import TUFFEngine
import TUFFModelCatalog

/// The concrete memory settings Auto resolves for one model on one Mac.
public struct AppAutomaticMemoryPlan: Equatable, Sendable {
    public let profile: AppAutomaticMemoryProfile
    public let contextTokens: Int
    public let expertCacheSlots: Int
    public let estimatedWorkingSetBytes: UInt64
    public let safeBudgetBytes: UInt64

    public init(profile: AppAutomaticMemoryProfile = .balanced,
                contextTokens: Int,
                expertCacheSlots: Int,
                estimatedWorkingSetBytes: UInt64,
                safeBudgetBytes: UInt64) {
        self.profile = profile
        self.contextTokens = contextTokens
        self.expertCacheSlots = expertCacheSlots
        self.estimatedWorkingSetBytes = estimatedWorkingSetBytes
        self.safeBudgetBytes = safeBudgetBytes
    }
}

/// Resolves a per-model memory profile from the Mac's detected unified memory
/// and the checkpoint's measured working-set profile.
///
/// The profiles differ in context length and nothing else, because context is
/// the only thing measurement says is worth buying. On Gemma 4 26B-A4B, a
/// 16 GB Mac, 24 generated tokens:
///
///     8K context, 16 slots   7.69 tok/s   1.93 GB peak
///     8K context, 128 slots  6.28 tok/s   2.99 GB peak
///     16K context, 16 slots  7.67 tok/s   1.93 GB peak
///
/// Doubling the context is free. Filling the budget with routed-expert slots
/// costs 18% of throughput and a gigabyte of memory: each slot is its own
/// Metal buffer, and a command buffer that references hundreds of them pays
/// for all of them. Auto therefore keeps the checkpoint's qualified slot
/// count, which is the count it was validated at, and spends nothing else.
/// Raising it stays available by hand for anyone who wants to trade that
/// throughput for fewer SSD reads.
public enum AppAutomaticMemoryPlanner {
    public static func plan(
        for descriptor: AppModelInstallDescriptor,
        on device: TUFFDeviceCapabilities,
        profile: AppAutomaticMemoryProfile = .balanced
    ) -> AppAutomaticMemoryPlan? {
        guard let id = descriptor.catalogID,
              let catalog = TUFFModelCatalog.model(id: id) else { return nil }

        let budget = device.safeAppMemoryBudgetBytes
        let memory = catalog.memory
        let qualifiedContext = catalog.runtimeDefaults.contextTokens
        func fits(context: Int, slots: Int) -> Bool {
            memory.estimatedWorkingSetBytes(contextTokens: context,
                                            expertCacheSlots: slots) <= budget
        }

        // The qualified slot count, clamped to what this model can use, then
        // raised just far enough to reach chunked prefill. GPT-OSS 20B is
        // qualified at four slots, and the chunked prefill path needs sixteen;
        // leaving it at four would trade a much slower prompt for memory Auto
        // is not otherwise spending. Nothing grows past that — see the note
        // above for why filling the budget with slots is a loss.
        let slots: Int
        if descriptor.usesExpertCache {
            let options = descriptor.usefulExpertCacheSlotCounts.sorted()
            let qualified = catalog.runtimeDefaults.expertCacheSlots
            let clamped = options.contains(qualified)
                ? qualified
                : (options.last(where: { $0 <= qualified })
                    ?? options.first
                    ?? qualified)
            let prefillFloor = options.first {
                $0 >= RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill
            }
            if let prefillFloor, prefillFloor > clamped,
               fits(context: qualifiedContext, slots: prefillFloor) {
                slots = prefillFloor
            } else {
                slots = clamped
            }
        } else {
            slots = catalog.runtimeDefaults.expertCacheSlots
        }

        // Context ceiling for this profile, expressed in tokens.
        let contextCeiling = profile.contextGrowthLimit.map {
            qualifiedContext * $0
        } ?? Int.max
        let contextOptions = AppContextLengthOption.allCases
            .map(\.tokens)
            .filter { $0 <= contextCeiling }
            .sorted()

        // Longest context this profile allows that still fits. Never drop below
        // the qualified default: that is the length the checkpoint was
        // validated at, and hardware eligibility — not Auto — is the gate on a
        // model this Mac cannot host at all.
        var context = qualifiedContext
        for candidate in contextOptions where candidate > context {
            guard fits(context: candidate, slots: slots) else { break }
            context = candidate
        }

        return AppAutomaticMemoryPlan(
            profile: profile,
            contextTokens: context,
            expertCacheSlots: slots,
            estimatedWorkingSetBytes: memory.estimatedWorkingSetBytes(
                contextTokens: context,
                expertCacheSlots: slots),
            safeBudgetBytes: budget)
    }

    public static func applying(
        _ profile: AppModelSettingsProfile,
        for descriptor: AppModelInstallDescriptor,
        on device: TUFFDeviceCapabilities
    ) -> AppModelSettingsProfile {
        guard profile.automaticMemory,
              let plan = plan(for: descriptor,
                              on: device,
                              profile: profile.automaticMemoryProfile) else { return profile }
        var resolved = profile
        resolved.contextTokens = plan.contextTokens
        resolved.expertCacheSlots = plan.expertCacheSlots
        // Auto owns the memory-dependent prefill decision too. This lets a
        // GPT-OSS profile whose manual four-slot cache disables prefill gain
        // the faster chunked path when Auto can afford at least 16 slots.
        resolved.prefillEnabled = resolved.expertCacheSlots
            >= RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill
        return resolved
    }
}
