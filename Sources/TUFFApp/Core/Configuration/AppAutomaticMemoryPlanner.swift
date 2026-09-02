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
/// Every profile ends by spending whatever budget is left on routed-expert
/// slots, because that is the only memory that reduces work during
/// generation; they differ in how much context they take first. A dense model
/// has no expert cache, so for it the profiles differ in context alone — which
/// is the honest answer, since extra memory cannot make a dense model faster.
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
        let slotOptions = descriptor.usesExpertCache
            ? descriptor.usefulExpertCacheSlotCounts.sorted()
            : [catalog.runtimeDefaults.expertCacheSlots]
        // Always leave a runnable answer: hardware eligibility, not Auto, is
        // the gate on a model this Mac cannot host at all.
        let smallestSlots = slotOptions.first ?? catalog.runtimeDefaults.expertCacheSlots

        func fits(context: Int, slots: Int) -> Bool {
            memory.estimatedWorkingSetBytes(contextTokens: context,
                                            expertCacheSlots: slots) <= budget
        }

        // Context ceiling for this profile, expressed in tokens.
        let contextCeiling = profile.contextGrowthLimit.map {
            qualifiedContext * $0
        } ?? Int.max
        let contextOptions = AppContextLengthOption.allCases
            .map(\.tokens)
            .filter { $0 <= contextCeiling }
            .sorted()

        // Longest context this profile allows that still fits beside the
        // model's minimum cache. Never drop below the qualified default: that
        // is the length the checkpoint was validated at.
        var context = qualifiedContext
        for candidate in contextOptions where candidate > context {
            guard fits(context: candidate, slots: smallestSlots) else { break }
            context = candidate
        }

        // Spend the remaining budget on resident experts.
        var slots = smallestSlots
        for candidate in slotOptions where candidate > slots {
            guard fits(context: context, slots: candidate) else { break }
            slots = candidate
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
