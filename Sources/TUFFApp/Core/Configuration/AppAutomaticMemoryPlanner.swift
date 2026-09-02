import TUFFEngine
import TUFFModelCatalog

/// The concrete memory settings Auto resolves for one model on one Mac.
public struct AppAutomaticMemoryPlan: Equatable, Sendable {
    public let contextTokens: Int
    public let expertCacheSlots: Int
    public let estimatedWorkingSetBytes: UInt64
    public let safeBudgetBytes: UInt64

    public init(contextTokens: Int,
                expertCacheSlots: Int,
                estimatedWorkingSetBytes: UInt64,
                safeBudgetBytes: UInt64) {
        self.contextTokens = contextTokens
        self.expertCacheSlots = expertCacheSlots
        self.estimatedWorkingSetBytes = estimatedWorkingSetBytes
        self.safeBudgetBytes = safeBudgetBytes
    }
}

/// Resolves a stable per-model memory profile from the Mac's detected unified
/// memory. Auto preserves the checkpoint's qualified context length and spends
/// additional safe capacity on routed-expert slots, where RAM directly avoids
/// SSD reads during decode. Dense models keep their qualified defaults because
/// a larger KV reservation does not make token generation faster.
public enum AppAutomaticMemoryPlanner {
    public static func plan(
        for descriptor: AppModelInstallDescriptor,
        on device: TUFFDeviceCapabilities
    ) -> AppAutomaticMemoryPlan? {
        guard let id = descriptor.catalogID,
              let catalog = TUFFModelCatalog.model(id: id) else { return nil }

        let context = catalog.runtimeDefaults.contextTokens
        let budget = device.safeAppMemoryBudgetBytes
        let memory = catalog.memory
        var slots = catalog.runtimeDefaults.expertCacheSlots

        if descriptor.usesExpertCache {
            let candidates = descriptor.usefulExpertCacheSlotCounts
            if let smallest = candidates.first {
                // Always leave a runnable result. Hardware eligibility remains
                // the hard gate if even this model's minimum cannot fit.
                slots = smallest
                for candidate in candidates {
                    let estimate = memory.estimatedWorkingSetBytes(
                        contextTokens: context,
                        expertCacheSlots: candidate)
                    guard estimate <= budget else { break }
                    slots = candidate
                }
            }
        }

        return AppAutomaticMemoryPlan(
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
              let plan = plan(for: descriptor, on: device) else { return profile }
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
