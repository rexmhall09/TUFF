import Testing
import TUFFEngine
import TUFFModelCatalog
@testable import TUFFAppCore

@Suite struct AppAutomaticMemoryPlannerTests {
    private func device(
        _ gibibytes: UInt64,
        availableGibibytes: UInt64? = nil
    ) -> TUFFDeviceCapabilities {
        TUFFDeviceCapabilities(
            unifiedMemoryBytes: gibibytes * TUFFModelCatalog.oneGiB,
            availableMemoryBytes: availableGibibytes.map {
                $0 * TUFFModelCatalog.oneGiB
            },
            macOSMajorVersion: 26,
            appleSiliconGeneration: 5)
    }

    @Test func miniMaxStaysAtItsQualifiedCacheOn16GB() throws {
        let plan = try #require(AppAutomaticMemoryPlanner.plan(
            for: .minimaxM27,
            on: device(16)))

        #expect(plan.contextTokens == 4_096)
        #expect(plan.expertCacheSlots == 16)
        #expect(plan.estimatedWorkingSetBytes <= plan.safeBudgetBytes)
    }

    @Test func miniMaxSpendsAdditionalSafeRAMOnExpertResidency() throws {
        let plan16 = try #require(AppAutomaticMemoryPlanner.plan(
            for: .minimaxM27,
            on: device(16)))
        let plan64 = try #require(AppAutomaticMemoryPlanner.plan(
            for: .minimaxM27,
            on: device(64)))

        #expect(plan64.contextTokens == plan16.contextTokens)
        #expect(plan64.expertCacheSlots == 96)
        #expect(plan64.expertCacheSlots > plan16.expertCacheSlots)
        #expect(plan64.estimatedWorkingSetBytes <= plan64.safeBudgetBytes)
    }

    @Test func cacheGrowthStopsAtAUsefulModelAndRuntimeLimit() throws {
        let gpt20 = try #require(AppModelInstallDescriptor.descriptor(
            for: ModelVariant.gptOss_20B))
        let gptPlan = try #require(AppAutomaticMemoryPlanner.plan(
            for: gpt20,
            on: device(128)))
        let miniMaxPlan = try #require(AppAutomaticMemoryPlanner.plan(
            for: .minimaxM27,
            on: device(128)))

        #expect(gpt20.usefulExpertCacheSlotCounts.last == 32)
        #expect(gptPlan.expertCacheSlots == 32)
        #expect(miniMaxPlan.expertCacheSlots == 128)
    }

    @Test func denseModelsKeepQualifiedDefaults() throws {
        let plan = try #require(AppAutomaticMemoryPlanner.plan(
            for: .gemma4E4B,
            on: device(128)))

        #expect(plan.contextTokens
            == TUFFModelCatalog.gemma4_E4B.runtimeDefaults.contextTokens)
        #expect(plan.expertCacheSlots
            == TUFFModelCatalog.gemma4_E4B.runtimeDefaults.expertCacheSlots)
    }

    @Test func turningAutoOffPreservesManualMemorySettings() {
        let manual = AppModelSettingsProfile(
            automaticMemory: false,
            contextTokens: 32_768,
            expertCacheSlots: 24)

        let resolved = AppAutomaticMemoryPlanner.applying(
            manual,
            for: .minimaxM27,
            on: device(64))

        #expect(resolved == manual)
    }

    @Test func autoEnablesPrefillWhenItCanAffordTheRequiredCache() throws {
        let descriptor = try #require(AppModelInstallDescriptor.descriptor(
            for: ModelVariant.gptOss_20B))
        let manual = AppModelSettingsProfile(
            automaticMemory: true,
            expertCacheSlots: 4,
            prefillEnabled: false)

        let resolved = AppAutomaticMemoryPlanner.applying(
            manual,
            for: descriptor,
            on: device(128))

        #expect(resolved.expertCacheSlots == 32)
        #expect(resolved.prefillEnabled)
    }

    @Test func safeBudgetRetainsSystemHeadroom() {
        #expect(device(8).safeAppMemoryBudgetBytes == 6 * TUFFModelCatalog.oneGiB)
        #expect(device(16).safeAppMemoryBudgetBytes
            == 16 * TUFFModelCatalog.oneGiB
                - (16 * TUFFModelCatalog.oneGiB / 5))
    }

    @Test func autoRespondsToRAMAvailableAtLaunch() throws {
        let idle = try #require(AppAutomaticMemoryPlanner.plan(
            for: .minimaxM27,
            on: device(64)))
        let busy = try #require(AppAutomaticMemoryPlanner.plan(
            for: .minimaxM27,
            on: device(64, availableGibibytes: 16)))

        #expect(busy.safeBudgetBytes < idle.safeBudgetBytes)
        #expect(busy.expertCacheSlots < idle.expertCacheSlots)
        #expect(busy.expertCacheSlots == 24)
    }
}
