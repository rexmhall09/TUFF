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

    @Test func speedKeepsTheQualifiedContextOn16GB() throws {
        let plan = try #require(AppAutomaticMemoryPlanner.plan(
            for: .minimaxM27,
            on: device(16),
            profile: .speed))

        #expect(plan.contextTokens == 4_096)
        #expect(plan.estimatedWorkingSetBytes <= plan.safeBudgetBytes)
    }

    /// Auto does not buy expert-cache slots with spare memory. Measured on
    /// Gemma 4 26B-A4B: 16 slots gave 7.69 tokens per second at 1.93 GB peak,
    /// 128 slots gave 6.28 at 2.99 GB. More slots is slower and larger, so a
    /// roomier Mac gets the same cache and a longer context instead.
    @Test func aRoomierMacDoesNotGetMoreExpertSlots() throws {
        let plan16 = try #require(AppAutomaticMemoryPlanner.plan(
            for: .minimaxM27, on: device(16), profile: .speed))
        let plan64 = try #require(AppAutomaticMemoryPlanner.plan(
            for: .minimaxM27, on: device(64), profile: .speed))

        #expect(plan64.expertCacheSlots == plan16.expertCacheSlots)
        #expect(plan64.estimatedWorkingSetBytes <= plan64.safeBudgetBytes)
    }

    /// Models already qualified at or above the chunked-prefill floor keep
    /// exactly their qualified count, whatever the Mac.
    @Test func autoKeepsEachModelsQualifiedSlotCount() throws {
        for descriptor in [AppModelInstallDescriptor.minimaxM27, .gemma4E4B] {
            let qualified = try #require(descriptor.catalogID
                .flatMap(TUFFModelCatalog.model(id:)))
                .runtimeDefaults.expertCacheSlots
            for gibibytes in [16 as UInt64, 128] {
                let plan = try #require(AppAutomaticMemoryPlanner.plan(
                    for: descriptor, on: device(gibibytes), profile: .speed))
                #expect(plan.expertCacheSlots == qualified)
            }
        }
    }

    // MARK: - Profiles

    @Test func speedNeverRaisesContextAboveTheQualifiedDefault() throws {
        for descriptor in [AppModelInstallDescriptor.gemma4E4B, .minimaxM27] {
            let plan = try #require(AppAutomaticMemoryPlanner.plan(
                for: descriptor, on: device(128), profile: .speed))
            let qualified = try #require(descriptor.catalogID
                .flatMap(TUFFModelCatalog.model(id:)))
                .runtimeDefaults.contextTokens
            #expect(plan.contextTokens == qualified)
        }
    }

    @Test func balancedStopsAtTwiceTheQualifiedContext() throws {
        let qualified = TUFFModelCatalog.gemma4_E4B.runtimeDefaults.contextTokens
        let plan = try #require(AppAutomaticMemoryPlanner.plan(
            for: .gemma4E4B, on: device(128), profile: .balanced))

        #expect(plan.contextTokens <= qualified * 2)
        #expect(plan.contextTokens > qualified)
    }

    @Test func contextTakesTheLongestWindowThatFits() throws {
        let balanced = try #require(AppAutomaticMemoryPlanner.plan(
            for: .gemma4E4B, on: device(128), profile: .balanced))
        let longest = try #require(AppAutomaticMemoryPlanner.plan(
            for: .gemma4E4B, on: device(128), profile: .context))

        #expect(longest.contextTokens > balanced.contextTokens)
        #expect(longest.contextTokens
            == AppContextLengthOption.allCases.map(\.tokens).max())
        #expect(longest.estimatedWorkingSetBytes <= longest.safeBudgetBytes)
    }

    /// The three profiles order by context and stay inside the budget wherever
    /// the model's own minimum fits. On a Mac too small for the model at all,
    /// the plan is the minimum rather than nothing: hardware eligibility, not
    /// Auto, is what refuses that model.
    @Test(arguments: [8 as UInt64, 16, 64, 128])
    func profilesAreOrderedAndFitWhereverTheyCan(gibibytes: UInt64) throws {
        let capabilities = device(gibibytes)
        let minimum = try #require(AppAutomaticMemoryPlanner.plan(
            for: .minimaxM27, on: capabilities, profile: .speed))
        let modelFitsAtAll = minimum.estimatedWorkingSetBytes <= minimum.safeBudgetBytes
        var previous = 0
        for profile in [AppAutomaticMemoryProfile.speed, .balanced, .context] {
            let plan = try #require(AppAutomaticMemoryPlanner.plan(
                for: .minimaxM27, on: capabilities, profile: profile))
            #expect(plan.contextTokens >= previous)
            if modelFitsAtAll {
                #expect(plan.estimatedWorkingSetBytes <= plan.safeBudgetBytes)
            } else {
                #expect(plan.contextTokens
                    == TUFFModelCatalog.minimaxM27.runtimeDefaults.contextTokens)
            }
            previous = plan.contextTokens
        }
    }

    /// A model whose minimum does not fit still resolves to something the app
    /// can show and run under a bypass, rather than to nothing.
    @Test func aTightMacStillResolvesARunnablePlan() throws {
        let plan = try #require(AppAutomaticMemoryPlanner.plan(
            for: .minimaxM27, on: device(8), profile: .context))

        #expect(plan.contextTokens
            == TUFFModelCatalog.minimaxM27.runtimeDefaults.contextTokens)
        #expect(plan.expertCacheSlots
            == TUFFModelCatalog.minimaxM27.runtimeDefaults.expertCacheSlots)
    }

    @Test func denseModelsGainContextRatherThanCacheSlots() throws {
        let speed = try #require(AppAutomaticMemoryPlanner.plan(
            for: .gemma4E4B, on: device(128), profile: .speed))
        let longest = try #require(AppAutomaticMemoryPlanner.plan(
            for: .gemma4E4B, on: device(128), profile: .context))

        #expect(speed.expertCacheSlots == longest.expertCacheSlots)
        #expect(longest.contextTokens > speed.contextTokens)
    }

    // MARK: - Applying

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

    @Test func applyingHonoursTheSelectedProfile() throws {
        var profile = AppModelSettingsProfile(automaticMemory: true)
        profile.automaticMemoryProfile = .context
        let resolved = AppAutomaticMemoryPlanner.applying(
            profile, for: .gemma4E4B, on: device(128))
        let expected = try #require(AppAutomaticMemoryPlanner.plan(
            for: .gemma4E4B, on: device(128), profile: .context))

        #expect(resolved.contextTokens == expected.contextTokens)
        #expect(resolved.expertCacheSlots == expected.expertCacheSlots)
    }

    @Test func autoEnablesPrefillWhenItCanAffordTheRequiredCache() throws {
        let descriptor = try #require(AppModelInstallDescriptor.descriptor(
            for: ModelVariant.gptOss_20B))
        var manual = AppModelSettingsProfile(
            automaticMemory: true,
            expertCacheSlots: 4,
            prefillEnabled: false)
        manual.automaticMemoryProfile = .speed

        let resolved = AppAutomaticMemoryPlanner.applying(
            manual,
            for: descriptor,
            on: device(128))

        // Raised to the chunked-prefill floor, not to the 32 Auto used to
        // grow to and not left at the 4 the checkpoint is qualified with.
        #expect(resolved.expertCacheSlots
            == RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill)
        #expect(resolved.prefillEnabled)
    }

    // MARK: - Budget

    @Test func safeBudgetIsAFixedShareOfInstalledMemory() {
        #expect(device(8).safeAppMemoryBudgetBytes == 6 * TUFFModelCatalog.oneGiB)
        #expect(device(16).safeAppMemoryBudgetBytes == 12 * TUFFModelCatalog.oneGiB)
        #expect(device(64).safeAppMemoryBudgetBytes == 48 * TUFFModelCatalog.oneGiB)
    }

    /// The budget used to be capped by whatever was reclaimable at launch,
    /// which made a 16 GB Mac refuse models qualified for 16 GB whenever
    /// something else was open. It now describes the machine, not the moment.
    @Test func launchTimeAvailabilityNoLongerShrinksTheBudget() throws {
        let idle = try #require(AppAutomaticMemoryPlanner.plan(
            for: .minimaxM27, on: device(64), profile: .speed))
        let busy = try #require(AppAutomaticMemoryPlanner.plan(
            for: .minimaxM27,
            on: device(64, availableGibibytes: 16),
            profile: .speed))

        #expect(busy.safeBudgetBytes == idle.safeBudgetBytes)
        #expect(busy.expertCacheSlots == idle.expertCacheSlots)
    }
}
