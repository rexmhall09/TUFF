import Testing
import TUFFModelCatalog
@testable import TUFFAppCore

/// The catalogue shows a minimum and a recommendation. The minimum is a hard
/// gate the model is refused below; the recommendation is where its defaults
/// fit with macOS and other applications still running.
@Suite struct RecommendedMemoryTests {
    @Test func everyModelRecommendsAtLeastItsMinimum() {
        for descriptor in AppModelInstallDescriptor.catalog {
            #expect(descriptor.recommendedUnifiedMemoryBytes
                    >= descriptor.hardwareEligibility(
                        on: TUFFDeviceCapabilities(
                            unifiedMemoryBytes: 128 * TUFFModelCatalog.oneGiB,
                            macOSMajorVersion: 26,
                            appleSiliconGeneration: 5)).minimumUnifiedMemoryBytes,
                    "a model recommends less than its own floor")
        }
    }

    /// A recommendation nobody can buy is not a recommendation.
    @Test func everyRecommendationIsASizeMacsAreSoldIn() {
        for descriptor in AppModelInstallDescriptor.catalog {
            #expect(AppModelInstallDescriptor.unifiedMemoryTiers
                .contains(descriptor.recommendedUnifiedMemoryBytes),
                    "a model recommends a size Macs are not sold in")
        }
    }

    /// Qwen3.8 Flash Next holds a 16 GB floor, and recommends 24 GB.
    ///
    /// The recommendation was 32 GB while the expert cache defaulted to 48
    /// slots. Measuring that choice showed 48 to be 1.7x slower than 32 as
    /// well as 2.4 GB larger, so the default moved and the recommendation
    /// followed it down — it is the working set doubled plus 4 GiB, rounded to
    /// a size Macs ship in, and the working set genuinely shrank. The floor is
    /// unchanged, because the floor is the hardware gate rather than a
    /// function of the defaults.
    @Test func flashNextIsSixteenGigabyteMinimumAndTwentyFourRecommended() {
        let descriptor = AppModelInstallDescriptor.qwen38FlashNext
        let catalog = TUFFModelCatalog.qwen38FlashNext
        #expect(catalog.hardware.minimumUnifiedMemoryBytes
                == 16 * TUFFModelCatalog.oneGiB)
        #expect(descriptor.recommendedUnifiedMemoryBytes
                == 24 * TUFFModelCatalog.oneGiB)

        // The defaults have to fit inside a 16 GB Mac's budget, or the floor
        // is a floor the model cannot actually stand on.
        let sixteen = TUFFDeviceCapabilities(
            unifiedMemoryBytes: 16 * TUFFModelCatalog.oneGiB,
            macOSMajorVersion: 26,
            appleSiliconGeneration: 2)
        let working = catalog.memory.estimatedWorkingSetBytes(
            contextTokens: catalog.runtimeDefaults.contextTokens,
            expertCacheSlots: catalog.runtimeDefaults.expertCacheSlots)
        #expect(working <= sixteen.safeAppMemoryBudgetBytes)
    }

    /// Flash Next has completed its validation run, so nothing but hardware
    /// stands between a Mac and the download. Its 110 GB is guarded by the
    /// memory floor alone now, which has to hold in both directions.
    @Test func flashNextIsOfferedOnceItsMemoryFloorIsMet() {
        let huge = TUFFDeviceCapabilities(
            unifiedMemoryBytes: 128 * TUFFModelCatalog.oneGiB,
            macOSMajorVersion: 26,
            appleSiliconGeneration: 5)
        let eligibility = AppModelInstallDescriptor.qwen38FlashNext
            .hardwareEligibility(on: huge)
        #expect(eligibility.isCompatible)
        #expect(eligibility.issues.isEmpty)

        let small = TUFFDeviceCapabilities(
            unifiedMemoryBytes: 8 * TUFFModelCatalog.oneGiB,
            macOSMajorVersion: 26,
            appleSiliconGeneration: 2)
        let refused = AppModelInstallDescriptor.qwen38FlashNext
            .hardwareEligibility(on: small)
        #expect(!refused.isCompatible)
        #expect(refused.explanation != nil)
    }

    /// The context default is not a preference: past 2,048 keys the reference
    /// switches to sparse attention that the runner does not implement, so a
    /// larger default would answer from a different model than the one this
    /// was validated against.
    @Test func flashNextDefaultsToTheContextItsAttentionCovers() {
        let catalog = TUFFModelCatalog.qwen38FlashNext
        #expect(catalog.runtimeDefaults.contextTokens == 2_048)
        #expect(catalog.memory.defaultContextTokens == 2_048)
    }

    /// The heaviest model must not recommend the same as the lightest, or the
    /// figure carries no information.
    @Test func theRecommendationVariesWithTheModel() {
        let values = Set(AppModelInstallDescriptor.catalog.map(\.recommendedUnifiedMemoryBytes))
        #expect(values.count > 1)
    }

}
