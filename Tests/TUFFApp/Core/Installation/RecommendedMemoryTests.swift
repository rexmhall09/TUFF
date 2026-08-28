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

    /// The heaviest model must not recommend the same as the lightest, or the
    /// figure carries no information.
    @Test func theRecommendationVariesWithTheModel() {
        let values = Set(AppModelInstallDescriptor.catalog.map(\.recommendedUnifiedMemoryBytes))
        #expect(values.count > 1)
    }

}
