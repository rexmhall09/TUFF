import Testing
import TUFFAppCore
@testable import TUFFMac

/// The catalogue is grouped by how a model is run, because that is what decides
/// what it needs from the Mac: a streamed model is bounded by free disk, a
/// resident one by memory. Listed together they read as a size ranking.
@Suite @MainActor struct ModelCatalogGroupTests {
    @Test func everyCatalogueModelLandsInExactlyOneGroup() {
        for descriptor in AppModelInstallDescriptor.catalog {
            let groups = ModelCatalogGroup.allCases.filter { $0.contains(descriptor) }
            #expect(groups.count == 1,
                    "\(descriptor.displayName) belongs to \(groups.count) groups")
        }
    }

    /// The split is the expert cache, which is the thing that makes a model
    /// stream rather than load.
    @Test func groupingFollowsTheExpertCache() {
        for descriptor in AppModelInstallDescriptor.catalog {
            let streamed = ModelCatalogGroup.streaming.contains(descriptor)
            #expect(streamed == descriptor.usesExpertCache)
        }
    }

    /// Both groups have to be populated, or the headers are noise.
    @Test func theShippedCatalogueUsesBothGroups() {
        for group in ModelCatalogGroup.allCases {
            #expect(AppModelInstallDescriptor.catalog.contains { group.contains($0) },
                    "\(group.title) is empty in the shipped catalogue")
        }
    }
}
