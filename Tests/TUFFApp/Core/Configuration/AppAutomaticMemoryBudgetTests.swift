import Testing
import TUFFModelCatalog
@testable import TUFFAppCore

/// The budget has to admit the models the catalog says a machine can run.
///
/// This suite exists because it once did not: the budget was capped by however
/// much memory happened to be reclaimable when the app launched, so a 16 GB Mac
/// with a browser open refused models qualified for 16 GB. A share of installed
/// memory cannot drift that way, and these tests pin the consequence rather
/// than the formula.
@Suite struct AppAutomaticMemoryBudgetTests {
    private func device(_ gibibytes: UInt64) -> TUFFDeviceCapabilities {
        TUFFDeviceCapabilities(
            unifiedMemoryBytes: gibibytes * TUFFModelCatalog.oneGiB,
            macOSMajorVersion: 26,
            appleSiliconGeneration: 5)
    }

    /// Every model whose stated minimum this Mac meets must fit the budget at
    /// its qualified defaults. A model the catalog offers on a machine and then
    /// refuses to load on it is a contradiction, not a safety margin.
    @Test(arguments: [8 as UInt64, 16, 24, 32, 64, 128])
    func qualifiedDefaultsFitWhereverTheModelIsOffered(gibibytes: UInt64) throws {
        let capabilities = device(gibibytes)
        for descriptor in TUFFModelCatalog.all {
            guard capabilities.unifiedMemoryBytes
                >= descriptor.hardware.minimumUnifiedMemoryBytes else { continue }
            let estimate = descriptor.memory.estimatedWorkingSetBytes(
                contextTokens: descriptor.runtimeDefaults.contextTokens,
                expertCacheSlots: descriptor.runtimeDefaults.expertCacheSlots)
            #expect(estimate <= capabilities.safeAppMemoryBudgetBytes,
                    "\(descriptor.displayName) on \(gibibytes) GB")
        }
    }

    /// The same statement through the gate the app actually consults.
    @Test(arguments: [16 as UInt64, 32, 64, 128])
    func compatibilityAgreesWithTheCatalogsOwnMinimum(gibibytes: UInt64) {
        let capabilities = device(gibibytes)
        for descriptor in TUFFModelCatalog.all {
            guard capabilities.unifiedMemoryBytes
                >= descriptor.hardware.minimumUnifiedMemoryBytes else { continue }
            let compatibility = descriptor.compatibility(
                with: capabilities,
                contextTokens: descriptor.runtimeDefaults.contextTokens,
                expertCacheSlots: descriptor.runtimeDefaults.expertCacheSlots)
            #expect(compatibility.isCompatible, "\(descriptor.displayName)")
        }
    }

    /// Auto's own answer, for every model and every profile, on the machine
    /// this project is developed and benchmarked on.
    @Test func autoResolvesARunnablePlanForEveryModelOn16GB() throws {
        let capabilities = device(16)
        for descriptor in TUFFModelCatalog.all {
            let install = AppModelInstallDescriptor(catalog: descriptor)
            guard capabilities.unifiedMemoryBytes
                >= descriptor.hardware.minimumUnifiedMemoryBytes else { continue }
            for profile in AppAutomaticMemoryProfile.allCases {
                let plan = try #require(AppAutomaticMemoryPlanner.plan(
                    for: install, on: capabilities, profile: profile))
                #expect(plan.estimatedWorkingSetBytes <= plan.safeBudgetBytes,
                        "\(descriptor.displayName) · \(profile.title)")
                #expect(plan.contextTokens
                    >= descriptor.runtimeDefaults.contextTokens)
            }
        }
    }

    @Test func theBudgetIsThreeQuartersOfInstalledMemory() {
        for gibibytes in [8 as UInt64, 16, 24, 32, 64, 128] {
            #expect(device(gibibytes).safeAppMemoryBudgetBytes
                == gibibytes * TUFFModelCatalog.oneGiB / 4 * 3)
        }
    }
}
