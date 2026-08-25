import Testing
@testable import TUFFModelCatalog

@Suite struct ModelCatalogTests {
    @Test func currentCatalogOrderAndSelectorsAreStable() {
        #expect(TUFFModelCatalog.all.map(\.selector) == ["gemma4", "qwen36"])
        #expect(TUFFModelCatalog.default.id == .gemma4_26B_A4B)
        #expect(TUFFModelCatalog.model(selector: "gemma4")?.id == .gemma4_26B_A4B)
        #expect(TUFFModelCatalog.model(selector: "qwen36")?.id == .qwen36_35B_A3B)
        #expect(TUFFModelCatalog.model(selector: "unknown") == nil)
    }

    @Test func sourcesRemainPinnedAndUnique() {
        #expect(Set(TUFFModelCatalog.all.map(\.source.repoID)).count
                == TUFFModelCatalog.all.count)
        for model in TUFFModelCatalog.all {
            #expect(model.source.revision.count == 40)
            #expect(model.source.sourceIndexSHA256.count == 64)
            #expect(model.source.approximateDownloadBytes > 0)
            #expect(model.source.installedBytes > 0)
            #expect(model.installDirectoryName.hasSuffix(".gturbo"))
            #expect(TUFFModelCatalog.model(manifestModelID:
                    model.source.manifestModelID)?.id == model.id)
        }
    }

    @Test func imageAddonsRemainSeparateAndM2Gated() {
        for model in TUFFModelCatalog.all {
            let addon = model.addons[0]
            #expect(addon.kind == .imageInput)
            #expect(addon.hardware.minimumAppleSiliconGeneration == 2)
            #expect(addon.source.repoID == model.source.repoID)
        }
    }

    @Test func architectureProfilesAreIndependentFromCheckpointCapabilities() {
        let gemma = TUFFModelCatalog.gemma4_26B_A4B
        let qwen = TUFFModelCatalog.qwen36_35B_A3B

        #expect(TUFFArchitectureProfile.gemma4E4B.feedForwardKind == .dense)
        #expect(TUFFArchitectureProfile.gemma4E4B.weightLayout == .affine)
        #expect(gemma.architecture.id == .gemma4_26B_A4B)
        #expect(qwen.architecture.id == .qwen36_35B_A3B)
        #expect(gemma.architecture.feedForwardKind == .mixtureOfExperts)
        #expect(qwen.architecture.feedForwardKind == .mixtureOfExperts)
        #expect(gemma.architecture.weightLayout == .affine)
        #expect(qwen.architecture.weightLayout == .affine)
        #expect(gemma.capabilities.contains(.imageInput))
        #expect(qwen.capabilities.contains(.reasoning))
    }
}
