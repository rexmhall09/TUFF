import Testing
import TUFFModelCatalog
@testable import TUFFEngine

/// Architecture facts transcribed from the pinned 12B QAT checkpoint. These
/// are execution semantics, not display metadata: a wrong KV shape, layer mask,
/// or vision graph can load successfully and still produce invalid output.
@Suite struct Gemma4Unified12BTests {
    private var config: ArchConfig { .gemma4_12B_QAT }

    @Test func languageShapesMatchPinnedCheckpoint() {
        #expect(config.hiddenSize == 3_840)
        #expect(config.intermediateSize == 15_360)
        #expect(config.numLayers == 48)
        #expect(config.numHeads == 16)
        #expect(config.numKVHeads == 8)
        #expect(config.numFullKVHeads == 1)
        #expect(config.headDim == 256)
        #expect(config.fullHeadDim == 512)
        #expect(config.slidingWindow == 1_024)
        #expect(config.attentionKEqV)
        #expect(!config.hasPerLayerInputs)
        #expect(config.feedForwardKind == .dense)
    }

    @Test func fullAttentionOccursEverySixthLayer() {
        let full = config.fullAttentionLayerMask.enumerated()
            .filter { $0.element == 1 }
            .map(\.offset)
        #expect(full == [5, 11, 17, 23, 29, 35, 41, 47])
    }

    @Test func unifiedVisionContractIsDistinctFromLegacyGemma() {
        let vision = VisionConfig(family: .gemma4, modelVariant: .gemma4_12B_QAT)
        #expect(vision.architecture == .gemma4Unified12B)
        #expect(vision.patchSize == 48)
        #expect(vision.patchDimension == 6_912)
        #expect(vision.maximumPatches == 280)
        #expect(vision.poolingKernel == 1)
        #expect(vision.textHiddenSize == 3_840)
        #expect(vision.numLayers == 0)
    }

    @Test func eSeriesVisionWidthsFollowTheirTextModels() {
        let e2b = VisionConfig(family: .gemma4, modelVariant: .gemma4_E2B)
        let e4b = VisionConfig(family: .gemma4, modelVariant: .gemma4_E4B)
        #expect(e2b.textHiddenSize == 1_536)
        #expect(e4b.textHiddenSize == 2_560)
        // Both pinned E-series checkpoints carry the same Gemma vision tower;
        // only their final language projection width differs.
        #expect(e2b.hiddenSize == 768)
        #expect(e4b.hiddenSize == 768)
        #expect(e2b.intermediateSize == 3_072)
        #expect(e4b.intermediateSize == 3_072)
        #expect(e2b.numLayers == 16)
        #expect(e4b.numLayers == 16)
        #expect(e2b.usesESeriesVisionTower)
        #expect(e4b.usesESeriesVisionTower)
        #expect(VisionConfig(
            family: .gemma4, modelVariant: .gemma4_26B_A4B).textHiddenSize == 2_816)
    }

    /// The download figure is the *coalesced range* total for this checkpoint's
    /// own tensor layout. It was previously E2B's figure plus the payload
    /// delta, which understated E4B by 143 MB: the two checkpoints interleave
    /// their vision tensors differently even though the towers are identical.
    @Test func e4bImageAddonMatchesThePinnedPack() {
        let addon = TUFFModelCatalog.gemma4_E4B.addons[0]
        #expect(addon.source.approximateDownloadBytes == 1_313_596_274)
        #expect(addon.source.installedBytes == 337_376_704)
    }

    @Test func catalogPinAndAddonAreExact() {
        let model = TUFFModelCatalog.gemma4_12B_QAT
        #expect(model.source.repoID == "mlx-community/gemma-4-12B-it-qat-4bit")
        #expect(model.source.revision
                == "e70c6b3ba0979b3357dcd2f223ad8bde7787a6b6")
        #expect(model.source.sourceIndexSHA256
                == "b87c93774de5d13ca9d0e21b045793e42e5df032fb5e7622212524f56f9695f2")
        #expect(model.addons.count == 1)
        #expect(model.qualification == .qualified)
        #expect(model.addons[0].source.approximateDownloadBytes == 102_556_672)
        #expect(model.addons[0].source.installedBytes == 40_581_222)
        #expect(ArchConfig.registeredArchitectures[.gemma4_12B_QAT] == config)
    }
}
