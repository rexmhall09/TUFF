import Testing
import TUFFModelCatalog
@testable import TUFFEngine

/// Gemma 4 E2B, transcribed from the pinned checkpoint's `config.json`.
///
/// The transcription method was validated by running it against E4B and
/// comparing the result to the `gemma4_E4B` entry the runtime already generates
/// correct output from — every field matched, the layer mask included. These
/// tests pin the parts of E2B that a careless edit could silently get wrong.
@Suite struct Gemma4E2BTests {
    private var config: ArchConfig { .gemma4_E2B }

    @Test func shapesMatchThePinnedCheckpoint() {
        #expect(config.hiddenSize == 1536)
        #expect(config.intermediateSize == 6144)
        #expect(config.numLayers == 35)
        #expect(config.numHeads == 8)
        #expect(config.numKVHeads == 1)
        #expect(config.headDim == 256)
        #expect(config.fullHeadDim == 512)
        #expect(config.vocabSize == 262_144)
        #expect(config.slidingWindow == 512)
    }

    /// E2B is an E-series Gemma: per-layer embeddings and shared KV
    /// projections, the same machinery E4B runs on.
    @Test func itKeepsTheESeriesMachinery() {
        #expect(config.hasPerLayerInputs)
        #expect(config.hiddenSizePerLayerInput == 256)
        #expect(config.vocabSizePerLayerInput == 262_144)
        #expect(config.numKVSharedLayers == 20)
        #expect(config.feedForwardKind == .dense)
        #expect(config.numExperts == 0)
    }

    /// `layer_types` in the checkpoint marks full attention every fifth layer.
    /// Getting this wrong changes which layers use the long-range rope and
    /// produces subtly wrong output rather than an error.
    @Test func theLayerMaskMatchesTheCheckpoint() {
        let full = config.fullAttentionLayerMask.enumerated()
            .filter { $0.element == 1 }
            .map(\.offset)
        #expect(full == [4, 9, 14, 19, 24, 29, 34])
        #expect(config.fullAttentionLayerMask.count == 35)
    }

    @Test func ropeMatchesTheCheckpoint() {
        #expect(config.fullRopeTheta == 1_000_000.0)
        #expect(config.ropeTheta == 10_000.0)
        #expect(config.partialRotaryFactor == 0.25)
        #expect(config.finalLogitSoftcap == 30.0)
        #expect(config.tieWordEmbeddings)
    }

    /// `use_double_wide_mlp` is not a whole-model property. E2B's first 15
    /// layers hold a 6,144-wide MLP and its last 20 hold a 12,288-wide one.
    /// Sizing every layer from `intermediateSize` rejected the real checkpoint
    /// at layer 15 with an affine metadata mismatch.
    @Test func theLastTwentyLayersAreTwiceAsWide() {
        #expect(config.doubleWideFFNFromLayer == 15)
        #expect(config.ffnIntermediateSize(layer: 0) == 6_144)
        #expect(config.ffnIntermediateSize(layer: 14) == 6_144)
        #expect(config.ffnIntermediateSize(layer: 15) == 12_288)
        #expect(config.ffnIntermediateSize(layer: 34) == 12_288)
        // Scratch is shared across layers, so it is sized for the widest.
        #expect(config.maxFFNIntermediateSize == 12_288)
    }

    /// Every other model keeps one width throughout, and must be untouched.
    @Test func modelsWithoutWideLayersAreUnaffected() {
        for config in [ArchConfig.gemma4_E4B, .gemma4_26B_A4B, .qwen36_35B_A3B] {
            #expect(config.doubleWideFFNFromLayer == nil)
            #expect(config.maxFFNIntermediateSize == config.intermediateSize)
            #expect(config.ffnIntermediateSize(layer: 0) == config.intermediateSize)
            #expect(config.ffnIntermediateSize(layer: config.numLayers - 1)
                    == config.intermediateSize)
        }
    }

    @Test func itIsRegisteredAndReachable() {
        #expect(ArchConfig.registeredArchitectures[.gemma4_E2B] != nil)
        let descriptor = TUFFModelCatalog.model(id: .gemma4_E2B)
        #expect(descriptor?.architecture.feedForwardKind == .dense)
        #expect(descriptor?.source.repoID == "mlx-community/gemma-4-e2b-it-4bit")
    }

    /// Every catalogue source is pinned to a commit and an index digest, and
    /// no two models may share either.
    @Test func thePinIsCompleteAndUnique() {
        let source = TUFFModelCatalog.model(id: .gemma4_E2B)?.source
        #expect(source?.revision.count == 40)
        #expect(source?.sourceIndexSHA256.count == 64)
        let digests = TUFFModelCatalog.all.map(\.source.sourceIndexSHA256)
        #expect(Set(digests).count == digests.count)
    }
}
