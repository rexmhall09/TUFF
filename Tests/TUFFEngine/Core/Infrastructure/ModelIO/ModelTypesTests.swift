import Testing
import Foundation
@testable import TUFFEngine

@Suite struct ModelTypesTests {

    @Test func archConfigGemma4BaselineMatchesDocs() {
        let a = ArchConfig.gemma4_26B_A4B
        #expect(a.hiddenSize == 2816)
        #expect(a.intermediateSize == 2112)
        #expect(a.moeIntermediateSize == 704)
        #expect(a.numLayers == 30)
        #expect(a.numExperts == 128)
        #expect(a.topKExperts == 8)
        #expect(a.vocabSize == 262144)
        #expect(a.tieWordEmbeddings == true)
        #expect(a.finalLogitSoftcap == 30.0)
        #expect(a.fullAttentionLayerMask.count == 30)
        let fullCount = a.fullAttentionLayerMask.reduce(0) { $0 + Int($1) }
        #expect(fullCount == 5, "Gemma 4 has 5 full-attention layers, got \(fullCount)")
        // Mask flags layers 5, 11, 17, 23, 29.
        for L in [5, 11, 17, 23, 29] {
            #expect(a.fullAttentionLayerMask[L] == 1, "layer \(L) should be full-attention")
        }
    }

    @Test func modelErrorDescriptionsContainKeyFacts() {
        let e1 = ModelError.archMismatch(field: "hiddenSize", expected: "2816", actual: "4096")
        #expect(e1.description.contains("2816") && e1.description.contains("4096"))
        let e2 = ModelError.unsupportedVersion(major: 2, minor: 0)
        #expect(e2.description.contains("2"))
        let e3 = ModelError.checksumMismatch(file: "model_weights.bin")
        #expect(e3.description.contains("model_weights.bin"))
    }

    @Test func gptOss20BProfileMatchesPinnedCheckpoint() {
        let config = ArchConfig.gptOss_20B
        #expect(config.family == .gptOss)
        #expect(config.variant == .gptOss_20B)
        #expect(config.hiddenSize == 2_880)
        #expect(config.numLayers == 24)
        #expect(config.numHeads == 64)
        #expect(config.numKVHeads == 8)
        #expect(config.headDim == 64)
        #expect(config.numExperts == 32)
        #expect(config.topKExperts == 4)
        #expect(config.slidingWindow == 128)
        #expect(config.fullAttentionLayerMask == (0..<24).map {
            $0.isMultiple(of: 2) ? 0 : 1
        })
        #expect(config.attentionSinks)
        #expect(config.attentionScale == 0.125)
        #expect(config.swigluLimit == 7)
        #expect(config.yarnRope == YaRNRopeConfig(
            originalContextLength: 4_096,
            scalingFactor: 32,
            betaFast: 32,
            betaSlow: 1))
        #expect(config.decodeInt4GEMVShapes.isEmpty)
    }

    @Test func gptOss120BProfileMatchesPinnedCheckpoint() {
        let config = ArchConfig.gptOss_120B
        #expect(config.family == .gptOss)
        #expect(config.variant == .gptOss_120B)
        #expect(config.hiddenSize == 2_880)
        #expect(config.numLayers == 36)
        #expect(config.numHeads == 64)
        #expect(config.numKVHeads == 8)
        #expect(config.headDim == 64)
        #expect(config.numExperts == 128)
        #expect(config.topKExperts == 4)
        #expect(config.slidingWindow == 128)
        #expect(config.fullAttentionLayerMask == (0..<36).map {
            $0.isMultiple(of: 2) ? 0 : 1
        })
        #expect(config.attentionSinks)
        #expect(config.attentionScale == 0.125)
        #expect(config.swigluLimit == 7)
        #expect(config.yarnRope == ArchConfig.gptOss_20B.yarnRope)
        #expect(config.decodeInt4GEMVShapes.isEmpty)
    }
}
