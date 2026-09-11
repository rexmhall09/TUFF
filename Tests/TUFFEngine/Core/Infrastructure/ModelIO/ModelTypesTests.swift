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

    @Test func qwen38FlashNextProfileMatchesPinnedCheckpoint() {
        let config = ArchConfig.qwen38FlashNext
        #expect(config.family == .qwen4Exp)
        #expect(config.variant == .qwen38FlashNext)
        #expect(ArchConfig.registeredArchitectures[.qwen38FlashNext] == config)
        #expect(config.hiddenSize == 2_560)
        #expect(config.numLayers == 48)
        #expect(config.numExperts == 512)
        #expect(config.topKExperts == 10)
        #expect(config.vocabSize == 248_320)
        #expect(!config.tieWordEmbeddings)

        // 12 full-attention layers, every fourth; the rest gated DeltaNet.
        #expect(config.fullAttentionLayerMask.count == 48)
        #expect((0..<48).filter { config.layerIsFull($0) } == Array(stride(
            from: 3, to: 48, by: 4)))
        #expect((0..<48).filter { config.layerIsLinear($0) }.count == 36)
        #expect(config.hasLinearAttentionLayers)
        #expect(config.hasSharedExpert)

        // The gated-DeltaNet bundle is Qwen3.6's contract at wider dimensions.
        #expect(config.linearAttention.qkvDim == 10_240)
        #expect(config.linearAttention.valueDim == 6_144)

        // A token costs 24 KiB of KV cache: 12 full-attention layers, two KV
        // heads, a 256 head dim, key and value, at two bytes each.
        #expect(config.numFullKVHeads * config.fullHeadDim * 2 * 2 * 12 == 24_576)

        #expect(config.hyperConnection.streamCount == 4)
        #expect(config.hyperConnection.lowRank == 320)
        #expect(config.hyperConnection.stackedWidth(
            hiddenSize: config.hiddenSize) == 10_240)
        #expect(config.ngramEmbedding.isEnabled)
        #expect(config.ngramEmbedding.layer == 1)
        #expect(config.ngramEmbedding.shardCount == 128)
        #expect(config.ngramEmbedding.headDim == 160)
        #expect(config.attentionIndexer.isEnabled)
        #expect(config.attentionIndexer.budget == 2_048)
        // Four query heads and one key head through one fused projection.
        #expect(config.attentionIndexer.projectionRows == 640)
    }

    /// Every other architecture keeps one residual stream, no n-gram table and
    /// dense attention, which is what the manifest's absent fields mean.
    @Test func onlyQwen4ExpCarriesTheNewMechanisms() {
        for (variant, config) in ArchConfig.registeredArchitectures
        where variant != .qwen38FlashNext {
            #expect(config.hyperConnection == .none, "\(variant)")
            #expect(config.ngramEmbedding == .none, "\(variant)")
            #expect(config.attentionIndexer == .none, "\(variant)")
        }
    }

    @Test func minimaxM27ProfileMatchesPinnedCheckpoint() {
        let config = ArchConfig.minimaxM27
        #expect(config.family == .minimaxM2)
        #expect(config.variant == .minimaxM27)
        #expect(config.hiddenSize == 3_072)
        #expect(config.intermediateSize == 1_536)
        #expect(config.numLayers == 62)
        #expect(config.numHeads == 48)
        #expect(config.numKVHeads == 8)
        #expect(config.headDim == 128)
        #expect(config.numExperts == 256)
        #expect(config.topKExperts == 8)
        #expect(config.vocabSize == 200_064)
        #expect(config.fullAttentionLayerMask == [UInt8](repeating: 1, count: 62))
        #expect(config.partialRotaryFactor == 0.5)
        #expect(config.usesProjectionWideQKNorm)
        #expect(config.usesSigmoidCorrectionRouter)
        #expect(!config.hasSharedExpert)
    }
}
