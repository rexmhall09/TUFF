import Foundation
import Testing
@testable import TurboFieldfare

@Suite struct Gemma4E4BArchitectureTests {
    @Test func productionProfileMatchesPinnedCheckpoint() {
        let config = ArchConfig.gemma4_E4B
        #expect(config.variant == .gemma4_E4B)
        #expect(config.feedForwardKind == .dense)
        #expect(config.hiddenSize == 2_560)
        #expect(config.intermediateSize == 10_240)
        #expect(config.numLayers == 42)
        #expect(config.numExperts == 0)
        #expect(config.hiddenSizePerLayerInput == 256)
        #expect(config.vocabSizePerLayerInput == 262_144)
        #expect(config.numKVSharedLayers == 18)
        #expect(config.fullAttentionLayerMask.enumerated().compactMap {
            $0.element == 1 ? $0.offset : nil
        } == [5, 11, 17, 23, 29, 35, 41])
    }

    @Test func perLayerEmbeddingPlanSlicesAndScalesToySignals() {
        let plan = PerLayerEmbeddingPlan(config: .gemma4_E4B)
        #expect(plan.packedWidth == 10_752)
        #expect(plan.packedRange(layer: 0) == 0..<256)
        #expect(plan.packedRange(layer: 41) == 10_496..<10_752)
        #expect(abs(plan.tokenIdentityScale - 16) < 0.000_001)
        #expect(abs(plan.contextProjectionScale - (1 / sqrt(Float(2_560)))) < 0.000_001)

        let token = [Float](repeating: 0.25, count: 256)
        let context = [Float](repeating: -0.5, count: 256)
        let combined = plan.combine(tokenIdentity: token, contextAware: context)
        let expected = (Float(4) - 0.5) / sqrt(Float(2))
        #expect(combined.allSatisfy { abs($0 - expected) < 0.000_001 })
    }

    @Test func sharedKVLayersResolveByAttentionKind() {
        let config = ArchConfig.gemma4_E4B
        #expect(config.firstKVSharedLayer == 24)
        #expect(config.kvSourceLayer(for: 23) == nil)
        #expect(config.kvSourceLayer(for: 24) == 22)
        #expect(config.kvSourceLayer(for: 29) == 23)
        #expect(config.kvSourceLayer(for: 41) == 23)
    }

    @Test func denseProfileDoesNotCompileRouterShapes() {
        let config = ArchConfig.gemma4_E4B
        #expect(config.decodeInt8GEMVShapes.isEmpty)
        #expect(config.decodeInt4GEMVShapes.contains { $0 == (10_752, 2_560) })
        #expect(config.decodeInt4GEMVShapes.contains { $0 == (256, 2_560) })
        #expect(config.decodeInt4GEMVShapes.contains { $0 == (2_560, 256) })
    }
}
