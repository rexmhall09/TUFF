import Darwin
import Metal
import Testing
@testable import TUFFEngine

@Suite struct QwenVisionPrimitiveTests {
    @Test func architectureAndMergerContractMatchQwen36() {
        let config = VisionConfig(family: .qwen36)
        #expect(config.numLayers == 27)
        #expect(config.hiddenSize == 1_152)
        #expect(config.intermediateSize == 4_304)
        #expect(config.numHeads == 16)
        #expect(config.patchSize == 16)
        #expect(config.temporalPatchSize == 2)
        #expect(abs(config.attentionScale - 1 / sqrt(Float(72))) < 0.000_001)
        #expect(config.patchDimension == 1_536)
        #expect(config.textHiddenSize == 2_048)
        #expect(config.positionEmbeddingSize == 2_304)
        #expect(config.positionGridSide == 48)
        #expect(config.poolingKernel == 2)
        #expect(config.ropeTheta == 10_000)
        #expect(VisionRuntime.qwenMergerTensorNames.count == 6)
    }

    @Test func affineLayerNormAndExactMergerGELUMatchCPUOracles() throws {
        let context = try MetalContext()
        let primitives = try VisionPrimitives(context: context)
        func bf16Buffer(_ values: [Float]) throws -> MTLBuffer {
            let bits = values.map(Quantization.bf16Bits)
            return try #require(context.device.makeBuffer(
                bytes: bits, length: bits.count * 2,
                options: .storageModeShared))
        }
        func floats(_ buffer: MTLBuffer, count: Int) -> [Float] {
            let pointer = buffer.contents().bindMemory(to: UInt16.self, capacity: count)
            return (0..<count).map { Quantization.bf16ToFloat(pointer[$0]) }
        }

        let input = try bf16Buffer([1, 2, 3, 4])
        let affine = try bf16Buffer([1, 1, 1, 1, 0, 0, 0, 0])
        let normalized = try #require(context.device.makeBuffer(
            length: 8, options: .storageModeShared))
        let layerNormCommands = try #require(context.queue.makeCommandBuffer())
        primitives.encodeQwenLayerNorm(
            commandBuffer: layerNormCommands,
            input: input, output: normalized,
            weights: affine, weightOffset: 0, biasOffset: 8,
            rows: 1, width: 4)
        layerNormCommands.commit()
        layerNormCommands.waitUntilCompleted()
        #expect(layerNormCommands.error == nil)
        let actualNorm = floats(normalized, count: 4)
        let expectedNorm: [Float] = [-1.3416402, -0.4472134, 0.4472134, 1.3416402]
        #expect(zip(actualNorm, expectedNorm).allSatisfy { abs($0 - $1) < 0.01 })

        let geluInputValues: [Float] = [-2, -1, 0, 1, 2]
        let gelu = try bf16Buffer(geluInputValues)
        let zeroBias = try bf16Buffer([0, 0, 0, 0, 0])
        let geluCommands = try #require(context.queue.makeCommandBuffer())
        primitives.encodeQwenGELUErfBias(
            commandBuffer: geluCommands,
            values: gelu, weights: zeroBias, biasOffset: 0,
            count: geluInputValues.count, width: geluInputValues.count)
        geluCommands.commit()
        geluCommands.waitUntilCompleted()
        #expect(geluCommands.error == nil)
        let actualGELU = floats(gelu, count: geluInputValues.count)
        let expectedGELU = geluInputValues.map {
            0.5 * $0 * (1 + Float(erf(Double($0) / sqrt(2))))
        }
        #expect(zip(actualGELU, expectedGELU).allSatisfy { abs($0 - $1) < 0.01 })
    }
}
