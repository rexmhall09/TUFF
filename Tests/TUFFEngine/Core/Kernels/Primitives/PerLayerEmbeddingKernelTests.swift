import Testing
import Metal
@testable import TUFFEngine

@Suite struct PerLayerEmbeddingKernelTests {
    @Test func packedLayerPreparationMatchesCPUReference() throws {
        let context = try MetalContext()
        let kernel = try PerLayerEmbedding(context: context)
        let rows = 2
        let layers = 2
        let width = 64
        let packed = layers * width
        let identity = (0..<(rows * packed)).map {
            Float16(Float(($0 % 17) - 8) * 0.015625)
        }
        let projected = (0..<(rows * packed)).map {
            Float16(Float(($0 % 23) - 11) * 0.03125)
        }
        let normBits = (0..<width).map {
            Quantization.bf16Bits(0.75 + Float($0 % 7) * 0.05)
        }
        let identityBuffer = context.device.makeBuffer(
            bytes: identity, length: identity.count * 2, options: .storageModeShared)!
        let projectedBuffer = context.device.makeBuffer(
            bytes: projected, length: projected.count * 2, options: .storageModeShared)!
        let normBuffer = context.device.makeBuffer(
            bytes: normBits, length: normBits.count * 2, options: .storageModeShared)!
        let output = context.device.makeBuffer(
            length: rows * width * 2, options: .storageModeShared)!
        let commandBuffer = context.queue.makeCommandBuffer()!
        kernel.encodePrepareLayer(
            commandBuffer: commandBuffer,
            identity: identityBuffer,
            context: projectedBuffer,
            normWeight: normBuffer,
            out: output,
            rows: rows,
            packedWidth: packed,
            pleWidth: width,
            layer: 1,
            contextScale: 0.125,
            combinedScale: Float(1 / Double(2).squareRoot()),
            eps: 1e-6)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try #require(commandBuffer.error == nil)

        let actual = output.contents().bindMemory(
            to: Float16.self, capacity: rows * width)
        for row in 0..<rows {
            let base = row * packed + width
            let scaled = (0..<width).map { Float(projected[base + $0]) * 0.125 }
            let meanSquare = scaled.reduce(Float(0)) { $0 + $1 * $1 } / Float(width)
            let inv = 1 / sqrt(meanSquare + 1e-6)
            for column in 0..<width {
                let weight = Quantization.bf16ToFloat(normBits[column])
                let expected = (scaled[column] * inv * weight
                    + Float(identity[base + column])) / sqrt(2)
                #expect(abs(Float(actual[row * width + column]) - expected) < 0.004)
            }
        }
    }
}
