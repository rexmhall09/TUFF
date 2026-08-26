import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct ElementwiseTests {
    @Test func floatResidualAddPreservesValuesBeyondFloat16Range() throws {
        let context = try MetalContext()
        let kernel = try Elementwise(context: context)
        let hiddenPrefix: [Float] = [31, -31]
        let hidden: [Float] = [100_000, -100_000, 70_000, -70_000, 65_600]
        let deltaPrefix: [Float16] = [3]
        let delta: [Float16] = [100, -100, 1, -1, 8]
        let hiddenBuffer = try #require(context.device.makeBuffer(
            bytes: hiddenPrefix + hidden,
            length: (hiddenPrefix.count + hidden.count) * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let deltaBuffer = try #require(Fp16Buffer.make(
            context.device, halves: deltaPrefix + delta))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        kernel.encodeFloatResidualAdd(
            commandBuffer: commandBuffer,
            hidden: hiddenBuffer,
            hiddenOffset: hiddenPrefix.count * MemoryLayout<Float>.stride,
            delta: deltaBuffer,
            deltaOffset: deltaPrefix.count * MemoryLayout<Float16>.stride,
            count: hidden.count)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try checkCommandBufferError(commandBuffer.error)

        let pointer = hiddenBuffer.contents().assumingMemoryBound(to: Float.self)
        #expect(pointer[0] == 31)
        #expect(pointer[1] == -31)
        let actual = (0..<hidden.count).map { pointer[hiddenPrefix.count + $0] }
        let expected = zip(hidden, delta).map { $0 + Float($1) }
        #expect(actual == expected)
        #expect(actual.allSatisfy { abs($0) > Float(Float16.greatestFiniteMagnitude) })
    }
}
