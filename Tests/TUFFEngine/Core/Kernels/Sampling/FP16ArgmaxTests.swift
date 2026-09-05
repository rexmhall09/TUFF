import Metal
import Testing
@testable import TUFFEngine
import TUFFValidationSupport

@Suite struct FP16ArgmaxTests {
    @Test(arguments: [false, true])
    func ignoresNonfiniteValuesAndUsesZeroWhenNoneAreFinite(allInvalid: Bool) throws {
        let context = try MetalContext()
        let kernel = try FP16Argmax(context: context)
        var values = (0..<513).map { index -> Float16 in
            switch index % 3 {
            case 0: return .nan
            case 1: return .infinity
            default: return -.infinity
            }
        }
        if !allInvalid {
            values[37] = 4
            values[300] = 4
        }
        let input = try #require(Fp16Buffer.make(context.device, halves: values))
        let output = try #require(context.device.makeBuffer(
            length: MemoryLayout<UInt32>.stride, options: .storageModeShared))
        let command = try #require(context.queue.makeCommandBuffer())
        kernel.encode(commandBuffer: command, values: input,
                      count: values.count, output: output)
        command.commit()
        command.waitUntilCompleted()
        try checkCommandBufferError(command.error)

        #expect(output.contents().load(as: UInt32.self) == (allInvalid ? 0 : 37))
    }
}
