import Metal
import Testing
@testable import TUFFEngine
import TUFFValidationSupport

@Suite struct MXFP4GEMVTests {
    private static func runAndCompare(rows: Int, columns: Int,
                                      includesBias: Bool = false) throws {
        precondition(columns % Quantization.mxfp4GroupSize == 0)
        let packed = (0..<(rows * columns / 2)).map {
            UInt8(truncatingIfNeeded: $0 * 73 + rows * 19 + columns)
        }
        let scales = (0..<(rows * columns / Quantization.mxfp4GroupSize)).map {
            UInt8(120 + (($0 * 7 + rows) % 13))
        }
        let input = (0..<columns).map {
            Float16(Float(($0 * 29 + 3) % 127 - 63) / 64)
        }
        let reference = MXFP4Reference.gemv(
            packed: packed, scales: scales,
            x: input.map(Float.init), rows: rows, columns: columns)
        let biasBits = (0..<rows).map {
            Quantization.bf16Bits(Float(($0 * 11) % 17 - 8) / 16)
        }
        let biasedReference = zip(reference, biasBits).map {
            $0 + (includesBias ? Quantization.bf16ToFloat($1) : 0)
        }

        let context = try MetalContext()
        let kernel = try MXFP4GEMV(context: context)
        guard let weightsBuffer = context.device.makeBuffer(
                bytes: packed, length: packed.count, options: .storageModeShared),
              let scalesBuffer = context.device.makeBuffer(
                bytes: scales, length: scales.count, options: .storageModeShared),
              let biasBuffer = context.device.makeBuffer(
                bytes: biasBits,
                length: biasBits.count * MemoryLayout<UInt16>.stride,
                options: .storageModeShared),
              let inputBuffer = Fp16Buffer.make(context.device, halves: input),
              let outputBuffer = Fp16Buffer.make(context.device, count: rows),
              let commandBuffer = context.queue.makeCommandBuffer() else {
            Issue.record("failed to allocate MXFP4 Metal resources")
            return
        }
        kernel.encode(
            commandBuffer: commandBuffer,
            weights: weightsBuffer, scales: scalesBuffer,
            input: inputBuffer, output: outputBuffer,
            bias: includesBias ? biasBuffer : nil,
            rows: UInt32(rows), columns: UInt32(columns))
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try checkCommandBufferError(commandBuffer.error)

        let actual = Fp16Buffer.read(outputBuffer, count: rows)
        let relative = RelError.compute(actual: actual, reference: biasedReference)
        #expect(relative < Tolerance.fp16Reduction,
                "rows=\(rows) columns=\(columns) relative=\(relative)")
    }

    @Test func oneBlockAndPartialThreadgroup() throws {
        try Self.runAndCompare(rows: 7, columns: 32)
    }

    @Test func crossesThreadgroupAndBlockBoundaries() throws {
        try Self.runAndCompare(rows: 17, columns: 96)
    }

    @Test func productionInputWidth() throws {
        try Self.runAndCompare(rows: 33, columns: 2_880)
    }

    @Test func bf16BiasMatchesReference() throws {
        try Self.runAndCompare(rows: 19, columns: 96, includesBias: true)
    }
}
