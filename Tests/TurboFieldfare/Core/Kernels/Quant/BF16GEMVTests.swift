import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct BF16GEMVTests {
    private static func view(_ buffer: MTLBuffer, elements: Int,
                             rows: Int, columns: Int) -> TensorView {
        TensorView(buffer: buffer,
                   offset: 0,
                   length: UInt64(elements * MemoryLayout<UInt16>.stride),
                   scaleOffset: 0,
                   scaleLength: 0,
                   biasOffset: 0,
                   biasLength: 0,
                   shape: (UInt32(rows), UInt32(columns), 0, 0),
                   dtype: 1)
    }

    @Test func halfAndFloatOutputsMatchBF16Reference() throws {
        let context = try MetalContext()
        let kernel = try BF16GEMV(context: context)
        let rows = 13
        let columns = 96
        let weights = (0..<(rows * columns)).map {
            Quantization.bf16Bits(Float(($0 * 17 + 5) % 61 - 30) / 64)
        }
        let biases = (0..<rows).map {
            Quantization.bf16Bits(Float(($0 * 11 + 3) % 19 - 9) / 32)
        }
        let input = (0..<columns).map {
            Float16(Float(($0 * 7 + 1) % 29 - 14) / 32)
        }
        let weightValues = weights.map(Quantization.bf16ToFloat)
        let biasValues = biases.map(Quantization.bf16ToFloat)
        let reference = (0..<rows).map { row in
            var sum = biasValues[row]
            for column in 0..<columns {
                sum += weightValues[row * columns + column] * Float(input[column])
            }
            return sum
        }

        let weightBuffer = try #require(context.device.makeBuffer(
            bytes: weights,
            length: weights.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))
        let biasBuffer = try #require(context.device.makeBuffer(
            bytes: biases,
            length: biases.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))
        let inputBuffer = try #require(Fp16Buffer.make(
            context.device, halves: input))
        let halfOutput = try #require(context.device.makeBuffer(
            length: (rows + 4) * MemoryLayout<Float16>.stride,
            options: .storageModeShared))
        let floatOutput = try #require(context.device.makeBuffer(
            length: rows * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        memset(halfOutput.contents(), 0, halfOutput.length)

        let weightsView = Self.view(weightBuffer, elements: weights.count,
                                    rows: rows, columns: columns)
        let biasView = Self.view(biasBuffer, elements: biases.count,
                                 rows: 1, columns: rows)
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        kernel.encodeHalf(commandBuffer: commandBuffer,
                          weights: weightsView,
                          input: inputBuffer,
                          output: halfOutput,
                          outputOffset: 2 * MemoryLayout<Float16>.stride,
                          bias: biasView,
                          rows: rows,
                          columns: columns)
        kernel.encodeFloat(commandBuffer: commandBuffer,
                           weights: weightsView,
                           input: inputBuffer,
                           output: floatOutput,
                           bias: biasView,
                           rows: rows,
                           columns: columns)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try checkCommandBufferError(commandBuffer.error)

        let halfPointer = halfOutput.contents()
            .assumingMemoryBound(to: Float16.self)
        let actualHalf = (0..<rows).map { Float(halfPointer[$0 + 2]) }
        let floatPointer = floatOutput.contents()
            .assumingMemoryBound(to: Float.self)
        let actualFloat = (0..<rows).map { floatPointer[$0] }
        #expect(RelError.compute(actual: actualHalf, reference: reference)
                < Tolerance.fp16Reduction)
        #expect(RelError.compute(actual: actualFloat, reference: reference)
                < 0.00001)
        #expect(halfPointer[0] == 0)
        #expect(halfPointer[1] == 0)
    }

    @Test func embeddingLookupUsesBF16Rows() throws {
        let context = try MetalContext()
        let kernel = try BF16GEMV(context: context)
        let vocab = 7
        let hidden = 65
        let table = (0..<(vocab * hidden)).map {
            Quantization.bf16Bits(Float(($0 * 13) % 47 - 23) / 32)
        }
        let tableBuffer = try #require(context.device.makeBuffer(
            bytes: table,
            length: table.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))
        let output = try #require(Fp16Buffer.make(context.device, count: hidden))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        kernel.encodeEmbedding(
            commandBuffer: commandBuffer,
            table: Self.view(tableBuffer, elements: table.count,
                             rows: vocab, columns: hidden),
            token: 5,
            output: output,
            hiddenSize: hidden)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try checkCommandBufferError(commandBuffer.error)

        let actual = Fp16Buffer.read(output, count: hidden)
        let expected = table[(5 * hidden)..<(6 * hidden)].map {
            Quantization.bf16ToFloat($0)
        }
        #expect(actual == expected)
    }
}
