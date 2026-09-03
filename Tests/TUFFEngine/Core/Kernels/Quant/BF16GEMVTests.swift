import Metal
import Testing
@testable import TUFFEngine
import TUFFValidationSupport

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
        let inputPrefix: [Float16] = [99, -99]
        let inputBuffer = try #require(Fp16Buffer.make(
            context.device, halves: inputPrefix + input))
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
                          inputOffset: inputPrefix.count
                            * MemoryLayout<Float16>.stride,
                          output: halfOutput,
                          outputOffset: 2 * MemoryLayout<Float16>.stride,
                          bias: biasView,
                          rows: rows,
                          columns: columns)
        kernel.encodeFloat(commandBuffer: commandBuffer,
                           weights: weightsView,
                           input: inputBuffer,
                           inputOffset: inputPrefix.count
                            * MemoryLayout<Float16>.stride,
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
        let output = try #require(Fp16Buffer.make(
            context.device, halves: [42, -42]
                + [Float16](repeating: 0, count: hidden)))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        kernel.encodeEmbedding(
            commandBuffer: commandBuffer,
            table: Self.view(tableBuffer, elements: table.count,
                             rows: vocab, columns: hidden),
            token: 5,
            output: output,
            outputOffset: 2 * MemoryLayout<Float16>.stride,
            hiddenSize: hidden)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try checkCommandBufferError(commandBuffer.error)

        let values = Fp16Buffer.read(output, count: hidden + 2)
        #expect(Array(values[0..<2]) == [42, -42])
        let actual = Array(values.dropFirst(2))
        let expected = table[(5 * hidden)..<(6 * hidden)].map {
            Quantization.bf16ToFloat($0)
        }
        #expect(actual == expected)
    }

    @Test func floatEmbeddingPreservesValuesBeyondFloat16Range() throws {
        let context = try MetalContext()
        let kernel = try BF16GEMV(context: context)
        let vocab = 3
        let hidden = 67
        var table = [UInt16](repeating: Quantization.bf16Bits(0),
                             count: vocab * hidden)
        for index in 0..<hidden {
            let value = index.isMultiple(of: 2)
                ? Float(100_000 + index * 256)
                : Float(-100_000 - index * 256)
            table[hidden + index] = Quantization.bf16Bits(value)
        }
        let tableBuffer = try #require(context.device.makeBuffer(
            bytes: table,
            length: table.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))
        let prefix: [Float] = [42, -42]
        let output = try #require(context.device.makeBuffer(
            bytes: prefix + [Float](repeating: 0, count: hidden),
            length: (prefix.count + hidden) * MemoryLayout<Float>.stride,
            options: .storageModeShared))
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        kernel.encodeFloatEmbedding(
            commandBuffer: commandBuffer,
            table: Self.view(tableBuffer, elements: table.count,
                             rows: vocab, columns: hidden),
            token: 1,
            output: output,
            outputOffset: prefix.count * MemoryLayout<Float>.stride,
            hiddenSize: hidden)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try checkCommandBufferError(commandBuffer.error)

        let pointer = output.contents().assumingMemoryBound(to: Float.self)
        #expect(pointer[0] == 42)
        #expect(pointer[1] == -42)
        let actual = (0..<hidden).map { pointer[prefix.count + $0] }
        let expected = table[hidden..<(2 * hidden)].map {
            Quantization.bf16ToFloat($0)
        }
        #expect(actual == expected)
        #expect(actual.allSatisfy { abs($0) > Float(Float16.greatestFiniteMagnitude) })
    }

    @Test func batchedArgmaxMatchesPerRowProjectionAndArgmax() throws {
        let context = try MetalContext()
        let kernel = try BF16GEMV(context: context, maxRows: 4,
                                  maxVocabularyRows: 64)
        let argmax = try FP16Argmax(context: context)
        let rows = 37
        let columns = 64
        let batch = 4
        let stride = columns + 5
        let weights = (0..<(rows * columns)).map { index in
            Quantization.bf16Bits(Float((index * 19 + 7) % 101 - 50) / 64)
        }
        let inputs = (0..<(batch * stride)).map { index in
            index % stride < columns
                ? Float16(Float((index * 13 + 3) % 67 - 33) / 32)
                : Float16(91)
        }
        let weightBuffer = try #require(context.device.makeBuffer(
            bytes: weights,
            length: weights.count * MemoryLayout<UInt16>.stride,
            options: .storageModeShared))
        let inputBuffer = try #require(Fp16Buffer.make(
            context.device, halves: [Float16(-17), Float16(23)] + inputs))
        let oldLogits = try #require(Fp16Buffer.make(
            context.device, count: batch * rows))
        let oldTokens = try #require(context.device.makeBuffer(
            length: batch * MemoryLayout<UInt32>.stride,
            options: .storageModeShared))
        let newTokens = try #require(context.device.makeBuffer(
            length: (batch + 1) * MemoryLayout<UInt32>.stride,
            options: .storageModeShared))
        let weightsView = Self.view(weightBuffer, elements: weights.count,
                                    rows: rows, columns: columns)
        let inputOffset = 2 * MemoryLayout<Float16>.stride
        let commandBuffer = try #require(context.queue.makeCommandBuffer())
        for row in 0..<batch {
            let inputRowOffset = inputOffset
                + row * stride * MemoryLayout<Float16>.stride
            let logitsOffset = row * rows * MemoryLayout<Float16>.stride
            kernel.encodeHalf(commandBuffer: commandBuffer,
                              weights: weightsView,
                              input: inputBuffer,
                              inputOffset: inputRowOffset,
                              output: oldLogits,
                              outputOffset: logitsOffset,
                              rows: rows,
                              columns: columns)
            argmax.encode(commandBuffer: commandBuffer,
                          values: oldLogits,
                          valuesOffset: logitsOffset,
                          count: rows,
                          output: oldTokens,
                          outputOffset: row * MemoryLayout<UInt32>.stride)
        }
        kernel.encodeHalfArgmaxRows(
            commandBuffer: commandBuffer,
            weights: weightsView,
            input: inputBuffer,
            inputOffset: inputOffset,
            inputStrideElements: stride,
            output: newTokens,
            outputOffset: MemoryLayout<UInt32>.stride,
            rowCount: batch,
            rows: rows,
            columns: columns)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try checkCommandBufferError(commandBuffer.error)

        let oldPointer = oldTokens.contents().assumingMemoryBound(to: UInt32.self)
        let newPointer = newTokens.contents().assumingMemoryBound(to: UInt32.self)
        for row in 0..<batch {
            #expect(newPointer[row + 1] == oldPointer[row])
        }
    }
}
