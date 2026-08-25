import Foundation
import Metal
import Testing
@testable import TurboFieldfare
import TurboFieldfareValidationSupport

@Suite struct GPTOSSExpertRuntimeTests {
    private struct ExpertFixture {
        let view: TensorView
        let offsets: GPTOSSExpertOffsets
        let mlp1Packed: [UInt8]
        let mlp1Scales: [UInt8]
        let mlp1Bias: [UInt16]
        let mlp2Packed: [UInt8]
        let mlp2Scales: [UInt8]
        let mlp2Bias: [UInt16]
    }

    private static func append(_ values: [UInt16], to bytes: inout [UInt8]) {
        for value in values {
            let little = value.littleEndian
            bytes.append(UInt8(truncatingIfNeeded: little))
            bytes.append(UInt8(truncatingIfNeeded: little >> 8))
        }
    }

    private static func fixture(context: MetalContext,
                                hidden: Int,
                                intermediate: Int,
                                seed: Int) throws -> ExpertFixture {
        let mlp1Packed = (0..<(2 * intermediate * hidden / 2)).map {
            UInt8(truncatingIfNeeded: $0 * 29 + seed * 17)
        }
        let mlp1Scales = (0..<(2 * intermediate * hidden / 32)).map {
            UInt8(122 + ($0 + seed) % 9)
        }
        let mlp1Bias = (0..<(2 * intermediate)).map {
            Quantization.bf16Bits(Float(($0 * 7 + seed) % 13 - 6) / 32)
        }
        let mlp2Packed = (0..<(hidden * intermediate / 2)).map {
            UInt8(truncatingIfNeeded: $0 * 43 + seed * 11)
        }
        let mlp2Scales = (0..<(hidden * intermediate / 32)).map {
            UInt8(121 + ($0 * 3 + seed) % 10)
        }
        let mlp2Bias = (0..<hidden).map {
            Quantization.bf16Bits(Float(($0 * 5 + seed) % 11 - 5) / 32)
        }

        var bytes: [UInt8] = []
        func pad16() {
            while !bytes.count.isMultiple(of: 16) { bytes.append(0) }
        }
        let mlp1Weights = bytes.count
        bytes.append(contentsOf: mlp1Packed)
        pad16()
        let mlp1ScaleOffset = bytes.count
        bytes.append(contentsOf: mlp1Scales)
        pad16()
        let mlp1BiasOffset = bytes.count
        append(mlp1Bias, to: &bytes)
        pad16()
        let mlp2Weights = bytes.count
        bytes.append(contentsOf: mlp2Packed)
        pad16()
        let mlp2ScaleOffset = bytes.count
        bytes.append(contentsOf: mlp2Scales)
        pad16()
        let mlp2BiasOffset = bytes.count
        append(mlp2Bias, to: &bytes)
        pad16()
        let buffer = try #require(context.device.makeBuffer(
            bytes: bytes, length: bytes.count, options: .storageModeShared))
        return ExpertFixture(
            view: TensorView(
                buffer: buffer,
                offset: 0,
                length: UInt64(bytes.count),
                scaleOffset: 0,
                scaleLength: 0,
                biasOffset: 0,
                biasLength: 0,
                shape: (0, 0, 0, 0),
                dtype: 0),
            offsets: GPTOSSExpertOffsets(
                mlp1Weights: mlp1Weights,
                mlp1Scales: mlp1ScaleOffset,
                mlp1Bias: mlp1BiasOffset,
                mlp2Weights: mlp2Weights,
                mlp2Scales: mlp2ScaleOffset,
                mlp2Bias: mlp2BiasOffset),
            mlp1Packed: mlp1Packed,
            mlp1Scales: mlp1Scales,
            mlp1Bias: mlp1Bias,
            mlp2Packed: mlp2Packed,
            mlp2Scales: mlp2Scales,
            mlp2Bias: mlp2Bias)
    }

    private static func reference(fixture: ExpertFixture,
                                  input: [Float],
                                  hidden: Int,
                                  intermediate: Int) -> [Float] {
        var projected = MXFP4Reference.gemv(
            packed: fixture.mlp1Packed,
            scales: fixture.mlp1Scales,
            x: input,
            rows: 2 * intermediate,
            columns: hidden)
        for index in projected.indices {
            projected[index] += Quantization.bf16ToFloat(fixture.mlp1Bias[index])
            projected[index] = Float(Float16(projected[index]))
        }
        let gate = (0..<intermediate).map { projected[$0 * 2] }
        let linear = (0..<intermediate).map { projected[$0 * 2 + 1] }
        let activated = GPTOSSMoERef.cappedSwiGLU(gate: gate, linear: linear)
            .map { Float(Float16($0)) }
        var output = MXFP4Reference.gemv(
            packed: fixture.mlp2Packed,
            scales: fixture.mlp2Scales,
            x: activated,
            rows: hidden,
            columns: intermediate)
        for index in output.indices {
            output[index] += Quantization.bf16ToFloat(fixture.mlp2Bias[index])
            output[index] = Float(Float16(output[index]))
        }
        return output
    }

    @Test func streamedDecodeAndPrefillRoutesMatchCPUReference() throws {
        let hidden = 64
        let intermediate = 32
        let queries = 3
        let topK = 4
        let context = try MetalContext()
        let runtime = try GPTOSSExpertRuntime(context: context)
        let layout = GPTOSSExpertScratchLayout(
            hiddenSize: hidden,
            intermediateSize: intermediate,
            topK: topK,
            queryCapacity: queries)
        let scratch = try GPTOSSExpertScratchBuffers.allocate(
            device: context.device, layout: layout)
        let fixtures = try (0..<topK).map {
            try Self.fixture(context: context,
                             hidden: hidden,
                             intermediate: intermediate,
                             seed: $0 + 1)
        }
        let inputs = (0..<(queries * hidden)).map {
            Float16(Float(($0 * 19 + 3) % 101 - 50) / 64)
        }
        let residual = (0..<(queries * hidden)).map {
            Float16(Float(($0 * 13 + 5) % 73 - 36) / 32)
        }
        let weights = (0..<(queries * topK)).map { index in
            Float16([0.4, 0.3, 0.2, 0.1][index % topK])
        }
        let inputBuffer = try #require(Fp16Buffer.make(context.device, halves: inputs))
        let residualBuffer = try #require(Fp16Buffer.make(context.device, halves: residual))
        let output = try #require(Fp16Buffer.make(context.device, count: queries * hidden))
        let weightPointer = scratch.routeWeights.contents()
            .bindMemory(to: Float16.self, capacity: weights.count)
        for index in weights.indices { weightPointer[index] = weights[index] }
        let commandBuffer = try #require(context.queue.makeCommandBuffer())

        for query in 0..<queries {
            for slot in 0..<topK {
                try runtime.encodeExpert(
                    commandBuffer: commandBuffer,
                    blob: fixtures[slot].view,
                    offsets: fixtures[slot].offsets,
                    input: inputBuffer,
                    queryIndex: query,
                    routeSlot: slot,
                    scratch: scratch)
            }
        }
        try runtime.encodeReduce(
            commandBuffer: commandBuffer,
            scratch: scratch,
            residual: residualBuffer,
            output: output,
            queryStart: 0,
            queryCount: 1)
        try runtime.encodeReduce(
            commandBuffer: commandBuffer,
            scratch: scratch,
            residual: residualBuffer,
            output: output,
            queryStart: 1,
            queryCount: queries - 1)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        try checkCommandBufferError(commandBuffer.error)

        var expected = [Float](repeating: 0, count: queries * hidden)
        for query in 0..<queries {
            let input = inputs[(query * hidden)..<((query + 1) * hidden)].map(Float.init)
            let expertOutputs = fixtures.map {
                Self.reference(fixture: $0, input: input,
                               hidden: hidden, intermediate: intermediate)
            }
            for dimension in 0..<hidden {
                var value = Float(residual[query * hidden + dimension])
                for slot in 0..<topK {
                    value += Float(weights[query * topK + slot])
                        * expertOutputs[slot][dimension]
                }
                expected[query * hidden + dimension] = value
            }
        }
        let actual = Fp16Buffer.read(output, count: expected.count)
        let error = RelError.compute(actual: actual, reference: expected)
        #expect(error < Tolerance.fp16ChainedReduction, "rel=\(error)")
    }

    @Test func productionScratchStaysBoundedAcrossDecodeAndPrefill() {
        let decode = GPTOSSExpertScratchLayout(
            hiddenSize: 2_880, intermediateSize: 2_880, queryCapacity: 1)
        let prefill = GPTOSSExpertScratchLayout(
            hiddenSize: 2_880, intermediateSize: 2_880, queryCapacity: 256)
        let clamped = GPTOSSExpertScratchLayout(
            hiddenSize: 2_880, intermediateSize: 2_880, queryCapacity: 10_000)
        #expect(decode.totalBytes == 40_328)
        #expect(prefill.totalBytes == 5_917_568)
        #expect(prefill.totalBytes < 6 * 1_048_576)
        #expect(clamped == prefill)
    }

    @Test func invalidBlobRangeFailsBeforeEncoding() throws {
        let offsets = GPTOSSExpertOffsets(
            mlp1Weights: 0, mlp1Scales: 16, mlp1Bias: 32,
            mlp2Weights: 48, mlp2Scales: 64, mlp2Bias: 80)
        #expect(throws: GPTOSSExpertRuntimeError.invalidBlobRange("mlp1")) {
            try offsets.validate(blobBytes: 96, hiddenSize: 64, intermediateSize: 32)
        }
    }
}
