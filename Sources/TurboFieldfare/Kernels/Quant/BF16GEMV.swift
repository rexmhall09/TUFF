import Metal

/// BF16-weight projection with FP16 activations. GPT-OSS keeps its resident
/// matrices in BF16 while only routed expert matrices use MXFP4.
final class BF16GEMV {
    private static let rowsPerThreadgroup = 8
    private let halfPipeline: MTLComputePipelineState
    private let floatPipeline: MTLComputePipelineState
    private let embeddingPipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        halfPipeline = try context.pipeline(
            "bf16_gemv_half_simd", constants: [],
            maxTotalThreadsPerThreadgroup: 32 * Self.rowsPerThreadgroup)
        floatPipeline = try context.pipeline(
            "bf16_gemv_float_simd", constants: [],
            maxTotalThreadsPerThreadgroup: 32 * Self.rowsPerThreadgroup)
        embeddingPipeline = try context.pipeline("bf16_embedding_lookup_half")
    }

    func encodeHalf(
        commandBuffer: MTLCommandBuffer,
        weights: TensorView,
        input: MTLBuffer,
        output: MTLBuffer,
        outputOffset: Int = 0,
        bias: TensorView? = nil,
        rows: Int,
        columns: Int
    ) {
        encode(commandBuffer: commandBuffer,
               pipeline: halfPipeline,
               weights: weights,
               input: input,
               output: output,
               outputOffset: outputOffset,
               bias: bias,
               rows: rows,
               columns: columns)
    }

    func encodeFloat(
        commandBuffer: MTLCommandBuffer,
        weights: TensorView,
        input: MTLBuffer,
        output: MTLBuffer,
        outputOffset: Int = 0,
        bias: TensorView? = nil,
        rows: Int,
        columns: Int
    ) {
        encode(commandBuffer: commandBuffer,
               pipeline: floatPipeline,
               weights: weights,
               input: input,
               output: output,
               outputOffset: outputOffset,
               bias: bias,
               rows: rows,
               columns: columns)
    }

    func encodeEmbedding(
        commandBuffer: MTLCommandBuffer,
        table: TensorView,
        token: UInt32,
        output: MTLBuffer,
        hiddenSize: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(embeddingPipeline)
        encoder.setBuffer(table.buffer, offset: Int(table.offset), index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        var tokenValue = token
        var hiddenValue = UInt32(hiddenSize)
        encoder.setBytes(&tokenValue, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&hiddenValue, length: MemoryLayout<UInt32>.size, index: 3)
        let width = min(hiddenSize, embeddingPipeline.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: hiddenSize, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: max(1, width), height: 1, depth: 1))
        encoder.endEncoding()
    }

    private func encode(
        commandBuffer: MTLCommandBuffer,
        pipeline: MTLComputePipelineState,
        weights: TensorView,
        input: MTLBuffer,
        output: MTLBuffer,
        outputOffset: Int,
        bias: TensorView?,
        rows: Int,
        columns: Int
    ) {
        precondition(rows > 0 && columns > 0)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weights.buffer, offset: Int(weights.offset), index: 0)
        encoder.setBuffer(input, offset: 0, index: 1)
        encoder.setBuffer(output, offset: outputOffset, index: 2)
        encoder.setBuffer(bias?.buffer ?? weights.buffer,
                          offset: bias.map { Int($0.offset) } ?? Int(weights.offset),
                          index: 3)
        var rowValue = UInt32(rows)
        var columnValue = UInt32(columns)
        var hasBias: UInt32 = bias == nil ? 0 : 1
        encoder.setBytes(&rowValue, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&columnValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&hasBias, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.dispatchThreadgroups(
            MTLSize(width: (rows + Self.rowsPerThreadgroup - 1)
                    / Self.rowsPerThreadgroup,
                    height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: 32 * Self.rowsPerThreadgroup, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
