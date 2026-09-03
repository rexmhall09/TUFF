import Metal

/// BF16-weight projection with FP16 activations. GPT-OSS keeps its resident
/// matrices in BF16 while only routed expert matrices use MXFP4.
final class BF16GEMV {
    private static let rowsPerThreadgroup = 8
    private static let defaultMaxVocabularyRows = 262_144
    private let halfPipeline: MTLComputePipelineState
    private let floatPipeline: MTLComputePipelineState
    private let argmaxRowsPipeline: MTLComputePipelineState
    private let argmaxRowsReducePipeline: MTLComputePipelineState
    private let embeddingPipeline: MTLComputePipelineState
    private let floatEmbeddingPipeline: MTLComputePipelineState
    private let argmaxRowsSummaries: MTLBuffer
    private let maxRows: Int
    private let maxVocabularyRows: Int

    init(context: MetalContext,
         maxRows: Int = 8,
         maxVocabularyRows: Int = BF16GEMV.defaultMaxVocabularyRows) throws {
        precondition(maxRows > 0)
        precondition(maxVocabularyRows > 0)
        halfPipeline = try context.pipeline(
            "bf16_gemv_half_simd", constants: [],
            maxTotalThreadsPerThreadgroup: 32 * Self.rowsPerThreadgroup)
        floatPipeline = try context.pipeline(
            "bf16_gemv_float_simd", constants: [],
            maxTotalThreadsPerThreadgroup: 32 * Self.rowsPerThreadgroup)
        argmaxRowsPipeline = try context.pipeline(
            "bf16_gemv_argmax_rows", constants: [],
            maxTotalThreadsPerThreadgroup: 32 * Self.rowsPerThreadgroup)
        argmaxRowsReducePipeline = try context.pipeline(
            "bf16_gemv_argmax_rows_reduce", constants: [],
            maxTotalThreadsPerThreadgroup: 256)
        embeddingPipeline = try context.pipeline("bf16_embedding_lookup_half")
        floatEmbeddingPipeline = try context.pipeline("bf16_embedding_lookup_float")
        self.maxRows = maxRows
        self.maxVocabularyRows = maxVocabularyRows
        let rowGroups = (maxVocabularyRows + Self.rowsPerThreadgroup - 1)
            / Self.rowsPerThreadgroup
        guard let summaries = context.device.makeBuffer(
            length: maxRows * rowGroups * 2 * MemoryLayout<Float>.stride,
            options: .storageModePrivate) else {
            throw MetalError.noDevice
        }
        summaries.label = "bf16.gemv.argmaxRowsSummaries"
        argmaxRowsSummaries = summaries
    }

    func encodeHalf(
        commandBuffer: MTLCommandBuffer,
        weights: TensorView,
        input: MTLBuffer,
        inputOffset: Int = 0,
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
               inputOffset: inputOffset,
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
        inputOffset: Int = 0,
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
               inputOffset: inputOffset,
               output: output,
               outputOffset: outputOffset,
               bias: bias,
               rows: rows,
               columns: columns)
    }

    /// Computes one greedy token for each input row without materializing a
    /// vocabulary-sized output for every row. The kernel compares the same
    /// FP16-rounded projection values produced by `encodeHalf`, preserving
    /// the existing target argmax semantics while allowing all rows to run in
    /// one 2-D dispatch.
    func encodeHalfArgmaxRows(
        commandBuffer: MTLCommandBuffer,
        weights: TensorView,
        input: MTLBuffer,
        inputOffset: Int = 0,
        inputStrideElements: Int,
        output: MTLBuffer,
        outputOffset: Int = 0,
        bias: TensorView? = nil,
        rowCount: Int,
        rows: Int,
        columns: Int
    ) {
        precondition(rowCount > 0 && rowCount <= maxRows)
        precondition(rows > 0 && rows <= maxVocabularyRows)
        precondition(columns > 0)
        precondition(inputOffset >= 0)
        precondition(inputStrideElements >= columns)
        precondition(outputOffset >= 0)
        precondition(inputOffset % MemoryLayout<Float16>.alignment == 0)
        precondition(outputOffset % MemoryLayout<UInt32>.alignment == 0)

        let rowGroups = (rows + Self.rowsPerThreadgroup - 1)
            / Self.rowsPerThreadgroup
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        encoder.setComputePipelineState(argmaxRowsPipeline)
        encoder.setBuffer(weights.buffer, offset: Int(weights.offset), index: 0)
        encoder.setBuffer(input, offset: inputOffset, index: 1)
        encoder.setBuffer(bias?.buffer ?? weights.buffer,
                          offset: bias.map { Int($0.offset) }
                              ?? Int(weights.offset),
                          index: 2)
        encoder.setBuffer(argmaxRowsSummaries, offset: 0, index: 3)
        var rowValue = UInt32(rows)
        var columnValue = UInt32(columns)
        var hasBias: UInt32 = bias == nil ? 0 : 1
        var inputStrideValue = UInt32(inputStrideElements)
        var batchValue = UInt32(rowCount)
        encoder.setBytes(&rowValue, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&columnValue, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.setBytes(&hasBias, length: MemoryLayout<UInt32>.size, index: 6)
        encoder.setBytes(&inputStrideValue,
                         length: MemoryLayout<UInt32>.size, index: 7)
        encoder.setBytes(&batchValue, length: MemoryLayout<UInt32>.size, index: 8)
        encoder.dispatchThreadgroups(
            MTLSize(width: rowGroups, height: rowCount, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: 32 * Self.rowsPerThreadgroup, height: 1, depth: 1))
        encoder.endEncoding()

        guard let reducer = commandBuffer.makeComputeCommandEncoder() else {
            return
        }
        reducer.setComputePipelineState(argmaxRowsReducePipeline)
        reducer.setBuffer(argmaxRowsSummaries, offset: 0, index: 0)
        reducer.setBuffer(output, offset: outputOffset, index: 1)
        var rowGroupValue = UInt32(rowGroups)
        reducer.setBytes(&rowGroupValue,
                         length: MemoryLayout<UInt32>.size, index: 2)
        reducer.dispatchThreadgroups(
            MTLSize(width: rowCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        reducer.endEncoding()
    }

    func encodeEmbedding(
        commandBuffer: MTLCommandBuffer,
        table: TensorView,
        token: UInt32,
        output: MTLBuffer,
        outputOffset: Int = 0,
        hiddenSize: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(embeddingPipeline)
        encoder.setBuffer(table.buffer, offset: Int(table.offset), index: 0)
        encoder.setBuffer(output, offset: outputOffset, index: 1)
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

    func encodeFloatEmbedding(
        commandBuffer: MTLCommandBuffer,
        table: TensorView,
        token: UInt32,
        output: MTLBuffer,
        outputOffset: Int = 0,
        hiddenSize: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(floatEmbeddingPipeline)
        encoder.setBuffer(table.buffer, offset: Int(table.offset), index: 0)
        encoder.setBuffer(output, offset: outputOffset, index: 1)
        var tokenValue = token
        var hiddenValue = UInt32(hiddenSize)
        encoder.setBytes(&tokenValue, length: MemoryLayout<UInt32>.size, index: 2)
        encoder.setBytes(&hiddenValue, length: MemoryLayout<UInt32>.size, index: 3)
        let width = min(hiddenSize, floatEmbeddingPipeline.maxTotalThreadsPerThreadgroup)
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
        inputOffset: Int,
        output: MTLBuffer,
        outputOffset: Int,
        bias: TensorView?,
        rows: Int,
        columns: Int
    ) {
        precondition(rows > 0 && columns > 0)
        precondition(inputOffset % MemoryLayout<Float16>.alignment == 0)
        precondition(outputOffset % MemoryLayout<Float16>.alignment == 0)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weights.buffer, offset: Int(weights.offset), index: 0)
        encoder.setBuffer(input, offset: inputOffset, index: 1)
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
