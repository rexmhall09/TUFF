import Metal

/// Bounded GPU argmax used by block verification. It returns one token ID and
/// never copies a vocabulary-sized vector to the CPU.
final class FP16Argmax {
    private let pipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        pipeline = try context.pipeline("argmax_fp16")
    }

    func encode(commandBuffer: MTLCommandBuffer,
                values: MTLBuffer,
                valuesOffset: Int = 0,
                count: Int,
                output: MTLBuffer,
                outputOffset: Int = 0) {
        precondition(count > 0)
        precondition(valuesOffset >= 0)
        precondition(outputOffset >= 0)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(values, offset: valuesOffset, index: 0)
        encoder.setBuffer(output, offset: outputOffset, index: 1)
        var countValue = UInt32(count)
        encoder.setBytes(&countValue, length: MemoryLayout<UInt32>.size, index: 2)
        let threads = min(256, pipeline.maxTotalThreadsPerThreadgroup)
        let threadgroup = MTLSize(width: max(1, threads), height: 1, depth: 1)
        encoder.dispatchThreads(threadgroup, threadsPerThreadgroup: threadgroup)
        encoder.endEncoding()
    }
}
