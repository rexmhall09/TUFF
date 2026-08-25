import Metal

/// Small GPT-OSS-specific MoE operations shared by decode and prefill.
final class GPTOSSMoEPrimitives {
    private let cappedSwiGLU: MTLComputePipelineState
    private let routerTop4: MTLComputePipelineState

    init(context: MetalContext) throws {
        cappedSwiGLU = try context.pipeline("gptoss_capped_swiglu")
        routerTop4 = try context.pipeline("gptoss_router_top4")
    }

    func encodeCappedSwiGLU(commandBuffer: MTLCommandBuffer,
                            gate: MTLBuffer,
                            gateOffset: Int = 0,
                            linear: MTLBuffer,
                            linearOffset: Int = 0,
                            output: MTLBuffer,
                            outputOffset: Int = 0,
                            count: UInt32,
                            limit: Float = 7) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(cappedSwiGLU)
        encoder.setBuffer(gate, offset: gateOffset, index: 0)
        encoder.setBuffer(linear, offset: linearOffset, index: 1)
        encoder.setBuffer(output, offset: outputOffset, index: 2)
        var countValue = count
        var limitValue = limit
        encoder.setBytes(&countValue, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.setBytes(&limitValue, length: MemoryLayout<Float>.size, index: 4)
        let width = min(Int(count), cappedSwiGLU.maxTotalThreadsPerThreadgroup)
        encoder.dispatchThreads(
            MTLSize(width: Int(count), height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: max(1, width), height: 1, depth: 1))
        encoder.endEncoding()
    }

    func encodeRouterTop4(commandBuffer: MTLCommandBuffer,
                          logits: MTLBuffer,
                          logitsOffset: Int = 0,
                          outputIndices: MTLBuffer,
                          outputWeights: MTLBuffer,
                          numExperts: UInt32) {
        precondition((4...128).contains(numExperts),
                     "GPT-OSS supports 4...128 routed experts")
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(routerTop4)
        encoder.setBuffer(logits, offset: logitsOffset, index: 0)
        encoder.setBuffer(outputIndices, offset: 0, index: 1)
        encoder.setBuffer(outputWeights, offset: 0, index: 2)
        var expertCount = numExperts
        encoder.setBytes(&expertCount, length: MemoryLayout<UInt32>.size, index: 3)
        encoder.dispatchThreadgroups(
            MTLSize(width: 1, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
