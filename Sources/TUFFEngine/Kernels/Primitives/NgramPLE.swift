import Foundation
import Metal

/// Swift wrapper for Qwen4-Exp's n-gram per-layer embedding.
///
/// The projections around it — `key_proj`, `value_proj` — are ordinary INT4
/// GEMVs, and the three norms are the grouped centered RMSNorm. The lookup
/// itself is `NgramTableReader`, on the host. This owns the parts with no
/// counterpart elsewhere: the gate between the looked-up keys and the layer's
/// queries, and the dilated depthwise convolution.
///
/// See `ngram_ple.metal` for the arithmetic.
final class NgramPLE {

    private let gatePSO: MTLComputePipelineState
    private let convPSO: MTLComputePipelineState
    private let historyPushPSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.gatePSO = try context.pipeline("ple_gate_values_fp16")
        self.convPSO = try context.pipeline("ple_depthwise_conv_silu_fp16")
        self.historyPushPSO = try context.pipeline("ple_conv_history_push_fp16")
    }

    /// `out[s][d] = sigmoid(signedSqrt(keys[s]·queries[s] / sqrt(D))) * values[d]`
    func encodeGateValues(commandBuffer: MTLCommandBuffer,
                          keys: MTLBuffer, keysOffset: Int = 0,
                          queries: MTLBuffer, queriesOffset: Int = 0,
                          values: MTLBuffer, valuesOffset: Int = 0,
                          out: MTLBuffer, outOffset: Int = 0,
                          hiddenSize: Int,
                          streamCount: Int) {
        precondition(hiddenSize > 0 && streamCount > 0)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(gatePSO)
        enc.setBuffer(keys, offset: keysOffset, index: 0)
        enc.setBuffer(queries, offset: queriesOffset, index: 1)
        enc.setBuffer(values, offset: valuesOffset, index: 2)
        enc.setBuffer(out, offset: outOffset, index: 3)
        var d = UInt32(hiddenSize)
        var invSqrtD = 1.0 / Float(hiddenSize).squareRoot()
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&invSqrtD, length: MemoryLayout<Float>.size, index: 5)
        let width = min(gatePSO.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreadgroups(
            MTLSize(width: streamCount, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// One output row of the causal dilated depthwise convolution, through
    /// silu. `history` must hold `(taps - 1) * dilation + 1` rows, the last
    /// being the current one.
    func encodeDepthwiseConv(commandBuffer: MTLCommandBuffer,
                             history: MTLBuffer, historyOffset: Int = 0,
                             weight: MTLBuffer, weightOffset: Int = 0,
                             out: MTLBuffer, outOffset: Int = 0,
                             width: Int, taps: Int, dilation: Int) {
        precondition(width > 0 && taps > 0 && dilation > 0)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(convPSO)
        enc.setBuffer(history, offset: historyOffset, index: 0)
        enc.setBuffer(weight, offset: weightOffset, index: 1)
        enc.setBuffer(out, offset: outOffset, index: 2)
        var w = UInt32(width)
        var k = UInt32(taps)
        var dil = UInt32(dilation)
        enc.setBytes(&w, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&k, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&dil, length: MemoryLayout<UInt32>.size, index: 5)
        dispatch(enc, pipeline: convPSO, threads: width)
        enc.endEncoding()
    }

    /// Drop the oldest convolution row and append `row`.
    func encodeHistoryPush(commandBuffer: MTLCommandBuffer,
                           history: MTLBuffer, historyOffset: Int = 0,
                           row: MTLBuffer, rowOffset: Int = 0,
                           width: Int, length: Int) {
        precondition(width > 0 && length > 0)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(historyPushPSO)
        enc.setBuffer(history, offset: historyOffset, index: 0)
        enc.setBuffer(row, offset: rowOffset, index: 1)
        var w = UInt32(width)
        var len = UInt32(length)
        enc.setBytes(&w, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&len, length: MemoryLayout<UInt32>.size, index: 3)
        dispatch(enc, pipeline: historyPushPSO, threads: width)
        enc.endEncoding()
    }

    private func dispatch(_ encoder: MTLComputeCommandEncoder,
                          pipeline: MTLComputePipelineState,
                          threads: Int) {
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        let groups = (threads + width - 1) / width
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }
}
