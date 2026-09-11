import Foundation
import Metal

/// Swift wrapper for Qwen4-Exp's multi-stream residual arithmetic.
///
/// The three projections a hyper-connection performs — down, up and the
/// per-stream injection gate — are ordinary INT4 GEMVs and go through the
/// existing dequant path. This owns the pieces between them: the low-rank
/// activation, the weighted collapse of the streams into the width a block
/// expects, and the gated accumulation back into all of them.
///
/// See `hyper_connection.metal` for the arithmetic and where each step sits.
final class HyperConnection {

    private let lowRankSiluPSO: MTLComputePipelineState
    private let combinePSO: MTLComputePipelineState
    private let injectPSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.lowRankSiluPSO = try context.pipeline("hc_lowrank_silu_fp16")
        self.combinePSO = try context.pipeline("hc_combine_fp16")
        self.injectPSO = try context.pipeline("hc_inject_fp16")
    }

    /// `out[i] = silu(x[i] / streamCount)` over the low-rank projection.
    func encodeLowRankSilu(commandBuffer: MTLCommandBuffer,
                           x: MTLBuffer, xOffset: Int = 0,
                           out: MTLBuffer, outOffset: Int = 0,
                           count: Int,
                           streamCount: Int) {
        precondition(count > 0 && streamCount > 0)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(lowRankSiluPSO)
        enc.setBuffer(x, offset: xOffset, index: 0)
        enc.setBuffer(out, offset: outOffset, index: 1)
        var countVar = UInt32(count)
        var scale = 1.0 / Float(streamCount)
        enc.setBytes(&countVar, length: MemoryLayout<UInt32>.size, index: 2)
        enc.setBytes(&scale, length: MemoryLayout<Float>.size, index: 3)
        dispatch(enc, pipeline: lowRankSiluPSO, threads: count)
        enc.endEncoding()
    }

    /// `mixed[i] = mean over streams of sigmoid(up[g][i]) * normed[g][i]`.
    ///
    /// `up` is the raw up-projection: the sigmoid is applied here.
    func encodeCombine(commandBuffer: MTLCommandBuffer,
                       up: MTLBuffer, upOffset: Int = 0,
                       normed: MTLBuffer, normedOffset: Int = 0,
                       mixed: MTLBuffer, mixedOffset: Int = 0,
                       hiddenSize: Int,
                       streamCount: Int) {
        precondition(hiddenSize > 0 && streamCount > 0)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(combinePSO)
        enc.setBuffer(up, offset: upOffset, index: 0)
        enc.setBuffer(normed, offset: normedOffset, index: 1)
        enc.setBuffer(mixed, offset: mixedOffset, index: 2)
        var d = UInt32(hiddenSize)
        var groups = UInt32(streamCount)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&groups, length: MemoryLayout<UInt32>.size, index: 4)
        dispatch(enc, pipeline: combinePSO, threads: hiddenSize)
        enc.endEncoding()
    }

    /// `hyper[g][i] += branch[i] * 2 * sigmoid(injectionRaw[g] / streamCount)`,
    /// accumulating a block's output into every residual stream.
    ///
    /// `injectionRaw` is the raw injection projection, `streamCount` values
    /// wide; the gate is applied here.
    func encodeInject(commandBuffer: MTLCommandBuffer,
                      hyper: MTLBuffer, hyperOffset: Int = 0,
                      branch: MTLBuffer, branchOffset: Int = 0,
                      injectionRaw: MTLBuffer, injectionOffset: Int = 0,
                      hiddenSize: Int,
                      streamCount: Int) {
        precondition(hiddenSize > 0 && streamCount > 0)
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(injectPSO)
        enc.setBuffer(hyper, offset: hyperOffset, index: 0)
        enc.setBuffer(branch, offset: branchOffset, index: 1)
        enc.setBuffer(injectionRaw, offset: injectionOffset, index: 2)
        var d = UInt32(hiddenSize)
        var groups = UInt32(streamCount)
        var scale = 1.0 / Float(streamCount)
        enc.setBytes(&d, length: MemoryLayout<UInt32>.size, index: 3)
        enc.setBytes(&groups, length: MemoryLayout<UInt32>.size, index: 4)
        enc.setBytes(&scale, length: MemoryLayout<Float>.size, index: 5)
        dispatch(enc, pipeline: injectPSO, threads: hiddenSize * streamCount)
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
