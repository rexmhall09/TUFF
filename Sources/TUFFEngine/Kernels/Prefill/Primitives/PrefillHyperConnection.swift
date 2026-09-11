import Foundation
import Metal

/// Chunk-scale hyper-connection kernels: the same arithmetic
/// `HyperConnection` performs for one token, over a whole prefill chunk.
///
/// The residual and its norm carry every stream stacked per token, so these
/// address `[token][stream][d]` where the decode path addresses `[stream][d]`.
/// See `prefill.metal` for the kernels themselves.
final class PrefillHyperConnection {

    private let groupedCenteredNormPSO: MTLComputePipelineState
    private let lowRankSiluPSO: MTLComputePipelineState
    private let combinePSO: MTLComputePipelineState
    private let injectPSO: MTLComputePipelineState
    private let tileEmbeddingPSO: MTLComputePipelineState

    /// The combine kernel splits each token's hidden dimension across this
    /// many threadgroups; it must match `kPrefillHCCombineGroups`.
    private static let combineGroups = 4

    init(context: MetalContext) throws {
        self.groupedCenteredNormPSO = try context.pipeline(
            "prefill_rmsnorm_bf16w_grouped_centered_block")
        self.lowRankSiluPSO = try context.pipeline("prefill_hc_lowrank_silu_block")
        self.combinePSO = try context.pipeline("prefill_hc_combine_block")
        self.injectPSO = try context.pipeline("prefill_hc_inject_block")
        self.tileEmbeddingPSO = try context.pipeline("prefill_hc_tile_embedding_block")
    }

    /// Grouped RMSNorm with checkpoint weights centered at zero.
    func encodeGroupedCenteredNorm(commandBuffer: MTLCommandBuffer,
                                   x: MTLBuffer,
                                   weight: MTLBuffer, weightOffset: Int,
                                   out: MTLBuffer,
                                   tokens: Int, hiddenSize: Int,
                                   streamCount: Int, eps: Float) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(groupedCenteredNormPSO)
        enc.setBuffer(x, offset: 0, index: 0)
        enc.setBuffer(weight, offset: weightOffset, index: 1)
        enc.setBuffer(out, offset: 0, index: 2)
        var t = UInt32(tokens), d = UInt32(hiddenSize), s = UInt32(streamCount)
        var e = eps
        enc.setBytes(&t, length: 4, index: 3)
        enc.setBytes(&d, length: 4, index: 4)
        enc.setBytes(&s, length: 4, index: 5)
        enc.setBytes(&e, length: MemoryLayout<Float>.size, index: 6)
        let width = min(groupedCenteredNormPSO.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreadgroups(
            MTLSize(width: streamCount, height: tokens, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// `x = silu(x / streamCount)` in place.
    func encodeLowRankSilu(commandBuffer: MTLCommandBuffer,
                           x: MTLBuffer,
                           tokens: Int, lowRank: Int, streamCount: Int) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(lowRankSiluPSO)
        enc.setBuffer(x, offset: 0, index: 0)
        var total = UInt32(tokens * lowRank)
        var scale = 1.0 / Float(streamCount)
        enc.setBytes(&total, length: 4, index: 1)
        enc.setBytes(&scale, length: MemoryLayout<Float>.size, index: 2)
        dispatch(enc, pipeline: lowRankSiluPSO, threads: tokens * lowRank)
        enc.endEncoding()
    }

    /// `mixed = mean_s sigmoid(up[s]) * normed[s]`.
    func encodeCombine(commandBuffer: MTLCommandBuffer,
                       up: MTLBuffer, normed: MTLBuffer, mixed: MTLBuffer,
                       tokens: Int, hiddenSize: Int, streamCount: Int) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(combinePSO)
        enc.setBuffer(up, offset: 0, index: 0)
        enc.setBuffer(normed, offset: 0, index: 1)
        enc.setBuffer(mixed, offset: 0, index: 2)
        var t = UInt32(tokens), d = UInt32(hiddenSize), s = UInt32(streamCount)
        enc.setBytes(&t, length: 4, index: 3)
        enc.setBytes(&d, length: 4, index: 4)
        enc.setBytes(&s, length: 4, index: 5)
        let width = min(combinePSO.maxTotalThreadsPerThreadgroup, 256)
        enc.dispatchThreadgroups(
            MTLSize(width: Self.combineGroups, height: tokens, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        enc.endEncoding()
    }

    /// Accumulate a block's output into every stream through its gate.
    func encodeInject(commandBuffer: MTLCommandBuffer,
                      hidden: MTLBuffer, branch: MTLBuffer, injectionRaw: MTLBuffer,
                      tokens: Int, hiddenSize: Int, streamCount: Int) {
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(injectPSO)
        enc.setBuffer(hidden, offset: 0, index: 0)
        enc.setBuffer(branch, offset: 0, index: 1)
        enc.setBuffer(injectionRaw, offset: 0, index: 2)
        var t = UInt32(tokens), d = UInt32(hiddenSize), s = UInt32(streamCount)
        var scale = 1.0 / Float(streamCount)
        enc.setBytes(&t, length: 4, index: 3)
        enc.setBytes(&d, length: 4, index: 4)
        enc.setBytes(&s, length: 4, index: 5)
        enc.setBytes(&scale, length: MemoryLayout<Float>.size, index: 6)
        dispatch(enc, pipeline: injectPSO, threads: tokens * streamCount * hiddenSize)
        enc.endEncoding()
    }

    /// Copy stream 0 — the embedding — across the remaining streams.
    func encodeTileEmbedding(commandBuffer: MTLCommandBuffer,
                             hidden: MTLBuffer,
                             tokens: Int, hiddenSize: Int, streamCount: Int) {
        guard streamCount > 1 else { return }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(tileEmbeddingPSO)
        enc.setBuffer(hidden, offset: 0, index: 0)
        var t = UInt32(tokens), d = UInt32(hiddenSize), s = UInt32(streamCount)
        enc.setBytes(&t, length: 4, index: 1)
        enc.setBytes(&d, length: 4, index: 2)
        enc.setBytes(&s, length: 4, index: 3)
        dispatch(enc, pipeline: tileEmbeddingPSO,
                 threads: tokens * (streamCount - 1) * hiddenSize)
        enc.endEncoding()
    }

    private func dispatch(_ encoder: MTLComputeCommandEncoder,
                          pipeline: MTLComputePipelineState,
                          threads: Int) {
        guard threads > 0 else { return }
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        let groups = (threads + width - 1) / width
        encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }
}
