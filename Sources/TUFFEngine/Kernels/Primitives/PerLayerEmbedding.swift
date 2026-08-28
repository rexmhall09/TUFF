import Foundation
import Metal

/// Gemma 4 Per-Layer Embedding (PLE) preparation. Both packed inputs are
/// token-major `[rows, numLayers * pleWidth]`; `out` receives the selected
/// contiguous `[rows, pleWidth]` slice.
final class PerLayerEmbedding {
    private let preparePSO: MTLComputePipelineState

    init(context: MetalContext) throws {
        self.preparePSO = try context.pipeline("ple_prepare_layer_fp16")
    }

    func encodePrepareLayer(commandBuffer: MTLCommandBuffer,
                            identity: MTLBuffer,
                            context: MTLBuffer,
                            normWeight: MTLBuffer,
                            normWeightOffset: Int = 0,
                            out: MTLBuffer,
                            rows: Int,
                            packedWidth: Int,
                            pleWidth: Int,
                            layer: Int,
                            contextScale: Float,
                            combinedScale: Float,
                            eps: Float) {
        precondition(rows > 0 && pleWidth > 0 && packedWidth >= pleWidth)
        precondition(layer >= 0 && (layer + 1) * pleWidth <= packedWidth)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(preparePSO)
        encoder.setBuffer(identity, offset: 0, index: 0)
        encoder.setBuffer(context, offset: 0, index: 1)
        encoder.setBuffer(normWeight, offset: normWeightOffset, index: 2)
        encoder.setBuffer(out, offset: 0, index: 3)
        var rowCount = UInt32(rows)
        var packed = UInt32(packedWidth)
        var width = UInt32(pleWidth)
        var layerIndex = UInt32(layer)
        var contextFactor = contextScale
        var combinedFactor = combinedScale
        var epsilon = eps
        encoder.setBytes(&rowCount, length: 4, index: 4)
        encoder.setBytes(&packed, length: 4, index: 5)
        encoder.setBytes(&width, length: 4, index: 6)
        encoder.setBytes(&layerIndex, length: 4, index: 7)
        encoder.setBytes(&contextFactor, length: 4, index: 8)
        encoder.setBytes(&combinedFactor, length: 4, index: 9)
        encoder.setBytes(&epsilon, length: 4, index: 10)
        let widthThreads = min(preparePSO.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: widthThreads, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
