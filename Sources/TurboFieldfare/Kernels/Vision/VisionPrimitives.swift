import Metal

package final class VisionPrimitives {
    private let normalize: MTLComputePipelineState
    private let addPosition: MTLComputePipelineState
    private let rmsnorm: MTLComputePipelineState
    private let postnormResidual: MTLComputePipelineState
    private let qkvNormRoPE: MTLComputePipelineState
    private let qkvNormRoPEPadded: MTLComputePipelineState
    private let geglu: MTLComputePipelineState
    private let pool: MTLComputePipelineState
    private let bfloatToHalf: MTLComputePipelineState
    private let qwenCopyPadded: MTLComputePipelineState
    private let qwenAddPosition: MTLComputePipelineState
    private let qwenLayerNorm: MTLComputePipelineState
    private let qwenQKVBiasRoPE: MTLComputePipelineState
    private let qwenResidualBias: MTLComputePipelineState
    private let qwenAddBias: MTLComputePipelineState
    private let qwenGELUTanhBias: MTLComputePipelineState
    private let qwenGELUErfBias: MTLComputePipelineState

    package init(context: MetalContext,
                 environment: [String: String] = ProcessInfo.processInfo.environment) throws {
        normalize = try context.pipeline("vision_normalize_patches")
        addPosition = try context.pipeline("vision_add_position")
        if environment["TURBO_FIELDFARE_VISION_SAFE_RMS"] == "1" {
            let library = try MetalContext.privateLibrary(
                device: context.device, module: "vision", mathMode: .safe)
            guard let rmsFunction = library.makeFunction(name: "vision_rmsnorm_rows"),
                  let residualFunction = library.makeFunction(
                    name: "vision_postnorm_residual") else {
                throw MetalError.missingFunction("vision safe RMSNorm")
            }
            rmsnorm = try context.device.makeComputePipelineState(function: rmsFunction)
            postnormResidual = try context.device.makeComputePipelineState(
                function: residualFunction)
        } else {
            rmsnorm = try context.pipeline("vision_rmsnorm_rows")
            postnormResidual = try context.pipeline("vision_postnorm_residual")
        }
        qkvNormRoPE = try context.pipeline("vision_qkv_norm_rope")
        qkvNormRoPEPadded = try context.pipeline("vision_qkv_norm_rope_padded")
        geglu = try context.pipeline("vision_geglu")
        pool = try context.pipeline("vision_pool_standardize_norm")
        bfloatToHalf = try context.pipeline("vision_bfloat_to_half")
        qwenCopyPadded = try context.pipeline("vision_qwen_copy_padded")
        qwenAddPosition = try context.pipeline("vision_qwen_add_interpolated_position")
        qwenLayerNorm = try context.pipeline("vision_qwen_layernorm_rows")
        qwenQKVBiasRoPE = try context.pipeline("vision_qwen_qkv_bias_rope")
        qwenResidualBias = try context.pipeline("vision_qwen_residual_bias")
        qwenAddBias = try context.pipeline("vision_qwen_add_bias")
        qwenGELUTanhBias = try context.pipeline("vision_qwen_gelu_tanh_bias")
        qwenGELUErfBias = try context.pipeline("vision_qwen_gelu_erf_bias")
    }

    package func encodeNormalize(commandBuffer: MTLCommandBuffer,
                                 input: MTLBuffer, output: MTLBuffer,
                                 rows: Int, paddedRows: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(normalize)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        var real = UInt32(rows), padded = UInt32(paddedRows)
        encoder.setBytes(&real, length: 4, index: 2)
        encoder.setBytes(&padded, length: 4, index: 3)
        dispatch1D(encoder, count: paddedRows * 768, pipeline: normalize)
    }

    package func encodeAddPosition(commandBuffer: MTLCommandBuffer,
                                   hidden: MTLBuffer,
                                   table: MTLBuffer, tableOffset: Int,
                                   positions: MTLBuffer, rows: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(addPosition)
        encoder.setBuffer(hidden, offset: 0, index: 0)
        encoder.setBuffer(table, offset: tableOffset, index: 1)
        encoder.setBuffer(positions, offset: 0, index: 2)
        var count = UInt32(rows)
        encoder.setBytes(&count, length: 4, index: 3)
        dispatch1D(encoder, count: rows * 1_152, pipeline: addPosition)
    }

    package func encodeRMSNorm(commandBuffer: MTLCommandBuffer,
                               input: MTLBuffer, output: MTLBuffer,
                               weightBuffer: MTLBuffer, weightOffset: Int,
                               rows: Int, width: Int = 1_152) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(rmsnorm)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weightBuffer, offset: weightOffset, index: 1)
        encoder.setBuffer(output, offset: 0, index: 2)
        var rowCount = UInt32(rows), dimension = UInt32(width)
        encoder.setBytes(&rowCount, length: 4, index: 3)
        encoder.setBytes(&dimension, length: 4, index: 4)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 288, height: 1, depth: 1))
        encoder.endEncoding()
    }

    package func encodePostnormResidual(commandBuffer: MTLCommandBuffer,
                                        residual: MTLBuffer, branch: MTLBuffer,
                                        weightBuffer: MTLBuffer, weightOffset: Int,
                                        output: MTLBuffer, rows: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(postnormResidual)
        encoder.setBuffer(residual, offset: 0, index: 0)
        encoder.setBuffer(branch, offset: 0, index: 1)
        encoder.setBuffer(weightBuffer, offset: weightOffset, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var rowCount = UInt32(rows), width: UInt32 = 1_152
        encoder.setBytes(&rowCount, length: 4, index: 4)
        encoder.setBytes(&width, length: 4, index: 5)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 288, height: 1, depth: 1))
        encoder.endEncoding()
    }

    package func encodeQKV(commandBuffer: MTLCommandBuffer,
                           q: MTLBuffer, k: MTLBuffer, v: MTLBuffer,
                           weights: MTLBuffer, qWeightOffset: Int, kWeightOffset: Int,
                           positions: MTLBuffer, rows: Int, heads: Int,
                           paddedRowStride: Int = 0) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        let padded = paddedRowStride != 0
        let pipeline = padded ? qkvNormRoPEPadded : qkvNormRoPE
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(q, offset: 0, index: 0)
        encoder.setBuffer(k, offset: 0, index: 1)
        encoder.setBuffer(v, offset: 0, index: 2)
        encoder.setBuffer(weights, offset: qWeightOffset, index: 3)
        encoder.setBuffer(weights, offset: kWeightOffset, index: 4)
        encoder.setBuffer(positions, offset: 0, index: 5)
        var rowCount = UInt32(rows), headCount = UInt32(heads)
        encoder.setBytes(&rowCount, length: 4, index: 6)
        encoder.setBytes(&headCount, length: 4, index: 7)
        if padded {
            var stride = UInt32(paddedRowStride)
            encoder.setBytes(&stride, length: 4, index: 8)
        }
        dispatch1D(encoder, count: rows * heads, pipeline: pipeline)
    }

    package func encodeGeGLU(commandBuffer: MTLCommandBuffer,
                             gate: MTLBuffer, up: MTLBuffer, count: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(geglu)
        encoder.setBuffer(gate, offset: 0, index: 0)
        encoder.setBuffer(up, offset: 0, index: 1)
        var value = UInt32(count)
        encoder.setBytes(&value, length: 4, index: 2)
        dispatch1D(encoder, count: count, pipeline: geglu)
    }

    package func encodePool(commandBuffer: MTLCommandBuffer,
                            input: MTLBuffer,
                            weights: MTLBuffer,
                            stdBiasOffset: Int, stdScaleOffset: Int,
                            output: MTLBuffer,
                            patchWidth: Int, patchHeight: Int) {
        let rows = (patchWidth / 3) * (patchHeight / 3)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pool)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: stdBiasOffset, index: 1)
        encoder.setBuffer(weights, offset: stdScaleOffset, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var width = UInt32(patchWidth), height = UInt32(patchHeight)
        encoder.setBytes(&width, length: 4, index: 4)
        encoder.setBytes(&height, length: 4, index: 5)
        encoder.dispatchThreadgroups(MTLSize(width: rows, height: 1, depth: 1),
                                     threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.endEncoding()
    }

    package func encodeBFloatToHalf(commandBuffer: MTLCommandBuffer,
                                    input: MTLBuffer, output: MTLBuffer,
                                    count: Int) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(bfloatToHalf)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        var value = UInt32(count)
        encoder.setBytes(&value, length: 4, index: 2)
        dispatch1D(encoder, count: count, pipeline: bfloatToHalf)
    }

    package func encodeQwenCopyPadded(
        commandBuffer: MTLCommandBuffer,
        input: MTLBuffer, output: MTLBuffer,
        rows: Int, paddedRows: Int, width: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(qwenCopyPadded)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        var real = UInt32(rows), padded = UInt32(paddedRows), dimension = UInt32(width)
        encoder.setBytes(&real, length: 4, index: 2)
        encoder.setBytes(&padded, length: 4, index: 3)
        encoder.setBytes(&dimension, length: 4, index: 4)
        dispatch1D(encoder, count: paddedRows * width, pipeline: qwenCopyPadded)
    }

    package func encodeQwenAddPosition(
        commandBuffer: MTLCommandBuffer,
        hidden: MTLBuffer,
        table: MTLBuffer, tableOffset: Int,
        positions: MTLBuffer, rows: Int,
        gridHeight: Int, gridWidth: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(qwenAddPosition)
        encoder.setBuffer(hidden, offset: 0, index: 0)
        encoder.setBuffer(table, offset: tableOffset, index: 1)
        encoder.setBuffer(positions, offset: 0, index: 2)
        var count = UInt32(rows), height = UInt32(gridHeight), width = UInt32(gridWidth)
        encoder.setBytes(&count, length: 4, index: 3)
        encoder.setBytes(&height, length: 4, index: 4)
        encoder.setBytes(&width, length: 4, index: 5)
        dispatch1D(encoder, count: rows * 1_152, pipeline: qwenAddPosition)
    }

    package func encodeQwenLayerNorm(
        commandBuffer: MTLCommandBuffer,
        input: MTLBuffer, output: MTLBuffer,
        weights: MTLBuffer, weightOffset: Int, biasOffset: Int,
        rows: Int, width: Int = 1_152
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(qwenLayerNorm)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: weightOffset, index: 1)
        encoder.setBuffer(weights, offset: biasOffset, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var rowCount = UInt32(rows), dimension = UInt32(width)
        encoder.setBytes(&rowCount, length: 4, index: 4)
        encoder.setBytes(&dimension, length: 4, index: 5)
        encoder.dispatchThreadgroups(
            MTLSize(width: rows, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 288, height: 1, depth: 1))
        encoder.endEncoding()
    }

    package func encodeQwenQKV(
        commandBuffer: MTLCommandBuffer,
        q: MTLBuffer, k: MTLBuffer, v: MTLBuffer,
        weights: MTLBuffer,
        qBiasOffset: Int, kBiasOffset: Int, vBiasOffset: Int,
        positions: MTLBuffer, rows: Int, heads: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(qwenQKVBiasRoPE)
        encoder.setBuffer(q, offset: 0, index: 0)
        encoder.setBuffer(k, offset: 0, index: 1)
        encoder.setBuffer(v, offset: 0, index: 2)
        encoder.setBuffer(weights, offset: qBiasOffset, index: 3)
        encoder.setBuffer(weights, offset: kBiasOffset, index: 4)
        encoder.setBuffer(weights, offset: vBiasOffset, index: 5)
        encoder.setBuffer(positions, offset: 0, index: 6)
        var rowCount = UInt32(rows), headCount = UInt32(heads)
        encoder.setBytes(&rowCount, length: 4, index: 7)
        encoder.setBytes(&headCount, length: 4, index: 8)
        dispatch1D(encoder, count: rows * heads, pipeline: qwenQKVBiasRoPE)
    }

    package func encodeQwenResidualBias(
        commandBuffer: MTLCommandBuffer,
        residual: MTLBuffer, branch: MTLBuffer,
        weights: MTLBuffer, biasOffset: Int,
        output: MTLBuffer, rows: Int, width: Int
    ) {
        encodeQwenElementwise(
            pipeline: qwenResidualBias, commandBuffer: commandBuffer,
            first: residual, second: branch, weights: weights,
            biasOffset: biasOffset, output: output,
            count: rows * width, width: width)
    }

    package func encodeQwenAddBias(
        commandBuffer: MTLCommandBuffer,
        values: MTLBuffer, weights: MTLBuffer, biasOffset: Int,
        count: Int, width: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(qwenAddBias)
        encoder.setBuffer(values, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: biasOffset, index: 1)
        var elementCount = UInt32(count), dimension = UInt32(width)
        encoder.setBytes(&elementCount, length: 4, index: 2)
        encoder.setBytes(&dimension, length: 4, index: 3)
        dispatch1D(encoder, count: count, pipeline: qwenAddBias)
    }

    package func encodeQwenGELUTanhBias(
        commandBuffer: MTLCommandBuffer,
        values: MTLBuffer, weights: MTLBuffer, biasOffset: Int,
        count: Int, width: Int
    ) {
        encodeQwenGELUBias(
            pipeline: qwenGELUTanhBias, commandBuffer: commandBuffer,
            values: values, weights: weights, biasOffset: biasOffset,
            count: count, width: width)
    }

    package func encodeQwenGELUErfBias(
        commandBuffer: MTLCommandBuffer,
        values: MTLBuffer, weights: MTLBuffer, biasOffset: Int,
        count: Int, width: Int
    ) {
        encodeQwenGELUBias(
            pipeline: qwenGELUErfBias, commandBuffer: commandBuffer,
            values: values, weights: weights, biasOffset: biasOffset,
            count: count, width: width)
    }

    private func encodeQwenGELUBias(
        pipeline: MTLComputePipelineState,
        commandBuffer: MTLCommandBuffer,
        values: MTLBuffer, weights: MTLBuffer, biasOffset: Int,
        count: Int, width: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(values, offset: 0, index: 0)
        encoder.setBuffer(weights, offset: biasOffset, index: 1)
        var elementCount = UInt32(count), dimension = UInt32(width)
        encoder.setBytes(&elementCount, length: 4, index: 2)
        encoder.setBytes(&dimension, length: 4, index: 3)
        dispatch1D(encoder, count: count, pipeline: pipeline)
    }

    private func encodeQwenElementwise(
        pipeline: MTLComputePipelineState,
        commandBuffer: MTLCommandBuffer,
        first: MTLBuffer, second: MTLBuffer,
        weights: MTLBuffer, biasOffset: Int,
        output: MTLBuffer, count: Int, width: Int
    ) {
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(first, offset: 0, index: 0)
        encoder.setBuffer(second, offset: 0, index: 1)
        encoder.setBuffer(weights, offset: biasOffset, index: 2)
        encoder.setBuffer(output, offset: 0, index: 3)
        var elementCount = UInt32(count), dimension = UInt32(width)
        encoder.setBytes(&elementCount, length: 4, index: 4)
        encoder.setBytes(&dimension, length: 4, index: 5)
        dispatch1D(encoder, count: count, pipeline: pipeline)
    }

    private func dispatch1D(_ encoder: MTLComputeCommandEncoder,
                            count: Int, pipeline: MTLComputePipelineState) {
        let width = min(pipeline.maxTotalThreadsPerThreadgroup, 256)
        encoder.dispatchThreads(MTLSize(width: count, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
        encoder.endEncoding()
    }

}
