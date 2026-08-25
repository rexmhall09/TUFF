import Metal

/// GPT-OSS MXFP4 E2M1 matrix-vector multiplication.
///
/// `weights` are row-major packed nibbles and `scales` contain one UE8M0 byte
/// per 32 input columns. Inputs and outputs are FP16; accumulation is FP32.
final class MXFP4GEMV {
    private static let rowsPerThreadgroup = 8
    private let pipeline: MTLComputePipelineState

    init(context: MetalContext) throws {
        pipeline = try context.pipeline(
            "mxfp4_gemv_simd", constants: [],
            maxTotalThreadsPerThreadgroup: 32 * Self.rowsPerThreadgroup)
    }

    func encode(
        commandBuffer: MTLCommandBuffer,
        weights: MTLBuffer,
        weightsOffset: Int = 0,
        scales: MTLBuffer,
        scalesOffset: Int = 0,
        input: MTLBuffer,
        inputOffset: Int = 0,
        output: MTLBuffer,
        outputOffset: Int = 0,
        rows: UInt32,
        columns: UInt32
    ) {
        precondition(rows > 0 && columns > 0)
        precondition(columns % UInt32(Quantization.mxfp4GroupSize) == 0,
                     "MXFP4 columns must be a multiple of \(Quantization.mxfp4GroupSize)")
        precondition(inputOffset % MemoryLayout<Float16>.alignment == 0)
        precondition(outputOffset % MemoryLayout<Float16>.alignment == 0)
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else { return }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(weights, offset: weightsOffset, index: 0)
        encoder.setBuffer(scales, offset: scalesOffset, index: 1)
        encoder.setBuffer(input, offset: inputOffset, index: 2)
        encoder.setBuffer(output, offset: outputOffset, index: 3)
        var rowCount = rows
        var columnCount = columns
        encoder.setBytes(&rowCount, length: MemoryLayout<UInt32>.size, index: 4)
        encoder.setBytes(&columnCount, length: MemoryLayout<UInt32>.size, index: 5)
        encoder.dispatchThreadgroups(
            MTLSize(
                width: (Int(rows) + Self.rowsPerThreadgroup - 1)
                    / Self.rowsPerThreadgroup,
                height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(
                width: 32 * Self.rowsPerThreadgroup, height: 1, depth: 1))
        encoder.endEncoding()
    }
}
