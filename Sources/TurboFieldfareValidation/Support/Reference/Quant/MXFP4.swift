import Accelerate
import Foundation

/// Independent FP32 oracle for GPT-OSS MXFP4 matrices.
///
/// This intentionally duplicates the public E2M1 table instead of calling the
/// runtime decoder. Metal tests therefore compare two different code paths:
/// scalar nibble expansion plus Accelerate here, and in-flight unpacking plus
/// SIMD reduction in the production primitive.
public enum MXFP4Reference {
    public static let groupSize = 32

    private static let values: [Float] = [
        0, 0.5, 1, 1.5, 2, 3, 4, 6,
        -0.0, -0.5, -1, -1.5, -2, -3, -4, -6,
    ]

    public static func decode(
        packed: [UInt8],
        scales: [UInt8],
        rows: Int,
        columns: Int
    ) -> [Float] {
        precondition(rows > 0 && columns > 0)
        precondition(columns % groupSize == 0)
        precondition(packed.count == rows * columns / 2)
        precondition(scales.count == rows * columns / groupSize)

        let groupsPerRow = columns / groupSize
        var output = [Float](repeating: 0, count: rows * columns)
        for row in 0..<rows {
            for group in 0..<groupsPerRow {
                let exponent = Int(scales[row * groupsPerRow + group]) - 127
                let scale = Float(sign: .plus, exponent: exponent, significand: 1)
                let byteBase = row * columns / 2 + group * groupSize / 2
                let valueBase = row * columns + group * groupSize
                for byteIndex in 0..<(groupSize / 2) {
                    let byte = packed[byteBase + byteIndex]
                    output[valueBase + byteIndex * 2] =
                        values[Int(byte & 0x0F)] * scale
                    output[valueBase + byteIndex * 2 + 1] =
                        values[Int(byte >> 4)] * scale
                }
            }
        }
        return output
    }

    public static func gemv(
        packed: [UInt8],
        scales: [UInt8],
        x: [Float],
        rows: Int,
        columns: Int
    ) -> [Float] {
        precondition(x.count == columns)
        let weights = decode(
            packed: packed, scales: scales, rows: rows, columns: columns)
        var output = [Float](repeating: 0, count: rows)
        for row in 0..<rows {
            weights.withUnsafeBufferPointer { weightsBuffer in
                x.withUnsafeBufferPointer { inputBuffer in
                    vDSP_dotpr(
                        weightsBuffer.baseAddress!.advanced(by: row * columns), 1,
                        inputBuffer.baseAddress!, 1,
                        &output[row], vDSP_Length(columns))
                }
            }
        }
        return output
    }
}
