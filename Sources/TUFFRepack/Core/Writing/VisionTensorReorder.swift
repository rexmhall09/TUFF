import Foundation

/// Reorders a Transformers Conv3D kernel to the channels-last layout MLX
/// uses, in place in a written pack: `[out, channels, t, h, w]` becomes
/// `[out, t, h, w, channels]`. Only Qwen's patch embedding needs it, about
/// 3.5 MB, so it is read whole.
enum VisionTensorReorder {
    static func channelsLastInPlace(path: String, offset: UInt64, sourceShape: [UInt64]) throws {
        guard sourceShape.count == 5 else {
            throw RepackError.configurationInvalid(detail: "patch embedding must have five dimensions")
        }
        let (out, channels, t, h, w) = (Int(sourceShape[0]), Int(sourceShape[1]), Int(sourceShape[2]),
                                        Int(sourceShape[3]), Int(sourceShape[4]))
        let count = out * channels * t * h * w
        let handle = try FileHandle(forUpdating: URL(fileURLWithPath: path))
        defer { try? handle.close() }
        try handle.seek(toOffset: offset)
        guard let data = try handle.read(upToCount: count * 2), data.count == count * 2 else {
            throw RepackError.configurationInvalid(detail: "patch embedding is truncated")
        }
        let source = data.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
        var reordered = [UInt16](repeating: 0, count: count)
        let spatial = t * h * w
        for o in 0..<out {
            for c in 0..<channels {
                for s in 0..<spatial {
                    reordered[(o * spatial + s) * channels + c] = source[(o * channels + c) * spatial + s]
                }
            }
        }
        try handle.seek(toOffset: offset)
        try reordered.withUnsafeBytes { try handle.write(contentsOf: Data($0)) }
        try handle.synchronize()
    }
}
