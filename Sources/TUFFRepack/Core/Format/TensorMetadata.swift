import Foundation

/// One physical tensor in a safetensors shard. Coordinates are absolute file
/// offsets so the writer can map and copy a single tensor without re-parsing
/// the shard header.
struct SourceTensor: Sendable, Hashable {
    enum Dtype: UInt8, Sendable, Hashable {
        case u32  = 0
        case bf16 = 1
        case fp16 = 2
        case fp32 = 3
        case u8   = 4
        /// The n-gram PLE carries three small integer tensors — the rolling
        /// hash multipliers and each head's offset and vocabulary size — that
        /// are read as values rather than decoded as weights.
        case i64  = 5

        var elementBytes: Int {
            switch self {
            case .u32: 4
            case .bf16: 2
            case .fp16: 2
            case .fp32: 4
            case .u8: 1
            case .i64: 8
            }
        }
    }

    let name: String
    let shardPath: String
    let dtype: Dtype
    let shape: [UInt64]
    let absoluteOffset: UInt64
    let sizeBytes: UInt64
}

/// Bit-width override resolved from `config.json -> quantization`.
/// Affine group sizes the runtime can decode. Mirrored from the engine's
/// `Quantization.supportedGroupSizes`; the repack target deliberately has no
/// dependency on the runtime module, so the pair is kept in step by the
/// format-compatibility tests rather than by a shared symbol.
enum RepackQuantGroup {
    static let supported: Set<Int> = [32, 64]
}

struct QuantSpec: Sendable, Hashable {
    let bits: Int
    /// Values sharing one scale and bias. Follows the bit width rather than
    /// the model: Qwen3.8 Flash Next groups its 4-bit tensors at 32 and its
    /// 8-bit router and shared-expert gate at 64, in one checkpoint.
    let groupSize: Int

    init(bits: Int, groupSize: Int) {
        self.bits = bits
        self.groupSize = groupSize
    }
}
