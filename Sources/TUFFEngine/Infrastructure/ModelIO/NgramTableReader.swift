import Foundation
import Darwin

/// Row reader for Qwen4-Exp's n-gram per-layer-embedding table.
///
/// The table is 29.80 GiB in the pinned checkpoint — an order of magnitude
/// more than every other resident tensor combined — while one token reads
/// sixteen of its 320 million rows. It is neither resident nor slot-cached.
///
/// Rows are read with `pread`, avoiding a mapping of the entire table. A small
/// bounded cache retains dequantized rows: image-pad runs and repeated text
/// otherwise issue the same scattered reads and dequantization again and again.
/// The reader belongs to one serial runner.
///
/// Weight, scale and bias sit in three separate regions per shard, so a row is
/// three reads. Interleaving them at repack time would make it one, but that
/// needs a strided scatter the range copier cannot express.
final class NgramTableReader {

    struct Shard {
        let rowStart: UInt64
        let rowCount: UInt64
        let weightOffset: UInt64
        let scaleOffset: UInt64
        let biasOffset: UInt64
    }

    /// Quantized values per row.
    let rowWidth: Int
    /// Values sharing one scale and bias.
    let groupSize: Int
    let rowCount: UInt64
    private let shards: [Shard]
    private let descriptor: Int32
    private let path: String
    private let cacheCapacity = 4_096
    private var rowCache: [Int64: [Float16]] = [:]
    private var cacheOrder: [Int64] = []
    private var nextEviction = 0
    private(set) var physicalRowReads = 0

    /// Bytes of packed weight in one row: two 4-bit values per byte.
    private var packedBytesPerRow: Int { rowWidth / 2 }
    /// Scale and bias values in one row.
    private var groupsPerRow: Int { rowWidth / groupSize }

    init(fileURL: URL,
         rowWidth: Int,
         groupSize: Int,
         rowCount: UInt64,
         shards: [Shard]) throws {
        precondition(rowWidth > 0 && groupSize > 0)
        guard rowWidth % groupSize == 0 else {
            throw ModelError.indexCorrupt(detail:
                "n-gram row width \(rowWidth) is not a multiple of \(groupSize)")
        }
        guard rowWidth.isMultiple(of: 2) else {
            throw ModelError.indexCorrupt(detail:
                "n-gram row width \(rowWidth) does not pack into whole bytes")
        }
        guard !shards.isEmpty else {
            throw ModelError.indexCorrupt(detail: "n-gram table has no shards")
        }
        // Contiguous and ordered, or a global row cannot be resolved to one.
        var expected: UInt64 = 0
        for shard in shards {
            guard shard.rowStart == expected, shard.rowCount > 0 else {
                throw ModelError.indexCorrupt(detail:
                    "n-gram shards must be positive, contiguous and ordered")
            }
            expected += shard.rowCount
        }
        guard expected == rowCount else {
            throw ModelError.indexCorrupt(detail:
                "n-gram shards cover \(expected) rows, manifest says \(rowCount)")
        }

        let fd = open(fileURL.path, O_RDONLY | O_CLOEXEC)
        guard fd >= 0 else {
            throw ModelError.posixFailed(call: "open(\(fileURL.path))",
                                         errno: errno)
        }
        var info = stat()
        guard fstat(fd, &info) == 0, (info.st_mode & S_IFMT) == S_IFREG else {
            close(fd)
            throw ModelError.indexCorrupt(detail:
                "the n-gram table is not a regular file")
        }
        self.descriptor = fd
        self.path = fileURL.path
        self.rowWidth = rowWidth
        self.groupSize = groupSize
        self.rowCount = rowCount
        self.shards = shards
        // Random access by construction: the rows a token wants are scattered
        // by a hash, so read-ahead would only evict what the next token needs.
        _ = posix_fadvise_random(fd)
    }

    deinit { close(descriptor) }

    /// The shard holding `row`, found by binary search over the ordered spans.
    private func shardIndex(for row: UInt64) throws -> Int {
        var low = 0
        var high = shards.count - 1
        while low <= high {
            let mid = (low + high) / 2
            let shard = shards[mid]
            if row < shard.rowStart {
                high = mid - 1
            } else if row >= shard.rowStart + shard.rowCount {
                low = mid + 1
            } else {
                return mid
            }
        }
        throw ModelError.indexCorrupt(detail:
            "n-gram row \(row) is outside the table's \(rowCount) rows")
    }

    private func read(into buffer: UnsafeMutableRawPointer,
                      count: Int,
                      offset: UInt64) throws {
        var read = 0
        while read < count {
            let n = pread(descriptor,
                          buffer.advanced(by: read),
                          count - read,
                          off_t(offset) + off_t(read))
            if n < 0 {
                if errno == EINTR { continue }
                throw ModelError.posixFailed(call: "pread(\(path))", errno: errno)
            }
            guard n > 0 else {
                throw ModelError.indexCorrupt(detail:
                    "the n-gram table ended early at offset \(offset)")
            }
            read += n
        }
    }

    /// Dequantize the rows `rows` names, concatenated in order, into
    /// `rows.count * rowWidth` values.
    func embedding(rows: [Int64]) throws -> [Float16] {
        var out = [Float16](repeating: 0, count: rows.count * rowWidth)
        var packed = [UInt8](repeating: 0, count: packedBytesPerRow)
        var scaleBits = [UInt16](repeating: 0, count: groupsPerRow)
        var biasBits = [UInt16](repeating: 0, count: groupsPerRow)

        for (slot, row) in rows.enumerated() {
            guard row >= 0, UInt64(row) < rowCount else {
                throw ModelError.indexCorrupt(detail:
                    "n-gram row \(row) is outside the table")
            }
            if let cached = rowCache[row] {
                out.replaceSubrange((slot * rowWidth)..<((slot + 1) * rowWidth), with: cached)
                continue
            }
            let shard = shards[try shardIndex(for: UInt64(row))]
            let local = UInt64(row) - shard.rowStart

            try packed.withUnsafeMutableBytes {
                try read(into: $0.baseAddress!, count: packedBytesPerRow,
                         offset: shard.weightOffset
                            + local * UInt64(packedBytesPerRow))
            }
            let auxBytes = groupsPerRow * MemoryLayout<UInt16>.size
            try scaleBits.withUnsafeMutableBytes {
                try read(into: $0.baseAddress!, count: auxBytes,
                         offset: shard.scaleOffset + local * UInt64(auxBytes))
            }
            try biasBits.withUnsafeMutableBytes {
                try read(into: $0.baseAddress!, count: auxBytes,
                         offset: shard.biasOffset + local * UInt64(auxBytes))
            }

            let base = slot * rowWidth
            for i in 0..<rowWidth {
                let byte = packed[i >> 1]
                let nibble = (i & 1) == 0 ? (byte & 0x0F) : (byte >> 4)
                let group = i / groupSize
                let scale = Quantization.bf16ToFloat(scaleBits[group])
                let bias = Quantization.bf16ToFloat(biasBits[group])
                out[base + i] = Float16(Float(nibble) * scale + bias)
            }
            physicalRowReads += 1
            if cacheOrder.count == cacheCapacity {
                rowCache.removeValue(forKey: cacheOrder[nextEviction])
                cacheOrder[nextEviction] = row
                nextEviction = (nextEviction + 1) % cacheCapacity
            } else {
                cacheOrder.append(row)
            }
            rowCache[row] = Array(out[base..<(base + rowWidth)])
        }
        return out
    }
}

/// `posix_fadvise` is not exposed on Darwin; the equivalent is an `fcntl` that
/// turns read-ahead off, which is what random access wants.
// F_RDAHEAD takes the integer value itself; a pointer would enable read-ahead.
private func posix_fadvise_random(_ fd: Int32) -> Int32 {
    return fcntl(fd, F_RDAHEAD, 0)
}
