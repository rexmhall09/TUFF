import Testing
import Foundation
@testable import TUFFEngine

/// Reads rows out of a synthetic n-gram table laid out exactly as the repack
/// planner writes one: shard after shard, and within each shard three
/// contiguous regions — packed weights, then scales, then biases.
///
/// The thing worth testing is addressing. A row id is global across 128
/// shards, and resolving it to the wrong shard, or to the right shard at the
/// wrong local offset, returns a perfectly plausible embedding belonging to a
/// different n-gram. Nothing downstream can notice.
@Suite struct NgramTableReaderTests {

    @Test func repeatedRowsReuseDequantizedValues() throws {
        let fixture = try Self.makeFixture()
        defer { try? FileManager.default.removeItem(at: fixture.url.deletingLastPathComponent()) }
        let first = try fixture.reader.embedding(rows: [3, 42, 3])
        #expect(fixture.reader.physicalRowReads == 2)
        let again = try fixture.reader.embedding(rows: [3, 42, 3])
        #expect(first == again)
        #expect(fixture.reader.physicalRowReads == 2)
    }

    private static let rowWidth = 160      // the checkpoint's own row width
    private static let groupSize = 32
    private static let groupsPerRow = rowWidth / groupSize

    /// A value that depends on both the row and the position within it, so a
    /// misaddressed read cannot coincidentally match.
    private static func nibble(row: Int, index: Int) -> UInt8 {
        UInt8((row * 7 + index * 3) % 16)
    }
    private static func scale(row: Int, group: Int) -> Float {
        0.01 + Float((row + group) % 5) * 0.002
    }
    private static func bias(row: Int, group: Int) -> Float {
        -0.05 + Float((row * 2 + group) % 7) * 0.01
    }

    private struct Fixture {
        let url: URL
        let reader: NgramTableReader
        let rowsPerShard: Int
        let shardCount: Int
        var totalRows: Int { rowsPerShard * shardCount }
    }

    private static func makeFixture(shardCount: Int = 4,
                                    rowsPerShard: Int = 37) throws -> Fixture {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ngram-table-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir,
                                                withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("ngram_ple.bin")

        let packedBytes = rowWidth / 2
        var bytes = [UInt8]()
        var shards: [NgramTableReader.Shard] = []
        for shard in 0..<shardCount {
            let weightOffset = UInt64(bytes.count)
            for local in 0..<rowsPerShard {
                let row = shard * rowsPerShard + local
                for byteIndex in 0..<packedBytes {
                    let low = nibble(row: row, index: byteIndex * 2)
                    let high = nibble(row: row, index: byteIndex * 2 + 1)
                    bytes.append(low | (high << 4))
                }
            }
            let scaleOffset = UInt64(bytes.count)
            for local in 0..<rowsPerShard {
                let row = shard * rowsPerShard + local
                for group in 0..<groupsPerRow {
                    let bits = Quantization.bf16Bits(scale(row: row, group: group))
                    bytes.append(UInt8(truncatingIfNeeded: bits))
                    bytes.append(UInt8(truncatingIfNeeded: bits >> 8))
                }
            }
            let biasOffset = UInt64(bytes.count)
            for local in 0..<rowsPerShard {
                let row = shard * rowsPerShard + local
                for group in 0..<groupsPerRow {
                    let bits = Quantization.bf16Bits(bias(row: row, group: group))
                    bytes.append(UInt8(truncatingIfNeeded: bits))
                    bytes.append(UInt8(truncatingIfNeeded: bits >> 8))
                }
            }
            shards.append(NgramTableReader.Shard(
                rowStart: UInt64(shard * rowsPerShard),
                rowCount: UInt64(rowsPerShard),
                weightOffset: weightOffset,
                scaleOffset: scaleOffset,
                biasOffset: biasOffset))
        }
        try Data(bytes).write(to: url)

        let reader = try NgramTableReader(
            fileURL: url, rowWidth: rowWidth, groupSize: groupSize,
            rowCount: UInt64(shardCount * rowsPerShard), shards: shards)
        return Fixture(url: url, reader: reader,
                       rowsPerShard: rowsPerShard, shardCount: shardCount)
    }

    private static func expected(row: Int) -> [Float] {
        (0..<rowWidth).map { i in
            let group = i / groupSize
            return Float(nibble(row: row, index: i)) * scale(row: row, group: group)
                + bias(row: row, group: group)
        }
    }

    @Test func readsRowsFromEveryShard() throws {
        let f = try Self.makeFixture()
        defer { try? FileManager.default.removeItem(
            at: f.url.deletingLastPathComponent()) }

        // One row from each shard, plus the first and last of the table.
        var rows = (0..<f.shardCount).map { Int64($0 * f.rowsPerShard + 5) }
        rows.append(0)
        rows.append(Int64(f.totalRows - 1))

        let values = try f.reader.embedding(rows: rows)
        #expect(values.count == rows.count * Self.rowWidth)
        for (slot, row) in rows.enumerated() {
            let actual = Array(values[(slot * Self.rowWidth)..<((slot + 1) * Self.rowWidth)])
                .map { Float($0) }
            let want = Self.expected(row: Int(row))
            let maxDiff = zip(actual, want).map { abs($0 - $1) }.max() ?? 0
            #expect(maxDiff < 1e-3, "row \(row) max diff \(maxDiff)")
        }
    }

    /// Shard boundaries are where an off-by-one in the local offset shows.
    @Test func resolvesRowsEitherSideOfAShardBoundary() throws {
        let f = try Self.makeFixture()
        defer { try? FileManager.default.removeItem(
            at: f.url.deletingLastPathComponent()) }

        for boundary in 1..<f.shardCount {
            let last = Int64(boundary * f.rowsPerShard - 1)
            let first = Int64(boundary * f.rowsPerShard)
            let values = try f.reader.embedding(rows: [last, first])
            for (slot, row) in [last, first].enumerated() {
                let actual = Array(
                    values[(slot * Self.rowWidth)..<((slot + 1) * Self.rowWidth)]
                ).map { Float($0) }
                let want = Self.expected(row: Int(row))
                let maxDiff = zip(actual, want).map { abs($0 - $1) }.max() ?? 0
                #expect(maxDiff < 1e-3, "boundary row \(row) diff \(maxDiff)")
            }
        }
    }

    /// Two different rows must not come back the same — otherwise the tests
    /// above would pass on a reader that ignored its row argument.
    @Test func differentRowsGiveDifferentEmbeddings() throws {
        let f = try Self.makeFixture()
        defer { try? FileManager.default.removeItem(
            at: f.url.deletingLastPathComponent()) }
        let values = try f.reader.embedding(rows: [3, 4])
        let first = Array(values[0..<Self.rowWidth])
        let second = Array(values[Self.rowWidth..<(2 * Self.rowWidth)])
        #expect(first != second)
    }

    @Test func aRowOutsideTheTableIsRefused() throws {
        let f = try Self.makeFixture()
        defer { try? FileManager.default.removeItem(
            at: f.url.deletingLastPathComponent()) }
        #expect(throws: (any Error).self) {
            _ = try f.reader.embedding(rows: [Int64(f.totalRows)])
        }
        #expect(throws: (any Error).self) {
            _ = try f.reader.embedding(rows: [-1])
        }
    }

    /// Shards that skip or overlap rows make a global id ambiguous, so the
    /// reader has to refuse them at construction rather than resolve one
    /// arbitrarily.
    @Test func nonContiguousShardsAreRefused() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ngram-bad-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir,
                                                withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("ngram_ple.bin")
        try Data(count: 1024).write(to: url)

        #expect(throws: (any Error).self) {
            _ = try NgramTableReader(
                fileURL: url, rowWidth: Self.rowWidth, groupSize: Self.groupSize,
                rowCount: 20,
                shards: [
                    .init(rowStart: 0, rowCount: 10, weightOffset: 0,
                          scaleOffset: 0, biasOffset: 0),
                    // Starts at 11, leaving row 10 unreachable.
                    .init(rowStart: 11, rowCount: 9, weightOffset: 0,
                          scaleOffset: 0, biasOffset: 0),
                ])
        }
    }
}
