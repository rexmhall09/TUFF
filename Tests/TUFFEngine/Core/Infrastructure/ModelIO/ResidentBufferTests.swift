import Testing
import Foundation
import Metal
import Darwin
@testable import TUFFEngine

@Suite struct ResidentBufferTests {

    private static func entry(
        _ name: String,
        weights: (UInt64, UInt64),
        scales: (UInt64, UInt64) = (0, 0),
        biases: (UInt64, UInt64) = (0, 0)
    ) -> ResidentIndexEntry {
        ResidentIndexEntry(
            name: name, dtype: 0,
            fileOffset: weights.0, sizeBytes: weights.1,
            shape: (1, 1, 1, 1),
            scaleOffset: scales.0, scaleSize: scales.1,
            biasOffset: biases.0, biasSize: biases.1)
    }

    @Test func regionPlannerSplitsOnlyBetweenCompleteTensors() throws {
        let plans = try ResidentBufferSet.planRegions(
            entries: [
                Self.entry("a", weights: (100, 40),
                           scales: (140, 8), biases: (148, 8)),
                Self.entry("b", weights: (160, 50)),
                Self.entry("c", weights: (214, 20),
                           scales: (234, 4), biases: (238, 4)),
            ],
            payloadOffset: 100,
            payloadSize: 300,
            maximumLength: 90)

        #expect(plans == [
            .init(fileOffset: 100, size: 56, tensorNames: ["a"]),
            .init(fileOffset: 160, size: 82, tensorNames: ["b", "c"]),
        ])
    }

    @Test func regionPlannerRejectsTensorWhoseRelatedFieldsExceedOneBuffer() {
        #expect(throws: ModelError.self) {
            try ResidentBufferSet.planRegions(
                entries: [
                    Self.entry("split-tensor", weights: (100, 20),
                               scales: (190, 10), biases: (200, 10)),
                ],
                payloadOffset: 100,
                payloadSize: 200,
                maximumLength: 64)
        }
    }

    @Test func wrapsResidentRegionAndReadsBytes() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-resident-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }

        // 24-byte fake header (zeros) followed by 16-byte fake index region
        // followed by 1 KB of resident payload with a recognizable pattern.
        let preamble = Data(repeating: 0, count: 24 + 16)
        var payload = Data(count: 1024)
        for i in 0..<payload.count { payload[i] = UInt8(i & 0xFF) }
        try (preamble + payload).write(to: url)

        let resident = try ResidentBuffer(
            fileURL: url,
            fileOffset: UInt64(preamble.count),
            residentSize: UInt64(payload.count),
            device: device)
        #expect(resident.buffer.length == payload.count)
        for i in 0..<payload.count {
            let got = resident.buffer.contents().load(
                fromByteOffset: i, as: UInt8.self)
            #expect(got == UInt8(i & 0xFF), "byte \(i)")
        }
    }

    @Test func unalignedFileOffsetStillExposesPayloadAtOffsetZero() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-resident-unaligned-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: url) }

        // Offset that is not a multiple of getpagesize() — say 137.
        let preLen = 137
        let preamble = Data(repeating: 0x55, count: preLen)
        var payload = Data(count: 64)
        for i in 0..<payload.count { payload[i] = UInt8(0x80 | (i & 0x7F)) }
        try (preamble + payload).write(to: url)

        let resident = try ResidentBuffer(
            fileURL: url, fileOffset: UInt64(preLen),
            residentSize: UInt64(payload.count), device: device)
        for i in 0..<payload.count {
            let got = resident.buffer.contents().load(
                fromByteOffset: i, as: UInt8.self)
            #expect(got == UInt8(0x80 | (i & 0x7F)), "byte \(i)")
        }
    }

    @Test func followsCallerSelectedSymlinkToRegularFile() throws {
        let device = try #require(MTLCreateSystemDefaultDevice())
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-resident-symlink-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let target = directory.appendingPathComponent("target.bin")
        let alias = directory.appendingPathComponent("alias.bin")
        try Data(repeating: 0x5A, count: 64).write(to: target)
        #expect(symlink(target.path, alias.path) == 0)
        let resident = try ResidentBuffer(
            fileURL: alias, fileOffset: 0, residentSize: 64, device: device)
        #expect(resident.buffer.contents().load(as: UInt8.self) == 0x5A)
    }
}
