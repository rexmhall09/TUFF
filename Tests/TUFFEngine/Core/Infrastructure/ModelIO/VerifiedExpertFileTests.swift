import Darwin
import Foundation
import Testing
@testable import TUFFEngine

@Suite struct VerifiedExpertFileTests {
    @Test func checksTheOwnedDescriptorAfterPathReplacement() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-verified-expert-\(UUID()).bin")
        let original = Data("original expert weights".utf8)
        try original.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let fd = open(url.path, O_RDONLY)
        #expect(fd >= 0)
        let pending = VerifiedExpertFile(
            descriptor: fd, name: "expert", expectedSHA256: Sha256Verifier.hashData(original))
        // Atomic replacement must not switch verification to another inode.
        try Data("replacement weights".utf8).write(to: url, options: .atomic)
        try pending.waitUntilVerified()
        var bytes = [UInt8](repeating: 0, count: original.count)
        #expect(pread(fd, &bytes, bytes.count, 0) == original.count)
        #expect(Data(bytes) == original)
        withExtendedLifetime(pending) {}
    }

    @Test func asynchronousChecksumFailureIsNotLost() throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-bad-expert-\(UUID()).bin")
        try Data("bad weights".utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let fd = open(url.path, O_RDONLY)
        #expect(fd >= 0)
        let pending = VerifiedExpertFile(descriptor: fd, name: "expert",
                                         expectedSHA256: String(repeating: "0", count: 64))
        #expect(throws: ModelError.checksumMismatch(file: "expert")) {
            try pending.waitUntilVerified()
        }
    }
}
