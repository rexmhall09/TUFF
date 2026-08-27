import Foundation
import Testing
@testable import TurboFieldfareRepackCore

@Suite("Blocked install paths")
struct BlockedInstallPathTests {
    private func makeDirectory() throws -> URL {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("tuff-blocked-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    @Test("a symlink where the model belongs is reported and cleared")
    func clearsSymlink() throws {
        let root = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let target = root.appendingPathComponent("real-model", isDirectory: true)
        try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
        try Data("weights".utf8).write(to: target.appendingPathComponent("model_weights.bin"))
        let destination = root.appendingPathComponent("qwen36.gturbo")
        try FileManager.default.createSymbolicLink(at: destination, withDestinationURL: target)

        let paths = try RemoteInstallPaths(outputDirectory: destination.path)
        #expect(throws: RepackError.self) { try paths.validateEntryTypes() }

        let blocked = try paths.blockingEntries()
        #expect(blocked.count == 1)
        #expect(blocked.first?.kind == .symlink)

        let removed = try paths.removeBlockingEntries()
        #expect(removed.count == 1)
        try paths.validateEntryTypes()

        // The symlink is gone; whatever it pointed at is untouched.
        #expect(!FileManager.default.fileExists(atPath: destination.path))
        #expect(FileManager.default.fileExists(
            atPath: target.appendingPathComponent("model_weights.bin").path))
    }

    @Test("a regular file standing in for the model directory is cleared")
    func clearsRegularFile() throws {
        let root = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("gemma4.gturbo")
        try Data("not a directory".utf8).write(to: destination)

        let paths = try RemoteInstallPaths(outputDirectory: destination.path)
        #expect(try paths.blockingEntries().count == 1)
        try paths.removeBlockingEntries()
        try paths.validateEntryTypes()
        #expect(!FileManager.default.fileExists(atPath: destination.path))
    }

    @Test("a clean destination reports nothing to clear")
    func cleanDestination() throws {
        let root = try makeDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let destination = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        let paths = try RemoteInstallPaths(outputDirectory: destination.path)
        #expect(try paths.blockingEntries().isEmpty)
        #expect(try paths.removeBlockingEntries().isEmpty)
        #expect(FileManager.default.fileExists(atPath: destination.path))
    }
}
