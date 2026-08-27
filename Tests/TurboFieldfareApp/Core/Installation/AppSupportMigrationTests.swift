import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite("Application Support migration")
struct AppSupportMigrationTests {
    private func makeSupportDirectory() throws -> URL {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("tuff-migration-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }

    @Test("model directories move out of the legacy folder")
    func movesModels() throws {
        let support = try makeSupportDirectory()
        defer { try? FileManager.default.removeItem(at: support) }
        let legacy = support.appendingPathComponent("TurboFieldfare", isDirectory: true)
        let model = legacy.appendingPathComponent("qwen36.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try Data("weights".utf8).write(
            to: model.appendingPathComponent("model_weights.bin"))

        let outcome = AppSupportMigration.migrateModels(applicationSupport: support)

        #expect(outcome.movedEntries == ["qwen36.gturbo"])
        let moved = support.appendingPathComponent("TUFF/Models/qwen36.gturbo/model_weights.bin")
        #expect(FileManager.default.fileExists(atPath: moved.path))
        #expect(!FileManager.default.fileExists(atPath: legacy.path))
    }

    @Test("an existing destination is never overwritten")
    func keepsExistingDestination() throws {
        let support = try makeSupportDirectory()
        defer { try? FileManager.default.removeItem(at: support) }
        let legacy = support.appendingPathComponent("TurboFieldfare/gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: legacy, withIntermediateDirectories: true)
        try Data("old".utf8).write(to: legacy.appendingPathComponent("marker"))
        let destination = support.appendingPathComponent("TUFF/Models/gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        try Data("current".utf8).write(to: destination.appendingPathComponent("marker"))

        let outcome = AppSupportMigration.migrateModels(applicationSupport: support)

        #expect(outcome.movedEntries.isEmpty)
        #expect(outcome.skippedEntries == ["gemma4.gturbo"])
        let kept = try String(
            contentsOf: destination.appendingPathComponent("marker"), encoding: .utf8)
        #expect(kept == "current")
    }

    @Test("a missing legacy directory is not an error")
    func toleratesMissingLegacyDirectory() throws {
        let support = try makeSupportDirectory()
        defer { try? FileManager.default.removeItem(at: support) }
        let outcome = AppSupportMigration.migrateModels(applicationSupport: support)
        #expect(outcome == AppSupportMigration.Outcome())
    }

    @Test("models resolve under the TUFF directory")
    func resolvesUnderTUFF() {
        let support = URL(fileURLWithPath: "/tmp/support", isDirectory: true)
        let resolved = AppModelLocation.resolve(
            explicitURL: nil,
            executableURL: nil,
            currentDirectoryURL: URL(fileURLWithPath: "/", isDirectory: true),
            applicationSupportURL: support,
            fileExists: { _ in false },
            installDirectoryName: "qwen36.gturbo")
        #expect(resolved.path == "/tmp/support/TUFF/Models/qwen36.gturbo")
    }
}
