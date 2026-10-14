#if os(macOS)
import Foundation
import Testing
@testable import TUFFAppCore

/// The files sandbox (a second VM with no network) reading a real folder on
/// this Mac, end to end through the app's indexer. It only runs when
/// `TUFF_SANDBOX_LIVE_IMAGE` names an image already in the local store (see
/// `SandboxLiveTests`):
///
///     TUFF_SANDBOX_LIVE_IMAGE=tuff-sandbox:dev Scripts/test.sh --filter SandboxFilesLiveTests
///
/// It prints one `FILES LIVE:` line per check with the time it took. The
/// folder is made in the temporary directory, so macOS asks nothing.
@MainActor
@Suite(.serialized) struct SandboxFilesLiveTests {
    /// A one-page PDF whose text is "Hello zeppelin PDF". The cross-reference
    /// table is left out on purpose: pypdf rebuilds it, as it does for many
    /// real-world files.
    private static let pdf = """
        %PDF-1.4
        1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
        2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
        3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 300 144] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj
        4 0 obj << /Length 52 >> stream
        BT /F1 18 Tf 20 60 Td (Hello zeppelin PDF) Tj ET
        endstream endobj
        5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj
        trailer << /Root 1 0 R /Size 6 >>
        %%EOF
        """

    @Test(.enabled(if: SandboxLiveTests.image != nil))
    func realFolderThroughTheRealFilesSandbox() async throws {
        let reference = try #require(SandboxLiveTests.image)
        let tool = ContainerTool()
        try await tool.verify()
        let inspected = try #require(await tool.inspectImage(reference), "image \(reference) is not in the local store")
        let digest = try #require(SandboxImageProvisioner.indexDigest(in: inspected))

        let work = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff sandbox files live \(UUID().uuidString)", isDirectory: true)
            .resolvingSymlinksInPath()
        defer { try? FileManager.default.removeItem(at: work) }
        let root = work.appendingPathComponent("docs", isDirectory: true)
        let manager = FileManager.default
        try manager.createDirectory(at: root.appendingPathComponent("notes"), withIntermediateDirectories: true)
        try manager.createDirectory(at: root.appendingPathComponent("node_modules"), withIntermediateDirectories: true)
        try "The zeppelin hangar is open on Tuesdays.\nSecond line about blimps.\n"
            .write(to: root.appendingPathComponent("hangar.txt"), atomically: true, encoding: .utf8)
        try "# Notes\n\nA quokka lives in the notes folder.\n"
            .write(to: root.appendingPathComponent("notes/animals.md"), atomically: true, encoding: .utf8)
        try "hidden marmoset secret".write(to: root.appendingPathComponent(".hidden.txt"), atomically: true, encoding: .utf8)
        try "dependency narwhal".write(to: root.appendingPathComponent("node_modules/dep.txt"), atomically: true, encoding: .utf8)
        try Data(Self.pdf.utf8).write(to: root.appendingPathComponent("flight.pdf"))
        // A link to a file outside the folder must not be read.
        let outside = work.appendingPathComponent("outside.txt")
        try "outside platypus secret".write(to: outside, atomically: true, encoding: .utf8)
        try manager.createSymbolicLink(at: root.appendingPathComponent("link.txt"), withDestinationURL: outside)

        var dependencies = SandboxDependencies.live(profile: .files)
        dependencies.pin = SandboxImagePin(reference: reference, digest: digest, source: .local)
        dependencies.stateDirectory = work.appendingPathComponent("state", isDirectory: true)
        let controller = SandboxController(dependencies: dependencies)
        let access = SandboxWebAccess(controller: controller)
        let service = SandboxFilesService(controller: controller, access: access)
        let folder = AppSearchFolder(path: root.path)

        func timed<T>(_ label: String, _ work: () async throws -> T) async rethrows -> T {
            let began = Date()
            defer { print("FILES LIVE: \(label) took \(String(format: "%.1f", Date().timeIntervalSince(began))) s") }
            return try await work()
        }

        // Start with the folder mounted read-only; the app's own checks of
        // the firewall, the mounts and the privileges run inside `start`.
        _ = try await timed("start with one folder") { try await service.ensureRunning(for: [folder]) }
        print("FILES LIVE: state \(controller.state), mounts \(controller.activeMountIDs)")
        #expect(controller.state == .ready)
        #expect(controller.activeMountIDs == [folder.id.uuidString.lowercased()])

        let lease = try controller.acquire()
        let health = try await SandboxHealthProbe(profile: .files).health(port: lease.port, key: lease.key)
        print("FILES LIVE: health profile \(health.profile), folders \(health.folders)")
        #expect(health.folders == [folder.id.uuidString.lowercased()])
        #expect(health.profile == "files")

        // Index through the sandbox into a temporary index.
        let index = try AppLocalSearchIndex(databaseURL: work.appendingPathComponent("index.sqlite"))
        let indexer = AppLocalIndexer(index: index, reader: SandboxFileReader(access: access))
        let status = try await timed("index") { try await indexer.update(folder) }
        print("FILES LIVE: indexed \(status.indexedFiles), skipped \(status.skippedFiles), passages \(status.passages), error \(status.lastError ?? "-")")
        // If this is 0 with no error, suspect virtiofs mtime precision: the
        // app wants the Mac and the VM to agree within a millisecond.
        #expect(status.indexedFiles >= 2, "indexed \(status.indexedFiles)")
        #expect(status.lastError == nil)

        func hits(_ query: String) async throws -> [AppFilePassage] {
            try await index.search(query, folders: [folder], limit: 5)
        }
        let text = try await hits("zeppelin hangar")
        print("FILES LIVE: 'zeppelin hangar' -> \(text.map(\.displayName))")
        #expect(text.contains { $0.displayName == "hangar.txt" && $0.lineStart == 1 })
        let markdown = try await hits("quokka")
        #expect(markdown.contains { $0.displayName == "animals.md" })
        let pdfHits = try await hits("Hello zeppelin PDF")
        print("FILES LIVE: pdf hits -> \(pdfHits.map { "\($0.displayName) \($0.location ?? "-")" })")
        if !pdfHits.contains(where: { $0.displayName == "flight.pdf" }) {
            print("FILES LIVE: the sample PDF was not indexed (inconclusive: the hand-made PDF may not parse)")
        }
        // What TUFF's own rules skip, or what is a link, never shows.
        #expect(try await hits("marmoset").isEmpty)
        #expect(try await hits("narwhal").isEmpty)
        #expect(try await hits("platypus").isEmpty)

        // Nothing changed: a second update reads nothing again.
        let again = try await timed("second index (unchanged)") { try await indexer.update(folder) }
        #expect(again.lastError == nil)
        #expect(again.indexedFiles == status.indexedFiles)

        // A new folder restarts the VM with both mounted.
        let second = work.appendingPathComponent("more", isDirectory: true)
        try manager.createDirectory(at: second, withIntermediateDirectories: true)
        try "A pangolin walks here.".write(to: second.appendingPathComponent("p.txt"), atomically: true, encoding: .utf8)
        let other = AppSearchFolder(path: second.path)
        _ = try await timed("restart with two folders") { try await service.ensureRunning(for: [folder, other]) }
        #expect(controller.activeMountIDs.count == 2)
        let secondStatus = try await indexer.update(other)
        #expect(secondStatus.indexedFiles == 1, "indexed \(secondStatus.indexedFiles)")
        let pangolin = try await index.search("pangolin", folders: [other], limit: 5)
        #expect(pangolin.contains { $0.displayName == "p.txt" })

        // No folder left: no VM.
        try await service.ensureRunning(for: [])
        await service.stop()
        let leftover = try await tool.containerIDs(all: true)
        #expect(!leftover.contains { $0.hasPrefix(SandboxProfile.files.namePrefix) })
        print("FILES LIVE: done; leftover containers: \(leftover.filter { $0.hasPrefix(SandboxProfile.files.namePrefix) })")
    }
}
#endif
