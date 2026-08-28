import Foundation

/// v2.0.0 kept chats under `Application Support/TUFF` while models stayed in the
/// older `Application Support/TurboFieldfare` directory. Everything the app owns
/// now lives under `TUFF`, so an existing install has its model directories moved
/// across once, on launch, before any of them is opened.
public enum AppSupportMigration {
    public static let legacyDirectoryName = "TurboFieldfare"
    public static let currentDirectoryName = "TUFF"
    public static let modelsDirectoryName = "Models"

    public struct Outcome: Equatable, Sendable {
        public var movedEntries: [String] = []
        public var skippedEntries: [String] = []
        public var removedLegacyRoot = false
    }

    /// Move every entry from the legacy directory into `TUFF/Models`. A rename
    /// inside one volume is cheap no matter how large the model is, so this stays
    /// fast even for a 61 GiB checkpoint. Anything already present at the
    /// destination is left untouched rather than overwritten.
    @discardableResult
    public static func migrateModels(
        applicationSupport: URL,
        fileManager: FileManager = .default
    ) -> Outcome {
        var outcome = Outcome()
        let legacyRoot = applicationSupport
            .appendingPathComponent(legacyDirectoryName, isDirectory: true)
        var isDirectory: ObjCBool = false
        guard fileManager.fileExists(atPath: legacyRoot.path, isDirectory: &isDirectory),
              isDirectory.boolValue else { return outcome }

        let destinationRoot = applicationSupport
            .appendingPathComponent(currentDirectoryName, isDirectory: true)
            .appendingPathComponent(modelsDirectoryName, isDirectory: true)
        guard let entries = try? fileManager.contentsOfDirectory(
            atPath: legacyRoot.path) else { return outcome }

        if !entries.isEmpty {
            try? fileManager.createDirectory(
                at: destinationRoot, withIntermediateDirectories: true)
        }

        for entry in entries.sorted() where entry != ".DS_Store" {
            let source = legacyRoot.appendingPathComponent(entry)
            let destination = destinationRoot.appendingPathComponent(entry)
            if fileManager.fileExists(atPath: destination.path) {
                outcome.skippedEntries.append(entry)
                continue
            }
            do {
                try fileManager.moveItem(at: source, to: destination)
                outcome.movedEntries.append(entry)
            } catch {
                outcome.skippedEntries.append(entry)
            }
        }

        // Only clear the old root once nothing of consequence is left in it. A
        // stray .DS_Store should not keep an empty directory alive forever.
        let remaining = (try? fileManager.contentsOfDirectory(atPath: legacyRoot.path)) ?? []
        if remaining.allSatisfy({ $0 == ".DS_Store" }) {
            if (try? fileManager.removeItem(at: legacyRoot)) != nil {
                outcome.removedLegacyRoot = true
            }
        }
        return outcome
    }

    public static func applicationSupportURL(fileManager: FileManager = .default) -> URL {
        fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? fileManager.homeDirectoryForCurrentUser
                .appendingPathComponent("Library/Application Support", isDirectory: true)
    }
}
