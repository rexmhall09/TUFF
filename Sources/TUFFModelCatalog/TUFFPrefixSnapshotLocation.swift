import CryptoKit
import Foundation

/// Where saved prompt-prefix snapshots live. The engine writes them; the app
/// only needs the folders so deleting a model or a chat can take its
/// snapshots with it.
public enum TUFFPrefixSnapshotLocation {
    /// `Caches/TUFF/PrefixSnapshots`, holding one folder per model. A scoped
    /// data root keeps the cache inside it, as local builds expect.
    public static func rootDirectory(environment: [String: String] = ProcessInfo.processInfo.environment,
                                     fileManager: FileManager = .default) -> URL? {
        let caches: URL
        if let scoped = environment["TUFF_APPLICATION_SUPPORT_ROOT"],
           scoped.hasPrefix("/"), !scoped.contains("\0") {
            caches = URL(fileURLWithPath: scoped, isDirectory: true).standardizedFileURL
                .appendingPathComponent("TUFF/Caches", isDirectory: true)
        } else if let normal = fileManager.urls(for: .cachesDirectory, in: .userDomainMask).first {
            caches = normal
        } else { return nil }
        return caches.appendingPathComponent("TUFF/PrefixSnapshots", isDirectory: true)
    }

    /// One model's folder, holding one folder per build.
    public static func modelDirectory(modelID: String,
                                      environment: [String: String] = ProcessInfo.processInfo.environment,
                                      fileManager: FileManager = .default) -> URL? {
        let component = safeComponent(modelID)
        // "." and ".." survive the character filter; either would name a
        // parent folder that holds every model's snapshots.
        guard !component.isEmpty, component != ".", component != ".." else { return nil }
        return rootDirectory(environment: environment, fileManager: fileManager)?
            .appendingPathComponent(component, isDirectory: true)
    }

    /// A saved chat's file, the same in every model and build folder, so the
    /// app can find a chat's state without knowing which model wrote it. The
    /// key is hashed because a chat id is not a file name.
    public static func chatSnapshotFileName(conversationKey: String) -> String {
        let digest = SHA256.hash(data: Data(conversationKey.utf8))
            .map { String(format: "%02x", $0) }.joined()
        return "chat-\(digest).tuffprefix"
    }

    /// Removes a chat's saved state for every model and build. It holds the
    /// conversation's computed state, so it goes when the chat does.
    public static func removeChatSnapshots(conversationKey: String,
                                           environment: [String: String] = ProcessInfo.processInfo.environment,
                                           fileManager: FileManager = .default) {
        guard let root = rootDirectory(environment: environment, fileManager: fileManager),
              let models = try? fileManager.contentsOfDirectory(
                at: root, includingPropertiesForKeys: nil) else { return }
        let name = chatSnapshotFileName(conversationKey: conversationKey)
        for model in models {
            for build in (try? fileManager.contentsOfDirectory(
                at: model, includingPropertiesForKeys: nil)) ?? [] {
                try? fileManager.removeItem(at: build.appendingPathComponent(name))
            }
        }
    }

    /// Letters, digits, `-` and `.` pass through; anything else becomes `_`.
    public static func safeComponent(_ text: String) -> String {
        String(text.map { $0.isLetter || $0.isNumber || $0 == "-" || $0 == "." ? $0 : "_" })
    }
}
