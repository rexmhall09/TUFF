#if os(macOS)
import Foundation

/// The two kinds of sandbox the app runs from the same image. They differ in
/// what is mounted, what the firewall allows, and which endpoints the server
/// offers (see `Sandbox/server.py`, `SANDBOX_PROFILE`).
enum SandboxProfile: String, Sendable, Equatable {
    /// Fetches web pages and search results. Network, no folders.
    case web
    /// Reads the folders the person chose for file search. Folders mounted
    /// read-only, no network at all.
    case files

    /// The start of every container name of this profile.
    var namePrefix: String { "tuff-sandbox-\(rawValue)-" }

    /// The nft table the VM's entrypoint loads (`Sandbox/firewall-*.nft`).
    var tableName: String { "tuff_sandbox_\(rawValue)" }

    /// What the signed `/health` answer calls this profile.
    var healthName: String { rawValue }

    var lockFileName: String {
        switch self {
        case .web: "instance.lock"
        case .files: "instance-files.lock"
        }
    }

    var secretsDirectoryName: String {
        switch self {
        case .web: "secrets"
        case .files: "secrets-files"
        }
    }

    /// A phrase for messages.
    var title: String {
        switch self {
        case .web: "web sandbox"
        case .files: "file sandbox"
        }
    }
}

/// A folder mounted read-only into the files sandbox at `/files/<id>`.
struct SandboxFolderMount: Equatable, Sendable {
    /// The lowercase UUID text of the search folder, which is also the name
    /// the server and the app use for it.
    let id: String
    /// The folder on the Mac, canonical (symbolic links resolved).
    let path: String

    static let target = "/files"
    /// The most folders one files sandbox mounts (and reports in `/health`).
    static let maximumCount = 64

    /// The home folder, or a folder that contains it, would give the sandbox
    /// the person's keys and app data. Folders inside the home folder are not
    /// refused, except `~/.ssh` (see `isKeyFolder`): TUFF's own indexer reads
    /// them the same way, so choosing `~/Library` on purpose is the person's
    /// decision.
    static func isTooBroad(_ path: String, home: String = NSHomeDirectory()) -> Bool {
        let canonical = canonicalPath(home) ?? home
        let slash = path.hasSuffix("/") ? path : path + "/"
        for candidate in [home, canonical] where path == candidate || candidate.hasPrefix(slash) {
            return true
        }
        // The same folder under another name (the firmlink /System/Volumes/Data
        // is not a symbolic link): compare what the paths are, not what they say.
        guard let target = identity(of: path) else { return false }
        // A folder that holds the home folder under another name, such as the
        // data volume at /System/Volumes/Data (it holds /Users, firmlinked): the
        // home folder's path, taken inside this folder, is the home folder.
        if let homeID = identity(of: canonical),
           identity(of: (path.hasSuffix("/") ? String(path.dropLast()) : path) + canonical) == homeID {
            return true
        }
        var ancestor = canonical
        while ancestor != "/", !ancestor.isEmpty {
            if identity(of: ancestor) == target { return true }
            ancestor = (ancestor as NSString).deletingLastPathComponent
        }
        return false
    }

    /// The folder of the person's SSH keys, or anything inside it. It is the
    /// one folder inside the home folder that is never given to the sandbox
    /// (the person's decision, 13 Oct 2026): private keys have no business in
    /// a search index. Compared by what the folders are, so a link to it does
    /// not help.
    static func isKeyFolder(_ path: String, home: String = NSHomeDirectory()) -> Bool {
        let canonical = canonicalPath(home) ?? home
        let given = path.hasSuffix("/") && path.count > 1 ? String(path.dropLast()) : path
        for base in Set([home, canonical]) {
            let ssh = base + "/.ssh"
            if given == ssh || given.hasPrefix(ssh + "/") { return true }
        }
        guard let sshID = identity(of: canonical + "/.ssh") else { return false }
        var ancestor = given
        while ancestor != "/", !ancestor.isEmpty {
            if identity(of: ancestor) == sshID { return true }
            ancestor = (ancestor as NSString).deletingLastPathComponent
        }
        return false
    }

    private static func identity(of path: String) -> [UInt64]? {
        var info = stat()
        guard stat(path, &info) == 0 else { return nil }
        return [UInt64(bitPattern: Int64(info.st_dev)), UInt64(info.st_ino)]
    }

    var targetPath: String { "\(Self.target)/\(id)" }

    /// The mount option as `container run --mount` takes it.
    var argument: String { "type=bind,source=\(path),target=\(targetPath),readonly" }

    /// Whether `text` is a folder id as the server accepts it: a lowercase UUID.
    static func isFolderID(_ text: String) -> Bool {
        text.wholeMatch(of: #/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/#) != nil
    }

    /// The path goes into one comma-separated option, so a comma or a line
    /// break cannot be mounted; nor can a NUL or a relative path.
    static func isMountable(_ path: String) -> Bool {
        path.hasPrefix("/") && !path.contains(",") && !path.contains("\n")
            && !path.contains("\r") && !path.unicodeScalars.contains { $0.value == 0 }
    }
}
#endif
