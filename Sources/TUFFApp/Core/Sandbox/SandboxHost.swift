#if os(macOS)
import Darwin
import Foundation
import Security

/// The Mac's own network addresses, which the sandbox must never connect to
/// (Sandbox/server.py and the VM's firewall refuse every address passed in
/// `SANDBOX_BLOCKED_ADDRESSES`).
enum HostAddresses {
    /// The most the server accepts in its list.
    static let maximumCount = 256

    /// Every IPv4 and IPv6 address of every interface that is up, loopback
    /// and link-local included.
    static func current() -> [String] {
        var list: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&list) == 0, let first = list else { return [] }
        defer { freeifaddrs(list) }
        var found: [String] = []
        var cursor: UnsafeMutablePointer<ifaddrs>? = first
        while let entry = cursor {
            defer { cursor = entry.pointee.ifa_next }
            guard Int32(entry.pointee.ifa_flags) & IFF_UP != 0,
                  let address = entry.pointee.ifa_addr else { continue }
            let family = Int32(address.pointee.sa_family)
            guard family == AF_INET || family == AF_INET6 else { continue }
            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            let hostLength = socklen_t(host.count)
            if getnameinfo(address, socklen_t(address.pointee.sa_len), &host, hostLength,
                           nil, 0, NI_NUMERICHOST) == 0 {
                found.append(cString(host))
            }
        }
        return found
    }

    /// The addresses as the server reads them: no zone id (`fe80::1%en0`),
    /// lowercase, an IPv4-mapped IPv6 address as the IPv4 address it carries,
    /// each once, in a fixed order. Anything that is not an IP address is
    /// dropped.
    static func normalized(_ raw: [String]) -> [String] {
        var unique = Set<String>()
        for entry in raw {
            let withoutZone = entry.split(separator: "%", maxSplits: 1,
                                          omittingEmptySubsequences: false).first.map(String.init) ?? entry
            if let text = canonical(withoutZone.trimmingCharacters(in: .whitespaces)) {
                unique.insert(text)
            }
        }
        return unique.sorted()
    }

    /// The address as `inet_ntop` prints it, with a mapped IPv4 address
    /// reduced to the IPv4 form; nil for text that is not an address.
    static func canonical(_ text: String) -> String? {
        // macOS accepts `fe80::1%en0` and puts the interface number into the
        // address; a zone is removed by `normalized`, never passed on here.
        guard !text.contains("%") else { return nil }
        var v4 = in_addr()
        if inet_pton(AF_INET, text, &v4) == 1 { return render(v4) }
        var v6 = in6_addr()
        guard inet_pton(AF_INET6, text, &v6) == 1 else { return nil }
        let bytes = withUnsafeBytes(of: &v6) { Array($0) }
        if bytes[0..<10].allSatisfy({ $0 == 0 }), bytes[10] == 0xff, bytes[11] == 0xff {
            var mapped = in_addr()
            withUnsafeMutableBytes(of: &mapped) { target in
                for index in 0..<4 { target[index] = bytes[12 + index] }
            }
            return render(mapped)
        }
        return render(v6)
    }

    private static func render(_ value: in_addr) -> String? {
        var address = value
        var buffer = [CChar](repeating: 0, count: Int(INET_ADDRSTRLEN))
        let length = socklen_t(buffer.count)
        guard inet_ntop(AF_INET, &address, &buffer, length) != nil else { return nil }
        return cString(buffer)
    }

    private static func render(_ value: in6_addr) -> String? {
        var address = value
        var buffer = [CChar](repeating: 0, count: Int(INET6_ADDRSTRLEN))
        let length = socklen_t(buffer.count)
        guard inet_ntop(AF_INET6, &address, &buffer, length) != nil else { return nil }
        return cString(buffer)
    }

    /// The text of a C buffer up to its first zero byte.
    private static func cString(_ buffer: [CChar]) -> String {
        String(decoding: buffer.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }
}

enum LoopbackPort {
    /// A TCP port on 127.0.0.1 nothing is listening on right now. Another
    /// program could take it before the container publishes it; the start
    /// then fails with a clear error instead of talking to the wrong server,
    /// because every answer has to carry the secret-based signature.
    static func findFree() throws -> Int {
        let descriptor = socket(AF_INET, SOCK_STREAM, 0)
        guard descriptor >= 0 else { throw SandboxError.noFreePort }
        defer { close(descriptor) }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        address.sin_port = 0
        address.sin_addr = in_addr(s_addr: UInt32(0x7f00_0001).bigEndian)
        let size = socklen_t(MemoryLayout<sockaddr_in>.size)
        let bound = withUnsafePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(descriptor, $0, size) }
        }
        guard bound == 0 else { throw SandboxError.noFreePort }
        var length = size
        let named = withUnsafeMutablePointer(to: &address) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { getsockname(descriptor, $0, &length) }
        }
        guard named == 0 else { throw SandboxError.noFreePort }
        let port = Int(UInt16(bigEndian: address.sin_port))
        guard port > 0 else { throw SandboxError.noFreePort }
        return port
    }
}

/// Only one copy of TUFF runs the sandbox at a time. The lock belongs to the
/// process: the system drops it when the process ends, so a crash never
/// leaves a stale lock, and whoever holds it knows any container left behind
/// by an earlier run has no owner.
final class SandboxInstanceLock: @unchecked Sendable {
    private let lock = NSLock()
    private var descriptor: Int32

    private init(descriptor: Int32) { self.descriptor = descriptor }

    /// The lock on the file at `url`, or nil if another process holds it
    /// (or the file cannot be opened).
    static func acquire(at url: URL) -> SandboxInstanceLock? {
        let descriptor = open(url.path, O_CREAT | O_RDWR | O_CLOEXEC, 0o600)
        guard descriptor >= 0 else { return nil }
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            close(descriptor)
            return nil
        }
        return SandboxInstanceLock(descriptor: descriptor)
    }

    func release() {
        lock.withLock {
            guard descriptor >= 0 else { return }
            flock(descriptor, LOCK_UN)
            close(descriptor)
            descriptor = -1
        }
    }

    deinit { release() }
}

/// A fresh secret for one start of the sandbox, in a folder only this user
/// can read. The folder is mounted read-only into the VM, where the server
/// reads the secret once before it gives up its privileges.
struct SandboxSecret: Sendable {
    static let fileName = "secret"

    let folder: URL
    /// The content of the secret file: 64 hexadecimal characters. The
    /// signature key is these characters' bytes, not the 32 random bytes
    /// they spell out (see Sandbox/server.py, `load_secret`).
    let key: Data

    /// Makes `base/<random name>/secret`. `base` and the new folder are
    /// readable by this user only (0700) and the file is 0600.
    static func create(under base: URL) throws -> SandboxSecret {
        var bytes = [UInt8](repeating: 0, count: 32)
        guard SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) == errSecSuccess else {
            throw SandboxError.secretUnavailable("no random numbers")
        }
        let text = bytes.map { String(format: "%02x", $0) }.joined()
        let key = Data(text.utf8)
        let manager = FileManager.default
        let folder = base.appendingPathComponent(UUID().uuidString, isDirectory: true)
        do {
            try manager.createDirectory(at: base, withIntermediateDirectories: true,
                                        attributes: [.posixPermissions: 0o700])
            try manager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: base.path)
            try manager.createDirectory(at: folder, withIntermediateDirectories: false,
                                        attributes: [.posixPermissions: 0o700])
            let file = folder.appendingPathComponent(fileName)
            guard manager.createFile(atPath: file.path, contents: key,
                                     attributes: [.posixPermissions: 0o600]) else {
                throw SandboxError.secretUnavailable("the file could not be written")
            }
        } catch let error as SandboxError {
            try? manager.removeItem(at: folder)
            throw error
        } catch {
            try? manager.removeItem(at: folder)
            throw SandboxError.secretUnavailable(SandboxText.oneLine(error.localizedDescription))
        }
        return SandboxSecret(folder: folder, key: key)
    }

    /// The secret folder's path is passed to `--mount` as one comma-separated
    /// option, so a path with a comma (or a line break) cannot be mounted.
    var isMountable: Bool {
        !folder.path.contains(",") && !folder.path.contains("\n")
    }

    func remove() {
        try? FileManager.default.removeItem(at: folder)
    }

    /// Removes every secret folder under `base`, for leftovers of a run that
    /// did not end normally. Only call it while holding the instance lock.
    static func removeAll(under base: URL) {
        let manager = FileManager.default
        guard let entries = try? manager.contentsOfDirectory(
            at: base, includingPropertiesForKeys: nil) else { return }
        for entry in entries { try? manager.removeItem(at: entry) }
    }
}

enum SandboxLocations {
    /// `~/Library/Application Support/TUFF/Sandbox`
    static func defaultStateDirectory() -> URL {
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Library/Application Support", isDirectory: true)
        return support.appendingPathComponent("TUFF/Sandbox", isDirectory: true)
    }

    /// Each profile has a folder and a lock file of its own, so the web and
    /// the files sandbox neither share a lock nor remove each other's secret.
    static func secretsDirectory(in state: URL, profile: SandboxProfile = .web) -> URL {
        state.appendingPathComponent(profile.secretsDirectoryName, isDirectory: true)
    }

    static func lockFile(in state: URL, profile: SandboxProfile = .web) -> URL {
        state.appendingPathComponent(profile.lockFileName)
    }
}
#endif
