import CryptoKit
import Foundation
import Metal

/// Runner state at the end of a shared prompt prefix, kept on disk so a new
/// process can skip reading an agent's system prompt and tools again.
///
/// A snapshot is written once, when its prefix is first read, and never while
/// tokens are generated: those writes would compete with expert reads on the
/// same SSD. It is loaded only when the model, its runtime settings, the app
/// build and every prefix token match, and any mismatch, damage or layout
/// change falls back to an ordinary cold prefill. The directory is capped at
/// `budgetBytes`, and the least recently used files go first.
public final class PrefixSnapshotDiskStore: @unchecked Sendable {
    public let directory: URL
    public let budgetBytes: Int
    private let buildIdentity: String
    private let clearsSiblingBuilds: Bool
    private let writes = DispatchQueue(label: "tuff.prefix-snapshots")
    private let lock = NSLock()

    static let magic = Data("TUFFPFX1".utf8)
    static let formatVersion = 1

    /// With `clearsSiblingBuilds`, `directory` is this build's folder and
    /// writing a snapshot removes the other builds' folders beside it.
    public init(directory: URL, budgetBytes: Int, buildIdentity: String,
                clearsSiblingBuilds: Bool = false) {
        self.directory = directory
        self.budgetBytes = max(0, budgetBytes)
        self.buildIdentity = buildIdentity
        self.clearsSiblingBuilds = clearsSiblingBuilds
    }

    /// The default location and budget. `TUFF_PREFIX_DISK_CACHE_MB` lowers the
    /// budget, and zero turns the store off.
    public static func standard(modelID: String, buildIdentity: String,
                                environment: [String: String] = ProcessInfo.processInfo.environment)
        -> PrefixSnapshotDiskStore? {
        var budget = defaultBudgetBytes
        if let override = environment["TUFF_PREFIX_DISK_CACHE_MB"],
           let megabytes = Int(override.trimmingCharacters(in: .whitespaces)) {
            budget = min(budget, max(0, megabytes) << 20)
        }
        guard budget > 0,
              let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
        else { return nil }
        func safe(_ text: String) -> String {
            String(text.map { $0.isLetter || $0.isNumber || $0 == "-" || $0 == "." ? $0 : "_" })
        }
        // One folder per build, so an update can clear what it can never use.
        return PrefixSnapshotDiskStore(
            directory: caches.appendingPathComponent(
                "TUFF/PrefixSnapshots/\(safe(modelID))/\(safe(buildIdentity))"),
            budgetBytes: budget, buildIdentity: buildIdentity, clearsSiblingBuilds: true)
    }

    /// 2 GiB: a few dozen agent prompts for most models.
    public static let defaultBudgetBytes = 2 << 30

    private struct Header: Codable, Equatable {
        struct Segment: Codable, Equatable {
            let label: String
            let sourceOffset: Int
            let snapshotOffset: Int
            let length: Int
        }
        let version: Int
        let identity: String
        let tokens: [Int32]
        let position: Int
        let ngramContext: [Int32]
        let ropeDelta: Int32
        let segments: [Segment]
        let payloadBytes: Int
        let payloadSHA256: String
    }

    /// Everything outside the tokens a snapshot depends on.
    private func identity(_ domain: ConversationCacheDomain) -> String {
        [buildIdentity, domain.modelID, domain.sourceSnapshotHash ?? "-",
         domain.runtimeProfileHash, String(domain.maximumContext), domain.kvStorage,
         String(domain.fp16RingEnabled), domain.templateSHA256].joined(separator: "|")
    }

    private func fileURL(tokens: [Int32], domain: ConversationCacheDomain) -> URL {
        var hasher = SHA256()
        hasher.update(data: Data(identity(domain).utf8))
        tokens.withUnsafeBytes { hasher.update(bufferPointer: $0) }
        let name = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        return directory.appendingPathComponent(name + ".tuffprefix")
    }

    public func contains(tokens: [Int32], domain: ConversationCacheDomain) -> Bool {
        FileManager.default.fileExists(atPath: fileURL(tokens: tokens, domain: domain).path)
    }

    /// Writes `snapshot` in the background. It must hold the state after
    /// exactly `tokens`; a failed write leaves nothing behind.
    public func save(_ snapshot: RunnerStateSnapshot, tokens: [Int32],
                     domain: ConversationCacheDomain) {
        guard snapshot.position == tokens.count, budgetBytes > 0,
              let length = snapshot.storage?.length, length <= budgetBytes else { return }
        let url = fileURL(tokens: tokens, domain: domain)
        let identity = identity(domain)
        writes.async { [self] in
            guard let storage = snapshot.storage else { return }
            // Read in place: copying a few hundred megabytes of KV would
            // double the memory this costs while it is written.
            let payload = Data(bytesNoCopy: storage.contents(), count: storage.length,
                               deallocator: .none)
            let header = Header(
                version: Self.formatVersion, identity: identity, tokens: tokens,
                position: snapshot.host.position, ngramContext: snapshot.host.ngramContext,
                ropeDelta: snapshot.host.ropeDelta,
                segments: snapshot.segments.map {
                    .init(label: $0.label, sourceOffset: $0.sourceOffset,
                          snapshotOffset: $0.snapshotOffset, length: $0.length)
                },
                payloadBytes: payload.count,
                payloadSHA256: SHA256.hash(data: payload).map { String(format: "%02x", $0) }.joined())
            guard let headerData = try? JSONEncoder().encode(header) else { return }
            var prefix = Self.magic
            var headerLength = UInt32(headerData.count).littleEndian
            prefix.append(Data(bytes: &headerLength, count: 4))
            prefix.append(headerData)
            lock.withLock {
                let temporary = directory.appendingPathComponent(".\(UUID().uuidString).partial")
                do {
                    try FileManager.default.createDirectory(at: directory,
                                                            withIntermediateDirectories: true)
                    FileManager.default.createFile(atPath: temporary.path, contents: nil)
                    let handle = try FileHandle(forWritingTo: temporary)
                    defer { try? handle.close() }
                    try handle.write(contentsOf: prefix)
                    try handle.write(contentsOf: payload)
                    try handle.close()
                    // A reader sees either no file or a whole one.
                    _ = try? FileManager.default.removeItem(at: url)
                    try FileManager.default.moveItem(at: temporary, to: url)
                    evict(keeping: url)
                    removeOtherBuilds()
                } catch {
                    try? FileManager.default.removeItem(at: temporary)
                }
            }
            withExtendedLifetime(snapshot) {}
        }
    }

    /// Waits for writes already started, for tests and orderly shutdown.
    public func flush() { writes.sync {} }

    /// The snapshot for exactly `tokens`, rebuilt for `runner`, or nil. A
    /// damaged or mismatched file is removed.
    public func load(tokens: [Int32], domain: ConversationCacheDomain,
                     runner: any StateSnapshottingRunner, device: MTLDevice) -> RunnerStateSnapshot? {
        let url = fileURL(tokens: tokens, domain: domain)
        guard let file = try? Data(contentsOf: url, options: .mappedIfSafe) else { return nil }
        guard let snapshot = decode(file, tokens: tokens, identity: identity(domain),
                                    owner: runner.snapshotOwner, device: device) else {
            try? FileManager.default.removeItem(at: url)
            return nil
        }
        // Marks it recently used for eviction.
        try? FileManager.default.setAttributes([.modificationDate: Date()], ofItemAtPath: url.path)
        return snapshot
    }

    private func decode(_ file: Data, tokens: [Int32], identity: String,
                        owner: ObjectIdentifier, device: MTLDevice) -> RunnerStateSnapshot? {
        let magic = Self.magic
        guard file.count > magic.count + 4, file.prefix(magic.count) == magic else { return nil }
        let lengthStart = file.startIndex + magic.count
        let headerLength = file[lengthStart..<(lengthStart + 4)].withUnsafeBytes {
            Int(UInt32(littleEndian: $0.loadUnaligned(as: UInt32.self)))
        }
        let headerStart = lengthStart + 4
        guard headerLength > 0, file.count >= magic.count + 4 + headerLength,
              let header = try? JSONDecoder().decode(
                Header.self, from: file[headerStart..<(headerStart + headerLength)]),
              header.version == Self.formatVersion, header.identity == identity,
              header.tokens == tokens, header.position == tokens.count else { return nil }
        let payload = file[(headerStart + headerLength)...]
        guard payload.count == header.payloadBytes,
              SHA256.hash(data: payload).map({ String(format: "%02x", $0) }).joined()
                == header.payloadSHA256 else { return nil }
        let storage: MTLBuffer?
        if payload.isEmpty {
            storage = nil
        } else {
            guard let buffer = payload.withUnsafeBytes({
                device.makeBuffer(bytes: $0.baseAddress!, length: payload.count,
                                  options: .storageModeShared)
            }) else { return nil }
            buffer.label = "conversation.snapshot.disk"
            storage = buffer
        }
        // The runner's restore checks every segment against its own layout,
        // so a file from a differently shaped runner is refused there.
        return RunnerStateSnapshot(
            owner: owner, storage: storage,
            segments: header.segments.map {
                .init(label: $0.label, sourceOffset: $0.sourceOffset,
                      snapshotOffset: $0.snapshotOffset, length: $0.length)
            },
            host: .init(position: header.position, ngramContext: header.ngramContext,
                        ropeDelta: header.ropeDelta))
    }

    /// Snapshots from another build never load, so they only take space.
    private func removeOtherBuilds() {
        guard clearsSiblingBuilds else { return }
        let parent = directory.deletingLastPathComponent()
        guard let folders = try? FileManager.default.contentsOfDirectory(
            at: parent, includingPropertiesForKeys: [.isDirectoryKey]) else { return }
        for folder in folders where folder.lastPathComponent != directory.lastPathComponent
            && (try? folder.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true {
            try? FileManager.default.removeItem(at: folder)
        }
    }

    /// Removes the least recently used files until the directory fits.
    private func evict(keeping kept: URL) {
        let keys: [URLResourceKey] = [.contentModificationDateKey, .fileSizeKey]
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: keys) else { return }
        for stale in files where stale.pathExtension == "partial" && stale != kept {
            // Left behind by a process that stopped mid-write.
            if let date = try? stale.resourceValues(forKeys: [.contentModificationDateKey])
                .contentModificationDate, date < Date().addingTimeInterval(-600) {
                try? FileManager.default.removeItem(at: stale)
            }
        }
        var entries = files.filter { $0.pathExtension == "tuffprefix" }.compactMap { url -> (URL, Date, Int)? in
            guard let values = try? url.resourceValues(forKeys: Set(keys)) else { return nil }
            return (url, values.contentModificationDate ?? .distantPast, values.fileSize ?? 0)
        }
        var total = entries.reduce(0) { $0 + $1.2 }
        entries.sort { $0.1 < $1.1 }
        for (url, _, size) in entries where total > budgetBytes && url != kept {
            try? FileManager.default.removeItem(at: url)
            total -= size
        }
    }
}
