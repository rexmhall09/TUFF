import Darwin
import Foundation

public struct RemoteInstallPaths: Sendable, Equatable {
    public let parentDirectory: String
    public let finalDirectory: String
    public let partialDirectory: String
    public let lockFile: String
    public let checkpointFile: String
    public let rangeTemporaryFile: String
    public let metadataDirectory: String

    public init(outputDirectory: String) throws {
        let standardized = URL(fileURLWithPath: outputDirectory).standardizedFileURL
        let basename = standardized.lastPathComponent.precomposedStringWithCanonicalMapping
        guard !basename.isEmpty,
              basename != ".",
              basename != "..",
              !basename.contains("/") else {
            throw RepackError.installPathUnsafe(
                path: outputDirectory,
                detail: "invalid output basename")
        }
        let requestedParent = standardized.deletingLastPathComponent().path
        try Posix.mkdirP(requestedParent)
        let parent = try Posix.physicalPath(requestedParent)
        let final = (parent as NSString).appendingPathComponent(basename)
        let partial = final + ".partial"
        self.parentDirectory = parent
        self.finalDirectory = final
        self.partialDirectory = partial
        self.lockFile = final + ".install.lock"
        self.checkpointFile = final + ".resume.json"
        self.rangeTemporaryFile = (partial as NSString)
            .appendingPathComponent(".range.tmp")
        self.metadataDirectory = (partial as NSString)
            .appendingPathComponent(".remote-metadata")
    }

    public func validateEntryTypes() throws {
        try require(finalDirectory, kinds: [.absent, .directory])
        try require(partialDirectory, kinds: [.absent, .directory])
        try require(checkpointFile, kinds: [.absent, .regular])
        try require(lockFile, kinds: [.absent, .regular])
    }

    private func require(_ path: String, kinds: Set<Posix.EntryKind>) throws {
        let kind = try Posix.entryKind(path)
        guard kinds.contains(kind) else {
            throw RepackError.installPathUnsafe(
                path: path,
                detail: "unexpected \(kind) entry")
        }
    }

    /// One install path occupied by something the installer will not write over.
    public struct BlockingEntry: Equatable, Sendable {
        public let path: String
        public let kind: Posix.EntryKind

        public init(path: String, kind: Posix.EntryKind) {
            self.path = path
            self.kind = kind
        }
    }

    private var requirements: [(path: String, kinds: Set<Posix.EntryKind>)] {
        [(finalDirectory, [.absent, .directory]),
         (partialDirectory, [.absent, .directory]),
         (checkpointFile, [.absent, .regular]),
         (lockFile, [.absent, .regular])]
    }

    /// The entries that make `validateEntryTypes` throw. A symlink left where a
    /// model directory belongs is the common case: the installer refuses to
    /// follow it, and nothing in the normal download flow can clear it.
    public func blockingEntries() throws -> [BlockingEntry] {
        try requirements.compactMap { requirement in
            let kind = try Posix.entryKind(requirement.path)
            guard !requirement.kinds.contains(kind) else { return nil }
            return BlockingEntry(path: requirement.path, kind: kind)
        }
    }

    /// Remove exactly those entries. Removing a symlink unlinks the link and
    /// never touches whatever it pointed at.
    @discardableResult
    public func removeBlockingEntries() throws -> [BlockingEntry] {
        let blocked = try blockingEntries()
        for entry in blocked {
            try Posix.removeEntry(entry.path)
        }
        return blocked
    }
}

public final class InstallLock: @unchecked Sendable {
    public let paths: RemoteInstallPaths
    private let descriptor: Int32

    private init(paths: RemoteInstallPaths, descriptor: Int32) {
        self.paths = paths
        self.descriptor = descriptor
    }

    deinit {
        _ = flock(descriptor, LOCK_UN)
        close(descriptor)
    }

    public static func acquire(outputDirectory: String) throws -> InstallLock {
        let paths = try RemoteInstallPaths(outputDirectory: outputDirectory)
        try paths.validateEntryTypes()
        let descriptor = try Posix.openLock(paths.lockFile)
        guard flock(descriptor, LOCK_EX | LOCK_NB) == 0 else {
            let savedErrno = errno
            close(descriptor)
            if savedErrno == EWOULDBLOCK {
                throw RepackError.installBusy(path: paths.lockFile)
            }
            throw RepackError.fileOpenFailed(path: paths.lockFile, errno: savedErrno)
        }
        do {
            try paths.validateEntryTypes()
            guard try Posix.descriptorMatchesPath(descriptor, path: paths.lockFile) else {
                throw RepackError.installPathUnsafe(
                    path: paths.lockFile,
                    detail: "lock path was replaced during acquisition")
            }
            return InstallLock(paths: paths, descriptor: descriptor)
        } catch {
            _ = flock(descriptor, LOCK_UN)
            close(descriptor)
            throw error
        }
    }
}
