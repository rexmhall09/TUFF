#if os(macOS)
import Darwin
import Foundation
import Testing
@testable import TUFFAppCore

// MARK: Helpers

private func hostTestTemporaryDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("tuff-sandbox-host-test-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

private func hostTestPermissions(of url: URL) throws -> Int {
    let attributes = try FileManager.default.attributesOfItem(atPath: url.path)
    let value = try #require((attributes[.posixPermissions] as? NSNumber)?.intValue)
    return value & 0o777
}

/// A TCP socket bound to 127.0.0.1:`port`, or nil when the port is taken.
private func hostTestBindLoopbackSocket(port: Int) -> Int32? {
    let descriptor = socket(AF_INET, SOCK_STREAM, 0)
    guard descriptor >= 0 else { return nil }
    var address = sockaddr_in()
    address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
    address.sin_family = sa_family_t(AF_INET)
    address.sin_port = UInt16(port).bigEndian
    address.sin_addr = in_addr(s_addr: UInt32(0x7f00_0001).bigEndian)
    let size = socklen_t(MemoryLayout<sockaddr_in>.size)
    let bound = withUnsafePointer(to: &address) { pointer in
        pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(descriptor, $0, size) }
    }
    guard bound == 0 else {
        close(descriptor)
        return nil
    }
    return descriptor
}

// MARK: HostAddresses

@Suite struct HostAddressesTests {
    @Test func theServersLimitIs256() {
        #expect(HostAddresses.maximumCount == 256)
    }

    @Test func normalizedStripsTheZoneId() {
        #expect(HostAddresses.normalized(["fe80::1%en0"]) == ["fe80::1"])
        #expect(HostAddresses.normalized(["fe80::abcd%lo0", "fe80::abcd%en1"]) == ["fe80::abcd"])
        #expect(HostAddresses.normalized(["10.0.0.1%en0"]) == ["10.0.0.1"])
    }

    @Test func normalizedLowercasesAndCompressesIPv6() {
        #expect(HostAddresses.normalized(["FE80::ABCD", "2001:DB8::A"]) == ["2001:db8::a", "fe80::abcd"])
        #expect(HostAddresses.normalized(["2001:0db8:0000:0000:0000:0000:0000:0001"]) == ["2001:db8::1"])
    }

    @Test func normalizedReducesIPv4MappedAddressesToIPv4() {
        #expect(HostAddresses.normalized(["::ffff:192.168.1.5"]) == ["192.168.1.5"])
        #expect(HostAddresses.normalized(["::FFFF:C0A8:105"]) == ["192.168.1.5"])
        #expect(HostAddresses.normalized(["::ffff:127.0.0.1%lo0"]) == ["127.0.0.1"])
    }

    @Test func normalizedKeepsOrdinaryIPv6AndTheLoopback() {
        #expect(HostAddresses.normalized(["2001:db8::1"]) == ["2001:db8::1"])
        #expect(HostAddresses.normalized(["::1"]) == ["::1"])
        #expect(HostAddresses.normalized(["::"]) == ["::"])
    }

    @Test func normalizedListsEachAddressOnce() {
        #expect(HostAddresses.normalized(["10.0.0.1", "10.0.0.1", "::ffff:10.0.0.1", "10.0.0.1%en0"])
            == ["10.0.0.1"])
        #expect(HostAddresses.normalized(["2001:db8::1", "2001:0DB8:0:0:0:0:0:1", "2001:db8:0::1"])
            == ["2001:db8::1"])
    }

    @Test func normalizedDropsEverythingThatIsNotAnAddress() {
        let garbage = ["not an address", "", "   ", "999.1.1.1", "1.2.3.4.5", "gggg::1", "%en0",
                       "10.0.0.1/8", "localhost"]
        #expect(HostAddresses.normalized(garbage) == [])
        #expect(HostAddresses.normalized(["garbage", "127.0.0.1", "", "::1"]) == ["127.0.0.1", "::1"])
        #expect(HostAddresses.normalized([]) == [])
    }

    @Test func normalizedTrimsSpacesAroundAnAddress() {
        #expect(HostAddresses.normalized([" 10.0.0.1 ", "\t::1"]) == ["10.0.0.1", "::1"])
    }

    @Test func normalizedOrderIsFixedWhateverTheInputOrder() {
        let raw = ["fe80::1", "192.168.1.20", "10.0.0.5", "9.9.9.9", "::1"]
        let expected = ["10.0.0.5", "192.168.1.20", "9.9.9.9", "::1", "fe80::1"]
        #expect(HostAddresses.normalized(raw) == expected)
        #expect(HostAddresses.normalized(raw.reversed()) == expected)
        #expect(HostAddresses.normalized(raw + raw) == expected)
        #expect(HostAddresses.normalized(expected) == expected)
    }

    @Test func normalizedIsIdempotent() {
        let once = HostAddresses.normalized(["FE80::1%en0", "::ffff:10.1.2.3", "10.1.2.3", "junk", "::1"])
        #expect(HostAddresses.normalized(once) == once)
    }

    @Test func canonicalPrintsAddressesTheWayInetNtopDoes() {
        #expect(HostAddresses.canonical("127.0.0.1") == "127.0.0.1")
        #expect(HostAddresses.canonical("0.0.0.0") == "0.0.0.0")
        #expect(HostAddresses.canonical("255.255.255.255") == "255.255.255.255")
        #expect(HostAddresses.canonical("::1") == "::1")
        #expect(HostAddresses.canonical("FE80::1") == "fe80::1")
        #expect(HostAddresses.canonical("2001:0db8:0000:0000:0000:0000:0000:0001") == "2001:db8::1")
    }

    @Test func canonicalReducesAMappedAddress() {
        #expect(HostAddresses.canonical("::ffff:192.168.1.5") == "192.168.1.5")
        #expect(HostAddresses.canonical("::ffff:1.2.3.4") == "1.2.3.4")
        let notMapped = HostAddresses.canonical("::fffe:c0a8:105")
        #expect(notMapped != nil)
        #expect(notMapped != "192.168.1.5")
        #expect(notMapped?.contains(":") == true)
    }

    @Test func canonicalRejectsTextThatIsNotAnAddress() {
        #expect(HostAddresses.canonical("garbage") == nil)
        #expect(HostAddresses.canonical("") == nil)
        #expect(HostAddresses.canonical("1.2.3") == nil)
        #expect(HostAddresses.canonical("256.0.0.1") == nil)
        #expect(HostAddresses.canonical(" 127.0.0.1") == nil)
        // The zone id is removed by `normalized`, not by `canonical`.
        #expect(HostAddresses.canonical("fe80::1%en0") == nil)
    }

    @Test func currentListsTheMacsOwnAddresses() {
        let raw = HostAddresses.current()
        #expect(!raw.isEmpty)
        for entry in raw {
            let withoutZone = entry.split(separator: "%", maxSplits: 1, omittingEmptySubsequences: false)
                .first.map(String.init) ?? entry
            #expect(HostAddresses.canonical(withoutZone) != nil, "\(entry) is not an address")
        }
        let normalized = HostAddresses.normalized(raw)
        #expect(!normalized.isEmpty)
        #expect(normalized.contains("127.0.0.1") || normalized.contains("::1"))
        #expect(normalized.count <= raw.count)
        #expect(normalized == normalized.sorted())
        for entry in normalized {
            #expect(HostAddresses.canonical(entry) == entry)
        }
    }
}

// MARK: LoopbackPort

@Suite struct LoopbackPortTests {
    @Test func findsAPortInTheValidRange() throws {
        let port = try LoopbackPort.findFree()
        #expect((1...65_535).contains(port))
    }

    @Test func thePortIsReallyFree() throws {
        let port = try LoopbackPort.findFree()
        let descriptor = try #require(hostTestBindLoopbackSocket(port: port))
        defer { close(descriptor) }
        // The helper itself can tell a taken port: a second bind must fail.
        let second = hostTestBindLoopbackSocket(port: port)
        if let second { close(second) }
        #expect(second == nil)
    }

    @Test func consecutiveResultsAreBothUsable() throws {
        for _ in 0..<3 {
            let first = try LoopbackPort.findFree()
            let second = try LoopbackPort.findFree()
            #expect((1...65_535).contains(first))
            #expect((1...65_535).contains(second))
            for port in [first, second] {
                let descriptor = try #require(hostTestBindLoopbackSocket(port: port), "port \(port) was not free")
                close(descriptor)
            }
        }
    }

    @Test func aPortThatIsInUseIsNotOffered() throws {
        let taken = try LoopbackPort.findFree()
        let descriptor = try #require(hostTestBindLoopbackSocket(port: taken))
        defer { close(descriptor) }
        for _ in 0..<20 {
            #expect(try LoopbackPort.findFree() != taken)
        }
    }
}

// MARK: SandboxInstanceLock

@Suite struct SandboxInstanceLockTests {
    @Test func theFirstAcquireSucceedsAndCreatesTheFile() throws {
        let directory = try hostTestTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = SandboxLocations.lockFile(in: directory)
        let lock = try #require(SandboxInstanceLock.acquire(at: file))
        #expect(FileManager.default.fileExists(atPath: file.path))
        #expect(try hostTestPermissions(of: file) == 0o600)
        lock.release()
    }

    @Test func aSecondAcquireFailsWhileTheFirstIsHeld() throws {
        let directory = try hostTestTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = SandboxLocations.lockFile(in: directory)
        let first = try #require(SandboxInstanceLock.acquire(at: file))
        defer { first.release() }
        #expect(SandboxInstanceLock.acquire(at: file) == nil)
        #expect(SandboxInstanceLock.acquire(at: file) == nil)
    }

    @Test func afterReleaseTheLockCanBeTakenAgain() throws {
        let directory = try hostTestTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = SandboxLocations.lockFile(in: directory)
        let first = try #require(SandboxInstanceLock.acquire(at: file))
        first.release()
        let second = try #require(SandboxInstanceLock.acquire(at: file))
        #expect(SandboxInstanceLock.acquire(at: file) == nil)
        second.release()
        let third = try #require(SandboxInstanceLock.acquire(at: file))
        third.release()
    }

    @Test func releasingTwiceIsHarmless() throws {
        let directory = try hostTestTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = SandboxLocations.lockFile(in: directory)
        let first = try #require(SandboxInstanceLock.acquire(at: file))
        first.release()
        first.release()
        let second = try #require(SandboxInstanceLock.acquire(at: file))
        // The second release of `first` must not have dropped anyone else's lock.
        first.release()
        #expect(SandboxInstanceLock.acquire(at: file) == nil)
        second.release()
    }

    @Test func dropTheLockWhenTheObjectGoesAway() throws {
        let directory = try hostTestTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = SandboxLocations.lockFile(in: directory)
        do {
            let temporary = try #require(SandboxInstanceLock.acquire(at: file))
            #expect(SandboxInstanceLock.acquire(at: file) == nil)
            withExtendedLifetime(temporary) {}
        }
        let again = try #require(SandboxInstanceLock.acquire(at: file))
        again.release()
    }

    @Test func differentFilesDoNotBlockEachOther() throws {
        let directory = try hostTestTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let one = try #require(SandboxInstanceLock.acquire(at: directory.appendingPathComponent("one.lock")))
        let two = try #require(SandboxInstanceLock.acquire(at: directory.appendingPathComponent("two.lock")))
        one.release()
        two.release()
    }

    @Test func aFileThatCannotBeOpenedGivesNil() throws {
        let directory = try hostTestTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let missing = directory.appendingPathComponent("missing-folder/instance.lock")
        #expect(SandboxInstanceLock.acquire(at: missing) == nil)
    }
}

// MARK: SandboxSecret

@Suite struct SandboxSecretTests {
    private func hexText(of file: URL) throws -> String {
        try String(contentsOf: file, encoding: .utf8)
    }

    @Test func createsAFolderUnderTheBase() throws {
        let directory = try hostTestTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let base = directory.appendingPathComponent("secrets", isDirectory: true)
        let secret = try SandboxSecret.create(under: base)
        #expect(secret.folder.deletingLastPathComponent().path == base.path)
        #expect(UUID(uuidString: secret.folder.lastPathComponent) != nil)
        var isFolder: ObjCBool = false
        let exists = FileManager.default.fileExists(atPath: secret.folder.path, isDirectory: &isFolder)
        #expect(exists)
        #expect(isFolder.boolValue)
    }

    @Test func theFileIsNamedSecretAndHolds64LowercaseHexCharacters() throws {
        let directory = try hostTestTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let secret = try SandboxSecret.create(
            under: directory.appendingPathComponent("secrets", isDirectory: true))
        #expect(SandboxSecret.fileName == "secret")
        let file = secret.folder.appendingPathComponent("secret")
        let text = try hexText(of: file)
        #expect(text.count == 64)
        #expect(text.allSatisfy { "0123456789abcdef".contains($0) })
        #expect(secret.key == Data(text.utf8))
        #expect(secret.key.count == 64)
        #expect(try Data(contentsOf: file) == secret.key)
        // Only the secret is in the folder, and it has no trailing newline.
        let names = try FileManager.default.contentsOfDirectory(atPath: secret.folder.path)
        #expect(names == ["secret"])
    }

    @Test func permissionsAreOwnerOnly() throws {
        let directory = try hostTestTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let base = directory.appendingPathComponent("secrets", isDirectory: true)
        let secret = try SandboxSecret.create(under: base)
        #expect(try hostTestPermissions(of: base) == 0o700)
        #expect(try hostTestPermissions(of: secret.folder) == 0o700)
        #expect(try hostTestPermissions(of: secret.folder.appendingPathComponent("secret")) == 0o600)
    }

    @Test func aBaseThatAlreadyExistsWithOpenPermissionsIsTightened() throws {
        let directory = try hostTestTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let base = directory.appendingPathComponent("secrets", isDirectory: true)
        try FileManager.default.createDirectory(
            at: base, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o755])
        _ = try SandboxSecret.create(under: base)
        #expect(try hostTestPermissions(of: base) == 0o700)
    }

    @Test func everySecretIsNew() throws {
        let directory = try hostTestTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let base = directory.appendingPathComponent("secrets", isDirectory: true)
        let first = try SandboxSecret.create(under: base)
        let second = try SandboxSecret.create(under: base)
        #expect(first.key != second.key)
        #expect(first.folder != second.folder)
        #expect(first.folder.deletingLastPathComponent().path == second.folder.deletingLastPathComponent().path)
        let names = try FileManager.default.contentsOfDirectory(atPath: base.path)
        #expect(names.count == 2)
    }

    @Test func aBaseThatIsAFileThrowsASandboxError() throws {
        let directory = try hostTestTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent("not-a-folder")
        try Data("x".utf8).write(to: file)
        #expect(throws: SandboxError.self) {
            _ = try SandboxSecret.create(under: file)
        }
    }

    @Test func aPathWithACommaOrLineBreakCannotBeMounted() {
        let key = Data("k".utf8)
        let plain = SandboxSecret(folder: URL(fileURLWithPath: "/Users/me/Library/secrets/ABC", isDirectory: true),
                                  key: key)
        let comma = SandboxSecret(folder: URL(fileURLWithPath: "/Users/me, jr/Library/secrets/ABC", isDirectory: true),
                                  key: key)
        let lineBreak = SandboxSecret(folder: URL(fileURLWithPath: "/Users/me\njr/Library/secrets/ABC", isDirectory: true),
                                      key: key)
        #expect(plain.isMountable)
        #expect(!comma.isMountable)
        #expect(!lineBreak.isMountable)
    }

    @Test func aFreshSecretUnderAPlainPathIsMountable() throws {
        let directory = try hostTestTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let secret = try SandboxSecret.create(
            under: directory.appendingPathComponent("secrets", isDirectory: true))
        #expect(secret.isMountable)
    }

    @Test func removeDeletesTheFolderAndLeavesTheBase() throws {
        let directory = try hostTestTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let base = directory.appendingPathComponent("secrets", isDirectory: true)
        let kept = try SandboxSecret.create(under: base)
        let removed = try SandboxSecret.create(under: base)
        removed.remove()
        #expect(!FileManager.default.fileExists(atPath: removed.folder.path))
        #expect(FileManager.default.fileExists(atPath: kept.folder.path))
        #expect(FileManager.default.fileExists(atPath: base.path))
        // Removing again does nothing.
        removed.remove()
    }

    @Test func removeAllClearsEveryLeftoverUnderTheBase() throws {
        let directory = try hostTestTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let base = directory.appendingPathComponent("secrets", isDirectory: true)
        let secrets = try (0..<3).map { _ in try SandboxSecret.create(under: base) }
        try Data("stray".utf8).write(to: base.appendingPathComponent("stray-file"))
        #expect(try FileManager.default.contentsOfDirectory(atPath: base.path).count == 4)
        SandboxSecret.removeAll(under: base)
        #expect(try FileManager.default.contentsOfDirectory(atPath: base.path).isEmpty)
        for secret in secrets {
            #expect(!FileManager.default.fileExists(atPath: secret.folder.path))
        }
        #expect(FileManager.default.fileExists(atPath: base.path))
    }

    @Test func removeAllOnAMissingBaseDoesNothing() throws {
        let directory = try hostTestTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        SandboxSecret.removeAll(under: directory.appendingPathComponent("never-created", isDirectory: true))
        #expect(try FileManager.default.contentsOfDirectory(atPath: directory.path).isEmpty)
    }
}

// MARK: SandboxLocations

@Suite struct SandboxLocationsTests {
    @Test func pathsInsideTheStateDirectory() {
        let state = URL(fileURLWithPath: "/tmp/tuff-state", isDirectory: true)
        #expect(SandboxLocations.secretsDirectory(in: state).path == "/tmp/tuff-state/secrets")
        #expect(SandboxLocations.lockFile(in: state).path == "/tmp/tuff-state/instance.lock")
        #expect(SandboxLocations.secretsDirectory(in: state).lastPathComponent == "secrets")
        #expect(SandboxLocations.lockFile(in: state).lastPathComponent == "instance.lock")
    }

    @Test func theDefaultStateDirectoryIsUnderApplicationSupport() {
        let path = SandboxLocations.defaultStateDirectory().path
        #expect(path.hasSuffix("/Library/Application Support/TUFF/Sandbox"))
    }
}
#endif
