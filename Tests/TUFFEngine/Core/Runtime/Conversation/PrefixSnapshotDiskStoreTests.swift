import Testing
import Foundation
import Metal
@testable import TUFFEngine

/// Shared-prefix snapshots on disk: loaded only for exactly the same model,
/// settings, build and tokens, and never trusted when damaged.
@Suite(.serialized) struct PrefixSnapshotDiskStoreTests {
    let device = MTLCreateSystemDefaultDevice()!
    final class Owner: StateSnapshottingRunner {
        var stateSnapshotByteEstimate: Int? { nil }
        func captureState() throws -> RunnerStateSnapshot { throw RunnerStateSnapshotError.unsupported("") }
        func restoreState(_ snapshot: RunnerStateSnapshot) throws {}
    }

    static let domain = ConversationCacheDomain(
        modelID: "gpt-oss-20b", sourceSnapshotHash: "s", runtimeProfileHash: "r",
        maximumContext: 4_096, kvStorage: "fp16", fp16RingEnabled: true, templateSHA256: "t")

    func store(budget: Int = 1 << 20, build: String = "8.3.4") -> (PrefixSnapshotDiskStore, URL) {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-prefix-\(UUID().uuidString)")
        return (PrefixSnapshotDiskStore(directory: directory, budgetBytes: budget,
                                        buildIdentity: build), directory)
    }

    func snapshot(tokens: [Int32], bytes: Int = 4_096, owner: AnyObject) -> RunnerStateSnapshot {
        let buffer = device.makeBuffer(length: bytes, options: .storageModeShared)!
        let pointer = buffer.contents().bindMemory(to: UInt8.self, capacity: bytes)
        for index in 0..<bytes { pointer[index] = UInt8(truncatingIfNeeded: index * 7) }
        return RunnerStateSnapshot(
            owner: ObjectIdentifier(owner), storage: buffer,
            segments: [.init(label: "kv.K.0", sourceOffset: 0, snapshotOffset: 0, length: bytes)],
            host: .init(position: tokens.count, ngramContext: [3, 4], ropeDelta: 0))
    }

    @Test func aSavedSnapshotLoadsBackExactly() throws {
        let (store, directory) = store()
        defer { try? FileManager.default.removeItem(at: directory) }
        let owner = Owner(), next = Owner()
        let tokens: [Int32] = Array(1...300)
        let saved = snapshot(tokens: tokens, owner: owner)
        store.save(saved, tokens: tokens, domain: Self.domain)
        store.flush()
        #expect(store.contains(tokens: tokens, domain: Self.domain))

        let loaded = try #require(store.load(tokens: tokens, domain: Self.domain,
                                             runner: next, device: device))
        #expect(loaded.owner == next.snapshotOwner)
        #expect(loaded.position == 300)
        #expect(loaded.host.ngramContext == [3, 4])
        #expect(loaded.segments.map(\.label) == ["kv.K.0"])
        let a = Data(bytes: saved.storage!.contents(), count: 4_096)
        let b = Data(bytes: loaded.storage!.contents(), count: 4_096)
        #expect(a == b)
    }

    @Test func anythingDifferentLoadsNothing() throws {
        let (store, directory) = store()
        defer { try? FileManager.default.removeItem(at: directory) }
        let owner = Owner()
        let tokens: [Int32] = Array(1...300)
        store.save(snapshot(tokens: tokens, owner: owner), tokens: tokens, domain: Self.domain)
        store.flush()
        var other = tokens
        other[150] = 9_999
        #expect(store.load(tokens: other, domain: Self.domain, runner: owner, device: device) == nil)
        let otherModel = ConversationCacheDomain(
            modelID: "gpt-oss-120b", sourceSnapshotHash: "s", runtimeProfileHash: "r",
            maximumContext: 4_096, kvStorage: "fp16", fp16RingEnabled: true, templateSHA256: "t")
        #expect(store.load(tokens: tokens, domain: otherModel, runner: owner, device: device) == nil)
        let newerBuild = PrefixSnapshotDiskStore(directory: directory, budgetBytes: 1 << 20,
                                                 buildIdentity: "8.3.5")
        #expect(newerBuild.load(tokens: tokens, domain: Self.domain, runner: owner,
                                device: device) == nil)
        // The original still loads.
        #expect(store.load(tokens: tokens, domain: Self.domain, runner: owner, device: device) != nil)
    }

    @Test func aDamagedFileIsRemovedNotUsed() throws {
        let (store, directory) = store()
        defer { try? FileManager.default.removeItem(at: directory) }
        let owner = Owner()
        let tokens: [Int32] = Array(1...300)
        store.save(snapshot(tokens: tokens, owner: owner), tokens: tokens, domain: Self.domain)
        store.flush()
        let file = try #require(try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil).first)
        var bytes = try Data(contentsOf: file)
        bytes[bytes.count - 10] ^= 0xFF
        try bytes.write(to: file)
        #expect(store.load(tokens: tokens, domain: Self.domain, runner: owner, device: device) == nil)
        #expect(!FileManager.default.fileExists(atPath: file.path))
    }

    @Test func theOldestFilesGoWhenTheBudgetIsFull() throws {
        let (store, directory) = store(budget: 10_000)
        defer { try? FileManager.default.removeItem(at: directory) }
        let owner = Owner()
        let first: [Int32] = Array(1...300), second: [Int32] = Array(2...301),
            third: [Int32] = Array(3...302)
        for tokens in [first, second, third] {
            store.save(snapshot(tokens: tokens, owner: owner), tokens: tokens, domain: Self.domain)
            store.flush()
            Thread.sleep(forTimeInterval: 0.02)
        }
        #expect(!store.contains(tokens: first, domain: Self.domain))
        #expect(store.contains(tokens: third, domain: Self.domain))
    }

    @Test func anUpdateClearsTheOldBuildsFolder() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-prefix-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        let owner = Owner()
        let tokens: [Int32] = Array(1...300)
        let old = PrefixSnapshotDiskStore(directory: root.appendingPathComponent("8.3.3"),
                                          budgetBytes: 1 << 20, buildIdentity: "8.3.3",
                                          clearsSiblingBuilds: true)
        old.save(snapshot(tokens: tokens, owner: owner), tokens: tokens, domain: Self.domain)
        old.flush()
        #expect(FileManager.default.fileExists(atPath: root.appendingPathComponent("8.3.3").path))
        let new = PrefixSnapshotDiskStore(directory: root.appendingPathComponent("8.3.4"),
                                          budgetBytes: 1 << 20, buildIdentity: "8.3.4",
                                          clearsSiblingBuilds: true)
        new.save(snapshot(tokens: tokens, owner: owner), tokens: tokens, domain: Self.domain)
        new.flush()
        #expect(!FileManager.default.fileExists(atPath: root.appendingPathComponent("8.3.3").path))
        #expect(new.contains(tokens: tokens, domain: Self.domain))
    }
}
