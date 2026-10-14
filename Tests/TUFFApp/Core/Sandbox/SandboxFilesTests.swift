#if os(macOS)
import Foundation
import Testing
import TUFFEngine
import TUFFModelCatalog
@testable import TUFFAppCore

private let folderOne = "11111111-2222-3333-4444-555555555555"
private let folderTwo = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

private func mount(_ id: String = folderOne, _ path: String = "/Users/test/Notes") -> SandboxFolderMount {
    SandboxFolderMount(id: id, path: path)
}

// MARK: The files profile of the controller

@MainActor
@Suite struct SandboxFilesProfileTests {
    @Test func theFilesLaunchHasNoNetworkAndMountsEachFolderReadOnly() {
        let launch = SandboxLaunch(
            name: "tuff-sandbox-files-00000001", image: "img@sha256:aa",
            secretFolder: URL(fileURLWithPath: "/tmp/secret-folder"), blockedAddresses: [], port: 51_000,
            profile: .files, mounts: [mount(), mount(folderTwo, "/Users/test/Other Folder")])
        #expect(launch.arguments == [
            "--detach", "--name", "tuff-sandbox-files-00000001",
            "--read-only", "--tmpfs", "/tmp",
            "--cap-drop", "ALL",
            "--cap-add", "NET_ADMIN", "--cap-add", "SETUID",
            "--cap-add", "SETGID", "--cap-add", "SETPCAP",
            "--ulimit", "nproc=512", "--cpus", "2", "--memory", "1G",
            "--no-dns",
            "--mount", "type=bind,source=/tmp/secret-folder,target=/run/tuff-sandbox,readonly",
            "--mount", "type=bind,source=/Users/test/Notes,target=/files/\(folderOne),readonly",
            "--mount", "type=bind,source=/Users/test/Other Folder,target=/files/\(folderTwo),readonly",
            "--env", "SANDBOX_PROFILE=files",
            "--publish", "127.0.0.1:51000:9100",
            "img@sha256:aa",
        ])
    }

    @Test func theWebLaunchIsExactlyWhatItWasBeforeFilesExisted() {
        let launch = SandboxLaunch(
            name: "tuff-sandbox-web-00000001", image: "img@sha256:aa",
            secretFolder: URL(fileURLWithPath: "/tmp/secret-folder"),
            blockedAddresses: ["10.0.0.1", "::1"], port: 51_000)
        #expect(launch.arguments == [
            "--detach", "--name", "tuff-sandbox-web-00000001",
            "--read-only", "--tmpfs", "/tmp",
            "--cap-drop", "ALL",
            "--cap-add", "NET_ADMIN", "--cap-add", "SETUID",
            "--cap-add", "SETGID", "--cap-add", "SETPCAP",
            "--ulimit", "nproc=512", "--cpus", "2", "--memory", "1G",
            "--mount", "type=bind,source=/tmp/secret-folder,target=/run/tuff-sandbox,readonly",
            "--env", "SANDBOX_BLOCKED_ADDRESSES=10.0.0.1,::1",
            "--publish", "127.0.0.1:51000:9100",
            "img@sha256:aa",
        ])
    }

    @Test func mountablePaths() {
        #expect(SandboxFolderMount.isMountable("/Users/test/My Notes"))
        #expect(!SandboxFolderMount.isMountable("/Users/test/a,b"))
        #expect(!SandboxFolderMount.isMountable("/Users/test/a\nb"))
        #expect(!SandboxFolderMount.isMountable("/Users/test/a\rb"))
        #expect(!SandboxFolderMount.isMountable("Users/test"))
        #expect(!SandboxFolderMount.isMountable("/Users/\u{0}test"))
    }

    @Test func folderIDsAreLowercaseUUIDs() {
        #expect(SandboxFolderMount.isFolderID(folderOne))
        // Uppercasing changes only letters, which folderOne does not have.
        #expect(SandboxFolderMount.isFolderID("abcdef01-2222-3333-4444-555555555555"))
        #expect(!SandboxFolderMount.isFolderID("ABCDEF01-2222-3333-4444-555555555555"))
        #expect(!SandboxFolderMount.isFolderID("../etc"))
        #expect(!SandboxFolderMount.isFolderID(folderOne + "x"))
    }

    @Test func aFilesSandboxStartsWithTheFoldersAndChecksItself() async throws {
        let h = SandboxHarness.make(profile: .files)
        defer { h.cleanUp() }
        h.controller.folderMounts = [mount(), mount(folderTwo, "/Users/test/Other")]
        await h.controller.start()
        #expect(h.controller.state == .ready)
        #expect(h.controller.activeMountIDs == [folderOne, folderTwo])
        let arguments = try #require(h.world.lastRunArguments)
        #expect(arguments.contains("--no-dns"))
        #expect(arguments.contains("SANDBOX_PROFILE=files"))
        #expect(!arguments.contains { $0.hasPrefix("SANDBOX_BLOCKED_ADDRESSES") })
        let name = try #require(h.world.runningContainers.first)
        #expect(name.hasPrefix("tuff-sandbox-files-"))
        #expect(h.world.count("exec nft") == 1)
    }

    @Test func aFilesStartWithNoFolderFailsBeforeAnythingRuns() async {
        let h = SandboxHarness.make(profile: .files)
        defer { h.cleanUp() }
        await h.controller.start()
        #expect(sandboxFailureText(h.controller.state) == SandboxError.noFoldersToMount.description)
        #expect(h.world.count("run") == 0)
        #expect(h.secretFolders.isEmpty)
    }

    @Test func aFolderThatCannotBeMountedFailsTheStart() async {
        let h = SandboxHarness.make(profile: .files)
        defer { h.cleanUp() }
        h.controller.folderMounts = [mount(folderOne, "/Users/test/a,b")]
        await h.controller.start()
        #expect(sandboxFailureText(h.controller.state) == SandboxError.unmountableFolder.description)
        #expect(h.world.count("run") == 0)
    }

    @Test func macOSRefusingAFolderNamesWhereToAllowIt() async {
        let h = SandboxHarness.make(profile: .files)
        defer { h.cleanUp() }
        h.world.failures["run"] = SandboxProcessResult(
            status: 1, output: "Error: unable to open /Users/test/Documents: Operation not permitted\n")
        h.controller.folderMounts = [mount()]
        await h.controller.start()
        let text = sandboxFailureText(h.controller.state) ?? ""
        #expect(text.contains("container-runtime-linux"))
        #expect(text.contains("Files and Folders"))
        #expect(h.world.runningContainers.isEmpty)
        #expect(h.secretFolders.isEmpty)
    }

    @Test func aRefusalIsOnlyReadIntoAFailedStartWithFolders() async {
        let h = SandboxHarness.make()
        defer { h.cleanUp() }
        h.world.failures["run"] = SandboxProcessResult(status: 1, output: "Operation not permitted\n")
        await h.controller.start()
        #expect(sandboxFailureText(h.controller.state)?.contains("container-runtime-linux") == false)
    }

    @Test func aFilesSandboxWithAWritableMountIsNotUsed() async {
        let h = SandboxHarness.make(profile: .files)
        defer { h.cleanUp() }
        h.world.mountOutputOverride = "/files/\(folderOne) rw\n"
        h.controller.folderMounts = [mount()]
        await h.controller.start()
        #expect(sandboxFailureText(h.controller.state)?.contains("writable") == true)
        #expect(h.world.runningContainers.isEmpty)
    }

    @Test func aFilesSandboxWithOtherFoldersThanGivenIsNotUsed() async {
        let h = SandboxHarness.make(profile: .files)
        defer { h.cleanUp() }
        h.world.mountOutputOverride = "/files/\(folderOne) ro\n/files/\(folderTwo) ro\n"
        h.controller.folderMounts = [mount()]
        await h.controller.start()
        #expect(sandboxFailureText(h.controller.state)?.contains("not the ones it was given") == true)
    }

    @Test func aFilesSandboxThatLooksAtOtherFoldersIsNotUsed() async {
        let h = SandboxHarness.make(profile: .files)
        defer { h.cleanUp() }
        h.probe.handler = { _, _, _ in
            SandboxHealth(profile: "files", api: 1, blockedAddresses: 0, folders: [folderTwo])
        }
        h.controller.folderMounts = [mount()]
        await h.controller.start()
        #expect(sandboxFailureText(h.controller.state)?.contains("other folders") == true)
    }

    @Test func aFilesSandboxWithAnOpenFirewallIsNotUsed() async {
        let h = SandboxHarness.make(profile: .files)
        defer { h.cleanUp() }
        h.world.nftListingOverride = FakeContainerWorld.goodFilesNftListing()
            .replacingOccurrences(of: "ct state established,related accept\n\t}\n\n\tchain input",
                                  with: "ct state established,related accept\n\t\tmeta nfproto ipv4 accept\n\t}\n\n\tchain input")
        h.controller.folderMounts = [mount()]
        await h.controller.start()
        #expect(sandboxFailureText(h.controller.state)?.contains("lets traffic out") == true)
        #expect(h.world.runningContainers.isEmpty)
    }

    @Test func theWebAndTheFilesSandboxRunSideBySideWithoutTouchingEachOther() async {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-sandbox-test-\(UUID().uuidString)", isDirectory: true)
        let web = SandboxHarness.make(stateDirectory: directory)
        let files = SandboxHarness.make(world: web.world, stateDirectory: directory, profile: .files)
        defer { web.cleanUp() }
        files.controller.folderMounts = [mount()]
        await web.controller.start()
        await files.controller.start()
        #expect(web.controller.state == .ready)
        #expect(files.controller.state == .ready)
        #expect(web.world.runningContainers.count == 2)
        // Each has its own secret folder.
        #expect(web.secretFolders.count == 1)
        #expect(files.secretFolders.count == 1)

        await files.controller.stop()
        #expect(files.controller.state == .off)
        #expect(web.controller.state == .ready)
        #expect(web.world.runningContainers.count == 1)
        #expect(web.world.runningContainers.allSatisfy { $0.hasPrefix("tuff-sandbox-web-") })
        #expect(web.secretFolders.count == 1)
        #expect(files.secretFolders.isEmpty)
        await web.controller.stop()
    }

    @Test func aFilesStartRemovesItsOwnLeftoversAndLeavesTheWebSandboxAlone() async {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-sandbox-test-\(UUID().uuidString)", isDirectory: true)
        let files = SandboxHarness.make(stateDirectory: directory, profile: .files)
        defer { files.cleanUp() }
        files.world.addContainer("tuff-sandbox-web-deadbeef", running: true)
        files.world.addContainer("tuff-sandbox-files-cafe0001", running: false)
        files.controller.folderMounts = [mount()]
        await files.controller.start()
        #expect(files.world.runningContainers.contains("tuff-sandbox-web-deadbeef"))
        #expect(!files.world.stoppedContainers.contains("tuff-sandbox-files-cafe0001"))
        #expect(files.world.runningContainers.contains { $0.hasPrefix("tuff-sandbox-files-") })
    }

    @Test func ownContainerNamesAreTheProfilesPrefixAndEightHexDigits() {
        #expect(SandboxController.isOurContainer("tuff-sandbox-files-0a1b2c3d", profile: .files))
        #expect(!SandboxController.isOurContainer("tuff-sandbox-files-0a1b2c3d"))
        #expect(!SandboxController.isOurContainer("tuff-sandbox-web-0a1b2c3d", profile: .files))
        #expect(!SandboxController.isOurContainer("tuff-sandbox-files-0a1b2c3d0", profile: .files))
        #expect(!SandboxController.isOurContainer("tuff-sandbox-files-selftest", profile: .files))
    }

    @Test func theTwoProfilesHaveSeparateLocksAndSecretFolders() {
        let state = URL(fileURLWithPath: "/tmp/tuff-state")
        #expect(SandboxLocations.lockFile(in: state).lastPathComponent == "instance.lock")
        #expect(SandboxLocations.lockFile(in: state, profile: .files).lastPathComponent == "instance-files.lock")
        #expect(SandboxLocations.secretsDirectory(in: state).lastPathComponent == "secrets")
        #expect(SandboxLocations.secretsDirectory(in: state, profile: .files).lastPathComponent == "secrets-files")
    }
}

// MARK: The checks of the files sandbox

@Suite struct SandboxFilesProtectionTests {
    @Test func theExpectedFilesFirewallPasses() {
        #expect(SandboxProtection.filesFirewallProblem(FakeContainerWorld.goodFilesNftListing()) == nil)
    }

    @Test func aWebTableIsNotAFilesTable() {
        #expect(SandboxProtection.filesFirewallProblem(
            FakeContainerWorld.goodNftListing(blocked: ["10.0.0.1"])) != nil)
    }

    @Test(arguments: [
        ("policy drop;\n\t\toifname", "policy accept;\n\t\toifname"),
        ("ct state established,related accept\n\t}\n\n\tchain input",
         "ct state established,related accept\n\t\taccept\n\t}\n\n\tchain input"),
        ("ct state established,related accept\n\t}\n\n\tchain input",
         "ct state established,related accept\n\t\tip daddr 1.1.1.1 tcp dport 443 accept\n\t}\n\n\tchain input"),
        ("ct state established,related accept\n\t}\n\n\tchain input",
         "ct state established,related accept\n\t\tjump other\n\t}\n\n\tchain input"),
        ("hook input priority filter; policy drop;", "hook input priority filter; policy accept;"),
        ("ip saddr 192.168.64.1 tcp dport 9100 accept",
         "ip saddr 192.168.64.1 tcp dport 9100 accept\n\t\ttcp dport 22 accept"),
        ("ip saddr 192.168.64.1 tcp dport 9100 accept", "tcp dport 9100 accept"),
    ])
    func aWeakerFirewallIsRefused(original: String, replacement: String) {
        let listing = FakeContainerWorld.goodFilesNftListing()
        #expect(listing.contains(original))
        #expect(SandboxProtection.filesFirewallProblem(
            listing.replacingOccurrences(of: original, with: replacement)) != nil)
    }

    @Test func mountsMustBeExactlyTheGivenOnesAndReadOnly() {
        let wanted = [mount(), mount(folderTwo, "/Users/test/Other")]
        let good = "/files/\(folderOne) ro\n/files/\(folderTwo) ro\n"
        #expect(SandboxProtection.mountProblem(good, status: 0, mounts: wanted) == nil)
        #expect(SandboxProtection.mountProblem(good, status: 1, mounts: wanted) != nil)
        #expect(SandboxProtection.mountProblem("/files/\(folderOne) ro\n", status: 0, mounts: wanted) != nil)
        #expect(SandboxProtection.mountProblem(good + "/files/extra ro\n", status: 0, mounts: wanted) != nil)
        #expect(SandboxProtection.mountProblem(
            "/files/\(folderOne) ro\n/files/\(folderTwo) rw\n", status: 0, mounts: wanted) != nil)
        #expect(SandboxProtection.mountProblem(good + good, status: 0, mounts: wanted) != nil)
        #expect(SandboxProtection.mountProblem("", status: 0, mounts: wanted) != nil)
    }

    @Test func theMountCheckReadsProcMounts() {
        #expect(SandboxProtection.mountCheck.contains("/proc/mounts"))
        #expect(SandboxProtection.mountCheck.contains("/files/"))
    }

    @Test func theHealthProbeOfAFilesSandboxWantsThatProfile() throws {
        let key = Data("k".utf8)
        let nonce = String(repeating: "a", count: 64)
        func answer(_ body: String, profile: SandboxProfile) throws -> SandboxHealth {
            let data = Data(body.utf8)
            let signature = SandboxSigning.responseAuth(
                key: key, method: "GET", path: "/health", nonce: nonce, status: 200, body: data)
            return try SandboxHealthProbe.check(status: 200, signature: signature, body: data,
                                                nonce: nonce, key: key, profile: profile)
        }
        let files = "{\"status\":\"ok\",\"profile\":\"files\",\"api\":1,\"pdf\":true,"
            + "\"blocked_addresses\":0,\"folders\":[\"\(folderOne)\"]}"
        let web = "{\"status\":\"ok\",\"profile\":\"web\",\"api\":1,\"blocked_addresses\":3}"
        #expect(try answer(files, profile: .files)
            == SandboxHealth(profile: "files", api: 1, blockedAddresses: 0, folders: [folderOne]))
        #expect(try answer(web, profile: .web) == SandboxHealth(profile: "web", api: 1, blockedAddresses: 3))
        #expect(throws: SandboxError.self) { _ = try answer(web, profile: .files) }
        #expect(throws: SandboxError.self) { _ = try answer(files, profile: .web) }
        let badFolder = files.replacingOccurrences(of: folderOne, with: "../..")
        #expect(throws: SandboxError.self) { _ = try answer(badFolder, profile: .files) }
    }
}

// MARK: The reader

@MainActor
private struct FilesHarness {
    let sandbox: SandboxHarness
    let reader: SandboxFileReader
    let folder: AppSearchFolder
    let base: URL

    var root: URL { URL(fileURLWithPath: folder.path, isDirectory: true) }
    var id: String { folder.id.uuidString.lowercased() }

    static func ready(files: [String] = [],
                      handler: @escaping @Sendable (SandboxStubProtocol.Seen) -> SandboxStubProtocol.Reply)
        async throws -> FilesHarness {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-sandbox-files-\(UUID().uuidString)")
        let root = base.appendingPathComponent("Notes")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        for path in files {
            let url = root.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            try "text".write(to: url, atomically: true, encoding: .utf8)
        }
        let folder = try AppSearchFolderStore.folder(for: root)
        let sandbox = SandboxHarness.make(freePort: SandboxStubProtocol.port, profile: .files)
        sandbox.controller.folderMounts = SandboxFilesService.mounts(for: [folder])
        await sandbox.controller.start()
        try #require(sandbox.controller.state == .ready)
        let lease = try sandbox.controller.acquire()
        let key = lease.key
        sandbox.controller.release(lease)
        SandboxStubProtocol.install(key: key, handler: handler)
        let access = SandboxWebHarness.access(for: sandbox.controller)
        return FilesHarness(sandbox: sandbox, reader: SandboxFileReader(access: access),
                            folder: folder, base: base)
    }

    func cleanUp() {
        sandbox.cleanUp()
        try? FileManager.default.removeItem(at: base)
    }

    func found(_ relative: String, size: Int = 4, mtime: Double = 1_700_000_000.5) -> AppLocalIndexer.FoundFile {
        AppLocalIndexer.FoundFile(
            path: (canonicalPath(folder.path) ?? folder.path) + "/" + relative, size: size,
            mtime: Date(timeIntervalSince1970: mtime).timeIntervalSinceReferenceDate)
    }
}

private final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0
    func next() -> Int { lock.withLock { value += 1; return value } }
}

private func listReply(_ entries: [(String, Int, Double)], next: String? = nil) -> SandboxStubProtocol.Reply {
    SandboxStubProtocol.json([
        "entries": entries.map { ["path": $0.0, "size": $0.1, "mtime": $0.2] as [String: Any] },
        "next_cursor": next.map { $0 as Any } ?? NSNull(),
    ])
}

private func passageReply(size: Int = 4, mtime: Double = 1_700_000_000.5,
                          _ passages: [[String: Any]]) -> SandboxStubProtocol.Reply {
    SandboxStubProtocol.json(["size": size, "mtime": mtime, "passages": passages])
}

private func linePassage(_ text: String, _ start: Int = 1, _ end: Int = 1) -> [String: Any] {
    ["page": NSNull(), "line_start": start, "line_end": end, "text": text]
}

extension SandboxStubTests {
    @MainActor
    @Suite struct SandboxFileReaderScanTests {
        @Test func theListIsReadPageByPageAndKeepsTheAppsPathsAndStamps() async throws {
            let h = try await FilesHarness.ready { seen in
                let cursor = seen.json["cursor"] as? String
                if cursor == nil {
                    return listReply([("a.md", 10, 1_700_000_000.25), ("sub/b.txt", 20, 1_700_000_100)],
                                     next: "sub/b.txt")
                }
                return listReply([("sub/c.swift", 30, 1_700_000_200)])
            }
            defer { h.cleanUp() }
            let result = try await h.reader.scan(h.folder, limits: .standard)
            #expect(result.reachedLimit == false)
            let root = canonicalPath(h.folder.path) ?? h.folder.path
            #expect(result.files.map(\.path) == [root + "/a.md", root + "/sub/b.txt", root + "/sub/c.swift"])
            #expect(result.files.map(\.size) == [10, 20, 30])
            let reference = Date(timeIntervalSince1970: 1_700_000_000.25).timeIntervalSinceReferenceDate
            #expect(abs(result.files[0].mtime - reference) < 0.0001)

            let seen = SandboxStubProtocol.seen
            #expect(seen.count == 2)
            #expect(seen.allSatisfy { $0.path == "/v1/files/list" })
            #expect(seen[0].json["folder"] as? String == h.id)
            #expect(seen[0].json["cursor"] == nil)
            #expect(seen[1].json["cursor"] as? String == "sub/b.txt")
            #expect(seen[0].json["limit"] as? Int == SandboxFileReader.listPageSize)
        }

        @Test func theAppAppliesItsOwnRulesToWhatTheSandboxListed() async throws {
            let tooDeep = (1...26).map { "d\($0)" }.joined(separator: "/") + "/f.txt"
            let atLimit = (1...24).map { "d\($0)" }.joined(separator: "/") + "/f.txt"
            let h = try await FilesHarness.ready(files: ["Thing.app/inside.txt"]) { _ in
                listReply([
                    ("good.md", 1, 1_700_000_000), ("README", 1, 1_700_000_000),
                    ("Makefile", 1, 1_700_000_000), (".hidden.md", 1, 1_700_000_000),
                    (".git/config.txt", 1, 1_700_000_000), ("node_modules/x/y.js", 1, 1_700_000_000),
                    ("a/.secret/n.txt", 1, 1_700_000_000), ("pic.png", 1, 1_700_000_000),
                    ("../escape.txt", 1, 1_700_000_000), ("/etc/passwd.txt", 1, 1_700_000_000),
                    ("a//b.txt", 1, 1_700_000_000), ("a/./b.txt", 1, 1_700_000_000),
                    ("bad\u{1}name.txt", 1, 1_700_000_000), ("neg.txt", -1, 1_700_000_000),
                    ("good.md", 2, 1_700_000_000), (tooDeep, 1, 1_700_000_000),
                    (atLimit, 1, 1_700_000_000), ("Thing.app/inside.txt", 1, 1_700_000_000),
                    ("", 1, 1_700_000_000),
                ])
            }
            defer { h.cleanUp() }
            let result = try await h.reader.scan(h.folder, limits: .standard)
            let root = (canonicalPath(h.folder.path) ?? h.folder.path) + "/"
            let names = result.files.map { String($0.path.dropFirst(root.count)) }
            #expect(names == ["good.md", "README", atLimit])
            #expect(result.files.first?.size == 1)
        }

        @Test func aListThatDoesNotMoveForwardIsRefused() async throws {
            let h = try await FilesHarness.ready { _ in listReply([("a.md", 1, 1_700_000_000)], next: "a.md") }
            defer { h.cleanUp() }
            await #expect(throws: AppLocalSearchError.self) {
                _ = try await h.reader.scan(h.folder, limits: .standard)
            }
            #expect(SandboxStubProtocol.seen.count == 2)
        }

        @Test func aShortOrEmptyPageWithAnUnlistedCursorIsFollowed() async throws {
            let counter = Counter()
            let h = try await FilesHarness.ready { _ in
                switch counter.next() {
                case 1: listReply([], next: ".cache/odd\u{1}name")
                case 2: listReply([("a.md", 1, 1_700_000_000)], next: "a.md")
                default: listReply([("b.md", 1, 1_700_000_000)])
                }
            }
            defer { h.cleanUp() }
            let result = try await h.reader.scan(h.folder, limits: .standard)
            let root = (canonicalPath(h.folder.path) ?? h.folder.path) + "/"
            #expect(result.files.map { String($0.path.dropFirst(root.count)) } == ["a.md", "b.md"])
            let seen = SandboxStubProtocol.seen
            #expect(seen.count == 3)
            #expect(seen[1].json["cursor"] as? String == ".cache/odd\u{1}name")
        }

        @Test func aCursorThatIsNotAPathIsRefused() async throws {
            let h = try await FilesHarness.ready { _ in listReply([("a.md", 1, 1_700_000_000)], next: "../x") }
            defer { h.cleanUp() }
            await #expect(throws: AppLocalSearchError.self) {
                _ = try await h.reader.scan(h.folder, limits: .standard)
            }
        }

        @Test func runningOutOfPagesKeepsWhatWasFoundAndSaysTheLimitWasReached() async throws {
            let counter = Counter()
            let h = try await FilesHarness.ready { _ in
                let n = counter.next()
                return listReply([("f\(n).txt", 1, 1_700_000_000)], next: "f\(n).txt")
            }
            defer { h.cleanUp() }
            let result = try await h.reader.scan(h.folder, limits: .standard)
            #expect(result.reachedLimit)
            #expect(result.files.count == SandboxFileReader.maximumListPages)
        }

        @Test func anEmptyListWithUnreadableDirectoriesIsNotBelieved() async throws {
            let h = try await FilesHarness.ready(files: ["Projects/2024/deep.pdf"]) { _ in
                SandboxStubProtocol.json(["entries": [], "next_cursor": NSNull(), "unreadable": 2])
            }
            defer { h.cleanUp() }
            do {
                _ = try await h.reader.scan(h.folder, limits: .standard)
                Issue.record("expected an error")
            } catch {
                #expect("\(error)".contains("container-runtime-linux"))
            }
        }

        @Test func unreadableDirectoriesDoNotHideTheFilesThatWereListed() async throws {
            let h = try await FilesHarness.ready { _ in
                SandboxStubProtocol.json([
                    "entries": [["path": "a.md", "size": 1, "mtime": 1_700_000_000.0]],
                    "next_cursor": NSNull(), "unreadable": 1])
            }
            defer { h.cleanUp() }
            let result = try await h.reader.scan(h.folder, limits: .standard)
            #expect(result.files.count == 1)
        }

        @Test func aNegativeUnreadableCountIsRefused() async throws {
            let h = try await FilesHarness.ready { _ in
                SandboxStubProtocol.json(["entries": [], "next_cursor": NSNull(), "unreadable": -1])
            }
            defer { h.cleanUp() }
            await #expect(throws: AppLocalSearchError.self) {
                _ = try await h.reader.scan(h.folder, limits: .standard)
            }
        }

        @Test func aFolderTheSandboxCannotReadNamesWhereToAllowIt() async throws {
            let h = try await FilesHarness.ready { _ in
                SandboxStubProtocol.failure("folder_unreadable", "the folder cannot be read", status: 403)
            }
            defer { h.cleanUp() }
            do {
                _ = try await h.reader.scan(h.folder, limits: .standard)
                Issue.record("expected an error")
            } catch {
                #expect("\(error)".contains("container-runtime-linux"))
            }
        }

        @Test func aFolderThatReallyIsEmptyIsBelieved() async throws {
            let h = try await FilesHarness.ready { _ in listReply([]) }
            defer { h.cleanUp() }
            let result = try await h.reader.scan(h.folder, limits: .standard)
            #expect(result.files.isEmpty)
        }

        @Test func linksAndHiddenDirectoriesTheSandboxListedAreNotFollowedOnTheMac() async throws {
            let h = try await FilesHarness.ready(files: ["real/inside.md"]) { _ in
                listReply([("link/inside.md", 1, 1_700_000_000), ("real/inside.md", 1, 1_700_000_000)])
            }
            defer { h.cleanUp() }
            try FileManager.default.createSymbolicLink(
                at: h.root.appendingPathComponent("link"),
                withDestinationURL: h.root.appendingPathComponent("real"))
            let result = try await h.reader.scan(h.folder, limits: .standard)
            let root = (canonicalPath(h.folder.path) ?? h.folder.path) + "/"
            #expect(result.files.map { String($0.path.dropFirst(root.count)) } == ["real/inside.md"])
        }

        @Test func theHomeFolderAndFoldersAboveItAreTooBroad() {
            #expect(SandboxFolderMount.isTooBroad("/Users/test", home: "/Users/test"))
            #expect(SandboxFolderMount.isTooBroad("/Users", home: "/Users/test"))
            // A real folder is its own home, and the folder above it contains it.
            let real = FileManager.default.temporaryDirectory.path
            #expect(SandboxFolderMount.isTooBroad(real, home: real))
            #expect(SandboxFolderMount.isTooBroad(
                FileManager.default.temporaryDirectory.deletingLastPathComponent().path, home: real))
            #expect(!SandboxFolderMount.isTooBroad("/Users/test/Documents", home: "/Users/test"))
            #expect(!SandboxFolderMount.isTooBroad("/Users/tester", home: "/Users/test"))
            #expect(!SandboxFolderMount.isTooBroad("/Volumes/Data", home: "/Users/test"))
        }

        @Test func theSSHKeysFolderAndWhatIsInsideItAreNeverMounted() {
            let home = "/Users/test"
            #expect(SandboxFolderMount.isKeyFolder("/Users/test/.ssh", home: home))
            #expect(SandboxFolderMount.isKeyFolder("/Users/test/.ssh/", home: home))
            #expect(SandboxFolderMount.isKeyFolder("/Users/test/.ssh/keys/old", home: home))
            // Other folders in the home folder, also sensitive ones, stay allowed.
            #expect(!SandboxFolderMount.isKeyFolder("/Users/test/Library", home: home))
            #expect(!SandboxFolderMount.isKeyFolder("/Users/test/.sshx", home: home))
            #expect(!SandboxFolderMount.isKeyFolder("/Users/test/Documents/.ssh-notes", home: home))
        }

        @Test func aLinkToTheSSHKeysFolderIsRecognisedByWhatItIs() throws {
            let home = FileManager.default.temporaryDirectory
                .appendingPathComponent("tuff files home \(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(
                at: home.appendingPathComponent(".ssh/inner"), withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: home) }
            let realHome = canonicalPath(home.path) ?? home.path
            #expect(SandboxFolderMount.isKeyFolder(realHome + "/.ssh/inner", home: realHome))
            let link = home.appendingPathComponent("elsewhere")
            try FileManager.default.createSymbolicLink(
                at: link, withDestinationURL: home.appendingPathComponent(".ssh"))
            // `canonicalPath` resolves links, so the sandbox sees the real path; a path
            // that still names the keys folder through a link is caught by identity.
            #expect(SandboxFolderMount.isKeyFolder(link.path, home: realHome))
        }

        @Test(.enabled(if: FileManager.default.fileExists(atPath: "/System/Volumes/Data")))
        func theDataVolumeRootIsTooBroadUnderItsOtherName() {
            // The firmlinked data volume holds /Users, so it holds the home
            // folder, although the home folder's path does not run through it.
            #expect(SandboxFolderMount.isTooBroad("/System/Volumes/Data", home: NSHomeDirectory()))
        }

        @Test func aFolderWithOnlyPackageContentsIsNotTakenForAnUnreadableOne() throws {
            let base = FileManager.default.temporaryDirectory
                .appendingPathComponent("tuff files package \(UUID().uuidString)", isDirectory: true)
            defer { try? FileManager.default.removeItem(at: base) }
            let package = base.appendingPathComponent("Thing.app", isDirectory: true)
            try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
            try "text".write(to: package.appendingPathComponent("inside.txt"), atomically: true, encoding: .utf8)
            let isPackage = try package.resourceValues(forKeys: [.isPackageKey]).isPackage == true
            try #require(isPackage, "this system does not treat .app as a package")
            #expect(!SandboxFileReader.macSeesIndexableFiles(in: base.resolvingSymlinksInPath().path))
            try "text".write(to: base.appendingPathComponent("plain.txt"), atomically: true, encoding: .utf8)
            #expect(SandboxFileReader.macSeesIndexableFiles(in: base.resolvingSymlinksInPath().path))
        }

        @Test func theFileLimitStopsTheScan() async throws {
            let h = try await FilesHarness.ready { _ in
                listReply((1...10).map { ("f\($0).txt", 1, 1_700_000_000) }, next: "f10.txt")
            }
            defer { h.cleanUp() }
            var limits = AppLocalIndexLimits()
            limits.maximumFilesPerFolder = 4
            let result = try await h.reader.scan(h.folder, limits: limits)
            #expect(result.files.count == 4)
            #expect(result.reachedLimit)
        }

        @Test func nothingListedWhileTheMacSeesFilesPointsAtMacOSPermission() async throws {
            let h = try await FilesHarness.ready(files: ["notes.md"]) { _ in listReply([]) }
            defer { h.cleanUp() }
            do {
                _ = try await h.reader.scan(h.folder, limits: .standard)
                Issue.record("expected an error")
            } catch {
                #expect("\(error)".contains("container-runtime-linux"))
            }
        }

        @Test func nothingListedInAFolderWithNothingToIndexIsFine() async throws {
            let h = try await FilesHarness.ready(files: ["pic.png"]) { _ in listReply([]) }
            defer { h.cleanUp() }
            let result = try await h.reader.scan(h.folder, limits: .standard)
            #expect(result.files.isEmpty)
        }

        @Test func aFolderPathWithACommaIsRefusedWithoutAskingTheSandbox() async throws {
            let h = try await FilesHarness.ready { _ in listReply([]) }
            defer { h.cleanUp() }
            let odd = AppSearchFolder(id: h.folder.id, path: h.folder.path + "/a,b")
            await #expect(throws: AppLocalSearchError.self) {
                _ = try await h.reader.scan(odd, limits: .standard)
            }
            #expect(SandboxStubProtocol.seen.isEmpty)
        }

        @Test func theSandboxsOwnRefusalOfAFolderReadsAsAMessage() async throws {
            let h = try await FilesHarness.ready { _ in
                SandboxStubProtocol.failure("not_found", "folder is not mounted", status: 404)
            }
            defer { h.cleanUp() }
            do {
                _ = try await h.reader.scan(h.folder, limits: .standard)
                Issue.record("expected an error")
            } catch {
                #expect("\(error)".contains("does not have this folder"))
            }
        }

        @Test func anUnsignedAnswerIsNotUsed() async throws {
            let h = try await FilesHarness.ready { _ in
                SandboxStubProtocol.json(["entries": [], "next_cursor": NSNull()], signature: .wrongKey)
            }
            defer { h.cleanUp() }
            await #expect(throws: AppLocalSearchError.self) {
                _ = try await h.reader.scan(h.folder, limits: .standard)
            }
        }
    }

    @MainActor
    @Suite struct SandboxFileReaderPassageTests {
        @Test func passagesAreCleanedAndKeepTheirPlace() async throws {
            let h = try await FilesHarness.ready { _ in
                passageReply([
                    linePassage("First\u{200B} passage\u{7}", 1, 3),
                    linePassage("   ", 4, 4),
                    linePassage("Second", 5, 9),
                ])
            }
            defer { h.cleanUp() }
            let passages = try await h.reader.passages(for: h.found("a.md"), in: h.folder, limits: .standard)
            #expect(passages.count == 2)
            #expect(passages[0].0 == nil && passages[0].1 == 1 && passages[0].2 == 3)
            #expect(passages[0].3 == "First passage")
            #expect(passages[1].3 == "Second")
            let seen = try #require(SandboxStubProtocol.seen.first)
            #expect(seen.path == "/v1/files/passages")
            #expect(seen.json["folder"] as? String == h.id)
            #expect(seen.json["path"] as? String == "a.md")
        }

        @Test func aPDFsPassagesCarryPageNumbers() async throws {
            let h = try await FilesHarness.ready { _ in
                passageReply([["page": 2, "line_start": NSNull(), "line_end": NSNull(), "text": "On page two"]])
            }
            defer { h.cleanUp() }
            let passages = try await h.reader.passages(for: h.found("m.pdf"), in: h.folder, limits: .standard)
            #expect(passages.count == 1)
            #expect(passages[0].0 == 2 && passages[0].1 == nil && passages[0].2 == nil)
        }

        struct BadPassage: Sendable, CustomTestStringConvertible {
            let path: String
            let page: Int?
            let lineStart: Int?
            let lineEnd: Int?
            var testDescription: String { "\(path) page \(String(describing: page)) lines \(String(describing: lineStart))-\(String(describing: lineEnd))" }
        }

        @Test(arguments: [
            // A PDF passage with lines, a page beyond the limit, no page; a
            // text passage with a page, a bad line range, no lines.
            BadPassage(path: "m.pdf", page: 1, lineStart: 1, lineEnd: 1),
            BadPassage(path: "m.pdf", page: 501, lineStart: nil, lineEnd: nil),
            BadPassage(path: "m.pdf", page: nil, lineStart: nil, lineEnd: nil),
            BadPassage(path: "a.md", page: 1, lineStart: 1, lineEnd: 1),
            BadPassage(path: "a.md", page: nil, lineStart: 0, lineEnd: 1),
            BadPassage(path: "a.md", page: nil, lineStart: 5, lineEnd: 2),
            BadPassage(path: "a.md", page: nil, lineStart: nil, lineEnd: nil),
        ])
        func aPassageOfTheWrongShapeStopsTheUpdate(bad: BadPassage) async throws {
            let h = try await FilesHarness.ready { _ in
                passageReply([["page": bad.page.map { $0 as Any } ?? NSNull(),
                               "line_start": bad.lineStart.map { $0 as Any } ?? NSNull(),
                               "line_end": bad.lineEnd.map { $0 as Any } ?? NSNull(),
                               "text": "x"]])
            }
            defer { h.cleanUp() }
            do {
                _ = try await h.reader.passages(for: h.found(bad.path), in: h.folder, limits: .standard)
                Issue.record("expected an error")
            } catch {
                #expect(!(error is AppLocalFileSkip) && !(error is AppLocalFileChanged))
            }
        }

        @Test func aFileThatChangedIsLeftForTheNextUpdate() async throws {
            let h = try await FilesHarness.ready { _ in passageReply(size: 5, [linePassage("x")]) }
            defer { h.cleanUp() }
            await #expect(throws: AppLocalFileChanged.self) {
                _ = try await h.reader.passages(for: h.found("a.md"), in: h.folder, limits: .standard)
            }
        }

        @Test func aDifferentStampAlsoMeansItChanged() async throws {
            let h = try await FilesHarness.ready { _ in passageReply(mtime: 1_700_000_001.5, [linePassage("x")]) }
            defer { h.cleanUp() }
            await #expect(throws: AppLocalFileChanged.self) {
                _ = try await h.reader.passages(for: h.found("a.md"), in: h.folder, limits: .standard)
            }
        }

        @Test func theSandboxSayingItChangedWhileReadingMeansTheSame() async throws {
            let h = try await FilesHarness.ready { _ in
                SandboxStubProtocol.failure("changed_while_reading", "changed while reading", status: 409)
            }
            defer { h.cleanUp() }
            await #expect(throws: AppLocalFileChanged.self) {
                _ = try await h.reader.passages(for: h.found("a.md"), in: h.folder, limits: .standard)
            }
        }

        @Test(arguments: ["oversized", "unreadable", "empty", "too_many_passages", "not_a_file", "not_found",
                          "timed_out"])
        func aFileTheSandboxCannotReadIsSkippedWithAReason(code: String) async throws {
            let h = try await FilesHarness.ready { _ in SandboxStubProtocol.failure(code, "no", status: 422) }
            defer { h.cleanUp() }
            do {
                _ = try await h.reader.passages(for: h.found("a.md"), in: h.folder, limits: .standard)
                Issue.record("expected a skip")
            } catch let skip as AppLocalFileSkip {
                #expect(!skip.reason.isEmpty)
            }
        }

        @Test func aProblemWithTheSandboxItselfIsNotAReasonToSkipTheFile() async throws {
            let h = try await FilesHarness.ready { _ in SandboxStubProtocol.failure("busy", "too busy", status: 503) }
            defer { h.cleanUp() }
            do {
                _ = try await h.reader.passages(for: h.found("a.md"), in: h.folder, limits: .standard)
                Issue.record("expected an error")
            } catch {
                #expect(!(error is AppLocalFileSkip))
            }
        }

        @Test func anOversizedFileIsSkippedWithoutAskingTheSandbox() async throws {
            let h = try await FilesHarness.ready { _ in passageReply([linePassage("x")]) }
            defer { h.cleanUp() }
            let big = h.found("a.md", size: AppLocalIndexLimits().maximumTextFileBytes + 1)
            await #expect(throws: AppLocalFileSkip.self) {
                _ = try await h.reader.passages(for: big, in: h.folder, limits: .standard)
            }
            let bigPDF = h.found("m.pdf", size: AppLocalIndexLimits().maximumPDFBytes + 1)
            await #expect(throws: AppLocalFileSkip.self) {
                _ = try await h.reader.passages(for: bigPDF, in: h.folder, limits: .standard)
            }
            #expect(SandboxStubProtocol.seen.isEmpty)
        }

        @Test func aPathOutsideTheFolderIsNeverSent() async throws {
            let h = try await FilesHarness.ready { _ in passageReply([linePassage("x")]) }
            defer { h.cleanUp() }
            let outside = AppLocalIndexer.FoundFile(path: "/etc/passwd.txt", size: 4, mtime: 0)
            await #expect(throws: AppLocalFileSkip.self) {
                _ = try await h.reader.passages(for: outside, in: h.folder, limits: .standard)
            }
            let dotdot = h.found("../x.txt")
            await #expect(throws: AppLocalFileSkip.self) {
                _ = try await h.reader.passages(for: dotdot, in: h.folder, limits: .standard)
            }
            #expect(SandboxStubProtocol.seen.isEmpty)
        }

        @Test func aFileWithNoUsableTextIsSkipped() async throws {
            let h = try await FilesHarness.ready { _ in passageReply([linePassage("\u{200B}\u{200B}")]) }
            defer { h.cleanUp() }
            await #expect(throws: AppLocalFileSkip.self) {
                _ = try await h.reader.passages(for: h.found("a.md"), in: h.folder, limits: .standard)
            }
        }
    }
}

// MARK: The indexer with a reader

private struct StubReader: AppLocalFileReading {
    var files: [String: (size: Int, mtime: Double)] = [:]
    var onPassages: @Sendable (String) throws -> [(Int?, Int?, Int?, String)]

    func scan(_ folder: AppSearchFolder, limits: AppLocalIndexLimits) async throws
        -> (files: [AppLocalIndexer.FoundFile], reachedLimit: Bool) {
        let root = (canonicalPath(folder.path) ?? folder.path) + "/"
        return (files.keys.sorted().map {
            AppLocalIndexer.FoundFile(path: root + $0, size: files[$0]!.size, mtime: files[$0]!.mtime)
        }, false)
    }

    func passages(for file: AppLocalIndexer.FoundFile, in folder: AppSearchFolder,
                  limits: AppLocalIndexLimits) async throws -> [(Int?, Int?, Int?, String)] {
        try onPassages((file.path as NSString).lastPathComponent)
    }
}

@Suite(.serialized) struct SandboxIndexerReaderTests {
    private struct Box {
        let base: URL
        let folder: AppSearchFolder
        let index: AppLocalSearchIndex

        func stamp(_ name: String) throws -> (size: Int, mtime: Double) {
            let path = (canonicalPath(folder.path) ?? folder.path) + "/" + name
            return try #require(AppLocalSearchIndex.fileStamp(path))
        }

        func cleanUp() { try? FileManager.default.removeItem(at: base) }
    }

    private func makeBox(_ names: [String]) throws -> Box {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-indexer-reader-\(UUID().uuidString)")
        let root = base.appendingPathComponent("Notes")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        for name in names { try "text of \(name)".write(to: root.appendingPathComponent(name),
                                                        atomically: true, encoding: .utf8) }
        return Box(base: base, folder: try AppSearchFolderStore.folder(for: root),
                   index: try AppLocalSearchIndex(databaseURL: base.appendingPathComponent("index.sqlite")))
    }

    private func reader(_ box: Box, names: [String],
                        onPassages: @escaping @Sendable (String) throws -> [(Int?, Int?, Int?, String)])
        throws -> StubReader {
        var reader = StubReader(onPassages: onPassages)
        for name in names {
            let stamp = try box.stamp(name)
            reader.files[name] = stamp
        }
        return reader
    }

    @Test func passagesFromTheReaderAreIndexedAndFound() async throws {
        let box = try makeBox(["a.md", "b.md"])
        defer { box.cleanUp() }
        let stub = try reader(box, names: ["a.md", "b.md"]) { name in
            [(nil, 1, 1, "The \(name) file talks about zeppelins.")]
        }
        let indexer = AppLocalIndexer(index: box.index, reader: stub)
        let status = try await indexer.update(box.folder)
        #expect(status.indexedFiles == 2)
        let found = try await box.index.search("zeppelins", folders: [box.folder], limit: 5)
        #expect(found.count == 2)
    }

    @Test func stampsThatNeverMatchTheMacAreReportedNotSilent() async throws {
        let box = try makeBox(["a.md", "b.md"])
        defer { box.cleanUp() }
        var stub = try reader(box, names: ["a.md", "b.md"]) { _ in [(nil, 1, 1, "text")] }
        // The sandbox's clock is off by a few microseconds on every file.
        for (name, stamp) in stub.files { stub.files[name] = (stamp.size, stamp.mtime + 0.01) }
        let status = try await AppLocalIndexer(index: box.index, reader: stub).update(box.folder)
        #expect(status.indexedFiles == 0)
        #expect(status.lastError?.contains("modification times") == true)
    }

    @Test func aSkipIsRememberedAndTheOthersAreStillIndexed() async throws {
        let box = try makeBox(["a.md", "bad.md"])
        defer { box.cleanUp() }
        let stub = try reader(box, names: ["a.md", "bad.md"]) { name in
            if name == "bad.md" { throw AppLocalFileSkip(reason: "it has no text") }
            return [(nil, 1, 1, "Readable words.")]
        }
        let status = try await AppLocalIndexer(index: box.index, reader: stub).update(box.folder)
        #expect(status.indexedFiles == 1)
        #expect(status.skippedFiles == 1)
    }

    @Test func aFileThatChangedIsNeitherIndexedNorSkipped() async throws {
        let box = try makeBox(["a.md"])
        defer { box.cleanUp() }
        let stub = try reader(box, names: ["a.md"]) { _ in throw AppLocalFileChanged() }
        let status = try await AppLocalIndexer(index: box.index, reader: stub).update(box.folder)
        #expect(status.indexedFiles == 0)
        #expect(status.skippedFiles == 0)
    }

    @Test func aProblemWithTheReaderStopsTheUpdateAndSkipsNothing() async throws {
        let box = try makeBox(["a.md", "b.md"])
        defer { box.cleanUp() }
        let stub = try reader(box, names: ["a.md", "b.md"]) { _ in throw SandboxError.notReady }
        await #expect(throws: SandboxError.notReady) {
            _ = try await AppLocalIndexer(index: box.index, reader: stub).update(box.folder)
        }
        let counts = try await box.index.counts(folder: box.folder.id)
        #expect(counts.skipped == 0 && counts.indexed == 0)
    }

    @Test func aFileTheMacSeesChangedIsNotStored() async throws {
        let box = try makeBox(["a.md"])
        defer { box.cleanUp() }
        var stub = try reader(box, names: ["a.md"]) { _ in [(nil, 1, 1, "words")] }
        stub.files["a.md"] = (size: 999, mtime: stub.files["a.md"]!.mtime)
        let status = try await AppLocalIndexer(index: box.index, reader: stub).update(box.folder)
        #expect(status.indexedFiles == 0)
    }

    @Test func withoutAReaderTheAppReadsTheFilesItselfAsBefore() async throws {
        let box = try makeBox(["a.md"])
        defer { box.cleanUp() }
        let status = try await AppLocalIndexer(index: box.index).update(box.folder)
        #expect(status.indexedFiles == 1)
    }
}

// MARK: Keeping the sandbox in step with the folders

@MainActor
@Suite struct SandboxFilesServiceTests {
    private func makeService(_ h: SandboxHarness) -> SandboxFilesService {
        SandboxFilesService(controller: h.controller,
                            access: SandboxWebAccess(controller: h.controller))
    }

    private func folders(_ count: Int) throws -> (base: URL, folders: [AppSearchFolder]) {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-service-\(UUID().uuidString)")
        var list: [AppSearchFolder] = []
        for index in 0..<count {
            let root = base.appendingPathComponent("F\(index)")
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            list.append(try AppSearchFolderStore.folder(for: root))
        }
        return (base, list)
    }

    @Test func mountsAreLowercaseSortedAndSkipTheUnmountable() throws {
        let made = try folders(2)
        defer { try? FileManager.default.removeItem(at: made.base) }
        let comma = AppSearchFolder(path: made.folders[0].path + ",x")
        let missing = AppSearchFolder(path: "/nonexistent-\(UUID().uuidString)")
        let mounts = SandboxFilesService.mounts(for: made.folders + [comma, missing, made.folders[0]])
        #expect(mounts.count == 2)
        #expect(mounts.map(\.id) == mounts.map(\.id).sorted())
        #expect(mounts.allSatisfy { SandboxFolderMount.isFolderID($0.id) })
        #expect(mounts.allSatisfy { SandboxFolderMount.isMountable($0.path) })
    }

    @Test func theSandboxStartsWithTheFoldersAndIsKeptWhileTheyStay() async throws {
        let h = SandboxHarness.make(profile: .files)
        let made = try folders(1)
        defer { h.cleanUp(); try? FileManager.default.removeItem(at: made.base) }
        let service = makeService(h)
        try await service.ensureRunning(for: made.folders)
        #expect(h.controller.state == .ready)
        #expect(h.world.count("run") == 1)
        try await service.ensureRunning(for: made.folders)
        #expect(h.world.count("run") == 1)
    }

    @Test func aChangeOfFoldersRestartsTheSandboxWithTheNewOnes() async throws {
        let h = SandboxHarness.make(profile: .files)
        let made = try folders(2)
        defer { h.cleanUp(); try? FileManager.default.removeItem(at: made.base) }
        let service = makeService(h)
        try await service.ensureRunning(for: [made.folders[0]])
        try await service.ensureRunning(for: made.folders)
        #expect(h.world.count("run") == 2)
        #expect(h.world.count("delete") >= 1)
        #expect(h.controller.activeMountIDs.count == 2)
        #expect(h.world.runningContainers.count == 1)
    }

    @Test func noFolderToMountMeansNoSandbox() async throws {
        let h = SandboxHarness.make(profile: .files)
        let made = try folders(1)
        defer { h.cleanUp(); try? FileManager.default.removeItem(at: made.base) }
        let service = makeService(h)
        try await service.ensureRunning(for: made.folders)
        try await service.ensureRunning(for: [])
        #expect(h.controller.state == .off)
        #expect(h.world.runningContainers.isEmpty)
    }

    @Test func oneFolderMacOSRefusesIsLeftOutAndNamedWhileTheOthersAreRead() async throws {
        let h = SandboxHarness.make(profile: .files)
        let made = try folders(3)
        defer { h.cleanUp(); try? FileManager.default.removeItem(at: made.base) }
        h.world.refusedMountPaths = ["/F1,"]
        let service = makeService(h)
        let blocked = try await service.ensureRunning(for: made.folders)
        let refusedID = made.folders[1].id.uuidString.lowercased()
        #expect(Array(blocked.keys) == [refusedID])
        let message = blocked[refusedID] ?? ""
        #expect(message.contains("F1"))
        #expect(message.contains("container-runtime-linux"))
        #expect(!message.contains("F0"))
        let expected = [made.folders[0], made.folders[2]].map { $0.id.uuidString.lowercased() }.sorted()
        #expect(h.controller.state == .ready)
        #expect(h.controller.activeMountIDs == expected)
        #expect(h.world.runningContainers.count == 1)
        // Allowed since: the next call reads all three.
        h.world.refusedMountPaths = []
        let allowed = try await service.ensureRunning(for: made.folders)
        #expect(allowed.isEmpty)
        #expect(h.controller.activeMountIDs.count == 3)
        #expect(h.world.runningContainers.count == 1)
    }

    @Test func whenEveryFolderIsRefusedItThrowsAndNamesThemAndNothingRuns() async throws {
        let h = SandboxHarness.make(profile: .files)
        let made = try folders(2)
        defer { h.cleanUp(); try? FileManager.default.removeItem(at: made.base) }
        h.world.refusedMountPaths = ["/F0,", "/F1,"]
        let service = makeService(h)
        do {
            try await service.ensureRunning(for: made.folders)
            Issue.record("expected an error")
        } catch {
            let text = "\(error)"
            #expect(text.contains("F0") && text.contains("F1"))
            #expect(text.contains("container-runtime-linux"))
        }
        #expect(h.world.runningContainers.isEmpty)
    }

    @Test func aSingleRefusedFolderIsNamed() async throws {
        let h = SandboxHarness.make(profile: .files)
        let made = try folders(1)
        defer { h.cleanUp(); try? FileManager.default.removeItem(at: made.base) }
        h.world.refusedMountPaths = ["/F0,"]
        let service = makeService(h)
        do {
            try await service.ensureRunning(for: made.folders)
            Issue.record("expected an error")
        } catch {
            #expect("\(error)".contains("F0"))
            #expect("\(error)".contains("Files and Folders"))
        }
        #expect(h.world.count("run") == 1)
    }

    @Test func tooManyFoldersAreNotTriedOneByOne() async throws {
        let h = SandboxHarness.make(profile: .files)
        let made = try folders(SandboxFilesService.maximumProbes + 1)
        defer { h.cleanUp(); try? FileManager.default.removeItem(at: made.base) }
        h.world.refusedMountPaths = ["/F3,"]
        let service = makeService(h)
        do {
            try await service.ensureRunning(for: made.folders)
            Issue.record("expected an error")
        } catch {
            #expect("\(error)".contains("container-runtime-linux"))
        }
        #expect(h.world.count("run") == 1)
        #expect(h.world.runningContainers.isEmpty)
    }

    @Test func theRefusalOfTheVirtualMachineFrameworkIsRecognised() {
        let detail = "Error: failed to bootstrap container (cause: VZErrorDomain Code=2 "
            + "\"A directory sharing device configuration is invalid.\")"
        let error = SandboxError.containerCommandFailed(action: "start the sandbox", detail: detail)
        #expect(SandboxController.folderAccessProblem(in: error, mounts: [mount()]) == .folderAccessDenied)
        #expect(SandboxController.folderAccessProblem(in: error, mounts: []) == nil)
    }

    @Test func aFailedStartIsReportedAndNothingKeepsRunning() async throws {
        let h = SandboxHarness.make(profile: .files)
        let made = try folders(1)
        defer { h.cleanUp(); try? FileManager.default.removeItem(at: made.base) }
        h.world.failures["run"] = SandboxProcessResult(status: 1, output: "boom\n")
        let service = makeService(h)
        do {
            try await service.ensureRunning(for: made.folders)
            Issue.record("expected an error")
        } catch {
            #expect("\(error)".contains("Could not start the sandbox"))
        }
        #expect(h.world.runningContainers.isEmpty)
    }

    @Test func aFailedSandboxIsStartedAgainByTheNextCall() async throws {
        let h = SandboxHarness.make(profile: .files)
        let made = try folders(1)
        defer { h.cleanUp(); try? FileManager.default.removeItem(at: made.base) }
        h.world.failures["run"] = SandboxProcessResult(status: 1, output: "boom\n")
        let service = makeService(h)
        _ = try? await service.ensureRunning(for: made.folders)
        h.world.failures["run"] = nil
        try await service.ensureRunning(for: made.folders)
        #expect(h.controller.state == .ready)
    }

    @Test func aCancelledRunDoesNotBringTheSandboxBack() async throws {
        let h = SandboxHarness.make(profile: .files)
        let made = try folders(1)
        defer { h.cleanUp(); try? FileManager.default.removeItem(at: made.base) }
        let service = makeService(h)
        let folders = made.folders
        let task = Task { @MainActor in try await service.ensureRunning(for: folders) }
        task.cancel()
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(h.world.count("run") == 0)
        #expect(h.controller.state == .off)
    }

    @Test func theHomeFolderIsNeverMounted() throws {
        let made = try folders(1)
        defer { try? FileManager.default.removeItem(at: made.base) }
        let home = AppSearchFolder(path: NSHomeDirectory())
        let mounts = SandboxFilesService.mounts(for: made.folders + [home])
        #expect(mounts.count == 1)
        #expect(mounts.first?.id == made.folders[0].id.uuidString.lowercased())
    }

    @Test func stoppingTheServiceStopsTheSandbox() async throws {
        let h = SandboxHarness.make(profile: .files)
        let made = try folders(1)
        defer { h.cleanUp(); try? FileManager.default.removeItem(at: made.base) }
        let service = makeService(h)
        try await service.ensureRunning(for: made.folders)
        await service.stop()
        #expect(h.controller.state == .off)
        #expect(h.world.runningContainers.isEmpty)
    }
}

// MARK: The setting, the tool store and the model

private let filesMac = TUFFDeviceCapabilities(
    unifiedMemoryBytes: 32 * TUFFModelCatalog.oneGiB, macOSMajorVersion: 26, appleSiliconGeneration: 2)

@Suite struct SandboxFilesSettingFileTests {
    @Test func theSettingIsOffByDefaultAndOldFilesLoadOff() throws {
        #expect(!MacAppSettings().runFilesInSandbox)
        let data = try JSONEncoder().encode(MacAppSettings())
        var legacy = try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
        #expect(legacy["runFilesInSandbox"] != nil)
        legacy.removeValue(forKey: "runFilesInSandbox")
        let migrated = try JSONDecoder().decode(
            MacAppSettings.self, from: JSONSerialization.data(withJSONObject: legacy))
        #expect(!migrated.runFilesInSandbox)
        #expect(migrated.version == MacAppSettings.currentVersion)
    }

    @Test func theSettingRoundTripsAndIsIndependentOfTheWebOne() throws {
        var settings = MacAppSettings()
        settings.runFilesInSandbox = true
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: JSONEncoder().encode(settings))
        #expect(decoded.runFilesInSandbox)
        #expect(!decoded.runWebInSandbox)
        #expect(decoded == settings)
    }
}

/// One folder with a note in it, a files sandbox on the fake `container`
/// whose clients talk to `SandboxStubProtocol`, and a stub that lists and
/// reads that note.
@MainActor
private struct IndexingRig {
    let base: URL
    let root: URL
    let sandbox: SandboxHarness
    let service: SandboxFilesService
    let store: AppToolStore

    static let noteText = "The zeppelin hangar opens at dawn."

    static func make() throws -> IndexingRig {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-files-rig-\(UUID().uuidString)")
        let root = base.appendingPathComponent("Notes")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let text = noteText
        try text.write(to: root.appendingPathComponent("note.md"), atomically: true, encoding: .utf8)
        let sandbox = SandboxHarness.make(freePort: SandboxStubProtocol.port, profile: .files)
        let world = sandbox.world
        let note = (canonicalPath(root.path) ?? root.path) + "/note.md"
        let stamp = AppLocalSearchIndex.fileStamp(note)!
        let unix = Date(timeIntervalSinceReferenceDate: stamp.mtime).timeIntervalSince1970
        SandboxStubProtocol.install(key: Data()) { seen in
            if seen.path == "/v1/files/list" {
                return listReply([("note.md", stamp.size, unix)])
            }
            return passageReply(size: stamp.size, mtime: unix, [linePassage(text, 1, 1)])
        }
        // The sandbox's secret is new at every start: answer with it.
        sandbox.probe.handler = { _, key, _ in
            SandboxStubProtocol.setKey(key)
            return SandboxHealth(profile: "files", api: 1, blockedAddresses: 0, folders: world.lastRunFolderIDs)
        }
        let service = SandboxFilesService(controller: sandbox.controller,
                                          access: SandboxWebHarness.access(for: sandbox.controller))
        let store = AppToolStore.inMemory(indexURL: base.appendingPathComponent("index.sqlite"))
        return IndexingRig(base: base, root: root, sandbox: sandbox, service: service, store: store)
    }

    func cleanUp() {
        store.stopIndexing()
        sandbox.cleanUp()
        try? FileManager.default.removeItem(at: base)
    }

    func waitUntilIndexed(folder id: UUID) async -> Bool {
        for _ in 0..<500 {
            if let status = store.folderStatus[id], !store.isIndexing,
               status.indexedFiles > 0 || status.lastError != nil { return true }
            try? await Task.sleep(for: .milliseconds(10))
        }
        return false
    }
}

extension SandboxStubTests {
    @MainActor
    @Suite struct SandboxFilesIndexingTests {
        @Test func withTheSettingOnTheFilesAreReadOnlyThroughTheSandbox() async throws {
            let rig = try IndexingRig.make()
            defer { rig.cleanUp() }
            rig.store.sandboxFiles = rig.service
            rig.store.indexInSandbox = true
            rig.store.addFolders([rig.root])
            let id = try #require(rig.store.folders.first?.id)
            #expect(await rig.waitUntilIndexed(folder: id))
            let status = try #require(rig.store.folderStatus[id])
            #expect(status.lastError == nil)
            #expect(status.indexedFiles == 1)
            #expect(rig.sandbox.world.count("run") == 1)
            let mount = try #require(rig.sandbox.world.lastRunFolderMounts.first)
            #expect(mount.hasSuffix(",readonly"))
            #expect(mount.contains("target=/files/\(id.uuidString.lowercased())"))
            #expect(SandboxStubProtocol.seen.map(\.path) == ["/v1/files/list", "/v1/files/passages"])
            let results = try await rig.store.makeToolbox(provider: .duckDuckGo).searchFiles("zeppelin hangar", 3)
            #expect(results.count == 1)
            #expect(results.first?.text.contains("zeppelin") == true)
        }

        @Test func withTheSettingOnAndNoSandboxNothingIsReadDirectly() async throws {
            let rig = try IndexingRig.make()
            defer { rig.cleanUp() }
            rig.store.indexInSandbox = true
            rig.store.addFolders([rig.root])
            let id = try #require(rig.store.folders.first?.id)
            #expect(rig.store.folderStatus[id]?.lastError == "The file sandbox is not available.")
            try await Task.sleep(for: .milliseconds(100))
            #expect(rig.store.folderStatus[id]?.indexedFiles ?? 0 == 0)
            let results = try await rig.store.makeToolbox(provider: .duckDuckGo).searchFiles("zeppelin", 3)
            #expect(results.isEmpty)
        }

        @Test func aSandboxThatCannotStartStopsIndexingWithItsMessageAndReadsNothing() async throws {
            let rig = try IndexingRig.make()
            defer { rig.cleanUp() }
            rig.sandbox.world.failures["run"] = SandboxProcessResult(status: 1, output: "boom\n")
            rig.store.sandboxFiles = rig.service
            rig.store.indexInSandbox = true
            rig.store.addFolders([rig.root])
            let id = try #require(rig.store.folders.first?.id)
            #expect(await rig.waitUntilIndexed(folder: id))
            #expect(rig.store.folderStatus[id]?.lastError?.contains("Could not start the sandbox") == true)
            #expect(rig.store.folderStatus[id]?.indexedFiles ?? 0 == 0)
            #expect(SandboxStubProtocol.seen.isEmpty)
            let results = try await rig.store.makeToolbox(provider: .duckDuckGo).searchFiles("zeppelin", 3)
            #expect(results.isEmpty)
        }

        @Test func removingEveryFolderStopsTheSandboxSoNothingStaysMounted() async throws {
            let rig = try IndexingRig.make()
            defer { rig.cleanUp() }
            rig.store.sandboxFiles = rig.service
            rig.store.indexInSandbox = true
            rig.store.addFolders([rig.root])
            let id = try #require(rig.store.folders.first?.id)
            #expect(await rig.waitUntilIndexed(folder: id))
            #expect(rig.sandbox.controller.state == .ready)
            rig.store.removeAllFolders()
            for _ in 0..<500 where rig.sandbox.controller.state != .off {
                try await Task.sleep(for: .milliseconds(10))
            }
            #expect(rig.sandbox.controller.state == .off)
            #expect(rig.sandbox.world.runningContainers.isEmpty)
        }

        @Test func skippedFilesAreReadAgainWhenTheyAreForgotten() async throws {
            let rig = try IndexingRig.make()
            defer { rig.cleanUp() }
            rig.store.addFolders([rig.root])
            let id = try #require(rig.store.folders.first?.id)
            #expect(await rig.waitUntilIndexed(folder: id))
            let index = try AppLocalSearchIndex(databaseURL: rig.base.appendingPathComponent("index.sqlite"))
            let note = (canonicalPath(rig.root.path) ?? rig.root.path) + "/note.md"
            let stamp = try #require(AppLocalSearchIndex.fileStamp(note))
            try await index.skip(path: note, folder: id, size: stamp.size, mtime: stamp.mtime, reason: "x")
            #expect(try await index.counts(folder: id).skipped == 1)
            try await index.forgetSkipped()
            #expect(try await index.counts(folder: id).skipped == 0)
        }

        @Test func withTheSettingOffTheAppReadsTheFilesItselfAndNeverStartsTheSandbox() async throws {
            let rig = try IndexingRig.make()
            defer { rig.cleanUp() }
            rig.store.sandboxFiles = rig.service
            rig.store.addFolders([rig.root])
            let id = try #require(rig.store.folders.first?.id)
            #expect(await rig.waitUntilIndexed(folder: id))
            #expect(rig.store.folderStatus[id]?.indexedFiles == 1)
            #expect(rig.sandbox.world.count("run") == 0)
            #expect(SandboxStubProtocol.seen.isEmpty)
        }
    }

    @MainActor
    @Suite struct SandboxFilesSettingModelTests {
        private func makeModel(_ rig: IndexingRig, directory: URL? = nil) -> AppModel {
            let model = makeAppModel(modelDirectory: directory, settingsPersistenceEnabled: directory != nil,
                                     deviceCapabilities: filesMac, toolStore: rig.store)
            model.installFilesSandbox(rig.sandbox.controller) {
                SandboxClient(lease: $0, protocolClasses: [SandboxStubProtocol.self])
            }
            return model
        }

        @Test func theSettingIsOffByDefault() throws {
            let rig = try IndexingRig.make()
            defer { rig.cleanUp() }
            let model = makeModel(rig)
            #expect(!model.runFilesInSandbox)
            #expect(!rig.store.indexInSandbox)
            #expect(model.isSandboxSupported)
        }

        @Test func theSettingPersistsAcrossRelaunch() async throws {
            let rig = try IndexingRig.make()
            let root = FileManager.default.temporaryDirectory
                .appendingPathComponent("tuff-files-setting-\(UUID().uuidString)", isDirectory: true)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            defer { rig.cleanUp(); try? FileManager.default.removeItem(at: root) }
            let directory = root.appendingPathComponent("gemma4.gturbo", isDirectory: true)
            let first = makeModel(rig, directory: directory)
            first.runFilesInSandbox = true
            await first.finishSandboxWork()
            first.flushPendingSettings()
            let other = try IndexingRig.make()
            defer { other.cleanUp() }
            let relaunched = makeModel(other, directory: directory)
            #expect(relaunched.runFilesInSandbox)
            #expect(other.store.indexInSandbox)
        }

        @Test func turningItOnReadsTheFoldersInTheSandboxAndOffStopsIt() async throws {
            let rig = try IndexingRig.make()
            defer { rig.cleanUp() }
            let model = makeModel(rig)
            rig.store.addFolders([rig.root])
            let id = try #require(rig.store.folders.first?.id)
            model.fileSearchEnabled = true
            #expect(await rig.waitUntilIndexed(folder: id))
            #expect(rig.sandbox.world.count("run") == 0)

            model.runFilesInSandbox = true
            await model.finishSandboxWork()
            for _ in 0..<500 where rig.sandbox.controller.state != .ready {
                try await Task.sleep(for: .milliseconds(10))
            }
            #expect(rig.sandbox.controller.state == .ready)
            #expect(rig.store.indexInSandbox)
            #expect(rig.sandbox.world.count("run") == 1)

            model.runFilesInSandbox = false
            await model.finishSandboxWork()
            #expect(rig.sandbox.controller.state == .off)
            #expect(rig.sandbox.world.runningContainers.isEmpty)
            #expect(!rig.store.indexInSandbox)
        }

        @Test func quittingStopsTheFilesSandboxToo() async throws {
            let rig = try IndexingRig.make()
            defer { rig.cleanUp() }
            let model = makeModel(rig)
            try await rig.service.ensureRunning(for: [try AppSearchFolderStore.folder(for: rig.root)])
            #expect(rig.sandbox.controller.state == .ready)
            model.stopSandboxWhenQuitting()
            #expect(rig.sandbox.controller.state == .off)
            #expect(rig.sandbox.world.runningContainers.isEmpty)
        }
    }
}
#endif
