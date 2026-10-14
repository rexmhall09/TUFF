#if os(macOS)
import Foundation
import Testing
@testable import TUFFAppCore

private func makeTool(_ world: FakeContainerWorld, installed: Bool = true,
                      team: String? = nil) -> ContainerTool {
    ContainerTool(runner: world, path: FakeContainerWorld.containerPath,
                  isExecutable: { _ in installed }, fileExists: { _ in installed },
                  requiredTeamIdentifier: team)
}

/// The `SandboxError` a call throws, or nil when it does not throw.
private func thrownError(_ body: () async throws -> Void) async -> SandboxError? {
    do {
        try await body()
    } catch let error as SandboxError {
        return error
    } catch {
        Issue.record("Unexpected error: \(error)")
    }
    return nil
}

private func commandNames(_ world: FakeContainerWorld) -> [String] {
    world.calls.map { FakeContainerWorld.name(of: $0) }
}

/// A runner whose `run` always throws and whose `runBlocking` never answers.
private struct ThrowingRunner: SandboxProcessRunning {
    struct Failure: LocalizedError {
        var errorDescription: String? { "disk\nexploded" }
    }

    func run(executable: String, arguments: [String],
             timeout: TimeInterval) async throws -> SandboxProcessResult {
        throw Failure()
    }

    func runBlocking(executable: String, arguments: [String],
                     timeout: TimeInterval) -> SandboxProcessResult? {
        nil
    }
}

@Suite struct ContainerVersionTests {
    @Test func parsesTheRealVersionOutput() {
        let version = ContainerVersion.parse("container CLI version 1.5.0 (build: release, commit: d265d66)")
        #expect(version == ContainerVersion(major: 1, minor: 5, patch: 0))
    }

    @Test func parsesThreePartVersions() {
        #expect(ContainerVersion.parse("1.2.3") == ContainerVersion(major: 1, minor: 2, patch: 3))
        #expect(ContainerVersion.parse("container 10.20.30") == ContainerVersion(major: 10, minor: 20, patch: 30))
    }

    @Test func aMissingPatchNumberIsZero() {
        #expect(ContainerVersion.parse("1.5") == ContainerVersion(major: 1, minor: 5, patch: 0))
        #expect(ContainerVersion.parse("version 2.10 beta") == ContainerVersion(major: 2, minor: 10, patch: 0))
    }

    @Test func onlyTheFirstVersionCounts() {
        #expect(ContainerVersion.parse("1.2.3 and 4.5.6") == ContainerVersion(major: 1, minor: 2, patch: 3))
        #expect(ContainerVersion.parse("1.2.3.4") == ContainerVersion(major: 1, minor: 2, patch: 3))
    }

    @Test func numbersWithoutADotAreSkipped() {
        #expect(ContainerVersion.parse("build 12 version 1.5.0") == ContainerVersion(major: 1, minor: 5, patch: 0))
    }

    @Test func suffixesAreIgnored() {
        #expect(ContainerVersion.parse("1.5.0-beta.2") == ContainerVersion(major: 1, minor: 5, patch: 0))
        #expect(ContainerVersion.parse("v1.6.1\n") == ContainerVersion(major: 1, minor: 6, patch: 1))
    }

    @Test func garbageGivesNil() {
        #expect(ContainerVersion.parse("") == nil)
        #expect(ContainerVersion.parse("no digits here") == nil)
        #expect(ContainerVersion.parse("v1") == nil)
        #expect(ContainerVersion.parse("1.") == nil)
        #expect(ContainerVersion.parse(".5") == nil)
        #expect(ContainerVersion.parse("a.b.c") == nil)
    }

    @Test func aNumberTooBigForAnIntGivesNil() {
        #expect(ContainerVersion.parse("99999999999999999999.1.0") == nil)
    }

    @Test func descriptionHasThreeParts() {
        #expect(ContainerVersion(major: 1, minor: 5, patch: 0).description == "1.5.0")
        #expect(ContainerVersion(major: 12, minor: 0, patch: 7).description == "12.0.7")
    }

    @Test func versionsAreOrderedNumerically() {
        let v149 = ContainerVersion(major: 1, minor: 4, patch: 9)
        let v150 = ContainerVersion(major: 1, minor: 5, patch: 0)
        let v151 = ContainerVersion(major: 1, minor: 5, patch: 1)
        let v1100 = ContainerVersion(major: 1, minor: 10, patch: 0)
        let v200 = ContainerVersion(major: 2, minor: 0, patch: 0)
        #expect(v149 < v150)
        #expect(v150 < v151)
        #expect(v151 < v1100)
        #expect(v1100 < v200)
        #expect(!(v150 < v150))
        #expect(v150 >= v150)
        #expect(v200 > v1100)
        #expect([v200, v149, v1100, v150, v151].sorted() == [v149, v150, v151, v1100, v200])
    }

    @Test func minimumVersionIsOnePointFive() {
        #expect(ContainerTool.minimumVersion == ContainerVersion(major: 1, minor: 5, patch: 0))
        #expect(ContainerTool.installedPath == "/usr/local/bin/container")
    }
}

@Suite struct ContainerToolVerifyTests {
    @Test func succeedsAndReturnsTheVersion() async throws {
        let world = FakeContainerWorld()
        let version = try await makeTool(world).verify()
        #expect(version == ContainerVersion(major: 1, minor: 5, patch: 0))
    }

    @Test func checksTheSignatureBeforeRunningTheTool() async throws {
        let world = FakeContainerWorld()
        _ = try await makeTool(world).verify()
        #expect(world.calls == [
            FakeContainerWorld.Call(
                executable: "/usr/bin/codesign",
                arguments: ["--verify", "--strict", FakeContainerWorld.containerPath]),
            FakeContainerWorld.Call(
                executable: "/usr/bin/codesign",
                arguments: ["-dv", "--verbose=2", FakeContainerWorld.containerPath]),
            FakeContainerWorld.Call(
                executable: FakeContainerWorld.containerPath, arguments: ["--version"]),
        ])
        #expect(commandNames(world) == ["codesign", "codesign", "version"])
    }

    @Test func notInstalledThrowsWithoutRunningAnything() async {
        let world = FakeContainerWorld()
        let error = await thrownError { _ = try await makeTool(world, installed: false).verify() }
        #expect(error == SandboxError.containerNotInstalled)
        #expect(world.calls.isEmpty)
    }

    @Test func aToolThatIsThereButNotTrustedIsNotCalledNotInstalled() async {
        let world = FakeContainerWorld()
        let tool = ContainerTool(runner: world, path: FakeContainerWorld.containerPath,
                                 isExecutable: { _ in false }, fileExists: { _ in true })
        let error = await thrownError { _ = try await tool.verify() }
        #expect(error == SandboxError.containerNotTrusted)
        #expect(world.calls.isEmpty)
    }

    @Test func isExecutableIsAskedAboutTheConfiguredPath() async throws {
        final class Seen: @unchecked Sendable {
            private let lock = NSLock()
            private var paths: [String] = []
            func add(_ path: String) { lock.withLock { paths.append(path) } }
            var all: [String] { lock.withLock { paths } }
        }
        let seen = Seen()
        let world = FakeContainerWorld()
        let tool = ContainerTool(runner: world, path: "/opt/somewhere/container",
                                 isExecutable: { seen.add($0); return true })
        _ = try await tool.verify()
        #expect(seen.all == ["/opt/somewhere/container"])
        #expect(world.calls.first?.arguments == ["--verify", "--strict", "/opt/somewhere/container"])
        #expect(world.calls.last?.executable == "/opt/somewhere/container")
    }

    @Test func aFailedSignatureCheckThrowsNotSignedAndNeverRunsTheTool() async {
        let world = FakeContainerWorld()
        world.codesignStatus = 1
        let error = await thrownError { _ = try await makeTool(world).verify() }
        #expect(error == SandboxError.containerNotSigned("code object is not signed at all"))
        #expect(world.count("version") == 0)
        #expect(commandNames(world) == ["codesign"])
    }

    @Test func aTimedOutSignatureCheckThrowsNotSigned() async {
        let world = FakeContainerWorld()
        world.failures["codesign"] = SandboxProcessResult(status: 0, output: "", timedOut: true)
        let error = await thrownError { _ = try await makeTool(world).verify() }
        #expect(error == SandboxError.containerNotSigned(""))
        #expect(world.count("version") == 0)
    }

    @Test func theSignatureErrorKeepsOnlyTheLastTwoLines() async {
        let world = FakeContainerWorld()
        world.failures["codesign"] = SandboxProcessResult(status: 3, output: "one\ntwo\nthree\n")
        let error = await thrownError { _ = try await makeTool(world).verify() }
        #expect(error == SandboxError.containerNotSigned("two | three"))
    }

    @Test func aMatchingTeamIdentifierPasses() async throws {
        let world = FakeContainerWorld()
        let version = try await makeTool(world, team: "TESTTEAM01").verify()
        #expect(version == ContainerVersion(major: 1, minor: 5, patch: 0))
    }

    @Test func theTeamIsCheckedBetweenTheSignatureAndTheVersion() async throws {
        let world = FakeContainerWorld()
        _ = try await makeTool(world, team: "TESTTEAM01").verify()
        #expect(commandNames(world) == ["codesign", "codesign", "version"])
        #expect(world.calls[1].arguments == ["-dv", "--verbose=2", FakeContainerWorld.containerPath])
        #expect(world.calls[1].executable == "/usr/bin/codesign")
    }

    @Test func withoutAPinnedTeamAnyDeveloperTeamPasses() async throws {
        let world = FakeContainerWorld()
        _ = try await makeTool(world, team: nil).verify()
        #expect(world.calls.contains(where: { $0.arguments.first == "-dv" }))
    }

    @Test func aPinnedTeamIsAlsoRequiredFromTheSignatureCheckItself() async throws {
        let world = FakeContainerWorld()
        _ = try await makeTool(world, team: "TESTTEAM01").verify()
        let arguments = world.calls[0].arguments
        #expect(arguments.first == "--verify")
        #expect(arguments.contains("-R=anchor apple generic and certificate leaf[subject.OU] = \"TESTTEAM01\""))
        #expect(arguments.last == FakeContainerWorld.containerPath)
    }

    @Test func theAppPinsTheTeamOfAppleContainerSignature() {
        #expect(ContainerTool.appleTeamIdentifier == "UPBK2H6LZM")
        #expect(SandboxDependencies.live().tool.requiredTeamIdentifier == "UPBK2H6LZM")
    }

    @Test func anAdHocSignatureIsRefused() async {
        let world = FakeContainerWorld()
        world.codesignDetails = "Identifier=container\nSignature=adhoc\nTeamIdentifier=not set\n"
        let error = await thrownError { _ = try await makeTool(world, team: nil).verify() }
        #expect(error == SandboxError.containerNotSigned("it is not signed by a developer"))
        #expect(world.count("version") == 0)
    }

    @Test func noTeamAtAllIsRefusedWithoutAPin() async {
        let world = FakeContainerWorld()
        world.codesignDetails = "Identifier=container\nTeamIdentifier=not set\n"
        let error = await thrownError { _ = try await makeTool(world, team: nil).verify() }
        #expect(error == SandboxError.containerNotSigned("it does not name a developer team"))
        #expect(world.count("version") == 0)
    }

    @Test func aFileOnlyRootCanChangeIsTrusted() {
        #expect(ContainerTool.isTrustedFile("/bin/ls"))
        #expect(!ContainerTool.isTrustedFile("/nonexistent/container"))
    }

    @Test func aFileThePersonOwnsIsNotTrusted() throws {
        guard getuid() != 0 else { return }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-sandbox-fake-container-\(UUID().uuidString)")
        try "#!/bin/sh\n".write(to: url, atomically: true, encoding: .utf8)
        defer { try? FileManager.default.removeItem(at: url) }
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
        #expect(!ContainerTool.isTrustedFile(url.path))
    }

    @Test func aDifferentTeamIdentifierIsRefused() async {
        let world = FakeContainerWorld()
        let error = await thrownError { _ = try await makeTool(world, team: "OTHERTEAM9").verify() }
        #expect(error == SandboxError.containerNotSigned("the signer is not the expected one"))
        #expect(world.count("version") == 0)
    }

    @Test func aTeamIdentifierThatOnlyStartsTheSameIsRefused() async {
        let world = FakeContainerWorld()
        let error = await thrownError { _ = try await makeTool(world, team: "TESTTEAM").verify() }
        #expect(error == SandboxError.containerNotSigned("the signer is not the expected one"))
    }

    @Test func aMissingTeamLineIsRefused() async {
        let world = FakeContainerWorld()
        world.codesignDetails = "Identifier=com.apple.container\n"
        let error = await thrownError { _ = try await makeTool(world, team: "TESTTEAM01").verify() }
        #expect(error == SandboxError.containerNotSigned("the signer is not the expected one"))
    }

    @Test func theTeamLineMayBeAnywhereInTheDetails() async throws {
        let world = FakeContainerWorld()
        world.codesignDetails = "Executable=/usr/local/bin/container\nIdentifier=x\nTeamIdentifier=TESTTEAM01\nSealed Resources=none\n"
        let version = try await makeTool(world, team: "TESTTEAM01").verify()
        #expect(version.major == 1)
    }

    @Test func aTooOldVersionIsRefused() async {
        let world = FakeContainerWorld()
        world.versionOutput = "container CLI version 1.4.2 (build: release)"
        let error = await thrownError { _ = try await makeTool(world).verify() }
        #expect(error == SandboxError.containerTooOld(found: "1.4.2", minimum: "1.5.0"))
    }

    @Test func theMinimumAndNewerVersionsPass() async throws {
        let world = FakeContainerWorld()
        let tool = makeTool(world)
        world.versionOutput = "container CLI version 1.5.0"
        let minimum = try await tool.verify()
        #expect(minimum == ContainerVersion(major: 1, minor: 5, patch: 0))
        world.versionOutput = "container CLI version 1.5.1"
        let patch = try await tool.verify()
        #expect(patch == ContainerVersion(major: 1, minor: 5, patch: 1))
        world.versionOutput = "container CLI version 2.0.0"
        let major = try await tool.verify()
        #expect(major == ContainerVersion(major: 2, minor: 0, patch: 0))
        world.versionOutput = "container CLI version 1.10.0"
        let minor = try await tool.verify()
        #expect(minor == ContainerVersion(major: 1, minor: 10, patch: 0))
    }

    @Test func aTwoPartVersionBelowTheMinimumIsRefused() async {
        let world = FakeContainerWorld()
        world.versionOutput = "version 1.4"
        let error = await thrownError { _ = try await makeTool(world).verify() }
        #expect(error == SandboxError.containerTooOld(found: "1.4.0", minimum: "1.5.0"))
    }

    @Test func anUnparseableVersionThrowsCommandFailed() async {
        let world = FakeContainerWorld()
        world.versionOutput = "garbage"
        let error = await thrownError { _ = try await makeTool(world).verify() }
        #expect(error == SandboxError.containerCommandFailed(
            action: "read the version of the container tool", detail: "garbage"))
    }

    @Test func aFailingVersionCommandThrowsCommandFailedWithTheTail() async {
        let world = FakeContainerWorld()
        world.failures["version"] = SandboxProcessResult(status: 2, output: "first\nsecond\nthird\n")
        let error = await thrownError { _ = try await makeTool(world).verify() }
        #expect(error == SandboxError.containerCommandFailed(
            action: "read the version of the container tool", detail: "second | third"))
    }

    @Test func aTimedOutVersionCommandThrowsCommandFailed() async {
        let world = FakeContainerWorld()
        world.failures["version"] = SandboxProcessResult(
            status: 0, output: "container CLI version 1.5.0\n", timedOut: true)
        let error = await thrownError { _ = try await makeTool(world).verify() }
        guard case .containerCommandFailed(let action, _)? = error else {
            Issue.record("Expected containerCommandFailed, got \(String(describing: error))")
            return
        }
        #expect(action == "read the version of the container tool")
    }

    @Test func aRunnerThatThrowsDuringVerifyPropagatesItsError() async {
        let tool = ContainerTool(runner: ThrowingRunner(), path: FakeContainerWorld.containerPath,
                                 isExecutable: { _ in true })
        await #expect(throws: (any Error).self) {
            _ = try await tool.verify()
        }
    }
}

@Suite struct ContainerToolCommandTests {
    // MARK: system

    @Test func systemIsRunningWhenStatusSucceeds() async {
        let world = FakeContainerWorld()
        world.systemRunning = true
        let running = await makeTool(world).systemIsRunning()
        #expect(running)
        #expect(world.containerCalls == [["system", "status"]])
    }

    @Test func systemIsNotRunningWhenStatusFails() async {
        let world = FakeContainerWorld()
        world.systemRunning = false
        let running = await makeTool(world).systemIsRunning()
        #expect(running == false)
    }

    @Test func systemIsNotRunningWhenStatusTimesOut() async {
        let world = FakeContainerWorld()
        world.failures["system status"] = SandboxProcessResult(status: 0, output: "", timedOut: true)
        let running = await makeTool(world).systemIsRunning()
        #expect(running == false)
    }

    @Test func systemIsNotRunningWhenTheRunnerThrows() async {
        let tool = ContainerTool(runner: ThrowingRunner(), path: FakeContainerWorld.containerPath,
                                 isExecutable: { _ in true })
        let running = await tool.systemIsRunning()
        #expect(running == false)
    }

    @Test func startSystemEnablesKernelInstall() async throws {
        let world = FakeContainerWorld()
        world.systemRunning = false
        let tool = makeTool(world)
        try await tool.startSystem()
        #expect(world.containerCalls == [["system", "start", "--enable-kernel-install"]])
        let running = await tool.systemIsRunning()
        #expect(running)
    }

    @Test func startSystemFailureIsCommandFailed() async {
        let world = FakeContainerWorld()
        world.failures["system start"] = SandboxProcessResult(status: 1, output: "kernel download failed\n")
        let error = await thrownError { try await makeTool(world).startSystem() }
        #expect(error == SandboxError.containerCommandFailed(
            action: "start Apple container's system service", detail: "kernel download failed"))
    }

    @Test func startSystemTimeoutIsTimedOut() async {
        let world = FakeContainerWorld()
        world.failures["system start"] = SandboxProcessResult(status: 0, output: "", timedOut: true)
        let error = await thrownError { try await makeTool(world).startSystem() }
        #expect(error == SandboxError.timedOut(action: "start Apple container's system service"))
    }

    @Test func aTimeoutIsReportedEvenWhenTheStatusIsNonZero() async {
        let world = FakeContainerWorld()
        world.failures["system start"] = SandboxProcessResult(status: 15, output: "killed\n", timedOut: true)
        let error = await thrownError { try await makeTool(world).startSystem() }
        #expect(error == SandboxError.timedOut(action: "start Apple container's system service"))
    }

    @Test func aThrowingRunnerBecomesCommandFailedWithACleanedMessage() async {
        let tool = ContainerTool(runner: ThrowingRunner(), path: FakeContainerWorld.containerPath,
                                 isExecutable: { _ in true })
        let error = await thrownError { try await tool.startSystem() }
        #expect(error == SandboxError.containerCommandFailed(
            action: "start Apple container's system service", detail: "disk exploded"))
    }

    @Test func failureDetailIsTheLastFourLinesOnOneLine() async {
        let world = FakeContainerWorld()
        world.failures["system start"] = SandboxProcessResult(
            status: 1, output: "l1\nl2\n\nl3\nl4\nl5\n")
        let error = await thrownError { try await makeTool(world).startSystem() }
        #expect(error == SandboxError.containerCommandFailed(
            action: "start Apple container's system service", detail: "l2 | l3 | l4 | l5"))
    }

    // MARK: list

    @Test func containerIDsInSkipsTheHeader() {
        let output = "ID  IMAGE  OS  ARCH  STATE  ADDR\nalpha  img  linux  arm64  running  192.168.64.2\nbeta  img  linux  arm64  stopped\n"
        #expect(ContainerTool.containerIDs(in: output) == ["alpha", "beta"])
    }

    @Test func containerIDsInIgnoresBlankLines() {
        let output = "ID IMAGE\n\nalpha x\n   \n\t\nbeta y\n\n"
        #expect(ContainerTool.containerIDs(in: output) == ["alpha", "beta"])
    }

    @Test func containerIDsInTakesOnlyTheFirstColumn() {
        let output = "ID\tIMAGE\tOS\tARCH\nalpha\timg\tlinux\tarm64\tmany\textra\tcolumns\n   indented   x   y\n"
        #expect(ContainerTool.containerIDs(in: output) == ["alpha", "indented"])
    }

    @Test func containerIDsInHandlesEmptyAndHeaderOnlyOutput() {
        #expect(ContainerTool.containerIDs(in: "") == [])
        #expect(ContainerTool.containerIDs(in: "ID IMAGE OS\n") == [])
        #expect(ContainerTool.containerIDs(in: "\n\n") == [])
    }

    @Test func containerIDsInHandlesWindowsLineEndingsAndNoHeader() {
        #expect(ContainerTool.containerIDs(in: "ID  IMAGE\r\nalpha  x\r\nbeta  y\r\n") == ["alpha", "beta"])
        #expect(ContainerTool.containerIDs(in: "alpha x\nbeta y") == ["alpha", "beta"])
    }

    @Test func containerIDsInKeepsOrderAndDuplicates() {
        #expect(ContainerTool.containerIDs(in: "b x\na x\nb x\n") == ["b", "a", "b"])
    }

    @Test func listPassesNoFlagsWithoutAll() async throws {
        let world = FakeContainerWorld()
        world.addContainer("run1", running: true)
        world.addContainer("old1", running: false)
        let ids = try await makeTool(world).containerIDs(all: false)
        #expect(ids == ["run1"])
        #expect(world.containerCalls == [["list", "--quiet"]])
    }

    @Test func listPassesAllWhenAskedForStoppedContainersToo() async throws {
        let world = FakeContainerWorld()
        world.addContainer("run1", running: true)
        world.addContainer("old1", running: false)
        let ids = try await makeTool(world).containerIDs(all: true)
        #expect(ids == ["run1", "old1"])
        #expect(world.containerCalls == [["list", "--all", "--quiet"]])
    }

    @Test func listOfNothingIsEmpty() async throws {
        let world = FakeContainerWorld()
        let ids = try await makeTool(world).containerIDs(all: true)
        #expect(ids == [])
    }

    @Test func aFailingListThrowsCommandFailed() async {
        let world = FakeContainerWorld()
        world.failures["list"] = SandboxProcessResult(status: 1, output: "service not running\n")
        let error = await thrownError { _ = try await makeTool(world).containerIDs(all: false) }
        #expect(error == SandboxError.containerCommandFailed(
            action: "list the containers", detail: "service not running"))
    }

    // MARK: run

    @Test func runDetachedPassesTheArgumentsAfterRun() async throws {
        let world = FakeContainerWorld()
        try await makeTool(world).runDetached(arguments: ["--name", "tuff-1", "--detach", "image@sha256:x"])
        #expect(world.containerCalls == [["run", "--name", "tuff-1", "--detach", "image@sha256:x"]])
        #expect(world.runningContainers == ["tuff-1"])
    }

    @Test func runDetachedFailureIsCommandFailed() async {
        let world = FakeContainerWorld()
        world.failures["run"] = SandboxProcessResult(status: 125, output: "cannot start\n")
        let error = await thrownError { try await makeTool(world).runDetached(arguments: ["--name", "x"]) }
        #expect(error == SandboxError.containerCommandFailed(
            action: "start the sandbox", detail: "cannot start"))
    }

    @Test func runDetachedTimeoutIsTimedOut() async {
        let world = FakeContainerWorld()
        world.failures["run"] = SandboxProcessResult(status: 0, output: "", timedOut: true)
        let error = await thrownError { try await makeTool(world).runDetached(arguments: ["--name", "x"]) }
        #expect(error == SandboxError.timedOut(action: "start the sandbox"))
    }

    // MARK: stop and delete

    @Test func stopAndDeleteStopsThenDeletes() async throws {
        let world = FakeContainerWorld()
        world.addContainer("c1", running: true)
        let tool = makeTool(world)
        await tool.stopAndDelete("c1")
        #expect(world.containerCalls == [["stop", "c1"], ["delete", "c1"]])
        let ids = try await tool.containerIDs(all: true)
        #expect(ids == [])
    }

    @Test func stopAndDeleteStillDeletesWhenStopFails() async {
        let world = FakeContainerWorld()
        world.failures["stop"] = SandboxProcessResult(status: 1, output: "no such container\n")
        await makeTool(world).stopAndDelete("c1")
        #expect(world.containerCalls == [["stop", "c1"], ["delete", "c1"]])
    }

    @Test func stopAndDeleteDoesNotThrowWhenTheRunnerThrows() async {
        let tool = ContainerTool(runner: ThrowingRunner(), path: FakeContainerWorld.containerPath,
                                 isExecutable: { _ in true })
        await tool.stopAndDelete("c1")
    }

    @Test func anUndeletableContainerStaysInTheListAfterStopAndDelete() async throws {
        let world = FakeContainerWorld()
        world.addContainer("stuck", running: true)
        world.undeletable = ["stuck"]
        let tool = makeTool(world)
        await tool.stopAndDelete("stuck")
        let ids = try await tool.containerIDs(all: true)
        #expect(ids == ["stuck"])
    }

    @Test func blockingStopAndDeleteReturnsTrueWhenTheContainerIsGone() {
        let world = FakeContainerWorld()
        world.addContainer("c1", running: true)
        world.addContainer("other", running: true)
        let gone = makeTool(world).stopAndDeleteBlocking("c1")
        #expect(gone)
        #expect(world.containerCalls == [["stop", "c1"], ["delete", "c1"], ["list", "--all", "--quiet"]])
        #expect(world.runningContainers == ["other"])
    }

    @Test func blockingStopAndDeleteReturnsTrueForAContainerThatNeverExisted() {
        let world = FakeContainerWorld()
        #expect(makeTool(world).stopAndDeleteBlocking("ghost"))
    }

    @Test func blockingStopAndDeleteReturnsFalseWhenTheContainerStays() {
        let world = FakeContainerWorld()
        world.addContainer("stuck", running: true)
        world.undeletable = ["stuck"]
        #expect(makeTool(world).stopAndDeleteBlocking("stuck") == false)
    }

    @Test func blockingStopAndDeleteReturnsFalseWhenTheListFails() {
        let world = FakeContainerWorld()
        world.failures["list"] = SandboxProcessResult(status: 1, output: "service not running\n")
        #expect(makeTool(world).stopAndDeleteBlocking("c1") == false)
    }

    @Test func blockingStopAndDeleteReturnsFalseWhenTheRunnerNeverAnswers() {
        let tool = ContainerTool(runner: ThrowingRunner(), path: FakeContainerWorld.containerPath,
                                 isExecutable: { _ in true })
        #expect(tool.stopAndDeleteBlocking("c1") == false)
    }

    // MARK: exec, logs

    @Test func execRunsTheCommandInsideTheContainer() async throws {
        let world = FakeContainerWorld()
        let result = try await makeTool(world).exec("c1", ["echo", "hi"])
        #expect(world.containerCalls == [["exec", "c1", "echo", "hi"]])
        #expect(result.status == 0)
    }

    @Test func execReturnsANonZeroStatusInsteadOfThrowing() async throws {
        let world = FakeContainerWorld()
        world.failures["exec echo"] = SandboxProcessResult(status: 3, output: "bad\n")
        let result = try await makeTool(world).exec("c1", ["echo", "hi"])
        #expect(result.status == 3)
        #expect(result.output == "bad\n")
        #expect(result.timedOut == false)
    }

    @Test func execPropagatesARunnerError() async {
        let tool = ContainerTool(runner: ThrowingRunner(), path: FakeContainerWorld.containerPath,
                                 isExecutable: { _ in true })
        await #expect(throws: (any Error).self) {
            _ = try await tool.exec("c1", ["true"])
        }
    }

    @Test func logTailIsTheLastThreeLinesOnOneLine() async {
        let world = FakeContainerWorld()
        world.logsOutput = "l1\nl2\nl3\nl4\nl5\n"
        let tail = await makeTool(world).logTail("c1")
        #expect(tail == "l3 | l4 | l5")
        #expect(world.containerCalls == [["logs", "c1"]])
    }

    @Test func logTailOfNothingIsEmpty() async {
        let world = FakeContainerWorld()
        world.logsOutput = ""
        let tail = await makeTool(world).logTail("c1")
        #expect(tail == "")
    }

    @Test func logTailIsEmptyWhenTheRunnerThrows() async {
        let tool = ContainerTool(runner: ThrowingRunner(), path: FakeContainerWorld.containerPath,
                                 isExecutable: { _ in true })
        let tail = await tool.logTail("c1")
        #expect(tail == "")
    }

    // MARK: images

    @Test func inspectImageReturnsTheOutputWhenThereIsOne() async throws {
        let world = FakeContainerWorld()
        let output = await makeTool(world).inspectImage("registry.example/img")
        let text = try #require(output)
        #expect(text.contains(FakeContainerWorld.digest))
        #expect(world.containerCalls == [["image", "inspect", "registry.example/img"]])
    }

    @Test func inspectImageIsNilWhenTheImageIsMissing() async {
        let world = FakeContainerWorld()
        world.imageDigests = nil
        let output = await makeTool(world).inspectImage("registry.example/img")
        #expect(output == nil)
    }

    @Test func inspectImageIsNilWhenTheCommandTimesOut() async {
        let world = FakeContainerWorld()
        world.failures["image inspect"] = SandboxProcessResult(status: 0, output: "{}", timedOut: true)
        let output = await makeTool(world).inspectImage("registry.example/img")
        #expect(output == nil)
    }

    @Test func inspectImageIsNilWhenTheRunnerThrows() async {
        let tool = ContainerTool(runner: ThrowingRunner(), path: FakeContainerWorld.containerPath,
                                 isExecutable: { _ in true })
        let output = await tool.inspectImage("registry.example/img")
        #expect(output == nil)
    }

    @Test func pullImagePullsTheReference() async throws {
        let world = FakeContainerWorld()
        world.imageDigests = nil
        try await makeTool(world).pullImage("registry.example/img@sha256:abc")
        #expect(world.containerCalls == [["image", "pull", "registry.example/img@sha256:abc"]])
        #expect(world.imageDigests == world.digestsAfterPull)
    }

    @Test func pullImageFailureIsCommandFailed() async {
        let world = FakeContainerWorld()
        world.failures["image pull"] = SandboxProcessResult(status: 1, output: "denied\n")
        let error = await thrownError { try await makeTool(world).pullImage("registry.example/img") }
        #expect(error == SandboxError.containerCommandFailed(
            action: "download the sandbox image", detail: "denied"))
    }

    @Test func pullImageTimeoutIsTimedOut() async {
        let world = FakeContainerWorld()
        world.failures["image pull"] = SandboxProcessResult(status: 0, output: "", timedOut: true)
        let error = await thrownError { try await makeTool(world).pullImage("registry.example/img") }
        #expect(error == SandboxError.timedOut(action: "download the sandbox image"))
    }
}
#endif
