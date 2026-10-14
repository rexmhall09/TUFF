#if os(macOS)
import Foundation
import Testing
@testable import TUFFAppCore

/// Lets a test hold a fake `container` command at one point and let it go
/// later.
actor SandboxGate {
    private var isOpen = false
    private var hasArrived = false
    private var held: [CheckedContinuation<Void, Never>] = []
    private var watchers: [CheckedContinuation<Void, Never>] = []

    /// Called by the code under test; returns once the gate is open.
    func pass() async {
        hasArrived = true
        let waiting = watchers
        watchers = []
        for watcher in waiting { watcher.resume() }
        if isOpen { return }
        await withCheckedContinuation { held.append($0) }
    }

    func waitUntilArrived() async {
        if hasArrived { return }
        await withCheckedContinuation { watchers.append($0) }
    }

    func open() {
        isOpen = true
        let waiting = held
        held = []
        for waiter in waiting { waiter.resume() }
    }
}

/// Polls a condition for up to `timeout` seconds.
@MainActor
func sandboxWaitFor(timeout: Double = 5, _ condition: @MainActor () -> Bool) async -> Bool {
    let deadline = Date().addingTimeInterval(timeout)
    while !condition() {
        if Date() > deadline { return false }
        try? await Task.sleep(for: .milliseconds(5))
    }
    return true
}

func sandboxFailureText(_ state: SandboxController.State) -> String? {
    if case .failed(let text) = state { return text }
    return nil
}

@MainActor
@Suite struct SandboxControllerTests {
    static let first = SandboxController.containerNamePrefix + "00000001"
    static let second = SandboxController.containerNamePrefix + "00000002"

    // MARK: A normal start and stop

    @Test func startsChecksAndIsReady() async {
        let h = SandboxHarness.make()
        defer { h.cleanUp() }
        #expect(h.controller.state == .off)
        await h.controller.start()
        #expect(h.controller.state == .ready)
        #expect(h.world.runningContainers == [Self.first])
        #expect(h.world.count("run") == 1)
        #expect(h.secretFolders.count == 1)
        #expect(h.probe.attemptCount == 1)
        await h.controller.stop()
        #expect(h.controller.state == .off)
        #expect(h.world.runningContainers.isEmpty)
        #expect(h.world.stoppedContainers.isEmpty)
        #expect(h.secretFolders.isEmpty)
    }

    @Test func checksRunInTheDocumentedOrder() async throws {
        let h = SandboxHarness.make()
        defer { h.cleanUp() }
        await h.controller.start()
        #expect(h.controller.state == .ready)
        let names = h.world.calls.map(FakeContainerWorld.name(of:))
        // The signature is checked before the tool is ever run.
        #expect(names.first == "codesign")
        let order = ["codesign", "version", "system status", "image inspect", "list", "run", "exec nft", "exec python3"]
        var last = -1
        for name in order {
            let index = try #require(names.firstIndex(of: name), "missing \(name)")
            #expect(index > last, "\(name) came too early")
            last = index
        }
        await h.controller.stop()
    }

    @Test func startsTheSystemServiceOnlyWhenItIsNotRunning() async {
        let running = SandboxHarness.make()
        defer { running.cleanUp() }
        await running.controller.start()
        #expect(running.world.count("system start") == 0)
        await running.controller.stop()

        let world = FakeContainerWorld()
        world.systemRunning = false
        let cold = SandboxHarness.make(world: world)
        defer { cold.cleanUp() }
        await cold.controller.start()
        #expect(cold.controller.state == .ready)
        #expect(world.count("system start") == 1)
        let startCall = world.containerCalls.first { $0.first == "system" && $0.dropFirst().first == "start" }
        #expect(startCall?.contains("--enable-kernel-install") == true)
        await cold.controller.stop()
    }

    @Test func containerRunUsesTheSafeFlags() async throws {
        let h = SandboxHarness.make(freePort: 50_123)
        defer { h.cleanUp() }
        await h.controller.start()
        let arguments = try #require(h.world.lastRunArguments)
        #expect(arguments.first == "run")
        for flag in ["--detach", "--read-only"] { #expect(arguments.contains(flag)) }
        #expect(h.world.lastRunValue(after: "--cap-drop") == "ALL")
        let added = arguments.indices.filter { arguments[$0] == "--cap-add" }.map { arguments[$0 + 1] }
        #expect(Set(added) == ["NET_ADMIN", "SETUID", "SETGID", "SETPCAP"])
        #expect(!arguments.contains("--privileged"))
        #expect(!arguments.contains("--network"))
        #expect(!arguments.contains("--volume") && !arguments.contains("-v"))
        #expect(h.world.lastRunValue(after: "--publish") == "127.0.0.1:50123:9100")
        #expect(h.world.lastRunPort == 50_123)
        #expect(h.world.lastRunValue(after: "--name") == Self.first)
        #expect(arguments.last == "registry.example/tuff-sandbox@\(FakeContainerWorld.digest)")
        // One mount only, of the secret folder, read-only.
        let mounts = arguments.indices.filter { arguments[$0] == "--mount" }.map { arguments[$0 + 1] }
        #expect(mounts.count == 1)
        let folder = try #require(h.secretFolders.first)
        let secrets = SandboxLocations.secretsDirectory(in: h.stateDirectory)
        #expect(mounts.first == "type=bind,source=\(secrets.appendingPathComponent(folder).path),target=/run/tuff-sandbox,readonly")
        await h.controller.stop()
    }

    @Test func blocksTheMacsOwnAddresses() async {
        let h = SandboxHarness.make()
        defer { h.cleanUp() }
        await h.controller.start()
        let expected = HostAddresses.normalized(SandboxHarness.hostAddresses)
        #expect(h.world.lastRunBlockedAddresses == expected)
        #expect(!h.world.lastRunBlockedAddresses.contains { $0.contains("%") })
        #expect(h.world.lastRunBlockedAddresses.contains("192.168.1.20"))
        await h.controller.stop()
    }

    @Test func theKeyAndPortAreWhatTheProbeAndTheLeaseUse() async throws {
        let h = SandboxHarness.make(freePort: 50_200)
        defer { h.cleanUp() }
        await h.controller.start()
        let lease = try h.controller.acquire()
        #expect(h.controller.requestsInFlight == 1)
        let request = try #require(h.probe.requests.last)
        #expect(request.port == 50_200)
        #expect(request.key == lease.key)
        // The key is the 64 hexadecimal characters of the secret file.
        #expect(lease.key.count == 64)
        let folder = try #require(h.secretFolders.first)
        let file = SandboxLocations.secretsDirectory(in: h.stateDirectory)
            .appendingPathComponent(folder).appendingPathComponent(SandboxSecret.fileName)
        #expect(try Data(contentsOf: file) == lease.key)
        h.controller.release(lease)
        #expect(h.controller.requestsInFlight == 0)
        await h.controller.stop()
    }

    @Test func eachStartGetsANewSecret() async throws {
        let h = SandboxHarness.make()
        defer { h.cleanUp() }
        await h.controller.start()
        let one = try h.controller.acquire().key
        await h.controller.stop()
        await h.controller.start()
        let two = try h.controller.acquire().key
        #expect(one != two)
        #expect(h.secretFolders.count == 1)
        await h.controller.stop()
    }

    // MARK: Start and stop at awkward moments

    @Test func twoStartsAtOnceStartOneContainer() async {
        let h = SandboxHarness.make()
        defer { h.cleanUp() }
        let one = Task { await h.controller.start() }
        let two = Task { await h.controller.start() }
        await one.value
        await two.value
        #expect(h.controller.state == .ready)
        #expect(h.world.count("run") == 1)
        #expect(h.world.runningContainers.count == 1)
        await h.controller.stop()
    }

    @Test func startingWhenReadyDoesNothing() async {
        let h = SandboxHarness.make()
        defer { h.cleanUp() }
        await h.controller.start()
        let generation = h.controller.generation
        await h.controller.start()
        #expect(h.controller.state == .ready)
        #expect(h.controller.generation == generation)
        #expect(h.world.count("run") == 1)
        await h.controller.stop()
    }

    @Test func stopWhenOffDoesNothing() async {
        let h = SandboxHarness.make()
        defer { h.cleanUp() }
        await h.controller.stop()
        #expect(h.controller.state == .off)
        #expect(h.world.calls.isEmpty)
    }

    @Test func stopDuringStartLeavesNothingRunning() async {
        let world = FakeContainerWorld()
        let gate = SandboxGate()
        world.beforeRun = { await gate.pass() }
        let h = SandboxHarness.make(world: world)
        defer { h.cleanUp() }
        let starting = Task { await h.controller.start() }
        await gate.waitUntilArrived()
        #expect(h.controller.state == .starting)
        await h.controller.stop()
        #expect(h.controller.state == .off)
        await gate.open()
        await starting.value
        #expect(h.controller.state == .off)
        #expect(world.runningContainers.isEmpty)
        #expect(world.stoppedContainers.isEmpty)
        #expect(h.secretFolders.isEmpty)
        // And it can be started again.
        await h.controller.start()
        #expect(h.controller.state == .ready)
        await h.controller.stop()
    }

    @Test func leasesAreRefusedUntilReady() async throws {
        let world = FakeContainerWorld()
        let gate = SandboxGate()
        world.beforeRun = { await gate.pass() }
        let h = SandboxHarness.make(world: world)
        defer { h.cleanUp() }
        #expect(throws: SandboxError.notReady) { try h.controller.acquire() }
        let starting = Task { await h.controller.start() }
        await gate.waitUntilArrived()
        #expect(throws: SandboxError.stillStarting) { try h.controller.acquire() }
        await gate.open()
        await starting.value
        #expect(h.controller.state == .ready)
        let lease = try h.controller.acquire()
        h.controller.release(lease)
        await h.controller.stop()
        #expect(throws: SandboxError.notReady) { try h.controller.acquire() }
    }

    @Test func stopDuringARequestInvalidatesTheLease() async throws {
        let h = SandboxHarness.make()
        defer { h.cleanUp() }
        await h.controller.start()
        let lease = try h.controller.acquire()
        await h.controller.stop()
        #expect(h.controller.state == .off)
        #expect(h.controller.requestsInFlight == 0)
        #expect(h.world.runningContainers.isEmpty)
        // A late release of the old lease changes nothing, even after a new start.
        await h.controller.start()
        let fresh = try h.controller.acquire()
        #expect(fresh.generation != lease.generation)
        h.controller.release(lease)
        #expect(h.controller.requestsInFlight == 1)
        h.controller.release(fresh)
        #expect(h.controller.requestsInFlight == 0)
        await h.controller.stop()
    }

    // MARK: Idle stop

    @Test func stopsWhenIdle() async {
        let h = SandboxHarness.make()
        defer { h.cleanUp() }
        await h.controller.start()
        let waiting = await sandboxWaitFor { h.clock.waitingCount == 1 }
        #expect(waiting)
        h.clock.fireIdle()
        let stopped = await sandboxWaitFor { h.controller.state == .off }
        #expect(stopped)
        #expect(h.world.runningContainers.isEmpty)
        #expect(h.secretFolders.isEmpty)
    }

    @Test func aLeaseKeepsTheSandboxFromIdling() async throws {
        let h = SandboxHarness.make()
        defer { h.cleanUp() }
        await h.controller.start()
        _ = await sandboxWaitFor { h.clock.waitingCount == 1 }
        let lease = try h.controller.acquire()
        let cancelled = await sandboxWaitFor { h.clock.waitingCount == 0 }
        #expect(cancelled)
        h.clock.fireIdle()
        try? await Task.sleep(for: .milliseconds(50))
        #expect(h.controller.state == .ready)
        // Giving the lease back starts the countdown again.
        h.controller.release(lease)
        let again = await sandboxWaitFor { h.clock.waitingCount == 1 }
        #expect(again)
        h.clock.fireIdle()
        let stopped = await sandboxWaitFor { h.controller.state == .off }
        #expect(stopped)
    }

    // MARK: Failures: each stops at the step that failed and leaves nothing

    private func expectFailed(_ h: SandboxHarness, containing text: String,
                              sourceLocation: SourceLocation = #_sourceLocation) {
        let message = sandboxFailureText(h.controller.state)
        #expect(message?.contains(text) == true, "state was \(h.controller.state)", sourceLocation: sourceLocation)
        #expect(h.world.runningContainers.isEmpty, sourceLocation: sourceLocation)
        #expect(h.world.stoppedContainers.isEmpty, sourceLocation: sourceLocation)
        #expect(h.secretFolders.isEmpty, sourceLocation: sourceLocation)
    }

    @Test func failsBeforeAnythingOnAnOldSystem() async {
        let h = SandboxHarness.make(osSupported: false)
        defer { h.cleanUp() }
        await h.controller.start()
        expectFailed(h, containing: "macOS 26")
        #expect(h.world.calls.isEmpty)
    }

    @Test func failsWhenContainerIsNotInstalled() async {
        let h = SandboxHarness.make(isInstalled: false)
        defer { h.cleanUp() }
        await h.controller.start()
        expectFailed(h, containing: "not installed")
        #expect(h.world.calls.isEmpty)
    }

    @Test func failsWhenContainerIsNotSigned() async {
        let world = FakeContainerWorld()
        world.codesignStatus = 3
        let h = SandboxHarness.make(world: world)
        defer { h.cleanUp() }
        await h.controller.start()
        expectFailed(h, containing: "valid signature")
        #expect(world.containerCalls.isEmpty, "the tool must not run before its signature is checked")
    }

    @Test func failsWhenTheSignerIsNotTheExpectedOne() async {
        let h = SandboxHarness.make(requiredTeamIdentifier: "SOMEONEELSE")
        defer { h.cleanUp() }
        await h.controller.start()
        expectFailed(h, containing: "valid signature")
        #expect(h.world.containerCalls.isEmpty)
    }

    @Test func failsWhenContainerIsTooOld() async {
        let world = FakeContainerWorld()
        world.versionOutput = "container CLI version 1.4.2"
        let h = SandboxHarness.make(world: world)
        defer { h.cleanUp() }
        await h.controller.start()
        expectFailed(h, containing: "too old")
        #expect(world.count("run") == 0)
    }

    @Test func failsWhenTheSystemServiceWillNotStart() async {
        let world = FakeContainerWorld()
        world.systemRunning = false
        world.failures["system start"] = SandboxProcessResult(status: 1, output: "kernel install declined\n")
        let h = SandboxHarness.make(world: world)
        defer { h.cleanUp() }
        await h.controller.start()
        expectFailed(h, containing: "system service")
        #expect(world.count("run") == 0)
    }

    @Test func failsWhenNoImageHasBeenPublished() async {
        let h = SandboxHarness.make(pin: .release)
        defer { h.cleanUp() }
        await h.controller.start()
        expectFailed(h, containing: "has not been published")
        #expect(h.world.count("image pull") == 0)
        #expect(h.world.count("run") == 0)
    }

    @Test func pullsAMissingImageOnce() async {
        let world = FakeContainerWorld()
        world.imageDigests = nil
        let h = SandboxHarness.make(world: world)
        defer { h.cleanUp() }
        await h.controller.start()
        #expect(h.controller.state == .ready)
        #expect(world.count("image pull") == 1)
        await h.controller.stop()
    }

    @Test func doesNotPullAnImageThatIsThere() async {
        let h = SandboxHarness.make()
        defer { h.cleanUp() }
        await h.controller.start()
        #expect(h.world.count("image pull") == 0)
        await h.controller.stop()
    }

    @Test func failsAndRunsNothingWhenThePulledImageIsDifferent() async {
        let world = FakeContainerWorld()
        world.imageDigests = nil
        world.digestsAfterPull = [FakeContainerWorld.otherDigest]
        let h = SandboxHarness.make(world: world)
        defer { h.cleanUp() }
        await h.controller.start()
        expectFailed(h, containing: "not the one this version of TUFF expects")
        #expect(world.count("run") == 0)
    }

    @Test func failsWhenTheContainerCannotBeStarted() async {
        let world = FakeContainerWorld()
        world.failures["run"] = SandboxProcessResult(status: 1, output: "image not found\n")
        let h = SandboxHarness.make(world: world)
        defer { h.cleanUp() }
        await h.controller.start()
        expectFailed(h, containing: "start the sandbox")
        #expect(world.count("exec nft") == 0)
    }

    @Test func failsWhenThePathHasAComma() async {
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff,comma-\(UUID().uuidString)", isDirectory: true)
        let h = SandboxHarness.make(stateDirectory: base)
        defer { h.cleanUp() }
        await h.controller.start()
        expectFailed(h, containing: "comma")
        #expect(h.world.count("run") == 0)
    }

    @Test func failsWhenTheMacHasTooManyAddresses() async {
        let many = (1...300).map { "10.1.\($0 / 200).\($0 % 200 + 1)" }
        let h = SandboxHarness.make(addresses: many)
        defer { h.cleanUp() }
        await h.controller.start()
        expectFailed(h, containing: "more network addresses")
        #expect(h.world.count("run") == 0)
    }

    @Test func failsWhenTheSandboxNeverAnswers() async {
        let h = SandboxHarness.make(healthAttempts: 6)
        defer { h.cleanUp() }
        h.probe.handler = { _, _, _ in throw SandboxError.notHealthy("connection refused") }
        await h.controller.start()
        expectFailed(h, containing: "does not answer correctly")
        #expect(sandboxFailureText(h.controller.state)?.contains("connection refused") == true)
        #expect(h.probe.attemptCount == 6)
        #expect(h.world.count("exec nft") == 0)
    }

    @Test func failsWhenTheAnswerIsNotSignedWithTheSecret() async {
        let h = SandboxHarness.make(healthAttempts: 3)
        defer { h.cleanUp() }
        h.probe.handler = { _, _, _ in
            throw SandboxError.notHealthy("the answer is not signed with this start's secret")
        }
        await h.controller.start()
        expectFailed(h, containing: "not signed")
        #expect(h.world.count("exec nft") == 0)
    }

    @Test func failsFastWhenTheContainerEndedByItself() async {
        let world = FakeContainerWorld()
        world.containersStayUp = false
        world.logsOutput = "entrypoint: cannot load the firewall\n"
        let h = SandboxHarness.make(world: world, healthAttempts: 40)
        defer { h.cleanUp() }
        h.probe.handler = { _, _, _ in throw SandboxError.notHealthy("connection refused") }
        await h.controller.start()
        expectFailed(h, containing: "the container stopped")
        #expect(sandboxFailureText(h.controller.state)?.contains("cannot load the firewall") == true)
        #expect(h.probe.attemptCount < 40)
    }

    @Test func failsWhenTheBlockedAddressCountDiffers() async {
        let h = SandboxHarness.make()
        defer { h.cleanUp() }
        h.blockedCount.set(99)
        await h.controller.start()
        expectFailed(h, containing: "protection could not be verified")
    }

    @Test func failsWhenTheServerRunsAsRoot() async {
        let world = FakeContainerWorld()
        world.privilegeOutput = "0 none"
        let h = SandboxHarness.make(world: world)
        defer { h.cleanUp() }
        await h.controller.start()
        expectFailed(h, containing: "protection could not be verified")
    }

    @Test func failsWhenTheFirewallIsNotLoaded() async {
        let world = FakeContainerWorld()
        world.failures["exec nft"] = SandboxProcessResult(status: 1, output: "No such file or directory\n")
        let h = SandboxHarness.make(world: world)
        defer { h.cleanUp() }
        await h.controller.start()
        expectFailed(h, containing: "firewall is not loaded")
    }

    @Test func failsWhenTheFirewallLetsEverythingOut() async {
        let world = FakeContainerWorld()
        let blocked = HostAddresses.normalized(SandboxHarness.hostAddresses)
        world.nftListingOverride = FakeContainerWorld.goodNftListing(blocked: blocked)
            .replacingOccurrences(of: "policy drop", with: "policy accept")
        let h = SandboxHarness.make(world: world)
        defer { h.cleanUp() }
        await h.controller.start()
        expectFailed(h, containing: "drop outbound traffic")
    }

    @Test func canStartAgainAfterAFailure() async {
        let h = SandboxHarness.make()
        defer { h.cleanUp() }
        h.blockedCount.set(99)
        await h.controller.start()
        #expect(sandboxFailureText(h.controller.state) != nil)
        h.blockedCount.set(nil)
        await h.controller.start()
        #expect(h.controller.state == .ready)
        #expect(h.world.count("run") == 2)
        #expect(h.world.runningContainers == [Self.second])
        await h.controller.stop()
    }

    // MARK: Containers left behind

    @Test func removesContainersLeftByACrash() async {
        let world = FakeContainerWorld()
        let leftRunning = SandboxController.containerNamePrefix + "deadbeef"
        let leftStopped = SandboxController.containerNamePrefix + "cafe0001"
        world.addContainer(leftRunning, running: true)
        world.addContainer(leftStopped, running: false)
        world.addContainer("someone-elses-container", running: true)
        let h = SandboxHarness.make(world: world)
        defer { h.cleanUp() }
        // A secret folder from the crashed run.
        let old = SandboxLocations.secretsDirectory(in: h.stateDirectory).appendingPathComponent("old")
        try? FileManager.default.createDirectory(at: old, withIntermediateDirectories: true)
        await h.controller.start()
        #expect(h.controller.state == .ready)
        #expect(Set(world.runningContainers) == [Self.first, "someone-elses-container"])
        #expect(world.stoppedContainers.isEmpty)
        #expect(h.secretFolders.count == 1)
        #expect(!h.secretFolders.contains("old"))
        await h.controller.stop()
        #expect(world.runningContainers == ["someone-elses-container"])
    }

    @Test func cleanUpOnLaunchRemovesLeftoversWithoutStarting() async {
        let world = FakeContainerWorld()
        world.addContainer(SandboxController.containerNamePrefix + "deadbeef", running: true)
        world.addContainer("someone-elses-container", running: true)
        let h = SandboxHarness.make(world: world)
        defer { h.cleanUp() }
        await h.controller.cleanUpLeftovers()
        #expect(world.runningContainers == ["someone-elses-container"])
        #expect(world.count("run") == 0)
        #expect(h.controller.state == .off)
    }

    @Test func cleanUpOnLaunchDoesNothingWithoutContainer() async {
        let world = FakeContainerWorld()
        world.addContainer(SandboxController.containerNamePrefix + "deadbeef", running: true)
        let h = SandboxHarness.make(world: world, isInstalled: false)
        defer { h.cleanUp() }
        await h.controller.cleanUpLeftovers()
        #expect(world.calls.isEmpty)
    }

    @Test func cleanUpOnLaunchLeavesTheSystemServiceAlone() async {
        let world = FakeContainerWorld()
        world.systemRunning = false
        let h = SandboxHarness.make(world: world)
        defer { h.cleanUp() }
        await h.controller.cleanUpLeftovers()
        #expect(world.count("system start") == 0)
        #expect(world.count("list") == 0)
    }

    // MARK: Two copies of TUFF

    @Test func theSecondCopyCannotUseTheSandbox() async {
        let one = SandboxHarness.make()
        defer { one.cleanUp() }
        await one.controller.start()
        #expect(one.controller.state == .ready)

        let two = SandboxHarness.make(stateDirectory: one.stateDirectory)
        await two.controller.start()
        #expect(sandboxFailureText(two.controller.state)?.contains("Another copy") == true)
        #expect(two.world.calls.isEmpty)

        // It must not touch the first copy's container either.
        await two.controller.cleanUpLeftovers()
        #expect(two.world.calls.isEmpty)
        #expect(one.world.runningContainers == [Self.first])

        await one.controller.stop()
        await two.controller.start()
        #expect(two.controller.state == .ready)
        await two.controller.stop()
    }

    // MARK: A container that will not go away

    @Test func reportsAContainerThatCannotBeConfirmedGone() async {
        let world = FakeContainerWorld()
        world.undeletable = [Self.first]
        let h = SandboxHarness.make(world: world)
        defer { h.cleanUp() }
        await h.controller.start()
        #expect(h.controller.state == .ready)
        await h.controller.stop()
        let message = sandboxFailureText(h.controller.state)
        #expect(message?.contains("could not be confirmed gone") == true)
        #expect(message?.contains(Self.first) == true)
        // The lock stays held, so no other copy starts next to a stray container.
        #expect(SandboxInstanceLock.acquire(at: SandboxLocations.lockFile(in: h.stateDirectory)) == nil)
        #expect(h.secretFolders.isEmpty)
        #expect(throws: SandboxError.notReady) { try h.controller.acquire() }

        // Once the container can be removed, this copy can try again.
        world.undeletable = []
        await h.controller.start()
        #expect(h.controller.state == .ready)
        #expect(world.runningContainers == [Self.second])
        await h.controller.stop()
        #expect(h.controller.state == .off)
        #expect(SandboxInstanceLock.acquire(at: SandboxLocations.lockFile(in: h.stateDirectory)) != nil)
    }

    // MARK: Quitting

    @Test func quittingStopsTheContainerAndFreesTheLock() async {
        let h = SandboxHarness.make()
        defer { h.cleanUp() }
        await h.controller.start()
        h.controller.stopWhenQuitting()
        #expect(h.world.runningContainers.isEmpty)
        #expect(h.world.stoppedContainers.isEmpty)
        #expect(h.secretFolders.isEmpty)
        let lock = SandboxInstanceLock.acquire(at: SandboxLocations.lockFile(in: h.stateDirectory))
        #expect(lock != nil)
        lock?.release()
    }

    @Test func quittingKeepsTheLockWhenTheContainerRemains() async {
        let world = FakeContainerWorld()
        world.undeletable = [Self.first]
        let h = SandboxHarness.make(world: world)
        defer { h.cleanUp() }
        await h.controller.start()
        h.controller.stopWhenQuitting()
        #expect(SandboxInstanceLock.acquire(at: SandboxLocations.lockFile(in: h.stateDirectory)) == nil)
    }

    @Test func quittingWithNothingRunningCallsNothing() {
        let h = SandboxHarness.make()
        defer { h.cleanUp() }
        h.controller.stopWhenQuitting()
        #expect(h.world.calls.isEmpty)
    }

    // MARK: Checking a running sandbox

    @Test func confirmRunningPassesWhileTheSandboxAnswers() async throws {
        let h = SandboxHarness.make()
        defer { h.cleanUp() }
        await h.controller.start()
        let before = h.probe.attemptCount
        try await h.controller.confirmRunning()
        #expect(h.probe.attemptCount == before + 1)
        #expect(h.controller.state == .ready)
        await h.controller.stop()
    }

    @Test func confirmRunningStopsASandboxThatNoLongerAnswers() async throws {
        let h = SandboxHarness.make()
        defer { h.cleanUp() }
        await h.controller.start()
        h.probe.handler = { _, _, _ in throw SandboxError.notHealthy("connection refused") }
        await #expect(throws: SandboxError.notHealthy("connection refused")) {
            try await h.controller.confirmRunning()
        }
        #expect(sandboxFailureText(h.controller.state) == "The sandbox stopped answering.")
        #expect(h.world.runningContainers.isEmpty)
        #expect(h.secretFolders.isEmpty)
        #expect(throws: SandboxError.notReady) { try h.controller.acquire() }
    }

    @Test func confirmRunningNeedsAReadySandbox() async {
        let h = SandboxHarness.make()
        defer { h.cleanUp() }
        await #expect(throws: SandboxError.notReady) { try await h.controller.confirmRunning() }
        #expect(h.probe.attemptCount == 0)
    }

    // MARK: Whose containers

    @Test func onlyContainersWithTheAppsOwnNamesAreRemoved() async {
        let world = FakeContainerWorld()
        let selftest = SandboxController.containerNamePrefix + "selftest"
        let longer = SandboxController.containerNamePrefix + "deadbeef0"
        world.addContainer(selftest, running: true)
        world.addContainer(longer, running: true)
        world.addContainer(SandboxController.containerNamePrefix + "deadbeef", running: true)
        let h = SandboxHarness.make(world: world)
        defer { h.cleanUp() }
        await h.controller.cleanUpLeftovers()
        #expect(Set(world.runningContainers) == [selftest, longer])
        #expect(SandboxController.isOurContainer(SandboxController.containerNamePrefix + "0a1b2c3d"))
        #expect(!SandboxController.isOurContainer(SandboxController.containerNamePrefix + "0A1B2C3D"))
        #expect(!SandboxController.isOurContainer("tuff-sandbox-web-"))
    }

    // MARK: Stopping at other moments

    @Test func stopDuringTheSystemServiceStartDoesNotLeaveItPreparing() async {
        let world = FakeContainerWorld()
        world.systemRunning = false
        let gate = SandboxGate()
        world.beforeSystemStart = { await gate.pass() }
        let h = SandboxHarness.make(world: world)
        defer { h.cleanUp() }
        let starting = Task { await h.controller.start() }
        await gate.waitUntilArrived()
        await h.controller.stop()
        await gate.open()
        await starting.value
        #expect(h.controller.state == .off)
        #expect(world.count("run") == 0)
    }

    @Test func quittingDuringAStartEndsTheStart() async {
        let world = FakeContainerWorld()
        let gate = SandboxGate()
        world.beforeRun = { await gate.pass() }
        let h = SandboxHarness.make(world: world)
        defer { h.cleanUp() }
        let starting = Task { await h.controller.start() }
        await gate.waitUntilArrived()
        h.controller.stopWhenQuitting()
        #expect(h.controller.state == .off)
        #expect(throws: SandboxError.notReady) { try h.controller.acquire() }
        await gate.open()
        await starting.value
        #expect(h.controller.state == .off)
        #expect(world.runningContainers.isEmpty)
        #expect(h.secretFolders.isEmpty)
    }

    @Test func aStartWaitsForALeftoverCleanUp() async {
        let h = SandboxHarness.make()
        defer { h.cleanUp() }
        let cleaning = Task { await h.controller.cleanUpLeftovers() }
        await Task.yield()
        await h.controller.start()
        await cleaning.value
        #expect(h.controller.state == .ready)
        await h.controller.stop()
    }

    @Test func failsWhenTheMacsAddressesCannotBeRead() async {
        let h = SandboxHarness.make(addresses: ["not an address"])
        defer { h.cleanUp() }
        await h.controller.start()
        expectFailed(h, containing: "could not read this Mac")
        #expect(h.world.count("run") == 0)
    }

    @Test func aStartThatWaitedDoesNotBringTheSandboxBackAfterAStop() async {
        let world = FakeContainerWorld()
        let gate = SandboxGate()
        world.beforeRun = { await gate.pass() }
        let h = SandboxHarness.make(world: world)
        defer { h.cleanUp() }
        let first = Task { await h.controller.start() }
        await gate.waitUntilArrived()
        let second = Task { await h.controller.start() }
        await Task.yield()
        await h.controller.stop()
        await gate.open()
        await first.value
        await second.value
        #expect(h.controller.state == .off)
        #expect(world.runningContainers.isEmpty)
        #expect(world.count("run") == 1)
    }

    @Test func twoStopsAtOnceBothWaitForTheStop() async {
        let h = SandboxHarness.make()
        defer { h.cleanUp() }
        await h.controller.start()
        let one = Task { await h.controller.stop() }
        let two = Task { await h.controller.stop() }
        await one.value
        await two.value
        #expect(h.controller.state == .off)
        #expect(h.world.runningContainers.isEmpty)
    }

    // MARK: Smaller pieces

    @Test func busyStates() {
        #expect(SandboxController.State.preparing("x").isBusy)
        #expect(SandboxController.State.starting.isBusy)
        #expect(SandboxController.State.stopping.isBusy)
        #expect(!SandboxController.State.off.isBusy)
        #expect(!SandboxController.State.ready.isBusy)
        #expect(!SandboxController.State.failed("x").isBusy)
    }

    @Test func launchArgumentsEndWithTheImageAndHaveNoAutoRemove() {
        let launch = SandboxLaunch(
            name: "tuff-sandbox-web-1", image: "img@sha256:aa",
            secretFolder: URL(fileURLWithPath: "/tmp/secret-folder"),
            blockedAddresses: ["10.0.0.1", "::1"], port: 51_000)
        let arguments = launch.arguments
        #expect(arguments.last == "img@sha256:aa")
        #expect(!arguments.contains("--rm"))
        #expect(arguments.contains("SANDBOX_BLOCKED_ADDRESSES=10.0.0.1,::1"))
        #expect(arguments.contains("127.0.0.1:51000:9100"))
        #expect(arguments.contains("type=bind,source=/tmp/secret-folder,target=/run/tuff-sandbox,readonly"))
    }
}
#endif
