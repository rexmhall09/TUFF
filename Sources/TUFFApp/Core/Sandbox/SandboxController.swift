#if os(macOS)
import Foundation
import Observation

/// Permission to send one request to the running sandbox. Take one with
/// `SandboxController.acquire()` and give it back with `release(_:)` when the
/// request is over; while any is out the sandbox is not stopped for being idle.
public struct SandboxLease: Sendable, Equatable {
    /// Which start of the sandbox this belongs to. After a stop it is stale.
    public let generation: Int
    let port: Int
    let key: Data
}

/// Everything the controller reaches outside itself, so tests can run it
/// with a fake `container`, a fake health answer and a fake clock.
struct SandboxDependencies: Sendable {
    var tool: ContainerTool
    var probe: any SandboxHealthProbing
    var pin: SandboxImagePin
    var stateDirectory: URL
    var addresses: @Sendable () -> [String]
    var freePort: @Sendable () throws -> Int
    var makeName: @Sendable () -> String
    var osSupported: @Sendable () -> Bool
    var sleep: @Sendable (Duration) async throws -> Void
    /// How long the sandbox may sit unused before it is stopped.
    var idleTimeout: Duration
    var healthAttempts: Int
    var healthInterval: Duration
    /// Seconds between the looks at a container when quitting during a start.
    var quitRetryDelay: TimeInterval = 1
    /// How long a signed answer from the sandbox (or the start's health check)
    /// counts as proof that the port is still the sandbox's. A request after
    /// a longer quiet time is preceded by a signed health check.
    var proofValidity: Duration = .seconds(5)
    var clock: @Sendable () -> ContinuousClock.Instant = { ContinuousClock.now }

    static func live() -> SandboxDependencies {
        SandboxDependencies(
            tool: ContainerTool(requiredTeamIdentifier: ContainerTool.appleTeamIdentifier),
            probe: SandboxHealthProbe(),
            pin: SandboxImagePin.fromEnvironment() ?? .release,
            stateDirectory: SandboxLocations.defaultStateDirectory(),
            addresses: { HostAddresses.current() },
            freePort: { try LoopbackPort.findFree() },
            makeName: { SandboxController.containerNamePrefix + String(UUID().uuidString.prefix(8)).lowercased() },
            osSupported: {
                ProcessInfo.processInfo.isOperatingSystemAtLeast(
                    OperatingSystemVersion(majorVersion: 26, minorVersion: 0, patchVersion: 0))
            },
            sleep: { try await Task.sleep(for: $0) },
            idleTimeout: .seconds(600),
            healthAttempts: 60,
            healthInterval: .milliseconds(500))
    }
}

/// The arguments of `container run` for the web sandbox. The flags are the
/// ones `Scripts/sandbox_selftest.sh start` uses and checks: a read-only
/// root, no capabilities except the four the entrypoint needs to load the
/// firewall and drop privileges, limits, the secret folder mounted read-only,
/// and the one published port bound to loopback.
struct SandboxLaunch: Equatable, Sendable {
    static let serverPort = 9100
    static let secretMount = "/run/tuff-sandbox"

    let name: String
    let image: String
    let secretFolder: URL
    let blockedAddresses: [String]
    let port: Int

    var arguments: [String] {
        [
            "--detach", "--name", name,
            "--read-only", "--tmpfs", "/tmp",
            "--cap-drop", "ALL",
            "--cap-add", "NET_ADMIN", "--cap-add", "SETUID",
            "--cap-add", "SETGID", "--cap-add", "SETPCAP",
            "--ulimit", "nproc=512", "--cpus", "2", "--memory", "1G",
            "--mount", "type=bind,source=\(secretFolder.path),target=\(Self.secretMount),readonly",
            "--env", "SANDBOX_BLOCKED_ADDRESSES=\(blockedAddresses.joined(separator: ","))",
            "--publish", "127.0.0.1:\(port):\(Self.serverPort)",
            image,
        ]
    }
}

/// Starts, checks and stops the web sandbox, a small Linux VM made by Apple's
/// `container` tool that fetches web pages and search results for TUFF.
///
/// The app owns one (`AppModel.sandbox`). It is started when the person turns
/// on "Run web search and page reads in a sandbox", at app launch if that is
/// on, and before an answer that uses the web if it is on and the sandbox is
/// not running. It never switches itself on.
///
/// A start does these steps in order, and stops at the first that fails
/// (there is no fallback and nothing is left running):
/// 1. macOS 26 or newer, and this is the only copy of TUFF using the sandbox;
/// 2. `container` is installed at its fixed path, validly signed and recent
///    enough, and its system service runs;
/// 3. the image with the digest pinned in TUFF is there;
/// 4. containers left behind by a crashed run are removed;
/// 5. a fresh secret and a free loopback port are made, and the container is
///    started with the Mac's own addresses blocked;
/// 6. the sandbox answers a signed `/health`, its firewall is loaded with
///    those addresses blocked, and its server runs without privileges.
@MainActor @Observable
public final class SandboxController {
    public enum State: Equatable, Sendable {
        case off
        /// Before the container runs, with a sentence for the state row.
        case preparing(String)
        case starting
        case ready
        case stopping
        case failed(String)

        public var isBusy: Bool {
            switch self {
            case .preparing, .starting, .stopping: true
            case .off, .ready, .failed: false
            }
        }
    }

    nonisolated public static let containerNamePrefix = "tuff-sandbox-web-"

    public private(set) var state: State = .off
    /// Increases at every start and every stop, so work that started under
    /// an earlier one can tell it is out of date.
    public private(set) var generation = 0
    /// Leases taken and not yet released.
    public private(set) var requestsInFlight = 0
    /// When the sandbox of the current generation last proved, with a signed
    /// answer, that it is the one holding its port.
    @ObservationIgnored private var lastProof: ContinuousClock.Instant?

    @ObservationIgnored private let dependencies: SandboxDependencies
    @ObservationIgnored private var startTask: Task<Void, Never>?
    @ObservationIgnored private var cleanupTask: Task<Void, Never>?
    @ObservationIgnored private var stopTask: Task<Void, Never>?
    @ObservationIgnored private var startToken = 0
    @ObservationIgnored private var idleTask: Task<Void, Never>?
    @ObservationIgnored private var lock: SandboxInstanceLock?
    @ObservationIgnored private var secret: SandboxSecret?
    @ObservationIgnored private var containerName: String?
    @ObservationIgnored private var activePort = 0
    @ObservationIgnored private var activeKey = Data()

    public convenience init() {
        self.init(dependencies: .live())
    }

    init(dependencies: SandboxDependencies) {
        self.dependencies = dependencies
    }

    // MARK: Start

    /// Starts the sandbox. Does nothing if it is ready or busy; a call made
    /// while another start runs waits for that start. On any failure the
    /// state is `.failed` with the reason, and nothing is left running.
    public func start() async {
        await cleanupTask?.value
        if let stopping = stopTask { await stopping.value }
        if let running = startTask {
            // A call made while a start runs only waits for it. If a stop
            // ended that start, this call must not bring the sandbox back.
            await running.value
            return
        }
        switch state {
        case .off, .failed: break
        case .preparing, .starting, .ready, .stopping: return
        }
        generation += 1
        let mine = generation
        startToken += 1
        let token = startToken
        state = .preparing("Checking Apple container")
        let task = Task {
            await self.performStart(generation: mine)
            if self.startToken == token { self.startTask = nil }
        }
        startTask = task
        await task.value
    }

    private func performStart(generation mine: Int) async {
        var createdName: String?
        var createdSecret: SandboxSecret?
        do {
            guard dependencies.osSupported() else { throw SandboxError.unsupportedSystem }
            try Self.prepare(directory: dependencies.stateDirectory)
            // A stop that could not confirm the container gone keeps the
            // lock; a second try then uses it instead of failing against it.
            if lock == nil {
                guard let acquired = SandboxInstanceLock.acquire(
                    at: SandboxLocations.lockFile(in: dependencies.stateDirectory)) else {
                    throw SandboxError.anotherInstanceRunning
                }
                lock = acquired
            }
            let tool = dependencies.tool
            try await tool.verify()
            try ensureCurrent(mine)
            let systemRunning = await tool.systemIsRunning()
            try ensureCurrent(mine)
            if !systemRunning {
                state = .preparing("Starting Apple container's service")
                try await tool.startSystem()
                try ensureCurrent(mine)
            }
            state = .preparing("Checking the sandbox image")
            try await SandboxImageProvisioner(tool: tool).ensure(dependencies.pin) { message in
                Task { @MainActor [weak self] in self?.note(message, generation: mine) }
            }
            try ensureCurrent(mine)
            try await removeLeftovers(tool: tool)
            try ensureCurrent(mine)

            state = .starting
            let blocked = HostAddresses.normalized(dependencies.addresses())
            guard !blocked.isEmpty else { throw SandboxError.hostAddressesUnknown }
            guard blocked.count <= HostAddresses.maximumCount else { throw SandboxError.tooManyAddresses }
            let made = try SandboxSecret.create(
                under: SandboxLocations.secretsDirectory(in: dependencies.stateDirectory))
            createdSecret = made
            secret = made
            guard made.isMountable else { throw SandboxError.unsafeMountPath }
            let port = try dependencies.freePort()
            let name = dependencies.makeName()
            createdName = name
            containerName = name
            let launch = SandboxLaunch(name: name, image: dependencies.pin.runReference,
                                       secretFolder: made.folder, blockedAddresses: blocked, port: port)
            try await tool.runDetached(arguments: launch.arguments)
            try ensureCurrent(mine)

            let health = try await waitUntilHealthy(port: port, key: made.key, name: name, generation: mine)
            // The signed health answer is the proof; the checks below take time.
            let provedAt = dependencies.clock()
            try ensureCurrent(mine)
            try await SandboxProtection.verify(tool: tool, container: name, blocked: blocked, health: health)
            try ensureCurrent(mine)

            activePort = port
            activeKey = made.key
            lastProof = provedAt
            state = .ready
            scheduleIdleStop()
        } catch {
            // A stop that came in meanwhile has set the state; either way
            // whatever this start made is removed. The removal runs in a task
            // of its own: a cancelled start must still be able to run it.
            let name = createdName
            let made = createdSecret
            let leftover = await Task { await self.tearDown(extraName: name, extraSecret: made) }.value
            guard generation == mine else {
                // The stop that cut this start short has already reported
                // "off". If this start's container could not be removed,
                // that must not stay hidden.
                if let leftover, state == .off, let name {
                    containerName = name
                    if lock == nil {
                        lock = SandboxInstanceLock.acquire(
                            at: SandboxLocations.lockFile(in: dependencies.stateDirectory))
                    }
                    state = .failed(leftover)
                }
                return
            }
            state = .failed(Self.message(for: error, leftover: leftover))
        }
    }

    private func note(_ message: String, generation mine: Int) {
        guard generation == mine, case .preparing = state else { return }
        state = .preparing(message)
    }

    private func ensureCurrent(_ mine: Int) throws {
        if generation != mine { throw SandboxError.stoppedWhileStarting }
    }

    private func waitUntilHealthy(port: Int, key: Data, name: String,
                                  generation mine: Int) async throws -> SandboxHealth {
        var lastProblem = "no answer yet"
        for attempt in 0..<max(1, dependencies.healthAttempts) {
            try ensureCurrent(mine)
            do {
                return try await dependencies.probe.health(port: port, key: key)
            } catch SandboxError.notHealthy(let detail) {
                lastProblem = detail
            } catch let error as SandboxError {
                lastProblem = error.description
            } catch {
                lastProblem = SandboxText.oneLine(error.localizedDescription)
            }
            // A container that ended, such as one whose entrypoint refused
            // to start, never answers: look at its log instead of waiting.
            if attempt % 4 == 3,
               let running = try? await dependencies.tool.containerIDs(all: false),
               !running.contains(name) {
                let log = await dependencies.tool.logTail(name)
                throw SandboxError.notHealthy(log.isEmpty ? "the container stopped" : "the container stopped (\(log))")
            }
            try await dependencies.sleep(dependencies.healthInterval)
        }
        throw SandboxError.notHealthy(lastProblem)
    }

    /// Removes containers an earlier run of TUFF left behind. Only called
    /// while this process holds the instance lock, so none has an owner.
    private func removeLeftovers(tool: ContainerTool) async throws {
        for id in try await tool.containerIDs(all: true) where Self.isOurContainer(id) {
            await tool.stopAndDelete(id)
        }
        SandboxSecret.removeAll(under: SandboxLocations.secretsDirectory(in: dependencies.stateDirectory))
    }

    /// Removes containers and secret folders a crashed run of TUFF left
    /// behind, when nothing is running. For app launch, so a crash does not
    /// leave a VM running while the setting is off. Does nothing if another
    /// copy of TUFF holds the sandbox or `container` is not usable.
    public func cleanUpLeftovers() async {
        guard state == .off, startTask == nil, cleanupTask == nil else { return }
        let task = Task { await self.performCleanUp() }
        cleanupTask = task
        await task.value
        cleanupTask = nil
    }

    /// A `start()` made meanwhile waits for this, so the two never compete
    /// for the instance lock.
    private func performCleanUp() async {
        try? Self.prepare(directory: dependencies.stateDirectory)
        guard let held = SandboxInstanceLock.acquire(
            at: SandboxLocations.lockFile(in: dependencies.stateDirectory)) else { return }
        defer { held.release() }
        let tool = dependencies.tool
        guard (try? await tool.verify()) != nil, await tool.systemIsRunning() else { return }
        try? await removeLeftovers(tool: tool)
    }

    /// Only containers TUFF named itself: the prefix and 8 hexadecimal
    /// digits. A container someone started by hand with a similar name, such
    /// as the self-test's, is left alone.
    nonisolated static func isOurContainer(_ id: String) -> Bool {
        id.wholeMatch(of: #/tuff-sandbox-web-[0-9a-f]{8}/#) != nil
    }

    // MARK: Stop

    /// Stops the sandbox and checks that its container is gone. If a start
    /// is under way it ends with `stoppedWhileStarting`.
    public func stop() async {
        if let stopping = stopTask {
            await stopping.value
            return
        }
        if state == .off { return }
        generation += 1
        requestsInFlight = 0
        state = .stopping
        // A start under way ends: its command is cut short, and it removes
        // what it made. The removal runs in a task of its own, so it also
        // works when the caller was cancelled.
        startTask?.cancel()
        let task = Task {
            let leftover = await self.tearDown(extraName: nil, extraSecret: nil)
            self.state = leftover.map { State.failed($0) } ?? State.off
            self.stopTask = nil
        }
        stopTask = task
        await task.value
    }

    /// Seconds after which no further looks at a container are made while
    /// quitting (one look can still take up to 15 s on a stuck service).
    private static let quitBudget: TimeInterval = 6

    /// Stops the sandbox while the app quits, waiting a few seconds for it.
    public func stopWhenQuitting() {
        // Whatever is under way ends, and nothing can take a lease any more.
        generation += 1
        requestsInFlight = 0
        startTask?.cancel()
        idleTask?.cancel()
        idleTask = nil
        activePort = 0
        activeKey = Data()
        let wasStarting = startTask != nil
        let began = ProcessInfo.processInfo.systemUptime
        var gone = containerName.map { dependencies.tool.stopAndDeleteBlocking($0) } ?? true
        if wasStarting, let name = containerName {
            // `container run` may still be creating the container: look
            // again a few times, so one "not there yet" is not believed.
            for _ in 0..<3 {
                // Quitting must not hang on a stuck container service: once
                // the first look has taken this long, no more are made and the
                // next launch removes what is left.
                guard ProcessInfo.processInfo.systemUptime - began < Self.quitBudget else { break }
                if dependencies.quitRetryDelay > 0 { Thread.sleep(forTimeInterval: dependencies.quitRetryDelay) }
                gone = dependencies.tool.stopAndDeleteBlocking(name)
            }
        }
        secret?.remove()
        secret = nil
        guard gone else {
            // Keep the lock and the name: the next start of TUFF removes it.
            state = .failed("The sandbox's container could not be confirmed gone.")
            return
        }
        containerName = nil
        lock?.release()
        lock = nil
        state = .off
    }

    /// Everything a start made: its container, its secret folder and the
    /// instance lock. Safe to call twice. Returns what to tell the person
    /// when the container could not be confirmed gone, in which case the
    /// lock and the container's name are kept for another try.
    private func tearDown(extraName: String?, extraSecret: SandboxSecret?) async -> String? {
        idleTask?.cancel()
        idleTask = nil
        var names = Set<String>()
        if let containerName { names.insert(containerName) }
        if let extraName { names.insert(extraName) }
        var problem: String?
        let tool = dependencies.tool
        for name in names.sorted() {
            await tool.stopAndDelete(name)
            if !(await waitUntilGone(name)) {
                problem = "The sandbox's container \(name) could not be confirmed gone. "
                    + "Run `container list --all` in Terminal to look."
            }
        }
        secret?.remove()
        extraSecret?.remove()
        secret = nil
        activeKey = Data()
        activePort = 0
        if problem == nil {
            containerName = nil
            lock?.release()
            lock = nil
        }
        return problem
    }

    private func waitUntilGone(_ name: String) async -> Bool {
        for attempt in 1...5 {
            if let ids = try? await dependencies.tool.containerIDs(all: true), !ids.contains(name) {
                return true
            }
            if attempt < 5 { try? await dependencies.sleep(.milliseconds(500)) }
        }
        return false
    }

    // MARK: Use

    /// Takes a lease for one request. Throws `stillStarting` or `notReady`
    /// unless the sandbox is ready: nothing here starts it, and there is no
    /// other path to the web.
    public func acquire() throws -> SandboxLease {
        switch state {
        case .ready: break
        case .preparing, .starting: throw SandboxError.stillStarting
        case .off, .stopping, .failed: throw SandboxError.notReady
        }
        idleTask?.cancel()
        idleTask = nil
        requestsInFlight += 1
        return SandboxLease(generation: generation, port: activePort, key: activeKey)
    }

    /// Asks the running sandbox, with a signed request, whether it still
    /// answers. Call it before sending a request that carries a secret (a
    /// search key): if the sandbox has died, something else may hold its port.
    /// On failure the sandbox is stopped, the state is `.failed`, and this
    /// throws.
    public func confirmRunning() async throws {
        guard state == .ready else { throw SandboxError.notReady }
        try await probeRunning(port: activePort, key: activeKey, generation: generation)
    }

    /// The same check for the sandbox a lease belongs to: it throws
    /// `notReady` if that sandbox has been stopped since, and asks the
    /// lease's own port with the lease's own key.
    public func confirmRunning(_ lease: SandboxLease) async throws {
        guard isCurrent(lease) else { throw SandboxError.notReady }
        try await probeRunning(port: lease.port, key: lease.key, generation: lease.generation)
    }

    /// Whether the sandbox has not proved itself for longer than
    /// `proofValidity`: after that, something else may hold its port, so the
    /// next request is preceded by a signed health check.
    public func needsProof(_ lease: SandboxLease) -> Bool {
        guard isCurrent(lease), let proof = lastProof else { return true }
        return dependencies.clock() - proof > dependencies.proofValidity
    }

    /// Records that the sandbox of the lease has just given a signed answer.
    public func noteAnswered(_ lease: SandboxLease) {
        guard isCurrent(lease) else { return }
        lastProof = dependencies.clock()
    }

    /// Whether the lease belongs to the sandbox that is ready right now.
    public func isCurrent(_ lease: SandboxLease) -> Bool {
        state == .ready && lease.generation == generation
    }

    private func probeRunning(port: Int, key: Data, generation mine: Int) async throws {
        do {
            do {
                _ = try await dependencies.probe.health(port: port, key: key)
                if generation == mine, state == .ready { lastProof = dependencies.clock() }
            } catch {
                // One busy moment is not a dead sandbox: ask once more.
                try Task.checkCancellation()
                try? await dependencies.sleep(.milliseconds(500))
                _ = try await dependencies.probe.health(port: port, key: key)
                if generation == mine, state == .ready { lastProof = dependencies.clock() }
            }
        } catch {
            if error is CancellationError || Task.isCancelled { throw CancellationError() }
            if generation == mine, state == .ready {
                await stop()
                if state == .off { state = .failed("The sandbox stopped answering.") }
            }
            throw (error as? SandboxError) ?? SandboxError.notHealthy(
                SandboxText.oneLine(error.localizedDescription))
        }
    }

    /// Gives a lease back. A lease from before a stop is ignored.
    public func release(_ lease: SandboxLease) {
        guard lease.generation == generation, requestsInFlight > 0 else { return }
        requestsInFlight -= 1
        if requestsInFlight == 0, state == .ready { scheduleIdleStop() }
    }

    private func scheduleIdleStop() {
        idleTask?.cancel()
        let mine = generation
        idleTask = Task { [weak self] in
            guard let self else { return }
            try? await self.dependencies.sleep(self.dependencies.idleTimeout)
            guard !Task.isCancelled else { return }
            await self.idleTimerFired(generation: mine)
        }
    }

    private func idleTimerFired(generation mine: Int) async {
        guard mine == generation, state == .ready, requestsInFlight == 0 else { return }
        await stop()
    }

    // MARK: Helpers

    private static func prepare(directory: URL) throws {
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true,
            attributes: [.posixPermissions: 0o700])
    }

    private static func message(for error: Error, leftover: String?) -> String {
        let text = (error as? SandboxError)?.description
            ?? SandboxText.oneLine(error.localizedDescription)
        guard let leftover else { return text }
        return text + " " + leftover
    }
}
#endif
