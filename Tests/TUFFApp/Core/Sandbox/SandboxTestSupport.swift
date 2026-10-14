#if os(macOS)
import Foundation
@testable import TUFFAppCore

/// A pretend `container` command line and `codesign`, with just enough state
/// to start, list, stop and inspect containers. It answers the commands
/// `ContainerTool`, `SandboxImageProvisioner`, `SandboxProtection` and
/// `SandboxController` run, records every call, and can be told to fail a
/// command or to wait before one. Nothing here starts a real process.
final class FakeContainerWorld: SandboxProcessRunning, @unchecked Sendable {
    struct Call: Equatable, Sendable {
        let executable: String
        let arguments: [String]
    }

    static let digest = "sha256:" + String(repeating: "ab", count: 32)
    static let otherDigest = "sha256:" + String(repeating: "cd", count: 32)
    static let containerPath = "/usr/local/bin/container"

    private let lock = NSLock()
    private var recorded: [Call] = []
    private var running: [String] = []
    private var stopped: [String] = []

    // MARK: What the pretend tools say. Set these before the code under test runs.

    var versionOutput = "container CLI version 1.5.0 (build: release, commit: d265d66)"
    var codesignStatus: Int32 = 0
    var codesignDetails = "Identifier=com.apple.container\nTeamIdentifier=TESTTEAM01\n"
    var systemRunning = true
    /// The digests `image inspect` prints for the pinned image, or nil when
    /// the image is not in the local store.
    var imageDigests: [String]? = [FakeContainerWorld.digest]
    /// What a successful pull leaves in the store.
    var digestsAfterPull: [String]? = [FakeContainerWorld.digest]
    var logsOutput = ""
    var privilegeOutput = "10001 none"
    /// The listing `nft -nn list table inet tuff_sandbox_web` prints. When nil
    /// it is built from the blocked addresses of the last `container run`.
    var nftListingOverride: String?
    /// Results to give instead of the normal answer, by command name:
    /// `"version"`, `"codesign"`, `"system status"`, `"system start"`, `"list"`, `"run"`, `"stop"`, `"delete"`,
    /// `"exec nft"`, `"exec python3"`, `"image inspect"`, `"image pull"`, `"logs"`.
    var failures: [String: SandboxProcessResult] = [:]
    /// Containers that stay in the list after `stop` and `delete`.
    var undeletable: Set<String> = []
    /// Awaited inside `container run`, before the container exists.
    var beforeRun: (@Sendable () async -> Void)?
    /// Awaited inside `container system start`.
    var beforeSystemStart: (@Sendable () async -> Void)?
    /// Whether a container that was started stays running. When false it
    /// ends at once, as one whose entrypoint refuses to start would.
    var containersStayUp = true

    // MARK: Look at what happened

    var calls: [Call] { lock.withLock { recorded } }

    /// The arguments of every call to the container tool, without its path.
    var containerCalls: [[String]] {
        calls.filter { $0.executable == Self.containerPath }.map(\.arguments)
    }

    func count(_ command: String) -> Int {
        calls.filter { Self.name(of: $0) == command }.count
    }

    var runningContainers: [String] { lock.withLock { running } }
    var stoppedContainers: [String] { lock.withLock { stopped } }

    /// Puts containers into the pretend system, as a crashed run would.
    func addContainer(_ name: String, running isRunning: Bool) {
        lock.withLock {
            if isRunning { running.append(name) } else { stopped.append(name) }
        }
    }

    /// The arguments of the last `container run`.
    var lastRunArguments: [String]? {
        containerCalls.last(where: { $0.first == "run" })
    }

    /// The value after `flag` in the last `container run`.
    func lastRunValue(after flag: String) -> String? {
        guard let arguments = lastRunArguments, let index = arguments.firstIndex(of: flag),
              index + 1 < arguments.count else { return nil }
        return arguments[index + 1]
    }

    /// The addresses of `--env SANDBOX_BLOCKED_ADDRESSES=…` in the last run.
    var lastRunBlockedAddresses: [String] {
        let prefix = "SANDBOX_BLOCKED_ADDRESSES="
        guard let arguments = lastRunArguments,
              let value = arguments.first(where: { $0.hasPrefix(prefix) }) else { return [] }
        return value.dropFirst(prefix.count).split(separator: ",").map(String.init)
    }

    /// The port of `--publish 127.0.0.1:PORT:9100` in the last run.
    var lastRunPort: Int? {
        guard let publish = lastRunValue(after: "--publish") else { return nil }
        let parts = publish.split(separator: ":")
        return parts.count == 3 ? Int(parts[1]) : nil
    }

    // MARK: SandboxProcessRunning

    func run(executable: String, arguments: [String],
             timeout: TimeInterval) async throws -> SandboxProcessResult {
        let call = Call(executable: executable, arguments: arguments)
        lock.withLock { recorded.append(call) }
        let command = Self.name(of: call)
        if command == "run", let gate = beforeRun { await gate() }
        if command == "system start", let gate = beforeSystemStart { await gate() }
        if let failure = lock.withLock({ failures[command] }) { return failure }
        return answer(command, call)
    }

    func runBlocking(executable: String, arguments: [String],
                     timeout: TimeInterval) -> SandboxProcessResult? {
        let call = Call(executable: executable, arguments: arguments)
        lock.withLock { recorded.append(call) }
        let command = Self.name(of: call)
        if let failure = lock.withLock({ failures[command] }) { return failure }
        return answer(command, call)
    }

    /// `"stop"`, `"exec nft"`, `"image pull"` and so on.
    static func name(of call: Call) -> String {
        if call.executable == "/usr/bin/codesign" { return "codesign" }
        guard let first = call.arguments.first else { return "" }
        switch first {
        case "--version": return "version"
        case "system", "image":
            return "\(first) \(call.arguments.dropFirst().first ?? "")"
        case "exec":
            return "exec \(call.arguments.dropFirst(2).first ?? "")"
        default: return first
        }
    }

    private func answer(_ command: String, _ call: Call) -> SandboxProcessResult {
        let arguments = call.arguments
        switch command {
        case "codesign":
            if arguments.first == "-dv" {
                return SandboxProcessResult(status: codesignStatus, output: codesignDetails)
            }
            return SandboxProcessResult(status: codesignStatus,
                                        output: codesignStatus == 0 ? "" : "code object is not signed at all\n")
        case "version":
            return SandboxProcessResult(status: 0, output: versionOutput + "\n")
        case "system status":
            return SandboxProcessResult(status: systemRunning ? 0 : 1, output: "")
        case "system start":
            lock.withLock { systemRunning = true }
            return SandboxProcessResult(status: 0, output: "")
        case "list":
            return lock.withLock {
                var names = running
                if arguments.contains("--all") { names += stopped }
                if arguments.contains("--quiet") {
                    return SandboxProcessResult(status: 0, output: names.map { $0 + "\n" }.joined())
                }
                let header = "ID  IMAGE  OS  ARCH  STATE  ADDR\n"
                let rows = names.map { "\($0)  img  linux  arm64  running  192.168.64.2\n" }
                return SandboxProcessResult(status: 0, output: header + rows.joined())
            }
        case "run":
            guard let index = arguments.firstIndex(of: "--name"), index + 1 < arguments.count else {
                return SandboxProcessResult(status: 1, output: "no name\n")
            }
            let name = arguments[index + 1]
            lock.withLock {
                if containersStayUp { running.append(name) } else { stopped.append(name) }
            }
            return SandboxProcessResult(status: 0, output: name + "\n")
        case "stop":
            let name = arguments.dropFirst().first ?? ""
            lock.withLock {
                guard !undeletable.contains(name), let index = running.firstIndex(of: name) else { return }
                running.remove(at: index)
                stopped.append(name)
            }
            return SandboxProcessResult(status: 0, output: "")
        case "delete":
            let name = arguments.dropFirst().first ?? ""
            lock.withLock {
                guard !undeletable.contains(name) else { return }
                running.removeAll { $0 == name }
                stopped.removeAll { $0 == name }
            }
            return SandboxProcessResult(status: 0, output: "")
        case "exec nft":
            return SandboxProcessResult(status: 0, output: nftListing())
        case "exec python3":
            return SandboxProcessResult(status: 0, output: privilegeOutput + "\n")
        case "image inspect":
            guard let digests = lock.withLock({ imageDigests }) else {
                return SandboxProcessResult(status: 1, output: "image not found\n")
            }
            return SandboxProcessResult(status: 0, output: Self.inspectOutput(digests))
        case "image pull":
            lock.withLock { imageDigests = digestsAfterPull }
            return SandboxProcessResult(status: 0, output: "pulled\n")
        case "logs":
            return SandboxProcessResult(status: 0, output: logsOutput)
        default:
            return SandboxProcessResult(status: 0, output: "")
        }
    }

    static func inspectOutput(_ digests: [String]) -> String {
        // The real layout: the first digest is the image index's
        // (`configuration.descriptor`), the others are platform manifests.
        let index = digests.first.map { "{\"digest\": \"\($0)\"}" } ?? "{}"
        let variants = digests.dropFirst().map { "{\"digest\": \"\($0)\"}" }.joined(separator: ", ")
        return "[{\"configuration\": {\"descriptor\": \(index)}, \"id\": \"image\", "
            + "\"variants\": [\(variants)]}]\n"
    }

    private func nftListing() -> String {
        if let override = lock.withLock({ nftListingOverride }) { return override }
        return Self.goodNftListing(blocked: lastRunBlockedAddresses)
    }

    /// What `nft -nn list table inet tuff_sandbox_web` prints for the table
    /// `Sandbox/firewall-web.nft` makes, with the entrypoint's additions: the
    /// blocked addresses and a DNS exception at the top of the output chain.
    static func goodNftListing(blocked: [String]) -> String {
        let four = blocked.filter { !$0.contains(":") }.joined(separator: ", ")
        let six = blocked.filter { $0.contains(":") }.joined(separator: ", ")
        return """
        table inet tuff_sandbox_web {
        \tset private4 {
        \t\ttype ipv4_addr
        \t\tflags interval
        \t\telements = { 0.0.0.0/8, 10.0.0.0/8, 100.64.0.0/10, 127.0.0.0/8,
        \t\t\t     169.254.0.0/16, 172.16.0.0/12, 192.168.0.0/16, 224.0.0.0/3 }
        \t}

        \tset blocked4 {
        \t\ttype ipv4_addr
        \(four.isEmpty ? "" : "\t\telements = { \(four) }")
        \t}

        \tset blocked6 {
        \t\ttype ipv6_addr
        \(six.isEmpty ? "" : "\t\telements = { \(six) }")
        \t}

        \tset refused6 {
        \t\ttype ipv6_addr
        \t\tflags interval
        \t\telements = { 2001::/32, 2001:db8::/32, 2002::/16 }
        \t}

        \tchain output {
        \t\ttype filter hook output priority filter; policy drop;
        \t\tip daddr 192.168.64.1 udp dport 53 accept
        \t\tip daddr 192.168.64.1 tcp dport 53 accept
        \t\toifname "lo" accept
        \t\tct state established,related accept
        \t\tip daddr @blocked4 meta l4proto tcp reject with tcp reset
        \t\tip daddr @blocked4 reject
        \t\tip6 daddr @blocked6 meta l4proto tcp reject with tcp reset
        \t\tip6 daddr @blocked6 reject
        \t\tip daddr @private4 meta l4proto tcp reject with tcp reset
        \t\tip daddr @private4 reject
        \t\tmeta nfproto ipv4 accept
        \t\tip6 daddr @refused6 meta l4proto tcp reject with tcp reset
        \t\tip6 daddr @refused6 reject
        \t\tip6 daddr 2000::/3 accept
        \t\tmeta l4proto tcp reject with tcp reset
        \t\treject
        \t}

        \tchain input {
        \t\ttype filter hook input priority filter; policy drop;
        \t\tiifname "lo" accept
        \t\tct state established,related accept
        \t\tip saddr 192.168.64.1 tcp dport 9100 accept
        \t}
        }

        """
    }
}

/// A health probe that answers from a closure.
final class FakeHealthProbe: SandboxHealthProbing, @unchecked Sendable {
    typealias Handler = @Sendable (_ port: Int, _ key: Data, _ attempt: Int) async throws -> SandboxHealth

    private let lock = NSLock()
    private var attempts = 0
    private var seen: [(port: Int, key: Data)] = []
    var handler: Handler

    init(handler: @escaping Handler) { self.handler = handler }

    /// Answers like a healthy sandbox that was given `blocked` addresses.
    static func healthy(blocked: @escaping @Sendable () -> Int) -> FakeHealthProbe {
        FakeHealthProbe { _, _, _ in
            SandboxHealth(profile: "web", api: 1, blockedAddresses: blocked())
        }
    }

    var attemptCount: Int { lock.withLock { attempts } }
    var requests: [(port: Int, key: Data)] { lock.withLock { seen } }

    func health(port: Int, key: Data) async throws -> SandboxHealth {
        let attempt = lock.withLock { () -> Int in
            attempts += 1
            seen.append((port, key))
            return attempts
        }
        return try await handler(port, key, attempt)
    }
}

/// A clock for the controller. Short sleeps (the health check's interval)
/// return at once; long sleeps (the idle timeout) wait until `fireIdle()`.
final class FakeIdleClock: @unchecked Sendable {
    private let lock = NSLock()
    private var waiters: [UUID: CheckedContinuation<Void, Error>] = [:]

    var waitingCount: Int { lock.withLock { waiters.count } }

    func sleep(_ duration: Duration) async throws {
        if duration < .seconds(60) {
            await Task.yield()
            return
        }
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                lock.withLock { waiters[id] = continuation }
                if Task.isCancelled, let cancelled = lock.withLock({ waiters.removeValue(forKey: id) }) {
                    cancelled.resume(throwing: CancellationError())
                }
            }
        } onCancel: {
            let waiting = lock.withLock { waiters.removeValue(forKey: id) }
            waiting?.resume(throwing: CancellationError())
        }
    }

    /// Lets every waiting idle timeout run out.
    func fireIdle() {
        let all = lock.withLock { () -> [CheckedContinuation<Void, Error>] in
            let values = Array(waiters.values)
            waiters.removeAll()
            return values
        }
        for waiting in all { waiting.resume() }
    }
}

/// The controller's clock for "how long since the sandbox last proved itself".
/// It stands still until a test moves it.
final class FakeProofClock: @unchecked Sendable {
    private let lock = NSLock()
    private let start = ContinuousClock.now
    private var offset = Duration.zero

    var now: ContinuousClock.Instant { lock.withLock { start + offset } }

    func advance(by duration: Duration) { lock.withLock { offset += duration } }
}

/// A controller wired to the fakes, with its own temporary state folder.
@MainActor
struct SandboxHarness {
    let world: FakeContainerWorld
    let probe: FakeHealthProbe
    let clock: FakeIdleClock
    let proofClock: FakeProofClock
    let stateDirectory: URL
    let controller: SandboxController
    let blockedCount: BlockedCount

    /// How many addresses the fake sandbox reports as blocked. Set to a
    /// different number to make the protection check fail.
    final class BlockedCount: @unchecked Sendable {
        private let lock = NSLock()
        private var override: Int?
        func set(_ value: Int?) { lock.withLock { override = value } }
        func value(default fallback: Int) -> Int { lock.withLock { override ?? fallback } }
    }

    static let hostAddresses = ["127.0.0.1", "::1", "192.168.1.20", "fe80::1%en0"]

    static func make(world: FakeContainerWorld = FakeContainerWorld(),
                     addresses: [String] = SandboxHarness.hostAddresses,
                     pin: SandboxImagePin = SandboxImagePin(
                         reference: "registry.example/tuff-sandbox", digest: FakeContainerWorld.digest,
                         source: .registry),
                     stateDirectory: URL? = nil,
                     osSupported: Bool = true,
                     requiredTeamIdentifier: String? = nil,
                     isInstalled: Bool = true,
                     freePort: Int = 49_152,
                     healthAttempts: Int = 8) -> SandboxHarness {
        let directory = stateDirectory ?? FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-sandbox-test-\(UUID().uuidString)", isDirectory: true)
        let clock = FakeIdleClock()
        let proofClock = FakeProofClock()
        let blockedCount = BlockedCount()
        let probe = FakeHealthProbe.healthy(blocked: { [world] in
            blockedCount.value(default: world.lastRunBlockedAddresses.count)
        })
        let counter = NameCounter()
        let tool = ContainerTool(
            runner: world, path: FakeContainerWorld.containerPath,
            isExecutable: { _ in isInstalled }, fileExists: { _ in isInstalled },
            requiredTeamIdentifier: requiredTeamIdentifier)
        let dependencies = SandboxDependencies(
            tool: tool, probe: probe, pin: pin, stateDirectory: directory,
            addresses: { addresses },
            freePort: { freePort },
            makeName: { SandboxController.containerNamePrefix + String(format: "%08d", counter.next()) },
            osSupported: { osSupported },
            sleep: { try await clock.sleep($0) },
            idleTimeout: .seconds(600),
            healthAttempts: healthAttempts,
            healthInterval: .milliseconds(500),
            quitRetryDelay: 0,
            clock: { proofClock.now })
        return SandboxHarness(world: world, probe: probe, clock: clock, proofClock: proofClock,
                              stateDirectory: directory,
                              controller: SandboxController(dependencies: dependencies),
                              blockedCount: blockedCount)
    }

    func cleanUp() {
        try? FileManager.default.removeItem(at: stateDirectory)
    }

    /// The folders under `secrets/`, one per start that has not been cleaned up.
    var secretFolders: [String] {
        let secrets = SandboxLocations.secretsDirectory(in: stateDirectory)
        return ((try? FileManager.default.contentsOfDirectory(atPath: secrets.path)) ?? []).sorted()
    }
}

final class NameCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var value = 0
    func next() -> Int { lock.withLock { value += 1; return value } }
}
#endif
