#if os(macOS)
import Foundation

struct ContainerVersion: Comparable, Equatable, Sendable, CustomStringConvertible {
    let major: Int
    let minor: Int
    let patch: Int

    var description: String { "\(major).\(minor).\(patch)" }

    static func < (left: ContainerVersion, right: ContainerVersion) -> Bool {
        (left.major, left.minor, left.patch) < (right.major, right.minor, right.patch)
    }

    /// The first `1.5` or `1.5.0` in the output of `container --version`.
    static func parse(_ output: String) -> ContainerVersion? {
        guard let match = output.firstMatch(of: #/(\d+)\.(\d+)(?:\.(\d+))?/#),
              let major = Int(match.1), let minor = Int(match.2) else { return nil }
        let patch = match.3.flatMap { Int($0) } ?? 0
        return ContainerVersion(major: major, minor: minor, patch: patch)
    }
}

/// Apple's `container` command-line tool. TUFF runs only the file at
/// `/usr/local/bin/container`, where Apple's installer puts it, never a copy
/// found through PATH, and checks its signature and version before the first
/// command. Every command has a time limit and turns a failure into a
/// `SandboxError` with a short, cleaned tail of what the command printed.
struct ContainerTool: Sendable {
    static let installedPath = "/usr/local/bin/container"
    static let minimumVersion = ContainerVersion(major: 1, minor: 5, patch: 0)
    /// The team of "Developer ID Application: Apple Inc. - Containerization",
    /// which signs Apple's release (read from the installed tool with
    /// `codesign -dv`).
    static let appleTeamIdentifier = "UPBK2H6LZM"

    let runner: any SandboxProcessRunning
    let path: String
    let isExecutable: @Sendable (String) -> Bool
    /// Whether an executable file is there at all, trusted or not: it tells
    /// "not installed" from "installed, but not run because the person or
    /// their group could replace it".
    let fileExists: @Sendable (String) -> Bool
    /// When set, the tool's code signature must name this team. The app
    /// leaves it nil until the team identifier of Apple's release is pinned;
    /// the signature must still be valid, unmodified, not ad hoc and name
    /// some developer team.
    let requiredTeamIdentifier: String?

    init(runner: any SandboxProcessRunning = FoundationSandboxProcessRunner(),
         path: String = ContainerTool.installedPath,
         isExecutable: @escaping @Sendable (String) -> Bool = ContainerTool.isTrustedFile,
         fileExists: @escaping @Sendable (String) -> Bool = { FileManager.default.isExecutableFile(atPath: $0) },
         requiredTeamIdentifier: String? = nil) {
        self.runner = runner
        self.path = path
        self.isExecutable = isExecutable
        self.fileExists = fileExists
        self.requiredTeamIdentifier = requiredTeamIdentifier
    }

    /// An executable file that only root can change. A program the person
    /// could replace, or that any program of theirs could, is not run.
    @Sendable static func isTrustedFile(_ path: String) -> Bool {
        let manager = FileManager.default
        guard manager.isExecutableFile(atPath: path) else { return false }
        // The real file, and every folder above it, belongs to root and
        // cannot be changed by the person or their group.
        // Both the path as given and the real file are checked, so a link
        // placed in a folder the person can write does not help.
        let given = URL(fileURLWithPath: path)
        return ownedByRootAbove(given) && ownedByRootAbove(given.resolvingSymlinksInPath())
    }

    private static func ownedByRootAbove(_ start: URL) -> Bool {
        let manager = FileManager.default
        var url = start
        while true {
            guard let attributes = try? manager.attributesOfItem(atPath: url.path),
                  (attributes[.ownerAccountID] as? NSNumber)?.intValue == 0,
                  let permissions = (attributes[.posixPermissions] as? NSNumber)?.intValue,
                  permissions & 0o022 == 0 else { return false }
            let parent = url.deletingLastPathComponent()
            if parent.path == url.path || url.path == "/" { return true }
            url = parent
        }
    }

    /// Checks that the tool is installed, validly signed and recent enough.
    /// The signature is checked before the tool is run for the first time.
    @discardableResult
    func verify() async throws -> ContainerVersion {
        guard isExecutable(path) else {
            // Installed but not owned by root alone is a different problem
            // from not installed, and installing again does not fix it.
            throw fileExists(path)
                ? SandboxError.containerNotTrusted : SandboxError.containerNotInstalled
        }
        // With a pinned team the signature must also chain to Apple's
        // certificate authority and carry that team in the certificate, so a
        // certificate someone made themselves does not pass.
        var verifyArguments = ["--verify", "--strict"]
        if let team = requiredTeamIdentifier {
            verifyArguments.append("-R=anchor apple generic and certificate leaf[subject.OU] = \"\(team)\"")
        }
        let signature = try await runner.run(
            executable: "/usr/bin/codesign", arguments: verifyArguments + [path], timeout: 30)
        guard signature.status == 0, !signature.timedOut else {
            throw SandboxError.containerNotSigned(signature.tail(lines: 2))
        }
        let details = try await runner.run(
            executable: "/usr/bin/codesign", arguments: ["-dv", "--verbose=2", path], timeout: 30)
        let lines = details.output.split(whereSeparator: \.isNewline).map(String.init)
        guard details.status == 0, !details.timedOut, !lines.contains("Signature=adhoc") else {
            throw SandboxError.containerNotSigned("it is not signed by a developer")
        }
        let team = lines.first { $0.hasPrefix("TeamIdentifier=") }
            .map { String($0.dropFirst("TeamIdentifier=".count)) }
        if let required = requiredTeamIdentifier {
            guard team == required else {
                throw SandboxError.containerNotSigned("the signer is not the expected one")
            }
        } else if team == nil || team == "not set" {
            throw SandboxError.containerNotSigned("it does not name a developer team")
        }
        let result = try await runner.run(executable: path, arguments: ["--version"], timeout: 15)
        guard result.status == 0, !result.timedOut, let version = ContainerVersion.parse(result.output) else {
            throw SandboxError.containerCommandFailed(
                action: "read the version of the container tool", detail: result.tail(lines: 2))
        }
        guard version >= Self.minimumVersion else {
            throw SandboxError.containerTooOld(
                found: version.description, minimum: Self.minimumVersion.description)
        }
        return version
    }

    /// Whether the container system service answers.
    func systemIsRunning() async -> Bool {
        let result = try? await runner.run(
            executable: path, arguments: ["system", "status"], timeout: 30)
        return result?.status == 0 && result?.timedOut == false
    }

    /// Starts the system service. `--enable-kernel-install` answers the
    /// kernel prompt in advance, and the command has no input to wait on, so
    /// it never stops at a question.
    func startSystem() async throws {
        _ = try await run(["system", "start", "--enable-kernel-install"],
                          action: "start Apple container's system service", timeout: 600)
    }

    /// The IDs in `container list --quiet`, one per line (a header or extra
    /// columns, if a version prints them, are skipped). With `all`, stopped
    /// containers are listed too.
    func containerIDs(all: Bool) async throws -> [String] {
        let result = try await run(all ? ["list", "--all", "--quiet"] : ["list", "--quiet"],
                                   action: "list the containers", timeout: 30)
        return Self.containerIDs(in: result.output)
    }

    static func containerIDs(in output: String) -> [String] {
        output.split(whereSeparator: \.isNewline).compactMap { line in
            let first = line.split(whereSeparator: \.isWhitespace).first.map(String.init)
            return first == "ID" ? nil : first
        }
    }

    /// Starts a container in the background; the arguments are everything
    /// after `container run`.
    func runDetached(arguments: [String]) async throws {
        // In a task of its own: cancelling the caller must not cut the
        // command off half way, when the container may or may not exist yet.
        // The caller checks whether it is still wanted afterwards.
        try await Task { _ = try await self.run(["run"] + arguments, action: "start the sandbox", timeout: 180) }.value
    }

    /// Stops and deletes a container. Neither fails when it is already gone;
    /// the caller checks the list afterwards.
    func stopAndDelete(_ name: String) async {
        _ = try? await runner.run(executable: path, arguments: ["stop", name], timeout: 60)
        _ = try? await runner.run(executable: path, arguments: ["delete", name], timeout: 60)
    }

    /// The same for the quit path, which cannot wait for an async call.
    /// Returns true only when the list afterwards no longer shows the
    /// container.
    func stopAndDeleteBlocking(_ name: String) -> Bool {
        _ = runner.runBlocking(executable: path, arguments: ["stop", name], timeout: 8)
        _ = runner.runBlocking(executable: path, arguments: ["delete", name], timeout: 4)
        guard let listed = runner.runBlocking(
            executable: path, arguments: ["list", "--all", "--quiet"], timeout: 3),
              listed.status == 0 else { return false }
        return !Self.containerIDs(in: listed.output).contains(name)
    }

    /// A command inside a running container. A non-zero status is returned,
    /// not thrown: the caller decides what it means.
    func exec(_ name: String, _ command: [String]) async throws -> SandboxProcessResult {
        try await runner.run(executable: path, arguments: ["exec", name] + command, timeout: 30)
    }

    /// The last lines a container printed, for an error message. Empty if
    /// they cannot be read.
    func logTail(_ name: String) async -> String {
        let result = try? await runner.run(executable: path, arguments: ["logs", name], timeout: 15)
        return result?.tail(lines: 3) ?? ""
    }

    /// What `container image inspect` printed, or nil if the image is not
    /// there (or the command failed).
    func inspectImage(_ reference: String) async -> String? {
        guard let result = try? await runner.run(
            executable: path, arguments: ["image", "inspect", reference], timeout: 60),
              result.status == 0, !result.timedOut else { return nil }
        return result.output
    }

    func pullImage(_ reference: String) async throws {
        _ = try await run(["image", "pull", reference],
                          action: "download the sandbox image", timeout: 1_800)
    }

    private func run(_ arguments: [String], action: String,
                     timeout: TimeInterval) async throws -> SandboxProcessResult {
        let result: SandboxProcessResult
        do {
            result = try await runner.run(executable: path, arguments: arguments, timeout: timeout)
        } catch {
            throw SandboxError.containerCommandFailed(
                action: action, detail: SandboxText.oneLine(error.localizedDescription))
        }
        if result.timedOut { throw SandboxError.timedOut(action: action) }
        guard result.status == 0 else {
            throw SandboxError.containerCommandFailed(action: action, detail: result.tail())
        }
        return result
    }
}
#endif
