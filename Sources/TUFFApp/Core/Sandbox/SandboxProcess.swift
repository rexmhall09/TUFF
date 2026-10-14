#if os(macOS)
import Darwin
import Foundation
import os

/// What a finished command printed and how it ended.
struct SandboxProcessResult: Equatable, Sendable {
    let status: Int32
    /// Standard output and standard error together.
    let output: String
    /// True when the command was still running after its time limit and was
    /// ended by TUFF.
    let timedOut: Bool

    init(status: Int32, output: String, timedOut: Bool = false) {
        self.status = status
        self.output = output
        self.timedOut = timedOut
    }

    /// The last non-empty lines on one line, cleaned for an error message.
    func tail(lines: Int = 4) -> String {
        output.split(whereSeparator: \.isNewline)
            .map { SandboxText.oneLine(String($0)) }
            .filter { !$0.isEmpty }
            .suffix(lines)
            .joined(separator: " | ")
    }
}

/// Where a failed command's whole output goes, so it can be read later: the
/// message on screen keeps only its last lines. Standard error ends up in
/// the terminal TUFF was started from; the unified log keeps it too, with
/// the folder paths in it marked private.
enum SandboxLog {
    private static let logger = Logger(subsystem: "TUFF", category: "sandbox")

    static func commandFailed(action: String, status: Int32, output: String) {
        let text = "sandbox: could not \(action) (exit \(status)):\n\(output)\n"
        logger.error("\(text, privacy: .private)")
        FileHandle.standardError.write(Data(text.utf8))
    }
}

/// Text from a command or from the VM, made safe for a message on screen.
enum SandboxText {
    /// Control, format and separator characters become one space, runs of
    /// spaces collapse, and the result is cut to `limit` characters.
    static func oneLine(_ text: String, limit: Int = 300) -> String {
        var cleaned = String.UnicodeScalarView()
        var lastWasSpace = true
        for scalar in text.unicodeScalars {
            let category = scalar.properties.generalCategory
            let isBlank = scalar.properties.isWhitespace
                || category == .control || category == .format
                || category == .lineSeparator || category == .paragraphSeparator
            if isBlank {
                if !lastWasSpace { cleaned.append(" ") }
                lastWasSpace = true
            } else {
                cleaned.append(scalar)
                lastWasSpace = false
            }
        }
        let line = String(cleaned).trimmingCharacters(in: .whitespaces)
        return line.count > limit ? String(line.prefix(limit)) + "…" : line
    }
}

/// Runs a command-line tool to completion. Tests replace it with a fake; the
/// app uses `FoundationSandboxProcessRunner`.
protocol SandboxProcessRunning: Sendable {
    /// Ends the process with SIGTERM if it is still running after `timeout`
    /// seconds (and with SIGKILL three seconds later), so a hung `container`
    /// command cannot block the app.
    func run(executable: String, arguments: [String],
             timeout: TimeInterval) async throws -> SandboxProcessResult

    /// The same for the quit path, which cannot wait for an async call.
    /// Returns nil when the command could not start or did not end in time.
    func runBlocking(executable: String, arguments: [String],
                     timeout: TimeInterval) -> SandboxProcessResult?
}

struct FoundationSandboxProcessRunner: SandboxProcessRunning {
    /// Output is kept to this many bytes.
    static let maximumOutputBytes = 1_048_576

    init() {}

    /// A fixed environment: nothing the person's shell set, and no PATH
    /// search, since every executable is named by its full path.
    static var environment: [String: String] {
        let source = ProcessInfo.processInfo.environment
        var environment = ["PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "LANG": "C"]
        for key in ["HOME", "USER", "LOGNAME", "TMPDIR"] {
            if let value = source[key] { environment[key] = value }
        }
        return environment
    }

    /// Output goes to a temporary file rather than a pipe. `container system
    /// start` launches long-lived services, and a pipe they inherited would
    /// never reach end-of-file.
    func run(executable: String, arguments: [String],
             timeout: TimeInterval) async throws -> SandboxProcessResult {
        let log = Self.makeLog()
        guard FileManager.default.createFile(atPath: log.path, contents: nil) else {
            throw CocoaError(.fileWriteUnknown)
        }
        let handle = try FileHandle(forWritingTo: log)
        let progress = SandboxProcessProgress()
        let holder = SandboxProcessHandle()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
            let process = Process()
            process.executableURL = URL(fileURLWithPath: executable)
            process.arguments = arguments
            process.environment = Self.environment
            process.standardInput = FileHandle.nullDevice
            process.standardOutput = handle
            process.standardError = handle
            process.terminationHandler = { ended in
                progress.markFinished()
                try? handle.close()
                let output = Self.readAndDelete(log)
                continuation.resume(returning: SandboxProcessResult(
                    status: ended.terminationStatus, output: output,
                    timedOut: progress.timedOut))
            }
            do {
                try process.run()
                let pid = process.processIdentifier
                if holder.attach(pid) { Self.terminate(pid, progress: progress) }
                DispatchQueue.global().asyncAfter(deadline: .now() + timeout) {
                    // The flag keeps a reused process id from being signalled.
                    guard progress.markTimedOutIfRunning() else { return }
                    kill(pid, SIGTERM)
                    DispatchQueue.global().asyncAfter(deadline: .now() + 3) {
                        if !progress.isFinished { kill(pid, SIGKILL) }
                    }
                }
            } catch {
                try? handle.close()
                try? FileManager.default.removeItem(at: log)
                continuation.resume(throwing: error)
            }
            }
        } onCancel: {
            if let pid = holder.cancel() { Self.terminate(pid, progress: progress) }
        }
    }

    /// Ends a command that is no longer wanted: SIGTERM, and SIGKILL if it
    /// is still there 3 seconds later.
    private static func terminate(_ pid: Int32, progress: SandboxProcessProgress) {
        guard !progress.isFinished else { return }
        kill(pid, SIGTERM)
        DispatchQueue.global().asyncAfter(deadline: .now() + 3) {
            if !progress.isFinished { kill(pid, SIGKILL) }
        }
    }

    func runBlocking(executable: String, arguments: [String],
                     timeout: TimeInterval) -> SandboxProcessResult? {
        let log = Self.makeLog()
        guard FileManager.default.createFile(atPath: log.path, contents: nil),
              let handle = try? FileHandle(forWritingTo: log) else { return nil }
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        process.environment = Self.environment
        process.standardInput = FileHandle.nullDevice
        process.standardOutput = handle
        process.standardError = handle
        do { try process.run() } catch {
            try? handle.close()
            try? FileManager.default.removeItem(at: log)
            return nil
        }
        let deadline = Date().addingTimeInterval(timeout)
        while process.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }
        let finished = !process.isRunning
        if !finished {
            process.terminate()
            let grace = Date().addingTimeInterval(0.5)
            while process.isRunning && Date() < grace { Thread.sleep(forTimeInterval: 0.02) }
            if process.isRunning { kill(process.processIdentifier, SIGKILL) }
        }
        try? handle.close()
        let output = Self.readAndDelete(log)
        guard finished else { return nil }
        return SandboxProcessResult(status: process.terminationStatus, output: output)
    }

    private static func makeLog() -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-sandbox-\(UUID().uuidString).log")
    }

    private static func readAndDelete(_ log: URL) -> String {
        var data = Data()
        if let reader = try? FileHandle(forReadingFrom: log) {
            data = (try? reader.read(upToCount: maximumOutputBytes)) ?? Data()
            try? reader.close()
        }
        try? FileManager.default.removeItem(at: log)
        return String(decoding: data.prefix(maximumOutputBytes), as: UTF8.self)
    }
}

/// Whether a running command has ended or been timed out, shared between the
/// termination handler and the timer.
private final class SandboxProcessProgress: @unchecked Sendable {
    private let lock = NSLock()
    private var finished = false
    private var didTimeOut = false

    var isFinished: Bool { lock.withLock { finished } }
    var timedOut: Bool { lock.withLock { didTimeOut } }

    func markFinished() { lock.withLock { finished = true } }

    /// True when the command was still running, now marked as timed out.
    func markTimedOutIfRunning() -> Bool {
        lock.withLock {
            guard !finished else { return false }
            didTimeOut = true
            return true
        }
    }
}

/// Links a running command to the task that waits for it, so cancelling the
/// task ends the command.
private final class SandboxProcessHandle: @unchecked Sendable {
    private let lock = NSLock()
    private var pid: Int32?
    private var cancelled = false

    /// Records the command; true if the task was cancelled before it started.
    func attach(_ pid: Int32) -> Bool {
        lock.withLock {
            self.pid = pid
            return cancelled
        }
    }

    /// Marks the task cancelled; the command's id if it is already running.
    func cancel() -> Int32? {
        lock.withLock {
            cancelled = true
            return pid
        }
    }
}
#endif
