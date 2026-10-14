#if os(macOS)
import Foundation
import Testing
@testable import TUFFAppCore

@Suite struct SandboxTextTests {
    @Test func plainTextIsUnchanged() {
        #expect(SandboxText.oneLine("hello world") == "hello world")
        #expect(SandboxText.oneLine("") == "")
    }

    @Test func newlinesTabsAndRunsOfSpacesCollapseToOneSpace() {
        #expect(SandboxText.oneLine("a\nb") == "a b")
        #expect(SandboxText.oneLine("a\r\nb") == "a b")
        #expect(SandboxText.oneLine("a\n\n\nb") == "a b")
        #expect(SandboxText.oneLine("a\t\tb") == "a b")
        #expect(SandboxText.oneLine("a     b") == "a b")
    }

    @Test func leadingAndTrailingBlanksAreRemoved() {
        #expect(SandboxText.oneLine("  \n\t a b \n  ") == "a b")
    }

    @Test func controlFormatAndSeparatorCharactersBecomeSpaces() {
        #expect(SandboxText.oneLine("a\u{0007}b") == "a b")
        #expect(SandboxText.oneLine("a\u{0000}b") == "a b")
        #expect(SandboxText.oneLine("a\u{001B}b") == "a b")
        #expect(SandboxText.oneLine("a\u{200B}b") == "a b")
        #expect(SandboxText.oneLine("a\u{2028}b") == "a b")
        #expect(SandboxText.oneLine("a\u{2029}b") == "a b")
        #expect(SandboxText.oneLine("a\u{00A0}b") == "a b")
    }

    @Test func textMadeOnlyOfBlanksBecomesEmpty() {
        #expect(SandboxText.oneLine("\u{0007}\u{0000}\n\t \u{200B}") == "")
    }

    @Test func resultIsNeverMultiLine() {
        let nasty = "one\ntwo\rthree\u{2028}four\u{0085}five\u{000B}six\u{000C}seven"
        let cleaned = SandboxText.oneLine(nasty)
        #expect(cleaned.rangeOfCharacter(from: .newlines) == nil)
        #expect(cleaned == "one two three four five six seven")
    }

    @Test func ordinaryNonAsciiTextIsKept() {
        #expect(SandboxText.oneLine("ok \u{2713} caf\u{00E9}") == "ok \u{2713} caf\u{00E9}")
    }

    @Test func textLongerThanTheLimitIsCutAndMarked() {
        let long = String(repeating: "a", count: 10)
        #expect(SandboxText.oneLine(long, limit: 5) == "aaaaa\u{2026}")
    }

    @Test func textAtTheLimitIsNotCut() {
        let exact = String(repeating: "a", count: 10)
        #expect(SandboxText.oneLine(exact, limit: 10) == exact)
        #expect(SandboxText.oneLine(exact, limit: 11) == exact)
    }

    @Test func defaultLimitIsThreeHundredCharacters() {
        let atLimit = String(repeating: "x", count: 300)
        #expect(SandboxText.oneLine(atLimit) == atLimit)

        let over = String(repeating: "x", count: 301)
        let cut = SandboxText.oneLine(over)
        #expect(cut == String(repeating: "x", count: 300) + "\u{2026}")
        #expect(cut.count == 301)
    }

    @Test func limitCountsTheCleanedTextNotTheInput() {
        // Ten newlines collapse to nothing, so only the visible text counts.
        let input = String(repeating: "\n", count: 10) + "abc" + String(repeating: "\n", count: 10)
        #expect(SandboxText.oneLine(input, limit: 3) == "abc")
    }
}

@Suite struct SandboxProcessResultTests {
    @Test func initializerDefaultsToNotTimedOut() {
        let result = SandboxProcessResult(status: 3, output: "x")
        #expect(result.status == 3)
        #expect(result.output == "x")
        #expect(result.timedOut == false)
    }

    @Test func timedOutIsPartOfEquality() {
        let normal = SandboxProcessResult(status: 0, output: "")
        let timedOut = SandboxProcessResult(status: 0, output: "", timedOut: true)
        #expect(normal != timedOut)
        #expect(normal == SandboxProcessResult(status: 0, output: "", timedOut: false))
    }

    @Test func tailKeepsTheLastFourLinesByDefault() {
        let result = SandboxProcessResult(status: 1, output: "a\nb\nc\nd\ne\nf\n")
        #expect(result.tail() == "c | d | e | f")
    }

    @Test func tailHonoursTheLineCount() {
        let result = SandboxProcessResult(status: 1, output: "a\nb\nc\nd\n")
        #expect(result.tail(lines: 2) == "c | d")
        #expect(result.tail(lines: 1) == "d")
        #expect(result.tail(lines: 0) == "")
        #expect(result.tail(lines: 10) == "a | b | c | d")
    }

    @Test func tailSkipsEmptyAndBlankLines() {
        let result = SandboxProcessResult(status: 1, output: "a\n\n   \n\t\nb\n\n")
        #expect(result.tail() == "a | b")
    }

    @Test func tailOfNoOutputIsEmpty() {
        #expect(SandboxProcessResult(status: 0, output: "").tail() == "")
        #expect(SandboxProcessResult(status: 0, output: "\n\n").tail() == "")
    }

    @Test func tailCleansEachLine() {
        let result = SandboxProcessResult(status: 1, output: "  first \t line  \nsecond\u{200B}line\n")
        #expect(result.tail() == "first line | second line")
    }

    @Test func tailHandlesWindowsLineEndings() {
        let result = SandboxProcessResult(status: 1, output: "a\r\nb\r\nc\r\n")
        #expect(result.tail(lines: 2) == "b | c")
    }

    @Test func tailLimitsEachLineLikeOneLine() {
        let long = String(repeating: "z", count: 400)
        let result = SandboxProcessResult(status: 1, output: long)
        #expect(result.tail() == String(repeating: "z", count: 300) + "\u{2026}")
    }
}

@Suite struct SandboxErrorTests {
    static let everyCase: [SandboxError] = [
        .unsupportedSystem,
        .containerNotInstalled,
        .containerNotTrusted,
        .containerTooOld(found: "1.4.0", minimum: "1.5.0"),
        .containerNotSigned("code object is not signed"),
        .containerNotSigned(""),
        .containerCommandFailed(action: "start the sandbox", detail: "boom"),
        .containerCommandFailed(action: "start the sandbox", detail: ""),
        .timedOut(action: "start the sandbox"),
        .imageNotPublished,
        .digestMismatch(expected: "sha256:aa", found: "sha256:bb"),
        .digestMismatch(expected: "sha256:aa", found: nil),
        .anotherInstanceRunning,
        .unsafeMountPath,
        .noFreePort,
        .secretUnavailable("no keychain"),
        .tooManyAddresses,
        .hostAddressesUnknown,
        .notHealthy("no answer"),
        .protectionNotVerified("rules missing"),
        .stoppedWhileStarting,
        .notReady,
        .stillStarting,
        .noFoldersToMount,
        .folderAccessDenied,
        .unmountableFolder,
    ]

    @Test func everyCaseHasANonEmptySingleLineDescription() {
        for error in Self.everyCase {
            let text = error.description
            #expect(!text.isEmpty, "\(error)")
            #expect(text.rangeOfCharacter(from: .newlines) == nil, "\(error)")
            #expect(text == text.trimmingCharacters(in: .whitespaces), "\(error)")
        }
    }

    @Test func descriptionIsWhatStringConversionsUse() {
        let error = SandboxError.noFreePort
        #expect("\(error)" == error.description)
        #expect(String(describing: error) == error.description)
    }

    @Test func descriptionsAreDistinctAcrossKindsOfError() {
        let descriptions = Set(Self.everyCase.map(\.description))
        // Two cases appear twice with different payloads, so nothing collapses.
        #expect(descriptions.count == Self.everyCase.count)
    }

    @Test func simpleMessages() {
        #expect(SandboxError.unsupportedSystem.description == "The sandbox needs macOS 26 or newer.")
        #expect(SandboxError.notReady.description == "The sandbox is not ready.")
        #expect(SandboxError.stillStarting.description == "The sandbox is still starting.")
        #expect(SandboxError.stoppedWhileStarting.description == "The sandbox was stopped while it was starting.")
        #expect(SandboxError.containerNotInstalled.description.contains("not installed"))
    }

    @Test func tooOldNamesBothVersions() {
        let text = SandboxError.containerTooOld(found: "1.4.2", minimum: "1.5.0").description
        #expect(text == "Apple container 1.4.2 is too old; the sandbox needs 1.5.0 or newer.")
    }

    @Test func notSignedAddsTheDetailInParentheses() {
        let bare = SandboxError.containerNotSigned("").description
        #expect(bare == "The container tool at /usr/local/bin/container does not have a valid signature.")
        let detailed = SandboxError.containerNotSigned("bad seal").description
        #expect(detailed == "The container tool at /usr/local/bin/container does not have a valid signature (bad seal).")
    }

    @Test func commandFailedAppendsTheDetailOnlyWhenThereIsOne() {
        #expect(SandboxError.containerCommandFailed(action: "list the containers", detail: "").description
                == "Could not list the containers.")
        #expect(SandboxError.containerCommandFailed(action: "list the containers", detail: "oops").description
                == "Could not list the containers: oops")
    }

    @Test func timedOutNamesTheAction() {
        #expect(SandboxError.timedOut(action: "start the sandbox").description
                == "Timed out while trying to start the sandbox.")
    }

    @Test func digestMismatchNamesBothDigestsOrNothing() {
        let withFound = SandboxError.digestMismatch(expected: "sha256:aa", found: "sha256:bb").description
        #expect(withFound.contains("wanted sha256:aa"))
        #expect(withFound.contains("found sha256:bb"))
        let without = SandboxError.digestMismatch(expected: "sha256:aa", found: nil).description
        #expect(without.contains("wanted sha256:aa"))
        #expect(without.contains("found nothing"))
    }

    @Test func detailCarryingCasesIncludeTheDetail() {
        #expect(SandboxError.secretUnavailable("no keychain").description.contains("no keychain"))
        #expect(SandboxError.notHealthy("no answer").description.contains("no answer"))
        #expect(SandboxError.protectionNotVerified("rules missing").description.contains("rules missing"))
    }

    @Test func equalityComparesPayloads() {
        #expect(SandboxError.timedOut(action: "a") == SandboxError.timedOut(action: "a"))
        #expect(SandboxError.timedOut(action: "a") != SandboxError.timedOut(action: "b"))
        #expect(SandboxError.digestMismatch(expected: "x", found: nil)
                != SandboxError.digestMismatch(expected: "x", found: "y"))
        #expect(SandboxError.notReady != SandboxError.stillStarting)
    }
}

/// Runs real, harmless programs from /bin and /usr/bin.
@Suite struct FoundationSandboxProcessRunnerTests {
    let runner = FoundationSandboxProcessRunner()

    // MARK: run

    @Test func capturesStandardOutput() async throws {
        let result = try await runner.run(executable: "/bin/echo", arguments: ["hello", "sandbox"], timeout: 30)
        #expect(result.status == 0)
        #expect(result.output == "hello sandbox\n")
        #expect(result.timedOut == false)
    }

    @Test func reportsANonZeroExitStatus() async throws {
        let result = try await runner.run(executable: "/usr/bin/false", arguments: [], timeout: 30)
        #expect(result.status != 0)
        #expect(result.timedOut == false)
    }

    @Test func reportsTheExactExitStatus() async throws {
        let result = try await runner.run(executable: "/bin/sh", arguments: ["-c", "exit 7"], timeout: 30)
        #expect(result.status == 7)
        #expect(result.output == "")
    }

    @Test func standardErrorIsMergedIntoTheOutput() async throws {
        let result = try await runner.run(
            executable: "/bin/sh", arguments: ["-c", "echo out; echo err 1>&2"], timeout: 30)
        #expect(result.status == 0)
        #expect(result.output.contains("out"))
        #expect(result.output.contains("err"))
    }

    @Test func aCommandThatCannotStartThrows() async {
        await #expect(throws: (any Error).self) {
            _ = try await runner.run(
                executable: "/nonexistent/tuff-sandbox-test-binary", arguments: [], timeout: 5)
        }
    }

    @Test func aCommandStillRunningAtTheTimeLimitIsEnded() async throws {
        let clock = ContinuousClock()
        let start = clock.now
        let result = try await runner.run(executable: "/bin/sleep", arguments: ["30"], timeout: 1)
        let elapsed = start.duration(to: clock.now)
        #expect(result.timedOut)
        #expect(result.status != 0)
        #expect(elapsed < .seconds(15))
    }

    @Test func aFastCommandIsNotMarkedTimedOut() async throws {
        let result = try await runner.run(executable: "/bin/echo", arguments: ["fast"], timeout: 30)
        #expect(result.timedOut == false)
    }

    @Test func outputIsCappedAtTheMaximum() async throws {
        #expect(FoundationSandboxProcessRunner.maximumOutputBytes == 1_048_576)
        // Two megabytes of NUL bytes; only the first megabyte is kept.
        let result = try await runner.run(
            executable: "/bin/sh",
            arguments: ["-c", "/bin/dd if=/dev/zero bs=1024 count=2048 2>/dev/null"],
            timeout: 60)
        #expect(result.status == 0)
        #expect(result.output.utf8.count == FoundationSandboxProcessRunner.maximumOutputBytes)
    }

    // MARK: environment

    @Test func environmentIsFixed() {
        let environment = FoundationSandboxProcessRunner.environment
        #expect(environment["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin")
        #expect(environment["LANG"] == "C")
        let allowed: Set<String> = ["PATH", "LANG", "HOME", "USER", "LOGNAME", "TMPDIR"]
        #expect(Set(environment.keys).isSubset(of: allowed))
        if let home = ProcessInfo.processInfo.environment["HOME"] {
            #expect(environment["HOME"] == home)
        }
    }

    @Test func aVariableOfTheTestProcessIsNotPassedOn() async throws {
        let name = "TUFF_SANDBOX_TEST_LEAK_" + UUID().uuidString.replacingOccurrences(of: "-", with: "")
        _ = setenv(name, "leaked-value", 1)
        defer { _ = unsetenv(name) }

        #expect(FoundationSandboxProcessRunner.environment[name] == nil)

        let result = try await runner.run(executable: "/usr/bin/env", arguments: [], timeout: 30)
        #expect(result.status == 0)
        #expect(!result.output.contains(name))
        #expect(!result.output.contains("leaked-value"))
        let lines = result.output.split(whereSeparator: \.isNewline).map(String.init)
        #expect(lines.contains("PATH=/usr/bin:/bin:/usr/sbin:/sbin"))
        #expect(lines.contains("LANG=C"))
    }

    @Test func theBlockingRunnerAlsoUsesTheFixedEnvironment() {
        let name = "TUFF_SANDBOX_TEST_LEAK_" + UUID().uuidString.replacingOccurrences(of: "-", with: "")
        _ = setenv(name, "leaked-value", 1)
        defer { _ = unsetenv(name) }

        let result = runner.runBlocking(executable: "/usr/bin/env", arguments: [], timeout: 30)
        #expect(result?.status == 0)
        #expect(result?.output.contains(name) == false)
        #expect(result?.output.contains("PATH=/usr/bin:/bin:/usr/sbin:/sbin") == true)
    }

    // MARK: runBlocking

    @Test func blockingRunCapturesOutput() {
        let result = runner.runBlocking(executable: "/bin/echo", arguments: ["hi"], timeout: 30)
        #expect(result == SandboxProcessResult(status: 0, output: "hi\n"))
    }

    @Test func blockingRunReportsANonZeroStatus() {
        let result = runner.runBlocking(executable: "/usr/bin/false", arguments: [], timeout: 30)
        #expect(result != nil)
        #expect(result?.status != 0)
        #expect(result?.timedOut == false)
    }

    @Test func blockingRunReturnsNilWhenTheCommandOutlastsTheTimeout() {
        let clock = ContinuousClock()
        let start = clock.now
        let result = runner.runBlocking(executable: "/bin/sleep", arguments: ["30"], timeout: 1)
        let elapsed = start.duration(to: clock.now)
        #expect(result == nil)
        #expect(elapsed < .seconds(15))
    }

    @Test func blockingRunReturnsNilWhenTheCommandCannotStart() {
        let result = runner.runBlocking(
            executable: "/nonexistent/tuff-sandbox-test-binary", arguments: [], timeout: 5)
        #expect(result == nil)
    }
}
#endif
