import Foundation
import Testing
import TUFFModelCatalog
@testable import TUFFCommandCore

@Suite("Unified tuff command")
struct TUFFCommandTests {
    private let repository = URL(fileURLWithPath: "/repo", isDirectory: true)
    private let appSupport = URL(
        fileURLWithPath: "/Users/test/Library/Application Support", isDirectory: true)

    @Test func promptUsesSelectedCatalogModelAndCatalogDefaults() throws {
        let executable = repository.appendingPathComponent(".build/debug/TUFFCommand")
        let existing: Set<String> = [
            "/repo/Package.swift",
            "/repo/Sources/TUFFApp/Mac",
            "/repo/.build/debug/TUFFCLI",
        ]
        let plan = try TUFFCommand.plan(
            arguments: ["prompt", "What is 2 + 2?"],
            executableURL: executable,
            currentDirectoryURL: repository,
            applicationSupportURL: appSupport,
            selectedModel: "minimax-m2.7",
            fileExists: existing.contains)

        guard case .run(let child, let arguments) = plan else {
            Issue.record("expected a child process")
            return
        }
        #expect(child.path == "/repo/.build/debug/TUFFCLI")
        #expect(arguments.prefix(4) == [
            "--model", "/repo/scratch/minimax-m2.7.gturbo",
            "--chat-prompt", "What is 2 + 2?",
        ])
        #expect(option("--max-context", in: arguments) == "4096")
        #expect(option("--expert-cache-slots", in: arguments) == "16")
        #expect(option("--thinking", in: arguments) == "on")
        #expect(option("--system-prompt", in: arguments)
            == TUFFModelCatalog.minimaxM27.defaultSystemPrompt)
        #expect(option("--prefill-chunk-tokens", in: arguments) == "auto")
    }

    @Test func promptAcceptsAnExplicitModelPathAndRawCLIOptions() throws {
        let executable = repository.appendingPathComponent(".build/debug/TUFFCommand")
        let existing: Set<String> = ["/repo/.build/debug/TUFFCLI"]
        let plan = try TUFFCommand.plan(
            arguments: [
                "prompt", "--model", "/models/custom.gturbo",
                "--prompt", "raw", "--temperature", "0",
            ],
            executableURL: executable,
            currentDirectoryURL: repository,
            applicationSupportURL: appSupport,
            fileExists: existing.contains)
        guard case .run(_, let arguments) = plan else {
            Issue.record("expected a child process")
            return
        }
        #expect(arguments.prefix(4) == [
            "--model", "/models/custom.gturbo", "--prompt", "raw",
        ])
        #expect(option("--temperature", in: arguments) == "0")
    }

    @Test func serveUsesSafePerModelContextDefaults() throws {
        let executable = repository.appendingPathComponent(".build/debug/TUFFCommand")
        let existing: Set<String> = [
            "/repo/Package.swift",
            "/repo/Sources/TUFFApp/Mac",
            "/repo/.build/debug/TUFFServer",
        ]
        let plan = try TUFFCommand.plan(
            arguments: ["serve", "--model", "minimax"],
            executableURL: executable,
            currentDirectoryURL: repository,
            applicationSupportURL: appSupport,
            fileExists: existing.contains)
        guard case .run(let child, let arguments) = plan else {
            Issue.record("expected a child process")
            return
        }
        #expect(child.lastPathComponent == "TUFFServer")
        #expect(option("--model", in: arguments)
            == "/repo/scratch/minimax-m2.7.gturbo")
        #expect(option("--max-context", in: arguments) == "4096")
        #expect(option("--expert-cache-slots", in: arguments) == "16")
    }

    @Test func loadSelectsInstalledModelAndLaunchesTheContainingApp() throws {
        let executable = URL(fileURLWithPath:
            "/Applications/TUFF.app/Contents/Resources/bin/tuff")
        let model = "/Users/test/Library/Application Support/TUFF/Models/minimax-m2.7.gturbo"
        let existing: Set<String> = [
            model + "/manifest.json",
            model + "/verified-install.json",
            "/Applications/TUFF.app/Contents/MacOS/TUFF",
        ]
        let plan = try TUFFCommand.plan(
            arguments: ["load", "minimax-m2.7"],
            executableURL: executable,
            currentDirectoryURL: URL(fileURLWithPath: "/tmp", isDirectory: true),
            applicationSupportURL: appSupport,
            fileExists: existing.contains)
        #expect(plan == .load(
            launcherURL: URL(fileURLWithPath: "/usr/bin/open"),
            arguments: [
                "-n", "/Applications/TUFF.app", "--args", "--load-model",
            ],
            selection: "minimax-m2.7",
            modelURL: URL(fileURLWithPath: model, isDirectory: true)))
    }

    @Test func loadRefusesASelectionThatIsNotInstalled() {
        #expect(throws: TUFFCommandError.modelNotInstalled(
            "/Users/test/Library/Application Support/TUFF/Models/minimax-m2.7.gturbo")) {
            _ = try TUFFCommand.plan(
                arguments: ["load", "minimax"],
                executableURL: URL(fileURLWithPath:
                    "/Applications/TUFF.app/Contents/Resources/bin/tuff"),
                currentDirectoryURL: URL(fileURLWithPath: "/tmp", isDirectory: true),
                applicationSupportURL: appSupport,
                fileExists: { _ in false })
        }
    }

    @Test func helpAndInvalidCommandsAreExplicit() throws {
        #expect(try TUFFCommand.plan(
            arguments: [],
            executableURL: repository.appendingPathComponent("tuff"),
            currentDirectoryURL: repository,
            applicationSupportURL: appSupport,
            fileExists: { _ in false }) == .help)
        #expect(throws: TUFFCommandError.unknownCommand("pull")) {
            _ = try TUFFCommand.plan(
                arguments: ["pull"],
                executableURL: repository.appendingPathComponent("tuff"),
                currentDirectoryURL: repository,
                applicationSupportURL: appSupport,
                fileExists: { _ in false })
        }
    }

    private func option(_ flag: String, in arguments: [String]) -> String? {
        guard let index = arguments.firstIndex(of: flag), index + 1 < arguments.count else {
            return nil
        }
        return arguments[index + 1]
    }
}
