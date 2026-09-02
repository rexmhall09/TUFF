import Foundation
import TUFFCommandCore

private func writeError(_ text: String) {
    FileHandle.standardError.write(Data((text + "\n").utf8))
}

private func run(_ executableURL: URL, arguments: [String]) throws -> Int32 {
    let process = Process()
    process.executableURL = executableURL
    process.arguments = arguments
    process.standardInput = FileHandle.standardInput
    process.standardOutput = FileHandle.standardOutput
    process.standardError = FileHandle.standardError
    try process.run()
    process.waitUntilExit()
    return process.terminationStatus
}

let fileManager = FileManager.default
let currentDirectoryURL = URL(
    fileURLWithPath: fileManager.currentDirectoryPath, isDirectory: true)
let applicationSupportURL = (try? fileManager.url(
    for: .applicationSupportDirectory,
    in: .userDomainMask,
    appropriateFor: nil,
    create: false)) ?? fileManager.homeDirectoryForCurrentUser
let rawExecutableURL = URL(fileURLWithPath: CommandLine.arguments[0])
let executableURL = CommandLine.arguments[0].hasPrefix("/")
    ? rawExecutableURL
    : currentDirectoryURL.appendingPathComponent(CommandLine.arguments[0])
let preferences = UserDefaults(suiteName: "TUFF")

do {
    let plan = try TUFFCommand.plan(
        arguments: Array(CommandLine.arguments.dropFirst()),
        executableURL: executableURL,
        currentDirectoryURL: currentDirectoryURL,
        applicationSupportURL: applicationSupportURL,
        environment: ProcessInfo.processInfo.environment,
        selectedModel: preferences?.string(forKey: "model"))
    switch plan {
    case .help:
        print(TUFFCommand.usage)
        exit(0)
    case .run(let child, let arguments):
        exit(try run(child, arguments: arguments))
    case .load(let launcher, let arguments, let selection, let modelURL):
        preferences?.set(selection, forKey: "model")
        let status = try run(launcher, arguments: arguments)
        if status == 0 {
            print("Loading \(selection) from \(modelURL.path)")
        }
        exit(status)
    }
} catch {
    writeError("error: \(error)")
    writeError("")
    writeError(TUFFCommand.usage)
    exit(2)
}
