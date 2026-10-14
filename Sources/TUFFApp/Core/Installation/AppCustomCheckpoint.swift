import Foundation
import Darwin
import TUFFModelCatalog
import TUFFModelMetadata
import TUFFRuntimeDescriptions
import TUFFRepackCore

public struct AppCustomCheckpoint: Codable, Equatable, Sendable, Identifiable {
    public let id: String
    public let name: String
    public let sourcePath: String
    public let templateID: TUFFModelID
    public let sourceIndexSHA256: String
    public let manifestSHA256: String
    public let installedBytes: UInt64

    public init(id: String, name: String, sourcePath: String, templateID: TUFFModelID,
                sourceIndexSHA256: String, manifestSHA256: String, installedBytes: UInt64) {
        self.id = id
        self.name = name
        self.sourcePath = sourcePath
        self.templateID = templateID
        self.sourceIndexSHA256 = sourceIndexSHA256
        self.manifestSHA256 = manifestSHA256
        self.installedBytes = installedBytes
    }

    public var descriptor: AppModelInstallDescriptor {
        AppModelInstallDescriptor(displayName: name, repoID: "local/\(id)", revision: manifestSHA256,
            sourceIndexSHA256: sourceIndexSHA256, approximateDownloadBytes: 0,
            installedBytes: installedBytes, rangeStagingBytes: 0, reserveBytes: 0,
            customIdentity: id, customArchitectureTemplate: templateID,
            customManifestSHA256: manifestSHA256)
    }

    public static func inspect(directory: URL, name: String, templateID: TUFFModelID) throws -> Self {
        let source = directory.standardizedFileURL.resolvingSymlinksInPath()
        let label = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard source.pathExtension == "gturbo", !label.isEmpty, label.count <= 80,
              !label.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) }),
              let template = TUFFModelCatalog.model(id: templateID),
              let variant = ModelVariant(rawValue: template.architecture.id.rawValue),
              let architecture = ArchConfig.registeredArchitectures[variant] else {
            throw ImportError("Choose a supported architecture, a name, and a complete .gturbo folder.")
        }
        let manifest = try ManifestReader.load(directoryURL: source, expecting: architecture)
        guard let fingerprint = manifest.sourceSnapshotHash, fingerprint.hasPrefix("sha256:"),
              fingerprint.dropFirst(7).count == 64,
              fingerprint.dropFirst(7).allSatisfy({ "0123456789abcdef".contains($0) }) else {
            throw ImportError("The checkpoint needs a valid source fingerprint.")
        }
        let receipt = try VerifiedInstallReceiptReader.load(directoryURL: source)
        let manifestHash = try Sha256Verifier.hashFile(at: source.appendingPathComponent("manifest.json"))
        try VerifiedInstallReceiptReader.validateManifestBinding(receipt, directoryURL: source,
            manifestSha256: manifestHash)
        let access = try GTurboModelDirectory(rootURL: source)
        for path in ["tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json", "tokenizer/chat_template.jinja"] {
            guard try access.fileSize(path) > 0 else { throw ImportError("Missing or empty \(path).") }
        }
        let tokenizerData = try access.readMetadata("tokenizer/tokenizer.json", maxBytes: 64 << 20)
        guard let tokenizer = try JSONSerialization.jsonObject(with: tokenizerData) as? [String: Any],
              let tokenizerModel = tokenizer["model"] as? [String: Any],
              let kind = tokenizerModel["type"] as? String, ["BPE", "Unigram"].contains(kind) else {
            throw ImportError("Unsupported tokenizer format. A matching BPE or Unigram tokenizer is required.")
        }
        if let vocabulary = tokenizerModel["vocab"] as? [String: Int] {
            guard !vocabulary.isEmpty, vocabulary.values.allSatisfy({ $0 >= 0 && $0 < architecture.vocabSize }) else {
                throw ImportError("Tokenizer IDs do not fit the selected architecture.")
            }
        } else if let vocabulary = tokenizerModel["vocab"] as? [[Any]], !vocabulary.isEmpty,
                  vocabulary.count <= architecture.vocabSize {
            // Unigram token IDs are the vocabulary's array positions.
        } else { throw ImportError("Invalid tokenizer vocabulary.") }
        for token in tokenizer["added_tokens"] as? [[String: Any]] ?? [] {
            guard let id = token["id"] as? Int, id >= 0, id < architecture.vocabSize else {
                throw ImportError("Added tokenizer IDs do not fit the selected architecture.")
            }
        }
        _ = try JSONSerialization.jsonObject(with: access.readMetadata("tokenizer/tokenizer_config.json", maxBytes: 4 << 20))
        guard String(data: try access.readMetadata("tokenizer/chat_template.jinja", maxBytes: 1 << 20), encoding: .utf8) != nil else {
            throw ImportError("The chat template must be valid UTF-8.")
        }
        let checked = try VerifiedInstallTool.run(options: .init(inputGTurbo: source.path, writeReceipt: false))
        guard checked.unexpectedEntries.isEmpty else {
            throw ImportError("Unexpected checkpoint entries: \(checked.unexpectedEntries.joined(separator: ", ")).")
        }
        try ModelRuntimeSchemaValidator.validate(directory: source, expecting: architecture)
        _ = try ModelWeightLayoutSummary.read(directory: source, expecting: architecture)
        return Self(id: "custom-" + UUID().uuidString.lowercased(), name: label,
            sourcePath: source.path, templateID: templateID,
            sourceIndexSHA256: String(fingerprint.dropFirst(7)), manifestSHA256: manifestHash,
            installedBytes: checked.bytesVerified)
    }

    public struct ImportError: LocalizedError {
        public let errorDescription: String?
        public init(_ message: String) { errorDescription = message }
    }
}

/// Custom checkpoints as other processes see them, such as `tuff serve`,
/// which offers each one under a name a client can type.
public enum AppCustomCheckpointRegistry {
    /// The registered checkpoints beside `modelsRoot`'s models. An unreadable
    /// registry offers none rather than failing whoever asked.
    public static func checkpoints(modelsRoot: URL) -> [AppCustomCheckpoint] {
        (try? AppCustomCheckpointStore.load(
            at: modelsRoot.appendingPathComponent("custom-checkpoints.json"))) ?? []
    }

    /// One API name per checkpoint, from its display name: lowercase letters,
    /// digits and hyphens. A name a catalog model already answers to gets
    /// `-custom`; a repeat gets `-2`, `-3`, in registry order, so names stay
    /// put while the registry does.
    public static func apiNames(for checkpoints: [AppCustomCheckpoint]) -> [String: String] {
        let reserved = Set(TUFFModelCatalog.all.flatMap {
            [$0.apiModelID, $0.selector, $0.id.rawValue] + $0.aliases
        } + ["default"])
        var taken = reserved
        var names: [String: String] = [:]
        for checkpoint in checkpoints {
            var base = slug(checkpoint.name)
            if base.isEmpty { base = "custom" }
            if reserved.contains(base) { base += "-custom" }
            var name = base
            var suffix = 2
            while taken.contains(name) {
                name = "\(base)-\(suffix)"
                suffix += 1
            }
            taken.insert(name)
            names[checkpoint.id] = name
        }
        return names
    }

    static func slug(_ text: String) -> String {
        var result = ""
        for scalar in text.lowercased().unicodeScalars {
            if ("a"..."z").contains(scalar) || ("0"..."9").contains(scalar) {
                result.unicodeScalars.append(scalar)
            } else if !result.isEmpty, !result.hasSuffix("-") {
                result += "-"
            }
        }
        result = String(result.prefix(64))
        while result.hasSuffix("-") { result.removeLast() }
        return result
    }
}

struct AppCustomCheckpointStore {
    struct Archive: Codable { var schemaVersion = 1; var checkpoints: [AppCustomCheckpoint] = [] }
    static var directory: URL { AppModelLocation.defaultURL(descriptor: .default).deletingLastPathComponent() }
    static var url: URL { directory.appendingPathComponent("custom-checkpoints.json") }
    static func load(at location: URL = url) throws -> [AppCustomCheckpoint] {
        guard FileManager.default.fileExists(atPath: location.path) else { return [] }
        let data = try GTurboModelDirectory(rootURL: location.deletingLastPathComponent()).readMetadata(location.lastPathComponent, maxBytes: 256 << 10)
        guard data.count <= 256 << 10 else { throw AppCustomCheckpoint.ImportError("Custom checkpoint registry is too large.") }
        let archive = try JSONDecoder().decode(Archive.self, from: data)
        guard archive.schemaVersion == 1, archive.checkpoints.count <= 64,
              Set(archive.checkpoints.map(\.id)).count == archive.checkpoints.count,
              archive.checkpoints.allSatisfy({ $0.id.hasPrefix("custom-") && UUID(uuidString: String($0.id.dropFirst(7))) != nil
                  && $0.sourcePath.hasPrefix("/") && !$0.sourcePath.contains("\0")
                  && !$0.name.isEmpty && $0.name.count <= 80 && $0.sourceIndexSHA256.count == 64
                  && $0.manifestSHA256.count == 64 && TUFFModelCatalog.model(id: $0.templateID) != nil }) else {
            throw AppCustomCheckpoint.ImportError("Invalid custom checkpoint registry.")
        }
        return archive.checkpoints
    }
    static func save(_ checkpoints: [AppCustomCheckpoint], at location: URL = url) throws {
        guard checkpoints.count <= 64 else { throw AppCustomCheckpoint.ImportError("At most 64 custom checkpoints are supported.") }
        try FileManager.default.createDirectory(at: location.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONEncoder().encode(Archive(checkpoints: checkpoints)).write(to: location, options: .atomic)
    }
}

struct AppCustomCheckpointInstaller: AppModelInstallerClient {
    let descriptor: AppModelInstallDescriptor
    func checkInstallRequirement(outputDirectory: URL) throws -> AppModelInstallRequirement {
        throw AppCustomCheckpoint.ImportError("Custom checkpoints are local. Restore the original folder or import it again.")
    }
    func installDefaultModel(outputDirectory: URL) -> AsyncThrowingStream<AppModelInstallEvent, Error> {
        AsyncThrowingStream { $0.finish(throwing: AppCustomCheckpoint.ImportError("Import a complete local checkpoint in Advanced settings.")) }
    }
    func discardPartialInstall(outputDirectory: URL) async throws {}
    // Custom checkpoints are removed by `removeCustomCheckpoint`, which unlinks
    // the alias and leaves the user's own folder alone.
    func removeInstalledModel(outputDirectory: URL) async throws {}
    func blockedInstallEntries(outputDirectory: URL) -> [String] { [] }
    func clearBlockedInstallPath(outputDirectory: URL) async throws {}
    func cancel() {}
}
