import Foundation

public enum ModelIntegrityPolicy: Sendable, Equatable {
    case fullSha256
    case sizeCheckTrustedReceipt
}

public struct VerifiedInstallReceipt: Codable, Equatable, Sendable {
    public struct FileEntry: Codable, Equatable, Sendable {
        public let size: UInt64
        public let sha256: String

        public init(size: UInt64, sha256: String) {
            self.size = size
            self.sha256 = sha256
        }
    }

    public let schemaVersion: Int
    public let manifestSha256: String
    public let modelDirectoryPath: String
    public let sourceRepoID: String?
    public let sourceRevision: String?
    public let verificationTimestamp: String
    public let toolVersion: String
    public let files: [String: FileEntry]

    public init(schemaVersion: Int = 1,
                manifestSha256: String,
                modelDirectoryPath: String,
                sourceRepoID: String? = nil,
                sourceRevision: String? = nil,
                verificationTimestamp: String,
                toolVersion: String,
                files: [String: FileEntry]) {
        self.schemaVersion = schemaVersion
        self.manifestSha256 = manifestSha256
        self.modelDirectoryPath = modelDirectoryPath
        self.sourceRepoID = sourceRepoID
        self.sourceRevision = sourceRevision
        self.verificationTimestamp = verificationTimestamp
        self.toolVersion = toolVersion
        self.files = files
    }
}

public enum VerifiedInstallReceiptReader {
    public static let fileName = "verified-install.json"
    public static let defaultMaxBytes: UInt64 = 4 * 1024 * 1024

    public static func load(directoryURL: URL,
                            maxBytes: UInt64 = defaultMaxBytes) throws -> VerifiedInstallReceipt {
        do {
            let directory = try GTurboModelDirectory(rootURL: directoryURL)
            let data = try directory.readMetadata(fileName, maxBytes: maxBytes)
            return try decode(data: data)
        } catch ModelError.missingFile {
            throw ModelError.trustedReceiptInvalid(detail: "\(fileName) is missing")
        } catch let error as ModelError {
            if case .trustedReceiptInvalid = error { throw error }
            throw ModelError.trustedReceiptInvalid(detail: "\(fileName): \(error)")
        } catch {
            throw ModelError.trustedReceiptInvalid(detail: "\(fileName): \(error)")
        }
    }

    package static func decode(data: Data) throws -> VerifiedInstallReceipt {
        do { return try JSONDecoder().decode(VerifiedInstallReceipt.self, from: data) }
        catch { throw ModelError.trustedReceiptInvalid(detail: "\(fileName): \(error)") }
    }

    public static func validate(_ receipt: VerifiedInstallReceipt,
                                directoryURL: URL,
                                manifest: Manifest,
                                manifestSha256: String,
                                manifestSize: UInt64) throws {
        try validateManifestBinding(receipt,
                                    directoryURL: directoryURL,
                                    manifestSha256: manifestSha256)
        var expectedFiles = Set(manifest.files.keys)
        expectedFiles.insert("manifest.json")
        let receiptFiles = Set(receipt.files.keys)
        guard receiptFiles == expectedFiles else {
            let missing = expectedFiles.subtracting(receiptFiles).sorted()
            let extra = receiptFiles.subtracting(expectedFiles).sorted()
            throw ModelError.trustedReceiptInvalid(
                detail: "receipt file set mismatch missing=\(missing) extra=\(extra)")
        }
        guard let manifestReceiptEntry = receipt.files["manifest.json"] else {
            throw ModelError.trustedReceiptInvalid(detail: "receipt missing manifest.json")
        }
        guard manifestReceiptEntry.size == manifestSize else {
            throw ModelError.trustedReceiptInvalid(detail: "manifest.json size mismatch")
        }
        guard manifestReceiptEntry.sha256.lowercased() == manifestSha256.lowercased() else {
            throw ModelError.trustedReceiptInvalid(detail: "manifest.json SHA mismatch")
        }

        for (rel, manifestEntry) in manifest.files {
            guard let receiptEntry = receipt.files[rel] else {
                throw ModelError.trustedReceiptInvalid(detail: "receipt missing \(rel)")
            }
            guard receiptEntry.size == manifestEntry.size else {
                throw ModelError.trustedReceiptInvalid(detail: "receipt size mismatch for \(rel)")
            }
            guard receiptEntry.sha256.lowercased() == manifestEntry.sha256.lowercased() else {
                throw ModelError.trustedReceiptInvalid(detail: "receipt SHA mismatch for \(rel)")
            }
        }
    }

    public static func validateManifestBinding(_ receipt: VerifiedInstallReceipt,
                                               directoryURL: URL,
                                               manifestSha256: String) throws {
        guard receipt.schemaVersion == 1 else {
            throw ModelError.trustedReceiptInvalid(
                detail: "unsupported schemaVersion \(receipt.schemaVersion)")
        }
        guard receipt.manifestSha256.lowercased() == manifestSha256.lowercased() else {
            throw ModelError.trustedReceiptInvalid(detail: "manifest SHA mismatch")
        }

        guard directoryBindingMatches(
            receiptPath: receipt.modelDirectoryPath,
            openedDirectory: directoryURL
        ) else {
            throw ModelError.trustedReceiptInvalid(detail: "model directory mismatch")
        }
    }

    /// A receipt written for the real directory may be opened through a root
    /// alias, which is how a packaged app can reuse a development install
    /// without copying it. Keep the binding asymmetric: a receipt written for
    /// a symlink remains bound to that exact alias and cannot authorize opening
    /// its target by another path.
    private static func directoryBindingMatches(
        receiptPath: String,
        openedDirectory: URL
    ) -> Bool {
        let receiptURL = URL(
            fileURLWithPath: receiptPath,
            isDirectory: true
        ).standardizedFileURL
        let openedURL = openedDirectory.standardizedFileURL
        if receiptURL.path == openedURL.path { return true }
        guard (try? FileManager.default.destinationOfSymbolicLink(
            atPath: receiptURL.path
        )) == nil else { return false }
        return receiptURL.resolvingSymlinksInPath().path
            == openedURL.resolvingSymlinksInPath().path
    }
}
