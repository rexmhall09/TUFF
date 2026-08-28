import Foundation

/// Snapshot fingerprints pinned by the project. Adding a new entry means the
/// importer has been validated against a fresh upload of the source.
public enum SourceFingerprint {
    public static let knownFingerprints: [String: String] = Dictionary(
        uniqueKeysWithValues: SupportedModelSource.all.map {
            ($0.modelID, $0.sourceIndexSHA256)
        })

    /// Returns the recognised model ID for a given index.json SHA-256, or nil.
    public static func modelID(forIndexSha256 sha256Hex: String) -> String? {
        for (id, sha) in knownFingerprints where sha == sha256Hex { return id }
        return nil
    }

    /// Resolves a fingerprint only when it belongs to the requested supported
    /// repository. Model IDs are manifest identifiers and are not required to
    /// equal Hugging Face repository IDs (Qwen intentionally uses different
    /// values for the two).
    public static func modelID(forIndexSha256 sha256Hex: String,
                               repoID: String) -> String? {
        guard let modelID = modelID(forIndexSha256: sha256Hex),
              SupportedModelSource.all.contains(where: {
                  $0.repoID == repoID && $0.modelID == modelID
              }) else { return nil }
        return modelID
    }
}
