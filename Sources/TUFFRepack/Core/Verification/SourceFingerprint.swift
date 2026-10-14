import Foundation
import TUFFModelCatalog

/// Snapshot fingerprints pinned by the project. Adding a new entry means the
/// importer has been validated against a fresh upload of the source.
public enum SourceFingerprint {
    public static let knownFingerprints: [String: String] = Dictionary(
        SupportedModelSource.all.map { ($0.modelID, $0.sourceIndexSHA256) }
            + imagePackSources.map { ($0.modelID, $0.sourceIndexSHA256) },
        uniquingKeysWith: { first, _ in first })

    /// Image add-ons whose pack comes from a checkpoint of their own, because
    /// the text conversion dropped the vision tower.
    static let imagePackSources: [(repoID: String, modelID: String, sourceIndexSHA256: String)] =
        TUFFModelCatalog.all.flatMap { model in
            model.addons.filter { $0.kind == .imageInput }.map {
                (repoID: $0.source.repoID, modelID: $0.source.manifestModelID,
                 sourceIndexSHA256: $0.source.sourceIndexSHA256)
            }
        }

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
              }) || imagePackSources.contains(where: {
                  $0.repoID == repoID && $0.modelID == modelID
              }) else { return nil }
        return modelID
    }
}
