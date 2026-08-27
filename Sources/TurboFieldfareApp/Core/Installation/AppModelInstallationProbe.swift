import Foundation
import TurboFieldfare

public enum AppModelInstallationStatus: Equatable, Sendable {
    case missing
    case partial(String)
    case complete
}

public enum AppModelInstallationProbe {
    /// `descriptor` pins the checkpoint the installation must match. When it
    /// is nil the probe derives the expectation from the family the manifest
    /// itself declares, so a multi-model app validates whichever model is
    /// actually installed rather than whichever one is currently selected.
    public static func status(
        at directory: URL,
        descriptor: AppModelInstallDescriptor? = nil
    ) -> AppModelInstallationStatus {
        let directory = directory.standardizedFileURL
        let manifestURL = directory.appendingPathComponent("manifest.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            return .missing
        }

        do {
            let baseline = try ManifestReader.resolveArchitecture(
                directoryURL: directory)
            let manifest = try ManifestReader.load(directoryURL: directory, expecting: baseline)
            // Validate the checkpoint against the descriptor for the family the
            // manifest itself declares, so the probe does not depend on which
            // model the app happens to have selected.
            guard let expected = descriptor
                    ?? AppModelInstallDescriptor.descriptor(for: baseline.variant) else {
                return .partial("no descriptor for variant \(baseline.variant.rawValue)")
            }
            let expectedSource = "sha256:" + expected.sourceIndexSHA256
            guard manifest.sourceSnapshotHash == expectedSource else {
                return .partial("installed checkpoint does not match \(expected.displayName)")
            }
            // Only a mixture-of-experts install carries a packed-expert
            // directory. The dense format forbids one, so requiring it here
            // failed every completed Gemma 4 E4B download.
            if manifest.expertsPerLayer > 0 {
                let layout = directory.appendingPathComponent("packed_experts/layout.json")
                guard FileManager.default.fileExists(atPath: layout.path) else {
                    return .partial("packed_experts/layout.json is missing")
                }
            }
            let receipt = try VerifiedInstallReceiptReader.load(directoryURL: directory)
            let manifestHash = try Sha256Verifier.hashFile(at: manifestURL, chunkBytes: 65_536)
            try VerifiedInstallReceiptReader.validateManifestBinding(
                receipt,
                directoryURL: directory,
                manifestSha256: manifestHash)
            return .complete
        } catch {
            return .partial("\(error)")
        }
    }
}
