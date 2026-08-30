import Foundation
import TUFFModelCatalog

/// A pinned upstream checkpoint the installer knows how to repack. Each value
/// fixes the repo, revision and index fingerprint so installs are exactly
/// reproducible.
public struct SupportedModelSource: Sendable, Equatable {
    /// CLI selector value (`--model <name>`).
    public let name: String
    public let aliases: [String]
    public let displayName: String
    public let repoID: String
    public let revision: String
    public let sourceIndexSHA256: String
    /// Value recorded as `manifest.modelID` when the source fingerprint matches.
    public let modelID: String
    public let approximateDownloadBytes: UInt64
    public let installedBytes: UInt64
    public let reserveBytes: UInt64

    public init(catalog descriptor: TUFFModelDescriptor) {
        let source = descriptor.source
        self.name = descriptor.selector
        self.aliases = descriptor.aliases
        self.displayName = descriptor.displayName
        self.repoID = source.repoID
        self.revision = source.revision
        self.sourceIndexSHA256 = source.sourceIndexSHA256
        self.modelID = source.manifestModelID
        self.approximateDownloadBytes = source.approximateDownloadBytes
        self.installedBytes = source.installedBytes
        self.reserveBytes = source.reserveBytes
    }

    public func installOptions(outputDirectory: URL,
                               overwrite: Bool,
                               token: String?,
                               resume: Bool = false)
        -> RemoteStreamingRepackOptions {
        RemoteStreamingRepackOptions(
            repoID: repoID,
            revision: revision,
            outputDir: outputDirectory.path,
            token: token,
            requireKnownSource: true,
            minFreeReserveBytes: reserveBytes,
            overwrite: overwrite,
            resume: resume)
    }

    public static let gemma4 = SupportedModelSource(catalog: TUFFModelCatalog.gemma4_26B_A4B)
    public static let gemma4E2B = SupportedModelSource(catalog: TUFFModelCatalog.gemma4_E2B)
    public static let gemma4E4B = SupportedModelSource(catalog: TUFFModelCatalog.gemma4_E4B)
    public static let gemma4_12B_QAT = SupportedModelSource(
        catalog: TUFFModelCatalog.gemma4_12B_QAT)

    /// Download estimate covers the `language_model.*` tensors plus tokenizer
    /// and metadata sidecars; the vision tower is never fetched. Installed
    /// bytes add the resident index and per-expert 16 KB page rounding
    /// (the 1,769,472-byte expert blob is already page-aligned) plus
    /// layout/manifest sidecars.
    public static let qwen36 = SupportedModelSource(catalog: TUFFModelCatalog.qwen36_35B_A3B)
    public static let gptOss20B = SupportedModelSource(catalog: TUFFModelCatalog.gptOss_20B)
    public static let gptOss120B = SupportedModelSource(catalog: TUFFModelCatalog.gptOss_120B)

    /// Default source when no `--model` selector is given.
    public static let `default` = gemma4

    public static let all: [SupportedModelSource] =
        TUFFModelCatalog.all.map(SupportedModelSource.init(catalog:))

    public static func named(_ name: String) -> SupportedModelSource? {
        all.first { $0.name == name || $0.aliases.contains(name) }
    }
}
