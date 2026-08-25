import Foundation
import TUFFModelCatalog
import TurboFieldfare
import TurboFieldfareRepackCore

public enum AppReasoningControl: Equatable, Sendable {
    case toggle
    case toggleWithPreservation
    case graded
}

public struct AppModelInstallDescriptor: Equatable, Sendable {
    public let displayName: String
    public let repoID: String
    public let revision: String
    public let sourceIndexSHA256: String
    public let approximateDownloadBytes: UInt64
    public let installedBytes: UInt64
    public let rangeStagingBytes: UInt64
    public let reserveBytes: UInt64

    public init(catalog descriptor: TUFFModelDescriptor) {
        let source = descriptor.source
        self.init(
            displayName: descriptor.displayName,
            repoID: source.repoID,
            revision: source.revision,
            sourceIndexSHA256: source.sourceIndexSHA256,
            approximateDownloadBytes: source.approximateDownloadBytes,
            installedBytes: source.installedBytes,
            rangeStagingBytes: UInt64(RemoteChunkPolicy.defaultBytes),
            reserveBytes: source.reserveBytes)
    }

    public init(addon: TUFFModelAddonDescriptor) {
        let source = addon.source
        self.init(
            displayName: addon.displayName,
            repoID: source.repoID,
            revision: source.revision,
            sourceIndexSHA256: source.sourceIndexSHA256,
            approximateDownloadBytes: source.approximateDownloadBytes,
            installedBytes: source.installedBytes,
            rangeStagingBytes: UInt64(RemoteChunkPolicy.defaultBytes),
            reserveBytes: source.reserveBytes)
    }

    public init(displayName: String,
                repoID: String,
                revision: String,
                sourceIndexSHA256: String,
                approximateDownloadBytes: UInt64,
                installedBytes: UInt64,
                rangeStagingBytes: UInt64,
                reserveBytes: UInt64) {
        self.displayName = displayName
        self.repoID = repoID
        self.revision = revision
        self.sourceIndexSHA256 = sourceIndexSHA256
        self.approximateDownloadBytes = approximateDownloadBytes
        self.installedBytes = installedBytes
        self.rangeStagingBytes = rangeStagingBytes
        self.reserveBytes = reserveBytes
    }

    public var requiredFreeBytes: UInt64 {
        installedBytes + rangeStagingBytes + reserveBytes
    }

    public static let `default` = AppModelInstallDescriptor(
        catalog: TUFFModelCatalog.gemma4_26B_A4B)

    public static let gemma4E4B = AppModelInstallDescriptor(
        catalog: TUFFModelCatalog.gemma4_E4B)

    public static let qwen36 = AppModelInstallDescriptor(
        catalog: TUFFModelCatalog.qwen36_35B_A3B)

    /// The shipped descriptor for a model family, if one exists.
    public static func descriptor(for family: ModelFamily) -> AppModelInstallDescriptor? {
        catalog.first { $0.family == family }
    }

    public static func descriptor(for variant: ModelVariant) -> AppModelInstallDescriptor? {
        catalog.first { $0.catalogDescriptor?.architecture.id.rawValue == variant.rawValue }
    }

    /// Basename of the installed `.gturbo` directory for this descriptor.
    public var installDirectoryName: String {
        catalogDescriptor?.installDirectoryName
            ?? (self == .qwen36 ? "qwen36.gturbo" : "gemma4.gturbo")
    }

    /// Stable identity for catalog lookups and UI selection. The install
    /// directory basename already distinguishes the shipped models and is what
    /// separates their downloads on disk, so it is the natural key.
    public var id: String { installDirectoryName }

    /// Stable registry identity used by settings and other persisted app data.
    /// Test-only/custom descriptors keep their install basename as a safe
    /// fallback so they never collide with a shipped model profile.
    public var catalogID: TUFFModelID? {
        catalogDescriptor?.id
    }

    public var settingsProfileKey: String {
        catalogID?.rawValue ?? id
    }

    /// The model family this descriptor installs.
    public var family: ModelFamily {
        catalogDescriptor.flatMap { ModelFamily(rawValue: $0.family.rawValue) }
            ?? (self == .qwen36 ? .qwen36 : .gemma4)
    }

    /// Whether this exact checkpoint has a qualified image companion. Custom
    /// descriptors retain the v1 behavior for tests and local builds.
    public var supportsImageInput: Bool {
        catalogDescriptor?.capabilities.contains(.imageInput) ?? true
    }

    public var reasoningControl: AppReasoningControl? {
        switch catalogDescriptor?.reasoningControl {
        case .toggle: .toggle
        case .toggleWithPreservation: .toggleWithPreservation
        case .graded: .graded
        case nil: nil
        }
    }

    /// Every model the app can install, in the order the UI lists them.
    public static let catalog: [AppModelInstallDescriptor] =
        TUFFModelCatalog.all.map(AppModelInstallDescriptor.init(catalog:))

    /// A short line describing what the model is good for, shown beside its
    /// name in the picker.
    public var summary: String {
        catalogDescriptor?.summary ?? displayName
    }

    public var shortName: String {
        catalogDescriptor?.shortName ?? displayName
    }

    /// The descriptor the app products select at launch. Defaults to Gemma 4.
    /// `TURBO_FIELDFARE_MODEL=qwen36` in the environment wins; otherwise the
    /// persisted preference (`defaults write TurboFieldfare model qwen36`)
    /// applies, so GUI launches without an environment also select Qwen.
    public static var selected: AppModelInstallDescriptor {
        let environmentValue = ProcessInfo.processInfo.environment["TURBO_FIELDFARE_MODEL"]
        let preferenceValue = UserDefaults(suiteName: "TurboFieldfare")?
            .string(forKey: "model")
        guard let selector = environmentValue ?? preferenceValue,
              let descriptor = TUFFModelCatalog.model(selector: selector) else {
            return .default
        }
        return AppModelInstallDescriptor(catalog: descriptor)
    }

    public static let visionCompanion = AppModelInstallDescriptor(
        addon: TUFFModelCatalog.gemma4_26B_A4B.addons[0])

    public static let qwen36VisionCompanion = AppModelInstallDescriptor(
        addon: TUFFModelCatalog.qwen36_35B_A3B.addons[0])

    public static func visionCompanion(
        for family: ModelFamily
    ) -> AppModelInstallDescriptor {
        switch family {
        case .gemma4: return .visionCompanion
        case .qwen36: return .qwen36VisionCompanion
        case .gptOss:
            preconditionFailure("GPT-OSS does not have a vision companion")
        }
    }

    private var catalogDescriptor: TUFFModelDescriptor? {
        TUFFModelCatalog.all.first {
            $0.source.repoID == repoID
                && $0.source.revision == revision
                && $0.source.sourceIndexSHA256 == sourceIndexSHA256
                && $0.source.installedBytes == installedBytes
        }
    }
}

public struct AppModelInstallRequirement: Equatable, Sendable {
    public let probePath: String
    public let requiredBytes: UInt64
    public let availableBytes: UInt64

    public init(probePath: String = "", requiredBytes: UInt64, availableBytes: UInt64) {
        self.probePath = probePath
        self.requiredBytes = requiredBytes
        self.availableBytes = availableBytes
    }

    public var canInstall: Bool { availableBytes >= requiredBytes }

    public var shortfallBytes: UInt64 {
        requiredBytes > availableBytes ? requiredBytes - availableBytes : 0
    }
}

public enum AppModelInstallReadiness: Equatable, Sendable {
    case checking
    case ready(AppModelInstallRequirement)
    case insufficientSpace(AppModelInstallRequirement)
    case failed(String)

    public var requirement: AppModelInstallRequirement? {
        switch self {
        case .ready(let requirement), .insufficientSpace(let requirement):
            return requirement
        case .checking, .failed:
            return nil
        }
    }
}
