import Foundation
import TUFFModelCatalog
import TurboFieldfare
import TurboFieldfareRepackCore

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

    public static let qwen36 = AppModelInstallDescriptor(
        catalog: TUFFModelCatalog.qwen36_35B_A3B)

    /// The shipped descriptor for a model family, if one exists.
    public static func descriptor(for family: ModelFamily) -> AppModelInstallDescriptor? {
        switch family {
        case .gemma4: return .default
        case .qwen36: return .qwen36
        }
    }

    /// Basename of the installed `.gturbo` directory for this descriptor.
    public var installDirectoryName: String {
        self == .qwen36 ? "qwen36.gturbo" : "gemma4.gturbo"
    }

    /// Stable identity for catalog lookups and UI selection. The install
    /// directory basename already distinguishes the shipped models and is what
    /// separates their downloads on disk, so it is the natural key.
    public var id: String { installDirectoryName }

    /// The model family this descriptor installs.
    public var family: ModelFamily {
        self == .qwen36 ? .qwen36 : .gemma4
    }

    /// Every model the app can install, in the order the UI lists them.
    public static let catalog: [AppModelInstallDescriptor] = [.default, .qwen36]

    /// A short line describing what the model is good for, shown beside its
    /// name in the picker.
    public var summary: String {
        switch family {
        case .gemma4:
            return "26B total, 3.9B active. The original TurboFieldfare target; "
                + "smallest install and lowest resident footprint."
        case .qwen36:
            return "35B total, 3B active. Hybrid linear/full attention, so its "
                + "KV cache stays small at long context; larger install."
        }
    }

    /// The descriptor the app products select at launch. Defaults to Gemma 4.
    /// `TURBO_FIELDFARE_MODEL=qwen36` in the environment wins; otherwise the
    /// persisted preference (`defaults write TurboFieldfare model qwen36`)
    /// applies, so GUI launches without an environment also select Qwen.
    public static var selected: AppModelInstallDescriptor {
        let environmentValue = ProcessInfo.processInfo.environment["TURBO_FIELDFARE_MODEL"]
        let preferenceValue = UserDefaults(suiteName: "TurboFieldfare")?
            .string(forKey: "model")
        switch environmentValue ?? preferenceValue {
        case "qwen36": return .qwen36
        default: return .default
        }
    }

    public static let visionCompanion = AppModelInstallDescriptor(
        addon: TUFFModelCatalog.gemma4_26B_A4B.addons[0])

    public static let qwen36VisionCompanion = AppModelInstallDescriptor(
        addon: TUFFModelCatalog.qwen36_35B_A3B.addons[0])

    public static func visionCompanion(
        for family: ModelFamily
    ) -> AppModelInstallDescriptor {
        family == .qwen36 ? .qwen36VisionCompanion : .visionCompanion
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
