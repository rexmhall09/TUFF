import Foundation

public enum TUFFModelID: String, Codable, CaseIterable, Sendable {
    case gemma4_26B_A4B = "gemma4-26b-a4b"
    case qwen36_35B_A3B = "qwen36-35b-a3b"
}

public enum TUFFModelFamily: String, Codable, Sendable {
    case gemma4
    case qwen36
}

public enum TUFFReasoningControl: String, Codable, Sendable {
    case toggle
    case toggleWithPreservation
}

public enum TUFFModelCapability: String, Codable, Sendable {
    case textGeneration
    case imageInput
    case reasoning
}

public enum TUFFModelAddonKind: String, Codable, Sendable {
    case imageInput
}

public struct TUFFModelSource: Codable, Equatable, Sendable {
    public let repoID: String
    public let revision: String
    public let sourceIndexSHA256: String
    public let manifestModelID: String
    public let approximateDownloadBytes: UInt64
    public let installedBytes: UInt64
    public let reserveBytes: UInt64

    public init(repoID: String,
                revision: String,
                sourceIndexSHA256: String,
                manifestModelID: String,
                approximateDownloadBytes: UInt64,
                installedBytes: UInt64,
                reserveBytes: UInt64) {
        self.repoID = repoID
        self.revision = revision
        self.sourceIndexSHA256 = sourceIndexSHA256
        self.manifestModelID = manifestModelID
        self.approximateDownloadBytes = approximateDownloadBytes
        self.installedBytes = installedBytes
        self.reserveBytes = reserveBytes
    }
}

public struct TUFFModelHardwareRequirements: Codable, Equatable, Sendable {
    public let minimumUnifiedMemoryBytes: UInt64
    public let minimumMacOSMajorVersion: Int
    public let minimumAppleSiliconGeneration: Int

    public init(minimumUnifiedMemoryBytes: UInt64,
                minimumMacOSMajorVersion: Int = 15,
                minimumAppleSiliconGeneration: Int = 1) {
        self.minimumUnifiedMemoryBytes = minimumUnifiedMemoryBytes
        self.minimumMacOSMajorVersion = minimumMacOSMajorVersion
        self.minimumAppleSiliconGeneration = minimumAppleSiliconGeneration
    }
}

public struct TUFFModelRuntimeDefaults: Codable, Equatable, Sendable {
    public let contextTokens: Int
    public let expertCacheSlots: Int
    public let temperature: Double
    public let topK: Int
    public let topP: Double

    public init(contextTokens: Int = 8_192,
                expertCacheSlots: Int = 16,
                temperature: Double = 0.2,
                topK: Int = 64,
                topP: Double = 0.95) {
        self.contextTokens = contextTokens
        self.expertCacheSlots = expertCacheSlots
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
    }
}

public struct TUFFModelAddonDescriptor: Codable, Equatable, Sendable {
    public let id: String
    public let displayName: String
    public let kind: TUFFModelAddonKind
    public let source: TUFFModelSource
    public let hardware: TUFFModelHardwareRequirements

    public init(id: String,
                displayName: String,
                kind: TUFFModelAddonKind,
                source: TUFFModelSource,
                hardware: TUFFModelHardwareRequirements) {
        self.id = id
        self.displayName = displayName
        self.kind = kind
        self.source = source
        self.hardware = hardware
    }
}

public struct TUFFModelDescriptor: Codable, Equatable, Sendable, Identifiable {
    public let id: TUFFModelID
    public let selector: String
    public let aliases: [String]
    public let displayName: String
    public let shortName: String
    public let summary: String
    public let family: TUFFModelFamily
    public let installDirectoryName: String
    public let source: TUFFModelSource
    public let hardware: TUFFModelHardwareRequirements
    public let runtimeDefaults: TUFFModelRuntimeDefaults
    public let capabilities: Set<TUFFModelCapability>
    public let reasoningControl: TUFFReasoningControl?
    public let addons: [TUFFModelAddonDescriptor]

    public init(id: TUFFModelID,
                selector: String,
                aliases: [String] = [],
                displayName: String,
                shortName: String,
                summary: String,
                family: TUFFModelFamily,
                installDirectoryName: String,
                source: TUFFModelSource,
                hardware: TUFFModelHardwareRequirements,
                runtimeDefaults: TUFFModelRuntimeDefaults = TUFFModelRuntimeDefaults(),
                capabilities: Set<TUFFModelCapability>,
                reasoningControl: TUFFReasoningControl?,
                addons: [TUFFModelAddonDescriptor] = []) {
        self.id = id
        self.selector = selector
        self.aliases = aliases
        self.displayName = displayName
        self.shortName = shortName
        self.summary = summary
        self.family = family
        self.installDirectoryName = installDirectoryName
        self.source = source
        self.hardware = hardware
        self.runtimeDefaults = runtimeDefaults
        self.capabilities = capabilities
        self.reasoningControl = reasoningControl
        self.addons = addons
    }
}

public enum TUFFModelCatalog {
    public static let oneGiB: UInt64 = 1_073_741_824
    public static let eightGiB: UInt64 = 8 * oneGiB

    private static let gemmaSource = TUFFModelSource(
        repoID: "mlx-community/gemma-4-26b-a4b-it-4bit",
        revision: "0d77464eeb233a2da68ebf9d7dc4edaac7db956d",
        sourceIndexSHA256:
            "bf198c9f5ea6462addca1966e5dd669c407537a876e82cf06db9084c5c850b13",
        manifestModelID: "mlx-community/gemma-4-26b-a4b-it-4bit",
        approximateDownloadBytes: 14_620_479_420,
        installedBytes: 14_291_921_884,
        reserveBytes: oneGiB)

    private static let qwenSource = TUFFModelSource(
        repoID: "mlx-community/Qwen3.6-35B-A3B-4bit",
        revision: "38740b847e4cb78f352aba30aa41c76e08e6eb46",
        sourceIndexSHA256:
            "0b28df60e33753a14e816d3b31577ae2c93884c58430a4a6de6ae9ea483842ea",
        manifestModelID: "qwen3.6-35b-a3b-4bit",
        approximateDownloadBytes: 19_529_025_048,
        installedBytes: 19_546_491_213,
        reserveBytes: oneGiB)

    public static let gemma4_26B_A4B = TUFFModelDescriptor(
        id: .gemma4_26B_A4B,
        selector: "gemma4",
        displayName: "Gemma 4 26B-A4B IT 4-bit",
        shortName: "Gemma 4 26B",
        summary: "26B total, 3.9B active. The original TurboFieldfare target; "
            + "smallest install and lowest resident footprint.",
        family: .gemma4,
        installDirectoryName: "gemma4.gturbo",
        source: gemmaSource,
        hardware: TUFFModelHardwareRequirements(minimumUnifiedMemoryBytes: eightGiB),
        capabilities: [.textGeneration, .imageInput, .reasoning],
        reasoningControl: .toggle,
        addons: [TUFFModelAddonDescriptor(
            id: "gemma4-image-input",
            displayName: "Gemma 4 Image Support",
            kind: .imageInput,
            source: TUFFModelSource(
                repoID: gemmaSource.repoID,
                revision: gemmaSource.revision,
                sourceIndexSHA256: gemmaSource.sourceIndexSHA256,
                manifestModelID: gemmaSource.manifestModelID,
                approximateDownloadBytes: 1_539_478_890,
                installedBytes: 1_148_567_552,
                reserveBytes: oneGiB),
            hardware: TUFFModelHardwareRequirements(
                minimumUnifiedMemoryBytes: eightGiB,
                minimumAppleSiliconGeneration: 2))])

    public static let qwen36_35B_A3B = TUFFModelDescriptor(
        id: .qwen36_35B_A3B,
        selector: "qwen36",
        displayName: "Qwen3.6 35B-A3B 4-bit",
        shortName: "Qwen3.6 35B",
        summary: "35B total, 3B active. Hybrid linear/full attention, so its "
            + "KV cache stays small at long context; larger install.",
        family: .qwen36,
        installDirectoryName: "qwen36.gturbo",
        source: qwenSource,
        hardware: TUFFModelHardwareRequirements(minimumUnifiedMemoryBytes: eightGiB),
        capabilities: [.textGeneration, .imageInput, .reasoning],
        reasoningControl: .toggleWithPreservation,
        addons: [TUFFModelAddonDescriptor(
            id: "qwen36-image-input",
            displayName: "Qwen3.6 Image Support",
            kind: .imageInput,
            source: TUFFModelSource(
                repoID: qwenSource.repoID,
                revision: qwenSource.revision,
                sourceIndexSHA256: qwenSource.sourceIndexSHA256,
                manifestModelID: qwenSource.manifestModelID,
                approximateDownloadBytes: 893_142_496,
                installedBytes: 900_808_704,
                reserveBytes: oneGiB),
            hardware: TUFFModelHardwareRequirements(
                minimumUnifiedMemoryBytes: eightGiB,
                minimumAppleSiliconGeneration: 2))])

    public static let all: [TUFFModelDescriptor] = [gemma4_26B_A4B, qwen36_35B_A3B]
    public static let `default` = gemma4_26B_A4B

    public static func model(id: TUFFModelID) -> TUFFModelDescriptor? {
        all.first { $0.id == id }
    }

    public static func model(selector: String) -> TUFFModelDescriptor? {
        all.first { $0.selector == selector || $0.aliases.contains(selector) }
    }

    public static func model(manifestModelID: String) -> TUFFModelDescriptor? {
        all.first { $0.source.manifestModelID == manifestModelID }
    }
}
