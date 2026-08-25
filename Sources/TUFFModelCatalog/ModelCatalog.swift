import Foundation
import Darwin

public enum TUFFModelID: String, Codable, CaseIterable, Sendable {
    case gemma4_E4B = "gemma4-e4b"
    case gemma4_26B_A4B = "gemma4-26b-a4b"
    case qwen36_35B_A3B = "qwen36-35b-a3b"
    case gptOss_20B = "gpt-oss-20b"
    case gptOss_120B = "gpt-oss-120b"
}

public enum TUFFModelFamily: String, Codable, Sendable {
    case gemma4
    case qwen36
    case gptOss = "gpt-oss"
}

public enum TUFFArchitectureID: String, Codable, Sendable {
    case gemma4_E4B = "gemma4-e4b"
    case gemma4_26B_A4B = "gemma4-26b-a4b"
    case qwen36_35B_A3B = "qwen36-35b-a3b"
    case gptOss_20B = "gpt-oss-20b"
    case gptOss_120B = "gpt-oss-120b"
}

public enum TUFFFeedForwardKind: String, Codable, Sendable {
    case dense
    case mixtureOfExperts = "moe"
}

public enum TUFFWeightLayout: String, Codable, Sendable {
    case affine
    case mxfp4
}

public struct TUFFArchitectureProfile: Codable, Equatable, Sendable {
    public let id: TUFFArchitectureID
    public let family: TUFFModelFamily
    public let feedForwardKind: TUFFFeedForwardKind
    public let weightLayout: TUFFWeightLayout

    public init(id: TUFFArchitectureID,
                family: TUFFModelFamily,
                feedForwardKind: TUFFFeedForwardKind,
                weightLayout: TUFFWeightLayout) {
        self.id = id
        self.family = family
        self.feedForwardKind = feedForwardKind
        self.weightLayout = weightLayout
    }
}

public extension TUFFArchitectureProfile {
    static let gemma4E4B = TUFFArchitectureProfile(
        id: .gemma4_E4B,
        family: .gemma4,
        feedForwardKind: .dense,
        weightLayout: .affine)

    static let gemma4_26B_A4B = TUFFArchitectureProfile(
        id: .gemma4_26B_A4B,
        family: .gemma4,
        feedForwardKind: .mixtureOfExperts,
        weightLayout: .affine)

    static let qwen36_35B_A3B = TUFFArchitectureProfile(
        id: .qwen36_35B_A3B,
        family: .qwen36,
        feedForwardKind: .mixtureOfExperts,
        weightLayout: .affine)

    static let gptOss_20B = TUFFArchitectureProfile(
        id: .gptOss_20B,
        family: .gptOss,
        feedForwardKind: .mixtureOfExperts,
        weightLayout: .mxfp4)

    static let gptOss_120B = TUFFArchitectureProfile(
        id: .gptOss_120B,
        family: .gptOss,
        feedForwardKind: .mixtureOfExperts,
        weightLayout: .mxfp4)
}

public enum TUFFReasoningControl: String, Codable, Sendable {
    case toggle
    case toggleWithPreservation
    case graded
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

public struct TUFFKVCacheProfile: Codable, Equatable, Sendable {
    public let fullAttentionBytesPerToken: UInt64
    public let slidingAttentionBytesPerToken: UInt64
    public let slidingWindowCapacityTokens: Int

    public init(fullAttentionBytesPerToken: UInt64,
                slidingAttentionBytesPerToken: UInt64 = 0,
                slidingWindowCapacityTokens: Int = 0) {
        self.fullAttentionBytesPerToken = fullAttentionBytesPerToken
        self.slidingAttentionBytesPerToken = slidingAttentionBytesPerToken
        self.slidingWindowCapacityTokens = slidingWindowCapacityTokens
    }

    public func estimatedBytes(contextTokens: Int) -> UInt64 {
        guard contextTokens > 0 else { return 0 }
        let full = fullAttentionBytesPerToken.multipliedReportingOverflow(
            by: UInt64(contextTokens))
        let slidingRows = min(contextTokens, max(0, slidingWindowCapacityTokens))
        let sliding = slidingAttentionBytesPerToken.multipliedReportingOverflow(
            by: UInt64(slidingRows))
        guard !full.overflow, !sliding.overflow else { return .max }
        let total = full.partialValue.addingReportingOverflow(sliding.partialValue)
        return total.overflow ? .max : total.partialValue
    }
}

public struct TUFFModelMemoryProfile: Codable, Equatable, Sendable {
    public let qualifiedDefaultWorkingSetBytes: UInt64
    public let defaultContextTokens: Int
    public let defaultExpertCacheSlots: Int
    public let expertCacheBytesPerSlot: UInt64
    public let kvCache: TUFFKVCacheProfile

    public init(qualifiedDefaultWorkingSetBytes: UInt64,
                defaultContextTokens: Int = 8_192,
                defaultExpertCacheSlots: Int = 16,
                expertCacheBytesPerSlot: UInt64,
                kvCache: TUFFKVCacheProfile) {
        self.qualifiedDefaultWorkingSetBytes = qualifiedDefaultWorkingSetBytes
        self.defaultContextTokens = defaultContextTokens
        self.defaultExpertCacheSlots = defaultExpertCacheSlots
        self.expertCacheBytesPerSlot = expertCacheBytesPerSlot
        self.kvCache = kvCache
    }

    public func estimatedWorkingSetBytes(contextTokens: Int,
                                         expertCacheSlots: Int) -> UInt64 {
        let defaultKV = kvCache.estimatedBytes(contextTokens: defaultContextTokens)
        let requestedKV = kvCache.estimatedBytes(contextTokens: contextTokens)
        let defaultSlots = expertCacheBytesPerSlot.multipliedReportingOverflow(
            by: UInt64(max(0, defaultExpertCacheSlots)))
        let requestedSlots = expertCacheBytesPerSlot.multipliedReportingOverflow(
            by: UInt64(max(0, expertCacheSlots)))
        guard defaultKV != .max, requestedKV != .max,
              !defaultSlots.overflow, !requestedSlots.overflow else { return .max }

        let removableKV = min(defaultKV, qualifiedDefaultWorkingSetBytes)
        let afterKV = qualifiedDefaultWorkingSetBytes - removableKV
        let removableSlots = min(defaultSlots.partialValue, afterKV)
        var estimate = afterKV - removableSlots
        for value in [requestedKV, requestedSlots.partialValue] {
            let addition = estimate.addingReportingOverflow(value)
            guard !addition.overflow else { return .max }
            estimate = addition.partialValue
        }
        return estimate
    }
}

public struct TUFFDeviceCapabilities: Codable, Equatable, Sendable {
    public let unifiedMemoryBytes: UInt64
    public let macOSMajorVersion: Int
    public let appleSiliconGeneration: Int

    public init(unifiedMemoryBytes: UInt64,
                macOSMajorVersion: Int,
                appleSiliconGeneration: Int) {
        self.unifiedMemoryBytes = unifiedMemoryBytes
        self.macOSMajorVersion = macOSMajorVersion
        self.appleSiliconGeneration = appleSiliconGeneration
    }

    public static func current() -> TUFFDeviceCapabilities {
        let memory = sysctlUInt64(named: "hw.memsize") ?? ProcessInfo.processInfo.physicalMemory
        let generation = sysctlString(named: "machdep.cpu.brand_string")
            .flatMap(appleSiliconGeneration(brandString:)) ?? 1
        return TUFFDeviceCapabilities(
            unifiedMemoryBytes: memory,
            macOSMajorVersion: ProcessInfo.processInfo.operatingSystemVersion.majorVersion,
            appleSiliconGeneration: generation)
    }

    static func appleSiliconGeneration(brandString: String) -> Int? {
        let normalized = brandString.lowercased()
        guard let marker = normalized.range(of: "apple m") else { return nil }
        let suffix = normalized[marker.upperBound...]
        let digits = suffix.prefix { $0.isNumber }
        guard !digits.isEmpty else { return nil }
        return Int(digits)
    }

    private static func sysctlUInt64(named name: String) -> UInt64? {
        var value: UInt64 = 0
        var size = MemoryLayout<UInt64>.size
        guard sysctlbyname(name, &value, &size, nil, 0) == 0,
              size == MemoryLayout<UInt64>.size else { return nil }
        return value
    }

    private static func sysctlString(named name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 1 else { return nil }
        var bytes = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &bytes, &size, nil, 0) == 0 else { return nil }
        return String(
            decoding: bytes.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) },
            as: UTF8.self)
    }
}

public enum TUFFCompatibilityIssue: Codable, Equatable, Sendable {
    case insufficientUnifiedMemory(requiredBytes: UInt64, actualBytes: UInt64)
    case unsupportedMacOS(requiredMajorVersion: Int, actualMajorVersion: Int)
    case unsupportedAppleSilicon(requiredGeneration: Int, actualGeneration: Int)
    case contextExceedsSafeMemory(estimatedBytes: UInt64, budgetBytes: UInt64)
}

public struct TUFFModelCompatibility: Codable, Equatable, Sendable {
    public let issues: [TUFFCompatibilityIssue]

    public init(issues: [TUFFCompatibilityIssue]) {
        self.issues = issues
    }

    public var isCompatible: Bool { issues.isEmpty }
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
    public let apiModelID: String
    public let displayName: String
    public let shortName: String
    public let summary: String
    public let family: TUFFModelFamily
    public let architecture: TUFFArchitectureProfile
    public let installDirectoryName: String
    public let source: TUFFModelSource
    public let hardware: TUFFModelHardwareRequirements
    public let memory: TUFFModelMemoryProfile
    public let runtimeDefaults: TUFFModelRuntimeDefaults
    public let capabilities: Set<TUFFModelCapability>
    public let reasoningControl: TUFFReasoningControl?
    public let addons: [TUFFModelAddonDescriptor]

    public init(id: TUFFModelID,
                selector: String,
                aliases: [String] = [],
                apiModelID: String,
                displayName: String,
                shortName: String,
                summary: String,
                family: TUFFModelFamily,
                architecture: TUFFArchitectureProfile,
                installDirectoryName: String,
                source: TUFFModelSource,
                hardware: TUFFModelHardwareRequirements,
                memory: TUFFModelMemoryProfile,
                runtimeDefaults: TUFFModelRuntimeDefaults = TUFFModelRuntimeDefaults(),
                capabilities: Set<TUFFModelCapability>,
                reasoningControl: TUFFReasoningControl?,
                addons: [TUFFModelAddonDescriptor] = []) {
        self.id = id
        self.selector = selector
        self.aliases = aliases
        self.apiModelID = apiModelID
        self.displayName = displayName
        self.shortName = shortName
        self.summary = summary
        self.family = family
        self.architecture = architecture
        self.installDirectoryName = installDirectoryName
        self.source = source
        self.hardware = hardware
        self.memory = memory
        self.runtimeDefaults = runtimeDefaults
        self.capabilities = capabilities
        self.reasoningControl = reasoningControl
        self.addons = addons
    }

    public func compatibility(
        with device: TUFFDeviceCapabilities,
        contextTokens: Int? = nil,
        expertCacheSlots: Int? = nil
    ) -> TUFFModelCompatibility {
        var issues: [TUFFCompatibilityIssue] = []
        if device.unifiedMemoryBytes < hardware.minimumUnifiedMemoryBytes {
            issues.append(.insufficientUnifiedMemory(
                requiredBytes: hardware.minimumUnifiedMemoryBytes,
                actualBytes: device.unifiedMemoryBytes))
        }
        if device.macOSMajorVersion < hardware.minimumMacOSMajorVersion {
            issues.append(.unsupportedMacOS(
                requiredMajorVersion: hardware.minimumMacOSMajorVersion,
                actualMajorVersion: device.macOSMajorVersion))
        }
        if device.appleSiliconGeneration < hardware.minimumAppleSiliconGeneration {
            issues.append(.unsupportedAppleSilicon(
                requiredGeneration: hardware.minimumAppleSiliconGeneration,
                actualGeneration: device.appleSiliconGeneration))
        }

        if let contextTokens {
            let estimate = memory.estimatedWorkingSetBytes(
                contextTokens: contextTokens,
                expertCacheSlots: expertCacheSlots ?? runtimeDefaults.expertCacheSlots)
            let twentyPercent = device.unifiedMemoryBytes / 5
            let reserve = max(2 * TUFFModelCatalog.oneGiB, twentyPercent)
            let budget = device.unifiedMemoryBytes > reserve
                ? device.unifiedMemoryBytes - reserve : 0
            if estimate > budget {
                issues.append(.contextExceedsSafeMemory(
                    estimatedBytes: estimate,
                    budgetBytes: budget))
            }
        }
        return TUFFModelCompatibility(issues: issues)
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

    private static let gemmaE4BSource = TUFFModelSource(
        repoID: "mlx-community/gemma-4-e4b-it-4bit",
        revision: "475b9088d29754a3379866cf5aeb6b41acd313c2",
        sourceIndexSHA256:
            "f8accac59ee7efe87e0c298c854610b262c3cadd477407503147c71209ff0093",
        manifestModelID: "mlx-community/gemma-4-e4b-it-4bit",
        approximateDownloadBytes: 4_231_600_000,
        installedBytes: 4_231_300_000,
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

    private static let gptOss20BSource = TUFFModelSource(
        repoID: "openai/gpt-oss-20b",
        revision: "6cee5e81ee83917806bbde320786a8fb61efebee",
        sourceIndexSHA256:
            "0e085b977c4c9942f85938828e8c989ed7d5cdabf852e4da6a67c116cd502cd1",
        manifestModelID: "openai/gpt-oss-20b",
        approximateDownloadBytes: 13_790_000_000,
        installedBytes: 13_792_000_000,
        reserveBytes: oneGiB)

    private static let gptOss120BSource = TUFFModelSource(
        repoID: "openai/gpt-oss-120b",
        revision: "b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
        sourceIndexSHA256:
            "ede2655fdc05008561983b6e0829c600727c28d591e071077377059f03a6c00e",
        manifestModelID: "openai/gpt-oss-120b",
        approximateDownloadBytes: 65_300_000_000,
        installedBytes: 65_400_000_000,
        reserveBytes: oneGiB)

    /// Small text-only launch model. Image and audio remain intentionally
    /// absent until separately packaged add-ons pass their own qualification.
    public static let gemma4_E4B = TUFFModelDescriptor(
        id: .gemma4_E4B,
        selector: "gemma4-e4b",
        aliases: ["e4b"],
        apiModelID: "gemma-4-e4b-it",
        displayName: "Gemma 4 E4B IT 4-bit",
        shortName: "Gemma 4 E4B",
        summary: "Small dense Gemma with per-layer embeddings and shared KV projections.",
        family: .gemma4,
        architecture: .gemma4E4B,
        installDirectoryName: "gemma4-e4b.gturbo",
        source: gemmaE4BSource,
        hardware: TUFFModelHardwareRequirements(minimumUnifiedMemoryBytes: eightGiB),
        memory: TUFFModelMemoryProfile(
            qualifiedDefaultWorkingSetBytes: 1_833_438_160,
            defaultExpertCacheSlots: 0,
            expertCacheBytesPerSlot: 0,
            kvCache: TUFFKVCacheProfile(
                fullAttentionBytesPerToken: 16_384,
                slidingAttentionBytesPerToken: 40_960,
                slidingWindowCapacityTokens: 512)),
        runtimeDefaults: TUFFModelRuntimeDefaults(
            contextTokens: 8_192,
            // Dense execution ignores expert residency, but the shared v1
            // runtime contract still requires a valid slot count.
            expertCacheSlots: 16,
            temperature: 1.0,
            topK: 64,
            topP: 0.95),
        capabilities: [.textGeneration, .reasoning],
        reasoningControl: .toggle)

    public static let gemma4_26B_A4B = TUFFModelDescriptor(
        id: .gemma4_26B_A4B,
        selector: "gemma4",
        apiModelID: "gemma-4-26b-a4b-it",
        displayName: "Gemma 4 26B-A4B IT 4-bit",
        shortName: "Gemma 4 26B",
        summary: "26B total, 3.9B active. The original TurboFieldfare target; "
            + "smallest install and lowest resident footprint.",
        family: .gemma4,
        architecture: .gemma4_26B_A4B,
        installDirectoryName: "gemma4.gturbo",
        source: gemmaSource,
        hardware: TUFFModelHardwareRequirements(minimumUnifiedMemoryBytes: eightGiB),
        memory: TUFFModelMemoryProfile(
            qualifiedDefaultWorkingSetBytes: 2_254_857_830,
            expertCacheBytesPerSlot: 100_663_296,
            kvCache: TUFFKVCacheProfile(
                fullAttentionBytesPerToken: 20_480,
                slidingAttentionBytesPerToken: 204_800,
                slidingWindowCapacityTokens: 1_304)),
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
        apiModelID: "qwen3.6-35b-a3b",
        displayName: "Qwen3.6 35B-A3B 4-bit",
        shortName: "Qwen3.6 35B",
        summary: "35B total, 3B active. Hybrid linear/full attention, so its "
            + "KV cache stays small at long context; larger install.",
        family: .qwen36,
        architecture: .qwen36_35B_A3B,
        installDirectoryName: "qwen36.gturbo",
        source: qwenSource,
        hardware: TUFFModelHardwareRequirements(minimumUnifiedMemoryBytes: eightGiB),
        memory: TUFFModelMemoryProfile(
            qualifiedDefaultWorkingSetBytes: 1_610_612_736,
            expertCacheBytesPerSlot: 75_497_472,
            kvCache: TUFFKVCacheProfile(fullAttentionBytesPerToken: 20_480)),
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

    /// Official Harmony checkpoint. The conservative 24 GB gate remains in
    /// place until the real 20B qualification commit records its measured
    /// peak working set and safe context limits.
    public static let gptOss_20B = TUFFModelDescriptor(
        id: .gptOss_20B,
        selector: "gpt-oss-20b",
        aliases: ["gpt-oss"],
        apiModelID: "gpt-oss-20b",
        displayName: "GPT-OSS 20B",
        shortName: "GPT-OSS 20B",
        summary: "Official MXFP4 checkpoint with Harmony reasoning and tool calls.",
        family: .gptOss,
        architecture: .gptOss_20B,
        installDirectoryName: "gpt-oss-20b.gturbo",
        source: gptOss20BSource,
        hardware: TUFFModelHardwareRequirements(
            minimumUnifiedMemoryBytes: 24 * oneGiB),
        memory: TUFFModelMemoryProfile(
            qualifiedDefaultWorkingSetBytes: 20 * oneGiB,
            defaultContextTokens: 4_096,
            defaultExpertCacheSlots: 4,
            expertCacheBytesPerSlot: 13_238_272,
            kvCache: TUFFKVCacheProfile(
                fullAttentionBytesPerToken: 24_576,
                slidingAttentionBytesPerToken: 24_576,
                slidingWindowCapacityTokens: 384)),
        runtimeDefaults: TUFFModelRuntimeDefaults(
            contextTokens: 4_096,
            expertCacheSlots: 4,
            temperature: 1.0,
            topK: 0,
            topP: 1.0),
        capabilities: [.textGeneration, .reasoning],
        reasoningControl: .graded)

    /// The pinned 120B checkpoint is installable with the same bounded-memory
    /// path as 20B. Its 96 GB gate is deliberately conservative until a real
    /// run on qualifying Apple Silicon records a safe measured working set.
    public static let gptOss_120B = TUFFModelDescriptor(
        id: .gptOss_120B,
        selector: "gpt-oss-120b",
        apiModelID: "gpt-oss-120b",
        displayName: "GPT-OSS 120B",
        shortName: "GPT-OSS 120B",
        summary: "Maximum-quality GPT-OSS checkpoint with streamed MXFP4 experts.",
        family: .gptOss,
        architecture: .gptOss_120B,
        installDirectoryName: "gpt-oss-120b.gturbo",
        source: gptOss120BSource,
        hardware: TUFFModelHardwareRequirements(
            minimumUnifiedMemoryBytes: 96 * oneGiB),
        memory: TUFFModelMemoryProfile(
            qualifiedDefaultWorkingSetBytes: 72 * oneGiB,
            defaultContextTokens: 4_096,
            defaultExpertCacheSlots: 4,
            expertCacheBytesPerSlot: 13_238_272,
            kvCache: TUFFKVCacheProfile(
                fullAttentionBytesPerToken: 36_864,
                slidingAttentionBytesPerToken: 36_864,
                slidingWindowCapacityTokens: 384)),
        runtimeDefaults: TUFFModelRuntimeDefaults(
            contextTokens: 4_096,
            expertCacheSlots: 4,
            temperature: 1.0,
            topK: 0,
            topP: 1.0),
        capabilities: [.textGeneration, .reasoning],
        reasoningControl: .graded)

    public static let all: [TUFFModelDescriptor] = [
        gemma4_E4B,
        gemma4_26B_A4B,
        qwen36_35B_A3B,
        gptOss_20B,
        gptOss_120B,
    ]
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
