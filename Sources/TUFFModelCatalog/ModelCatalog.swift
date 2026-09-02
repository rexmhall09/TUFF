import Foundation
import Darwin

public enum TUFFModelID: String, Codable, CaseIterable, Sendable {
    case gemma4_E2B = "gemma4-e2b"
    case gemma4_E4B = "gemma4-e4b"
    case gemma4_12B_QAT = "gemma4-12b-qat"
    case gemma4_26B_A4B = "gemma4-26b-a4b"
    case qwen36_35B_A3B = "qwen36-35b-a3b"
    case gptOss_20B = "gpt-oss-20b"
    case gptOss_120B = "gpt-oss-120b"
    case minimaxM27 = "minimax-m2.7"
}

public enum TUFFModelFamily: String, Codable, Sendable {
    case gemma4
    case qwen36
    case gptOss = "gpt-oss"
    case minimaxM2 = "minimax-m2"
}

public enum TUFFArchitectureID: String, Codable, Sendable {
    case gemma4_E2B = "gemma4-e2b"
    case gemma4_E4B = "gemma4-e4b"
    case gemma4_12B_QAT = "gemma4-12b-qat"
    case gemma4_26B_A4B = "gemma4-26b-a4b"
    case qwen36_35B_A3B = "qwen36-35b-a3b"
    case gptOss_20B = "gpt-oss-20b"
    case gptOss_120B = "gpt-oss-120b"
    case minimaxM27 = "minimax-m2.7"
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
    static let gemma4E2B = TUFFArchitectureProfile(
        id: .gemma4_E2B,
        family: .gemma4,
        feedForwardKind: .dense,
        weightLayout: .affine)

    static let gemma4E4B = TUFFArchitectureProfile(
        id: .gemma4_E4B,
        family: .gemma4,
        feedForwardKind: .dense,
        weightLayout: .affine)

    static let gemma4_12B_QAT = TUFFArchitectureProfile(
        id: .gemma4_12B_QAT,
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

    static let minimaxM27 = TUFFArchitectureProfile(
        id: .minimaxM27,
        family: .minimaxM2,
        feedForwardKind: .mixtureOfExperts,
        weightLayout: .affine)
}

public enum TUFFReasoningControl: String, Codable, Sendable {
    case toggle
    case toggleWithPreservation
    case graded
    case alwaysOn
}

public enum TUFFModelQualification: String, Codable, Sendable {
    case qualified
    case requiresRealModelValidation
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
    /// Free, inactive, and speculative pages reclaimable when TUFF launched.
    /// Nil for synthetic/test capability descriptions and older archives.
    ///
    /// Recorded as a diagnostic only. It used to cap `safeAppMemoryBudgetBytes`,
    /// which made the budget depend on whatever else was open at launch and
    /// refused models a Mac was qualified for; see that property.
    public let availableMemoryBytes: UInt64?
    public let macOSMajorVersion: Int
    public let appleSiliconGeneration: Int

    public init(unifiedMemoryBytes: UInt64,
                availableMemoryBytes: UInt64? = nil,
                macOSMajorVersion: Int,
                appleSiliconGeneration: Int) {
        self.unifiedMemoryBytes = unifiedMemoryBytes
        self.availableMemoryBytes = availableMemoryBytes
        self.macOSMajorVersion = macOSMajorVersion
        self.appleSiliconGeneration = appleSiliconGeneration
    }

    public static func current() -> TUFFDeviceCapabilities {
        let memory = sysctlUInt64(named: "hw.memsize") ?? ProcessInfo.processInfo.physicalMemory
        let generation = sysctlString(named: "machdep.cpu.brand_string")
            .flatMap(appleSiliconGeneration(brandString:)) ?? 1
        return TUFFDeviceCapabilities(
            unifiedMemoryBytes: memory,
            availableMemoryBytes: hostAvailableMemoryBytes(),
            macOSMajorVersion: ProcessInfo.processInfo.operatingSystemVersion.majorVersion,
            appleSiliconGeneration: generation)
    }

    /// Share of installed unified memory TUFF plans against.
    public static let safeAppMemoryShareNumerator: UInt64 = 3
    public static let safeAppMemoryShareDenominator: UInt64 = 4

    /// Memory TUFF can use while retaining a system reserve: a fixed 75% of
    /// installed unified memory.
    ///
    /// This was previously capped by how much memory happened to be
    /// reclaimable when the app launched. That made the budget depend on
    /// whatever else was open at the time, and on a 16 GB Mac it routinely
    /// refused the very models that were qualified on 16 GB — the machine
    /// could host them, but the snapshot said otherwise. Weights are
    /// memory-mapped and their pages are evictable, so the kernel can reclaim
    /// under pressure; a fixed share of installed memory describes what the
    /// machine is, which is the thing a model requirement is written against.
    public var safeAppMemoryBudgetBytes: UInt64 {
        unifiedMemoryBytes / Self.safeAppMemoryShareDenominator
            * Self.safeAppMemoryShareNumerator
    }

    private static func hostAvailableMemoryBytes() -> UInt64? {
        var pageSize: vm_size_t = 0
        guard host_page_size(mach_host_self(), &pageSize) == KERN_SUCCESS else {
            return nil
        }
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(
            MemoryLayout<vm_statistics64_data_t>.size
                / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &stats) { pointer in
            pointer.withMemoryRebound(
                to: integer_t.self,
                capacity: Int(count)
            ) { rebound in
                host_statistics64(
                    mach_host_self(),
                    HOST_VM_INFO64,
                    rebound,
                    &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        let pages = UInt64(stats.free_count)
            + UInt64(stats.inactive_count)
            + UInt64(stats.speculative_count)
        let bytes = pages.multipliedReportingOverflow(by: UInt64(pageSize))
        return bytes.overflow ? nil : bytes.partialValue
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
    case requiresRealModelValidation
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
    public let qualification: TUFFModelQualification
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
                qualification: TUFFModelQualification = .qualified,
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
        self.qualification = qualification
        self.runtimeDefaults = runtimeDefaults
        self.capabilities = capabilities
        self.reasoningControl = reasoningControl
        self.addons = addons
    }

    public var defaultSystemPrompt: String {
        "You are \(shortName), a helpful AI assistant. "
            + "You are running in a SSD MoE streaming app on Mac called TUFF."
    }

    public func compatibility(
        with device: TUFFDeviceCapabilities,
        contextTokens: Int? = nil,
        expertCacheSlots: Int? = nil
    ) -> TUFFModelCompatibility {
        var issues: [TUFFCompatibilityIssue] = []
        if qualification == .requiresRealModelValidation {
            issues.append(.requiresRealModelValidation)
        }
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
            let budget = device.safeAppMemoryBudgetBytes
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

    /// Pins verified against the live repository: `revision` is the commit the
    /// Hugging Face API reports for `main`, and `sourceIndexSHA256` is the
    /// SHA-256 of `model.safetensors.index.json` at that commit.
    ///
    /// The byte figures are the tensors this installer actually streams — the
    /// `language_model.*` ranges plus the tokenizer — not the whole file. The
    /// checkpoint also carries a vision tower and an audio tower that the text
    /// install skips, which is why the download is smaller than the repository.
    ///
    /// The method was checked by recomputing E4B's pinned fingerprint from its
    /// own repository: it reproduced `f8accac5…0093` exactly, and the byte sum
    /// landed within 400 KB of the figure pinned below.
    private static let gemmaE2BSource = TUFFModelSource(
        repoID: "mlx-community/gemma-4-e2b-it-4bit",
        revision: "238767527555cb75a05732a84dff5d6ba0dd6809",
        sourceIndexSHA256:
            "edb157dbf495e23f37377af4a628a9ad13c4ee7937f93ccb36ec9e9a19940f16",
        manifestModelID: "mlx-community/gemma-4-e2b-it-4bit",
        approximateDownloadBytes: 2_636_500_000,
        // Measured from a completed install: 2,636,263,860 bytes on disk. The
        // previous figure rounded *down* past that, so the free-space gate
        // asked for less than the install actually needs.
        installedBytes: 2_636_400_000,
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

    private static let gemma12BQATSource = TUFFModelSource(
        repoID: "mlx-community/gemma-4-12B-it-qat-4bit",
        revision: "e70c6b3ba0979b3357dcd2f223ad8bde7787a6b6",
        sourceIndexSHA256:
            "b87c93774de5d13ca9d0e21b045793e42e5df032fb5e7622212524f56f9695f2",
        manifestModelID: "mlx-community/gemma-4-12B-it-qat-4bit",
        approximateDownloadBytes: 10_947_021_920,
        installedBytes: 10_978_056_429,
        reserveBytes: oneGiB)

    private static let qwenSource = TUFFModelSource(
        repoID: "mlx-community/Qwen3.6-35B-A3B-4bit",
        revision: "38740b847e4cb78f352aba30aa41c76e08e6eb46",
        sourceIndexSHA256:
            "0b28df60e33753a14e816d3b31577ae2c93884c58430a4a6de6ae9ea483842ea",
        manifestModelID: "qwen3.6-35b-a3b-4bit",
        // The planner's own totalSourceBytes for this exact revision. The
        // previous figure understated the transfer by 617 MB.
        approximateDownloadBytes: 20_146_183_200,
        // Measured from a completed install: 19,551,402,438 bytes on disk,
        // which the previous planned figure understated by 4.9 MB.
        installedBytes: 19_551_500_000,
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

    private static let minimaxM27Source = TUFFModelSource(
        repoID: "mlx-community/MiniMax-M2.7-4bit",
        revision: "66d2e5cb7c5cda05251b4625c504af4b034df7ff",
        sourceIndexSHA256:
            "8b2204b5a4741cb323a49d3ad6cfc5523c72c7c8ffa6e668a5627fb36ee13e52",
        manifestModelID: "mlx-community/MiniMax-M2.7-4bit",
        approximateDownloadBytes: 128_700_000_000,
        installedBytes: 128_700_000_000,
        reserveBytes: oneGiB)

    /// Small text-only launch model. Image and audio remain intentionally
    /// absent until separately packaged add-ons pass their own qualification.
    public static let gemma4_E2B = TUFFModelDescriptor(
        id: .gemma4_E2B,
        selector: "gemma4-e2b",
        aliases: ["e2b"],
        apiModelID: "gemma-4-e2b-it",
        displayName: "Gemma 4 E2B IT 4-bit",
        shortName: "Gemma 4 E2B",
        summary: "The smallest Gemma. Same architecture as E4B — per-layer embeddings and shared KV projections — at roughly half the weights.",
        family: .gemma4,
        architecture: .gemma4E2B,
        installDirectoryName: "gemma4-e2b.gturbo",
        source: gemmaE2BSource,
        hardware: TUFFModelHardwareRequirements(minimumUnifiedMemoryBytes: eightGiB),
        // Retain E4B's measured profile as a conservative compatibility
        // guardrail. E2B completed a pinned install and an 8K-context runtime
        // generation on the audit machine, but that single short run is not a
        // replacement for the cross-device qualification sweep that produced
        // the catalogue figures. E2B has fewer layers, KV heads and weights,
        // although its final 20 MLPs are wider than E4B's, so the old claim
        // that it was smaller in every dimension was not true.
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
            expertCacheSlots: 16,
            temperature: 1.0,
            topK: 64,
            topP: 0.95),
        capabilities: [.textGeneration, .imageInput, .reasoning],
        reasoningControl: .toggle,
        addons: [TUFFModelAddonDescriptor(
            id: "gemma4-e2b-image-input",
            displayName: "Gemma 4 E2B Image Support",
            kind: .imageInput,
            source: TUFFModelSource(
                repoID: gemmaE2BSource.repoID,
                revision: gemmaE2BSource.revision,
                sourceIndexSHA256: gemmaE2BSource.sourceIndexSHA256,
                manifestModelID: gemmaE2BSource.manifestModelID,
                approximateDownloadBytes: 1_169_504_854,
                installedBytes: 336_950_725,
                reserveBytes: oneGiB),
            hardware: TUFFModelHardwareRequirements(
                minimumUnifiedMemoryBytes: eightGiB,
                minimumAppleSiliconGeneration: 2))])

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
        capabilities: [.textGeneration, .imageInput, .reasoning],
        reasoningControl: .toggle,
        addons: [TUFFModelAddonDescriptor(
            id: "gemma4-e4b-image-input",
            displayName: "Gemma 4 E4B Image Support",
            kind: .imageInput,
            source: TUFFModelSource(
                repoID: gemmaE4BSource.repoID,
                revision: gemmaE4BSource.revision,
                sourceIndexSHA256: gemmaE4BSource.sourceIndexSHA256,
                manifestModelID: gemmaE4BSource.manifestModelID,
                // Coalesced range total for E4B's own layout, not E2B's plus
                // the payload delta: the two checkpoints interleave their
                // vision tensors differently, so E4B needs 31 requests where
                // E2B needs 23 and pulls 143 MB more of the gaps between them.
                approximateDownloadBytes: 1_313_596_274,
                installedBytes: 337_376_704,
                reserveBytes: oneGiB),
            hardware: TUFFModelHardwareRequirements(
                minimumUnifiedMemoryBytes: eightGiB,
                minimumAppleSiliconGeneration: 2))])

    public static let gemma4_12B_QAT = TUFFModelDescriptor(
        id: .gemma4_12B_QAT,
        selector: "gemma4-12b-qat",
        aliases: ["12b", "12b-qat"],
        apiModelID: "gemma-4-12b-it-qat",
        displayName: "Gemma 4 12B IT QAT 4-bit",
        shortName: "Gemma 4 12B QAT",
        summary: "Dense Gemma 4 Unified with 8-bit QAT MLPs and optional image input.",
        family: .gemma4,
        architecture: .gemma4_12B_QAT,
        installDirectoryName: "gemma4-12b-qat.gturbo",
        source: gemma12BQATSource,
        hardware: TUFFModelHardwareRequirements(
            minimumUnifiedMemoryBytes: 16 * oneGiB),
        memory: TUFFModelMemoryProfile(
            qualifiedDefaultWorkingSetBytes: 6_000_000_000,
            defaultExpertCacheSlots: 0,
            expertCacheBytesPerSlot: 0,
            kvCache: TUFFKVCacheProfile(
                fullAttentionBytesPerToken: 16_384,
                slidingAttentionBytesPerToken: 327_680,
                slidingWindowCapacityTokens: 1_024)),
        // Qualified with a complete pinned install plus real text and image
        // generation on a 16 GB Apple-silicon Mac. Keep the 6 GB catalog
        // estimate conservative: file-backed Metal mappings are not fully
        // represented by the process RSS reported by `time`.
        runtimeDefaults: TUFFModelRuntimeDefaults(
            contextTokens: 8_192,
            expertCacheSlots: 16,
            temperature: 1.0,
            topK: 64,
            topP: 0.95),
        capabilities: [.textGeneration, .imageInput, .reasoning],
        reasoningControl: .toggle,
        addons: [TUFFModelAddonDescriptor(
            id: "gemma4-12b-qat-image-input",
            displayName: "Gemma 4 12B QAT Image Support",
            kind: .imageInput,
            source: TUFFModelSource(
                repoID: gemma12BQATSource.repoID,
                revision: gemma12BQATSource.revision,
                sourceIndexSHA256: gemma12BQATSource.sourceIndexSHA256,
                manifestModelID: gemma12BQATSource.manifestModelID,
                approximateDownloadBytes: 102_556_672,
                installedBytes: 40_581_222,
                reserveBytes: oneGiB),
            hardware: TUFFModelHardwareRequirements(
                minimumUnifiedMemoryBytes: 16 * oneGiB,
                minimumAppleSiliconGeneration: 2))])

    public static let gemma4_26B_A4B = TUFFModelDescriptor(
        id: .gemma4_26B_A4B,
        selector: "gemma4",
        apiModelID: "gemma-4-26b-a4b-it",
        displayName: "Gemma 4 26B-A4B IT 4-bit",
        shortName: "Gemma 4 26B",
        summary: "26B total, 3.9B active. A balanced Gemma model with optional image input.",
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
                // The coalesced range total, not the vision payload sum. Qwen
                // interleaves its 333 vision tensors through shard 1 across
                // 129 gaps, so fetching them costs 245 MB more than the
                // tensors themselves weigh.
                approximateDownloadBytes: 1_137_999_008,
                installedBytes: 900_808_704,
                reserveBytes: oneGiB),
            hardware: TUFFModelHardwareRequirements(
                minimumUnifiedMemoryBytes: eightGiB,
                minimumAppleSiliconGeneration: 2))])

    /// Official Harmony checkpoint, qualified at 4K context on a 16 GB M2.
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
            minimumUnifiedMemoryBytes: 16 * oneGiB),
        memory: TUFFModelMemoryProfile(
            qualifiedDefaultWorkingSetBytes: 5_487_695_296,
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

    /// Official Harmony checkpoint, qualified at 4K context on a 16 GB M2.
    /// Streamed experts keep the full 61 GiB installation out of the working
    /// set while an FP32 residual stream prevents late-layer overflow.
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
            minimumUnifiedMemoryBytes: 16 * oneGiB),
        memory: TUFFModelMemoryProfile(
            qualifiedDefaultWorkingSetBytes: 7_990_582_952,
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

    /// MiniMax M2.7 streamed from the pinned MLX 4-bit checkpoint. Its 229B
    /// total parameters remain file-backed; only the selected eight experts
    /// per layer enter the bounded cache.
    public static let minimaxM27 = TUFFModelDescriptor(
        id: .minimaxM27,
        selector: "minimax-m2.7",
        aliases: ["minimax", "m2.7"],
        apiModelID: "minimax-m2.7",
        displayName: "MiniMax M2.7 4-bit",
        shortName: "MiniMax M2.7",
        summary: "229B total with eight active experts, streamed for 16 GB Macs.",
        family: .minimaxM2,
        architecture: .minimaxM27,
        installDirectoryName: "minimax-m2.7.gturbo",
        source: minimaxM27Source,
        hardware: TUFFModelHardwareRequirements(
            minimumUnifiedMemoryBytes: 16 * oneGiB,
            minimumAppleSiliconGeneration: 2),
        memory: TUFFModelMemoryProfile(
            qualifiedDefaultWorkingSetBytes: 11_250_000_000,
            defaultContextTokens: 4_096,
            defaultExpertCacheSlots: 16,
            expertCacheBytesPerSlot: 493_682_688,
            kvCache: TUFFKVCacheProfile(fullAttentionBytesPerToken: 253_952)),
        runtimeDefaults: TUFFModelRuntimeDefaults(
            contextTokens: 4_096,
            expertCacheSlots: 16,
            temperature: 1.0,
            topK: 40,
            topP: 0.95),
        capabilities: [.textGeneration, .reasoning],
        reasoningControl: .alwaysOn)

    public static let all: [TUFFModelDescriptor] = [
        gemma4_E2B,
        gemma4_E4B,
        gemma4_12B_QAT,
        gemma4_26B_A4B,
        qwen36_35B_A3B,
        gptOss_20B,
        gptOss_120B,
        minimaxM27,
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
