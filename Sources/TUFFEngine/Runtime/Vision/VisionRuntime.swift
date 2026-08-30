import Darwin
import Foundation
import Metal
import TUFFFormat

public enum VisionRuntimeError: Error, CustomStringConvertible {
    case disabled
    case invalidInput(String)
    case commandFailed(String)
    case unsupportedKernel(String)

    public var description: String {
        switch self {
        case .disabled: "vision runtime is unavailable"
        case .invalidInput(let detail): "invalid vision input: \(detail)"
        case .commandFailed(let detail): "vision Metal command failed: \(detail)"
        case .unsupportedKernel(let detail): "vision kernel unavailable: \(detail)"
        }
    }
}

public struct VisionFeatures: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let tokenCount: Int
    public let hiddenSize: Int
    public let family: ModelFamily
    public let patchGridWidth: Int
    public let patchGridHeight: Int
    public let gpuNanoseconds: UInt64
    public let scratchBytes: Int
    public let attentionVariant: VisionAttention.Variant
    public let projectorPath: MPPPrefillInt4QMM.Path
    public let expertResidencyTransition: VisionExpertResidencyTransition?
    public let preprocessing: VisionPreprocessingReport?
    public var wall: VisionWallBreakdown = .unmeasured

    public init(
        buffer: MTLBuffer,
        tokenCount: Int,
        hiddenSize: Int,
        family: ModelFamily = .gemma4,
        patchGridWidth: Int = 0,
        patchGridHeight: Int = 0,
        gpuNanoseconds: UInt64,
        scratchBytes: Int,
        attentionVariant: VisionAttention.Variant,
        projectorPath: MPPPrefillInt4QMM.Path,
        expertResidencyTransition: VisionExpertResidencyTransition?,
        preprocessing: VisionPreprocessingReport?,
        wall: VisionWallBreakdown = .unmeasured
    ) {
        self.buffer = buffer
        self.tokenCount = tokenCount
        self.hiddenSize = hiddenSize
        self.family = family
        self.patchGridWidth = patchGridWidth
        self.patchGridHeight = patchGridHeight
        self.gpuNanoseconds = gpuNanoseconds
        self.scratchBytes = scratchBytes
        self.attentionVariant = attentionVariant
        self.projectorPath = projectorPath
        self.expertResidencyTransition = expertResidencyTransition
        self.preprocessing = preprocessing
        self.wall = wall
    }
}

/// Splits tower wall time into the buckets that decide whether GPU-time work or
/// pipelining work is worth doing: scratch allocation, weight mapping and page
/// faults, blocking GPU waits, and the CPU encode remainder.
public struct VisionWallBreakdown: Sendable, Equatable {
    public static let unmeasured = VisionWallBreakdown(
        scratchAllocationNanoseconds: 0, weightMapNanoseconds: 0,
        gpuWaitNanoseconds: 0, totalNanoseconds: 0)

    public let scratchAllocationNanoseconds: UInt64
    public let weightMapNanoseconds: UInt64
    public let gpuWaitNanoseconds: UInt64
    public let totalNanoseconds: UInt64

    public var encodeNanoseconds: UInt64 {
        totalNanoseconds
            - min(totalNanoseconds,
                  scratchAllocationNanoseconds + weightMapNanoseconds
                      + gpuWaitNanoseconds)
    }
}

public struct VisionPreprocessingReport: Sendable, Equatable {
    public let metadata: VisionImageMetadata
    public let geometry: Gemma4ImageGeometry
    public let wallNanoseconds: UInt64
    public let allocatedBytes: Int
}

enum VisionStageClass {
    case patchPosition
    case attention
    case linear
    case elementwise
    case poolProjector
}

/// Diagnostic-only attribution produced under TUFF_VISION_STAGE_PROFILE=1,
/// which splits stage classes into separate command buffers. Its per-buffer sync
/// overhead makes the totals incomparable to the promotion gpu_ms.
public struct VisionStageProfile: Sendable, Equatable {
    public let patchPositionNanoseconds: UInt64
    public let attentionNanoseconds: UInt64
    public let attentionPadNanoseconds: UInt64
    public let attentionCoreNanoseconds: UInt64
    public let attentionUnpadNanoseconds: UInt64
    public let linearNanoseconds: UInt64
    public let elementwiseNanoseconds: UInt64
    public let poolProjectorNanoseconds: UInt64
    public let commandBufferCount: Int
    public let layerCount: Int

    public var totalNanoseconds: UInt64 {
        patchPositionNanoseconds + attentionNanoseconds + linearNanoseconds
            + elementwiseNanoseconds + poolProjectorNanoseconds
    }
}

public final class VisionRuntime {
    public let config: VisionConfig

    private let context: MetalContext
    private let store: VisionWeightStore
    private let useLease: VisionPackUseLease
    private let linear: VisionLinearBF16
    private let primitives: VisionPrimitives
    private let attention: VisionAttention
    private let usesBF16Projector: Bool
    private let splitsStagesForProfile: Bool
    private let fusesGeGLU: Bool
    private let unifiedPatchProjectorKernel: MPPPrefillInt4QMM
    public private(set) var lastStageProfile: VisionStageProfile?
    /// Mapped weight regions retained across images under `.keepReady`. Mapping
    /// and first-touching the 1.08 GB the tower reads costs the same every image
    /// otherwise; holding the mappings pays it once. Pages are clean, file-backed
    /// and evictable, like the routed expert pool.
    private var cachedRegions: [String: VisionMappedWeightRegion] = [:]

    /// Drops retained mappings. Called when residency returns to on-demand.
    public func releaseCachedWeightRegions() {
        cachedRegions.removeAll(keepingCapacity: false)
    }

    /// Maps every tower region up front and retains them, as `.keepReady` does
    /// lazily on its first encode.
    ///
    /// Without this, choosing Keep Ready changed nothing until an image was
    /// sent: the first image still paid the mapping cost the setting exists to
    /// avoid, and any figure reporting held bytes read zero until then, which
    /// makes the setting look inert. Pages are clean and file-backed, so this
    /// costs mappings and page-cache residency, not process footprint.
    @discardableResult
    public func prewarmWeightRegions() throws -> Int {
        if config.architecture == .gemma4Unified12B {
            for names in [Self.unifiedPatchTensorNames, Self.unifiedProjectorTensorNames] {
                let key = names.first ?? ""
                guard cachedRegions[key] == nil else { continue }
                cachedRegions[key] = try store.mapRegion(
                    tensorNames: names, device: context.device)
            }
            return retainedWeightBytes
        }
        for layer in 0..<config.numLayers {
            let prefix = config.family == .gemma4
                ? "vision_tower.encoder.layers.\(layer)."
                : "vision_tower.blocks.\(layer)."
            let names = store.tensorNames(withPrefix: prefix)
            guard !names.isEmpty else { continue }
            let key = names.first ?? ""
            if cachedRegions[key] == nil {
                cachedRegions[key] = try store.mapRegion(
                    tensorNames: names, device: context.device)
            }
        }
        // The same groups, in the same order, that `encodePreparedPatches` maps —
        // the cache is keyed on a group's first tensor name, so prewarming a
        // different grouping fills it with entries no encode ever looks up. The
        // first version walked `vision_tower.embeddings.`, which matches nothing
        // in this pack, and an `embed_vision.` group whose key differed from the
        // projector's: the patch embedder was never prewarmed and the projector
        // was mapped a second time and then held for the life of the runtime.
        let gemmaProjector = config.usesESeriesVisionTower
            ? Array(Self.projectorTensorNames.suffix(3))
            : Self.projectorTensorNames
        let fixedGroups = config.family == .gemma4
            ? [Self.patchEmbedderTensorNames, gemmaProjector]
            : [Self.qwenPatchTensorNames, Self.qwenMergerTensorNames]
        for names in fixedGroups {
            let key = names.first ?? ""
            guard cachedRegions[key] == nil else { continue }
            cachedRegions[key] = try store.mapRegion(
                tensorNames: names, device: context.device)
        }
        return retainedWeightBytes
    }

    /// Bytes of tower weights this runtime is holding mapped.
    ///
    /// They are clean and file-backed, so they never appear in the process
    /// footprint however many are held — only in resident bytes, and only once
    /// touched. Reporting them separately is the only way to answer "what is
    /// keep-ready costing me", which neither figure can.
    public var retainedWeightBytes: Int {
        cachedRegions.values.reduce(0) { $0 + $1.buffer.length }
    }

    public static func open(
        textModelURL: URL,
        context: MetalContext,
        visionPackURL: URL? = nil,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) throws -> VisionRuntime {
        try requireSupportedDevice(context.device)
        let baseline = try ManifestReader.resolveArchitecture(
            directoryURL: textModelURL)
        let textFamily = baseline.family
        let manifest = try ManifestReader.load(
            directoryURL: textModelURL,
            expecting: baseline)
        guard let source = manifest.sourceSnapshotHash else {
            throw VisionRuntimeError.invalidInput("text model has no source identity")
        }
        let textManifestSHA = try Sha256Verifier.hashFile(
            at: textModelURL.appendingPathComponent("manifest.json"))
        let companion = try visionPackURL
            ?? VisionPackLocation.companionURL(forTextModel: textModelURL)
        // The pack has to exist before anything is created beside it. The lease
        // opens its lock with O_CREAT, so taking it first meant a text-only
        // install got a permanent stray dotfile, and a pack on a read-only
        // volume failed with "could not open use lock ... errno 30" — a lock
        // error standing in for a perfectly valid pack.
        guard FileManager.default.fileExists(atPath: companion.path) else {
            // Absence is not invalid metadata. Saying "metadata is invalid" for
            // a pack nobody installed sends the reader looking for corruption
            // in a file that is not there.
            throw VisionPackError.packNotFound(companion.path)
        }
        // ... and it has to look like a pack, for the same reason the existence
        // check above exists. A directory that is merely *there* — a `.partial`
        // download, a mistyped path — also got a permanent stray lock file
        // beside it, because the lease is created before the store validates
        // anything. One `lstat` moves that decision ahead of the `O_CREAT`.
        let manifestPath = companion
            .appendingPathComponent(GTurboVisionFormatV1.manifestFile).path
        guard FileManager.default.fileExists(atPath: manifestPath) else {
            throw VisionPackError.invalidMetadata(
                "no \(GTurboVisionFormatV1.manifestFile) in \(companion.path)")
        }
        let useLease = try VisionPackUseLease.acquireShared(companionURL: companion)
        let store = try VisionWeightStore.open(
            directoryURL: companion,
            expectedFamily: textFamily,
            compatibleTextSourceSnapshotHash: source,
            compatibleTextManifestSha256: textManifestSHA)
        let runtimeEnvironment = resolvedKernelEnvironment(
            environment, family: textFamily)
        return try VisionRuntime(
            context: context,
            store: store,
            useLease: useLease,
            family: textFamily,
            modelVariant: baseline.variant,
            environment: runtimeEnvironment)
    }

    static func resolvedKernelEnvironment(
        _ environment: [String: String],
        family: ModelFamily = .gemma4
    ) -> [String: String] {
        var runtimeEnvironment = environment
        if environment["TUFF_VISION_BASELINE_KERNELS"] != "1" {
            let explicitAttention = [
                "TUFF_VISION_ATTENTION_Q8",
                "TUFF_VISION_ATTENTION_Q16",
                "TUFF_VISION_ATTENTION_MPP",
                "TUFF_VISION_ATTENTION_PAD80",
                "TUFF_VISION_MLX_METALLIB",
            ].contains { environment[$0] != nil }
            if !explicitAttention {
                runtimeEnvironment[
                    family == .gemma4
                        ? "TUFF_VISION_ATTENTION_MPP"
                        : "TUFF_VISION_ATTENTION_Q16"] = "1"
            }
            let explicitLinear = [
                "TUFF_VISION_REGISTER_GEMM",
                "TUFF_VISION_REGISTER_ATTENTION",
                "TUFF_VISION_REGISTER_MLP",
                "TUFF_VISION_MLX_GEMM_METALLIB",
            ].contains { environment[$0] != nil }
            if !explicitLinear {
                runtimeEnvironment["TUFF_VISION_REGISTER_GEMM"] = "1"
            }
        }
        return runtimeEnvironment
    }

    public static func isSupported(on device: MTLDevice) -> Bool {
        device.supportsFamily(.apple8)
    }

    public static var isSupportedOnDefaultDevice: Bool {
        guard let device = MTLCreateSystemDefaultDevice() else { return false }
        return isSupported(on: device)
    }

    public static func requireSupportedDevice(_ device: MTLDevice) throws {
        try requireSupportedDevice(supportsApple8: isSupported(on: device))
    }

    public static func requireSupportedDevice(supportsApple8: Bool) throws {
        guard supportsApple8 else {
            throw VisionRuntimeError.unsupportedKernel(
                "the image tower requires an M2 or newer Apple Silicon Mac")
        }
    }

    init(context: MetalContext, store: VisionWeightStore,
                 useLease: VisionPackUseLease,
                 family: ModelFamily = .gemma4,
                 modelVariant: ModelVariant? = nil,
                 environment: [String: String]) throws {
        self.context = context
        self.store = store
        self.useLease = useLease
        self.config = VisionConfig(family: family, modelVariant: modelVariant)
        let shape = config
        let linear = try VisionLinearBF16(context: context, environment: environment)
        self.linear = linear
        self.primitives = try VisionPrimitives(
            context: context,
            headDimension: shape.headDimension > 0 ? shape.headDimension : 72,
            environment: environment)
        // The fused-padded-layout decision has one owner: the attention object,
        // told here whether the linear path can write the padded head-major
        // store. Re-deriving the AND at encode time let the two sides disagree
        // and run a different kernel than the one diagnostics reported.
        var attentionEnvironment = environment
        if shape.architecture == .gemma4Unified12B {
            attentionEnvironment["TUFF_VISION_ATTENTION_Q8"] = "1"
            attentionEnvironment.removeValue(forKey: "TUFF_VISION_ATTENTION_MPP")
        }
        self.attention = try VisionAttention(
            context: context, environment: attentionEnvironment,
            scoreScale: max(shape.attentionScale, 1),
            headDimension: max(shape.headDimension, 64),
            allowFusedPaddedLayout: linear.supportsPaddedHeadStore(
                n: shape.hiddenSize, k: shape.hiddenSize)
                && shape.headDimension == VisionAttention.headDimension)
        self.usesBF16Projector = context.device.supportsFamily(.apple10)
            && environment["TUFF_VISION_FP16_PROJECTOR"] != "1"
        self.splitsStagesForProfile =
            environment["TUFF_VISION_STAGE_PROFILE"] == "1"
        self.fusesGeGLU =
            environment["TUFF_VISION_SPLIT_GEGLU"] != "1"
        // The reduction runs on the GPU unless the environment says otherwise.
        // Both paths apply the same weights, so this is where the arithmetic
        // happens, not which arithmetic it is; the flag exists to fall back if
        // a device ever disagrees with the CPU oracle.
        self.imageResize = environment["TUFF_VISION_CPU_RESIZE"] == "1"
            ? nil
            : try? VisionResize(context: context)
        // Built once per runtime; the tensorops library behind it is shared
        // with the linear and attention kernels through
        // `MetalContext.privateLibrary`'s cache instead of compiled per
        // constructor.
        self.projectorKernel = MPPPrefillInt4QMM(
            context: context,
            variant: usesBF16Projector
                ? .apple10BF16
                : context.device.supportsFamily(.apple10) ? .apple10V1 : .control)
        self.unifiedPatchProjectorKernel = MPPPrefillInt4QMM(
            context: context, variant: .apple10BF16Output)
    }

    private let projectorKernel: MPPPrefillInt4QMM
    private let imageResize: VisionResize?

    /// The tensor groups the encode path maps, named once so prewarm cannot
    /// map a different grouping into the same cache.
    static let patchEmbedderTensorNames = [
        "vision_tower.patch_embedder.input_proj.weight",
        "vision_tower.patch_embedder.position_embedding_table",
    ]
    static let projectorTensorNames = [
        "vision_tower.std_bias",
        "vision_tower.std_scale",
        "embed_vision.embedding_projection.weight",
        "embed_vision.embedding_projection.scales",
        "embed_vision.embedding_projection.biases",
    ]
    static let qwenPatchTensorNames = [
        "vision_tower.patch_embed.proj.weight",
        "vision_tower.patch_embed.proj.bias",
        "vision_tower.pos_embed.weight",
    ]
    static let qwenMergerTensorNames = [
        "vision_tower.merger.norm.weight",
        "vision_tower.merger.norm.bias",
        "vision_tower.merger.linear_fc1.weight",
        "vision_tower.merger.linear_fc1.bias",
        "vision_tower.merger.linear_fc2.weight",
        "vision_tower.merger.linear_fc2.bias",
    ]
    static let unifiedPatchTensorNames = [
        "vision_embedder.patch_ln1.weight",
        "vision_embedder.patch_ln1.bias",
        "vision_embedder.patch_dense.weight",
        "vision_embedder.patch_dense.scales",
        "vision_embedder.patch_dense.biases",
        "vision_embedder.patch_dense.bias",
        "vision_embedder.patch_ln2.weight",
        "vision_embedder.patch_ln2.bias",
        "vision_embedder.pos_embedding",
        "vision_embedder.pos_norm.weight",
        "vision_embedder.pos_norm.bias",
    ]
    static let unifiedProjectorTensorNames = [
        "embed_vision.embedding_projection.weight",
        "embed_vision.embedding_projection.scales",
        "embed_vision.embedding_projection.biases",
    ]

    public func preprocessImage(at fileURL: URL) throws -> VisionPixelBuffer {
        try Gemma4ImagePreprocessor(device: context.device, config: config,
                                    gpuResize: imageResize)
            .preprocess(fileURL: fileURL)
    }

    public func encodeImage(
        at fileURL: URL,
        languageModel: Model? = nil,
        residencyPolicy: VisionResidencyPolicy = .defaultPolicy,
        checkCancellation: () throws -> Void = {}
    ) throws -> VisionFeatures {
        let preprocessor = Gemma4ImagePreprocessor(device: context.device, config: config,
                                                   gpuResize: imageResize)
        return try encodeImage(
            plan: try preprocessor.plan(fileURL: fileURL),
            languageModel: languageModel,
            residencyPolicy: residencyPolicy,
            checkCancellation: checkCancellation)
    }

    /// Encodes an image the caller has already planned.
    ///
    /// Callers plan first to learn the token count, and re-planning here opened
    /// the file a second time for another ImageIO sniff, metadata parse and
    /// marker walk — per image, per turn. It also validated the encode against
    /// metadata that a file swapped between the two opens would no longer match.
    public func encodeImage(
        plan: VisionImagePlan,
        languageModel: Model? = nil,
        residencyPolicy: VisionResidencyPolicy = .defaultPolicy,
        checkCancellation: () throws -> Void = {}
    ) throws -> VisionFeatures {
        try checkCancellation()
        let preprocessor = Gemma4ImagePreprocessor(device: context.device, config: config,
                                                   gpuResize: imageResize)
        try checkCancellation()
        let gpuDrainNanoseconds = try languageModel != nil && residencyPolicy == .onDemand
            ? drainGPU() : 0
        try checkCancellation()
        let transition = languageModel?.prepareExpertResidencyForVision(
            residencyPolicy, gpuDrainNanoseconds: gpuDrainNanoseconds)
        try checkCancellation()
        let input = try preprocessor.preprocess(plan)
        return try encodePreparedPatches(
            patchesBF16: input.patchesBF16,
            positionsInt32x2: input.positionsInt32x2,
            patchGridWidth: input.geometry.patchGridWidth,
            patchGridHeight: input.geometry.patchGridHeight,
            expertResidencyTransition: transition,
            preprocessing: VisionPreprocessingReport(
                metadata: input.metadata,
                geometry: input.geometry,
                wallNanoseconds: input.wallNanoseconds,
                allocatedBytes: input.allocatedBytes),
            retainsWeightRegions: residencyPolicy == .keepReady,
            checkCancellation: checkCancellation)
    }

    public func encodePatches(
        patchesBF16: MTLBuffer,
        positionsInt32x2: MTLBuffer,
        patchGridWidth: Int,
        patchGridHeight: Int,
        languageModel: Model? = nil,
        residencyPolicy: VisionResidencyPolicy = .defaultPolicy,
        checkCancellation: () throws -> Void = {}
    ) throws -> VisionFeatures {
        try checkCancellation()
        let rows = patchGridWidth * patchGridHeight
        guard patchGridWidth > 0, patchGridHeight > 0,
              patchGridWidth.isMultiple(of: config.poolingKernel),
              patchGridHeight.isMultiple(of: config.poolingKernel),
              rows <= config.maximumPatches else {
            throw VisionRuntimeError.invalidInput(
                "patch grid must be positive, merge-aligned, and within the family patch budget")
        }
        // Coordinates are validated here, once, rather than per element in the
        // kernel: `vision.metal` reinterprets them as unsigned and multiplies
        // them into the position table, so a -1 from a caller building its own
        // buffers wraps to 4294967295 and reads far past the mapping — a GPU
        // fault, or a confident description of an image nobody sent.
        // `contents()` is invalid for private storage, so the CPU-side loop
        // below would fault on exactly the caller-built buffer it validates.
        guard positionsInt32x2.storageMode != .private else {
            throw VisionRuntimeError.invalidInput(
                "patch positions must be CPU-visible (shared) so they can be validated")
        }
        let positionCount = positionsInt32x2.length / (2 * MemoryLayout<Int32>.stride)
        if positionCount >= rows {
            let coordinates = positionsInt32x2.contents()
                .bindMemory(to: Int32.self, capacity: positionCount * 2)
            let limit = Int32(config.positionGridSide)
            for index in 0..<(rows * 2) {
                let value = coordinates[index]
                guard value >= 0, value < limit else {
                    throw VisionRuntimeError.invalidInput(
                        "patch position \(value) outside 0..<\(limit)")
                }
            }
        }
        guard patchesBF16.length >= rows * config.patchDimension * 2,
              positionsInt32x2.length >= rows * 2 * MemoryLayout<Int32>.stride else {
            throw VisionRuntimeError.invalidInput("patch or position buffer is too small")
        }

        let gpuDrainNanoseconds = try languageModel != nil && residencyPolicy == .onDemand
            ? drainGPU() : 0
        try checkCancellation()
        let transition = languageModel?.prepareExpertResidencyForVision(
            residencyPolicy, gpuDrainNanoseconds: gpuDrainNanoseconds)
        try checkCancellation()
        return try encodePreparedPatches(
            patchesBF16: patchesBF16,
            positionsInt32x2: positionsInt32x2,
            patchGridWidth: patchGridWidth,
            patchGridHeight: patchGridHeight,
            expertResidencyTransition: transition,
            preprocessing: nil,
            retainsWeightRegions: residencyPolicy == .keepReady,
            checkCancellation: checkCancellation)
    }

    private func encodePreparedPatches(
        patchesBF16: MTLBuffer,
        positionsInt32x2: MTLBuffer,
        patchGridWidth: Int,
        patchGridHeight: Int,
        expertResidencyTransition: VisionExpertResidencyTransition?,
        preprocessing: VisionPreprocessingReport?,
        retainsWeightRegions: Bool,
        checkCancellation: () throws -> Void
    ) throws -> VisionFeatures {
        if !retainsWeightRegions { releaseCachedWeightRegions() }
        try checkCancellation()
        let rows = patchGridWidth * patchGridHeight
        if config.family == .qwen36 {
            return try encodeQwenPreparedPatches(
                patchesBF16: patchesBF16,
                positionsInt32x2: positionsInt32x2,
                patchGridWidth: patchGridWidth,
                patchGridHeight: patchGridHeight,
                expertResidencyTransition: expertResidencyTransition,
                preprocessing: preprocessing,
                retainsWeightRegions: retainsWeightRegions,
                checkCancellation: checkCancellation)
        }
        if config.architecture == .gemma4Unified12B {
            return try encodeUnified12BPreparedPatches(
                patchesBF16: patchesBF16,
                positionsInt32x2: positionsInt32x2,
                patchGridWidth: patchGridWidth,
                patchGridHeight: patchGridHeight,
                expertResidencyTransition: expertResidencyTransition,
                preprocessing: preprocessing,
                retainsWeightRegions: retainsWeightRegions,
                checkCancellation: checkCancellation)
        }

        let wallStarted = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        // The QKV projections write the padded head-major layout directly when the
        // register GEMM and the MPP attention kernel both support it, which removes
        // the pad/unpad dispatches and the hidden-width K/V scratch. The attention
        // object owns the decision — it was constructed with the linear path's
        // capability — so it is read, never re-derived, here.
        let fusedPaddedLayout = attention.usesFusedPaddedLayout
        // The up projection folds GeGLU into its epilogue, which removes the
        // separate elementwise dispatch and the `up` scratch buffer.
        let fusedGeGLU = fusesGeGLU
            && linear.supportsFusedGeGLU(n: config.intermediateSize,
                                         k: config.hiddenSize)
        let scratch = try VisionScratch(device: context.device, rows: rows,
                                        config: config,
                                        needsSeparateKV: !fusedPaddedLayout,
                                        needsSeparateUp: !fusedGeGLU)
        let qBuffer = fusedPaddedLayout ? (attention.fusedQ ?? scratch.q) : scratch.q
        let kBuffer = fusedPaddedLayout ? (attention.fusedK ?? scratch.q) : (scratch.k ?? scratch.q)
        let vBuffer = fusedPaddedLayout ? (attention.fusedV ?? scratch.q) : (scratch.v ?? scratch.q)
        let paddedHeadStore = fusedPaddedLayout
            ? VisionLinearBF16.PaddedHeadStore(
                rowStride: scratch.paddedRows,
                headDim: VisionAttention.headDimension,
                paddedHeadDim: VisionAttention.paddedHeadDimension)
            : nil
        let scratchAllocationNanoseconds =
            clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - wallStarted
        var weightMapNanoseconds: UInt64 = 0
        var gpuWaitNanoseconds: UInt64 = 0
        func mapRegion(_ tensorNames: [String]) throws -> VisionMappedWeightRegion {
            let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            defer {
                weightMapNanoseconds += clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started
            }
            let key = tensorNames.first ?? ""
            if retainsWeightRegions, let cached = cachedRegions[key] {
                return cached
            }
            let region = try store.mapRegion(
                tensorNames: tensorNames, device: context.device)
            if retainsWeightRegions { cachedRegions[key] = region }
            return region
        }
        func commitMeasured(_ commandBuffer: MTLCommandBuffer) throws -> UInt64 {
            let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            defer {
                gpuWaitNanoseconds += clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started
            }
            return try commitAndMeasure(commandBuffer)
        }
        var gpuNanoseconds: UInt64 = 0
        // The non-profile encode pipelines command buffers: the CPU encodes
        // the next layer while the GPU executes the previous one, instead of
        // a blocking drain per layer — 29 per image before this. An in-flight
        // entry keeps its mapped weight region alive until its buffer
        // completes, and the window bounds `.onDemand` residency to
        // `pipelineDepth + 1` regions instead of the whole tower.
        var inFlight: [(buffer: MTLCommandBuffer, region: VisionMappedWeightRegion?)] = []
        let pipelineDepth = 2
        func drainOldest() throws {
            let entry = inFlight.removeFirst()
            let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            entry.buffer.waitUntilCompleted()
            gpuWaitNanoseconds += clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started
            if let error = entry.buffer.error {
                throw VisionRuntimeError.commandFailed(String(describing: error))
            }
            gpuNanoseconds += UInt64(
                max(0, entry.buffer.gpuEndTime - entry.buffer.gpuStartTime)
                    * 1_000_000_000)
        }
        func commitPipelined(_ buffer: MTLCommandBuffer,
                             retaining region: VisionMappedWeightRegion?) throws {
            buffer.commit()
            inFlight.append((buffer, region))
            while inFlight.count > pipelineDepth { try drainOldest() }
        }
        // A throw mid-encode must not release mapped regions while the GPU is
        // still reading them; defers run before locals, so drain first.
        defer {
            for entry in inFlight { entry.buffer.waitUntilCompleted() }
        }
        var stagePatchPosition: UInt64 = 0
        var stageAttention: UInt64 = 0
        var stageLinear: UInt64 = 0
        var stageElementwise: UInt64 = 0
        var stagePoolProjector: UInt64 = 0
        var stageCommandBuffers = 0
        var stageAttentionPad: UInt64 = 0
        var stageAttentionCore: UInt64 = 0
        var stageAttentionUnpad: UInt64 = 0
        var lastStageElapsed: UInt64 = 0
        lastStageProfile = nil

        // In stage-profile mode each stage class runs in its own command buffer so
        // its GPU time is attributable; otherwise `shared` carries the whole layer.
        func stage(_ kind: VisionStageClass, on shared: MTLCommandBuffer?,
                   _ body: (MTLCommandBuffer) throws -> Void) throws {
            if let shared {
                try body(shared)
                return
            }
            let buffer = try makeCommandBuffer()
            try body(buffer)
            let elapsed = try commitMeasured(buffer)
            gpuNanoseconds += elapsed
            lastStageElapsed = elapsed
            stageCommandBuffers += 1
            switch kind {
            case .patchPosition: stagePatchPosition += elapsed
            case .attention: stageAttention += elapsed
            case .linear: stageLinear += elapsed
            case .elementwise: stageElementwise += elapsed
            case .poolProjector: stagePoolProjector += elapsed
            }
        }

        let patchWeight = Self.patchEmbedderTensorNames[0]
        let positionTable = Self.patchEmbedderTensorNames[1]
        do {
            let fixed = try mapRegion(Self.patchEmbedderTensorNames)
            let shared = splitsStagesForProfile ? nil : try makeCommandBuffer()
            try stage(.patchPosition, on: shared) { commandBuffer in
                try scratch.encodeClear(commandBuffer: commandBuffer)
                attention.encodeClearPadding(commandBuffer: commandBuffer)
                primitives.encodeNormalize(commandBuffer: commandBuffer,
                                           input: patchesBF16,
                                           output: scratch.normalizedPatches,
                                           rows: rows, paddedRows: scratch.paddedRows,
                                           patchDimension: config.patchDimension)
                linear.encode(
                    commandBuffer: commandBuffer,
                    input: scratch.normalizedPatches,
                    weights: fixed.buffer, weightsOffset: try fixed.offset(of: patchWeight),
                    output: scratch.hiddenA,
                    m: scratch.paddedRows, n: config.hiddenSize, k: config.patchDimension)
                primitives.encodeAddPosition(
                    commandBuffer: commandBuffer,
                    hidden: scratch.hiddenA,
                    table: fixed.buffer, tableOffset: try fixed.offset(of: positionTable),
                    positions: positionsInt32x2, rows: rows,
                    width: config.hiddenSize,
                    positionSize: config.positionEmbeddingSize)
            }
            if let shared {
                try commitPipelined(shared, retaining: fixed)
            }
        }

        for layer in 0..<config.numLayers {
            try checkCancellation()
            let prefix = "vision_tower.encoder.layers.\(layer)."
            let names = store.tensorNames(withPrefix: prefix)
            guard names.count == 13 else {
                throw VisionRuntimeError.invalidInput(
                    "layer \(layer) expected 13 tensors, found \(names.count)")
            }
            let weights = try mapRegion(names)
            let shared = splitsStagesForProfile ? nil : try makeCommandBuffer()
            func offset(_ suffix: String) throws -> Int {
                try weights.offset(of: prefix + suffix)
            }

            try stage(.elementwise, on: shared) { commandBuffer in
                primitives.encodeRMSNorm(
                    commandBuffer: commandBuffer,
                    input: scratch.hiddenA, output: scratch.hiddenB,
                    weightBuffer: weights.buffer,
                    weightOffset: try offset("input_layernorm.weight"), rows: rows,
                    width: config.hiddenSize)
            }
            try stage(.linear, on: shared) { commandBuffer in
                for (suffix, output) in [
                    ("self_attn.q_proj.linear.weight", qBuffer),
                    ("self_attn.k_proj.linear.weight", kBuffer),
                    ("self_attn.v_proj.linear.weight", vBuffer),
                ] {
                    linear.encode(
                        commandBuffer: commandBuffer,
                        input: scratch.hiddenB,
                        weights: weights.buffer, weightsOffset: try offset(suffix),
                        output: output,
                        m: scratch.paddedRows, n: config.hiddenSize, k: config.hiddenSize,
                        paddedHeadStore: paddedHeadStore)
                }
            }
            try stage(.elementwise, on: shared) { commandBuffer in
                primitives.encodeQKV(
                    commandBuffer: commandBuffer,
                    q: qBuffer, k: kBuffer, v: vBuffer,
                    weights: weights.buffer,
                    qWeightOffset: try offset("self_attn.q_norm.weight"),
                    kWeightOffset: try offset("self_attn.k_norm.weight"),
                    positions: positionsInt32x2,
                    rows: rows, heads: config.numHeads,
                    paddedRowStride: fusedPaddedLayout ? scratch.paddedRows : 0)
            }
            let inputRowStride = fusedPaddedLayout ? scratch.paddedRows : 0
            if let shared {
                attention.encode(commandBuffer: shared,
                                 q: qBuffer, k: kBuffer, v: vBuffer,
                                 output: scratch.attention,
                                 sequenceLength: rows, numHeads: config.numHeads,
                                 inputRowStride: inputRowStride)
            } else {
                var phaseError: Error?
                attention.encodePhased(
                    q: qBuffer, k: kBuffer, v: vBuffer,
                    output: scratch.attention,
                    sequenceLength: rows, numHeads: config.numHeads,
                    inputRowStride: inputRowStride
                ) { phase, body in
                    guard phaseError == nil else { return }
                    do {
                        try stage(.attention, on: nil) { commandBuffer in
                            body(commandBuffer)
                        }
                        switch phase {
                        case .pad: stageAttentionPad += lastStageElapsed
                        case .core: stageAttentionCore += lastStageElapsed
                        case .unpad: stageAttentionUnpad += lastStageElapsed
                        }
                    } catch {
                        phaseError = error
                    }
                }
                if let phaseError { throw phaseError }
            }
            try stage(.linear, on: shared) { commandBuffer in
                linear.encode(
                    commandBuffer: commandBuffer,
                    input: scratch.attention,
                    weights: weights.buffer,
                    weightsOffset: try offset("self_attn.o_proj.linear.weight"),
                    output: scratch.q,
                    m: scratch.paddedRows, n: config.hiddenSize, k: config.hiddenSize)
            }
            try stage(.elementwise, on: shared) { commandBuffer in
                primitives.encodePostnormResidual(
                    commandBuffer: commandBuffer,
                    residual: scratch.hiddenA, branch: scratch.q,
                    weightBuffer: weights.buffer,
                    weightOffset: try offset("post_attention_layernorm.weight"),
                    output: scratch.hiddenB, rows: rows,
                    width: config.hiddenSize)
                primitives.encodeRMSNorm(
                    commandBuffer: commandBuffer,
                    input: scratch.hiddenB, output: scratch.q,
                    weightBuffer: weights.buffer,
                    weightOffset: try offset("pre_feedforward_layernorm.weight"), rows: rows,
                    width: config.hiddenSize)
            }
            try stage(.linear, on: shared) { commandBuffer in
                linear.encode(
                    commandBuffer: commandBuffer,
                    input: scratch.q,
                    weights: weights.buffer,
                    weightsOffset: try offset("mlp.gate_proj.linear.weight"),
                    output: scratch.gate,
                    m: scratch.paddedRows, n: config.intermediateSize,
                    k: config.hiddenSize)
                linear.encode(
                    commandBuffer: commandBuffer,
                    input: scratch.q,
                    weights: weights.buffer,
                    weightsOffset: try offset("mlp.up_proj.linear.weight"),
                    output: fusedGeGLU ? scratch.gate : (scratch.up ?? scratch.gate),
                    m: scratch.paddedRows, n: config.intermediateSize,
                    k: config.hiddenSize,
                    fusesGeGLU: fusedGeGLU)
            }
            if !fusedGeGLU, let up = scratch.up {
                try stage(.elementwise, on: shared) { commandBuffer in
                    primitives.encodeGeGLU(commandBuffer: commandBuffer,
                                           gate: scratch.gate, up: up,
                                           count: rows * config.intermediateSize)
                }
            }
            try stage(.linear, on: shared) { commandBuffer in
                linear.encode(
                    commandBuffer: commandBuffer,
                    input: scratch.gate,
                    weights: weights.buffer,
                    weightsOffset: try offset("mlp.down_proj.linear.weight"),
                    output: scratch.q,
                    m: scratch.paddedRows, n: config.hiddenSize, k: config.intermediateSize)
            }
            try stage(.elementwise, on: shared) { commandBuffer in
                primitives.encodePostnormResidual(
                    commandBuffer: commandBuffer,
                    residual: scratch.hiddenB, branch: scratch.q,
                    weightBuffer: weights.buffer,
                    weightOffset: try offset("post_feedforward_layernorm.weight"),
                    output: scratch.hiddenA, rows: rows,
                    width: config.hiddenSize)
            }
            if let shared {
                try commitPipelined(shared, retaining: weights)
            }
        }

        let outputRows = (patchGridWidth / config.poolingKernel)
            * (patchGridHeight / config.poolingKernel)
        try checkCancellation()
        let projectorNames = config.usesESeriesVisionTower
            ? Array(Self.projectorTensorNames.suffix(3))
            : Self.projectorTensorNames
        let projector = try mapRegion(projectorNames)
        guard let features = context.device.makeBuffer(
            length: outputRows * config.textHiddenSize * MemoryLayout<Float16>.stride,
            options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        guard projectorKernel.isAvailable else {
            throw VisionRuntimeError.unsupportedKernel(
                projectorKernel.unavailableReason ?? "INT4 projector unavailable")
        }
        let shared = splitsStagesForProfile ? nil : try makeCommandBuffer()
        var projectorMetadata: MPPPrefillInt4QMM.PathMetadata?
        try stage(.poolProjector, on: shared) { commandBuffer in
            if config.usesESeriesVisionTower {
                primitives.encodePoolNorm(
                    commandBuffer: commandBuffer,
                    input: scratch.hiddenA,
                    output: scratch.attention,
                    patchWidth: patchGridWidth,
                    patchHeight: patchGridHeight,
                    width: config.hiddenSize,
                    kernel: config.poolingKernel)
            } else {
                primitives.encodePool(
                    commandBuffer: commandBuffer,
                    input: scratch.hiddenA,
                    weights: projector.buffer,
                    stdBiasOffset: try projector.offset(of: "vision_tower.std_bias"),
                    stdScaleOffset: try projector.offset(of: "vision_tower.std_scale"),
                    output: scratch.attention,
                    patchWidth: patchGridWidth, patchHeight: patchGridHeight)
            }
            if !usesBF16Projector {
                primitives.encodeBFloatToHalf(
                    commandBuffer: commandBuffer,
                    input: scratch.attention, output: scratch.hiddenB,
                    count: outputRows * config.hiddenSize)
            }
            projectorMetadata = projectorKernel.encode(
                commandBuffer: commandBuffer,
                weights: projector.buffer,
                weightsOffset: try projector.offset(
                    of: "embed_vision.embedding_projection.weight"),
                scales: projector.buffer,
                scalesOffset: try projector.offset(
                    of: "embed_vision.embedding_projection.scales"),
                biases: projector.buffer,
                biasesOffset: try projector.offset(
                    of: "embed_vision.embedding_projection.biases"),
                x: usesBF16Projector ? scratch.attention : scratch.hiddenB,
                y: features,
                m: outputRows, n: config.textHiddenSize, k: config.hiddenSize)
        }
        guard let projectorMetadata, projectorMetadata.path != .fallback else {
            throw VisionRuntimeError.unsupportedKernel("INT4 projector fell back")
        }
        if let shared {
            try commitPipelined(shared, retaining: projector)
        }
        while !inFlight.isEmpty { try drainOldest() }
        if splitsStagesForProfile {
            lastStageProfile = VisionStageProfile(
                patchPositionNanoseconds: stagePatchPosition,
                attentionNanoseconds: stageAttention,
                attentionPadNanoseconds: stageAttentionPad,
                attentionCoreNanoseconds: stageAttentionCore,
                attentionUnpadNanoseconds: stageAttentionUnpad,
                linearNanoseconds: stageLinear,
                elementwiseNanoseconds: stageElementwise,
                poolProjectorNanoseconds: stagePoolProjector,
                commandBufferCount: stageCommandBuffers,
                layerCount: config.numLayers)
        }

        return VisionFeatures(buffer: features,
                              tokenCount: outputRows,
                              hiddenSize: config.textHiddenSize,
                              family: config.family,
                              patchGridWidth: patchGridWidth,
                              patchGridHeight: patchGridHeight,
                              gpuNanoseconds: gpuNanoseconds,
                              scratchBytes: scratch.allocatedBytes
                                  + attention.additionalScratchBytes,
                              attentionVariant: attention.variant,
                              projectorPath: projectorMetadata.path,
                              expertResidencyTransition: expertResidencyTransition,
                              preprocessing: preprocessing,
                              wall: VisionWallBreakdown(
                                  scratchAllocationNanoseconds: scratchAllocationNanoseconds,
                                  weightMapNanoseconds: weightMapNanoseconds,
                                  gpuWaitNanoseconds: gpuWaitNanoseconds,
                                  totalNanoseconds: clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                                      - wallStarted))
    }

    /// Gemma 4 Unified folds the entire image encoder into two quantized
    /// projections around normalization and factorized 2-D position stages.
    /// It deliberately bypasses the legacy SigLIP tower: the two formats share
    /// a family name but no layer graph or tensor naming contract.
    private func encodeUnified12BPreparedPatches(
        patchesBF16: MTLBuffer,
        positionsInt32x2: MTLBuffer,
        patchGridWidth: Int,
        patchGridHeight: Int,
        expertResidencyTransition: VisionExpertResidencyTransition?,
        preprocessing: VisionPreprocessingReport?,
        retainsWeightRegions: Bool,
        checkCancellation: () throws -> Void
    ) throws -> VisionFeatures {
        let wallStarted = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let rows = patchGridWidth * patchGridHeight
        func buffer(_ elements: Int, shared: Bool = false) throws -> MTLBuffer {
            guard let value = context.device.makeBuffer(
                length: elements * MemoryLayout<UInt16>.stride,
                options: shared ? .storageModeShared : .storageModePrivate) else {
                throw MetalError.noDevice
            }
            return value
        }
        let patchNormalized = try buffer(rows * config.patchDimension)
        let qmmInput = try buffer(rows * config.patchDimension)
        let denseBF16 = try buffer(rows * config.hiddenSize)
        let hiddenA = try buffer(rows * config.hiddenSize)
        let hiddenB = try buffer(rows * config.hiddenSize)
        let features = try buffer(rows * config.textHiddenSize, shared: true)
        let allocationFinished = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)

        var mapNanoseconds: UInt64 = 0
        func map(_ names: [String]) throws -> VisionMappedWeightRegion {
            let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            defer { mapNanoseconds += clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started }
            let key = names.first ?? ""
            if retainsWeightRegions, let cached = cachedRegions[key] { return cached }
            let region = try store.mapRegion(tensorNames: names, device: context.device)
            if retainsWeightRegions { cachedRegions[key] = region }
            return region
        }

        try checkCancellation()
        let patch = try map(Self.unifiedPatchTensorNames)
        let projector = try map(Self.unifiedProjectorTensorNames)
        guard projectorKernel.isAvailable, unifiedPatchProjectorKernel.isAvailable else {
            throw VisionRuntimeError.unsupportedKernel(
                unifiedPatchProjectorKernel.unavailableReason
                    ?? projectorKernel.unavailableReason
                    ?? "INT4 image projection unavailable")
        }
        let commandBuffer = try makeCommandBuffer()
        primitives.encodeQwenLayerNorm(
            commandBuffer: commandBuffer,
            input: patchesBF16, output: patchNormalized,
            weights: patch.buffer,
            weightOffset: try patch.offset(of: "vision_embedder.patch_ln1.weight"),
            biasOffset: try patch.offset(of: "vision_embedder.patch_ln1.bias"),
            rows: rows, width: config.patchDimension)
        if !context.device.supportsFamily(.apple10) {
            primitives.encodeBFloatToHalf(
                commandBuffer: commandBuffer,
                input: patchNormalized, output: qmmInput,
                count: rows * config.patchDimension)
        }
        let patchMetadata = unifiedPatchProjectorKernel.encode(
            commandBuffer: commandBuffer,
            weights: patch.buffer,
            weightsOffset: try patch.offset(of: "vision_embedder.patch_dense.weight"),
            scales: patch.buffer,
            scalesOffset: try patch.offset(of: "vision_embedder.patch_dense.scales"),
            biases: patch.buffer,
            biasesOffset: try patch.offset(of: "vision_embedder.patch_dense.biases"),
            x: context.device.supportsFamily(.apple10) ? patchNormalized : qmmInput,
            y: denseBF16,
            m: rows, n: config.hiddenSize, k: config.patchDimension)
        guard patchMetadata.path != .fallback else {
            throw VisionRuntimeError.unsupportedKernel("unified patch projection fell back")
        }
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw VisionRuntimeError.unsupportedKernel("unified BF16 copy unavailable")
        }
        blit.copy(from: denseBF16, sourceOffset: 0, to: hiddenA,
                  destinationOffset: 0, size: denseBF16.length)
        blit.endEncoding()
        primitives.encodeQwenAddBias(
            commandBuffer: commandBuffer,
            values: hiddenA, weights: patch.buffer,
            biasOffset: try patch.offset(of: "vision_embedder.patch_dense.bias"),
            count: rows * config.hiddenSize, width: config.hiddenSize)
        primitives.encodeQwenLayerNorm(
            commandBuffer: commandBuffer,
            input: hiddenA, output: hiddenB,
            weights: patch.buffer,
            weightOffset: try patch.offset(of: "vision_embedder.patch_ln2.weight"),
            biasOffset: try patch.offset(of: "vision_embedder.patch_ln2.bias"),
            rows: rows, width: config.hiddenSize)
        primitives.encodeUnifiedAddPosition(
            commandBuffer: commandBuffer,
            hidden: hiddenB, table: patch.buffer,
            tableOffset: try patch.offset(of: "vision_embedder.pos_embedding"),
            positions: positionsInt32x2, rows: rows, width: config.hiddenSize)
        primitives.encodeQwenLayerNorm(
            commandBuffer: commandBuffer,
            input: hiddenB, output: hiddenA,
            weights: patch.buffer,
            weightOffset: try patch.offset(of: "vision_embedder.pos_norm.weight"),
            biasOffset: try patch.offset(of: "vision_embedder.pos_norm.bias"),
            rows: rows, width: config.hiddenSize)
        primitives.encodeRMSNormNoScale(
            commandBuffer: commandBuffer,
            input: hiddenA, output: hiddenB,
            rows: rows, width: config.hiddenSize)
        if !usesBF16Projector {
            primitives.encodeBFloatToHalf(
                commandBuffer: commandBuffer,
                input: hiddenB, output: qmmInput,
                count: rows * config.hiddenSize)
        }
        let projectorMetadata = projectorKernel.encode(
            commandBuffer: commandBuffer,
            weights: projector.buffer,
            weightsOffset: try projector.offset(
                of: "embed_vision.embedding_projection.weight"),
            scales: projector.buffer,
            scalesOffset: try projector.offset(
                of: "embed_vision.embedding_projection.scales"),
            biases: projector.buffer,
            biasesOffset: try projector.offset(
                of: "embed_vision.embedding_projection.biases"),
            x: usesBF16Projector ? hiddenB : qmmInput,
            y: features,
            m: rows, n: config.textHiddenSize, k: config.hiddenSize)
        guard projectorMetadata.path != .fallback else {
            throw VisionRuntimeError.unsupportedKernel("unified embedding projection fell back")
        }
        let waitStarted = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let waitNanoseconds = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - waitStarted
        if let error = commandBuffer.error {
            throw VisionRuntimeError.commandFailed(String(describing: error))
        }
        let gpuNanoseconds = UInt64(
            max(0, commandBuffer.gpuEndTime - commandBuffer.gpuStartTime)
                * 1_000_000_000)
        let scratchBytes = patchNormalized.length + qmmInput.length
            + denseBF16.length + hiddenA.length + hiddenB.length
        return VisionFeatures(
            buffer: features,
            tokenCount: rows,
            hiddenSize: config.textHiddenSize,
            family: config.family,
            patchGridWidth: patchGridWidth,
            patchGridHeight: patchGridHeight,
            gpuNanoseconds: gpuNanoseconds,
            scratchBytes: scratchBytes,
            attentionVariant: attention.variant,
            projectorPath: projectorMetadata.path,
            expertResidencyTransition: expertResidencyTransition,
            preprocessing: preprocessing,
            wall: VisionWallBreakdown(
                scratchAllocationNanoseconds: allocationFinished - wallStarted,
                weightMapNanoseconds: mapNanoseconds,
                gpuWaitNanoseconds: waitNanoseconds,
                totalNanoseconds: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - wallStarted))
    }

    /// Qwen3.6's tower shares the Gemma allocation, GEMM, attention and pack
    /// machinery, but owns its tensor names and block/merger graph. Keeping the
    /// family branch here prevents Qwen biases, LayerNorms and 2x2 merger from
    /// leaking into the Gemma path.
    private func encodeQwenPreparedPatches(
        patchesBF16: MTLBuffer,
        positionsInt32x2: MTLBuffer,
        patchGridWidth: Int,
        patchGridHeight: Int,
        expertResidencyTransition: VisionExpertResidencyTransition?,
        preprocessing: VisionPreprocessingReport?,
        retainsWeightRegions: Bool,
        checkCancellation: () throws -> Void
    ) throws -> VisionFeatures {
        let wallStarted = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let rows = patchGridWidth * patchGridHeight
        let scratch = try VisionScratch(
            device: context.device, rows: rows, config: config,
            needsSeparateKV: true, needsSeparateUp: false)
        let allocationNanoseconds = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - wallStarted
        var mapNanoseconds: UInt64 = 0
        var waitNanoseconds: UInt64 = 0
        var gpuNanoseconds: UInt64 = 0

        func map(_ names: [String]) throws -> VisionMappedWeightRegion {
            let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            defer { mapNanoseconds += clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started }
            let key = names.first ?? ""
            if retainsWeightRegions, let cached = cachedRegions[key] { return cached }
            let region = try store.mapRegion(tensorNames: names, device: context.device)
            if retainsWeightRegions { cachedRegions[key] = region }
            return region
        }
        func finish(_ commandBuffer: MTLCommandBuffer) throws {
            let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            waitNanoseconds += clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started
            if let error = commandBuffer.error {
                throw VisionRuntimeError.commandFailed(String(describing: error))
            }
            gpuNanoseconds += UInt64(
                max(0, commandBuffer.gpuEndTime - commandBuffer.gpuStartTime)
                    * 1_000_000_000)
        }

        let patch = try map(Self.qwenPatchTensorNames)
        do {
            let commandBuffer = try makeCommandBuffer()
            try scratch.encodeClear(commandBuffer: commandBuffer)
            primitives.encodeQwenCopyPadded(
                commandBuffer: commandBuffer,
                input: patchesBF16, output: scratch.normalizedPatches,
                rows: rows, paddedRows: scratch.paddedRows,
                width: config.patchDimension)
            linear.encode(
                commandBuffer: commandBuffer,
                input: scratch.normalizedPatches,
                weights: patch.buffer,
                weightsOffset: try patch.offset(of: "vision_tower.patch_embed.proj.weight"),
                output: scratch.hiddenA,
                m: scratch.paddedRows, n: config.hiddenSize, k: config.patchDimension)
            primitives.encodeQwenAddBias(
                commandBuffer: commandBuffer,
                values: scratch.hiddenA, weights: patch.buffer,
                biasOffset: try patch.offset(of: "vision_tower.patch_embed.proj.bias"),
                count: rows * config.hiddenSize, width: config.hiddenSize)
            primitives.encodeQwenAddPosition(
                commandBuffer: commandBuffer,
                hidden: scratch.hiddenA,
                table: patch.buffer,
                tableOffset: try patch.offset(of: "vision_tower.pos_embed.weight"),
                positions: positionsInt32x2, rows: rows,
                gridHeight: patchGridHeight, gridWidth: patchGridWidth)
            try finish(commandBuffer)
        }

        let componentWeightBytes = config.hiddenSize * config.hiddenSize
            * MemoryLayout<UInt16>.stride
        let componentBiasBytes = config.hiddenSize * MemoryLayout<UInt16>.stride
        for layer in 0..<config.numLayers {
            try checkCancellation()
            let prefix = "vision_tower.blocks.\(layer)."
            let names = store.tensorNames(withPrefix: prefix)
            guard names.count == 12 else {
                throw VisionRuntimeError.invalidInput(
                    "Qwen vision block \(layer) expected 12 tensors, found \(names.count)")
            }
            let weights = try map(names)
            func offset(_ suffix: String) throws -> Int {
                try weights.offset(of: prefix + suffix)
            }
            let commandBuffer = try makeCommandBuffer()
            primitives.encodeQwenLayerNorm(
                commandBuffer: commandBuffer,
                input: scratch.hiddenA, output: scratch.hiddenB,
                weights: weights.buffer,
                weightOffset: try offset("norm1.weight"),
                biasOffset: try offset("norm1.bias"), rows: rows)
            let qkvWeight = try offset("attn.qkv.weight")
            linear.encode(
                commandBuffer: commandBuffer,
                input: scratch.hiddenB, weights: weights.buffer,
                weightsOffset: qkvWeight,
                output: scratch.q,
                m: scratch.paddedRows, n: config.hiddenSize, k: config.hiddenSize)
            guard let k = scratch.k, let v = scratch.v else {
                throw VisionRuntimeError.invalidInput("Qwen QKV scratch is unavailable")
            }
            linear.encode(
                commandBuffer: commandBuffer,
                input: scratch.hiddenB, weights: weights.buffer,
                weightsOffset: qkvWeight + componentWeightBytes,
                output: k,
                m: scratch.paddedRows, n: config.hiddenSize, k: config.hiddenSize)
            linear.encode(
                commandBuffer: commandBuffer,
                input: scratch.hiddenB, weights: weights.buffer,
                weightsOffset: qkvWeight + componentWeightBytes * 2,
                output: v,
                m: scratch.paddedRows, n: config.hiddenSize, k: config.hiddenSize)
            let qkvBias = try offset("attn.qkv.bias")
            primitives.encodeQwenQKV(
                commandBuffer: commandBuffer,
                q: scratch.q, k: k, v: v,
                weights: weights.buffer,
                qBiasOffset: qkvBias,
                kBiasOffset: qkvBias + componentBiasBytes,
                vBiasOffset: qkvBias + componentBiasBytes * 2,
                positions: positionsInt32x2, rows: rows, heads: config.numHeads)
            attention.encode(
                commandBuffer: commandBuffer,
                q: scratch.q, k: k, v: v,
                output: scratch.attention,
                sequenceLength: rows, numHeads: config.numHeads)
            linear.encode(
                commandBuffer: commandBuffer,
                input: scratch.attention, weights: weights.buffer,
                weightsOffset: try offset("attn.proj.weight"),
                output: scratch.q,
                m: scratch.paddedRows, n: config.hiddenSize, k: config.hiddenSize)
            primitives.encodeQwenResidualBias(
                commandBuffer: commandBuffer,
                residual: scratch.hiddenA, branch: scratch.q,
                weights: weights.buffer,
                biasOffset: try offset("attn.proj.bias"),
                output: scratch.hiddenB, rows: rows, width: config.hiddenSize)
            primitives.encodeQwenLayerNorm(
                commandBuffer: commandBuffer,
                input: scratch.hiddenB, output: scratch.hiddenA,
                weights: weights.buffer,
                weightOffset: try offset("norm2.weight"),
                biasOffset: try offset("norm2.bias"), rows: rows)
            linear.encode(
                commandBuffer: commandBuffer,
                input: scratch.hiddenA, weights: weights.buffer,
                weightsOffset: try offset("mlp.linear_fc1.weight"),
                output: scratch.gate,
                m: scratch.paddedRows, n: config.intermediateSize, k: config.hiddenSize)
            primitives.encodeQwenGELUTanhBias(
                commandBuffer: commandBuffer,
                values: scratch.gate, weights: weights.buffer,
                biasOffset: try offset("mlp.linear_fc1.bias"),
                count: rows * config.intermediateSize,
                width: config.intermediateSize)
            linear.encode(
                commandBuffer: commandBuffer,
                input: scratch.gate, weights: weights.buffer,
                weightsOffset: try offset("mlp.linear_fc2.weight"),
                output: scratch.q,
                m: scratch.paddedRows, n: config.hiddenSize, k: config.intermediateSize)
            primitives.encodeQwenResidualBias(
                commandBuffer: commandBuffer,
                residual: scratch.hiddenB, branch: scratch.q,
                weights: weights.buffer,
                biasOffset: try offset("mlp.linear_fc2.bias"),
                output: scratch.hiddenA, rows: rows, width: config.hiddenSize)
            try finish(commandBuffer)
        }

        try checkCancellation()
        let outputRows = rows / (config.poolingKernel * config.poolingKernel)
        let merger = try map(Self.qwenMergerTensorNames)
        guard let featuresBF16 = context.device.makeBuffer(
                  length: outputRows * config.textHiddenSize
                      * MemoryLayout<UInt16>.stride,
                  options: .storageModePrivate),
              let features = context.device.makeBuffer(
                  length: outputRows * config.textHiddenSize
                      * MemoryLayout<Float16>.stride,
                  options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        do {
            let commandBuffer = try makeCommandBuffer()
            primitives.encodeQwenLayerNorm(
                commandBuffer: commandBuffer,
                input: scratch.hiddenA, output: scratch.hiddenB,
                weights: merger.buffer,
                weightOffset: try merger.offset(of: "vision_tower.merger.norm.weight"),
                biasOffset: try merger.offset(of: "vision_tower.merger.norm.bias"),
                rows: rows)
            let mergedWidth = config.hiddenSize * config.poolingKernel
                * config.poolingKernel
            linear.encode(
                commandBuffer: commandBuffer,
                input: scratch.hiddenB, weights: merger.buffer,
                weightsOffset: try merger.offset(
                    of: "vision_tower.merger.linear_fc1.weight"),
                output: scratch.gate,
                m: outputRows, n: mergedWidth, k: mergedWidth)
            primitives.encodeQwenGELUErfBias(
                commandBuffer: commandBuffer,
                values: scratch.gate, weights: merger.buffer,
                biasOffset: try merger.offset(
                    of: "vision_tower.merger.linear_fc1.bias"),
                count: outputRows * mergedWidth, width: mergedWidth)
            linear.encode(
                commandBuffer: commandBuffer,
                input: scratch.gate, weights: merger.buffer,
                weightsOffset: try merger.offset(
                    of: "vision_tower.merger.linear_fc2.weight"),
                output: featuresBF16,
                m: outputRows, n: config.textHiddenSize, k: mergedWidth)
            primitives.encodeQwenAddBias(
                commandBuffer: commandBuffer,
                values: featuresBF16, weights: merger.buffer,
                biasOffset: try merger.offset(
                    of: "vision_tower.merger.linear_fc2.bias"),
                count: outputRows * config.textHiddenSize,
                width: config.textHiddenSize)
            primitives.encodeBFloatToHalf(
                commandBuffer: commandBuffer,
                input: featuresBF16, output: features,
                count: outputRows * config.textHiddenSize)
            try finish(commandBuffer)
        }

        return VisionFeatures(
            buffer: features,
            tokenCount: outputRows,
            hiddenSize: config.textHiddenSize,
            family: .qwen36,
            patchGridWidth: patchGridWidth,
            patchGridHeight: patchGridHeight,
            gpuNanoseconds: gpuNanoseconds,
            scratchBytes: scratch.allocatedBytes + featuresBF16.length,
            attentionVariant: attention.variant,
            projectorPath: .fallback,
            expertResidencyTransition: expertResidencyTransition,
            preprocessing: preprocessing,
            wall: VisionWallBreakdown(
                scratchAllocationNanoseconds: allocationNanoseconds,
                weightMapNanoseconds: mapNanoseconds,
                gpuWaitNanoseconds: waitNanoseconds,
                totalNanoseconds: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - wallStarted))
    }

    private func makeCommandBuffer() throws -> MTLCommandBuffer {
        guard let value = context.queue.makeCommandBuffer() else { throw MetalError.noQueue }
        return value
    }

    private func drainGPU() throws -> UInt64 {
        let started = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let commandBuffer = try makeCommandBuffer()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw VisionRuntimeError.commandFailed(String(describing: error))
        }
        return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - started
    }

    private func commitAndMeasure(_ commandBuffer: MTLCommandBuffer) throws -> UInt64 {
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw VisionRuntimeError.commandFailed(String(describing: error))
        }
        return UInt64(max(0, commandBuffer.gpuEndTime - commandBuffer.gpuStartTime)
                      * 1_000_000_000)
    }

}
