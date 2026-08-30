import Foundation
import Metal
import Darwin
import TUFFFormat

public struct ModelLoadStats: Sendable {
    public var manifestSha256Nanos: UInt64
    public var receiptValidationNanos: UInt64
    public var eagerSha256Nanos: UInt64

    public init(manifestSha256Nanos: UInt64 = 0,
                receiptValidationNanos: UInt64 = 0,
                eagerSha256Nanos: UInt64 = 0) {
        self.manifestSha256Nanos = manifestSha256Nanos
        self.receiptValidationNanos = receiptValidationNanos
        self.eagerSha256Nanos = eagerSha256Nanos
    }
}

/// Bounded routed-expert cache configuration.
public enum ExpertStreamingMode: Sendable {
    /// Read each expert into one of `slotCount` 2 MB-aligned cache slots.
    case pread(slotCount: Int)
}

/// Loaded `.gturbo/` model. Resident weights live behind tensor-safe mmap'd
/// `MTLBuffer` regions; routed expert weights live behind per-layer streaming
/// backends opened lazily on first touch.
public struct Model {
    public let device: MTLDevice
    public let config: ArchConfig
    public let streamingMode: ExpertStreamingMode
    public let expertCachePolicy: ExpertCachePolicy
    public let integrityPolicy: ModelIntegrityPolicy
    public var modelID: String { manifest.modelID }
    public var sourceSnapshotHash: String? { manifest.sourceSnapshotHash }
    public var sharedExpertWeightBits: Int { manifest.quant?.sharedExpert.weightBits ?? 8 }

    let residentBuffers: ResidentBufferSet
    let residentIndex: ResidentIndex
    let packedExpertsLayout: PackedExpertsLayout
    let manifest: Manifest
    let directoryURL: URL
    let modelDirectory: GTurboModelDirectory

    /// Lazy state. Held inside a reference box so `Model` can stay a struct
    /// while still letting accessors mutate layer state via a serial queue.
    let streamersBox: StreamersBox
    let streamersQueue: DispatchQueue

    final class StreamersBox: @unchecked Sendable {
        var streamers: [PreadExpertStreamer?]
        var layerVerified: [Bool]
        init(numLayers: Int) {
            self.streamers = Array(repeating: nil, count: numLayers)
            self.layerVerified = Array(repeating: false, count: numLayers)
        }
    }

    init(device: MTLDevice,
         config: ArchConfig,
         streamingMode: ExpertStreamingMode,
         expertCachePolicy: ExpertCachePolicy,
         integrityPolicy: ModelIntegrityPolicy,
         residentBuffers: ResidentBufferSet,
         residentIndex: ResidentIndex,
         packedExpertsLayout: PackedExpertsLayout,
         manifest: Manifest,
         directoryURL: URL,
         modelDirectory: GTurboModelDirectory) {
        self.device = device
        self.config = config
        self.streamingMode = streamingMode
        self.expertCachePolicy = expertCachePolicy
        self.integrityPolicy = integrityPolicy
        self.residentBuffers = residentBuffers
        self.residentIndex = residentIndex
        self.packedExpertsLayout = packedExpertsLayout
        self.manifest = manifest
        self.directoryURL = directoryURL
        self.modelDirectory = modelDirectory
        self.streamersBox = StreamersBox(numLayers: packedExpertsLayout.numLayers)
        self.streamersQueue = DispatchQueue(label: "turbo-fieldfare.expert-streamers")
    }

    // MARK: - Resident accessors

    public var embedding: TensorView {
        try! resident(name: "language_model.model.embed_tokens.weight")
    }

    /// Gemma 4 E2B/E4B token-identity PLE table. Its logical row width is
    /// `numLayers * hiddenSizePerLayerInput`.
    public var perLayerEmbedding: TensorView {
        try! resident(name: "language_model.model.embed_tokens_per_layer.weight")
    }

    /// Context projection from the main embedding into all packed PLE rows.
    public var perLayerModelProjection: TensorView {
        try! resident(name: "language_model.model.per_layer_model_projection.weight")
    }

    /// One shared gain vector applied independently to every packed PLE row.
    public var perLayerProjectionNorm: TensorView {
        try! resident(name: "language_model.model.per_layer_projection_norm.weight")
    }

    /// Gemma 4 ties lm_head to the embedding; Qwen 3.6 carries a separate
    /// `lm_head` tensor. The transpose for the lm_head GEMV path is the
    /// kernel's job, not the loader's.
    public var lmHead: TensorView {
        if config.tieWordEmbeddings { return embedding }
        return try! resident(name: "language_model.lm_head.weight")
    }

    public func qProj(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.q_proj.weight")
    }
    public func kProj(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.k_proj.weight")
    }
    public func vProj(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.v_proj.weight")
    }
    public func oProj(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.o_proj.weight")
    }
    public func qProjBias(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.q_proj.bias")
    }
    public func kProjBias(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.k_proj.bias")
    }
    public func vProjBias(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.v_proj.bias")
    }
    public func oProjBias(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.o_proj.bias")
    }
    public func attentionSinks(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.sinks")
    }
    /// Gemma's writer emits `.router.proj.weight` (no `.mlp.` segment);
    /// Qwen's router is the source-named `.mlp.gate.weight`.
    public func router(layer L: Int) throws -> TensorView {
        switch config.family {
        case .gemma4:
            return try resident(name: "language_model.model.layers.\(L).router.proj.weight")
        case .qwen36:
            return try resident(name: "language_model.model.layers.\(L).mlp.gate.weight")
        case .gptOss:
            return try resident(name: "language_model.model.layers.\(L).mlp.router.weight")
        }
    }
    public func routerBias(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).mlp.router.bias")
    }
    /// Shared-expert FFN. Gemma emits `.mlp.{gate,up,down}_proj.weight`
    /// without a `.shared_expert.` segment; Qwen keeps the source's
    /// `.mlp.shared_expert.{gate,up,down}_proj.weight` names.
    public func sharedExpertGate(layer L: Int) throws -> TensorView {
        try resident(name: sharedExpertName("gate_proj", layer: L))
    }
    public func sharedExpertUp(layer L: Int) throws -> TensorView {
        try resident(name: sharedExpertName("up_proj", layer: L))
    }
    public func sharedExpertDown(layer L: Int) throws -> TensorView {
        try resident(name: sharedExpertName("down_proj", layer: L))
    }
    private func sharedExpertName(_ proj: String, layer L: Int) -> String {
        switch config.family {
        case .gemma4:
            return "language_model.model.layers.\(L).mlp.\(proj).weight"
        case .qwen36:
            return "language_model.model.layers.\(L).mlp.shared_expert.\(proj).weight"
        case .gptOss:
            return "language_model.model.layers.\(L).mlp.\(proj).weight"
        }
    }
    /// Qwen-only scalar gate on the shared-expert branch: a `[1, hidden]`
    /// 8-bit projection whose sigmoid multiplies the shared FFN output.
    public func sharedExpertScalarGate(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).mlp.shared_expert_gate.weight")
    }
    public func inputNorm(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).input_layernorm.weight")
    }
    public func postAttnNorm(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).post_attention_layernorm.weight")
    }
    public var finalNorm: TensorView {
        try! resident(name: "language_model.model.norm.weight")
    }

    // MARK: - Per-head attention norms (Q/K only)
    //
    // `q_norm` and `k_norm` are RMSNorm with learnable scale, applied per head
    // before RoPE. `v_norm` has **no learnable weight** (no-scale RMSNorm) and
    // is therefore not stored as a tensor — the runtime uses an
    // explicit no-scale variant rather than consuming a unit-weight buffer.

    public func qNorm(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.q_norm.weight")
    }
    public func kNorm(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).self_attn.k_norm.weight")
    }

    // MARK: - Feed-forward norms
    //
    // The Gemma 4 sandwich wraps two parallel FFN branches:
    //   pre_feedforward_layernorm        -> dense MLP input
    //   pre_feedforward_layernorm_2      -> routed expert input
    //   post_feedforward_layernorm_1     -> dense MLP output
    //   post_feedforward_layernorm_2     -> routed expert output
    //   post_feedforward_layernorm       -> combined (h1+h2) output

    public func preFFN(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).pre_feedforward_layernorm.weight")
    }
    public func preFFN2(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).pre_feedforward_layernorm_2.weight")
    }
    public func postFFN1(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).post_feedforward_layernorm_1.weight")
    }
    public func postFFN2(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).post_feedforward_layernorm_2.weight")
    }
    public func postFFN(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).post_feedforward_layernorm.weight")
    }

    // MARK: - Per-layer embedding mapping (dense Gemma 4)

    public func perLayerInputGate(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).per_layer_input_gate.weight")
    }
    public func perLayerProjection(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).per_layer_projection.weight")
    }
    public func postPerLayerInputNorm(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).post_per_layer_input_norm.weight")
    }

    // MARK: - Router auxiliaries
    //
    // `router.scale` is a per-feature multiplier on the router's input
    // (post-RMSNorm), fused with 1/sqrt(hidden_size). `per_expert_scale` is
    // applied to the top-k routing weights after softmax over top-k.

    public func routerScale(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).router.scale")
    }
    public func routerPerExpertScale(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).router.per_expert_scale")
    }

    /// Per-layer scalar gain applied to the entire residual stream at the end
    /// of the layer; shape `[1]`, BF16.
    public func layerScalar(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).layer_scalar")
    }

    // MARK: - Gated-DeltaNet linear attention (Qwen 3.6)
    //
    // Layers whose mask value is 2 replace full/sliding attention with the
    // gated delta rule. Projections are 4-bit affine; the depthwise conv
    // weight, A_log, dt_bias, and the gated output norm are BF16.

    public func linearInProjQKV(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.in_proj_qkv.weight")
    }
    public func linearInProjZ(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.in_proj_z.weight")
    }
    public func linearInProjA(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.in_proj_a.weight")
    }
    public func linearInProjB(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.in_proj_b.weight")
    }
    public func linearOutProj(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.out_proj.weight")
    }
    /// Depthwise causal conv weight, source shape `[convDim, kernel, 1]`, BF16.
    public func linearConv1d(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.conv1d.weight")
    }
    /// Per-value-head decay base, shape `[numVHeads]`, BF16.
    public func linearALog(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.A_log")
    }
    /// Per-value-head dt bias, shape `[numVHeads]`, BF16.
    public func linearDtBias(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.dt_bias")
    }
    /// Gated RMSNorm weight over the value head dim, shape `[valueHeadDim]`.
    public func linearNorm(layer L: Int) throws -> TensorView {
        try resident(name: "language_model.model.layers.\(L).linear_attn.norm.weight")
    }

    /// Resolve a tensor name to a `TensorView` against its resident region.
    /// Absolute file offsets are converted to offsets relative to that
    /// region's `MTLBuffer`.
    func resident(name: String) throws -> TensorView {
        guard let entry = residentIndex.entries[name] else {
            throw ModelError.tensorNotFound(name: name)
        }
        let resolved = try residentBuffers.resolve(tensorName: name)
        let residentFileOffset = resolved.regionFileOffset
        func checkedRelativeOffset(_ absolute: UInt64,
                                   size: UInt64,
                                   field: String) throws -> UInt64 {
            if size == 0 {
                guard absolute == 0 else {
                    throw ModelError.indexCorrupt(detail: "\(name).\(field) has an absent nonzero offset")
                }
                return 0
            }
            guard absolute >= residentFileOffset else {
                throw ModelError.indexCorrupt(detail: "\(name).\(field) precedes its resident region")
            }
            let relative = absolute - residentFileOffset
            guard relative <= UInt64(resolved.buffer.length),
                  size <= UInt64(resolved.buffer.length) - relative else {
                throw ModelError.indexCorrupt(detail: "\(name).\(field) exceeds its resident region")
            }
            return relative
        }
        let relativeOffset = try checkedRelativeOffset(
            entry.fileOffset, size: entry.sizeBytes, field: "weights")
        let scaleRel = try checkedRelativeOffset(
            entry.scaleOffset, size: entry.scaleSize, field: "scales")
        let biasRel = try checkedRelativeOffset(
            entry.biasOffset, size: entry.biasSize, field: "biases")
        return TensorView(
            buffer: resolved.buffer,
            offset: relativeOffset,
            length: entry.sizeBytes,
            scaleOffset: scaleRel, scaleLength: entry.scaleSize,
            biasOffset:  biasRel,  biasLength:  entry.biasSize,
            shape: entry.shape,
            dtype: entry.dtype)
    }

    // MARK: - Routed expert (lazy)

    /// First touch of layer L opens its backend + verifies SHA-256; subsequent
    /// touches reuse the open backend. The backend resolves the expert to an
    /// cache-slot `(MTLBuffer, offset)` pair.
    public func routedExpert(layer L: Int, expert E: Int) throws -> TensorView {
        try ensureLayerOpened(L)
        let backend = streamersQueue.sync { streamersBox.streamers[L]! }
        let r = try backend.loadExpert(layer: 0, expert: E)
        return TensorView(
            buffer: r.buffer,
            offset: r.offset,
            length: r.size,
            scaleOffset: 0, scaleLength: 0,
            biasOffset:  0, biasLength:  0,
            shape: (UInt32(L), UInt32(E), 0, 0),
            dtype: GTurboFormatV1.DType.u32.rawValue)
    }

    /// Open layer L's file + verify SHA, idempotent.
    func ensureLayerOpened(_ L: Int) throws {
        try streamersQueue.sync {
            try openLayerLocked(L)
        }
    }

    /// Best-effort overlap hook for prefill: starts the same lazy layer open on
    /// the model's streamer queue without waiting for the first expert fetch.
    public func beginOpeningRoutedExpertStreamer(layer L: Int) {
        nonisolated(unsafe) let model = self
        streamersQueue.async {
            try? model.openLayerLocked(L)
        }
    }

    private func openLayerLocked(_ L: Int) throws {
        if streamersBox.streamers[L] != nil {
            return
        }
        let basename = packedExpertsLayout.layers[L].file
        let url = directoryURL
            .appendingPathComponent("packed_experts")
            .appendingPathComponent(basename)
        let manifestRel = "packed_experts/\(basename)"
        let layerFD = try modelDirectory.openFile(manifestRel)
        defer { close(layerFD) }
        if !streamersBox.layerVerified[L] {
            guard let entry = manifest.files[manifestRel] else {
                throw ModelError.missingFile(name: manifestRel)
            }
            let actualSize = try modelDirectory.fileSize(
                fileDescriptor: layerFD, relativePath: manifestRel)
            guard actualSize == entry.size else {
                throw ModelError.tensorSizeMismatch(
                    name: manifestRel, expected: entry.size, actual: actualSize)
            }
            switch integrityPolicy {
            case .fullSha256:
                try Sha256Verifier.verifyFile(fileDescriptor: layerFD,
                                              named: manifestRel,
                                              expectedHex: entry.sha256)
            case .sizeCheckTrustedReceipt:
                break
            }
        }
        let streamSize = UInt64(packedExpertsLayout.expertsPerLayer)
            * packedExpertsLayout.expertStride
        let layout = StreamLayout(
            path: url.path,
            streamOffset: 0,
            streamSize: streamSize,
            expertsPerLayer: packedExpertsLayout.expertsPerLayer,
            expertStride: packedExpertsLayout.expertStride,
            expertOffsets: packedExpertsLayout.layers[L].experts.map(\.offset))
        let slotCount: Int
        switch streamingMode {
        case .pread(let configuredSlotCount):
            slotCount = configuredSlotCount
        }
        streamersBox.streamers[L] = try PreadExpertStreamer(
            layout: layout,
            device: device,
            slotCount: slotCount,
            cachePolicy: expertCachePolicy,
            fileDescriptor: layerFD)
        streamersBox.layerVerified[L] = true
    }

    /// Test hook: how many layer files have been opened so far.
    public func openLayerFileCount() -> Int {
        streamersQueue.sync { streamersBox.streamers.compactMap { $0 }.count }
    }

}

extension Model {

    /// Open a `.gturbo/` directory with the architecture auto-detected from
    /// `manifest.json -> arch.family` (absent means Gemma 4).
    public static func load(directoryURL: URL,
                            device: MTLDevice,
                            streamingMode: ExpertStreamingMode = .pread(slotCount: 16),
                            expertCachePolicy: ExpertCachePolicy = PreadExpertStreamer.cachePolicyDefault,
                            integrityPolicy: ModelIntegrityPolicy? = nil,
                            loadStats: UnsafeMutablePointer<ModelLoadStats>? = nil) throws -> Model {
        let baseline = try ManifestReader.resolveArchitecture(
            directoryURL: directoryURL)
        return try load(directoryURL: directoryURL,
                        device: device,
                        expecting: baseline,
                        streamingMode: streamingMode,
                        expertCachePolicy: expertCachePolicy,
                        integrityPolicy: integrityPolicy,
                        loadStats: loadStats)
    }

    /// Open a `.gturbo/` directory and return a typed handle. Eagerly verifies
    /// SHA-256 of `model_weights.bin` and `packed_experts/layout.json`; layer
    /// files are verified lazily on first `routedExpert(...)` touch.
    public static func load(directoryURL: URL,
                            device: MTLDevice,
                            expecting: ArchConfig,
                            streamingMode: ExpertStreamingMode = .pread(slotCount: 16),
                            expertCachePolicy: ExpertCachePolicy = PreadExpertStreamer.cachePolicyDefault,
                            integrityPolicy: ModelIntegrityPolicy? = nil,
                            loadStats: UnsafeMutablePointer<ModelLoadStats>? = nil) throws -> Model {
        var stats = ModelLoadStats()
        defer {
            loadStats?.pointee = stats
        }
        let resolvedIntegrityPolicy = integrityPolicy ?? .fullSha256
        let modelDirectory = try GTurboModelDirectory(rootURL: directoryURL)
        let manifestFD: Int32
        do { manifestFD = try modelDirectory.openFile("manifest.json") }
        catch ModelError.missingFile { throw ModelError.partialInstall(path: directoryURL.path) }
        defer { close(manifestFD) }
        let manifestData = try modelDirectory.readMetadata(
            fileDescriptor: manifestFD, relativePath: "manifest.json",
            maxBytes: ManifestReader.defaultMaxBytes)
        let manifestSize = UInt64(manifestData.count)
        let manifestShaStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let manifestSha = Sha256Verifier.hashData(manifestData)
        stats.manifestSha256Nanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - manifestShaStart
        let receipt: VerifiedInstallReceipt?
        if resolvedIntegrityPolicy == .sizeCheckTrustedReceipt {
            let receiptStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let receiptFD: Int32
            do {
                receiptFD = try modelDirectory.openFile(VerifiedInstallReceiptReader.fileName)
            } catch ModelError.missingFile {
                throw ModelError.trustedReceiptInvalid(
                    detail: "\(VerifiedInstallReceiptReader.fileName) is missing")
            }
            defer { close(receiptFD) }
            let receiptData = try modelDirectory.readMetadata(
                fileDescriptor: receiptFD,
                relativePath: VerifiedInstallReceiptReader.fileName,
                maxBytes: VerifiedInstallReceiptReader.defaultMaxBytes)
            let loadedReceipt = try VerifiedInstallReceiptReader.decode(data: receiptData)
            try VerifiedInstallReceiptReader.validateManifestBinding(
                loadedReceipt,
                directoryURL: directoryURL,
                manifestSha256: manifestSha)
            stats.receiptValidationNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - receiptStart
            receipt = loadedReceipt
        } else {
            receipt = nil
        }

        let manifest = try ManifestReader.decode(
            data: manifestData, expecting: expecting)
        if let receipt {
            let receiptStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            try VerifiedInstallReceiptReader.validate(receipt,
                                                      directoryURL: directoryURL,
                                                      manifest: manifest,
                                                      manifestSha256: manifestSha,
                                                      manifestSize: manifestSize)
            stats.receiptValidationNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - receiptStart
        }

        // Verify the small, always-touched files before mapping model data.
        let weightsURL = directoryURL.appendingPathComponent("model_weights.bin")
        guard let weightsEntry = manifest.files["model_weights.bin"] else {
            throw ModelError.missingFile(name: "model_weights.bin")
        }
        let weightsFD = try modelDirectory.openFile("model_weights.bin")
        defer { close(weightsFD) }
        let weightsSize = try modelDirectory.fileSize(
            fileDescriptor: weightsFD, relativePath: "model_weights.bin")
        guard weightsSize == weightsEntry.size else {
            throw ModelError.tensorSizeMismatch(
                name: "model_weights.bin",
                expected: weightsEntry.size,
                actual: weightsSize)
        }
        let eagerShaStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        try Sha256Verifier.verifyFile(fileDescriptor: weightsFD,
                                      named: "model_weights.bin",
                                      expectedHex: weightsEntry.sha256)
        let layout: PackedExpertsLayout
        if expecting.feedForwardKind == .dense {
            layout = .dense(numLayers: expecting.numLayers)
        } else {
            guard let layoutEntry = manifest.files["packed_experts/layout.json"] else {
                throw ModelError.missingFile(name: "packed_experts/layout.json")
            }
            let layoutFD = try modelDirectory.openFile("packed_experts/layout.json")
            defer { close(layoutFD) }
            let layoutData = try modelDirectory.readMetadata(
                fileDescriptor: layoutFD, relativePath: "packed_experts/layout.json",
                maxBytes: PackedExpertsLayoutReader.defaultMaxBytes)
            guard UInt64(layoutData.count) == layoutEntry.size else {
                throw ModelError.tensorSizeMismatch(
                    name: "packed_experts/layout.json",
                    expected: layoutEntry.size,
                    actual: UInt64(layoutData.count))
            }
            guard Sha256Verifier.hashData(layoutData).lowercased()
                    == layoutEntry.sha256.lowercased() else {
                throw ModelError.checksumMismatch(file: "packed_experts/layout.json")
            }
            layout = try PackedExpertsLayoutReader.decode(data: layoutData,
                                                           manifest: manifest)
        }
        stats.eagerSha256Nanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - eagerShaStart

        if resolvedIntegrityPolicy == .sizeCheckTrustedReceipt,
           expecting.feedForwardKind == .mixtureOfExperts {
            let receiptStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            try validateTrustedReceiptLayerLayout(modelDirectory: modelDirectory,
                                                  manifest: manifest,
                                                  layout: layout)
            stats.receiptValidationNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - receiptStart
        }

        let residentIndex = try ResidentIndexReader.load(
            fileDescriptor: weightsFD, displayPath: "model_weights.bin")
        try validateRuntimeSchema(residentIndex: residentIndex,
                                  layout: layout,
                                  manifest: manifest,
                                  config: expecting)

        // The resident index must account for the complete weights file.
        let fileSize = weightsSize
        let (expectedSize, overflow) = residentIndex.header.indexSize
            .addingReportingOverflow(residentIndex.header.residentSize)
        if overflow || fileSize != expectedSize {
            throw ModelError.indexCorrupt(detail: """
                model_weights.bin size \(fileSize) != indexSize \
                \(residentIndex.header.indexSize) + residentSize \
                \(residentIndex.header.residentSize) = \(expectedSize)
                """)
        }

        let residentBuffers = try ResidentBufferSet(
            fileURL: weightsURL,
            index: residentIndex,
            device: device,
            fileDescriptor: weightsFD)

        return Model(
            device: device,
            config: expecting,
            streamingMode: streamingMode,
            expertCachePolicy: expertCachePolicy,
            integrityPolicy: resolvedIntegrityPolicy,
            residentBuffers: residentBuffers,
            residentIndex: residentIndex,
            packedExpertsLayout: layout,
            manifest: manifest,
            directoryURL: directoryURL,
            modelDirectory: modelDirectory)
    }

    private static func validateTrustedReceiptLayerLayout(modelDirectory: GTurboModelDirectory,
                                                          manifest: Manifest,
                                                          layout: PackedExpertsLayout) throws {
        for layer in layout.layers {
            let relativePath = "packed_experts/\(layer.file)"
            guard let manifestEntry = manifest.files[relativePath] else {
                throw ModelError.trustedReceiptInvalid(detail: "manifest missing \(relativePath)")
            }
            let actualSize: UInt64
            do {
                let fd = try modelDirectory.openFile(relativePath)
                defer { close(fd) }
                actualSize = try modelDirectory.fileSize(
                    fileDescriptor: fd, relativePath: relativePath)
            }
            guard actualSize == manifestEntry.size else {
                throw ModelError.trustedReceiptInvalid(
                    detail: "\(relativePath) size \(actualSize) != \(manifestEntry.size)")
            }
        }
    }

    static func validateRuntimeSchema(residentIndex: ResidentIndex,
                                      layout: PackedExpertsLayout,
                                      manifest: Manifest,
                                      config: ArchConfig) throws {
        guard let quant = manifest.quant else {
            throw ModelError.indexCorrupt(
                detail: "manifest.quant is required by the executable runtime schema")
        }

        func checkedMultiply(_ lhs: UInt64, _ rhs: UInt64, field: String) throws -> UInt64 {
            let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
            guard !overflow else {
                throw ModelError.indexCorrupt(detail: "\(field) byte count overflows UInt64")
            }
            return value
        }

        func checkedIntMultiply(_ lhs: Int, _ rhs: Int, field: String) throws -> Int {
            let (value, overflow) = lhs.multipliedReportingOverflow(by: rhs)
            guard !overflow else {
                throw ModelError.indexCorrupt(detail: "\(field) dimension overflows Int")
            }
            return value
        }

        func dimensions(_ rows: Int, _ columns: Int, field: String) throws -> (UInt32, UInt32) {
            guard let r = UInt32(exactly: rows), let c = UInt32(exactly: columns),
                  r > 0, c > 0 else {
                throw ModelError.indexCorrupt(detail: "\(field) has invalid dimensions")
            }
            return (r, c)
        }

        func requireBF16(_ name: String, count: Int) throws {
            guard let entry = residentIndex.entries[name] else {
                throw ModelError.indexCorrupt(detail: "missing required resident tensor \(name)")
            }
            guard let logicalCount = UInt32(exactly: count), logicalCount > 0 else {
                throw ModelError.indexCorrupt(detail: "\(name) has invalid dimensions")
            }
            let expectedBytes = try checkedMultiply(
                UInt64(logicalCount), UInt64(MemoryLayout<UInt16>.size), field: name)
            guard entry.dtype == GTurboFormatV1.DType.bf16.rawValue,
                  entry.shape.0 == logicalCount,
                  entry.shape.1 == 0, entry.shape.2 == 0, entry.shape.3 == 0,
                  entry.sizeBytes == expectedBytes,
                  entry.scaleOffset == 0, entry.scaleSize == 0,
                  entry.biasOffset == 0, entry.biasSize == 0,
                  entry.fileOffset % UInt64(MemoryLayout<UInt16>.alignment) == 0 else {
                throw ModelError.indexCorrupt(detail: "\(name) does not match the required BF16 schema")
            }
        }

        func affineSizes(rows: Int,
                         columns: Int,
                         slot: ManifestQuantSlot,
                         field: String) throws -> (shape: (UInt32, UInt32), weight: UInt64, aux: UInt64) {
            let shape = try dimensions(rows, columns, field: field)
            guard slot.weightBits == 4 || slot.weightBits == 8,
                  slot.groupSize > 0,
                  columns % slot.groupSize == 0 else {
                throw ModelError.indexCorrupt(detail: "\(field) has unsupported affine quantization")
            }
            let elements = try checkedMultiply(UInt64(rows), UInt64(columns), field: field)
            let bitCount = try checkedMultiply(elements, UInt64(slot.weightBits), field: field)
            guard bitCount % 8 == 0 else {
                throw ModelError.indexCorrupt(detail: "\(field) packed byte count is fractional")
            }
            let groups = UInt64(columns / slot.groupSize)
            let auxElements = try checkedMultiply(UInt64(shape.0), groups, field: field)
            let auxBytes = try checkedMultiply(
                auxElements, UInt64(MemoryLayout<UInt16>.size), field: field)
            return (shape, bitCount / 8, auxBytes)
        }

        func requireAffine(_ name: String,
                           rows: Int,
                           columns: Int,
                           slot: ManifestQuantSlot) throws {
            guard let entry = residentIndex.entries[name] else {
                throw ModelError.indexCorrupt(detail: "missing required resident tensor \(name)")
            }
            let expected = try affineSizes(
                rows: rows, columns: columns, slot: slot, field: name)
            let primaryAlignment: UInt64 = slot.weightBits == 4
                ? UInt64(MemoryLayout<UInt16>.alignment)
                : 1
            guard entry.dtype == GTurboFormatV1.DType.u32.rawValue,
                  entry.shape.0 == expected.shape.0,
                  entry.shape.1 == expected.shape.1,
                  entry.shape.2 == 0, entry.shape.3 == 0,
                  entry.sizeBytes == expected.weight,
                  entry.scaleSize == expected.aux,
                  entry.biasSize == expected.aux,
                  entry.fileOffset % primaryAlignment == 0,
                  entry.scaleOffset % UInt64(MemoryLayout<UInt16>.alignment) == 0,
                  entry.biasOffset % UInt64(MemoryLayout<UInt16>.alignment) == 0 else {
                throw ModelError.indexCorrupt(
                    detail: "\(name) affine metadata mismatch: dtype=\(entry.dtype), shape=[\(entry.shape.0),\(entry.shape.1),\(entry.shape.2),\(entry.shape.3)], bytes=\(entry.sizeBytes), scales=\(entry.scaleSize), biases=\(entry.biasSize), expected shape=[\(expected.shape.0),\(expected.shape.1),0,0], bytes=\(expected.weight), aux=\(expected.aux)")
            }
        }

        /// BF16 tensor with a non-vector logical shape (Qwen's depthwise conv
        /// kernel is the only one). `requireBF16` insists on `[n, 0, 0, 0]`.
        func requireBF16Shaped(_ name: String, shape: [UInt32]) throws {
            guard let entry = residentIndex.entries[name] else {
                throw ModelError.indexCorrupt(detail: "missing required resident tensor \(name)")
            }
            let padded = shape + Array(repeating: UInt32(0), count: max(0, 4 - shape.count))
            let elements = shape.reduce(UInt64(1)) { $0 * UInt64(max($1, 1)) }
            let expectedBytes = try checkedMultiply(
                elements, UInt64(MemoryLayout<UInt16>.size), field: name)
            guard entry.dtype == GTurboFormatV1.DType.bf16.rawValue,
                  entry.shape.0 == padded[0], entry.shape.1 == padded[1],
                  entry.shape.2 == padded[2], entry.shape.3 == padded[3],
                  entry.sizeBytes == expectedBytes,
                  entry.scaleOffset == 0, entry.scaleSize == 0,
                  entry.biasOffset == 0, entry.biasSize == 0,
                  entry.fileOffset % UInt64(MemoryLayout<UInt16>.alignment) == 0 else {
                throw ModelError.indexCorrupt(
                    detail: "\(name) does not match the required BF16 schema")
            }
        }

        if config.family == .gptOss {
            try requireBF16Shaped(
                "language_model.model.embed_tokens.weight",
                shape: [UInt32(config.vocabSize), UInt32(config.hiddenSize)])
        } else {
            try requireAffine(
                "language_model.model.embed_tokens.weight",
                rows: config.vocabSize,
                columns: config.hiddenSize,
                slot: quant.embedding)
        }
        try requireBF16("language_model.model.norm.weight", count: config.hiddenSize)
        // Gemma ties lm_head to the embedding; an untied family carries its own.
        if !config.tieWordEmbeddings {
            if config.family == .gptOss {
                try requireBF16Shaped(
                    "language_model.lm_head.weight",
                    shape: [UInt32(config.vocabSize), UInt32(config.hiddenSize)])
            } else {
                try requireAffine("language_model.lm_head.weight",
                                  rows: config.vocabSize,
                                  columns: config.hiddenSize,
                                  slot: quant.embedding)
            }
        }

        // Dense Gemma has no router or packed-expert files. Its resident
        // contract also includes the packed PLE tables and per-layer mapping.
        if config.feedForwardKind == .dense {
            try validateDenseGemma4LayerSchema(
                config: config, quant: quant,
                requireBF16: requireBF16,
                requireAffine: requireAffine,
                checkedIntMultiply: checkedIntMultiply)
        } else if config.family == .gptOss {
            for layer in 0..<config.numLayers {
                let prefix = "language_model.model.layers.\(layer)"
                try requireBF16("\(prefix).input_layernorm.weight",
                                count: config.hiddenSize)
                try requireBF16("\(prefix).post_attention_layernorm.weight",
                                count: config.hiddenSize)
                try requireBF16Shaped(
                    "\(prefix).self_attn.q_proj.weight",
                    shape: [UInt32(config.numHeads * config.headDim),
                            UInt32(config.hiddenSize)])
                try requireBF16("\(prefix).self_attn.q_proj.bias",
                                count: config.numHeads * config.headDim)
                for projection in ["k_proj", "v_proj"] {
                    try requireBF16Shaped(
                        "\(prefix).self_attn.\(projection).weight",
                        shape: [UInt32(config.numKVHeads * config.headDim),
                                UInt32(config.hiddenSize)])
                    try requireBF16("\(prefix).self_attn.\(projection).bias",
                                    count: config.numKVHeads * config.headDim)
                }
                try requireBF16Shaped(
                    "\(prefix).self_attn.o_proj.weight",
                    shape: [UInt32(config.hiddenSize),
                            UInt32(config.numHeads * config.headDim)])
                try requireBF16("\(prefix).self_attn.o_proj.bias",
                                count: config.hiddenSize)
                try requireBF16("\(prefix).self_attn.sinks",
                                count: config.numHeads)
                try requireBF16Shaped(
                    "\(prefix).mlp.router.weight",
                    shape: [UInt32(config.numExperts), UInt32(config.hiddenSize)])
                try requireBF16("\(prefix).mlp.router.bias",
                                count: config.numExperts)
            }
        } else if config.family == .qwen36 {
            try validateQwen36LayerSchema(
                config: config, quant: quant,
                requireBF16: requireBF16,
                requireBF16Shaped: requireBF16Shaped,
                requireAffine: requireAffine,
                checkedIntMultiply: checkedIntMultiply)
        } else {
        for layer in 0..<config.numLayers {
            let prefix = "language_model.model.layers.\(layer)"
            let isFull = config.fullAttentionLayerMask[layer] != 0
            let headDimension = isFull ? config.fullHeadDim : config.headDim
            let kvHeads = isFull ? config.numFullKVHeads : config.numKVHeads
            let queryDimension = try checkedIntMultiply(
                config.numHeads, headDimension, field: "layer \(layer) query")
            let kvDimension = try checkedIntMultiply(
                kvHeads, headDimension, field: "layer \(layer) key/value")

            for name in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "pre_feedforward_layernorm_2.weight",
                "post_feedforward_layernorm_1.weight",
                "post_feedforward_layernorm_2.weight",
                "post_feedforward_layernorm.weight",
                "router.scale",
            ] {
                try requireBF16("\(prefix).\(name)", count: config.hiddenSize)
            }
            try requireBF16("\(prefix).self_attn.q_norm.weight", count: headDimension)
            try requireBF16("\(prefix).self_attn.k_norm.weight", count: headDimension)
            try requireBF16("\(prefix).router.per_expert_scale", count: config.numExperts)
            try requireBF16("\(prefix).layer_scalar", count: 1)

            try requireAffine("\(prefix).self_attn.q_proj.weight",
                              rows: queryDimension, columns: config.hiddenSize,
                              slot: quant.attention)
            try requireAffine("\(prefix).self_attn.k_proj.weight",
                              rows: kvDimension, columns: config.hiddenSize,
                              slot: quant.attention)
            if !isFull {
                try requireAffine("\(prefix).self_attn.v_proj.weight",
                                  rows: kvDimension, columns: config.hiddenSize,
                                  slot: quant.attention)
            }
            try requireAffine("\(prefix).self_attn.o_proj.weight",
                              rows: config.hiddenSize, columns: queryDimension,
                              slot: quant.attention)
            let ffn = config.ffnIntermediateSize(layer: layer)
            try requireAffine("\(prefix).mlp.gate_proj.weight",
                              rows: ffn, columns: config.hiddenSize,
                              slot: quant.sharedExpert)
            try requireAffine("\(prefix).mlp.up_proj.weight",
                              rows: ffn, columns: config.hiddenSize,
                              slot: quant.sharedExpert)
            try requireAffine("\(prefix).mlp.down_proj.weight",
                              rows: config.hiddenSize, columns: ffn,
                              slot: quant.sharedExpert)
            try requireAffine("\(prefix).router.proj.weight",
                              rows: config.numExperts, columns: config.hiddenSize,
                              slot: quant.router)
        }
        }

        guard config.feedForwardKind == .mixtureOfExperts else {
            guard layout.layers.isEmpty,
                  layout.expertsPerLayer == 0,
                  layout.expertStride == 0 else {
                throw ModelError.indexCorrupt(
                    detail: "dense architecture must not carry routed experts")
            }
            return
        }

        if config.family == .gptOss {
            try validateGPTOSSRoutedLayout(config: config, layout: layout)
            return
        }

        let routedShapes: [(String, Int, Int)] = [
            ("gate", config.moeIntermediateSize, config.hiddenSize),
            ("up", config.moeIntermediateSize, config.hiddenSize),
            ("down", config.hiddenSize, config.moeIntermediateSize),
        ]
        for layer in layout.layers {
            guard let reference = layer.experts.first else {
                throw ModelError.indexCorrupt(
                    detail: "routed layer \(layer.layer) has no experts")
            }
            for (role, rows, columns) in routedShapes {
                let sizes = try affineSizes(
                    rows: rows, columns: columns,
                    slot: quant.routedExpert,
                    field: "routed layer \(layer.layer) \(role)")
                let expectedRoles: [(String, String, [UInt32], Int?, UInt64, UInt64)] = [
                    (role, "U32", [sizes.shape.0, sizes.shape.1],
                     quant.routedExpert.weightBits, sizes.weight,
                     UInt64(MemoryLayout<UInt32>.alignment)),
                    ("\(role)_scales", "BF16",
                     [sizes.shape.0, UInt32(columns / quant.routedExpert.groupSize)],
                     nil, sizes.aux, UInt64(MemoryLayout<UInt16>.alignment)),
                    ("\(role)_biases", "BF16",
                     [sizes.shape.0, UInt32(columns / quant.routedExpert.groupSize)],
                     nil, sizes.aux, UInt64(MemoryLayout<UInt16>.alignment)),
                ]
                for (name, dtype, shape, bits, size, alignment) in expectedRoles {
                    guard let expected = reference.subTensors[name] else {
                        throw ModelError.indexCorrupt(
                            detail: "routed layer \(layer.layer) is missing role \(name)")
                    }
                    let (end, overflow) = expected.offset.addingReportingOverflow(expected.size)
                    guard expected.dtype == dtype,
                          expected.shape == shape,
                          expected.bits == bits,
                          expected.size == size,
                          expected.offset % alignment == 0,
                          !overflow,
                          end <= reference.size,
                          end <= UInt64(UInt32.max) + 1 else {
                        throw ModelError.indexCorrupt(
                            detail: "routed layer \(layer.layer) role \(name) does not match the required schema")
                    }
                    for expert in layer.experts.dropFirst()
                        where expert.subTensors[name] != expected {
                        throw ModelError.indexCorrupt(
                            detail: "routed layer \(layer.layer) role \(name) metadata differs across experts")
                    }
                }
            }
        }
    }

    private static func validateGPTOSSRoutedLayout(
        config: ArchConfig,
        layout: PackedExpertsLayout
    ) throws {
        let hidden = config.hiddenSize
        let intermediate = config.moeIntermediateSize
        let expected: [(String, String, [UInt32], Int?, UInt64, UInt64)] = [
            ("mlp1", "U8", [UInt32(2 * intermediate), UInt32(hidden)], 4,
             UInt64(2 * intermediate * hidden / 2), 1),
            ("mlp1_scales", "U8",
             [UInt32(2 * intermediate), UInt32(hidden / 32)], nil,
             UInt64(2 * intermediate * hidden / 32), 1),
            ("mlp1_bias", "BF16", [UInt32(2 * intermediate)], nil,
             UInt64(2 * intermediate * MemoryLayout<UInt16>.stride), 2),
            ("mlp2", "U8", [UInt32(hidden), UInt32(intermediate)], 4,
             UInt64(hidden * intermediate / 2), 1),
            ("mlp2_scales", "U8",
             [UInt32(hidden), UInt32(intermediate / 32)], nil,
             UInt64(hidden * intermediate / 32), 1),
            ("mlp2_bias", "BF16", [UInt32(hidden)], nil,
             UInt64(hidden * MemoryLayout<UInt16>.stride), 2),
        ]
        guard layout.numLayers == config.numLayers,
              layout.expertsPerLayer == config.numExperts else {
            throw ModelError.indexCorrupt(
                detail: "GPT-OSS routed layout dimensions do not match the architecture")
        }
        for layer in layout.layers {
            guard let reference = layer.experts.first else {
                throw ModelError.indexCorrupt(
                    detail: "GPT-OSS routed layer \(layer.layer) has no experts")
            }
            for (name, dtype, shape, bits, size, alignment) in expected {
                guard let tensor = reference.subTensors[name] else {
                    throw ModelError.indexCorrupt(
                        detail: "GPT-OSS routed layer \(layer.layer) is missing \(name)")
                }
                let (end, overflow) = tensor.offset.addingReportingOverflow(tensor.size)
                guard tensor.dtype == dtype,
                      tensor.shape == shape,
                      tensor.bits == bits,
                      tensor.size == size,
                      tensor.offset % alignment == 0,
                      !overflow,
                      end <= reference.size else {
                    throw ModelError.indexCorrupt(
                        detail: "GPT-OSS routed layer \(layer.layer) role \(name) does not match the required schema")
                }
                for expert in layer.experts.dropFirst()
                    where expert.subTensors[name] != tensor {
                    throw ModelError.indexCorrupt(
                        detail: "GPT-OSS routed layer \(layer.layer) role \(name) differs across experts")
                }
            }
        }
    }

    private static func validateDenseGemma4LayerSchema(
        config: ArchConfig,
        quant: ManifestQuant,
        requireBF16: (String, Int) throws -> Void,
        requireAffine: (String, Int, Int, ManifestQuantSlot) throws -> Void,
        checkedIntMultiply: (Int, Int, String) throws -> Int
    ) throws {
        if config.hasPerLayerInputs {
            let packedPLE = try checkedIntMultiply(
                config.numLayers, config.hiddenSizePerLayerInput, "packed PLE width")
            try requireAffine("language_model.model.embed_tokens_per_layer.weight",
                              config.vocabSizePerLayerInput, packedPLE, quant.embedding)
            try requireAffine("language_model.model.per_layer_model_projection.weight",
                              packedPLE, config.hiddenSize, quant.sharedExpert)
            try requireBF16("language_model.model.per_layer_projection_norm.weight",
                            config.hiddenSizePerLayerInput)
        }

        for layer in 0..<config.numLayers {
            let prefix = "language_model.model.layers.\(layer)"
            let isFull = config.layerIsFull(layer)
            let headDimension = isFull ? config.fullHeadDim : config.headDim
            let kvHeads = isFull ? config.numFullKVHeads : config.numKVHeads
            let queryDimension = try checkedIntMultiply(
                config.numHeads, headDimension, "layer \(layer) query")
            let kvDimension = try checkedIntMultiply(
                kvHeads, headDimension, "layer \(layer) key/value")

            var normNames = [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "post_feedforward_layernorm.weight",
            ]
            if config.hasPerLayerInputs {
                normNames.append("post_per_layer_input_norm.weight")
            }
            for name in normNames {
                try requireBF16("\(prefix).\(name)", config.hiddenSize)
            }
            try requireBF16("\(prefix).self_attn.q_norm.weight", headDimension)
            try requireBF16("\(prefix).layer_scalar", 1)
            try requireAffine("\(prefix).self_attn.q_proj.weight",
                              queryDimension, config.hiddenSize, quant.attention)

            if !config.layerSharesKV(layer) {
                try requireAffine("\(prefix).self_attn.k_proj.weight",
                                  kvDimension, config.hiddenSize, quant.attention)
                if !isFull || !config.attentionKEqV {
                    try requireAffine("\(prefix).self_attn.v_proj.weight",
                                      kvDimension, config.hiddenSize, quant.attention)
                }
                try requireBF16("\(prefix).self_attn.k_norm.weight", headDimension)
            }
            try requireAffine("\(prefix).self_attn.o_proj.weight",
                              config.hiddenSize, queryDimension, quant.attention)
            let ffn = config.ffnIntermediateSize(layer: layer)
            try requireAffine("\(prefix).mlp.gate_proj.weight",
                              ffn, config.hiddenSize, quant.sharedExpert)
            try requireAffine("\(prefix).mlp.up_proj.weight",
                              ffn, config.hiddenSize, quant.sharedExpert)
            try requireAffine("\(prefix).mlp.down_proj.weight",
                              config.hiddenSize, ffn, quant.sharedExpert)
            if config.hasPerLayerInputs {
                try requireAffine("\(prefix).per_layer_input_gate.weight",
                                  config.hiddenSizePerLayerInput, config.hiddenSize,
                                  quant.sharedExpert)
                try requireAffine("\(prefix).per_layer_projection.weight",
                                  config.hiddenSize, config.hiddenSizePerLayerInput,
                                  quant.sharedExpert)
            }
        }
    }

    /// Per-layer resident contract for Qwen 3.6.
    ///
    /// The layer graph is hybrid, so the required tensors depend on the layer
    /// kind: mask-2 layers carry the gated-DeltaNet block (`linear_attn.*`) and
    /// no KV projections, mask-1 layers carry gated full attention with a
    /// packed `[query ; gate]` q_proj. Neither kind has Gemma's sandwich norms,
    /// router scales, or per-layer scalar.
    private static func validateQwen36LayerSchema(
        config: ArchConfig,
        quant: ManifestQuant,
        requireBF16: (String, Int) throws -> Void,
        requireBF16Shaped: (String, [UInt32]) throws -> Void,
        requireAffine: (String, Int, Int, ManifestQuantSlot) throws -> Void,
        checkedIntMultiply: (Int, Int, String) throws -> Int
    ) throws {
        let linear = config.linearAttention
        for layer in 0..<config.numLayers {
            let prefix = "language_model.model.layers.\(layer)"
            try requireBF16("\(prefix).input_layernorm.weight", config.hiddenSize)
            try requireBF16("\(prefix).post_attention_layernorm.weight", config.hiddenSize)

            try requireAffine("\(prefix).mlp.gate.weight",
                              config.numExperts, config.hiddenSize, quant.router)
            try requireAffine("\(prefix).mlp.shared_expert_gate.weight",
                              1, config.hiddenSize, quant.router)
            try requireAffine("\(prefix).mlp.shared_expert.gate_proj.weight",
                              config.intermediateSize, config.hiddenSize, quant.sharedExpert)
            try requireAffine("\(prefix).mlp.shared_expert.up_proj.weight",
                              config.intermediateSize, config.hiddenSize, quant.sharedExpert)
            try requireAffine("\(prefix).mlp.shared_expert.down_proj.weight",
                              config.hiddenSize, config.intermediateSize, quant.sharedExpert)

            if config.layerIsLinear(layer) {
                try requireAffine("\(prefix).linear_attn.in_proj_qkv.weight",
                                  linear.qkvDim, config.hiddenSize, quant.attention)
                try requireAffine("\(prefix).linear_attn.in_proj_z.weight",
                                  linear.valueDim, config.hiddenSize, quant.attention)
                try requireAffine("\(prefix).linear_attn.in_proj_a.weight",
                                  linear.numVHeads, config.hiddenSize, quant.attention)
                try requireAffine("\(prefix).linear_attn.in_proj_b.weight",
                                  linear.numVHeads, config.hiddenSize, quant.attention)
                try requireAffine("\(prefix).linear_attn.out_proj.weight",
                                  config.hiddenSize, linear.valueDim, quant.attention)
                try requireBF16Shaped(
                    "\(prefix).linear_attn.conv1d.weight",
                    [UInt32(linear.qkvDim), UInt32(linear.convKernelSize), 1])
                try requireBF16("\(prefix).linear_attn.A_log", linear.numVHeads)
                try requireBF16("\(prefix).linear_attn.dt_bias", linear.numVHeads)
                try requireBF16("\(prefix).linear_attn.norm.weight", linear.valueHeadDim)
                continue
            }

            let queryDimension = try checkedIntMultiply(
                config.numHeads, config.fullHeadDim, "layer \(layer) query")
            let kvDimension = try checkedIntMultiply(
                config.numFullKVHeads, config.fullHeadDim, "layer \(layer) key/value")
            // q_proj is packed `[query ; gate]`, so it has twice the query rows.
            let qProjRows = try checkedIntMultiply(2, queryDimension, "layer \(layer) q_proj")
            try requireAffine("\(prefix).self_attn.q_proj.weight",
                              qProjRows, config.hiddenSize, quant.attention)
            try requireAffine("\(prefix).self_attn.k_proj.weight",
                              kvDimension, config.hiddenSize, quant.attention)
            try requireAffine("\(prefix).self_attn.v_proj.weight",
                              kvDimension, config.hiddenSize, quant.attention)
            try requireAffine("\(prefix).self_attn.o_proj.weight",
                              config.hiddenSize, queryDimension, quant.attention)
            try requireBF16("\(prefix).self_attn.q_norm.weight", config.fullHeadDim)
            try requireBF16("\(prefix).self_attn.k_norm.weight", config.fullHeadDim)
        }
    }

    /// Releases routed-expert streamers and their slot scratch before vision
    /// encoding, so the tower's bounded scratch does not stack on top of the
    /// expert cache on an 8 GB machine.
    ///
    /// The caller must serialize this transition against forward execution.
    public func prepareExpertResidencyForVision(
        _ policy: VisionResidencyPolicy,
        gpuDrainNanoseconds: UInt64 = 0
    ) -> VisionExpertResidencyTransition {
        let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        return streamersQueue.sync {
            let releasedLayerCount = streamersBox.streamers.compactMap { $0 }.count
            let releasedBytes = streamersBox.streamers.compactMap { $0 }.reduce(UInt64(0)) {
                $0 + $1.diagnosticSlotScratchBytes
            }
            if policy == .onDemand {
                var released = streamersBox.streamers
                streamersBox.streamers = Array(
                    repeating: nil, count: packedExpertsLayout.numLayers)
                // `layerVerified` deliberately survives the release. Clearing it
                // would make every image turn from the second on re-verify all
                // 30 packed expert files, ~12.9 GB, inside this queue.
                released.removeAll()
            }
            let remaining = streamersBox.streamers.compactMap { $0 }
            return VisionExpertResidencyTransition(
                policy: policy,
                gpuDrainNanoseconds: gpuDrainNanoseconds,
                releasedLayerCount: policy == .onDemand ? releasedLayerCount : 0,
                releasedSlotScratchBytes: policy == .onDemand ? releasedBytes : 0,
                remainingOpenLayerCount: remaining.count,
                remainingSlotScratchBytes: remaining.reduce(UInt64(0)) {
                    $0 + $1.diagnosticSlotScratchBytes
                },
                wallNanoseconds: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start)
        }
    }

}
