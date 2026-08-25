import Foundation
import Metal

/// Model family discriminator. Selects the tensor-name contract, the layer
/// graph shape (norm sandwich vs plain pre-norm), and family-specific kernel
/// behavior. Stored in `manifest.json -> arch.family`; absent means Gemma 4
/// (the format's original architecture).
public enum ModelFamily: String, Sendable, Equatable {
    case gemma4 = "gemma4"
    case qwen36 = "qwen36"
    case gptOss = "gpt-oss"
}

public enum ModelVariant: String, Sendable, Equatable {
    case gemma4_E4B = "gemma4-e4b"
    case gemma4_26B_A4B = "gemma4-26b-a4b"
    case qwen36_35B_A3B = "qwen36-35b-a3b"
    case gptOss_20B = "gpt-oss-20b"

    static func legacyDefault(for family: ModelFamily) -> ModelVariant {
        switch family {
        case .gemma4: return .gemma4_26B_A4B
        case .qwen36: return .qwen36_35B_A3B
        case .gptOss: return .gptOss_20B
        }
    }
}

public enum FeedForwardKind: String, Sendable, Equatable {
    case dense
    case mixtureOfExperts = "moe"
}

/// YaRN RoPE scaling parameters used by GPT-OSS. Frequencies inside the
/// original context are blended between extrapolation and interpolation;
/// positions beyond it keep the same continuous frequency schedule.
public struct YaRNRopeConfig: Sendable, Equatable {
    public let originalContextLength: Int
    public let scalingFactor: Double
    public let betaFast: Double
    public let betaSlow: Double

    public init(originalContextLength: Int,
                scalingFactor: Double,
                betaFast: Double,
                betaSlow: Double) {
        self.originalContextLength = originalContextLength
        self.scalingFactor = scalingFactor
        self.betaFast = betaFast
        self.betaSlow = betaSlow
    }
}

/// Gated-DeltaNet (linear attention) dimensions. Zeroed for architectures
/// without linear-attention layers.
public struct LinearAttentionConfig: Sendable, Equatable {
    public let numKHeads: Int
    public let numVHeads: Int
    public let keyHeadDim: Int
    public let valueHeadDim: Int
    public let convKernelSize: Int

    public init(numKHeads: Int, numVHeads: Int,
                keyHeadDim: Int, valueHeadDim: Int,
                convKernelSize: Int) {
        self.numKHeads = numKHeads
        self.numVHeads = numVHeads
        self.keyHeadDim = keyHeadDim
        self.valueHeadDim = valueHeadDim
        self.convKernelSize = convKernelSize
    }

    public static let none = LinearAttentionConfig(
        numKHeads: 0, numVHeads: 0, keyHeadDim: 0, valueHeadDim: 0,
        convKernelSize: 0)

    /// Fused qkv projection rows: 2 * K-dim + V-dim. Also the depthwise conv
    /// channel count.
    public var qkvDim: Int { 2 * numKHeads * keyHeadDim + numVHeads * valueHeadDim }
    /// Value dim, also the z-gate projection rows and out_proj columns.
    public var valueDim: Int { numVHeads * valueHeadDim }
}

/// Compile-time architecture baseline. `manifest.json -> arch` must match this
/// field-by-field at load time; mismatches throw `ModelError.archMismatch`.
///
/// `fullAttentionLayerMask` values: 0 = sliding-window attention,
/// 1 = full attention, 2 = gated-DeltaNet linear attention.
public struct ArchConfig: Sendable, Equatable {
    public let hiddenSize: Int
    public let intermediateSize: Int          // shared expert FFN (== ffnIntermediate in manifest)
    public let moeIntermediateSize: Int       // per-expert FFN
    public let numHeads: Int
    public let numKVHeads: Int
    public let numFullKVHeads: Int
    public let headDim: Int
    public let fullHeadDim: Int
    public let vocabSize: Int
    public let slidingWindow: Int
    public let finalLogitSoftcap: Double
    public let ropeTheta: Double
    public let fullRopeTheta: Double
    public let partialRotaryFactor: Double
    public let numLayers: Int
    public let numExperts: Int
    public let topKExperts: Int
    public let tieWordEmbeddings: Bool
    public let attentionKEqV: Bool
    public let fullAttentionLayerMask: [UInt8]
    public let hiddenActivation: String

    /// Per-layer embeddings (PLE) used by Gemma 4 E2B/E4B. Zero means the
    /// architecture does not carry the auxiliary per-token residual stream.
    public let hiddenSizePerLayerInput: Int
    public let vocabSizePerLayerInput: Int
    /// Final decoder layers that reuse K/V projections from the most recent
    /// non-sharing layer of the same attention kind.
    public let numKVSharedLayers: Int

    // Family-dependent extensions. Defaults describe Gemma 4 so that legacy
    // manifests (which omit them) validate unchanged.
    public let family: ModelFamily
    /// Checkpoint architecture identity. Unlike `family`, this distinguishes
    /// dense and MoE variants that share tokenizer and prompt behavior.
    public let variant: ModelVariant
    public let feedForwardKind: FeedForwardKind
    /// Full-attention q_proj emits `2 * numHeads * fullHeadDim` rows: per-head
    /// [query ; gate] halves. Attention output is multiplied by sigmoid(gate)
    /// before o_proj.
    public let attnOutputGate: Bool
    /// Softmax scale for full attention. Gemma 4 uses 1.0.
    public let attentionScale: Double
    /// Embedding lookup is multiplied by sqrt(hiddenSize) (Gemma) or not (Qwen).
    public let embeddingScaledBySqrtHidden: Bool
    /// Router has `router.scale` (input multiplier) and `per_expert_scale`
    /// tensors (Gemma). False: plain quantized linear router with renormalized
    /// top-k softmax weights and no auxiliary scale tensors.
    public let routerScaled: Bool
    /// Gemma's dual-branch FFN sandwich: pre/post feedforward norms plus a
    /// per-layer residual scalar. False = plain pre-norm residual block.
    public let ffnSandwichNorms: Bool
    /// Shared expert output is gated by sigmoid(shared_expert_gate(x)) (Qwen).
    public let sharedExpertGated: Bool
    /// Partial RoPE convention. False (Gemma): pairs (i, headDim/2 + i) for
    /// i < rotatedPairs with frequency divisor = headDim. True (Qwen/NeoX
    /// sub-dim): rotation confined to the first `rotaryDim` elements, pairing
    /// (i, rotaryDim/2 + i), frequency divisor = rotaryDim.
    public let ropeNeoxSubdim: Bool
    /// Each query head has a learned scalar softmax logit whose value vector
    /// is zero. GPT-OSS uses this as an attention sink.
    public let attentionSinks: Bool
    /// Optional YaRN frequency scaling. Nil means unscaled RoPE.
    public let yarnRope: YaRNRopeConfig?
    /// Upper clamp used by GPT-OSS capped SwiGLU. Zero means ordinary SwiGLU.
    public let swigluLimit: Double
    /// Gated-DeltaNet dimensions for layers with mask value 2.
    public let linearAttention: LinearAttentionConfig

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        moeIntermediateSize: Int,
        numHeads: Int,
        numKVHeads: Int,
        numFullKVHeads: Int,
        headDim: Int,
        fullHeadDim: Int,
        vocabSize: Int,
        slidingWindow: Int,
        finalLogitSoftcap: Double,
        ropeTheta: Double,
        fullRopeTheta: Double,
        partialRotaryFactor: Double,
        numLayers: Int,
        numExperts: Int,
        topKExperts: Int,
        tieWordEmbeddings: Bool,
        attentionKEqV: Bool,
        fullAttentionLayerMask: [UInt8],
        hiddenActivation: String,
        hiddenSizePerLayerInput: Int = 0,
        vocabSizePerLayerInput: Int = 0,
        numKVSharedLayers: Int = 0,
        family: ModelFamily = .gemma4,
        variant: ModelVariant? = nil,
        feedForwardKind: FeedForwardKind = .mixtureOfExperts,
        attnOutputGate: Bool = false,
        attentionScale: Double = 1.0,
        embeddingScaledBySqrtHidden: Bool = true,
        routerScaled: Bool = true,
        ffnSandwichNorms: Bool = true,
        sharedExpertGated: Bool = false,
        ropeNeoxSubdim: Bool = false,
        attentionSinks: Bool = false,
        yarnRope: YaRNRopeConfig? = nil,
        swigluLimit: Double = 0,
        linearAttention: LinearAttentionConfig = .none
    ) {
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.numFullKVHeads = numFullKVHeads
        self.headDim = headDim
        self.fullHeadDim = fullHeadDim
        self.vocabSize = vocabSize
        self.slidingWindow = slidingWindow
        self.finalLogitSoftcap = finalLogitSoftcap
        self.ropeTheta = ropeTheta
        self.fullRopeTheta = fullRopeTheta
        self.partialRotaryFactor = partialRotaryFactor
        self.numLayers = numLayers
        self.numExperts = numExperts
        self.topKExperts = topKExperts
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionKEqV = attentionKEqV
        self.fullAttentionLayerMask = fullAttentionLayerMask
        self.hiddenActivation = hiddenActivation
        self.hiddenSizePerLayerInput = hiddenSizePerLayerInput
        self.vocabSizePerLayerInput = vocabSizePerLayerInput
        self.numKVSharedLayers = numKVSharedLayers
        self.family = family
        self.variant = variant ?? ModelVariant.legacyDefault(for: family)
        self.feedForwardKind = feedForwardKind
        self.attnOutputGate = attnOutputGate
        self.attentionScale = attentionScale
        self.embeddingScaledBySqrtHidden = embeddingScaledBySqrtHidden
        self.routerScaled = routerScaled
        self.ffnSandwichNorms = ffnSandwichNorms
        self.sharedExpertGated = sharedExpertGated
        self.ropeNeoxSubdim = ropeNeoxSubdim
        self.attentionSinks = attentionSinks
        self.yarnRope = yarnRope
        self.swigluLimit = swigluLimit
        self.linearAttention = linearAttention
    }

    /// Canonical Gemma 4 26B-A4B baseline, checked against the installed
    /// model manifest.
    /// `intermediateSize = 2112` is the shared-expert FFN width (3 × moe).
    public static let gemma4_26B_A4B = ArchConfig(
        hiddenSize: 2816,
        intermediateSize: 2112,
        moeIntermediateSize: 704,
        numHeads: 16,
        numKVHeads: 8,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 512,
        vocabSize: 262144,
        slidingWindow: 1024,
        finalLogitSoftcap: 30.0,
        ropeTheta: 10_000.0,
        fullRopeTheta: 1_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 30,
        numExperts: 128,
        topKExperts: 8,
        tieWordEmbeddings: true,
        attentionKEqV: true,
        fullAttentionLayerMask: Self.gemma4LayerMask(),
        hiddenActivation: "gelu_pytorch_tanh"
    )

    /// Canonical Gemma 4 E4B text architecture. The final 18 layers reuse K/V
    /// projections by attention kind, and every layer consumes a 256-wide PLE
    /// signal alongside the main 2,560-wide residual stream.
    public static let gemma4_E4B = ArchConfig(
        hiddenSize: 2560,
        intermediateSize: 10_240,
        moeIntermediateSize: 0,
        numHeads: 8,
        numKVHeads: 2,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 512,
        vocabSize: 262_144,
        slidingWindow: 512,
        finalLogitSoftcap: 30.0,
        ropeTheta: 10_000.0,
        fullRopeTheta: 1_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 42,
        numExperts: 0,
        topKExperts: 0,
        tieWordEmbeddings: true,
        attentionKEqV: false,
        fullAttentionLayerMask: Self.gemma4E4BLayerMask(),
        hiddenActivation: "gelu_pytorch_tanh",
        hiddenSizePerLayerInput: 256,
        vocabSizePerLayerInput: 262_144,
        numKVSharedLayers: 18,
        variant: .gemma4_E4B,
        feedForwardKind: .dense)

    private static func gemma4LayerMask() -> [UInt8] {
        var mask = [UInt8](repeating: 0, count: 30)
        for i in stride(from: 5, to: 30, by: 6) { mask[i] = 1 }
        return mask
    }

    private static func gemma4E4BLayerMask() -> [UInt8] {
        var mask = [UInt8](repeating: 0, count: 42)
        for i in stride(from: 5, to: 42, by: 6) { mask[i] = 1 }
        return mask
    }

    /// Canonical Qwen3.6-35B-A3B baseline: a 40-layer hybrid of 30
    /// gated-DeltaNet linear-attention layers and 10 full-attention layers
    /// (every 4th layer), 256 routed experts (top-8) plus a sigmoid-gated
    /// shared expert, SwiGLU activations, untied lm_head, no logit softcap.
    ///
    /// The sliding-window slots (`numKVHeads`/`headDim`/`slidingWindow`/
    /// `ropeTheta`) mirror the full-attention values; the architecture has no
    /// sliding-window layers so they are never used to size storage.
    public static let qwen36_35B_A3B = ArchConfig(
        hiddenSize: 2048,
        intermediateSize: 512,
        moeIntermediateSize: 512,
        numHeads: 16,
        numKVHeads: 2,
        numFullKVHeads: 2,
        headDim: 256,
        fullHeadDim: 256,
        vocabSize: 248_320,
        slidingWindow: 0,
        finalLogitSoftcap: 0.0,
        ropeTheta: 10_000_000.0,
        fullRopeTheta: 10_000_000.0,
        partialRotaryFactor: 0.25,
        numLayers: 40,
        numExperts: 256,
        topKExperts: 8,
        tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: Self.qwen36LayerMask(),
        hiddenActivation: "silu",
        family: .qwen36,
        variant: .qwen36_35B_A3B,
        attnOutputGate: true,
        attentionScale: 0.0625,   // 256^-0.5
        embeddingScaledBySqrtHidden: false,
        routerScaled: false,
        ffnSandwichNorms: false,
        sharedExpertGated: true,
        ropeNeoxSubdim: true,
        linearAttention: LinearAttentionConfig(
            numKHeads: 16, numVHeads: 32,
            keyHeadDim: 128, valueHeadDim: 128,
            convKernelSize: 4)
    )

    private static func qwen36LayerMask() -> [UInt8] {
        // Layer kinds: 2 = gated-DeltaNet linear, 1 = full attention on every
        // 4th layer ((i + 1) % 4 == 0).
        var mask = [UInt8](repeating: 2, count: 40)
        for i in stride(from: 3, to: 40, by: 4) { mask[i] = 1 }
        return mask
    }

    /// Canonical GPT-OSS 20B profile from the pinned OpenAI checkpoint. The
    /// model alternates 128-token sliding and full attention, routes each
    /// token through four of 32 experts, and stores routed matrices as MXFP4.
    public static let gptOss_20B = ArchConfig(
        hiddenSize: 2_880,
        intermediateSize: 2_880,
        moeIntermediateSize: 2_880,
        numHeads: 64,
        numKVHeads: 8,
        numFullKVHeads: 8,
        headDim: 64,
        fullHeadDim: 64,
        vocabSize: 201_088,
        slidingWindow: 128,
        finalLogitSoftcap: 0,
        ropeTheta: 150_000,
        fullRopeTheta: 150_000,
        partialRotaryFactor: 1,
        numLayers: 24,
        numExperts: 32,
        topKExperts: 4,
        tieWordEmbeddings: false,
        attentionKEqV: false,
        fullAttentionLayerMask: Self.gptOss20BLayerMask(),
        hiddenActivation: "swiglu_capped",
        family: .gptOss,
        variant: .gptOss_20B,
        attentionScale: 0.125,
        embeddingScaledBySqrtHidden: false,
        routerScaled: false,
        ffnSandwichNorms: false,
        sharedExpertGated: false,
        ropeNeoxSubdim: true,
        attentionSinks: true,
        yarnRope: YaRNRopeConfig(
            originalContextLength: 4_096,
            scalingFactor: 32,
            betaFast: 32,
            betaSlow: 1),
        swigluLimit: 7)

    private static func gptOss20BLayerMask() -> [UInt8] {
        (0..<24).map { $0.isMultiple(of: 2) ? 0 : 1 }
    }

    /// Variant registry used by both v1.1 manifests and the legacy-family
    /// resolver. Family alone is intentionally not a registry key because
    /// multiple checkpoints can share prompt behavior without sharing shapes.
    public static let registeredArchitectures: [ModelVariant: ArchConfig] = [
        .gemma4_E4B: .gemma4_E4B,
        .gemma4_26B_A4B: .gemma4_26B_A4B,
        .qwen36_35B_A3B: .qwen36_35B_A3B,
        .gptOss_20B: .gptOss_20B,
    ]

    /// Resident INT4 GEMV shapes this architecture issues during decode, for
    /// pipeline specialization. Constant-folding the loop bounds measurably
    /// raises achieved bandwidth on the narrower projections.
    public var decodeInt4GEMVShapes: [(m: Int, n: Int)] {
        if family == .gptOss { return [] }
        var shapes: [(m: Int, n: Int)] = []
        if attnOutputGate {
            shapes.append((m: 2 * numHeads * fullHeadDim, n: hiddenSize))
        } else {
            shapes.append((m: numHeads * fullHeadDim, n: hiddenSize))
        }
        shapes.append((m: numFullKVHeads * fullHeadDim, n: hiddenSize))
        shapes.append((m: hiddenSize, n: numHeads * fullHeadDim))
        if hasLinearAttentionLayers {
            let la = linearAttention
            shapes.append((m: la.qkvDim, n: hiddenSize))
            shapes.append((m: la.valueDim, n: hiddenSize))
            shapes.append((m: hiddenSize, n: la.valueDim))
        }
        shapes.append((m: intermediateSize, n: hiddenSize))
        shapes.append((m: hiddenSize, n: intermediateSize))
        if hiddenSizePerLayerInput > 0 {
            shapes.append((m: numLayers * hiddenSizePerLayerInput, n: hiddenSize))
            shapes.append((m: hiddenSizePerLayerInput, n: hiddenSize))
            shapes.append((m: hiddenSize, n: hiddenSizePerLayerInput))
        }
        return shapes
    }

    /// Resident INT8 GEMV shapes issued during decode (router and, when the
    /// architecture has one, the shared-expert scalar gate).
    public var decodeInt8GEMVShapes: [(m: Int, n: Int)] {
        var shapes: [(m: Int, n: Int)] = []
        if feedForwardKind == .mixtureOfExperts {
            shapes.append((m: numExperts, n: hiddenSize))
        }
        if sharedExpertGated { shapes.append((m: 1, n: hiddenSize)) }
        return shapes
    }

    /// Layer kind helpers over the mask encoding.
    public func layerIsFull(_ layer: Int) -> Bool { fullAttentionLayerMask[layer] == 1 }
    public func layerIsLinear(_ layer: Int) -> Bool { fullAttentionLayerMask[layer] == 2 }
    public var hasLinearAttentionLayers: Bool { fullAttentionLayerMask.contains(2) }
    public var hasPerLayerInputs: Bool { hiddenSizePerLayerInput > 0 }
    public var firstKVSharedLayer: Int { numLayers - numKVSharedLayers }
    public func layerSharesKV(_ layer: Int) -> Bool {
        numKVSharedLayers > 0 && layer >= firstKVSharedLayer
    }
    public func kvSourceLayer(for layer: Int) -> Int? {
        guard layerSharesKV(layer) else { return nil }
        let kind = fullAttentionLayerMask[layer]
        return stride(from: firstKVSharedLayer - 1, through: 0, by: -1)
            .first { fullAttentionLayerMask[$0] == kind }
    }
}

/// Failure modes for the validation gates in `Model.load`.
enum ModelError: Error, CustomStringConvertible, Equatable {
    case partialInstall(path: String)
    case notAGTurboDirectory
    case unsupportedVersion(major: Int, minor: Int)
    case unknownFlag(name: String)
    case archMismatch(field: String, expected: String, actual: String)
    case expertStrideNotPageAligned(stride: UInt64, pageSize: Int)
    case missingFile(name: String)
    case checksumMismatch(file: String)
    case tensorNotFound(name: String)
    case tensorSizeMismatch(name: String, expected: UInt64, actual: UInt64)
    case residentBufferWrapFailed
    case indexCorrupt(detail: String)
    case posixFailed(call: String, errno: Int32)
    case trustedReceiptInvalid(detail: String)

    public var description: String {
        switch self {
        case .partialInstall(let p):
            return "model.gturbo directory at \(p) is missing manifest.json"
        case .notAGTurboDirectory:
            return "manifest.json magic does not equal \"GTURBO\""
        case .unsupportedVersion(let maj, let min):
            return "manifest version \(maj).\(min) is not supported (need 1.x)"
        case .unknownFlag(let n):
            return "manifest.flags contains unknown key \"\(n)\""
        case .archMismatch(let field, let exp, let act):
            return "manifest.arch.\(field) = \(act); expected \(exp)"
        case .expertStrideNotPageAligned(let s, let p):
            return "expertStride \(s) is not a multiple of page size \(p)"
        case .missingFile(let n):
            return "model.gturbo is missing required file \(n)"
        case .checksumMismatch(let f):
            return "SHA-256 of \(f) does not match manifest.files[\(f)].sha256"
        case .tensorNotFound(let n):
            return "no IndexEntry named \(n) in model_weights.bin"
        case .tensorSizeMismatch(let n, let e, let a):
            return "tensor \(n) size \(a) does not match expected \(e)"
        case .residentBufferWrapFailed:
            return "MTLDevice.makeBuffer(bytesNoCopy:...) returned nil"
        case .indexCorrupt(let d):
            return "resident index is corrupt: \(d)"
        case .posixFailed(let c, let e):
            return "\(c) failed with errno \(e)"
        case .trustedReceiptInvalid(let detail):
            return "trusted install receipt invalid: \(detail)"
        }
    }
}

/// View into a tensor that lives inside one of the loader's resident or
/// streamed `MTLBuffer`s. No `MTLBuffer` is allocated per tensor — the
/// `buffer` reference is shared across many `TensorView` instances and
/// addressed by byte offsets.
public struct TensorView: @unchecked Sendable {
    public let buffer: MTLBuffer
    public let offset: UInt64
    public let length: UInt64
    public let scaleOffset: UInt64
    public let scaleLength: UInt64
    public let biasOffset: UInt64
    public let biasLength: UInt64
    public let shape: (UInt32, UInt32, UInt32, UInt32)
    /// Dtype byte. 0 = U32, 1 = BF16, 2 = FP16, 3 = FP32.
    public let dtype: UInt8

    public init(buffer: MTLBuffer,
                offset: UInt64, length: UInt64,
                scaleOffset: UInt64, scaleLength: UInt64,
                biasOffset: UInt64, biasLength: UInt64,
                shape: (UInt32, UInt32, UInt32, UInt32),
                dtype: UInt8) {
        self.buffer = buffer
        self.offset = offset
        self.length = length
        self.scaleOffset = scaleOffset
        self.scaleLength = scaleLength
        self.biasOffset = biasOffset
        self.biasLength = biasLength
        self.shape = shape
        self.dtype = dtype
    }
}
