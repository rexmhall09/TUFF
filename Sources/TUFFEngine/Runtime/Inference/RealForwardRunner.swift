import Foundation
import Metal

public enum RDAdvicePolicyMode: String, Codable, Sendable, Equatable {
    case `default`
    case off
    case bounded
    case adaptive

    public static func parse(_ raw: String?) -> RDAdvicePolicyMode {
        switch raw?.lowercased() {
        case "off", "none", "disabled":
            return .off
        case "bounded":
            return .bounded
        case "adaptive":
            return .adaptive
        default:
            return .default
        }
    }
}

public struct RDAdviceAdaptivePolicyConfig: Sendable, Equatable {
    public var missCap: Int
    public var byteCap: UInt64
    public var slowCallNanos: UInt64

    public init(missCap: Int,
                byteCap: UInt64,
                slowCallNanos: UInt64) {
        self.missCap = missCap
        self.byteCap = byteCap
        self.slowCallNanos = slowCallNanos
    }

    public static let conservative = RDAdviceAdaptivePolicyConfig(
        missCap: 12,
        byteCap: 384 * 1_048_576,
        slowCallNanos: 1_000_000)
}

struct RDAdviceAdaptivePolicyState: Sendable, Equatable {
    var config: RDAdviceAdaptivePolicyConfig
    private var skipUntilPosition: Int = -1
    private(set) var recentSlowCallNanos: UInt64 = 0

    init(config: RDAdviceAdaptivePolicyConfig = .conservative) {
        self.config = config
    }

    mutating func reset() {
        skipUntilPosition = -1
        recentSlowCallNanos = 0
    }

    func shouldSkip(position: Int,
                    requestedMisses: Int,
                    estimatedBytes: UInt64,
                    canOverlapUsefulGPUWork: Bool) -> Bool {
        position <= skipUntilPosition ||
        !canOverlapUsefulGPUWork ||
        requestedMisses > config.missCap ||
        estimatedBytes > config.byteCap
    }

    mutating func update(after result: ExpertIOAdviceResult,
                                position: Int) {
        recentSlowCallNanos = max(recentSlowCallNanos, result.maxCallNanos)
        if result.maxCallNanos >= config.slowCallNanos {
            skipUntilPosition = max(skipUntilPosition, position)
        }
    }
}

/// Gemma 4 real-forward decode pass.
///
/// Composes the production kernels against the `.gturbo` model:
///
///   embed_lookup_int4(token) * sqrt(H)
///   for L in 0..<30:
///     a = rmsnorm_bf16w(h, input_layernorm)
///     Q = q_proj(a)    K = k_proj(a)    V = (SWA) v_proj(a) | (full) k_proj(a)
///     per-head q/k_norm (bf16w), per-head v_norm (no_scale)
///     NeoX RoPE on Q + K (default for SWA, proportional for full)
///     write K and V into separate cache slots
///     attn = attention(scale=1.0, SWA window or full causal)
///     attn = o_proj(attn)
///     h = h + rmsnorm_bf16w(attn, post_attention_layernorm)
///     h1 = rmsnorm_bf16w(h, pre_feedforward_layernorm)
///     h1 = SharedExpertInt8(h1)
///     h1 = rmsnorm_bf16w(h1, post_feedforward_layernorm_1)
///     // router + routed branch
///     xr   = rmsnorm_no_scale(h)
///     idx, w = router_topk_gemma4(xr, effective_scale[L], per_expert_scale[L])
///     h2 = rmsnorm_bf16w(h, pre_feedforward_layernorm_2)
///     h2 = moe_fused_ffn_streamed_routed(h2, residual=0, routedBlobs=fetch(idx), w)
///     h2 = rmsnorm_bf16w(h2, post_feedforward_layernorm_2)
///     h = h + rmsnorm_bf16w(h1 + h2, post_feedforward_layernorm)
///     h = h * layer_scalar[L]
///   logits = DequantInt4GEMV(rmsnorm_bf16w(h, model.norm), embed_table^T)
///   // final softcap and softmax happen in the Sampler.
///
/// Direct against `Model`; this is the only production decode forward path.
internal enum PrefillProjectionFamily: Sendable, Equatable {
    case q
    case kv
    case o
    case shared
    case routed
}

internal enum PrefillProjectionDispatch: Sendable, Equatable {
    case repeatedGEMV
    case qmm
}

internal enum PrefillProjectionDispatchPolicy {
    static func selectedDispatch(for family: PrefillProjectionFamily,
                                 chunkTokens: Int) -> PrefillProjectionDispatch {
        guard chunkTokens >= 32 else {
            return .repeatedGEMV
        }
        switch family {
        case .q:
            return .repeatedGEMV
        case .kv, .o, .shared, .routed:
            return .qmm
        }
    }
}

public final class RealForwardRunner: ChunkedPrefillRunner, MultimodalPrefillRunner, ContextWindowReporting, ContinuableLogitProducer, SpeculativeVerificationRunner, @unchecked Sendable {
    private struct LayerSharedExpertProjections {
        let gate: SharedExpertInt8Proj
        let up: SharedExpertInt8Proj
        let down: SharedExpertInt8Proj
        /// Gemma-only post_feedforward_layernorm_1; nil when the arch has no
        /// FFN sandwich norms.
        let postF1: TensorView?
        /// Qwen-only [1, hidden] scalar gate on the shared expert branch.
        let scalarGate: TensorView?
    }

    private let model: Model
    private let ctx: MetalContext
    private let kv: KVCacheManager?
    private let cfg: ArchConfig

    // Kernels
    private let embedInt4: EmbedLookupInt4
    private let rms: RMSNorm
    private let int4: DequantInt4GEMV
    private let attention: Attention
    private let shared: SharedExpertRuntime
    private let moe: MoE?
    private let fusionHead: LMHeadChainInt4
    private let fusedQKVGEMV: FusedQKVGEMV
    private let fusedQKVEpilogue: FusedQKVEpilogue
    private let fusedPostAttentionSetup: FusedPostAttentionSetup
    private let fusedTail: FusedLayerTail

    // Qwen 3.6 kernels. Nil on architectures that never dispatch them.
    private let elementwise: Elementwise?
    private let gdn: GDN?
    private let gdnState: GDNStateManager?
    private let rope: RoPE?
    private let perLayerEmbeddingKernel: PerLayerEmbedding?
    private let int8ScalarGate: DequantInt8GEMV?
    /// Difference between Qwen's logical multimodal RoPE position and the
    /// physical KV index. It remains zero for text-only and every Gemma run.
    private var qwenMultimodalRopeDelta: Int32 = 0

    // Prefill kernels. These are initialized once per runner so the chunk path
    // cannot accidentally rebuild PSOs inside a per-layer loop.
    private let prefillEmbed: PrefillEmbedLookupInt4
    private let prefillRMS: PrefillRMSNorm
    private let prefillQMM: PrefillInt4QMM
    private let prefillMPPAffineInt4: MPPPrefillInt4QMM?
    private let prefillQKVEpilogue: PrefillQKVEpilogue
    private let prefillAttention: PrefillAttention
    private let prefillPostAttention: PrefillPostAttentionSetup
    private let prefillRouter: PrefillRouter
    private let prefillSharedExpert: PrefillSharedExpert
    private let prefillGroupedMoE: PrefillGroupedRoutedMoE
    private let prefillMoE: PrefillMoE
    private let prefillLayerTail: PrefillLayerTail
    private let prefillFinalRowHead: PrefillFinalRowHeadInt4

    // Scratch — preallocated per spec'd D / F / vocab.
    private let hidden: MTLBuffer        // [D] FP16
    private let normed: MTLBuffer        // [D] FP16
    private let attnOut: MTLBuffer       // [N_HEADS * head_dim] FP16
    private let qScratch: MTLBuffer      // [N_HEADS * head_dim] FP16
    private let kStage: MTLBuffer        // [max KV heads * head_dim] FP16, current token
    private let vStage: MTLBuffer        // [max KV heads * head_dim] FP16, current token
    private let oOut: MTLBuffer          // [D] FP16
    private let h1Buf: MTLBuffer         // [D] FP16 (dense MLP output)
    private let h2Buf: MTLBuffer         // [D] FP16 (routed output)
    private let routedX: MTLBuffer       // [D] FP16 (pre_feedforward_layernorm_2 output)
    private let denseX: MTLBuffer        // [D] FP16 (pre_feedforward_layernorm output)
    private let denseScratchGate: MTLBuffer // [F=2112] FP16
    private let denseScratchUp: MTLBuffer   // [F=2112] FP16
    private let denseScratchAct: MTLBuffer  // [F=2112] FP16
    private let routerInput: MTLBuffer   // [D] FP16 (rmsnorm_no_scale(h))
    private let zeroResidual: MTLBuffer  // [D] FP16 zeros — for routed branch base
    private let outIndices: MTLBuffer    // [topK] UInt32
    private let outWeights: MTLBuffer    // [topK] FP16
    // Persistent MoE scratch, allocated once; about 56 KiB at production shape.
    private let moeActs: MTLBuffer       // [topK * FmoE] FP16
    private let moeHitActiveSlots: MTLBuffer // [topK] UInt32
    private let moeMissActiveSlots: MTLBuffer // [topK] UInt32
    private let greedyTokenBuf: MTLBuffer // 4 B UInt32 fused-head output
    private let speculativeTokenBuffer: MTLBuffer // bounded candidate IDs
    private let speculativeTargetTokenBuf: MTLBuffer // one argmax per candidate row
    // Qwen 3.6 decode scratch (nil on architectures that never use it).
    private let qPackedScratch: MTLBuffer?   // [2 * N_HEADS * head_dim] packed [q ; gate]
    private let attnGateScratch: MTLBuffer?  // [N_HEADS * head_dim]
    private let gdnQKVRaw: MTLBuffer?        // [qkvDim] raw in_proj_qkv output
    private let gdnConvOut: MTLBuffer?       // [qkvDim] conv + SiLU output
    private let gdnZ: MTLBuffer?             // [valueDim]
    private let gdnA: MTLBuffer?             // [numVHeads]
    private let gdnB: MTLBuffer?             // [numVHeads]
    private let gdnY: MTLBuffer?             // [valueDim] delta-rule output
    private let gdnOut: MTLBuffer?           // [valueDim] gated-norm output
    private let sharedScalarGateBuf: MTLBuffer? // [1] shared-expert gate logit
    private let pleIdentity: MTLBuffer?       // [numLayers * P] token PLE lookup
    private let pleContext: MTLBuffer?        // [numLayers * P] context projection
    private let pleLayer: MTLBuffer?          // [P] selected, combined PLE row
    private let pleGate: MTLBuffer?           // [P] mapping gate / activation
    /// BF16 ones over [numExperts]; neutral per_expert_scale when the router
    /// has no auxiliary scale tensors.
    private let onesPerExpertScale: MTLBuffer?
    private var prefillChunkState = PrefillChunkCommitState()
    private var prefillScratch: PrefillChunkScratchBuffers?
    private var speculativeStartPosition: Int?
    private var speculativeProcessedTokens = 0
    private var collectingSpeculativeMetrics = false
    private var speculativeExpertReads: UInt64 = 0
    private var speculativeExpertBytes: UInt64 = 0
    private var speculativeExpertCacheHits: UInt64 = 0
    private var speculativeExpertCacheMisses: UInt64 = 0
    private var speculativeTargetCommandBuffers: UInt64 = 0

    private static let rdadviseBoundedMissCap = 12
    private static let rdadviseBoundedMaxCallNanos: UInt64 = 250_000
    private static let rdadviseAdaptiveMissCap = 12
    private static let rdadviseAdaptiveByteCap: UInt64 = 384 * 1_048_576
    private static let rdadviseAdaptiveSlowCallNanos: UInt64 = 1_000_000
    private static let prefillRoutedTileSchedulerConfig = PrefillRoutedTileSchedulerConfig()

    /// Per-layer `router.scale * D^-0.5` pre-folded into one BF16 buffer
    /// allocation per layer. ~168 KB total at 30 layers × 2816 BF16 — bounded
    /// host work done once at init.
    private let effectiveScaleBuffers: [MTLBuffer]
    private let sharedExpertProjections: [LayerSharedExpertProjections?]

    public let maxContext: Int

    /// Per-instance head and RDADVISE modes. The fused head (default) skips the
    /// 512 KB logits write and leaves a greedy argmax in `lastGreedyToken`;
    /// callers that sample from the logits buffer (non-greedy configs) must pass
    /// `forceLogitsHead: true` or they read a never-written buffer.
    private let useFusedGreedyHead: Bool
    private let prefillAttentionPath: RuntimePrefillAttentionPath
    public let rdadviseEnabled: Bool
    public let rdadvisePolicyMode: RDAdvicePolicyMode
    private var rdadviseSkipUntilPosition: Int = -1
    private var rdadviseAdaptiveState: RDAdviceAdaptivePolicyState
    private var rdadviseAdaptivePosition: Int = -1
    private var rdadviseAdaptivePositionBytes: UInt64 = 0
    public init(model: Model, context: MetalContext, maxContext: Int,
                runtimeConfiguration: RuntimeConfiguration = .production) throws {
        let config = model.config
        if config.feedForwardKind == .mixtureOfExperts,
           let effectiveSlots = model.routedExpertCacheSlotCount(layer: 0),
           effectiveSlots < config.topKExperts {
            throw RuntimeConfigurationError.expertCacheTooSmall(
                configured: effectiveSlots,
                required: config.topKExperts)
        }
        self.model = model
        self.ctx = context
        self.cfg = config
        self.maxContext = maxContext
        self.useFusedGreedyHead = runtimeConfiguration.headPath == .fusedRows
        self.prefillAttentionPath = runtimeConfiguration.prefillAttentionPath
        let useFP16Ring = runtimeConfiguration.fp16RingEnabled
        self.rdadvisePolicyMode = runtimeConfiguration.rdadvisePolicy
        self.rdadviseAdaptiveState = RDAdviceAdaptivePolicyState(
            config: RDAdviceAdaptivePolicyConfig(
                missCap: Self.rdadviseAdaptiveMissCap,
                byteCap: Self.rdadviseAdaptiveByteCap,
                slowCallNanos: Self.rdadviseAdaptiveSlowCallNanos))
        self.rdadviseEnabled = runtimeConfiguration.rdadviseEnabled
        let maximumVisionTokens = (cfg.family == .gemma4 || cfg.family == .qwen36)
            ? VisionConfig(family: cfg.family).maximumPooledTokens : 0
        self.kv = try KVCacheManager(device: context.device,
                                     config: cfg,
                                     maxContext: maxContext,
                                     fp16RingEnabled: useFP16Ring,
                                     slidingWindow: cfg.slidingWindow,
                                     maxPrefillChunkTokens: max(
                                        PrefillRuntimeConfig.maxChunkTokens,
                                        min(maxContext, maximumVisionTokens)))

        let silu = cfg.hiddenActivation == "silu"
        self.embedInt4 = try EmbedLookupInt4(context: context)
        self.rms       = try RMSNorm(context: context)
        self.int4      = try DequantInt4GEMV(
            context: context,
            additionalShapes: cfg.decodeInt4GEMVShapes)
        self.attention = try Attention(context: context)
        self.shared    = try SharedExpertRuntime(context: context,
                                                  weightBits: model.sharedExpertWeightBits,
                                                  siluActivation: silu)
        self.moe       = cfg.feedForwardKind == .mixtureOfExperts
            ? try MoE(context: context,
                      siluActivation: silu,
                      specializedD: UInt32(cfg.hiddenSize),
                      specializedF: UInt32(cfg.moeIntermediateSize),
                      specializedNumExperts: UInt32(cfg.numExperts))
            : nil
        self.fusionHead = try LMHeadChainInt4(context: context,
                                              maxD: cfg.hiddenSize,
                                              maxVocab: cfg.vocabSize)
        self.fusedQKVGEMV = try FusedQKVGEMV(context: context)
        self.fusedQKVEpilogue = try FusedQKVEpilogue(context: context)
        self.fusedPostAttentionSetup = try FusedPostAttentionSetup(context: context)
        self.fusedTail = try FusedLayerTail(context: context)
        self.prefillEmbed = try PrefillEmbedLookupInt4(context: context)
        self.prefillRMS = try PrefillRMSNorm(context: context)
        self.prefillQMM = try PrefillInt4QMM(context: context)
        self.prefillMPPAffineInt4 = MPPPrefillInt4QMM(context: context)
        self.prefillQKVEpilogue = try PrefillQKVEpilogue(context: context)
        self.prefillAttention = try PrefillAttention(context: context)
        self.prefillPostAttention = try PrefillPostAttentionSetup(context: context)
        self.prefillRouter = try PrefillRouter(context: context)
        self.prefillSharedExpert = try PrefillSharedExpert(
            context: context,
            weightBits: model.sharedExpertWeightBits,
            siluActivation: silu)
        self.prefillGroupedMoE = try PrefillGroupedRoutedMoE(context: context,
                                                             siluActivation: silu)
        self.prefillMoE = try PrefillMoE(context: context)
        self.prefillLayerTail = try PrefillLayerTail(context: context)
        self.prefillFinalRowHead = try PrefillFinalRowHeadInt4(context: context,
                                                               maxD: cfg.hiddenSize)

        // Qwen 3.6 kernels, keyed off the data flags so architectures that
        // never dispatch them pay no PSO compile cost.
        let needsElementwise = cfg.attnOutputGate
            || cfg.sharedExpertGated
            || !cfg.ffnSandwichNorms
            || cfg.hasLinearAttentionLayers
            || cfg.feedForwardKind == .dense
            || cfg.hasPerLayerInputs
        self.elementwise = needsElementwise ? try Elementwise(context: context) : nil
        if cfg.hasLinearAttentionLayers {
            self.gdn = try GDN(context: context, config: cfg.linearAttention,
                               specializedHiddenSize: cfg.hiddenSize)
            self.gdnState = try GDNStateManager(device: context.device, config: cfg)
        } else {
            self.gdn = nil
            self.gdnState = nil
        }
        self.rope = (cfg.ropeNeoxSubdim || cfg.numKVSharedLayers > 0)
            ? try RoPE(context: context) : nil
        self.perLayerEmbeddingKernel = cfg.hasPerLayerInputs
            ? try PerLayerEmbedding(context: context) : nil
        self.int8ScalarGate = cfg.sharedExpertGated
            ? try DequantInt8GEMV(context: context,
                                  additionalShapes: cfg.decodeInt8GEMVShapes)
            : nil

        let device = context.device
        let D = cfg.hiddenSize
        // Scratch is shared across layers, so it is sized for the widest
        // feed-forward block rather than the nominal one. E2B's last 20 layers
        // are twice as wide as its first 15.
        let F = cfg.maxFFNIntermediateSize
        let maxQ = cfg.numHeads * max(cfg.headDim, cfg.fullHeadDim)

        func buf(_ count: Int, _ stride: Int = MemoryLayout<Float16>.size) throws -> MTLBuffer {
            guard let b = device.makeBuffer(length: max(count, 1) * stride,
                                            options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            return b
        }
        self.hidden        = try buf(D)
        self.normed        = try buf(D)
        self.attnOut       = try buf(maxQ)
        self.qScratch      = try buf(maxQ)
        self.kStage        = try buf(max(cfg.numKVHeads * cfg.headDim,
                                         cfg.numFullKVHeads * cfg.fullHeadDim))
        self.vStage        = try buf(max(cfg.numKVHeads * cfg.headDim,
                                         cfg.numFullKVHeads * cfg.fullHeadDim))
        self.oOut          = try buf(D)
        self.h1Buf         = try buf(D)
        self.h2Buf         = try buf(D)
        self.routedX       = try buf(D)
        self.denseX        = try buf(D)
        self.denseScratchGate = try buf(F)
        self.denseScratchUp   = try buf(F)
        self.denseScratchAct  = try buf(F)
        self.routerInput   = try buf(D)
        self.zeroResidual  = try buf(D)
        // The routed MoE kernel seeds y[d] = residual[d]; pinning this buffer
        // to zero once at init makes the routed branch's residual contribution
        // exactly zero (it's combined with the dense MLP downstream).
        memset(self.zeroResidual.contents(), 0, self.zeroResidual.length)
        self.outIndices    = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size)
        self.outWeights    = try buf(cfg.topKExperts)
        self.moeActs       = try buf(cfg.topKExperts * cfg.moeIntermediateSize)
        self.moeHitActiveSlots = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size)
        self.moeMissActiveSlots = try buf(cfg.topKExperts, MemoryLayout<UInt32>.size)
        guard let tok = device.makeBuffer(length: MemoryLayout<UInt32>.size,
                                          options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        self.greedyTokenBuf = tok
        self.speculativeTokenBuffer = try buf(8, MemoryLayout<UInt32>.size)
        self.speculativeTargetTokenBuf = try buf(8, MemoryLayout<UInt32>.size)

        // Qwen 3.6 decode scratch — allocated once here, never in the hot path.
        if cfg.attnOutputGate {
            self.qPackedScratch = try buf(2 * maxQ)
            self.attnGateScratch = try buf(maxQ)
        } else {
            self.qPackedScratch = nil
            self.attnGateScratch = nil
        }
        if cfg.hasLinearAttentionLayers {
            let la = cfg.linearAttention
            self.gdnQKVRaw = try buf(la.qkvDim)
            self.gdnConvOut = try buf(la.qkvDim)
            self.gdnZ = try buf(la.valueDim)
            self.gdnA = try buf(la.numVHeads)
            self.gdnB = try buf(la.numVHeads)
            self.gdnY = try buf(la.valueDim)
            self.gdnOut = try buf(la.valueDim)
        } else {
            self.gdnQKVRaw = nil
            self.gdnConvOut = nil
            self.gdnZ = nil
            self.gdnA = nil
            self.gdnB = nil
            self.gdnY = nil
            self.gdnOut = nil
        }
        self.sharedScalarGateBuf = cfg.sharedExpertGated ? try buf(1) : nil
        if cfg.hasPerLayerInputs {
            let packedPLE = cfg.numLayers * cfg.hiddenSizePerLayerInput
            self.pleIdentity = try buf(packedPLE)
            self.pleContext = try buf(packedPLE)
            self.pleLayer = try buf(cfg.hiddenSizePerLayerInput)
            self.pleGate = try buf(cfg.hiddenSizePerLayerInput)
        } else {
            self.pleIdentity = nil
            self.pleContext = nil
            self.pleLayer = nil
            self.pleGate = nil
        }

        // The validated tensor is authoritative for each projection's shape.
        // Scratch uses the model-wide maximum, but E2B's first 15 MLPs are
        // narrower than its final 20; stamping the maximum onto every view
        // makes those early GEMVs read into the next packed tensor.
        func sharedProj(_ view: TensorView) -> SharedExpertProjection {
            SharedExpertProjection(weights: view.buffer,
                                 scales: view.buffer,
                                 biases: view.buffer,
                                 weightsOffset: Int(view.offset),
                                 scalesOffset: Int(view.scaleOffset),
                                 biasesOffset: Int(view.biasOffset),
                                 rows: view.shape.0,
                                 cols: view.shape.1)
        }
        var sharedViews: [LayerSharedExpertProjections?] = []
        sharedViews.reserveCapacity(cfg.numLayers)
        for L in 0..<cfg.numLayers {
            guard cfg.hasSharedExpert else {
                sharedViews.append(nil)
                continue
            }
            let gate = try model.sharedExpertGate(layer: L)
            let up = try model.sharedExpertUp(layer: L)
            let down = try model.sharedExpertDown(layer: L)
            sharedViews.append(LayerSharedExpertProjections(
                gate: sharedProj(gate),
                up: sharedProj(up),
                down: sharedProj(down),
                postF1: cfg.feedForwardKind == .dense
                    ? try model.postFFN(layer: L)
                    : (cfg.ffnSandwichNorms ? try model.postFFN1(layer: L) : nil),
                scalarGate: cfg.sharedExpertGated
                    ? try model.sharedExpertScalarGate(layer: L) : nil))
        }
        self.sharedExpertProjections = sharedViews

        func bf16OnesBuffer(count: Int, label: String) throws -> MTLBuffer {
            guard let buf = device.makeBuffer(length: count * MemoryLayout<UInt16>.size,
                                              options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            let dst = buf.contents().assumingMemoryBound(to: UInt16.self)
            for i in 0..<count { dst[i] = 0x3F80 }  // BF16 1.0
            buf.label = label
            return buf
        }

        if cfg.feedForwardKind == .dense {
            self.effectiveScaleBuffers = []
            self.onesPerExpertScale = nil
        } else if cfg.routerScaled {
            // Pre-fold 1/sqrt(D) into router.scale per layer. Each layer gets
            // its own BF16 [D] buffer — the kernel reads `effective_scale[i]`
            // and we pay for the multiply once per generation, not per token.
            var perLayer: [MTLBuffer] = []
            perLayer.reserveCapacity(cfg.numLayers)
            let invSqrtD = Float(1.0) / Float(D).squareRoot()
            let dInts = D
            for L in 0..<cfg.numLayers {
                let scaleView = try model.routerScale(layer: L)
                guard let buf = device.makeBuffer(length: dInts * MemoryLayout<UInt16>.size,
                                                  options: .storageModeShared) else {
                    throw ModelError.residentBufferWrapFailed
                }
                let src = scaleView.buffer.contents()
                    .advanced(by: Int(scaleView.offset))
                    .assumingMemoryBound(to: UInt16.self)
                let dst = buf.contents().assumingMemoryBound(to: UInt16.self)
                for i in 0..<dInts {
                    let v = Quantization.bf16ToFloat(src[i]) * invSqrtD
                    dst[i] = Quantization.bf16Bits(v)
                }
                buf.label = "effective_scale.L\(L)"
                perLayer.append(buf)
            }
            self.effectiveScaleBuffers = perLayer
            self.onesPerExpertScale = nil
        } else {
            // Plain linear router (Qwen): one shared BF16 ones buffer keeps
            // the router kernel's effective_scale multiply neutral, and a ones
            // per_expert_scale keeps the top-k weights untouched. (Softmax
            // over top-k then renormalize equals Qwen's softmax over all
            // experts then renormalize the selected top-k.)
            let ones = try bf16OnesBuffer(count: D, label: "effective_scale.ones")
            self.effectiveScaleBuffers = [MTLBuffer](repeating: ones,
                                                     count: cfg.numLayers)
            self.onesPerExpertScale = try bf16OnesBuffer(count: cfg.numExperts,
                                                         label: "per_expert_scale.ones")
        }
    }

    public func reset() {
        kv?.reset()
        gdnState?.reset()
        qwenMultimodalRopeDelta = 0
        speculativeStartPosition = nil
        speculativeProcessedTokens = 0
        resetTransientState()
    }

    public var continuationPosition: Int {
        kv?.position ?? 0
    }

    public func prepareForContinuation(expectedPosition: Int) throws {
        guard let kv else {
            throw PrefillError.prefillCursorMismatch(
                "continuation requires an initialized KV cache")
        }
        guard expectedPosition > 0, kv.position == expectedPosition else {
            throw PrefillError.prefillCursorMismatch(
                "continuation expected KV position \(expectedPosition), current \(kv.position)")
        }
        speculativeStartPosition = nil
        speculativeProcessedTokens = 0
        resetTransientState()
    }

    private func resetTransientState() {
        prefillChunkState.reset()
        rdadviseSkipUntilPosition = -1
        rdadviseAdaptiveState.reset()
        rdadviseAdaptivePosition = -1
        rdadviseAdaptivePositionBytes = 0
    }

    public private(set) var totalIoNanos: UInt64 = 0
    public private(set) var totalCb1Nanos: UInt64 = 0
    public private(set) var totalCb2Nanos: UInt64 = 0
    public private(set) var totalHeadNanos: UInt64 = 0
    public private(set) var totalHeadFusedNanos: UInt64 = 0
    /// GPU-side execution time (gpuStartTime→gpuEndTime) of the per-layer
    /// command buffers and of the head, for TUFF_PHASES diagnostics.
    public private(set) var totalGPULayerNanos: UInt64 = 0
    public private(set) var totalGPUHeadNanos: UInt64 = 0
    public private(set) var totalGPULayerCommandBuffers: UInt64 = 0
    public private(set) var totalGPURoutedNanos: UInt64 = 0
    public private(set) var totalGPUSharedNanos: UInt64 = 0
    /// Decode-only routed-expert traffic. A read is a cache miss that required
    /// an SSD-backed expert fetch; cache hits are reported separately.
    public private(set) var totalRoutedExpertReads: UInt64 = 0
    public private(set) var totalRoutedExpertBytes: UInt64 = 0
    public private(set) var totalRoutedExpertCacheHits: UInt64 = 0
    public private(set) var totalRoutedExpertCacheMisses: UInt64 = 0
    public private(set) var lastGreedyToken: UInt32 = 0
    public var usesFusedGreedyHead: Bool { useFusedGreedyHead }
    /// The Qwen Gated-DeltaNet state is recurrent rather than rewindable KV,
    /// so this POC advertises verification only for the fused-head affine
    /// path without linear-attention layers.
    public var supportsSpeculativeVerification: Bool {
        useFusedGreedyHead && gdnState == nil
    }
    public private(set) var totalRDAdviseNanos: UInt64 = 0
    public private(set) var totalRDAdviseCalls: UInt64 = 0
    public private(set) var totalRDAdviseBytes: UInt64 = 0
    public private(set) var totalRDAdviseFailures: UInt64 = 0
    public private(set) var totalRDAdviseSkipped: UInt64 = 0

    private func recordRDAdvice(_ result: ExpertIOAdviceResult, wallNanos: UInt64) {
        totalRDAdviseNanos &+= wallNanos
        totalRDAdviseCalls &+= UInt64(result.calls)
        totalRDAdviseBytes &+= result.bytes
        totalRDAdviseFailures &+= UInt64(result.failed)
        totalRDAdviseSkipped &+= UInt64(result.skipped)
    }

    private func shouldSkipRDAdvice(position: Int,
                                    requestedMisses: Int,
                                    estimatedBytes: UInt64,
                                    canOverlapUsefulGPUWork: Bool) -> ExpertIOAdviceResult? {
        switch rdadvisePolicyMode {
        case .bounded:
            if position <= rdadviseSkipUntilPosition {
                return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                    bytes: estimatedBytes)
            }
            if requestedMisses > Self.rdadviseBoundedMissCap {
                return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                    bytes: estimatedBytes)
            }
            return nil
        case .adaptive:
            if position != rdadviseAdaptivePosition {
                rdadviseAdaptivePosition = position
                rdadviseAdaptivePositionBytes = 0
            }
            let cumulativeEstimatedBytes = rdadviseAdaptivePositionBytes &+ estimatedBytes
            let shouldSkip = rdadviseAdaptiveState.shouldSkip(
                position: position,
                requestedMisses: requestedMisses,
                estimatedBytes: cumulativeEstimatedBytes,
                canOverlapUsefulGPUWork: canOverlapUsefulGPUWork)
            rdadviseAdaptivePositionBytes = cumulativeEstimatedBytes
            guard shouldSkip else { return nil }
            return ExpertIOAdviceResult.skipped(requested: requestedMisses,
                                                bytes: estimatedBytes)
        case .default, .off:
            return nil
        }
    }

    private func updateRDAdvicePolicy(after result: ExpertIOAdviceResult,
                                      position: Int) {
        switch rdadvisePolicyMode {
        case .bounded:
            if result.maxCallNanos > Self.rdadviseBoundedMaxCallNanos {
                rdadviseSkipUntilPosition = max(rdadviseSkipUntilPosition, position + 1)
            }
        case .adaptive:
            rdadviseAdaptiveState.update(after: result, position: position)
        case .default, .off:
            break
        }
    }

    public func verifySpeculativeBlock(tokens: [Int32],
                                       startPosition: Int,
                                       into logits: MTLBuffer) async throws
        -> SpeculativeVerificationResult {
        guard supportsSpeculativeVerification else {
            throw PrefillError.chunkedUnsupported(
                "speculative verification requires the fused affine head without GDN state")
        }
        guard (1...8).contains(tokens.count) else {
            throw SpeculativeDecodingError.invalidBlockSize(
                requested: tokens.count, maximum: 8)
        }
        guard speculativeStartPosition == nil else {
            throw PrefillError.chunkedRunnerDirty(
                "a speculative verification transaction is already active")
        }
        guard let kv, kv.position == startPosition else {
            throw SpeculativeDecodingError.invalidStartPosition(
                expected: kv?.position ?? 0, actual: startPosition)
        }
        guard tokens.allSatisfy({ $0 >= 0 && Int($0) < cfg.vocabSize }) else {
            throw GeneratorError.invalidGenerationConfig(
                "speculative token is outside the model vocabulary")
        }
        guard let scratch = try? ensurePrefillScratch(
            config: .production(chunkTokens: 32)) else {
            throw PrefillError.chunkedUnsupported(
                "speculative verification could not allocate bounded prefill scratch")
        }

        speculativeExpertReads = 0
        speculativeExpertBytes = 0
        speculativeExpertCacheHits = 0
        speculativeExpertCacheMisses = 0
        speculativeTargetCommandBuffers = 0
        collectingSpeculativeMetrics = true
        defer { collectingSpeculativeMetrics = false }

        let candidatePointer = speculativeTokenBuffer
            .contents().assumingMemoryBound(to: UInt32.self)
        for (index, token) in tokens.enumerated() {
            candidatePointer[index] = UInt32(bitPattern: token)
        }
        let boundaryToken = lastGreedyToken
        let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        do {
            try await executePrefillChunk(
                tokens: tokens[...],
                startPosition: startPosition,
                outputMode: .greedyIfAvailable,
                logits: logits,
                scratch: scratch,
                config: .production(chunkTokens: 32),
                writeFinalHead: true,
                speculativeTargetTokens: speculativeTargetTokenBuf,
                tokenBufferOverride: speculativeTokenBuffer)
        } catch {
            // The chunk path marks itself dirty before dispatching layers. A
            // failed/cancelled verification must restore both logical state
            // and the reusable prefill scratch state before propagating.
            kv.rewind(to: startPosition)
            prefillChunkState.markCommitted()
            throw error
        }
        let wallNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start

        let targetPointer = speculativeTargetTokenBuf
            .contents().assumingMemoryBound(to: UInt32.self)
        var targetTokens = [Int32(bitPattern: boundaryToken)]
        targetTokens.reserveCapacity(tokens.count + 1)
        for index in 0..<tokens.count {
            targetTokens.append(Int32(bitPattern: targetPointer[index]))
        }
        speculativeStartPosition = startPosition
        speculativeProcessedTokens = tokens.count
        return SpeculativeVerificationResult(
            startPosition: startPosition,
            proposedTokenIDs: tokens,
            targetTokenIDs: targetTokens,
            processedTokens: tokens.count,
            newPosition: startPosition + tokens.count,
            metrics: SpeculativeVerificationMetrics(
                wallNanos: wallNanos,
                targetCommandBuffers: speculativeTargetCommandBuffers,
                expertReads: speculativeExpertReads,
                expertBytes: speculativeExpertBytes,
                expertCacheHits: speculativeExpertCacheHits,
                expertCacheMisses: speculativeExpertCacheMisses))
    }

    public func speculativeBoundaryToken() async throws -> Int32? {
        guard supportsSpeculativeVerification else { return nil }
        return Int32(bitPattern: lastGreedyToken)
    }

    private func recordSpeculativeFetch(layer: Int,
                                        fetch: PrefillStreamedTileFetchResult) {
        guard collectingSpeculativeMetrics else { return }
        let misses = fetch.usedPlannedFetch
            ? fetch.plannedMissSlots.count
            : fetch.expertIDs.count
        let hits = fetch.usedPlannedFetch ? fetch.plannedHits : 0
        recordDecodeExpertFetch(layer: layer, misses: misses, hits: hits)
        speculativeExpertReads &+= UInt64(misses)
        speculativeExpertCacheMisses &+= UInt64(misses)
        speculativeExpertCacheHits &+= UInt64(hits)
        if let bytes = try? model.routedExpertAdviceByteEstimate(
            layer: layer, missCount: misses) {
            speculativeExpertBytes &+= bytes
        }
    }

    private func makePrefillCommandBuffer() -> MTLCommandBuffer? {
        let commandBuffer = ctx.queue.makeCommandBuffer()
        if collectingSpeculativeMetrics, commandBuffer != nil {
            speculativeTargetCommandBuffers &+= 1
        }
        return commandBuffer
    }

    private func recordDecodeExpertFetch(layer: Int,
                                         plan: RoutedExpertFetchPlan?) {
        guard let plan else { return }
        recordDecodeExpertFetch(layer: layer,
                                misses: plan.misses.count,
                                hits: plan.hits)
    }

    private func recordDecodeExpertFetch(layer: Int,
                                         misses: Int,
                                         hits: Int) {
        guard misses >= 0, hits >= 0 else { return }
        totalRoutedExpertReads &+= UInt64(misses)
        totalRoutedExpertCacheMisses &+= UInt64(misses)
        totalRoutedExpertCacheHits &+= UInt64(hits)
        if let bytes = try? model.routedExpertAdviceByteEstimate(
            layer: layer, missCount: misses) {
            totalRoutedExpertBytes &+= bytes
        }
    }

    public func commitSpeculativePrefix(_ count: Int) throws {
        guard let start = speculativeStartPosition else {
            throw SpeculativeDecodingError.noActiveTransaction
        }
        guard (0...speculativeProcessedTokens).contains(count) else {
            throw SpeculativeDecodingError.invalidCommitCount(
                requested: count, processed: speculativeProcessedTokens)
        }
        kv?.rewind(to: start + count)
        speculativeStartPosition = nil
        speculativeProcessedTokens = 0
    }

    public func produce(token: Int32, position: Int, into logits: MTLBuffer,
                        appendingToFinalCommandBuffer epilogue: (MTLCommandBuffer) -> Void)
        async throws -> Bool {
        try prefillChunkState.requireClean(operation: "produce")
        var encoded = false
        // `produceToken` calls the epilogue during its own execution, before
        // returning, so borrowing the non-escaping closure is safe here.
        try await withoutActuallyEscaping(epilogue) { epilogue in
            try await produceToken(token: token, position: position, into: logits,
                                   emitHead: true, outputMode: .greedyIfAvailable,
                                   epilogue: { cb in
                                       encoded = true
                                       epilogue(cb)
                                   })
        }
        return encoded
    }

    public func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try prefillChunkState.requireClean(operation: "produce")
        try await produceToken(token: token,
                               position: position,
                               into: logits,
                               emitHead: true,
                               outputMode: .greedyIfAvailable)
    }

    public func prefillChunked(tokens: ArraySlice<Int32>,
                               startPosition: Int,
                               outputMode: PrefillOutputMode,
                               config: PrefillRuntimeConfig,
                               into logits: MTLBuffer,
                               onProgress: (Int) -> Void) async throws -> PrefillResult {
        try prefillChunkState.requireClean(operation: "prefillChunked")
        guard config.mode == .chunked else {
            throw PrefillError.chunkedUnsupported(
                "prefillChunked requires PrefillRuntimeConfig.mode == .chunked")
        }
        guard startPosition >= 0 else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill startPosition must be non-negative")
        }
        let kvPosition = kv?.position ?? 0
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        guard tokens.count <= maxContext - startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range starting at \(startPosition) with \(tokens.count) tokens exceeds maxContext \(maxContext)")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten)
        }
        let scratch = try ensurePrefillScratch(config: config)
        let spans = PrefillChunkPlanner.spans(tokenCount: tokens.count,
                                              startPosition: startPosition,
                                              config: config)
        try await PrefillSpanIteration.forEachSpan(spans) { spanIndex, span in
            let lower = tokens.index(tokens.startIndex, offsetBy: span.tokenOffset)
            let upper = tokens.index(lower, offsetBy: span.tokenCount)
            try await executePrefillChunk(
                tokens: tokens[lower..<upper],
                startPosition: span.startPosition,
                outputMode: outputMode,
                logits: logits,
                scratch: scratch,
                config: config,
                writeFinalHead: spanIndex == spans.count - 1)
            onProgress(span.completedCount)
        }
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            return PrefillResult(newPosition: startPosition + tokens.count,
                                 seed: .greedyToken(lastGreedyToken))
        }
        return PrefillResult(newPosition: startPosition + tokens.count,
                             seed: .logitsWritten)
    }


    public func prefillMultimodal(input: MultimodalPrefillInput,
                                  startPosition: Int,
                                  outputMode: PrefillOutputMode,
                                  config: PrefillRuntimeConfig,
                                  into logits: MTLBuffer,
                                  onProgress: (Int) -> Void) async throws -> PrefillResult {
        try prefillChunkState.requireClean(operation: "prefillMultimodal")
        // The same property `coercedForImagePrompt()` repairs, so the guard and
        // the coercion cannot drift apart.
        guard config.servesImagePrompt else {
            throw PrefillError.chunkedUnsupported(
                "multimodal prefill requires the complete chunked prefill path")
        }
        let tokens = input.embeddingTokenIDs
        guard startPosition >= 0,
              startPosition + tokens.count <= maxContext else {
            throw PrefillError.chunkedUnsupported(
                "multimodal prefill range exceeds maxContext \(maxContext)")
        }
        let kvPosition = kv?.position ?? 0
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "multimodal prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition, seed: .logitsWritten)
        }
        guard input.imageSpans.allSatisfy({
            $0.features.hiddenSize == cfg.hiddenSize
                && $0.features.tokenCount == $0.tokenRange.count
        }) else {
            throw PrefillError.chunkedUnsupported(
                "image features do not match the loaded model architecture")
        }

        // The planner applies the same clamp the scratch layout does. Cutting at
        // the raw `config.chunkTokens` let a caller size a 280-token scratch and
        // then hand it a larger chunk, which died on the guard.
        let textChunkTokens = min(config.chunkTokens, PrefillRuntimeConfig.maxChunkTokens)
        let work = PrefillChunkPlanner.multimodalWork(
            tokenCount: tokens.count,
            imageRanges: input.imageSpans.map(\.tokenRange),
            chunkTokens: config.chunkTokens)
        let imageBuffers = input.imageSpans.map(\.features.buffer)

        // One layout for the whole turn, sized for the largest item in it.
        // Keying the single-slot cache on each item's exact chunk size made a
        // text-image-text prompt free and reallocate the entire chunk scratch at
        // every boundary, twice per image span, every turn.
        let pooledTokens = input.imageSpans.map(\.tokenRange.count).max() ?? 0
        let hasImages = work.contains(where: \.isImage)
        let layoutTokens = hasImages
            ? max(config.chunkTokens, pooledTokens)
            : config.chunkTokens
        let layoutLimit = hasImages
            ? max(PrefillRuntimeConfig.maxChunkTokens, pooledTokens)
            : PrefillRuntimeConfig.maxChunkTokens
        let scratch = try ensurePrefillScratch(
            config: config.replacingChunkTokens(layoutTokens),
            chunkTokenLimit: layoutLimit)

        var resolvedPositions = input.positionIDs
        var resolvedRopeDelta = input.ropeDelta
        if let positions = resolvedPositions, startPosition > 0,
           positions.temporal.first == 0,
           positions.height.first == 0,
           positions.width.first == 0 {
            let base = Int32(startPosition) + qwenMultimodalRopeDelta
            resolvedRopeDelta = qwenMultimodalRopeDelta + positions.ropeDelta
            resolvedPositions = try positions.offsettingPositions(
                by: base, ropeDelta: resolvedRopeDelta)
        }

        var completed = 0
        for (index, item) in work.enumerated() {
            try Task.checkCancellation()
            let isImage = item.isImage
            let imageFeatures = item.imageIndex.map { imageBuffers[$0] }
            // Every item executes under the chunk size it was planned at.
            let chunkConfig = config.replacingChunkTokens(
                isImage ? item.range.count : textChunkTokens)
            let lower = tokens.index(tokens.startIndex, offsetBy: item.range.lowerBound)
            let upper = tokens.index(tokens.startIndex, offsetBy: item.range.upperBound)
            try await executePrefillChunk(
                tokens: tokens[lower..<upper],
                startPosition: startPosition + item.range.lowerBound,
                outputMode: outputMode,
                logits: logits,
                scratch: scratch,
                config: chunkConfig,
                writeFinalHead: index == work.count - 1,
                embeddingOverride: imageFeatures,
                bidirectionalBlock: isImage && input.usesBidirectionalImageAttention
                    ? (startPosition + item.range.lowerBound)..<(startPosition + item.range.upperBound)
                    : nil,
                positionIDs: try resolvedPositions?.slice(item.range))
            completed += item.range.count
            onProgress(completed)
        }
        if input.family == .qwen36 {
            qwenMultimodalRopeDelta = resolvedRopeDelta
        }
        if outputMode == .greedyIfAvailable, useFusedGreedyHead {
            return PrefillResult(newPosition: startPosition + tokens.count,
                                 seed: .greedyToken(lastGreedyToken))
        }
        return PrefillResult(newPosition: startPosition + tokens.count,
                             seed: .logitsWritten)
    }

    @discardableResult
    private func ensurePrefillScratch(
        config: PrefillRuntimeConfig,
        chunkTokenLimit: Int = PrefillRuntimeConfig.maxChunkTokens
    ) throws -> PrefillChunkScratchBuffers {
        let layout = PrefillChunkScratchLayout(config: cfg,
                                               chunkTokens: config.chunkTokens,
                                               chunkTokenLimit: chunkTokenLimit)
        if let scratch = prefillScratch, scratch.layout == layout {
            return scratch
        }
        let scratch = try PrefillChunkScratchBuffers.allocate(device: ctx.device, layout: layout)
        prefillScratch = scratch
        return scratch
    }

    private func executePrefillChunk(tokens: ArraySlice<Int32>,
                                     startPosition: Int,
                                     outputMode: PrefillOutputMode,
                                     logits: MTLBuffer,
                                     scratch: PrefillChunkScratchBuffers,
                                     config: PrefillRuntimeConfig,
                                     writeFinalHead: Bool,
                                     embeddingOverride: MTLBuffer? = nil,
                                     bidirectionalBlock: Range<Int>? = nil,
                                     positionIDs: MultimodalPositionIDs? = nil,
                                     speculativeTargetTokens: MTLBuffer? = nil,
                                     tokenBufferOverride: MTLBuffer? = nil) async throws {
        guard !tokens.isEmpty else { return }
        guard kv != nil else {
            throw PrefillError.chunkedUnsupported("chunked prefill attention requires FP16 KV")
        }
        let kvPosition = kv?.position ?? 0
        guard kvPosition == startPosition else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill cursor \(kvPosition) != startPosition \(startPosition)")
        }
        guard startPosition >= 0, startPosition + tokens.count <= maxContext else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill range [\(startPosition), \(startPosition + tokens.count)) exceeds maxContext \(maxContext)")
        }
        guard tokens.count <= scratch.layout.chunkTokens else {
            throw PrefillError.chunkedUnsupported(
                "chunked prefill token count \(tokens.count) exceeds scratch chunk size \(scratch.layout.chunkTokens)")
        }
        if let kv, kv.fp16RingEnabled, let ringLayer = (0..<cfg.numLayers).first(where: {
            kv.ringCapacity(layer: $0) > 0
        }) {
            let requiredCapacity = min(maxContext, cfg.slidingWindow + config.chunkTokens)
            let ringCapacity = kv.ringCapacity(layer: ringLayer)
            guard requiredCapacity <= ringCapacity else {
                throw PrefillError.chunkedUnsupported(
                    "FP16 KV ring capacity \(ringCapacity) cannot hold required capacity \(requiredCapacity) for maxContext \(maxContext), slidingWindow \(cfg.slidingWindow), and prefillChunkTokens \(config.chunkTokens)")
            }
        }

        struct LayerPrefillQKVViews {
            let inputNorm: TensorView
            let postAttention: TensorView
            let router: TensorView?
            // Softmax-attention layers only (nil on linear-attention layers).
            let q: TensorView?
            let k: TensorView?
            let v: TensorView?
            let o: TensorView?
            let qNorm: TensorView?
            let kNorm: TensorView?
            // Gemma FFN sandwich only.
            let preFFN: TensorView?
            let preFFN2: TensorView?
            let postFFN2: TensorView?
            let postFFN: TensorView?
            let layerScalar: TensorView?
            let routerPerExpertScale: TensorView?
            // Dense Gemma PLE mapping.
            let perLayerInputGate: TensorView?
            let perLayerProjection: TensorView?
            let postPerLayerInputNorm: TensorView?
            // Gated-DeltaNet linear-attention layers only.
            let linQKV: TensorView?
            let linZ: TensorView?
            let linA: TensorView?
            let linB: TensorView?
            let linOut: TensorView?
            let linConv: TensorView?
            let linALog: TensorView?
            let linDtBias: TensorView?
            let linNorm: TensorView?
        }

        let layerViews = try (0..<cfg.numLayers).map { L in
            let isFull = cfg.fullAttentionLayerMask[L] == 1
            let isLinear = cfg.layerIsLinear(L)
            let dense = cfg.feedForwardKind == .dense
            let sandwich = cfg.ffnSandwichNorms && !dense
            let sharesKV = cfg.layerSharesKV(L)
            return LayerPrefillQKVViews(
                inputNorm: try model.inputNorm(layer: L),
                postAttention: try model.postAttnNorm(layer: L),
                router: dense ? nil : try model.router(layer: L),
                q: isLinear ? nil : try model.qProj(layer: L),
                k: (isLinear || sharesKV) ? nil : try model.kProj(layer: L),
                v: (isLinear || sharesKV) ? nil
                    : ((isFull && cfg.attentionKEqV)
                        ? (try model.kProj(layer: L))
                        : (try model.vProj(layer: L))),
                o: isLinear ? nil : try model.oProj(layer: L),
                qNorm: isLinear ? nil : try model.qNorm(layer: L),
                kNorm: (isLinear || sharesKV) ? nil : try model.kNorm(layer: L),
                preFFN: (sandwich || dense) ? try model.preFFN(layer: L) : nil,
                preFFN2: sandwich ? try model.preFFN2(layer: L) : nil,
                postFFN2: sandwich ? try model.postFFN2(layer: L) : nil,
                postFFN: (sandwich || dense) ? try model.postFFN(layer: L) : nil,
                layerScalar: (sandwich || dense) ? try model.layerScalar(layer: L) : nil,
                routerPerExpertScale: cfg.routerScaled
                    && !dense ? try model.routerPerExpertScale(layer: L) : nil,
                perLayerInputGate: dense && cfg.hasPerLayerInputs
                    ? try model.perLayerInputGate(layer: L) : nil,
                perLayerProjection: dense && cfg.hasPerLayerInputs
                    ? try model.perLayerProjection(layer: L) : nil,
                postPerLayerInputNorm: dense && cfg.hasPerLayerInputs
                    ? try model.postPerLayerInputNorm(layer: L) : nil,
                linQKV: isLinear ? try model.linearInProjQKV(layer: L) : nil,
                linZ: isLinear ? try model.linearInProjZ(layer: L) : nil,
                linA: isLinear ? try model.linearInProjA(layer: L) : nil,
                linB: isLinear ? try model.linearInProjB(layer: L) : nil,
                linOut: isLinear ? try model.linearOutProj(layer: L) : nil,
                linConv: isLinear ? try model.linearConv1d(layer: L) : nil,
                linALog: isLinear ? try model.linearALog(layer: L) : nil,
                linDtBias: isLinear ? try model.linearDtBias(layer: L) : nil,
                linNorm: isLinear ? try model.linearNorm(layer: L) : nil)
        }

        let tokenIDs = tokens.map { UInt32(bitPattern: $0) }
        let tokenBuffer: MTLBuffer
        if let tokenBufferOverride {
            guard tokenBufferOverride.length
                >= tokenIDs.count * MemoryLayout<UInt32>.stride else {
                throw ModelError.residentBufferWrapFailed
            }
            tokenIDs.withUnsafeBytes { bytes in
                if let baseAddress = bytes.baseAddress {
                    memcpy(tokenBufferOverride.contents(), baseAddress, bytes.count)
                }
            }
            tokenBuffer = tokenBufferOverride
        } else {
            guard let allocated = ctx.device.makeBuffer(
                bytes: tokenIDs,
                length: tokenIDs.count * MemoryLayout<UInt32>.stride,
                options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            tokenBuffer = allocated
        }
        let mropeBuffers: (temporal: MTLBuffer, height: MTLBuffer, width: MTLBuffer)?
        if let positionIDs {
            guard positionIDs.count == tokens.count,
                  let temporal = ctx.device.makeBuffer(
                    bytes: positionIDs.temporal,
                    length: positionIDs.count * MemoryLayout<Int32>.stride,
                    options: .storageModeShared),
                  let height = ctx.device.makeBuffer(
                    bytes: positionIDs.height,
                    length: positionIDs.count * MemoryLayout<Int32>.stride,
                    options: .storageModeShared),
                  let width = ctx.device.makeBuffer(
                    bytes: positionIDs.width,
                    length: positionIDs.count * MemoryLayout<Int32>.stride,
                    options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            mropeBuffers = (temporal, height, width)
        } else {
            mropeBuffers = nil
        }
        let D = cfg.hiddenSize
        let eps: Float = 1e-6
        let embedOutScale = cfg.embeddingScaledBySqrtHidden
            ? Float(D).squareRoot()
            : 1.0
        let t = tokens.count
        let emb = model.embedding

        func encodeInt4Projection(commandBuffer: MTLCommandBuffer,
                                  family: PrefillProjectionFamily,
                                  weights: TensorView,
                                  x: MTLBuffer,
                                  y: MTLBuffer,
                                  rows: Int,
                                  columns: Int,
                                  tokenCount: Int,
                                  xStrideElements: Int,
                                  yStrideElements: Int) {
            if tokenCount >= 32,
               family == .q || family == .kv || family == .o,
               let candidate = prefillMPPAffineInt4 {
                let metadata = candidate.encode(
                    commandBuffer: commandBuffer,
                    weights: weights.buffer,
                    weightsOffset: Int(weights.offset),
                    scales: weights.buffer,
                    scalesOffset: Int(weights.scaleOffset),
                    biases: weights.buffer,
                    biasesOffset: Int(weights.biasOffset),
                    x: x,
                    y: y,
                    m: tokenCount,
                    n: rows,
                    k: columns)
                if metadata.path == .affineThreadgroupF16 {
                    return
                }
            }
            if PrefillProjectionDispatchPolicy.selectedDispatch(for: family,
                                                                chunkTokens: tokenCount) == .qmm {
                prefillQMM.encode(commandBuffer: commandBuffer,
                                  weights: weights.buffer,
                                  weightsOffset: Int(weights.offset),
                                  scales: weights.buffer,
                                  scalesOffset: Int(weights.scaleOffset),
                                  biases: weights.buffer,
                                  biasesOffset: Int(weights.biasOffset),
                                  x: x,
                                  y: y,
                                  t: tokenCount,
                                  n: rows,
                                  k: columns)
                return
            }
            for row in 0..<tokenCount {
                int4.encode(commandBuffer: commandBuffer,
                            weights: weights.buffer,
                            weightsOffset: Int(weights.offset),
                            scales: weights.buffer,
                            scalesOffset: Int(weights.scaleOffset),
                            biases: weights.buffer,
                            biasesOffset: Int(weights.biasOffset),
                            x: x,
                            xOffset: row * xStrideElements * MemoryLayout<Float16>.stride,
                            y: y,
                            yOffset: row * yStrideElements * MemoryLayout<Float16>.stride,
                            m: UInt32(rows),
                            n: UInt32(columns))
            }
        }

        func copyPrefillKV(commandBuffer: MTLCommandBuffer,
                           source: MTLBuffer,
                           destination: (buffer: MTLBuffer, offset: Int, stride: Int),
                           sourceTokenOffset: Int,
                           tokenCount: Int,
                           bytesPerToken: Int) throws {
            guard tokenCount > 0 else { return }
            guard let blit = commandBuffer.makeBlitCommandEncoder() else {
                throw ModelError.residentBufferWrapFailed
            }
            blit.copy(from: source,
                      sourceOffset: sourceTokenOffset * bytesPerToken,
                      to: destination.buffer,
                      destinationOffset: destination.offset,
                      size: tokenCount * bytesPerToken)
            blit.endEncoding()
        }

        func copyPrefillKVToCache(commandBuffer: MTLCommandBuffer,
                                  kv: KVCacheManager,
                                  layer: Int,
                                  startPosition: Int,
                                  tokenCount: Int,
                                  keySource: MTLBuffer,
                                  valueSource: MTLBuffer,
                                  bytesPerToken: Int) throws {
            let capacity = kv.capacity(layer: layer)
            let physicalStart = startPosition % capacity
            let firstSpan = min(tokenCount, capacity - physicalStart)
            let keyFirst = kv.kRange(layer: layer, start: startPosition, count: firstSpan)
            let valueFirst = kv.vRange(layer: layer, start: startPosition, count: firstSpan)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: keySource,
                              destination: keyFirst,
                              sourceTokenOffset: 0,
                              tokenCount: firstSpan,
                              bytesPerToken: bytesPerToken)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: valueSource,
                              destination: valueFirst,
                              sourceTokenOffset: 0,
                              tokenCount: firstSpan,
                              bytesPerToken: bytesPerToken)
            guard firstSpan < tokenCount else { return }

            let secondCount = tokenCount - firstSpan
            let secondStart = startPosition + firstSpan
            let keySecond = kv.kRange(layer: layer, start: secondStart, count: secondCount)
            let valueSecond = kv.vRange(layer: layer, start: secondStart, count: secondCount)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: keySource,
                              destination: keySecond,
                              sourceTokenOffset: firstSpan,
                              tokenCount: secondCount,
                              bytesPerToken: bytesPerToken)
            try copyPrefillKV(commandBuffer: commandBuffer,
                              source: valueSource,
                              destination: valueSecond,
                              sourceTokenOffset: firstSpan,
                              tokenCount: secondCount,
                              bytesPerToken: bytesPerToken)
        }

        prefillChunkState.markDirty(startPosition: startPosition, tokenCount: tokens.count)

        guard var cb = makePrefillCommandBuffer() else {
            throw ModelError.residentBufferWrapFailed
        }
        if let embeddingOverride {
            // The override spans the whole chunk — the planner emits image spans
            // as standalone chunks — so the INT4 gather it would cover is
            // skipped rather than encoded and fully overwritten.
            let bytes = t * D * MemoryLayout<Float16>.stride
            guard embeddingOverride.length >= bytes,
                  let blit = cb.makeBlitCommandEncoder() else {
                throw VisionRuntimeError.invalidInput(
                    "projected image feature buffer is too small")
            }
            blit.copy(from: embeddingOverride,
                      sourceOffset: 0,
                      to: scratch.hidden,
                      destinationOffset: 0,
                      size: bytes)
            blit.endEncoding()
        } else {
            prefillEmbed.encode(commandBuffer: cb,
                                table: emb.buffer,
                                tableOffset: Int(emb.offset),
                                scales: emb.buffer,
                                scalesOffset: Int(emb.scaleOffset),
                                biases: emb.buffer,
                                biasesOffset: Int(emb.biasOffset),
                                tokens: tokenBuffer,
                                out: scratch.hidden,
                                t: UInt32(t),
                                d: UInt32(D),
                                outScale: embedOutScale)
        }

        if cfg.hasPerLayerInputs {
            let plan = PerLayerEmbeddingPlan(config: cfg)
            let identity = model.perLayerEmbedding
            prefillEmbed.encode(
                commandBuffer: cb,
                table: identity.buffer,
                tableOffset: Int(identity.offset),
                scales: identity.buffer,
                scalesOffset: Int(identity.scaleOffset),
                biases: identity.buffer,
                biasesOffset: Int(identity.biasOffset),
                tokens: tokenBuffer,
                out: scratch.pleIdentity,
                t: UInt32(t),
                d: UInt32(plan.packedWidth),
                outScale: plan.tokenIdentityScale)
            encodeInt4Projection(
                commandBuffer: cb,
                family: .shared,
                weights: model.perLayerModelProjection,
                x: scratch.hidden,
                y: scratch.pleContext,
                rows: plan.packedWidth,
                columns: D,
                tokenCount: t,
                xStrideElements: D,
                yStrideElements: plan.packedWidth)
        }

        for L in 0..<cfg.numLayers {
            if cfg.feedForwardKind == .mixtureOfExperts {
                model.beginOpeningRoutedExpertStreamer(layer: L)
            }
            let views = layerViews[L]
            let isLinear = cfg.layerIsLinear(L)
            let isFull = cfg.fullAttentionLayerMask[L] == 1
            let headDim = isFull ? cfg.fullHeadDim : cfg.headDim
            let numKVHeads = isFull ? cfg.numFullKVHeads : cfg.numKVHeads
            let qDim = cfg.numHeads * headDim
            let kvDim = numKVHeads * headDim

            prefillRMS.encodeBF16W(commandBuffer: cb,
                                   x: scratch.hidden,
                                   weight: views.inputNorm.buffer,
                                   weightOffset: Int(views.inputNorm.offset),
                                   out: scratch.normed,
                                   t: UInt32(t),
                                   d: UInt32(D),
                                   eps: eps)
            if isLinear {
                // Gated-DeltaNet linear attention over the chunk: batched
                // projections, causal conv (+ tail carry), delta-rule
                // recurrence, gated norm, out_proj. No KV writes, no
                // attention, no blit.
                guard let gdn, let gdnState else {
                    preconditionFailure("linear-attention layer without GDN kernels")
                }
                let la = cfg.linearAttention
                encodeInt4Projection(commandBuffer: cb,
                                     family: .q,
                                     weights: views.linQKV!,
                                     x: scratch.normed,
                                     y: scratch.q,
                                     rows: la.qkvDim,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: la.qkvDim)
                encodeInt4Projection(commandBuffer: cb,
                                     family: .kv,
                                     weights: views.linZ!,
                                     x: scratch.normed,
                                     y: scratch.gdnZ,
                                     rows: la.valueDim,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: la.valueDim)
                encodeInt4Projection(commandBuffer: cb,
                                     family: .kv,
                                     weights: views.linA!,
                                     x: scratch.normed,
                                     y: scratch.gdnA,
                                     rows: la.numVHeads,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: la.numVHeads)
                encodeInt4Projection(commandBuffer: cb,
                                     family: .kv,
                                     weights: views.linB!,
                                     x: scratch.normed,
                                     y: scratch.gdnB,
                                     rows: la.numVHeads,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: la.numVHeads)
                let convW = views.linConv!
                let tail = gdnState.convTailBuffer(layer: L)
                gdn.encodeConvPrefill(commandBuffer: cb,
                                      tail: tail,
                                      qkvRows: scratch.q,
                                      convWeight: convW.buffer,
                                      convWeightOffset: Int(convW.offset),
                                      out: scratch.gdnConvOut,
                                      rows: t)
                gdn.encodeConvTailUpdate(commandBuffer: cb,
                                         tail: tail,
                                         qkvRows: scratch.q,
                                         rows: t)
                gdn.encodeQKNorm(commandBuffer: cb,
                                 convOut: scratch.gdnConvOut,
                                 rows: t)
                let aLog = views.linALog!
                let dtBias = views.linDtBias!
                gdn.encodeDeltaStepPrefill(commandBuffer: cb,
                                           convOut: scratch.gdnConvOut,
                                           aProj: scratch.gdnA,
                                           bProj: scratch.gdnB,
                                           aLog: aLog.buffer,
                                           aLogOffset: Int(aLog.offset),
                                           dtBias: dtBias.buffer,
                                           dtBiasOffset: Int(dtBias.offset),
                                           state: gdnState.stateBuffer(layer: L),
                                           y: scratch.gdnY,
                                           rows: t)
                let gatedNormW = views.linNorm!
                gdn.encodeGatedNorm(commandBuffer: cb,
                                    y: scratch.gdnY,
                                    z: scratch.gdnZ,
                                    weight: gatedNormW.buffer,
                                    weightOffset: Int(gatedNormW.offset),
                                    out: scratch.attentionOutput,
                                    rows: t)
                encodeInt4Projection(commandBuffer: cb,
                                     family: .o,
                                     weights: views.linOut!,
                                     x: scratch.attentionOutput,
                                     y: scratch.h1,
                                     rows: D,
                                     columns: la.valueDim,
                                     tokenCount: t,
                                     xStrideElements: la.valueDim,
                                     yStrideElements: D)
            } else {
                let qProjRows = cfg.attnOutputGate ? 2 * qDim : qDim
                encodeInt4Projection(commandBuffer: cb,
                                     family: .q,
                                     weights: views.q!,
                                     x: scratch.normed,
                                     y: scratch.q,
                                     rows: qProjRows,
                                     columns: D,
                                     tokenCount: t,
                                     xStrideElements: D,
                                     yStrideElements: qProjRows)
                if !cfg.layerSharesKV(L) {
                    encodeInt4Projection(commandBuffer: cb,
                                         family: .kv,
                                         weights: views.k!,
                                         x: scratch.normed,
                                         y: scratch.kStage,
                                         rows: kvDim,
                                         columns: D,
                                         tokenCount: t,
                                         xStrideElements: D,
                                         yStrideElements: kvDim)
                    encodeInt4Projection(commandBuffer: cb,
                                         family: .kv,
                                         weights: views.v!,
                                         x: scratch.normed,
                                         y: scratch.vStage,
                                         rows: kvDim,
                                         columns: D,
                                         tokenCount: t,
                                         xStrideElements: D,
                                         yStrideElements: kvDim)
                }

                // The attention input Q: the packed q_proj output is split
                // into per-head query/gate halves for gated architectures.
                let attnQ: MTLBuffer
                if cfg.attnOutputGate {
                    elementwise!.encodeSplitQGate(commandBuffer: cb,
                                                  packed: scratch.q,
                                                  q: scratch.attnQ,
                                                  gate: scratch.attnGate,
                                                  heads: cfg.numHeads,
                                                  dim: headDim,
                                                  rows: t)
                    attnQ = scratch.attnQ
                } else {
                    attnQ = scratch.q
                }

                if cfg.layerSharesKV(L) {
                    let rotatedPairs = isFull
                        ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                        : UInt32(headDim / 2)
                    prefillQKVEpilogue.encodeGemmaQOnly(
                        commandBuffer: cb,
                        q: attnQ,
                        qWeight: views.qNorm!.buffer,
                        qWeightOffset: Int(views.qNorm!.offset),
                        startPosition: UInt32(startPosition),
                        queryCount: UInt32(t),
                        headDim: UInt32(headDim),
                        numQHeads: UInt32(cfg.numHeads),
                        qTokenStrideElements: UInt32(qDim),
                        theta: isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                        rotatedPairs: rotatedPairs,
                        eps: eps)
                } else if cfg.usesProjectionWideQKNorm {
                    let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)
                    prefillRMS.encodeBF16W(
                        commandBuffer: cb,
                        x: attnQ,
                        weight: views.qNorm!.buffer,
                        weightOffset: Int(views.qNorm!.offset),
                        out: attnQ,
                        t: UInt32(t), d: UInt32(qDim), eps: eps)
                    prefillRMS.encodeBF16W(
                        commandBuffer: cb,
                        x: scratch.kStage,
                        weight: views.kNorm!.buffer,
                        weightOffset: Int(views.kNorm!.offset),
                        out: scratch.kStage,
                        t: UInt32(t), d: UInt32(kvDim), eps: eps)
                    prefillQKVEpilogue.encodeProjectionWideNeoxSubdimRoPE(
                        commandBuffer: cb,
                        q: attnQ, k: scratch.kStage,
                        startPosition: UInt32(startPosition),
                        queryCount: UInt32(t),
                        headDim: UInt32(headDim),
                        numQHeads: UInt32(cfg.numHeads),
                        numKVHeads: UInt32(numKVHeads),
                        qTokenStrideElements: UInt32(qDim),
                        kvTokenStrideElements: UInt32(kvDim),
                        theta: Float(cfg.fullRopeTheta),
                        rotaryDim: rotaryDim)
                } else if cfg.ropeNeoxSubdim {
                    let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)
                    if let mropeBuffers {
                        prefillQKVEpilogue.encodeMRoPENeoxSubdimNoVNorm(
                            commandBuffer: cb,
                            q: attnQ,
                            k: scratch.kStage,
                            qWeight: views.qNorm!.buffer,
                            qWeightOffset: Int(views.qNorm!.offset),
                            kWeight: views.kNorm!.buffer,
                            kWeightOffset: Int(views.kNorm!.offset),
                            queryCount: UInt32(t),
                            headDim: UInt32(headDim),
                            numQHeads: UInt32(cfg.numHeads),
                            numKVHeads: UInt32(numKVHeads),
                            qTokenStrideElements: UInt32(qDim),
                            kvTokenStrideElements: UInt32(kvDim),
                            theta: Float(cfg.fullRopeTheta),
                            rotaryDim: rotaryDim,
                            eps: eps,
                            temporalPositions: mropeBuffers.temporal,
                            heightPositions: mropeBuffers.height,
                            widthPositions: mropeBuffers.width)
                    } else {
                        prefillQKVEpilogue.encodeNeoxSubdimNoVNorm(
                        commandBuffer: cb,
                        q: attnQ,
                        k: scratch.kStage,
                        qWeight: views.qNorm!.buffer,
                        qWeightOffset: Int(views.qNorm!.offset),
                        kWeight: views.kNorm!.buffer,
                        kWeightOffset: Int(views.kNorm!.offset),
                        startPosition: UInt32(startPosition),
                        queryCount: UInt32(t),
                        headDim: UInt32(headDim),
                        numQHeads: UInt32(cfg.numHeads),
                        numKVHeads: UInt32(numKVHeads),
                        qTokenStrideElements: UInt32(qDim),
                        kvTokenStrideElements: UInt32(kvDim),
                        theta: Float(cfg.fullRopeTheta),
                        rotaryDim: rotaryDim,
                        eps: eps)
                    }
                } else {
                    let rotatedPairs = isFull
                        ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                        : UInt32(headDim / 2)
                    prefillQKVEpilogue.encode(commandBuffer: cb,
                                               q: attnQ,
                                               k: scratch.kStage,
                                               v: scratch.vStage,
                                               qWeight: views.qNorm!.buffer,
                                               qWeightOffset: Int(views.qNorm!.offset),
                                               kWeight: views.kNorm!.buffer,
                                               kWeightOffset: Int(views.kNorm!.offset),
                                               startPosition: UInt32(startPosition),
                                               queryCount: UInt32(t),
                                               headDim: UInt32(headDim),
                                               numQHeads: UInt32(cfg.numHeads),
                                               numKVHeads: UInt32(numKVHeads),
                                               qTokenStrideElements: UInt32(qDim),
                                               kvTokenStrideElements: UInt32(kvDim),
                                               theta: isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                                               rotatedPairs: rotatedPairs,
                                               eps: eps)
                }

                if let kv, !cfg.layerSharesKV(L) {
                    let bytes = t * kvDim * MemoryLayout<Float16>.stride
                    try copyPrefillKVToCache(commandBuffer: cb,
                                             kv: kv,
                                             layer: L,
                                             startPosition: startPosition,
                                             tokenCount: t,
                                             keySource: scratch.kStage,
                                             valueSource: scratch.vStage,
                                             bytesPerToken: bytes / t)
                }
                let params = PrefillAttentionParams(
                        startPosition: UInt32(startPosition),
                        queryCount: UInt32(t),
                        headDim: UInt32(headDim),
                        numQHeads: UInt32(cfg.numHeads),
                        numKVHeads: UInt32(numKVHeads),
                        kvValidCount: UInt32(startPosition + t),
                        slidingWindow: isFull ? UInt32(startPosition + t) : UInt32(cfg.slidingWindow),
                        kvTokenStrideElements: UInt32(kvDim),
                        qTokenStrideElements: UInt32(qDim),
                        oTokenStrideElements: UInt32(qDim),
                        scale: Float(cfg.attentionScale),
                        bidirectionalBlockStart: UInt32(
                            bidirectionalBlock?.lowerBound ?? 0),
                        bidirectionalBlockEnd: UInt32(
                            bidirectionalBlock?.upperBound ?? 0))
                if let kv {
                        let kvLayer = cfg.kvSourceLayer(for: L) ?? L
                        let keyBuffer = kv.keyBuffer(layer: kvLayer, validTokenCount: startPosition + t)
                        let valueBuffer = kv.valueBuffer(layer: kvLayer, validTokenCount: startPosition + t)
                        let ringCapacity = kv.ringCapacity(layer: kvLayer)
                        let activeRingCapacity = ringCapacity > 0 && startPosition + t > ringCapacity
                            ? UInt32(ringCapacity)
                            : 0
                        prefillAttention.encodeCausal(commandBuffer: cb,
                                                      q: attnQ,
                                                      k: keyBuffer,
                                                      v: valueBuffer,
                                                      out: scratch.attentionOutput,
                                                      params: params,
                                                      kvRingCapacity: activeRingCapacity,
                                                      layerKind: isFull ? .full : .slidingWindow,
                                                      allowsBidirectionalFullAttention:
                                                          cfg.variant == .gemma4_12B_QAT,
                                                      path: prefillAttentionPath)
                } else {
                    throw PrefillError.chunkedUnsupported(
                        "chunked prefill attention requires FP16 KV")
                }
                if cfg.attnOutputGate {
                    elementwise!.encodeSigmoidGateMul(commandBuffer: cb,
                                                      out: scratch.attentionOutput,
                                                      gate: scratch.attnGate,
                                                      count: t * qDim)
                }
                encodeInt4Projection(commandBuffer: cb,
                                         family: .o,
                                         weights: views.o!,
                                         x: scratch.attentionOutput,
                                         y: scratch.h1,
                                         rows: D,
                                         columns: qDim,
                                         tokenCount: t,
                                         xStrideElements: qDim,
                                         yStrideElements: D)
            }
            if cfg.feedForwardKind == .dense {
                // Official Gemma 4 dense order: post-normalized attention
                // residual, dense FFN sandwich, PLE mapping residual, scale.
                prefillRMS.encodeBF16W(
                    commandBuffer: cb,
                    x: scratch.h1,
                    weight: views.postAttention.buffer,
                    weightOffset: Int(views.postAttention.offset),
                    out: scratch.h2,
                    t: UInt32(t), d: UInt32(D), eps: eps)
                elementwise!.encodeResidualAdd(commandBuffer: cb,
                                               hidden: scratch.hidden,
                                               delta: scratch.h2,
                                               count: t * D)
                prefillRMS.encodeBF16W(
                    commandBuffer: cb,
                    x: scratch.hidden,
                    weight: views.preFFN!.buffer,
                    weightOffset: Int(views.preFFN!.offset),
                    out: scratch.denseX,
                    t: UInt32(t), d: UInt32(D), eps: eps)
                let denseProjections = sharedExpertProjections[L]!
                try prefillSharedExpert.encodeBlock(
                    commandBuffer: cb,
                    x: scratch.denseX,
                    y: scratch.h1,
                    gate: denseProjections.gate,
                    up: denseProjections.up,
                    down: denseProjections.down,
                    scratchGate: scratch.sharedGateScratch,
                    scratchUp: scratch.sharedUpScratch,
                    scratchAct: scratch.sharedActScratch,
                    queryCount: t,
                    d: D,
                    intermediate: cfg.ffnIntermediateSize(layer: L),
                    xStrideElements: D,
                    yStrideElements: D)
                prefillRMS.encodeBF16W(
                    commandBuffer: cb,
                    x: scratch.h1,
                    weight: views.postFFN!.buffer,
                    weightOffset: Int(views.postFFN!.offset),
                    out: scratch.h1,
                    t: UInt32(t), d: UInt32(D), eps: eps)
                elementwise!.encodeResidualAdd(commandBuffer: cb,
                                               hidden: scratch.hidden,
                                               delta: scratch.h1,
                                               count: t * D)

                let scalarView = views.layerScalar!
                let scalarBits = scalarView.buffer.contents()
                    .advanced(by: Int(scalarView.offset))
                    .assumingMemoryBound(to: UInt16.self)[0]
                let layerScale = Quantization.bf16ToFloat(scalarBits)
                guard cfg.hasPerLayerInputs else {
                    elementwise!.encodeScale(
                        commandBuffer: cb,
                        values: scratch.hidden,
                        count: t * D,
                        scale: layerScale)
                    continue
                }

                let plePlan = PerLayerEmbeddingPlan(config: cfg)
                let pleNorm = model.perLayerProjectionNorm
                perLayerEmbeddingKernel!.encodePrepareLayer(
                    commandBuffer: cb,
                    identity: scratch.pleIdentity,
                    context: scratch.pleContext,
                    normWeight: pleNorm.buffer,
                    normWeightOffset: Int(pleNorm.offset),
                    out: scratch.pleLayer,
                    rows: t,
                    packedWidth: plePlan.packedWidth,
                    pleWidth: plePlan.perLayerHiddenSize,
                    layer: L,
                    contextScale: plePlan.contextProjectionScale,
                    combinedScale: plePlan.combinedScale,
                    eps: eps)
                encodeInt4Projection(
                    commandBuffer: cb,
                    family: .shared,
                    weights: views.perLayerInputGate!,
                    x: scratch.hidden,
                    y: scratch.pleGate,
                    rows: plePlan.perLayerHiddenSize,
                    columns: D,
                    tokenCount: t,
                    xStrideElements: D,
                    yStrideElements: plePlan.perLayerHiddenSize)
                elementwise!.encodeGELUMul(
                    commandBuffer: cb,
                    gate: scratch.pleGate,
                    value: scratch.pleLayer,
                    out: scratch.pleGate,
                    count: t * plePlan.perLayerHiddenSize)
                encodeInt4Projection(
                    commandBuffer: cb,
                    family: .shared,
                    weights: views.perLayerProjection!,
                    x: scratch.pleGate,
                    y: scratch.h1,
                    rows: D,
                    columns: plePlan.perLayerHiddenSize,
                    tokenCount: t,
                    xStrideElements: plePlan.perLayerHiddenSize,
                    yStrideElements: D)
                prefillRMS.encodeBF16W(
                    commandBuffer: cb,
                    x: scratch.h1,
                    weight: views.postPerLayerInputNorm!.buffer,
                    weightOffset: Int(views.postPerLayerInputNorm!.offset),
                    out: scratch.h1,
                    t: UInt32(t), d: UInt32(D), eps: eps)
                elementwise!.encodeResidualAddScale(
                    commandBuffer: cb,
                    hidden: scratch.hidden,
                    delta: scratch.h1,
                    count: t * D,
                    scale: layerScale)
                continue
            }
            if cfg.ffnSandwichNorms {
                prefillPostAttention.encode(commandBuffer: cb,
                                                hidden: scratch.hidden,
                                                attn: scratch.h1,
                                                denseX: scratch.denseX,
                                                routedX: scratch.routedX,
                                                routerX: scratch.routerX,
                                                postAttentionWeight: views.postAttention.buffer,
                                                postAttentionWeightOffset: Int(views.postAttention.offset),
                                                preFFNWeight: views.preFFN!.buffer,
                                                preFFNWeightOffset: Int(views.preFFN!.offset),
                                                preFFN2Weight: views.preFFN2!.buffer,
                                                preFFN2WeightOffset: Int(views.preFFN2!.offset),
                                                queryCount: UInt32(t),
                                                d: UInt32(D),
                                                hiddenStrideElements: UInt32(D),
                                                attnStrideElements: UInt32(D),
                                                denseStrideElements: UInt32(D),
                                                routedStrideElements: UInt32(D),
                                                routerStrideElements: UInt32(D),
                                                eps: eps)
            } else {
                // Plain pre-norm residual block: hidden += attention branch,
                // then one post-attention norm feeds router, shared expert,
                // and routed phase 1 (routedX doubles as moeX).
                elementwise!.encodeResidualAdd(commandBuffer: cb,
                                               hidden: scratch.hidden,
                                               delta: scratch.h1,
                                               count: t * D)
                prefillRMS.encodeBF16W(commandBuffer: cb,
                                       x: scratch.hidden,
                                       weight: views.postAttention.buffer,
                                       weightOffset: Int(views.postAttention.offset),
                                       out: scratch.routedX,
                                       t: UInt32(t),
                                       d: UInt32(D),
                                       eps: eps)
            }
            let perExpertScale: (buffer: MTLBuffer, offset: Int)
            if cfg.routerScaled {
                let view = views.routerPerExpertScale!
                perExpertScale = (view.buffer, Int(view.offset))
            } else {
                perExpertScale = (onesPerExpertScale!, 0)
            }
            if cfg.usesSigmoidCorrectionRouter {
                let correction = try model.routerCorrectionBias(layer: L)
                prefillRouter.encodeMiniMaxBlock(
                        commandBuffer: cb,
                        weights: views.router!.buffer,
                        weightsOffset: Int(views.router!.offset),
                        scales: views.router!.buffer,
                        scalesOffset: Int(views.router!.scaleOffset),
                        biases: views.router!.buffer,
                        biasesOffset: Int(views.router!.biasOffset),
                        hidden: scratch.routedX,
                        effectiveScale: effectiveScaleBuffers[L],
                        correctionBias: correction.buffer,
                        correctionBiasOffset: Int(correction.offset),
                        outIndices: scratch.routeIDs,
                        outWeights: scratch.routeWeights,
                        queryCount: UInt32(t),
                        numExperts: UInt32(cfg.numExperts),
                        d: UInt32(D),
                        topK: UInt32(cfg.topKExperts),
                        hiddenStrideElements: UInt32(D))
            } else {
            prefillRouter.encodeGemma4Block(
                        commandBuffer: cb,
                        weights: views.router!.buffer,
                        weightsOffset: Int(views.router!.offset),
                        scales: views.router!.buffer,
                        scalesOffset: Int(views.router!.scaleOffset),
                        biases: views.router!.buffer,
                        biasesOffset: Int(views.router!.biasOffset),
                        hidden: cfg.ffnSandwichNorms ? scratch.routerX : scratch.routedX,
                        effectiveScale: effectiveScaleBuffers[L],
                        perExpertScale: perExpertScale.buffer,
                        perExpertScaleOffset: perExpertScale.offset,
                        outIndices: scratch.routeIDs,
                        outWeights: scratch.routeWeights,
                        queryCount: UInt32(t),
                        numExperts: UInt32(cfg.numExperts),
                        d: UInt32(D),
                        topK: UInt32(cfg.topKExperts),
                        hiddenStrideElements: UInt32(D))
            }

                    cb.commit()
                    try waitForCompletion(cb)

                    let routeCount = t * cfg.topKExperts
                    let idPtr = scratch.routeIDs.contents()
                        .bindMemory(to: UInt32.self, capacity: routeCount)
                    let weightPtr = scratch.routeWeights.contents()
                        .bindMemory(to: Float16.self, capacity: routeCount)
                    var routeIDs = [UInt32]()
                    routeIDs.reserveCapacity(routeCount)
                    var routeWeights = [Float16]()
                    routeWeights.reserveCapacity(routeCount)
                    for i in 0..<routeCount {
                        routeIDs.append(min(idPtr[i], UInt32(cfg.numExperts - 1)))
                        routeWeights.append(weightPtr[i])
                    }
                    let pairs = PrefillRouter.makeTokenExpertPairs(indices: routeIDs,
                                                                   weights: routeWeights,
                                                                   queryCount: t,
                                                                   topK: cfg.topKExperts)
                    let schedulerConfig = Self.prefillRoutedTileSchedulerConfig
                    let routeTileExpertCount: Int
                    if let slotCount = model.routedExpertCacheSlotCount(layer: L) {
                        guard schedulerConfig.fitsSlotBudget(slotCount: slotCount) else {
                            throw PrefillError.chunkedUnsupported(
                                "prefill routed tile depth \(schedulerConfig.maxPendingDepth) with \(schedulerConfig.tileExperts) experts/tile needs \((schedulerConfig.maxPendingDepth + 1) * schedulerConfig.tileExperts) slots, has \(slotCount)")
                        }
                        routeTileExpertCount = min(schedulerConfig.tileExperts, slotCount)
                    } else {
                        routeTileExpertCount = schedulerConfig.tileExperts
                    }
                    let routes = try PrefillMoEGrouping.groupTokenExpertPairs(
                        pairs,
                        queryCount: t,
                        topK: cfg.topKExperts,
                        numExperts: cfg.numExperts,
                        tileExpertCount: routeTileExpertCount,
                        expertSortKeys: model.routedExpertPhysicalOffsets(layer: L))

                    guard let sharedCB = makePrefillCommandBuffer() else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    if let sharedProj = sharedExpertProjections[L] {
                    try prefillSharedExpert.encodeBlock(commandBuffer: sharedCB,
                                                        x: cfg.ffnSandwichNorms
                                                            ? scratch.denseX
                                                            : scratch.routedX,
                                                        y: scratch.h1,
                                                        gate: sharedProj.gate,
                                                        up: sharedProj.up,
                                                        down: sharedProj.down,
                                                        scratchGate: scratch.sharedGateScratch,
                                                        scratchUp: scratch.sharedUpScratch,
                                                        scratchAct: scratch.sharedActScratch,
                                                        queryCount: t,
                                                        d: D,
                                                        intermediate: cfg.ffnIntermediateSize(layer: L),
                                                        xStrideElements: D,
                                                        yStrideElements: D)
                    if cfg.ffnSandwichNorms {
                        let postF1 = sharedProj.postF1!
                        prefillRMS.encodeBF16W(commandBuffer: sharedCB,
                                               x: scratch.h1,
                                               weight: postF1.buffer,
                                               weightOffset: Int(postF1.offset),
                                               out: scratch.h1,
                                               t: UInt32(t),
                                               d: UInt32(D),
                                               eps: eps)
                    } else if cfg.sharedExpertGated {
                        // out = sigmoid(shared_expert_gate(moeX)) * shared_mlp(moeX),
                        // per chunk row.
                        let gateView = sharedProj.scalarGate!
                        let halfBytes = MemoryLayout<Float16>.stride
                        for row in 0..<t {
                            int8ScalarGate!.encode(
                                commandBuffer: sharedCB,
                                weights: gateView.buffer,
                                weightsOffset: Int(gateView.offset),
                                scales: gateView.buffer,
                                scalesOffset: Int(gateView.scaleOffset),
                                biases: gateView.buffer,
                                biasesOffset: Int(gateView.biasOffset),
                                x: scratch.routedX,
                                xOffset: row * D * halfBytes,
                                y: scratch.sharedScalarGate,
                                yOffset: row * halfBytes,
                                m: 1, n: UInt32(D))
                        }
                        for row in 0..<t {
                            elementwise!.encodeSigmoidScalarMul(
                                commandBuffer: sharedCB,
                                y: scratch.h1,
                                yOffset: row * D * halfBytes,
                                gate: scratch.sharedScalarGate,
                                gateOffset: row * halfBytes,
                                count: D)
                        }
                    }
                    }
                    sharedCB.commit()
                    try waitForCompletion(sharedCB)

                    let metadata = try prefillGroupedMoE.makeStreamedMetadataBuffers(
                        device: ctx.device,
                        routes: routes)
                    let routedOffsets = model.routedExpertOffsets(layer: L)
                    struct PendingPrefillTile {
                        let tileIndex: Int
                        let commandBuffer: MTLCommandBuffer
                        let fetch: PrefillStreamedTileFetchResult
                        let argumentBuffer: PrefillStreamedTileArgumentBuffer
                    }
                    var pendingTiles: [PendingPrefillTile] = []
                    var tileLifetime = PrefillStreamedTileSlotLifetime()
                    func drainOldestPendingTile() throws {
                        guard !pendingTiles.isEmpty else { return }
                        let pending = pendingTiles.removeFirst()
                        try withExtendedLifetime((pending.fetch, pending.argumentBuffer)) {
                            try waitForCompletion(pending.commandBuffer)
                        }
                        if !pending.fetch.plannedMissSlots.isEmpty {
                            try tileLifetime.complete(tileIndex: pending.tileIndex)
                        }
                    }

                    let routedTileScheduler = PrefillRoutedTileScheduler(config: schedulerConfig)
                    for (tileIndex, tile) in routes.tiles.enumerated() {
                        let expertIDs = try PrefillStreamedTileBinding.expertIDs(
                            forTile: tileIndex,
                            routes: routes)
                        var plannedFetch: RoutedExpertFetchPlan?
                        if !pendingTiles.isEmpty {
                            let pendingAssignedSlots = pendingTiles.flatMap(\.fetch.plannedAssignedSlots)
                            if !pendingAssignedSlots.isEmpty {
                                let pendingSlots = Set(pendingAssignedSlots)
                                let plan = try model.planRoutedExpertsIfPossible(
                                    layer: L,
                                    experts: expertIDs,
                                    avoidingSlots: pendingSlots)
                                let decision = routedTileScheduler.decide(
                                    PrefillRoutedTileSchedulerInput(
                                        hasPendingTile: true,
                                        pendingDepth: pendingTiles.count,
                                        pendingAssignedSlots: pendingAssignedSlots,
                                        avoidingSlotPlanAvailable: plan != nil))
                                switch decision {
                                case .prefetchNext:
                                    guard let plan else {
                                        throw ModelError.indexCorrupt(
                                            detail: "routed tile scheduler requested missing plan")
                                    }
                                    plannedFetch = plan
                                case .drainBeforeIssue:
                                    try drainOldestPendingTile()
                                case .issueWithoutPending:
                                    throw ModelError.indexCorrupt(
                                        detail: "routed tile scheduler ignored pending tile")
                                }
                            } else {
                                let decision = routedTileScheduler.decide(
                                    PrefillRoutedTileSchedulerInput(
                                        hasPendingTile: true,
                                        pendingDepth: pendingTiles.count,
                                        pendingAssignedSlots: [],
                                        avoidingSlotPlanAvailable: false))
                                switch decision {
                                case .drainBeforeIssue:
                                    try drainOldestPendingTile()
                                case .issueWithoutPending, .prefetchNext:
                                    throw ModelError.indexCorrupt(
                                        detail: "routed tile scheduler failed to drain empty-slot pending tile")
                                }
                            }
                        } else {
                            let decision = routedTileScheduler.decide(
                                PrefillRoutedTileSchedulerInput(
                                    hasPendingTile: false,
                                    pendingAssignedSlots: [],
                                    avoidingSlotPlanAvailable: false))
                            switch decision {
                            case .issueWithoutPending:
                                break
                            case .prefetchNext, .drainBeforeIssue:
                                throw ModelError.indexCorrupt(
                                    detail: "routed tile scheduler requested pending action without pending tile")
                            }
                        }
                        let fetch = try await PrefillStreamedTileBinding.fetchBindingForTile(
                            model: model,
                            layer: L,
                            tileIndex: tileIndex,
                            routes: routes,
                            plannedFetch: plannedFetch,
                            avoidingSlots: Set(pendingTiles.flatMap(\.fetch.plannedAssignedSlots)))
                        recordSpeculativeFetch(layer: L, fetch: fetch)
                        try fetch.binding.validateCoversPairs(routes.sortedPairs,
                                                              pairStart: Int(tile.pairStart),
                                                              pairCount: Int(tile.pairCount))
                        if !fetch.plannedMissSlots.isEmpty {
                            try tileLifetime.begin(tileIndex: tileIndex,
                                                   plannedSlots: fetch.plannedMissSlots)
                        }
                        let argumentBuffer = try prefillGroupedMoE.makeStreamedArgumentBuffer(
                            device: ctx.device,
                            binding: fetch.binding)
                        let streamedParams = PrefillGroupedRoutedMoEStreamedParams(
                            pairStart: tile.pairStart,
                            pairCount: tile.pairCount,
                            d: UInt32(D),
                            routedIntermediate: UInt32(cfg.moeIntermediateSize),
                            topK: UInt32(cfg.topKExperts),
                            hiddenStrideElements: UInt32(D),
                            binding: fetch.binding,
                            offsets: routedOffsets)
                        guard let tileCB = makePrefillCommandBuffer() else {
                            throw ModelError.residentBufferWrapFailed
                        }
                        _ = prefillGroupedMoE.encodeStreamedBatched(
                            commandBuffer: tileCB,
                            hidden: scratch.routedX,
                            sortedPairs: metadata.sortedPairs,
                            routePartials: scratch.routePartials,
                            gateUpActScratch: scratch.routedGateUpActScratch,
                            downScratch: scratch.routedDownScratch,
                            argumentBuffer: argumentBuffer,
                            binding: fetch.binding,
                            params: streamedParams,
                            pairMicrobatchRows: scratch.layout.routedPairMicrobatchRows)
                        tileCB.commit()
                        pendingTiles.append(PendingPrefillTile(tileIndex: tileIndex,
                                                               commandBuffer: tileCB,
                                                               fetch: fetch,
                                                               argumentBuffer: argumentBuffer))
                        while pendingTiles.count > schedulerConfig.maxPendingDepth {
                            try drainOldestPendingTile()
                        }
                    }
                    while !pendingTiles.isEmpty {
                        try drainOldestPendingTile()
                    }
                    guard let tailCB = makePrefillCommandBuffer() else {
                        throw ModelError.residentBufferWrapFailed
                    }
                    prefillMoE.encodeReduceTokenMajor(commandBuffer: tailCB,
                                                      routePartials: scratch.routePartials,
                                                      routeWeights: scratch.routeWeights,
                                                      h2: scratch.h2,
                                                      queryCount: UInt32(t),
                                                      topK: UInt32(cfg.topKExperts),
                                                      d: UInt32(D))
                    if cfg.ffnSandwichNorms {
                        let layerScalarView = views.layerScalar!
                        let scalarBits = layerScalarView.buffer.contents()
                            .advanced(by: Int(layerScalarView.offset))
                            .assumingMemoryBound(to: UInt16.self)[0]
                        prefillLayerTail.encode(commandBuffer: tailCB,
                                                h2: scratch.h2,
                                                h1: scratch.h1,
                                                hidden: scratch.hidden,
                                                postFFN2Weight: views.postFFN2!.buffer,
                                                postFFN2WeightOffset: Int(views.postFFN2!.offset),
                                                postFFNWeight: views.postFFN!.buffer,
                                                postFFNWeightOffset: Int(views.postFFN!.offset),
                                                queryCount: UInt32(t),
                                                d: UInt32(D),
                                                h2StrideElements: UInt32(D),
                                                h1StrideElements: UInt32(D),
                                                hiddenStrideElements: UInt32(D),
                                                eps: eps,
                                                layerScalar: Quantization.bf16ToFloat(scalarBits))
                    } else {
                        // Plain pre-norm tail: hidden += gated shared branch
                        // + routed branch.
                        if cfg.hasSharedExpert {
                            elementwise!.encodeResidualAdd(commandBuffer: tailCB,
                                                           hidden: scratch.hidden,
                                                           delta: scratch.h1,
                                                           count: t * D)
                        }
                        elementwise!.encodeResidualAdd(commandBuffer: tailCB,
                                                       hidden: scratch.hidden,
                                                       delta: scratch.h2,
                                                       count: t * D)
                    }
                    tailCB.commit()
                    try withExtendedLifetime(metadata) {
                        try waitForCompletion(tailCB)
                    }
                    if L + 1 < cfg.numLayers {
                        guard let nextCB = makePrefillCommandBuffer() else {
                            throw ModelError.residentBufferWrapFailed
                        }
                        cb = nextCB
                    }
                    continue
        }

        if cfg.feedForwardKind == .dense {
            cb.commit()
            try waitForCompletion(cb)
        }

        if writeFinalHead {
            let finalNorm = model.finalNorm
            let lm = model.lmHead
            guard let finalCB = makePrefillCommandBuffer() else {
                throw ModelError.residentBufferWrapFailed
            }
            if let speculativeTargetTokens {
                fusionHead.encodeGreedyDecodeRows(
                    commandBuffer: finalCB,
                    hidden: scratch.hidden,
                    hiddenStride: D,
                    rowCount: t,
                    normWeight: finalNorm.buffer,
                    normOffset: Int(finalNorm.offset),
                    weights: lm.buffer,
                    weightsOffset: Int(lm.offset),
                    scales: lm.buffer,
                    scalesOffset: Int(lm.scaleOffset),
                    biases: lm.buffer,
                    biasesOffset: Int(lm.biasOffset),
                    outTokens: speculativeTargetTokens,
                    d: UInt32(D),
                    vocab: UInt32(cfg.vocabSize),
                    rmsEps: eps)
            } else if outputMode == .greedyIfAvailable, useFusedGreedyHead {
                fusionHead.encodeGreedyDecode(
                    commandBuffer: finalCB,
                    hidden: scratch.hidden,
                    hiddenOffset: (t - 1) * D * MemoryLayout<Float16>.stride,
                    normWeight: finalNorm.buffer,
                    normOffset: Int(finalNorm.offset),
                    weights: lm.buffer,
                    weightsOffset: Int(lm.offset),
                    scales: lm.buffer,
                    scalesOffset: Int(lm.scaleOffset),
                    biases: lm.buffer,
                    biasesOffset: Int(lm.biasOffset),
                    outToken: greedyTokenBuf,
                    d: UInt32(D),
                    vocab: UInt32(cfg.vocabSize),
                    rmsEps: eps)
            } else {
                prefillFinalRowHead.encodeLogits(commandBuffer: finalCB,
                                                 hiddenBlock: scratch.hidden,
                                                 row: t - 1,
                                                 rowStrideElements: D,
                                                 normWeight: finalNorm.buffer,
                                                 normWeightOffset: Int(finalNorm.offset),
                                                 weights: lm.buffer,
                                                 weightsOffset: Int(lm.offset),
                                                 scales: lm.buffer,
                                                 scalesOffset: Int(lm.scaleOffset),
                                                 biases: lm.buffer,
                                                 biasesOffset: Int(lm.biasOffset),
                                                 logits: logits,
                                                 d: UInt32(D),
                                                 vocab: UInt32(cfg.vocabSize),
                                                 rmsEps: eps)
            }
            finalCB.commit()
            try waitForCompletion(finalCB)
            if speculativeTargetTokens == nil,
               outputMode == .greedyIfAvailable, useFusedGreedyHead {
                lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
            }
        }

        kv?.advance(by: tokens.count)
        prefillChunkState.markCommitted()
    }

    private func produceToken(token: Int32,
                              position: Int,
                              into logits: MTLBuffer,
                              emitHead: Bool,
                              outputMode: PrefillOutputMode,
                              epilogue: ((MTLCommandBuffer) -> Void)? = nil) async throws {
        let kvPosition = kv?.position ?? 0
        guard kvPosition == position else {
            throw PrefillError.prefillCursorMismatch(
                "produce cursor \(kvPosition) != position \(position)")
        }
        guard position < maxContext else {
            throw PrefillError.prefillCursorMismatch(
                "produce position \(position) exceeds maxContext \(maxContext)")
        }
        let D    = UInt32(cfg.hiddenSize)
        let FmoE = UInt32(cfg.moeIntermediateSize)
        let eps: Float = 1e-6
        let embedOutScale = cfg.embeddingScaledBySqrtHidden
            ? Float(cfg.hiddenSize).squareRoot()
            : 1.0
        struct PendingRoutedCommand {
            let cb: MTLCommandBuffer
            let sharedCB: MTLCommandBuffer?
            let phase1HitCB: MTLCommandBuffer?
            let encodeAndCommitNanos: UInt64
        }
        var pendingRoutedCommand: PendingRoutedCommand?

        func finishPendingRoutedCommand(_ pending: PendingRoutedCommand,
                                        waitIfNeeded: Bool) throws {
            if waitIfNeeded {
                if let sharedCB = pending.sharedCB {
                    waitUntilCompleted(sharedCB)
                }
                if let phase1HitCB = pending.phase1HitCB {
                    waitUntilCompleted(phase1HitCB)
                }
                waitUntilCompleted(pending.cb)
            }
            if let sharedCB = pending.sharedCB {
                try checkCommandBufferError(sharedCB.error)
            }
            if let phase1HitCB = pending.phase1HitCB {
                try checkCommandBufferError(phase1HitCB.error)
            }
            try checkCommandBufferError(pending.cb.error)
            totalGPURoutedNanos &+= UInt64(max(0, (pending.cb.gpuEndTime - pending.cb.gpuStartTime) * 1e9))
            if let sharedCB = pending.sharedCB {
                totalGPUSharedNanos &+= UInt64(max(0, (sharedCB.gpuEndTime - sharedCB.gpuStartTime) * 1e9))
            }
            totalCb2Nanos &+= pending.encodeAndCommitNanos
        }

        func writeActiveSlots(_ slots: [UInt32], into buffer: MTLBuffer) {
            let ptr = buffer.contents().assumingMemoryBound(to: UInt32.self)
            for i in 0..<slots.count { ptr[i] = slots[i] }
        }

        // Embed lookup + sqrt(H) fused. Keep these encodes on the first
        // layer's command buffer: nothing on the CPU consumes `hidden` or the
        // optional PLE scratch before that layer, so the old standalone
        // command-buffer commit and synchronous wait were pure token overhead.
        let emb = model.embedding
        var firstLayerCommandBuffer: MTLCommandBuffer? = ctx.queue.makeCommandBuffer()!
        embedInt4.encode(commandBuffer: firstLayerCommandBuffer!,
                         table:  emb.buffer, tableOffset:  Int(emb.offset),
                         scales: emb.buffer, scalesOffset: Int(emb.scaleOffset),
                         biases: emb.buffer, biasesOffset: Int(emb.biasOffset),
                         out: hidden,
                         tokenId: UInt32(bitPattern: token),
                         d: D,
                         outScale: embedOutScale)
        if cfg.hasPerLayerInputs {
            let plan = PerLayerEmbeddingPlan(config: cfg)
            let identity = model.perLayerEmbedding
            embedInt4.encode(
                commandBuffer: firstLayerCommandBuffer!,
                table: identity.buffer, tableOffset: Int(identity.offset),
                scales: identity.buffer, scalesOffset: Int(identity.scaleOffset),
                biases: identity.buffer, biasesOffset: Int(identity.biasOffset),
                out: pleIdentity!,
                tokenId: UInt32(bitPattern: token),
                d: UInt32(plan.packedWidth),
                outScale: plan.tokenIdentityScale)
            let projection = model.perLayerModelProjection
            int4.encode(
                commandBuffer: firstLayerCommandBuffer!,
                weights: projection.buffer, weightsOffset: Int(projection.offset),
                scales: projection.buffer, scalesOffset: Int(projection.scaleOffset),
                biases: projection.buffer, biasesOffset: Int(projection.biasOffset),
                x: hidden, y: pleContext!,
                m: UInt32(plan.packedWidth), n: D)
        }

        for L in 0..<cfg.numLayers {
            let isLinear = cfg.layerIsLinear(L)
            let isFull = cfg.fullAttentionLayerMask[L] == 1
            let headDimL = isFull ? cfg.fullHeadDim : cfg.headDim
            let numKVL   = isFull ? cfg.numFullKVHeads : cfg.numKVHeads
            let qDim     = UInt32(cfg.numHeads * headDimL)
            let kvDim    = UInt32(numKVL * headDimL)
            let seqLen   = UInt32(position + 1)

            let inNorm   = try model.inputNorm(layer: L)
            let postAttn = try model.postAttnNorm(layer: L)
            let sharedProj = sharedExpertProjections[L]

            let tCb1Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            // Everything up to and including the router runs in a single CB:
            // the only reason to break is the CPU readback of router indices
            // needed to issue I/O for the routed-expert blobs.
            let cb: MTLCommandBuffer
            if let first = firstLayerCommandBuffer {
                cb = first
                firstLayerCommandBuffer = nil
            } else {
                cb = ctx.queue.makeCommandBuffer()!
            }
            rms.encodeBF16W(commandBuffer: cb,
                            x: hidden,
                            weight: inNorm.buffer, weightOffset: Int(inNorm.offset),
                            out: normed,
                            d: D, eps: eps)

            if isLinear {
                // Gated-DeltaNet linear attention: no KV slots, no RoPE — a
                // fixed-size recurrent state updated in place.
                try encodeLinearAttentionDecode(cb, layer: L)
            } else if cfg.attnOutputGate {
                // Qwen full attention: packed [query ; gate] q_proj, real
                // v_proj, no V norm, NeoX sub-dim RoPE, sigmoid output gate.
                try encodeGatedFullAttentionDecode(cb, layer: L,
                                                   position: position,
                                                   seqLen: seqLen)
            } else if let kvSourceLayer = cfg.kvSourceLayer(for: L) {
                guard let kv, let rope else {
                    preconditionFailure("shared-KV attention requires KV and RoPE kernels")
                }
                let q = try model.qProj(layer: L)
                let o = try model.oProj(layer: L)
                let qNorm = try model.qNorm(layer: L)
                int4.encode(commandBuffer: cb,
                            weights: q.buffer, weightsOffset: Int(q.offset),
                            scales: q.buffer, scalesOffset: Int(q.scaleOffset),
                            biases: q.buffer, biasesOffset: Int(q.biasOffset),
                            x: normed, y: qScratch, m: qDim, n: D)
                rms.encodeBF16WPerHead(commandBuffer: cb,
                                       x: qScratch,
                                       weight: qNorm.buffer,
                                       weightOffset: Int(qNorm.offset),
                                       out: qScratch,
                                       headDim: UInt32(headDimL),
                                       numHeads: cfg.numHeads,
                                       eps: eps)
                let rotated = isFull
                    ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                    : UInt32(headDimL / 2)
                if isFull {
                    rope.encodeProportionalNeox(
                        commandBuffer: cb, data: qScratch,
                        position: UInt32(position), headDim: UInt32(headDimL),
                        numHeads: UInt32(cfg.numHeads), rotatedPairs: rotated,
                        theta: Float(cfg.fullRopeTheta))
                } else {
                    rope.encodeDefaultNeox(
                        commandBuffer: cb, data: qScratch,
                        position: UInt32(position), headDim: UInt32(headDimL),
                        numHeads: UInt32(cfg.numHeads), theta: Float(cfg.ropeTheta))
                }
                let keyBuffer = kv.keyBuffer(layer: kvSourceLayer,
                                             validTokenCount: position + 1)
                let valueBuffer = kv.valueBuffer(layer: kvSourceLayer,
                                                 validTokenCount: position + 1)
                if isFull {
                    attention.encodeFull(commandBuffer: cb,
                                         q: qScratch,
                                         k: keyBuffer, kOffset: 0,
                                         v: valueBuffer, vOffset: 0,
                                         out: attnOut,
                                         headDim: UInt32(headDimL),
                                         numQHeads: UInt32(cfg.numHeads),
                                         numKVHeads: UInt32(numKVL),
                                         seqLen: seqLen,
                                         scale: Float(cfg.attentionScale))
                } else {
                    let ringCapacity = kv.ringCapacity(layer: kvSourceLayer)
                    let activeRingCapacity = ringCapacity > 0 && Int(seqLen) > ringCapacity
                        ? UInt32(ringCapacity) : 0
                    attention.encodeSWA(commandBuffer: cb,
                                        q: qScratch,
                                        k: keyBuffer, kOffset: 0,
                                        v: valueBuffer, vOffset: 0,
                                        out: attnOut,
                                        headDim: UInt32(headDimL),
                                        numQHeads: UInt32(cfg.numHeads),
                                        numKVHeads: UInt32(numKVL),
                                        seqLen: seqLen,
                                        window: UInt32(cfg.slidingWindow),
                                        scale: Float(cfg.attentionScale),
                                        ringCapacity: activeRingCapacity)
                }
                int4.encode(commandBuffer: cb,
                            weights: o.buffer, weightsOffset: Int(o.offset),
                            scales: o.buffer, scalesOffset: Int(o.scaleOffset),
                            biases: o.buffer, biasesOffset: Int(o.biasOffset),
                            x: attnOut, y: oOut, m: D, n: qDim)
            } else {
                let kSlot = kv?.kSlot(layer: L, position: position) ?? (buffer: kStage, offset: 0)
                let vSlot = kv?.vSlot(layer: L, position: position) ?? (buffer: vStage, offset: 0)
                let q     = try model.qProj(layer: L)
                let k     = try model.kProj(layer: L)
                // Under the K=V quirk full layers reuse k_proj; otherwise
                // v_proj is a real tensor.
                let vProj = (isFull && cfg.attentionKEqV) ? k : (try model.vProj(layer: L))
                let o     = try model.oProj(layer: L)
                let qNorm = try model.qNorm(layer: L)
                let kNorm = try model.kNorm(layer: L)

                fusedQKVGEMV.encode(commandBuffer: cb,
                                    qWeights: q.buffer, qWeightsOffset: Int(q.offset),
                                    qScales: q.buffer, qScalesOffset: Int(q.scaleOffset),
                                    qBiases: q.buffer, qBiasesOffset: Int(q.biasOffset),
                                    kWeights: k.buffer, kWeightsOffset: Int(k.offset),
                                    kScales: k.buffer, kScalesOffset: Int(k.scaleOffset),
                                    kBiases: k.buffer, kBiasesOffset: Int(k.biasOffset),
                                    vWeights: vProj.buffer, vWeightsOffset: Int(vProj.offset),
                                    vScales: vProj.buffer, vScalesOffset: Int(vProj.scaleOffset),
                                    vBiases: vProj.buffer, vBiasesOffset: Int(vProj.biasOffset),
                                    x: normed,
                                    qOut: qScratch,
                                    kOut: kSlot.buffer, kOutOffset: kSlot.offset,
                                    vOut: vSlot.buffer, vOutOffset: vSlot.offset,
                                    qRows: qDim,
                                    kvRows: kvDim,
                                    n: D)

                let rotated = isFull
                    ? UInt32(Double(cfg.fullHeadDim) * cfg.partialRotaryFactor / 2.0)
                    : UInt32(headDimL / 2)
                if cfg.usesProjectionWideQKNorm {
                    guard let rope else {
                        preconditionFailure("MiniMax attention requires RoPE kernels")
                    }
                    rms.encodeBF16W(commandBuffer: cb,
                                    x: qScratch,
                                    weight: qNorm.buffer,
                                    weightOffset: Int(qNorm.offset),
                                    out: qScratch,
                                    d: qDim, eps: eps)
                    rms.encodeBF16W(commandBuffer: cb,
                                    x: kSlot.buffer,
                                    xOffset: kSlot.offset,
                                    weight: kNorm.buffer,
                                    weightOffset: Int(kNorm.offset),
                                    out: kSlot.buffer,
                                    outOffset: kSlot.offset,
                                    d: kvDim, eps: eps)
                    let rotaryDim = UInt32(Double(headDimL) * cfg.partialRotaryFactor)
                    rope.encodeNeoxSubdim(
                        commandBuffer: cb, data: qScratch,
                        position: UInt32(position), headDim: UInt32(headDimL),
                        numHeads: UInt32(cfg.numHeads), rotaryDim: rotaryDim,
                        theta: Float(cfg.fullRopeTheta))
                    rope.encodeNeoxSubdim(
                        commandBuffer: cb, data: kSlot.buffer, dataOffset: kSlot.offset,
                        position: UInt32(position), headDim: UInt32(headDimL),
                        numHeads: UInt32(numKVL), rotaryDim: rotaryDim,
                        theta: Float(cfg.fullRopeTheta))
                } else {
                fusedQKVEpilogue.encode(commandBuffer: cb,
                                        q: qScratch,
                                        k: kSlot.buffer,
                                        kOffset: kSlot.offset,
                                        v: vSlot.buffer,
                                        vOffset: vSlot.offset,
                                        qWeight: qNorm.buffer,
                                        qWeightOffset: Int(qNorm.offset),
                                        kWeight: kNorm.buffer,
                                        kWeightOffset: Int(kNorm.offset),
                                        headDim: UInt32(headDimL),
                                        numQHeads: UInt32(cfg.numHeads),
                                        numKVHeads: UInt32(numKVL),
                                        position: UInt32(position),
                                        theta: isFull ? Float(cfg.fullRopeTheta) : Float(cfg.ropeTheta),
                                        rotatedPairs: rotated,
                                        eps: eps)
                }

                guard kv != nil else {
                    preconditionFailure("FP16 attention requires an FP16 KV cache")
                }
                if isFull {
                    attention.encodeFull(commandBuffer: cb,
                                         q: qScratch,
                                         k: kSlot.buffer, kOffset: 0,
                                         v: vSlot.buffer, vOffset: 0,
                                         out: attnOut,
                                         headDim: UInt32(headDimL),
                                         numQHeads: UInt32(cfg.numHeads),
                                         numKVHeads: UInt32(numKVL),
                                         seqLen: seqLen,
                                         scale: Float(cfg.attentionScale))
                } else {
                    let ringCapacity = kv?.ringCapacity(layer: L) ?? 0
                    let activeRingCapacity = ringCapacity > 0 && Int(seqLen) > ringCapacity
                        ? UInt32(ringCapacity)
                        : 0
                    attention.encodeSWA(commandBuffer: cb,
                                        q: qScratch,
                                        k: kSlot.buffer, kOffset: 0,
                                        v: vSlot.buffer, vOffset: 0,
                                        out: attnOut,
                                        headDim: UInt32(headDimL),
                                        numQHeads: UInt32(cfg.numHeads),
                                        numKVHeads: UInt32(numKVL),
                                        seqLen: seqLen,
                                        window: UInt32(cfg.slidingWindow),
                                        scale: Float(cfg.attentionScale),
                                        ringCapacity: activeRingCapacity)
                }
                int4.encode(commandBuffer: cb,
                            weights: o.buffer, weightsOffset: Int(o.offset),
                            scales:  o.buffer, scalesOffset:  Int(o.scaleOffset),
                            biases:  o.buffer, biasesOffset:  Int(o.biasOffset),
                            x: attnOut, y: oOut, m: D, n: qDim)
            }

            if cfg.feedForwardKind == .dense {
                try encodeDenseGemmaLayerDecode(
                    cb, layer: L, postAttention: postAttn,
                    projections: sharedProj!, d: D, eps: eps)
                // A dense layer produces nothing the CPU reads, so the whole
                // token stays on one command buffer: the layer's work is
                // handed to the next iteration instead of being committed and
                // waited on here. Each round trip cost ~130 us of wall time
                // against ~500 us of GPU work, so 35 of them per token were
                // most of the gap between GPU time and tokens per second.
                firstLayerCommandBuffer = cb
                totalCb1Nanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb1Start
                continue
            }

            if cfg.ffnSandwichNorms {
                let preFFN   = try model.preFFN(layer: L)
                let preFFN2  = try model.preFFN2(layer: L)
                fusedPostAttentionSetup.encode(commandBuffer: cb,
                                               hidden: hidden,
                                               attn: oOut,
                                               denseX: denseX,
                                               routedX: routedX,
                                               routerX: routerInput,
                                               postAttentionWeight: postAttn.buffer,
                                               postAttentionWeightOffset: Int(postAttn.offset),
                                               preFFNWeight: preFFN.buffer,
                                               preFFNWeightOffset: Int(preFFN.offset),
                                               preFFN2Weight: preFFN2.buffer,
                                               preFFN2WeightOffset: Int(preFFN2.offset),
                                               d: D,
                                               eps: eps)
            } else {
                // Plain pre-norm residual block: hidden += attention branch,
                // then one post-attention norm feeds router, shared expert,
                // and routed phase 1 (routedX doubles as moeX).
                elementwise!.encodeResidualAdd(commandBuffer: cb,
                                               hidden: hidden,
                                               delta: oOut,
                                               count: cfg.hiddenSize)
                rms.encodeBF16W(commandBuffer: cb,
                                x: hidden,
                                weight: postAttn.buffer,
                                weightOffset: Int(postAttn.offset),
                                out: routedX,
                                d: D, eps: eps)
            }

            let moe = self.moe!
            let routerW = try model.router(layer: L)
            let perExpertScale: (buffer: MTLBuffer, offset: Int)
            if cfg.routerScaled {
                let view = try model.routerPerExpertScale(layer: L)
                perExpertScale = (view.buffer, Int(view.offset))
            } else {
                perExpertScale = (onesPerExpertScale!, 0)
            }

            if cfg.usesSigmoidCorrectionRouter {
                let correction = try model.routerCorrectionBias(layer: L)
                moe.encodeRouterMiniMax(commandBuffer: cb,
                    weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                    scales: routerW.buffer, scalesOffset: Int(routerW.scaleOffset),
                    biases: routerW.buffer, biasesOffset: Int(routerW.biasOffset),
                    hidden: routedX,
                    effectiveScale: effectiveScaleBuffers[L],
                    correctionBias: correction.buffer,
                    correctionBiasOffset: Int(correction.offset),
                    outIndices: outIndices, outWeights: outWeights,
                    numExperts: UInt32(cfg.numExperts), d: D,
                    topK: UInt32(cfg.topKExperts))
            } else {
            moe.encodeRouterGemma4(commandBuffer: cb,
                weights: routerW.buffer, weightsOffset: Int(routerW.offset),
                scales:  routerW.buffer, scalesOffset:  Int(routerW.scaleOffset),
                biases:  routerW.buffer, biasesOffset:  Int(routerW.biasOffset),
                hidden: cfg.ffnSandwichNorms ? routerInput : routedX,
                effectiveScale: effectiveScaleBuffers[L],
                perExpertScale: perExpertScale.buffer,
                perExpertScaleOffset: perExpertScale.offset,
                outIndices: outIndices, outWeights: outWeights,
                numExperts: UInt32(cfg.numExperts), d: D, topK: UInt32(cfg.topKExperts))
            }
            cb.commit()
            let tWait = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            waitUntilCompleted(cb)
            let waitNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tWait
            if let pending = pendingRoutedCommand {
                try finishPendingRoutedCommand(pending, waitIfNeeded: false)
                pendingRoutedCommand = nil
            }
            try checkCommandBufferError(cb.error)
            totalGPULayerNanos &+= UInt64(max(0, (cb.gpuEndTime - cb.gpuStartTime) * 1e9))
            totalGPULayerCommandBuffers &+= 1
            totalCb1Nanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb1Start - waitNanos

            // CPU readback to fetch routed-expert blobs from disk.
            let idxPtr = outIndices.contents().bindMemory(to: UInt32.self,
                                                          capacity: cfg.topKExperts)
            var experts = [Int](repeating: 0, count: cfg.topKExperts)
            for i in 0..<cfg.topKExperts {
                experts[i] = min(Int(idxPtr[i]), cfg.numExperts - 1)
            }

            let routedOffsets = model.routedExpertOffsets(layer: L)
            let topK = UInt32(cfg.topKExperts)
            let canPlanPhase1HitSplit =
                cfg.topKExperts <= MoE.maxStreamedExperts
            let plannedFetch = canPlanPhase1HitSplit
                ? try model.planRoutedExperts(layer: L, experts: experts)
                : nil
            recordDecodeExpertFetch(layer: L, plan: plannedFetch)
            var phase1HitCB: MTLCommandBuffer?
            var phase1HitSplitArgBuf: MTLBuffer?
            var phase1HitSplitRoutedBufs: [MTLBuffer] = []
            var phase1HitSlots: [UInt32] = []
            var phase1MissSlots: [UInt32] = []

            if let plan = plannedFetch {
                phase1HitSlots.reserveCapacity(cfg.topKExperts - plan.misses.count)
                for slot in 0..<cfg.topKExperts where !plan.misses.contains(slot) {
                    phase1HitSlots.append(UInt32(slot))
                }
                phase1MissSlots = plan.misses.map { UInt32($0) }
            }
            func encodeRoutedPhase1Full(
                _ cb: MTLCommandBuffer,
                argBuf: MTLBuffer,
                routedBufs: [MTLBuffer]
            ) {
                moe.encodeRoutedPersistentPhase1U16Load(commandBuffer: cb,
                                                        routedArgBuffer: argBuf,
                                                        routedBlobs: routedBufs,
                                                        routedOffsets: routedOffsets,
                                                        x: routedX,
                                                        acts: moeActs,
                                                        d: D,
                                                        f: FmoE,
                                                        topK: topK)
            }

            func encodeRoutedPhase1Subset(
                _ cb: MTLCommandBuffer,
                argBuf: MTLBuffer,
                routedBufs: [MTLBuffer],
                activeSlots: MTLBuffer,
                activeSlotIndices: [UInt32],
                activeCount: UInt32
            ) {
                moe.encodeRoutedPersistentPhase1SubsetU16Load(
                    commandBuffer: cb,
                    routedArgBuffer: argBuf,
                    routedBlobs: routedBufs,
                    routedOffsets: routedOffsets,
                    x: routedX,
                    acts: moeActs,
                    activeSlots: activeSlots,
                    activeSlotIndices: activeSlotIndices,
                    activeCount: activeCount,
                    d: D,
                    f: FmoE,
                    topK: topK)
            }

            if let plan = plannedFetch,
               plan.hits > 0,
               !plan.misses.isEmpty {
                let plannedBlobs = try model.routedExpertBuffers(for: plan)
                phase1HitSplitRoutedBufs = plannedBlobs.map { $0.buffer }
                phase1HitSplitArgBuf = moe.makeRoutedArgumentBuffer(
                    routedBlobs: phase1HitSplitRoutedBufs,
                    topK: topK)
                if let argBuf = phase1HitSplitArgBuf, plan.hits > 0, !plan.misses.isEmpty {
                    writeActiveSlots(phase1HitSlots, into: moeHitActiveSlots)
                    let cb = ctx.queue.makeCommandBuffer()!
                    encodeRoutedPhase1Subset(
                        cb,
                        argBuf: argBuf,
                        routedBufs: phase1HitSplitRoutedBufs,
                        activeSlots: moeHitActiveSlots,
                        activeSlotIndices: phase1HitSlots,
                        activeCount: UInt32(phase1HitSlots.count))
                    phase1HitCB = cb
                }
            }

            // The shared dense MLP depends only on its normed input, not on
            // the routed experts. Commit it without waiting so its GPU work
            // overlaps the routed-expert pread. The routed CB follows it on
            // the same queue, so the combine sees h1Buf.
            let sharedCB = sharedProj.map { _ in ctx.queue.makeCommandBuffer()! }
            if let sharedCB, let sharedProj {
            try! shared.encode(commandBuffer: sharedCB,
                               x: cfg.ffnSandwichNorms ? denseX : routedX,
                               gate: sharedProj.gate,
                               up: sharedProj.up,
                               down: sharedProj.down,
                               y: h1Buf,
                               scratchGate: denseScratchGate,
                               scratchUp: denseScratchUp,
                               scratchAct: denseScratchAct)
            if cfg.ffnSandwichNorms {
                let postF1 = sharedProj.postF1!
                rms.encodeBF16W(commandBuffer: sharedCB, x: h1Buf,
                                weight: postF1.buffer,
                                weightOffset: Int(postF1.offset),
                                out: h1Buf, d: D, eps: eps)
            } else if cfg.sharedExpertGated {
                // out = sigmoid(shared_expert_gate(moeX)) * shared_mlp(moeX)
                let gateView = sharedProj.scalarGate!
                int8ScalarGate!.encode(commandBuffer: sharedCB,
                                       weights: gateView.buffer,
                                       weightsOffset: Int(gateView.offset),
                                       scales: gateView.buffer,
                                       scalesOffset: Int(gateView.scaleOffset),
                                       biases: gateView.buffer,
                                       biasesOffset: Int(gateView.biasOffset),
                                       x: routedX,
                                       y: sharedScalarGateBuf!,
                                       m: 1, n: D)
                elementwise!.encodeSigmoidScalarMul(commandBuffer: sharedCB,
                                                    y: h1Buf,
                                                    gate: sharedScalarGateBuf!,
                                                    count: cfg.hiddenSize)
            }
            sharedCB.commit()
            }
            if let cb = phase1HitCB {
                cb.commit()
            }
            if rdadviseEnabled && rdadvisePolicyMode != .off {
                let requestedMisses = plannedFetch?.misses.count ?? experts.count
                let estimatedAdviceBytes = try model.routedExpertAdviceByteEstimate(
                    layer: L,
                    missCount: requestedMisses)
                if let skipped = shouldSkipRDAdvice(position: position,
                                                    requestedMisses: requestedMisses,
                                                    estimatedBytes: estimatedAdviceBytes,
                                                    canOverlapUsefulGPUWork: true) {
                    recordRDAdvice(skipped, wallNanos: 0)
                } else {
                    let tAdvice = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                    let result: ExpertIOAdviceResult
                    if let plannedFetch {
                        result = try model.adviseRoutedExperts(plan: plannedFetch)
                    } else {
                        result = try model.adviseRoutedExperts(layer: L, experts: experts)
                    }
                    let wallNanos = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tAdvice
                    recordRDAdvice(result, wallNanos: wallNanos)
                    updateRDAdvicePolicy(after: result, position: position)
                }
            }

            // Routed-expert pread — overlaps the shared MLP GPU work above.
            let tIoStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let blobs: [TensorView]
            if let plannedFetch {
                blobs = try await model.fetchRoutedExperts(plan: plannedFetch)
            } else {
                blobs = try await model.fetchRoutedExperts(layer: L, experts: experts)
            }
            let layerIo = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tIoStart
            totalIoNanos &+= layerIo
            let routedBufs = blobs.map { $0.buffer }
            let tCb2Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let gTail: (MTLCommandBuffer) -> Void
            if cfg.ffnSandwichNorms {
                let postF2 = try model.postFFN2(layer: L)
                let postF = try model.postFFN(layer: L)
                let layerScalarView = try model.layerScalar(layer: L)
                let scalarPtr = layerScalarView.buffer.contents()
                    .advanced(by: Int(layerScalarView.offset))
                    .assumingMemoryBound(to: UInt16.self)
                let layerScalar = Quantization.bf16ToFloat(scalarPtr[0])
                gTail = { [self] cb in
                    fusedTail.encode(commandBuffer: cb,
                                     h2: h2Buf,
                                     h1: h1Buf,
                                     hidden: hidden,
                                     postFFN2Weight: postF2.buffer,
                                     postFFN2WeightOffset: Int(postF2.offset),
                                     postFFNWeight: postF.buffer,
                                     postFFNWeightOffset: Int(postF.offset),
                                     d: D,
                                     eps: eps,
                                     layerScalar: layerScalar)
                }
            } else {
                // The phase-2 reduce already folded the shared branch (h1Buf
                // as its residual); the tail is a plain residual add.
                gTail = { [self] cb in
                    elementwise!.encodeResidualAdd(commandBuffer: cb,
                                                   hidden: hidden,
                                                   delta: h2Buf,
                                                   count: cfg.hiddenSize)
                }
            }
            let routedCB = ctx.queue.makeCommandBuffer()!
            let splitArgBuf = phase1HitCB != nil && !phase1MissSlots.isEmpty
                ? phase1HitSplitArgBuf
                : nil
            let argBuf = splitArgBuf ?? moe.makeReusedRoutedArgumentBuffer(
                routedBlobs: routedBufs,
                topK: topK)
            if splitArgBuf != nil {
                writeActiveSlots(phase1MissSlots, into: moeMissActiveSlots)
                encodeRoutedPhase1Subset(
                    routedCB,
                    argBuf: argBuf,
                    routedBufs: routedBufs,
                    activeSlots: moeMissActiveSlots,
                    activeSlotIndices: phase1MissSlots,
                    activeCount: UInt32(phase1MissSlots.count))
            } else {
                encodeRoutedPhase1Full(routedCB,
                                       argBuf: argBuf,
                                       routedBufs: routedBufs)
            }
            moe.encodeRoutedPersistentPhase2Reduce(commandBuffer: routedCB,
                                                   routedArgBuffer: argBuf,
                                                   routedBlobs: routedBufs,
                                                   routedOffsets: routedOffsets,
                                                   acts: moeActs,
                                                   routingWeights: outWeights,
                                                   residual: (!cfg.ffnSandwichNorms && cfg.hasSharedExpert)
                                                       ? h1Buf : zeroResidual,
                                                   y: h2Buf,
                                                   d: D,
                                                   f: FmoE,
                                                   topK: topK)
            gTail(routedCB)
            routedCB.commit()
            precondition(pendingRoutedCommand == nil,
                         "routed command-buffer pipeline drained before queuing the next layer")
            pendingRoutedCommand = PendingRoutedCommand(
                cb: routedCB,
                sharedCB: sharedCB,
                phase1HitCB: phase1HitCB,
                encodeAndCommitNanos: clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tCb2Start)
            continue
        }
        if let pending = pendingRoutedCommand {
            try finishPendingRoutedCommand(pending, waitIfNeeded: true)
            pendingRoutedCommand = nil
        }

        // The fused head skips the vocab buffer and leaves a greedy token in
        // greedyTokenBuf; the logits path writes the complete vector.
        let fNorm = model.finalNorm
        let lm    = model.lmHead
        let gFinalNorm: (MTLCommandBuffer) -> Void = { cb in
            self.rms.encodeBF16W(commandBuffer: cb, x: self.hidden,
                                 weight: fNorm.buffer, weightOffset: Int(fNorm.offset),
                                 out: self.normed, d: D, eps: eps)
        }
        let gLmHead: (MTLCommandBuffer) -> Void = { cb in
            self.int4.encode(commandBuffer: cb,
                             weights: lm.buffer, weightsOffset: Int(lm.offset),
                             scales:  lm.buffer, scalesOffset:  Int(lm.scaleOffset),
                             biases:  lm.buffer, biasesOffset:  Int(lm.biasOffset),
                             x: self.normed, y: logits, m: UInt32(self.cfg.vocabSize), n: D)
        }
        let gFusionHead: (MTLCommandBuffer) -> Void = { cb in
            self.fusionHead.encodeGreedyDecode(
                commandBuffer: cb,
                hidden: self.hidden,
                normWeight: fNorm.buffer, normOffset: Int(fNorm.offset),
                weights: lm.buffer, weightsOffset: Int(lm.offset),
                scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                outToken: self.greedyTokenBuf,
                d: D, vocab: UInt32(self.cfg.vocabSize),
                rmsEps: eps)
        }
        // A dense token carries its layers on `firstLayerCommandBuffer`; the
        // head joins them so the token costs exactly one commit and one wait.
        let pendingTokenCommandBuffer = firstLayerCommandBuffer
        firstLayerCommandBuffer = nil
        if emitHead {
            let useFusedHeadForThisToken = useFusedGreedyHead && outputMode == .greedyIfAvailable
            let tHead = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            let head = useFusedHeadForThisToken ? gFusionHead : { cb in
                gFinalNorm(cb)
                gLmHead(cb)
            }
            if let cb = pendingTokenCommandBuffer {
                head(cb)
                epilogue?(cb)
                cb.commit()
                waitUntilCompleted(cb)
                try checkCommandBufferError(cb.error)
                totalGPULayerNanos &+= UInt64(max(0, (cb.gpuEndTime - cb.gpuStartTime) * 1e9))
                totalGPULayerCommandBuffers &+= 1
            } else {
                try runSync(head)
            }
            if useFusedHeadForThisToken {
                totalHeadFusedNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
                lastGreedyToken = greedyTokenBuf.contents().load(as: UInt32.self)
            } else {
                totalHeadNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - tHead
            }
        } else if let cb = pendingTokenCommandBuffer {
            cb.commit()
            waitUntilCompleted(cb)
            try checkCommandBufferError(cb.error)
            totalGPULayerNanos &+= UInt64(max(0, (cb.gpuEndTime - cb.gpuStartTime) * 1e9))
            totalGPULayerCommandBuffers &+= 1
        }

        kv?.advance()
    }

    /// Dense Gemma 4 tail, including the per-layer embedding mapping:
    /// post-attention sandwich, dense gated MLP, PLE residual, layer scale.
    private func encodeDenseGemmaLayerDecode(
        _ cb: MTLCommandBuffer,
        layer: Int,
        postAttention: TensorView,
        projections: LayerSharedExpertProjections,
        d: UInt32,
        eps: Float
    ) throws {
        guard let elementwise else {
            preconditionFailure("dense Gemma layer requires elementwise kernels")
        }
        let preFFN = try model.preFFN(layer: layer)
        rms.encodeBF16W(commandBuffer: cb,
                        x: oOut,
                        weight: postAttention.buffer,
                        weightOffset: Int(postAttention.offset),
                        out: h1Buf, d: d, eps: eps)
        elementwise.encodeResidualAdd(commandBuffer: cb,
                                      hidden: hidden, delta: h1Buf,
                                      count: cfg.hiddenSize)
        rms.encodeBF16W(commandBuffer: cb,
                        x: hidden,
                        weight: preFFN.buffer,
                        weightOffset: Int(preFFN.offset),
                        out: denseX, d: d, eps: eps)
        try shared.encode(commandBuffer: cb,
                          x: denseX,
                          gate: projections.gate,
                          up: projections.up,
                          down: projections.down,
                          y: h1Buf,
                          scratchGate: denseScratchGate,
                          scratchUp: denseScratchUp,
                          scratchAct: denseScratchAct)
        let postFFN = projections.postF1!
        rms.encodeBF16W(commandBuffer: cb,
                        x: h1Buf,
                        weight: postFFN.buffer,
                        weightOffset: Int(postFFN.offset),
                        out: h1Buf, d: d, eps: eps)
        elementwise.encodeResidualAdd(commandBuffer: cb,
                                      hidden: hidden, delta: h1Buf,
                                      count: cfg.hiddenSize)

        let layerScalar = try model.layerScalar(layer: layer)
        let scalarBits = layerScalar.buffer.contents()
            .advanced(by: Int(layerScalar.offset))
            .assumingMemoryBound(to: UInt16.self)[0]
        let layerScale = Quantization.bf16ToFloat(scalarBits)
        guard cfg.hasPerLayerInputs else {
            elementwise.encodeScale(commandBuffer: cb, values: hidden,
                                    count: cfg.hiddenSize, scale: layerScale)
            return
        }
        guard let perLayerEmbeddingKernel,
              let pleIdentity, let pleContext, let pleLayer, let pleGate else {
            preconditionFailure("PLE architecture requires PLE kernels and scratch")
        }

        let plan = PerLayerEmbeddingPlan(config: cfg)
        let pleNorm = model.perLayerProjectionNorm
        perLayerEmbeddingKernel.encodePrepareLayer(
            commandBuffer: cb,
            identity: pleIdentity,
            context: pleContext,
            normWeight: pleNorm.buffer,
            normWeightOffset: Int(pleNorm.offset),
            out: pleLayer,
            rows: 1,
            packedWidth: plan.packedWidth,
            pleWidth: plan.perLayerHiddenSize,
            layer: layer,
            contextScale: plan.contextProjectionScale,
            combinedScale: plan.combinedScale,
            eps: eps)
        let inputGate = try model.perLayerInputGate(layer: layer)
        int4.encode(commandBuffer: cb,
                    weights: inputGate.buffer, weightsOffset: Int(inputGate.offset),
                    scales: inputGate.buffer, scalesOffset: Int(inputGate.scaleOffset),
                    biases: inputGate.buffer, biasesOffset: Int(inputGate.biasOffset),
                    x: hidden, y: pleGate,
                    m: UInt32(plan.perLayerHiddenSize), n: d)
        elementwise.encodeGELUMul(commandBuffer: cb,
                                  gate: pleGate, value: pleLayer, out: pleGate,
                                  count: plan.perLayerHiddenSize)
        let projection = try model.perLayerProjection(layer: layer)
        int4.encode(commandBuffer: cb,
                    weights: projection.buffer, weightsOffset: Int(projection.offset),
                    scales: projection.buffer, scalesOffset: Int(projection.scaleOffset),
                    biases: projection.buffer, biasesOffset: Int(projection.biasOffset),
                    x: pleGate, y: h2Buf,
                    m: d, n: UInt32(plan.perLayerHiddenSize))
        let postPLE = try model.postPerLayerInputNorm(layer: layer)
        rms.encodeBF16W(commandBuffer: cb,
                        x: h2Buf,
                        weight: postPLE.buffer,
                        weightOffset: Int(postPLE.offset),
                        out: h2Buf, d: d, eps: eps)
        elementwise.encodeResidualAddScale(
            commandBuffer: cb,
            hidden: hidden,
            delta: h2Buf,
            count: cfg.hiddenSize,
            scale: layerScale)
    }

    /// Gated-DeltaNet linear attention (layer mask 2), one decode step.
    /// Reads `normed`, updates the layer's recurrent state + conv tail in
    /// place, and leaves the attention-branch output in `oOut`.
    private func encodeLinearAttentionDecode(_ cb: MTLCommandBuffer, layer L: Int) throws {
        guard let gdn, let gdnState, let gdnQKVRaw, let gdnConvOut,
              let gdnZ, let gdnA, let gdnB, let gdnY, let gdnOut else {
            preconditionFailure("linear-attention layer without GDN kernels")
        }
        let la = cfg.linearAttention
        let D = UInt32(cfg.hiddenSize)
        let qkvW = try model.linearInProjQKV(layer: L)
        let zW = try model.linearInProjZ(layer: L)
        let aW = try model.linearInProjA(layer: L)
        let bW = try model.linearInProjB(layer: L)
        let outW = try model.linearOutProj(layer: L)
        let convW = try model.linearConv1d(layer: L)
        let aLog = try model.linearALog(layer: L)
        let dtBias = try model.linearDtBias(layer: L)
        let gatedNormW = try model.linearNorm(layer: L)

        // One dispatch over the concatenated qkv/z/a/b row space instead of four
        // separate GEMVs (a and b were 4 threadgroups each).
        gdn.encodeInputProjections(commandBuffer: cb,
                                   x: normed,
                                   qkv: qkvW, qkvOut: gdnQKVRaw,
                                   z: zW, zOut: gdnZ,
                                   a: aW, aOut: gdnA,
                                   b: bW, bOut: gdnB,
                                   hiddenSize: cfg.hiddenSize)

        gdn.encodeConvDecode(commandBuffer: cb,
                             tail: gdnState.convTailBuffer(layer: L),
                             qkv: gdnQKVRaw,
                             convWeight: convW.buffer,
                             convWeightOffset: Int(convW.offset),
                             out: gdnConvOut)
        gdn.encodeQKNorm(commandBuffer: cb, convOut: gdnConvOut)
        gdn.encodeDeltaStepDecode(commandBuffer: cb,
                                  convOut: gdnConvOut,
                                  aProj: gdnA,
                                  bProj: gdnB,
                                  aLog: aLog.buffer, aLogOffset: Int(aLog.offset),
                                  dtBias: dtBias.buffer, dtBiasOffset: Int(dtBias.offset),
                                  state: gdnState.stateBuffer(layer: L),
                                  y: gdnY)
        gdn.encodeGatedNorm(commandBuffer: cb,
                            y: gdnY,
                            z: gdnZ,
                            weight: gatedNormW.buffer,
                            weightOffset: Int(gatedNormW.offset),
                            out: gdnOut)
        int4.encode(commandBuffer: cb,
                    weights: outW.buffer, weightsOffset: Int(outW.offset),
                    scales: outW.buffer, scalesOffset: Int(outW.scaleOffset),
                    biases: outW.buffer, biasesOffset: Int(outW.biasOffset),
                    x: gdnOut, y: oOut, m: D, n: UInt32(la.valueDim))
    }

    /// Qwen full attention (attn_output_gate), one decode step: packed
    /// [query ; gate] q_proj split per head, weighted per-head q/k norms
    /// (no V norm), NeoX sub-dim RoPE, full attention with the configured
    /// scale, sigmoid output gate, then o_proj into `oOut`.
    private func encodeGatedFullAttentionDecode(_ cb: MTLCommandBuffer,
                                                layer L: Int,
                                                position: Int,
                                                seqLen: UInt32) throws {
        guard let elementwise, let rope, let qPackedScratch, let attnGateScratch else {
            preconditionFailure("attn_output_gate layer without gate kernels")
        }
        guard let kv else {
            preconditionFailure("FP16 attention requires an FP16 KV cache")
        }
        let D = UInt32(cfg.hiddenSize)
        let eps: Float = 1e-6
        let headDim = cfg.fullHeadDim
        let numKV = cfg.numFullKVHeads
        let qDim = UInt32(cfg.numHeads * headDim)
        let kvDim = UInt32(numKV * headDim)
        let kSlot = kv.kSlot(layer: L, position: position)
        let vSlot = kv.vSlot(layer: L, position: position)
        let q = try model.qProj(layer: L)
        let k = try model.kProj(layer: L)
        let v = try model.vProj(layer: L)
        let o = try model.oProj(layer: L)
        let qNormW = try model.qNorm(layer: L)
        let kNormW = try model.kNorm(layer: L)
        let rotaryDim = UInt32(Double(headDim) * cfg.partialRotaryFactor)

        fusedQKVGEMV.encode(commandBuffer: cb,
                            qWeights: q.buffer, qWeightsOffset: Int(q.offset),
                            qScales: q.buffer, qScalesOffset: Int(q.scaleOffset),
                            qBiases: q.buffer, qBiasesOffset: Int(q.biasOffset),
                            kWeights: k.buffer, kWeightsOffset: Int(k.offset),
                            kScales: k.buffer, kScalesOffset: Int(k.scaleOffset),
                            kBiases: k.buffer, kBiasesOffset: Int(k.biasOffset),
                            vWeights: v.buffer, vWeightsOffset: Int(v.offset),
                            vScales: v.buffer, vScalesOffset: Int(v.scaleOffset),
                            vBiases: v.buffer, vBiasesOffset: Int(v.biasOffset),
                            x: normed,
                            qOut: qPackedScratch,
                            kOut: kSlot.buffer, kOutOffset: kSlot.offset,
                            vOut: vSlot.buffer, vOutOffset: vSlot.offset,
                            qRows: 2 * qDim,
                            kvRows: kvDim,
                            n: D)
        elementwise.encodeSplitQGate(commandBuffer: cb,
                                     packed: qPackedScratch,
                                     q: qScratch,
                                     gate: attnGateScratch,
                                     heads: cfg.numHeads,
                                     dim: headDim)
        rms.encodeBF16WPerHead(commandBuffer: cb,
                               x: qScratch,
                               weight: qNormW.buffer,
                               weightOffset: Int(qNormW.offset),
                               out: qScratch,
                               headDim: UInt32(headDim),
                               numHeads: cfg.numHeads,
                               eps: eps)
        rms.encodeBF16WPerHead(commandBuffer: cb,
                               x: kSlot.buffer, xOffset: kSlot.offset,
                               weight: kNormW.buffer,
                               weightOffset: Int(kNormW.offset),
                               out: kSlot.buffer, outOffset: kSlot.offset,
                               headDim: UInt32(headDim),
                               numHeads: numKV,
                               eps: eps)
        let ropePosition = position + Int(qwenMultimodalRopeDelta)
        precondition(ropePosition >= 0, "Qwen multimodal RoPE position must be nonnegative")
        rope.encodeNeoxSubdim(commandBuffer: cb,
                              data: qScratch,
                              position: UInt32(ropePosition),
                              headDim: UInt32(headDim),
                              numHeads: UInt32(cfg.numHeads),
                              rotaryDim: rotaryDim,
                              theta: Float(cfg.fullRopeTheta))
        rope.encodeNeoxSubdim(commandBuffer: cb,
                              data: kSlot.buffer,
                              dataOffset: kSlot.offset,
                              position: UInt32(ropePosition),
                              headDim: UInt32(headDim),
                              numHeads: UInt32(numKV),
                              rotaryDim: rotaryDim,
                              theta: Float(cfg.fullRopeTheta))
        attention.encodeFull(commandBuffer: cb,
                             q: qScratch,
                             k: kSlot.buffer, kOffset: 0,
                             v: vSlot.buffer, vOffset: 0,
                             out: attnOut,
                             headDim: UInt32(headDim),
                             numQHeads: UInt32(cfg.numHeads),
                             numKVHeads: UInt32(numKV),
                             seqLen: seqLen,
                             scale: Float(cfg.attentionScale))
        elementwise.encodeSigmoidGateMul(commandBuffer: cb,
                                         out: attnOut,
                                         gate: attnGateScratch,
                                         count: Int(qDim))
        int4.encode(commandBuffer: cb,
                    weights: o.buffer, weightsOffset: Int(o.offset),
                    scales: o.buffer, scalesOffset: Int(o.scaleOffset),
                    biases: o.buffer, biasesOffset: Int(o.biasOffset),
                    x: attnOut, y: oOut, m: D, n: qDim)
    }

    private func runSync(_ body: (MTLCommandBuffer) -> Void) throws {
        let cb = ctx.queue.makeCommandBuffer()!
        body(cb)
        cb.commit()
        try waitForCompletion(cb)
        totalGPUHeadNanos &+= UInt64(max(0, (cb.gpuEndTime - cb.gpuStartTime) * 1e9))
    }

    private nonisolated func waitForCompletion(_ cb: MTLCommandBuffer) throws {
        waitUntilCompleted(cb)
        try checkCommandBufferError(cb.error)
    }

    private nonisolated func waitUntilCompleted(_ cb: MTLCommandBuffer) {
        cb.waitUntilCompleted()
    }

}

// MARK: - Decode kernel benchmark (diagnostic)

extension RealForwardRunner {
    /// Times the decode projections against this model's own resident tensors
    /// and reports achieved memory bandwidth, then times command-buffer round
    /// trips. Decode is memory-bound, so those two numbers say whether a token
    /// is limited by the kernels or by submission latency. The CLI exposes it
    /// behind `TUFF_KERNEL_BENCH=1`.
    public func benchmarkDecodeKernels(repetitions: Int = 20) throws -> String {
        var report = ""
        let D = UInt32(cfg.hiddenSize)
        guard let scratchLogits = ctx.device.makeBuffer(
            length: cfg.vocabSize * MemoryLayout<Float16>.size,
            options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        // Real activations: an all-zero input can be measured optimistically.
        for buffer in [normed, denseX, attnOut, hidden, denseScratchAct] {
            let count = buffer.length / MemoryLayout<Float16>.size
            let pointer = buffer.contents().assumingMemoryBound(to: Float16.self)
            for i in 0..<count { pointer[i] = Float16(sin(Float(i) * 0.37) * 0.5) }
        }
        func gemvBytes(_ view: TensorView) -> Double {
            Double(view.length + view.scaleLength + view.biasLength)
        }
        /// Best of several rounds. Each timed buffer is preceded by a burst of
        /// real work so nothing is measured at idle GPU clocks.
        func time(_ label: String, bytesPerRep: Double,
                  _ body: @escaping (MTLCommandBuffer) throws -> Void) throws {
            let cb = ctx.queue.makeCommandBuffer()!
            try body(cb)
            cb.commit()
            cb.waitUntilCompleted()
            let lm = model.lmHead
            var best = Double.infinity
            for _ in 0..<4 {
                let heater = ctx.queue.makeCommandBuffer()!
                for _ in 0..<2 {
                    int4.encode(commandBuffer: heater,
                                weights: lm.buffer, weightsOffset: Int(lm.offset),
                                scales: lm.buffer, scalesOffset: Int(lm.scaleOffset),
                                biases: lm.buffer, biasesOffset: Int(lm.biasOffset),
                                x: normed, y: scratchLogits,
                                m: UInt32(cfg.vocabSize), n: D)
                }
                heater.commit()
                let timed = ctx.queue.makeCommandBuffer()!
                for _ in 0..<repetitions { try body(timed) }
                timed.commit()
                timed.waitUntilCompleted()
                try checkCommandBufferError(timed.error)
                best = min(best, timed.gpuEndTime - timed.gpuStartTime)
            }
            report += String(format: "%-34@ %9.1f us  %7.1f GB/s  (%.1f MB)\n",
                             label as NSString,
                             best / Double(repetitions) * 1e6,
                             bytesPerRep * Double(repetitions) / best / 1e9,
                             bytesPerRep / 1e6)
        }
        func gemv(_ label: String, _ view: TensorView, x: MTLBuffer, y: MTLBuffer) throws {
            try time("\(label) \(view.shape.0)x\(view.shape.1)",
                     bytesPerRep: gemvBytes(view)) { [self] cb in
                int4.encode(commandBuffer: cb,
                            weights: view.buffer, weightsOffset: Int(view.offset),
                            scales: view.buffer, scalesOffset: Int(view.scaleOffset),
                            biases: view.buffer, biasesOffset: Int(view.biasOffset),
                            x: x, y: y, m: view.shape.0, n: view.shape.1)
            }
        }
        for L in Set([0, cfg.numLayers / 2, cfg.numLayers - 1]).sorted() {
            try gemv("L\(L) q_proj", try model.qProj(layer: L), x: normed, y: qScratch)
            try gemv("L\(L) o_proj", try model.oProj(layer: L), x: attnOut, y: oOut)
            if cfg.hasSharedExpert {
                try gemv("L\(L) gate_proj", try model.sharedExpertGate(layer: L),
                         x: denseX, y: denseScratchGate)
                try gemv("L\(L) down_proj", try model.sharedExpertDown(layer: L),
                         x: denseScratchAct, y: h1Buf)
            }
        }
        try gemv("lm_head", model.lmHead, x: normed, y: scratchLogits)

        // Wall-clock cost of a command-buffer round trip. Every one of these a
        // token pays is latency no kernel change can recover.
        let inputNorm = try model.inputNorm(layer: 0)
        func wall(_ label: String, iterations: Int, _ body: () -> Void) {
            for _ in 0..<3 { body() }
            var best = UInt64.max
            for _ in 0..<5 {
                let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
                for _ in 0..<iterations { body() }
                best = min(best, clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start)
            }
            report += String(format: "%-34@ %9.1f us wall\n", label as NSString,
                             Double(best) / Double(iterations) / 1000.0)
        }
        func encodeNorms(_ cb: MTLCommandBuffer, count: Int) {
            for _ in 0..<count {
                rms.encodeBF16W(commandBuffer: cb, x: hidden,
                                weight: inputNorm.buffer,
                                weightOffset: Int(inputNorm.offset),
                                out: normed, d: D, eps: 1e-6)
            }
        }
        wall("empty CB commit+wait", iterations: 200) { [self] in
            let cb = ctx.queue.makeCommandBuffer()!
            cb.commit()
            cb.waitUntilCompleted()
        }
        wall("16 norms, one CB", iterations: 100) { [self] in
            let cb = ctx.queue.makeCommandBuffer()!
            encodeNorms(cb, count: 16)
            cb.commit()
            cb.waitUntilCompleted()
        }
        wall("16 norms, 16 CBs", iterations: 20) { [self] in
            for _ in 0..<16 {
                let cb = ctx.queue.makeCommandBuffer()!
                encodeNorms(cb, count: 1)
                cb.commit()
                cb.waitUntilCompleted()
            }
        }
        return report
    }
}
