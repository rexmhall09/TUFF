import Metal

/// Qwen4-Exp's compressed-block indexer. State belongs to a runner, and each
/// full-attention layer owns its raw and pooled index keys. The ordinary KV
/// cache remains authoritative; selection only changes which rows are read.
final class QwenSparseAttention {
    struct Parameters {
        var start, tokens, dim, heads, ratio, topBlocks, blocks, rotary: UInt32
        var attentionDim, attentionHeads, kvHeads, partitions: UInt32
        var theta, scale: Float
    }
    struct State {
        let rawKeys, positions, pooled: MTLBuffer
    }
    private let context: MetalContext
    private let config: ArchConfig
    private let capacity: Int
    private var states: [Int: State] = [:]
    private let pipelines: [String: MTLComputePipelineState]
    private var scratch: [String: MTLBuffer] = [:]

    init(context: MetalContext, config: ArchConfig, maxContext: Int) throws {
        let index = config.attentionIndexer
        guard index.isEnabled, index.numKVHeads == 1,
              index.headDim <= 128, config.fullHeadDim <= 256,
              index.compressRatio > 0, index.budget % index.compressRatio == 0 else {
            throw PrefillError.chunkedUnsupported("unsupported Qwen sparse-attention geometry")
        }
        self.context = context
        self.config = config
        self.capacity = maxContext
        var pipelines: [String: MTLComputePipelineState] = [:]
        for name in ["prepare", "pool", "scores", "select", "attention_partial", "attention_combine"] {
            pipelines[name] = try context.pipeline("qsa_" + name)
        }
        self.pipelines = pipelines
    }

    private func buffer(_ name: String, bytes: Int) throws -> MTLBuffer {
        if let old = scratch[name], old.length >= bytes { return old }
        guard let value = context.device.makeBuffer(length: max(4, bytes), options: .storageModeShared) else {
            throw MetalError.noDevice
        }
        scratch[name] = value
        return value
    }

    func projectionBuffer(tokens: Int) throws -> MTLBuffer {
        try buffer("projection", bytes: tokens * config.attentionIndexer.projectionRows * 2)
    }

    /// Encode after index_qk_proj and before attention, on the same queue.
    /// Scratch is shared across layers; GPU ordering, not CPU waits, protects it.
    func encode(commandBuffer cb: MTLCommandBuffer, model: Model, layer: Int,
                projected: MTLBuffer, start: Int, tokens: Int,
                positions: MultimodalPositionIDs?, ropeDelta: Int32,
                q: MTLBuffer, k: MTLBuffer, v: MTLBuffer, out: MTLBuffer) throws {
        let index = config.attentionIndexer
        let state: State
        if let existing = states[layer] { state = existing }
        else {
            func allocate(_ bytes: Int) throws -> MTLBuffer {
                guard let b = context.device.makeBuffer(length: max(4, bytes), options: .storageModePrivate) else {
                    throw MetalError.noDevice
                }
                return b
            }
            state = try State(
                rawKeys: allocate(capacity * index.headDim * 2),
                positions: allocate(capacity * 12),
                pooled: allocate((capacity / index.compressRatio + 1) * index.headDim * 2))
            states[layer] = state
        }
        // Keep position data separate for each encoded call so the host cannot
        // overwrite a previous layer's values while the GPU is still using them.
        var xyz: [Int32] = []
        xyz.reserveCapacity(tokens * 3)
        for t in 0..<tokens {
            let text = Int32(start + t) + ropeDelta
            xyz.append(positions?.temporal[t] ?? text)
            xyz.append(positions?.height[t] ?? text)
            xyz.append(positions?.width[t] ?? text)
        }
        // A distinct buffer per call also handles image chunks above setBytes'
        // 4 KiB limit. Metal retains it until this command buffer completes.
        let positionBuffer = try xyz.withUnsafeBytes { bytes -> MTLBuffer in
            guard let b = context.device.makeBuffer(bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared) else { throw MetalError.noDevice }
            return b
        }
        var p = Parameters(
            start: UInt32(start), tokens: UInt32(tokens), dim: UInt32(index.headDim),
            heads: UInt32(index.numHeads), ratio: UInt32(index.compressRatio),
            topBlocks: UInt32(index.budget / index.compressRatio),
            blocks: UInt32((start + tokens) / index.compressRatio),
            rotary: UInt32(Double(config.fullHeadDim) * config.partialRotaryFactor),
            attentionDim: UInt32(config.fullHeadDim), attentionHeads: UInt32(config.numHeads),
            kvHeads: UInt32(config.numFullKVHeads), partitions: 16,
            theta: Float(config.fullRopeTheta), scale: Float(config.attentionScale))
        let queries = try buffer("queries", bytes: tokens * index.numHeads * index.headDim * 2)
        let prefix = "language_model.model.layers.\(layer).self_attn.indexer."
        let qNorm = try model.resident(name: prefix + "q_layernorm.weight")
        let kNorm = try model.resident(name: prefix + "k_layernorm.weight")
        func dispatch(_ name: String, _ buffers: [(MTLBuffer, Int)], _ shape: MTLSize, width: Int) throws {
            guard let e = cb.makeComputeCommandEncoder() else { throw MetalError.noDevice }
            e.setComputePipelineState(pipelines[name]!)
            for (i, b) in buffers.enumerated() { e.setBuffer(b.0, offset: b.1, index: i) }
            e.setBytes(&p, length: MemoryLayout<Parameters>.stride, index: buffers.count)
            e.dispatchThreadgroups(shape, threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
            e.endEncoding()
        }
        try dispatch("prepare", [(projected, 0), (qNorm.buffer, Int(qNorm.offset)), (queries, 0), (state.rawKeys, 0), (state.positions, 0), (positionBuffer, 0)], MTLSize(width: index.numHeads + 1, height: tokens, depth: 1), width: 128)
        let newBlocks = Int(p.blocks) - start / index.compressRatio
        if newBlocks > 0 {
            try dispatch("pool", [(state.rawKeys, 0), (kNorm.buffer, Int(kNorm.offset)), (state.positions, 0), (state.pooled, 0)], MTLSize(width: newBlocks, height: 1, depth: 1), width: 128)
        }
        // Below the sparse threshold the caller uses the existing dense path.
        guard Int(p.blocks) > Int(p.topBlocks) else { return }
        let scores = try buffer("scores", bytes: tokens * Int(p.blocks) * 4)
        let selected = try buffer("selected", bytes: tokens * Int(p.topBlocks) * 4)
        let partial = try buffer("partial", bytes: tokens * config.numHeads * Int(p.partitions) * (config.fullHeadDim + 2) * 4)
        try dispatch("scores", [(queries, 0), (state.pooled, 0), (scores, 0)], MTLSize(width: Int(p.blocks), height: tokens, depth: 1), width: 32)
        try dispatch("select", [(scores, 0), (selected, 0)], MTLSize(width: tokens, height: 1, depth: 1), width: 256)
        try dispatch("attention_partial", [(q, 0), (k, 0), (v, 0), (selected, 0), (partial, 0)], MTLSize(width: config.numHeads, height: tokens, depth: Int(p.partitions)), width: 32)
        try dispatch("attention_combine", [(partial, 0), (out, 0)], MTLSize(width: config.numHeads, height: tokens, depth: 1), width: 32)
    }

    func usesSparseAttention(endPosition: Int) -> Bool {
        let i = config.attentionIndexer
        return endPosition / i.compressRatio > i.budget / i.compressRatio
    }
}
