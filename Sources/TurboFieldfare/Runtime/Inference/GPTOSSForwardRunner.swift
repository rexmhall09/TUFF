import Foundation
import Metal

/// One-token-at-a-time GPT-OSS forward path. The resident BF16 projections
/// stay mapped while the four routed MXFP4 experts are fetched and executed
/// serially for each layer. This is also the correctness fallback for prefill;
/// a later optimization may batch those same operations without changing the
/// model contract or KV layout.
final class GPTOSSForwardRunner: ChunkedPrefillRunner, ContextWindowReporting,
    ContinuableLogitProducer, @unchecked Sendable {
    private struct LayerViews {
        let inputNorm: TensorView
        let postAttentionNorm: TensorView
        let qWeight: TensorView
        let qBias: TensorView
        let kWeight: TensorView
        let kBias: TensorView
        let vWeight: TensorView
        let vBias: TensorView
        let oWeight: TensorView
        let oBias: TensorView
        let sinks: TensorView
        let routerWeight: TensorView
        let routerBias: TensorView
        let expertOffsets: GPTOSSExpertOffsets
    }

    private let model: Model
    private let context: MetalContext
    private let config: ArchConfig
    private let kv: KVCacheManager
    private let layers: [LayerViews]

    private let bf16: BF16GEMV
    private let rms: RMSNorm
    private let rope: RoPE
    private let attention: Attention
    private let elementwise: Elementwise
    private let moePrimitives: GPTOSSMoEPrimitives
    private let expertRuntime: GPTOSSExpertRuntime

    private let hidden: MTLBuffer
    private let normed: MTLBuffer
    private let query: MTLBuffer
    private let attentionOutput: MTLBuffer
    private let projectedAttention: MTLBuffer
    private let routerLogits: MTLBuffer
    private let routedIndices: MTLBuffer
    private let expertScratch: GPTOSSExpertScratchBuffers

    let maxContext: Int
    private(set) var totalIoNanos: UInt64 = 0
    private(set) var totalCb1Nanos: UInt64 = 0
    private(set) var totalCb2Nanos: UInt64 = 0
    private(set) var totalHeadNanos: UInt64 = 0

    init(model: Model, context: MetalContext, maxContext: Int,
         runtimeConfiguration: RuntimeConfiguration) throws {
        let config = model.config
        guard config.family == .gptOss,
              config.feedForwardKind == .mixtureOfExperts,
              config.topKExperts == 4,
              config.attentionSinks,
              let yarn = config.yarnRope,
              yarn.originalContextLength > 0 else {
            throw ModelError.archMismatch(
                field: "family",
                expected: ModelFamily.gptOss.rawValue,
                actual: config.family.rawValue)
        }
        guard maxContext > 0 else {
            throw PrefillError.chunkedUnsupported(
                "GPT-OSS maxContext must be positive")
        }

        self.model = model
        self.context = context
        self.config = config
        self.maxContext = maxContext
        self.kv = try KVCacheManager(
            device: context.device,
            config: config,
            maxContext: maxContext,
            fp16RingEnabled: runtimeConfiguration.fp16RingEnabled,
            slidingWindow: config.slidingWindow,
            maxPrefillChunkTokens: PrefillRuntimeConfig.maxChunkTokens)

        bf16 = try BF16GEMV(context: context)
        rms = try RMSNorm(context: context)
        rope = try RoPE(context: context)
        attention = try Attention(context: context)
        elementwise = try Elementwise(context: context)
        moePrimitives = try GPTOSSMoEPrimitives(context: context)
        expertRuntime = try GPTOSSExpertRuntime(context: context)

        var resolvedLayers: [LayerViews] = []
        resolvedLayers.reserveCapacity(config.numLayers)
        for layer in 0..<config.numLayers {
            resolvedLayers.append(LayerViews(
                inputNorm: try model.inputNorm(layer: layer),
                postAttentionNorm: try model.postAttnNorm(layer: layer),
                qWeight: try model.qProj(layer: layer),
                qBias: try model.qProjBias(layer: layer),
                kWeight: try model.kProj(layer: layer),
                kBias: try model.kProjBias(layer: layer),
                vWeight: try model.vProj(layer: layer),
                vBias: try model.vProjBias(layer: layer),
                oWeight: try model.oProj(layer: layer),
                oBias: try model.oProjBias(layer: layer),
                sinks: try model.attentionSinks(layer: layer),
                routerWeight: try model.router(layer: layer),
                routerBias: try model.routerBias(layer: layer),
                expertOffsets: try model.gptOssRoutedExpertOffsets(layer: layer)))
        }
        layers = resolvedLayers

        func sharedBuffer(elements: Int, stride: Int, label: String) throws -> MTLBuffer {
            guard let buffer = context.device.makeBuffer(
                length: max(1, elements) * stride,
                options: .storageModeShared) else {
                throw ModelError.residentBufferWrapFailed
            }
            buffer.label = label
            return buffer
        }
        let hiddenSize = config.hiddenSize
        let querySize = config.numHeads * config.headDim
        hidden = try sharedBuffer(elements: hiddenSize,
                                  stride: MemoryLayout<Float16>.stride,
                                  label: "gptoss.hidden")
        normed = try sharedBuffer(elements: hiddenSize,
                                  stride: MemoryLayout<Float16>.stride,
                                  label: "gptoss.normed")
        query = try sharedBuffer(elements: querySize,
                                 stride: MemoryLayout<Float16>.stride,
                                 label: "gptoss.query")
        attentionOutput = try sharedBuffer(elements: querySize,
                                           stride: MemoryLayout<Float16>.stride,
                                           label: "gptoss.attention")
        projectedAttention = try sharedBuffer(elements: hiddenSize,
                                              stride: MemoryLayout<Float16>.stride,
                                              label: "gptoss.projectedAttention")
        routerLogits = try sharedBuffer(elements: config.numExperts,
                                        stride: MemoryLayout<Float>.stride,
                                        label: "gptoss.routerLogits")
        routedIndices = try sharedBuffer(elements: config.topKExperts,
                                         stride: MemoryLayout<UInt32>.stride,
                                         label: "gptoss.routedIndices")
        expertScratch = try GPTOSSExpertScratchBuffers.allocate(
            device: context.device,
            layout: GPTOSSExpertScratchLayout(
                hiddenSize: hiddenSize,
                intermediateSize: config.moeIntermediateSize,
                topK: config.topKExperts,
                queryCapacity: 1))
    }

    func reset() {
        kv.reset()
    }

    var continuationPosition: Int { kv.position }

    func prepareForContinuation(expectedPosition: Int) throws {
        guard expectedPosition > 0, kv.position == expectedPosition else {
            throw PrefillError.prefillCursorMismatch(
                "GPT-OSS continuation expected KV position \(expectedPosition), current \(kv.position)")
        }
    }

    func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
        try await executeToken(token: token, position: position,
                               emitHead: true, logits: logits)
    }

    func prefillChunked(tokens: ArraySlice<Int32>,
                        startPosition: Int,
                        outputMode _: PrefillOutputMode,
                        config prefillConfig: PrefillRuntimeConfig,
                        into logits: MTLBuffer,
                        onProgress: (Int) -> Void) async throws -> PrefillResult {
        guard prefillConfig.mode == .chunked else {
            throw PrefillError.chunkedUnsupported(
                "GPT-OSS chunked prefill requires chunked mode")
        }
        guard startPosition == kv.position else {
            throw PrefillError.prefillCursorMismatch(
                "GPT-OSS prefill cursor \(kv.position) != startPosition \(startPosition)")
        }
        guard startPosition >= 0,
              tokens.count <= maxContext - startPosition else {
            throw PrefillError.chunkedUnsupported(
                "GPT-OSS prefill exceeds maxContext \(maxContext)")
        }
        guard !tokens.isEmpty else {
            return PrefillResult(newPosition: startPosition,
                                 seed: .logitsWritten)
        }

        for (offset, token) in tokens.enumerated() {
            try Task.checkCancellation()
            try await executeToken(
                token: token,
                position: startPosition + offset,
                emitHead: offset == tokens.count - 1,
                logits: logits)
            onProgress(offset + 1)
        }
        return PrefillResult(newPosition: startPosition + tokens.count,
                             seed: .logitsWritten)
    }

    private func executeToken(token: Int32, position: Int,
                              emitHead: Bool, logits: MTLBuffer) async throws {
        guard position == kv.position else {
            throw PrefillError.prefillCursorMismatch(
                "GPT-OSS token position \(position) != KV position \(kv.position)")
        }
        guard position >= 0, position < maxContext else {
            throw PrefillError.chunkedUnsupported(
                "GPT-OSS position \(position) exceeds maxContext \(maxContext)")
        }
        guard token >= 0, Int(token) < config.vocabSize else {
            throw GeneratorError.invalidGenerationConfig(
                "GPT-OSS token ID \(token) is outside vocab \(config.vocabSize)")
        }
        if emitHead && logits.length < config.vocabSize * MemoryLayout<Float16>.stride {
            throw PrefillError.chunkedUnsupported(
                "GPT-OSS logits buffer is smaller than the model vocabulary")
        }

        let embeddingCB = context.queue.makeCommandBuffer()!
        bf16.encodeEmbedding(commandBuffer: embeddingCB,
                             table: model.embedding,
                             token: UInt32(token),
                             output: hidden,
                             hiddenSize: config.hiddenSize)
        embeddingCB.commit()
        try waitForCompletion(embeddingCB)

        for layer in 0..<config.numLayers {
            try Task.checkCancellation()
            try await executeLayer(layer, position: position)
        }
        kv.advance()

        guard emitHead else { return }
        let headStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let headCB = context.queue.makeCommandBuffer()!
        rms.encodeBF16W(commandBuffer: headCB,
                        x: hidden,
                        weight: model.finalNorm.buffer,
                        weightOffset: Int(model.finalNorm.offset),
                        out: normed,
                        d: UInt32(config.hiddenSize),
                        eps: 1e-5)
        bf16.encodeHalf(commandBuffer: headCB,
                        weights: model.lmHead,
                        input: normed,
                        output: logits,
                        rows: config.vocabSize,
                        columns: config.hiddenSize)
        headCB.commit()
        try waitForCompletion(headCB)
        totalHeadNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - headStart
    }

    private func executeLayer(_ layer: Int, position: Int) async throws {
        let views = layers[layer]
        let hiddenSize = config.hiddenSize
        let qRows = config.numHeads * config.headDim
        let kvRows = config.numKVHeads * config.headDim
        let kSlot = kv.kSlot(layer: layer, position: position)
        let vSlot = kv.vSlot(layer: layer, position: position)

        let cb1Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let cb1 = context.queue.makeCommandBuffer()!
        rms.encodeBF16W(commandBuffer: cb1,
                        x: hidden,
                        weight: views.inputNorm.buffer,
                        weightOffset: Int(views.inputNorm.offset),
                        out: normed,
                        d: UInt32(hiddenSize),
                        eps: 1e-5)
        bf16.encodeHalf(commandBuffer: cb1,
                        weights: views.qWeight,
                        input: normed,
                        output: query,
                        bias: views.qBias,
                        rows: qRows,
                        columns: hiddenSize)
        bf16.encodeHalf(commandBuffer: cb1,
                        weights: views.kWeight,
                        input: normed,
                        output: kSlot.buffer,
                        outputOffset: kSlot.offset,
                        bias: views.kBias,
                        rows: kvRows,
                        columns: hiddenSize)
        bf16.encodeHalf(commandBuffer: cb1,
                        weights: views.vWeight,
                        input: normed,
                        output: vSlot.buffer,
                        outputOffset: vSlot.offset,
                        bias: views.vBias,
                        rows: kvRows,
                        columns: hiddenSize)
        encodeRoPE(commandBuffer: cb1, data: query, dataOffset: 0,
                   position: position, heads: config.numHeads)
        encodeRoPE(commandBuffer: cb1, data: kSlot.buffer,
                   dataOffset: kSlot.offset,
                   position: position, heads: config.numKVHeads)

        let sequenceLength = UInt32(position + 1)
        if config.layerIsFull(layer) {
            attention.encodeFull(
                commandBuffer: cb1,
                q: query,
                k: kSlot.buffer,
                v: vSlot.buffer,
                out: attentionOutput,
                headDim: UInt32(config.headDim),
                numQHeads: UInt32(config.numHeads),
                numKVHeads: UInt32(config.numKVHeads),
                seqLen: sequenceLength,
                scale: Float(config.attentionScale),
                sinks: views.sinks.buffer,
                sinksOffset: Int(views.sinks.offset))
        } else {
            attention.encodeSWA(
                commandBuffer: cb1,
                q: query,
                k: kSlot.buffer,
                v: vSlot.buffer,
                out: attentionOutput,
                headDim: UInt32(config.headDim),
                numQHeads: UInt32(config.numHeads),
                numKVHeads: UInt32(config.numKVHeads),
                seqLen: sequenceLength,
                window: UInt32(config.slidingWindow),
                scale: Float(config.attentionScale),
                sinks: views.sinks.buffer,
                sinksOffset: Int(views.sinks.offset),
                ringCapacity: UInt32(kv.ringCapacity(layer: layer)))
        }
        bf16.encodeHalf(commandBuffer: cb1,
                        weights: views.oWeight,
                        input: attentionOutput,
                        output: projectedAttention,
                        bias: views.oBias,
                        rows: hiddenSize,
                        columns: qRows)
        elementwise.encodeResidualAdd(commandBuffer: cb1,
                                      hidden: hidden,
                                      delta: projectedAttention,
                                      count: hiddenSize)
        rms.encodeBF16W(commandBuffer: cb1,
                        x: hidden,
                        weight: views.postAttentionNorm.buffer,
                        weightOffset: Int(views.postAttentionNorm.offset),
                        out: normed,
                        d: UInt32(hiddenSize),
                        eps: 1e-5)
        bf16.encodeFloat(commandBuffer: cb1,
                         weights: views.routerWeight,
                         input: normed,
                         output: routerLogits,
                         bias: views.routerBias,
                         rows: config.numExperts,
                         columns: hiddenSize)
        moePrimitives.encodeRouterTop4(
            commandBuffer: cb1,
            logits: routerLogits,
            outputIndices: routedIndices,
            outputWeights: expertScratch.routeWeights,
            numExperts: UInt32(config.numExperts))
        cb1.commit()
        try waitForCompletion(cb1)
        totalCb1Nanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - cb1Start

        let indexPointer = routedIndices.contents()
            .assumingMemoryBound(to: UInt32.self)
        let experts = (0..<config.topKExperts).map { Int(indexPointer[$0]) }

        let ioStart = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let blobs = try await model.fetchRoutedExperts(
            layer: layer, experts: experts)
        totalIoNanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - ioStart

        let cb2Start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
        let cb2 = context.queue.makeCommandBuffer()!
        for route in 0..<config.topKExperts {
            try expertRuntime.encodeExpert(
                commandBuffer: cb2,
                blob: blobs[route],
                offsets: views.expertOffsets,
                input: normed,
                queryIndex: 0,
                routeSlot: route,
                scratch: expertScratch,
                swigluLimit: Float(config.swigluLimit))
        }
        try expertRuntime.encodeReduce(
            commandBuffer: cb2,
            scratch: expertScratch,
            residual: hidden,
            output: hidden,
            queryCount: 1)
        cb2.commit()
        try waitForCompletion(cb2)
        totalCb2Nanos &+= clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - cb2Start
    }

    private func encodeRoPE(commandBuffer: MTLCommandBuffer,
                            data: MTLBuffer,
                            dataOffset: Int,
                            position: Int,
                            heads: Int) {
        let yarn = config.yarnRope!
        rope.encodeYaRNNeox(
            commandBuffer: commandBuffer,
            data: data,
            dataOffset: dataOffset,
            position: UInt32(position),
            headDim: UInt32(config.headDim),
            numHeads: UInt32(heads),
            theta: Float(config.ropeTheta),
            originalContextLength: UInt32(yarn.originalContextLength),
            scalingFactor: Float(yarn.scalingFactor),
            betaFast: Float(yarn.betaFast),
            betaSlow: Float(yarn.betaSlow))
    }

    private nonisolated func waitForCompletion(
        _ commandBuffer: MTLCommandBuffer
    ) throws {
        commandBuffer.waitUntilCompleted()
        try checkCommandBufferError(commandBuffer.error)
    }
}
