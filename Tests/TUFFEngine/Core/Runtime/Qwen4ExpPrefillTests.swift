import Testing
import Foundation
import Metal
@testable import TUFFEngine
import TUFFValidationSupport

/// Chunked prefill has to agree with decoding the same tokens one at a time.
/// The two paths share the KV cache, the gated-DeltaNet recurrent state, the
/// convolution tails and the n-gram history, and they run entirely different
/// kernels over the four-stream residual — so equivalence is the only check
/// that covers the layer graph end to end.
@Suite struct Qwen4ExpPrefillTests {

    private func makeRunner(maxContext: Int = 128,
                            config: ArchConfig = .qwen4ExpToy())
        throws -> (URL, MetalContext, RealForwardRunner) {
        let dir = try Qwen4ExpToySynthetic.write(config: config)
        let ctx = try MetalContext()
        let model = try Model.load(directoryURL: dir, device: ctx.device,
                                   expecting: config)
        let runner = try RealForwardRunner(model: model, context: ctx,
                                           maxContext: maxContext)
        return (dir, ctx, runner)
    }

    private func makeLogits(_ ctx: MetalContext, vocab: Int) throws -> MTLBuffer {
        guard let buf = ctx.device.makeBuffer(
            length: vocab * MemoryLayout<Float16>.stride,
            options: .storageModeShared) else {
            throw ModelError.residentBufferWrapFailed
        }
        return buf
    }

    @Test func imageGatherPlaceholdersDoNotChangeNgramSemantics() async throws {
        let config = ArchConfig.qwen4ExpToy(ngramLayer: 1)
        let (dir, ctx, runner) = try makeRunner(config: config)
        defer { try? FileManager.default.removeItem(at: dir) }
        let featuresBuffer = try #require(ctx.device.makeBuffer(length: config.hiddenSize * 2, options: .storageModeShared))
        let data = featuresBuffer.contents().assumingMemoryBound(to: Float16.self)
        for d in 0..<config.hiddenSize { data[d] = Float16(sin(Float(d))) }
        let features = VisionFeatures(buffer: featuresBuffer, tokenCount: 1,
            hiddenSize: config.hiddenSize, family: .qwen4Exp,
            patchGridWidth: 2, patchGridHeight: 2, gpuNanoseconds: 0, scratchBytes: 0,
            attentionVariant: .native72, projectorPath: .fallback,
            expertResidencyTransition: nil, preprocessing: nil)
        let ids: [Int32] = [11, 8, 7, 19]
        var reference: [Float] = []
        for gatherID: Int32 in [8, 0] {
            runner.reset()
            let input = try MultimodalPrefillInput(effectiveTokenIDs: ids,
                embeddingTokenIDs: [11, gatherID, 7, 19],
                imageSpans: [.init(tokenRange: 1..<2, features: features)], family: .qwen4Exp)
            let logits = try makeLogits(ctx, vocab: config.vocabSize)
            _ = try await runner.prefillMultimodal(input: input, startPosition: 0,
                outputMode: .logits, config: .production(chunkTokens: 32), into: logits,
                onProgress: { _ in })
            let values = Fp16Buffer.read(logits, count: config.vocabSize)
            if reference.isEmpty { reference = values }
            else { #expect(RelError.compute(actual: values, reference: reference) < 1e-5) }
        }
    }

    @Test func textPrefillAfterImageMatchesSequentialDecode() async throws {
        let config = ArchConfig.qwen4ExpToy(ngramLayer: 1, indexer: .init(
            budget: 4, compressRatio: 2, headDim: 32, numHeads: 2, numKVHeads: 1))
        let (dir, ctx, runner) = try makeRunner(config: config)
        defer { try? FileManager.default.removeItem(at: dir) }
        let b = try #require(ctx.device.makeBuffer(length: 4 * config.hiddenSize * 2, options: .storageModeShared))
        let data = b.contents().assumingMemoryBound(to: Float16.self)
        for d in 0..<(4 * config.hiddenSize) { data[d] = Float16(sin(Float(d) * 0.3)) }
        let features = VisionFeatures(buffer: b, tokenCount: 4, hiddenSize: config.hiddenSize,
            family: .qwen4Exp, patchGridWidth: 4, patchGridHeight: 4,
            gpuNanoseconds: 0, scratchBytes: 0, attentionVariant: .native72,
            projectorPath: .fallback, expertResidencyTransition: nil, preprocessing: nil)
        let input = try MultimodalPrefillInput(effectiveTokenIDs: [11, 8, 8, 8, 8, 19],
            embeddingTokenIDs: [11, 0, 0, 0, 0, 19], imageSpans: [.init(tokenRange: 1..<5, features: features)], family: .qwen4Exp)
        #expect(input.ropeDelta != 0)
        let logits = try makeLogits(ctx, vocab: config.vocabSize)
        let followup: [Int32] = [5, 9, 3]
        var reference: [Float] = []
        for chunked in [false, true] {
            runner.reset()
            _ = try await runner.prefillMultimodal(input: input, startPosition: 0,
                outputMode: .logits, config: .production(chunkTokens: 32), into: logits, onProgress: { _ in })
            if chunked {
                _ = try await runner.prefillChunked(tokens: followup[...], startPosition: 6,
                    outputMode: .logits, config: .production(chunkTokens: 32), into: logits, onProgress: { _ in })
                #expect(RelError.compute(actual: Fp16Buffer.read(logits, count: config.vocabSize), reference: reference) < 0.025)
            } else {
                for (p, token) in followup.enumerated() { try await runner.produce(token: token, position: 6 + p, into: logits) }
                reference = Fp16Buffer.read(logits, count: config.vocabSize)
            }
        }
    }

    @Test func sparsePrefillMatchesDecodeAcrossBlockBoundaries() async throws {
        let config = ArchConfig.qwen4ExpToy(indexer: .init(
            budget: 8, compressRatio: 4, headDim: 32, numHeads: 2, numKVHeads: 1))
        let (dir, ctx, runner) = try makeRunner(config: config)
        defer { try? FileManager.default.removeItem(at: dir) }
        let tokens: [Int32] = (0..<19).map { Int32(($0 * 7 + 3) % 31) }
        let logits = try makeLogits(ctx, vocab: config.vocabSize)
        for (p, token) in tokens.enumerated() { try await runner.produce(token: token, position: p, into: logits) }
        let reference = Fp16Buffer.read(logits, count: config.vocabSize)
        runner.reset()
        // 7-token pieces force incomplete compressed blocks across calls.
        for start in stride(from: 0, to: tokens.count, by: 7) {
            let end = min(start + 7, tokens.count)
            _ = try await runner.prefillChunked(tokens: tokens[start..<end], startPosition: start,
                outputMode: .logits, config: .production(chunkTokens: 32), into: logits,
                onProgress: { _ in })
        }
        let actual = Fp16Buffer.read(logits, count: config.vocabSize)
        #expect(RelError.compute(actual: actual, reference: reference) < 0.025)
    }

    @Test func theArchitectureNowSupportsChunkedPrefill() throws {
        let (dir, _, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(runner.supportsChunkedPrefill)
    }

    @Test func continuationPreservesNgramGDNAndSparseAttentionState() async throws {
        let config = ArchConfig.qwen4ExpToy(ngramLayer: 1, indexer: .init(
            budget: 8, compressRatio: 4, headDim: 32, numHeads: 2, numKVHeads: 1))
        let (dir, ctx, runner) = try makeRunner(config: config)
        defer { try? FileManager.default.removeItem(at: dir) }
        let logits = try makeLogits(ctx, vocab: config.vocabSize)
        let tokens: [Int32] = (0..<19).map { Int32(($0 * 7 + 3) % 31) }
        var reference: [Float] = []
        for resumes in [false, true] {
            runner.reset()
            _ = try await runner.prefillChunked(tokens: tokens[..<7], startPosition: 0,
                outputMode: .logits, config: .production(chunkTokens: 32),
                into: logits, onProgress: { _ in })
            // The assistant's decode advances both recurrent and KV state.
            for p in 7..<11 {
                try await runner.produce(token: tokens[p], position: p, into: logits)
            }
            if resumes { try runner.prepareForContinuation(expectedPosition: 11) }
            var computed = 0
            _ = try await runner.prefillChunked(tokens: tokens[11...], startPosition: 11,
                outputMode: .logits, config: .production(chunkTokens: 32),
                into: logits, onProgress: { computed = $0 })
            #expect(computed == 8)
            #expect(runner.continuationPosition == tokens.count)
            let actual = Fp16Buffer.read(logits, count: config.vocabSize)
            if resumes { #expect(actual == reference) }
            else { reference = actual }
        }
    }

    /// Prefilling the whole prompt must land the same logits as decoding it.
    @Test(arguments: [1, 2, 5, 8, 32, 65])
    func chunkedPrefillMatchesTokenAtATimeDecode(count: Int) async throws {
        let config = ArchConfig.qwen4ExpToy()
        let tokens: [Int32] = (0..<count).map { Int32(($0 * 7 + 11) % 31) }

        let (dirA, ctxA, decodeRunner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dirA) }
        let decodeLogits = try makeLogits(ctxA, vocab: config.vocabSize)
        for (position, token) in tokens.enumerated() {
            try await decodeRunner.produce(token: token, position: position,
                                           into: decodeLogits)
        }
        let reference = Fp16Buffer.read(decodeLogits, count: config.vocabSize)

        let (dirB, ctxB, prefillRunner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dirB) }
        let prefillLogits = try makeLogits(ctxB, vocab: config.vocabSize)
        _ = try await prefillRunner.prefillChunked(
            tokens: tokens[...],
            startPosition: 0,
            outputMode: .logits,
            config: .production(chunkTokens: 32),
            into: prefillLogits,
            onProgress: { _ in })
        let actual = Fp16Buffer.read(prefillLogits, count: config.vocabSize)

        let relErr = RelError.compute(actual: actual, reference: reference)
        #expect(relErr < 0.02, "count=\(count) relErr=\(relErr)")
        let refArgmax = reference.enumerated().max(by: { $0.element < $1.element })!.offset
        let gotArgmax = actual.enumerated().max(by: { $0.element < $1.element })!.offset
        #expect(refArgmax == gotArgmax, "count=\(count) argmax \(gotArgmax) != \(refArgmax)")
    }

    /// The n-gram layer is the one place the chunk path runs a per-token loop,
    /// and its embedding rows are gathered on the CPU while the chunk is being
    /// encoded. A single shared row left every token in a chunk reading the
    /// last one's n-gram, which no fixture without an n-gram layer can catch.
    @Test(arguments: [2, 5, 8])
    func chunkedPrefillMatchesDecodeWithTheNgramLayer(count: Int) async throws {
        let config = ArchConfig.qwen4ExpToy(ngramLayer: 1)
        let prompt: [Int32] = [11, 4, 7, 2, 19, 3, 8, 5]
        let tokens = Array(prompt.prefix(count))

        let (dirA, ctxA, decodeRunner) = try makeRunner(config: config)
        defer { try? FileManager.default.removeItem(at: dirA) }
        let decodeLogits = try makeLogits(ctxA, vocab: config.vocabSize)
        for (position, token) in tokens.enumerated() {
            try await decodeRunner.produce(token: token, position: position,
                                           into: decodeLogits)
        }
        let reference = Fp16Buffer.read(decodeLogits, count: config.vocabSize)

        let (dirB, ctxB, prefillRunner) = try makeRunner(config: config)
        defer { try? FileManager.default.removeItem(at: dirB) }
        let prefillLogits = try makeLogits(ctxB, vocab: config.vocabSize)
        _ = try await prefillRunner.prefillChunked(
            tokens: tokens[...], startPosition: 0, outputMode: .logits,
            config: .production(chunkTokens: 32), into: prefillLogits,
            onProgress: { _ in })
        let actual = Fp16Buffer.read(prefillLogits, count: config.vocabSize)

        let relErr = RelError.compute(actual: actual, reference: reference)
        #expect(relErr < 0.02, "count=\(count) relErr=\(relErr)")
        let refArgmax = reference.enumerated().max(by: { $0.element < $1.element })!.offset
        let gotArgmax = actual.enumerated().max(by: { $0.element < $1.element })!.offset
        #expect(refArgmax == gotArgmax, "count=\(count) argmax \(gotArgmax) != \(refArgmax)")
    }

    /// An image span is emitted as a single chunk of its own pooled length,
    /// and `prefillMultimodal` sizes the chunk layout from that span rather
    /// than from `maxChunkTokens`. The n-gram layer stages one embedding row
    /// per token in the chunk, so that staging has to come from the layout
    /// too — sizing it from the constant wrote past the buffer and killed the
    /// decode service with a bus error on the first photo big enough to pool
    /// past 256 tokens.
    @Test func ngramStagingHasARowForEveryTokenAnImageSpanCanCarry() {
        let config = ArchConfig.qwen4ExpToy(ngramLayer: 1)
        let pooled = PrefillRuntimeConfig.maxChunkTokens + 512
        let layout = PrefillChunkScratchLayout(
            config: config,
            chunkTokens: pooled,
            chunkTokenLimit: pooled)
        #expect(layout.chunkTokens == pooled)
        #expect(layout.ngramEmbedDim == config.ngramEmbedding.embedDim)
        #expect(layout.ngramEmbeddingElements
                == pooled * config.ngramEmbedding.embedDim)

        // And the buffer really is that big, since the guard that would have
        // caught the overflow reads its length rather than the layout.
        let ctx = try? MetalContext()
        guard let ctx else { return }
        guard let buffers = try? PrefillChunkScratchBuffers.allocate(
            device: ctx.device, layout: layout) else {
            Issue.record("scratch allocation failed"); return
        }
        #expect(buffers.ngramEmbedding.length
                >= pooled * config.ngramEmbedding.embedDim
                    * MemoryLayout<Float16>.stride)
    }

    /// An architecture without an n-gram layer must not pay for the staging.
    @Test func ngramStagingIsAbsentWithoutAnNgramLayer() {
        let layout = PrefillChunkScratchLayout(
            config: .qwen4ExpToy(), chunkTokens: 64)
        #expect(layout.ngramEmbedDim == 0)
        #expect(layout.ngramEmbeddingElements == 0)
    }

    /// A prompt split across several chunks must match one big chunk — this
    /// is what carries the recurrent state and the n-gram window across a
    /// boundary.
    @Test func severalChunksMatchOne() async throws {
        let config = ArchConfig.qwen4ExpToy()
        let tokens: [Int32] = [11, 4, 7, 2, 19, 3]

        let (dirA, ctxA, single) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dirA) }
        let oneLogits = try makeLogits(ctxA, vocab: config.vocabSize)
        _ = try await single.prefillChunked(
            tokens: tokens[...], startPosition: 0, outputMode: .logits,
            config: .production(chunkTokens: 32), into: oneLogits,
            onProgress: { _ in })
        let reference = Fp16Buffer.read(oneLogits, count: config.vocabSize)

        let (dirB, ctxB, split) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dirB) }
        let manyLogits = try makeLogits(ctxB, vocab: config.vocabSize)
        _ = try await split.prefillChunked(
            tokens: tokens[...], startPosition: 0, outputMode: .logits,
            config: .production(chunkTokens: 32), into: manyLogits,
            onProgress: { _ in })
        let actual = Fp16Buffer.read(manyLogits, count: config.vocabSize)
        #expect(RelError.compute(actual: actual, reference: reference) < 1e-3)
    }

    /// Prefill then decode: the continuation has to see the state the prefill
    /// left behind, or the recurrent halves of this architecture are stale.
    @Test func prefillThenDecodeMatchesPureDecode() async throws {
        let config = ArchConfig.qwen4ExpToy()
        let (dirA, ctxA, decodeRunner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dirA) }
        let a = try makeLogits(ctxA, vocab: config.vocabSize)
        for (position, token) in [Int32(11), 4, 7].enumerated() {
            try await decodeRunner.produce(token: token, position: position, into: a)
        }
        let reference = Fp16Buffer.read(a, count: config.vocabSize)

        let (dirB, ctxB, mixed) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: dirB) }
        let b = try makeLogits(ctxB, vocab: config.vocabSize)
        let head: [Int32] = [11, 4]
        _ = try await mixed.prefillChunked(
            tokens: head[...], startPosition: 0, outputMode: .logits,
            config: .production(chunkTokens: 32), into: b, onProgress: { _ in })
        try await mixed.produce(token: 7, position: 2, into: b)
        let actual = Fp16Buffer.read(b, count: config.vocabSize)
        let relErr = RelError.compute(actual: actual, reference: reference)
        #expect(relErr < 0.02, "relErr=\(relErr)")
    }
}
