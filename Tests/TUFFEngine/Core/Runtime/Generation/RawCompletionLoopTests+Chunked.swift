import Testing

@testable import TUFFEngine

extension RawCompletionLoopTests {
    /// A producer with no chunked path runs the prompt a token at a time
    /// rather than refusing the turn. Prefill mode is a performance choice —
    /// the same reasoning `coercedForImagePrompt` applies in the other
    /// direction — and a hyper-connection architecture has no chunked kernels
    /// at all, so refusing here meant Qwen3.8 Flash Next could not answer.
    @Test func chunkedModeFallsBackWhenTheProducerHasNoChunkedPath() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let tokenA = tokenizer.encode("a", addBOS: false).first!
        let promptIDs = tokenizer.encode("one two three", addBOS: true)
        let producer = CountingProducer(
            vocabSize: tokenizer.vocabSize,
            step: automaton([tokenA], end: tokenizer.eosID))
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        var prefills: [(Int, Int)] = []

        let result = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: promptIDs,
            config: GenerationConfig(maxNewTokens: 4, temperature: 0),
            context: context,
            scratch: scratch,
            prefillConfig: .production(chunkTokens: 32)) { progress in
                if case .prefill(let done, let total) = progress {
                    prefills.append((done, total))
                }
            }

        // Every prompt token went through the sequential entry point, and the
        // turn produced its token rather than throwing.
        #expect(producer.produceCalls >= promptIDs.count)
        #expect(result.newTokens >= 1)
        #expect(prefills.count == promptIDs.count)
        #expect(prefills.last?.0 == promptIDs.count)
        #expect(prefills.last?.1 == promptIDs.count)
    }

    @Test func chunkedModeUsesChunkedRunnerEntryPoint() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let tokenA = tokenizer.encode("a", addBOS: false).first!
        let producer = ChunkedTestProducer(vocabSize: tokenizer.vocabSize, firstToken: tokenA)
        let promptIDs = tokenizer.encode("go", addBOS: true)
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        var prefills: [(Int, Int)] = []

        let result = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: promptIDs,
            config: GenerationConfig(maxNewTokens: 1, temperature: 0),
            context: context,
            scratch: scratch,
            prefillConfig: .production(chunkTokens: 32)) { progress in
                if case .prefill(let done, let total) = progress {
                    prefills.append((done, total))
                }
            }

        #expect(result.newTokens == 1)
        #expect(producer.chunkedCalls == 1)
        #expect(producer.produceCalls == 0)
        #expect(producer.lastOutputMode == .logits)
        #expect(producer.lastConfig == .production(chunkTokens: 32))
        #expect(prefills.count == 1)
        #expect(prefills.first?.0 == promptIDs.count)
        #expect(prefills.first?.1 == promptIDs.count)
    }

    @Test func chunkedLogitsSeedProducesFirstToken() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let tokenA = tokenizer.encode("a", addBOS: false).first!
        let producer = ChunkedTestProducer(vocabSize: tokenizer.vocabSize, firstToken: tokenA)
        let promptIDs = tokenizer.encode("go", addBOS: true)
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)
        var tokens: [Int32] = []

        let result = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: promptIDs,
            config: GenerationConfig(maxNewTokens: 1, temperature: 0),
            context: context,
            scratch: scratch,
            prefillConfig: .production(chunkTokens: 32)) { progress in
                if case .token(_, let id, _) = progress {
                    tokens.append(id)
                }
            }

        #expect(result.newTokens == 1)
        #expect(tokens == [tokenA])
        #expect(producer.chunkedCalls == 1)
        #expect(producer.produceCalls == 0)
        #expect(producer.lastOutputMode == .logits)
    }

    @Test func chunkedPrefillRejectsGreedySeedWhenLogitsRequested() async throws {
        let context = try MetalContext()
        let tokenizer = try await GFTokenizer.load()
        let tokenA = tokenizer.encode("a", addBOS: false).first!
        let producer = ChunkedTestProducer(
            vocabSize: tokenizer.vocabSize,
            firstToken: tokenA,
            seed: .greedyToken(UInt32(bitPattern: tokenA)))
        let promptIDs = tokenizer.encode("go", addBOS: true)
        let scratch = try RawCompletionScratch(context: context, vocab: tokenizer.vocabSize)

        do {
            _ = try await runRawCompletion(
                producer: producer,
                tokenizer: tokenizer,
                promptIds: promptIDs,
                config: GenerationConfig(maxNewTokens: 1, temperature: 0.7),
                context: context,
                scratch: scratch,
                prefillConfig: .production(chunkTokens: 32)) { _ in }
            Issue.record("expected unsupported chunked prefill seed")
        } catch let error as PrefillError {
            guard case .unsupportedPrefillSeed(let reason) = error else {
                Issue.record("unexpected PrefillError \(error)")
                return
            }
            #expect(reason.contains("RawCompletion chunked prefill requested logits"))
        }

        #expect(producer.chunkedCalls == 1)
        #expect(producer.produceCalls == 0)
        #expect(producer.lastOutputMode == .logits)
    }
}
