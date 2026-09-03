import Foundation
import Metal
import Testing

@testable import TUFFEngine

extension RawCompletionLoopTests {
    /// A deterministic target with an explicit logical cursor. Verification
    /// advances the cursor speculatively, while commit rewinds it to the
    /// accepted prefix. This is deliberately independent of Metal kernels so
    /// the generation policy can be tested before production backends exist.
    final class ScriptedSpeculativeTarget: SpeculativeVerificationRunner,
        ContinuableLogitProducer, ContextWindowReporting, @unchecked Sendable
    {
        let vocabSize: Int
        let maxContext: Int = Int.max
        let promptCount: Int
        let targetTokenIDs: [Int32]
        private(set) var cursor: Int
        var continuationPosition: Int { cursor }
        private(set) var resetCalls = 0
        private(set) var prepareCalls: [Int] = []
        private(set) var commitCounts: [Int] = []
        private(set) var produceCalls = 0
        var cancelDuringVerification = false

        private var activeStart: Int?
        private var activeCount = 0

        init(vocabSize: Int,
             promptCount: Int,
             targetTokenIDs: [Int32],
             initialCursor: Int = 0) {
            self.vocabSize = vocabSize
            self.promptCount = promptCount
            self.targetTokenIDs = targetTokenIDs
            self.cursor = initialCursor
        }

        func reset() {
            resetCalls += 1
            cursor = 0
            activeStart = nil
            activeCount = 0
            commitCounts.removeAll(keepingCapacity: true)
            produceCalls = 0
        }

        func prepareForContinuation(expectedPosition: Int) throws {
            guard cursor == expectedPosition else {
                throw PrefillError.prefillCursorMismatch(
                    "scripted speculative cursor mismatch")
            }
            prepareCalls.append(expectedPosition)
        }

        func produce(token: Int32, position: Int, into logits: MTLBuffer) async throws {
            guard cursor == position else {
                throw PrefillError.prefillCursorMismatch(
                    "scripted speculative scalar cursor mismatch")
            }
            writePrediction(at: predictionIndex(for: position), into: logits)
            cursor += 1
            produceCalls += 1
        }

        func verifySpeculativeBlock(tokens: [Int32],
                                    startPosition: Int,
                                    into logits: MTLBuffer) async throws
            -> SpeculativeVerificationResult
        {
            guard cursor == startPosition else {
                throw PrefillError.prefillCursorMismatch(
                    "scripted speculative verification cursor mismatch")
            }
            let start = max(0, startPosition - promptCount)
            let predictions = (0...tokens.count).map {
                targetToken(at: start + $0)
            }
            writePrediction(at: start, into: logits)
            activeStart = startPosition
            activeCount = tokens.count
            cursor = startPosition + tokens.count

            if cancelDuringVerification {
                withUnsafeCurrentTask { $0?.cancel() }
                throw CancellationError()
            }

            return SpeculativeVerificationResult(
                startPosition: startPosition,
                proposedTokenIDs: tokens,
                targetTokenIDs: predictions,
                processedTokens: tokens.count,
                newPosition: cursor,
                metrics: SpeculativeVerificationMetrics(targetCommandBuffers: 1))
        }

        func commitSpeculativePrefix(_ count: Int) throws {
            guard let activeStart else {
                throw SpeculativeDecodingError.noActiveTransaction
            }
            guard (0...activeCount).contains(count) else {
                throw SpeculativeDecodingError.invalidCommitCount(
                    requested: count, processed: activeCount)
            }
            cursor = activeStart + count
            commitCounts.append(count)
            self.activeStart = nil
            activeCount = 0
        }

        private func predictionIndex(for position: Int) -> Int {
            max(0, position - promptCount + 1)
        }

        private func targetToken(at index: Int) -> Int32 {
            guard targetTokenIDs.indices.contains(index) else {
                return targetTokenIDs.last ?? 0
            }
            return targetTokenIDs[index]
        }

        private func writePrediction(at index: Int, into logits: MTLBuffer) {
            let pointer = logits.contents().bindMemory(to: Float16.self,
                                                        capacity: vocabSize)
            for token in 0..<vocabSize { pointer[token] = -30 }
            let prediction = targetToken(at: index)
            guard targetTokenIDs.indices.contains(index),
                  prediction >= 0, Int(prediction) < vocabSize else { return }
            pointer[Int(prediction)] = 30
        }
    }

    final class ScriptedDraftProducer: DraftTokenProducer, @unchecked Sendable {
        let targetTokenIDs: [Int32]
        let promptCount: Int
        let overrides: [[Int32]]
        var cancelOnFirstProposal = false
        private(set) var calls = 0

        init(targetTokenIDs: [Int32], promptCount: Int,
             overrides: [[Int32]] = []) {
            self.targetTokenIDs = targetTokenIDs
            self.promptCount = promptCount
            self.overrides = overrides
        }

        func reset() { calls = 0 }

        func propose(history: [Int32], maxTokens: Int,
                     startPosition: Int) async throws -> DraftProposal {
            defer { calls += 1 }
            if cancelOnFirstProposal, calls == 0 {
                withUnsafeCurrentTask { $0?.cancel() }
            }
            if calls < overrides.count {
                return DraftProposal(tokenIDs: Array(overrides[calls].prefix(maxTokens)))
            }
            let start = max(0, history.count - promptCount)
            guard start < targetTokenIDs.count else {
                return DraftProposal(tokenIDs: [])
            }
            return DraftProposal(tokenIDs: Array(
                targetTokenIDs.dropFirst(start).prefix(maxTokens)))
        }
    }

    private func oneToken(_ text: String, tokenizer: GFTokenizer) -> Int32 {
        tokenizer.encode(text, addBOS: false).first!
    }

    private func runScriptedCase(
        targetTokenIDs: [Int32],
        tokenizer: GFTokenizer,
        promptIDs: [Int32],
        config: GenerationConfig,
        overrides: [[Int32]] = [],
        start: RawCompletionStart = .reset,
        initialCursor: Int = 0,
        cancelDuringVerification: Bool = false,
        draftProducer: ScriptedDraftProducer? = nil
    ) async throws -> (RawDecodeResult, ScriptedSpeculativeTarget) {
        let context = try MetalContext()
        let producer = ScriptedSpeculativeTarget(
            vocabSize: tokenizer.vocabSize,
            promptCount: promptIDs.count,
            targetTokenIDs: targetTokenIDs,
            initialCursor: initialCursor)
        producer.cancelDuringVerification = cancelDuringVerification
        let scratch = try RawCompletionScratch(context: context,
                                                vocab: tokenizer.vocabSize)
        let drafter = draftProducer ?? ScriptedDraftProducer(
            targetTokenIDs: targetTokenIDs,
            promptCount: promptIDs.count,
            overrides: overrides)
        let result = try await runRawCompletion(
            producer: producer,
            tokenizer: tokenizer,
            promptIds: promptIDs,
            config: config,
            context: context,
            scratch: scratch,
            prefillConfig: .off,
            start: start,
            draftProducer: drafter) { _ in }
        return (result, producer)
    }

    @Test func speculativeOutputMatchesBaselineWhenAllCandidatesAccepted() async throws {
        let tokenizer = try await GFTokenizer.load()
        let prompt = tokenizer.encode("go", addBOS: true)
        let target = ["a", "b", "c", "d"].map {
            oneToken($0, tokenizer: tokenizer)
        } + [tokenizer.eosID]
        let config = GenerationConfig(maxNewTokens: 20, temperature: 0)

        let context = try MetalContext()
        let baselineProducer = ScriptedSpeculativeTarget(
            vocabSize: tokenizer.vocabSize,
            promptCount: prompt.count,
            targetTokenIDs: target)
        let baseline = try await runRawCompletion(
            producer: baselineProducer, tokenizer: tokenizer, promptIds: prompt,
            config: config, context: context,
            scratch: try RawCompletionScratch(context: context,
                                               vocab: tokenizer.vocabSize),
            prefillConfig: .off) { _ in }

        let (speculative, producer) = try await runScriptedCase(
            targetTokenIDs: target, tokenizer: tokenizer, promptIDs: prompt,
            config: GenerationConfig(
                maxNewTokens: 20, temperature: 0,
                speculative: SpeculativeDecodeConfig(mode: .greedy,
                                                     draftTokens: 4)))

        #expect(speculative.kvBackedTokenIDs == baseline.kvBackedTokenIDs)
        #expect(speculative.uncommittedBoundaryTokenIDs
                == baseline.uncommittedBoundaryTokenIDs)
        #expect(speculative.reason == baseline.reason)
        #expect(speculative.newTokens == baseline.newTokens)
        #expect(speculative.kvPosition == baseline.kvPosition)
        #expect(producer.commitCounts == [4])
        #expect(speculative.speculative.acceptedTokens == 4)
        #expect(speculative.speculative.correctionTokens == 0)
    }

    @Test func automaticSpeculationGrowsFromAConservativeBlock() async throws {
        let tokenizer = try await GFTokenizer.load()
        let prompt = tokenizer.encode("go", addBOS: true)
        let target = ["a", "b", "c", "d", "e", "f", "g"].map {
            oneToken($0, tokenizer: tokenizer)
        } + [tokenizer.eosID]

        let (result, producer) = try await runScriptedCase(
            targetTokenIDs: target,
            tokenizer: tokenizer,
            promptIDs: prompt,
            config: GenerationConfig(
                maxNewTokens: 20,
                temperature: 0,
                speculative: SpeculativeDecodeConfig(mode: .auto,
                                                     draftTokens: 8)))

        #expect(result.kvBackedTokenIDs == prompt + Array(target.dropLast()))
        #expect(result.uncommittedBoundaryTokenIDs == [tokenizer.eosID])
        #expect(result.speculative.minimumVerificationBlockTokens == 2)
        #expect(result.speculative.maximumVerificationBlockTokens == 4)
        #expect(!result.speculative.adaptiveDisabled)
        #expect(producer.commitCounts == [2, 4])
    }

    @Test func firstRejectionDoesNotLeakDraftTokens() async throws {
        let tokenizer = try await GFTokenizer.load()
        let prompt = tokenizer.encode("go", addBOS: true)
        let target = ["a", "b", "c", "d"].map {
            oneToken($0, tokenizer: tokenizer)
        } + [tokenizer.eosID]
        let bad = oneToken("z", tokenizer: tokenizer)
        let config = GenerationConfig(
            maxNewTokens: 20, temperature: 0,
            speculative: SpeculativeDecodeConfig(mode: .greedy,
                                                 draftTokens: 4))

        let (result, producer) = try await runScriptedCase(
            targetTokenIDs: target, tokenizer: tokenizer, promptIDs: prompt,
            config: config, overrides: [[bad, bad, bad, bad]])

        #expect(result.kvBackedTokenIDs == prompt + Array(target.dropLast()))
        #expect(result.uncommittedBoundaryTokenIDs == [tokenizer.eosID])
        #expect(result.speculative.acceptedTokens == 4)
        #expect(result.speculative.rejectedTokens == 4)
        #expect(result.speculative.correctionTokens == 1)
        #expect(producer.commitCounts == [0, 3])
        #expect(!result.kvBackedTokenIDs.contains(bad))
    }

    @Test func partialAcceptanceCommitsOnlyTheMatchingPrefix() async throws {
        let tokenizer = try await GFTokenizer.load()
        let prompt = tokenizer.encode("go", addBOS: true)
        let target = ["a", "b", "c", "d"].map {
            oneToken($0, tokenizer: tokenizer)
        } + [tokenizer.eosID]
        let bad = oneToken("z", tokenizer: tokenizer)
        let config = GenerationConfig(
            maxNewTokens: 20, temperature: 0,
            speculative: SpeculativeDecodeConfig(mode: .greedy,
                                                 draftTokens: 4))

        let (result, producer) = try await runScriptedCase(
            targetTokenIDs: target, tokenizer: tokenizer, promptIDs: prompt,
            config: config, overrides: [[target[0], target[1], bad, bad]])

        #expect(result.kvBackedTokenIDs == prompt + Array(target.dropLast()))
        #expect(result.uncommittedBoundaryTokenIDs == [tokenizer.eosID])
        #expect(producer.commitCounts == [2, 1])
        #expect(result.speculative.acceptedTokens == 4)
        #expect(result.speculative.rejectedTokens == 2)
        #expect(result.speculative.correctionTokens == 1)
    }

    @Test func stopStringAcrossSpeculativeTokensMatchesScalarDecode() async throws {
        let tokenizer = try await GFTokenizer.load()
        let prompt = tokenizer.encode("go", addBOS: true)
        let a = oneToken("a", tokenizer: tokenizer)
        let b = oneToken("b", tokenizer: tokenizer)
        let c = oneToken("c", tokenizer: tokenizer)
        let target = [a, b, c, tokenizer.eosID]
        let stop = tokenizer.decode([a, b], skipSpecialTokens: true)
        let config = GenerationConfig(maxNewTokens: 20, temperature: 0,
                                      stopStrings: [stop])
        let context = try MetalContext()
        let baseline = try await runRawCompletion(
            producer: ScriptedSpeculativeTarget(
                vocabSize: tokenizer.vocabSize,
                promptCount: prompt.count,
                targetTokenIDs: target),
            tokenizer: tokenizer, promptIds: prompt, config: config,
            context: context,
            scratch: try RawCompletionScratch(context: context,
                                               vocab: tokenizer.vocabSize),
            prefillConfig: .off) { _ in }
        let (speculative, producer) = try await runScriptedCase(
            targetTokenIDs: target, tokenizer: tokenizer, promptIDs: prompt,
            config: GenerationConfig(
                maxNewTokens: 20, temperature: 0, stopStrings: [stop],
                speculative: SpeculativeDecodeConfig(mode: .greedy,
                                                     draftTokens: 4)))

        #expect(speculative.reason == .stopString)
        #expect(speculative.reason == baseline.reason)
        #expect(speculative.kvBackedTokenIDs == baseline.kvBackedTokenIDs)
        #expect(speculative.uncommittedBoundaryTokenIDs
                == baseline.uncommittedBoundaryTokenIDs)
        #expect(producer.commitCounts == [1])
    }

    @Test func maxTokensInsideBlockLeavesOnlyTheVisiblePrefixInKV() async throws {
        let tokenizer = try await GFTokenizer.load()
        let prompt = tokenizer.encode("go", addBOS: true)
        let target = ["a", "b", "c", "d"].map {
            oneToken($0, tokenizer: tokenizer)
        } + [tokenizer.eosID]
        let config = GenerationConfig(
            maxNewTokens: 2, temperature: 0,
            speculative: SpeculativeDecodeConfig(mode: .greedy,
                                                 draftTokens: 8))
        let (result, producer) = try await runScriptedCase(
            targetTokenIDs: target, tokenizer: tokenizer, promptIDs: prompt,
            config: config)

        #expect(result.reason == .maxTokens)
        #expect(result.newTokens == 2)
        #expect(result.kvBackedTokenIDs == prompt + [target[0]])
        #expect(result.uncommittedBoundaryTokenIDs == [target[1]])
        #expect(result.kvPosition == prompt.count + 1)
        #expect(producer.commitCounts == [1])
    }

    @Test func eosInsideAcceptedBlockStopsWithoutAnExtraTargetDecode() async throws {
        let tokenizer = try await GFTokenizer.load()
        let prompt = tokenizer.encode("go", addBOS: true)
        let target = [oneToken("a", tokenizer: tokenizer), tokenizer.eosID,
                      oneToken("b", tokenizer: tokenizer)]
        let (result, producer) = try await runScriptedCase(
            targetTokenIDs: target, tokenizer: tokenizer, promptIDs: prompt,
            config: GenerationConfig(
                maxNewTokens: 20, temperature: 0,
                speculative: SpeculativeDecodeConfig(mode: .greedy,
                                                     draftTokens: 4)))

        #expect(result.reason == .eos)
        #expect(result.newTokens == 2)
        #expect(result.kvBackedTokenIDs == prompt + [target[0]])
        #expect(result.uncommittedBoundaryTokenIDs == [tokenizer.eosID])
        #expect(result.kvPosition == prompt.count + 1)
        #expect(producer.commitCounts == [1])
        #expect(producer.produceCalls == prompt.count)
    }

    @Test func eosCorrectionStopsWithoutAnExtraTargetDecode() async throws {
        let tokenizer = try await GFTokenizer.load()
        let prompt = tokenizer.encode("go", addBOS: true)
        let a = oneToken("a", tokenizer: tokenizer)
        let bad = oneToken("z", tokenizer: tokenizer)
        let target = [a, tokenizer.eosID, oneToken("b", tokenizer: tokenizer)]
        let (result, producer) = try await runScriptedCase(
            targetTokenIDs: target, tokenizer: tokenizer, promptIDs: prompt,
            config: GenerationConfig(
                maxNewTokens: 20, temperature: 0,
                speculative: SpeculativeDecodeConfig(mode: .greedy,
                                                     draftTokens: 4)),
            overrides: [[a, bad, bad, bad]])

        #expect(result.reason == .eos)
        #expect(result.newTokens == 2)
        #expect(result.kvBackedTokenIDs == prompt + [a])
        #expect(result.uncommittedBoundaryTokenIDs == [tokenizer.eosID])
        #expect(result.kvPosition == prompt.count + 1)
        #expect(producer.commitCounts == [1])
        #expect(producer.produceCalls == prompt.count)
    }

    @Test func nonGreedyGenerationDoesNotSilentlyUseGreedySpeculation() async throws {
        let tokenizer = try await GFTokenizer.load()
        let prompt = tokenizer.encode("go", addBOS: true)
        let target = [oneToken("a", tokenizer: tokenizer), tokenizer.eosID]
        let config = GenerationConfig(
            maxNewTokens: 5, temperature: 0.7, seed: 7,
            speculative: SpeculativeDecodeConfig(mode: .greedy,
                                                 draftTokens: 4))
        let (result, _) = try await runScriptedCase(
            targetTokenIDs: target, tokenizer: tokenizer, promptIDs: prompt,
            config: config)
        #expect(result.speculative.rounds == 0)
        #expect(result.speculative.proposedTokens == 0)
    }

    @Test func cancellationDuringDraftingDoesNotTouchTargetTail() async throws {
        let tokenizer = try await GFTokenizer.load()
        let prompt = tokenizer.encode("go", addBOS: true)
        let target = [oneToken("a", tokenizer: tokenizer), tokenizer.eosID]
        let context = try MetalContext()
        let producer = ScriptedSpeculativeTarget(
            vocabSize: tokenizer.vocabSize, promptCount: prompt.count,
            targetTokenIDs: target)
        let drafter = ScriptedDraftProducer(targetTokenIDs: target,
                                            promptCount: prompt.count)
        drafter.cancelOnFirstProposal = true
        let task = Task {
            try await runRawCompletion(
                producer: producer, tokenizer: tokenizer, promptIds: prompt,
                config: GenerationConfig(
                    maxNewTokens: 5, temperature: 0,
                    speculative: SpeculativeDecodeConfig(mode: .greedy,
                                                         draftTokens: 4)),
                context: context,
                scratch: try RawCompletionScratch(context: context,
                                                   vocab: tokenizer.vocabSize),
                prefillConfig: .off, draftProducer: drafter) { _ in }
        }
        await #expect(throws: CancellationError.self) { _ = try await task.value }
        #expect(producer.cursor == prompt.count)
        #expect(producer.commitCounts.isEmpty)
    }

    @Test func cancellationDuringVerificationDiscardsTheWholeTail() async throws {
        let tokenizer = try await GFTokenizer.load()
        let prompt = tokenizer.encode("go", addBOS: true)
        let target = [oneToken("a", tokenizer: tokenizer), tokenizer.eosID]
        let producer = ScriptedSpeculativeTarget(
            vocabSize: tokenizer.vocabSize, promptCount: prompt.count,
            targetTokenIDs: target)
        producer.cancelDuringVerification = true
        let context = try MetalContext()
        let drafter = ScriptedDraftProducer(targetTokenIDs: target,
                                            promptCount: prompt.count)
        let task = Task {
            try await runRawCompletion(
                producer: producer, tokenizer: tokenizer, promptIds: prompt,
                config: GenerationConfig(
                    maxNewTokens: 5, temperature: 0,
                    speculative: SpeculativeDecodeConfig(mode: .greedy,
                                                         draftTokens: 4)),
                context: context,
                scratch: try RawCompletionScratch(context: context,
                                                   vocab: tokenizer.vocabSize),
                prefillConfig: .off, draftProducer: drafter) { _ in }
        }
        await #expect(throws: CancellationError.self) { _ = try await task.value }
        #expect(producer.cursor == prompt.count)
        #expect(producer.commitCounts == [0])
    }

    @Test func cancellationDuringAcceptedEmissionCommitsVisiblePrefix() async throws {
        let tokenizer = try await GFTokenizer.load()
        let prompt = tokenizer.encode("go", addBOS: true)
        let target = ["a", "b", "c"].map {
            oneToken($0, tokenizer: tokenizer)
        } + [tokenizer.eosID]
        let producer = ScriptedSpeculativeTarget(
            vocabSize: tokenizer.vocabSize, promptCount: prompt.count,
            targetTokenIDs: target)
        let drafter = ScriptedDraftProducer(targetTokenIDs: target,
                                            promptCount: prompt.count)
        let context = try MetalContext()
        let task = Task {
            try await runRawCompletion(
                producer: producer, tokenizer: tokenizer, promptIds: prompt,
                config: GenerationConfig(
                    maxNewTokens: 5, temperature: 0,
                    speculative: SpeculativeDecodeConfig(mode: .greedy,
                                                         draftTokens: 4)),
                context: context,
                scratch: try RawCompletionScratch(context: context,
                                                   vocab: tokenizer.vocabSize),
                prefillConfig: .off, draftProducer: drafter) { progress in
                    if case .token(let index, _, _) = progress, index == 0 {
                        withUnsafeCurrentTask { $0?.cancel() }
                    }
                }
        }
        await #expect(throws: CancellationError.self) { _ = try await task.value }
        #expect(producer.cursor == prompt.count + 1)
        #expect(producer.commitCounts == [1])
    }

    @Test func continuationAfterSpeculationKeepsCachedHistoryAndCursorAligned() async throws {
        let tokenizer = try await GFTokenizer.load()
        let prompt = tokenizer.encode("one two three four", addBOS: true)
        let cached = prompt.count - 1
        let target = [oneToken("a", tokenizer: tokenizer), tokenizer.eosID]
        let producer = ScriptedSpeculativeTarget(
            vocabSize: tokenizer.vocabSize, promptCount: prompt.count,
            targetTokenIDs: target, initialCursor: cached)
        let drafter = ScriptedDraftProducer(targetTokenIDs: target,
                                            promptCount: prompt.count)
        let context = try MetalContext()
        let result = try await runRawCompletion(
            producer: producer, tokenizer: tokenizer, promptIds: prompt,
            config: GenerationConfig(
                maxNewTokens: 5, temperature: 0,
                speculative: SpeculativeDecodeConfig(mode: .greedy,
                                                     draftTokens: 4)),
            context: context,
            scratch: try RawCompletionScratch(context: context,
                                               vocab: tokenizer.vocabSize),
            prefillConfig: .off,
            start: .resume(cachedPromptTokens: cached),
            draftProducer: drafter) { _ in }

        #expect(producer.prepareCalls == [cached])
        #expect(result.kvBackedTokenIDs == prompt + [target[0]])
        #expect(result.kvPosition == prompt.count + 1)
        #expect(result.uncommittedBoundaryTokenIDs == [tokenizer.eosID])
    }
}
