import Foundation
import Testing
import TUFFEngine
@testable import TUFFAppCore

@Suite struct AppPromptReuseStateTests {
    @Test func bridgesOnlyTheUnchangedCompletedExchange() async throws {
        let tokenizer = try await GFTokenizer.load()
        let initial = AppGenerationRequest(modelDirectory: URL(fileURLWithPath: "/tmp/model"),
                                           prompt: "first")
        let result = RawDecodeResult(prefillTokens: 2, cachedPromptTokens: 0,
            computedPrefillTokens: 2, prefillSeconds: 0, newTokens: 2,
            decodeSeconds: 0, reason: .endOfTurn, kvPosition: 3,
            kvBackedTokenIDs: [1, 2, 3], uncommittedBoundaryTokenIDs: [tokenizer.endOfTurnID])
        var cache = AppPromptReuseState()
        cache.record(result, hasImages: false, request: initial, response: "answer")
        var next = initial
        next.prompt = "second"
        next.history = [AppChatTurn(prompt: "first", response: "answer")]
        let bridge = tokenizer.encodeTextContinuation(userContent: "second",
                                                       modelVariant: .gemma4_E4B)
        #expect(cache.textContinuation(request: next, tokenizer: tokenizer,
            modelVariant: .gemma4_E4B) == [1, 2, 3] + bridge)
        for change in 0..<6 {
            var edited = next
            switch change {
            case 0: edited.history[0].response = "edited"
            case 1: edited.systemPrompt = "new instructions"
            case 2: edited.reasoning = .on
            case 3: edited.history[0].prompt = "edited"
            case 4: edited.modelDirectory = URL(fileURLWithPath: "/tmp/other-model")
            default: edited.assistantPrefix = "continue"
            }
            #expect(cache.textContinuation(request: edited, tokenizer: tokenizer,
                modelVariant: .gemma4_E4B) == nil)
        }
    }
    private func result(position: Int = 3) -> RawDecodeResult {
        RawDecodeResult(prefillTokens: 2, cachedPromptTokens: 0,
                        computedPrefillTokens: 2, prefillSeconds: 0,
                        newTokens: 2, decodeSeconds: 0, reason: .endOfTurn,
                        kvPosition: position, kvBackedTokenIDs: [1, 2, 3],
                        uncommittedBoundaryTokenIDs: [4])
    }

    @Test func followupReusesOnlyTokensAlreadyBackedByRunnerState() {
        var cache = AppPromptReuseState()
        cache.record(result(), hasImages: false)
        // The uncommitted final output token still needs to be processed.
        #expect(cache.takeStart(promptIDs: [1, 2, 3, 4, 5], position: 3,
                                hasImages: false) == .resume(cachedPromptTokens: 3))
        // A failure or cancellation after takeStart cannot reuse old state.
        #expect(cache.takeStart(promptIDs: [1, 2, 3, 4, 5], position: 3,
                                hasImages: false) == .reset)
    }

    @Test func editsTrimmingRegenerationAndCursorMismatchStartFresh() {
        for (ids, position) in [([1, 8, 3, 4], 3), ([2, 3, 4], 3),
                                ([1, 2], 3), ([1, 2, 3], 3), ([1, 2, 3, 4], 4)] {
            var cache = AppPromptReuseState()
            cache.record(result(), hasImages: false)
            #expect(cache.takeStart(promptIDs: ids.map(Int32.init), position: position,
                                    hasImages: false) == .reset)
        }
    }

    @Test func imagesInvalidResultsAndUnloadCannotReuseTokenOnlyState() {
        for mode in 0..<4 {
            var cache = AppPromptReuseState()
            cache.record(result(position: mode == 0 ? 2 : 3), hasImages: mode == 1)
            if mode == 2 { cache.clear() }
            #expect(cache.takeStart(promptIDs: [1, 2, 3, 4], position: 3,
                                    hasImages: mode == 3) == .reset)
        }
    }
}
