import Foundation
import Testing
import TurboFieldfare

@testable import TurboFieldfareAppCore

/// Multi-turn behavior: a finished exchange becomes context for the next one,
/// an unfinished one does not, and a conversation that outgrows the context
/// window loses its oldest turns rather than failing.
@Suite("Conversation history")
struct ConversationHistoryTests {

    private func request(prompt: String,
                         history: [AppChatTurn],
                         maxContext: Int = 4096) -> AppGenerationRequest {
        AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.gturbo"),
            prompt: prompt,
            history: history,
            maxContextTokens: maxContext)
    }

    @Test("A rendered conversation contains every turn in order")
    func rendersEveryTurn() async throws {
        let tokenizer = try await GFTokenizer.load()
        let history = [
            AppChatTurn(prompt: "What is prefill?", response: "Processing the prompt."),
            AppChatTurn(prompt: "And decode?", response: "Emitting one token at a time."),
        ]
        let rendered = try RealInferenceSession.renderConversation(
            request: request(prompt: "Which is slower?", history: history),
            tokenizer: tokenizer,
            maxContext: 4096)

        #expect(!rendered.trim.didTrim)
        #expect(rendered.trim.promptTokens == rendered.tokens.count)

        let text = tokenizer.decode(rendered.tokens, skipSpecialTokens: false)
        // Every turn, in order, with the new prompt last.
        let markers = ["What is prefill?", "Processing the prompt.",
                       "And decode?", "Emitting one token at a time.",
                       "Which is slower?"]
        var searchStart = text.startIndex
        for marker in markers {
            let found = try #require(
                text.range(of: marker, range: searchStart..<text.endIndex),
                "\(marker) missing or out of order")
            searchStart = found.upperBound
        }
    }

    @Test("An empty history renders exactly the single-turn prompt")
    func emptyHistoryMatchesSingleTurn() async throws {
        let tokenizer = try await GFTokenizer.load()
        let rendered = try RealInferenceSession.renderConversation(
            request: request(prompt: "Hello.", history: []),
            tokenizer: tokenizer,
            maxContext: 4096)
        let expected = tokenizer.encode(
            try tokenizer.applyChatTemplate([
                GFTokenizer.Message(role: .user, content: "Hello.")
            ]),
            addBOS: false)

        #expect(rendered.tokens == expected)
        #expect(rendered.trim.droppedTurns == 0)
    }

    @Test("Oldest turns are dropped until the conversation fits")
    func trimsOldestTurnsToFit() async throws {
        let tokenizer = try await GFTokenizer.load()
        let filler = String(repeating: "context ", count: 60)
        let history = (0..<12).map {
            AppChatTurn(prompt: "Question \($0). \(filler)",
                        response: "Answer \($0). \(filler)")
        }
        // Deliberately too small for the whole conversation.
        let maxContext = 512
        let rendered = try RealInferenceSession.renderConversation(
            request: request(prompt: "Final question.", history: history),
            tokenizer: tokenizer,
            maxContext: maxContext)

        #expect(rendered.trim.didTrim)
        #expect(rendered.tokens.count < maxContext)
        let text = tokenizer.decode(rendered.tokens, skipSpecialTokens: false)
        // The newest turns survive; the oldest are gone.
        #expect(text.contains("Final question."))
        #expect(!text.contains("Question 0."))
        #expect(text.contains("Question 11."))
    }

    @Test("A prompt that cannot fit on its own is still a context overflow")
    func promptAloneOverflowing() async throws {
        let tokenizer = try await GFTokenizer.load()
        let huge = String(repeating: "overflow ", count: 4000)
        await #expect(throws: AppInferenceError.self) {
            _ = try RealInferenceSession.renderConversation(
                request: request(prompt: huge, history: [
                    AppChatTurn(prompt: "earlier", response: "earlier"),
                ]),
                tokenizer: tokenizer,
                maxContext: 256)
        }
    }
}

/// The state side: which exchanges become history, and which do not. Driven
/// through the real run path with the fake client, so the assertions cover
/// AppModel's actual completion handling rather than a shortcut into it.
@Suite("Conversation state")
struct ConversationStateTests {

    @MainActor
    private func loadedModel(_ tag: String) async throws -> (AppModel, URL) {
        let directory = try makeCompleteModelInstall("conversation-\(tag)")
        let model = AppModel(
            modelDirectory: directory,
            client: FakeInferenceClient(eventDelay: .zero),
            installer: MockModelInstallerClient())
        model.loadModel()
        try await waitForFakeClient { model.loadState.isReady }
        return (model, directory)
    }

    @MainActor
    private func ask(_ model: AppModel, _ prompt: String) async throws {
        model.promptText = prompt
        model.run()
        try await waitForFakeClient { model.diagnostics != nil && !model.isRunning }
    }

    @MainActor
    @Test func aFinishedExchangeBecomesHistory() async throws {
        let (model, directory) = try await loadedModel("finished")
        defer { try? FileManager.default.removeItem(at: directory) }

        try await ask(model, "First question")

        #expect(model.conversation.count == 1)
        #expect(model.conversation[0].prompt == "First question")
        #expect(!model.conversation[0].response.isEmpty)
    }

    @MainActor
    @Test func theNextRequestCarriesTheConversation() async throws {
        let (model, directory) = try await loadedModel("carries")
        defer { try? FileManager.default.removeItem(at: directory) }

        try await ask(model, "First question")
        model.promptText = "Follow-up"
        let request = try model.makeRequest()

        #expect(request.prompt == "Follow-up")
        #expect(request.history.count == 1)
        #expect(request.history[0].prompt == "First question")
    }

    @MainActor
    @Test func historyAccumulatesAcrossTurns() async throws {
        let (model, directory) = try await loadedModel("accumulates")
        defer { try? FileManager.default.removeItem(at: directory) }

        try await ask(model, "One")
        try await ask(model, "Two")
        try await ask(model, "Three")

        #expect(model.conversation.map(\.prompt) == ["One", "Two", "Three"])
        #expect(model.conversationTurnCount == 3)
    }

    @MainActor
    @Test func startingANewConversationDropsHistory() async throws {
        let (model, directory) = try await loadedModel("new")
        defer { try? FileManager.default.removeItem(at: directory) }

        try await ask(model, "First question")
        #expect(model.hasOutputTranscript)

        model.clearOutput()

        #expect(model.conversation.isEmpty)
        #expect(!model.hasOutputTranscript)
        model.promptText = "Fresh start"
        #expect(try model.makeRequest().history.isEmpty)
    }

    @MainActor
    @Test func aCancelledExchangeLeavesNoHistory() async throws {
        let directory = try makeCompleteModelInstall("conversation-cancelled")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(
            modelDirectory: directory,
            client: FakeInferenceClient(eventDelay: .milliseconds(5)),
            installer: MockModelInstallerClient())
        model.loadModel()
        try await waitForFakeClient { model.loadState.isReady }

        model.promptText = "Cancel me"
        model.run()
        try await waitForFakeClient { !model.outputResponsePlainText.isEmpty }
        model.cancel()
        try await waitForFakeClient { !model.isRunning }

        // A truncated answer must not become context the model treats as its
        // own settled output.
        #expect(model.conversation.isEmpty)
    }
}

/// Poll the main actor until `predicate` holds or the budget runs out.
@MainActor
private func waitForFakeClient(_ predicate: @escaping @MainActor () -> Bool) async throws {
    for _ in 0..<400 {
        if predicate() { return }
        try await Task.sleep(for: .milliseconds(5))
    }
    Issue.record("timed out waiting for conversation state")
}
