import Foundation
import Testing
@testable import TUFFAppCore

/// Rewinding a conversation, the context estimate, and standing instructions.
@Suite(.serialized) @MainActor struct AppRewindAndInstructionsTests {
    private func makeStore() -> (URL, AppConversationStore) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-rewind-\(UUID().uuidString)", isDirectory: true)
        return (root, AppConversationStore(
            repository: AppConversationRepository(rootURL: root)))
    }

    private func record(_ store: AppConversationStore, _ prompt: String,
                        _ response: String) {
        store.recordCompletedTurn(
            AppChatTurn(prompt: prompt, response: response),
            attachments: [], modelID: "gemma4-e4b")
    }

    // MARK: - Truncation

    @Test func truncatingDropsTheTurnAndEverythingAfterIt() throws {
        let (root, store) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        record(store, "first", "one")
        record(store, "second", "two")
        record(store, "third", "three")
        let second = try #require(store.conversation.first { $0.prompt == "second" })

        let removed = store.truncate(from: second.id)

        #expect(removed?.prompt == "second")
        #expect(store.conversation.map(\.prompt) == ["first"])
        #expect(store.selectedConversation?.turns.map(\.prompt) == ["first"])
    }

    @Test func truncationIsWrittenThrough() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-rewind-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let repository = AppConversationRepository(rootURL: root)
        let first = AppConversationStore(repository: repository)
        first.recordCompletedTurn(
            AppChatTurn(prompt: "keep", response: "yes"),
            attachments: [], modelID: "gemma4-e4b")
        first.recordCompletedTurn(
            AppChatTurn(prompt: "drop", response: "no"),
            attachments: [], modelID: "gemma4-e4b")
        let doomed = try #require(first.conversation.last?.id)

        first.truncate(from: doomed)

        let reloaded = AppConversationStore(repository: repository)
        #expect(reloaded.conversation.map(\.prompt) == ["keep"])
    }

    @Test func truncatingAnUnknownTurnChangesNothing() {
        let (root, store) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        record(store, "only", "one")

        #expect(store.truncate(from: UUID()) == nil)
        #expect(store.conversation.count == 1)
    }

    // MARK: - Editing and regenerating

    @Test func editingAMessagePutsItBackInTheComposer() throws {
        let (root, store) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = AppModel(conversationStore: store)
        record(store, "first", "one")
        record(store, "second", "two")
        let second = try #require(model.conversation.first { $0.prompt == "second" })

        #expect(model.editMessage(second))

        #expect(model.promptText == "second")
        #expect(model.conversation.map(\.prompt) == ["first"],
                "the edited message and its answer should have been rewound")
    }

    /// A conversation cannot be rewound underneath a run that is reading it.
    @Test func rewindingIsRefusedWhileRunning() throws {
        let (root, store) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = AppModel(conversationStore: store)
        record(store, "first", "one")
        let turn = try #require(model.conversation.first)
        model.runState = .running

        #expect(!model.canRewind(to: turn))
        #expect(!model.editMessage(turn))
        #expect(model.conversation.count == 1)
    }

    // MARK: - Standing instructions

    @Test func aChatOverridesTheModelDefault() {
        let (root, store) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = AppModel(conversationStore: store)
        store.startNewConversation(modelID: "gemma4-e4b")
        model.modelSystemPrompt = "Be brief."

        #expect(model.effectiveSystemPrompt == "Be brief.")

        model.setConversationSystemPrompt("Answer in French.")
        #expect(model.effectiveSystemPrompt == "Answer in French.")
    }

    /// "This chat has none" is a different thing from "this chat follows the
    /// model", and must not quietly become the model's.
    @Test func anEmptyOverrideMeansNoInstructionsRatherThanTheDefault() {
        let (root, store) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = AppModel(conversationStore: store)
        store.startNewConversation(modelID: "gemma4-e4b")
        model.modelSystemPrompt = "Be brief."

        model.setConversationSystemPrompt("")
        #expect(model.effectiveSystemPrompt == nil)

        model.setConversationSystemPrompt(nil)
        #expect(model.effectiveSystemPrompt == "Be brief.")
    }

    @Test func theOverrideSurvivesAReload() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-rewind-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let repository = AppConversationRepository(rootURL: root)
        let first = AppConversationStore(repository: repository)
        first.recordCompletedTurn(
            AppChatTurn(prompt: "hi", response: "hello"),
            attachments: [], modelID: "gemma4-e4b")
        first.setSystemPrompt("Answer in French.")

        let reloaded = AppConversationStore(repository: repository)
        #expect(reloaded.selectedConversation?.systemPrompt == "Answer in French.")
    }

    // MARK: - Context estimate

    @Test func theEstimateGrowsWithTheDraftAndTheHistory() {
        let (root, store) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = AppModel(conversationStore: store)
        let empty = model.contextUsage.estimatedTokens

        model.promptText = String(repeating: "word ", count: 400)
        let withDraft = model.contextUsage.estimatedTokens
        #expect(withDraft > empty)

        record(store, String(repeating: "x", count: 4_000), "answer")
        #expect(model.contextUsage.estimatedTokens > withDraft)
    }

    @Test func aDraftLargerThanTheWindowIsReportedAsSuchBeforeSending() {
        let (root, store) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = AppModel(conversationStore: store)
        model.maxContextTokens = 4_096
        model.promptText = String(repeating: "x", count: 4 * 5_000)

        let usage = model.contextUsage
        #expect(usage.draftAloneOverflows)
        #expect(usage.fraction == 1)
    }

    @Test func anIdleChatReportsNothingWorthShowing() {
        let (root, store) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = AppModel(conversationStore: store)
        let usage = model.contextUsage
        #expect(usage.estimatedTokens == 0)
        #expect(!usage.isTight)
        #expect(!usage.willDropOldestTurns)
    }

    // MARK: - Trimming

    @Test func droppedTurnsReachTheApp() {
        let (root, store) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = AppModel(conversationStore: store)
        #expect(model.lastRunDroppedTurns == nil)

        model.diagnostics = AppDiagnostics(
            generatedTokens: 1, stopReason: .eos, promptTokenCount: 10,
            prefillSeconds: 0, timeToFirstTokenSeconds: 0, decodeSeconds: 1,
            tokensPerSecond: 1, peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions(), droppedTurns: 3)

        #expect(model.lastRunDroppedTurns == 3)
    }
}
