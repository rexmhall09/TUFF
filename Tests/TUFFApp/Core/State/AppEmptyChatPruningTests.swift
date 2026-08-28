import Foundation
import Testing
@testable import TUFFAppCore

/// A chat is written to disk the moment New Chat is pressed, so one abandoned
/// before its first message used to stay in the sidebar forever as another
/// identical row called "New Chat". The sidebar is now always on screen, so
/// that litter buries the real chats.
@Suite(.serialized) @MainActor struct AppEmptyChatPruningTests {
    private func makeStore() -> (URL, AppConversationStore) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-prune-\(UUID().uuidString)", isDirectory: true)
        return (root, AppConversationStore(
            repository: AppConversationRepository(rootURL: root)))
    }

    @Test func repeatedNewChatLeavesOneEmptyConversation() {
        let (root, store) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }

        store.startNewConversation(modelID: "gemma4-e4b")
        store.startNewConversation(modelID: "gemma4-e4b")
        store.startNewConversation(modelID: "gemma4-e4b")

        #expect(store.conversations.count == 1)
        #expect(store.selectedConversation?.turns.isEmpty == true)
    }

    @Test func aChatWithATurnIsNeverDiscarded() {
        let (root, store) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }

        store.recordCompletedTurn(
            AppChatTurn(prompt: "keep me", response: "kept"),
            attachments: [], modelID: "gemma4-e4b")
        let kept = store.selectedConversationID
        store.startNewConversation(modelID: "gemma4-e4b")
        store.startNewConversation(modelID: "gemma4-e4b")

        #expect(store.conversations.count == 2)
        #expect(store.conversations.contains { $0.id == kept })
    }

    /// Opening a saved chat sweeps the empty one that was left behind, and the
    /// chat being opened stays selected.
    @Test func openingASavedChatDiscardsTheAbandonedEmptyOne() throws {
        let (root, store) = makeStore()
        defer { try? FileManager.default.removeItem(at: root) }

        store.recordCompletedTurn(
            AppChatTurn(prompt: "first", response: "answer"),
            attachments: [], modelID: "gemma4-e4b")
        let saved = try #require(store.selectedConversationID)
        store.startNewConversation(modelID: "gemma4-e4b")
        #expect(store.conversations.count == 2)

        store.selectConversation(id: saved)

        #expect(store.conversations.count == 1)
        #expect(store.selectedConversationID == saved)
        #expect(store.conversation.count == 1)
    }

    /// The pruning is written through to disk, not just to the in-memory list.
    @Test func theDiscardSurvivesAReload() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-prune-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let repository = AppConversationRepository(rootURL: root)

        let first = AppConversationStore(repository: repository)
        first.recordCompletedTurn(
            AppChatTurn(prompt: "first", response: "answer"),
            attachments: [], modelID: "gemma4-e4b")
        first.startNewConversation(modelID: "gemma4-e4b")
        let saved = try #require(first.conversations.first { !$0.turns.isEmpty }?.id)
        first.selectConversation(id: saved)

        let reloaded = AppConversationStore(repository: repository)
        #expect(reloaded.conversations.count == 1)
        #expect(reloaded.selectedConversationID == saved)
    }
}
