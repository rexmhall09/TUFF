import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite(.serialized) struct AppConversationPersistenceTests {
    private func makeRepository() -> (URL, AppConversationRepository) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-chats-\(UUID().uuidString)", isDirectory: true)
        return (root, AppConversationRepository(rootURL: root))
    }

    @Test func archiveRoundTripsThroughAtomicFile() throws {
        let (root, repository) = makeRepository()
        defer { try? FileManager.default.removeItem(at: root) }
        let id = UUID()
        let archive = AppConversationArchive(
            selectedConversationID: id,
            conversations: [AppConversationRecord(
                id: id,
                title: "A saved chat",
                modelID: "gemma4-e4b",
                createdAt: Date(timeIntervalSince1970: 1_700_000_000),
                updatedAt: Date(timeIntervalSince1970: 1_700_000_001),
                turns: [AppPersistedChatTurn(prompt: "hello", response: "hi")])])

        try repository.save(archive)

        #expect(try repository.load() == archive)
        let siblings = try FileManager.default.contentsOfDirectory(
            at: root, includingPropertiesForKeys: nil)
        #expect(siblings.map(\.lastPathComponent) == ["conversations.json"])
    }

    @MainActor
    @Test func newerArchiveIsNeverOverwritten() throws {
        let (root, repository) = makeRepository()
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let original = Data("{\"schemaVersion\":99,\"sentinel\":true}".utf8)
        try original.write(to: repository.archiveURL)

        let store = AppConversationStore(repository: repository)
        store.startNewConversation(modelID: "gemma4-e4b")

        #expect(store.persistenceError != nil)
        #expect(try Data(contentsOf: repository.archiveURL) == original)
    }

    @MainActor
    @Test func completedTurnsRestoreWithNameAndModelBinding() {
        let (root, repository) = makeRepository()
        defer { try? FileManager.default.removeItem(at: root) }
        let first = AppConversationStore(repository: repository)
        first.recordCompletedTurn(
            AppChatTurn(
                prompt: "Explain why local models are useful",
                response: "Privacy.",
                thinking: "The user asked for one benefit."),
            attachments: [],
            modelID: "gpt-oss-20b",
            now: Date(timeIntervalSince1970: 1_700_000_000))

        let restored = AppConversationStore(repository: repository)

        #expect(restored.conversations.count == 1)
        #expect(restored.selectedConversation?.title
            == "Explain why local models are useful")
        #expect(restored.selectedConversation?.modelID == "gpt-oss-20b")
        #expect(restored.conversation == [
            AppChatTurn(
                id: restored.conversation[0].id,
                prompt: "Explain why local models are useful",
                response: "Privacy.",
                thinking: "The user asked for one benefit.")
        ])

        let chat = restored.selectedConversation!
        restored.renameConversation(id: chat.id, title: "  local   privacy  ")
        #expect(restored.selectedConversation?.title == "local privacy")
    }

    @MainActor
    @Test func managedAttachmentsSurviveRestartAndLeaveWithChat() throws {
        let (root, repository) = makeRepository()
        defer { try? FileManager.default.removeItem(at: root) }
        let staging = AppImageAttachmentStore(directoryURL: root
            .appendingPathComponent("staging", isDirectory: true))
        let attachment = try staging.stage(
            data: Data("image fixture".utf8),
            displayName: "photo.png")
        let store = AppConversationStore(repository: repository)
        let turn = AppChatTurn(prompt: "What is this?", response: "A fixture.")

        store.recordCompletedTurn(
            turn,
            attachments: [attachment],
            modelID: "gemma4-26b-a4b")
        staging.removeAll()

        let restored = AppConversationStore(repository: repository)
        let durable = try #require(restored.attachments(for: turn.id).first)
        #expect(try Data(contentsOf: durable.fileURL) == Data("image fixture".utf8))
        #expect(durable.fileURL.path.hasPrefix(repository.attachmentsRootURL.path))

        let chat = try #require(restored.selectedConversation)
        restored.deleteConversation(id: chat.id)
        #expect(!FileManager.default.fileExists(atPath: durable.fileURL.path))
        #expect(try repository.load().conversations.isEmpty)
    }

    @MainActor
    @Test func changingModelAfterMessagesStartsFreshChat() {
        let store = AppConversationStore()
        store.recordCompletedTurn(
            AppChatTurn(prompt: "hello", response: "hi"),
            attachments: [],
            modelID: AppModelInstallDescriptor.default.settingsProfileKey)
        let qwenInstaller = MockModelInstallerClient(descriptor: .qwen36)
        let qwen = ModelInstallCoordinator(
            descriptor: .qwen36,
            directoryURL: FileManager.default.temporaryDirectory
                .appendingPathComponent("qwen-switch-\(UUID().uuidString).gturbo"),
            client: qwenInstaller)
        let model = AppModel(
            modelDirectory: FileManager.default.temporaryDirectory
                .appendingPathComponent("gemma-switch-\(UUID().uuidString).gturbo"),
            installer: MockModelInstallerClient(descriptor: .default),
            otherInstalls: [qwen],
            conversationStore: store)

        model.selectModel(qwen)

        #expect(model.selectedModelID == qwen.id)
        #expect(model.conversation.isEmpty)
        #expect(store.conversations.count == 2)
        #expect(store.selectedConversation?.modelID
            == AppModelInstallDescriptor.qwen36.settingsProfileKey)

        let qwenChat = store.selectedConversation!
        model.deleteConversation(qwenChat)
        #expect(store.selectedConversation?.modelID
            == AppModelInstallDescriptor.default.settingsProfileKey)
        #expect(model.selectedModelID == AppModelInstallDescriptor.default.id)
    }

    @Test func attachmentPathsCannotEscapeManagedStorage() {
        let (root, repository) = makeRepository()
        defer { try? FileManager.default.removeItem(at: root) }
        let escaped = AppConversationAttachment(
            id: UUID(),
            relativePath: "attachments/../outside",
            displayName: "outside.png",
            encodedBytes: 1,
            sha256: String(repeating: "0", count: 64))

        #expect(throws: AppConversationPersistenceError.self) {
            try repository.resolve(escaped)
        }
    }

    @Test func decodeServiceAcceptsManagedApplicationSupportAttachments() {
        let managed = AppConversationRepository.defaultRootURL
            .appendingPathComponent("attachments/chat/image")

        #expect(AppImageAttachmentStore.contains(managed))
        #expect(!AppImageAttachmentStore.contains(
            FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("unmanaged-image.png")))
    }
}
