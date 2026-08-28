import Foundation
import Testing
@testable import TUFFAppCore

/// Images stay in the conversation.
///
/// A picture used to reach the model on exactly the message it was attached to.
/// Every follow-up — "what colour is the shirt?", "read the second line" — was
/// answered by a model that could no longer see it, with nothing on screen
/// saying so. These pin the rules that keep an image in context, and the budget
/// that stops a long chat re-encoding a dozen of them on every turn.
@Suite struct ConversationImageContextTests {
    private func makeImage(_ name: String) -> AppImageAttachment {
        AppImageAttachment(
            fileURL: URL(fileURLWithPath: "/tmp/\(name)"),
            displayName: name,
            encodedBytes: 1_024,
            sha256: String(repeating: "a", count: 64))
    }

    private func writeImage(
        _ root: URL, _ name: String
    ) throws -> AppImageAttachment {
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        let url = root.appendingPathComponent(name)
        try Data("fixture".utf8).write(to: url)
        return AppImageAttachment(
            fileURL: url,
            displayName: name,
            encodedBytes: 7,
            sha256: String(repeating: "b", count: 64))
    }

    /// The current message is encoded first: if the context leaves room for one
    /// image, it has to be the one that was just attached.
    @Test func theCurrentMessageIsEncodedAheadOfTheConversation() {
        let current = makeImage("now.png")
        let older = makeImage("older.png")
        let newer = makeImage("newer.png")
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.gturbo"),
            prompt: "and this one?",
            history: [
                AppChatTurn(prompt: "first", response: "a", images: [older]),
                AppChatTurn(prompt: "second", response: "b", images: [newer]),
            ],
            imageAttachments: [current])

        let ordered = RealInferenceSession.attachmentsToEncode(for: request)

        #expect(ordered.map(\.displayName) == ["now.png", "newer.png", "older.png"])
    }

    @Test func anImageIsEncodedOnceHoweverManyTurnsReferToIt() {
        let repeated = makeImage("same.png")
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.gturbo"),
            prompt: "again",
            history: [
                AppChatTurn(prompt: "first", response: "a", images: [repeated]),
                AppChatTurn(prompt: "second", response: "b", images: [repeated]),
            ],
            imageAttachments: [])

        #expect(RealInferenceSession.attachmentsToEncode(for: request).count == 1)
    }

    @Test func aTextOnlyConversationNeedsNoVisionRuntime() {
        let request = AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.gturbo"),
            prompt: "hello",
            history: [AppChatTurn(prompt: "first", response: "a")])

        #expect(RealInferenceSession.attachmentsToEncode(for: request).isEmpty)
    }

    /// Newest first, and nothing past the budget. Encoding is the expensive part
    /// of an image prompt and the context cannot hold more than the budget
    /// anyway, so carrying every image a long chat has collected would pay for
    /// encodes the renderer then throws away.
    @MainActor
    @Test func onlyTheNewestImagesThatFitTheBudgetAreCarried() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("carry-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = AppModel()
        let older = try writeImage(root, "older.png")
        let newer = try writeImage(root, "newer.png")
        model.conversationStore.conversation = [
            AppChatTurn(prompt: "first", response: "a", images: [older]),
            AppChatTurn(prompt: "second", response: "b", images: [newer]),
        ]

        let history = model.requestHistory(imageBudget: 1)

        #expect(history.count == 2)
        #expect(history[0].images.isEmpty, "the oldest images went past the budget")
        #expect(history[1].images.map(\.displayName) == ["newer.png"])
        #expect(history[0].prompt == "first", "the text of a trimmed turn still counts")
    }

    @MainActor
    @Test func noBudgetMeansTheConversationIsCarriedAsTextAlone() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("carry-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let model = AppModel()
        model.conversationStore.conversation = [
            AppChatTurn(
                prompt: "look", response: "a",
                images: [try writeImage(root, "only.png")]),
        ]

        #expect(model.requestHistory(imageBudget: 0)[0].images.isEmpty)
    }

    /// The archive can lose a file — a backup restore, a manual delete. Sending
    /// its path fails the whole run rather than one image, so it is dropped.
    @MainActor
    @Test func anImageTheArchiveNoLongerHoldsIsDroppedNotSent() {
        let model = AppModel()
        model.conversationStore.conversation = [
            AppChatTurn(
                prompt: "look", response: "a",
                images: [makeImage("gone-\(UUID().uuidString).png")]),
        ]

        #expect(model.requestHistory(imageBudget: 4)[0].images.isEmpty)
    }
}
