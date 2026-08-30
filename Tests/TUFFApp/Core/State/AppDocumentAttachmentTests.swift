import Foundation
import Testing
@testable import TUFFAppCore

/// Documents are attachments, not prompt text.
///
/// They used to be pasted into the prompt box: a two-line question with a PDF on
/// it became a ten-thousand-word prompt with no way to take the file back off,
/// and the transcript then showed that wall of text as what the user had asked.
/// The text still reaches the model — every model reads text — but only at the
/// point where the request is built.
@Suite struct AppDocumentAttachmentTests {
    private func writeDocument(
        _ name: String, _ contents: String
    ) throws -> (root: URL, url: URL) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("documents-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        let url = root.appendingPathComponent(name)
        try Data(contents.utf8).write(to: url)
        return (root, url)
    }

    @MainActor
    @Test func attachingAFileLeavesThePromptBoxAlone() throws {
        let (root, url) = try writeDocument("notes.md", "# Notes\n\nSomething useful.")
        defer { try? FileManager.default.removeItem(at: root) }
        let model = makeAppModel()
        model.promptText = "Summarise this"

        model.attachDocuments([url])

        #expect(model.promptText == "Summarise this",
                "the file's text was pasted into the message again")
        #expect(model.documentAttachments.count == 1)
        #expect(model.documentAttachments[0].displayName == "notes.md")
        #expect(model.documentAttachmentNotice != nil,
                "the size has to be visible before the message is sent")
    }

    @MainActor
    @Test func theModelSeesTheFileAheadOfTheTypedMessage() throws {
        let (root, url) = try writeDocument("data.csv", "a,b\n1,2")
        defer { try? FileManager.default.removeItem(at: root) }
        let model = makeAppModel()
        model.promptText = "What is in column b?"
        model.attachDocuments([url])

        let composed = model.composedPromptText
        #expect(composed.hasPrefix("<document name=\"data.csv\">"))
        #expect(composed.contains("a,b\n1,2"))
        #expect(composed.hasSuffix("What is in column b?"))
    }

    @MainActor
    @Test func aFileOnItsOwnIsEnoughToSend() throws {
        let (root, url) = try writeDocument("report.txt", "Quarterly results.")
        defer { try? FileManager.default.removeItem(at: root) }
        let model = makeAppModel()

        #expect(!model.hasComposedInput)
        model.attachDocuments([url])
        #expect(model.hasComposedInput,
                "a file with no question typed is still a message")
        #expect(model.composedPromptText.contains("Quarterly results."))
    }

    @MainActor
    @Test func aFileCanBeTakenBackOff() throws {
        let (root, first) = try writeDocument("one.txt", "first")
        defer { try? FileManager.default.removeItem(at: root) }
        let second = root.appendingPathComponent("two.txt")
        try Data("second".utf8).write(to: second)
        let model = makeAppModel()
        model.attachDocuments([first, second])
        #expect(model.documentAttachments.count == 2)

        model.removeDocument(id: model.documentAttachments[0].id)

        #expect(model.documentAttachments.map(\.displayName) == ["two.txt"])
        #expect(!model.composedPromptText.contains("first"))

        model.clearDocuments()
        #expect(model.documentAttachments.isEmpty)
        #expect(model.documentAttachmentNotice == nil)
        #expect(model.composedPromptText.isEmpty)
    }

    @MainActor
    @Test func anUnreadableFileIsRefusedRatherThanAttachedEmpty() throws {
        let (root, url) = try writeDocument("notes.rtf", "not supported")
        defer { try? FileManager.default.removeItem(at: root) }
        let model = makeAppModel()

        model.attachDocuments([url])

        #expect(model.documentAttachments.isEmpty)
        #expect(model.error != nil)
    }

    @MainActor
    @Test func theFileCountIsCapped() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("documents-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        var urls: [URL] = []
        for index in 0..<(AppModel.maximumDocumentAttachments + 3) {
            let url = root.appendingPathComponent("file-\(index).txt")
            try Data("contents \(index)".utf8).write(to: url)
            urls.append(url)
        }
        let model = makeAppModel()

        model.attachDocuments(urls)

        #expect(model.documentAttachments.count == AppModel.maximumDocumentAttachments)
        #expect(model.documentAttachmentNotice?.contains("At most") == true,
                "silently dropping the rest is how attachments go missing")
    }

    /// The flattening happens once, where the request is built, and history has
    /// to be flattened the same way or a follow-up question loses the file the
    /// first answer was about.
    @Test func aRecordedTurnStillCarriesItsFileIntoLaterTurns() {
        let turn = AppChatTurn(
            prompt: "What does it say?",
            response: "It is a receipt.",
            documents: [AppDocumentAttachment(
                displayName: "receipt.txt", text: "Total: $12")])

        #expect(turn.modelPrompt
            == "<document name=\"receipt.txt\">\nTotal: $12\n</document>\n\n"
                + "What does it say?")
        #expect(turn.requestTurn(carrying: []).prompt == turn.modelPrompt)
        #expect(turn.requestTurn(carrying: []).documents.isEmpty,
                "the documents are already in the prompt; sending both duplicates them")
    }

    @Test func aTokenEstimateIsPrintedAsAnEstimate() {
        #expect(AppDocumentAttachment.tokensLabel(940) == "~940")
        #expect(AppDocumentAttachment.tokensLabel(1_240) == "~1.2k")
        #expect(AppDocumentAttachment(displayName: "x.txt", text: String(repeating: "a", count: 4_000))
            .estimatedTokensLabel == "~1.0k")
    }
}
