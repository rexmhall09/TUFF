import Foundation

/// One completed exchange: what the user asked, and what the model answered.
///
/// Only finished turns become history. A cancelled or failed generation leaves
/// no turn behind, so the model never sees a truncated answer of its own as
/// established context.
///
/// `prompt` is what the user typed. The attachments that came with it are held
/// separately rather than folded into that string, so the transcript can show a
/// file as a file and an image as an image; `modelPrompt` is the one place they
/// are flattened back into the text a model actually reads.
public struct AppChatTurn: Equatable, Sendable, Codable, Identifiable {
    public let id: UUID
    public var prompt: String
    public var response: String
    public var thinking: String?
    /// Files attached to this message. Their text is prepended to the prompt on
    /// every turn that carries them, so a follow-up question can still refer to
    /// the document.
    public var documents: [AppDocumentAttachment]
    /// Images attached to this message. Carried into the context of later turns
    /// so "what colour is it?" has something to look at.
    public var images: [AppImageAttachment]
    /// `settingsProfileKey` of the model that produced `response`. Nil for turns
    /// saved before chats recorded it, and for a message not yet answered.
    public var modelID: String?

    public init(
        id: UUID = UUID(),
        prompt: String,
        response: String,
        thinking: String? = nil,
        documents: [AppDocumentAttachment] = [],
        images: [AppImageAttachment] = [],
        modelID: String? = nil
    ) {
        self.id = id
        self.prompt = prompt
        self.response = response
        self.thinking = thinking
        self.documents = documents
        self.images = images
        self.modelID = modelID
    }

    /// The user message as the model sees it: attached documents first, then
    /// what was typed.
    public var modelPrompt: String {
        let preamble = documents.promptPreamble
        if preamble.isEmpty { return prompt }
        if prompt.isEmpty { return preamble }
        return preamble + "\n\n" + prompt
    }

    /// This turn as it is sent back through the inference client: the prompt
    /// flattened, the documents dropped because they are already in it, and
    /// exactly the images the caller decided still fit the image budget.
    ///
    /// The images are passed in rather than defaulted, because "carry none" is
    /// a decision the conversation makes per turn and a default would make it
    /// silently.
    public func requestTurn(carrying images: [AppImageAttachment]) -> AppChatTurn {
        AppChatTurn(
            id: id,
            prompt: modelPrompt,
            response: response,
            thinking: thinking,
            images: images,
            modelID: modelID)
    }
}

/// How much of a conversation survived the context budget.
public struct AppConversationTrim: Equatable, Sendable {
    /// Turns dropped from the start of the conversation to make the prompt fit.
    public let droppedTurns: Int
    /// Token count of the render that was actually sent.
    public let promptTokens: Int

    public init(droppedTurns: Int, promptTokens: Int) {
        self.droppedTurns = droppedTurns
        self.promptTokens = promptTokens
    }

    public var didTrim: Bool { droppedTurns > 0 }
}
