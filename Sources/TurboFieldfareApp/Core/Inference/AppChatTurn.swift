import Foundation

/// One completed exchange: what the user asked, and what the model answered.
///
/// Only finished turns become history. A cancelled or failed generation leaves
/// no turn behind, so the model never sees a truncated answer of its own as
/// established context.
public struct AppChatTurn: Equatable, Sendable, Codable, Identifiable {
    public let id: UUID
    public var prompt: String
    public var response: String

    public init(id: UUID = UUID(), prompt: String, response: String) {
        self.id = id
        self.prompt = prompt
        self.response = response
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
