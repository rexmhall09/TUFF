import Foundation

/// One completed exchange, as the transcript document renders it.
///
/// Deliberately its own type rather than the app's `AppChatTurn`: the
/// presentation layer renders text and knows nothing about conversation state
/// or identity.
public struct TranscriptTurn: Equatable, Sendable {
    public let prompt: String
    public let response: String

    public init(prompt: String, response: String) {
        self.prompt = prompt
        self.response = response
    }
}
