import Foundation

/// What each of the two copy buttons on an answer puts on the pasteboard.
///
/// The choice lives here rather than inside the transcript view so it can be
/// tested. The whole point of the pair is that they produce different text for
/// the same answer, and a view cannot assert that about itself.
public enum AnswerCopyFormat {
    /// The answer as it reads on screen, with the Markdown resolved. What
    /// someone pasting into a mail or a note wants.
    case text
    /// The Markdown the model actually produced, untouched, for pasting
    /// somewhere that renders it again.
    case markdown

    @MainActor
    public func pasteboardText(
        for response: String,
        renderer: ResponseMarkdownRenderer
    ) -> String {
        switch self {
        case .text: renderer.plainText(response)
        case .markdown: response
        }
    }
}
