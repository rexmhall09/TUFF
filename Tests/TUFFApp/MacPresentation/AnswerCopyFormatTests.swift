import Foundation
import Testing
@testable import TUFFMacPresentation

@MainActor
@Suite struct AnswerCopyFormatTests {
    private static let answer = """
    # Heading

    A **bold** sentence with `inlineCode` and a [link](https://example.com).

    - first
    - second

    ```swift
    let answer = 42
    ```
    """

    @Test func markdownCopiesTheSourceUntouched() {
        let copy = AnswerCopyFormat.markdown.pasteboardText(
            for: Self.answer, renderer: ResponseMarkdownRenderer())

        #expect(copy == Self.answer)
    }

    @Test func textCopiesTheAnswerAsItReadsOnScreen() {
        let copy = AnswerCopyFormat.text.pasteboardText(
            for: Self.answer, renderer: ResponseMarkdownRenderer())

        // The syntax is resolved away, but every word survives it.
        #expect(!copy.contains("**"))
        #expect(!copy.contains("```"))
        #expect(!copy.contains("# Heading"))
        #expect(!copy.contains("(https://example.com)"))
        #expect(copy.contains("Heading"))
        #expect(copy.contains("bold"))
        #expect(copy.contains("inlineCode"))
        #expect(copy.contains("link"))
        #expect(copy.contains("let answer = 42"))
    }

    /// The reason the second button exists: the same answer has to reach the
    /// pasteboard as two different strings.
    @Test func theTwoFormatsDisagreeOnAMarkdownAnswer() {
        let renderer = ResponseMarkdownRenderer()

        #expect(AnswerCopyFormat.text.pasteboardText(for: Self.answer, renderer: renderer)
            != AnswerCopyFormat.markdown.pasteboardText(for: Self.answer, renderer: renderer))
    }

    /// An answer with no Markdown in it is the one case where they agree, and
    /// neither may mangle it.
    @Test func plainProseIsUnchangedByEitherFormat() {
        let renderer = ResponseMarkdownRenderer()
        let prose = "A sentence with no Markdown in it."

        #expect(AnswerCopyFormat.text.pasteboardText(for: prose, renderer: renderer) == prose)
        #expect(AnswerCopyFormat.markdown.pasteboardText(for: prose, renderer: renderer) == prose)
    }

    @Test func anEmptyAnswerCopiesAsNothing() {
        let renderer = ResponseMarkdownRenderer()

        #expect(AnswerCopyFormat.text.pasteboardText(for: "", renderer: renderer).isEmpty)
        #expect(AnswerCopyFormat.markdown.pasteboardText(for: "", renderer: renderer).isEmpty)
    }
}
