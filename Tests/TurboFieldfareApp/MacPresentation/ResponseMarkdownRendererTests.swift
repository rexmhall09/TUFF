import AppKit
import Foundation
import Testing
@testable import TurboFieldfareMacPresentation

@MainActor
@Suite struct ResponseMarkdownRendererTests {
    @Test func rendersSupportedMarkdownWithNativeAttributes() throws {
        let source = """
        # Heading

        A **bold** and *italic* sentence with ~~obsolete~~ text, `inlineCode`, and a [link](https://example.com).

        - first
        - second

        > quoted text

        ```swift
        let answer = 42
        ```

        ---
        """

        let result = ResponseMarkdownRenderer().render(source)
        let text = result.attributedString.string

        #expect(!result.usedFallback)
        #expect(text.contains("Heading"))
        #expect(text.contains("bold"))
        #expect(text.contains("italic"))
        #expect(text.contains("•\tfirst\n•\tsecond"))
        #expect(text.contains("│\tquoted text"))
        #expect(text.contains("let answer = 42"))
        #expect(text.contains("────────────────"))
        #expect(!text.contains("**"))
        #expect(!text.contains("```"))

        let linkRange = (text as NSString).range(of: "link")
        #expect(result.attributedString.attribute(.link, at: linkRange.location,
                                                  effectiveRange: nil) == nil)
        let linkColor = result.attributedString.attribute(
            .foregroundColor, at: linkRange.location, effectiveRange: nil) as? NSColor
        #expect(linkColor?.isEqual(NSColor.linkColor) == true)
        #expect(result.attributedString.attribute(.underlineStyle,
                                                  at: linkRange.location,
                                                  effectiveRange: nil) as? Int
            == NSUnderlineStyle.single.rawValue)

        let codeRange = (text as NSString).range(of: "inlineCode")
        let codeFont = try #require(result.attributedString.attribute(
            .font, at: codeRange.location, effectiveRange: nil) as? NSFont)
        #expect(codeFont.fontDescriptor.symbolicTraits.contains(.monoSpace))
        #expect(result.attributedString.attribute(
            .backgroundColor, at: codeRange.location, effectiveRange: nil) != nil)

        let strikeRange = (text as NSString).range(of: "obsolete")
        #expect(result.attributedString.attribute(
            .strikethroughStyle, at: strikeRange.location, effectiveRange: nil) != nil)
    }

    @Test func unfinishedFenceFallsBackToExactRawText() {
        let source = "Before\n\n```python\nprint('unfinished')"
        let result = ResponseMarkdownRenderer().render(source)

        #expect(result.usedFallback)
        #expect(result.attributedString.string == source)
    }

    @Test func unsupportedHTMLTableAndImageStayReadableAsRawText() {
        let renderer = ResponseMarkdownRenderer()
        let samples = [
            "<div>Never execute this</div>",
            "| A | B |\n|---|---|\n| 1 | 2 |",
            "![remote](https://example.com/image.png)",
        ]

        for source in samples {
            let result = renderer.render(source)
            #expect(result.usedFallback)
            #expect(result.attributedString.string == source)
        }
    }

    @Test func latexRendersAsNativeMathAlongsideMarkdown() throws {
        let source = "Cosine is **$\\frac{u \\cdot v}{\\lVert u\\rVert \\lVert v\\rVert}$**."
        let result = ResponseMarkdownRenderer().render(source)

        #expect(!result.usedFallback)
        let attachmentRange = (result.attributedString.string as NSString)
            .range(of: "\u{fffc}")
        try #require(attachmentRange.location != NSNotFound)
        let attachment = try #require(result.attributedString.attribute(
            .attachment,
            at: attachmentRange.location,
            effectiveRange: nil) as? NSTextAttachment)
        #expect(attachment.image?.size.width ?? 0 > 0)
        #expect(attachment.image?.size.height ?? 0 > 0)
        #expect(attachment.image?.accessibilityDescription?.contains("\\frac") == true)
        #expect(result.attributedString.attribute(
            .tuffLaTeXSource,
            at: attachmentRange.location,
            effectiveRange: nil) as? String == "$\\frac{u \\cdot v}{\\lVert u\\rVert \\lVert v\\rVert}$")
    }

    @Test func supportsDisplayAndBracketedLaTeXDelimiters() {
        let source = """
        Inline \\(x^2 + y^2 = z^2\\).

        $$\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}$$

        \\[\\int_0^1 x^2 \\, dx = \\frac{1}{3}\\]
        """
        let result = ResponseMarkdownRenderer().render(source)
        let attachments = result.attributedString.string.filter { $0 == "\u{fffc}" }

        #expect(!result.usedFallback)
        #expect(attachments.count == 3)
    }

    @Test func codeAndEscapedCurrencyNeverBecomeMath() {
        let source = """
        `let price = "$5"`

        ```text
        $x^2$
        ```

        A ticket costs \\$20.
        """
        let result = ResponseMarkdownRenderer().render(source)

        #expect(!result.attributedString.string.contains("\u{fffc}"))
        #expect(result.attributedString.string.contains("$x^2$"))
        #expect(result.attributedString.string.contains("$20"))
    }

    @Test func malformedLaTeXFallsBackToItsExactSource() {
        let source = "Before $\\frac{1}{2$ after"
        let result = ResponseMarkdownRenderer().render(source)

        #expect(result.usedFallback)
        #expect(result.attributedString.string == source)
    }

    @Test func boldOnlyModelHeadingStaysOnItsOwnLine() {
        let source = "**Origins**\nFieldfares arrive from northern Europe."
        let result = ResponseMarkdownRenderer().render(source)

        #expect(!result.usedFallback)
        #expect(result.attributedString.string
            == "Origins\n\nFieldfares arrive from northern Europe.")
    }
}

@MainActor
@Suite struct InstructionTranscriptDocumentControllerTests {
    @Test func appAccentMatchesProductRGB() {
        let color = TurboFieldfareMacTheme.accentNSColor
            .usingColorSpace(.sRGB)
        #expect(color != nil)
        #expect(abs((color?.redComponent ?? 0) - 111.0 / 255.0) < 0.000_001)
        #expect(abs((color?.greenComponent ?? 0) - 77.0 / 255.0) < 0.000_001)
        #expect(abs((color?.blueComponent ?? 0) - 255.0 / 255.0) < 0.000_001)
    }

    @Test func appAccentKeepsWhiteControlLabelsReadable() {
        let components = [111.0, 77.0, 255.0].map { value in
            let channel = value / 255.0
            return channel <= 0.03928
                ? channel / 12.92
                : pow((channel + 0.055) / 1.055, 2.4)
        }
        let luminance = 0.2126 * components[0]
            + 0.7152 * components[1]
            + 0.0722 * components[2]
        let contrast = 1.05 / (luminance + 0.05)

        #expect(contrast >= 4.5)
    }

    @Test func rebuildsThenAppendsOnlyNewResponseSuffix() {
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController()

        let first = controller.synchronize(
            storage: storage,
            prompt: "Explain this",
            response: "Hel",
            isTerminal: false)
        let second = controller.synchronize(
            storage: storage,
            prompt: "Explain this",
            response: "Hello",
            isTerminal: false)

        #expect(first.mutation == .rebuilt)
        #expect(second.mutation == .appended)
        #expect(storage.string == "You\nExplain this\n\nAnswer\nHello")
        #expect(storage.string.components(separatedBy: "Answer").count == 2)
        let answerRange = (storage.string as NSString).range(of: "Answer")
        let answerColor = storage.attribute(
            .foregroundColor,
            at: answerRange.location,
            effectiveRange: nil) as? NSColor
        #expect(answerColor?.isEqual(TurboFieldfareMacTheme.accentNSColor) == true)
        #expect(controller.response == "Hello")
    }

    @Test func animatedPrefillPlaceholderIsPresentationOnlyAndFirstResponseRemovesIt() {
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController()

        let prefilling = controller.synchronize(
            storage: storage,
            prompt: "Explain this",
            response: "",
            isTerminal: false,
            showsPrefillPlaceholder: true)

        #expect(prefilling.mutation == .rebuilt)
        #expect(controller.showsPrefillPlaceholder)
        #expect(storage.string == "You\nExplain this\n\nAnswer\nProcessing your prompt")
        #expect(controller.response.isEmpty)
        #expect(controller.assistantRange.length == 0)

        #expect(controller.advancePrefillAnimation(storage: storage))
        #expect(storage.string.hasSuffix("Processing your prompt."))
        #expect(controller.advancePrefillAnimation(storage: storage))
        #expect(storage.string.hasSuffix("Processing your prompt.."))
        #expect(controller.advancePrefillAnimation(storage: storage))
        #expect(storage.string.hasSuffix("Processing your prompt..."))
        #expect(controller.advancePrefillAnimation(storage: storage))
        #expect(storage.string.hasSuffix("Processing your prompt"))

        let responding = controller.synchronize(
            storage: storage,
            prompt: "Explain this",
            response: "Hello",
            isTerminal: false,
            showsPrefillPlaceholder: true)

        #expect(responding.mutation == .rebuilt)
        #expect(!controller.showsPrefillPlaceholder)
        #expect(storage.string == "You\nExplain this\n\nAnswer\nHello")
        #expect(!storage.string.contains("Processing your prompt"))
        #expect((storage.string as NSString).substring(with: responding.assistantRange)
            == "Hello")
    }

    @Test func processingAnimationPolicyStopsForTextAndTerminalStates() {
        #expect(InstructionTranscriptDocumentController.shouldRunPrefillAnimation(
            response: "", isTerminal: false, requested: true))
        #expect(!InstructionTranscriptDocumentController.shouldRunPrefillAnimation(
            response: "First token", isTerminal: false, requested: true))
        #expect(!InstructionTranscriptDocumentController.shouldRunPrefillAnimation(
            response: "", isTerminal: true, requested: true))
        #expect(!InstructionTranscriptDocumentController.shouldRunPrefillAnimation(
            response: "", isTerminal: false, requested: false))
    }

    @Test func promptChangeOrResponseResetRebuildsWithoutStaleBytes() {
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController()
        _ = controller.synchronize(
            storage: storage, prompt: "Old", response: "Long response", isTerminal: false)

        let result = controller.synchronize(
            storage: storage, prompt: "New", response: "Short", isTerminal: false)

        #expect(result.mutation == .rebuilt)
        #expect(storage.string == "You\nNew\n\nAnswer\nShort")
        #expect(!storage.string.contains("Old"))
        #expect(!storage.string.contains("Long response"))
    }

    @Test func terminalUpdateFormatsOnlyAssistantRangeAndKeepsRawResponse() {
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController()
        _ = controller.synchronize(
            storage: storage,
            prompt: "Question",
            response: "**Bold answer**",
            isTerminal: false)

        let result = controller.synchronize(
            storage: storage,
            prompt: "Question",
            response: "**Bold answer**",
            isTerminal: true)

        #expect(result.mutation == .finalized)
        #expect(controller.isFinalized)
        #expect(controller.response == "**Bold answer**")
        #expect(storage.string == "You\nQuestion\n\nAnswer\nBold answer")
        #expect((storage.string as NSString).substring(with: result.assistantRange)
            == "Bold answer")

        let unchanged = storage.copy() as! NSAttributedString
        let repeated = controller.synchronize(
            storage: storage,
            prompt: "Question",
            response: "**Bold answer**",
            isTerminal: true)
        #expect(repeated.mutation == .none)
        #expect(storage.isEqual(to: unchanged))
    }

    @Test func terminalPartialOutputIsReadableAndNextRunRestoresStreamingSource() {
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController()
        _ = controller.synchronize(
            storage: storage,
            prompt: "Question",
            response: "Partial **answer**",
            isTerminal: true)
        #expect(storage.string.hasSuffix("Partial answer"))

        let result = controller.synchronize(
            storage: storage,
            prompt: "Question",
            response: "Partial **answer**",
            isTerminal: false)
        #expect(result.mutation == .rebuilt)
        #expect(!controller.isFinalized)
        #expect(storage.string.hasSuffix("Partial **answer**"))
    }

    @Test func terminalResponseRendersAgainWhenClosingFenceArrivesLate() {
        let storage = NSMutableAttributedString()
        let controller = InstructionTranscriptDocumentController()
        let partial = "```cpp\nkernel void matmul() {}"
        let complete = partial + "\n```"

        let first = controller.synchronize(
            storage: storage,
            prompt: "Write a Metal kernel",
            response: partial,
            isTerminal: true)
        #expect(first.mutation == .finalized)
        #expect(storage.string.hasSuffix(partial))

        let updated = controller.synchronize(
            storage: storage,
            prompt: "Write a Metal kernel",
            response: complete,
            isTerminal: true)

        #expect(updated.mutation == .finalized)
        #expect(controller.response == complete)
        #expect((storage.string as NSString).substring(with: updated.assistantRange)
            == "kernel void matmul() {}\n")
        #expect(!storage.string.contains("```"))
    }

    @Test func selectionRangesClampToCurrentStorage() {
        let ranges = InstructionTranscriptDocumentController.clampedRanges([
            NSRange(location: 3, length: 20),
            NSRange(location: 50, length: 2),
        ], toLength: 10)

        #expect(ranges == [
            NSRange(location: 3, length: 7),
            NSRange(location: 10, length: 0),
        ])
    }

    @Test func terminalFormattingAlwaysScrollsToBottom() {
        #expect(InstructionTranscriptDocumentController.shouldScrollToBottom(
            wasAtBottom: false,
            mutation: .finalized))
        #expect(InstructionTranscriptDocumentController.shouldScrollToBottom(
            wasAtBottom: true,
            mutation: .appended))
        #expect(!InstructionTranscriptDocumentController.shouldScrollToBottom(
            wasAtBottom: false,
            mutation: .appended))
    }
}
