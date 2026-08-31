import AppKit
import TUFFAppCore
import TUFFMacPresentation
import SwiftUI

/// One exchange: what was asked, with the attachments it was asked with, and
/// what came back — from the model named on the answer.
///
/// The same view draws a finished exchange and the one being generated. Only
/// the answer differs: a finished one is laid out once from rendered Markdown,
/// while the live one streams through `StreamingResponseView`. Everything
/// around it — the images, the files, the reasoning — belongs to the message,
/// which is why it is drawn here rather than above the conversation.
struct TranscriptMessageView: View {
    let prompt: String
    let response: String
    let thinking: String
    let images: [AppImageAttachment]
    let documents: [AppDocumentAttachment]
    let modelName: String
    let renderer: ResponseMarkdownRenderer
    /// Nil for a finished message. Present while this message is the live one.
    var live: LiveResponse?
    /// What this message can be rewound to. Absent while a run is in flight,
    /// and on the live message, which has nothing to rewind yet.
    var actions: MessageActions?

    struct MessageActions {
        let edit: () -> Void
        let regenerate: () -> Void
        /// Present only on the newest message, and only when the answer looks
        /// cut off rather than finished.
        let carryOn: (() -> Void)?
    }

    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @State private var thinkingExpanded = false
    @State private var promptHeight: CGFloat = 0
    @State private var responseHeight: CGFloat = 0
    @State private var liveHeight: CGFloat = 0
    /// Which of the two copy buttons last fired, so only that one shows the
    /// checkmark. A single flag lit both.
    @State private var copied: AnswerCopyFormat?

    struct LiveResponse {
        let mailbox: GenerationTranscriptMailbox?
        let isTerminal: Bool
        let showsPrefillPlaceholder: Bool
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            if hasUserMessage { userMessage }
            if hasAssistantMessage { assistantMessage }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var hasUserMessage: Bool {
        !prompt.isEmpty || !images.isEmpty || !documents.isEmpty
    }

    private var hasAssistantMessage: Bool {
        live != nil || !response.isEmpty || !thinking.isEmpty
    }

    private var userMessage: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                MessageRoleLabel(text: "You")
                if let actions {
                    MessageActionButton(
                        symbol: "pencil",
                        help: "Edit this message and send it again",
                        action: actions.edit)
                }
                Spacer(minLength: 0)
            }
            MessageAttachmentsView(images: images, documents: documents)
            if !prompt.isEmpty {
                // Rendered the same way an answer is: a prompt with a list, a
                // code block, or a formula in it was the one half of the
                // conversation still shown as raw source.
                AttributedTextView(
                    attributed: renderer.render(prompt).attributedString,
                    height: $promptHeight)
                    .frame(height: max(promptHeight, 1))
            }
        }
    }

    private var assistantMessage: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 6) {
                MessageRoleLabel(text: modelName)
                if showsActions { answerActions }
                Spacer(minLength: 0)
            }
            if !thinking.isEmpty {
                ThinkingDisclosure(
                    text: thinking,
                    isRunning: live?.isTerminal == false && response.isEmpty,
                    isExpanded: $thinkingExpanded,
                    reduceTransparency: reduceTransparency)
            }
            responseBody
        }
    }

    /// Copy, copy as Markdown, regenerate and continue, beside the name of
    /// whoever produced the answer. They used to sit under it and appear only
    /// on hover, which meant the reader had to know they were there before
    /// they would show.
    private var answerActions: some View {
        HStack(spacing: 6) {
            MessageActionButton(
                symbol: copied == .text ? "checkmark" : "doc.on.doc",
                help: copied == .text ? "Copied" : "Copy this answer",
                tint: copied == .text ? TUFFMacTheme.accentColor : nil,
                action: { copyResponse(.text) })
            MessageActionButton(
                symbol: copied == .markdown
                    ? "checkmark" : "chevron.left.forwardslash.chevron.right",
                help: copied == .markdown
                    ? "Copied" : "Copy this answer as Markdown",
                tint: copied == .markdown ? TUFFMacTheme.accentColor : nil,
                action: { copyResponse(.markdown) })
            if let actions {
                MessageActionButton(
                    symbol: "arrow.clockwise",
                    help: "Send this message again for a different answer",
                    action: actions.regenerate)
                if let carryOn = actions.carryOn {
                    MessageActionButton(
                        symbol: "text.append",
                        help: "Carry on from where this answer stopped",
                        tint: TUFFMacTheme.accentColor,
                        action: carryOn)
                }
            }
        }
        .task(id: copied) {
            guard copied != nil else { return }
            try? await Task.sleep(for: .seconds(1.5))
            guard !Task.isCancelled else { return }
            copied = nil
        }
    }

    private var showsActions: Bool {
        !response.isEmpty && live?.isTerminal != false
    }

    @ViewBuilder
    private var responseBody: some View {
        if let live {
            StreamingResponseView(
                text: response,
                mailbox: live.mailbox,
                isTerminal: live.isTerminal,
                showsPrefillPlaceholder: live.showsPrefillPlaceholder,
                height: $liveHeight)
                .frame(height: max(liveHeight, 1))
        } else if !response.isEmpty {
            AttributedTextView(
                attributed: renderer.render(response).attributedString,
                height: $responseHeight)
                .frame(height: max(responseHeight, 1))
        }
    }

    private func copyResponse(_ format: AnswerCopyFormat) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(
            format.pasteboardText(for: response, renderer: renderer),
            forType: .string)
        copied = format
    }
}

/// One icon-only control on a message: copy, edit, regenerate, continue.
///
/// Small, quiet, and always present. Text labels beside the role name made the
/// line read as a sentence rather than a heading with tools on it.
struct MessageActionButton: View {
    let symbol: String
    let help: String
    var tint: Color?
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Image(systemName: symbol)
                .font(.caption)
                .contentTransition(.symbolEffect(.replace))
                .frame(width: 18, height: 18)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(tint ?? .secondary)
        .help(help)
        .accessibilityLabel(help)
    }
}

/// Who is speaking. Both sides are the app's purple: the conversation is one
/// surface, and colouring only one half of it made the other read as chrome.
struct MessageRoleLabel: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.caption.weight(.semibold))
            .foregroundStyle(TUFFMacTheme.accentColor)
            .lineLimit(1)
            .accessibilityAddTraits(.isHeader)
    }
}

/// Collapsed reasoning that belongs to one message.
///
/// Expanded, it flows into the message rather than into a box of its own that
/// scrolls: one conversation, one scroll.
struct ThinkingDisclosure: View {
    let text: String
    let isRunning: Bool
    @Binding var isExpanded: Bool
    let reduceTransparency: Bool

    var body: some View {
        DisclosureGroup(isExpanded: $isExpanded) {
            Text(text)
                .font(.callout)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 8)
        } label: {
            Label(isRunning ? "Thinking…" : "Thinking", systemImage: "brain")
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)
        }
        .padding(10)
        .background(
            TUFFMacTheme.surfaceStyle(
                reduceTransparency: reduceTransparency,
                material: .thin),
            in: RoundedRectangle(cornerRadius: 10))
        .accessibilityLabel("Model thinking")
        .accessibilityHint("Shows or hides the reasoning for this message")
    }
}
