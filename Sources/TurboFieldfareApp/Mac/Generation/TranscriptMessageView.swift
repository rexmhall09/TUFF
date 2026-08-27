import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

/// One completed exchange: what was asked, what came back, and the controls that
/// belong to that message rather than to the whole conversation.
struct TranscriptMessageView: View {
    let turn: AppChatTurn
    let attachments: [AppImageAttachment]
    let renderer: ResponseMarkdownRenderer
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency

    @State private var thinkingExpanded = false
    @State private var promptHeight: CGFloat = 0
    @State private var responseHeight: CGFloat = 0
    @State private var copied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            promptSection
            if let thinking = turn.thinking, !thinking.isEmpty {
                ThinkingDisclosure(
                    text: thinking,
                    isRunning: false,
                    isExpanded: $thinkingExpanded,
                    reduceTransparency: reduceTransparency)
            }
            responseSection
            copyRow
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var promptSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("You")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            // Attachments render with the message they were sent with, so a
            // prompt that carried images never shows as bare text.
            if !attachments.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(attachments, id: \.id) { attachment in
                            SubmittedImageThumbnail(attachment: attachment)
                        }
                    }
                }
            }
            Text(turn.prompt)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var responseSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Answer")
                .font(.caption.weight(.semibold))
                .foregroundStyle(TurboFieldfareMacTheme.accentColor)
            AttributedTextView(
                attributed: renderer.render(turn.response).attributedString,
                height: $responseHeight)
                .frame(height: max(responseHeight, 1))
        }
    }

    private var copyRow: some View {
        HStack(spacing: 8) {
            Button {
                copyResponse()
            } label: {
                Label(copied ? "Copied" : "Copy",
                      systemImage: copied ? "checkmark" : "doc.on.doc")
                    .font(.caption)
                    .labelStyle(.titleAndIcon)
            }
            .buttonStyle(.borderless)
            .foregroundStyle(.secondary)
            .help("Copy this answer")
            Spacer(minLength: 0)
        }
        .task(id: copied) {
            guard copied else { return }
            try? await Task.sleep(for: .seconds(1.5))
            guard !Task.isCancelled else { return }
            copied = false
        }
    }

    private func copyResponse() {
        let plain = renderer.plainText(turn.response)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(plain, forType: .string)
        copied = true
    }
}

/// Collapsed reasoning that belongs to one message.
struct ThinkingDisclosure: View {
    let text: String
    let isRunning: Bool
    @Binding var isExpanded: Bool
    let reduceTransparency: Bool

    var body: some View {
        DisclosureGroup(isExpanded: $isExpanded) {
            ScrollView {
                Text(text)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.top, 8)
            }
            .frame(maxHeight: 220)
        } label: {
            Label(isRunning ? "Thinking…" : "Thinking", systemImage: "brain")
                .font(.caption.weight(.medium))
                .foregroundStyle(.secondary)
        }
        .padding(10)
        .background(
            TurboFieldfareMacTheme.surfaceStyle(
                reduceTransparency: reduceTransparency,
                material: .thin),
            in: RoundedRectangle(cornerRadius: 10))
        .accessibilityLabel("Model thinking")
        .accessibilityHint("Shows or hides the reasoning for this message")
    }
}
