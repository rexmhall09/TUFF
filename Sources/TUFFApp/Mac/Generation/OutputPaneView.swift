import AppKit
import TUFFEngine
import TUFFAppCore
import TUFFMacPresentation
import SwiftUI

/// The conversation: one scrolling surface holding every message.
///
/// Each exchange is a view that sizes to its own content, including the one
/// being generated. Nothing inside a message scrolls on its own — the reasoning
/// disclosure and the attachment row used to, which put a scroller inside a
/// scroller and made the wheel do different things depending on what the
/// pointer was over.
struct OutputPaneView: View {
    let model: AppModel
    /// True while the view should keep itself at the newest message. Set by the
    /// reader arriving at the bottom and cleared by the reader leaving it, so
    /// following the answer never fights someone reading back.
    @State private var isPinnedToBottom = true

    var body: some View {
        Group {
            if model.hasOutputTranscript {
                transcript
            } else {
                placeholder
            }
        }
        .contextMenu {
            Button("Copy conversation") {
                copy(conversationPlainText)
            }
            .disabled(conversationPlainText.isEmpty)

            Divider()

            Button("New Chat") { model.clearOutput() }
                .disabled(model.isRunning || !model.hasOutputTranscript)
        }
    }

    private var placeholder: some View {
        EmptyConversationLayout(spacing: 8) {
            EmptyPlaceholderIcon(systemName: placeholderSymbol)
                .frame(width: 32, height: 32)

            emptyPlaceholderContent
        }
        .padding(.horizontal, 24)
        .padding(.vertical, 20)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private static let bottomID = "tuff.transcript.bottom"

    private var transcript: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 26) {
                    ForEach(model.conversation) { turn in
                        recordedMessage(turn)
                    }
                    if model.hasLiveMessage {
                        liveMessage.id(model.runIdentity)
                    }
                    // The scroll target. Anchoring on the last message instead
                    // stopped short whenever that message was taller than the
                    // viewport, which is most of them.
                    Color.clear
                        .frame(height: 1)
                        .id(Self.bottomID)
                        .accessibilityHidden(true)
                }
                .padding(.horizontal, 26)
                .padding(.vertical, 22)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            .defaultScrollAnchor(.bottom, for: .initialOffset)
            .onScrollPhaseChange { _, phase in
                // A programmatic scroll reports `.animating`, so this reads the
                // reader only.
                if phase == .interacting || phase == .tracking {
                    isPinnedToBottom = false
                }
            }
            .onScrollGeometryChange(for: TranscriptGeometry.self) { geometry in
                TranscriptGeometry(
                    offset: geometry.contentOffset.y,
                    contentHeight: geometry.contentSize.height,
                    viewportHeight: geometry.containerSize.height)
            } action: { previous, current in
                if current.isAtBottom {
                    isPinnedToBottom = true
                } else if isPinnedToBottom,
                          current.contentHeight > previous.contentHeight {
                    // The answer grew under a view that was at the bottom.
                    scrollToBottom(proxy)
                }
            }
            .onChange(of: model.runIdentity) { _, _ in
                // A new message is the thing that was just asked for: it goes
                // on screen whatever the reader was looking at.
                isPinnedToBottom = true
                scrollToBottom(proxy)
            }
        }
    }

    private func scrollToBottom(_ proxy: ScrollViewProxy) {
        var transaction = Transaction()
        transaction.disablesAnimations = true
        withTransaction(transaction) {
            proxy.scrollTo(Self.bottomID, anchor: .bottom)
        }
    }

    private func recordedMessage(_ turn: AppChatTurn) -> some View {
        TranscriptMessageView(
            prompt: turn.prompt,
            response: turn.response,
            thinking: turn.thinking ?? "",
            images: turn.images,
            documents: turn.documents,
            modelName: model.modelShortName(forProfileKey: turn.modelID),
            renderer: Self.sharedRenderer,
            actions: actions(for: turn))
    }

    /// Rewind controls, offered only while the chat is idle enough to rewind.
    private func actions(for turn: AppChatTurn) -> TranscriptMessageView.MessageActions? {
        guard model.canRewind(to: turn) else { return nil }
        let isNewest = model.conversation.last?.id == turn.id
        return TranscriptMessageView.MessageActions(
            edit: { model.editMessage(turn) },
            regenerate: { model.regenerate(turn) },
            carryOn: isNewest && model.canContinueLastAnswer
                ? { model.continueLastAnswer() } : nil)
    }

    private var liveMessage: some View {
        TranscriptMessageView(
            prompt: model.outputPromptText,
            response: model.outputResponsePlainText,
            thinking: model.outputThinkingText,
            images: model.outputImageAttachments,
            documents: model.outputDocumentAttachments,
            modelName: model.selectedDescriptor.shortName,
            renderer: Self.sharedRenderer,
            live: TranscriptMessageView.LiveResponse(
                mailbox: model.hasLiveMessage
                    ? model.generationTranscriptMailbox : nil,
                isTerminal: !model.isRunning,
                showsPrefillPlaceholder: model.isRunning
                    && model.outputResponsePlainText.isEmpty))
    }

    private static let sharedRenderer = ResponseMarkdownRenderer()

    /// The whole conversation as plain text, each answer labelled with the
    /// model that produced it.
    private var conversationPlainText: String {
        var blocks: [String] = []
        for turn in model.conversation {
            let name = model.modelShortName(forProfileKey: turn.modelID)
            blocks.append("You:\n\(turn.modelPrompt)\n\n\(name):\n\(turn.response)")
        }
        let live = model.outputConversationPlainText
        if !live.isEmpty { blocks.append(live) }
        return blocks.joined(separator: "\n\n")
    }

    private var emptyPlaceholderContent: some View {
        VStack(spacing: 8) {
            if !needsModelLoad {
                Text("Choose a predefined example or write your own prompt.")
                    .font(.headline)
                Text("Describe the goal, relevant context, and any constraints.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
            }
            if isLoadingModel {
                LoadingModelText()
                    .font(.callout)
                    .foregroundStyle(.tertiary)
            } else if let placeholderHint {
                Text(placeholderHint)
                    .font(.callout)
                    .foregroundStyle(.tertiary)
            }
            if let detail = model.presentation.detail {
                Text(detail)
                    .font(.caption)
                    .foregroundStyle(model.presentation.severity == .error ? .red : .secondary)
                    .multilineTextAlignment(.center)
            }
            if model.canLoadModel {
                Button(model.loadState.isFailed ? "Retry Load" : "Load Model",
                       action: model.loadModel)
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
            } else if isLoadingModel {
                Button("Load Model", action: {})
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .hidden()
                    .accessibilityHidden(true)
            } else if model.canReloadModel {
                Button("Reload Model", action: model.reloadModel)
                    .buttonStyle(.borderedProminent)
            }
        }
        .frame(maxWidth: .infinity)
    }

    private var needsModelLoad: Bool {
        !model.loadState.isReady
    }

    private var isLoadingModel: Bool {
        if case .loading = model.loadState { return true }
        return false
    }

    private var placeholderSymbol: String {
        "cube.transparent"
    }

    private var placeholderHint: String? {
        if model.loadState.isFailed { return "The model could not be loaded" }
        if model.hasStaleLoadedRuntime { return "Reload the model to use changed settings" }
        return needsModelLoad ? "Load the model to begin" : nil
    }

    private func copy(_ text: String) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(text, forType: .string)
    }
}

/// What the transcript needs from its scroll view to decide whether to follow.
private struct TranscriptGeometry: Equatable {
    var offset: CGFloat = 0
    var contentHeight: CGFloat = 0
    var viewportHeight: CGFloat = 0

    /// A band rather than an exact match: a streaming answer grows by a line at
    /// a time, and an exact test read every one of those as the reader leaving.
    var isAtBottom: Bool {
        offset + viewportHeight >= contentHeight - 32
    }
}

struct SubmittedImageThumbnail: View {
    let attachment: AppImageAttachment
    let maximumSize: CGSize
    @State private var image: NSImage?

    init(
        attachment: AppImageAttachment,
        maximumSize: CGSize = CGSize(width: 48, height: 48)
    ) {
        self.attachment = attachment
        self.maximumSize = maximumSize
    }

    var body: some View {
        Group {
            if let image {
                // Filled and cropped to a tile, not fitted inside one. Fitting
                // gave every attachment a different height — a screenshot came
                // out a third the height of a portrait photo — so a row of them
                // was ragged, and the remove badge, pinned to the tile, floated
                // clear of the short ones.
                Image(nsImage: image)
                    .resizable()
                    .scaledToFill()
                    .frame(width: maximumSize.width, height: maximumSize.height)
                    .clipped()
            } else {
                Image(systemName: "photo")
                    .foregroundStyle(.tertiary)
                    .frame(width: maximumSize.width, height: maximumSize.height)
            }
        }
        .frame(width: maximumSize.width, height: maximumSize.height)
        .background(.quaternary)
        .clipShape(RoundedRectangle(cornerRadius: 10))
        .overlay {
            RoundedRectangle(cornerRadius: 10)
                .stroke(.separator.opacity(0.5), lineWidth: 0.5)
        }
        .accessibilityLabel("Attached image \(attachment.displayName)")
        .help(attachment.displayName)
        .task(id: "\(attachment.id)-\(maximumSize.width)x\(maximumSize.height)") {
            let url = attachment.fileURL
            let key = attachment.sha256
            let pixels = Int(ceil(max(maximumSize.width, maximumSize.height) * 2))
            // This closure runs on the main actor, and a source near the decode
            // budget takes long enough that decoding here stalled the window.
            // The decode is done off the main thread and left in the cache; the
            // read below is then a lookup.
            await Task.detached(priority: .userInitiated) {
                _ = Self.loadThumbnail(
                    at: url, maximumPixelSize: pixels, cacheKey: key)
            }.value
            image = Self.loadThumbnail(
                at: url, maximumPixelSize: pixels, cacheKey: key)
        }
    }

    /// Every attached-image decode in the app goes through here, so the decode
    /// budget cannot be applied to one caller and forgotten on the next. Nil
    /// means refused or unreadable; both callers draw their placeholder for it.
    nonisolated static func loadThumbnail(
        at url: URL,
        maximumPixelSize: Int,
        cacheKey: String? = nil
    ) -> NSImage? {
        TranscriptImageLoader.thumbnail(
            at: url,
            maximumPixelSize: maximumPixelSize,
            budget: decodeBudget,
            cacheKey: cacheKey)
    }

    /// The single point where the app binds the runtime's limits, so the two
    /// cannot drift apart: see `VisionImageLimits` in
    /// Runtime/Vision/Preprocessing/ImageMetadataReader.swift.
    nonisolated private static let decodeBudget: TranscriptImageLoader.Budget = {
        let limits = VisionImageLimits()
        return TranscriptImageLoader.Budget(
            maximumSourcePixels: limits.maximumSourcePixels,
            maximumSourceDimension: limits.maximumSourceDimension,
            maximumDecodedBytes: limits.maximumDecodedBytes,
            allowedTypeIdentifiers: limits.allowedTypeIdentifiers)
    }()
}

private struct EmptyPlaceholderIcon: View {
    let systemName: String

    var body: some View {
        Image(systemName: systemName)
            .font(.title2)
            .foregroundStyle(.quaternary)
            .accessibilityHidden(true)
    }
}

private struct EmptyConversationLayout: Layout {
    let spacing: CGFloat

    func sizeThatFits(
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) -> CGSize {
        proposal.replacingUnspecifiedDimensions()
    }

    func placeSubviews(
        in bounds: CGRect,
        proposal: ProposedViewSize,
        subviews: Subviews,
        cache: inout ()
    ) {
        guard subviews.count == 2 else { return }

        let iconSize = subviews[0].sizeThatFits(.unspecified)
        let iconCenter = CGPoint(x: bounds.midX, y: bounds.midY)
        subviews[0].place(
            at: iconCenter,
            anchor: .center,
            proposal: ProposedViewSize(
                width: iconSize.width,
                height: iconSize.height))

        subviews[1].place(
            at: CGPoint(
                x: bounds.midX,
                y: iconCenter.y + iconSize.height / 2 + spacing),
            anchor: .top,
            proposal: ProposedViewSize(width: bounds.width, height: nil))
    }
}

private struct LoadingModelText: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var animationStart = Date()

    var body: some View {
        if reduceMotion {
            label(dotCount: 3)
        } else {
            TimelineView(.periodic(from: .now, by: 0.25)) { context in
                let elapsed = max(0, context.date.timeIntervalSince(animationStart))
                label(dotCount: Int(elapsed / 0.25) % 4)
            }
        }
    }

    private func label(dotCount: Int) -> some View {
        ZStack(alignment: .leading) {
            Text("Loading Model...").hidden()
            Text("Loading Model" + String(repeating: ".", count: dotCount))
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Loading Model")
    }
}

#if DEBUG && !TUFF_NO_PREVIEWS
private struct MessagePreview: View {
    let prompt: String
    let response: String
    var thinking = ""

    var body: some View {
        TranscriptMessageView(
            prompt: prompt,
            response: response,
            thinking: thinking,
            images: [],
            documents: [],
            modelName: "Gemma 4 E4B",
            renderer: ResponseMarkdownRenderer())
            .padding(24)
            .frame(width: 720)
    }
}

#Preview("Empty") {
    VStack(spacing: 8) {
        Image(systemName: "cube.transparent")
            .font(.title2)
            .foregroundStyle(.quaternary)
        Text("Choose a predefined example or write your own prompt.")
            .font(.headline)
        Text("Describe the goal, relevant context, and any constraints.")
            .foregroundStyle(.secondary)
    }
    .frame(width: 720, height: 420)
}

#Preview("Completed prose") {
    MessagePreview(
        prompt: "Explain this clearly.",
        response: "# A clear answer\n\nHere is a concise explanation with **useful emphasis**.\n\n- First point\n- Second point")
}

#Preview("Completed code") {
    MessagePreview(
        prompt: "Write `fibonacci`.",
        response: "Use `fibonacci(7)`:\n\n```python\ndef fibonacci(n: int) -> list[int]:\n    return []\n```")
}

#Preview("With reasoning") {
    MessagePreview(
        prompt: "What is 17 × 23?",
        response: "391.",
        thinking: "17 × 23 = 17 × 20 + 17 × 3 = 340 + 51 = 391.")
}
#endif
