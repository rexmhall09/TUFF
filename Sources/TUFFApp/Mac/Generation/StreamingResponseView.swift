import AppKit
import TUFFAppCore
import TUFFMacPresentation
import SwiftUI

/// The answer currently being generated.
///
/// Tokens arrive faster than SwiftUI wants to rebuild a view, so this one keeps
/// the streaming fast path: a mailbox drained on a timer, the new suffix
/// appended to the text storage, and the whole response re-rendered as Markdown
/// exactly once, when the run ends. It reports the height it needs, so it sits
/// in the transcript's scroll view as an ordinary message rather than bringing
/// a second scroll view along with it.
struct StreamingResponseView: NSViewRepresentable {
    /// The response as the model state knows it. The mailbox is the fast path;
    /// this is the fallback and the source of truth when there is no mailbox.
    var text: String
    var mailbox: GenerationTranscriptMailbox?
    var isTerminal: Bool
    var showsPrefillPlaceholder: Bool
    @Binding var height: CGFloat

    @MainActor
    final class Coordinator: NSObject {
        weak var textView: MessageTextView?
        var mailbox: GenerationTranscriptMailbox?
        var isTerminal = false
        var showsPrefillPlaceholder = false
        var timer: Timer?
        var prefillAnimationTimer: Timer?
        let documentController = StreamingResponseDocumentController()

        func attach(_ textView: MessageTextView) {
            self.textView = textView
            guard timer == nil else { return }
            let timer = Timer(timeInterval: 0.1, target: self,
                              selector: #selector(drainMailbox),
                              userInfo: nil, repeats: true)
            timer.tolerance = 0.02
            RunLoop.main.add(timer, forMode: .common)
            self.timer = timer
        }

        func synchronize(
            text: String,
            mailbox: GenerationTranscriptMailbox?,
            isTerminal: Bool,
            showsPrefillPlaceholder: Bool
        ) {
            self.mailbox = mailbox
            self.isTerminal = isTerminal
            self.showsPrefillPlaceholder = showsPrefillPlaceholder
            apply(response: mailbox?.drain().completeText ?? text)
        }

        @objc private func drainMailbox() {
            guard let mailbox else { return }
            let snapshot = mailbox.drain()
            guard !snapshot.pendingText.isEmpty
                    || snapshot.completeText != documentController.response else {
                return
            }
            apply(response: snapshot.completeText)
        }

        @objc private func animatePrefillPlaceholderIfNeeded() {
            guard documentController.showsPrefillPlaceholder,
                  let textView,
                  let storage = textView.textStorage else { return }
            let selection = textView.selectedRanges.map(\.rangeValue)
            storage.beginEditing()
            let changed = documentController.advancePrefillAnimation(storage: storage)
            storage.endEditing()
            guard changed else { return }
            restoreSelection(selection, in: textView, storage: storage)
            textView.reportHeight()
        }

        private func apply(response: String) {
            guard let textView, let storage = textView.textStorage else { return }
            let selection = textView.selectedRanges.map(\.rangeValue)
            storage.beginEditing()
            let update = documentController.synchronize(
                storage: storage,
                response: response,
                isTerminal: isTerminal,
                showsPrefillPlaceholder: showsPrefillPlaceholder)
            storage.endEditing()
            updatePrefillAnimationTimer()
            guard update.mutation != .none else { return }
            restoreSelection(selection, in: textView, storage: storage)
            textView.reportHeight()
        }

        private func restoreSelection(
            _ selection: [NSRange],
            in textView: NSTextView,
            storage: NSTextStorage
        ) {
            let restored = StreamingResponseDocumentController.clampedRanges(
                selection, toLength: storage.length)
            if restored.isEmpty {
                textView.setSelectedRange(NSRange(location: storage.length, length: 0))
            } else {
                textView.selectedRanges = restored.map(NSValue.init(range:))
            }
        }

        private func updatePrefillAnimationTimer() {
            if documentController.showsPrefillPlaceholder {
                guard prefillAnimationTimer == nil else { return }
                let timer = Timer(
                    timeInterval: 0.25,
                    target: self,
                    selector: #selector(animatePrefillPlaceholderIfNeeded),
                    userInfo: nil,
                    repeats: true)
                timer.tolerance = 0.025
                RunLoop.main.add(timer, forMode: .common)
                prefillAnimationTimer = timer
            } else {
                prefillAnimationTimer?.invalidate()
                prefillAnimationTimer = nil
            }
        }

        func invalidate() {
            timer?.invalidate()
            timer = nil
            prefillAnimationTimer?.invalidate()
            prefillAnimationTimer = nil
            mailbox = nil
        }
    }

    func makeCoordinator() -> Coordinator { Coordinator() }

    func makeNSView(context: Context) -> MessageTextView {
        let textView = MessageTextView.make()
        textView.setAccessibilityLabel("Model answer")
        textView.onHeightChange = { measured in height = measured }
        return textView
    }

    func updateNSView(_ textView: MessageTextView, context: Context) {
        textView.onHeightChange = { measured in height = measured }
        context.coordinator.attach(textView)
        context.coordinator.synchronize(
            text: text,
            mailbox: mailbox,
            isTerminal: isTerminal,
            showsPrefillPlaceholder: showsPrefillPlaceholder)
    }

    static func dismantleNSView(_ nsView: MessageTextView, coordinator: Coordinator) {
        coordinator.invalidate()
    }
}
