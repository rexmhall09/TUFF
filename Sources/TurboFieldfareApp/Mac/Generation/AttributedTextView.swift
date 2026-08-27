import AppKit
import SwiftUI

/// A static, selectable attributed-text block that reports its own height.
///
/// Rendered responses carry math as `NSTextAttachment`, which SwiftUI's `Text`
/// cannot draw, so a completed message still needs a text view. This one holds
/// a single message rather than the whole conversation: it lays out once and
/// never streams, which is what lets the transcript be a list of messages.
struct AttributedTextView: NSViewRepresentable {
    let attributed: NSAttributedString
    @Binding var height: CGFloat

    func makeNSView(context: Context) -> NSTextView {
        let textView = NSTextView()
        textView.isEditable = false
        textView.isSelectable = true
        textView.drawsBackground = false
        textView.textContainerInset = .zero
        textView.textContainer?.lineFragmentPadding = 0
        textView.textContainer?.widthTracksTextView = true
        textView.isAutomaticLinkDetectionEnabled = false
        textView.isAutomaticDataDetectionEnabled = false
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.setContentHuggingPriority(.defaultHigh, for: .vertical)
        return textView
    }

    func updateNSView(_ textView: NSTextView, context: Context) {
        if textView.textStorage?.isEqual(to: attributed) != true {
            textView.textStorage?.setAttributedString(attributed)
        }
        recalculateHeight(textView)
    }

    private func recalculateHeight(_ textView: NSTextView) {
        guard let container = textView.textContainer,
              let layoutManager = textView.layoutManager else { return }
        layoutManager.ensureLayout(for: container)
        let measured = ceil(layoutManager.usedRect(for: container).height)
        guard measured > 0, abs(measured - height) > 0.5 else { return }
        DispatchQueue.main.async { height = measured }
    }
}
