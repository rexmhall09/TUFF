import AppKit
import Foundation

/// The text storage of the one answer still being generated.
///
/// Only the live answer goes through here. Completed messages are ordinary
/// views that lay out once, so this controller holds a single response rather
/// than the whole conversation: the transcript is a list of messages, and the
/// streaming fast path — append the suffix, format once at the end — applies to
/// the last of them.
@MainActor
public final class StreamingResponseDocumentController {
    public enum Mutation: Equatable {
        case none
        case rebuilt
        case appended
        case finalized
    }

    public struct UpdateResult {
        public let mutation: Mutation
        public let assistantRange: NSRange

        public init(mutation: Mutation, assistantRange: NSRange) {
            self.mutation = mutation
            self.assistantRange = assistantRange
        }
    }

    public private(set) var response = ""
    public private(set) var isFinalized = false
    public private(set) var showsPrefillPlaceholder = false
    public private(set) var assistantRange = NSRange(location: 0, length: 0)
    private var prefillPlaceholderRange: NSRange?
    private var prefillDotCount = 0

    private let renderer: ResponseMarkdownRenderer
    /// The zoom this document is set at. The streamed text is an
    /// `NSAttributedString`, so the size is baked in here as well as in the
    /// renderer, and both have to agree or the answer changes size as it
    /// finishes.
    private let scale: CGFloat

    public init(
        renderer: ResponseMarkdownRenderer? = nil,
        scale: CGFloat = 1
    ) {
        self.scale = scale
        self.renderer = renderer ?? ResponseMarkdownRenderer(scale: scale)
    }

    public static func clampedRanges(
        _ ranges: [NSRange],
        toLength length: Int
    ) -> [NSRange] {
        ranges.map { range in
            let location = min(max(range.location, 0), length)
            let available = max(0, length - location)
            return NSRange(location: location, length: min(max(range.length, 0), available))
        }
    }

    public static func shouldRunPrefillAnimation(
        response: String,
        isTerminal: Bool,
        requested: Bool
    ) -> Bool {
        requested && response.isEmpty && !isTerminal
    }

    @discardableResult
    public func synchronize(
        storage: NSMutableAttributedString,
        response: String,
        isTerminal: Bool,
        showsPrefillPlaceholder: Bool = false
    ) -> UpdateResult {
        let responseChanged = response != self.response
        let displaysPrefillPlaceholder = Self.shouldRunPrefillAnimation(
            response: response,
            isTerminal: isTerminal,
            requested: showsPrefillPlaceholder)
        let needsRebuild = !response.hasPrefix(self.response)
            || (isFinalized && !isTerminal)
            || displaysPrefillPlaceholder != self.showsPrefillPlaceholder

        var mutation: Mutation = .none
        if needsRebuild
            || storage.length == 0
                && (!response.isEmpty || displaysPrefillPlaceholder) {
            rebuild(
                storage: storage,
                response: response,
                showsPrefillPlaceholder: displaysPrefillPlaceholder)
            mutation = .rebuilt
        } else if response.count > self.response.count {
            let delta = String(response.dropFirst(self.response.count))
            storage.append(NSAttributedString(
                string: delta,
                attributes: Self.responseAttributes(scale: scale)))
            assistantRange.length += (delta as NSString).length
            mutation = .appended
        }

        self.response = response
        self.showsPrefillPlaceholder = displaysPrefillPlaceholder

        if isTerminal && (!isFinalized || responseChanged) {
            let rendered = renderer.render(response).attributedString
            storage.replaceCharacters(in: assistantRange, with: rendered)
            assistantRange.length = rendered.length
            isFinalized = true
            mutation = .finalized
        } else if !isTerminal {
            isFinalized = false
        }

        return UpdateResult(mutation: mutation, assistantRange: assistantRange)
    }

    @discardableResult
    public func advancePrefillAnimation(
        storage: NSMutableAttributedString
    ) -> Bool {
        guard showsPrefillPlaceholder, var range = prefillPlaceholderRange else {
            return false
        }
        prefillDotCount = (prefillDotCount + 1) % 4
        let replacement = NSAttributedString(
            string: Self.prefillPlaceholder(dotCount: prefillDotCount),
            attributes: Self.prefillPlaceholderAttributes(scale: scale))
        storage.replaceCharacters(in: range, with: replacement)
        range.length = replacement.length
        prefillPlaceholderRange = range
        return true
    }

    private func rebuild(
        storage: NSMutableAttributedString,
        response: String,
        showsPrefillPlaceholder: Bool
    ) {
        let document = NSMutableAttributedString()
        assistantRange = NSRange(location: 0, length: 0)
        document.append(NSAttributedString(
            string: response,
            attributes: Self.responseAttributes(scale: scale)))
        assistantRange.length = (response as NSString).length
        prefillDotCount = 0
        prefillPlaceholderRange = nil
        if showsPrefillPlaceholder {
            let placeholder = NSAttributedString(
                string: Self.prefillPlaceholder(dotCount: prefillDotCount),
                attributes: Self.prefillPlaceholderAttributes(scale: scale))
            prefillPlaceholderRange = NSRange(
                location: document.length,
                length: placeholder.length)
            document.append(placeholder)
        }
        storage.setAttributedString(document)
        isFinalized = false
    }

    public static func responseAttributes(scale: CGFloat = 1) -> [NSAttributedString.Key: Any] {
        let style = NSMutableParagraphStyle()
        style.lineSpacing = 3 * scale
        style.paragraphSpacing = 6 * scale
        return [
            .font: NSFont.systemFont(ofSize: NSFont.systemFontSize * scale),
            .foregroundColor: NSColor.labelColor,
            .paragraphStyle: style,
        ]
    }

    private static func prefillPlaceholderAttributes(scale: CGFloat = 1) -> [NSAttributedString.Key: Any] {
        var attributes = responseAttributes(scale: scale)
        attributes[.foregroundColor] = NSColor.secondaryLabelColor
        return attributes
    }

    private static func prefillPlaceholder(dotCount: Int) -> String {
        "Processing your prompt" + String(repeating: ".", count: dotCount)
    }
}
