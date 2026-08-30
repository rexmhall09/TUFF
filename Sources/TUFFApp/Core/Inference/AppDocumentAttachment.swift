import Foundation

/// A file the user attached to a message, kept as a first-class attachment
/// rather than pasted into the prompt box.
///
/// The text still reaches the model as prompt text — that is the only thing
/// every model can read — but the composer and the transcript show the file,
/// not its contents. Dumping a PDF into the prompt made a two-line question
/// look like a ten-thousand-word one, and there was no way to take the file
/// back off without hunting for where its text ended.
public struct AppDocumentAttachment: Codable, Equatable, Sendable, Identifiable {
    public let id: UUID
    public let displayName: String
    /// Extracted plain text, exactly as it will be sent.
    public let text: String

    public init(id: UUID = UUID(), displayName: String, text: String) {
        self.id = id
        self.displayName = displayName
        self.text = text
    }

    /// Characters divided by four, and labelled as an estimate wherever it is
    /// shown: the real count depends on the model's tokenizer.
    public var estimatedTokens: Int { max(1, text.count / 4) }

    public var fileExtension: String {
        (displayName as NSString).pathExtension.lowercased()
    }

    /// The document as it is added to a prompt. The delimiters are plain text
    /// so every model reads them the same way, and the name is kept so the
    /// answer can refer to the file.
    public var promptBlock: String {
        documentPromptBlock(displayName: displayName, text: text)
    }

    /// A short badge for the tile: "PDF", "MD", "TXT".
    public var kindLabel: String {
        fileExtension.isEmpty ? "TXT" : fileExtension.uppercased()
    }

    /// Deliberately coarse, and always prefixed with a tilde. The figure is
    /// characters over four; printing it to the unit invited it to be read as a
    /// measurement of what the model's tokenizer will actually produce.
    public var estimatedTokensLabel: String {
        Self.tokensLabel(estimatedTokens)
    }

    public static func tokensLabel(_ count: Int) -> String {
        count < 1_000
            ? "~\(count)"
            : "~\(String(format: "%.1f", Double(count) / 1_000))k"
    }
}

/// Keep a filename from changing the prompt wrapper itself. File contents are
/// intentionally unescaped plain text, but the name lives inside a quoted
/// attribute and macOS filenames may contain quotes or line breaks.
func documentPromptBlock(displayName: String, text: String) -> String {
    let escapedName = displayName
        .replacingOccurrences(of: "&", with: "&amp;")
        .replacingOccurrences(of: "\"", with: "&quot;")
        .replacingOccurrences(of: "<", with: "&lt;")
        .replacingOccurrences(of: ">", with: "&gt;")
        .replacingOccurrences(of: "\n", with: "&#10;")
        .replacingOccurrences(of: "\r", with: "&#13;")
        .replacingOccurrences(of: "\t", with: "&#9;")
    return "<document name=\"\(escapedName)\">\n\(text)\n</document>"
}

public extension Array where Element == AppDocumentAttachment {
    /// The text these documents contribute to a prompt, ahead of what the user
    /// typed. Empty when there are no documents, so a plain message is
    /// unchanged by their existence.
    var promptPreamble: String {
        isEmpty ? "" : map(\.promptBlock).joined(separator: "\n\n")
    }

    var estimatedTokens: Int {
        reduce(0) { $0 + $1.estimatedTokens }
    }
}
