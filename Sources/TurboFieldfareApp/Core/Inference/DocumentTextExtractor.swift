import Foundation
import PDFKit

/// Text pulled out of a file the user attached, with enough context to show what
/// happened before anything is sent.
///
/// Documents are not a model capability: every model reads text, so a document
/// becomes prompt text rather than a companion pack. What matters is that the
/// user can see how much text arrived, because a long file will not fit the
/// context window and silently trimming it would answer a different question
/// than the one asked.
public struct ExtractedDocument: Equatable, Sendable {
    public let displayName: String
    public let text: String
    /// Characters divided by four. Deliberately an estimate, and labelled as one
    /// wherever it is shown: the real count depends on the model's tokenizer.
    public var estimatedTokens: Int { max(1, text.count / 4) }

    public init(displayName: String, text: String) {
        self.displayName = displayName
        self.text = text
    }
}

public enum DocumentExtractionError: Error, Equatable, CustomStringConvertible {
    case unsupportedType(String)
    case unreadable(String)
    case empty(String)

    public var description: String {
        switch self {
        case .unsupportedType(let ext):
            return "TUFF cannot read .\(ext) files. Plain text, Markdown, CSV, "
                + "JSON, and PDF are supported."
        case .unreadable(let name):
            return "\(name) could not be read."
        case .empty(let name):
            return "\(name) has no extractable text. A scanned PDF needs OCR first."
        }
    }
}

public enum DocumentTextExtractor {
    public static let supportedExtensions: Set<String> = [
        "txt", "md", "markdown", "text", "csv", "tsv", "json", "yaml", "yml", "pdf",
    ]

    public static func extract(from url: URL) throws -> ExtractedDocument {
        let name = url.lastPathComponent
        let ext = url.pathExtension.lowercased()
        guard supportedExtensions.contains(ext) else {
            throw DocumentExtractionError.unsupportedType(ext)
        }

        let text: String
        if ext == "pdf" {
            guard let document = PDFDocument(url: url) else {
                throw DocumentExtractionError.unreadable(name)
            }
            var pages: [String] = []
            for index in 0..<document.pageCount {
                if let page = document.page(at: index), let content = page.string {
                    pages.append(content)
                }
            }
            text = pages.joined(separator: "\n\n")
        } else {
            guard let content = try? String(contentsOf: url, encoding: .utf8) else {
                throw DocumentExtractionError.unreadable(name)
            }
            text = content
        }

        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw DocumentExtractionError.empty(name) }
        return ExtractedDocument(displayName: name, text: trimmed)
    }

    /// The document as it is added to a prompt. The delimiters are plain text so
    /// every model reads them the same way, and the name is kept so the answer
    /// can refer to the file.
    public static func promptBlock(for document: ExtractedDocument) -> String {
        """
        <document name="\(document.displayName)">
        \(document.text)
        </document>
        """
    }
}
