import Foundation
import Testing
@testable import TUFFAppCore

@Suite("Document extraction")
struct DocumentTextExtractorTests {
    private func write(_ contents: String, as name: String) throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-doc-\(UUID().uuidString)-\(name)")
        try contents.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    @Test("plain text and Markdown are read as they are")
    func readsPlainText() throws {
        let url = try write("# Title\n\nBody text.", as: "notes.md")
        defer { try? FileManager.default.removeItem(at: url) }
        let document = try DocumentTextExtractor.extract(from: url)
        #expect(document.text == "# Title\n\nBody text.")
        #expect(document.displayName.hasSuffix("notes.md"))
    }

    @Test("extensionless plain-text files are accepted")
    func readsExtensionlessPlainText() throws {
        let url = try write("Build instructions", as: "README")
        defer { try? FileManager.default.removeItem(at: url) }

        #expect(DocumentTextExtractor.canExtract(from: url))
        #expect(try DocumentTextExtractor.extract(from: url).text
                == "Build instructions")
    }

    @Test("an unsupported type is refused by name")
    func refusesUnsupportedType() throws {
        let url = try write("binary-ish", as: "model.safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: DocumentExtractionError.self) {
            try DocumentTextExtractor.extract(from: url)
        }
    }

    @Test("an empty file is refused rather than attached as nothing")
    func refusesEmptyFile() throws {
        let url = try write("   \n\n  ", as: "blank.txt")
        defer { try? FileManager.default.removeItem(at: url) }
        #expect(throws: DocumentExtractionError.self) {
            try DocumentTextExtractor.extract(from: url)
        }
    }

    @Test("the prompt block keeps the file name")
    func promptBlockNamesTheFile() {
        let document = ExtractedDocument(displayName: "report.txt", text: "hello")
        let block = DocumentTextExtractor.promptBlock(for: document)
        #expect(block.contains("name=\"report.txt\""))
        #expect(block.contains("hello"))
    }

    @Test("filename punctuation cannot break the prompt wrapper")
    func promptBlockEscapesTheFileName() {
        let document = ExtractedDocument(
            displayName: "report\"&\n</document>.txt", text: "hello")
        let block = DocumentTextExtractor.promptBlock(for: document)
        #expect(block.hasPrefix(
            "<document name=\"report&quot;&amp;&#10;&lt;/document&gt;.txt\">"))
        #expect(block.components(separatedBy: "</document>").count == 2)
    }

    @Test("the token estimate scales with length and is never zero")
    func estimatesTokens() {
        #expect(ExtractedDocument(displayName: "a", text: "x").estimatedTokens == 1)
        let long = ExtractedDocument(
            displayName: "a", text: String(repeating: "x", count: 4_000))
        #expect(long.estimatedTokens == 1_000)
    }

    /// The open panel offers `public.plain-text`, because `.txt` is that type,
    /// so anything conforming to it can be chosen. The extractor has to accept
    /// the same set or the panel offers files it then refuses.
    @Test("plain-text files beyond the named list are accepted")
    func acceptsAnyPlainTextFile() {
        for name in ["Model.swift", "server.log", "build.py", "run.sh", "main.c"] {
            #expect(DocumentTextExtractor.canExtract(
                from: URL(fileURLWithPath: "/tmp/\(name)")),
                    "\(name) is plain text and the panel will offer it")
        }
    }

    /// Markup containers conform to `public.text` but not to `public.plain-text`.
    /// Reading one as UTF-8 would hand the model its markup and call it the
    /// document, so the boundary is drawn at plain text.
    @Test("markup and binary documents are still refused")
    func refusesNonPlainTextDocuments() {
        for name in ["notes.rtf", "page.html", "data.xml", "report.docx",
                     "deck.key", "sheet.pages"] {
            #expect(!DocumentTextExtractor.canExtract(
                from: URL(fileURLWithPath: "/tmp/\(name)")),
                    "\(name) is not plain text and must not be read as UTF-8")
        }
    }

    /// The named extensions carry types that are not plain text — JSON and YAML
    /// among them — so the list is load-bearing rather than documentation.
    @Test("the named extensions are accepted whatever their type says")
    func acceptsEveryNamedExtension() {
        for ext in DocumentTextExtractor.supportedExtensions {
            #expect(DocumentTextExtractor.canExtract(
                from: URL(fileURLWithPath: "/tmp/file.\(ext)")))
        }
    }

    /// Whatever the panel offers, the extractor must accept.
    @Test("every offered content type is one the extractor reads")
    func theOpenPanelOffersNothingItWouldRefuse() {
        for type in DocumentTextExtractor.allowedContentTypes {
            guard let ext = type.preferredFilenameExtension as String? else {
                continue
            }
            #expect(DocumentTextExtractor.canExtract(
                from: URL(fileURLWithPath: "/tmp/file.\(ext)")),
                    "the panel offers .\(ext) but the extractor refuses it")
        }
    }
}
