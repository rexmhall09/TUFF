import Foundation
import Testing
@testable import TurboFieldfareAppCore

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

    @Test("the token estimate scales with length and is never zero")
    func estimatesTokens() {
        #expect(ExtractedDocument(displayName: "a", text: "x").estimatedTokens == 1)
        let long = ExtractedDocument(
            displayName: "a", text: String(repeating: "x", count: 4_000))
        #expect(long.estimatedTokens == 1_000)
    }
}
