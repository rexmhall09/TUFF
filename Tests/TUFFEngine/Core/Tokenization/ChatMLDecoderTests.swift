import Foundation
import Testing
@testable import TUFFEngine

/// StructuredAssistantDecoder in ChatML mode: `<think>` separation and
/// `<tool_call>` buffering driven by the fixture tokenizer's special-token IDs.
@Suite("ChatML decoder")
struct ChatMLDecoderTests {
    let tok: GFTokenizer

    init() async throws {
        self.tok = try await GFTokenizer.load(from: ChatMLTemplateTests.fixtureFolder())
    }

    private func decoder(allowedTools: Set<String> = ["get_weather"]) -> StructuredAssistantDecoder {
        StructuredAssistantDecoder(tokenizer: tok,
                                   allowedTools: allowedTools,
                                   idGenerator: { "call_fixed" })
    }

    /// Feeds text through the streaming detokenizer so each token carries the
    /// same delta the generation loop would produce.
    private func feed(_ text: String,
                      into decoder: StructuredAssistantDecoder) throws -> [StructuredAssistantEvent] {
        var events: [StructuredAssistantEvent] = []
        var detok = GFDetokenizer(tokenizer: tok)
        for id in tok.encode(text, addBOS: false) {
            events += try decoder.consume(tokenID: id, delta: detok.push(id))
        }
        return events
    }

    private func visibleText(_ events: [StructuredAssistantEvent]) -> String {
        events.reduce(into: "") { result, event in
            if case .content(let delta) = event { result += delta }
        }
    }

    private func thinkingText(_ events: [StructuredAssistantEvent]) -> String {
        events.reduce(into: "") { result, event in
            if case .thinking(let delta) = event { result += delta }
        }
    }

    @Test("Visible text streams through unchanged")
    func plainText() throws {
        let d = decoder()
        let events = try feed("Hello there!", into: d)
        #expect(visibleText(events) == "Hello there!")
        try d.finish()
        #expect(!d.hasToolCalls)
    }

    @Test("Think spans are separated, text after them is visible")
    func thinkSeparation() throws {
        let d = decoder()
        let events = try feed("<think>\nhidden reasoning\n</think>\n\nvisible answer", into: d)
        let text = visibleText(events)
        #expect(!text.contains("hidden reasoning"))
        #expect(text.contains("visible answer"))
        #expect(thinkingText(events).contains("hidden reasoning"))
        try d.finish()
    }

    @Test("Tool call spans buffer and emit a parsed call")
    func toolCallBuffering() throws {
        let d = decoder()
        let events = try feed(
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
            into: d)
        #expect(events == [.toolCall(ParsedToolCall(
            id: "call_fixed",
            name: "get_weather",
            arguments: .object(["city": .string("Paris")]),
            argumentsJSON: #"{"city":"Paris"}"#))])
        #expect(d.hasToolCalls)
        try d.finish()
    }

    @Test("Preamble text before the tool call stays visible")
    func preambleThenToolCall() throws {
        let d = decoder()
        let events = try feed(
            "Checking the weather now.\n\n<tool_call>\n<function=get_weather>\n</function>\n</tool_call>",
            into: d)
        #expect(visibleText(events) == "Checking the weather now.\n\n")
        #expect(d.hasToolCalls)
        try d.finish()
    }

    @Test("Unknown tool inside a call fails closed")
    func unknownToolFails() {
        let d = decoder(allowedTools: [])
        #expect(throws: ToolCallParserError.unknownTool("get_weather")) {
            _ = try feed(
                "<tool_call>\n<function=get_weather>\n</function>\n</tool_call>",
                into: d)
        }
    }

    @Test("Nested tool-call start is malformed")
    func nestedToolCallStart() throws {
        let d = decoder()
        _ = try d.consume(tokenID: tok.toolCallStartID, delta: "")
        #expect(throws: ToolCallParserError.malformed) {
            _ = try d.consume(tokenID: tok.toolCallStartID, delta: "")
        }
    }

    @Test("Tool-call end without a start is malformed")
    func endWithoutStart() {
        let d = decoder()
        #expect(throws: ToolCallParserError.malformed) {
            _ = try d.consume(tokenID: tok.toolCallEndID, delta: "")
        }
    }

    @Test("Finish with an unterminated tool call is malformed")
    func unterminatedToolCall() throws {
        let d = decoder()
        _ = try d.consume(tokenID: tok.toolCallStartID, delta: "")
        #expect(throws: ToolCallParserError.malformed) {
            try d.finish()
        }
    }

    /// The ChatML generation prompt for thinking-on ends with `<think>`, so the
    /// model continues inside the thought channel and never emits the token
    /// that opens it. A decoder that starts in `.visible` therefore streams the
    /// whole chain of thought as the answer — which is exactly what Qwen3.6
    /// did: every reply opened with "Thinking Process: 1. Analyze the
    /// Request…" before getting to the answer.
    @Test("A prompt that pre-opens the thought channel keeps reasoning hidden")
    func promptOpenedThinkingIsSeparated() throws {
        let d = StructuredAssistantDecoder(
            tokenizer: tok, allowedTools: [], promptOpensThinking: true)
        let events = try feed("hidden reasoning\n</think>\n\nvisible answer", into: d)

        #expect(thinkingText(events).contains("hidden reasoning"))
        let text = visibleText(events)
        #expect(!text.contains("hidden reasoning"),
                "the model's reasoning was streamed as the answer")
        #expect(text.contains("visible answer"))
    }

    @Test("Bytes flushed by the closing thought token stay hidden")
    func closingThoughtTokenKeepsItsFlushInTheThoughtChannel() throws {
        let d = StructuredAssistantDecoder(
            tokenizer: tok, allowedTools: [], promptOpensThinking: true)
        let events = try d.consume(tokenID: tok.thinkEndID!, delta: "last thought")
        #expect(thinkingText(events) == "last thought")
        #expect(visibleText(events).isEmpty)
        #expect(try d.consume(tokenID: 0, delta: "answer") == [.content("answer")])
    }

    /// The same continuation without the flag is the pre-fix behaviour, kept so
    /// the flag is shown to be what makes the difference.
    @Test("Without the flag the same tokens leak into the answer")
    func promptOpenedThinkingLeaksWithoutTheFlag() throws {
        let events = try feed(
            "hidden reasoning\n</think>\n\nvisible answer", into: decoder())
        #expect(visibleText(events).contains("hidden reasoning"))
    }

    /// Only ChatML with reasoning on pre-opens the channel. Gemma names its
    /// channels inline and Harmony has its own decoder.
    @Test("The flag is derived from the dialect and the reasoning setting")
    func promptOpensThinkingRule() {
        #expect(StructuredAssistantDecoder.promptOpensThinking(
            tokenizer: tok, reasoning: .on))
        #expect(!StructuredAssistantDecoder.promptOpensThinking(
            tokenizer: tok, reasoning: .off))
        #expect(StructuredAssistantDecoder.promptOpensThinking(
            dialect: .minimax, reasoning: .off))
        #expect(StructuredAssistantDecoder.promptOpensThinking(
            dialect: .minimax, reasoning: .on))
        #expect(!StructuredAssistantDecoder.promptOpensThinking(
            dialect: .gemma, reasoning: .on))
    }
}
