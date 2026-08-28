import Foundation
import Testing
@testable import TUFFEngine

@Suite("Harmony prompt renderer")
struct HarmonyPromptRendererTests {
    private typealias Message = GFTokenizer.Message

    @Test("Official special-token IDs are pinned")
    func tokenIDs() {
        let ids = HarmonySpecialTokenIDs.official
        #expect(ids.startOfText == 199_998)
        #expect(ids.endOfText == 199_999)
        #expect(ids.return == 200_002)
        #expect(ids.constrain == 200_003)
        #expect(ids.channel == 200_005)
        #expect(ids.start == 200_006)
        #expect(ids.end == 200_007)
        #expect(ids.message == 200_008)
        #expect(ids.call == 200_012)
        #expect(ids.structuralMarkerIDs == [
            200_002, 200_003, 200_005, 200_006, 200_007, 200_008, 200_012,
        ])
    }

    @Test("Single user turn matches the official Harmony framing")
    func singleUserTurn() throws {
        let rendered = try HarmonyPromptRenderer().render(
            messages: [Message(role: .user, content: "Hi")],
            reasoningEffort: .medium,
            currentDate: "2026-08-25")
        #expect(rendered ==
            "<|start|>system<|message|>"
            + "You are ChatGPT, a large language model trained by OpenAI.\n"
            + "Knowledge cutoff: 2024-06\n"
            + "Current date: 2026-08-25\n\n"
            + "Reasoning: medium\n\n"
            + "# Valid channels: analysis, commentary, final. "
            + "Channel must be included for every message."
            + "<|end|>"
            + "<|start|>user<|message|>Hi<|end|>"
            + "<|start|>assistant")
    }

    @Test("Reasoning effort is rendered without changing the conversation")
    func reasoningEffort() throws {
        for effort in GPTOSSReasoningEffort.allCases {
            let rendered = try HarmonyPromptRenderer().render(
                messages: [Message(role: .user, content: "Hi")],
                reasoningEffort: effort,
                currentDate: "2026-08-25")
            #expect(rendered.contains("\nReasoning: \(effort.rawValue)\n"))
        }
    }

    @Test("Instructions and function schemas render in the developer turn")
    func tools() throws {
        let rendered = try HarmonyPromptRenderer().render(
            messages: [
                Message(role: .developer, content: "Be concise."),
                Message(role: .user, content: "Weather?"),
            ],
            tools: [
                .init(
                    name: "get_weather",
                    description: "Look up weather",
                    parameters: .object([
                        "type": .string("object"),
                        "required": .array([.string("city")]),
                        "properties": .object([
                            "units": .object([
                                "type": .string("string"),
                                "enum": .array([.string("c"), .string("f")]),
                            ]),
                            "city": .object([
                                "type": .string("string"),
                                "description": .string("City name"),
                            ]),
                        ]),
                    ])),
            ],
            reasoningEffort: .high,
            currentDate: "2026-08-25")
        #expect(rendered.contains(
            "Calls to these tools must go to the commentary channel: 'functions'."))
        #expect(rendered.contains(
            "<|start|>developer<|message|># Instructions\n\nBe concise.\n\n# Tools\n\n"))
        #expect(rendered.contains("namespace functions {"))
        #expect(rendered.contains("// Look up weather\ntype get_weather = (_: {"))
        #expect(rendered.contains("// City name\ncity: string,"))
        #expect(rendered.contains("units?: \"c\" | \"f\","))
    }

    @Test("Tool calls and results use Harmony recipients and JSON")
    func toolHistory() throws {
        let call = GFTokenizer.HistoricalToolCall(
            id: "call_1",
            name: "get_weather",
            arguments: .object(["city": .string("Paris")]))
        let rendered = try HarmonyPromptRenderer().render(
            messages: [
                Message(role: .user, content: "Weather?"),
                Message(
                    role: .assistant,
                    content: nil,
                    thinking: "Need current data.",
                    toolCalls: [call]),
                Message(
                    role: .tool,
                    content: "sunny",
                    toolCallID: "call_1",
                    name: "get_weather"),
                Message(role: .assistant, content: "It is sunny."),
                Message(role: .user, content: "Thanks"),
            ],
            tools: [],
            currentDate: "2026-08-25")
        #expect(!rendered.contains("Need current data."),
                "analysis preceding a later final answer must not be preserved")
        #expect(rendered.contains(
            "<|start|>assistant to=functions.get_weather<|channel|>commentary json"
            + "<|message|>{\"city\":\"Paris\"}<|call|>"))
        #expect(rendered.contains(
            "<|start|>functions.get_weather to=assistant<|channel|>commentary"
            + "<|message|>\"sunny\"<|end|>"))
        #expect(rendered.contains(
            "<|start|>assistant<|channel|>final<|message|>It is sunny.<|end|>"))
    }

    @Test("Unresolved tool-call analysis is preserved")
    func preservesUnresolvedAnalysis() throws {
        let call = GFTokenizer.HistoricalToolCall(
            id: "call_1", name: "lookup", arguments: .object([:]))
        let rendered = try HarmonyPromptRenderer().render(
            messages: [
                Message(role: .user, content: "Look it up"),
                Message(
                    role: .assistant,
                    content: nil,
                    thinking: "Need the lookup tool.",
                    toolCalls: [call]),
            ],
            currentDate: "2026-08-25")
        #expect(rendered.contains(
            "<|start|>assistant<|channel|>analysis<|message|>"
            + "Need the lookup tool.<|end|>"))
    }

    @Test("Training render terminates the final assistant message with return")
    func trainingFinal() throws {
        let rendered = try HarmonyPromptRenderer().render(
            messages: [
                Message(role: .user, content: "Hi"),
                Message(
                    role: .assistant,
                    content: "Hello",
                    thinking: "Greet briefly."),
            ],
            currentDate: "2026-08-25",
            addGenerationPrompt: false)
        #expect(rendered.hasSuffix(
            "<|start|>assistant<|channel|>analysis<|message|>Greet briefly.<|end|>"
            + "<|start|>assistant<|channel|>final<|message|>Hello<|return|>"))
    }

    @Test("Malformed histories fail closed")
    func invalidHistory() {
        #expect(throws: GFTokenizerError.self) {
            _ = try HarmonyPromptRenderer().render(
                messages: [
                    Message(role: .user, content: "Hi"),
                    Message(role: .developer, content: "Too late"),
                ],
                currentDate: "2026-08-25")
        }
        #expect(throws: GFTokenizerError.self) {
            _ = try HarmonyPromptRenderer().render(
                messages: [Message(role: .tool, content: "orphan")],
                currentDate: "2026-08-25")
        }
        #expect(throws: GFTokenizerError.self) {
            _ = try HarmonyPromptRenderer().render(
                messages: [
                    Message(role: .user, content: "Hi"),
                    Message(
                        role: .assistant,
                        content: "analysis",
                        thinking: "more analysis",
                        toolCalls: [.init(
                            id: "call_1", name: "lookup", arguments: .object([:]))]),
                ],
                currentDate: "2026-08-25")
        }
        #expect(throws: GFTokenizerError.self) {
            _ = try HarmonyPromptRenderer().render(
                messages: [Message(role: .user, content: "Hi")],
                currentDate: "08/25/2026")
        }
    }
}
