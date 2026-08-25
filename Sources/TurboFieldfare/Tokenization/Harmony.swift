import Foundation

/// GPT-OSS exposes graded reasoning effort through the Harmony system message.
/// It is intentionally separate from `ChatReasoning`, whose only valid values
/// are the native on/off controls used by Gemma and Qwen.
public enum GPTOSSReasoningEffort: String, Codable, Sendable, Equatable, CaseIterable {
    case low
    case medium
    case high
}

/// The special-token contract shared by the official 20B and 120B GPT-OSS
/// tokenizers. Instances resolved from an installed tokenizer are compared to
/// these values during load, so a foreign Harmony-like vocabulary fails closed.
public struct HarmonySpecialTokenIDs: Sendable, Equatable {
    public let startOfText: Int32
    public let endOfText: Int32
    public let `return`: Int32
    public let constrain: Int32
    public let channel: Int32
    public let start: Int32
    public let end: Int32
    public let message: Int32
    public let call: Int32

    public static let official = HarmonySpecialTokenIDs(
        startOfText: 199_998,
        endOfText: 199_999,
        return: 200_002,
        constrain: 200_003,
        channel: 200_005,
        start: 200_006,
        end: 200_007,
        message: 200_008,
        call: 200_012)

    var structuralMarkerIDs: Set<Int32> {
        [self.return, constrain, channel, start, end, message, call]
    }
}

/// Foundation-only renderer for the official GPT-OSS Harmony conversation
/// format. It keeps the current date explicit so tests, benchmarks, and cached
/// prompts can be reproduced byte-for-byte.
public struct HarmonyPromptRenderer: Sendable {
    public static let defaultModelIdentity =
        "You are ChatGPT, a large language model trained by OpenAI."
    public static let knowledgeCutoff = "2024-06"

    public init() {}

    /// Calendar date embedded in the Harmony system message. Keeping the
    /// formatter here gives the CLI, server, and app one reproducible contract.
    public static func calendarDate(
        _ date: Date = Date(),
        timeZone: TimeZone = .current
    ) -> String {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = timeZone
        let components = calendar.dateComponents([.year, .month, .day], from: date)
        return String(
            format: "%04d-%02d-%02d",
            components.year ?? 0,
            components.month ?? 0,
            components.day ?? 0)
    }

    public func render(
        messages: [GFTokenizer.Message],
        tools: [GFTokenizer.FunctionDefinition] = [],
        reasoningEffort: GPTOSSReasoningEffort = .medium,
        currentDate: String,
        modelIdentity: String = Self.defaultModelIdentity,
        addGenerationPrompt: Bool = true
    ) throws -> String {
        guard currentDate.range(
            of: #"^\d{4}-\d{2}-\d{2}$"#,
            options: .regularExpression) != nil else {
            throw GFTokenizerError.invalidChatTemplate(
                "Harmony current date must use YYYY-MM-DD")
        }
        guard !modelIdentity.contains(Self.startMark),
              !modelIdentity.contains(Self.endMark) else {
            throw GFTokenizerError.invalidChatTemplate(
                "Harmony model identity contains reserved control tokens")
        }

        var output = Self.startMark + "system" + Self.messageMark
        output += modelIdentity + "\n"
        output += "Knowledge cutoff: \(Self.knowledgeCutoff)\n"
        output += "Current date: \(currentDate)\n\n"
        output += "Reasoning: \(reasoningEffort.rawValue)\n\n"
        output += "# Valid channels: analysis, commentary, final. "
            + "Channel must be included for every message."
        if !tools.isEmpty {
            output += "\nCalls to these tools must go to the commentary channel: 'functions'."
        }
        output += Self.endMark

        var conversation = messages[...]
        var instructions = ""
        if let first = messages.first,
           first.role == .system || first.role == .developer {
            guard let content = first.content else {
                throw GFTokenizerError.invalidChatTemplate(
                    "Harmony instruction message requires text content")
            }
            instructions = content
            conversation = messages.dropFirst()
        }
        guard !conversation.contains(where: {
            $0.role == .system || $0.role == .developer
        }) else {
            throw GFTokenizerError.invalidChatTemplate(
                "Harmony instruction message must be first")
        }

        if !instructions.isEmpty || !tools.isEmpty {
            output += Self.startMark + "developer" + Self.messageMark
            if !instructions.isEmpty {
                output += "# Instructions\n\n" + instructions + "\n\n"
            }
            if !tools.isEmpty {
                output += "# Tools\n\n"
                output += try renderToolNamespace(tools)
            }
            output += Self.endMark
        }

        var lastTool: (id: String, name: String)?
        for index in conversation.indices {
            let message = conversation[index]
            switch message.role {
            case .user:
                guard let content = message.content else {
                    throw GFTokenizerError.invalidChatTemplate(
                        "Harmony user message requires text content")
                }
                output += Self.startMark + "user" + Self.messageMark
                    + content + Self.endMark
                lastTool = nil

            case .assistant where !message.toolCalls.isEmpty:
                guard message.toolCalls.count == 1, let call = message.toolCalls.first else {
                    throw GFTokenizerError.invalidChatTemplate(
                        "Harmony supports one tool call per assistant message")
                }
                guard message.content == nil || message.thinking == nil else {
                    throw GFTokenizerError.invalidChatTemplate(
                        "Harmony tool call cannot contain both content and thinking")
                }
                try validateToolName(call.name)
                let laterFinal = conversation[conversation.index(after: index)...]
                    .contains { $0.role == .assistant && $0.toolCalls.isEmpty }
                if !laterFinal, let thinking = message.thinking ?? message.content,
                   !thinking.isEmpty {
                    output += Self.startMark + "assistant" + Self.channelMark
                        + "analysis" + Self.messageMark + thinking + Self.endMark
                }
                output += Self.startMark + "assistant to=functions.\(call.name)"
                    + Self.channelMark + "commentary json" + Self.messageMark
                    + (try call.arguments.encoded()) + Self.callMark
                lastTool = (call.id, call.name)

            case .assistant:
                guard let content = message.content else {
                    throw GFTokenizerError.invalidChatTemplate(
                        "Harmony assistant message requires text content")
                }
                if !addGenerationPrompt, index == conversation.indices.last,
                   let thinking = message.thinking, !thinking.isEmpty {
                    output += Self.startMark + "assistant" + Self.channelMark
                        + "analysis" + Self.messageMark + thinking + Self.endMark
                }
                output += Self.startMark + "assistant" + Self.channelMark
                    + "final" + Self.messageMark + content
                    + (!addGenerationPrompt && index == conversation.indices.last
                        ? Self.returnMark : Self.endMark)
                lastTool = nil

            case .tool:
                guard let prior = lastTool,
                      message.toolCallID == nil || message.toolCallID == prior.id,
                      let content = message.content else {
                    throw GFTokenizerError.invalidChatTemplate(
                        "Harmony tool result does not match the preceding call")
                }
                let quoted = try JSONValue.string(content).encoded()
                output += Self.startMark + "functions.\(prior.name) to=assistant"
                    + Self.channelMark + "commentary" + Self.messageMark
                    + quoted + Self.endMark
                lastTool = nil

            case .system, .developer:
                throw GFTokenizerError.invalidChatTemplate(
                    "Harmony instruction message must be first")
            }
        }

        if addGenerationPrompt {
            output += Self.startMark + "assistant"
        }
        return output
    }

    private func renderToolNamespace(
        _ tools: [GFTokenizer.FunctionDefinition]
    ) throws -> String {
        var result = "## functions\n\nnamespace functions {\n\n"
        for tool in tools {
            try validateToolName(tool.name)
            if !tool.description.isEmpty {
                result += "// \(tool.description)\n"
            }
            result += "type \(tool.name) = "
            guard case .object(let schema) = tool.parameters else {
                throw GFTokenizerError.invalidChatTemplate(
                    "Harmony tool parameters must be an object schema")
            }
            let properties = schema["properties"]?.objectValue ?? [:]
            let required: Set<String>
            if case .array(let values) = schema["required"] {
                required = Set(values.compactMap {
                    if case .string(let value) = $0 { return value }
                    return nil
                })
            } else {
                required = []
            }
            if properties.isEmpty {
                result += "() => any;\n\n"
                continue
            }
            result += "(_: {\n"
            for name in properties.keys.sorted() {
                guard let property = properties[name] else { continue }
                if case .object(let spec) = property,
                   case .string(let description) = spec["description"] {
                    result += "// \(description)\n"
                }
                result += name + (required.contains(name) ? "" : "?")
                    + ": " + typescriptType(property) + ",\n"
            }
            result += "}) => any;\n\n"
        }
        result += "} // namespace functions"
        return result
    }

    private func typescriptType(_ schema: JSONValue) -> String {
        guard case .object(let object) = schema else { return "any" }
        if case .array(let choices) = object["enum"] {
            let values = choices.compactMap { value -> String? in
                guard case .string(let text) = value else { return nil }
                return try? JSONValue.string(text).encoded()
            }
            if !values.isEmpty { return values.joined(separator: " | ") }
        }
        if case .array(let types) = object["type"] {
            let names = types.compactMap {
                if case .string(let value) = $0 { return value }
                return nil
            }
            if !names.isEmpty { return names.joined(separator: " | ") }
        }
        guard case .string(let type) = object["type"] else { return "any" }
        switch type {
        case "string": return "string" + nullableSuffix(object)
        case "number", "integer": return "number"
        case "boolean": return "boolean"
        case "array":
            guard let items = object["items"] else { return "any[]" }
            let inner = typescriptType(items)
            return (inner.contains(" | ") ? "(\(inner))" : inner) + "[]"
        case "object": return "object"
        default: return "any"
        }
    }

    private func nullableSuffix(_ object: [String: JSONValue]) -> String {
        if case .bool(true) = object["nullable"] { return " | null" }
        return ""
    }

    private func validateToolName(_ name: String) throws {
        guard name.range(
            of: #"^[A-Za-z0-9_-]{1,64}$"#,
            options: .regularExpression) != nil else {
            throw GFTokenizerError.invalidChatTemplate(
                "Harmony tool name is invalid")
        }
    }

    private static let startMark = "<|start|>"
    private static let endMark = "<|end|>"
    private static let returnMark = "<|return|>"
    private static let channelMark = "<|channel|>"
    private static let messageMark = "<|message|>"
    private static let callMark = "<|call|>"
}

/// Parses the JSON payload between a Harmony message header and `<|call|>`.
public struct HarmonyToolCallParser: Sendable {
    public static let maximumBytes = 256 * 1024

    public init() {}

    public func parse(
        recipient: String,
        argumentsJSON: String,
        allowedTools: Set<String>,
        id: String
    ) throws -> ParsedToolCall {
        guard argumentsJSON.utf8.count <= Self.maximumBytes else {
            throw ToolCallParserError.oversized
        }
        let prefix = "functions."
        guard recipient.hasPrefix(prefix) else {
            throw ToolCallParserError.malformed
        }
        let name = String(recipient.dropFirst(prefix.count))
        guard name.range(
            of: #"^[A-Za-z0-9_-]{1,64}$"#,
            options: .regularExpression) != nil else {
            throw ToolCallParserError.malformed
        }
        guard allowedTools.contains(name) else {
            throw ToolCallParserError.unknownTool(name)
        }
        let value: JSONValue
        do {
            value = try JSONDecoder().decode(
                JSONValue.self, from: Data(argumentsJSON.utf8))
        } catch {
            throw ToolCallParserError.malformed
        }
        guard value.objectValue != nil else {
            throw ToolCallParserError.malformed
        }
        return ParsedToolCall(
            id: id,
            name: name,
            arguments: value,
            argumentsJSON: try value.encoded())
    }
}

/// Token-driven streaming decoder for Harmony assistant messages. Analysis is
/// retained separately from visible content, so clients can hide, collapse, or
/// omit it without ever mistaking it for the final answer.
final class HarmonyAssistantDecoder: @unchecked Sendable {
    private enum State: Equatable {
        case header(expectsRole: Bool)
        case channel(recipient: String?)
        case content(channel: String, recipient: String?)
        case awaitingStart
        case completed
    }

    private let tokens: HarmonySpecialTokenIDs
    private let allowedTools: Set<String>
    private let idGenerator: @Sendable () -> String
    private var state: State = .header(expectsRole: false)
    private var buffer = ""
    private var emittedCalls = 0
    private var failed = false

    init(
        tokens: HarmonySpecialTokenIDs,
        allowedTools: Set<String>,
        idGenerator: @escaping @Sendable () -> String
    ) {
        self.tokens = tokens
        self.allowedTools = allowedTools
        self.idGenerator = idGenerator
    }

    func consume(tokenID: Int32, delta: String) throws -> [StructuredAssistantEvent] {
        guard !failed else { throw ToolCallParserError.malformed }
        let control = tokens.structuralMarkerIDs.contains(tokenID)
        var events: [StructuredAssistantEvent] = []
        if control, !delta.isEmpty {
            events += try append(delta)
        }

        do {
            if tokenID == tokens.start {
                guard state == .awaitingStart else { throw ToolCallParserError.malformed }
                state = .header(expectsRole: true)
                buffer = ""
                return events
            }
            if tokenID == tokens.channel {
                guard case .header(let expectsRole) = state else {
                    throw ToolCallParserError.malformed
                }
                let recipient = try parseHeader(buffer, expectsRole: expectsRole)
                state = .channel(recipient: recipient)
                buffer = ""
                return events
            }
            if tokenID == tokens.constrain {
                guard case .channel = state else { throw ToolCallParserError.malformed }
                return events
            }
            if tokenID == tokens.message {
                guard case .channel(let recipient) = state else {
                    throw ToolCallParserError.malformed
                }
                let channel = try parseChannel(buffer, recipient: recipient)
                state = .content(channel: channel, recipient: recipient)
                buffer = ""
                return events
            }
            if tokenID == tokens.end {
                guard case .content(_, let recipient) = state, recipient == nil else {
                    throw ToolCallParserError.malformed
                }
                state = .awaitingStart
                buffer = ""
                return events
            }
            if tokenID == tokens.return {
                guard case .content(let channel, let recipient) = state,
                      channel == "final", recipient == nil else {
                    throw ToolCallParserError.malformed
                }
                state = .completed
                buffer = ""
                return events
            }
            if tokenID == tokens.call {
                guard case .content(let channel, let recipient?) = state,
                      channel == "commentary" else {
                    throw ToolCallParserError.malformed
                }
                let call = try HarmonyToolCallParser().parse(
                    recipient: recipient,
                    argumentsJSON: buffer,
                    allowedTools: allowedTools,
                    id: idGenerator())
                emittedCalls += 1
                state = .completed
                buffer = ""
                return events + [.toolCall(call)]
            }
            if tokenID == tokens.endOfText {
                guard state == .completed || state == .awaitingStart else {
                    throw ToolCallParserError.malformed
                }
                state = .completed
                return events
            }
            return events + (try append(delta))
        } catch {
            failed = true
            throw error
        }
    }

    func consumeTail(_ text: String) throws -> [StructuredAssistantEvent] {
        guard !failed else { throw ToolCallParserError.malformed }
        return try append(text)
    }

    func finish() throws {
        guard !failed else { throw ToolCallParserError.malformed }
        switch state {
        case .content(let channel, let recipient):
            guard recipient == nil, channel == "analysis" || channel == "final" else {
                throw ToolCallParserError.malformed
            }
        case .awaitingStart, .completed:
            break
        case .header, .channel:
            throw ToolCallParserError.malformed
        }
    }

    var hasToolCalls: Bool { emittedCalls > 0 }

    private func append(_ text: String) throws -> [StructuredAssistantEvent] {
        guard !text.isEmpty else { return [] }
        switch state {
        case .header, .channel:
            buffer += text
            guard buffer.utf8.count <= 4_096 else {
                throw ToolCallParserError.oversized
            }
            return []
        case .content(let channel, let recipient):
            if recipient != nil {
                buffer += text
                guard buffer.utf8.count <= HarmonyToolCallParser.maximumBytes else {
                    throw ToolCallParserError.oversized
                }
                return []
            }
            return channel == "final" ? [.content(text)] : [.thinking(text)]
        case .awaitingStart, .completed:
            guard text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw ToolCallParserError.malformed
            }
            return []
        }
    }

    private func parseHeader(_ text: String, expectsRole: Bool) throws -> String? {
        var header = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if expectsRole {
            guard header.hasPrefix("assistant") else {
                throw ToolCallParserError.malformed
            }
            header.removeFirst("assistant".count)
            header = header.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard !header.isEmpty else { return nil }
        let prefix = "to="
        guard header.hasPrefix(prefix) else { throw ToolCallParserError.malformed }
        let recipient = String(header.dropFirst(prefix.count))
        guard !recipient.isEmpty, !recipient.contains(where: \.isWhitespace) else {
            throw ToolCallParserError.malformed
        }
        return recipient
    }

    private func parseChannel(_ text: String, recipient: String?) throws -> String {
        let components = text.split(whereSeparator: \.isWhitespace).map(String.init)
        guard let channel = components.first else { throw ToolCallParserError.malformed }
        if recipient == nil {
            guard components.count == 1, channel == "analysis" || channel == "final" else {
                throw ToolCallParserError.malformed
            }
        } else {
            guard channel == "commentary",
                  components.count == 1 || components == ["commentary", "json"] else {
                throw ToolCallParserError.malformed
            }
        }
        return channel
    }
}
