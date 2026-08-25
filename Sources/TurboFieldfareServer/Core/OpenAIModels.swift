import Foundation
import TurboFieldfare

public struct OpenAIErrorEnvelope: Codable, Equatable, Sendable {
    public struct Detail: Codable, Equatable, Sendable {
        public let message: String
        public let type: String
        public let param: String?
        public let code: String
    }

    public let error: Detail

    public init(message: String, param: String? = nil, code: String,
                type: String = "invalid_request_error") {
        error = Detail(message: message,
                       type: type,
                       param: param,
                       code: code)
    }
}

public struct OpenAIImageURL: Codable, Equatable, Sendable {
    public let url: String
    public let detail: String?
}

public struct OpenAIContentPart: Codable, Equatable, Sendable {
    public let type: String
    public let text: String?
    public let imageURL: OpenAIImageURL?

    enum CodingKeys: String, CodingKey {
        case type, text
        case imageURL = "image_url"
    }
}

public typealias OpenAITextPart = OpenAIContentPart

public enum OpenAIMessageContent: Codable, Equatable, Sendable {
    case text(String)
    case parts([OpenAIContentPart])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let text = try? container.decode(String.self) {
            self = .text(text)
        } else {
            self = .parts(try container.decode([OpenAIContentPart].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .text(let text): try container.encode(text)
        case .parts(let parts): try container.encode(parts)
        }
    }

}

public struct OpenAIFunctionCall: Codable, Equatable, Sendable {
    public let name: String
    public let arguments: String
}

public struct OpenAIToolCall: Codable, Equatable, Sendable {
    public let id: String
    public let type: String
    public let function: OpenAIFunctionCall
}

public struct OpenAIChatMessage: Codable, Equatable, Sendable {
    public let role: String
    public let content: OpenAIMessageContent?
    public let toolCalls: [OpenAIToolCall]?
    public let toolCallID: String?
    public let name: String?

    enum CodingKeys: String, CodingKey {
        case role, content, name
        case toolCalls = "tool_calls"
        case toolCallID = "tool_call_id"
    }
}

public struct OpenAIFunctionDefinition: Codable, Equatable, Sendable {
    public let name: String
    public let description: String?
    public let parameters: JSONValue
}

public struct OpenAITool: Codable, Equatable, Sendable {
    public let type: String
    public let function: OpenAIFunctionDefinition
}

public enum OpenAIStop: Codable, Equatable, Sendable {
    case one(String)
    case many([String])

    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let one = try? container.decode(String.self) {
            self = .one(one)
        } else {
            self = .many(try container.decode([String].self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .one(let value): try container.encode(value)
        case .many(let value): try container.encode(value)
        }
    }

    var values: [String] {
        switch self {
        case .one(let value): [value]
        case .many(let value): value
        }
    }
}

public struct OpenAIStreamOptions: Codable, Equatable, Sendable {
    public let includeUsage: Bool?

    enum CodingKeys: String, CodingKey {
        case includeUsage = "include_usage"
    }
}

public struct OpenAIChatRequest: Codable, Equatable, Sendable {
    public let model: String
    public let messages: [OpenAIChatMessage]
    public let stream: Bool?
    public let streamOptions: OpenAIStreamOptions?
    public let temperature: Float?
    public let topP: Float?
    public let maxTokens: Int?
    public let maxCompletionTokens: Int?
    public let stop: OpenAIStop?
    public let seed: UInt64?
    public let tools: [OpenAITool]?
    public let toolChoice: JSONValue?
    public let parallelToolCalls: Bool?
    public let topK: Int?
    public let repetitionPenalty: Float?
    public let n: Int?
    public let logprobs: Bool?
    public let presencePenalty: Float?
    public let frequencyPenalty: Float?
    /// TUFF's model-aware on/off reasoning control for Gemma 4 and Qwen.
    public let enableThinking: Bool?
    /// GPT-OSS graded reasoning control. Harmony defaults to Medium when this
    /// field is absent; other model families reject it.
    public let reasoningEffort: GPTOSSReasoningEffort?

    enum CodingKeys: String, CodingKey {
        case model, messages, stream, temperature, stop, seed, tools, n, logprobs
        case streamOptions = "stream_options"
        case topP = "top_p"
        case maxTokens = "max_tokens"
        case maxCompletionTokens = "max_completion_tokens"
        case toolChoice = "tool_choice"
        case parallelToolCalls = "parallel_tool_calls"
        case topK = "top_k"
        case repetitionPenalty = "repetition_penalty"
        case presencePenalty = "presence_penalty"
        case frequencyPenalty = "frequency_penalty"
        case enableThinking = "enable_thinking"
        case reasoningEffort = "reasoning_effort"
    }
}

public struct OpenAIUsage: Codable, Equatable, Sendable {
    public struct PromptTokensDetails: Codable, Equatable, Sendable {
        public let cachedTokens: Int

        enum CodingKeys: String, CodingKey {
            case cachedTokens = "cached_tokens"
        }

        public init(cachedTokens: Int) {
            self.cachedTokens = cachedTokens
        }
    }

    public let promptTokens: Int
    public let completionTokens: Int
    public let totalTokens: Int
    public let promptTokensDetails: PromptTokensDetails

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case completionTokens = "completion_tokens"
        case totalTokens = "total_tokens"
        case promptTokensDetails = "prompt_tokens_details"
    }

    public init(promptTokens: Int,
                completionTokens: Int,
                totalTokens: Int,
                cachedTokens: Int = 0) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = totalTokens
        self.promptTokensDetails = PromptTokensDetails(cachedTokens: cachedTokens)
    }
}

public struct OpenAIModelList: Codable, Equatable, Sendable {
    public struct Model: Codable, Equatable, Sendable {
        public let id: String
        public let object: String
        public let created: Int
        public let ownedBy: String
        public let capabilities: [String]?

        enum CodingKeys: String, CodingKey {
            case id, object, created, capabilities
            case ownedBy = "owned_by"
        }

        public init(id: String, object: String, created: Int,
                    ownedBy: String, capabilities: [String]? = nil) {
            self.id = id
            self.object = object
            self.created = created
            self.ownedBy = ownedBy
            self.capabilities = capabilities
        }
    }

    public let object: String
    public let data: [Model]
}

public enum ServerRequestError: Error, Equatable, Sendable {
    case invalid(message: String, param: String?, code: String)
    case unknownModel
    case queueFull

    public var envelope: OpenAIErrorEnvelope {
        switch self {
        case .invalid(let message, let param, let code):
            OpenAIErrorEnvelope(message: message, param: param, code: code)
        case .unknownModel:
            OpenAIErrorEnvelope(message: "requested model is not available",
                                param: "model", code: "model_not_found")
        case .queueFull:
            OpenAIErrorEnvelope(message: "generation queue is full",
                                code: "queue_full")
        }
    }
}

public struct ValidatedChatRequest: Sendable {
    public let messages: [GFTokenizer.Message]
    public let multimodalMessages: [MultimodalMessage]?
    public let imageFiles: [UUID: URL]
    /// Content SHA-256 of each message's images, in order, aligned with
    /// `messages`. Staged image UUIDs are fresh per request, so they cannot
    /// identify an image across turns; the content hash can. Empty for
    /// text-only requests.
    public let imageIdentities: [[String]]
    public let tools: [GFTokenizer.FunctionDefinition]
    public let stream: Bool
    public let includeUsage: Bool
    public let generationConfig: GenerationConfig
    public let maximumCompletionTokens: Int
    public let reasoning: ChatReasoning
    public let reasoningEffort: GPTOSSReasoningEffort?
    /// The local calendar day rendered into Harmony's system message. It is
    /// captured during validation so queued requests and cache comparisons do
    /// not change identity across midnight.
    public let harmonyCurrentDate: String?
    /// Every staging directory this request's image files live in. The parser
    /// and the validator's store each stage under their own lease, and a
    /// request may carry files from both, so dropping either would delete
    /// files the other path staged before `generate` reads them.
    fileprivate let attachmentLeases: [ServerAttachmentLease]

    public init(
        messages: [GFTokenizer.Message],
        multimodalMessages: [MultimodalMessage]? = nil,
        imageFiles: [UUID: URL] = [:],
        imageIdentities: [[String]] = [],
        tools: [GFTokenizer.FunctionDefinition],
        stream: Bool,
        includeUsage: Bool,
        generationConfig: GenerationConfig,
        maximumCompletionTokens: Int,
        reasoning: ChatReasoning = .off,
        reasoningEffort: GPTOSSReasoningEffort? = nil,
        harmonyCurrentDate: String? = nil
    ) {
        self.messages = messages
        self.multimodalMessages = multimodalMessages
        self.imageFiles = imageFiles
        self.imageIdentities = imageIdentities
        self.tools = tools
        self.stream = stream
        self.includeUsage = includeUsage
        self.generationConfig = generationConfig
        self.maximumCompletionTokens = maximumCompletionTokens
        self.reasoning = reasoning
        self.reasoningEffort = reasoningEffort
        self.harmonyCurrentDate = harmonyCurrentDate
        self.attachmentLeases = []
    }

    fileprivate init(
        messages: [GFTokenizer.Message],
        multimodalMessages: [MultimodalMessage]?,
        imageFiles: [UUID: URL],
        imageIdentities: [[String]],
        tools: [GFTokenizer.FunctionDefinition],
        stream: Bool,
        includeUsage: Bool,
        generationConfig: GenerationConfig,
        maximumCompletionTokens: Int,
        reasoning: ChatReasoning,
        reasoningEffort: GPTOSSReasoningEffort?,
        harmonyCurrentDate: String?,
        attachmentLeases: [ServerAttachmentLease]
    ) {
        self.messages = messages
        self.multimodalMessages = multimodalMessages
        self.imageFiles = imageFiles
        self.imageIdentities = imageIdentities
        self.tools = tools
        self.stream = stream
        self.includeUsage = includeUsage
        self.generationConfig = generationConfig
        self.maximumCompletionTokens = maximumCompletionTokens
        self.reasoning = reasoning
        self.reasoningEffort = reasoningEffort
        self.harmonyCurrentDate = harmonyCurrentDate
        self.attachmentLeases = attachmentLeases
    }
}

private enum OpenAIToolName {
    static let maximumLength = 64

    static func isValid(_ name: String) -> Bool {
        let bytes = name.utf8
        guard !bytes.isEmpty, bytes.count <= maximumLength else { return false }
        return bytes.allSatisfy { byte in
            switch byte {
            case 45, 48...57, 65...90, 95, 97...122:
                true
            default:
                false
            }
        }
    }

    static func validationMessage(for name: String) -> String {
        let prefix = name.prefix(maximumLength + 1)
        let displayed = String(prefix.prefix(maximumLength))
            + (prefix.count > maximumLength ? "..." : "")
        return "tool name \(String(reflecting: displayed)) must contain 1 to 64 ASCII letters, numbers, underscores, or hyphens"
    }
}

public enum OpenAIRequestValidator {
    public static func validate(_ request: OpenAIChatRequest,
                                modelID: String,
                                dialect: ChatDialect = .gemma) throws -> ValidatedChatRequest {
        try validate(
            request,
            modelID: modelID,
            dialect: dialect,
            preStagedImages: [:],
            attachmentLease: nil)
    }

    static func validate(_ request: OpenAIChatRequest,
                         modelID: String,
                         dialect: ChatDialect = .gemma,
                         preStagedImages: [String: ServerStagedImage],
                         attachmentLease: ServerAttachmentLease?) throws -> ValidatedChatRequest {
        guard request.model == modelID else { throw ServerRequestError.unknownModel }
        guard request.n == nil || request.n == 1 else {
            throw invalid("only n=1 is supported", "n", "unsupported_value")
        }
        guard request.logprobs != true else {
            throw invalid("logprobs are not supported", "logprobs", "unsupported_value")
        }
        guard request.presencePenalty == nil || request.presencePenalty == 0 else {
            throw invalid("presence_penalty must be zero", "presence_penalty", "unsupported_value")
        }
        guard request.frequencyPenalty == nil || request.frequencyPenalty == 0 else {
            throw invalid("frequency_penalty must be zero", "frequency_penalty", "unsupported_value")
        }
        guard request.parallelToolCalls != false else {
            throw invalid("parallel_tool_calls=false is not supported",
                          "parallel_tool_calls", "unsupported_value")
        }

        let temperature = request.temperature ?? 0.2
        guard temperature >= 0, temperature <= 2 else {
            throw invalid("temperature must be between 0 and 2",
                          "temperature", "invalid_value")
        }
        let topP = request.topP ?? 0.95
        guard topP > 0, topP <= 1 else {
            throw invalid("top_p must be greater than 0 and at most 1",
                          "top_p", "invalid_value")
        }
        let topK = request.topK ?? 64
        guard (1...256).contains(topK) else {
            throw invalid("top_k must be between 1 and 256", "top_k", "invalid_value")
        }
        let repetitionPenalty = request.repetitionPenalty ?? 1
        guard repetitionPenalty > 0 else {
            throw invalid("repetition_penalty must be positive",
                          "repetition_penalty", "invalid_value")
        }
        let maximum = request.maxCompletionTokens ?? request.maxTokens ?? 4096
        guard maximum > 0 else {
            throw invalid("maximum completion tokens must be positive",
                          request.maxCompletionTokens != nil ? "max_completion_tokens" : "max_tokens",
                          "invalid_value")
        }
        let reasoning: ChatReasoning
        let reasoningEffort: GPTOSSReasoningEffort?
        let harmonyCurrentDate: String?
        if dialect == .harmony {
            guard request.enableThinking == nil else {
                throw invalid(
                    "enable_thinking is not supported by GPT-OSS; use reasoning_effort",
                    "enable_thinking", "unsupported_parameter")
            }
            reasoning = .off
            reasoningEffort = request.reasoningEffort ?? .medium
            harmonyCurrentDate = HarmonyPromptRenderer.calendarDate()
        } else {
            guard request.reasoningEffort == nil else {
                throw invalid(
                    "reasoning_effort is only supported by GPT-OSS",
                    "reasoning_effort", "unsupported_parameter")
            }
            reasoning = request.enableThinking == true ? .on : .off
            reasoningEffort = nil
            harmonyCurrentDate = nil
        }

        let includeTools: Bool
        switch request.toolChoice {
        case nil, .some(.string("auto")):
            includeTools = true
        case .some(.string("none")):
            includeTools = false
        case .some(.string("required")):
            throw invalid("tool_choice=required is not supported",
                          "tool_choice", "unsupported_value")
        default:
            throw invalid("named tool choices are not supported",
                          "tool_choice", "unsupported_value")
        }

        let tools = try (includeTools ? request.tools ?? [] : []).map {
            try validateTool($0, dialect: dialect)
        }
        let validatedMessages = try validateMessages(
            request.messages,
            dialect: dialect,
            preStagedImages: preStagedImages,
            attachmentLease: attachmentLease)
        let config = GenerationConfig(maxNewTokens: maximum,
                                      temperature: temperature,
                                      topK: topK,
                                      topP: topP,
                                      repetitionPenalty: repetitionPenalty,
                                      seed: request.seed,
                                      stopStrings: request.stop?.values ?? [])
        return ValidatedChatRequest(messages: validatedMessages.messages,
                                    multimodalMessages: validatedMessages.multimodal,
                                    imageFiles: validatedMessages.imageFiles,
                                    imageIdentities: validatedMessages.imageIdentities,
                                    tools: tools,
                                    stream: request.stream ?? false,
                                    includeUsage: request.streamOptions?.includeUsage ?? false,
                                    generationConfig: config,
                                    maximumCompletionTokens: maximum,
                                    reasoning: reasoning,
                                    reasoningEffort: reasoningEffort,
                                    harmonyCurrentDate: harmonyCurrentDate,
                                    attachmentLeases: validatedMessages.leases)
    }

    private static func validateTool(_ tool: OpenAITool,
                                     dialect: ChatDialect) throws -> GFTokenizer.FunctionDefinition {
        guard tool.type == "function" else {
            throw invalid("only function tools are supported", "tools", "unsupported_tool")
        }
        let name = tool.function.name
        guard OpenAIToolName.isValid(name) else {
            throw invalid(OpenAIToolName.validationMessage(for: name),
                          "tools", "invalid_tool_name")
        }
        guard tool.function.parameters.objectValue != nil else {
            throw invalid("tool parameters must be an object schema",
                          "tools", "invalid_tool_schema")
        }
        try validateSchemaKeys(tool.function.parameters, dialect: dialect)
        // The Gemma template can only render a restricted schema subset, so it
        // needs the adaptation pass. ChatML and Harmony preserve the schema.
        let parameters = dialect == .gemma
            ? try GemmaToolSchema.adapted(tool.function.parameters, toolName: name)
            : tool.function.parameters
        guard (try? parameters.jinjaSendableValue()) != nil else {
            throw invalid("tool schema contains a number that cannot be represented exactly",
                          "tools", "invalid_tool_schema")
        }
        return GFTokenizer.FunctionDefinition(name: name,
                                              description: tool.function.description ?? "",
                                              parameters: parameters)
    }

    private static func validateSchemaKeys(_ schema: JSONValue,
                                           dialect: ChatDialect) throws {
        switch schema {
        case .object(let object):
            for (schemaKey, value) in object {
                if schemaKey == "properties" {
                    guard case .object(let definitions) = value else {
                        throw invalid("tool schema properties must be an object",
                                      "tools", "invalid_tool_schema")
                    }
                    for (key, definition) in definitions {
                        // Gemma's tool-call DSL cannot round-trip arbitrary
                        // parameter names; ChatML and Harmony calls are JSON.
                        guard dialect != .gemma
                                || GemmaToolCallParser.isRepresentableObjectKey(key) else {
                            throw invalid(
                                "tool parameter names may contain only letters, numbers, _, -, ., and $",
                                "tools",
                                "invalid_tool_schema")
                        }
                        try validateSchemaKeys(definition, dialect: dialect)
                    }
                } else {
                    try validateSchemaKeys(value, dialect: dialect)
                }
            }
        case .array(let values):
            for value in values {
                try validateSchemaKeys(value, dialect: dialect)
            }
        default:
            break
        }
    }

    private struct ValidatedMessages {
        let messages: [GFTokenizer.Message]
        let multimodal: [MultimodalMessage]?
        let imageFiles: [UUID: URL]
        let imageIdentities: [[String]]
        let leases: [ServerAttachmentLease]
    }

    private static func validateMessages(
        _ input: [OpenAIChatMessage],
        dialect: ChatDialect,
        preStagedImages: [String: ServerStagedImage],
        attachmentLease: ServerAttachmentLease?
    ) throws -> ValidatedMessages {
        guard !input.isEmpty else {
            throw invalid("messages must not be empty", "messages", "invalid_message")
        }
        var knownCalls: [String: (name: String, resolved: Bool)] = [:]
        var result: [GFTokenizer.Message] = []
        var multimodal: [MultimodalMessage] = []
        var imageFiles: [UUID: URL] = [:]
        var imageIdentities: [[String]] = []
        var messageIdentities: [String] = []
        var store: ServerAttachmentStore?
        var sawConversationMessage = false
        for message in input {
            guard let role = GFTokenizer.Role(rawValue: message.role) else {
                throw invalid("unsupported message role \(message.role)",
                              "messages", "invalid_message")
            }
            if role == .system || role == .developer {
                guard !sawConversationMessage else {
                    throw invalid("system or developer guidance must precede the conversation",
                                  "messages", "invalid_message")
                }
            } else {
                sawConversationMessage = true
            }
            var orderedContent: [MultimodalContentPart] = []
            let content: String?
            switch message.content {
            case nil:
                content = nil
            case .text(let text):
                content = text
                orderedContent = [.text(text)]
            case .parts(let parts):
                var joined = ""
                for part in parts {
                    switch part.type {
                    case "text":
                        guard let text = part.text else {
                            throw invalid("text content part requires text",
                                          "messages", "invalid_message")
                        }
                        joined += text
                        orderedContent.append(.text(text))
                    case "image_url":
                        guard role == .user else {
                            throw invalid("image_url is supported only in user messages",
                                          "messages", "unsupported_content")
                        }
                        guard let image = part.imageURL else {
                            throw invalid("image_url content part requires an image_url object",
                                          "messages", "invalid_message")
                        }
                        guard image.detail == nil || image.detail == "auto" else {
                            throw invalid("image detail must be absent or auto",
                                          "messages", "unsupported_value")
                        }
                        let staged: ServerStagedImage
                        let prefix = "turbofieldfare-attachment:"
                        if image.url.hasPrefix(prefix) {
                            let token = String(image.url.dropFirst(prefix.count))
                            guard let existing = preStagedImages[token] else {
                                throw invalid("image attachment lease is missing",
                                              "messages", "invalid_image")
                            }
                            staged = existing
                        } else {
                            if store == nil { store = try ServerAttachmentStore() }
                            staged = try store!.stage(dataURL: image.url)
                        }
                        imageFiles[staged.id] = staged.fileURL
                        messageIdentities.append(staged.sha256)
                        orderedContent.append(.image(id: staged.id))
                    default:
                        throw invalid("unsupported content part \(part.type)",
                                      "messages", "unsupported_content")
                    }
                }
                content = joined
            }
            // The semantic bound on images is the context budget, checked
            // against the model's actual context in `ServerModelSession.prepare`
            // where the per-image token cost is known. Staging enforces its own
            // per-file, per-request, and count resource caps; no further count
            // check belongs here.

            let calls: [GFTokenizer.HistoricalToolCall] = try (message.toolCalls ?? []).map { call in
                guard role == .assistant, call.type == "function",
                      !call.id.isEmpty, knownCalls[call.id] == nil else {
                    throw invalid("invalid or duplicate historical tool call",
                                  "messages", "invalid_tool_call")
                }
                guard OpenAIToolName.isValid(call.function.name) else {
                    throw invalid(OpenAIToolName.validationMessage(for: call.function.name),
                                  "messages", "invalid_tool_call")
                }
                let data = Data(call.function.arguments.utf8)
                let arguments = try JSONDecoder().decode(JSONValue.self, from: data)
                guard arguments.objectValue != nil else {
                    throw invalid("historical tool arguments must be a JSON object",
                                  "messages", "invalid_tool_arguments")
                }
                guard dialect == .chatml
                        || (try? arguments.gemmaToolArgumentBody()) != nil,
                      (try? arguments.jinjaSendableValue()) != nil else {
                    throw invalid(
                        "historical tool arguments cannot be represented exactly: "
                            + "unsupported value",
                        "messages",
                        "invalid_tool_arguments")
                }
                knownCalls[call.id] = (call.function.name, false)
                return GFTokenizer.HistoricalToolCall(
                    id: call.id, name: call.function.name, arguments: arguments)
            }
            if role == .tool {
                guard let id = message.toolCallID,
                      let call = knownCalls[id], !call.resolved else {
                    throw invalid("tool result must reference one unresolved call",
                                  "messages", "invalid_tool_result")
                }
                knownCalls[id] = (call.name, true)
                guard content != nil else {
                    throw invalid("tool result content is required",
                                  "messages", "invalid_tool_result")
                }
            } else if content == nil && calls.isEmpty {
                throw invalid("message content is required",
                              "messages", "invalid_message")
            }
            result.append(GFTokenizer.Message(role: role,
                                              content: content,
                                              toolCalls: calls,
                                              toolCallID: message.toolCallID,
                                              name: message.name))
            if orderedContent.isEmpty, let content {
                orderedContent = [.text(content)]
            }
            multimodal.append(MultimodalMessage(
                role: role,
                content: orderedContent,
                toolCalls: calls,
                toolCallID: message.toolCallID,
                name: message.name))
            imageIdentities.append(messageIdentities)
            messageIdentities = []
        }
        return ValidatedMessages(
            messages: result,
            multimodal: imageFiles.isEmpty ? nil : multimodal,
            imageFiles: imageFiles,
            imageIdentities: imageFiles.isEmpty ? [] : imageIdentities,
            leases: [attachmentLease, store?.lease].compactMap { $0 })
    }

    private static func invalid(_ message: String,
                                _ param: String?,
                                _ code: String) -> ServerRequestError {
        .invalid(message: message, param: param, code: code)
    }
}
