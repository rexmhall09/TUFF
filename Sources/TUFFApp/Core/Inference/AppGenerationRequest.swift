import Foundation
import TUFFEngine

public struct AppGenerationRequest: Equatable, Sendable {
    public var modelDirectory: URL
    public var prompt: String
    /// The system prompt, rendered as a system message ahead of the whole
    /// conversation. Empty means none.
    public var systemPrompt: String
    /// Text the model should treat as the beginning of its own reply, appended
    /// after the generation prompt so decoding carries it on. Empty means a
    /// fresh answer. Used by Continue.
    public var assistantPrefix: String
    /// Completed turns preceding `prompt`, oldest first. Empty is the
    /// single-turn behavior this type had before.
    public var history: [AppChatTurn]
    /// Fully structured messages used by the app-hosted OpenAI-compatible
    /// server. Nil keeps the simpler Chat conversation path above.
    public var structuredMessages: [GFTokenizer.Message]?
    public var multimodalMessages: [MultimodalMessage]?
    public var tools: [GFTokenizer.FunctionDefinition]
    public var imageAttachments: [AppImageAttachment]
    public var maxNewTokens: Int
    public var maxContextTokens: Int
    public var reasoning: ChatReasoning
    public var reasoningEffort: GPTOSSReasoningEffort?
    public var preserveThinking: Bool
    public var temperature: Float
    public var topK: Int?
    public var topP: Float?
    public var repetitionPenalty: Float
    public var seed: UInt64?
    public var stopStrings: [String]
    public var harmonyCurrentDate: String?
    public var runtimeOptions: AppRuntimeOptions

    public init(modelDirectory: URL,
                prompt: String,
                systemPrompt: String = "",
                assistantPrefix: String = "",
                history: [AppChatTurn] = [],
                structuredMessages: [GFTokenizer.Message]? = nil,
                multimodalMessages: [MultimodalMessage]? = nil,
                tools: [GFTokenizer.FunctionDefinition] = [],
                imageAttachments: [AppImageAttachment] = [],
                maxNewTokens: Int = 4_096,
                maxContextTokens: Int = 4096,
                reasoning: ChatReasoning = .off,
                reasoningEffort: GPTOSSReasoningEffort? = nil,
                preserveThinking: Bool = false,
                temperature: Float = 0.2,
                topK: Int? = 64,
                topP: Float? = 0.95,
                repetitionPenalty: Float = 1.0,
                seed: UInt64? = nil,
                stopStrings: [String] = [],
                harmonyCurrentDate: String? = nil,
                runtimeOptions: AppRuntimeOptions = AppRuntimeOptions()) {
        self.modelDirectory = modelDirectory
        self.prompt = prompt
        self.systemPrompt = systemPrompt
        self.assistantPrefix = assistantPrefix
        self.history = history
        self.structuredMessages = structuredMessages
        self.multimodalMessages = multimodalMessages
        self.tools = tools
        self.imageAttachments = imageAttachments
        self.maxNewTokens = maxNewTokens
        self.maxContextTokens = maxContextTokens
        self.reasoning = reasoning
        self.reasoningEffort = reasoningEffort
        self.preserveThinking = preserveThinking
        self.temperature = temperature
        self.topK = topK
        self.topP = topP
        self.repetitionPenalty = repetitionPenalty
        self.seed = seed
        self.stopStrings = stopStrings
        self.harmonyCurrentDate = harmonyCurrentDate
        self.runtimeOptions = runtimeOptions
    }

    public var isPureGreedy: Bool {
        temperature == 0 && repetitionPenalty == 1
    }

    public func validate(fileManager: FileManager = .default,
                         requireModelDirectory: Bool = true) throws {
        guard structuredMessages?.isEmpty == false
                || !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                || !imageAttachments.isEmpty else {
            throw AppInferenceError.invalidRequest("Prompt or image cannot be empty.")
        }
        // The same context-derived rule the server uses. A fixed four here
        // meant a request the API accepted was refused in the app.
        guard Set(imageAttachments.map(\.id)).count == imageAttachments.count else {
            throw AppInferenceError.invalidRequest("Images must be distinct.")
        }
        if let multimodalMessages {
            let referenced = Set(multimodalMessages.flatMap { message in
                message.content.compactMap { part -> UUID? in
                    if case .image(let id) = part { return id }
                    return nil
                }
            })
            guard referenced == Set(imageAttachments.map(\.id)) else {
                throw AppInferenceError.invalidRequest(
                    "Structured image references do not match staged attachments.")
            }
        } else if !imageAttachments.isEmpty, structuredMessages != nil {
            throw AppInferenceError.invalidRequest(
                "Structured image requests require multimodal messages.")
        }
        let family = (try? ManifestReader.resolveArchitecture(
            directoryURL: modelDirectory).family)
            ?? .gemma4
        try Self.validateReasoning(
            family: family,
            reasoning: reasoning,
            reasoningEffort: reasoningEffort)
        try Self.validateImageCapacity(
            imageCount: imageAttachments.count,
            maxContextTokens: maxContextTokens,
            family: family)
        for attachment in imageAttachments {
            guard attachment.fileURL.isFileURL,
                  attachment.encodedBytes >= 0,
                  attachment.encodedBytes <= VisionImageLimits().maximumEncodedBytes,
                  attachment.sha256.count == 64,
                  attachment.sha256.unicodeScalars.allSatisfy({
                      (48...57).contains($0.value) || (97...102).contains($0.value)
                  }) else {
                throw AppInferenceError.invalidRequest(
                    "Invalid image attachment \(attachment.displayName).")
            }
        }
        guard maxNewTokens > 0 else {
            throw AppInferenceError.invalidRequest("Max response length must be greater than zero.")
        }
        guard maxContextTokens > 0 else {
            throw AppInferenceError.invalidRequest("Max context must be greater than zero.")
        }
        guard temperature >= 0 else {
            throw AppInferenceError.invalidRequest("Temperature cannot be negative.")
        }
        if let topK {
            guard (1...256).contains(topK) else {
                throw AppInferenceError.invalidRequest("Top-K must be between 1 and 256.")
            }
        }
        if let topP {
            guard topP > 0, topP <= 1 else {
                throw AppInferenceError.invalidRequest("Top-P must be greater than 0 and at most 1.")
            }
            if temperature > 0, topP < 1, topK == nil {
                throw AppInferenceError.invalidRequest(
                    "Top-P below 1 requires Top-K to be enabled.")
            }
        }
        if structuredMessages == nil {
            guard repetitionPenalty >= 1 else {
                throw AppInferenceError.invalidRequest(
                    "Repetition penalty must be at least 1.")
            }
        } else {
            guard repetitionPenalty > 0 else {
                throw AppInferenceError.invalidRequest(
                    "Repetition penalty must be positive.")
            }
        }
        guard stopStrings.allSatisfy({ !$0.isEmpty }) else {
            throw AppInferenceError.invalidRequest("Stop strings cannot be empty.")
        }
        try runtimeOptions.validate()

        if requireModelDirectory {
            var isDirectory = ObjCBool(false)
            guard fileManager.fileExists(atPath: modelDirectory.path, isDirectory: &isDirectory),
                  isDirectory.boolValue else {
                throw AppInferenceError.modelNotFound(modelDirectory.path)
            }
        }
    }

    static func validateReasoning(
        family: ModelFamily,
        reasoning: ChatReasoning,
        reasoningEffort: GPTOSSReasoningEffort?
    ) throws {
        if family == .gptOss {
            guard reasoning == .off else {
                throw AppInferenceError.invalidRequest(
                    "GPT-OSS uses graded reasoning effort, not an on/off switch.")
            }
        } else if reasoningEffort != nil {
            throw AppInferenceError.invalidRequest(
                "Reasoning effort is only supported by GPT-OSS.")
        }
    }

    static func validateImageCapacity(
        imageCount: Int,
        maxContextTokens: Int,
        family: ModelFamily
    ) throws {
        guard imageCount > 0 else { return }
        guard family != .gptOss else {
            throw AppInferenceError.invalidRequest(
                "GPT-OSS does not support image input.")
        }
        let maximumTokensPerImage = VisionImageTokenBudget.maximumTokensPerImage(
            family: family)
        let capacity = VisionImageTokenBudget.capacity(
            maxContext: maxContextTokens,
            reservedTextTokens: 0,
            family: family)
        guard imageCount <= capacity else {
            throw AppInferenceError.invalidRequest(
                "\(imageCount) images need up to "
                    + "\(imageCount * maximumTokensPerImage) "
                    + "tokens, beyond the \(maxContextTokens)-token context.")
        }
    }
}
