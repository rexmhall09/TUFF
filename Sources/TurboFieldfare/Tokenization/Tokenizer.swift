import Foundation
import Hub
import Tokenizers

public enum GFTokenizerError: Error, CustomStringConvertible {
    case missingSpecialToken(String)
    case invalidChatTemplate(String)
    case missingToolTemplate
    case missingTokenizerConfig
    case unsupportedDecoder(actual: String)
    case invalidTokenID(token: String, id: Int)
    case unsupportedForDialect(String)
    case specialTokenMismatch(token: String, expected: Int32, resolved: Int32)

    public var description: String {
        switch self {
        case .missingSpecialToken(let t): return "tokenizer missing required special token: \(t)"
        case .invalidChatTemplate(let detail): return "invalid chat messages: \(detail)"
        case .missingToolTemplate:
            return "installed tokenizer is missing chat_template.jinja; reinstall the model"
        case .missingTokenizerConfig:
            return "tokenizer_config.json is missing or unreadable"
        case .unsupportedDecoder(let actual):
            return "tokenizer decoder is not the pinned Gemma 4 sequence "
                + "Sequence[Replace(▁→␣), ByteFallback, Fuse]; found: \(actual)"
        case .invalidTokenID(let token, let id):
            return "tokenizer declares out-of-range ID \(id) for token \(token)"
        case .unsupportedForDialect(let operation):
            return "operation is not supported for this tokenizer's chat dialect: \(operation)"
        case .specialTokenMismatch(let token, let expected, let resolved):
            return "tokenizer resolves \(token) to \(resolved), but the runtime "
                + "expects \(expected); the tokenizer does not match this build"
        }
    }
}

/// Chat framing dialect, resolved from the loaded tokenizer's special tokens.
///
/// `.chatml` is detected by the presence of the `<|im_end|>` special token
/// (Qwen-style ChatML); everything else uses the Gemma 4 contract.
public enum ChatDialect: String, Sendable {
    case gemma
    case chatml
}

/// Detokenization pipeline declared by `tokenizer.json`. Resolved from the
/// decoder declaration rather than from the chat dialect, because it is the
/// decoder — not the chat framing — that decides how token text becomes bytes.
public enum TokenDecoding: String, Sendable {
    /// `Sequence[Replace("▁" -> " "), ByteFallback, Fuse]` — see `GemmaDecoding`.
    case gemmaByteFallback
    /// GPT-2 `bytes_to_unicode` — see `ByteLevelDecoding`.
    case byteLevel
}

/// Tokenizer wrapper for the supported model families (Gemma 4 and ChatML/Qwen).
///
/// Prefers tokenizer sidecars in a completed `.gturbo/tokenizer/` directory,
/// then falls back to the IT variant's Hugging Face Hub tokenizer cache. Exposes
/// typed accessors for the IDs the generator actually needs (BOS / EOS / pad /
/// end-of-turn) and adapts encode/decode to Int32 to match the buffer types
/// kernels consume.
///
/// TurboFieldfare owns the minimal chat framing because the upstream
/// `tokenizer_config.json` has no `chat_template`. Literal control-token text in
/// user content is accepted as a trusted-input research-runtime limitation.
public struct GFTokenizer: @unchecked Sendable {
    public static let modelID = "google/gemma-4-26B-A4B-it"
    public static let chatTemplateIdentity = "gemma4-it-text-no-tools-v1"
    public static let toolChatTemplateIdentity = "gemma4-it-tools-jinja-v1"

    public let dialect: ChatDialect
    /// Decoder pipeline declared by the installed `tokenizer.json`.
    public let decoding: TokenDecoding
    /// Nominal BOS. For ChatML this is `<|endoftext|>` (the config's unused
    /// `bos_token_id`); it is never prepended — see `encode(_:addBOS:)`.
    public let bosID: Int32
    public let eosID: Int32
    public let padID: Int32
    public let endOfTurnID: Int32
    public let toolCallStartID: Int32
    public let toolCallEndID: Int32
    public let toolResponseID: Int32
    public let toolResponseEndID: Int32
    /// For ChatML these alias the `<think>` / `</think>` markers, the dialect's
    /// closest analog of Gemma's hidden-channel delimiters.
    public let channelStartID: Int32
    public let channelEndID: Int32
    /// ChatML `<think>` / `</think>` special-token IDs; nil for Gemma.
    public let thinkStartID: Int32?
    public let thinkEndID: Int32?
    public let stopTokenIDs: Set<Int32>
    public let vocabSize: Int
    /// The channel/tool markers that structure assistant output. Streaming
    /// treats them as detokenizer barriers: a byte-fallback run must not span
    /// one, or its text would surface after the marker and be routed under the
    /// wrong channel state.
    public var structuralMarkerIDs: Set<Int32> {
        [toolCallStartID, toolCallEndID, toolResponseID, toolResponseEndID,
         channelStartID, channelEndID]
    }
    /// IDs that `decode(skipSpecialTokens: true)` strips — the
    /// `added_tokens[special == true]` set from `tokenizer.json`, identical to
    /// the filter the library's own decode applies before its decoder chain.
    let specialTokenIDs: Set<Int32>

    /// BOS actually prepended by `encode(_:addBOS:)`; nil for dialects that
    /// never use a BOS prefix (ChatML).
    private let bosPrefixID: Int32?

    @usableFromInline
    let tokenizer: any Tokenizer

    public static func load() async throws -> GFTokenizer {
        try await GFTokenizerLoadCoordinator.shared.load(.pretrained(modelID))
    }

    public static func load(from folder: URL) async throws -> GFTokenizer {
        try await GFTokenizerLoadCoordinator.shared.load(.local(folder.standardizedFileURL.path))
    }

    public static func load(forModelDirectory modelDirectory: URL,
                            environment: [String: String] = ProcessInfo.processInfo.environment) async throws -> GFTokenizer {
        if let folder = tokenizerFolder(forModelDirectory: modelDirectory, environment: environment) {
            return try await load(from: folder)
        }
        return try await load()
    }

    public static func tokenizerFolder(forModelDirectory modelDirectory: URL,
                                       environment: [String: String] = ProcessInfo.processInfo.environment,
                                       fileManager: FileManager = .default) -> URL? {
        let sidecar = modelDirectory
            .standardizedFileURL
            .appendingPathComponent("tokenizer", isDirectory: true)
        if hasTokenizerJSON(in: sidecar, fileManager: fileManager) {
            return sidecar
        }

        guard let override = environment["TURBO_FIELDFARE_TOKENIZER_DIR"], !override.isEmpty else {
            return nil
        }
        let overrideURL = URL(fileURLWithPath: override).standardizedFileURL
        return hasTokenizerJSON(in: overrideURL, fileManager: fileManager) ? overrideURL : nil
    }

    static func loadUncached(pretrained modelID: String = Self.modelID) async throws -> GFTokenizer {
        try await make(from: LanguageModelConfigurationFromHub(modelName: modelID))
    }

    static func loadUncached(from folder: URL) async throws -> GFTokenizer {
        try await make(from: LanguageModelConfigurationFromHub(modelFolder: folder))
    }

    /// Build from the raw tokenizer configs so the decoder pipeline and the
    /// special-token set can be validated against `tokenizer.json` itself.
    /// Same fetch/caching path `AutoTokenizer.from(pretrained:)` uses internally.
    private static func make(from hub: LanguageModelConfigurationFromHub) async throws -> GFTokenizer {
        guard let tokenizerConfig = try await hub.tokenizerConfig else {
            throw GFTokenizerError.missingTokenizerConfig
        }
        let tokenizerData = try await hub.tokenizerData
        let underlying = try AutoTokenizer.from(
            tokenizerConfig: tokenizerConfig, tokenizerData: tokenizerData)
        return try GFTokenizer(tokenizer: underlying, tokenizerData: tokenizerData)
    }

    private static func hasTokenizerJSON(in folder: URL, fileManager: FileManager) -> Bool {
        fileManager.fileExists(atPath: folder.appendingPathComponent("tokenizer.json").path)
    }

    /// Reject a tokenizer whose declared decoder is not the pinned Gemma 4
    /// sequence.
    ///
    /// `GemmaDecoding` reproduces `Sequence[Replace("▁" -> " "), ByteFallback,
    /// Fuse]` rather than calling `Tokenizers.decode`, so that decode stays
    /// lossless and per-token (see `GemmaDecoding`). The installed tokenizer is
    /// pinned and hash-validated, but `TURBO_FIELDFARE_TOKENIZER_DIR` can point
    /// at any directory. Without this check a tokenizer declaring a different
    /// decoder would decode subtly wrong text instead of failing.
    ///
    /// The check reads `tokenizer.json`'s decoder declaration structurally, so
    /// it rejects any foreign decoder — including one whose behavior happens to
    /// coincide on a handful of probe strings — without asserting the library's
    /// exact runtime output, which a benign dependency bump may change.
    /// Behavioral agreement with the library is pinned by the differential
    /// tests instead.
    @discardableResult
    static func verifyDecoderConfiguration(_ tokenizerData: Config) throws -> TokenDecoding {
        let decoder = tokenizerData["decoder"]
        // ChatML/Qwen declares a bare ByteLevel decoder, reproduced by
        // `ByteLevelDecoding` on the same terms.
        if decoder.type.string() == "ByteLevel" { return .byteLevel }
        let steps = decoder.decoders.array(or: [])
        guard decoder.type.string() == "Sequence",
              steps.count == 3,
              steps[0].type.string() == "Replace",
              steps[0].pattern.String.string() == "▁",
              steps[0].content.string() == " ",
              steps[1].type.string() == "ByteFallback",
              steps[2].type.string() == "Fuse"
        else {
            throw GFTokenizerError.unsupportedDecoder(actual: decoder.description)
        }
        return .gemmaByteFallback
    }

    /// Resolve a special token to its ID, rejecting `<unk>` substitution.
    ///
    /// BPE's `convertTokenToId` returns the unknown-token ID — not `nil` — for
    /// a token absent from the vocab, so a plain `guard let` never fires and a
    /// missing marker would silently bind to `<unk>`, colliding with every
    /// other missing marker. The round-trip through `convertIdToToken` detects
    /// any substitution.
    static func requireTokenID(_ tokenizer: any Tokenizer, _ token: String) throws -> Int32 {
        guard let id = tokenizer.convertTokenToId(token),
              tokenizer.convertIdToToken(id) == token else {
            throw GFTokenizerError.missingSpecialToken(token)
        }
        return try int32ID(token, id)
    }

    /// Config-supplied IDs are full-range `Int`; a trapping `Int32(_:)` would
    /// crash on a corrupt tokenizer instead of taking the typed error path
    /// every other malformed-config case uses.
    private static func int32ID(_ token: String, _ id: Int) throws -> Int32 {
        guard let value = Int32(exactly: id) else {
            throw GFTokenizerError.invalidTokenID(token: token, id: id)
        }
        return value
    }

    public init(tokenizer: any Tokenizer, tokenizerData: Config) throws {
        self.tokenizer = tokenizer
        self.decoding = try Self.verifyDecoderConfiguration(tokenizerData)

        // The same `added_tokens[special == true]` ID set the library's
        // `decode(skipSpecialTokens: true)` filters before running its decoder.
        var specials: Set<Int32> = []
        for added in tokenizerData["addedTokens"].array(or: []) {
            guard added["special"].boolean(or: false),
                  let id = added["id"].integer() else { continue }
            try specials.insert(Self.int32ID(added.content.string() ?? "added token", id))
        }
        self.specialTokenIDs = specials

        let dialect: ChatDialect =
            Self.specialTokenID(tokenizer, Self.imEndMark) != nil ? .chatml : .gemma
        let resolved = dialect == .chatml
            ? try Self.resolveChatMLTokens(tokenizer)
            : try Self.resolveGemmaTokens(tokenizer)

        self.dialect = dialect
        self.bosID = resolved.bosID
        self.bosPrefixID = resolved.bosPrefixID
        self.eosID = resolved.eosID
        self.padID = resolved.padID
        self.endOfTurnID = resolved.endOfTurnID
        self.toolCallStartID = resolved.toolCallStartID
        self.toolCallEndID = resolved.toolCallEndID
        self.toolResponseID = resolved.toolResponseID
        self.toolResponseEndID = resolved.toolResponseEndID
        self.channelStartID = resolved.channelStartID
        self.channelEndID = resolved.channelEndID
        self.thinkStartID = resolved.thinkStartID
        self.thinkEndID = resolved.thinkEndID
        self.stopTokenIDs = resolved.stopTokenIDs
        self.vocabSize = resolved.vocabSize
    }

    private struct ResolvedSpecialTokens {
        let bosID: Int32
        let bosPrefixID: Int32?
        let eosID: Int32
        let padID: Int32
        let endOfTurnID: Int32
        let toolCallStartID: Int32
        let toolCallEndID: Int32
        let toolResponseID: Int32
        let toolResponseEndID: Int32
        let channelStartID: Int32
        let channelEndID: Int32
        let thinkStartID: Int32?
        let thinkEndID: Int32?
        let stopTokenIDs: Set<Int32>
        let vocabSize: Int
    }

    private static func resolveGemmaTokens(
        _ tokenizer: any Tokenizer
    ) throws -> ResolvedSpecialTokens {
        guard let bos = tokenizer.bosTokenId else {
            throw GFTokenizerError.missingSpecialToken("<bos>")
        }
        guard let eos = tokenizer.eosTokenId else {
            throw GFTokenizerError.missingSpecialToken("<eos>")
        }
        // `requireTokenID` round-trips each marker through `convertIdToToken`;
        // a plain `convertTokenToId` would silently bind a missing marker to
        // `<unk>` and collide every missing marker onto the same ID.
        let bosID = try Self.int32ID("<bos>", bos)
        let eosID = try Self.int32ID("<eos>", eos)
        let pad = try Self.requireTokenID(tokenizer, "<pad>")
        let eot = try Self.requireTokenID(tokenizer, "<turn|>")
        let toolCallStart = try Self.requireTokenID(tokenizer, "<|tool_call>")
        let toolCallEnd = try Self.requireTokenID(tokenizer, "<tool_call|>")
        let toolResponse = try Self.requireTokenID(tokenizer, "<|tool_response>")
        let toolResponseEnd = try Self.requireTokenID(tokenizer, "<tool_response|>")
        let channelStart = try Self.requireTokenID(tokenizer, "<|channel>")
        let channelEnd = try Self.requireTokenID(tokenizer, "<channel|>")
        // Multimodal prompt layout uses compile-time marker IDs, so verify this
        // Gemma tokenizer resolves the same vocabulary positions before any
        // image prompt can be constructed.
        for (token, expected) in [
            ("<|image>", MultimodalPromptRenderer.beginImageTokenID),
            ("<|image|>", MultimodalPromptRenderer.imageTokenID),
            ("<image|>", MultimodalPromptRenderer.endImageTokenID),
        ] {
            let resolved = try Self.requireTokenID(tokenizer, token)
            guard resolved == expected else {
                throw GFTokenizerError.specialTokenMismatch(
                    token: token, expected: expected, resolved: resolved)
            }
        }
        return ResolvedSpecialTokens(
            bosID: bosID,
            bosPrefixID: bosID,
            eosID: eosID,
            padID: pad,
            endOfTurnID: eot,
            toolCallStartID: toolCallStart,
            toolCallEndID: toolCallEnd,
            toolResponseID: toolResponse,
            toolResponseEndID: toolResponseEnd,
            channelStartID: channelStart,
            channelEndID: channelEnd,
            thinkStartID: nil,
            thinkEndID: nil,
            stopTokenIDs: [eosID, eot, toolResponse],
            vocabSize: 262_144)
    }

    /// Resolves a token string to its ID, rejecting the unk-token fallback
    /// some tokenizers substitute for out-of-vocabulary strings.
    private static func specialTokenID(_ tokenizer: any Tokenizer, _ token: String) -> Int? {
        guard let id = tokenizer.convertTokenToId(token),
              tokenizer.convertIdToToken(id) == token else { return nil }
        return id
    }

    private static func resolveChatMLTokens(
        _ tokenizer: any Tokenizer
    ) throws -> ResolvedSpecialTokens {
        func id(_ token: String) throws -> Int32 {
            guard let value = specialTokenID(tokenizer, token) else {
                throw GFTokenizerError.missingSpecialToken(token)
            }
            return Int32(value)
        }
        // `<|im_start|>` is required even though no stored property holds it;
        // template rendering relies on the tokenizer recognizing its text.
        _ = try id(Self.imStartMark)
        let imEnd = try id(Self.imEndMark)
        let endOfText = try id("<|endoftext|>")
        let toolCallStart = try id("<tool_call>")
        let toolCallEnd = try id("</tool_call>")
        let toolResponse = try id("<tool_response>")
        let toolResponseEnd = try id("</tool_response>")
        let thinkStart = try id("<think>")
        let thinkEnd = try id("</think>")
        return ResolvedSpecialTokens(
            bosID: endOfText,
            bosPrefixID: nil,
            eosID: endOfText,
            padID: endOfText,
            endOfTurnID: imEnd,
            toolCallStartID: toolCallStart,
            toolCallEndID: toolCallEnd,
            toolResponseID: toolResponse,
            toolResponseEndID: toolResponseEnd,
            channelStartID: thinkStart,
            channelEndID: thinkEnd,
            thinkStartID: thinkStart,
            thinkEndID: thinkEnd,
            stopTokenIDs: [imEnd, endOfText],
            // The model's padded embedding/lm_head row count, not the
            // tokenizer's actual vocab (248 077) — logits buffers use this.
            vocabSize: 248_320)
    }

    /// Encode UTF-8 text to token IDs. `addBOS = true` prepends `<bos>`.
    ///
    /// The library's `addSpecialTokens: true` flag is a no-op for the Gemma 4 IT
    /// tokenizer (its config has `add_bos_token = false`; BOS is expected to come
    /// from the chat template). We prepend manually so the kernel-facing API stays
    /// the same regardless of upstream defaults. ChatML has no BOS, so `addBOS`
    /// is a no-op for that dialect.
    public func encode(_ text: String, addBOS: Bool = true) -> [Int32] {
        let base = tokenizer.encode(text: text, addSpecialTokens: false).map(Int32.init)
        guard addBOS, let bosPrefixID else { return base }
        return [bosPrefixID] + base
    }

    /// Decode token IDs to text. `skipSpecialTokens` strips BOS/EOS/turn markers from the output.
    ///
    /// Runs the pinned Gemma decoder sequence directly (see `GemmaDecoding`)
    /// rather than `Tokenizers.decode`, whose trailing
    /// `clean_up_tokenization_spaces` pass defaults to on for this tokenizer and
    /// rewrites model output (`"he said ' ok ' now"` -> `"he said'ok'now"`),
    /// breaking `decode(encode(x)) == x`. Batch decode is a push-loop over
    /// `GFDetokenizer`, so batch and streaming decode agree by construction.
    public func decode(_ ids: [Int32], skipSpecialTokens: Bool = true) -> String {
        var detok = GFDetokenizer(tokenizer: self, skipSpecialTokens: skipSpecialTokens)
        var text = ""
        for id in ids {
            text += detok.push(id)
        }
        return text + detok.flush()
    }

    // MARK: - Chat template

    public enum Role: String, Sendable { case system, developer, user, assistant, tool }
    public struct HistoricalToolCall: Sendable, Equatable {
        public let id: String
        public let name: String
        public let arguments: JSONValue

        public init(id: String, name: String, arguments: JSONValue) {
            self.id = id
            self.name = name
            self.arguments = arguments
        }
    }

    public struct FunctionDefinition: Sendable, Equatable {
        public let name: String
        public let description: String
        public let parameters: JSONValue

        public init(name: String, description: String, parameters: JSONValue) {
            self.name = name
            self.description = description
            self.parameters = parameters
        }
    }

    public struct Message: Sendable, Equatable {
        public let role: Role
        public let content: String?
        public let toolCalls: [HistoricalToolCall]
        public let toolCallID: String?
        public let name: String?

        public init(role: Role, content: String) {
            self.role = role
            self.content = content
            self.toolCalls = []
            self.toolCallID = nil
            self.name = nil
        }

        public init(role: Role,
                    content: String?,
                    toolCalls: [HistoricalToolCall] = [],
                    toolCallID: String? = nil,
                    name: String? = nil) {
            self.role = role
            self.content = content
            self.toolCalls = toolCalls
            self.toolCallID = toolCallID
            self.name = name
        }
    }

    /// Text-only, no-tool rendering of the pinned checkpoint's bundled
    /// `chat_template.jinja`, with thinking disabled. Keeping this narrow makes
    /// unsupported tool/media behavior explicit instead of approximating it.
    private static let turnOpen    = "<|turn>"
    private static let turnClose   = "<turn|>"
    private static let bosMark     = "<bos>"
    private static let imStartMark = "<|im_start|>"
    private static let imEndMark   = "<|im_end|>"
    /// Generation prompt with thinking disabled, matching the Jinja template's
    /// `add_generation_prompt` + `enable_thinking=false` branch.
    private static let chatMLGenerationSuffix =
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"

    public func applyChatTemplate(_ messages: [Message]) throws -> String {
        switch dialect {
        case .gemma: return try gemmaChatTemplate(messages)
        case .chatml: return try chatMLChatTemplate(messages)
        }
    }

    private func gemmaChatTemplate(_ messages: [Message]) throws -> String {
        var s = Self.bosMark
        for (index, message) in messages.enumerated() {
            guard let rawContent = message.content else {
                throw GFTokenizerError.invalidChatTemplate("text-only messages require content")
            }
            let content = rawContent.trimmingCharacters(in: .whitespacesAndNewlines)
            if message.role == .system && index != 0 {
                throw GFTokenizerError.invalidChatTemplate("system message must be first")
            }
            let role = message.role == .assistant ? "model" : message.role.rawValue
            s += Self.turnOpen + role + "\n" + content + Self.turnClose + "\n"
        }
        s += Self.turnOpen + "model\n<|channel>thought\n<channel|>"
        return s
    }

    private func chatMLChatTemplate(_ messages: [Message]) throws -> String {
        var s = ""
        for (index, message) in messages.enumerated() {
            guard let rawContent = message.content else {
                throw GFTokenizerError.invalidChatTemplate("text-only messages require content")
            }
            let content = rawContent.trimmingCharacters(in: .whitespacesAndNewlines)
            if message.role == .system && index != 0 {
                throw GFTokenizerError.invalidChatTemplate("system message must be first")
            }
            s += Self.imStartMark + message.role.rawValue + "\n" + content + Self.imEndMark + "\n"
        }
        s += Self.chatMLGenerationSuffix
        return s
    }

    public func encodeToolChat(messages: [Message],
                               tools: [FunctionDefinition]) throws -> [Int32] {
        guard tokenizer.hasChatTemplate else {
            throw GFTokenizerError.missingToolTemplate
        }
        let upstreamMessages: [Tokenizers.Message] = try messages.map { message in
            var value: Tokenizers.Message = [
                "role": message.role.rawValue,
                "content": message.content,
            ]
            if !message.toolCalls.isEmpty {
                value["tool_calls"] = try message.toolCalls.map { call -> [String: any Sendable] in
                    [
                        "id": call.id,
                        "type": "function",
                        "function": [
                            "name": call.name,
                            "arguments": try call.arguments.jinjaSendableValue(),
                        ] as [String: any Sendable],
                    ]
                }
            }
            if let toolCallID = message.toolCallID { value["tool_call_id"] = toolCallID }
            if let name = message.name { value["name"] = name }
            return value
        }
        let upstreamTools: [ToolSpec] = try tools.map { tool in
            [
                "type": "function",
                "function": [
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": try tool.parameters.jinjaSendableValue(),
                ] as [String: any Sendable],
            ]
        }
        return try tokenizer.applyChatTemplate(
            messages: upstreamMessages,
            chatTemplate: nil,
            addGenerationPrompt: true,
            truncation: false,
            maxLength: nil,
            tools: upstreamTools,
            additionalContext: ["enable_thinking": false]
        ).map(Int32.init)
    }

    public func encodeTextContinuation(userContent: String) -> [Int32] {
        let content = userContent.trimmingCharacters(in: .whitespacesAndNewlines)
        switch dialect {
        case .gemma:
            return [endOfTurnID] + encode(
                "\n\(Self.turnOpen)user\n\(content)\(Self.turnClose)\n"
                    + "\(Self.turnOpen)model\n<|channel>thought\n<channel|>",
                addBOS: false)
        case .chatml:
            return [endOfTurnID] + encode(
                "\n\(Self.imStartMark)user\n\(content)\(Self.imEndMark)\n"
                    + Self.chatMLGenerationSuffix,
                addBOS: false)
        }
    }

    /// A user turn carrying images, encoded as a continuation onto an existing
    /// KV prefix. Mirrors `encodeTextContinuation` exactly and then expands each
    /// image marker the way `MultimodalPromptRenderer` does, so the result is
    /// the same tokens a full render of that turn would produce. Token counts
    /// come from image geometry, so no image has to be encoded to build this.
    /// `openingConversation` builds the turn as the *first* turn of a
    /// conversation instead of a continuation: the chat template's opening,
    /// including `<bos>`, rather than a continuation's leading end-of-turn.
    /// Without it a fresh conversation's first image turn started with a
    /// dangling end-of-turn and no BOS, which this model is sensitive to.
    public func encodeMultimodalUserContinuation(
        textAndImages: [MultimodalContinuationPart],
        imageTokenCounts: [Int],
        openingConversation: Bool = false
    ) throws -> MultimodalContinuationTokens {
        var text = ""
        var expected = 0
        for part in textAndImages {
            switch part {
            case .text(let value):
                guard !value.contains(MultimodalPromptRenderer.placeholder) else {
                    throw MultimodalPromptRendererError.reservedImageMarker
                }
                text += value
            case .image:
                text += MultimodalPromptRenderer.placeholder
                expected += 1
            }
        }
        guard expected == imageTokenCounts.count, expected > 0 else {
            throw MultimodalPromptRendererError.placeholderMismatch
        }
        guard text.components(
            separatedBy: MultimodalPromptRenderer.placeholder).count - 1 == expected else {
            throw MultimodalPromptRendererError.reservedImageMarker
        }
        let content = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let template: [Int32]
        if openingConversation {
            // The same rendering the text path uses for an empty KV, so the
            // first turn of a conversation is identical whether or not it
            // carries an image.
            template = encode(
                try applyChatTemplate([Message(role: .user, content: content)]),
                addBOS: false)
        } else {
            template = [endOfTurnID] + encode(
                "\n\(Self.turnOpen)user\n\(content)\(Self.turnClose)\n"
                    + "\(Self.turnOpen)model\n<|channel>thought\n<channel|>",
                addBOS: false)
        }
        // Count the placeholders the tokenizer actually produced before indexing
        // anything by them. The per-part marker check above cannot see a marker
        // split across two text parts, which concatenation would reassemble into
        // a real image token and leave more placeholders than counts.
        let placeholders = template.reduce(into: 0) {
            if $1 == MultimodalPromptRenderer.imageTokenID { $0 += 1 }
        }
        guard placeholders == imageTokenCounts.count else {
            throw MultimodalPromptRendererError.placeholderMismatch
        }

        var effective: [Int32] = []
        var embedding: [Int32] = []
        var ranges: [Range<Int>] = []
        var index = 0
        for token in template {
            guard token == MultimodalPromptRenderer.imageTokenID else {
                effective.append(token)
                embedding.append(token)
                continue
            }
            let count = imageTokenCounts[index]
            effective.append(MultimodalPromptRenderer.beginImageTokenID)
            embedding.append(MultimodalPromptRenderer.beginImageTokenID)
            let lower = effective.count
            effective.append(contentsOf: repeatElement(
                MultimodalPromptRenderer.imageTokenID, count: count))
            embedding.append(contentsOf: repeatElement(Int32(0), count: count))
            ranges.append(lower..<effective.count)
            effective.append(MultimodalPromptRenderer.endImageTokenID)
            embedding.append(MultimodalPromptRenderer.endImageTokenID)
            index += 1
        }
        guard index == imageTokenCounts.count else {
            throw MultimodalPromptRendererError.placeholderMismatch
        }
        return MultimodalContinuationTokens(
            effectiveTokenIDs: effective,
            embeddingTokenIDs: embedding,
            imageTokenRanges: ranges)
    }

    public func encodeToolResultContinuation(
        cachedMessages: [Message],
        assistant: Message,
        incomingMessages: [Message],
        tools: [FunctionDefinition]
    ) throws -> [Int32] {
        // The ChatML template's `<think>` stripping depends on each assistant
        // turn's position relative to the last user query, so a re-rendered
        // prefix is not guaranteed to be a token prefix of the full render.
        // Callers (ServerPromptCache) fall back to prefix matching.
        guard dialect == .gemma else {
            throw GFTokenizerError.unsupportedForDialect("tool-result KV continuation")
        }
        let prefix = try encodeToolChat(
            messages: cachedMessages + [assistant],
            tools: tools)
        let full = try encodeToolChat(messages: incomingMessages, tools: tools)
        let callCount = assistant.toolCalls.count
        let starts = prefix.indices.filter { prefix[$0] == toolCallStartID }
        guard callCount > 0, starts.count >= callCount,
              let callEnd = prefix.lastIndex(of: toolCallEndID) else {
            throw GFTokenizerError.invalidChatTemplate(
                "cached assistant tool-call boundary is missing")
        }
        let callStart = starts[starts.count - callCount]
        let callSequence = Array(prefix[callStart...callEnd])
        let matches = full.subsequenceStartIndices(matching: callSequence)
        // The boundary is unambiguous by POSITION even when the sequence repeats.
        // `prefix` and `full` render the same leading history, and the generation
        // prompt is appended past the boundary, so the occurrence at `callStart`
        // is the cached call itself rather than an identical earlier twin.
        //
        // Requiring a unique match cost agentic loops their cache: as soon as a
        // model repeated a call verbatim — `ls` of the same path, a re-read of
        // the same file — the sequence occurred twice, the bridge refused to
        // encode, and the cache stayed dead for the rest of the conversation,
        // because every later request carried both twins too.
        let boundary: Int
        if matches.contains(callStart) {
            boundary = callStart
        } else if matches.count == 1 {
            boundary = matches[0]
        } else {
            throw GFTokenizerError.invalidChatTemplate(
                "cached assistant tool-call boundary is ambiguous")
        }
        let suffixStart = boundary + callSequence.count
        let suffix = Array(full[suffixStart...])
        guard suffix.first == toolResponseID else {
            throw GFTokenizerError.invalidChatTemplate(
                "tool-result continuation does not begin at the KV boundary")
        }
        return suffix
    }
}

private extension Array where Element: Equatable {
    func subsequenceStartIndices(matching needle: [Element]) -> [Int] {
        guard !needle.isEmpty, needle.count <= count else { return [] }
        return indices.dropLast(needle.count - 1).filter { start in
            self[start..<(start + needle.count)].elementsEqual(needle)
        }
    }
}

private enum GFTokenizerLoadSource: Hashable {
    case pretrained(String)
    case local(String)
}

private actor GFTokenizerLoadCoordinator {
    static let shared = GFTokenizerLoadCoordinator()

    private var tasks: [GFTokenizerLoadSource: Task<GFTokenizer, Error>] = [:]

    func load(_ source: GFTokenizerLoadSource) async throws -> GFTokenizer {
        if let task = tasks[source] {
            return try await task.value
        }

        // Keep the CPU-heavy tokenizer build off the coordinator actor; callers
        // share the task result instead of owning its cancellation.
        let task = Task.detached(priority: .userInitiated) { () throws -> GFTokenizer in
            switch source {
            case .pretrained(let modelID):
                return try await GFTokenizer.loadUncached(pretrained: modelID)
            case .local(let path):
                return try await GFTokenizer.loadUncached(from: URL(fileURLWithPath: path))
            }
        }
        tasks[source] = task

        do {
            return try await task.value
        } catch {
            tasks[source] = nil
            throw error
        }
    }
}
