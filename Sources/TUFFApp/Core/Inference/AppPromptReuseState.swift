import TUFFEngine

/// One text prefix backed by the current runner's KV and recurrent state.
/// Matching the complete token prefix avoids rewinding models with GDN or
/// n-gram state. Image-pad tokens alone cannot identify image embeddings.
struct AppPromptReuseState {
    private var tokenIDs: [Int32] = []
    private var completedTurn: (request: AppGenerationRequest, response: String,
                                boundary: [Int32], reason: StopReason)?

    mutating func clear() {
        tokenIDs.removeAll(keepingCapacity: false)
        completedTurn = nil
    }

    /// ChatML omits the empty thinking block when rendering an old assistant
    /// message. Bridge an unchanged completed exchange using the same native
    /// continuation encoder as the server, retaining the tokens actually seen
    /// by the runner. Restrict this path to ordinary, non-thinking text turns.
    func textContinuation(request: AppGenerationRequest, tokenizer: GFTokenizer,
                          modelVariant: ModelVariant) -> [Int32]? {
        guard let entry = completedTurn,
              Self.canBridge(request), Self.canBridge(entry.request),
              request.modelDirectory == entry.request.modelDirectory,
              request.systemPrompt == entry.request.systemPrompt,
              request.history.count == entry.request.history.count + 1,
              zip(request.history.dropLast(), entry.request.history).allSatisfy({ lhs, rhs in
                  lhs.prompt == rhs.prompt && lhs.response == rhs.response
                      && lhs.thinking == rhs.thinking && lhs.documents == rhs.documents
                      && lhs.images == rhs.images
              }),
              let last = request.history.last,
              last.prompt == entry.request.prompt, last.response == entry.response,
              last.thinking?.isEmpty != false, last.images.isEmpty, last.documents.isEmpty,
              entry.boundary.count == 1,
              entry.reason == .endOfTurn || entry.reason == .maxTokens else { return nil }
        var bridge = tokenizer.encodeTextContinuation(userContent: request.prompt,
            modelVariant: modelVariant, reasoning: request.reasoning)
        guard !bridge.isEmpty else { return nil }
        if entry.reason == .maxTokens { bridge = entry.boundary + bridge }
        else if bridge.first != entry.boundary.first { return nil }
        return tokenIDs + bridge
    }

    private static func canBridge(_ request: AppGenerationRequest) -> Bool {
        request.structuredMessages == nil && request.multimodalMessages == nil
            && request.tools.isEmpty && request.assistantPrefix.isEmpty
            && request.imageAttachments.isEmpty && request.reasoning == .off
            && !request.preserveThinking && request.stopStrings.isEmpty
            && request.history.allSatisfy { $0.images.isEmpty }
    }

    mutating func takeStart(promptIDs: [Int32], position: Int,
                            hasImages: Bool) -> RawCompletionStart {
        defer { clear() }
        guard !hasImages, !tokenIDs.isEmpty,
              position == tokenIDs.count,
              promptIDs.count > tokenIDs.count,
              promptIDs.starts(with: tokenIDs) else { return .reset }
        return .resume(cachedPromptTokens: tokenIDs.count)
    }

    mutating func record(_ result: RawDecodeResult, hasImages: Bool,
                         request: AppGenerationRequest? = nil, response: String = "") {
        clear()
        guard !hasImages, result.kvPosition == result.kvBackedTokenIDs.count else { return }
        tokenIDs = result.kvBackedTokenIDs
        if let request {
            completedTurn = (request, response, result.uncommittedBoundaryTokenIDs, result.reason)
        }
    }
}
