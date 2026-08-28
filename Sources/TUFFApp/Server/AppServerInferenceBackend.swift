import Foundation
import TUFFEngine
import TUFFAppCore
import TUFFServerCore

public struct AppServerRuntimeConfiguration: Sendable, Equatable {
    public var modelDirectory: URL
    public var maxContextTokens: Int
    public var runtimeOptions: AppRuntimeOptions
    public var preserveThinking: Bool

    public init(
        modelDirectory: URL,
        maxContextTokens: Int,
        runtimeOptions: AppRuntimeOptions,
        preserveThinking: Bool = false
    ) {
        self.modelDirectory = modelDirectory
        self.maxContextTokens = maxContextTokens
        self.runtimeOptions = runtimeOptions
        self.preserveThinking = preserveThinking
    }
}

/// Adapts validated OpenAI-compatible requests to the app's one shared decode
/// service. Server-staged images cross the decode service's trust boundary only
/// after they are copied into the app attachment store and re-keyed.
public final class AppServerInferenceBackend: ServerInferenceBackend, Sendable {
    private let broker: SharedInferenceBroker
    private let runtime: AppServerRuntimeConfiguration

    public init(
        broker: SharedInferenceBroker,
        runtime: AppServerRuntimeConfiguration
    ) {
        self.broker = broker
        self.runtime = runtime
    }

    public func generate(
        _ request: ValidatedChatRequest,
        onEvent: @escaping @Sendable (ServerInferenceEvent) -> Void
    ) async throws -> ServerCompletion {
        let attachmentStore = AppImageAttachmentStore()
        defer { attachmentStore.removeAll() }

        let staged = try stageImages(
            request.imageFiles, in: attachmentStore)
        let multimodal = try remap(
            request.multimodalMessages,
            imageIDs: staged.idByServerID)
        let config = request.generationConfig
        let appRequest = AppGenerationRequest(
            modelDirectory: runtime.modelDirectory,
            prompt: "",
            structuredMessages: request.messages,
            multimodalMessages: multimodal,
            tools: request.tools,
            imageAttachments: staged.attachments,
            maxNewTokens: request.maximumCompletionTokens,
            maxContextTokens: runtime.maxContextTokens,
            reasoning: request.reasoning,
            reasoningEffort: request.reasoningEffort,
            preserveThinking: runtime.preserveThinking,
            temperature: config.temperature,
            topK: config.topK,
            topP: config.topP,
            repetitionPenalty: config.repetitionPenalty,
            seed: config.seed,
            stopStrings: config.stopStrings,
            harmonyCurrentDate: request.harmonyCurrentDate,
            runtimeOptions: runtime.runtimeOptions)
        try appRequest.validate()

        var content = ""
        var toolCalls: [ParsedToolCall] = []
        var diagnostics: AppDiagnostics?
        var terminalError: AppInferenceError?
        var wasCancelled = false
        do {
            for try await event in broker.generateForServer(appRequest) {
                switch event {
                case .token(let token):
                    guard !token.textDelta.isEmpty else { continue }
                    content += token.textDelta
                    onEvent(.content(token.textDelta))
                case .toolCall(let call):
                    toolCalls.append(call)
                    onEvent(.toolCall(call))
                case .finished(let value):
                    diagnostics = value
                case .cancelled(let value):
                    diagnostics = value
                    wasCancelled = true
                case .failed(let error, let partial):
                    terminalError = error
                    diagnostics = partial
                case .prefillProgress, .memorySample, .thinking:
                    break
                }
            }
        } catch {
            if let terminalError { throw terminalError }
            throw error
        }
        if let terminalError { throw terminalError }
        if wasCancelled { throw CancellationError() }
        guard let diagnostics else {
            throw AppInferenceError.unknown(
                "decode service ended without generation diagnostics")
        }

        let promptTokens = diagnostics.promptTokenCount ?? 0
        let completionTokens = diagnostics.generatedTokens
        let total = promptTokens.addingReportingOverflow(completionTokens)
        return ServerCompletion(
            content: content,
            toolCalls: toolCalls,
            finishReason: finishReason(
                diagnostics.stopReason, hasToolCalls: !toolCalls.isEmpty),
            usage: OpenAIUsage(
                promptTokens: promptTokens,
                completionTokens: completionTokens,
                totalTokens: total.overflow ? .max : total.partialValue))
    }

    private func stageImages(
        _ files: [UUID: URL],
        in store: AppImageAttachmentStore
    ) throws -> (attachments: [AppImageAttachment], idByServerID: [UUID: UUID]) {
        var attachments: [AppImageAttachment] = []
        var idByServerID: [UUID: UUID] = [:]
        attachments.reserveCapacity(files.count)
        idByServerID.reserveCapacity(files.count)
        for (serverID, fileURL) in files.sorted(by: {
            $0.key.uuidString < $1.key.uuidString
        }) {
            let attachment = try store.stage(fileURL)
            attachments.append(attachment)
            idByServerID[serverID] = attachment.id
        }
        return (attachments, idByServerID)
    }

    private func remap(
        _ messages: [MultimodalMessage]?,
        imageIDs: [UUID: UUID]
    ) throws -> [MultimodalMessage]? {
        guard let messages else { return nil }
        return try messages.map { message in
            let content = try message.content.map { part in
                switch part {
                case .text:
                    return part
                case .image(let serverID):
                    guard let appID = imageIDs[serverID] else {
                        throw AppInferenceError.invalidRequest(
                            "server image reference was not staged")
                    }
                    return .image(id: appID)
                }
            }
            return MultimodalMessage(
                role: message.role,
                content: content,
                thinking: message.thinking,
                toolCalls: message.toolCalls,
                toolCallID: message.toolCallID,
                name: message.name)
        }
    }

    private func finishReason(
        _ stopReason: AppStopReason,
        hasToolCalls: Bool
    ) -> String {
        if hasToolCalls || stopReason == .toolCalls { return "tool_calls" }
        if stopReason == .maxTokens { return "length" }
        return "stop"
    }
}
