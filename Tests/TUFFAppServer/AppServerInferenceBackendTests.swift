import Foundation
import Synchronization
import Testing
import TUFFEngine
import TUFFAppCore
@testable import TUFFAppServer
import TUFFServerCore

@Suite struct AppServerInferenceBackendTests {
    @Test func structuredControlsEventsAndUsageCrossTheAdapter() async throws {
        let client = CapturingServerClient(events: [
            .thinking(AppTokenEvent(
                index: 0, textDelta: "hidden", elapsedDecodeSeconds: 0.1)),
            .token(AppTokenEvent(
                index: 1, textDelta: "Paris", elapsedDecodeSeconds: 0.2)),
            .toolCall(ParsedToolCall(
                id: "call_1",
                name: "weather",
                arguments: .object(["city": .string("Paris")]),
                argumentsJSON: #"{"city":"Paris"}"#)),
            .finished(diagnostics(
                promptTokens: 20,
                generatedTokens: 2,
                stopReason: .toolCalls)),
        ])
        let backend = backend(client: client, preserveThinking: true)
        let request = ValidatedChatRequest(
            messages: [GFTokenizer.Message(role: .user, content: "weather")],
            tools: [GFTokenizer.FunctionDefinition(
                name: "weather",
                description: "Get weather.",
                parameters: .object(["type": .string("object")]))],
            stream: true,
            includeUsage: true,
            generationConfig: GenerationConfig(
                maxNewTokens: 17,
                temperature: 0.4,
                topK: 12,
                topP: 0.8,
                repetitionPenalty: 0.7,
                seed: 99,
                stopStrings: ["DONE"]),
            maximumCompletionTokens: 17,
            reasoning: .on,
            harmonyCurrentDate: "2026-08-25")
        let emitted = Mutex<[ServerInferenceEvent]>([])

        let completion = try await backend.generate(request) { event in
            emitted.withLock { $0.append(event) }
        }
        let captured = try #require(client.capturedRequest)

        #expect(captured.structuredMessages == request.messages)
        #expect(captured.tools == request.tools)
        #expect(captured.maxNewTokens == 17)
        #expect(captured.reasoning == ChatReasoning.on)
        #expect(captured.preserveThinking)
        #expect(captured.temperature == 0.4)
        #expect(captured.topK == 12)
        #expect(captured.topP == 0.8)
        #expect(captured.repetitionPenalty == 0.7)
        #expect(captured.seed == 99)
        #expect(captured.stopStrings == ["DONE"])
        #expect(captured.harmonyCurrentDate == "2026-08-25")
        #expect(completion.content == "Paris")
        #expect(completion.toolCalls.map { $0.name } == ["weather"])
        #expect(completion.finishReason == "tool_calls")
        #expect(completion.usage.promptTokens == 20)
        #expect(completion.usage.completionTokens == 2)
        #expect(completion.usage.totalTokens == 22)
        #expect(emitted.withLock { $0 } == [
            .content("Paris"),
            .toolCall(try #require(completion.toolCalls.first)),
        ])
    }

    @Test func serverImagesAreRekeyedIntoTrustedStoreThenRemoved() async throws {
        let temporary = FileManager.default.temporaryDirectory
            .appendingPathComponent("app-server-adapter-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let source = temporary.appendingPathComponent("image.bin")
        try Data([0x89, 0x50, 0x4E, 0x47]).write(to: source)
        let serverID = UUID()
        let client = CapturingServerClient(events: [
            .finished(diagnostics(
                promptTokens: 8,
                generatedTokens: 1,
                stopReason: .eos)),
        ])
        let backend = backend(client: client)
        let request = ValidatedChatRequest(
            messages: [GFTokenizer.Message(role: .user, content: "describe")],
            multimodalMessages: [MultimodalMessage(
                role: .user,
                content: [.image(id: serverID), .text("describe")])],
            imageFiles: [serverID: source],
            imageIdentities: [[String(repeating: "a", count: 64)]],
            tools: [],
            stream: false,
            includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 4),
            maximumCompletionTokens: 4)

        _ = try await backend.generate(request) { _ in }
        let captured = try #require(client.capturedRequest)
        let attachment = try #require(captured.imageAttachments.first)
        #expect(attachment.id != serverID)
        #expect(AppImageAttachmentStore.contains(attachment.fileURL))
        #expect(captured.multimodalMessages?.first?.content == [
            .image(id: attachment.id), .text("describe"),
        ])
        #expect(!FileManager.default.fileExists(atPath: attachment.fileURL.path))
        #expect(FileManager.default.fileExists(atPath: source.path))
    }

    @Test func completionStopReasonsMapToOpenAIValues() async throws {
        for (stopReason, expected) in [
            (AppStopReason.maxTokens, "length"),
            (.stopString, "stop"),
            (.endOfTurn, "stop"),
        ] {
            let client = CapturingServerClient(events: [
                .finished(diagnostics(
                    promptTokens: 1,
                    generatedTokens: 1,
                    stopReason: stopReason)),
            ])
            let completion = try await backend(client: client).generate(
                textRequest()) { _ in }
            #expect(completion.finishReason == expected)
        }
    }

    @Test func cancelledGenerationDoesNotBecomeACompletion() async {
        let client = CapturingServerClient(events: [
            .cancelled(diagnostics(
                promptTokens: 1,
                generatedTokens: 0,
                stopReason: .cancelled)),
        ])
        do {
            _ = try await backend(client: client).generate(textRequest()) { _ in }
            Issue.record("cancelled server generation should throw")
        } catch {
            #expect(error is CancellationError)
        }
    }

    private func backend(
        client: CapturingServerClient,
        preserveThinking: Bool = false
    ) -> AppServerInferenceBackend {
        AppServerInferenceBackend(
            broker: SharedInferenceBroker(client: client),
            runtime: AppServerRuntimeConfiguration(
                modelDirectory: FileManager.default.temporaryDirectory,
                maxContextTokens: 4_096,
                runtimeOptions: AppRuntimeOptions(),
                preserveThinking: preserveThinking))
    }

    private func textRequest() -> ValidatedChatRequest {
        ValidatedChatRequest(
            messages: [GFTokenizer.Message(role: .user, content: "hello")],
            tools: [],
            stream: false,
            includeUsage: false,
            generationConfig: GenerationConfig(maxNewTokens: 4),
            maximumCompletionTokens: 4)
    }

}

private final class CapturingServerClient: AppModelLifecycleClient, Sendable {
    private struct State: Sendable {
        var capturedRequest: AppGenerationRequest?
        var cancelCount = 0
    }

    private let state = Mutex(State())
    private let events: [AppInferenceEvent]

    init(events: [AppInferenceEvent]) {
        self.events = events
    }

    var capturedRequest: AppGenerationRequest? {
        state.withLock { $0.capturedRequest }
    }

    func ensureLoaded(
        modelDirectory: URL,
        maxContextTokens: Int,
        options: AppRuntimeOptions,
        forceLogitsHead: Bool,
        onState: @escaping @Sendable (AppModelLoadState) -> Void
    ) async throws {}

    func unload() async {}

    func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        state.withLock { $0.capturedRequest = request }
        return AsyncThrowingStream { continuation in
            for event in events { continuation.yield(event) }
            continuation.finish()
        }
    }

    func cancel() {
        state.withLock { $0.cancelCount += 1 }
    }
}

private func diagnostics(
    promptTokens: Int,
    generatedTokens: Int,
    stopReason: AppStopReason
) -> AppDiagnostics {
    AppDiagnostics(
        generatedTokens: generatedTokens,
        stopReason: stopReason,
        promptTokenCount: promptTokens,
        timeToFirstTokenSeconds: nil,
        decodeSeconds: 0,
        tokensPerSecond: 0,
        peakMemoryBytes: nil,
        runtimeOptions: AppRuntimeOptions())
}
