import Foundation
import Synchronization
import Testing
@testable import TUFFAppCore

@Suite struct SharedInferenceBrokerTests {
    @Test func chatAndServerRequestsRunFIFOWithOneActiveGeneration() async throws {
        let client = BrokerProbeClient()
        let broker = SharedInferenceBroker(client: client)
        let chat = Task { try await collect(broker.generate(request("chat"))) }
        #expect(await client.waitForStart("chat"))

        let server = Task {
            try await collect(broker.generateForServer(request("server")))
        }
        #expect(await waitUntil { broker.activity.queuedServerRequests == 1 })
        #expect(client.starts == ["chat"])
        #expect(client.maximumConcurrentGenerations == 1)

        client.finish("chat")
        #expect(await client.waitForStart("server"))
        client.finish("server")
        _ = try await chat.value
        _ = try await server.value

        #expect(client.starts == ["chat", "server"])
        #expect(client.maximumConcurrentGenerations == 1)
        #expect(broker.activity == SharedInferenceActivity())
    }

    @Test func cancellingQueuedChatDoesNotCancelActiveServer() async throws {
        let client = BrokerProbeClient()
        let broker = SharedInferenceBroker(client: client)
        let server = Task {
            try await collect(broker.generateForServer(request("server")))
        }
        #expect(await client.waitForStart("server"))
        let chat = Task { try await collect(broker.generate(request("chat"))) }
        #expect(await waitUntil { broker.activity.queuedChatRequests == 1 })

        broker.cancel()
        do {
            _ = try await chat.value
            Issue.record("queued Chat generation should have been cancelled")
        } catch {
            #expect(error is CancellationError)
        }
        #expect(client.cancelCount == 0)
        #expect(client.starts == ["server"])
        #expect(broker.activity.activeConsumer == .server)

        client.finish("server")
        _ = try await server.value
    }

    @Test func activeChatCancellationWaitsForItsTerminalBeforeServerStarts() async throws {
        let client = BrokerProbeClient()
        let broker = SharedInferenceBroker(client: client)
        let chat = Task { try await collect(broker.generate(request("chat"))) }
        #expect(await client.waitForStart("chat"))
        let server = Task {
            try await collect(broker.generateForServer(request("server")))
        }
        #expect(await waitUntil { broker.activity.queuedServerRequests == 1 })

        broker.cancel()
        #expect(await waitUntil { client.cancelCount == 1 })
        let chatEvents = try await chat.value
        #expect(chatEvents.contains { event in
            if case .cancelled = event { return true }
            return false
        })
        #expect(await client.waitForStart("server"))
        client.finish("server")
        _ = try await server.value

        #expect(client.starts == ["chat", "server"])
        #expect(client.maximumConcurrentGenerations == 1)
    }

    @Test func lifecycleWorkWaitsBehindGenerationLease() async throws {
        let client = BrokerProbeClient()
        let broker = SharedInferenceBroker(client: client)
        let server = Task {
            try await collect(broker.generateForServer(request("server")))
        }
        #expect(await client.waitForStart("server"))
        let load = Task {
            try await broker.ensureLoaded(
                modelDirectory: URL(fileURLWithPath: "/tmp/model.gturbo"),
                maxContextTokens: 4_096,
                options: AppRuntimeOptions(),
                forceLogitsHead: false) { _ in }
        }

        #expect(await waitUntil {
            broker.activity.queuedLifecycleOperations == 1
        })
        #expect(client.loadCount == 0)
        client.finish("server")
        _ = try await server.value
        try await load.value

        #expect(client.loadCount == 1)
        #expect(client.operationOrder == ["generate:server", "load"])
    }

    @Test func brokerForwardsServiceMemoryAndTranscriptReporting() {
        let client = BrokerProbeClient()
        let broker = SharedInferenceBroker(client: client)

        #expect(broker.currentInferenceMemoryBytes == 123)
        #expect(broker.currentInferenceResidentBytes == 456)
        #expect(broker.currentInferenceTowerBytes == 789)
        #expect(broker.generationTranscriptMailbox === client.generationTranscriptMailbox)
    }

    private func request(_ prompt: String) -> AppGenerationRequest {
        AppGenerationRequest(
            modelDirectory: URL(fileURLWithPath: "/tmp/model.gturbo"),
            prompt: prompt,
            maxNewTokens: 1,
            temperature: 0)
    }

    private func collect(
        _ stream: AsyncThrowingStream<AppInferenceEvent, Error>
    ) async throws -> [AppInferenceEvent] {
        var events: [AppInferenceEvent] = []
        for try await event in stream { events.append(event) }
        return events
    }

    private func waitUntil(
        _ predicate: @escaping @Sendable () -> Bool
    ) async -> Bool {
        for _ in 0..<400 {
            if predicate() { return true }
            try? await Task.sleep(for: .milliseconds(5))
        }
        return false
    }
}

private final class BrokerProbeClient: AppModelLifecycleClient,
    AppInferenceMemoryReporting, AppInferenceTranscriptReporting, Sendable {
    private struct State: Sendable {
        var continuations: [String:
            AsyncThrowingStream<AppInferenceEvent, Error>.Continuation] = [:]
        var starts: [String] = []
        var operationOrder: [String] = []
        var active: Set<String> = []
        var maximumConcurrentGenerations = 0
        var cancelCount = 0
        var loadCount = 0
    }

    private let state = Mutex(State())
    let generationTranscriptMailbox = GenerationTranscriptMailbox()
    let currentInferenceMemoryBytes: UInt64? = 123
    let currentInferenceResidentBytes: UInt64? = 456
    let currentInferenceTowerBytes: UInt64? = 789

    var starts: [String] { state.withLock { $0.starts } }
    var operationOrder: [String] { state.withLock { $0.operationOrder } }
    var maximumConcurrentGenerations: Int {
        state.withLock { $0.maximumConcurrentGenerations }
    }
    var cancelCount: Int { state.withLock { $0.cancelCount } }
    var loadCount: Int { state.withLock { $0.loadCount } }

    func ensureLoaded(
        modelDirectory: URL,
        maxContextTokens: Int,
        options: AppRuntimeOptions,
        forceLogitsHead: Bool,
        onState: @escaping @Sendable (AppModelLoadState) -> Void
    ) async throws {
        state.withLock {
            $0.loadCount += 1
            $0.operationOrder.append("load")
        }
        onState(.ready(modelDirectory: modelDirectory, loadSeconds: 0))
    }

    func unload() async {
        state.withLock { $0.operationOrder.append("unload") }
    }

    func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { continuation in
            state.withLock {
                $0.continuations[request.prompt] = continuation
                $0.starts.append(request.prompt)
                $0.operationOrder.append("generate:\(request.prompt)")
                $0.active.insert(request.prompt)
                $0.maximumConcurrentGenerations = max(
                    $0.maximumConcurrentGenerations, $0.active.count)
            }
        }
    }

    func cancel() {
        let cancelled: (String,
            AsyncThrowingStream<AppInferenceEvent, Error>.Continuation)? =
            state.withLock { state in
                state.cancelCount += 1
                guard let prompt = state.active.first,
                      let continuation = state.continuations.removeValue(
                        forKey: prompt) else { return nil }
                state.active.remove(prompt)
                return (prompt, continuation)
            }
        guard let (_, continuation) = cancelled else { return }
        continuation.yield(.cancelled(diagnostics(stopReason: .cancelled)))
        continuation.finish()
    }

    func finish(_ prompt: String) {
        let continuation = state.withLock { state in
            state.active.remove(prompt)
            return state.continuations.removeValue(forKey: prompt)
        }
        continuation?.yield(.finished(diagnostics(stopReason: .eos)))
        continuation?.finish()
    }

    func waitForStart(_ prompt: String) async -> Bool {
        for _ in 0..<400 {
            if starts.contains(prompt) { return true }
            try? await Task.sleep(for: .milliseconds(5))
        }
        return false
    }

    private func diagnostics(stopReason: AppStopReason) -> AppDiagnostics {
        AppDiagnostics(
            generatedTokens: 0,
            stopReason: stopReason,
            timeToFirstTokenSeconds: nil,
            decodeSeconds: 0,
            tokensPerSecond: 0,
            peakMemoryBytes: nil,
            runtimeOptions: AppRuntimeOptions())
    }
}
