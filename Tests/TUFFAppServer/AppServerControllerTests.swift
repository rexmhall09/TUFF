import Foundation
import Testing
import TUFFAppCore
@testable import TUFFAppServer

@Suite struct AppServerControllerTests {
    @MainActor
    @Test func startChecksHealthAndStopClearsPresentationState() async throws {
        let client = ControllerServerClient()
        let broker = SharedInferenceBroker(client: client)
        let store = AppServerStore()
        let controller = AppServerController(broker: broker, store: store)
        let configuration = AppHostedServerConfiguration(
            modelID: "controller-test",
            chatDialect: .gemma,
            visionCapability: "missing",
            port: 0,
            queueLimit: 2,
            runtime: AppServerRuntimeConfiguration(
                modelDirectory: FileManager.default.temporaryDirectory,
                maxContextTokens: 4_096,
                runtimeOptions: AppRuntimeOptions()))

        controller.start(configuration)
        #expect(await waitUntil { store.status == .running })
        #expect(try #require(store.boundPort) > 0)
        #expect(store.modelID == "controller-test")
        #expect(await waitUntil { store.health == .healthy })
        #expect(controller.url?.host == "127.0.0.1")

        controller.stop()
        #expect(await waitUntil { store.status == .stopped })
        #expect(store.boundPort == nil)
        #expect(store.modelID == nil)
        #expect(store.health == .unknown)
        #expect(store.activeRequests == 0)
        #expect(store.queuedRequests == 0)
    }

    @MainActor
    private func waitUntil(_ predicate: () -> Bool) async -> Bool {
        for _ in 0..<400 {
            if predicate() { return true }
            try? await Task.sleep(for: .milliseconds(5))
        }
        return false
    }
}

private final class ControllerServerClient: AppModelLifecycleClient, Sendable {
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
        AsyncThrowingStream { continuation in
            continuation.yield(.finished(AppDiagnostics(
                generatedTokens: 0,
                stopReason: .eos,
                promptTokenCount: 1,
                timeToFirstTokenSeconds: nil,
                decodeSeconds: 0,
                tokensPerSecond: 0,
                peakMemoryBytes: nil,
                runtimeOptions: request.runtimeOptions)))
            continuation.finish()
        }
    }

    func cancel() {}
}
