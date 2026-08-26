import Foundation
import Synchronization
import Testing
import TurboFieldfare
import TurboFieldfareAppCore
import TurboFieldfareServerCore
@testable import TurboFieldfareAppServer

@Suite struct AppHostedServerTests {
    @Test func loopbackLayerStartsOnAnEphemeralPortAndStops() async throws {
        let broker = SharedInferenceBroker(client: HostedServerNoopClient())
        let server = AppHostedServer(broker: broker)
        let configuration = AppHostedServerConfiguration(
            modelID: "test-model",
            chatDialect: .gemma,
            visionCapability: "missing",
            port: 0,
            queueLimit: 1,
            runtime: AppServerRuntimeConfiguration(
                modelDirectory: FileManager.default.temporaryDirectory,
                maxContextTokens: 4_096,
                runtimeOptions: AppRuntimeOptions()))

        let activities = Mutex<[ServerCoordinatorActivity]>([])
        let errors = Mutex<[String]>([])
        let port = try await server.start(
            configuration,
            onActivity: { activity in
                activities.withLock { $0.append(activity) }
            },
            onError: { error in
                errors.withLock { $0.append(error) }
            })
        #expect(port > 0)
        #expect(await server.isRunning)
        #expect(await server.boundPort == port)
        #expect(await server.modelID == "test-model")
        await #expect(throws: AppHostedServerError.alreadyRunning) {
            try await server.start(configuration)
        }

        var request = URLRequest(
            url: try #require(URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.httpBody = Data(#"{"model":"test-model","messages":[{"role":"user","content":"hello"}]}"#.utf8)
        let (_, response) = try await URLSession.shared.data(for: request)
        #expect((response as? HTTPURLResponse)?.statusCode == 500)
        #expect(await waitUntil { !errors.withLock { $0 }.isEmpty })
        #expect(activities.withLock { $0 }.contains(ServerCoordinatorActivity(
            activeRequests: 1, queuedRequests: 0)))
        #expect(activities.withLock { $0 }.last == ServerCoordinatorActivity(
            activeRequests: 0, queuedRequests: 0))

        try await server.stop()
        #expect(!(await server.isRunning))
        #expect(await server.boundPort == nil)
        try await server.stop()
    }

    private func waitUntil(_ predicate: @escaping @Sendable () -> Bool) async -> Bool {
        for _ in 0..<400 {
            if predicate() { return true }
            try? await Task.sleep(for: .milliseconds(5))
        }
        return false
    }
}

private final class HostedServerNoopClient: AppModelLifecycleClient, Sendable {
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
            continuation.finish()
        }
    }

    func cancel() {}
}
