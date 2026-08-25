import Foundation
import Testing
import TurboFieldfare
import TurboFieldfareAppCore
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

        let port = try await server.start(configuration)
        #expect(port > 0)
        #expect(await server.isRunning)
        #expect(await server.boundPort == port)
        #expect(await server.modelID == "test-model")
        await #expect(throws: AppHostedServerError.alreadyRunning) {
            try await server.start(configuration)
        }

        try await server.stop()
        #expect(!(await server.isRunning))
        #expect(await server.boundPort == nil)
        try await server.stop()
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
