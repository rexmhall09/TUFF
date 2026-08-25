import Foundation
import NIOCore
import TurboFieldfare
import TurboFieldfareAppCore
import TurboFieldfareServerCore

public enum AppHostedServerError: Error, Equatable, Sendable {
    case alreadyRunning
    case missingBoundPort
}

public struct AppHostedServerConfiguration: Sendable, Equatable {
    public var modelID: String
    public var chatDialect: ChatDialect
    public var visionCapability: String
    public var port: Int
    public var queueLimit: Int
    public var runtime: AppServerRuntimeConfiguration

    public init(
        modelID: String,
        chatDialect: ChatDialect,
        visionCapability: String,
        port: Int = 8_080,
        queueLimit: Int = 4,
        runtime: AppServerRuntimeConfiguration
    ) {
        self.modelID = modelID
        self.chatDialect = chatDialect
        self.visionCapability = visionCapability
        self.port = port
        self.queueLimit = queueLimit
        self.runtime = runtime
    }
}

/// Owns only the app's loopback HTTP layer. Model lifecycle stays with the
/// shared broker, so starting this server cannot launch another decode service.
public actor AppHostedServer {
    private let broker: SharedInferenceBroker
    private var server: TurboFieldfareHTTPServer?
    public private(set) var boundPort: Int?
    public private(set) var modelID: String?

    public init(broker: SharedInferenceBroker) {
        self.broker = broker
    }

    public var isRunning: Bool { server != nil }

    @discardableResult
    public func start(_ configuration: AppHostedServerConfiguration) async throws -> Int {
        guard server == nil else { throw AppHostedServerError.alreadyRunning }
        let backend = AppServerInferenceBackend(
            broker: broker,
            runtime: configuration.runtime)
        let candidate = TurboFieldfareHTTPServer(
            modelID: configuration.modelID,
            queueLimit: configuration.queueLimit,
            backend: backend,
            chatDialect: configuration.chatDialect,
            visionCapability: configuration.visionCapability)
        do {
            let channel = try await candidate.start(port: configuration.port)
            guard let port = channel.localAddress?.port else {
                try await candidate.shutdown()
                throw AppHostedServerError.missingBoundPort
            }
            server = candidate
            boundPort = port
            modelID = configuration.modelID
            return port
        } catch {
            try? await candidate.shutdown()
            throw error
        }
    }

    public func stop() async throws {
        guard let server else { return }
        self.server = nil
        boundPort = nil
        modelID = nil
        try await server.shutdown()
    }
}
