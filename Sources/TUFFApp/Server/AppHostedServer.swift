import Foundation
import NIOCore
import TUFFEngine
import TUFFAppCore
import TUFFServerCore

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
    private var server: TUFFHTTPServer?
    private var pendingServer: TUFFHTTPServer?
    public private(set) var boundPort: Int?
    public private(set) var modelID: String?

    public init(broker: SharedInferenceBroker) {
        self.broker = broker
    }

    public var isRunning: Bool { server != nil }

    @discardableResult
    public func start(
        _ configuration: AppHostedServerConfiguration,
        onActivity: @escaping @Sendable (ServerCoordinatorActivity) -> Void = { _ in },
        onError: @escaping @Sendable (String) -> Void = { _ in }
    ) async throws -> Int {
        guard server == nil, pendingServer == nil else {
            throw AppHostedServerError.alreadyRunning
        }
        let backend = AppServerInferenceBackend(
            broker: broker,
            runtime: configuration.runtime)
        let candidate = TUFFHTTPServer(
            modelID: configuration.modelID,
            queueLimit: configuration.queueLimit,
            backend: backend,
            chatDialect: configuration.chatDialect,
            visionCapability: configuration.visionCapability,
            onRequestActivity: onActivity,
            onRequestError: onError)
        pendingServer = candidate
        do {
            let channel = try await candidate.start(port: configuration.port)
            guard pendingServer === candidate else {
                try await candidate.shutdown()
                throw CancellationError()
            }
            guard let port = channel.localAddress?.port else {
                try await candidate.shutdown()
                throw AppHostedServerError.missingBoundPort
            }
            pendingServer = nil
            server = candidate
            boundPort = port
            modelID = configuration.modelID
            return port
        } catch {
            if pendingServer === candidate { pendingServer = nil }
            try? await candidate.shutdown()
            throw error
        }
    }

    public func stop() async throws {
        let server = server ?? pendingServer
        self.server = nil
        pendingServer = nil
        boundPort = nil
        modelID = nil
        try await server?.shutdown()
    }
}
