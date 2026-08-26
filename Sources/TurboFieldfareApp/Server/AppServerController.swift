import Foundation
import Observation
import TurboFieldfareAppCore
import TurboFieldfareServerCore

@MainActor
@Observable
public final class AppServerController {
    public let store: AppServerStore
    private let server: AppHostedServer
    private var operationGeneration: UInt64 = 0
    private var operationTask: Task<Void, Never>?
    private var healthTask: Task<Void, Never>?

    public init(broker: SharedInferenceBroker, store: AppServerStore) {
        self.store = store
        self.server = AppHostedServer(broker: broker)
    }

    public var url: URL? {
        guard let port = store.boundPort else { return nil }
        return URL(string: "http://127.0.0.1:\(port)")
    }

    public func start(_ configuration: AppHostedServerConfiguration) {
        guard !store.isBusy else { return }
        operationGeneration &+= 1
        let generation = operationGeneration
        store.status = .starting
        store.health = .checking
        store.recentErrors.removeAll()
        operationTask = Task { [weak self, server] in
            do {
                let port = try await server.start(
                    configuration,
                    onActivity: { [weak self] activity in
                        Task { @MainActor in self?.apply(activity) }
                    },
                    onError: { [weak self] message in
                        Task { @MainActor in self?.recordError(message) }
                    })
                guard let self, generation == self.operationGeneration else {
                    try? await server.stop()
                    return
                }
                self.store.boundPort = port
                self.store.modelID = configuration.modelID
                self.store.status = .running
                self.operationTask = nil
                self.refreshHealth()
            } catch is CancellationError {
                guard let self, generation == self.operationGeneration else { return }
                self.store.status = .stopped
                self.store.health = .unknown
                self.operationTask = nil
            } catch {
                guard let self, generation == self.operationGeneration else { return }
                self.store.status = .failed(String(describing: error))
                self.store.health = .unreachable
                self.recordError(String(describing: error))
                self.operationTask = nil
            }
        }
    }

    public func stop() {
        guard store.isBusy else { return }
        operationGeneration &+= 1
        let generation = operationGeneration
        store.status = .stopping
        store.health = .unknown
        healthTask?.cancel()
        operationTask?.cancel()
        operationTask = Task { [weak self, server] in
            do {
                try await server.stop()
                guard let self, generation == self.operationGeneration else { return }
                self.finishStopped()
            } catch {
                guard let self, generation == self.operationGeneration else { return }
                self.store.status = .failed(String(describing: error))
                self.recordError(String(describing: error))
                self.operationTask = nil
            }
        }
    }

    public func refreshHealth() {
        guard case .running = store.status,
              let healthURL = url?.appendingPathComponent("health") else { return }
        store.health = .checking
        healthTask?.cancel()
        healthTask = Task { [weak self] in
            do {
                let (data, response) = try await URLSession.shared.data(from: healthURL)
                try Task.checkCancellation()
                guard let http = response as? HTTPURLResponse,
                      http.statusCode == 200,
                      let object = try JSONSerialization.jsonObject(with: data)
                        as? [String: Any],
                      object["status"] as? String == "ok" else {
                    throw URLError(.badServerResponse)
                }
                guard let self else { return }
                self.store.health = .healthy
                self.healthTask = nil
            } catch is CancellationError {
            } catch {
                guard let self else { return }
                self.store.health = .unreachable
                self.recordError("Health check failed: \(error)")
                self.healthTask = nil
            }
        }
    }

    private func apply(_ activity: ServerCoordinatorActivity) {
        store.activeRequests = activity.activeRequests
        store.queuedRequests = activity.queuedRequests
    }

    private func recordError(_ message: String) {
        store.recordError(message)
    }

    private func finishStopped() {
        store.status = .stopped
        store.health = .unknown
        store.boundPort = nil
        store.modelID = nil
        store.activeRequests = 0
        store.queuedRequests = 0
        operationTask = nil
    }
}
