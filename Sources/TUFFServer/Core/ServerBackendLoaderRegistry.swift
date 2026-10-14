import Foundation
import Synchronization
import TUFFModelCatalog
import TUFFRuntimeManagement
import TUFFRuntimeDescriptions
import TUFFAppCore

/// Hosts register a downloaded-helper loader; engine executables can register
/// the in-process adapter. HTTP parsing and routing never require Metal.
public enum ServerBackendLoaderRegistry {
    public typealias Loader = @Sendable (TUFFModelDescriptor, URL, ServerInstalledModels) async throws -> any ServerInferenceBackend
    private static let loader = Mutex<Loader?>(nil)

    public static func register(_ value: @escaping Loader) {
        loader.withLock { $0 = value }
    }

    static func load(descriptor: TUFFModelDescriptor, directory: URL,
                     installed: ServerInstalledModels) async throws -> any ServerInferenceBackend {
        if let value = loader.withLock({ $0 }) {
            return try await value(descriptor, directory, installed)
        }
        let installation = try await InferenceEngineEnvironment.acquireModel(at: directory)
        func launch(_ installation: InferenceEngineInstallation) async throws -> any ServerInferenceBackend {
            try Task.checkCancellation()
            guard let automatic = installed.automaticPlan(for: descriptor) else {
                throw ServerRequestError.modelUnavailable("No safe automatic allocation is available for this model on this Mac.")
            }
            let densePolicy = automatic.denseResidencyPolicy
            ServerLoadedPlans.set(automatic, for: directory)
            return try await ProcessBackedServerModelSession.load(
                executableURL: installation.executableURL, engineLockFileURL: installation.lockFileURL,
                architectureID: descriptor.architecture.id.rawValue,
                expectedModelID: installed.engineReportedModelID(for: descriptor),
                request: ServerEngineLoadRequest(modelPath: directory.path,
                    maxContextTokens: automatic.contextTokens,
                    retainedConversationBytes: automatic.retainedConversationBytes ?? 0,
                    runtimeConfiguration: installed.runtimeConfiguration(for: descriptor, resolvedPlan: automatic),
                    denseResidencyPolicy: densePolicy),
                installation: installation)
        }
        do { return try await launch(installation) }
        catch {
            await installation.release()
            try Task.checkCancellation()
            guard let fallback = try? await InferenceEngineEnvironment.acquireKnownGoodModel(
                at: directory, excluding: installation.identity) else { throw error }
            do { return try await launch(fallback) }
            catch { await fallback.release(); throw error }
        }
    }
}
