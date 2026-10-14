import Foundation
import TUFFEngine
import TUFFServerCore

/// Installed by engine executables. This adapter keeps the previous server
/// runner intact while the host registers its process-backed implementation.
public enum ServerEngineRuntime {
    private final class SharedContext: @unchecked Sendable {
        private let lock = NSLock()
        private var context: MetalContext?
        func get() throws -> MetalContext {
            lock.lock()
            defer { lock.unlock() }
            if let context { return context }
            let created = try MetalContext()
            context = created
            return created
        }
    }

    public static func registerInProcessLoader() {
        let shared = SharedContext()
        ServerBackendLoaderRegistry.register { descriptor, directory, installed in
            let session = try await ServerModelSession.load(modelDirectory: directory,
                maxContext: installed.inferencePlan(for: descriptor).contextTokens,
                promptCacheMode: .singlePrefix,
                retainedConversationBytes: installed.retainedConversationBytes(for: descriptor),
                runtimeConfiguration: installed.runtimeConfiguration(for: descriptor),
                context: try shared.get(), integrityPolicy: .sizeCheckTrustedReceipt)
            guard session.defaultModelID == installed.engineReportedModelID(for: descriptor),
                  session.modelVariant.rawValue == descriptor.architecture.id.rawValue else {
                throw ServerRequestError.modelUnavailable("The installed pack does not match the requested model.")
            }
            return session
        }
    }
}
