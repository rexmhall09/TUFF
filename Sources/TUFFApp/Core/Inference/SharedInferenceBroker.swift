import Foundation
import Synchronization

public enum SharedInferenceConsumer: String, Sendable, Equatable {
    case chat
    case server
    case lifecycle
}

public struct SharedInferenceActivity: Sendable, Equatable {
    public var activeConsumer: SharedInferenceConsumer?
    public var queuedChatRequests: Int
    public var queuedServerRequests: Int
    public var queuedLifecycleOperations: Int

    public init(
        activeConsumer: SharedInferenceConsumer? = nil,
        queuedChatRequests: Int = 0,
        queuedServerRequests: Int = 0,
        queuedLifecycleOperations: Int = 0
    ) {
        self.activeConsumer = activeConsumer
        self.queuedChatRequests = queuedChatRequests
        self.queuedServerRequests = queuedServerRequests
        self.queuedLifecycleOperations = queuedLifecycleOperations
    }
}

private final class SharedInferenceActivityBox: Sendable {
    struct State: Sendable {
        var snapshot = SharedInferenceActivity()
        var handler: (@Sendable (SharedInferenceActivity) -> Void)?
    }

    let state = Mutex(State())

    func publish(_ snapshot: SharedInferenceActivity) {
        let handler = state.withLock { state in
            state.snapshot = snapshot
            return state.handler
        }
        handler?(snapshot)
    }
}

private final class SharedInferenceChatJobs: Sendable {
    let ids = Mutex<Set<UUID>>([])
}

private actor SharedInferenceAdmission {
    struct Lease: Sendable, Equatable {
        let id: UUID
        let consumer: SharedInferenceConsumer
    }

    private struct Waiter {
        let lease: Lease
        let continuation: CheckedContinuation<Lease, Error>
    }

    private var active: Lease?
    private var waiters: [Waiter] = []
    private var cancelled: Set<UUID> = []
    private var cancellationRequested: Set<UUID> = []
    private let onActivity: @Sendable (SharedInferenceActivity) -> Void

    init(onActivity: @escaping @Sendable (SharedInferenceActivity) -> Void) {
        self.onActivity = onActivity
    }

    func acquire(id: UUID, consumer: SharedInferenceConsumer) async throws -> Lease {
        let lease = Lease(id: id, consumer: consumer)
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                if cancelled.remove(id) != nil || Task.isCancelled {
                    continuation.resume(throwing: CancellationError())
                } else if active == nil {
                    active = lease
                    publishActivity()
                    continuation.resume(returning: lease)
                } else {
                    waiters.append(Waiter(lease: lease, continuation: continuation))
                    publishActivity()
                }
            }
        } onCancel: {
            Task { await self.cancel(id: id, cancelActive: {}) }
        }
    }

    func release(_ lease: Lease) {
        guard active == lease else { return }
        active = nil
        cancellationRequested.remove(lease.id)
        admitNext()
        publishActivity()
    }

    /// Cancels one broker job without ever targeting a different active job.
    /// An active job keeps its lease until the underlying stream acknowledges
    /// cancellation, so the next Chat or Server request cannot start early.
    func cancel(id: UUID, cancelActive: @Sendable () -> Void) {
        if active?.id == id {
            guard cancellationRequested.insert(id).inserted else { return }
            cancelActive()
            return
        }
        if let index = waiters.firstIndex(where: { $0.lease.id == id }) {
            let waiter = waiters.remove(at: index)
            cancelled.remove(id)
            waiter.continuation.resume(throwing: CancellationError())
            publishActivity()
            return
        }
        cancelled.insert(id)
    }

    private func admitNext() {
        guard active == nil else { return }
        while !waiters.isEmpty {
            let waiter = waiters.removeFirst()
            if cancelled.remove(waiter.lease.id) != nil {
                waiter.continuation.resume(throwing: CancellationError())
                continue
            }
            active = waiter.lease
            waiter.continuation.resume(returning: waiter.lease)
            return
        }
    }

    private func publishActivity() {
        onActivity(SharedInferenceActivity(
            activeConsumer: active?.consumer,
            queuedChatRequests: waiters.count { $0.lease.consumer == .chat },
            queuedServerRequests: waiters.count { $0.lease.consumer == .server },
            queuedLifecycleOperations: waiters.count {
                $0.lease.consumer == .lifecycle
            }))
    }
}

/// The app's single admission point for model lifecycle and generation work.
/// Chat and the loopback server share the same decode-service client; every
/// operation is FIFO and one active lease is held until its service response
/// reaches a terminal event.
public final class SharedInferenceBroker: AppModelLifecycleClient,
    AppInferenceMemoryReporting, AppInferenceTranscriptReporting, @unchecked Sendable {
    private let client: any AppModelLifecycleClient
    private let admission: SharedInferenceAdmission
    private let activityBox: SharedInferenceActivityBox
    private let chatJobs = SharedInferenceChatJobs()
    private let fallbackTranscriptMailbox = GenerationTranscriptMailbox()

    public init(client: any AppModelLifecycleClient) {
        let activityBox = SharedInferenceActivityBox()
        self.client = client
        self.activityBox = activityBox
        admission = SharedInferenceAdmission { snapshot in
            activityBox.publish(snapshot)
        }
    }

    public var activity: SharedInferenceActivity {
        activityBox.state.withLock { $0.snapshot }
    }

    public func setActivityHandler(
        _ handler: (@Sendable (SharedInferenceActivity) -> Void)?
    ) {
        let snapshot = activityBox.state.withLock { state in
            state.handler = handler
            return state.snapshot
        }
        handler?(snapshot)
    }

    public var currentInferenceMemoryBytes: UInt64? {
        (client as? any AppInferenceMemoryReporting)?.currentInferenceMemoryBytes
    }

    public var currentInferenceResidentBytes: UInt64? {
        (client as? any AppInferenceMemoryReporting)?.currentInferenceResidentBytes
    }

    public var currentInferenceTowerBytes: UInt64? {
        (client as? any AppInferenceMemoryReporting)?.currentInferenceTowerBytes
    }

    public var generationTranscriptMailbox: GenerationTranscriptMailbox {
        (client as? any AppInferenceTranscriptReporting)?.generationTranscriptMailbox
            ?? fallbackTranscriptMailbox
    }

    public func ensureLoaded(
        modelDirectory: URL,
        maxContextTokens: Int,
        options: AppRuntimeOptions,
        forceLogitsHead: Bool,
        onState: @escaping @Sendable (AppModelLoadState) -> Void
    ) async throws {
        let lease = try await admission.acquire(
            id: UUID(), consumer: .lifecycle)
        do {
            try await client.ensureLoaded(
                modelDirectory: modelDirectory,
                maxContextTokens: maxContextTokens,
                options: options,
                forceLogitsHead: forceLogitsHead,
                onState: onState)
            await admission.release(lease)
        } catch {
            await admission.release(lease)
            throw error
        }
    }

    public func unload() async {
        do {
            let lease = try await admission.acquire(
                id: UUID(), consumer: .lifecycle)
            await client.unload()
            await admission.release(lease)
        } catch {
            // A cancelled owner no longer wants this lifecycle operation.
        }
    }

    public func generate(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        generationStream(request, consumer: .chat)
    }

    public func generateForServer(_ request: AppGenerationRequest)
        -> AsyncThrowingStream<AppInferenceEvent, Error> {
        generationStream(request, consumer: .server)
    }

    /// AppModel's Cancel action is scoped to Chat. Server work is cancelled by
    /// the server request or shutdown path, never by this method.
    public func cancel() {
        let ids = chatJobs.ids.withLock { $0 }
        for id in ids {
            Task { [client, admission] in
                await admission.cancel(id: id) { client.cancel() }
            }
        }
    }

    private func generationStream(
        _ request: AppGenerationRequest,
        consumer: SharedInferenceConsumer
    ) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        let id = UUID()
        if consumer == .chat {
            _ = chatJobs.ids.withLock { $0.insert(id) }
        }

        return AsyncThrowingStream { [client, admission, chatJobs] continuation in
            Task {
                defer {
                    if consumer == .chat {
                        _ = chatJobs.ids.withLock { $0.remove(id) }
                    }
                }
                do {
                    let lease = try await admission.acquire(
                        id: id, consumer: consumer)
                    do {
                        for try await event in client.generate(request) {
                            continuation.yield(event)
                        }
                        await admission.release(lease)
                        continuation.finish()
                    } catch {
                        await admission.release(lease)
                        continuation.finish(throwing: error)
                    }
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { termination in
                guard case .cancelled = termination else { return }
                Task {
                    await admission.cancel(id: id) { client.cancel() }
                }
            }
        }
    }
}
