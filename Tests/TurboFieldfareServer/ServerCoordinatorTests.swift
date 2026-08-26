import Testing
import Synchronization
@testable import TurboFieldfareServerCore

private actor TestGate {
    private var continuation: CheckedContinuation<Void, Never>?

    func wait() async {
        await withCheckedContinuation { continuation = $0 }
    }

    func open() {
        continuation?.resume()
        continuation = nil
    }
}

@Suite("Server coordinator")
struct ServerCoordinatorTests {
    @Test func boundsFIFOAndRecoversAfterCancellation() async throws {
        let coordinator = ServerCoordinator(queueLimit: 1)
        let gate = TestGate()
        let active = Task {
            try await coordinator.run {
                await gate.wait()
                return 1
            }
        }
        while await !coordinator.isActive {
            await Task.yield()
        }
        let queued = Task {
            try await coordinator.run { 2 }
        }
        while await coordinator.queuedCount != 1 {
            await Task.yield()
        }
        await #expect(throws: ServerRequestError.queueFull) {
            try await coordinator.run { 3 }
        }
        queued.cancel()
        _ = try? await queued.value
        await gate.open()
        #expect(try await active.value == 1)
        #expect(await coordinator.queuedCount == 0)
    }

    @Test func reportsActiveAndQueuedRequestCounts() async throws {
        let snapshots = Mutex<[ServerCoordinatorActivity]>([])
        let coordinator = ServerCoordinator(queueLimit: 2) { activity in
            snapshots.withLock { $0.append(activity) }
        }
        let gate = TestGate()
        let active = Task {
            try await coordinator.run {
                await gate.wait()
                return 1
            }
        }
        while await !coordinator.isActive { await Task.yield() }
        let queued = Task { try await coordinator.run { 2 } }
        while await coordinator.queuedCount != 1 { await Task.yield() }
        await gate.open()
        _ = try await active.value
        _ = try await queued.value

        let values = snapshots.withLock { $0 }
        #expect(values.contains(ServerCoordinatorActivity(
            activeRequests: 1, queuedRequests: 0)))
        #expect(values.contains(ServerCoordinatorActivity(
            activeRequests: 1, queuedRequests: 1)))
        #expect(values.last == ServerCoordinatorActivity(
            activeRequests: 0, queuedRequests: 0))
    }
}
