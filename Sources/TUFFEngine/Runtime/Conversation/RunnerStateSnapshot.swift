import Foundation
import Metal

/// A copy of everything a runner carries from one token to the next, taken
/// between requests so a different conversation can use the runner and this
/// one can resume later without prefilling again.
///
/// The copy is compact: it holds only the rows a sequence has written, not the
/// runner's full allocation, so a short conversation costs little. Every byte
/// lives in one shared-storage buffer, and both directions are blit copies on
/// the runner's queue. That handles GPU-private state such as the Qwen sparse
/// indexer's keys the same way as shared KV rows.
///
/// A snapshot is only valid for the runner that produced it. `owner` records
/// that runner, and restore refuses any other one, because the layout of
/// segments depends on the runner's architecture, context and ring geometry.
public final class RunnerStateSnapshot: @unchecked Sendable {
    /// One contiguous range copied out of a runner buffer.
    struct Segment {
        let label: String
        let sourceOffset: Int
        let snapshotOffset: Int
        let length: Int
    }

    /// Host-side values that are not in Metal buffers.
    struct HostState: Equatable {
        var position: Int
        var ngramContext: [Int32]
        var ropeDelta: Int32
    }

    let owner: ObjectIdentifier
    let storage: MTLBuffer?
    let segments: [Segment]
    let host: HostState

    /// The KV position the snapshot resumes at.
    public var position: Int { host.position }
    /// Bytes held by this snapshot. This is what the conversation store
    /// charges against its memory budget.
    public var byteCount: Int { storage?.length ?? 0 }

    init(owner: ObjectIdentifier, storage: MTLBuffer?, segments: [Segment],
         host: HostState) {
        self.owner = owner
        self.storage = storage
        self.segments = segments
        self.host = host
    }
}

public enum RunnerStateSnapshotError: Error, Equatable, CustomStringConvertible {
    case unsupported(String)
    case foreignSnapshot
    case allocationFailed(Int)
    case layoutMismatch(String)

    public var description: String {
        switch self {
        case .unsupported(let reason):
            return "runner state cannot be retained: \(reason)"
        case .foreignSnapshot:
            return "the snapshot belongs to a different runner"
        case .allocationFailed(let bytes):
            return "could not allocate \(bytes) bytes for a retained conversation"
        case .layoutMismatch(let detail):
            return "retained conversation layout no longer matches the runner: \(detail)"
        }
    }
}

/// A runner whose sequence state can be copied out and back.
public protocol StateSnapshottingRunner: AnyObject {
    /// Bytes a snapshot taken now would occupy, or nil when the runner cannot
    /// snapshot its current state. Computed without allocating anything.
    var stateSnapshotByteEstimate: Int? { get }
    /// Copies the current sequence state. The runner is unchanged.
    func captureState() throws -> RunnerStateSnapshot
    /// Replaces the runner's sequence state with `snapshot`. On a throw the
    /// runner has been reset and holds no sequence.
    func restoreState(_ snapshot: RunnerStateSnapshot) throws
    /// The runner whose layout snapshots describe: this one, or the backend a
    /// wrapper forwards to. A snapshot read from disk is made out to it.
    var snapshotOwner: ObjectIdentifier { get }
}

extension StateSnapshottingRunner {
    public var snapshotOwner: ObjectIdentifier { ObjectIdentifier(self) }
}

/// A runner that can return to an earlier position of the sequence it holds.
///
/// Rows that later tokens only append after stay where they are, so going
/// back needs a copy of just the rows that are overwritten in place, such as
/// a sliding-window ring. A checkpoint holds those rows and its position.
public protocol PrefixCheckpointingRunner: AnyObject {
    /// Whether this runner can take checkpoints at all.
    var supportsPrefixCheckpoints: Bool { get }
    /// Copies what returning to the current position later needs. The runner
    /// is unchanged.
    func capturePrefixCheckpoint() throws -> RunnerStateSnapshot
    /// Returns to `checkpoint`'s position. The runner must still hold the
    /// sequence the checkpoint was taken from, at or past that position. On a
    /// throw the runner has been reset and holds no sequence.
    func rewind(to checkpoint: RunnerStateSnapshot) throws
}

/// Collects the ranges one snapshot copies, then performs the copy.
///
/// Callers describe ranges as (buffer, offset, length). The builder lays them
/// out back to back in one shared buffer and records where each landed, so
/// restore writes the same bytes back to the same places.
struct RunnerStateSnapshotBuilder {
    private(set) var ranges: [(label: String, buffer: MTLBuffer, offset: Int, length: Int)] = []

    var byteCount: Int {
        ranges.reduce(0) { $0 + Self.aligned($1.length) }
    }

    /// `label` must be unique within one snapshot; restore matches by it.
    mutating func add(_ label: String, _ buffer: MTLBuffer, offset: Int = 0, length: Int) {
        precondition(offset >= 0 && length >= 0 && offset + length <= buffer.length,
                     "snapshot range \(label) exceeds its buffer")
        guard length > 0 else { return }
        ranges.append((label, buffer, offset, length))
    }

    /// Blit offsets and sizes must be multiples of 4 for some buffer types;
    /// 16 keeps every segment aligned for any element type it holds.
    static func aligned(_ length: Int) -> Int { (length + 15) & ~15 }

    func capture(owner: AnyObject, queue: MTLCommandQueue,
                 host: RunnerStateSnapshot.HostState) throws -> RunnerStateSnapshot {
        let total = byteCount
        guard total > 0 else {
            return RunnerStateSnapshot(owner: ObjectIdentifier(owner), storage: nil,
                                       segments: [], host: host)
        }
        guard let storage = queue.device.makeBuffer(length: total, options: .storageModeShared) else {
            throw RunnerStateSnapshotError.allocationFailed(total)
        }
        storage.label = "conversation.snapshot"
        var segments: [RunnerStateSnapshot.Segment] = []
        segments.reserveCapacity(ranges.count)
        var cursor = 0
        guard let cb = queue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else { throw MetalError.noQueue }
        cb.label = "conversation.snapshot.capture"
        for range in ranges {
            blit.copy(from: range.buffer, sourceOffset: range.offset,
                      to: storage, destinationOffset: cursor, size: range.length)
            segments.append(.init(label: range.label,
                                  sourceOffset: range.offset,
                                  snapshotOffset: cursor,
                                  length: range.length))
            cursor += Self.aligned(range.length)
        }
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        try checkCommandBufferError(cb)
        return RunnerStateSnapshot(owner: ObjectIdentifier(owner), storage: storage,
                                   segments: segments, host: host)
    }

    /// Writes `snapshot` back. `ranges` must describe the same layout the
    /// snapshot was captured with, which the caller rebuilds from the
    /// snapshot's position; any difference is refused rather than copied.
    func restore(_ snapshot: RunnerStateSnapshot, queue: MTLCommandQueue) throws {
        guard snapshot.segments.count == ranges.count else {
            throw RunnerStateSnapshotError.layoutMismatch(
                "\(snapshot.segments.count) segments captured, \(ranges.count) expected")
        }
        guard !ranges.isEmpty else { return }
        guard let storage = snapshot.storage else {
            throw RunnerStateSnapshotError.layoutMismatch("snapshot holds no storage")
        }
        for (segment, range) in zip(snapshot.segments, ranges) {
            // Buffers are matched by label and range, not identity: a full
            // attention layer that grew since the capture has a new buffer
            // holding the same rows.
            guard segment.label == range.label,
                  segment.sourceOffset == range.offset,
                  segment.length == range.length else {
                throw RunnerStateSnapshotError.layoutMismatch(segment.label)
            }
        }
        guard let cb = queue.makeCommandBuffer(),
              let blit = cb.makeBlitCommandEncoder() else { throw MetalError.noQueue }
        cb.label = "conversation.snapshot.restore"
        for (segment, range) in zip(snapshot.segments, ranges) {
            blit.copy(from: storage, sourceOffset: segment.snapshotOffset,
                      to: range.buffer, destinationOffset: range.offset,
                      size: range.length)
        }
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        try checkCommandBufferError(cb)
    }
}
