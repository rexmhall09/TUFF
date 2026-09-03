import Metal

/// Counters measured by one target verification block. Zero means that a
/// backend does not expose that counter, not that the operation was free.
public struct SpeculativeVerificationMetrics: Sendable, Equatable {
    public let wallNanos: UInt64
    public let targetCommandBuffers: UInt64
    public let expertReads: UInt64
    public let expertBytes: UInt64
    public let expertCacheHits: UInt64
    public let expertCacheMisses: UInt64

    public init(wallNanos: UInt64 = 0,
                targetCommandBuffers: UInt64 = 0,
                expertReads: UInt64 = 0,
                expertBytes: UInt64 = 0,
                expertCacheHits: UInt64 = 0,
                expertCacheMisses: UInt64 = 0) {
        self.wallNanos = wallNanos
        self.targetCommandBuffers = targetCommandBuffers
        self.expertReads = expertReads
        self.expertBytes = expertBytes
        self.expertCacheHits = expertCacheHits
        self.expertCacheMisses = expertCacheMisses
    }
}

/// Target predictions for a verified block.
///
/// `targetTokenIDs[0]` is the target prediction at `startPosition`, before the
/// first proposed token. Each following entry is the prediction after the
/// corresponding proposed token has been processed. An N-token block therefore
/// has N+1 predictions, which provides both the first correction token and the
/// all-accepted bonus token without another target pass.
public struct SpeculativeVerificationResult: Sendable, Equatable {
    public let startPosition: Int
    public let proposedTokenIDs: [Int32]
    public let targetTokenIDs: [Int32]
    public let processedTokens: Int
    public let newPosition: Int
    public let metrics: SpeculativeVerificationMetrics

    public init(startPosition: Int,
                proposedTokenIDs: [Int32],
                targetTokenIDs: [Int32],
                processedTokens: Int,
                newPosition: Int,
                metrics: SpeculativeVerificationMetrics = .init()) {
        precondition(startPosition >= 0, "speculative start position must be non-negative")
        precondition(processedTokens == proposedTokenIDs.count,
                     "verification must process every proposed token")
        precondition(targetTokenIDs.count == proposedTokenIDs.count + 1,
                     "verification must return one boundary prediction per candidate plus a bonus")
        precondition(newPosition == startPosition + processedTokens,
                     "verification newPosition must include the processed candidate count")
        self.startPosition = startPosition
        self.proposedTokenIDs = proposedTokenIDs
        self.targetTokenIDs = targetTokenIDs
        self.processedTokens = processedTokens
        self.newPosition = newPosition
        self.metrics = metrics
    }
}

/// A target runner that can process a bounded candidate block as one explicit
/// transaction. The verifier owns the speculative tail until
/// `commitSpeculativePrefix(_:)` is called. Passing zero discards the complete
/// block; passing `processedTokens` commits it all. Intermediate values retain
/// only the accepted prefix logically, while physical tail bytes may be reused
/// by the next correction token.
public protocol SpeculativeVerificationRunner: LogitProducer {
    func verifySpeculativeBlock(tokens: [Int32],
                                startPosition: Int,
                                into logits: MTLBuffer) async throws
        -> SpeculativeVerificationResult

    func commitSpeculativePrefix(_ count: Int) throws
}

/// The first POC's greedy acceptance rule. For a mismatch, the target token at
/// that boundary is the correction token. If every candidate matches, the
/// target token after the block is the bonus token.
public struct SpeculativeGreedyDecision: Sendable, Equatable {
    public let acceptedDraftCount: Int
    public let firstMismatchIndex: Int?
    public let nextTargetToken: Int32

    public var allDraftTokensAccepted: Bool { firstMismatchIndex == nil }

    public init(acceptedDraftCount: Int,
                firstMismatchIndex: Int?,
                nextTargetToken: Int32) {
        self.acceptedDraftCount = acceptedDraftCount
        self.firstMismatchIndex = firstMismatchIndex
        self.nextTargetToken = nextTargetToken
    }
}

public enum SpeculativeGreedyAcceptance {
    public static func evaluate(
        proposedTokenIDs: [Int32],
        targetTokenIDs: [Int32]
    ) throws -> SpeculativeGreedyDecision {
        guard !proposedTokenIDs.isEmpty else {
            throw SpeculativeDecodingError.emptyProposal
        }
        guard targetTokenIDs.count == proposedTokenIDs.count + 1 else {
            throw SpeculativeDecodingError.invalidPredictionCount(
                expected: proposedTokenIDs.count + 1,
                actual: targetTokenIDs.count)
        }

        for index in proposedTokenIDs.indices {
            guard proposedTokenIDs[index] == targetTokenIDs[index] else {
                return SpeculativeGreedyDecision(
                    acceptedDraftCount: index,
                    firstMismatchIndex: index,
                    nextTargetToken: targetTokenIDs[index])
            }
        }
        return SpeculativeGreedyDecision(
            acceptedDraftCount: proposedTokenIDs.count,
            firstMismatchIndex: nil,
            nextTargetToken: targetTokenIDs[proposedTokenIDs.count])
    }
}

public enum SpeculativeDecodingError: Error, Equatable, CustomStringConvertible {
    case emptyProposal
    case invalidPredictionCount(expected: Int, actual: Int)
    case invalidBlockSize(requested: Int, maximum: Int)
    case invalidStartPosition(expected: Int, actual: Int)
    case noActiveTransaction
    case invalidCommitCount(requested: Int, processed: Int)

    public var description: String {
        switch self {
        case .emptyProposal:
            return "speculative proposal must contain at least one token"
        case .invalidPredictionCount(let expected, let actual):
            return "speculative verifier returned \(actual) predictions; expected \(expected)"
        case .invalidBlockSize(let requested, let maximum):
            return "speculative block has \(requested) tokens; maximum is \(maximum)"
        case .invalidStartPosition(let expected, let actual):
            return "speculative verifier expected start position \(expected), got \(actual)"
        case .noActiveTransaction:
            return "no speculative verification transaction is active"
        case .invalidCommitCount(let requested, let processed):
            return "cannot commit \(requested) speculative tokens; transaction processed \(processed)"
        }
    }
}
