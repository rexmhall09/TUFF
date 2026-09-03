import Metal

public enum SpeculativeDecodeMode: String, Codable, Sendable, Equatable {
    case off
    case greedy
}

/// Experimental speculative-decoding controls. It is disabled by default and
/// currently enables only greedy verification. Other sampling configurations
/// intentionally continue through the ordinary target path.
public struct SpeculativeDecodeConfig: Codable, Sendable, Equatable {
    public var mode: SpeculativeDecodeMode
    public var draftTokens: Int

    public init(mode: SpeculativeDecodeMode = .off,
                draftTokens: Int = 4) {
        self.mode = mode
        self.draftTokens = draftTokens
    }

    public static let off = SpeculativeDecodeConfig()
    public var isEnabled: Bool { mode != .off }
}

/// End-to-end counters returned with a raw completion. These are kept separate
/// from target-runner counters so scripted and future remote drafters can report
/// the policy economics without knowing Metal or model internals.
public struct SpeculativeDecodeMetrics: Sendable, Equatable {
    public let rounds: Int
    public let proposedTokens: Int
    public let acceptedTokens: Int
    public let rejectedTokens: Int
    public let correctionTokens: Int
    public let verificationBlocks: Int
    public let verificationTokens: Int
    public let verificationWallNanos: UInt64
    public let verificationCommandBuffers: UInt64
    public let verificationExpertReads: UInt64
    public let verificationExpertBytes: UInt64
    public let verificationExpertCacheHits: UInt64
    public let verificationExpertCacheMisses: UInt64
    public let draftWallNanos: UInt64
    public let normalFallbackDecodes: Int

    public init(rounds: Int = 0,
                proposedTokens: Int = 0,
                acceptedTokens: Int = 0,
                rejectedTokens: Int = 0,
                correctionTokens: Int = 0,
                verificationBlocks: Int = 0,
                verificationTokens: Int = 0,
                verificationWallNanos: UInt64 = 0,
                verificationCommandBuffers: UInt64 = 0,
                verificationExpertReads: UInt64 = 0,
                verificationExpertBytes: UInt64 = 0,
                verificationExpertCacheHits: UInt64 = 0,
                verificationExpertCacheMisses: UInt64 = 0,
                draftWallNanos: UInt64 = 0,
                normalFallbackDecodes: Int = 0) {
        self.rounds = rounds
        self.proposedTokens = proposedTokens
        self.acceptedTokens = acceptedTokens
        self.rejectedTokens = rejectedTokens
        self.correctionTokens = correctionTokens
        self.verificationBlocks = verificationBlocks
        self.verificationTokens = verificationTokens
        self.verificationWallNanos = verificationWallNanos
        self.verificationCommandBuffers = verificationCommandBuffers
        self.verificationExpertReads = verificationExpertReads
        self.verificationExpertBytes = verificationExpertBytes
        self.verificationExpertCacheHits = verificationExpertCacheHits
        self.verificationExpertCacheMisses = verificationExpertCacheMisses
        self.draftWallNanos = draftWallNanos
        self.normalFallbackDecodes = normalFallbackDecodes
    }

    public var acceptanceRate: Double {
        guard proposedTokens > 0 else { return 0 }
        return Double(acceptedTokens) / Double(proposedTokens)
    }
}

/// The generation-policy half of one speculative round. It deliberately does
/// not commit KV: RawCompletion must inspect each accepted token for stop,
/// cancellation, and max-token boundaries before deciding how much of the
/// verifier transaction is safe to keep.
public struct SpeculativeRoundPlan: Sendable, Equatable {
    public let verification: SpeculativeVerificationResult
    public let decision: SpeculativeGreedyDecision

    public init(verification: SpeculativeVerificationResult,
                decision: SpeculativeGreedyDecision) {
        self.verification = verification
        self.decision = decision
    }
}

public struct SpeculativeDecoder: Sendable {
    public let maximumBlockTokens: Int

    public init(maximumBlockTokens: Int = 8) {
        precondition(maximumBlockTokens > 0,
                     "maximum speculative block size must be positive")
        self.maximumBlockTokens = maximumBlockTokens
    }

    public func verifyGreedyRound(
        proposal: DraftProposal,
        startPosition: Int,
        verifier: any SpeculativeVerificationRunner,
        into logits: MTLBuffer
    ) async throws -> SpeculativeRoundPlan {
        guard proposal.count > 0 else {
            throw SpeculativeDecodingError.emptyProposal
        }
        guard proposal.count <= maximumBlockTokens else {
            throw SpeculativeDecodingError.invalidBlockSize(
                requested: proposal.count,
                maximum: maximumBlockTokens)
        }
        let result = try await verifier.verifySpeculativeBlock(
            tokens: proposal.tokenIDs,
            startPosition: startPosition,
            into: logits)
        guard result.startPosition == startPosition else {
            throw SpeculativeDecodingError.invalidStartPosition(
                expected: startPosition,
                actual: result.startPosition)
        }
        guard result.proposedTokenIDs == proposal.tokenIDs else {
            throw SpeculativeDecodingError.invalidPredictionCount(
                expected: proposal.count,
                actual: result.proposedTokenIDs.count)
        }
        let decision = try SpeculativeGreedyAcceptance.evaluate(
            proposedTokenIDs: proposal.tokenIDs,
            targetTokenIDs: result.targetTokenIDs)
        return SpeculativeRoundPlan(verification: result, decision: decision)
    }
}
