import Metal

public enum SpeculativeDecodeMode: String, Codable, Sendable, Equatable {
    case off
    case greedy
    case auto
}

/// Speculative-decoding controls. It is disabled by default and currently
/// enables only greedy verification. Other sampling configurations intentionally
/// continue through the ordinary target path.
///
/// `greedy` is a fixed-size mode for reproducible experiments. `auto` treats
/// `draftTokens` as an upper bound, starts at a conservative two-token block,
/// and adapts from measured acceptance and target time. The adaptive mode is
/// deliberately opt-in so a caller can make a clean baseline comparison.
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

/// Runtime controller for adaptive greedy speculation.
///
/// The controller uses the scalar target advance following each completed
/// round as a local cost estimate. A block is considered useful only when its
/// verification plus drafting work is below the estimated scalar cost of the
/// tokens it enabled. This is intentionally conservative: a bad first block
/// can be discarded without making the rest of a completion slower.
public struct SpeculativeDecodeController: Sendable, Equatable {
    public let mode: SpeculativeDecodeMode
    public let maximumBlockTokens: Int
    public private(set) var nextBlockTokens: Int
    public private(set) var disabled: Bool

    private var scalarNanosPerToken: Double?
    private var unprofitableRounds = 0

    public init(config: SpeculativeDecodeConfig) {
        self.mode = config.mode
        self.maximumBlockTokens = min(max(config.draftTokens, 1), 8)
        self.nextBlockTokens = config.mode == .auto
            ? min(2, min(max(config.draftTokens, 1), 8))
            : min(max(config.draftTokens, 1), 8)
        self.disabled = config.mode == .off
    }

    public var isEnabled: Bool { !disabled && mode != .off }

    /// Returns the current bounded proposal size. Zero means the adaptive
    /// policy has decided that scalar decode is safer for the rest of the run.
    public func blockSize(remainingTokens: Int) -> Int {
        guard isEnabled, remainingTokens > 0 else { return 0 }
        return min(nextBlockTokens, remainingTokens)
    }

    /// Incorporate one completed speculative round. The boundary advance is
    /// the scalar target call used to continue after the correction/bonus
    /// token; it is optional because a stop token may end the completion first.
    public mutating func record(proposedTokens: Int,
                                acceptedTokens: Int,
                                verificationWallNanos: UInt64,
                                draftWallNanos: UInt64,
                                boundaryAdvanceNanos: UInt64?) {
        guard mode == .auto, !disabled, proposedTokens > 0 else { return }

        if let boundaryAdvanceNanos {
            let sample = max(Double(boundaryAdvanceNanos), 1)
            if let scalarNanosPerToken {
                // A slow SSD read or a transient scheduler delay should not
                // make the next block grow immediately.
                self.scalarNanosPerToken = scalarNanosPerToken * 0.75 + sample * 0.25
            } else {
                scalarNanosPerToken = sample
            }
        }

        // With no accepted draft token there is no possible amortization from
        // this drafter. Disable immediately rather than paying another known
        // speculative block and then falling back anyway.
        guard acceptedTokens > 0 else {
            disabled = true
            return
        }

        let acceptance = Double(acceptedTokens) / Double(proposedTokens)
        let speculativeNanos = Double(verificationWallNanos)
            + Double(draftWallNanos)
            + Double(boundaryAdvanceNanos ?? 0)
        let scalarBaseline = scalarNanosPerToken.map {
            $0 * Double(acceptedTokens + 1)
        }
        let unprofitable = scalarBaseline.map {
            speculativeNanos > $0 * 1.05
        } ?? false
        let lowAcceptance = acceptance < 0.5

        if lowAcceptance || unprofitable {
            unprofitableRounds += 1
            if nextBlockTokens > 2 {
                nextBlockTokens = max(2, nextBlockTokens / 2)
            } else if unprofitableRounds >= 2 || lowAcceptance {
                disabled = true
            }
            return
        }

        unprofitableRounds = 0
        if acceptance >= 0.75,
           scalarBaseline.map({ speculativeNanos < $0 * 0.9 }) == true {
            nextBlockTokens = min(maximumBlockTokens,
                                  max(2, nextBlockTokens * 2))
        }
    }
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
    public let minimumVerificationBlockTokens: Int
    public let maximumVerificationBlockTokens: Int
    public let verificationWallNanos: UInt64
    public let verificationCommandBuffers: UInt64
    public let verificationExpertReads: UInt64
    public let verificationExpertBytes: UInt64
    public let verificationExpertCacheHits: UInt64
    public let verificationExpertCacheMisses: UInt64
    public let draftWallNanos: UInt64
    public let normalFallbackDecodes: Int
    public let adaptiveDisabled: Bool

    public init(rounds: Int = 0,
                proposedTokens: Int = 0,
                acceptedTokens: Int = 0,
                rejectedTokens: Int = 0,
                correctionTokens: Int = 0,
                verificationBlocks: Int = 0,
                verificationTokens: Int = 0,
                minimumVerificationBlockTokens: Int = 0,
                maximumVerificationBlockTokens: Int = 0,
                verificationWallNanos: UInt64 = 0,
                verificationCommandBuffers: UInt64 = 0,
                verificationExpertReads: UInt64 = 0,
                verificationExpertBytes: UInt64 = 0,
                verificationExpertCacheHits: UInt64 = 0,
                verificationExpertCacheMisses: UInt64 = 0,
                draftWallNanos: UInt64 = 0,
                normalFallbackDecodes: Int = 0,
                adaptiveDisabled: Bool = false) {
        self.rounds = rounds
        self.proposedTokens = proposedTokens
        self.acceptedTokens = acceptedTokens
        self.rejectedTokens = rejectedTokens
        self.correctionTokens = correctionTokens
        self.verificationBlocks = verificationBlocks
        self.verificationTokens = verificationTokens
        self.minimumVerificationBlockTokens = minimumVerificationBlockTokens
        self.maximumVerificationBlockTokens = maximumVerificationBlockTokens
        self.verificationWallNanos = verificationWallNanos
        self.verificationCommandBuffers = verificationCommandBuffers
        self.verificationExpertReads = verificationExpertReads
        self.verificationExpertBytes = verificationExpertBytes
        self.verificationExpertCacheHits = verificationExpertCacheHits
        self.verificationExpertCacheMisses = verificationExpertCacheMisses
        self.draftWallNanos = draftWallNanos
        self.normalFallbackDecodes = normalFallbackDecodes
        self.adaptiveDisabled = adaptiveDisabled
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
