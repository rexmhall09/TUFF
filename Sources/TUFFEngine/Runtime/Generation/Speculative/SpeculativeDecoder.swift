import Metal

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
