import Foundation
import Metal
import Testing

@testable import TUFFEngine

@Suite struct SpeculativeDecodingTests {
    @Test func allCandidatesAcceptedUsesBonusPrediction() throws {
        let proposal: [Int32] = [42, 17, 81, 90]
        let target: [Int32] = [42, 17, 81, 90, 123]

        let decision = try SpeculativeGreedyAcceptance.evaluate(
            proposedTokenIDs: proposal,
            targetTokenIDs: target)

        #expect(decision.acceptedDraftCount == 4)
        #expect(decision.firstMismatchIndex == nil)
        #expect(decision.allDraftTokensAccepted)
        #expect(decision.nextTargetToken == 123)
    }

    @Test func firstCandidateRejectedDoesNotAcceptDraftTokens() throws {
        let decision = try SpeculativeGreedyAcceptance.evaluate(
            proposedTokenIDs: [42, 17, 81],
            targetTokenIDs: [65, 17, 81, 90])

        #expect(decision.acceptedDraftCount == 0)
        #expect(decision.firstMismatchIndex == 0)
        #expect(decision.nextTargetToken == 65)
    }

    @Test func partialPrefixStopsAtFirstMismatch() throws {
        let decision = try SpeculativeGreedyAcceptance.evaluate(
            proposedTokenIDs: [42, 17, 81, 90],
            targetTokenIDs: [42, 17, 65, 12, 99])

        #expect(decision.acceptedDraftCount == 2)
        #expect(decision.firstMismatchIndex == 2)
        #expect(decision.nextTargetToken == 65)
    }

    @Test func malformedPredictionCountIsRejected() {
        #expect(throws: SpeculativeDecodingError.invalidPredictionCount(
            expected: 3, actual: 2)) {
            try SpeculativeGreedyAcceptance.evaluate(
                proposedTokenIDs: [1, 2],
                targetTokenIDs: [1, 2])
        }
    }

    @Test func emptyProposalIsRejected() {
        #expect(throws: SpeculativeDecodingError.emptyProposal) {
            try SpeculativeGreedyAcceptance.evaluate(
                proposedTokenIDs: [], targetTokenIDs: [])
        }
    }

    @Test func verifierTransactionCommitsOnlyAcceptedPrefix() async throws {
        let context = try MetalContext()
        let verifier = TransactionalVerifier(
            targetTokenIDs: [10, 20, 99, 40, 50])
        let proposal = DraftProposal(tokenIDs: [10, 20, 30, 40])
        let decoder = SpeculativeDecoder(maximumBlockTokens: 8)
        let plan = try await decoder.verifyGreedyRound(
            proposal: proposal,
            startPosition: 12,
            verifier: verifier,
            into: try #require(context.device.makeBuffer(
                length: 4, options: .storageModeShared)))

        #expect(plan.decision.acceptedDraftCount == 2)
        #expect(verifier.position == 16)
        try verifier.commitSpeculativePrefix(plan.decision.acceptedDraftCount)
        #expect(verifier.position == 14)
        #expect(verifier.commits == [2])
    }

    @Test func zeroCommitDiscardsEntireSpeculativeTail() async throws {
        let context = try MetalContext()
        let verifier = TransactionalVerifier(
            targetTokenIDs: [99, 20, 30, 40])
        _ = try await verifier.verifySpeculativeBlock(
            tokens: [10, 20, 30],
            startPosition: 7,
            into: try #require(context.device.makeBuffer(
                length: 4, options: .storageModeShared)))

        try verifier.commitSpeculativePrefix(0)
        #expect(verifier.position == 7)
        #expect(verifier.commits == [0])
    }

    @Test func commitCannotExceedProcessedBlock() async throws {
        let context = try MetalContext()
        let verifier = TransactionalVerifier(targetTokenIDs: [1, 2, 3])
        _ = try await verifier.verifySpeculativeBlock(
            tokens: [1, 2],
            startPosition: 4,
            into: try #require(context.device.makeBuffer(
                length: 4, options: .storageModeShared)))

        #expect(throws: SpeculativeDecodingError.invalidCommitCount(
            requested: 3, processed: 2)) {
            try verifier.commitSpeculativePrefix(3)
        }
    }

    @Test func decoderRejectsProposalLargerThanBound() async throws {
        let context = try MetalContext()
        let verifier = TransactionalVerifier(targetTokenIDs: [1, 2, 3, 4])
        let decoder = SpeculativeDecoder(maximumBlockTokens: 2)

        await #expect(throws: SpeculativeDecodingError.invalidBlockSize(
            requested: 3, maximum: 2)) {
            _ = try await decoder.verifyGreedyRound(
                proposal: DraftProposal(tokenIDs: [1, 2, 3]),
                startPosition: 0,
                verifier: verifier,
                into: try #require(context.device.makeBuffer(
                    length: 4, options: .storageModeShared)))
        }
        #expect(verifier.verifyCalls == 0)
    }

    @Test func adaptiveControllerStartsSmallAndGrowsAfterProfitableRounds() {
        var controller = SpeculativeDecodeController(
            config: SpeculativeDecodeConfig(mode: .auto, draftTokens: 8))

        #expect(controller.blockSize(remainingTokens: 32) == 2)
        controller.record(proposedTokens: 2,
                          acceptedTokens: 2,
                          verificationWallNanos: 1,
                          draftWallNanos: 0,
                          boundaryAdvanceNanos: 100)
        #expect(controller.nextBlockTokens == 4)
        controller.record(proposedTokens: 4,
                          acceptedTokens: 4,
                          verificationWallNanos: 1,
                          draftWallNanos: 0,
                          boundaryAdvanceNanos: 100)
        #expect(controller.nextBlockTokens == 8)
        #expect(controller.isEnabled)
    }

    @Test func adaptiveControllerDisablesAfterZeroAcceptance() {
        var controller = SpeculativeDecodeController(
            config: SpeculativeDecodeConfig(mode: .auto, draftTokens: 8))

        controller.record(proposedTokens: 4,
                          acceptedTokens: 0,
                          verificationWallNanos: 1_000,
                          draftWallNanos: 1,
                          boundaryAdvanceNanos: 100)

        #expect(controller.disabled)
        #expect(controller.blockSize(remainingTokens: 8) == 0)
    }

    final class TransactionalVerifier: SpeculativeVerificationRunner,
        @unchecked Sendable {
        let targetTokenIDs: [Int32]
        private(set) var position = 0
        private(set) var verifyCalls = 0
        private(set) var commits: [Int] = []
        private var activeStart: Int?
        private var activeCount = 0

        init(targetTokenIDs: [Int32]) {
            self.targetTokenIDs = targetTokenIDs
        }

        func reset() {
            position = 0
            activeStart = nil
            activeCount = 0
            verifyCalls = 0
            commits.removeAll()
        }

        func produce(token _: Int32, position: Int, into _: MTLBuffer) async throws {
            self.position = position + 1
        }

        func verifySpeculativeBlock(tokens: [Int32],
                                    startPosition: Int,
                                    into _: MTLBuffer) async throws
            -> SpeculativeVerificationResult {
            verifyCalls += 1
            activeStart = startPosition
            activeCount = tokens.count
            position = startPosition + tokens.count
            return SpeculativeVerificationResult(
                startPosition: startPosition,
                proposedTokenIDs: tokens,
                targetTokenIDs: Array(targetTokenIDs.prefix(tokens.count + 1)),
                processedTokens: tokens.count,
                newPosition: startPosition + tokens.count)
        }

        func commitSpeculativePrefix(_ count: Int) throws {
            guard activeStart != nil else {
                throw SpeculativeDecodingError.noActiveTransaction
            }
            guard (0...activeCount).contains(count) else {
                throw SpeculativeDecodingError.invalidCommitCount(
                    requested: count, processed: activeCount)
            }
            position = activeStart! + count
            commits.append(count)
            activeStart = nil
            activeCount = 0
        }
    }
}
