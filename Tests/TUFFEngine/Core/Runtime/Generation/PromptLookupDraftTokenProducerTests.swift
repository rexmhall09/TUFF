import Testing
@testable import TUFFEngine

@Suite struct PromptLookupDraftTokenProducerTests {
    @Test func proposesContinuationAfterTheLongestRepeatedSuffix() async throws {
        let drafter = PromptLookupDraftTokenProducer()
        let proposal = try await drafter.propose(
            history: [1, 2, 3, 4, 2, 3, 4], maxTokens: 4, startPosition: 7)
        #expect(proposal.tokenIDs == [2, 3, 4])
    }

    @Test func emptyWhenHistoryHasNoRepeatOrOnlyOneToken() async throws {
        let drafter = PromptLookupDraftTokenProducer()
        let one = try await drafter.propose(
            history: [1], maxTokens: 4, startPosition: 1)
        let noRepeat = try await drafter.propose(
            history: [1, 2, 3, 4], maxTokens: 4, startPosition: 4)
        #expect(one.isEmpty)
        #expect(noRepeat.isEmpty)
    }

    @Test func proposalIsBoundedByRequestedBlockSize() async throws {
        let drafter = PromptLookupDraftTokenProducer()
        let proposal = try await drafter.propose(
            history: [1, 2, 3, 4, 1, 2, 3, 4], maxTokens: 2, startPosition: 8)
        #expect(proposal.count == 2)
        #expect(proposal.tokenIDs == [1, 2])
    }

    @Test func prefersAnOlderMatchWithAFullerContinuation() async throws {
        let drafter = PromptLookupDraftTokenProducer()
        let proposal = try await drafter.propose(
            history: [1, 2, 3, 1, 2, 3, 4, 5, 6, 1, 2, 3],
            maxTokens: 7,
            startPosition: 12)

        #expect(proposal.tokenIDs == [1, 2, 3, 4, 5, 6, 1])
    }
}
