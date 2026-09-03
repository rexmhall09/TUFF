/// A tiny tokenizer-independent drafter for the first end-to-end experiment.
/// It finds a repeated suffix in the existing token history and proposes the
/// tokens that followed the earlier occurrence. It has no model state or
/// weights, so a low acceptance rate measures the verifier honestly rather
/// than hiding a second neural model's cost in the experiment.
public final class PromptLookupDraftTokenProducer: DraftTokenProducer,
    @unchecked Sendable {
    public let minimumNGram: Int
    public let maximumNGram: Int
    public let maximumSearchTokens: Int

    public init(minimumNGram: Int = 2,
                maximumNGram: Int = 8,
                maximumSearchTokens: Int = 4096) {
        precondition(minimumNGram > 0)
        precondition(maximumNGram >= minimumNGram)
        precondition(maximumSearchTokens > 0)
        self.minimumNGram = minimumNGram
        self.maximumNGram = maximumNGram
        self.maximumSearchTokens = maximumSearchTokens
    }

    public func reset() {}

    public func propose(history: [Int32],
                        maxTokens: Int,
                        startPosition _: Int) async throws -> DraftProposal {
        guard maxTokens > 0, history.count >= minimumNGram + 1 else {
            return DraftProposal(tokenIDs: [])
        }

        let suffixLimit = min(maximumNGram, history.count - 1)
        for nGram in stride(from: suffixLimit,
                            through: minimumNGram,
                            by: -1) {
            let suffixStart = history.count - nGram
            let searchStart = max(0, suffixStart - maximumSearchTokens)
            if suffixStart <= searchStart { continue }

            // Prefer the occurrence with the longest available continuation,
            // not merely the newest occurrence. The newest match is often
            // adjacent to the current suffix and can contribute only one or
            // two tokens even though an older identical context has a full
            // block behind it.
            var bestCandidateStart: Int?
            var bestContinuationCount = 0
            for candidateStart in stride(from: suffixStart - 1,
                                          through: searchStart,
                                          by: -1) {
                var matches = true
                for offset in 0..<nGram
                    where history[candidateStart + offset]
                    != history[suffixStart + offset] {
                    matches = false
                    break
                }
                guard matches else { continue }

                let continuationStart = candidateStart + nGram
                guard continuationStart < history.count else { continue }
                let continuationCount = min(maxTokens,
                                            history.count - continuationStart)
                if continuationCount > bestContinuationCount
                    || (continuationCount == bestContinuationCount
                        && candidateStart > (bestCandidateStart ?? -1)) {
                    bestCandidateStart = candidateStart
                    bestContinuationCount = continuationCount
                }
            }
            if let bestCandidateStart, bestContinuationCount > 0 {
                let continuationStart = bestCandidateStart + nGram
                return DraftProposal(tokenIDs: Array(
                    history[continuationStart..<(continuationStart + bestContinuationCount)]))
            }
        }
        return DraftProposal(tokenIDs: [])
    }
}
