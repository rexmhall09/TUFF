import Testing
import Foundation
@testable import TUFFEngine

/// Checks the n-gram row selection against mlx-vlm's own hash.
///
/// The fixture is not a transcription: `Scripts/capture_qwen4exp_ngram_rows.py`
/// constructs the reference's `Qwen4ExpNGramEmbedding` with the constants the
/// checkpoint stores, substitutes a recording stand-in for the 30 GiB table,
/// and captures the row ids the reference's own code computes.
///
/// This matters more than it looks. A row id that is wrong by one selects a
/// different embedding out of 320 million, and nothing downstream can tell —
/// the model would simply be slightly, unfixably wrong.
@Suite struct NgramRowIndexTests {

    private struct Golden: Decodable {
        let eosTokenId: Int32
        let ngramSize: Int
        let headsPerNgram: Int
        let multipliers: [Int64]
        let headOffsets: [Int64]
        let headVocabSizes: [Int64]
        let tokens: [Int32]
        let rows: [[Int64]]
    }

    private static func loadGolden() throws -> Golden? {
        guard let url = Bundle.module.url(forResource: "ngram-rows",
                                          withExtension: "json",
                                          subdirectory: "qwen4exp") else {
            Issue.record("n-gram row fixture is missing from the bundle")
            return nil
        }
        return try JSONDecoder().decode(Golden.self,
                                        from: Data(contentsOf: url))
    }

    private static func index(_ g: Golden) -> NgramRowIndex {
        NgramRowIndex(multipliers: g.multipliers,
                      headOffsets: g.headOffsets,
                      headVocabSizes: g.headVocabSizes,
                      ngramSize: g.ngramSize,
                      headsPerNgram: g.headsPerNgram,
                      eosTokenID: g.eosTokenId)
    }

    @Test func rowsMatchTheReferenceHash() throws {
        guard let g = try Self.loadGolden() else { return }
        let index = Self.index(g)
        #expect(index.heads == 16)
        #expect(index.contextLength == 2)

        let (rows, _) = index.rows(for: g.tokens,
                                   context: index.initialContext())
        #expect(rows.count == g.tokens.count * index.heads)
        for (position, expected) in g.rows.enumerated() {
            let slice = Array(rows[(position * index.heads)..<((position + 1) * index.heads)])
            #expect(slice == expected, "position \(position)")
        }
    }

    /// The fixture's token sequence contains an EOS in the middle, and the
    /// shift is segment-aware: positions right after it must read EOS rather
    /// than reach back into the previous turn. If that were ignored the rows
    /// after the boundary would differ from the reference's.
    @Test func theSegmentBoundaryIsHonoured() throws {
        guard let g = try Self.loadGolden() else { return }
        #expect(g.tokens.contains(g.eosTokenId),
                "the fixture no longer exercises a segment boundary")
        let index = Self.index(g)
        let boundary = g.tokens.firstIndex(of: g.eosTokenId)!

        let (rows, _) = index.rows(for: g.tokens,
                                   context: index.initialContext())
        // The token straight after the boundary has no in-segment history, so
        // its rows must equal those of that same token opening a sequence.
        let after = boundary + 1
        let (fresh, _) = index.rows(for: [g.tokens[after]],
                                    context: index.initialContext())
        let observed = Array(rows[(after * index.heads)..<((after + 1) * index.heads)])
        #expect(observed == fresh)
    }

    /// Feeding tokens one at a time while carrying the context forward has to
    /// give what one batched call gives — that is what makes decode agree with
    /// prefill.
    @Test func streamingOneTokenAtATimeMatchesABatch() throws {
        guard let g = try Self.loadGolden() else { return }
        let index = Self.index(g)
        let (batched, batchedContext) = index.rows(
            for: g.tokens, context: index.initialContext())

        var context = index.initialContext()
        var streamed: [Int64] = []
        for token in g.tokens {
            let (rows, next) = index.rows(for: [token], context: context)
            streamed.append(contentsOf: rows)
            context = next
        }
        #expect(streamed == batched)
        #expect(context == batchedContext)
    }

    /// Every row has to land inside its own head's slice of the table, or the
    /// gather reads another head's embeddings.
    @Test func everyRowLandsInsideItsHeadsRange() throws {
        guard let g = try Self.loadGolden() else { return }
        let index = Self.index(g)
        let (rows, _) = index.rows(for: g.tokens,
                                   context: index.initialContext())
        for (slot, row) in rows.enumerated() {
            let head = slot % index.heads
            let lower = g.headOffsets[head]
            let upper = lower + g.headVocabSizes[head]
            #expect(row >= lower && row < upper,
                    "head \(head) row \(row) outside [\(lower), \(upper))")
        }
    }
}
