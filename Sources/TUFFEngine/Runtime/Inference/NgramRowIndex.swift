import Foundation

/// Row selection for Qwen4-Exp's n-gram per-layer-embedding table.
///
/// The table is one row per hashed n-gram, and a token selects `heads` of its
/// 320 million rows. This computes which — a rolling hash over the last
/// `ngramSize` tokens, folded into each head's own vocabulary — so the lookup
/// itself is an ordinary gather.
///
/// The hash constants are read from the checkpoint rather than regenerated:
/// the reference derives the multipliers from a seed and the per-head
/// vocabulary sizes from a prime search, but it also stores the results, and a
/// stored table is the thing the weights were actually trained against.
///
/// Everything here is `Int64` with wrapping arithmetic, matching the
/// reference's int64 tensors. Overflow is not an error; it is the hash.
struct NgramRowIndex {

    /// Hash multipliers, one per n-gram position (`ngramSize` of them).
    let multipliers: [Int64]
    /// First global row of each head, `heads` entries.
    let headOffsets: [Int64]
    /// Each head's own vocabulary size, `heads` entries.
    let headVocabSizes: [Int64]
    let ngramSize: Int
    let headsPerNgram: Int
    /// Token that both pads the start of a sequence and bounds a segment.
    let eosTokenID: Int32

    /// Tokens carried across calls so the hash sees the same history a
    /// contiguous sequence would. `ngramSize - 1` of them.
    var contextLength: Int { ngramSize - 1 }
    /// Rows one token selects, `(ngramSize - 1) * headsPerNgram`.
    var heads: Int { (ngramSize - 1) * headsPerNgram }

    init(multipliers: [Int64],
         headOffsets: [Int64],
         headVocabSizes: [Int64],
         ngramSize: Int,
         headsPerNgram: Int,
         eosTokenID: Int32) {
        precondition(ngramSize > 1, "an n-gram needs at least two positions")
        precondition(multipliers.count == ngramSize,
                     "expected \(ngramSize) multipliers, got \(multipliers.count)")
        let expectedHeads = (ngramSize - 1) * headsPerNgram
        precondition(headOffsets.count == expectedHeads
                        && headVocabSizes.count == expectedHeads,
                     "expected \(expectedHeads) head entries")
        precondition(headVocabSizes.allSatisfy { $0 > 0 },
                     "a head vocabulary size must be positive")
        self.multipliers = multipliers
        self.headOffsets = headOffsets
        self.headVocabSizes = headVocabSizes
        self.ngramSize = ngramSize
        self.headsPerNgram = headsPerNgram
        self.eosTokenID = eosTokenID
    }

    /// The token history a fresh sequence starts from: EOS repeated, which is
    /// what the reference fills its context with.
    func initialContext() -> [Int32] {
        [Int32](repeating: eosTokenID, count: contextLength)
    }

    /// Rows for every token in `tokens`, given the `contextLength` tokens that
    /// preceded them, as `tokens.count * heads` global row ids in token-major
    /// order. Also returns the context to carry into the next call.
    ///
    /// The shift is EOS-aware: a position closer than `shift` to the start of
    /// its segment reads EOS instead of reaching across the boundary, so the
    /// n-grams of one turn never mix with the previous one's.
    func rows(for tokens: [Int32],
              context: [Int32]) -> (rows: [Int64], nextContext: [Int32]) {
        precondition(context.count == contextLength,
                     "expected \(contextLength) context tokens")
        guard !tokens.isEmpty else { return ([], context) }

        let history = context + tokens
        let count = history.count

        // Where each position's segment begins: one past the most recent
        // preceding EOS. `-1` until the first one, so the first segment starts
        // at zero.
        var segmentStart = [Int](repeating: 0, count: count)
        var previousEOS = -1
        for index in 0..<count {
            segmentStart[index] = previousEOS + 1
            if history[index] == eosTokenID { previousEOS = index }
        }

        // shifted[s][i] is the token `s` places before position i, or EOS when
        // that would leave the segment.
        var shifted = [[Int64]](repeating: [], count: ngramSize)
        for shift in 0..<ngramSize {
            var row = [Int64](repeating: Int64(eosTokenID), count: count)
            for index in 0..<count {
                let source = index - shift
                let withinSegment = index - segmentStart[index] >= shift
                row[index] = (source >= 0 && withinSegment)
                    ? Int64(history[source])
                    : Int64(eosTokenID)
            }
            shifted[shift] = row
        }

        // One block of heads per n-gram order, from 2 up to `ngramSize`.
        let first = count - tokens.count
        var rows = [Int64](repeating: 0, count: tokens.count * heads)
        for order in 2...ngramSize {
            let block = (order - 2) * headsPerNgram
            for (slot, index) in (first..<count).enumerated() {
                var mixed = shifted[0][index] &* multipliers[0]
                for position in 1..<order {
                    mixed ^= shifted[position][index] &* multipliers[position]
                }
                for head in 0..<headsPerNgram {
                    let global = block + head
                    // Swift's % keeps the sign of the dividend; the reference
                    // folds into [0, size) via a non-negative modulus.
                    var folded = mixed % headVocabSizes[global]
                    if folded < 0 { folded &+= headVocabSizes[global] }
                    rows[slot * heads + global] = folded &+ headOffsets[global]
                }
            }
        }

        let nextContext = Array(history.suffix(contextLength))
        return (rows, nextContext)
    }
}
