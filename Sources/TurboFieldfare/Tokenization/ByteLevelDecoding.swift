import Foundation

/// The ByteLevel detokenization pipeline used by the ChatML/Qwen tokenizer.
///
/// Qwen's `tokenizer.json` declares `decoder: ByteLevel`, not the Gemma
/// `Sequence[Replace, ByteFallback, Fuse]`. ByteLevel is GPT-2's
/// `bytes_to_unicode` mapping: every one of the 256 byte values is rendered as a
/// single printable scalar, so a token string is a byte string in disguise.
/// Decoding maps each scalar back to its byte and UTF-8 decodes the result.
///
/// Like `GemmaDecoding`, this stops at the declared decoder and skips HF's
/// `clean_up_tokenization_spaces` pass, so batch and streaming decode agree and
/// `decode(encode(x)) == x`.
enum ByteLevelDecoding {
    /// Byte for one scalar of a ByteLevel token, or nil when the scalar is not
    /// in the mapping. Added tokens (`<think>`, `<|im_end|>`, …) are stored
    /// literally rather than byte-encoded; every ASCII scalar except space maps
    /// to itself, so they survive this table unchanged, and anything outside it
    /// is passed through verbatim by `GFDetokenizer`.
    static func byteValue(_ scalar: Unicode.Scalar) -> UInt8? {
        inverseTable[scalar]
    }

    /// GPT-2 `bytes_to_unicode`, inverted.
    ///
    /// The printable ASCII range plus two Latin-1 runs map to themselves; the
    /// remaining 68 byte values are displaced to U+0100 and up, in byte order.
    private static let inverseTable: [Unicode.Scalar: UInt8] = {
        var direct: [UInt8] = []
        for range in [UInt8(0x21)...UInt8(0x7E), UInt8(0xA1)...UInt8(0xAC), UInt8(0xAE)...UInt8(0xFF)] {
            direct.append(contentsOf: range)
        }
        let directSet = Set(direct)
        var table: [Unicode.Scalar: UInt8] = [:]
        for byte in direct {
            table[Unicode.Scalar(byte)] = byte
        }
        var displaced: UInt32 = 0x100
        for value in UInt8.min...UInt8.max where !directSet.contains(value) {
            table[Unicode.Scalar(displaced)!] = value
            displaced += 1
        }
        return table
    }()
}

/// Incremental UTF-8 assembler for a ByteLevel token stream.
///
/// ByteLevel gives one byte per scalar, and a multi-byte codepoint can straddle
/// a token boundary, so bytes are buffered only as long as a codepoint is
/// genuinely incomplete. As soon as the buffer closes a codepoint it is emitted,
/// which keeps ASCII — the common case — streaming a token at a time instead of
/// waiting for a run to end.
///
/// Invalid input decodes lossily with the reference decoder's
/// `from_utf8_lossy` semantics: one U+FFFD per *maximal invalid subpart*, not
/// per byte. So `E6 41 42` decodes to `"\u{FFFD}AB"` — the lone lead byte is
/// one subpart, and the ASCII that follows it is still ordinary text.
struct ByteLevelRun {
    private var bytes: [UInt8] = []
    /// Continuation bytes still owed by the current lead, and the allowed range
    /// for the next one (RFC 3629 constrains the first continuation after
    /// E0/ED/F0/F4 leads).
    private var pendingContinuations = 0
    private var nextContinuation: ClosedRange<UInt8> = 0x80...0xBF

    /// Text this byte contributes: `""` while a codepoint is still incomplete.
    mutating func push(_ byte: UInt8) -> String {
        if pendingContinuations > 0 {
            guard nextContinuation.contains(byte) else {
                // The buffered lead and its continuations so far are one
                // maximal invalid subpart; `byte` is not part of it and starts
                // fresh.
                reset()
                return "\u{FFFD}" + push(byte)
            }
            bytes.append(byte)
            pendingContinuations -= 1
            nextContinuation = 0x80...0xBF
            return pendingContinuations == 0 ? take() : ""
        }
        switch byte {
        case 0x00...0x7F:
            bytes.append(byte)
            return take()
        case 0xC2...0xDF: pendingContinuations = 1
        case 0xE0: pendingContinuations = 2; nextContinuation = 0xA0...0xBF
        case 0xE1...0xEC, 0xEE, 0xEF: pendingContinuations = 2
        case 0xED: pendingContinuations = 2; nextContinuation = 0x80...0x9F
        case 0xF0: pendingContinuations = 3; nextContinuation = 0x90...0xBF
        case 0xF1...0xF3: pendingContinuations = 3
        case 0xF4: pendingContinuations = 3; nextContinuation = 0x80...0x8F
        default:
            // Stray continuation, overlong lead, or > U+10FFFF: a one-byte
            // maximal invalid subpart.
            return "\u{FFFD}"
        }
        bytes.append(byte)
        return ""
    }

    /// Close the run: the stream ends, or a barrier marker interrupts it. A
    /// codepoint left incomplete is one truncated subpart, so it degrades to a
    /// single U+FFFD.
    mutating func commit() -> String {
        guard !bytes.isEmpty else { return "" }
        reset()
        return "\u{FFFD}"
    }

    /// Emit the completed codepoint held in the buffer.
    private mutating func take() -> String {
        defer { reset() }
        return String(bytes: bytes, encoding: .utf8) ?? "\u{FFFD}"
    }

    private mutating func reset() {
        bytes.removeAll(keepingCapacity: true)
        pendingContinuations = 0
        nextContinuation = 0x80...0xBF
    }
}
