import Foundation
import Testing
@testable import TurboFieldfare

/// `ByteLevelDecoding` reproduces GPT-2's `bytes_to_unicode` inverse, and
/// `ByteLevelRun` assembles those bytes into UTF-8 incrementally. Both are
/// exercised directly here; no tokenizer fixture or network is involved.
@Suite("ByteLevel decoding")
struct ByteLevelDecodingTests {

    /// The reference GPT-2 table, rebuilt independently of the implementation.
    private static let referenceTable: [UInt8: Unicode.Scalar] = {
        var direct: [UInt8] = []
        for range in [UInt8(0x21)...UInt8(0x7E), UInt8(0xA1)...UInt8(0xAC), UInt8(0xAE)...UInt8(0xFF)] {
            direct.append(contentsOf: range)
        }
        let directSet = Set(direct)
        var table: [UInt8: Unicode.Scalar] = [:]
        for byte in direct { table[byte] = Unicode.Scalar(byte) }
        var displaced: UInt32 = 0x100
        for value in UInt8.min...UInt8.max where !directSet.contains(value) {
            table[value] = Unicode.Scalar(displaced)!
            displaced += 1
        }
        return table
    }()

    @Test("Every byte round-trips through the mapping")
    func everyByteRoundTrips() {
        for byte in UInt8.min...UInt8.max {
            let scalar = Self.referenceTable[byte]!
            #expect(ByteLevelDecoding.byteValue(scalar) == byte,
                    "byte \(byte) via U+\(String(scalar.value, radix: 16))")
        }
    }

    @Test("Space is displaced and printable ASCII is identity")
    func knownMappings() {
        // 0x20 is outside the direct ranges, so it lands on the first
        // displaced slot, U+0100 ("Ġ") — the marker Qwen tokens carry.
        #expect(ByteLevelDecoding.byteValue("Ġ") == 0x20)
        #expect(ByteLevelDecoding.byteValue("A") == 0x41)
        #expect(ByteLevelDecoding.byteValue("~") == 0x7E)
    }

    @Test("A scalar outside the mapping has no byte")
    func unmappedScalarIsNil() {
        #expect(ByteLevelDecoding.byteValue("漢") == nil)
    }

    /// Push a whole byte string through the run and concatenate the output.
    private func decode(_ bytes: [UInt8], flush: Bool = true) -> String {
        var run = ByteLevelRun()
        var text = ""
        for byte in bytes { text += run.push(byte) }
        if flush { text += run.commit() }
        return text
    }

    @Test("ASCII streams one byte at a time", arguments: [
        "hello world", "", "TurboFieldfare 0123456789",
    ])
    func asciiStreamsImmediately(_ input: String) {
        var run = ByteLevelRun()
        var text = ""
        for byte in Array(input.utf8) {
            let delta = run.push(byte)
            // No ASCII byte is ever held back.
            #expect(delta.count == 1 || byte >= 0x80)
            text += delta
        }
        #expect(text == input)
        #expect(run.commit() == "")
    }

    @Test("Multi-byte codepoints emit only once complete", arguments: [
        "漢字", "Здравствуй", "🦝", "café", "Ελληνικά", "a漢b🦝c",
    ])
    func multiByteCodepointsAssemble(_ input: String) {
        #expect(decode(Array(input.utf8)) == input)
    }

    @Test("A codepoint split across pushes is held until its last byte")
    func splitCodepointIsHeld() {
        var run = ByteLevelRun()
        let bytes = Array("漢".utf8) // 3 bytes
        #expect(run.push(bytes[0]) == "")
        #expect(run.push(bytes[1]) == "")
        #expect(run.push(bytes[2]) == "漢")
        #expect(run.commit() == "")
    }

    @Test("An incomplete trailing codepoint degrades at commit")
    func incompleteTrailingCodepointDegrades() {
        var run = ByteLevelRun()
        let bytes = Array("漢".utf8)
        _ = run.push(bytes[0])
        _ = run.push(bytes[1])
        // One truncated subpart, so one replacement character.
        #expect(run.commit() == "\u{FFFD}")
        // The run is reusable and clean after a commit.
        #expect(run.push(0x41) == "A")
    }

    /// Every expectation here is the output Rust's `String::from_utf8_lossy`
    /// produces for the same bytes: one U+FFFD per maximal invalid subpart.
    @Test("Invalid sequences decode lossily and the run recovers")
    func invalidSequencesRecover() {
        // A stray continuation byte cannot start a codepoint.
        #expect(decode([0x80]) == "\u{FFFD}")
        // A lone lead byte is one subpart; the ASCII after it is ordinary text.
        #expect(decode([0xE6, 0x41, 0x42]) == "\u{FFFD}AB")
        // A lead plus one valid continuation is still a single subpart.
        #expect(decode([0xE6, 0x80, 0x41]) == "\u{FFFD}A")
        // Overlong and out-of-range leads are rejected outright.
        #expect(decode([0xC0]) == "\u{FFFD}")
        #expect(decode([0xF5]) == "\u{FFFD}")
        // Valid text on either side of the damage survives intact.
        #expect(decode(Array("a".utf8) + [0xFF] + Array("漢".utf8)) == "a\u{FFFD}漢")
    }

    @Test("Surrogate and overlong ranges are rejected per RFC 3629")
    func rfc3629RangesRejected() {
        // ED A0 80 would encode a UTF-16 surrogate: ED is rejected as its own
        // subpart, then A0 and 80 are each stray continuations.
        #expect(decode([0xED, 0xA0, 0x80]) == "\u{FFFD}\u{FFFD}\u{FFFD}")
        // E0 80 80 is an overlong two-byte value in three-byte form.
        #expect(decode([0xE0, 0x80, 0x80]) == "\u{FFFD}\u{FFFD}\u{FFFD}")
    }

    @Test("Decoding a ByteLevel token string reproduces its text")
    func tokenStringDecodes() {
        // "Ġthe" is how a ByteLevel vocab spells " the".
        let bytes = "Ġthe".unicodeScalars.compactMap(ByteLevelDecoding.byteValue)
        #expect(bytes.count == 4)
        #expect(decode(bytes) == " the")
    }
}
