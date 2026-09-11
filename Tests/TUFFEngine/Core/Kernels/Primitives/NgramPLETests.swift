import Testing
import Foundation
import Metal
@testable import TUFFEngine
import TUFFValidationSupport

/// The n-gram PLE's arithmetic, against a transcription of mlx-vlm's
/// `Qwen4ExpPLELayer`.
///
/// Two details here are easy to get wrong and impossible to notice afterwards:
/// the gate takes a *signed* square root of the dot product rather than a
/// plain one, so a negative match stays negative instead of folding to
/// positive; and the convolution is dilated by the n-gram size, so its four
/// taps read every third row of the history rather than four adjacent ones.
@Suite struct NgramPLETests {

    private static let streams = 4
    private static let hidden = 128

    private static func sigmoid(_ v: Float) -> Float { 1 / (1 + exp(-v)) }
    private static func silu(_ v: Float) -> Float { v * sigmoid(v) }

    private static func fp16(_ device: MTLDevice, _ v: [Float]) -> MTLBuffer? {
        Fp16Buffer.make(device, halves: v.map { Float16($0) })
    }

    @Test func theGateTakesASignedSquareRootOfTheMatch() throws {
        let total = Self.streams * Self.hidden
        var rng = SeedTree(0x504C_4547).key("ple-gate")
        // Deliberately mixed signs: half the streams should end up with a
        // negative gate, which is where the signed square root shows.
        let keys = (0..<total).map { _ in rng.uniform(-1.5, 1.5) }
        let queries = (0..<total).map { i in
            (i / Self.hidden) % 2 == 0 ? rng.uniform(0.2, 1.5) : rng.uniform(-1.5, -0.2)
        }
        let values = (0..<Self.hidden).map { _ in rng.uniform(-1.0, 1.0) }

        let ctx = try MetalContext()
        let kernel = try NgramPLE(context: ctx)
        guard let kBuf = Self.fp16(ctx.device, keys),
              let qBuf = Self.fp16(ctx.device, queries),
              let vBuf = Self.fp16(ctx.device, values),
              let out = Fp16Buffer.make(ctx.device, count: total) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeGateValues(commandBuffer: cb, keys: kBuf, queries: qBuf,
                                values: vBuf, out: out,
                                hiddenSize: Self.hidden,
                                streamCount: Self.streams)
        cb.commit(); cb.waitUntilCompleted()

        let kR = keys.map { Float(Float16($0)) }
        let qR = queries.map { Float(Float16($0)) }
        let vR = values.map { Float(Float16($0)) }
        var expected = [Float](repeating: 0, count: total)
        var sawNegative = false
        for s in 0..<Self.streams {
            var dot: Float = 0
            for i in 0..<Self.hidden {
                dot += kR[s * Self.hidden + i] * qR[s * Self.hidden + i]
            }
            var gate = dot / Float(Self.hidden).squareRoot()
            if gate < 0 { sawNegative = true }
            let magnitude = max(abs(gate), 1e-6).squareRoot()
            gate = gate < 0 ? -magnitude : magnitude
            let g = Self.sigmoid(gate)
            for i in 0..<Self.hidden {
                expected[s * Self.hidden + i] = g * vR[i]
            }
        }
        #expect(sawNegative, "the fixture no longer exercises a negative gate")

        let actual = Fp16Buffer.read(out, count: total)
        #expect(RelError.compute(actual: actual, reference: expected)
                < Tolerance.fp16Reduction)

        // An unsigned square root would make every gate positive, so the
        // streams with a negative match would come out larger than they should.
        let wrong = (0..<Self.streams).map { s -> Float in
            var dot: Float = 0
            for i in 0..<Self.hidden {
                dot += kR[s * Self.hidden + i] * qR[s * Self.hidden + i]
            }
            return Self.sigmoid(max(abs(dot / Float(Self.hidden).squareRoot()),
                                    1e-6).squareRoot())
        }
        let firstOfEachStream = (0..<Self.streams).map { actual[$0 * Self.hidden] }
        let wrongFirst = (0..<Self.streams).map { wrong[$0] * vR[0] }
        #expect(RelError.maxAbsDiff(firstOfEachStream, wrongFirst) > 0.05,
                "the kernel dropped the gate's sign")
    }

    /// Four taps at dilation three read rows 0, 3, 6 and 9 of a ten-row
    /// history — the last being the current token.
    @Test func theConvolutionIsDilatedNotAdjacent() throws {
        let width = 64
        let taps = 4
        let dilation = 3
        let rows = (taps - 1) * dilation + 1
        var rng = SeedTree(0x504C_4543).key("ple-conv")
        let history = (0..<(rows * width)).map { _ in rng.uniform(-1.0, 1.0) }
        let weightF = (0..<(width * taps)).map { _ in rng.uniform(-0.8, 0.8) }

        let ctx = try MetalContext()
        let kernel = try NgramPLE(context: ctx)
        guard let hBuf = Self.fp16(ctx.device, history),
              let out = Fp16Buffer.make(ctx.device, count: width),
              let wBuf = ctx.device.makeBuffer(length: weightF.count * 2,
                                               options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }
        let wBits = weightF.map { Quantization.bf16Bits($0) }
        let wPtr = wBuf.contents().bindMemory(to: UInt16.self,
                                              capacity: wBits.count)
        for i in 0..<wBits.count { wPtr[i] = wBits[i] }

        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeDepthwiseConv(commandBuffer: cb, history: hBuf,
                                   weight: wBuf, out: out,
                                   width: width, taps: taps, dilation: dilation)
        cb.commit(); cb.waitUntilCompleted()

        let hR = history.map { Float(Float16($0)) }
        let wR = wBits.map { Quantization.bf16ToFloat($0) }
        let expected = (0..<width).map { channel -> Float in
            var acc: Float = 0
            for k in 0..<taps {
                acc += wR[channel * taps + k] * hR[(k * dilation) * width + channel]
            }
            return Self.silu(acc)
        }
        #expect(RelError.compute(actual: Fp16Buffer.read(out, count: width),
                                 reference: expected) < Tolerance.fp16Reduction)

        // Adjacent taps would read rows 0..3 instead, which is a different sum.
        let adjacent = (0..<width).map { channel -> Float in
            var acc: Float = 0
            for k in 0..<taps {
                acc += wR[channel * taps + k] * hR[k * width + channel]
            }
            return Self.silu(acc)
        }
        #expect(RelError.maxAbsDiff(expected, adjacent) > 0.05,
                "the fixture cannot tell dilated from adjacent taps")
    }

    /// Pushing a row drops the oldest and keeps the order, so the next token's
    /// taps land on the same relative positions.
    @Test func historyPushShiftsAndAppends() throws {
        let width = 32
        let length = 10
        var rng = SeedTree(0x504C_4348).key("ple-history")
        let history = (0..<(length * width)).map { _ in rng.uniform(-1.0, 1.0) }
        let row = (0..<width).map { _ in rng.uniform(2.0, 3.0) }

        let ctx = try MetalContext()
        let kernel = try NgramPLE(context: ctx)
        guard let hBuf = Self.fp16(ctx.device, history),
              let rBuf = Self.fp16(ctx.device, row) else {
            Issue.record("alloc failed"); return
        }
        let before = Fp16Buffer.read(hBuf, count: length * width)
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeHistoryPush(commandBuffer: cb, history: hBuf, row: rBuf,
                                 width: width, length: length)
        cb.commit(); cb.waitUntilCompleted()
        let after = Fp16Buffer.read(hBuf, count: length * width)

        for r in 0..<(length - 1) {
            let moved = Array(after[(r * width)..<((r + 1) * width)])
            let source = Array(before[((r + 1) * width)..<((r + 2) * width)])
            #expect(moved == source, "row \(r) did not shift down")
        }
        let appended = Array(after[((length - 1) * width)..<(length * width)])
        #expect(appended == row.map { Float(Float16($0)) })
    }
}
