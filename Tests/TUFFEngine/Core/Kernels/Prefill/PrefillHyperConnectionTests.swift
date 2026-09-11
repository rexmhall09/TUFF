import Testing
import Foundation
import Metal
@testable import TUFFEngine
import TUFFValidationSupport

/// The chunk-scale hyper-connection has to agree with the one-token path it
/// replaces, token for token. The two differ only in addressing — `[token]
/// [stream][d]` against `[stream][d]` — which is exactly the kind of mistake
/// that produces plausible output and no error, so it is pinned directly
/// rather than inferred from a forward pass.
@Suite struct PrefillHyperConnectionTests {

    private static let hiddenSize = 64
    private static let streams = 4
    private static let lowRank = 16

    private struct Fixture {
        let ctx: MetalContext
        let hidden: [Float]      // [T, S * D]
        let normWeight: MTLBuffer
        let tokens: Int
    }

    private static func make(tokens: Int, seed: UInt64) throws -> Fixture {
        let ctx = try MetalContext()
        var rng = SplitMix64(seed: seed)
        let width = hiddenSize * streams
        let hidden = (0..<(tokens * width)).map { _ in Float(rng.uniform(-2.0, 2.0)) }
        // Centered weights: the checkpoint stores `w`, the kernel applies 1 + w.
        let bits = (0..<width).map { _ in Quantization.bf16Bits(rng.uniform(-0.3, 0.3)) }
        let weight = ctx.device.makeBuffer(length: bits.count * 2,
                                           options: .storageModeShared)!
        bits.withUnsafeBytes {
            weight.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
        }
        return Fixture(ctx: ctx, hidden: hidden, normWeight: weight, tokens: tokens)
    }

    /// Grouped centered RMSNorm over a chunk equals the same norm applied to
    /// each token on its own.
    @Test(arguments: [1, 3, 8])
    func groupedCenteredNormMatchesTheDecodeKernel(tokens: Int) throws {
        let f = try Self.make(tokens: tokens, seed: 0x484E_0001)
        let ctx = f.ctx
        let width = Self.hiddenSize * Self.streams
        let prefill = try PrefillHyperConnection(context: ctx)
        let decode = try RMSNorm(context: ctx)

        let x = Fp16Buffer.make(ctx.device, halves: f.hidden.map { Float16($0) })!
        let chunkOut = Fp16Buffer.make(ctx.device, count: tokens * width)!
        let cb = ctx.queue.makeCommandBuffer()!
        prefill.encodeGroupedCenteredNorm(
            commandBuffer: cb, x: x, weight: f.normWeight, weightOffset: 0,
            out: chunkOut, tokens: tokens, hiddenSize: Self.hiddenSize,
            streamCount: Self.streams, eps: 1e-6)
        cb.commit(); cb.waitUntilCompleted()

        var reference: [Float] = []
        for t in 0..<tokens {
            let row = Array(f.hidden[(t * width)..<((t + 1) * width)])
            let rowBuf = Fp16Buffer.make(ctx.device, halves: row.map { Float16($0) })!
            let out = Fp16Buffer.make(ctx.device, count: width)!
            let one = ctx.queue.makeCommandBuffer()!
            decode.encodeBF16WGroupedCentered(
                commandBuffer: one, x: rowBuf, weight: f.normWeight, weightOffset: 0,
                out: out, d: UInt32(Self.hiddenSize), groups: Self.streams, eps: 1e-6)
            one.commit(); one.waitUntilCompleted()
            reference += Fp16Buffer.read(out, count: width)
        }
        let actual = Fp16Buffer.read(chunkOut, count: tokens * width)
        let relErr = RelError.compute(actual: actual, reference: reference)
        #expect(relErr < 1e-3, "tokens=\(tokens) relErr=\(relErr)")
    }

    /// The combine collapses the streams the same way for a chunk as for a
    /// token, and the low-rank silu carries the same 1/streams scale.
    @Test(arguments: [1, 5])
    func combineAndLowRankMatchTheDecodeKernels(tokens: Int) throws {
        let f = try Self.make(tokens: tokens, seed: 0x484E_0002)
        let ctx = f.ctx
        var rng = SplitMix64(seed: 0x484E_0003)
        let width = Self.hiddenSize * Self.streams
        let prefill = try PrefillHyperConnection(context: ctx)
        let decode = try HyperConnection(context: ctx)

        let up = (0..<(tokens * width)).map { _ in Float(rng.uniform(-3.0, 3.0)) }
        let low = (0..<(tokens * Self.lowRank)).map { _ in Float(rng.uniform(-4.0, 4.0)) }

        let upBuf = Fp16Buffer.make(ctx.device, halves: up.map { Float16($0) })!
        let normedBuf = Fp16Buffer.make(ctx.device, halves: f.hidden.map { Float16($0) })!
        let mixedBuf = Fp16Buffer.make(ctx.device, count: tokens * Self.hiddenSize)!
        let lowBuf = Fp16Buffer.make(ctx.device, halves: low.map { Float16($0) })!
        let cb = ctx.queue.makeCommandBuffer()!
        prefill.encodeLowRankSilu(commandBuffer: cb, x: lowBuf, tokens: tokens,
                                  lowRank: Self.lowRank, streamCount: Self.streams)
        prefill.encodeCombine(commandBuffer: cb, up: upBuf, normed: normedBuf,
                              mixed: mixedBuf, tokens: tokens,
                              hiddenSize: Self.hiddenSize, streamCount: Self.streams)
        cb.commit(); cb.waitUntilCompleted()

        var wantMixed: [Float] = []
        var wantLow: [Float] = []
        for t in 0..<tokens {
            let upRow = Array(up[(t * width)..<((t + 1) * width)])
            let nRow = Array(f.hidden[(t * width)..<((t + 1) * width)])
            let lowRow = Array(low[(t * Self.lowRank)..<((t + 1) * Self.lowRank)])
            let u = Fp16Buffer.make(ctx.device, halves: upRow.map { Float16($0) })!
            let n = Fp16Buffer.make(ctx.device, halves: nRow.map { Float16($0) })!
            let m = Fp16Buffer.make(ctx.device, count: Self.hiddenSize)!
            let l = Fp16Buffer.make(ctx.device, halves: lowRow.map { Float16($0) })!
            let one = ctx.queue.makeCommandBuffer()!
            decode.encodeLowRankSilu(commandBuffer: one, x: l, out: l,
                                     count: Self.lowRank, streamCount: Self.streams)
            decode.encodeCombine(commandBuffer: one, up: u, normed: n, mixed: m,
                                 hiddenSize: Self.hiddenSize,
                                 streamCount: Self.streams)
            one.commit(); one.waitUntilCompleted()
            wantMixed += Fp16Buffer.read(m, count: Self.hiddenSize)
            wantLow += Fp16Buffer.read(l, count: Self.lowRank)
        }
        let mixedErr = RelError.compute(
            actual: Fp16Buffer.read(mixedBuf, count: tokens * Self.hiddenSize),
            reference: wantMixed)
        let lowErr = RelError.compute(
            actual: Fp16Buffer.read(lowBuf, count: tokens * Self.lowRank),
            reference: wantLow)
        #expect(mixedErr < 1e-3, "mixed tokens=\(tokens) relErr=\(mixedErr)")
        #expect(lowErr < 1e-3, "lowRank tokens=\(tokens) relErr=\(lowErr)")
    }

    /// The injection gate writes the branch into every stream with the same
    /// `2 * sigmoid(raw / streams)` weight the decode path applies.
    @Test(arguments: [1, 6])
    func injectionMatchesTheDecodeKernel(tokens: Int) throws {
        let f = try Self.make(tokens: tokens, seed: 0x484E_0004)
        let ctx = f.ctx
        var rng = SplitMix64(seed: 0x484E_0005)
        let width = Self.hiddenSize * Self.streams
        let prefill = try PrefillHyperConnection(context: ctx)
        let decode = try HyperConnection(context: ctx)

        let branch = (0..<(tokens * Self.hiddenSize)).map { _ in Float(rng.uniform(-1.0, 1.0)) }
        let raw = (0..<(tokens * Self.streams)).map { _ in Float(rng.uniform(-5.0, 5.0)) }

        let hiddenBuf = Fp16Buffer.make(ctx.device, halves: f.hidden.map { Float16($0) })!
        let branchBuf = Fp16Buffer.make(ctx.device, halves: branch.map { Float16($0) })!
        let rawBuf = Fp16Buffer.make(ctx.device, halves: raw.map { Float16($0) })!
        let cb = ctx.queue.makeCommandBuffer()!
        prefill.encodeInject(commandBuffer: cb, hidden: hiddenBuf, branch: branchBuf,
                             injectionRaw: rawBuf, tokens: tokens,
                             hiddenSize: Self.hiddenSize, streamCount: Self.streams)
        cb.commit(); cb.waitUntilCompleted()

        var reference: [Float] = []
        for t in 0..<tokens {
            let hRow = Array(f.hidden[(t * width)..<((t + 1) * width)])
            let bRow = Array(branch[(t * Self.hiddenSize)..<((t + 1) * Self.hiddenSize)])
            let rRow = Array(raw[(t * Self.streams)..<((t + 1) * Self.streams)])
            let h = Fp16Buffer.make(ctx.device, halves: hRow.map { Float16($0) })!
            let b = Fp16Buffer.make(ctx.device, halves: bRow.map { Float16($0) })!
            let r = Fp16Buffer.make(ctx.device, halves: rRow.map { Float16($0) })!
            let one = ctx.queue.makeCommandBuffer()!
            decode.encodeInject(commandBuffer: one, hyper: h, branch: b,
                                injectionRaw: r, hiddenSize: Self.hiddenSize,
                                streamCount: Self.streams)
            one.commit(); one.waitUntilCompleted()
            reference += Fp16Buffer.read(h, count: width)
        }
        let relErr = RelError.compute(
            actual: Fp16Buffer.read(hiddenBuf, count: tokens * width),
            reference: reference)
        #expect(relErr < 1e-3, "tokens=\(tokens) relErr=\(relErr)")
    }

    /// Tiling has to leave every stream holding the token's own embedding —
    /// and a token's copy must not leak into its neighbour.
    @Test func tilingCopiesEachTokenEmbeddingAcrossItsOwnStreams() throws {
        let tokens = 5
        let ctx = try MetalContext()
        let prefill = try PrefillHyperConnection(context: ctx)
        let width = Self.hiddenSize * Self.streams
        var values = [Float](repeating: 0, count: tokens * width)
        for t in 0..<tokens {
            for d in 0..<Self.hiddenSize {
                values[t * width + d] = Float(t * 1_000 + d)
            }
        }
        let buf = Fp16Buffer.make(ctx.device, halves: values.map { Float16($0) })!
        let cb = ctx.queue.makeCommandBuffer()!
        prefill.encodeTileEmbedding(commandBuffer: cb, hidden: buf, tokens: tokens,
                                    hiddenSize: Self.hiddenSize,
                                    streamCount: Self.streams)
        cb.commit(); cb.waitUntilCompleted()
        let out = Fp16Buffer.read(buf, count: tokens * width)
        for t in 0..<tokens {
            for s in 0..<Self.streams {
                for d in 0..<Self.hiddenSize {
                    #expect(out[t * width + s * Self.hiddenSize + d]
                            == Float(Float16(Float(t * 1_000 + d))),
                            "token \(t) stream \(s) index \(d)")
                }
            }
        }
    }
}
