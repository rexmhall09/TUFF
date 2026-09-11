import Testing
import Foundation
import Metal
@testable import TUFFEngine
import TUFFValidationSupport

/// Checks the multi-stream residual arithmetic against a plain Swift
/// transcription of mlx-vlm's `Qwen4ExpGatedResidual`.
///
/// The reference is written out in full below rather than reduced, because the
/// details are exactly where this can go quietly wrong: the stream count
/// divides the projection *before* each activation rather than after, the
/// injection gate carries a factor of two, and the residual that survives a
/// block is the unnormalized input, not the normalized one.
@Suite struct HyperConnectionTests {

    private static let streams = 4
    private static let hidden = 320

    private static func sigmoid(_ v: Float) -> Float { 1 / (1 + exp(-v)) }

    private static func fp16Buffer(_ device: MTLDevice,
                                   _ values: [Float]) -> MTLBuffer? {
        Fp16Buffer.make(device, halves: values.map { Float16($0) })
    }

    @Test func lowRankActivationDividesBeforeTheNonlinearity() throws {
        let count = 320
        var rng = SeedTree(0x4843_4C52).key("hc-lowrank")
        let x = (0..<count).map { _ in rng.uniform(-4.0, 4.0) }
        let xRounded = x.map { Float(Float16($0)) }

        let ctx = try MetalContext()
        let kernel = try HyperConnection(context: ctx)
        guard let xBuf = Self.fp16Buffer(ctx.device, x),
              let outBuf = Fp16Buffer.make(ctx.device, count: count) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeLowRankSilu(commandBuffer: cb, x: xBuf, out: outBuf,
                                 count: count, streamCount: Self.streams)
        cb.commit(); cb.waitUntilCompleted()

        let expected = xRounded.map { value -> Float in
            let v = value / Float(Self.streams)
            return v * Self.sigmoid(v)
        }
        let actual = Fp16Buffer.read(outBuf, count: count)
        #expect(RelError.compute(actual: actual, reference: expected)
                < Tolerance.fp16Reduction)

        // Dividing after the activation instead of before would agree at zero
        // and diverge everywhere else; make sure the test could tell.
        let wrong = xRounded.map { value -> Float in
            (value * Self.sigmoid(value)) / Float(Self.streams)
        }
        #expect(RelError.maxAbsDiff(expected, wrong) > 0.1)
    }

    @Test func combineCollapsesStreamsByTheirGatedMean() throws {
        let total = Self.streams * Self.hidden
        var rng = SeedTree(0x4843_4D58).key("hc-combine")
        let up = (0..<total).map { _ in rng.uniform(-3.0, 3.0) }
        let normed = (0..<total).map { _ in rng.uniform(-2.0, 2.0) }
        let upRounded = up.map { Float(Float16($0)) }
        let normedRounded = normed.map { Float(Float16($0)) }

        let ctx = try MetalContext()
        let kernel = try HyperConnection(context: ctx)
        guard let upBuf = Self.fp16Buffer(ctx.device, up),
              let normedBuf = Self.fp16Buffer(ctx.device, normed),
              let outBuf = Fp16Buffer.make(ctx.device, count: Self.hidden) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeCombine(commandBuffer: cb, up: upBuf, normed: normedBuf,
                             mixed: outBuf, hiddenSize: Self.hidden,
                             streamCount: Self.streams)
        cb.commit(); cb.waitUntilCompleted()

        var expected = [Float](repeating: 0, count: Self.hidden)
        for i in 0..<Self.hidden {
            var acc: Float = 0
            for g in 0..<Self.streams {
                let index = g * Self.hidden + i
                acc += Self.sigmoid(upRounded[index]) * normedRounded[index]
            }
            expected[i] = acc / Float(Self.streams)
        }
        let actual = Fp16Buffer.read(outBuf, count: Self.hidden)
        #expect(RelError.compute(actual: actual, reference: expected)
                < Tolerance.fp16Reduction)
    }

    @Test func injectionAccumulatesIntoEveryStreamWithItsOwnGate() throws {
        let total = Self.streams * Self.hidden
        var rng = SeedTree(0x4843_494E).key("hc-inject")
        let hyper = (0..<total).map { _ in rng.uniform(-2.0, 2.0) }
        let branch = (0..<Self.hidden).map { _ in rng.uniform(-2.0, 2.0) }
        let injRaw = (0..<Self.streams).map { _ in rng.uniform(-3.0, 3.0) }
        let hyperRounded = hyper.map { Float(Float16($0)) }
        let branchRounded = branch.map { Float(Float16($0)) }
        let injRounded = injRaw.map { Float(Float16($0)) }

        let ctx = try MetalContext()
        let kernel = try HyperConnection(context: ctx)
        guard let hyperBuf = Self.fp16Buffer(ctx.device, hyper),
              let branchBuf = Self.fp16Buffer(ctx.device, branch),
              let injBuf = Self.fp16Buffer(ctx.device, injRaw) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeInject(commandBuffer: cb, hyper: hyperBuf,
                            branch: branchBuf, injectionRaw: injBuf,
                            hiddenSize: Self.hidden, streamCount: Self.streams)
        cb.commit(); cb.waitUntilCompleted()

        var expected = hyperRounded
        for g in 0..<Self.streams {
            let gate = 2 * Self.sigmoid(injRounded[g] / Float(Self.streams))
            for i in 0..<Self.hidden {
                expected[g * Self.hidden + i] += branchRounded[i] * gate
            }
        }
        let actual = Fp16Buffer.read(hyperBuf, count: total)
        #expect(RelError.compute(actual: actual, reference: expected)
                < Tolerance.fp16Reduction)

        // Each stream must get a different gate. If the kernel indexed the
        // gate by element rather than by stream, or dropped the factor of two,
        // the per-stream deltas would stop being distinguishable multiples of
        // the branch.
        let deltas = (0..<Self.streams).map { g -> Float in
            actual[g * Self.hidden] - hyperRounded[g * Self.hidden]
        }
        let ratios = deltas.map { $0 / branchRounded[0] }
        #expect(Set(ratios.map { ($0 * 100).rounded() }).count == Self.streams,
                "streams did not receive distinct gates: \(ratios)")
        for (g, ratio) in ratios.enumerated() {
            let gate = 2 * Self.sigmoid(injRounded[g] / Float(Self.streams))
            #expect(abs(ratio - gate) < 0.02, "stream \(g) gate \(ratio) vs \(gate)")
        }
    }

    /// The whole block, in the order `Qwen4ExpGatedResidual` runs it, with the
    /// projections done on the host so only the kernel arithmetic is on trial.
    @Test func theBlockReproducesTheReferenceSequence() throws {
        let lowRank = 64
        let total = Self.streams * Self.hidden
        var rng = SeedTree(0x4843_5351).key("hc-sequence")

        let normed = (0..<total).map { _ in rng.uniform(-1.5, 1.5) }
        let hyper = (0..<total).map { _ in rng.uniform(-1.5, 1.5) }
        let down = (0..<lowRank).map { _ in rng.uniform(-3.0, 3.0) }
        let branch = (0..<Self.hidden).map { _ in rng.uniform(-1.0, 1.0) }
        let injRaw = (0..<Self.streams).map { _ in rng.uniform(-2.0, 2.0) }

        // A stand-in up-projection: the kernels never see the weights, so any
        // fixed map from the low-rank vector to G * D will do.
        func upProjection(_ t: [Float]) -> [Float] {
            (0..<total).map { index in
                t[index % lowRank] * (0.5 + Float(index % 7) * 0.1)
            }
        }

        let ctx = try MetalContext()
        let kernel = try HyperConnection(context: ctx)
        guard let downBuf = Self.fp16Buffer(ctx.device, down),
              let tBuf = Fp16Buffer.make(ctx.device, count: lowRank) else {
            Issue.record("alloc failed"); return
        }
        var cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeLowRankSilu(commandBuffer: cb, x: downBuf, out: tBuf,
                                 count: lowRank, streamCount: Self.streams)
        cb.commit(); cb.waitUntilCompleted()
        let t = Fp16Buffer.read(tBuf, count: lowRank)
        let up = upProjection(t)

        guard let upBuf = Self.fp16Buffer(ctx.device, up),
              let normedBuf = Self.fp16Buffer(ctx.device, normed),
              let mixedBuf = Fp16Buffer.make(ctx.device, count: Self.hidden),
              let hyperBuf = Self.fp16Buffer(ctx.device, hyper),
              let branchBuf = Self.fp16Buffer(ctx.device, branch),
              let injBuf = Self.fp16Buffer(ctx.device, injRaw) else {
            Issue.record("alloc failed"); return
        }
        cb = ctx.queue.makeCommandBuffer()!
        kernel.encodeCombine(commandBuffer: cb, up: upBuf, normed: normedBuf,
                             mixed: mixedBuf, hiddenSize: Self.hidden,
                             streamCount: Self.streams)
        kernel.encodeInject(commandBuffer: cb, hyper: hyperBuf,
                            branch: branchBuf, injectionRaw: injBuf,
                            hiddenSize: Self.hidden, streamCount: Self.streams)
        cb.commit(); cb.waitUntilCompleted()

        let normedRounded = normed.map { Float(Float16($0)) }
        let upRounded = up.map { Float(Float16($0)) }
        var expectedMixed = [Float](repeating: 0, count: Self.hidden)
        for i in 0..<Self.hidden {
            var acc: Float = 0
            for g in 0..<Self.streams {
                let index = g * Self.hidden + i
                acc += Self.sigmoid(upRounded[index]) * normedRounded[index]
            }
            expectedMixed[i] = acc / Float(Self.streams)
        }
        #expect(RelError.compute(actual: Fp16Buffer.read(mixedBuf,
                                                         count: Self.hidden),
                                 reference: expectedMixed)
                < Tolerance.fp16Reduction)

        // The residual that leaves the block is built from the unnormalized
        // input. Building it from `normed` is the mistake this catches.
        var expectedHyper = hyper.map { Float(Float16($0)) }
        let branchRounded = branch.map { Float(Float16($0)) }
        let injRounded = injRaw.map { Float(Float16($0)) }
        for g in 0..<Self.streams {
            let gate = 2 * Self.sigmoid(injRounded[g] / Float(Self.streams))
            for i in 0..<Self.hidden {
                expectedHyper[g * Self.hidden + i] += branchRounded[i] * gate
            }
        }
        #expect(RelError.compute(actual: Fp16Buffer.read(hyperBuf, count: total),
                                 reference: expectedHyper)
                < Tolerance.fp16Reduction)
        #expect(RelError.maxAbsDiff(expectedHyper, normedRounded) > 0.1)
    }
}
