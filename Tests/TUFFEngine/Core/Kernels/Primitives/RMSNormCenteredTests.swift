import Testing
import Foundation
import Metal
@testable import TUFFEngine
import TUFFValidationSupport

/// Qwen4-Exp's norms store their weights centered at zero and scale by
/// `1 + w`. Qwen3.6's are centered at one and scale by `w`, so the same tensor
/// name means different arithmetic in the two architectures and reusing the
/// existing kernel would be quietly wrong rather than loudly broken.
///
/// The reference is `RmsNormRef` — the same Accelerate pipeline the plain
/// kernel is checked against — fed weights with the one already added, so the
/// only thing under test is that the kernel adds it and adds it in float.
@Suite struct RMSNormCenteredTests {

    private static let eps: Float = 1e-6

    private struct Fixture {
        let xBuf: MTLBuffer
        let yBuf: MTLBuffer
        let wBuf: MTLBuffer
        let xRef: [Float]
        let wRef: [Float]
        let ctx: MetalContext
        let kernel: RMSNorm
    }

    /// Weights are drawn around zero, which is where this architecture's
    /// actually sit; drawing them around one would let a kernel that forgot
    /// the offset still look close.
    private static func makeFixture(count: Int, seed: UInt64,
                                    label: String) throws -> Fixture? {
        var rng = SeedTree(seed).key(label)
        let xFp32 = (0..<count).map { _ in rng.uniform(-1.0, 1.0) }
        let wFp32 = (0..<count).map { _ in rng.uniform(-0.4, 0.4) }

        let xFp16 = xFp32.map { Float16($0) }
        let xRef = xFp16.map { Float($0) }
        let wBits = wFp32.map { Quantization.bf16Bits($0) }
        // The kernel reads BF16 and adds one in float, so the reference has to
        // round first and add second.
        let wRef = wBits.map { 1.0 + Quantization.bf16ToFloat($0) }

        let ctx = try MetalContext()
        let kernel = try RMSNorm(context: ctx)
        guard let xBuf = Fp16Buffer.make(ctx.device, halves: xFp16),
              let yBuf = Fp16Buffer.make(ctx.device, count: count),
              let wBuf = ctx.device.makeBuffer(length: wBits.count * 2,
                                               options: .storageModeShared) else {
            return nil
        }
        let wPtr = wBuf.contents().bindMemory(to: UInt16.self,
                                              capacity: wBits.count)
        for i in 0..<wBits.count { wPtr[i] = wBits[i] }
        return Fixture(xBuf: xBuf, yBuf: yBuf, wBuf: wBuf,
                       xRef: xRef, wRef: wRef, ctx: ctx, kernel: kernel)
    }

    /// One group is an ordinary centered norm.
    @Test(arguments: [256, 2_560])
    func singleGroupMatchesTheCenteredReference(d: Int) throws {
        guard let f = try Self.makeFixture(
            count: d, seed: 0x4E47_524D, label: "centered-single-d\(d)") else {
            Issue.record("alloc failed"); return
        }
        let cb = f.ctx.queue.makeCommandBuffer()!
        f.kernel.encodeBF16WGroupedCentered(
            commandBuffer: cb, x: f.xBuf, weight: f.wBuf, out: f.yBuf,
            d: UInt32(d), groups: 1, eps: Self.eps)
        cb.commit(); cb.waitUntilCompleted()

        let ref = RmsNormRef.apply(x: f.xRef, weight: f.wRef, eps: Self.eps)
        let actual = Fp16Buffer.read(f.yBuf, count: d)
        let relErr = RelError.compute(actual: actual, reference: ref)
        #expect(relErr < Tolerance.fp16Reduction,
                "centered D=\(d): relErr=\(relErr)")
    }

    /// The hyper-connection shape: four residual streams of 2,560 normalized
    /// independently, each against its own slice of a 10,240-wide weight. A
    /// kernel that normalized across the whole width would pass the
    /// single-group test and fail this one.
    @Test func groupsNormalizeIndependentlyWithTheirOwnWeightSlice() throws {
        let groups = 4
        let d = 2_560
        let total = groups * d
        guard let f = try Self.makeFixture(
            count: total, seed: 0x4843_4E4F, label: "centered-hc") else {
            Issue.record("alloc failed"); return
        }
        let cb = f.ctx.queue.makeCommandBuffer()!
        f.kernel.encodeBF16WGroupedCentered(
            commandBuffer: cb, x: f.xBuf, weight: f.wBuf, out: f.yBuf,
            d: UInt32(d), groups: groups, eps: Self.eps)
        cb.commit(); cb.waitUntilCompleted()

        let actual = Fp16Buffer.read(f.yBuf, count: total)
        for group in 0..<groups {
            let range = (group * d)..<((group + 1) * d)
            let ref = RmsNormRef.apply(x: Array(f.xRef[range]),
                                       weight: Array(f.wRef[range]),
                                       eps: Self.eps)
            let relErr = RelError.compute(actual: Array(actual[range]),
                                          reference: ref)
            #expect(relErr < Tolerance.fp16Reduction,
                    "hc group \(group): relErr=\(relErr)")
        }
    }

    /// q_norm and k_norm: one weight, every head normalized on its own.
    @Test func perHeadSharesOneWeightAcrossHeads() throws {
        let headDim = 256
        let heads = 24
        guard let f = try Self.makeFixture(
            count: heads * headDim, seed: 0x5148_4541,
            label: "centered-perhead") else {
            Issue.record("alloc failed"); return
        }
        // A shared [headDim] weight rather than one slice per head.
        guard let wBuf = f.ctx.device.makeBuffer(length: headDim * 2,
                                                 options: .storageModeShared) else {
            Issue.record("alloc failed"); return
        }
        var rng = SeedTree(0x5148_4541).key("centered-perhead-weight")
        let wBits = (0..<headDim).map { _ in
            Quantization.bf16Bits(rng.uniform(-0.4, 0.4))
        }
        let wRef = wBits.map { 1.0 + Quantization.bf16ToFloat($0) }
        let wPtr = wBuf.contents().bindMemory(to: UInt16.self, capacity: headDim)
        for i in 0..<headDim { wPtr[i] = wBits[i] }

        let cb = f.ctx.queue.makeCommandBuffer()!
        f.kernel.encodeBF16WPerHeadCentered(
            commandBuffer: cb, x: f.xBuf, weight: wBuf, out: f.yBuf,
            headDim: UInt32(headDim), numHeads: heads, eps: Self.eps)
        cb.commit(); cb.waitUntilCompleted()

        let actual = Fp16Buffer.read(f.yBuf, count: heads * headDim)
        for head in 0..<heads {
            let range = (head * headDim)..<((head + 1) * headDim)
            let ref = RmsNormRef.apply(x: Array(f.xRef[range]),
                                       weight: wRef, eps: Self.eps)
            let relErr = RelError.compute(actual: Array(actual[range]),
                                          reference: ref)
            #expect(relErr < Tolerance.fp16Reduction,
                    "head \(head): relErr=\(relErr)")
        }
    }

    /// The offset is the whole point: with weights centered at zero the
    /// centered kernel and the plain one must disagree substantially.
    @Test func centeredAndPlainDisagreeOnZeroCenteredWeights() throws {
        let d = 512
        guard let f = try Self.makeFixture(
            count: d, seed: 0x4449_4646, label: "centered-vs-plain") else {
            Issue.record("alloc failed"); return
        }
        guard let plainOut = Fp16Buffer.make(f.ctx.device, count: d) else {
            Issue.record("alloc failed"); return
        }
        let cb = f.ctx.queue.makeCommandBuffer()!
        f.kernel.encodeBF16WGroupedCentered(
            commandBuffer: cb, x: f.xBuf, weight: f.wBuf, out: f.yBuf,
            d: UInt32(d), groups: 1, eps: Self.eps)
        f.kernel.encodeBF16W(
            commandBuffer: cb, x: f.xBuf, weight: f.wBuf, out: plainOut,
            d: UInt32(d), eps: Self.eps)
        cb.commit(); cb.waitUntilCompleted()

        let centered = Fp16Buffer.read(f.yBuf, count: d)
        let plain = Fp16Buffer.read(plainOut, count: d)
        #expect(RelError.maxAbsDiff(centered, plain) > 0.1,
                "the centered kernel is not applying its offset")
    }
}
