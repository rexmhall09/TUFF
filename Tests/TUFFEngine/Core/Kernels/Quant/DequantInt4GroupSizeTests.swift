import Testing
import Foundation
import Metal
@testable import TUFFEngine
import TUFFValidationSupport

/// The INT4 GEMV decoded one scale and bias per 64 values from the day it was
/// written, because every checkpoint in the lineup was quantized that way.
/// Qwen3.8 Flash Next is quantized at 32, and not by preference: its n-gram
/// PLE rows are 160 values wide, and 160 is not divisible by 64, so no
/// group-64 conversion of that architecture can exist.
///
/// The group size is now a function constant. These tests check that the new
/// path decodes correctly and — just as important — that the old one is
/// untouched, since every shipping model runs through it.
@Suite struct DequantInt4GroupSizeTests {

    private struct Fixture {
        let weights: MTLBuffer
        let scales: MTLBuffer
        let biases: MTLBuffer
        let x: MTLBuffer
        let y: MTLBuffer
        let reference: [Float]
    }

    /// Build an `m x n` INT4 matrix and an FP16 vector, plus the exact product
    /// of the dequantized weights with that vector.
    private static func makeFixture(device: MTLDevice,
                                    m: Int, n: Int,
                                    groupSize: Int,
                                    seed: UInt64) -> Fixture? {
        var rng = SeedTree(seed).key("int4-g\(groupSize)-\(m)x\(n)")
        let groupsPerRow = n / groupSize

        var packed = [UInt8]()
        var scaleBits = [UInt16]()
        var biasBits = [UInt16]()
        var dequantized = [[Float]]()
        packed.reserveCapacity(m * n / 2)

        for _ in 0..<m {
            let row = (0..<n).map { _ in rng.uniform(-1.0, 1.0) }
            let quantized = Quantization.quantizeInt4Affine(row, groupSize: groupSize)
            packed.append(contentsOf: quantized.packed)
            scaleBits.append(contentsOf: quantized.scales)
            biasBits.append(contentsOf: quantized.biases)
            dequantized.append(Quantization.dequantizeInt4Affine(
                quantized, n: n, groupSize: groupSize))
        }

        let xFp32 = (0..<n).map { _ in rng.uniform(-1.0, 1.0) }
        let xFp16 = xFp32.map { Float16($0) }
        let xRef = xFp16.map { Float($0) }

        // The kernel accumulates in float; so does this.
        let reference = (0..<m).map { row -> Float in
            var acc: Float = 0
            for i in 0..<n { acc += dequantized[row][i] * xRef[i] }
            return acc
        }

        guard let wBuf = device.makeBuffer(length: packed.count,
                                           options: .storageModeShared),
              let sBuf = device.makeBuffer(length: scaleBits.count * 2,
                                           options: .storageModeShared),
              let bBuf = device.makeBuffer(length: biasBits.count * 2,
                                           options: .storageModeShared),
              let xBuf = Fp16Buffer.make(device, halves: xFp16),
              let yBuf = Fp16Buffer.make(device, count: m) else {
            return nil
        }
        packed.withUnsafeBytes {
            wBuf.contents().copyMemory(from: $0.baseAddress!, byteCount: packed.count)
        }
        scaleBits.withUnsafeBytes {
            sBuf.contents().copyMemory(from: $0.baseAddress!, byteCount: scaleBits.count * 2)
        }
        biasBits.withUnsafeBytes {
            bBuf.contents().copyMemory(from: $0.baseAddress!, byteCount: biasBits.count * 2)
        }
        #expect(scaleBits.count == m * groupsPerRow)
        return Fixture(weights: wBuf, scales: sBuf, biases: bBuf,
                       x: xBuf, y: yBuf, reference: reference)
    }

    private static func run(m: Int, n: Int, groupSize: Int,
                            seed: UInt64) throws -> (actual: [Float],
                                                     reference: [Float])? {
        let ctx = try MetalContext()
        let kernel = try DequantInt4GEMV(context: ctx, groupSize: groupSize)
        #expect(kernel.groupSize == groupSize)
        guard let f = makeFixture(device: ctx.device, m: m, n: n,
                                  groupSize: groupSize, seed: seed) else {
            Issue.record("alloc failed"); return nil
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encode(commandBuffer: cb,
                      weights: f.weights, scales: f.scales, biases: f.biases,
                      x: f.x, y: f.y, m: UInt32(m), n: UInt32(n))
        cb.commit(); cb.waitUntilCompleted()
        return (Fp16Buffer.read(f.y, count: m), f.reference)
    }

    /// Qwen3.8 Flash Next's real decode widths. 2,560 is the hidden size,
    /// 6,144 the gated attention output, 640 the expert intermediate — none of
    /// which are multiples of 64 groups in a way that would hide an indexing
    /// slip: 640/32 = 20 groups, which is not a multiple of the eight groups a
    /// 128-byte block spans, so the remainder loop runs.
    @Test(arguments: [(64, 2_560), (128, 640), (256, 6_144), (24, 320)])
    func groupOf32MatchesTheDequantizedProduct(shape: (m: Int, n: Int)) throws {
        guard let r = try Self.run(m: shape.m, n: shape.n, groupSize: 32,
                                   seed: 0x4732_3200) else { return }
        let relErr = RelError.compute(actual: r.actual, reference: r.reference)
        #expect(relErr < Tolerance.fp16Reduction,
                "g32 \(shape.m)x\(shape.n): relErr=\(relErr)")
    }

    /// The path every shipping model uses. If this moves, something that was
    /// working is now broken.
    @Test(arguments: [(64, 2_816), (128, 2_048), (256, 4_096)])
    func groupOf64IsUnchanged(shape: (m: Int, n: Int)) throws {
        guard let r = try Self.run(m: shape.m, n: shape.n, groupSize: 64,
                                   seed: 0x4736_3400) else { return }
        let relErr = RelError.compute(actual: r.actual, reference: r.reference)
        #expect(relErr < Tolerance.fp16Reduction,
                "g64 \(shape.m)x\(shape.n): relErr=\(relErr)")
    }

    /// Decoding group-32 data with the group-64 kernel must produce garbage.
    /// Without this the two tests above could both pass on a kernel that
    /// quietly ignored the constant.
    @Test func theGroupSizeActuallyChangesTheDecode() throws {
        let m = 64, n = 2_560
        let ctx = try MetalContext()
        guard let f = Self.makeFixture(device: ctx.device, m: m, n: n,
                                       groupSize: 32, seed: 0x4D49_5358) else {
            Issue.record("alloc failed"); return
        }
        // Same group-32 bytes, read by a kernel expecting groups of 64.
        let wrong = try DequantInt4GEMV(context: ctx, groupSize: 64)
        let cb = ctx.queue.makeCommandBuffer()!
        wrong.encode(commandBuffer: cb,
                     weights: f.weights, scales: f.scales, biases: f.biases,
                     x: f.x, y: f.y, m: UInt32(m), n: UInt32(n))
        cb.commit(); cb.waitUntilCompleted()
        let mismatched = Fp16Buffer.read(f.y, count: m)
        let relErr = RelError.compute(actual: mismatched, reference: f.reference)
        #expect(relErr > 0.05, """
            the group-64 kernel reproduced group-32 data; the function             constant is not reaching the kernel
            """)
    }

    /// The kernels the decode path actually strings together, each built at
    /// group 32 and each asked for a width that is a multiple of 32 but not of
    /// 64. Constructing them is the test: a wrapper that kept a `% 64`
    /// precondition, or forgot to pass the constant to one of its pipelines,
    /// fails here rather than at the end of a 110 GB install.
    @Test func theDecodeChainConstructsAtGroupOf32() throws {
        let ctx = try MetalContext()

        let embed = try EmbedLookupInt4(context: ctx, groupSize: 32)
        #expect(embed.groupSize == 32)

        let gemv = try DequantInt4GEMV(context: ctx, groupSize: 32)
        #expect(gemv.groupSize == 32)

        // Ten routed experts and group-32 expert weights, as the checkpoint
        // presents them.
        let moe = try MoE(context: ctx,
                          siluActivation: true,
                          specializedD: 2_560,
                          specializedF: 640,
                          specializedNumExperts: 512,
                          topKExperts: 10,
                          groupSize: 32)
        #expect(moe.groupSize == 32)

        let shared = try SharedExpertInt4(context: ctx,
                                          siluActivation: true,
                                          groupSize: 32)
        _ = shared

        let head = try LMHeadChainInt4(context: ctx,
                                       maxD: 2_560,
                                       maxVocab: 248_320,
                                       groupSize: 32)
        #expect(head.groupSize == 32)
    }

    /// 160 is the n-gram PLE row width, and the reason the checkpoint cannot
    /// be quantized at 64 at all: it is a multiple of 32 and not of 64. A
    /// width like this has to reach the kernel rather than trip a precondition
    /// left over from when 64 was the only group size.
    @Test func aWidthOnlyValidAtGroupOf32IsAccepted() throws {
        #expect(160 % 32 == 0)
        #expect(160 % 64 != 0)
        guard let r = try Self.run(m: 64, n: 160, groupSize: 32,
                                   seed: 0x504C_4531) else { return }
        let relErr = RelError.compute(actual: r.actual, reference: r.reference)
        #expect(relErr < Tolerance.fp16Reduction, "160-wide: relErr=\(relErr)")
    }

    /// Specialized shape variants have to carry the group size too, or a model
    /// would decode correctly at one width and wrongly at another.
    @Test func shapeSpecializedVariantsKeepTheGroupSize() throws {
        let m = 256, n = 6_144
        let ctx = try MetalContext()
        let kernel = try DequantInt4GEMV(
            context: ctx, additionalShapes: [(m: m, n: n)], groupSize: 32)
        guard let f = Self.makeFixture(device: ctx.device, m: m, n: n,
                                       groupSize: 32, seed: 0x5350_4543) else {
            Issue.record("alloc failed"); return
        }
        let cb = ctx.queue.makeCommandBuffer()!
        kernel.encode(commandBuffer: cb,
                      weights: f.weights, scales: f.scales, biases: f.biases,
                      x: f.x, y: f.y, m: UInt32(m), n: UInt32(n))
        cb.commit(); cb.waitUntilCompleted()
        let relErr = RelError.compute(actual: Fp16Buffer.read(f.y, count: m),
                                      reference: f.reference)
        #expect(relErr < Tolerance.fp16Reduction,
                "specialized g32 \(m)x\(n): relErr=\(relErr)")
    }
}
