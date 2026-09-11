import Testing
import Foundation
import Metal
@testable import TUFFEngine
import TUFFValidationSupport

/// Checks the hyper-connection kernels against mlx-vlm's own
/// `Qwen4ExpGatedResidual`, run over the pinned checkpoint's layer-0 weights.
///
/// `HyperConnectionTests` checks the same kernels against a transcription of
/// the reference written by hand. That catches a kernel that does not do what
/// the transcription says; this catches a transcription that does not say what
/// the reference does. Neither subsumes the other, and the second failure mode
/// is the one that silently produces a model which loads and talks nonsense.
///
/// See `Fixtures/qwen4exp/README.md` for how the vectors are regenerated.
@Suite struct HyperConnectionGoldenTests {

    // MARK: - Fixture

    private struct Golden {
        let tensors: [String: (shape: [Int], values: [Float])]

        subscript(name: String) -> [Float] {
            guard let entry = tensors[name] else {
                Issue.record("golden fixture has no tensor \(name)")
                return []
            }
            return entry.values
        }

        func shape(_ name: String) -> [Int] { tensors[name]?.shape ?? [] }
    }

    private static func loadGolden() throws -> Golden? {
        guard let indexURL = Bundle.module.url(forResource: "golden",
                                               withExtension: "json",
                                               subdirectory: "qwen4exp"),
              let blobURL = Bundle.module.url(forResource: "golden",
                                              withExtension: "bin",
                                              subdirectory: "qwen4exp") else {
            Issue.record("qwen4_exp golden fixture is missing from the bundle")
            return nil
        }
        struct Index: Decodable {
            struct Tensor: Decodable {
                let name: String
                let shape: [Int]
                let offset: Int
                let byteCount: Int
            }
            let tensors: [Tensor]
            let revision: String
        }
        let index = try JSONDecoder().decode(
            Index.self, from: Data(contentsOf: indexURL))
        // The vectors are only meaningful for the revision the catalogue pins.
        #expect(index.revision == "07b5dc6c54600a359b87f1e53e7adf6351c72a2c")

        let blob = try Data(contentsOf: blobURL)
        var tensors: [String: (shape: [Int], values: [Float])] = [:]
        for tensor in index.tensors {
            let end = tensor.offset + tensor.byteCount
            guard end <= blob.count else {
                Issue.record("golden tensor \(tensor.name) runs past the blob")
                return nil
            }
            let slice = blob[tensor.offset..<end]
            let values = slice.withUnsafeBytes { raw -> [Float] in
                Array(raw.bindMemory(to: Float.self))
            }
            tensors[tensor.name] = (tensor.shape, values)
        }
        return Golden(tensors: tensors)
    }

    private static let streams = 4
    private static let hidden = 2_560

    private static func fp16Buffer(_ device: MTLDevice,
                                   _ values: [Float]) -> MTLBuffer? {
        Fp16Buffer.make(device, halves: values.map { Float16($0) })
    }

    /// Both hyper-connections in the layer, so a kernel that happens to suit
    /// one weight set is not mistaken for a correct one.
    private static let slots = ["attn", "mlp"]

    // MARK: - Tests

    /// `normed = hc_norm(hyper)`: four streams of 2,560 normalized
    /// independently, scaled by `1 + w` against a 10,240-wide weight.
    ///
    /// This is the check that would have caught reusing Qwen3.6's plain
    /// `x * inv * w` kernel here.
    @Test func groupedCenteredNormMatchesTheReference() throws {
        guard let golden = try Self.loadGolden() else { return }
        let ctx = try MetalContext()
        let kernel = try RMSNorm(context: ctx)

        for slot in Self.slots {
            let tag = "module.layer00.\(slot)_hc"
            let input = golden["\(tag).input"]
            let weight = golden["\(tag).hc_norm_weight"]
            let expected = golden["\(tag).normed"]
            let positions = input.count / (Self.streams * Self.hidden)
            #expect(positions > 0)

            guard let wBuf = ctx.device.makeBuffer(
                length: weight.count * 2, options: .storageModeShared) else {
                Issue.record("alloc failed"); return
            }
            let wPtr = wBuf.contents().bindMemory(to: UInt16.self,
                                                  capacity: weight.count)
            for i in 0..<weight.count {
                wPtr[i] = Quantization.bf16Bits(weight[i])
            }

            for position in 0..<positions {
                let span = Self.streams * Self.hidden
                let range = (position * span)..<((position + 1) * span)
                guard let xBuf = Self.fp16Buffer(ctx.device,
                                                 Array(input[range])),
                      let yBuf = Fp16Buffer.make(ctx.device, count: span) else {
                    Issue.record("alloc failed"); return
                }
                let cb = ctx.queue.makeCommandBuffer()!
                kernel.encodeBF16WGroupedCentered(
                    commandBuffer: cb, x: xBuf, weight: wBuf, out: yBuf,
                    d: UInt32(Self.hidden), groups: Self.streams, eps: 1e-6)
                cb.commit(); cb.waitUntilCompleted()

                let actual = Fp16Buffer.read(yBuf, count: span)
                let relErr = RelError.compute(actual: actual,
                                              reference: Array(expected[range]))
                #expect(relErr < Tolerance.fp16Reduction,
                        "\(tag) position \(position): relErr=\(relErr)")
            }
        }
    }

    /// `lowrank = silu(mix_down / streams)`.
    @Test func lowRankActivationMatchesTheReference() throws {
        guard let golden = try Self.loadGolden() else { return }
        let ctx = try MetalContext()
        let kernel = try HyperConnection(context: ctx)

        for slot in Self.slots {
            let tag = "module.layer00.\(slot)_hc"
            let down = golden["\(tag).mix_down"]
            let expected = golden["\(tag).mix_lowrank"]
            guard let xBuf = Self.fp16Buffer(ctx.device, down),
                  let yBuf = Fp16Buffer.make(ctx.device, count: down.count) else {
                Issue.record("alloc failed"); return
            }
            let cb = ctx.queue.makeCommandBuffer()!
            kernel.encodeLowRankSilu(commandBuffer: cb, x: xBuf, out: yBuf,
                                     count: down.count,
                                     streamCount: Self.streams)
            cb.commit(); cb.waitUntilCompleted()

            let actual = Fp16Buffer.read(yBuf, count: down.count)
            #expect(RelError.compute(actual: actual, reference: expected)
                    < Tolerance.fp16Reduction, "\(tag) low-rank activation")
        }
    }

    /// `mixed = mean over streams of sigmoid(mix_up) * normed`, the value the
    /// attention or MLP block actually consumes.
    @Test func combineMatchesTheReference() throws {
        guard let golden = try Self.loadGolden() else { return }
        let ctx = try MetalContext()
        let kernel = try HyperConnection(context: ctx)

        for slot in Self.slots {
            let tag = "module.layer00.\(slot)_hc"
            let up = golden["\(tag).mix_up"]
            let normed = golden["\(tag).normed"]
            let expected = golden["\(tag).mixed"]
            let span = Self.streams * Self.hidden
            let positions = up.count / span

            for position in 0..<positions {
                let range = (position * span)..<((position + 1) * span)
                let outRange = (position * Self.hidden)..<((position + 1) * Self.hidden)
                guard let upBuf = Self.fp16Buffer(ctx.device, Array(up[range])),
                      let normedBuf = Self.fp16Buffer(ctx.device,
                                                      Array(normed[range])),
                      let outBuf = Fp16Buffer.make(ctx.device,
                                                   count: Self.hidden) else {
                    Issue.record("alloc failed"); return
                }
                let cb = ctx.queue.makeCommandBuffer()!
                kernel.encodeCombine(commandBuffer: cb, up: upBuf,
                                     normed: normedBuf, mixed: outBuf,
                                     hiddenSize: Self.hidden,
                                     streamCount: Self.streams)
                cb.commit(); cb.waitUntilCompleted()

                let actual = Fp16Buffer.read(outBuf, count: Self.hidden)
                let relErr = RelError.compute(actual: actual,
                                              reference: Array(expected[outRange]))
                #expect(relErr < Tolerance.fp16Reduction,
                        "\(tag) position \(position): relErr=\(relErr)")
            }
        }
    }

    /// The injection gate, `2 * sigmoid(injection_raw / streams)`, and the
    /// accumulation it drives.
    @Test func injectionGateAndAccumulationMatchTheReference() throws {
        guard let golden = try Self.loadGolden() else { return }
        let ctx = try MetalContext()
        let kernel = try HyperConnection(context: ctx)

        for slot in Self.slots {
            let tag = "module.layer00.\(slot)_hc"
            let raw = golden["\(tag).injection_raw"]
            let gates = golden["\(tag).injection_weights"]
            let hyper = golden["\(tag).input"]
            let span = Self.streams * Self.hidden
            let positions = raw.count / Self.streams

            for position in 0..<positions {
                let gateRange = (position * Self.streams)..<((position + 1) * Self.streams)
                let hyperRange = (position * span)..<((position + 1) * span)
                // A branch output of all ones turns the accumulated delta into
                // the gate itself, so the kernel's gate is read back directly
                // rather than inferred.
                let branch = [Float](repeating: 1, count: Self.hidden)
                guard let hyperBuf = Self.fp16Buffer(ctx.device,
                                                     Array(hyper[hyperRange])),
                      let branchBuf = Self.fp16Buffer(ctx.device, branch),
                      let rawBuf = Self.fp16Buffer(ctx.device,
                                                   Array(raw[gateRange])) else {
                    Issue.record("alloc failed"); return
                }
                let before = Fp16Buffer.read(hyperBuf, count: span)
                let cb = ctx.queue.makeCommandBuffer()!
                kernel.encodeInject(commandBuffer: cb, hyper: hyperBuf,
                                    branch: branchBuf, injectionRaw: rawBuf,
                                    hiddenSize: Self.hidden,
                                    streamCount: Self.streams)
                cb.commit(); cb.waitUntilCompleted()
                let after = Fp16Buffer.read(hyperBuf, count: span)

                for stream in 0..<Self.streams {
                    let index = stream * Self.hidden
                    let applied = after[index] - before[index]
                    let expected = gates[gateRange.lowerBound + stream]
                    let detail = "\(tag) position \(position) stream "
                        + "\(stream): applied \(applied) vs \(expected)"
                    #expect(abs(applied - expected) < 0.01, "\(detail)")
                }
            }
        }
    }

    /// The residual that leaves a hyper-connection is the input, untouched.
    /// The reference returns it alongside the normalized stream, and confusing
    /// the two is the kind of mistake that still produces fluent output.
    @Test func theResidualPassesThroughUnnormalized() throws {
        guard let golden = try Self.loadGolden() else { return }
        for slot in Self.slots {
            let tag = "module.layer00.\(slot)_hc"
            let input = golden["\(tag).input"]
            let passthrough = golden["\(tag).passthrough"]
            let normed = golden["\(tag).normed"]
            #expect(RelError.maxAbsDiff(input, passthrough) == 0,
                    "\(tag): the reference did not pass its input through")
            #expect(RelError.maxAbsDiff(input, normed) > 0.1,
                    "\(tag): normalization was a no-op, so this proves nothing")
        }
    }
}
