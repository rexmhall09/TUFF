import Testing
import Metal
import Foundation
@testable import TUFFEngine

@Suite struct QwenSparseAttentionTests {
    private func buffer<T>(_ values: [T], _ ctx: MetalContext) throws -> MTLBuffer {
        try values.withUnsafeBytes {
            try #require(ctx.device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared))
        }
    }
    private func parameters(blocks: Int, top: Int, tokens: Int = 1) -> QwenSparseAttention.Parameters {
        .init(start: UInt32(blocks * 4 + 2 - tokens), tokens: UInt32(tokens), dim: 128,
              heads: 4, ratio: 4, topBlocks: UInt32(top), blocks: UInt32(blocks), rotary: 64,
              attentionDim: 64, attentionHeads: 4, kvHeads: 2, partitions: 16, theta: 10_000_000, scale: 0.125)
    }
    @Test(arguments: [17, 513, 65_536])
    func radixSelectionMatchesSortedReference(blocks: Int) throws {
        let ctx = try MetalContext()
        let top = min(512, blocks - 1)
        // Includes many ties and zero scores; cutoff ties choose lower IDs.
        let values = (0..<blocks).map { Float(($0 * 71) % 991) / 37 }
        let input = try buffer(values, ctx)
        let selected = try buffer([UInt32](repeating: .max, count: top), ctx)
        var p = parameters(blocks: blocks, top: top)
        let cb = try #require(ctx.queue.makeCommandBuffer())
        let e = try #require(cb.makeComputeCommandEncoder())
        e.setComputePipelineState(try ctx.pipeline("qsa_select"))
        e.setBuffer(input, offset: 0, index: 0)
        e.setBuffer(selected, offset: 0, index: 1)
        e.setBytes(&p, length: MemoryLayout.size(ofValue: p), index: 2)
        e.dispatchThreadgroups(.init(width: 1, height: 1, depth: 1), threadsPerThreadgroup: .init(width: 256, height: 1, depth: 1))
        e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        try checkCommandBufferError(cb.error)
        let actual = Set(UnsafeBufferPointer(start: selected.contents().assumingMemoryBound(to: UInt32.self), count: top).map(Int.init))
        let expected = Set((0..<blocks).sorted { values[$0] == values[$1] ? $0 < $1 : values[$0] > values[$1] }.prefix(top))
        #expect(actual == expected)
    }

    @Test func indexQueriesAndPooledKeysMatchNormalizationAndMRoPEReference() throws {
        let ctx = try MetalContext()
        let count = 9, dim = 128, heads = 4
        var p = parameters(blocks: 2, top: 1, tokens: count)
        p.start = 0
        let values = (0..<(count * (heads + 1) * dim)).map { Float16(sin(Float($0) * 0.19) * 0.7) }
        let normValues = (0..<dim).map { Quantization.bf16Bits(Float($0 % 11) * 0.013 - 0.03) }
        var xyz: [Int32] = []
        for t in 0..<count {
            xyz.append(Int32(t / 4))
            xyz.append(Int32(t / 2))
            xyz.append(Int32(t % 3))
        }
        let projected = try buffer(values, ctx), norm = try buffer(normValues, ctx)
        let positions = try buffer([Int32](repeating: 0, count: count * 3), ctx), current = try buffer(xyz, ctx)
        let raw = try buffer([Float16](repeating: 0, count: count * dim), ctx)
        let queries = try buffer([Float16](repeating: 0, count: count * heads * dim), ctx)
        let pooled = try buffer([Float16](repeating: 0, count: 2 * dim), ctx)
        let cb = try #require(ctx.queue.makeCommandBuffer())
        for name in ["prepare", "pool"] {
            let e = try #require(cb.makeComputeCommandEncoder())
            e.setComputePipelineState(try ctx.pipeline("qsa_" + name))
            let buffers = name == "prepare" ? [projected, norm, queries, raw, positions, current] : [raw, norm, positions, pooled]
            for (i, b) in buffers.enumerated() { e.setBuffer(b, offset: 0, index: i) }
            e.setBytes(&p, length: MemoryLayout.size(ofValue: p), index: buffers.count)
            e.dispatchThreadgroups(.init(width: name == "prepare" ? heads + 1 : 2,
                height: name == "prepare" ? count : 1, depth: 1),
                threadsPerThreadgroup: .init(width: 128, height: 1, depth: 1))
            e.endEncoding()
        }
        cb.commit(); cb.waitUntilCompleted(); try checkCommandBufferError(cb.error)
        func reference(_ x: [Float], token: Int) -> [Float] {
            let inv = 1 / sqrt(x.reduce(Float(0)) { $0 + $1 * $1 } / Float(dim) + 1e-6)
            let normalized = x.enumerated().map { d, v in Float(Float16(v * inv * (1 + Quantization.bf16ToFloat(normValues[d])))) }
            var out = normalized
            for i in 0..<32 {
                let axis = i % 3 == 1 && i < 33 ? 1 : (i % 3 == 2 && i < 30 ? 2 : 0)
                let angle = Float(xyz[token * 3 + axis]) * pow(p.theta, -Float(2 * i) / 64)
                out[i] = Float(Float16(normalized[i] * cos(angle) - normalized[i + 32] * sin(angle)))
                out[i + 32] = Float(Float16(normalized[i + 32] * cos(angle) + normalized[i] * sin(angle)))
            }
            return out
        }
        var maxError: Float = 0
        let q = queries.contents().assumingMemoryBound(to: Float16.self)
        for t in 0..<count {
            for h in 0..<heads {
                let start = (t * (heads + 1) + h) * dim
                let expected = reference(values[start..<(start + dim)].map(Float.init), token: t)
                for d in 0..<dim { maxError = max(maxError, abs(expected[d] - Float(q[(t * heads + h) * dim + d]))) }
            }
        }
        let k = pooled.contents().assumingMemoryBound(to: Float16.self)
        for b in 0..<2 {
            let mean = (0..<dim).map { d -> Float in
                var sum: Float = 0
                for t in 0..<4 { sum += Float(values[((b * 4 + t) * (heads + 1) + heads) * dim + d]) }
                return Float(Float16(sum / 4))
            }
            let expected = reference(mean, token: b * 4)
            for d in 0..<dim { maxError = max(maxError, abs(expected[d] - Float(k[b * dim + d]))) }
        }
        #expect(maxError < 0.004)
    }

    @Test func sparseAttentionMatchesCPUWithCausalTailAndDensePrefix() throws {
        let ctx = try MetalContext()
        let tokens = 9, blocks = 4, top = 2, dim = 64, heads = 4, kvHeads = 2
        var p = parameters(blocks: blocks, top: top, tokens: tokens)
        let end = Int(p.start) + tokens
        let qs = (0..<(tokens * heads * dim)).map { Float16(sin(Float($0) * 0.17) * 0.3) }
        let ks = (0..<(end * kvHeads * dim)).map { Float16(cos(Float($0) * 0.11) * 0.4) }
        let vs = (0..<(end * kvHeads * dim)).map { Float16(sin(Float($0) * 0.31)) }
        // Select the first and last complete block, with no future rows.
        let ids = (0..<tokens).flatMap { t -> [UInt32] in
            let complete = (Int(p.start) + t + 1) / 4
            return [0, UInt32(max(1, complete - 1))]
        }
        let q = try buffer(qs, ctx), k = try buffer(ks, ctx), v = try buffer(vs, ctx)
        let selected = try buffer(ids, ctx)
        let partial = try buffer([Float](repeating: 0, count: tokens * heads * 16 * (dim + 2)), ctx)
        let output = try buffer([Float16](repeating: 0, count: qs.count), ctx)
        let cb = try #require(ctx.queue.makeCommandBuffer())
        for name in ["partial", "combine"] {
            let e = try #require(cb.makeComputeCommandEncoder())
            e.setComputePipelineState(try ctx.pipeline("qsa_attention_" + name))
            let buffers = name == "partial" ? [q, k, v, selected, partial] : [partial, output]
            for (i, b) in buffers.enumerated() { e.setBuffer(b, offset: 0, index: i) }
            e.setBytes(&p, length: MemoryLayout.size(ofValue: p), index: buffers.count)
            e.dispatchThreadgroups(.init(width: heads, height: tokens, depth: name == "partial" ? 16 : 1), threadsPerThreadgroup: .init(width: 32, height: 1, depth: 1))
            e.endEncoding()
        }
        cb.commit(); cb.waitUntilCompleted(); try checkCommandBufferError(cb.error)
        let actual = output.contents().assumingMemoryBound(to: Float16.self)
        var maxError: Float = 0
        for t in 0..<tokens {
            let visible = Int(p.start) + t + 1, complete = visible / 4
            var rows = (0..<min(top, complete)).flatMap { b in (0..<4).map { Int(ids[t * top + b]) * 4 + $0 } }
            rows += Array((complete * 4)..<visible)
            for h in 0..<heads {
                let scores = rows.map { row -> Float in
                    var dot: Float = 0
                    for d in 0..<dim { dot += Float(qs[(t * heads + h) * dim + d]) * Float(ks[(row * kvHeads + h / 2) * dim + d]) }
                    return dot * p.scale
                }
                let mx = scores.max()!, weights = scores.map { exp($0 - mx) }, sum = weights.reduce(0, +)
                for d in 0..<dim {
                    var reference: Float = 0
                    for (i, row) in rows.enumerated() { reference += weights[i] * Float(vs[(row * kvHeads + h / 2) * dim + d]) / sum }
                    maxError = max(maxError, abs(reference - Float(actual[(t * heads + h) * dim + d])))
                }
            }
        }
        #expect(maxError < 0.002)
    }
}
