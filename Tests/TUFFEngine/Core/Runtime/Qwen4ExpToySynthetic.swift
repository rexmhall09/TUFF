import Foundation
@testable import TUFFEngine
import TUFFFormat
@testable import TUFFRepackCore

/// Synthetic Qwen4-Exp toy fixture: a tiny runnable `.gturbo/` directory with
/// the `qwen4_exp` tensor-name contract.
///
/// What it exercises that `QwenToySynthetic` cannot:
///
/// - **Hyper-connections.** Four residual streams instead of one, and *no*
///   `input_layernorm`, `post_attention_layernorm` or `model.norm` anywhere —
///   the two per-layer hyper-connection modules and the model-level mixer
///   carry every norm the architecture has.
/// - **Group-32 INT4.** Every 4-bit tensor is grouped at 32 while the 8-bit
///   router and shared-expert gate stay at 64, as the real checkpoint has them.
/// - **Top-10 routing**, past the eight the kernels were written for.
///
/// The weights are uniform stand-ins, so this says nothing about numerical
/// agreement with the reference — `HyperConnectionGoldenTests` covers that.
/// What it proves is that the shapes, tensor names, manifest and layer graph
/// compose into something the runner can execute.
enum Qwen4ExpToySynthetic {

    /// The toy n-gram table: four heads over a few thousand rows, split in
    /// two shards. Small enough to write, large enough that a row resolved to
    /// the wrong shard reads different values.
    enum Ngram {
        static let headVocabSizes: [Int64] = [1009, 1013, 1019, 1021]
        static let headOffsets: [Int64] = [0, 1009, 2022, 3041]
        static let multipliers: [Int64] = [
            23_703_573_157_769, 20_109_073_645_365, 8_052_911_324_071,
        ]
        static let shardCount = 2
        static var rowCount: Int64 { headOffsets[3] + headVocabSizes[3] }
        static let eosTokenID: Int32 = 2

        /// Deterministic contents, varying with both row and position so a
        /// misaddressed read cannot coincidentally match.
        static func nibble(row: Int, index: Int) -> UInt8 {
            UInt8((row * 5 + index * 3) % 16)
        }
        static func scale(row: Int) -> Float { 0.01 + Float(row % 5) * 0.001 }
        static func bias(row: Int) -> Float { Float(row % 3) * 0.002 - 0.002 }
    }

    /// `config` defaults to the four-layer toy; pass a narrower one to
    /// isolate a single layer kind while debugging.
    static func write(config: ArchConfig = .qwen4ExpToy()) throws -> URL {
        let toy = config
        let la = toy.linearAttention
        let hc = toy.hyperConnection
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-qwen4exp-toy-\(UUID().uuidString)")
        let exp = dir.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(at: exp, withIntermediateDirectories: true)

        struct ResidentSpec {
            let name: String
            let dtype: UInt8
            let shape: [UInt32]
            let weightBytes: UInt64
            let scaleBytes: UInt64
            let biasBytes: UInt64
        }

        let d = toy.hiddenSize
        let stacked = hc.streamCount * d
        let u16 = MemoryLayout<UInt16>.stride
        // 4-bit tensors are grouped at 32 here; 8-bit ones stay at 64.
        let int4Groups = toy.int4GroupSize
        let int8Groups = Quantization.groupSize

        func int4AffineSpec(_ name: String, rows: Int, cols: Int) -> ResidentSpec {
            precondition(cols % int4Groups == 0,
                         "\(name): \(cols) is not a multiple of \(int4Groups)")
            let groups = cols / int4Groups
            let auxBytes = UInt64(rows * groups * u16)
            return ResidentSpec(name: name, dtype: 0,
                                shape: [UInt32(rows), UInt32(cols), 0, 0],
                                weightBytes: UInt64(rows * cols / 2),
                                scaleBytes: auxBytes, biasBytes: auxBytes)
        }

        func int8AffineSpec(_ name: String, rows: Int, cols: Int) -> ResidentSpec {
            precondition(cols % int8Groups == 0,
                         "\(name): \(cols) is not a multiple of \(int8Groups)")
            let groups = cols / int8Groups
            let auxBytes = UInt64(rows * groups * u16)
            return ResidentSpec(name: name, dtype: 0,
                                shape: [UInt32(rows), UInt32(cols), 0, 0],
                                weightBytes: UInt64(rows * cols),
                                scaleBytes: auxBytes, biasBytes: auxBytes)
        }

        func bf16Spec(_ name: String, shape: [UInt32], count: Int) -> ResidentSpec {
            ResidentSpec(name: name, dtype: 1, shape: shape,
                         weightBytes: UInt64(count * u16),
                         scaleBytes: 0, biasBytes: 0)
        }

        /// One hyper-connection's four tensors. `layer` nil builds the
        /// model-level mixer, which collapses the streams and therefore has no
        /// injection gate.
        func hyperConnectionSpecs(slot: String, layer: Int?) -> [ResidentSpec] {
            let prefix = layer.map { "language_model.model.layers.\($0).\(slot)" }
                ?? "language_model.model.hyper_connection_mixer"
            var out: [ResidentSpec] = [
                bf16Spec("\(prefix).hc_norm.weight",
                         shape: [UInt32(stacked), 0, 0, 0], count: stacked),
                int4AffineSpec("\(prefix).input_mix_weight_down.weight",
                               rows: hc.lowRank, cols: stacked),
                int4AffineSpec("\(prefix).input_mix_weight_up.weight",
                               rows: stacked, cols: hc.lowRank),
            ]
            if layer != nil {
                out.append(int4AffineSpec("\(prefix).block_inject_weight.weight",
                                          rows: hc.streamCount, cols: stacked))
            }
            return out
        }

        // 1. Resident specs. No model.norm: the mixer is the last thing before
        //    the head, and it is a hyper-connection rather than an RMSNorm.
        var specs: [ResidentSpec] = [
            int4AffineSpec("language_model.model.embed_tokens.weight",
                           rows: toy.vocabSize, cols: d),
            int4AffineSpec("language_model.lm_head.weight",
                           rows: toy.vocabSize, cols: d),
        ]
        specs.append(contentsOf: hyperConnectionSpecs(slot: "", layer: nil))

        let ngram = toy.ngramEmbedding
        func i64Spec(_ name: String, count: Int) -> ResidentSpec {
            ResidentSpec(name: name, dtype: 5,
                         shape: [UInt32(count), 0, 0, 0],
                         weightBytes: UInt64(count * 8),
                         scaleBytes: 0, biasBytes: 0)
        }

        for L in 0..<toy.numLayers {
            let prefix = "language_model.model.layers.\(L)"
            if ngram.isEnabled && L == ngram.layer {
                let ple = "\(prefix).ple"
                specs.append(int4AffineSpec("\(ple).key_proj.weight",
                                            rows: stacked, cols: ngram.embedDim))
                specs.append(int4AffineSpec("\(ple).value_proj.weight",
                                            rows: d, cols: ngram.embedDim))
                for norm in ["norm_key", "norm_query", "norm_conv"] {
                    specs.append(bf16Spec("\(ple).\(norm).weight",
                                          shape: [UInt32(stacked), 0, 0, 0],
                                          count: stacked))
                }
                specs.append(bf16Spec(
                    "\(ple).conv1d.weight",
                    shape: [UInt32(stacked), UInt32(ngram.convKernelSize), 1, 0],
                    count: stacked * ngram.convKernelSize))
                let embed = "\(ple).ple_embedding"
                specs.append(i64Spec("\(embed).layer_multipliers",
                                     count: ngram.ngramSize))
                specs.append(i64Spec("\(embed).ngram_heads_offsets",
                                     count: ngram.heads))
                specs.append(i64Spec("\(embed).ngram_heads_vocab_sizes",
                                     count: ngram.heads))
            }
            specs.append(contentsOf: hyperConnectionSpecs(
                slot: "attn_hyper_connection", layer: L))
            specs.append(contentsOf: hyperConnectionSpecs(
                slot: "mlp_hyper_connection", layer: L))
            specs.append(int8AffineSpec("\(prefix).mlp.gate.weight",
                                        rows: toy.numExperts, cols: d))
            specs.append(int8AffineSpec("\(prefix).mlp.shared_expert_gate.weight",
                                        rows: 1, cols: d))
            specs.append(int8AffineSpec("\(prefix).mlp.shared_expert.gate_proj.weight",
                                        rows: toy.intermediateSize, cols: d))
            specs.append(int8AffineSpec("\(prefix).mlp.shared_expert.up_proj.weight",
                                        rows: toy.intermediateSize, cols: d))
            specs.append(int8AffineSpec("\(prefix).mlp.shared_expert.down_proj.weight",
                                        rows: d, cols: toy.intermediateSize))
            if toy.layerIsLinear(L) {
                specs.append(int4AffineSpec("\(prefix).linear_attn.in_proj_qkv.weight",
                                            rows: la.qkvDim, cols: d))
                specs.append(int4AffineSpec("\(prefix).linear_attn.in_proj_z.weight",
                                            rows: la.valueDim, cols: d))
                specs.append(int4AffineSpec("\(prefix).linear_attn.in_proj_a.weight",
                                            rows: la.numVHeads, cols: d))
                specs.append(int4AffineSpec("\(prefix).linear_attn.in_proj_b.weight",
                                            rows: la.numVHeads, cols: d))
                specs.append(int4AffineSpec("\(prefix).linear_attn.out_proj.weight",
                                            rows: d, cols: la.valueDim))
                specs.append(bf16Spec("\(prefix).linear_attn.conv1d.weight",
                                      shape: [UInt32(la.qkvDim),
                                              UInt32(la.convKernelSize), 1, 0],
                                      count: la.qkvDim * la.convKernelSize))
                specs.append(bf16Spec("\(prefix).linear_attn.A_log",
                                      shape: [UInt32(la.numVHeads), 0, 0, 0],
                                      count: la.numVHeads))
                specs.append(bf16Spec("\(prefix).linear_attn.dt_bias",
                                      shape: [UInt32(la.numVHeads), 0, 0, 0],
                                      count: la.numVHeads))
                specs.append(bf16Spec("\(prefix).linear_attn.norm.weight",
                                      shape: [UInt32(la.valueHeadDim), 0, 0, 0],
                                      count: la.valueHeadDim))
            } else {
                if toy.attentionIndexer.isEnabled {
                    let i = toy.attentionIndexer
                    specs.append(int4AffineSpec("\(prefix).self_attn.indexer.index_qk_proj.weight", rows: i.projectionRows, cols: d))
                    for norm in ["q_layernorm", "k_layernorm"] {
                        specs.append(bf16Spec("\(prefix).self_attn.indexer.\(norm).weight", shape: [UInt32(i.headDim), 0, 0, 0], count: i.headDim))
                    }
                }
                let qDim = toy.numHeads * toy.fullHeadDim
                let kvDim = toy.numFullKVHeads * toy.fullHeadDim
                specs.append(int4AffineSpec("\(prefix).self_attn.q_proj.weight",
                                            rows: 2 * qDim, cols: d))
                specs.append(int4AffineSpec("\(prefix).self_attn.k_proj.weight",
                                            rows: kvDim, cols: d))
                specs.append(int4AffineSpec("\(prefix).self_attn.v_proj.weight",
                                            rows: kvDim, cols: d))
                specs.append(int4AffineSpec("\(prefix).self_attn.o_proj.weight",
                                            rows: d, cols: qDim))
                specs.append(bf16Spec("\(prefix).self_attn.q_norm.weight",
                                      shape: [UInt32(toy.fullHeadDim), 0, 0, 0],
                                      count: toy.fullHeadDim))
                specs.append(bf16Spec("\(prefix).self_attn.k_norm.weight",
                                      shape: [UInt32(toy.fullHeadDim), 0, 0, 0],
                                      count: toy.fullHeadDim))
            }
        }

        // 2. Resident index + payload.
        let names = specs.map(\.name)
        let stringTable = names.joined().data(using: .utf8)!
        let headerBytes = GTurboBinary.indexHeaderBytes
        let entryBytes  = GTurboBinary.indexEntryBytes
        let entriesBase = headerBytes
        let stringTableBase = entriesBase + names.count * entryBytes
        var nameAbsOffsets: [UInt32] = []
        var cursor = 0
        for n in names {
            nameAbsOffsets.append(UInt32(stringTableBase + cursor))
            cursor += n.utf8.count
        }
        let rawIndexBytes = UInt64(stringTableBase + stringTable.count)
        let alignment = GTurboFormatV1.alignmentBytes
        let indexBytes = ((rawIndexBytes + alignment - 1) / alignment) * alignment

        var entries: [ResidentEntry] = []
        entries.reserveCapacity(specs.count)
        var payloadCursor = indexBytes
        for spec in specs {
            let weightOffset = payloadCursor
            let scaleOffset = spec.scaleBytes > 0 ? weightOffset + spec.weightBytes : 0
            let biasOffset = spec.biasBytes > 0 ? scaleOffset + spec.scaleBytes : 0
            entries.append(ResidentEntry(
                name: spec.name, dtype: spec.dtype,
                logicalShape4: spec.shape,
                fileOffset: weightOffset, sizeBytes: spec.weightBytes,
                scaleOffset: scaleOffset, scaleSize: spec.scaleBytes,
                biasOffset: biasOffset, biasSize: spec.biasBytes,
                quantSpec: nil,
                sourceWeight: ModelLoaderTests.dummySource(spec.name),
                sourceScales: nil, sourceBiases: nil))
            payloadCursor += spec.weightBytes + spec.scaleBytes + spec.biasBytes
        }
        let residentSize = payloadCursor - indexBytes
        let totalBytes = Int(indexBytes + residentSize)

        var fileBuf = [UInt8](repeating: 0, count: totalBytes)
        fileBuf.withUnsafeMutableBytes { raw in
            let base = raw.baseAddress!
            GTurboBinary.writeIndexHeader(into: base,
                                          indexSize: indexBytes,
                                          residentSize: residentSize,
                                          entryCount: UInt64(entries.count))
            for (i, e) in entries.enumerated() {
                GTurboBinary.writeIndexEntry(
                    into: base.advanced(by: entriesBase + i * entryBytes),
                    entry: e, nameOffset: nameAbsOffsets[i])
            }
            _ = stringTable.withUnsafeBytes { sb in
                memcpy(base.advanced(by: stringTableBase),
                       sb.baseAddress!, stringTable.count)
            }
            for entry in entries where entry.dtype == 0 {
                if entry.name.hasSuffix("block_inject_weight.weight") {
                    // One row per residual stream, and they have to differ:
                    // identical rows give every stream the same gate, the
                    // streams never diverge, and the extra three cost memory
                    // while carrying nothing. Uniform stand-ins elsewhere are
                    // fine; here the variation is the point.
                    let rowBytes = Int(entry.sizeBytes) / hc.streamCount
                    let rows = base.advanced(by: Int(entry.fileOffset))
                        .assumingMemoryBound(to: UInt8.self)
                    for stream in 0..<hc.streamCount {
                        let nibble = UInt8(1 + stream * 3)
                        let byte = nibble | (nibble << 4)
                        memset(rows.advanced(by: stream * rowBytes),
                               Int32(byte), rowBytes)
                    }
                } else {
                    memset(base.advanced(by: Int(entry.fileOffset)), 0x11,
                           Int(entry.sizeBytes))
                }
                if entry.scaleSize > 0 {
                    let scales = base.advanced(by: Int(entry.scaleOffset))
                        .assumingMemoryBound(to: UInt16.self)
                    for i in 0..<(Int(entry.scaleSize) / u16) {
                        scales[i] = Quantization.bf16Bits(0.01)
                    }
                }
            }
            for entry in entries where entry.dtype == 5 {
                let dst = base.advanced(by: Int(entry.fileOffset))
                    .assumingMemoryBound(to: Int64.self)
                let values: [Int64]
                if entry.name.hasSuffix("layer_multipliers") {
                    values = Ngram.multipliers
                } else if entry.name.hasSuffix("ngram_heads_offsets") {
                    values = Ngram.headOffsets
                } else {
                    values = Ngram.headVocabSizes
                }
                for (i, v) in values.enumerated() { dst[i] = v }
            }
            // Norm weights are centered at zero in this architecture, so a
            // stand-in of 0 means "scale by 1" rather than "scale by 0".
            for entry in entries where entry.dtype == 1 {
                let dst = base.advanced(by: Int(entry.fileOffset))
                    .assumingMemoryBound(to: UInt16.self)
                let isCenteredNorm = entry.name.contains(".hc_norm.")
                    || entry.name.contains("_norm.weight")
                let value: Float = isCenteredNorm ? 0.0 : 1.0
                for i in 0..<(Int(entry.sizeBytes) / u16) {
                    dst[i] = Quantization.bf16Bits(value)
                }
            }
        }
        let weightsURL = dir.appendingPathComponent("model_weights.bin")
        try Data(fileBuf).write(to: weightsURL)
        let weightsSha = try Sha256Verifier.hashFile(at: weightsURL)

        // 3. Packed experts, grouped at 32 like the rest of the 4-bit weights.
        func appendU16(_ values: [UInt16], to bytes: inout [UInt8]) {
            for value in values {
                bytes.append(UInt8(truncatingIfNeeded: value))
                bytes.append(UInt8(truncatingIfNeeded: value >> 8))
            }
        }

        func toyExpertRows(rows: Int, cols: Int, expert: Int, role: Int) -> [[Float]] {
            (0..<rows).map { row in
                (0..<cols).map { col in
                    Float(expert + 1) * 0.001 + Float(role + 1) * 0.003
                        + Float((row % 7) - 3) * 0.0004
                        + Float((col % 11) - 5) * 0.0002
                }
            }
        }

        func toyExpertBlob(expert: Int) -> (bytes: [UInt8],
                                            tensors: [String: [String: Any]]) {
            var bytes: [UInt8] = []
            var tensors: [String: [String: Any]] = [:]

            func addProjection(prefix: String, rows: Int, cols: Int, role: Int) {
                let projectionRows = toyExpertRows(rows: rows, cols: cols,
                                                   expert: expert, role: role)
                let quantized = projectionRows.map {
                    Quantization.quantizeInt4Affine($0, groupSize: int4Groups)
                }
                let packedOffset = bytes.count
                for row in quantized { bytes.append(contentsOf: row.packed) }
                tensors[prefix] = [
                    "offset": packedOffset, "size": bytes.count - packedOffset,
                    "dtype": "U32", "shape": [rows, cols], "bits": 4,
                ]
                let scalesOffset = bytes.count
                for row in quantized { appendU16(row.scales, to: &bytes) }
                tensors["\(prefix)_scales"] = [
                    "offset": scalesOffset, "size": bytes.count - scalesOffset,
                    "dtype": "BF16", "shape": [rows, cols / int4Groups],
                ]
                let biasesOffset = bytes.count
                for row in quantized { appendU16(row.biases, to: &bytes) }
                tensors["\(prefix)_biases"] = [
                    "offset": biasesOffset, "size": bytes.count - biasesOffset,
                    "dtype": "BF16", "shape": [rows, cols / int4Groups],
                ]
            }

            addProjection(prefix: "gate", rows: toy.moeIntermediateSize,
                          cols: d, role: 0)
            addProjection(prefix: "up", rows: toy.moeIntermediateSize,
                          cols: d, role: 1)
            addProjection(prefix: "down", rows: d,
                          cols: toy.moeIntermediateSize, role: 2)
            return (bytes, tensors)
        }

        // A 128-wide hidden size doubles every projection against the
        // Qwen3.6 toy, so one expert no longer fits a single 16 KiB page.
        let expertStride: UInt64 = 32768
        let layerBytes = Int(expertStride) * toy.numExperts
        for L in 0..<toy.numLayers {
            var payload = Data(count: layerBytes)
            for E in 0..<toy.numExperts {
                let blob = toyExpertBlob(expert: E).bytes
                precondition(blob.count <= Int(expertStride),
                             "toy expert blob exceeds stride")
                let baseB = E * Int(expertStride)
                for (i, byte) in blob.enumerated() { payload[baseB + i] = byte }
            }
            try payload.write(to: exp.appendingPathComponent(
                String(format: "layer_%02d.bin", L)))
        }
        var layerShaByName: [String: String] = [:]
        for L in 0..<toy.numLayers {
            let basename = String(format: "layer_%02d.bin", L)
            layerShaByName["packed_experts/\(basename)"] =
                try Sha256Verifier.hashFile(at: exp.appendingPathComponent(basename))
        }

        // 3b. The n-gram table: shard after shard, three contiguous regions
        //     each, exactly as `RepackPlanner.planNgramTable` lays one out.
        var ngramShards: [[String: Any]] = []
        if ngram.isEnabled {
            let rowWidth = ngram.headDim
            let packedBytes = rowWidth / 2
            let groups = rowWidth / int4Groups
            let rowsPerShard = Int(Ngram.rowCount) / Ngram.shardCount
            var bytes = [UInt8]()
            for shard in 0..<Ngram.shardCount {
                let weightOffset = bytes.count
                for local in 0..<rowsPerShard {
                    let row = shard * rowsPerShard + local
                    for byteIndex in 0..<packedBytes {
                        let low = Ngram.nibble(row: row, index: byteIndex * 2)
                        let high = Ngram.nibble(row: row, index: byteIndex * 2 + 1)
                        bytes.append(low | (high << 4))
                    }
                }
                let scaleOffset = bytes.count
                for local in 0..<rowsPerShard {
                    let row = shard * rowsPerShard + local
                    let bits = Quantization.bf16Bits(Ngram.scale(row: row))
                    for _ in 0..<groups { appendU16([bits], to: &bytes) }
                }
                let biasOffset = bytes.count
                for local in 0..<rowsPerShard {
                    let row = shard * rowsPerShard + local
                    let bits = Quantization.bf16Bits(Ngram.bias(row: row))
                    for _ in 0..<groups { appendU16([bits], to: &bytes) }
                }
                ngramShards.append([
                    "rowStart": shard * rowsPerShard,
                    "rowCount": rowsPerShard,
                    "weightOffset": weightOffset,
                    "scaleOffset": scaleOffset,
                    "biasOffset": biasOffset,
                ])
            }
            try Data(bytes).write(to: dir.appendingPathComponent("ngram_ple.bin"))
        }

        // 4. layout.json
        var layersArr: [[String: Any]] = []
        for L in 0..<toy.numLayers {
            var experts: [[String: Any]] = []
            for E in 0..<toy.numExperts {
                experts.append([
                    "expert": E,
                    "offset": UInt64(E) * expertStride,
                    "size": expertStride,
                    "tensors": toyExpertBlob(expert: E).tensors,
                ])
            }
            layersArr.append([
                "layer": L,
                "file": String(format: "layer_%02d.bin", L),
                "experts": experts,
            ])
        }
        let layoutData = try JSONSerialization.data(withJSONObject: [
            "expertStride": expertStride,
            "numLayers": toy.numLayers,
            "expertsPerLayer": toy.numExperts,
            "layers": layersArr,
        ] as [String: Any], options: [.sortedKeys])
        let layoutURL = exp.appendingPathComponent("layout.json")
        try layoutData.write(to: layoutURL)
        let layoutSha = try Sha256Verifier.hashFile(at: layoutURL)

        // 5. manifest.json
        var files: [String: [String: Any]] = [
            "model_weights.bin": ["size": totalBytes, "sha256": weightsSha],
            "packed_experts/layout.json": ["size": layoutData.count,
                                           "sha256": layoutSha],
        ]
        for (rel, sha) in layerShaByName {
            files[rel] = ["size": layerBytes, "sha256": sha]
        }

        let archDict: [String: Any] = [
            "hiddenSize": toy.hiddenSize, "ffnIntermediate": toy.intermediateSize,
            "moeIntermediateSize": toy.moeIntermediateSize,
            "numHeads": toy.numHeads, "numKVHeads": toy.numKVHeads,
            "numFullKVHeads": toy.numFullKVHeads,
            "headDim": toy.headDim, "fullHeadDim": toy.fullHeadDim,
            "vocabSize": toy.vocabSize, "slidingWindow": toy.slidingWindow,
            "finalLogitSoftcap": toy.finalLogitSoftcap,
            "ropeTheta": toy.ropeTheta, "fullRopeTheta": toy.fullRopeTheta,
            "partialRotaryFactor": toy.partialRotaryFactor,
            "numLayers": toy.numLayers, "numExperts": toy.numExperts,
            "topKExperts": toy.topKExperts,
            "tieWordEmbeddings": toy.tieWordEmbeddings,
            "attentionKEqV": toy.attentionKEqV,
            "hiddenActivation": toy.hiddenActivation,
            "fullAttentionLayerMask": toy.fullAttentionLayerMask.map { Int($0) },
            "family": toy.family.rawValue,
            "variant": toy.variant.rawValue,
            "attnOutputGate": toy.attnOutputGate,
            "attentionScale": toy.attentionScale,
            "embeddingScaledBySqrtHidden": toy.embeddingScaledBySqrtHidden,
            "routerScaled": toy.routerScaled,
            "ffnSandwichNorms": toy.ffnSandwichNorms,
            "sharedExpertGated": toy.sharedExpertGated,
            "ropeNeoxSubdim": toy.ropeNeoxSubdim,
            "linearNumKHeads": la.numKHeads,
            "linearNumVHeads": la.numVHeads,
            "linearKeyHeadDim": la.keyHeadDim,
            "linearValueHeadDim": la.valueHeadDim,
            "linearConvKernelSize": la.convKernelSize,
            "attentionIndexer": [
                "budget": toy.attentionIndexer.budget,
                "compressRatio": toy.attentionIndexer.compressRatio,
                "headDim": toy.attentionIndexer.headDim,
                "numHeads": toy.attentionIndexer.numHeads,
                "numKVHeads": toy.attentionIndexer.numKVHeads,
            ],
            "hyperConnection": [
                "streamCount": hc.streamCount,
                "lowRank": hc.lowRank,
            ],
        ].merging(ngram.isEnabled ? [
            "ngramEmbedding": [
                "layer": ngram.layer,
                "ngramSize": ngram.ngramSize,
                "heads": ngram.heads,
                "headsPerNgram": ngram.headsPerNgram,
                "vocabSizeBase": ngram.vocabSizeBase,
                "shardCount": ngram.shardCount,
                "embedDim": ngram.embedDim,
                "convKernelSize": ngram.convKernelSize,
                "eosTokenID": Int(ngram.eosTokenID),
            ] as [String: Any],
        ] : [:]) { current, _ in current }
        func quantSlot(_ weightBits: Int, _ groupSize: Int) -> [String: Any] {
            [
                "weightBits": weightBits, "scheme": "affine",
                "scaleType": "BF16", "biasType": "BF16",
                "groupSize": groupSize,
            ]
        }
        let manifestRoot: [String: Any] = [
            "magic": "GTURBO",
            "versionMajor": 1,
            "versionMinor": 0,
            "flags": ["streamingPresent": true, "turboQuantKV": false,
                      "aneSharedExpert": false],
            "modelID": "qwen4exp-toy",
            "arch": archDict,
            // 4-bit at 32, 8-bit at 64 — the split the checkpoint has.
            "quant": [
                "embedding": quantSlot(4, int4Groups),
                "attention": quantSlot(4, int4Groups),
                "router": quantSlot(8, int8Groups),
                "sharedExpert": quantSlot(8, int8Groups),
                "routedExpert": quantSlot(4, int4Groups),
            ],
            "files": files,
            "expertsPerLayer": toy.numExperts,
            "numLayers": toy.numLayers,
            "expertStride": expertStride,
        ].merging(ngram.isEnabled ? [
            "ngramTable": [
                "file": "ngram_ple.bin",
                "layerIndex": ngram.layer,
                "rowWidth": ngram.headDim,
                "groupSize": int4Groups,
                "rowCount": Int(Ngram.rowCount),
                "shards": ngramShards,
            ] as [String: Any],
        ] : [:]) { current, _ in current }
        try JSONSerialization.data(withJSONObject: manifestRoot,
                                   options: [.sortedKeys, .withoutEscapingSlashes])
            .write(to: dir.appendingPathComponent("manifest.json"))
        return dir
    }
}

extension ArchConfig {
    /// Tiny Qwen4-Exp baseline: 4 layers alternating gated-DeltaNet and full
    /// attention, four residual streams, 16 experts with 10 routed, 4-bit
    /// weights grouped at 32.
    ///
    /// Every width here has to clear a divisibility constraint that the real
    /// checkpoint also clears: the stacked residual (4 x 128 = 512) and the
    /// low rank (64) are both multiples of 32, so the mix projections in both
    /// directions are decodable.
    /// `ngramLayer` opts the n-gram per-layer embedding in on that layer;
    /// nil leaves it out, which keeps the fixture usable for tests about the
    /// rest of the graph.
    static func qwen4ExpToy(
        layerMask: [UInt8] = [2, 1, 2, 1],
        int4GroupSize: Int = 32,
        ngramLayer: Int? = nil,
        indexer: AttentionIndexerConfig = .none
    ) -> ArchConfig {
        ArchConfig(
            hiddenSize: 128,
            intermediateSize: 128,
            moeIntermediateSize: 64,
            numHeads: 4,
            numKVHeads: 2,
            numFullKVHeads: 2,
            headDim: 32,
            fullHeadDim: 32,
            vocabSize: 1024,
            slidingWindow: 0,
            finalLogitSoftcap: 0.0,
            ropeTheta: 10_000_000.0,
            fullRopeTheta: 10_000_000.0,
            partialRotaryFactor: 0.25,
            numLayers: layerMask.count,
            numExperts: 16,
            topKExperts: 10,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: layerMask,
            hiddenActivation: "silu",
            family: .qwen4Exp,
            variant: .qwen38FlashNext,
            attnOutputGate: true,
            attentionScale: 0.125,
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: true,
            ropeNeoxSubdim: true,
            linearAttention: LinearAttentionConfig(
                numKHeads: 2, numVHeads: 4,
                keyHeadDim: 32, valueHeadDim: 32,
                convKernelSize: 4),
            hyperConnection: HyperConnectionConfig(streamCount: 4, lowRank: 64),
            ngramEmbedding: ngramLayer.map { layer in
                // Four heads over a 128-wide embedding: 32 values a row, which
                // is exactly one quantization group.
                NgramEmbeddingConfig(
                    layer: layer,
                    ngramSize: 3,
                    heads: 4,
                    headsPerNgram: 2,
                    vocabSizeBase: 1_000,
                    shardCount: Qwen4ExpToySynthetic.Ngram.shardCount,
                    embedDim: 128,
                    convKernelSize: 4,
                    eosTokenID: Qwen4ExpToySynthetic.Ngram.eosTokenID)
            } ?? .none,
            attentionIndexer: indexer,
            int4GroupSize: int4GroupSize)
    }
}
