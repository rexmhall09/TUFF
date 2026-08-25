import Foundation
@testable import TurboFieldfare
import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore

/// Tiny runnable dense Gemma 4 `.gturbo` fixture. It intentionally has no
/// `packed_experts` directory, includes PLE, and makes the final two layers
/// reuse layer-0/layer-1 KV by attention kind.
enum DenseGemmaToySynthetic {
    static func write() throws -> URL {
        let toy = ArchConfig.gemma4E4BToy()
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-e4b-toy-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)

        struct ResidentSpec {
            let name: String
            let dtype: UInt8
            let shape: [UInt32]
            let weightBytes: UInt64
            let scaleBytes: UInt64
            let biasBytes: UInt64
        }
        let d = toy.hiddenSize
        let p = toy.hiddenSizePerLayerInput
        let packedPLE = toy.numLayers * p
        let u16 = MemoryLayout<UInt16>.stride

        func affine(_ name: String, rows: Int, columns: Int) -> ResidentSpec {
            let groups = columns / Quantization.groupSize
            let auxiliaries = UInt64(rows * groups * u16)
            return ResidentSpec(name: name, dtype: 0,
                                shape: [UInt32(rows), UInt32(columns), 0, 0],
                                weightBytes: UInt64(rows * columns / 2),
                                scaleBytes: auxiliaries,
                                biasBytes: auxiliaries)
        }
        func bf16(_ name: String, count: Int) -> ResidentSpec {
            ResidentSpec(name: name, dtype: 1,
                         shape: [UInt32(count), 0, 0, 0],
                         weightBytes: UInt64(count * u16),
                         scaleBytes: 0, biasBytes: 0)
        }

        var specs: [ResidentSpec] = [
            affine("language_model.model.embed_tokens.weight",
                   rows: toy.vocabSize, columns: d),
            affine("language_model.model.embed_tokens_per_layer.weight",
                   rows: toy.vocabSizePerLayerInput, columns: packedPLE),
            affine("language_model.model.per_layer_model_projection.weight",
                   rows: packedPLE, columns: d),
            bf16("language_model.model.per_layer_projection_norm.weight", count: p),
            bf16("language_model.model.norm.weight", count: d),
        ]
        for layer in 0..<toy.numLayers {
            let prefix = "language_model.model.layers.\(layer)"
            let headDimension = toy.layerIsFull(layer) ? toy.fullHeadDim : toy.headDim
            let kvHeads = toy.layerIsFull(layer) ? toy.numFullKVHeads : toy.numKVHeads
            let qDimension = toy.numHeads * headDimension
            let kvDimension = kvHeads * headDimension
            specs.append(contentsOf: [
                affine("\(prefix).self_attn.q_proj.weight",
                       rows: qDimension, columns: d),
                affine("\(prefix).self_attn.o_proj.weight",
                       rows: d, columns: qDimension),
                bf16("\(prefix).self_attn.q_norm.weight", count: headDimension),
                affine("\(prefix).mlp.gate_proj.weight",
                       rows: toy.intermediateSize, columns: d),
                affine("\(prefix).mlp.up_proj.weight",
                       rows: toy.intermediateSize, columns: d),
                affine("\(prefix).mlp.down_proj.weight",
                       rows: d, columns: toy.intermediateSize),
                affine("\(prefix).per_layer_input_gate.weight", rows: p, columns: d),
                affine("\(prefix).per_layer_projection.weight", rows: d, columns: p),
                bf16("\(prefix).input_layernorm.weight", count: d),
                bf16("\(prefix).post_attention_layernorm.weight", count: d),
                bf16("\(prefix).pre_feedforward_layernorm.weight", count: d),
                bf16("\(prefix).post_feedforward_layernorm.weight", count: d),
                bf16("\(prefix).post_per_layer_input_norm.weight", count: d),
                bf16("\(prefix).layer_scalar", count: 1),
            ])
            if !toy.layerSharesKV(layer) {
                specs.append(affine("\(prefix).self_attn.k_proj.weight",
                                    rows: kvDimension, columns: d))
                specs.append(affine("\(prefix).self_attn.v_proj.weight",
                                    rows: kvDimension, columns: d))
                specs.append(bf16("\(prefix).self_attn.k_norm.weight",
                                  count: headDimension))
            }
        }

        let names = specs.map(\.name)
        let stringTable = names.joined().data(using: .utf8)!
        let headerBytes = GTurboBinary.indexHeaderBytes
        let entryBytes = GTurboBinary.indexEntryBytes
        let entriesBase = headerBytes
        let stringBase = entriesBase + names.count * entryBytes
        var nameOffsets: [UInt32] = []
        var stringCursor = 0
        for name in names {
            nameOffsets.append(UInt32(stringBase + stringCursor))
            stringCursor += name.utf8.count
        }
        let alignment = GTurboFormatV1.alignmentBytes
        let rawIndexBytes = UInt64(stringBase + stringTable.count)
        let indexBytes = ((rawIndexBytes + alignment - 1) / alignment) * alignment
        var entries: [ResidentEntry] = []
        var payloadCursor = indexBytes
        for spec in specs {
            let weightOffset = payloadCursor
            let scaleOffset = spec.scaleBytes == 0 ? 0 : weightOffset + spec.weightBytes
            let biasOffset = spec.biasBytes == 0 ? 0 : scaleOffset + spec.scaleBytes
            entries.append(ResidentEntry(
                name: spec.name,
                dtype: spec.dtype,
                logicalShape4: spec.shape,
                fileOffset: weightOffset,
                sizeBytes: spec.weightBytes,
                scaleOffset: scaleOffset,
                scaleSize: spec.scaleBytes,
                biasOffset: biasOffset,
                biasSize: spec.biasBytes,
                quantSpec: nil,
                sourceWeight: ModelLoaderTests.dummySource(spec.name),
                sourceScales: nil,
                sourceBiases: nil))
            payloadCursor += spec.weightBytes + spec.scaleBytes + spec.biasBytes
        }
        let residentSize = payloadCursor - indexBytes
        var file = [UInt8](repeating: 0, count: Int(payloadCursor))
        file.withUnsafeMutableBytes { raw in
            let base = raw.baseAddress!
            GTurboBinary.writeIndexHeader(into: base,
                                          indexSize: indexBytes,
                                          residentSize: residentSize,
                                          entryCount: UInt64(entries.count))
            for (index, entry) in entries.enumerated() {
                GTurboBinary.writeIndexEntry(
                    into: base.advanced(by: entriesBase + index * entryBytes),
                    entry: entry,
                    nameOffset: nameOffsets[index])
            }
            _ = stringTable.withUnsafeBytes {
                memcpy(base.advanced(by: stringBase), $0.baseAddress!, stringTable.count)
            }
            for (index, entry) in entries.enumerated() where entry.dtype == 0 {
                // Different small affine codes per tensor avoid making every
                // branch algebraically identical while staying deterministic.
                memset(base.advanced(by: Int(entry.fileOffset)),
                       Int32(0x11 + (index % 5)), Int(entry.sizeBytes))
                let scales = base.advanced(by: Int(entry.scaleOffset))
                    .assumingMemoryBound(to: UInt16.self)
                let biases = base.advanced(by: Int(entry.biasOffset))
                    .assumingMemoryBound(to: UInt16.self)
                for i in 0..<(Int(entry.scaleSize) / u16) {
                    scales[i] = Quantization.bf16Bits(0.01 + Float(index % 3) * 0.002)
                    biases[i] = Quantization.bf16Bits(-0.02 + Float(index % 4) * 0.005)
                }
            }
            for entry in entries where entry.dtype == 1 {
                let values = base.advanced(by: Int(entry.fileOffset))
                    .assumingMemoryBound(to: UInt16.self)
                for i in 0..<(Int(entry.sizeBytes) / u16) {
                    values[i] = Quantization.bf16Bits(1.0)
                }
            }
        }
        let weightsURL = directory.appendingPathComponent("model_weights.bin")
        try Data(file).write(to: weightsURL)
        let weightsSHA = try Sha256Verifier.hashFile(at: weightsURL)

        func quantSlot(_ bits: Int) -> [String: Any] {
            ["weightBits": bits, "scheme": "affine", "scaleType": "BF16",
             "biasType": "BF16", "groupSize": Quantization.groupSize]
        }
        let architecture: [String: Any] = [
            "hiddenSize": toy.hiddenSize,
            "ffnIntermediate": toy.intermediateSize,
            "moeIntermediateSize": 0,
            "numHeads": toy.numHeads,
            "numKVHeads": toy.numKVHeads,
            "numFullKVHeads": toy.numFullKVHeads,
            "headDim": toy.headDim,
            "fullHeadDim": toy.fullHeadDim,
            "vocabSize": toy.vocabSize,
            "slidingWindow": toy.slidingWindow,
            "finalLogitSoftcap": toy.finalLogitSoftcap,
            "ropeTheta": toy.ropeTheta,
            "fullRopeTheta": toy.fullRopeTheta,
            "partialRotaryFactor": toy.partialRotaryFactor,
            "numLayers": toy.numLayers,
            "numExperts": 0,
            "topKExperts": 0,
            "tieWordEmbeddings": true,
            "attentionKEqV": false,
            "hiddenActivation": toy.hiddenActivation,
            "fullAttentionLayerMask": toy.fullAttentionLayerMask.map(Int.init),
            "family": toy.family.rawValue,
            "variant": toy.variant.rawValue,
            "feedForwardKind": toy.feedForwardKind.rawValue,
            "hiddenSizePerLayerInput": p,
            "vocabSizePerLayerInput": toy.vocabSizePerLayerInput,
            "numKVSharedLayers": toy.numKVSharedLayers,
            "attnOutputGate": false,
            "attentionScale": toy.attentionScale,
            "embeddingScaledBySqrtHidden": true,
            "routerScaled": true,
            "ffnSandwichNorms": true,
            "sharedExpertGated": false,
            "ropeNeoxSubdim": false,
            "linearNumKHeads": 0,
            "linearNumVHeads": 0,
            "linearKeyHeadDim": 0,
            "linearValueHeadDim": 0,
            "linearConvKernelSize": 0,
        ]
        let manifest: [String: Any] = [
            "magic": "GTURBO",
            "versionMajor": 1,
            "versionMinor": 1,
            "flags": [
                "streamingPresent": false,
                "turboQuantKV": false,
                "aneSharedExpert": false,
                "denseFFN": true,
            ],
            "modelID": "gemma4-e4b-toy",
            "arch": architecture,
            "quant": [
                "embedding": quantSlot(4),
                "attention": quantSlot(4),
                "router": quantSlot(8),
                "sharedExpert": quantSlot(4),
                "routedExpert": quantSlot(4),
            ],
            "files": [
                "model_weights.bin": ["size": file.count, "sha256": weightsSHA],
            ],
            "expertsPerLayer": 0,
            "numLayers": toy.numLayers,
            "expertStride": 0,
        ]
        let manifestData = try JSONSerialization.data(
            withJSONObject: manifest, options: [.sortedKeys, .withoutEscapingSlashes])
        try manifestData.write(to: directory.appendingPathComponent("manifest.json"))
        return directory
    }
}

extension ArchConfig {
    static func gemma4E4BToy() -> ArchConfig {
        ArchConfig(
            hiddenSize: 64,
            intermediateSize: 128,
            moeIntermediateSize: 0,
            numHeads: 2,
            numKVHeads: 1,
            numFullKVHeads: 1,
            headDim: 32,
            fullHeadDim: 64,
            vocabSize: 256,
            slidingWindow: 32,
            finalLogitSoftcap: 30,
            ropeTheta: 10_000,
            fullRopeTheta: 1_000_000,
            partialRotaryFactor: 0.25,
            numLayers: 4,
            numExperts: 0,
            topKExperts: 0,
            tieWordEmbeddings: true,
            attentionKEqV: false,
            fullAttentionLayerMask: [0, 1, 0, 1],
            hiddenActivation: "gelu_pytorch_tanh",
            hiddenSizePerLayerInput: 64,
            vocabSizePerLayerInput: 256,
            numKVSharedLayers: 2,
            variant: .gemma4_E4B,
            feedForwardKind: .dense)
    }
}
