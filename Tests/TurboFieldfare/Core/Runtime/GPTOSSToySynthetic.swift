import Foundation
@testable import TurboFieldfare
import TurboFieldfareFormat
@testable import TurboFieldfareRepackCore

/// Tiny executable GPT-OSS `.gturbo` fixture with one sliding-window layer,
/// one full-attention layer, BF16 resident projections, and four MXFP4 routed
/// experts. The shapes are small but retain every production layout rule.
enum GPTOSSToySynthetic {
    static func write() throws -> URL {
        let config = ArchConfig.gptOssToy()
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("gturbo-gptoss-toy-\(UUID().uuidString)")
        let expertsDirectory = directory.appendingPathComponent("packed_experts")
        try FileManager.default.createDirectory(
            at: expertsDirectory, withIntermediateDirectories: true)

        struct ResidentSpec {
            let name: String
            let shape: [UInt32]
            let count: Int
        }
        let hidden = config.hiddenSize
        let query = config.numHeads * config.headDim
        let kv = config.numKVHeads * config.headDim
        func matrix(_ name: String, rows: Int, columns: Int) -> ResidentSpec {
            ResidentSpec(name: name,
                         shape: [UInt32(rows), UInt32(columns), 0, 0],
                         count: rows * columns)
        }
        func vector(_ name: String, count: Int) -> ResidentSpec {
            ResidentSpec(name: name,
                         shape: [UInt32(count), 0, 0, 0],
                         count: count)
        }

        var specs: [ResidentSpec] = [
            matrix("language_model.model.embed_tokens.weight",
                   rows: config.vocabSize, columns: hidden),
            matrix("language_model.lm_head.weight",
                   rows: config.vocabSize, columns: hidden),
            vector("language_model.model.norm.weight", count: hidden),
        ]
        for layer in 0..<config.numLayers {
            let prefix = "language_model.model.layers.\(layer)"
            specs.append(contentsOf: [
                vector("\(prefix).input_layernorm.weight", count: hidden),
                vector("\(prefix).post_attention_layernorm.weight", count: hidden),
                matrix("\(prefix).self_attn.q_proj.weight",
                       rows: query, columns: hidden),
                vector("\(prefix).self_attn.q_proj.bias", count: query),
                matrix("\(prefix).self_attn.k_proj.weight",
                       rows: kv, columns: hidden),
                vector("\(prefix).self_attn.k_proj.bias", count: kv),
                matrix("\(prefix).self_attn.v_proj.weight",
                       rows: kv, columns: hidden),
                vector("\(prefix).self_attn.v_proj.bias", count: kv),
                matrix("\(prefix).self_attn.o_proj.weight",
                       rows: hidden, columns: query),
                vector("\(prefix).self_attn.o_proj.bias", count: hidden),
                vector("\(prefix).self_attn.sinks", count: config.numHeads),
                matrix("\(prefix).mlp.router.weight",
                       rows: config.numExperts, columns: hidden),
                vector("\(prefix).mlp.router.bias", count: config.numExperts),
            ])
        }

        let names = specs.map(\.name)
        let stringTable = names.joined().data(using: .utf8)!
        let entriesBase = GTurboBinary.indexHeaderBytes
        let stringBase = entriesBase + names.count * GTurboBinary.indexEntryBytes
        var nameOffsets: [UInt32] = []
        var nameCursor = 0
        for name in names {
            nameOffsets.append(UInt32(stringBase + nameCursor))
            nameCursor += name.utf8.count
        }
        let rawIndexSize = UInt64(stringBase + stringTable.count)
        let alignment = GTurboFormatV1.alignmentBytes
        let indexSize = ((rawIndexSize + alignment - 1) / alignment) * alignment
        var payloadCursor = indexSize
        var entries: [ResidentEntry] = []
        for spec in specs {
            let byteCount = UInt64(spec.count * MemoryLayout<UInt16>.stride)
            entries.append(ResidentEntry(
                name: spec.name,
                dtype: GTurboFormatV1.DType.bf16.rawValue,
                logicalShape4: spec.shape,
                fileOffset: payloadCursor,
                sizeBytes: byteCount,
                scaleOffset: 0,
                scaleSize: 0,
                biasOffset: 0,
                biasSize: 0,
                quantSpec: nil,
                sourceWeight: ModelLoaderTests.dummySource(spec.name),
                sourceScales: nil,
                sourceBiases: nil))
            payloadCursor += byteCount
        }
        var residentBytes = [UInt8](repeating: 0, count: Int(payloadCursor))
        residentBytes.withUnsafeMutableBytes { raw in
            let base = raw.baseAddress!
            GTurboBinary.writeIndexHeader(
                into: base,
                indexSize: indexSize,
                residentSize: payloadCursor - indexSize,
                entryCount: UInt64(entries.count))
            for (index, entry) in entries.enumerated() {
                GTurboBinary.writeIndexEntry(
                    into: base.advanced(
                        by: entriesBase + index * GTurboBinary.indexEntryBytes),
                    entry: entry,
                    nameOffset: nameOffsets[index])
            }
            _ = stringTable.withUnsafeBytes {
                memcpy(base.advanced(by: stringBase),
                       $0.baseAddress!, stringTable.count)
            }
            for (specIndex, entry) in entries.enumerated() {
                let spec = specs[specIndex]
                let output = base.advanced(by: Int(entry.fileOffset))
                    .assumingMemoryBound(to: UInt16.self)
                for element in 0..<spec.count {
                    let value: Float
                    if spec.name.contains("layernorm")
                        || spec.name == "language_model.model.norm.weight" {
                        value = 1
                    } else if spec.name.hasSuffix(".sinks") {
                        value = -0.75 + Float(element) * 0.125
                    } else if spec.name.hasSuffix(".bias") {
                        value = Float((element + specIndex * 3) % 11 - 5) / 256
                    } else {
                        value = Float((element * 17 + specIndex * 13) % 31 - 15)
                            / 512
                    }
                    output[element] = Quantization.bf16Bits(value)
                }
            }
        }
        let weightsURL = directory.appendingPathComponent("model_weights.bin")
        try Data(residentBytes).write(to: weightsURL)
        let weightsSHA = try Sha256Verifier.hashFile(at: weightsURL)

        let expertLayout = makeExpertLayout(config: config)
        let expertStride: UInt64 = 16_384
        let layerBytes = Int(expertStride) * config.numExperts
        var layerFiles: [String: [String: Any]] = [:]
        for layer in 0..<config.numLayers {
            var payload = Data(count: layerBytes)
            for expert in 0..<config.numExperts {
                let base = expert * Int(expertStride)
                let bytes = makeExpertBlob(
                    config: config, expert: expert, layer: layer,
                    offsets: expertLayout.offsets)
                payload.replaceSubrange(base..<(base + bytes.count), with: bytes)
            }
            let basename = String(format: "layer_%02d.bin", layer)
            let url = expertsDirectory.appendingPathComponent(basename)
            try payload.write(to: url)
            layerFiles["packed_experts/\(basename)"] = [
                "size": payload.count,
                "sha256": try Sha256Verifier.hashFile(at: url),
            ]
        }

        var layoutLayers: [[String: Any]] = []
        for layer in 0..<config.numLayers {
            let experts: [[String: Any]] = (0..<config.numExperts).map { expert in
                [
                    "expert": expert,
                    "offset": UInt64(expert) * expertStride,
                    "size": expertStride,
                    "tensors": expertLayout.tensors,
                ]
            }
            layoutLayers.append([
                "layer": layer,
                "file": String(format: "layer_%02d.bin", layer),
                "experts": experts,
            ])
        }
        let layoutRoot: [String: Any] = [
            "expertStride": expertStride,
            "numLayers": config.numLayers,
            "expertsPerLayer": config.numExperts,
            "layers": layoutLayers,
        ]
        let layoutData = try JSONSerialization.data(
            withJSONObject: layoutRoot, options: [.sortedKeys])
        let layoutURL = expertsDirectory.appendingPathComponent("layout.json")
        try layoutData.write(to: layoutURL)

        var files: [String: [String: Any]] = [
            "model_weights.bin": [
                "size": residentBytes.count,
                "sha256": weightsSHA,
            ],
            "packed_experts/layout.json": [
                "size": layoutData.count,
                "sha256": try Sha256Verifier.hashFile(at: layoutURL),
            ],
        ]
        for (name, entry) in layerFiles { files[name] = entry }

        let architecture: [String: Any] = [
            "hiddenSize": config.hiddenSize,
            "ffnIntermediate": config.intermediateSize,
            "moeIntermediateSize": config.moeIntermediateSize,
            "numHeads": config.numHeads,
            "numKVHeads": config.numKVHeads,
            "numFullKVHeads": config.numFullKVHeads,
            "headDim": config.headDim,
            "fullHeadDim": config.fullHeadDim,
            "vocabSize": config.vocabSize,
            "slidingWindow": config.slidingWindow,
            "finalLogitSoftcap": config.finalLogitSoftcap,
            "ropeTheta": config.ropeTheta,
            "fullRopeTheta": config.fullRopeTheta,
            "partialRotaryFactor": config.partialRotaryFactor,
            "numLayers": config.numLayers,
            "numExperts": config.numExperts,
            "topKExperts": config.topKExperts,
            "tieWordEmbeddings": config.tieWordEmbeddings,
            "attentionKEqV": config.attentionKEqV,
            "hiddenActivation": config.hiddenActivation,
            "fullAttentionLayerMask": config.fullAttentionLayerMask.map(Int.init),
            "family": config.family.rawValue,
            "variant": config.variant.rawValue,
            "feedForwardKind": config.feedForwardKind.rawValue,
            "attnOutputGate": config.attnOutputGate,
            "attentionScale": config.attentionScale,
            "embeddingScaledBySqrtHidden": config.embeddingScaledBySqrtHidden,
            "routerScaled": config.routerScaled,
            "ffnSandwichNorms": config.ffnSandwichNorms,
            "sharedExpertGated": config.sharedExpertGated,
            "ropeNeoxSubdim": config.ropeNeoxSubdim,
            "linearNumKHeads": 0,
            "linearNumVHeads": 0,
            "linearKeyHeadDim": 0,
            "linearValueHeadDim": 0,
            "linearConvKernelSize": 0,
        ]
        let residentQuant: [String: Any] = [
            "weightBits": 16,
            "scheme": "BF16",
            "scaleType": "none",
            "biasType": "none",
            "groupSize": 1,
        ]
        let routedQuant: [String: Any] = [
            "weightBits": 4,
            "scheme": "MXFP4",
            "scaleType": "UE8M0",
            "biasType": "none",
            "groupSize": Quantization.mxfp4GroupSize,
        ]
        let manifest: [String: Any] = [
            "magic": "GTURBO",
            "versionMajor": 1,
            "versionMinor": 1,
            "flags": [
                "streamingPresent": true,
                "turboQuantKV": false,
                "aneSharedExpert": false,
                "mxfp4Weights": true,
            ],
            "modelID": "gpt-oss-toy",
            "arch": architecture,
            "quant": [
                "embedding": residentQuant,
                "attention": residentQuant,
                "router": residentQuant,
                "sharedExpert": residentQuant,
                "routedExpert": routedQuant,
            ],
            "files": files,
            "expertsPerLayer": config.numExperts,
            "numLayers": config.numLayers,
            "expertStride": expertStride,
        ]
        let manifestData = try JSONSerialization.data(
            withJSONObject: manifest,
            options: [.sortedKeys, .withoutEscapingSlashes])
        try manifestData.write(
            to: directory.appendingPathComponent("manifest.json"))
        return directory
    }

    private struct ExpertLayout {
        let offsets: GPTOSSExpertOffsets
        let tensors: [String: [String: Any]]
    }

    private static func makeExpertLayout(config: ArchConfig) -> ExpertLayout {
        let hidden = config.hiddenSize
        let intermediate = config.moeIntermediateSize
        var offset = 0
        var tensors: [String: [String: Any]] = [:]
        func add(_ name: String, size: Int, dtype: String,
                 shape: [Int], bits: Int? = nil) -> Int {
            let start = offset
            var entry: [String: Any] = [
                "offset": start,
                "size": size,
                "dtype": dtype,
                "shape": shape,
            ]
            if let bits { entry["bits"] = bits }
            tensors[name] = entry
            offset += size
            return start
        }
        let mlp1 = add("mlp1", size: 2 * intermediate * hidden / 2,
                       dtype: "U8", shape: [2 * intermediate, hidden], bits: 4)
        let mlp1Scales = add(
            "mlp1_scales", size: 2 * intermediate * hidden / 32,
            dtype: "U8", shape: [2 * intermediate, hidden / 32])
        let mlp1Bias = add("mlp1_bias", size: 2 * intermediate * 2,
                           dtype: "BF16", shape: [2 * intermediate])
        let mlp2 = add("mlp2", size: hidden * intermediate / 2,
                       dtype: "U8", shape: [hidden, intermediate], bits: 4)
        let mlp2Scales = add(
            "mlp2_scales", size: hidden * intermediate / 32,
            dtype: "U8", shape: [hidden, intermediate / 32])
        let mlp2Bias = add("mlp2_bias", size: hidden * 2,
                           dtype: "BF16", shape: [hidden])
        return ExpertLayout(
            offsets: GPTOSSExpertOffsets(
                mlp1Weights: mlp1,
                mlp1Scales: mlp1Scales,
                mlp1Bias: mlp1Bias,
                mlp2Weights: mlp2,
                mlp2Scales: mlp2Scales,
                mlp2Bias: mlp2Bias),
            tensors: tensors)
    }

    private static func makeExpertBlob(
        config: ArchConfig,
        expert: Int,
        layer: Int,
        offsets: GPTOSSExpertOffsets
    ) -> Data {
        let hidden = config.hiddenSize
        let intermediate = config.moeIntermediateSize
        let finalSize = offsets.mlp2Bias + hidden * 2
        var bytes = Data(count: finalSize)
        bytes.withUnsafeMutableBytes { raw in
            let base = raw.baseAddress!
            let mlp1Count = 2 * intermediate * hidden / 2
            let mlp2Count = hidden * intermediate / 2
            memset(base.advanced(by: offsets.mlp1Weights),
                   Int32(0x11 + ((expert + layer) % 4)), mlp1Count)
            memset(base.advanced(by: offsets.mlp2Weights),
                   Int32(0x11 + ((expert * 2 + layer) % 4)), mlp2Count)
            let scale = UInt8(120 + ((expert + layer) % 3))
            memset(base.advanced(by: offsets.mlp1Scales), Int32(scale),
                   2 * intermediate * hidden / 32)
            memset(base.advanced(by: offsets.mlp2Scales), Int32(scale),
                   hidden * intermediate / 32)
            let firstBias = base.advanced(by: offsets.mlp1Bias)
                .assumingMemoryBound(to: UInt16.self)
            for index in 0..<(2 * intermediate) {
                firstBias[index] = Quantization.bf16Bits(
                    Float((index + expert) % 7 - 3) / 512)
            }
            let secondBias = base.advanced(by: offsets.mlp2Bias)
                .assumingMemoryBound(to: UInt16.self)
            for index in 0..<hidden {
                secondBias[index] = Quantization.bf16Bits(
                    Float((index + layer) % 5 - 2) / 512)
            }
        }
        return bytes
    }
}

extension ArchConfig {
    static func gptOssToy() -> ArchConfig {
        ArchConfig(
            hiddenSize: 64,
            intermediateSize: 64,
            moeIntermediateSize: 64,
            numHeads: 2,
            numKVHeads: 1,
            numFullKVHeads: 1,
            headDim: 32,
            fullHeadDim: 32,
            vocabSize: 128,
            slidingWindow: 32,
            finalLogitSoftcap: 0,
            ropeTheta: 150_000,
            fullRopeTheta: 150_000,
            partialRotaryFactor: 1,
            numLayers: 2,
            numExperts: 4,
            topKExperts: 4,
            tieWordEmbeddings: false,
            attentionKEqV: false,
            fullAttentionLayerMask: [0, 1],
            hiddenActivation: "swiglu_capped",
            family: .gptOss,
            variant: .gptOss_20B,
            attentionScale: 1 / sqrt(32),
            embeddingScaledBySqrtHidden: false,
            routerScaled: false,
            ffnSandwichNorms: false,
            sharedExpertGated: false,
            ropeNeoxSubdim: true,
            attentionSinks: true,
            yarnRope: YaRNRopeConfig(
                originalContextLength: 64,
                scalingFactor: 2,
                betaFast: 32,
                betaSlow: 1),
            swigluLimit: 7)
    }
}
