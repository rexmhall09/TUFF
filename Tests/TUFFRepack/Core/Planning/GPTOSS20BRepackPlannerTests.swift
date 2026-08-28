import Foundation
import Testing
@testable import TUFFRepackCore

@Suite struct GPTOSS20BRepackPlannerTests {
    @Test func pinnedSourceAndSelectorsAreStable() {
        let source = SupportedModelSource.gptOss20B
        #expect(source.name == "gpt-oss-20b")
        #expect(source.aliases == ["gpt-oss"])
        #expect(source.repoID == "openai/gpt-oss-20b")
        #expect(source.revision == "6cee5e81ee83917806bbde320786a8fb61efebee")
        #expect(source.sourceIndexSHA256
            == "0e085b977c4c9942f85938828e8c989ed7d5cdabf852e4da6a67c116cd502cd1")
        #expect(SourceFingerprint.modelID(
            forIndexSha256: source.sourceIndexSHA256,
            repoID: source.repoID) == source.modelID)
    }

    @Test func officialArchitectureConfigParsesAndCrossChecks() throws {
        let root = temporaryRoot("production-config")
        defer { try? FileManager.default.removeItem(atPath: root) }
        try FileManager.default.createDirectory(
            atPath: root, withIntermediateDirectories: true)
        let path = (root as NSString).appendingPathComponent("config.json")
        try writeConfig(path: path, production: true)

        let arch = try ArchInfo.load(configPath: path)
        #expect(arch.family == .gptOss)
        #expect(arch.variant == .gptOss_20B)
        #expect(arch.hiddenSize == 2_880)
        #expect(arch.numLayers == 24)
        #expect(arch.numExperts == 32)
        #expect(arch.topKExperts == 4)
        #expect(arch.fullAttentionLayerMask
            == (0..<24).map { UInt8($0.isMultiple(of: 2) ? 0 : 1) })
        #expect(arch.hiddenActivation == "swiglu_capped")
        #expect(arch.attentionScale == 0.125)
    }

    @Test func pinned120BSourceAndArchitectureAreStable() throws {
        let source = SupportedModelSource.gptOss120B
        #expect(source.name == "gpt-oss-120b")
        #expect(source.aliases.isEmpty)
        #expect(source.repoID == "openai/gpt-oss-120b")
        #expect(source.revision == "b5c939de8f754692c1647ca79fbf85e8c1e70f8a")
        #expect(source.sourceIndexSHA256
            == "ede2655fdc05008561983b6e0829c600727c28d591e071077377059f03a6c00e")
        #expect(SourceFingerprint.modelID(
            forIndexSha256: source.sourceIndexSHA256,
            repoID: source.repoID) == source.modelID)

        let root = temporaryRoot("production-120b-config")
        defer { try? FileManager.default.removeItem(atPath: root) }
        try FileManager.default.createDirectory(
            atPath: root, withIntermediateDirectories: true)
        let path = (root as NSString).appendingPathComponent("config.json")
        try writeConfig(path: path, production: true, model120B: true)

        let arch = try ArchInfo.load(configPath: path)
        #expect(arch.family == .gptOss)
        #expect(arch.variant == .gptOss_120B)
        #expect(arch.numLayers == 36)
        #expect(arch.numExperts == 128)
        #expect(arch.topKExperts == 4)
        #expect(arch.fullAttentionLayerMask
            == (0..<36).map { UInt8($0.isMultiple(of: 2) ? 0 : 1) })
    }

    @Test func mxfp4MetadataAndU8SafetensorsAreAccepted() throws {
        let root = temporaryRoot("metadata")
        defer { try? FileManager.default.removeItem(atPath: root) }
        try FileManager.default.createDirectory(
            atPath: root, withIntermediateDirectories: true)
        try writeConfig(
            path: (root as NSString).appendingPathComponent("config.json"),
            production: false)
        let index: [String: Any] = [
            "weight_map": ["model.embed_tokens.weight": "model-00000-of-00002.safetensors"],
        ]
        try JSONSerialization.data(withJSONObject: index, options: [.sortedKeys])
            .write(to: URL(fileURLWithPath: (root as NSString)
                .appendingPathComponent("model.safetensors.index.json")))
        let metadata = try IndexLoader.load(snapshotDir: root)
        #expect(metadata.baseMode == "mxfp4")
        #expect(metadata.baseBits == 4)
        #expect(metadata.baseGroupSize == 32)
        #expect(metadata.bitsOverrides.isEmpty)

        let object: [String: Any] = [
            "tensor": [
                "dtype": "U8",
                "shape": [2, 16],
                "data_offsets": [0, 32],
            ],
        ]
        let bytes = try JSONSerialization.data(withJSONObject: object)
        let header = try Safetensors.parseHeaderBytes(
            path: "toy.safetensors",
            fileSize: UInt64(8 + bytes.count + 32),
            headerBytes: bytes)
        #expect(header.tensors[0].dtype == .u8)
        #expect(header.tensors[0].sizeBytes == 32)
    }

    @Test func plannerKeepsBF16ResidentAndStreamsMXFP4Experts() throws {
        let fixture = try makeToyPlan(tag: "plan")
        defer { fixture.cleanup() }
        let plan = fixture.plan

        #expect(plan.baseMode == "mxfp4")
        #expect(plan.resident.entries.first?.name
            == "language_model.model.embed_tokens.weight")
        #expect(plan.resident.entries.last?.name
            == "language_model.lm_head.weight")
        #expect(plan.resident.entries.allSatisfy {
            $0.dtype == 1 && $0.quantSpec == nil
        })
        #expect(plan.layers.count == 2)
        for layer in plan.layers {
            #expect(layer.expertsPerLayer == 2)
            #expect(layer.subTensors.map { "\($0.role).\($0.component)" } == [
                "mlp1.weights", "mlp1.scales", "mlp1.bias",
                "mlp2.weights", "mlp2.scales", "mlp2.bias",
            ])
            #expect(layer.expertStride % 16_384 == 0)
            #expect(layer.subTensors.filter { $0.component != "bias" }
                .allSatisfy { $0.dtype == SourceTensor.Dtype.u8.rawValue })
        }

        let layoutData = try GTurboJSON.encodeLayout(
            plan: plan, expertStride: plan.layers[0].expertStride)
        let layout = try JSONSerialization.jsonObject(with: layoutData)
            as! [String: Any]
        let layers = layout["layers"] as! [[String: Any]]
        let experts = layers[0]["experts"] as! [[String: Any]]
        let tensors = experts[0]["tensors"] as! [String: [String: Any]]
        #expect(Set(tensors.keys) == [
            "mlp1", "mlp1_scales", "mlp1_bias",
            "mlp2", "mlp2_scales", "mlp2_bias",
        ])
        #expect(tensors["mlp1"]?["dtype"] as? String == "U8")
        #expect(tensors["mlp1_scales"]?["dtype"] as? String == "U8")
        #expect(tensors["mlp1_bias"]?["dtype"] as? String == "BF16")

        let rangePlan = try RangeCopyPlanner.plan(
            repackPlan: plan, rangeChunkBytes: 512 * 1024)
        #expect(rangePlan.scalarCopies.count
            == plan.resident.entries.count + 2 * 2 * 6)
        #expect(rangePlan.remoteBytesToDownload > 0)
    }

    @Test func manifestUsesV11AndExplicitMXFP4Contract() throws {
        let fixture = try makeToyPlan(tag: "manifest")
        defer { fixture.cleanup() }
        let plan = fixture.plan
        let data = try GTurboJSON.encodeManifest(
            plan: plan,
            modelID: "openai/gpt-oss-20b",
            sourceSnapshotHash: "sha256:toy",
            files: [],
            expertsPerLayer: plan.arch.numExperts,
            numLayers: plan.arch.numLayers,
            expertStride: plan.layers[0].expertStride,
            bitWidths: .init(
                embedding: 16, attention: 16, router: 16,
                sharedExpert: 16, routedExpert: 4))
        let root = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let flags = root["flags"] as! [String: Bool]
        let arch = root["arch"] as! [String: Any]
        let quant = root["quant"] as! [String: [String: Any]]
        #expect(root["versionMinor"] as? Int == 1)
        #expect(flags["mxfp4Weights"] == true)
        #expect(flags["streamingPresent"] == true)
        #expect(arch["family"] as? String == "gpt-oss")
        #expect(arch["variant"] as? String == "gpt-oss-20b")
        #expect(arch["feedForwardKind"] as? String == "moe")
        #expect(quant["embedding"]?["scheme"] as? String == "bf16")
        #expect(quant["embedding"]?["weightBits"] as? Int == 16)
        #expect(quant["routedExpert"]?["scheme"] as? String == "mxfp4")
        #expect(quant["routedExpert"]?["scaleType"] as? String == "UE8M0")
        #expect(quant["routedExpert"]?["groupSize"] as? Int == 32)
    }
}

private struct GPTOSSToyFixture {
    let plan: RepackPlan
    let sourceRoot: String
    let outputRoot: String

    func cleanup() {
        try? FileManager.default.removeItem(atPath: sourceRoot)
        try? FileManager.default.removeItem(atPath: outputRoot)
    }
}

private func makeToyPlan(tag: String) throws -> GPTOSSToyFixture {
    let source = temporaryRoot(tag + "-source")
    let output = temporaryRoot(tag + "-output")
    try FileManager.default.createDirectory(
        atPath: source, withIntermediateDirectories: true)
    let configPath = (source as NSString).appendingPathComponent("config.json")
    try writeConfig(path: configPath, production: false)
    let arch = try ArchInfo.load(configPath: configPath)

    var tensors: [SourceTensor] = []
    var offset: UInt64 = 24_000
    func add(_ name: String, _ dtype: SourceTensor.Dtype, _ shape: [UInt64]) {
        let elements = shape.reduce(UInt64(1), *)
        let size = elements * UInt64(dtype.elementBytes)
        tensors.append(SourceTensor(
            name: name,
            shardPath: "model-00000-of-00002.safetensors",
            dtype: dtype,
            shape: shape,
            absoluteOffset: offset,
            sizeBytes: size))
        offset += size
    }

    add("model.embed_tokens.weight", .bf16, [128, 64])
    for layer in 0..<2 {
        let prefix = "model.layers.\(layer)."
        add(prefix + "input_layernorm.weight", .bf16, [64])
        add(prefix + "self_attn.q_proj.weight", .bf16, [64, 64])
        add(prefix + "self_attn.q_proj.bias", .bf16, [64])
        add(prefix + "self_attn.k_proj.weight", .bf16, [32, 64])
        add(prefix + "self_attn.k_proj.bias", .bf16, [32])
        add(prefix + "self_attn.v_proj.weight", .bf16, [32, 64])
        add(prefix + "self_attn.v_proj.bias", .bf16, [32])
        add(prefix + "self_attn.o_proj.weight", .bf16, [64, 64])
        add(prefix + "self_attn.o_proj.bias", .bf16, [64])
        add(prefix + "self_attn.sinks", .bf16, [4])
        add(prefix + "post_attention_layernorm.weight", .bf16, [64])
        add(prefix + "mlp.router.weight", .bf16, [2, 64])
        add(prefix + "mlp.router.bias", .bf16, [2])
        let experts = prefix + "mlp.experts."
        add(experts + "gate_up_proj_blocks", .u8, [2, 128, 2, 16])
        add(experts + "gate_up_proj_scales", .u8, [2, 128, 2])
        add(experts + "gate_up_proj_bias", .bf16, [2, 128])
        add(experts + "down_proj_blocks", .u8, [2, 64, 2, 16])
        add(experts + "down_proj_scales", .u8, [2, 64, 2])
        add(experts + "down_proj_bias", .bf16, [2, 64])
    }
    add("model.norm.weight", .bf16, [64])
    add("lm_head.weight", .bf16, [128, 64])

    let metadata = IndexLoader.SourceMetadata(
        indexPath: (source as NSString)
            .appendingPathComponent("model.safetensors.index.json"),
        configPath: configPath,
        indexSha256Hex: SupportedModelSource.gptOss20B.sourceIndexSHA256,
        weightMap: Dictionary(uniqueKeysWithValues: tensors.map {
            ($0.name, $0.shardPath)
        }),
        baseBits: 4,
        baseGroupSize: 32,
        baseMode: "mxfp4",
        bitsOverrides: [:],
        shardFilenames: ["model-00000-of-00002.safetensors"])
    let header = Safetensors.Header(
        path: "model-00000-of-00002.safetensors",
        payloadBaseOffset: 24_000,
        tensors: tensors)
    let plan = try RepackPlanner.plan(
        meta: metadata,
        arch: arch,
        shardHeaders: [header],
        outputDir: output)
    return GPTOSSToyFixture(
        plan: plan, sourceRoot: source, outputRoot: output)
}

private func writeConfig(path: String,
                         production: Bool,
                         model120B: Bool = false) throws {
    let layers = production ? (model120B ? 36 : 24) : 2
    let hidden = production ? 2_880 : 64
    let heads = production ? 64 : 4
    let kvHeads = production ? 8 : 2
    let headDim = production ? 64 : 16
    let experts = production ? (model120B ? 128 : 32) : 2
    let intermediate = production ? 2_880 : 64
    let vocab = production ? 201_088 : 128
    let root: [String: Any] = [
        "model_type": "gpt_oss",
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_attention_heads": heads,
        "num_key_value_heads": kvHeads,
        "head_dim": headDim,
        "vocab_size": vocab,
        "num_hidden_layers": layers,
        "sliding_window": 128,
        "rope_theta": 150_000.0,
        "num_local_experts": experts,
        "experts_per_token": min(4, experts),
        "tie_word_embeddings": false,
        "attention_bias": true,
        "swiglu_limit": 7.0,
        "initial_context_length": 4_096,
        "max_position_embeddings": 131_072,
        "rope_scaling": [
            "rope_type": "yarn",
            "factor": 32.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "original_max_position_embeddings": 4_096,
        ],
        "layer_types": (0..<layers).map {
            $0.isMultiple(of: 2) ? "sliding_attention" : "full_attention"
        },
        "quantization_config": ["quant_method": "mxfp4"],
    ]
    try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
        .write(to: URL(fileURLWithPath: path))
}

private func temporaryRoot(_ tag: String) -> String {
    (NSTemporaryDirectory() as NSString)
        .appendingPathComponent("tuff-gptoss-\(tag)-\(UUID().uuidString)")
}
