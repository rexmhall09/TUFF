import Foundation
import Testing
@testable import TurboFieldfareRepackCore

@Suite struct GemmaE4BRepackPlannerTests {
    @Test func pinnedSourceAndSelectorsAreStable() {
        let source = SupportedModelSource.gemma4E4B
        #expect(source.name == "gemma4-e4b")
        #expect(source.aliases == ["e4b"])
        #expect(source.repoID == "mlx-community/gemma-4-e4b-it-4bit")
        #expect(source.revision == "475b9088d29754a3379866cf5aeb6b41acd313c2")
        #expect(SourceFingerprint.modelID(
            forIndexSha256: source.sourceIndexSHA256,
            repoID: source.repoID) == source.modelID)
    }

    @Test func productionE4BConfigParsesAndCrossChecks() throws {
        let root = temporaryE4BRoot("production-config")
        defer { try? FileManager.default.removeItem(atPath: root) }
        try FileManager.default.createDirectory(
            atPath: root, withIntermediateDirectories: true)
        let path = (root as NSString).appendingPathComponent("config.json")
        try writeProductionE4BConfig(path: path)
        let arch = try ArchInfo.load(configPath: path)
        #expect(arch.hiddenSize == 2_560)
        #expect(arch.intermediateSize == 10_240)
        #expect(arch.numFullKVHeads == 2)
        #expect(arch.fullAttentionLayerMask.filter { $0 == 1 }.count == 7)
        #expect(arch.hiddenSizePerLayerInput == 256)
        #expect(arch.numKVSharedLayers == 18)
    }

    @Test func denseConfigAndPlanCarryPLEWithoutExpertFiles() throws {
        let snapshotDir = temporaryE4BRoot("plan-source")
        let outputDir = temporaryE4BRoot("plan-output")
        defer {
            try? FileManager.default.removeItem(atPath: snapshotDir)
            try? FileManager.default.removeItem(atPath: outputDir)
        }
        let snapshot = try SyntheticSnapshot.buildDenseGemmaE4B(at: snapshotDir)
        let metadata = try IndexLoader.load(snapshotDir: snapshotDir)
        let arch = try ArchInfo.load(configPath:
            (snapshotDir as NSString).appendingPathComponent("config.json"))
        let header = try parseE4BHeader(path: snapshot.shardPath)
        let plan = try RepackPlanner.plan(
            meta: metadata, arch: arch, shardHeaders: [header], outputDir: outputDir)

        #expect(arch.variant == .gemma4_E4B)
        #expect(arch.feedForwardKind == .dense)
        #expect(arch.hiddenSizePerLayerInput == 64)
        #expect(arch.numKVSharedLayers == 2)
        #expect(plan.layers.isEmpty)
        #expect(plan.resident.entries.contains {
            $0.name == "language_model.model.embed_tokens_per_layer.weight"
        })
        #expect(plan.resident.entries.contains {
            $0.name == "language_model.model.layers.0.per_layer_input_gate.weight"
        })
        #expect(plan.excludedMultimodalTensorNames.sorted() == [
            "audio_tower.layers.0.norm_out.weight",
            "embed_audio.embedding_projection.biases",
            "embed_audio.embedding_projection.scales",
            "embed_audio.embedding_projection.weight",
        ])
    }

    @Test func denseManifestUsesV11AndExplicitFeatureFlag() throws {
        let (plan, arch, cleanup) = try makeE4BPlan(tag: "manifest")
        defer { cleanup() }
        let data = try GTurboJSON.encodeManifest(
            plan: plan,
            modelID: "dense-e4b-toy",
            sourceSnapshotHash: "sha256:toy",
            files: [],
            expertsPerLayer: 0,
            numLayers: arch.numLayers,
            expertStride: 0,
            bitWidths: .init(
                embedding: 4, attention: 4, router: 8,
                sharedExpert: 4, routedExpert: 4))
        let root = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let flags = root["flags"] as! [String: Bool]
        let manifestArch = root["arch"] as! [String: Any]
        #expect(root["versionMajor"] as? Int == 1)
        #expect(root["versionMinor"] as? Int == 1)
        #expect(flags["denseFFN"] == true)
        #expect(flags["streamingPresent"] == false)
        #expect(manifestArch["variant"] as? String == "gemma4-e4b")
        #expect(manifestArch["feedForwardKind"] as? String == "dense")
        #expect(manifestArch["hiddenSizePerLayerInput"] as? Int == 64)
        #expect(manifestArch["vocabSizePerLayerInput"] as? Int == 256)
        #expect(manifestArch["numKVSharedLayers"] as? Int == 2)
    }
}

private func makeE4BPlan(tag: String) throws
    -> (RepackPlan, ArchInfo, () -> Void) {
    let source = temporaryE4BRoot(tag + "-source")
    let output = temporaryE4BRoot(tag + "-output")
    let snapshot = try SyntheticSnapshot.buildDenseGemmaE4B(at: source)
    let metadata = try IndexLoader.load(snapshotDir: source)
    let arch = try ArchInfo.load(configPath:
        (source as NSString).appendingPathComponent("config.json"))
    let plan = try RepackPlanner.plan(
        meta: metadata, arch: arch,
        shardHeaders: [try parseE4BHeader(path: snapshot.shardPath)],
        outputDir: output)
    return (plan, arch, {
        try? FileManager.default.removeItem(atPath: source)
        try? FileManager.default.removeItem(atPath: output)
    })
}

private func parseE4BHeader(path: String) throws -> Safetensors.Header {
    try Safetensors.parseHeader(path: path)
}

private func temporaryE4BRoot(_ tag: String) -> String {
    (NSTemporaryDirectory() as NSString)
        .appendingPathComponent("tuff-e4b-\(tag)-\(UUID().uuidString)")
}

private func writeProductionE4BConfig(path: String) throws {
    let layerTypes = (0..<42).map {
        ($0 + 1).isMultiple(of: 6) ? "full_attention" : "sliding_attention"
    }
    let text: [String: Any] = [
        "hidden_size": 2_560,
        "intermediate_size": 10_240,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "global_head_dim": 512,
        "vocab_size": 262_144,
        "num_hidden_layers": 42,
        "sliding_window": 512,
        "final_logit_softcapping": 30.0,
        "rope_parameters": [
            "sliding_attention": ["rope_theta": 10_000.0],
            "full_attention": ["rope_theta": 1_000_000.0,
                               "partial_rotary_factor": 0.25],
        ],
        "layer_types": layerTypes,
        "tie_word_embeddings": true,
        "attention_k_eq_v": false,
        "hidden_activation": "gelu_pytorch_tanh",
        "enable_moe_block": false,
        "hidden_size_per_layer_input": 256,
        "vocab_size_per_layer_input": 262_144,
        "num_kv_shared_layers": 18,
    ]
    let root: [String: Any] = ["model_type": "gemma4", "text_config": text]
    try JSONSerialization.data(withJSONObject: root, options: [.sortedKeys])
        .write(to: URL(fileURLWithPath: path))
}
