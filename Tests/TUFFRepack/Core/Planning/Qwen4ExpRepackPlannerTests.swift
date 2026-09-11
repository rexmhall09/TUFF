import Foundation
import Testing
@testable import TUFFRepackCore

@Suite struct Qwen4ExpRepackPlannerTests {
    /// The pinned `text_config`, reduced to the keys the loader reads. Values
    /// are transcribed from the checkpoint at revision `07b5dc6c`.
    private func pinnedConfig() -> [String: Any] {
        var layerTypes = [String](repeating: "linear_attention", count: 48)
        for i in stride(from: 3, to: 48, by: 4) { layerTypes[i] = "full_attention" }
        return [
            "model_type": "qwen4_exp",
            "text_config": [
                // The token the n-gram hash treats as a segment boundary.
                "eos_token_id": 248_044,
                "hidden_size": 2_560,
                "shared_expert_intermediate_size": 640,
                "moe_intermediate_size": 640,
                "num_attention_heads": 24,
                "num_key_value_heads": 2,
                "head_dim": 256,
                "vocab_size": 248_320,
                "num_hidden_layers": 48,
                "num_experts": 512,
                "num_experts_per_tok": 10,
                "tie_word_embeddings": false,
                "hidden_act": "silu",
                "layer_types": layerTypes,
                "rope_parameters": [
                    "rope_theta": 10_000_000,
                    "partial_rotary_factor": 0.25,
                ],
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 48,
                "linear_key_head_dim": 128,
                "linear_value_head_dim": 128,
                "linear_conv_kernel_dim": 4,
                "hc_count": 4,
                "hc_lowrank": 320,
                "ple_layer_ids": [2],
                "ngram_size": 3,
                "heads_per_ngram": 8,
                "ngram_vocab_size_base": 20_000_000,
                "split_ngram_parts": 128,
                "ple_embed_dim": 2_560,
                "ple_conv_kernel_size": 4,
                "indexer_budget": 2_048,
                "indexer_compress_ratio": 4,
                "indexer_head_dim": 128,
                "indexer_n_heads": 4,
                "indexer_kv_heads": 1,
            ] as [String: Any],
        ]
    }

    private func withConfig(
        _ config: [String: Any],
        _ body: (String) throws -> Void
    ) throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-qwen4exp-config-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let configURL = directory.appendingPathComponent("config.json")
        try JSONSerialization.data(withJSONObject: config).write(to: configURL)
        try body(configURL.path)
    }

    @Test func pinnedSourceIsRecognized() {
        let source = SupportedModelSource.named("qwen38-flash-next")
        #expect(source?.repoID == "mlx-community/Qwen3.8-Flash-Next-4bit")
        #expect(source?.revision == "07b5dc6c54600a359b87f1e53e7adf6351c72a2c")
        #expect(source?.sourceIndexSHA256
            == "3581f8d40a330d40009d0417359f5f75b7cf79e9f8fc48ba0b1461eabd43dd5f")
        // The fingerprint table is keyed by model ID, so a colliding ID would
        // trap at initialization rather than fail here.
        #expect(SourceFingerprint.modelID(
            forIndexSha256: source!.sourceIndexSHA256,
            repoID: source!.repoID) == source!.modelID)
        #expect(SupportedModelSource.named("flash-next")?.repoID == source?.repoID)
        #expect(SupportedModelSource.named("qwen-flash")?.repoID == source?.repoID)
    }

    @Test func configParsesIntoThePinnedArchitecture() throws {
        try withConfig(pinnedConfig()) { path in
            let arch = try ArchInfo.load(configPath: path)
            #expect(arch.family == .qwen4Exp)
            #expect(arch.variant == .qwen38FlashNext)
            #expect(arch.numLayers == 48)
            #expect(arch.numExperts == 512)
            #expect(arch.topKExperts == 10)
            #expect(arch.attnOutputGate)
            #expect(arch.sharedExpertGated)
            #expect(arch.ropeNeoxSubdim)
            #expect(arch.attentionScale == 1.0 / 16.0)

            // 36 linear-attention layers, 12 full-attention layers.
            #expect(arch.fullAttentionLayerMask.filter { $0 == 2 }.count == 36)
            #expect(arch.fullAttentionLayerMask.filter { $0 == 1 }.count == 12)
            #expect(arch.fullAttentionLayerMask[3] == 1)
            #expect(arch.fullAttentionLayerMask[47] == 1)

            #expect(arch.hyperConnectionStreamCount == 4)
            #expect(arch.hyperConnectionLowRank == 320)
            #expect(arch.indexerBudget == 2_048)
            #expect(arch.indexerNumHeads == 4)
            #expect(arch.indexerNumKVHeads == 1)
        }
    }

    /// `ple_layer_ids` names layer 2; the checkpoint emits the tensors under
    /// `layers.1`, and tensor names are what the loader resolves against.
    @Test func ngramModuleTakesTheLayerItsTensorsAreUnder() throws {
        try withConfig(pinnedConfig()) { path in
            let arch = try ArchInfo.load(configPath: path)
            #expect(arch.ngramLayer == 1)
            #expect(arch.ngramSize == 3)
            #expect(arch.ngramShardCount == 128)
            #expect(arch.ngramHeadsPerNgram == 8)
            // 128 shards split 8 ways per head; the checkpoint's
            // ngram_heads_offsets carries exactly this many entries.
            #expect(arch.ngramHeads == 16)
            #expect(arch.ngramEmbedDim == 2_560)
        }
    }

    @Test func aSplitThatDoesNotDivideIntoHeadsIsRejected() throws {
        var config = pinnedConfig()
        var text = config["text_config"] as! [String: Any]
        text["split_ngram_parts"] = 127
        config["text_config"] = text
        try withConfig(config) { path in
            #expect(throws: RepackError.self) {
                _ = try ArchInfo.load(configPath: path)
            }
        }
    }

    /// A config claiming the production shape while disagreeing on a field the
    /// repack depends on must not produce a plan.
    @Test func productionShapeIsCrossChecked() throws {
        var config = pinnedConfig()
        var text = config["text_config"] as! [String: Any]
        text["num_experts_per_tok"] = 8
        config["text_config"] = text
        try withConfig(config) { path in
            #expect(throws: RepackError.self) {
                _ = try ArchInfo.load(configPath: path)
            }
        }
    }

    @Test func sourceTensorNamesSplitResidentAndStreamedExperts() {
        let prefix = "language_model.model.layers.7"
        #expect(RepackPlanner.classify(
            "\(prefix).mlp.switch_mlp.gate_proj.weight",
            numLayers: 48, family: .qwen4Exp) == .routedExpert(role: "gate", layer: 7))
        #expect(RepackPlanner.classify(
            "\(prefix).mlp.switch_mlp.down_proj.weight",
            numLayers: 48, family: .qwen4Exp) == .routedExpert(role: "down", layer: 7))
        #expect(RepackPlanner.classify(
            "\(prefix).mlp.shared_expert.gate_proj.weight",
            numLayers: 48, family: .qwen4Exp) == .lmResident)
        #expect(RepackPlanner.classify(
            "\(prefix).attn_hyper_connection.block_inject_weight.weight",
            numLayers: 48, family: .qwen4Exp) == .lmResident)
        #expect(RepackPlanner.classify(
            "\(prefix).self_attn.indexer.index_qk_proj.weight",
            numLayers: 48, family: .qwen4Exp) == .lmResident)
        #expect(RepackPlanner.classify(
            "language_model.lm_head.weight",
            numLayers: 48, family: .qwen4Exp) == .lmResident)
    }

    /// The n-gram table is 29.80 GiB against 2.91 GiB for every other resident
    /// tensor combined, so it must never fall into the resident bucket.
    @Test func ngramShardsAreTheirOwnBucket() {
        let name = "language_model.model.layers.1.ple.ple_embedding"
            + ".ngram_embedding.shards.37.weight"
        #expect(RepackPlanner.classify(name, numLayers: 48, family: .qwen4Exp)
            == .ngramEmbeddingShard(shard: 37, layer: 1))
        // The projections around the table are ordinary resident tensors.
        #expect(RepackPlanner.classify(
            "language_model.model.layers.1.ple.key_proj.weight",
            numLayers: 48, family: .qwen4Exp) == .lmResident)
        #expect(RepackPlanner.ngramShardIndex(
            in: "language_model.model.layers.1.ple.conv1d.weight") == nil)
    }
}
