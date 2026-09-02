import Foundation
import Testing
@testable import TUFFRepackCore

@Suite struct MiniMaxRepackPlannerTests {
    @Test func pinnedSourceAndArchitectureAreRecognized() throws {
        let source = SupportedModelSource.named("minimax-m2.7")
        #expect(source?.repoID == "mlx-community/MiniMax-M2.7-4bit")
        #expect(source?.revision == "66d2e5cb7c5cda05251b4625c504af4b034df7ff")
        #expect(SourceFingerprint.modelID(
            forIndexSha256: source!.sourceIndexSHA256,
            repoID: source!.repoID) == source!.modelID)

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-minimax-config-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let configURL = directory.appendingPathComponent("config.json")
        let config: [String: Any] = [
            "model_type": "minimax_m2",
            "hidden_size": 3_072,
            "intermediate_size": 1_536,
            "num_hidden_layers": 62,
            "num_attention_heads": 48,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rotary_dim": 64,
            "vocab_size": 200_064,
            "rope_theta": 5_000_000,
            "num_local_experts": 256,
            "num_experts_per_tok": 8,
            "tie_word_embeddings": false,
            "hidden_act": "silu",
            "use_qk_norm": true,
        ]
        try JSONSerialization.data(withJSONObject: config).write(to: configURL)
        let arch = try ArchInfo.load(configPath: configURL.path)
        #expect(arch.family == .minimaxM2)
        #expect(arch.variant == .minimaxM27)
        #expect(arch.fullAttentionLayerMask == [UInt8](repeating: 1, count: 62))
        #expect(arch.partialRotaryFactor == 0.5)
    }

    @Test func sourceTensorNamesSplitResidentAndStreamedExperts() {
        #expect(RepackPlanner.classify(
            "model.layers.7.block_sparse_moe.switch_mlp.gate_proj.weight",
            numLayers: 62, family: .minimaxM2) == .routedExpert(role: "gate", layer: 7))
        #expect(RepackPlanner.classify(
            "model.layers.7.block_sparse_moe.e_score_correction_bias",
            numLayers: 62, family: .minimaxM2) == .lmResident)
        #expect(RepackPlanner.classify(
            "lm_head.weight", numLayers: 62, family: .minimaxM2) == .lmResident)
    }
}
