import Testing
import Foundation
import Metal
@testable import TUFFEngine

@Suite(.serialized) struct DenseGemmaRunnerTests {
    private func makeRunner() throws -> (URL, MetalContext, Model, RealForwardRunner) {
        let directory = try DenseGemmaToySynthetic.write()
        let context = try MetalContext()
        let model = try Model.load(directoryURL: directory,
                                   device: context.device,
                                   expecting: .gemma4E4BToy())
        let runner = try RealForwardRunner(model: model,
                                           context: context,
                                           maxContext: 64)
        return (directory, context, model, runner)
    }

    private func logits(_ context: MetalContext) -> MTLBuffer {
        context.device.makeBuffer(length: 256 * MemoryLayout<Float16>.stride,
                                  options: .storageModeShared)!
    }

    @Test func denseInstallLoadsWithoutExpertDirectory() throws {
        let (directory, _, model, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }
        #expect(model.config.feedForwardKind == .dense)
        #expect(model.openLayerFileCount() == 0)
        #expect(runner.maxContext == 64)
    }

    @Test func decodeUsesPLEAndSharedKVWithoutOpeningExperts() async throws {
        let (directory, context, model, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }
        let output = logits(context)
        try await runner.produce(token: 7, position: 0, into: output)
        #expect(runner.lastGreedyToken < 256)
        try await runner.produce(token: Int32(runner.lastGreedyToken),
                                 position: 1, into: output)
        #expect(runner.continuationPosition == 2)
        #expect(model.openLayerFileCount() == 0)
    }

    @Test func chunkedPrefillContinuesIntoDecode() async throws {
        let (directory, context, _, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }
        let output = logits(context)
        let prompt: [Int32] = [2, 5, 8, 13]
        let result = try await runner.prefillChunked(
            tokens: prompt[...],
            startPosition: 0,
            outputMode: .greedyIfAvailable,
            config: .production(chunkTokens: 32),
            into: output,
            onProgress: { _ in })
        #expect(result.newPosition == prompt.count)
        try await runner.produce(token: 21, position: prompt.count, into: output)
        #expect(runner.continuationPosition == prompt.count + 1)
        #expect(runner.lastGreedyToken < 256)
    }

    @Test func prefillThenDecodeMatchesPureDecodeArgmax() async throws {
        let (directory, context, _, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }
        let output = logits(context)
        try await runner.produce(token: 3, position: 0, into: output)
        try await runner.produce(token: 9, position: 1, into: output)
        let reference = runner.lastGreedyToken

        runner.reset()
        let first: [Int32] = [3]
        _ = try await runner.prefillChunked(
            tokens: first[...],
            startPosition: 0,
            outputMode: .greedyIfAvailable,
            config: .production(chunkTokens: 32),
            into: output,
            onProgress: { _ in })
        try await runner.produce(token: 9, position: 1, into: output)
        #expect(runner.lastGreedyToken == reference)
    }

    @Test func speculativeBlockMatchesScalarArgmaxAndCommitsAllRows() async throws {
        let (directory, context, _, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }
        let output = logits(context)
        try await runner.produce(token: 7, position: 0, into: output)
        let boundary = Int32(bitPattern: runner.lastGreedyToken)
        let candidates: [Int32] = [9, 11, 13, 17]
        var scalarTargets = [boundary]
        for (index, candidate) in candidates.enumerated() {
            try await runner.produce(token: candidate,
                                     position: index + 1,
                                     into: output)
            scalarTargets.append(Int32(bitPattern: runner.lastGreedyToken))
        }

        runner.reset()
        try await runner.produce(token: 7, position: 0, into: output)
        let result = try await runner.verifySpeculativeBlock(
            tokens: candidates, startPosition: 1, into: output)
        #expect(result.targetTokenIDs == scalarTargets)
        #expect(runner.continuationPosition == 5)
        try runner.commitSpeculativePrefix(candidates.count)
        #expect(runner.continuationPosition == 5)
    }

    @Test func speculativeRollbackLeavesAUsableKVBoundary() async throws {
        let (directory, context, _, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }
        let output = logits(context)
        try await runner.produce(token: 7, position: 0, into: output)
        let candidates: [Int32] = [9, 11, 13, 17]
        let result = try await runner.verifySpeculativeBlock(
            tokens: candidates, startPosition: 1, into: output)
        try runner.commitSpeculativePrefix(2)
        #expect(runner.continuationPosition == 3)
        try await runner.produce(token: result.targetTokenIDs[2],
                                 position: 3, into: output)
        #expect(runner.continuationPosition == 4)
    }
}
