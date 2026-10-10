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

    /// Generating past the checkpoint overwrites the sliding-window ring
    /// (window 32 here, with PLE and shared KV layers). Rewinding must bring
    /// back exactly the state at the checkpoint, so continuing from it equals
    /// a fresh run, logit for logit.
    @Test func rewindingToACheckpointMatchesAFreshRun() async throws {
        let directory = try DenseGemmaToySynthetic.write()
        defer { try? FileManager.default.removeItem(at: directory) }
        let context = try MetalContext()
        let model = try Model.load(directoryURL: directory, device: context.device,
                                   expecting: .gemma4E4BToy())
        let runner = try RealForwardRunner(
            model: model, context: context, maxContext: 512,
            runtimeConfiguration: RuntimeConfiguration(forceLogitsHead: true))
        #expect(runner.supportsPrefixCheckpoints)
        let output = logits(context)
        func values() -> [Float16] {
            let pointer = output.contents().assumingMemoryBound(to: Float16.self)
            return (0..<256).map { pointer[$0] }
        }
        let prompt = (0..<40).map { Int32(($0 * 7) % 200 + 1) }
        let next: [Int32] = [17, 4, 61]

        for (position, token) in prompt.enumerated() {
            try await runner.produce(token: token, position: position, into: output)
        }
        let checkpoint = try runner.capturePrefixCheckpoint()
        #expect(checkpoint.position == 40)
        for offset in 0..<300 {
            try await runner.produce(token: Int32(offset % 90 + 2), position: 40 + offset,
                                     into: output)
        }
        try runner.rewind(to: checkpoint)
        #expect(runner.continuationPosition == 40)
        try runner.prepareForContinuation(expectedPosition: 40)
        for (offset, token) in next.enumerated() {
            try await runner.produce(token: token, position: 40 + offset, into: output)
        }
        let resumed = values()

        runner.reset()
        for (position, token) in (prompt + next).enumerated() {
            try await runner.produce(token: token, position: position, into: output)
        }
        #expect(resumed == values())
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

    @Test func speculativeVerifierMicrobenchmark() async throws {
        let (directory, context, _, runner) = try makeRunner()
        defer { try? FileManager.default.removeItem(at: directory) }
        let output = logits(context)
        let candidates: [Int32] = [9, 11, 13, 17, 19, 23, 29, 31]
        for blockSize in [2, 4, 6, 8] {
            var scalarNanos: [UInt64] = []
            var blockNanos: [UInt64] = []
            var lastMetrics = SpeculativeVerificationMetrics()
            for _ in 0..<4 {
                runner.reset()
                try await runner.produce(token: 7, position: 0, into: output)
                let scalarStart = DispatchTime.now().uptimeNanoseconds
                for (index, candidate) in candidates.prefix(blockSize).enumerated() {
                    try await runner.produce(token: candidate,
                                             position: index + 1,
                                             into: output)
                }
                scalarNanos.append(DispatchTime.now().uptimeNanoseconds - scalarStart)

                runner.reset()
                try await runner.produce(token: 7, position: 0, into: output)
                let blockStart = DispatchTime.now().uptimeNanoseconds
                let result = try await runner.verifySpeculativeBlock(
                    tokens: Array(candidates.prefix(blockSize)),
                    startPosition: 1,
                    into: output)
                try runner.commitSpeculativePrefix(blockSize)
                blockNanos.append(DispatchTime.now().uptimeNanoseconds - blockStart)
                lastMetrics = result.metrics
                #expect(result.metrics.wallNanos > 0)
            }
            let scalarMedian = scalarNanos.sorted()[scalarNanos.count / 2]
            let blockMedian = blockNanos.sorted()[blockNanos.count / 2]
            let speedup = Double(scalarMedian) / Double(blockMedian)
            print("[spec-benchmark backend=dense-gemma-toy block=\(blockSize) "
                  + "repeats=4 scalarMedianMs=\(Double(scalarMedian) / 1e6) "
                  + "blockMedianMs=\(Double(blockMedian) / 1e6) "
                  + "speedup=\(speedup) expertReads=\(lastMetrics.expertReads) "
                  + "expertBytes=\(lastMetrics.expertBytes)]")
        }
    }
}
