import Foundation
import Testing
@testable import TUFFEngine

@Suite("Tokenizer load coordinator")
struct TokenizerLoadCoordinatorTests {
    @Test("Concurrent loads return equivalent tokenizer metadata")
    func concurrentLoadsShareLoadedTokenizer() async throws {
        let tokenizers = try await withThrowingTaskGroup(of: GFTokenizer.self) { group in
            for _ in 0..<8 {
                group.addTask {
                    try await GFTokenizer.load()
                }
            }

            var loaded: [GFTokenizer] = []
            loaded.reserveCapacity(8)
            for try await tokenizer in group {
                loaded.append(tokenizer)
            }
            return loaded
        }

        #expect(tokenizers.count == 8)
        let first = try #require(tokenizers.first)
        for tokenizer in tokenizers {
            #expect(tokenizer.bosID == first.bosID)
            #expect(tokenizer.eosID == first.eosID)
            #expect(tokenizer.padID == first.padID)
            #expect(tokenizer.endOfTurnID == first.endOfTurnID)
            #expect(tokenizer.encode("The capital of France is", addBOS: true)
                    == first.encode("The capital of France is", addBOS: true))
        }
    }

    @Test("Consecutive loads reuse the completed process cache")
    func consecutiveLoadsReuseCompletedTask() async throws {
        let first = try await GFTokenizer.load()
        let second = try await GFTokenizer.load()

        #expect(second.bosID == first.bosID)
        #expect(second.eosID == first.eosID)
        #expect(second.endOfTurnID == first.endOfTurnID)
        #expect(second.decode(first.encode("cache check", addBOS: false)) == "cache check")
    }

    @Test("Model tokenizer sidecar is discovered")
    func modelTokenizerSidecarIsDiscovered() throws {
        let root = try temporaryDirectory()
        let model = root.appendingPathComponent("model.gturbo", isDirectory: true)
        let modelTokenizer = model.appendingPathComponent("tokenizer", isDirectory: true)
        try FileManager.default.createDirectory(at: modelTokenizer, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: modelTokenizer.appendingPathComponent("tokenizer.json"))

        let resolved = GFTokenizer.tokenizerFolder(forModelDirectory: model)

        #expect(resolved == modelTokenizer.standardizedFileURL)
    }

    @Test("Missing model tokenizer sidecar returns nil")
    func missingModelTokenizerSidecarReturnsNil() throws {
        let root = try temporaryDirectory()
        let model = root.appendingPathComponent("model.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)

        let resolved = GFTokenizer.tokenizerFolder(forModelDirectory: model)

        #expect(resolved == nil)
    }

    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("gf-tokenizer-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}

/// The hub fetch is the one part of loading a tokenizer that can fail without
/// anything being wrong. A single timed-out request once failed a whole CI run
/// of roughly seventy tests that each resolve a tokenizer.
@Suite("Tokenizer load retry")
struct TokenizerLoadRetryTests {
    private struct Transient: Error {}

    @Test("A transient failure is retried and the load succeeds")
    func retriesTransientFailures() async throws {
        actor Counter {
            var calls = 0
            func next() -> Int { calls += 1; return calls }
        }
        let counter = Counter()
        let value = try await GFTokenizerLoadRetry.run(
            attempts: GFTokenizerLoadRetry.networkAttempts,
            sleep: { _ in }
        ) {
            guard await counter.next() >= 3 else { throw Transient() }
            return "loaded"
        }

        #expect(value == "loaded")
        #expect(await counter.calls == 3)
    }

    @Test("The last failure is thrown once the attempts run out")
    func stopsAfterTheAllowedAttempts() async throws {
        actor Counter {
            var calls = 0
            func next() { calls += 1 }
        }
        let counter = Counter()
        await #expect(throws: Transient.self) {
            try await GFTokenizerLoadRetry.run(attempts: 3, sleep: { _ in }) {
                await counter.next()
                throw Transient()
            }
        }

        #expect(await counter.calls == 3)
    }

    @Test("A single attempt does not retry")
    func oneAttemptRunsOnce() async throws {
        actor Counter {
            var calls = 0
            func next() { calls += 1 }
        }
        let counter = Counter()
        await #expect(throws: Transient.self) {
            try await GFTokenizerLoadRetry.run(attempts: 1, sleep: { _ in }) {
                await counter.next()
                throw Transient()
            }
        }

        #expect(await counter.calls == 1)
    }

    /// A folder on disk either has the files or does not, so retrying it only
    /// repeats the same answer.
    @Test("Only network sources are retried")
    func onlyNetworkSourcesRetry() {
        #expect(GFTokenizerLoadRetry.attemptCount(
            for: .pretrained("some/model"), networkExhausted: false)
            == GFTokenizerLoadRetry.networkAttempts)
        #expect(GFTokenizerLoadRetry.attemptCount(
            for: .local("/tmp/tokenizer"), networkExhausted: false) == 1)
    }

    /// Once a network load has spent its attempts, a machine that simply has no
    /// network should fail fast rather than make every later test wait again.
    @Test("Retries stop being offered after the network has failed once")
    func exhaustedNetworkStopsRetrying() {
        #expect(GFTokenizerLoadRetry.attemptCount(
            for: .pretrained("some/model"), networkExhausted: true) == 1)
    }
}
