import Foundation
import Darwin

/// Decode expert lookahead: routing layer L's feed-forward input through
/// layer L+1's router predicts L+1's experts, and their SSD reads start while
/// layer L is still computing.
///
/// On a 16 GB M2, decode waited on expert reads for 42-56% of its time on
/// Gemma 4 26B, Qwen 3.6 and Qwen 3.8 Flash Next, serially after each layer's
/// router. A prediction only decides what is read ahead into layer L+1's own
/// cache slots; the layer still routes on its real input, so a wrong guess
/// costs bandwidth and a cache slot, never correctness. Lookahead turns itself
/// off for the session when its predictions are mostly wrong.
struct ExpertLookahead {
    /// A read in flight, or a prediction whose experts were already cached.
    struct Pending {
        let layer: Int
        let experts: [Int]
        let reads: DispatchGroup?
    }

    /// Predictions scored before lookahead may turn itself off.
    static let minimumSample = 1_024
    /// Fraction of predicted experts the real router must pick to keep going.
    static let minimumPrecision = 0.35

    private(set) var predicted = 0
    private(set) var correct = 0
    private(set) var readsIssued = 0
    private(set) var exposedWaitNanos: UInt64 = 0
    private(set) var enabled = true
    var pending: Pending?
    /// How many layers ahead a runner that supports it predicts. GPT-OSS
    /// reads two ahead, so a read has two layers of compute to hide behind,
    /// and tops up one ahead with whatever the earlier guess missed.
    let depth: Int
    /// Reads for layers further ahead than the next, by layer.
    private var ahead: [Int: Pending] = [:]

    // Read once when the runner is created, never while GPU work is in flight.
    init(environment: [String: String] = ProcessInfo.processInfo.environment) {
        enabled = environment["TUFF_EXPERT_LOOKAHEAD"] != "off"
        depth = environment["TUFF_EXPERT_LOOKAHEAD_DEPTH"] == "1" ? 1 : 2
    }

    var precision: Double {
        predicted == 0 ? 0 : Double(correct) / Double(predicted)
    }

    mutating func record(predicted experts: [Int], actual: [Int]) {
        predicted += experts.count
        correct += Set(experts).intersection(actual).count
        if enabled, predicted >= Self.minimumSample, precision < Self.minimumPrecision {
            enabled = false
        }
    }

    mutating func noteRead() { readsIssued += 1 }

    /// Waits for the in-flight read, if any, and returns what it predicted.
    /// Every path that plans or fetches experts drains first, so a background
    /// read never races the layer it fills. Reads for layers further ahead
    /// are drained too.
    mutating func drain() -> Pending? {
        for layer in ahead.keys.sorted() { _ = drain(layer: layer) }
        guard let pending else { return nil }
        wait(for: pending)
        self.pending = nil
        return pending
    }

    /// Adds a prediction for `layer`, with the read that fills it, to any
    /// already made. One group per layer, so draining waits for all of them.
    mutating func expect(layer: Int, experts: [Int], reads: DispatchGroup?) {
        let earlier = ahead[layer]
        let merged = (earlier?.experts ?? []) + experts.filter { !(earlier?.experts.contains($0) ?? false) }
        ahead[layer] = Pending(layer: layer, experts: merged, reads: reads ?? earlier?.reads)
    }

    /// Whether `layer`'s reads have all landed, without waiting. Its streamer
    /// may be planned again only then.
    func isSettled(layer: Int) -> Bool {
        guard let reads = ahead[layer]?.reads else { return true }
        return reads.wait(timeout: .now()) == .success
    }

    /// Waits for `layer`'s reads and returns everything predicted for it.
    mutating func drain(layer: Int) -> Pending? {
        guard let pending = ahead.removeValue(forKey: layer) else { return nil }
        wait(for: pending)
        return pending
    }

    private mutating func wait(for pending: Pending) {
        if let reads = pending.reads, reads.wait(timeout: .now()) == .timedOut {
            let start = clock_gettime_nsec_np(CLOCK_UPTIME_RAW)
            reads.wait()
            exposedWaitNanos += clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - start
        }
    }
}
