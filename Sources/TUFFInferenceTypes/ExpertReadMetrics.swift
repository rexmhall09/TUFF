import Foundation

/// Logical pread traffic, including OS-cache hits; physical SSD traffic is not measured.
public struct ExpertReadMetrics: Codable, Sendable, Equatable {
    /// Expert requests accepted by cache planning, including hits.
    public var demandRequests: UInt64 = 0
    /// Speculative expert requests, including predictions already cached.
    public var predictionRequests: UInt64 = 0
    public var demandReads: UInt64 = 0
    public var demandBytes: UInt64 = 0
    public var demandFailures: UInt64 = 0
    public var prefetchReads: UInt64 = 0
    public var prefetchBytes: UInt64 = 0
    public var prefetchFailures: UInt64 = 0
    /// Completed speculative records subsequently hit by demand, once per insertion.
    public var usefulPrefetchReads: UInt64 = 0
    /// Completed speculative records evicted before their first demand hit.
    public var unusedPrefetchEvictions: UInt64 = 0
    /// Demand misses on an expert whose slot was last taken by a prediction:
    /// the reads a wrong guess cost by pushing out something needed.
    public var pollutedDemandMisses: UInt64 = 0

    public init() {}

    private enum CodingKeys: String, CodingKey {
        case demandRequests, predictionRequests
        case demandReads, demandBytes, demandFailures, prefetchReads, prefetchBytes, prefetchFailures
        case usefulPrefetchReads, unusedPrefetchEvictions, pollutedDemandMisses
    }

    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        demandRequests = try values.decodeIfPresent(UInt64.self, forKey: .demandRequests) ?? 0
        predictionRequests = try values.decodeIfPresent(UInt64.self, forKey: .predictionRequests) ?? 0
        demandReads = try values.decodeIfPresent(UInt64.self, forKey: .demandReads) ?? 0
        demandBytes = try values.decodeIfPresent(UInt64.self, forKey: .demandBytes) ?? 0
        demandFailures = try values.decodeIfPresent(UInt64.self, forKey: .demandFailures) ?? 0
        prefetchReads = try values.decodeIfPresent(UInt64.self, forKey: .prefetchReads) ?? 0
        prefetchBytes = try values.decodeIfPresent(UInt64.self, forKey: .prefetchBytes) ?? 0
        prefetchFailures = try values.decodeIfPresent(UInt64.self, forKey: .prefetchFailures) ?? 0
        usefulPrefetchReads = try values.decodeIfPresent(UInt64.self, forKey: .usefulPrefetchReads) ?? 0
        unusedPrefetchEvictions = try values.decodeIfPresent(UInt64.self, forKey: .unusedPrefetchEvictions) ?? 0
        pollutedDemandMisses = try values.decodeIfPresent(UInt64.self, forKey: .pollutedDemandMisses) ?? 0
    }

    public func subtracting(_ baseline: Self) -> Self {
        var delta = Self()
        delta.demandRequests = demandRequests &- baseline.demandRequests
        delta.predictionRequests = predictionRequests &- baseline.predictionRequests
        delta.demandReads = demandReads &- baseline.demandReads
        delta.demandBytes = demandBytes &- baseline.demandBytes
        delta.demandFailures = demandFailures &- baseline.demandFailures
        delta.prefetchReads = prefetchReads &- baseline.prefetchReads
        delta.prefetchBytes = prefetchBytes &- baseline.prefetchBytes
        delta.prefetchFailures = prefetchFailures &- baseline.prefetchFailures
        delta.usefulPrefetchReads = usefulPrefetchReads &- baseline.usefulPrefetchReads
        delta.unusedPrefetchEvictions = unusedPrefetchEvictions &- baseline.unusedPrefetchEvictions
        delta.pollutedDemandMisses = pollutedDemandMisses &- baseline.pollutedDemandMisses
        return delta
    }

}
