import Foundation
import Testing

@testable import TUFFEngine

extension PreadExpertStreamerTests {
  /// A demand miss on an expert a prediction pushed out is counted; one on an
  /// expert demand itself evicted is not.
  @Test func demandMissesCausedByPredictionsAreCounted() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: MetalContext().device, slotCount: 1)
    _ = try streamer.loadExpertsCached(experts: [0])
    _ = try streamer.executeExpertCachePlan(
      streamer.planExpertsCached(experts: [1], purpose: .prefetch))
    _ = streamer.planExpertsCached(experts: [0])
    #expect(streamer.readMetrics.pollutedDemandMisses == 1)
    _ = streamer.planExpertsCached(experts: [2])
    _ = streamer.planExpertsCached(experts: [0])
    #expect(streamer.readMetrics.pollutedDemandMisses == 1)
  }
}
