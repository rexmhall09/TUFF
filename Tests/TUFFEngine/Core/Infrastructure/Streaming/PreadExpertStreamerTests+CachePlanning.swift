import Darwin
import Foundation
import Metal
import Testing

@testable import TUFFEngine

extension PreadExpertStreamerTests {
  @Test(arguments: [ExpertCachePolicy.lru, .lfu])
  func allHitPlanPreservesBytesAndUpdatesEvictionPolicy(policy: ExpertCachePolicy) throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: MetalContext().device,
      slotCount: 2, cachePolicy: policy)
    let warmed = try streamer.loadExpertsCached(experts: [0, 1])
    let hit = streamer.planExpertsCached(experts: [0], avoidingSlots: [0, 1])
    #expect(hit.hits == 1)
    #expect(hit.misses.isEmpty)
    let cached = try streamer.executeExpertCachePlan(hit)
    #expect(cached[0].buffer === warmed[0].buffer)
    #expect(Self.bytes(of: cached[0].buffer, offset: 0, count: Self.expertStride)
      .allSatisfy { $0 == Self.tagByte(0) })

    // The hit must still count as a use under both LRU and LFU.
    let replacement = streamer.planExpertsCached(experts: [2])
    #expect(replacement.assignedSlots == [1])
    _ = try streamer.executeExpertCachePlan(replacement)
    #expect(streamer.planExpertsCached(experts: [0, 2]).hits == 2)
  }

  @Test func failedSingleMissIsNotPublishedAsACacheHit() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: MetalContext().device, slotCount: 2)
    _ = try streamer.loadExpertsCached(experts: [0])
    #expect(truncate(url.path, off_t(Self.streamOffset + UInt64(Self.expertStride))) == 0)
    let plan = streamer.planExpertsCached(experts: [0, 1])
    #expect(plan.misses == [1])
    #expect(throws: StreamerError.self) {
      _ = try streamer.executeExpertCachePlan(plan)
    }
    let retry = streamer.planExpertsCached(experts: [0, 1])
    #expect(retry.hits == 1)
    #expect(retry.misses == [1])
  }

  @Test func cachedBatchWithoutExecutorLoadsTaggedBytes() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    let results = try streamer.loadExpertsCached(experts: [3, 1, 2])
    for (index, result) in results.enumerated() {
      let expert = [3, 1, 2][index]
      let got = Self.bytes(of: result.buffer, offset: 0, count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(expert) })
    }
  }

  @Test func adviseExpertsDoesNotChangeLoadedBytes() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)
    let experts = [0, 2, 3]

    let advice = streamer.adviseExperts(experts: experts)
    #expect(advice.requested == experts.count)
    #if os(macOS) || os(iOS) || os(tvOS) || os(watchOS)
      #expect(advice.failed == 0)
    #else
      #expect(advice.failed == experts.count)
    #endif

    let results = try streamer.loadExpertsCached(experts: experts)
    for (index, result) in results.enumerated() {
      let got = Self.bytes(of: result.buffer, offset: 0, count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(experts[index]) })
    }
  }

  @Test func adviseExpertMissesSkipsResidentSlots() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    _ = try streamer.loadExpertsCached(experts: [0])
    let advice = streamer.adviseExpertMisses(experts: [0, 1, 2])

    #expect(advice.requested == 2)
    #expect(advice.calls == 1)
    #if os(macOS) || os(iOS) || os(tvOS) || os(watchOS)
      #expect(advice.failed == 0)
    #else
      #expect(advice.failed == 1)
    #endif
  }

  @Test func plannedCacheLoadExecutesSameMisses() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    _ = try streamer.loadExpertsCached(experts: [0])
    let experts = [0, 1, 2]
    let plan = streamer.planExpertsCached(experts: experts)

    #expect(plan.hits == 1)
    #expect(plan.misses.map { experts[$0] } == [1, 2])

    let results = try streamer.executeExpertCachePlan(plan)
    for (index, result) in results.enumerated() {
      let got = Self.bytes(of: result.buffer, offset: 0, count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(experts[index]) })
    }
  }

  @Test func plannedCacheBuffersExposeReservedSlotsBeforeExecute() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    _ = try streamer.loadExpertsCached(experts: [0])
    let experts = [0, 1, 2]
    let plan = streamer.planExpertsCached(experts: experts)
    let reserved = streamer.expertCachePlanBuffers(plan)

    let hitBytes = Self.bytes(of: reserved[0].buffer, offset: 0, count: Self.expertStride)
    #expect(hitBytes.allSatisfy { $0 == Self.tagByte(0) })

    let executed = try streamer.executeExpertCachePlan(plan)
    for i in 0..<experts.count {
      #expect(reserved[i].buffer === executed[i].buffer)
      let got = Self.bytes(of: executed[i].buffer, offset: 0, count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(experts[i]) })
    }
  }

  @Test func plannedCacheAvoidsInFlightSlotsForHitsAndMisses() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    let warmed = try streamer.loadExpertsCached(experts: [0, 1])
    let plan = streamer.planExpertsCached(
      experts: [0, 2],
      avoidingSlots: [0, 1])

    #expect(plan.assignedSlots == [0, 2])
    #expect(plan.hits == 1)
    #expect(plan.misses == [1])

    let executed = try streamer.executeExpertCachePlan(plan)
    for (index, expert) in plan.experts.enumerated() {
      let got = Self.bytes(of: executed[index].buffer, offset: 0, count: Self.expertStride)
      #expect(got.allSatisfy { $0 == Self.tagByte(expert) })
    }

    let avoidedBytes = Self.bytes(of: warmed[0].buffer, offset: 0, count: Self.expertStride)
    #expect(avoidedBytes.allSatisfy { $0 == Self.tagByte(0) })
  }

  @Test func plannedCacheReturnsNilWhenMissesCannotAvoidInFlightSlots() throws {
    let url = try Self.writeSyntheticLayer()
    defer { try? FileManager.default.removeItem(at: url) }
    let device = try MetalContext().device
    let streamer = try PreadExpertStreamer(
      layout: Self.makeLayout(path: url.path), device: device, slotCount: 4)

    _ = try streamer.loadExpertsCached(experts: [0, 1])
    let plan = streamer.planExpertsCachedIfPossible(
      experts: [0, 2, 3, 4],
      avoidingSlots: [0, 1])

    #expect(plan == nil)
  }

}
