import Foundation
import Synchronization
import Testing
import TUFFModelCatalog
import TUFFInferenceTypes
@testable import TUFFAppCore

@Suite struct AppAutoCalibratorTests {
    @Test func cancelledPrefillPreservesHealthButCannotBecomeATimingSample() {
        let empty = InferencePlanObservation(candidateID: "prefill", workloadKey: "cold-short",
            decodeTokens: 0, decodeSeconds: 0, timeToFirstTokenSeconds: 0)
        let paging = InferencePlanObservation(candidateID: "prefill", workloadKey: "cold-short",
            decodeTokens: 0, decodeSeconds: 0, timeToFirstTokenSeconds: 0, swapGrowthBytes: 4096)
        let pressured = InferencePlanObservation(candidateID: "prefill", workloadKey: "cold-short",
            decodeTokens: 0, decodeSeconds: 0, timeToFirstTokenSeconds: 0, pressure: .warning)
        #expect(!AppAutoPerformanceRecords.accepts(empty))
        #expect(AppAutoPerformanceRecords.accepts(paging))
        #expect(AppAutoPerformanceRecords.accepts(pressured))
        #expect(!AppAutoPerformanceRecords.accepts(.init(candidateID: "invalid", workloadKey: "cold-short",
            decodeTokens: -1, decodeSeconds: 0, timeToFirstTokenSeconds: 0, swapGrowthBytes: 4096)))
    }
    /// The app, `tuff prompt` and `tuff serve` name a run's settings the same
    /// way, in the form records written before 9.1.2 already use.
    @Test func candidateNamesMatchTheRecordedForm() {
        #expect(AppAutoPerformanceRecords.candidateID(contextTokens: 16_384, expertCacheSlots: 16,
            residentDenseLayers: 30, prefillChunkTokens: 512) == "c16384-e16-d30-p512")
    }

    @Test func pagingWritesCountEvenWhenNetSwapOccupancyFalls() {
        #expect(AppAutoMemoryPressureMonitor.pagingBytes(startUsed: 1_000, endUsed: 900,
            startWritten: 2_000, endWritten: 2_500) == 500)
        #expect(AppAutoMemoryPressureMonitor.pagingBytes(startUsed: 1_000, endUsed: 1_200,
            startWritten: 2_000, endWritten: 2_050) == 200)
        #expect(AppAutoMemoryPressureMonitor.pagingBytes(startUsed: 1_000, endUsed: 900,
            startWritten: 2_000, endWritten: 1_900) == 0)
    }
    @Test func smallerExpertCacheProbeCanBuyMoreResidentProjections() throws {
        let descriptor = AppModelInstallDescriptor(catalog: TUFFModelCatalog.ornith15_397B)
        let summary = try #require(AppAutomaticMemoryPlanner.predictedLayout(for: descriptor))
        let device = TUFFDeviceCapabilities(unifiedMemoryBytes: 16 << 30, macOSMajorVersion: 26, appleSiliconGeneration: 2)
        let legal = AppAutomaticLayoutPlanner.candidates(catalog: TUFFModelCatalog.ornith15_397B,
            descriptor: descriptor, summary: summary, budget: device.safeAppMemoryBudgetBytes,
            contextOverride: 2_048, recommendedChunk: 2_048)
        let baseline = try #require(legal.filter { $0.slots == 16 && $0.chunk == 2_048 }
            .max { $0.denseLayers.count < $1.denseLayers.count })
        let probe = try #require(AppAutoCalibrator.cacheTradeoff(legal,
            targetSlots: baseline.slots / 2, baselineChunk: baseline.chunk))
        #expect(probe.slots < baseline.slots)
        #expect(probe.denseLayers.count > baseline.denseLayers.count)
        #expect(probe.selection.peakAllocatedBytes <= device.safeAppMemoryBudgetBytes)
    }
    private final class Client: AppModelLifecycleClient, @unchecked Sendable {
        let loaded = Mutex<AppRuntimeOptions?>(nil)
        func ensureLoaded(modelDirectory: URL, maxContextTokens: Int, options: AppRuntimeOptions,
                          forceLogitsHead: Bool, onState: @escaping @Sendable (AppModelLoadState) -> Void) async throws {
            loaded.withLock { $0 = options }
        }
        func unload() async {}
        func cancel() {}
        func generate(_ request: AppGenerationRequest) -> AsyncThrowingStream<AppInferenceEvent, Error> {
            AsyncThrowingStream { continuation in
                let slots = request.runtimeOptions.expertCacheSlots
                let seconds = slots == 16 ? 1.0 : (slots > 16 ? 3.0 : 2.0)
                continuation.yield(.finished(AppDiagnostics(generatedTokens: 16, stopReason: .maxTokens,
                    promptTokenCount: 24, prefillSeconds: 0.5, timeToFirstTokenSeconds: 0.1,
                    decodeSeconds: seconds, tokensPerSecond: 16 / seconds,
                    peakMemoryBytes: nil, runtimeOptions: request.runtimeOptions)))
                continuation.finish()
            }
        }
    }

    @Test func repeatedProbesRejectASlowerLargerCacheAndRestoreTheSelectedRuntime() async throws {
        let descriptor = AppModelInstallDescriptor.qwen36
        let summary = try #require(AppAutomaticMemoryPlanner.predictedLayout(for: descriptor))
        let device = TUFFDeviceCapabilities(unifiedMemoryBytes: 16 << 30, macOSMajorVersion: 26, appleSiliconGeneration: 2)
        let baseline = try #require(AppAutomaticMemoryPlanner.plan(for: descriptor, on: device,
            profile: .speed, layoutSummary: summary, contextTokens: 2_048))
        let client = Client()
        let records = Mutex<[InferencePlanObservation]>([])
        let selected = try await AppAutoCalibrator.run(client: client, lifecycle: client,
            directory: URL(fileURLWithPath: "/synthetic-fixture"), descriptor: descriptor, device: device,
            summary: summary, baseline: baseline, options: AppRuntimeOptions(), forceLogitsHead: true,
            record: { _, observation in records.withLock { $0.append(observation) } },
            observations: { _ in records.withLock { $0 } })
        #expect(selected.expertCacheSlots == 16)
        #expect(selected.evidence == .observed)
        #expect(records.withLock { $0.count } <= AppAutoCalibrator.maximumCandidates * 3)
        #expect(records.withLock { $0.filter { $0.candidateID == selected.candidateID }.count } >= 3)
        #expect(client.loaded.withLock { $0?.expertCacheSlots } == selected.expertCacheSlots)
        #expect(client.loaded.withLock { $0?.prefillChunkTokens } == selected.prefillChunkTokens)
    }
}
