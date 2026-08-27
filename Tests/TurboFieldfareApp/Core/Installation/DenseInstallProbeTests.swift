import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite("Dense install probing")
struct DenseInstallProbeTests {
    /// A dense install carries no `packed_experts/` directory: the format
    /// rejects a dense manifest that references one. Demanding that layout file
    /// made every completed Gemma 4 E4B download report
    /// "completed install did not pass metadata validation", so the app could
    /// never finish installing its own smallest model.
    @Test("a completed dense install probes as complete without packed experts")
    func denseInstallProbesComplete() throws {
        let model = try makeCompleteDenseModelInstall("probe")
        defer { try? FileManager.default.removeItem(at: model) }

        #expect(!FileManager.default.fileExists(
            atPath: model.appendingPathComponent("packed_experts").path))
        let status = AppModelInstallationProbe.status(at: model, descriptor: .gemma4E4B)
        #expect(status == .complete)
    }

    /// A mixture-of-experts install still has to carry its expert layout.
    @Test("a mixture-of-experts install still requires its expert layout")
    func expertInstallStillRequiresLayout() throws {
        let model = try makeCompleteModelInstall("probe-moe")
        defer { try? FileManager.default.removeItem(at: model) }
        #expect(AppModelInstallationProbe.status(at: model) == .complete)

        try FileManager.default.removeItem(
            at: model.appendingPathComponent("packed_experts/layout.json"))
        let status = AppModelInstallationProbe.status(at: model)
        #expect(status != .complete)
        if case .partial(let reason) = status {
            #expect(reason.contains("packed_experts/layout.json"))
        } else {
            Issue.record("expected a partial status, got \(status)")
        }
    }
}
