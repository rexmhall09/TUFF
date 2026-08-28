import Foundation
import Testing
@testable import TUFFRepackCore

extension RemotePayloadCopyTests {
    @Test func denseGemmaRemoteInstallOmitsExpertArtifactsAndVerifies() async throws {
        let snapshotDir = tmpDirForRemote("e4b-snap")
        let output = tmpPathForRemote("e4b-output")
        defer { cleanUpRemote([snapshotDir, output]) }
        let snapshot = try SyntheticSnapshot.buildDenseGemmaE4B(at: snapshotDir)

        resetFakeHF()
        FakeHFURLProtocol.files = try remoteFiles(
            snapshotDir: snapshotDir,
            snap: snapshot,
            includeRequiredTokenizer: true,
            includeOptionalTokenizer: true)
        let result = try await RemoteStreamingRepacker(options: remoteOptions(
            outputDir: output,
            session: fakeHFSession())).run()

        #expect(result.plan.arch.feedForwardKind == .dense)
        #expect(result.plan.layers.isEmpty)
        #expect(FileManager.default.fileExists(atPath:
            (output as NSString).appendingPathComponent("model_weights.bin")))
        #expect(!FileManager.default.fileExists(atPath:
            (output as NSString).appendingPathComponent("packed_experts")))
        let manifestData = try Data(contentsOf: URL(fileURLWithPath:
            (output as NSString).appendingPathComponent("manifest.json")))
        let manifest = try JSONSerialization.jsonObject(with: manifestData)
            as! [String: Any]
        let flags = manifest["flags"] as! [String: Bool]
        #expect(manifest["versionMinor"] as? Int == 1)
        #expect(flags["denseFFN"] == true)

        let verification = try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputGTurbo: output))
        #expect(verification.unexpectedEntries.isEmpty)
        #expect(verification.bytesVerified > 0)
    }
}
