import Foundation
import Testing
@testable import TurboFieldfareRepackCore

extension RemotePayloadCopyTests {
    @Test func gptOssRemoteInstallStreamsMXFP4AndVerifies() async throws {
        let snapshotDir = tmpDirForRemote("gptoss-snap")
        let output = tmpPathForRemote("gptoss-output")
        defer { cleanUpRemote([snapshotDir, output]) }
        let snapshot = try SyntheticSnapshot.buildGPTOSS(at: snapshotDir)

        resetFakeHF()
        FakeHFURLProtocol.files = try remoteFiles(
            snapshotDir: snapshotDir,
            snap: snapshot,
            includeRequiredTokenizer: true,
            includeOptionalTokenizer: true)
        let result = try await RemoteStreamingRepacker(options: remoteOptions(
            outputDir: output,
            session: fakeHFSession())).run()

        #expect(result.plan.arch.family == .gptOss)
        #expect(result.plan.baseMode == "mxfp4")
        #expect(result.plan.layers.count == 2)
        #expect(result.downloadedThisRunBytes == result.remoteBytesToDownload)
        for relativePath in [
            "model_weights.bin",
            "packed_experts/layout.json",
            "packed_experts/layer_00.bin",
            "packed_experts/layer_01.bin",
            "manifest.json",
        ] {
            #expect(FileManager.default.fileExists(atPath:
                (output as NSString).appendingPathComponent(relativePath)))
        }

        let manifestData = try Data(contentsOf: URL(fileURLWithPath:
            (output as NSString).appendingPathComponent("manifest.json")))
        let manifest = try JSONSerialization.jsonObject(with: manifestData)
            as! [String: Any]
        let flags = manifest["flags"] as! [String: Bool]
        let arch = manifest["arch"] as! [String: Any]
        let quant = manifest["quant"] as! [String: [String: Any]]
        #expect(manifest["versionMinor"] as? Int == 1)
        #expect(flags["mxfp4Weights"] == true)
        #expect(arch["family"] as? String == "gpt-oss")
        #expect(arch["variant"] as? String == "gpt-oss-20b")
        #expect(quant["embedding"]?["scheme"] as? String == "bf16")
        #expect(quant["routedExpert"]?["scheme"] as? String == "mxfp4")

        let verification = try VerifiedInstallTool.run(
            options: VerifyInstallOptions(inputGTurbo: output))
        #expect(verification.unexpectedEntries.isEmpty)
        #expect(verification.bytesVerified > 0)
    }
}
