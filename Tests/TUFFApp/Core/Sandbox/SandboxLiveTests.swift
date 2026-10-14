#if os(macOS)
import Foundation
import Testing
@testable import TUFFAppCore

/// Starts and stops the real sandbox with Apple's `container`. It only runs
/// when `TUFF_SANDBOX_LIVE_IMAGE` names an image already in the local store,
/// such as the one `Scripts/sandbox_selftest.sh build` makes (`tuff-sandbox:dev`):
///
///     TUFF_SANDBOX_LIVE_IMAGE=tuff-sandbox:dev Scripts/test.sh --filter SandboxLiveTests
///
/// The image is trusted by name, so use this only on a developer's Mac. With
/// `TUFF_SANDBOX_LIVE_COLD=1` the test also runs `container system stop`
/// first, to time a start from a cold system service (this stops every
/// container, so do not use it while anything else runs).
@MainActor
@Suite(.serialized) struct SandboxLiveTests {
    nonisolated static let image = ProcessInfo.processInfo.environment["TUFF_SANDBOX_LIVE_IMAGE"]
    nonisolated static let cold = ProcessInfo.processInfo.environment["TUFF_SANDBOX_LIVE_COLD"] == "1"

    @Test(.enabled(if: SandboxLiveTests.image != nil))
    func startsAndStopsTheRealSandbox() async throws {
        let reference = try #require(Self.image)
        let tool = ContainerTool()
        try await tool.verify()
        // The image's digest comes from the local store, read while the
        // service is up (with COLD it is stopped right after); the controller
        // then runs exactly that image and checks it again.
        let inspected = try #require(await tool.inspectImage(reference), "image \(reference) is not in the local store")
        let digest = try #require(SandboxImageProvisioner.indexDigest(in: inspected))
        if Self.cold {
            _ = try await FoundationSandboxProcessRunner().run(
                executable: ContainerTool.installedPath, arguments: ["system", "stop"], timeout: 120)
        }
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff sandbox live \(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        var dependencies = SandboxDependencies.live()
        dependencies.pin = SandboxImagePin(reference: reference, digest: digest, source: .local)
        dependencies.stateDirectory = directory
        let controller = SandboxController(dependencies: dependencies)

        let began = Date()
        await controller.start()
        let seconds = Date().timeIntervalSince(began)
        print("SANDBOX LIVE: start took \(String(format: "%.1f", seconds)) s (cold system service: \(Self.cold)); state \(controller.state)")
        #expect(controller.state == .ready, "\(controller.state)")
        guard controller.state == .ready else {
            await controller.stop()
            return
        }

        // The real signed health check, from outside the VM.
        let lease = try controller.acquire()
        let health = try await SandboxHealthProbe().health(port: lease.port, key: lease.key)
        #expect(health.profile == "web")
        #expect(health.blockedAddresses > 0)
        // An answer without this start's key must not pass.
        await #expect(throws: SandboxError.self) {
            _ = try await SandboxHealthProbe().health(port: lease.port, key: Data("wrong".utf8))
        }
        controller.release(lease)

        let ids = try await tool.containerIDs(all: true)
        #expect(ids.contains { $0.hasPrefix(SandboxController.containerNamePrefix) })

        let stopBegan = Date()
        await controller.stop()
        print("SANDBOX LIVE: stop took \(String(format: "%.1f", Date().timeIntervalSince(stopBegan))) s")
        #expect(controller.state == .off)
        let after = try await tool.containerIDs(all: true)
        #expect(!after.contains { $0.hasPrefix(SandboxController.containerNamePrefix) })
        #expect(!FileManager.default.fileExists(
            atPath: SandboxLocations.secretsDirectory(in: directory).path)
            || ((try? FileManager.default.contentsOfDirectory(
                atPath: SandboxLocations.secretsDirectory(in: directory).path))?.isEmpty ?? true))
    }
}
#endif
