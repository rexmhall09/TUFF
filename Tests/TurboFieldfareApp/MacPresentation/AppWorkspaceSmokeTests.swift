import AppKit
import Testing
import TUFFModelCatalog
import TurboFieldfareAppCore
import TurboFieldfareAppServer
import TurboFieldfareAppUpdater
@testable import TurboFieldfareMac
import TurboFieldfareMacPresentation
import SwiftUI

@Suite(.serialized) @MainActor struct AppWorkspaceSmokeTests {
    @Test func everyDestinationRendersAtTheMinimumWindowSize() throws {
        let client = WorkspaceSmokeLifecycleClient()
        let broker = SharedInferenceBroker(client: client)
        let model = AppModel(
            modelDirectory: FileManager.default.temporaryDirectory
                .appendingPathComponent(
                    "TUFFWorkspaceSmoke-\(UUID().uuidString).gturbo",
                    isDirectory: true),
            client: broker,
            otherInstalls: [],
            deviceCapabilities: TUFFDeviceCapabilities(
                unifiedMemoryBytes: 16 * 1_024 * 1_024 * 1_024,
                macOSMajorVersion: 15,
                appleSiliconGeneration: 2))
        let serverController = AppServerController(
            broker: broker, store: model.serverStore)
        let updateController = AppUpdateController(infoDictionary: nil)
        for destination in AppDestination.allCases {
            let content = AppWorkspaceView(
                destination: destination,
                model: model,
                serverController: serverController,
                updateController: updateController)
                .frame(
                    width: AppWindowLayout.detailMinimumWidth,
                    height: AppWindowLayout.minimumHeight)
                .transaction { $0.disablesAnimations = true }
            let renderer = ImageRenderer(content: content)
            renderer.scale = 1
            let image = try #require(renderer.nsImage)
            let data = try #require(image.tiffRepresentation)

            #expect(image.size == NSSize(
                width: AppWindowLayout.detailMinimumWidth,
                height: AppWindowLayout.minimumHeight))
            #expect(!data.isEmpty)
        }
    }
}

private final class WorkspaceSmokeLifecycleClient: AppModelLifecycleClient,
    @unchecked Sendable {
    func generate(
        _ request: AppGenerationRequest
    ) -> AsyncThrowingStream<AppInferenceEvent, Error> {
        AsyncThrowingStream { $0.finish() }
    }

    func cancel() {}

    func ensureLoaded(
        modelDirectory: URL,
        maxContextTokens: Int,
        options: AppRuntimeOptions,
        forceLogitsHead: Bool,
        onState: @escaping @Sendable (AppModelLoadState) -> Void
    ) async throws {}

    func unload() async {}
}
