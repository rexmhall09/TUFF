import Foundation
import Testing
@testable import TUFFAppCore

/// What the chat's model picker is allowed to offer.
///
/// It used to list the whole catalogue. Choosing a model that was not on the
/// Mac replaced the conversation with the downloader — the chat vanished behind
/// an install screen because of a menu that reads like a preference. Downloads
/// live on the Models screen; the picker switches between what is already here.
@Suite struct AppChatModelPickerTests {
    private func missingDirectory(_ tag: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("\(tag)-\(UUID().uuidString).gturbo",
                                    isDirectory: true)
    }

    @MainActor
    @Test func onlyDownloadedModelsAreOffered() throws {
        let installed = try makeVisionReadyModelInstall("picker-installed")
        defer { try? FileManager.default.removeItem(at: installed) }
        let absent = ModelInstallCoordinator(
            descriptor: .qwen36,
            directoryURL: missingDirectory("picker-absent"),
            client: MockModelInstallerClient(descriptor: .qwen36))
        let model = AppModel(modelDirectory: installed, otherInstalls: [absent])

        #expect(model.installs.count == 2)
        #expect(model.selectedInstall.isInstalled)
        #expect(!absent.isInstalled)
        #expect(model.installedInstalls.map(\.id) == [model.selectedModelID],
                "the picker was offering a model that is not on this Mac")
    }

    /// A picker whose selection is not in its own list has no valid tag, and
    /// SwiftUI draws it blank. The selected model is therefore always present,
    /// downloaded or not — which is the state the app starts in before the
    /// first download finishes.
    @MainActor
    @Test func theSelectedModelIsOfferedEvenBeforeItIsDownloaded() {
        let absent = ModelInstallCoordinator(
            descriptor: .qwen36,
            directoryURL: missingDirectory("picker-other"),
            client: MockModelInstallerClient(descriptor: .qwen36))
        let model = AppModel(
            modelDirectory: missingDirectory("picker-selected"),
            otherInstalls: [absent])

        #expect(!model.selectedInstall.isInstalled)
        #expect(model.installedInstalls.map(\.id) == [model.selectedModelID])
    }
}
