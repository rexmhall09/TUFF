import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppStoresTests {
    @MainActor
    @Test func appModelProxiesConversationAndSettingsState() {
        let model = AppModel(installer: MockModelInstallerClient(descriptor: .default))

        model.promptText = "draft"
        model.temperature = 0.45

        #expect(model.conversationStore.promptText == "draft")
        #expect(model.settingsStore.temperature == 0.45)

        model.conversationStore.outputText = "answer"
        model.settingsStore.topK = 32

        #expect(model.outputText == "answer")
        #expect(model.topK == 32)
    }

    @MainActor
    @Test func appModelProxiesLibraryAndInferenceState() {
        let model = AppModel(installer: MockModelInstallerClient(descriptor: .default))

        model.modelLibraryStore.modelPathText = "/tmp/example.gturbo"
        model.inferenceStore.liveTokenCount = 12
        model.inferenceStore.runState = .running

        #expect(model.modelPathText == "/tmp/example.gturbo")
        #expect(model.liveTokenCount == 12)
        #expect(model.isRunning)
    }

    @MainActor
    @Test func storeInstancesDoNotShareMutableState() {
        let first = AppConversationStore()
        let second = AppConversationStore()

        first.promptText = "only the first store"

        #expect(second.promptText.isEmpty)
        #expect(AppServerStore().status == .stopped)
    }

    @MainActor
    @Test func serverStoreBoundsRecentErrorsNewestFirst() {
        let store = AppServerStore()
        for index in 0...AppServerStore.maximumRecentErrors {
            store.recordError("error \(index)")
        }

        #expect(store.recentErrors.count == AppServerStore.maximumRecentErrors)
        #expect(store.recentErrors.first == "error \(AppServerStore.maximumRecentErrors)")
        #expect(store.recentErrors.last == "error 1")
    }
}
