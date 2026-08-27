import Foundation
import Testing
@testable import TurboFieldfareAppCore

@Suite struct AppModelLoadStateTests {
    @MainActor
    @Test func runRequiresReadyStateButSubmissionCanLoadAnInstalledModel() throws {
        let directory = try makeCompleteModelInstall("submit-eligibility")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel()
        model.setModelURL(directory)
        model.promptText = "go"

        model.loadState = .notLoaded
        #expect(!model.canRun)
        #expect(model.canSubmit)
        model.loadState = .loading(.verifyingWeights)
        #expect(!model.canRun)
        #expect(!model.canSubmit)
        model.loadState = .failed(.modelLoadFailed("boom"))
        #expect(!model.canRun)
        #expect(model.canSubmit)
        model.loadState = .ready(modelDirectory: URL(fileURLWithPath: "/tmp/m.gturbo"), loadSeconds: 1.2)
        #expect(model.canRun)
        #expect(model.canSubmit)
    }

    @MainActor
    @Test func submitLoadsThenRunsExactlyOnce() async throws {
        let directory = try makeCompleteModelInstall("load-then-submit")
        defer { try? FileManager.default.removeItem(at: directory) }
        let client = MockLifecycleInferenceClient()
        client.suspendLoads = true
        let model = AppModel(modelDirectory: directory, client: client)
        model.promptText = "Explain local inference"

        model.submit()
        await client.waitForLoadStart()
        #expect(model.isLoadingForSubmission)
        #expect(!model.canSubmit)
        #expect(client.generationCallCount() == 0)

        model.submit()
        #expect(client.ensureLoadedCallCount() == 1)
        client.releaseLoads()
        for _ in 0..<200 where client.generationCallCount() == 0 {
            try? await Task.sleep(for: .milliseconds(5))
        }

        #expect(client.generationCallCount() == 1)
        #expect(client.lastGenerationRequest()?.prompt == "Explain local inference")
        #expect(!model.isLoadingForSubmission)
    }

    @MainActor
    @Test func failedLoadKeepsDraftAndNeverRunsQueuedSubmission() async throws {
        let directory = try makeCompleteModelInstall("failed-load-submit")
        defer { try? FileManager.default.removeItem(at: directory) }
        let client = MockLifecycleInferenceClient()
        client.suspendLoads = true
        let model = AppModel(modelDirectory: directory, client: client)
        model.promptText = "Keep this draft"

        model.submit()
        await client.waitForLoadStart()
        client.failNextLoad(.modelLoadFailed("synthetic"))
        for _ in 0..<200 where !model.loadState.isFailed {
            try? await Task.sleep(for: .milliseconds(5))
        }

        #expect(client.generationCallCount() == 0)
        #expect(model.promptText == "Keep this draft")
        #expect(!model.isLoadingForSubmission)
        #expect(model.canSubmit)
    }

    @MainActor
    @Test func canLoadModelOnlyFromNotLoadedOrFailed() throws {
        let directory = try makeCompleteModelInstall("can-load")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory)
        model.loadState = .notLoaded
        #expect(model.canLoadModel)
        model.loadState = .failed(.modelLoadFailed("boom"))
        #expect(model.canLoadModel)
        model.loadState = .loading(.tokenizer)
        #expect(!model.canLoadModel)
        model.loadState = .ready(modelDirectory: URL(fileURLWithPath: "/tmp/m.gturbo"), loadSeconds: 1.2)
        #expect(!model.canLoadModel)
    }

    @MainActor
    @Test func loadingStateBlocksRun() {
        let model = AppModel()
        model.promptText = "go"
        model.loadState = .loading(.preparingRunner)
        #expect(!model.canRun)
        #expect(model.loadState.isLoading)
    }

    @MainActor
    @Test func explicitLoadReloadAndUnloadActionsAreMutuallyExclusive() throws {
        let directory = try makeCompleteModelInstall("action-matrix")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory,
                             client: MockLifecycleInferenceClient())
        #expect(model.canLoadModel)
        #expect(!model.canReloadModel)
        #expect(!model.canUnloadModel)

        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))
        #expect(!model.canLoadModel)
        #expect(model.canUnloadModel)

        model.runtimeOptions.rdadvisePolicy = .bounded
        #expect(model.canReloadModel)
        #expect(model.canUnloadModel)
        #expect(!model.canRun)
    }

    @MainActor
    @Test func unloadWaitsForLifecycleAndPreservesTranscript() async throws {
        let directory = try makeCompleteModelInstall("unload")
        defer { try? FileManager.default.removeItem(at: directory) }
        let client = MockLifecycleInferenceClient()
        client.suspendUnloads = true
        let model = AppModel(modelDirectory: directory, client: client)
        model.outputText = "keep me"
        model.promptText = "keep this draft"
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))

        model.unloadModel()
        await client.waitForUnloadStart()
        #expect(model.loadState == .unloading)
        client.releaseUnloads()
        for _ in 0..<200 where model.loadState != .notLoaded {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
        #expect(model.loadState == .notLoaded)
        #expect(model.outputText == "keep me")
        #expect(model.promptText == "keep this draft")
    }

    @MainActor
    @Test func reloadPreservesTranscriptAndDraft() async throws {
        let directory = try makeCompleteModelInstall("reload-preserves-conversation")
        defer { try? FileManager.default.removeItem(at: directory) }
        let client = MockLifecycleInferenceClient()
        let model = AppModel(modelDirectory: directory, client: client)
        model.outputText = "keep me"
        model.promptText = "keep this draft"
        model.applyLoadState(.ready(modelDirectory: directory, loadSeconds: 0))
        model.runtimeOptions.rdadvisePolicy = .bounded

        model.reloadModel()
        for _ in 0..<200 where !model.loadState.isReady {
            try? await Task.sleep(for: .milliseconds(5))
        }

        #expect(model.loadState.isReady)
        #expect(model.outputText == "keep me")
        #expect(model.promptText == "keep this draft")
    }

    @MainActor
    @Test func cancelLoadWaitsForUnloadAndRejectsLateReady() async throws {
        let directory = try makeCompleteModelInstall("cancel-load")
        defer { try? FileManager.default.removeItem(at: directory) }
        let client = MockLifecycleInferenceClient()
        client.suspendLoads = true
        client.suspendUnloads = true
        let model = AppModel(modelDirectory: directory, client: client)
        model.outputText = "keep me"
        model.promptText = "keep this draft"

        model.loadModel()
        await client.waitForLoadStart()
        for _ in 0..<200 where !model.canCancelLoad {
            try? await Task.sleep(for: .milliseconds(5))
        }
        model.cancelLoad()
        await client.waitForUnloadStart()
        #expect(model.loadState == .cancelling)

        client.emitLoadState(.ready(modelDirectory: directory, loadSeconds: 0), callIndex: 0)
        await Task.yield()
        #expect(model.loadState == .cancelling)

        client.releaseUnloads()
        for _ in 0..<200 where model.loadState != .notLoaded {
            try? await Task.sleep(for: .milliseconds(5))
        }
        #expect(model.loadState == .notLoaded)
        #expect(model.outputText == "keep me")
        #expect(model.promptText == "keep this draft")
    }

    @MainActor
    @Test func failedLoadIsRetryableAndOldAttemptCannotCompleteRetry() async throws {
        let directory = try makeCompleteModelInstall("retry-load")
        defer { try? FileManager.default.removeItem(at: directory) }
        let client = MockLifecycleInferenceClient()
        client.suspendLoads = true
        let model = AppModel(modelDirectory: directory, client: client)
        model.outputText = "keep me"
        model.promptText = "keep this draft"

        model.loadModel()
        await client.waitForLoadStart()
        client.failNextLoad(.modelLoadFailed("synthetic"))
        for _ in 0..<200 where !model.loadState.isFailed {
            try? await Task.sleep(for: .milliseconds(5))
        }
        #expect(model.loadState == .failed(.modelLoadFailed("synthetic")))
        #expect(model.canLoadModel)
        #expect(model.outputText == "keep me")
        #expect(model.promptText == "keep this draft")

        client.suspendLoads = true
        model.loadModel()
        await client.waitForLoadStart(2)
        client.emitLoadState(.ready(modelDirectory: directory, loadSeconds: 0), callIndex: 0)
        await Task.yield()
        #expect(model.loadState.isLoading)

        client.releaseLoads()
        for _ in 0..<200 where !model.loadState.isReady {
            try? await Task.sleep(for: .milliseconds(5))
        }
        #expect(model.loadState.isReady)
        #expect(client.ensureLoadedCallCount() == 2)
        #expect(model.outputText == "keep me")
        #expect(model.promptText == "keep this draft")
    }

    @MainActor
    @Test func loadModelWithoutLifecycleClientFails() throws {
        let directory = try makeCompleteModelInstall("no-lifecycle")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory, client: MockInferenceClient())
        model.loadModel()
        #expect(model.loadState.isFailed)
    }

    @MainActor
    @Test func applyLoadStateUpdatesPresentation() throws {
        let directory = try makeCompleteModelInstall("load-presentation")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(modelDirectory: directory)
        model.applyLoadState(.loading(.verifyingWeights))
        #expect(model.presentation.label == AppModelLoadPhase.verifyingWeights.label)
        model.applyLoadState(.failed(.modelLoadFailed("boom")))
        #expect(model.presentation.detail == "Model load failed: boom")
    }

    @MainActor
    @Test func loadModelWaitsForPendingUnloadAfterModelPathChange() async throws {
        let client = MockLifecycleInferenceClient()
        client.suspendUnloads = true
        let oldURL = try makeCompleteModelInstall("old-load")
        let newURL = try makeCompleteModelInstall("new-load")
        defer {
            try? FileManager.default.removeItem(at: oldURL)
            try? FileManager.default.removeItem(at: newURL)
        }
        let model = AppModel(modelDirectory: oldURL, client: client)
        model.loadState = .ready(modelDirectory: oldURL, loadSeconds: 1)

        model.setModelURL(newURL)
        await client.waitForUnloadStart()
        model.loadModel()

        #expect(client.ensureLoadedCallCount() == 0)
        client.releaseUnloads()
        await client.waitForEnsureLoadedCallCount(1)
        #expect(client.ensureLoadedCallCount() == 1)
    }

    @MainActor
    @Test func staleReadyStateForOldModelPathIsIgnored() {
        let model = AppModel(client: MockInferenceClient())
        let oldURL = URL(fileURLWithPath: "/tmp/old.gturbo")
        let newURL = URL(fileURLWithPath: "/tmp/new.gturbo")
        model.modelPathText = oldURL.path

        model.setModelURL(newURL)
        model.applyLoadState(.ready(modelDirectory: oldURL, loadSeconds: 0))

        #expect(model.loadState == .notLoaded)
        #expect(model.loadedRuntimeKey == nil)
    }
}

@Suite("Blocked install recovery")
@MainActor
struct BlockedInstallRecoveryTests {
    @Test("a blocked destination offers recovery instead of a download")
    func offersRecovery() async throws {
        let client = MockModelInstallerClient()
        let coordinator = ModelInstallCoordinator(
            descriptor: .default,
            directoryURL: URL(fileURLWithPath: "/tmp/support/TUFF/Models/qwen36.gturbo"),
            client: client)
        client.setBlockedEntries(["/tmp/support/TUFF/Models/qwen36.gturbo"])

        #expect(coordinator.isInstallPathBlocked)
        #expect(coordinator.canClearInstallPath)
        #expect(coordinator.blockedInstallEntries.count == 1)
    }

    @Test("clearing the path unblocks the install")
    func clearingUnblocks() async throws {
        let client = MockModelInstallerClient()
        let coordinator = ModelInstallCoordinator(
            descriptor: .default,
            directoryURL: URL(fileURLWithPath: "/tmp/support/TUFF/Models/qwen36.gturbo"),
            client: client)
        client.setBlockedEntries(["/tmp/support/TUFF/Models/qwen36.gturbo"])

        coordinator.clearBlockedInstallPath()
        try await Task.sleep(nanoseconds: 200_000_000)

        #expect(client.clearBlockedCallCount == 1)
        #expect(!coordinator.isInstallPathBlocked)
        #expect(!coordinator.canClearInstallPath)
    }

    @Test("a clean destination offers no recovery button")
    func noRecoveryWhenClean() {
        let client = MockModelInstallerClient()
        let coordinator = ModelInstallCoordinator(
            descriptor: .default,
            directoryURL: URL(fileURLWithPath: "/tmp/support/TUFF/Models/qwen36.gturbo"),
            client: client)
        #expect(!coordinator.isInstallPathBlocked)
        #expect(!coordinator.canClearInstallPath)
    }
}
