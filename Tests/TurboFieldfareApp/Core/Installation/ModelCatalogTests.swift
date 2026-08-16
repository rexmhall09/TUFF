import Foundation
import Testing
import TurboFieldfareRepackCore

@testable import TurboFieldfareAppCore

/// The catalog's contract: every model has its own install, and starting,
/// cancelling, or finishing one never touches another. Selection is a separate
/// concern from downloading — picking a model decides what loads, and must not
/// start or stop any transfer.
@Suite struct ModelCatalogTests {

    private func temporaryInstallPath(_ tag: String) -> URL {
        FileManager.default.temporaryDirectory
            .appendingPathComponent(
                "turbofieldfare-catalog-\(tag)-\(UUID().uuidString).gturbo")
    }

    @MainActor
    private func makeModel(
        selectedDirectory: URL,
        selectedInstaller: MockModelInstallerClient,
        otherDirectory: URL,
        otherInstaller: MockModelInstallerClient
    ) -> AppModel {
        let other = ModelInstallCoordinator(
            descriptor: otherInstaller.descriptor,
            directoryURL: otherDirectory,
            client: otherInstaller)
        return AppModel(
            modelDirectory: selectedDirectory,
            client: MockLifecycleInferenceClient(),
            installer: selectedInstaller,
            otherInstalls: [other])
    }

    @MainActor
    @Test func catalogListsEveryModelAndSelectsTheInjectedOne() {
        let model = makeModel(
            selectedDirectory: temporaryInstallPath("catalog-gemma"),
            selectedInstaller: MockModelInstallerClient(descriptor: .default),
            otherDirectory: temporaryInstallPath("catalog-qwen"),
            otherInstaller: MockModelInstallerClient(descriptor: .qwen36))

        #expect(model.installs.count == 2)
        #expect(model.selectedModelID == AppModelInstallDescriptor.default.id)
        #expect(model.selectedDescriptor == .default)
        #expect(model.installs.map(\.descriptor) == [.default, .qwen36])
    }

    @MainActor
    @Test func downloadingOneModelLeavesTheOtherAlone() async {
        let selected = MockModelInstallerClient(descriptor: .default, holdOpen: true)
        let other = MockModelInstallerClient(descriptor: .qwen36, holdOpen: true)
        let model = makeModel(
            selectedDirectory: temporaryInstallPath("independent-gemma"),
            selectedInstaller: selected,
            otherDirectory: temporaryInstallPath("independent-qwen"),
            otherInstaller: other)
        let qwen = try! #require(model.installs.first { $0.descriptor == .qwen36 })

        qwen.install()
        await Task.yield()

        #expect(qwen.isInstalling)
        #expect(model.isInstallingAnyModel)
        // The selected model is untouched: its own state, and the app-level
        // single-model properties that report it, stay idle.
        #expect(!model.selectedInstall.isInstalling)
        #expect(model.installState == .idle)
        #expect(!model.isInstallingModel)
        #expect(!selected.cancelCalled)
    }

    @MainActor
    @Test func bothModelsCanDownloadAtOnce() async {
        let selected = MockModelInstallerClient(descriptor: .default, holdOpen: true)
        let other = MockModelInstallerClient(descriptor: .qwen36, holdOpen: true)
        let model = makeModel(
            selectedDirectory: temporaryInstallPath("both-gemma"),
            selectedInstaller: selected,
            otherDirectory: temporaryInstallPath("both-qwen"),
            otherInstaller: other)
        let qwen = try! #require(model.installs.first { $0.descriptor == .qwen36 })

        model.installModel()
        qwen.install()
        await Task.yield()

        #expect(model.selectedInstall.isInstalling)
        #expect(qwen.isInstalling)
    }

    @MainActor
    @Test func cancellingOneDownloadLeavesTheOtherRunning() async {
        let selected = MockModelInstallerClient(descriptor: .default, holdOpen: true)
        let other = MockModelInstallerClient(descriptor: .qwen36, holdOpen: true)
        let model = makeModel(
            selectedDirectory: temporaryInstallPath("cancel-gemma"),
            selectedInstaller: selected,
            otherDirectory: temporaryInstallPath("cancel-qwen"),
            otherInstaller: other)
        let qwen = try! #require(model.installs.first { $0.descriptor == .qwen36 })

        model.installModel()
        qwen.install()
        await Task.yield()
        qwen.cancel()
        // `.cancelling` is still an installing state; wait for it to settle.
        try? await waitUntil { !qwen.isInstalling }

        // `cancelCalled` is not the signal here: starting an install calls
        // cancel on its own client first, to clear any earlier attempt. What
        // matters is that only the cancelled model left the installing state.
        #expect(qwen.state == .cancelled)
        #expect(model.selectedInstall.isInstalling)
        #expect(model.isInstallingModel)
    }

    @MainActor
    @Test func selectingAnotherModelSwitchesTheActiveDirectory() {
        let qwenDirectory = temporaryInstallPath("select-qwen")
        let model = makeModel(
            selectedDirectory: temporaryInstallPath("select-gemma"),
            selectedInstaller: MockModelInstallerClient(descriptor: .default),
            otherDirectory: qwenDirectory,
            otherInstaller: MockModelInstallerClient(descriptor: .qwen36))
        let qwen = try! #require(model.installs.first { $0.descriptor == .qwen36 })

        #expect(model.canSelectModel(qwen))
        model.selectModel(qwen)

        #expect(model.selectedModelID == AppModelInstallDescriptor.qwen36.id)
        #expect(model.modelPathText == qwenDirectory.standardizedFileURL.path)
        #expect(model.installDescriptor == .qwen36)
        // Selecting the model already selected is a no-op, not a reset.
        #expect(!model.canSelectModel(qwen))
    }

    @MainActor
    @Test func selectingAModelDoesNotDisturbAnyDownload() async {
        let selected = MockModelInstallerClient(descriptor: .default, holdOpen: true)
        let other = MockModelInstallerClient(descriptor: .qwen36, holdOpen: true)
        let model = makeModel(
            selectedDirectory: temporaryInstallPath("switch-gemma"),
            selectedInstaller: selected,
            otherDirectory: temporaryInstallPath("switch-qwen"),
            otherInstaller: other)
        let qwen = try! #require(model.installs.first { $0.descriptor == .qwen36 })

        model.installModel()
        qwen.install()
        await Task.yield()

        model.selectModel(qwen)
        await Task.yield()

        #expect(model.installs.allSatisfy { $0.isInstalling })
        #expect(model.selectedModelID == AppModelInstallDescriptor.qwen36.id)
    }

    @MainActor
    @Test func loadedModelBlocksSelectionWhileGenerating() {
        let model = makeModel(
            selectedDirectory: temporaryInstallPath("busy-gemma"),
            selectedInstaller: MockModelInstallerClient(descriptor: .default),
            otherDirectory: temporaryInstallPath("busy-qwen"),
            otherInstaller: MockModelInstallerClient(descriptor: .qwen36))
        let qwen = try! #require(model.installs.first { $0.descriptor == .qwen36 })

        model.runState = .running

        #expect(!model.canSelectModel(qwen))
        model.selectModel(qwen)
        #expect(model.selectedModelID == AppModelInstallDescriptor.default.id)
    }

    @MainActor
    @Test func aFinishedDownloadSelectsItsModelWhenNothingIsInstalled() async throws {
        let qwenDirectory = temporaryInstallPath("finish-qwen")
        let selected = MockModelInstallerClient(descriptor: .default)
        let installed = try makeCompleteModelInstall(
            "finish-selects", stampedAs: .qwen36)
        defer { try? FileManager.default.removeItem(at: installed) }
        let other = MockModelInstallerClient(
            events: [.installed(installed)],
            descriptor: .qwen36)
        let coordinator = ModelInstallCoordinator(
            descriptor: .qwen36,
            directoryURL: qwenDirectory,
            client: other)
        let model = AppModel(
            modelDirectory: temporaryInstallPath("finish-gemma"),
            client: MockLifecycleInferenceClient(),
            installer: selected,
            otherInstalls: [coordinator])

        #expect(!model.isModelInstalled)
        coordinator.install()
        try await waitUntil { coordinator.isInstalled }

        #expect(model.selectedModelID == AppModelInstallDescriptor.qwen36.id)
        #expect(model.isModelInstalled)
        #expect(model.modelPathText == installed.standardizedFileURL.path)
    }
}

/// Poll the main actor until `condition` holds or the budget runs out.
@MainActor
private func waitUntil(_ condition: () -> Bool) async throws {
    for _ in 0..<200 {
        if condition() { return }
        try await Task.sleep(for: .milliseconds(5))
    }
    Issue.record("condition did not become true")
}
