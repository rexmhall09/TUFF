import Foundation
import Testing
import TurboFieldfare
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

    @Test func appCatalogKeepsStableRegistryIdentities() {
        #expect(AppModelInstallDescriptor.catalog.map(\.settingsProfileKey) == [
            "gemma4-e4b",
            "gemma4-26b-a4b",
            "qwen36-35b-a3b",
        ])
        #expect(AppModelInstallDescriptor.catalog.map(\.installDirectoryName) == [
            "gemma4-e4b.gturbo",
            "gemma4.gturbo",
            "qwen36.gturbo",
        ])
        #expect(AppModelInstallDescriptor.gemma4E4B.supportsImageInput == false)
        #expect(MacModelSettings.defaults(
            for: AppModelInstallDescriptor.gemma4E4B.settingsProfileKey).isValid())
        #expect(AppModelInstallDescriptor.default.supportsImageInput)
        #expect(AppModelInstallDescriptor.descriptor(
            for: ModelVariant.gemma4_E4B) == .gemma4E4B)
        #expect(AppModelInstallDescriptor.descriptor(
            for: ModelVariant.qwen36_35B_A3B) == .qwen36)
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
    @Test func installedQwenOffersItsOwnOptionalVisionPack() throws {
        let directory = try makeCompleteModelInstall(
            "vision-qwen", stampedAs: .qwen36)
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = AppModel(
            modelDirectory: directory,
            client: MockLifecycleInferenceClient(),
            installer: MockModelInstallerClient(descriptor: .qwen36),
            visionInstaller: MockVisionPackInstallerClient(
                descriptor: .qwen36VisionCompanion))

        #expect(model.selectedDescriptor == .qwen36)
        #expect(model.visionInstallDescriptor == .qwen36VisionCompanion)
        #expect(!model.isImageInputAvailable)
        #expect(model.canInstallVisionPack)
        #expect(model.visionInstallReadiness.requirement?.canInstall == true)

        // Preparing the optional companion is a separate download. It may run
        // while the text model stays loaded; only activation needs an unload.
        model.loadState = .ready(modelDirectory: directory, loadSeconds: 0)
        #expect(model.canInstallVisionPack)
        model.visionInstallState = .readyToActivate(directory)
        #expect(!model.canActivateVisionPack)
        model.loadState = .notLoaded
        #expect(model.canActivateVisionPack)
    }

    @MainActor
    @Test func catalogOffersSeparateVisionDownloadsWithoutSwitchingModels() async throws {
        let gemmaDirectory = try makeCompleteModelInstall(
            "separate-vision-gemma", stampedAs: .default)
        let qwenDirectory = try makeCompleteModelInstall(
            "separate-vision-qwen", stampedAs: .qwen36)
        defer {
            try? FileManager.default.removeItem(at: gemmaDirectory)
            try? FileManager.default.removeItem(at: qwenDirectory)
        }
        let qwenInstall = ModelInstallCoordinator(
            descriptor: .qwen36,
            directoryURL: qwenDirectory,
            client: MockModelInstallerClient(descriptor: .qwen36))
        let visionInstaller = MockVisionPackInstallerClient(holdOpen: true)
        let model = AppModel(
            modelDirectory: gemmaDirectory,
            client: MockLifecycleInferenceClient(),
            installer: MockModelInstallerClient(descriptor: .default),
            otherInstalls: [qwenInstall],
            visionInstaller: visionInstaller)
        let gemma = try #require(model.installs.first { $0.descriptor == .default })
        let qwen = try #require(model.installs.first { $0.descriptor == .qwen36 })

        #expect(model.visionDownloadButtonLabel(for: gemma)
            == "Download Gemma 4 Image Support")
        #expect(model.visionDownloadButtonLabel(for: qwen)
            == "Download Qwen3.6 Image Support")
        #expect(model.canInstallVisionPack(for: gemma))
        #expect(model.canInstallVisionPack(for: qwen))

        model.installVisionPack(for: qwen)
        await Task.yield()

        #expect(model.selectedModelID == gemma.id)
        #expect(model.visionInstallTargetModelID == qwen.id)
        #expect(!model.selectedModelOwnsVisionInstallState)
        #expect(visionInstaller.installedTextModelDirectories == [
            qwenDirectory.standardizedFileURL,
        ])
        #expect(!model.canSelectModel(qwen))

        model.cancelVisionInstall()
        try await waitUntil { !model.isInstallingVisionPack }
    }

    @MainActor
    @Test func preparedVisionDownloadFollowsOnlyItsMatchingModel() async throws {
        let gemmaDirectory = try makeCompleteModelInstall(
            "prepared-vision-gemma", stampedAs: .default)
        let qwenDirectory = try makeCompleteModelInstall(
            "prepared-vision-qwen", stampedAs: .qwen36)
        defer {
            try? FileManager.default.removeItem(at: gemmaDirectory)
            try? FileManager.default.removeItem(at: qwenDirectory)
        }
        let qwenInstall = ModelInstallCoordinator(
            descriptor: .qwen36,
            directoryURL: qwenDirectory,
            client: MockModelInstallerClient(descriptor: .qwen36))
        let preparedDirectory = qwenDirectory.deletingPathExtension()
            .appendingPathExtension("vision.gturbo")
        let model = AppModel(
            modelDirectory: gemmaDirectory,
            client: MockLifecycleInferenceClient(),
            installer: MockModelInstallerClient(descriptor: .default),
            otherInstalls: [qwenInstall],
            visionInstaller: MockVisionPackInstallerClient(events: [
                .readyToActivate(preparedDirectory),
            ]))
        let qwen = try #require(model.installs.first { $0.descriptor == .qwen36 })

        model.installVisionPack(for: qwen)
        try await waitUntil {
            if case .readyToActivate = model.visionInstallState { return true }
            return false
        }

        #expect(model.selectedDescriptor == .default)
        #expect(model.canSelectModel(qwen))
        model.selectModel(qwen)

        #expect(model.selectedDescriptor == .qwen36)
        #expect(model.modelPathText == qwenDirectory.standardizedFileURL.path)
        #expect(model.visionInstallTargetModelID == qwen.id)
        #expect(model.selectedModelOwnsVisionInstallState)
        #expect(model.canActivateVisionPack)
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

    /// Free space is shared, so a running download has to shrink what the other
    /// model believes it can use. Without this, both models pass their own
    /// check on a disk that fits only one, and the second fails deep into a
    /// multi-gigabyte transfer instead of before it starts.
    @MainActor
    @Test func aRunningDownloadReservesSpaceAgainstTheOtherModel() async {
        // Room for one model but not both.
        let diskBytes = AppModelInstallDescriptor.default.requiredFreeBytes
            + AppModelInstallDescriptor.qwen36.installedBytes / 2
        func installer(_ descriptor: AppModelInstallDescriptor)
            -> MockModelInstallerClient {
            MockModelInstallerClient(
                requirement: AppModelInstallRequirement(
                    requiredBytes: descriptor.requiredFreeBytes,
                    availableBytes: diskBytes),
                descriptor: descriptor,
                holdOpen: true)
        }
        let model = makeModel(
            selectedDirectory: temporaryInstallPath("reserve-gemma"),
            selectedInstaller: installer(.default),
            otherDirectory: temporaryInstallPath("reserve-qwen"),
            otherInstaller: installer(.qwen36))
        let qwen = try! #require(model.installs.first { $0.descriptor == .qwen36 })

        // On an idle disk each model fits on its own.
        #expect(model.selectedInstall.canInstall)
        #expect(qwen.canInstall)

        qwen.install()
        await Task.yield()
        #expect(qwen.isInstalling)

        // Qwen now owes its whole installed size, which is no longer free.
        model.installModel()
        #expect(!model.selectedInstall.isInstalling)
        if case .insufficientSpace(let requirement) = model.installReadiness {
            #expect(requirement.shortfallBytes > 0)
        } else {
            Issue.record("expected insufficientSpace, got \(model.installReadiness)")
        }

        // Cancelling the first download releases the reservation.
        qwen.cancel()
        try? await waitUntil { !qwen.isInstalling }
        model.refreshInstallReadiness()
        #expect(model.selectedInstall.canInstall)
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
