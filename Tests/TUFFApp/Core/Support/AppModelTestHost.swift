import Foundation
import TUFFModelCatalog
@testable import TUFFAppCore

extension TUFFDeviceCapabilities {
    /// A Mac that qualifies for every model in the catalog.
    ///
    /// Install, load, and companion actions are gated on the host's real
    /// unified memory, so a suite that took `AppModel`'s `.current()` default
    /// was asserting the RAM of whatever machine happened to run it. The
    /// Apple-silicon CI runners have 7 GB — under the 8 GB catalog floor —
    /// which turned every one of those actions off and failed the app suites
    /// for a reason that has nothing to do with the code under test.
    ///
    /// Hardware and context gating keep their own coverage in
    /// `ModelCatalogTests`, where each test states the device it means.
    static let qualifyingTestHost = TUFFDeviceCapabilities(
        unifiedMemoryBytes: 32 * TUFFModelCatalog.oneGiB,
        macOSMajorVersion: 15,
        appleSiliconGeneration: 2)
}

/// `AppModel` on a qualifying Mac, whatever the suite is running on.
///
/// Every parameter mirrors `AppModel.init`, defaults included. Only the device
/// differs: it is pinned rather than measured. A test that means to assert
/// gating passes its own `deviceCapabilities`, exactly as it would to the
/// initializer.
@MainActor
func makeAppModel(
    modelDirectory: URL? = nil,
    client: any AppInferenceClient = RealInferenceClient(),
    installer: any AppModelInstallerClient
        = RepackModelInstallerClient(descriptor: .default),
    otherInstalls: [ModelInstallCoordinator]? = nil,
    visionInstaller: any AppVisionPackInstallerClient
        = RepackVisionPackInstallerClient(),
    memorySampler: AppMemorySampler = AppMemorySampler(),
    attachmentStore: AppImageAttachmentStore = AppImageAttachmentStore(),
    conversationStore: AppConversationStore = AppConversationStore(),
    visionRuntimeSupported: Bool = true,
    settingsPersistenceEnabled: Bool = false,
    deviceCapabilities: TUFFDeviceCapabilities = .qualifyingTestHost
) -> AppModel {
    AppModel(
        modelDirectory: modelDirectory,
        client: client,
        installer: installer,
        otherInstalls: otherInstalls,
        visionInstaller: visionInstaller,
        memorySampler: memorySampler,
        attachmentStore: attachmentStore,
        conversationStore: conversationStore,
        visionRuntimeSupported: visionRuntimeSupported,
        settingsPersistenceEnabled: settingsPersistenceEnabled,
        deviceCapabilities: deviceCapabilities)
}
