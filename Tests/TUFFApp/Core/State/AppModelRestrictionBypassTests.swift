import Foundation
import Testing
import TUFFModelCatalog
@testable import TUFFAppCore

/// The escape hatch and the explanation that goes with it.
///
/// Both exist for the same complaint: a model the app refuses gives no visible
/// reason, and there is no way to overrule the refusal on a machine whose
/// owner is willing to accept the swapping.
@Suite struct AppModelRestrictionBypassTests {
    /// A Mac below every model's floor, so the hardware gate is the thing
    /// under test rather than a rounding difference.
    private let tinyMac = TUFFDeviceCapabilities(
        unifiedMemoryBytes: 4 * TUFFModelCatalog.oneGiB,
        macOSMajorVersion: 26,
        appleSiliconGeneration: 5)

    @MainActor
    @Test func hardwareGateRefusesDownloadAndSelectionUntilBypassed() {
        let model = makeAppModel(deviceCapabilities: tinyMac)
        let install = model.selectedInstall

        #expect(!model.hardwareRequirementsSatisfied(for: install))
        #expect(!model.canInstallModel(install))

        model.bypassModelRestrictions = true

        #expect(model.hardwareRequirementsSatisfied(for: install))
        #expect(model.canInstallModel(install))
    }

    /// The gate is waived, but the requirement is still reported: bypassing is
    /// a decision to proceed, not a claim that the Mac is big enough.
    @MainActor
    @Test func bypassLeavesTheHardwareRequirementVisible() {
        let model = makeAppModel(deviceCapabilities: tinyMac)
        model.bypassModelRestrictions = true

        let eligibility = model.hardwareEligibility(for: model.selectedInstall)
        #expect(!eligibility.isCompatible)
        #expect(eligibility.explanation != nil)
    }

    /// The other half of the complaint: settings raised past the budget, on a
    /// Mac that meets the model's hardware floor. Nothing about the model is
    /// wrong; the chosen context is simply larger than the budget.
    @MainActor
    @Test func settingsAboveTheBudgetBlockLoadingUntilBypassed() throws {
        let directory = try makeCompleteModelInstall("over-budget")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = makeAppModel(
            modelDirectory: directory,
            deviceCapabilities: TUFFDeviceCapabilities(
                unifiedMemoryBytes: 16 * TUFFModelCatalog.oneGiB,
                macOSMajorVersion: 26,
                appleSiliconGeneration: 5))
        model.automaticMemory = false
        model.maxContextTokens = 65_536
        model.runtimeOptions.expertCacheSlots = 128

        #expect(!model.memoryRequirementsSatisfied)
        #expect(!model.canLoadModel)
        let reason = try #require(model.loadBlockedReason)
        #expect(reason.contains("Context"))

        model.bypassModelRestrictions = true
        #expect(model.memoryRequirementsSatisfied)
        #expect(model.canLoadModel)
        #expect(model.loadBlockedReason == nil)
    }

    // MARK: - The explanation

    @MainActor
    @Test func aBlockedModelSaysWhyAndNamesTheWayOut() throws {
        let directory = try makeCompleteModelInstall("blocked")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = makeAppModel(modelDirectory: directory,
                                 deviceCapabilities: tinyMac)

        #expect(model.isModelInstalled)
        #expect(!model.canLoadModel)
        let reason = try #require(model.loadBlockedReason)
        #expect(reason.contains("Bypass model restrictions"))
    }

    @MainActor
    @Test func bypassingClearsTheExplanationAndAllowsTheLoad() throws {
        let directory = try makeCompleteModelInstall("bypassed")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = makeAppModel(modelDirectory: directory,
                                 deviceCapabilities: tinyMac)
        model.bypassModelRestrictions = true

        #expect(model.loadBlockedReason == nil)
        #expect(model.canLoadModel)
    }

    @MainActor
    @Test func aQualifyingMacHasNothingToExplain() throws {
        let directory = try makeCompleteModelInstall("qualifying")
        defer { try? FileManager.default.removeItem(at: directory) }
        let model = makeAppModel(modelDirectory: directory)

        #expect(model.loadBlockedReason == nil)
    }

    /// The reason has to reach the conversation. It resolves to a warning with
    /// no action, because there is no button that would help — which is
    /// exactly the state that used to render an enabled "Load Model" button
    /// that did nothing, or nothing at all.
    @Test func theConversationShowsTheReasonWithoutOfferingAButton() throws {
        let state = AppPresentationState.resolve(AppPresentationSnapshot(
            requiresInstallation: false,
            installState: .idle,
            installReadiness: .ready(AppModelInstallRequirement(
                requiredBytes: 0, availableBytes: .max)),
            loadState: .notLoaded,
            hasStaleRuntime: false,
            isRunning: false,
            isGenerationCancellationPending: false,
            generationPhase: .idle,
            loadBlockedReason: "Estimated memory is 20 GB; this Mac's safe app "
                + "budget is 12 GB."))

        #expect(state.severity == .warning)
        #expect(state.primaryAction == nil)
        #expect(state.conversationAction == nil)
        #expect(try #require(state.conversationNotice).contains("12 GB"))
    }

    @Test func anOrdinaryUnloadedModelStillOffersLoad() {
        let state = AppPresentationState.resolve(AppPresentationSnapshot(
            requiresInstallation: false,
            installState: .idle,
            installReadiness: .ready(AppModelInstallRequirement(
                requiredBytes: 0, availableBytes: .max)),
            loadState: .notLoaded,
            hasStaleRuntime: false,
            isRunning: false,
            isGenerationCancellationPending: false,
            generationPhase: .idle))

        #expect(state.primaryAction == AppModelAction.load)
        #expect(state.conversationNotice == nil)
    }

    // MARK: - Persistence

    @MainActor
    @Test func theBypassSurvivesARelaunch() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("tuff-bypass-\(UUID().uuidString)", isDirectory: true)
        let modelDirectory = directory.appendingPathComponent(
            "gemma4.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(
            at: modelDirectory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let first = makeAppModel(modelDirectory: modelDirectory,
                                 settingsPersistenceEnabled: true)
        first.bypassModelRestrictions = true

        let second = makeAppModel(modelDirectory: modelDirectory,
                                  settingsPersistenceEnabled: true)
        #expect(second.bypassModelRestrictions)
    }
}
