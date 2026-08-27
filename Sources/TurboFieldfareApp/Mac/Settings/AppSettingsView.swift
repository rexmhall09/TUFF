import TurboFieldfare
import TurboFieldfareAppCore
import TurboFieldfareAppUpdater
import TurboFieldfareMacPresentation
import SwiftUI

struct AppSettingsView: View {
    @Bindable var model: AppModel
    let updateController: AppUpdateController
    @State private var section: AppSettingsSection = .general
    @State private var profileModelID = ""

    var body: some View {
        VStack(spacing: 0) {
            Picker("Settings section", selection: $section) {
                ForEach(AppSettingsSection.allCases) { item in
                    Label(item.title, systemImage: item.systemImage)
                        .tag(item)
                }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .controlSize(.large)
            .frame(maxWidth: 520)
            .padding(.horizontal, 28)
            .padding(.top, 20)
            .padding(.bottom, 10)
            .accessibilityLabel("Settings section")

            Group {
                switch section {
                case .general: generalSettings
                case .models: modelSettings(advanced: false)
                case .advanced: modelSettings(advanced: true)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            // Settings controls read as full-size rows rather than the compact
            // system default, so a pop-up here is a comfortable target.
            .controlSize(.large)
            .id(section)
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var generalSettings: some View {
        Form {
            Section("Chat") {
                Picker("Send message with", selection: newlineShortcutBinding) {
                    ForEach(AppNewlineShortcut.sendMessageOptions) { shortcut in
                        Text(shortcut.sendMessageLabel).tag(shortcut)
                    }
                }
                Picker("After sending", selection: sentPromptBehaviorBinding) {
                    ForEach(AppSentPromptBehavior.allCases) { behavior in
                        Text(behavior.settingsLabel).tag(behavior)
                    }
                }
                Toggle("Show prompt examples", isOn: showPromptExamplesBinding)
            }
            Section("Startup") {
                Toggle("Load the selected model at launch",
                       isOn: loadModelOnLaunchBinding)
                Text("TUFF will only load when the model is installed and its "
                    + "saved memory profile is safe for this Mac.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Section("Updates") {
                if updateController.isAvailable {
                    Toggle(
                        "Check for updates automatically",
                        isOn: automaticUpdateChecksBinding)
                    Toggle(
                        "Download and install updates automatically",
                        isOn: automaticUpdateDownloadsBinding)
                        .disabled(!updateController.automaticallyChecksForUpdates
                            || !updateController.allowsAutomaticUpdates)
                    Button("Check for Updates Now") {
                        updateController.checkForUpdates()
                    }
                    .disabled(!updateController.canCheckForUpdates)
                    Text("Updates come from TUFF's GitHub Releases feed. Each "
                        + "download must pass Sparkle's embedded EdDSA signature "
                        + "check before it can be installed.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    Label(
                        updateController.unavailableReason
                            ?? "Updates are unavailable in this build.",
                        systemImage: "exclamationmark.shield")
                        .foregroundStyle(.secondary)
                    Text("Clone builds stay usable, but only packaged builds with "
                        + "TUFF's update-signing public key can install releases.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
    }

    private func modelSettings(advanced: Bool) -> some View {
        Form {
            Section("Profile") {
                Picker("Model", selection: profileModelBinding) {
                    ForEach(model.installs) { install in
                        Text(install.descriptor.shortName).tag(install.id)
                    }
                }
                if selectedProfileInstall.id == model.selectedModelID,
                   model.hasStaleLoadedRuntime {
                    Label("Reload the model to apply memory changes.",
                          systemImage: "arrow.clockwise")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            if advanced {
                advancedProfileSettings
            } else {
                memorySettings
                reasoningSettings
                samplingSettings
            }
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .disabled(model.isRunning || model.loadState.isLoading
            || model.isVisionCompanionOperationInProgress)
    }

    private var memorySettings: some View {
        Section("Memory") {
            Picker("Context", selection: profileBinding(\.contextTokens)) {
                ForEach(AppContextLengthOption.allCases) { option in
                    let eligibility = contextEligibility(
                        tokens: option.tokens,
                        slots: selectedProfile.expertCacheSlots)
                    Text(contextLabel(option, eligibility: eligibility))
                        .tag(option.tokens)
                        .disabled(!eligibility.isCompatible)
                }
            }
            if selectedProfileInstall.descriptor.usesExpertCache {
                Picker("Expert cache", selection: expertCacheSlotsBinding) {
                    ForEach(AppRuntimeOptions.allowedSlotCounts, id: \.self) { slots in
                        let eligibility = contextEligibility(
                            tokens: selectedProfile.contextTokens,
                            slots: slots)
                        Text("\(slots) slots · "
                            + MetricFormat.storage(eligibility.estimatedWorkingSetBytes))
                            .tag(slots)
                            .disabled(!eligibility.isCompatible)
                    }
                }
            } else {
                LabeledContent("Expert cache", value: "Not used by this dense model")
                    .foregroundStyle(.secondary)
            }
            let eligibility = contextEligibility(
                tokens: selectedProfile.contextTokens,
                slots: selectedProfile.expertCacheSlots)
            LabeledContent("Estimated working set") {
                Text(MetricFormat.storage(eligibility.estimatedWorkingSetBytes))
                    .monospacedDigit()
            }
            if let explanation = eligibility.explanation {
                Label(explanation, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.orange)
            }
        }
    }

    @ViewBuilder
    private var reasoningSettings: some View {
        if let control = selectedProfileInstall.descriptor.reasoningControl {
            Section("Reasoning default") {
                switch control {
                case .toggle, .toggleWithPreservation:
                    Picker("Thinking", selection: profileBinding(\.defaultReasoning)) {
                        Text("Off").tag(ChatReasoning.off)
                        Text("On").tag(ChatReasoning.on)
                    }
                    .pickerStyle(.segmented)
                case .graded:
                    Picker("Effort", selection: profileBinding(\.defaultReasoningEffort)) {
                        ForEach(GPTOSSReasoningEffort.allCases, id: \.self) { effort in
                            Text(effort.rawValue.capitalized).tag(effort)
                        }
                    }
                    .pickerStyle(.segmented)
                }
                Text("Chat can override this for the current conversation.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var samplingSettings: some View {
        Section("Sampling") {
            LabeledContent("Temperature") {
                HStack(spacing: 10) {
                    Slider(value: profileBinding(\.temperature), in: 0...2, step: 0.05)
                    Text(selectedProfile.temperature,
                         format: .number.precision(.fractionLength(2)))
                        .monospacedDigit()
                        .frame(width: 38, alignment: .trailing)
                }
            }
            Toggle("Top-K", isOn: profileBinding(\.topKEnabled))
            if selectedProfile.topKEnabled {
                Stepper(value: profileBinding(\.topK), in: 1...256) {
                    LabeledContent("K value", value: "\(selectedProfile.topK)")
                }
                Toggle("Top-P", isOn: profileBinding(\.topPEnabled))
                if selectedProfile.topPEnabled {
                    LabeledContent("P value") {
                        HStack(spacing: 10) {
                            Slider(value: profileBinding(\.topP),
                                   in: 0.01...1, step: 0.01)
                            Text(selectedProfile.topP,
                                 format: .number.precision(.fractionLength(2)))
                                .monospacedDigit()
                                .frame(width: 38, alignment: .trailing)
                        }
                    }
                }
            }
        }
    }

    private var advancedProfileSettings: some View {
        Group {
            Section("Runtime") {
                Toggle("Chunked prefill", isOn: profileBinding(\.prefillEnabled))
                    .disabled(selectedProfile.expertCacheSlots
                        < RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill)
                if selectedProfile.expertCacheSlots
                    < RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill {
                    Text("Chunked prefill needs at least "
                        + "\(RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill) "
                        + "expert-cache slots.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Picker("RDADVISE", selection: profileBinding(\.rdadvisePolicy)) {
                    ForEach(AppRDAdvicePolicy.allCases) { policy in
                        Text(policy.label).tag(policy)
                    }
                }
                Text("RDADVISE is experimental. Changes to memory mapping apply "
                    + "after the selected model is reloaded.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if selectedProfileInstall.descriptor.reasoningControl
                == .toggleWithPreservation {
                Section("Thinking history") {
                    Toggle("Preserve thinking in later prompts",
                           isOn: profileBinding(\.preserveThinking))
                    Text("When enabled, Qwen receives its earlier thinking text as "
                        + "part of conversation history. TUFF always keeps that text "
                        + "available for you in the transcript.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
            Section("Location") {
                Text(selectedProfileInstall.directoryURL.path)
                    .font(.caption.monospaced())
                    .textSelection(.enabled)
            }
            if selectedProfileInstall.id == model.selectedModelID,
               let diagnostics = model.diagnostics {
                RunnerDiagnosticsSection(diagnostics: diagnostics)
            }
        }
    }

    private var selectedProfileInstall: ModelInstallCoordinator {
        model.installs.first { $0.id == profileModelID } ?? model.selectedInstall
    }

    private var selectedProfile: AppModelSettingsProfile {
        model.settingsProfile(for: selectedProfileInstall)
    }

    private var profileModelBinding: Binding<String> {
        Binding {
            profileModelID.isEmpty ? model.selectedModelID : profileModelID
        } set: { profileModelID = $0 }
    }

    private var automaticUpdateChecksBinding: Binding<Bool> {
        Binding {
            updateController.automaticallyChecksForUpdates
        } set: { enabled in
            updateController.setAutomaticallyChecksForUpdates(enabled)
        }
    }

    private var automaticUpdateDownloadsBinding: Binding<Bool> {
        Binding {
            updateController.automaticallyDownloadsUpdates
        } set: { enabled in
            updateController.setAutomaticallyDownloadsUpdates(enabled)
        }
    }

    private func profileBinding<Value>(
        _ keyPath: WritableKeyPath<AppModelSettingsProfile, Value>
    ) -> Binding<Value> {
        Binding {
            selectedProfile[keyPath: keyPath]
        } set: { value in
            var profile = selectedProfile
            profile[keyPath: keyPath] = value
            model.updateSettingsProfile(profile, for: selectedProfileInstall)
        }
    }

    private var expertCacheSlotsBinding: Binding<Int> {
        Binding {
            selectedProfile.expertCacheSlots
        } set: { slots in
            var profile = selectedProfile
            profile.expertCacheSlots = slots
            if slots < RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill {
                profile.prefillEnabled = false
            }
            model.updateSettingsProfile(profile, for: selectedProfileInstall)
        }
    }

    private func contextEligibility(
        tokens: Int,
        slots: Int
    ) -> AppModelContextEligibility {
        model.contextEligibility(
            for: selectedProfileInstall,
            contextTokens: tokens,
            expertCacheSlots: slots)
    }

    private func contextLabel(
        _ option: AppContextLengthOption,
        eligibility: AppModelContextEligibility
    ) -> String {
        option.shortLabel + " · "
            + MetricFormat.storage(eligibility.estimatedWorkingSetBytes)
    }

    private var newlineShortcutBinding: Binding<AppNewlineShortcut> {
        Binding { model.newlineShortcut } set: { model.setNewlineShortcut($0) }
    }

    private var sentPromptBehaviorBinding: Binding<AppSentPromptBehavior> {
        Binding { model.sentPromptBehavior } set: { model.setSentPromptBehavior($0) }
    }

    private var showPromptExamplesBinding: Binding<Bool> {
        Binding { model.showPromptExamples } set: { model.setShowPromptExamples($0) }
    }

    private var loadModelOnLaunchBinding: Binding<Bool> {
        Binding { model.loadModelOnLaunch } set: { model.setLoadModelOnLaunch($0) }
    }
}
