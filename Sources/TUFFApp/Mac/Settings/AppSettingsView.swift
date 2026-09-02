import AppKit
import TUFFEngine
import TUFFAppCore
import TUFFAppUpdater
import TUFFMacPresentation
import SwiftUI

struct AppSettingsView: View {
    @Bindable var model: AppModel
    let updateController: AppUpdateController
    @State private var section: AppSettingsSection = .general
    @State private var profileModelID = ""
    /// Holds text the user is mid-typing into the hex field until it parses,
    /// so an incomplete value (e.g. "6F4") isn't snapped back to the last
    /// valid one on every keystroke.
    @State private var pendingAccentHex: String?

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
            Section("Appearance") {
                Picker("Accent color", selection: accentColorModeBinding) {
                    ForEach(AppAccentColorMode.allCases) { mode in
                        Text(mode.settingsLabel).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                if model.accentColorMode == .custom {
                    ColorPicker("Custom color", selection: customAccentColorBinding, supportsOpacity: false)
                    LabeledContent("Hex value") {
                        TextField("6F4DFF", text: customAccentColorHexFieldBinding)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 100)
                            .multilineTextAlignment(.trailing)
                    }
                } else if model.accentColorMode == .system {
                    Text("Follows the accent color set in System Settings.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
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
                systemPromptSettings
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
            Toggle("Auto", isOn: automaticMemoryBinding)
            if selectedProfile.automaticMemory,
               let plan = model.automaticMemoryPlan(for: selectedProfileInstall) {
                Text(automaticMemorySummary(plan))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Group {
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
                        ForEach(selectedProfileInstall.descriptor.usefulExpertCacheSlotCounts,
                                id: \.self) { slots in
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
            }
            .disabled(selectedProfile.automaticMemory)
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

    /// This model's system prompt. Stored with the rest of its profile, so
    /// each model carries its own.
    private var systemPromptSettings: some View {
        Section("System prompt") {
            TextEditor(text: profileBinding(\.systemPrompt))
                .font(.body)
                .frame(minHeight: 96)
                .overlay(alignment: .topLeading) {
                    if selectedProfile.systemPrompt.isEmpty {
                        Text("Sent ahead of every chat with this model.")
                            .font(.body)
                            .foregroundStyle(.tertiary)
                            .padding(.leading, 5)
                            .padding(.top, 8)
                            .allowsHitTesting(false)
                    }
                }
                .accessibilityLabel("System prompt")
            HStack {
                Spacer()
                Button("Restore Default") {
                    var profile = selectedProfile
                    profile.systemPrompt = selectedProfileInstall.descriptor.defaultSystemPrompt
                    model.updateSettingsProfile(profile, for: selectedProfileInstall)
                }
                .disabled(selectedProfile.systemPrompt
                    == selectedProfileInstall.descriptor.defaultSystemPrompt)
                .help("Restore TUFF's default prompt for this model")
            }
            Text("Sent as a system message before the conversation. It applies "
                 + "from the next message; answers already given were not "
                 + "written with it.")
                .font(.caption)
                .foregroundStyle(.secondary)
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
                case .alwaysOn:
                    LabeledContent("Thinking", value: "Always On")
                    Text("This model always reasons before answering. TUFF keeps "
                        + "that reasoning separate from the visible answer.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                if control != .alwaysOn {
                    Text("Chat can override this for the current conversation.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
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
                    .disabled(selectedProfile.automaticMemory
                        || selectedProfile.expertCacheSlots
                        < RuntimeConfiguration.minimumExpertCacheSlotsForChunkedPrefill)
                if selectedProfile.automaticMemory {
                    Text("Auto manages chunked prefill with this model's memory plan.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else if selectedProfile.expertCacheSlots
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

    private var automaticMemoryBinding: Binding<Bool> {
        Binding {
            selectedProfile.automaticMemory
        } set: { enabled in
            model.setAutomaticMemory(enabled, for: selectedProfileInstall)
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

    private func automaticMemorySummary(_ plan: AppAutomaticMemoryPlan) -> String {
        let detected = MetricFormat.storage(model.deviceCapabilities.unifiedMemoryBytes)
        let selected = MetricFormat.storage(plan.estimatedWorkingSetBytes)
        let budget = MetricFormat.storage(plan.safeBudgetBytes)
        let available = model.deviceCapabilities.availableMemoryBytes.map {
            " \(MetricFormat.storage($0)) was available at launch."
        } ?? ""
        return "Detected \(detected) unified memory.\(available) TUFF selected "
            + "\(selected) of its \(budget) safe app budget for this model."
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

    private var accentColorModeBinding: Binding<AppAccentColorMode> {
        Binding { model.accentColorMode } set: { model.setAccentColorMode($0) }
    }

    private var customAccentColorBinding: Binding<Color> {
        Binding {
            let hex = AppHexColor(hexString: model.customAccentColorHex) ?? .defaultPurple
            return Color(red: hex.red, green: hex.green, blue: hex.blue)
        } set: { newColor in
            guard let hex = Self.hexColor(from: newColor) else { return }
            pendingAccentHex = nil
            model.setCustomAccentColorHex(hex.hexString)
        }
    }

    private var customAccentColorHexFieldBinding: Binding<String> {
        Binding {
            pendingAccentHex
                ?? model.customAccentColorHex.trimmingCharacters(in: CharacterSet(charactersIn: "#"))
        } set: { newValue in
            pendingAccentHex = newValue
            let candidate = "#" + newValue.trimmingCharacters(in: CharacterSet(charactersIn: "#"))
            guard AppHexColor(hexString: candidate) != nil else { return }
            model.setCustomAccentColorHex(candidate)
            pendingAccentHex = nil
        }
    }

    private static func hexColor(from color: Color) -> AppHexColor? {
        guard let rgb = NSColor(color).usingColorSpace(.deviceRGB) else { return nil }
        return AppHexColor(
            red: rgb.redComponent, green: rgb.greenComponent, blue: rgb.blueComponent)
    }
}
