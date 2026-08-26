import AppKit
import TurboFieldfareAppCore
import TurboFieldfareAppServer
import TurboFieldfareAppUpdater
import TurboFieldfareMacPresentation
import SwiftUI

struct RootView: View {
    let model: AppModel
    let serverController: AppServerController
    let updateController: AppUpdateController
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @State private var navigation = AppNavigationState()
    @State private var renameTarget: AppConversationRecord?
    @State private var renameText = ""

    var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(
                    min: AppWindowLayout.sidebarMinimumWidth,
                    ideal: AppWindowLayout.sidebarIdealWidth,
                    max: AppWindowLayout.sidebarMaximumWidth)
        } detail: {
            destinationView
        }
        .navigationSplitViewStyle(.balanced)
        .focusedSceneValue(
            \.appNavigationAction,
            AppNavigationAction { navigation.select($0) })
        .containerBackground(for: .window) {
            LinearGradient(
                colors: [
                    Color(nsColor: .windowBackgroundColor),
                    Color(nsColor: .windowBackgroundColor).mix(
                        with: TurboFieldfareMacTheme.accentColor,
                        by: 0.04),
                ],
                startPoint: .top,
                endPoint: .bottom)
        }
        .tint(TurboFieldfareMacTheme.accentColor)
        .animation(.smooth(duration: 0.25), value: navigation.destination)
        .animation(.smooth(duration: 0.3), value: model.requiresModelInstallation)
        .animation(.smooth(duration: 0.25), value: model.error)
        .transaction { transaction in
            if model.isRunning { transaction.animation = nil }
        }
        .alert("Rename Chat", isPresented: renameAlertIsPresented) {
            TextField("Name", text: $renameText)
            Button("Cancel", role: .cancel) { renameTarget = nil }
            Button("Rename") {
                if let renameTarget {
                    model.renameConversation(renameTarget, to: renameText)
                }
                renameTarget = nil
            }
            .disabled(renameText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
    }

    private var sidebar: some View {
        List {
            Section {
                ForEach(AppDestination.allCases) { item in
                    Button {
                        navigation.select(item)
                    } label: {
                        HStack(spacing: 9) {
                            Label(item.title, systemImage: item.systemImage)
                            Spacer(minLength: 0)
                        }
                        .contentShape(Rectangle())
                    }
                        .buttonStyle(.plain)
                        .listRowBackground(destinationRowBackground(for: item))
                        .foregroundStyle(navigation.destination == item
                                         ? TurboFieldfareMacTheme.accentColor
                                         : .primary)
                        .accessibilityIdentifier("navigation.\(item.rawValue)")
                        .accessibilityValue(navigation.destination == item
                                            ? "Selected" : "")
                        .accessibilityAddTraits(
                            navigation.destination == item ? .isSelected : [])
                }
            }
            if navigation.destination == .chat {
                Section {
                    Button {
                        model.clearOutput()
                    } label: {
                        Label("New Chat", systemImage: "square.and.pencil")
                    }
                    .disabled(model.isRunning || (!model.hasOutputTranscript
                        && model.conversationStore.selectedConversation != nil))
                }
                Section("Chats") {
                    if model.conversationStore.conversations.isEmpty {
                        Text("No saved chats")
                            .font(.caption)
                            .foregroundStyle(.tertiary)
                    } else {
                        ForEach(model.conversationStore.conversations) { conversation in
                            conversationRow(conversation)
                        }
                    }
                }
            }
        }
        .listStyle(.sidebar)
        .safeAreaInset(edge: .top, spacing: 0) { brandHeader }
        .safeAreaInset(edge: .bottom, spacing: 0) { selectedModelFooter }
        .navigationTitle("TUFF")
    }

    private func destinationRowBackground(for item: AppDestination) -> Color {
        navigation.destination == item
            ? TurboFieldfareMacTheme.accentColor.opacity(0.14)
            : .clear
    }

    private func conversationRow(_ conversation: AppConversationRecord) -> some View {
        Button {
            model.selectConversation(conversation)
        } label: {
            HStack(spacing: 8) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(conversation.title)
                        .lineLimit(1)
                    Text(modelName(for: conversation.modelID))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }
                Spacer(minLength: 4)
                if model.conversationStore.selectedConversationID == conversation.id {
                    Circle()
                        .fill(TurboFieldfareMacTheme.accentColor)
                        .frame(width: 6, height: 6)
                        .accessibilityHidden(true)
                }
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .disabled(model.isRunning || model.loadState.isLoading)
        .contextMenu {
            Button("Rename") {
                renameText = conversation.title
                renameTarget = conversation
            }
            Divider()
            Button("Delete", role: .destructive) {
                model.deleteConversation(conversation)
            }
            .disabled(model.isRunning)
        }
        .accessibilityLabel("\(conversation.title), \(modelName(for: conversation.modelID))")
        .accessibilityValue(model.conversationStore.selectedConversationID == conversation.id
                            ? "Selected" : "")
        .accessibilityHint("Opens this saved chat")
        .accessibilityAddTraits(
            model.conversationStore.selectedConversationID == conversation.id
                ? .isSelected : [])
        .accessibilityAction(named: "Rename chat") {
            renameText = conversation.title
            renameTarget = conversation
        }
        .accessibilityAction(named: "Delete chat") {
            model.deleteConversation(conversation)
        }
    }

    private func modelName(for profileKey: String) -> String {
        model.installs.first {
            $0.descriptor.settingsProfileKey == profileKey
        }?.descriptor.shortName ?? profileKey
    }

    private var renameAlertIsPresented: Binding<Bool> {
        Binding {
            renameTarget != nil
        } set: { presented in
            if !presented { renameTarget = nil }
        }
    }

    private var brandHeader: some View {
        HStack(spacing: 10) {
            TUFFMarkView()
                .padding(3)
                .frame(width: 28, height: 28)
                .background(
                    TurboFieldfareMacTheme.surfaceStyle(
                        reduceTransparency: reduceTransparency,
                        material: .thin),
                    in: RoundedRectangle(cornerRadius: 8))
            VStack(alignment: .leading, spacing: 1) {
                Text("TUFF").font(.headline)
                Text("Runs on this Mac")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
    }

    private var selectedModelFooter: some View {
        Button {
            navigation.select(.models)
        } label: {
            HStack(alignment: .center, spacing: 8) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("CURRENT MODEL")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.tertiary)
                    Text(model.selectedDescriptor.displayName)
                        .font(.caption.weight(.medium))
                        .lineLimit(2)
                    Label(model.presentation.label, systemImage: model.loadState.isReady
                          ? "circle.fill" : "circle")
                        .font(.caption2)
                        .foregroundStyle(model.loadState.isReady
                                         ? TurboFieldfareMacTheme.accentColor
                                         : .secondary)
                }
                Spacer(minLength: 0)
                Image(systemName: "chevron.right")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.tertiary)
                    .accessibilityHidden(true)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(TurboFieldfareMacTheme.surfaceStyle(
            reduceTransparency: reduceTransparency,
            material: .thin))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Current model")
        .accessibilityValue("\(model.selectedDescriptor.displayName), \(model.presentation.label)")
        .accessibilityHint("Opens Models")
    }

    @ViewBuilder
    private var destinationView: some View {
        switch navigation.destination {
        case .chat:
            ChatWorkspaceView(model: model)
        case .models:
            ModelsWorkspaceView(model: model)
        case .server:
            ServerWorkspaceView(
                model: model, controller: serverController)
        case .settings:
            SettingsWorkspaceView(
                model: model, updateController: updateController)
        }
    }
}

private struct ChatWorkspaceView: View {
    let model: AppModel
    @State private var conversationChromeHeight: CGFloat = 0

    var body: some View {
        primaryContent
            .frame(
                minWidth: AppWindowLayout.detailMinimumWidth,
                maxWidth: .infinity,
                maxHeight: .infinity)
            .safeAreaInset(edge: .top, spacing: 0) {
                StatusHUDView(model: model)
            }
    }

    private var primaryContent: some View {
        Group {
            if model.requiresModelInstallation {
                ModelInstallView(model: model)
            } else {
                conversationView
            }
        }
    }

    private var conversationView: some View {
        GeometryReader { geometry in
            ZStack(alignment: .bottom) {
                if model.hasOutputTranscript {
                    OutputPaneView(model: model)
                        .padding(.bottom, conversationChromeHeight)
                } else if conversationChromeHeight > 0 {
                    OutputPaneView(model: model)
                        .frame(height: max(0, geometry.size.height - conversationChromeHeight))
                        .frame(maxHeight: .infinity, alignment: .top)
                }

                conversationChrome
                    .background {
                        GeometryReader { chromeGeometry in
                            Color.clear.preference(
                                key: ConversationChromeHeightKey.self,
                                value: chromeGeometry.size.height)
                        }
                    }
            }
            .onPreferenceChange(ConversationChromeHeightKey.self) { height in
                guard height > 0 else { return }
                var transaction = Transaction()
                transaction.disablesAnimations = true
                withTransaction(transaction) { conversationChromeHeight = height }
            }
        }
    }

    private var conversationChrome: some View {
        VStack(spacing: 10) {
            ChatControlsView(model: model)
            ErrorBanner(model: model)
            if model.promptText.isEmpty && model.showPromptExamples && !model.isRunning {
                PromptExamplesView { preset in model.promptText = preset.prompt }
            }
            ModelActionBanner(model: model)
            PromptComposerView(model: model)
        }
        .padding(.horizontal, 20)
        .padding(.bottom, 16)
        .animation(.smooth(duration: 0.2), value: model.promptText.isEmpty)
        .animation(.smooth(duration: 0.2), value: model.showPromptExamples)
    }
}

private struct ModelsWorkspaceView: View {
    let model: AppModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                WorkspaceTitle(
                    title: "Models",
                    subtitle: "Download and manage the models that run on this Mac.")
                ModelLibraryColumnsView(model: model)
            }
            .frame(maxWidth: 1_180, alignment: .leading)
            .padding(28)
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }
}

private struct ServerWorkspaceView: View {
    let model: AppModel
    let controller: AppServerController
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                WorkspaceTitle(
                    title: "Server",
                    subtitle: "Use TUFF through an OpenAI-compatible loopback endpoint.")
                statusCard
                configurationCard
                activityCard
                if !model.serverStore.recentErrors.isEmpty { errorsCard }
            }
            .frame(maxWidth: 920, alignment: .leading)
            .padding(28)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var statusCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 12) {
                ZStack {
                    Circle()
                        .fill(TurboFieldfareMacTheme.accentColor.opacity(0.12))
                    Image(systemName: "network")
                        .font(.title2)
                        .foregroundStyle(TurboFieldfareMacTheme.accentColor)
                }
                .frame(width: 44, height: 44)
                VStack(alignment: .leading, spacing: 3) {
                    Text("Local server").font(.headline)
                    Text(serverStatus)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                statusControl
            }
            Divider()
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Endpoint")
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    Text(endpointText)
                        .font(.body.monospaced())
                        .textSelection(.enabled)
                }
                Spacer()
                Button {
                    guard let url = controller.url else { return }
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(
                        url.absoluteString, forType: .string)
                } label: {
                    Label("Copy URL", systemImage: "doc.on.doc")
                }
                .disabled(controller.url == nil)
                .accessibilityHint("Copies the loopback server URL")
            }
        }
        .padding(18)
        .background(cardBackground, in: RoundedRectangle(cornerRadius: 16))
    }

    @ViewBuilder
    private var statusControl: some View {
        switch model.serverStore.status {
        case .starting, .stopping:
            ProgressView()
                .controlSize(.small)
                .accessibilityLabel(serverStatus)
        case .running:
            Button("Stop", role: .destructive) { controller.stop() }
                .keyboardShortcut("s", modifiers: [.command, .option])
        case .stopped, .failed:
            if model.loadState.isReady, !model.hasStaleLoadedRuntime {
                Button("Start") {
                    if let configuration { controller.start(configuration) }
                }
                .buttonStyle(.borderedProminent)
                .disabled(configuration == nil)
                .keyboardShortcut("s", modifiers: [.command, .option])
            } else if model.hasStaleLoadedRuntime {
                Button("Reload Model", action: model.reloadModel)
                    .disabled(!model.canReloadModel)
            } else {
                Button("Load Model", action: model.loadModel)
                    .disabled(!model.canLoadModel)
            }
        }
    }

    private var configurationCard: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Configuration")
                .font(.headline)
            Picker("Model", selection: selectedModelBinding) {
                ForEach(model.installs) { install in
                    Text(install.descriptor.displayName).tag(install.id)
                }
            }
            .disabled(model.serverStore.isBusy || model.isRunning)
            .accessibilityHint("Selects the model Chat and Server share")

            LabeledContent("API model ID") {
                Text(model.selectedDescriptor.apiModelID)
                    .font(.callout.monospaced())
                    .textSelection(.enabled)
            }
            HStack(spacing: 18) {
                TextField("Port", value: desiredPortBinding, format: .number)
                    .frame(maxWidth: 150)
                    .textFieldStyle(.roundedBorder)
                    .disabled(model.serverStore.isBusy)
                Stepper(
                    "Queue: \(model.serverStore.queueLimit)",
                    value: queueLimitBinding,
                    in: 1...16)
                    .disabled(model.serverStore.isBusy)
                Spacer()
            }
            Text("The server binds only to 127.0.0.1. Chat and API requests share one model process and run one at a time.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(18)
        .background(cardBackground, in: RoundedRectangle(cornerRadius: 16))
    }

    private var activityCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Health and activity").font(.headline)
                Spacer()
                Button {
                    controller.refreshHealth()
                } label: {
                    Label("Check", systemImage: "arrow.clockwise")
                }
                .disabled(model.serverStore.status != .running
                          || model.serverStore.health == .checking)
            }
            HStack(spacing: 12) {
                ServerMetric(
                    title: "Health",
                    value: healthLabel,
                    systemImage: healthSystemImage,
                    accent: model.serverStore.health == .healthy)
                ServerMetric(
                    title: "Active",
                    value: "\(model.serverStore.activeRequests)",
                    systemImage: "bolt.horizontal.circle",
                    accent: model.serverStore.activeRequests > 0)
                ServerMetric(
                    title: "Queued",
                    value: "\(model.serverStore.queuedRequests)",
                    systemImage: "list.number",
                    accent: model.serverStore.queuedRequests > 0)
            }
        }
        .padding(18)
        .background(cardBackground, in: RoundedRectangle(cornerRadius: 16))
    }

    private var errorsCard: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Recent errors").font(.headline)
            ForEach(Array(model.serverStore.recentErrors.enumerated()), id: \.offset) {
                _, message in
                Label(message, systemImage: "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
            }
        }
        .padding(18)
        .background(cardBackground, in: RoundedRectangle(cornerRadius: 16))
    }

    private var cardBackground: AnyShapeStyle {
        TurboFieldfareMacTheme.surfaceStyle(
            reduceTransparency: reduceTransparency)
    }

    private var selectedModelBinding: Binding<String> {
        Binding {
            model.selectedModelID
        } set: { id in
            guard let install = model.installs.first(where: { $0.id == id }) else { return }
            model.selectModel(install)
        }
    }

    private var desiredPortBinding: Binding<Int> {
        Binding {
            model.serverStore.desiredPort
        } set: { value in
            model.serverStore.desiredPort = min(max(value, 1), 65_535)
        }
    }

    private var queueLimitBinding: Binding<Int> {
        Binding {
            model.serverStore.queueLimit
        } set: { model.serverStore.queueLimit = $0 }
    }

    private var configuration: AppHostedServerConfiguration? {
        guard model.loadState.isReady,
              !model.hasStaleLoadedRuntime,
              let loaded = model.loadedRuntimeKey,
              (1...65_535).contains(model.serverStore.desiredPort) else { return nil }
        let visionCapability = if !model.visionRuntimeEnabled {
            "disabled"
        } else if model.isImageInputAvailable {
            "ready"
        } else {
            "missing"
        }
        return AppHostedServerConfiguration(
            modelID: model.selectedDescriptor.apiModelID,
            chatDialect: model.selectedDescriptor.chatDialect,
            visionCapability: visionCapability,
            port: model.serverStore.desiredPort,
            queueLimit: model.serverStore.queueLimit,
            runtime: AppServerRuntimeConfiguration(
                modelDirectory: loaded.modelDirectory,
                maxContextTokens: loaded.maxContextTokens,
                runtimeOptions: loaded.options(
                    prefillEnabled: model.runtimeOptions.prefillEnabled,
                    prefillChunkTokens: model.runtimeOptions.prefillChunkTokens),
                preserveThinking: model.selectedDescriptor.family == .qwen36
                    && model.preserveThinking))
    }

    private var endpointText: String {
        controller.url?.absoluteString
            ?? "http://127.0.0.1:\(model.serverStore.desiredPort)"
    }

    private var serverStatus: String {
        switch model.serverStore.status {
        case .stopped: "Stopped"
        case .starting: "Starting on this Mac…"
        case .running: "Running on 127.0.0.1"
        case .stopping: "Stopping…"
        case .failed(let detail): "Could not start: \(detail)"
        }
    }

    private var healthLabel: String {
        switch model.serverStore.health {
        case .unknown: "Not checked"
        case .checking: "Checking"
        case .healthy: "Healthy"
        case .unreachable: "Unavailable"
        }
    }

    private var healthSystemImage: String {
        switch model.serverStore.health {
        case .unknown: "minus.circle"
        case .checking: "arrow.clockwise.circle"
        case .healthy: "checkmark.circle.fill"
        case .unreachable: "xmark.circle.fill"
        }
    }
}

private struct ServerMetric: View {
    let title: String
    let value: String
    let systemImage: String
    let accent: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Image(systemName: systemImage)
                .foregroundStyle(accent
                    ? TurboFieldfareMacTheme.accentColor : .secondary)
            Text(value)
                .font(.title3.weight(.semibold))
            Text(title)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color.primary.opacity(0.035), in: RoundedRectangle(cornerRadius: 12))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(title)
        .accessibilityValue(value)
    }
}

private struct SettingsWorkspaceView: View {
    let model: AppModel
    let updateController: AppUpdateController

    var body: some View {
        VStack(spacing: 0) {
            WorkspaceTitle(
                title: "Settings",
                subtitle: "Tune model, memory, and generation behavior.")
                .padding(.horizontal, 28)
                .padding(.top, 24)
                .frame(maxWidth: .infinity, alignment: .leading)
            AppSettingsView(
                model: model, updateController: updateController)
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }
}

private struct WorkspaceTitle: View {
    let title: String
    let subtitle: String

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text(title)
                .font(.largeTitle.bold())
                .accessibilityHeading(.h1)
            Text(subtitle)
                .font(.callout)
                .foregroundStyle(.secondary)
        }
        .accessibilityElement(children: .contain)
    }
}

private struct ConversationChromeHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}
