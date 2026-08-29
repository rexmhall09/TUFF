import AppKit
import TUFFAppCore
import TUFFAppServer
import TUFFAppUpdater
import TUFFMacPresentation
import SwiftUI

/// Whether the window is in full screen.
///
/// The title bar is hidden there, so the chrome that normally sits clear of it
/// — the sidebar header, the status strip's room for the traffic lights — has
/// to make its own space instead.
private struct WindowIsFullScreenKey: EnvironmentKey {
    static let defaultValue = false
}

extension EnvironmentValues {
    var windowIsFullScreen: Bool {
        get { self[WindowIsFullScreenKey.self] }
        set { self[WindowIsFullScreenKey.self] = newValue }
    }
}

struct RootView: View {
    let model: AppModel
    let serverController: AppServerController
    let updateController: AppUpdateController
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @State private var navigation = AppNavigationState()
    @State private var renameTarget: AppConversationRecord?
    @State private var renameText = ""
    @State private var isFullScreen = false

    var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(
                    min: AppWindowLayout.sidebarMinimumWidth,
                    ideal: AppWindowLayout.sidebarIdealWidth,
                    max: AppWindowLayout.sidebarMaximumWidth)
                // Full screen keeps the band the toolbar occupies even though
                // there is no title bar in it, so the sidebar panel began about
                // 31pt below the top of the screen while sitting 5pt from the
                // bottom and 5pt from every edge of a windowed frame.
                //
                // The panel is let through that band and then inset by the same
                // small gap it has everywhere else, so full screen matches a
                // window rather than either sinking the panel or — the previous
                // attempt — running it flush into the black.
                .ignoresSafeArea(.container, edges: isFullScreen ? .top : [])
                .padding(.top, isFullScreen ? Self.sidebarEdgeInset : 0)
        } detail: {
            AppWorkspaceView(
                destination: navigation.destination,
                model: model,
                serverController: serverController,
                updateController: updateController)
        }
        .navigationSplitViewStyle(.balanced)
        .environment(\.windowIsFullScreen, isFullScreen)
        .background {
            WindowFullScreenReader(isFullScreen: $isFullScreen)
                .accessibilityHidden(true)
        }
        .focusedSceneValue(
            \.appNavigationAction,
            AppNavigationAction { navigation.select($0) })
        .containerBackground(for: .window) {
            LinearGradient(
                colors: [
                    Color(nsColor: .windowBackgroundColor),
                    Color(nsColor: .windowBackgroundColor).mix(
                        with: TUFFMacTheme.accentColor,
                        by: 0.04),
                ],
                startPoint: .top,
                endPoint: .bottom)
        }
        .tint(TUFFMacTheme.accentColor)
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
                                         ? TUFFMacTheme.accentColor
                                         : .primary)
                        .accessibilityIdentifier("navigation.\(item.rawValue)")
                        .accessibilityValue(navigation.destination == item
                                            ? "Selected" : "")
                        .accessibilityAddTraits(
                            navigation.destination == item ? .isSelected : [])
                }
            }
            // Always, not only on the Chat screen. Chats are the thing the app
            // is for; hiding them behind the destination meant leaving Chat to
            // check a download made every conversation disappear, and coming
            // back was a two-step trip.
            Section {
                Button {
                    navigation.select(.chat)
                    model.clearOutput()
                } label: {
                    Label("New Chat", systemImage: "square.and.pencil")
                }
                .disabled(!canStartNewChat)
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
        .listStyle(.sidebar)
        .safeAreaInset(edge: .top, spacing: 0) { brandHeader }
        .safeAreaInset(edge: .bottom, spacing: 0) { selectedModelFooter }
        .navigationTitle("TUFF")
    }

    private func destinationRowBackground(for item: AppDestination) -> Color {
        navigation.destination == item
            ? TUFFMacTheme.accentColor.opacity(0.14)
            : .clear
    }

    /// Whether the New Chat button does anything. A chat that has not been
    /// written to yet is already a new chat.
    private var canStartNewChat: Bool {
        guard !model.isRunning else { return false }
        guard navigation.destination == .chat else { return true }
        return model.hasOutputTranscript
            || model.conversationStore.selectedConversation == nil
    }

    private func isOpen(_ conversation: AppConversationRecord) -> Bool {
        navigation.destination == .chat
            && model.conversationStore.selectedConversationID == conversation.id
    }

    /// The open chat stays enabled even though clicking it changes nothing:
    /// disabling it would grey out the one row that is meant to look selected.
    private func canOpen(_ conversation: AppConversationRecord) -> Bool {
        isOpen(conversation) || model.canSelectConversation(conversation)
    }

    private func conversationRow(_ conversation: AppConversationRecord) -> some View {
        Button {
            // Opening a chat means showing it. Selecting one from Models or
            // Settings used to load it silently behind whatever screen was up,
            // so the click looked like it had done nothing.
            navigation.select(.chat)
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
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        // The same treatment the destinations get, rather than a 6pt dot: with
        // the list always on screen, which chat is open has to be readable at a
        // glance.
        .listRowBackground(isOpen(conversation)
                           ? TUFFMacTheme.accentColor.opacity(0.14)
                           : Color.clear)
        .foregroundStyle(isOpen(conversation)
                         ? TUFFMacTheme.accentColor
                         : .primary)
        .disabled(!canOpen(conversation))
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
        .accessibilityValue(isOpen(conversation) ? "Selected" : "")
        .accessibilityHint("Opens this saved chat")
        .accessibilityAddTraits(isOpen(conversation) ? .isSelected : [])
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
                    TUFFMacTheme.surfaceStyle(
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
        // Windowed, the title bar sits above this and supplies the top margin.
        // Full screen removes it, so the same margin is added back explicitly —
        // measured from AppKit rather than guessed, so it stays right if the
        // system metric ever changes.
        .padding(.top, 10 + (isFullScreen ? Self.titleBarHeight : 0))
        .padding(.bottom, 10)
    }

    /// The gap the window leaves between its own edge and the sidebar panel.
    /// Matched by hand to the inset a windowed frame gives the panel on its
    /// other three edges, because full screen supplies none of its own.
    private static let sidebarEdgeInset: CGFloat = 5

    /// Height of a standard title bar: the difference between a titled
    /// window's frame and its content rect.
    private static let titleBarHeight: CGFloat = {
        let frame = NSWindow.frameRect(
            forContentRect: .zero, styleMask: [.titled])
        return max(0, frame.height)
    }()

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
                                         ? TUFFMacTheme.accentColor
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
        .background(TUFFMacTheme.surfaceStyle(
            reduceTransparency: reduceTransparency,
            material: .thin))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Current model")
        .accessibilityValue("\(model.selectedDescriptor.displayName), \(model.presentation.label)")
        .accessibilityHint("Opens Models")
    }

}

/// Reports whether the hosting window is in full screen.
///
/// The style mask is read on attach as well as on the notifications: a window
/// restored into full screen never posts one, and the sidebar would sit against
/// the top of the screen until the user toggled full screen off and on again.
private struct WindowFullScreenReader: NSViewRepresentable {
    @Binding var isFullScreen: Bool

    func makeNSView(context: Context) -> NSView {
        let view = ObservingView()
        view.onChange = { isFullScreen = $0 }
        return view
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        (nsView as? ObservingView)?.onChange = { isFullScreen = $0 }
    }

    final class ObservingView: NSView {
        var onChange: ((Bool) -> Void)?

        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            NotificationCenter.default.removeObserver(self)
            guard let window else { return }
            for name in [NSWindow.didEnterFullScreenNotification,
                         NSWindow.didExitFullScreenNotification,
                         // A window restored into full screen posts neither of
                         // the above; it does become key, and the style mask is
                         // correct by then.
                         NSWindow.didBecomeKeyNotification] {
                NotificationCenter.default.addObserver(
                    self, selector: #selector(windowFullScreenChanged),
                    name: name, object: window)
            }
            report()
        }

        @objc private func windowFullScreenChanged() { report() }

        private func report() {
            let value = window?.styleMask.contains(.fullScreen) ?? false
            // Deferred: this runs inside AppKit's view-hierarchy change, which
            // SwiftUI counts as its own update pass.
            DispatchQueue.main.async { [weak self] in self?.onChange?(value) }
        }
    }
}

struct AppWorkspaceView: View {
    let destination: AppDestination
    let model: AppModel
    let serverController: AppServerController
    let updateController: AppUpdateController

    var body: some View {
        Group {
            switch destination {
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
        .accessibilityIdentifier("workspace.\(destination.rawValue)")
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
            ErrorBanner(model: model)
            // Not just an empty prompt box: a message that is so far only an
            // attached file is still a message being written, and offering
            // "try one of these instead" over it reads as a dead end.
            if !model.hasComposedInput && model.showPromptExamples && !model.isRunning {
                PromptExamplesView { preset in model.promptText = preset.prompt }
            }
            ModelActionBanner(model: model)
            PromptComposerView(model: model)
        }
        .padding(.horizontal, 20)
        .padding(.bottom, 16)
        .animation(.smooth(duration: 0.2), value: model.hasComposedInput)
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
                        .fill(TUFFMacTheme.accentColor.opacity(0.12))
                    Image(systemName: "network")
                        .font(.title2)
                        .foregroundStyle(TUFFMacTheme.accentColor)
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
        TUFFMacTheme.surfaceStyle(
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
                    ? TUFFMacTheme.accentColor : .secondary)
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
                subtitle: "App behavior and per-model generation defaults.")
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
