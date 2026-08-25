import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct RootView: View {
    let model: AppModel
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @State private var destination: AppDestination? = .chat
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
            AppNavigationAction { destination = $0 })
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
        .animation(.smooth(duration: 0.25), value: destination)
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
        List(selection: $destination) {
            Section {
                ForEach(AppDestination.allCases) { item in
                    Label(item.title, systemImage: item.systemImage)
                        .tag(Optional(item))
                        .accessibilityIdentifier("navigation.\(item.rawValue)")
                }
            }
            if destination == .chat {
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
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(TurboFieldfareMacTheme.surfaceStyle(
            reduceTransparency: reduceTransparency,
            material: .thin))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Current model")
        .accessibilityValue("\(model.selectedDescriptor.displayName), \(model.presentation.label)")
    }

    @ViewBuilder
    private var destinationView: some View {
        switch destination ?? .chat {
        case .chat:
            ChatWorkspaceView(model: model)
        case .models:
            ModelsWorkspaceView(model: model)
        case .server:
            ServerWorkspaceView(model: model)
        case .settings:
            SettingsWorkspaceView(model: model)
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
                ModelCatalogView(model: model)
            }
            .frame(maxWidth: 860, alignment: .leading)
            .padding(28)
        }
        .background(Color(nsColor: .windowBackgroundColor))
    }
}

private struct ServerWorkspaceView: View {
    let model: AppModel
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            WorkspaceTitle(
                title: "Server",
                subtitle: "Use TUFF through an OpenAI-compatible loopback endpoint.")
            HStack(spacing: 12) {
                Image(systemName: "network")
                    .font(.title2)
                    .foregroundStyle(TurboFieldfareMacTheme.accentColor)
                VStack(alignment: .leading, spacing: 4) {
                    Text("Local server").font(.headline)
                    Text("127.0.0.1 · \(serverStatus)")
                        .font(.callout.monospaced())
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding(18)
            .background(
                TurboFieldfareMacTheme.surfaceStyle(
                    reduceTransparency: reduceTransparency),
                in: RoundedRectangle(cornerRadius: 16))
            .accessibilityElement(children: .combine)
            Spacer()
        }
        .padding(28)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

    private var serverStatus: String {
        switch model.serverStore.status {
        case .stopped: "Stopped"
        case .starting: "Starting"
        case .running: "Running"
        case .stopping: "Stopping"
        case .failed: "Error"
        }
    }
}

private struct SettingsWorkspaceView: View {
    let model: AppModel

    var body: some View {
        VStack(spacing: 0) {
            WorkspaceTitle(
                title: "Settings",
                subtitle: "Tune model, memory, and generation behavior.")
                .padding(.horizontal, 28)
                .padding(.top, 24)
                .frame(maxWidth: .infinity, alignment: .leading)
            InspectorView(model: model)
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
