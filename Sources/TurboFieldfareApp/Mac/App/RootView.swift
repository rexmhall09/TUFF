import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct RootView: View {
    let model: AppModel
    @State private var destination: AppDestination? = .chat

    var body: some View {
        NavigationSplitView {
            sidebar
                .navigationSplitViewColumnWidth(min: 176, ideal: 208, max: 240)
        } detail: {
            destinationView
        }
        .navigationSplitViewStyle(.balanced)
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
        }
        .listStyle(.sidebar)
        .safeAreaInset(edge: .top, spacing: 0) { brandHeader }
        .safeAreaInset(edge: .bottom, spacing: 0) { selectedModelFooter }
        .navigationTitle("TUFF")
    }

    private var brandHeader: some View {
        HStack(spacing: 10) {
            Image(systemName: "triangle.fill")
                .font(.title3)
                .foregroundStyle(TurboFieldfareMacTheme.accentColor)
                .frame(width: 28, height: 28)
                .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 8))
                .accessibilityHidden(true)
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
        .background(.thinMaterial)
        .accessibilityElement(children: .combine)
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
            .frame(minWidth: 520, maxWidth: .infinity, maxHeight: .infinity)
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
            .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 16))
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
            Text(title).font(.largeTitle.bold())
            Text(subtitle)
                .font(.callout)
                .foregroundStyle(.secondary)
        }
        .accessibilityElement(children: .combine)
    }
}

private struct ConversationChromeHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}
