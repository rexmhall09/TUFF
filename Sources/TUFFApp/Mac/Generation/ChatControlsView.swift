import TUFFEngine
import TUFFAppCore
import TUFFMacPresentation
import SwiftUI

/// Model and reasoning pickers, sized to sit inside the prompt bar.
///
/// These read as quiet capsules rather than system pop-up buttons: they are part
/// of the composer, and the send button is the only emphasized control there.
struct ChatControlsView: View {
    @Bindable var model: AppModel

    var body: some View {
        HStack(spacing: 6) {
            // Only what is on this Mac. Offering the rest of the catalogue here
            // meant picking one replaced the conversation with the downloader,
            // which is not what choosing a model inside a chat should do — the
            // Models screen is where downloads live.
            Picker("Model", selection: modelSelection) {
                ForEach(model.installedInstalls) { install in
                    Text(install.descriptor.shortName).tag(install.id)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .fixedSize()
            .buttonStyle(.borderless)
            .padding(.horizontal, 8)
            .frame(height: 28)
            .background(pillBackground)
            .help("Model for this chat")
            .accessibilityLabel("Chat model")

            reasoningControl
        }
        .font(.callout)
        .disabled(model.isRunning || model.loadState.isLoading)
    }

    private var pillBackground: some View {
        Capsule()
            .fill(Color.primary.opacity(0.06))
            .overlay {
                Capsule().stroke(Color.primary.opacity(0.08), lineWidth: 0.5)
            }
    }

    @ViewBuilder
    private var reasoningControl: some View {
        switch model.selectedDescriptor.reasoningControl {
        case .toggle, .toggleWithPreservation:
            Picker("Thinking", selection: $model.reasoning) {
                Text("Thinking Off").tag(ChatReasoning.off)
                Text("Thinking On").tag(ChatReasoning.on)
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .fixedSize()
            .buttonStyle(.borderless)
            .padding(.horizontal, 8)
            .frame(height: 28)
            .background(pillBackground)
            .accessibilityLabel("Thinking")
        case .graded:
            Picker("Reasoning", selection: $model.reasoningEffort) {
                Text("Reasoning Low").tag(GPTOSSReasoningEffort.low)
                Text("Reasoning Medium").tag(GPTOSSReasoningEffort.medium)
                Text("Reasoning High").tag(GPTOSSReasoningEffort.high)
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .fixedSize()
            .buttonStyle(.borderless)
            .padding(.horizontal, 8)
            .frame(height: 28)
            .background(pillBackground)
            .accessibilityLabel("Reasoning effort")
        case .alwaysOn:
            Text("Thinking On")
                .foregroundStyle(TUFFMacTheme.accentColor)
                .padding(.horizontal, 10)
                .frame(height: 28)
                .background(pillBackground)
                .help("This model always reasons before answering")
                .accessibilityLabel("Thinking always on")
        case nil:
            EmptyView()
        }
    }

    private var modelSelection: Binding<String> {
        Binding {
            model.selectedModelID
        } set: { id in
            guard let install = model.installedInstalls.first(where: { $0.id == id })
            else { return }
            model.selectModel(install)
        }
    }
}
