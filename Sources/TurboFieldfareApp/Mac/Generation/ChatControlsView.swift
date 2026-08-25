import TurboFieldfare
import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct ChatControlsView: View {
    @Bindable var model: AppModel

    var body: some View {
        HStack(spacing: 10) {
            Picker("Model", selection: modelSelection) {
                ForEach(model.installs) { install in
                    Text(install.descriptor.shortName).tag(install.id)
                }
            }
            .labelsHidden()
            .pickerStyle(.menu)
            .fixedSize()
            .help("Model for this chat")
            .accessibilityLabel("Chat model")

            reasoningControl

            Spacer(minLength: 0)

            if model.conversationTurnCount > 0 {
                Text("\(model.conversationTurnCount) turn\(model.conversationTurnCount == 1 ? "" : "s")")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .monospacedDigit()
            }
        }
        .controlSize(.small)
        .padding(.horizontal, 2)
        .disabled(model.isRunning || model.loadState.isLoading)
    }

    @ViewBuilder
    private var reasoningControl: some View {
        switch model.selectedDescriptor.reasoningControl {
        case .toggle, .toggleWithPreservation:
            Picker("Thinking", selection: $model.reasoning) {
                Text("Thinking Off").tag(ChatReasoning.off)
                Text("Thinking On").tag(ChatReasoning.on)
            }
            .pickerStyle(.menu)
            .fixedSize()
            .accessibilityLabel("Thinking")
        case .graded:
            Picker("Reasoning", selection: $model.reasoningEffort) {
                Text("Low").tag(GPTOSSReasoningEffort.low)
                Text("Medium").tag(GPTOSSReasoningEffort.medium)
                Text("High").tag(GPTOSSReasoningEffort.high)
            }
            .pickerStyle(.menu)
            .fixedSize()
            .accessibilityLabel("Reasoning effort")
        case nil:
            EmptyView()
        }
    }

    private var modelSelection: Binding<String> {
        Binding {
            model.selectedModelID
        } set: { id in
            guard let install = model.installs.first(where: { $0.id == id }) else { return }
            model.selectModel(install)
        }
    }
}
