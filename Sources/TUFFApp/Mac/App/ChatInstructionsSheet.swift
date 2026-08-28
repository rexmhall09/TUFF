import TUFFAppCore
import TUFFMacPresentation
import SwiftUI

/// Instructions for one chat, overriding the model's own.
///
/// Two states that look alike and are not: following the model's default, and
/// deliberately having none. A chat that has cleared its instructions must not
/// silently pick the model's back up, so "no instructions" is stored as an
/// empty override rather than as an absent one.
struct ChatInstructionsSheet: View {
    let model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var text = ""
    @State private var overrides = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Chat Instructions")
                .font(.title3.weight(.semibold))

            Toggle("Use instructions just for this chat", isOn: $overrides)
                .toggleStyle(.switch)

            if overrides {
                TextEditor(text: $text)
                    .font(.body)
                    .frame(minHeight: 150)
                    .overlay(alignment: .topLeading) {
                        if text.isEmpty {
                            Text("Leave empty to give this chat no instructions "
                                 + "at all.")
                                .font(.body)
                                .foregroundStyle(.tertiary)
                                .padding(.leading, 5)
                                .padding(.top, 8)
                                .allowsHitTesting(false)
                        }
                    }
                    .accessibilityLabel("Chat instructions")
            } else {
                modelDefault
            }

            Text("Sent as a system message ahead of the conversation. It applies "
                 + "from the next message; answers already given were not "
                 + "written with it.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { dismiss() }
                Button("Save") { save() }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(width: 460)
        .onAppear {
            overrides = model.conversationSystemPrompt != nil
            text = model.conversationSystemPrompt ?? model.modelSystemPrompt
        }
    }

    private var modelDefault: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Following \(model.selectedDescriptor.shortName)")
                .font(.caption.weight(.semibold))
                .foregroundStyle(TUFFMacTheme.accentColor)
            ScrollView {
                Text(model.modelSystemPrompt.isEmpty
                     ? "This model has no instructions set."
                     : model.modelSystemPrompt)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
            }
            .frame(height: 110)
        }
    }

    private func save() {
        model.setConversationSystemPrompt(overrides ? text : nil)
        dismiss()
    }
}
