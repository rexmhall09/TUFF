import TUFFAppCore
import TUFFMacPresentation
import SwiftUI

struct ModelActionBanner: View {
    @Bindable var model: AppModel

    var body: some View {
        if let action = model.presentation.conversationAction,
           model.hasOutputTranscript {
            banner(tint: iconColor(for: action)) {
                Image(systemName: iconName(for: action))
                    .foregroundStyle(iconColor(for: action))
                    .accessibilityHidden(true)
                Text(message(for: action))
                    .font(.callout)
                    .lineLimit(2)
                Spacer(minLength: 8)
                Button(buttonTitle(for: action)) {
                    model.perform(action)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
            }
        } else if let notice = model.presentation.conversationNotice {
            // Shown whether or not anything has been said yet: this is the
            // state where the composer refuses and there is no button to
            // press, so the reason has to be on screen from the start.
            banner(tint: .orange) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(.orange)
                    .accessibilityHidden(true)
                Text(notice)
                    .font(.callout)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 8)
            }
        }
    }

    @ViewBuilder
    private func banner(
        tint: Color,
        @ViewBuilder content: () -> some View
    ) -> some View {
        HStack(spacing: 10) {
            content()
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
        .background {
            RoundedRectangle(cornerRadius: 14)
                .fill(Color(nsColor: .controlBackgroundColor))
                .overlay {
                    RoundedRectangle(cornerRadius: 14)
                        .stroke(tint.opacity(0.5), lineWidth: 1)
                }
        }
        .accessibilityElement(children: .contain)
        .transition(.move(edge: .bottom).combined(with: .opacity))
    }

    private func message(for action: AppModelAction) -> String {
        switch action {
        case .load:
            return "The model is unloaded. Load it to generate."
        case .retryLoad:
            return model.presentation.detail ?? "The model could not be loaded."
        case .reload:
            return "Settings changed. Reload the model to continue."
        case .install, .cancelInstall, .cancelLoad, .unload:
            return model.presentation.label
        }
    }

    private func buttonTitle(for action: AppModelAction) -> String {
        switch action {
        case .load: return "Load Model"
        case .retryLoad: return "Retry Load"
        case .reload: return "Reload Model"
        case .install: return "Install"
        case .cancelInstall: return "Cancel"
        case .cancelLoad: return "Cancel Load"
        case .unload: return "Unload Model"
        }
    }

    private func iconName(for action: AppModelAction) -> String {
        switch action {
        case .retryLoad: return "exclamationmark.triangle.fill"
        case .reload: return "arrow.clockwise.circle.fill"
        case .load, .install, .cancelInstall, .cancelLoad, .unload:
            return "externaldrive.fill"
        }
    }

    /// An unloaded model is a step to take, not a fault: it wears the app's
    /// purple like every other "do this next" control. Orange is kept for the
    /// things that really are wrong, and red for a load that failed.
    private func iconColor(for action: AppModelAction) -> Color {
        action == .retryLoad ? .red : TUFFMacTheme.accentColor
    }
}
