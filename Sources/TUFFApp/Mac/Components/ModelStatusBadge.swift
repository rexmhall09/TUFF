import TUFFAppCore
import TUFFMacPresentation
import SwiftUI

struct ModelStatusBadge: View {
    let model: AppModel

    var body: some View {
        HStack(spacing: 6) {
            statusDot
            Text(model.installDescriptor.displayName)
                .appFont(.callout.weight(.semibold))
                .lineLimit(1)
                .help(model.installDescriptor.repoID)
                .accessibilityLabel("Model")
                .accessibilityValue(model.installDescriptor.repoID)
        }
    }

    @ViewBuilder
    private var statusDot: some View {
        switch model.presentation.severity {
        case .neutral: dot(.gray)
        // Working, not failing: the same purple the phase label and the send
        // button use, rather than a caution colour for an ordinary load.
        case .active: dot(TUFFMacTheme.accentColor)
        case .warning: dot(.orange)
        case .success: dot(.green)
        case .error: dot(.red)
        }
    }

    private func dot(_ color: Color) -> some View {
        Circle().fill(color).frame(width: 8, height: 8).accessibilityHidden(true)
    }
}
