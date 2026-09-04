import SwiftUI

struct HUDMetricView: View {
    let value: String
    let label: String
    var animated = true

    var body: some View {
        VStack(spacing: 1) {
            Text(value)
                .appFont(.system(.callout, design: .rounded).weight(.semibold))
                .monospacedDigit()
                .contentTransition(animated ? .numericText() : .identity)
                .animation(animated ? .snappy(duration: 0.25) : nil, value: value)
            Text(label)
                .appFont(.caption2)
                .textCase(.uppercase)
                .foregroundStyle(.secondary)
        }
        .frame(minWidth: 56)
    }
}
