import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

struct ModelLibraryColumnsView: View {
    let model: AppModel

    var body: some View {
        ViewThatFits(in: .horizontal) {
            HStack(alignment: .top, spacing: 22) {
                modelColumn
                    .frame(minWidth: 390, maxWidth: .infinity, alignment: .top)
                Divider()
                addOnColumn
                    .frame(width: 300, alignment: .top)
            }
            .frame(minWidth: 720)

            VStack(alignment: .leading, spacing: 24) {
                modelColumn
                Divider()
                addOnColumn
            }
        }
    }

    private var modelColumn: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionTitle(
                "Text models",
                subtitle: "Five pinned checkpoints, installed independently.")
            ModelCatalogView(model: model, showsAddOns: false)
        }
    }

    private var addOnColumn: some View {
        VStack(alignment: .leading, spacing: 12) {
            sectionTitle(
                "Add-ons",
                subtitle: "Verified capabilities kept separate from text weights.")
            ForEach(imageCapableInstalls) { install in
                ModelAddOnSummaryCard(model: model, install: install)
            }
        }
    }

    private var imageCapableInstalls: [ModelInstallCoordinator] {
        model.installs.filter(\.descriptor.supportsImageInput)
    }

    private func sectionTitle(_ title: String, subtitle: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(title).font(.title3.weight(.semibold))
            Text(subtitle)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .accessibilityElement(children: .combine)
    }
}

private struct ModelAddOnSummaryCard: View {
    let model: AppModel
    let install: ModelInstallCoordinator

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: "photo.on.rectangle.angled")
                    .font(.title3)
                    .foregroundStyle(TurboFieldfareMacTheme.accentColor)
                    .frame(width: 28, height: 28)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Image input")
                        .font(.headline)
                    Text(install.descriptor.shortName)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 4)
                Text(status.label)
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(status.color)
            }
            Text("Adds local image understanding without changing the text model.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            HStack {
                Text(MetricFormat.storage(descriptor.approximateDownloadBytes))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                Spacer()
                if !install.isInstalled {
                    Text("Install text model first")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
        }
        .padding(14)
        .background(Color(nsColor: .controlBackgroundColor),
                    in: RoundedRectangle(cornerRadius: 14))
        .overlay {
            RoundedRectangle(cornerRadius: 14)
                .stroke(Color(nsColor: .separatorColor).opacity(0.5), lineWidth: 0.5)
        }
        .accessibilityElement(children: .combine)
    }

    private var descriptor: AppModelInstallDescriptor {
        model.visionInstallDescriptor(for: install)
    }

    private var status: (label: String, color: Color) {
        guard model.isVisionRuntimeSupported else {
            return ("Requires M2", .orange)
        }
        switch model.visionInstallationStatus(for: install) {
        case .complete: return ("Installed", .green)
        case .partial: return ("Needs repair", .orange)
        case .missing: return (install.isInstalled ? "Available" : "Optional", .secondary)
        case .unsupportedLayout: return ("Unavailable", .secondary)
        }
    }
}
