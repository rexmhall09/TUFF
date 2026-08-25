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
    @State private var showingDiscardConfirmation = false
    @State private var showingRemoveConfirmation = false

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
            HStack(alignment: .firstTextBaseline) {
                Text(MetricFormat.storage(descriptor.approximateDownloadBytes))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                Spacer()
                Text("Separate download")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            progress
            message
            actions
        }
        .padding(14)
        .background(Color(nsColor: .controlBackgroundColor),
                    in: RoundedRectangle(cornerRadius: 14))
        .overlay {
            RoundedRectangle(cornerRadius: 14)
                .stroke(Color(nsColor: .separatorColor).opacity(0.5), lineWidth: 0.5)
        }
        .confirmationDialog(
            "Discard the saved \(descriptor.displayName) download?",
            isPresented: $showingDiscardConfirmation,
            titleVisibility: .visible
        ) {
            Button("Discard Download", role: .destructive) {
                model.discardVisionPackDownload(for: install)
            }
            Button("Keep Download", role: .cancel) {}
        } message: {
            Text("Downloaded ranges for this add-on will be removed. "
                + "The text model is untouched.")
        }
        .confirmationDialog(
            "Remove \(descriptor.displayName)?",
            isPresented: $showingRemoveConfirmation,
            titleVisibility: .visible
        ) {
            Button("Remove Image Support", role: .destructive) {
                model.removeVisionPack(for: install)
            }
            Button("Keep Image Support", role: .cancel) {}
        } message: {
            Text("Text generation will continue to work. Restoring image input "
                + "requires downloading this add-on again.")
        }
    }

    private var descriptor: AppModelInstallDescriptor {
        model.visionInstallDescriptor(for: install)
    }

    private var status: (label: String, color: Color) {
        guard model.isVisionRuntimeSupported else {
            return ("Requires M2", .orange)
        }
        if model.isVisionInstallTarget(install), model.visionInstallState != .idle {
            return (model.visionInstallPhaseLabel,
                    model.isInstallingVisionPack
                        ? TurboFieldfareMacTheme.accentColor : .secondary)
        }
        switch model.visionInstallationStatus(for: install) {
        case .complete: return ("Installed", .green)
        case .partial: return ("Needs repair", .orange)
        case .missing: return (install.isInstalled ? "Available" : "Optional", .secondary)
        case .unsupportedLayout: return ("Unavailable", .secondary)
        }
    }

    @ViewBuilder
    private var progress: some View {
        if model.isVisionInstallTarget(install), model.isInstallingVisionPack {
            if let fraction = model.visionInstallProgressFraction {
                ProgressView(value: fraction)
                    .accessibilityLabel("\(descriptor.displayName) download")
                    .accessibilityValue(MetricFormat.percent(fraction * 100))
                HStack {
                    Text(MetricFormat.percent(fraction * 100))
                    Spacer()
                    if let eta = model.visionInstallETAText { Text(eta) }
                }
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
            } else {
                ProgressView()
                    .controlSize(.small)
                    .accessibilityLabel(model.visionInstallPhaseLabel)
            }
        }
    }

    @ViewBuilder
    private var message: some View {
        if !install.isInstalled {
            Label("Install the text model first", systemImage: "arrow.down.circle")
                .foregroundStyle(.secondary)
        } else if !model.hardwareEligibility(for: install).isCompatible,
                  let explanation = model.hardwareEligibility(for: install).explanation {
            Label(explanation, systemImage: "memorychip")
                .foregroundStyle(.orange)
        } else if !model.isVisionRuntimeSupported {
            Label("Image input requires an M2 or newer Mac", systemImage: "memorychip")
                .foregroundStyle(.orange)
        } else if model.isVisionInstallTarget(install),
                  case .failed(let text) = model.visionInstallState {
            Label(text, systemImage: "exclamationmark.triangle.fill")
                .foregroundStyle(.red)
        } else if model.isVisionInstallTarget(install),
                  case .recoverable(let text) = model.visionInstallState {
            Label(text, systemImage: "exclamationmark.triangle")
                .foregroundStyle(.orange)
        } else if let requirement = model.visionInstallRequirement(for: install),
                  !requirement.canInstall, !visionIsInstalled {
            Label("Free \(MetricFormat.storage(requirement.shortfallBytes)) more storage.",
                  systemImage: "internaldrive")
                .foregroundStyle(.orange)
        }
    }

    private var actions: some View {
        HStack(spacing: 8) {
            if model.isVisionInstallTarget(install), model.isInstallingVisionPack {
                Button("Cancel") { model.cancelVisionInstall() }
                    .disabled(!model.canCancelVisionInstall)
            } else if visionIsInstalled {
                Button("Remove", role: .destructive) {
                    showingRemoveConfirmation = true
                }
                .disabled(!model.canRemoveVisionPack(for: install))
            } else if visionIsPrepared {
                Button("Discard", role: .destructive) {
                    showingDiscardConfirmation = true
                }
                .disabled(!model.canDiscardVisionPackDownload(for: install))
                activationAction
            } else {
                if model.hasVisionPackDirectory(for: install) {
                    Button("Remove", role: .destructive) {
                        showingRemoveConfirmation = true
                    }
                    .disabled(!model.canRemoveVisionPack(for: install))
                } else if model.hasPartialVisionPackDownload(for: install) {
                    Button("Discard", role: .destructive) {
                        showingDiscardConfirmation = true
                    }
                    .disabled(!model.canDiscardVisionPackDownload(for: install))
                }
                Button(addOnDownloadLabel) {
                    model.installVisionPack(for: install)
                }
                .buttonStyle(.borderedProminent)
                .disabled(!model.canInstallVisionPack(for: install))
            }
            Spacer(minLength: 0)
        }
        .controlSize(.small)
    }

    @ViewBuilder
    private var activationAction: some View {
        if install.id == model.selectedModelID {
            if model.loadState.isReady {
                Button("Unload to Activate") { model.unloadModel() }
                    .disabled(!model.canUnloadModel)
            } else {
                Button("Activate") { model.activateVisionPack() }
                    .buttonStyle(.borderedProminent)
                    .disabled(!model.canActivateVisionPack)
            }
        } else {
            Button("Use to Activate") { model.selectModel(install) }
                .disabled(!model.canSelectModel(install))
        }
    }

    private var visionIsInstalled: Bool {
        model.isVisionPackInstalled(for: install)
    }

    private var visionIsPrepared: Bool {
        guard model.isVisionInstallTarget(install) else { return false }
        if case .readyToActivate = model.visionInstallState { return true }
        return false
    }

    private var addOnDownloadLabel: String {
        if model.hasPartialVisionPackDownload(for: install) { return "Resume" }
        if model.hasVisionPackDirectory(for: install) { return "Repair" }
        return "Download"
    }
}
