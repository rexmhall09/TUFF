import TurboFieldfareAppCore
import TurboFieldfareMacPresentation
import SwiftUI

/// The model catalog: every supported model, each with its own download.
///
/// A row's download controls act on that model alone, so a download can run
/// for a model the app is not currently using — including while another model
/// is loaded and generating. Selection is a separate action from downloading:
/// picking a model decides which one loads, and never starts or stops a
/// transfer.
struct ModelCatalogView: View {
    let model: AppModel
    /// Compact drops the per-model summary and storage detail for the
    /// inspector, where the row sits in a narrow column.
    var compact: Bool = false

    var body: some View {
        VStack(spacing: compact ? 8 : 14) {
            ForEach(model.installs) { install in
                ModelCatalogRow(model: model, install: install, compact: compact)
            }
        }
    }
}

struct ModelCatalogRow: View {
    let model: AppModel
    let install: ModelInstallCoordinator
    var compact: Bool

    @State private var showingDiscardConfirmation = false

    private var isSelected: Bool { install.id == model.selectedModelID }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            header
            if !compact {
                Text(install.descriptor.summary)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                storage
            }
            if install.isInstalling {
                progress
            }
            message
            actions
        }
        .padding(compact ? 12 : 16)
        .background {
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(nsColor: .controlBackgroundColor))
                .overlay {
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(isSelected
                                ? TurboFieldfareMacTheme.accentColor.opacity(0.65)
                                : Color(nsColor: .separatorColor).opacity(0.5),
                                lineWidth: isSelected ? 1.5 : 0.5)
                }
        }
        .confirmationDialog(
            "Discard the saved \(install.descriptor.displayName) download?",
            isPresented: $showingDiscardConfirmation,
            titleVisibility: .visible
        ) {
            Button("Discard Download", role: .destructive) { install.discard() }
            Button("Keep Download", role: .cancel) {}
        } message: {
            Text("Downloaded ranges for this model will be removed. "
                 + "Other models and any installed model are untouched.")
        }
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            Text(install.descriptor.displayName)
                .font(compact ? .callout.weight(.medium) : .headline)
            if isSelected {
                Text("Selected")
                    .font(.caption2.weight(.semibold))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(TurboFieldfareMacTheme.accentColor.opacity(0.18),
                                in: Capsule())
            }
            Spacer(minLength: 8)
            statusBadge
        }
    }

    @ViewBuilder
    private var statusBadge: some View {
        switch status {
        case .installed:
            Label("Installed", systemImage: "checkmark.circle.fill")
                .foregroundStyle(.green)
                .labelStyle(.titleAndIcon)
                .font(.caption)
        case .downloading:
            Label("Downloading", systemImage: "arrow.down.circle")
                .foregroundStyle(TurboFieldfareMacTheme.accentColor)
                .font(.caption)
        case .paused:
            Label("Paused", systemImage: "pause.circle")
                .foregroundStyle(.secondary)
                .font(.caption)
        case .notInstalled:
            Text(MetricFormat.storage(install.descriptor.approximateDownloadBytes))
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
        }
    }

    private enum RowStatus { case installed, downloading, paused, notInstalled }

    private var status: RowStatus {
        if install.isInstalled { return .installed }
        if install.isInstalling { return .downloading }
        if install.hasPartialDownload { return .paused }
        return .notInstalled
    }

    private var storage: some View {
        VStack(spacing: 6) {
            StorageRow(label: "Download",
                       value: MetricFormat.storage(
                        install.descriptor.approximateDownloadBytes))
            StorageRow(label: "Installed size",
                       value: MetricFormat.storage(install.descriptor.installedBytes))
            if let requirement = install.requirement, !install.isInstalled {
                StorageRow(label: "Available on this Mac",
                           value: MetricFormat.storage(requirement.availableBytes))
                if !requirement.canInstall {
                    Label("\(MetricFormat.storage(requirement.shortfallBytes)) more is required",
                          systemImage: "exclamationmark.triangle.fill")
                        .font(.caption)
                        .foregroundStyle(.orange)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
    }

    @ViewBuilder
    private var progress: some View {
        VStack(alignment: .leading, spacing: 6) {
            if case .copyingPayload(let reused, let downloaded, let total) = install.state,
               total > 0 {
                let completed = reused + downloaded
                let fraction = min(1, Double(completed) / Double(total))
                ProgressView(value: fraction)
                    .accessibilityLabel("\(install.descriptor.displayName) download")
                    .accessibilityValue(Text(accessibleProgressValue(fraction: fraction)))
                HStack {
                    Text("\(MetricFormat.storage(completed)) of \(MetricFormat.storage(total))")
                    Spacer(minLength: 12)
                    Text(MetricFormat.percent(fraction * 100))
                    if let eta = install.etaText {
                        Text(eta)
                    }
                }
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
                if reused > 0 {
                    Text("Reused \(MetricFormat.storage(reused)) from the saved download")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } else {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(stageLabel).font(.caption).foregroundStyle(.secondary)
                    Spacer()
                }
            }
        }
    }

    @ViewBuilder
    private var message: some View {
        switch install.state {
        case .failed(let text):
            Label(text, systemImage: "exclamationmark.triangle.fill")
                .font(.caption)
                .foregroundStyle(.red)
                .fixedSize(horizontal: false, vertical: true)
        case .recoverable(let text):
            Label(text, systemImage: "exclamationmark.triangle")
                .font(.caption)
                .foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
        default:
            if case .failed(let text) = install.readiness, !install.isInstalled {
                Label(text, systemImage: "exclamationmark.triangle.fill")
                    .font(.caption)
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    private var actions: some View {
        HStack(spacing: 8) {
            if install.isInstalling {
                Button("Cancel") { install.cancel() }
                    .disabled(!install.canCancel)
            } else {
                if install.hasPartialDownload && !install.isInstalled {
                    Button("Discard", role: .destructive) {
                        showingDiscardConfirmation = true
                    }
                    .disabled(!install.canDiscard)
                }
                if !install.isInstalled {
                    Button(install.hasPartialDownload ? "Resume" : "Download") {
                        install.install()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!install.canInstall)
                }
            }
            Spacer(minLength: 8)
            if !isSelected {
                Button("Use This Model") { model.selectModel(install) }
                    .disabled(!model.canSelectModel(install))
                    .help(install.isInstalled
                          ? "Load and generate with this model"
                          : "Make this the model the app loads once it is installed")
            }
        }
        .controlSize(compact ? .small : .regular)
    }

    private var stageLabel: String {
        switch install.state {
        case .checking: return "Checking"
        case .downloadingMetadata: return "Reading the checkpoint index"
        case .planning: return "Planning the repack"
        case .reservingOutput: return "Reserving disk space"
        case .hashingOutput(let file): return "Verifying \(file)"
        case .finalizing: return "Finalizing"
        case .cancelling: return "Cancelling"
        case .discarding: return "Discarding the saved download"
        default: return "Working"
        }
    }

    private func accessibleProgressValue(fraction: Double) -> String {
        let percent = MetricFormat.percent(fraction * 100)
        guard let eta = install.etaText else { return percent }
        return "\(percent), \(eta)"
    }
}

struct StorageRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label).foregroundStyle(.secondary)
            Spacer(minLength: 16)
            Text(value)
                .monospacedDigit()
                .multilineTextAlignment(.trailing)
        }
        .font(.callout)
    }
}
