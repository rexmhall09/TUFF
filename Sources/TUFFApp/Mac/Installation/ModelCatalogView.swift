import TUFFAppCore
import TUFFMacPresentation
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
    var showsAddOns: Bool = true

    var body: some View {
        VStack(alignment: .leading, spacing: compact ? 8 : 14) {
            // Grouped by how the model is run, because that is the difference
            // that decides whether it fits this Mac. A streaming model's size
            // is governed by disk and a resident one's by memory, so listing a
            // 61 GiB model beside a 4 GiB one with nothing to separate them
            // made the catalogue look like a straight size ranking.
            ForEach(ModelCatalogGroup.allCases) { group in
                let installs = model.installs.filter {
                    group.contains($0.descriptor)
                }
                if !installs.isEmpty {
                    if !compact { groupHeader(group) }
                    ForEach(installs) { install in
                        ModelCatalogRow(
                            model: model,
                            install: install,
                            compact: compact,
                            showsAddOns: showsAddOns)
                    }
                }
            }
        }
    }

    private func groupHeader(_ group: ModelCatalogGroup) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(group.title)
                .font(.headline)
            Text(group.subtitle)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.top, 4)
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(.isHeader)
    }
}

/// How a model is run, which is what decides what it needs from the Mac.
enum ModelCatalogGroup: String, CaseIterable, Identifiable {
    case streaming
    case resident

    var id: String { rawValue }

    func contains(_ descriptor: AppModelInstallDescriptor) -> Bool {
        switch self {
        case .streaming: descriptor.usesExpertCache
        case .resident: !descriptor.usesExpertCache
        }
    }

    var title: String {
        switch self {
        case .streaming: "Streamed from SSD"
        case .resident: "Held in memory"
        }
    }

    var subtitle: String {
        switch self {
        case .streaming:
            "Mixture-of-experts models. TUFF keeps the shared weights resident "
                + "and reads each token's experts from disk, so how large a "
                + "model you can run is governed by free storage rather than by "
                + "RAM. Decode is slower in exchange."
        case .resident:
            "Dense models, loaded whole. Every weight is used for every token, "
                + "so these are the fastest to decode and are bounded by memory "
                + "rather than by disk."
        }
    }
}

struct ModelCatalogRow: View {
    let model: AppModel
    let install: ModelInstallCoordinator
    var compact: Bool
    var showsAddOns: Bool

    @State private var showingDiscardConfirmation = false
    @State private var showingVisionDiscardConfirmation = false
    @State private var showingVisionRemoveConfirmation = false

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
            if showsAddOns && install.descriptor.supportsImageInput {
                visionSupport
            }
        }
        .padding(compact ? 12 : 16)
        .background {
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(nsColor: .controlBackgroundColor))
                .overlay {
                    RoundedRectangle(cornerRadius: 16)
                        .stroke(isSelected
                                ? TUFFMacTheme.accentColor.opacity(0.65)
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
        .confirmationDialog(
            "Discard the saved \(visionDisplayName) download?",
            isPresented: $showingVisionDiscardConfirmation,
            titleVisibility: .visible
        ) {
            Button("Discard Download", role: .destructive) {
                model.discardVisionPackDownload(for: install)
            }
            Button("Keep Download", role: .cancel) {}
        } message: {
            Text("Downloaded add-on ranges will be removed. The text model is untouched.")
        }
        .confirmationDialog(
            "Remove \(visionDisplayName)?",
            isPresented: $showingVisionRemoveConfirmation,
            titleVisibility: .visible
        ) {
            Button("Remove Image Support", role: .destructive) {
                model.removeVisionPack(for: install)
            }
            Button("Keep Image Support", role: .cancel) {}
        } message: {
            Text("Text generation will keep working. Restoring image input requires downloading this add-on again.")
        }
        .saturation(hardwareEligibility.isCompatible ? 1 : 0)
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
                    .background(TUFFMacTheme.accentColor.opacity(0.18),
                                in: Capsule())
            }
            if install.descriptor.isRecommended {
                Text("Recommended")
                    .font(.caption2.weight(.semibold))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .foregroundStyle(TUFFMacTheme.accentColor)
                    .background(TUFFMacTheme.accentColor.opacity(0.12),
                                in: Capsule())
            }
            Spacer(minLength: 8)
            statusBadge
        }
    }

    private var visionDisplayName: String {
        guard install.descriptor.supportsImageInput else { return "image support" }
        return visionDescriptor.displayName
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
                .foregroundStyle(TUFFMacTheme.accentColor)
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
            StorageRow(
                label: "Unified memory",
                value: "\(MetricFormat.memory(hardwareEligibility.minimumUnifiedMemoryBytes)) min")
            if hardwareEligibility.isCompatible {
                Label("Compatible with this Mac", systemImage: "checkmark.circle")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else if let explanation = hardwareEligibility.explanation {
                Label(explanation, systemImage: "memorychip")
                    .font(.caption)
                    .foregroundStyle(.orange)
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
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

    private var blockedPathHelp: String {
        let entries = install.blockedInstallEntries
        let names = entries.map { ($0 as NSString).lastPathComponent }.joined(separator: ", ")
        return "Delete \(names) and download again. A symlink is unlinked; "
            + "whatever it points at is left alone."
    }

    @ViewBuilder
    private var message: some View {
        if install.isInstallPathBlocked {
            Label("Something is already in the way at this location. "
                  + "Remove and Retry clears it and downloads again.",
                  systemImage: "exclamationmark.triangle.fill")
                .font(.caption)
                .foregroundStyle(.orange)
                .fixedSize(horizontal: false, vertical: true)
        }
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
                if install.isInstallPathBlocked {
                    Button("Remove and Retry", role: .destructive) {
                        install.clearBlockedInstallPath()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!install.canClearInstallPath)
                    .help(blockedPathHelp)
                } else if !install.isInstalled {
                    Button(install.hasPartialDownload ? "Resume" : "Download") {
                        model.installModel(install)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!model.canInstallModel(install))
                    .help(downloadHelp)
                }
            }
            Spacer(minLength: 8)
            if !isSelected {
                Button("Use This Model") { model.selectModel(install) }
                    .disabled(!model.canSelectModel(install))
                    .help(selectionHelp)
            }
        }
        .controlSize(compact ? .small : .regular)
    }

    private var visionSupport: some View {
        VStack(alignment: .leading, spacing: 8) {
            Divider()
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Label(visionDescriptor.displayName, systemImage: "photo")
                    .font(.caption.weight(.semibold))
                Spacer(minLength: 8)
                Text(visionStatusText)
                    .font(.caption2)
                    .foregroundStyle(visionStatusColor)
            }
            Text("Adds local image understanding without reinstalling or changing the text model.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            if model.isVisionInstallTarget(install),
               let fraction = model.visionInstallProgressFraction {
                ProgressView(value: fraction)
                    .accessibilityLabel("\(visionDescriptor.displayName) download")
                    .accessibilityValue(Text(MetricFormat.percent(fraction * 100)))
                HStack {
                    Text(MetricFormat.percent(fraction * 100))
                    Spacer()
                    if let eta = model.visionInstallETAText { Text(eta) }
                }
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
            }

            visionMessage

            HStack(alignment: .firstTextBaseline, spacing: 8) {
                visionAction
                Spacer(minLength: 8)
                if !visionIsInstalled && !visionIsPrepared {
                    Text(MetricFormat.storage(visionDescriptor.approximateDownloadBytes))
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }
            .controlSize(compact ? .small : .regular)
        }
    }

    @ViewBuilder
    private var visionAction: some View {
        if model.isVisionInstallTarget(install), model.isInstallingVisionPack {
            Button("Cancel \(visionDescriptor.displayName)") {
                model.cancelVisionInstall()
            }
            .disabled(!model.canCancelVisionInstall)
        } else if visionIsInstalled {
            Label("Image support installed", systemImage: "checkmark.circle.fill")
                .font(.caption)
                .foregroundStyle(.green)
            Button("Remove", role: .destructive) {
                showingVisionRemoveConfirmation = true
            }
            .disabled(!model.canRemoveVisionPack(for: install))
        } else if visionIsPrepared {
            if isSelected {
                if model.loadState.isReady {
                    Button("Unload to Activate \(visionDescriptor.displayName)") {
                        model.unloadModel()
                    }
                    .disabled(!model.canUnloadModel)
                } else {
                    Button("Activate \(visionDescriptor.displayName)") {
                        model.activateVisionPack()
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!model.canActivateVisionPack)
                }
            } else {
                Button("Use This Model to Activate") {
                    model.selectModel(install)
                }
                .disabled(!model.canSelectModel(install))
            }
        } else {
            if model.hasPartialVisionPackDownload(for: install) {
                Button("Discard", role: .destructive) {
                    showingVisionDiscardConfirmation = true
                }
                .disabled(!model.canDiscardVisionPackDownload(for: install))
            }
            Button(model.visionDownloadButtonLabel(for: install)) {
                model.installVisionPack(for: install)
            }
            .buttonStyle(.bordered)
            .disabled(!model.canInstallVisionPack(for: install))
            .help("Optional download for this model only. The text model stays installed.")
        }
    }

    @ViewBuilder
    private var visionMessage: some View {
        if !install.isInstalled {
            Label("Install the text model first", systemImage: "arrow.down.circle")
                .foregroundStyle(.secondary)
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

    private var visionDescriptor: AppModelInstallDescriptor {
        model.visionInstallDescriptor(for: install)
    }

    private var hardwareEligibility: AppModelHardwareEligibility {
        model.hardwareEligibility(for: install)
    }

    private var downloadHelp: String {
        if let explanation = hardwareEligibility.explanation { return explanation }
        if let requirement = install.requirement, !requirement.canInstall {
            return "Requires \(MetricFormat.storage(requirement.shortfallBytes)) more disk space."
        }
        return "Download and install \(install.descriptor.displayName)"
    }

    private var selectionHelp: String {
        if let explanation = hardwareEligibility.explanation { return explanation }
        return install.isInstalled
            ? "Load and generate with this model"
            : "Make this the model the app loads once it is installed"
    }

    private var visionIsInstalled: Bool {
        model.isVisionPackInstalled(for: install)
    }

    private var visionIsPrepared: Bool {
        guard model.isVisionInstallTarget(install) else { return false }
        if case .readyToActivate = model.visionInstallState { return true }
        return false
    }

    private var visionStatusText: String {
        guard model.isVisionRuntimeSupported else { return "Requires M2 or newer" }
        if model.isVisionInstallTarget(install) {
            if model.visionInstallState != .idle {
                return model.visionInstallPhaseLabel
            }
        }
        switch model.visionInstallationStatus(for: install) {
        case .complete: return "Installed"
        case .partial: return "Needs repair"
        case .missing: return "Optional"
        case .unsupportedLayout: return "Unavailable"
        }
    }

    private var visionStatusColor: Color {
        if visionIsInstalled { return .green }
        if case .partial = model.visionInstallationStatus(for: install) {
            return .orange
        }
        return .secondary
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
