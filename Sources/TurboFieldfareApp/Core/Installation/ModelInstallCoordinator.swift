import Foundation
import Observation
import TurboFieldfareRepackCore

/// One model's install: its own directory, installer client, progress, and
/// cancellation.
///
/// Each catalog entry owns a coordinator, so downloads are independent of both
/// each other and of which model is currently selected for loading. Starting a
/// Qwen download does not disturb a Gemma download already in flight, and
/// neither one has to be the selected model to run.
@MainActor
@Observable
public final class ModelInstallCoordinator: Identifiable {
    public let descriptor: AppModelInstallDescriptor

    /// Where this model installs. Mutable because the user can point the
    /// selected model at a directory of their own.
    public private(set) var directoryURL: URL

    public private(set) var state: AppModelInstallState = .idle
    public private(set) var readiness: AppModelInstallReadiness = .checking
    public private(set) var installationStatus: AppModelInstallationStatus
    public private(set) var etaPresentation: DownloadETAPresentation = .hidden
    public private(set) var etaText: String?

    /// Called on the main actor when an install completes, so the owner can
    /// react (clear a stale load, select the freshly installed model).
    public var onInstalled: ((ModelInstallCoordinator) -> Void)?

    public nonisolated var id: String { descriptor.id }

    private let client: any AppModelInstallerClient
    private var task: Task<Void, Never>?
    private var generation: UInt64 = 0
    private var etaEstimator = DownloadETAEstimator()
    private let etaClock: SuspendingClock
    private let etaOrigin: SuspendingClock.Instant

    public init(descriptor: AppModelInstallDescriptor,
                directoryURL: URL,
                client: any AppModelInstallerClient) {
        let directory = directoryURL.standardizedFileURL
        self.descriptor = descriptor
        self.directoryURL = directory
        self.client = client
        // Probed against this coordinator's own descriptor: a catalog entry is
        // responsible for one model, so a directory holding a different
        // supported model is a mismatch to report, not an install to adopt.
        self.installationStatus = AppModelInstallationProbe.status(
            at: directory, descriptor: descriptor)
        let clock = SuspendingClock()
        self.etaClock = clock
        self.etaOrigin = clock.now
        refreshReadiness()
    }

    /// A coordinator for a catalog entry installing to its default location.
    public static func shipped(_ descriptor: AppModelInstallDescriptor)
        -> ModelInstallCoordinator {
        ModelInstallCoordinator(
            descriptor: descriptor,
            directoryURL: AppModelLocation.defaultURL(descriptor: descriptor),
            client: RepackModelInstallerClient(descriptor: descriptor))
    }

    // MARK: - Derived state

    public var isInstalled: Bool { installationStatus == .complete }

    public var isInstalling: Bool { state.isInstalling }

    public var requirement: AppModelInstallRequirement? { readiness.requirement }

    public var canInstall: Bool {
        guard case .ready = readiness else { return false }
        return !isInstalling && !isInstalled
    }

    public var canCancel: Bool { state.canCancel && task != nil }

    public var hasPartialDownload: Bool {
        guard let paths = try? RemoteInstallPaths(outputDirectory: directoryURL.path) else {
            return false
        }
        return FileManager.default.fileExists(atPath: paths.partialDirectory)
            || FileManager.default.fileExists(atPath: paths.checkpointFile)
    }

    public var canDiscard: Bool { hasPartialDownload && !isInstalling }

    // MARK: - Commands

    /// Point this model at a different directory, abandoning any install in
    /// flight against the old one.
    public func setDirectory(_ url: URL) {
        let directory = url.standardizedFileURL
        guard directory != directoryURL else { return }
        cancelInFlight()
        directoryURL = directory
        state = .idle
        refresh()
    }

    public func install() {
        refreshReadiness()
        guard canInstall else { return }
        cancelInFlight()
        resetETA()
        generation &+= 1
        let generation = self.generation
        let outputDirectory = directoryURL
        state = .checking
        task = Task { [weak self, client] in
            do {
                for try await event in client.installDefaultModel(
                    outputDirectory: outputDirectory) {
                    guard let self else { return }
                    self.apply(event, generation: generation)
                }
                self?.finishStream(generation: generation)
            } catch is CancellationError {
                self?.finishCancellation(generation: generation)
            } catch {
                self?.finishFailure(error, generation: generation)
            }
        }
    }

    public func cancel() {
        guard canCancel else { return }
        state = .cancelling
        client.cancel()
    }

    public func discard() {
        guard canDiscard else { return }
        cancelInFlight()
        generation &+= 1
        let generation = self.generation
        let outputDirectory = directoryURL
        state = .discarding
        task = Task { [weak self, client] in
            do {
                try await client.discardPartialInstall(outputDirectory: outputDirectory)
                guard let self, generation == self.generation else { return }
                self.task = nil
                self.state = .idle
                self.refresh()
            } catch {
                self?.finishFailure(error, generation: generation)
            }
        }
    }

    /// Re-probe the directory and recompute free-space readiness.
    public func refresh() {
        installationStatus = AppModelInstallationProbe.status(
            at: directoryURL, descriptor: descriptor)
        refreshReadiness()
    }

    // MARK: - Internals

    private func refreshReadiness() {
        guard !isInstalled else { return }
        readiness = .checking
        do {
            let requirement = try client.checkInstallRequirement(
                outputDirectory: directoryURL)
            readiness = requirement.canInstall
                ? .ready(requirement)
                : .insufficientSpace(requirement)
        } catch {
            readiness = .failed("\(error)")
        }
    }

    private func cancelInFlight() {
        generation &+= 1
        task?.cancel()
        client.cancel()
        task = nil
    }

    private func apply(_ event: AppModelInstallEvent, generation: UInt64) {
        guard generation == self.generation else { return }
        switch event {
        case .checking:
            resetETA()
            state = .checking
        case .downloadingMetadata:
            resetETA()
            state = .downloadingMetadata
        case .planning:
            resetETA()
            state = .planning
        case .reservingOutput:
            resetETA()
            state = .reservingOutput
        case .copyingPayload(let reused, let downloadedThisRun, let total):
            state = .copyingPayload(
                reusedBytes: reused,
                downloadedThisRunBytes: downloadedThisRun,
                totalBytes: total)
            updateETA(
                reusedBytes: reused,
                downloadedThisRunBytes: downloadedThisRun,
                totalBytes: total)
        case .hashingOutput(let file):
            resetETA()
            state = .hashingOutput(file)
        case .finalizing:
            resetETA()
            state = .finalizing
        case .installed(let url):
            resetETA()
            let directory = url.standardizedFileURL
            directoryURL = directory
            installationStatus = AppModelInstallationProbe.status(
                at: directory, descriptor: descriptor)
            guard installationStatus == .complete else {
                finishFailure(
                    RepackError.configurationInvalid(
                        detail: "completed install did not pass metadata validation"),
                    generation: generation)
                return
            }
            state = .installed(modelDirectory: directory)
            task = nil
            onInstalled?(self)
        }
    }

    private func finishStream(generation: UInt64) {
        guard generation == self.generation, task != nil else { return }
        if state == .cancelling {
            finishCancellation(generation: generation)
        } else if !isInstalled {
            finishFailure(
                RepackError.configurationInvalid(
                    detail: "installer ended before completion"),
                generation: generation)
        }
    }

    private func finishCancellation(generation: UInt64) {
        guard generation == self.generation else { return }
        task = nil
        state = .cancelled
        resetETA()
        refresh()
    }

    private func finishFailure(_ error: Error, generation: UInt64) {
        guard generation == self.generation else { return }
        task = nil
        resetETA()
        // A saved partial download makes the failure resumable rather than
        // terminal, so the UI can offer Resume instead of only Download.
        let hasSavedDownload = hasPartialDownload
        state = hasSavedDownload ? .recoverable("\(error)") : .failed("\(error)")
        if let repackError = error as? RepackError,
           case .diskSpaceInsufficient(let path, let required, let available) = repackError {
            // Keep the installer's own measurement; re-probing would report
            // free space after the failed run released its reservation.
            readiness = .insufficientSpace(
                AppModelInstallRequirement(probePath: path,
                                           requiredBytes: required,
                                           availableBytes: available))
        } else {
            refresh()
            if hasSavedDownload { state = .recoverable("\(error)") }
        }
    }

    private func updateETA(reusedBytes: UInt64,
                           downloadedThisRunBytes: UInt64,
                           totalBytes: UInt64) {
        let observation = DownloadETAObservation(
            reusedBytes: reusedBytes,
            downloadedThisRunBytes: downloadedThisRunBytes,
            totalBytes: totalBytes)
        setETA(etaEstimator.update(observation, timestamp: etaTimestamp))
    }

    private var etaTimestamp: Double {
        let components = etaOrigin.duration(to: etaClock.now).components
        return Double(components.seconds)
            + Double(components.attoseconds) / 1_000_000_000_000_000_000
    }

    private func resetETA() {
        etaEstimator.reset()
        etaPresentation = .hidden
        etaText = nil
    }

    private func setETA(_ presentation: DownloadETAPresentation) {
        etaPresentation = presentation
        etaText = DownloadETAFormatter.string(for: presentation)
    }
}
