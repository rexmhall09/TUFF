import Foundation
import Observation
import Sparkle

public struct AppUpdateConfiguration: Equatable, Sendable {
    public let feedURL: URL
    public let publicKey: String

    public init(feedURL: URL, publicKey: String) {
        self.feedURL = feedURL
        self.publicKey = publicKey
    }

    public static func resolve(
        infoDictionary: [String: Any]?
    ) -> Result<Self, AppUpdateConfigurationError> {
        guard let infoDictionary,
              let feed = (infoDictionary["SUFeedURL"] as? String)?
                .trimmingCharacters(in: .whitespacesAndNewlines),
              let key = (infoDictionary["SUPublicEDKey"] as? String)?
                .trimmingCharacters(in: .whitespacesAndNewlines),
              !feed.isEmpty, !key.isEmpty else {
            return .failure(.missingConfiguration)
        }
        guard let feedURL = URL(string: feed),
              feedURL.scheme?.lowercased() == "https",
              feedURL.host != nil else {
            return .failure(.insecureOrInvalidFeedURL)
        }
        guard let keyData = Data(base64Encoded: key), keyData.count == 32 else {
            return .failure(.invalidPublicKey)
        }
        return .success(Self(feedURL: feedURL, publicKey: key))
    }
}

public enum AppUpdateConfigurationError: Error, Equatable, Sendable {
    case missingConfiguration
    case insecureOrInvalidFeedURL
    case invalidPublicKey

    public var userMessage: String {
        switch self {
        case .missingConfiguration:
            "Update signing is not configured in this build."
        case .insecureOrInvalidFeedURL:
            "The update feed must be a valid HTTPS URL."
        case .invalidPublicKey:
            "The embedded update-signing public key is invalid."
        }
    }
}

/// Owns Sparkle's standard UI and persists update preferences through
/// Sparkle's host-bundle defaults. Invalid or unsigned configurations never
/// start an updater.
@MainActor
@Observable
public final class AppUpdateController {
    public let configuration: AppUpdateConfiguration?
    public private(set) var unavailableReason: String?
    public private(set) var automaticallyChecksForUpdates = false
    public private(set) var automaticallyDownloadsUpdates = false
    public private(set) var allowsAutomaticUpdates = false

    @ObservationIgnored
    private var standardController: SPUStandardUpdaterController?

    public init(infoDictionary: [String: Any]? = Bundle.main.infoDictionary) {
        switch AppUpdateConfiguration.resolve(infoDictionary: infoDictionary) {
        case .failure(let error):
            configuration = nil
            unavailableReason = error.userMessage
        case .success(let configuration):
            self.configuration = configuration
            let controller = SPUStandardUpdaterController(
                startingUpdater: false,
                updaterDelegate: nil,
                userDriverDelegate: nil)
            do {
                try controller.updater.start()
                standardController = controller
                automaticallyChecksForUpdates =
                    controller.updater.automaticallyChecksForUpdates
                automaticallyDownloadsUpdates =
                    controller.updater.automaticallyDownloadsUpdates
                allowsAutomaticUpdates = controller.updater.allowsAutomaticUpdates
                // Sparkle's own scheduler only checks once its interval has
                // elapsed since the last check, so a launch shortly after the
                // previous one would otherwise check for nothing. Sparkle's
                // header docs recommend calling this once, right after
                // starting, to force a check on every launch instead.
                if automaticallyChecksForUpdates {
                    controller.updater.checkForUpdatesInBackground()
                }
            } catch {
                unavailableReason = "The updater could not start: \(error)"
            }
        }
    }

    public var isAvailable: Bool { standardController != nil }

    public var canCheckForUpdates: Bool {
        standardController?.updater.canCheckForUpdates ?? false
    }

    public func checkForUpdates() {
        guard let standardController,
              standardController.updater.canCheckForUpdates else { return }
        standardController.checkForUpdates(nil)
    }

    public func setAutomaticallyChecksForUpdates(_ enabled: Bool) {
        guard let updater = standardController?.updater else { return }
        updater.automaticallyChecksForUpdates = enabled
        automaticallyChecksForUpdates = updater.automaticallyChecksForUpdates
        allowsAutomaticUpdates = updater.allowsAutomaticUpdates
        if !automaticallyChecksForUpdates {
            setAutomaticallyDownloadsUpdates(false)
        }
    }

    public func setAutomaticallyDownloadsUpdates(_ enabled: Bool) {
        guard let updater = standardController?.updater else { return }
        updater.automaticallyDownloadsUpdates = enabled
        automaticallyDownloadsUpdates = updater.automaticallyDownloadsUpdates
    }
}
