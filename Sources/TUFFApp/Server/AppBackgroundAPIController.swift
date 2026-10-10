import AppKit
import Foundation
import Observation
import ServiceManagement
import TUFFModelCatalog
import TUFFServerCore

@MainActor protocol BackgroundAgentService {
    var status: SMAppService.Status { get }
    func register() throws
    func unregister() async throws
}
extension SMAppService: BackgroundAgentService {}

@MainActor @Observable
public final class AppBackgroundAPIController {
    public private(set) var settings: TUFFBackgroundServerSettings
    public private(set) var status: ServerControlStatus?
    public private(set) var message: String?
    public private(set) var isChanging = false
    public let isAvailable: Bool
    public var requiresApproval: Bool { service?.status == .requiresApproval }
    private let settingsURL: URL
    private let service: (any BackgroundAgentService)?

    public convenience init(bundle: Bundle = .main, settingsURL: URL? = nil) {
        let name = (bundle.bundleIdentifier ?? "") + ".server.plist"
        let plist = bundle.bundleURL.appendingPathComponent("Contents/Library/LaunchAgents/\(name)")
        let available = bundle.bundleURL.pathExtension == "app"
            && FileManager.default.fileExists(atPath: plist.path)
        self.init(service: available ? SMAppService.agent(plistName: name) : nil,
            settingsURL: settingsURL ?? TUFFBackgroundServerSettingsStore.fileURL(
                applicationSupport: RouterServerRuntime.applicationSupportURL()))
    }

    init(service: (any BackgroundAgentService)?, settingsURL: URL) {
        self.service = service
        self.settingsURL = settingsURL
        isAvailable = service != nil
        if case .loaded(let saved) = TUFFBackgroundServerSettingsStore.load(from: self.settingsURL) {
            settings = saved
        } else {
            settings = TUFFBackgroundServerSettings()
        }
    }

    public var endpoint: String { "http://127.0.0.1:\(settings.port)/v1" }
    public var defaultModelID: String {
        ServerInstalledModels.descriptor(named: settings.defaultModel)?.apiModelID ?? settings.defaultModel
    }

    public func updatePort(_ input: String) {
        guard let port = Int(input.trimmingCharacters(in: .whitespacesAndNewlines)),
              (1...65_535).contains(port) else {
            message = "Port must be a number from 1 to 65535."
            return
        }
        guard port != settings.port else { return }
        update { $0.port = port }
    }

    public func update(_ change: (inout TUFFBackgroundServerSettings) -> Void) {
        guard isAvailable, !isChanging, let service else { return }
        var candidate = settings
        change(&candidate)
        let previous = settings
        isChanging = true
        message = nil
        Task {
            defer { isChanging = false }
            do {
                try TUFFBackgroundServerSettingsStore.save(candidate, to: settingsURL)
                settings = candidate
                if !candidate.enabled || !previous.enabled || previous.port != candidate.port {
                    if service.status == .enabled || service.status == .requiresApproval {
                        try await service.unregister()
                    }
                }
                if candidate.enabled && service.status != .enabled {
                    try service.register()
                    recordRegistration()
                }
                if requiresApproval {
                    message = "Allow TUFF in Login Items to start the Background API."
                    SMAppService.openSystemSettingsLoginItems()
                }
                await refreshStatus()
            } catch {
                // Keep the UI and hosted-server guard aligned with a failed
                // registration or restart, rather than leaving a false "on".
                if service.status != .requiresApproval {
                    try? TUFFBackgroundServerSettingsStore.save(previous, to: settingsURL)
                    settings = previous
                    if previous.enabled && service.status == .notRegistered {
                        try? service.register()
                    }
                }
                message = "Could not change Background API settings: \(error.localizedDescription)"
            }
        }
    }

    /// Registers the login item again after an update. TUFF is signed
    /// without a developer team, so macOS pins the item to the exact binary
    /// that registered it, and after an update launchd refuses the new one
    /// ("Launch Constraint Violation") and the Background API stays down.
    /// Registering again from the new build points it at the new binary.
    public func refreshRegistrationAfterUpdate(build: String = TUFFVersion.current) {
        guard isAvailable, let service, settings.enabled, !isChanging,
              registeredBuild != build else { return }
        isChanging = true
        Task {
            defer { isChanging = false }
            do {
                if service.status == .enabled || service.status == .requiresApproval {
                    try await service.unregister()
                }
                try service.register()
                recordRegistration(build: build)
                await refreshStatus()
            } catch {
                message = "Could not restart the Background API after updating: "
                    + error.localizedDescription
            }
        }
    }

    /// The build the login item was last registered from, kept beside the
    /// settings so the settings format itself does not change.
    private var registrationURL: URL {
        settingsURL.deletingLastPathComponent().appendingPathComponent("registered-build")
    }

    var registeredBuild: String? {
        (try? String(contentsOf: registrationURL, encoding: .utf8))?
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func recordRegistration(build: String = TUFFVersion.current) {
        try? FileManager.default.createDirectory(
            at: registrationURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? build.write(to: registrationURL, atomically: true, encoding: .utf8)
    }

    public func openLoginItems() { SMAppService.openSystemSettingsLoginItems() }

    /// Where the login item writes requests, loads and errors.
    public var logURL: URL { RouterServerRuntime.logURL() }
    public var hasLog: Bool { FileManager.default.fileExists(atPath: logURL.path) }
    public func openLog() { NSWorkspace.shared.open(logURL) }

    public func refreshStatus() async {
        guard isAvailable, settings.enabled else { status = nil; return }
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(settings.port)/tuff/v1/status")!)
        request.timeoutInterval = 2
        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
            status = try JSONDecoder().decode(ServerControlStatus.self, from: data)
        } catch {
            status = nil
        }
    }
}
