import Foundation
import Observation
import TurboFieldfare

public enum AppRunState: Equatable, Sendable {
    case idle
    case running
}

/// Chat drafts, transcript state, and attachment ownership. Persistence is
/// layered onto this store so the UI does not have to know where chats live.
@MainActor
@Observable
public final class AppConversationStore {
    public var promptText = ""
    public var imageAttachments: [AppImageAttachment] = []
    public var imageAttachmentError: String?
    var addingImagesCount = 0
    public var outputPromptText = ""
    public var outputImageAttachments: [AppImageAttachment] = []
    public var outputText = ""
    public var conversation: [AppChatTurn] = []
    public var runIdentity = 0

    public init() {}
}

/// Download, selection, and optional add-on state for the model library.
@MainActor
@Observable
public final class AppModelLibraryStore {
    public var modelPathText = ""
    public var installs: [ModelInstallCoordinator] = []
    public var selectedModelID = ""
    public var isConfirmingVisionPackRemoval = false
    public var visionInstallETAPresentation: DownloadETAPresentation = .hidden
    public var visionInstallETAText: String?
    public var visionInstallState: AppModelInstallState = .idle
    public var visionActivationProgress: Double?
    public var visionInstallReadiness: AppModelInstallReadiness = .checking
    public var visionInstallationStatus: AppVisionPackInstallationStatus = .missing
    public var visionInstallTargetModelID: String?

    public init() {}
}

/// User-configurable defaults. Model profiles are loaded into this store when
/// selection changes, while persistence remains coordinated by AppModel.
@MainActor
@Observable
public final class AppSettingsStore {
    public var runtimeOptions = AppRuntimeOptions()
    public var maxNewTokensOverride: Int?
    public var maxContextTokens = 4_096
    public var temperature = 0.2
    public var topKEnabled = true
    public var topK = 64
    public var topPEnabled = true
    public var topP = 0.95
    public var newlineShortcut: AppNewlineShortcut = .return
    public var showPromptExamples = true
    public var sentPromptBehavior: AppSentPromptBehavior = .keep
    public var loadModelOnLaunch = false

    public init() {}
}

/// State shared by Chat and, in Part 5, the app-hosted local server. The
/// inference client and task ownership stay in AppModel until the broker lands.
@MainActor
@Observable
public final class AppSharedInferenceStore {
    public var runState: AppRunState = .idle
    public var diagnostics: AppDiagnostics?
    public var error: AppInferenceError?
    public var loadState: AppModelLoadState = .notLoaded
    public var loadedRuntimeKey: AppLoadedRuntimeKey?
    public var phase: AppGenerationPhase = .idle
    public var liveTokenCount = 0
    public var liveElapsedDecodeSeconds = 0.0
    public var livePrefillDone = 0
    public var livePrefillTotal = 0
    public var liveMemoryBytes: UInt64?
    public var liveResidentBytes: UInt64?
    public var visionTowerMappedBytes: UInt64?
    public var isCancellationPending = false

    public init() {}
}

public enum AppServerStatus: Equatable, Sendable {
    case stopped
    case starting
    case running
    case stopping
    case failed(String)
}

/// Local-server presentation state. The HTTP adapter arrives in Part 5, but
/// the navigation and app-state boundary are established with the v2 shell.
@MainActor
@Observable
public final class AppServerStore {
    public var status: AppServerStatus = .stopped
    public var queuedRequests = 0
    public var activeRequests = 0
    public var recentErrors: [String] = []

    public init() {}
}
