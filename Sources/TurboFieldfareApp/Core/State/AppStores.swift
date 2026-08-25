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
    public var outputThinkingText = ""
    public var conversation: [AppChatTurn] = []
    public var runIdentity = 0
    public private(set) var conversations: [AppConversationRecord] = []
    public private(set) var selectedConversationID: UUID?
    public private(set) var persistenceError: String?

    private let repository: AppConversationRepository?
    private var archive = AppConversationArchive()
    private var persistenceIsReadOnly = false

    public init(repository: AppConversationRepository? = nil) {
        self.repository = repository
        guard let repository else { return }
        do {
            archive = try repository.load()
            conversations = archive.conversations.sorted { $0.updatedAt > $1.updatedAt }
            selectedConversationID = archive.selectedConversationID.flatMap { selected in
                conversations.contains(where: { $0.id == selected }) ? selected : nil
            } ?? conversations.first?.id
            restoreSelectedTurns()
        } catch {
            // Do not overwrite an archive this build cannot understand or
            // decode. The in-memory chat remains usable and the error is shown
            // in Settings/Chat once that screen lands.
            persistenceError = String(describing: error)
            persistenceIsReadOnly = true
        }
    }

    public static func persistentDefault() -> AppConversationStore {
        AppConversationStore(repository: AppConversationRepository())
    }

    public var selectedConversation: AppConversationRecord? {
        guard let selectedConversationID else { return nil }
        return conversations.first { $0.id == selectedConversationID }
    }

    public var hasCompletedMessages: Bool {
        !conversation.isEmpty
    }

    public func startNewConversation(modelID: String, now: Date = Date()) {
        let record = AppConversationRecord(modelID: modelID, createdAt: now, updatedAt: now)
        conversations.insert(record, at: 0)
        selectedConversationID = record.id
        conversation = []
        outputPromptText = ""
        outputImageAttachments = []
        outputText = ""
        outputThinkingText = ""
        persistArchive()
    }

    public func bindEmptyConversation(to modelID: String) {
        guard let selectedConversationID,
              let index = conversations.firstIndex(where: { $0.id == selectedConversationID }),
              conversations[index].turns.isEmpty else { return }
        conversations[index].modelID = modelID
        conversations[index].updatedAt = Date()
        moveConversationToFront(at: index)
        persistArchive()
    }

    public func selectConversation(id: UUID) {
        guard conversations.contains(where: { $0.id == id }) else { return }
        selectedConversationID = id
        outputPromptText = ""
        outputImageAttachments = []
        outputText = ""
        outputThinkingText = ""
        restoreSelectedTurns()
        persistArchive()
    }

    public func renameConversation(id: UUID, title: String) {
        let cleaned = Self.cleanedTitle(title)
        guard !cleaned.isEmpty,
              let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        conversations[index].title = cleaned
        conversations[index].updatedAt = Date()
        moveConversationToFront(at: index)
        persistArchive()
    }

    public func deleteConversation(id: UUID) {
        guard let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        conversations.remove(at: index)
        if selectedConversationID == id {
            selectedConversationID = conversations.first?.id
            outputPromptText = ""
            outputImageAttachments = []
            outputText = ""
            outputThinkingText = ""
            restoreSelectedTurns()
        }
        do {
            try repository?.deleteAttachments(conversationID: id)
        } catch {
            persistenceError = String(describing: error)
        }
        persistArchive()
    }

    public func recordCompletedTurn(
        _ turn: AppChatTurn,
        attachments: [AppImageAttachment],
        modelID: String,
        now: Date = Date()
    ) {
        var index: Int
        if let selectedConversationID,
           let selected = conversations.firstIndex(where: { $0.id == selectedConversationID }),
           conversations[selected].modelID == modelID {
            index = selected
        } else {
            let record = AppConversationRecord(
                modelID: modelID, createdAt: now, updatedAt: now)
            conversations.insert(record, at: 0)
            selectedConversationID = record.id
            index = 0
        }

        let durableAttachments: [AppConversationAttachment]
        do {
            durableAttachments = try repository?.persistAttachments(
                attachments,
                conversationID: conversations[index].id) ?? []
        } catch {
            // The text is still useful and remains model context. Image
            // persistence fails visibly instead of pretending the images were
            // saved when their managed copies do not exist.
            persistenceError = String(describing: error)
            durableAttachments = []
        }

        conversations[index].turns.append(AppPersistedChatTurn(
            id: turn.id,
            prompt: turn.prompt,
            response: turn.response,
            thinking: turn.thinking,
            attachments: durableAttachments))
        if conversations[index].title == "New Chat" {
            conversations[index].title = Self.automaticTitle(for: turn.prompt)
        }
        conversations[index].updatedAt = now
        moveConversationToFront(at: index)
        conversation.append(turn)
        persistArchive()
    }

    public func attachments(for turnID: UUID) -> [AppImageAttachment] {
        guard let record = selectedConversation,
              let turn = record.turns.first(where: { $0.id == turnID }),
              let repository else { return [] }
        return turn.attachments.compactMap { try? repository.resolve($0) }
    }

    private func restoreSelectedTurns() {
        conversation = selectedConversation?.turns.map(\.chatTurn) ?? []
    }

    private func persistArchive() {
        guard let repository, !persistenceIsReadOnly else { return }
        archive = AppConversationArchive(
            selectedConversationID: selectedConversationID,
            conversations: conversations)
        do {
            try repository.save(archive)
        } catch {
            persistenceError = String(describing: error)
            persistenceIsReadOnly = true
        }
    }

    private func moveConversationToFront(at index: Int) {
        guard index > 0 else { return }
        let record = conversations.remove(at: index)
        conversations.insert(record, at: 0)
    }

    private static func automaticTitle(for prompt: String) -> String {
        let cleaned = cleanedTitle(prompt)
        guard cleaned.count > 52 else { return cleaned.isEmpty ? "New Chat" : cleaned }
        let end = cleaned.index(cleaned.startIndex, offsetBy: 49)
        return String(cleaned[..<end]).trimmingCharacters(in: .whitespaces) + "…"
    }

    private static func cleanedTitle(_ value: String) -> String {
        value.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    }
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
    public var reasoning: ChatReasoning = .off
    public var reasoningEffort: GPTOSSReasoningEffort = .medium
    public var preserveThinking = false
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
