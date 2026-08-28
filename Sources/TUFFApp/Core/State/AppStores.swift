import Foundation
import Observation
import TUFFEngine

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
    public var documentAttachments: [AppDocumentAttachment] = []
    public var imageAttachmentError: String?
    var addingImagesCount = 0
    public var outputPromptText = ""
    public var outputImageAttachments: [AppImageAttachment] = []
    public var outputDocumentAttachments: [AppDocumentAttachment] = []
    /// Hard links this session's runs staged so the transcript keeps showing
    /// what was sent. Released when the conversation itself moves on, not when
    /// a message is folded into history: a chat whose images could not be saved
    /// to the archive is still drawing — and still sending — these.
    var retainedRunImages: [AppImageAttachment] = []
    /// Whether the exchange in the output slot has already been folded into
    /// `conversation`. The slot is kept afterwards — the copy commands and the
    /// diagnostics still describe the last run — so the transcript needs to be
    /// told, or it draws the same exchange twice.
    public internal(set) var outputTurnIsRecorded = false
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
        discardEmptyConversations(keeping: record.id)
        conversations.insert(record, at: 0)
        selectedConversationID = record.id
        conversation = []
        clearPendingTurn()
        outputPromptText = ""
        outputImageAttachments = []
        outputDocumentAttachments = []
        outputText = ""
        outputThinkingText = ""
        outputTurnIsRecorded = false
        persistArchive()
    }

    /// Point the selected conversation at another model, keeping its history.
    /// Switching models mid-chat re-renders that history through the new
    /// model's prompt format on the next turn, so the chat continues rather
    /// than starting over.
    public func rebindConversation(to modelID: String) {
        guard let selectedConversationID,
              let index = conversations.firstIndex(where: { $0.id == selectedConversationID })
        else { return }
        conversations[index].modelID = modelID
        conversations[index].updatedAt = Date()
        moveConversationToFront(at: index)
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
        discardEmptyConversations(keeping: id)
        selectedConversationID = id
        clearPendingTurn()
        outputPromptText = ""
        outputImageAttachments = []
        outputDocumentAttachments = []
        outputText = ""
        outputThinkingText = ""
        outputTurnIsRecorded = false
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
            outputDocumentAttachments = []
            outputText = ""
            outputThinkingText = ""
            outputTurnIsRecorded = false
            restoreSelectedTurns()
        }
        do {
            try repository?.deleteAttachments(conversationID: id)
        } catch {
            persistenceError = String(describing: error)
        }
        persistArchive()
    }

    /// The turn that has been sent but not yet answered, and the chat it went
    /// into. Nil whenever no run is in flight.
    private var pendingTurn: (conversationID: UUID, turnID: UUID)?

    /// Saves a message the moment it is sent, before a single token comes back.
    ///
    /// A chat used to come into existence only when an answer finished, so a
    /// run stopped halfway — or one the app was quit during — left nothing at
    /// all: no chat in the sidebar, no title, nothing on disk. Stopping the
    /// model was indistinguishable from never having asked.
    ///
    /// The turn is written to the archive here and stays out of `conversation`,
    /// which is the transcript's history: the message being generated is drawn
    /// from the live output slot instead, so it keeps the streaming fast path.
    /// `completeTurn` moves it across when the run ends.
    public func beginTurn(
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
            response: "",
            thinking: nil,
            attachments: durableAttachments,
            documents: turn.documents,
            modelID: modelID))
        if conversations[index].title == "New Chat" {
            conversations[index].title = Self.automaticTitle(for: Self.titleSource(turn))
        }
        conversations[index].updatedAt = now
        moveConversationToFront(at: index)
        pendingTurn = (conversationID: conversations[0].id, turnID: turn.id)
        pendingImageAttachments = durableAttachments
        pendingStagedAttachments = attachments
        persistArchive()
    }

    /// The images `beginTurn` saved, kept so `completeTurn` can hand the
    /// transcript the durable copies rather than the run's staged links.
    private var pendingImageAttachments: [AppConversationAttachment] = []
    private var pendingStagedAttachments: [AppImageAttachment] = []

    /// Files the answer, whatever the run produced, and folds the turn into
    /// history.
    ///
    /// Called for every terminal outcome, not only a clean finish: an answer
    /// cut off halfway is what the reader has, and throwing it away was the
    /// behaviour this replaced.
    public func completeTurn(
        response: String,
        thinking: String?,
        now: Date = Date()
    ) {
        guard let pending = pendingTurn,
              let index = conversations.firstIndex(
                where: { $0.id == pending.conversationID }),
              let turnIndex = conversations[index].turns.firstIndex(
                where: { $0.id == pending.turnID }) else { return }
        conversations[index].turns[turnIndex].response = response
        conversations[index].turns[turnIndex].thinking = thinking
        conversations[index].updatedAt = now
        moveConversationToFront(at: index)

        // The saved copies, not the run's staged links: the next run releases
        // those, and the turn has to keep showing — and re-sending — its images
        // long after that. When nothing durable was written, the staged copies
        // are all there is, which is the same reach the display had before.
        var recorded = conversations.first { $0.id == pending.conversationID }?
            .turns.first { $0.id == pending.turnID }?.chatTurn
            ?? AppChatTurn(prompt: "", response: response)
        let resolved = pendingImageAttachments.compactMap { try? repository?.resolve($0) }
        recorded.images = resolved.count == pendingStagedAttachments.count
            ? resolved : pendingStagedAttachments
        conversation.append(recorded)
        outputTurnIsRecorded = true
        clearPendingTurn()
        persistArchive()
    }

    private func clearPendingTurn() {
        pendingTurn = nil
        pendingImageAttachments = []
        pendingStagedAttachments = []
    }

    /// A whole exchange at once, for a caller that already has the answer.
    ///
    /// The app always goes through `beginTurn` and `completeTurn`, because a run
    /// has a gap between the two. This is the same thing with no gap.
    public func recordCompletedTurn(
        _ turn: AppChatTurn,
        attachments: [AppImageAttachment],
        modelID: String,
        now: Date = Date()
    ) {
        beginTurn(turn, attachments: attachments, modelID: modelID, now: now)
        completeTurn(response: turn.response, thinking: turn.thinking, now: now)
    }

    /// What a chat is named after: the typed message, or the first thing
    /// attached when nothing was typed.
    private static func titleSource(_ turn: AppChatTurn) -> String {
        if !turn.prompt.isEmpty { return turn.prompt }
        if let document = turn.documents.first { return document.displayName }
        return turn.images.first?.displayName ?? ""
    }

    /// Drops saved chats that never received a turn, except the one being
    /// opened.
    ///
    /// A chat is written to disk the moment New Chat is pressed, so one
    /// abandoned before its first message stayed in the list forever as another
    /// identical row called "New Chat". That was survivable while the list only
    /// appeared on the Chat screen; now that it is always on screen, a week of
    /// use buries the real chats under them. A chat with a turn in it is never
    /// touched.
    private func discardEmptyConversations(keeping keptID: UUID?) {
        let discarded = conversations.filter { $0.turns.isEmpty && $0.id != keptID }
        guard !discarded.isEmpty else { return }
        conversations.removeAll { $0.turns.isEmpty && $0.id != keptID }
        for record in discarded {
            // An empty chat has recorded no turn, so it can own no attachment.
            // Asking anyway costs nothing and cannot strand one.
            try? repository?.deleteAttachments(conversationID: record.id)
        }
        if let current = selectedConversationID,
           !conversations.contains(where: { $0.id == current }) {
            selectedConversationID = keptID ?? conversations.first?.id
        }
    }

    private func restoreSelectedTurns() {
        guard let record = selectedConversation else {
            conversation = []
            return
        }
        conversation = record.turns.map { persisted in
            var turn = persisted.chatTurn
            turn.images = persisted.attachments.compactMap {
                try? repository?.resolve($0)
            }
            return turn
        }
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
    public var newlineShortcut: AppNewlineShortcut = .shiftReturn
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

public enum AppServerHealth: Equatable, Sendable {
    case unknown
    case checking
    case healthy
    case unreachable
}

/// Local-server presentation state. The HTTP adapter arrives in Part 5, but
/// the navigation and app-state boundary are established with the v2 shell.
@MainActor
@Observable
public final class AppServerStore {
    public static let maximumRecentErrors = 8
    public var status: AppServerStatus = .stopped
    public var health: AppServerHealth = .unknown
    public var desiredPort = 8_080
    public var queueLimit = 4
    public var boundPort: Int?
    public var modelID: String?
    public var queuedRequests = 0
    public var activeRequests = 0
    public var recentErrors: [String] = []

    public init() {}

    public var isBusy: Bool {
        switch status {
        case .starting, .running, .stopping: true
        case .stopped, .failed: false
        }
    }

    public func recordError(_ message: String) {
        guard !message.isEmpty else { return }
        recentErrors.insert(message, at: 0)
        if recentErrors.count > Self.maximumRecentErrors {
            recentErrors.removeLast(recentErrors.count - Self.maximumRecentErrors)
        }
    }
}
