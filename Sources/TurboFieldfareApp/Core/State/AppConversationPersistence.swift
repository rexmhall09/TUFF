import Darwin
import Foundation

public struct AppConversationAttachment: Codable, Equatable, Sendable, Identifiable {
    public let id: UUID
    public let relativePath: String
    public let displayName: String
    public let encodedBytes: Int
    public let sha256: String

    public init(
        id: UUID,
        relativePath: String,
        displayName: String,
        encodedBytes: Int,
        sha256: String
    ) {
        self.id = id
        self.relativePath = relativePath
        self.displayName = displayName
        self.encodedBytes = encodedBytes
        self.sha256 = sha256
    }
}

public struct AppPersistedChatTurn: Codable, Equatable, Sendable, Identifiable {
    public let id: UUID
    public var prompt: String
    public var response: String
    public var attachments: [AppConversationAttachment]

    public init(
        id: UUID = UUID(),
        prompt: String,
        response: String,
        attachments: [AppConversationAttachment] = []
    ) {
        self.id = id
        self.prompt = prompt
        self.response = response
        self.attachments = attachments
    }

    public var chatTurn: AppChatTurn {
        AppChatTurn(id: id, prompt: prompt, response: response)
    }
}

public struct AppConversationRecord: Codable, Equatable, Sendable, Identifiable {
    public let id: UUID
    public var title: String
    public var modelID: String
    public let createdAt: Date
    public var updatedAt: Date
    public var turns: [AppPersistedChatTurn]

    public init(
        id: UUID = UUID(),
        title: String = "New Chat",
        modelID: String,
        createdAt: Date = Date(),
        updatedAt: Date = Date(),
        turns: [AppPersistedChatTurn] = []
    ) {
        self.id = id
        self.title = title
        self.modelID = modelID
        self.createdAt = createdAt
        self.updatedAt = updatedAt
        self.turns = turns
    }
}

public struct AppConversationArchive: Codable, Equatable, Sendable {
    public static let currentSchemaVersion = 1

    public var schemaVersion: Int
    public var selectedConversationID: UUID?
    public var conversations: [AppConversationRecord]

    public init(
        schemaVersion: Int = currentSchemaVersion,
        selectedConversationID: UUID? = nil,
        conversations: [AppConversationRecord] = []
    ) {
        self.schemaVersion = schemaVersion
        self.selectedConversationID = selectedConversationID
        self.conversations = conversations
    }
}

public enum AppConversationPersistenceError: Error, Equatable, LocalizedError {
    case newerSchema(found: Int, supported: Int)
    case invalidAttachmentPath(String)
    case missingAttachment(String)

    public var errorDescription: String? {
        switch self {
        case .newerSchema(let found, let supported):
            return "The chat archive uses schema \(found), but this version supports \(supported)."
        case .invalidAttachmentPath(let path):
            return "The chat archive contains an invalid attachment path: \(path)"
        case .missingAttachment(let name):
            return "A saved attachment is missing: \(name)"
        }
    }
}

/// Versioned, atomic storage for named chats and their managed attachments.
/// The archive contains relative paths only, so moving Application Support or
/// restoring it from a backup does not leave absolute paths behind.
public struct AppConversationRepository: Sendable {
    public static var defaultRootURL: URL {
        let support = FileManager.default.urls(
            for: .applicationSupportDirectory,
            in: .userDomainMask).first
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent("Library/Application Support", isDirectory: true)
        return support.appendingPathComponent("TUFF/Chats/v1", isDirectory: true)
    }

    public let rootURL: URL

    public init(rootURL: URL = AppConversationRepository.defaultRootURL) {
        self.rootURL = rootURL.standardizedFileURL
    }

    public var archiveURL: URL {
        rootURL.appendingPathComponent("conversations.json")
    }

    public var attachmentsRootURL: URL {
        rootURL.appendingPathComponent("attachments", isDirectory: true)
    }

    public func load() throws -> AppConversationArchive {
        guard FileManager.default.fileExists(atPath: archiveURL.path) else {
            return AppConversationArchive()
        }
        let data = try Data(contentsOf: archiveURL)
        let header = try JSONDecoder().decode(SchemaHeader.self, from: data)
        guard header.schemaVersion <= AppConversationArchive.currentSchemaVersion else {
            throw AppConversationPersistenceError.newerSchema(
                found: header.schemaVersion,
                supported: AppConversationArchive.currentSchemaVersion)
        }
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(AppConversationArchive.self, from: data)
    }

    public func save(_ archive: AppConversationArchive) throws {
        try FileManager.default.createDirectory(
            at: rootURL, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(archive)
        // Foundation's atomic option writes a sibling temporary file and
        // renames it over the archive. A crash therefore leaves either the old
        // complete archive or the new complete archive, never half of either.
        try data.write(to: archiveURL, options: [.atomic])
    }

    public func persistAttachments(
        _ attachments: [AppImageAttachment],
        conversationID: UUID
    ) throws -> [AppConversationAttachment] {
        guard !attachments.isEmpty else { return [] }
        let directory = attachmentsRootURL.appendingPathComponent(
            conversationID.uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)

        var persisted: [AppConversationAttachment] = []
        var createdFiles: [URL] = []
        var activeTemporary: URL?
        do {
            for attachment in attachments {
                guard FileManager.default.fileExists(atPath: attachment.fileURL.path) else {
                    throw AppConversationPersistenceError.missingAttachment(
                        attachment.displayName)
                }
                let fileName = attachment.id.uuidString
                let destination = directory.appendingPathComponent(fileName)
                if !FileManager.default.fileExists(atPath: destination.path) {
                    let temporary = directory.appendingPathComponent(
                        ".\(fileName).\(UUID().uuidString).partial")
                    activeTemporary = temporary
                    try FileManager.default.copyItem(
                        at: attachment.fileURL, to: temporary)
                    guard chmod(temporary.path, S_IRUSR) == 0 else {
                        try? FileManager.default.removeItem(at: temporary)
                        throw CocoaError(.fileWriteNoPermission)
                    }
                    try FileManager.default.moveItem(at: temporary, to: destination)
                    activeTemporary = nil
                    createdFiles.append(destination)
                }
                persisted.append(AppConversationAttachment(
                    id: attachment.id,
                    relativePath: "attachments/\(conversationID.uuidString)/\(fileName)",
                    displayName: attachment.displayName,
                    encodedBytes: attachment.encodedBytes,
                    sha256: attachment.sha256))
            }
        } catch {
            // The conversation has not referenced this directory yet, so a
            // failed save owns every file it just created. Existing files may
            // belong to an earlier saved turn and are deliberately preserved.
            if let activeTemporary {
                try? FileManager.default.removeItem(at: activeTemporary)
            }
            for url in createdFiles { try? FileManager.default.removeItem(at: url) }
            if (try? FileManager.default.contentsOfDirectory(atPath: directory.path))?.isEmpty
                == true { try? FileManager.default.removeItem(at: directory) }
            throw error
        }
        return persisted
    }

    public func resolve(
        _ attachment: AppConversationAttachment
    ) throws -> AppImageAttachment {
        let components = NSString(string: attachment.relativePath).pathComponents
        guard !attachment.relativePath.hasPrefix("/"),
              !components.contains(".."),
              components.first == "attachments" else {
            throw AppConversationPersistenceError.invalidAttachmentPath(
                attachment.relativePath)
        }
        let url = rootURL.appendingPathComponent(attachment.relativePath)
            .standardizedFileURL.resolvingSymlinksInPath()
        let rootComponents = attachmentsRootURL.standardizedFileURL
            .resolvingSymlinksInPath().pathComponents
        guard Array(url.pathComponents.prefix(rootComponents.count)) == rootComponents,
              FileManager.default.fileExists(atPath: url.path) else {
            throw AppConversationPersistenceError.missingAttachment(
                attachment.displayName)
        }
        return AppImageAttachment(
            id: attachment.id,
            fileURL: url,
            displayName: attachment.displayName,
            encodedBytes: attachment.encodedBytes,
            sha256: attachment.sha256)
    }

    public func deleteAttachments(conversationID: UUID) throws {
        let directory = attachmentsRootURL.appendingPathComponent(
            conversationID.uuidString, isDirectory: true)
        guard FileManager.default.fileExists(atPath: directory.path) else { return }
        try FileManager.default.removeItem(at: directory)
    }

    private struct SchemaHeader: Decodable {
        let schemaVersion: Int
    }
}
