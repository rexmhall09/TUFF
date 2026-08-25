import Foundation
import TUFFModelCatalog
import TurboFieldfare

struct MacModelSettings: Codable, Equatable, Sendable {
    var contextTokens: Int
    var expertCacheSlots: Int
    var temperature: Double
    var topKEnabled: Bool
    var topK: Int
    var topPEnabled: Bool
    var topP: Double
    var prefillEnabled: Bool
    var visionResidencyPolicy: VisionResidencyPolicy
    var rdadvisePolicy: AppRDAdvicePolicy

    init(contextTokens: Int = AppContextLengthOption.eightK.tokens,
         expertCacheSlots: Int = 16,
         temperature: Double = 0.2,
         topKEnabled: Bool = true,
         topK: Int = 64,
         topPEnabled: Bool = true,
         topP: Double = 0.95,
         prefillEnabled: Bool = true,
         visionResidencyPolicy: VisionResidencyPolicy = .onDemand,
         rdadvisePolicy: AppRDAdvicePolicy = .off) {
        self.contextTokens = contextTokens
        self.expertCacheSlots = expertCacheSlots
        self.temperature = temperature
        self.topKEnabled = topKEnabled
        self.topK = topK
        self.topPEnabled = topPEnabled
        self.topP = topP
        self.prefillEnabled = prefillEnabled
        self.visionResidencyPolicy = visionResidencyPolicy
        self.rdadvisePolicy = rdadvisePolicy
    }

    static func defaults(for profileKey: String) -> MacModelSettings {
        guard let id = TUFFModelID(rawValue: profileKey),
              let descriptor = TUFFModelCatalog.model(id: id) else {
            return MacModelSettings()
        }
        let defaults = descriptor.runtimeDefaults
        return MacModelSettings(
            contextTokens: defaults.contextTokens,
            expertCacheSlots: defaults.expertCacheSlots,
            temperature: defaults.temperature,
            topK: defaults.topK,
            topP: defaults.topP)
    }

    func isValid() -> Bool {
        AppContextLengthOption.allCases.contains { $0.tokens == contextTokens }
            && AppRuntimeOptions.allowedSlotCounts.contains(expertCacheSlots)
            && temperature.isFinite && (0...2).contains(temperature)
            && (1...256).contains(topK)
            && topP.isFinite && (0.01...1).contains(topP)
    }
}

struct MacAppSettings: Codable, Equatable, Sendable {
    static let fileName = "mac-app-settings.json"
    static let currentVersion = 3
    static let defaultProfileKey = TUFFModelCatalog.default.id.rawValue

    var version: Int = currentVersion
    var modelProfiles: [String: MacModelSettings]
    var newlineShortcut: AppNewlineShortcut = .return
    var showPromptExamples: Bool = true
    var sentPromptBehavior: AppSentPromptBehavior = .keep
    var loadModelOnLaunch: Bool = false

    // Source compatibility for the v1 app and focused tests. These map only to
    // Gemma's stable profile; app code uses `profile(for:)` explicitly.
    var contextTokens: Int {
        get { profile(for: Self.defaultProfileKey).contextTokens }
        set { updateDefaultProfile { $0.contextTokens = newValue } }
    }
    var expertCacheSlots: Int {
        get { profile(for: Self.defaultProfileKey).expertCacheSlots }
        set { updateDefaultProfile { $0.expertCacheSlots = newValue } }
    }
    var temperature: Double {
        get { profile(for: Self.defaultProfileKey).temperature }
        set { updateDefaultProfile { $0.temperature = newValue } }
    }
    var topKEnabled: Bool {
        get { profile(for: Self.defaultProfileKey).topKEnabled }
        set { updateDefaultProfile { $0.topKEnabled = newValue } }
    }
    var topK: Int {
        get { profile(for: Self.defaultProfileKey).topK }
        set { updateDefaultProfile { $0.topK = newValue } }
    }
    var topPEnabled: Bool {
        get { profile(for: Self.defaultProfileKey).topPEnabled }
        set { updateDefaultProfile { $0.topPEnabled = newValue } }
    }
    var topP: Double {
        get { profile(for: Self.defaultProfileKey).topP }
        set { updateDefaultProfile { $0.topP = newValue } }
    }
    var prefillEnabled: Bool {
        get { profile(for: Self.defaultProfileKey).prefillEnabled }
        set { updateDefaultProfile { $0.prefillEnabled = newValue } }
    }
    var visionResidencyPolicy: VisionResidencyPolicy {
        get { profile(for: Self.defaultProfileKey).visionResidencyPolicy }
        set { updateDefaultProfile { $0.visionResidencyPolicy = newValue } }
    }
    var rdadvisePolicy: AppRDAdvicePolicy {
        get { profile(for: Self.defaultProfileKey).rdadvisePolicy }
        set { updateDefaultProfile { $0.rdadvisePolicy = newValue } }
    }

    private enum CodingKeys: String, CodingKey {
        case version
        case modelProfiles
        case newlineShortcut
        case showPromptExamples
        case sentPromptBehavior
        case loadModelOnLaunch
    }

    private enum LegacyCodingKeys: String, CodingKey {
        case version
        case contextTokens
        case expertCacheSlots
        case temperature
        case topKEnabled
        case topK
        case topPEnabled
        case topP
        case prefillEnabled
        case newlineShortcut
        case showPromptExamples
        case sentPromptBehavior
        case visionResidencyPolicy
        case rdadvisePolicy
        case loadModelOnLaunch
    }

    init(version: Int = currentVersion,
         contextTokens: Int = AppContextLengthOption.eightK.tokens,
         expertCacheSlots: Int = 16,
         temperature: Double = 0.2,
         topKEnabled: Bool = true,
         topK: Int = 64,
         topPEnabled: Bool = true,
         topP: Double = 0.95,
         prefillEnabled: Bool = true,
         newlineShortcut: AppNewlineShortcut = .return,
         showPromptExamples: Bool = true,
         sentPromptBehavior: AppSentPromptBehavior = .keep,
         visionResidencyPolicy: VisionResidencyPolicy = .onDemand,
         rdadvisePolicy: AppRDAdvicePolicy = .off,
         loadModelOnLaunch: Bool = false,
         modelProfiles: [String: MacModelSettings]? = nil) {
        self.version = version
        self.modelProfiles = modelProfiles ?? [Self.defaultProfileKey: MacModelSettings(
            contextTokens: contextTokens,
            expertCacheSlots: expertCacheSlots,
            temperature: temperature,
            topKEnabled: topKEnabled,
            topK: topK,
            topPEnabled: topPEnabled,
            topP: topP,
            prefillEnabled: prefillEnabled,
            visionResidencyPolicy: visionResidencyPolicy,
            rdadvisePolicy: rdadvisePolicy)]
        self.newlineShortcut = newlineShortcut
        self.showPromptExamples = showPromptExamples
        self.sentPromptBehavior = sentPromptBehavior
        self.loadModelOnLaunch = loadModelOnLaunch
    }

    init(from decoder: Decoder) throws {
        let stamp = try decoder.container(keyedBy: LegacyCodingKeys.self)
        version = try stamp.decode(Int.self, forKey: .version)
        if version >= Self.currentVersion {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            modelProfiles = try container.decode(
                [String: MacModelSettings].self, forKey: .modelProfiles)
            newlineShortcut = try container.decodeIfPresent(
                AppNewlineShortcut.self, forKey: .newlineShortcut) ?? .return
            showPromptExamples = try container.decodeIfPresent(
                Bool.self, forKey: .showPromptExamples) ?? true
            sentPromptBehavior = try container.decodeIfPresent(
                AppSentPromptBehavior.self, forKey: .sentPromptBehavior) ?? .keep
            loadModelOnLaunch = try container.decodeIfPresent(
                Bool.self, forKey: .loadModelOnLaunch) ?? false
            return
        }

        modelProfiles = [Self.defaultProfileKey: MacModelSettings(
            contextTokens: try stamp.decode(Int.self, forKey: .contextTokens),
            expertCacheSlots: try stamp.decode(Int.self, forKey: .expertCacheSlots),
            temperature: try stamp.decode(Double.self, forKey: .temperature),
            topKEnabled: try stamp.decode(Bool.self, forKey: .topKEnabled),
            topK: try stamp.decode(Int.self, forKey: .topK),
            topPEnabled: try stamp.decode(Bool.self, forKey: .topPEnabled),
            topP: try stamp.decode(Double.self, forKey: .topP),
            prefillEnabled: try stamp.decode(Bool.self, forKey: .prefillEnabled),
            visionResidencyPolicy: try stamp.decodeIfPresent(
                VisionResidencyPolicy.self, forKey: .visionResidencyPolicy) ?? .onDemand,
            rdadvisePolicy: try stamp.decodeIfPresent(
                AppRDAdvicePolicy.self, forKey: .rdadvisePolicy) ?? .off)]
        newlineShortcut = try stamp.decodeIfPresent(
            AppNewlineShortcut.self, forKey: .newlineShortcut) ?? .return
        showPromptExamples = try stamp.decodeIfPresent(
            Bool.self, forKey: .showPromptExamples) ?? true
        sentPromptBehavior = try stamp.decodeIfPresent(
            AppSentPromptBehavior.self, forKey: .sentPromptBehavior) ?? .keep
        loadModelOnLaunch = try stamp.decodeIfPresent(
            Bool.self, forKey: .loadModelOnLaunch) ?? false
    }

    func profile(for key: String) -> MacModelSettings {
        modelProfiles[key] ?? .defaults(for: key)
    }

    mutating func setProfile(_ profile: MacModelSettings, for key: String) {
        modelProfiles[key] = profile
    }

    func isValid() -> Bool {
        version == Self.currentVersion
            && !modelProfiles.isEmpty
            && modelProfiles.keys.allSatisfy { !$0.isEmpty }
            && modelProfiles.values.allSatisfy { $0.isValid() }
    }

    private mutating func updateDefaultProfile(
        _ update: (inout MacModelSettings) -> Void
    ) {
        var value = profile(for: Self.defaultProfileKey)
        update(&value)
        modelProfiles[Self.defaultProfileKey] = value
    }
}

/// Just enough of the file to route on, so a newer build's schema cannot throw
/// before its version has been read.
private struct VersionStamp: Decodable {
    let version: Int
}

enum MacAppSettingsFileStore {
    static func fileURL(forModelDirectory modelDirectory: URL) -> URL {
        modelDirectory.standardizedFileURL
            .deletingLastPathComponent()
            .appendingPathComponent(MacAppSettings.fileName, isDirectory: false)
    }

    static func loadOrCreate(forModelDirectory modelDirectory: URL,
                             profileKey: String = MacAppSettings.defaultProfileKey,
                             fileManager: FileManager = .default) -> MacAppSettings {
        let fileURL = fileURL(forModelDirectory: modelDirectory)
        if fileManager.fileExists(atPath: fileURL.path) {
            do {
                let data = try Data(contentsOf: fileURL)
                // The version is read on its own, before the full decode, because
                // nine keys decode with a hard `decode` and a newer build is
                // entitled to have moved any of them. Decoding first threw on
                // exactly the files this branch exists to protect, and the throw
                // reached the `catch` below, which deletes the file - so an older
                // build silently replaced a newer one's settings with its own
                // defaults. A version gate that cannot survive a schema change
                // gates nothing.
                let stamp = try JSONDecoder().decode(VersionStamp.self, from: data)
                // This build does not know what a later version means, so it runs
                // on defaults and leaves the file exactly as its owner wrote it.
                guard stamp.version <= MacAppSettings.currentVersion else {
                    return MacAppSettings(modelProfiles: [
                        profileKey: .defaults(for: profileKey)
                    ])
                }
                var settings = try JSONDecoder().decode(MacAppSettings.self, from: data)
                let needsMigration = settings.version < MacAppSettings.currentVersion
                if needsMigration {
                    // v1/v2 had one global runtime profile. Move it to the model
                    // active during the first v3 launch, preserving every
                    // deliberate value and leaving other models on defaults.
                    let legacy = settings.profile(for: MacAppSettings.defaultProfileKey)
                    settings.modelProfiles = [profileKey: legacy]
                    settings.version = MacAppSettings.currentVersion
                }
                guard settings.isValid() else { throw InvalidSettings() }
                if needsMigration {
                    try? save(settings, forModelDirectory: modelDirectory,
                              fileManager: fileManager)
                }
                return settings
            } catch {
                try? fileManager.removeItem(at: fileURL)
            }
        }

        let settings = MacAppSettings(modelProfiles: [
            profileKey: .defaults(for: profileKey)
        ])
        try? save(settings, forModelDirectory: modelDirectory, fileManager: fileManager)
        return settings
    }

    static func save(_ settings: MacAppSettings,
                     forModelDirectory modelDirectory: URL,
                     fileManager: FileManager = .default) throws {
        guard settings.isValid() else { throw InvalidSettings() }
        let fileURL = fileURL(forModelDirectory: modelDirectory)
        try fileManager.createDirectory(
            at: fileURL.deletingLastPathComponent(),
            withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        var data = try encoder.encode(settings)
        data.append(0x0A)
        try data.write(to: fileURL, options: .atomic)
    }

    private struct InvalidSettings: Error {}
}
