public enum AppAccentColorMode: String, CaseIterable, Codable, Sendable, Identifiable, Hashable {
    case appDefault = "default"
    case system
    case custom

    public var id: String { rawValue }

    public var settingsLabel: String {
        switch self {
        case .appDefault: return "Default"
        case .system: return "System"
        case .custom: return "Custom"
        }
    }
}
