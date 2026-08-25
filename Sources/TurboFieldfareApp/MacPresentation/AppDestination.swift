public enum AppDestination: String, CaseIterable, Hashable, Identifiable, Sendable {
    case chat
    case models
    case server
    case settings

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .chat: "Chat"
        case .models: "Models"
        case .server: "Server"
        case .settings: "Settings"
        }
    }

    public var systemImage: String {
        switch self {
        case .chat: "bubble.left.and.bubble.right"
        case .models: "shippingbox"
        case .server: "network"
        case .settings: "gearshape"
        }
    }
}
