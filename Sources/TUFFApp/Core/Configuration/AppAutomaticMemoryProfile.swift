import Foundation

/// Automatic context targets scale with both this model's KV cost and the
/// Mac's memory. Expert caches retain their measured, qualified slot counts.
public enum AppAutomaticMemoryProfile: String, Codable, CaseIterable,
                                       Identifiable, Sendable {
    case speed
    case balanced
    case context

    public var id: String { rawValue }
    public var title: String {
        switch self {
        case .speed: "Speed"
        case .balanced: "Balanced"
        case .context: "Context"
        }
    }

    public var explanation: String {
        switch self {
        case .speed:
            "Uses about a quarter of the context this model can fit in your Mac's memory, keeping conversations lighter."
        case .balanced:
            "Uses about half of the context this model can fit in your Mac's memory, with room for longer conversations."
        case .context:
            "Uses the longest supported context that fits in your Mac's memory. Long conversations take more time to process."
        }
    }

    var contextCapacityDivisor: Int {
        switch self {
        case .speed: 4
        case .balanced: 2
        case .context: 1
        }
    }
}
