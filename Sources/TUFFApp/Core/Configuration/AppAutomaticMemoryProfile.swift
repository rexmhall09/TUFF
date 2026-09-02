import Foundation

/// How Auto spends this Mac's spare memory on one model.
///
/// The two things Auto can buy differ in what they are worth. Routed-expert
/// cache slots keep expert weights resident and cut SSD reads during
/// generation. Context tokens buy a longer conversation and cost KV memory,
/// and they do not make generation faster. A single automatic answer therefore
/// cannot serve someone who wants throughput and someone who wants a long
/// document in the same window, which is what these three profiles separate.
public enum AppAutomaticMemoryProfile: String, Codable, CaseIterable,
                                       Identifiable, Sendable {
    /// Keep the checkpoint's qualified context and spend the rest on experts.
    case speed
    /// Raise context up to twice the qualified default, then spend the rest.
    case balanced
    /// Take the longest context that fits, then spend the rest on experts.
    case context

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .speed: return "Speed"
        case .balanced: return "Balanced"
        case .context: return "Context"
        }
    }

    public var explanation: String {
        switch self {
        case .speed:
            return "Keeps this model's qualified context and spends the rest of "
                + "the budget on resident experts, which is the only memory "
                + "that reduces work during generation."
        case .balanced:
            return "Raises context to at most twice the qualified default, "
                + "then spends what is left on resident experts."
        case .context:
            return "Takes the longest context that fits, then spends what is "
                + "left on resident experts. Longer conversations, no faster."
        }
    }

    /// Context ceiling as a multiple of the checkpoint's qualified default.
    /// `nil` means the budget is the only limit.
    var contextGrowthLimit: Int? {
        switch self {
        case .speed: return 1
        case .balanced: return 2
        case .context: return nil
        }
    }
}
