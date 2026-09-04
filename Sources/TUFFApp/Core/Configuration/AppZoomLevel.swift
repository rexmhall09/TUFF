import CoreGraphics

/// How much larger or smaller the whole interface is drawn.
///
/// macOS has no Dynamic Type, so `dynamicTypeSize` does nothing here and the
/// text styles resolve to fixed point sizes. This scale is therefore applied
/// to the fonts themselves — see `AppFont` — rather than to the rendered
/// window: magnifying the view with `scaleEffect` rasterises at the base size
/// and blows the bitmap up, which leaves every glyph soft. Asking for larger
/// fonts keeps the text drawn at full resolution, and the icons and control
/// chrome sized against those fonts grow with them.
public enum AppZoomLevel: String, CaseIterable, Codable, Sendable, Identifiable, Hashable {
    case percent90 = "90"
    case percent100 = "100"
    case percent110 = "110"
    case percent125 = "125"
    case percent150 = "150"

    public static let `default` = AppZoomLevel.percent100

    public var id: String { rawValue }

    public var settingsLabel: String { "\(rawValue)%" }

    public var scale: CGFloat {
        switch self {
        case .percent90: return 0.9
        case .percent100: return 1
        case .percent110: return 1.1
        case .percent125: return 1.25
        case .percent150: return 1.5
        }
    }

    /// One step up, clamped at the largest level.
    public var zoomedIn: AppZoomLevel {
        let levels = Self.allCases
        let index = levels.firstIndex(of: self) ?? 0
        return levels[min(index + 1, levels.count - 1)]
    }

    /// One step down, clamped at the smallest level.
    public var zoomedOut: AppZoomLevel {
        let levels = Self.allCases
        let index = levels.firstIndex(of: self) ?? 0
        return levels[max(index - 1, 0)]
    }
}
