import AppKit
import SwiftUI

/// How much larger TUFF draws its text than the system default.
///
/// macOS has no Dynamic Type: `Font.body` and friends resolve to fixed point
/// sizes and `dynamicTypeSize` does nothing. Scaling the rendered view with
/// `scaleEffect` is not an answer either — it rasterises at the base size and
/// then magnifies the bitmap, so every glyph goes soft. The only way to get
/// larger text that is still drawn at full resolution is to ask for larger
/// fonts, which is what this scale feeds.
///
/// `RootView` publishes the zoom here; `appFont(_:)` reads it.
private struct AppFontScaleKey: EnvironmentKey {
    static let defaultValue: CGFloat = 1
}

extension EnvironmentValues {
    public var appFontScale: CGFloat {
        get { self[AppFontScaleKey.self] }
        set { self[AppFontScaleKey.self] = newValue }
    }
}

/// A font request that has not been given a size yet.
///
/// It mirrors the part of SwiftUI's `Font` API the app actually uses, so a call
/// site reads the same as it did before (`.caption.weight(.semibold)`), but the
/// point size is resolved against the current zoom instead of being baked in.
public struct AppFont: Equatable, Sendable {
    /// The system text styles, which carry the sizes TUFF starts from.
    public enum Style: Equatable, Sendable {
        case largeTitle, title, title2, title3, headline, subheadline
        case body, callout, footnote, caption, caption2

        var nsTextStyle: NSFont.TextStyle {
            switch self {
            case .largeTitle: return .largeTitle
            case .title: return .title1
            case .title2: return .title2
            case .title3: return .title3
            case .headline: return .headline
            case .subheadline: return .subheadline
            case .body: return .body
            case .callout: return .callout
            case .footnote: return .footnote
            case .caption: return .caption1
            case .caption2: return .caption2
            }
        }

        /// `.headline` is the one style that is not regular weight.
        var defaultWeight: Font.Weight? {
            self == .headline ? .semibold : nil
        }
    }

    private enum Base: Equatable, Sendable {
        case style(Style)
        case fixed(CGFloat)
    }

    private var base: Base
    private var weightOverride: Font.Weight?
    private var design: Font.Design?
    private var isBold = false
    private var isItalic = false
    private var usesMonospacedDigits = false

    private init(base: Base) { self.base = base }

    public static let largeTitle = AppFont(base: .style(.largeTitle))
    public static let title = AppFont(base: .style(.title))
    public static let title2 = AppFont(base: .style(.title2))
    public static let title3 = AppFont(base: .style(.title3))
    public static let headline = AppFont(base: .style(.headline))
    public static let subheadline = AppFont(base: .style(.subheadline))
    public static let body = AppFont(base: .style(.body))
    public static let callout = AppFont(base: .style(.callout))
    public static let footnote = AppFont(base: .style(.footnote))
    public static let caption = AppFont(base: .style(.caption))
    public static let caption2 = AppFont(base: .style(.caption2))

    public static func system(
        size: CGFloat,
        weight: Font.Weight? = nil,
        design: Font.Design? = nil
    ) -> AppFont {
        var font = AppFont(base: .fixed(size))
        font.weightOverride = weight
        font.design = design
        return font
    }

    public static func system(_ style: Style, design: Font.Design? = nil) -> AppFont {
        var font = AppFont(base: .style(style))
        font.design = design
        return font
    }

    public func weight(_ weight: Font.Weight) -> AppFont {
        var font = self
        font.weightOverride = weight
        return font
    }

    public func bold() -> AppFont {
        var font = self
        font.isBold = true
        return font
    }

    public func italic() -> AppFont {
        var font = self
        font.isItalic = true
        return font
    }

    public func monospaced() -> AppFont {
        var font = self
        font.design = .monospaced
        return font
    }

    public func monospacedDigit() -> AppFont {
        var font = self
        font.usesMonospacedDigits = true
        return font
    }

    /// The unscaled point size this font starts from. Read from the system
    /// text style rather than hard-coded, so TUFF keeps matching macOS.
    @MainActor
    public var basePointSize: CGFloat {
        switch base {
        case .fixed(let size): return size
        case .style(let style):
            return NSFont.preferredFont(forTextStyle: style.nsTextStyle).pointSize
        }
    }

    @MainActor
    public func resolved(scale: CGFloat) -> Font {
        let weight: Font.Weight? = {
            if isBold { return .bold }
            if let weightOverride { return weightOverride }
            if case .style(let style) = base { return style.defaultWeight }
            return nil
        }()
        var font = Font.system(
            size: basePointSize * scale, weight: weight, design: design)
        if usesMonospacedDigits { font = font.monospacedDigit() }
        if isItalic { font = font.italic() }
        return font
    }
}

extension View {
    /// Sets the font the way `font(_:)` does, at the current zoom.
    public func appFont(_ font: AppFont) -> some View {
        modifier(AppFontModifier(font: font))
    }
}

private struct AppFontModifier: ViewModifier {
    @Environment(\.appFontScale) private var scale
    let font: AppFont

    func body(content: Content) -> some View {
        content.font(font.resolved(scale: scale))
    }
}
