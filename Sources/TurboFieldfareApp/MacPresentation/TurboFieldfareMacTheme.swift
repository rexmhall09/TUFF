import AppKit
import SwiftUI

public enum TurboFieldfareMacTheme {
    public static let accentNSColor = NSColor(
        srgbRed: 111.0 / 255.0,
        green: 77.0 / 255.0,
        blue: 255.0 / 255.0,
        alpha: 1)

    public static let accentColor = Color(nsColor: accentNSColor)

    public static func surfaceStyle(
        reduceTransparency: Bool,
        material: Material = .regular
    ) -> AnyShapeStyle {
        if reduceTransparency {
            return AnyShapeStyle(Color(nsColor: .controlBackgroundColor))
        }
        return AnyShapeStyle(material)
    }
}
