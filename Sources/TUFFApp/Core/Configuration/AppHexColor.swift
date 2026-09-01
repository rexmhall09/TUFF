import Foundation

/// A platform-independent RGB color, parsed from and formatted as a hex
/// string, so the custom accent color can be validated and persisted without
/// this module depending on AppKit or SwiftUI.
public struct AppHexColor: Equatable, Sendable {
    public let red: Double
    public let green: Double
    public let blue: Double

    public init(red: Double, green: Double, blue: Double) {
        self.red = red
        self.green = green
        self.blue = blue
    }

    /// Accepts "#RRGGBB", "RRGGBB", "#RGB", or "RGB". Anything else is nil.
    public init?(hexString: String) {
        var hex = hexString.trimmingCharacters(in: .whitespacesAndNewlines)
        if hex.hasPrefix("#") { hex.removeFirst() }
        let expanded: String
        switch hex.count {
        case 3:
            expanded = hex.map { "\($0)\($0)" }.joined()
        case 6:
            expanded = hex
        default:
            return nil
        }
        guard let value = UInt32(expanded, radix: 16) else { return nil }
        red = Double((value >> 16) & 0xFF) / 255.0
        green = Double((value >> 8) & 0xFF) / 255.0
        blue = Double(value & 0xFF) / 255.0
    }

    public var hexString: String {
        String(format: "#%02X%02X%02X",
               Int((red * 255).rounded()),
               Int((green * 255).rounded()),
               Int((blue * 255).rounded()))
    }

    /// TUFF's original fixed purple, used for `.appDefault` and as the
    /// fallback when a stored custom hex string somehow fails to parse.
    public static let defaultPurple = AppHexColor(
        red: 111.0 / 255.0, green: 77.0 / 255.0, blue: 255.0 / 255.0)
}
