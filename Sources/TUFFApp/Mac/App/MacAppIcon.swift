import AppKit

// Bundle.module resolves only inside this target, so the icon cannot be loaded
// from the presentation library.
enum MacAppIcon {
    private static let resourceBundle: Bundle = {
        if let resources = Bundle.main.resourceURL,
           let packaged = Bundle(
               url: resources.appendingPathComponent(
                   "TUFF_TUFFMac.bundle",
                   isDirectory: true
               )
           ) {
            return packaged
        }
        return Bundle.module
    }()

    /// Only used as a Dock fallback by bare SwiftPM launches. App bundles use
    /// their compiled icon; the PNG's artwork fills its canvas and needs the
    /// normal macOS inset when supplied directly to NSApplication.
    static func dockFallback() -> NSImage? {
        guard let source = load() else { return nil }
        let size = NSSize(width: 1024, height: 1024)
        let result = NSImage(size: size)
        result.lockFocus()
        source.draw(in: NSRect(x: 100, y: 100, width: 824, height: 824),
                    from: .zero, operation: .sourceOver, fraction: 1)
        result.unlockFocus()
        return result
    }

    static func load() -> NSImage? {
        guard let url = resourceBundle.url(
            forResource: "tuff-app-icon",
            withExtension: "png"
        ) else { return nil }
        return NSImage(contentsOf: url)
    }
}
