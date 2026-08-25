import AppKit

// Bundle.module resolves only inside this target, so the icon cannot be loaded
// from the presentation library.
enum MacAppIcon {
    private static let resourceBundle: Bundle = {
        if let resources = Bundle.main.resourceURL,
           let packaged = Bundle(
               url: resources.appendingPathComponent(
                   "TurboFieldfare_TurboFieldfareMac.bundle",
                   isDirectory: true
               )
           ) {
            return packaged
        }
        return Bundle.module
    }()

    static func load() -> NSImage? {
        guard let url = resourceBundle.url(
            forResource: "tuff-app-icon",
            withExtension: "png"
        ) else { return nil }
        return NSImage(contentsOf: url)
    }
}
