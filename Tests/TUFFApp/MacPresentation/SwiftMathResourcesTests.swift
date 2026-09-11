import Foundation
import Testing
@testable import SwiftMath

/// A packaged app keeps SwiftMath's resource bundle in Contents/Resources.
/// SwiftPM's own lookup never checks there, so 5.0.0 found its math fonts only
/// while the build directory that compiled it still existed, and a restored
/// chat containing a formula stopped the app at launch.
@Suite struct SwiftMathResourcesTests {
    @Test func packagedResourcesAreFoundInContentsResources() throws {
        let app = FileManager.default.temporaryDirectory
            .appendingPathComponent("SwiftMathResourcesTests-\(UUID().uuidString).app",
                                    isDirectory: true)
        defer { try? FileManager.default.removeItem(at: app) }
        let resources = app.appendingPathComponent("Contents/Resources", isDirectory: true)
        let packaged = resources.appendingPathComponent(
            "SwiftMath_SwiftMath.bundle", isDirectory: true)
        try FileManager.default.createDirectory(
            at: resources, withIntermediateDirectories: true)
        try FileManager.default.copyItem(
            at: Bundle.swiftMathResources.bundleURL, to: packaged)

        let bundle = try #require(Bundle.packagedSwiftMathResources(in: resources))
        #expect(bundle.bundleURL.standardizedFileURL == packaged.standardizedFileURL)
        let fontsURL = try #require(
            bundle.url(forResource: "mathFonts", withExtension: "bundle"))
        let fonts = try #require(Bundle(url: fontsURL))
        let font = MathFont.latinModernFont.rawValue
        #expect(fonts.url(forResource: font, withExtension: "otf") != nil)
        #expect(fonts.url(forResource: font, withExtension: "plist") != nil)
    }

    @Test func absentPackagedResourcesAreNotAMatch() {
        #expect(Bundle.packagedSwiftMathResources(in: nil) == nil)
        let empty = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        #expect(Bundle.packagedSwiftMathResources(in: empty) == nil)
    }
}
