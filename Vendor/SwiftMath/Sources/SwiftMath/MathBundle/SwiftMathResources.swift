//
//  SwiftMathResources.swift
//
//  Added for TUFF; not part of upstream SwiftMath. See Vendor/SwiftMath/README.md.
//

import Foundation

extension Bundle {
    /// The bundle that holds `mathFonts.bundle`.
    ///
    /// SwiftPM's generated `Bundle.module` looks beside the main bundle and at
    /// the absolute path of the build directory that compiled it. For an app,
    /// "beside the main bundle" is the root of the `.app`, where code signing
    /// allows nothing but `Contents`, so a packaged app keeps this bundle in
    /// `Contents/Resources`, and `Bundle.module` finds it only while that
    /// original build directory still exists. When it does not, `Bundle.module`
    /// stops the process the first time a formula is rendered.
    static let swiftMathResources: Bundle =
        packagedSwiftMathResources(in: Bundle.main.resourceURL) ?? .module

    /// The SwiftMath resource bundle inside `resources`, if one is there.
    /// SwiftPM names it `<package>_<target>.bundle`. The name is spelled out
    /// rather than read from `Bundle.module`, because touching `Bundle.module`
    /// in a packaged app is the failure this avoids.
    static func packagedSwiftMathResources(in resources: URL?) -> Bundle? {
        guard let resources else { return nil }
        return Bundle(url: resources.appendingPathComponent(
            "SwiftMath_SwiftMath.bundle", isDirectory: true))
    }
}
