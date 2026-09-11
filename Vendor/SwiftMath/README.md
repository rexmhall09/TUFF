# SwiftMath (vendored)

This is [SwiftMath](https://github.com/mgriebling/SwiftMath) 1.7.3, revision
`fa8244ed032f4a1ade4cb0571bf87d2f1a9fd2d7`, under its [MIT license](LICENSE),
with two changes for TUFF.

**Resource lookup.** `Sources/SwiftMath/MathBundle/SwiftMathResources.swift`
finds SwiftMath's resource bundle in the app's `Contents/Resources` before
falling back to SwiftPM's `Bundle.module`, and the three places that load
`mathFonts.bundle` use it. SwiftPM's generated accessor checks only beside the
main bundle and the absolute build directory. Beside the main bundle means the
root of the `.app`, where code signing allows nothing but `Contents`. Every
packaged TUFF therefore loaded its math fonts from the build directory of the
Mac that built it. On any other Mac, or once that directory was cleaned, the
first formula in a chat stopped the app. TUFF 5.0.0 would not open when a
restored chat contained math.

**Fonts.** `mathFonts.bundle` keeps only Latin Modern Math, the font
`MathImage` uses by default, with its GUST Font License and the MathChat
license for the math tables. The other eleven fonts, the upstream tests, and
the test target are left out.

No upstream release changes the lookup. To update, copy the new release over
this directory, then reapply both changes.
