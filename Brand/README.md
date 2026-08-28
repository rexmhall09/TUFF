# TUFF brand source

TUFF Violet is `#6F4DFF`; the wing and beak use `#A78BFA`. The mark is a bird
built from one equilateral triangle and four quarter circles of equal radius:
two quarters make the belly, one makes the shoulder, one makes the raised wing,
and the triangle is the beak sitting on the shoulder's vertical edge. Every
coordinate follows from that construction, so the shape can be rederived rather
than redrawn. `TUFFLogo.svg` is the flat version on black. It has no gradients,
shadows, highlights, outlines, or baked depth.

`TUFF.icon` is the editable 1024 by 1024 Icon Composer source. Its black canvas
and two layers let Icon Composer produce the default, dark, clear, and tinted
system renditions.

Icon Composer rims each layer's own silhouette, so anything that must not have a
glass edge running through it has to share a layer. The beak therefore sits on
the body layer even though it is the lighter violet: the glass edge follows the
outside of the bird, and the beak reads as a colour change rather than a
separate pane. The wing is genuinely detached, so it keeps its own layer and its
own edge. The beak also tucks three units under the body so no hairline shows
along the seam.

`Scripts/package_app.sh` compiles `TUFF.icon` the way Xcode does, with
`actool`. That writes the layered icon into `Assets.car`, which macOS 26 uses to
render the default, dark, clear, and tinted appearances live, and a flattened
`TUFF.icns` beside it for older systems. The script reads both `CFBundleIconName`
and `CFBundleIconFile` out of actool's partial plist rather than hard-coding
them, and fails if either is missing:

```bash
xcrun actool --compile "$build" --platform macosx \
  --minimum-deployment-target 15.0 --app-icon TUFF \
  --output-partial-info-plist "$partial" Brand/TUFF.icon
```

`Sources/TUFFApp/Mac/Resources/tuff-app-icon.png` is no longer what
the app installs as its icon. It is a flattened 1024 preview that the app shows
inside its own interface and that the README uses. Regenerate it after changing
the mark:

```bash
/Applications/Xcode.app/Contents/Applications/Icon\ Composer.app/Contents/Executables/ictool \
  Brand/TUFF.icon --export-image \
  --output-file /tmp/tuff-app-icon-16bit.png \
  --platform macOS --rendition Default --width 1024 --height 1024 --scale 1
magick /tmp/tuff-app-icon-16bit.png -depth 8 \
  PNG32:Sources/TUFFApp/Mac/Resources/tuff-app-icon.png
ruby Scripts/check_brand_assets.rb
```

To preview any single appearance, pass `--rendition` one of `Default`, `Dark`,
`ClearLight`, `ClearDark`, `TintedLight`, or `TintedDark`.
