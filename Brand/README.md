# TUFF brand source

TUFF Violet is `#6F4DFF`. `TUFFLogo.svg` is the transparent black-and-violet mark for documentation and other flat uses.

`TUFF.icon` is the editable 1024 by 1024 Icon Composer source. Its graphite canvas and three separate violet facet groups produce the default, dark, clear, and tinted system renditions. Open it in Icon Composer to tune material settings without baking blur, shadows, or highlights into the SVG artwork.

The packaged app uses `Sources/TurboFieldfareApp/Mac/Resources/tuff-app-icon.png` as its flattened compatibility source. Apple’s renderer exports a 16-bit PNG, while `iconutil` requires an 8-bit source for the largest iconset member. Regenerate and normalize it with:

```bash
/Applications/Xcode.app/Contents/Applications/Icon\ Composer.app/Contents/Executables/ictool \
  Brand/TUFF.icon --export-image \
  --output-file /tmp/tuff-app-icon-16bit.png \
  --platform macOS --rendition Default --width 1024 --height 1024 --scale 1
magick /tmp/tuff-app-icon-16bit.png -depth 8 \
  PNG32:Sources/TurboFieldfareApp/Mac/Resources/tuff-app-icon.png
ruby Scripts/check_brand_assets.rb
```
