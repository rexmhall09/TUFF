#!/usr/bin/env bash

set -euo pipefail

version="${1:-}"
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "usage: Scripts/package_app.sh VERSION [OUTPUT_DIRECTORY]" >&2
  exit 64
fi

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "$script_directory/.." && pwd)"
output_directory="${2:-$repository_root/dist}"
mkdir -p "$output_directory"
output_directory="$(cd -- "$output_directory" && pwd)"

app="$output_directory/TUFF.app"
archive_name="TUFF-v${version}-macos-arm64.zip"
archive="$output_directory/$archive_name"
checksum="$archive.sha256"
update_feed_url="${TUFF_UPDATE_FEED_URL:-https://github.com/rexmhall09/TUFF/releases/latest/download/appcast.xml}"
update_public_key="${TUFF_UPDATE_PUBLIC_ED_KEY:-}"
update_public_key_file="$repository_root/Config/UpdateSigningPublicKey.txt"
if [[ -z "$update_public_key" && -f "$update_public_key_file" ]]; then
  IFS= read -r update_public_key < "$update_public_key_file"
fi

rm -rf "$app"
rm -f "$archive" "$checksum"

temporary_directory="$(mktemp -d "${TMPDIR:-/tmp}/tuff-package.XXXXXX")"
trap 'rm -rf "$temporary_directory"' EXIT

cd "$repository_root"
ruby Scripts/check_brand_assets.rb
swift build -c release
binary_directory="$(swift build -c release --show-bin-path)"

required_binaries=(TUFF TUFFDecodeService)
required_bundles=(
  TurboFieldfare_TurboFieldfare.bundle
  TurboFieldfare_TurboFieldfareAppCore.bundle
  TurboFieldfare_TurboFieldfareMac.bundle
  swift-crypto_Crypto.bundle
  swift-nio_NIOPosix.bundle
  swift-transformers_Hub.bundle
  SwiftMath_SwiftMath.bundle
)
required_frameworks=(Sparkle.framework)

for name in "${required_binaries[@]}" "${required_bundles[@]}" \
  "${required_frameworks[@]}"; do
  if [[ ! -e "$binary_directory/$name" ]]; then
    echo "missing release product: $binary_directory/$name" >&2
    exit 1
  fi
done

mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources" \
  "$app/Contents/Frameworks"
signed_binaries="$temporary_directory/signed-binaries"
mkdir -p "$signed_binaries"
install -m 0755 "$binary_directory/TUFF" "$signed_binaries/TUFF"
install -m 0755 "$binary_directory/TUFFDecodeService" \
  "$signed_binaries/TUFFDecodeService"
if ! otool -l "$signed_binaries/TUFF" \
  | grep -Fq '@executable_path/../Frameworks'; then
  install_name_tool -add_rpath '@executable_path/../Frameworks' \
    "$signed_binaries/TUFF"
fi
codesign --force --sign - --timestamp=none "$signed_binaries/TUFF"
codesign --force --sign - --timestamp=none "$signed_binaries/TUFFDecodeService"
codesign --verify --strict --verbose=2 "$signed_binaries/TUFF"
codesign --verify --strict --verbose=2 "$signed_binaries/TUFFDecodeService"
install -m 0755 "$signed_binaries/TUFF" "$app/Contents/MacOS/TUFF"
install -m 0755 "$signed_binaries/TUFFDecodeService" \
  "$app/Contents/MacOS/TUFFDecodeService"

# Keep resource bundles in the standard sealed app resource directory. TUFF's
# resource lookups prefer this location in a packaged app and fall back to
# SwiftPM's generated Bundle.module path for command-line and test builds.
for name in "${required_bundles[@]}"; do
  ditto "$binary_directory/$name" "$app/Contents/Resources/$name"
done
for name in "${required_frameworks[@]}"; do
  ditto "$binary_directory/$name" "$app/Contents/Frameworks/$name"
done

install -m 0644 LICENSE "$app/Contents/Resources/LICENSE"
install -m 0644 README.md "$app/Contents/Resources/README.md"
install -m 0644 THIRD_PARTY_NOTICES.md \
  "$app/Contents/Resources/THIRD_PARTY_NOTICES.md"

iconset="$temporary_directory/TUFF.iconset"
mkdir -p "$iconset"
icon_source="Sources/TurboFieldfareApp/Mac/Resources/tuff-app-icon.png"

make_icon() {
  local pixels="$1"
  local filename="$2"
  sips -z "$pixels" "$pixels" "$icon_source" \
    --out "$iconset/$filename" >/dev/null
}

make_icon 16 icon_16x16.png
make_icon 32 icon_16x16@2x.png
make_icon 32 icon_32x32.png
make_icon 64 icon_32x32@2x.png
make_icon 128 icon_128x128.png
make_icon 256 icon_128x128@2x.png
make_icon 256 icon_256x256.png
make_icon 512 icon_256x256@2x.png
make_icon 512 icon_512x512.png
make_icon 1024 icon_512x512@2x.png
iconutil -c icns "$iconset" -o "$app/Contents/Resources/TUFF.icns"

info_plist="$app/Contents/Info.plist"
plutil -create xml1 "$info_plist"
plutil -insert CFBundleDevelopmentRegion -string en "$info_plist"
plutil -insert CFBundleDisplayName -string TUFF "$info_plist"
plutil -insert CFBundleExecutable -string TUFF "$info_plist"
plutil -insert CFBundleIconFile -string TUFF.icns "$info_plist"
plutil -insert CFBundleIdentifier -string com.rexmhall09.TUFF "$info_plist"
plutil -insert CFBundleInfoDictionaryVersion -string 6.0 "$info_plist"
plutil -insert CFBundleName -string TUFF "$info_plist"
plutil -insert CFBundlePackageType -string APPL "$info_plist"
plutil -insert CFBundleShortVersionString -string "$version" "$info_plist"
plutil -insert CFBundleVersion -string "$version" "$info_plist"
plutil -insert LSApplicationCategoryType -string public.app-category.utilities \
  "$info_plist"
plutil -insert LSMinimumSystemVersion -string 15.0 "$info_plist"
plutil -insert LSRequiresNativeExecution -bool true "$info_plist"
plutil -insert NSHighResolutionCapable -bool true "$info_plist"
if [[ -n "$update_public_key" ]]; then
  if ! ruby -rbase64 -e \
    'decoded = Base64.strict_decode64(ARGV.fetch(0)); exit(decoded.bytesize == 32 ? 0 : 1)' \
    "$update_public_key" 2>/dev/null; then
    echo "invalid Sparkle EdDSA public key" >&2
    exit 1
  fi
  if [[ "$update_feed_url" != https://* ]]; then
    echo "update feed must use HTTPS" >&2
    exit 1
  fi
  plutil -insert SUFeedURL -string "$update_feed_url" "$info_plist"
  plutil -insert SUPublicEDKey -string "$update_public_key" "$info_plist"
  plutil -insert SUEnableAutomaticChecks -bool true "$info_plist"
  plutil -insert SUAllowsAutomaticUpdates -bool true "$info_plist"
  plutil -insert SUAutomaticallyUpdate -bool true "$info_plist"
else
  echo "warning: no Sparkle update public key; in-app updates are disabled" >&2
fi

plutil -lint "$info_plist"
if [[ -n "$update_public_key" ]]; then
  [[ "$(plutil -extract SUEnableAutomaticChecks raw "$info_plist")" == "true" ]]
  [[ "$(plutil -extract SUAllowsAutomaticUpdates raw "$info_plist")" == "true" ]]
  [[ "$(plutil -extract SUAutomaticallyUpdate raw "$info_plist")" == "true" ]]
fi

main_architectures="$(lipo -archs "$app/Contents/MacOS/TUFF")"
service_architectures="$(lipo -archs "$app/Contents/MacOS/TUFFDecodeService")"
if [[ "$main_architectures" != "arm64" || "$service_architectures" != "arm64" ]]; then
  echo "release app must contain arm64-only executables" >&2
  exit 1
fi

# Sign after the complete bundle has been assembled so Info.plist and resources
# are sealed as part of the app rather than leaving a standalone executable
# signature inside an otherwise unsigned bundle.
codesign --force --deep --sign - --timestamp=none \
  "$app/Contents/Frameworks/Sparkle.framework"
codesign --force --deep --sign - --timestamp=none "$app"
codesign --verify --deep --strict --verbose=2 "$app"

ditto -c -k --sequesterRsrc --keepParent "$app" "$archive"
(
  cd "$output_directory"
  shasum -a 256 "$archive_name" > "$archive_name.sha256"
)

extracted="$temporary_directory/extracted"
mkdir -p "$extracted"
ditto -x -k "$archive" "$extracted"
codesign --verify --deep --strict --verbose=2 "$extracted/TUFF.app"
test -f "$extracted/TUFF.app/Contents/Frameworks/Sparkle.framework/Sparkle"
otool -L "$extracted/TUFF.app/Contents/MacOS/TUFF" \
  | grep -Fq '@rpath/Sparkle.framework/Versions/B/Sparkle'

echo "created $archive"
echo "created $checksum"
