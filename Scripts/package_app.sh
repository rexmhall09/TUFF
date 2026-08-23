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

rm -rf "$app"
rm -f "$archive" "$checksum"

temporary_directory="$(mktemp -d "${TMPDIR:-/tmp}/tuff-package.XXXXXX")"
trap 'rm -rf "$temporary_directory"' EXIT

cd "$repository_root"
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
)

for name in "${required_binaries[@]}" "${required_bundles[@]}"; do
  if [[ ! -e "$binary_directory/$name" ]]; then
    echo "missing release product: $binary_directory/$name" >&2
    exit 1
  fi
done

mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
signed_binaries="$temporary_directory/signed-binaries"
mkdir -p "$signed_binaries"
install -m 0755 "$binary_directory/TUFF" "$signed_binaries/TUFF"
install -m 0755 "$binary_directory/TUFFDecodeService" \
  "$signed_binaries/TUFFDecodeService"
codesign --force --sign - --timestamp=none "$signed_binaries/TUFF"
codesign --force --sign - --timestamp=none "$signed_binaries/TUFFDecodeService"
codesign --verify --strict --verbose=2 "$signed_binaries/TUFF"
codesign --verify --strict --verbose=2 "$signed_binaries/TUFFDecodeService"
install -m 0755 "$signed_binaries/TUFF" "$app/Contents/MacOS/TUFF"
install -m 0755 "$signed_binaries/TUFFDecodeService" \
  "$app/Contents/MacOS/TUFFDecodeService"

# SwiftPM's generated Bundle.module accessors look beside Bundle.main.bundleURL.
# For a packaged app that is the .app root, so keep these bundles there rather
# than moving them under Contents/Resources.
for name in "${required_bundles[@]}"; do
  ditto "$binary_directory/$name" "$app/$name"
done

install -m 0644 LICENSE "$app/Contents/Resources/LICENSE"
install -m 0644 README.md "$app/Contents/Resources/README.md"

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

plutil -lint "$info_plist"

main_architectures="$(lipo -archs "$app/Contents/MacOS/TUFF")"
service_architectures="$(lipo -archs "$app/Contents/MacOS/TUFFDecodeService")"
if [[ "$main_architectures" != "arm64" || "$service_architectures" != "arm64" ]]; then
  echo "release app must contain arm64-only executables" >&2
  exit 1
fi

ditto -c -k --sequesterRsrc --keepParent "$app" "$archive"
(
  cd "$output_directory"
  shasum -a 256 "$archive_name" > "$archive_name.sha256"
)

extracted="$temporary_directory/extracted"
mkdir -p "$extracted"
ditto -x -k "$archive" "$extracted"
install -m 0755 "$extracted/TUFF.app/Contents/MacOS/TUFF" \
  "$temporary_directory/extracted-TUFF"
install -m 0755 "$extracted/TUFF.app/Contents/MacOS/TUFFDecodeService" \
  "$temporary_directory/extracted-TUFFDecodeService"
codesign --verify --strict --verbose=2 "$temporary_directory/extracted-TUFF"
codesign --verify --strict --verbose=2 \
  "$temporary_directory/extracted-TUFFDecodeService"

echo "created $archive"
echo "created $checksum"
