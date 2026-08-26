#!/usr/bin/env bash

set -euo pipefail

version="${1:-}"
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "usage: Scripts/generate_update_appcast.sh VERSION [ARCHIVE_DIRECTORY]" >&2
  exit 64
fi

script_directory="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd -- "$script_directory/.." && pwd)"
archive_directory="${2:-$repository_root/dist}"
archive="TUFF-v${version}-macos-arm64.zip"
generator="$repository_root/.build/artifacts/sparkle/Sparkle/bin/generate_appcast"

if [[ ! -x "$generator" ]]; then
  echo "missing Sparkle appcast generator; run swift package resolve first" >&2
  exit 1
fi
if [[ ! -f "$archive_directory/$archive" ]]; then
  echo "missing packaged update: $archive_directory/$archive" >&2
  exit 1
fi

"$generator" \
  --account TUFF \
  --download-url-prefix \
    "https://github.com/rexmhall09/TUFF/releases/download/v${version}/" \
  --link "https://github.com/rexmhall09/TUFF" \
  --embed-release-notes \
  "$archive_directory"

echo "created $archive_directory/appcast.xml"
