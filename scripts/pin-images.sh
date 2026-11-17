#!/bin/sh
# Pin the root images of an `images-*` release (made by the "Root images"
# workflow, .github/workflows/images.yml): replace src/image/pinned.txt with
# the release's pinned.txt, after checking it against the release's SHA256SUMS.
#
#   scripts/pin-images.sh images-3.22.2-1
#
# Then commit; the next nixvm release ships the new pins.
set -eu

tag=${1:?usage: scripts/pin-images.sh <images-tag>}
repo=${NIXVM_REPO:-KarpelesLab/nixvm}
base="https://github.com/$repo/releases/download/$tag"
here=$(cd "$(dirname "$0")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

curl -fsSL --retry 5 -o "$tmp/pinned.txt" "$base/pinned.txt"
curl -fsSL --retry 5 -o "$tmp/SHA256SUMS" "$base/SHA256SUMS"

grep -qx "release $tag" "$tmp/pinned.txt" || {
    echo "pin-images: $base/pinned.txt is not for $tag" >&2
    exit 1
}
# Every pinned digest must be the release's.
grep -v '^#' "$tmp/pinned.txt" | grep -v '^release ' | while read -r name ver arch sha size; do
    [ -n "$name" ] || continue
    grep -qx "$sha  $name-$ver-$arch.sqfs" "$tmp/SHA256SUMS" || {
        echo "pin-images: $name-$ver-$arch.sqfs: $sha is not in SHA256SUMS" >&2
        exit 1
    }
done

cp "$tmp/pinned.txt" "$here/src/image/pinned.txt"
echo "Pinned $tag:"
grep -v '^#' "$here/src/image/pinned.txt"
