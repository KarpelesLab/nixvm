#!/bin/sh
# Build an uncompressed x86-64 Alpine root with bubblewrap and util-linux's namespace
# tools, for tests/namespace_tools.rs.
#
#   scripts/alpine-with-bwrap.sh <alpine-minirootfs.tar> <output.tar>
set -eu
base=$1
out=$2
mirror=https://dl-cdn.alpinelinux.org/alpine/v3.20/main/x86_64
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir "$work/root"
tar -xf "$base" -C "$work/root"
curl -fsSL --retry 5 "$mirror/APKINDEX.tar.gz" | tar -xzf - -C "$work" APKINDEX
for name in bubblewrap libcap2 util-linux-misc libblkid libmount libsmartcols; do
    version=$(awk -v n="$name" 'BEGIN { RS = ""; FS = "\n" } {
        p = ""; v = ""
        for (i = 1; i <= NF; i++) {
            if ($i ~ /^P:/) p = substr($i, 3)
            if ($i ~ /^V:/) v = substr($i, 3)
        }
        if (p == n) print v
    }' "$work/APKINDEX")
    curl -fsSL --retry 5 "$mirror/$name-$version.apk" | tar -xzf - -C "$work/root" 2>/dev/null
done
rm -f "$work"/root/.PKGINFO "$work"/root/.SIGN* "$work"/root/.pre-* "$work"/root/.post-*
tar -C "$work/root" -cf "$out" .
