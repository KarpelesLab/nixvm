#!/bin/sh
# Build a guest rootfs that can run Claude Code inside nixvm.
#
#   Alpine minirootfs (musl + busybox + ca-certificates)
#   + @anthropic-ai/claude-code-linux-x64-musl — Claude Code ships as a native
#     single-file executable (a Bun build); the musl variant is a plain
#     dynamically-linked ELF whose only NEEDED is libc.musl, i.e. exactly the
#     ld-musl path nixvm already boots Alpine on.
#   + the caller's ~/.claude config and credentials, copied to the guest's /root
#
# Usage: scripts/build-claude-root.sh [outdir]
# Default outdir: target/claude-root
set -eu

ALPINE_VER=3.20
ALPINE_REL=3.20.0
ARCH=x86_64
MIRROR=https://dl-cdn.alpinelinux.org/alpine
OUT=${1:-target/claude-root}
CACHE=${NIXVM_BUILD_CACHE:-target/.rootcache}

mkdir -p "$CACHE"
rm -rf "$OUT"
mkdir -p "$OUT"

# ---- 1. Alpine base -------------------------------------------------------
BASE="alpine-minirootfs-$ALPINE_REL-$ARCH.tar.gz"
if [ ! -f "$CACHE/$BASE" ]; then
    echo "==> fetching $BASE"
    curl -fsSL -o "$CACHE/$BASE" "$MIRROR/v$ALPINE_VER/releases/$ARCH/$BASE"
fi
echo "==> unpacking Alpine base"
tar xzf "$CACHE/$BASE" -C "$OUT"

# ---- 2. TLS trust store (Claude Code talks HTTPS to the API) --------------
fetch_apk() {
    pkg=$1
    idx="$CACHE/APKINDEX.main"
    if [ ! -f "$idx" ]; then
        curl -fsSL "$MIRROR/v$ALPINE_VER/main/$ARCH/APKINDEX.tar.gz" \
            | tar xzO APKINDEX > "$idx"
    fi
    ver=$(awk -v p="$pkg" '
        /^P:/ { name = substr($0,3) }
        /^V:/ { if (name == p) { print substr($0,3); exit } }
    ' "$idx")
    [ -n "$ver" ] || { echo "!! package $pkg not found" >&2; exit 1; }
    file="$pkg-$ver.apk"
    if [ ! -f "$CACHE/$file" ]; then
        echo "==> fetching $file"
        curl -fsSL -o "$CACHE/$file" "$MIRROR/v$ALPINE_VER/main/$ARCH/$file"
    fi
    tar xzf "$CACHE/$file" -C "$OUT" --exclude='.*' 2>/dev/null || true
}
for p in ca-certificates ca-certificates-bundle; do fetch_apk "$p"; done

# ---- 3. Claude Code (native musl binary) ---------------------------------
echo "==> fetching @anthropic-ai/claude-code-linux-x64-musl"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
( cd "$TMP" && npm pack --silent @anthropic-ai/claude-code-linux-x64-musl >/dev/null )
tar xzf "$TMP"/*.tgz -C "$TMP"
install -Dm755 "$TMP/package/claude" "$OUT/usr/bin/claude"
VER=$(sed -n 's/.*"version": *"\([^"]*\)".*/\1/p' "$TMP/package/package.json" | head -1)
echo "==> claude $VER installed"

# ---- 4. Guest /etc + the caller's Claude config ---------------------------
mkdir -p "$OUT/root" "$OUT/tmp"
chmod 1777 "$OUT/tmp"
printf 'nameserver 1.1.1.1\nnameserver 8.8.8.8\n' > "$OUT/etc/resolv.conf"
printf 'root:x:0:0:root:/root:/bin/sh\n' > "$OUT/etc/passwd"
printf 'root:x:0:\n' > "$OUT/etc/group"
printf 'localhost\n' > "$OUT/etc/hostname"

[ -f "$HOME/.claude.json" ] && cp "$HOME/.claude.json" "$OUT/root/.claude.json"
if [ -d "$HOME/.claude" ]; then
    mkdir -p "$OUT/root/.claude"
    # Credentials + settings only — not the whole history/cache/projects tree.
    for f in .credentials.json settings.json; do
        [ -f "$HOME/.claude/$f" ] && cp "$HOME/.claude/$f" "$OUT/root/.claude/$f"
    done
fi

echo "==> rootfs ready at $OUT  ($(du -sh "$OUT" | cut -f1))"
echo "    run: NIXVM_NET=host cargo run --release -- run --root $OUT -- /usr/bin/claude -p 'say hi'"
