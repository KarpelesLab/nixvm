#!/usr/bin/env bash
# Run the HVF (Hypervisor.framework) tests on macOS/arm64.
#
# HVF needs the `com.apple.security.hypervisor` entitlement, so the test binary
# must be codesigned before it runs — which `cargo test` cannot do for itself.
# This builds the test binary, ad-hoc signs it with tests/hvf.entitlements, then
# runs the (otherwise `#[ignore]`d) HVF tests with NIXVM_HVF=1.
#
# Usage: scripts/hvf-test.sh [extra `cargo test --no-run` args] [-- <test filter>]
# Examples:
#   scripts/hvf-test.sh                      # all HVF lib tests
#   scripts/hvf-test.sh -- bringup           # just the bring-up test
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$(uname -sm)" != "Darwin arm64" ]]; then
  echo "HVF tests only run on macOS/arm64 (this is $(uname -sm))." >&2
  exit 1
fi

# Split args into cargo-build args (before `--`) and a test filter (after `--`).
build_args=()
filter=()
seen_sep=0
for a in "$@"; do
  if [[ $seen_sep == 0 && "$a" == "--" ]]; then seen_sep=1; continue; fi
  if [[ $seen_sep == 0 ]]; then build_args+=("$a"); else filter+=("$a"); fi
done

echo "==> building lib test binary"
bin=$(cargo test --lib --no-run --message-format=json "${build_args[@]}" 2>/dev/null \
  | python3 -c "import sys,json
for line in sys.stdin:
    try: o=json.loads(line)
    except Exception: continue
    if o.get('profile',{}).get('test') and o.get('target',{}).get('name')=='nixvm' and o.get('executable'):
        print(o['executable'])" | head -1)

if [[ -z "${bin:-}" || ! -x "$bin" ]]; then
  echo "could not locate the test binary" >&2
  exit 1
fi

echo "==> codesigning $bin"
codesign --sign - --force --entitlements tests/hvf.entitlements "$bin"

echo "==> running HVF tests (NIXVM_HVF=1)"
NIXVM_HVF=1 "$bin" --ignored --nocapture "${filter[@]}"
