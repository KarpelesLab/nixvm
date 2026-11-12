#!/usr/bin/env bash
# Ad-hoc codesign nixvm binaries with the hypervisor entitlement, so they run
# guests on Hypervisor.framework (macOS/arm64) instead of the interpreter.
#
# Usage: scripts/sign-macos.sh [binary...]   (default: target/release/{nixvm,run-elf})
#
# Without the entitlement `hv_vm_create` is denied and nixvm quietly falls back
# to the software interpreter; check with `NIXVM_INTERP=1` for comparison.
# Any rebuild — including the one `cargo test --release` does of these same
# binaries — replaces them unsigned, so sign again after building.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "codesigning is macOS-only" >&2
  exit 1
fi
bins=("$@")
if [[ ${#bins[@]} -eq 0 ]]; then
  bins=(target/release/nixvm target/release/run-elf)
fi
for b in "${bins[@]}"; do
  codesign --sign - --force --entitlements tests/hvf.entitlements "$b"
  echo "signed $b"
done
