//! Host-platform cfgs the crate keys its OS layer on.
//!
//! `cfg(unix)` / `target_os = "linux"` are not enough to recognise a Linux host:
//! the libc-free `x86_64-unknown-linux-fullrust` target speaks the Linux kernel
//! ABI but is `target_os = "fullrust"` and not in the `unix` family (its std has
//! its own raw-syscall platform layer, but still provides `std::os::{fd, unix,
//! linux}`). So this script works out what the host really is and emits:
//!
//! * `nixvm_linux` — the host runs the Linux kernel ABI on an arch the raw
//!   syscall layer (`src/linux_raw.rs`) knows (x86-64, arm64): `target_os =
//!   "linux"`, `target_os = "fullrust"`, or any OS with the `linux-raw` feature
//!   (a custom libc-free target). Host-directory mounts, KVM, egress socket
//!   options, the tty ioctls and guest-RAM `mmap` then go through raw system
//!   calls, never libc.
//! * `nixvm_libc_free` — on top of that, there is no libc to call at all
//!   (fullrust, or `linux-raw` forced): the few places that still prefer a libc
//!   entry point on a normal Linux build — `clock_gettime`, which libc serves
//!   from the vDSO without entering the kernel — use the raw syscall instead.

use std::env;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(nixvm_linux)");
    println!("cargo::rustc-check-cfg=cfg(nixvm_libc_free)");
    println!("cargo::rerun-if-changed=build.rs");

    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let family = env::var("CARGO_CFG_TARGET_FAMILY").unwrap_or_default();
    let family: Vec<&str> = family.split(',').collect();
    // An OS known not to be Linux (macOS, the BSDs, Windows) keeps its own
    // layer even with `linux-raw` on (an `--all-features` build, say).
    let other_os = (family.contains(&"unix") && os != "linux") || family.contains(&"windows");
    let forced = env::var_os("CARGO_FEATURE_LINUX_RAW").is_some() && !other_os;
    let raw_arch = matches!(arch.as_str(), "x86_64" | "aarch64");
    let fullrust = os == "fullrust";

    if raw_arch && (os == "linux" || fullrust || forced) {
        println!("cargo::rustc-cfg=nixvm_linux");
        if fullrust || forced {
            println!("cargo::rustc-cfg=nixvm_libc_free");
        }
    }
}
