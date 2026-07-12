# nixvm

[![CI](https://github.com/KarpelesLab/nixvm/actions/workflows/ci.yml/badge.svg)](https://github.com/KarpelesLab/nixvm/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/nixvm.svg)](https://crates.io/crates/nixvm)
[![docs.rs](https://img.shields.io/docsrs/nixvm)](https://docs.rs/nixvm)
[![License: Apache-2.0 OR MIT](https://img.shields.io/badge/license-Apache--2.0%20OR%20MIT-blue.svg)](LICENSE)
[![Browser demo](https://img.shields.io/badge/demo-live-brightgreen.svg)](https://karpeleslab.github.io/nixvm/)

**A portable, VM-style sandbox that runs a real Linux userland by emulating
Linux syscalls directly — not by emulating hardware.**

nixvm lets you run normally-dangerous tasks (`npm install`, `pip install`,
untrusted build scripts, `apk add`, …) inside an isolated Linux environment,
with or without hardware virtualization.

## How it's different

A traditional VM (QEMU, Firecracker) boots a full guest Linux kernel and
emulates hardware: a CPU, interrupt controller, virtio devices, block devices.
That's heavy and slow to start.

nixvm takes the [gVisor](https://gvisor.dev/)-style approach instead: **there
is no guest kernel.** A real Linux *userland* (Alpine) runs directly on the
host CPU or a software interpreter. When a guest process executes a
`syscall`/`svc` instruction, control traps out and **nixvm's own "kernel" —
written in Rust — services the syscall.** Files, memory, processes, threads,
signals, and networking are all implemented in userspace, under our control.

## Try it in your browser

**▶ [karpeleslab.github.io/nixvm](https://karpeleslab.github.io/nixvm/)**

Because the whole thing — the CPU interpreters *and* the syscall kernel — is
portable Rust with no OS dependencies, it compiles to WebAssembly and runs
entirely in the page. There's no server: a real Alpine Linux userland boots
locally, in a sandbox, inside your browser's sandbox.

- **Pick a guest architecture** — arm64 or x86-64 — and press Start: the
  matching stock Alpine minirootfs boots to an interactive `busybox sh`
  (dynamic linking, real `ld-musl`, fork/exec — the same kernel the native
  build runs).
- Everything is client-side: the rootfs is repacked into an in-memory
  squashfs under a copy-on-write tmpfs upper, and the guest runs on the
  software CPU interpreter for whichever arch you picked.

CI rebuilds and redeploys it to GitHub Pages on every push to `master` (see
[`.github/workflows/pages.yml`](.github/workflows/pages.yml)); the frontend
is the Vue app in `web/` and the wasm entry point is `src/wasm.rs` (which
also still exposes the original static-ELF runner, `run_elf`).

## Status

Functional on the software-interpreter path: a real Rust syscall kernel, two
working CPU interpreters (aarch64 and a growing x86-64), static, static-PIE, and
**dynamically-linked** ELF loading (real `ld-musl` boots stock Alpine),
multi-threaded/multi-process scheduling with an SMP worker-thread pool, an in-VM
network stack, and several filesystem backends — all covered by 456 tests
(`cargo test`). Hardware acceleration is live on both target hosts: the **HVF
backend (macOS/arm64) runs a static program end-to-end**, and the **KVM backend
(Linux/x86-64) runs a real statically-linked glibc binary end-to-end on real
hardware** (long mode at ring 3, `syscall` trapped via an `LSTAR` trampoline,
TLS via `arch_prctl`). `vcpu::select` prefers hardware and falls back to the
interpreter, which remains the portable default and the everywhere-CI path.
See [ROADMAP.md](ROADMAP.md) for the phased plan and what's next.

## What works today

| Area | Status |
| --- | --- |
| **Guest architectures** | aarch64 (interpreter, primary target); x86-64 (interpreter, growing) |
| **Execution backends** | software interpreter — **working** (portable, CI-tested). HVF (macOS/arm64) — **runs a static program end-to-end on hardware** (guest MMU-off at EL0, `svc` trapped via an EL1 stub); needs a codesigned binary (`scripts/hvf-test.sh`). KVM (Linux/x86-64) — **runs a real static glibc binary end-to-end on hardware** (guest at CPL3 in long mode over fixed identity page tables, `syscall` → `LSTAR` trampoline → `KVM_EXIT_HLT`); no entitlement needed, its tests run under plain `cargo test` wherever `/dev/kvm` is accessible and skip elsewhere. `vcpu::select` prefers hardware and falls back to the interpreter (`NIXVM_INTERP=1` forces it); dynamic linking + multi-process on the hardware backends are the follow-up. KVM/arm64 — not started. |
| **ELF loading** | static (`ET_EXEC`), static-PIE (`ET_DYN` with `R_*_RELATIVE`/`R_*_IRELATIVE` fixups), and **dynamic linking** (`PT_INTERP` → `ld-musl`, file-backed `mmap` of `.so`s) — working; real Alpine boots on the interpreter. |
| **Processes & threads** | `clone`/`fork`/`execve`/`wait4`/`exit(_group)`; `CLONE_VM`+`CLONE_THREAD` shared-address-space threads; real `futex` WAIT/WAKE parking |
| **SMP scheduler** | a pool of `ncpus` host worker threads run guest compute in parallel; syscalls are serviced serially on the main thread (a big-kernel-lock model, `Kernel::set_ncpus`/`NIXVM_CPUS`) |
| **Signals** | `rt_sigaction`/`rt_sigprocmask`/`kill`/`tgkill` and default-disposition delivery (terminate/ignore) — **custom handler invocation (frame push + PC redirect) is not implemented** |
| **Filesystem** | `tmpfs` (rw in-memory), `overlay` (COW upper/lower), `passthrough` (host dir, **symlink/TOCTOU-safe** — see below), `devfs`, `procfs`, `sysfs`; a real on-disk squashfs/ext reader (`fstoolfs`, via the optional `fstool` crate) exists but isn't wired into the default mount table yet |
| **Networking** | in-VM AF_UNIX + AF_INET/AF_INET6 loopback (TCP stream + UDP datagram); no real host/internet networking yet |
| **I/O multiplexing** | `poll`/`ppoll`/`select`/`pselect6`, `epoll_create1`/`ctl`/`wait`/`pwait2`, `eventfd2`, `timerfd_*` |
| **Browser demo (wasm)** | working — [**live demo**](https://karpeleslab.github.io/nixvm/): `wasm32-unknown-unknown` + interpreter, built and deployed to GitHub Pages by CI on every push to `main` ([details](#try-it-in-your-browser)) |

The syscall dispatch table in `src/kernel/mod.rs` covers process/thread
lifecycle, fd/file I/O, `mmap`/`brk`/`mprotect`/`mremap`, signals, networking,
poll/epoll/eventfd/timerfd, and a set of always-succeed/no-op syscalls
(`uid`/`gid` queries, `sched_*` setters, `Mlock*`, …) real programs probe at
startup. Anything not in the table returns `ENOSYS` and is recorded in an
`unsupported()` ledger you can inspect after a run.

## Quickstart

```sh
cargo build --release
cargo test
```

On macOS/arm64 the HVF backend is exercised by a separate, `#[ignore]`d test
suite (it needs a binary codesigned with the `com.apple.security.hypervisor`
entitlement, which `cargo test` can't do for itself):

```sh
scripts/hvf-test.sh          # builds, ad-hoc codesigns, runs the HVF tests
```

The `nixvm` CLI and the public `Sandbox::run()` API drive the full
image-based pipeline (resolve a cached Alpine squashfs → mount → load → run),
but image fetch/caching is still a stub (ROADMAP Phase 11) and the default
mount table doesn't yet wire in the squashfs reader — so that path isn't
runnable end-to-end yet.

What **does** work today is running a static ELF directly, via the dev
harnesses or the embeddable `Sandbox::exec_elf()`:

```sh
# Run a statically-linked aarch64 ELF on the interpreter:
cargo run --bin run-elf -- ./some-static-aarch64-binary

# Same, for a statically-linked x86-64 ELF:
cargo run --bin run-elf-x86 -- ./some-static-x86_64-binary
```

Useful environment variables (both harnesses):

- `NIXVM_ROOT=/path/to/alpine-rootfs` — mount `/` as
  `overlay(passthrough(NIXVM_ROOT) read-only, tmpfs)` instead of a bare tmpfs,
  so the guest sees a real Alpine tree with a copy-on-write upper.
- `NIXVM_CPUS=N` — run guest compute on `N` host worker threads (the SMP
  scheduler); default is `1` (single-threaded cooperative scheduling).
- `NIXVM_TRACE=1` — log every dispatched syscall (`pid`, `pc`, name, raw
  number, args) to stderr.
- `NIXVM_INTERP=1` — skip the hardware-backend probe in `vcpu::select` and run
  on the software interpreter (a debugging/parity escape hatch).

## Default sandbox layout

```
/           tmpfs by default; overlay(passthrough(NIXVM_ROOT), tmpfs) when set
/work       the host's current working directory, passed through read-write
/tmp,/dev,  synthesized in-sandbox (tmpfs / devfs / procfs / sysfs)
/proc,/sys
```

## Design notes

- **Single crate, feature-gated deps.** Everything lives under one `nixvm`
  crate (`abi`, `vcpu`, `loader`, `fs`, `kernel`, `image`, `sandbox`) instead
  of a workspace; heavy or platform-specific dependencies are opt-in cargo
  features (`hvf`, `kvm`, `interp`, `fstool`, `wasm`, `cli`) rather than
  separate crates. The core builds fully offline with zero third-party
  dependencies.
- **`unsafe` policy.** `unsafe` is confined to four documented sites; the
  interpreter and kernel paths have none. (1) `vcpu::hvf` — the
  Hypervisor.framework FFI (`hv_vm_*`/`hv_vcpu_*`) — and `vcpu::kvm`, its
  Linux equivalent (hand-rolled `/dev/kvm` ioctls, structs verified against
  the kernel headers, plus the mmap'd per-vcpu `kvm_run` page).
  (2) `vcpu::region` — one 16 KiB-aligned `alloc_zeroed`
  allocation of guest RAM (a hardware backend maps its raw pointer into the
  guest; safe `std` can't express over-aligned allocation), with all raw
  pointer access wrapped in checked copy methods. (3) `fs::passthrough` —
  hand-declares a handful of dirfd-relative `*at(2)` FFI calls (`openat`/
  `unlinkat`/`mkdirat`/`symlinkat`/`renameat`/`readlinkat`/`fdopendir`/
  `readdir`) because safe, TOCTOU-free path confinement genuinely requires them
  and `std` exposes none of it. Resolution walks the host path one component at a time
  from a dirfd opened on the mount root, with `O_NOFOLLOW` on every
  component; a symlink is never handed to the kernel to auto-follow — its
  target is read and re-spliced into the walk, re-anchored so an absolute
  target or a `..` chain can never resolve above the mount root. The final,
  actual I/O syscall is always issued directly against `(parent_dirfd, name)`
  with `O_NOFOLLOW`, so even a symlink swapped in mid-race fails safely
  instead of redirecting the I/O.

## License

Apache-2.0 OR MIT, at your option.
