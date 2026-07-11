# nixvm

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

Because the whole thing — the CPU interpreter *and* the syscall kernel — is
portable Rust with no OS dependencies, it compiles to WebAssembly and runs
entirely in the page. There's no server: the ELF you give it is executed
locally, in a sandbox, inside your browser's sandbox.

- **Drag in (or pick) a static Linux ELF** — aarch64 or x86-64; the arch is
  detected from the ELF header and the matching interpreter runs it.
- **Bundled samples** let you see output with zero setup.
- You get the program's **stdout/stderr, its exit code, and the ledger of any
  syscalls it hit that nixvm doesn't implement yet** — the same
  `unsupported()` view you'd get from the native `run-elf` harness.

It's a try-before-you-install demo, so it's deliberately minimal: a
statically-linked ELF runner (dynamic linking and a full Alpine shell run on
the native build, not the wasm demo yet). CI rebuilds and redeploys it to
GitHub Pages on every push to `main` (see
[`.github/workflows/pages.yml`](.github/workflows/pages.yml)); the page is
`web/index.html` and the wasm entry point is `src/wasm.rs`.

## Status

Early but functional on the software-interpreter path: a real Rust syscall
kernel, two working CPU interpreters (aarch64 and a growing x86-64), static +
static-PIE ELF loading, multi-threaded/multi-process scheduling with an SMP
worker-thread pool, an in-VM network stack, and several filesystem backends —
all covered by 253 unit tests + 8 integration tests + 1 doctest (`cargo test`).
Hardware acceleration (HVF, KVM) is not wired up yet; everything below runs on
the portable interpreter today. See [ROADMAP.md](ROADMAP.md) for the phased
plan and what's next.

## What works today

| Area | Status |
| --- | --- |
| **Guest architectures** | aarch64 (interpreter, primary target); x86-64 (interpreter, growing) |
| **Execution backends** | software interpreter — **working**. HVF (macOS/arm64) — module scaffolded, `new_vcpu` still returns "not implemented" (planned). KVM (Linux) — not started (planned). |
| **ELF loading** | static (`ET_EXEC`) and static-PIE (`ET_DYN` with `R_*_RELATIVE`/`R_*_IRELATIVE` fixups) — working. Dynamic linking (`PT_INTERP` → `ld-musl`) — **not implemented** (planned). |
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
- **`unsafe` policy.** `unsafe` is meant to live only in the hardware vcpu
  backends (`vcpu::hvf`, later `vcpu::kvm`) — the interpreter path has none.
  One deliberate, documented exception: `fs::passthrough` hand-declares a
  handful of dirfd-relative `*at(2)` FFI calls (`openat`/`unlinkat`/
  `mkdirat`/`symlinkat`/`renameat`/`readlinkat`/`fdopendir`/`readdir`) because
  safe, TOCTOU-free path confinement genuinely requires them and `std`
  exposes none of it. Resolution walks the host path one component at a time
  from a dirfd opened on the mount root, with `O_NOFOLLOW` on every
  component; a symlink is never handed to the kernel to auto-follow — its
  target is read and re-spliced into the walk, re-anchored so an absolute
  target or a `..` chain can never resolve above the mount root. The final,
  actual I/O syscall is always issued directly against `(parent_dirfd, name)`
  with `O_NOFOLLOW`, so even a symlink swapped in mid-race fails safely
  instead of redirecting the I/O.

## License

Apache-2.0 OR MIT, at your option.
