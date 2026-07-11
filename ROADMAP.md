# nixvm — Roadmap

nixvm is a portable, VM-style sandbox that runs a **real Linux userland by
emulating Linux syscalls directly** — no guest kernel, no device/interrupt
emulation. This document is the plan: the architecture, the phased milestones,
and the exit criteria that tell us a phase is done.

> Reference: an adjacent project, `univdreams`, already emulates the Linux
> syscall surface (for reverse engineering) with a proven **engine/adapter**
> split and a **mount-table VFS**. nixvm reuses those patterns but targets a
> *portable sandbox/jail*, and fills the gaps univdreams left open: a
> **Hypervisor.framework backend**, **squashfs**, **host passthrough**, and a
> **copy-on-write overlay**.

---

## 1. Architecture

### 1.1 The core idea

A traditional VM boots a guest kernel and emulates hardware. nixvm does neither.
It runs guest *user* code (Alpine's busybox, apk, node, …) directly on the CPU
at the lowest privilege level, and the instant that code executes a syscall
(`svc #0` on arm64, `syscall` on x86-64) the CPU **traps out to the host**.
nixvm's Rust "kernel" services the syscall — files, memory, processes, signals,
sockets — entirely in userspace, then resumes the guest. This is the
[gVisor](https://gvisor.dev/) model, implemented in Rust.

```
        guest process (Alpine userland, ring3/EL0)
                   │  svc #0 / syscall  →  TRAP (VM exit)
                   ▼
     ┌───────────────────────────────┐
     │  nixvm kernel  (crate: nixvm) │
     │  ── syscall dispatch ──────────│   services the call against:
     │  fd table · mm · signals · …   │     · fs::MountTable  (files)
     └───────────────────────────────┘     · vcpu::GuestMemory (mem)
                   ▲  set return reg, resume
                   │
        run again via vcpu backend (HVF / KVM / interp)
```

### 1.2 Module seams (single crate, `nixvm`)

| Module          | Responsibility                                                        |
| --------------- | --------------------------------------------------------------------- |
| `abi`           | The Linux ABI as *data*: `Errno`, per-arch syscall tables → `Sysno`.  |
| `vcpu`          | Execution backends (`hvf`, `kvm`, `interp`) behind the `Vcpu` trait.  |
| `vcpu::mem`     | `GuestMemory` — the guest address space (mapping, protections).       |
| `loader`        | ELF64 loading, stack + auxv, dynamic-linker (`PT_INTERP`) handoff.     |
| `fs`            | `MountTable` + `MountFs` backends (squashfs/overlay/passthrough/…).    |
| `kernel`        | Arch-agnostic syscall engine, fd table, process/thread state.         |
| `image`         | Resolve/download/verify/cache guest root images (Alpine squashfs).    |
| `sandbox`       | Public `Sandbox` builder wiring the pipeline together.                |

Design rules carried from univdreams:

- **Engine/adapter split.** Handlers are written once against the normalized
  `Sysno` enum and the `Vcpu`/`GuestMemory`/`MountFs` trait seams. The guest
  arch and the concrete backend are invisible to handler code.
- **`unsafe` is quarantined** to the hardware backends (`vcpu::hvf`, later
  `vcpu::kvm`). Everything else is safe Rust; the interpreter path has no
  `unsafe` and no heavy deps.
- **Heavy/platform deps are feature-gated**, not split into crates: `hvf`,
  `kvm`, `interp`, `cli`.
- **Read-only-by-default filesystems.** `MountFs` requires only `stat`,
  `read_at`, `readdir`; every mutation defaults to `EROFS`.

### 1.3 Default sandbox layout

```
/           overlay:  read-only squashfs (Alpine)  +  writable tmpfs upper (ephemeral, COW)
/work       passthrough to the host's current working directory (read-write)
/tmp        tmpfs
/proc,/sys  synthesized procfs / sysfs
/dev        devtmpfs (null, zero, full, random, urandom, tty, pts)
```

### 1.4 Backend & arch matrix

| Host              | Backend | Guest arch     | Phase |
| ----------------- | ------- | -------------- | ----- |
| macOS / arm64     | HVF     | arm64          | 1     |
| Linux / arm64     | KVM     | arm64          | 10    |
| Linux / x86-64    | KVM     | x86-64         | 10    |
| anywhere          | interp  | arm64 / x86-64 | 10    |

The primary development target is **macOS/arm64 + HVF + arm64 Alpine**.

---

## 2. Phases

Each phase is a vertical slice that ends in something runnable and testable.
"Syscalls" lists the *new* surface introduced. Numbers are guidance, not
contracts.

> **Build order note (in progress, 2026-07-11).** The **interpreter path is
> being built first**, ahead of HVF: the hardware backend needs macOS
> entitlements and can't run in CI, whereas the interpreter makes the entire
> syscall engine testable on any machine (and is exactly what the wasm demo
> needs). So Phase 1's *observable outcome* (a guest makes a syscall and exits)
> and much of Phase 2/3 already work on the interpreter, with a growing subset
> of Phase 10's ISA, while `vcpu::hvf` remains a stub. Every step ships with
> tests. Status is marked per-phase below.

### Phase 0 — Scaffold ✅ (this commit)

Workspace-free single crate; module seams (`abi`, `vcpu`, `loader`, `fs`,
`kernel`, `image`, `sandbox`); normalized `Sysno` + per-arch decode tables;
`Vcpu`/`Backend`/`MountFs` traits; `Sandbox` builder wiring the full pipeline to
its first unimplemented frontier; `nixvm` CLI (`run`/`shell`/`version`).

- **Exit criteria:** `cargo build`, `cargo test`, `cargo clippy` all clean;
  `nixvm run -- <cmd>` walks the pipeline and reports the current frontier.

### Phase 1 — HVF backend + first syscall  🟡 outcome met on interpreter; HVF pending

Bring up Hypervisor.framework on macOS/arm64. Create a VM, map a flat
`GuestMemory`, create a vcpu at EL0/EL1, and trap `svc #0` into `Exit::Syscall`.
Hand-load a tiny **static** arm64 program (raw bytes, no ELF yet) that does
`write(1, "hi\n", 3); exit_group(0)`.

- **New:** `vcpu::hvf` FFI (`hv_vm_*`, `hv_vcpu_*`), ESR decode, register
  get/set, PC advance. The crate's first `unsafe`.
- **Syscalls:** `write` (to host stdio only), `exit_group`, `exit`.
- **Exit criteria:** a static arm64 blob prints to stdout and exits with a
  chosen code, entirely through the HVF run/serve loop.

### Phase 2 — Memory manager + static ELF loader  ✅ (interpreter path)

Replace the flat stub with a page-granular `GuestMemory` (region tree,
protections, host-backed pages mapped into HVF). Implement `loader::load_static`
for ELF64: map `PT_LOAD`, build the initial stack (`argc`/`argv`/`envp`/auxv),
report entry + SP. Wire `brk`/`mmap`(anon)/`munmap`/`mprotect`.

- **New:** `object` (ELF parsing) behind a feature; `GuestMemory::{read,write,map,protect}`.
- **Syscalls:** `brk`, `mmap`(anon), `munmap`, `mprotect`, `set_tid_address`,
  `set_robust_list`, `rt_sigprocmask` (stub), `getrandom` (for `AT_RANDOM`).
- **Exit criteria:** a **statically-linked musl** `busybox echo`/`true` runs from
  a real ELF and exits correctly.

### Phase 3 — Syscall breadth for static binaries  🟡 in progress

Enough of the syscall surface to run non-trivial static programs against an
in-memory VFS. Reads/writes of guest pointers go through `GuestMemory`; file ops
go through `MountTable` (backed by a temporary in-memory fs until Phase 4).

- **Syscalls:** `read`, `readv`, `writev`, `openat`, `close`, `lseek`,
  `fstat`/`newfstatat`/`statx`, `getdents64`, `readlinkat`, `ioctl`(TCGETS/TIOCGWINSZ),
  `uname`, `getpid`/`gettid`/`getuid`/`getgid`/`geteuid`/`getegid`,
  `clock_gettime`, `gettimeofday`, `nanosleep`, `sched_yield`, `fcntl`,
  `sysinfo`, `prlimit64`.
- **Exit criteria:** static `busybox` multi-applet (`ls`, `cat`, `sha256sum`)
  runs against a seeded in-memory fs; `strace`-level parity on the covered set.

### Phase 4 — Real filesystem: squashfs + overlay + passthrough

The actual root. Implement the `MountFs` backends and compose them:

- `squashfs` — read-only reader for the Alpine root image (own reader or
  `backhand`).
- `tmpfs` — in-memory read-write (overlay upper, `/tmp`).
- `overlay` — copy-up semantics over `(lower=squashfs, upper=tmpfs)`.
- `passthrough` — host directory ↔ `/work`, read-write, with path sandboxing
  (no escaping the mapped root; symlink containment).

- **New:** squashfs dep; `rustix`/`libc` for passthrough host I/O.
- **Syscalls:** write side — `write`(files), `mkdirat`, `unlinkat`, `renameat2`,
  `symlinkat`, `linkat`, `ftruncate`, `fchmodat`, `fchownat`, `utimensat`,
  `statfs`, `getcwd`, `chdir`, `fchdir`, `faccessat2`, `umask`.
- **Exit criteria:** `nixvm run -- sh -c 'ls -l / && echo hi > /work/out && cat /work/out'`
  reads the real Alpine root and writes a file visible on the host.

### Phase 5 — Dynamic linking

Load `PT_INTERP` (`ld-musl-aarch64.so.1`) from the guest rootfs, map file-backed
segments, and provide a **vDSO** (`clock_gettime`/`gettimeofday`/`time`/
`getcpu`) plus `AT_SYSINFO_EHDR`. TLS setup (`TPIDR_EL0` / `arch_prctl` on x86).

- **Syscalls:** `mmap`(file-backed), `mremap`, `madvise`, `arch_prctl`(x86),
  `rseq` (stub/handle), `membarrier`.
- **Exit criteria:** dynamically-linked `/bin/sh` and `/bin/ls` from stock
  Alpine run to completion.

### Phase 6 — Processes, threads, signals

The hard core. A process/thread table; `clone`/`clone3` for both threads
(shared address space) and processes (`fork` via COW); a scheduler mapping guest
threads onto host vcpus/threads; futexes; signal delivery and return.

**State partitioning (drives a `Kernel` refactor).** Today `Kernel` holds the
fd table, `GuestMemory`, brk/mmap arena, and `cwd` as one flat process. That
splits into three layers:

- **Task (per thread):** its own **vcpu** (registers/pc/sp — *one vcpu per
  thread*), its own **cwd**, `clear_child_tid`, signal mask. The scheduler owns
  the task table and runs each task's vcpu.
- **Process (shared by a thread group):** address space (`GuestMemory`), fd
  table, brk/mmap arena, signal handlers, exit state. `clone(CLONE_VM|CLONE_FILES
  |CLONE_THREAD)` shares these; `fork` copies them (COW `GuestMemory`).
- **Kernel-global:** the mount table and the scheduler.

The **scheduler** (`kernel::sched`) replaces the single-vcpu `Kernel::run` loop:
round-robin (or step-budget preemptive) across ready tasks, `Exit::Interrupted`
as the yield point, futex/`wait4` as block/wake edges — mirroring univdreams'
per-thread `Cpu` loop.

- **New:** scheduler (`kernel::sched`), `Task`/`Process` split, per-task cwd,
  per-thread vcpu ownership, COW fork of `GuestMemory`.
- **Syscalls:** `clone`/`clone3`, `fork`/`vfork`, `execve`/`execveat`, `wait4`,
  `exit` (thread), `futex`(WAIT/WAKE/REQUEUE/PI subset), `tgkill`/`kill`,
  `rt_sigaction`, `rt_sigprocmask`, `rt_sigreturn`, `rt_sigpending`,
  `rt_sigtimedwait`, `sigaltstack`, `getpgid`/`setpgid`/`setsid`.
- **Exit criteria:** a shell script that spawns subprocesses and pipelines runs;
  `busybox sh` job control basics; `apk` reaches network (fails cleanly until
  Phase 8).

### Phase 7 — /proc, /sys, /dev, and IO multiplexing

Synthesized pseudo-filesystems and the fd machinery real programs assume.

- **New:** `fs::procfs`, `fs::sysfs`, `fs::devfs` backends.
- **Content:** `/proc/self/{maps,exe,fd,cmdline,status,stat}`, `/proc/cpuinfo`,
  `/proc/meminfo`, `/proc/mounts`, `/sys` minimal; `/dev/{null,zero,full,random,
  urandom,tty}`, `/dev/pts` + a pty.
- **Syscalls:** `pipe2`, `dup`/`dup2`/`dup3`, `poll`/`ppoll`, `pselect6`,
  `epoll_create1`/`epoll_ctl`/`epoll_pwait`, `eventfd2`, `signalfd4`,
  `timerfd_*`, `inotify_*` (stub), `memfd_create`, `close_range`.
- **Exit criteria:** programs using epoll and ptys work (`bash -i`, a
  select/poll-based server loop locally).

### Phase 8 — Networking

A socket layer. Start with loopback + Unix sockets in-process; then egress via a
userspace TCP/IP stack (`smoltcp`) NAT'd to the host, or host-socket passthrough
under policy. DNS.

- **New:** `kernel::net`, address translation, per-sandbox network policy
  (off / loopback-only / NAT).
- **Syscalls:** `socket`, `socketpair`, `bind`, `listen`, `accept4`, `connect`,
  `send*`/`recv*`, `getsockopt`/`setsockopt`, `getsockname`/`getpeername`,
  `shutdown`, `getaddrinfo` path (`/etc/resolv.conf` + UDP:53).
- **Exit criteria:** `apk update && apk add <pkg>` and `npm install <small pkg>`
  complete over the network inside the sandbox.

### Phase 9 — Resource limits & isolation policy

Turn it into a real *jail*: enforce the limits that make running dangerous tasks
safe.

- **Limits:** guest RAM ceiling (already sized) with real accounting; CPU time /
  wall-clock deadline; max pids/threads; max open fds; disk quota on the overlay
  upper; `prlimit64` honored.
- **Policy:** syscall-filter policy (allow/deny/log, gVisor-style), no-network
  mode, read-only `/work`, env scrubbing, drop-privilege semantics
  (`uid`/`gid`/`no-new-privs`).
- **Exit criteria:** a fork bomb, a memory hog, and an infinite loop are each
  contained and terminated with a clear diagnostic; policy denials are logged.

### Phase 10 — Portability backends (KVM + interpreter) & x86-64 guests  🟡 interpreter partially live

Second and third backends, and the second guest arch. (A usable subset of the
aarch64 interpreter already exists — it's the primary development backend; the
work remaining here is full ISA coverage, KVM, and the x86-64 guest.)

- `vcpu::kvm` — Linux; the `syscall`-trap-via-trampoline technique proven in
  univdreams' `kvm.rs` (LSTAR → `hlt;sysretq`, `KVM_EXIT_HLT` serviced by the
  same engine). arm64 KVM too.
- `vcpu::interp` — software CPU (arm64 + x86-64 decode/execute), the
  no-acceleration fallback; the syscall gate is just another trap.
- x86-64 guest ABI adapter fully populated.
- **Exit criteria:** the Phase 4 and Phase 6 test suites pass on Linux/KVM and,
  more slowly, on the interpreter; an x86-64 Alpine root runs.

### Phase 11 — Image management & developer experience

- `image` fetch: download Alpine squashfs from a mirror, verify by sha256 /
  minisign, cache under `~/.nixvm`, pin versions.
- Config file (`nixvm.toml`): mounts, env, limits, network policy, image.
- CLI polish (`cli` feature): `clap`, `--mount`, `--env`, `--net`, `--ro`,
  `--mem`, `--cpus`, `--timeout`; `tracing` logs; `nixvm pull`, `nixvm images`.
- Library API: stabilize `Sandbox`/`Config`; `stdin`/`stdout` wiring, exit codes
  and signals surfaced to the caller.
- **Exit criteria:** `nixvm run -- npm install` works from a clean machine with
  one command (auto-downloads the image); documented embeddable API.

### Phase 12 — Hardening, performance, 1.0

- Fuzz the syscall surface (guest-pointer handling, path resolution, ELF/auxv).
- Differential testing vs a real Linux kernel for covered syscalls.
- Perf: reduce VM-exit cost, batch small syscalls, fast-path `read`/`write`, mmap
  copy-avoidance; benchmark `npm install`/`cargo build` vs Docker.
- Security review of the passthrough boundary and the syscall filter.
- Docs, examples, semver-stable `0.1`/`1.0`.
- **Exit criteria:** sustained real-world workloads (a full `npm ci`, a
  `pip install` with native builds) run correctly and within a target overhead
  of a native run.

---

## 3. Cross-cutting workstreams

Run continuously alongside the phases:

- **Testing:** golden static blobs (Phase 1+), an `strace`-style trace harness
  for parity, a corpus of real Alpine binaries, per-phase integration tests
  gated on the backend feature.
- **Observability:** an env-gated syscall trace (`NIXVM_TRACE`), and the
  `Kernel::unsupported()` ledger so "what's missing to run program X" is always
  answerable.
- **CI:** build + clippy + test on macOS/arm64 and Linux/x86-64; MSRV (1.89)
  job; the interpreter backend makes syscall tests host-independent. GitHub
  Actions also builds the wasm demo (below) and deploys it to GitHub Pages.

### Browser demo (wasm)

A zero-install *try-before-you-install* page: a real Alpine shell in a browser
tab, running entirely client-side on the software interpreter — nothing touches
the visitor's machine, which is the whole value proposition demonstrated by
construction. It doubles as (a) a **host-independent correctness oracle** — the
same syscall engine as the native build, so a browser test corpus validates the
kernel on any machine with no toolchain — and (b) a **compile-time check** that
the portable path leaked no host dependencies (if it builds for wasm, the
`cfg`/feature discipline held).

- **Target:** `wasm32-unknown-unknown`, `interp` backend only (no HVF/KVM in a
  browser); in-memory / overlay / tmpfs fs backends; the Alpine squashfs
  `fetch()`ed into memory. `passthrough`, `kvm`, `hvf` and host networking are
  all `cfg`-ed out — wasm is the enforcement mechanism for that boundary.
- **Glue:** a wasm-bindgen shim for stdio (xterm.js), time, and
  `crypto.getRandomValues`; networking disabled by default (optional WebSocket
  proxy later). Cooperative single-thread scheduling by default; Web Workers +
  `SharedArrayBuffer` (needs COOP/COEP cross-origin isolation) optional for
  parallelism.
- **Delivery:** built and deployed by **GitHub Actions → GitHub Pages** on push
  to `master`; the same wasm artifact backs the in-browser test corpus in CI.
- **Depends on:** a minimal interpreter + the Phase 4 filesystem — *not* on HVF.
  See the sequencing question in §4.

---

## 4. Key risks & open questions

| Risk / question                                                                 | Approach |
| ------------------------------------------------------------------------------- | -------- |
| **HVF syscall-trap ergonomics** — cleanest way to trap `svc` at low overhead.   | Prototype early in Phase 1; measure exit cost; consider running guest at EL1 with a minimal trap vector vs EL0 + EL1 stub. |
| **Address-space model** — one flat guest AS per process; how to isolate procs.  | Per-process `GuestMemory`; COW at `fork`; HVF/KVM stage-2 or per-process VM. Decide in Phase 6. |
| **Signals on a trap-only model** — delivering async signals to guest threads.   | Interrupt the vcpu (`Exit::Interrupted`), push a signal frame, redirect PC — mirrors univdreams' `deliver_signal`. |
| **Networking fidelity** — userspace TCP/IP vs host passthrough.                  | `smoltcp` NAT by default for isolation; opt-in host passthrough under policy. |
| **Passthrough/hole escape** — a host symlink inside a shared path, or a TOCTOU swap of a component for a symlink by a concurrent thread, redirecting a lookup *outside* the mapped directory. | Race-free resolution **beneath the root**: hold the mount root as a dir fd and walk components with `openat(O_NOFOLLOW)` — `openat2(RESOLVE_BENEATH\|RESOLVE_NO_MAGICLINKS)` on Linux, a per-component `O_NOFOLLOW`+`fstatat` walk on macOS. Symlinks resolve in *our* VFS within the sandbox root, never on the host. Current passthrough is only lexically contained (`..` rejection) — this is the gap to close before writable holes of untrusted dirs are safe. |
| **Performance of the trap-per-syscall model.**                                  | Benchmark continuously from Phase 1; fast-path hot syscalls; the point of comparison is Docker/gVisor, not a bare VM. |
| **Demo-vs-native sequencing** — the interpreter sits at Phase 10, but the browser demo needs only it + the Phase 4 fs (not HVF). | If try-before-install is an adoption priority, pull a *minimal* interpreter (arm64 integer ISA) forward as its own early public milestone, decoupled from the full Phase 10 backend. Decide before Phase 4 lands. |

---

## 5. Definition of done (v1.0)

From a clean machine, one command — `nixvm run -- npm install` — downloads a
minimal Alpine image on first use, runs the install inside an isolated Linux
userland with the current directory at `/work`, enforces memory/CPU/network
limits, writes results back to the host cwd, and exits with the guest's status —
on macOS/arm64 (HVF) and Linux (KVM), with a software fallback everywhere else.
