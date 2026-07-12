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

> **Build order note.** The **interpreter path was built first**, ahead of the
> hardware backends: the interpreter makes the entire syscall engine testable on
> any machine and in CI (and is exactly what the wasm demo needs), whereas HVF
> needs a macOS entitlement + codesign to run. So Phases 1-8 and Phase 10's
> aarch64/x86-64 ISA all work end-to-end on the interpreter (~446 tests). The
> **HVF backend is now real** — a static program runs end-to-end on Apple
> Silicon (Phase 1) — with `vcpu::select` falling back to the interpreter when
> unentitled so CI stays green; `vcpu::kvm` is next and reuses the same seam.
> Every step ships with tests. Status is marked per-phase below.

### Phase 0 — Scaffold ✅ (this commit)

Workspace-free single crate; module seams (`abi`, `vcpu`, `loader`, `fs`,
`kernel`, `image`, `sandbox`); normalized `Sysno` + per-arch decode tables;
`Vcpu`/`Backend`/`MountFs` traits; `Sandbox` builder wiring the full pipeline to
its first unimplemented frontier; `nixvm` CLI (`run`/`shell`/`version`).

- **Exit criteria:** `cargo build`, `cargo test`, `cargo clippy` all clean;
  `nixvm run -- <cmd>` walks the pipeline and reports the current frontier.

### Phase 1 — HVF backend + first syscall  ✅ static program runs on hardware

Bring up Hypervisor.framework on macOS/arm64. Create a VM, map a flat
`GuestMemory`, create a vcpu at EL0, and trap `svc #0` into `Exit::Syscall`.

- **New:** `vcpu::hvf` (`hvf/{sys,vm,stub,vcpu,mod}.rs`) — hand-rolled
  `hv_vm_*`/`hv_vcpu_*` FFI (the register constants are the ARM `MRS`/`MSR`
  encodings), one process-global VM (`OnceLock`; Apple Silicon's small VM
  quota), ESR decode, register get/set. The crate's hardware-virtualization
  `unsafe`.
- **Trap model:** the guest runs **MMU-off at EL0**, so its virtual addresses
  are the IPAs of the contiguous `GuestMemory` region `hv_vm_map`'d into the VM
  (guest VA == IPA — the same flat model the interpreter uses). A guest `svc`
  traps to a process-global EL1 stub page (`hvc #0` in every vector slot) that
  `hvc`s out to the host → `Exit::Syscall`; `set_syscall_ret` emulates the
  `eret` back to EL0 from the captured `ELR_EL1`/`SPSR_EL1`. A guest access
  outside the mapped region is a stage-2 abort → `Exit::MemFault`.
- **Status: DONE for a single static program.** A guest doing
  `write(1,"hi\n",3); exit(0)` runs entirely through HVF, driven by the real
  `Kernel` run/serve loop, with syscalls dispatched off the hardware vcpu's
  registers. `vcpu::select` probes `hv_vm_create` and **falls back to the
  interpreter** when the process isn't entitled, so default `cargo test`/CI stay
  green. Running HVF needs a codesigned binary — `scripts/hvf-test.sh` builds,
  ad-hoc signs with `tests/hvf.entitlements` (`com.apple.security.hypervisor`),
  and runs the `#[ignore]`d, `NIXVM_HVF=1`-gated tests; 3 pass on Apple Silicon,
  incl. `program_write_exit_through_kernel`.
- **Deferred to follow-up:** dynamic linking through HVF; multi-process (the one
  IPA space holds a single process today — remap-on-context-switch via
  `backing_generation` is the seam); SMP vcpu thread-affinity (`hv_vcpu` is
  thread-bound, so M1 forces `ncpus=1` / the serial scheduler); and lazy/shared
  copy-on-write via stage-2 `hv_vm_protect` (fork is eager-copy for now — see
  the memory-model note in Phase 10).

### Phase 2 — Memory manager + static ELF loader  ✅ (interpreter path)

Replace the flat stub with a page-granular `GuestMemory` (region tree,
protections, host-backed pages mapped into HVF). Implement `loader::load_static`
for ELF64: map `PT_LOAD`, build the initial stack (`argc`/`argv`/`envp`/auxv),
report entry + SP. Wire `brk`/`mmap`(anon)/`munmap`/`mprotect`.

- **New:** hand-rolled ELF64 parsing (no external dep — `object` was not
  needed); `GuestMemory::{read,write,map,protect}` (flat, bounds- and
  permission-checked, 4 KiB pages).
- **Syscalls:** `brk`, `mmap`(anon), `mremap`, `madvise`, `mincore`, `munmap`,
  `mprotect`, `set_tid_address`, `set_robust_list`, `rt_sigprocmask`,
  `getrandom` (for `AT_RANDOM`).
- **Exit criteria:** a **statically-linked musl** `busybox echo`/`true` runs from
  a real ELF and exits correctly. Met (`tests/hello_elf.rs`, `tests/mm_brk.rs`,
  `tests/mm_mmap.rs`, `tests/sandbox_exec.rs`).
- **Beyond plan:** `loader::load_static` also loads **static-PIE** (`ET_DYN`
  with no `PT_INTERP`) by picking a load bias and applying its `R_*_RELATIVE`
  fixups from `PT_DYNAMIC` — musl's default static-executable output — on both
  aarch64 and x86-64.

### Phase 3 — Syscall breadth for static binaries  ✅ (broad coverage; unvalidated against real busybox)

Enough of the syscall surface to run non-trivial static programs against an
in-memory VFS. Reads/writes of guest pointers go through `GuestMemory`; file ops
go through `MountTable`.

- **Syscalls implemented:** `read`, `readv`, `write`, `writev`, `openat`,
  `close`, `lseek`, `fstat`/`newfstatat`, `getdents64`, `getcwd`/`chdir`,
  `statfs`/`fstatfs`, `readlinkat`, `symlinkat`, `mkdirat`, `unlinkat`,
  `renameat`/`renameat2`, `faccessat`/`faccessat2`/`access`, `umask`, `fcntl`
  (`F_DUPFD`/`F_GETFL` subset), `uname`, `getpid`/`gettid`/`getppid`,
  `clock_gettime`/`gettimeofday`/`clock_getres`/`nanosleep`/
  `clock_nanosleep`/`time`, `sched_getaffinity`/`sched_getparam`,
  `getrusage`/`sysinfo`/`times`/`getcpu`/`capget`/`prlimit64`/`getrlimit`,
  `prctl`. `ioctl` returns `ENOTTY` (no terminal-control modeling yet).
- **Exit criteria:** static `busybox` multi-applet (`ls`, `cat`, `sha256sum`)
  runs against a seeded in-memory fs; `strace`-level parity on the covered set.
  The syscall surface above now covers this in principle; running a real
  Alpine busybox against it (via `NIXVM_ROOT`) has not yet been recorded as a
  passing test.

### Phase 4 — Real filesystem: squashfs + overlay + passthrough  🟡 in progress

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
- **Status:**
  - `fs::TmpFs`, `fs::Overlay` (copy-up + whiteouts over any two `MountFs`
    backends), and `fs::Passthrough` are implemented and unit-tested.
    `Passthrough` write-side syscalls (`mkdirat`, `unlinkat`, `renameat2`,
    `symlinkat`, `statfs`, `getcwd`/`chdir`, `faccessat2`, `umask`) are wired.
  - `Passthrough` resolution is **symlink/TOCTOU-safe**, closing the gap this
    phase originally flagged: every lookup walks the host path one component
    at a time from a dirfd on the mount root with `O_NOFOLLOW`, so neither a
    pre-existing symlink nor one swapped in mid-race can resolve outside the
    mapped directory (see README's `unsafe` policy note, and
    `src/fs/passthrough.rs`'s tests).
  - A real squashfs/ext reader exists (`fs::fstoolfs::FsToolMount`, via the
    optional `fstool` cargo feature) but is **not yet wired into
    `Sandbox::build_mounts`** — `/` there is still a bare `tmpfs`, and the
    `run-elf`/`run-elf-x86` dev harnesses mount a real rootfs via
    `Passthrough::read_only` + `Overlay` (`NIXVM_ROOT`) rather than squashfs.
    `image::ImageStore::ensure` (download/cache) is still the Phase 11 stub,
    so the `nixvm run -- <cmd>` CLI path isn't runnable end-to-end yet.

### Phase 5 — Dynamic linking  ⬜ not started (static-PIE landed ahead of it)

Load `PT_INTERP` (`ld-musl-aarch64.so.1`) from the guest rootfs, map file-backed
segments, and provide a **vDSO** (`clock_gettime`/`gettimeofday`/`time`/
`getcpu`) plus `AT_SYSINFO_EHDR`. TLS setup (`TPIDR_EL0` / `arch_prctl` on x86).

- **Syscalls:** `mmap`(file-backed), `mremap` ✅ (added in Phase 2's follow-on
  work), `madvise` ✅, `arch_prctl`(x86) — not yet, `rseq` (stub/handle),
  `membarrier` ✅ (no-op).
- **Exit criteria:** dynamically-linked `/bin/sh` and `/bin/ls` from stock
  Alpine run to completion.
- **Status:** `loader::load_static` handles static PIEs (see Phase 2) but
  `PT_INTERP` is not read and no dynamic linker is loaded — `LoadError`
  doesn't even have a variant for it yet. `mmap` is anonymous-only (no
  file-backed mapping), so a real `ld-musl` couldn't map its own segments this
  way regardless. TLS is set only via `CLONE_SETTLS`/`Vcpu::set_tls`
  (aarch64's `TPIDR_EL0`); x86-64's `arch_prctl`(`ARCH_SET_FS`) path doesn't
  exist, so x86-64 threads have no working TLS yet.

### Phase 6 — Processes, threads, signals  🟡 mostly done; real signal-handler invocation still missing

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

The **scheduler** (`kernel::sched`) replaces the single-vcpu `Kernel::run` loop.
The model mirrors a real SMP kernel: spin up **one host thread per vcpu, sized to
the physical CPU count** — each host thread *is* a CPU. The scheduler hands a
runnable task to a free vcpu-thread, which owns and runs it until it blocks
(futex/`wait4`), yields (`Exit::Interrupted` / step-budget), or exits; then the
thread picks up the next runnable task. This is exactly how the hardware
backends must work — an HVF/KVM vcpu *is* a host thread running guest code — so
the same scheduler drives the interpreter and the hardware backends uniformly;
only the "run this task's registers until the next exit" primitive differs per
backend. Guest threads/processes migrate across vcpu-threads like tasks across
CPUs, rather than pinning one host thread per guest thread.

- **New:** scheduler (`kernel::sched`), `Task`/`Process` split, per-task cwd,
  per-thread vcpu ownership, COW fork of `GuestMemory`.
- **Syscalls:** `clone`/`clone3`, `fork`/`vfork`, `execve`/`execveat`, `wait4`,
  `exit` (thread), `futex`(WAIT/WAKE/REQUEUE/PI subset), `tgkill`/`kill`,
  `rt_sigaction`, `rt_sigprocmask`, `rt_sigreturn`, `rt_sigpending`,
  `rt_sigtimedwait`, `sigaltstack`, `getpgid`/`setpgid`/`setsid`.
- **Exit criteria:** a shell script that spawns subprocesses and pipelines runs;
  `busybox sh` job control basics; `apk` reaches network (fails cleanly until
  Phase 8).
- **Status:** the `ProcInfo`/`Process` split and the address-space table
  (`Kernel::spaces: Vec<Arc<Mutex<GuestMemory>>>`, one slot per distinct `mm`,
  shared across `CLONE_VM` threads) are implemented, exactly as planned above.
  `sys_clone` implements both `fork` (fresh `mm`, COW-by-clone of
  `GuestMemory`) and `CLONE_VM|CLONE_THREAD` threads (shared `mm`, shared
  `tgid`, distinct `pid`/tid, not reaped by `wait4`), including
  `CLONE_SETTLS`/`CLONE_PARENT_SETTID`/`CLONE_CHILD_SETTID`/
  `CLONE_CHILD_CLEARTID`. `futex` `FUTEX_WAIT`/`FUTEX_WAKE`(`_BITSET`) is a
  real park/wake (a lone waiter gets a spurious wake instead of deadlocking).
  `execve` replaces the image in place (no `execveat`, no `vfork` distinction —
  `vfork` isn't decoded separately). The scheduler exists in **two modes**
  rather than a dedicated `kernel::sched` module: `Kernel::schedule_serial`
  (cooperative single-thread round-robin, default) and `Kernel::schedule_smp`
  (`Kernel::set_ncpus`/`NIXVM_CPUS`> 1 — a pool of host worker threads run
  `vcpu.run()` in parallel while syscalls are serviced serially on the main
  thread, matching the big-kernel-lock model this section calls for). Signals:
  `rt_sigaction`/`rt_sigprocmask`/`rt_sigpending`/`kill`/`tkill`/`tgkill` are
  implemented and default dispositions (terminate/ignore) are applied after
  every syscall — but **a registered custom handler is never actually invoked**
  (no signal-frame push, no PC redirect, no `rt_sigreturn` trampoline); a
  pending signal with a real handler address is silently dropped rather than
  delivered, specifically to avoid deadlocking the scheduler. `getpgid`/
  `setpgid`/`setsid` are not implemented.

### Phase 7 — /proc, /sys, /dev, and IO multiplexing  🟡 mostly done; no real pty

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
- **Status:** `fs::ProcFs` serves a real, rendered `/proc/self/*` (`maps`,
  `exe`/`cwd` symlinks, `cmdline`, `status`, `stat`, `fd/<n>` sized to the
  actual fd table via `ProcFs::set_self`) plus static `version`/`filesystems`/
  `mounts`/`cpuinfo`/`meminfo`; `/proc/<pid>` aliases `/proc/self`. `fs::SysFs`
  serves a static `/sys` skeleton with CPU topology sized from
  `available_parallelism`. `fs::DevFs` covers `null`/`zero`/`full`/`random`/
  `urandom`/`tty`/`console`/`ptmx`/`kmsg` plus `/dev/fd`, `/dev/std{in,out,err}`
  symlinks and empty `/dev/pts`, `/dev/shm` directories — there is no real pty
  allocation yet (`ptmx` reads as EOF, doesn't hand back a pty pair). `poll`/
  `ppoll`/`select`/`pselect6`, `epoll_create1`/`ctl`/`wait`/`pwait`/`pwait2`,
  `eventfd2`, and `timerfd_create`/`settime`/`gettime` are implemented
  (readiness computed synchronously; socket fds are reported best-effort
  always-ready since `net.rs`'s connection state is private to it).
  `signalfd4`, `inotify_*`, `memfd_create`, and `close_range` are not
  implemented.

### Phase 8 — Networking  🟡 loopback done; no egress yet

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
- **Status:** `kernel::net::Net` implements AF_UNIX stream sockets and an
  AF_INET/AF_INET6 loopback (TCP stream via a connected `Pair` of byte
  buffers, UDP datagram via per-port queues), entirely in-process — `socket`,
  `socketpair`, `bind`, `listen`, `accept4`, `connect`, `sendto`/`recvfrom`
  (address-aware), `getsockname`/`getpeername`, `setsockopt`/`getsockopt`
  (`SO_REUSEADDR`/`SO_TYPE`/`SO_SNDBUF`/`SO_RCVBUF`/`SO_ERROR`), `shutdown` are
  all wired into dispatch. There is **no egress**: no `smoltcp` stack, no NAT,
  no host-socket passthrough, no DNS/`resolv.conf` — only endpoints that both
  live inside the same VM can talk to each other, so `apk`/`npm` against the
  real internet still fails cleanly (connection refused / no route), as
  expected pre-Phase-8-egress.

### Phase 9 — Resource limits & isolation policy  ⬜ not started

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
- **Status:** `prlimit64`/`getrlimit` return plausible fixed values rather than
  tracking or enforcing real limits; `Mlock*`/`Setrlimit`/scheduling setters
  are no-ops. No CPU/wall-clock deadline, pid/fd ceiling, disk quota, or
  syscall-filter policy exists yet — an infinite loop or fork bomb inside the
  guest is not currently contained by nixvm itself.

### Phase 10 — Portability backends (KVM + interpreter) & x86-64 guests  🟡 interpreters live; HVF runs static programs; KVM not started

Second and third backends, and the second guest arch.

- `vcpu::kvm` — Linux; the `syscall`-trap-via-trampoline technique proven in
  univdreams' `kvm.rs` (LSTAR → `hlt;sysretq`, `KVM_EXIT_HLT` serviced by the
  same engine). arm64 KVM too. **Not started** — no `vcpu::kvm` module exists;
  `vcpu::select` has a `// TODO(Phase 10): KVM on Linux` marker and (on Linux)
  falls back to an interpreter.

  **KVM reuses the exact backend seam HVF just proved** (see Phase 1): implement
  `trait Vcpu` + `trait Backend` in a new `vcpu::kvm` module gated
  `#[cfg(target_os = "linux")]`, and add it to `vcpu::select` with the **same
  graceful-fallback pattern** — probe (`open("/dev/kvm")` + `KVM_CREATE_VM`),
  and on failure fall through to the interpreter so non-KVM Linux / CI stays
  green. Unlike HVF, KVM needs no entitlement and **can run in CI** (a Linux
  runner with `/dev/kvm`), so its tests need not be `#[ignore]`d.

  Memory: `GuestMemory` is now **one contiguous host allocation**
  (`vcpu::region::Region`) exactly so a hardware backend can map it — use
  `host_base()` (raw host pointer) as the `userspace_addr` of a
  `KVM_SET_USER_MEMORY_REGION` (Linux host pages are 4 KiB, so there's none of
  HVF's 16 KiB-alignment constraint), and `backing_generation()` to re-issue the
  memory region when the backing changes (fork/execve) — the same
  reconcile-on-generation-change the HVF vcpu does. `Exit::MemFault` + the
  `cow_fault` seam are ready to drive EPT/stage-2 copy-on-write later. Like HVF,
  the interpreter's flat model maps cleanly onto MMU-off guest-physical == guest
  -virtual; for x86-64, prefer the univdreams long-mode + `hlt`-trampoline
  approach; for arm64 KVM, mirror the HVF EL0 + `svc`-trap setup
  (`KVM_ARM_VCPU_INIT`, exception/`HVC` exits).
- `vcpu::interp` — software CPU (arm64 + x86-64 decode/execute), the
  no-acceleration fallback; the syscall gate is just another trap. **Live on
  both guest architectures.** The aarch64 interpreter (`src/vcpu/interp.rs`,
  ~3900 lines) covers move-wide/PC-relative addressing, add/sub/logical
  (immediate, shifted, extended, with flags), bitfield move + aliases,
  conditional compare/select, bit manipulation, compares, branches/`BL`/`BLR`/
  `RET`, load/store (immediate, unscaled/pre/post-index, register-offset,
  pair, exclusive/acquire-release), ARMv8.1 LSE atomics (`CAS`/`CASP`, `SWP`,
  `LD<op>`/`ST<op>`), and a growing slice of NEON/SIMD (`DUP`/`INS`/`UMOV`/
  `SMOV`, `LD1`/`ST1`, vector ALU/compare/shift, `ADDV`/`UADDLV`, vector FP)
  plus scalar FP (`FMADD`/`FMSUB`, `FSQRT`, `FRINT*`, `FCVT*` incl. half
  precision, `FMAX(NM)`/`FMIN(NM)`, `FCMP`/`FCCMP`/`FCSEL`, `SCVTF`/`UCVTF`,
  `FMOV`). The x86-64 interpreter (`src/vcpu/interp_x86.rs`, ~3300 lines)
  covers `MOV`/`MOVZX`/`MOVSX`/`MOVSXD`/`LEA`, the ALU group with full flags,
  `MUL`/`IMUL`/`DIV`/`IDIV`, `CMOVcc`/`SETcc`, `PUSH`/`POP`/`CALL`/`JMP`/`RET`/
  `LEAVE`, `Jcc`, `INC`/`DEC`, shifts, `XCHG`, `REP`-prefixed string ops,
  `SYSCALL`, and SSE/SSE2 (xmm regs, scalar+packed FP arithmetic/compare,
  int↔float conversions, packed-integer logic/compare). Both surface anything
  unimplemented as `Exit::IllegalInstruction` rather than silently
  misbehaving.
- x86-64 guest ABI adapter: the syscall table is fully populated
  (`e1b1d6b feat(abi,bin): complete x86-64 syscall table`).
- **Exit criteria:** the Phase 4 and Phase 6 test suites pass on Linux/KVM and,
  more slowly, on the interpreter; an x86-64 Alpine root runs. Met for the
  interpreter (`tests/x86_smoke.rs` and the shared kernel test suite run on
  both `interp`/`interp_x86`); KVM is unstarted, so the Linux/KVM half is
  outstanding, and no x86-64 Alpine root has been run end-to-end yet (dynamic
  linking/TLS from Phase 5 block that).

### Phase 11 — Image management & developer experience  ⬜ not started (API shape exists, fetch is a stub)

- `image` fetch: download Alpine squashfs from a mirror, verify by sha256 /
  minisign, cache under `~/.nixvm`, pin versions.
- Config file (`nixvm.toml`): mounts, env, limits, network policy, image.
- CLI polish (`cli` feature): `clap`, `--mount`, `--env`, `--net`, `--ro`,
  `--mem`, `--cpus`, `--timeout`; `tracing` logs; `nixvm pull`, `nixvm images`.
- Library API: stabilize `Sandbox`/`Config`; `stdin`/`stdout` wiring, exit codes
  and signals surfaced to the caller.
- **Exit criteria:** `nixvm run -- npm install` works from a clean machine with
  one command (auto-downloads the image); documented embeddable API.
- **Status:** `image::ImageRef`/`ImageStore` exist (naming convention, cache
  location via `NIXVM_CACHE`/`~/.nixvm`) but `ImageStore::ensure` only checks
  whether the file is already present locally — no download, no digest
  verification. There is no `nixvm.toml`. The `nixvm` binary is a small
  std-only arg handler (`run [--mem] [--workdir] -- <cmd>`, `shell`,
  `version`) — no `clap`, no `--net`/`--ro`/`--cpus`/`--timeout`, no `tracing`.
  `Sandbox`/`SandboxBuilder` (`command`, `work_dir`, `mem_bytes`,
  `prefer_interp`, `bind`/`bind_ro`) and `Sandbox::exec_elf` are the stable,
  working embeddable surface today; `Sandbox::run()` (the image-based path) is
  blocked on the fetch stub above.

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
- **CI:** the interpreter backend makes syscall tests host-independent, so
  `cargo test` (253 unit + 8 integration tests + 1 doctest) needs no
  hypervisor and runs anywhere. The only GitHub Actions workflow today
  (`.github/workflows/pages.yml`) builds and deploys the wasm demo on push to
  `main`; a build+clippy+test matrix across macOS/arm64 and Linux/x86-64, and
  an MSRV (1.89) job, have not been set up yet.

### Browser demo (wasm)  ✅ shipped (in a smaller shape than first planned)

A zero-install *try-before-you-install* page running entirely client-side on
the software interpreter — nothing touches the visitor's machine. It doubles
as (a) a **host-independent correctness oracle** — the same syscall engine as
the native build — and (b) a **compile-time check** that the portable path
leaked no host dependencies (if it builds for wasm, the `cfg`/feature
discipline held).

- **Target:** `wasm32-unknown-unknown`, `interp` backend only (no HVF/KVM in a
  browser); `TmpFs`/`DevFs`/`ProcFs`/`SysFs` only — no `Passthrough` (`cfg`-ed
  out on wasm32) and no squashfs-backed Alpine root yet (the demo takes a
  user-picked static ELF, not a full rootfs).
- **What it actually is today:** `src/wasm.rs` exposes one `#[wasm_bindgen]`
  function, `run_elf(bytes: &[u8]) -> String`, that loads a static ELF the
  visitor picks, runs it to completion on the aarch64 interpreter, and returns
  its captured stdout/stderr/exit-code as JSON; `web/index.html` is a single
  static page (file picker + `<pre>` output, no xterm.js, no interactive
  shell) that calls it. Not yet the "real Alpine shell in a browser tab"
  originally envisioned — that needs the squashfs-into-wasm and a real pty,
  neither of which exist yet.
- **Delivery:** built (`wasm-pack build --target web --no-default-features
  --features wasm -- --lib`) and deployed by **GitHub Actions → GitHub
  Pages** (`.github/workflows/pages.yml`) on every push to `main` that touches
  `src/`, `web/`, or the manifest.
- **Depends on:** the interpreter + `TmpFs`/`DevFs`/`ProcFs`/`SysFs` — *not* on
  HVF. The sequencing question in §4 is resolved: the demo shipped ahead of
  the full Phase 10 backend, as a static-ELF-runner rather than a full shell.

---

## 4. Key risks & open questions

| Risk / question                                                                 | Approach |
| ------------------------------------------------------------------------------- | -------- |
| **HVF syscall-trap ergonomics** — cleanest way to trap `svc` at low overhead.   | ✅ **Resolved (Phase 1).** Guest at **EL0 + a minimal EL1 stub** (`hvc #0` in every vector slot) — `svc` traps to the stub, which `hvc`s out to the host; `set_syscall_ret` emulates the `eret`. Chosen because an EL0 `svc` traps to EL1 (`VBAR_EL1`), never straight to the host. Exit-cost measurement is a later optimization. |
| **Address-space model** — one flat guest AS per process; how to isolate procs.  | ⏳ **Partially resolved.** `GuestMemory` is now **one contiguous host allocation per process** (`vcpu::region`), mapped into a **single shared IPA space** on the hardware backend; a hardware vcpu re-maps that window when `backing_generation()` changes (the seam for remap-on-context-switch across processes). Single-process works today; multi-process IPA multiplexing and stage-2 COW (`hv_vm_protect`/EPT driving the `cow_fault` seam) are the follow-up (M2). Interpreter isolates per-process by holding a distinct `GuestMemory`. |
| **Signals on a trap-only model** — delivering async signals to guest threads.   | Interrupt the vcpu (`Exit::Interrupted`), push a signal frame, redirect PC — mirrors univdreams' `deliver_signal`. |
| **Networking fidelity** — userspace TCP/IP vs host passthrough.                  | `smoltcp` NAT by default for isolation; opt-in host passthrough under policy. |
| **Passthrough/hole escape** — a host symlink inside a shared path, or a TOCTOU swap of a component for a symlink by a concurrent thread, redirecting a lookup *outside* the mapped directory. | ✅ **Resolved.** `fs::passthrough` resolves every lookup component-by-component from a dirfd on the mount root with `O_NOFOLLOW`; a symlink's target is read and re-spliced into the walk (re-anchored so absolute targets and `..` chains can't climb above the root); the final syscall is always issued directly against `(parent_dirfd, name)` so a last-instant swap fails safely instead of redirecting. See README's `unsafe` policy note and `src/fs/passthrough.rs`'s tests. |
| **Performance of the trap-per-syscall model.**                                  | Benchmark continuously from Phase 1; fast-path hot syscalls; the point of comparison is Docker/gVisor, not a bare VM. Not yet benchmarked. |
| **Demo-vs-native sequencing** — the interpreter sits at Phase 10, but the browser demo needs only it + a minimal fs (not HVF). | ✅ **Resolved as planned.** The interpreter and `TmpFs`/`DevFs`/`ProcFs`/`SysFs` were pulled forward as an early, standalone milestone (`src/wasm.rs` + `web/` + CI Pages deploy), decoupled from HVF/KVM and from the full Phase 4 squashfs pipeline — see the Browser demo section above. |

---

## 5. Definition of done (v1.0)

From a clean machine, one command — `nixvm run -- npm install` — downloads a
minimal Alpine image on first use, runs the install inside an isolated Linux
userland with the current directory at `/work`, enforces memory/CPU/network
limits, writes results back to the host cwd, and exits with the guest's status —
on macOS/arm64 (HVF) and Linux (KVM), with a software fallback everywhere else.
