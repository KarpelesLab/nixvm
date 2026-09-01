# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.1](https://github.com/KarpelesLab/nixvm/compare/v0.0.0...v0.0.1) - 2026-09-01

### Fixed

- *(wasm)* remove `*/` from Terminal doc — it broke the generated pkg/nixvm.js

### Other

- fair virtual-runtime scheduling (nice-proportional, no SMP starvation)
- don't call std::time on wasm32 (fixes x86 demo "boot failed: unreachable")
- a forked child inherits the parent's process group (fixes wait4(0))
- implement aarch64 rt_sigframe delivery (was x86-64-only → crash)
- MAP_SHARED|MAP_ANONYMOUS is shared across fork (real shared memory)
- Merge branch 'worktree-agent-ad150a3681a5fa607'
- capture & implement the full clone(2)/clone3(2) flag set
- Merge branch 'worktree-agent-a6e2719ef42f12687'
- fsync/fdatasync EBADF on a bad fd; dup3 equal-fd EINVAL ([#8](https://github.com/KarpelesLab/nixvm/pull/8))
- fallocate FALLOC_FL_PUNCH_HOLE zeroes the range ([#6](https://github.com/KarpelesLab/nixvm/pull/6))
- make metadata syscalls actually take effect + path-resolution errnos
- reset caught signal handlers to SIG_DFL
- honor VMIN/VTIME on non-canonical reads (fixes read hang)
- honor SA_NODEFER (fixes deadlock) and SA_RESETHAND
- seekable directory fds (rewinddir/telldir/seekdir)
- idle in-VM sockets are not readable (fixes event-loop busy-spin)
- declare MIT consistently (matches the LICENSE file)
- describe pipe EOF/broken readiness by observed behavior, not internals
- reject reads from an O_WRONLY file with EBADF
- reject writes to an O_RDONLY file with EBADF
- enforce O_EXCL, O_DIRECTORY, O_NOFOLLOW, and EISDIR
- real hard links on host-backed mounts (was always a copy)
- sigwaitinfo/sigtimedwait report the queued si_code and si_value too
- carry si_code/si_value through sigqueue to the handler
- honor IN_NONBLOCK so a watcher read returns EAGAIN, not a deadlock
- readlink on the mount root is EINVAL, not EISDIR — fixes realpath
- broken-pipe write end reports POLLOUT|POLLERR, not POLLERR alone
- encode a signal death as WIFSIGNALED, not WIFEXITED
- a pipe at EOF reports POLLHUP alone, not POLLHUP|POLLIN
- honor O_NONBLOCK on reads — empty non-blocking pipe returns EAGAIN
- F_GETLK reports the region unlocked; F_SETLK/OFD locks granted
- /proc/self/{comm,cmdline,stat,status,fd} report the live program
- prctl NO_NEW_PRIVS / PDEATHSIG / DUMPABLE round-trip; getpriority ABI
- raise SIGPIPE on writes to a reader-less pipe/socket
- honor EPOLLONESHOT (disarm after one event)
- map rt_sigqueueinfo/rt_tgsigqueueinfo (sigqueue) to delivery
- support AF_UNIX SOCK_DGRAM (syslog /dev/log, unix-dgram IPC)
- renameat2 flags, F_GETPIPE_SZ, O_TMPFILE fallback
- report non-zero uptime and free RAM
- honor O_APPEND — writes go to end-of-file
- implement chmod/fchmod (were no-ops) so mode changes stick
- access(X_OK) checks execute bits; implement SEEK_DATA/SEEK_HOLE
- page-round st_blocks (du/ls -s), matching tmpfs
- make /proc/self/{exe,cwd,root} resolve to live per-task state
- report nixvm's own core count, not the host's (sysfs leak)
- track FD_CLOEXEC (close on exec) and honor F_DUPFD's minimum
- deliver signals through the fd (was a never-readable stub)
- implement utimensat (set file mtime) instead of a no-op
- getitimer/setitimer honor `which` (VIRTUAL/PROF read back disarmed)
- implement alarm/setitimer/getitimer (ITIMER_REAL → SIGALRM)
- implement rt_sigtimedwait (sigwait/sigtimedwait/sigwaitinfo)
- honor the ppoll/pselect6/epoll_pwait signal mask
- POSIX process-group targeting (0 / -1 / -pgrp) + negative-pid fix
- fill rusage, honor the pid filter, accumulate children CPU
- real nanosleep + per-task CPU accounting (clock_gettime/getrusage/times)
- report real CPU time, consistent with clock_gettime
- full compliant clock set, correct per-clock semantics
- let the software backend reach the vDSO (any libc, not just KVM)
- interrupt every blocking syscall correctly (SA_RESTART vs EINTR)
- wire ISIG signal generation (^C/^\/^Z → fg process group)
- pseudo-terminal support (/dev/ptmx + /dev/pts/N)
- virtual tty — forward terminal ioctls to the host tty
- general socket/network ioctls (SIOCGIF*, FIONREAD, SIOCOUTQ)
- implement the fd-flag ioctls (FIONBIO/FIOCLEX/FIONCLEX)
- edge-trigger EPOLLET so a signalled eventfd stops the loop spinning
- clock vDSO — serve clock_gettime/gettimeofday without a syscall
- service clock reads lock-free (no big `sh` lock)
- implement fcntl(F_SETFL) O_NONBLOCK — claude completes its request
- Revert "poll: honor EPOLLET (edge-triggered) in epoll_wait"
- poll for host I/O instead of declaring deadlock while a connection is live
- honor EPOLLET (edge-triggered) in epoll_wait
- install host egress in the `nixvm run` path (NIXVM_NET=host)
- fix two tests double-freeing map's data frames
- run signal handlers at CPL3 — fixes SMP fork/signal memory corruption
- don't double-apply the syscall result — fixes the interpreter
- serialize runs of a shared (CLONE_VM) address space — fixes SMP claude
- decouple guest virtual space from the physical RAM pool + lazy pool — claude runs
- async signal delivery to handlers + real rt_sigsuspend + SIGCHLD-on-exit fixes the `& wait` livelock
- B5 — peel `eventfds`/`timerfds`/`epolls` onto `Kernel.pollfds`
- B4 — peel `pipes` onto its own `Kernel.pipes` lock
- B3 — peel `net` onto its own `Kernel.net` lock
- B2 — peel `mounts` onto its own `Kernel.vfs` lock
- B1 — extract Shared behind one coarse Mutex; servicing takes &self
- refresh module doc for the ServiceCtx extraction (Phase A)
- extract per-task servicing state into a passed-in ServiceCtx (Phase A)
- per-address-space kernel stack — fixes SMP fork/exec corruption
- TLB shootdowns for host-side page-table edits (fork/CoW/munmap/mprotect)
- demand paging + faithful triple-fault decode; runs the full JSC engine
- retire the flat identity model — shared frame pool + per-process page tables
- real x86-64 AddrSpace over the Phase-1 frame pool
- physical RAM pool + 4 KiB frame allocator (MMU refactor, Phase 1)
- Send bounds on the fstool-backed mount so the feature build compiles
- service syscalls in place under a big kernel lock — kill the per-syscall hand-off
- run KVM_RUN lockless for true intra-process parallelism
- real time-based preemption for both vcpu backends
- resolve /proc/self/fd/<n> against the live fd table
- MADV_DONTNEED preserves file-backed (ELF segment) pages
- gated load-verify + open tracing debug aids
- preemptive time-slicing (NIXVM_SLICE) so a busy thread can't starve others
- guest-debug watchpoints + execute breakpoints (NIXVM_WATCHPOINT/NIXVM_EBP)
- NIXVM_NOWX debug gate to force uniform RWX
- accurate #PF handling, CPL3 restore, AVX state, grow-down stack
- enforce per-page W^X — guest page tables from the protection map
- enforce NX on instruction fetch (no more executing writable pages)
- synchronous signal delivery (SIGSEGV/SIGILL/SIGBUS to guest handlers)
- real mmap allocator, sched_yield, 8 MiB stack, 64 GiB KVM ceiling
- dependency-free soft-float — true 80-bit x87 + MXCSR directed rounding
- implement the x87 F-row transcendentals (FPREM/FSIN/FYL2X/…)
- share the anonymous-mmap arena per address space, not per task
- implement the SSE ops V8's JIT emits — TurboFan runs
- node runs on the software interpreter (POP r/m + IMUL/shift flags)
- add PALIGNR and PTEST (0F 3A 0F / 0F 38 17)
- syscall writes RCX←RIP and R11←RFLAGS, like hardware
- fix imm16 decode length + add SSE/stack ops V8 exercises
- share fd tables across threads + futex requeue + timer wakeups — node runs
- record the node event-loop limitation (futex/epoll busy-spin)
- bound the fd-cloexec loop — fcntl EBADF on closed fds + capped RLIMIT_NOFILE
- Phase 5 dynamic linking is done — record the two large-program fixes
- give upper and lower layers disjoint inode numbers
- flush writable MAP_SHARED file mappings back to their file
- record the broadened syscall coverage
- broad syscall coverage — ~60 new syscalls across both arches
- record host-egress (Phase 8) landing — apk works over the internet
- host egress — apk update && apk add work over the real internet
- surface a wasm panic instead of hanging at the prompt
- flock + sendmsg/recvmsg — apk update runs to a clean network failure
- don't panic on wasm32 — the browser terminal died on `ls`
- pick the guest architecture (arm64 / x86-64) before booting the VM
- x86-64 Alpine boots on the interpreter — xchg/AND/OR decode fixes
- boot a full Alpine userland — per-arch stat, legacy syscalls, KVM retry fix
- hardware backend for x86-64 guests — static glibc runs end-to-end
- run stock static glibc binaries — arch_prctl TLS, brk fix, decode gaps
- sync README/ROADMAP/Cargo for the HVF milestone + KVM readiness
- default-on select() with interp fallback + static program e2e
- implement HvfVcpu — EL0 execution, svc trap, syscall resume
- Hypervisor.framework FFI + process VM + bring-up test
- unify guest memory on one contiguous host-aligned region
- page-granular copy-on-write with fault-driven privatization
- export PWD=/ to match the shell's starting cwd
- fix vfork clobbering the parent's address space
