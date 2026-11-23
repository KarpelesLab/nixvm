# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.10](https://github.com/KarpelesLab/nixvm/compare/v0.0.9...v0.0.10) - 2026-11-23

### Other

- an access the page tables allow by now is run again
- shared file mappings belong to the address space, not the task
- a reaped child's CPU time is its whole process's
- a MAP_FIXED mapping over backed pages flushes the TLB
- the address space starts at 0, so the low 64 KiB can be mapped
- vm.mmap_min_addr, and below it CAP_SYS_RAWIO, as cap_mmap_addr has it
- SO_PASSRIGHTS, an AF_UNIX end that refuses SCM_RIGHTS (Linux 6.16)
- cpu_timers, CPU-time timers and itimers against Linux
- CPU-clock timers expire on CPU time, as posix-cpu-timers.c has them
- a call interrupted under a tracer that detaches still restarts
- ptrace_regs watches where the KVM guest keeps its supervisor pages
- a debugger's breakpoint on the guest's supervisor pages stays off
- ptrace_regs checks restarts, kernel accesses to watched memory, per-thread watchpoints
- an interrupted traced call returns -ERESTART*, restarted from where rip is then
- a new task reads its parent's DR6 and DR7, and what Linux leaves out
- a traced task's group stop is a stop for its tracer

## [0.0.9](https://github.com/KarpelesLab/nixvm/compare/v0.0.8...v0.0.9) - 2026-11-19

### Other

- kvm tests: clippy's needless_continue in the preemption retry
- kvm tests: a run the preemption timer breaks before the guest's syscall is run again
- ptrace_regs32's XMM check asks for SSE2, which Ubuntu's i386 gcc leaves off
- the SCM_PIDFD emission's flags, Dgram's Debug, place_32bit
- unix_scm checks SCM_PIDFD and SO_PEERPIDFD; ia32 an i386 SCM_PIDFD
- SO_PASSPIDFD and SCM_PIDFD, and SO_PEERPIDFD, on AF_UNIX sockets
- clocks finds the aarch64 vDSO's clock functions by their names
- mmap_place, MAP_32BIT, mmap hints and the fixed mappings' checks
- MAP_32BIT, and 64-bit mmap hints taken, as x86's get_unmapped_area
- clocks, clock ids and what each call refuses; ia32's vDSO with them
- the clock id is a 32-bit clockid_t, compared unsigned
- its five clocks, an absolute expiry on its own clock, and a read that wakes
- clock ids checked as Linux's clockid_to_kclock, and the CPU clocks by pid
- rustfmt and clippy: the debug registers and PTRACE_LISTEN
- ptrace_regs checks an execve's exit stop and the SIGTRAP after it
- the stops at an execve's end show the call, and its SIGTRAP comes next
- ptrace_regs and ptrace_regs32, si_errno, SIGCONT, syscall stops and debug registers
- the debug registers, in struct user's u_debugreg
- RF is a flag, and a read/write watchpoint fires on reads
- a continued task takes its signals before it runs more of its code
- syscall stops have the FP state, and a tracer's rip there is where the call returns
- chdir of an empty path is ENOENT
- a queued signal carries si_errno, and seccomp's SIGSYS the filter's data there
- an i386 recvmsg closes the descriptors its control buffer has no room for
- ptrace32 and ptrace_ia32, ptrace and seccomp of i386 calls
- from and of i386 tasks, as Linux's compat paths
- a call through the i386 entry is AUDIT_ARCH_I386, its i386 number and registers
- tkill and tgkill send SI_TKILL, and refuse a tid that isn't positive
- a siginfo's layout by Linux's per-signal code limits
- check the fixed layout at compile time; the README's compat32_insn
- compat32_insn runs 16-bit code from a modify_ldt code segment
- 16-bit code and stack segments in compatibility mode
- ia32 and compat32 check that 32-bit mappings stay below 4 GiB
- 32-bit callers' mappings stay below 4 GiB, and their hints are taken
- ia32 checks the 32-bit vDSO's clocks against the system calls
- the clock functions, reading the 64-bit vDSO's vvar page
- gettimeofday fills in the timezone, as the vDSOs do
- ia32 passes its SCM_RIGHTS over SOCK_DGRAM again
- unix_scm, ancillary data on AF_UNIX datagram and seqpacket sockets
- AF_UNIX datagrams carry their own SCM_RIGHTS and SCM_CREDENTIALS
- a release gives back at most what its file was charged
- count disk by where the host says the data is, not st_blocks
- files are sparse, so a hole holds no memory
- the disk quota counts allocated blocks, so a sparse file costs only what's written
- the /usr/include/asm link gcc-multilib would make, for the ia32 build
- show why an i386 program didn't build
- a 32-bit libc on the KVM runner so ia32 runs there; rustfmt
- %ebp holds a fast entry's sixth argument; PER_LINUX32's uname
- clippy and rustdoc: the compat layer's lints, and the GDT pool's KVM-only gate
- the fast entry the CPU vendor has, and an i386 mmap2 from 64-bit code
- modify_ldt and LDT segments, and 600 threads with TLS entries
- LDT selectors, and a private GDT per host thread instead of per thread
- modify_ldt, the process's LDT
- ia32, an i386 program run as an ELFCLASS32 process
- the 32-bit vDSO enters with sysenter (the CPUID vendor is Intel)
- compatibility-mode linear addresses wrap at 4 GiB; keep the 64-bit path lean
- compat32_insn, the instructions only 32-bit code has
- BCD adjustments, BOUND, ARPL, SALC and far-pointer loads
- sysenter enters as the 32-bit vDSO's int $0x80
- segment bases and limits, LDT selectors and 16-bit addressing in compatibility mode
- the whole i386 system-call table, with compat translation
- syscall and sysenter from 32-bit code, as Linux on the CPU's vendor
- ELFCLASS32 (i386) programs, with a 32-bit stack, auxv and vDSO
- the 32-bit vDSO image, and per-process LDT and vDSO32 slots in GuestMemory
- a thread killed by a sibling's exit_group never runs again on SMP
- the private-GDT pool is KVM-only, so wasm and macOS builds don't warn
- compat32, 32-bit code and int $0x80 in a 64-bit process
- the i386 int $0x80 entry, set_thread_area, and signals in 32-bit code
- compatibility mode — 32-bit code, far transfers, segment loads, int $0x80
- Linux's GDT, 32-bit code in compatibility mode, and the int $0x80 gate
- a call that finds another task's memory busy runs again soon
- a wake landing while a task is in flight isn't lost
- plain code spans for items behind the images feature, so docs build without it
- iretq and retfq back to the 64-bit user code segment
- NETLINK_KOBJECT_UEVENT sockets, which never see an event
- TF set by popf single-steps the program
- a handler starts with TF, DF and RF clear
- signal frames carry the FPU state (the XSAVE area), as on Linux
- a trailing slash follows the last symlink, even for lstat
- select, pselect6: honor the timeout, and write back the time left
- the address space ends at 1 TiB exactly, a power of two as on Linux
- ARCH_SET_GS/ARCH_GET_GS set and read a real GS base
- a static PIE is a loader, mapped halfway up the address space, not at its base
- root images pulled from OCI registries by digest, over rsurl; no images of our own
- quote a test filter YAML read as a mapping ("image:: fs::overlay")
- overlay, tests: the persistent upper on the libc-free Linux; root image tests install their environment
- released root images, fetched and verified in pure Rust, mounted without unpacking
- an upper layer that outlives the guest — a host directory, whiteouts as .wh. files
- fs, sandbox: protect and a writable root directory on the libc-free Linux too
- build and test the libc-free path
- passthrough, KVM, egress and the clock on a libc-free Linux
- raw Linux system calls for a host with or without libc
- Sandbox::execute, an embedding API to run one command ([#14](https://github.com/KarpelesLab/nixvm/pull/14))
- read no environment, print nothing — NIXVM_* via nixvm::env, kernel notices to a sink
- protect(paths) keeps paths read-only by identity ([#15](https://github.com/KarpelesLab/nixvm/pull/15))
- a symlink mid-path is followed, not ENOTDIR
- pin wasm-pack — "latest" resolved to v0.9.1 and broke the Pages build
- a thread killed by a sibling's execve never runs or is serviced in the new image

## [0.0.8](https://github.com/KarpelesLab/nixvm/compare/v0.0.7...v0.0.8) - 2026-11-16

### Other

- a clear to an exact half stores the same value on every host
- out of guest RAM, a fault is an out-of-memory kill, not SIGSEGV at 0
- execbuffers return fences — Chromium's frames reach the compositor finished
- a real-time child takes its place in the queue at fork
- 1 GiB of guest RAM — llvmpipe ran 256 MiB dry (es2gears segfault)
- ROADMAP — desktop GL 3.3 on GLES 3.0/WebGL2, what's emulated, the state cache
- compositing on the GPU, the browser's display canvas, Chromium's render passes
- gpu_display through the GL compositor — interpreter, shared memory
- indexed draws start where the index buffer's offset points
- with the GPU on, the display canvas is the GPU's WebGL2 canvas
- display_present — show the frame on the GPU's WebGL2 canvas
- a GL state cache, and cheap hashing on the renderer's per-draw maps
- gpu_gl33 — desktop GL 3.3 core against llvmpipe, natively and in the browser
- desktop OpenGL 3.3 (and WebGL 2 in Chromium) on a GLES 3.0 / WebGL2 host
- a Compositor interface; with a GPU, frames are made on the GPU
- GpuDevice::with_gl — the host GL context for callers outside the guest
- README — GL apps on the display with the GPU
- PRIME, the display's dmabuf support and what runs on it
- the display's read-back reads what GLES/WebGL2 always can
- the WebGL2 backend's first browser runs — results, fixes, what's left
- a gpu toggle — the guest's Mesa renders with WebGL2 in the page
- what WebGL2 rejects — fixed-index restart, read formats, float targets
- gl_FragCoord is TGSI's fragment position as it is
- zwp_linux_dmabuf_v1 — show what clients render on the GPU
- winding on window-system buffers (flipped viewport)
- PRIME — GPU buffers as dma-buf descriptors

## [0.0.7](https://github.com/KarpelesLab/nixvm/compare/v0.0.6...v0.0.7) - 2026-11-14

### Other

- bound what a guest can make the host allocate
- the GPU acceleration design and status in the roadmap
- Gallium's blend factors, winding and mip filters, checked against llvmpipe
- suggest havoc alone; mention the display on the page
- boot_squashfs_env; a live test runs a Wayland terminal on the display
- the guest's display on a canvas, with keyboard and mouse
- forkpty's child owns its terminal; termios2 for newer glibc
- a Terminal can have a display
- boot with extra environment, attach the built-in display
- a virtio-gpu render node for the guest's Mesa, rendered with GLES 3
- bound what a client can make the host hold
- the built-in Wayland display, in the README and the CLI's usage
- frame callbacks before new input; bad opcodes say so
- a Wayland compositor nixvm runs for the guest
- the master stays open while any fd on it does
- a socket nixvm serves in the guest, and the display skeleton

## [0.0.6](https://github.com/KarpelesLab/nixvm/compare/v0.0.5...v0.0.6) - 2026-11-13

### Other

- render another task's maps only when its maps is read
- socket events wake their waiters; epoll reports fds oldest-ready first
- --user works on any root, not only an image's
- /proc/<pid>/maps walks runs, a chunk of pages at a time
- a read of a /proc file renders the address space only for maps
- a regular file's read is all of it; connect again is 0, then EISCONN
- what Chromium's sandbox needs: SCM_CREDENTIALS, CLONE_FS chroot, /proc/self/fdinfo, tmpfs mount options
- any thread waits for the process's children; mapping a file is one read
- a PID namespace's init drops default-action signals when they're delivered, not only when sent
- filemap's links resolve in the public docs
- gdb runs, breaks, steps, catches faults and attaches
- single steps, fault stops, and the FP registers
- the module doc says what RET_TRACE does now
- file mappings map the file's pages; shared memory lives in guest memory
- the creator owns what it makes, less the umask; a bound socket has a name
- CPU exceptions in user code get IDT gates of their own
- file capabilities, newuidmap, /proc/self/task; userns_caps runs on Ubuntu's CI
- a task blocked in a syscall shows as asleep
- seccomp sees the syscall's return address; /proc/<pid>/syscall
- kill() no longer finds threads that have exited
- processes sharing an fd table without sharing memory
- SOCK_SEQPACKET socketpairs
- seccomp user notification; /proc/<pid>/mem; the rest of the seccomp API
- a blocking recvmsg leaves the msghdr alone until it has data
- ptrace between guest tasks, enough for strace -f
- a guest ptrace program (exec trap, syscall stops, peek/poke, signals, fork events, seize, exitkill, seccomp RET_TRACE)
- capabilities are privilege, not uid 0; setns checks every namespace; bubblewrap runs
- every arrival is a fresh epoll edge; stalls re-check host I/O
- execve takes the whole process; 1 TiB address spaces; a versioned vDSO; MADV_DONTNEED frees
- threads share the program break; host connects don't freeze the VM; dual-stack sockets
- index-free register setup in two tests, for clippy 1.100
- fstool 0.5.4, so Ubuntu images run
- guest randomness from purecrypto's OsRng, the browser's crypto in wasm
- pull images in the browser over the guest's network link, not fetch()
- don't reap a child still finishing its exit on another CPU
- cgroups waits for the small task when Linux counts a second kill
- rlimits says what waitpid returned when it isn't the child
- run Docker/OCI images without a daemon
- ASIDs per address space, lazy register reads; no flush for unchanged leaves
- fault-around for sequential first touches
- named FIFOs
- a rebuild leaves the macOS binaries unsigned; sign again
- fail before the point of no return; follow the linker's last link
- a signal to a task running on another vcpu stops it at once (HVF)
- a memory.high penalty owed from user mode is paid before the next syscall
- a failing conformance program shows what it printed, not the runner's lines

## [0.0.5](https://github.com/KarpelesLab/nixvm/compare/v0.0.4...v0.0.5) - 2026-11-12

### Other

- an aarch64 vDSO, calibrated from the host timebase on HVF
- build the backend on every macOS/arm64 target; sign the macOS release
- no link to a private item from PtFormat
- run on the shared pool with arm64 page tables; preempt and signal spinning guests
- disks, the page cache, reclaim with memory.min/low, the io controller
- real-time tasks run ahead of fair ones, FIFO/RR in turn, with the fair server
- cpu.weight is shared by the group; sched_setaffinity is honored
- threaded subtrees
- Linux may also kill the small task when memory.max is lowered
- say what lowering memory.max killed
- lowering memory.max kills until the cgroup fits; memory.high throttles
- no link to a private item
- the memory controller (memory.max, out-of-memory kills)
- the cpu controller (cpu.max bandwidth, cpu.weight)
- a nonzero SO_LINGER makes close wait for the host peer's ack

## [0.0.4](https://github.com/KarpelesLab/nixvm/compare/v0.0.3...v0.0.4) - 2026-11-07

### Other

- enforce RLIMIT_RTTIME; a fork child's CPU time starts at zero
- no rename, as in cgroup v2 (EPERM)
- signal info posted to a running task isn't lost
- cgroup.stat in Linux's layout, with the pids controller's counts
- Linux's no-internal-process rule for a threaded controller
- a real cgroup v2 hierarchy, with the pids controller
- UDP multicast between the VM's sockets
- pid namespaces: a recorded pid of a task that has gone keeps its number
- an ignored signal is discarded as it's sent; a self-stop stops
- a reaped task's vcpu no longer replaces a new task's
- the embedder's syscall policy, with logged denials
- mount namespaces: propagation (shared, slave, private, unbindable)
- the conformance programs run on an aarch64 guest too
- mount namespaces: open files survive setns into another one
- net namespaces: lo starts down; empty buffers never fault
- signals and ids: si_pid, SIGCHLD info, signalfd, SO_PEERCRED, PID translation
- keepalive and abortive linger reach host connections
- enforce MEMLOCK, NICE, RTPRIO, SIGPENDING and MSGQUEUE
- real bubblewrap runs in the VM; /proc/<pid> and tests in CI
- user and time namespaces
- mount isolation, real chroot, pivot_root and bind mounts
- PID isolation
- network isolation
- real UTS and IPC isolation, /proc/<pid>/ns and setns
- set them through the Sandbox and Vm APIs; fix a macOS lint
- deliver signals to code that makes no syscalls on KVM
- disk protection and VM-wide time limits
- real per-process rlimits
- bound the kernel's other queues; fit guest RAM to the host
- bounded pipes and socket queues; enforce descriptor limits
- survive any number of forks; a full pool no longer crashes
- a VM-wide kernel memory quota; bound guest-sized allocations
- make everyday socket options work; refuse the dangerous ones
- threads share one live handler table (CLONE_SIGHAND)
- fix the invalid alpine job (runner context in job-level env)
- boot a real Alpine image on KVM and on the interpreter
- pump_for test grows its workload until it outlasts the budget
- bump GitHub Actions to current majors; build Pages on Node 24
- bump vue, vite and @vitejs/plugin-vue, and refresh the lockfile
- fstool 0.5, compcol 0.7, ureq 3; raise MSRV to 1.95
- refresh Cargo.lock to the latest semver-compatible versions
- resolve PID 1 symlinks inside the guest root; pick the guest arch from the ELF
- XSAVE signal frames with the full extended state; Linux fault signals
- on hosts without FEAT_AFP, accept Rosetta's NaN sign and RCP estimates
- Merge branch 'worktree-agent-a3b22575688121458'
- x86 interpreter: RORX requires VEX.vvvv = 1111; coverage scan checks invalid encodings too
- deliver signals at preemption points; time-slice the serial scheduler
- /dev/tty is the controlling terminal, with working termios
- cpuinfo Features derived from the advertised HWCAPs; real /proc/self/auxv
- Linux-accurate arm64/x86-64 frames with FP/SIMD state; SIGTRAP/SIGBUS fault kinds
- x86_diff covers the YMM state, VEX (AVX/AVX2/FMA/F16C/BMI) and XSAVE
- x86 interpreter: x86-64-v3 — AVX, AVX2, FMA, F16C, BMI1/2, LZCNT, MOVBE, XSAVE
- tolerate the M1's FJCVTZS Z flag for FZ-flushed subnormals
- x86 interpreter: factor SSE lane math into pure helpers; exact f64 fast path near underflow
- don't link private items from public docs (rustdoc -D warnings)
- curated default word list for the aarch64 diff test; skip host-missing features
- a 32 GiB virtual window over the physical pool, so Go runs
- *(aarch64)* regression tests for instructions real programs hit
- x86 interpreter: unit-test 0x67 addressing, 32-bit string registers, fs: RIP-relative
- x86 interpreter: group-7 UMIP spoofing, prefix-tolerant group 15/MOVNTI/EMMS
- x86_diff covers branches (Jcc/LOOPcc/JrCXZ/JMP) via skippable filler
- a fork child's close keeps the parent's registrations
- x86 interpreter: cache the executing code page within a run
- Linux argument checks, keep protection, move shared frames, MREMAP_DONTUNMAP
- *(aarch64)* FP16 arithmetic; advertise the verified extensions
- x86 interpreter: x86-64-v2 (SSE3, SSSE3, SSE4.1, SSE4.2) and an e2e program test
- *(aarch64)* cache decoded code pages for instruction fetch
- real hard links in tmpfs and the overlay; reflink ioctls answer EOPNOTSUPP
- x86-64 differential tester against real execution (Rosetta 2)
- x86 interpreter: complete baseline ISA, true 80-bit x87, MMX, SSE/SSE2
- *(aarch64)* implement post-v8.0 extensions, and LSE2/SP alignment faults
- never let a guest-supplied length size a host allocation
- real mount/umount of pseudo filesystems, supplementary groups, syslog, arch_prctl
- argument validation and subcommand fixes across mm, signals, clone, ioctl
- the full socket-option namespace, Linux's defaults, truncating getsockopt
- complete prctl, enforced seccomp, child subreapers
- real file locks, complete fcntl, O_APPEND per open file, memfd flags and seals
- newfstatat/statx share one resolver: AT_EMPTY_PATH on any fd, statx follows links
- dup'd and inherited descriptors share their file offset
- orphans fall back to their own directory when the mount root refuses
- open files survive unlink and rename; real O_TMPFILE
- a page cache makes MAP_SHARED file mappings really shared
- /dev/shm is a real writable tmpfs
- futex timeouts, process-shared keys, futex_waitv, futex2, WAKE_OP, PI locks
- *(aarch64)* rewrite the A64 decoder as a complete, verified Armv8.0 EL0 ISA
- POSIX message queues as pollable descriptors
- System V IPC — message queues, semaphores, and truly shared memory
- real state for scheduling, ioprio, NUMA, pkeys, mseal, rseq, membarrier, caps
- pidfd_open/send_signal/getfd, waitid(P_PIDFD), kcmp, process_vm_*
- POSIX timers with real signal delivery; fix timer/signal inheritance
- splice/tee/vmsplice, preadv2/pwritev2, openat2, cachestat
- real extended attributes on the in-memory filesystems
- decode every syscall number on both arches; refuse the unavailable set deliberately
- Make nginx run: directory symlinks, NEON/FP gaps, socket and epoll lifetimes
- show the tunnel as tun0 in ip addr/ip route/ifconfig
- getsockname on netlink sockets returns a full sockaddr_nl (fixes `ip addr`)
- clicking the net address copies it; on/off moves to a separate ⏻ toggle

## [0.0.3](https://github.com/KarpelesLab/nixvm/compare/v0.0.2...v0.0.3) - 2026-10-03

### Other

- NIXVM_WANT_NODE restricts the SYN probe to one grouterd node
- SYN probe prints each failed flow (UTC start, destination, inside source port)
- SYN probe (NIXVM_SYN_PROBE=n) in the tunnel throughput test
- run `#!` scripts through their interpreter (fixes apk scripts)
- keep the tab responsive while the guest computes (time-sliced pump)
- assert_is_empty also covers the negated form (assert_ne!)
- satisfy Rust 1.99's clippy::assert_is_empty
- show the IPv6 validity as an HH:MM:SS countdown
- show how long the guest's IPv6 address stays valid
- show the guest's IPv6 address in the net button, not the NATed IPv4
- pktkit 0.1.11
- classify the tunnel's bad TCP checksums; dump raw packets
- MSS sweep and inbound TCP checksum check in the tunnel throughput test
- keep a task visible while its syscall is serviced (fixes flaky smp_e2e)

## [0.0.2](https://github.com/KarpelesLab/nixvm/compare/v0.0.1...v0.0.2) - 2026-09-29

### Other

- pktkit 0.1.10
- multi-target tunnel throughput test; bridge with a TCP_NODELAY switch
- pktkit 0.1.8; add a live tunnel throughput diagnostic
- Ctrl-C interrupts the running command; sub-millisecond guest clocks
- arm64 guest fixes: apk update over HTTPS works (NEON, mremap, renameat, kills)
- *(aarch64)* NEON single-structure loads/stores, PMULL; coverage scanner
- ICMP sockets (ping) over the tunnel, answered locally on loopback
- use the guest arch's O_DIRECTORY/O_NOFOLLOW (fixes arm64 symlinks)

## [0.0.1](https://github.com/KarpelesLab/nixvm/compare/v0.0.0...v0.0.1) - 2026-09-27

### Fixed

- *(wasm)* remove `*/` from Terminal doc — it broke the generated pkg/nixvm.js

### Other

- disable semver checks (published 0.0.0 baseline no longer builds)
- enable the fstool filesystem features nixvm uses (fixes Release-plz)
- fix the Linux-only lints (KVM docs links, raw borrow in clock, KvmVcpu bools)
- reuse the tunnel token across reconnects; banner when tokens run out
- fix clippy, rustdoc and a flaky test so CI can go green
- build the HVF backend only with the `hvf` feature
- cargo fmt (unblocks CI's format check)
- guest networking over a WebSocket IP tunnel (pktkit TCP/IP stack)
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
