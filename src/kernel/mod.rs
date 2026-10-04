//! The nixvm "kernel": an arch-agnostic engine that services guest syscalls
//! and schedules multiple guest processes.
//!
//! State is split between **global** kernel state (mount table, pipes, stdio,
//! process table) and the **running task's** state (`ServiceCtx`: its
//! `ProcInfo` — fds, cwd, brk, mmap arena, pid — plus the per-syscall `block`/
//! `yield_now`/`exec_ok` flags). The servicer owns a `ServiceCtx` for the
//! duration of a slice (built from the task's `ProcInfo`, restored after), and
//! threads `&mut cx` through the syscall handlers, which read/write `cx.cur.*`
//! for per-task state and `self.*` for globals. Making that state a passed-in
//! value rather than a single `Kernel` field is what lets several tasks be
//! serviced concurrently once the global lock is split (a later phase); today
//! it is still one servicer at a time. The scheduler ([`Kernel::run`]) is a
//! cooperative round-robin over `Process`es; a syscall that would block re-traps
//! later (we simply don't advance the guest PC), which the interpreter turns
//! back into the same syscall on the next slice.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::io::{self, Read, Write};
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Condvar, Mutex, mpsc};

use crate::abi::Arch;
use crate::abi::arch::{self, Sysno};
use crate::abi::errno::Errno;
use crate::fs::{Attrs, MountTable, NodeKind};
use crate::loader::{ProcessSpec, interp_path, load_dynamic, load_static};
use crate::vcpu::mem::{PAGE_SIZE, Prot};
use crate::vcpu::{Exit, GuestMemory, Vcpu, VcpuError};

mod attrs;
pub mod egress;
mod fcntl;
mod fd;
mod fs_ext;
mod futex;
mod ipc;
mod mem_syscalls;
mod mqueue;
mod net;
mod orphan;
mod pagecache;
mod path;
mod poll;
mod prctl;
mod procx;
mod ptimer;
mod pty;
mod seccomp;
mod signal;
mod sockopt;
mod splice;
mod stat;
mod sys_misc;
#[cfg(test)]
mod testutil;
mod time;
mod unavailable;
mod xattr;

pub use fd::{Fd, FdTable, FileOffset};
use net::Net;
use poll::{EventFdInst, PidfdInst, PollFds};

/// The most one `read`/`write` transfers (`MAX_RW_COUNT`, `INT_MAX` rounded
/// down to a page); larger requests are short transfers, as on Linux.
const MAX_RW_COUNT: u64 = 0x7fff_f000;

/// `dirfd` value meaning "resolve relative to the current working directory".
const AT_FDCWD: i64 = -100;
/// Max symlink hops before `ELOOP` (matches Linux's `MAXSYMLINKS`).
const SYMLINK_MAX: u32 = 40;

/// Per-process kernel-side state (swapped into `Kernel::cur` while running).
#[derive(Clone)]
#[allow(clippy::struct_excessive_bools)]
struct ProcInfo {
    fds: FdTable,
    /// Current working directory. Shared across a `CLONE_FS` group via
    /// [`Shared::cwd_tables`] (indexed by [`ProcInfo::fs`]): checked out into this
    /// field while the task runs, checked back in between slices so siblings see a
    /// `chdir`. A plain `fork` gets a private copy.
    cwd: String,
    brk: u64,
    heap_start: u64,
    heap_limit: u64,
    mmap_cursor: u64,
    mmap_floor: u64,
    /// Lowest address the initial thread's stack may grow to. A fault in
    /// `[stack_limit, stack_top)` on an unmapped page grows the stack there
    /// (Linux's `VM_GROWSDOWN`); only a small window is mapped at startup so a
    /// runtime that probes its own stack size doesn't measure the whole
    /// reservation. `stack_top` is the address space's top (`base + size`).
    stack_limit: u64,
    /// Task id (a.k.a. tid): unique per task, returned by `gettid`.
    pid: i32,
    ppid: i32,
    /// Thread-group id, returned by `getpid`. For a single-threaded process
    /// `tgid == pid`; threads created with `CLONE_THREAD` share the leader's
    /// `tgid` but keep distinct `pid`s.
    tgid: i32,
    /// True for a `CLONE_THREAD` task (a thread, not a child process). Threads
    /// are not reaped by their parent's `wait4`.
    is_thread: bool,
    /// Signal delivered to the parent when this task terminates (the low byte of
    /// `clone`'s flags, or `clone3`'s `exit_signal`; `SIGCHLD`=17 for a plain
    /// `fork`). `0` for a `CLONE_THREAD` task — an individual thread's death does
    /// not signal the parent; only the thread group's exit does, and it carries
    /// the group *leader's* `exit_signal`.
    exit_signal: u8,
    /// Address-space id: an index into [`Kernel::spaces`]. Threads that share
    /// memory (`CLONE_VM`) share one `mm`; a forked child gets a fresh copy.
    mm: usize,
    /// File-descriptor-table id: an index into [`Kernel::file_tables`]. Threads
    /// created with `CLONE_FILES` (every pthread) share one table, so an fd
    /// opened by one thread is visible to all — load-bearing for libuv, whose
    /// async wakeups write an eventfd from one thread that another polls. A
    /// forked child gets a private copy. While a task runs its slice its table
    /// is *checked out* into [`ProcInfo::fds`]; between slices it lives in
    /// `file_tables[files]` (and `fds` holds an empty placeholder).
    files: usize,
    /// Filesystem-context id: an index into [`Shared::cwd_tables`]. Tasks created
    /// with `CLONE_FS` (every pthread, via `CLONE_THREAD`'s implied fs-sharing)
    /// share one entry, so a `chdir` in one is seen by all; a plain `fork` gets a
    /// private copy. Like [`ProcInfo::files`] the entry is *checked out* into
    /// [`ProcInfo::cwd`] while the task runs, and lives in `cwd_tables[fs]`
    /// between slices. (The `umask` half of a filesystem context is already a
    /// single global in [`Shared::umask`], and there is no per-task chroot root,
    /// so `cwd` is the only per-task fs state a `CLONE_FS` share must unify.)
    fs: usize,
    /// `set_tid_address` / `CLONE_CHILD_CLEARTID`: on exit, zero this guest
    /// word and futex-wake it (lets `pthread_join` return). 0 = unset.
    clear_child_tid: u64,
    /// When `Some((mm, uaddr))`, this task is parked in `FUTEX_WAIT` on that
    /// address; cleared when woken.
    futex_wait: Option<futex::FutexKey>,
    /// The `FUTEX_WAIT_BITSET` mask of the current wait (all ones for a plain
    /// wait); a wake only releases waiters whose mask intersects its own.
    futex_bitset: u32,
    /// `futex_waitv`: every word the task is parked on (empty otherwise); a
    /// wake on any of them releases it, recording which in `futex_waitv_idx`.
    futex_waitv: Vec<futex::FutexKey>,
    futex_waitv_idx: usize,
    /// The current futex wait is a PI lock wait (`FUTEX_LOCK_PI`): only
    /// `FUTEX_UNLOCK_PI` (which hands over ownership) releases it.
    futex_pi: bool,
    /// Set by `FUTEX_WAKE` to release a parked waiter on its next slice.
    futex_woken: bool,
    run: RunState,
    /// Job control: set `true` when a posted SIGCONT resumed this task from
    /// `RunState::Stopped`, latching a "continued" event for a
    /// `wait4(WCONTINUED)`/`waitid(WCONTINUED)` parent to report. Cleared when
    /// reported. Irrelevant across fork (a fresh child starts `false`).
    continued: bool,
    /// Job control: set `true` once a `wait4(WUNTRACED)`/`waitid(WSTOPPED)`
    /// parent has reported this task's current stop, so the same stop isn't
    /// reported again on a later wait. Reset each time the task stops afresh (or
    /// is continued). Irrelevant across fork.
    stop_reported: bool,
    /// Per-signal disposition (handler address / `SIG_DFL` / `SIG_IGN`, plus the
    /// flags, restorer, and mask from `rt_sigaction`). Indexed by signal number
    /// (1..=64); index 0 is unused.
    handlers: [SigAction; 65],
    /// Alternate signal stack (`sigaltstack`): `(base, size, flags)`. A handler
    /// registered `SA_ONSTACK` runs here instead of the interrupted stack —
    /// which is exactly how a runtime catches its own stack-overflow fault.
    altstack: (u64, u64, u64),
    /// Blocked-signal mask (bit `sig-1` set = blocked).
    blocked: u64,
    /// Pending-signal mask (bit `sig-1` set = pending).
    pending: u64,
    /// While a `sigsuspend` is in progress, the signal mask to restore when it
    /// returns (POSIX: `sigsuspend` installs a temporary mask, then restores the
    /// pre-call mask once a signal is delivered). `None` when no `sigsuspend` is
    /// active. Taken by the delivered handler (used as its `uc_sigmask`) or, if
    /// the wake was on an ignored signal, restored by `deliver_pending_signals`.
    sigsuspend_prev: Option<u64>,
    /// Process-group id (`setpgid`/`getpgid`/`getpgrp`). `0` means "not set
    /// yet — defaults to `pid`". Inherited across `fork`.
    pgid: i32,
    /// Session id (`setsid`/`getsid`). `0` means "defaults to `pid`".
    sid: i32,
    /// Real/effective/saved/fs user and group ids. The VM starts as root
    /// (all 0); a process may drop privileges (`setuid`/`setgid`/`setres*`),
    /// after which regaining them is `EPERM` — so a dropped-privilege program
    /// behaves correctly instead of silently staying root.
    creds: Creds,
    /// Scheduling attributes, reported back to the guest (the cooperative
    /// scheduler doesn't actually honor them, but a program that sets and reads
    /// them back must see what it set): policy (`SCHED_OTHER`=0/`FIFO`=1/`RR`=2/
    /// …), real-time priority, `nice` (−20..19), and CPU-affinity mask.
    sched_policy: i32,
    sched_priority: i32,
    nice: i32,
    affinity: u64,
    /// Parked: the task blocked on its last slice (futex/poll/wait4/stdin) and
    /// should not be re-run until something might wake it. Distinct from
    /// `RunState::Running` so the scheduler doesn't busy-spin re-running a
    /// blocked task, and so "is another task runnable?" excludes parked
    /// siblings (else a thread group all parks itself into a false deadlock).
    parked: bool,
    /// Writable file-backed `MAP_SHARED` mappings, flushed back to their file
    /// on `munmap`/`msync`/exit. This is how `apk` (and `install`, `cp
    /// --sparse`, …) writes large extracted files: create → `ftruncate` →
    /// `mmap(MAP_SHARED, PROT_WRITE)` → memcpy → `munmap`. Without write-back
    /// the file stays zero-filled at the right size.
    shared_maps: Vec<SharedMap>,
    /// Absolute wall-clock deadline (ns since the UNIX epoch) at which a timed
    /// wait (`poll`/`ppoll`/`epoll_pwait` with a finite timeout) gives up and
    /// returns 0. `None` when the task holds no timed wait. Set on the first
    /// re-trap of the blocking syscall and checked on each later re-trap; once
    /// the wall clock passes it, the syscall completes with "timed out" instead
    /// of re-parking. This is what makes `setTimeout` fire — libuv sleeps in
    /// `epoll_pwait(timeout)` until the next timer is due.
    wake_deadline: Option<u128>,
    /// CPU time consumed by this task, in nanoseconds: the wall time it has spent
    /// actually executing (each dispatched step's run + service), accumulated by
    /// the schedulers. A parked/blocked task doesn't advance it — which is what
    /// makes `CLOCK_THREAD_CPUTIME_ID` (this field) and `CLOCK_PROCESS_CPUTIME_ID`
    /// (the thread-group sum) measure CPU rather than wall time, per task rather
    /// than per host process.
    cpu_ns: u128,
    /// Weighted *virtual* runtime for fair scheduling (CFS-style), in nice-0
    /// nanosecond units: each executed step adds its CPU delta scaled by
    /// `NICE0_WEIGHT / nice_weight(nice)`, so a low-priority (higher-`nice`) task's
    /// vruntime climbs faster and it is picked less. Both schedulers always run
    /// the runnable task with the *least* vruntime (clamped up to
    /// [`Shared::min_vruntime`] so a long-blocked task can't hoard the CPU on
    /// wake). Unlike [`Self::cpu_ns`] this is not real time — it exists only to
    /// order the run queue. A forked child inherits the parent's value (via the
    /// `ProcInfo` clone) so it gets no free head start.
    vruntime: u128,
    /// CPU time (ns) of this task's *reaped* children, accumulated on `wait4`/
    /// `waitid` — Linux's `ru_utime` for `RUSAGE_CHILDREN` and `tms_cutime`.
    child_cpu_ns: u128,
    /// `ITIMER_REAL` (`alarm`/`setitimer`): absolute wall-clock deadline (ns since
    /// the epoch) at which `SIGALRM` is posted, or `None` when disarmed. The
    /// scheduler wakes a parked task at this deadline (like [`Self::wake_deadline`])
    /// so a real-time timer can interrupt a blocking syscall.
    alarm_deadline: Option<u128>,
    /// `ITIMER_REAL` reload interval (ns); `0` for a one-shot `alarm`. When the
    /// timer fires, `alarm_deadline` advances by this if non-zero, else disarms.
    alarm_interval_ns: u128,
    /// Absolute path of the running program (last `execve`, or the initial image),
    /// for the `/proc/self/exe` symlink. Empty until set.
    exe: String,
    /// The command name (`/proc/self/comm`, the `stat`/`status` name field, and
    /// what `PR_GET_NAME` returns). Initialized to the executable's basename at
    /// `execve`/boot, truncated to 15 bytes; `PR_SET_NAME`/`pthread_setname_np`
    /// overwrite it. Per-task so threads can name themselves independently.
    comm: String,
    /// The full launch command line (`/proc/self/cmdline`): the task's `argv`
    /// joined by NULs, exactly as the kernel presents it. Set at `execve`/boot.
    cmdline: Vec<u8>,
    /// The auxiliary vector the image was started with (`/proc/self/auxv`,
    /// Linux's `mm->saved_auxv`): raw `(type, value)` words through
    /// `AT_NULL`, captured at `execve`/boot and inherited across `fork`.
    auxv: Vec<u8>,
    /// arm64: the address of this image's `rt_sigreturn` trampoline page —
    /// what Linux's vDSO `__kernel_rt_sigreturn` provides to handlers
    /// installed without `SA_RESTORER` (Go's). Mapped on first need, 0 until
    /// then; inherited across `fork` (the page is copied), reset by `execve`.
    sigtramp: u64,
    /// `PR_SET_NO_NEW_PRIVS` latch (sandboxing setups set and re-check it).
    no_new_privs: bool,
    /// `PR_SET_PDEATHSIG`: signal to send when the parent dies (stored/reported;
    /// delivery-on-parent-death is not yet wired). `0` = none.
    pdeathsig: u64,
    /// `PR_SET_DUMPABLE` state: `1` = `SUID_DUMP_USER` (the default), `0` =
    /// not dumpable, `2` = dumpable-by-root. Sandboxes set this and re-read it,
    /// so it must round-trip even though we never produce core dumps.
    dumpable: u64,
    /// The siginfo accompanying each pending *standard* signal (index = signal
    /// number), so an `SA_SIGINFO` handler sees the right `si_code`/`si_pid`/
    /// `si_value`. Standard signals coalesce (one `pending` bit), so this keeps
    /// the most recent info; `None` for a signal posted without siginfo (a bare
    /// `kill`, a fault). Real-time signals use [`Self::rt_queue`] instead.
    queued_siginfo: [Option<QueuedSig>; NSIG_SLOTS],
    /// Real-time signals (`>= SIGRTMIN`, 32) *queue* rather than coalesce: each
    /// send is delivered separately, FIFO, carrying its own `si_value`. Keyed by
    /// signal number; the `pending` bit for an RT signal mirrors "its queue is
    /// non-empty".
    rt_queue: BTreeMap<u32, VecDeque<QueuedSig>>,
    /// POSIX timers (`timer_create`) whose expirations are delivered to this
    /// task — see [`ptimer`]. Like `ITIMER_REAL` they wake the task at their
    /// deadline ([`ProcInfo::timer_deadline`]); unlike it, neither `fork`
    /// children nor new threads inherit them, and `execve` deletes them.
    ptimers: Vec<ptimer::PosixTimer>,
    /// `ioprio_set` value (`class << 13 | level`); 0 = never set (reads back
    /// as the nice-derived best-effort default).
    ioprio: u16,
    /// `set_mempolicy` mode and nodemask (node 0 is the only node).
    mempolicy: (u16, u64),
    /// Registered `rseq` area: `(address, length, signature)`. Per thread: a
    /// new thread starts unregistered, a fork child keeps the registration
    /// (same address in its copied memory), `execve` drops it.
    rseq: Option<(u64, u32, u32)>,
    /// `personality(2)` persona (0 = `PER_LINUX`).
    personality: u32,
    /// `set_robust_list` head (reported by `get_robust_list`).
    robust_list: u64,
    /// Capability sets `[effective, permitted, inheritable]` after a
    /// `capset`; `None` = the default for the task's uid (see `attrs.rs`).
    caps: Option<[u64; 3]>,
    /// The System V semaphore this task is parked on in `semop`:
    /// `(semid, semnum, waiting-for-zero)` — what `GETNCNT`/`GETZCNT` count.
    sem_wait: Option<(i32, u16, bool)>,
    /// Supplementary group ids (`setgroups`/`getgroups`).
    groups: Vec<u32>,
    /// x86-64: the FS base last installed (`arch_prctl(ARCH_SET_FS)` or
    /// `CLONE_SETTLS`), reported by `ARCH_GET_FS`.
    fs_base: u64,
    /// Assorted `prctl` state (see [`prctl`]).
    pr: prctl::PrctlState,
    /// seccomp mode and filters (see [`seccomp`]); inherited by children,
    /// kept across `execve`.
    seccomp: seccomp::SeccompState,
}

/// The subset of `siginfo_t` a queued/sent signal carries beyond its number,
/// filled by `rt_sigqueueinfo`/`kill` and written into the delivered frame.
#[derive(Clone, Copy, Debug)]
struct QueuedSig {
    /// `si_code`: `SI_USER` (0, from `kill`), `SI_QUEUE` (-1, from `sigqueue`), …
    code: i32,
    /// `si_pid`: the sending process.
    pid: i32,
    /// `si_uid`: the sending user (always 0 — the VM is single-user root).
    uid: u32,
    /// `si_value` (the `sigqueue` payload): an 8-byte union of int and pointer.
    value: u64,
}

/// Signal-table size: signals `1..=64` plus the unused index 0.
const NSIG_SLOTS: usize = 65;

/// A writable file-backed `MAP_SHARED` region awaiting flush-back.
#[derive(Clone, Debug)]
struct SharedMap {
    base: u64,
    len: u64,
    path: String,
    offset: u64,
}

impl Default for ProcInfo {
    fn default() -> Self {
        Self {
            fds: FdTable::with_standard_streams(),
            cwd: "/".to_string(),
            brk: 0,
            heap_start: 0,
            heap_limit: 0,
            mmap_cursor: 0,
            mmap_floor: 0,
            stack_limit: 0,
            pid: 0,
            ppid: 0,
            tgid: 0,
            is_thread: false,
            exit_signal: SIGCHLD as u8, // a plain fork signals SIGCHLD on exit
            mm: 0,
            files: 0,
            fs: 0,
            clear_child_tid: 0,
            futex_wait: None,
            futex_bitset: u32::MAX,
            futex_waitv: Vec::new(),
            futex_waitv_idx: 0,
            futex_pi: false,
            futex_woken: false,
            run: RunState::Running,
            continued: false,
            stop_reported: false,
            handlers: [SigAction::default(); 65],
            altstack: (0, 0, SS_DISABLE),
            blocked: 0,
            pending: 0,
            sigsuspend_prev: None,
            pgid: 0,
            sid: 0,
            parked: false,
            shared_maps: Vec::new(),
            wake_deadline: None,
            cpu_ns: 0,
            vruntime: 0,
            child_cpu_ns: 0,
            alarm_deadline: None,
            alarm_interval_ns: 0,
            exe: String::new(),
            comm: String::new(),
            cmdline: Vec::new(),
            auxv: Vec::new(),
            sigtramp: 0,
            no_new_privs: false,
            pdeathsig: 0,
            dumpable: 1,
            queued_siginfo: [None; NSIG_SLOTS],
            rt_queue: BTreeMap::new(),
            ptimers: Vec::new(),
            ioprio: 0,
            mempolicy: (0, 0),
            rseq: None,
            personality: 0,
            robust_list: 0,
            caps: None,
            sem_wait: None,
            groups: Vec::new(),
            fs_base: 0,
            pr: prctl::PrctlState::default(),
            seccomp: seccomp::SeccompState::default(),
            creds: Creds::default(),
            sched_policy: 0, // SCHED_OTHER
            sched_priority: 0,
            nice: 0,
            affinity: 0, // 0 = "all CPUs" (default; never a real mask)
        }
    }
}

/// A process's real/effective/saved-set/filesystem user and group ids. Default
/// is all-zero (root), and privileged (`euid == 0`) transitions can set any id;
/// once dropped, only the current real/effective/saved set may be restored.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Creds {
    ruid: u32,
    euid: u32,
    suid: u32,
    fsuid: u32,
    rgid: u32,
    egid: u32,
    sgid: u32,
    fsgid: u32,
}

impl ProcInfo {
    /// The earliest real-time timer deadline this task holds — its
    /// `ITIMER_REAL` (`alarm`/`setitimer`) or any armed POSIX timer — which the
    /// schedulers sleep until / wake the task at, like [`Self::wake_deadline`].
    fn timer_deadline(&self) -> Option<u128> {
        self.ptimers
            .iter()
            .filter_map(|t| t.deadline)
            .chain(self.alarm_deadline)
            .min()
    }

    /// Record the siginfo for a just-posted signal. Real-time signals (`>=
    /// SIGRTMIN`) append to their FIFO queue (each delivery is distinct);
    /// standard signals coalesce, keeping the newest info.
    fn post_siginfo(&mut self, sig: u64, qs: QueuedSig) {
        if sig >= SIGRTMIN {
            self.rt_queue.entry(sig as u32).or_default().push_back(qs);
        } else if (sig as usize) < NSIG_SLOTS {
            self.queued_siginfo[sig as usize] = Some(qs);
        }
    }

    /// Consume the siginfo for a signal about to be delivered/dequeued, and
    /// report whether *more* of that signal remains queued (only possible for
    /// real-time signals, whose `pending` bit must then stay set to redeliver).
    fn take_siginfo(&mut self, sig: u64) -> (Option<QueuedSig>, bool) {
        if sig >= SIGRTMIN {
            if let Some(q) = self.rt_queue.get_mut(&(sig as u32)) {
                let info = q.pop_front();
                let more = !q.is_empty();
                if !more {
                    self.rt_queue.remove(&(sig as u32));
                }
                return (info, more);
            }
            return (None, false);
        }
        let info = if (sig as usize) < NSIG_SLOTS {
            self.queued_siginfo[sig as usize].take()
        } else {
            None
        };
        (info, false)
    }

    /// Drop a real-time signal's whole queue (it was ignored / the process is
    /// dying). A no-op for standard signals.
    fn drain_rt(&mut self, sig: u64) {
        if sig >= SIGRTMIN {
            self.rt_queue.remove(&(sig as u32));
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RunState {
    Running,
    /// Job-control stopped by SIGSTOP/SIGTSTP/SIGTTIN/SIGTTOU; carries the stop
    /// signal (for `WSTOPSIG`/`si_status`). A stopped task is not `Running`, so
    /// the scheduler's runnable predicates never dispatch it; a posted SIGCONT
    /// flips it back to `Running`. Reported to a `wait4(WUNTRACED)`/
    /// `waitid(WSTOPPED)` parent, then latched via `stop_reported`.
    Stopped(i32),
    Zombie(ExitCause),
}

/// How a task died, so `wait4`/`waitid` can encode the distinction Linux makes
/// between a normal exit and a signal death. The old model squashed both into a
/// single "shell exit code" (`128 + signal`), which `wait4` then always encoded
/// as `WIFEXITED` — so a child killed by a signal reported `WIFSIGNALED == 0`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ExitCause {
    /// Terminated normally via `exit`/`exit_group`; carries the exit code (only
    /// the low 8 bits are meaningful to `waitpid`).
    Exited(i32),
    /// Terminated by a signal's default (fatal) action or an uncaught fatal
    /// fault; carries the signal number.
    Signaled(i32),
}

impl ExitCause {
    /// The `waitpid` status word: `code << 8` for a normal exit (`WIFEXITED`),
    /// or the bare signal number for a signal death (`WIFSIGNALED`). The
    /// core-dump bit (`0x80`) is never set — nixvm produces no core files.
    fn wait_status(self) -> u32 {
        match self {
            ExitCause::Exited(c) => ((c & 0xff) as u32) << 8,
            ExitCause::Signaled(s) => (s & 0x7f) as u32,
        }
    }

    /// The scalar exit code a shell reports in `$?` (and nixvm returns as pid 1's
    /// process exit): the exit code, or `128 + signal` for a signal death.
    fn shell_code(self) -> i32 {
        match self {
            ExitCause::Exited(c) => c & 0xff,
            ExitCause::Signaled(s) => 128 + s,
        }
    }

    /// `waitid` `si_code`: `CLD_EXITED` (1) or `CLD_KILLED` (2).
    fn si_code(self) -> i32 {
        match self {
            ExitCause::Exited(_) => 1,
            ExitCause::Signaled(_) => 2,
        }
    }

    /// `waitid` `si_status`: the exit code for a normal exit, else the signal.
    fn si_status(self) -> i32 {
        match self {
            ExitCause::Exited(c) => c & 0xff,
            ExitCause::Signaled(s) => s,
        }
    }
}

/// Outcome of a [`Kernel::pump`] step in interactive mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pumped {
    /// pid 1 exited with this code; the machine is done.
    Exited(i32),
    /// Every runnable task is parked waiting for input; feed stdin and pump
    /// again to resume (e.g. the shell is blocked on a `read` of its terminal).
    Blocked,
    /// [`Kernel::pump_for`]'s time budget ran out with work still to do: the
    /// guest is computing. Pump again soon (after letting the embedder's
    /// event loop run).
    Busy,
}

/// The result of servicing one guest exit, telling the scheduler what to do
/// with the task's vcpu next.
/// Unmapped guard reserved between the initial stack's low bound and the top of
/// the anonymous-`mmap` arena, mirroring Linux's `stack_guard_gap` (256 pages).
///
/// A runtime that measures its own stack by probing downward until it hits
/// unmapped memory (JSC/Bun does this with `mremap`, to size its JS-recursion
/// limit) must find that boundary at the real stack bottom. Without the gap the
/// arena's first mapping sits flush against the stack, so the probe walks past
/// the true bottom and the runtime concludes it has a far larger stack than is
/// mapped — then recurses off the end of it into the heap.
const STACK_GUARD_GAP: u64 = 256 * PAGE_SIZE;

/// `sigaltstack` disabled (`SS_DISABLE`).
const SS_DISABLE: u64 = 2;
/// `sigaction` flag: run the handler on the alternate signal stack.
const SA_ONSTACK: u64 = 0x0800_0000;
/// `sigaction` flag: restart an interruptible syscall after the handler returns
/// (rather than failing it with `EINTR`).
const SA_RESTART: u64 = 0x1000_0000;
/// `sigaction` flag: don't block the signal itself while its handler runs, so
/// the handler can be re-entered by another instance of the same signal.
const SA_NODEFER: u64 = 0x4000_0000;
/// `sigaction` flag: reset the disposition to `SIG_DFL` on entry to the handler
/// (a one-shot handler; the classic `signal()` semantics).
const SA_RESETHAND: u64 = 0x8000_0000;
/// Lowest real-time signal (`SIGRTMIN` at the kernel ABI). Signals `>=` this
/// queue rather than coalesce.
const SIGRTMIN: u64 = 32;
/// The synchronous fault signals this kernel can deliver to a handler.
const SIGILL: u64 = 4;
const SIGTRAP: u64 = 5;
const SIGBUS: u64 = 7;
const SIGSEGV: u64 = 11;
/// Posted when an `ITIMER_REAL` (`alarm`/`setitimer`) deadline passes.
const SIGALRM: u64 = 14;
/// Posted to a process that writes to a pipe/socket with no reader.
const SIGPIPE: u64 = 13;
/// Sent to a process's parent when a child terminates (so a blocked `wait`
/// wakes to reap it).
const SIGCHLD: u64 = 17;

// ---- clone(2)/clone3(2) flags (asm-generic; identical on x86-64 & aarch64) ---
// The child's termination signal is the low byte of `clone`'s `flags` (or
// `clone3`'s `exit_signal` field); these named bits occupy the high 3 bytes.
/// Share the address space with the caller (a thread) rather than copying it.
const CLONE_VM: u64 = 0x0000_0100;
/// Share the filesystem context (cwd/umask/root) — a `chdir` is seen groupwide.
const CLONE_FS: u64 = 0x0000_0200;
/// Share the open-file-descriptor table.
const CLONE_FILES: u64 = 0x0000_0400;
/// Share the table of signal handlers (implied by `CLONE_THREAD`).
const CLONE_SIGHAND: u64 = 0x0000_0800;
/// Allocate a pidfd for the child and store it at the `parent_tid`/`pidfd` slot.
const CLONE_PIDFD: u64 = 0x0000_1000;
/// Continue tracing the child if the caller is being traced (ptrace).
const CLONE_PTRACE: u64 = 0x0000_2000;
/// Suspend the caller until the child `execve`s or exits (`vfork`).
const CLONE_VFORK: u64 = 0x0000_4000;
/// Make the child a sibling: its parent is the caller's parent, not the caller.
const CLONE_PARENT: u64 = 0x0000_8000;
/// Put the child in the caller's thread group (shared tgid; a thread).
const CLONE_THREAD: u64 = 0x0001_0000;
/// New mount namespace.
const CLONE_NEWNS: u64 = 0x0002_0000;
/// Share System V semaphore undo state.
const CLONE_SYSVSEM: u64 = 0x0004_0000;
/// Seed the child's thread-pointer (TLS) register.
const CLONE_SETTLS: u64 = 0x0008_0000;
/// Write the child's tid into the caller's memory at `parent_tid`.
const CLONE_PARENT_SETTID: u64 = 0x0010_0000;
/// Zero+futex-wake the child's `child_tid` word when the child exits (join).
const CLONE_CHILD_CLEARTID: u64 = 0x0020_0000;
/// Obsolete no-op (historically "detach"; the kernel ignores it).
const CLONE_DETACHED: u64 = 0x0040_0000;
/// Forbid the tracing process from forcing `CLONE_PTRACE` on this child.
const CLONE_UNTRACED: u64 = 0x0080_0000;
/// Write the child's tid into the child's memory at `child_tid`.
const CLONE_CHILD_SETTID: u64 = 0x0100_0000;
/// New cgroup namespace.
const CLONE_NEWCGROUP: u64 = 0x0200_0000;
/// New UTS (hostname) namespace.
const CLONE_NEWUTS: u64 = 0x0400_0000;
/// New System V IPC namespace.
const CLONE_NEWIPC: u64 = 0x0800_0000;
/// New user namespace.
const CLONE_NEWUSER: u64 = 0x1000_0000;
/// New PID namespace.
const CLONE_NEWPID: u64 = 0x2000_0000;
/// New network namespace.
const CLONE_NEWNET: u64 = 0x4000_0000;
/// Share the I/O context (block-layer scheduling).
const CLONE_IO: u64 = 0x8000_0000;
/// (`clone3` only) Reset all signal handlers to `SIG_DFL` in the child.
const CLONE_CLEAR_SIGHAND: u64 = 0x1_0000_0000;
/// (`clone3` only) Place the child into the cgroup named by `cl_args.cgroup`.
const CLONE_INTO_CGROUP: u64 = 0x2_0000_0000;
/// Every `clone`/`clone3` flag nixvm recognizes (the union of the bits above),
/// so an unknown high bit can be reported as `EINVAL` the way Linux does.
const CLONE_ALL_FLAGS: u64 = CLONE_VM
    | CLONE_FS
    | CLONE_FILES
    | CLONE_SIGHAND
    | CLONE_PIDFD
    | CLONE_PTRACE
    | CLONE_VFORK
    | CLONE_PARENT
    | CLONE_THREAD
    | CLONE_NEWNS
    | CLONE_SYSVSEM
    | CLONE_SETTLS
    | CLONE_PARENT_SETTID
    | CLONE_CHILD_CLEARTID
    | CLONE_DETACHED
    | CLONE_UNTRACED
    | CLONE_CHILD_SETTID
    | CLONE_NEWCGROUP
    | CLONE_NEWUTS
    | CLONE_NEWIPC
    | CLONE_NEWUSER
    | CLONE_NEWPID
    | CLONE_NEWNET
    | CLONE_IO
    | CLONE_CLEAR_SIGHAND
    | CLONE_INTO_CGROUP;
// nixvm models a single global namespace of each `CLONE_NEW*` kind, and runs as
// root (for whom real Linux *succeeds* and creates a namespace), so those flags
// are ACCEPTED as no-ops — the child runs against the one global view — rather
// than returning `EPERM`/`EINVAL`. That is the closest faithful behavior a
// namespace-less VM can offer; failing would wrongly break a root program that
// legitimately asks for a fresh namespace.

/// The fully-decoded arguments of a `clone`/`clone3`, normalized so the shared
/// [`Kernel::do_clone`] core is arch- and syscall-independent. Legacy `clone`
/// packs the termination signal into the low byte of `flags` and (for
/// `CLONE_PIDFD`) reuses the `parent_tid` pointer as the pidfd output; `clone3`
/// carries both as their own fields. Both are lowered to this.
struct CloneArgs {
    /// The clone flags with the exit-signal byte masked off.
    flags: u64,
    /// Child stack pointer (top of the stack region; grows down). 0 = inherit.
    stack_ptr: u64,
    /// Guest address to write the child tid to (`CLONE_PARENT_SETTID`).
    parent_tid: u64,
    /// Guest address for the child-tid word (`CLONE_CHILD_SETTID`/`CLEARTID`).
    child_tid: u64,
    /// New thread pointer (`CLONE_SETTLS`).
    tls: u64,
    /// Signal delivered to the parent on the child's exit (0 for a thread).
    exit_signal: u64,
    /// Guest address to write the allocated pidfd to (`CLONE_PIDFD`), else 0.
    pidfd_ptr: u64,
}

/// One signal's disposition, as `rt_sigaction` records it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct SigAction {
    /// Handler address, `SIG_DFL` (0), or `SIG_IGN` (1).
    handler: u64,
    flags: u64,
    /// Trampoline the handler returns *to*; it invokes `rt_sigreturn`.
    restorer: u64,
    /// Signals blocked for the duration of the handler.
    mask: u64,
}

/// Top of the anonymous-`mmap` arena given the initial stack's low bound and the
/// arena floor: a guard gap below the stack (full [`STACK_GUARD_GAP`] when the
/// arena is roomy, one page when it is tiny — as in unit tests — so the arena
/// stays usable), clamped to the floor.
fn arena_top(stack_bottom: u64, floor: u64) -> u64 {
    let room = stack_bottom.saturating_sub(floor);
    let guard = if room > STACK_GUARD_GAP * 4 {
        STACK_GUARD_GAP
    } else {
        PAGE_SIZE
    };
    stack_bottom.saturating_sub(guard).max(floor)
}

/// The anonymous-`mmap` arena for one address space: a region `[floor, top)`
/// carved downward from `top`, plus a free list of ranges returned by `munmap`.
///
/// The free list is what makes this an allocator rather than a bump pointer. A
/// long-running guest that maps and unmaps repeatedly — a JS engine cycling JIT
/// code buffers and heap blocks is the extreme case — would otherwise walk the
/// cursor to the floor and start failing with `ENOMEM` while most of the arena
/// sat unused, since nothing ever reclaimed it.
#[derive(Debug, Clone, Default)]
struct Arena {
    /// Next bump allocation ends here (allocations grow *down* from `top`).
    cursor: u64,
    /// Allocations may not go below this.
    floor: u64,
    /// The arena's high bound — the initial `cursor`. Used to tell an address
    /// inside the arena (reclaimable) from one outside it (an image segment).
    top: u64,
    /// Freed ranges `(addr, len)`, sorted by address and coalesced.
    free: Vec<(u64, u64)>,
}

impl Arena {
    fn new(top: u64, floor: u64) -> Self {
        Self {
            cursor: top,
            floor,
            top,
            free: Vec::new(),
        }
    }

    /// Carve `len` bytes: reuse a freed range if one fits, else bump the cursor.
    /// `None` when the arena is exhausted.
    fn alloc(&mut self, len: u64) -> Option<u64> {
        // First fit over the free list, splitting the remainder back in.
        if let Some(i) = self.free.iter().position(|&(_, flen)| flen >= len) {
            let (addr, flen) = self.free[i];
            if flen == len {
                self.free.remove(i);
            } else {
                // Keep the low part free, hand back the high part, so adjacent
                // frees still coalesce downward.
                self.free[i] = (addr, flen - len);
            }
            return Some(addr + (flen - len));
        }
        let new_top = self.cursor.checked_sub(len)?;
        if new_top < self.floor {
            return None;
        }
        self.cursor = new_top;
        Some(new_top)
    }

    /// Whether `[addr, addr+len)` lies entirely inside one freed range — i.e.
    /// arena space nothing owns, which a caller may [`Arena::claim`].
    fn is_free(&self, addr: u64, len: u64) -> bool {
        let Some(end) = addr.checked_add(len) else {
            return false;
        };
        self.free.iter().any(|&(a, l)| a <= addr && end <= a + l)
    }

    /// Take `[addr, addr+len)` out of the arena's free space because something
    /// now occupies it without going through [`Arena::alloc`] (an in-place
    /// `mremap` grow, a `MAP_FIXED` mapping). Otherwise a later `alloc` would
    /// hand the same pages out again and the new mapping would zero-fill them
    /// under their owner. Covers both the free list and the never-used bump
    /// region below the cursor; ranges outside the arena are left alone.
    fn claim(&mut self, addr: u64, len: u64) {
        let Some(end) = addr.checked_add(len) else {
            return;
        };
        if len == 0 {
            return;
        }
        // Drop the overlap from every freed range, keeping any remainders.
        let mut kept = Vec::with_capacity(self.free.len() + 1);
        for &(a, l) in &self.free {
            let e = a + l;
            if e <= addr || a >= end {
                kept.push((a, l));
                continue;
            }
            if a < addr {
                kept.push((a, addr - a));
            }
            if e > end {
                kept.push((end, e - end));
            }
        }
        self.free = kept;
        // Inside the bump region: lower the cursor past the claimed range and
        // keep whatever lies between it and the old cursor as free space.
        if addr < self.cursor && end > self.floor {
            let old = self.cursor;
            self.cursor = addr.max(self.floor);
            if end < old {
                let pos = self.free.partition_point(|&(a, _)| a < end);
                self.free.insert(pos, (end, old - end));
            }
        }
    }

    /// Return `[addr, addr+len)` to the arena, coalescing with its neighbours.
    ///
    /// The guest is not trusted here: it may `munmap` an image segment, a
    /// `MAP_FIXED` range we never handed out, or the same range twice. Anything
    /// outside the *allocated* window `[cursor, top)` is ignored, and a range
    /// already on the free list is ignored — otherwise a double free would walk
    /// the cursor past `top` and later allocations would hand out addresses
    /// above the arena, i.e. inside the initial stack.
    fn free_range(&mut self, addr: u64, len: u64) {
        let Some(end) = addr.checked_add(len) else {
            return;
        };
        if len == 0 || addr < self.cursor || end > self.top {
            return;
        }
        // Already free? (double munmap, or a sub-range of a freed block)
        if self.free.iter().any(|&(a, l)| addr < a + l && a < end) {
            return;
        }
        // Sitting right on the cursor: give it straight back to the bump region
        // and absorb anything that just became adjacent.
        if addr == self.cursor {
            self.cursor = end;
            while let Some(i) = self.free.iter().position(|&(a, _)| a == self.cursor) {
                self.cursor += self.free[i].1;
                self.free.remove(i);
            }
            debug_assert!(self.cursor <= self.top);
            return;
        }
        let pos = self.free.partition_point(|&(a, _)| a < addr);
        self.free.insert(pos, (addr, len));
        // Coalesce with the next, then the previous, entry.
        if pos + 1 < self.free.len() {
            let (na, nl) = self.free[pos + 1];
            if end == na {
                self.free[pos].1 += nl;
                self.free.remove(pos + 1);
            }
        }
        if pos > 0 {
            let (pa, pl) = self.free[pos - 1];
            if pa + pl == addr {
                self.free[pos - 1].1 += self.free[pos].1;
                self.free.remove(pos);
            }
        }
    }
}

enum Serviced {
    /// Syscall done. `service` has already written the result into the vcpu (it
    /// does so before delivering any pending signal, so an interrupted syscall's
    /// signal frame captures the real return value); the caller just resumes,
    /// honoring `yield_now`. It must NOT re-write the result — the interpreter's
    /// `set_syscall_ret` advances the pc past `syscall`, so a second call drifts.
    SetRet,
    /// Resume compute without touching the result register (interrupt / execve
    /// replaced the image).
    Resume,
    /// The syscall would block; leave the guest PC on the `svc` and retry later.
    Blocked,
    /// The task became a zombie (exit, fault, or halt).
    Ended,
}

/// What one in-place SMP service step decided about the task's *next* step,
/// after [`Kernel::smp_service_step`] applied the syscall result to the vcpu.
/// The worker uses this to keep running the same vcpu (the hot path — no thread
/// hand-off) or to end its slice and report back to the scheduler.
enum SliceStep {
    /// Progress made; keep running the same vcpu.
    Continue,
    /// `sched_yield`: the task is still runnable but wants to give siblings a
    /// turn, so end the slice.
    Yielded,
    /// The task blocked (futex/poll/wait4/…); end the slice and park it.
    Blocked,
    /// The task became a zombie; end the slice.
    Ended,
}

/// Why an SMP worker's slice ended, shipped back to the scheduler main loop so
/// it can park/re-dispatch/reap the task. Carries the vcpu back so its home
/// worker keeps it (KVM vcpu→thread affinity).
enum SliceOutcome {
    /// The task blocked; `bool` is whether it serviced any syscall before
    /// blocking (i.e. made progress worth waking other blocked waiters for).
    Blocked(bool),
    /// The task became a zombie.
    Ended,
    /// The task yielded (still runnable).
    Yielded,
    /// The slice hit the syscall-count quantum (`slice_cap`) without blocking;
    /// the task is still runnable.
    Preempted,
    /// A backend error surfaced from `run`/`reconcile`.
    Err(VcpuError),
}

/// Poll interval while every task is parked but a host connection is live: the
/// scheduler sleeps this long, then retries a round that re-checks host-socket
/// readiness. Short enough that an arriving HTTP response is picked up promptly,
/// long enough that the idle poll isn't a busy spin.
const HOST_IO_POLL_NS: u128 = 1_000_000; // 1 ms

/// What the SMP scheduler does when every task is blocked and nothing is in
/// flight — mirrors the serial scheduler's stall handling.
enum StallAction {
    /// A timed wait is pending; sleep the main thread to this absolute deadline
    /// (ns since the epoch), then force a retry so the waiter re-checks it.
    SleepUntil(u128),
    /// No timer, first stall since the last progress: force one retry round to
    /// catch a lost wake / host I/O / a freshly-reaped child.
    Retry,
    /// No timer and the forced retry made no progress: a genuine deadlock.
    Deadlock,
}

/// One SMP worker's slice: run the guest lockless (KVM) or under the memory
/// lock (interpreter) to its next exit, then service that exit **in place** —
/// acquire the big kernel lock and call [`Kernel::smp_service_step`] — and, as
/// long as the task stays runnable, loop and run it again on this same thread.
/// This is the core of the syscall hot path: a guest doing millions of
/// `clock_gettime`s never leaves its worker thread, paying only an uncontended
/// lock per syscall instead of a full worker→main→worker hand-off.
///
/// Lock order is **memory lock → kernel lock**: the service step takes the
/// task's `Arc<Mutex<GuestMemory>>` first and the kernel lock second, and the
/// only other lock sites (a locked interpreter `run`, or a KVM `reconcile`) take
/// the memory lock alone. So the kernel lock is always the last lock acquired —
/// a worker never blocks on the memory lock while holding the kernel lock — and
/// with the service step only ever touching its *own* task's space there is no
/// lock cycle. (Holding the memory lock across the kernel lock, rather than the
/// reverse, keeps a long interpreter run in one address space from stalling
/// syscall servicing for every *other* address space.)
fn run_slice_smp(
    kernel: &Kernel,
    slice_cap: u32,
    i: usize,
    mut vcpu: Box<dyn Vcpu>,
    space: &Arc<Mutex<GuestMemory>>,
) -> (usize, Box<dyn Vcpu>, SliceOutcome) {
    let mut count: u32 = 0;
    let mut progressed = false;
    loop {
        // Step start, for CPU-time accounting (charged in `smp_service_step`).
        let step_start = crate::clock::now_monotonic().as_nanos();
        // ---- run phase: no kernel lock held ----
        // Interpreter reads/writes guest memory *through* GuestMemory, so it must
        // hold the memory lock for the whole run. KVM executes against the mapped
        // memslot: take the lock only to reconcile the memslot + shadow page
        // tables, then drop it so KVM_RUN runs in parallel with siblings of the
        // same address space.
        // A *shared* address space (CLONE_VM threads) must run serialized: hold the
        // per-space memory lock across the whole run so only one of its threads is
        // in KVM_RUN at a time. They share one page-table tree and one kstack
        // frame, so running them lockless-in-parallel corrupts each other's `#PF`
        // exception frame and races on page-table edits. Distinct processes have
        // distinct spaces (and locks), so this never serializes across processes.
        // The lock is dropped at the end of this block — before the service phase
        // re-acquires it — so a locked run never self-deadlocks.
        let exit = {
            let mut mem = space.lock().unwrap();
            if vcpu.needs_locked_run() || mem.is_shared() {
                vcpu.run(&mut mem)
            } else {
                let reconciled = vcpu.reconcile(&mut mem);
                drop(mem);
                match reconciled {
                    Ok(()) => vcpu.run_bare(),
                    Err(e) => Err(e),
                }
            }
        };
        let exit = match exit {
            Ok(e) => e,
            Err(e) => return (i, vcpu, SliceOutcome::Err(e)),
        };
        let is_syscall = matches!(exit, Exit::Syscall);
        // A time-quantum interrupt (mid-compute preemption) ends the slice so
        // the scheduler regains control: a syscall-free hot loop turns into a
        // stream of `Exit::Interrupted`s, and without ending the slice here the
        // worker would run that one task forever — never yielding its home
        // siblings a turn, and never letting the scheduler observe pid-1 exit or
        // drain the pool at shutdown.
        let is_interrupt = matches!(exit, Exit::Interrupted);
        // ---- service phase: hold the memory lock across the step (it is
        // outermost, and it serializes same-address-space siblings' service
        // phases). `smp_service_step` takes the kernel lock only briefly, for
        // the checkout/check-in bookkeeping; the syscall itself takes its own
        // per-handler locks (sh before vfs) while `sh` is *not* held, so other
        // workers service their syscalls concurrently (step B2). ----
        let step = {
            let mut mem = space.lock().unwrap();
            kernel.smp_service_step(i, exit, vcpu.as_mut(), &mut mem, step_start)
        };
        match step {
            SliceStep::Continue => {
                progressed = true;
                if is_interrupt {
                    return (i, vcpu, SliceOutcome::Preempted);
                }
                // Only real syscalls count toward the preemption quantum (a COW/
                // stack-grow fault resume does not), mirroring how `service`
                // increments `slice_syscalls`.
                if is_syscall {
                    count += 1;
                    if slice_cap != 0 && count >= slice_cap {
                        return (i, vcpu, SliceOutcome::Preempted);
                    }
                }
            }
            SliceStep::Yielded => return (i, vcpu, SliceOutcome::Yielded),
            SliceStep::Blocked => return (i, vcpu, SliceOutcome::Blocked(progressed)),
            SliceStep::Ended => return (i, vcpu, SliceOutcome::Ended),
        }
    }
}

/// A guest task (process or thread): its vcpu and per-task state. Its address
/// space lives in [`Kernel::spaces`] at `info.mm`, shared with any sibling
/// threads created via `CLONE_VM`. `vcpu` is `None` while the task is in
/// flight on an SMP worker thread (its compute running off the main thread).
struct Process {
    vcpu: Option<Box<dyn Vcpu>>,
    info: ProcInfo,
}

/// An in-kernel pipe: a byte buffer with reference counts for the open ends.
#[derive(Debug, Default)]
struct Pipe {
    buf: VecDeque<u8>,
    readers: usize,
    writers: usize,
    /// `O_NONBLOCK` on the read/write open file descriptions (set by
    /// `pipe2(O_NONBLOCK)` or `fcntl(F_SETFL)`/`ioctl(FIONBIO)`). A read of an
    /// empty non-blocking pipe returns `EAGAIN` instead of parking the task —
    /// without this a program that sets its pipe non-blocking (libuv's wakeup
    /// pipe, a self-pipe) deadlocks the whole VM on the first empty read.
    read_nonblock: bool,
    write_nonblock: bool,
}

/// The running task's mutable servicing state, owned by the servicer for the
/// duration of a slice instead of living in [`Kernel`]. Making it a passed-in
/// value (threaded as `&mut cx` through the syscall-servicing call graph) rather
/// than a `Kernel` field is what lets several tasks be serviced at once once the
/// kernel lock is split (a later phase); today the kernel stays single-servicer
/// under its big lock, so exactly one `ServiceCtx` is live at a time.
#[derive(Default)]
#[allow(clippy::struct_excessive_bools)] // independent one-shot flags, not a state enum
pub(super) struct ServiceCtx {
    /// The current task's per-process state (was `Kernel::cur`). Swapped/`take`n
    /// out of [`Kernel::procs`] for the slice and written back when it ends.
    cur: ProcInfo,
    /// Set by a handler when the syscall would block (re-trap it later). (Was
    /// `Kernel::block`.)
    block: bool,
    /// Set by `sched_yield`: end this task's slice but leave it *runnable*.
    /// Distinct from `block` (which parks the task until a wake) — a yielding
    /// task wants to go around again, just not before its siblings do. Without
    /// this the cooperative scheduler never leaves a thread that spins on
    /// `sched_yield` waiting for a sibling to make progress, and the whole
    /// process livelocks (Bun's event loop does exactly that). (Was
    /// `Kernel::yield_now`.)
    yield_now: bool,
    /// Set by `execve`/`rt_sigreturn` when it replaced the process image (resume
    /// at the new PC without setting a syscall return). (Was `Kernel::exec_ok`.)
    exec_ok: bool,
    /// Whether a syscall that sets `block` may be *restarted* (re-executed after a
    /// signal handler runs, per `SA_RESTART`). Defaults to `true` — the common
    /// case (`read`/`write`/`wait`/`accept`/`recv`/`futex`/…). The syscalls Linux
    /// never restarts (`poll`/`select`/`epoll_wait`/`pause`) clear it, so an
    /// interrupting handler makes them fail with `EINTR` regardless of `SA_RESTART`.
    restartable: bool,
    /// Set by the syscall-interruption path when a blocking syscall is being
    /// restarted (an `SA_RESTART` handler interrupted it): [`Kernel::deliver_async_signal`]
    /// rewinds the guest PC to re-execute the `syscall` after the handler returns.
    restart_syscall: bool,
    /// Syscalls serviced in the current slice (preemption quantum counter). (Was
    /// `Kernel::slice_syscalls`.)
    slice_syscalls: u32,
}

#[cfg(test)]
impl ServiceCtx {
    /// A minimal context for unit tests that exercise a single handler directly
    /// (a default task, restartable, no pending flags).
    pub(super) fn for_test() -> Self {
        Self {
            cur: ProcInfo::default(),
            block: false,
            yield_now: false,
            exec_ok: false,
            restartable: true,
            restart_syscall: false,
            slice_syscalls: 0,
        }
    }
}

/// The kernel: immutable-during-servicing config plus the coarse lock over all
/// mutable state (`Shared`).
///
/// Only the config fields live directly on `Kernel` (written just by `new`/the
/// pre-boot `set_*` setters, never during servicing); everything mutated while
/// servicing a syscall lives in `Shared` behind `shared`. Servicing therefore
/// takes `&self` + `&mut Shared` (B1): still exactly one coarse lock held for
/// one syscall at a time (the "big kernel lock"), but with the `&mut Kernel`
/// requirement gone so later steps can peel individual subsystems onto their own
/// locks.
#[allow(clippy::struct_excessive_bools)] // independent one-shot flags, not a state enum
pub struct Kernel {
    arch: Arch,
    /// Interactive mode (the browser terminal): guest reads of fd 0 draw from
    /// `Shared::stdin_buf` and *block* (re-trap) when it is empty rather than
    /// hitting the host `stdin`, so the embedder can pump input in between runs.
    interactive: bool,
    /// When set, the guest's stdio (fds 0/1/2) is the host process's own stdio,
    /// so terminal ioctls (`TCGETS`, `TIOCGWINSZ`, …) are forwarded to the real
    /// host tty — giving the guest an accurate virtual terminal (size, raw mode,
    /// echo). Cleared for paths that redirect stdio into a capture sink, where
    /// the host tty is unrelated to where the guest's output actually goes.
    host_tty: bool,
    trace: bool,
    /// Debug (`NIXVM_SCHEDTRACE`): log every scheduler slice (pid, syscalls run,
    /// how it ended) to see how threads interleave.
    schedtrace: bool,
    /// Preemption quantum: end a running task's slice after this many serviced
    /// syscalls even if it never blocks, so no task monopolizes the single CPU
    /// (a busy-waiting thread otherwise starves the workers it is waiting on).
    /// Tunable via `NIXVM_SLICE`; 0 disables preemption (old run-until-block).
    slice_cap: u32,
    /// Seed template for the initial process (pid 1): the pre-boot setters
    /// ([`Kernel::set_heap`]/[`Kernel::set_mmap_area`]/[`Kernel::set_cwd`])
    /// stash the first task's `ProcInfo` here, and [`Kernel::run`]/[`Kernel::boot`]
    /// `take` it to build pid 1. It is never touched during servicing — the
    /// running task's mutable state lives in a passed-in [`ServiceCtx`], not on
    /// the kernel, so several tasks can be serviced re-entrantly once the kernel
    /// lock is split (a later phase).
    seed: ProcInfo,
    /// Number of virtual CPUs: how many host worker threads run guest compute
    /// in parallel. `1` uses the single-threaded cooperative scheduler.
    ncpus: usize,
    /// Every field mutated during syscall servicing, behind the coarse kernel
    /// lock. Servicing acquires this once per service step and holds it for the
    /// whole step — the behavior-preserving big kernel lock.
    shared: Mutex<Shared>,
    /// The filesystem (`mounts`), peeled out of [`Shared`] onto its own lock
    /// (step B2) so a slow fstool/disk read holds only this lock and other
    /// tasks' non-FS syscalls run concurrently on `shared` instead of stalling
    /// on the big lock. **Lock order is strict and inviolable: `shared` (sh) is
    /// ALWAYS acquired BEFORE `vfs`; a `vfs` guard is NEVER held while acquiring
    /// `shared`.** Two locks in a consistent order ⇒ no deadlock cycle. The
    /// per-space memory lock stays outermost (memory → sh → vfs → net → pipes).
    vfs: Mutex<MountTable>,
    /// The network subsystem (`net`), peeled out of the coarse [`Shared`] lock
    /// onto its own sibling lock (step B3) so socket I/O holds only this lock
    /// and other tasks' non-socket syscalls run concurrently on `shared`
    /// instead of stalling on the big lock. **The order is strict and
    /// inviolable — memory → sh → vfs → net → pipes. A `net` guard is NEVER held
    /// while acquiring `shared` or `vfs`.** Handlers that need net plus others
    /// acquire in order sh → vfs → net.
    net: Mutex<Net>,
    /// The pipe subsystem (`pipes`), peeled out of the coarse [`Shared`] lock
    /// onto its own sibling lock (step B4) so a pipe read/write holds only this
    /// lock and other tasks' non-pipe syscalls run concurrently on `shared`
    /// instead of stalling on the big lock. **`pipes` is the innermost/LAST
    /// lock: the order is strict and inviolable — memory → sh → vfs → net →
    /// pipes. A `pipes` guard is NEVER held while acquiring `shared`, `vfs`, or
    /// `net`.** Handlers that need pipes plus others acquire in order
    /// sh → (vfs) → net → pipes.
    pipes: Mutex<Vec<Pipe>>,
    /// The poll/event subsystem (`eventfds`/`timerfds`/`epolls`), peeled out of
    /// the coarse [`Shared`] lock onto its own sibling lock (step B5) so the
    /// event-fd / epoll syscalls hold only this lock and other tasks' unrelated
    /// syscalls run concurrently on `shared` instead of stalling on the big
    /// lock. **`pollfds` is now the innermost/LAST lock: the order is strict and
    /// inviolable — memory → sh → vfs → net → pipes → pollfds. A `pollfds` guard
    /// is NEVER held while acquiring `shared`, `vfs`, `net`, or `pipes`.**
    /// Handlers that need pollfds plus others acquire in order
    /// sh → (vfs) → net → pipes → pollfds. The three tables are grouped behind
    /// one lock because the poll/select/epoll readiness scan touches them as a
    /// unit (see [`PollFds`]).
    pollfds: Mutex<PollFds>,
    /// Pseudo-terminals (/dev/ptmx + /dev/pts/N). Innermost data lock like
    /// `pipes`; opened/read/written/polled independently of `sh`.
    ptys: Mutex<pty::Ptys>,
    /// Subcommands of *known* syscalls that no handler recognized — an
    /// `ioctl` request, `fcntl`/`prctl` command, socket option, … — keyed by
    /// `(syscall name, subcommand)` with a hit count. The sibling of
    /// [`Shared::unsupported`] one level down: a syscall can be decoded and
    /// handled yet still be asked for something nobody implemented, and this is
    /// what makes such gaps visible after a run ([`Kernel::unsupported_subcommands`]).
    /// A leaf lock: nothing else is ever acquired while it is held, so any
    /// handler may record into it whatever locks it already holds.
    unsupported_sub: Mutex<BTreeMap<(&'static str, u64), u64>>,
    /// The page cache behind `MAP_SHARED` file mappings (see [`pagecache`]).
    /// A leaf lock taken after `vfs`; nothing is acquired while it is held.
    page_cache: Mutex<pagecache::PageCache>,
    /// Open-but-unlinked files, by their hidden path (see [`orphan`]). A leaf
    /// lock; the counter mints the hidden names.
    orphans: Mutex<BTreeSet<String>>,
    orphan_seq: AtomicU64,
    /// Record/OFD/`flock` locks and memfd seals (see [`fcntl`]). A leaf lock.
    locks: Mutex<fcntl::FileLocks>,
}

/// All kernel state mutated while a syscall is serviced, behind [`Kernel`]'s
/// coarse lock. Servicing takes `&self` (the config) plus `&mut Shared` (this),
/// so a field here is reached as `sh.<field>` instead of `self.<field>`.
#[allow(clippy::struct_excessive_bools)] // independent one-shot flags, not a state enum
pub(super) struct Shared {
    stdin: Box<dyn Read + Send>,
    stdout: Box<dyn Write + Send>,
    stderr: Box<dyn Write + Send>,
    /// Buffered terminal input for interactive mode (see [`Kernel::feed_stdin`]).
    stdin_buf: VecDeque<u8>,
    /// Whether interactive stdin has been closed (EOF / Ctrl-D).
    stdin_closed: bool,
    /// The last interactive-stdin read found the buffer empty and parked: the
    /// guest is waiting for the user (see [`Kernel::awaiting_input`]).
    stdin_waiting: bool,
    rng_state: u64,
    /// The tracked `RLIMIT_NOFILE` `(soft, hard)`. Programs (node/V8) binary-
    /// search `setrlimit` to raise it to the maximum, then loop over `[0,
    /// soft)` marking fds cloexec — so the hard cap must be *bounded* or that
    /// loop runs to `1<<20`. A `setrlimit` that always "succeeds" made node
    /// conclude it could raise the limit to a million fds and spin there.
    rlimit_nofile: (u64, u64),
    /// Monotonic counter for `memfd_create` backing-file names.
    memfd_seq: u64,
    /// The process file-creation mask (`umask`); global for our single session.
    umask: u32,
    unsupported: BTreeMap<u64, u64>,
    /// All tasks; the running one is `take`n out during its slice, so its
    /// slot is temporarily `None` (making `fork`/`wait4` on the table clean).
    procs: Vec<Option<Process>>,
    /// Address-space table indexed by `ProcInfo::mm`, each behind its own lock
    /// so a task's guest memory can be handed to an SMP worker thread while the
    /// main thread keeps servicing other tasks' syscalls. Threads that share
    /// memory (`CLONE_VM`) share one `Arc`; the per-space `Mutex` serializes
    /// access between a worker running compute and the main thread servicing a
    /// syscall against the same address space.
    spaces: Vec<Arc<Mutex<GuestMemory>>>,
    /// File-descriptor tables indexed by [`ProcInfo::files`]. A `CLONE_FILES`
    /// thread group shares one entry; a forked child gets its own. The slot is
    /// `None` while its owning task is mid-slice (the table is checked out into
    /// [`ProcInfo::fds`]); see [`Shared::check_out_files`].
    file_tables: Vec<Option<FdTable>>,
    /// Working directories indexed by [`ProcInfo::fs`]. A `CLONE_FS` group (every
    /// pthread) shares one entry so a `chdir` is seen groupwide; a forked child
    /// gets its own. Checked out into [`ProcInfo::cwd`] while the owning task runs
    /// (slot `None` meanwhile), alongside the fd table — see
    /// [`Shared::check_out_files`].
    cwd_tables: Vec<Option<String>>,
    /// Anonymous-`mmap` arenas indexed by [`ProcInfo::mm`] — one per address
    /// space, so every `CLONE_VM` thread allocates from the same arena and two
    /// threads can never be handed overlapping ranges.
    mmap_areas: Vec<Arena>,
    next_pid: i32,
    /// Fair-scheduling floor: the monotonic minimum weighted virtual runtime of
    /// the run queue (CFS's `min_vruntime`). Advanced to the vruntime of each task
    /// as it is picked; a task about to run is clamped *up* to this floor first,
    /// so a task that was blocked for a long time (its vruntime frozen while
    /// parked) rejoins at the current front of the queue instead of monopolizing
    /// the CPU until its stale vruntime catches up. See [`ProcInfo::vruntime`].
    min_vruntime: u128,
    /// `NIXVM_WATCHCODE` debug watch: address whose 8 bytes are checked after
    /// every syscall, and the last value seen there.
    watch_addr: Option<u64>,
    watch_last: u64,
    /// `sethostname`/`setdomainname`: what `uname` reports.
    hostname: String,
    domainname: String,
    /// `mseal`ed ranges: `(mm, start, end)`.
    sealed: Vec<(usize, u64, u64)>,
    /// `membarrier` registrations per address space (`MEMBARRIER_CMD_REGISTER_*`
    /// bits).
    membarrier: BTreeMap<usize, u64>,
    /// The System V IPC namespace (message queues, semaphores, shared memory).
    ipc: ipc::Ipc,
}

// The SMP scheduler ([`Kernel::schedule_smp`]) shares `&Kernel` across its worker
// threads and services each guest's syscall in place under the coarse kernel lock
// (`Kernel::shared`, the "big kernel lock") instead of shipping every exit to a
// central servicer thread. That requires `Kernel: Send + Sync` — `Sync` because
// the workers borrow `&Kernel` concurrently. `Mutex<Shared>` is `Sync` when
// `Shared: Send`, and the config fields are `Sync`, so both hold; assert them
// here so a future non-`Send`/`Sync` field breaks the build at its source rather
// than deep inside `schedule_smp`'s `thread::scope`.
const _: fn() = || {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<Kernel>();
    assert_sync::<Kernel>();
};

impl std::fmt::Debug for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut d = f.debug_struct("Kernel");
        d.field("arch", &self.arch);
        // Non-deadlocking: a `try_lock` failure just omits the shared counts.
        if let Ok(sh) = self.shared.try_lock() {
            d.field("procs", &sh.procs.len());
            d.field("unsupported", &sh.unsupported);
        }
        // Non-deadlocking (and order-safe): `try_lock` never blocks, so probing
        // `vfs` here can't violate the sh→vfs order even though sh may be held.
        if let Ok(vfs) = self.vfs.try_lock() {
            d.field("mounts", &*vfs);
        }
        d.finish_non_exhaustive()
    }
}

impl Shared {
    /// pid 1's exit code, if it has become a zombie.
    fn pid1_code(&self) -> Option<i32> {
        self.procs.iter().flatten().find_map(|p| match p.info.run {
            RunState::Zombie(c) if p.info.pid == 1 => Some(c.shell_code()),
            _ => None,
        })
    }

    fn any_running(&self) -> bool {
        self.procs
            .iter()
            .flatten()
            .any(|p| p.info.run == RunState::Running)
    }

    /// If any live task holds a timed-wait deadline, sleep the host thread until
    /// the earliest one (so the wall clock actually advances) and return `true`;
    /// the caller re-sweeps and the waiter, re-checking its now-passed deadline,
    /// returns "timed out". Returns `false` when nothing is timed — a genuine
    /// deadlock. This is what lets a fully-parked machine make `setTimeout`
    /// progress instead of being declared deadlocked.
    fn wait_for_timer(&self) -> bool {
        let now = poll::now_ns();
        let Some(dl) = self
            .procs
            .iter()
            .flatten()
            .filter(|p| p.info.run == RunState::Running)
            .flat_map(|p| [p.info.wake_deadline, p.info.timer_deadline()])
            .flatten()
            .min()
        else {
            return false;
        };
        if dl > now {
            let ns = (dl - now).min(3_600_000_000_000) as u64; // cap at 1h
            std::thread::sleep(std::time::Duration::from_nanos(ns));
        }
        true
    }

    /// True if every live, non-zombie task is parked (blocked). Used to break
    /// out of the unpark/re-sweep loop when a re-check produced no progress.
    fn everything_parked(&self) -> bool {
        let mut any_live = false;
        for p in self.procs.iter().flatten() {
            if p.info.run == RunState::Running {
                any_live = true;
                if !p.info.parked {
                    return false;
                }
            }
        }
        any_live
    }

    /// Wake parked tasks so they re-check their block condition on the next
    /// sweep. Called when the scheduler would otherwise stall — it catches
    /// wakeups that don't flow through an explicit unpark (a futex value that
    /// changed under a "lost" wake, host-socket data arriving, a child that
    /// became a zombie). Returns whether anything was parked (i.e. worth a
    /// re-sweep).
    fn unpark_all(&mut self) -> bool {
        let mut any = false;
        for p in self.procs.iter_mut().flatten() {
            if p.info.parked {
                p.info.parked = false;
                any = true;
            }
        }
        any
    }

    /// The earliest absolute wake deadline (ns since the epoch) held by any live
    /// task, or `None` if no task holds a timed wait — the SMP twin of
    /// [`Shared::wait_for_timer`]'s deadline scan.
    fn earliest_deadline(&self) -> Option<u128> {
        self.procs
            .iter()
            .flatten()
            .filter(|p| p.info.run == RunState::Running)
            .flat_map(|p| [p.info.wake_deadline, p.info.timer_deadline()])
            .flatten()
            .min()
    }

    /// Admit task `i` to the CPU: clamp its virtual runtime up to the run-queue
    /// floor ([`Shared::min_vruntime`]) and advance the floor to it. The clamp is
    /// what stops a task that sat blocked for a long time — its vruntime frozen
    /// while parked — from monopolizing the CPU on wake until its stale vruntime
    /// catches up; instead it rejoins at the front of the queue. Returns `i` for
    /// chaining. See [`ProcInfo::vruntime`].
    fn admit_fair(&mut self, i: usize) -> usize {
        let eff = self.procs[i]
            .as_ref()
            .unwrap()
            .info
            .vruntime
            .max(self.min_vruntime);
        self.procs[i].as_mut().unwrap().info.vruntime = eff;
        self.min_vruntime = eff;
        i
    }

    /// Pick the fairest runnable task for the serial scheduler / interactive
    /// pump: the `Running`, un-parked task holding its vcpu with the *least*
    /// virtual runtime (nice-weighted CPU consumed). Running one such task per
    /// call — the callers loop — gives least-vruntime-first, proportional-share
    /// scheduling instead of the old fixed pid-table order.
    fn pick_serial_runnable(&mut self) -> Option<usize> {
        let floor = self.min_vruntime;
        let best = (0..self.procs.len())
            .filter(|&i| {
                matches!(self.procs.get(i),
                    Some(Some(p)) if p.info.run == RunState::Running && !p.info.parked && p.vcpu.is_some())
            })
            .min_by_key(|&i| self.procs[i].as_ref().unwrap().info.vruntime.max(floor));
        best.map(|i| self.admit_fair(i))
    }

    /// Pick a runnable task for an SMP worker: `Running`, holding its vcpu (not
    /// already in flight), and not parked at the current progress epoch — and,
    /// among those, the one with the least virtual runtime. Choosing by vruntime
    /// (rather than the first free pid index) is what makes N CPU-bound processes
    /// on M<N workers share the cores fairly instead of the low-index ones
    /// starving the rest, and makes `nice` proportional here too.
    fn pick_smp_runnable(
        &mut self,
        blocked_at: &BTreeMap<usize, u64>,
        epoch: u64,
    ) -> Option<usize> {
        let floor = self.min_vruntime;
        let best = (0..self.procs.len())
            .filter(|&i| {
                let Some(Some(p)) = self.procs.get(i) else {
                    return false;
                };
                p.info.run == RunState::Running
                    && p.vcpu.is_some()
                    && blocked_at.get(&i).copied() != Some(epoch)
            })
            .min_by_key(|&i| self.procs[i].as_ref().unwrap().info.vruntime.max(floor));
        best.map(|i| self.admit_fair(i))
    }

    /// Check the running task's shared fd table out of [`Shared::file_tables`]
    /// into `cur.fds` — and its working directory out of [`Shared::cwd_tables`]
    /// into `cur.cwd` — for the duration of its slice. Called right after `cur`
    /// is swapped in. Its sibling threads (same `files`/`fs` id) are parked, so
    /// the slots are free; servicing is single-threaded, so no two tasks are ever
    /// checked out at once.
    fn check_out_files(&mut self, cx: &mut ServiceCtx) {
        let f = cx.cur.files;
        cx.cur.fds = self.file_tables[f]
            .take()
            .expect("fd table already checked out");
        let s = cx.cur.fs;
        cx.cur.cwd = self.cwd_tables[s].take().expect("cwd already checked out");
    }

    /// Check the running task's fd table back into [`Shared::file_tables`] and its
    /// cwd back into [`Shared::cwd_tables`] so its siblings see any changes it
    /// made (a `chdir` in a `CLONE_FS` group, an fd opened by a `CLONE_FILES`
    /// thread). Called right before `cur` is swapped out. If the task exited as
    /// the last user of its fd table, `cur.fds` was drained and we store the
    /// emptied table back (its slot is now idle).
    fn check_in_files(&mut self, cx: &mut ServiceCtx) {
        let f = cx.cur.files;
        self.file_tables[f] = Some(std::mem::take(&mut cx.cur.fds));
        let s = cx.cur.fs;
        self.cwd_tables[s] = Some(std::mem::take(&mut cx.cur.cwd));
    }

    /// The running task's `mmap` arena — the one shared by every task in its
    /// address space, so `CLONE_VM` siblings allocate from a single pool.
    fn arena(&mut self, cx: &mut ServiceCtx) -> &mut Arena {
        let mm = cx.cur.mm;
        &mut self.mmap_areas[mm]
    }
}

impl Kernel {
    #[must_use]
    pub fn new(arch: Arch, mounts: MountTable) -> Self {
        Self {
            arch,
            trace: std::env::var_os("NIXVM_TRACE").is_some(),
            schedtrace: std::env::var_os("NIXVM_SCHEDTRACE").is_some(),
            slice_cap: std::env::var("NIXVM_SLICE")
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(1024),
            seed: ProcInfo::default(),
            ncpus: 1,
            interactive: false,
            host_tty: false,
            vfs: Mutex::new(mounts),
            net: Mutex::new(Net::default()),
            pipes: Mutex::new(Vec::new()),
            pollfds: Mutex::new(PollFds::default()),
            ptys: Mutex::new(pty::Ptys::default()),
            unsupported_sub: Mutex::new(BTreeMap::new()),
            page_cache: Mutex::new(pagecache::PageCache::default()),
            orphans: Mutex::new(BTreeSet::new()),
            orphan_seq: AtomicU64::new(0),
            locks: Mutex::new(fcntl::FileLocks::default()),
            shared: Mutex::new(Shared {
                stdin: Box::new(std::io::stdin()),
                stdout: Box::new(std::io::stdout()),
                stderr: Box::new(std::io::stderr()),
                rng_state: 0,
                rlimit_nofile: (1024, 4096),
                memfd_seq: 0,
                umask: 0o022,
                unsupported: BTreeMap::new(),
                procs: Vec::new(),
                min_vruntime: 0,
                spaces: Vec::new(),
                file_tables: Vec::new(),
                cwd_tables: Vec::new(),
                mmap_areas: Vec::new(),
                stdin_buf: VecDeque::new(),
                stdin_closed: false,
                stdin_waiting: false,
                next_pid: 2,
                watch_addr: std::env::var("NIXVM_WATCHCODE")
                    .ok()
                    .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok()),
                watch_last: 0,
                hostname: "nixvm".to_string(),
                domainname: "(none)".to_string(),
                sealed: Vec::new(),
                membarrier: BTreeMap::new(),
                ipc: ipc::Ipc::default(),
            }),
        }
    }

    /// Redirect the sink backing guest fd 1 (`stdout`).
    pub fn set_stdout(&mut self, w: Box<dyn Write + Send>) {
        self.shared.get_mut().unwrap().stdout = w;
    }
    /// Redirect the sink backing guest fd 2 (`stderr`).
    pub fn set_stderr(&mut self, w: Box<dyn Write + Send>) {
        self.shared.get_mut().unwrap().stderr = w;
    }
    /// Redirect the source backing guest fd 0 (`stdin`).
    pub fn set_stdin(&mut self, r: Box<dyn Read + Send>) {
        self.shared.get_mut().unwrap().stdin = r;
    }

    /// Install a host-network egress backend: guest `connect`s to routable
    /// addresses (and UDP/DNS) are bridged onto real host sockets, so
    /// `apk`/`curl`/`npm` reach the internet. Without this the network is
    /// loopback-only. See [`crate::kernel::egress`].
    pub fn set_egress(&mut self, egress: Box<dyn egress::Egress>) {
        self.net.get_mut().unwrap().set_egress(egress);
    }

    /// Set the initial heap window for the first process: `start` is the program
    /// break, `limit` the highest address the heap may reach.
    pub fn set_heap(&mut self, start: u64, limit: u64) {
        self.seed.heap_start = start;
        self.seed.brk = start;
        self.seed.heap_limit = limit;
    }

    /// Set the initial anonymous-`mmap` arena for the first process. `top` is
    /// the initial stack's low bound; the arena is placed a guard gap below it
    /// so an unmapped region separates the stack from any `mmap` — see
    /// `STACK_GUARD_GAP`.
    pub fn set_mmap_area(&mut self, top: u64, floor: u64) {
        self.seed.stack_limit = top; // `top` is the stack's growth floor
        self.seed.mmap_cursor = arena_top(top, floor);
        self.seed.mmap_floor = floor;
    }

    /// Set the first process's current working directory.
    pub fn set_cwd(&mut self, dir: impl Into<String>) {
        self.seed.cwd = path::normalize(&dir.into());
    }

    /// Set the first process's program path (for `/proc/self/exe`); later
    /// `execve`s update it themselves.
    pub fn set_exe(&mut self, path: impl Into<String>) {
        self.seed.exe = path.into();
    }

    /// Set pid 1's launch `argv`, backing `/proc/self/cmdline` and the initial
    /// `comm` (`argv[0]`'s basename). Later `execve`s refresh both themselves.
    pub fn set_cmdline<I, S>(&mut self, argv: I)
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let argv: Vec<String> = argv.into_iter().map(Into::into).collect();
        if let Some(argv0) = argv.first() {
            self.seed.comm = comm_from_path(argv0);
        }
        self.seed.cmdline = cmdline_bytes(&argv);
    }

    /// Set the number of virtual CPUs (host worker threads that run guest
    /// compute in parallel). `0` is treated as `1`. With more than one CPU the
    /// SMP scheduler runs; guest compute for independent tasks proceeds on
    /// separate host threads while syscalls are serviced serially on the main
    /// thread (a big-kernel-lock model that maps cleanly onto KVM/HVF later).
    pub fn set_ncpus(&mut self, n: usize) {
        self.ncpus = n.max(1);
    }

    /// Run the machine: `vcpu`/`mem` become the initial process (pid 1), then
    /// the scheduler drives all processes until pid 1 exits. Returns pid 1's
    /// exit code.
    pub fn run(&mut self, vcpu: Box<dyn Vcpu>, mem: GuestMemory) -> Result<i32, VcpuError> {
        let ncpus = self.ncpus;
        let mut info = std::mem::take(&mut self.seed);
        {
            let sh = self.shared.get_mut().unwrap();
            info.pid = 1;
            info.ppid = 0;
            info.tgid = 1;
            info.mm = sh.spaces.len();
            info.run = RunState::Running;
            info.files = sh.file_tables.len();
            sh.file_tables.push(Some(std::mem::take(&mut info.fds)));
            info.fs = sh.cwd_tables.len();
            sh.cwd_tables.push(Some(std::mem::take(&mut info.cwd)));
            sh.mmap_areas
                .push(Arena::new(info.mmap_cursor, info.mmap_floor));
            sh.spaces.push(Arc::new(Mutex::new(mem)));
            sh.procs.push(Some(Process {
                vcpu: Some(vcpu),
                info,
            }));
        }
        if ncpus > 1 {
            self.schedule_smp()
        } else {
            self.schedule_serial()
        }
    }

    /// Cooperative single-CPU round-robin scheduler.
    fn schedule_serial(&self) -> Result<i32, VcpuError> {
        loop {
            if let Some(code) = self.shared.lock().unwrap().pid1_code() {
                return Ok(code);
            }
            if self.serial_sweep()? {
                continue;
            }
            // No runnable task made progress. Wake the parked tasks so they
            // re-check their conditions (a futex value that changed under a
            // lost wake, a child that exited, host I/O). If nothing was parked
            // to re-check, it's a genuine deadlock.
            if !self.shared.lock().unwrap().unpark_all() {
                let sh = self.shared.lock().unwrap();
                if sh.any_running() {
                    return Err(VcpuError::Backend(
                        "deadlock: every process is blocked".into(),
                    ));
                }
                return Ok(sh.pid1_code().unwrap_or(0));
            }
            // Re-sweep the just-unparked tasks. If they all immediately re-park
            // without progress, either a timer is pending (sleep until it, then
            // the re-run of that task's wait sees its deadline passed and
            // returns) or it's a genuine deadlock.
            if !self.serial_sweep()? && self.shared.lock().unwrap().everything_parked() {
                if self.shared.lock().unwrap().wait_for_timer() {
                    continue;
                }
                // A live host connection may still deliver data asynchronously;
                // poll for it (short sleep + re-sweep) rather than declaring a
                // deadlock. Only a machine with no timer and no host I/O pending
                // is genuinely stuck.
                if self.net.lock().unwrap().has_pending_host_io() {
                    std::thread::sleep(std::time::Duration::from_nanos(HOST_IO_POLL_NS as u64));
                    continue;
                }
                return Err(VcpuError::Backend(
                    "deadlock: every process is blocked".into(),
                ));
            }
        }
    }

    /// Run the fairest runnable task that makes progress — one slice — on the
    /// current thread. "Fairest" is least virtual runtime (nice-weighted CPU),
    /// so calling this in a loop (as [`Kernel::schedule_serial`] and the
    /// interactive [`Kernel::pump`] both do) gives least-vruntime-first,
    /// proportional-share scheduling rather than the old fixed pid-table order.
    /// Returns whether a task made progress; `false` means nothing was runnable
    /// (or every runnable task blocked immediately without progress), so the
    /// caller runs its unpark / timer / deadlock logic.
    ///
    /// A task that blocks immediately with no progress is parked and skipped
    /// past *within* this call, so one poll of an unready fd doesn't cost a whole
    /// unpark round; as soon as one task makes progress we return, so the caller
    /// re-picks by vruntime for the next slice.
    ///
    /// The coarse kernel lock ([`Kernel::shared`]) is taken only for the
    /// check-out/check-in bookkeeping; the slice itself runs holding just the
    /// per-address-space memory lock (outermost), so a syscall's own per-handler
    /// locks nest correctly under it.
    fn serial_sweep(&self) -> Result<bool, VcpuError> {
        let mut progressed = false;
        loop {
            // The embedder's time budget (`pump_for`) is spent: hand back.
            if crate::vcpu::yield_due() {
                return Ok(progressed);
            }
            // Pick the least-vruntime runnable task and check it out (slot → `None`,
            // fd table into `cx`), releasing `sh` before running the slice so the
            // slice's syscalls can take their own per-handler locks.
            let Some((i, mut proc, mut vcpu, space_arc, mut cx)) = ({
                let mut sh = self.shared.lock().unwrap();
                sh.pick_serial_runnable().map(|i| {
                    let mut proc = sh.procs[i].take().unwrap();
                    let mm = proc.info.mm;
                    let vcpu = proc.vcpu.take().expect("runnable task has a vcpu");
                    let space_arc = Arc::clone(&sh.spaces[mm]);
                    let mut cx = ServiceCtx {
                        cur: std::mem::take(&mut proc.info),
                        ..ServiceCtx::default()
                    };
                    sh.check_out_files(&mut cx);
                    (i, proc, vcpu, space_arc, cx)
                })
            }) else {
                return Ok(progressed);
            };
            let mut guard = space_arc.lock().unwrap();
            let made = self.run_slice(&mut cx, &mut vcpu, &mut guard)?;
            if self.schedtrace {
                let end = if matches!(cx.cur.run, RunState::Zombie(_)) {
                    "ended"
                } else if cx.block {
                    "blocked"
                } else {
                    "yield"
                };
                eprintln!(
                    "[sched] pid={} slot={} vr={} syscalls={} end={end}",
                    cx.cur.pid, i, cx.cur.vruntime, cx.slice_syscalls
                );
            }
            // The slice ended by exiting, yielding, preemption, or blocking;
            // `cx.block` reflects the last syscall. A blocked task parks.
            let blocked = cx.block;
            drop(guard); // memory is outermost — drop before re-taking `sh`
            {
                let mut sh = self.shared.lock().unwrap();
                sh.check_in_files(&mut cx);
                proc.info = cx.cur;
                proc.info.parked = blocked && proc.info.run == RunState::Running;
                proc.vcpu = Some(vcpu);
                sh.procs[i] = Some(proc);
            }
            progressed |= made;
            // Made progress → hand control back so the caller re-picks fairly.
            // No progress but the task blocked → try the next fairest task in
            // this same call (it is now parked, so it won't be re-picked). No
            // progress and didn't block → nothing more to do (avoids a busy spin
            // re-picking the same task).
            if made || !blocked {
                return Ok(progressed);
            }
        }
    }

    // ---- interactive driver (the browser terminal) -----------------------

    /// Enable interactive mode: guest reads of fd 0 draw from the buffer fed via
    /// [`Kernel::feed_stdin`] and block when empty, instead of the host stdin.
    /// Mark the guest's stdio as the host process's own, so terminal ioctls are
    /// forwarded to the real host tty (see `Kernel::host_tty`). The `nixvm run`
    /// CLI sets this; capture/redirect paths leave it clear.
    pub fn set_host_tty(&mut self, yes: bool) {
        self.host_tty = yes;
    }

    pub fn set_interactive(&mut self, yes: bool) {
        self.interactive = yes;
    }

    /// Append bytes to the interactive terminal-input buffer (keystrokes).
    pub fn feed_stdin(&mut self, bytes: &[u8]) {
        self.shared
            .get_mut()
            .unwrap()
            .stdin_buf
            .extend(bytes.iter().copied());
    }

    /// Signal end-of-input on the interactive stdin (Ctrl-D).
    pub fn close_stdin(&mut self) {
        self.shared.get_mut().unwrap().stdin_closed = true;
    }

    /// Ctrl-C on the interactive terminal: post `SIGINT` to the running
    /// command — every live process except the session's shell (pid 1). The
    /// terminal here is not a tty with a line discipline, so a `^C` byte in
    /// stdin would just be data; and the demo's `sh` runs without job control
    /// (one process group), where a real interactive shell ignores `SIGINT`
    /// while its foreground command receives it. Background jobs started with
    /// `&` have `SIGINT` ignored by the shell, so they carry on. Returns
    /// whether any process was signalled (i.e. a command was running).
    pub fn interrupt(&mut self) -> bool {
        const SIGINT_BIT: u64 = 1 << (2 - 1);
        let mut any = false;
        for p in self.shared.get_mut().unwrap().procs.iter_mut().flatten() {
            if p.info.pid != 1 && p.info.run == RunState::Running {
                p.info.pending |= SIGINT_BIT;
                p.info.parked = false;
                any = true;
            }
        }
        any
    }

    /// Whether the guest is parked reading the interactive terminal with
    /// nothing buffered — i.e. the command the user typed has finished and the
    /// shell wants the next line. A [`Pumped::Blocked`] that is *not* this is
    /// the guest waiting on something else (a timer, the network), and the
    /// embedder should pump again later without prompting.
    #[must_use]
    pub fn awaiting_input(&self) -> bool {
        let sh = self.shared.lock().unwrap();
        sh.stdin_waiting && sh.stdin_buf.is_empty() && !sh.stdin_closed
    }

    /// Whether a parked guest will make progress without new input: some task
    /// holds a timed wait, or a host-bridged socket may deliver data. The
    /// single-threaded embedder (the browser) re-pumps on a timer while this
    /// holds, since nothing wakes the cooperative loop from outside.
    #[must_use]
    pub fn has_pending_work(&self) -> bool {
        self.shared.lock().unwrap().earliest_deadline().is_some()
            || self.net.lock().unwrap().has_pending_host_io()
    }

    /// Seed the initial process (pid 1) without running it, for the incremental
    /// [`Kernel::pump`] driver. Use instead of [`Kernel::run`] when the embedder
    /// wants to interleave guest execution with feeding input (e.g. a terminal).
    pub fn boot(&mut self, vcpu: Box<dyn Vcpu>, mem: GuestMemory) {
        let mut info = std::mem::take(&mut self.seed);
        let sh = self.shared.get_mut().unwrap();
        info.pid = 1;
        info.ppid = 0;
        info.tgid = 1;
        info.mm = sh.spaces.len();
        info.run = RunState::Running;
        info.auxv = crate::loader::read_auxv(&mem, vcpu.sp());
        // Check the initial fd table (the standard streams) into slot 0; the
        // scheduler checks it out into `cur.fds` for each slice.
        info.files = sh.file_tables.len();
        sh.file_tables.push(Some(std::mem::take(&mut info.fds)));
        info.fs = sh.cwd_tables.len();
        sh.cwd_tables.push(Some(std::mem::take(&mut info.cwd)));
        sh.mmap_areas
            .push(Arena::new(info.mmap_cursor, info.mmap_floor));
        sh.spaces.push(Arc::new(Mutex::new(mem)));
        sh.procs.push(Some(Process {
            vcpu: Some(vcpu),
            info,
        }));
    }

    /// Drive the (single-CPU) machine until pid 1 exits or every task is parked
    /// waiting for input. Call after [`Kernel::boot`], re-calling after each
    /// [`Kernel::feed_stdin`] to resume. Unlike [`Kernel::run`], a full sweep
    /// with no progress is reported as [`Pumped::Blocked`] (needs input), not a
    /// deadlock error.
    pub fn pump(&self) -> Result<Pumped, VcpuError> {
        self.pump_inner()
    }

    /// [`Kernel::pump`] with a time budget: return [`Pumped::Busy`] once
    /// `budget` has elapsed even though the guest could keep running, so a
    /// single-threaded embedder (the browser tab, whose UI, timers and
    /// WebSocket all share the thread) is never frozen by a compute-heavy
    /// guest. The running vcpu notices the deadline within a few thousand
    /// instructions; its task stays runnable and resumes on the next call.
    pub fn pump_for(&self, budget: std::time::Duration) -> Result<Pumped, VcpuError> {
        crate::vcpu::set_yield_deadline(Some(crate::clock::now_monotonic() + budget));
        let r = self.pump_inner();
        crate::vcpu::set_yield_deadline(None);
        r
    }

    fn pump_inner(&self) -> Result<Pumped, VcpuError> {
        loop {
            if let Some(code) = self.shared.lock().unwrap().pid1_code() {
                return Ok(Pumped::Exited(code));
            }
            if crate::vcpu::yield_due() {
                let sh = self.shared.lock().unwrap();
                return Ok(if sh.any_running() {
                    Pumped::Busy
                } else {
                    Pumped::Exited(sh.pid1_code().unwrap_or(0))
                });
            }
            if self.serial_sweep()? {
                continue;
            }
            // Stalled. Re-check parked tasks once (catches lost futex wakes,
            // host-socket data, child exits). If the re-check makes progress,
            // keep going; otherwise the machine is genuinely parked — for the
            // interactive driver that means "waiting for input" (the embedder
            // feeds stdin / host I/O completes and re-pumps), not a deadlock.
            if self.shared.lock().unwrap().unpark_all() && self.serial_sweep()? {
                continue;
            }
            // Genuinely parked. A task holding a timed-wait deadline (setTimeout
            // → epoll_pwait) isn't waiting for input — it just needs the wall
            // clock to advance. We don't sleep here (this drives the single-
            // threaded wasm terminal too), so the embedder must re-pump; each
            // re-pump re-checks the deadline and fires the timer once it passes.
            let sh = self.shared.lock().unwrap();
            return Ok(if sh.any_running() {
                Pumped::Blocked
            } else {
                Pumped::Exited(sh.pid1_code().unwrap_or(0))
            });
        }
    }

    /// Run one process until it blocks or exits. Returns whether it made
    /// progress (completed at least one syscall, or exited).
    fn run_slice(
        &self,
        cx: &mut ServiceCtx,
        vcpu: &mut Box<dyn Vcpu>,
        mem: &mut GuestMemory,
    ) -> Result<bool, VcpuError> {
        let mut progressed = false;
        loop {
            // Charge this step's wall time (guest execution + syscall servicing)
            // to the task's CPU total — a blocked task ends its slice here and
            // stops accruing, so this tracks CPU rather than wall time.
            let step_start = crate::clock::now_monotonic().as_nanos();
            let exit = vcpu.run(mem)?;
            // Fire a due ITIMER_REAL / POSIX timer before servicing, so a
            // blocking syscall this step sees its signal pending and is
            // interrupted.
            if cx.cur.alarm_deadline.is_some() || !cx.cur.ptimers.is_empty() {
                fire_timers_if_due(&mut cx.cur, poll::now_ns());
            }
            let flow = self.service(cx, exit, vcpu.as_mut(), mem);
            let delta = crate::clock::now_monotonic()
                .as_nanos()
                .saturating_sub(step_start);
            cx.cur.cpu_ns = cx.cur.cpu_ns.saturating_add(delta);
            charge_vruntime(&mut cx.cur, delta);
            match flow {
                Serviced::SetRet => {
                    // The result was already written to the vcpu inside `service`
                    // (before signal delivery, so an interrupted syscall's frame
                    // captures it). Re-writing it here would call the backend's
                    // `set_syscall_ret` twice — harmless for KVM but a double pc
                    // advance for the interpreter (it steps past the 2-byte
                    // `syscall`), drifting into the middle of the next instruction.
                    progressed = true;
                    // `sched_yield`: the call succeeded but ends the slice so
                    // siblings run. The task is *not* parked — `cx.block` stays clear.
                    if cx.yield_now {
                        cx.yield_now = false;
                        return Ok(true);
                    }
                }
                Serviced::Resume => progressed = true,
                Serviced::Blocked => return Ok(progressed),
                Serviced::Ended => return Ok(true),
            }
            // Preemption: after a full quantum of syscalls, end the slice so a
            // sibling can run even though this task never blocked. This keeps a
            // busy-waiting thread from monopolizing the single CPU while the
            // worker it is spinning on starves. The task stays runnable (not
            // parked) — `cx.block` is clear — so the next sweep resumes it.
            if self.slice_cap != 0 && cx.slice_syscalls >= self.slice_cap {
                return Ok(progressed);
            }
            // The embedder's time budget (`pump_for`) is spent: end the slice
            // with the task still runnable.
            if crate::vcpu::yield_due() {
                return Ok(progressed);
            }
        }
    }

    /// Service one guest exit against the current task (`self.cur`): dispatch a
    /// syscall, or turn a fault/halt into a zombie. Shared by the serial and
    /// SMP schedulers. Does NOT touch the vcpu's result register — the caller
    /// applies [`Serviced::SetRet`] — so the same logic works whether the vcpu
    /// lives on the main thread or is round-tripping through a worker.
    fn service(
        &self,
        cx: &mut ServiceCtx,
        exit: Exit,
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> Serviced {
        match exit {
            Exit::Syscall => {
                let raw = vcpu.syscall_nr();
                let sys = arch::decode(self.arch, raw);
                let args = vcpu.syscall_args();
                cx.slice_syscalls = cx.slice_syscalls.saturating_add(1);
                cx.block = false;
                cx.exec_ok = false;
                cx.restartable = true; // syscalls are restartable unless they opt out
                cx.restart_syscall = false;
                // No lock is held here: `dispatch` acquires exactly the lock(s)
                // each handler needs (sh before vfs). This is what lets other
                // workers service their own syscalls concurrently (step B2).
                // seccomp filters see every syscall before it runs.
                let verdict = if cx.cur.seccomp.active() {
                    self.seccomp_check(cx, sys, raw, &args, vcpu.pc(), mem)
                } else {
                    None
                };
                let mut ret = match verdict {
                    Some(r) => r,
                    None => self.dispatch(cx, sys, raw, &args, vcpu, mem),
                };
                // A syscall that wants to block but has a signal with a real
                // handler pending must not park — POSIX requires it to either
                // restart after the handler (`SA_RESTART`) or fail with `-EINTR`,
                // and the handler must run now. (A default-*terminate* signal is
                // handled by the `Zombie` check below, which wins over `block`;
                // `SIG_IGN`/default-ignored signals correctly leave it blocked.)
                if cx.block
                    && let Some(sig) = self.first_handled_signal(cx)
                {
                    cx.block = false;
                    if cx.cur.handlers[sig].flags & SA_RESTART != 0 && cx.restartable {
                        // Re-run the `syscall` once the handler returns; the
                        // re-run re-establishes any wait bookkeeping (e.g. a
                        // futex's), so leave it in place.
                        cx.restart_syscall = true;
                    } else {
                        ret = err(Errno::EINTR);
                        // Abandoning the syscall (not restarting): drop the
                        // futex-wait bookkeeping so a later FUTEX_WAKE on the
                        // old address can't spuriously flag this now-running
                        // task. (`wake_deadline` is cleared below since the
                        // syscall no longer blocks.)
                        cx.cur.futex_wait = None;
                        cx.cur.futex_waitv.clear();
                        cx.cur.futex_pi = false;
                        cx.cur.futex_woken = false;
                    }
                }
                // Land the syscall's result in the vcpu *before* delivering any
                // pending signal: if this syscall is interrupted by a handler
                // (e.g. `sigsuspend` → `-EINTR`), the `rt_sigframe` must capture
                // the real result so `rt_sigreturn` restores it — otherwise the
                // interrupted syscall would resume with a stale return register.
                // Skipped when the task re-blocks (re-traps the same syscall), is
                // being restarted (RAX must keep the syscall number), or exec'd a
                // new image (resumes at its entry, no return value).
                if !cx.block && !cx.exec_ok && !cx.restart_syscall {
                    vcpu.set_syscall_ret(ret as u64);
                }
                let delivered = self.deliver_pending_signals(cx, vcpu, mem);
                // Flush the running vcpu's TLB if the page tables were edited in
                // place — so it can't keep serving a stale entry for a now-
                // unmapped, re-protected, or CoW-replaced page. This MUST run
                // AFTER `deliver_pending_signals`, not before: delivering a signal
                // writes the handler's `rt_sigframe` onto the guest stack, which
                // privatizes a copy-on-write-shared stack page (a fresh frame,
                // remapped). Flushing before delivery would miss that, and the
                // handler would then run with a stale TLB entry pointing at the
                // old (still-shared) frame — corrupting memory shared with a
                // concurrent sibling thread or a not-yet-exec'd forked child (only
                // visible under SMP, where such a sibling runs at the same time).
                // No-op for the interpreter (no TLB).
                if mem.take_tlb_dirty() {
                    vcpu.flush_tlb();
                }
                // A syscall that returns (didn't re-block) has consumed any
                // timed-wait deadline it set; the next blocking syscall starts
                // a fresh one.
                if !cx.block {
                    cx.cur.wake_deadline = None;
                }
                self.watch_code(vcpu, mem, sys);
                if let RunState::Zombie(_) = cx.cur.run {
                    Serviced::Ended
                } else if cx.block {
                    Serviced::Blocked
                } else if cx.exec_ok {
                    Serviced::Resume // resume the new image at its entry
                } else if delivered {
                    // A handler was set up (pc/sp/regs redirected, and the return
                    // value already written above so its sigframe captured it):
                    // resume into the handler rather than re-applying the ret.
                    Serviced::Resume
                } else {
                    Serviced::SetRet
                }
            }
            Exit::Interrupted => Serviced::Resume,
            Exit::MemFault { addr, write } => {
                // A fault on a mapped-but-unbacked page is demand paging: mint the
                // frame and re-run the access (the software mirror of a hardware
                // MMU faulting in a lazily-committed page). Anonymous reservations
                // and freshly-`mmap`ped ranges are backed here on first touch.
                // Each of these resolutions edits this address space's page tables
                // in place (from the host, behind the running vcpu), so the vcpu's
                // TLB is flushed before it retries — otherwise a stale
                // write-protected (copy-on-write) entry would keep faulting or a
                // stale mapping would be used. `flush_tlb` is a no-op for the
                // interpreter (no TLB) and a not-present demand fault leaves no
                // stale entry, but flushing uniformly keeps the seam simple.
                //
                // A write fault on a copy-on-write page is resolved by
                // privatizing the page and re-running the instruction (the vcpu
                // left PC on the faulting store). Anything else — a read fault, a
                // write to read-only/unmapped memory, or an already-private page
                // — is a genuine segfault. This is the software mirror of a
                // hardware MMU's page-fault-driven COW.
                if mem.demand_fault(addr) || mem.cow_fault(addr, write) {
                    vcpu.flush_tlb();
                    Serviced::Resume
                } else if self.grow_stack(cx, addr, mem) {
                    // A fault in the reserved stack region grows it (VM_GROWSDOWN)
                    // and re-runs the faulting instruction.
                    vcpu.flush_tlb();
                    Serviced::Resume
                } else if vcpu.shadow_stale(mem, addr) {
                    // SMP/KVM only: a sibling mapped or re-protected this page
                    // (serviced here on the main thread) while this vcpu was mid
                    // run with shadow page tables synced at its last dispatch, so
                    // its hardware walk faulted on a page that is in fact
                    // accessible. Re-dispatch reconciles the tables and re-runs
                    // the faulting instruction. Never true for the interpreter or
                    // the serial path, which are always coherent with `mem`.
                    Serviced::Resume
                } else if self.deliver_fault_signal(
                    cx,
                    signal::Fault::segv(
                        self.arch,
                        addr,
                        write,
                        mem.page_prot(addr).is_some(),
                        addr == vcpu.pc(),
                    ),
                    vcpu,
                    mem,
                ) {
                    // The guest caught it (JIT trap handler): run the handler.
                    Serviced::Resume
                } else {
                    eprintln!(
                        "[fault] pid {} memory fault at {addr:#x} (write={write}, pc={:#x})",
                        cx.cur.pid,
                        vcpu.pc()
                    );
                    self.dump_fault_context(vcpu, mem);
                    self.die_of_signal(cx, SIGSEGV as u32, mem);
                    Serviced::Ended
                }
            }
            Exit::IllegalInstruction { pc } => {
                // Dump the raw bytes at the fault so an interpreter decode gap
                // is identifiable from the report alone (the pc is under a
                // load bias for PIEs/`ld-musl`, so it can't be looked up in
                // the on-disk ELF directly).
                if self.deliver_fault_signal(cx, signal::Fault::ill(self.arch, pc), vcpu, mem) {
                    return Serviced::Resume; // guest's SIGILL handler (JIT trap)
                }
                let bytes = mem.read_vec(pc, 16).unwrap_or_default();
                let hex: Vec<String> = bytes.iter().map(|b| format!("{b:02x}")).collect();
                self.dump_fault_context(vcpu, mem);
                eprintln!(
                    "[fault] pid {} illegal instruction at {pc:#x} [{}]",
                    cx.cur.pid,
                    hex.join(" ")
                );
                self.die_of_signal(cx, SIGILL as u32, mem);
                Serviced::Ended
            }
            Exit::Breakpoint { pc, .. } => {
                // `BRK`/`int3`: SIGTRAP (a debugger-less process dies of it with
                // a core, like Linux; a guest handler — Go, sanitizers — runs).
                if self.deliver_fault_signal(cx, signal::Fault::brk(self.arch, pc), vcpu, mem) {
                    return Serviced::Resume;
                }
                eprintln!("[fault] pid {} breakpoint trap at {pc:#x}", cx.cur.pid);
                self.dump_fault_context(vcpu, mem);
                self.die_of_signal(cx, SIGTRAP as u32, mem);
                Serviced::Ended
            }
            Exit::Misaligned { addr, write } => {
                let fault = signal::Fault::misaligned(self.arch, addr, write, vcpu.pc(), vcpu.sp());
                if self.deliver_fault_signal(cx, fault, vcpu, mem) {
                    return Serviced::Resume;
                }
                eprintln!(
                    "[fault] pid {} alignment fault at {addr:#x} (write={write}, pc={:#x})",
                    cx.cur.pid,
                    vcpu.pc()
                );
                self.dump_fault_context(vcpu, mem);
                self.die_of_signal(cx, SIGBUS as u32, mem);
                Serviced::Ended
            }
            Exit::Halt => {
                cx.cur.run = RunState::Zombie(ExitCause::Exited(0));
                Serviced::Ended
            }
        }
    }

    /// SMP scheduler: a pool of `ncpus` host worker threads run guest compute
    /// **and service their own syscalls in place**. Each worker runs its vcpu to
    /// an exit, then — under a single global "big kernel lock" — services that
    /// exit ([`run_slice_smp`] → [`Kernel::smp_service_step`]) and, while the
    /// task stays runnable, keeps running the *same* vcpu on the same thread.
    /// The scheduler main loop only dispatches slices to their home worker and,
    /// when a slice ends, parks/reaps/re-dispatches the task.
    ///
    /// # The lock model
    /// The whole `Kernel` (mounts, pipes, process table, scheduler bookkeeping)
    /// sits behind one `Mutex` — the *kernel lock*, held only while servicing a
    /// syscall or making a scheduling decision. Because exactly one thread holds
    /// it at a time, at most one syscall is serviced at once: big-kernel-lock
    /// semantics are preserved, so global kernel state is never touched
    /// concurrently and stays race-free. Guest compute runs with the kernel lock
    /// **not** held (KVM runs with *no* lock; the interpreter holds only the
    /// per-space memory lock), so vCPUs still execute in parallel.
    ///
    /// Two lock classes, always taken **memory lock → kernel lock** (never the
    /// reverse — see [`run_slice_smp`]): the per-space `Arc<Mutex<GuestMemory>>`
    /// and the kernel lock. The scheduler main loop only ever takes the kernel
    /// lock; servicing takes the memory lock first, then the kernel lock; a
    /// locked interpreter run and a KVM reconcile take the memory lock alone.
    /// The kernel lock is therefore always the last lock acquired, so no worker
    /// blocks on the memory lock while holding the kernel lock, and there is no
    /// lock cycle.
    ///
    /// vcpu→thread affinity (task `i` always runs on worker `i % nworkers`) is
    /// kept from the previous design: KVM penalizes running a vcpu from a
    /// rotating set of threads (a vcpu-migration cost measured at ~27 ms vs
    /// ~2 µs same-thread), so a task's vcpu returns to its home worker across
    /// slices. In-place servicing makes that automatic within a slice.
    #[allow(clippy::too_many_lines)] // the worker pool + scheduler loop reads best as one unit
    fn schedule_smp(&self) -> Result<i32, VcpuError> {
        // Work handed to a worker: run a slice for this vcpu on this address
        // space. `Stop` drains the pool at shutdown.
        enum Work {
            Run(usize, Box<dyn Vcpu>, Arc<Mutex<GuestMemory>>),
            Stop,
        }
        type Done = (usize, Box<dyn Vcpu>, SliceOutcome);

        let nworkers = self.ncpus;
        // `slice_cap` is fixed for the run; snapshot it so workers need no lock
        // to read it.
        let slice_cap = self.slice_cap;
        // One queue per worker (home affinity, see the doc comment).
        let queues: Vec<Arc<(Mutex<VecDeque<Work>>, Condvar)>> = (0..nworkers)
            .map(|_| Arc::new((Mutex::new(VecDeque::new()), Condvar::new())))
            .collect();
        let (done_tx, done_rx) = mpsc::channel::<Done>();

        // Share `&Kernel` across the workers; each services its own guest's
        // syscall in place under the coarse kernel lock (`self.shared`, the "big
        // kernel lock"). `Kernel: Sync` (asserted above) makes the shared borrow
        // sound; `thread::scope` joins every worker before the borrow of `self`
        // ends. This is the behavior-preserving replacement for the former
        // `Mutex<&mut Kernel>` — still exactly one lock, still one syscall at a
        // time — dropped so the `&mut Kernel` requirement is gone.
        let kernel: &Kernel = self;

        std::thread::scope(|scope| {
            for home in &queues {
                let q = Arc::clone(home);
                let out = done_tx.clone();
                scope.spawn(move || {
                    loop {
                        let work = {
                            let (lock, cv) = &*q;
                            let mut g = lock.lock().unwrap();
                            loop {
                                if let Some(w) = g.pop_front() {
                                    break w;
                                }
                                g = cv.wait(g).unwrap();
                            }
                        };
                        match work {
                            Work::Stop => break,
                            Work::Run(id, vcpu, space) => {
                                let done = run_slice_smp(kernel, slice_cap, id, vcpu, &space);
                                if out.send(done).is_err() {
                                    break;
                                }
                            }
                        }
                    }
                });
            }
            drop(done_tx);

            // Route a run to its home worker (`task index % nworkers`).
            let push_run = |i: usize, vcpu, space| {
                let (lock, cv) = &*queues[i % nworkers];
                lock.lock().unwrap().push_back(Work::Run(i, vcpu, space));
                cv.notify_one();
            };

            // A task that blocked records the progress epoch at which it did; it
            // is not re-dispatched until the epoch advances (some other slice
            // made real progress that might satisfy its wait) — avoiding a busy
            // spin. `stalled` guards the deadlock/timer path: it is cleared by
            // any real progress and set once we force a no-timer retry round, so
            // a genuinely deadlocked machine is detected after exactly one
            // fruitless retry instead of spinning or erroring prematurely.
            let mut blocked_at: BTreeMap<usize, u64> = BTreeMap::new();
            let mut epoch: u64 = 0;
            let mut inflight = 0usize;
            let mut stalled = false;
            let outcome = loop {
                // Fill idle workers with runnable tasks (kernel lock held only
                // for the dispatch decision, not while awaiting results).
                let dispatch = {
                    let mut sh = self.shared.lock().unwrap();
                    if let Some(code) = sh.pid1_code() {
                        break Ok(code);
                    }
                    let mut batch = Vec::new();
                    while inflight + batch.len() < nworkers {
                        let Some(i) = sh.pick_smp_runnable(&blocked_at, epoch) else {
                            break;
                        };
                        let mm = sh.procs[i].as_ref().unwrap().info.mm;
                        let space = Arc::clone(&sh.spaces[mm]);
                        let vcpu = sh.procs[i].as_mut().unwrap().vcpu.take().unwrap();
                        batch.push((i, vcpu, space));
                    }
                    batch
                };
                for (i, vcpu, space) in dispatch {
                    push_run(i, vcpu, space);
                    inflight += 1;
                }

                if inflight == 0 {
                    // Nothing runnable and nothing in flight. Decide under the
                    // kernel lock, mirroring the serial scheduler's stall logic.
                    let action = {
                        let sh = self.shared.lock().unwrap();
                        if !sh.any_running() {
                            break Ok(sh.pid1_code().unwrap_or(0));
                        }
                        // A pending timed wait (poll/epoll timeout, setTimeout)
                        // isn't a deadlock — it just needs the wall clock to
                        // advance. Sleep to the earliest deadline, then force a
                        // retry so the waiter re-checks its now-passed deadline.
                        if let Some(dl) = sh.earliest_deadline() {
                            StallAction::SleepUntil(dl)
                        } else if self.net.lock().unwrap().has_pending_host_io() {
                            // A live host connection may still deliver data
                            // asynchronously (an in-flight HTTP response). There
                            // is no host-side wakeup into this cooperative loop,
                            // so poll for it — a short sleep, then a retry round
                            // that re-checks socket readiness — rather than
                            // mistaking the wait for a deadlock.
                            StallAction::SleepUntil(poll::now_ns() + HOST_IO_POLL_NS)
                        } else if !stalled {
                            // No timer: catch a lost futex wake / a child that
                            // became a zombie with one forced retry round before
                            // declaring deadlock.
                            StallAction::Retry
                        } else {
                            StallAction::Deadlock
                        }
                    };
                    match action {
                        StallAction::SleepUntil(dl) => {
                            let now = poll::now_ns();
                            if dl > now {
                                let ns = (dl - now).min(3_600_000_000_000) as u64;
                                std::thread::sleep(std::time::Duration::from_nanos(ns));
                            }
                            stalled = false;
                            epoch += 1;
                        }
                        StallAction::Retry => {
                            stalled = true;
                            epoch += 1;
                        }
                        StallAction::Deadlock => {
                            break Err(VcpuError::Backend(
                                "deadlock: every task is blocked".into(),
                            ));
                        }
                    }
                    continue;
                }

                // Await one slice result (kernel lock released while we wait, so
                // other workers keep servicing).
                let (i, vcpu, out) = done_rx.recv().expect("workers outlive the scheduler");
                inflight -= 1;
                let mut sh = self.shared.lock().unwrap();
                // Re-attach the vcpu to its task slot — unless the task was
                // *reaped while in flight*. A task's own worker services its
                // `exit` in place, marking it a `Zombie` under the kernel lock
                // before shipping the vcpu back here; in that window a sibling's
                // `wait4`/`waitid` (also under the kernel lock) can reap the
                // zombie and clear its slot to `None`. The orphaned vcpu is then
                // simply dropped: the task is gone. Only a just-exited task can
                // hit this (a runnable/blocked task is never a reap target), but
                // guarding every arm keeps the invariant local.
                let reattach = |sh: &mut Shared, vcpu| {
                    if let Some(p) = sh.procs[i].as_mut() {
                        p.vcpu = Some(vcpu);
                    }
                };
                match out {
                    SliceOutcome::Err(e) => {
                        reattach(&mut sh, vcpu);
                        break Err(e);
                    }
                    SliceOutcome::Blocked(made_progress) => {
                        reattach(&mut sh, vcpu);
                        if made_progress {
                            epoch += 1;
                            stalled = false;
                        }
                        // Parked at the (post-progress) epoch: it won't re-run
                        // until some *later* progress advances the epoch.
                        blocked_at.insert(i, epoch);
                    }
                    SliceOutcome::Ended => {
                        reattach(&mut sh, vcpu);
                        epoch += 1;
                        stalled = false;
                    }
                    SliceOutcome::Yielded | SliceOutcome::Preempted => {
                        // Still runnable; make it immediately re-dispatchable.
                        reattach(&mut sh, vcpu);
                        blocked_at.remove(&i);
                        epoch += 1;
                        stalled = false;
                    }
                }
            };

            // Drain any still-in-flight slices so their vcpus are returned and
            // the workers go idle before we stop them (a slice that errored/
            // exited may have left siblings running). A drained task may already
            // have been reaped (see the re-attach note above), so tolerate a
            // `None` slot.
            while inflight > 0 {
                if let Ok((i, vcpu, _)) = done_rx.recv()
                    && let Some(p) = self.shared.lock().unwrap().procs[i].as_mut()
                {
                    p.vcpu = Some(vcpu);
                }
                inflight -= 1;
            }
            // One Stop per worker, into its own queue; the scope joins them.
            for q in &queues {
                q.0.lock().unwrap().push_back(Work::Stop);
                q.1.notify_one();
            }
            outcome
        })
    }

    /// Service one guest exit for task `i` **in place** on an SMP worker: swap
    /// the task's per-process state into a local `cx` (its slot in `sh.procs` is
    /// `take`n out for the duration, exactly as the serial scheduler does, so
    /// `fork`/`wait4`/`futex` scans don't see the running task), run the shared
    /// [`Kernel::service`] logic, then swap it back.
    ///
    /// The kernel lock is taken only for the checkout and the check-in — NOT
    /// across `service`, which acquires its own per-handler locks (sh before
    /// vfs) so sibling workers service their syscalls concurrently (step B2).
    /// The caller holds this address space's memory lock across the whole call,
    /// which serializes the service phases of tasks that share it (so the fd
    /// table can never be checked out twice at once). Returns what the worker
    /// should do next with the same vcpu.
    fn smp_service_step(
        &self,
        i: usize,
        exit: Exit,
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
        step_start: u128,
    ) -> SliceStep {
        // Checkout under `sh`, then release it before servicing.
        let (mut proc, mut cx, out_pending, out_ppid) = {
            let mut sh = self.shared.lock().unwrap();
            let mut proc = sh.procs[i].take().expect("dispatched task is in the table");
            // Own the task's per-step servicing state. `yield_now`/`block`/
            // `exec_ok` start clear so we observe only this step's value (the
            // serial path resets them likewise).
            let mut cx = ServiceCtx {
                cur: std::mem::take(&mut proc.info),
                ..ServiceCtx::default()
            };
            // Leave a stand-in in the slot while the syscall is serviced
            // outside the lock: other workers service theirs concurrently and
            // must still see this task — a parent's `wait4` otherwise found a
            // child mid-`exit` missing, got ECHILD, and exited early (the flaky
            // smp_e2e sum); `kill` by pid and reparenting look it up too.
            sh.procs[i] = Some(Process {
                vcpu: None,
                info: cx.cur.clone(),
            });
            sh.check_out_files(&mut cx);
            let (pending, ppid) = (cx.cur.pending, cx.cur.ppid);
            (proc, cx, pending, ppid)
        };
        if cx.cur.alarm_deadline.is_some() || !cx.cur.ptimers.is_empty() {
            fire_timers_if_due(&mut cx.cur, poll::now_ns());
        }
        let flow = self.service(&mut cx, exit, vcpu, mem);
        // Charge this step's wall time (run + service) to the task's CPU total,
        // and its nice-weighted share to the fair-scheduling virtual runtime.
        let delta = crate::clock::now_monotonic()
            .as_nanos()
            .saturating_sub(step_start);
        cx.cur.cpu_ns = cx.cur.cpu_ns.saturating_add(delta);
        charge_vruntime(&mut cx.cur, delta);
        {
            let mut sh = self.shared.lock().unwrap();
            // Fold in what other workers did to the stand-in meanwhile: signals
            // posted to this task, and a new parent (ours exited and reparented
            // us to init). Anything else on it was a copy of our own state.
            if let Some(stand_in) = sh.procs[i].take() {
                cx.cur.pending |= stand_in.info.pending & !out_pending;
                if stand_in.info.ppid != out_ppid {
                    cx.cur.ppid = stand_in.info.ppid;
                }
            }
            sh.check_in_files(&mut cx);
            // A task that just exited: wake its parent again now that the
            // zombie is visible. The exit's own wake-up went out while the
            // stand-in still read "running", so a parent whose `wait4` ran in
            // between parked on it and would otherwise wait for a stall retry.
            let ppid = matches!(cx.cur.run, RunState::Zombie(_)).then_some(cx.cur.ppid);
            proc.info = cx.cur;
            sh.procs[i] = Some(proc);
            if let Some(ppid) = ppid {
                for p in sh.procs.iter_mut().flatten() {
                    if p.info.pid == ppid {
                        p.info.parked = false;
                    }
                }
            }
        }
        match flow {
            Serviced::SetRet => {
                // Result already written in `service` (see the serial path); do
                // NOT re-write it — that double-advances the interpreter's pc.
                if cx.yield_now {
                    SliceStep::Yielded
                } else {
                    SliceStep::Continue
                }
            }
            Serviced::Resume => SliceStep::Continue,
            Serviced::Blocked => SliceStep::Blocked,
            Serviced::Ended => SliceStep::Ended,
        }
    }

    /// The syscall table. Returns the value the guest sees in its result
    /// register: a non-negative result, or a negative errno.
    /// Print registers and the top of the stack at a fatal guest fault. A guest
    /// that dies deep inside a JIT is otherwise a bare address; the register
    /// file plus the words at `rsp` usually say immediately whether control flow
    /// was corrupted (a `ret` to a data address) or a pointer was simply null.
    /// Debug: watch a guest address (`NIXVM_WATCHCODE=0xADDR`) for its 8 bytes
    /// changing, printing the syscall/pc window it changed in — for tracking
    /// down a wild write that corrupts a code page real hardware would fault on.
    fn watch_code(&self, vcpu: &dyn Vcpu, mem: &GuestMemory, after: Sysno) {
        // Debug-only: acquires `sh` on its own (no other lock is held here).
        let mut sh = self.shared.lock().unwrap();
        let Some(addr) = sh.watch_addr else {
            return;
        };
        let now = mem.read_u64(addr).unwrap_or(0);
        if now != sh.watch_last {
            eprintln!(
                "[watch] {addr:#x}: {:#018x} -> {now:#018x} in the window before {after:?} (pc={:#x})",
                sh.watch_last,
                vcpu.pc()
            );
            sh.watch_last = now;
        }
    }

    /// Grow the initial thread's stack to cover a fault at `addr` (Linux's
    /// `VM_GROWSDOWN`): if `addr` lies in the reserved-but-unmapped stack region
    /// `[stack_limit, stack_top)`, map from its page up to the existing stack
    /// and return `true` so the faulting instruction re-runs. Like Linux ≥ 6.5,
    /// any access down to the reservation floor grows the stack (the old
    /// "must be near `sp`" heuristic was removed upstream). This is why only a
    /// small stack window is mapped at startup — the rest materializes on
    /// demand, and a runtime that measures its stack sees a fresh-looking size.
    #[allow(clippy::unused_self)]
    fn grow_stack(&self, cx: &mut ServiceCtx, addr: u64, mem: &mut GuestMemory) -> bool {
        let stack_top = mem.base() + mem.size();
        if addr < cx.cur.stack_limit || addr >= stack_top {
            return false;
        }
        // Only grow genuinely-unmapped pages (a fault on a mapped stack page is
        // a real protection error, not a growth request).
        if mem.page_prot(addr).is_some() {
            return false;
        }
        let page = addr - addr % PAGE_SIZE;
        // Map from the faulting page up to the first already-mapped page, so a
        // large downward sweep (JSC zeroing a frame) grows in one step rather
        // than faulting per page.
        let mut end = page;
        while end < stack_top && mem.page_prot(end).is_none() {
            end += PAGE_SIZE;
        }
        mem.map(page, end - page, crate::vcpu::mem::Prot::rw())
            .is_ok()
    }

    #[allow(clippy::unused_self)] // reads self.cur.pid context in the caller; kept a method for symmetry
    fn dump_fault_context(&self, vcpu: &dyn Vcpu, mem: &GuestMemory) {
        const NAMES: [&str; 16] = [
            "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11",
            "r12", "r13", "r14", "r15",
        ];
        let line: Vec<String> = NAMES
            .iter()
            .enumerate()
            .map(|(i, n)| format!("{n}={:#x}", vcpu.reg(i)))
            .collect();
        eprintln!("[fault]   regs: {}", line.join(" "));
        let pc = vcpu.pc();
        if let Ok(b) = mem.read_vec(pc, 16) {
            let hex: Vec<String> = b.iter().map(|x| format!("{x:02x}")).collect();
            eprintln!("[fault]   code@pc: {}", hex.join(" "));
        }
        let sp = vcpu.sp();
        let stack: Vec<String> = (0..8u64)
            .map(|i| match mem.read_u64(sp + i * 8) {
                Ok(v) => format!("{v:#x}"),
                Err(_) => "<unmapped>".to_string(),
            })
            .collect();
        eprintln!("[fault]   [rsp+0..64]: {}", stack.join(" "));
    }

    /// `NIXVM_TRACE` wrapper around [`Kernel::dispatch_inner`]: logs each call
    /// *and its return value*, since a syscall's result (an `-errno`, or the
    /// address an `mmap` actually handed back) is usually what explains a guest
    /// that aborts right after the call.
    #[allow(clippy::too_many_arguments)]
    fn dispatch(
        &self,
        cx: &mut ServiceCtx,
        sys: Sysno,
        raw: u64,
        args: &[u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        if !self.trace {
            return self.dispatch_impl(cx, sys, raw, args, vcpu, mem);
        }
        let (pid, pc) = (cx.cur.pid, vcpu.pc());
        eprintln!("[trace] pid={pid} pc={pc:#x} {sys:?} raw={raw} args={args:x?}");
        let ret = self.dispatch_impl(cx, sys, raw, args, vcpu, mem);
        if (-4095..0).contains(&ret) {
            eprintln!("[trace]   = {ret} (errno {})", -ret);
        } else {
            eprintln!("[trace]   = {ret:#x}");
        }
        ret
    }

    /// Route a syscall to the lock discipline it needs (steps B2/B3/B4/B5). No
    /// lock is pre-held here — each category acquires exactly the lock(s) it
    /// touches, always in the strict order `shared` (sh) → `vfs` → `net` →
    /// `pipes` → `pollfds` (`pollfds` is last):
    /// - **net-only** (the pure socket syscalls): take only `net` via
    ///   [`Self::dispatch_net`] — no sh, no vfs.
    /// - **pipes-only** (`pipe2`): take only `pipes` — no sh, no vfs, no net.
    /// - **pollfds-only** (`eventfd`/`timerfd_*`/`epoll_create`/`epoll_ctl`/
    ///   `inotify_init1`/`signalfd4`): take only `pollfds` via
    ///   [`Self::dispatch_pollfds`] — no sh, vfs, net, or pipes.
    /// - **fd-polymorphic** (`read`/`write`/`readv`/`writev`): peek the fd type
    ///   from `cx` (no lock), then take *one* of sh/vfs/net/pipes/pollfds (a file
    ///   op → vfs, a socket → net, a pipe → pipes, an eventfd/timerfd → pollfds,
    ///   every other target → sh) — never more than one.
    /// - **both** (`mmap`/`memfd_create`): take sh then vfs and hold both (they
    ///   mutate `shared` state *and* the mount table atomically). `sendfile`
    ///   takes sh → vfs, and net (socket dst) or pipes (pipe dst) too, last.
    /// - **vfs-only** (the FS hot path): take only `vfs` via [`Self::dispatch_vfs`].
    /// - **everything else**: take only `sh` via [`Self::dispatch_shared`] (the
    ///   B1 table; poll/select/epoll_wait additionally take `net` then `pipes`
    ///   then `pollfds` — after sh — for the readiness scan).
    #[allow(clippy::too_many_lines)]
    fn dispatch_impl(
        &self,
        cx: &mut ServiceCtx,
        sys: Sysno,
        raw: u64,
        args: &[u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        match sys {
            // fd-polymorphic: exactly one of sh/vfs, chosen from the fd type.
            Sysno::Write => self.sys_write(cx, args[0], args[1], args[2], mem),
            Sysno::Read => self.sys_read(cx, args[0], args[1], args[2], mem),
            Sysno::Readv => self.sys_readv(cx, args[0], args[1], args[2], mem),
            Sysno::Writev => self.sys_writev(cx, args[0], args[1], args[2], mem),
            // Self-locking: these move bytes between two fds of arbitrary kinds
            // and take the non-pipe side's lock, then `pipes`, themselves.
            Sysno::Splice => self.sys_splice(cx, args, mem),
            Sysno::Tee => self.sys_tee(cx, args[0], args[1], args[2], args[3]),
            Sysno::Vmsplice => self.sys_vmsplice(cx, args[0], args[1], args[2], args[3], mem),
            Sysno::Preadv2 => self.sys_rwv2(cx, args, false, mem),
            Sysno::Pwritev2 => self.sys_rwv2(cx, args, true, mem),
            // both: sh THEN vfs, held together for atomicity.
            Sysno::Mmap => {
                let mut sh = self.shared.lock().unwrap();
                let mut vfs = self.vfs.lock().unwrap();
                self.sys_mmap(&mut sh, &mut vfs, cx, args, mem)
            }
            Sysno::MemfdCreate => {
                let mut sh = self.shared.lock().unwrap();
                let mut vfs = self.vfs.lock().unwrap();
                self.sys_memfd_create(&mut sh, &mut vfs, cx, args[0], args[1], mem)
            }
            Sysno::Sendfile => {
                let mut sh = self.shared.lock().unwrap();
                let mut vfs = self.vfs.lock().unwrap();
                self.sys_sendfile(
                    &mut sh, &mut vfs, cx, args[0], args[1], args[2], args[3], mem,
                )
            }
            // vfs-only (the FS hot path): a single `vfs` lock for the whole group.
            Sysno::Openat
            | Sysno::Open
            | Sysno::Creat
            | Sysno::Lseek
            | Sysno::Pread64
            | Sysno::Pwrite64
            | Sysno::Preadv
            | Sysno::Pwritev
            | Sysno::Ftruncate
            | Sysno::Truncate
            | Sysno::Fallocate
            | Sysno::CopyFileRange
            | Sysno::Link
            | Sysno::Linkat
            | Sysno::Statx
            | Sysno::Fstat
            | Sysno::Newfstatat
            | Sysno::Stat
            | Sysno::Lstat
            | Sysno::Getdents64
            | Sysno::Getdents
            | Sysno::Chdir
            | Sysno::Fchdir
            | Sysno::Statfs
            | Sysno::Readlinkat
            | Sysno::Readlink
            | Sysno::Symlinkat
            | Sysno::Symlink
            | Sysno::Mkdirat
            | Sysno::Mkdir
            | Sysno::Utimensat
            | Sysno::Utime
            | Sysno::Utimes
            | Sysno::Futimesat
            | Sysno::Fchmodat
            | Sysno::Fchmodat2
            | Sysno::Chmod
            | Sysno::Fchmod
            | Sysno::Fchownat
            | Sysno::Fchown
            | Sysno::Chown
            | Sysno::Lchown
            | Sysno::Mknod
            | Sysno::Mknodat
            | Sysno::Faccessat
            | Sysno::Faccessat2
            | Sysno::Access
            | Sysno::Msync
            | Sysno::Mount
            | Sysno::Umount2
            | Sysno::Getxattr
            | Sysno::Lgetxattr
            | Sysno::Fgetxattr
            | Sysno::Getxattrat
            | Sysno::Setxattr
            | Sysno::Lsetxattr
            | Sysno::Fsetxattr
            | Sysno::Setxattrat
            | Sysno::Listxattr
            | Sysno::Llistxattr
            | Sysno::Flistxattr
            | Sysno::Listxattrat
            | Sysno::Removexattr
            | Sysno::Lremovexattr
            | Sysno::Fremovexattr
            | Sysno::Removexattrat
            | Sysno::Openat2
            | Sysno::Cachestat => {
                let mut vfs = self.vfs.lock().unwrap();
                self.dispatch_vfs(&mut vfs, cx, sys, args, mem)
            }
            // Namespace changes: sh THEN vfs, so an unlink/rename can see every
            // task's descriptors and keep open files alive (see `orphan.rs`).
            Sysno::Unlinkat
            | Sysno::Unlink
            | Sysno::Rmdir
            | Sysno::Renameat
            | Sysno::Renameat2
            | Sysno::Rename => {
                let mut sh = self.shared.lock().unwrap();
                let mut vfs = self.vfs.lock().unwrap();
                self.dispatch_namespace(&mut sh, &mut vfs, cx, sys, args, mem)
            }
            // net-only: the pure socket syscalls, holding ONLY `net` (the last
            // lock) via `dispatch_net` — no sh, no vfs may be taken below it.
            Sysno::Socket
            | Sysno::Socketpair
            | Sysno::Bind
            | Sysno::Listen
            | Sysno::Accept
            | Sysno::Accept4
            | Sysno::Connect
            | Sysno::Getsockname
            | Sysno::Getpeername
            | Sysno::Setsockopt
            | Sysno::Getsockopt
            | Sysno::Shutdown
            | Sysno::Sendto
            | Sysno::Recvfrom
            | Sysno::Sendmsg
            | Sysno::Recvmsg
            | Sysno::Sendmmsg
            | Sysno::Recvmmsg => {
                let mut net = self.net.lock().unwrap();
                self.dispatch_net(&mut net, cx, sys, args, mem)
            }
            // pipes-only: `pipe2` just allocates a fresh pipe, holding ONLY
            // `pipes` (the innermost/last lock) — no sh, no vfs, no net.
            Sysno::Pipe2 => {
                let mut pipes = self.pipes.lock().unwrap();
                self.sys_pipe2(&mut pipes, cx, args[0], args[1], mem)
            }
            // Legacy x86-64 `pipe(fds)`: no flags argument at all — the second
            // register holds whatever the caller left there.
            Sysno::Pipe => {
                let mut pipes = self.pipes.lock().unwrap();
                self.sys_pipe2(&mut pipes, cx, args[0], 0, mem)
            }
            // pollfds-only: the pure eventfd/timerfd/epoll-setup syscalls, holding
            // ONLY `pollfds` (the innermost/last lock) via `dispatch_pollfds` —
            // no sh, no vfs, no net, no pipes may be taken below it.
            Sysno::Eventfd
            | Sysno::Eventfd2
            | Sysno::TimerfdCreate
            | Sysno::TimerfdSettime
            | Sysno::TimerfdGettime
            | Sysno::EpollCreate
            | Sysno::EpollCreate1
            | Sysno::EpollCtl
            | Sysno::InotifyInit1
            | Sysno::InotifyInit
            | Sysno::Signalfd4
            | Sysno::Signalfd
            | Sysno::MqOpen
            | Sysno::MqUnlink
            | Sysno::MqNotify
            | Sysno::MqGetsetattr => {
                let mut pf = self.pollfds.lock().unwrap();
                self.dispatch_pollfds(&mut pf, cx, sys, args, mem)
            }
            // Pure clock/time reads: they touch no shared kernel state (only the
            // host wall clock and the caller's buffer), so they take NO lock.
            // These dominate the syscall stream of clock-polling runtimes
            // (Bun/JSC issues ~89% `clock_gettime`), where routing each through
            // the big `sh` lock cost an acquire/release on the hot path and
            // needless cross-thread contention under SMP.
            Sysno::ClockGettime => self.sys_clock_gettime(cx, args[0], args[1], mem),
            Sysno::Gettimeofday => time::sys_gettimeofday(args[0], mem),
            Sysno::ClockGetres => time::sys_clock_getres(args[1], mem),
            Sysno::Time => time::sys_time(args[0], mem),
            // everything else: a single `sh` lock, running the B1 syscall table.
            _ => {
                let mut sh = self.shared.lock().unwrap();
                self.dispatch_shared(&mut sh, cx, sys, raw, args, vcpu, mem)
            }
        }
    }

    /// The net-only syscalls (the pure socket path): called with `net` — and
    /// *only* `net` — held. `net` is the innermost/last lock, so **no
    /// `self.shared.lock()` and no `self.vfs.lock()` may appear anywhere below
    /// this** (that would take sh or vfs after net and invert the order). Every
    /// arm here touches just the socket table (plus per-task `cx`).
    #[allow(clippy::too_many_lines)]
    fn dispatch_net(
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
        sys: Sysno,
        args: &[u64; 6],
        mem: &mut GuestMemory,
    ) -> i64 {
        match sys {
            Sysno::Socket => self.sys_socket(net, cx, args[0], args[1], args[2]),
            Sysno::Socketpair => {
                self.sys_socketpair(net, cx, args[0], args[1], args[2], args[3], mem)
            }
            Sysno::Bind => self.sys_bind(net, cx, args[0], args[1], args[2], mem),
            Sysno::Listen => self.sys_listen(net, cx, args[0]),
            // `accept` is `accept4` with no flags.
            Sysno::Accept => self.sys_accept4(net, cx, args[0], args[1], args[2], 0, mem),
            Sysno::Accept4 => self.sys_accept4(net, cx, args[0], args[1], args[2], args[3], mem),
            Sysno::Connect => self.sys_connect(net, cx, args[0], args[1], args[2], mem),
            Sysno::Getsockname => self.sys_getsockname(net, cx, args[0], args[1], args[2], mem),
            Sysno::Getpeername => self.sys_getpeername(net, cx, args[0], args[1], args[2], mem),
            Sysno::Setsockopt => {
                self.sys_setsockopt(net, cx, args[0], args[1], args[2], args[3], args[4], mem)
            }
            Sysno::Getsockopt => {
                self.sys_getsockopt(net, cx, args[0], args[1], args[2], args[3], args[4], mem)
            }
            Sysno::Shutdown => self.sys_shutdown(net, cx, args[0], args[1]),
            // sendto/recvfrom carry an optional peer address (UDP) beyond
            // write/read; the `mmsg` forms loop the single-message path.
            Sysno::Sendto => self.sys_sendto(
                net, cx, args[0], args[1], args[2], args[3], args[4], args[5], mem,
            ),
            Sysno::Recvfrom => self.sys_recvfrom(
                net, cx, args[0], args[1], args[2], args[3], args[4], args[5], mem,
            ),
            Sysno::Sendmsg => self.sys_sendmsg(net, cx, args[0], args[1], args[2], mem),
            Sysno::Recvmsg => self.sys_recvmsg(net, cx, args[0], args[1], args[2], mem),
            Sysno::Sendmmsg => self.sys_sendmmsg(net, cx, args[0], args[1], args[2], mem),
            Sysno::Recvmmsg => self.sys_recvmmsg(net, cx, args[0], args[1], args[2], args[3], mem),
            // Unreachable: `dispatch_impl` only routes the syscalls above here.
            _ => unreachable!("dispatch_net: {sys:?} is not a net-only syscall"),
        }
    }

    /// The pollfds-only syscalls (the pure eventfd/timerfd/epoll-setup path):
    /// called with `pollfds` — and *only* `pollfds` — held. `pollfds` is the
    /// innermost/last lock, so **no `self.shared.lock()`, `self.vfs.lock()`,
    /// `self.net.lock()`, or `self.pipes.lock()` may appear anywhere below
    /// this** (that would take an outer lock after pollfds and invert the
    /// order). Every arm here touches just the event/timer/epoll tables (plus
    /// per-task `cx`).
    fn dispatch_pollfds(
        &self,
        pf: &mut PollFds,
        cx: &mut ServiceCtx,
        sys: Sysno,
        args: &[u64; 6],
        mem: &mut GuestMemory,
    ) -> i64 {
        match sys {
            Sysno::Eventfd => self.sys_eventfd2(pf, cx, args[0], 0),
            Sysno::Eventfd2 => self.sys_eventfd2(pf, cx, args[0], args[1]),
            Sysno::TimerfdCreate => self.sys_timerfd_create(pf, cx, args[0], args[1]),
            Sysno::TimerfdSettime => {
                self.sys_timerfd_settime(pf, cx, args[0], args[1], args[2], args[3], mem)
            }
            Sysno::TimerfdGettime => self.sys_timerfd_gettime(pf, cx, args[0], args[1], mem),
            Sysno::EpollCreate | Sysno::EpollCreate1 => self.sys_epoll_create1(pf, cx, args[0]),
            Sysno::EpollCtl => self.sys_epoll_ctl(pf, cx, args[0], args[1], args[2], args[3], mem),
            // inotify gets an eventfd-backed descriptor that never becomes
            // readable (no filesystem events delivered — a safe degradation).
            Sysno::InotifyInit1 => self.sys_inotify_init1(pf, cx, args[0]),
            Sysno::InotifyInit => self.sys_inotify_init1(pf, cx, 0),
            // signalfd4(fd, mask, sizemask, flags): a real signal-reading fd. The
            // `fd` is a 32-bit int (`-1` = create), read via `as i32`.
            Sysno::Signalfd4 => {
                self.sys_signalfd4(pf, cx, i64::from(args[0] as i32), args[1], args[3], mem)
            }
            // POSIX message queues (see `mqueue.rs`); send/receive also signal
            // and wake, so they run under `sh` instead.
            Sysno::MqOpen => self.sys_mq_open(pf, cx, args, mem),
            Sysno::MqUnlink => self.sys_mq_unlink(pf, args[0], mem),
            Sysno::MqNotify => self.sys_mq_notify(pf, cx, args[0], args[1], mem),
            Sysno::MqGetsetattr => self.sys_mq_getsetattr(pf, cx, args[0], args[1], args[2], mem),
            // Legacy x86-64 `signalfd(fd, mask, sizemask)`: signalfd4 with no flags.
            Sysno::Signalfd => {
                self.sys_signalfd4(pf, cx, i64::from(args[0] as i32), args[1], 0, mem)
            }
            // Unreachable: `dispatch_impl` only routes the syscalls above here.
            _ => unreachable!("dispatch_pollfds: {sys:?} is not a pollfds-only syscall"),
        }
    }

    /// The vfs-only syscalls (the filesystem hot path): called with `vfs` — and
    /// *only* `vfs` — held, so no `self.shared.lock()` may appear anywhere below
    /// (that would take sh after vfs and break the lock order). Every arm here
    /// touches just the mount table (plus per-task `cx`).
    #[allow(clippy::too_many_lines)]
    fn dispatch_vfs(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        sys: Sysno,
        args: &[u64; 6],
        mem: &mut GuestMemory,
    ) -> i64 {
        match sys {
            Sysno::Openat => {
                self.sys_openat(vfs, cx, args[0] as i64, args[1], args[2], args[3], mem)
            }
            Sysno::Open => self.sys_openat(vfs, cx, AT_FDCWD, args[0], args[1], args[2], mem),
            Sysno::Openat2 => self.sys_openat2(
                vfs,
                cx,
                i64::from(args[0] as i32),
                args[1],
                args[2],
                args[3],
                mem,
            ),
            Sysno::Cachestat => {
                self.sys_cachestat(vfs, cx, args[0], args[1], args[2], args[3], mem)
            }
            Sysno::Creat => {
                const O_WRONLY_CREAT_TRUNC: u64 = 0o1101;
                self.sys_openat(
                    vfs,
                    cx,
                    AT_FDCWD,
                    args[0],
                    O_WRONLY_CREAT_TRUNC,
                    args[1],
                    mem,
                )
            }
            Sysno::Lseek => self.sys_lseek(vfs, cx, args[0], args[1] as i64, args[2]),
            Sysno::Pread64 => self.sys_pread(vfs, cx, args[0], args[1], args[2], args[3], mem),
            Sysno::Pwrite64 => self.sys_pwrite(vfs, cx, args[0], args[1], args[2], args[3], mem),
            Sysno::Preadv => self.sys_preadv(vfs, cx, args[0], args[1], args[2], args[3], mem),
            Sysno::Pwritev => self.sys_pwritev(vfs, cx, args[0], args[1], args[2], args[3], mem),
            Sysno::Ftruncate => self.sys_ftruncate(vfs, cx, args[0], args[1]),
            Sysno::Truncate => self.sys_truncate(vfs, cx, args[0], args[1], mem),
            Sysno::Fallocate => self.sys_fallocate(vfs, cx, args[0], args[1], args[2], args[3]),
            Sysno::CopyFileRange => self.sys_copy_file_range(vfs, cx, args, mem),
            Sysno::Link => self.sys_linkat(vfs, cx, AT_FDCWD, args[0], AT_FDCWD, args[1], 0, mem),
            Sysno::Linkat => self.sys_linkat(
                vfs,
                cx,
                args[0] as i64,
                args[1],
                args[2] as i64,
                args[3],
                args[4],
                mem,
            ),
            Sysno::Statx => self.sys_statx(vfs, cx, args[0] as i64, args[1], args[2], args[4], mem),
            Sysno::Fstat => self.sys_fstat(vfs, cx, args[0], args[1], mem),
            Sysno::Newfstatat => {
                self.sys_newfstatat(vfs, cx, args[0] as i64, args[1], args[2], args[3], mem)
            }
            Sysno::Stat => self.sys_newfstatat(vfs, cx, AT_FDCWD, args[0], args[1], 0, mem),
            Sysno::Lstat => {
                const AT_SYMLINK_NOFOLLOW: u64 = 0x100;
                self.sys_newfstatat(
                    vfs,
                    cx,
                    AT_FDCWD,
                    args[0],
                    args[1],
                    AT_SYMLINK_NOFOLLOW,
                    mem,
                )
            }
            Sysno::Getdents64 => {
                self.sys_getdents64(vfs, cx, args[0], args[1], args[2], false, mem)
            }
            Sysno::Getdents => self.sys_getdents64(vfs, cx, args[0], args[1], args[2], true, mem),
            Sysno::Chdir => self.sys_chdir(vfs, cx, args[0], mem),
            Sysno::Fchdir => self.sys_fchdir(vfs, cx, args[0]),
            Sysno::Statfs => self.sys_statfs(vfs, cx, args[0], args[1], mem),
            Sysno::Readlinkat => {
                self.sys_readlinkat(vfs, cx, args[0] as i64, args[1], args[2], args[3], mem)
            }
            Sysno::Readlink => {
                self.sys_readlinkat(vfs, cx, AT_FDCWD, args[0], args[1], args[2], mem)
            }
            Sysno::Symlinkat => self.sys_symlinkat(vfs, cx, args[0], args[1] as i64, args[2], mem),
            Sysno::Symlink => self.sys_symlinkat(vfs, cx, args[0], AT_FDCWD, args[1], mem),
            Sysno::Mkdirat => self.sys_mkdirat(vfs, cx, args[0] as i64, args[1], args[2], mem),
            Sysno::Mkdir => self.sys_mkdirat(vfs, cx, AT_FDCWD, args[0], args[1], mem),
            Sysno::Utimensat => {
                self.sys_utimensat(vfs, cx, args[0] as i64, args[1], args[2], args[3], mem)
            }
            // The pre-`utimensat` x86-64 spellings: `utime` takes a `struct
            // utimbuf` (whole seconds), `utimes`/`futimesat` a `timeval[2]`.
            Sysno::Utime => self.sys_utimes_legacy(vfs, cx, AT_FDCWD, args[0], args[1], true, mem),
            Sysno::Utimes => {
                self.sys_utimes_legacy(vfs, cx, AT_FDCWD, args[0], args[1], false, mem)
            }
            Sysno::Futimesat => self.sys_utimes_legacy(
                vfs,
                cx,
                i64::from(args[0] as i32),
                args[1],
                args[2],
                false,
                mem,
            ),
            // legacy chmod(path, mode) vs fchmodat(dirfd, path, mode, flags).
            Sysno::Chmod => self.sys_fchmodat(vfs, cx, AT_FDCWD, args[0], args[1], mem),
            Sysno::Fchmodat => {
                self.sys_fchmodat(vfs, cx, i64::from(args[0] as i32), args[1], args[2], mem)
            }
            Sysno::Fchmod => self.sys_fchmod(vfs, cx, args[0], args[1]),
            Sysno::Fchmodat2 => self.sys_fchmodat2(
                vfs,
                cx,
                i64::from(args[0] as i32),
                args[1],
                args[2],
                args[3],
                mem,
            ),
            // chown follows symlinks; lchown acts on the link (AT_SYMLINK_NOFOLLOW=0x100).
            Sysno::Chown => self.sys_fchownat(vfs, cx, AT_FDCWD, args[0], args[1], args[2], 0, mem),
            Sysno::Lchown => {
                self.sys_fchownat(vfs, cx, AT_FDCWD, args[0], args[1], args[2], 0x100, mem)
            }
            Sysno::Fchownat => self.sys_fchownat(
                vfs,
                cx,
                i64::from(args[0] as i32),
                args[1],
                args[2],
                args[3],
                args[4],
                mem,
            ),
            Sysno::Fchown => self.sys_fchown(vfs, cx, args[0], args[1], args[2]),
            // mknod(path, mode, dev); mknodat(dirfd, path, mode, dev). mkfifo is
            // glibc's mknod with S_IFIFO. `dev` is ignored (no device nodes).
            Sysno::Mknod => self.sys_mknodat(vfs, cx, AT_FDCWD, args[0], args[1], mem),
            Sysno::Mknodat => {
                self.sys_mknodat(vfs, cx, i64::from(args[0] as i32), args[1], args[2], mem)
            }
            Sysno::Faccessat | Sysno::Faccessat2 => {
                self.sys_faccessat(vfs, cx, args[0] as i64, args[1], args[2], mem)
            }
            Sysno::Access => self.sys_faccessat(vfs, cx, AT_FDCWD, args[0], args[1], mem),
            Sysno::Msync => self.sys_msync(vfs, cx, args[0], args[1], args[2], mem),
            Sysno::Mount => self.sys_mount(vfs, cx, args, mem),
            Sysno::Umount2 => self.sys_umount2(vfs, cx, args[0], args[1], mem),
            // Extended attributes: the path / no-follow / fd / *at spellings all
            // lower onto one handler per operation (see `xattr.rs`).
            Sysno::Getxattr | Sysno::Lgetxattr | Sysno::Fgetxattr => {
                let t = xattr_target(sys, args[0]);
                self.sys_getxattr(vfs, cx, &t, args[1], args[2], args[3], mem)
            }
            Sysno::Setxattr | Sysno::Lsetxattr | Sysno::Fsetxattr => {
                let t = xattr_target(sys, args[0]);
                self.sys_setxattr(vfs, cx, &t, args[1], args[2], args[3], args[4], mem)
            }
            Sysno::Listxattr | Sysno::Llistxattr | Sysno::Flistxattr => {
                let t = xattr_target(sys, args[0]);
                self.sys_listxattr(vfs, cx, &t, args[1], args[2], mem)
            }
            Sysno::Removexattr | Sysno::Lremovexattr | Sysno::Fremovexattr => {
                let t = xattr_target(sys, args[0]);
                self.sys_removexattr(vfs, cx, &t, args[1], mem)
            }
            // (dirfd, path, at_flags, name, struct xattr_args *, usize)
            Sysno::Getxattrat | Sysno::Setxattrat => {
                let t = match xattr::XattrTarget::at(args[0], args[1], args[2]) {
                    Ok(t) => t,
                    Err(e) => return e,
                };
                let (value, size, flags) = match Self::read_xattr_args(mem, args[4], args[5]) {
                    Ok(a) => a,
                    Err(e) => return e,
                };
                if sys == Sysno::Setxattrat {
                    self.sys_setxattr(vfs, cx, &t, args[3], value, size, flags, mem)
                } else if flags != 0 {
                    err(Errno::EINVAL) // getxattrat takes no flags
                } else {
                    self.sys_getxattr(vfs, cx, &t, args[3], value, size, mem)
                }
            }
            // (dirfd, path, at_flags, list, size)
            Sysno::Listxattrat => match xattr::XattrTarget::at(args[0], args[1], args[2]) {
                Ok(t) => self.sys_listxattr(vfs, cx, &t, args[3], args[4], mem),
                Err(e) => e,
            },
            // (dirfd, path, at_flags, name)
            Sysno::Removexattrat => match xattr::XattrTarget::at(args[0], args[1], args[2]) {
                Ok(t) => self.sys_removexattr(vfs, cx, &t, args[3], mem),
                Err(e) => e,
            },
            // Unreachable: `dispatch_impl` only routes the syscalls above here.
            _ => unreachable!("dispatch_vfs: {sys:?} is not a vfs-only syscall"),
        }
    }

    /// The namespace-changing file syscalls (`unlink`/`rmdir`/`rename` and
    /// their `*at` forms), called with `sh` then `vfs` held: they consult every
    /// task's descriptor table to keep open files reachable (see `orphan.rs`).
    fn dispatch_namespace(
        &self,
        sh: &mut Shared,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        sys: Sysno,
        args: &[u64; 6],
        mem: &mut GuestMemory,
    ) -> i64 {
        const AT_REMOVEDIR: u64 = 0x200;
        match sys {
            Sysno::Unlinkat => {
                self.sys_unlink_keep_open(sh, vfs, cx, args[0] as i64, args[1], args[2], mem)
            }
            Sysno::Unlink => self.sys_unlink_keep_open(sh, vfs, cx, AT_FDCWD, args[0], 0, mem),
            Sysno::Rmdir => self.sys_unlinkat(vfs, cx, AT_FDCWD, args[0], AT_REMOVEDIR, mem),
            // renameat has no flags; renameat2's flags are arg 4.
            Sysno::Renameat | Sysno::Renameat2 => self.sys_rename_keep_open(
                sh,
                vfs,
                cx,
                args[0] as i64,
                args[1],
                args[2] as i64,
                args[3],
                if sys == Sysno::Renameat2 { args[4] } else { 0 },
                mem,
            ),
            Sysno::Rename => {
                self.sys_rename_keep_open(sh, vfs, cx, AT_FDCWD, args[0], AT_FDCWD, args[1], 0, mem)
            }
            _ => unreachable!("dispatch_namespace: {sys:?} is not a namespace syscall"),
        }
    }

    /// The B1 syscall table: every syscall that touches `shared` (and nothing in
    /// the mount table), run with `sh` held. Unchanged from B1 except that the
    /// FS / fd-polymorphic / mmap-family arms moved to [`Self::dispatch_impl`]/
    /// [`Self::dispatch_vfs`]. A handler here that *also* needs the mount table
    /// (`execve`, `exit`, `munmap`, …) acquires `vfs` internally — always after
    /// `sh`, never before. Handlers that also touch `net`/`pipes` (the
    /// poll/select/epoll readiness scans, `bump_pipe`/`clone`'s pipe- and
    /// socket-refcount bumps) acquire them internally — always after `sh`
    /// (sh → net → pipes), never before.
    #[allow(clippy::too_many_lines)] // one arm per syscall; a flat table is clearest.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_shared(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        sys: Sysno,
        raw: u64,
        args: &[u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        match sys {
            // `sched_yield` succeeds *and* ends the slice, so a sibling gets the
            // CPU — see [`Kernel::yield_now`].
            Sysno::SchedYield => {
                cx.yield_now = true;
                0
            }
            Sysno::Brk => self.sys_brk(cx, args[0], mem),
            // An `mseal`ed range can't be unmapped or remapped.
            Sysno::Munmap | Sysno::Mremap if Self::is_sealed(sh, cx, args[0], args[1]) => {
                err(Errno::EPERM)
            }
            Sysno::Munmap => self.sys_munmap(sh, cx, args[0], args[1], mem),
            Sysno::Mprotect => self.sys_mprotect_sealed(sh, cx, args[0], args[1], args[2], mem),
            Sysno::Mremap => {
                self.sys_mremap(sh, cx, args[0], args[1], args[2], args[3], args[4], mem)
            }
            Sysno::Madvise => self.sys_madvise(args[0], args[1], args[2], mem),
            Sysno::Mincore => self.sys_mincore(args[0], args[1], args[2], mem),
            Sysno::Uname => self.sys_uname(sh, args[0], mem),
            // ClockGettime/Gettimeofday/ClockGetres/Time are handled in the fast
            // `dispatch_impl` table (they never reach here). nanosleep's interval
            // is relative (no clock/flags); clock_nanosleep carries a clock id and
            // flags (TIMER_ABSTIME).
            // Real-time timers (ITIMER_REAL) → SIGALRM.
            Sysno::Alarm => self.sys_alarm(cx, args[0]),
            // POSIX timers (see `ptimer.rs`).
            Sysno::TimerCreate => self.sys_timer_create(sh, cx, args[0], args[1], args[2], mem),
            Sysno::TimerSettime => {
                self.sys_timer_settime(sh, cx, args[0], args[1], args[2], args[3], mem)
            }
            Sysno::TimerGettime => self.sys_timer_gettime(sh, cx, args[0], args[1], mem),
            Sysno::TimerGetoverrun => self.sys_timer_getoverrun(sh, cx, args[0]),
            Sysno::TimerDelete => self.sys_timer_delete(sh, cx, args[0]),
            Sysno::MqTimedsend => self.sys_mq_timedsend(sh, cx, args, mem),
            Sysno::MqTimedreceive => self.sys_mq_timedreceive(sh, cx, args, mem),
            // System V IPC (see `ipc.rs`).
            Sysno::Msgget => self.sys_msgget(sh, cx, args[0], args[1]),
            Sysno::Msgsnd => self.sys_msgsnd(sh, cx, args, mem),
            Sysno::Msgrcv => self.sys_msgrcv(sh, cx, args, mem),
            Sysno::Msgctl => self.sys_msgctl(sh, cx, args[0], args[1], args[2], mem),
            Sysno::Semget => self.sys_semget(sh, cx, args[0], args[1], args[2]),
            Sysno::Semop => self.sys_semtimedop(sh, cx, args, 0, mem),
            Sysno::Semtimedop => self.sys_semtimedop(sh, cx, args, args[3], mem),
            Sysno::Semctl => self.sys_semctl(sh, cx, args, mem),
            Sysno::Shmget => self.sys_shmget(sh, cx, args[0], args[1], args[2], mem),
            Sysno::Shmat => self.sys_shmat(sh, cx, args[0], args[1], args[2], mem),
            Sysno::Shmdt => self.sys_shmdt(sh, cx, args[0], mem),
            Sysno::Shmctl => self.sys_shmctl(sh, cx, args[0], args[1], args[2], mem),
            // Cross-process: pidfds, kcmp, process_vm_* (see `procx.rs`).
            Sysno::PidfdOpen => self.sys_pidfd_open(sh, cx, args[0], args[1]),
            Sysno::PidfdSendSignal => {
                self.sys_pidfd_send_signal(sh, cx, args[0], args[1], args[2], args[3], mem)
            }
            Sysno::PidfdGetfd => self.sys_pidfd_getfd(sh, cx, args[0], args[1], args[2]),
            Sysno::Kcmp => self.sys_kcmp(sh, cx, args[0], args[1], args[2], args[3], args[4]),
            Sysno::ProcessVmReadv => self.sys_process_vm(sh, cx, args, false, mem),
            Sysno::ProcessVmWritev => self.sys_process_vm(sh, cx, args, true, mem),
            Sysno::ProcessMadvise => {
                self.sys_process_madvise(sh, cx, args[0], args[1], args[2], args[3], args[4], mem)
            }
            Sysno::ProcessMrelease => self.sys_process_mrelease(sh, cx, args[0], args[1]),
            Sysno::Setitimer => self.sys_setitimer(cx, args[0], args[1], args[2], mem),
            Sysno::Getitimer => self.sys_getitimer(cx, args[0], args[1], mem),
            Sysno::Nanosleep => self.sys_nanosleep(cx, 0, 0, args[0], args[1], mem),
            Sysno::ClockNanosleep => self.sys_nanosleep(cx, args[0], args[1], args[2], args[3], mem),
            // The guest does not own the host clock: refuse to set it. ptrace
            // is refused too (no debugging surface).
            Sysno::Settimeofday | Sysno::ClockSettime | Sysno::Ptrace => err(Errno::EPERM),
            // Closing the last descriptor of an unlinked file deletes it.
            // Closing a descriptor drops the locks it carried, and closing the
            // last descriptor of an unlinked file deletes it.
            Sysno::Close => {
                let closed = cx.cur.fds.get(args[0] as i32).cloned();
                let r = self.sys_close(cx, args[0] as i32);
                if let Some(f) = closed {
                    self.release_fd_locks(sh, cx, &f);
                }
                self.reap_orphans_locked(sh, cx);
                r
            }
            Sysno::CloseRange => self.sys_close_range(sh, cx, args[0], args[1], args[2]),
            // Credentials: the VM starts as root but a process may drop
            // privileges; the ids are tracked per task (see `Creds`).
            Sysno::Getuid => i64::from(cx.cur.creds.ruid),
            Sysno::Geteuid => i64::from(cx.cur.creds.euid),
            Sysno::Getgid => i64::from(cx.cur.creds.rgid),
            Sysno::Getegid => i64::from(cx.cur.creds.egid),
            Sysno::Getresuid => {
                let c = cx.cur.creds;
                self.sys_getres_id([c.ruid, c.euid, c.suid], args[0], args[1], args[2], mem)
            }
            Sysno::Getresgid => {
                let c = cx.cur.creds;
                self.sys_getres_id([c.rgid, c.egid, c.sgid], args[0], args[1], args[2], mem)
            }
            Sysno::Setuid => self.sys_setuid(cx, args[0] as u32),
            Sysno::Setgid => self.sys_setgid(cx, args[0] as u32),
            Sysno::Setreuid => self.sys_setreuid(cx, args[0], args[1]),
            Sysno::Setregid => self.sys_setregid(cx, args[0], args[1]),
            Sysno::Setresuid => self.sys_setresuid(cx, args[0], args[1], args[2]),
            Sysno::Setresgid => self.sys_setresgid(cx, args[0], args[1], args[2]),
            Sysno::Setfsuid => self.sys_setfsuid(cx, args[0]),
            Sysno::Setfsgid => self.sys_setfsgid(cx, args[0]),
            // Process groups / sessions.
            Sysno::Setpgid => self.sys_setpgid(sh, cx, args[0] as i32, args[1] as i32),
            Sysno::Getpgid => self.sys_getpgid(sh, cx, args[0] as i32),
            Sysno::Getpgrp => i64::from(pgid_of(&cx.cur)),
            Sysno::Setsid => self.sys_setsid(cx),
            Sysno::Getsid => self.sys_getsid(sh, cx, args[0] as i32),
            // Process lifecycle.
            Sysno::Waitid => self.sys_waitid(sh, cx, args[0], i64::from(args[1] as i32), args[2], args[3], args[4], mem),
            Sysno::Clone3 => self.sys_clone3(sh, cx, args[0], args[1], vcpu, mem),
            Sysno::Execveat => {
                self.sys_execveat(sh, cx, args[0] as i64, args[1], args[2], args[3], args[4], vcpu, mem)
            }
            // A watch descriptor the guest can pass to inotify_rm_watch (which
            // is a no-op in the always-succeed group below).
            Sysno::InotifyAddWatch => 1,
            // restart_syscall reports the interrupted call didn't resume.
            Sysno::RestartSyscall => err(Errno::EINTR),
            // pause() blocks until a signal; with our minimal signal delivery
            // it simply parks (the guest re-traps).
            Sysno::Pause => {
                cx.block = true;
                cx.restartable = false; // pause() always returns -EINTR when caught
                0
            }
            Sysno::Getcwd => self.sys_getcwd(cx, args[0], args[1], mem),
            Sysno::Fstatfs => self.sys_fstatfs(cx, args[0], args[1], mem),
            Sysno::Umask => self.sys_umask(sh, args[0]),
            // Sync family that takes an fd: nothing is durably backed, so there's
            // nothing to flush — but a bad fd must still report EBADF (real
            // fsync/fdatasync/syncfs/sync_file_range validate the descriptor).
            Sysno::Fsync | Sysno::Fdatasync | Sysno::Syncfs | Sysno::SyncFileRange => {
                if cx.cur.fds.get(args[0] as i32).is_some() {
                    0
                } else {
                    err(Errno::EBADF)
                }
            }
            Sysno::Getrandom => self.sys_getrandom(sh, args[0], args[1], args[2], mem),
            // mlock/munlock/mlock2(addr, len[, flags]): nothing is ever paged
            // out, so locking is a no-op — after Linux's checks: the range must
            // be mapped (ENOMEM), mlock2 takes only MLOCK_ONFAULT, and mlockall
            // needs MCL_CURRENT and/or MCL_FUTURE (MCL_ONFAULT alone is EINVAL).
            Sysno::Mlock2 if args[2] & !1 != 0 => err(Errno::EINVAL),
            Sysno::Mlock | Sysno::Munlock | Sysno::Mlock2 => {
                let (start, end) = (page_down(args[0]), args[0].saturating_add(args[1]));
                let mut p = start;
                while p < end {
                    if mem.page_prot(p).is_none() {
                        return err(Errno::ENOMEM);
                    }
                    p += PAGE_SIZE;
                }
                0
            }
            Sysno::Mlockall => {
                if args[0] & !7 != 0 || args[0] & 3 == 0 {
                    err(Errno::EINVAL)
                } else {
                    0
                }
            }
            Sysno::Ioctl => self.sys_ioctl(cx, args[0], args[1], args[2], mem),
            Sysno::Fcntl => self.sys_fcntl(sh, cx, args[0], args[1], args[2], mem),
            Sysno::Flock => self.sys_flock(sh, cx, args[0], args[1]),
            Sysno::Futex => self.sys_futex(sh, cx, args, mem),
            Sysno::FutexWaitv => self.sys_futex_waitv(sh, cx, args, mem),
            Sysno::FutexWake => self.sys_futex2_wake(sh, cx, args, mem),
            Sysno::FutexWait => self.sys_futex2_wait(sh, cx, args, mem),
            Sysno::FutexRequeue => self.sys_futex2_requeue(sh, cx, args, mem),
            // Event-notification / readiness scans. `sh` stays held (outermost)
            // by this dispatcher; each scan additionally acquires
            // net → pipes → pollfds internally (order sh → net → pipes →
            // pollfds), so it takes no `sh` param. The pure eventfd/timerfd/
            // epoll-setup syscalls are pollfds-only and routed via
            // `dispatch_pollfds` in `dispatch_impl` (they never touch `sh`).
            Sysno::Poll => self.sys_poll(cx, args[0], args[1], args[2] as i64, mem),
            Sysno::Ppoll => self.sys_ppoll(cx, args[0], args[1], args[2], args[3], args[4], mem),
            Sysno::Select => self.sys_select(cx, args[0], args[1], args[2], args[3], args[4], mem),
            Sysno::Pselect6 => {
                self.sys_pselect6(cx, args[0], args[1], args[2], args[3], args[4], args[5], mem)
            }
            Sysno::EpollWait => self.sys_epoll_wait(cx, args[0], args[1], args[2], args[3] as i64, mem),
            // epoll_pwait/pwait2 carry a sigmask (arg 4, a direct sigset pointer)
            // installed for the wait, exactly like ppoll.
            Sysno::EpollPwait => {
                poll::install_poll_sigmask(cx, args[4], mem);
                self.sys_epoll_wait(cx, args[0], args[1], args[2], args[3] as i64, mem)
            }
            Sysno::EpollPwait2 => {
                poll::install_poll_sigmask(cx, args[4], mem);
                self.sys_epoll_pwait2(cx, args[0], args[1], args[2], args[3], mem)
            }
            Sysno::Dup => self.sys_dup(cx, args[0]),
            // dup2 has no flags (pass 0); dup3's 3rd arg is O_CLOEXEC.
            Sysno::Dup2 | Sysno::Dup3 => {
                let replaced = cx.cur.fds.get(args[1] as i32).cloned();
                let r = if sys == Sysno::Dup3 {
                    self.sys_dup2(cx, args[0], args[1], args[2], true)
                } else {
                    self.sys_dup2(cx, args[0], args[1], 0, false)
                };
                if r >= 0
                    && args[0] != args[1]
                    && let Some(f) = replaced
                {
                    self.release_fd_locks(sh, cx, &f);
                }
                self.reap_orphans_locked(sh, cx);
                r
            }
            Sysno::Clone => self.sys_clone(sh, cx, args, vcpu, mem),
            // x86-64's legacy spellings of clone: `fork` is
            // `clone(SIGCHLD, ...)`, `vfork` is `clone(CLONE_VM|CLONE_VFORK|
            // SIGCHLD, ...)` — aarch64 never had either as its own syscall.
            Sysno::Fork => self.sys_clone(sh, cx, &[0x11, 0, 0, 0, 0, 0], vcpu, mem),
            Sysno::Vfork => self.sys_clone(sh, cx, &[0x4111, 0, 0, 0, 0, 0], vcpu, mem),
            Sysno::Execve => self.sys_execve(sh, cx, args[0], args[1], args[2], vcpu, mem),
            // `pid` is a 32-bit int: `as i32 as i64` sign-extends a `-1` the guest
            // passed as a zero-extended `0xFFFF_FFFF` (else `waitpid(-1)` breaks).
            // WNOHANG | WUNTRACED | WCONTINUED | __WNOTHREAD | __WALL | __WCLONE.
            Sysno::Wait4 if args[2] & !(1 | 2 | 8 | 0xe000_0000) != 0 => err(Errno::EINVAL),
            Sysno::Wait4 => self.sys_wait4(sh, cx, i64::from(args[0] as i32), args[1], args[2], args[3], mem),
            Sysno::Exit => self.sys_exit(sh, cx, args[0] as i32, mem),
            Sysno::ExitGroup => self.sys_exit_group(sh, cx, args[0] as i32, mem),
            // The rt_sig* calls take the kernel sigset size, which must be 8.
            Sysno::RtSigaction | Sysno::RtSigprocmask | Sysno::RtSigtimedwait
                if args[3] != 8 =>
            {
                err(Errno::EINVAL)
            }
            Sysno::RtSigsuspend if args[1] != 8 => err(Errno::EINVAL),
            Sysno::RtSigaction => self.sys_rt_sigaction(cx, args[0], args[1], args[2], mem),
            Sysno::Sigaltstack => self.sys_sigaltstack(cx, args[0], args[1], mem),
            Sysno::RtSigreturn => {
                self.sys_rt_sigreturn(cx, vcpu, mem);
                // The return value is whatever the restored context's rax holds;
                // it was just written into the vcpu, so don't overwrite it.
                cx.exec_ok = true;
                0
            }
            Sysno::RtSigprocmask => self.sys_rt_sigprocmask(cx, args[0], args[1], args[2], mem),
            Sysno::RtSigsuspend => self.sys_rt_sigsuspend(cx, args[0], mem),
            Sysno::RtSigpending => self.sys_rt_sigpending(cx, args[0], mem),
            Sysno::RtSigtimedwait => self.sys_rt_sigtimedwait(cx, args[0], args[1], args[2], mem),
            // `pid`/`tid` is a 32-bit int: `as i32 as i64` recovers a negative
            // `kill(-pgrp)` the guest passed zero-extended as `0xFFFF_FFFF`.
            Sysno::Kill | Sysno::Tkill => self.sys_kill(sh, cx, i64::from(args[0] as i32), args[1]),
            Sysno::Tgkill => self.sys_kill(sh, cx, i64::from(args[1] as i32), args[2]),
            // sigqueue/pthread_sigqueue: deliver the signal and carry its
            // siginfo (si_code/si_value) so an SA_SIGINFO handler sees the real
            // payload. tgsigqueueinfo targets a tid (args: tgid, tid, sig, uinfo).
            Sysno::RtSigqueueinfo => self.sys_rt_sigqueueinfo(sh, cx, i64::from(args[0] as i32), args[1], args[2], mem),
            Sysno::RtTgsigqueueinfo => self.sys_rt_sigqueueinfo(sh, cx, i64::from(args[1] as i32), args[2], args[3], mem),
            // getpid = thread-group id; gettid = this task's id.
            Sysno::Getpid => i64::from(cx.cur.tgid),
            Sysno::Gettid => i64::from(cx.cur.pid),
            // set_tid_address records the CHILD_CLEARTID word and returns the tid.
            Sysno::SetTidAddress => {
                cx.cur.clear_child_tid = args[0];
                i64::from(cx.cur.pid)
            }
            Sysno::Getppid => i64::from(cx.cur.ppid),
            // Resource / scheduling / process-attribute syscalls. The scheduling
            // attrs (policy/priority/nice/affinity) are recorded and reported
            // back — the cooperative scheduler doesn't honor them, but a program
            // that sets and re-reads them must see what it set.
            Sysno::SchedGetaffinity => {
                // The task's mask, or the default all-CPUs set when unset.
                let bits = if cx.cur.affinity != 0 {
                    cx.cur.affinity
                } else if self.ncpus >= 64 {
                    u64::MAX
                } else {
                    (1u64 << self.ncpus) - 1
                };
                sys_misc::sys_sched_getaffinity(bits, args[1], args[2], mem)
            }
            Sysno::SchedSetaffinity => self.sys_sched_setaffinity(cx, args[1], args[2], mem),
            Sysno::SchedGetparam => sys_misc::sys_sched_getparam(cx.cur.sched_priority, args[1], mem),
            Sysno::SchedGetscheduler => i64::from(cx.cur.sched_policy),
            Sysno::SchedSetscheduler => self.sys_sched_setscheduler(cx, args[1] as i32, args[2], mem),
            Sysno::SchedGetPriorityMax => self.sys_sched_priority_bound(args[0], true),
            Sysno::SchedGetPriorityMin => self.sys_sched_priority_bound(args[0], false),
            Sysno::Getrusage => self.sys_getrusage(sh, cx, args[0], args[1], mem),
            Sysno::Sysinfo => sys_misc::sys_sysinfo(args[0], mem),
            Sysno::Times => self.sys_times(sh, cx, args[0], mem),
            Sysno::Getcpu => sys_misc::sys_getcpu(args[0], args[1], mem),
            Sysno::Capget => self.sys_capget(sh, cx, args[0], args[1], mem),
            Sysno::Capset => self.sys_capset(cx, args[0], args[1], mem),
            // Scheduling / I/O priority / NUMA / pkeys / sealing / rseq /
            // membarrier / clock discipline / names / robust list / namespaces
            // (see `attrs.rs`).
            Sysno::SchedSetparam => self.sys_sched_setparam(sh, cx, args[0], args[1], mem),
            Sysno::SchedRrGetInterval => self.sys_sched_rr_get_interval(sh, cx, args[0], args[1], mem),
            Sysno::SchedSetattr => self.sys_sched_setattr(sh, cx, args[0], args[1], args[2], mem),
            Sysno::SchedGetattr => {
                self.sys_sched_getattr(sh, cx, args[0], args[1], args[2], args[3], mem)
            }
            Sysno::IoprioSet => self.sys_ioprio(sh, cx, args[0], args[1], Some(args[2])),
            Sysno::IoprioGet => self.sys_ioprio(sh, cx, args[0], args[1], None),
            Sysno::SetMempolicy => self.sys_set_mempolicy(cx, args[0], args[1], args[2], mem),
            Sysno::GetMempolicy => {
                self.sys_get_mempolicy(cx, args[0], args[1], args[2], args[3], args[4], mem)
            }
            Sysno::Mbind => self.sys_mbind(args, mem),
            Sysno::MovePages => self.sys_move_pages(sh, cx, args, mem),
            // migrate_pages(pid, maxnode, old, new): every page is already on
            // the one node; the count of pages that could not move is 0.
            Sysno::MigratePages => match (
                attrs_check_pid(sh, cx, args[0]),
                attrs_nodes_ok(mem, args[2], args[1]),
                attrs_nodes_ok(mem, args[3], args[1]),
            ) {
                (Err(e), _, _) | (_, Err(e), _) | (_, _, Err(e)) => e,
                _ => 0,
            },
            // set_mempolicy_home_node(start, len, home_node, flags): only node 0
            // exists; ranges without an explicit bind policy are skipped (0).
            Sysno::SetMempolicyHomeNode => {
                if args[3] != 0 || args[2] != 0 || !args[0].is_multiple_of(PAGE_SIZE) {
                    err(Errno::EINVAL)
                } else {
                    0
                }
            }
            Sysno::PkeyAlloc => self.sys_pkey_alloc(args[0], args[1]),
            Sysno::PkeyMprotect => self.sys_pkey_mprotect(sh, cx, args, mem),
            Sysno::Mseal => self.sys_mseal(sh, cx, args[0], args[1], args[2], mem),
            Sysno::Membarrier => self.sys_membarrier(sh, cx, args[0], args[1]),
            Sysno::Rseq => self.sys_rseq(cx, args[0], args[1], args[2], args[3], mem),
            Sysno::Adjtimex => self.sys_adjtimex(0, args[0], mem),
            Sysno::ClockAdjtime => self.sys_adjtimex(args[0], args[1], mem),
            Sysno::Personality => self.sys_personality(cx, args[0]),
            Sysno::Sethostname => self.sys_setname(sh, cx, args[0], args[1], false, mem),
            Sysno::Setdomainname => self.sys_setname(sh, cx, args[0], args[1], true, mem),
            Sysno::SetRobustList => self.sys_robust_list(sh, cx, args, false, mem),
            Sysno::GetRobustList => self.sys_robust_list(sh, cx, args, true, mem),
            Sysno::Unshare => self.sys_unshare(sh, cx, args[0]),
            Sysno::Seccomp => self.sys_seccomp(sh, cx, args[0], args[1], args[2], mem),
            Sysno::Setns => self.sys_setns(cx, args[0], args[1]),
            Sysno::Ustat => self.sys_ustat(args[1], mem),
            Sysno::Sysfs => self.sys_sysfs(args[0], args[1], args[2], mem),
            // vhangup: hang up the controlling terminal — needs
            // CAP_SYS_TTY_CONFIG; for root a no-op (the session's tty is the
            // host's or an in-VM pty that the caller is about to reopen).
            Sysno::Vhangup => {
                if cx.cur.creds.euid == 0 {
                    0
                } else {
                    err(Errno::EPERM)
                }
            }
            Sysno::Iopl => self.sys_ioport(args, false),
            Sysno::Ioperm => self.sys_ioport(args, true),
            // pkey_free: no key was ever allocated. remap_file_pages: Linux
            // emulates it only on MAP_SHARED file mappings (it is deprecated);
            // nixvm's file mappings are private copies, so the call can only be
            // refused as for any other mapping.
            Sysno::PkeyFree | Sysno::RemapFilePages => err(Errno::EINVAL),
            Sysno::Prlimit64 => self.sys_prlimit64(sh, args[1], args[2], args[3], mem),
            Sysno::Getrlimit => self.sys_getrlimit(sh, args[0], args[1], mem),
            // setrlimit(resource, rlim) is prlimit64 on the caller without the
            // old-value output (so RLIMIT_NOFILE is tracked either way).
            Sysno::Setrlimit => self.sys_prlimit64(sh, args[0], args[1], 0, mem),
            Sysno::Prctl => self.sys_prctl(cx, args, mem),
            // getpriority returns the kernel ABI value 20 - nice (glibc converts
            // it back to the nice value); setpriority records the nice.
            Sysno::Getpriority => i64::from(20 - cx.cur.nice),
            Sysno::Setpriority => self.sys_setpriority(cx, i64::from(args[2] as i32)),
            // arch_prctl(ARCH_SET_FS) — how an x86-64 guest installs its TLS
            // register (FS.base; aarch64 uses the MSR-like TPIDR_EL0 via
            // CLONE_SETTLS instead, so this arm only ever fires for x86-64).
            // The GS and GET_* subcommands aren't modeled.
            Sysno::ArchPrctl => self.sys_arch_prctl(cx, args[0], args[1], vcpu, mem),
            // Succeed as root / no-op: uid queries, signal setup, robust list,
            // permission/ownership/timestamp changes, socket options, clock
            // adjustment (TIME_OK), and scheduling/process-attr setters — none
            // modeled yet.
            // Locking/sync setters: no-ops (there is no swap to keep pages from).
            | Sysno::Munlockall
            // Sync family: nothing is durably backed (in-memory / host
            // passthrough), so there's nothing to flush. `sync()` takes no fd;
            // the fd-taking members (fsync/fdatasync/syncfs/sync_file_range) are
            // handled separately below so a bad fd reports EBADF.
            | Sysno::Sync
            // chroot is accepted without confining path resolution (a per-
            // process root isn't modeled — documented limitation).
            | Sysno::Chroot
            | Sysno::InotifyRmWatch => 0,
            Sysno::Setgroups => self.sys_setgroups(cx, args[0], args[1], mem),
            Sysno::Getgroups => self.sys_getgroups(cx, args[0], args[1], mem),
            Sysno::Syslog => self.sys_syslog(cx, args[0], args[1], args[2], mem),
            // readahead(fd, off, count) / fadvise64(fd, off, len, advice):
            // hints with nothing to act on in memory, after Linux's checks.
            Sysno::Readahead | Sysno::Fadvise64 => match cx.cur.fds.get(args[0] as i32) {
                None => err(Errno::EBADF),
                Some(Fd::PipeRead(_) | Fd::PipeWrite(_)) if sys == Sysno::Fadvise64 => {
                    err(Errno::ESPIPE)
                }
                Some(Fd::File { .. }) if sys == Sysno::Readahead || args[3] <= 5 => 0,
                Some(Fd::Dir { .. }) if sys == Sysno::Fadvise64 && args[3] <= 5 => 0,
                _ => err(Errno::EINVAL),
            },
            _ => {
                // Known-but-refused syscalls answer their documented errno and
                // stay out of the unknown ledger.
                if let Some(ret) = self.sys_unavailable(sys, args) {
                    return ret;
                }
                *sh.unsupported.entry(raw).or_default() += 1;
                err(Errno::ENOSYS)
            }
        }
    }

    // ---- process lifecycle ------------------------------------------------

    /// `clone(flags, stack, ...)` — the one primitive behind both `fork` (a new
    /// process with a copied address space) and `pthread_create` (a thread that
    /// shares the caller's address space).
    ///
    /// `CLONE_VM` shares the address space (the new task's `mm` points at the
    /// same [`Kernel::spaces`] slot); otherwise the space is copied. The one
    /// exception is `vfork` (`CLONE_VM | CLONE_VFORK`, no `CLONE_THREAD`), which
    /// is copied anyway — see the `is_vfork` comment below. `CLONE_THREAD`
    /// puts the new task in the caller's thread group (shared `tgid`, distinct
    /// `pid`/tid, not reaped by `wait4`). `CLONE_SETTLS` seeds the thread pointer;
    /// the `*_SETTID`/`CHILD_CLEARTID` flags write/clear the tid words musl's
    /// pthread layer relies on. `CLONE_FILES` shares the fd table (every pthread
    /// sets it); without it — fork — the child gets a private copy.
    fn sys_clone(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        args: &[u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        let flags = args[0];
        let stack = args[1];
        // clone's tls/child_tid argument order differs by arch:
        //   aarch64: clone(flags, stack, parent_tid, tls, child_tid)
        //   x86-64:  clone(flags, stack, parent_tid, child_tid, tls)
        let parent_tid = args[2];
        let (tls, child_tid) = match self.arch {
            Arch::X86_64 => (args[4], args[3]),
            Arch::Aarch64 => (args[3], args[4]),
        };
        // Legacy `clone` packs the child's termination signal into the low byte
        // of `flags`, and `CLONE_PIDFD` reuses the `parent_tid` pointer as the
        // pidfd output (so it is mutually exclusive with `CLONE_PARENT_SETTID`,
        // which also writes through `parent_tid` — EINVAL together, as on
        // Linux). `clone3` splits both out; we lower legacy to the same
        // normalized [`CloneArgs`].
        if flags & CLONE_PIDFD != 0 && flags & CLONE_PARENT_SETTID != 0 {
            return err(Errno::EINVAL);
        }
        let ca = CloneArgs {
            flags: flags & !0xff,
            stack_ptr: stack,
            parent_tid,
            child_tid,
            tls,
            exit_signal: flags & 0xff,
            pidfd_ptr: if flags & CLONE_PIDFD != 0 {
                parent_tid
            } else {
                0
            },
        };
        self.do_clone(sh, cx, &ca, vcpu, mem)
    }

    /// The shared `clone`/`clone3` core: create a task per the fully-decoded
    /// [`CloneArgs`], implementing every flag nixvm honors and gracefully
    /// accepting the rest. Returns the child pid, or a negative errno.
    #[allow(clippy::too_many_lines)]
    fn do_clone(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        ca: &CloneArgs,
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        let flags = ca.flags;
        let stack = ca.stack_ptr;
        let parent_tid = ca.parent_tid;
        let child_tid = ca.child_tid;
        let tls = ca.tls;
        // An unknown high bit is a malformed request — Linux rejects it EINVAL.
        // (`CLONE_ALL_FLAGS` already spans the two clone3-only flags, so a valid
        // `CLONE_INTO_CGROUP`/`CLONE_CLEAR_SIGHAND` is not caught here.)
        if flags & !CLONE_ALL_FLAGS != 0 {
            return err(Errno::EINVAL);
        }
        // Linux's copy_process() consistency rules: a new mount or user
        // namespace can't share the fs context; a thread shares the handler
        // table, and a shared handler table needs a shared address space.
        if (flags & CLONE_FS != 0 && flags & (CLONE_NEWNS | CLONE_NEWUSER) != 0)
            || (flags & CLONE_THREAD != 0 && flags & CLONE_SIGHAND == 0)
            || (flags & CLONE_SIGHAND != 0 && flags & CLONE_VM == 0)
        {
            return err(Errno::EINVAL);
        }
        // `vfork` (CLONE_VM | CLONE_VFORK, no CLONE_THREAD) asks to *borrow* the
        // parent's address space, relying on real page tables: the child runs in
        // it only until it `execve`s (which installs a fresh mm) or `_exit`s,
        // with the parent frozen meanwhile. This kernel's `execve` replaces the
        // space in place (`*mem = new_mem`), so a truly shared slot would be
        // clobbered out from under the parent shell — the classic symptom being
        // `vi` and the shell fighting for the console. We instead give `vfork` a
        // copied address space (plain-fork semantics), which is the standard
        // user-mode emulation (QEMU does the same) and correct for how libc uses
        // it: the child only ever `execve`s or `_exit`s before touching memory.
        // Genuine threads always set CLONE_THREAD and keep sharing.
        let is_thread = flags & CLONE_THREAD != 0;
        let is_vfork = flags & CLONE_VFORK != 0;
        let share_vm = flags & CLONE_VM != 0 && (is_thread || !is_vfork);
        let share_files = flags & CLONE_FILES != 0;
        // CLONE_FS shares the filesystem context (cwd); CLONE_THREAD implies it,
        // and every pthread sets it, so a `chdir` in one thread is seen by all.
        let share_fs = flags & CLONE_FS != 0;
        // The namespace flags (CLONE_NEW*), CLONE_SYSVSEM, CLONE_IO, CLONE_PTRACE,
        // CLONE_UNTRACED, CLONE_DETACHED and CLONE_INTO_CGROUP need no code here:
        // nixvm models one global namespace of each kind (and runs as root, for
        // whom real Linux *creates* a new namespace successfully), a single SysV
        // sem/IO/cgroup context, and no ptrace — so each is ACCEPTED as a no-op,
        // the child simply running against the one global view. See the flag
        // definitions for why acceptance (not EPERM) is the faithful choice.
        // CLONE_SIGHAND asks to *share* the handler table; nixvm copies handlers
        // per task (`info = cx.cur.clone()` below), so a CLONE_SIGHAND/THREAD
        // child gets a snapshot of the caller's handlers rather than a live-shared
        // table. This matches the common case (handlers are installed before
        // threads spawn and rarely change afterward); true live sharing would need
        // a handler table like `file_tables` and is deferred.

        let pid = sh.next_pid;
        sh.next_pid += 1;
        let mut info = cx.cur.clone();
        info.pid = pid;
        info.run = RunState::Running;
        info.futex_wait = None;
        info.futex_waitv = Vec::new();
        info.futex_pi = false;
        info.futex_woken = false;
        // Per-task state a new task starts without (Linux: "the child's set of
        // pending signals is initially empty"; interval timers, alarms and
        // POSIX timers are not inherited by a fork child, and a new thread
        // doesn't get a second copy of the process's timers — a clone of them
        // would fire every expiration twice).
        info.pending = 0;
        info.queued_siginfo = [None; NSIG_SLOTS];
        info.rt_queue = BTreeMap::new();
        info.sigsuspend_prev = None;
        info.wake_deadline = None;
        info.alarm_deadline = None;
        info.alarm_interval_ns = 0;
        info.ptimers = Vec::new();
        // The robust-futex head is per thread and reset for every new task
        // (the new thread's libc registers its own); an rseq registration
        // survives into a fork child (same address in its copied memory) but
        // not into a new thread, which registers its own area.
        info.robust_list = 0;
        if flags & CLONE_VM != 0 {
            info.rseq = None;
        }
        // The parent-death signal is cleared for a fork child, and being a
        // child subreaper is a property of the process, not inherited.
        info.pdeathsig = 0;
        if !is_thread {
            info.pr.child_subreaper = false;
        }
        // A child inherits the parent's *process group*, so resolve the `pgid == 0`
        // ("group leader = self") sentinel to the parent's effective pgid here.
        // Left as 0 it would default to the child's *own* pid (`pgid_of`), putting
        // every forked child in its own group — then a parent's `wait4(0)`
        // (same-process-group) or `kill(0, …)` would miss all its children.
        info.pgid = pgid_of(&cx.cur);
        if is_thread {
            // A thread joins the caller's group: shared tgid, the leader's parent,
            // and no exit signal (only the group's termination notifies the parent).
            info.tgid = cx.cur.tgid;
            info.ppid = cx.cur.ppid;
            info.is_thread = true;
            info.exit_signal = 0;
        } else {
            info.tgid = pid;
            // CLONE_PARENT makes the child a *sibling* of the caller: its parent is
            // the caller's parent, so it is reaped by (and signals) the grandparent.
            info.ppid = if flags & CLONE_PARENT != 0 {
                cx.cur.ppid
            } else {
                cx.cur.pid
            };
            info.is_thread = false;
            // The termination signal is the low byte of `flags` (SIGCHLD for fork).
            info.exit_signal = (ca.exit_signal & 0xff) as u8;
        }
        // CLONE_CLEAR_SIGHAND (clone3): the child starts with default dispositions.
        if flags & CLONE_CLEAR_SIGHAND != 0 {
            info.handlers = [SigAction::default(); 65];
        }

        // Address space: share the caller's slot (CLONE_VM), or fork a
        // copy-on-write child (both parent and child pages become shared and
        // read-on-write until the first store privatizes a page).
        let mut child_mem = if share_vm { None } else { Some(mem.fork()) };
        info.mm = if share_vm {
            // A thread shares this address space: mark it so the SMP scheduler
            // runs its tasks serialized (one page-table tree + one kstack frame
            // can't be run concurrently without corruption). `mem` IS the shared
            // space (checked out from `sh.spaces[cx.cur.mm]`), so the child, which
            // shares the same `Arc<Mutex<GuestMemory>>`, sees the flag too.
            mem.mark_shared();
            cx.cur.mm
        } else {
            sh.spaces.len()
        };

        info.clear_child_tid = if flags & CLONE_CHILD_CLEARTID != 0 {
            child_tid
        } else {
            0
        };

        // tid notifications. The parent word lives in the caller's space (`mem`);
        // the child word lives in the child's space (shared `mem`, or the fresh
        // copy we are about to install). In *legacy* clone, CLONE_PIDFD repurposes
        // the `parent_tid` pointer for its pidfd output — but that flag is mutually
        // exclusive with CLONE_PARENT_SETTID, so at most one of the two writes ever
        // fires for a given pointer and no guard is needed here. (`clone3` gives
        // pidfd its own field, so the two never collide there either.)
        if flags & CLONE_PARENT_SETTID != 0 && parent_tid != 0 {
            let _ = mem.write(parent_tid, &(pid as u32).to_le_bytes());
        }
        if flags & CLONE_CHILD_SETTID != 0 && child_tid != 0 {
            match child_mem.as_mut() {
                Some(cm) => {
                    let _ = cm.write(child_tid, &(pid as u32).to_le_bytes());
                }
                None => {
                    let _ = mem.write(child_tid, &(pid as u32).to_le_bytes());
                }
            }
        }

        // File-descriptor table. `info.fds` is only a placeholder — the real
        // table lives in `sh.file_tables`, checked out into `cur.fds` while a
        // task runs. `CLONE_FILES` (every pthread) shares the caller's table id,
        // so both threads see the same open fds — libuv relies on this: one
        // thread's `uv_async_send` writes an eventfd another thread polls.
        // Without it (fork) the child gets a private copy, and its fds hold
        // independent references, so bump pipe/socket refcounts for the copy.
        info.fds = FdTable::default();
        if share_files {
            info.files = cx.cur.files;
        } else {
            let copy = cx.cur.fds.clone();
            let (mut r, mut w, mut socks) = (Vec::new(), Vec::new(), Vec::new());
            for fd in copy.values() {
                match fd {
                    Fd::PipeRead(i) => r.push(*i),
                    Fd::PipeWrite(i) => w.push(*i),
                    Fd::Socket { .. } => socks.push(fd.clone()),
                    _ => {}
                }
            }
            // Socket refcounts live in `net`, pipe refcounts in `pipes`: bump
            // each set under one guard, acquired *after* `sh` and in the strict
            // order sh → net → pipes (net first, then pipes — the innermost),
            // each acquired once around its loop and released here.
            if !socks.is_empty() {
                let mut net = self.net.lock().unwrap();
                for fd in &socks {
                    net.bump(fd, true);
                }
            }
            if !r.is_empty() || !w.is_empty() {
                let mut pipes = self.pipes.lock().unwrap();
                for i in r {
                    pipes[i].readers += 1;
                }
                for i in w {
                    pipes[i].writers += 1;
                }
            }
            info.files = sh.file_tables.len();
            sh.file_tables.push(Some(copy));
        }

        // Filesystem context (cwd). CLONE_FS shares the caller's `cwd_tables`
        // entry (a `chdir` in either is seen by both); a fork gets a private copy
        // of the current cwd. `cx.cur.cwd` holds the caller's live value (checked
        // out for the slice), so the fork snapshots it here.
        if share_fs {
            info.fs = cx.cur.fs;
        } else {
            info.fs = sh.cwd_tables.len();
            sh.cwd_tables.push(Some(cx.cur.cwd.clone()));
        }

        if let Some(cm) = child_mem.take() {
            // Seals are a property of the mappings, which the copy inherits.
            let child_mm = sh.spaces.len();
            let seals: Vec<_> = sh
                .sealed
                .iter()
                .filter(|s| s.0 == cx.cur.mm)
                .map(|&(_, a, b)| (child_mm, a, b))
                .collect();
            sh.sealed.extend(seals);
            // ...as are SysV shared-memory attachments (the pages alias).
            sh.ipc.fork_mm(cx.cur.mm, child_mm);
            // A forked address space inherits the parent's arena position (its
            // pages were copied); `CLONE_VM` threads instead share the parent's
            // `mmap_areas[mm]` entry and never reach here.
            let inherited = sh.mmap_areas[cx.cur.mm].clone();
            sh.mmap_areas.push(inherited);
            sh.spaces.push(Arc::new(Mutex::new(cm)));
        }

        let mut child_vcpu = vcpu.fork();
        if stack != 0 {
            child_vcpu.set_sp(stack);
        }
        if flags & CLONE_SETTLS != 0 {
            child_vcpu.set_tls(tls);
            info.fs_base = tls;
        }
        child_vcpu.set_syscall_ret(0); // child returns 0 and advances past the svc
        // A copy-on-write fork (`mem.fork()`) downgraded *this* (parent) address
        // space's pages to read-only behind the running parent vcpu's back. Flush
        // its TLB so the parent's next store faults into `cow_fault` instead of
        // writing through a stale writable entry into the now-shared frame. (Free
        // for a `CLONE_VM` thread, which shares the mm and downgraded nothing.)
        vcpu.flush_tlb();
        // CLONE_PIDFD: allocate a process descriptor for the child in the caller's
        // fd table and write its number to the requested slot (`parent_tid` for
        // legacy `clone`, the `pidfd` field for `clone3`). The pidfd becomes
        // `POLLIN`-readable when the child exits (see `sys_exit`), so a parent can
        // `poll` it instead of catching SIGCHLD. `pollfds` is the innermost lock;
        // acquired after `sh` (which we already hold) it respects the lock order.
        if flags & CLONE_PIDFD != 0 {
            let pidfd_idx = {
                let mut pf = self.pollfds.lock().unwrap();
                let idx = pf.pidfds.len();
                pf.pidfds.push(PidfdInst {
                    target_pid: pid,
                    exited: false,
                    nonblock: false,
                });
                idx
            };
            let pidfd = cx.cur.fds.alloc(Fd::Pidfd(pidfd_idx));
            if ca.pidfd_ptr != 0 {
                let _ = mem.write(ca.pidfd_ptr, &(pidfd as u32).to_le_bytes());
            }
        }
        sh.procs.push(Some(Process {
            vcpu: Some(child_vcpu),
            info,
        }));
        i64::from(pid)
    }

    /// `execve(path, argv, envp)` — replace the process image with a new ELF
    /// read from the mount table (following symlinks). Static and static-PIE
    /// images load directly; a dynamic executable's `PT_INTERP` linker is read
    /// from the same root and loaded alongside it.
    #[allow(clippy::too_many_arguments)]
    fn sys_execve(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        path_ptr: u64,
        argv_ptr: u64,
        envp_ptr: u64,
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some(rel) = read_path(mem, path_ptr) else {
            return err(Errno::EFAULT);
        };
        // `sh` is already held (this handler runs under `dispatch_shared`);
        // acquiring `vfs` here keeps the mandated sh→vfs order.
        let mut vfs = self.vfs.lock().unwrap();
        let Some(abs) = self.resolve_exec(&mut vfs, cx, &rel) else {
            return err(Errno::ENOENT);
        };
        let argv = read_string_array(mem, argv_ptr);
        let envp = read_string_array(mem, envp_ptr);
        self.exec_image(sh, &mut vfs, cx, &abs, argv, envp, vcpu, mem)
    }

    /// `execveat(dirfd, path, argv, envp, flags)` — like `execve` but resolves
    /// `path` relative to `dirfd`, and (with `AT_EMPTY_PATH`) can exec the file
    /// `dirfd` itself refers to.
    #[allow(clippy::too_many_arguments)]
    fn sys_execveat(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        dirfd: i64,
        path_ptr: u64,
        argv_ptr: u64,
        envp_ptr: u64,
        flags: u64,
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        const AT_EMPTY_PATH: u64 = 0x1000;
        let Some(rel) = read_path(mem, path_ptr) else {
            return err(Errno::EFAULT);
        };
        let abs = if rel.is_empty() && flags & AT_EMPTY_PATH != 0 {
            match cx.cur.fds.get(dirfd as i32) {
                Some(Fd::File { path, .. }) => path.clone(),
                _ => return err(Errno::EBADF),
            }
        } else {
            self.resolve_path(cx, dirfd, &rel)
        };
        let argv = read_string_array(mem, argv_ptr);
        let envp = read_string_array(mem, envp_ptr);
        // sh held (dispatch_shared) → acquire vfs in the mandated order.
        let mut vfs = self.vfs.lock().unwrap();
        self.exec_image(sh, &mut vfs, cx, &abs, argv, envp, vcpu, mem)
    }

    /// Load `abs` (following `PT_INTERP` for dynamic executables) into a fresh
    /// address space and reset the vcpu onto it — the shared core of
    /// `execve`/`execveat`. Reads the image from `vfs` and resets the arena in
    /// `sh`, so its caller holds both (sh→vfs).
    #[allow(clippy::too_many_arguments)]
    fn exec_image(
        &self,
        sh: &mut Shared,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        abs: &str,
        argv: Vec<String>,
        envp: Vec<String>,
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        // A `#!` script runs its interpreter instead (Linux's binfmt_script):
        // argv becomes [interpreter, (its optional argument), script path,
        // argv[1..]]. The interpreter may itself be a script, up to 4 levels
        // deep (BINPRM_MAX_RECURSION), as on Linux. `comm` stays the script's
        // name; `/proc/self/exe` is the interpreter that actually runs.
        let script_name = abs.to_string();
        let (mut abs, mut argv) = (abs.to_string(), argv);
        let mut depth = 0;
        let elf = loop {
            let Some(data) = self.read_file(vfs, &abs) else {
                return err(Errno::ENOENT);
            };
            if !data.starts_with(b"#!") {
                break data;
            }
            depth += 1;
            if depth > 4 {
                return err(Errno::ELOOP);
            }
            // Only the first line counts, and only its first 256 bytes.
            let head = &data[2..data.len().min(256)];
            let line = head
                .iter()
                .position(|&b| b == b'\n')
                .map_or(head, |n| &head[..n]);
            let line = String::from_utf8_lossy(line);
            let line = line.trim();
            let (interp, arg) = match line.split_once([' ', '\t']) {
                Some((i, a)) => (i, Some(a.trim()).filter(|a| !a.is_empty())),
                None => (line, None),
            };
            if interp.is_empty() {
                return err(Errno::ENOEXEC);
            }
            let Some(interp_abs) = self.resolve_exec(vfs, cx, interp) else {
                return err(Errno::ENOENT);
            };
            let mut next = vec![interp.to_string()];
            next.extend(arg.map(str::to_string));
            next.push(abs.clone());
            next.extend(argv.into_iter().skip(1));
            argv = next;
            abs = interp_abs;
        };
        let abs = abs.as_str();
        // Reject an obviously non-ELF64 image *before* tearing down the current
        // one, so a bad `execve` leaves the process intact (real semantics) rather
        // than stranded on an empty address space.
        if elf.len() < 64 || elf[0..4] != [0x7f, b'E', b'L', b'F'] || elf[4] != 2 {
            return err(Errno::ENOEXEC);
        }
        // Capture the launch identity for /proc/self/{cmdline,comm} before argv
        // is moved into the spec: cmdline is argv NUL-joined; comm is the new
        // program's basename (truncated to 15), which execve resets (a prior
        // PR_SET_NAME does not survive exec).
        cx.cur.cmdline = cmdline_bytes(&argv);
        cx.cur.comm = comm_from_path(&script_name);
        let spec = ProcessSpec { argv, envp };
        // Point of no return. Record the new program image for `/proc/self/exe`.
        cx.cur.exe = abs.to_string();
        // execve resets every *caught* signal handler to SIG_DFL — the handler
        // addresses point into the image being replaced, so keeping them would
        // jump the new program to garbage on the next signal. Ignored (SIG_IGN)
        // dispositions are preserved, as Linux does; the blocked mask and pending
        // signals survive too. The alternate signal stack does not survive exec.
        for h in &mut cx.cur.handlers {
            if h.handler != 1 {
                // != SIG_IGN
                *h = SigAction::default(); // SIG_DFL, no flags/mask/restorer
            }
        }
        cx.cur.altstack = (0, 0, SS_DISABLE);
        // Close every `FD_CLOEXEC` descriptor (dropping its pipe/socket backing)
        // — this is what execve is *for*, so a process doesn't leak private fds
        // into the program it launches.
        for fd in cx.cur.fds.close_cloexec() {
            self.bump_pipe(&fd, false);
            self.release_fd_locks_with(sh, cx, vfs, &fd);
        }
        self.reap_orphans(sh, cx, vfs);
        // POSIX timers are destroyed by execve (an ITIMER_REAL survives it), as
        // are the per-image registrations: rseq, the robust list, mseal seals
        // and membarrier registrations (they describe the old address space).
        cx.cur.ptimers.clear();
        cx.cur.rseq = None;
        cx.cur.robust_list = 0;
        let mm = cx.cur.mm;
        sh.sealed.retain(|s| s.0 != mm);
        sh.membarrier.remove(&mm);
        // execve detaches every SysV shared-memory segment.
        sh.ipc.detach_mm(mm, mem);
        // Writable shared file mappings die with the old image: flush them to
        // their files now (their bytes are the source of truth) and forget them,
        // or the exit-time flush would write whatever the *new* image later
        // maps at those addresses back over the files.
        if !cx.cur.shared_maps.is_empty() {
            self.flush_shared_maps(vfs, cx, 0, 0, mem);
            cx.cur.shared_maps.clear();
        }
        // Replace the image *in place*: tear down the old page tables (returning
        // their frames to the shared pool) and rebuild within the SAME pool, so
        // the one KVM memslot stays valid and the process just gets a new cr3.
        mem.exec_reset();
        self.pc_gc(mem);
        let loaded = if let Some(interp) = interp_path(&elf) {
            let Some(interp_elf) = self.read_file(vfs, &interp) else {
                return err(Errno::ENOENT); // interpreter missing
            };
            load_dynamic(mem, &elf, &interp_elf, &spec)
        } else {
            load_static(mem, &elf, &spec)
        };
        let Ok(img) = loaded else {
            return err(Errno::ENOEXEC);
        };
        vcpu.reset(img.entry, img.stack_pointer);
        cx.cur.auxv = crate::loader::read_auxv(mem, img.stack_pointer);
        cx.cur.sigtramp = 0;
        let mid = page_down(img.program_break + (img.stack_bottom - img.program_break) / 2);
        cx.cur.brk = img.program_break;
        cx.cur.heap_start = img.program_break;
        cx.cur.heap_limit = mid;
        cx.cur.stack_limit = img.stack_bottom; // stack grows down to here on demand
        // Arena top sits a guard gap below the stack (see STACK_GUARD_GAP).
        let top = arena_top(img.stack_bottom, mid);
        cx.cur.mmap_cursor = top;
        cx.cur.mmap_floor = mid;
        // The image was replaced in place: the arena starts over, free list and all.
        let mm = cx.cur.mm;
        sh.mmap_areas[mm] = Arena::new(top, mid);
        cx.exec_ok = true;
        0
    }

    /// `wait4(pid, wstatus, options, rusage)` — reap a zombie child, honoring the
    /// `pid` filter (`>0` a specific child, `-1` any, `0` the caller's group,
    /// `<-1` group `-pid`) and filling `rusage` with the child's CPU time.
    #[allow(clippy::unused_self, clippy::too_many_arguments)]
    fn sys_wait4(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        pid: i64,
        wstatus: u64,
        options: u64,
        rusage: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const WNOHANG: u64 = 1;
        const WUNTRACED: u64 = 2; // also report a child that job-control-stopped
        const WCONTINUED: u64 = 8; // also report a child that was continued
        let cur = cx.cur.pid;
        let cur_pgid = pgid_of(&cx.cur);
        // Threads (CLONE_THREAD) are not reaped by wait4; only child processes
        // matching the `pid` filter.
        let wanted = |p: &ProcInfo| -> bool {
            if p.ppid != cur || p.is_thread {
                return false;
            }
            match pid {
                p_ if p_ > 0 => i64::from(p.pid) == pid,
                -1 => true,
                0 => pgid_of(p) == cur_pgid,
                _ => i64::from(pgid_of(p)) == -pid, // pid < -1: group -pid
            }
        };
        let mut zombie = None;
        let mut stopped = None; // (pid, stop signal) — an unreported stop
        let mut continued = None; // pid of a continued child
        let mut has_child = false;
        for p in sh.procs.iter().flatten() {
            if wanted(&p.info) {
                has_child = true;
                match p.info.run {
                    RunState::Zombie(code) if zombie.is_none() => {
                        zombie = Some((p.info.pid, code, p.info.cpu_ns));
                    }
                    RunState::Stopped(sig) if !p.info.stop_reported && stopped.is_none() => {
                        stopped = Some((p.info.pid, sig));
                    }
                    _ => {}
                }
                if p.info.continued && continued.is_none() {
                    continued = Some(p.info.pid);
                }
            }
        }
        // A zombie reap takes priority over a stop/continue report.
        if let Some((child, code, child_cpu)) = zombie {
            if wstatus != 0 {
                // WIFEXITED for a normal exit, WIFSIGNALED for a signal death.
                let _ = mem.write(wstatus, &code.wait_status().to_le_bytes());
            }
            if rusage != 0 {
                let _ = mem.write(rusage, &rusage_bytes(child_cpu));
            }
            cx.cur.child_cpu_ns = cx.cur.child_cpu_ns.saturating_add(child_cpu);
            for slot in &mut sh.procs {
                if slot.as_ref().is_some_and(|p| p.info.pid == child) {
                    *slot = None;
                    break;
                }
            }
            return i64::from(child);
        }
        // WUNTRACED: report a stopped child (WIFSTOPPED | WSTOPSIG<<8), latching
        // it so the same stop isn't re-reported; the child is NOT reaped.
        if options & WUNTRACED != 0
            && let Some((child, sig)) = stopped
        {
            if wstatus != 0 {
                let status = ((sig as u32) << 8) | 0x7f;
                let _ = mem.write(wstatus, &status.to_le_bytes());
            }
            for slot in sh.procs.iter_mut().flatten() {
                if slot.info.pid == child {
                    slot.info.stop_reported = true;
                    break;
                }
            }
            return i64::from(child);
        }
        // WCONTINUED: report a continued child (WIFCONTINUED = 0xffff), clearing
        // the latch; the child is NOT reaped.
        if options & WCONTINUED != 0
            && let Some(child) = continued
        {
            if wstatus != 0 {
                let _ = mem.write(wstatus, &0xffffu32.to_le_bytes());
            }
            for slot in sh.procs.iter_mut().flatten() {
                if slot.info.pid == child {
                    slot.info.continued = false;
                    break;
                }
            }
            return i64::from(child);
        }
        if !has_child {
            return err(Errno::ECHILD);
        }
        if options & WNOHANG != 0 {
            return 0;
        }
        cx.block = true; // wait for a child to exit / stop / continue
        0
    }

    /// `waitid(idtype, id, infop, options, rusage)` — the siginfo-based wait.
    /// Reaps a zombie child (or, with `WNOWAIT`, reports without reaping) and
    /// fills a `siginfo_t` instead of `wait4`'s status word.
    #[allow(clippy::too_many_arguments, clippy::unused_self)]
    fn sys_waitid(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        idtype: u64,
        id: i64,
        infop: u64,
        options: u64,
        rusage: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const P_ALL: u64 = 0;
        const P_PID: u64 = 1;
        const P_PGID: u64 = 2;
        const P_PIDFD: u64 = 3;
        const WNOHANG: u64 = 1;
        const WSTOPPED: u64 = 2; // report a job-control-stopped child
        const WEXITED: u64 = 4; // report an exited child
        const WCONTINUED: u64 = 8; // report a continued child
        const WNOWAIT: u64 = 0x0100_0000;
        // __WNOTHREAD | __WALL | __WCLONE: accepted (threads are never waitable
        // children here, so they change nothing).
        const WLINUX: u64 = 0xe000_0000;
        const SIGCHLD: i32 = 17;
        const SIGCONT: i32 = 18;
        const CLD_STOPPED: i32 = 5;
        const CLD_CONTINUED: i32 = 6;
        if options & !(WNOHANG | WSTOPPED | WEXITED | WCONTINUED | WNOWAIT | WLINUX) != 0
            || options & (WEXITED | WSTOPPED | WCONTINUED) == 0
        {
            return err(Errno::EINVAL);
        }
        // P_PIDFD names the child by a pidfd (`pidfd_open`/`CLONE_PIDFD`): the
        // same wait as P_PID on its target, except that an O_NONBLOCK pidfd
        // makes a would-block wait EAGAIN instead of parking.
        let (idtype, id, pidfd_nonblock) = match idtype {
            P_PIDFD => match self.pidfd_target(cx, id as i32) {
                Ok((pid, nonblock)) => (P_PID, i64::from(pid), nonblock),
                Err(e) => return e,
            },
            P_ALL | P_PID | P_PGID => (idtype, id, false),
            _ => return err(Errno::EINVAL),
        };
        let cur = cx.cur.pid;
        let matches_id = |p: &ProcInfo| match idtype {
            P_ALL => true,
            P_PID => i64::from(p.pid) == id,
            P_PGID => i64::from(pgid_of(p)) == id,
            _ => false,
        };
        let mut zombie = None;
        let mut stopped = None; // (pid, stop signal)
        let mut continued = None; // pid
        let mut has_child = false;
        for p in sh.procs.iter().flatten() {
            if p.info.ppid == cur && !p.info.is_thread && matches_id(&p.info) {
                has_child = true;
                match p.info.run {
                    RunState::Zombie(code) if zombie.is_none() && options & WEXITED != 0 => {
                        zombie = Some((p.info.pid, code, p.info.cpu_ns));
                    }
                    RunState::Stopped(sig) if !p.info.stop_reported && stopped.is_none() => {
                        stopped = Some((p.info.pid, sig));
                    }
                    _ => {}
                }
                if p.info.continued && continued.is_none() {
                    continued = Some(p.info.pid);
                }
            }
        }
        // A `siginfo_t` for a child job-control event (CLD_STOPPED/CLD_CONTINUED):
        // si_signo=SIGCHLD, si_code=the event, si_pid=child, si_status=the signal.
        let write_child_si = |mem: &mut GuestMemory, code: i32, child: i32, status: i32| {
            if infop != 0 {
                let mut si = [0u8; 128];
                si[0..4].copy_from_slice(&SIGCHLD.to_le_bytes());
                si[8..12].copy_from_slice(&code.to_le_bytes());
                si[16..20].copy_from_slice(&child.to_le_bytes());
                si[24..28].copy_from_slice(&status.to_le_bytes());
                let _ = mem.write(infop, &si);
            }
        };
        if let Some((child, code, child_cpu)) = zombie {
            if infop != 0 {
                // siginfo_t: si_signo(0)=SIGCHLD(17), si_errno(4)=0,
                // si_code(8)=CLD_EXITED/CLD_KILLED, si_pid(16), si_uid(20),
                // si_status(24)=exit code or the killing signal.
                let mut si = [0u8; 128];
                si[0..4].copy_from_slice(&17i32.to_le_bytes());
                si[8..12].copy_from_slice(&code.si_code().to_le_bytes());
                si[16..20].copy_from_slice(&child.to_le_bytes());
                si[24..28].copy_from_slice(&code.si_status().to_le_bytes());
                let _ = mem.write(infop, &si);
            }
            if rusage != 0 {
                let _ = mem.write(rusage, &rusage_bytes(child_cpu));
            }
            // WNOWAIT reports without reaping — so it also doesn't collect the
            // child's CPU into the parent (that happens when it's actually reaped).
            if options & WNOWAIT == 0 {
                cx.cur.child_cpu_ns = cx.cur.child_cpu_ns.saturating_add(child_cpu);
                for slot in &mut sh.procs {
                    if slot.as_ref().is_some_and(|p| p.info.pid == child) {
                        *slot = None;
                        break;
                    }
                }
            }
            return 0;
        }
        // WSTOPPED: report a job-control-stopped child. WNOWAIT leaves the stop
        // un-latched so a later wait sees it again; otherwise latch it. Not reaped.
        if options & WSTOPPED != 0
            && let Some((child, sig)) = stopped
        {
            write_child_si(mem, CLD_STOPPED, child, sig);
            if options & WNOWAIT == 0 {
                for slot in sh.procs.iter_mut().flatten() {
                    if slot.info.pid == child {
                        slot.info.stop_reported = true;
                        break;
                    }
                }
            }
            return 0;
        }
        // WCONTINUED: report a continued child (si_status = SIGCONT). WNOWAIT
        // leaves the "continued" latch set; otherwise clear it. Not reaped.
        if options & WCONTINUED != 0
            && let Some(child) = continued
        {
            write_child_si(mem, CLD_CONTINUED, child, SIGCONT);
            if options & WNOWAIT == 0 {
                for slot in sh.procs.iter_mut().flatten() {
                    if slot.info.pid == child {
                        slot.info.continued = false;
                        break;
                    }
                }
            }
            return 0;
        }
        if !has_child {
            return err(Errno::ECHILD);
        }
        if options & WNOHANG != 0 {
            // Nothing waitable: Linux zeroes the siginfo (callers tell "no
            // child changed state" from a report by `si_pid == 0`).
            if infop != 0 {
                let _ = mem.write(infop, &[0u8; 128]);
            }
            return 0;
        }
        if pidfd_nonblock {
            return err(Errno::EAGAIN);
        }
        cx.block = true;
        0
    }

    /// `clone3(cl_args, size)` — the modern `clone`. Reads the full `clone_args`
    /// struct and forwards it to the shared [`Kernel::do_clone`] core. Unlike
    /// legacy `clone`, the termination signal and pidfd pointer are their own
    /// fields (not packed into `flags`/`parent_tid`), and `stack`+`stack_size`
    /// give the region rather than a pre-computed stack pointer.
    fn sys_clone3(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        args_ptr: u64,
        size: u64,
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        // The struct is versioned by size: 64 bytes (v0) through 88 (with the
        // set_tid/cgroup fields). A short size is malformed (EINVAL).
        if size < 64 {
            return err(Errno::EINVAL);
        }
        // struct clone_args: flags@0, pidfd@8, child_tid@16, parent_tid@24,
        // exit_signal@32, stack@40, stack_size@48, tls@56, set_tid@64,
        // set_tid_size@72, cgroup@80.
        let rd = |off: u64| mem.read_u64(args_ptr + off).unwrap_or(0);
        let flags = rd(0);
        let pidfd_ptr = rd(8);
        let child_tid = rd(16);
        let parent_tid = rd(24);
        let exit_signal = rd(32);
        let stack = rd(40);
        let stack_size = rd(48);
        let tls = rd(56);
        // set_tid[] (request specific pids per namespace) and cgroup are read for
        // completeness but not honored: nixvm assigns pids from its own counter
        // and models a single global cgroup, so a set_tid request cannot be
        // satisfied — Linux would return EINVAL if the requested pid is taken, but
        // since we never reuse pids the field is simply ignored. `CLONE_INTO_CGROUP`
        // (whose target lives in `cgroup`) is accepted as a no-op like the other
        // namespace/context flags.
        let set_tid = if size >= 80 { rd(64) } else { 0 };
        let set_tid_size = if size >= 80 { rd(72) } else { 0 };
        let _cgroup = if size >= 88 { rd(80) } else { 0 };
        let _ = (set_tid, set_tid_size);
        // The child SP is the top of the provided stack region (grows down); a
        // zero stack means "inherit the caller's" (do_clone leaves SP untouched).
        let sp = if stack == 0 {
            0
        } else {
            stack.wrapping_add(stack_size)
        };
        let ca = CloneArgs {
            // clone3's flags never carry the exit signal in their low byte.
            flags,
            stack_ptr: sp,
            parent_tid,
            child_tid,
            tls,
            exit_signal,
            pidfd_ptr,
        };
        self.do_clone(sh, cx, &ca, vcpu, mem)
    }

    /// `close_range(first, last, flags)` — close every open fd in `[first,
    /// last]`. `flags` (e.g. `CLOSE_RANGE_CLOEXEC`) is ignored beyond the
    /// close itself.
    /// `close_range(first, last, flags)`: close every open descriptor in
    /// `[first, last]` — or, with `CLOSE_RANGE_CLOEXEC`, just mark them
    /// close-on-exec; `CLOSE_RANGE_UNSHARE` first gives the caller a private
    /// copy of a shared descriptor table (so siblings keep theirs).
    fn sys_close_range(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        first: u64,
        last: u64,
        flags: u64,
    ) -> i64 {
        const CLOSE_RANGE_UNSHARE: u64 = 1 << 1;
        const CLOSE_RANGE_CLOEXEC: u64 = 1 << 2;
        let (first, last) = (first as u32, last as u32);
        if flags & !(CLOSE_RANGE_UNSHARE | CLOSE_RANGE_CLOEXEC) != 0 || first > last {
            return err(Errno::EINVAL);
        }
        if flags & CLOSE_RANGE_UNSHARE != 0 {
            const CLONE_FILES: u64 = 0x400;
            let r = self.sys_unshare(sh, cx, CLONE_FILES);
            if r < 0 {
                return r;
            }
        }
        let targets: Vec<i32> = cx
            .cur
            .fds
            .iter()
            .map(|(n, _)| n)
            .filter(|&n| (first..=last).contains(&(n as u32)))
            .collect();
        for n in targets {
            if flags & CLOSE_RANGE_CLOEXEC != 0 {
                cx.cur.fds.set_cloexec(n, true);
            } else {
                let closed = cx.cur.fds.get(n).cloned();
                let _ = self.sys_close(cx, n);
                if let Some(f) = closed {
                    self.release_fd_locks(sh, cx, &f);
                }
            }
        }
        self.reap_orphans_locked(sh, cx);
        0
    }

    /// `getresuid`/`getresgid` — write `(real, effective, saved)` = `(0,0,0)`
    /// (this VM is single-user root).
    #[allow(clippy::unused_self)] // method form keeps the dispatch table uniform
    fn sys_getres_id(&self, ids: [u32; 3], a: u64, b: u64, c: u64, mem: &mut GuestMemory) -> i64 {
        for (p, v) in [(a, ids[0]), (b, ids[1]), (c, ids[2])] {
            if p != 0 && mem.write(p, &v.to_le_bytes()).is_err() {
                return err(Errno::EFAULT);
            }
        }
        0
    }

    /// `setpgid(pid, pgid)` — set the process group of `pid` (0 = self) to
    /// `pgid` (0 = the target's own pid). Only the current task is tracked.
    #[allow(clippy::unused_self)]
    fn sys_setpgid(&self, sh: &mut Shared, cx: &mut ServiceCtx, pid: i32, pgid: i32) -> i64 {
        if pgid < 0 {
            return err(Errno::EINVAL);
        }
        if pid == 0 || pid == cx.cur.pid {
            cx.cur.pgid = if pgid == 0 { cx.cur.pid } else { pgid };
            return 0;
        }
        // Setting a child's process group (what a shell does to build a job): the
        // target must be an existing process. `pgid == 0` means the target's pid.
        for slot in sh.procs.iter_mut().flatten() {
            if slot.info.pid == pid {
                slot.info.pgid = if pgid == 0 { pid } else { pgid };
                return 0;
            }
        }
        err(Errno::ESRCH)
    }

    /// `getpgid(pid)` — the process group of `pid` (0 = self).
    #[allow(clippy::unused_self)]
    fn sys_getpgid(&self, sh: &mut Shared, cx: &mut ServiceCtx, pid: i32) -> i64 {
        if pid == 0 || pid == cx.cur.pid {
            return i64::from(pgid_of(&cx.cur));
        }
        for p in sh.procs.iter().flatten() {
            if p.info.pid == pid {
                return i64::from(pgid_of(&p.info));
            }
        }
        err(Errno::ESRCH)
    }

    /// `setsid()` — start a new session: sid = pgid = the caller's pid.
    #[allow(clippy::unused_self)]
    fn sys_setsid(&self, cx: &mut ServiceCtx) -> i64 {
        cx.cur.sid = cx.cur.pid;
        cx.cur.pgid = cx.cur.pid;
        i64::from(cx.cur.pid)
    }

    /// `getsid(pid)` — the session id of `pid` (0 = self).
    #[allow(clippy::unused_self)]
    fn sys_getsid(&self, sh: &mut Shared, cx: &mut ServiceCtx, pid: i32) -> i64 {
        if pid == 0 || pid == cx.cur.pid {
            return i64::from(if cx.cur.sid == 0 {
                cx.cur.pid
            } else {
                cx.cur.sid
            });
        }
        for p in sh.procs.iter().flatten() {
            if p.info.pid == pid {
                return i64::from(if p.info.sid == 0 {
                    p.info.pid
                } else {
                    p.info.sid
                });
            }
        }
        err(Errno::ESRCH)
    }

    /// `statx(dirfd, path, flags, mask, buf)` — the modern `stat`. Fills the
    /// basic-stats fields of `struct statx` from the resolved node's [`Attrs`].
    #[allow(clippy::too_many_arguments)]
    fn sys_statx(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        dirfd: i64,
        path_ptr: u64,
        flags: u64,
        buf: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        // AT_SYMLINK_NOFOLLOW | AT_NO_AUTOMOUNT | AT_EMPTY_PATH and the
        // AT_STATX_SYNC_TYPE field (FORCE_SYNC / DONT_SYNC — nothing to sync).
        const VALID_FLAGS: u64 = 0x100 | 0x800 | 0x1000 | 0x6000;
        const AT_STATX_SYNC_TYPE: u64 = 0x6000;
        if flags & !VALID_FLAGS != 0 || flags & AT_STATX_SYNC_TYPE == AT_STATX_SYNC_TYPE {
            return err(Errno::EINVAL);
        }
        let a = match self.stat_at(vfs, cx, dirfd, path_ptr, flags, mem) {
            Ok(a) => a,
            Err(e) => return e,
        };
        let buf_bytes = stat::encode_statx(&a);
        if mem.write(buf, &buf_bytes).is_err() {
            return err(Errno::EFAULT);
        }
        0
    }

    /// `memfd_create(name, flags)` — an anonymous, initially-empty read/write
    /// file: an *orphan* (see [`orphan`]) at the root, so it is in no
    /// directory and vanishes with its last descriptor, while its pages stay
    /// alive in the page cache for whoever still maps it. `MFD_CLOEXEC` and
    /// `MFD_ALLOW_SEALING` (else the memfd starts sealed against sealing) are
    /// honored; `MFD_EXEC`/`MFD_NOEXEC_SEAL` (`F_SEAL_EXEC`) are accepted;
    /// hugetlb memfds are refused (`EINVAL`, no huge pages).
    #[allow(clippy::unused_self)]
    fn sys_memfd_create(
        &self,
        sh: &mut Shared,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        name_ptr: u64,
        flags: u64,
        mem: &GuestMemory,
    ) -> i64 {
        const MFD_CLOEXEC: u64 = 1;
        const MFD_ALLOW_SEALING: u64 = 2;
        const MFD_HUGETLB: u64 = 4;
        const MFD_NOEXEC_SEAL: u64 = 8;
        const MFD_EXEC: u64 = 0x10;
        if flags & !(MFD_CLOEXEC | MFD_ALLOW_SEALING | MFD_NOEXEC_SEAL | MFD_EXEC) != 0
            || flags & MFD_HUGETLB != 0
            || (flags & MFD_EXEC != 0 && flags & MFD_NOEXEC_SEAL != 0)
        {
            return err(Errno::EINVAL);
        }
        let Some(name) = read_path(mem, name_ptr) else {
            return err(Errno::EFAULT);
        };
        if name.len() > 249 {
            return err(Errno::EINVAL);
        }
        sh.memfd_seq += 1;
        let path = format!("/{}memfd.{}", orphan::ORPHAN_PREFIX, sh.memfd_seq);
        if vfs.create(&path, 0o777).is_err() {
            return err(Errno::ENOSPC);
        }
        self.orphans.lock().unwrap().insert(path.clone());
        // MFD_NOEXEC_SEAL implies a sealable memfd with F_SEAL_EXEC set.
        self.memfd_register(&path, flags & (MFD_ALLOW_SEALING | MFD_NOEXEC_SEAL) != 0);
        let fd = cx.cur.fds.alloc(Fd::File {
            path,
            offset: FileOffset::new(0),
            readable: true,
            writable: true,
        });
        cx.cur.fds.set_cloexec(fd, flags & MFD_CLOEXEC != 0);
        i64::from(fd)
    }

    /// `inotify_init1(flags)` stub — an eventfd-backed descriptor that is always
    /// empty (no filesystem events are delivered). Programs get a valid fd and
    /// simply never see events, a safe degradation for optional watching.
    ///
    /// `IN_NONBLOCK` must be honored: the fd never becomes readable, so a
    /// *blocking* `read` on it would park forever — and for a single-threaded
    /// watcher that is the whole VM, tripping the deadlock detector. A
    /// non-blocking reader instead gets `EAGAIN`, which is exactly how `fs.watch`
    /// / chokidar / Bun drive an inotify fd (non-blocking + epoll/poll).
    #[allow(clippy::unused_self)]
    fn sys_inotify_init1(&self, pf: &mut PollFds, cx: &mut ServiceCtx, flags: u64) -> i64 {
        const IN_NONBLOCK: u64 = 0o4000; // == O_NONBLOCK
        const IN_CLOEXEC: u64 = 0o2000000; // == O_CLOEXEC
        let idx = pf.eventfds.len();
        pf.eventfds.push(EventFdInst::default());
        if flags & IN_NONBLOCK != 0 {
            pf.set_nonblock(&Fd::Eventfd(idx), true);
        }
        let fd = cx.cur.fds.alloc(Fd::Eventfd(idx));
        cx.cur.fds.set_cloexec(fd, flags & IN_CLOEXEC != 0);
        i64::from(fd)
    }

    /// `exit` — terminate just this task: run its `CLONE_CHILD_CLEARTID`
    /// notification (so a joiner wakes), close its fds (so pipe peers see EOF),
    /// and become a zombie until reaped.
    fn sys_exit(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        code: i32,
        mem: &mut GuestMemory,
    ) -> i64 {
        self.exit_task(sh, cx, ExitCause::Exited(code & 0xff), mem)
    }

    /// Tear down the current task and make it a zombie with `cause`: the
    /// common tail of `exit`, `exit_group` and death by a fatal signal. Closes
    /// its fds (so pipe/socket peers see EOF), releases its memory, wakes a
    /// `CLONE_CHILD_CLEARTID` waiter, and notifies/reparents as below.
    fn exit_task(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        cause: ExitCause,
        mem: &mut GuestMemory,
    ) -> i64 {
        // Flush any un-munmap'd writable shared file mappings first. `sh` is
        // held; scope `vfs` to just the flush (sh→vfs order), dropping it before
        // the rest of teardown.
        if !cx.cur.shared_maps.is_empty() {
            let mut vfs = self.vfs.lock().unwrap();
            self.flush_shared_maps(&mut vfs, cx, 0, 0, mem);
        }
        let ctid = cx.cur.clear_child_tid;
        let mm = cx.cur.mm;
        if ctid != 0 {
            let _ = mem.write(ctid, &0u32.to_le_bytes());
            // The kernel's CLONE_CHILD_CLEARTID wake is a shared-key wake.
            let key = Self::futex_key(cx, mem, ctid, false);
            Self::futex_wake_key(sh, key, i64::from(i32::MAX), u32::MAX);
        }
        // Only close the fds when this is the last user of the shared table: a
        // thread exiting while siblings live must leave the (`CLONE_FILES`)
        // table — and its pipe/socket references — intact. `check_in_files`
        // stores whatever remains back for the survivors.
        let files = cx.cur.files;
        let others_share = sh
            .procs
            .iter()
            .flatten()
            .any(|p| p.info.files == files && !matches!(p.info.run, RunState::Zombie(_)));
        if !others_share {
            for fd in cx.cur.fds.drain() {
                self.bump_pipe(&fd, false);
                self.release_fd_locks(sh, cx, &fd);
            }
            self.reap_orphans_locked(sh, cx);
        }
        // Last task of this address space: return its frames to the shared pool
        // (page tables + private data pages), so a long-lived process tree does
        // not accumulate dead processes' frames. Threads sharing the mm keep it.
        if !self.has_cowaiter(sh, mm) {
            sh.ipc.detach_mm(mm, mem);
            mem.release();
            self.pc_gc(mem);
        }
        // The thread group's last task applies its SEM_UNDO adjustments (and
        // wakes anyone those unblock).
        let tgid = cx.cur.tgid;
        if !sh
            .procs
            .iter()
            .flatten()
            .any(|p| p.info.tgid == tgid && !matches!(p.info.run, RunState::Zombie(_)))
            && sh.ipc.exit_group(tgid)
        {
            sh.unpark_all();
        }
        if !sh
            .procs
            .iter()
            .flatten()
            .any(|p| p.info.tgid == tgid && !matches!(p.info.run, RunState::Zombie(_)))
        {
            self.release_process_locks(sh, tgid);
        }
        cx.cur.run = RunState::Zombie(cause);
        // The signal a terminating child sends its parent. A plain fork uses
        // SIGCHLD, but `clone` can request any signal (or none). For a process
        // (non-thread) it is the child's own `exit_signal`. A thread's individual
        // death sends nothing — UNLESS the whole group is going down (exit_group
        // already zombified the leader), in which case the parent must receive the
        // *leader's* exit signal, not the (signal-less) thread's; find the leader
        // and use its signal only if it, too, is now a zombie.
        let exit_sig: u64 = if cx.cur.is_thread {
            sh.procs
                .iter()
                .flatten()
                .find(|p| p.info.pid == cx.cur.tgid)
                .filter(|p| matches!(p.info.run, RunState::Zombie(_)))
                .map_or(0, |p| u64::from(p.info.exit_signal))
        } else {
            u64::from(cx.cur.exit_signal)
        };
        // Notify the parent: post `exit_sig` (if any) and unpark it so a
        // `wait`/`sigsuspend` blocked for it re-checks and reaps this zombie. The
        // unpark is unconditional even when no signal is sent (a thread's death,
        // or `clone` with exit signal 0), so a parked `wait4` still re-runs and
        // reaps. A parent that left SIGCHLD at its default disposition just ignores
        // the signal (it is in the default-ignored set); one with a handler (the
        // shell) gets it delivered. `exit_group` funnels through here for the
        // current task, so this covers both exit paths.
        let ppid = cx.cur.ppid;
        let me = cx.cur.pid;
        // Orphans go to the nearest living ancestor that declared itself a
        // child subreaper (PR_SET_CHILD_SUBREAPER), else to init.
        let new_parent = {
            let mut up = cx.cur.ppid;
            let mut found = 1;
            for _ in 0..64 {
                let Some(p) = sh.procs.iter().flatten().find(|p| p.info.pid == up) else {
                    break;
                };
                if p.info.pr.child_subreaper && !matches!(p.info.run, RunState::Zombie(_)) {
                    found = up;
                    break;
                }
                up = p.info.ppid;
            }
            found
        };
        for slot in sh.procs.iter_mut().flatten() {
            if slot.info.pid == ppid {
                if (1..=64).contains(&exit_sig) {
                    slot.info.pending |= 1u64 << (exit_sig - 1);
                }
                slot.info.parked = false;
            }
            // Reparent our children (to a subreaper or init): an orphan must
            // not keep its dead parent's pid as `getppid()`, and its eventual
            // zombie must be reaped rather than stranding forever. A child that
            // armed PR_SET_PDEATHSIG gets that signal now that its parent died.
            if slot.info.ppid == me && slot.info.pid != me {
                slot.info.ppid = new_parent;
                let ds = slot.info.pdeathsig;
                if (1..=64).contains(&ds) {
                    slot.info.pending |= 1u64 << (ds - 1);
                    slot.info.parked = false;
                }
            }
        }
        // Make any `CLONE_PIDFD` descriptor pointing at this task readable, so a
        // parent polling the pidfd (rather than catching the exit signal) wakes.
        // Only a whole process becoming reapable satisfies a pidfd — an individual
        // thread's death does not (its group leader is still alive), so skip the
        // mark for a thread whose leader has not also exited.
        let group_gone = !cx.cur.is_thread
            || sh
                .procs
                .iter()
                .flatten()
                .find(|p| p.info.pid == cx.cur.tgid)
                .is_none_or(|p| matches!(p.info.run, RunState::Zombie(_)));
        if group_gone {
            let mut pf = self.pollfds.lock().unwrap();
            for pfd in &mut pf.pidfds {
                if pfd.target_pid == me || pfd.target_pid == cx.cur.tgid {
                    pfd.exited = true;
                }
            }
        }
        0
    }

    /// `exit_group` — terminate the whole thread group: this task plus every
    /// sibling sharing our `tgid`. Each dying task closes its fds; the running
    /// task also runs its `CLONE_CHILD_CLEARTID` notification.
    fn sys_exit_group(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        code: i32,
        mem: &mut GuestMemory,
    ) -> i64 {
        self.exit_group_with(sh, cx, ExitCause::Exited(code & 0xff), mem)
    }

    /// Terminate the current task's whole thread group with a fatal `sig`
    /// (a default-action signal, or an unhandled fault): the same teardown as
    /// `exit_group` — without it a killed process kept its fds open, so e.g. a
    /// pipe's reader never saw EOF and the shell hung — recorded as
    /// signal-terminated for `wait`. Must be called without `shared` held.
    pub(super) fn die_of_signal(&self, cx: &mut ServiceCtx, sig: u32, mem: &mut GuestMemory) {
        let mut sh = self.shared.lock().unwrap();
        self.exit_group_with(&mut sh, cx, ExitCause::Signaled(sig as i32), mem);
    }

    /// `exit_group` with an explicit exit cause (see [`Self::die_of_signal`]).
    fn exit_group_with(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        cause: ExitCause,
        mem: &mut GuestMemory,
    ) -> i64 {
        // Flush any un-munmap'd writable shared file mappings first (sh→vfs,
        // scoped so the tail `sys_exit` can re-acquire vfs without a re-lock).
        if !cx.cur.shared_maps.is_empty() {
            let mut vfs = self.vfs.lock().unwrap();
            self.flush_shared_maps(&mut vfs, cx, 0, 0, mem);
        }
        let tgid = cx.cur.tgid;
        // Zombify every sibling and note the distinct fd-table ids they used.
        // Their `info.fds` are placeholders — the real tables live in
        // `file_tables` (each shared table drained once, below).
        let mut files_ids: Vec<usize> = Vec::new();
        for slot in &mut sh.procs {
            let Some(p) = slot.as_mut() else { continue };
            if p.info.tgid != tgid || matches!(p.info.run, RunState::Zombie(_)) {
                continue;
            }
            if !files_ids.contains(&p.info.files) {
                files_ids.push(p.info.files);
            }
            p.info.run = RunState::Zombie(cause);
        }
        // Close each distinct table's fds (`bump_pipe` briefly takes `pipes`
        // for a pipe fd or `net` for a socket fd — after `sh`, which this holds
        // — so the fds are collected first, then bumped after the `sh.procs`
        // borrow ends). The current task's table
        // is checked out into `cur.fds` (its slot is `None`), so it's skipped
        // here and closed by the `sys_exit` tail call.
        let mut to_close: Vec<Fd> = Vec::new();
        for f in files_ids {
            if let Some(Some(t)) = sh.file_tables.get_mut(f) {
                to_close.extend(t.drain());
            }
        }
        for fd in to_close {
            self.bump_pipe(&fd, false);
        }
        // `cx.cur` is this task, taken out of the table for its slice.
        self.exit_task(sh, cx, cause, mem)
    }

    // ---- files & fds ------------------------------------------------------

    /// `write(fd, buf, count)` — stdio sinks (fd 1/2), files, and pipes.
    /// `write(fd, buf, count)`. **fd-polymorphic**: a file write touches only
    /// the mount table (holds just `vfs`); a socket write holds just `net`; a
    /// pipe write holds just `pipes`; every other target (stdout/stderr/eventfd)
    /// lives in `shared` (holds just `sh`). The fd type is read from `cx`
    /// *without a lock*, then exactly one of the four locks is taken — never
    /// more than one — so a file write and another task's non-FS syscall run
    /// concurrently.
    fn sys_write(
        &self,
        cx: &mut ServiceCtx,
        fd: u64,
        buf: u64,
        count: u64,
        mem: &GuestMemory,
    ) -> i64 {
        match cx.cur.fds.get(fd as i32) {
            Some(Fd::File { .. }) => {
                let mut vfs = self.vfs.lock().unwrap();
                self.write_file_fd(&mut vfs, cx, fd, buf, count, mem)
            }
            Some(Fd::Socket { .. }) => {
                let mut net = self.net.lock().unwrap();
                self.write_socket_fd(&mut net, cx, fd, buf, count, mem)
            }
            Some(Fd::PipeWrite(..)) => {
                let mut pipes = self.pipes.lock().unwrap();
                self.write_pipe_fd(&mut pipes, cx, fd, buf, count, mem)
            }
            Some(Fd::Eventfd(..)) => {
                let mut pf = self.pollfds.lock().unwrap();
                self.write_pollfd_fd(&mut pf, cx, fd, buf, count, mem)
            }
            Some(Fd::PtyMaster(..) | Fd::PtySlave(..)) => {
                self.write_pty_fd(cx, fd, buf, count, mem)
            }
            _ => {
                let mut sh = self.shared.lock().unwrap();
                self.write_shared_fd(&mut sh, cx, fd, buf, count, mem)
            }
        }
    }

    /// `write` to a pty end: master writes are terminal *input* (line
    /// discipline), slave writes are terminal *output* (post-processing). All
    /// bytes are accepted (nixvm's pty buffers are unbounded).
    fn write_pty_fd(
        &self,
        cx: &mut ServiceCtx,
        fd: u64,
        buf: u64,
        count: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let f = cx.cur.fds.get(fd as i32).cloned();
        let Ok(data) = mem.read_vec(buf, count as usize) else {
            return err(Errno::EFAULT);
        };
        // Run the line discipline under the `ptys` lock, then release it before
        // taking `sh` to deliver any `ISIG`-generated signals (sh sorts *before*
        // ptys, so it must never be acquired while ptys is held).
        let signals = {
            let mut ptys = self.ptys.lock().unwrap();
            match f {
                Some(Fd::PtyMaster(n)) => ptys.master_write(n, &data),
                Some(Fd::PtySlave(n)) => {
                    ptys.slave_write(n, &data);
                    (Vec::new(), 0)
                }
                _ => return err(Errno::EBADF),
            }
        };
        let (sigs, pgrp) = signals;
        if !sigs.is_empty() && pgrp != 0 {
            let mut sh = self.shared.lock().unwrap();
            for sig in sigs {
                self.signal_pgrp(&mut sh, cx, pgrp, sig);
            }
        }
        data.len() as i64
    }

    /// Post signal `sig` to every task in process group `pgrp` (the pty
    /// foreground-group `^C`/`^\`/`^Z` path) and un-park them so a blocked
    /// `read`/`wait` re-checks. The signalling task's own `cur` is out of
    /// `sh.procs` during its slice, so it is handled separately.
    #[allow(clippy::unused_self)]
    fn signal_pgrp(&self, sh: &mut Shared, cx: &mut ServiceCtx, pgrp: i32, sig: u32) {
        if sig == 0 || u64::from(sig) > signal::NSIG {
            return;
        }
        let bit = 1u64 << (sig - 1);
        if pgid_of(&cx.cur) == pgrp {
            cx.cur.pending |= bit;
        }
        for slot in sh.procs.iter_mut().flatten() {
            if pgid_of(&slot.info) == pgrp {
                slot.info.pending |= bit;
                slot.info.parked = false;
            }
        }
    }

    /// The `Fd::Eventfd` arm of [`Self::sys_write`]/[`Self::sys_writev`]: add to
    /// the eventfd counter. `pollfds`-only (the innermost lock). A full counter
    /// on a blocking eventfd sets the block flag and returns 0 (the caller drops
    /// the lock and re-traps) — it never blocks in place holding the lock.
    fn write_pollfd_fd(
        &self,
        pf: &mut PollFds,
        cx: &mut ServiceCtx,
        fd: u64,
        buf: u64,
        count: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let Some(Fd::Eventfd(i)) = cx.cur.fds.get(fd as i32).cloned() else {
            return err(Errno::EBADF);
        };
        let Ok(data) = mem.read_vec(buf, count as usize) else {
            return err(Errno::EFAULT);
        };
        self.write_eventfd(pf, cx, i, &data)
    }

    /// The `Fd::Socket` arm of [`Self::sys_write`]/[`Self::sys_writev`]: send
    /// `count` bytes on the socket. `net`-only.
    fn write_socket_fd(
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
        fd: u64,
        buf: u64,
        count: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let Some(Fd::Socket { sock, end }) = cx.cur.fds.get(fd as i32).cloned() else {
            return err(Errno::EBADF);
        };
        let Ok(data) = mem.read_vec(buf, count as usize) else {
            return err(Errno::EFAULT);
        };
        self.write_socket(net, cx, sock, end, &data, false)
    }

    /// The `Fd::PipeWrite` arm of [`Self::sys_write`]/[`Self::sys_writev`]:
    /// append `count` bytes to the pipe. `pipes`-only.
    fn write_pipe_fd(
        &self,
        pipes: &mut [Pipe],
        cx: &mut ServiceCtx,
        fd: u64,
        buf: u64,
        count: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let Some(Fd::PipeWrite(i)) = cx.cur.fds.get(fd as i32).cloned() else {
            return err(Errno::EBADF);
        };
        let Ok(data) = mem.read_vec(buf, count as usize) else {
            return err(Errno::EFAULT);
        };
        self.write_pipe(pipes, cx, i, &data, false)
    }

    /// The `Fd::File` arm of [`Self::sys_write`]: write `count` bytes at the
    /// fd's offset and advance it. `vfs`-only.
    #[allow(clippy::unused_self)]
    fn write_file_fd(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        fd: u64,
        buf: u64,
        count: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let Some(Fd::File {
            path,
            offset: ofs,
            writable,
            ..
        }) = cx.cur.fds.get(fd as i32).cloned()
        else {
            return err(Errno::EBADF);
        };
        let offset = ofs.get();
        if !writable {
            return err(Errno::EBADF); // fd opened O_RDONLY
        }
        let Ok(data) = mem.read_vec(buf, count as usize) else {
            return err(Errno::EFAULT);
        };
        // O_APPEND: every write goes to (the current) end of file, ignoring the
        // fd offset, then the offset lands past what was written.
        let write_off = if cx.cur.fds.is_append(fd as i32) {
            vfs.stat(&path).map_or(offset, |a| a.size)
        } else {
            offset
        };
        match self.vfs_write(vfs, &path, write_off, &data) {
            Ok(n) => {
                ofs.set(write_off + n as u64);
                n as i64
            }
            Err(e) => io_errno(&e),
        }
    }

    /// The non-`File`, non-`Socket`, non-`PipeWrite`, non-`Eventfd` arms of
    /// [`Self::sys_write`] (stdout/stderr), backed by `shared`. Sockets go
    /// through [`Self::write_socket_fd`] under `net`; pipes through
    /// [`Self::write_pipe_fd`] under `pipes`; eventfds through
    /// [`Self::write_pollfd_fd`] under `pollfds`.
    #[allow(clippy::unused_self)]
    fn write_shared_fd(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        fd: u64,
        buf: u64,
        count: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let Ok(data) = mem.read_vec(buf, count as usize) else {
            return err(Errno::EFAULT);
        };
        // fd 1/2 fall back to the host sinks only when still the standard stream.
        match cx.cur.fds.get(fd as i32).cloned() {
            Some(Fd::Stdout) => match sh.stdout.write_all(&data) {
                Ok(()) => count as i64,
                Err(_) => err(Errno::EIO),
            },
            Some(Fd::Stderr) => match sh.stderr.write_all(&data) {
                Ok(()) => count as i64,
                Err(_) => err(Errno::EIO),
            },
            _ => err(Errno::EBADF),
        }
    }

    /// `read(fd, buf, count)` — stdin, files, and pipes. **fd-polymorphic**,
    /// exactly like [`Self::sys_write`]: a file read holds only `vfs`, a socket
    /// read only `net`, a pipe read only `pipes`, every other source only `sh`.
    fn sys_read(
        &self,
        cx: &mut ServiceCtx,
        fd: u64,
        buf: u64,
        count: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        match cx.cur.fds.get(fd as i32) {
            Some(Fd::File { .. }) => {
                let mut vfs = self.vfs.lock().unwrap();
                self.read_file_fd(&mut vfs, cx, fd, buf, count, mem)
            }
            Some(Fd::Socket { .. }) => {
                let mut net = self.net.lock().unwrap();
                self.read_socket_fd(&mut net, cx, fd, buf, count, mem)
            }
            Some(Fd::PipeRead(..)) => {
                let mut pipes = self.pipes.lock().unwrap();
                self.read_pipe_fd(&mut pipes, cx, fd, buf, count, mem)
            }
            Some(Fd::Eventfd(..) | Fd::Timerfd(..) | Fd::Signalfd(..)) => {
                let mut pf = self.pollfds.lock().unwrap();
                self.read_pollfd_fd(&mut pf, cx, fd, buf, count, mem)
            }
            Some(Fd::PtyMaster(..) | Fd::PtySlave(..)) => self.read_pty_fd(cx, fd, buf, count, mem),
            // A pidfd carries no data — `read` is EINVAL (it is only pollable).
            Some(Fd::Pidfd(..)) => err(Errno::EINVAL),
            _ => {
                let mut sh = self.shared.lock().unwrap();
                self.read_shared_fd(&mut sh, cx, fd, buf, count, mem)
            }
        }
    }

    /// `read` from a pty end: master reads terminal output, slave reads terminal
    /// input (whole canonical lines when `ICANON`). Empty with the other end
    /// still open blocks (or `EAGAIN` if `O_NONBLOCK`); empty with it closed is
    /// EOF (0).
    fn read_pty_fd(
        &self,
        cx: &mut ServiceCtx,
        fd: u64,
        buf: u64,
        count: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let f = cx.cur.fds.get(fd as i32).cloned();
        // A slave read in *non-canonical* mode is governed by VMIN/VTIME; the
        // master and canonical-slave paths keep the simple block-until-ready
        // behavior. Without VMIN/VTIME, a VMIN=0 read blocked forever instead of
        // returning after the VTIME timeout (a whole-VM hang for a lone reader).
        if let Some(Fd::PtySlave(n)) = f {
            let (canon, vmin, vtime_ds) = self.ptys.lock().unwrap().slave_read_params(n);
            if !canon {
                return self.read_pty_slave_noncanon(cx, n, vmin, vtime_ds, buf, count, mem);
            }
        }
        let (res, nonblock) = {
            let mut ptys = self.ptys.lock().unwrap();
            match f {
                Some(Fd::PtyMaster(n)) => (
                    ptys.master_read(n, count as usize),
                    ptys.is_nonblock(n, true),
                ),
                Some(Fd::PtySlave(n)) => (
                    ptys.slave_read(n, count as usize),
                    ptys.is_nonblock(n, false),
                ),
                _ => return err(Errno::EBADF),
            }
        };
        match res {
            None => {
                if nonblock {
                    return err(Errno::EAGAIN);
                }
                // Block; the dispatcher interrupts this with EINTR / SA_RESTART if
                // a caught signal (e.g. `^C` → SIGINT) is pending for this task.
                cx.block = true;
                0
            }
            Some(data) if data.is_empty() => 0, // EOF
            Some(data) => {
                if mem.write(buf, &data).is_err() {
                    return err(Errno::EFAULT);
                }
                data.len() as i64
            }
        }
    }

    /// A non-canonical (`ICANON` off) slave read, honoring `VMIN`/`VTIME`:
    /// - `VMIN==0`: return immediately with whatever is buffered; if nothing and
    ///   `VTIME>0`, wait up to `VTIME` deciseconds then return 0 (never blocks
    ///   forever); with `VTIME==0` it's a pure non-blocking poll.
    /// - `VMIN>0`: block until at least `VMIN` bytes are available (as Linux does
    ///   for `VTIME==0`), then return them. `VTIME` acts as an inter-byte timer
    ///   once some data has arrived.
    #[allow(clippy::too_many_arguments)]
    fn read_pty_slave_noncanon(
        &self,
        cx: &mut ServiceCtx,
        n: usize,
        vmin: usize,
        vtime_ds: u64,
        buf: u64,
        count: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let (avail, master_open) = self.ptys.lock().unwrap().slave_input(n);
        if avail == 0 && !master_open {
            cx.cur.wake_deadline = None;
            return 0; // EOF
        }
        let nonblock = self.ptys.lock().unwrap().is_nonblock(n, false);
        let ready = if vmin == 0 { avail > 0 } else { avail >= vmin };
        if ready {
            cx.cur.wake_deadline = None;
            let data = self
                .ptys
                .lock()
                .unwrap()
                .slave_read(n, count as usize)
                .unwrap_or_default();
            if mem.write(buf, &data).is_err() {
                return err(Errno::EFAULT);
            }
            return data.len() as i64;
        }
        // Not enough data yet.
        if nonblock {
            // VMIN==0 returns 0 (no data now); VMIN>0 with too few bytes → EAGAIN.
            return if vmin == 0 { 0 } else { err(Errno::EAGAIN) };
        }
        // A VTIME timer applies for VMIN==0, or for VMIN>0 once some bytes exist
        // (the inter-byte timer). VMIN==0/VTIME==0 is a pure poll → return 0.
        let timed = vtime_ds > 0 && (vmin == 0 || avail > 0);
        if !timed {
            if vmin == 0 {
                cx.cur.wake_deadline = None;
                return 0; // poll: nothing available
            }
            cx.block = true; // VMIN>0/VTIME==0: wait for VMIN bytes
            return 0;
        }
        let deadline = if let Some(dl) = cx.cur.wake_deadline {
            dl
        } else {
            let dl = poll::now_ns() + u128::from(vtime_ds) * 100_000_000; // deciseconds → ns
            cx.cur.wake_deadline = Some(dl);
            dl
        };
        if poll::now_ns() >= deadline {
            // Timed out: return whatever is buffered (0 for VMIN==0).
            cx.cur.wake_deadline = None;
            let data = self
                .ptys
                .lock()
                .unwrap()
                .slave_read(n, count as usize)
                .unwrap_or_default();
            if mem.write(buf, &data).is_err() {
                return err(Errno::EFAULT);
            }
            return data.len() as i64;
        }
        cx.block = true; // park until the deadline or until more data arrives
        0
    }

    /// The `Fd::Eventfd`/`Fd::Timerfd` arm of [`Self::sys_read`]/
    /// [`Self::sys_readv`]: drain the eventfd counter or the timerfd expiration
    /// count. `pollfds`-only (the innermost lock). An empty counter on a
    /// blocking fd sets the block flag and returns 0 (the caller drops the lock
    /// and re-traps) — it never blocks in place holding the lock.
    fn read_pollfd_fd(
        &self,
        pf: &mut PollFds,
        cx: &mut ServiceCtx,
        fd: u64,
        buf: u64,
        count: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        match cx.cur.fds.get(fd as i32).cloned() {
            Some(Fd::Eventfd(i)) => self.read_eventfd(pf, cx, i, buf, count, mem),
            Some(Fd::Timerfd(i)) => self.read_timerfd(pf, cx, i, buf, count, mem),
            Some(Fd::Signalfd(i)) => self.read_signalfd(pf, cx, i, buf, count, mem),
            _ => err(Errno::EBADF),
        }
    }

    /// The `Fd::Socket` arm of [`Self::sys_read`]/[`Self::sys_readv`]: receive
    /// up to `count` bytes from the socket. `net`-only.
    fn read_socket_fd(
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
        fd: u64,
        buf: u64,
        count: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some(Fd::Socket { sock, end }) = cx.cur.fds.get(fd as i32).cloned() else {
            return err(Errno::EBADF);
        };
        self.read_socket(net, cx, sock, end, buf, count, mem)
    }

    /// The `Fd::PipeRead` arm of [`Self::sys_read`]/[`Self::sys_readv`]: drain
    /// up to `count` bytes from the pipe. `pipes`-only. An empty pipe with
    /// writers still open sets the block flag and returns 0 (the caller drops
    /// the lock and re-traps) — it never blocks in place holding the lock.
    fn read_pipe_fd(
        &self,
        pipes: &mut [Pipe],
        cx: &mut ServiceCtx,
        fd: u64,
        buf: u64,
        count: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some(Fd::PipeRead(i)) = cx.cur.fds.get(fd as i32).cloned() else {
            return err(Errno::EBADF);
        };
        self.read_pipe(pipes, cx, i, buf, count, mem)
    }

    /// The `Fd::File` arm of [`Self::sys_read`]: read at the fd's offset and
    /// advance it. `vfs`-only.
    #[allow(clippy::unused_self)]
    fn read_file_fd(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        fd: u64,
        buf: u64,
        count: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some(Fd::File {
            path,
            offset: ofs,
            readable,
            ..
        }) = cx.cur.fds.get(fd as i32).cloned()
        else {
            return err(Errno::EBADF);
        };
        if !readable {
            return err(Errno::EBADF); // fd opened O_WRONLY
        }
        let r = self.read_file_chunked(vfs, &path, ofs.get(), buf, count, mem);
        if r > 0 {
            ofs.add(r as u64);
        }
        r
    }

    /// Read up to `count` bytes of `path` at `off` into guest `buf`, in
    /// bounded host chunks (a guest asking for `SSIZE_MAX` bytes must not
    /// make the host allocate that), stopping at EOF. Returns the bytes read,
    /// `EFAULT` if the buffer faults before anything was read.
    fn read_file_chunked(
        &self,
        vfs: &mut MountTable,
        path: &str,
        off: u64,
        buf: u64,
        count: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const CHUNK: u64 = 1 << 20;
        let count = count.min(MAX_RW_COUNT);
        let mut total = 0u64;
        while total < count {
            let want = (count - total).min(CHUNK);
            let mut tmp = vec![0u8; want as usize];
            let n = match self.vfs_read(vfs, path, off + total, &mut tmp) {
                Ok(n) => n,
                Err(e) if total == 0 => return io_errno(&e),
                Err(_) => break,
            };
            if mem.write(buf + total, &tmp[..n]).is_err() {
                return if total > 0 {
                    total as i64
                } else {
                    err(Errno::EFAULT)
                };
            }
            total += n as u64;
            if (n as u64) < want {
                break; // EOF
            }
        }
        total as i64
    }

    /// The non-`File`, non-`Socket`, non-`PipeRead`, non-`Eventfd`/`Timerfd`
    /// arms of [`Self::sys_read`] (stdin), backed by `shared`. Sockets go
    /// through [`Self::read_socket_fd`] under `net`; pipes through
    /// [`Self::read_pipe_fd`] under `pipes`; eventfds/timerfds through
    /// [`Self::read_pollfd_fd`] under `pollfds`.
    fn read_shared_fd(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        fd: u64,
        buf: u64,
        count: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        match cx.cur.fds.get(fd as i32).cloned() {
            Some(Fd::Stdin) if self.interactive => {
                // Draw from the buffered terminal input; block (re-trap) when it
                // is empty and not yet closed, so the embedder can pump more.
                if sh.stdin_buf.is_empty() {
                    if sh.stdin_closed {
                        sh.stdin_waiting = false;
                        return 0; // EOF
                    }
                    sh.stdin_waiting = true;
                    cx.block = true;
                    return 0;
                }
                sh.stdin_waiting = false;
                let n = (count as usize).min(sh.stdin_buf.len());
                let chunk: Vec<u8> = sh.stdin_buf.drain(..n).collect();
                if mem.write(buf, &chunk).is_err() {
                    return err(Errno::EFAULT);
                }
                n as i64
            }
            Some(Fd::Stdin) => {
                let mut tmp = vec![0u8; count.min(1 << 20) as usize];
                match sh.stdin.read(&mut tmp) {
                    Ok(n) => {
                        if mem.write(buf, &tmp[..n]).is_err() {
                            return err(Errno::EFAULT);
                        }
                        n as i64
                    }
                    Err(_) => err(Errno::EIO),
                }
            }
            _ => err(Errno::EBADF),
        }
    }

    /// Read from pipe `i`. Empty with writers still open -> block; empty with no
    /// writers -> EOF (0).
    #[allow(clippy::unused_self)]
    fn read_pipe(
        &self,
        pipes: &mut [Pipe],
        cx: &mut ServiceCtx,
        i: usize,
        buf: u64,
        count: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        if pipes[i].buf.is_empty() {
            if pipes[i].writers > 0 {
                // Empty but still open: a non-blocking reader gets EAGAIN; a
                // blocking one parks (re-trap) until a writer feeds the pipe.
                if pipes[i].read_nonblock {
                    return err(Errno::EAGAIN);
                }
                cx.block = true;
            }
            return 0;
        }
        let n = count.min(pipes[i].buf.len() as u64) as usize;
        let data: Vec<u8> = pipes[i].buf.drain(..n).collect();
        if mem.write(buf, &data).is_err() {
            return err(Errno::EFAULT);
        }
        n as i64
    }

    /// Write to pipe `i` (`EPIPE` if all readers are gone). A broken-pipe write
    /// also raises `SIGPIPE` on the writer (so `producer | consumer` dies when
    /// the consumer exits) unless `nosignal` — the write's own default action
    /// then terminates the process, or it sees `EPIPE` if SIGPIPE is caught/ignored.
    fn write_pipe(
        &self,
        pipes: &mut [Pipe],
        cx: &mut ServiceCtx,
        i: usize,
        data: &[u8],
        nosignal: bool,
    ) -> i64 {
        if pipes[i].readers == 0 {
            if !nosignal {
                self.raise_sigpipe(cx);
            }
            return err(Errno::EPIPE);
        }
        pipes[i].buf.extend(data.iter().copied());
        data.len() as i64
    }

    /// Post `SIGPIPE` to the current task — a write to a reader-less pipe/socket.
    /// Its default action terminates the process; a caught/ignored SIGPIPE lets
    /// the write's `EPIPE` return surface instead.
    #[allow(clippy::unused_self)]
    pub(super) fn raise_sigpipe(&self, cx: &mut ServiceCtx) {
        cx.cur.pending |= 1u64 << (SIGPIPE - 1);
    }

    /// `pread64(fd, buf, count, offset)` — read at `offset` without moving the
    /// fd's position. Files only (a pipe/socket has no position → `ESPIPE`).
    #[allow(clippy::too_many_arguments, clippy::unused_self)]
    fn sys_pread(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        fd: u64,
        buf: u64,
        count: u64,
        offset: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some(Fd::File { path, readable, .. }) = cx.cur.fds.get(fd as i32).cloned() else {
            return err(Errno::ESPIPE);
        };
        if !readable {
            return err(Errno::EBADF); // fd opened O_WRONLY
        }
        self.read_file_chunked(vfs, &path, offset, buf, count, mem)
    }

    /// `pwrite64(fd, buf, count, offset)` — write at `offset` without moving
    /// the fd's position.
    #[allow(clippy::too_many_arguments, clippy::unused_self)]
    fn sys_pwrite(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        fd: u64,
        buf: u64,
        count: u64,
        offset: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let Some(Fd::File { path, writable, .. }) = cx.cur.fds.get(fd as i32).cloned() else {
            return err(Errno::ESPIPE);
        };
        if !writable {
            return err(Errno::EBADF); // fd opened O_RDONLY
        }
        let Ok(data) = mem.read_vec(buf, count as usize) else {
            return err(Errno::EFAULT);
        };
        match self.vfs_write(vfs, &path, offset, &data) {
            Ok(n) => n as i64,
            Err(e) => io_errno(&e),
        }
    }

    /// `preadv(fd, iov, iovcnt, offset)` — scatter a positioned read across
    /// iovecs. `offset` is `pos_l` (`pos_h`, the 32-bit-compat high word, is 0
    /// for 64-bit callers).
    #[allow(clippy::too_many_arguments)]
    fn sys_preadv(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        fd: u64,
        iov: u64,
        iovcnt: u64,
        offset: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let mut cur = offset;
        let mut total = 0i64;
        for i in 0..iovcnt {
            let ent = iov + i * 16;
            let (Ok(base), Ok(len)) = (mem.read_u64(ent), mem.read_u64(ent + 8)) else {
                return if total > 0 { total } else { err(Errno::EFAULT) };
            };
            if len == 0 {
                continue;
            }
            let r = self.sys_pread(vfs, cx, fd, base, len, cur, mem);
            if r < 0 {
                return if total > 0 { total } else { r };
            }
            total += r;
            cur += r as u64;
            if (r as u64) < len {
                break;
            }
        }
        total
    }

    /// `pwritev(fd, iov, iovcnt, offset)` — gather a positioned write.
    #[allow(clippy::too_many_arguments)]
    fn sys_pwritev(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        fd: u64,
        iov: u64,
        iovcnt: u64,
        offset: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let mut cur = offset;
        let mut total = 0i64;
        for i in 0..iovcnt {
            let ent = iov + i * 16;
            let (Ok(base), Ok(len)) = (mem.read_u64(ent), mem.read_u64(ent + 8)) else {
                return if total > 0 { total } else { err(Errno::EFAULT) };
            };
            if len == 0 {
                continue;
            }
            let r = self.sys_pwrite(vfs, cx, fd, base, len, cur, mem);
            if r < 0 {
                return if total > 0 { total } else { r };
            }
            total += r;
            cur += r as u64;
            if (r as u64) < len {
                break;
            }
        }
        total
    }

    /// `ftruncate(fd, len)` — resize the file the fd refers to.
    #[allow(clippy::unused_self)]
    fn sys_ftruncate(&self, vfs: &mut MountTable, cx: &mut ServiceCtx, fd: u64, len: u64) -> i64 {
        let Some(Fd::File { path, writable, .. }) = cx.cur.fds.get(fd as i32).cloned() else {
            return err(Errno::EBADF);
        };
        if !writable {
            return err(Errno::EBADF); // ftruncate needs an fd open for writing
        }
        match self.vfs_truncate(vfs, &path, len) {
            Ok(()) => 0,
            Err(e) => io_errno(&e),
        }
    }

    /// `truncate(path, len)` — resize by path.
    fn sys_truncate(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        pathptr: u64,
        len: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let Some(rel) = read_path(mem, pathptr) else {
            return err(Errno::EFAULT);
        };
        let abs = self.resolve_path(cx, AT_FDCWD, &rel);
        match self.vfs_truncate(vfs, &abs, len) {
            Ok(()) => 0,
            Err(e) => io_errno(&e),
        }
    }

    /// `fallocate(fd, mode, offset, len)`. Default mode (0) grows the file to at
    /// least `offset + len`. `FALLOC_FL_PUNCH_HOLE` (which must be combined with
    /// `FALLOC_FL_KEEP_SIZE`) zeroes the byte range without changing the file
    /// size. Other modes are accepted as no-ops.
    #[allow(clippy::unused_self)]
    fn sys_fallocate(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        fd: u64,
        mode: u64,
        offset: u64,
        len: u64,
    ) -> i64 {
        const FALLOC_FL_KEEP_SIZE: u64 = 0x01;
        const FALLOC_FL_PUNCH_HOLE: u64 = 0x02;
        let Some(Fd::File { path, writable, .. }) = cx.cur.fds.get(fd as i32).cloned() else {
            return err(Errno::EBADF);
        };
        if !writable {
            return err(Errno::EBADF); // fallocate needs a writable fd
        }
        let cur = vfs.stat(&path).map_or(0, |a| a.size);
        if mode & FALLOC_FL_PUNCH_HOLE != 0 {
            // PUNCH_HOLE requires KEEP_SIZE and must not extend the file: zero
            // only the portion of [offset, offset+len) that lies within EOF.
            if mode & FALLOC_FL_KEEP_SIZE == 0 {
                return err(Errno::EINVAL);
            }
            let end = offset.saturating_add(len).min(cur);
            if end <= offset {
                return 0;
            }
            let zeros = vec![0u8; (end - offset) as usize];
            return match self.vfs_write(vfs, &path, offset, &zeros) {
                Ok(_) => 0,
                Err(e) => io_errno(&e),
            };
        }
        // Default allocate/extend: grow the file if the range runs past EOF.
        let want = offset.saturating_add(len);
        if want > cur {
            match vfs.truncate(&path, want) {
                Ok(()) => 0,
                Err(e) => io_errno(&e),
            }
        } else {
            0
        }
    }

    /// `sendfile(out_fd, in_fd, offset_ptr, count)` — copy up to `count` bytes
    /// from `in_fd` to `out_fd`. If `offset_ptr` is non-null it names the start
    /// offset in `in_fd` (and is advanced), and `in_fd`'s own position is left
    /// alone; otherwise `in_fd`'s position is used and advanced.
    #[allow(clippy::too_many_arguments)]
    fn sys_sendfile(
        &self,
        sh: &mut Shared,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        out_fd: u64,
        in_fd: u64,
        offset_ptr: u64,
        count: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        // Resolve the source position.
        let use_ptr = offset_ptr != 0;
        let start = if use_ptr {
            match mem.read_u64(offset_ptr) {
                Ok(v) => v,
                Err(_) => return err(Errno::EFAULT),
            }
        } else {
            match cx.cur.fds.get(in_fd as i32) {
                Some(Fd::File { offset, .. }) => offset.get(),
                _ => return err(Errno::EINVAL),
            }
        };
        let Some(Fd::File { path, readable, .. }) = cx.cur.fds.get(in_fd as i32).cloned() else {
            return err(Errno::EINVAL);
        };
        if !readable {
            return err(Errno::EBADF); // source opened O_WRONLY
        }
        // No more than the source holds past `start` (callers commonly pass
        // SIZE_MAX), and no more than MAX_RW_COUNT.
        let avail = vfs.stat(&path).map_or(0, |a| a.size.saturating_sub(start));
        let mut buf = vec![0u8; count.min(avail).min(MAX_RW_COUNT) as usize];
        let n = match self.vfs_read(vfs, &path, start, &mut buf) {
            Ok(n) => n,
            Err(e) => return io_errno(&e),
        };
        buf.truncate(n);
        // Write it out through the normal write path (files, pipes, sockets).
        let written = match cx.cur.fds.get(out_fd as i32).cloned() {
            Some(Fd::File {
                writable: false, ..
            }) => err(Errno::EBADF), // out fd is O_RDONLY
            Some(Fd::File { path, offset, .. }) => {
                match self.vfs_write(vfs, &path, offset.get(), &buf) {
                    Ok(w) => {
                        offset.add(w as u64);
                        w as i64
                    }
                    Err(e) => io_errno(&e),
                }
            }
            Some(Fd::Stdout) => sh
                .stdout
                .write_all(&buf)
                .map_or(err(Errno::EIO), |()| buf.len() as i64),
            Some(Fd::Stderr) => sh
                .stderr
                .write_all(&buf)
                .map_or(err(Errno::EIO), |()| buf.len() as i64),
            // Destination is a pipe: its buffer lives in `pipes`, taken *after*
            // sh, vfs (and it never coexists with `net` here) — pipes is last.
            Some(Fd::PipeWrite(i)) => {
                self.write_pipe(&mut self.pipes.lock().unwrap(), cx, i, &buf, false)
            }
            // Destination is a socket: its state lives in `net`, taken *after*
            // sh and vfs (sh → vfs → net order) and released with the arm.
            Some(Fd::Socket { sock, end }) => {
                self.write_socket(&mut self.net.lock().unwrap(), cx, sock, end, &buf, false)
            }
            _ => err(Errno::EBADF),
        };
        if written < 0 {
            return written;
        }
        let advanced = written as u64;
        if use_ptr {
            let _ = mem.write_u64(offset_ptr, start + advanced);
        } else if let Some(Fd::File { offset, .. }) = cx.cur.fds.get_mut(in_fd as i32) {
            offset.add(advanced);
        }
        written
    }

    /// `copy_file_range(fd_in, off_in, fd_out, off_out, len, flags)` — copy
    /// between two files, honoring the optional in/out offset pointers.
    #[allow(clippy::unused_self)]
    fn sys_copy_file_range(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        a: &[u64; 6],
        mem: &mut GuestMemory,
    ) -> i64 {
        let (fd_in, off_in_p, fd_out, off_out_p, len) = (a[0], a[1], a[2], a[3], a[4]);
        let Some(Fd::File {
            path: in_path,
            offset: in_pos,
            readable: in_r,
            ..
        }) = cx.cur.fds.get(fd_in as i32).cloned()
        else {
            return err(Errno::EBADF);
        };
        let in_pos = in_pos.get();
        if !in_r {
            return err(Errno::EBADF); // source opened O_WRONLY
        }
        let in_off = if off_in_p != 0 {
            mem.read_u64(off_in_p).unwrap_or(in_pos)
        } else {
            in_pos
        };
        // coreutils `cp` passes SIZE_MAX: bound by what the source holds.
        let avail = vfs
            .stat(&in_path)
            .map_or(0, |a| a.size.saturating_sub(in_off));
        let mut buf = vec![0u8; len.min(avail).min(MAX_RW_COUNT) as usize];
        let n = match self.vfs_read(vfs, &in_path, in_off, &mut buf) {
            Ok(n) => n,
            Err(e) => return io_errno(&e),
        };
        buf.truncate(n);
        let Some(Fd::File {
            path: out_path,
            offset: out_pos,
            writable: out_w,
            ..
        }) = cx.cur.fds.get(fd_out as i32).cloned()
        else {
            return err(Errno::EBADF);
        };
        let out_pos = out_pos.get();
        if !out_w {
            return err(Errno::EBADF); // destination opened O_RDONLY
        }
        let out_off = if off_out_p != 0 {
            mem.read_u64(off_out_p).unwrap_or(out_pos)
        } else {
            out_pos
        };
        let w = match self.vfs_write(vfs, &out_path, out_off, &buf) {
            Ok(w) => w,
            Err(e) => return io_errno(&e),
        };
        // Advance the offsets (pointer or fd position).
        if off_in_p != 0 {
            let _ = mem.write_u64(off_in_p, in_off + w as u64);
        } else if let Some(Fd::File { offset, .. }) = cx.cur.fds.get_mut(fd_in as i32) {
            offset.add(w as u64);
        }
        if off_out_p != 0 {
            let _ = mem.write_u64(off_out_p, out_off + w as u64);
        } else if let Some(Fd::File { offset, .. }) = cx.cur.fds.get_mut(fd_out as i32) {
            offset.add(w as u64);
        }
        w as i64
    }

    /// `linkat(olddirfd, old, newdirfd, new, flags)` (and plain `link`) — the
    /// mount table has no true hard-link primitive, so this copies the source
    /// file's contents to the new path (correct for the overwhelmingly common
    /// use — same-content at a second name; the shared-inode nuance is lost).
    #[allow(clippy::too_many_arguments)]
    fn sys_linkat(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        olddirfd: i64,
        oldp: u64,
        newdirfd: i64,
        newp: u64,
        flags: u64,
        mem: &GuestMemory,
    ) -> i64 {
        const AT_SYMLINK_FOLLOW: u64 = 0x400;
        const AT_EMPTY_PATH: u64 = 0x1000;
        if flags & !(AT_SYMLINK_FOLLOW | AT_EMPTY_PATH) != 0 {
            return err(Errno::EINVAL);
        }
        let (Some(orel), Some(nrel)) = (read_path(mem, oldp), read_path(mem, newp)) else {
            return err(Errno::EFAULT);
        };
        if orel.is_empty() && flags & AT_EMPTY_PATH == 0 {
            return err(Errno::ENOENT);
        }
        let mut old_abs = self.resolve_path(cx, olddirfd, &orel);
        // AT_SYMLINK_FOLLOW links the symlink's target instead of the link.
        if flags & AT_SYMLINK_FOLLOW != 0 {
            old_abs = match self.follow_or_eloop(vfs, &old_abs) {
                Ok(p) => p,
                Err(e) => return e,
            };
        }
        let new_abs = self.resolve_path(cx, newdirfd, &nrel);
        // Naming an unnamed (O_TMPFILE) or unlinked file moves it into place.
        if let Some(r) = self.link_orphan(vfs, cx, &old_abs, &new_abs) {
            return r;
        }
        let Some(attrs) = vfs.stat(&old_abs) else {
            return err(Errno::ENOENT);
        };
        if attrs.kind == NodeKind::Dir {
            return err(Errno::EPERM); // can't hard-link a directory
        }
        if vfs.stat(&new_abs).is_some() {
            return err(Errno::EEXIST);
        }
        // Make a real hard link (tmpfs, the overlay's upper layer, host-backed
        // mounts): one inode, st_nlink, shared writes. EOPNOTSUPP means the
        // backend can't (a read-only image, a FIFO/symlink in tmpfs);
        // crossing mounts is EXDEV, as on Linux. Any other error is real.
        match vfs.link(&old_abs, &new_abs) {
            Ok(()) => return 0,
            Err(e) if e.raw_os_error() == Some(95) => {} // EOPNOTSUPP: recreate below
            Err(e) => return io_errno(&e),
        }
        // No inode to share: recreate the node under the new name — a
        // symlink as a symlink, a FIFO as a FIFO, a file as a copy.
        if attrs.kind == NodeKind::Symlink {
            return match vfs
                .readlink(&old_abs)
                .and_then(|t| vfs.symlink(&t, &new_abs))
            {
                Ok(()) => 0,
                Err(e) => io_errno(&e),
            };
        }
        if attrs.kind == NodeKind::Fifo {
            return match vfs.mknod(&new_abs, attrs.mode) {
                Ok(()) => 0,
                Err(e) => io_errno(&e),
            };
        }
        let mut data = vec![0u8; attrs.size as usize];
        if vfs.read_at(&old_abs, 0, &mut data).is_err() {
            return err(Errno::EIO);
        }
        if let Err(e) = vfs.create(&new_abs, attrs.mode & 0o7777) {
            return io_errno(&e);
        }
        match vfs.write_at(&new_abs, 0, &data) {
            Ok(_) => 0,
            Err(e) => io_errno(&e),
        }
    }

    /// `readv(fd, iov, iovcnt)` — scatter a read across `struct iovec` entries.
    /// A short read (or a blocking fd) stops after the first partially-filled
    /// iovec, like the real syscall.
    #[allow(clippy::too_many_lines)] // one repetitive scatter block per fd-lock kind
    fn sys_readv(
        &self,
        cx: &mut ServiceCtx,
        fd: u64,
        iov: u64,
        iovcnt: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        // fd-polymorphic, and atomic across iovecs: peek the fd type once (no
        // lock), then hold a single lock for the whole scatter — a file readv
        // holds only `vfs`, a socket readv only `net`, a pipe readv only
        // `pipes`, every other source `sh`.
        if let Some(Fd::File { .. }) = cx.cur.fds.get(fd as i32) {
            let mut vfs = self.vfs.lock().unwrap();
            let mut total = 0i64;
            for i in 0..iovcnt {
                let ent = iov + i * 16;
                let (Ok(base), Ok(len)) = (mem.read_u64(ent), mem.read_u64(ent + 8)) else {
                    return if total > 0 { total } else { err(Errno::EFAULT) };
                };
                if len == 0 {
                    continue;
                }
                let r = self.read_file_fd(&mut vfs, cx, fd, base, len, mem);
                if r < 0 {
                    return if total > 0 { total } else { r };
                }
                total += r;
                if (r as u64) < len {
                    break; // short read: don't touch the remaining iovecs
                }
            }
            return total;
        }
        if let Some(Fd::Socket { .. }) = cx.cur.fds.get(fd as i32) {
            let mut net = self.net.lock().unwrap();
            let mut total = 0i64;
            for i in 0..iovcnt {
                let ent = iov + i * 16;
                let (Ok(base), Ok(len)) = (mem.read_u64(ent), mem.read_u64(ent + 8)) else {
                    return if total > 0 { total } else { err(Errno::EFAULT) };
                };
                if len == 0 {
                    continue;
                }
                let r = self.read_socket_fd(&mut net, cx, fd, base, len, mem);
                if r < 0 {
                    return if total > 0 { total } else { r };
                }
                total += r;
                if (r as u64) < len {
                    break; // short read: don't touch the remaining iovecs
                }
            }
            return total;
        }
        if let Some(Fd::PipeRead(..)) = cx.cur.fds.get(fd as i32) {
            let mut pipes = self.pipes.lock().unwrap();
            let mut total = 0i64;
            for i in 0..iovcnt {
                let ent = iov + i * 16;
                let (Ok(base), Ok(len)) = (mem.read_u64(ent), mem.read_u64(ent + 8)) else {
                    return if total > 0 { total } else { err(Errno::EFAULT) };
                };
                if len == 0 {
                    continue;
                }
                let r = self.read_pipe_fd(&mut pipes, cx, fd, base, len, mem);
                if r < 0 {
                    return if total > 0 { total } else { r };
                }
                total += r;
                if (r as u64) < len {
                    break; // short read (or empty pipe): stop scattering
                }
            }
            return total;
        }
        if let Some(Fd::Eventfd(..) | Fd::Timerfd(..)) = cx.cur.fds.get(fd as i32) {
            let mut pf = self.pollfds.lock().unwrap();
            let mut total = 0i64;
            for i in 0..iovcnt {
                let ent = iov + i * 16;
                let (Ok(base), Ok(len)) = (mem.read_u64(ent), mem.read_u64(ent + 8)) else {
                    return if total > 0 { total } else { err(Errno::EFAULT) };
                };
                if len == 0 {
                    continue;
                }
                let r = self.read_pollfd_fd(&mut pf, cx, fd, base, len, mem);
                if r < 0 {
                    return if total > 0 { total } else { r };
                }
                total += r;
                if (r as u64) < len {
                    break; // short read (or empty counter): stop scattering
                }
            }
            return total;
        }
        if let Some(Fd::PtyMaster(..) | Fd::PtySlave(..)) = cx.cur.fds.get(fd as i32) {
            let mut total = 0i64;
            for i in 0..iovcnt {
                let ent = iov + i * 16;
                let (Ok(base), Ok(len)) = (mem.read_u64(ent), mem.read_u64(ent + 8)) else {
                    return if total > 0 { total } else { err(Errno::EFAULT) };
                };
                if len == 0 {
                    continue;
                }
                let r = self.read_pty_fd(cx, fd, base, len, mem);
                if r < 0 {
                    return if total > 0 { total } else { r };
                }
                total += r;
                if (r as u64) < len {
                    break;
                }
            }
            return total;
        }
        let mut sh = self.shared.lock().unwrap();
        let mut total = 0i64;
        for i in 0..iovcnt {
            let ent = iov + i * 16;
            let (Ok(base), Ok(len)) = (mem.read_u64(ent), mem.read_u64(ent + 8)) else {
                return if total > 0 { total } else { err(Errno::EFAULT) };
            };
            if len == 0 {
                continue;
            }
            let r = self.read_shared_fd(&mut sh, cx, fd, base, len, mem);
            if r < 0 {
                return if total > 0 { total } else { r };
            }
            total += r;
            if (r as u64) < len {
                break;
            }
        }
        total
    }

    /// `writev(fd, iov, iovcnt)` — gather `struct iovec { base; len }` entries.
    /// fd-polymorphic and atomic across iovecs, exactly like [`Self::sys_readv`].
    #[allow(clippy::too_many_lines)] // one repetitive gather block per fd-lock kind
    fn sys_writev(
        &self,
        cx: &mut ServiceCtx,
        fd: u64,
        iov: u64,
        iovcnt: u64,
        mem: &GuestMemory,
    ) -> i64 {
        if let Some(Fd::File { .. }) = cx.cur.fds.get(fd as i32) {
            let mut vfs = self.vfs.lock().unwrap();
            let mut total = 0i64;
            for i in 0..iovcnt {
                let ent = iov + i * 16;
                let (Ok(base), Ok(len)) = (mem.read_u64(ent), mem.read_u64(ent + 8)) else {
                    return if total > 0 { total } else { err(Errno::EFAULT) };
                };
                if len == 0 {
                    continue;
                }
                let r = self.write_file_fd(&mut vfs, cx, fd, base, len, mem);
                if r < 0 {
                    return if total > 0 { total } else { r };
                }
                total += r;
                if (r as u64) < len {
                    break;
                }
            }
            return total;
        }
        if let Some(Fd::Socket { .. }) = cx.cur.fds.get(fd as i32) {
            let mut net = self.net.lock().unwrap();
            let mut total = 0i64;
            for i in 0..iovcnt {
                let ent = iov + i * 16;
                let (Ok(base), Ok(len)) = (mem.read_u64(ent), mem.read_u64(ent + 8)) else {
                    return if total > 0 { total } else { err(Errno::EFAULT) };
                };
                if len == 0 {
                    continue;
                }
                let r = self.write_socket_fd(&mut net, cx, fd, base, len, mem);
                if r < 0 {
                    return if total > 0 { total } else { r };
                }
                total += r;
                if (r as u64) < len {
                    break;
                }
            }
            return total;
        }
        if let Some(Fd::PipeWrite(..)) = cx.cur.fds.get(fd as i32) {
            let mut pipes = self.pipes.lock().unwrap();
            let mut total = 0i64;
            for i in 0..iovcnt {
                let ent = iov + i * 16;
                let (Ok(base), Ok(len)) = (mem.read_u64(ent), mem.read_u64(ent + 8)) else {
                    return if total > 0 { total } else { err(Errno::EFAULT) };
                };
                if len == 0 {
                    continue;
                }
                let r = self.write_pipe_fd(&mut pipes, cx, fd, base, len, mem);
                if r < 0 {
                    return if total > 0 { total } else { r };
                }
                total += r;
                if (r as u64) < len {
                    break;
                }
            }
            return total;
        }
        if let Some(Fd::Eventfd(..)) = cx.cur.fds.get(fd as i32) {
            let mut pf = self.pollfds.lock().unwrap();
            let mut total = 0i64;
            for i in 0..iovcnt {
                let ent = iov + i * 16;
                let (Ok(base), Ok(len)) = (mem.read_u64(ent), mem.read_u64(ent + 8)) else {
                    return if total > 0 { total } else { err(Errno::EFAULT) };
                };
                if len == 0 {
                    continue;
                }
                let r = self.write_pollfd_fd(&mut pf, cx, fd, base, len, mem);
                if r < 0 {
                    return if total > 0 { total } else { r };
                }
                total += r;
                if (r as u64) < len {
                    break;
                }
            }
            return total;
        }
        let mut sh = self.shared.lock().unwrap();
        let mut total = 0i64;
        for i in 0..iovcnt {
            let ent = iov + i * 16;
            let (Ok(base), Ok(len)) = (mem.read_u64(ent), mem.read_u64(ent + 8)) else {
                return if total > 0 { total } else { err(Errno::EFAULT) };
            };
            if len == 0 {
                continue;
            }
            if let Some(Fd::PtyMaster(..) | Fd::PtySlave(..)) = cx.cur.fds.get(fd as i32) {
                let r = self.write_pty_fd(cx, fd, base, len, mem);
                if r < 0 {
                    return if total > 0 { total } else { r };
                }
                total += r;
                continue;
            }
            let r = self.write_shared_fd(&mut sh, cx, fd, base, len, mem);
            if r < 0 {
                return if total > 0 { total } else { r };
            }
            total += r;
            if (r as u64) < len {
                break;
            }
        }
        total
    }

    /// The `(soft, hard)` limit pair for `resource`, consulting the tracked
    /// `RLIMIT_NOFILE` and the fixed values for everything else.
    #[allow(clippy::unused_self)]
    fn rlimit_pair(&self, sh: &mut Shared, resource: u64) -> (u64, u64) {
        if resource == sys_misc::RLIMIT_NOFILE {
            sh.rlimit_nofile
        } else {
            sys_misc::rlimit_for(resource)
        }
    }

    /// `prlimit64(pid, resource, new_limit, old_limit)` — report the current
    /// limit into `old_limit`, then apply `new_limit` (for `RLIMIT_NOFILE`,
    /// which is the only one we track; the hard limit is capped so a program
    /// can't raise it into a pathological fd-scan range).
    fn sys_prlimit64(
        &self,
        sh: &mut Shared,
        resource: u64,
        new_limit: u64,
        old_limit: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let (cur, max) = self.rlimit_pair(sh, resource);
        if old_limit != 0 {
            let r = sys_misc::write_rlimit(mem, old_limit, cur, max);
            if r < 0 {
                return r;
            }
        }
        if new_limit != 0 && resource == sys_misc::RLIMIT_NOFILE {
            let Some((mut new_cur, mut new_max)) = sys_misc::read_rlimit(mem, new_limit) else {
                return err(Errno::EFAULT);
            };
            new_max = new_max.min(sys_misc::NOFILE_HARD_CAP);
            new_cur = new_cur.min(new_max);
            sh.rlimit_nofile = (new_cur, new_max);
        }
        0
    }

    /// `getrlimit(resource, buf)` — report the current limit for `resource`.
    fn sys_getrlimit(
        &self,
        sh: &mut Shared,
        resource: u64,
        buf: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let (cur, max) = self.rlimit_pair(sh, resource);
        sys_misc::write_rlimit(mem, buf, cur, max)
    }

    /// `ioctl(fd, request, arg)` — only the fd-flag requests that work on any fd
    /// are honored; genuine terminal requests (`TCGETS`, `TIOCGWINSZ`, …) return
    /// `ENOTTY`, which is the correct answer for the pipe/file/socket fds nixvm
    /// hands out (there is no pty). The important one is `FIONBIO`: it is the
    /// ioctl spelling of `fcntl(F_SETFL, O_NONBLOCK)`, so a client that sets its
    /// socket non-blocking this way must not be silently left blocking (that
    /// strands an event loop, exactly like the `F_SETFL` gap did).
    fn sys_ioctl(
        &self,
        cx: &mut ServiceCtx,
        fd: u64,
        request: u64,
        arg: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const FIONBIO: u32 = 0x5421;
        const FIOCLEX: u32 = 0x5451;
        const FIONCLEX: u32 = 0x5450;
        const FIOASYNC: u32 = 0x5452;
        const FIGETBSZ: u32 = 2;
        const FICLONE: u32 = 0x4004_9409;
        const FICLONERANGE: u32 = 0x4020_940d;
        const FIDEDUPERANGE: u32 = 0xc018_9436;
        const FIOQSIZE: u32 = 0x5460;
        const FS_IOC_GETFLAGS: u32 = 0x8008_6601;
        const FS_IOC_SETFLAGS: u32 = 0x4008_6602;
        const FS_IOC_GETVERSION: u32 = 0x8008_7601;
        const FS_IOC_FSGETXATTR: u32 = 0x801c_581f;
        const FS_IOC_FSSETXATTR: u32 = 0x401c_5820;
        const FIONREAD: u32 = 0x541B; // == SIOCINQ
        const SIOCOUTQ: u32 = 0x5411;
        let Some(f) = cx.cur.fds.get(fd as i32).cloned() else {
            return err(Errno::EBADF);
        };
        // ioctl requests are 32-bit; some (`_IOR`-encoded) reach us sign-extended.
        let req = (request & 0xffff_ffff) as u32;
        // Pty ends answer their own termios/winsize/TIOCGPTN/TIOCSPTLCK/FIONREAD/
        // FIONBIO against the in-VM pty, never the host tty.
        if let Fd::PtyMaster(n) | Fd::PtySlave(n) = f {
            return self.pty_ioctl(n, matches!(f, Fd::PtyMaster(_)), req, arg, mem);
        }
        // Terminal-attribute ioctls on the guest's stdio: forward to the real
        // host tty when the guest's stdio is the host's own (the CLI path), so
        // the guest gets a working virtual terminal (size, raw mode, echo). The
        // host ioctl itself returns ENOTTY when the fd isn't actually a tty
        // (output piped), so isatty() stays honest.
        if self.host_tty && is_tty_ioctl(req) {
            let host_fd = match f {
                Fd::Stdin => Some(0),
                Fd::Stdout => Some(1),
                Fd::Stderr => Some(2),
                _ => None,
            };
            if let Some(hfd) = host_fd {
                return host_tty_ioctl(hfd, req, arg, mem);
            }
        }
        // Interface queries (`SIOCGIF*`) operate on any socket fd.
        if net::is_iface_ioctl(req) {
            return if matches!(f, Fd::Socket { .. }) {
                let link = self.net.lock().unwrap().link();
                net::iface_ioctl(req, arg, mem, link.as_ref())
            } else {
                err(Errno::ENOTTY)
            };
        }
        match req {
            // `arg` points at an `int`: nonzero sets `O_NONBLOCK`, zero clears it.
            FIONBIO => {
                let on = mem.read_u32(arg).is_ok_and(|v| v != 0);
                self.fd_set_nonblock(&f, on);
                0
            }
            // Set/clear close-on-exec, as fcntl(F_SETFD) does.
            FIOCLEX | FIONCLEX => {
                cx.cur.fds.set_cloexec(fd as i32, req == FIOCLEX);
                0
            }
            // `FIOASYNC` (signal-driven I/O): accepted, but no `SIGIO` is ever
            // sent — callers (nginx's master/worker channel) also poll the fd.
            FIOASYNC => 0,
            // Reflinks/dedupe (FICLONE, FICLONERANGE, FIDEDUPERANGE): no
            // backend shares extents — EOPNOTSUPP, which `cp --reflink=auto`
            // and friends fall back from to a plain copy.
            FICLONE | FICLONERANGE | FIDEDUPERANGE => err(Errno::EOPNOTSUPP),
            // File-only queries: FIGETBSZ (block size), FIOQSIZE (size), and
            // the ext2-style attribute ioctls `lsattr`/`chattr`/`cp -a` use:
            // no attribute flags are set and none can be (EOPNOTSUPP, as
            // tmpfs answers for the flags it lacks); the fsxattr form reads
            // all-zero.
            FIGETBSZ | FIOQSIZE | FS_IOC_GETFLAGS | FS_IOC_SETFLAGS | FS_IOC_GETVERSION
            | FS_IOC_FSGETXATTR | FS_IOC_FSSETXATTR => {
                let path = match &f {
                    Fd::File { path, .. } | Fd::Dir { path, .. } => path.clone(),
                    _ => return err(Errno::ENOTTY),
                };
                let out: Vec<u8> = match req {
                    FIGETBSZ => 4096u32.to_le_bytes().to_vec(),
                    FIOQSIZE => {
                        let size = self.vfs.lock().unwrap().stat(&path).map_or(0, |a| a.size);
                        size.to_le_bytes().to_vec()
                    }
                    FS_IOC_SETFLAGS => {
                        return match mem.read_u32(arg) {
                            Ok(0) => 0,
                            Ok(_) => err(Errno::EOPNOTSUPP),
                            Err(_) => err(Errno::EFAULT),
                        };
                    }
                    FS_IOC_FSSETXATTR => return 0,
                    FS_IOC_FSGETXATTR => vec![0u8; 28],
                    _ => 0u32.to_le_bytes().to_vec(), // GETFLAGS / GETVERSION
                };
                if mem.write(arg, &out).is_ok() {
                    0
                } else {
                    err(Errno::EFAULT)
                }
            }
            // Bytes available to read, written as an `int` at `arg`.
            FIONREAD => {
                let bytes = match &f {
                    Fd::PipeRead(i) => self
                        .pipes
                        .lock()
                        .unwrap()
                        .get(*i)
                        .map_or(0, |p| p.buf.len() as u64),
                    Fd::Eventfd(_) | Fd::Timerfd(_) | Fd::Signalfd(_) => {
                        self.pollfds.lock().unwrap().readable_bytes(&f)
                    }
                    Fd::Socket { sock, end } => {
                        let mut net = self.net.lock().unwrap();
                        self.socket_readable_bytes(&mut net, *sock, *end)
                    }
                    // Host stdin's count is not tracked, nor any other kind's.
                    _ => 0,
                };
                let v = u32::try_from(bytes).unwrap_or(u32::MAX);
                if mem.write(arg, &v.to_le_bytes()).is_ok() {
                    0
                } else {
                    err(Errno::EFAULT)
                }
            }
            // Bytes queued to send: nixvm flushes sockets straight to the host /
            // peer, so nothing is ever queued.
            SIOCOUTQ if matches!(f, Fd::Socket { .. }) => {
                if mem.write(arg, &0u32.to_le_bytes()).is_ok() {
                    0
                } else {
                    err(Errno::EFAULT)
                }
            }
            // Terminal requests on a non-terminal are a legitimate ENOTTY
            // (that is how `isatty` answers); anything else is a request no
            // handler knows — record it so the gap is visible, then answer
            // ENOTTY as Linux does for an fd type without that ioctl.
            _ => {
                // Type 'T' (0x54) is the terminal ioctl space.
                if (req >> 8) & 0xff != 0x54 {
                    self.note_unsupported("ioctl", u64::from(req));
                }
                err(Errno::ENOTTY)
            }
        }
    }

    /// ioctls on a pty end: `TCGETS`/`TCSETS`(`W`/`F`) and `TIOCGWINSZ`/
    /// `TIOCSWINSZ` against the pty's own termios/winsize, plus the master-only
    /// `TIOCGPTN` (slave number) and `TIOCSPTLCK` (unlock), and `FIONREAD`/
    /// `FIONBIO`. A successful `TCGETS` is also what makes `isatty()` true.
    fn pty_ioctl(
        &self,
        n: usize,
        is_master: bool,
        req: u32,
        arg: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const TCGETS: u32 = 0x5401;
        const TCSETS: u32 = 0x5402;
        const TCSETSW: u32 = 0x5403;
        const TCSETSF: u32 = 0x5404;
        const TIOCGWINSZ: u32 = 0x5413;
        const TIOCSWINSZ: u32 = 0x5414;
        const TIOCGPTN: u32 = 0x8004_5430;
        const TIOCSPTLCK: u32 = 0x4004_5431;
        const TIOCGPGRP: u32 = 0x540F;
        const TIOCSPGRP: u32 = 0x5410;
        const FIONREAD: u32 = 0x541B;
        const FIONBIO: u32 = 0x5421;
        const TCSBRK: u32 = 0x5409; // tcdrain (arg!=0) / tcsendbreak
        const TCXONC: u32 = 0x540A; // tcflow
        const TCFLSH: u32 = 0x540B; // tcflush
        const TCSBRKP: u32 = 0x5425; // tcsendbreak (POSIX)
        const TIOCOUTQ: u32 = 0x5411; // bytes still queued for output
        let mut ptys = self.ptys.lock().unwrap();
        match req {
            TCGETS => match ptys.get_termios(n) {
                Some(t) if mem.write(arg, &t).is_ok() => 0,
                Some(_) => err(Errno::EFAULT),
                None => err(Errno::EBADF),
            },
            TCSETS | TCSETSW | TCSETSF => match mem.read_vec(arg, pty::TERMIOS_LEN) {
                Ok(v) => {
                    let mut t = [0u8; pty::TERMIOS_LEN];
                    t.copy_from_slice(&v);
                    ptys.set_termios(n, t);
                    0
                }
                Err(_) => err(Errno::EFAULT),
            },
            TIOCGWINSZ => match ptys.get_winsize(n) {
                Some(w) if mem.write(arg, &w).is_ok() => 0,
                Some(_) => err(Errno::EFAULT),
                None => err(Errno::EBADF),
            },
            TIOCSWINSZ => match mem.read_vec(arg, pty::WINSIZE_LEN) {
                Ok(v) => {
                    let mut w = [0u8; pty::WINSIZE_LEN];
                    w.copy_from_slice(&v);
                    ptys.set_winsize(n, w);
                    0
                }
                Err(_) => err(Errno::EFAULT),
            },
            TIOCGPTN if is_master => {
                if mem.write(arg, &(n as u32).to_le_bytes()).is_ok() {
                    0
                } else {
                    err(Errno::EFAULT)
                }
            }
            TIOCSPTLCK if is_master => {
                ptys.set_lock(n, mem.read_u32(arg).is_ok_and(|v| v != 0));
                0
            }
            // `tcgetpgrp`/`tcsetpgrp`: the foreground process group, the target
            // of `ISIG`-generated signals. Both ends share one value.
            TIOCGPGRP => {
                let pgrp = ptys.fg_pgrp(n);
                if mem.write(arg, &pgrp.to_le_bytes()).is_ok() {
                    0
                } else {
                    err(Errno::EFAULT)
                }
            }
            TIOCSPGRP => match mem.read_u32(arg) {
                Ok(v) => {
                    ptys.set_fg_pgrp(n, v as i32);
                    0
                }
                Err(_) => err(Errno::EFAULT),
            },
            FIONREAD => {
                let bytes = if is_master {
                    ptys.master_avail(n)
                } else {
                    ptys.slave_avail(n)
                };
                let v = u32::try_from(bytes).unwrap_or(u32::MAX);
                if mem.write(arg, &v.to_le_bytes()).is_ok() {
                    0
                } else {
                    err(Errno::EFAULT)
                }
            }
            FIONBIO => {
                ptys.set_nonblock(n, is_master, mem.read_u32(arg).is_ok_and(|v| v != 0));
                0
            }
            // `tcflush`: `arg` is the queue selector, passed by value (not a
            // pointer). TCIFLUSH=0 (input), TCOFLUSH=1 (output), TCIOFLUSH=2.
            TCFLSH => {
                const TCIFLUSH: u64 = 0;
                const TCOFLUSH: u64 = 1;
                const TCIOFLUSH: u64 = 2;
                let (input, output) = match arg {
                    TCIFLUSH => (true, false),
                    TCOFLUSH => (false, true),
                    TCIOFLUSH => (true, true),
                    _ => return err(Errno::EINVAL),
                };
                ptys.flush(n, input, output);
                0
            }
            // `tcdrain`/`tcsendbreak` (output reaches the master queue instantly,
            // nothing to wait for) and `tcflow` (no software flow control is
            // modelled): accept every selector as a success no-op.
            TCSBRK | TCSBRKP | TCXONC => 0,
            // `TIOCOUTQ`: bytes still queued to transmit. Output is flushed to the
            // master immediately, so nothing is ever pending.
            TIOCOUTQ => {
                if mem.write(arg, &0u32.to_le_bytes()).is_ok() {
                    0
                } else {
                    err(Errno::EFAULT)
                }
            }
            _ => {
                self.note_unsupported("ioctl(pty)", u64::from(req));
                err(Errno::ENOTTY)
            }
        }
    }

    /// Apply an `fcntl(F_SETFL)` `O_NONBLOCK` change to whichever subsystem owns
    /// the fd (socket / eventfd / timerfd). Other fd kinds have no blocking mode
    /// to set. Acquires only the one relevant lock (order-safe: `sh` is held by
    /// the caller and both `net` and `pollfds` sort after it).
    fn fd_set_nonblock(&self, f: &Fd, nb: bool) {
        match f {
            Fd::Socket { sock, end } => self.net.lock().unwrap().set_nonblock(*sock, *end, nb),
            Fd::Eventfd(_) | Fd::Timerfd(_) | Fd::Signalfd(_) => {
                self.pollfds.lock().unwrap().set_nonblock(f, nb);
            }
            Fd::PtyMaster(n) => self.ptys.lock().unwrap().set_nonblock(*n, true, nb),
            Fd::PtySlave(n) => self.ptys.lock().unwrap().set_nonblock(*n, false, nb),
            Fd::PipeRead(i) => self.pipes.lock().unwrap()[*i].read_nonblock = nb,
            Fd::PipeWrite(i) => self.pipes.lock().unwrap()[*i].write_nonblock = nb,
            _ => {}
        }
    }

    /// The `O_NONBLOCK` state an `fcntl(F_GETFL)` should report.
    fn fd_is_nonblock(&self, f: &Fd) -> bool {
        match f {
            Fd::Socket { sock, end } => self.net.lock().unwrap().is_nonblock(*sock, *end),
            Fd::Eventfd(_) | Fd::Timerfd(_) | Fd::Signalfd(_) => {
                self.pollfds.lock().unwrap().is_nonblock(f)
            }
            Fd::PtyMaster(n) => self.ptys.lock().unwrap().is_nonblock(*n, true),
            Fd::PtySlave(n) => self.ptys.lock().unwrap().is_nonblock(*n, false),
            Fd::PipeRead(i) => self.pipes.lock().unwrap()[*i].read_nonblock,
            Fd::PipeWrite(i) => self.pipes.lock().unwrap()[*i].write_nonblock,
            _ => false,
        }
    }

    /// `openat(dirfd, path, flags, mode)` against the mount table.
    #[allow(clippy::too_many_arguments)]
    fn sys_openat(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        dirfd: i64,
        pathptr: u64,
        flags: u64,
        mode: u64,
        mem: &GuestMemory,
    ) -> i64 {
        // (O_DIRECTORY/O_NOFOLLOW are arch-specific: arm64 (asm-generic) uses
        // 0o40000/0o100000, and its 0o200000/0o400000 are O_DIRECT/O_LARGEFILE
        // — the x86-64 values. musl ORs O_LARGEFILE into every open, so using
        // the x86 values for an arm64 guest made every open through a symlink
        // fail with ELOOP (e.g. the dynamic linker loading libz.so.1). See
        // `open_path`.)

        let Some(rel) = read_path(mem, pathptr) else {
            return err(Errno::EFAULT);
        };
        self.open_path(vfs, cx, dirfd, &rel, flags, mode)
    }

    /// The body of [`Self::sys_openat`] once the path is in hand: resolve `rel`
    /// against `dirfd` and open it. Shared with `openat2`, which resolves the
    /// path itself (to apply its `RESOLVE_*` restrictions) and passes the
    /// result here as an absolute path.
    pub(super) fn open_path(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        dirfd: i64,
        rel: &str,
        flags: u64,
        mode: u64,
    ) -> i64 {
        const O_CREAT: u64 = 0o100;
        const O_EXCL: u64 = 0o200;
        const O_TRUNC: u64 = 0o1000;
        let (o_directory, o_nofollow): (u64, u64) = match self.arch {
            Arch::X86_64 => (0o200000, 0o400000),
            Arch::Aarch64 => (0o40000, 0o100000),
        };
        const O_ACCMODE: u64 = 0o3;
        // A trailing slash on the guest path demands a directory target.
        let had_slash = rel.len() > 1 && rel.ends_with('/');
        let resolved = self.resolve_path(cx, dirfd, rel);
        // O_TMPFILE (the __O_TMPFILE bit, same on both arches): an unnamed file
        // in the directory `rel` names (see `orphan.rs`).
        const O_TMPFILE: u64 = 0o20000000;
        if flags & O_TMPFILE != 0 {
            let dir = match self.follow_or_eloop(vfs, &resolved) {
                Ok(p) => p,
                Err(e) => return e,
            };
            return self.open_tmpfile(vfs, cx, &dir, flags, mode);
        }
        // O_NOFOLLOW: if the final component is itself a symlink, fail with ELOOP
        // rather than following it (a security check `open`ers rely on). Checked
        // against the *unfollowed* path; intermediate symlinks still resolve.
        if flags & o_nofollow != 0
            && vfs
                .stat(&resolved)
                .is_some_and(|a| a.kind == NodeKind::Symlink)
        {
            return err(Errno::ELOOP);
        }
        let abs = match self.follow_or_eloop(vfs, &resolved) {
            Ok(p) => p,
            Err(e) => return e,
        };
        if self.trace {
            eprintln!("[open] pid={} {abs:?}", cx.cur.pid);
        }
        // A `/proc` open (of a file to read or a directory to list) must see this
        // task's live identity — its program name, cmdline, and open fds — not the
        // boot-time placeholder. Refresh procfs's `self/` view before the fd is
        // created so the following read/readdir renders the running process.
        if (abs == "/proc" || abs.starts_with("/proc/"))
            && let Some(pf) = vfs.procfs_mut()
        {
            pf.update_self(self.proc_self_live(cx));
        }

        // Pseudo-terminals: `/dev/ptmx` allocates a fresh pty and returns its
        // master; `/dev/pts/N` opens the matching slave once unlocked.
        if abs == "/dev/ptmx" {
            let n = self.ptys.lock().unwrap().alloc();
            return i64::from(cx.cur.fds.alloc(Fd::PtyMaster(n)));
        }
        if let Some(rest) = abs.strip_prefix("/dev/pts/")
            && let Ok(n) = rest.parse::<usize>()
        {
            let mut ptys = self.ptys.lock().unwrap();
            if !ptys.slave_openable(n) {
                return err(Errno::ENXIO);
            }
            ptys.open_slave(n, pgid_of(&cx.cur));
            return i64::from(cx.cur.fds.alloc(Fd::PtySlave(n)));
        }

        match vfs.stat(&abs) {
            None => {
                if flags & O_CREAT != 0 {
                    // Creating a name with a trailing slash asks for a directory,
                    // which open(2) can't make → ENOTDIR (Linux via ENOTDIR/EISDIR).
                    if had_slash {
                        return err(Errno::ENOTDIR);
                    }
                    if let Err(e) = vfs.create(&abs, (mode & 0o777) as u32) {
                        return io_errno(&e);
                    }
                } else if self.has_nondir_component(vfs, &abs) {
                    return err(Errno::ENOTDIR);
                } else {
                    return err(Errno::ENOENT);
                }
            }
            // A trailing slash on a non-directory is ENOTDIR ("file/").
            Some(a) if had_slash && a.kind != NodeKind::Dir => {
                return err(Errno::ENOTDIR);
            }
            // O_CREAT|O_EXCL demands the file not already exist (atomic create —
            // the standard lock-file / mkstemp idiom); anything else is EEXIST.
            Some(_) if flags & O_CREAT != 0 && flags & O_EXCL != 0 => {
                return err(Errno::EEXIST);
            }
            Some(_) if flags & O_TRUNC != 0 => {
                let _ = self.vfs_truncate(vfs, &abs, 0);
            }
            Some(_) => {}
        }

        let Some(attrs) = vfs.stat(&abs) else {
            return err(Errno::ENOENT);
        };
        let is_dir = attrs.kind == NodeKind::Dir;
        // O_DIRECTORY requires a directory; a non-dir is ENOTDIR.
        if flags & o_directory != 0 && !is_dir {
            return err(Errno::ENOTDIR);
        }
        // A directory can't be opened for writing — EISDIR.
        if is_dir && flags & O_ACCMODE != 0 {
            return err(Errno::EISDIR);
        }
        let fd = if attrs.kind == NodeKind::Dir {
            cx.cur.fds.alloc(Fd::Dir { path: abs, pos: 0 })
        } else {
            cx.cur.fds.alloc(Fd::File {
                path: abs,
                offset: FileOffset::new(0),
                readable: flags & O_ACCMODE != 1, // not O_WRONLY
                writable: flags & O_ACCMODE != 0, // O_WRONLY or O_RDWR
            })
        };
        const O_CLOEXEC: u64 = 0o2000000;
        const O_APPEND: u64 = 0o2000;
        cx.cur.fds.set_cloexec(fd, flags & O_CLOEXEC != 0);
        cx.cur.fds.set_append(fd, flags & O_APPEND != 0);
        i64::from(fd)
    }

    /// `close(fd)`.
    fn sys_close(&self, cx: &mut ServiceCtx, fd: i32) -> i64 {
        match cx.cur.fds.close(fd) {
            Some(f) => {
                self.bump_pipe(&f, false);
                self.epoll_forget(cx, fd);
                match f {
                    Fd::PtyMaster(n) => self.ptys.lock().unwrap().close_master(n),
                    Fd::PtySlave(n) => self.ptys.lock().unwrap().close_slave(n),
                    _ => {}
                }
                0
            }
            None => err(Errno::EBADF),
        }
    }

    /// `pipe2(fds, flags)` — create an anonymous pipe. **pipes-only**.
    #[allow(clippy::unused_self)]
    fn sys_pipe2(
        &self,
        pipes: &mut Vec<Pipe>,
        cx: &mut ServiceCtx,
        fds_ptr: u64,
        flags: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const O_CLOEXEC: u64 = 0o2000000;
        const O_NONBLOCK: u64 = 0o4000;
        let nonblock = flags & O_NONBLOCK != 0;
        let idx = pipes.len();
        pipes.push(Pipe {
            buf: VecDeque::new(),
            readers: 1,
            writers: 1,
            read_nonblock: nonblock,
            write_nonblock: nonblock,
        });
        let rfd = cx.cur.fds.alloc(Fd::PipeRead(idx));
        let wfd = cx.cur.fds.alloc(Fd::PipeWrite(idx));
        let cloexec = flags & O_CLOEXEC != 0;
        cx.cur.fds.set_cloexec(rfd, cloexec);
        cx.cur.fds.set_cloexec(wfd, cloexec);
        let mut b = [0u8; 8];
        b[0..4].copy_from_slice(&rfd.to_le_bytes());
        b[4..8].copy_from_slice(&wfd.to_le_bytes());
        if mem.write(fds_ptr, &b).is_err() {
            return err(Errno::EFAULT);
        }
        0
    }

    /// `dup(oldfd)`.
    fn sys_dup(&self, cx: &mut ServiceCtx, oldfd: u64) -> i64 {
        let Some(fd) = cx.cur.fds.get(oldfd as i32).cloned() else {
            return err(Errno::EBADF);
        };
        self.bump_pipe(&fd, true);
        i64::from(cx.cur.fds.alloc(fd))
    }

    /// `dup2`/`dup3(oldfd, newfd, flags)`. `dup3` sets `FD_CLOEXEC` from
    /// `O_CLOEXEC`; `dup2` always clears it (via `insert`). `dup3` also differs
    /// on the `oldfd == newfd` case: `dup2` returns `newfd` unchanged, but
    /// `dup3` rejects it with `EINVAL`.
    fn sys_dup2(
        &self,
        cx: &mut ServiceCtx,
        oldfd: u64,
        newfd: u64,
        flags: u64,
        is_dup3: bool,
    ) -> i64 {
        const O_CLOEXEC: u64 = 0o2000000;
        // dup3 rejects equal fds with EINVAL *before* validating oldfd.
        if is_dup3 && oldfd == newfd {
            return err(Errno::EINVAL);
        }
        let Some(fd) = cx.cur.fds.get(oldfd as i32).cloned() else {
            return err(Errno::EBADF);
        };
        if oldfd == newfd {
            return newfd as i64;
        }
        if let Some(old) = cx.cur.fds.close(newfd as i32) {
            self.bump_pipe(&old, false);
            self.epoll_forget(cx, newfd as i32);
        }
        self.bump_pipe(&fd, true);
        cx.cur.fds.insert(newfd as i32, fd);
        cx.cur.fds.set_cloexec(newfd as i32, flags & O_CLOEXEC != 0);
        newfd as i64
    }

    /// Adjust the reader/writer refcount of the pipe (or socket) a fd refers to.
    /// A pipe fd's refcount lives in `pipes`, a socket fd's in `net`. Called
    /// with `sh` held by every caller (close/dup/fcntl/clone/exit), so the one
    /// lock this briefly takes — `pipes` or `net`, never both in a single call
    /// — is always acquired *after* `sh` (sh → net/pipes order) and released
    /// here; no other lock is taken under it, so the discipline holds.
    fn bump_pipe(&self, fd: &Fd, inc: bool) {
        let apply = |n: &mut usize| {
            if inc {
                *n += 1;
            } else {
                *n = n.saturating_sub(1);
            }
        };
        match fd {
            Fd::PipeRead(i) => apply(&mut self.pipes.lock().unwrap()[*i].readers),
            Fd::PipeWrite(i) => apply(&mut self.pipes.lock().unwrap()[*i].writers),
            Fd::Socket { .. } => self.net.lock().unwrap().bump(fd, inc),
            // Non-refcounted fds (files, stdio, eventfds, …): nothing to adjust.
            _ => {}
        }
    }

    /// `lseek(fd, offset, whence)`.
    #[allow(clippy::unused_self)]
    fn sys_lseek(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        fd: u64,
        offset: i64,
        whence: u64,
    ) -> i64 {
        // Directory fds are seekable too — this is what backs rewinddir (lseek
        // 0/SEEK_SET), telldir (lseek 0/SEEK_CUR), and seekdir. The position is
        // the getdents entry cursor, and the d_off cookies getdents emits are
        // exactly those cursor values, so seekdir(cookie) resumes at the right
        // entry. Without this, a second directory scan reads zero entries.
        if let Some(Fd::Dir { pos, .. }) = cx.cur.fds.get(fd as i32) {
            let cur = *pos as i64;
            let base = match whence {
                0 => 0,       // SEEK_SET
                1 | 2 => cur, // SEEK_CUR; SEEK_END has no meaningful dir size
                _ => return err(Errno::EINVAL),
            };
            let newpos = base + offset;
            if newpos < 0 {
                return err(Errno::EINVAL);
            }
            if let Some(Fd::Dir { pos, .. }) = cx.cur.fds.get_mut(fd as i32) {
                *pos = newpos as usize;
            }
            return newpos;
        }
        let (cur, path) = match cx.cur.fds.get(fd as i32) {
            Some(Fd::File { path, offset, .. }) => (offset.get(), path.clone()),
            _ => return err(Errno::ESPIPE),
        };
        let size = vfs.stat(&path).map_or(0, |a| a.size);
        // SEEK_DATA(3)/SEEK_HOLE(4): files here are non-sparse, so all data lies
        // in `[0, size)` with a single hole at EOF. `offset >= size` is ENXIO.
        const SEEK_DATA: u64 = 3;
        const SEEK_HOLE: u64 = 4;
        if whence == SEEK_DATA || whence == SEEK_HOLE {
            if offset < 0 || offset as u64 >= size {
                return err(Errno::ENXIO);
            }
            let pos = if whence == SEEK_DATA {
                offset as u64
            } else {
                size
            };
            if let Some(Fd::File { offset, .. }) = cx.cur.fds.get_mut(fd as i32) {
                offset.set(pos);
            }
            return pos as i64;
        }
        let base = match whence {
            0 => 0i64,
            1 => cur as i64,
            2 => size as i64,
            _ => return err(Errno::EINVAL),
        };
        let newpos = base + offset;
        if newpos < 0 {
            return err(Errno::EINVAL);
        }
        if let Some(Fd::File { offset, .. }) = cx.cur.fds.get_mut(fd as i32) {
            offset.set(newpos as u64);
        }
        newpos
    }

    /// `fstat(fd, statbuf)`.
    fn sys_fstat(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        fd: u64,
        statbuf: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        match Self::fd_attrs(vfs, cx, fd as i32) {
            Ok(attrs) => write_stat_or_fault(mem, statbuf, &attrs, self.arch),
            Err(e) => e,
        }
    }

    /// The metadata `fstat` reports for descriptor `fd` (`EBADF` if closed):
    /// the node for a file/dir, a synthesized char device/FIFO/socket/mqueue
    /// inode for the anonymous kinds. Shared by `fstat` and the
    /// `AT_EMPTY_PATH` forms of `newfstatat`/`statx`.
    fn fd_attrs(vfs: &mut MountTable, cx: &ServiceCtx, fd: i32) -> Result<Attrs, i64> {
        Ok(match cx.cur.fds.get(fd) {
            Some(Fd::File { path, .. } | Fd::Dir { path, .. }) => {
                let path = path.clone();
                match vfs.stat(&path) {
                    Some(a) => a,
                    None => return Err(err(Errno::ENOENT)),
                }
            }
            // A pty slave is the `/dev/pts/N` char device: resolve it through the
            // same path a bare `stat("/dev/pts/N")` takes, so the fd's `st_dev`/
            // `st_ino` match the path's — the equality `ttyname()` checks.
            Some(Fd::PtySlave(n)) => {
                let n = *n;
                vfs.stat(&format!("/dev/pts/{n}"))
                    .unwrap_or_else(stat::char_device_attrs)
            }
            // eventfd/timerfd/epoll are anonymous-inode char-device-like fds;
            // the pty master is a genuine tty char device.
            Some(
                Fd::Stdin
                | Fd::Stdout
                | Fd::Stderr
                | Fd::Eventfd(_)
                | Fd::Signalfd(_)
                | Fd::Timerfd(_)
                | Fd::Pidfd(_)
                | Fd::Epoll(_)
                | Fd::PtyMaster(_),
            ) => stat::char_device_attrs(),
            Some(Fd::PipeRead(_) | Fd::PipeWrite(_)) => stat::fifo_attrs(),
            Some(Fd::Socket { .. }) => stat::socket_attrs(),
            // A message queue is a regular file on the mqueue filesystem.
            Some(Fd::Mqueue { q, .. }) => Attrs {
                kind: NodeKind::File,
                size: 0,
                mode: 0o100_600,
                uid: 0,
                gid: 0,
                atime: 0,
                mtime: 0,
                inode: 0x4d51_0000 + *q as u64,
                nlink: 1,
                rdev: 0,
            },
            None => return Err(err(Errno::EBADF)),
        })
    }

    /// Resolve the `(dirfd, path, flags)` of `newfstatat`/`statx` to the
    /// metadata it names: `AT_EMPTY_PATH` with an empty (or, since 6.11,
    /// NULL) path is `dirfd` itself (`AT_FDCWD` = the cwd); otherwise the path,
    /// following a final symlink unless `AT_SYMLINK_NOFOLLOW`. A trailing slash
    /// on a non-directory, or a non-directory along the way, is `ENOTDIR`.
    fn stat_at(
        &self,
        vfs: &mut MountTable,
        cx: &ServiceCtx,
        dirfd: i64,
        pathptr: u64,
        flags: u64,
        mem: &GuestMemory,
    ) -> Result<Attrs, i64> {
        const AT_SYMLINK_NOFOLLOW: u64 = 0x100;
        const AT_EMPTY_PATH: u64 = 0x1000;
        let rel = if pathptr == 0 && flags & AT_EMPTY_PATH != 0 {
            String::new()
        } else {
            read_path(mem, pathptr).ok_or_else(|| err(Errno::EFAULT))?
        };
        if rel.is_empty() {
            if flags & AT_EMPTY_PATH == 0 {
                return Err(err(Errno::ENOENT));
            }
            if dirfd == AT_FDCWD {
                return vfs.stat(&cx.cur.cwd).ok_or_else(|| err(Errno::ENOENT));
            }
            return Self::fd_attrs(vfs, cx, dirfd as i32);
        }
        let had_slash = rel.len() > 1 && rel.ends_with('/');
        let mut abs = self.resolve_path(cx, dirfd, &rel);
        if flags & AT_SYMLINK_NOFOLLOW == 0 {
            abs = self.follow_or_eloop(vfs, &abs)?;
        }
        let Some(attrs) = vfs.stat(&abs) else {
            if self.has_nondir_component(vfs, &abs) {
                return Err(err(Errno::ENOTDIR));
            }
            return Err(err(Errno::ENOENT));
        };
        if had_slash && attrs.kind != NodeKind::Dir {
            return Err(err(Errno::ENOTDIR));
        }
        Ok(attrs)
    }

    /// `newfstatat(dirfd, path, statbuf, flags)`.
    #[allow(clippy::too_many_arguments)]
    fn sys_newfstatat(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        dirfd: i64,
        pathptr: u64,
        statbuf: u64,
        flags: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        // AT_SYMLINK_NOFOLLOW | AT_NO_AUTOMOUNT | AT_EMPTY_PATH.
        if flags & !(0x100 | 0x800 | 0x1000) != 0 {
            return err(Errno::EINVAL);
        }
        match self.stat_at(vfs, cx, dirfd, pathptr, flags, mem) {
            Ok(attrs) => write_stat_or_fault(mem, statbuf, &attrs, self.arch),
            Err(e) => e,
        }
    }

    /// `getdents64(fd, buf, count)`.
    #[allow(clippy::unused_self)]
    ///
    /// `legacy` selects x86-64's original `getdents` record layout
    /// (`linux_dirent`: no `d_type` after `d_reclen`, the type in the record's
    /// last byte instead) — the directory walk is otherwise identical.
    #[allow(clippy::too_many_arguments)]
    fn sys_getdents64(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        fd: u64,
        buf: u64,
        count: u64,
        legacy: bool,
        mem: &mut GuestMemory,
    ) -> i64 {
        let (path, pos) = match cx.cur.fds.get(fd as i32) {
            Some(Fd::Dir { path, pos }) => (path.clone(), *pos),
            _ => return err(Errno::ENOTDIR),
        };
        let entries = match vfs.readdir(&path) {
            Ok(e) => e,
            Err(e) => return io_errno(&e),
        };
        let mut all: Vec<(String, NodeKind, u64)> = vec![
            (".".into(), NodeKind::Dir, 1),
            ("..".into(), NodeKind::Dir, 1),
        ];
        all.extend(
            entries
                .into_iter()
                .filter(|e| !e.name.starts_with(orphan::ORPHAN_PREFIX))
                .map(|e| (e.name, e.kind, e.inode)),
        );

        let (bytes, consumed) = if legacy {
            stat::encode_dirents_legacy(&all, pos, count as usize)
        } else {
            stat::encode_dirents(&all, pos, count as usize)
        };
        if bytes.is_empty() && pos < all.len() {
            return err(Errno::EINVAL);
        }
        if mem.write(buf, &bytes).is_err() {
            return err(Errno::EFAULT);
        }
        if let Some(Fd::Dir { pos, .. }) = cx.cur.fds.get_mut(fd as i32) {
            *pos = consumed;
        }
        bytes.len() as i64
    }

    /// `getcwd(buf, size)`.
    #[allow(clippy::unused_self)]
    fn sys_getcwd(&self, cx: &mut ServiceCtx, buf: u64, size: u64, mem: &mut GuestMemory) -> i64 {
        let mut bytes = cx.cur.cwd.clone().into_bytes();
        bytes.push(0);
        if bytes.len() as u64 > size {
            return err(Errno::ERANGE);
        }
        if mem.write(buf, &bytes).is_err() {
            return err(Errno::EFAULT);
        }
        bytes.len() as i64
    }

    /// `chdir(path)`.
    fn sys_chdir(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        pathptr: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let Some(rel) = read_path(mem, pathptr) else {
            return err(Errno::EFAULT);
        };
        let abs = self.resolve_path(cx, AT_FDCWD, &rel);
        match vfs.stat(&abs) {
            Some(a) if a.kind == NodeKind::Dir => {
                cx.cur.cwd = abs;
                0
            }
            Some(_) => err(Errno::ENOTDIR),
            None => err(Errno::ENOENT),
        }
    }

    /// `fchdir(fd)` — change cwd to the directory `fd` refers to. apk's
    /// busybox post-install triggers `fchdir` back to a saved dir fd.
    #[allow(clippy::unused_self)]
    fn sys_fchdir(&self, vfs: &mut MountTable, cx: &mut ServiceCtx, fd: u64) -> i64 {
        let path = match cx.cur.fds.get(fd as i32) {
            Some(Fd::Dir { path, .. }) => path.clone(),
            Some(_) => return err(Errno::ENOTDIR),
            None => return err(Errno::EBADF),
        };
        match vfs.stat(&path) {
            Some(a) if a.kind == NodeKind::Dir => {
                cx.cur.cwd = path;
                0
            }
            _ => err(Errno::ENOTDIR),
        }
    }

    /// Resolve a possibly-relative guest path to an absolute, normalized path.
    #[allow(clippy::unused_self)]
    fn resolve_path(&self, cx: &ServiceCtx, dirfd: i64, p: &str) -> String {
        if p.starts_with('/') {
            return path::normalize(p);
        }
        let base = if dirfd == AT_FDCWD {
            cx.cur.cwd.clone()
        } else {
            match cx.cur.fds.get(dirfd as i32) {
                Some(Fd::Dir { path, .. } | Fd::File { path, .. }) => path.clone(),
                _ => cx.cur.cwd.clone(),
            }
        };
        path::normalize(&format!("{base}/{p}"))
    }

    /// True if any *non-final* component of `abs` exists but is not a directory
    /// (nor a symlink, which might point at one) — meaning a lookup that failed
    /// with a missing final component should really be `ENOTDIR`, not `ENOENT`
    /// (e.g. `stat("file/foo")` where `file` is a regular file). Consulted only
    /// on the failure path, so the common success case pays nothing. Works
    /// uniformly across backends since it walks via [`MountTable::stat`].
    #[allow(clippy::unused_self)]
    fn has_nondir_component(&self, vfs: &mut MountTable, abs: &str) -> bool {
        let comps: Vec<&str> = abs.split('/').filter(|c| !c.is_empty()).collect();
        if comps.len() < 2 {
            return false;
        }
        let mut prefix = String::new();
        for c in &comps[..comps.len() - 1] {
            prefix.push('/');
            prefix.push_str(c);
            match vfs.stat(&prefix) {
                Some(a) if a.kind != NodeKind::Dir && a.kind != NodeKind::Symlink => return true,
                // A missing intermediate component means the failure is a plain
                // ENOENT, not ENOTDIR — stop walking.
                None => return false,
                _ => {}
            }
        }
        false
    }

    /// Follow the final-component symlink chain, surfacing a cycle (or an
    /// over-long chain) as `ELOOP` (an encoded `-i64`) instead of silently
    /// falling back to the unresolved path — which would make a mutual
    /// `a -> b -> a` loop `stat`/`open` "succeed" on the link node. Returns the
    /// resolved path on success (including the case where the final component
    /// doesn't exist yet — that's `ENOENT` at the caller, not here).
    fn follow_or_eloop(&self, vfs: &mut MountTable, path: &str) -> Result<String, i64> {
        self.follow_symlinks(vfs, path)
            .ok_or_else(|| err(Errno::ELOOP))
    }

    /// Follow the final-component symlink chain (bounded), returning the target.
    /// `None` means the chain exceeded [`SYMLINK_MAX`] hops (a loop) — callers
    /// that must report `ELOOP` should go through [`Self::follow_or_eloop`].
    #[allow(clippy::unused_self)]
    fn follow_symlinks(&self, vfs: &mut MountTable, path: &str) -> Option<String> {
        let mut p = path.to_string();
        for _ in 0..SYMLINK_MAX {
            match vfs.stat(&p) {
                Some(a) if a.kind == NodeKind::Symlink => {
                    let target = vfs.readlink(&p).ok()?;
                    p = if target.starts_with('/') {
                        path::normalize(&target)
                    } else {
                        let dir = parent_of(&p);
                        path::normalize(&format!("{dir}/{target}"))
                    };
                }
                _ => return Some(p),
            }
        }
        None
    }

    /// Resolve an `execve` target: absolute-ize, then follow symlinks.
    fn resolve_exec(&self, vfs: &mut MountTable, cx: &mut ServiceCtx, p: &str) -> Option<String> {
        let abs = self.resolve_path(cx, AT_FDCWD, p);
        // `/proc/self/exe` (and `/proc/<pid>/exe`) is a magic symlink to the
        // running program — a common way to re-exec oneself. The mount table
        // has no such node, so resolve it from the live per-task path.
        if abs == "/proc/self/exe" || abs == format!("/proc/{}/exe", cx.cur.pid) {
            return (!cx.cur.exe.is_empty()).then(|| cx.cur.exe.clone());
        }
        self.follow_symlinks(vfs, &abs)
    }

    /// Read an entire file from the mount table.
    #[allow(clippy::unused_self)]
    fn read_file(&self, vfs: &mut MountTable, path: &str) -> Option<Vec<u8>> {
        let size = vfs.stat(path)?.size as usize;
        let mut buf = vec![0u8; size];
        let mut off = 0;
        while off < size {
            match vfs.read_at(path, off as u64, &mut buf[off..]) {
                Ok(0) => break,
                Ok(n) => off += n,
                Err(_) => return None,
            }
        }
        buf.truncate(off);
        Some(buf)
    }

    // ---- memory -----------------------------------------------------------

    /// `brk(addr)`.
    #[allow(clippy::unused_self)]
    fn sys_brk(&self, cx: &mut ServiceCtx, addr: u64, mem: &mut GuestMemory) -> i64 {
        if addr == 0 || addr < cx.cur.heap_start {
            return cx.cur.brk as i64;
        }
        if addr > cx.cur.brk {
            // Map from the first page NOT already backing the heap: the page a
            // mid-page `brk` sits on is live (`map` zero-fills, and glibc puts
            // its TCB — the TLS block and the stack-protector canary — in
            // early brk memory; rounding down here wiped it and every later
            // canary check "detected" smashing). A page-aligned `brk` is
            // exclusive, so its page is not yet part of the heap.
            let from = cx.cur.brk.div_ceil(PAGE_SIZE) * PAGE_SIZE;
            let to = addr.div_ceil(PAGE_SIZE) * PAGE_SIZE;
            if to > cx.cur.heap_limit
                || (to > from && mem.map(from, to - from, Prot::rw()).is_err())
            {
                return cx.cur.brk as i64;
            }
        } else if addr < cx.cur.brk {
            let from = addr.div_ceil(PAGE_SIZE) * PAGE_SIZE;
            let to = cx.cur.brk.div_ceil(PAGE_SIZE) * PAGE_SIZE;
            if to > from {
                let _ = mem.unmap(from, to - from);
            }
        }
        cx.cur.brk = addr;
        cx.cur.brk as i64
    }

    /// `mmap(addr, len, prot, flags, fd, off)`.
    ///
    /// Anonymous mappings carve from the downward-growing arena (or land at a
    /// `MAP_FIXED` address). A private (`MAP_PRIVATE`) file mapping copies the
    /// file's bytes from `off` into the fresh, zero-filled region — the
    /// mechanism the dynamic linker uses to map `ld-musl` and the shared
    /// libraries. A `MAP_SHARED` file mapping instead maps the file's pages
    /// from the shared page cache ([`pagecache`]), so every mapper — and a
    /// fork child — shares one copy that `read`/`write` stay coherent with; a
    /// writable one is also written back to the file on `munmap`/`msync`/exit.
    /// `MAP_SHARED` of `/dev/zero` is a fresh shared anonymous region, as on
    /// Linux.
    #[allow(clippy::unused_self)]
    fn sys_mmap(
        &self,
        sh: &mut Shared,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        a: &[u64; 6],
        mem: &mut GuestMemory,
    ) -> i64 {
        const MAP_SHARED: u64 = 0x01;
        const MAP_FIXED: u64 = 0x10;
        const MAP_ANONYMOUS: u64 = 0x20;
        // MAP_FIXED_NOREPLACE places at `addr` exactly (like MAP_FIXED) but fails
        // with EEXIST instead of clobbering an existing mapping in the range.
        const MAP_FIXED_NOREPLACE: u64 = 0x10_0000;

        const MAP_TYPE: u64 = 0x0f;
        const MAP_SHARED_VALIDATE: u64 = 0x03;
        const MAP_SYNC: u64 = 0x8_0000;
        // Every flag Linux defines outside the map type (incl. the hugetlb
        // size field and x86's MAP_32BIT).
        const MAP_KNOWN: u64 = 0x10
            | 0x20
            | 0x40
            | 0x100
            | 0x800
            | 0x1000
            | 0x2000
            | 0x4000
            | 0x8000
            | 0x1_0000
            | 0x2_0000
            | 0x4_0000
            | 0x8_0000
            | 0x10_0000
            | 0x400_0000
            | (0x3f << 26);
        let (addr, len, prot, flags) = (a[0], a[1], a[2], a[3]);
        let (fd, offset) = (a[4], a[5]);
        if len == 0 {
            return err(Errno::EINVAL);
        }
        // Exactly one of MAP_SHARED / MAP_PRIVATE / MAP_SHARED_VALIDATE; the
        // validating type refuses flags it doesn't know (and MAP_SYNC, which
        // needs DAX storage) with EOPNOTSUPP.
        match flags & MAP_TYPE {
            1 | 2 => {}
            MAP_SHARED_VALIDATE => {
                if flags & !(MAP_TYPE | MAP_KNOWN) != 0 || flags & MAP_SYNC != 0 {
                    return err(Errno::EOPNOTSUPP);
                }
            }
            _ => return err(Errno::EINVAL),
        }
        // PROT_READ|WRITE|EXEC, PROT_SEM, PROT_GROWSDOWN/UP, and arm64's
        // PROT_BTI/PROT_MTE.
        if prot & !(0x3f | 0x0100_0000 | 0x0200_0000) != 0 {
            return err(Errno::EINVAL);
        }
        // PR_SET_MDWE: never writable and executable at once.
        if cx.cur.pr.mdwe & prctl::MDWE_REFUSE_EXEC_GAIN != 0 && prot & 6 == 6 {
            return err(Errno::EACCES);
        }
        let len = len.div_ceil(PAGE_SIZE) * PAGE_SIZE;
        let prot = Prot((prot as u8) & 0x7);

        // For a file-backed mapping, resolve the backing path up front so a bad
        // fd fails before we disturb the address space.
        let file_src = if flags & MAP_ANONYMOUS == 0 {
            match cx.cur.fds.get(fd as i32) {
                // A shared mapping of /dev/zero is anonymous shared memory.
                Some(Fd::File { path, .. }) if path == "/dev/zero" && flags & MAP_SHARED != 0 => {
                    None
                }
                Some(Fd::File { path, .. }) => Some(path.clone()),
                Some(_) => return err(Errno::EACCES), // mmap of pipe/socket/dir
                None => return err(Errno::EBADF),
            }
        } else {
            None
        };
        if file_src.is_some() && !offset.is_multiple_of(PAGE_SIZE) {
            return err(Errno::EINVAL);
        }

        let noreplace = flags & MAP_FIXED_NOREPLACE != 0 && addr != 0;
        let base = if (flags & MAP_FIXED != 0 || noreplace) && addr != 0 {
            let base = addr - addr % PAGE_SIZE;
            // MAP_FIXED_NOREPLACE is atomic: if *any* page in the target range is
            // already mapped, fail with EEXIST and disturb nothing (don't relocate
            // and don't clobber). A free range is placed exactly at `base`.
            if noreplace {
                let end = base + len;
                let mut p = base;
                while p < end {
                    if mem.page_prot(p).is_some() {
                        return err(Errno::EEXIST);
                    }
                    p += PAGE_SIZE;
                }
            }
            // A fixed placement inside the arena is no longer free space.
            sh.arena(cx).claim(base, len);
            base
        } else {
            let Some(base) = sh.arena(cx).alloc(len) else {
                return err(Errno::ENOMEM);
            };
            base
        };
        // A writable MAP_SHARED|MAP_ANONYMOUS region must be shared across fork
        // (the standard shared-memory IPC primitive) — map it eagerly-backed and
        // shared so a forked child aliases the same frames rather than getting a
        // copy-on-write copy. Everything else is an ordinary (demand-paged) map.
        let shared_anon =
            file_src.is_none() && flags & MAP_SHARED != 0 && prot.contains(Prot::WRITE);
        // A shared file mapping maps the page cache's frames for the file.
        if flags & MAP_SHARED != 0
            && let Some(path) = file_src
        {
            if let Err(e) = self.pc_map_shared(vfs, mem, &path, offset, base, len, prot) {
                return e;
            }
            if prot.contains(Prot::WRITE) {
                cx.cur.shared_maps.push(SharedMap {
                    base,
                    len,
                    path,
                    offset,
                });
            }
            return base as i64;
        }
        let mapped = if shared_anon {
            mem.map_shared_anon(base, len, prot)
        } else {
            mem.map(base, len, prot)
        };
        if mapped.is_err() {
            return err(Errno::ENOMEM);
        }

        if let Some(path) = file_src {
            // Fill the mapping from the file: a zero-initialized page-sized
            // buffer, with the file's bytes (from `offset`, up to EOF) copied
            // over the front; the tail past EOF stays zero, as mmap requires.
            // Only the file's bytes need a host buffer; the mapping's pages
            // past EOF are already zero.
            let file_len = vfs.stat(&path).map_or(0, |a| a.size.saturating_sub(offset));
            let mut data = vec![0u8; len.min(file_len) as usize];
            let mut got = 0usize;
            while got < data.len() {
                match vfs.read_at(&path, offset + got as u64, &mut data[got..]) {
                    Ok(n) if n > 0 => got += n,
                    _ => break, // EOF or read error: leave the rest zero-filled
                }
            }
            // A private mapping snapshots the file as a reader would see it.
            self.pc_after_read(vfs, &path, offset, &mut data[..got]);
            // write_init bypasses page protection, so a read/exec-only mapping
            // (the common code-segment case) is still populated correctly.
            if mem.write_init(base, &data).is_err() {
                return err(Errno::ENOMEM);
            }
        }
        base as i64
    }

    /// Flush any writable `MAP_SHARED` file mappings overlapping `[addr, addr +
    /// len)` back to their backing files (their guest memory is the source of
    /// truth). `len == 0` flushes every shared mapping (process teardown).
    #[allow(clippy::unused_self)]
    fn flush_shared_maps(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        addr: u64,
        len: u64,
        mem: &GuestMemory,
    ) {
        let hit_all = len == 0;
        let (lo, hi) = (addr, addr.saturating_add(len));
        // Take the list out to avoid borrowing `self` twice; retained maps go back.
        let maps = std::mem::take(&mut cx.cur.shared_maps);
        for m in &maps {
            let overlaps = hit_all || (m.base < hi && m.base + m.len > lo);
            if !overlaps {
                continue;
            }
            // Don't grow the file past its real size (the mapping is page-
            // rounded, but the file was `ftruncate`d to the exact length).
            let file_size = vfs.stat(&m.path).map_or(m.len, |a| a.size);
            let writable = file_size.saturating_sub(m.offset).min(m.len);
            if writable == 0 {
                continue;
            }
            if let Ok(bytes) = mem.read_vec(m.base, writable as usize) {
                let _ = vfs.write_at(&m.path, m.offset, &bytes);
            }
        }
        // A partial munmap keeps mappings it didn't cover; a full flush drops all.
        if !hit_all {
            cx.cur.shared_maps = maps
                .into_iter()
                .filter(|m| !(m.base < hi && m.base + m.len > lo))
                .collect();
        }
    }

    /// Reserve `len` bytes (rounded up to a page) from the anonymous `mmap`
    /// arena, returning the base of the fresh region, or `None` if the arena is
    /// exhausted. Shares [`Self::sys_mmap`]'s allocator (free-list reuse, then
    /// bump) so relocating callers (`mremap` MAYMOVE) allocate the same way.
    #[allow(clippy::unused_self)]
    pub(super) fn alloc_mmap(&self, sh: &mut Shared, cx: &mut ServiceCtx, len: u64) -> Option<u64> {
        let len = len.div_ceil(PAGE_SIZE) * PAGE_SIZE;
        sh.arena(cx).alloc(len)
    }

    /// `munmap(addr, len)`.
    fn sys_munmap(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        addr: u64,
        len: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        if len == 0 {
            return err(Errno::EINVAL);
        }
        let len = len.div_ceil(PAGE_SIZE) * PAGE_SIZE;
        let base = addr - addr % PAGE_SIZE;
        // Flush any writable shared file mapping before the pages go away.
        // `sh` is held (dispatch_shared) → acquire `vfs` in the mandated order.
        if !cx.cur.shared_maps.is_empty() {
            let mut vfs = self.vfs.lock().unwrap();
            self.flush_shared_maps(&mut vfs, cx, base, len, mem);
        }
        let _ = mem.unmap(base, len);
        // Unmapping a SysV shared-memory attachment detaches it, and a shared
        // file page nobody maps any more leaves the page cache.
        sh.ipc.unmapped(cx.cur.mm, base, len, mem);
        self.pc_gc(mem);
        // Give the range back to the arena so it can be handed out again — a
        // guest that cycles mappings (a JS engine's JIT/heap blocks) would
        // otherwise exhaust the arena while most of it sat free.
        sh.arena(cx).free_range(base, len);
        0
    }

    /// `msync(addr, len, flags)` — flush a writable shared file mapping to its
    /// file without unmapping it.
    fn sys_msync(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        addr: u64,
        len: u64,
        flags: u64,
        mem: &GuestMemory,
    ) -> i64 {
        const MS_ASYNC: u64 = 1;
        const MS_INVALIDATE: u64 = 2;
        const MS_SYNC: u64 = 4;
        // Reject unknown flag bits, and MS_SYNC|MS_ASYNC together — the two are
        // mutually exclusive (one requests a blocking flush, the other a lazy
        // one), exactly as the kernel validates before touching the mapping.
        if flags & !(MS_ASYNC | MS_INVALIDATE | MS_SYNC) != 0
            || (flags & MS_SYNC != 0 && flags & MS_ASYNC != 0)
        {
            return err(Errno::EINVAL);
        }
        if !cx.cur.shared_maps.is_empty() {
            let len = len.div_ceil(PAGE_SIZE) * PAGE_SIZE;
            // msync flushes but keeps the mapping — re-add after the flush.
            let saved = cx.cur.shared_maps.clone();
            self.flush_shared_maps(vfs, cx, addr - addr % PAGE_SIZE, len.max(PAGE_SIZE), mem);
            cx.cur.shared_maps = saved;
        }
        0
    }

    /// `mprotect(addr, len, prot)`.
    #[allow(clippy::unused_self)]
    fn sys_mprotect(&self, addr: u64, len: u64, prot: u64, mem: &mut GuestMemory) -> i64 {
        if len == 0 {
            return 0;
        }
        let len = len.div_ceil(PAGE_SIZE) * PAGE_SIZE;
        match mem.protect(addr - addr % PAGE_SIZE, len, Prot((prot as u8) & 0x7)) {
            Ok(()) => 0,
            Err(_) => err(Errno::ENOMEM),
        }
    }

    // ---- misc -------------------------------------------------------------

    /// `getrandom(buf, len, flags)`.
    #[allow(clippy::unused_self)]
    ///
    /// The flags are validated (`GRND_NONBLOCK`, `GRND_RANDOM`,
    /// `GRND_INSECURE`; `RANDOM` with `INSECURE` is `EINVAL`); the pool is
    /// always "initialized", so none of them changes the output. The
    /// generator is reseeded from the host's entropy (`/dev/urandom`) when one
    /// is available, so guest keys and nonces aren't predictable from the
    /// boot time.
    fn sys_getrandom(
        &self,
        sh: &mut Shared,
        buf: u64,
        len: u64,
        flags: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const GRND_NONBLOCK: u64 = 1;
        const GRND_RANDOM: u64 = 2;
        const GRND_INSECURE: u64 = 4;
        if flags & !(GRND_NONBLOCK | GRND_RANDOM | GRND_INSECURE) != 0
            || flags & (GRND_RANDOM | GRND_INSECURE) == GRND_RANDOM | GRND_INSECURE
        {
            return err(Errno::EINVAL);
        }
        // Linux returns at most 32 MiB - 1 per call.
        let len = len.min((1 << 25) - 1);
        if sh.rng_state == 0 {
            let now = match crate::clock::now_unix().as_nanos() as u64 {
                0 => 0x9E37_79B9_7F4A_7C15,
                n => n,
            };
            sh.rng_state = (now ^ host_entropy()) | 1;
        }
        let mut out = vec![0u8; len as usize];
        for chunk in out.chunks_mut(8) {
            let mut s = sh.rng_state;
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            sh.rng_state = s;
            let bytes = s.to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
        if mem.write(buf, &out).is_err() {
            return err(Errno::EFAULT);
        }
        len as i64
    }

    /// `uname(buf)`.
    fn sys_uname(&self, sh: &Shared, buf: u64, mem: &mut GuestMemory) -> i64 {
        const FIELD: usize = 65;
        let mut data = [0u8; FIELD * 6];
        let fields: [&[u8]; 6] = [
            b"Linux",
            sh.hostname.as_bytes(),
            b"6.1.0-nixvm",
            b"#1 nixvm",
            self.arch.as_str().as_bytes(),
            sh.domainname.as_bytes(),
        ];
        for (i, f) in fields.iter().enumerate() {
            let n = f.len().min(FIELD - 1);
            data[i * FIELD..i * FIELD + n].copy_from_slice(&f[..n]);
        }
        match mem.write(buf, &data) {
            Ok(()) => 0,
            Err(_) => err(Errno::EFAULT),
        }
    }

    /// Syscalls the guest attempted that nixvm does not implement yet. Returns a
    /// snapshot (the counts live behind the kernel lock); called after the run.
    #[must_use]
    pub fn unsupported(&self) -> BTreeMap<u64, u64> {
        self.shared.lock().unwrap().unsupported.clone()
    }

    /// Subcommands of known syscalls that fell through every handler (see
    /// `Kernel::unsupported_sub`): `(syscall name, subcommand) -> count`.
    #[must_use]
    pub fn unsupported_subcommands(&self) -> BTreeMap<(&'static str, u64), u64> {
        self.unsupported_sub.lock().unwrap().clone()
    }

    /// Record that `syscall` was asked for subcommand `sub` (an ioctl request,
    /// fcntl/prctl command, sockopt `level << 32 | name`, …) that no handler
    /// recognizes. Safe to call with any other lock held (leaf lock).
    pub(super) fn note_unsupported(&self, syscall: &'static str, sub: u64) {
        *self
            .unsupported_sub
            .lock()
            .unwrap()
            .entry((syscall, sub))
            .or_default() += 1;
    }
}

/// Terminal-attribute ioctls forwarded to the host tty: `TCGETS`/`TCSETS`(`W`/
/// `F`) and `TIOCGWINSZ`/`TIOCSWINSZ`. Job-control ioctls (`TIOC[GS]PGRP`,
/// `TIOCSCTTY`) are deliberately excluded — forwarding them would drive the
/// *host's* terminal session with guest pids.
fn is_tty_ioctl(req: u32) -> bool {
    matches!(req, 0x5401 | 0x5402 | 0x5403 | 0x5404 | 0x5413 | 0x5414)
}

/// Forward a terminal ioctl to the host tty backing guest fd `host_fd` (0/1/2).
/// Guest and host are both x86-64 Linux, so `struct termios` (36 bytes) and
/// `struct winsize` (8 bytes) are byte-identical — copy the fixed-size struct
/// across. A host failure (notably `ENOTTY` when stdio is a pipe) maps straight
/// back to the guest as that negative errno, keeping `isatty()` honest.
#[cfg(all(unix, not(target_arch = "wasm32")))]
fn host_tty_ioctl(host_fd: i32, req: u32, arg: u64, mem: &mut GuestMemory) -> i64 {
    use core::ffi::{c_ulong, c_void};
    // Variadic to match the C prototype (and the vcpu crate's declaration).
    unsafe extern "C" {
        fn ioctl(fd: i32, request: c_ulong, ...) -> i32;
    }
    let (size, write) = match req {
        0x5401 => (36usize, false),    // TCGETS
        0x5402..=0x5404 => (36, true), // TCSETS/TCSETSW/TCSETSF
        0x5413 => (8, false),          // TIOCGWINSZ
        _ => (8, true),                // TIOCSWINSZ
    };
    let mut buf = if write {
        match mem.read_vec(arg, size) {
            Ok(b) => b,
            Err(_) => return err(Errno::EFAULT),
        }
    } else {
        vec![0u8; size]
    };
    // SAFETY: `host_fd` is one of this process's own std streams; `buf` is
    // exactly the `size` bytes the request reads or writes.
    let r = unsafe {
        ioctl(
            host_fd,
            c_ulong::from(req),
            buf.as_mut_ptr().cast::<c_void>(),
        )
    };
    if r < 0 {
        return -i64::from(std::io::Error::last_os_error().raw_os_error().unwrap_or(25));
    }
    if !write && mem.write(arg, &buf).is_err() {
        return err(Errno::EFAULT);
    }
    0
}

/// No host tty to forward to (wasm / non-unix): every terminal ioctl is ENOTTY.
#[cfg(not(all(unix, not(target_arch = "wasm32"))))]
fn host_tty_ioctl(_host_fd: i32, _req: u32, _arg: u64, _mem: &mut GuestMemory) -> i64 {
    err(Errno::ENOTTY)
}

impl Kernel {
    /// `clock_gettime(clk_id, timespec)`. Each Linux clock id is served from the
    /// matching source so the guest sees genuine wall / monotonic / CPU time — not
    /// wall time for all of them. `CLOCK_MONOTONIC` agrees with the vDSO fast path
    /// (both use [`crate::clock::now_monotonic`]); the CPU-time clocks the vDSO
    /// never fast-paths land here and read the scheduler's per-task accounting
    /// ([`ProcInfo::cpu_ns`]) — `CLOCK_THREAD_CPUTIME_ID` is this task, and
    /// `CLOCK_PROCESS_CPUTIME_ID` sums its whole thread group — so each guest
    /// process sees only its own CPU, not the host's. An unrecognized id falls
    /// back to the wall clock rather than failing.
    fn sys_clock_gettime(
        &self,
        cx: &ServiceCtx,
        clk_id: u64,
        ts: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const MONOTONIC: u64 = 1;
        const PROCESS_CPUTIME: u64 = 2;
        const THREAD_CPUTIME: u64 = 3;
        const MONOTONIC_RAW: u64 = 4;
        const MONOTONIC_COARSE: u64 = 6;
        const BOOTTIME: u64 = 7;
        let cpu = |ns: u128| std::time::Duration::from_nanos(u64::try_from(ns).unwrap_or(u64::MAX));
        let now = match clk_id {
            MONOTONIC | MONOTONIC_RAW | MONOTONIC_COARSE | BOOTTIME => {
                crate::clock::now_monotonic()
            }
            THREAD_CPUTIME => cpu(cx.cur.cpu_ns),
            // Sum the thread group. No `sh` is held on this call path (the fast
            // dispatch table reaches here without it), so lock it here.
            PROCESS_CPUTIME => cpu(process_cpu_ns(&self.shared.lock().unwrap(), cx)),
            _ => crate::clock::now_unix(), // REALTIME(0)/REALTIME_COARSE(5)/TAI(11)/…
        };
        let mut b = [0u8; 16];
        b[0..8].copy_from_slice(&(now.as_secs()).to_le_bytes());
        b[8..16].copy_from_slice(&u64::from(now.subsec_nanos()).to_le_bytes());
        match mem.write(ts, &b) {
            Ok(()) => 0,
            Err(_) => err(Errno::EFAULT),
        }
    }

    /// `alarm(seconds)` — arm a one-shot `ITIMER_REAL` that posts `SIGALRM` after
    /// `seconds`, returning the whole seconds left on any previous alarm (0 if
    /// none). `alarm(0)` cancels.
    #[allow(clippy::unused_self)]
    fn sys_alarm(&self, cx: &mut ServiceCtx, seconds: u64) -> i64 {
        let now = poll::now_ns();
        let remaining = cx
            .cur
            .alarm_deadline
            .map_or(0, |dl| dl.saturating_sub(now).div_ceil(1_000_000_000));
        cx.cur.alarm_interval_ns = 0;
        cx.cur.alarm_deadline = if seconds == 0 {
            None
        } else {
            Some(now + u128::from(seconds) * 1_000_000_000)
        };
        i64::try_from(remaining).unwrap_or(i64::MAX)
    }

    /// `setitimer(which, new, old)` / `getitimer(which, old)` — only `ITIMER_REAL`
    /// (which 0) is modeled (it posts `SIGALRM`); the CPU-time timers (`ITIMER_
    /// VIRTUAL`/`PROF`) are accepted as no-ops. `struct itimerval` is two
    /// `timeval`s: `it_interval` then `it_value`, each `{ i64 tv_sec; i64 tv_usec }`.
    fn sys_setitimer(
        &self,
        cx: &mut ServiceCtx,
        which: u64,
        new: u64,
        old: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const ITIMER_REAL: u64 = 0;
        let now = poll::now_ns();
        // Report the previous timer into `old` if requested (only ITIMER_REAL is
        // modeled; VIRTUAL/PROF read back as disarmed).
        if old != 0
            && self
                .write_itimer(cx, old, now, which == ITIMER_REAL, mem)
                .is_err()
        {
            return err(Errno::EFAULT);
        }
        if which != ITIMER_REAL {
            return 0; // VIRTUAL/PROF not modeled
        }
        if new == 0 {
            return 0; // query only
        }
        let (Ok(int_s), Ok(int_us), Ok(val_s), Ok(val_us)) = (
            mem.read_u64(new),
            mem.read_u64(new + 8),
            mem.read_u64(new + 16),
            mem.read_u64(new + 24),
        ) else {
            return err(Errno::EFAULT);
        };
        let interval = u128::from(int_s) * 1_000_000_000 + u128::from(int_us) * 1_000;
        let value = u128::from(val_s) * 1_000_000_000 + u128::from(val_us) * 1_000;
        cx.cur.alarm_interval_ns = interval;
        cx.cur.alarm_deadline = if value == 0 { None } else { Some(now + value) };
        0
    }

    /// Shared helper: write the current `ITIMER_REAL` state as a `struct itimerval`
    /// at `dst` (`it_interval`, then remaining `it_value`).
    #[allow(clippy::unused_self)]
    fn write_itimer(
        &self,
        cx: &ServiceCtx,
        dst: u64,
        now: u128,
        is_real: bool,
        mem: &mut GuestMemory,
    ) -> Result<(), ()> {
        let mut b = [0u8; 32];
        // Only ITIMER_REAL is modeled; the CPU-time timers read back as disarmed.
        if is_real {
            let remaining = cx.cur.alarm_deadline.map_or(0, |dl| dl.saturating_sub(now));
            let put = |b: &mut [u8; 32], off: usize, ns: u128| {
                b[off..off + 8].copy_from_slice(&((ns / 1_000_000_000) as i64).to_le_bytes());
                b[off + 8..off + 16]
                    .copy_from_slice(&((ns % 1_000_000_000 / 1_000) as i64).to_le_bytes());
            };
            put(&mut b, 0, cx.cur.alarm_interval_ns); // it_interval
            put(&mut b, 16, remaining); // it_value
        }
        mem.write(dst, &b).map_err(|_| ())
    }

    /// `getitimer(which, curr)` — write the current timer to `curr`.
    fn sys_getitimer(&self, cx: &ServiceCtx, which: u64, curr: u64, mem: &mut GuestMemory) -> i64 {
        const ITIMER_REAL: u64 = 0;
        if curr != 0
            && self
                .write_itimer(cx, curr, poll::now_ns(), which == ITIMER_REAL, mem)
                .is_err()
        {
            return err(Errno::EFAULT);
        }
        0
    }

    /// `nanosleep`/`clock_nanosleep` — genuinely suspend the caller until the
    /// deadline, using the scheduler's timed wait ([`ProcInfo::wake_deadline`]),
    /// instead of returning instantly. The absolute wall-clock deadline is seeded
    /// on the first entry and reused on every re-trap (the guest PC never advanced
    /// past the syscall), so a relative sleep converges instead of resetting.
    ///
    /// `clock_id`/`flags` come from `clock_nanosleep`; plain `nanosleep` passes a
    /// relative interval (`flags == 0`). `TIMER_ABSTIME` targets an absolute time
    /// in `clock_id`'s domain. A caught signal interrupts the sleep with `-EINTR`,
    /// writing the time remaining to `rem` (relative sleeps only), exactly as
    /// Linux does.
    fn sys_nanosleep(
        &self,
        cx: &mut ServiceCtx,
        clock_id: u64,
        flags: u64,
        req: u64,
        rem: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const TIMER_ABSTIME: u64 = 1;
        const CLOCK_REALTIME: u64 = 0;
        let abstime = flags & TIMER_ABSTIME != 0;

        // Seed the absolute wall-clock deadline once; re-traps reuse it.
        let deadline = if let Some(dl) = cx.cur.wake_deadline {
            dl
        } else {
            let (Ok(sec), Ok(nsec)) = (mem.read_u64(req), mem.read_u64(req + 8)) else {
                return err(Errno::EFAULT);
            };
            if nsec >= 1_000_000_000 || (sec as i64) < 0 {
                return err(Errno::EINVAL);
            }
            let want = u128::from(sec) * 1_000_000_000 + u128::from(nsec);
            let now_wall = poll::now_ns();
            let dl = if abstime {
                // `want` is an absolute time on `clock_id`; convert to how long
                // remains, then to a wall-clock deadline the scheduler tracks.
                let clock_now = if clock_id == CLOCK_REALTIME {
                    crate::clock::now_unix()
                } else {
                    crate::clock::now_monotonic()
                }
                .as_nanos();
                if want <= clock_now {
                    return 0; // the absolute deadline already passed
                }
                now_wall + (want - clock_now)
            } else {
                now_wall + want
            };
            cx.cur.wake_deadline = Some(dl);
            dl
        };

        let now = poll::now_ns();
        if now >= deadline {
            cx.cur.wake_deadline = None;
            if !abstime && rem != 0 {
                let _ = mem.write(rem, &[0u8; 16]); // slept the full interval
            }
            return 0;
        }
        // A caught signal cuts the sleep short: report the remaining time (a
        // relative sleep only) and fail with EINTR. Handled here (not by the
        // generic block-interrupt path) so `rem` is written correctly.
        if self.first_handled_signal(cx).is_some() {
            cx.cur.wake_deadline = None;
            if !abstime && rem != 0 {
                let left = deadline - now;
                let mut b = [0u8; 16];
                b[0..8].copy_from_slice(&((left / 1_000_000_000) as u64).to_le_bytes());
                b[8..16].copy_from_slice(&((left % 1_000_000_000) as u64).to_le_bytes());
                let _ = mem.write(rem, &b);
            }
            return err(Errno::EINTR);
        }
        // Not there yet, no signal: park until the deadline (or an earlier wake).
        // Not auto-restarted on SA_RESTART — Linux uses a restart-block that
        // re-computes the remaining time; we report EINTR instead.
        cx.block = true;
        cx.restartable = false;
        0
    }
}

/// CPU time (ns) of `cx`'s whole thread group — this task plus its checked-in
/// siblings sharing its `tgid`. `cx.cur` is out of `sh.procs` during its slice,
/// so it is added explicitly. Shared by `clock_gettime(CLOCK_PROCESS_CPUTIME_ID)`,
/// `getrusage`, and `times` so all three agree.
fn process_cpu_ns(sh: &Shared, cx: &ServiceCtx) -> u128 {
    let tgid = cx.cur.tgid;
    cx.cur.cpu_ns
        + sh.procs
            .iter()
            .flatten()
            .filter(|p| p.info.tgid == tgid)
            .map(|p| p.info.cpu_ns)
            .sum::<u128>()
}

/// Fire every real-time timer of `info` due at `now`: `ITIMER_REAL` and the
/// POSIX timers (see [`ptimer::fire_ptimers`]).
fn fire_timers_if_due(info: &mut ProcInfo, now: u128) {
    fire_alarm_if_due(info, now);
    if !info.ptimers.is_empty() {
        ptimer::fire_ptimers(info, now);
    }
}

/// Post `SIGALRM` to `info` if its `ITIMER_REAL` deadline has passed at `now`
/// (wall ns), re-arming a periodic timer or disarming a one-shot. Un-parks the
/// task so a blocking syscall wakes to take the signal. Returns whether it fired.
fn fire_alarm_if_due(info: &mut ProcInfo, now: u128) -> bool {
    let Some(dl) = info.alarm_deadline else {
        return false;
    };
    if now < dl {
        return false;
    }
    info.pending |= 1u64 << (SIGALRM - 1);
    info.parked = false;
    info.alarm_deadline = if info.alarm_interval_ns > 0 {
        // Advance past any deadlines missed while descheduled (no signal coalescing
        // beyond one pending bit, as on Linux).
        let mut next = dl + info.alarm_interval_ns;
        while next <= now {
            next += info.alarm_interval_ns;
        }
        Some(next)
    } else {
        None
    };
    true
}

/// Build a `struct rusage` (144 bytes) whose `ru_utime` carries `cpu_ns` of CPU
/// time (seconds + microseconds); every other counter is zero. Shared by
/// `getrusage` and `wait4`/`waitid` (the child-usage argument).
fn rusage_bytes(cpu_ns: u128) -> [u8; 144] {
    let mut ru = [0u8; 144];
    // ru_utime: struct timeval { i64 tv_sec; i64 tv_usec } at offset 0.
    ru[0..8].copy_from_slice(&((cpu_ns / 1_000_000_000) as i64).to_le_bytes());
    ru[8..16].copy_from_slice(&((cpu_ns % 1_000_000_000 / 1_000) as i64).to_le_bytes());
    ru
}

/// 64 bits from the host's entropy pool, or 0 where there is none (wasm).
fn host_entropy() -> u64 {
    #[cfg(all(unix, not(target_arch = "wasm32")))]
    {
        let mut b = [0u8; 8];
        if std::fs::File::open("/dev/urandom")
            .and_then(|mut f| f.read_exact(&mut b))
            .is_ok()
        {
            return u64::from_le_bytes(b);
        }
    }
    0
}

/// Encode an errno as a negative syscall return.
const fn err(e: Errno) -> i64 {
    -(e.0 as i64)
}

/// The [`xattr::XattrTarget`] of a non-`at` xattr syscall: the `f*` forms name
/// a descriptor, the `l*` forms a path whose final symlink is not followed.
fn xattr_target(sys: Sysno, arg0: u64) -> xattr::XattrTarget {
    match sys {
        Sysno::Fgetxattr | Sysno::Fsetxattr | Sysno::Flistxattr | Sysno::Fremovexattr => {
            xattr::XattrTarget::Fd(arg0 as i32)
        }
        Sysno::Lgetxattr | Sysno::Lsetxattr | Sysno::Llistxattr | Sysno::Lremovexattr => {
            xattr::XattrTarget::path(arg0, true)
        }
        _ => xattr::XattrTarget::path(arg0, false),
    }
}

/// `migrate_pages`'s target check: `pid` (0 = self) must exist.
fn attrs_check_pid(sh: &Shared, cx: &ServiceCtx, pid: u64) -> Result<(), i64> {
    let pid = pid as i32;
    if pid == 0 || pid == cx.cur.pid || sh.procs.iter().flatten().any(|p| p.info.pid == pid) {
        Ok(())
    } else {
        Err(err(Errno::ESRCH))
    }
}

/// A nodemask naming only node 0 (or nothing) — the one-node machine's
/// valid masks (`EINVAL` otherwise, `EFAULT` if unreadable).
fn attrs_nodes_ok(mem: &GuestMemory, ptr: u64, maxnode: u64) -> Result<(), i64> {
    if ptr == 0 || maxnode <= 1 {
        return Ok(());
    }
    let words = (maxnode - 1).div_ceil(64);
    for w in 0..words.min(1 << 14) {
        match mem.read_u64(ptr + w * 8) {
            Ok(v) if (w == 0 && v & !1 == 0) || v == 0 => {}
            Ok(_) => return Err(err(Errno::EINVAL)),
            Err(_) => return Err(err(Errno::EFAULT)),
        }
    }
    Ok(())
}

/// Read a NUL-terminated path string from guest memory.
fn read_path(mem: &GuestMemory, ptr: u64) -> Option<String> {
    let bytes = mem.read_cstr(ptr, 4096).ok()?;
    Some(String::from_utf8_lossy(&bytes).into_owned())
}

/// Read a NULL-terminated array of C-string pointers (argv/envp).
fn read_string_array(mem: &GuestMemory, mut ptr: u64) -> Vec<String> {
    let mut out = Vec::new();
    if ptr == 0 {
        return out;
    }
    while out.len() < 4096 {
        let Ok(p) = mem.read_u64(ptr) else { break };
        if p == 0 {
            break;
        }
        let Ok(bytes) = mem.read_cstr(p, 4096) else {
            break;
        };
        out.push(String::from_utf8_lossy(&bytes).into_owned());
        ptr += 8;
    }
    out
}

/// The parent directory of an absolute path (`/` for a top-level entry).
fn parent_of(p: &str) -> &str {
    match p.rfind('/') {
        Some(0) | None => "/",
        Some(i) => &p[..i],
    }
}

fn page_down(v: u64) -> u64 {
    v - v % PAGE_SIZE
}

/// The process group of `p` — its explicit `pgid`, or its pid when unset.
fn pgid_of(p: &ProcInfo) -> i32 {
    if p.pgid == 0 { p.pid } else { p.pgid }
}

/// The scheduling weight of a nice-0 task; `vruntime` accrues at `1×` here.
const NICE0_WEIGHT: u128 = 1024;

/// Scheduling weight for a `nice` value, on the standard CFS curve: each nice
/// level is worth ~1.25× the CPU, i.e. `weight = 1024 / 1.25^nice` (nice 0 →
/// 1024, so ~10% CPU per step). A step's `vruntime` grows by `cpu_delta *
/// NICE0_WEIGHT / nice_weight(nice)`, so a higher-`nice` (lower-weight) task's
/// vruntime climbs faster and the least-vruntime scheduler picks it less often —
/// the proportional-share behavior real Linux gives `nice`. Computed from the
/// documented 1.25-per-level rule (not copied from any table), cached once.
fn nice_weight(nice: i32) -> u128 {
    use std::sync::OnceLock;
    static W: OnceLock<[u128; 40]> = OnceLock::new();
    let table = W.get_or_init(|| {
        let mut t = [0u128; 40];
        for (i, w) in t.iter_mut().enumerate() {
            let n = i as i32 - 20; // index 0 → nice -20 … index 39 → nice 19
            *w = (1024.0 / 1.25f64.powi(n)).round().max(1.0) as u128;
        }
        t
    });
    table[(nice.clamp(-20, 19) + 20) as usize]
}

/// Add one executed step's weighted virtual runtime to a task, given the raw CPU
/// nanoseconds it consumed. Shared by the serial and SMP accounting sites so the
/// two paths order their run queues identically.
fn charge_vruntime(p: &mut ProcInfo, cpu_delta_ns: u128) {
    p.vruntime = p
        .vruntime
        .saturating_add(cpu_delta_ns.saturating_mul(NICE0_WEIGHT) / nice_weight(p.nice));
}

/// Join `argv` into the NUL-separated, NUL-terminated blob the kernel exposes
/// as `/proc/self/cmdline` (each argument followed by a `\0`).
fn cmdline_bytes(argv: &[String]) -> Vec<u8> {
    let mut out = Vec::new();
    for a in argv {
        out.extend_from_slice(a.as_bytes());
        out.push(0);
    }
    out
}

/// Derive the initial command name (`comm`) from an executable path: its final
/// path component, truncated to Linux's 15-byte `TASK_COMM_LEN - 1` limit.
fn comm_from_path(path: &str) -> String {
    let base = path.rsplit('/').next().unwrap_or(path);
    let n = base.len().min(15);
    String::from_utf8_lossy(&base.as_bytes()[..n]).into_owned()
}

/// Map a host `io::Error` to a negative guest errno.
fn io_errno(e: &io::Error) -> i64 {
    match e.raw_os_error() {
        Some(n) => -i64::from(n),
        None => err(Errno::EIO),
    }
}

/// Write `arch`'s `struct stat` for `attrs` at `addr`, or return `-EFAULT`.
fn write_stat_or_fault(mem: &mut GuestMemory, addr: u64, attrs: &Attrs, arch: Arch) -> i64 {
    let buf = stat::encode_stat(attrs, arch);
    if mem.write(addr, &buf).is_err() {
        err(Errno::EFAULT)
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fs::TmpFs;

    /// A no-op vcpu for the file/syscall unit tests.
    #[derive(Clone)]
    struct DummyVcpu;
    impl Vcpu for DummyVcpu {
        fn run(&mut self, _m: &mut GuestMemory) -> Result<Exit, VcpuError> {
            Ok(Exit::Halt)
        }
        fn syscall_nr(&self) -> u64 {
            0
        }
        fn syscall_args(&self) -> [u64; 6] {
            [0; 6]
        }
        fn set_syscall_ret(&mut self, _v: u64) {}
        fn reg(&self, _i: usize) -> u64 {
            0
        }
        fn set_reg(&mut self, _i: usize, _v: u64) {}
        fn pc(&self) -> u64 {
            0
        }
        fn set_pc(&mut self, _v: u64) {}
        fn sp(&self) -> u64 {
            0
        }
        fn set_sp(&mut self, _v: u64) {}
        fn set_tls(&mut self, _v: u64) {}
        fn fork(&self) -> Box<dyn Vcpu> {
            Box::new(self.clone())
        }
        fn reset(&mut self, _e: u64, _s: u64) {}
    }

    /// A vcpu that replays a fixed script of syscall numbers (one per `run`),
    /// then halts. Used to drive the scheduler (incl. the SMP path) without a
    /// real interpreter. A `fork` clone carries the remaining script, so a
    /// scripted `clone` syscall produces a child that finishes the rest.
    #[derive(Clone)]
    struct ScriptVcpu {
        ops: VecDeque<u64>,
        cur_nr: u64,
    }
    impl ScriptVcpu {
        fn boxed(ops: impl IntoIterator<Item = u64>) -> Box<dyn Vcpu> {
            Box::new(Self {
                ops: ops.into_iter().collect(),
                cur_nr: 0,
            })
        }
    }
    impl Vcpu for ScriptVcpu {
        fn run(&mut self, _m: &mut GuestMemory) -> Result<Exit, VcpuError> {
            match self.ops.pop_front() {
                Some(nr) => {
                    self.cur_nr = nr;
                    Ok(Exit::Syscall)
                }
                None => Ok(Exit::Halt),
            }
        }
        fn syscall_nr(&self) -> u64 {
            self.cur_nr
        }
        fn syscall_args(&self) -> [u64; 6] {
            [0; 6]
        }
        fn set_syscall_ret(&mut self, _v: u64) {}
        fn reg(&self, _i: usize) -> u64 {
            0
        }
        fn set_reg(&mut self, _i: usize, _v: u64) {}
        fn pc(&self) -> u64 {
            0
        }
        fn set_pc(&mut self, _v: u64) {}
        fn sp(&self) -> u64 {
            0
        }
        fn set_sp(&mut self, _v: u64) {}
        fn set_tls(&mut self, _v: u64) {}
        fn fork(&self) -> Box<dyn Vcpu> {
            Box::new(self.clone())
        }
        fn reset(&mut self, _e: u64, _s: u64) {}
    }

    fn kernel_only() -> Kernel {
        let mut mounts = MountTable::new();
        mounts.mount("/", Box::new(TmpFs::new()));
        Kernel::new(Arch::Aarch64, mounts)
    }

    // aarch64 syscall numbers used by the scripted SMP tests.
    const NR_READ: u64 = 63;
    const NR_GETPID: u64 = 172;
    const NR_CLONE: u64 = 220;

    /// A vcpu that keeps issuing `read(0, buf, 16)` until it gets a result
    /// (data or EOF), then halts. Models the re-trap of a blocking read: while
    /// the kernel parks the read (no `set_syscall_ret`), `run` re-issues the
    /// same syscall; once a result arrives it halts. Used to drive the
    /// interactive `pump` loop.
    #[derive(Clone)]
    struct ReadVcpu {
        buf: u64,
        done: bool,
    }
    impl Vcpu for ReadVcpu {
        fn run(&mut self, _m: &mut GuestMemory) -> Result<Exit, VcpuError> {
            if self.done {
                Ok(Exit::Halt)
            } else {
                Ok(Exit::Syscall)
            }
        }
        fn syscall_nr(&self) -> u64 {
            NR_READ
        }
        fn syscall_args(&self) -> [u64; 6] {
            [0, self.buf, 16, 0, 0, 0]
        }
        fn set_syscall_ret(&mut self, _v: u64) {
            self.done = true; // got a result (data or EOF): stop.
        }
        fn reg(&self, _i: usize) -> u64 {
            0
        }
        fn set_reg(&mut self, _i: usize, _v: u64) {}
        fn pc(&self) -> u64 {
            0
        }
        fn set_pc(&mut self, _v: u64) {}
        fn sp(&self) -> u64 {
            0
        }
        fn set_sp(&mut self, _v: u64) {}
        fn set_tls(&mut self, _v: u64) {}
        fn fork(&self) -> Box<dyn Vcpu> {
            Box::new(self.clone())
        }
        fn reset(&mut self, _e: u64, _s: u64) {}
    }

    #[test]
    fn interactive_stdin_blocks_then_delivers_then_eof() {
        let (mut k, mut mem, mut v, mut cx) = setup();
        k.set_interactive(true);
        let buf = 0x1_0000u64; // mapped by setup()

        // Empty buffer, not closed: the read parks (blocks).
        cx.block = false;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Read,
                [0, buf, 16, 0, 0, 0]
            ),
            0
        );
        assert!(cx.block, "read of empty interactive stdin blocks");

        // Feed input: the read now delivers it.
        k.feed_stdin(b"hi\n");
        cx.block = false;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Read,
                [0, buf, 16, 0, 0, 0]
            ),
            3
        );
        assert_eq!(&mem.read_vec(buf, 3).unwrap(), b"hi\n");
        assert!(!cx.block);

        // Closed + empty: EOF (0), no block.
        k.close_stdin();
        cx.block = false;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Read,
                [0, buf, 16, 0, 0, 0]
            ),
            0
        );
        assert!(!cx.block, "EOF does not block");
    }

    #[test]
    fn pump_blocks_on_empty_stdin_then_runs_to_exit_on_input() {
        let mut mounts = MountTable::new();
        mounts.mount("/", Box::new(TmpFs::new()));
        let mut k = Kernel::new(Arch::Aarch64, mounts);
        k.set_interactive(true);
        let mut mem = GuestMemory::new(0x1_0000, 16 * PAGE);
        mem.map(0x1_0000, PAGE, Prot::rw()).unwrap();

        k.boot(
            Box::new(ReadVcpu {
                buf: 0x1_0000,
                done: false,
            }),
            mem,
        );

        // Nothing to read yet: pump parks waiting for input.
        assert_eq!(k.pump().unwrap(), Pumped::Blocked);

        // Feed a line: the read completes and the task halts (exit 0).
        k.feed_stdin(b"go\n");
        assert_eq!(k.pump().unwrap(), Pumped::Exited(0));
    }

    #[test]
    fn smp_single_task_completes() {
        let mut k = kernel_only();
        k.set_ncpus(4);
        let mem = GuestMemory::new(0x1_0000, 16 * PAGE);
        // Three getpids then an implicit halt.
        let code = k
            .run(ScriptVcpu::boxed([NR_GETPID, NR_GETPID, NR_GETPID]), mem)
            .unwrap();
        assert_eq!(code, 0);
    }

    #[test]
    fn smp_fork_runs_child_on_the_pool() {
        let mut k = kernel_only();
        k.set_ncpus(4);
        let mem = GuestMemory::new(0x1_0000, 16 * PAGE);
        // pid 1: clone (fork) once, then two getpids, then halt. The child is
        // forked with the remaining script ([getpid, getpid]) and finishes it
        // on another worker thread.
        let code = k
            .run(ScriptVcpu::boxed([NR_CLONE, NR_GETPID, NR_GETPID]), mem)
            .unwrap();
        assert_eq!(code, 0, "pid 1 exits cleanly");
        assert!(
            k.shared
                .lock()
                .unwrap()
                .procs
                .iter()
                .flatten()
                .any(|p| p.info.pid == 2),
            "the forked child exists in the process table"
        );
    }

    #[test]
    fn smp_and_serial_agree() {
        let program = [NR_CLONE, NR_GETPID, NR_CLONE, NR_GETPID, NR_GETPID];
        let run_with = |ncpus: usize| {
            let mut k = kernel_only();
            k.set_ncpus(ncpus);
            let mem = GuestMemory::new(0x1_0000, 16 * PAGE);
            k.run(ScriptVcpu::boxed(program), mem).unwrap()
        };
        // The same program yields the same pid-1 exit code on 1 and 8 CPUs.
        assert_eq!(run_with(1), run_with(8));
    }

    #[test]
    fn smp_in_place_servicing_is_correct_and_deterministic() {
        // A program that forks several children interleaved with runs of
        // syscalls, so under the SMP scheduler each worker services many
        // syscalls *in place* (no per-syscall main-thread hand-off) while the
        // workers run their tasks concurrently. Exercises the big-kernel-lock
        // service path, the fork/process-table mutation under the lock, and the
        // block-free re-dispatch loop. Repeated many times to shake out any
        // scheduler race, deadlock, or nondeterminism (a race would surface as a
        // panic, a `deadlock` error from `run().unwrap()`, a hang, or a
        // mismatched result).
        let program = [
            NR_GETPID, NR_CLONE, NR_GETPID, NR_GETPID, NR_CLONE, NR_GETPID, NR_GETPID, NR_GETPID,
            NR_CLONE, NR_GETPID, NR_GETPID, NR_GETPID,
        ];
        // Run to completion and report (pid-1 exit code, number of tasks the
        // process table ended up holding) — both are deterministic functions of
        // the (deterministic) fork schedule, independent of CPU count.
        let run_with = |ncpus: usize| {
            let mut k = kernel_only();
            k.set_ncpus(ncpus);
            let mem = GuestMemory::new(0x1_0000, 16 * PAGE);
            let code = k.run(ScriptVcpu::boxed(program), mem).unwrap();
            let tasks = k.shared.lock().unwrap().procs.iter().flatten().count();
            (code, tasks)
        };
        let expected = run_with(1);
        assert_eq!(expected.1, 4, "three clones produce four tasks total");
        for _ in 0..50 {
            assert_eq!(
                run_with(4),
                expected,
                "SMP in-place servicing agrees with serial on every run"
            );
        }
    }

    const PAGE: u64 = 4096;
    const AT_CWD: u64 = (-100i64) as u64;

    fn setup() -> (Kernel, GuestMemory, DummyVcpu, ServiceCtx) {
        let mut mounts = MountTable::new();
        mounts.mount("/", Box::new(TmpFs::new()));
        let mut kernel = Kernel::new(Arch::Aarch64, mounts);
        let mut cx = ServiceCtx::default();
        cx.cur.pid = 1;
        cx.cur.tgid = 1;
        // Tests call syscall handlers directly (no boot/run), so give mm 0 its
        // mmap arena here — a small one inside the 16-page test region.
        cx.cur.mm = 0;
        kernel
            .shared
            .get_mut()
            .unwrap()
            .mmap_areas
            .push(Arena::new(0x1_8000, 0x1_5000));
        let mut mem = GuestMemory::new(0x1_0000, 16 * PAGE);
        mem.map(0x1_0000, 4 * PAGE, Prot::rw()).unwrap();
        (kernel, mem, DummyVcpu, cx)
    }

    fn call(
        k: &Kernel,
        cx: &mut ServiceCtx,
        mem: &mut GuestMemory,
        v: &mut DummyVcpu,
        s: Sysno,
        a: [u64; 6],
    ) -> i64 {
        // `dispatch` now takes its own per-handler locks; the caller must NOT
        // pre-hold `sh` (that would self-deadlock on the non-reentrant Mutex).
        k.dispatch(cx, s, 0, &a, v, mem)
    }

    #[test]
    fn nonblock_inotify_read_is_eagain_not_deadlock() {
        let (k, mut mem, mut v, mut cx) = setup();
        const IN_NONBLOCK: u64 = 0o4000;
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::InotifyInit1,
            [IN_NONBLOCK, 0, 0, 0, 0, 0],
        );
        assert!(fd >= 3);
        // The stub never delivers events; a non-blocking read must return EAGAIN
        // rather than parking (which for a lone watcher would deadlock the VM).
        let buf = 0x1_0000;
        let r = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Read,
            [fd as u64, buf, 16, 0, 0, 0],
        );
        assert_eq!(r, err(Errno::EAGAIN));
        assert!(!cx.block, "a non-blocking inotify read must not block");
    }

    #[test]
    fn nonblock_pipe_read_is_eagain_not_deadlock() {
        let (k, mut mem, mut v, mut cx) = setup();
        let fds = 0x1_0000;
        let buf = 0x1_1000;
        const O_NONBLOCK: u64 = 0o4000;
        // pipe2(fds, O_NONBLOCK).
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Pipe2,
                [fds, O_NONBLOCK, 0, 0, 0, 0]
            ),
            0
        );
        let rfd = u64::from(u32::from_le_bytes(
            mem.read_vec(fds, 4).unwrap().try_into().unwrap(),
        ));
        // Reading the empty non-blocking pipe returns EAGAIN and must NOT park.
        let r = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Read,
            [rfd, buf, 1, 0, 0, 0],
        );
        assert_eq!(r, err(Errno::EAGAIN));
        assert!(!cx.block, "a non-blocking read must not set the block flag");
    }

    #[test]
    fn fcntl_getlk_reports_unlocked() {
        let (k, mut mem, mut v, mut cx) = setup();
        let path = 0x1_0000;
        let flock = 0x1_1000;
        mem.write_init(path, b"/lk\0").unwrap();
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_CWD, path, 0o102, 0o644, 0, 0],
        );
        assert_eq!(fd, 3);
        // Caller seeds l_type = F_WRLCK (1); F_GETLK must overwrite it with
        // F_UNLCK (2) since nothing conflicts.
        mem.write_init(flock, &1u16.to_le_bytes()).unwrap();
        const F_SETLK: u64 = 6;
        const F_GETLK: u64 = 5;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Fcntl,
                [fd as u64, F_SETLK, flock, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Fcntl,
                [fd as u64, F_GETLK, flock, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            mem.read_vec(flock, 2).unwrap(),
            2u16.to_le_bytes(),
            "l_type should be F_UNLCK"
        );
    }

    #[test]
    fn write_to_readonly_fd_is_ebadf() {
        let (k, mut mem, mut v, mut cx) = setup();
        const O_WRONLY: u64 = 1;
        const O_RDWR: u64 = 2;
        const O_CREAT: u64 = 0o100;
        let path = 0x1_0000;
        let data = 0x1_1000;
        mem.write_init(path, b"/f\0").unwrap();
        mem.write_init(data, b"hi").unwrap();
        // Create + write via an O_RDWR fd works.
        let rw = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_CWD, path, O_CREAT | O_RDWR, 0o644, 0, 0],
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [rw as u64, data, 2, 0, 0, 0]
            ),
            2
        );
        // A write through an O_RDONLY fd (accmode 0) is EBADF and changes nothing.
        let ro = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_CWD, path, 0, 0, 0, 0],
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [ro as u64, data, 2, 0, 0, 0]
            ),
            err(Errno::EBADF)
        );
        // ftruncate on the read-only fd is likewise EBADF; on the writable one it works.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Ftruncate,
                [ro as u64, 0, 0, 0, 0, 0]
            ),
            err(Errno::EBADF)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Ftruncate,
                [rw as u64, 0, 0, 0, 0, 0]
            ),
            0
        );
        // Symmetric: a read through an O_WRONLY fd is EBADF.
        let wo = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_CWD, path, O_WRONLY, 0, 0, 0],
        );
        let rbuf = 0x1_2000;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Read,
                [wo as u64, rbuf, 2, 0, 0, 0]
            ),
            err(Errno::EBADF)
        );
    }

    #[test]
    fn arena_claim_removes_space_from_reuse() {
        const P: u64 = PAGE_SIZE;
        let mut a = Arena::new(0x100 * P, 0x10 * P);
        let x = a.alloc(4 * P).unwrap(); // [0xfc, 0x100)
        let y = a.alloc(4 * P).unwrap(); // [0xf8, 0xfc)
        a.free_range(x, 4 * P);
        assert!(a.is_free(x + P, 2 * P));
        // Claim the middle of the freed block: only its edges stay reusable.
        a.claim(x + P, 2 * P);
        assert!(!a.is_free(x + P, P));
        assert!(a.is_free(x, P) && a.is_free(x + 3 * P, P));
        for _ in 0..2 {
            let z = a.alloc(P).unwrap();
            assert!(z + P <= x + P || z >= x + 3 * P, "claimed pages not reused");
        }
        // A fixed placement below the cursor (never-used bump space): later
        // allocations must not land on it either, and the gap stays usable.
        let fixed = y - 8 * P;
        a.claim(fixed, 2 * P);
        let gap = a.alloc(6 * P).unwrap();
        assert_eq!(
            gap,
            fixed + 2 * P,
            "the gap above the fixed range is reused"
        );
        let below = a.alloc(P).unwrap();
        assert!(below + P <= fixed, "the bump region continues below it");
    }

    #[test]
    fn openat_rejects_bad_flag_combinations() {
        let (k, mut mem, mut v, mut cx) = setup();
        const O_WRONLY: u64 = 1;
        const O_CREAT: u64 = 0o100;
        const O_EXCL: u64 = 0o200;
        // arm64 values: `setup()` builds an aarch64 kernel.
        const O_DIRECTORY: u64 = 0o40000;
        const O_NOFOLLOW: u64 = 0o100000;
        let p = |s: &[u8], at: u64, m: &mut GuestMemory| {
            m.write_init(at, s).unwrap();
            at
        };
        // Create /f, mkdir /d, symlink /l -> f.
        let fpath = p(b"/f\0", 0x1_0000, &mut mem);
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Openat,
                [AT_CWD, fpath, O_CREAT | O_WRONLY, 0o644, 0, 0]
            ),
            3
        );
        let dpath = p(b"/d\0", 0x1_0100, &mut mem);
        call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Mkdirat,
            [AT_CWD, dpath, 0o755, 0, 0, 0],
        );
        let tgt = p(b"f\0", 0x1_0200, &mut mem);
        let lpath = p(b"/l\0", 0x1_0300, &mut mem);
        call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Symlinkat,
            [tgt, AT_CWD, lpath, 0, 0, 0],
        );

        // O_DIRECTORY on a file → ENOTDIR; O_WRONLY on a dir → EISDIR;
        // O_CREAT|O_EXCL on an existing file → EEXIST; O_NOFOLLOW on a symlink → ELOOP.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Openat,
                [AT_CWD, fpath, O_DIRECTORY, 0, 0, 0]
            ),
            err(Errno::ENOTDIR)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Openat,
                [AT_CWD, dpath, O_WRONLY, 0, 0, 0]
            ),
            err(Errno::EISDIR)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Openat,
                [AT_CWD, fpath, O_CREAT | O_EXCL | O_WRONLY, 0o644, 0, 0]
            ),
            err(Errno::EEXIST)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Openat,
                [AT_CWD, lpath, O_NOFOLLOW, 0, 0, 0]
            ),
            err(Errno::ELOOP)
        );
        // arm64's O_LARGEFILE (0o400000, x86-64's O_NOFOLLOW value) — which
        // musl sets on every open — must follow the symlink, not ELOOP.
        const O_LARGEFILE_ARM64: u64 = 0o400000;
        assert!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Openat,
                [AT_CWD, lpath, O_LARGEFILE_ARM64, 0, 0, 0]
            ) >= 0,
            "O_LARGEFILE open through a symlink follows it"
        );
    }

    #[test]
    fn openat_write_lseek_read_roundtrip() {
        let (k, mut mem, mut v, mut cx) = setup();
        let path = 0x1_0000;
        let msg = 0x1_1000;
        let buf = 0x1_2000;
        mem.write_init(path, b"/f\0").unwrap();
        mem.write_init(msg, b"Hi").unwrap();

        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_CWD, path, 0o102, 0o644, 0, 0],
        );
        assert_eq!(fd, 3);
        let fd = fd as u64;

        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [fd, msg, 2, 0, 0, 0]
            ),
            2
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Lseek,
                [fd, 0, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Read,
                [fd, buf, 2, 0, 0, 0]
            ),
            2
        );
        assert_eq!(mem.read_vec(buf, 2).unwrap(), b"Hi");

        let stbuf = 0x1_3000;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Fstat,
                [fd, stbuf, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_u64(stbuf + 48).unwrap(), 2);

        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Close,
                [fd, 0, 0, 0, 0, 0]
            ),
            0
        );
    }

    #[test]
    fn writev_gathers_iovecs() {
        use std::sync::{Arc, Mutex};
        struct Buf(Arc<Mutex<Vec<u8>>>);
        impl Write for Buf {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let (mut k, mut mem, mut v, mut cx) = setup();
        let cap = Arc::new(Mutex::new(Vec::new()));
        k.set_stdout(Box::new(Buf(cap.clone())));

        let d0 = 0x1_0000;
        let d1 = 0x1_0010;
        let iov = 0x1_0100;
        mem.write_init(d0, b"foo").unwrap();
        mem.write_init(d1, b"bar!").unwrap();
        mem.write_init(iov, &d0.to_le_bytes()).unwrap();
        mem.write_init(iov + 8, &3u64.to_le_bytes()).unwrap();
        mem.write_init(iov + 16, &d1.to_le_bytes()).unwrap();
        mem.write_init(iov + 24, &4u64.to_le_bytes()).unwrap();

        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Writev,
                [1, iov, 2, 0, 0, 0]
            ),
            7
        );
        assert_eq!(&*cap.lock().unwrap(), b"foobar!");
    }

    #[test]
    fn pipe_write_read_and_dup() {
        let (k, mut mem, mut v, mut cx) = setup();
        let fds = 0x1_0000;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Pipe2,
                [fds, 0, 0, 0, 0, 0]
            ),
            0
        );
        let rfd = u64::from(mem.read_u32(fds).unwrap());
        let wfd = u64::from(mem.read_u32(fds + 4).unwrap());
        assert!(rfd >= 3 && wfd >= 3 && rfd != wfd);

        let msg = 0x1_1000;
        mem.write_init(msg, b"pipe!").unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [wfd, msg, 5, 0, 0, 0]
            ),
            5
        );

        let dfd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Dup,
            [rfd, 0, 0, 0, 0, 0],
        );
        assert!(dfd >= 3);
        let buf = 0x1_2000;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Read,
                [dfd as u64, buf, 5, 0, 0, 0]
            ),
            5
        );
        assert_eq!(mem.read_vec(buf, 5).unwrap(), b"pipe!");

        // drained + writer still open -> blocks (returns 0 with the block flag)
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Read,
                [rfd, buf, 5, 0, 0, 0]
            ),
            0
        );
        assert!(cx.block);
    }

    #[test]
    fn read_from_stdin() {
        let (mut k, mut mem, mut v, mut cx) = setup();
        k.set_stdin(Box::new(std::io::Cursor::new(b"piped".to_vec())));
        let buf = 0x1_0000;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Read,
                [0, buf, 5, 0, 0, 0]
            ),
            5
        );
        assert_eq!(mem.read_vec(buf, 5).unwrap(), b"piped");
    }

    #[test]
    fn getrandom_fills_buffer() {
        let (k, mut mem, mut v, mut cx) = setup();
        let buf = 0x1_0000;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Getrandom,
                [buf, 16, 0, 0, 0, 0]
            ),
            16
        );
        assert!(mem.read_vec(buf, 16).unwrap().iter().any(|&b| b != 0));
    }

    #[test]
    fn clone_makes_a_child_and_wait4_reaps_it() {
        let (k, mut mem, mut v, mut cx) = setup();
        // clone(flags=0, stack=0, ...) -> child pid
        let child = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Clone,
            [0x11, 0, 0, 0, 0, 0],
        );
        assert_eq!(child, 2, "first child is pid 2");
        assert_eq!(
            k.shared.lock().unwrap().procs.len(),
            1,
            "child pushed to the process table"
        );

        // no zombie yet -> wait4 blocks
        let ws = 0x1_0000;
        call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Wait4,
            [child as u64, ws, 0, 0, 0, 0],
        );
        assert!(cx.block, "wait4 blocks while the child is alive");

        // make the child a zombie (exit code 7), then wait4 reaps it.
        if let Some(Some(p)) = k
            .shared
            .lock()
            .unwrap()
            .procs
            .iter_mut()
            .find(|s| s.as_ref().is_some_and(|p| p.info.pid == 2))
        {
            p.info.run = RunState::Zombie(ExitCause::Exited(7));
        }
        let reaped = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Wait4,
            [child as u64, ws, 0, 0, 0, 0],
        );
        assert_eq!(reaped, 2);
        // WIFEXITED status: (code & 0xff) << 8
        assert_eq!(mem.read_u32(ws).unwrap(), 7 << 8);
    }

    #[test]
    fn wait4_encodes_a_signal_death_as_wifsignaled() {
        let (k, mut mem, mut v, mut cx) = setup();
        // A child (pid 2) of the caller, killed by SIGKILL (9).
        let child = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Clone,
            [0, 0, 0, 0, 0, 0],
        );
        assert_eq!(child, 2);
        if let Some(Some(p)) = k
            .shared
            .lock()
            .unwrap()
            .procs
            .iter_mut()
            .find(|s| s.as_ref().is_some_and(|p| p.info.pid == 2))
        {
            p.info.run = RunState::Zombie(ExitCause::Signaled(9));
        }
        let ws = 0x1_0000;
        let reaped = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Wait4,
            [child as u64, ws, 0, 0, 0, 0],
        );
        assert_eq!(reaped, 2);
        // WIFSIGNALED: the low 7 bits are the signal, no (code << 8) exit part.
        let status = mem.read_u32(ws).unwrap();
        assert_eq!(status & 0x7f, 9, "termsig should be SIGKILL");
        assert_eq!(
            status & 0xff00,
            0,
            "no WIFEXITED exit code for a signal death"
        );
    }

    /// Helper: set a table process's `run` state, by pid.
    fn set_run(k: &Kernel, pid: i32, run: RunState) {
        for slot in k.shared.lock().unwrap().procs.iter_mut().flatten() {
            if slot.info.pid == pid {
                slot.info.run = run;
            }
        }
    }

    /// Helper: read a table process's fields, by pid.
    fn proc_field<T>(k: &Kernel, pid: i32, f: impl Fn(&ProcInfo) -> T) -> T {
        for slot in k.shared.lock().unwrap().procs.iter().flatten() {
            if slot.info.pid == pid {
                return f(&slot.info);
            }
        }
        panic!("no proc pid {pid}");
    }

    #[test]
    fn sigstop_delivery_stops_the_task_and_notifies_the_parent() {
        // The current task is a child (pid 2, ppid 1) with a parent in the table.
        let (k, mut mem, mut v, mut cx) = setup();
        k.shared
            .lock()
            .unwrap()
            .procs
            .push(Some(make_proc(1, 1, 0, false)));
        cx.cur.pid = 2;
        cx.cur.ppid = 1;
        // SIGSTOP (19) pending → deliver_pending_signals stops it (uncatchable).
        cx.cur.pending = 1 << (19 - 1);
        k.deliver_pending_signals(&mut cx, &mut v, &mut mem);
        assert!(
            matches!(cx.cur.run, RunState::Stopped(19)),
            "SIGSTOP stops the task"
        );
        assert!(!cx.cur.stop_reported, "a fresh stop is unreported");
        // The parent got SIGCHLD (17) posted and was unparked.
        assert_eq!(
            proc_field(&k, 1, |p| p.pending) & (1 << (17 - 1)),
            1 << (17 - 1)
        );
        assert!(!proc_field(&k, 1, |p| p.parked));
    }

    #[test]
    fn sigtstp_disposition_governs_whether_it_stops() {
        // SIGTSTP (20) is a *catchable* stop: at SIG_IGN it is dropped, not a
        // stop (the stop guard only fires at SIG_DFL / for SIGSTOP). SIGSTOP,
        // by contrast, always stops regardless of disposition.
        let (k, mut mem, mut v, mut cx) = setup();
        const SIG_IGN: u64 = 1;
        // SIGTSTP ignored → dropped, task stays Running.
        cx.cur.handlers[20] = SigAction {
            handler: SIG_IGN,
            flags: 0,
            restorer: 0,
            mask: 0,
        };
        cx.cur.pending = 1 << (20 - 1);
        k.deliver_pending_signals(&mut cx, &mut v, &mut mem);
        assert!(
            matches!(cx.cur.run, RunState::Running),
            "an ignored SIGTSTP does not stop"
        );
        assert_eq!(cx.cur.pending & (1 << (20 - 1)), 0, "and is dropped");
        // SIGTSTP at SIG_DFL → stop.
        cx.cur.handlers[20] = SigAction::default();
        cx.cur.pending = 1 << (20 - 1);
        k.deliver_pending_signals(&mut cx, &mut v, &mut mem);
        assert!(
            matches!(cx.cur.run, RunState::Stopped(20)),
            "a default SIGTSTP stops"
        );
    }

    #[test]
    fn wait4_wuntraced_reports_a_stopped_child_then_latches() {
        let (k, mut mem, mut v, mut cx) = setup();
        let child = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Clone,
            [0, 0, 0, 0, 0, 0],
        );
        assert_eq!(child, 2);
        set_run(&k, 2, RunState::Stopped(19));
        let ws = 0x1_0000;
        // WUNTRACED (0x2): WIFSTOPPED, WSTOPSIG == 19, child NOT reaped.
        let r = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Wait4,
            [child as u64, ws, 2, 0, 0, 0],
        );
        assert_eq!(r, 2);
        let st = mem.read_u32(ws).unwrap();
        assert_eq!(st & 0xff, 0x7f, "WIFSTOPPED");
        assert_eq!((st >> 8) & 0xff, 19, "WSTOPSIG == SIGSTOP");
        assert!(
            proc_field(&k, 2, |p| p.stop_reported),
            "the stop is latched"
        );
        // A second WUNTRACED wait doesn't re-report the same stop — it blocks.
        cx.block = false;
        let r2 = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Wait4,
            [child as u64, ws, 2, 0, 0, 0],
        );
        assert_eq!(r2, 0);
        assert!(cx.block, "an already-reported stop doesn't re-report");
    }

    #[test]
    fn sigcont_resumes_a_stopped_child_and_wcontinued_reports_it() {
        let (k, mut mem, mut v, mut cx) = setup();
        let child = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Clone,
            [0, 0, 0, 0, 0, 0],
        );
        assert_eq!(child, 2);
        set_run(&k, 2, RunState::Stopped(19));
        // kill(child, SIGCONT=18) resumes it and latches "continued".
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Kill,
                [2, 18, 0, 0, 0, 0]
            ),
            0
        );
        assert!(
            matches!(proc_field(&k, 2, |p| p.run), RunState::Running),
            "SIGCONT resumes"
        );
        assert!(proc_field(&k, 2, |p| p.continued), "continued latched");
        assert!(!proc_field(&k, 2, |p| p.parked), "and it's runnable again");
        // The parent (pid 1, the current task) got SIGCHLD from the resume.
        assert_eq!(cx.cur.pending & (1 << (17 - 1)), 1 << (17 - 1));
        // WCONTINUED (0x8): WIFCONTINUED == 0xffff, latch cleared, not reaped.
        let ws = 0x1_0000;
        let r = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Wait4,
            [child as u64, ws, 8, 0, 0, 0],
        );
        assert_eq!(r, 2);
        assert_eq!(mem.read_u32(ws).unwrap(), 0xffff, "WIFCONTINUED");
        assert!(
            !proc_field(&k, 2, |p| p.continued),
            "continued cleared after report"
        );
    }

    #[test]
    fn waitid_reports_stop_and_continue_via_siginfo() {
        let (k, mut mem, mut v, mut cx) = setup();
        let child = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Clone,
            [0, 0, 0, 0, 0, 0],
        );
        assert_eq!(child, 2);
        set_run(&k, 2, RunState::Stopped(19));
        let si = 0x1_0000;
        const P_PID: u64 = 1;
        const WSTOPPED: u64 = 2;
        const WCONTINUED: u64 = 8;
        const WNOWAIT: u64 = 0x0100_0000;
        // WNOWAIT stop report: CLD_STOPPED(5), si_status=19, and NOT latched.
        let r = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Waitid,
            [P_PID, 2, si, WSTOPPED | WNOWAIT, 0, 0],
        );
        assert_eq!(r, 0);
        assert_eq!(mem.read_u32(si).unwrap(), 17, "si_signo == SIGCHLD");
        assert_eq!(mem.read_u32(si + 8).unwrap(), 5, "si_code == CLD_STOPPED");
        assert_eq!(mem.read_u32(si + 16).unwrap(), 2, "si_pid == child");
        assert_eq!(mem.read_u32(si + 24).unwrap(), 19, "si_status == SIGSTOP");
        assert!(
            !proc_field(&k, 2, |p| p.stop_reported),
            "WNOWAIT does not latch"
        );
        // Now a real (non-WNOWAIT) WSTOPPED wait latches it.
        call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Waitid,
            [P_PID, 2, si, WSTOPPED, 0, 0],
        );
        assert!(proc_field(&k, 2, |p| p.stop_reported));
        // Continue it and report CLD_CONTINUED(6), si_status = SIGCONT(18).
        call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Kill,
            [2, 18, 0, 0, 0, 0],
        );
        let r = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Waitid,
            [P_PID, 2, si, WCONTINUED, 0, 0],
        );
        assert_eq!(r, 0);
        assert_eq!(mem.read_u32(si + 8).unwrap(), 6, "si_code == CLD_CONTINUED");
        assert_eq!(mem.read_u32(si + 24).unwrap(), 18, "si_status == SIGCONT");
    }

    #[test]
    fn stop_and_cont_pending_bits_annihilate() {
        // Posting SIGCONT clears a pending stop; posting a stop clears pending CONT.
        let (k, mut mem, mut v, mut cx) = setup();
        let child = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Clone,
            [0, 0, 0, 0, 0, 0],
        );
        assert_eq!(child, 2);
        // Pending SIGTSTP(20), then SIGCONT(18) cancels it.
        call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Kill,
            [2, 20, 0, 0, 0, 0],
        );
        assert_eq!(
            proc_field(&k, 2, |p| p.pending) & (1 << (20 - 1)),
            1 << (20 - 1)
        );
        call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Kill,
            [2, 18, 0, 0, 0, 0],
        );
        assert_eq!(
            proc_field(&k, 2, |p| p.pending) & (1 << (20 - 1)),
            0,
            "SIGCONT cancels the pending stop"
        );
        // Pending SIGCONT(18), then SIGSTOP(19) cancels it.
        call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Kill,
            [2, 18, 0, 0, 0, 0],
        );
        call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Kill,
            [2, 19, 0, 0, 0, 0],
        );
        assert_eq!(
            proc_field(&k, 2, |p| p.pending) & (1 << (18 - 1)),
            0,
            "a stop cancels the pending SIGCONT"
        );
    }

    #[test]
    fn vfork_copies_the_address_space_but_a_thread_shares_it() {
        // vfork = CLONE_VM | CLONE_VFORK (no CLONE_THREAD). Real Linux lets the
        // child borrow the parent's mm until it execs, but this kernel's execve
        // replaces the space in place, so a shared slot would be clobbered out
        // from under the parent (vi/sh fighting for the console). vfork must get
        // its own copied space; only genuine threads keep sharing.
        const CLONE_VM: u64 = 0x0000_0100;
        const CLONE_VFORK: u64 = 0x0000_4000;
        const CLONE_THREAD: u64 = 0x0001_0000;

        // Give the parent a real address-space slot at index 0.
        let (k, mut mem, mut v, mut cx) = setup();
        k.shared
            .lock()
            .unwrap()
            .spaces
            .push(Arc::new(Mutex::new(mem.fork())));

        cx.cur.mm = 0;

        let child = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Clone,
            [CLONE_VM | CLONE_VFORK, 0, 0, 0, 0, 0],
        );
        let cmm = k
            .shared
            .lock()
            .unwrap()
            .procs
            .iter()
            .flatten()
            .find(|p| p.info.pid == child as i32)
            .unwrap()
            .info
            .mm;
        assert_ne!(
            cmm, cx.cur.mm,
            "vfork child gets its own copied address space"
        );

        let thread = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Clone,
            [CLONE_VM | CLONE_THREAD | 0x800, 0, 0, 0, 0, 0],
        );
        let tmm = k
            .shared
            .lock()
            .unwrap()
            .procs
            .iter()
            .flatten()
            .find(|p| p.info.pid == thread as i32)
            .unwrap()
            .info
            .mm;
        assert_eq!(
            tmm, cx.cur.mm,
            "a real thread shares the caller's address space"
        );
    }

    /// Build a bare task record for scheduler/thread-table tests.
    fn make_proc(pid: i32, tgid: i32, mm: usize, is_thread: bool) -> Process {
        let mut info = ProcInfo {
            pid,
            tgid,
            is_thread,
            mm,
            ..ProcInfo::default()
        };
        info.run = RunState::Running;
        Process {
            vcpu: Some(Box::new(DummyVcpu)),
            info,
        }
    }

    #[test]
    fn getpid_is_tgid_gettid_is_pid() {
        let (k, mut mem, mut v, mut cx) = setup();
        cx.cur.pid = 7; // a thread's tid
        cx.cur.tgid = 1; // its process
        assert_eq!(
            call(&k, &mut cx, &mut mem, &mut v, Sysno::Getpid, [0; 6]),
            1
        );
        assert_eq!(
            call(&k, &mut cx, &mut mem, &mut v, Sysno::Gettid, [0; 6]),
            7
        );
    }

    #[test]
    fn clone_thread_shares_tgid_and_address_space() {
        let (k, mut mem, mut v, mut cx) = setup();
        // CLONE_VM | CLONE_THREAD | CLONE_SETTLS
        let flags = 0x0000_0100 | 0x0000_0800 | 0x0001_0000 | 0x0008_0000;
        let tid = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Clone,
            [flags, 0x2_0000, 0, 0xdead_beef, 0, 0],
        );
        assert_eq!(tid, 2, "new thread gets a fresh tid");
        let sh = k.shared.lock().unwrap();
        let spaces_before = sh.spaces.len();
        let child = sh
            .procs
            .iter()
            .flatten()
            .find(|p| p.info.pid == 2)
            .expect("thread in table");
        assert!(child.info.is_thread);
        assert_eq!(child.info.tgid, cx.cur.tgid, "thread shares the tgid");
        assert_eq!(child.info.mm, cx.cur.mm, "thread shares the address space");
        assert_eq!(
            spaces_before,
            sh.spaces.len(),
            "CLONE_VM does not allocate a new address space"
        );
    }

    #[test]
    fn fork_gets_its_own_address_space() {
        let (k, mut mem, mut v, mut cx) = setup();
        // Put the parent's space in the table (as run() would).
        k.shared
            .lock()
            .unwrap()
            .spaces
            .push(Arc::new(Mutex::new(GuestMemory::new(0x1_0000, PAGE))));
        cx.cur.mm = 0;
        let before = k.shared.lock().unwrap().spaces.len();
        // flags = SIGCHLD only (a plain fork), no CLONE_VM.
        let child = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Clone,
            [0x11, 0, 0, 0, 0, 0],
        );
        assert_eq!(child, 2);
        let sh = k.shared.lock().unwrap();
        let c = sh.procs.iter().flatten().find(|p| p.info.pid == 2).unwrap();
        assert!(!c.info.is_thread);
        assert_eq!(c.info.tgid, 2, "a forked process is its own group");
        assert_ne!(c.info.mm, cx.cur.mm, "fork copies the address space");
        assert_eq!(sh.spaces.len(), before + 1);
    }

    #[test]
    fn clone_records_the_exit_signal_and_thread_has_none() {
        let (k, mut mem, mut v, mut cx) = setup();
        // A plain fork (flags = SIGCHLD in the low byte) must record SIGCHLD (17).
        let c1 = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Clone,
            [0x11, 0, 0, 0, 0, 0],
        );
        // A clone requesting SIGUSR1 (10) as the exit signal records exactly that.
        let c2 = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Clone,
            [10, 0, 0, 0, 0, 0],
        );
        // A thread (CLONE_VM|CLONE_THREAD) has no exit signal at all.
        let t = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Clone,
            [0x0000_0100 | 0x0000_0800 | 0x0001_0000, 0, 0, 0, 0, 0],
        );
        let sh = k.shared.lock().unwrap();
        let sig = |pid: i64| {
            sh.procs
                .iter()
                .flatten()
                .find(|p| i64::from(p.info.pid) == pid)
                .unwrap()
                .info
                .exit_signal
        };
        assert_eq!(sig(c1), 17, "fork signals SIGCHLD");
        assert_eq!(sig(c2), 10, "clone honors a custom exit signal");
        assert_eq!(sig(t), 0, "a thread has no exit signal");
    }

    #[test]
    fn exiting_child_posts_its_exit_signal_to_the_parent() {
        let (k, mut mem, mut v, mut cx) = setup();
        // A parent process (pid 100) sits parked in the table waiting on a child.
        {
            let mut sh = k.shared.lock().unwrap();
            let mut parent = make_proc(100, 100, 0, false);
            parent.info.parked = true;
            sh.procs.push(Some(parent));
        }
        // The running task is that child: pid 200, parent 100, exit signal SIGUSR1.
        cx.cur.pid = 200;
        cx.cur.tgid = 200;
        cx.cur.ppid = 100;
        cx.cur.exit_signal = 10; // SIGUSR1
        call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Exit,
            [0, 0, 0, 0, 0, 0],
        );
        let sh = k.shared.lock().unwrap();
        let parent = sh
            .procs
            .iter()
            .flatten()
            .find(|p| p.info.pid == 100)
            .unwrap();
        assert!(
            parent.info.pending & (1 << (10 - 1)) != 0,
            "parent gets SIGUSR1, not SIGCHLD"
        );
        assert!(
            parent.info.pending & (1 << (17 - 1)) == 0,
            "no spurious SIGCHLD"
        );
        assert!(
            !parent.info.parked,
            "the parent is unparked so its wait re-checks"
        );
    }

    #[test]
    fn clone_parent_makes_the_child_a_sibling() {
        let (k, mut mem, mut v, mut cx) = setup();
        // The caller (pid 1) itself has a parent (pid 50).
        cx.cur.ppid = 50;
        const CLONE_PARENT: u64 = 0x0000_8000;
        // A plain fork's child is parented to the caller...
        let plain = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Clone,
            [0x11, 0, 0, 0, 0, 0],
        );
        // ...but CLONE_PARENT parents the child to the caller's parent (a sibling).
        let sib = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Clone,
            [CLONE_PARENT | 0x11, 0, 0, 0, 0, 0],
        );
        let sh = k.shared.lock().unwrap();
        let ppid = |pid: i64| {
            sh.procs
                .iter()
                .flatten()
                .find(|p| i64::from(p.info.pid) == pid)
                .unwrap()
                .info
                .ppid
        };
        assert_eq!(ppid(plain), 1, "a plain fork's parent is the caller");
        assert_eq!(
            ppid(sib),
            50,
            "CLONE_PARENT reparents to the caller's parent"
        );
    }

    #[test]
    fn clone_fs_shares_the_cwd_slot_a_fork_copies_it() {
        let (k, mut mem, mut v, mut cx) = setup();
        // Seed the caller's cwd slot (index 0) so a fork's fresh slot is distinct.
        k.shared
            .lock()
            .unwrap()
            .cwd_tables
            .push(Some("/".to_string()));
        cx.cur.fs = 0;
        const CLONE_FS: u64 = 0x0000_0200;
        let shared = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Clone,
            [CLONE_FS | 0x11, 0, 0, 0, 0, 0],
        );
        let forked = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Clone,
            [0x11, 0, 0, 0, 0, 0],
        );
        let sh = k.shared.lock().unwrap();
        let fs = |pid: i64| {
            sh.procs
                .iter()
                .flatten()
                .find(|p| i64::from(p.info.pid) == pid)
                .unwrap()
                .info
                .fs
        };
        assert_eq!(fs(shared), 0, "CLONE_FS shares the caller's cwd slot");
        assert_ne!(fs(forked), 0, "a fork gets its own cwd slot");
    }

    #[test]
    fn clone_pidfd_yields_an_fd_that_polls_ready_after_the_child_exits() {
        let (k, mut mem, mut v, mut cx) = setup();
        const CLONE_PIDFD: u64 = 0x0000_1000;
        let pidfd_out = 0x1_0000; // where the kernel writes the pidfd number
        let child = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Clone,
            [CLONE_PIDFD | 0x11, 0, pidfd_out, 0, 0, 0],
        );
        let pidfd = u64::from(mem.read_u32(pidfd_out).unwrap());
        assert!(pidfd >= 3, "a real fd was allocated for the pidfd");
        // Poll it before the child exits: not ready.
        let pollfds = 0x1_2000;
        mem.write_init(pollfds, &(pidfd as u32).to_le_bytes())
            .unwrap();
        mem.write_init(pollfds + 4, &1u16.to_le_bytes()).unwrap(); // POLLIN
        mem.write_init(pollfds + 6, &0u16.to_le_bytes()).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Poll,
                [pollfds, 1, 0, 0, 0, 0]
            ),
            0,
            "pidfd not ready while the child lives"
        );
        // The child exits (its proc stays in the table sharing mm, so nothing is
        // freed): drive its exit through a child ServiceCtx.
        let child_mm = k
            .shared
            .lock()
            .unwrap()
            .procs
            .iter()
            .flatten()
            .find(|p| i64::from(p.info.pid) == child)
            .unwrap()
            .info
            .mm;
        let mut cx_child = ServiceCtx::default();
        cx_child.cur.pid = child as i32;
        cx_child.cur.tgid = child as i32;
        cx_child.cur.ppid = 1;
        cx_child.cur.mm = child_mm;
        call(
            &k,
            &mut cx_child,
            &mut mem,
            &mut v,
            Sysno::Exit,
            [0, 0, 0, 0, 0, 0],
        );
        // Now the pidfd is POLLIN-readable.
        mem.write_init(pollfds + 6, &0u16.to_le_bytes()).unwrap();
        let n = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Poll,
            [pollfds, 1, 0, 0, 0, 0],
        );
        assert_eq!(n, 1, "pidfd is ready once the child exits");
        assert_eq!(
            mem.read_vec(pollfds + 6, 2).unwrap(),
            1u16.to_le_bytes(),
            "revents = POLLIN"
        );
    }

    #[test]
    fn clone_rejects_an_unknown_flag_bit() {
        let (k, mut mem, mut v, mut cx) = setup();
        // Bit 34 (0x4_0000_0000) is above every defined clone flag.
        let r = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Clone,
            [0x4_0000_0000, 0, 0, 0, 0, 0],
        );
        assert_eq!(
            r,
            err(Errno::EINVAL),
            "an undefined high flag bit is EINVAL"
        );
    }

    #[test]
    fn clone_into_a_new_namespace_as_root_succeeds() {
        let (k, mut mem, mut v, mut cx) = setup();
        // As root, real Linux creates the namespace and succeeds; nixvm accepts
        // the flags (single global namespace) rather than returning EPERM.
        const CLONE_NEWUSER: u64 = 0x1000_0000;
        const CLONE_NEWNET: u64 = 0x4000_0000;
        const CLONE_NEWPID: u64 = 0x2000_0000;
        let child = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Clone,
            [
                CLONE_NEWUSER | CLONE_NEWNET | CLONE_NEWPID | 0x11,
                0,
                0,
                0,
                0,
                0,
            ],
        );
        assert!(
            child > 0,
            "a namespace clone as root produces a child, not EPERM"
        );
    }

    #[test]
    fn clone3_honors_the_exit_signal_and_pidfd_fields() {
        let (k, mut mem, mut v, mut cx) = setup();
        const CLONE_PIDFD: u64 = 0x0000_1000;
        let args = 0x1_0000;
        let pidfd_out: u64 = 0x1_1000;
        // struct clone_args, 64 bytes: flags@0, pidfd@8, child_tid@16,
        // parent_tid@24, exit_signal@32, stack@40, stack_size@48, tls@56.
        for off in (0..64).step_by(8) {
            mem.write_init(args + off, &0u64.to_le_bytes()).unwrap();
        }
        mem.write_init(args, &CLONE_PIDFD.to_le_bytes()).unwrap();
        mem.write_init(args + 8, &pidfd_out.to_le_bytes()).unwrap();
        mem.write_init(args + 32, &10u64.to_le_bytes()).unwrap(); // exit_signal SIGUSR1
        let child = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Clone3,
            [args, 64, 0, 0, 0, 0],
        );
        assert!(child > 0);
        let pidfd = mem.read_u32(pidfd_out).unwrap();
        assert!(
            pidfd >= 3,
            "clone3 wrote a pidfd into the struct's pidfd field"
        );
        let sh = k.shared.lock().unwrap();
        let c = sh
            .procs
            .iter()
            .flatten()
            .find(|p| i64::from(p.info.pid) == child)
            .unwrap();
        assert_eq!(
            c.info.exit_signal, 10,
            "clone3 exit_signal is its own field, not flags"
        );
    }

    #[test]
    fn exit_group_zombies_the_whole_thread_group() {
        let (k, mut mem, mut v, mut cx) = setup();
        // Two sibling threads in the leader's group, plus an unrelated process.
        k.shared
            .lock()
            .unwrap()
            .procs
            .push(Some(make_proc(2, 1, 0, true)));
        k.shared
            .lock()
            .unwrap()
            .procs
            .push(Some(make_proc(3, 1, 0, true)));
        k.shared
            .lock()
            .unwrap()
            .procs
            .push(Some(make_proc(4, 4, 1, false)));

        call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::ExitGroup,
            [42, 0, 0, 0, 0, 0],
        );

        assert!(
            matches!(cx.cur.run, RunState::Zombie(ExitCause::Exited(42))),
            "leader exits"
        );
        let state = |pid| {
            k.shared
                .lock()
                .unwrap()
                .procs
                .iter()
                .flatten()
                .find(|p| p.info.pid == pid)
                .map(|p| p.info.run)
        };
        assert_eq!(
            state(2),
            Some(RunState::Zombie(ExitCause::Exited(42))),
            "sibling thread killed"
        );
        assert_eq!(
            state(3),
            Some(RunState::Zombie(ExitCause::Exited(42))),
            "sibling thread killed"
        );
        assert_eq!(
            state(4),
            Some(RunState::Running),
            "unrelated process untouched"
        );
    }

    #[test]
    fn futex_wake_releases_a_parked_waiter() {
        let (k, mut mem, mut v, mut cx) = setup();
        let uaddr = 0x1_0000;
        // A sibling parked in FUTEX_WAIT on (mm 0, uaddr).
        let mut waiter = make_proc(2, 1, 0, true);
        waiter.info.futex_wait = Some(futex::FutexKey::Private(0, uaddr));
        k.shared.lock().unwrap().procs.push(Some(waiter));

        // FUTEX_WAKE(uaddr, op=1, val=1) wakes exactly one waiter.
        let woken = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Futex,
            [uaddr, 1, 1, 0, 0, 0],
        );
        assert_eq!(woken, 1);
        let sh = k.shared.lock().unwrap();
        let w = sh.procs.iter().flatten().find(|p| p.info.pid == 2).unwrap();
        assert!(w.info.futex_woken, "waiter flagged for release");
    }

    #[test]
    fn clock_gettime_reads_each_clock_from_its_own_domain() {
        let (k, mut mem, _v, cx) = setup();
        let buf = 0x1_0000; // mapped rw by `setup`
        mem.map(buf, PAGE, Prot::rw()).unwrap();
        let mut secs = |clk: u64| -> i64 {
            assert_eq!(k.sys_clock_gettime(&cx, clk, buf, &mut mem), 0);
            let b = mem.read_vec(buf, 16).unwrap();
            i64::from_le_bytes(b[0..8].try_into().unwrap())
        };
        let realtime = secs(0); // CLOCK_REALTIME  -> wall seconds (post-2020)
        let monotonic = secs(1); // CLOCK_MONOTONIC -> since boot/start
        let proc_cpu = secs(2); // CLOCK_PROCESS_CPUTIME_ID -> CPU seconds
        let thr_cpu = secs(3); // CLOCK_THREAD_CPUTIME_ID
        assert!(realtime > 1_600_000_000, "realtime is a real wall time");
        // The monotonic and CPU clocks must NOT read as the wall epoch — the old
        // bug returned wall time for every id.
        assert!(monotonic < realtime, "monotonic is not the wall epoch");
        assert!(
            (0..realtime).contains(&proc_cpu),
            "process-cpu is not the wall epoch"
        );
        assert!(
            (0..realtime).contains(&thr_cpu),
            "thread-cpu is not the wall epoch"
        );
    }

    #[test]
    fn getrusage_and_times_report_cpu_time_not_zero_or_wall() {
        let (k, mut mem, _v, mut cx) = setup();
        let buf = 0x1_0000;
        mem.map(buf, PAGE, Prot::rw()).unwrap();
        // Give the task some accounted CPU so getrusage/times report it.
        cx.cur.cpu_ns = 250_000_000; // 250 ms
        let sh = k.shared.lock().unwrap();
        // getrusage(RUSAGE_SELF): ru_utime carries the CPU time (not the old
        // all-zeros, not wall time).
        assert_eq!(k.sys_getrusage(&sh, &cx, 0, buf, &mut mem), 0);
        let ru = mem.read_vec(buf, 144).unwrap();
        let sec = i64::from_le_bytes(ru[0..8].try_into().unwrap());
        let usec = i64::from_le_bytes(ru[8..16].try_into().unwrap());
        assert_eq!(sec, 0, "250 ms is 0 whole seconds");
        assert_eq!(usec, 250_000, "ru_utime microseconds reflect cpu_ns");
        // times(): tms_utime carries the CPU time in ticks (250 ms = 25 ticks),
        // the return (real elapsed ticks) is non-negative.
        let ret = k.sys_times(&sh, &cx, buf, &mut mem);
        assert!(ret >= 0, "times returns elapsed ticks");
        let tms = mem.read_vec(buf, 32).unwrap();
        assert_eq!(
            i64::from_le_bytes(tms[0..8].try_into().unwrap()),
            25,
            "tms_utime ticks"
        );
    }

    #[test]
    fn first_handled_signal_classifies_pending_signals() {
        let (k, _mem, _v, mut cx) = setup();
        let handler = SigAction {
            handler: 0x4000,
            flags: 0,
            restorer: 0,
            mask: 0,
        };
        let bit = |sig: u32| 1u64 << (sig - 1);

        // Nothing pending → nothing to interrupt.
        assert_eq!(k.first_handled_signal(&cx), None);

        // A real handler for a pending signal interrupts.
        cx.cur.pending = bit(2); // SIGINT
        cx.cur.handlers[2] = handler;
        assert_eq!(k.first_handled_signal(&cx), Some(2));

        // …but not while that signal is blocked.
        cx.cur.blocked = bit(2);
        assert_eq!(k.first_handled_signal(&cx), None);
        cx.cur.blocked = 0;

        // SIG_IGN does not interrupt.
        cx.cur.handlers[2] = SigAction {
            handler: 1,
            ..handler
        };
        assert_eq!(k.first_handled_signal(&cx), None);

        // A default-terminate signal (SIG_DFL) returns None: the caller lets the
        // Zombie path end the task rather than interrupting the syscall.
        cx.cur.handlers[2] = SigAction::default();
        cx.cur.pending = bit(15); // SIGTERM, no handler
        assert_eq!(k.first_handled_signal(&cx), None);

        // A default-ignored signal (SIGCHLD) does not interrupt.
        cx.cur.pending = bit(17);
        assert_eq!(k.first_handled_signal(&cx), None);

        // Alongside a default-ignored signal, the lowest *handled* one wins.
        cx.cur.pending = bit(17) | bit(10);
        cx.cur.handlers[10] = handler;
        assert_eq!(k.first_handled_signal(&cx), Some(10));
    }

    #[test]
    fn futex_wait_single_thread_does_not_deadlock() {
        let (k, mut mem, mut v, mut cx) = setup();
        let uaddr = 0x1_0000;
        mem.write_init(uaddr, &42u32.to_le_bytes()).unwrap();
        // Value matches and no other task could wake us: report a spurious wake
        // rather than parking (which would be a false deadlock).
        let r = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Futex,
            [uaddr, 0, 42, 0, 0, 0],
        );
        assert_eq!(r, 0);
        assert!(!cx.block, "lone waiter is not parked");
        // A mismatched value is EAGAIN.
        let r = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Futex,
            [uaddr, 0, 99, 0, 0, 0],
        );
        assert_eq!(r, -i64::from(Errno::EAGAIN.0));
    }

    #[test]
    fn futex_wait_parks_when_a_sibling_can_wake() {
        let (k, mut mem, mut v, mut cx) = setup();
        let uaddr = 0x1_0000;
        mem.write_init(uaddr, &42u32.to_le_bytes()).unwrap();
        // A runnable sibling exists, so a matching wait parks the caller.
        k.shared
            .lock()
            .unwrap()
            .procs
            .push(Some(make_proc(2, 1, 0, true)));
        let r = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Futex,
            [uaddr, 0, 42, 0, 0, 0],
        );
        assert_eq!(r, 0);
        assert!(cx.block, "caller parks awaiting a wake");
        assert_eq!(cx.cur.futex_wait, Some(futex::FutexKey::Private(0, uaddr)));
    }

    #[test]
    fn mmap_file_backed_copies_file_contents() {
        const MAP_FIXED: u64 = 0x10 | 0x02; // with MAP_PRIVATE: a map type is required
        const PROT_READ: u64 = 0x1;
        let (k, mut mem, mut v, mut cx) = setup();
        let path = 0x1_0000;
        let content = 0x1_1000;
        mem.write_init(path, b"/lib\0").unwrap();
        mem.write_init(content, &[0x11, 0x22, 0x33, 0x44]).unwrap();

        // Create /lib and write four bytes to it.
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_CWD, path, 0o102, 0o644, 0, 0],
        );
        assert_eq!(fd, 3);
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [fd as u64, content, 4, 0, 0, 0]
            ),
            4
        );

        // Map it read-only at a fixed address; the file bytes appear there.
        let addr = 0x1_5000u64;
        let ret = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Mmap,
            [addr, 4, PROT_READ, MAP_FIXED, fd as u64, 0],
        );
        assert_eq!(ret, addr as i64);
        assert_eq!(mem.read_u32(addr).unwrap(), 0x4433_2211);
    }

    #[test]
    fn mmap_file_backed_zero_fills_past_eof() {
        const MAP_FIXED: u64 = 0x10 | 0x02; // with MAP_PRIVATE: a map type is required
        let (k, mut mem, mut v, mut cx) = setup();
        let path = 0x1_0000;
        let content = 0x1_1000;
        mem.write_init(path, b"/x\0").unwrap();
        mem.write_init(content, &[0xAB, 0xCD]).unwrap();
        // Pre-dirty the target page so we can prove the tail is zeroed.
        mem.write(0x1_3000, &[0xFF; 8]).unwrap();
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_CWD, path, 0o102, 0o644, 0, 0],
        );
        call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Write,
            [fd as u64, content, 2, 0, 0, 0],
        );
        let addr = 0x1_3000u64;
        call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Mmap,
            [addr, 8, 0x1, MAP_FIXED, fd as u64, 0],
        );
        // First two bytes from the file, the rest zero-filled (not the old 0xFF).
        assert_eq!(mem.read_u32(addr).unwrap(), 0x0000_CDAB);
        assert_eq!(mem.read_u32(addr + 4).unwrap(), 0);
    }

    #[test]
    fn mmap_bad_and_nonfile_fd_rejected() {
        const MAP_FIXED: u64 = 0x10 | 0x02; // with MAP_PRIVATE: a map type is required
        let (k, mut mem, mut v, mut cx) = setup();
        // No such fd -> EBADF.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Mmap,
                [0x1_5000, 4, 1, MAP_FIXED, 99, 0]
            ),
            -i64::from(Errno::EBADF.0)
        );
        // fd 1 is stdout, not a file -> EACCES.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Mmap,
                [0x1_5000, 4, 1, MAP_FIXED, 1, 0]
            ),
            -i64::from(Errno::EACCES.0)
        );
    }

    #[cfg(unix)]
    #[test]
    fn reads_host_file_through_passthrough_hole() {
        use crate::fs::Passthrough;
        let dir = std::env::temp_dir().join(format!("nixvm-hole-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("probe"), b"Z").unwrap();

        let mut mounts = MountTable::new();
        mounts.mount("/", Box::new(TmpFs::new()));
        mounts.mount("/work", Box::new(Passthrough::new(dir.clone())));
        let k = Kernel::new(Arch::Aarch64, mounts);
        let mut cx = ServiceCtx::default();
        cx.cur.pid = 1;
        let mut mem = GuestMemory::new(0x1_0000, 16 * PAGE);
        mem.map(0x1_0000, 4 * PAGE, Prot::rw()).unwrap();
        let mut v = DummyVcpu;

        let path = 0x1_0000;
        mem.write_init(path, b"/work/probe\0").unwrap();
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_CWD, path, 0, 0, 0, 0],
        );
        assert!(fd >= 3, "open through hole failed: {fd}");
        let buf = 0x1_1000;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Read,
                [fd as u64, buf, 1, 0, 0, 0]
            ),
            1
        );
        assert_eq!(mem.read_vec(buf, 1).unwrap(), b"Z");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn time_syscalls() {
        let (k, mut mem, mut v, mut cx) = setup();
        let tv = 0x1_0000;

        // gettimeofday writes a nonzero tv_sec.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Gettimeofday,
                [tv, 0, 0, 0, 0, 0]
            ),
            0
        );
        assert!(mem.read_u64(tv).unwrap() > 0);

        // clock_getres writes {tv_sec: 0, tv_nsec: 1} (arg[1] is res).
        let res = 0x1_1000;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::ClockGetres,
                [0, res, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_u64(res).unwrap(), 0);
        assert_eq!(mem.read_u64(res + 8).unwrap(), 1);

        // nanosleep now genuinely suspends the caller: the first entry seeds the
        // deadline and parks (`cx.block`); a re-trap after it elapses completes,
        // writing rem = {0, 0}.
        let req = 0x1_2000;
        let rem = 0x1_2100;
        mem.write_init(req, &0u64.to_le_bytes()).unwrap();
        mem.write_init(req + 8, &20_000_000u64.to_le_bytes())
            .unwrap(); // 20 ms
        mem.write_init(rem, &7u64.to_le_bytes()).unwrap();
        mem.write_init(rem + 8, &7u64.to_le_bytes()).unwrap();
        cx.block = false;
        cx.cur.wake_deadline = None;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Nanosleep,
                [req, rem, 0, 0, 0, 0]
            ),
            0
        );
        assert!(cx.block, "nanosleep parks the caller until its deadline");
        assert!(cx.cur.wake_deadline.is_some());
        // Let the (20 ms) deadline pass, then re-trap: it completes. (A
        // sub-microsecond sleep could expire before the first check under a
        // loaded parallel test run, so the "parks" assertion raced.)
        std::thread::sleep(std::time::Duration::from_millis(30));
        cx.block = false;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Nanosleep,
                [req, rem, 0, 0, 0, 0]
            ),
            0
        );
        assert!(!cx.block, "nanosleep completes once the deadline passes");
        assert_eq!(cx.cur.wake_deadline, None);
        assert_eq!(mem.read_u64(rem).unwrap(), 0);
        assert_eq!(mem.read_u64(rem + 8).unwrap(), 0);

        // nanosleep with tv_nsec >= 1e9 returns -EINVAL.
        cx.block = false;
        cx.cur.wake_deadline = None;
        mem.write_init(req + 8, &1_000_000_000u64.to_le_bytes())
            .unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Nanosleep,
                [req, 0, 0, 0, 0, 0]
            ),
            err(Errno::EINVAL)
        );
    }

    #[test]
    fn lseek_on_directory_rewinds_getdents() {
        let (k, mut mem, mut v, mut cx) = setup();
        // Create a file so the root dir has content beyond "."/"..".
        let path = 0x1_0000;
        mem.write_init(path, b"/f\0").unwrap();
        call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_CWD, path, 0o102, 0o644, 0, 0],
        );
        let root = 0x1_1000;
        mem.write_init(root, b"/\0").unwrap();
        let dirfd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_CWD, root, 0, 0, 0, 0],
        ) as u64;
        let buf = 0x1_2000;
        // First scan consumes the directory.
        let n1 = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Getdents64,
            [dirfd, buf, PAGE, 0, 0, 0],
        );
        assert!(n1 > 0);
        // At EOF a second getdents returns 0.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Getdents64,
                [dirfd, buf, PAGE, 0, 0, 0]
            ),
            0
        );
        // lseek(0, SEEK_SET) rewinds (rewinddir); the next getdents re-reads all.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Lseek,
                [dirfd, 0, 0, 0, 0, 0]
            ),
            0
        );
        let n3 = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Getdents64,
            [dirfd, buf, PAGE, 0, 0, 0],
        );
        assert_eq!(n3, n1, "rewound scan re-reads the whole directory");
    }

    #[test]
    fn getdents_and_getcwd() {
        let (k, mut mem, mut v, mut cx) = setup();
        let path = 0x1_0000;
        mem.write_init(path, b"/a\0").unwrap();
        call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_CWD, path, 0o102, 0o644, 0, 0],
        );

        let root = 0x1_1000;
        mem.write_init(root, b"/\0").unwrap();
        let dirfd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_CWD, root, 0, 0, 0, 0],
        );
        let buf = 0x1_2000;
        let n = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Getdents64,
            [dirfd as u64, buf, PAGE, 0, 0, 0],
        );
        assert!(n > 0);

        let cbuf = 0x1_3000;
        let len = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Getcwd,
            [cbuf, 64, 0, 0, 0, 0],
        );
        assert_eq!(len, 2);
        assert_eq!(mem.read_vec(cbuf, 1).unwrap(), b"/");
    }

    #[test]
    fn fault_signal_delivery_and_rt_sigreturn_round_trip() {
        use crate::vcpu::Backend;
        // A real interpreter vcpu with distinctive register state.
        let backend = crate::vcpu::interp_x86::X86Backend::new(Arch::X86_64).unwrap();
        let mut vcpu = backend.new_vcpu(0x1_1111, 0x1_3000).unwrap();
        vcpu.set_reg(3, 0xdead); // rbx (callee-saved) — must survive the handler
        vcpu.set_reg(0, 0x1234); // rax
        let xmm: Vec<u8> = (0..=255u8).collect(); // XMM0..15, distinctive
        vcpu.set_simd_state(&xmm);
        let (orig_pc, orig_sp) = (vcpu.pc(), vcpu.sp());

        let (mut k, mut mem, _v, mut cx) = setup();
        k.arch = Arch::X86_64; // this test drives an x86-64 vcpu + frame layout
        mem.map(0x1_0000, 4 * PAGE, Prot::rw()).unwrap();
        cx.cur.mm = 0;
        cx.cur.handlers[11] = SigAction {
            handler: 0x2_0000,
            flags: 0,
            restorer: 0x2_1000,
            mask: 0,
        };

        // Deliver SIGSEGV (fault addr 0xcafe) → the vcpu enters the handler.
        assert!(k.deliver_fault_signal(
            &mut cx,
            signal::Fault::segv(k.arch, 0xcafe, false, false, false),
            vcpu.as_mut(),
            &mut mem
        ));
        assert_eq!(vcpu.pc(), 0x2_0000, "pc → handler");
        assert_eq!(vcpu.reg(7), 11, "rdi = signum");
        let frame = vcpu.sp();
        assert_eq!(vcpu.reg(2), frame + 8, "rdx = &ucontext");
        assert_eq!(
            vcpu.reg(6),
            frame + 8 + super::signal::signal_ucontext_size(),
            "rsi = &siginfo"
        );
        assert_eq!(
            mem.read_u64(frame).unwrap(),
            0x2_1000,
            "pretcode = restorer"
        );
        assert_eq!(
            cx.cur.blocked & (1 << 10),
            1 << 10,
            "SIGSEGV blocked in handler"
        );

        // uc_mcontext.fpstate → a 64-byte-aligned fxsave image above the
        // frame with the XMM file at +160.
        let fpstate = mem.read_u64(frame + 8 + 40 + 23 * 8).unwrap();
        assert!(fpstate > frame && fpstate % 64 == 0, "fpstate {fpstate:#x}");
        assert_eq!(mem.read_vec(fpstate + 160, 256).unwrap(), xmm);
        assert_eq!(mem.read_u32(fpstate + 24).unwrap(), 0x1f80, "mxcsr");

        // The handler clobbers rbx and the XMM file; rt_sigreturn restores them.
        vcpu.set_reg(3, 0);
        vcpu.set_simd_state(&[0u8; 256]);
        vcpu.set_sp(frame + 8); // as if the restorer's `ret` popped pretcode
        k.sys_rt_sigreturn(&mut cx, vcpu.as_mut(), &mem);
        assert_eq!(vcpu.pc(), orig_pc, "pc restored");
        assert_eq!(vcpu.sp(), orig_sp, "rsp restored");
        assert_eq!(vcpu.reg(3), 0xdead, "rbx restored");
        assert_eq!(vcpu.reg(0), 0x1234, "rax restored");
        assert_eq!(vcpu.simd_state(), xmm, "XMM restored");
        assert_eq!(cx.cur.blocked, 0, "signal mask restored");
    }

    #[test]
    fn queued_signal_delivers_si_code_and_si_value() {
        use crate::vcpu::Backend;
        let backend = crate::vcpu::interp_x86::X86Backend::new(Arch::X86_64).unwrap();
        let mut vcpu = backend.new_vcpu(0x1_1111, 0x1_3000).unwrap();
        let (mut k, mut mem, _v, mut cx) = setup();
        k.arch = Arch::X86_64; // this test drives an x86-64 vcpu + frame layout
        mem.map(0x1_0000, 4 * PAGE, Prot::rw()).unwrap();
        cx.cur.mm = 0;
        cx.cur.handlers[10] = SigAction {
            handler: 0x2_0000,
            flags: 0,
            restorer: 0x2_1000,
            mask: 0,
        };

        // A guest siginfo with si_code = SI_QUEUE (-1) and si_value = 0xABCD.
        let uinfo = 0x1_2000;
        mem.write_init(uinfo + 8, &(-1i32).to_le_bytes()).unwrap();
        mem.write_init(uinfo + 24, &0xABCDu64.to_le_bytes())
            .unwrap();

        // sigqueue(self, SIGUSR1, {0xABCD}) records both pending + siginfo.
        let target = i64::from(cx.cur.pid);
        assert_eq!(
            k.sys_rt_sigqueueinfo(
                &mut k.shared.lock().unwrap(),
                &mut cx,
                target,
                10,
                uinfo,
                &mem
            ),
            0
        );
        assert_ne!(cx.cur.pending & (1 << 9), 0, "SIGUSR1 pending");
        assert!(cx.cur.queued_siginfo[10].is_some(), "siginfo recorded");

        // Deliver: the frame's siginfo must carry si_code and si_value through.
        assert!(k.deliver_async_signal(&mut cx, 10, vcpu.as_mut(), &mut mem));
        let si = vcpu.sp() + 8 + super::signal::signal_ucontext_size();
        assert_eq!(mem.read_u32(si).unwrap(), 10, "si_signo");
        assert_eq!(
            mem.read_u32(si + 8).unwrap() as i32,
            -1,
            "si_code = SI_QUEUE"
        );
        assert_eq!(mem.read_u64(si + 24).unwrap(), 0xABCD, "si_value carried");
        // The info is consumed on delivery (not re-delivered on the next signal).
        assert!(cx.cur.queued_siginfo[10].is_none(), "siginfo consumed");
    }

    #[test]
    #[allow(clippy::cast_precision_loss)] // small integers; a ratio check
    fn nice_weight_follows_the_cfs_curve() {
        // nice 0 is the reference weight; the curve is monotone (lower nice =
        // more weight = more CPU), and each step is ~1.25×.
        assert_eq!(nice_weight(0), 1024);
        assert!(
            nice_weight(-20) > nice_weight(0),
            "negative nice weighs more"
        );
        assert!(
            nice_weight(19) < nice_weight(0),
            "positive nice weighs less"
        );
        for n in -19..=19 {
            assert!(nice_weight(n - 1) > nice_weight(n), "monotone at nice {n}");
        }
        // ~1.25 per level (allow rounding slack on the integer table).
        let ratio = nice_weight(0) as f64 / nice_weight(1) as f64;
        assert!(
            (ratio - 1.25).abs() < 0.05,
            "≈1.25× per nice level, got {ratio}"
        );
        // Out-of-range nice clamps to the table ends rather than panicking.
        assert_eq!(nice_weight(-100), nice_weight(-20));
        assert_eq!(nice_weight(100), nice_weight(19));
    }

    #[test]
    #[allow(clippy::cast_precision_loss)] // small integers; a ratio check
    fn charge_vruntime_makes_nice_proportional() {
        // Equal CPU consumed, but a higher-nice task accrues more virtual runtime
        // (so the least-vruntime scheduler picks it less) — proportional to the
        // inverse weight ratio, which is the whole point of the nice curve.
        let cpu = 10_000_000u128; // 10 ms
        let mut fast = ProcInfo {
            nice: 0,
            ..ProcInfo::default()
        };
        let mut slow = ProcInfo {
            nice: 5,
            ..ProcInfo::default()
        };
        charge_vruntime(&mut fast, cpu);
        charge_vruntime(&mut slow, cpu);
        assert!(
            slow.vruntime > fast.vruntime,
            "niced task's vruntime climbs faster"
        );
        let got = slow.vruntime as f64 / fast.vruntime as f64;
        let want = nice_weight(0) as f64 / nice_weight(5) as f64; // ≈ 1.25^5 ≈ 3.05
        assert!(
            (got - want).abs() / want < 0.02,
            "vruntime ratio ≈ weight ratio: {got} vs {want}"
        );
    }

    #[test]
    fn serial_pick_is_least_vruntime_and_clamps_woken_tasks() {
        fn task(pid: i32, vruntime: u128) -> Process {
            Process {
                vcpu: Some(Box::new(DummyVcpu)),
                info: ProcInfo {
                    pid,
                    vruntime,
                    run: RunState::Running,
                    ..ProcInfo::default()
                },
            }
        }
        let (k, _mem, _v, _cx) = setup();
        let mut sh = k.shared.lock().unwrap();
        sh.procs = vec![
            Some(task(10, 300)),
            Some(task(11, 100)),
            Some(task(12, 200)),
        ];

        // The least-vruntime task is picked (index 1, pid 11) — not pid order.
        assert_eq!(sh.pick_serial_runnable(), Some(1));
        // Picking advances the floor to that task's vruntime.
        assert_eq!(sh.min_vruntime, 100);

        // A task that blocked long ago carries a stale, tiny vruntime; on wake it
        // must not monopolize the CPU. Give index 0 a vruntime far below the
        // floor and confirm admission clamps it up to the floor rather than
        // letting it run unbounded ahead of the others.
        sh.procs[0].as_mut().unwrap().info.vruntime = 5;
        sh.min_vruntime = 250;
        assert_eq!(
            sh.pick_serial_runnable(),
            Some(0),
            "the woken task is picked (now lowest)"
        );
        assert_eq!(
            sh.procs[0].as_ref().unwrap().info.vruntime,
            250,
            "clamped up to the floor"
        );
        assert_eq!(sh.min_vruntime, 250, "floor stays monotonic");
    }

    #[test]
    fn aarch64_signal_delivery_and_rt_sigreturn_round_trip() {
        use crate::vcpu::Backend;
        // A real aarch64 interpreter vcpu with distinctive register/flag state.
        let backend = crate::vcpu::interp::InterpBackend::new(Arch::Aarch64).unwrap();
        let mut vcpu = backend.new_vcpu(0x1_1111, 0x1_3000).unwrap();
        vcpu.set_reg(19, 0xdead); // x19 (callee-saved) — must survive the handler
        vcpu.set_reg(0, 0x1234); // x0
        vcpu.set_rflags(1 << 30); // PSTATE.Z set — must round-trip
        // FP/SIMD state: FPCR rounding mode, a cumulative FPSR flag, and
        // distinctive V registers — all must survive the handler.
        let mut simd = vec![0u8; 520];
        simd[0..4].copy_from_slice(&0x10u32.to_le_bytes()); // FPSR.IXC
        simd[4..8].copy_from_slice(&0x0040_0000u32.to_le_bytes()); // FPCR.RMode = +inf
        for (i, b) in simd[8..].iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        vcpu.set_simd_state(&simd);
        let (orig_pc, orig_sp, orig_pstate) = (vcpu.pc(), vcpu.sp(), vcpu.rflags());

        let (k, mut mem, _v, mut cx) = setup(); // setup() is already Arch::Aarch64
        mem.map(0x1_0000, 4 * PAGE, Prot::rw()).unwrap();
        cx.cur.mm = 0;
        cx.cur.handlers[11] = SigAction {
            handler: 0x2_0000,
            flags: 0x0400_0000, // SA_RESTORER
            restorer: 0x2_1000,
            mask: 0,
        };

        // Deliver SIGSEGV (fault addr 0xcafe) → the vcpu enters the aarch64 handler.
        assert!(k.deliver_fault_signal(
            &mut cx,
            signal::Fault::segv(k.arch, 0xcafe, false, false, false),
            vcpu.as_mut(),
            &mut mem
        ));
        assert_eq!(vcpu.pc(), 0x2_0000, "pc → handler");
        assert_eq!(vcpu.reg(0), 11, "x0 = signum");
        assert_eq!(vcpu.reg(30), 0x2_1000, "x30 = sa_restorer");
        let frame = vcpu.sp();
        assert_eq!(vcpu.reg(1), frame, "x1 = &siginfo (frame base)");
        assert_eq!(vcpu.reg(2), frame + 128, "x2 = &ucontext");
        // siginfo at the frame base carries si_signo and the fault address.
        assert_eq!(mem.read_u32(frame).unwrap(), 11, "si_signo");
        assert_eq!(mem.read_u32(frame + 8).unwrap(), 1, "si_code SEGV_MAPERR");
        assert_eq!(mem.read_u64(frame + 16).unwrap(), 0xcafe, "si_addr");
        assert_eq!(
            cx.cur.blocked & (1 << 10),
            1 << 10,
            "SIGSEGV blocked in handler"
        );
        // uc_mcontext at +176 (the UAPI offset musl/glibc/Go compile in).
        let mctx = frame + 128 + 176;
        assert_eq!(mem.read_u64(mctx).unwrap(), 0xcafe, "fault_address");
        assert_eq!(mem.read_u64(mctx + 8 + 19 * 8).unwrap(), 0xdead, "regs[19]");
        assert_eq!(mem.read_u64(mctx + 264).unwrap(), orig_pc, "pc");
        // __reserved: fpsimd_context, esr_context (data abort, level-3
        // translation fault), then the null terminator.
        let rec = mctx + 288;
        assert_eq!(mem.read_u32(rec).unwrap(), 0x4650_8001, "FPSIMD_MAGIC");
        assert_eq!(mem.read_u32(rec + 4).unwrap(), 528);
        assert_eq!(mem.read_vec(rec + 8, 520).unwrap(), simd, "fpsr/fpcr/vregs");
        assert_eq!(mem.read_u32(rec + 528).unwrap(), 0x4553_5201, "ESR_MAGIC");
        assert_eq!(
            mem.read_u64(rec + 536).unwrap() >> 26,
            0x24,
            "EC = DABT_LOW"
        );
        assert_eq!(mem.read_u64(rec + 544).unwrap(), 0, "terminator");

        // The handler clobbers x19, sp, the flags and the FP state;
        // rt_sigreturn restores them.
        vcpu.set_reg(19, 0);
        vcpu.set_rflags(0);
        vcpu.set_simd_state(&[0u8; 520]);
        // sp still points at the frame base (the restorer trampoline doesn't move it).
        k.sys_rt_sigreturn(&mut cx, vcpu.as_mut(), &mem);
        assert_eq!(vcpu.pc(), orig_pc, "pc restored");
        assert_eq!(vcpu.sp(), orig_sp, "sp restored");
        assert_eq!(vcpu.reg(19), 0xdead, "x19 restored");
        assert_eq!(vcpu.reg(0), 0x1234, "x0 restored");
        assert_eq!(vcpu.rflags(), orig_pstate, "pstate (NZCV) restored");
        assert_eq!(cx.cur.blocked, 0, "signal mask restored");
    }

    #[test]
    fn fault_with_no_handler_is_not_delivered() {
        use crate::vcpu::Backend;
        let backend = crate::vcpu::interp_x86::X86Backend::new(Arch::X86_64).unwrap();
        let mut vcpu = backend.new_vcpu(0x1_1111, 0x1_3000).unwrap();
        let (k, mut mem, _v, mut cx) = setup();
        // SIG_DFL for SIGSEGV: not deliverable (stays a fatal fault).
        assert!(!k.deliver_fault_signal(
            &mut cx,
            signal::Fault::segv(k.arch, 0, false, false, false),
            vcpu.as_mut(),
            &mut mem
        ));
    }

    #[test]
    fn rt_sigaction_stores_and_returns_old_handler() {
        let (k, mut mem, mut v, mut cx) = setup();
        let act = 0x1_0000;
        let oldact = 0x1_0100;

        // Install handler 0xdead for SIGINT (2).
        mem.write_init(act, &0xdeadu64.to_le_bytes()).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::RtSigaction,
                [2, act, 0, 8, 0, 0]
            ),
            0
        );
        assert_eq!(cx.cur.handlers[2].handler, 0xdead);

        // Install 0xbeef and read back the previous (0xdead) via oldact.
        mem.write_init(act, &0xbeefu64.to_le_bytes()).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::RtSigaction,
                [2, act, oldact, 8, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_u64(oldact).unwrap(), 0xdead);
        assert_eq!(cx.cur.handlers[2].handler, 0xbeef);
    }

    #[test]
    fn rt_sigaction_rejects_sigkill() {
        let (k, mut mem, mut v, mut cx) = setup();
        let act = 0x1_0000;
        mem.write_init(act, &1u64.to_le_bytes()).unwrap();
        // SIGKILL (9) and SIGSTOP (19) dispositions cannot change.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::RtSigaction,
                [9, act, 0, 8, 0, 0]
            ),
            -22
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::RtSigaction,
                [19, act, 0, 8, 0, 0]
            ),
            -22
        );
    }

    #[test]
    fn rt_sigprocmask_setmask_and_readback() {
        let (k, mut mem, mut v, mut cx) = setup();
        let set = 0x1_0000;
        let oldset = 0x1_0100;
        mem.write_init(set, &0b1010u64.to_le_bytes()).unwrap();

        // SIG_SETMASK (2) replaces the mask.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::RtSigprocmask,
                [2, set, 0, 8, 0, 0]
            ),
            0
        );
        assert_eq!(cx.cur.blocked, 0b1010);

        // Read it back through oldset (set == 0 leaves the mask unchanged).
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::RtSigprocmask,
                [0, 0, oldset, 8, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_u64(oldset).unwrap(), 0b1010);
    }

    #[test]
    fn kill_self_then_deliver_terminates() {
        let (k, mut mem, mut v, mut cx) = setup();
        // kill(pid 1 == self, SIGTERM=15) sets the pending bit.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Kill,
                [1, 15, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(cx.cur.pending, 1 << 14);

        // Default disposition of SIGTERM is TERMINATE -> a signal death (which
        // wait4 encodes as WIFSIGNALED with termsig 15, not a WIFEXITED code).
        k.deliver_pending_signals(&mut cx, &mut v, &mut mem);
        assert!(matches!(
            cx.cur.run,
            RunState::Zombie(ExitCause::Signaled(15))
        ));
    }

    #[test]
    fn kill_nonexistent_pid_is_esrch() {
        let (k, mut mem, mut v, mut cx) = setup();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Kill,
                [999, 15, 0, 0, 0, 0]
            ),
            -3
        );
    }

    #[test]
    fn alarm_arms_a_timer_that_posts_sigalrm() {
        let (k, mut mem, mut v, mut cx) = setup();
        // alarm(0) with nothing armed returns 0 and stays disarmed.
        assert_eq!(call(&k, &mut cx, &mut mem, &mut v, Sysno::Alarm, [0; 6]), 0);
        assert_eq!(cx.cur.alarm_deadline, None);
        // alarm(5) arms a one-shot ~5s out; a re-arm returns the ~5s remaining.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Alarm,
                [5, 0, 0, 0, 0, 0]
            ),
            0
        );
        assert!(cx.cur.alarm_deadline.is_some());
        let r = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Alarm,
            [10, 0, 0, 0, 0, 0],
        );
        assert!((4..=5).contains(&r), "prior alarm had ~5s left, got {r}");

        // Firing at/after the deadline posts SIGALRM and disarms the one-shot.
        let dl = cx.cur.alarm_deadline.unwrap();
        assert!(!fire_alarm_if_due(&mut cx.cur, dl - 1), "not yet due");
        assert!(fire_alarm_if_due(&mut cx.cur, dl), "due → fires");
        assert_eq!(cx.cur.pending & (1 << (SIGALRM - 1)), 1 << (SIGALRM - 1));
        assert_eq!(cx.cur.alarm_deadline, None, "one-shot disarms");

        // A periodic timer re-arms to the next interval when it fires.
        cx.cur.pending = 0;
        cx.cur.alarm_interval_ns = 1_000_000_000; // 1s
        cx.cur.alarm_deadline = Some(1000);
        assert!(fire_alarm_if_due(&mut cx.cur, 1000));
        assert_eq!(
            cx.cur.alarm_deadline,
            Some(1000 + 1_000_000_000),
            "periodic re-arms"
        );
    }

    #[test]
    fn signalfd_reads_pending_masked_signals() {
        let (k, mut mem, mut v, mut cx) = setup();
        let maskptr = 0x1_0000;
        mem.write_init(maskptr, &(1u64 << 9).to_le_bytes()).unwrap(); // SIGUSR1(10)
        // signalfd4(-1, {SIGUSR1}, 8, 0) creates a new signalfd.
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Signalfd4,
            [(-1i64) as u64, maskptr, 8, 0, 0, 0],
        );
        assert!(fd >= 3, "a fresh fd, got {fd}");
        // With SIGUSR1 pending, a read returns one 128-byte siginfo and dequeues it.
        cx.cur.pending = 1u64 << 9;
        let buf = 0x1_1000;
        let r = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Read,
            [fd as u64, buf, 128, 0, 0, 0],
        );
        assert_eq!(r, 128);
        assert_eq!(mem.read_u32(buf).unwrap(), 10, "ssi_signo = SIGUSR1");
        assert_eq!(cx.cur.pending, 0, "the signal is consumed by the read");
    }

    #[test]
    fn rt_sigtimedwait_dequeues_pending_or_times_out() {
        let (k, mut mem, mut v, mut cx) = setup();
        let (set, timeout) = (0x1_0000, 0x1_0100);
        mem.write_init(set, &(1u64 << 9).to_le_bytes()).unwrap(); // wait for SIGUSR1(10)
        mem.write_init(timeout, &[0u8; 16]).unwrap(); // {0,0}: non-blocking poll
        // Nothing pending → EAGAIN.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::RtSigtimedwait,
                [set, 0, timeout, 8, 0, 0]
            ),
            err(Errno::EAGAIN)
        );
        // SIGUSR1 pending → returns 10 and dequeues it.
        cx.cur.pending = 1u64 << 9;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::RtSigtimedwait,
                [set, 0, timeout, 8, 0, 0]
            ),
            10
        );
        assert_eq!(cx.cur.pending, 0, "the accepted signal is dequeued");
    }

    #[test]
    fn kill_targets_process_groups() {
        let (k, mut mem, mut v, mut cx) = setup(); // caller pid 1, pgid 0 → group 1
        {
            let mut sh = k.shared.lock().unwrap();
            for (pid, pgid) in [(2, 7), (3, 7), (4, 9)] {
                let mut p = make_proc(pid, pid, 0, false);
                p.info.pgid = pgid;
                sh.procs.push(Some(p));
            }
        }
        let bit = 1u64 << 9; // SIGUSR1 = 10
        let pending = |pid: i32| {
            k.shared
                .lock()
                .unwrap()
                .procs
                .iter()
                .flatten()
                .find(|p| p.info.pid == pid)
                .unwrap()
                .info
                .pending
        };
        // kill(-7, SIGUSR1): both group-7 members, not group 9.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Kill,
                [(-7i64) as u64, 10, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(pending(2) & bit, bit);
        assert_eq!(pending(3) & bit, bit);
        assert_eq!(pending(4) & bit, 0, "a different group is untouched");
        // kill(-99): no such group → ESRCH.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Kill,
                [(-99i64) as u64, 10, 0, 0, 0, 0]
            ),
            err(Errno::ESRCH)
        );
        // kill(0): the caller's own group (pgid 1) — the caller gets it.
        cx.cur.pending = 0;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Kill,
                [0, 10, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(cx.cur.pending & bit, bit);
    }

    /// Open a file, seed it, and return its fd — for the I/O syscall tests.
    fn open_seeded(
        k: &mut Kernel,
        cx: &mut ServiceCtx,
        mem: &mut GuestMemory,
        v: &mut DummyVcpu,
        content: &[u8],
    ) -> u64 {
        let path = 0x1_0000;
        mem.write_init(path, b"/f\0").unwrap();
        let fd = call(
            k,
            cx,
            mem,
            v,
            Sysno::Openat,
            [AT_CWD, path, 0o102, 0o644, 0, 0],
        ) as u64;
        let src = 0x1_3000;
        mem.write_init(src, content).unwrap();
        call(
            k,
            cx,
            mem,
            v,
            Sysno::Write,
            [fd, src, content.len() as u64, 0, 0, 0],
        );
        fd
    }

    #[test]
    fn pread_pwrite_do_not_move_the_offset() {
        let (mut k, mut mem, mut v, mut cx) = setup();
        let fd = open_seeded(&mut k, &mut cx, &mut mem, &mut v, b"0123456789");
        // Read the fd position back to 0 via lseek, then pread at offset 4.
        call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Lseek,
            [fd, 0, 0, 0, 0, 0],
        );
        let buf = 0x1_2000;
        let n = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Pread64,
            [fd, buf, 3, 4, 0, 0],
        );
        assert_eq!(n, 3);
        assert_eq!(mem.read_vec(buf, 3).unwrap(), b"456");
        // The fd position is still 0, so a plain read starts at the beginning.
        let n = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Read,
            [fd, buf, 2, 0, 0, 0],
        );
        assert_eq!(n, 2);
        assert_eq!(mem.read_vec(buf, 2).unwrap(), b"01");
        // pwrite at offset 4 overwrites in place, again without moving the pos.
        let src = 0x1_1000;
        mem.write_init(src, b"XY").unwrap();
        call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Pwrite64,
            [fd, src, 2, 4, 0, 0],
        );
        call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Lseek,
            [fd, 0, 0, 0, 0, 0],
        );
        call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Read,
            [fd, buf, 10, 0, 0, 0],
        );
        assert_eq!(mem.read_vec(buf, 10).unwrap(), b"0123XY6789");
    }

    #[test]
    fn ftruncate_and_truncate_resize() {
        let (mut k, mut mem, mut v, mut cx) = setup();
        let fd = open_seeded(&mut k, &mut cx, &mut mem, &mut v, b"abcdef");
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Ftruncate,
                [fd, 3, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(k.vfs.lock().unwrap().stat("/f").unwrap().size, 3);
        // truncate by path can also grow (zero-extend).
        let path = 0x1_0000;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Truncate,
                [path, 8, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(k.vfs.lock().unwrap().stat("/f").unwrap().size, 8);
    }

    #[test]
    fn statx_reports_size_and_mode() {
        let (mut k, mut mem, mut v, mut cx) = setup();
        open_seeded(&mut k, &mut cx, &mut mem, &mut v, b"hello world");
        let path = 0x1_0000;
        let buf = 0x1_2000;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Statx,
                [AT_CWD, path, 0, 0x7ff, buf, 0]
            ),
            0
        );
        // stx_size @40, stx_mode @28.
        assert_eq!(
            u64::from_le_bytes(mem.read_vec(buf + 40, 8).unwrap().try_into().unwrap()),
            11
        );
        let mode = u16::from_le_bytes(mem.read_vec(buf + 28, 2).unwrap().try_into().unwrap());
        assert_eq!(mode & 0o170000, 0o100000, "S_IFREG");
    }

    #[test]
    fn sendfile_copies_between_files() {
        let (mut k, mut mem, mut v, mut cx) = setup();
        let infd = open_seeded(&mut k, &mut cx, &mut mem, &mut v, b"payload!");
        call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Lseek,
            [infd, 0, 0, 0, 0, 0],
        );
        // A second file as the destination.
        let path2 = 0x1_1000;
        mem.write_init(path2, b"/g\0").unwrap();
        let outfd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_CWD, path2, 0o102, 0o644, 0, 0],
        ) as u64;
        let n = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Sendfile,
            [outfd, infd, 0, 8, 0, 0],
        );
        assert_eq!(n, 8);
        assert_eq!(k.vfs.lock().unwrap().stat("/g").unwrap().size, 8);
        let buf = 0x1_2000;
        call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Lseek,
            [outfd, 0, 0, 0, 0, 0],
        );
        call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Read,
            [outfd, buf, 8, 0, 0, 0],
        );
        assert_eq!(mem.read_vec(buf, 8).unwrap(), b"payload!");
    }

    #[test]
    fn session_and_pgid_tracking() {
        let (k, mut mem, mut v, mut cx) = setup();
        cx.cur.pid = 5;
        // getpgid(0) defaults to the pid.
        assert_eq!(
            call(&k, &mut cx, &mut mem, &mut v, Sysno::Getpgid, [0; 6]),
            5
        );
        // setpgid(0, 42) sets it.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Setpgid,
                [0, 42, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(&k, &mut cx, &mut mem, &mut v, Sysno::Getpgid, [0; 6]),
            42
        );
        assert_eq!(
            call(&k, &mut cx, &mut mem, &mut v, Sysno::Getpgrp, [0; 6]),
            42
        );
        // setsid starts a new session: sid = pgid = pid.
        assert_eq!(
            call(&k, &mut cx, &mut mem, &mut v, Sysno::Setsid, [0; 6]),
            5
        );
        assert_eq!(
            call(&k, &mut cx, &mut mem, &mut v, Sysno::Getsid, [0; 6]),
            5
        );
        assert_eq!(
            call(&k, &mut cx, &mut mem, &mut v, Sysno::Getpgid, [0; 6]),
            5
        );
    }

    #[test]
    fn memfd_create_is_a_readwrite_fd() {
        let (k, mut mem, mut v, mut cx) = setup();
        let name = 0x1_0000;
        mem.write_init(name, b"scratch\0").unwrap();
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::MemfdCreate,
            [name, 0, 0, 0, 0, 0],
        );
        assert!(fd >= 3, "a real fd");
        let fd = fd as u64;
        let src = 0x1_2000;
        mem.write_init(src, b"data").unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [fd, src, 4, 0, 0, 0]
            ),
            4
        );
        call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Lseek,
            [fd, 0, 0, 0, 0, 0],
        );
        let buf = 0x1_3000;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Read,
                [fd, buf, 4, 0, 0, 0]
            ),
            4
        );
        assert_eq!(mem.read_vec(buf, 4).unwrap(), b"data");
    }

    #[test]
    fn close_range_closes_fds() {
        let (mut k, mut mem, mut v, mut cx) = setup();
        let fd = open_seeded(&mut k, &mut cx, &mut mem, &mut v, b"x");
        assert!(fd >= 3);
        // Close everything from `fd` up; a subsequent op on it is EBADF.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::CloseRange,
                [fd, u64::from(u32::MAX), 0, 0, 0, 0]
            ),
            0
        );
        let buf = 0x1_2000;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Read,
                [fd, buf, 1, 0, 0, 0]
            ),
            -9
        ); // EBADF
    }

    #[test]
    fn shared_file_mmap_flushes_writes_back() {
        // The apk large-file extraction pattern: create, ftruncate to size,
        // mmap(MAP_SHARED, PROT_WRITE), store into the mapping, munmap — and
        // the bytes must land in the file (this was the "node reads as zeros"
        // bug: MAP_SHARED writes were never flushed).
        let (mut k, mut mem, mut v, mut cx) = setup();
        // A small mmap arena inside the 16-page test region.
        let fd = open_seeded(&mut k, &mut cx, &mut mem, &mut v, b"");
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Ftruncate,
                [fd, 6, 0, 0, 0, 0]
            ),
            0
        );
        // mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0).
        let base = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Mmap,
            [0, 4096, 0x3, 0x1, fd, 0],
        );
        assert!(base > 0, "mmap returned {base}");
        let base = base as u64;
        // Store "hello!" into the mapping (as a guest memcpy would).
        mem.write(base, b"hello!").unwrap();
        // munmap flushes it back to the file.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Munmap,
                [base, 4096, 0, 0, 0, 0]
            ),
            0
        );
        // Read the file: it now holds the mapped bytes, not zeros.
        call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Lseek,
            [fd, 0, 0, 0, 0, 0],
        );
        let buf = 0x1_2000;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Read,
                [fd, buf, 6, 0, 0, 0]
            ),
            6
        );
        assert_eq!(mem.read_vec(buf, 6).unwrap(), b"hello!");
    }

    #[test]
    fn threads_sharing_an_address_space_get_disjoint_mmaps() {
        // Every task in one address space (CLONE_VM — every pthread) allocates
        // from the same per-mm `Arena`, so two threads can never be handed
        // overlapping ranges. Before the arena was shared, each thread bumped
        // its own copy of the cursor from the same start and they collided —
        // fatal once a JIT dropped code onto memory a sibling thought was free.
        let (k, mut mem, mut v, mut cx) = setup();
        cx.cur.mm = 0;

        let a = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Mmap,
            [0, 4096, 0x3, 0x22, u64::MAX, 0],
        );
        let b = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Mmap,
            [0, 4096, 0x3, 0x22, u64::MAX, 0],
        );
        assert!(a > 0 && b > 0, "mmaps returned {a}, {b}");
        let (a, b) = (a as u64, b as u64);
        assert!(
            a.abs_diff(b) >= 4096,
            "sibling mmaps overlap: A={a:#x} B={b:#x}"
        );
    }

    #[test]
    fn munmap_returns_the_range_to_the_arena_for_reuse() {
        // The arena must be an allocator, not a bump pointer: a guest that
        // cycles mappings (a JS engine recycling JIT/heap blocks) would
        // otherwise walk the cursor to the floor and start failing with ENOMEM
        // while nearly the whole arena sat free.
        let (k, mut mem, mut v, mut cx) = setup();
        cx.cur.mm = 0;
        // A 3-page arena: exactly three single-page mmaps fit.

        let anon = [0u64, 4096, 0x3, 0x22, u64::MAX, 0];

        let a = call(&k, &mut cx, &mut mem, &mut v, Sysno::Mmap, anon);
        let b = call(&k, &mut cx, &mut mem, &mut v, Sysno::Mmap, anon);
        let c = call(&k, &mut cx, &mut mem, &mut v, Sysno::Mmap, anon);
        assert!(a > 0 && b > 0 && c > 0);
        // Arena is now full: a fourth fails.
        assert_eq!(call(&k, &mut cx, &mut mem, &mut v, Sysno::Mmap, anon), -12); // ENOMEM

        // Free the middle one and the next mmap must reuse exactly that page,
        // rather than reporting the arena exhausted.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Munmap,
                [b as u64, 4096, 0, 0, 0, 0]
            ),
            0
        );
        let reused = call(&k, &mut cx, &mut mem, &mut v, Sysno::Mmap, anon);
        assert_eq!(reused, b, "munmap'd page must be handed out again");

        // Freeing all three coalesces back into one contiguous run, so a
        // 3-page mmap fits again.
        for p in [a, b, c] {
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Munmap,
                [p as u64, 4096, 0, 0, 0, 0],
            );
        }
        let big = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Mmap,
            [0, 3 * 4096, 0x3, 0x22, u64::MAX, 0],
        );
        assert!(
            big > 0,
            "coalesced free space must satisfy a 3-page mmap, got {big}"
        );
    }

    #[test]
    fn mmap_fixed_noreplace_fails_eexist_over_mapped_but_places_when_free() {
        // MAP_FIXED_NOREPLACE (0x100000) must be atomic: EEXIST over an occupied
        // range (never relocate, never clobber), exact placement over a free one.
        let (k, mut mem, mut v, mut cx) = setup();
        cx.cur.mm = 0;
        const NOREPLACE: u64 = 0x02 | 0x20 | 0x10_0000; // PRIVATE|ANON|FIXED_NOREPLACE
        // 0x1_0000..0x1_4000 is mapped by setup(): a NOREPLACE there is EEXIST.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Mmap,
                [0x1_0000, 4096, 0x3, NOREPLACE, u64::MAX, 0]
            ),
            err(Errno::EEXIST),
        );
        // A partial overlap (page 0x1_3000 is mapped) is still EEXIST, atomically.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Mmap,
                [0x1_3000, 2 * 4096, 0x3, NOREPLACE, u64::MAX, 0]
            ),
            err(Errno::EEXIST),
        );
        // A free, in-bounds page is placed exactly at the requested address.
        let want = 0x1_9000;
        let got = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Mmap,
            [want, 4096, 0x3, NOREPLACE, u64::MAX, 0],
        );
        assert_eq!(got, want as i64, "free NOREPLACE lands exactly at addr");
        mem.write_u64(want, 0xfeed).unwrap();
        assert_eq!(mem.read_u64(want).unwrap(), 0xfeed);
    }

    #[test]
    fn prlimit_nofile_is_tracked_and_hard_capped() {
        const NOFILE: u64 = 7;
        let (k, mut mem, mut v, mut cx) = setup();
        let buf = 0x1_2000;
        // getrlimit reports the default (1024, 4096).
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Getrlimit,
                [NOFILE, buf, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_u64(buf).unwrap(), 1024);
        assert_eq!(mem.read_u64(buf + 8).unwrap(), 4096);
        // Try to raise both soft and hard to a million (node/V8's binary
        // search). The hard limit is capped, and the soft is clamped to it.
        let newl = 0x1_2100;
        mem.write(newl, &1_048_576u64.to_le_bytes()).unwrap();
        mem.write(newl + 8, &1_048_576u64.to_le_bytes()).unwrap();
        // prlimit64(pid=0, NOFILE, new_limit, old_limit=0)
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Prlimit64,
                [0, NOFILE, newl, 0, 0, 0]
            ),
            0
        );
        // getrlimit now reports the capped values, not a million.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Getrlimit,
                [NOFILE, buf, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            mem.read_u64(buf).unwrap(),
            4096,
            "soft clamped to the hard cap"
        );
        assert_eq!(mem.read_u64(buf + 8).unwrap(), 4096, "hard capped");
    }

    #[test]
    fn fcntl_on_a_closed_fd_is_ebadf() {
        let (k, mut mem, mut v, mut cx) = setup();
        // F_SETFD (2) on an unopened fd must fail — else a "cloexec every fd
        // until EBADF" loop never terminates.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Fcntl,
                [99, 2, 1, 0, 0, 0]
            ),
            -9
        );
    }

    #[test]
    fn ioctl_fd_flag_requests_and_tty_fallback() {
        let (k, mut mem, mut v, mut cx) = setup();
        // A closed fd is EBADF (-9), so a "FIOCLEX every fd until EBADF" loop
        // terminates — the blanket ENOTTY stub used to spin such loops.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Ioctl,
                [99, 0x5451, 0, 0, 0, 0]
            ),
            -9
        );
        // FIOCLEX (0x5451) on an open fd (stdin) succeeds as an accepted no-op.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Ioctl,
                [0, 0x5451, 0, 0, 0, 0]
            ),
            0
        );
        // A terminal request (TIOCGWINSZ 0x5413) on a non-tty fd is ENOTTY (-25).
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Ioctl,
                [0, 0x5413, 0x1_2000, 0, 0, 0]
            ),
            -25
        );
    }

    #[test]
    fn credential_setters_succeed_as_root() {
        let (k, mut mem, mut v, mut cx) = setup();
        for s in [
            Sysno::Setuid,
            Sysno::Setgid,
            Sysno::Setresuid,
            Sysno::Setgroups,
        ] {
            assert_eq!(call(&k, &mut cx, &mut mem, &mut v, s, [0; 6]), 0, "{s:?}");
        }
        // getresuid writes (0,0,0).
        let (a, b, c) = (0x1_2000, 0x1_2010, 0x1_2020);
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Getresuid,
                [a, b, c, 0, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_u32(a).unwrap(), 0);
    }
}
