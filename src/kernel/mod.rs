//! The nixvm "kernel": an arch-agnostic engine that services guest syscalls
//! and schedules multiple guest processes.
//!
//! State is split between **global** kernel state (mount table, pipes, stdio,
//! process table) and the **running task's** state ([`ServiceCtx`]: its
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

use std::collections::{BTreeMap, VecDeque};
use std::io::{self, Read, Write};
use std::sync::{Arc, Condvar, Mutex, mpsc};

use crate::abi::Arch;
use crate::abi::arch::{self, Sysno};
use crate::abi::errno::Errno;
use crate::fs::{Attrs, MountTable, NodeKind};
use crate::loader::{ProcessSpec, interp_path, load_dynamic, load_static};
use crate::vcpu::mem::{PAGE_SIZE, Prot};
use crate::vcpu::{Exit, GuestMemory, Vcpu, VcpuError};

pub mod egress;
mod fd;
mod fs_ext;
mod mem_syscalls;
mod net;
mod path;
mod poll;
mod signal;
mod stat;
mod sys_misc;
mod time;

pub use fd::{Fd, FdTable};
use net::Net;
use poll::{EpollInst, EventFdInst, TimerFdInst};

/// `dirfd` value meaning "resolve relative to the current working directory".
const AT_FDCWD: i64 = -100;
/// Max symlink hops before `ELOOP`.
const SYMLINK_MAX: u32 = 16;

/// Per-process kernel-side state (swapped into `Kernel::cur` while running).
#[derive(Clone)]
struct ProcInfo {
    fds: FdTable,
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
    /// `set_tid_address` / `CLONE_CHILD_CLEARTID`: on exit, zero this guest
    /// word and futex-wake it (lets `pthread_join` return). 0 = unset.
    clear_child_tid: u64,
    /// When `Some((mm, uaddr))`, this task is parked in `FUTEX_WAIT` on that
    /// address; cleared when woken.
    futex_wait: Option<(usize, u64)>,
    /// Set by `FUTEX_WAKE` to release a parked waiter on its next slice.
    futex_woken: bool,
    run: RunState,
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
    /// Process-group id (`setpgid`/`getpgid`/`getpgrp`). `0` means "not set
    /// yet — defaults to `pid`". Inherited across `fork`.
    pgid: i32,
    /// Session id (`setsid`/`getsid`). `0` means "defaults to `pid`".
    sid: i32,
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
}

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
            mm: 0,
            files: 0,
            clear_child_tid: 0,
            futex_wait: None,
            futex_woken: false,
            run: RunState::Running,
            handlers: [SigAction::default(); 65],
            altstack: (0, 0, SS_DISABLE),
            blocked: 0,
            pending: 0,
            pgid: 0,
            sid: 0,
            parked: false,
            shared_maps: Vec::new(),
            wake_deadline: None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RunState {
    Running,
    Zombie(i32),
}

/// Outcome of a [`Kernel::pump`] step in interactive mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pumped {
    /// pid 1 exited with this code; the machine is done.
    Exited(i32),
    /// Every runnable task is parked waiting for input; feed stdin and pump
    /// again to resume (e.g. the shell is blocked on a `read` of its terminal).
    Blocked,
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
/// The synchronous fault signals this kernel can deliver to a handler.
const SIGILL: u64 = 4;
const SIGSEGV: u64 = 11;

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
    let guard = if room > STACK_GUARD_GAP * 4 { STACK_GUARD_GAP } else { PAGE_SIZE };
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
        Self { cursor: top, floor, top, free: Vec::new() }
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
    /// Syscall done; write this value into the vcpu's result register, resume.
    SetRet(i64),
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
        // ---- run phase: no kernel lock held ----
        // Interpreter reads/writes guest memory *through* GuestMemory, so it must
        // hold the memory lock for the whole run. KVM executes against the mapped
        // memslot: take the lock only to reconcile the memslot + shadow page
        // tables, then drop it so KVM_RUN runs in parallel with siblings of the
        // same address space.
        let exit = if vcpu.needs_locked_run() {
            let mut mem = space.lock().unwrap();
            vcpu.run(&mut mem)
        } else {
            let reconciled = {
                let mut mem = space.lock().unwrap();
                vcpu.reconcile(&mut mem)
            };
            match reconciled {
                Ok(()) => vcpu.run_bare(),
                Err(e) => Err(e),
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
        // ---- service phase: memory lock, then kernel lock (see above) ----
        let step = {
            let mut mem = space.lock().unwrap();
            let mut sh = kernel.shared.lock().unwrap();
            kernel.smp_service_step(i, exit, vcpu.as_mut(), &mut mem, &mut sh)
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
    /// Syscalls serviced in the current slice (preemption quantum counter). (Was
    /// `Kernel::slice_syscalls`.)
    slice_syscalls: u32,
}

/// The kernel: immutable-during-servicing config plus the coarse lock over all
/// mutable state ([`Shared`]).
///
/// Only the config fields live directly on `Kernel` (written just by `new`/the
/// pre-boot `set_*` setters, never during servicing); everything mutated while
/// servicing a syscall lives in [`Shared`] behind `shared`. Servicing therefore
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
}

/// All kernel state mutated while a syscall is serviced, behind [`Kernel`]'s
/// coarse lock. Servicing takes `&self` (the config) plus `&mut Shared` (this),
/// so a field here is reached as `sh.<field>` instead of `self.<field>`.
#[allow(clippy::struct_excessive_bools)] // independent one-shot flags, not a state enum
pub(super) struct Shared {
    mounts: MountTable,
    pipes: Vec<Pipe>,
    net: Net,
    /// `eventfd2` counters, indexed by [`Fd::Eventfd`].
    eventfds: Vec<EventFdInst>,
    /// `timerfd_create` timers, indexed by [`Fd::Timerfd`].
    timerfds: Vec<TimerFdInst>,
    /// `epoll_create1` instances, indexed by [`Fd::Epoll`].
    epolls: Vec<EpollInst>,
    stdin: Box<dyn Read + Send>,
    stdout: Box<dyn Write + Send>,
    stderr: Box<dyn Write + Send>,
    /// Buffered terminal input for interactive mode (see [`Kernel::feed_stdin`]).
    stdin_buf: VecDeque<u8>,
    /// Whether interactive stdin has been closed (EOF / Ctrl-D).
    stdin_closed: bool,
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
    /// The process name (`prctl(PR_SET_NAME)`); a fixed 16-byte, NUL-padded field.
    procname: [u8; 16],
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
    /// Anonymous-`mmap` arenas indexed by [`ProcInfo::mm`] — one per address
    /// space, so every `CLONE_VM` thread allocates from the same arena and two
    /// threads can never be handed overlapping ranges.
    mmap_areas: Vec<Arena>,
    next_pid: i32,
    /// `NIXVM_WATCHCODE` debug watch: address whose 8 bytes are checked after
    /// every syscall, and the last value seen there.
    watch_addr: Option<u64>,
    watch_last: u64,
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
        d.finish_non_exhaustive()
    }
}

impl Shared {
    /// pid 1's exit code, if it has become a zombie.
    fn pid1_code(&self) -> Option<i32> {
        self.procs.iter().flatten().find_map(|p| match p.info.run {
            RunState::Zombie(c) if p.info.pid == 1 => Some(c),
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
            .filter_map(|p| p.info.wake_deadline)
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
            .filter_map(|p| p.info.wake_deadline)
            .min()
    }

    /// Pick a runnable task for an SMP worker: `Running`, holding its vcpu (not
    /// already in flight), and not parked at the current progress epoch.
    fn pick_smp_runnable(&self, blocked_at: &BTreeMap<usize, u64>, epoch: u64) -> Option<usize> {
        (0..self.procs.len()).find(|&i| {
            let Some(Some(p)) = self.procs.get(i) else {
                return false;
            };
            p.info.run == RunState::Running
                && p.vcpu.is_some()
                && blocked_at.get(&i).copied() != Some(epoch)
        })
    }

    /// Check the running task's shared fd table out of [`Shared::file_tables`]
    /// into `cur.fds` for the duration of its slice. Called right after `cur` is
    /// swapped in. Its sibling threads (same `files` id) are parked, so the slot
    /// is free; servicing is single-threaded, so no two tasks are ever checked
    /// out at once.
    fn check_out_files(&mut self, cx: &mut ServiceCtx) {
        let f = cx.cur.files;
        cx.cur.fds = self.file_tables[f]
            .take()
            .expect("fd table already checked out");
    }

    /// Check the running task's fd table back into [`Shared::file_tables`] so its
    /// siblings see any changes it made. Called right before `cur` is swapped
    /// out. If the task exited as the last user of its table, `cur.fds` was
    /// drained and we store the emptied table back (its slot is now idle).
    fn check_in_files(&mut self, cx: &mut ServiceCtx) {
        let f = cx.cur.files;
        self.file_tables[f] = Some(std::mem::take(&mut cx.cur.fds));
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
            shared: Mutex::new(Shared {
                mounts,
                pipes: Vec::new(),
                net: Net::default(),
                eventfds: Vec::new(),
                timerfds: Vec::new(),
                epolls: Vec::new(),
                stdin: Box::new(std::io::stdin()),
                stdout: Box::new(std::io::stdout()),
                stderr: Box::new(std::io::stderr()),
                rng_state: 0,
                rlimit_nofile: (1024, 4096),
                memfd_seq: 0,
                umask: 0o022,
                procname: [0u8; 16],
                unsupported: BTreeMap::new(),
                procs: Vec::new(),
                spaces: Vec::new(),
                file_tables: Vec::new(),
                mmap_areas: Vec::new(),
                stdin_buf: VecDeque::new(),
                stdin_closed: false,
                next_pid: 2,
                watch_addr: std::env::var("NIXVM_WATCHCODE").ok().and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok()),
                watch_last: 0,
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
        self.shared.get_mut().unwrap().net.set_egress(egress);
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
    /// [`STACK_GUARD_GAP`].
    pub fn set_mmap_area(&mut self, top: u64, floor: u64) {
        self.seed.stack_limit = top; // `top` is the stack's growth floor
        self.seed.mmap_cursor = arena_top(top, floor);
        self.seed.mmap_floor = floor;
    }

    /// Set the first process's current working directory.
    pub fn set_cwd(&mut self, dir: impl Into<String>) {
        self.seed.cwd = path::normalize(&dir.into());
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
            sh.mmap_areas.push(Arena::new(info.mmap_cursor, info.mmap_floor));
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
                return Err(VcpuError::Backend(
                    "deadlock: every process is blocked".into(),
                ));
            }
        }
    }

    /// One pass over the process table on the current thread: run each runnable
    /// task's slice. Returns whether any task made progress. Shared by the
    /// blocking scheduler and the interactive [`Kernel::pump`] loop.
    ///
    /// The coarse kernel lock ([`Kernel::shared`]) is held for the whole sweep —
    /// behavior-identical to the old exclusive `&mut Kernel`, since the serial
    /// path is single-threaded and nothing else contends for it.
    fn serial_sweep(&self) -> Result<bool, VcpuError> {
        let mut sh = self.shared.lock().unwrap();
        let mut progressed = false;
        for i in 0..sh.procs.len() {
            // Run only tasks that are Running *and not parked*: a parked task
            // blocked last slice and won't make progress until woken.
            let runnable = matches!(
                sh.procs.get(i),
                Some(Some(p)) if p.info.run == RunState::Running && !p.info.parked
            );
            if !runnable {
                continue;
            }
            let mut proc = sh.procs[i].take().unwrap();
            let mm = proc.info.mm;
            let mut vcpu = proc.vcpu.take().expect("runnable task has a vcpu");
            let space_arc = Arc::clone(&sh.spaces[mm]);
            let mut guard = space_arc.lock().unwrap();
            // Own the task's per-slice servicing state for the duration of the
            // slice (was swapped into `self.cur`; now a passed-in value).
            let mut cx = ServiceCtx {
                cur: std::mem::take(&mut proc.info),
                ..ServiceCtx::default()
            };
            sh.check_out_files(&mut cx);
            let made = self.run_slice(&mut cx, &mut vcpu, &mut guard, &mut sh)?;
            if self.schedtrace {
                let end = if matches!(cx.cur.run, RunState::Zombie(_)) {
                    "ended"
                } else if cx.block {
                    "blocked"
                } else {
                    "yield"
                };
                eprintln!(
                    "[sched] pid={} slice={} syscalls={} end={end}",
                    cx.cur.pid, i, cx.slice_syscalls
                );
            }
            // The slice ended either by exiting or by blocking; `cx.block`
            // reflects the last syscall. A blocked task parks until a wake.
            let blocked = cx.block;
            sh.check_in_files(&mut cx);
            proc.info = cx.cur;
            proc.info.parked = blocked && proc.info.run == RunState::Running;
            drop(guard);
            proc.vcpu = Some(vcpu);
            sh.procs[i] = Some(proc);
            progressed |= made;
        }
        Ok(progressed)
    }

    // ---- interactive driver (the browser terminal) -----------------------

    /// Enable interactive mode: guest reads of fd 0 draw from the buffer fed via
    /// [`Kernel::feed_stdin`] and block when empty, instead of the host stdin.
    pub fn set_interactive(&mut self, yes: bool) {
        self.interactive = yes;
    }

    /// Append bytes to the interactive terminal-input buffer (keystrokes).
    pub fn feed_stdin(&mut self, bytes: &[u8]) {
        self.shared.get_mut().unwrap().stdin_buf.extend(bytes.iter().copied());
    }

    /// Signal end-of-input on the interactive stdin (Ctrl-D).
    pub fn close_stdin(&mut self) {
        self.shared.get_mut().unwrap().stdin_closed = true;
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
        // Check the initial fd table (the standard streams) into slot 0; the
        // scheduler checks it out into `cur.fds` for each slice.
        info.files = sh.file_tables.len();
        sh.file_tables.push(Some(std::mem::take(&mut info.fds)));
        sh.mmap_areas.push(Arena::new(info.mmap_cursor, info.mmap_floor));
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
        loop {
            if let Some(code) = self.shared.lock().unwrap().pid1_code() {
                return Ok(Pumped::Exited(code));
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
        &self, cx: &mut ServiceCtx,
        vcpu: &mut Box<dyn Vcpu>,
        mem: &mut GuestMemory,
        sh: &mut Shared,
    ) -> Result<bool, VcpuError> {
        let mut progressed = false;
        loop {
            let exit = vcpu.run(mem)?;
            match self.service(cx, exit, vcpu.as_mut(), mem, sh) {
                Serviced::SetRet(ret) => {
                    vcpu.set_syscall_ret(ret as u64);
                    progressed = true;
                    // `sched_yield`: the call succeeded (its return value is
                    // written above), but end the slice so siblings run. The
                    // task is *not* parked — `cx.block` stays clear.
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
        }
    }

    /// Service one guest exit against the current task (`self.cur`): dispatch a
    /// syscall, or turn a fault/halt into a zombie. Shared by the serial and
    /// SMP schedulers. Does NOT touch the vcpu's result register — the caller
    /// applies [`Serviced::SetRet`] — so the same logic works whether the vcpu
    /// lives on the main thread or is round-tripping through a worker.
    fn service(&self, cx: &mut ServiceCtx, exit: Exit, vcpu: &mut dyn Vcpu, mem: &mut GuestMemory, sh: &mut Shared) -> Serviced {
        match exit {
            Exit::Syscall => {
                let raw = vcpu.syscall_nr();
                let sys = arch::decode(self.arch, raw);
                let args = vcpu.syscall_args();
                cx.slice_syscalls = cx.slice_syscalls.saturating_add(1);
                cx.block = false;
                cx.exec_ok = false;
                let ret = self.dispatch(cx, sys, raw, &args, vcpu, mem, sh);
                // If the syscall edited this address space's page tables in place
                // (munmap, mprotect, mremap, a copy-on-write copy-in), flush the
                // running vcpu's TLB so it can't keep using a stale entry for a
                // now-unmapped or replaced page. No-op for the interpreter.
                if mem.take_tlb_dirty() {
                    vcpu.flush_tlb();
                }
                self.deliver_pending_signals(cx);
                // A syscall that returns (didn't re-block) has consumed any
                // timed-wait deadline it set; the next blocking syscall starts
                // a fresh one.
                if !cx.block {
                    cx.cur.wake_deadline = None;
                }
                self.watch_code(vcpu, mem, sys, sh);
                if let RunState::Zombie(_) = cx.cur.run {
                    Serviced::Ended
                } else if cx.block {
                    Serviced::Blocked
                } else if cx.exec_ok {
                    Serviced::Resume // resume the new image at its entry
                } else {
                    Serviced::SetRet(ret)
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
                if mem.demand_fault(addr) {
                    vcpu.flush_tlb();
                    Serviced::Resume
                }
                // A write fault on a copy-on-write page is resolved by
                // privatizing the page and re-running the instruction (the vcpu
                // left PC on the faulting store). Anything else — a read fault, a
                // write to read-only/unmapped memory, or an already-private page
                // — is a genuine segfault. This is the software mirror of a
                // hardware MMU's page-fault-driven COW.
                else if mem.cow_fault(addr, write) {
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
                } else if self.deliver_fault_signal(cx, SIGSEGV, addr, vcpu, mem) {
                    // The guest caught it (JIT trap handler): run the handler.
                    Serviced::Resume
                } else {
                    eprintln!(
                        "[fault] pid {} memory fault at {addr:#x} (write={write}, pc={:#x})",
                        cx.cur.pid,
                        vcpu.pc()
                    );
                    self.dump_fault_context(vcpu, mem);
                    cx.cur.run = RunState::Zombie(139);
                    Serviced::Ended
                }
            }
            Exit::IllegalInstruction { pc } => {
                // Dump the raw bytes at the fault so an interpreter decode gap
                // is identifiable from the report alone (the pc is under a
                // load bias for PIEs/`ld-musl`, so it can't be looked up in
                // the on-disk ELF directly).
                if self.deliver_fault_signal(cx, SIGILL, pc, vcpu, mem) {
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
                cx.cur.run = RunState::Zombie(132);
                Serviced::Ended
            }
            Exit::Halt => {
                cx.cur.run = RunState::Zombie(0);
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
                        } else if !stalled {
                            // No timer: catch a lost futex wake / host-socket
                            // data / a child that became a zombie with one
                            // forced retry round before declaring deadlock.
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
    /// the task's per-process state into `self.cur` (its slot in `sh.procs` is
    /// `take`n out for the duration, exactly as the serial scheduler does, so
    /// `fork`/`wait4`/`futex` scans don't see the running task), run the shared
    /// [`Kernel::service`] logic, then swap it back. Called with the kernel lock
    /// held, so servicing is serialized across all workers (the big kernel
    /// lock). Returns what the worker should do next with the same vcpu.
    fn smp_service_step(
        &self,
        i: usize,
        exit: Exit,
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
        sh: &mut Shared,
    ) -> SliceStep {
        let mut proc = sh.procs[i].take().expect("dispatched task is in the table");
        // Own the task's per-step servicing state (was swapped into `self.cur`;
        // now a passed-in value). `yield_now`/`block`/`exec_ok` start clear so we
        // observe only this step's value (the serial path resets them likewise).
        let mut cx = ServiceCtx {
            cur: std::mem::take(&mut proc.info),
            ..ServiceCtx::default()
        };
        sh.check_out_files(&mut cx);
        let flow = self.service(&mut cx, exit, vcpu, mem, sh);
        sh.check_in_files(&mut cx);
        proc.info = cx.cur;
        sh.procs[i] = Some(proc);
        match flow {
            Serviced::SetRet(ret) => {
                vcpu.set_syscall_ret(ret as u64);
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
    #[allow(clippy::unused_self)]
    fn watch_code(&self, vcpu: &dyn Vcpu, mem: &GuestMemory, after: Sysno, sh: &mut Shared) {
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
        mem.map(page, end - page, crate::vcpu::mem::Prot::rw()).is_ok()
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
        &self, cx: &mut ServiceCtx,
        sys: Sysno,
        raw: u64,
        args: &[u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
        sh: &mut Shared,
    ) -> i64 {
        if !self.trace {
            return self.dispatch_inner(cx, sys, raw, args, vcpu, mem, sh);
        }
        let (pid, pc) = (cx.cur.pid, vcpu.pc());
        eprintln!("[trace] pid={pid} pc={pc:#x} {sys:?} raw={raw} args={args:x?}");
        let ret = self.dispatch_inner(cx, sys, raw, args, vcpu, mem, sh);
        if (-4095..0).contains(&ret) {
            eprintln!("[trace]   = {ret} (errno {})", -ret);
        } else {
            eprintln!("[trace]   = {ret:#x}");
        }
        ret
    }

    #[allow(clippy::too_many_lines)] // one arm per syscall; a flat table is clearest.
    #[allow(clippy::too_many_arguments)]
    fn dispatch_inner(
        &self, cx: &mut ServiceCtx,
        sys: Sysno,
        raw: u64,
        args: &[u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
        sh: &mut Shared,
    ) -> i64 {
        match sys {
            Sysno::Write => self.sys_write(sh, cx, args[0], args[1], args[2], mem),
            Sysno::Read => self.sys_read(sh, cx, args[0], args[1], args[2], mem),
            // sendto/recvfrom carry an optional peer address (UDP) beyond
            // write/read; the address-aware path lives in net.rs.
            Sysno::Sendto => {
                self.sys_sendto(sh, cx, args[0], args[1], args[2], args[3], args[4], args[5], mem)
            }
            Sysno::Recvfrom => {
                self.sys_recvfrom(sh, cx, args[0], args[1], args[2], args[3], args[4], args[5], mem)
            }
            Sysno::Sendmsg => self.sys_sendmsg(sh, cx, args[0], args[1], args[2], mem),
            Sysno::Recvmsg => self.sys_recvmsg(sh, cx, args[0], args[1], args[2], mem),
            // `sched_yield` succeeds *and* ends the slice, so a sibling gets the
            // CPU — see [`Kernel::yield_now`].
            Sysno::SchedYield => {
                cx.yield_now = true;
                0
            }
            Sysno::Brk => self.sys_brk(cx, args[0], mem),
            Sysno::Mmap => self.sys_mmap(sh, cx, args, mem),
            Sysno::Munmap => self.sys_munmap(sh, cx, args[0], args[1], mem),
            Sysno::Msync => self.sys_msync(sh, cx, args[0], args[1], mem),
            Sysno::Mprotect => self.sys_mprotect(args[0], args[1], args[2], mem),
            Sysno::Mremap => {
                self.sys_mremap(sh, cx, args[0], args[1], args[2], args[3], args[4], mem)
            }
            Sysno::Madvise => self.sys_madvise(args[0], args[1], args[2], mem),
            Sysno::Mincore => self.sys_mincore(args[0], args[1], args[2], mem),
            Sysno::Uname => self.sys_uname(args[0], mem),
            Sysno::ClockGettime => sys_clock_gettime(args[1], mem),
            Sysno::Gettimeofday => time::sys_gettimeofday(args[0], mem),
            Sysno::ClockGetres => time::sys_clock_getres(args[1], mem),
            Sysno::Nanosleep => time::sys_nanosleep(args[0], args[1], mem),
            Sysno::ClockNanosleep => time::sys_nanosleep(args[2], args[3], mem),
            Sysno::Time => time::sys_time(args[0], mem),
            // The guest does not own the host clock: refuse to set it. ptrace
            // is refused too (no debugging surface).
            Sysno::Settimeofday | Sysno::ClockSettime | Sysno::Ptrace => err(Errno::EPERM),
            Sysno::Openat => self.sys_openat(sh, cx, args[0] as i64, args[1], args[2], args[3], mem),
            Sysno::Close => self.sys_close(sh, cx, args[0] as i32),
            Sysno::CloseRange => self.sys_close_range(sh, cx, args[0], args[1]),
            Sysno::Lseek => self.sys_lseek(sh, cx, args[0], args[1] as i64, args[2]),
            // Positioned & vectored I/O.
            Sysno::Pread64 => self.sys_pread(sh, cx, args[0], args[1], args[2], args[3], mem),
            Sysno::Pwrite64 => self.sys_pwrite(sh, cx, args[0], args[1], args[2], args[3], mem),
            Sysno::Preadv => self.sys_preadv(sh, cx, args[0], args[1], args[2], args[3], mem),
            Sysno::Pwritev => self.sys_pwritev(sh, cx, args[0], args[1], args[2], args[3], mem),
            // File size / sync. The sync family has no durable backing store to
            // flush (everything is in-memory or host-passthrough), so they just
            // succeed; `readahead`/`fadvise64`/`sync_file_range` are advisory.
            Sysno::Ftruncate => self.sys_ftruncate(sh, cx, args[0], args[1]),
            Sysno::Truncate => self.sys_truncate(sh, cx, args[0], args[1], mem),
            Sysno::Fallocate => self.sys_fallocate(sh, cx, args[0], args[2], args[3]),
            // File copy / link.
            Sysno::Sendfile => self.sys_sendfile(sh, cx, args[0], args[1], args[2], args[3], mem),
            Sysno::CopyFileRange => self.sys_copy_file_range(sh, cx, args, mem),
            Sysno::Link => self.sys_linkat(sh, cx, AT_FDCWD, args[0], AT_FDCWD, args[1], 0, mem),
            Sysno::Linkat => {
                self.sys_linkat(sh, cx, args[0] as i64, args[1], args[2] as i64, args[3], args[4], mem)
            }
            Sysno::Statx => self.sys_statx(sh, cx, args[0] as i64, args[1], args[2], args[4], mem),
            // Credentials: this VM runs as root and models no multi-user
            // policy, so the setters all succeed and the getters report root.
            Sysno::Getresuid | Sysno::Getresgid => {
                self.sys_getres_id(args[0], args[1], args[2], mem)
            }
            // Process groups / sessions.
            Sysno::Setpgid => self.sys_setpgid(cx, args[0] as i32, args[1] as i32),
            Sysno::Getpgid => self.sys_getpgid(sh, cx, args[0] as i32),
            Sysno::Getpgrp => i64::from(pgid_of(&cx.cur)),
            Sysno::Setsid => self.sys_setsid(cx),
            Sysno::Getsid => self.sys_getsid(sh, cx, args[0] as i32),
            // Process lifecycle.
            Sysno::Waitid => self.sys_waitid(sh, cx, args[0], args[1] as i64, args[2], args[3], mem),
            Sysno::Clone3 => self.sys_clone3(sh, cx, args[0], args[1], vcpu, mem),
            Sysno::Execveat => {
                self.sys_execveat(sh, cx, args[0] as i64, args[1], args[2], args[3], args[4], vcpu, mem)
            }
            // Sockets: `accept` is `accept4` with no flags; the `mmsg` forms
            // loop the single-message path over the message array.
            Sysno::Accept => self.sys_accept4(sh, cx, args[0], args[1], args[2], 0, mem),
            Sysno::Recvmmsg => self.sys_recvmmsg(sh, cx, args[0], args[1], args[2], args[3], mem),
            Sysno::Sendmmsg => self.sys_sendmmsg(sh, cx, args[0], args[1], args[2], mem),
            // Anonymous / notification fds. inotify/signalfd get an fd that
            // never becomes readable (no events/signals delivered — a safe
            // degradation for optional watching).
            Sysno::MemfdCreate => self.sys_memfd_create(sh, cx, args[0], mem),
            Sysno::InotifyInit1 | Sysno::Signalfd4 => self.sys_inotify_init1(sh, cx),
            // A watch descriptor the guest can pass to inotify_rm_watch (which
            // is a no-op in the always-succeed group below).
            Sysno::InotifyAddWatch => 1,
            // restart_syscall reports the interrupted call didn't resume.
            Sysno::RestartSyscall => err(Errno::EINTR),
            // pause() blocks until a signal; with our minimal signal delivery
            // it simply parks (the guest re-traps).
            Sysno::Pause => {
                cx.block = true;
                0
            }
            Sysno::Fstat => self.sys_fstat(sh, cx, args[0], args[1], mem),
            Sysno::Newfstatat => {
                self.sys_newfstatat(sh, cx, args[0] as i64, args[1], args[2], args[3], mem)
            }
            Sysno::Getdents64 => self.sys_getdents64(sh, cx, args[0], args[1], args[2], mem),
            Sysno::Getcwd => self.sys_getcwd(cx, args[0], args[1], mem),
            Sysno::Chdir => self.sys_chdir(sh, cx, args[0], mem),
            Sysno::Fchdir => self.sys_fchdir(sh, cx, args[0]),
            Sysno::Statfs => self.sys_statfs(sh, cx, args[0], args[1], mem),
            Sysno::Fstatfs => self.sys_fstatfs(cx, args[0], args[1], mem),
            Sysno::Readlinkat => {
                self.sys_readlinkat(sh, cx, args[0] as i64, args[1], args[2], args[3], mem)
            }
            Sysno::Symlinkat => self.sys_symlinkat(sh, cx, args[0], args[1] as i64, args[2], mem),
            Sysno::Mkdirat => self.sys_mkdirat(sh, cx, args[0] as i64, args[1], args[2], mem),
            Sysno::Unlinkat => self.sys_unlinkat(sh, cx, args[0] as i64, args[1], args[2], mem),
            Sysno::Renameat | Sysno::Renameat2 => {
                self.sys_renameat(sh, cx, args[0] as i64, args[1], args[2] as i64, args[3], mem)
            }
            Sysno::Faccessat | Sysno::Faccessat2 => {
                self.sys_faccessat(sh, cx, args[0] as i64, args[1], mem)
            }
            Sysno::Access => self.sys_faccessat(sh, cx, AT_FDCWD, args[0], mem),
            Sysno::Umask => self.sys_umask(sh, args[0]),
            // No extended attributes: report "no such attribute".
            Sysno::Getxattr | Sysno::Lgetxattr | Sysno::Fgetxattr => err(Errno::ENODATA),
            Sysno::Readv => self.sys_readv(sh, cx, args[0], args[1], args[2], mem),
            Sysno::Writev => self.sys_writev(sh, cx, args[0], args[1], args[2], mem),
            Sysno::Getrandom => self.sys_getrandom(sh, args[0], args[1], mem),
            Sysno::Ioctl => err(Errno::ENOTTY),
            Sysno::Fcntl => self.sys_fcntl(sh, cx, args[0], args[1]),
            Sysno::Futex => self.sys_futex(sh, cx, args, mem),
            Sysno::Pipe2 => self.sys_pipe2(sh, cx, args[0], mem),
            Sysno::Socket => self.sys_socket(sh, cx, args[0], args[1], args[2]),
            Sysno::Socketpair => self.sys_socketpair(sh, cx, args[0], args[1], args[2], args[3], mem),
            Sysno::Bind => self.sys_bind(sh, cx, args[0], args[1], args[2], mem),
            Sysno::Listen => self.sys_listen(sh, cx, args[0]),
            Sysno::Accept4 => self.sys_accept4(sh, cx, args[0], args[1], args[2], args[3], mem),
            Sysno::Connect => self.sys_connect(sh, cx, args[0], args[1], args[2], mem),
            Sysno::Getsockname => self.sys_getsockname(sh, cx, args[0], args[1], args[2], mem),
            Sysno::Getpeername => self.sys_getpeername(sh, cx, args[0], args[1], args[2], mem),
            Sysno::Setsockopt => {
                self.sys_setsockopt(sh, cx, args[0], args[1], args[2], args[3], args[4], mem)
            }
            Sysno::Getsockopt => {
                self.sys_getsockopt(sh, cx, args[0], args[1], args[2], args[3], args[4], mem)
            }
            Sysno::Shutdown => self.sys_shutdown(sh, cx, args[0], args[1]),
            // Event-notification / readiness syscalls.
            Sysno::Poll => self.sys_poll(sh, cx, args[0], args[1], args[2] as i64, mem),
            Sysno::Ppoll => self.sys_ppoll(sh, cx, args[0], args[1], args[2], args[3], args[4], mem),
            Sysno::Select => self.sys_select(sh, cx, args[0], args[1], args[2], args[3], args[4], mem),
            Sysno::Pselect6 => {
                self.sys_pselect6(sh, cx, args[0], args[1], args[2], args[3], args[4], args[5], mem)
            }
            Sysno::EpollCreate | Sysno::EpollCreate1 => self.sys_epoll_create1(sh, cx, args[0]),
            Sysno::EpollCtl => self.sys_epoll_ctl(sh, cx, args[0], args[1], args[2], args[3], mem),
            Sysno::EpollWait | Sysno::EpollPwait => {
                self.sys_epoll_wait(sh, cx, args[0], args[1], args[2], args[3] as i64, mem)
            }
            Sysno::EpollPwait2 => self.sys_epoll_pwait2(sh, cx, args[0], args[1], args[2], args[3], mem),
            Sysno::Eventfd => self.sys_eventfd2(sh, cx, args[0], 0),
            Sysno::Eventfd2 => self.sys_eventfd2(sh, cx, args[0], args[1]),
            Sysno::TimerfdCreate => self.sys_timerfd_create(sh, cx, args[0], args[1]),
            Sysno::TimerfdSettime => {
                self.sys_timerfd_settime(sh, cx, args[0], args[1], args[2], args[3], mem)
            }
            Sysno::TimerfdGettime => self.sys_timerfd_gettime(sh, cx, args[0], args[1], mem),
            Sysno::Dup => self.sys_dup(sh, cx, args[0]),
            Sysno::Dup2 | Sysno::Dup3 => self.sys_dup2(sh, cx, args[0], args[1]),
            Sysno::Clone => self.sys_clone(sh, cx, args, vcpu, mem),
            // x86-64's legacy spellings of clone: `fork` is
            // `clone(SIGCHLD, ...)`, `vfork` is `clone(CLONE_VM|CLONE_VFORK|
            // SIGCHLD, ...)` — aarch64 never had either as its own syscall.
            Sysno::Fork => self.sys_clone(sh, cx, &[0x11, 0, 0, 0, 0, 0], vcpu, mem),
            Sysno::Vfork => self.sys_clone(sh, cx, &[0x4111, 0, 0, 0, 0, 0], vcpu, mem),
            // x86-64's legacy path-based stat family, in terms of newfstatat.
            Sysno::Stat => self.sys_newfstatat(sh, cx, AT_FDCWD, args[0], args[1], 0, mem),
            Sysno::Lstat => {
                const AT_SYMLINK_NOFOLLOW: u64 = 0x100;
                self.sys_newfstatat(sh, cx, AT_FDCWD, args[0], args[1], AT_SYMLINK_NOFOLLOW, mem)
            }
            // x86-64's legacy path-based file syscalls, each the `AT_FDCWD`
            // special case of its `*at` successor.
            Sysno::Open => self.sys_openat(sh, cx, AT_FDCWD, args[0], args[1], args[2], mem),
            Sysno::Creat => {
                const O_WRONLY_CREAT_TRUNC: u64 = 0o1101;
                self.sys_openat(sh, cx, AT_FDCWD, args[0], O_WRONLY_CREAT_TRUNC, args[1], mem)
            }
            Sysno::Mkdir => self.sys_mkdirat(sh, cx, AT_FDCWD, args[0], args[1], mem),
            Sysno::Rmdir => {
                const AT_REMOVEDIR: u64 = 0x200;
                self.sys_unlinkat(sh, cx, AT_FDCWD, args[0], AT_REMOVEDIR, mem)
            }
            Sysno::Unlink => self.sys_unlinkat(sh, cx, AT_FDCWD, args[0], 0, mem),
            Sysno::Rename => self.sys_renameat(sh, cx, AT_FDCWD, args[0], AT_FDCWD, args[1], mem),
            Sysno::Symlink => self.sys_symlinkat(sh, cx, args[0], AT_FDCWD, args[1], mem),
            Sysno::Readlink => self.sys_readlinkat(sh, cx, AT_FDCWD, args[0], args[1], args[2], mem),
            Sysno::Execve => self.sys_execve(sh, cx, args[0], args[1], args[2], vcpu, mem),
            Sysno::Wait4 => self.sys_wait4(sh, cx, args[0] as i64, args[1], args[2], mem),
            Sysno::Exit => self.sys_exit(sh, cx, args[0] as i32, mem),
            Sysno::ExitGroup => self.sys_exit_group(sh, cx, args[0] as i32, mem),
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
            Sysno::RtSigpending => self.sys_rt_sigpending(cx, args[0], mem),
            Sysno::RtSigtimedwait => err(Errno::EAGAIN),
            Sysno::Kill | Sysno::Tkill => self.sys_kill(sh, cx, args[0] as i64, args[1]),
            Sysno::Tgkill => self.sys_kill(sh, cx, args[1] as i64, args[2]),
            // getpid = thread-group id; gettid = this task's id.
            Sysno::Getpid => i64::from(cx.cur.tgid),
            Sysno::Gettid => i64::from(cx.cur.pid),
            // set_tid_address records the CHILD_CLEARTID word and returns the tid.
            Sysno::SetTidAddress => {
                cx.cur.clear_child_tid = args[0];
                i64::from(cx.cur.pid)
            }
            Sysno::Getppid => i64::from(cx.cur.ppid),
            // Resource / scheduling / process-attribute syscalls (informational).
            Sysno::SchedGetaffinity => {
                sys_misc::sys_sched_getaffinity(args[1], args[2], mem)
            }
            Sysno::SchedGetparam => sys_misc::sys_sched_getparam(args[1], mem),
            Sysno::Getrusage => sys_misc::sys_getrusage(args[1], mem),
            Sysno::Sysinfo => sys_misc::sys_sysinfo(args[0], mem),
            Sysno::Times => sys_misc::sys_times(args[0], mem),
            Sysno::Getcpu => sys_misc::sys_getcpu(args[0], args[1], mem),
            Sysno::Capget => sys_misc::sys_capget(args[1], mem),
            Sysno::Prlimit64 => self.sys_prlimit64(sh, args[1], args[2], args[3], mem),
            Sysno::Getrlimit => self.sys_getrlimit(sh, args[0], args[1], mem),
            Sysno::Prctl => self.sys_prctl(sh, args, mem),
            // arch_prctl(ARCH_SET_FS) — how an x86-64 guest installs its TLS
            // register (FS.base; aarch64 uses the MSR-like TPIDR_EL0 via
            // CLONE_SETTLS instead, so this arm only ever fires for x86-64).
            // The GS and GET_* subcommands aren't modeled.
            Sysno::ArchPrctl => {
                const ARCH_SET_FS: u64 = 0x1002;
                if args[0] == ARCH_SET_FS {
                    vcpu.set_tls(args[1]);
                    0
                } else {
                    err(Errno::EINVAL)
                }
            }
            // Succeed as root / no-op: uid queries, signal setup, robust list,
            // permission/ownership/timestamp changes, socket options, clock
            // adjustment (TIME_OK), and scheduling/process-attr setters — none
            // modeled yet.
            Sysno::Adjtimex
            | Sysno::ClockAdjtime
            | Sysno::Getuid
            | Sysno::Geteuid
            | Sysno::Getgid
            | Sysno::Getegid
            | Sysno::RtSigsuspend
            | Sysno::SetRobustList
            | Sysno::Fchmodat
            | Sysno::Fchmod
            | Sysno::Fchownat
            | Sysno::Fchown
            | Sysno::Utimensat
            // Locking/sync + scheduling/process-attr setters: all no-ops.
            | Sysno::Mlock
            | Sysno::Mlock2
            | Sysno::Munlock
            | Sysno::Mlockall
            | Sysno::Munlockall
            | Sysno::SchedSetaffinity
            | Sysno::SchedGetscheduler
            | Sysno::SchedSetscheduler
            | Sysno::SchedGetPriorityMax
            | Sysno::SchedGetPriorityMin
            | Sysno::Setrlimit
            | Sysno::Getpriority
            | Sysno::Setpriority
            | Sysno::Personality
            | Sysno::Sethostname
            | Sysno::Setdomainname
            | Sysno::Capset
            // flock: advisory whole-file locks. One kernel instance runs one
            // cooperating process tree and nothing else can touch the in-VM
            // files, so granting every request immediately is safe — apk
            // locks its database this way.
            | Sysno::Flock
            // setitimer: wget/curl set an interval timer for request timeouts.
            // Not modeled (no SIGALRM delivery yet) — a no-op just means the
            // timeout never fires.
            | Sysno::Setitimer
            // Sync family: nothing is durably backed (in-memory / host
            // passthrough), so there's nothing to flush.
            | Sysno::Fsync
            | Sysno::Fdatasync
            | Sysno::Sync
            | Sysno::Syncfs
            | Sysno::Readahead
            | Sysno::Fadvise64
            | Sysno::SyncFileRange
            // Credential setters: single-user (root) VM, so they all succeed
            // (setfsuid/setfsgid return the previous id, which is always 0).
            | Sysno::Setuid
            | Sysno::Setgid
            | Sysno::Setreuid
            | Sysno::Setregid
            | Sysno::Setresuid
            | Sysno::Setresgid
            | Sysno::Setfsuid
            | Sysno::Setfsgid
            | Sysno::Setgroups
            // Namespacing/mount ops we accept but don't model (no real jail
            // layering yet): chroot, mount, umount2 succeed as no-ops.
            | Sysno::Chroot
            | Sysno::Mount
            | Sysno::Umount2
            // syslog: accept and drop (no kernel ring buffer to read).
            | Sysno::Syslog
            // rseq: accept the registration; single-cpu so the cached cpu_id
            // never goes stale. get_robust_list: nothing registered.
            | Sysno::Rseq
            | Sysno::GetRobustList
            | Sysno::InotifyRmWatch
            // getgroups: no supplementary groups (count 0).
            | Sysno::Getgroups
            | Sysno::Membarrier => 0,
            _ => {
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
    #[allow(clippy::too_many_lines)]
    fn sys_clone(&self, sh: &mut Shared, cx: &mut ServiceCtx, args: &[u64; 6], vcpu: &mut dyn Vcpu, mem: &mut GuestMemory) -> i64 {
        const CLONE_VM: u64 = 0x0000_0100;
        const CLONE_FILES: u64 = 0x0000_0400;
        const CLONE_VFORK: u64 = 0x0000_4000;
        const CLONE_THREAD: u64 = 0x0001_0000;
        const CLONE_SETTLS: u64 = 0x0008_0000;
        const CLONE_PARENT_SETTID: u64 = 0x0010_0000;
        const CLONE_CHILD_CLEARTID: u64 = 0x0020_0000;
        const CLONE_CHILD_SETTID: u64 = 0x0100_0000;

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

        let pid = sh.next_pid;
        sh.next_pid += 1;
        let mut info = cx.cur.clone();
        info.pid = pid;
        info.run = RunState::Running;
        info.futex_wait = None;
        info.futex_woken = false;
        if is_thread {
            info.tgid = cx.cur.tgid;
            info.ppid = cx.cur.ppid;
            info.is_thread = true;
        } else {
            info.tgid = pid;
            info.ppid = cx.cur.pid;
            info.is_thread = false;
        }

        // Address space: share the caller's slot (CLONE_VM), or fork a
        // copy-on-write child (both parent and child pages become shared and
        // read-on-write until the first store privatizes a page).
        let mut child_mem = if share_vm { None } else { Some(mem.fork()) };
        info.mm = if share_vm {
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
        // copy we are about to install).
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
            let (mut r, mut w) = (Vec::new(), Vec::new());
            for fd in copy.values() {
                match fd {
                    Fd::PipeRead(i) => r.push(*i),
                    Fd::PipeWrite(i) => w.push(*i),
                    Fd::Socket { .. } => sh.net.bump(fd, true),
                    _ => {}
                }
            }
            for i in r {
                sh.pipes[i].readers += 1;
            }
            for i in w {
                sh.pipes[i].writers += 1;
            }
            info.files = sh.file_tables.len();
            sh.file_tables.push(Some(copy));
        }

        if let Some(cm) = child_mem.take() {
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
        }
        child_vcpu.set_syscall_ret(0); // child returns 0 and advances past the svc
        // A copy-on-write fork (`mem.fork()`) downgraded *this* (parent) address
        // space's pages to read-only behind the running parent vcpu's back. Flush
        // its TLB so the parent's next store faults into `cow_fault` instead of
        // writing through a stale writable entry into the now-shared frame. (Free
        // for a `CLONE_VM` thread, which shares the mm and downgraded nothing.)
        vcpu.flush_tlb();
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
        &self, sh: &mut Shared, cx: &mut ServiceCtx,
        path_ptr: u64,
        argv_ptr: u64,
        envp_ptr: u64,
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some(rel) = read_path(mem, path_ptr) else {
            return err(Errno::EFAULT);
        };
        let Some(abs) = self.resolve_exec(sh, cx, &rel) else {
            return err(Errno::ENOENT);
        };
        let argv = read_string_array(mem, argv_ptr);
        let envp = read_string_array(mem, envp_ptr);
        self.exec_image(sh, cx, &abs, argv, envp, vcpu, mem)
    }

    /// `execveat(dirfd, path, argv, envp, flags)` — like `execve` but resolves
    /// `path` relative to `dirfd`, and (with `AT_EMPTY_PATH`) can exec the file
    /// `dirfd` itself refers to.
    #[allow(clippy::too_many_arguments)]
    fn sys_execveat(
        &self, sh: &mut Shared, cx: &mut ServiceCtx,
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
        self.exec_image(sh, cx, &abs, argv, envp, vcpu, mem)
    }

    /// Load `abs` (following `PT_INTERP` for dynamic executables) into a fresh
    /// address space and reset the vcpu onto it — the shared core of
    /// `execve`/`execveat`.
    #[allow(clippy::too_many_arguments)]
    fn exec_image(
        &self, sh: &mut Shared, cx: &mut ServiceCtx,
        abs: &str,
        argv: Vec<String>,
        envp: Vec<String>,
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some(elf) = self.read_file(sh, abs) else {
            return err(Errno::ENOENT);
        };
        // Reject an obviously non-ELF64 image *before* tearing down the current
        // one, so a bad `execve` leaves the process intact (real semantics) rather
        // than stranded on an empty address space.
        if elf.len() < 64 || elf[0..4] != [0x7f, b'E', b'L', b'F'] || elf[4] != 2 {
            return err(Errno::ENOEXEC);
        }
        let spec = ProcessSpec { argv, envp };
        // Replace the image *in place*: tear down the old page tables (returning
        // their frames to the shared pool) and rebuild within the SAME pool, so
        // the one KVM memslot stays valid and the process just gets a new cr3.
        mem.exec_reset();
        let loaded = if let Some(interp) = interp_path(&elf) {
            let Some(interp_elf) = self.read_file(sh, &interp) else {
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

    /// `wait4(pid, wstatus, options, rusage)` — reap a zombie child.
    #[allow(clippy::unused_self)]
    fn sys_wait4(&self, sh: &mut Shared, cx: &mut ServiceCtx, _pid: i64, wstatus: u64, options: u64, mem: &mut GuestMemory) -> i64 {
        const WNOHANG: u64 = 1;
        let cur = cx.cur.pid;
        let mut zombie = None;
        let mut has_child = false;
        for p in sh.procs.iter().flatten() {
            // Threads (CLONE_THREAD) are not reaped by wait4; only child procs.
            if p.info.ppid == cur && !p.info.is_thread {
                has_child = true;
                if let RunState::Zombie(code) = p.info.run {
                    zombie = Some((p.info.pid, code));
                    break;
                }
            }
        }
        if let Some((child, code)) = zombie {
            if wstatus != 0 {
                let status = ((code & 0xff) as u32) << 8; // WIFEXITED status
                let _ = mem.write(wstatus, &status.to_le_bytes());
            }
            for slot in &mut sh.procs {
                if slot.as_ref().is_some_and(|p| p.info.pid == child) {
                    *slot = None;
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
        cx.block = true; // wait for a child to exit
        0
    }

    /// `waitid(idtype, id, infop, options, rusage)` — the siginfo-based wait.
    /// Reaps a zombie child (or, with `WNOWAIT`, reports without reaping) and
    /// fills a `siginfo_t` instead of `wait4`'s status word.
    #[allow(clippy::too_many_arguments, clippy::unused_self)]
    fn sys_waitid(&self, sh: &mut Shared, cx: &mut ServiceCtx, idtype: u64, id: i64, infop: u64, options: u64, mem: &mut GuestMemory) -> i64 {
        const P_ALL: u64 = 0;
        const P_PID: u64 = 1;
        const P_PGID: u64 = 2;
        const WNOHANG: u64 = 1;
        const WNOWAIT: u64 = 0x0100_0000;
        const CLD_EXITED: i32 = 1;
        let cur = cx.cur.pid;
        let matches_id = |p: &ProcInfo| match idtype {
            P_ALL => true,
            P_PID => i64::from(p.pid) == id,
            P_PGID => i64::from(pgid_of(p)) == id,
            _ => false,
        };
        let mut zombie = None;
        let mut has_child = false;
        for p in sh.procs.iter().flatten() {
            if p.info.ppid == cur && !p.info.is_thread && matches_id(&p.info) {
                has_child = true;
                if let RunState::Zombie(code) = p.info.run {
                    zombie = Some((p.info.pid, code));
                    break;
                }
            }
        }
        if let Some((child, code)) = zombie {
            if infop != 0 {
                // siginfo_t: si_signo(0)=SIGCHLD(17), si_errno(4)=0,
                // si_code(8)=CLD_EXITED, si_pid(16), si_uid(20), si_status(24).
                let mut si = [0u8; 128];
                si[0..4].copy_from_slice(&17i32.to_le_bytes());
                si[8..12].copy_from_slice(&CLD_EXITED.to_le_bytes());
                si[16..20].copy_from_slice(&child.to_le_bytes());
                si[24..28].copy_from_slice(&(code & 0xff).to_le_bytes());
                let _ = mem.write(infop, &si);
            }
            if options & WNOWAIT == 0 {
                for slot in &mut sh.procs {
                    if slot.as_ref().is_some_and(|p| p.info.pid == child) {
                        *slot = None;
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
            return 0;
        }
        cx.block = true;
        0
    }

    /// `clone3(cl_args, size)` — the modern `clone`. Reads the `clone_args`
    /// struct and forwards to [`Kernel::sys_clone`] with the equivalent
    /// register arguments.
    fn sys_clone3(&self, sh: &mut Shared, cx: &mut ServiceCtx, args_ptr: u64, size: u64, vcpu: &mut dyn Vcpu, mem: &mut GuestMemory) -> i64 {
        if size < 64 {
            return err(Errno::EINVAL);
        }
        // clone_args: flags@0, pidfd@8, child_tid@16, parent_tid@24,
        // exit_signal@32, stack@40, stack_size@48, tls@56.
        let rd = |off: u64| mem.read_u64(args_ptr + off).unwrap_or(0);
        let flags = rd(0);
        let child_tid = rd(16);
        let parent_tid = rd(24);
        let exit_signal = rd(32);
        let stack = rd(40);
        let stack_size = rd(48);
        let tls = rd(56);
        // The child SP is the top of the provided stack region (grows down).
        let sp = stack.wrapping_add(stack_size);
        let legacy = [flags | (exit_signal & 0xff), sp, parent_tid, child_tid, tls, 0];
        self.sys_clone(sh, cx, &legacy, vcpu, mem)
    }

    /// `close_range(first, last, flags)` — close every open fd in `[first,
    /// last]`. `flags` (e.g. `CLOSE_RANGE_CLOEXEC`) is ignored beyond the
    /// close itself.
    fn sys_close_range(&self, sh: &mut Shared, cx: &mut ServiceCtx, first: u64, last: u64) -> i64 {
        let last = last.min(4095); // bound the sweep to a sane fd ceiling
        for fd in first..=last {
            let _ = self.sys_close(sh, cx, fd as i32);
        }
        0
    }

    /// `getresuid`/`getresgid` — write `(real, effective, saved)` = `(0,0,0)`
    /// (this VM is single-user root).
    #[allow(clippy::unused_self)] // method form keeps the dispatch table uniform
    fn sys_getres_id(&self, a: u64, b: u64, c: u64, mem: &mut GuestMemory) -> i64 {
        for p in [a, b, c] {
            if p != 0 && mem.write(p, &0u32.to_le_bytes()).is_err() {
                return err(Errno::EFAULT);
            }
        }
        0
    }

    /// `setpgid(pid, pgid)` — set the process group of `pid` (0 = self) to
    /// `pgid` (0 = the target's own pid). Only the current task is tracked.
    #[allow(clippy::unused_self)]
    fn sys_setpgid(&self, cx: &mut ServiceCtx, pid: i32, pgid: i32) -> i64 {
        if pid != 0 && pid != cx.cur.pid {
            // Setting another process's pgid isn't modeled; accept for self only.
            return err(Errno::ESRCH);
        }
        cx.cur.pgid = if pgid == 0 { cx.cur.pid } else { pgid };
        0
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
            return i64::from(if cx.cur.sid == 0 { cx.cur.pid } else { cx.cur.sid });
        }
        for p in sh.procs.iter().flatten() {
            if p.info.pid == pid {
                return i64::from(if p.info.sid == 0 { p.info.pid } else { p.info.sid });
            }
        }
        err(Errno::ESRCH)
    }

    /// `statx(dirfd, path, flags, mask, buf)` — the modern `stat`. Fills the
    /// basic-stats fields of `struct statx` from the resolved node's [`Attrs`].
    #[allow(clippy::too_many_arguments)]
    fn sys_statx(&self, sh: &mut Shared, cx: &mut ServiceCtx, dirfd: i64, path_ptr: u64, flags: u64, buf: u64, mem: &mut GuestMemory) -> i64 {
        const AT_EMPTY_PATH: u64 = 0x1000;
        let Some(rel) = read_path(mem, path_ptr) else {
            return err(Errno::EFAULT);
        };
        let attrs = if rel.is_empty() && flags & AT_EMPTY_PATH != 0 {
            match cx.cur.fds.get(dirfd as i32) {
                Some(Fd::File { path, .. } | Fd::Dir { path, .. }) => {
                    sh.mounts.stat(&path.clone())
                }
                Some(Fd::Stdin | Fd::Stdout | Fd::Stderr) => Some(stat::char_device_attrs()),
                _ => None,
            }
        } else {
            let abs = self.resolve_path(cx, dirfd, &rel);
            sh.mounts.stat(&abs)
        };
        let Some(a) = attrs else {
            return err(Errno::ENOENT);
        };
        let buf_bytes = stat::encode_statx(&a);
        if mem.write(buf, &buf_bytes).is_err() {
            return err(Errno::EFAULT);
        }
        0
    }

    /// `memfd_create(name, flags)` — an anonymous, initially-empty read/write
    /// file. Backed by a uniquely-named node in `/tmp` (a tmpfs), which gives
    /// the read/write/`ftruncate`/`mmap` behavior programs expect from a memfd
    /// (the "not linked into any directory" nuance is not modeled).
    #[allow(clippy::unused_self)]
    fn sys_memfd_create(&self, sh: &mut Shared, cx: &mut ServiceCtx, name_ptr: u64, mem: &GuestMemory) -> i64 {
        let name = read_path(mem, name_ptr).unwrap_or_default();
        let short: String = name.chars().take(64).filter(|c| *c != '/').collect();
        sh.memfd_seq += 1;
        // Back it at the (always-writable) root with a dot-prefixed name so it
        // stays out of ordinary `ls` output — the root is a tmpfs/overlay in
        // every configuration, so this doesn't depend on `/tmp` existing.
        let path = format!("/.memfd.{short}.{}", sh.memfd_seq);
        if sh.mounts.create(&path, 0o600).is_err() {
            return err(Errno::ENOSPC);
        }
        i64::from(cx.cur.fds.alloc(Fd::File { path, offset: 0 }))
    }

    /// `inotify_init1`/`signalfd4` stub — a descriptor that is always empty
    /// (never becomes readable). Programs get a valid fd and simply never see
    /// events/signals, which is a safe degradation for optional watching.
    #[allow(clippy::unused_self)]
    fn sys_inotify_init1(&self, sh: &mut Shared, cx: &mut ServiceCtx) -> i64 {
        let idx = sh.eventfds.len();
        sh.eventfds.push(EventFdInst::default());
        i64::from(cx.cur.fds.alloc(Fd::Eventfd(idx)))
    }

    /// `exit` — terminate just this task: run its `CLONE_CHILD_CLEARTID`
    /// notification (so a joiner wakes), close its fds (so pipe peers see EOF),
    /// and become a zombie until reaped.
    fn sys_exit(&self, sh: &mut Shared, cx: &mut ServiceCtx, code: i32, mem: &mut GuestMemory) -> i64 {
        // Flush any un-munmap'd writable shared file mappings first.
        if !cx.cur.shared_maps.is_empty() {
            self.flush_shared_maps(sh, cx, 0, 0, mem);
        }
        let ctid = cx.cur.clear_child_tid;
        let mm = cx.cur.mm;
        if ctid != 0 {
            let _ = mem.write(ctid, &0u32.to_le_bytes());
            self.futex_wake(sh, mm, ctid, i32::MAX);
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
                self.bump_pipe(sh, &fd, false);
            }
        }
        // Last task of this address space: return its frames to the shared pool
        // (page tables + private data pages), so a long-lived process tree does
        // not accumulate dead processes' frames. Threads sharing the mm keep it.
        if !self.has_cowaiter(sh, mm) {
            mem.release();
        }
        cx.cur.run = RunState::Zombie(code & 0xff);
        0
    }

    /// `exit_group` — terminate the whole thread group: this task plus every
    /// sibling sharing our `tgid`. Each dying task closes its fds; the running
    /// task also runs its `CLONE_CHILD_CLEARTID` notification.
    fn sys_exit_group(&self, sh: &mut Shared, cx: &mut ServiceCtx, code: i32, mem: &mut GuestMemory) -> i64 {
        // Flush any un-munmap'd writable shared file mappings first.
        if !cx.cur.shared_maps.is_empty() {
            self.flush_shared_maps(sh, cx, 0, 0, mem);
        }
        let tgid = cx.cur.tgid;
        let status = code & 0xff;
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
            p.info.run = RunState::Zombie(status);
        }
        // Close each distinct table's fds (touches `sh.pipes`/`sh.net`, so
        // it can't run while borrowing `sh.procs`). The current task's table
        // is checked out into `cur.fds` (its slot is `None`), so it's skipped
        // here and closed by the `sys_exit` tail call.
        let mut to_close: Vec<Fd> = Vec::new();
        for f in files_ids {
            if let Some(Some(t)) = sh.file_tables.get_mut(f) {
                to_close.extend(t.drain());
            }
        }
        for fd in to_close {
            self.bump_pipe(sh, &fd, false);
        }
        // `cx.cur` is this task, taken out of the table for its slice.
        self.sys_exit(sh, cx, code, mem)
    }

    /// `futex(uaddr, op, val, ...)` — the parking primitive under mutexes,
    /// condvars, and `pthread_join`.
    ///
    /// `FUTEX_WAIT`: if `*uaddr != val` the caller is already past the wait, so
    /// return `EAGAIN` immediately. Otherwise the caller parks — but only if
    /// another task could ever wake it; when this is the sole runnable task
    /// (the common single-threaded-musl case) parking would just deadlock, so
    /// we report a spurious wake (return 0) instead. A parked task re-traps the
    /// same `futex` on each slice (its PC never advanced) and returns once
    /// `FUTEX_WAKE` flips its `futex_woken` flag — decoupled from the value, as
    /// real futexes require. `FUTEX_WAKE` releases up to `val` parked waiters on
    /// `(mm, uaddr)`.
    fn sys_futex(&self, sh: &mut Shared, cx: &mut ServiceCtx, args: &[u64; 6], mem: &GuestMemory) -> i64 {
        const FUTEX_WAIT: u64 = 0;
        const FUTEX_WAKE: u64 = 1;
        const FUTEX_REQUEUE: u64 = 3;
        const FUTEX_CMP_REQUEUE: u64 = 4;
        const FUTEX_WAIT_BITSET: u64 = 9;
        const FUTEX_WAKE_BITSET: u64 = 10;
        let uaddr = args[0];
        let op = args[1] & 0x7f; // strip FUTEX_PRIVATE_FLAG / CLOCK_REALTIME
        let val = args[2] as u32;
        let mm = cx.cur.mm;
        match op {
            FUTEX_WAIT | FUTEX_WAIT_BITSET => {
                // Woken by an explicit FUTEX_WAKE (directly, or after being
                // requeued to another address by a condvar signal): consume it,
                // regardless of which address the wake targeted.
                if cx.cur.futex_woken {
                    cx.cur.futex_wait = None;
                    cx.cur.futex_woken = false;
                    return 0;
                }
                // Parked on a *different* address than this call names — i.e.
                // requeued (pthread_cond_signal moved us from the condvar futex
                // to the mutex futex). Stay parked; only an explicit wake on the
                // requeue target releases us, so don't re-compare this address's
                // value (which would spuriously return EAGAIN and desync the
                // condvar wait).
                if matches!(cx.cur.futex_wait, Some(w) if w != (mm, uaddr)) {
                    cx.block = true;
                    return 0;
                }
                // Fresh wait, or a re-check on the same address. Re-read the
                // word: if it no longer equals `val`, the wait is over (this is
                // what makes a "lost" plain wake safe — an unlock that changed
                // the word is caught here). A real futex compares atomically at
                // wait time; we compare on every re-run.
                match mem.read_u32(uaddr) {
                    Ok(cur) if cur != val => {
                        cx.cur.futex_wait = None;
                        cx.cur.futex_woken = false;
                        err(Errno::EAGAIN)
                    }
                    Ok(_) if !self.has_cowaiter(sh, mm) => {
                        // No sibling shares this address space, so no one can
                        // ever `FUTEX_WAKE` us — parking would be a false
                        // deadlock. Report a spurious wake instead (permitted by
                        // the futex contract; the caller re-checks its predicate
                        // and loops). This is the common single-threaded-musl
                        // case; a real thread group takes the parking path.
                        cx.cur.futex_wait = None;
                        cx.cur.futex_woken = false;
                        0
                    }
                    Ok(_) => {
                        cx.cur.futex_wait = Some((mm, uaddr));
                        cx.cur.futex_woken = false;
                        cx.block = true;
                        0
                    }
                    Err(_) => err(Errno::EFAULT),
                }
            }
            FUTEX_WAKE | FUTEX_WAKE_BITSET => self.futex_wake(sh, mm, uaddr, val as i32),
            // Requeue: wake up to `val` waiters on `uaddr`, then move up to
            // `val2` of the rest to wait on `uaddr2` instead. This is how
            // musl's pthread_cond_signal/broadcast hand a woken thread off to
            // the mutex — without it the condvar futex keeps the waiters and
            // the later mutex wake finds no one (the deadlock node hit).
            FUTEX_REQUEUE | FUTEX_CMP_REQUEUE => {
                if op == FUTEX_CMP_REQUEUE {
                    let expected = args[5] as u32;
                    match mem.read_u32(uaddr) {
                        Ok(cur) if cur != expected => return err(Errno::EAGAIN),
                        Err(_) => return err(Errno::EFAULT),
                        Ok(_) => {}
                    }
                }
                let nr_wake = i64::from(val);
                let nr_requeue = args[3] as i64;
                let uaddr2 = args[4];
                self.futex_requeue(sh, mm, uaddr, uaddr2, nr_wake, nr_requeue)
            }
            _ => 0,
        }
    }

    /// Whether any *other* live task shares this address space (`mm`) and could
    /// therefore issue a `FUTEX_WAKE` against it. `self.cur` is out of the table
    /// during its slice, so a scan of `sh.procs` sees only the siblings.
    #[allow(clippy::unused_self)]
    fn has_cowaiter(&self, sh: &mut Shared, mm: usize) -> bool {
        sh.procs
            .iter()
            .flatten()
            .any(|p| p.info.mm == mm && !matches!(p.info.run, RunState::Zombie(_)))
    }

    /// Wake up to `nr_wake` waiters on `(mm, uaddr)`, then requeue up to
    /// `nr_requeue` of the remaining waiters to wait on `(mm, uaddr2)`. Returns
    /// the number of waiters woken (Linux's `FUTEX_REQUEUE` return value).
    #[allow(clippy::unused_self)]
    fn futex_requeue(&self, sh: &mut Shared, mm: usize, uaddr: u64, uaddr2: u64, nr_wake: i64, nr_requeue: i64) -> i64 {
        let mut woken = 0i64;
        let mut requeued = 0i64;
        for p in sh.procs.iter_mut().flatten() {
            if p.info.futex_wait != Some((mm, uaddr)) || p.info.futex_woken {
                continue;
            }
            if woken < nr_wake {
                p.info.futex_woken = true;
                p.info.parked = false;
                woken += 1;
            } else if requeued < nr_requeue {
                // Move it to the new address; it stays parked until an explicit
                // wake on `uaddr2`.
                p.info.futex_wait = Some((mm, uaddr2));
                requeued += 1;
            } else {
                break;
            }
        }
        woken
    }

    /// Release up to `n` tasks parked in `FUTEX_WAIT` on `(mm, uaddr)`; returns
    /// how many were woken.
    #[allow(clippy::unused_self)]
    fn futex_wake(&self, sh: &mut Shared, mm: usize, uaddr: u64, n: i32) -> i64 {
        let mut woken = 0i64;
        for p in sh.procs.iter_mut().flatten() {
            if woken >= i64::from(n) {
                break;
            }
            if p.info.futex_wait == Some((mm, uaddr)) && !p.info.futex_woken {
                p.info.futex_woken = true;
                p.info.parked = false; // make it runnable so the sweep re-runs it
                woken += 1;
            }
        }
        woken
    }

    // ---- files & fds ------------------------------------------------------

    /// `write(fd, buf, count)` — stdio sinks (fd 1/2), files, and pipes.
    fn sys_write(&self, sh: &mut Shared, cx: &mut ServiceCtx, fd: u64, buf: u64, count: u64, mem: &GuestMemory) -> i64 {
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
            Some(Fd::File { path, offset }) => match sh.mounts.write_at(&path, offset, &data) {
                Ok(n) => {
                    if let Some(Fd::File { offset, .. }) = cx.cur.fds.get_mut(fd as i32) {
                        *offset += n as u64;
                    }
                    n as i64
                }
                Err(e) => io_errno(&e),
            },
            Some(Fd::PipeWrite(i)) => self.write_pipe(sh, i, &data),
            Some(Fd::Socket { sock, end }) => self.write_socket(sh, cx, sock, end, &data),
            Some(Fd::Eventfd(i)) => self.write_eventfd(sh, cx, i, &data),
            _ => err(Errno::EBADF),
        }
    }

    /// `read(fd, buf, count)` — stdin, files, and pipes.
    fn sys_read(&self, sh: &mut Shared, cx: &mut ServiceCtx, fd: u64, buf: u64, count: u64, mem: &mut GuestMemory) -> i64 {
        match cx.cur.fds.get(fd as i32).cloned() {
            Some(Fd::Stdin) if self.interactive => {
                // Draw from the buffered terminal input; block (re-trap) when it
                // is empty and not yet closed, so the embedder can pump more.
                if sh.stdin_buf.is_empty() {
                    if sh.stdin_closed {
                        return 0; // EOF
                    }
                    cx.block = true;
                    return 0;
                }
                let n = (count as usize).min(sh.stdin_buf.len());
                let chunk: Vec<u8> = sh.stdin_buf.drain(..n).collect();
                if mem.write(buf, &chunk).is_err() {
                    return err(Errno::EFAULT);
                }
                n as i64
            }
            Some(Fd::Stdin) => {
                let mut tmp = vec![0u8; count as usize];
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
            Some(Fd::PipeRead(i)) => self.read_pipe(sh, cx, i, buf, count, mem),
            Some(Fd::Socket { sock, end }) => self.read_socket(sh, cx, sock, end, buf, count, mem),
            Some(Fd::Eventfd(i)) => self.read_eventfd(sh, cx, i, buf, count, mem),
            Some(Fd::Timerfd(i)) => self.read_timerfd(sh, cx, i, buf, count, mem),
            Some(Fd::File { path, offset }) => {
                let mut tmp = vec![0u8; count as usize];
                match sh.mounts.read_at(&path, offset, &mut tmp) {
                    Ok(n) => {
                        if mem.write(buf, &tmp[..n]).is_err() {
                            return err(Errno::EFAULT);
                        }
                        if let Some(Fd::File { offset, .. }) = cx.cur.fds.get_mut(fd as i32) {
                            *offset += n as u64;
                        }
                        n as i64
                    }
                    Err(e) => io_errno(&e),
                }
            }
            _ => err(Errno::EBADF),
        }
    }

    /// Read from pipe `i`. Empty with writers still open -> block; empty with no
    /// writers -> EOF (0).
    #[allow(clippy::unused_self)]
    fn read_pipe(&self, sh: &mut Shared, cx: &mut ServiceCtx, i: usize, buf: u64, count: u64, mem: &mut GuestMemory) -> i64 {
        if sh.pipes[i].buf.is_empty() {
            if sh.pipes[i].writers > 0 {
                cx.block = true;
            }
            return 0;
        }
        let n = count.min(sh.pipes[i].buf.len() as u64) as usize;
        let data: Vec<u8> = sh.pipes[i].buf.drain(..n).collect();
        if mem.write(buf, &data).is_err() {
            return err(Errno::EFAULT);
        }
        n as i64
    }

    /// Write to pipe `i` (`EPIPE` if all readers are gone).
    #[allow(clippy::unused_self)]
    fn write_pipe(&self, sh: &mut Shared, i: usize, data: &[u8]) -> i64 {
        if sh.pipes[i].readers == 0 {
            return err(Errno::EPIPE);
        }
        sh.pipes[i].buf.extend(data.iter().copied());
        data.len() as i64
    }

    /// `pread64(fd, buf, count, offset)` — read at `offset` without moving the
    /// fd's position. Files only (a pipe/socket has no position → `ESPIPE`).
    #[allow(clippy::too_many_arguments, clippy::unused_self)]
    fn sys_pread(&self, sh: &mut Shared, cx: &mut ServiceCtx, fd: u64, buf: u64, count: u64, offset: u64, mem: &mut GuestMemory) -> i64 {
        let Some(Fd::File { path, .. }) = cx.cur.fds.get(fd as i32).cloned() else {
            return err(Errno::ESPIPE);
        };
        let mut tmp = vec![0u8; count as usize];
        match sh.mounts.read_at(&path, offset, &mut tmp) {
            Ok(n) => {
                if mem.write(buf, &tmp[..n]).is_err() {
                    return err(Errno::EFAULT);
                }
                n as i64
            }
            Err(e) => io_errno(&e),
        }
    }

    /// `pwrite64(fd, buf, count, offset)` — write at `offset` without moving
    /// the fd's position.
    #[allow(clippy::too_many_arguments, clippy::unused_self)]
    fn sys_pwrite(&self, sh: &mut Shared, cx: &mut ServiceCtx, fd: u64, buf: u64, count: u64, offset: u64, mem: &GuestMemory) -> i64 {
        let Some(Fd::File { path, .. }) = cx.cur.fds.get(fd as i32).cloned() else {
            return err(Errno::ESPIPE);
        };
        let Ok(data) = mem.read_vec(buf, count as usize) else {
            return err(Errno::EFAULT);
        };
        match sh.mounts.write_at(&path, offset, &data) {
            Ok(n) => n as i64,
            Err(e) => io_errno(&e),
        }
    }

    /// `preadv(fd, iov, iovcnt, offset)` — scatter a positioned read across
    /// iovecs. `offset` is `pos_l` (`pos_h`, the 32-bit-compat high word, is 0
    /// for 64-bit callers).
    #[allow(clippy::too_many_arguments)]
    fn sys_preadv(&self, sh: &mut Shared, cx: &mut ServiceCtx, fd: u64, iov: u64, iovcnt: u64, offset: u64, mem: &mut GuestMemory) -> i64 {
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
            let r = self.sys_pread(sh, cx, fd, base, len, cur, mem);
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
    fn sys_pwritev(&self, sh: &mut Shared, cx: &mut ServiceCtx, fd: u64, iov: u64, iovcnt: u64, offset: u64, mem: &GuestMemory) -> i64 {
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
            let r = self.sys_pwrite(sh, cx, fd, base, len, cur, mem);
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
    fn sys_ftruncate(&self, sh: &mut Shared, cx: &mut ServiceCtx, fd: u64, len: u64) -> i64 {
        let Some(Fd::File { path, .. }) = cx.cur.fds.get(fd as i32).cloned() else {
            return err(Errno::EBADF);
        };
        match sh.mounts.truncate(&path, len) {
            Ok(()) => 0,
            Err(e) => io_errno(&e),
        }
    }

    /// `truncate(path, len)` — resize by path.
    fn sys_truncate(&self, sh: &mut Shared, cx: &mut ServiceCtx, pathptr: u64, len: u64, mem: &GuestMemory) -> i64 {
        let Some(rel) = read_path(mem, pathptr) else {
            return err(Errno::EFAULT);
        };
        let abs = self.resolve_path(cx, AT_FDCWD, &rel);
        match sh.mounts.truncate(&abs, len) {
            Ok(()) => 0,
            Err(e) => io_errno(&e),
        }
    }

    /// `fallocate(fd, mode, offset, len)` — for the default allocate/extend
    /// mode, grow the file to at least `offset + len`; other modes (punch
    /// hole, and so on) are accepted as no-ops.
    #[allow(clippy::unused_self)]
    fn sys_fallocate(&self, sh: &mut Shared, cx: &mut ServiceCtx, fd: u64, offset: u64, len: u64) -> i64 {
        let Some(Fd::File { path, .. }) = cx.cur.fds.get(fd as i32).cloned() else {
            return err(Errno::EBADF);
        };
        let want = offset.saturating_add(len);
        let cur = sh.mounts.stat(&path).map_or(0, |a| a.size);
        if want > cur {
            match sh.mounts.truncate(&path, want) {
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
    fn sys_sendfile(&self, sh: &mut Shared, cx: &mut ServiceCtx, out_fd: u64, in_fd: u64, offset_ptr: u64, count: u64, mem: &mut GuestMemory) -> i64 {
        // Resolve the source position.
        let use_ptr = offset_ptr != 0;
        let start = if use_ptr {
            match mem.read_u64(offset_ptr) {
                Ok(v) => v,
                Err(_) => return err(Errno::EFAULT),
            }
        } else {
            match cx.cur.fds.get(in_fd as i32) {
                Some(Fd::File { offset, .. }) => *offset,
                _ => return err(Errno::EINVAL),
            }
        };
        let Some(Fd::File { path, .. }) = cx.cur.fds.get(in_fd as i32).cloned() else {
            return err(Errno::EINVAL);
        };
        let mut buf = vec![0u8; count as usize];
        let n = match sh.mounts.read_at(&path, start, &mut buf) {
            Ok(n) => n,
            Err(e) => return io_errno(&e),
        };
        buf.truncate(n);
        // Write it out through the normal write path (files, pipes, sockets).
        let written = match cx.cur.fds.get(out_fd as i32).cloned() {
            Some(Fd::File { path, offset }) => match sh.mounts.write_at(&path, offset, &buf) {
                Ok(w) => {
                    if let Some(Fd::File { offset, .. }) = cx.cur.fds.get_mut(out_fd as i32) {
                        *offset += w as u64;
                    }
                    w as i64
                }
                Err(e) => io_errno(&e),
            },
            Some(Fd::Stdout) => sh.stdout.write_all(&buf).map_or(err(Errno::EIO), |()| buf.len() as i64),
            Some(Fd::Stderr) => sh.stderr.write_all(&buf).map_or(err(Errno::EIO), |()| buf.len() as i64),
            Some(Fd::PipeWrite(i)) => self.write_pipe(sh, i, &buf),
            Some(Fd::Socket { sock, end }) => self.write_socket(sh, cx, sock, end, &buf),
            _ => err(Errno::EBADF),
        };
        if written < 0 {
            return written;
        }
        let advanced = written as u64;
        if use_ptr {
            let _ = mem.write_u64(offset_ptr, start + advanced);
        } else if let Some(Fd::File { offset, .. }) = cx.cur.fds.get_mut(in_fd as i32) {
            *offset += advanced;
        }
        written
    }

    /// `copy_file_range(fd_in, off_in, fd_out, off_out, len, flags)` — copy
    /// between two files, honoring the optional in/out offset pointers.
    #[allow(clippy::unused_self)]
    fn sys_copy_file_range(&self, sh: &mut Shared, cx: &mut ServiceCtx, a: &[u64; 6], mem: &mut GuestMemory) -> i64 {
        let (fd_in, off_in_p, fd_out, off_out_p, len) = (a[0], a[1], a[2], a[3], a[4]);
        let Some(Fd::File { path: in_path, offset: in_pos }) = cx.cur.fds.get(fd_in as i32).cloned() else {
            return err(Errno::EBADF);
        };
        let in_off = if off_in_p != 0 {
            mem.read_u64(off_in_p).unwrap_or(in_pos)
        } else {
            in_pos
        };
        let mut buf = vec![0u8; len as usize];
        let n = match sh.mounts.read_at(&in_path, in_off, &mut buf) {
            Ok(n) => n,
            Err(e) => return io_errno(&e),
        };
        buf.truncate(n);
        let Some(Fd::File { path: out_path, offset: out_pos }) = cx.cur.fds.get(fd_out as i32).cloned() else {
            return err(Errno::EBADF);
        };
        let out_off = if off_out_p != 0 {
            mem.read_u64(off_out_p).unwrap_or(out_pos)
        } else {
            out_pos
        };
        let w = match sh.mounts.write_at(&out_path, out_off, &buf) {
            Ok(w) => w,
            Err(e) => return io_errno(&e),
        };
        // Advance the offsets (pointer or fd position).
        if off_in_p != 0 {
            let _ = mem.write_u64(off_in_p, in_off + w as u64);
        } else if let Some(Fd::File { offset, .. }) = cx.cur.fds.get_mut(fd_in as i32) {
            *offset += w as u64;
        }
        if off_out_p != 0 {
            let _ = mem.write_u64(off_out_p, out_off + w as u64);
        } else if let Some(Fd::File { offset, .. }) = cx.cur.fds.get_mut(fd_out as i32) {
            *offset += w as u64;
        }
        w as i64
    }

    /// `linkat(olddirfd, old, newdirfd, new, flags)` (and plain `link`) — the
    /// mount table has no true hard-link primitive, so this copies the source
    /// file's contents to the new path (correct for the overwhelmingly common
    /// use — same-content at a second name; the shared-inode nuance is lost).
    #[allow(clippy::too_many_arguments)]
    fn sys_linkat(&self, sh: &mut Shared, cx: &mut ServiceCtx, olddirfd: i64, oldp: u64, newdirfd: i64, newp: u64, _flags: u64, mem: &GuestMemory) -> i64 {
        let (Some(orel), Some(nrel)) = (read_path(mem, oldp), read_path(mem, newp)) else {
            return err(Errno::EFAULT);
        };
        let old_abs = self.resolve_path(cx, olddirfd, &orel);
        let new_abs = self.resolve_path(cx, newdirfd, &nrel);
        let Some(attrs) = sh.mounts.stat(&old_abs) else {
            return err(Errno::ENOENT);
        };
        if attrs.kind == NodeKind::Dir {
            return err(Errno::EPERM); // can't hard-link a directory
        }
        // Read the whole source, create the target, copy.
        let mut data = vec![0u8; attrs.size as usize];
        if sh.mounts.read_at(&old_abs, 0, &mut data).is_err() {
            return err(Errno::EIO);
        }
        if let Err(e) = sh.mounts.create(&new_abs, attrs.mode & 0o7777) {
            return io_errno(&e);
        }
        match sh.mounts.write_at(&new_abs, 0, &data) {
            Ok(_) => 0,
            Err(e) => io_errno(&e),
        }
    }

    /// `readv(fd, iov, iovcnt)` — scatter a read across `struct iovec` entries.
    /// A short read (or a blocking fd) stops after the first partially-filled
    /// iovec, like the real syscall.
    fn sys_readv(&self, sh: &mut Shared, cx: &mut ServiceCtx, fd: u64, iov: u64, iovcnt: u64, mem: &mut GuestMemory) -> i64 {
        let mut total = 0i64;
        for i in 0..iovcnt {
            let ent = iov + i * 16;
            let (Ok(base), Ok(len)) = (mem.read_u64(ent), mem.read_u64(ent + 8)) else {
                return if total > 0 { total } else { err(Errno::EFAULT) };
            };
            if len == 0 {
                continue;
            }
            let r = self.sys_read(sh, cx, fd, base, len, mem);
            if r < 0 {
                return if total > 0 { total } else { r };
            }
            total += r;
            if (r as u64) < len {
                break; // short read: don't touch the remaining iovecs
            }
        }
        total
    }

    /// `writev(fd, iov, iovcnt)` — gather `struct iovec { base; len }` entries.
    fn sys_writev(&self, sh: &mut Shared, cx: &mut ServiceCtx, fd: u64, iov: u64, iovcnt: u64, mem: &GuestMemory) -> i64 {
        let mut total = 0i64;
        for i in 0..iovcnt {
            let ent = iov + i * 16;
            let (Ok(base), Ok(len)) = (mem.read_u64(ent), mem.read_u64(ent + 8)) else {
                return if total > 0 { total } else { err(Errno::EFAULT) };
            };
            if len == 0 {
                continue;
            }
            let r = self.sys_write(sh, cx, fd, base, len, mem);
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
    fn sys_prlimit64(&self, sh: &mut Shared, resource: u64, new_limit: u64, old_limit: u64, mem: &mut GuestMemory) -> i64 {
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
    fn sys_getrlimit(&self, sh: &mut Shared, resource: u64, buf: u64, mem: &mut GuestMemory) -> i64 {
        let (cur, max) = self.rlimit_pair(sh, resource);
        sys_misc::write_rlimit(mem, buf, cur, max)
    }

    /// `fcntl(fd, cmd, ...)` — the subset real programs need at startup.
    fn sys_fcntl(&self, sh: &mut Shared, cx: &mut ServiceCtx, fd: u64, cmd: u64) -> i64 {
        const F_DUPFD: u64 = 0;
        const F_GETFL: u64 = 3;
        const F_DUPFD_CLOEXEC: u64 = 1030;
        // Every fcntl command operates on an open fd. Returning success for a
        // closed fd breaks the common "mark every fd from 3 up cloexec until
        // EBADF" loop (node/libuv do this at startup) into an unbounded spin —
        // it must see EBADF to stop.
        if cx.cur.fds.get(fd as i32).is_none() {
            return err(Errno::EBADF);
        }
        match cmd {
            F_DUPFD | F_DUPFD_CLOEXEC => match cx.cur.fds.get(fd as i32).cloned() {
                Some(f) => {
                    self.bump_pipe(sh, &f, true);
                    i64::from(cx.cur.fds.alloc(f))
                }
                None => err(Errno::EBADF),
            },
            F_GETFL => 2,
            _ => 0,
        }
    }

    /// `openat(dirfd, path, flags, mode)` against the mount table.
    #[allow(clippy::too_many_arguments)]
    fn sys_openat(
        &self, sh: &mut Shared, cx: &mut ServiceCtx,
        dirfd: i64,
        pathptr: u64,
        flags: u64,
        mode: u64,
        mem: &GuestMemory,
    ) -> i64 {
        const O_CREAT: u64 = 0o100;
        const O_TRUNC: u64 = 0o1000;

        let Some(rel) = read_path(mem, pathptr) else {
            return err(Errno::EFAULT);
        };
        let abs = self.resolve_path(cx, dirfd, &rel);
        let abs = self.follow_symlinks(sh, &abs).unwrap_or(abs);
        if self.trace {
            eprintln!("[open] pid={} {abs:?}", cx.cur.pid);
        }

        if sh.mounts.stat(&abs).is_none() {
            if flags & O_CREAT != 0 {
                if let Err(e) = sh.mounts.create(&abs, (mode & 0o777) as u32) {
                    return io_errno(&e);
                }
            } else {
                return err(Errno::ENOENT);
            }
        } else if flags & O_TRUNC != 0 {
            let _ = sh.mounts.truncate(&abs, 0);
        }

        let Some(attrs) = sh.mounts.stat(&abs) else {
            return err(Errno::ENOENT);
        };
        let fd = if attrs.kind == NodeKind::Dir {
            cx.cur.fds.alloc(Fd::Dir { path: abs, pos: 0 })
        } else {
            cx.cur.fds.alloc(Fd::File {
                path: abs,
                offset: 0,
            })
        };
        i64::from(fd)
    }

    /// `close(fd)`.
    fn sys_close(&self, sh: &mut Shared, cx: &mut ServiceCtx, fd: i32) -> i64 {
        match cx.cur.fds.close(fd) {
            Some(f) => {
                self.bump_pipe(sh, &f, false);
                0
            }
            None => err(Errno::EBADF),
        }
    }

    /// `pipe2(fds, flags)` — create an anonymous pipe.
    #[allow(clippy::unused_self)]
    fn sys_pipe2(&self, sh: &mut Shared, cx: &mut ServiceCtx, fds_ptr: u64, mem: &mut GuestMemory) -> i64 {
        let idx = sh.pipes.len();
        sh.pipes.push(Pipe {
            buf: VecDeque::new(),
            readers: 1,
            writers: 1,
        });
        let rfd = cx.cur.fds.alloc(Fd::PipeRead(idx));
        let wfd = cx.cur.fds.alloc(Fd::PipeWrite(idx));
        let mut b = [0u8; 8];
        b[0..4].copy_from_slice(&rfd.to_le_bytes());
        b[4..8].copy_from_slice(&wfd.to_le_bytes());
        if mem.write(fds_ptr, &b).is_err() {
            return err(Errno::EFAULT);
        }
        0
    }

    /// `dup(oldfd)`.
    fn sys_dup(&self, sh: &mut Shared, cx: &mut ServiceCtx, oldfd: u64) -> i64 {
        let Some(fd) = cx.cur.fds.get(oldfd as i32).cloned() else {
            return err(Errno::EBADF);
        };
        self.bump_pipe(sh, &fd, true);
        i64::from(cx.cur.fds.alloc(fd))
    }

    /// `dup2`/`dup3(oldfd, newfd)`.
    fn sys_dup2(&self, sh: &mut Shared, cx: &mut ServiceCtx, oldfd: u64, newfd: u64) -> i64 {
        let Some(fd) = cx.cur.fds.get(oldfd as i32).cloned() else {
            return err(Errno::EBADF);
        };
        if oldfd == newfd {
            return newfd as i64;
        }
        if let Some(old) = cx.cur.fds.close(newfd as i32) {
            self.bump_pipe(sh, &old, false);
        }
        self.bump_pipe(sh, &fd, true);
        cx.cur.fds.insert(newfd as i32, fd);
        newfd as i64
    }

    /// Adjust the reader/writer refcount of the pipe a fd refers to (if any).
    #[allow(clippy::unused_self)]
    fn bump_pipe(&self, sh: &mut Shared, fd: &Fd, inc: bool) {
        let apply = |n: &mut usize| {
            if inc {
                *n += 1;
            } else {
                *n = n.saturating_sub(1);
            }
        };
        match fd {
            Fd::PipeRead(i) => apply(&mut sh.pipes[*i].readers),
            Fd::PipeWrite(i) => apply(&mut sh.pipes[*i].writers),
            f => sh.net.bump(f, inc),
        }
    }

    /// `lseek(fd, offset, whence)`.
    #[allow(clippy::unused_self)]
    fn sys_lseek(&self, sh: &mut Shared, cx: &mut ServiceCtx, fd: u64, offset: i64, whence: u64) -> i64 {
        let (cur, path) = match cx.cur.fds.get(fd as i32) {
            Some(Fd::File { path, offset }) => (*offset, path.clone()),
            _ => return err(Errno::ESPIPE),
        };
        let size = sh.mounts.stat(&path).map_or(0, |a| a.size);
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
            *offset = newpos as u64;
        }
        newpos
    }

    /// `fstat(fd, statbuf)`.
    fn sys_fstat(&self, sh: &mut Shared, cx: &mut ServiceCtx, fd: u64, statbuf: u64, mem: &mut GuestMemory) -> i64 {
        let attrs = match cx.cur.fds.get(fd as i32) {
            Some(Fd::File { path, .. } | Fd::Dir { path, .. }) => {
                let path = path.clone();
                match sh.mounts.stat(&path) {
                    Some(a) => a,
                    None => return err(Errno::ENOENT),
                }
            }
            // eventfd/timerfd/epoll are anonymous-inode char-device-like fds.
            Some(
                Fd::Stdin
                | Fd::Stdout
                | Fd::Stderr
                | Fd::Eventfd(_)
                | Fd::Timerfd(_)
                | Fd::Epoll(_),
            ) => stat::char_device_attrs(),
            Some(Fd::PipeRead(_) | Fd::PipeWrite(_)) => stat::fifo_attrs(),
            Some(Fd::Socket { .. }) => stat::socket_attrs(),
            None => return err(Errno::EBADF),
        };
        write_stat_or_fault(mem, statbuf, &attrs, self.arch)
    }

    /// `newfstatat(dirfd, path, statbuf, flags)`.
    #[allow(clippy::too_many_arguments)]
    fn sys_newfstatat(
        &self, sh: &mut Shared, cx: &mut ServiceCtx,
        dirfd: i64,
        pathptr: u64,
        statbuf: u64,
        flags: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const AT_SYMLINK_NOFOLLOW: u64 = 0x100;
        let Some(rel) = read_path(mem, pathptr) else {
            return err(Errno::EFAULT);
        };
        let mut abs = self.resolve_path(cx, dirfd, &rel);
        if flags & AT_SYMLINK_NOFOLLOW == 0 {
            abs = self.follow_symlinks(sh, &abs).unwrap_or(abs);
        }
        let Some(attrs) = sh.mounts.stat(&abs) else {
            return err(Errno::ENOENT);
        };
        write_stat_or_fault(mem, statbuf, &attrs, self.arch)
    }

    /// `getdents64(fd, buf, count)`.
    #[allow(clippy::unused_self)]
    fn sys_getdents64(&self, sh: &mut Shared, cx: &mut ServiceCtx, fd: u64, buf: u64, count: u64, mem: &mut GuestMemory) -> i64 {
        let (path, pos) = match cx.cur.fds.get(fd as i32) {
            Some(Fd::Dir { path, pos }) => (path.clone(), *pos),
            _ => return err(Errno::ENOTDIR),
        };
        let entries = match sh.mounts.readdir(&path) {
            Ok(e) => e,
            Err(e) => return io_errno(&e),
        };
        let mut all: Vec<(String, NodeKind, u64)> = vec![
            (".".into(), NodeKind::Dir, 1),
            ("..".into(), NodeKind::Dir, 1),
        ];
        all.extend(entries.into_iter().map(|e| (e.name, e.kind, e.inode)));

        let (bytes, consumed) = stat::encode_dirents(&all, pos, count as usize);
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
    fn sys_chdir(&self, sh: &mut Shared, cx: &mut ServiceCtx, pathptr: u64, mem: &GuestMemory) -> i64 {
        let Some(rel) = read_path(mem, pathptr) else {
            return err(Errno::EFAULT);
        };
        let abs = self.resolve_path(cx, AT_FDCWD, &rel);
        match sh.mounts.stat(&abs) {
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
    fn sys_fchdir(&self, sh: &mut Shared, cx: &mut ServiceCtx, fd: u64) -> i64 {
        let path = match cx.cur.fds.get(fd as i32) {
            Some(Fd::Dir { path, .. }) => path.clone(),
            Some(_) => return err(Errno::ENOTDIR),
            None => return err(Errno::EBADF),
        };
        match sh.mounts.stat(&path) {
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

    /// Follow the final-component symlink chain (bounded), returning the target.
    #[allow(clippy::unused_self)]
    fn follow_symlinks(&self, sh: &mut Shared, path: &str) -> Option<String> {
        let mut p = path.to_string();
        for _ in 0..SYMLINK_MAX {
            match sh.mounts.stat(&p) {
                Some(a) if a.kind == NodeKind::Symlink => {
                    let target = sh.mounts.readlink(&p).ok()?;
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
    fn resolve_exec(&self, sh: &mut Shared, cx: &mut ServiceCtx, p: &str) -> Option<String> {
        let abs = self.resolve_path(cx, AT_FDCWD, p);
        self.follow_symlinks(sh, &abs)
    }

    /// Read an entire file from the mount table.
    #[allow(clippy::unused_self)]
    fn read_file(&self, sh: &mut Shared, path: &str) -> Option<Vec<u8>> {
        let size = sh.mounts.stat(path)?.size as usize;
        let mut buf = vec![0u8; size];
        let mut off = 0;
        while off < size {
            match sh.mounts.read_at(path, off as u64, &mut buf[off..]) {
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
    /// `MAP_FIXED` address). File-backed mappings additionally copy the file's
    /// bytes from `off` into the fresh, zero-filled region — the mechanism the
    /// dynamic linker uses to map `ld-musl` and the shared libraries. We give
    /// every file mapping private (copy) semantics: `MAP_SHARED` writes are not
    /// flushed back to the backing file (documented limitation), which is
    /// correct for the read-only/executable maps loaders create.
    #[allow(clippy::unused_self)]
    fn sys_mmap(&self, sh: &mut Shared, cx: &mut ServiceCtx, a: &[u64; 6], mem: &mut GuestMemory) -> i64 {
        const MAP_SHARED: u64 = 0x01;
        const MAP_FIXED: u64 = 0x10;
        const MAP_ANONYMOUS: u64 = 0x20;

        let (addr, len, prot, flags) = (a[0], a[1], a[2], a[3]);
        let (fd, offset) = (a[4], a[5]);
        if len == 0 {
            return err(Errno::EINVAL);
        }
        let len = len.div_ceil(PAGE_SIZE) * PAGE_SIZE;
        let prot = Prot((prot as u8) & 0x7);

        // For a file-backed mapping, resolve the backing path up front so a bad
        // fd fails before we disturb the address space.
        let file_src = if flags & MAP_ANONYMOUS == 0 {
            match cx.cur.fds.get(fd as i32) {
                Some(Fd::File { path, .. }) => Some(path.clone()),
                Some(_) => return err(Errno::EACCES), // mmap of pipe/socket/dir
                None => return err(Errno::EBADF),
            }
        } else {
            None
        };

        let base = if flags & MAP_FIXED != 0 && addr != 0 {
            addr - addr % PAGE_SIZE
        } else {
            let Some(base) = sh.arena(cx).alloc(len) else {
                return err(Errno::ENOMEM);
            };
            base
        };
        if mem.map(base, len, prot).is_err() {
            return err(Errno::ENOMEM);
        }

        if let Some(path) = file_src {
            // Fill the mapping from the file: a zero-initialized page-sized
            // buffer, with the file's bytes (from `offset`, up to EOF) copied
            // over the front; the tail past EOF stays zero, as mmap requires.
            let mut data = vec![0u8; len as usize];
            let mut got = 0usize;
            while got < data.len() {
                match sh
                    .mounts
                    .read_at(&path, offset + got as u64, &mut data[got..])
                {
                    Ok(n) if n > 0 => got += n,
                    _ => break, // EOF or read error: leave the rest zero-filled
                }
            }
            // write_init bypasses page protection, so a read/exec-only mapping
            // (the common code-segment case) is still populated correctly.
            if mem.write_init(base, &data).is_err() {
                return err(Errno::ENOMEM);
            }
            // A writable MAP_SHARED file mapping must have the guest's later
            // stores flushed back to the file (on munmap/msync/exit).
            if flags & MAP_SHARED != 0 && prot.contains(Prot::WRITE) {
                cx.cur.shared_maps.push(SharedMap {
                    base,
                    len,
                    path,
                    offset,
                });
            }
        }
        base as i64
    }

    /// Flush any writable `MAP_SHARED` file mappings overlapping `[addr, addr +
    /// len)` back to their backing files (their guest memory is the source of
    /// truth). `len == 0` flushes every shared mapping (process teardown).
    #[allow(clippy::unused_self)]
    fn flush_shared_maps(&self, sh: &mut Shared, cx: &mut ServiceCtx, addr: u64, len: u64, mem: &GuestMemory) {
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
            let file_size = sh.mounts.stat(&m.path).map_or(m.len, |a| a.size);
            let writable = file_size.saturating_sub(m.offset).min(m.len);
            if writable == 0 {
                continue;
            }
            if let Ok(bytes) = mem.read_vec(m.base, writable as usize) {
                let _ = sh.mounts.write_at(&m.path, m.offset, &bytes);
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
    fn sys_munmap(&self, sh: &mut Shared, cx: &mut ServiceCtx, addr: u64, len: u64, mem: &mut GuestMemory) -> i64 {
        if len == 0 {
            return err(Errno::EINVAL);
        }
        let len = len.div_ceil(PAGE_SIZE) * PAGE_SIZE;
        let base = addr - addr % PAGE_SIZE;
        // Flush any writable shared file mapping before the pages go away.
        if !cx.cur.shared_maps.is_empty() {
            self.flush_shared_maps(sh, cx, base, len, mem);
        }
        let _ = mem.unmap(base, len);
        // Give the range back to the arena so it can be handed out again — a
        // guest that cycles mappings (a JS engine's JIT/heap blocks) would
        // otherwise exhaust the arena while most of it sat free.
        sh.arena(cx).free_range(base, len);
        0
    }

    /// `msync(addr, len, flags)` — flush a writable shared file mapping to its
    /// file without unmapping it.
    fn sys_msync(&self, sh: &mut Shared, cx: &mut ServiceCtx, addr: u64, len: u64, mem: &GuestMemory) -> i64 {
        if !cx.cur.shared_maps.is_empty() {
            let len = len.div_ceil(PAGE_SIZE) * PAGE_SIZE;
            // msync flushes but keeps the mapping — re-add after the flush.
            let saved = cx.cur.shared_maps.clone();
            self.flush_shared_maps(sh, cx, addr - addr % PAGE_SIZE, len.max(PAGE_SIZE), mem);
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
    fn sys_getrandom(&self, sh: &mut Shared, buf: u64, len: u64, mem: &mut GuestMemory) -> i64 {
        if sh.rng_state == 0 {
            let now = match crate::clock::now_unix().as_nanos() as u64 {
                0 => 0x9E37_79B9_7F4A_7C15,
                n => n,
            };
            sh.rng_state = now | 1;
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
    fn sys_uname(&self, buf: u64, mem: &mut GuestMemory) -> i64 {
        const FIELD: usize = 65;
        let mut data = [0u8; FIELD * 6];
        let fields: [&[u8]; 6] = [
            b"Linux",
            b"nixvm",
            b"6.1.0-nixvm",
            b"#1 nixvm",
            self.arch.as_str().as_bytes(),
            b"(none)",
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
}

/// `clock_gettime(clk_id, timespec)`.
fn sys_clock_gettime(ts: u64, mem: &mut GuestMemory) -> i64 {
    let now = crate::clock::now_unix();
    let mut b = [0u8; 16];
    b[0..8].copy_from_slice(&(now.as_secs()).to_le_bytes());
    b[8..16].copy_from_slice(&u64::from(now.subsec_nanos()).to_le_bytes());
    match mem.write(ts, &b) {
        Ok(()) => 0,
        Err(_) => err(Errno::EFAULT),
    }
}

/// Encode an errno as a negative syscall return.
const fn err(e: Errno) -> i64 {
    -(e.0 as i64)
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
            call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Read, [0, buf, 16, 0, 0, 0]),
            0
        );
        assert!(cx.block, "read of empty interactive stdin blocks");

        // Feed input: the read now delivers it.
        k.feed_stdin(b"hi\n");
        cx.block = false;
        assert_eq!(
            call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Read, [0, buf, 16, 0, 0, 0]),
            3
        );
        assert_eq!(&mem.read_vec(buf, 3).unwrap(), b"hi\n");
        assert!(!cx.block);

        // Closed + empty: EOF (0), no block.
        k.close_stdin();
        cx.block = false;
        assert_eq!(
            call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Read, [0, buf, 16, 0, 0, 0]),
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
            k.shared.lock().unwrap().procs.iter().flatten().any(|p| p.info.pid == 2),
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
            NR_GETPID, NR_CLONE, NR_GETPID, NR_GETPID, NR_CLONE, NR_GETPID,
            NR_GETPID, NR_GETPID, NR_CLONE, NR_GETPID, NR_GETPID, NR_GETPID,
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
        kernel.shared.get_mut().unwrap().mmap_areas.push(Arena::new(0x1_8000, 0x1_5000));
        let mut mem = GuestMemory::new(0x1_0000, 16 * PAGE);
        mem.map(0x1_0000, 4 * PAGE, Prot::rw()).unwrap();
        (kernel, mem, DummyVcpu, cx)
    }

    fn call(
        k: &Kernel,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        mem: &mut GuestMemory,
        v: &mut DummyVcpu,
        s: Sysno,
        a: [u64; 6],
    ) -> i64 {
        k.dispatch(cx, s, 0, &a, v, mem, sh)
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
            &k, &mut k.shared.lock().unwrap(),
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
                &k, &mut k.shared.lock().unwrap(),
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [fd, msg, 2, 0, 0, 0]
            ),
            2
        );
        assert_eq!(
            call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Lseek, [fd, 0, 0, 0, 0, 0]),
            0
        );
        assert_eq!(
            call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Read, [fd, buf, 2, 0, 0, 0]),
            2
        );
        assert_eq!(mem.read_vec(buf, 2).unwrap(), b"Hi");

        let stbuf = 0x1_3000;
        assert_eq!(
            call(
                &k, &mut k.shared.lock().unwrap(),
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
            call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Close, [fd, 0, 0, 0, 0, 0]),
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
                &k, &mut k.shared.lock().unwrap(),
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
            call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Pipe2, [fds, 0, 0, 0, 0, 0]),
            0
        );
        let rfd = u64::from(mem.read_u32(fds).unwrap());
        let wfd = u64::from(mem.read_u32(fds + 4).unwrap());
        assert!(rfd >= 3 && wfd >= 3 && rfd != wfd);

        let msg = 0x1_1000;
        mem.write_init(msg, b"pipe!").unwrap();
        assert_eq!(
            call(
                &k, &mut k.shared.lock().unwrap(),
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [wfd, msg, 5, 0, 0, 0]
            ),
            5
        );

        let dfd = call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Dup, [rfd, 0, 0, 0, 0, 0]);
        assert!(dfd >= 3);
        let buf = 0x1_2000;
        assert_eq!(
            call(
                &k, &mut k.shared.lock().unwrap(),
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
                &k, &mut k.shared.lock().unwrap(),
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
            call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Read, [0, buf, 5, 0, 0, 0]),
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
                &k, &mut k.shared.lock().unwrap(),
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
            &k, &mut k.shared.lock().unwrap(),
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Clone,
            [0x11, 0, 0, 0, 0, 0],
        );
        assert_eq!(child, 2, "first child is pid 2");
        assert_eq!(k.shared.lock().unwrap().procs.len(), 1, "child pushed to the process table");

        // no zombie yet -> wait4 blocks
        let ws = 0x1_0000;
        call(
            &k, &mut k.shared.lock().unwrap(),
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Wait4,
            [child as u64, ws, 0, 0, 0, 0],
        );
        assert!(cx.block, "wait4 blocks while the child is alive");

        // make the child a zombie (exit code 7), then wait4 reaps it.
        if let Some(Some(p)) = k.shared.lock().unwrap().procs
            .iter_mut()
            .find(|s| s.as_ref().is_some_and(|p| p.info.pid == 2))
        {
            p.info.run = RunState::Zombie(7);
        }
        let reaped = call(
            &k, &mut k.shared.lock().unwrap(),
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
        k.shared.lock().unwrap().spaces.push(Arc::new(Mutex::new(mem.fork())));

        cx.cur.mm = 0;

        let child = call(
            &k, &mut k.shared.lock().unwrap(),
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Clone,
            [CLONE_VM | CLONE_VFORK, 0, 0, 0, 0, 0],
        );
        let cmm = k.shared.lock().unwrap().procs
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
            &k, &mut k.shared.lock().unwrap(),
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Clone,
            [CLONE_VM | CLONE_THREAD, 0, 0, 0, 0, 0],
        );
        let tmm = k.shared.lock().unwrap().procs
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
        assert_eq!(call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Getpid, [0; 6]), 1);
        assert_eq!(call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Gettid, [0; 6]), 7);
    }

    #[test]
    fn clone_thread_shares_tgid_and_address_space() {
        let (k, mut mem, mut v, mut cx) = setup();
        // CLONE_VM | CLONE_THREAD | CLONE_SETTLS
        let flags = 0x0000_0100 | 0x0001_0000 | 0x0008_0000;
        let tid = call(
            &k, &mut k.shared.lock().unwrap(),
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Clone,
            [flags, 0x2_0000, 0, 0xdead_beef, 0, 0],
        );
        assert_eq!(tid, 2, "new thread gets a fresh tid");
        let sh = k.shared.lock().unwrap();
        let spaces_before = sh.spaces.len();
        let child = sh.procs
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
        k.shared.lock().unwrap().spaces
            .push(Arc::new(Mutex::new(GuestMemory::new(0x1_0000, PAGE))));
        cx.cur.mm = 0;
        let before = k.shared.lock().unwrap().spaces.len();
        // flags = SIGCHLD only (a plain fork), no CLONE_VM.
        let child = call(
            &k, &mut k.shared.lock().unwrap(),
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
    fn exit_group_zombies_the_whole_thread_group() {
        let (k, mut mem, mut v, mut cx) = setup();
        // Two sibling threads in the leader's group, plus an unrelated process.
        k.shared.lock().unwrap().procs.push(Some(make_proc(2, 1, 0, true)));
        k.shared.lock().unwrap().procs.push(Some(make_proc(3, 1, 0, true)));
        k.shared.lock().unwrap().procs.push(Some(make_proc(4, 4, 1, false)));

        call(
            &k, &mut k.shared.lock().unwrap(),
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::ExitGroup,
            [42, 0, 0, 0, 0, 0],
        );

        assert!(matches!(cx.cur.run, RunState::Zombie(42)), "leader exits");
        let state = |pid| {
            k.shared.lock().unwrap().procs
                .iter()
                .flatten()
                .find(|p| p.info.pid == pid)
                .map(|p| p.info.run)
        };
        assert_eq!(
            state(2),
            Some(RunState::Zombie(42)),
            "sibling thread killed"
        );
        assert_eq!(
            state(3),
            Some(RunState::Zombie(42)),
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
        waiter.info.futex_wait = Some((0, uaddr));
        k.shared.lock().unwrap().procs.push(Some(waiter));

        // FUTEX_WAKE(uaddr, op=1, val=1) wakes exactly one waiter.
        let woken = call(
            &k, &mut k.shared.lock().unwrap(),
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
    fn futex_wait_single_thread_does_not_deadlock() {
        let (k, mut mem, mut v, mut cx) = setup();
        let uaddr = 0x1_0000;
        mem.write_init(uaddr, &42u32.to_le_bytes()).unwrap();
        // Value matches and no other task could wake us: report a spurious wake
        // rather than parking (which would be a false deadlock).
        let r = call(
            &k, &mut k.shared.lock().unwrap(),
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
            &k, &mut k.shared.lock().unwrap(),
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
        k.shared.lock().unwrap().procs.push(Some(make_proc(2, 1, 0, true)));
        let r = call(
            &k, &mut k.shared.lock().unwrap(),
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Futex,
            [uaddr, 0, 42, 0, 0, 0],
        );
        assert_eq!(r, 0);
        assert!(cx.block, "caller parks awaiting a wake");
        assert_eq!(cx.cur.futex_wait, Some((0, uaddr)));
    }

    #[test]
    fn mmap_file_backed_copies_file_contents() {
        const MAP_FIXED: u64 = 0x10;
        const PROT_READ: u64 = 0x1;
        let (k, mut mem, mut v, mut cx) = setup();
        let path = 0x1_0000;
        let content = 0x1_1000;
        mem.write_init(path, b"/lib\0").unwrap();
        mem.write_init(content, &[0x11, 0x22, 0x33, 0x44]).unwrap();

        // Create /lib and write four bytes to it.
        let fd = call(
            &k, &mut k.shared.lock().unwrap(),
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_CWD, path, 0o102, 0o644, 0, 0],
        );
        assert_eq!(fd, 3);
        assert_eq!(
            call(
                &k, &mut k.shared.lock().unwrap(),
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
            &k, &mut k.shared.lock().unwrap(),
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
        const MAP_FIXED: u64 = 0x10;
        let (k, mut mem, mut v, mut cx) = setup();
        let path = 0x1_0000;
        let content = 0x1_1000;
        mem.write_init(path, b"/x\0").unwrap();
        mem.write_init(content, &[0xAB, 0xCD]).unwrap();
        // Pre-dirty the target page so we can prove the tail is zeroed.
        mem.write(0x1_3000, &[0xFF; 8]).unwrap();
        let fd = call(
            &k, &mut k.shared.lock().unwrap(),
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_CWD, path, 0o102, 0o644, 0, 0],
        );
        call(
            &k, &mut k.shared.lock().unwrap(),
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Write,
            [fd as u64, content, 2, 0, 0, 0],
        );
        let addr = 0x1_3000u64;
        call(
            &k, &mut k.shared.lock().unwrap(),
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
        const MAP_FIXED: u64 = 0x10;
        let (k, mut mem, mut v, mut cx) = setup();
        // No such fd -> EBADF.
        assert_eq!(
            call(
                &k, &mut k.shared.lock().unwrap(),
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
                &k, &mut k.shared.lock().unwrap(),
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
            &k, &mut k.shared.lock().unwrap(),
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
                &k, &mut k.shared.lock().unwrap(),
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
                &k, &mut k.shared.lock().unwrap(),
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
                &k, &mut k.shared.lock().unwrap(),
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

        // nanosleep with a valid req returns 0 and writes rem = {0, 0}.
        let req = 0x1_2000;
        let rem = 0x1_2100;
        mem.write_init(req, &0u64.to_le_bytes()).unwrap();
        mem.write_init(req + 8, &500u64.to_le_bytes()).unwrap();
        mem.write_init(rem, &7u64.to_le_bytes()).unwrap();
        mem.write_init(rem + 8, &7u64.to_le_bytes()).unwrap();
        assert_eq!(
            call(
                &k, &mut k.shared.lock().unwrap(),
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Nanosleep,
                [req, rem, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_u64(rem).unwrap(), 0);
        assert_eq!(mem.read_u64(rem + 8).unwrap(), 0);

        // nanosleep with tv_nsec >= 1e9 returns -EINVAL.
        mem.write_init(req + 8, &1_000_000_000u64.to_le_bytes())
            .unwrap();
        assert_eq!(
            call(
                &k, &mut k.shared.lock().unwrap(),
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
    fn getdents_and_getcwd() {
        let (k, mut mem, mut v, mut cx) = setup();
        let path = 0x1_0000;
        mem.write_init(path, b"/a\0").unwrap();
        call(
            &k, &mut k.shared.lock().unwrap(),
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_CWD, path, 0o102, 0o644, 0, 0],
        );

        let root = 0x1_1000;
        mem.write_init(root, b"/\0").unwrap();
        let dirfd = call(
            &k, &mut k.shared.lock().unwrap(),
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_CWD, root, 0, 0, 0, 0],
        );
        let buf = 0x1_2000;
        let n = call(
            &k, &mut k.shared.lock().unwrap(),
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Getdents64,
            [dirfd as u64, buf, PAGE, 0, 0, 0],
        );
        assert!(n > 0);

        let cbuf = 0x1_3000;
        let len = call(
            &k, &mut k.shared.lock().unwrap(),
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
        let (orig_pc, orig_sp) = (vcpu.pc(), vcpu.sp());

        let (k, mut mem, _v, mut cx) = setup();
        mem.map(0x1_0000, 4 * PAGE, Prot::rw()).unwrap();
        cx.cur.mm = 0;
        cx.cur.handlers[11] = SigAction { handler: 0x2_0000, flags: 0, restorer: 0x2_1000, mask: 0 };

        // Deliver SIGSEGV (fault addr 0xcafe) → the vcpu enters the handler.
        assert!(k.deliver_fault_signal(&mut cx, 11, 0xcafe, vcpu.as_mut(), &mut mem));
        assert_eq!(vcpu.pc(), 0x2_0000, "pc → handler");
        assert_eq!(vcpu.reg(7), 11, "rdi = signum");
        let frame = vcpu.sp();
        assert_eq!(vcpu.reg(2), frame + 8, "rdx = &ucontext");
        assert_eq!(vcpu.reg(6), frame + 8 + super::signal::signal_ucontext_size(), "rsi = &siginfo");
        assert_eq!(mem.read_u64(frame).unwrap(), 0x2_1000, "pretcode = restorer");
        assert_eq!(cx.cur.blocked & (1 << 10), 1 << 10, "SIGSEGV blocked in handler");

        // The handler clobbers rbx; rt_sigreturn must restore it.
        vcpu.set_reg(3, 0);
        vcpu.set_sp(frame + 8); // as if the restorer's `ret` popped pretcode
        k.sys_rt_sigreturn(&mut cx, vcpu.as_mut(), &mem);
        assert_eq!(vcpu.pc(), orig_pc, "pc restored");
        assert_eq!(vcpu.sp(), orig_sp, "rsp restored");
        assert_eq!(vcpu.reg(3), 0xdead, "rbx restored");
        assert_eq!(vcpu.reg(0), 0x1234, "rax restored");
        assert_eq!(cx.cur.blocked, 0, "signal mask restored");
    }

    #[test]
    fn fault_with_no_handler_is_not_delivered() {
        use crate::vcpu::Backend;
        let backend = crate::vcpu::interp_x86::X86Backend::new(Arch::X86_64).unwrap();
        let mut vcpu = backend.new_vcpu(0x1_1111, 0x1_3000).unwrap();
        let (k, mut mem, _v, mut cx) = setup();
        // SIG_DFL for SIGSEGV: not deliverable (stays a fatal fault).
        assert!(!k.deliver_fault_signal(&mut cx, 11, 0, vcpu.as_mut(), &mut mem));
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
                &k, &mut k.shared.lock().unwrap(),
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
                &k, &mut k.shared.lock().unwrap(),
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
                &k, &mut k.shared.lock().unwrap(),
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
                &k, &mut k.shared.lock().unwrap(),
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
                &k, &mut k.shared.lock().unwrap(),
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
                &k, &mut k.shared.lock().unwrap(),
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
            call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Kill, [1, 15, 0, 0, 0, 0]),
            0
        );
        assert_eq!(cx.cur.pending, 1 << 14);

        // Default disposition of SIGTERM is TERMINATE -> exit code 128 + 15.
        k.deliver_pending_signals(&mut cx);
        assert!(matches!(cx.cur.run, RunState::Zombie(143)));
    }

    #[test]
    fn kill_nonexistent_pid_is_esrch() {
        let (k, mut mem, mut v, mut cx) = setup();
        assert_eq!(
            call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Kill, [999, 15, 0, 0, 0, 0]),
            -3
        );
    }

    /// Open a file, seed it, and return its fd — for the I/O syscall tests.
    fn open_seeded(k: &mut Kernel, cx: &mut ServiceCtx, mem: &mut GuestMemory, v: &mut DummyVcpu, content: &[u8]) -> u64 {
        let path = 0x1_0000;
        mem.write_init(path, b"/f\0").unwrap();
        let fd = call(k, &mut k.shared.lock().unwrap(), cx, mem, v, Sysno::Openat, [AT_CWD, path, 0o102, 0o644, 0, 0]) as u64;
        let src = 0x1_3000;
        mem.write_init(src, content).unwrap();
        call(k, &mut k.shared.lock().unwrap(), cx, mem, v, Sysno::Write, [fd, src, content.len() as u64, 0, 0, 0]);
        fd
    }

    #[test]
    fn pread_pwrite_do_not_move_the_offset() {
        let (mut k, mut mem, mut v, mut cx) = setup();
        let fd = open_seeded(&mut k, &mut cx, &mut mem, &mut v, b"0123456789");
        // Read the fd position back to 0 via lseek, then pread at offset 4.
        call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Lseek, [fd, 0, 0, 0, 0, 0]);
        let buf = 0x1_2000;
        let n = call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Pread64, [fd, buf, 3, 4, 0, 0]);
        assert_eq!(n, 3);
        assert_eq!(mem.read_vec(buf, 3).unwrap(), b"456");
        // The fd position is still 0, so a plain read starts at the beginning.
        let n = call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Read, [fd, buf, 2, 0, 0, 0]);
        assert_eq!(n, 2);
        assert_eq!(mem.read_vec(buf, 2).unwrap(), b"01");
        // pwrite at offset 4 overwrites in place, again without moving the pos.
        let src = 0x1_1000;
        mem.write_init(src, b"XY").unwrap();
        call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Pwrite64, [fd, src, 2, 4, 0, 0]);
        call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Lseek, [fd, 0, 0, 0, 0, 0]);
        call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Read, [fd, buf, 10, 0, 0, 0]);
        assert_eq!(mem.read_vec(buf, 10).unwrap(), b"0123XY6789");
    }

    #[test]
    fn ftruncate_and_truncate_resize() {
        let (mut k, mut mem, mut v, mut cx) = setup();
        let fd = open_seeded(&mut k, &mut cx, &mut mem, &mut v, b"abcdef");
        assert_eq!(call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Ftruncate, [fd, 3, 0, 0, 0, 0]), 0);
        assert_eq!(k.shared.lock().unwrap().mounts.stat("/f").unwrap().size, 3);
        // truncate by path can also grow (zero-extend).
        let path = 0x1_0000;
        assert_eq!(call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Truncate, [path, 8, 0, 0, 0, 0]), 0);
        assert_eq!(k.shared.lock().unwrap().mounts.stat("/f").unwrap().size, 8);
    }

    #[test]
    fn statx_reports_size_and_mode() {
        let (mut k, mut mem, mut v, mut cx) = setup();
        open_seeded(&mut k, &mut cx, &mut mem, &mut v, b"hello world");
        let path = 0x1_0000;
        let buf = 0x1_2000;
        assert_eq!(
            call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Statx, [AT_CWD, path, 0, 0x7ff, buf, 0]),
            0
        );
        // stx_size @40, stx_mode @28.
        assert_eq!(u64::from_le_bytes(mem.read_vec(buf + 40, 8).unwrap().try_into().unwrap()), 11);
        let mode = u16::from_le_bytes(mem.read_vec(buf + 28, 2).unwrap().try_into().unwrap());
        assert_eq!(mode & 0o170000, 0o100000, "S_IFREG");
    }

    #[test]
    fn sendfile_copies_between_files() {
        let (mut k, mut mem, mut v, mut cx) = setup();
        let infd = open_seeded(&mut k, &mut cx, &mut mem, &mut v, b"payload!");
        call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Lseek, [infd, 0, 0, 0, 0, 0]);
        // A second file as the destination.
        let path2 = 0x1_1000;
        mem.write_init(path2, b"/g\0").unwrap();
        let outfd = call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Openat, [AT_CWD, path2, 0o102, 0o644, 0, 0]) as u64;
        let n = call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Sendfile, [outfd, infd, 0, 8, 0, 0]);
        assert_eq!(n, 8);
        assert_eq!(k.shared.lock().unwrap().mounts.stat("/g").unwrap().size, 8);
        let buf = 0x1_2000;
        call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Lseek, [outfd, 0, 0, 0, 0, 0]);
        call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Read, [outfd, buf, 8, 0, 0, 0]);
        assert_eq!(mem.read_vec(buf, 8).unwrap(), b"payload!");
    }

    #[test]
    fn session_and_pgid_tracking() {
        let (k, mut mem, mut v, mut cx) = setup();
        cx.cur.pid = 5;
        // getpgid(0) defaults to the pid.
        assert_eq!(call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Getpgid, [0; 6]), 5);
        // setpgid(0, 42) sets it.
        assert_eq!(call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Setpgid, [0, 42, 0, 0, 0, 0]), 0);
        assert_eq!(call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Getpgid, [0; 6]), 42);
        assert_eq!(call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Getpgrp, [0; 6]), 42);
        // setsid starts a new session: sid = pgid = pid.
        assert_eq!(call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Setsid, [0; 6]), 5);
        assert_eq!(call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Getsid, [0; 6]), 5);
        assert_eq!(call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Getpgid, [0; 6]), 5);
    }

    #[test]
    fn memfd_create_is_a_readwrite_fd() {
        let (k, mut mem, mut v, mut cx) = setup();
        let name = 0x1_0000;
        mem.write_init(name, b"scratch\0").unwrap();
        let fd = call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::MemfdCreate, [name, 0, 0, 0, 0, 0]);
        assert!(fd >= 3, "a real fd");
        let fd = fd as u64;
        let src = 0x1_2000;
        mem.write_init(src, b"data").unwrap();
        assert_eq!(call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Write, [fd, src, 4, 0, 0, 0]), 4);
        call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Lseek, [fd, 0, 0, 0, 0, 0]);
        let buf = 0x1_3000;
        assert_eq!(call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Read, [fd, buf, 4, 0, 0, 0]), 4);
        assert_eq!(mem.read_vec(buf, 4).unwrap(), b"data");
    }

    #[test]
    fn close_range_closes_fds() {
        let (mut k, mut mem, mut v, mut cx) = setup();
        let fd = open_seeded(&mut k, &mut cx, &mut mem, &mut v, b"x");
        assert!(fd >= 3);
        // Close everything from `fd` up; a subsequent op on it is EBADF.
        assert_eq!(call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::CloseRange, [fd, u64::from(u32::MAX), 0, 0, 0, 0]), 0);
        let buf = 0x1_2000;
        assert_eq!(call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Read, [fd, buf, 1, 0, 0, 0]), -9); // EBADF
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
        assert_eq!(call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Ftruncate, [fd, 6, 0, 0, 0, 0]), 0);
        // mmap(NULL, 4096, PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0).
        let base = call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Mmap, [0, 4096, 0x3, 0x1, fd, 0]);
        assert!(base > 0, "mmap returned {base}");
        let base = base as u64;
        // Store "hello!" into the mapping (as a guest memcpy would).
        mem.write(base, b"hello!").unwrap();
        // munmap flushes it back to the file.
        assert_eq!(call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Munmap, [base, 4096, 0, 0, 0, 0]), 0);
        // Read the file: it now holds the mapped bytes, not zeros.
        call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Lseek, [fd, 0, 0, 0, 0, 0]);
        let buf = 0x1_2000;
        assert_eq!(call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Read, [fd, buf, 6, 0, 0, 0]), 6);
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


        let a = call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Mmap, [0, 4096, 0x3, 0x22, u64::MAX, 0]);
        let b = call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Mmap, [0, 4096, 0x3, 0x22, u64::MAX, 0]);
        assert!(a > 0 && b > 0, "mmaps returned {a}, {b}");
        let (a, b) = (a as u64, b as u64);
        assert!(a.abs_diff(b) >= 4096, "sibling mmaps overlap: A={a:#x} B={b:#x}");
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

        let a = call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Mmap, anon);
        let b = call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Mmap, anon);
        let c = call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Mmap, anon);
        assert!(a > 0 && b > 0 && c > 0);
        // Arena is now full: a fourth fails.
        assert_eq!(call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Mmap, anon), -12); // ENOMEM

        // Free the middle one and the next mmap must reuse exactly that page,
        // rather than reporting the arena exhausted.
        assert_eq!(
            call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Munmap, [b as u64, 4096, 0, 0, 0, 0]),
            0
        );
        let reused = call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Mmap, anon);
        assert_eq!(reused, b, "munmap'd page must be handed out again");

        // Freeing all three coalesces back into one contiguous run, so a
        // 3-page mmap fits again.
        for p in [a, b, c] {
            call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Munmap, [p as u64, 4096, 0, 0, 0, 0]);
        }
        let big = call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Mmap, [0, 3 * 4096, 0x3, 0x22, u64::MAX, 0]);
        assert!(big > 0, "coalesced free space must satisfy a 3-page mmap, got {big}");
    }

    #[test]
    fn prlimit_nofile_is_tracked_and_hard_capped() {
        const NOFILE: u64 = 7;
        let (k, mut mem, mut v, mut cx) = setup();
        let buf = 0x1_2000;
        // getrlimit reports the default (1024, 4096).
        assert_eq!(call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Getrlimit, [NOFILE, buf, 0, 0, 0, 0]), 0);
        assert_eq!(mem.read_u64(buf).unwrap(), 1024);
        assert_eq!(mem.read_u64(buf + 8).unwrap(), 4096);
        // Try to raise both soft and hard to a million (node/V8's binary
        // search). The hard limit is capped, and the soft is clamped to it.
        let newl = 0x1_2100;
        mem.write(newl, &1_048_576u64.to_le_bytes()).unwrap();
        mem.write(newl + 8, &1_048_576u64.to_le_bytes()).unwrap();
        // prlimit64(pid=0, NOFILE, new_limit, old_limit=0)
        assert_eq!(call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Prlimit64, [0, NOFILE, newl, 0, 0, 0]), 0);
        // getrlimit now reports the capped values, not a million.
        assert_eq!(call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Getrlimit, [NOFILE, buf, 0, 0, 0, 0]), 0);
        assert_eq!(mem.read_u64(buf).unwrap(), 4096, "soft clamped to the hard cap");
        assert_eq!(mem.read_u64(buf + 8).unwrap(), 4096, "hard capped");
    }

    #[test]
    fn fcntl_on_a_closed_fd_is_ebadf() {
        let (k, mut mem, mut v, mut cx) = setup();
        // F_SETFD (2) on an unopened fd must fail — else a "cloexec every fd
        // until EBADF" loop never terminates.
        assert_eq!(call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Fcntl, [99, 2, 1, 0, 0, 0]), -9);
    }

    #[test]
    fn credential_setters_succeed_as_root() {
        let (k, mut mem, mut v, mut cx) = setup();
        for s in [Sysno::Setuid, Sysno::Setgid, Sysno::Setresuid, Sysno::Setgroups] {
            assert_eq!(call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, s, [0; 6]), 0, "{s:?}");
        }
        // getresuid writes (0,0,0).
        let (a, b, c) = (0x1_2000, 0x1_2010, 0x1_2020);
        assert_eq!(call(&k, &mut k.shared.lock().unwrap(), &mut cx, &mut mem, &mut v, Sysno::Getresuid, [a, b, c, 0, 0, 0]), 0);
        assert_eq!(mem.read_u32(a).unwrap(), 0);
    }
}
