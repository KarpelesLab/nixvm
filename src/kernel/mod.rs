//! The nixvm "kernel": an arch-agnostic engine that services guest syscalls
//! and schedules multiple guest processes.
//!
//! State is split between **global** kernel state (mount table, pipes, stdio,
//! process table) and **per-process** state ([`ProcInfo`]: fds, cwd, brk, mmap
//! arena, pid). The currently-running process's [`ProcInfo`] is swapped into
//! `self.cur` while it runs, so the syscall handlers read/write `self.cur.*`
//! for per-process state and `self.*` for globals — no per-handler `Process`
//! threading. The scheduler ([`Kernel::run`]) is a cooperative round-robin over
//! [`Process`]es; a syscall that would block re-traps later (we simply don't
//! advance the guest PC), which the interpreter turns back into the same
//! syscall on the next slice.

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
    /// `set_tid_address` / `CLONE_CHILD_CLEARTID`: on exit, zero this guest
    /// word and futex-wake it (lets `pthread_join` return). 0 = unset.
    clear_child_tid: u64,
    /// When `Some((mm, uaddr))`, this task is parked in `FUTEX_WAIT` on that
    /// address; cleared when woken.
    futex_wait: Option<(usize, u64)>,
    /// Set by `FUTEX_WAKE` to release a parked waiter on its next slice.
    futex_woken: bool,
    run: RunState,
    /// Per-signal disposition: `SIG_DFL` (0), `SIG_IGN` (1), or a handler
    /// address. Indexed by signal number (1..=64); index 0 is unused.
    handlers: [u64; 65],
    /// Blocked-signal mask (bit `sig-1` set = blocked).
    blocked: u64,
    /// Pending-signal mask (bit `sig-1` set = pending).
    pending: u64,
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
            pid: 0,
            ppid: 0,
            tgid: 0,
            is_thread: false,
            mm: 0,
            clear_child_tid: 0,
            futex_wait: None,
            futex_woken: false,
            run: RunState::Running,
            handlers: [0; 65],
            blocked: 0,
            pending: 0,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RunState {
    Running,
    Zombie(i32),
}

/// The result of servicing one guest exit, telling the scheduler what to do
/// with the task's vcpu next.
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

/// The kernel: global state plus the process table and scheduler.
pub struct Kernel {
    arch: Arch,
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
    rng_state: u64,
    trace: bool,
    /// The process file-creation mask (`umask`); global for our single session.
    umask: u32,
    /// The process name (`prctl(PR_SET_NAME)`); a fixed 16-byte, NUL-padded field.
    procname: [u8; 16],
    unsupported: BTreeMap<u64, u64>,
    /// The running process's per-process state (swapped in for the slice).
    cur: ProcInfo,
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
    /// Number of virtual CPUs: how many host worker threads run guest compute
    /// in parallel. `1` uses the single-threaded cooperative scheduler.
    ncpus: usize,
    next_pid: i32,
    /// Set by a handler when the syscall would block (re-trap it later).
    block: bool,
    /// Set by `execve` when it replaced the process image (resume at the new
    /// PC without setting a syscall return).
    exec_ok: bool,
}

impl std::fmt::Debug for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernel")
            .field("arch", &self.arch)
            .field("procs", &self.procs.len())
            .field("unsupported", &self.unsupported)
            .finish_non_exhaustive()
    }
}

impl Kernel {
    #[must_use]
    pub fn new(arch: Arch, mounts: MountTable) -> Self {
        Self {
            arch,
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
            trace: std::env::var_os("NIXVM_TRACE").is_some(),
            umask: 0o022,
            procname: [0u8; 16],
            unsupported: BTreeMap::new(),
            cur: ProcInfo::default(),
            procs: Vec::new(),
            spaces: Vec::new(),
            ncpus: 1,
            next_pid: 2,
            block: false,
            exec_ok: false,
        }
    }

    /// Redirect the sink backing guest fd 1 (`stdout`).
    pub fn set_stdout(&mut self, w: Box<dyn Write + Send>) {
        self.stdout = w;
    }
    /// Redirect the sink backing guest fd 2 (`stderr`).
    pub fn set_stderr(&mut self, w: Box<dyn Write + Send>) {
        self.stderr = w;
    }
    /// Redirect the source backing guest fd 0 (`stdin`).
    pub fn set_stdin(&mut self, r: Box<dyn Read + Send>) {
        self.stdin = r;
    }

    /// Set the initial heap window for the first process: `start` is the program
    /// break, `limit` the highest address the heap may reach.
    pub fn set_heap(&mut self, start: u64, limit: u64) {
        self.cur.heap_start = start;
        self.cur.brk = start;
        self.cur.heap_limit = limit;
    }

    /// Set the initial anonymous-`mmap` arena for the first process.
    pub fn set_mmap_area(&mut self, top: u64, floor: u64) {
        self.cur.mmap_cursor = top;
        self.cur.mmap_floor = floor;
    }

    /// Set the first process's current working directory.
    pub fn set_cwd(&mut self, dir: impl Into<String>) {
        self.cur.cwd = path::normalize(&dir.into());
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
        let mut info = std::mem::take(&mut self.cur);
        info.pid = 1;
        info.ppid = 0;
        info.tgid = 1;
        info.mm = self.spaces.len();
        info.run = RunState::Running;
        self.spaces.push(Arc::new(Mutex::new(mem)));
        self.procs.push(Some(Process {
            vcpu: Some(vcpu),
            info,
        }));
        if self.ncpus > 1 {
            self.schedule_smp()
        } else {
            self.schedule_serial()
        }
    }

    /// Cooperative single-CPU round-robin scheduler.
    fn schedule_serial(&mut self) -> Result<i32, VcpuError> {
        loop {
            if let Some(code) = self.pid1_code() {
                return Ok(code);
            }
            let mut progressed = false;
            for i in 0..self.procs.len() {
                let runnable = matches!(
                    self.procs.get(i),
                    Some(Some(p)) if p.info.run == RunState::Running
                );
                if !runnable {
                    continue;
                }
                let mut proc = self.procs[i].take().unwrap();
                let mm = proc.info.mm;
                let mut vcpu = proc.vcpu.take().expect("runnable task has a vcpu");
                let space_arc = Arc::clone(&self.spaces[mm]);
                let mut guard = space_arc.lock().unwrap();
                std::mem::swap(&mut self.cur, &mut proc.info);
                let made = self.run_slice(&mut vcpu, &mut guard)?;
                std::mem::swap(&mut self.cur, &mut proc.info);
                drop(guard);
                proc.vcpu = Some(vcpu);
                self.procs[i] = Some(proc);
                progressed |= made;
            }
            if !progressed {
                if self.any_running() {
                    return Err(VcpuError::Backend(
                        "deadlock: every process is blocked".into(),
                    ));
                }
                return Ok(self.pid1_code().unwrap_or(0));
            }
        }
    }

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

    /// Run one process until it blocks or exits. Returns whether it made
    /// progress (completed at least one syscall, or exited).
    fn run_slice(
        &mut self,
        vcpu: &mut Box<dyn Vcpu>,
        mem: &mut GuestMemory,
    ) -> Result<bool, VcpuError> {
        let mut progressed = false;
        loop {
            let exit = vcpu.run(mem)?;
            match self.service(exit, vcpu.as_mut(), mem) {
                Serviced::SetRet(ret) => {
                    vcpu.set_syscall_ret(ret as u64);
                    progressed = true;
                }
                Serviced::Resume => progressed = true,
                Serviced::Blocked => return Ok(progressed),
                Serviced::Ended => return Ok(true),
            }
        }
    }

    /// Service one guest exit against the current task (`self.cur`): dispatch a
    /// syscall, or turn a fault/halt into a zombie. Shared by the serial and
    /// SMP schedulers. Does NOT touch the vcpu's result register — the caller
    /// applies [`Serviced::SetRet`] — so the same logic works whether the vcpu
    /// lives on the main thread or is round-tripping through a worker.
    fn service(&mut self, exit: Exit, vcpu: &mut dyn Vcpu, mem: &mut GuestMemory) -> Serviced {
        match exit {
            Exit::Syscall => {
                let raw = vcpu.syscall_nr();
                let sys = arch::decode(self.arch, raw);
                let args = vcpu.syscall_args();
                self.block = false;
                self.exec_ok = false;
                let ret = self.dispatch(sys, raw, &args, vcpu, mem);
                self.deliver_pending_signals();
                if let RunState::Zombie(_) = self.cur.run {
                    Serviced::Ended
                } else if self.block {
                    Serviced::Blocked
                } else if self.exec_ok {
                    Serviced::Resume // resume the new image at its entry
                } else {
                    Serviced::SetRet(ret)
                }
            }
            Exit::Interrupted => Serviced::Resume,
            Exit::MemFault { addr, write } => {
                eprintln!(
                    "[fault] pid {} memory fault at {addr:#x} (write={write})",
                    self.cur.pid
                );
                self.cur.run = RunState::Zombie(139);
                Serviced::Ended
            }
            Exit::IllegalInstruction { pc } => {
                eprintln!("[fault] pid {} illegal instruction at {pc:#x}", self.cur.pid);
                self.cur.run = RunState::Zombie(132);
                Serviced::Ended
            }
            Exit::Halt => {
                self.cur.run = RunState::Zombie(0);
                Serviced::Ended
            }
        }
    }

    /// SMP scheduler: a pool of `ncpus` host worker threads run guest compute
    /// (`vcpu.run`) in parallel, while this main thread services every syscall
    /// serially. Only the `Box<dyn Vcpu>` and the task's `Arc<Mutex<GuestMemory>>`
    /// cross a thread boundary; the `Kernel` (mounts, pipes, process table)
    /// stays here and needs no locking — scheduling and syscall servicing are
    /// single-threaded, so the whole design is race-free by construction. This
    /// is the big-kernel-lock model a KVM/HVF backend will slot into: vCPUs run
    /// in parallel, exits are serviced on one thread.
    #[allow(clippy::too_many_lines)] // the worker pool + service loop reads best as one unit
    fn schedule_smp(&mut self) -> Result<i32, VcpuError> {
        // Work handed to a worker: run this vcpu on this address space until it
        // next exits. `Stop` drains the pool at shutdown.
        enum Work {
            Run(usize, Box<dyn Vcpu>, Arc<Mutex<GuestMemory>>),
            Stop,
        }
        type Done = (usize, Box<dyn Vcpu>, Result<Exit, VcpuError>);

        let queue: Arc<(Mutex<VecDeque<Work>>, Condvar)> =
            Arc::new((Mutex::new(VecDeque::new()), Condvar::new()));
        let (done_tx, done_rx) = mpsc::channel::<Done>();
        let mut workers = Vec::with_capacity(self.ncpus);
        for _ in 0..self.ncpus {
            let q = Arc::clone(&queue);
            let out = done_tx.clone();
            workers.push(std::thread::spawn(move || {
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
                        Work::Run(id, mut vcpu, space) => {
                            let exit = {
                                let mut mem = space.lock().unwrap();
                                vcpu.run(&mut mem)
                            };
                            if out.send((id, vcpu, exit)).is_err() {
                                break;
                            }
                        }
                    }
                }
            }));
        }
        drop(done_tx);

        let push = |w: Work| {
            let (lock, cv) = &*queue;
            lock.lock().unwrap().push_back(w);
            cv.notify_one();
        };

        // A task that blocked records the progress epoch at which it did; it is
        // not re-dispatched until the epoch advances (some other task made real
        // progress that might satisfy its wait) — avoiding a busy spin.
        let mut blocked_at: BTreeMap<usize, u64> = BTreeMap::new();
        let mut epoch: u64 = 0;
        let mut inflight = 0usize;
        let outcome = loop {
            if let Some(code) = self.pid1_code() {
                break Ok(code);
            }
            // Fill idle workers with runnable tasks.
            while inflight < self.ncpus {
                let Some(i) = self.pick_smp_runnable(&blocked_at, epoch) else {
                    break;
                };
                let mm = self.procs[i].as_ref().unwrap().info.mm;
                let space = Arc::clone(&self.spaces[mm]);
                let vcpu = self.procs[i].as_mut().unwrap().vcpu.take().unwrap();
                push(Work::Run(i, vcpu, space));
                inflight += 1;
            }
            if inflight == 0 {
                // Nothing runnable and nothing in flight: either everyone is
                // blocked (deadlock) or done.
                break if self.any_running() {
                    Err(VcpuError::Backend("deadlock: every task is blocked".into()))
                } else {
                    Ok(self.pid1_code().unwrap_or(0))
                };
            }
            let (i, vcpu, exit_res) = done_rx.recv().expect("workers outlive the scheduler");
            inflight -= 1;
            let exit = match exit_res {
                Ok(e) => e,
                Err(e) => break Err(e),
            };
            let mm = self.procs[i].as_ref().unwrap().info.mm;
            let space = Arc::clone(&self.spaces[mm]);
            let mut vcpu = vcpu;
            let flow = {
                let mut proc = self.procs[i].take().unwrap();
                std::mem::swap(&mut self.cur, &mut proc.info);
                let flow = {
                    let mut mem = space.lock().unwrap();
                    self.service(exit, vcpu.as_mut(), &mut mem)
                };
                std::mem::swap(&mut self.cur, &mut proc.info);
                self.procs[i] = Some(proc);
                flow
            };
            match flow {
                Serviced::SetRet(ret) => {
                    vcpu.set_syscall_ret(ret as u64);
                    self.procs[i].as_mut().unwrap().vcpu = Some(vcpu);
                    epoch += 1; // real progress: wake blocked waiters to retry
                }
                Serviced::Resume => {
                    self.procs[i].as_mut().unwrap().vcpu = Some(vcpu);
                    epoch += 1;
                }
                Serviced::Blocked => {
                    self.procs[i].as_mut().unwrap().vcpu = Some(vcpu);
                    blocked_at.insert(i, epoch);
                }
                Serviced::Ended => {
                    // Task became a zombie; keep the (now-idle) vcpu slot empty.
                    self.procs[i].as_mut().unwrap().vcpu = Some(vcpu);
                    epoch += 1;
                }
            }
        };

        for _ in 0..self.ncpus {
            push(Work::Stop);
        }
        for h in workers {
            let _ = h.join();
        }
        outcome
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

    /// The syscall table. Returns the value the guest sees in its result
    /// register: a non-negative result, or a negative errno.
    #[allow(clippy::too_many_lines)] // one arm per syscall; a flat table is clearest.
    fn dispatch(
        &mut self,
        sys: Sysno,
        raw: u64,
        args: &[u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        if self.trace {
            eprintln!(
                "[trace] pid={} pc={:#x} {sys:?} raw={raw} args={args:x?}",
                self.cur.pid,
                vcpu.pc()
            );
        }
        match sys {
            Sysno::Write => self.sys_write(args[0], args[1], args[2], mem),
            Sysno::Read => self.sys_read(args[0], args[1], args[2], mem),
            // sendto/recvfrom carry an optional peer address (UDP) beyond
            // write/read; the address-aware path lives in net.rs.
            Sysno::Sendto => {
                self.sys_sendto(args[0], args[1], args[2], args[3], args[4], args[5], mem)
            }
            Sysno::Recvfrom => {
                self.sys_recvfrom(args[0], args[1], args[2], args[3], args[4], args[5], mem)
            }
            Sysno::Brk => self.sys_brk(args[0], mem),
            Sysno::Mmap => self.sys_mmap(args, mem),
            Sysno::Munmap => self.sys_munmap(args[0], args[1], mem),
            Sysno::Mprotect => self.sys_mprotect(args[0], args[1], args[2], mem),
            Sysno::Mremap => {
                self.sys_mremap(args[0], args[1], args[2], args[3], args[4], mem)
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
            // The guest does not own the host clock: refuse to set it.
            Sysno::Settimeofday | Sysno::ClockSettime => err(Errno::EPERM),
            Sysno::Openat => self.sys_openat(args[0] as i64, args[1], args[2], args[3], mem),
            Sysno::Close => self.sys_close(args[0] as i32),
            Sysno::Lseek => self.sys_lseek(args[0], args[1] as i64, args[2]),
            Sysno::Fstat => self.sys_fstat(args[0], args[1], mem),
            Sysno::Newfstatat => {
                self.sys_newfstatat(args[0] as i64, args[1], args[2], args[3], mem)
            }
            Sysno::Getdents64 => self.sys_getdents64(args[0], args[1], args[2], mem),
            Sysno::Getcwd => self.sys_getcwd(args[0], args[1], mem),
            Sysno::Chdir => self.sys_chdir(args[0], mem),
            Sysno::Statfs => self.sys_statfs(args[0], args[1], mem),
            Sysno::Fstatfs => self.sys_fstatfs(args[0], args[1], mem),
            Sysno::Readlinkat => {
                self.sys_readlinkat(args[0] as i64, args[1], args[2], args[3], mem)
            }
            Sysno::Symlinkat => self.sys_symlinkat(args[0], args[1] as i64, args[2], mem),
            Sysno::Mkdirat => self.sys_mkdirat(args[0] as i64, args[1], args[2], mem),
            Sysno::Unlinkat => self.sys_unlinkat(args[0] as i64, args[1], args[2], mem),
            Sysno::Renameat | Sysno::Renameat2 => {
                self.sys_renameat(args[0] as i64, args[1], args[2] as i64, args[3], mem)
            }
            Sysno::Faccessat | Sysno::Faccessat2 => {
                self.sys_faccessat(args[0] as i64, args[1], mem)
            }
            Sysno::Access => self.sys_faccessat(AT_FDCWD, args[0], mem),
            Sysno::Umask => self.sys_umask(args[0]),
            // No extended attributes: report "no such attribute".
            Sysno::Getxattr | Sysno::Lgetxattr | Sysno::Fgetxattr => err(Errno::ENODATA),
            Sysno::Writev => self.sys_writev(args[0], args[1], args[2], mem),
            Sysno::Getrandom => self.sys_getrandom(args[0], args[1], mem),
            Sysno::Ioctl => err(Errno::ENOTTY),
            Sysno::Fcntl => self.sys_fcntl(args[0], args[1]),
            Sysno::Futex => self.sys_futex(args, mem),
            Sysno::Pipe2 => self.sys_pipe2(args[0], mem),
            Sysno::Socket => self.sys_socket(args[0], args[1], args[2]),
            Sysno::Socketpair => self.sys_socketpair(args[0], args[1], args[2], args[3], mem),
            Sysno::Bind => self.sys_bind(args[0], args[1], args[2], mem),
            Sysno::Listen => self.sys_listen(args[0]),
            Sysno::Accept4 => self.sys_accept4(args[0], args[1], args[2], args[3], mem),
            Sysno::Connect => self.sys_connect(args[0], args[1], args[2], mem),
            Sysno::Getsockname => self.sys_getsockname(args[0], args[1], args[2], mem),
            Sysno::Getpeername => self.sys_getpeername(args[0], args[1], args[2], mem),
            Sysno::Setsockopt => {
                self.sys_setsockopt(args[0], args[1], args[2], args[3], args[4], mem)
            }
            Sysno::Getsockopt => {
                self.sys_getsockopt(args[0], args[1], args[2], args[3], args[4], mem)
            }
            Sysno::Shutdown => self.sys_shutdown(args[0], args[1]),
            // Event-notification / readiness syscalls.
            Sysno::Poll => self.sys_poll(args[0], args[1], args[2] as i64, mem),
            Sysno::Ppoll => self.sys_ppoll(args[0], args[1], args[2], args[3], args[4], mem),
            Sysno::Select => self.sys_select(args[0], args[1], args[2], args[3], args[4], mem),
            Sysno::Pselect6 => {
                self.sys_pselect6(args[0], args[1], args[2], args[3], args[4], args[5], mem)
            }
            Sysno::EpollCreate | Sysno::EpollCreate1 => self.sys_epoll_create1(args[0]),
            Sysno::EpollCtl => self.sys_epoll_ctl(args[0], args[1], args[2], args[3], mem),
            Sysno::EpollWait | Sysno::EpollPwait => {
                self.sys_epoll_wait(args[0], args[1], args[2], args[3] as i64, mem)
            }
            Sysno::EpollPwait2 => self.sys_epoll_pwait2(args[0], args[1], args[2], args[3], mem),
            Sysno::Eventfd => self.sys_eventfd2(args[0], 0),
            Sysno::Eventfd2 => self.sys_eventfd2(args[0], args[1]),
            Sysno::TimerfdCreate => self.sys_timerfd_create(args[0], args[1]),
            Sysno::TimerfdSettime => {
                self.sys_timerfd_settime(args[0], args[1], args[2], args[3], mem)
            }
            Sysno::TimerfdGettime => self.sys_timerfd_gettime(args[0], args[1], mem),
            Sysno::Dup => self.sys_dup(args[0]),
            Sysno::Dup2 | Sysno::Dup3 => self.sys_dup2(args[0], args[1]),
            Sysno::Clone => self.sys_clone(args, vcpu, mem),
            Sysno::Execve => self.sys_execve(args[0], args[1], args[2], vcpu, mem),
            Sysno::Wait4 => self.sys_wait4(args[0] as i64, args[1], args[2], mem),
            Sysno::Exit => self.sys_exit(args[0] as i32, mem),
            Sysno::ExitGroup => self.sys_exit_group(args[0] as i32, mem),
            Sysno::RtSigaction => self.sys_rt_sigaction(args[0], args[1], args[2], mem),
            Sysno::RtSigprocmask => self.sys_rt_sigprocmask(args[0], args[1], args[2], mem),
            Sysno::RtSigpending => self.sys_rt_sigpending(args[0], mem),
            Sysno::RtSigtimedwait => err(Errno::EAGAIN),
            Sysno::Kill | Sysno::Tkill => self.sys_kill(args[0] as i64, args[1]),
            Sysno::Tgkill => self.sys_kill(args[1] as i64, args[2]),
            // getpid = thread-group id; gettid = this task's id.
            Sysno::Getpid => i64::from(self.cur.tgid),
            Sysno::Gettid => i64::from(self.cur.pid),
            // set_tid_address records the CHILD_CLEARTID word and returns the tid.
            Sysno::SetTidAddress => {
                self.cur.clear_child_tid = args[0];
                i64::from(self.cur.pid)
            }
            Sysno::Getppid => i64::from(self.cur.ppid),
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
            Sysno::Prlimit64 => sys_misc::sys_prlimit64(args[1], args[3], mem),
            Sysno::Getrlimit => sys_misc::sys_getrlimit(args[0], args[1], mem),
            Sysno::Prctl => self.sys_prctl(args, mem),
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
            | Sysno::Sigaltstack
            | Sysno::RtSigsuspend
            | Sysno::RtSigreturn
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
            | Sysno::Msync
            | Sysno::SchedYield
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
            | Sysno::Membarrier => 0,
            _ => {
                *self.unsupported.entry(raw).or_default() += 1;
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
    /// same [`Kernel::spaces`] slot); otherwise the space is copied. `CLONE_THREAD`
    /// puts the new task in the caller's thread group (shared `tgid`, distinct
    /// `pid`/tid, not reaped by `wait4`). `CLONE_SETTLS` seeds the thread pointer;
    /// the `*_SETTID`/`CHILD_CLEARTID` flags write/clear the tid words musl's
    /// pthread layer relies on. The fd table is still copied (a `CLONE_FILES`
    /// approximation: correct for fork, and fine for threads that don't pass
    /// fds between each other after creation).
    fn sys_clone(&mut self, args: &[u64; 6], vcpu: &mut dyn Vcpu, mem: &mut GuestMemory) -> i64 {
        const CLONE_VM: u64 = 0x0000_0100;
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
        let share_vm = flags & CLONE_VM != 0;
        let is_thread = flags & CLONE_THREAD != 0;

        let pid = self.next_pid;
        self.next_pid += 1;
        let mut info = self.cur.clone();
        info.pid = pid;
        info.run = RunState::Running;
        info.futex_wait = None;
        info.futex_woken = false;
        if is_thread {
            info.tgid = self.cur.tgid;
            info.ppid = self.cur.ppid;
            info.is_thread = true;
        } else {
            info.tgid = pid;
            info.ppid = self.cur.pid;
            info.is_thread = false;
        }

        // Address space: share the caller's slot (CLONE_VM) or take a fresh copy.
        let mut child_mem = if share_vm { None } else { Some(mem.clone()) };
        info.mm = if share_vm {
            self.cur.mm
        } else {
            self.spaces.len()
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

        // The child holds copies of every open fd; bump pipe/socket refcounts.
        let (mut r, mut w) = (Vec::new(), Vec::new());
        for fd in info.fds.values() {
            match fd {
                Fd::PipeRead(i) => r.push(*i),
                Fd::PipeWrite(i) => w.push(*i),
                Fd::Socket { .. } => self.net.bump(fd, true),
                _ => {}
            }
        }
        for i in r {
            self.pipes[i].readers += 1;
        }
        for i in w {
            self.pipes[i].writers += 1;
        }

        if let Some(cm) = child_mem.take() {
            self.spaces.push(Arc::new(Mutex::new(cm)));
        }

        let mut child_vcpu = vcpu.fork();
        if stack != 0 {
            child_vcpu.set_sp(stack);
        }
        if flags & CLONE_SETTLS != 0 {
            child_vcpu.set_tls(tls);
        }
        child_vcpu.set_syscall_ret(0); // child returns 0 and advances past the svc
        self.procs.push(Some(Process {
            vcpu: Some(child_vcpu),
            info,
        }));
        i64::from(pid)
    }

    /// `execve(path, argv, envp)` — replace the process image with a new ELF
    /// read from the mount table (following symlinks). Static and static-PIE
    /// images load directly; a dynamic executable's `PT_INTERP` linker is read
    /// from the same root and loaded alongside it.
    fn sys_execve(
        &mut self,
        path_ptr: u64,
        argv_ptr: u64,
        envp_ptr: u64,
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some(rel) = read_path(mem, path_ptr) else {
            return err(Errno::EFAULT);
        };
        let Some(abs) = self.resolve_exec(&rel) else {
            return err(Errno::ENOENT);
        };
        let Some(elf) = self.read_file(&abs) else {
            return err(Errno::ENOENT);
        };
        let argv = read_string_array(mem, argv_ptr);
        let envp = read_string_array(mem, envp_ptr);

        let (base, size) = (mem.base(), mem.size());
        let mut new_mem = GuestMemory::new(base, size);
        let spec = ProcessSpec { argv, envp };
        let loaded = if let Some(interp) = interp_path(&elf) {
            let Some(interp_elf) = self.read_file(&interp) else {
                return err(Errno::ENOENT); // interpreter missing
            };
            load_dynamic(&mut new_mem, &elf, &interp_elf, &spec)
        } else {
            load_static(&mut new_mem, &elf, &spec)
        };
        let Ok(img) = loaded else {
            return err(Errno::ENOEXEC);
        };
        *mem = new_mem;
        vcpu.reset(img.entry, img.stack_pointer);
        let mid = page_down(img.program_break + (img.stack_bottom - img.program_break) / 2);
        self.cur.brk = img.program_break;
        self.cur.heap_start = img.program_break;
        self.cur.heap_limit = mid;
        self.cur.mmap_cursor = img.stack_bottom;
        self.cur.mmap_floor = mid;
        self.exec_ok = true;
        0
    }

    /// `wait4(pid, wstatus, options, rusage)` — reap a zombie child.
    fn sys_wait4(&mut self, _pid: i64, wstatus: u64, options: u64, mem: &mut GuestMemory) -> i64 {
        const WNOHANG: u64 = 1;
        let cur = self.cur.pid;
        let mut zombie = None;
        let mut has_child = false;
        for p in self.procs.iter().flatten() {
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
            for slot in &mut self.procs {
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
        self.block = true; // wait for a child to exit
        0
    }

    /// `exit` — terminate just this task: run its `CLONE_CHILD_CLEARTID`
    /// notification (so a joiner wakes), close its fds (so pipe peers see EOF),
    /// and become a zombie until reaped.
    fn sys_exit(&mut self, code: i32, mem: &mut GuestMemory) -> i64 {
        let ctid = self.cur.clear_child_tid;
        let mm = self.cur.mm;
        if ctid != 0 {
            let _ = mem.write(ctid, &0u32.to_le_bytes());
            self.futex_wake(mm, ctid, i32::MAX);
        }
        for fd in self.cur.fds.drain() {
            self.bump_pipe(&fd, false);
        }
        self.cur.run = RunState::Zombie(code & 0xff);
        0
    }

    /// `exit_group` — terminate the whole thread group: this task plus every
    /// sibling sharing our `tgid`. Each dying task closes its fds; the running
    /// task also runs its `CLONE_CHILD_CLEARTID` notification.
    fn sys_exit_group(&mut self, code: i32, mem: &mut GuestMemory) -> i64 {
        let tgid = self.cur.tgid;
        let status = code & 0xff;
        // Collect the siblings' fds first: closing them touches `self.pipes` /
        // `self.net`, which we cannot borrow while iterating `self.procs`.
        let mut to_close: Vec<Fd> = Vec::new();
        for slot in &mut self.procs {
            let Some(p) = slot.as_mut() else { continue };
            if p.info.tgid != tgid || matches!(p.info.run, RunState::Zombie(_)) {
                continue;
            }
            to_close.extend(p.info.fds.drain());
            p.info.run = RunState::Zombie(status);
        }
        for fd in to_close {
            self.bump_pipe(&fd, false);
        }
        // `self.cur` is this task, taken out of the table for its slice.
        self.sys_exit(code, mem)
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
    fn sys_futex(&mut self, args: &[u64; 6], mem: &GuestMemory) -> i64 {
        const FUTEX_WAIT: u64 = 0;
        const FUTEX_WAKE: u64 = 1;
        const FUTEX_WAIT_BITSET: u64 = 9;
        const FUTEX_WAKE_BITSET: u64 = 10;
        let uaddr = args[0];
        let op = args[1] & 0x7f; // strip FUTEX_PRIVATE_FLAG / CLOCK_REALTIME
        let val = args[2] as u32;
        let mm = self.cur.mm;
        match op {
            FUTEX_WAIT | FUTEX_WAIT_BITSET => {
                // Already parked here: consult the wake flag only.
                if self.cur.futex_wait == Some((mm, uaddr)) {
                    if self.cur.futex_woken {
                        self.cur.futex_wait = None;
                        self.cur.futex_woken = false;
                        return 0;
                    }
                    self.block = true;
                    return 0;
                }
                match mem.read_u32(uaddr) {
                    Ok(cur) if cur != val => err(Errno::EAGAIN),
                    Ok(_) => {
                        // Park only if some other task could wake us; otherwise a
                        // lone waiter would deadlock, so fake a wake.
                        let others = self
                            .procs
                            .iter()
                            .flatten()
                            .any(|p| p.info.pid != self.cur.pid && p.info.run == RunState::Running);
                        if !others {
                            return 0;
                        }
                        self.cur.futex_wait = Some((mm, uaddr));
                        self.cur.futex_woken = false;
                        self.block = true;
                        0
                    }
                    Err(_) => err(Errno::EFAULT),
                }
            }
            FUTEX_WAKE | FUTEX_WAKE_BITSET => self.futex_wake(mm, uaddr, val as i32),
            _ => 0,
        }
    }

    /// Release up to `n` tasks parked in `FUTEX_WAIT` on `(mm, uaddr)`; returns
    /// how many were woken.
    fn futex_wake(&mut self, mm: usize, uaddr: u64, n: i32) -> i64 {
        let mut woken = 0i64;
        for p in self.procs.iter_mut().flatten() {
            if woken >= i64::from(n) {
                break;
            }
            if p.info.futex_wait == Some((mm, uaddr)) && !p.info.futex_woken {
                p.info.futex_woken = true;
                woken += 1;
            }
        }
        woken
    }

    // ---- files & fds ------------------------------------------------------

    /// `write(fd, buf, count)` — stdio sinks (fd 1/2), files, and pipes.
    fn sys_write(&mut self, fd: u64, buf: u64, count: u64, mem: &GuestMemory) -> i64 {
        let Ok(data) = mem.read_vec(buf, count as usize) else {
            return err(Errno::EFAULT);
        };
        // fd 1/2 fall back to the host sinks only when still the standard stream.
        match self.cur.fds.get(fd as i32).cloned() {
            Some(Fd::Stdout) => match self.stdout.write_all(&data) {
                Ok(()) => count as i64,
                Err(_) => err(Errno::EIO),
            },
            Some(Fd::Stderr) => match self.stderr.write_all(&data) {
                Ok(()) => count as i64,
                Err(_) => err(Errno::EIO),
            },
            Some(Fd::File { path, offset }) => match self.mounts.write_at(&path, offset, &data) {
                Ok(n) => {
                    if let Some(Fd::File { offset, .. }) = self.cur.fds.get_mut(fd as i32) {
                        *offset += n as u64;
                    }
                    n as i64
                }
                Err(e) => io_errno(&e),
            },
            Some(Fd::PipeWrite(i)) => self.write_pipe(i, &data),
            Some(Fd::Socket { sock, end }) => self.write_socket(sock, end, &data),
            Some(Fd::Eventfd(i)) => self.write_eventfd(i, &data),
            _ => err(Errno::EBADF),
        }
    }

    /// `read(fd, buf, count)` — stdin, files, and pipes.
    fn sys_read(&mut self, fd: u64, buf: u64, count: u64, mem: &mut GuestMemory) -> i64 {
        match self.cur.fds.get(fd as i32).cloned() {
            Some(Fd::Stdin) => {
                let mut tmp = vec![0u8; count as usize];
                match self.stdin.read(&mut tmp) {
                    Ok(n) => {
                        if mem.write(buf, &tmp[..n]).is_err() {
                            return err(Errno::EFAULT);
                        }
                        n as i64
                    }
                    Err(_) => err(Errno::EIO),
                }
            }
            Some(Fd::PipeRead(i)) => self.read_pipe(i, buf, count, mem),
            Some(Fd::Socket { sock, end }) => self.read_socket(sock, end, buf, count, mem),
            Some(Fd::Eventfd(i)) => self.read_eventfd(i, buf, count, mem),
            Some(Fd::Timerfd(i)) => self.read_timerfd(i, buf, count, mem),
            Some(Fd::File { path, offset }) => {
                let mut tmp = vec![0u8; count as usize];
                match self.mounts.read_at(&path, offset, &mut tmp) {
                    Ok(n) => {
                        if mem.write(buf, &tmp[..n]).is_err() {
                            return err(Errno::EFAULT);
                        }
                        if let Some(Fd::File { offset, .. }) = self.cur.fds.get_mut(fd as i32) {
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
    fn read_pipe(&mut self, i: usize, buf: u64, count: u64, mem: &mut GuestMemory) -> i64 {
        if self.pipes[i].buf.is_empty() {
            if self.pipes[i].writers > 0 {
                self.block = true;
            }
            return 0;
        }
        let n = count.min(self.pipes[i].buf.len() as u64) as usize;
        let data: Vec<u8> = self.pipes[i].buf.drain(..n).collect();
        if mem.write(buf, &data).is_err() {
            return err(Errno::EFAULT);
        }
        n as i64
    }

    /// Write to pipe `i` (`EPIPE` if all readers are gone).
    fn write_pipe(&mut self, i: usize, data: &[u8]) -> i64 {
        if self.pipes[i].readers == 0 {
            return err(Errno::EPIPE);
        }
        self.pipes[i].buf.extend(data.iter().copied());
        data.len() as i64
    }

    /// `writev(fd, iov, iovcnt)` — gather `struct iovec { base; len }` entries.
    fn sys_writev(&mut self, fd: u64, iov: u64, iovcnt: u64, mem: &GuestMemory) -> i64 {
        let mut total = 0i64;
        for i in 0..iovcnt {
            let ent = iov + i * 16;
            let (Ok(base), Ok(len)) = (mem.read_u64(ent), mem.read_u64(ent + 8)) else {
                return if total > 0 { total } else { err(Errno::EFAULT) };
            };
            if len == 0 {
                continue;
            }
            let r = self.sys_write(fd, base, len, mem);
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

    /// `fcntl(fd, cmd, ...)` — the subset real programs need at startup.
    fn sys_fcntl(&mut self, fd: u64, cmd: u64) -> i64 {
        const F_DUPFD: u64 = 0;
        const F_GETFL: u64 = 3;
        const F_DUPFD_CLOEXEC: u64 = 1030;
        match cmd {
            F_DUPFD | F_DUPFD_CLOEXEC => match self.cur.fds.get(fd as i32).cloned() {
                Some(f) => {
                    self.bump_pipe(&f, true);
                    i64::from(self.cur.fds.alloc(f))
                }
                None => err(Errno::EBADF),
            },
            F_GETFL => 2,
            _ => 0,
        }
    }

    /// `openat(dirfd, path, flags, mode)` against the mount table.
    fn sys_openat(
        &mut self,
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
        let abs = self.resolve_path(dirfd, &rel);
        let abs = self.follow_symlinks(&abs).unwrap_or(abs);

        if self.mounts.stat(&abs).is_none() {
            if flags & O_CREAT != 0 {
                if let Err(e) = self.mounts.create(&abs, (mode & 0o777) as u32) {
                    return io_errno(&e);
                }
            } else {
                return err(Errno::ENOENT);
            }
        } else if flags & O_TRUNC != 0 {
            let _ = self.mounts.truncate(&abs, 0);
        }

        let Some(attrs) = self.mounts.stat(&abs) else {
            return err(Errno::ENOENT);
        };
        let fd = if attrs.kind == NodeKind::Dir {
            self.cur.fds.alloc(Fd::Dir { path: abs, pos: 0 })
        } else {
            self.cur.fds.alloc(Fd::File {
                path: abs,
                offset: 0,
            })
        };
        i64::from(fd)
    }

    /// `close(fd)`.
    fn sys_close(&mut self, fd: i32) -> i64 {
        match self.cur.fds.close(fd) {
            Some(f) => {
                self.bump_pipe(&f, false);
                0
            }
            None => err(Errno::EBADF),
        }
    }

    /// `pipe2(fds, flags)` — create an anonymous pipe.
    fn sys_pipe2(&mut self, fds_ptr: u64, mem: &mut GuestMemory) -> i64 {
        let idx = self.pipes.len();
        self.pipes.push(Pipe {
            buf: VecDeque::new(),
            readers: 1,
            writers: 1,
        });
        let rfd = self.cur.fds.alloc(Fd::PipeRead(idx));
        let wfd = self.cur.fds.alloc(Fd::PipeWrite(idx));
        let mut b = [0u8; 8];
        b[0..4].copy_from_slice(&rfd.to_le_bytes());
        b[4..8].copy_from_slice(&wfd.to_le_bytes());
        if mem.write(fds_ptr, &b).is_err() {
            return err(Errno::EFAULT);
        }
        0
    }

    /// `dup(oldfd)`.
    fn sys_dup(&mut self, oldfd: u64) -> i64 {
        let Some(fd) = self.cur.fds.get(oldfd as i32).cloned() else {
            return err(Errno::EBADF);
        };
        self.bump_pipe(&fd, true);
        i64::from(self.cur.fds.alloc(fd))
    }

    /// `dup2`/`dup3(oldfd, newfd)`.
    fn sys_dup2(&mut self, oldfd: u64, newfd: u64) -> i64 {
        let Some(fd) = self.cur.fds.get(oldfd as i32).cloned() else {
            return err(Errno::EBADF);
        };
        if oldfd == newfd {
            return newfd as i64;
        }
        if let Some(old) = self.cur.fds.close(newfd as i32) {
            self.bump_pipe(&old, false);
        }
        self.bump_pipe(&fd, true);
        self.cur.fds.insert(newfd as i32, fd);
        newfd as i64
    }

    /// Adjust the reader/writer refcount of the pipe a fd refers to (if any).
    fn bump_pipe(&mut self, fd: &Fd, inc: bool) {
        let apply = |n: &mut usize| {
            if inc {
                *n += 1;
            } else {
                *n = n.saturating_sub(1);
            }
        };
        match fd {
            Fd::PipeRead(i) => apply(&mut self.pipes[*i].readers),
            Fd::PipeWrite(i) => apply(&mut self.pipes[*i].writers),
            f => self.net.bump(f, inc),
        }
    }

    /// `lseek(fd, offset, whence)`.
    fn sys_lseek(&mut self, fd: u64, offset: i64, whence: u64) -> i64 {
        let (cur, path) = match self.cur.fds.get(fd as i32) {
            Some(Fd::File { path, offset }) => (*offset, path.clone()),
            _ => return err(Errno::ESPIPE),
        };
        let size = self.mounts.stat(&path).map_or(0, |a| a.size);
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
        if let Some(Fd::File { offset, .. }) = self.cur.fds.get_mut(fd as i32) {
            *offset = newpos as u64;
        }
        newpos
    }

    /// `fstat(fd, statbuf)`.
    fn sys_fstat(&mut self, fd: u64, statbuf: u64, mem: &mut GuestMemory) -> i64 {
        let attrs = match self.cur.fds.get(fd as i32) {
            Some(Fd::File { path, .. } | Fd::Dir { path, .. }) => {
                let path = path.clone();
                match self.mounts.stat(&path) {
                    Some(a) => a,
                    None => return err(Errno::ENOENT),
                }
            }
            // eventfd/timerfd/epoll are anonymous-inode char-device-like fds.
            Some(Fd::Stdin | Fd::Stdout | Fd::Stderr | Fd::Eventfd(_) | Fd::Timerfd(_) | Fd::Epoll(_)) => {
                stat::char_device_attrs()
            }
            Some(Fd::PipeRead(_) | Fd::PipeWrite(_)) => stat::fifo_attrs(),
            Some(Fd::Socket { .. }) => stat::socket_attrs(),
            None => return err(Errno::EBADF),
        };
        write_stat_or_fault(mem, statbuf, &attrs)
    }

    /// `newfstatat(dirfd, path, statbuf, flags)`.
    fn sys_newfstatat(
        &mut self,
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
        let mut abs = self.resolve_path(dirfd, &rel);
        if flags & AT_SYMLINK_NOFOLLOW == 0 {
            abs = self.follow_symlinks(&abs).unwrap_or(abs);
        }
        let Some(attrs) = self.mounts.stat(&abs) else {
            return err(Errno::ENOENT);
        };
        write_stat_or_fault(mem, statbuf, &attrs)
    }

    /// `getdents64(fd, buf, count)`.
    fn sys_getdents64(&mut self, fd: u64, buf: u64, count: u64, mem: &mut GuestMemory) -> i64 {
        let (path, pos) = match self.cur.fds.get(fd as i32) {
            Some(Fd::Dir { path, pos }) => (path.clone(), *pos),
            _ => return err(Errno::ENOTDIR),
        };
        let entries = match self.mounts.readdir(&path) {
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
        if let Some(Fd::Dir { pos, .. }) = self.cur.fds.get_mut(fd as i32) {
            *pos = consumed;
        }
        bytes.len() as i64
    }

    /// `getcwd(buf, size)`.
    fn sys_getcwd(&mut self, buf: u64, size: u64, mem: &mut GuestMemory) -> i64 {
        let mut bytes = self.cur.cwd.clone().into_bytes();
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
    fn sys_chdir(&mut self, pathptr: u64, mem: &GuestMemory) -> i64 {
        let Some(rel) = read_path(mem, pathptr) else {
            return err(Errno::EFAULT);
        };
        let abs = self.resolve_path(AT_FDCWD, &rel);
        match self.mounts.stat(&abs) {
            Some(a) if a.kind == NodeKind::Dir => {
                self.cur.cwd = abs;
                0
            }
            Some(_) => err(Errno::ENOTDIR),
            None => err(Errno::ENOENT),
        }
    }

    /// Resolve a possibly-relative guest path to an absolute, normalized path.
    fn resolve_path(&self, dirfd: i64, p: &str) -> String {
        if p.starts_with('/') {
            return path::normalize(p);
        }
        let base = if dirfd == AT_FDCWD {
            self.cur.cwd.clone()
        } else {
            match self.cur.fds.get(dirfd as i32) {
                Some(Fd::Dir { path, .. } | Fd::File { path, .. }) => path.clone(),
                _ => self.cur.cwd.clone(),
            }
        };
        path::normalize(&format!("{base}/{p}"))
    }

    /// Follow the final-component symlink chain (bounded), returning the target.
    fn follow_symlinks(&mut self, path: &str) -> Option<String> {
        let mut p = path.to_string();
        for _ in 0..SYMLINK_MAX {
            match self.mounts.stat(&p) {
                Some(a) if a.kind == NodeKind::Symlink => {
                    let target = self.mounts.readlink(&p).ok()?;
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
    fn resolve_exec(&mut self, p: &str) -> Option<String> {
        let abs = self.resolve_path(AT_FDCWD, p);
        self.follow_symlinks(&abs)
    }

    /// Read an entire file from the mount table.
    fn read_file(&mut self, path: &str) -> Option<Vec<u8>> {
        let size = self.mounts.stat(path)?.size as usize;
        let mut buf = vec![0u8; size];
        let mut off = 0;
        while off < size {
            match self.mounts.read_at(path, off as u64, &mut buf[off..]) {
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
    fn sys_brk(&mut self, addr: u64, mem: &mut GuestMemory) -> i64 {
        if addr == 0 || addr < self.cur.heap_start {
            return self.cur.brk as i64;
        }
        if addr > self.cur.brk {
            let from = self.cur.brk - self.cur.brk % PAGE_SIZE;
            let to = addr.div_ceil(PAGE_SIZE) * PAGE_SIZE;
            if to > self.cur.heap_limit || mem.map(from, to - from, Prot::rw()).is_err() {
                return self.cur.brk as i64;
            }
        } else if addr < self.cur.brk {
            let from = addr.div_ceil(PAGE_SIZE) * PAGE_SIZE;
            let to = self.cur.brk.div_ceil(PAGE_SIZE) * PAGE_SIZE;
            if to > from {
                let _ = mem.unmap(from, to - from);
            }
        }
        self.cur.brk = addr;
        self.cur.brk as i64
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
    fn sys_mmap(&mut self, a: &[u64; 6], mem: &mut GuestMemory) -> i64 {
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
            match self.cur.fds.get(fd as i32) {
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
            let Some(new_top) = self.cur.mmap_cursor.checked_sub(len) else {
                return err(Errno::ENOMEM);
            };
            if new_top < self.cur.mmap_floor {
                return err(Errno::ENOMEM);
            }
            self.cur.mmap_cursor = new_top;
            new_top
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
                match self.mounts.read_at(&path, offset + got as u64, &mut data[got..]) {
                    Ok(n) if n > 0 => got += n,
                    _ => break, // EOF or read error: leave the rest zero-filled
                }
            }
            // write_init bypasses page protection, so a read/exec-only mapping
            // (the common code-segment case) is still populated correctly.
            if mem.write_init(base, &data).is_err() {
                return err(Errno::ENOMEM);
            }
        }
        base as i64
    }

    /// Reserve `len` bytes (rounded up to a page) from the downward-growing
    /// anonymous `mmap` arena, returning the base of the fresh region, or `None`
    /// if the arena is exhausted. Shares the cursor discipline of [`Self::sys_mmap`]
    /// so relocating callers (`mremap` MAYMOVE) allocate the same way.
    pub(super) fn alloc_mmap(&mut self, len: u64) -> Option<u64> {
        let len = len.div_ceil(PAGE_SIZE) * PAGE_SIZE;
        let new_top = self.cur.mmap_cursor.checked_sub(len)?;
        if new_top < self.cur.mmap_floor {
            return None;
        }
        self.cur.mmap_cursor = new_top;
        Some(new_top)
    }

    /// `munmap(addr, len)`.
    #[allow(clippy::unused_self)]
    fn sys_munmap(&mut self, addr: u64, len: u64, mem: &mut GuestMemory) -> i64 {
        if len == 0 {
            return err(Errno::EINVAL);
        }
        let len = len.div_ceil(PAGE_SIZE) * PAGE_SIZE;
        let _ = mem.unmap(addr - addr % PAGE_SIZE, len);
        0
    }

    /// `mprotect(addr, len, prot)`.
    #[allow(clippy::unused_self)]
    fn sys_mprotect(&mut self, addr: u64, len: u64, prot: u64, mem: &mut GuestMemory) -> i64 {
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
    fn sys_getrandom(&mut self, buf: u64, len: u64, mem: &mut GuestMemory) -> i64 {
        if self.rng_state == 0 {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0x9E37_79B9_7F4A_7C15, |d| d.as_nanos() as u64);
            self.rng_state = now | 1;
        }
        let mut out = vec![0u8; len as usize];
        for chunk in out.chunks_mut(8) {
            let mut s = self.rng_state;
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            self.rng_state = s;
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

    /// Syscalls the guest attempted that nixvm does not implement yet.
    #[must_use]
    pub fn unsupported(&self) -> &BTreeMap<u64, u64> {
        &self.unsupported
    }
}

/// `clock_gettime(clk_id, timespec)`.
fn sys_clock_gettime(ts: u64, mem: &mut GuestMemory) -> i64 {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
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

/// Map a host `io::Error` to a negative guest errno.
fn io_errno(e: &io::Error) -> i64 {
    match e.raw_os_error() {
        Some(n) => -i64::from(n),
        None => err(Errno::EIO),
    }
}

/// Write a `struct stat` for `attrs` at `addr`, or return `-EFAULT`.
fn write_stat_or_fault(mem: &mut GuestMemory, addr: u64, attrs: &Attrs) -> i64 {
    let buf = stat::encode_stat(attrs);
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
    const NR_GETPID: u64 = 172;
    const NR_CLONE: u64 = 220;

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
            k.procs.iter().flatten().any(|p| p.info.pid == 2),
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

    const PAGE: u64 = 4096;
    const AT_CWD: u64 = (-100i64) as u64;

    fn setup() -> (Kernel, GuestMemory, DummyVcpu) {
        let mut mounts = MountTable::new();
        mounts.mount("/", Box::new(TmpFs::new()));
        let mut kernel = Kernel::new(Arch::Aarch64, mounts);
        kernel.cur.pid = 1;
        kernel.cur.tgid = 1;
        let mut mem = GuestMemory::new(0x1_0000, 16 * PAGE);
        mem.map(0x1_0000, 4 * PAGE, Prot::rw()).unwrap();
        (kernel, mem, DummyVcpu)
    }

    fn call(
        k: &mut Kernel,
        mem: &mut GuestMemory,
        v: &mut DummyVcpu,
        s: Sysno,
        a: [u64; 6],
    ) -> i64 {
        k.dispatch(s, 0, &a, v, mem)
    }

    #[test]
    fn openat_write_lseek_read_roundtrip() {
        let (mut k, mut mem, mut v) = setup();
        let path = 0x1_0000;
        let msg = 0x1_1000;
        let buf = 0x1_2000;
        mem.write_init(path, b"/f\0").unwrap();
        mem.write_init(msg, b"Hi").unwrap();

        let fd = call(
            &mut k,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_CWD, path, 0o102, 0o644, 0, 0],
        );
        assert_eq!(fd, 3);
        let fd = fd as u64;

        assert_eq!(
            call(
                &mut k,
                &mut mem,
                &mut v,
                Sysno::Write,
                [fd, msg, 2, 0, 0, 0]
            ),
            2
        );
        assert_eq!(
            call(&mut k, &mut mem, &mut v, Sysno::Lseek, [fd, 0, 0, 0, 0, 0]),
            0
        );
        assert_eq!(
            call(&mut k, &mut mem, &mut v, Sysno::Read, [fd, buf, 2, 0, 0, 0]),
            2
        );
        assert_eq!(mem.read_vec(buf, 2).unwrap(), b"Hi");

        let stbuf = 0x1_3000;
        assert_eq!(
            call(
                &mut k,
                &mut mem,
                &mut v,
                Sysno::Fstat,
                [fd, stbuf, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_u64(stbuf + 48).unwrap(), 2);

        assert_eq!(
            call(&mut k, &mut mem, &mut v, Sysno::Close, [fd, 0, 0, 0, 0, 0]),
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
        let (mut k, mut mem, mut v) = setup();
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
                &mut k,
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
        let (mut k, mut mem, mut v) = setup();
        let fds = 0x1_0000;
        assert_eq!(
            call(&mut k, &mut mem, &mut v, Sysno::Pipe2, [fds, 0, 0, 0, 0, 0]),
            0
        );
        let rfd = u64::from(mem.read_u32(fds).unwrap());
        let wfd = u64::from(mem.read_u32(fds + 4).unwrap());
        assert!(rfd >= 3 && wfd >= 3 && rfd != wfd);

        let msg = 0x1_1000;
        mem.write_init(msg, b"pipe!").unwrap();
        assert_eq!(
            call(
                &mut k,
                &mut mem,
                &mut v,
                Sysno::Write,
                [wfd, msg, 5, 0, 0, 0]
            ),
            5
        );

        let dfd = call(&mut k, &mut mem, &mut v, Sysno::Dup, [rfd, 0, 0, 0, 0, 0]);
        assert!(dfd >= 3);
        let buf = 0x1_2000;
        assert_eq!(
            call(
                &mut k,
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
                &mut k,
                &mut mem,
                &mut v,
                Sysno::Read,
                [rfd, buf, 5, 0, 0, 0]
            ),
            0
        );
        assert!(k.block);
    }

    #[test]
    fn read_from_stdin() {
        let (mut k, mut mem, mut v) = setup();
        k.set_stdin(Box::new(std::io::Cursor::new(b"piped".to_vec())));
        let buf = 0x1_0000;
        assert_eq!(
            call(&mut k, &mut mem, &mut v, Sysno::Read, [0, buf, 5, 0, 0, 0]),
            5
        );
        assert_eq!(mem.read_vec(buf, 5).unwrap(), b"piped");
    }

    #[test]
    fn getrandom_fills_buffer() {
        let (mut k, mut mem, mut v) = setup();
        let buf = 0x1_0000;
        assert_eq!(
            call(
                &mut k,
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
        let (mut k, mut mem, mut v) = setup();
        // clone(flags=0, stack=0, ...) -> child pid
        let child = call(
            &mut k,
            &mut mem,
            &mut v,
            Sysno::Clone,
            [0x11, 0, 0, 0, 0, 0],
        );
        assert_eq!(child, 2, "first child is pid 2");
        assert_eq!(k.procs.len(), 1, "child pushed to the process table");

        // no zombie yet -> wait4 blocks
        let ws = 0x1_0000;
        call(
            &mut k,
            &mut mem,
            &mut v,
            Sysno::Wait4,
            [child as u64, ws, 0, 0, 0, 0],
        );
        assert!(k.block, "wait4 blocks while the child is alive");

        // make the child a zombie (exit code 7), then wait4 reaps it.
        if let Some(Some(p)) = k
            .procs
            .iter_mut()
            .find(|s| s.as_ref().is_some_and(|p| p.info.pid == 2))
        {
            p.info.run = RunState::Zombie(7);
        }
        let reaped = call(
            &mut k,
            &mut mem,
            &mut v,
            Sysno::Wait4,
            [child as u64, ws, 0, 0, 0, 0],
        );
        assert_eq!(reaped, 2);
        // WIFEXITED status: (code & 0xff) << 8
        assert_eq!(mem.read_u32(ws).unwrap(), 7 << 8);
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
        let (mut k, mut mem, mut v) = setup();
        k.cur.pid = 7; // a thread's tid
        k.cur.tgid = 1; // its process
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Getpid, [0; 6]), 1);
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Gettid, [0; 6]), 7);
    }

    #[test]
    fn clone_thread_shares_tgid_and_address_space() {
        let (mut k, mut mem, mut v) = setup();
        // CLONE_VM | CLONE_THREAD | CLONE_SETTLS
        let flags = 0x0000_0100 | 0x0001_0000 | 0x0008_0000;
        let tid = call(
            &mut k,
            &mut mem,
            &mut v,
            Sysno::Clone,
            [flags, 0x2_0000, 0, 0xdead_beef, 0, 0],
        );
        assert_eq!(tid, 2, "new thread gets a fresh tid");
        let spaces_before = k.spaces.len();
        let child = k
            .procs
            .iter()
            .flatten()
            .find(|p| p.info.pid == 2)
            .expect("thread in table");
        assert!(child.info.is_thread);
        assert_eq!(child.info.tgid, k.cur.tgid, "thread shares the tgid");
        assert_eq!(child.info.mm, k.cur.mm, "thread shares the address space");
        assert_eq!(
            spaces_before, k.spaces.len(),
            "CLONE_VM does not allocate a new address space"
        );
    }

    #[test]
    fn fork_gets_its_own_address_space() {
        let (mut k, mut mem, mut v) = setup();
        // Put the parent's space in the table (as run() would).
        k.spaces
            .push(Arc::new(Mutex::new(GuestMemory::new(0x1_0000, PAGE))));
        k.cur.mm = 0;
        let before = k.spaces.len();
        // flags = SIGCHLD only (a plain fork), no CLONE_VM.
        let child = call(&mut k, &mut mem, &mut v, Sysno::Clone, [0x11, 0, 0, 0, 0, 0]);
        assert_eq!(child, 2);
        let c = k.procs.iter().flatten().find(|p| p.info.pid == 2).unwrap();
        assert!(!c.info.is_thread);
        assert_eq!(c.info.tgid, 2, "a forked process is its own group");
        assert_ne!(c.info.mm, k.cur.mm, "fork copies the address space");
        assert_eq!(k.spaces.len(), before + 1);
    }

    #[test]
    fn exit_group_zombies_the_whole_thread_group() {
        let (mut k, mut mem, mut v) = setup();
        // Two sibling threads in the leader's group, plus an unrelated process.
        k.procs.push(Some(make_proc(2, 1, 0, true)));
        k.procs.push(Some(make_proc(3, 1, 0, true)));
        k.procs.push(Some(make_proc(4, 4, 1, false)));

        call(&mut k, &mut mem, &mut v, Sysno::ExitGroup, [42, 0, 0, 0, 0, 0]);

        assert!(matches!(k.cur.run, RunState::Zombie(42)), "leader exits");
        let state = |pid| {
            k.procs
                .iter()
                .flatten()
                .find(|p| p.info.pid == pid)
                .map(|p| p.info.run)
        };
        assert_eq!(state(2), Some(RunState::Zombie(42)), "sibling thread killed");
        assert_eq!(state(3), Some(RunState::Zombie(42)), "sibling thread killed");
        assert_eq!(
            state(4),
            Some(RunState::Running),
            "unrelated process untouched"
        );
    }

    #[test]
    fn futex_wake_releases_a_parked_waiter() {
        let (mut k, mut mem, mut v) = setup();
        let uaddr = 0x1_0000;
        // A sibling parked in FUTEX_WAIT on (mm 0, uaddr).
        let mut waiter = make_proc(2, 1, 0, true);
        waiter.info.futex_wait = Some((0, uaddr));
        k.procs.push(Some(waiter));

        // FUTEX_WAKE(uaddr, op=1, val=1) wakes exactly one waiter.
        let woken = call(
            &mut k,
            &mut mem,
            &mut v,
            Sysno::Futex,
            [uaddr, 1, 1, 0, 0, 0],
        );
        assert_eq!(woken, 1);
        let w = k.procs.iter().flatten().find(|p| p.info.pid == 2).unwrap();
        assert!(w.info.futex_woken, "waiter flagged for release");
    }

    #[test]
    fn futex_wait_single_thread_does_not_deadlock() {
        let (mut k, mut mem, mut v) = setup();
        let uaddr = 0x1_0000;
        mem.write_init(uaddr, &42u32.to_le_bytes()).unwrap();
        // Value matches and no other task could wake us: report a spurious wake
        // rather than parking (which would be a false deadlock).
        let r = call(
            &mut k,
            &mut mem,
            &mut v,
            Sysno::Futex,
            [uaddr, 0, 42, 0, 0, 0],
        );
        assert_eq!(r, 0);
        assert!(!k.block, "lone waiter is not parked");
        // A mismatched value is EAGAIN.
        let r = call(
            &mut k,
            &mut mem,
            &mut v,
            Sysno::Futex,
            [uaddr, 0, 99, 0, 0, 0],
        );
        assert_eq!(r, -i64::from(Errno::EAGAIN.0));
    }

    #[test]
    fn futex_wait_parks_when_a_sibling_can_wake() {
        let (mut k, mut mem, mut v) = setup();
        let uaddr = 0x1_0000;
        mem.write_init(uaddr, &42u32.to_le_bytes()).unwrap();
        // A runnable sibling exists, so a matching wait parks the caller.
        k.procs.push(Some(make_proc(2, 1, 0, true)));
        let r = call(
            &mut k,
            &mut mem,
            &mut v,
            Sysno::Futex,
            [uaddr, 0, 42, 0, 0, 0],
        );
        assert_eq!(r, 0);
        assert!(k.block, "caller parks awaiting a wake");
        assert_eq!(k.cur.futex_wait, Some((0, uaddr)));
    }

    #[test]
    fn mmap_file_backed_copies_file_contents() {
        const MAP_FIXED: u64 = 0x10;
        const PROT_READ: u64 = 0x1;
        let (mut k, mut mem, mut v) = setup();
        let path = 0x1_0000;
        let content = 0x1_1000;
        mem.write_init(path, b"/lib\0").unwrap();
        mem.write_init(content, &[0x11, 0x22, 0x33, 0x44]).unwrap();

        // Create /lib and write four bytes to it.
        let fd = call(
            &mut k,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_CWD, path, 0o102, 0o644, 0, 0],
        );
        assert_eq!(fd, 3);
        assert_eq!(
            call(&mut k, &mut mem, &mut v, Sysno::Write, [fd as u64, content, 4, 0, 0, 0]),
            4
        );

        // Map it read-only at a fixed address; the file bytes appear there.
        let addr = 0x1_5000u64;
        let ret = call(
            &mut k,
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
        let (mut k, mut mem, mut v) = setup();
        let path = 0x1_0000;
        let content = 0x1_1000;
        mem.write_init(path, b"/x\0").unwrap();
        mem.write_init(content, &[0xAB, 0xCD]).unwrap();
        // Pre-dirty the target page so we can prove the tail is zeroed.
        mem.write(0x1_3000, &[0xFF; 8]).unwrap();
        let fd = call(
            &mut k,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_CWD, path, 0o102, 0o644, 0, 0],
        );
        call(&mut k, &mut mem, &mut v, Sysno::Write, [fd as u64, content, 2, 0, 0, 0]);
        let addr = 0x1_3000u64;
        call(
            &mut k,
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
        let (mut k, mut mem, mut v) = setup();
        // No such fd -> EBADF.
        assert_eq!(
            call(&mut k, &mut mem, &mut v, Sysno::Mmap, [0x1_5000, 4, 1, MAP_FIXED, 99, 0]),
            -i64::from(Errno::EBADF.0)
        );
        // fd 1 is stdout, not a file -> EACCES.
        assert_eq!(
            call(&mut k, &mut mem, &mut v, Sysno::Mmap, [0x1_5000, 4, 1, MAP_FIXED, 1, 0]),
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
        let mut k = Kernel::new(Arch::Aarch64, mounts);
        k.cur.pid = 1;
        let mut mem = GuestMemory::new(0x1_0000, 16 * PAGE);
        mem.map(0x1_0000, 4 * PAGE, Prot::rw()).unwrap();
        let mut v = DummyVcpu;

        let path = 0x1_0000;
        mem.write_init(path, b"/work/probe\0").unwrap();
        let fd = call(
            &mut k,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_CWD, path, 0, 0, 0, 0],
        );
        assert!(fd >= 3, "open through hole failed: {fd}");
        let buf = 0x1_1000;
        assert_eq!(
            call(
                &mut k,
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
        let (mut k, mut mem, mut v) = setup();
        let tv = 0x1_0000;

        // gettimeofday writes a nonzero tv_sec.
        assert_eq!(
            call(
                &mut k,
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
                &mut k,
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
                &mut k,
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
                &mut k,
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
        let (mut k, mut mem, mut v) = setup();
        let path = 0x1_0000;
        mem.write_init(path, b"/a\0").unwrap();
        call(
            &mut k,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_CWD, path, 0o102, 0o644, 0, 0],
        );

        let root = 0x1_1000;
        mem.write_init(root, b"/\0").unwrap();
        let dirfd = call(
            &mut k,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_CWD, root, 0, 0, 0, 0],
        );
        let buf = 0x1_2000;
        let n = call(
            &mut k,
            &mut mem,
            &mut v,
            Sysno::Getdents64,
            [dirfd as u64, buf, PAGE, 0, 0, 0],
        );
        assert!(n > 0);

        let cbuf = 0x1_3000;
        let len = call(
            &mut k,
            &mut mem,
            &mut v,
            Sysno::Getcwd,
            [cbuf, 64, 0, 0, 0, 0],
        );
        assert_eq!(len, 2);
        assert_eq!(mem.read_vec(cbuf, 1).unwrap(), b"/");
    }

    #[test]
    fn rt_sigaction_stores_and_returns_old_handler() {
        let (mut k, mut mem, mut v) = setup();
        let act = 0x1_0000;
        let oldact = 0x1_0100;

        // Install handler 0xdead for SIGINT (2).
        mem.write_init(act, &0xdeadu64.to_le_bytes()).unwrap();
        assert_eq!(
            call(&mut k, &mut mem, &mut v, Sysno::RtSigaction, [2, act, 0, 8, 0, 0]),
            0
        );
        assert_eq!(k.cur.handlers[2], 0xdead);

        // Install 0xbeef and read back the previous (0xdead) via oldact.
        mem.write_init(act, &0xbeefu64.to_le_bytes()).unwrap();
        assert_eq!(
            call(
                &mut k,
                &mut mem,
                &mut v,
                Sysno::RtSigaction,
                [2, act, oldact, 8, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_u64(oldact).unwrap(), 0xdead);
        assert_eq!(k.cur.handlers[2], 0xbeef);
    }

    #[test]
    fn rt_sigaction_rejects_sigkill() {
        let (mut k, mut mem, mut v) = setup();
        let act = 0x1_0000;
        mem.write_init(act, &1u64.to_le_bytes()).unwrap();
        // SIGKILL (9) and SIGSTOP (19) dispositions cannot change.
        assert_eq!(
            call(&mut k, &mut mem, &mut v, Sysno::RtSigaction, [9, act, 0, 8, 0, 0]),
            -22
        );
        assert_eq!(
            call(&mut k, &mut mem, &mut v, Sysno::RtSigaction, [19, act, 0, 8, 0, 0]),
            -22
        );
    }

    #[test]
    fn rt_sigprocmask_setmask_and_readback() {
        let (mut k, mut mem, mut v) = setup();
        let set = 0x1_0000;
        let oldset = 0x1_0100;
        mem.write_init(set, &0b1010u64.to_le_bytes()).unwrap();

        // SIG_SETMASK (2) replaces the mask.
        assert_eq!(
            call(&mut k, &mut mem, &mut v, Sysno::RtSigprocmask, [2, set, 0, 8, 0, 0]),
            0
        );
        assert_eq!(k.cur.blocked, 0b1010);

        // Read it back through oldset (set == 0 leaves the mask unchanged).
        assert_eq!(
            call(
                &mut k,
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
        let (mut k, mut mem, mut v) = setup();
        // kill(pid 1 == self, SIGTERM=15) sets the pending bit.
        assert_eq!(
            call(&mut k, &mut mem, &mut v, Sysno::Kill, [1, 15, 0, 0, 0, 0]),
            0
        );
        assert_eq!(k.cur.pending, 1 << 14);

        // Default disposition of SIGTERM is TERMINATE -> exit code 128 + 15.
        k.deliver_pending_signals();
        assert!(matches!(k.cur.run, RunState::Zombie(143)));
    }

    #[test]
    fn kill_nonexistent_pid_is_esrch() {
        let (mut k, mut mem, mut v) = setup();
        assert_eq!(
            call(&mut k, &mut mem, &mut v, Sysno::Kill, [999, 15, 0, 0, 0, 0]),
            -3
        );
    }
}
