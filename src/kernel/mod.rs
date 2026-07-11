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

use crate::abi::Arch;
use crate::abi::arch::{self, Sysno};
use crate::abi::errno::Errno;
use crate::fs::{Attrs, MountTable, NodeKind};
use crate::loader::{ProcessSpec, load_static};
use crate::vcpu::mem::{PAGE_SIZE, Prot};
use crate::vcpu::{Exit, GuestMemory, Vcpu, VcpuError};

mod fd;
mod fs_ext;
mod net;
mod path;
mod stat;

pub use fd::{Fd, FdTable};
use net::Net;

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
    pid: i32,
    ppid: i32,
    run: RunState,
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
            run: RunState::Running,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum RunState {
    Running,
    Zombie(i32),
}

/// A guest process: its vcpu, address space, and per-process state.
struct Process {
    vcpu: Box<dyn Vcpu>,
    mem: GuestMemory,
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
    stdin: Box<dyn Read + Send>,
    stdout: Box<dyn Write + Send>,
    stderr: Box<dyn Write + Send>,
    rng_state: u64,
    trace: bool,
    /// The process file-creation mask (`umask`); global for our single session.
    umask: u32,
    unsupported: BTreeMap<u64, u64>,
    /// The running process's per-process state (swapped in for the slice).
    cur: ProcInfo,
    /// All processes; the running one is `take`n out during its slice, so its
    /// slot is temporarily `None` (making `fork`/`wait4` on the table clean).
    procs: Vec<Option<Process>>,
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
            stdin: Box::new(std::io::stdin()),
            stdout: Box::new(std::io::stdout()),
            stderr: Box::new(std::io::stderr()),
            rng_state: 0,
            trace: std::env::var_os("NIXVM_TRACE").is_some(),
            umask: 0o022,
            unsupported: BTreeMap::new(),
            cur: ProcInfo::default(),
            procs: Vec::new(),
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

    /// Run the machine: `vcpu`/`mem` become the initial process (pid 1), then
    /// the scheduler drives all processes until pid 1 exits. Returns pid 1's
    /// exit code.
    pub fn run(&mut self, vcpu: Box<dyn Vcpu>, mem: GuestMemory) -> Result<i32, VcpuError> {
        let mut info = std::mem::take(&mut self.cur);
        info.pid = 1;
        info.ppid = 0;
        info.run = RunState::Running;
        self.procs.push(Some(Process { vcpu, mem, info }));
        self.schedule()
    }

    /// Cooperative round-robin scheduler.
    fn schedule(&mut self) -> Result<i32, VcpuError> {
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
                std::mem::swap(&mut self.cur, &mut proc.info);
                let made = self.run_slice(&mut proc.vcpu, &mut proc.mem)?;
                std::mem::swap(&mut self.cur, &mut proc.info);
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
            match vcpu.run(mem)? {
                Exit::Syscall => {
                    let raw = vcpu.syscall_nr();
                    let sys = arch::decode(self.arch, raw);
                    let args = vcpu.syscall_args();
                    self.block = false;
                    self.exec_ok = false;
                    let ret = self.dispatch(sys, raw, &args, vcpu.as_mut(), mem);
                    if let RunState::Zombie(_) = self.cur.run {
                        return Ok(true);
                    }
                    if self.block {
                        return Ok(progressed);
                    }
                    if self.exec_ok {
                        progressed = true;
                        continue; // resume the new image at its entry
                    }
                    vcpu.set_syscall_ret(ret as u64);
                    progressed = true;
                }
                Exit::Interrupted => {}
                Exit::MemFault { addr, write } => {
                    eprintln!(
                        "[fault] pid {} memory fault at {addr:#x} (write={write})",
                        self.cur.pid
                    );
                    self.cur.run = RunState::Zombie(139);
                    return Ok(true);
                }
                Exit::IllegalInstruction { pc } => {
                    eprintln!(
                        "[fault] pid {} illegal instruction at {pc:#x}",
                        self.cur.pid
                    );
                    self.cur.run = RunState::Zombie(132);
                    return Ok(true);
                }
                Exit::Halt => {
                    self.cur.run = RunState::Zombie(0);
                    return Ok(true);
                }
            }
        }
    }

    /// The syscall table. Returns the value the guest sees in its result
    /// register: a non-negative result, or a negative errno.
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
            // sendto/recvfrom on a connected socket are just write/read.
            Sysno::Write | Sysno::Sendto => self.sys_write(args[0], args[1], args[2], mem),
            Sysno::Read | Sysno::Recvfrom => self.sys_read(args[0], args[1], args[2], mem),
            Sysno::Brk => self.sys_brk(args[0], mem),
            Sysno::Mmap => self.sys_mmap(args, mem),
            Sysno::Munmap => self.sys_munmap(args[0], args[1], mem),
            Sysno::Mprotect => self.sys_mprotect(args[0], args[1], args[2], mem),
            Sysno::Uname => self.sys_uname(args[0], mem),
            Sysno::ClockGettime => sys_clock_gettime(args[1], mem),
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
            Sysno::Futex => sys_futex(args, mem),
            Sysno::Pipe2 => self.sys_pipe2(args[0], mem),
            Sysno::Socket => self.sys_socket(args[0], args[1], args[2]),
            Sysno::Socketpair => self.sys_socketpair(args[0], args[1], args[2], args[3], mem),
            Sysno::Bind => self.sys_bind(args[0], args[1], args[2], mem),
            Sysno::Listen => self.sys_listen(args[0]),
            Sysno::Accept4 => self.sys_accept4(args[0], args[1], args[2], args[3], mem),
            Sysno::Connect => self.sys_connect(args[0], args[1], args[2], mem),
            Sysno::Getsockname => self.sys_getsockname(args[0], args[1], args[2], mem),
            Sysno::Getpeername => self.sys_getpeername(args[0], args[1], args[2], mem),
            Sysno::Shutdown => self.sys_shutdown(args[0], args[1]),
            Sysno::Dup => self.sys_dup(args[0]),
            Sysno::Dup2 | Sysno::Dup3 => self.sys_dup2(args[0], args[1]),
            Sysno::Clone => self.sys_clone(args, vcpu, mem),
            Sysno::Execve => self.sys_execve(args[0], args[1], args[2], vcpu, mem),
            Sysno::Wait4 => self.sys_wait4(args[0] as i64, args[1], args[2], mem),
            Sysno::ExitGroup | Sysno::Exit => self.sys_exit(args[0] as i32),
            // pid/tid = this process's pid (single-thread so tid == pid);
            // set_tid_address also returns the tid.
            Sysno::Getpid | Sysno::Gettid | Sysno::SetTidAddress => i64::from(self.cur.pid),
            Sysno::Getppid => i64::from(self.cur.ppid),
            // Succeed as root / no-op: uid queries, signal setup, robust list,
            // permission/ownership/timestamp changes, and socket options —
            // none modeled yet.
            Sysno::Getuid
            | Sysno::Geteuid
            | Sysno::Getgid
            | Sysno::Getegid
            | Sysno::RtSigaction
            | Sysno::RtSigprocmask
            | Sysno::SetRobustList
            | Sysno::Fchmodat
            | Sysno::Fchmod
            | Sysno::Fchownat
            | Sysno::Fchown
            | Sysno::Utimensat
            | Sysno::Setsockopt
            | Sysno::Getsockopt => 0,
            _ => {
                *self.unsupported.entry(raw).or_default() += 1;
                err(Errno::ENOSYS)
            }
        }
    }

    // ---- process lifecycle ------------------------------------------------

    /// `clone(flags, stack, ...)` — implemented as `fork` (a full copy of the
    /// address space, fd table, and registers). Threads that share memory
    /// (`CLONE_THREAD`) arrive later; the fork/vfork-then-exec pattern that
    /// shells use works with a copy.
    fn sys_clone(&mut self, args: &[u64; 6], vcpu: &mut dyn Vcpu, mem: &mut GuestMemory) -> i64 {
        let stack = args[1];
        let child_mem = mem.clone();
        let mut child_vcpu = vcpu.fork();
        let mut info = self.cur.clone();
        let pid = self.next_pid;
        self.next_pid += 1;
        info.pid = pid;
        info.ppid = self.cur.pid;
        info.run = RunState::Running;

        // The child holds copies of every open fd; bump pipe refcounts.
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

        if stack != 0 {
            child_vcpu.set_sp(stack);
        }
        child_vcpu.set_syscall_ret(0); // child returns 0 and advances past the svc
        self.procs.push(Some(Process {
            vcpu: child_vcpu,
            mem: child_mem,
            info,
        }));
        i64::from(pid)
    }

    /// `execve(path, argv, envp)` — replace the process image with a new static
    /// ELF read from the mount table (following symlinks).
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
        let Ok(img) = load_static(&mut new_mem, &elf, &spec) else {
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
            if p.info.ppid == cur {
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

    /// `exit`/`exit_group` — close all fds (so pipe peers see EOF) and become a
    /// zombie until the parent reaps us.
    fn sys_exit(&mut self, code: i32) -> i64 {
        for fd in self.cur.fds.drain() {
            self.bump_pipe(&fd, false);
        }
        self.cur.run = RunState::Zombie(code & 0xff);
        0
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
            Some(Fd::Stdin | Fd::Stdout | Fd::Stderr) => stat::char_device_attrs(),
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

    /// `mmap(addr, len, prot, flags, fd, off)` — anonymous mappings only.
    fn sys_mmap(&mut self, a: &[u64; 6], mem: &mut GuestMemory) -> i64 {
        const MAP_FIXED: u64 = 0x10;
        const MAP_ANONYMOUS: u64 = 0x20;

        let (addr, len, prot, flags) = (a[0], a[1], a[2], a[3]);
        if len == 0 {
            return err(Errno::EINVAL);
        }
        if flags & MAP_ANONYMOUS == 0 {
            return err(Errno::ENOSYS);
        }
        let len = len.div_ceil(PAGE_SIZE) * PAGE_SIZE;
        let prot = Prot((prot as u8) & 0x7);
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
        base as i64
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

/// `futex(uaddr, op, val, ...)` — the single-thread subset.
fn sys_futex(args: &[u64; 6], mem: &GuestMemory) -> i64 {
    const FUTEX_WAIT: u64 = 0;
    let uaddr = args[0];
    let op = args[1] & 0x7f;
    let val = args[2] as u32;
    match op {
        FUTEX_WAIT => match mem.read_u32(uaddr) {
            Ok(cur) if cur != val => err(Errno::EAGAIN),
            Ok(_) => 0,
            Err(_) => err(Errno::EFAULT),
        },
        _ => 0,
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

    const PAGE: u64 = 4096;
    const AT_CWD: u64 = (-100i64) as u64;

    fn setup() -> (Kernel, GuestMemory, DummyVcpu) {
        let mut mounts = MountTable::new();
        mounts.mount("/", Box::new(TmpFs::new()));
        let mut kernel = Kernel::new(Arch::Aarch64, mounts);
        kernel.cur.pid = 1;
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
}
