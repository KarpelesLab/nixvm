//! The nixvm "kernel": an arch-agnostic engine that services guest syscalls.
//!
//! Following the engine/adapter split proven in univdreams, everything here is
//! written in terms of the normalized [`crate::abi::arch::Sysno`] and the
//! [`crate::vcpu::Vcpu`] / [`crate::vcpu::GuestMemory`] seams. The backend and
//! guest arch stay invisible to the handlers.
//!
//! The core is the run/serve loop in [`Kernel::run`]: run the vcpu until it
//! traps on a syscall, decode + dispatch it, write the return value, repeat —
//! until the guest calls `exit_group`.
//!
//! Handlers are stubs in the scaffold; they come online across ROADMAP phases:
//! Phase 3 (files/stat/tty), Phase 6 (clone/futex/signals), Phase 8 (sockets).

use std::collections::BTreeMap;
use std::io::{self, Write};

use crate::abi::Arch;
use crate::abi::arch::{self, Sysno};
use crate::abi::errno::Errno;
use crate::fs::{Attrs, MountTable, NodeKind};
use crate::vcpu::mem::{PAGE_SIZE, Prot};
use crate::vcpu::{Exit, GuestMemory, Vcpu, VcpuError};

mod fd;
mod path;
mod stat;

pub use fd::{Fd, FdTable};

/// `dirfd` value meaning "resolve relative to the current working directory".
const AT_FDCWD: i64 = -100;

/// One guest process's kernel-side state.
///
/// (Multi-process / threads arrive in Phase 6; for now this models a single
/// address space and fd table.)
pub struct Kernel {
    arch: Arch,
    mounts: MountTable,
    fds: FdTable,
    /// Current working directory (absolute, normalized).
    cwd: String,
    /// Current program break (top of the heap).
    brk: u64,
    /// Lowest heap address (the program break at start-up); `brk` never drops
    /// below this.
    heap_start: u64,
    /// Upper bound the heap may not grow past (start of the mmap/stack area).
    heap_limit: u64,
    /// Downward-growing cursor for anonymous `mmap` allocations.
    mmap_cursor: u64,
    /// Lowest address `mmap` may reach.
    mmap_floor: u64,
    /// `getrandom` PRNG state (xorshift; seeded lazily from the host clock).
    rng_state: u64,
    /// Sinks for guest fd 1 and 2. Configurable so callers (and tests) can
    /// capture or redirect guest output.
    stdout: Box<dyn Write + Send>,
    stderr: Box<dyn Write + Send>,
    /// Set by `exit`/`exit_group`; ends the run loop.
    exit_code: Option<i32>,
    /// Raw guest syscall numbers we don't handle yet, with hit counts — an
    /// honest "what's missing" ledger surfaced at shutdown.
    unsupported: BTreeMap<u64, u64>,
}

impl std::fmt::Debug for Kernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Kernel")
            .field("arch", &self.arch)
            .field("exit_code", &self.exit_code)
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
            fds: FdTable::with_standard_streams(),
            cwd: "/".to_string(),
            brk: 0,
            heap_start: 0,
            heap_limit: 0,
            mmap_cursor: 0,
            mmap_floor: 0,
            rng_state: 0,
            stdout: Box::new(std::io::stdout()),
            stderr: Box::new(std::io::stderr()),
            exit_code: None,
            unsupported: BTreeMap::new(),
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

    /// Set the heap window: `start` is the initial program break (page-aligned,
    /// just past the loaded image) and `limit` is the highest address the heap
    /// may reach (the bottom of the mmap/stack area).
    pub fn set_heap(&mut self, start: u64, limit: u64) {
        self.heap_start = start;
        self.brk = start;
        self.heap_limit = limit;
    }

    /// Set the anonymous-`mmap` arena: allocations grow down from `top` and may
    /// not drop below `floor`.
    pub fn set_mmap_area(&mut self, top: u64, floor: u64) {
        self.mmap_cursor = top;
        self.mmap_floor = floor;
    }

    /// Set the initial current working directory (absolute path).
    pub fn set_cwd(&mut self, dir: impl Into<String>) {
        self.cwd = path::normalize(&dir.into());
    }

    /// Drive one vcpu until the guest exits, returning its exit code.
    pub fn run(&mut self, vcpu: &mut dyn Vcpu, mem: &mut GuestMemory) -> Result<i32, VcpuError> {
        loop {
            match vcpu.run(mem)? {
                Exit::Syscall => {
                    let raw = vcpu.syscall_nr();
                    let sys = arch::decode(self.arch, raw);
                    let args = vcpu.syscall_args();
                    let ret = self.dispatch(sys, raw, &args, vcpu, mem);
                    if let Some(code) = self.exit_code {
                        return Ok(code);
                    }
                    vcpu.set_syscall_ret(ret as u64);
                }
                Exit::Interrupted => { /* scheduler hook (Phase 6) */ }
                Exit::MemFault { addr, write } => {
                    // Phase 6 turns this into SIGSEGV delivery; for now it's fatal.
                    return Err(VcpuError::Backend(format!(
                        "guest memory fault at {addr:#x} (write={write})"
                    )));
                }
                Exit::IllegalInstruction { pc } => {
                    return Err(VcpuError::Backend(format!("illegal instruction at {pc:#x}")));
                }
                Exit::Halt => return Ok(self.exit_code.unwrap_or(0)),
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
        _vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        match sys {
            Sysno::Write => self.sys_write(args[0], args[1], args[2], mem),
            Sysno::Brk => self.sys_brk(args[0], mem),
            Sysno::Mmap => self.sys_mmap(args, mem),
            Sysno::Munmap => self.sys_munmap(args[0], args[1], mem),
            Sysno::Mprotect => self.sys_mprotect(args[0], args[1], args[2], mem),
            Sysno::Uname => self.sys_uname(args[0], mem),
            Sysno::ClockGettime => sys_clock_gettime(args[1], mem),
            Sysno::Openat => self.sys_openat(args[0] as i64, args[1], args[2], args[3], mem),
            Sysno::Read => self.sys_read(args[0], args[1], args[2], mem),
            Sysno::Close => self.sys_close(args[0] as i32),
            Sysno::Lseek => self.sys_lseek(args[0], args[1] as i64, args[2]),
            Sysno::Fstat => self.sys_fstat(args[0], args[1], mem),
            Sysno::Newfstatat => {
                self.sys_newfstatat(args[0] as i64, args[1], args[2], args[3], mem)
            }
            Sysno::Getdents64 => self.sys_getdents64(args[0], args[1], args[2], mem),
            Sysno::Getcwd => self.sys_getcwd(args[0], args[1], mem),
            Sysno::Chdir => self.sys_chdir(args[0], mem),
            Sysno::Writev => self.sys_writev(args[0], args[1], args[2], mem),
            Sysno::Getrandom => self.sys_getrandom(args[0], args[1], mem),
            // Terminal ioctls: report "not a tty" for our plain stdio pipes.
            Sysno::Ioctl => err(Errno::ENOTTY),
            Sysno::ExitGroup | Sysno::Exit => {
                self.exit_code = Some(args[0] as i32);
                0
            }
            // Single-process identity: pid/tid 1, running as root.
            Sysno::SetTidAddress | Sysno::Getpid | Sysno::Gettid => 1,
            // ppid 0, uid/gid 0, and signal setup succeeding (delivery is
            // Phase 6, so nothing fires yet) — all return 0.
            Sysno::Getppid
            | Sysno::Getuid
            | Sysno::Geteuid
            | Sysno::Getgid
            | Sysno::Getegid
            | Sysno::RtSigaction
            | Sysno::RtSigprocmask => 0,
            // Not wired up yet (Unknown or an unhandled variant). Record the
            // raw number and return -ENOSYS so the guest gets a well-formed
            // failure rather than a crash.
            _ => {
                *self.unsupported.entry(raw).or_default() += 1;
                err(Errno::ENOSYS)
            }
        }
    }

    /// `write(fd, buf, count)` — stdio sinks (fd 1/2) and open files.
    fn sys_write(&mut self, fd: u64, buf: u64, count: u64, mem: &GuestMemory) -> i64 {
        let Ok(data) = mem.read_vec(buf, count as usize) else {
            return err(Errno::EFAULT);
        };
        match fd {
            1 => match self.stdout.write_all(&data) {
                Ok(()) => count as i64,
                Err(_) => err(Errno::EIO),
            },
            2 => match self.stderr.write_all(&data) {
                Ok(()) => count as i64,
                Err(_) => err(Errno::EIO),
            },
            _ => {
                let Some(Fd::File { path, offset }) = self.fds.get(fd as i32).cloned() else {
                    return err(Errno::EBADF);
                };
                match self.mounts.write_at(&path, offset, &data) {
                    Ok(n) => {
                        if let Some(Fd::File { offset, .. }) = self.fds.get_mut(fd as i32) {
                            *offset += n as u64;
                        }
                        n as i64
                    }
                    Err(e) => io_errno(&e),
                }
            }
        }
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
                break; // short write
            }
        }
        total
    }

    /// `getrandom(buf, len, flags)` — fill `buf` with pseudorandom bytes.
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

    /// `read(fd, buf, count)` — stdin (currently EOF) and open files.
    fn sys_read(&mut self, fd: u64, buf: u64, count: u64, mem: &mut GuestMemory) -> i64 {
        let (path, offset) = match self.fds.get(fd as i32) {
            Some(Fd::File { path, offset }) => (path.clone(), *offset),
            Some(Fd::Stdin) => return 0, // no console input yet
            _ => return err(Errno::EBADF),
        };
        let mut tmp = vec![0u8; count as usize];
        match self.mounts.read_at(&path, offset, &mut tmp) {
            Ok(n) => {
                if mem.write(buf, &tmp[..n]).is_err() {
                    return err(Errno::EFAULT);
                }
                if let Some(Fd::File { offset, .. }) = self.fds.get_mut(fd as i32) {
                    *offset += n as u64;
                }
                n as i64
            }
            Err(e) => io_errno(&e),
        }
    }

    /// `openat(dirfd, path, flags, mode)` against the mount table.
    fn sys_openat(&mut self, dirfd: i64, pathptr: u64, flags: u64, mode: u64, mem: &GuestMemory) -> i64 {
        const O_CREAT: u64 = 0o100;
        const O_TRUNC: u64 = 0o1000;

        let Some(rel) = read_path(mem, pathptr) else {
            return err(Errno::EFAULT);
        };
        let abs = self.resolve_path(dirfd, &rel);

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
            self.fds.alloc(Fd::Dir { path: abs, pos: 0 })
        } else {
            self.fds.alloc(Fd::File { path: abs, offset: 0 })
        };
        i64::from(fd)
    }

    /// `close(fd)`.
    fn sys_close(&mut self, fd: i32) -> i64 {
        match self.fds.close(fd) {
            Some(_) => 0,
            None => err(Errno::EBADF),
        }
    }

    /// `lseek(fd, offset, whence)`.
    fn sys_lseek(&mut self, fd: u64, offset: i64, whence: u64) -> i64 {
        let (cur, path) = match self.fds.get(fd as i32) {
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
        if let Some(Fd::File { offset, .. }) = self.fds.get_mut(fd as i32) {
            *offset = newpos as u64;
        }
        newpos
    }

    /// `fstat(fd, statbuf)`.
    fn sys_fstat(&mut self, fd: u64, statbuf: u64, mem: &mut GuestMemory) -> i64 {
        let attrs = match self.fds.get(fd as i32) {
            Some(Fd::File { path, .. } | Fd::Dir { path, .. }) => {
                let path = path.clone();
                match self.mounts.stat(&path) {
                    Some(a) => a,
                    None => return err(Errno::ENOENT),
                }
            }
            Some(Fd::Stdin | Fd::Stdout | Fd::Stderr) => stat::char_device_attrs(),
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
        _flags: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some(rel) = read_path(mem, pathptr) else {
            return err(Errno::EFAULT);
        };
        let abs = self.resolve_path(dirfd, &rel);
        let Some(attrs) = self.mounts.stat(&abs) else {
            return err(Errno::ENOENT);
        };
        write_stat_or_fault(mem, statbuf, &attrs)
    }

    /// `getdents64(fd, buf, count)` — encode directory entries.
    fn sys_getdents64(&mut self, fd: u64, buf: u64, count: u64, mem: &mut GuestMemory) -> i64 {
        let (path, pos) = match self.fds.get(fd as i32) {
            Some(Fd::Dir { path, pos }) => (path.clone(), *pos),
            _ => return err(Errno::ENOTDIR),
        };
        let entries = match self.mounts.readdir(&path) {
            Ok(e) => e,
            Err(e) => return io_errno(&e),
        };
        // "." and ".." precede the real children.
        let mut all: Vec<(String, NodeKind, u64)> =
            vec![(".".into(), NodeKind::Dir, 1), ("..".into(), NodeKind::Dir, 1)];
        all.extend(entries.into_iter().map(|e| (e.name, e.kind, e.inode)));

        let (bytes, consumed) = stat::encode_dirents(&all, pos, count as usize);
        if bytes.is_empty() && pos < all.len() {
            return err(Errno::EINVAL); // buffer too small for even one entry
        }
        if mem.write(buf, &bytes).is_err() {
            return err(Errno::EFAULT);
        }
        if let Some(Fd::Dir { pos, .. }) = self.fds.get_mut(fd as i32) {
            *pos = consumed;
        }
        bytes.len() as i64
    }

    /// `getcwd(buf, size)` — returns the length including the NUL terminator.
    fn sys_getcwd(&mut self, buf: u64, size: u64, mem: &mut GuestMemory) -> i64 {
        let mut bytes = self.cwd.clone().into_bytes();
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
                self.cwd = abs;
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
            self.cwd.clone()
        } else {
            match self.fds.get(dirfd as i32) {
                Some(Fd::Dir { path, .. } | Fd::File { path, .. }) => path.clone(),
                _ => self.cwd.clone(),
            }
        };
        path::normalize(&format!("{base}/{p}"))
    }

    /// `brk(addr)` — move the program break. Returns the new break on success,
    /// or the unchanged break on failure (the Linux convention; libc computes
    /// success by comparing the result to what it asked for). `brk(0)` queries.
    fn sys_brk(&mut self, addr: u64, mem: &mut GuestMemory) -> i64 {
        if addr == 0 || addr < self.heap_start {
            return self.brk as i64;
        }
        if addr > self.brk {
            // Grow: map the pages newly covered by [old_brk, addr).
            let from = self.brk - self.brk % PAGE_SIZE;
            let to = addr.div_ceil(PAGE_SIZE) * PAGE_SIZE;
            if to > self.heap_limit || mem.map(from, to - from, Prot::rw()).is_err() {
                return self.brk as i64; // failure: break unchanged
            }
        } else if addr < self.brk {
            // Shrink: release whole pages above the new break.
            let from = addr.div_ceil(PAGE_SIZE) * PAGE_SIZE;
            let to = self.brk.div_ceil(PAGE_SIZE) * PAGE_SIZE;
            if to > from {
                let _ = mem.unmap(from, to - from);
            }
        }
        self.brk = addr;
        self.brk as i64
    }

    /// `mmap(addr, len, prot, flags, fd, off)` — anonymous mappings only for
    /// now (file-backed mappings arrive with dynamic linking, Phase 5).
    /// Non-fixed anonymous requests are placed in a downward-growing arena.
    fn sys_mmap(&mut self, a: &[u64; 6], mem: &mut GuestMemory) -> i64 {
        const MAP_FIXED: u64 = 0x10;
        const MAP_ANONYMOUS: u64 = 0x20;

        let (addr, len, prot, flags) = (a[0], a[1], a[2], a[3]);
        if len == 0 {
            return err(Errno::EINVAL);
        }
        if flags & MAP_ANONYMOUS == 0 {
            return err(Errno::ENOSYS); // file-backed mmap: Phase 5
        }
        let len = len.div_ceil(PAGE_SIZE) * PAGE_SIZE;
        let prot = Prot((prot as u8) & 0x7);

        let base = if flags & MAP_FIXED != 0 && addr != 0 {
            addr - addr % PAGE_SIZE
        } else {
            let Some(new_top) = self.mmap_cursor.checked_sub(len) else {
                return err(Errno::ENOMEM);
            };
            if new_top < self.mmap_floor {
                return err(Errno::ENOMEM);
            }
            self.mmap_cursor = new_top;
            new_top
        };
        if mem.map(base, len, prot).is_err() {
            return err(Errno::ENOMEM);
        }
        base as i64
    }

    /// `munmap(addr, len)` — release the covered pages.
    #[allow(clippy::unused_self)] // will reclaim arena space / update accounting later
    fn sys_munmap(&mut self, addr: u64, len: u64, mem: &mut GuestMemory) -> i64 {
        if len == 0 {
            return err(Errno::EINVAL);
        }
        let len = len.div_ceil(PAGE_SIZE) * PAGE_SIZE;
        let _ = mem.unmap(addr - addr % PAGE_SIZE, len);
        0
    }

    /// `mprotect(addr, len, prot)` — change protection on mapped pages.
    #[allow(clippy::unused_self)] // stays a method alongside the other mm syscalls
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

    /// `uname(buf)` — fill a `struct utsname` (six 65-byte NUL-padded fields).
    fn sys_uname(&self, buf: u64, mem: &mut GuestMemory) -> i64 {
        const FIELD: usize = 65;
        let mut data = [0u8; FIELD * 6];
        let fields: [&[u8]; 6] = [
            b"Linux",                    // sysname
            b"nixvm",                    // nodename
            b"6.1.0-nixvm",              // release
            b"#1 nixvm",                 // version
            self.arch.as_str().as_bytes(), // machine
            b"(none)",                   // domainname
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

/// `clock_gettime(clk_id, timespec)` — write host wall-clock time as a 16-byte
/// `struct timespec { i64 tv_sec; i64 tv_nsec; }`. All clock ids report the
/// same host time for now (per-clock semantics arrive with Phase 9 deadlines).
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

    /// A no-op vcpu: the file syscalls don't touch CPU registers.
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
    }

    const PAGE: u64 = 4096;
    const AT_CWD: u64 = (-100i64) as u64;

    fn setup() -> (Kernel, GuestMemory, DummyVcpu) {
        let mut mounts = MountTable::new();
        mounts.mount("/", Box::new(TmpFs::new()));
        let kernel = Kernel::new(Arch::Aarch64, mounts);
        let mut mem = GuestMemory::new(0x1_0000, 16 * PAGE);
        mem.map(0x1_0000, 4 * PAGE, Prot::rw()).unwrap();
        (kernel, mem, DummyVcpu)
    }

    fn call(k: &mut Kernel, mem: &mut GuestMemory, v: &mut DummyVcpu, s: Sysno, a: [u64; 6]) -> i64 {
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

        // openat(AT_FDCWD, "/f", O_CREAT|O_RDWR, 0644)
        let fd = call(&mut k, &mut mem, &mut v, Sysno::Openat, [AT_CWD, path, 0o102, 0o644, 0, 0]);
        assert_eq!(fd, 3);
        let fd = fd as u64;

        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Write, [fd, msg, 2, 0, 0, 0]), 2);
        // lseek(fd, 0, SEEK_SET)
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Lseek, [fd, 0, 0, 0, 0, 0]), 0);
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Read, [fd, buf, 2, 0, 0, 0]), 2);
        assert_eq!(mem.read_vec(buf, 2).unwrap(), b"Hi");

        // fstat: st_size (offset 48) == 2
        let stbuf = 0x1_3000;
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Fstat, [fd, stbuf, 0, 0, 0, 0]), 0);
        assert_eq!(mem.read_u64(stbuf + 48).unwrap(), 2);

        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Close, [fd, 0, 0, 0, 0, 0]), 0);
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

        // Two data buffers and a two-entry iovec array.
        let d0 = 0x1_0000;
        let d1 = 0x1_0010;
        let iov = 0x1_0100;
        mem.write_init(d0, b"foo").unwrap();
        mem.write_init(d1, b"bar!").unwrap();
        mem.write_init(iov, &d0.to_le_bytes()).unwrap();
        mem.write_init(iov + 8, &3u64.to_le_bytes()).unwrap();
        mem.write_init(iov + 16, &d1.to_le_bytes()).unwrap();
        mem.write_init(iov + 24, &4u64.to_le_bytes()).unwrap();

        // writev(1, iov, 2)
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Writev, [1, iov, 2, 0, 0, 0]), 7);
        assert_eq!(&*cap.lock().unwrap(), b"foobar!");
    }

    #[test]
    fn getrandom_fills_buffer() {
        let (mut k, mut mem, mut v) = setup();
        let buf = 0x1_0000;
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Getrandom, [buf, 16, 0, 0, 0, 0]), 16);
        // Extremely unlikely to be all-zero after fill.
        assert!(mem.read_vec(buf, 16).unwrap().iter().any(|&b| b != 0));
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
        let mut mem = GuestMemory::new(0x1_0000, 16 * PAGE);
        mem.map(0x1_0000, 4 * PAGE, Prot::rw()).unwrap();
        let mut v = DummyVcpu;

        let path = 0x1_0000;
        mem.write_init(path, b"/work/probe\0").unwrap();
        let fd = call(&mut k, &mut mem, &mut v, Sysno::Openat, [AT_CWD, path, 0, 0, 0, 0]);
        assert!(fd >= 3, "open through hole failed: {fd}");
        let buf = 0x1_1000;
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Read, [fd as u64, buf, 1, 0, 0, 0]), 1);
        assert_eq!(mem.read_vec(buf, 1).unwrap(), b"Z");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn getdents_and_getcwd() {
        let (mut k, mut mem, mut v) = setup();
        // create a file, then open "/" and list it.
        let path = 0x1_0000;
        mem.write_init(path, b"/a\0").unwrap();
        call(&mut k, &mut mem, &mut v, Sysno::Openat, [AT_CWD, path, 0o102, 0o644, 0, 0]);

        let root = 0x1_1000;
        mem.write_init(root, b"/\0").unwrap();
        let dirfd = call(&mut k, &mut mem, &mut v, Sysno::Openat, [AT_CWD, root, 0, 0, 0, 0]);
        let buf = 0x1_2000;
        let n = call(&mut k, &mut mem, &mut v, Sysno::Getdents64, [dirfd as u64, buf, PAGE, 0, 0, 0]);
        assert!(n > 0, "directory listing should be non-empty");

        // getcwd -> "/"
        let cbuf = 0x1_3000;
        let len = call(&mut k, &mut mem, &mut v, Sysno::Getcwd, [cbuf, 64, 0, 0, 0, 0]);
        assert_eq!(len, 2); // "/" + NUL
        assert_eq!(mem.read_vec(cbuf, 1).unwrap(), b"/");
    }
}
