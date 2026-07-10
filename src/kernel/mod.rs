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
use std::io::Write;

use crate::abi::Arch;
use crate::abi::arch::{self, Sysno};
use crate::abi::errno::Errno;
use crate::fs::MountTable;
use crate::vcpu::mem::{PAGE_SIZE, Prot};
use crate::vcpu::{Exit, GuestMemory, Vcpu, VcpuError};

mod fd;

pub use fd::{Fd, FdTable};

/// One guest process's kernel-side state.
///
/// (Multi-process / threads arrive in Phase 6; for now this models a single
/// address space and fd table.)
pub struct Kernel {
    arch: Arch,
    #[allow(dead_code)] // wired into file syscalls in Phase 3/4
    mounts: MountTable,
    #[allow(dead_code)]
    fds: FdTable,
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
            brk: 0,
            heap_start: 0,
            heap_limit: 0,
            mmap_cursor: 0,
            mmap_floor: 0,
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
            Sysno::ExitGroup | Sysno::Exit => {
                self.exit_code = Some(args[0] as i32);
                0
            }
            Sysno::SetTidAddress => 1, // pretend tid == 1 for now
            // Everything else is not wired up yet. Record and return -ENOSYS so
            // the guest gets a well-formed failure rather than a crash.
            Sysno::Unknown(nr) => {
                *self.unsupported.entry(nr).or_default() += 1;
                err(Errno::ENOSYS)
            }
            _ => {
                *self.unsupported.entry(raw).or_default() += 1;
                err(Errno::ENOSYS)
            }
        }
    }

    /// `write(fd, buf, count)` — currently only the stdio sinks (fd 1/2). File
    /// and pipe/socket descriptors arrive in Phases 4/7/8.
    fn sys_write(&mut self, fd: u64, buf: u64, count: u64, mem: &GuestMemory) -> i64 {
        let Ok(data) = mem.read_vec(buf, count as usize) else {
            return err(Errno::EFAULT);
        };
        let sink: &mut dyn Write = match fd {
            1 => &mut *self.stdout,
            2 => &mut *self.stderr,
            _ => return err(Errno::EBADF),
        };
        match sink.write_all(&data) {
            Ok(()) => count as i64,
            Err(_) => err(Errno::EIO),
        }
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

    /// Syscalls the guest attempted that nixvm does not implement yet.
    #[must_use]
    pub fn unsupported(&self) -> &BTreeMap<u64, u64> {
        &self.unsupported
    }
}

/// Encode an errno as a negative syscall return.
const fn err(e: Errno) -> i64 {
    -(e.0 as i64)
}
