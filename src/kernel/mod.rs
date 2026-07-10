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

use crate::abi::Arch;
use crate::abi::arch::{self, Sysno};
use crate::abi::errno::Errno;
use crate::fs::MountTable;
use crate::vcpu::{Exit, GuestMemory, Vcpu, VcpuError};

mod fd;

pub use fd::{Fd, FdTable};

/// One guest process's kernel-side state.
///
/// (Multi-process / threads arrive in Phase 6; for now this models a single
/// address space and fd table.)
#[derive(Debug)]
pub struct Kernel {
    arch: Arch,
    #[allow(dead_code)] // wired into file syscalls in Phase 3/4
    mounts: MountTable,
    #[allow(dead_code)]
    fds: FdTable,
    /// Program break for `brk`.
    #[allow(dead_code)]
    brk: u64,
    /// Bump pointer for anonymous `mmap`.
    #[allow(dead_code)]
    mmap_top: u64,
    /// Set by `exit`/`exit_group`; ends the run loop.
    exit_code: Option<i32>,
    /// Raw guest syscall numbers we don't handle yet, with hit counts — an
    /// honest "what's missing" ledger surfaced at shutdown.
    unsupported: BTreeMap<u64, u64>,
}

impl Kernel {
    #[must_use]
    pub fn new(arch: Arch, mounts: MountTable) -> Self {
        Self {
            arch,
            mounts,
            fds: FdTable::with_standard_streams(),
            brk: 0,
            mmap_top: 0,
            exit_code: None,
            unsupported: BTreeMap::new(),
        }
    }

    /// Drive one vcpu until the guest exits, returning its exit code.
    pub fn run(&mut self, vcpu: &mut dyn Vcpu, mem: &mut GuestMemory) -> Result<i32, VcpuError> {
        loop {
            match vcpu.run()? {
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
        _mem: &mut GuestMemory,
    ) -> i64 {
        match sys {
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
