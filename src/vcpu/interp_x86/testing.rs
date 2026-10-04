//! Single-instruction access to the x86-64 interpreter for differential
//! testing (`tests/x86_diff.rs` runs the same instruction here and on real
//! x86-64 execution, then compares the full architectural state). Not a
//! stable API — `#[doc(hidden)]` and only meant for that harness.

use super::{Step, X86Interp};
use crate::vcpu::GuestMemory;

/// The architectural state a differential case loads and compares: the GPRs
/// (`RAX..R15` in encoding order), `RIP`, `RFLAGS`, and the x87/MMX/SSE state
/// as a 512-byte `FXSAVE64` image.
#[derive(Clone, Debug)]
pub struct CpuState {
    pub gpr: [u64; 16],
    pub rip: u64,
    pub rflags: u64,
    pub fxsave: [u8; 512],
    pub fs_base: u64,
    pub gs_base: u64,
}

impl Default for CpuState {
    fn default() -> Self {
        Self {
            gpr: [0; 16],
            rip: 0,
            rflags: 0x202,
            fxsave: [0; 512],
            fs_base: 0,
            gs_base: 0,
        }
    }
}

/// How one executed instruction ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Completed (fell through or branched); the state was written back.
    Done,
    /// `SYSCALL`.
    Syscall,
    /// `#UD` — the Linux `SIGILL`.
    Illegal,
    /// A page fault on `addr` — the Linux `SIGSEGV` (or `SIGBUS`).
    Fault { addr: u64, write: bool },
    /// A non-memory exception, as the Linux signal it is delivered as
    /// (`SIGFPE` = 8 for `#DE`/`#MF`/`#XM`, `SIGTRAP` = 5 for `INT3`,
    /// `SIGSEGV` = 11 for `#GP`/privileged instructions).
    Signal(i32),
}

/// Execute exactly one instruction at `st.rip` from state `st`, writing the
/// resulting state back into `st` (also on a fault, where it is the
/// unmodified pre-instruction state — x86 faults are precise).
pub fn step(mem: &mut GuestMemory, st: &mut CpuState) -> Outcome {
    let mut cpu = X86Interp::new(st.rip, st.gpr[4]);
    cpu.quantum = None;
    cpu.gpr = st.gpr;
    cpu.set_rflags_user(st.rflags);
    cpu.fs_base = st.fs_base;
    cpu.gs_base = st.gs_base;
    if !cpu.fxrstor_image(&st.fxsave, true) {
        return Outcome::Signal(11);
    }
    let out = match cpu.exec(mem) {
        Step::Next | Step::Branched => Outcome::Done,
        Step::Syscall => Outcome::Syscall,
        Step::Illegal => Outcome::Illegal,
        Step::Fault { addr, write } => Outcome::Fault { addr, write },
        Step::Trap(t) => Outcome::Signal(t.signal()),
    };
    st.gpr = cpu.gpr;
    st.rip = cpu.rip;
    st.rflags = cpu.rflags_word();
    st.fxsave = cpu.fxsave_image(true);
    out
}
