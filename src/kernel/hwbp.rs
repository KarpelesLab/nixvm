//! The x86 debug registers a debugger gives a task (`u_debugreg` in the
//! ptrace `struct user`), as Linux keeps them: a breakpoint per `DR0`..`DR3`
//! once something is written there (`thread.ptrace_bps`), the `DR7` the
//! debugger last wrote successfully (`ptrace_dr7`), and a virtual `DR6` the
//! traps it causes set bits in (`virtual_dr6`). Writes are validated as
//! `ptrace_set_debugreg` and `ptrace_write_dr7` do, through the generic
//! breakpoint layer's checks (`arch_bp_generic_fields`,
//! `hw_breakpoint_arch_parse`, user addresses only). The enabled ones are
//! loaded into the task's vcpu ([`crate::vcpu::Vcpu::set_debug_regs`]),
//! which reports a hit as [`crate::vcpu::Exit::DebugTrap`].
//!
//! They belong to the task, not to its tracer: they stay across a detach,
//! a new task starts without any (`copy_thread`), and `execve` drops them
//! (`flush_thread`).

use super::err;
use crate::abi::errno::Errno;

/// `DR_CONTROL_RESERVED` (x86-64): `DR7` bits a write ignores.
const DR_CONTROL_RESERVED: u64 = 0xFFFF_FFFF_0000_FC00;
/// `DR6_RESERVED`: the bits that read as one; the virtual `DR6` is kept
/// with them flipped (`virtual_dr6 ^ DR6_RESERVED` is what reads back).
const DR6_RESERVED: u64 = 0xFFFF_0FF0;
/// `DR_STEP`: the trap is a single step.
pub(super) const DR_STEP: u64 = 0x4000;
/// `DR_TRAP0`..`DR_TRAP3`.
const DR_TRAP_BITS: u64 = 0xf;
/// `TASK_SIZE_MAX` with 4-level paging: no user breakpoint at or past it.
const TASK_SIZE_MAX: u64 = (1 << 47) - 4096;

// `X86_BREAKPOINT_LEN_*` and `X86_BREAKPOINT_*` (`asm/hw_breakpoint.h`):
// a `DR7` LEN field `| 0x40`, R/W field `| 0x80`.
const LEN_1: u8 = 0x40;
const LEN_2: u8 = 0x44;
const LEN_8: u8 = 0x48;
const LEN_4: u8 = 0x4c;
const EXECUTE: u8 = 0x80;
const WRITE: u8 = 0x81;
const RW: u8 = 0x83;

/// One of `DR0`..`DR3`'s breakpoints: its address, `X86_BREAKPOINT_LEN_*`
/// and `X86_BREAKPOINT_*` type, and whether `DR7` leaves it disabled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Bp {
    addr: u64,
    len: u8,
    ty: u8,
    disabled: bool,
}

/// A breakpoint at `addr` of x86 `len` and `ty`, if the breakpoint layer
/// takes it: `arch_bp_generic_fields` turns them into a generic length and
/// type, and `hw_breakpoint_parse` checks those (an execute breakpoint is
/// `sizeof(long)` long, the address is aligned to the length, and all of it
/// is a user address, since ptrace's exclude the kernel). `EINVAL` if not.
fn parse(addr: u64, len: u8, ty: u8) -> Result<(), i64> {
    let einval = Err(err(Errno::EINVAL));
    let size: u64 = match len {
        LEN_1 => 1,
        LEN_2 => 2,
        LEN_4 => 4,
        LEN_8 => 8,
        _ => return einval,
    };
    // An instruction breakpoint is `X86_BREAKPOINT_LEN_X` (the 1-byte code)
    // and nothing else; it checks as one byte.
    match ty {
        EXECUTE if len != LEN_1 => return einval,
        EXECUTE | WRITE | RW => {}
        _ => return einval,
    }
    let Some(end) = addr.checked_add(size - 1) else {
        return einval;
    };
    if addr & (size - 1) != 0 || addr >= TASK_SIZE_MAX || end >= TASK_SIZE_MAX {
        return einval;
    }
    Ok(())
}

/// The LEN and R/W fields of breakpoint `i` in `dr7`, as x86 codes, and
/// whether it's enabled (`decode_dr7`).
fn decode_dr7(dr7: u64, i: usize) -> (u8, u8, bool) {
    let info = (dr7 >> (16 + 4 * i)) as u8;
    ((info & 0xc) | 0x40, (info & 0x3) | 0x80, (dr7 >> (2 * i)) & 3 != 0)
}

/// A task's debug registers.
#[derive(Clone, Debug, Default)]
pub(super) struct DebugRegs {
    bps: [Option<Bp>; 4],
    /// `virtual_dr6`: what the traps said, with the reserved bits flipped.
    virtual_dr6: u64,
    /// `ptrace_dr7`: the last `DR7` written without error, as written.
    dr7: u64,
    /// The breakpoints changed since the vcpu last had them.
    pub(super) dirty: bool,
}

impl DebugRegs {
    /// `ptrace_get_debugreg(n)`: a breakpoint's address, the virtual `DR6`,
    /// `DR7`; 0 for `DR4` and `DR5`.
    pub(super) fn get(&self, n: u64) -> u64 {
        match n {
            0..=3 => self.bps[n as usize].map_or(0, |b| b.addr),
            6 => self.virtual_dr6 ^ DR6_RESERVED,
            7 => self.dr7,
            _ => 0,
        }
    }

    /// `ptrace_set_debugreg(n, v)`: `EIO` for `DR4` and `DR5`, `EINVAL` for
    /// a breakpoint the breakpoint layer refuses (nothing changes then).
    pub(super) fn set(&mut self, n: u64, v: u64) -> Result<(), i64> {
        match n {
            0..=3 => {
                let i = n as usize;
                // A first write makes a disabled 1-byte write breakpoint
                // (`ptrace_set_breakpoint_addr`); after that only the
                // address changes, checked against what it is.
                let bp = self.bps[i].unwrap_or(Bp {
                    addr: 0,
                    len: LEN_1,
                    ty: WRITE,
                    disabled: true,
                });
                parse(v, bp.len, bp.ty)?;
                self.bps[i] = Some(Bp { addr: v, ..bp });
                self.dirty = true;
                Ok(())
            }
            6 => {
                self.virtual_dr6 = v ^ DR6_RESERVED;
                Ok(())
            }
            7 => {
                self.write_dr7(v)?;
                self.dr7 = v;
                Ok(())
            }
            _ => Err(err(Errno::EIO)),
        }
    }

    /// `ptrace_get_dr7`: the enabled breakpoints, as `encode_dr7` puts them.
    fn encoded_dr7(&self) -> u64 {
        let mut dr7 = 0;
        for (i, b) in self.bps.iter().enumerate() {
            if let Some(b) = b.filter(|b| !b.disabled) {
                dr7 |= u64::from((b.len | b.ty) & 0xf) << (16 + 4 * i);
                dr7 |= 2 << (2 * i); // DR_GLOBAL_ENABLE
                dr7 |= 0x200; // DR_GLOBAL_SLOWDOWN
            }
        }
        dr7
    }

    /// `ptrace_write_dr7`: enable, disable and retype the breakpoints as
    /// `data` says, slot by slot; at the first that fails, put them back as
    /// they were and report it.
    fn write_dr7(&mut self, data: u64) -> Result<(), i64> {
        let old = self.encoded_dr7();
        let r = self.apply_dr7(data & !DR_CONTROL_RESERVED);
        if r.is_err() {
            let _ = self.apply_dr7(old);
        }
        self.dirty = true;
        r
    }

    fn apply_dr7(&mut self, data: u64) -> Result<(), i64> {
        for i in 0..4 {
            let (len, ty, enabled) = decode_dr7(data, i);
            let addr = match self.bps[i] {
                None if !enabled => continue,
                None => 0,
                Some(b) => b.addr,
            };
            parse(addr, len, ty)?;
            self.bps[i] = Some(Bp {
                addr,
                len,
                ty,
                disabled: !enabled,
            });
        }
        Ok(())
    }

    /// What the vcpu gets ([`crate::vcpu::Vcpu::set_debug_regs`]): the
    /// enabled breakpoints' addresses and a `DR7` with their local enables
    /// and fields.
    pub(super) fn hw(&self) -> [u64; 8] {
        let mut dr = [0u64; 8];
        for (i, b) in self.bps.iter().enumerate() {
            if let Some(b) = b.filter(|b| !b.disabled) {
                dr[i] = b.addr;
                dr[7] |= (1 << (2 * i)) | u64::from((b.len | b.ty) & 0xf) << (16 + 4 * i);
            }
        }
        dr
    }

    /// Whether any breakpoint is enabled.
    pub(super) fn armed(&self) -> bool {
        self.bps.iter().flatten().any(|b| !b.disabled)
    }

    /// A debug trap with hardware `dr6` (`exc_debug_user`): the virtual
    /// `DR6` starts over with its `DR_STEP`, and each enabled breakpoint
    /// that hit adds its bit (`ptrace_triggered`). The `DR6` the signal's
    /// `si_code` is chosen by (`get_si_code`).
    pub(super) fn trap(&mut self, dr6: u64) -> u64 {
        self.virtual_dr6 = dr6 & DR_STEP;
        for (i, b) in self.bps.iter().enumerate() {
            if dr6 & (1 << i) != 0 && b.is_some_and(|b| !b.disabled) {
                self.virtual_dr6 |= 1 << i;
            }
        }
        (dr6 & !DR_TRAP_BITS) | self.virtual_dr6
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EINVAL: i64 = -22;
    const EIO: i64 = -5;

    #[test]
    fn nothing_set_reads_as_linux_does() {
        let d = DebugRegs::default();
        assert_eq!(d.get(0), 0);
        assert_eq!(d.get(6), 0xffff_0ff0);
        assert_eq!(d.get(7), 0);
        assert_eq!(d.get(4), 0);
    }

    #[test]
    fn dr4_and_dr5_cannot_be_written() {
        let mut d = DebugRegs::default();
        assert_eq!(d.set(4, 1), Err(EIO));
        assert_eq!(d.set(5, 1), Err(EIO));
    }

    #[test]
    fn an_address_is_checked_against_the_breakpoint_it_has() {
        let mut d = DebugRegs::default();
        // A first write is a 1-byte breakpoint: any user address.
        assert_eq!(d.set(0, 0x1001), Ok(()));
        assert_eq!(d.get(0), 0x1001);
        assert_eq!(d.set(0, TASK_SIZE_MAX), Err(EINVAL));
        assert_eq!(d.set(0, u64::MAX), Err(EINVAL));
        assert_eq!(d.get(0), 0x1001);
        // An enabled 4-byte watchpoint at an unaligned address: refused,
        // and DR7 reads back what it was.
        assert_eq!(d.set(7, 0xd0001), Err(EINVAL));
        assert_eq!(d.get(7), 0);
        assert!(!d.armed());
        assert_eq!(d.set(0, 0x1000), Ok(()));
        assert_eq!(d.set(7, 0xd0001), Ok(()));
        assert_eq!(d.get(7), 0xd0001);
        assert!(d.armed());
        // Now 4 bytes long, so the address must stay 4-aligned.
        assert_eq!(d.set(0, 0x1002), Err(EINVAL));
        assert_eq!(d.set(0, 0x2000), Ok(()));
        assert_eq!(d.hw(), [0x2000, 0, 0, 0, 0, 0, 0, 0xd0001]);
    }

    #[test]
    fn dr7_types_and_lengths() {
        let mut d = DebugRegs::default();
        // An I/O breakpoint (R/W 10) is refused.
        assert_eq!(d.set(7, 0x2_0001), Err(EINVAL));
        // An execute breakpoint must be LEN 00.
        assert_eq!(d.set(7, 0x4_0001), Err(EINVAL));
        assert_eq!(d.set(7, 0x1), Ok(()));
        // 8-byte read/write.
        assert_eq!(d.set(1, 0x7fff_ffff_eff8), Ok(()));
        assert_eq!(d.set(7, 0xb0_0004), Ok(()));
        assert_eq!(d.hw()[1], 0x7fff_ffff_eff8);
        assert_eq!(d.hw()[7], 0xb0_0004);
        // The reserved bits are kept as written but not acted on.
        assert_eq!(d.set(7, 0xb0_0504), Ok(()));
        assert_eq!(d.get(7), 0xb0_0504);
    }

    #[test]
    fn a_failed_dr7_puts_back_what_was_there() {
        let mut d = DebugRegs::default();
        assert_eq!(d.set(1, 0x3001), Ok(()));
        assert_eq!(d.set(7, 0x1 | 0x4), Ok(()));
        // Slot 0 retyped fine, then slot 1 (1-byte aligned at 0x3001) as 2
        // bytes fails: slot 0 goes back to the execute breakpoint.
        assert_eq!(d.set(7, 0x51_0005), Err(EINVAL));
        assert_eq!(d.get(7), 0x5);
        assert_eq!(d.hw()[7], 0x5);
    }

    #[test]
    fn traps_set_the_virtual_dr6() {
        let mut d = DebugRegs::default();
        assert_eq!(d.set(2, 0x4000), Ok(()));
        assert_eq!(d.set(7, 0x10_0010), Ok(()));
        assert_eq!(d.trap(0xffff_0ff4), (0xffff_0ff4 & !0xf) | 4);
        assert_eq!(d.get(6), 0xffff_0ff4);
        // A single step: just DR_STEP.
        assert_eq!(d.trap(DR_STEP) & 0xf, 0);
        assert_eq!(d.get(6), 0xffff_4ff0);
        // A debugger clears it.
        assert_eq!(d.set(6, 0), Ok(()));
        assert_eq!(d.get(6), 0);
    }
}
