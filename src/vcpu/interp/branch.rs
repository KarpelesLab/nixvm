//! A64 "Branches, Exception Generating and System instructions" (`op0 =
//! 101x`): immediate/conditional/compare/test branches, branch-to-register,
//! `SVC`, hints, barriers, `MRS`/`MSR`, and the cache-maintenance `SYS` ops.
//!
//! The system-register surface is what Linux exposes to EL0 on a v8.0 CPU
//! with the features this interpreter advertises: direct access to `NZCV`,
//! `FPCR`/`FPSR`, `TPIDR_EL0`/`TPIDRRO_EL0`, `CTR_EL0`, `DCZID_EL0`,
//! `CNTVCT_EL0`/`CNTFRQ_EL0`; the kernel's trap-and-emulate view of `MIDR_EL1`/
//! `MPIDR_EL1`/`REVIDR_EL1` and the `ID_*` feature-register space
//! (`op0=3, op1=0, CRn=0, CRm=1..7`); and `DC ZVA`/`DC CVAC`/`DC CVAU`/
//! `DC CIVAC`/`IC IVAU` (`SCTLR_EL1.DZE`/`UCI` set). Everything else — `DAIF`
//! (`UMA` clear), the physical timer and counter, PMU and debug registers,
//! `MSR` (immediate), `TLBI`/`AT`/`DC IVAC`/`IC IALLU`, unimplemented-feature
//! registers like `RNDR`/`DIT`/`SSBS` — is UNDEFINED at EL0 and so returns
//! [`Step::Illegal`] (the guest gets `SIGILL`).

use super::alu::sign_extend;
use super::{Aarch64Interp, Flags, Step, reg_field};
use crate::vcpu::GuestMemory;

impl Aarch64Interp {
    pub(super) fn exec_branch_sys(&mut self, instr: u32, mem: &mut GuestMemory) -> Step {
        match instr >> 29 {
            // B / BL
            0b000 | 0b100 => {
                if instr >> 31 == 1 {
                    self.x[30] = self.pc.wrapping_add(4);
                }
                let off = sign_extend(u64::from(instr & 0x03ff_ffff), 26) << 2;
                self.branch(off)
            }
            // CBZ/CBNZ (bit 25 = 0), TBZ/TBNZ (bit 25 = 1)
            0b001 | 0b101 => {
                let rt = reg_field(instr, 0);
                let nonzero = (instr >> 24) & 1 == 1;
                let taken = if (instr >> 25) & 1 == 0 {
                    let v = self.read_x(rt);
                    let v = if instr >> 31 == 1 { v } else { v & 0xffff_ffff };
                    (v != 0) == nonzero
                } else {
                    let bit = ((instr >> 31) << 5) | ((instr >> 19) & 0x1f);
                    let set = (self.read_x(rt) >> bit) & 1 == 1;
                    set == nonzero
                };
                if taken {
                    let off = if (instr >> 25) & 1 == 0 {
                        sign_extend(u64::from((instr >> 5) & 0x7ffff), 19) << 2
                    } else {
                        sign_extend(u64::from((instr >> 5) & 0x3fff), 14) << 2
                    };
                    self.branch(off)
                } else {
                    Step::Next
                }
            }
            // B.cond (bit 4 = 1 would be FEAT_HBC's BC.cond).
            0b010 => {
                if (instr >> 24) & 3 != 0 || (instr >> 4) & 1 != 0 {
                    return Step::Illegal;
                }
                if self.cond_holds(instr & 0xf) {
                    self.branch(sign_extend(u64::from((instr >> 5) & 0x7ffff), 19) << 2)
                } else {
                    Step::Next
                }
            }
            0b110 => match (instr >> 24) & 3 {
                0b00 => {
                    // Exception generation: only SVC reaches the kernel. HVC/
                    // SMC/HLT/DCPSn are UNDEFINED at EL0; BRK is a debug
                    // exception (SIGTRAP/TRAP_BRKPT on Linux).
                    if instr & 0xFFE0_001F == 0xD400_0001 {
                        Step::Syscall
                    } else if instr & 0xFFE0_001F == 0xD420_0000 {
                        Step::Breakpoint {
                            imm: ((instr >> 5) & 0xffff) as u16,
                        }
                    } else {
                        Step::Illegal
                    }
                }
                0b01 => self.exec_system(instr, mem),
                _ => self.exec_branch_reg(instr),
            },
            _ => Step::Illegal,
        }
    }

    /// Unconditional branch (register): `BR`/`BLR`/`RET`. The PAuth forms
    /// (`BRAA`, `RETAA`, …), `ERET` and `DRPS` are UNDEFINED here.
    fn exec_branch_reg(&mut self, instr: u32) -> Step {
        if instr & 0xFE1F_FC1F != 0xD61F_0000 {
            return Step::Illegal; // op2 != 11111, op3 != 0, or op4 != 0
        }
        let target = self.read_x(reg_field(instr, 5));
        match (instr >> 21) & 0xf {
            0b0000 | 0b0010 => {}
            0b0001 => self.x[30] = self.pc.wrapping_add(4),
            _ => return Step::Illegal,
        }
        self.pc = target;
        Step::Branched
    }

    /// The System instruction class (bits 31:22 = `1101010100`).
    fn exec_system(&mut self, instr: u32, mem: &mut GuestMemory) -> Step {
        if (instr >> 22) & 3 != 0 {
            return Step::Illegal; // MRRS/MSRR/SYSP (FEAT_SYSREG128)
        }
        let l = (instr >> 21) & 1;
        let op0 = (instr >> 19) & 3;
        let rt = reg_field(instr, 0);
        match op0 {
            0b00 => {
                let op1 = (instr >> 16) & 7;
                let crn = (instr >> 12) & 0xf;
                let crm = (instr >> 8) & 0xf;
                let op2 = (instr >> 5) & 7;
                if l != 0 || rt != 31 {
                    return Step::Illegal; // WFET/WFIT, …
                }
                if crn == 0b0100 {
                    return self.exec_msr_imm(op1, crm, op2);
                }
                if op1 != 0b011 {
                    return Step::Illegal;
                }
                match crn {
                    // HINT space: NOP, YIELD, WFE, WFI, SEV, SEVL, and every
                    // hint a newer extension allocated (PACIASP/AUTIASP/BTI/
                    // XPACLRI/…) — architecturally NOPs on a CPU without the
                    // feature, which is exactly this one.
                    0b0010 => Step::Next,
                    // Barriers.
                    0b0011 => match op2 {
                        0b010 => {
                            self.excl_monitor = false; // CLREX
                            Step::Next
                        }
                        0b100 | 0b101 => Step::Next, // DSB / DMB
                        0b110 => {
                            self.invalidate_fetch(); // ISB
                            Step::Next
                        }
                        0b111 if crm == 0 => Step::Next, // SB (FEAT_SB)
                        _ => Step::Illegal,              // DSB nXS, TCOMMIT
                    },
                    _ => Step::Illegal,
                }
            }
            0b01 => {
                if l != 0 {
                    return Step::Illegal; // SYSL
                }
                self.exec_sys(instr, mem)
            }
            _ => {
                let key = (instr >> 5) & 0xffff; // op0:op1:CRn:CRm:op2
                if l == 1 {
                    let Some(v) = self.read_sysreg(key) else {
                        return Step::Illegal;
                    };
                    self.write_x(rt, v);
                } else if !self.write_sysreg(key, self.read_x(rt)) {
                    return Step::Illegal;
                }
                Step::Next
            }
        }
    }

    /// `MSR` (immediate) / the PSTATE-flag instructions (`CRn = 0100`): only
    /// `CFINV` (FEAT_FlagM), `XAFLAG`/`AXFLAG` (FEAT_FlagM2) and `MSR DIT`
    /// are permitted at EL0 on this CPU (`DAIFSet`/`DAIFClr` trap: UMA is
    /// clear; the rest are EL1-only or unimplemented).
    fn exec_msr_imm(&mut self, op1: u32, crm: u32, op2: u32) -> Step {
        let f = self.flags;
        match (op1, op2) {
            (0b000, 0b000) if crm == 0 => self.flags.c = !f.c, // CFINV
            (0b000, 0b001) if crm == 0 => {
                // XAFLAG: external (IEEE-style) to Arm flag format.
                self.flags = Flags {
                    n: !f.c && !f.z,
                    z: f.z && f.c,
                    c: f.c || f.z,
                    v: !f.c && f.z,
                };
            }
            (0b000, 0b010) if crm == 0 => {
                // AXFLAG
                self.flags = Flags {
                    n: false,
                    z: f.z || f.v,
                    c: f.c && !f.v,
                    v: false,
                };
            }
            (0b011, 0b010) => self.dit = crm & 1 == 1, // MSR DIT, #imm
            _ => return Step::Illegal,
        }
        Step::Next
    }

    /// `SYS` (op0 = 1): the EL0-permitted cache maintenance operations.
    fn exec_sys(&mut self, instr: u32, mem: &mut GuestMemory) -> Step {
        let rt = reg_field(instr, 0);
        match (instr >> 5) & 0x3fff {
            SYS_DC_ZVA => {
                // DC ZVA really zeroes guest memory (memset fast paths rely on
                // it). The block is DC_ZVA_BLOCK_BYTES, matching DCZID_EL0.
                let addr = self.read_x(rt) & !(DC_ZVA_BLOCK_BYTES - 1);
                self.note_store(addr, 0, DC_ZVA_BLOCK_BYTES as usize);
                if mem
                    .write_trap(addr, &[0u8; DC_ZVA_BLOCK_BYTES as usize])
                    .is_err()
                {
                    return Step::Fault { addr, write: true };
                }
                Step::Next
            }
            SYS_IC_IVAU => {
                self.invalidate_fetch();
                Step::Next
            }
            // DC CVAP / DC CVADP: FEAT_DPB / FEAT_DPB2.
            SYS_DC_CVAC | SYS_DC_CVAU | SYS_DC_CIVAC | SYS_DC_CVAP | SYS_DC_CVADP => Step::Next,
            _ => Step::Illegal,
        }
    }

    /// `MRS`: the value of the system register `key` (`op0:op1:CRn:CRm:op2`)
    /// as EL0 code under Linux sees it, or `None` if the access is UNDEFINED.
    #[allow(clippy::match_same_arms)] // distinct registers that read as zero
    pub(super) fn read_sysreg(&mut self, key: u32) -> Option<u64> {
        Some(match key {
            NZCV => u64::from(self.flags.nzcv()) << 28,
            DIT => u64::from(self.dit) << 24,
            FPCR => self.fpcr,
            FPSR => self.fpsr,
            TPIDR_EL0 => self.tpidr,
            TPIDRRO_EL0 => 0,
            CTR_EL0 => CTR_EL0_VAL,
            DCZID_EL0 => DCZID_EL0_VAL,
            CNTFRQ_EL0 => CNTFRQ_EL0_VAL,
            CNTVCT_EL0 => {
                // Advance on every read so `while (cntvct == cntvct) {}`-style
                // spins (and real "did time pass" checks) always terminate.
                self.cntvct = self.cntvct.wrapping_add(1);
                self.cntvct
            }
            // Linux emulates EL0 reads of the CPU identification space.
            MIDR_EL1 => MIDR_EL1_VAL,
            MPIDR_EL1 => MPIDR_EL1_VAL,
            REVIDR_EL1 => 0,
            ID_AA64PFR0_EL1 => ID_AA64PFR0_EL1_VAL,
            ID_AA64ISAR0_EL1 => ID_AA64ISAR0_EL1_VAL,
            ID_AA64DFR0_EL1 => ID_AA64DFR0_EL1_VAL,
            ID_AA64MMFR0_EL1 => ID_AA64MMFR0_EL1_VAL,
            ID_AA64ISAR1_EL1 => ID_AA64ISAR1_EL1_VAL,
            ID_AA64MMFR2_EL1 => ID_AA64MMFR2_EL1_VAL,
            // The rest of the ID space (op0=3, op1=0, CRn=0, CRm=1..7) reads
            // as zero: AArch32 ID registers (no AArch32 at EL0), and the
            // AArch64 ones whose every user-visible field is "not
            // implemented" for this CPU (ISAR1/ISAR2, PFR1, MMFR*, ZFR0, …).
            k if k & 0xff80 == 0xc000 && (1..=7).contains(&((k >> 3) & 0xf)) => 0,
            _ => return None,
        })
    }

    /// `MSR` (register): whether the write is permitted at EL0.
    pub(super) fn write_sysreg(&mut self, key: u32, value: u64) -> bool {
        match key {
            NZCV => self.flags = Flags::from_nzcv((value >> 28) as u32),
            DIT => self.dit = (value >> 24) & 1 == 1,
            // FPCR: only the fields this CPU implements are writable (AHP, DN,
            // FZ, RMode, FZ16; the trap enables are RES0 without trapping
            // support).
            FPCR => self.fpcr = value & FPCR_WRITABLE,
            FPSR => self.fpsr = value & FPSR_WRITABLE,
            TPIDR_EL0 => self.tpidr = value,
            _ => return false,
        }
        true
    }
}

// System-register keys: `op0:op1:CRn:CRm:op2` (instruction bits 20:5).
const fn sysreg(op0: u32, op1: u32, crn: u32, crm: u32, op2: u32) -> u32 {
    (op0 << 14) | (op1 << 11) | (crn << 7) | (crm << 3) | op2
}
const NZCV: u32 = sysreg(3, 3, 4, 2, 0);
const DIT: u32 = sysreg(3, 3, 4, 2, 5);
const FPCR: u32 = sysreg(3, 3, 4, 4, 0);
const FPSR: u32 = sysreg(3, 3, 4, 4, 1);
const TPIDR_EL0: u32 = sysreg(3, 3, 13, 0, 2);
const TPIDRRO_EL0: u32 = sysreg(3, 3, 13, 0, 3);
const CTR_EL0: u32 = sysreg(3, 3, 0, 0, 1);
const DCZID_EL0: u32 = sysreg(3, 3, 0, 0, 7);
const CNTFRQ_EL0: u32 = sysreg(3, 3, 14, 0, 0);
const CNTVCT_EL0: u32 = sysreg(3, 3, 14, 0, 2);
const MIDR_EL1: u32 = sysreg(3, 0, 0, 0, 0);
const MPIDR_EL1: u32 = sysreg(3, 0, 0, 0, 5);
const REVIDR_EL1: u32 = sysreg(3, 0, 0, 0, 6);
const ID_AA64PFR0_EL1: u32 = sysreg(3, 0, 0, 4, 0);
const ID_AA64DFR0_EL1: u32 = sysreg(3, 0, 0, 5, 0);
const ID_AA64ISAR0_EL1: u32 = sysreg(3, 0, 0, 6, 0);
const ID_AA64MMFR0_EL1: u32 = sysreg(3, 0, 0, 7, 0);
const ID_AA64ISAR1_EL1: u32 = sysreg(3, 0, 0, 6, 1);
const ID_AA64MMFR2_EL1: u32 = sysreg(3, 0, 0, 7, 2);

/// `FPCR` bits with an effect on this CPU: `AHP`(26) `DN`(25) `FZ`(24)
/// `RMode`(23:22) `FZ16`(19).
const FPCR_WRITABLE: u64 = 0x07C8_0000;
/// `FPSR`: `QC`(27), `IDC`(7) and the cumulative `IXC`/`UFC`/`OFC`/`DZC`/
/// `IOC` (4:0); everything else is RES0 in AArch64.
const FPSR_WRITABLE: u64 = 0x0800_009F;

/// `MIDR_EL1`: an Arm Neoverse-N1 r0p0 (implementer 0x41, part 0xD0C) — a
/// plain v8.2 server core; nothing in userland keys behaviour off it beyond
/// errata/tuning tables, and those only on exact matches.
const MIDR_EL1_VAL: u64 = 0x410F_D0C0;
/// `MPIDR_EL1` as Linux's EL0 emulation reports it (`SYS_MPIDR_SAFE_VAL`:
/// only the RES1 bit 31).
const MPIDR_EL1_VAL: u64 = 1 << 31;
/// `CTR_EL0`: bit 31 RES1, `CWG`=`ERG`=4 (64-byte writeback/reservation
/// granules), `DminLine`=`IminLine`=4 (64-byte lines), `L1Ip`=PIPT; `IDC`/
/// `DIC` clear, so code must (and does) issue `DC CVAU`/`IC IVAU`.
const CTR_EL0_VAL: u64 = 0x8444_C004;
/// `DCZID_EL0`: `BS`=4 → `DC ZVA` zeroes `4 << 4` = 64 bytes; `DZP`=0.
const DCZID_EL0_VAL: u64 = 0x4;
/// `DC ZVA`'s block size; must stay in sync with [`DCZID_EL0_VAL`].
const DC_ZVA_BLOCK_BYTES: u64 = 64;
/// 1 GHz: `CNTVCT_EL0` ticks once per read, so the rate is nominal.
const CNTFRQ_EL0_VAL: u64 = 1_000_000_000;
/// `ID_AA64ISAR0_EL1` (Linux's sanitised EL0 view): `AES`=2 (AES + PMULL),
/// `SHA1`=1, `SHA2`=2 (SHA-256 + SHA-512), `CRC32`=1, `Atomic`=2 (LSE),
/// `RDM`=1, `SHA3`=1, `DP`=1, `FHM`=1, `TS`=2 (FlagM + FlagM2); SM3/SM4,
/// TME, TLB and RNDR absent. Must agree with `HWCAP_AARCH64`/`HWCAP2_AARCH64`
/// in the loader.
const ID_AA64ISAR0_EL1_VAL: u64 = 0x0021_1001_1021_2120;
/// `ID_AA64ISAR1_EL1`: `DPB`=2 (DC CVAP + CVADP), `JSCVT`=1, `FCMA`=1,
/// `LRCPC`=2 (LDAPR + LDAPUR/STLUR), `FRINTTS`=1, `SB`=1, `BF16`=1,
/// `I8MM`=1; no pointer authentication.
const ID_AA64ISAR1_EL1_VAL: u64 = 0x0010_1011_0021_1002;
/// `ID_AA64PFR0_EL1`: `EL0`=`EL1`=1 (AArch64 only), `FP`=`AdvSIMD`=1
/// (with half-precision arithmetic), `DIT`=1.
const ID_AA64PFR0_EL1_VAL: u64 = 0x0001_0000_0011_0011;
/// `ID_AA64MMFR2_EL1`: `AT`=1 — LSE2's unaligned single-copy atomicity
/// (`HWCAP_USCAT`).
const ID_AA64MMFR2_EL1_VAL: u64 = 1 << 32;
/// `ID_AA64DFR0_EL1`: Linux exposes only `DebugVer`, as its safe value 6
/// (Armv8 debug).
const ID_AA64DFR0_EL1_VAL: u64 = 0x6;
/// `ID_AA64MMFR0_EL1`: every field hidden from EL0, so Linux reports the safe
/// values — `TGran4`/`TGran64` = 0xf ("not implemented"), the rest 0.
const ID_AA64MMFR0_EL1_VAL: u64 = 0xFF00_0000;

/// `op1:CRn:CRm:op2` (instruction bits 18:5) of the EL0 cache-maintenance ops.
const fn sys_op(op1: u32, crn: u32, crm: u32, op2: u32) -> u32 {
    (op1 << 11) | (crn << 7) | (crm << 3) | op2
}
const SYS_DC_ZVA: u32 = sys_op(3, 7, 4, 1);
const SYS_DC_CVAC: u32 = sys_op(3, 7, 10, 1);
const SYS_DC_CVAU: u32 = sys_op(3, 7, 11, 1);
const SYS_DC_CIVAC: u32 = sys_op(3, 7, 14, 1);
const SYS_IC_IVAU: u32 = sys_op(3, 7, 5, 1);
const SYS_DC_CVAP: u32 = sys_op(3, 7, 12, 1);
const SYS_DC_CVADP: u32 = sys_op(3, 7, 13, 1);
