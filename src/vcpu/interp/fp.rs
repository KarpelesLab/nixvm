//! A64 scalar floating-point instructions (the non-SIMD half of the "Data
//! Processing -- Scalar Floating-Point and Advanced SIMD" group, bit 30 = 0,
//! bits 28:24 = `1111x`): data-processing with 1/2/3 sources, `FCMP`/
//! `FCMPE`/`FCCMP`/`FCCMPE`, `FCSEL`, `FMOV` (register, immediate and
//! general-register forms), and the integer and fixed-point conversions.
//!
//! Single and double precision are supported throughout; half precision only
//! where Armv8.0 has it (`FCVT` to and from half). Half-precision arithmetic
//! (`FEAT_FP16`) and `FJCVTZS` (`FEAT_JSCVT`), `FRINT32*`/`FRINT64*`
//! (`FEAT_FRINTTS`) and `BFCVT` (`FEAT_BF16`) are not advertised, so their
//! encodings are UNDEFINED.

use super::fpu::{self, D, Fmt, H, Rounding, S};
use super::{Aarch64Interp, Flags, Step, reg_field};

/// The scalar format selected by an FP `ftype` field, if Armv8.0 arithmetic
/// supports it (single or double).
pub(super) fn arith_fmt(ftype: u32) -> Option<Fmt> {
    match ftype {
        0b00 => Some(S),
        0b01 => Some(D),
        _ => None,
    }
}

/// `VFPExpandImm`: the 8-bit `FMOV` immediate expanded to `fmt`.
pub(super) fn vfp_expand_imm(imm8: u32, fmt: Fmt) -> u64 {
    let sign = u64::from(imm8 >> 7);
    let b6 = u64::from((imm8 >> 6) & 1);
    let e = fmt.e;
    // exp = NOT(b6) : Replicate(b6, E-3) : imm8<5:4>
    let rep = if b6 == 1 { (1 << (e - 3)) - 1 } else { 0 };
    let exp = ((b6 ^ 1) << (e - 1)) | (rep << 2) | u64::from((imm8 >> 4) & 3);
    let frac = u64::from(imm8 & 0xf) << (fmt.f - 4);
    (sign << (fmt.n() - 1)) | (exp << fmt.f) | frac
}

impl Aarch64Interp {
    /// The floating-point environment for one instruction.
    #[inline]
    pub(super) fn fpenv(&self) -> fpu::Env {
        fpu::Env {
            fpcr: self.fpcr as u32,
            fpsr: self.fpsr as u32,
        }
    }

    /// Commit an instruction's accumulated `FPSR` flags.
    #[inline]
    pub(super) fn set_fpenv(&mut self, env: fpu::Env) {
        self.fpsr = u64::from(env.fpsr);
    }

    /// Read the low `fmt` bits of `V[n]`.
    #[inline]
    pub(super) fn vreg(&self, n: usize, fmt: Fmt) -> u64 {
        (self.v[n] as u64) & fmt.mask()
    }

    /// Write a scalar result (`fmt`-wide), zeroing the rest of `V[d]`.
    #[inline]
    pub(super) fn set_vreg(&mut self, d: usize, bits: u64, fmt: Fmt) {
        self.v[d] = u128::from(bits & fmt.mask());
    }

    pub(super) fn exec_fp(&mut self, instr: u32) -> Step {
        let rd = reg_field(instr, 0);
        let rn = reg_field(instr, 5);
        let rm = reg_field(instr, 16);
        let ftype = (instr >> 22) & 3;
        if (instr >> 24) & 1 == 1 {
            // FMADD/FMSUB/FNMADD/FNMSUB.
            if instr >> 29 != 0 {
                return Step::Illegal;
            }
            let Some(fmt) = arith_fmt(ftype) else {
                return Step::Illegal;
            };
            let ra = reg_field(instr, 10);
            let o1 = (instr >> 21) & 1 == 1;
            let o0 = (instr >> 15) & 1 == 1;
            let sign = 1u64 << (fmt.n() - 1);
            let mut a = self.vreg(ra, fmt);
            let mut n = self.vreg(rn, fmt);
            let m = self.vreg(rm, fmt);
            if o1 {
                a ^= sign;
            }
            if o1 != o0 {
                n ^= sign;
            }
            let mut env = self.fpenv();
            let r = fpu::mul_add(a, n, m, fmt, &mut env);
            self.set_fpenv(env);
            self.set_vreg(rd, r, fmt);
            return Step::Next;
        }
        if (instr >> 21) & 1 == 0 {
            return self.exec_fp_fixed(instr);
        }
        if (instr >> 10) & 0x3f == 0 {
            return self.exec_fp_int(instr);
        }
        if instr >> 29 != 0 {
            return Step::Illegal; // M and S must be zero
        }
        match (instr >> 10) & 3 {
            0b01 => {
                // FCCMP/FCCMPE
                let Some(fmt) = arith_fmt(ftype) else {
                    return Step::Illegal;
                };
                if self.cond_holds((instr >> 12) & 0xf) {
                    let mut env = self.fpenv();
                    let signal = (instr >> 4) & 1 == 1;
                    let nzcv = fpu::compare(
                        self.vreg(rn, fmt),
                        self.vreg(rm, fmt),
                        signal,
                        fmt,
                        &mut env,
                    );
                    self.set_fpenv(env);
                    self.flags = Flags::from_nzcv(nzcv);
                } else {
                    self.flags = Flags::from_nzcv(instr & 0xf);
                }
                Step::Next
            }
            0b10 => {
                // 2-source arithmetic.
                let Some(fmt) = arith_fmt(ftype) else {
                    return Step::Illegal;
                };
                let (a, b) = (self.vreg(rn, fmt), self.vreg(rm, fmt));
                let mut env = self.fpenv();
                let e = &mut env;
                let r = match (instr >> 12) & 0xf {
                    0b0000 => fpu::mul(a, b, false, fmt, e),
                    0b0001 => fpu::div(a, b, fmt, e),
                    0b0010 => fpu::add(a, b, false, fmt, e),
                    0b0011 => fpu::add(a, b, true, fmt, e),
                    0b0100 => fpu::max_min(a, b, true, false, fmt, e),
                    0b0101 => fpu::max_min(a, b, false, false, fmt, e),
                    0b0110 => fpu::max_min(a, b, true, true, fmt, e),
                    0b0111 => fpu::max_min(a, b, false, true, fmt, e),
                    0b1000 => fpu::mul(a, b, false, fmt, e) ^ (1 << (fmt.n() - 1)), // FNMUL
                    _ => return Step::Illegal,
                };
                self.set_fpenv(env);
                self.set_vreg(rd, r, fmt);
                Step::Next
            }
            0b11 => {
                // FCSEL
                let Some(fmt) = arith_fmt(ftype) else {
                    return Step::Illegal;
                };
                let src = if self.cond_holds((instr >> 12) & 0xf) {
                    rn
                } else {
                    rm
                };
                let r = self.vreg(src, fmt);
                self.set_vreg(rd, r, fmt);
                Step::Next
            }
            _ => {
                if (instr >> 12) & 1 == 1 {
                    // FMOV (scalar, immediate)
                    let Some(fmt) = arith_fmt(ftype) else {
                        return Step::Illegal;
                    };
                    if rn != 0 {
                        return Step::Illegal;
                    }
                    let r = vfp_expand_imm((instr >> 13) & 0xff, fmt);
                    self.set_vreg(rd, r, fmt);
                    Step::Next
                } else if (instr >> 10) & 0xf == 0b1000 {
                    // FCMP/FCMPE (register or #0.0)
                    let Some(fmt) = arith_fmt(ftype) else {
                        return Step::Illegal;
                    };
                    if (instr >> 14) & 3 != 0 || instr & 7 != 0 {
                        return Step::Illegal;
                    }
                    let with_zero = (instr >> 3) & 1 == 1;
                    let signal = (instr >> 4) & 1 == 1;
                    let b = if with_zero { 0 } else { self.vreg(rm, fmt) };
                    let mut env = self.fpenv();
                    let nzcv = fpu::compare(self.vreg(rn, fmt), b, signal, fmt, &mut env);
                    self.set_fpenv(env);
                    self.flags = Flags::from_nzcv(nzcv);
                    Step::Next
                } else if (instr >> 10) & 0x1f == 0b10000 {
                    self.exec_fp_1src(instr)
                } else {
                    Step::Illegal
                }
            }
        }
    }

    /// FP data-processing (1 source).
    fn exec_fp_1src(&mut self, instr: u32) -> Step {
        let rd = reg_field(instr, 0);
        let rn = reg_field(instr, 5);
        let ftype = (instr >> 22) & 3;
        let opcode = (instr >> 15) & 0x3f;
        if opcode & 0b111100 == 0b000100 {
            // FCVT between half, single and double.
            let from = match ftype {
                0b00 => S,
                0b01 => D,
                0b11 => H,
                _ => return Step::Illegal,
            };
            let to = match opcode & 3 {
                0b00 => S,
                0b01 => D,
                0b11 => H,
                _ => return Step::Illegal,
            };
            if from == to {
                return Step::Illegal;
            }
            let mut env = self.fpenv();
            let mode = env.rounding();
            let r = fpu::convert(self.vreg(rn, from), from, to, mode, &mut env);
            self.set_fpenv(env);
            self.set_vreg(rd, r, to);
            return Step::Next;
        }
        let Some(fmt) = arith_fmt(ftype) else {
            return Step::Illegal;
        };
        let a = self.vreg(rn, fmt);
        let sign = 1u64 << (fmt.n() - 1);
        let mut env = self.fpenv();
        let r = match opcode {
            0b000000 => a,
            0b000001 => a & !sign,
            0b000010 => a ^ sign,
            0b000011 => fpu::sqrt(a, fmt, &mut env),
            0b001000..=0b001111 => {
                let (mode, exact) = match opcode & 7 {
                    0b000 => (Rounding::TieEven, false),
                    0b001 => (Rounding::PosInf, false),
                    0b010 => (Rounding::NegInf, false),
                    0b011 => (Rounding::Zero, false),
                    0b100 => (Rounding::TieAway, false),
                    0b110 => (env.rounding(), true),
                    0b111 => (env.rounding(), false),
                    _ => return Step::Illegal,
                };
                fpu::round_int(a, mode, exact, fmt, &mut env)
            }
            _ => return Step::Illegal,
        };
        self.set_fpenv(env);
        self.set_vreg(rd, r, fmt);
        Step::Next
    }

    /// Conversion between floating-point and integer, and `FMOV` to/from a
    /// general-purpose register (`sf 0 S 11110 ftype 1 rmode opcode 000000`).
    fn exec_fp_int(&mut self, instr: u32) -> Step {
        let sf = instr >> 31 == 1;
        let rd = reg_field(instr, 0);
        let rn = reg_field(instr, 5);
        let ftype = (instr >> 22) & 3;
        let rmode = (instr >> 19) & 3;
        let opcode = (instr >> 16) & 7;
        if (instr >> 29) & 1 != 0 {
            return Step::Illegal;
        }
        let ibits = if sf { 64 } else { 32 };
        match (rmode, opcode) {
            (_, 0b000 | 0b001) | (0b00, 0b100 | 0b101) => {
                let Some(fmt) = arith_fmt(ftype) else {
                    return Step::Illegal;
                };
                let mode = if opcode & 4 != 0 {
                    Rounding::TieAway
                } else {
                    Rounding::from_rmode(rmode)
                };
                let mut env = self.fpenv();
                let r = fpu::to_fixed(
                    self.vreg(rn, fmt),
                    fmt,
                    0,
                    opcode & 1 == 1,
                    ibits,
                    mode,
                    &mut env,
                );
                self.set_fpenv(env);
                self.write_x(rd, r);
            }
            (0b00, 0b010 | 0b011) => {
                let Some(fmt) = arith_fmt(ftype) else {
                    return Step::Illegal;
                };
                let mut env = self.fpenv();
                let mode = env.rounding();
                let r = fpu::from_fixed(
                    self.read_x(rn),
                    ibits,
                    opcode & 1 == 0,
                    0,
                    fmt,
                    mode,
                    &mut env,
                );
                self.set_fpenv(env);
                self.set_vreg(rd, r, fmt);
            }
            (0b00, 0b110 | 0b111) => {
                // FMOV Wd<->Sn / Xd<->Dn (half precision needs FEAT_FP16).
                let fmt = match (sf, ftype) {
                    (false, 0b00) => S,
                    (true, 0b01) => D,
                    _ => return Step::Illegal,
                };
                if opcode == 0b110 {
                    let r = self.vreg(rn, fmt);
                    self.write_x(rd, r);
                } else {
                    let r = self.read_x(rn);
                    self.set_vreg(rd, r, fmt);
                }
            }
            (0b01, 0b110 | 0b111) => {
                // FMOV Xd, Vn.D[1] / FMOV Vd.D[1], Xn
                if !sf || ftype != 0b10 {
                    return Step::Illegal;
                }
                if opcode == 0b110 {
                    self.write_x(rd, (self.v[rn] >> 64) as u64);
                } else {
                    let x = self.read_x(rn);
                    self.v[rd] = (self.v[rd] & u128::from(u64::MAX)) | (u128::from(x) << 64);
                }
            }
            _ => return Step::Illegal,
        }
        Step::Next
    }

    /// Conversion between floating-point and fixed-point
    /// (`sf 0 S 11110 ftype 0 rmode opcode scale Rn Rd`).
    fn exec_fp_fixed(&mut self, instr: u32) -> Step {
        let sf = instr >> 31 == 1;
        let rd = reg_field(instr, 0);
        let rn = reg_field(instr, 5);
        let scale = (instr >> 10) & 0x3f;
        if (instr >> 29) & 1 != 0 || (!sf && scale < 32) {
            return Step::Illegal;
        }
        let Some(fmt) = arith_fmt((instr >> 22) & 3) else {
            return Step::Illegal;
        };
        let fbits = 64 - scale;
        let ibits = if sf { 64 } else { 32 };
        let mut env = self.fpenv();
        match (instr >> 16) & 0x1f {
            0b00010 | 0b00011 => {
                let signed = (instr >> 16) & 1 == 0;
                let mode = env.rounding();
                let r = fpu::from_fixed(self.read_x(rn), ibits, signed, fbits, fmt, mode, &mut env);
                self.set_vreg(rd, r, fmt);
            }
            0b11000 | 0b11001 => {
                let unsigned = (instr >> 16) & 1 == 1;
                let r = fpu::to_fixed(
                    self.vreg(rn, fmt),
                    fmt,
                    fbits,
                    unsigned,
                    ibits,
                    Rounding::Zero,
                    &mut env,
                );
                self.write_x(rd, r);
            }
            _ => return Step::Illegal,
        }
        self.set_fpenv(env);
        Step::Next
    }
}
