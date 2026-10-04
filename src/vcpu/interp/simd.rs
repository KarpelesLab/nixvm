//! Advanced SIMD (vector and scalar) and the Cryptographic Extension: the
//! AdvSIMD half of the "Data Processing -- Scalar Floating-Point and Advanced
//! SIMD" encoding group.
//!
//! Decoding follows the ARM ARM's class split — three same, three different,
//! two-register miscellaneous, across lanes, copy, modified immediate, shift
//! by immediate, by element, permute, extract, table lookup, scalar pairwise —
//! and within each class the `U`/`opcode`/`size` tables, with every reserved
//! size/`Q` combination UNDEFINED. Floating-point lanes go through [`fpu`], so
//! they honour `FPCR` (rounding, `FZ`, `DN`) and accumulate `FPSR` flags just
//! like the scalar instructions; saturating integer ops set `FPSR.QC`.
//!
//! Not implemented because not advertised (UNDEFINED, like a v8.0 core):
//! half-precision arithmetic (`FEAT_FP16`), `SDOT`/`UDOT` (DotProd),
//! `SQRDMLAH`/`SQRDMLSH` (RDM), `FMLAL` (FHM), `FCMLA`/`FCADD` (FCMA), `I8MM`,
//! BF16, SHA-512/SHA-3/SM3/SM4, `FRINT32*`/`FRINT64*`.

// Decode keys are written grouped by encoding field (`U`_`a`_`opcode`), which
// is how the ARM ARM tables read; and decode tables naturally have distinct
// encodings sharing an implementation.
#![allow(clippy::unusual_byte_groupings, clippy::match_same_arms)]

use super::alu::{ones, sign_extend};
use super::crypto;
use super::fp::vfp_expand_imm;
use super::fpu::{self, Cmp, D, Fmt, H, QC, Rounding, S};
use super::{Aarch64Interp, Step, reg_field};

/// Element `i` (of `esize` bits) of `v`.
#[inline]
fn elem(v: u128, i: u32, esize: u32) -> u64 {
    ((v >> (i * esize)) as u64) & ones(esize)
}

/// An element's integer value, signed or unsigned.
#[inline]
fn ival(x: u64, esize: u32, unsigned: bool) -> i128 {
    if unsigned {
        i128::from(x)
    } else {
        i128::from(sign_extend(x, esize))
    }
}

/// `SatQ`: saturate `v` to `esize` bits (signed or unsigned), returning the
/// bit pattern and whether it saturated.
#[inline]
fn sat(v: i128, esize: u32, unsigned: bool) -> (u64, bool) {
    let (lo, hi) = if unsigned {
        (0, (1i128 << esize) - 1)
    } else {
        (-(1i128 << (esize - 1)), (1i128 << (esize - 1)) - 1)
    };
    if v < lo {
        (lo as u64 & ones(esize), true)
    } else if v > hi {
        (hi as u64 & ones(esize), true)
    } else {
        (v as u64 & ones(esize), false)
    }
}

/// All-ones of `esize` bits if `c`.
#[inline]
fn mask_if(c: bool, esize: u32) -> u64 {
    if c { ones(esize) } else { 0 }
}

/// The register shift of `SSHL`/`USHL`/`SRSHL`/`SQSHL`/`SQRSHL`/… : shift
/// `a` left by the signed amount `sh` (right if negative), with optional
/// rounding and saturation. Returns `(bits, saturated)`.
fn shift_lane(
    a: u64,
    sh: i64,
    esize: u32,
    unsigned: bool,
    round: bool,
    saturate: bool,
) -> (u64, bool) {
    let v = ival(a, esize, unsigned);
    let r = if sh >= 0 {
        let sh = sh as u32;
        if sh >= esize {
            if !saturate || v == 0 {
                return (0, false);
            }
            if v > 0 { i128::MAX } else { i128::MIN }
        } else {
            v << sh
        }
    } else {
        let n = (-sh) as u32;
        if n >= 127 {
            if !round && v < 0 { -1 } else { 0 }
        } else if round {
            (v + (1i128 << (n - 1))) >> n
        } else {
            v >> n
        }
    };
    if saturate {
        sat(r, esize, unsigned)
    } else {
        (r as u64 & ones(esize), false)
    }
}

/// Carry-less multiply of two 8-bit polynomials (`PMUL`/`PMULL` 8-bit).
fn pmul8(a: u64, b: u64) -> u64 {
    (0..8)
        .filter(|i| (b >> i) & 1 == 1)
        .fold(0, |acc, i| acc ^ (a << i))
}

/// The vector FP format for a `sz` bit and `Q`: single (2 or 4 lanes) or
/// double (2 lanes, `Q` required).
fn fp_vec(q: bool, sz: u32) -> Option<(Fmt, u32, u32)> {
    match (sz, q) {
        (0, false) => Some((S, 32, 2)),
        (0, true) => Some((S, 32, 4)),
        (1, true) => Some((D, 64, 2)),
        _ => None,
    }
}

impl Aarch64Interp {
    /// Data Processing -- Scalar FP and Advanced SIMD (`op0 = x111`).
    pub(super) fn exec_simd_fp(&mut self, instr: u32) -> Step {
        let scalar = (instr >> 28) & 1 == 1;
        if scalar && (instr >> 30) & 1 == 0 {
            return self.exec_fp(instr);
        }
        if instr >> 31 != 0 {
            return Step::Illegal; // SHA-512/SHA-3/SM3/SM4, …
        }
        let bit10 = (instr >> 10) & 1 == 1;
        let bit21 = (instr >> 21) & 1 == 1;
        if (instr >> 24) & 1 == 1 {
            // x1111: modified immediate / shift by immediate / by element.
            if !bit10 {
                return self.simd_by_elem(instr, scalar);
            }
            if (instr >> 23) & 1 != 0 {
                return Step::Illegal;
            }
            if (instr >> 19) & 0xf == 0 {
                return if scalar {
                    Step::Illegal
                } else {
                    self.simd_mod_imm(instr)
                };
            }
            return self.simd_shift_imm(instr, scalar);
        }
        if bit21 {
            if bit10 {
                return self.simd_three_same(instr, scalar);
            }
            if (instr >> 11) & 1 == 0 {
                return self.simd_three_diff(instr, scalar);
            }
            return match (instr >> 17) & 0xf {
                0b0000 => self.simd_misc(instr, scalar),
                0b1000 => {
                    if scalar {
                        self.simd_scalar_pairwise(instr)
                    } else {
                        self.simd_across(instr)
                    }
                }
                0b0100 => self.crypto_two_reg(instr, scalar),
                _ => Step::Illegal,
            };
        }
        if bit10 {
            if (instr >> 15) & 1 == 0 && (instr >> 21) & 7 == 0 {
                return self.simd_copy(instr, scalar);
            }
            return Step::Illegal; // three-same extra / FP16
        }
        if (instr >> 15) & 1 != 0 {
            return Step::Illegal;
        }
        if scalar {
            if (instr >> 11) & 1 == 0 && (instr >> 29) & 1 == 0 && (instr >> 22) & 3 == 0 {
                return self.crypto_sha3(instr);
            }
            return Step::Illegal;
        }
        if (instr >> 29) & 1 == 1 {
            self.simd_ext(instr)
        } else if (instr >> 11) & 1 == 0 {
            self.simd_tbl(instr)
        } else {
            self.simd_permute(instr)
        }
    }

    /// Write a vector result, clearing the upper half for a 64-bit (`Q=0`)
    /// operation.
    #[inline]
    fn set_vec(&mut self, rd: usize, r: u128, q: bool) {
        self.v[rd] = if q { r } else { r & u128::from(u64::MAX) };
    }

    /// Record saturation in `FPSR.QC`.
    #[inline]
    fn set_qc(&mut self, saturated: bool) {
        if saturated {
            self.fpsr |= u64::from(QC);
        }
    }

    // ---- three same --------------------------------------------------------

    fn simd_three_same(&mut self, instr: u32, scalar: bool) -> Step {
        let opcode = (instr >> 11) & 0x1f;
        if opcode >= 0b11000 {
            return self.simd_fp_three_same(instr, scalar);
        }
        let q = (instr >> 30) & 1 == 1;
        let u = (instr >> 29) & 1;
        let size = (instr >> 22) & 3;
        let (rd, rn, rm) = (
            reg_field(instr, 0),
            reg_field(instr, 5),
            reg_field(instr, 16),
        );
        let (n, m, d) = (self.v[rn], self.v[rm], self.v[rd]);
        if opcode == 0b00011 {
            // Bitwise: AND/BIC/ORR/ORN (U=0), EOR/BSL/BIT/BIF (U=1).
            if scalar {
                return Step::Illegal;
            }
            let r = match (u, size) {
                (0, 0) => n & m,
                (0, 1) => n & !m,
                (0, 2) => n | m,
                (0, _) => n | !m,
                (_, 0) => n ^ m,
                (_, 1) => (d & n) | (!d & m),
                (_, 2) => (d & !m) | (n & m),
                _ => (d & m) | (n & !m),
            };
            self.set_vec(rd, r, q);
            return Step::Next;
        }
        let esize = 8u32 << size;
        if scalar {
            let ok = match opcode {
                0b00001 | 0b00101 | 0b01001 | 0b01011 => true,
                0b00110 | 0b00111 | 0b01000 | 0b01010 | 0b10000 | 0b10001 => size == 3,
                0b10110 => size == 1 || size == 2,
                _ => false,
            };
            if !ok {
                return Step::Illegal;
            }
        } else {
            if size == 3 && !q {
                return Step::Illegal;
            }
            let ok = match opcode {
                0b00000 | 0b00010 | 0b00100 | 0b01100 | 0b01101 | 0b01110 | 0b01111 | 0b10010
                | 0b10100 | 0b10101 => size != 3,
                0b10011 => {
                    if u == 1 {
                        size == 0
                    } else {
                        size != 3
                    }
                }
                0b10110 => size == 1 || size == 2,
                0b10111 => u == 0,
                _ => true,
            };
            if !ok {
                return Step::Illegal;
            }
        }
        let lanes = if scalar {
            1
        } else {
            (if q { 128 } else { 64 }) / esize
        };
        let unsigned = u == 1;
        let pairwise = matches!(opcode, 0b10100 | 0b10101 | 0b10111);
        let mut r = 0u128;
        let mut qc = false;
        for i in 0..lanes {
            let (a, b) = if pairwise {
                let half = lanes / 2;
                let src = if i < half { n } else { m };
                let j = (i % half) * 2;
                (elem(src, j, esize), elem(src, j + 1, esize))
            } else {
                (elem(n, i, esize), elem(m, i, esize))
            };
            let (ia, ib) = (ival(a, esize, unsigned), ival(b, esize, unsigned));
            let x = match opcode {
                0b00000 => ((ia + ib) >> 1) as u64,
                0b00001 | 0b00101 => {
                    let v = if opcode == 0b00001 { ia + ib } else { ia - ib };
                    let (x, s) = sat(v, esize, unsigned);
                    qc |= s;
                    x
                }
                0b00010 => ((ia + ib + 1) >> 1) as u64,
                0b00100 => ((ia - ib) >> 1) as u64,
                0b00110 => mask_if(ia > ib, esize),
                0b00111 => mask_if(ia >= ib, esize),
                0b01000..=0b01011 => {
                    let sh = sign_extend(b & 0xff, 8);
                    let (x, s) =
                        shift_lane(a, sh, esize, unsigned, opcode & 2 != 0, opcode & 1 != 0);
                    qc |= s;
                    x
                }
                0b01100 | 0b10100 => {
                    if ia >= ib {
                        a
                    } else {
                        b
                    }
                }
                0b01101 | 0b10101 => {
                    if ia <= ib {
                        a
                    } else {
                        b
                    }
                }
                0b01110 => (ia - ib).unsigned_abs() as u64,
                0b01111 => elem(d, i, esize).wrapping_add((ia - ib).unsigned_abs() as u64),
                0b10000 => {
                    if unsigned {
                        a.wrapping_sub(b)
                    } else {
                        a.wrapping_add(b)
                    }
                }
                0b10001 => {
                    if unsigned {
                        mask_if(a == b, esize)
                    } else {
                        mask_if(a & b != 0, esize)
                    }
                }
                0b10010 => {
                    let p = a.wrapping_mul(b);
                    let acc = elem(d, i, esize);
                    if unsigned {
                        acc.wrapping_sub(p)
                    } else {
                        acc.wrapping_add(p)
                    }
                }
                0b10011 => {
                    if unsigned {
                        pmul8(a, b)
                    } else {
                        a.wrapping_mul(b)
                    }
                }
                0b10110 => {
                    let sa = ival(a, esize, false);
                    let sb = ival(b, esize, false);
                    let round = if unsigned { 1i128 << (esize - 1) } else { 0 };
                    let (x, s) = sat((2 * sa * sb + round) >> esize, esize, false);
                    qc |= s;
                    x
                }
                _ => a.wrapping_add(b), // 0b10111 ADDP
            };
            r |= u128::from(x & ones(esize)) << (i * esize);
        }
        self.set_qc(qc);
        self.set_vec(rd, r, q || scalar);
        Step::Next
    }

    fn simd_fp_three_same(&mut self, instr: u32, scalar: bool) -> Step {
        let q = (instr >> 30) & 1 == 1;
        let u = (instr >> 29) & 1;
        let a_bit = (instr >> 23) & 1;
        let sz = (instr >> 22) & 1;
        let key = (u << 4) | (a_bit << 3) | ((instr >> 11) & 7);
        let (fmt, esize, lanes) = if scalar {
            if !matches!(
                key,
                0b0_0011
                    | 0b0_0100
                    | 0b0_0111
                    | 0b0_1111
                    | 0b1_0100
                    | 0b1_0101
                    | 0b1_1010
                    | 0b1_1100
                    | 0b1_1101
            ) {
                return Step::Illegal;
            }
            if sz == 1 { (D, 64, 1) } else { (S, 32, 1) }
        } else {
            let Some(x) = fp_vec(q, sz) else {
                return Step::Illegal;
            };
            x
        };
        if matches!(
            key,
            0b0_0101 | 0b0_1011 | 0b0_1100 | 0b0_1101 | 0b1_0001 | 0b1_1001 | 0b1_1011 | 0b1_1111
        ) {
            return Step::Illegal; // FMLAL/FMLSL (FHM) and unallocated
        }
        let (rd, rn, rm) = (
            reg_field(instr, 0),
            reg_field(instr, 5),
            reg_field(instr, 16),
        );
        let (n, m, d) = (self.v[rn], self.v[rm], self.v[rd]);
        let pairwise = matches!(key, 0b1_0000 | 0b1_0010 | 0b1_0110 | 0b1_1000 | 0b1_1110);
        let sign = 1u64 << (esize - 1);
        let mut env = self.fpenv();
        let e = &mut env;
        let mut r = 0u128;
        for i in 0..lanes {
            let (a, b) = if pairwise {
                let half = lanes / 2;
                let src = if i < half { n } else { m };
                let j = (i % half) * 2;
                (elem(src, j, esize), elem(src, j + 1, esize))
            } else {
                (elem(n, i, esize), elem(m, i, esize))
            };
            let x = match key {
                0b0_0000 | 0b1_0000 => fpu::max_min(a, b, true, true, fmt, e),
                0b0_0001 => fpu::mul_add(elem(d, i, esize), a, b, fmt, e),
                0b0_0010 | 0b1_0010 => fpu::add(a, b, false, fmt, e),
                0b0_0011 => fpu::mul(a, b, true, fmt, e),
                0b0_0100 => mask_if(fpu::compare_op(a, b, Cmp::Eq, fmt, e), esize),
                0b0_0110 | 0b1_0110 => fpu::max_min(a, b, true, false, fmt, e),
                0b0_0111 => fpu::step_fused(a, b, false, fmt, e),
                0b0_1000 | 0b1_1000 => fpu::max_min(a, b, false, true, fmt, e),
                0b0_1001 => fpu::mul_add(elem(d, i, esize), a ^ sign, b, fmt, e),
                0b0_1010 => fpu::add(a, b, true, fmt, e),
                0b0_1110 | 0b1_1110 => fpu::max_min(a, b, false, false, fmt, e),
                0b0_1111 => fpu::step_fused(a, b, true, fmt, e),
                0b1_0011 => fpu::mul(a, b, false, fmt, e),
                0b1_0100 => mask_if(fpu::compare_op(a, b, Cmp::Ge, fmt, e), esize),
                0b1_0101 => mask_if(
                    fpu::compare_op(a & !sign, b & !sign, Cmp::Ge, fmt, e),
                    esize,
                ),
                0b1_0111 => fpu::div(a, b, fmt, e),
                0b1_1010 => fpu::add(a, b, true, fmt, e) & !sign,
                0b1_1100 => mask_if(fpu::compare_op(a, b, Cmp::Gt, fmt, e), esize),
                _ => mask_if(
                    fpu::compare_op(a & !sign, b & !sign, Cmp::Gt, fmt, e),
                    esize,
                ), // 0b1_1101 FACGT
            };
            r |= u128::from(x) << (i * esize);
        }
        self.set_fpenv(env);
        self.set_vec(rd, r, q || scalar);
        Step::Next
    }

    // ---- three different -----------------------------------------------------

    fn simd_three_diff(&mut self, instr: u32, scalar: bool) -> Step {
        let q = (instr >> 30) & 1 == 1;
        let u = (instr >> 29) & 1;
        let size = (instr >> 22) & 3;
        let opcode = (instr >> 12) & 0xf;
        let (rd, rn, rm) = (
            reg_field(instr, 0),
            reg_field(instr, 5),
            reg_field(instr, 16),
        );
        let (n, m, d) = (self.v[rn], self.v[rm], self.v[rd]);
        if opcode == 0b1110 && u == 0 && !scalar {
            // PMULL/PMULL2: 8x8->16 (size 00) or 64x64->128 (size 11).
            let half = |v: u128| if q { (v >> 64) as u64 } else { v as u64 };
            let (a, b) = (half(n), half(m));
            self.v[rd] = match size {
                0 => (0..8).fold(0u128, |acc, i| {
                    acc | (u128::from(pmul8((a >> (8 * i)) & 0xff, (b >> (8 * i)) & 0xff))
                        << (16 * i))
                }),
                3 => crypto::clmul64(a, b),
                _ => return Step::Illegal,
            };
            return Step::Next;
        }
        if size == 3 || opcode == 0b1111 {
            return Step::Illegal;
        }
        let sqd = matches!(opcode, 0b1001 | 0b1011 | 0b1101);
        if sqd && (u == 1 || size == 0) {
            return Step::Illegal;
        }
        if scalar && !sqd {
            return Step::Illegal;
        }
        if opcode == 0b1110 {
            return Step::Illegal; // PMULL with U=1
        }
        let esize = 8u32 << size;
        let wide = esize * 2;
        let unsigned = u == 1;
        let lanes = if scalar { 1 } else { 64 / esize };
        let part = if q && !scalar { 64 } else { 0 };
        let narrow_src = |v: u128, i: u32| elem(v >> part, i, esize);
        let mut qc = false;
        if matches!(opcode, 0b0100 | 0b0110) {
            // ADDHN/RADDHN/SUBHN/RSUBHN: wide + wide -> high narrow half.
            let mut r = 0u128;
            for i in 0..lanes {
                let (a, b) = (elem(n, i, wide), elem(m, i, wide));
                let mut s = if opcode == 0b0100 {
                    a.wrapping_add(b)
                } else {
                    a.wrapping_sub(b)
                };
                if unsigned {
                    s = s.wrapping_add(1 << (esize - 1));
                }
                r |= u128::from((s >> esize) & ones(esize)) << (i * esize);
            }
            self.v[rd] = if q {
                (d & u128::from(u64::MAX)) | (r << 64)
            } else {
                r
            };
            return Step::Next;
        }
        let mut r = 0u128;
        for i in 0..lanes {
            let x = match opcode {
                0b0001 | 0b0011 => ival(elem(n, i, wide), wide, unsigned),
                _ => ival(narrow_src(n, i), esize, unsigned),
            };
            let y = ival(narrow_src(m, i), esize, unsigned);
            let acc = elem(d, i, wide);
            let out: u64 = match opcode {
                0b0000 | 0b0001 => (x + y) as u64,
                0b0010 | 0b0011 => (x - y) as u64,
                0b0101 => acc.wrapping_add((x - y).unsigned_abs() as u64),
                0b0111 => (x - y).unsigned_abs() as u64,
                0b1000 => acc.wrapping_add((x * y) as u64),
                0b1010 => acc.wrapping_sub((x * y) as u64),
                0b1100 => (x * y) as u64,
                _ => {
                    // SQDMLAL/SQDMLSL/SQDMULL
                    let (p, s1) = sat(2 * x * y, wide, false);
                    qc |= s1;
                    if opcode == 0b1101 {
                        p
                    } else {
                        let pv = ival(p, wide, false);
                        let av = ival(acc, wide, false);
                        let v = if opcode == 0b1001 { av + pv } else { av - pv };
                        let (o, s2) = sat(v, wide, false);
                        qc |= s2;
                        o
                    }
                }
            };
            r |= u128::from(out & ones(wide)) << (i * wide);
        }
        self.set_qc(qc);
        self.v[rd] = r;
        Step::Next
    }

    // ---- two-register miscellaneous --------------------------------------------

    fn simd_misc(&mut self, instr: u32, scalar: bool) -> Step {
        let q = (instr >> 30) & 1 == 1;
        let u = (instr >> 29) & 1;
        let size = (instr >> 22) & 3;
        let opcode = (instr >> 12) & 0x1f;
        let (rd, rn) = (reg_field(instr, 0), reg_field(instr, 5));
        let (n, d) = (self.v[rn], self.v[rd]);
        // FP forms: opcode 1xxxx (except the narrow/long ones) and 011xx with
        // size<1> set.
        let fp_form = (opcode >= 0b11000) || (opcode >= 0b01100 && size >= 2 && opcode < 0b10000);
        if fp_form {
            return self.simd_fp_misc(instr, scalar);
        }
        if matches!(opcode, 0b10110 | 0b10111) {
            return self.simd_fcvt_narrow_long(instr, scalar);
        }
        let esize = 8u32 << size;
        let unsigned = u == 1;
        // Narrowing: XTN/SQXTUN/SQXTN/UQXTN.
        if opcode == 0b10010 || opcode == 0b10100 {
            if size == 3 || (scalar && opcode == 0b10010 && u == 0) {
                return Step::Illegal;
            }
            let wide = esize * 2;
            let lanes = if scalar { 1 } else { 64 / esize };
            let mut r = 0u128;
            let mut qc = false;
            for i in 0..lanes {
                let x = elem(n, i, wide);
                let out = match (opcode, u) {
                    (0b10010, 0) => x,
                    (0b10010, _) => {
                        let (o, s) = sat(ival(x, wide, false), esize, true);
                        qc |= s;
                        o
                    }
                    _ => {
                        let (o, s) = sat(ival(x, wide, unsigned), esize, unsigned);
                        qc |= s;
                        o
                    }
                };
                r |= u128::from(out & ones(esize)) << (i * esize);
            }
            self.set_qc(qc);
            self.v[rd] = if q && !scalar {
                (d & u128::from(u64::MAX)) | (r << 64)
            } else {
                r
            };
            return Step::Next;
        }
        if opcode == 0b10011 {
            // SHLL/SHLL2 (U=1 only).
            if u == 0 || scalar || size == 3 {
                return Step::Illegal;
            }
            let src = if q { n >> 64 } else { n };
            let mut r = 0u128;
            for i in 0..64 / esize {
                r |= u128::from(elem(src, i, esize) << esize) << (i * 2 * esize);
            }
            self.v[rd] = r;
            return Step::Next;
        }
        if scalar {
            let ok = match opcode {
                0b00011 | 0b00111 => true,
                0b01000 | 0b01001 | 0b01011 => size == 3,
                0b01010 => size == 3 && u == 0,
                _ => false,
            };
            if !ok {
                return Step::Illegal;
            }
        } else {
            if size == 3 && !q {
                return Step::Illegal;
            }
            let ok = match (u, opcode) {
                (0, 0b00000) | (_, 0b00010 | 0b00100 | 0b00110) => size != 3,
                (1, 0b00000 | 0b00101) => size < 2,
                (0, 0b00001 | 0b00101) => size == 0,
                (_, 0b00011 | 0b00111 | 0b01000 | 0b01001 | 0b01011) | (0, 0b01010) => true,
                _ => false,
            };
            if !ok {
                return Step::Illegal;
            }
        }
        let width = if scalar {
            esize
        } else if q {
            128
        } else {
            64
        };
        let lanes = width / esize;
        let mut qc = false;
        let r: u128 = match (u, opcode) {
            (_, 0b00000) | (0, 0b00001) => {
                // REV64 / REV32 / REV16: reverse elements within containers.
                let container = match (u, opcode) {
                    (0, 0b00000) => 64,
                    (1, _) => 32,
                    _ => 16,
                };
                let per = container / esize;
                let mut r = 0u128;
                for i in 0..lanes {
                    let base = i - i % per;
                    let j = base + (per - 1 - i % per);
                    r |= u128::from(elem(n, i, esize)) << (j * esize);
                }
                r
            }
            (_, 0b00010 | 0b00110) => {
                // SADDLP/UADDLP/SADALP/UADALP.
                let wide = esize * 2;
                let mut r = 0u128;
                for i in 0..lanes / 2 {
                    let s = ival(elem(n, 2 * i, esize), esize, unsigned)
                        + ival(elem(n, 2 * i + 1, esize), esize, unsigned);
                    let mut x = s as u64;
                    if opcode == 0b00110 {
                        x = x.wrapping_add(elem(d, i, wide));
                    }
                    r |= u128::from(x & ones(wide)) << (i * wide);
                }
                r
            }
            (0, 0b00101) => {
                let mut r = 0u128;
                for i in 0..lanes {
                    r |= u128::from(elem(n, i, 8).count_ones()) << (i * 8);
                }
                r
            }
            (1, 0b00101) => {
                if size == 0 {
                    !n
                } else {
                    let mut r = 0u128;
                    for i in 0..width / 8 {
                        r |= u128::from((elem(n, i, 8) as u8).reverse_bits()) << (i * 8);
                    }
                    r
                }
            }
            _ => {
                let mut r = 0u128;
                for i in 0..lanes {
                    let a = elem(n, i, esize);
                    let sa = ival(a, esize, false);
                    let x = match (u, opcode) {
                        (0, 0b00011) => {
                            // SUQADD: signed accumulate of an unsigned value.
                            let (o, s) = sat(
                                ival(elem(d, i, esize), esize, false) + i128::from(a),
                                esize,
                                false,
                            );
                            qc |= s;
                            o
                        }
                        (_, 0b00011) => {
                            // USQADD
                            let (o, s) = sat(i128::from(elem(d, i, esize)) + sa, esize, true);
                            qc |= s;
                            o
                        }
                        (0, 0b00100) => {
                            // CLS
                            let x = if sa < 0 { !a & ones(esize) } else { a };
                            u64::from(x.leading_zeros() - (64 - esize)) - 1
                        }
                        (_, 0b00100) => u64::from(a.leading_zeros() - (64 - esize)),
                        (0, 0b00111) => {
                            let (o, s) = sat(sa.abs(), esize, false);
                            qc |= s;
                            o
                        }
                        (_, 0b00111) => {
                            let (o, s) = sat(-sa, esize, false);
                            qc |= s;
                            o
                        }
                        (0, 0b01000) => mask_if(sa > 0, esize),
                        (_, 0b01000) => mask_if(sa >= 0, esize),
                        (0, 0b01001) => mask_if(sa == 0, esize),
                        (_, 0b01001) => mask_if(sa <= 0, esize),
                        (_, 0b01010) => mask_if(sa < 0, esize),
                        (0, _) => sa.unsigned_abs() as u64, // ABS
                        _ => (sa.wrapping_neg()) as u64,    // NEG
                    };
                    r |= u128::from(x & ones(esize)) << (i * esize);
                }
                r
            }
        };
        self.set_qc(qc);
        self.set_vec(rd, r, q || scalar);
        Step::Next
    }

    /// `FCVTN`/`FCVTN2`/`FCVTXN`/`FCVTXN2`/`FCVTL`/`FCVTL2`.
    fn simd_fcvt_narrow_long(&mut self, instr: u32, scalar: bool) -> Step {
        let q = (instr >> 30) & 1 == 1;
        let u = (instr >> 29) & 1;
        let size = (instr >> 22) & 3;
        let opcode = (instr >> 12) & 0x1f;
        let (rd, rn) = (reg_field(instr, 0), reg_field(instr, 5));
        if size >= 2 {
            return Step::Illegal; // BFCVTN (BF16) and unallocated
        }
        let sz = size & 1;
        let (n, d) = (self.v[rn], self.v[rd]);
        let mut env = self.fpenv();
        let r = match (u, opcode) {
            (_, 0b10110) => {
                // FCVTN (U=0) / FCVTXN (U=1, double -> single, round to odd).
                if u == 1 && sz == 0 {
                    return Step::Illegal;
                }
                if scalar && u == 0 {
                    return Step::Illegal;
                }
                let (from, to, fe, te) = if sz == 1 {
                    (D, S, 64, 32)
                } else {
                    (S, H, 32, 16)
                };
                let mode = if u == 1 {
                    Rounding::Odd
                } else {
                    env.rounding()
                };
                let lanes = if scalar { 1 } else { 64 / te };
                let mut r = 0u128;
                for i in 0..lanes {
                    let x = fpu::convert(elem(n, i, fe), from, to, mode, &mut env);
                    r |= u128::from(x) << (i * te);
                }
                if q && !scalar {
                    (d & u128::from(u64::MAX)) | (r << 64)
                } else {
                    r
                }
            }
            (0, _) => {
                // FCVTL/FCVTL2
                if scalar {
                    return Step::Illegal;
                }
                let (from, to, fe, te) = if sz == 1 {
                    (S, D, 32, 64)
                } else {
                    (H, S, 16, 32)
                };
                let src = if q { n >> 64 } else { n };
                let mut r = 0u128;
                let mode = env.rounding();
                for i in 0..128 / te {
                    let x = fpu::convert(elem(src, i, fe), from, to, mode, &mut env);
                    r |= u128::from(x) << (i * te);
                }
                r
            }
            _ => return Step::Illegal,
        };
        self.set_fpenv(env);
        self.v[rd] = r;
        Step::Next
    }

    /// Floating-point two-register miscellaneous (vector and scalar).
    fn simd_fp_misc(&mut self, instr: u32, scalar: bool) -> Step {
        let q = (instr >> 30) & 1 == 1;
        let u = (instr >> 29) & 1;
        let a = (instr >> 23) & 1;
        let sz = (instr >> 22) & 1;
        let opcode = (instr >> 12) & 0x1f;
        let (rd, rn) = (reg_field(instr, 0), reg_field(instr, 5));
        let key = (u << 6) | (a << 5) | opcode;
        // Which ops exist, and which also have a scalar form.
        let (exists, scalar_ok) = match key {
            // U=0 a=0: FRINTN FRINTM FCVTNS FCVTMS FCVTAS SCVTF
            0b0_0_11000 | 0b0_0_11001 => (true, false),
            0b0_0_11010..=0b0_0_11101 => (true, true),
            // U=0 a=1: FCMGT0 FCMEQ0 FCMLT0 FABS FRINTP FRINTZ FCVTPS FCVTZS
            // URECPE FRECPE FRECPX
            0b0_1_01100..=0b0_1_01110 => (true, true),
            0b0_1_01111 | 0b0_1_11000 | 0b0_1_11001 => (true, false),
            0b0_1_11010 | 0b0_1_11011 | 0b0_1_11101 => (true, true),
            0b0_1_11100 => (sz == 0, false),
            0b0_1_11111 => (scalar, true),
            // U=1 a=0: FRINTA FRINTX FCVTNU FCVTMU FCVTAU UCVTF
            0b1_0_11000 | 0b1_0_11001 => (true, false),
            0b1_0_11010..=0b1_0_11101 => (true, true),
            // U=1 a=1: FCMGE0 FCMLE0 FNEG FRINTI FCVTPU FCVTZU URSQRTE FRSQRTE
            // FSQRT
            0b1_1_01100 | 0b1_1_01101 => (true, true),
            0b1_1_01111 | 0b1_1_11001 | 0b1_1_11111 => (true, false),
            0b1_1_11010 | 0b1_1_11011 | 0b1_1_11101 => (true, true),
            0b1_1_11100 => (sz == 0, false),
            _ => (false, false),
        };
        if !exists || (scalar && !scalar_ok) {
            return Step::Illegal;
        }
        let (fmt, esize, lanes) = if scalar {
            if sz == 1 { (D, 64, 1) } else { (S, 32, 1) }
        } else {
            let Some(x) = fp_vec(q, sz) else {
                return Step::Illegal;
            };
            x
        };
        let n = self.v[rn];
        let sign = 1u64 << (esize - 1);
        let mut env = self.fpenv();
        let e = &mut env;
        let rm = e.rounding();
        let mut r = 0u128;
        for i in 0..lanes {
            let x = elem(n, i, esize);
            let unsigned = u == 1;
            let out = match key {
                0b0_0_11000 => fpu::round_int(x, Rounding::TieEven, false, fmt, e),
                0b0_0_11001 => fpu::round_int(x, Rounding::NegInf, false, fmt, e),
                0b0_1_11000 => fpu::round_int(x, Rounding::PosInf, false, fmt, e),
                0b0_1_11001 => fpu::round_int(x, Rounding::Zero, false, fmt, e),
                0b1_0_11000 => fpu::round_int(x, Rounding::TieAway, false, fmt, e),
                0b1_0_11001 => fpu::round_int(x, rm, true, fmt, e),
                0b1_1_11001 => fpu::round_int(x, rm, false, fmt, e),
                0b0_0_11010 | 0b1_0_11010 => {
                    fpu::to_fixed(x, fmt, 0, unsigned, esize, Rounding::TieEven, e)
                }
                0b0_0_11011 | 0b1_0_11011 => {
                    fpu::to_fixed(x, fmt, 0, unsigned, esize, Rounding::NegInf, e)
                }
                0b0_0_11100 | 0b1_0_11100 => {
                    fpu::to_fixed(x, fmt, 0, unsigned, esize, Rounding::TieAway, e)
                }
                0b0_1_11010 | 0b1_1_11010 => {
                    fpu::to_fixed(x, fmt, 0, unsigned, esize, Rounding::PosInf, e)
                }
                0b0_1_11011 | 0b1_1_11011 => {
                    fpu::to_fixed(x, fmt, 0, unsigned, esize, Rounding::Zero, e)
                }
                0b0_0_11101 | 0b1_0_11101 => fpu::from_fixed(x, esize, !unsigned, 0, fmt, rm, e),
                0b0_1_01100 => mask_if(fpu::compare_op(x, 0, Cmp::Gt, fmt, e), esize),
                0b0_1_01101 => mask_if(fpu::compare_op(x, 0, Cmp::Eq, fmt, e), esize),
                0b0_1_01110 => mask_if(fpu::compare_op(0, x, Cmp::Gt, fmt, e), esize),
                0b1_1_01100 => mask_if(fpu::compare_op(x, 0, Cmp::Ge, fmt, e), esize),
                0b1_1_01101 => mask_if(fpu::compare_op(0, x, Cmp::Ge, fmt, e), esize),
                0b0_1_01111 => x & !sign,
                0b1_1_01111 => x ^ sign,
                0b0_1_11100 => u64::from(fpu::urecpe(x as u32)),
                0b1_1_11100 => u64::from(fpu::ursqrte(x as u32)),
                0b0_1_11101 => fpu::recip_est(x, fmt, e),
                0b1_1_11101 => fpu::rsqrt_est(x, fmt, e),
                0b0_1_11111 => fpu::recpx(x, fmt, e),
                _ => fpu::sqrt(x, fmt, e), // 0b1_1_11111
            };
            r |= u128::from(out) << (i * esize);
        }
        self.set_fpenv(env);
        self.set_vec(rd, r, q || scalar);
        Step::Next
    }

    // ---- across lanes / scalar pairwise ----------------------------------------

    fn simd_across(&mut self, instr: u32) -> Step {
        let q = (instr >> 30) & 1 == 1;
        let u = (instr >> 29) & 1;
        let size = (instr >> 22) & 3;
        let opcode = (instr >> 12) & 0x1f;
        let (rd, rn) = (reg_field(instr, 0), reg_field(instr, 5));
        let n = self.v[rn];
        if opcode == 0b01100 || opcode == 0b01111 {
            // FMAXNMV/FMINNMV/FMAXV/FMINV: 4S only (U=0 is FP16).
            if u == 0 || size & 1 != 0 || !q {
                return Step::Illegal;
            }
            let max = size >> 1 == 0;
            let num = opcode == 0b01100;
            let mut env = self.fpenv();
            let e = &mut env;
            let lo = fpu::max_min(elem(n, 0, 32), elem(n, 1, 32), max, num, S, e);
            let hi = fpu::max_min(elem(n, 2, 32), elem(n, 3, 32), max, num, S, e);
            let r = fpu::max_min(lo, hi, max, num, S, e);
            self.set_fpenv(env);
            self.v[rd] = u128::from(r);
            return Step::Next;
        }
        if size == 3 || (size == 2 && !q) {
            return Step::Illegal;
        }
        let esize = 8u32 << size;
        let lanes = (if q { 128 } else { 64 }) / esize;
        let unsigned = u == 1;
        let vals = (0..lanes).map(|i| ival(elem(n, i, esize), esize, unsigned));
        let (r, width) = match (u, opcode) {
            (_, 0b00011) => (vals.sum::<i128>() as u64, esize * 2), // SADDLV/UADDLV
            (_, 0b01010) => (vals.max().unwrap_or(0) as u64, esize), // SMAXV/UMAXV
            (_, 0b11010) => (vals.min().unwrap_or(0) as u64, esize), // SMINV/UMINV
            (0, 0b11011) => (vals.sum::<i128>() as u64, esize),     // ADDV
            _ => return Step::Illegal,
        };
        self.v[rd] = u128::from(r & ones(width));
        Step::Next
    }

    /// Scalar pairwise: `ADDP` (D) and `FADDP`/`FMAXP`/`FMINP`/`FMAXNMP`/
    /// `FMINNMP` (S/D).
    fn simd_scalar_pairwise(&mut self, instr: u32) -> Step {
        let u = (instr >> 29) & 1;
        let size = (instr >> 22) & 3;
        let opcode = (instr >> 12) & 0x1f;
        let (rd, rn) = (reg_field(instr, 0), reg_field(instr, 5));
        let n = self.v[rn];
        if u == 0 {
            if opcode != 0b11011 || size != 3 {
                return Step::Illegal; // (U=0 FP forms are FP16)
            }
            self.v[rd] = u128::from((n as u64).wrapping_add((n >> 64) as u64));
            return Step::Next;
        }
        let (fmt, esize) = if size & 1 == 1 { (D, 64) } else { (S, 32) };
        let (a, b) = (elem(n, 0, esize), elem(n, 1, esize));
        let min = size >> 1 == 1;
        let mut env = self.fpenv();
        let r = match (opcode, min) {
            (0b01100, _) => fpu::max_min(a, b, !min, true, fmt, &mut env),
            (0b01101, false) => fpu::add(a, b, false, fmt, &mut env),
            (0b01111, _) => fpu::max_min(a, b, !min, false, fmt, &mut env),
            _ => return Step::Illegal,
        };
        self.set_fpenv(env);
        self.v[rd] = u128::from(r);
        Step::Next
    }

    // ---- copy / modified immediate -------------------------------------------

    fn simd_copy(&mut self, instr: u32, scalar: bool) -> Step {
        let q = (instr >> 30) & 1 == 1;
        let op = (instr >> 29) & 1;
        let imm5 = (instr >> 16) & 0x1f;
        let imm4 = (instr >> 11) & 0xf;
        let (rd, rn) = (reg_field(instr, 0), reg_field(instr, 5));
        let size = imm5.trailing_zeros();
        if size > 3 {
            return Step::Illegal;
        }
        let esize = 8u32 << size;
        let index = imm5 >> (size + 1);
        if scalar {
            // DUP (element), scalar form.
            if op != 0 || imm4 != 0 {
                return Step::Illegal;
            }
            self.v[rd] = u128::from(elem(self.v[rn], index, esize));
            return Step::Next;
        }
        if op == 1 {
            // INS (element)
            if !q {
                return Step::Illegal;
            }
            let src = imm4 >> size;
            let x = elem(self.v[rn], src, esize);
            let sh = index * esize;
            self.v[rd] = (self.v[rd] & !(u128::from(ones(esize)) << sh)) | (u128::from(x) << sh);
            return Step::Next;
        }
        match imm4 {
            0b0000 | 0b0001 => {
                // DUP (element) / DUP (general)
                if size == 3 && !q {
                    return Step::Illegal;
                }
                let x = if imm4 == 0 {
                    elem(self.v[rn], index, esize)
                } else {
                    self.read_x(rn) & ones(esize)
                };
                let mut r = 0u128;
                for i in 0..128 / esize {
                    r |= u128::from(x) << (i * esize);
                }
                self.set_vec(rd, r, q);
            }
            0b0011 => {
                // INS (general)
                if !q {
                    return Step::Illegal;
                }
                let x = self.read_x(rn) & ones(esize);
                let sh = index * esize;
                self.v[rd] =
                    (self.v[rd] & !(u128::from(ones(esize)) << sh)) | (u128::from(x) << sh);
            }
            0b0101 => {
                // SMOV
                if size == 3 || (size == 2 && !q) {
                    return Step::Illegal;
                }
                let x = sign_extend(elem(self.v[rn], index, esize), esize) as u64;
                self.write_x(rd, if q { x } else { x & 0xffff_ffff });
            }
            0b0111 => {
                // UMOV
                if q != (size == 3) {
                    return Step::Illegal;
                }
                self.write_x(rd, elem(self.v[rn], index, esize));
            }
            _ => return Step::Illegal,
        }
        Step::Next
    }

    fn simd_mod_imm(&mut self, instr: u32) -> Step {
        let q = (instr >> 30) & 1 == 1;
        let op = (instr >> 29) & 1;
        let cmode = (instr >> 12) & 0xf;
        let o2 = (instr >> 11) & 1;
        let rd = reg_field(instr, 0);
        let imm8 = (((instr >> 16) & 7) << 5) | ((instr >> 5) & 0x1f);
        if o2 != 0 || (cmode == 0b1111 && op == 1 && !q) {
            return Step::Illegal; // FMOV (half) is FP16
        }
        let imm = adv_simd_expand_imm(op, cmode, imm8);
        let imm128 = u128::from(imm) | (u128::from(imm) << 64);
        let d = self.v[rd];
        let r = if cmode < 0b1100 && cmode & 1 == 1 {
            // ORR/BIC (vector, immediate)
            if op == 0 { d | imm128 } else { d & !imm128 }
        } else if op == 1 && cmode < 0b1110 {
            !imm128 // MVNI
        } else {
            imm128 // MOVI / FMOV
        };
        self.set_vec(rd, r, q);
        Step::Next
    }

    // ---- shift by immediate -------------------------------------------------------

    fn simd_shift_imm(&mut self, instr: u32, scalar: bool) -> Step {
        let q = (instr >> 30) & 1 == 1;
        let u = (instr >> 29) & 1;
        let immh = (instr >> 19) & 0xf;
        let immhb = (instr >> 16) & 0x7f;
        let opcode = (instr >> 11) & 0x1f;
        let (rd, rn) = (reg_field(instr, 0), reg_field(instr, 5));
        let (n, d) = (self.v[rn], self.v[rd]);
        let hsb = immh.ilog2();
        let esize = 8u32 << hsb;
        let unsigned = u == 1;
        let narrow = matches!(opcode, 0b10000..=0b10011);
        let long = opcode == 0b10100;
        if narrow || long {
            if immh & 8 != 0 {
                return Step::Illegal;
            }
            if scalar && (long || (opcode & 0b10010 == 0b10000 && u == 0)) {
                return Step::Illegal; // SSHLL, SHRN/RSHRN have no scalar form
            }
            let wide = esize * 2;
            if long {
                // SSHLL/USHLL (SXTL/UXTL aliases)
                let shift = immhb - esize;
                let src = if q { n >> 64 } else { n };
                let mut r = 0u128;
                for i in 0..64 / esize {
                    let x = ival(elem(src, i, esize), esize, unsigned) << shift;
                    r |= u128::from(x as u64 & ones(wide)) << (i * wide);
                }
                self.v[rd] = r;
                return Step::Next;
            }
            let shift = 2 * esize - immhb;
            let round = opcode & 1 == 1;
            let lanes = if scalar { 1 } else { 64 / esize };
            let mut r = 0u128;
            let mut qc = false;
            for i in 0..lanes {
                // SHRN/RSHRN (U=0, 1000x) are plain truncations; SQSHRUN/
                // SQRSHRUN (U=1, 1000x) take signed input, saturate unsigned;
                // SQSHRN/UQSHRN/SQRSHRN/UQRSHRN (1001x) saturate in kind.
                let (src_unsigned, out) = match (opcode & 0b10, u) {
                    (0, 0) => (true, None),
                    (0, _) => (false, Some(true)),
                    (_, _) => (unsigned, Some(unsigned)),
                };
                let x = ival(elem(n, i, wide), wide, src_unsigned);
                let v = if round {
                    (x + (1i128 << (shift - 1))) >> shift
                } else {
                    x >> shift
                };
                let o = match out {
                    None => v as u64 & ones(esize),
                    Some(us) => {
                        let (o, s) = sat(v, esize, us);
                        qc |= s;
                        o
                    }
                };
                r |= u128::from(o) << (i * esize);
            }
            self.set_qc(qc);
            self.v[rd] = if q && !scalar {
                (d & u128::from(u64::MAX)) | (r << 64)
            } else {
                r
            };
            return Step::Next;
        }
        if matches!(opcode, 0b11100 | 0b11111) {
            // SCVTF/UCVTF/FCVTZS/FCVTZU (vector/scalar, fixed-point).
            let fmt = match esize {
                32 => S,
                64 => D,
                _ => return Step::Illegal, // FP16 / reserved
            };
            if !scalar && esize == 64 && !q {
                return Step::Illegal;
            }
            let fbits = 2 * esize - immhb;
            let lanes = if scalar {
                1
            } else {
                (if q { 128 } else { 64 }) / esize
            };
            let mut env = self.fpenv();
            let rm = env.rounding();
            let mut r = 0u128;
            for i in 0..lanes {
                let x = elem(n, i, esize);
                let o = if opcode == 0b11100 {
                    fpu::from_fixed(x, esize, !unsigned, fbits, fmt, rm, &mut env)
                } else {
                    fpu::to_fixed(x, fmt, fbits, unsigned, esize, Rounding::Zero, &mut env)
                };
                r |= u128::from(o) << (i * esize);
            }
            self.set_fpenv(env);
            self.set_vec(rd, r, q || scalar);
            return Step::Next;
        }
        let only64 = matches!(
            opcode,
            0b00000 | 0b00010 | 0b00100 | 0b00110 | 0b01000 | 0b01010
        );
        if scalar && only64 && esize != 64 {
            return Step::Illegal;
        }
        if !scalar && esize == 64 && !q {
            return Step::Illegal;
        }
        let ok = match opcode {
            0b00000 | 0b00010 | 0b00100 | 0b00110 | 0b01010 | 0b01110 => true,
            0b01000 | 0b01100 => u == 1,
            _ => false,
        };
        if !ok {
            return Step::Illegal;
        }
        let rshift = 2 * esize - immhb; // 1..=esize
        let lshift = immhb - esize; // 0..esize
        let lanes = if scalar {
            1
        } else {
            (if q { 128 } else { 64 }) / esize
        };
        let mut r = 0u128;
        let mut qc = false;
        for i in 0..lanes {
            let x = elem(n, i, esize);
            let acc = elem(d, i, esize);
            let o = match opcode {
                0b00000..=0b00110 => {
                    // SSHR/USHR, SSRA/USRA, SRSHR/URSHR, SRSRA/URSRA
                    let v = ival(x, esize, unsigned);
                    let s = if opcode & 0b100 != 0 {
                        (v + (1i128 << (rshift - 1))) >> rshift
                    } else {
                        v >> rshift
                    } as u64;
                    if opcode & 0b010 != 0 {
                        acc.wrapping_add(s)
                    } else {
                        s
                    }
                }
                0b01000 => {
                    // SRI
                    let mask = if rshift >= 64 {
                        0
                    } else {
                        ones(esize) >> rshift
                    };
                    let s = if rshift >= 64 { 0 } else { x >> rshift };
                    (acc & !mask) | (s & mask)
                }
                0b01010 => {
                    let s = x << lshift;
                    if unsigned {
                        let mask = ones(esize) << lshift;
                        (acc & !mask) | (s & mask) // SLI
                    } else {
                        s // SHL
                    }
                }
                _ => {
                    // SQSHLU (01100) / SQSHL, UQSHL (01110)
                    let (src_unsigned, out_unsigned) = if opcode == 0b01100 {
                        (false, true)
                    } else {
                        (unsigned, unsigned)
                    };
                    let (o, s) = sat(ival(x, esize, src_unsigned) << lshift, esize, out_unsigned);
                    qc |= s;
                    o
                }
            };
            r |= u128::from(o & ones(esize)) << (i * esize);
        }
        self.set_qc(qc);
        self.set_vec(rd, r, q || scalar);
        Step::Next
    }

    // ---- by element ---------------------------------------------------------------

    fn simd_by_elem(&mut self, instr: u32, scalar: bool) -> Step {
        let q = (instr >> 30) & 1 == 1;
        let u = (instr >> 29) & 1;
        let size = (instr >> 22) & 3;
        let l = (instr >> 21) & 1;
        let m_bit = (instr >> 20) & 1;
        let opcode = (instr >> 12) & 0xf;
        let h = (instr >> 11) & 1;
        let (rd, rn) = (reg_field(instr, 0), reg_field(instr, 5));
        let key = (u << 4) | opcode;
        let fp = matches!(key, 0b0_0001 | 0b0_0101 | 0b0_1001 | 0b1_1001);
        if fp {
            // FMLA/FMLS/FMUL/FMULX (by element), single or double.
            if size < 2 {
                return Step::Illegal; // FP16 / unallocated
            }
            let sz = size & 1;
            let (fmt, esize, lanes) = if scalar {
                if sz == 1 { (D, 64, 1) } else { (S, 32, 1) }
            } else {
                let Some(x) = fp_vec(q, sz) else {
                    return Step::Illegal;
                };
                x
            };
            if sz == 1 && l == 1 {
                return Step::Illegal;
            }
            let index = if sz == 1 { h } else { (h << 1) | l };
            let rm = reg_field(instr, 16);
            let b = elem(self.v[rm], index, esize);
            let (n, d) = (self.v[rn], self.v[rd]);
            let sign = 1u64 << (esize - 1);
            let mut env = self.fpenv();
            let e = &mut env;
            let mut r = 0u128;
            for i in 0..lanes {
                let a = elem(n, i, esize);
                let o = match key {
                    0b0_0001 => fpu::mul_add(elem(d, i, esize), a, b, fmt, e),
                    0b0_0101 => fpu::mul_add(elem(d, i, esize), a ^ sign, b, fmt, e),
                    0b0_1001 => fpu::mul(a, b, false, fmt, e),
                    _ => fpu::mul(a, b, true, fmt, e),
                };
                r |= u128::from(o) << (i * esize);
            }
            self.set_fpenv(env);
            self.set_vec(rd, r, q || scalar);
            return Step::Next;
        }
        // Integer forms: H (index H:L:M, Rm in V0-V15) or S (index H:L).
        let (esize, index, rm) = match size {
            1 => (
                16u32,
                (h << 2) | (l << 1) | m_bit,
                ((instr >> 16) & 0xf) as usize,
            ),
            2 => (32, (h << 1) | l, reg_field(instr, 16)),
            _ => return Step::Illegal,
        };
        let long = matches!(
            key,
            0b0_0010
                | 0b0_0011
                | 0b0_0110
                | 0b0_0111
                | 0b0_1010
                | 0b0_1011
                | 0b1_0010
                | 0b1_0110
                | 0b1_1010
        );
        let ok = match key {
            0b0_0011 | 0b0_0111 | 0b0_1011 | 0b0_1100 | 0b0_1101 => true, // SQD*
            0b0_1000 | 0b1_0000 | 0b1_0100 => !scalar,                    // MUL/MLA/MLS
            0b0_0010 | 0b0_0110 | 0b0_1010 | 0b1_0010 | 0b1_0110 | 0b1_1010 => !scalar,
            _ => false,
        };
        if !ok {
            return Step::Illegal;
        }
        let unsigned = u == 1;
        let b_raw = elem(self.v[rm], index, esize);
        let (n, d) = (self.v[rn], self.v[rd]);
        let mut r = 0u128;
        let mut qc = false;
        if long {
            let wide = esize * 2;
            let lanes = if scalar { 1 } else { 64 / esize };
            let src = if q && !scalar { n >> 64 } else { n };
            let y = ival(b_raw, esize, unsigned);
            for i in 0..lanes {
                let x = ival(elem(src, i, esize), esize, unsigned);
                let acc = elem(d, i, wide);
                let o = match key {
                    0b0_0010 | 0b1_0010 => acc.wrapping_add((x * y) as u64),
                    0b0_0110 | 0b1_0110 => acc.wrapping_sub((x * y) as u64),
                    0b0_1010 | 0b1_1010 => (x * y) as u64,
                    _ => {
                        let (p, s1) = sat(2 * x * y, wide, false);
                        qc |= s1;
                        if key == 0b0_1011 {
                            p
                        } else {
                            let pv = ival(p, wide, false);
                            let av = ival(acc, wide, false);
                            let v = if key == 0b0_0011 { av + pv } else { av - pv };
                            let (o, s2) = sat(v, wide, false);
                            qc |= s2;
                            o
                        }
                    }
                };
                r |= u128::from(o & ones(wide)) << (i * wide);
            }
            self.set_qc(qc);
            self.v[rd] = r;
            return Step::Next;
        }
        let lanes = if scalar {
            1
        } else {
            (if q { 128 } else { 64 }) / esize
        };
        for i in 0..lanes {
            let a = elem(n, i, esize);
            let o = match key {
                0b0_1000 => a.wrapping_mul(b_raw),
                0b1_0000 => elem(d, i, esize).wrapping_add(a.wrapping_mul(b_raw)),
                0b1_0100 => elem(d, i, esize).wrapping_sub(a.wrapping_mul(b_raw)),
                _ => {
                    // SQDMULH / SQRDMULH
                    let p = 2 * ival(a, esize, false) * ival(b_raw, esize, false);
                    let round = if key == 0b0_1101 {
                        1i128 << (esize - 1)
                    } else {
                        0
                    };
                    let (o, s) = sat((p + round) >> esize, esize, false);
                    qc |= s;
                    o
                }
            };
            r |= u128::from(o & ones(esize)) << (i * esize);
        }
        self.set_qc(qc);
        self.set_vec(rd, r, q || scalar);
        Step::Next
    }

    // ---- permute / extract / table -----------------------------------------------

    fn simd_permute(&mut self, instr: u32) -> Step {
        let q = (instr >> 30) & 1 == 1;
        let size = (instr >> 22) & 3;
        let opcode = (instr >> 12) & 7;
        if (size == 3 && !q) || opcode & 3 == 0 {
            return Step::Illegal;
        }
        let (rd, rn, rm) = (
            reg_field(instr, 0),
            reg_field(instr, 5),
            reg_field(instr, 16),
        );
        let (n, m) = (self.v[rn], self.v[rm]);
        let esize = 8u32 << size;
        let lanes = (if q { 128 } else { 64 }) / esize;
        let half = lanes / 2;
        let second = opcode >> 2; // 1 for the "2" variants
        let mut r = 0u128;
        for i in 0..lanes {
            let x = match opcode & 3 {
                // UZP1/UZP2: even/odd elements of n:m.
                0b01 => {
                    let k = 2 * i + second;
                    if k < lanes {
                        elem(n, k, esize)
                    } else {
                        elem(m, k - lanes, esize)
                    }
                }
                // TRN1/TRN2
                0b10 => {
                    let base = i & !1;
                    if i & 1 == 0 {
                        elem(n, base + second, esize)
                    } else {
                        elem(m, base + second, esize)
                    }
                }
                // ZIP1/ZIP2
                _ => {
                    let k = second * half + i / 2;
                    if i & 1 == 0 {
                        elem(n, k, esize)
                    } else {
                        elem(m, k, esize)
                    }
                }
            };
            r |= u128::from(x) << (i * esize);
        }
        self.set_vec(rd, r, q);
        Step::Next
    }

    fn simd_ext(&mut self, instr: u32) -> Step {
        let q = (instr >> 30) & 1 == 1;
        let imm4 = (instr >> 11) & 0xf;
        if (instr >> 22) & 3 != 0 || (!q && imm4 >= 8) {
            return Step::Illegal;
        }
        let (rd, rn, rm) = (
            reg_field(instr, 0),
            reg_field(instr, 5),
            reg_field(instr, 16),
        );
        let (n, m) = (self.v[rn], self.v[rm]);
        let sh = imm4 * 8;
        let r = if q {
            if sh == 0 {
                n
            } else {
                (n >> sh) | (m << (128 - sh))
            }
        } else {
            let lo = u128::from(n as u64) | (u128::from(m as u64) << 64);
            (lo >> sh) & u128::from(u64::MAX)
        };
        self.set_vec(rd, r, q);
        Step::Next
    }

    fn simd_tbl(&mut self, instr: u32) -> Step {
        let q = (instr >> 30) & 1 == 1;
        if (instr >> 22) & 3 != 0 {
            return Step::Illegal;
        }
        let (rd, rn, rm) = (
            reg_field(instr, 0),
            reg_field(instr, 5),
            reg_field(instr, 16),
        );
        let len = ((instr >> 13) & 3) as usize + 1;
        let tbx = (instr >> 12) & 1 == 1;
        let mut table = [0u8; 64];
        for i in 0..len {
            table[i * 16..(i + 1) * 16].copy_from_slice(&self.v[(rn + i) % 32].to_le_bytes());
        }
        let idx = self.v[rm].to_le_bytes();
        let mut out = self.v[rd].to_le_bytes();
        let bytes = if q { 16 } else { 8 };
        for i in 0..bytes {
            let k = idx[i] as usize;
            if k < len * 16 {
                out[i] = table[k];
            } else if !tbx {
                out[i] = 0;
            }
        }
        self.set_vec(rd, u128::from_le_bytes(out), q);
        Step::Next
    }

    // ---- cryptographic extension ---------------------------------------------------

    /// AES (vector form, `0x4E28_0800` class) and SHA-1/SHA-256 two-register
    /// (scalar form, `0x5E28_0800` class).
    fn crypto_two_reg(&mut self, instr: u32, scalar: bool) -> Step {
        let (rd, rn) = (reg_field(instr, 0), reg_field(instr, 5));
        let opcode = (instr >> 12) & 0x1f;
        if scalar {
            if instr & 0xFFFE_0C00 != 0x5E28_0800 {
                return Step::Illegal;
            }
            self.v[rd] = match opcode {
                0b00000 => u128::from((self.v[rn] as u32).rotate_left(30)), // SHA1H
                0b00001 => crypto::sha1_su1(self.v[rd], self.v[rn]),
                0b00010 => crypto::sha256_su0(self.v[rd], self.v[rn]),
                _ => return Step::Illegal,
            };
        } else {
            if instr & 0xFFFE_0C00 != 0x4E28_0800 {
                return Step::Illegal;
            }
            self.v[rd] = match opcode {
                0b00100 => crypto::aes_round(self.v[rd], self.v[rn], true),
                0b00101 => crypto::aes_round(self.v[rd], self.v[rn], false),
                0b00110 => crypto::aes_mix_columns(self.v[rn], true),
                0b00111 => crypto::aes_mix_columns(self.v[rn], false),
                _ => return Step::Illegal,
            };
        }
        Step::Next
    }

    /// SHA-1/SHA-256 three-register (`0x5E00_0000` class).
    fn crypto_sha3(&mut self, instr: u32) -> Step {
        let (rd, rn, rm) = (
            reg_field(instr, 0),
            reg_field(instr, 5),
            reg_field(instr, 16),
        );
        let (d, n, m) = (self.v[rd], self.v[rn], self.v[rm]);
        let e = n as u32;
        self.v[rd] = match (instr >> 12) & 7 {
            0b000 => crypto::sha1_quad_round(d, e, m, crypto::Sha1Op::Choose),
            0b001 => crypto::sha1_quad_round(d, e, m, crypto::Sha1Op::Parity),
            0b010 => crypto::sha1_quad_round(d, e, m, crypto::Sha1Op::Majority),
            0b011 => crypto::sha1_su0(d, n, m),
            0b100 => crypto::sha256_hash(d, n, m, false),
            0b101 => crypto::sha256_hash(n, d, m, true),
            0b110 => crypto::sha256_su1(d, n, m),
            _ => return Step::Illegal,
        };
        Step::Next
    }
}

/// `AdvSIMDExpandImm`: the 64-bit pattern for a modified-immediate `op`/
/// `cmode`/`imm8`.
fn adv_simd_expand_imm(op: u32, cmode: u32, imm8: u32) -> u64 {
    let imm8 = u64::from(imm8);
    let rep32 = |x: u64| x | (x << 32);
    let rep16 = |x: u64| x | (x << 16) | (x << 32) | (x << 48);
    match cmode >> 1 {
        0b000 => rep32(imm8),
        0b001 => rep32(imm8 << 8),
        0b010 => rep32(imm8 << 16),
        0b011 => rep32(imm8 << 24),
        0b100 => rep16(imm8),
        0b101 => rep16(imm8 << 8),
        0b110 => {
            if cmode & 1 == 0 {
                rep32((imm8 << 8) | 0xff)
            } else {
                rep32((imm8 << 16) | 0xffff)
            }
        }
        _ => {
            if cmode & 1 == 0 {
                if op == 0 {
                    imm8 * 0x0101_0101_0101_0101
                } else {
                    (0..8).fold(0u64, |acc, i| {
                        if (imm8 >> i) & 1 == 1 {
                            acc | (0xff << (8 * i))
                        } else {
                            acc
                        }
                    })
                }
            } else if op == 0 {
                rep32(vfp_expand_imm(imm8 as u32, S))
            } else {
                vfp_expand_imm(imm8 as u32, D)
            }
        }
    }
}
