//! Advanced SIMD and crypto instructions from the post-v8.0 extensions this
//! CPU implements and advertises: DotProd (`SDOT`/`UDOT`), RDM
//! (`SQRDMLAH`/`SQRDMLSH`), I8MM (`SMMLA`/`UMMLA`/`USMMLA`/`USDOT`/`SUDOT`),
//! FCMA (`FCMLA`/`FCADD`), SHA-512 (`SHA512H`/`H2`/`SU0`/`SU1`) and SHA-3
//! (`EOR3`/`RAX1`/`XAR`/`BCAX`). Each was diff-tested against native
//! execution (see `tests/aarch64_diff.rs`).

// Decode keys follow the ARM ARM table layout (`U`_`opcode`).
#![allow(clippy::unusual_byte_groupings, clippy::match_same_arms)]

use super::alu::{ones, sign_extend};
use super::fpu::{self, D, Fmt, QC, S};
use super::{Aarch64Interp, Step, reg_field};

#[inline]
fn elem(v: u128, i: u32, esize: u32) -> u64 {
    ((v >> (i * esize)) as u64) & ones(esize)
}

#[inline]
fn ival(x: u64, esize: u32, unsigned: bool) -> i64 {
    if unsigned {
        x as i64
    } else {
        sign_extend(x, esize)
    }
}

/// `SQRDMLAH`/`SQRDMLSH` on one element: `sat((d << esize) ± 2·n·m + round)
/// >> esize`.
fn sqrdmlah(d: u64, n: u64, m: u64, esize: u32, sub: bool) -> (u64, bool) {
    let (d, n, m) = (
        i128::from(sign_extend(d, esize)),
        i128::from(sign_extend(n, esize)),
        i128::from(sign_extend(m, esize)),
    );
    let p = 2 * n * m;
    let acc = (d << esize) + if sub { -p } else { p } + (1i128 << (esize - 1));
    let v = acc >> esize;
    let (lo, hi) = (-(1i128 << (esize - 1)), (1i128 << (esize - 1)) - 1);
    if v < lo {
        (lo as u64 & ones(esize), true)
    } else if v > hi {
        (hi as u64 & ones(esize), true)
    } else {
        (v as u64 & ones(esize), false)
    }
}

/// Four-way byte dot product of 32-bit group `g` of `n` and `m`.
fn dot4(n: u128, gn: u32, n_unsigned: bool, m: u128, gm: u32, m_unsigned: bool) -> i64 {
    (0..4)
        .map(|k| {
            ival(elem(n, gn * 4 + k, 8), 8, n_unsigned)
                * ival(elem(m, gm * 4 + k, 8), 8, m_unsigned)
        })
        .sum()
}

impl Aarch64Interp {
    /// AdvSIMD three-same (extra): bit 21 = 0, bit 15 = 1, bit 10 = 1.
    pub(super) fn simd_three_same_extra(&mut self, instr: u32, scalar: bool) -> Step {
        let q = (instr >> 30) & 1 == 1;
        let u = (instr >> 29) & 1;
        let size = (instr >> 22) & 3;
        let opcode = (instr >> 11) & 0xf;
        let (rd, rn, rm) = (
            reg_field(instr, 0),
            reg_field(instr, 5),
            reg_field(instr, 16),
        );
        let (n, m, d) = (self.v[rn], self.v[rm], self.v[rd]);
        let width = if q { 128 } else { 64 };
        match (u, opcode) {
            (1, 0b0000 | 0b0001) => {
                // SQRDMLAH / SQRDMLSH (vector and scalar)
                if size == 0 || size == 3 {
                    return Step::Illegal;
                }
                let esize = 8u32 << size;
                let lanes = if scalar { 1 } else { width / esize };
                let mut r = 0u128;
                let mut qc = false;
                for i in 0..lanes {
                    let (x, s) = sqrdmlah(
                        elem(d, i, esize),
                        elem(n, i, esize),
                        elem(m, i, esize),
                        esize,
                        opcode == 1,
                    );
                    qc |= s;
                    r |= u128::from(x) << (i * esize);
                }
                if qc {
                    self.fpsr |= u64::from(QC);
                }
                self.v[rd] = if q || scalar {
                    r
                } else {
                    r & u128::from(u64::MAX)
                };
                Step::Next
            }
            _ if scalar => Step::Illegal,
            (_, 0b0010) | (0, 0b0011) => {
                // SDOT/UDOT (DotProd), USDOT (I8MM)
                if size != 2 {
                    return Step::Illegal;
                }
                let (nu, mu) = match (u, opcode) {
                    (0, 0b0010) => (false, false),
                    (1, _) => (true, true),
                    _ => (true, false),
                };
                let mut r = 0u128;
                for i in 0..width / 32 {
                    let acc = elem(d, i, 32).wrapping_add(dot4(n, i, nu, m, i, mu) as u64);
                    r |= u128::from(acc & 0xffff_ffff) << (32 * i);
                }
                self.v[rd] = r;
                Step::Next
            }
            (_, 0b0100) | (0, 0b0101) => {
                // SMMLA/UMMLA/USMMLA (I8MM): 2x8 · (2x8)^T into a 2x2 of i32.
                if size != 2 || !q {
                    return Step::Illegal;
                }
                let (nu, mu) = match (u, opcode) {
                    (0, 0b0100) => (false, false),
                    (1, _) => (true, true),
                    _ => (true, false),
                };
                let mut r = 0u128;
                for i in 0..2 {
                    for j in 0..2 {
                        let s: i64 = (0..8)
                            .map(|k| {
                                ival(elem(n, i * 8 + k, 8), 8, nu)
                                    * ival(elem(m, j * 8 + k, 8), 8, mu)
                            })
                            .sum();
                        let idx = i * 2 + j;
                        let acc = elem(d, idx, 32).wrapping_add(s as u64);
                        r |= u128::from(acc & 0xffff_ffff) << (32 * idx);
                    }
                }
                self.v[rd] = r;
                Step::Next
            }
            (1, 0b1000..=0b1011) => {
                // FCMLA (vector)
                let Some((fmt, esize)) = fcma_fmt(size, q) else {
                    return Step::Illegal;
                };
                let lanes = width / esize;
                let rot = opcode & 3;
                let mut env = self.fpenv();
                let mut r = 0u128;
                for e in 0..lanes / 2 {
                    let (re, im) = fcmla_pair(
                        (elem(d, 2 * e, esize), elem(d, 2 * e + 1, esize)),
                        (elem(n, 2 * e, esize), elem(n, 2 * e + 1, esize)),
                        (elem(m, 2 * e, esize), elem(m, 2 * e + 1, esize)),
                        rot,
                        fmt,
                        &mut env,
                    );
                    r |= (u128::from(re) << (2 * e * esize))
                        | (u128::from(im) << ((2 * e + 1) * esize));
                }
                self.set_fpenv(env);
                self.v[rd] = r;
                Step::Next
            }
            (1, 0b1111) if size == 1 => {
                // BFDOT (vector)
                let mut r = 0u128;
                for e in 0..width / 32 {
                    let x = fpu::bf_dot_add(
                        elem(d, e, 32),
                        elem(n, 2 * e, 16),
                        elem(n, 2 * e + 1, 16),
                        elem(m, 2 * e, 16),
                        elem(m, 2 * e + 1, 16),
                    );
                    r |= u128::from(x) << (32 * e);
                }
                self.v[rd] = r;
                Step::Next
            }
            (1, 0b1111) if size == 3 => {
                // BFMLALB (Q=0) / BFMLALT (Q=1): even/odd BF16 lanes widened
                // and fused-multiply-added in single precision.
                let sel = u32::from(q);
                let mut env = self.fpenv();
                let mut r = 0u128;
                for e in 0..4 {
                    let a = elem(n, 2 * e + sel, 16) << 16;
                    let b = elem(m, 2 * e + sel, 16) << 16;
                    let x = fpu::mul_add(elem(d, e, 32), a, b, S, &mut env);
                    r |= u128::from(x) << (32 * e);
                }
                self.set_fpenv(env);
                self.v[rd] = r;
                Step::Next
            }
            (1, 0b1101) if size == 1 && q => {
                // BFMMLA: 2x4 · (2x4)^T BF16 into a 2x2 of single.
                let mut r = 0u128;
                for i in 0..2 {
                    for j in 0..2 {
                        let mut sum = elem(d, 2 * i + j, 32);
                        for k in 0..2 {
                            sum = fpu::bf_dot_add(
                                sum,
                                elem(n, 4 * i + 2 * k, 16),
                                elem(n, 4 * i + 2 * k + 1, 16),
                                elem(m, 4 * j + 2 * k, 16),
                                elem(m, 4 * j + 2 * k + 1, 16),
                            );
                        }
                        r |= u128::from(sum) << (32 * (2 * i + j));
                    }
                }
                self.v[rd] = r;
                Step::Next
            }
            (1, 0b1100 | 0b1110) => {
                // FCADD
                let Some((fmt, esize)) = fcma_fmt(size, q) else {
                    return Step::Illegal;
                };
                let lanes = width / esize;
                let sign = 1u64 << (esize - 1);
                let rot270 = opcode == 0b1110;
                let mut env = self.fpenv();
                let mut r = 0u128;
                for e in 0..lanes / 2 {
                    let (n_re, n_im) = (elem(n, 2 * e, esize), elem(n, 2 * e + 1, esize));
                    let (m_re, m_im) = (elem(m, 2 * e, esize), elem(m, 2 * e + 1, esize));
                    let (e2, e4) = if rot270 {
                        (m_im, m_re ^ sign)
                    } else {
                        (m_im ^ sign, m_re)
                    };
                    let re = fpu::add(n_re, e2, false, fmt, &mut env);
                    let im = fpu::add(n_im, e4, false, fmt, &mut env);
                    r |= (u128::from(re) << (2 * e * esize))
                        | (u128::from(im) << ((2 * e + 1) * esize));
                }
                self.set_fpenv(env);
                self.v[rd] = r;
                Step::Next
            }
            _ => Step::Illegal,
        }
    }

    /// `FMLAL`/`FMLSL`/`FMLAL2`/`FMLSL2` (vector, FEAT_FHM): half-precision
    /// products of the low (or, for the `2` forms, high) halves accumulated
    /// into single precision with one rounding.
    pub(super) fn simd_fhm(&mut self, instr: u32, sub: bool, upper: bool) -> Step {
        let q = (instr >> 30) & 1 == 1;
        let (rd, rn, rm) = (
            reg_field(instr, 0),
            reg_field(instr, 5),
            reg_field(instr, 16),
        );
        let width = if q { 128 } else { 64 };
        let part = if upper { width / 2 } else { 0 };
        let (n, m, d) = (self.v[rn] >> part, self.v[rm] >> part, self.v[rd]);
        let mut env = self.fpenv();
        let mut r = 0u128;
        for e in 0..width / 32 {
            let mut a = elem(n, e, 16);
            if sub {
                a ^= 0x8000;
            }
            let x = fpu::mul_add_mixed(elem(d, e, 32), a, elem(m, e, 16), S, fpu::H, &mut env);
            r |= u128::from(x) << (32 * e);
        }
        self.set_fpenv(env);
        self.v[rd] = if q { r } else { r & u128::from(u64::MAX) };
        Step::Next
    }

    /// The by-element forms of the extensions; `None` if `instr` isn't one.
    pub(super) fn simd_by_elem_ext(&mut self, instr: u32, scalar: bool) -> Option<Step> {
        let q = (instr >> 30) & 1 == 1;
        let u = (instr >> 29) & 1;
        let size = (instr >> 22) & 3;
        let l = (instr >> 21) & 1;
        let m_bit = (instr >> 20) & 1;
        let opcode = (instr >> 12) & 0xf;
        let h = (instr >> 11) & 1;
        let (rd, rn) = (reg_field(instr, 0), reg_field(instr, 5));
        let width = if q { 128 } else { 64 };
        let key = (u << 4) | opcode;
        match key {
            0b0_1111 if size == 1 || size == 3 => {
                if scalar {
                    return Some(Step::Illegal);
                }
                let (n, d) = (self.v[rn], self.v[rd]);
                let mut r = 0u128;
                if size == 1 {
                    // BFDOT (by element): a BF16 pair of Vm (index H:L).
                    let rm = reg_field(instr, 16);
                    let mm = self.v[rm];
                    let index = (h << 1) | l;
                    for e in 0..width / 32 {
                        let x = fpu::bf_dot_add(
                            elem(d, e, 32),
                            elem(n, 2 * e, 16),
                            elem(n, 2 * e + 1, 16),
                            elem(mm, 2 * index, 16),
                            elem(mm, 2 * index + 1, 16),
                        );
                        r |= u128::from(x) << (32 * e);
                    }
                    self.v[rd] = r;
                } else {
                    // BFMLALB/BFMLALT (by element): index H:L:M, Rm in V0-V15.
                    let rm = ((instr >> 16) & 0xf) as usize;
                    let index = (h << 2) | (l << 1) | m_bit;
                    let b = elem(self.v[rm], index, 16) << 16;
                    let sel = u32::from(q);
                    let mut env = self.fpenv();
                    for e in 0..4 {
                        let a = elem(n, 2 * e + sel, 16) << 16;
                        let x = fpu::mul_add(elem(d, e, 32), a, b, S, &mut env);
                        r |= u128::from(x) << (32 * e);
                    }
                    self.set_fpenv(env);
                    self.v[rd] = r;
                }
                Some(Step::Next)
            }
            0b0_0000 | 0b0_0100 | 0b1_1000 | 0b1_1100 => {
                // FMLAL/FMLSL/FMLAL2/FMLSL2 (by element), FEAT_FHM.
                if scalar || size != 2 {
                    return Some(Step::Illegal);
                }
                let rm = ((instr >> 16) & 0xf) as usize;
                let index = (h << 2) | (l << 1) | m_bit;
                let b = elem(self.v[rm], index, 16);
                let sub = opcode & 0b0100 != 0;
                let part = if u == 1 { width / 2 } else { 0 };
                let (n, d) = (self.v[rn] >> part, self.v[rd]);
                let mut env = self.fpenv();
                let mut r = 0u128;
                for e in 0..width / 32 {
                    let mut a = elem(n, e, 16);
                    if sub {
                        a ^= 0x8000;
                    }
                    let x = fpu::mul_add_mixed(elem(d, e, 32), a, b, S, fpu::H, &mut env);
                    r |= u128::from(x) << (32 * e);
                }
                self.set_fpenv(env);
                self.v[rd] = if q { r } else { r & u128::from(u64::MAX) };
                Some(Step::Next)
            }
            0b0_1110 | 0b1_1110 | 0b0_1111 => {
                // SDOT/UDOT (DotProd); SUDOT (size 00) / USDOT (size 10) (I8MM)
                if scalar {
                    return Some(Step::Illegal);
                }
                let (nu, mu) = match (key, size) {
                    (0b0_1110, 2) => (false, false),
                    (0b1_1110, 2) => (true, true),
                    (0b0_1111, 0) => (false, true),
                    (0b0_1111, 2) => (true, false),
                    _ => return Some(Step::Illegal),
                };
                let rm = reg_field(instr, 16);
                let index = (h << 1) | l;
                let (n, m, d) = (self.v[rn], self.v[rm], self.v[rd]);
                let mut r = 0u128;
                for i in 0..width / 32 {
                    let acc = elem(d, i, 32).wrapping_add(dot4(n, i, nu, m, index, mu) as u64);
                    r |= u128::from(acc & 0xffff_ffff) << (32 * i);
                }
                self.v[rd] = r;
                Some(Step::Next)
            }
            0b1_1101 | 0b1_1111 => {
                // SQRDMLAH / SQRDMLSH (by element)
                let (esize, index, rm) = match size {
                    1 => (
                        16u32,
                        (h << 2) | (l << 1) | m_bit,
                        ((instr >> 16) & 0xf) as usize,
                    ),
                    2 => (32, (h << 1) | l, reg_field(instr, 16)),
                    _ => return Some(Step::Illegal),
                };
                let b = elem(self.v[rm], index, esize);
                let (n, d) = (self.v[rn], self.v[rd]);
                let lanes = if scalar { 1 } else { width / esize };
                let mut r = 0u128;
                let mut qc = false;
                for i in 0..lanes {
                    let (x, s) = sqrdmlah(
                        elem(d, i, esize),
                        elem(n, i, esize),
                        b,
                        esize,
                        key == 0b1_1111,
                    );
                    qc |= s;
                    r |= u128::from(x) << (i * esize);
                }
                if qc {
                    self.fpsr |= u64::from(QC);
                }
                self.v[rd] = if q || scalar {
                    r
                } else {
                    r & u128::from(u64::MAX)
                };
                Some(Step::Next)
            }
            0b1_0001 | 0b1_0011 | 0b1_0101 | 0b1_0111 => {
                // FCMLA (by element): H (index H:L) or S (index H, Q=1).
                if scalar {
                    return Some(Step::Illegal);
                }
                let (fmt, esize, index) = match size {
                    2 if l == 0 && q => (S, 32u32, h),
                    _ => return Some(Step::Illegal),
                };
                let rm = reg_field(instr, 16);
                let rot = (opcode >> 1) & 3;
                let (n, d) = (self.v[rn], self.v[rd]);
                let mm = self.v[rm];
                let mut env = self.fpenv();
                let mut r = 0u128;
                for e in 0..width / esize / 2 {
                    let mp = (elem(mm, 2 * index, esize), elem(mm, 2 * index + 1, esize));
                    let (re, im) = fcmla_pair(
                        (elem(d, 2 * e, esize), elem(d, 2 * e + 1, esize)),
                        (elem(n, 2 * e, esize), elem(n, 2 * e + 1, esize)),
                        mp,
                        rot,
                        fmt,
                        &mut env,
                    );
                    r |= (u128::from(re) << (2 * e * esize))
                        | (u128::from(im) << ((2 * e + 1) * esize));
                }
                self.set_fpenv(env);
                self.v[rd] = if q { r } else { r & u128::from(u64::MAX) };
                Some(Step::Next)
            }
            _ => None,
        }
    }

    /// The `0xCE` crypto group: SHA-512 and SHA-3 (SM3/SM4 are not
    /// implemented).
    pub(super) fn crypto_ext(&mut self, instr: u32) -> Step {
        if instr >> 24 != 0xCE {
            return Step::Illegal;
        }
        let (rd, rn, rm) = (
            reg_field(instr, 0),
            reg_field(instr, 5),
            reg_field(instr, 16),
        );
        let (d, n, m) = (self.v[rd], self.v[rn], self.v[rm]);
        let lo = |v: u128| v as u64;
        let hi = |v: u128| (v >> 64) as u64;
        let pack = |l: u64, h: u64| u128::from(l) | (u128::from(h) << 64);
        let r = match (instr >> 21) & 7 {
            // EOR3 / BCAX (four-register, bit 15 = 0)
            0b000 | 0b001 if (instr >> 15) & 1 == 0 => {
                let a = self.v[reg_field(instr, 10)];
                if (instr >> 21) & 1 == 0 {
                    n ^ m ^ a
                } else {
                    n ^ (m & !a)
                }
            }
            // XAR
            0b100 => {
                let imm6 = (instr >> 10) & 0x3f;
                let x = n ^ m;
                pack(lo(x).rotate_right(imm6), hi(x).rotate_right(imm6))
            }
            // SHA512H/SHA512H2/SHA512SU1/RAX1 (bits 15:14 = 10, O = 0)
            0b011 if (instr >> 12) & 0xf == 0b1000 => match (instr >> 10) & 3 {
                0b00 => sha512h(d, n, m),
                0b01 => sha512h2(d, n, m),
                0b10 => {
                    let su1 = |x: u64| x.rotate_right(19) ^ x.rotate_right(61) ^ (x >> 6);
                    pack(
                        lo(d).wrapping_add(su1(lo(n))).wrapping_add(lo(m)),
                        hi(d).wrapping_add(su1(hi(n))).wrapping_add(hi(m)),
                    )
                }
                _ => pack(lo(n) ^ lo(m).rotate_left(1), hi(n) ^ hi(m).rotate_left(1)),
            },
            // SHA512SU0 (two-register)
            0b110 if instr & 0xFFFF_FC00 == 0xCEC0_8000 => {
                let su0 = |x: u64| x.rotate_right(1) ^ x.rotate_right(8) ^ (x >> 7);
                pack(
                    lo(d).wrapping_add(su0(hi(d))),
                    hi(d).wrapping_add(su0(lo(n))),
                )
            }
            _ => return Step::Illegal,
        };
        self.v[rd] = r;
        Step::Next
    }
}

/// The FP format of `FCMLA`/`FCADD`: half (FP16), single, or double (`Q`).
fn fcma_fmt(size: u32, q: bool) -> Option<(Fmt, u32)> {
    match size {
        2 => Some((S, 32)),
        3 if q => Some((D, 64)),
        _ => None,
    }
}

/// One complex element of `FCMLA` with rotation `rot` (0/90/180/270 as
/// 0..=3): returns the new `(real, imaginary)`.
fn fcmla_pair(
    d: (u64, u64),
    n: (u64, u64),
    m: (u64, u64),
    rot: u32,
    fmt: Fmt,
    env: &mut fpu::Env,
) -> (u64, u64) {
    let neg = |x: u64| x ^ (1u64 << (fmt.n() - 1));
    let (e1, e2, e3, e4) = match rot {
        0 => (m.0, n.0, m.1, n.0),
        1 => (neg(m.1), n.1, m.0, n.1),
        2 => (neg(m.0), n.0, neg(m.1), n.0),
        _ => (m.1, n.1, neg(m.0), n.1),
    };
    (
        fpu::mul_add(d.0, e2, e1, fmt, env),
        fpu::mul_add(d.1, e4, e3, fmt, env),
    )
}

fn sha512h(w: u128, x: u128, y: u128) -> u128 {
    let (xl, xh) = (x as u64, (x >> 64) as u64);
    let (yl, yh) = (y as u64, (y >> 64) as u64);
    let (wl, wh) = (w as u64, (w >> 64) as u64);
    let sigma1 = |v: u64| v.rotate_right(14) ^ v.rotate_right(18) ^ v.rotate_right(41);
    let hi = ((yh & xl) ^ (!yh & xh))
        .wrapping_add(sigma1(yh))
        .wrapping_add(wh);
    let tmp = hi.wrapping_add(yl);
    let lo = ((tmp & yh) ^ (!tmp & xl))
        .wrapping_add(sigma1(tmp))
        .wrapping_add(wl);
    u128::from(lo) | (u128::from(hi) << 64)
}

fn sha512h2(w: u128, x: u128, y: u128) -> u128 {
    let xl = x as u64;
    let (yl, yh) = (y as u64, (y >> 64) as u64);
    let (wl, wh) = (w as u64, (w >> 64) as u64);
    let sigma0 = |v: u64| v.rotate_right(28) ^ v.rotate_right(34) ^ v.rotate_right(39);
    let hi = ((xl & yh) ^ (xl & yl) ^ (yh & yl))
        .wrapping_add(sigma0(yl))
        .wrapping_add(wh);
    let lo = ((hi & yl) ^ (hi & yh) ^ (yh & yl))
        .wrapping_add(sigma0(hi))
        .wrapping_add(wl);
    u128::from(lo) | (u128::from(hi) << 64)
}
