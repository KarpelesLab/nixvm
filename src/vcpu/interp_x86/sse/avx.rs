//! The VEX-encoded instructions of x86-64-v3: AVX and AVX2 (the SSE
//! operations widened to 256 bits with a non-destructive source, plus
//! broadcasts, permutes, lane inserts/extracts, masked moves, variable
//! shifts and gathers), FMA, F16C, and the general-purpose BMI1/BMI2
//! instructions (`ANDN`, `BLS*`, `BZHI`, `PDEP`/`PEXT`, `MULX`, `BEXTR`,
//! `SARX`/`SHLX`/`SHRX`, `RORX`).
//!
//! A VEX instruction names its destination in ModRM.reg, its first source
//! in `VEX.vvvv` and its second in ModRM.rm; `VEX.L` selects 128 or 256 bits.
//! Writing an XMM destination zeroes bits 255:128 of the YMM register (the
//! legacy SSE forms leave them). 256-bit operations work on two 128-bit
//! halves with the same per-lane semantics as the SSE forms (reusing their
//! lane helpers); the few that cross lanes are spelled out. VEX memory
//! operands need no alignment, except for the explicitly aligned moves.

#![allow(clippy::too_many_lines)]

use super::sse4::{
    blend_imm, blendv, dpp, fp_horiz, mpsadbw, palignr, phminposuw, pmov_ext, round_lanes,
    sse41_op, ssse3_op,
};
use super::{
    Ff, Mp, cvt_lane, fp_arith, fp_cmp, fp_shape, from_int_lane, int_op, lane, movmskb,
    rcp_estimate, set_lane, shift, sx, to_int_lane, unpack,
};
use crate::vcpu::GuestMemory;
use crate::vcpu::interp_x86::{
    AVX, AVX2, BMI1, BMI2, F16C, FMA, Flags, MXCSR_MASK, ModRm, Pfx, RDX, Rex, RmKind, Step, Trap,
    X86Interp, fetch, rd_fault,
};
use crate::vcpu::softfloat::{self as sf, FMT16, FMT32, INVALID, Mx, Round};

/// A YMM value: the low and high 128-bit halves.
type Y = [u128; 2];

/// The decoded VEX prefix.
#[derive(Clone, Copy, Debug)]
struct Vex {
    /// Opcode map: 1 = `0F`, 2 = `0F 38`, 3 = `0F 3A`.
    map: u8,
    /// The implied mandatory prefix.
    pp: Mp,
    /// `VEX.L`: 256-bit.
    l: bool,
    /// `VEX.W`.
    w: bool,
    /// The register `VEX.vvvv` names (stored inverted; `0` when unused).
    v: usize,
}

impl Vex {
    /// The operand size in bytes.
    const fn bytes(self) -> usize {
        if self.l { 32 } else { 16 }
    }
}

/// The register-file and memory helpers of the VEX forms.
impl X86Interp {
    fn yget(&self, r: usize) -> Y {
        [self.xmm[r], self.ymm_hi[r]]
    }

    /// Write a VEX destination: 256 bits, or 128 with bits 255:128 zeroed.
    fn yset(&mut self, r: usize, v: Y, l: bool) {
        self.xmm[r] = v[0];
        self.ymm_hi[r] = if l { v[1] } else { 0 };
    }

    fn yread(mem: &GuestMemory, a: u64, n: usize, align: u64) -> Result<Y, Step> {
        if align > 1 && a & (align - 1) != 0 {
            return Err(Step::Trap(Trap::Protection));
        }
        let mut b = [0u8; 32];
        mem.read(a, &mut b[..n]).map_err(|_| rd_fault(a))?;
        Ok([
            u128::from_le_bytes(b[..16].try_into().unwrap()),
            u128::from_le_bytes(b[16..].try_into().unwrap()),
        ])
    }

    fn ywrite(
        &mut self,
        mem: &mut GuestMemory,
        a: u64,
        v: Y,
        n: usize,
        align: u64,
    ) -> Result<(), Step> {
        if align > 1 && a & (align - 1) != 0 {
            return Err(Step::Trap(Trap::Protection));
        }
        let mut b = [0u8; 32];
        b[..16].copy_from_slice(&v[0].to_le_bytes());
        b[16..].copy_from_slice(&v[1].to_le_bytes());
        self.store(mem, a, &b[..n])
    }

    /// The linear address of a memory r/m (`None` for a register).
    fn yaddr(&self, m: &ModRm, end: u64) -> Option<u64> {
        self.ea_of(m.kind, end).map(|a| self.lin(a))
    }

    /// An r/m source: a whole YMM register, or `n` bytes of memory
    /// (zero-extended), unaligned.
    fn ysrc(&self, mem: &GuestMemory, m: &ModRm, end: u64, n: usize) -> Result<Y, Step> {
        match m.kind {
            RmKind::Reg(r) => Ok(self.yget(r)),
            _ => Self::yread(mem, self.yaddr(m, end).unwrap_or(0), n, 1),
        }
    }

    /// `dst = f(vvvv, r/m)` per 128-bit half (an integer/bitwise op).
    fn v_nds(
        &mut self,
        mem: &GuestMemory,
        m: &ModRm,
        end: u64,
        vx: Vex,
        f: impl Fn(u128, u128) -> u128,
    ) -> Step {
        let b = fetch!(self.ysrc(mem, m, end, vx.bytes()));
        let a = self.yget(vx.v);
        let hi = if vx.l { f(a[1], b[1]) } else { 0 };
        self.yset(m.reg, [f(a[0], b[0]), hi], vx.l);
        self.next(end)
    }

    /// `dst = f(r/m)` per 128-bit half (a two-operand op: `vvvv` must be
    /// unused).
    fn v_un(
        &mut self,
        mem: &GuestMemory,
        m: &ModRm,
        end: u64,
        vx: Vex,
        f: impl Fn(u128, usize) -> u128,
    ) -> Step {
        if vx.v != 0 {
            return Step::Illegal;
        }
        let b = fetch!(self.ysrc(mem, m, end, vx.bytes()));
        let hi = if vx.l { f(b[1], 1) } else { 0 };
        self.yset(m.reg, [f(b[0], 0), hi], vx.l);
        self.next(end)
    }

    /// A floating-point `dst = f(vvvv, r/m)` with exception flags: per half
    /// for the packed forms; for a scalar form (`scalar` = its memory
    /// operand size) on the low lane, with the rest of `vvvv`'s low half
    /// passed through and bits 255:128 zeroed whatever `VEX.L`.
    fn v_fp(
        &mut self,
        mem: &GuestMemory,
        m: &ModRm,
        end: u64,
        vx: Vex,
        scalar: Option<usize>,
        f: impl Fn(u128, u128, Mx, usize) -> (u128, u32),
    ) -> Step {
        let mx = self.mx();
        let a = self.yget(vx.v);
        let (r, flags, l) = if let Some(n) = scalar {
            let b = fetch!(self.ysrc(mem, m, end, n));
            let (r, fl) = f(a[0], b[0], mx, 0);
            ([r, 0], fl, false)
        } else {
            let b = fetch!(self.ysrc(mem, m, end, vx.bytes()));
            let (lo, f0) = f(a[0], b[0], mx, 0);
            let (hi, f1) = if vx.l { f(a[1], b[1], mx, 1) } else { (0, 0) };
            ([lo, hi], f0 | f1, vx.l)
        };
        fetch!(self.sse_flags(flags));
        self.yset(m.reg, r, l);
        self.next(end)
    }

    /// Like [`X86Interp::v_fp`] for a two-operand packed op (`vvvv` unused).
    fn v_fp_un(
        &mut self,
        mem: &GuestMemory,
        m: &ModRm,
        end: u64,
        vx: Vex,
        f: impl Fn(u128, Mx, usize) -> (u128, u32),
    ) -> Step {
        if vx.v != 0 {
            return Step::Illegal;
        }
        self.v_fp(mem, m, end, vx, None, |_, b, mx, h| f(b, mx, h))
    }

    /// A store or register move whose destination is the r/m operand.
    #[allow(clippy::too_many_arguments)]
    fn v_store(
        &mut self,
        mem: &mut GuestMemory,
        m: &ModRm,
        end: u64,
        v: Y,
        n: usize,
        align: u64,
        l: bool,
    ) -> Step {
        match m.kind {
            RmKind::Reg(r) => self.yset(r, v, l),
            _ => {
                let a = self.yaddr(m, end).unwrap_or(0);
                fetch!(self.ywrite(mem, a, v, n, align));
            }
        }
        self.next(end)
    }
}

// ---- pure helpers -----------------------------------------------------------------

/// `w`-byte lanes `0..n` of a YMM value, as one sequence across both halves.
fn ylane(v: Y, w: usize, i: usize) -> u64 {
    let per = 16 / w;
    lane(v[i / per], w, i % per)
}

/// Pack `w`-byte lanes into a YMM value (unused lanes zero).
fn ypack(lanes: &[u64], w: usize) -> Y {
    let per = 16 / w;
    let mut out = [0u128; 2];
    for (i, &x) in lanes.iter().enumerate() {
        out[i / per] = set_lane(out[i / per], w, i % per, x);
    }
    out
}

/// The sign bits of the `w`-byte lanes of the low `n` bytes of `v`.
fn sign_mask(v: Y, w: usize, n: usize) -> u64 {
    (0..n / w).fold(0, |acc, i| acc | ((ylane(v, w, i) >> (8 * w - 1)) & 1) << i)
}

/// `PSHUFD`/`PSHUFHW`/`PSHUFLW` (`mp` = 66/F3/F2) of one 128-bit half.
fn pshuf(v: u128, mp: Mp, imm: u8) -> u128 {
    let sel = |w: usize, base: usize, i: usize| -> u128 {
        u128::from(lane(v, w, base + usize::from((imm >> (2 * i)) & 3)))
    };
    match mp {
        Mp::P66 => (0..4).fold(0, |acc, i| acc | (sel(4, 0, i) << (32 * i))),
        Mp::F3 => {
            let words = (0..4).fold(0u128, |acc, i| acc | (sel(2, 4, i) << (16 * i)));
            (v & u128::from(u64::MAX)) | (words << 64)
        }
        _ => {
            let words = (0..4).fold(0u128, |acc, i| acc | (sel(2, 0, i) << (16 * i)));
            (v & !u128::from(u64::MAX)) | words
        }
    }
}

/// `SHUFPS` (same `imm` for both halves) / `SHUFPD` (`imm` bits `2h..2h+1`
/// for half `h`) of one half.
fn shufp(a: u128, b: u128, double: bool, imm: u8, half: usize) -> u128 {
    if double {
        let k = imm >> (2 * half);
        let s = |v: u128, k: u8| (v >> (64 * u32::from(k & 1))) as u64;
        u128::from(s(a, k)) | (u128::from(s(b, k >> 1)) << 64)
    } else {
        let s = |v: u128, k: u8| (v >> (32 * u32::from(k & 3))) as u32;
        u128::from(s(a, imm))
            | (u128::from(s(a, imm >> 2)) << 32)
            | (u128::from(s(b, imm >> 4)) << 64)
            | (u128::from(s(b, imm >> 6)) << 96)
    }
}

/// The immediate shifts of groups 12-14 (`/2` logical right, `/4`
/// arithmetic right, `/6` left, `/3`/`/7` whole-half byte shifts) on one
/// half; `None` for an invalid extension.
fn shift_imm_half(v: u128, op: u8, ext: usize, imm: u8) -> Option<u128> {
    let w = match op {
        0x71 => 2,
        0x72 => 4,
        _ => 8,
    };
    let count = u64::from(imm);
    Some(match (op, ext) {
        (_, 2) => shift(v, w, 16, count, 0),
        (0x71 | 0x72, 4) => shift(v, w, 16, count, 2),
        (_, 6) => shift(v, w, 16, count, 1),
        (0x73, 3) => {
            if imm >= 16 {
                0
            } else {
                v >> (8 * u32::from(imm))
            }
        }
        (0x73, 7) => {
            if imm >= 16 {
                0
            } else {
                v << (8 * u32::from(imm))
            }
        }
        _ => return None,
    })
}

/// `CVTDQ2PS` (none), `CVTPS2DQ` (66), `CVTTPS2DQ` (F3) of one half.
fn cvt_5b(mp: Mp, v: u128, mx: Mx) -> (u128, u32) {
    let mut out = 0u128;
    let mut flags = 0;
    for i in 0..4 {
        let x = lane(v, 4, i);
        let (r, f) = if mp == Mp::None {
            from_int_lane(Ff::S, i64::from(x as u32 as i32), mx)
        } else {
            to_int_lane(Ff::S, x, 32, mx, mp == Mp::F3)
        };
        out |= u128::from(r) << (32 * i);
        flags |= f;
    }
    (out, flags)
}

/// `RCPPS`/`RSQRTPS` (`op` 53/52) of one half; scalar: the low lane only,
/// the rest from `a`.
fn rcp_lanes(op: u8, a: u128, b: u128, scalar: bool) -> u128 {
    let mut r = a;
    for i in 0..if scalar { 1 } else { 4 } {
        let e = rcp_estimate(lane(b, 4, i) as u32, op == 0x52);
        r = set_lane(r, 4, i, u64::from(e));
    }
    r
}

/// `VPERMILPS`/`VPERMILPD` by a control vector (per-half, in-lane).
fn permil_var(a: u128, ctl: u128, double: bool) -> u128 {
    if double {
        (0..2).fold(0, |acc, i| {
            let k = ((lane(ctl, 8, i) >> 1) & 1) as usize;
            set_lane(acc, 8, i, lane(a, 8, k))
        })
    } else {
        (0..4).fold(0, |acc, i| {
            let k = (lane(ctl, 4, i) & 3) as usize;
            set_lane(acc, 4, i, lane(a, 4, k))
        })
    }
}

/// `VPSRLV`/`VPSRAV`/`VPSLLV` (`op` 45/46/47) of one half: each lane shifted
/// by the corresponding lane of `cnt`.
fn shift_var(op: u8, a: u128, cnt: u128, w: usize) -> u128 {
    let bits = 8 * w as u64;
    let mut out = 0u128;
    for i in 0..16 / w {
        let (x, c) = (lane(a, w, i), lane(cnt, w, i));
        let r = match op {
            0x45 => {
                if c >= bits {
                    0
                } else {
                    x >> c
                }
            }
            0x46 => (sx(x, w) >> c.min(bits - 1)) as u64,
            _ => {
                if c >= bits {
                    0
                } else {
                    x << c
                }
            }
        };
        out = set_lane(out, w, i, r);
    }
    out
}

/// `VPERM2F128`/`VPERM2I128`: each result half picks a half of `a` or `b`
/// (`imm[1:0]`, `imm[5:4]`) or zero (`imm[3]`, `imm[7]`).
fn perm2x128(a: Y, b: Y, imm: u8) -> Y {
    let pick = |k: u8| -> u128 {
        if k & 8 != 0 {
            return 0;
        }
        match k & 3 {
            0 => a[0],
            1 => a[1],
            2 => b[0],
            _ => b[1],
        }
    };
    [pick(imm), pick(imm >> 4)]
}

/// One FMA lane: `±(a·b) ± c` rounded once, from the three operands in
/// instruction order (`x1` = destination, `x2` = `vvvv`, `x3` = r/m) and
/// the 132/213/231 operand form (`form` 0/1/2) that maps them to `a`, `b`,
/// `c`. A NaN operand yields the first NaN of `a`, `b`, `c`, quieted (and
/// unnegated); `∞·0` with a NaN addend is that NaN, without `IE`.
#[allow(clippy::fn_params_excessive_bools)]
fn fma_lane(ff: Ff, x: [u64; 3], form: u8, neg_prod: bool, neg_add: bool, mx: Mx) -> (u64, u32) {
    let u = x.map(|v| ff.unpack(v));
    let abc = match form {
        0 => [u[0], u[2], u[1]],
        1 => [u[1], u[0], u[2]],
        _ => [u[1], u[2], u[0]],
    };
    if let Some(n) = abc.iter().find(|v| v.is_nan()) {
        let f = if abc.iter().any(sf::Fp::is_snan) {
            INVALID
        } else {
            0
        };
        return (ff.pack(&n.quieted()), f);
    }
    let mut d = 0;
    let [a, b, c] = abc.map(|v| {
        let (v, de) = mx.input(v, ff.fmt());
        d |= de;
        v
    });
    let a = if neg_prod { a.neg() } else { a };
    let c = if neg_add { c.neg() } else { c };
    let (v, f) = mx.output(sf::fma(a, b, c, ff.fmt(), mx.mode));
    (ff.pack(&v), f | d)
}

/// `VCVTPH2PS` lane: exact (half denormals are normal singles), an SNaN is
/// quieted with `IE`.
fn half_to_single(h: u16) -> (u64, u32) {
    let v = sf::unpack_f16(h);
    if v.is_nan() {
        let f = if v.is_snan() { INVALID } else { 0 };
        return (u64::from(sf::pack_f32(&v.quieted())), f);
    }
    (u64::from(sf::pack_f32(&v)), 0)
}

/// `VCVTPS2PH` lane: round to half precision under `mode` (`DAZ` applies to
/// the input; `FTZ` does not apply to the result).
fn single_to_half(s: u32, mode: Round, mx: Mx) -> (u16, u32) {
    let (v, d) = mx.input(sf::unpack_f32(s), FMT32);
    if v.is_nan() {
        let f = if v.is_snan() { INVALID } else { 0 };
        return (sf::pack_f16(&v.quieted()), f | d);
    }
    let r = sf::round_fp(v, FMT16, mode);
    (sf::pack_f16(&r.v), r.flags | d)
}

impl X86Interp {
    /// A VEX-prefixed instruction (`c` = `C4`/`C5`, the prefix bytes start
    /// at `pc`). A `66`/`F2`/`F3`/`REX`/`LOCK` prefix before VEX is `#UD`.
    pub(in crate::vcpu::interp_x86) fn exec_vex(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        p: Pfx,
        c: u8,
    ) -> Step {
        if p.opsize || p.rep != 0 || p.lock || p.has_rex {
            return Step::Illegal;
        }
        let (b1, pc) = fetch!(self.fetch8(pc));
        let (rex, map, b2, pc) = if c == 0xC5 {
            let rex = Rex {
                r: b1 & 0x80 == 0,
                ..Rex::default()
            };
            (rex, 1, b1, pc)
        } else {
            let (b2, pc) = fetch!(self.fetch8(pc));
            let rex = Rex {
                w: b2 & 0x80 != 0,
                r: b1 & 0x80 == 0,
                x: b1 & 0x40 == 0,
                b: b1 & 0x20 == 0,
            };
            (rex, b1 & 0x1f, b2, pc)
        };
        let pp = match b2 & 3 {
            0 => Mp::None,
            1 => Mp::P66,
            2 => Mp::F3,
            _ => Mp::F2,
        };
        let vx = Vex {
            map,
            pp,
            l: b2 & 4 != 0,
            w: rex.w,
            v: usize::from((!b2 >> 3) & 15),
        };
        let vp = Pfx {
            rex,
            has_rex: true,
            opsize: pp == Mp::P66,
            rep: match pp {
                Mp::F3 => 1,
                Mp::F2 => 2,
                _ => 0,
            },
            lock: false,
        };
        let (op, pc) = fetch!(self.fetch8(pc));
        if matches!((map, op), (2, 0xF2 | 0xF3 | 0xF5..=0xF7) | (3, 0xF0)) {
            return self.vex_bmi(mem, pc, vp, vx, op);
        }
        if !AVX {
            return Step::Illegal;
        }
        match map {
            1 => self.vex_0f(mem, pc, vp, vx, op),
            2 => self.vex_0f38(mem, pc, vp, vx, op),
            3 => self.vex_0f3a(mem, pc, vp, vx, op),
            _ => Step::Illegal,
        }
    }

    /// The VEX `0F` map.
    fn vex_0f(&mut self, mem: &mut GuestMemory, pc: u64, p: Pfx, vx: Vex, op: u8) -> Step {
        let mp = vx.pp;
        let l = vx.l;
        let nb = vx.bytes();
        if op == 0x77 && mp == Mp::None {
            // VZEROUPPER / VZEROALL
            if vx.v != 0 {
                return Step::Illegal;
            }
            self.ymm_hi = [0; 16];
            if l {
                self.xmm = [0; 16];
            }
            return self.next(pc);
        }
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        let int_form = matches!(
            op,
            0x60..=0x6D | 0x74..=0x76 | 0xD1..=0xD5 | 0xD8..=0xE5 | 0xE8..=0xEF | 0xF1..=0xF6 | 0xF8..=0xFE
        );
        if mp == Mp::P66 && int_form {
            if l && !AVX2 {
                return Step::Illegal;
            }
            // Shifts by a count take it from an xmm/m128 for both halves.
            let by_xmm = matches!(op, 0xD1..=0xD3 | 0xE1 | 0xE2 | 0xF1..=0xF3);
            let b = fetch!(self.ysrc(mem, &m, end, if by_xmm { 16 } else { nb }));
            let a = self.yget(vx.v);
            let lo = int_op(op, a[0], b[0], 16).unwrap_or(0);
            let hi = if l {
                int_op(op, a[1], if by_xmm { b[0] } else { b[1] }, 16).unwrap_or(0)
            } else {
                0
            };
            self.yset(m.reg, [lo, hi], l);
            return self.next(end);
        }
        let mem_op = !matches!(m.kind, RmKind::Reg(_));
        match (op, mp) {
            // ---- moves ----
            (0x10 | 0x11, Mp::None | Mp::P66) => {
                if vx.v != 0 {
                    return Step::Illegal;
                }
                if op == 0x10 {
                    let v = fetch!(self.ysrc(mem, &m, end, nb));
                    self.yset(m.reg, v, l);
                    self.next(end)
                } else {
                    let v = self.yget(m.reg);
                    self.v_store(mem, &m, end, v, nb, 1, l)
                }
            }
            (0x10 | 0x11, Mp::F3 | Mp::F2) => {
                let n = if mp == Mp::F3 { 4 } else { 8 };
                let lm: u128 = (1u128 << (8 * n)) - 1;
                match (m.kind, op) {
                    (RmKind::Reg(r), 0x10) => {
                        let v = (self.xmm[vx.v] & !lm) | (self.xmm[r] & lm);
                        self.yset(m.reg, [v, 0], false);
                    }
                    (RmKind::Reg(r), _) => {
                        let v = (self.xmm[vx.v] & !lm) | (self.xmm[m.reg] & lm);
                        self.yset(r, [v, 0], false);
                    }
                    (_, 0x10) => {
                        if vx.v != 0 {
                            return Step::Illegal;
                        }
                        let v = fetch!(self.ysrc(mem, &m, end, n));
                        self.yset(m.reg, [v[0], 0], false);
                    }
                    _ => {
                        if vx.v != 0 {
                            return Step::Illegal;
                        }
                        let a = self.yaddr(&m, end).unwrap_or(0);
                        fetch!(self.ywrite(mem, a, [self.xmm[m.reg], 0], n, 1));
                    }
                }
                self.next(end)
            }
            (0x12 | 0x16, Mp::None | Mp::P66) => {
                // VMOVLPS/VMOVLPD/VMOVHPS/VMOVHPD xmm, xmm, m64 and
                // VMOVHLPS/VMOVLHPS xmm, xmm, xmm.
                const LO: u128 = u64::MAX as u128;
                if l || (mp == Mp::P66 && !mem_op) {
                    return Step::Illegal;
                }
                let s = fetch!(self.ysrc(mem, &m, end, 8))[0];
                let a = self.xmm[vx.v];
                let v = match (op, mem_op) {
                    (0x12, true) => (a & !LO) | (s & LO),
                    (0x12, false) => (a & !LO) | (s >> 64),
                    _ => (a & LO) | (s << 64),
                };
                self.yset(m.reg, [v, 0], false);
                self.next(end)
            }
            (0x13 | 0x17, Mp::None | Mp::P66) => {
                // VMOVLPS/VMOVLPD/VMOVHPS/VMOVHPD m64, xmm.
                if l || vx.v != 0 || !mem_op {
                    return Step::Illegal;
                }
                let x = self.xmm[m.reg];
                let v = if op == 0x17 { x >> 64 } else { x };
                let a = self.yaddr(&m, end).unwrap_or(0);
                fetch!(self.ywrite(mem, a, [v, 0], 8, 1));
                self.next(end)
            }
            (0x12 | 0x16, Mp::F3) => {
                // VMOVSLDUP / VMOVSHDUP
                let s0 = usize::from(op == 0x16);
                self.v_un(mem, &m, end, vx, |v, _| {
                    let pick = |i: usize| u128::from(lane(v, 4, i));
                    pick(s0) | (pick(s0) << 32) | (pick(s0 + 2) << 64) | (pick(s0 + 2) << 96)
                })
            }
            (0x12, Mp::F2) => {
                // VMOVDDUP: the even qwords (an m64 for the 128-bit form).
                if vx.v != 0 {
                    return Step::Illegal;
                }
                let v = fetch!(self.ysrc(mem, &m, end, if l { 32 } else { 8 }));
                let d = |x: u128| (x & u128::from(u64::MAX)) * ((1u128 << 64) | 1);
                self.yset(m.reg, [d(v[0]), d(v[1])], l);
                self.next(end)
            }
            (0x14 | 0x15, Mp::None | Mp::P66) => {
                let w = if mp == Mp::None { 4 } else { 8 };
                self.v_nds(mem, &m, end, vx, |a, b| unpack(a, b, w, 16, op == 0x15))
            }
            (0x28 | 0x29, Mp::None | Mp::P66) | (0x6F | 0x7F, Mp::P66 | Mp::F3) => {
                // VMOVAPS/VMOVAPD/VMOVDQA (aligned) and VMOVDQU.
                if vx.v != 0 {
                    return Step::Illegal;
                }
                let align = if mp == Mp::F3 { 1 } else { nb as u64 };
                if matches!(op, 0x28 | 0x6F) {
                    let v = match m.kind {
                        RmKind::Reg(r) => self.yget(r),
                        _ => fetch!(Self::yread(
                            mem,
                            self.yaddr(&m, end).unwrap_or(0),
                            nb,
                            align
                        )),
                    };
                    self.yset(m.reg, v, l);
                    self.next(end)
                } else {
                    let v = self.yget(m.reg);
                    self.v_store(mem, &m, end, v, nb, align, l)
                }
            }
            (0x2B, Mp::None | Mp::P66) | (0xE7, Mp::P66) => {
                // VMOVNTPS/VMOVNTPD/VMOVNTDQ (aligned stores).
                if vx.v != 0 || !mem_op {
                    return Step::Illegal;
                }
                let v = self.yget(m.reg);
                self.v_store(mem, &m, end, v, nb, nb as u64, l)
            }
            (0xF0, Mp::F2) => {
                // VLDDQU
                if vx.v != 0 || !mem_op {
                    return Step::Illegal;
                }
                let v = fetch!(self.ysrc(mem, &m, end, nb));
                self.yset(m.reg, v, l);
                self.next(end)
            }
            (0x50, Mp::None | Mp::P66) => {
                // VMOVMSKPS / VMOVMSKPD
                let RmKind::Reg(r) = m.kind else {
                    return Step::Illegal;
                };
                if vx.v != 0 {
                    return Step::Illegal;
                }
                let w = if mp == Mp::None { 4 } else { 8 };
                self.gpr[m.reg] = sign_mask(self.yget(r), w, nb);
                self.next(end)
            }
            (0xD7, Mp::P66) => {
                // VPMOVMSKB
                let RmKind::Reg(r) = m.kind else {
                    return Step::Illegal;
                };
                if vx.v != 0 || (l && !AVX2) {
                    return Step::Illegal;
                }
                let v = self.yget(r);
                let hi = if l { movmskb(v[1], 16) << 16 } else { 0 };
                self.gpr[m.reg] = movmskb(v[0], 16) | hi;
                self.next(end)
            }
            (0x6E, Mp::P66) => {
                // VMOVD/VMOVQ xmm, r/m32|64
                if l || vx.v != 0 {
                    return Step::Illegal;
                }
                let w = if vx.w { 64 } else { 32 };
                let v = fetch!(self.read_operand(mem, self.op_of(m.kind, end), w));
                self.yset(m.reg, [u128::from(v), 0], false);
                self.next(end)
            }
            (0x7E, Mp::P66) => {
                // VMOVD/VMOVQ r/m32|64, xmm
                if l || vx.v != 0 {
                    return Step::Illegal;
                }
                let w = if vx.w { 64 } else { 32 };
                let v = self.xmm[m.reg] as u64;
                fetch!(self.write_operand(mem, self.op_of(m.kind, end), v, w));
                self.next(end)
            }
            (0x7E, Mp::F3) => {
                // VMOVQ xmm, xmm/m64
                if l || vx.v != 0 {
                    return Step::Illegal;
                }
                let v = fetch!(self.ysrc(mem, &m, end, 8))[0];
                self.yset(m.reg, [v & u128::from(u64::MAX), 0], false);
                self.next(end)
            }
            (0xD6, Mp::P66) => {
                // VMOVQ xmm/m64, xmm
                if l || vx.v != 0 {
                    return Step::Illegal;
                }
                let v = self.xmm[m.reg] & u128::from(u64::MAX);
                self.v_store(mem, &m, end, [v, 0], 8, 1, false)
            }
            (0x70, Mp::P66 | Mp::F3 | Mp::F2) => {
                // VPSHUFD / VPSHUFHW / VPSHUFLW
                if l && !AVX2 {
                    return Step::Illegal;
                }
                let (imm, end) = fetch!(self.fetch8(end));
                self.v_un(mem, &m, end, vx, |v, _| pshuf(v, mp, imm))
            }
            (0x71..=0x73, Mp::P66) => {
                // Shift by immediate: the destination is vvvv.
                let RmKind::Reg(r) = m.kind else {
                    return Step::Illegal;
                };
                if l && !AVX2 {
                    return Step::Illegal;
                }
                let (imm, end) = fetch!(self.fetch8(end));
                let src = self.yget(r);
                let Some(lo) = shift_imm_half(src[0], op, m.ext(), imm) else {
                    return Step::Illegal;
                };
                let hi = shift_imm_half(src[1], op, m.ext(), imm).unwrap_or(0);
                self.yset(vx.v, [lo, hi], l);
                self.next(end)
            }
            (0xC4, Mp::P66) => {
                // VPINSRW xmm, xmm, r32/m16, imm8
                if l {
                    return Step::Illegal;
                }
                let (imm, end) = fetch!(self.fetch8(end));
                let v = fetch!(self.read_operand(mem, self.op_of(m.kind, end), 16));
                let r = set_lane(self.xmm[vx.v], 2, usize::from(imm & 7), v);
                self.yset(m.reg, [r, 0], false);
                self.next(end)
            }
            (0xC5, Mp::P66) => {
                // VPEXTRW r32, xmm, imm8
                let RmKind::Reg(r) = m.kind else {
                    return Step::Illegal;
                };
                if l || vx.v != 0 {
                    return Step::Illegal;
                }
                let (imm, end) = fetch!(self.fetch8(end));
                self.gpr[m.reg] = lane(self.xmm[r], 2, usize::from(imm & 7));
                self.next(end)
            }
            (0xC6, Mp::None | Mp::P66) => {
                // VSHUFPS / VSHUFPD
                let (imm, end) = fetch!(self.fetch8(end));
                let b = fetch!(self.ysrc(mem, &m, end, nb));
                let a = self.yget(vx.v);
                let d = mp == Mp::P66;
                let hi = if l { shufp(a[1], b[1], d, imm, 1) } else { 0 };
                self.yset(m.reg, [shufp(a[0], b[0], d, imm, 0), hi], l);
                self.next(end)
            }
            (0xF7, Mp::P66) => {
                // VMASKMOVDQU
                if l || vx.v != 0 {
                    return Step::Illegal;
                }
                self.maskmov(mem, pc, p, true)
            }
            (0xAE, Mp::None) if matches!(m.ext(), 2 | 3) && mem_op => {
                // VLDMXCSR / VSTMXCSR
                if l || vx.v != 0 {
                    return Step::Illegal;
                }
                let a = self.yaddr(&m, end).unwrap_or(0);
                if m.ext() == 2 {
                    let v = fetch!(Self::read_mem(mem, a, 32)) as u32;
                    if v & !MXCSR_MASK != 0 {
                        return Step::Trap(Trap::Protection);
                    }
                    self.mxcsr = v;
                } else {
                    fetch!(self.write_mem(mem, a, u64::from(self.mxcsr), 32));
                }
                self.next(end)
            }
            // ---- bitwise float ops ----
            (0x54..=0x57, Mp::None | Mp::P66) => self.v_nds(mem, &m, end, vx, |a, b| match op {
                0x54 => a & b,
                0x55 => !a & b,
                0x56 => a | b,
                _ => a ^ b,
            }),
            // ---- float arithmetic ----
            (0x51, Mp::None | Mp::P66) => {
                self.v_fp_un(mem, &m, end, vx, |b, mx, _| fp_arith(op, mp, 0, b, mx))
            }
            (0x51 | 0x58 | 0x59 | 0x5C..=0x5F, _) => {
                let scalar = matches!(mp, Mp::F3 | Mp::F2).then_some(fp_shape(mp).2);
                self.v_fp(mem, &m, end, vx, scalar, |a, b, mx, _| {
                    fp_arith(op, mp, a, b, mx)
                })
            }
            (0x52 | 0x53, Mp::None) => {
                if vx.v != 0 {
                    return Step::Illegal;
                }
                self.v_un(mem, &m, end, vx, |b, _| rcp_lanes(op, 0, b, false))
            }
            (0x52 | 0x53, Mp::F3) => {
                let b = fetch!(self.ysrc(mem, &m, end, 4))[0];
                let r = rcp_lanes(op, self.xmm[vx.v], b, true);
                self.yset(m.reg, [r, 0], false);
                self.next(end)
            }
            (0xC2, _) => {
                let (imm, end) = fetch!(self.fetch8(end));
                let scalar = matches!(mp, Mp::F3 | Mp::F2).then_some(fp_shape(mp).2);
                self.v_fp(mem, &m, end, vx, scalar, |a, b, mx, _| {
                    fp_cmp(mp, a, b, mx, imm & 31)
                })
            }
            (0x2E | 0x2F, Mp::None | Mp::P66) => {
                if vx.v != 0 {
                    return Step::Illegal;
                }
                self.sse_comis(mem, pc, p, mp == Mp::P66, op == 0x2F)
            }
            (0xD0 | 0x7C | 0x7D, Mp::P66 | Mp::F2) => {
                let ff = if mp == Mp::P66 { Ff::D } else { Ff::S };
                let kind = match op {
                    0xD0 => 0,
                    0x7C => 1,
                    _ => 2,
                };
                self.v_fp(mem, &m, end, vx, None, |a, b, mx, _| {
                    fp_horiz(ff, kind, a, b, mx)
                })
            }
            // ---- conversions ----
            (0x2A, Mp::F3 | Mp::F2) => {
                // VCVTSI2SS / VCVTSI2SD xmm, xmm, r/m32|64
                let w = if vx.w { 64 } else { 32 };
                let raw = fetch!(self.read_operand(mem, self.op_of(m.kind, end), w));
                let v = if w == 64 {
                    raw as i64
                } else {
                    i64::from(raw as u32 as i32)
                };
                let ff = if mp == Mp::F3 { Ff::S } else { Ff::D };
                let (r, f) = from_int_lane(ff, v, self.mx());
                fetch!(self.sse_flags(f));
                let out = set_lane(self.xmm[vx.v], ff.bytes(), 0, r);
                self.yset(m.reg, [out, 0], false);
                self.next(end)
            }
            (0x2C | 0x2D, Mp::F3 | Mp::F2) => {
                // VCVT[T]SS2SI / VCVT[T]SD2SI r32|64, xmm/m
                if vx.v != 0 {
                    return Step::Illegal;
                }
                let ff = if mp == Mp::F3 { Ff::S } else { Ff::D };
                let v = fetch!(self.ysrc(mem, &m, end, ff.bytes()))[0];
                let bits = if vx.w { 64 } else { 32 };
                let (r, f) = to_int_lane(ff, lane(v, ff.bytes(), 0), bits, self.mx(), op == 0x2C);
                fetch!(self.sse_flags(f));
                self.set_reg(m.reg, r, bits);
                self.next(end)
            }
            (0x5A, Mp::F3 | Mp::F2) => {
                // VCVTSS2SD / VCVTSD2SS xmm, xmm, xmm/m
                let (from, to) = if mp == Mp::F3 {
                    (Ff::S, Ff::D)
                } else {
                    (Ff::D, Ff::S)
                };
                self.v_fp(mem, &m, end, vx, Some(from.bytes()), |a, b, mx, _| {
                    let (r, f) = cvt_lane(from, to, lane(b, from.bytes(), 0), mx);
                    (set_lane(a, to.bytes(), 0, r), f)
                })
            }
            (0x5A, Mp::None) | (0xE6, Mp::F3) => {
                // VCVTPS2PD / VCVTDQ2PD: widen the low 2 (4) lanes.
                if vx.v != 0 {
                    return Step::Illegal;
                }
                let n = if l { 4 } else { 2 };
                let s = fetch!(self.ysrc(mem, &m, end, 4 * n))[0];
                let mx = self.mx();
                let mut out = [0u64; 4];
                let mut flags = 0;
                for (i, o) in out.iter_mut().enumerate().take(n) {
                    let x = lane(s, 4, i);
                    let (r, f) = if op == 0x5A {
                        cvt_lane(Ff::S, Ff::D, x, mx)
                    } else {
                        from_int_lane(Ff::D, i64::from(x as u32 as i32), mx)
                    };
                    *o = r;
                    flags |= f;
                }
                fetch!(self.sse_flags(flags));
                self.yset(m.reg, ypack(&out[..n], 8), l);
                self.next(end)
            }
            (0x5A, Mp::P66) | (0xE6, Mp::P66 | Mp::F2) => {
                // VCVTPD2PS / VCVTTPD2DQ / VCVTPD2DQ: narrow 2 (4) doubles
                // into an XMM.
                if vx.v != 0 {
                    return Step::Illegal;
                }
                let n = if l { 4 } else { 2 };
                let s = fetch!(self.ysrc(mem, &m, end, nb));
                let mx = self.mx();
                let mut out = [0u64; 4];
                let mut flags = 0;
                for (i, o) in out.iter_mut().enumerate().take(n) {
                    let x = ylane(s, 8, i);
                    let (r, f) = if op == 0x5A {
                        cvt_lane(Ff::D, Ff::S, x, mx)
                    } else {
                        to_int_lane(Ff::D, x, 32, mx, mp == Mp::P66)
                    };
                    *o = r;
                    flags |= f;
                }
                fetch!(self.sse_flags(flags));
                self.yset(m.reg, [ypack(&out[..n], 4)[0], 0], false);
                self.next(end)
            }
            (0x5B, Mp::None | Mp::P66 | Mp::F3) => {
                self.v_fp_un(mem, &m, end, vx, |b, mx, _| cvt_5b(mp, b, mx))
            }
            _ => Step::Illegal,
        }
    }

    /// The VEX `0F 38` map (66 forms; the BMI opcodes are dispatched
    /// earlier).
    fn vex_0f38(&mut self, mem: &mut GuestMemory, pc: u64, p: Pfx, vx: Vex, op: u8) -> Step {
        if vx.pp != Mp::P66 {
            return Step::Illegal;
        }
        let l = vx.l;
        let nb = vx.bytes();
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        let mem_op = !matches!(m.kind, RmKind::Reg(_));
        // Instructions defined only with VEX.W0.
        let w0_only = matches!(
            op,
            0x0C..=0x0F | 0x13 | 0x16 | 0x18..=0x1A | 0x2C..=0x2F | 0x36 | 0x46 | 0x58..=0x5A | 0x78 | 0x79
        );
        if w0_only && vx.w {
            return Step::Illegal;
        }
        let int_256 = matches!(
            op,
            0x00..=0x0B | 0x1C..=0x25 | 0x28..=0x2B | 0x30..=0x3F | 0x40 | 0x45..=0x47
        );
        if l && int_256 && !AVX2 {
            return Step::Illegal;
        }
        match op {
            0x00..=0x0B => self.v_nds(mem, &m, end, vx, |a, b| ssse3_op(op, a, b, 16).unwrap_or(0)),
            0x1C..=0x1E => self.v_un(mem, &m, end, vx, |b, _| ssse3_op(op, 0, b, 16).unwrap_or(0)),
            0x28 | 0x29 | 0x2B | 0x37..=0x40 => {
                self.v_nds(mem, &m, end, vx, |a, b| sse41_op(op, a, b).unwrap_or(0))
            }
            0x0C | 0x0D => self.v_nds(mem, &m, end, vx, |a, ctl| permil_var(a, ctl, op == 0x0D)),
            0x0E | 0x0F | 0x17 => {
                // VTESTPS / VTESTPD (sign bits only) / VPTEST
                if vx.v != 0 {
                    return Step::Illegal;
                }
                let b = fetch!(self.ysrc(mem, &m, end, nb));
                let a = self.yget(m.reg);
                let sel: u128 = match op {
                    0x0E => 0x8000_0000_8000_0000_8000_0000_8000_0000,
                    0x0F => 0x8000_0000_0000_0000_8000_0000_0000_0000,
                    _ => u128::MAX,
                };
                let halves = if l { 2 } else { 1 };
                let zf = (0..halves).all(|h| a[h] & b[h] & sel == 0);
                let cf = (0..halves).all(|h| !a[h] & b[h] & sel == 0);
                self.flags = Flags {
                    cf,
                    zf,
                    ..Flags::default()
                };
                self.next(end)
            }
            0x13 => {
                // VCVTPH2PS
                if !F16C || vx.v != 0 {
                    return Step::Illegal;
                }
                let n = if l { 8 } else { 4 };
                let s = fetch!(self.ysrc(mem, &m, end, 2 * n))[0];
                let mut out = [0u64; 8];
                let mut flags = 0;
                for (i, o) in out.iter_mut().enumerate().take(n) {
                    let (r, f) = half_to_single(lane(s, 2, i) as u16);
                    *o = r;
                    flags |= f;
                }
                fetch!(self.sse_flags(flags));
                self.yset(m.reg, ypack(&out[..n], 4), l);
                self.next(end)
            }
            0x16 | 0x36 => {
                // VPERMPS / VPERMD: dst[i] = src2[src1[i] & 7]
                if !l || !AVX2 {
                    return Step::Illegal;
                }
                let b = fetch!(self.ysrc(mem, &m, end, 32));
                let idx = self.yget(vx.v);
                let out: [u64; 8] =
                    core::array::from_fn(|i| ylane(b, 4, (ylane(idx, 4, i) & 7) as usize));
                self.yset(m.reg, ypack(&out, 4), true);
                self.next(end)
            }
            0x18 | 0x19 | 0x58 | 0x59 | 0x78 | 0x79 => {
                // VBROADCASTSS/SD, VPBROADCASTD/Q/B/W
                let w = match op {
                    0x18 | 0x58 => 4,
                    0x19 | 0x59 => 8,
                    0x78 => 1,
                    _ => 2,
                };
                // (The register-source and integer forms are AVX2.)
                if vx.v != 0 || (op == 0x19 && !l) || ((op >= 0x58 || !mem_op) && !AVX2) {
                    return Step::Illegal;
                }
                let x = lane(fetch!(self.ysrc(mem, &m, end, w))[0], w, 0);
                let v = (0..16 / w).fold(0u128, |acc, i| set_lane(acc, w, i, x));
                self.yset(m.reg, [v, v], l);
                self.next(end)
            }
            0x1A | 0x5A => {
                // VBROADCASTF128 / VBROADCASTI128
                if !l || vx.v != 0 || !mem_op || (op == 0x5A && !AVX2) {
                    return Step::Illegal;
                }
                let v = fetch!(self.ysrc(mem, &m, end, 16))[0];
                self.yset(m.reg, [v, v], true);
                self.next(end)
            }
            0x20..=0x25 | 0x30..=0x35 => {
                // VPMOVSX* / VPMOVZX*
                if vx.v != 0 {
                    return Step::Illegal;
                }
                let (from, to) = match op & 0xF {
                    0 => (1, 2),
                    1 => (1, 4),
                    2 => (1, 8),
                    3 => (2, 4),
                    4 => (2, 8),
                    _ => (4, 8),
                };
                let per_half = 16 * from / to; // source bytes per result half
                let n = if l { 2 * per_half } else { per_half };
                let s = fetch!(self.ysrc(mem, &m, end, n))[0];
                let signed = op < 0x30;
                let lo = pmov_ext(s, from, to, signed);
                let hi = if l {
                    pmov_ext(s >> (8 * per_half), from, to, signed)
                } else {
                    0
                };
                self.yset(m.reg, [lo, hi], l);
                self.next(end)
            }
            0x2A => {
                // VMOVNTDQA (aligned load)
                if vx.v != 0 || !mem_op {
                    return Step::Illegal;
                }
                let a = self.yaddr(&m, end).unwrap_or(0);
                let v = fetch!(Self::yread(mem, a, nb, nb as u64));
                self.yset(m.reg, v, l);
                self.next(end)
            }
            0x2C | 0x2D | 0x8C => {
                // VMASKMOVPS/PD, VPMASKMOVD/Q loads: masked-off lanes read as
                // zero and never fault.
                if !mem_op || (op == 0x8C && !AVX2) {
                    return Step::Illegal;
                }
                let w = if op == 0x2C || (op == 0x8C && !vx.w) {
                    4
                } else {
                    8
                };
                let mask = self.yget(vx.v);
                let a = self.yaddr(&m, end).unwrap_or(0);
                let mut out = [0u64; 8];
                for (i, o) in out.iter_mut().enumerate().take(nb / w) {
                    if ylane(mask, w, i) >> (8 * w - 1) != 0 {
                        let ea = a.wrapping_add((i * w) as u64);
                        *o = fetch!(Self::read_mem(mem, ea, 8 * w as u32));
                    }
                }
                self.yset(m.reg, ypack(&out[..nb / w], w), l);
                self.next(end)
            }
            0x2E | 0x2F | 0x8E => {
                // VMASKMOVPS/PD, VPMASKMOVD/Q stores.
                if !mem_op || (op == 0x8E && !AVX2) {
                    return Step::Illegal;
                }
                let w = if op == 0x2E || (op == 0x8E && !vx.w) {
                    4
                } else {
                    8
                };
                let mask = self.yget(vx.v);
                let src = self.yget(m.reg);
                let a = self.yaddr(&m, end).unwrap_or(0);
                for i in 0..nb / w {
                    if ylane(mask, w, i) >> (8 * w - 1) != 0 {
                        let ea = a.wrapping_add((i * w) as u64);
                        fetch!(self.write_mem(mem, ea, ylane(src, w, i), 8 * w as u32));
                    }
                }
                self.next(end)
            }
            0x41 => {
                if l {
                    return Step::Illegal;
                }
                self.v_un(mem, &m, end, vx, |b, _| phminposuw(b))
            }
            0x45..=0x47 => {
                if !AVX2 || (op == 0x46 && vx.w) {
                    return Step::Illegal;
                }
                let w = if vx.w { 8 } else { 4 };
                self.v_nds(mem, &m, end, vx, |a, c| shift_var(op, a, c, w))
            }
            0x90..=0x93 => self.vex_gather(mem, pc, p, vx, op),
            0x96..=0x9F | 0xA6..=0xAF | 0xB6..=0xBF => self.vex_fma(mem, &m, end, vx, op),
            _ => Step::Illegal,
        }
    }

    /// FMA (`0F 38 96..9F/A6..AF/B6..BF`): `VFMADD`/`VFMSUB`/`VFNMADD`/
    /// `VFNMSUB` (packed and scalar) and `VFMADDSUB`/`VFMSUBADD`, in the
    /// 132/213/231 operand orders; `VEX.W` selects double precision.
    fn vex_fma(&mut self, mem: &GuestMemory, m: &ModRm, end: u64, vx: Vex, op: u8) -> Step {
        if !FMA {
            return Step::Illegal;
        }
        let form = (op >> 4) - 9; // 9x: 132, Ax: 213, Bx: 231
        let kind = op & 0xF;
        let ff = if vx.w { Ff::D } else { Ff::S };
        let w = ff.bytes();
        let scalar = kind >= 8 && kind & 1 == 1;
        let (neg_prod, neg_add) = match kind & 0xE {
            0x8 => (false, false),
            0xA => (false, true),
            0xC => (true, false),
            0xE => (true, true),
            _ => (false, false), // 6/7: alternating, below
        };
        let mx = self.mx();
        let x1 = self.yget(m.reg);
        let x2 = self.yget(vx.v);
        let (x3, lanes, l) = if scalar {
            (fetch!(self.ysrc(mem, m, end, w)), 1, false)
        } else {
            (
                fetch!(self.ysrc(mem, m, end, vx.bytes())),
                vx.bytes() / w,
                vx.l,
            )
        };
        let mut out = if scalar { [x1[0], 0] } else { [0, 0] };
        let mut flags = 0;
        for i in 0..lanes {
            let x = [ylane(x1, w, i), ylane(x2, w, i), ylane(x3, w, i)];
            // FMADDSUB: even lanes subtract; FMSUBADD: even lanes add.
            let neg_add = match kind {
                6 => i % 2 == 0,
                7 => i % 2 == 1,
                _ => neg_add,
            };
            let (r, f) = fma_lane(ff, x, form, neg_prod, neg_add, mx);
            let per = 16 / w;
            out[i / per] = set_lane(out[i / per], w, i % per, r);
            flags |= f;
        }
        fetch!(self.sse_flags(flags));
        self.yset(m.reg, out, l);
        self.next(end)
    }

    /// `VPGATHERDD/DQ/QD/QQ`, `VGATHERDPS/DPD/QPS/QPD` (`0F 38 90..93`): a
    /// VSIB memory operand (`SIB.index` names a vector register of dword or
    /// qword indices), the destination in ModRM.reg and the element mask in
    /// `vvvv`. Elements load in order, each clearing its mask element, so a
    /// fault leaves the completed ones done (the instruction restarts from
    /// there); at the end the whole mask is zero.
    fn vex_gather(&mut self, mem: &GuestMemory, pc: u64, p: Pfx, vx: Vex, op: u8) -> Step {
        if !AVX2 {
            return Step::Illegal;
        }
        let (modrm, pc) = fetch!(self.fetch8(pc));
        let md = modrm >> 6;
        if md == 3 || modrm & 7 != 4 {
            return Step::Illegal;
        }
        let (sib, pc) = fetch!(self.fetch8(pc));
        let scale = sib >> 6;
        let idx = usize::from((sib >> 3) & 7) | (usize::from(p.rex.x) << 3);
        let base_f = usize::from(sib & 7) | (usize::from(p.rex.b) << 3);
        let (base, disp, end) = if sib & 7 == 5 && md == 0 {
            let (d, e) = fetch!(self.fetch_i32(pc));
            (0, i64::from(d), e)
        } else {
            let b = self.gpr[base_f];
            match md {
                1 => {
                    let (d, e) = fetch!(self.fetch_i8(pc));
                    (b, i64::from(d), e)
                }
                2 => {
                    let (d, e) = fetch!(self.fetch_i32(pc));
                    (b, i64::from(d), e)
                }
                _ => (b, 0, pc),
            }
        };
        let dst = usize::from((modrm >> 3) & 7) | (usize::from(p.rex.r) << 3);
        let mask_r = vx.v;
        if dst == idx || dst == mask_r || idx == mask_r {
            return Step::Illegal;
        }
        let qidx = op & 1 == 1; // 91/93: qword indices
        let ew = if vx.w { 8 } else { 4 }; // element width
        let iw = if qidx { 8 } else { 4 };
        let n = if vx.l { 32 } else { 16 } / iw.max(ew);
        let index = self.yget(idx);
        // The destination/mask span: n elements (a QD/QPS form's elements
        // fill only an XMM).
        let span_l = vx.l && !(qidx && ew == 4);
        for i in 0..n {
            let mask = self.yget(mask_r);
            if ylane(mask, ew, i) >> (8 * ew - 1) == 0 {
                continue;
            }
            let ix = if qidx {
                ylane(index, 8, i)
            } else {
                i64::from(ylane(index, 4, i) as u32 as i32) as u64
            };
            let ea = base.wrapping_add(ix << scale).wrapping_add(disp as u64);
            let ea = if self.addr32 { ea & 0xffff_ffff } else { ea };
            let v = fetch!(Self::read_mem(mem, self.lin(ea), 8 * ew as u32));
            let mut d = self.yget(dst);
            let per = 16 / ew;
            d[i / per] = set_lane(d[i / per], ew, i % per, v);
            let mut mk = mask;
            mk[i / per] = set_lane(mk[i / per], ew, i % per, 0);
            self.xmm[dst] = d[0];
            self.ymm_hi[dst] = d[1];
            self.xmm[mask_r] = mk[0];
            self.ymm_hi[mask_r] = mk[1];
        }
        // Elements beyond n (and the upper half for a 128-bit span) are
        // zeroed in the destination; the mask is cleared entirely.
        let mut d = self.yget(dst);
        let keep = n * ew; // bytes of real elements
        if keep < 16 {
            d[0] &= (1u128 << (8 * keep)) - 1;
        }
        self.yset(dst, d, span_l);
        self.yset(mask_r, [0, 0], false);
        self.next(end)
    }

    /// The VEX `0F 3A` map (66 forms; `RORX` is dispatched earlier). Every
    /// instruction here takes an `imm8`.
    fn vex_0f3a(&mut self, mem: &mut GuestMemory, pc: u64, p: Pfx, vx: Vex, op: u8) -> Step {
        if vx.pp != Mp::P66 {
            return Step::Illegal;
        }
        let l = vx.l;
        let nb = vx.bytes();
        let (m, imm, end) = fetch!(self.modrm_imm(pc, p, true));
        let w0_only = matches!(op, 0x02 | 0x04..=0x06 | 0x18 | 0x19 | 0x1D | 0x38 | 0x39 | 0x46 | 0x4A..=0x4C);
        if (w0_only && vx.w) || (matches!(op, 0x00 | 0x01) && !vx.w) {
            return Step::Illegal;
        }
        if l && matches!(op, 0x0E | 0x0F | 0x42 | 0x4C) && !AVX2 {
            return Step::Illegal;
        }
        match op {
            0x00 | 0x01 => {
                // VPERMQ / VPERMPD ymm, ymm/m256, imm8
                if !l || vx.v != 0 || !AVX2 {
                    return Step::Illegal;
                }
                let s = fetch!(self.ysrc(mem, &m, end, 32));
                let out: [u64; 4] =
                    core::array::from_fn(|i| ylane(s, 8, usize::from((imm >> (2 * i)) & 3)));
                self.yset(m.reg, ypack(&out, 8), true);
                self.next(end)
            }
            0x02 => {
                // VPBLENDD
                if !AVX2 {
                    return Step::Illegal;
                }
                let b = fetch!(self.ysrc(mem, &m, end, nb));
                let a = self.yget(vx.v);
                let out: [u64; 8] = core::array::from_fn(|i| {
                    let s = if (imm >> i) & 1 != 0 { b } else { a };
                    ylane(s, 4, i)
                });
                self.yset(m.reg, ypack(&out[..nb / 4], 4), l);
                self.next(end)
            }
            0x04 => self.v_un(mem, &m, end, vx, |v, _| {
                // VPERMILPS by imm8 (same selection in both halves)
                (0..4).fold(0, |acc, i| {
                    set_lane(acc, 4, i, lane(v, 4, usize::from((imm >> (2 * i)) & 3)))
                })
            }),
            0x05 => self.v_un(mem, &m, end, vx, |v, h| {
                // VPERMILPD by imm8 (bits 2h, 2h+1 for half h)
                let k = imm >> (2 * h);
                (0..2).fold(0, |acc, i| {
                    set_lane(acc, 8, i, lane(v, 8, usize::from((k >> i) & 1)))
                })
            }),
            0x06 | 0x46 => {
                // VPERM2F128 / VPERM2I128
                if !l || (op == 0x46 && !AVX2) {
                    return Step::Illegal;
                }
                let b = fetch!(self.ysrc(mem, &m, end, 32));
                let a = self.yget(vx.v);
                self.yset(m.reg, perm2x128(a, b, imm), true);
                self.next(end)
            }
            0x08 | 0x09 => {
                self.v_fp_un(mem, &m, end, vx, |b, mx, _| round_lanes(op, 0, b, imm, mx))
            }
            0x0A | 0x0B => {
                let n = if op == 0x0A { 4 } else { 8 };
                self.v_fp(mem, &m, end, vx, Some(n), |a, b, mx, _| {
                    round_lanes(op, a, b, imm, mx)
                })
            }
            0x0C..=0x0E => {
                // VBLENDPS (imm bits 4..7 for the high half) / VBLENDPD
                // (bits 2..3) / VPBLENDW (the same 8 bits for both)
                let sh = match op {
                    0x0C => 4,
                    0x0D => 2,
                    _ => 0,
                };
                let b = fetch!(self.ysrc(mem, &m, end, nb));
                let a = self.yget(vx.v);
                let hi = if l {
                    blend_imm(op, a[1], b[1], imm >> sh)
                } else {
                    0
                };
                self.yset(m.reg, [blend_imm(op, a[0], b[0], imm), hi], l);
                self.next(end)
            }
            0x0F => self.v_nds(mem, &m, end, vx, |a, b| palignr(a, b, imm, 16)),
            0x14..=0x17 => {
                // VPEXTRB/W/D/Q, VEXTRACTPS: as the SSE4.1 forms.
                if l || vx.v != 0 {
                    return Step::Illegal;
                }
                self.exec_0f3a(mem, pc, p, op)
            }
            0x18 | 0x38 => {
                // VINSERTF128 / VINSERTI128
                if !l || (op == 0x38 && !AVX2) {
                    return Step::Illegal;
                }
                let s = fetch!(self.ysrc(mem, &m, end, 16))[0];
                let mut a = self.yget(vx.v);
                a[usize::from(imm & 1)] = s;
                self.yset(m.reg, a, true);
                self.next(end)
            }
            0x19 | 0x39 => {
                // VEXTRACTF128 / VEXTRACTI128
                if !l || vx.v != 0 || (op == 0x39 && !AVX2) {
                    return Step::Illegal;
                }
                let v = self.yget(m.reg)[usize::from(imm & 1)];
                self.v_store(mem, &m, end, [v, 0], 16, 1, false)
            }
            0x1D => {
                // VCVTPS2PH xmm/m64|m128, xmm|ymm, imm8
                if !F16C || vx.v != 0 {
                    return Step::Illegal;
                }
                let mx = self.mx();
                let mode = if imm & 4 != 0 {
                    mx.mode
                } else {
                    Round::from_x86(u32::from(imm & 3))
                };
                let s = self.yget(m.reg);
                let n = if l { 8 } else { 4 };
                let mut out = [0u64; 8];
                let mut flags = 0;
                for (i, o) in out.iter_mut().enumerate().take(n) {
                    let (h, f) = single_to_half(ylane(s, 4, i) as u32, mode, mx);
                    *o = u64::from(h);
                    flags |= f;
                }
                fetch!(self.sse_flags(flags));
                let v = ypack(&out[..n], 2);
                self.v_store(mem, &m, end, v, 2 * n, 1, false)
            }
            0x20 | 0x22 => {
                // VPINSRB / VPINSRD|Q
                if l {
                    return Step::Illegal;
                }
                let (w, idx) = match op {
                    0x20 => (8, usize::from(imm & 15)),
                    _ if vx.w => (64, usize::from(imm & 1)),
                    _ => (32, usize::from(imm & 3)),
                };
                let v = match m.kind {
                    RmKind::Reg(r) => self.gpr[r],
                    _ => fetch!(Self::read_mem(mem, self.yaddr(&m, end).unwrap_or(0), w)),
                };
                let r = set_lane(self.xmm[vx.v], (w / 8) as usize, idx, v);
                self.yset(m.reg, [r, 0], false);
                self.next(end)
            }
            0x21 => {
                // VINSERTPS
                if l {
                    return Step::Illegal;
                }
                let src = match m.kind {
                    RmKind::Reg(r) => lane(self.xmm[r], 4, usize::from(imm >> 6)),
                    _ => fetch!(Self::read_mem(mem, self.yaddr(&m, end).unwrap_or(0), 32)),
                };
                let mut x = set_lane(self.xmm[vx.v], 4, usize::from((imm >> 4) & 3), src);
                for i in 0..4 {
                    if (imm >> i) & 1 != 0 {
                        x = set_lane(x, 4, i, 0);
                    }
                }
                self.yset(m.reg, [x, 0], false);
                self.next(end)
            }
            0x40 => self.v_fp(mem, &m, end, vx, None, |a, b, mx, _| {
                dpp(a, b, true, imm, mx)
            }),
            0x41 => {
                if l {
                    return Step::Illegal;
                }
                self.v_fp(mem, &m, end, vx, None, |a, b, mx, _| {
                    dpp(a, b, false, imm, mx)
                })
            }
            0x42 => {
                // VMPSADBW (the high half uses imm bits 3..5)
                let b = fetch!(self.ysrc(mem, &m, end, nb));
                let a = self.yget(vx.v);
                let hi = if l { mpsadbw(a[1], b[1], imm >> 3) } else { 0 };
                self.yset(m.reg, [mpsadbw(a[0], b[0], imm), hi], l);
                self.next(end)
            }
            0x4A..=0x4C => {
                // VBLENDVPS / VBLENDVPD / VPBLENDVB: the mask register is
                // imm8[7:4].
                let w = match op {
                    0x4A => 4,
                    0x4B => 8,
                    _ => 1,
                };
                let mask = self.yget(usize::from(imm >> 4));
                let b = fetch!(self.ysrc(mem, &m, end, nb));
                let a = self.yget(vx.v);
                let hi = if l { blendv(a[1], b[1], mask[1], w) } else { 0 };
                self.yset(m.reg, [blendv(a[0], b[0], mask[0], w), hi], l);
                self.next(end)
            }
            0x60..=0x63 => {
                // VPCMPESTRM / VPCMPESTRI / VPCMPISTRM / VPCMPISTRI
                if l || vx.v != 0 {
                    return Step::Illegal;
                }
                let b = fetch!(self.ysrc(mem, &m, end, 16))[0];
                self.pcmpstr_run(op, imm, self.xmm[m.reg], b, vx.w);
                if op & 1 == 0 {
                    self.ymm_hi[0] = 0;
                }
                self.next(end)
            }
            _ => Step::Illegal,
        }
    }

    /// The VEX-encoded general-purpose instructions (BMI1/BMI2): `0F 38
    /// F2/F3/F5/F6/F7` and `0F 3A F0`. `VEX.W` selects 64-bit operands;
    /// `VEX.L` must be 0.
    fn vex_bmi(&mut self, mem: &mut GuestMemory, pc: u64, p: Pfx, vx: Vex, op: u8) -> Step {
        let w: u32 = if vx.w { 64 } else { 32 };
        let mask = if vx.w { u64::MAX } else { 0xffff_ffff };
        let (bmi1, bmi2) = match (vx.map, op, vx.pp) {
            (2, 0xF2 | 0xF3 | 0xF7, Mp::None) => (true, false),
            (2, 0xF5, Mp::None | Mp::F3 | Mp::F2)
            | (2, 0xF6, Mp::F2)
            | (2, 0xF7, Mp::P66 | Mp::F3 | Mp::F2)
            | (3, 0xF0, Mp::F2) => (false, true),
            _ => return Step::Illegal,
        };
        if vx.l || (bmi1 && !BMI1) || (bmi2 && !BMI2) {
            return Step::Illegal;
        }
        let (m, imm, end) = fetch!(self.modrm_imm(pc, p, vx.map == 3));
        let src = fetch!(self.read_operand(mem, self.op_of(m.kind, end), w));
        let v = self.gpr[vx.v] & mask;
        let szf = |r: u64| -> (bool, bool) { (r == 0, (r >> (w - 1)) & 1 != 0) };
        match (vx.map, op, vx.pp) {
            (2, 0xF2, _) => {
                // ANDN: dst = !vvvv & r/m
                let r = !v & src & mask;
                let (zf, sf) = szf(r);
                self.flags = Flags {
                    zf,
                    sf,
                    ..Flags::default()
                };
                self.set_reg(m.reg, r, w);
            }
            (2, 0xF3, _) => {
                // BLSR (/1), BLSMSK (/2), BLSI (/3): dst = vvvv
                let r = match m.ext() {
                    1 => src.wrapping_sub(1) & src,
                    2 => src.wrapping_sub(1) ^ src,
                    3 => src.wrapping_neg() & src,
                    _ => return Step::Illegal,
                } & mask;
                let (zf, sf) = szf(r);
                self.flags = Flags {
                    zf,
                    sf,
                    cf: if m.ext() == 3 { src != 0 } else { src == 0 },
                    ..Flags::default()
                };
                self.set_reg(vx.v, r, w);
            }
            (2, 0xF5, Mp::None) => {
                // BZHI: zero the bits from index vvvv[7:0] up
                let n = u32::from(v as u8);
                let r = if n < w { src & ((1u64 << n) - 1) } else { src };
                let (zf, sf) = szf(r);
                self.flags = Flags {
                    zf,
                    sf,
                    cf: n > w - 1,
                    ..Flags::default()
                };
                self.set_reg(m.reg, r, w);
            }
            (2, 0xF5, Mp::F3) => {
                // PEXT: gather the bits of vvvv selected by r/m
                let (mut r, mut k) = (0u64, 0);
                for i in 0..w {
                    if (src >> i) & 1 != 0 {
                        r |= ((v >> i) & 1) << k;
                        k += 1;
                    }
                }
                self.set_reg(m.reg, r, w);
            }
            (2, 0xF5, _) => {
                // PDEP: scatter the low bits of vvvv to the r/m mask bits
                let (mut r, mut k) = (0u64, 0);
                for i in 0..w {
                    if (src >> i) & 1 != 0 {
                        r |= ((v >> k) & 1) << i;
                        k += 1;
                    }
                }
                self.set_reg(m.reg, r, w);
            }
            (2, 0xF6, _) => {
                // MULX: reg:vvvv = rDX * r/m (unsigned), no flags; the high
                // half wins when both name the same register.
                let prod = u128::from(self.gpr[RDX] & mask) * u128::from(src);
                let (lo, hi) = (prod as u64 & mask, (prod >> w) as u64);
                self.set_reg(vx.v, lo, w);
                self.set_reg(m.reg, hi, w);
            }
            (2, 0xF7, Mp::None) => {
                // BEXTR: the vvvv[15:8]-bit field of r/m at vvvv[7:0]
                let start = u32::from(v as u8);
                let len = u32::from((v >> 8) as u8);
                let r = if start >= w {
                    0
                } else {
                    let x = src >> start;
                    if len >= 64 {
                        x
                    } else {
                        x & ((1u64 << len) - 1)
                    }
                };
                self.flags = Flags {
                    zf: r == 0,
                    ..Flags::default()
                };
                self.set_reg(m.reg, r, w);
            }
            (2, 0xF7, _) => {
                // SHLX (66) / SARX (F3) / SHRX (F2): count = vvvv mod width
                let c = (v & u64::from(w - 1)) as u32;
                let r = match vx.pp {
                    Mp::P66 => src << c,
                    Mp::F3 => {
                        let s = 64 - w;
                        ((((src << s) as i64) >> s) >> c) as u64
                    }
                    _ => src >> c,
                };
                self.set_reg(m.reg, r & mask, w);
            }
            _ => {
                // RORX
                let c = u32::from(imm) & (w - 1);
                let r = if w == 64 {
                    src.rotate_right(c)
                } else {
                    u64::from((src as u32).rotate_right(c))
                };
                self.set_reg(m.reg, r, w);
            }
        }
        self.next(end)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp, clippy::unreadable_literal)]

    use super::*;
    use crate::vcpu::Prot;
    use crate::vcpu::interp_x86::{RAX, RBX, RCX};

    const CODE: u64 = 0x1_1000;
    const STACK: u64 = 0x1_F000;
    const DATA: u64 = 0x1_4000;

    fn mem() -> GuestMemory {
        let mut m = GuestMemory::new(0x1_0000, 16 * crate::vcpu::mem::PAGE_SIZE);
        m.map(0x1_0000, 16 * crate::vcpu::mem::PAGE_SIZE, Prot::rwx())
            .unwrap();
        m
    }

    fn run(
        m: &mut GuestMemory,
        code: &[u8],
        setup: impl FnOnce(&mut X86Interp),
    ) -> (X86Interp, Step) {
        m.write_init(CODE, code).unwrap();
        let mut c = X86Interp::new(CODE, STACK);
        setup(&mut c);
        let s = c.exec(m);
        (c, s)
    }

    fn ps(v: [f32; 4]) -> u128 {
        v.iter()
            .enumerate()
            .fold(0, |acc, (i, x)| acc | (u128::from(x.to_bits()) << (32 * i)))
    }

    #[test]
    fn vaddps_256_and_128_upper_zeroing() {
        let mut m = mem();
        // vaddps ymm0, ymm1, ymm2
        let (c, s) = run(&mut m, &[0xC5, 0xF4, 0x58, 0xC2], |c| {
            c.xmm[1] = ps([1.0, 2.0, 3.0, 4.0]);
            c.ymm_hi[1] = ps([5.0, 6.0, 7.0, 8.0]);
            c.xmm[2] = ps([0.5; 4]);
            c.ymm_hi[2] = ps([10.0; 4]);
        });
        assert!(matches!(s, Step::Next));
        assert_eq!(c.xmm[0], ps([1.5, 2.5, 3.5, 4.5]));
        assert_eq!(c.ymm_hi[0], ps([15.0, 16.0, 17.0, 18.0]));
        // vaddps xmm0, xmm1, xmm2 zeroes bits 255:128; legacy addps keeps them.
        let (c, _) = run(&mut m, &[0xC5, 0xF0, 0x58, 0xC2], |c| c.ymm_hi[0] = 7);
        assert_eq!(c.ymm_hi[0], 0);
        let (c, _) = run(&mut m, &[0x0F, 0x58, 0xC2], |c| c.ymm_hi[0] = 7);
        assert_eq!(c.ymm_hi[0], 7);
    }

    #[test]
    fn vzeroupper_vzeroall_and_vex_prefix_rules() {
        let mut m = mem();
        let set = |c: &mut X86Interp| {
            c.xmm = [1; 16];
            c.ymm_hi = [2; 16];
        };
        let (c, _) = run(&mut m, &[0xC5, 0xF8, 0x77], set);
        assert_eq!((c.xmm, c.ymm_hi), ([1; 16], [0; 16]));
        let (c, _) = run(&mut m, &[0xC5, 0xFC, 0x77], set);
        assert_eq!((c.xmm, c.ymm_hi), ([0; 16], [0; 16]));
        // VEX.vvvv must be 1111 for VZEROUPPER.
        assert!(matches!(
            run(&mut m, &[0xC5, 0xF0, 0x77], set).1,
            Step::Illegal
        ));
        // A 66/F2/F3/REX/LOCK prefix in front of VEX is #UD.
        for pfx in [0x66u8, 0xF2, 0xF3, 0x40, 0xF0] {
            let code = [pfx, 0xC5, 0xF4, 0x58, 0xC2];
            assert!(matches!(run(&mut m, &code, |_| {}).1, Step::Illegal));
        }
        // A 256-bit VMOVAPS needs 32-byte alignment.
        let (_, s) = run(&mut m, &[0xC5, 0xFC, 0x28, 0x03], |c| {
            c.gpr[RBX] = DATA + 16;
        });
        assert!(matches!(s, Step::Trap(Trap::Protection)));
        let (_, s) = run(&mut m, &[0xC5, 0xFC, 0x28, 0x03], |c| {
            c.gpr[RBX] = DATA + 32;
        });
        assert!(matches!(s, Step::Next));
    }

    #[test]
    fn gather_loads_elements_and_clears_the_mask() {
        let mut m = mem();
        for i in 0..16u32 {
            m.write_init(DATA + 4 * u64::from(i), &(100 + i).to_le_bytes())
                .unwrap();
        }
        // vpgatherdd xmm0, [rbx + xmm1*4], xmm2: elements 0, 2, 3 enabled.
        let code = [0xC4, 0xE2, 0x69, 0x90, 0x04, 0x8B];
        let (c, s) = run(&mut m, &code, |c| {
            c.gpr[RBX] = DATA;
            c.xmm[1] = 3 | (5 << 32) | (7 << 64) | (11 << 96);
            c.xmm[2] = 0x8000_0000 | (0x8000_0000u128 << 64) | (0x8000_0000u128 << 96);
            c.xmm[0] = 0xdead_0000_0000;
            c.ymm_hi[0] = 1;
        });
        assert!(matches!(s, Step::Next));
        assert_eq!(
            c.xmm[0],
            0x67 | (0xdead << 32) | (0x6b << 64) | (0x6f << 96)
        );
        assert_eq!((c.xmm[2], c.ymm_hi[2], c.ymm_hi[0]), (0, 0, 0));
        // The same register as mask and index is #UD.
        let (_, s) = run(&mut m, &[0xC4, 0xE2, 0x71, 0x90, 0x04, 0x8B], |_| {});
        assert!(matches!(s, Step::Illegal));
    }

    #[test]
    fn fma_rounds_once() {
        let mut m = mem();
        // vfmadd231sd xmm0, xmm1, xmm2: xmm0 = xmm1 * xmm2 + xmm0
        let (c, _) = run(&mut m, &[0xC4, 0xE2, 0xF1, 0xB9, 0xC2], |c| {
            c.xmm[1] = u128::from(0x3ff0_0000_0000_0001u64); // 1 + 2^-52
            c.xmm[2] = u128::from(0x3fef_ffff_ffff_ffffu64); // 1 - 2^-53
            c.xmm[0] = u128::from((-1.0f64).to_bits()) | (0x1234 << 64);
        });
        // Exactly 2^-53 - 2^-105 (a separate multiply would round to 0).
        assert_eq!(
            c.xmm[0],
            u128::from(0x3c9f_ffff_ffff_fffeu64) | (0x1234 << 64)
        );
        assert_eq!(c.mxcsr & 0x3f, 0);
    }

    #[test]
    fn f16c_conversions() {
        let mut m = mem();
        let src = ps([1.0, 65520.0, -0.5, 5.960_464_5e-8]);
        // vcvtps2ph xmm0, xmm1, 0 (nearest)
        let (c, _) = run(&mut m, &[0xC4, 0xE3, 0x79, 0x1D, 0xC8, 0x00], |c| {
            c.xmm[1] = src;
        });
        assert_eq!(c.xmm[0], 0x0001_b800_7c00_3c00);
        // ... and 3 (toward zero): 65520 stays finite, the max half.
        let (c, _) = run(&mut m, &[0xC4, 0xE3, 0x79, 0x1D, 0xC8, 0x03], |c| {
            c.xmm[1] = src;
        });
        assert_eq!(c.xmm[0], 0x0001_b800_7bff_3c00);
        // vcvtph2ps xmm0, xmm1: a half denormal is a normal single.
        let (c, _) = run(&mut m, &[0xC4, 0xE2, 0x79, 0x13, 0xC1], |c| {
            c.xmm[1] = 0x0001_b800_7c00_3c00;
        });
        assert_eq!(c.xmm[0], ps([1.0, f32::INFINITY, -0.5, 5.960_464_5e-8]));
    }

    #[test]
    fn avx2_permutes_and_broadcasts() {
        let mut m = mem();
        let q = |a: u64, b: u64| u128::from(a) | (u128::from(b) << 64);
        // vpermq ymm0, ymm1, 0x1b (reverse the quadwords)
        let (c, _) = run(&mut m, &[0xC4, 0xE3, 0xFD, 0x00, 0xC1, 0x1B], |c| {
            c.xmm[1] = q(0, 1);
            c.ymm_hi[1] = q(2, 3);
        });
        assert_eq!((c.xmm[0], c.ymm_hi[0]), (q(3, 2), q(1, 0)));
        // vperm2i128 ymm0, ymm1, ymm2, 0x21: [ymm1.hi, ymm2.lo]
        let (c, _) = run(&mut m, &[0xC4, 0xE3, 0x75, 0x46, 0xC2, 0x21], |c| {
            c.ymm_hi[1] = 11;
            c.xmm[2] = 22;
        });
        assert_eq!((c.xmm[0], c.ymm_hi[0]), (11, 22));
        // vpbroadcastb ymm0, xmm1
        let (c, _) = run(&mut m, &[0xC4, 0xE2, 0x7D, 0x78, 0xC1], |c| c.xmm[1] = 0x5a);
        let b = u128::MAX / 255 * 0x5a;
        assert_eq!((c.xmm[0], c.ymm_hi[0]), (b, b));
        // vpmovmskb eax, ymm1
        let (c, _) = run(&mut m, &[0xC5, 0xFD, 0xD7, 0xC1], |c| {
            c.xmm[1] = 0x80;
            c.ymm_hi[1] = 0x80u128 << 120;
        });
        assert_eq!(c.gpr[RAX], 0x8000_0001);
        // vinserti128 ymm0, ymm1, xmm2, 1 / vextracti128 xmm0, ymm1, 1
        let (c, _) = run(&mut m, &[0xC4, 0xE3, 0x75, 0x38, 0xC2, 0x01], |c| {
            c.xmm[1] = 5;
            c.xmm[2] = 6;
        });
        assert_eq!((c.xmm[0], c.ymm_hi[0]), (5, 6));
        let (c, _) = run(&mut m, &[0xC4, 0xE3, 0x7D, 0x39, 0xC8, 0x01], |c| {
            c.ymm_hi[1] = 9;
            c.ymm_hi[0] = 1;
        });
        assert_eq!((c.xmm[0], c.ymm_hi[0]), (9, 0));
        // vpsllvd ymm0, ymm1, ymm2: per-lane counts, 32+ zeroes the lane.
        let (c, _) = run(&mut m, &[0xC4, 0xE2, 0x75, 0x47, 0xC2], |c| {
            c.xmm[1] = 1 | (1 << 32);
            c.xmm[2] = 4 | (32 << 32);
        });
        assert_eq!(c.xmm[0], 16);
    }

    #[test]
    fn masked_load_skips_unmapped_lanes() {
        let mut m = mem();
        // vmaskmovps ymm0, ymm1, [rbx]: only lane 0 enabled, at the very end
        // of the mapping — the disabled lanes beyond it must not fault.
        let end = 0x1_0000 + 16 * crate::vcpu::mem::PAGE_SIZE;
        m.write_init(end - 4, &7u32.to_le_bytes()).unwrap();
        let (c, s) = run(&mut m, &[0xC4, 0xE2, 0x75, 0x2C, 0x03], |c| {
            c.gpr[RBX] = end - 4;
            c.xmm[1] = 0x8000_0000;
            c.xmm[0] = u128::MAX;
        });
        assert!(matches!(s, Step::Next));
        assert_eq!((c.xmm[0], c.ymm_hi[0]), (7, 0));
    }

    #[test]
    fn bmi_and_friends() {
        let mut m = mem();
        let r = |m: &mut GuestMemory, code: &[u8], rax: u64, rcx: u64| {
            run(m, code, |c| {
                c.gpr[RAX] = rax;
                c.gpr[RCX] = rcx;
                c.gpr[RDX] = 0x5555;
            })
            .0
        };
        // pdep rdx, rax, rcx / pext rdx, rax, rcx
        let c = r(&mut m, &[0xC4, 0xE2, 0xFB, 0xF5, 0xD1], 0b101, 0xF0F0);
        assert_eq!(c.gpr[RDX], 0x50);
        let c = r(&mut m, &[0xC4, 0xE2, 0xFA, 0xF5, 0xD1], 0xA0A0, 0xF0F0);
        assert_eq!(c.gpr[RDX], 0xAA);
        // bzhi rdx, rax, rcx: keep the low 12 bits; an index ≥ 64 sets CF.
        let c = r(&mut m, &[0xC4, 0xE2, 0xF0, 0xF5, 0xD0], u64::MAX, 12);
        assert_eq!((c.gpr[RDX], c.flags.cf), (0xfff, false));
        let c = r(&mut m, &[0xC4, 0xE2, 0xF0, 0xF5, 0xD0], u64::MAX, 64);
        assert_eq!((c.gpr[RDX], c.flags.cf), (u64::MAX, true));
        // andn rdx, rax, rcx
        let c = r(&mut m, &[0xC4, 0xE2, 0xF8, 0xF2, 0xD1], 0xFF00, 0xFFF0);
        assert_eq!(c.gpr[RDX], 0xF0);
        // blsr rdx, rax
        let c = r(&mut m, &[0xC4, 0xE2, 0xE8, 0xF3, 0xC8], 0b1100, 0);
        assert_eq!((c.gpr[RDX], c.flags.cf, c.flags.zf), (0b1000, false, false));
        // mulx rdx, rbx, rcx: rdx:rbx = rdx * rcx, flags untouched
        let (c, _) = run(&mut m, &[0xC4, 0xE2, 0xE3, 0xF6, 0xD1], |c| {
            c.gpr[RDX] = u64::MAX;
            c.gpr[RCX] = 3;
            c.flags.cf = true;
        });
        assert_eq!(
            (c.gpr[RDX], c.gpr[RBX], c.flags.cf),
            (2, u64::MAX - 2, true)
        );
        // shlx rdx, rax, rcx (count mod 64) / rorx rdx, rax, 8
        assert_eq!(
            r(&mut m, &[0xC4, 0xE2, 0xF1, 0xF7, 0xD0], 1, 65).gpr[RDX],
            2
        );
        let c = r(&mut m, &[0xC4, 0xE3, 0xFB, 0xF0, 0xD0, 0x08], 0x1234, 0);
        assert_eq!(c.gpr[RDX], 0x3400_0000_0000_0012);
        // bextr rdx, rax, rcx: 8 bits from bit 4
        let c = r(&mut m, &[0xC4, 0xE2, 0xF0, 0xF7, 0xD0], 0xABCD, 0x0804);
        assert_eq!(c.gpr[RDX], 0xBC);
        // lzcnt / tzcnt rdx, rax: a zero input gives the width and CF.
        let c = r(&mut m, &[0xF3, 0x48, 0x0F, 0xBD, 0xD0], 1, 0);
        assert_eq!((c.gpr[RDX], c.flags.cf), (63, false));
        let c = r(&mut m, &[0xF3, 0x48, 0x0F, 0xBC, 0xD0], 0, 0);
        assert_eq!((c.gpr[RDX], c.flags.cf, c.flags.zf), (64, true, false));
        // movbe eax, [rbx] / movbe [rbx], eax
        m.write_init(DATA, &[1, 2, 3, 4]).unwrap();
        let (c, _) = run(&mut m, &[0x0F, 0x38, 0xF0, 0x03], |c| c.gpr[RBX] = DATA);
        assert_eq!(c.gpr[RAX], 0x0102_0304);
        run(&mut m, &[0x0F, 0x38, 0xF1, 0x03], |c| {
            c.gpr[RBX] = DATA;
            c.gpr[RAX] = 0x1122_3344;
        });
        assert_eq!(m.read_vec(DATA, 4).unwrap(), [0x11, 0x22, 0x33, 0x44]);
    }

    #[test]
    fn xsave_xrstor_and_xgetbv() {
        let mut m = mem();
        let (c, _) = run(&mut m, &[0x0F, 0x01, 0xD0], |c| c.gpr[RCX] = 0);
        assert_eq!((c.gpr[RAX], c.gpr[RDX]), (7, 0));
        let (_, s) = run(&mut m, &[0x0F, 0x01, 0xD0], |c| c.gpr[RCX] = 1);
        assert!(matches!(s, Step::Trap(Trap::Protection)));
        // xsave64 [rbx], then xrstor64 [rbx] into a fresh CPU.
        let (src, s) = run(&mut m, &[0x48, 0x0F, 0xAE, 0x23], |c| {
            c.gpr[RBX] = DATA;
            c.gpr[RAX] = 7;
            for i in 0..16 {
                c.xmm[i] = 100 + i as u128;
                c.ymm_hi[i] = 200 + i as u128;
            }
            c.mxcsr = 0x3f80;
        });
        assert!(matches!(s, Step::Next));
        assert_eq!(m.read_vec(DATA + 512, 8).unwrap(), 7u64.to_le_bytes());
        let (dst, s) = run(&mut m, &[0x48, 0x0F, 0xAE, 0x2B], |c| {
            c.gpr[RBX] = DATA;
            c.gpr[RAX] = 7;
        });
        assert!(matches!(s, Step::Next));
        assert_eq!(
            (dst.xmm, dst.ymm_hi, dst.mxcsr),
            (src.xmm, src.ymm_hi, 0x3f80)
        );
        // XSTATE_BV clear for AVX: the upper halves are initialized (zero).
        m.write_init(DATA + 512, &3u64.to_le_bytes()).unwrap();
        let (dst, _) = run(&mut m, &[0x48, 0x0F, 0xAE, 0x2B], |c| {
            c.gpr[RBX] = DATA;
            c.gpr[RAX] = 7;
            c.ymm_hi = [1; 16];
        });
        assert_eq!((dst.xmm, dst.ymm_hi), (src.xmm, [0; 16]));
        // A nonzero XCOMP_BV (the unsupported compacted form), a misaligned
        // area, or an XSTATE_BV bit outside XCR0 is #GP.
        for (off, bytes, rbx) in [
            (520u64, 1u64, DATA),
            (512, 3, DATA + 16),
            (512, 8 | 3, DATA),
        ] {
            m.write_init(DATA + 512, &[0; 64]).unwrap();
            m.write_init(DATA + off, &bytes.to_le_bytes()).unwrap();
            let (_, s) = run(&mut m, &[0x48, 0x0F, 0xAE, 0x2B], |c| {
                c.gpr[RBX] = rbx;
                c.gpr[RAX] = 7;
            });
            assert!(matches!(s, Step::Trap(Trap::Protection)), "{off} {bytes}");
        }
    }

    #[test]
    fn cpuid_advertises_v3_consistently() {
        let mut m = mem();
        let cpuid = |m: &mut GuestMemory, leaf: u64, sub: u64| {
            let (c, _) = run(m, &[0x0F, 0xA2], |c| {
                c.gpr[RAX] = leaf;
                c.gpr[RCX] = sub;
            });
            (c.gpr[RAX], c.gpr[RBX], c.gpr[RCX], c.gpr[RDX])
        };
        let (_, _, ecx, _) = cpuid(&mut m, 1, 0);
        for bit in [12, 22, 26, 27, 28, 29] {
            assert_ne!(ecx & (1 << bit), 0, "leaf 1 ecx bit {bit}");
        }
        let (_, ebx, _, _) = cpuid(&mut m, 7, 0);
        assert_eq!(ebx & 0x128, 0x128); // BMI1, AVX2, BMI2
        assert_eq!(cpuid(&mut m, 0xD, 0), (7, 0x340, 0x340, 0));
        assert_eq!(cpuid(&mut m, 0xD, 2), (0x100, 0x240, 0, 0));
        assert_ne!(cpuid(&mut m, 0x8000_0001, 0).2 & (1 << 5), 0); // LZCNT
        assert!(cpuid(&mut m, 0, 0).0 >= 0xD);
    }
}
