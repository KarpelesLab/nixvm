//! MMX, SSE and SSE2 (the SIMD opcodes of the `0F` map, selected by the
//! "mandatory" prefix: none = packed single / MMX, `66` = packed double /
//! 128-bit integer, `F3` = scalar single, `F2` = scalar double).
//!
//! * **MMX** registers alias the low 64 bits of the x87 physical registers;
//!   every MMX instruction (except `EMMS`) puts the FPU in MMX mode (`TOP =
//!   0`, all tags valid) and writing an MMX register sets the aliased
//!   register's exponent field to all ones, exactly as hardware does.
//! * **Floating point** goes through [`super::super::softfloat`] with the
//!   `MXCSR` rounding mode, `DAZ`/`FTZ`, the x86 NaN rules (first operand
//!   wins), and per-lane exception flags; a host-arithmetic fast path handles
//!   the common normal-operand round-to-nearest case bit-exactly. Unmasked
//!   exceptions raise `#XM` without writing the destination.
//! * Legacy-SSE 128-bit memory operands must be 16-byte aligned (`#GP`)
//!   except for the explicitly unaligned forms (`MOVUPS`, `MOVDQU`, …).
//! * `RCPPS`/`RSQRTPS` reproduce the hardware 12-bit estimate tables.

#![allow(clippy::match_same_arms, clippy::single_match_else)]

use super::{Flags, GuestMemory, ModRm, Pfx, RDI, RmKind, Step, Trap, X86Interp, fetch, rd_fault};
use crate::vcpu::softfloat::F80;
use crate::vcpu::softfloat::{
    self as sf, Class, FMT32, FMT64, Fp, INEXACT, INVALID, Mx, Op, Round,
};

mod sse4;

/// SSE3 (`ADDSUB*`, `HADD*`/`HSUB*`, `MOV*DUP`, `LDDQU`, `FISTTP`) — not yet
/// advertised, so these encodings are `#UD`.
pub(super) const SSE3: bool = true;

/// The mandatory-prefix class of a SIMD opcode.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mp {
    None,
    P66,
    F3,
    F2,
}

/// `F2`/`F3` take precedence over `66` (which then just sets operand size,
/// meaningless here).
fn mp(p: Pfx) -> Mp {
    match p.rep {
        1 => Mp::F3,
        2 => Mp::F2,
        _ if p.opsize => Mp::P66,
        _ => Mp::None,
    }
}

// ---- lane helpers ----------------------------------------------------------------

/// Lane `i` of width `w` bytes.
#[inline]
fn lane(v: u128, w: usize, i: usize) -> u64 {
    let bits = 8 * w;
    let x = v >> (bits * i);
    if bits == 64 {
        x as u64
    } else {
        (x as u64) & ((1u64 << bits) - 1)
    }
}

/// Apply `f` lane-wise over the low `n` bytes (`n` = 8 for MMX, 16 for XMM).
#[inline]
fn map2(a: u128, b: u128, w: usize, n: usize, f: impl Fn(u64, u64) -> u64) -> u128 {
    let bits = 8 * w;
    let mask: u128 = if bits == 64 {
        u128::from(u64::MAX)
    } else {
        (1u128 << bits) - 1
    };
    let mut out = 0u128;
    for i in 0..n / w {
        let r = f(lane(a, w, i), lane(b, w, i));
        out |= (u128::from(r) & mask) << (bits * i);
    }
    out
}

const fn sx(v: u64, w: usize) -> i64 {
    let s = 64 - 8 * w as u32;
    ((v << s) as i64) >> s
}

fn sat_s(v: i64, w: usize) -> u64 {
    let bits = 8 * w as u32;
    let max = (1i64 << (bits - 1)) - 1;
    let min = -(1i64 << (bits - 1));
    v.clamp(min, max) as u64
}

fn sat_u(v: i64, w: usize) -> u64 {
    let bits = 8 * w as u32;
    v.clamp(0, (1i64 << bits) - 1) as u64
}

/// Interleave the low (`high == false`) or high halves of `a` and `b` in
/// `w`-byte lanes (`PUNPCKL*`/`PUNPCKH*`, `UNPCKLPS`/`UNPCKHPS`/…).
fn unpack(a: u128, b: u128, w: usize, n: usize, high: bool) -> u128 {
    let lanes = n / w / 2;
    let base = if high { lanes } else { 0 };
    let bits = 8 * w;
    let mut out = 0u128;
    for i in 0..lanes {
        out |= u128::from(lane(a, w, base + i)) << (bits * 2 * i);
        out |= u128::from(lane(b, w, base + i)) << (bits * (2 * i + 1));
    }
    out
}

/// `PACKSSWB`/`PACKUSWB`/`PACKSSDW`: narrow `w`-byte signed lanes of `a`
/// then `b` with signed or unsigned saturation.
fn pack(a: u128, b: u128, w: usize, n: usize, signed: bool) -> u128 {
    let lanes = n / w;
    let ow = w / 2;
    let mut out = 0u128;
    for (k, v) in [a, b].iter().enumerate() {
        for i in 0..lanes {
            let x = sx(lane(*v, w, i), w);
            let r = if signed { sat_s(x, ow) } else { sat_u(x, ow) };
            let r = r & ((1u64 << (8 * ow)) - 1);
            out |= u128::from(r) << (8 * ow * (k * lanes + i));
        }
    }
    out
}

/// Logical/arithmetic per-lane shift by `count` (a count ≥ the lane width
/// zeroes the lane, or fills it with the sign for arithmetic right shifts).
fn shift(v: u128, w: usize, n: usize, count: u64, kind: u8) -> u128 {
    let bits = 8 * w as u64;
    map2(v, 0, w, n, |x, _| match kind {
        0 => {
            // right logical
            if count >= bits { 0 } else { x >> count }
        }
        1 => {
            // left
            if count >= bits { 0 } else { x << count }
        }
        _ => {
            // right arithmetic
            (sx(x, w) >> count.min(bits - 1)) as u64
        }
    })
}

/// Expand per-byte "high bit set" flags to all-ones bytes (SWAR).
fn byte_mask_from_high(t: u128) -> u128 {
    ((t >> 7) & 0x0101_0101_0101_0101_0101_0101_0101_0101) * 0xff
}

/// `PCMPEQB` without a lane loop: a byte of `a ^ b` is zero iff equal.
fn pcmpeqb(a: u128, b: u128) -> u128 {
    const LO7: u128 = 0x7f7f_7f7f_7f7f_7f7f_7f7f_7f7f_7f7f_7f7f;
    let x = a ^ b;
    let t = !(((x & LO7) + LO7) | x | LO7);
    byte_mask_from_high(t)
}

/// `PMOVMSKB` over `n` bytes.
fn movmskb(v: u128, n: usize) -> u64 {
    let gather = |h: u64| ((h & 0x8080_8080_8080_8080).wrapping_mul(0x0002_0408_1020_4081)) >> 56;
    let lo = gather(v as u64);
    if n == 8 {
        lo
    } else {
        lo | (gather((v >> 64) as u64) << 8)
    }
}

/// The packed-integer binary ops shared by MMX (`n == 8`) and SSE2 (`n ==
/// 16`): `op` is the second opcode byte. `None` for opcodes not in this set.
#[allow(clippy::too_many_lines)]
fn int_op(op: u8, a: u128, b: u128, n: usize) -> Option<u128> {
    let cnt = b as u64; // shift-by-register count: the low 64 bits
    Some(match op {
        0x60 => unpack(a, b, 1, n, false),
        0x61 => unpack(a, b, 2, n, false),
        0x62 => unpack(a, b, 4, n, false),
        0x63 => pack(a, b, 2, n, true),
        0x64 => map2(
            a,
            b,
            1,
            n,
            |x, y| if sx(x, 1) > sx(y, 1) { 0xff } else { 0 },
        ),
        0x65 => map2(
            a,
            b,
            2,
            n,
            |x, y| if sx(x, 2) > sx(y, 2) { 0xffff } else { 0 },
        ),
        0x66 => map2(a, b, 4, n, |x, y| {
            if sx(x, 4) > sx(y, 4) { 0xffff_ffff } else { 0 }
        }),
        0x67 => pack(a, b, 2, n, false),
        0x68 => unpack(a, b, 1, n, true),
        0x69 => unpack(a, b, 2, n, true),
        0x6A => unpack(a, b, 4, n, true),
        0x6B => pack(a, b, 4, n, true),
        0x6C => unpack(a, b, 8, n, false),
        0x6D => unpack(a, b, 8, n, true),
        0x74 => {
            let r = pcmpeqb(a, b);
            if n == 8 { r & u128::from(u64::MAX) } else { r }
        }
        0x75 => map2(a, b, 2, n, |x, y| if x == y { 0xffff } else { 0 }),
        0x76 => map2(a, b, 4, n, |x, y| if x == y { 0xffff_ffff } else { 0 }),
        0xD1 => shift(a, 2, n, cnt, 0),
        0xD2 => shift(a, 4, n, cnt, 0),
        0xD3 => shift(a, 8, n, cnt, 0),
        0xD4 => map2(a, b, 8, n, u64::wrapping_add),
        0xD5 => map2(a, b, 2, n, u64::wrapping_mul),
        0xD8 => map2(a, b, 1, n, u64::saturating_sub),
        0xD9 => map2(a, b, 2, n, u64::saturating_sub),
        0xDA => map2(a, b, 1, n, u64::min),
        0xDB => a & b,
        0xDC => map2(a, b, 1, n, |x, y| (x + y).min(0xff)),
        0xDD => map2(a, b, 2, n, |x, y| (x + y).min(0xffff)),
        0xDE => map2(a, b, 1, n, u64::max),
        0xDF => !a & b,
        0xE0 => map2(a, b, 1, n, |x, y| (x + y + 1) >> 1),
        0xE1 => shift(a, 2, n, cnt, 2),
        0xE2 => shift(a, 4, n, cnt, 2),
        0xE3 => map2(a, b, 2, n, |x, y| (x + y + 1) >> 1),
        0xE4 => map2(a, b, 2, n, |x, y| (x * y) >> 16),
        0xE5 => map2(a, b, 2, n, |x, y| ((sx(x, 2) * sx(y, 2)) >> 16) as u64),
        0xE8 => map2(a, b, 1, n, |x, y| sat_s(sx(x, 1) - sx(y, 1), 1)),
        0xE9 => map2(a, b, 2, n, |x, y| sat_s(sx(x, 2) - sx(y, 2), 2)),
        0xEA => map2(a, b, 2, n, |x, y| if sx(x, 2) < sx(y, 2) { x } else { y }),
        0xEB => a | b,
        0xEC => map2(a, b, 1, n, |x, y| sat_s(sx(x, 1) + sx(y, 1), 1)),
        0xED => map2(a, b, 2, n, |x, y| sat_s(sx(x, 2) + sx(y, 2), 2)),
        0xEE => map2(a, b, 2, n, |x, y| if sx(x, 2) > sx(y, 2) { x } else { y }),
        0xEF => a ^ b,
        0xF1 => shift(a, 2, n, cnt, 1),
        0xF2 => shift(a, 4, n, cnt, 1),
        0xF3 => shift(a, 8, n, cnt, 1),
        0xF4 => map2(a, b, 8, n, |x, y| (x & 0xffff_ffff) * (y & 0xffff_ffff)),
        0xF5 => map2(a, b, 4, n, |x, y| {
            let lo = sx(x & 0xffff, 2) * sx(y & 0xffff, 2);
            let hi = sx(x >> 16, 2) * sx(y >> 16, 2);
            lo.wrapping_add(hi) as u64
        }),
        0xF6 => map2(a, b, 8, n, |x, y| {
            (0..8)
                .map(|i| ((x >> (8 * i)) & 0xff).abs_diff((y >> (8 * i)) & 0xff))
                .sum()
        }),
        0xF8 => map2(a, b, 1, n, u64::wrapping_sub),
        0xF9 => map2(a, b, 2, n, u64::wrapping_sub),
        0xFA => map2(a, b, 4, n, u64::wrapping_sub),
        0xFB => map2(a, b, 8, n, u64::wrapping_sub),
        0xFC => map2(a, b, 1, n, u64::wrapping_add),
        0xFD => map2(a, b, 2, n, u64::wrapping_add),
        0xFE => map2(a, b, 4, n, u64::wrapping_add),
        _ => return None,
    })
}

// ---- floating-point lane helpers ----------------------------------------------

/// A float lane format: `f32` (4-byte lanes) or `f64` (8-byte).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ff {
    S,
    D,
}

impl Ff {
    const fn bytes(self) -> usize {
        match self {
            Ff::S => 4,
            Ff::D => 8,
        }
    }
    fn fmt(self) -> sf::Fmt {
        match self {
            Ff::S => FMT32,
            Ff::D => FMT64,
        }
    }
    fn unpack(self, v: u64) -> Fp {
        match self {
            Ff::S => sf::unpack_f32(v as u32),
            Ff::D => sf::unpack_f64(v),
        }
    }
    fn pack(self, v: &Fp) -> u64 {
        match self {
            Ff::S => u64::from(sf::pack_f32(v)),
            Ff::D => sf::pack_f64(v),
        }
    }
    /// Is the bit pattern a normal (not zero/denormal/inf/NaN) value?
    fn normal(self, v: u64) -> bool {
        match self {
            Ff::S => {
                let e = (v >> 23) & 0xff;
                e != 0 && e != 0xff
            }
            Ff::D => {
                let e = (v >> 52) & 0x7ff;
                e != 0 && e != 0x7ff
            }
        }
    }
}

// Exact float equality is the point: these are error-free exactness checks.
#[allow(clippy::float_cmp)]
/// The host-arithmetic fast path for an SSE op in round-to-nearest: valid
/// when both operands and the result are normal, where host IEEE arithmetic
/// is bit-identical; inexactness comes from an error-free transformation.
fn fast_arith(op: Op, ff: Ff, a: u64, b: u64) -> Option<(u64, u32)> {
    if !ff.normal(a) || !ff.normal(b) {
        return None;
    }
    match ff {
        Ff::D => {
            let (x, y) = (f64::from_bits(a), f64::from_bits(b));
            let (r, exact) = match op {
                Op::Add | Op::Sub => {
                    let y = if op == Op::Sub { -y } else { y };
                    let s = x + y;
                    // TwoSum error term
                    let bb = s - x;
                    let err = (x - (s - bb)) + (y - bb);
                    (s, err == 0.0)
                }
                Op::Mul => {
                    let p = x * y;
                    (p, x.mul_add(y, -p) == 0.0)
                }
                Op::Div => {
                    let q = x / y;
                    (q, (-q).mul_add(y, x) == 0.0)
                }
            };
            let bits = r.to_bits();
            ff.normal(bits)
                .then_some((bits, if exact { 0 } else { INEXACT }))
        }
        Ff::S => {
            let (x, y) = (f32::from_bits(a as u32), f32::from_bits(b as u32));
            // f32 products are exact in f64 (and so is a quotient times the
            // divisor); sums use the f32 TwoSum error term.
            let (r, exact) = match op {
                Op::Add | Op::Sub => {
                    let y = if op == Op::Sub { -y } else { y };
                    let s = x + y;
                    let bb = s - x;
                    let err = (x - (s - bb)) + (y - bb);
                    (s, err == 0.0)
                }
                Op::Mul => {
                    let p = x * y;
                    (p, f64::from(p) == f64::from(x) * f64::from(y))
                }
                Op::Div => {
                    let q = x / y;
                    (q, f64::from(q) * f64::from(y) == f64::from(x))
                }
            };
            let bits = u64::from(r.to_bits());
            ff.normal(bits)
                .then_some((bits, if exact { 0 } else { INEXACT }))
        }
    }
}

/// One SSE arithmetic lane (`ADD`/`SUB`/`MUL`/`DIV`).
fn arith_lane(op: Op, ff: Ff, a: u64, b: u64, mx: Mx) -> (u64, u32) {
    if mx.mode == Round::Nearest
        && let Some(r) = fast_arith(op, ff, a, b)
    {
        return r;
    }
    let (v, f) = sf::sse_arith(op, ff.unpack(a), ff.unpack(b), ff.fmt(), mx);
    (ff.pack(&v), f)
}

/// One `SQRT` lane.
#[allow(clippy::float_cmp)] // an exactness check
fn sqrt_lane(ff: Ff, a: u64, mx: Mx) -> (u64, u32) {
    if mx.mode == Round::Nearest && ff.normal(a) && (a >> (8 * ff.bytes() - 1)) & 1 == 0 {
        let (bits, exact) = match ff {
            Ff::D => {
                let x = f64::from_bits(a);
                let s = x.sqrt();
                (s.to_bits(), s.mul_add(s, -x) == 0.0)
            }
            Ff::S => {
                let x = f32::from_bits(a as u32);
                let s = x.sqrt();
                (
                    u64::from(s.to_bits()),
                    f64::from(s) * f64::from(s) == f64::from(x),
                )
            }
        };
        return (bits, if exact { 0 } else { INEXACT });
    }
    let (v, f) = sf::sse_sqrt(ff.unpack(a), ff.fmt(), mx);
    (ff.pack(&v), f)
}

/// `MIN`/`MAX` lane: a NaN in either operand, or two zeros, return the
/// second operand; `IE` for any NaN (these are signaling compares).
fn minmax_lane(ff: Ff, a: u64, b: u64, mx: Mx, max: bool) -> (u64, u32) {
    let (ua, da) = mx.input(ff.unpack(a), ff.fmt());
    let (ub, db) = mx.input(ff.unpack(b), ff.fmt());
    let d = da | db;
    if ua.is_nan() || ub.is_nan() {
        // The second operand as read (a NaN as-is, a DAZ-flushed zero).
        let r = if ub.is_nan() { b } else { ff.pack(&ub) };
        return (r, INVALID | d);
    }
    if ua.class == Class::Zero && ub.class == Class::Zero {
        return (ff.pack(&ub), d);
    }
    let ord = sf::compare(&ua, &ub);
    let first = if max {
        ord == core::cmp::Ordering::Greater
    } else {
        ord == core::cmp::Ordering::Less
    };
    let pick = if first { ua } else { ub };
    // A DAZ-flushed operand is returned as the zero it was read as.
    (ff.pack(&pick), d)
}

/// `CMPccPS/PD/SS/SD` lane: all-ones if predicate `pred` (0..7) holds.
fn cmp_lane(ff: Ff, a: u64, b: u64, mx: Mx, pred: u8) -> (u64, u32) {
    use core::cmp::Ordering::{Equal, Less};
    let (ua, da) = mx.input(ff.unpack(a), ff.fmt());
    let (ub, db) = mx.input(ff.unpack(b), ff.fmt());
    let mut f = da | db;
    let ord = if ua.is_nan() || ub.is_nan() {
        let signaling = matches!(pred & 7, 1 | 2 | 5 | 6);
        if signaling || ua.is_snan() || ub.is_snan() {
            f |= INVALID;
        }
        None
    } else {
        Some(sf::compare(&ua, &ub))
    };
    let hit = match pred & 7 {
        0 => ord == Some(Equal),
        1 => ord == Some(Less),
        2 => matches!(ord, Some(Less | Equal)),
        3 => ord.is_none(),
        4 => ord != Some(Equal),
        5 => ord != Some(Less),
        6 => !matches!(ord, Some(Less | Equal)),
        _ => ord.is_some(),
    };
    let ones = if ff == Ff::S { 0xffff_ffff } else { u64::MAX };
    (if hit { ones } else { 0 }, f)
}

/// Float → `i32`/`i64` conversion lane (`CVT*2SI`, `CVT*2DQ`, `CVT*2PI`):
/// rounding per `MXCSR` or truncating; NaN/out-of-range → the integer
/// indefinite with `IE`.
fn to_int_lane(ff: Ff, a: u64, bits: u32, mx: Mx, truncate: bool) -> (u64, u32) {
    let (u, _) = mx.input(ff.unpack(a), ff.fmt());
    let indef = 1u64 << (bits - 1);
    if u.is_nan() || u.class == Class::Inf {
        return (indef, INVALID);
    }
    let mode = if truncate { Round::Zero } else { mx.mode };
    match sf::to_int(u, bits, mode) {
        Some((v, f, _)) => {
            let m = if bits == 64 {
                u64::MAX
            } else {
                (1u64 << bits) - 1
            };
            ((v as u64) & m, f)
        }
        None => (indef, INVALID),
    }
}

/// Integer → float lane (`CVTSI2S*`, `CVTDQ2P*`, `CVTPI2P*`).
fn from_int_lane(ff: Ff, v: i64, mx: Mx) -> (u64, u32) {
    let r = sf::round_fp(sf::from_int(v), ff.fmt(), mx.mode);
    (ff.pack(&r.v), r.flags)
}

/// Float format conversion lane (`CVTSS2SD`/`CVTSD2SS`/`CVTPS2PD`/…).
fn cvt_lane(from: Ff, to: Ff, a: u64, mx: Mx) -> (u64, u32) {
    let (u, d) = mx.input(from.unpack(a), from.fmt());
    if u.is_nan() {
        let f = if u.is_snan() { INVALID } else { 0 };
        return (to.pack(&u.quieted()), f | d);
    }
    let (v, f) = mx.output(sf::round_fp(u, to.fmt(), mx.mode));
    (to.pack(&v), f | d)
}

/// The `RCPSS`/`RSQRTSS` estimate of an `f32` (no exceptions; denormal
/// inputs and outputs flush to zero).
fn rcp_estimate(a: u32, rsqrt: bool) -> u32 {
    let sign = a & 0x8000_0000;
    let e = (a >> 23) & 0xff;
    let m = a & 0x7f_ffff;
    if e == 0xff {
        if m != 0 {
            return a | 0x40_0000; // NaN: quieted
        }
        // ±∞: 1/∞ = ±0; 1/√+∞ = +0; 1/√-∞ = indefinite.
        return if rsqrt && sign != 0 {
            0xffc0_0000
        } else {
            sign
        };
    }
    if e == 0 {
        // ±0 or denormal (read as zero): ±∞.
        return sign | 0x7f80_0000;
    }
    if rsqrt {
        if sign != 0 {
            return 0xffc0_0000;
        }
        let odd = e & 1; // e odd ↔ unbiased exponent even
        let t = u32::from(RSQRT_TABLE[(odd as usize) * 1024 + (m >> 13) as usize]);
        // [1,2) (odd e): out exp 126 - (e-127)/2; [0.5,1): 127 - (e-126)/2.
        let oe = if odd == 1 {
            126 - (e as i32 - 127) / 2
        } else {
            127 - (e as i32 - 126) / 2
        };
        ((oe as u32) << 23) | (t << 11)
    } else {
        let t = u32::from(RCP_TABLE[(m >> 12) as usize]);
        let oe = 253 - e as i32;
        if oe <= 0 {
            return sign; // denormal result flushes to zero
        }
        sign | ((oe as u32) << 23) | (t << 11)
    }
}

impl X86Interp {
    // ---- register files ------------------------------------------------------------

    /// MMX register `i` (the low 64 bits of physical x87 register `R_i`).
    fn mm_get(&self, i: usize) -> u64 {
        self.st[i & 7].0 as u64
    }

    /// Write MMX register `i`: the aliased x87 register's sign/exponent field
    /// becomes all ones.
    fn mm_set(&mut self, i: usize, v: u64) {
        self.st[i & 7] = F80((0xffffu128 << 64) | u128::from(v));
    }

    /// The x87→MMX transition every MMX instruction performs: `TOP = 0`, all
    /// tags valid. A pending unmasked x87 exception raises `#MF` first.
    fn mmx_enter(&mut self) -> Result<(), Step> {
        if self.fpu_pending() {
            return Err(Step::Trap(Trap::X87));
        }
        self.fpu_top = 0;
        self.fpu_tag = 0xff;
        Ok(())
    }

    /// The `MXCSR` environment for a floating-point op.
    fn mx(&self) -> Mx {
        Mx::from_mxcsr(self.mxcsr)
    }

    /// Accumulate SSE exception `flags`; `Err(#XM)` if any is unmasked (the
    /// caller then leaves the destination unwritten).
    fn sse_flags(&mut self, flags: u32) -> Result<(), Step> {
        let f = flags & 0x3f;
        self.mxcsr |= f;
        if f & !(self.mxcsr >> 7) & 0x3f != 0 {
            Err(Step::Trap(Trap::Simd))
        } else {
            Ok(())
        }
    }

    // ---- operand access ------------------------------------------------------------

    /// Read `n` bytes (≤ 16) of a memory operand, `#GP` if `align` and the
    /// address isn't `align`-aligned.
    fn mem_read(mem: &GuestMemory, addr: u64, n: usize, align: u64) -> Result<u128, Step> {
        if align > 1 && addr & (align - 1) != 0 {
            return Err(Step::Trap(Trap::Protection));
        }
        let mut b = [0u8; 16];
        mem.read(addr, &mut b[..n]).map_err(|_| rd_fault(addr))?;
        Ok(u128::from_le_bytes(b))
    }

    fn mem_write(
        &mut self,
        mem: &mut GuestMemory,
        addr: u64,
        v: u128,
        n: usize,
        align: u64,
    ) -> Result<(), Step> {
        if align > 1 && addr & (align - 1) != 0 {
            return Err(Step::Trap(Trap::Protection));
        }
        self.store(mem, addr, &v.to_le_bytes()[..n])
    }

    /// An `xmm/m128` source (aligned unless `unaligned`), or for scalar forms
    /// `xmm/m32`/`xmm/m64` (`n` bytes; a register source reads its full value).
    fn xsrc(
        &self,
        mem: &GuestMemory,
        m: &ModRm,
        end: u64,
        n: usize,
        align: bool,
    ) -> Result<u128, Step> {
        match m.kind {
            RmKind::Reg(r) => Ok(self.xmm[r]),
            _ => {
                let a = self.lin(self.ea_of(m.kind, end).unwrap_or(0));
                let al = if align && n == 16 { 16 } else { 1 };
                Self::mem_read(mem, a, n, al)
            }
        }
    }

    /// An `mm/m64` source.
    fn msrc(&self, mem: &GuestMemory, m: &ModRm, end: u64) -> Result<u64, Step> {
        match m.kind {
            RmKind::Reg(r) => Ok(self.mm_get(r)),
            _ => {
                let a = self.lin(self.ea_of(m.kind, end).unwrap_or(0));
                Self::mem_read(mem, a, 8, 1).map(|v| v as u64)
            }
        }
    }

    /// Decode ModRM and an optional trailing `imm8`.
    fn modrm_imm(&self, pc: u64, p: Pfx, imm: bool) -> Result<(ModRm, u8, u64), Step> {
        let (m, pc2) = self.modrm(pc, p.rex)?;
        if imm {
            let (i, end) = self.fetch8(pc2)?;
            Ok((m, i, end))
        } else {
            Ok((m, 0, pc2))
        }
    }

    // ---- dispatch ------------------------------------------------------------------

    /// The SIMD opcodes of the `0F` map (`op` = second opcode byte; the ModRM
    /// starts at `pc`).
    #[allow(clippy::too_many_lines)]
    pub(super) fn exec_simd(&mut self, mem: &mut GuestMemory, pc: u64, p: Pfx, op: u8) -> Step {
        let mp = mp(p);
        if let Some(s) = self.exec_sse3(mem, pc, p, op) {
            return s;
        }
        // ---- the packed-integer ops: MMX (no prefix) and SSE2 (66) forms ----
        let int_form = matches!(op, 0x60..=0x6B | 0x74..=0x76)
            || (op >= 0xD1 && !matches!(op, 0xD6 | 0xD7 | 0xE6 | 0xE7 | 0xF7 | 0xFF))
            || (mp == Mp::P66 && matches!(op, 0x6C | 0x6D));
        if int_form && mp == Mp::None && int_op(op, 0, 0, 8).is_some() {
            let (m, end) = fetch!(self.modrm(pc, p.rex));
            let b = fetch!(self.msrc(mem, &m, end));
            fetch!(self.mmx_enter());
            let a = self.mm_get(m.reg);
            let r = int_op(op, u128::from(a), u128::from(b), 8).unwrap_or(0);
            self.mm_set(m.reg, r as u64);
            return self.next(end);
        }
        if int_form && mp == Mp::P66 && int_op(op, 0, 0, 16).is_some() {
            let (m, end) = fetch!(self.modrm(pc, p.rex));
            let b = fetch!(self.xsrc(mem, &m, end, 16, true));
            let a = self.xmm[m.reg];
            self.xmm[m.reg] = int_op(op, a, b, 16).unwrap_or(0);
            return self.next(end);
        }
        match (op, mp) {
            // ---- moves ----
            (0x10 | 0x11, _) => self.sse_mov_10(mem, pc, p, mp, op == 0x11),
            (0x12 | 0x13 | 0x16 | 0x17, _) => self.sse_mov_half(mem, pc, p, mp, op),
            (0x14 | 0x15, Mp::None | Mp::P66) => {
                let w = if mp == Mp::None { 4 } else { 8 };
                self.xmm_binop(mem, pc, p, true, |a, b| unpack(a, b, w, 16, op == 0x15))
            }
            (0x28 | 0x29, Mp::None | Mp::P66) | (0x6F | 0x7F, Mp::P66 | Mp::F3) => {
                // MOVAPS/MOVAPD/MOVDQA (aligned) and MOVDQU.
                let aligned = mp != Mp::F3;
                self.sse_mov128(mem, pc, p, matches!(op, 0x29 | 0x7F), aligned)
            }
            (0x2B, Mp::None | Mp::P66) | (0xE7, Mp::P66) => {
                // MOVNTPS/MOVNTPD/MOVNTDQ m128, xmm (memory only, aligned).
                let (m, end) = fetch!(self.modrm(pc, p.rex));
                let a = fetch!(self.mem_only(m.kind, end));
                fetch!(self.mem_write(mem, a, self.xmm[m.reg], 16, 16));
                self.next(end)
            }
            (0xE7, Mp::None) => {
                // MOVNTQ m64, mm.
                let (m, end) = fetch!(self.modrm(pc, p.rex));
                let a = fetch!(self.mem_only(m.kind, end));
                fetch!(self.mmx_enter());
                fetch!(self.mem_write(mem, a, u128::from(self.mm_get(m.reg)), 8, 1));
                self.next(end)
            }
            (0x6E, Mp::None | Mp::P66) => {
                // MOVD/MOVQ mm|xmm, r/m32|64 (zero-extended).
                let (m, end) = fetch!(self.modrm(pc, p.rex));
                let w = if p.rex.w { 64 } else { 32 };
                let v = fetch!(self.read_operand(mem, self.op_of(m.kind, end), w));
                if mp == Mp::None {
                    fetch!(self.mmx_enter());
                    self.mm_set(m.reg, v);
                } else {
                    self.xmm[m.reg] = u128::from(v);
                }
                self.next(end)
            }
            (0x7E, Mp::None | Mp::P66) => {
                // MOVD/MOVQ r/m32|64, mm|xmm.
                let (m, end) = fetch!(self.modrm(pc, p.rex));
                let w = if p.rex.w { 64 } else { 32 };
                let v = if mp == Mp::None {
                    fetch!(self.mmx_enter());
                    self.mm_get(m.reg)
                } else {
                    self.xmm[m.reg] as u64
                };
                fetch!(self.write_operand(mem, self.op_of(m.kind, end), v, w));
                self.next(end)
            }
            (0x7E, Mp::F3) => {
                // MOVQ xmm, xmm/m64 (zero the upper half).
                let (m, end) = fetch!(self.modrm(pc, p.rex));
                let v = fetch!(self.xsrc(mem, &m, end, 8, false));
                self.xmm[m.reg] = v & u128::from(u64::MAX);
                self.next(end)
            }
            (0xD6, Mp::P66) => {
                // MOVQ xmm/m64, xmm (a register destination's upper half zeroed).
                let (m, end) = fetch!(self.modrm(pc, p.rex));
                let v = self.xmm[m.reg] & u128::from(u64::MAX);
                match m.kind {
                    RmKind::Reg(r) => self.xmm[r] = v,
                    _ => {
                        let a = self.lin(self.ea_of(m.kind, end).unwrap_or(0));
                        fetch!(self.mem_write(mem, a, v, 8, 1));
                    }
                }
                self.next(end)
            }
            (0xD6, Mp::F3 | Mp::F2) => {
                // MOVQ2DQ xmm, mm / MOVDQ2Q mm, xmm (register forms only).
                let (m, end) = fetch!(self.modrm(pc, p.rex));
                let RmKind::Reg(r) = m.kind else {
                    return Step::Illegal;
                };
                fetch!(self.mmx_enter());
                if mp == Mp::F3 {
                    self.xmm[m.reg] = u128::from(self.mm_get(r));
                } else {
                    let v = self.xmm[r] as u64;
                    self.mm_set(m.reg, v);
                }
                self.next(end)
            }
            (0x6F | 0x7F, Mp::None) => {
                // MOVQ mm, mm/m64 / MOVQ mm/m64, mm.
                let (m, end) = fetch!(self.modrm(pc, p.rex));
                if op == 0x6F {
                    let v = fetch!(self.msrc(mem, &m, end));
                    fetch!(self.mmx_enter());
                    self.mm_set(m.reg, v);
                } else {
                    fetch!(self.mmx_enter());
                    let v = self.mm_get(m.reg);
                    match m.kind {
                        RmKind::Reg(r) => self.mm_set(r, v),
                        _ => {
                            let a = self.lin(self.ea_of(m.kind, end).unwrap_or(0));
                            fetch!(self.mem_write(mem, a, u128::from(v), 8, 1));
                        }
                    }
                }
                self.next(end)
            }
            (0x77, Mp::None | Mp::P66) => {
                // EMMS: all tags empty.
                if self.fpu_pending() {
                    return Step::Trap(Trap::X87);
                }
                self.fpu_tag = 0;
                self.next(pc)
            }
            (0x50, Mp::None | Mp::P66) => {
                // MOVMSKPS/MOVMSKPD r32, xmm.
                let (m, end) = fetch!(self.modrm(pc, p.rex));
                let RmKind::Reg(r) = m.kind else {
                    return Step::Illegal;
                };
                let v = self.xmm[r];
                let bits = if mp == Mp::None {
                    (0..4).fold(0u64, |acc, i| {
                        acc | (((v >> (32 * i + 31)) as u64 & 1) << i)
                    })
                } else {
                    ((v >> 63) as u64 & 1) | (((v >> 127) as u64 & 1) << 1)
                };
                self.gpr[m.reg] = bits;
                self.next(end)
            }
            (0xD7, Mp::None | Mp::P66) => {
                // PMOVMSKB r32, mm|xmm.
                let (m, end) = fetch!(self.modrm(pc, p.rex));
                let RmKind::Reg(r) = m.kind else {
                    return Step::Illegal;
                };
                let v = if mp == Mp::None {
                    fetch!(self.mmx_enter());
                    movmskb(u128::from(self.mm_get(r)), 8)
                } else {
                    movmskb(self.xmm[r], 16)
                };
                self.gpr[m.reg] = v;
                self.next(end)
            }
            (0xF7, Mp::None | Mp::P66) => self.maskmov(mem, pc, p, mp == Mp::P66),
            (0x70, _) => self.pshuf(mem, pc, p, mp),
            (0x71..=0x73, Mp::None | Mp::P66) => self.shift_imm(pc, p, mp == Mp::P66, op),
            (0xC4, Mp::None | Mp::P66) => {
                // PINSRW mm|xmm, r32/m16, imm8.
                let (m, imm, end) = fetch!(self.modrm_imm(pc, p, true));
                let v = fetch!(self.read_operand(mem, self.op_of(m.kind, end), 16));
                if mp == Mp::None {
                    fetch!(self.mmx_enter());
                    let sh = 16 * u32::from(imm & 3);
                    let old = self.mm_get(m.reg);
                    self.mm_set(m.reg, (old & !(0xffff << sh)) | (v << sh));
                } else {
                    let sh = 16 * u32::from(imm & 7);
                    let x = &mut self.xmm[m.reg];
                    *x = (*x & !(0xffffu128 << sh)) | (u128::from(v) << sh);
                }
                self.next(end)
            }
            (0xC5, Mp::None | Mp::P66) => {
                // PEXTRW r32, mm|xmm, imm8 (register source only).
                let (m, imm, end) = fetch!(self.modrm_imm(pc, p, true));
                let RmKind::Reg(r) = m.kind else {
                    return Step::Illegal;
                };
                let v = if mp == Mp::None {
                    fetch!(self.mmx_enter());
                    (self.mm_get(r) >> (16 * u32::from(imm & 3))) & 0xffff
                } else {
                    (self.xmm[r] >> (16 * u32::from(imm & 7))) as u64 & 0xffff
                };
                self.gpr[m.reg] = v;
                self.next(end)
            }
            (0xC6, Mp::None | Mp::P66) => {
                // SHUFPS / SHUFPD.
                let (m, imm, end) = fetch!(self.modrm_imm(pc, p, true));
                let b = fetch!(self.xsrc(mem, &m, end, 16, true));
                let a = self.xmm[m.reg];
                let r = if mp == Mp::None {
                    let s = |v: u128, k: u8| (v >> (32 * u32::from(k & 3))) as u32;
                    u128::from(s(a, imm))
                        | (u128::from(s(a, imm >> 2)) << 32)
                        | (u128::from(s(b, imm >> 4)) << 64)
                        | (u128::from(s(b, imm >> 6)) << 96)
                } else {
                    let s = |v: u128, k: u8| (v >> (64 * u32::from(k & 1))) as u64;
                    u128::from(s(a, imm)) | (u128::from(s(b, imm >> 1)) << 64)
                };
                self.xmm[m.reg] = r;
                self.next(end)
            }
            // ---- bitwise float ops ----
            (0x54..=0x57, Mp::None | Mp::P66) => {
                self.xmm_binop(mem, pc, p, true, |a, b| match op {
                    0x54 => a & b,
                    0x55 => !a & b,
                    0x56 => a | b,
                    _ => a ^ b,
                })
            }
            // ---- float arithmetic ----
            (0x51 | 0x58 | 0x59 | 0x5C..=0x5F, _) => self.sse_arith(mem, pc, p, mp, op),
            (0x52 | 0x53, Mp::None | Mp::F3) => {
                // RSQRTPS/RSQRTSS, RCPPS/RCPSS.
                let scalar = mp == Mp::F3;
                let (m, end) = fetch!(self.modrm(pc, p.rex));
                let b = fetch!(self.xsrc(mem, &m, end, if scalar { 4 } else { 16 }, true));
                let mut r = self.xmm[m.reg];
                for i in 0..if scalar { 1 } else { 4 } {
                    let e = rcp_estimate(lane(b, 4, i) as u32, op == 0x52);
                    r = (r & !(0xffff_ffffu128 << (32 * i))) | (u128::from(e) << (32 * i));
                }
                self.xmm[m.reg] = r;
                self.next(end)
            }
            (0xC2, _) => self.sse_cmp(mem, pc, p, mp),
            (0x2E | 0x2F, Mp::None | Mp::P66) => {
                self.sse_comis(mem, pc, p, mp == Mp::P66, op == 0x2F)
            }
            // ---- conversions ----
            (0x2A, _) => self.cvt_2a(mem, pc, p, mp),
            (0x2C | 0x2D, _) => self.cvt_2c(mem, pc, p, mp, op == 0x2C),
            (0x5A, _) => self.cvt_5a(mem, pc, p, mp),
            (0x5B, Mp::None | Mp::P66 | Mp::F3) => self.cvt_5b(mem, pc, p, mp),
            (0xE6, Mp::P66 | Mp::F3 | Mp::F2) => self.cvt_e6(mem, pc, p, mp),
            _ => Step::Illegal,
        }
    }

    // ---- moves -----------------------------------------------------------------------

    /// `xmm = f(xmm, xmm/m128)` for a bitwise/shuffle op (`aligned` memory).
    fn xmm_binop(
        &mut self,
        mem: &GuestMemory,
        pc: u64,
        p: Pfx,
        aligned: bool,
        f: impl Fn(u128, u128) -> u128,
    ) -> Step {
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        let b = fetch!(self.xsrc(mem, &m, end, 16, aligned));
        self.xmm[m.reg] = f(self.xmm[m.reg], b);
        self.next(end)
    }

    /// `MOVAPS`/`MOVAPD`/`MOVDQA` (`aligned`) and `MOVDQU`, load or store.
    fn sse_mov128(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        p: Pfx,
        store: bool,
        aligned: bool,
    ) -> Step {
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        let al = if aligned { 16 } else { 1 };
        match (m.kind, store) {
            (RmKind::Reg(r), false) => self.xmm[m.reg] = self.xmm[r],
            (RmKind::Reg(r), true) => self.xmm[r] = self.xmm[m.reg],
            (_, false) => {
                let a = self.lin(self.ea_of(m.kind, end).unwrap_or(0));
                self.xmm[m.reg] = fetch!(Self::mem_read(mem, a, 16, al));
            }
            (_, true) => {
                let a = self.lin(self.ea_of(m.kind, end).unwrap_or(0));
                fetch!(self.mem_write(mem, a, self.xmm[m.reg], 16, al));
            }
        }
        self.next(end)
    }

    /// `0F 10/11`: `MOVUPS`/`MOVUPD` (unaligned 128-bit) and `MOVSS`/`MOVSD`
    /// (a register-to-register scalar move keeps the destination's upper
    /// lanes; a load from memory zeroes them).
    fn sse_mov_10(&mut self, mem: &mut GuestMemory, pc: u64, p: Pfx, mp: Mp, store: bool) -> Step {
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        let n = match mp {
            Mp::F3 => 4,
            Mp::F2 => 8,
            _ => 16,
        };
        let lmask: u128 = if n == 16 {
            u128::MAX
        } else {
            (1u128 << (8 * n)) - 1
        };
        match (m.kind, store) {
            (RmKind::Reg(r), _) => {
                let (d, s) = if store { (r, m.reg) } else { (m.reg, r) };
                self.xmm[d] = (self.xmm[d] & !lmask) | (self.xmm[s] & lmask);
            }
            (_, false) => {
                let a = self.lin(self.ea_of(m.kind, end).unwrap_or(0));
                self.xmm[m.reg] = fetch!(Self::mem_read(mem, a, n, 1));
            }
            (_, true) => {
                let a = self.lin(self.ea_of(m.kind, end).unwrap_or(0));
                fetch!(self.mem_write(mem, a, self.xmm[m.reg], n, 1));
            }
        }
        self.next(end)
    }

    /// `0F 12/13/16/17`: the 64-bit half moves — `MOVLPS`/`MOVLPD`,
    /// `MOVHPS`/`MOVHPD` (memory), `MOVHLPS`/`MOVLHPS` (register), and the
    /// SSE3 duplicating loads `MOVSLDUP`/`MOVSHDUP`/`MOVDDUP`.
    fn sse_mov_half(&mut self, mem: &mut GuestMemory, pc: u64, p: Pfx, mp: Mp, op: u8) -> Step {
        const LO: u128 = u64::MAX as u128;
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        let high = op >= 0x16;
        let addr = self.ea_of(m.kind, end).map(|a| self.lin(a));
        match (mp, op & 1 == 1, addr) {
            (Mp::None | Mp::P66, false, Some(a)) => {
                // MOVLPS/MOVLPD/MOVHPS/MOVHPD xmm, m64.
                let v = fetch!(Self::mem_read(mem, a, 8, 1));
                let x = &mut self.xmm[m.reg];
                *x = if high {
                    (*x & LO) | (v << 64)
                } else {
                    (*x & !LO) | v
                };
            }
            (Mp::None, false, None) => {
                // MOVHLPS (12) / MOVLHPS (16).
                let RmKind::Reg(r) = m.kind else {
                    return Step::Illegal;
                };
                let s = self.xmm[r];
                let x = &mut self.xmm[m.reg];
                *x = if high {
                    (*x & LO) | (s << 64)
                } else {
                    (*x & !LO) | (s >> 64)
                };
            }
            (Mp::None | Mp::P66, true, Some(a)) => {
                // MOVLPS/MOVLPD/MOVHPS/MOVHPD m64, xmm.
                let x = self.xmm[m.reg];
                let v = if high { x >> 64 } else { x & LO };
                fetch!(self.mem_write(mem, a, v, 8, 1));
            }
            (Mp::F3, false, _) if SSE3 => {
                // MOVSLDUP (12) / MOVSHDUP (16).
                let v = fetch!(self.xsrc(mem, &m, end, 16, true));
                let pick = |i: u32| (v >> (32 * i)) & 0xffff_ffff;
                let (s0, s1) = if high { (1, 3) } else { (0, 2) };
                self.xmm[m.reg] = pick(s0) | (pick(s0) << 32) | (pick(s1) << 64) | (pick(s1) << 96);
            }
            (Mp::F2, false, _) if SSE3 && !high => {
                // MOVDDUP xmm, xmm/m64.
                let v = fetch!(self.xsrc(mem, &m, end, 8, false)) & LO;
                self.xmm[m.reg] = v | (v << 64);
            }
            _ => return Step::Illegal,
        }
        self.next(end)
    }

    /// `MASKMOVQ mm, mm` / `MASKMOVDQU xmm, xmm`: store the bytes of the
    /// first operand whose mask byte (second operand) has its top bit set, to
    /// `[seg:rDI]`.
    fn maskmov(&mut self, mem: &mut GuestMemory, pc: u64, p: Pfx, xmm: bool) -> Step {
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        let RmKind::Reg(r) = m.kind else {
            return Step::Illegal;
        };
        let (data, mask, n) = if xmm {
            (self.xmm[m.reg], self.xmm[r], 16)
        } else {
            fetch!(self.mmx_enter());
            (
                u128::from(self.mm_get(m.reg)),
                u128::from(self.mm_get(r)),
                8,
            )
        };
        let base = self.lin(self.sreg(RDI));
        for i in 0..n {
            if (mask >> (8 * i + 7)) & 1 != 0 {
                let a = base.wrapping_add(i as u64);
                if let Err(e) = self.store(mem, a, &[(data >> (8 * i)) as u8]) {
                    return e;
                }
            }
        }
        self.next(end)
    }

    /// `0F 70`: `PSHUFW mm` (none), `PSHUFD` (66), `PSHUFHW` (F3), `PSHUFLW` (F2).
    fn pshuf(&mut self, mem: &GuestMemory, pc: u64, p: Pfx, mp: Mp) -> Step {
        let (m, imm, end) = fetch!(self.modrm_imm(pc, p, true));
        let sel = |v: u128, w: usize, base: usize, k: u8, i: usize| -> u128 {
            u128::from(lane(v, w, base + usize::from((k >> (2 * i)) & 3)))
        };
        match mp {
            Mp::None => {
                let v = fetch!(self.msrc(mem, &m, end));
                fetch!(self.mmx_enter());
                let v = u128::from(v);
                let r = (0..4).fold(0u128, |acc, i| acc | (sel(v, 2, 0, imm, i) << (16 * i)));
                self.mm_set(m.reg, r as u64);
            }
            Mp::P66 => {
                let v = fetch!(self.xsrc(mem, &m, end, 16, true));
                let r = (0..4).fold(0u128, |acc, i| acc | (sel(v, 4, 0, imm, i) << (32 * i)));
                self.xmm[m.reg] = r;
            }
            Mp::F3 | Mp::F2 => {
                let v = fetch!(self.xsrc(mem, &m, end, 16, true));
                let (base, keep_hi) = if mp == Mp::F3 { (4, false) } else { (0, true) };
                let words =
                    (0..4).fold(0u128, |acc, i| acc | (sel(v, 2, base, imm, i) << (16 * i)));
                self.xmm[m.reg] = if keep_hi {
                    (v & !u128::from(u64::MAX)) | words
                } else {
                    (v & u128::from(u64::MAX)) | (words << 64)
                };
            }
        }
        self.next(end)
    }

    /// Groups 12/13/14 (`0F 71/72/73`): shift `mm`/`xmm` lanes by `imm8`, and
    /// the 66-only whole-register byte shifts `PSRLDQ`/`PSLLDQ`.
    fn shift_imm(&mut self, pc: u64, p: Pfx, xmm: bool, op: u8) -> Step {
        let (m, imm, end) = fetch!(self.modrm_imm(pc, p, true));
        let RmKind::Reg(r) = m.kind else {
            return Step::Illegal;
        };
        let w = match op {
            0x71 => 2,
            0x72 => 4,
            _ => 8,
        };
        let kind = match (op, m.ext()) {
            (_, 2) => 0,
            (0x71 | 0x72, 4) => 2,
            (_, 6) => 1,
            (0x73, 3) if xmm => 3, // PSRLDQ
            (0x73, 7) if xmm => 4, // PSLLDQ
            _ => return Step::Illegal,
        };
        let count = u64::from(imm);
        if xmm {
            let v = self.xmm[r];
            self.xmm[r] = match kind {
                3 => {
                    if imm >= 16 {
                        0
                    } else {
                        v >> (8 * u32::from(imm))
                    }
                }
                4 => {
                    if imm >= 16 {
                        0
                    } else {
                        v << (8 * u32::from(imm))
                    }
                }
                k => shift(v, w, 16, count, k),
            };
        } else {
            fetch!(self.mmx_enter());
            let v = u128::from(self.mm_get(r));
            self.mm_set(r, shift(v, w, 8, count, kind) as u64);
        }
        self.next(end)
    }

    // ---- floating point ---------------------------------------------------------------

    /// `0F 51/58/59/5C/5D/5E/5F`: `SQRT`/`ADD`/`MUL`/`SUB`/`MIN`/`DIV`/`MAX`
    /// in all four forms (`PS`/`PD`/`SS`/`SD`).
    fn sse_arith(&mut self, mem: &GuestMemory, pc: u64, p: Pfx, mp: Mp, op: u8) -> Step {
        let (ff, lanes, n) = match mp {
            Mp::None => (Ff::S, 4, 16),
            Mp::P66 => (Ff::D, 2, 16),
            Mp::F3 => (Ff::S, 1, 4),
            Mp::F2 => (Ff::D, 1, 8),
        };
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        let b = fetch!(self.xsrc(mem, &m, end, n, true));
        let a = self.xmm[m.reg];
        let mx = self.mx();
        let w = ff.bytes();
        let mut out = a;
        let mut flags = 0;
        for i in 0..lanes {
            let (x, y) = (lane(a, w, i), lane(b, w, i));
            let (r, f) = match op {
                0x51 => sqrt_lane(ff, y, mx),
                0x58 => arith_lane(Op::Add, ff, x, y, mx),
                0x59 => arith_lane(Op::Mul, ff, x, y, mx),
                0x5C => arith_lane(Op::Sub, ff, x, y, mx),
                0x5E => arith_lane(Op::Div, ff, x, y, mx),
                0x5D => minmax_lane(ff, x, y, mx, false),
                _ => minmax_lane(ff, x, y, mx, true),
            };
            let sh = 8 * w * i;
            let lm: u128 = if w == 8 {
                u128::from(u64::MAX)
            } else {
                0xffff_ffff
            };
            out = (out & !(lm << sh)) | (u128::from(r) << sh);
            flags |= f;
        }
        fetch!(self.sse_flags(flags));
        self.xmm[m.reg] = out;
        self.next(end)
    }

    /// `CMPPS/PD/SS/SD xmm, xmm/m, imm8` (predicate in `imm8[2:0]`).
    fn sse_cmp(&mut self, mem: &GuestMemory, pc: u64, p: Pfx, mp: Mp) -> Step {
        let (ff, lanes, n) = match mp {
            Mp::None => (Ff::S, 4, 16),
            Mp::P66 => (Ff::D, 2, 16),
            Mp::F3 => (Ff::S, 1, 4),
            Mp::F2 => (Ff::D, 1, 8),
        };
        let (m, imm, end) = fetch!(self.modrm_imm(pc, p, true));
        let b = fetch!(self.xsrc(mem, &m, end, n, true));
        let a = self.xmm[m.reg];
        let mx = self.mx();
        let w = ff.bytes();
        let mut out = a;
        let mut flags = 0;
        for i in 0..lanes {
            let (r, f) = cmp_lane(ff, lane(a, w, i), lane(b, w, i), mx, imm);
            let sh = 8 * w * i;
            let lm: u128 = if w == 8 {
                u128::from(u64::MAX)
            } else {
                0xffff_ffff
            };
            out = (out & !(lm << sh)) | (u128::from(r) << sh);
            flags |= f;
        }
        fetch!(self.sse_flags(flags));
        self.xmm[m.reg] = out;
        self.next(end)
    }

    /// `UCOMISS`/`COMISS`/`UCOMISD`/`COMISD`: `ZF`/`PF`/`CF` = unordered
    /// `111`, greater `000`, less `001`, equal `100`; `OF`/`SF`/`AF` cleared.
    fn sse_comis(
        &mut self,
        mem: &GuestMemory,
        pc: u64,
        p: Pfx,
        dbl: bool,
        signaling: bool,
    ) -> Step {
        let ff = if dbl { Ff::D } else { Ff::S };
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        let b = fetch!(self.xsrc(mem, &m, end, ff.bytes(), false));
        let mx = self.mx();
        let (ua, da) = mx.input(ff.unpack(lane(self.xmm[m.reg], ff.bytes(), 0)), ff.fmt());
        let (ub, db) = mx.input(ff.unpack(lane(b, ff.bytes(), 0)), ff.fmt());
        let mut f = da | db;
        let ord = if ua.is_nan() || ub.is_nan() {
            if signaling || ua.is_snan() || ub.is_snan() {
                f |= INVALID;
            }
            None
        } else {
            Some(sf::compare(&ua, &ub))
        };
        fetch!(self.sse_flags(f));
        use core::cmp::Ordering;
        self.flags = Flags {
            cf: matches!(ord, None | Some(Ordering::Less)),
            pf: ord.is_none(),
            af: false,
            zf: matches!(ord, None | Some(Ordering::Equal)),
            sf: false,
            of: false,
        };
        self.next(end)
    }

    /// `0F 2A`: `CVTPI2PS` (none), `CVTPI2PD` (66), `CVTSI2SS` (F3), `CVTSI2SD` (F2).
    fn cvt_2a(&mut self, mem: &GuestMemory, pc: u64, p: Pfx, mp: Mp) -> Step {
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        let mx = self.mx();
        match mp {
            Mp::None | Mp::P66 => {
                let v = fetch!(self.msrc(mem, &m, end));
                if let RmKind::Reg(_) = m.kind {
                    fetch!(self.mmx_enter());
                }
                let ints = [v as u32 as i32, (v >> 32) as u32 as i32];
                if mp == Mp::None {
                    let mut out = self.xmm[m.reg] & !u128::from(u64::MAX);
                    let mut flags = 0;
                    for (i, &x) in ints.iter().enumerate() {
                        let (r, f) = from_int_lane(Ff::S, i64::from(x), mx);
                        out |= u128::from(r) << (32 * i);
                        flags |= f;
                    }
                    fetch!(self.sse_flags(flags));
                    self.xmm[m.reg] = out;
                } else {
                    let lo = from_int_lane(Ff::D, i64::from(ints[0]), mx).0;
                    let hi = from_int_lane(Ff::D, i64::from(ints[1]), mx).0;
                    self.xmm[m.reg] = u128::from(lo) | (u128::from(hi) << 64);
                }
            }
            Mp::F3 | Mp::F2 => {
                let w = if p.rex.w { 64 } else { 32 };
                let raw = fetch!(self.read_operand(mem, self.op_of(m.kind, end), w));
                let v = if w == 64 {
                    raw as i64
                } else {
                    i64::from(raw as u32 as i32)
                };
                let ff = if mp == Mp::F3 { Ff::S } else { Ff::D };
                let (r, f) = from_int_lane(ff, v, mx);
                fetch!(self.sse_flags(f));
                let lm: u128 = if ff == Ff::S {
                    0xffff_ffff
                } else {
                    u128::from(u64::MAX)
                };
                self.xmm[m.reg] = (self.xmm[m.reg] & !lm) | u128::from(r);
            }
        }
        self.next(end)
    }

    /// `0F 2C/2D`: `CVT[T]PS2PI` (none), `CVT[T]PD2PI` (66), `CVT[T]SS2SI`
    /// (F3), `CVT[T]SD2SI` (F2).
    fn cvt_2c(&mut self, mem: &GuestMemory, pc: u64, p: Pfx, mp: Mp, truncate: bool) -> Step {
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        let mx = self.mx();
        match mp {
            Mp::None | Mp::P66 => {
                let (ff, n) = if mp == Mp::None {
                    (Ff::S, 8)
                } else {
                    (Ff::D, 16)
                };
                let v = fetch!(self.xsrc(mem, &m, end, n, true));
                let w = ff.bytes();
                let (r0, f0) = to_int_lane(ff, lane(v, w, 0), 32, mx, truncate);
                let (r1, f1) = to_int_lane(ff, lane(v, w, 1), 32, mx, truncate);
                fetch!(self.sse_flags(f0 | f1));
                fetch!(self.mmx_enter());
                self.mm_set(m.reg, r0 | (r1 << 32));
            }
            Mp::F3 | Mp::F2 => {
                let ff = if mp == Mp::F3 { Ff::S } else { Ff::D };
                let v = fetch!(self.xsrc(mem, &m, end, ff.bytes(), false));
                let bits = if p.rex.w { 64 } else { 32 };
                let (r, f) = to_int_lane(ff, lane(v, ff.bytes(), 0), bits, mx, truncate);
                fetch!(self.sse_flags(f));
                self.set_reg(m.reg, r, bits);
            }
        }
        self.next(end)
    }

    /// `0F 5A`: `CVTPS2PD` (none), `CVTPD2PS` (66), `CVTSS2SD` (F3), `CVTSD2SS` (F2).
    fn cvt_5a(&mut self, mem: &GuestMemory, pc: u64, p: Pfx, mp: Mp) -> Step {
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        let mx = self.mx();
        let (from, to, lanes, n) = match mp {
            Mp::None => (Ff::S, Ff::D, 2, 8),
            Mp::P66 => (Ff::D, Ff::S, 2, 16),
            Mp::F3 => (Ff::S, Ff::D, 1, 4),
            Mp::F2 => (Ff::D, Ff::S, 1, 8),
        };
        let v = fetch!(self.xsrc(mem, &m, end, n, n == 16));
        let mut out: u128 = match mp {
            Mp::F3 => self.xmm[m.reg] & !u128::from(u64::MAX),
            Mp::F2 => self.xmm[m.reg] & !0xffff_ffffu128,
            _ => 0, // packed: unused lanes (CVTPD2PS's upper half) zeroed
        };
        let mut flags = 0;
        for i in 0..lanes {
            let (r, f) = cvt_lane(from, to, lane(v, from.bytes(), i), mx);
            out |= u128::from(r) << (8 * to.bytes() * i);
            flags |= f;
        }
        fetch!(self.sse_flags(flags));
        self.xmm[m.reg] = out;
        self.next(end)
    }

    /// `0F 5B`: `CVTDQ2PS` (none), `CVTPS2DQ` (66), `CVTTPS2DQ` (F3).
    fn cvt_5b(&mut self, mem: &GuestMemory, pc: u64, p: Pfx, mp: Mp) -> Step {
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        let v = fetch!(self.xsrc(mem, &m, end, 16, true));
        let mx = self.mx();
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
        fetch!(self.sse_flags(flags));
        self.xmm[m.reg] = out;
        self.next(end)
    }

    /// `0F E6`: `CVTTPD2DQ` (66), `CVTDQ2PD` (F3), `CVTPD2DQ` (F2).
    fn cvt_e6(&mut self, mem: &GuestMemory, pc: u64, p: Pfx, mp: Mp) -> Step {
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        let mx = self.mx();
        let (out, flags) = if mp == Mp::F3 {
            let v = fetch!(self.xsrc(mem, &m, end, 8, false));
            let lo = from_int_lane(Ff::D, i64::from(v as u32 as i32), mx).0;
            let hi = from_int_lane(Ff::D, i64::from((v >> 32) as u32 as i32), mx).0;
            (u128::from(lo) | (u128::from(hi) << 64), 0)
        } else {
            let v = fetch!(self.xsrc(mem, &m, end, 16, true));
            let trunc = mp == Mp::P66;
            let (r0, f0) = to_int_lane(Ff::D, lane(v, 8, 0), 32, mx, trunc);
            let (r1, f1) = to_int_lane(Ff::D, lane(v, 8, 1), 32, mx, trunc);
            (u128::from(r0) | (u128::from(r1) << 32), f0 | f1)
        };
        fetch!(self.sse_flags(flags));
        self.xmm[m.reg] = out;
        self.next(end)
    }
}

/// The `RCPSS`/`RCPPS` estimate table: 12-bit mantissas for inputs `1.m`,
/// indexed by the top 11 bits of `m`; the result is `(4096 + T)·2^-13`
/// (exponent field 126). Reproduces the hardware's estimates bit for bit.
const RCP_TABLE: [u16; 2048] = include!("rcp_table.in");

/// The `RSQRTSS`/`RSQRTPS` estimate table: entries `0..1024` for inputs in
/// `[0.5, 1)` (result exponent field 127), `1024..2048` for `[1, 2)` (field
/// 126), indexed by the top 10 bits of the input mantissa.
const RSQRT_TABLE: [u16; 2048] = include!("rsqrt_table.in");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swar_pcmpeqb_and_movmskb() {
        let a = 0x00ff_1234_5678_9abc_def0_0000_ffff_0102u128;
        let b = 0x00fe_1234_0078_9abc_def0_0001_ffff_0102u128;
        let r = pcmpeqb(a, b);
        let slow = map2(a, b, 1, 16, |x, y| if x == y { 0xff } else { 0 });
        assert_eq!(r, slow);
        assert_eq!(movmskb(r, 16), {
            (0..16).fold(0u64, |acc, i| acc | (((r >> (8 * i + 7)) as u64 & 1) << i))
        });
    }

    #[test]
    fn rcp_estimates_match_hardware() {
        assert_eq!(rcp_estimate(0x3f80_0000, false), 0x3f7f_f000); // 1/1
        assert_eq!(rcp_estimate(0x4000_0000, false), 0x3eff_f000); // 1/2
        assert_eq!(rcp_estimate(0x3f80_0000, true), 0x3f7f_f000); // 1/√1
        assert_eq!(rcp_estimate(0x4080_0000, true), 0x3eff_f000); // 1/√4
        assert_eq!(rcp_estimate(0x4000_0000, true), 0x3f34_f800); // 1/√2
        assert_eq!(rcp_estimate(0x3f00_0000, true), 0x3fb4_f800); // 1/√0.5
        assert_eq!(rcp_estimate(0x7f7f_ffff, false), 0); // flushes
        assert_eq!(rcp_estimate(0x0040_0000, false), 0x7f80_0000);
        assert_eq!(rcp_estimate(0xbf80_0000, true), 0xffc0_0000);
    }
}
