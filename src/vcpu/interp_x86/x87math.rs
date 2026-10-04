//! Extended-precision math behind the x87 transcendental instructions
//! (`F2XM1`, `FYL2X`, `FYL2XP1`, `FPATAN`, `FSIN`/`FCOS`/`FSINCOS`/`FPTAN`),
//! the inexact constants (`FLDPI` …) and the exact partial remainder of
//! `FPREM`/`FPREM1`.
//!
//! Results are computed in a 128-bit-significand software format ([`X`]),
//! roughly 60 bits more than the 64-bit x87 significand, and then rounded
//! once under the control word's rounding mode — so they are correctly
//! rounded except in astronomically rare near-halfway cases. Trigonometric
//! arguments are reduced modulo π/2 against a 448-bit `2/π` (Payne–Hanek), so
//! even arguments near multiples of π/2 up to the instructions' 2^63 limit
//! keep full relative accuracy.

use super::F80;
use crate::vcpu::softfloat::{
    self as sf, Class, DENORMAL, DIVZERO, FMT80, Fp, INEXACT, INVALID, Round, UNDERFLOW,
};

// ---- a minimal 256-bit unsigned integer --------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct U256 {
    hi: u128,
    lo: u128,
}

impl U256 {
    const fn new(hi: u128, lo: u128) -> Self {
        Self { hi, lo }
    }

    /// The full product of two `u128`s.
    fn mul(a: u128, b: u128) -> Self {
        let (a1, a0) = (a >> 64, a & u128::from(u64::MAX));
        let (b1, b0) = (b >> 64, b & u128::from(u64::MAX));
        let p00 = a0 * b0;
        let p01 = a0 * b1;
        let p10 = a1 * b0;
        let p11 = a1 * b1;
        let mid = (p00 >> 64) + (p01 & u128::from(u64::MAX)) + (p10 & u128::from(u64::MAX));
        let lo = (p00 & u128::from(u64::MAX)) | (mid << 64);
        let hi = p11 + (p01 >> 64) + (p10 >> 64) + (mid >> 64);
        Self { hi, lo }
    }

    fn add(self, o: Self) -> Self {
        let (lo, c) = self.lo.overflowing_add(o.lo);
        Self {
            hi: self.hi.wrapping_add(o.hi).wrapping_add(u128::from(c)),
            lo,
        }
    }

    fn sub(self, o: Self) -> Self {
        let (lo, b) = self.lo.overflowing_sub(o.lo);
        Self {
            hi: self.hi.wrapping_sub(o.hi).wrapping_sub(u128::from(b)),
            lo,
        }
    }

    fn shl(self, n: u32) -> Self {
        match n {
            0 => self,
            1..=127 => Self {
                hi: (self.hi << n) | (self.lo >> (128 - n)),
                lo: self.lo << n,
            },
            128..=255 => Self {
                hi: self.lo << (n - 128),
                lo: 0,
            },
            _ => Self::new(0, 0),
        }
    }

    fn shr(self, n: u32) -> Self {
        match n {
            0 => self,
            1..=127 => Self {
                hi: self.hi >> n,
                lo: (self.lo >> n) | (self.hi << (128 - n)),
            },
            128..=255 => Self {
                hi: 0,
                lo: self.hi >> (n - 128),
            },
            _ => Self::new(0, 0),
        }
    }

    fn is_zero(self) -> bool {
        self.hi == 0 && self.lo == 0
    }

    fn leading_zeros(self) -> u32 {
        if self.hi == 0 {
            128 + self.lo.leading_zeros()
        } else {
            self.hi.leading_zeros()
        }
    }
}

// ---- the 128-bit working format ------------------------------------------------

/// A working value `(-1)^neg · m · 2^(e - 127)`: `m` normalized (bit 127 set)
/// unless the value is zero (`m == 0`). Operations truncate, so each loses
/// at most one unit in the 128th bit.
#[derive(Clone, Copy, Debug)]
struct X {
    neg: bool,
    m: u128,
    e: i32,
}

impl X {
    const ZERO: X = X {
        neg: false,
        m: 0,
        e: 0,
    };

    /// `m · 2^(e - 127)` from a not-necessarily-normalized 256-bit mantissa
    /// whose binary point sits `point` bits up from bit 0 (value =
    /// `wide · 2^-point · 2^e0`).
    fn from_wide(neg: bool, wide: U256, e_of_bit255: i32) -> X {
        if wide.is_zero() {
            return X::ZERO;
        }
        let lz = wide.leading_zeros();
        let w = wide.shl(lz);
        X {
            neg,
            m: w.hi,
            e: e_of_bit255 - lz as i32,
        }
    }

    fn from_fp(f: &Fp) -> X {
        if f.class != Class::Finite {
            return X::ZERO;
        }
        let m = sf::msb(f.sig);
        X {
            neg: f.sign,
            m: f.sig << (127 - m),
            e: f.exp + m as i32,
        }
    }

    fn from_u128(neg: bool, v: u128, e_of_bit0: i32) -> X {
        if v == 0 {
            return X::ZERO;
        }
        let m = v.ilog2();
        X {
            neg,
            m: v << (127 - m),
            e: e_of_bit0 + m as i32,
        }
    }

    const fn constant(m: u128, e: i32) -> X {
        X { neg: false, m, e }
    }

    fn one() -> X {
        X::constant(1 << 127, 0)
    }

    fn is_zero(self) -> bool {
        self.m == 0
    }

    fn neg(self) -> X {
        X {
            neg: !self.neg,
            ..self
        }
    }

    fn abs(self) -> X {
        X { neg: false, ..self }
    }

    fn scale(self, k: i32) -> X {
        if self.is_zero() {
            self
        } else {
            X {
                e: self.e + k,
                ..self
            }
        }
    }

    fn mul(self, o: X) -> X {
        if self.is_zero() || o.is_zero() {
            return X::ZERO;
        }
        X::from_wide(self.neg ^ o.neg, U256::mul(self.m, o.m), self.e + o.e + 1)
    }

    fn add(self, o: X) -> X {
        if self.is_zero() {
            return o;
        }
        if o.is_zero() {
            return self;
        }
        let (a, b) = if self.e >= o.e { (self, o) } else { (o, self) };
        let d = (a.e - b.e) as u32;
        // Work with a 256-bit window: a at bits 254..127, b shifted down.
        let wa = U256::new(a.m >> 1, a.m << 127);
        let mut wb = U256::new(b.m >> 1, b.m << 127).shr(d);
        if wb.is_zero() {
            // Far below the window: keep it as one unit at the very bottom so
            // the sum still lies strictly on the right side of `a` (a later
            // rounding with "inexact" must see 1 - tiny as below 1).
            wb = U256::new(0, 1);
        }
        if a.neg == b.neg {
            X::from_wide(a.neg, wa.add(wb), a.e + 1)
        } else if wa >= wb {
            X::from_wide(a.neg, wa.sub(wb), a.e + 1)
        } else {
            X::from_wide(b.neg, wb.sub(wa), a.e + 1)
        }
    }

    fn sub(self, o: X) -> X {
        self.add(o.neg())
    }

    /// `self / o` (`o` nonzero).
    fn div(self, o: X) -> X {
        if self.is_zero() {
            return X::ZERO;
        }
        // Restoring division of the two normalized mantissas: 129 quotient
        // bits (the quotient is in (1/2, 2)).
        let d = o.m;
        let mut rem_hi = false; // the 129th remainder bit
        let mut rem = self.m;
        let mut q: u128 = 0;
        let mut extra = false;
        for i in 0..129 {
            let ge = rem_hi || rem >= d;
            if i == 0 {
                extra = ge;
            } else {
                q = (q << 1) | u128::from(ge);
            }
            if ge {
                rem = rem.wrapping_sub(d);
            }
            rem_hi = rem >> 127 != 0;
            rem <<= 1;
        }
        // `extra` is the 2^0 bit; `q` holds 128 fraction bits below it.
        let neg = self.neg ^ o.neg;
        if extra {
            X {
                neg,
                m: (1 << 127) | (q >> 1),
                e: self.e - o.e,
            }
        } else {
            X {
                neg,
                m: q,
                e: self.e - o.e - 1,
            }
        }
    }

    /// Divide by a small positive integer.
    fn div_small(self, d: u32) -> X {
        if self.is_zero() {
            return self;
        }
        let d = u128::from(d);
        let hi = self.m / d;
        let rem = self.m % d;
        let lo = (rem << 64) / d;
        // value = (hi + lo·2^-64) · 2^(e-127)
        let wide = U256::new(hi >> 64, (hi << 64) | lo);
        X::from_wide(self.neg, wide, self.e + 64)
    }

    /// `√self` (non-negative).
    fn sqrt(self) -> X {
        if self.is_zero() {
            return self;
        }
        // value = m·2^(e-127). Make (e - 127) even and take the integer
        // square root of m·2^k as a 256-bit number.
        let (wide, e2) = if (self.e - 127) & 1 == 0 {
            (U256::new(self.m, 0), self.e - 127 - 128)
        } else {
            (U256::new(self.m >> 1, self.m << 127), self.e - 127 - 127)
        };
        // Bitwise restoring square root (128 result bits).
        let mut rem = U256::new(0, 0);
        let mut root: u128 = 0;
        let mut n = wide;
        for _ in 0..128 {
            rem = rem.shl(2).add(U256::new(0, n.hi >> 126));
            n = n.shl(2);
            let trial = U256::new(0, root).shl(2).add(U256::new(0, 1));
            root <<= 1;
            if rem >= trial {
                rem = rem.sub(trial);
                root |= 1;
            }
        }
        // value = root · 2^(e2/2)
        X::from_u128(false, root, e2 / 2)
    }

    /// Round into an x87 value under `mode` (the approximation is inexact).
    fn round(self, mode: Round) -> sf::Rounded {
        if self.is_zero() {
            return sf::round(self.neg, 0, 0, false, FMT80, mode);
        }
        sf::round(self.neg, self.m, self.e - 127, true, FMT80, mode)
    }
}

// Constants (bit 127 = the leading bit; `e` = its exponent).
const PI: X = X::constant(0xc90f_daa2_2168_c234_c4c6_628b_80dc_1cd1, 1);
const LN2: X = X::constant(0xb172_17f7_d1cf_79ab_c9e3_b398_03f2_f6af, -1);
const LOG2E: X = X::constant(0xb8aa_3b29_5c17_f0bb_be87_fed0_691d_3e88, 0);
const LOG2_10: X = X::constant(0xd49a_784b_cd1b_8afe_492b_f6ff_4daf_db4c, 1);
const LOG10_2: X = X::constant(0x9a20_9a84_fbcf_f798_8f89_59ac_0b7c_9178, -2);

/// `FLD1`/`FLDL2T`/`FLDL2E`/`FLDPI`/`FLDLG2`/`FLDLN2`/`FLDZ` (`D9 E8+i`),
/// the inexact ones rounded per `RC`.
pub(super) fn constant(i: u8, mode: Round) -> Option<F80> {
    let c = match i {
        0 => return Some(F80::ONE),
        1 => LOG2_10,
        2 => LOG2E,
        3 => PI,
        4 => LOG10_2,
        5 => LN2,
        6 => return Some(F80::ZERO),
        _ => return None,
    };
    Some(F80::pack(&c.round(mode).v))
}

// ---- elementary functions on X ---------------------------------------------------

/// `e^t - 1` for `|t| < 1`.
fn expm1(t: X) -> X {
    let mut sum = t;
    let mut term = t;
    for n in 2..80u32 {
        term = term.mul(t).div_small(n);
        if term.is_zero() {
            break;
        }
        // A negligible term still nudges the sum (see `X::add`), so the
        // final rounding sees which side of a representable value it is on.
        let done = term.e < sum.e - 132;
        sum = sum.add(term);
        if done {
            break;
        }
    }
    sum
}

/// `2·atanh(z) = ln((1+z)/(1-z))` for small `|z|`.
fn two_atanh(z: X) -> X {
    let z2 = z.mul(z);
    let mut sum = z;
    let mut pow = z;
    for k in 1..400u32 {
        pow = pow.mul(z2);
        let term = pow.div_small(2 * k + 1);
        if term.is_zero() {
            break;
        }
        // A negligible term still nudges the sum (see `X::add`), so the
        // final rounding sees which side of a representable value it is on.
        let done = term.e < sum.e - 132;
        sum = sum.add(term);
        if done {
            break;
        }
    }
    sum.scale(1)
}

/// `ln(x)` for finite `x > 0` given as `x = sig·2^exp`.
fn ln(x: &Fp) -> X {
    ln_x(X::from_fp(x))
}

/// `ln(x)` for a positive working value.
fn ln_x(x: X) -> X {
    // x = m·2^k with m in [√2/2, √2).
    let mut k = x.e;
    let mut m = X { e: 0, ..x };
    // m in [1, 2); fold the top half down.
    if m.m > 0xb504_f333_f9de_6484_597d_89b3_754a_be9f {
        m = m.scale(-1);
        k += 1;
    }
    let one = X::one();
    let z = m.sub(one).div(m.add(one));
    two_atanh(z).add(X::from_u128(k < 0, u128::from(k.unsigned_abs()), 0).mul(LN2))
}

/// `atan(t)` for `0 ≤ t ≤ 1`.
fn atan_unit(t: X) -> X {
    // Halve the angle three times: atan(t) = 2·atan(t / (1 + √(1+t²))).
    let mut t = t;
    let one = X::one();
    for _ in 0..3 {
        t = t.div(one.add(one.add(t.mul(t)).sqrt()));
    }
    let t2 = t.mul(t);
    let mut sum = t;
    let mut pow = t;
    for k in 1..200u32 {
        pow = pow.mul(t2).neg();
        let term = pow.div_small(2 * k + 1);
        if term.is_zero() {
            break;
        }
        // A negligible term still nudges the sum (see `X::add`), so the
        // final rounding sees which side of a representable value it is on.
        let done = term.e < sum.e - 132;
        sum = sum.add(term);
        if done {
            break;
        }
    }
    sum.scale(3)
}

/// `(sin r, cos r)` for `|r| ≤ π/4`.
fn sincos_small(r: X) -> (X, X) {
    let r2 = r.mul(r);
    let mut s = r;
    let mut term = r;
    for n in 1..60u32 {
        term = term.mul(r2).div_small((2 * n) * (2 * n + 1)).neg();
        if term.is_zero() {
            break;
        }
        // A negligible term still nudges the sum (see `X::add`), so the
        // final rounding sees which side of a representable value it is on.
        let done = term.e < s.e - 132;
        s = s.add(term);
        if done {
            break;
        }
    }
    let mut c = X::one();
    let mut term = X::one();
    for n in 1..60u32 {
        term = term.mul(r2).div_small((2 * n - 1) * (2 * n)).neg();
        if term.is_zero() {
            break;
        }
        let done = term.e < -134;
        c = c.add(term);
        if done {
            break;
        }
    }
    (s, c)
}

/// π rounded to 66 significant bits — the constant the x87 uses to reduce
/// trigonometric arguments (Intel SDM Vol. 1 §8.3.8): value `PI66 · 2^-64`.
const PI66: u128 = 0x3_243f_6a88_85a3_08d3;

/// Reduce finite `x` (`|x| < 2^63`) modulo π/2 the way the x87 does: exactly,
/// but against its 66-bit approximation of π, so `x = n·π₆₆/2 + r` with
/// `|r| ≤ π₆₆/4` (the source of the hardware's well-known loss of accuracy
/// near large multiples of π, which this reproduces). Returns `(n mod 4, r)`.
fn reduce_pio2(x: &Fp) -> (u32, X) {
    let le = x.lead_exp();
    if le < -1 {
        // |x| < 1/2 < π/4: no reduction.
        return (0, X::from_fp(x));
    }
    // x = mm·2^e with mm the 64-bit significand, e in [-65, -1].
    let msb = sf::msb(x.sig);
    let mm = x.sig << (63 - msb);
    let e = le - 63;
    // x = N·2^-65 and π₆₆/2 = PI66·2^-65 (PI66 is π·2^64), so
    // n = round(N / PI66) and r = (N - n·PI66)·2^-65, all exact.
    let num = mm << (e + 65);
    let mut q = num / PI66;
    let mut rem = num % PI66;
    let mut neg = false;
    if 2 * rem > PI66 {
        q += 1;
        rem = PI66 - rem;
        neg = true;
    }
    let r = X::from_u128(neg, rem, -65);
    let n = (q & 3) as u32;
    // The sign of x flips the whole reduction.
    if x.sign {
        ((4 - n) & 3, r.neg())
    } else {
        (n, r)
    }
}

/// `v` moved by one unit in the 128th bit toward (`up == false`) or away from
/// zero — for results that sit a sub-representable distance from an exactly
/// representable value, so the final rounding sees the right side of it.
fn nudge(v: X, up: bool) -> X {
    if v.is_zero() {
        return v;
    }
    if up {
        match v.m.checked_add(1) {
            Some(m) => X { m, ..v },
            None => X {
                m: 1 << 127,
                e: v.e + 1,
                ..v
            },
        }
    } else if v.m == 1 << 127 {
        X {
            m: u128::MAX,
            e: v.e - 1,
            ..v
        }
    } else {
        X { m: v.m - 1, ..v }
    }
}

/// `(sin x, cos x)` for finite `|x| < 2^63`.
fn sincos(x: &Fp) -> (X, X) {
    if x.lead_exp() < -40 {
        // sin x = x - x³/6…, cos x = 1 - x²/2…: the correction is far below
        // the working precision, but its direction decides directed rounding.
        let xv = X::from_fp(x);
        return (nudge(xv, false), nudge(X::one(), false));
    }
    let (n, r) = reduce_pio2(x);
    let (s, c) = sincos_small(r);
    match n {
        0 => (s, c),
        1 => (c, s.neg()),
        2 => (s.neg(), c.neg()),
        _ => (c.neg(), s),
    }
}

// ---- the instruction-level evaluation ----------------------------------------------

/// The outcome of a transcendental instruction.
pub(super) struct Eval {
    /// The primary result (`ST(0)`, or `ST(1)` for the popping forms).
    pub a: F80,
    /// `FSINCOS`'s pushed cosine.
    pub b: F80,
    pub flags: u32,
    pub c1: bool,
    pub c2: bool,
    /// The operand was out of range (`|x| ≥ 2^63` for the trig forms): `C2`
    /// is set and nothing is written.
    pub incomplete: bool,
}

/// One rounded result with its flags (`UE` when tiny, `PE`, `C1`).
fn finish(x: X, mode: Round) -> (F80, u32, bool) {
    let r = x.round(mode);
    let mut f = r.flags;
    if r.tiny {
        f |= UNDERFLOW;
    }
    (F80::pack(&r.v), f | INEXACT, r.up)
}

/// Evaluate transcendental `code` (see `X86Interp::x87_transcendental`) on
/// `x = ST(0)` and `y = ST(1)`.
#[allow(clippy::too_many_lines)]
pub(super) fn eval(code: usize, x: F80, y: F80, mode: Round) -> Eval {
    let mut out = Eval {
        a: F80::INDEFINITE,
        b: F80::INDEFINITE,
        flags: 0,
        c1: false,
        c2: false,
        incomplete: false,
    };
    let two_operand = matches!(code, 1 | 3 | 9);
    let unsupported = sf::f80_unsupported(x.0) || (two_operand && sf::f80_unsupported(y.0));
    if unsupported {
        out.flags = INVALID;
        return out;
    }
    let ux = x.unpack();
    let uy = y.unpack();
    let denorm = |v: F80| v.exp_field() == 0 && v.mant() != 0;
    let mut dflag = if denorm(x) { DENORMAL } else { 0 };
    if two_operand && denorm(y) {
        dflag |= DENORMAL;
    }
    // NaN operands propagate (quieted); an SNaN signals.
    let nan = if two_operand {
        super::x87::x87_nan2(&uy, &ux)
    } else if ux.is_nan() {
        Some((ux.quieted(), if ux.is_snan() { INVALID } else { 0 }))
    } else {
        None
    };
    if let Some((n, f)) = nan {
        out.a = F80::pack(&n);
        out.b = out.a;
        out.flags = f;
        return out;
    }
    let pack = |v: Fp| F80::pack(&v);
    match code {
        0 => {
            // F2XM1: 2^x - 1.
            out.flags = dflag;
            match ux.class {
                Class::Zero => out.a = x,
                Class::Inf => {
                    out.a = if ux.sign { pack(sf::from_int(-1)) } else { x };
                }
                _ => {
                    let xv = X::from_fp(&ux);
                    if xv.abs().e == 0 && xv.m == 1 << 127 {
                        // ±1 exactly: 1 or -1/2.
                        out.a = if ux.sign {
                            pack(Fp::finite(true, 1, -1))
                        } else {
                            F80::ONE
                        };
                    } else {
                        let r = expm1(xv.mul(LN2));
                        let (v, f, up) = finish(r, mode);
                        out.a = v;
                        out.flags |= f;
                        out.c1 = up;
                    }
                }
            }
        }
        1 | 9 => {
            // FYL2X: y·log2(x); FYL2XP1: y·log2(x + 1).
            out.flags = dflag;
            let lnv: Option<X>; // None = special-cased below
            let arg_zero;
            if code == 1 {
                if ux.sign && ux.class != Class::Zero {
                    out.flags |= INVALID;
                    return out;
                }
                arg_zero = ux.class == Class::Zero;
                lnv = if ux.class == Class::Finite {
                    Some(ln(&ux))
                } else {
                    None
                };
            } else {
                arg_zero = false;
                lnv = match ux.class {
                    Class::Finite => {
                        // ln(1 + x) = 2·atanh(x / (2 + x)) — accurate for the
                        // instruction's |x| < 1 - √2/2 domain; outside it (where
                        // the result is architecturally undefined) fall back
                        // to ln of the exact sum.
                        let xv = X::from_fp(&ux);
                        let one = X::one();
                        let sum = one.add(xv);
                        if sum.neg || sum.is_zero() {
                            out.flags |= INVALID;
                            return out;
                        }
                        if ux.lead_exp() < -2 {
                            Some(two_atanh(xv.div(sum.add(one))))
                        } else {
                            Some(ln_x(sum))
                        }
                    }
                    _ => None,
                };
            }
            match (lnv, uy.class) {
                (Some(l), Class::Finite) => {
                    if l.is_zero() {
                        out.a = pack(Fp::zero(uy.sign));
                    } else {
                        let r = l.mul(LOG2E).mul(X::from_fp(&uy));
                        let (v, f, up) = finish(r, mode);
                        out.a = v;
                        out.flags |= f;
                        out.c1 = up;
                    }
                }
                (Some(l), Class::Zero) => {
                    // ±0 · log: a signed zero (log sign XOR y sign).
                    out.a = pack(Fp::zero(uy.sign ^ (l.neg && !l.is_zero())));
                }
                (Some(l), Class::Inf) => {
                    if l.is_zero() {
                        out.flags |= INVALID;
                    } else {
                        out.a = pack(Fp::inf(uy.sign ^ l.neg));
                    }
                }
                (Some(_), Class::Nan) => {}
                (None, _) => {
                    if code == 9 {
                        // FYL2XP1 with x = ±0: ±0·y; x = ±∞ out of range.
                        if ux.class == Class::Zero {
                            if uy.class == Class::Inf {
                                out.flags |= INVALID;
                            } else {
                                out.a = pack(Fp::zero(ux.sign ^ uy.sign));
                            }
                        } else if ux.sign || uy.class == Class::Zero {
                            out.flags |= INVALID;
                        } else {
                            out.a = pack(Fp::inf(uy.sign));
                        }
                    } else if arg_zero {
                        // log2(±0) = -∞.
                        match uy.class {
                            Class::Zero => out.flags |= INVALID,
                            Class::Inf => out.a = pack(Fp::inf(!uy.sign)),
                            _ => {
                                out.flags |= DIVZERO;
                                out.a = pack(Fp::inf(!uy.sign));
                            }
                        }
                    } else {
                        // x = +∞: log2 = +∞.
                        if uy.class == Class::Zero {
                            out.flags |= INVALID;
                        } else {
                            out.a = pack(Fp::inf(uy.sign));
                        }
                    }
                }
            }
        }
        3 => {
            // FPATAN: atan2(y = ST(1), x = ST(0)).
            out.flags = dflag;
            let pi = PI;
            let res: Option<X> = match (uy.class, ux.class) {
                (Class::Zero, _) => {
                    if ux.sign {
                        Some(pi)
                    } else {
                        out.a = pack(Fp::zero(uy.sign));
                        None
                    }
                }
                (Class::Inf, Class::Inf) => Some(if ux.sign {
                    pi.mul(X::constant(3 << 126, 1)).scale(-2)
                } else {
                    pi.scale(-2)
                }),
                (Class::Inf, _) | (_, Class::Zero) => Some(pi.scale(-1)),
                (_, Class::Inf) => {
                    if ux.sign {
                        Some(pi)
                    } else {
                        out.a = pack(Fp::zero(uy.sign));
                        None
                    }
                }
                _ => {
                    let (ay, ax) = (X::from_fp(&uy).abs(), X::from_fp(&ux).abs());
                    let le = (ay.e, ay.m) <= (ax.e, ax.m);
                    let mut a = if le {
                        atan_unit(ay.div(ax))
                    } else {
                        pi.scale(-1).sub(atan_unit(ax.div(ay)))
                    };
                    if ux.sign {
                        a = pi.sub(a);
                    }
                    Some(a)
                }
            };
            if let Some(r) = res {
                let r = if uy.sign { r.neg() } else { r };
                let (v, f, up) = finish(r, mode);
                out.a = v;
                out.flags |= f;
                out.c1 = up;
            }
        }
        2 | 11 | 14 | 15 => {
            // FPTAN, FSINCOS, FSIN, FCOS.
            out.flags = dflag;
            match ux.class {
                Class::Inf => {
                    out.flags |= INVALID;
                    return out;
                }
                Class::Zero => {
                    // sin/tan(±0) = ±0 exactly, cos(±0) = 1.
                    out.a = if code == 15 { F80::ONE } else { x };
                    out.b = F80::ONE;
                    return out;
                }
                _ => {}
            }
            if ux.lead_exp() >= 63 {
                out.c2 = true;
                out.incomplete = true;
                out.flags = dflag;
                return out;
            }
            let (s, c) = sincos(&ux);
            let primary = match code {
                // tan x = x + x³/3… just above |x| for tiny arguments.
                2 if ux.lead_exp() < -40 => nudge(X::from_fp(&ux), true),
                2 => s.div(c),
                15 => c,
                _ => s,
            };
            let (v, f, up) = finish(primary, mode);
            out.a = v;
            out.flags |= f;
            out.c1 = up;
            if code == 11 {
                let (v, f, up) = finish(c, mode);
                out.b = v;
                out.flags |= f;
                out.c1 = up;
            }
        }
        _ => {}
    }
    out
}

/// The exact `FPREM`/`FPREM1` step for finite nonzero `x`, `y`: returns the
/// (exact) remainder, the low quotient bits, and whether the reduction was
/// only partial (exponent difference ≥ 64).
pub(super) fn fprem(x: &Fp, y: &Fp, nearest: bool) -> (Fp, u64, bool) {
    // Normalize both significands to 64 bits.
    let nx = sf::msb(x.sig);
    let ny = sf::msb(y.sig);
    let a = x.sig << (63 - nx);
    let b = y.sig << (63 - ny);
    let ea = x.exp - (63 - nx as i32);
    let eb = y.exp - (63 - ny as i32);
    let d = ea - eb;
    if d < 0 {
        // |x| < |y|: quotient 0 (FPREM), or possibly ±1 (FPREM1).
        if nearest && d == -1 && a > b {
            // |x| in (|y|/2, |y|): r = x - sign·y.
            let r = (b << 1) - a; // magnitude of |y| - |x| at exponent ea
            let r = Fp::finite(!x.sign, r, ea);
            return (r, 1, false);
        }
        return (*x, 0, false);
    }
    let (n, partial) = if d >= 64 {
        (32 + (d % 32), true)
    } else {
        (d, false)
    };
    // Divide a·2^n by b exactly (n < 64 → a·2^n < 2^127).
    let num = a << n;
    let mut q = num / b;
    let mut r = num % b;
    let mut sign = x.sign;
    if nearest && !partial && (2 * r > b || (2 * r == b && q & 1 == 1)) {
        q += 1;
        r = b - r;
        sign = !sign;
    }
    // The remainder is r at the exponent of b's lsb shifted up by (d - n).
    let re = eb + (d - n);
    let rv = if r == 0 {
        Fp::zero(x.sign)
    } else {
        Fp::finite(sign, r, re)
    };
    (rv, q as u64, partial)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f80(x: f64) -> F80 {
        F80::from_f64(x.to_bits())
    }

    #[test]
    fn constants_round_like_hardware() {
        assert_eq!(
            constant(3, Round::Nearest).unwrap().0,
            0x4000_c90f_daa2_2168_c235
        );
        assert_eq!(
            constant(3, Round::Zero).unwrap().0,
            0x4000_c90f_daa2_2168_c234
        );
        assert_eq!(
            constant(2, Round::Nearest).unwrap().0,
            0x3fff_b8aa_3b29_5c17_f0bc
        );
        assert_eq!(
            constant(5, Round::Nearest).unwrap().0,
            0x3ffe_b172_17f7_d1cf_79ac
        );
    }

    #[test]
    fn sin_cos_of_one_are_correctly_rounded() {
        let r = eval(14, f80(1.0), F80::ZERO, Round::Nearest);
        assert_eq!(r.a.0, 0x3ffe_d76a_a478_4867_7021);
        let r = eval(15, f80(1.0), F80::ZERO, Round::Nearest);
        assert_eq!(r.a.0, 0x3ffe_8a51_407d_a834_5c92);
    }

    #[test]
    fn log_exp_atan_match_reference() {
        // 3 · log2(log2(e)): FYL2X with ST(0) = log2(e), ST(1) = 3.
        let l2e = constant(2, Round::Nearest).unwrap();
        let r = eval(1, l2e, f80(3.0), Round::Nearest);
        assert_eq!(r.a.0, 0x3fff_cb0b_d97a_88c8_a6b8);
        // atan(1/3 / 1/3) = π/4
        let third = F80(0x3ffd_aaaa_aaaa_aaaa_aaab);
        let r = eval(3, third, third, Round::Nearest);
        assert_eq!(r.a.0, 0x3ffe_c90f_daa2_2168_c235);
        // 2^(1/3) - 1
        let r = eval(0, third, F80::ZERO, Round::Nearest);
        assert_eq!(r.a.0, 0x3ffd_8514_5f31_ae51_5c45);
    }

    #[test]
    fn fprem_exact() {
        let x = f80(10.0).unpack();
        let y = f80(3.0).unpack();
        let (r, q, partial) = fprem(&x, &y, false);
        assert!(!partial);
        assert_eq!(q, 3);
        assert_eq!(F80::pack(&r), f80(1.0));
        let (r, q, _) = fprem(&f80(11.0).unpack(), &y, true);
        assert_eq!(q, 4);
        assert_eq!(F80::pack(&r), f80(-1.0));
    }
}
