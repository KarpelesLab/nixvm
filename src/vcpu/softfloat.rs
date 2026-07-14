//! A small, dependency-free software floating-point core, precise enough to be
//! *bit-exact* with real x86 hardware.
//!
//! The interpreter needs two things `f64` hardware alone can't give it:
//!
//! * **true 80-bit x87 extended precision** — the x87 register stack computes
//!   in a 64-bit-significand format with a 15-bit exponent; modelling it in
//!   `f64` loses the low bits every intermediate touches.
//! * **directed rounding** — SSE code can select round-toward-zero/±∞ via
//!   `MXCSR`, and x87 via its control word. You cannot get a correctly directed
//!   result by re-rounding an `f64` computed in round-to-nearest: that
//!   double-rounds and produces the wrong low bit.
//!
//! Both fall out of one design: every operation is carried out on an *unpacked*
//! value (`sign · sig · 2^exp`, `sig` an integer with all the bits the op can
//! produce) and rounded exactly once, by [`round`], into the destination
//! [`Fmt`] under the requested [`Round`] mode. The same code serves `f32`,
//! `f64`, and `f80` — only the format parameters differ. Every rounding path
//! also reports the IEEE exception flags (`invalid`, `div-by-zero`, `overflow`,
//! `underflow`, `inexact`, `denormal`) the guest reads back through `MXCSR` /
//! `FNSTSW`.
//!
//! Correctness is pinned two ways (see the tests and the KVM differential
//! harness): for round-to-nearest the `f32`/`f64` results must equal the host's
//! native arithmetic exactly, and the `f80` and directed-rounding results are
//! diffed against real x87/SSE hardware.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    // The arithmetic core is dense with one-letter math names (a, b, q, r, e…).
    clippy::many_single_char_names,
    // Special-value arms (inf/zero/nan combinations) share result expressions
    // but read far clearer kept separate and in IEEE order.
    clippy::match_same_arms
)]

/// IEEE rounding mode, encoded as x86 does (`MXCSR[14:13]` / x87 `CW[11:10]`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Round {
    /// Round to nearest, ties to even (the reset default).
    Nearest,
    /// Round toward −∞ (`01`).
    Down,
    /// Round toward +∞ (`10`).
    Up,
    /// Round toward zero / truncate (`11`).
    Zero,
}

impl Round {
    /// Decode the 2-bit rounding-control field shared by `MXCSR` and the x87
    /// control word.
    #[must_use]
    pub fn from_x86(rc: u32) -> Self {
        match rc & 3 {
            1 => Round::Down,
            2 => Round::Up,
            3 => Round::Zero,
            _ => Round::Nearest,
        }
    }
}

/// IEEE exception flags, in `MXCSR`/x87-status bit positions (`IE`,`DE`,`ZE`,
/// `OE`,`UE`,`PE`). Accumulated by every op and OR-ed into the guest's status
/// word by the caller.
pub const INVALID: u32 = 0x01;
pub const DENORMAL: u32 = 0x02;
pub const DIVZERO: u32 = 0x04;
pub const OVERFLOW: u32 = 0x08;
pub const UNDERFLOW: u32 = 0x10;
pub const INEXACT: u32 = 0x20;

/// A binary floating-point format: significand precision (including the leading
/// bit) and the exponent of the leading bit for the largest/smallest normals.
#[derive(Clone, Copy, Debug)]
pub struct Fmt {
    /// Significand bits, counting the implicit/explicit integer bit (24/53/64).
    pub prec: u32,
    /// Leading-bit exponent of the largest finite normal (127/1023/16383).
    pub emax: i32,
    /// Leading-bit exponent of the smallest normal (`1 - emax`).
    pub emin: i32,
}

pub const FMT32: Fmt = Fmt { prec: 24, emax: 127, emin: -126 };
pub const FMT64: Fmt = Fmt { prec: 53, emax: 1023, emin: -1022 };
/// x87 80-bit extended: 64-bit significand with an *explicit* integer bit,
/// 15-bit exponent (bias 16383).
pub const FMT80: Fmt = Fmt { prec: 64, emax: 16383, emin: -16382 };

/// The category of a value, kept separate so arithmetic can branch on it
/// without decoding bit patterns repeatedly.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    Zero,
    Inf,
    Nan,
    Finite,
}

/// An exact, unpacked value: `(-1)^sign · sig · 2^exp` for `Finite`; for `Nan`,
/// `sig` carries the significand payload (right-justified in the source
/// format's mantissa). `sig` is a plain integer with no assumed normalization —
/// arithmetic fills it with as many bits as the operation produces and [`round`]
/// is the only place bits are dropped.
#[derive(Clone, Copy, Debug)]
struct Unpacked {
    sign: bool,
    class: Class,
    sig: u128,
    exp: i32,
}

impl Unpacked {
    const fn zero(sign: bool) -> Self {
        Self { sign, class: Class::Zero, sig: 0, exp: 0 }
    }
    const fn inf(sign: bool) -> Self {
        Self { sign, class: Class::Inf, sig: 0, exp: 0 }
    }
    /// A quiet NaN carrying `payload` (the source mantissa, quiet bit optional —
    /// callers set it). The canonical x86 QNaN ("real indefinite") has just the
    /// quiet bit set, which callers request with `payload == 0` after quieting.
    const fn nan(sign: bool, payload: u128) -> Self {
        Self { sign, class: Class::Nan, sig: payload, exp: 0 }
    }
}

/// Position of the most-significant set bit (0-based); `sig` must be nonzero.
#[inline]
fn msb(sig: u128) -> u32 {
    sig.ilog2()
}

// ---- unpack: format bits -> Unpacked (exact) -------------------------------

fn unpack_f32(bits: u32) -> Unpacked {
    let sign = bits >> 31 != 0;
    let exp = (bits >> 23) & 0xff;
    let mant = u128::from(bits & 0x7f_ffff);
    match exp {
        0xff if mant == 0 => Unpacked::inf(sign),
        0xff => Unpacked::nan(sign, mant),
        0 if mant == 0 => Unpacked::zero(sign),
        // Subnormal: value = mant · 2^(1-127-23).
        0 => Unpacked { sign, class: Class::Finite, sig: mant, exp: -149 },
        // Normal: value = (2^23 | mant) · 2^(exp-127-23).
        _ => Unpacked {
            sign,
            class: Class::Finite,
            sig: mant | (1 << 23),
            exp: exp as i32 - 150,
        },
    }
}

fn unpack_f64(bits: u64) -> Unpacked {
    let sign = bits >> 63 != 0;
    let exp = (bits >> 52) & 0x7ff;
    let mant = u128::from(bits & 0xf_ffff_ffff_ffff);
    match exp {
        0x7ff if mant == 0 => Unpacked::inf(sign),
        0x7ff => Unpacked::nan(sign, mant),
        0 if mant == 0 => Unpacked::zero(sign),
        0 => Unpacked { sign, class: Class::Finite, sig: mant, exp: -1074 },
        _ => Unpacked {
            sign,
            class: Class::Finite,
            sig: mant | (1 << 52),
            exp: exp as i32 - 1075,
        },
    }
}

/// Unpack an x87 80-bit value (low 80 bits of `bits`). Unlike IEEE binary32/64
/// the integer bit is explicit, so "unnormal"/pseudo forms are representable;
/// we treat the significand as given (bit 63 is the integer bit).
fn unpack_f80(bits: u128) -> Unpacked {
    let sign = (bits >> 79) & 1 != 0;
    let exp = ((bits >> 64) & 0x7fff) as u32;
    let sig = bits & 0xffff_ffff_ffff_ffff; // 64-bit significand, integer bit at 63
    let frac = sig & 0x7fff_ffff_ffff_ffff;
    match exp {
        0x7fff if frac == 0 && (sig >> 63) & 1 == 1 => Unpacked::inf(sign),
        // NaN (or pseudo-inf/unnormal with max exp): carry the low mantissa.
        0x7fff => Unpacked::nan(sign, frac),
        0 if sig == 0 => Unpacked::zero(sign),
        // Both normals (integer bit set) and subnormals (clear) reduce to the
        // same exact value = sig · 2^(exp - 16383 - 63).
        _ => Unpacked { sign, class: Class::Finite, sig, exp: exp as i32 - 16446 },
    }
}

// ---- pack: rounded Unpacked -> format bits ---------------------------------
//
// After `round`, a Finite value has `sig` holding exactly the destination
// significand (prec bits when normal, fewer when subnormal) and `exp` the
// exponent of its least-significant bit.

fn pack_f32(v: &Unpacked) -> u32 {
    let s = u32::from(v.sign) << 31;
    match v.class {
        Class::Zero => s,
        Class::Inf => s | (0xff << 23),
        Class::Nan => s | (0xff << 23) | (1 << 22) | (v.sig as u32 & 0x3f_ffff),
        Class::Finite => {
            let e = v.exp + msb(v.sig) as i32; // leading-bit exponent
            if e < FMT32.emin {
                // Subnormal: mantissa = sig · 2^(exp + 149).
                let m = (v.sig << (v.exp + 149)) as u32;
                s | m
            } else {
                let exp_field = (e + 127) as u32;
                let mant = (v.sig as u32) & 0x7f_ffff;
                s | (exp_field << 23) | mant
            }
        }
    }
}

fn pack_f64(v: &Unpacked) -> u64 {
    let s = u64::from(v.sign) << 63;
    match v.class {
        Class::Zero => s,
        Class::Inf => s | (0x7ff << 52),
        Class::Nan => s | (0x7ff << 52) | (1 << 51) | (v.sig as u64 & 0x7_ffff_ffff_ffff),
        Class::Finite => {
            let e = v.exp + msb(v.sig) as i32;
            if e < FMT64.emin {
                let m = (v.sig << (v.exp + 1074)) as u64;
                s | m
            } else {
                let exp_field = (e + 1023) as u64;
                let mant = (v.sig as u64) & 0xf_ffff_ffff_ffff;
                s | (exp_field << 52) | mant
            }
        }
    }
}

fn pack_f80(v: &Unpacked) -> u128 {
    let s = u128::from(v.sign) << 79;
    match v.class {
        Class::Zero => s,
        Class::Inf => s | (0x7fff << 64) | (1 << 63), // integer bit set
        Class::Nan => {
            // Canonical/propagated QNaN: max exp, integer bit + quiet bit set.
            s | (0x7fff << 64) | (1 << 63) | (1 << 62) | (v.sig & 0x3fff_ffff_ffff_ffff)
        }
        Class::Finite => {
            let e = v.exp + msb(v.sig) as i32;
            if e < FMT80.emin {
                // Subnormal: exp field 0, integer bit clear, significand shifted.
                let m = v.sig << (v.exp + 16446);
                s | (m & 0xffff_ffff_ffff_ffff)
            } else {
                let exp_field = (e + 16383) as u128;
                // Explicit integer bit is part of the stored 64-bit significand.
                s | (exp_field << 64) | (v.sig & 0xffff_ffff_ffff_ffff)
            }
        }
    }
}

// ---- the one rounding point ------------------------------------------------

/// Round the exact finite value `(-1)^sign · sig · 2^exp` (with `extra_sticky`
/// recording nonzero bits already dropped below bit 0 of `sig`) into `fmt`
/// under `mode`, returning the rounded [`Unpacked`] plus exception flags.
///
/// This is where precision is lost — exactly once — so directed rounding is
/// correct: the true result's guard/round/sticky bits and sign drive the
/// decision, never a previously-rounded intermediate.
fn round(
    sign: bool,
    sig: u128,
    exp: i32,
    extra_sticky: bool,
    fmt: Fmt,
    mode: Round,
) -> (Unpacked, u32) {
    if sig == 0 {
        // Exact zero; a spurious extra_sticky can't happen with sig==0 here.
        return (Unpacked::zero(sign), 0);
    }
    let leading = msb(sig) as i32;
    let e = exp + leading; // leading-bit exponent of the value

    // Exponent of the result's least-significant bit: fixed at the subnormal
    // granularity when the value is below the normal range, else prec-1 below
    // the leading bit.
    let ulp_exp = if e < fmt.emin {
        fmt.emin - (fmt.prec as i32 - 1)
    } else {
        e - (fmt.prec as i32 - 1)
    };

    // Split sig at ulp_exp into the kept quotient q and the discarded low part.
    let shift = ulp_exp - exp;
    let (mut q, round_bit, mut sticky) = if shift <= 0 {
        // No low bits dropped; may need to move up (exact).
        (sig << (-shift) as u32, false, false)
    } else if (shift as u32) >= 128 {
        (0u128, false, sig != 0)
    } else {
        let sh = shift as u32;
        let q = sig >> sh;
        let round_bit = (sig >> (sh - 1)) & 1 != 0;
        let sticky = sig & ((1u128 << (sh - 1)) - 1) != 0;
        (q, round_bit, sticky)
    };
    sticky |= extra_sticky;

    let inexact = round_bit || sticky;
    // Decide whether to round the magnitude up by one ulp.
    let round_up = match mode {
        Round::Nearest => round_bit && (sticky || (q & 1 != 0)),
        Round::Zero => false,
        Round::Up => inexact && !sign,
        Round::Down => inexact && sign,
    };
    if round_up {
        q += 1;
    }

    let mut flags = 0;
    if inexact {
        flags |= INEXACT;
    }

    // A round-up carry can widen q to prec+1 bits; renormalize.
    let mut result_ulp = ulp_exp;
    if q != 0 && (msb(q) as i32) >= fmt.prec as i32 && e >= fmt.emin {
        // q grew past prec bits: drop the new low bit (it is zero) and bump exp.
        q >>= 1;
        result_ulp += 1;
    }
    // Subnormal that rounded up exactly to the smallest normal is fine: it now
    // has prec bits and its computed leading-bit exponent equals emin.

    let final_e = if q == 0 { fmt.emin } else { result_ulp + msb(q) as i32 };

    // Overflow: past the largest representable exponent.
    if final_e > fmt.emax {
        flags |= OVERFLOW | INEXACT;
        let to_inf = match mode {
            Round::Nearest => true,
            Round::Zero => false,
            Round::Up => !sign,
            Round::Down => sign,
        };
        if to_inf {
            return (Unpacked::inf(sign), flags);
        }
        // Largest finite: significand all ones at emax.
        let maxsig = (1u128 << fmt.prec) - 1;
        return (
            Unpacked { sign, class: Class::Finite, sig: maxsig, exp: fmt.emax - (fmt.prec as i32 - 1) },
            flags,
        );
    }

    // Underflow (IEEE "tininess before rounding, inexact result"): flagged when
    // the pre-rounding value was subnormal and the result is inexact.
    if e < fmt.emin && inexact {
        flags |= UNDERFLOW;
    }

    if q == 0 {
        return (Unpacked::zero(sign), flags);
    }
    (Unpacked { sign, class: Class::Finite, sig: q, exp: result_ulp }, flags)
}

// ---- NaN handling ----------------------------------------------------------

/// Quiet a NaN and set INVALID if either operand was a signaling NaN. Returns
/// the propagated NaN (x86 keeps the source NaN, quieted, preferring the first).
fn propagate_nan(a: &Unpacked, b: Option<&Unpacked>, fmt: Fmt) -> (Unpacked, u32) {
    let quiet_bit = 1u128 << (fmt.prec - 2);
    let is_snan = |v: &Unpacked| v.class == Class::Nan && (v.sig & quiet_bit == 0);
    let mut flags = 0;
    if is_snan(a) || b.is_some_and(is_snan) {
        flags |= INVALID;
    }
    let src = if a.class == Class::Nan {
        a
    } else {
        b.expect("propagate_nan called with no NaN operand")
    };
    (Unpacked::nan(src.sign, src.sig | quiet_bit), flags)
}

/// The default/indefinite QNaN a generated invalid result yields (negative sign,
/// only the quiet bit set — matching x86's "real indefinite").
fn default_nan() -> (Unpacked, u32) {
    (Unpacked::nan(true, 0), INVALID)
}

/// OR the DENORMAL flag if a finite operand is subnormal (as SSE reports).
fn denormal_flag(v: &Unpacked, fmt: Fmt) -> u32 {
    if v.class == Class::Finite && v.sig != 0 && v.exp + (msb(v.sig) as i32) < fmt.emin {
        DENORMAL
    } else {
        0
    }
}

// ---- arithmetic on Unpacked ------------------------------------------------

fn add_unpacked(a: Unpacked, b: Unpacked, fmt: Fmt, mode: Round) -> (Unpacked, u32) {
    use Class::{Finite, Inf, Nan, Zero};
    let dflag = denormal_flag(&a, fmt) | denormal_flag(&b, fmt);
    match (a.class, b.class) {
        (Nan, _) | (_, Nan) => propagate_nan(&a, Some(&b), fmt),
        (Inf, Inf) => {
            if a.sign == b.sign {
                (Unpacked::inf(a.sign), 0)
            } else {
                default_nan() // (+∞) + (−∞)
            }
        }
        (Inf, _) => (Unpacked::inf(a.sign), 0),
        (_, Inf) => (Unpacked::inf(b.sign), 0),
        (Zero, Zero) => {
            // −0 + −0 = −0; every other zero-sum is +0 except toward −∞.
            let sign = if a.sign == b.sign { a.sign } else { mode == Round::Down };
            (Unpacked::zero(sign), 0)
        }
        (Zero, _) => (b, dflag),
        (_, Zero) => (a, dflag),
        (Finite, Finite) => {
            let (r, f) = add_finite(a, b, fmt, mode);
            (r, f | dflag)
        }
    }
}

/// Left-justify `sig` so its most-significant bit sits at bit 63, returning the
/// justified significand and the value's leading-bit exponent.
fn norm63(sig: u128, exp: i32) -> (u128, i32) {
    let m = msb(sig);
    (sig << (63 - m), exp + m as i32)
}

/// Right shift capturing whether any set bit was shifted out (the sticky bit).
fn shr_sticky(x: u128, n: u32) -> (u128, bool) {
    if n == 0 {
        (x, false)
    } else if n >= 128 {
        (0, x != 0)
    } else {
        (x >> n, x & ((1u128 << n) - 1) != 0)
    }
}

/// Core finite add/subtract. Both significands are left-justified to bit 63,
/// widened to bit 126 (leaving carry room above and 63 alignment bits below),
/// aligned by exponent with a sticky bit, then combined and rounded once. The
/// far-operand subtraction borrow is exact because deep cancellation only
/// happens at equal leading exponents (where the shift is zero and there is no
/// sticky bit to borrow from).
fn add_finite(a: Unpacked, b: Unpacked, fmt: Fmt, mode: Round) -> (Unpacked, u32) {
    let (sa, ea) = norm63(a.sig, a.exp);
    let (sb, eb) = norm63(b.sig, b.exp);
    // `hi` has the larger (or equal) leading-bit exponent.
    let (hs, he, hsign, ls, _le, lsign) = if ea >= eb {
        (sa, ea, a.sign, sb, eb, b.sign)
    } else {
        (sb, eb, b.sign, sa, ea, a.sign)
    };
    let d = (he - if ea >= eb { eb } else { ea }) as u32;
    let big = hs << 63; // MSB at bit 126
    let (small, sticky) = shr_sticky(ls << 63, d);

    if hsign == lsign {
        let sum = big + small; // < 2^128
        round(hsign, sum, he - 126, sticky, fmt, mode)
    } else if d == 0 {
        // Equal leading exponents: exact subtraction, no sticky bit.
        match big.cmp(&small) {
            core::cmp::Ordering::Equal => (Unpacked::zero(mode == Round::Down), 0),
            core::cmp::Ordering::Greater => round(hsign, big - small, he - 126, false, fmt, mode),
            core::cmp::Ordering::Less => round(lsign, small - big, he - 126, false, fmt, mode),
        }
    } else {
        // d ≥ 1: `big` dominates. The exact small operand is `small + frac`
        // (frac < 1 ulp, present iff sticky), so subtract the borrow.
        let mut diff = big - small - u128::from(sticky);
        if diff == 0 && sticky {
            // Vanishingly rare: result is a sub-ulp positive remnant. Represent
            // it as the smallest nonzero at this scale rather than losing it.
            diff = 1;
            return round(hsign, diff, he - 127, true, fmt, mode);
        }
        round(hsign, diff, he - 126, sticky, fmt, mode)
    }
}

fn mul_unpacked(a: Unpacked, b: Unpacked, fmt: Fmt, mode: Round) -> (Unpacked, u32) {
    use Class::{Finite, Inf, Nan, Zero};
    let dflag = denormal_flag(&a, fmt) | denormal_flag(&b, fmt);
    match (a.class, b.class) {
        (Nan, _) | (_, Nan) => propagate_nan(&a, Some(&b), fmt),
        (Inf, Zero) | (Zero, Inf) => default_nan(),
        (Inf, _) | (_, Inf) => (Unpacked::inf(a.sign ^ b.sign), 0),
        (Zero, _) | (_, Zero) => (Unpacked::zero(a.sign ^ b.sign), 0),
        (Finite, Finite) => {
            // Significands ≤64 bits each → product ≤128 bits, exact in u128.
            let prod = a.sig * b.sig;
            let (r, f) = round(a.sign ^ b.sign, prod, a.exp + b.exp, false, fmt, mode);
            (r, f | dflag)
        }
    }
}

fn div_unpacked(a: Unpacked, b: Unpacked, fmt: Fmt, mode: Round) -> (Unpacked, u32) {
    use Class::{Finite, Inf, Nan, Zero};
    let dflag = denormal_flag(&a, fmt) | denormal_flag(&b, fmt);
    let sign = a.sign ^ b.sign;
    match (a.class, b.class) {
        (Nan, _) | (_, Nan) => propagate_nan(&a, Some(&b), fmt),
        (Inf, Inf) | (Zero, Zero) => default_nan(),
        (Inf, _) => (Unpacked::inf(sign), 0),
        (_, Inf) => (Unpacked::zero(sign), 0),
        (_, Zero) => (Unpacked::inf(sign), DIVZERO), // finite / 0
        (Zero, _) => (Unpacked::zero(sign), 0),
        (Finite, Finite) => {
            // Bit-serial long division producing prec+3 quotient bits + sticky.
            // Normalize both significands to bit 126 so the remainder never
            // overflows when shifted left each step (rem < db < 2^127).
            let na = msb(a.sig);
            let nb = msb(b.sig);
            let da = a.sig << (126 - na);
            let db = b.sig << (126 - nb);
            let a_exp = a.exp + na as i32 - 126;
            let b_exp = b.exp + nb as i32 - 126;
            // da,db ∈ [2^126, 2^127); quotient da/db ∈ [0.5, 2).
            let nbits = fmt.prec + 3;
            let mut rem = da;
            let mut q: u128 = 0;
            for _ in 0..nbits {
                q <<= 1;
                if rem >= db {
                    rem -= db;
                    q |= 1;
                }
                rem <<= 1; // rem < db < 2^127, so this cannot overflow u128
            }
            let sticky = rem != 0;
            // Each step doubles q once and shifts rem once, so q carries an
            // extra factor of two: value = q · 2^(a_exp - b_exp - (nbits-1)).
            let (r, f) = round(sign, q, a_exp - b_exp - (nbits as i32 - 1), sticky, fmt, mode);
            (r, f | dflag)
        }
    }
}

/// Integer square root of a `u128` (floor), plus whether it was exact.
fn isqrt128(n: u128) -> (u128, bool) {
    if n == 0 {
        return (0, true);
    }
    // Initial estimate from the bit length, then Newton iteration in u128.
    let mut x = 1u128 << (msb(n) / 2 + 1);
    loop {
        let nx = x.midpoint(n / x);
        if nx >= x {
            break;
        }
        x = nx;
    }
    // x is floor(sqrt(n)) (or one too high by construction); correct down.
    while x > 0 && x > n / x {
        x -= 1;
    }
    while (x + 1) <= n / (x + 1) {
        x += 1;
    }
    (x, x * x == n)
}

fn sqrt_unpacked(a: Unpacked, fmt: Fmt, mode: Round) -> (Unpacked, u32) {
    use Class::{Finite, Inf, Nan, Zero};
    match a.class {
        Nan => propagate_nan(&a, None, fmt),
        Zero => (a, 0), // √±0 = ±0
        Inf if a.sign => default_nan(), // √−∞
        Inf => (a, 0),
        Finite if a.sign => default_nan(), // √(negative)
        Finite => {
            // value = sig · 2^exp. Make exp even, then scale sig (by an even
            // amount, preserving the value's evenness) so its integer square
            // root lands in exactly `prec` bits — [2^(2·prec-2), 2^(2·prec)),
            // which fits u128 for every format we support (2·prec ≤ 128).
            let mut sig = a.sig;
            let mut exp = a.exp;
            if exp & 1 != 0 {
                sig <<= 1;
                exp -= 1;
            }
            let target = 2 * fmt.prec - 2; // desired msb of the scaled significand
            let cur = msb(sig);
            // Pick an even shift landing msb at `target` or `target+1`.
            let mut k = target - cur;
            if !k.is_multiple_of(2) {
                k += 1;
            }
            sig <<= k;
            exp -= k as i32;

            // q = ⌊√sig⌋ has exactly `prec` bits; r is the remainder. The exact
            // root lies in [q, q+1); it exceeds the q+½ midpoint iff r > q (and
            // a midpoint is never hit exactly, since (q+½)² is not an integer),
            // so ordinary round-to-nearest needs no tie rule.
            let (q, _) = isqrt128(sig);
            let r = sig - q * q;
            let inexact = r != 0;
            let round_up = match mode {
                Round::Nearest => r > q,
                Round::Zero => false,
                Round::Up => inexact, // √ is non-negative
                Round::Down => false,
            };
            let qr = q + u128::from(round_up);
            // qr already carries the correctly-rounded `prec` bits; `round` only
            // normalizes a possible carry-out — it drops no significant bits, so
            // there is no double rounding.
            let (mut res, mut flags) = round(false, qr, exp / 2, false, fmt, Round::Nearest);
            if inexact {
                flags |= INEXACT;
                if res.class == Class::Finite
                    && res.exp + (msb(res.sig) as i32) < fmt.emin
                {
                    flags |= UNDERFLOW;
                }
            }
            res.sign = false;
            (res, flags | denormal_flag(&a, fmt))
        }
    }
}

// ---- public, format-typed API ---------------------------------------------
//
// x87 keeps values as `F80` (an 80-bit pattern) between operations to preserve
// the extended precision; SSE goes bits→op→bits per instruction.

/// An x87 80-bit extended value, stored as its 80-bit encoding in the low bits.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct F80(pub u128);

impl core::fmt::Debug for F80 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "F80({:#022x}={})", self.0 & 0xffff_ffff_ffff_ffff_ffff, self.to_f64())
    }
}

impl F80 {
    #[must_use]
    pub fn from_f64(bits: u64) -> Self {
        // Widening f64→f80 is exact; RNE with F80 never rounds a finite f64.
        let (v, _) = round_unpacked(unpack_f64(bits), FMT80, Round::Nearest);
        F80(pack_f80(&v))
    }
    #[must_use]
    pub fn from_f32(bits: u32) -> Self {
        let (v, _) = round_unpacked(unpack_f32(bits), FMT80, Round::Nearest);
        F80(pack_f80(&v))
    }
    #[must_use]
    pub fn to_f64_round(self, mode: Round) -> (u64, u32) {
        let (v, f) = round_unpacked(unpack_f80(self.0), FMT64, mode);
        (pack_f64(&v), f)
    }
    #[must_use]
    pub fn to_f32_round(self, mode: Round) -> (u32, u32) {
        let (v, f) = round_unpacked(unpack_f80(self.0), FMT32, mode);
        (pack_f32(&v), f)
    }
    /// Lossy `to_f64` for the transcendental fallbacks (RNE, flags dropped).
    #[must_use]
    pub fn to_f64(self) -> f64 {
        f64::from_bits(self.to_f64_round(Round::Nearest).0)
    }
    #[must_use]
    pub fn from_f64_val(x: f64) -> Self {
        Self::from_f64(x.to_bits())
    }
    #[must_use]
    pub fn from_i64(v: i64) -> Self {
        if v == 0 {
            return F80(0);
        }
        let sign = v < 0;
        let mag = u128::from(v.unsigned_abs());
        let (u, _) = round(sign, mag, 0, false, FMT80, Round::Nearest);
        F80(pack_f80(&u))
    }
    /// Convert to a 64-bit integer, rounding per `mode`; out-of-range/NaN yields
    /// the x86 "integer indefinite" (`i64::MIN`).
    #[must_use]
    pub fn to_i64_round(self, mode: Round) -> i64 {
        let u = unpack_f80(self.0);
        match u.class {
            Class::Zero => 0,
            Class::Nan | Class::Inf => i64::MIN,
            Class::Finite => {
                // A leading bit above 2^63 can't fit; reject before shifting so
                // `sig << exp` can't overflow u128 (guaranteed here: ≤64 bits).
                if u.exp + msb(u.sig) as i32 > 63 {
                    return i64::MIN;
                }
                let (mag, _) = round_to_int_mag(u.sign, u.sig, u.exp, mode);
                if u.sign {
                    if mag > 1u128 << 63 {
                        return i64::MIN;
                    }
                    (mag as i64).wrapping_neg()
                } else {
                    if mag > i64::MAX as u128 {
                        return i64::MIN;
                    }
                    mag as i64
                }
            }
        }
    }

    pub fn add(self, o: F80, mode: Round) -> (F80, u32) {
        pack_res(add_unpacked(unpack_f80(self.0), unpack_f80(o.0), FMT80, mode))
    }
    pub fn sub(self, o: F80, mode: Round) -> (F80, u32) {
        let mut b = unpack_f80(o.0);
        b.sign = !b.sign;
        pack_res(add_unpacked(unpack_f80(self.0), b, FMT80, mode))
    }
    pub fn mul(self, o: F80, mode: Round) -> (F80, u32) {
        pack_res(mul_unpacked(unpack_f80(self.0), unpack_f80(o.0), FMT80, mode))
    }
    pub fn div(self, o: F80, mode: Round) -> (F80, u32) {
        pack_res(div_unpacked(unpack_f80(self.0), unpack_f80(o.0), FMT80, mode))
    }
    pub fn sqrt(self, mode: Round) -> (F80, u32) {
        pack_res(sqrt_unpacked(unpack_f80(self.0), FMT80, mode))
    }
    /// Round to an integral value in the 80-bit format, per `mode` (`FRNDINT`).
    #[must_use]
    pub fn round_to_int(self, mode: Round) -> F80 {
        let u = unpack_f80(self.0);
        if u.class != Class::Finite || u.exp >= 0 {
            return self; // non-finite or already integral
        }
        let (mag, _) = round_to_int_mag(u.sign, u.sig, u.exp, mode);
        if mag == 0 {
            return F80(u128::from(u.sign) << 79); // signed zero
        }
        // mag·2^0 is an exact integer; round(...RNE) just normalizes/packs it.
        let (r, _) = round(u.sign, mag, 0, false, FMT80, Round::Nearest);
        F80(pack_f80(&r))
    }
    /// `x - y·trunc/round(x/y)` (`FPREM`/`FPREM1`); returns the remainder and
    /// the low 3 quotient bits (`Q0,Q1,Q2`) x87 exposes via `C1,C3,C0`.
    #[must_use]
    pub fn remainder(self, y: F80, nearest: bool) -> (F80, u64) {
        let a = self.to_f64();
        let b = y.to_f64();
        let q = if nearest { (a / b).round_ties_even() } else { (a / b).trunc() };
        // Compute the remainder at 80-bit precision: x - y·q.
        let (yq, _) = y.mul(F80::from_f64_val(q), Round::Nearest);
        let (r, _) = self.sub(yq, Round::Nearest);
        (r, q.abs() as i64 as u64)
    }
    /// `FCHS`: flip the sign bit (exact for every value, including NaN/∞/0).
    #[must_use]
    pub fn neg(self) -> F80 {
        F80(self.0 ^ (1 << 79))
    }
    /// `FABS`: clear the sign bit.
    #[must_use]
    pub fn abs(self) -> F80 {
        F80(self.0 & !(1u128 << 79))
    }
    /// Ordered compare; `None` when unordered (either operand NaN).
    #[must_use]
    pub fn partial_cmp(self, o: F80) -> Option<core::cmp::Ordering> {
        compare(unpack_f80(self.0), unpack_f80(o.0))
    }
}

/// Whether to round a magnitude up by one ulp, given the lsb of the kept part,
/// the round bit, the sticky bit, the sign, and the mode. Shared by [`round`]
/// and the integer conversions so directed rounding is defined in one place.
#[allow(clippy::fn_params_excessive_bools)] // guard/round/sticky/sign are the IEEE rounding inputs
fn round_decision(lsb: bool, round_bit: bool, sticky: bool, sign: bool, mode: Round) -> bool {
    match mode {
        Round::Nearest => round_bit && (sticky || lsb),
        Round::Zero => false,
        Round::Up => (round_bit || sticky) && !sign,
        Round::Down => (round_bit || sticky) && sign,
    }
}

/// Round the magnitude `sig · 2^exp` to an integer under `mode`, returning the
/// integer magnitude and whether it was inexact. The caller guarantees the
/// result fits (for `exp ≥ 0`, `sig << exp` must not overflow `u128`).
fn round_to_int_mag(sign: bool, sig: u128, exp: i32, mode: Round) -> (u128, bool) {
    if exp >= 0 {
        return (sig << exp, false);
    }
    let shift = (-exp) as u32;
    if shift >= 128 {
        let sticky = sig != 0;
        return (u128::from(round_decision(false, false, sticky, sign, mode)), sticky);
    }
    let q = sig >> shift;
    let round_bit = (sig >> (shift - 1)) & 1 != 0;
    let sticky = sig & ((1u128 << (shift - 1)) - 1) != 0;
    let up = round_decision(q & 1 != 0, round_bit, sticky, sign, mode);
    (q + u128::from(up), round_bit || sticky)
}

fn round_unpacked(u: Unpacked, fmt: Fmt, mode: Round) -> (Unpacked, u32) {
    match u.class {
        Class::Finite => round(u.sign, u.sig, u.exp, false, fmt, mode),
        Class::Nan => {
            // Re-quiet/canonicalize into the destination width.
            let quiet = 1u128 << (fmt.prec - 2);
            (Unpacked::nan(u.sign, quiet), 0)
        }
        _ => (u, 0),
    }
}

fn compare(a: Unpacked, b: Unpacked) -> Option<core::cmp::Ordering> {
    use core::cmp::Ordering;
    if a.class == Class::Nan || b.class == Class::Nan {
        return None;
    }
    let val_sign = |v: &Unpacked| v.class != Class::Zero && v.sign;
    // ±0 compare equal.
    if a.class == Class::Zero && b.class == Class::Zero {
        return Some(Ordering::Equal);
    }
    let (as_, bs_) = (val_sign(&a), val_sign(&b));
    if as_ != bs_ {
        return Some(if as_ { Ordering::Less } else { Ordering::Greater });
    }
    // Same sign (or both non-negative). Compare magnitudes, then apply sign.
    let mag = magnitude_cmp(&a, &b);
    Some(if as_ { mag.reverse() } else { mag })
}

fn magnitude_cmp(a: &Unpacked, b: &Unpacked) -> core::cmp::Ordering {
    use core::cmp::Ordering;
    let rank = |v: &Unpacked| match v.class {
        Class::Zero => 0,
        Class::Finite => 1,
        Class::Inf => 2,
        Class::Nan => 3,
    };
    match rank(a).cmp(&rank(b)) {
        Ordering::Equal if a.class == Class::Finite => {
            let ea = a.exp + msb(a.sig) as i32;
            let eb = b.exp + msb(b.sig) as i32;
            match ea.cmp(&eb) {
                Ordering::Equal => {
                    // Align significands and compare.
                    let (sa, sb) = (a.sig, b.sig);
                    let na = msb(sa);
                    let nb = msb(sb);
                    let (sa, sb) = if na >= nb {
                        (sa, sb << (na - nb))
                    } else {
                        (sa << (nb - na), sb)
                    };
                    sa.cmp(&sb)
                }
                o => o,
            }
        }
        o => o,
    }
}

fn pack_res((u, f): (Unpacked, u32)) -> (F80, u32) {
    (F80(pack_f80(&u)), f)
}

// ---- SSE (f32/f64) format-typed helpers ------------------------------------

/// Which SSE arithmetic op to perform (min/max don't round, handled inline).
#[derive(Clone, Copy, Debug)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

/// One scalar f64 SSE op with directed rounding + IEEE flags.
#[must_use]
pub fn f64_op(a: u64, b: u64, op: Op, mode: Round) -> (u64, u32) {
    let (ua, ub) = (unpack_f64(a), unpack_f64(b));
    let (r, f) = match op {
        Op::Add => add_unpacked(ua, ub, FMT64, mode),
        Op::Sub => {
            let mut nb = ub;
            nb.sign = !nb.sign;
            add_unpacked(ua, nb, FMT64, mode)
        }
        Op::Mul => mul_unpacked(ua, ub, FMT64, mode),
        Op::Div => div_unpacked(ua, ub, FMT64, mode),
    };
    (pack_f64(&r), f)
}

/// One scalar f32 SSE op with directed rounding + IEEE flags.
#[must_use]
pub fn f32_op(a: u32, b: u32, op: Op, mode: Round) -> (u32, u32) {
    let (ua, ub) = (unpack_f32(a), unpack_f32(b));
    let (r, f) = match op {
        Op::Add => add_unpacked(ua, ub, FMT32, mode),
        Op::Sub => {
            let mut nb = ub;
            nb.sign = !nb.sign;
            add_unpacked(ua, nb, FMT32, mode)
        }
        Op::Mul => mul_unpacked(ua, ub, FMT32, mode),
        Op::Div => div_unpacked(ua, ub, FMT32, mode),
    };
    (pack_f32(&r), f)
}

#[must_use]
pub fn f64_sqrt(a: u64, mode: Round) -> (u64, u32) {
    let (r, f) = sqrt_unpacked(unpack_f64(a), FMT64, mode);
    (pack_f64(&r), f)
}

#[must_use]
pub fn f32_sqrt(a: u32, mode: Round) -> (u32, u32) {
    let (r, f) = sqrt_unpacked(unpack_f32(a), FMT32, mode);
    (pack_f32(&r), f)
}

/// Convert a signed integer to f64/f32 with directed rounding.
#[must_use]
pub fn i64_to_f64(v: i64, mode: Round) -> (u64, u32) {
    if v == 0 {
        return (0, 0);
    }
    let (r, f) = round(v < 0, u128::from(v.unsigned_abs()), 0, false, FMT64, mode);
    (pack_f64(&r), f)
}

#[must_use]
pub fn i64_to_f32(v: i64, mode: Round) -> (u32, u32) {
    if v == 0 {
        return (0, 0);
    }
    let (r, f) = round(v < 0, u128::from(v.unsigned_abs()), 0, false, FMT32, mode);
    (pack_f32(&r), f)
}

/// f64→f32 narrowing with directed rounding (`CVTSD2SS`).
#[must_use]
pub fn f64_to_f32(a: u64, mode: Round) -> (u32, u32) {
    let (r, f) = round_unpacked(unpack_f64(a), FMT32, mode);
    (pack_f32(&r), f)
}

/// f32→f64 widening (exact; never rounds a finite value).
#[must_use]
pub fn f32_to_f64(a: u32) -> u64 {
    let (r, _) = round_unpacked(unpack_f32(a), FMT64, Round::Nearest);
    pack_f64(&r)
}

/// Convert f64/f32 to a signed 64-bit integer per `mode`; NaN/overflow yields
/// the x86 integer-indefinite `i64::MIN`.
#[must_use]
pub fn f64_to_i64(a: u64, mode: Round) -> i64 {
    F80::from_f64(a).to_i64_round(mode)
}
#[must_use]
pub fn f32_to_i64(a: u32, mode: Round) -> i64 {
    F80::from_f32(a).to_i64_round(mode)
}

/// Ordered compare of two f64 values; `None` if unordered.
#[must_use]
pub fn f64_cmp(a: u64, b: u64) -> Option<core::cmp::Ordering> {
    compare(unpack_f64(a), unpack_f64(b))
}
#[must_use]
pub fn f32_cmp(a: u32, b: u32) -> Option<core::cmp::Ordering> {
    compare(unpack_f32(a), unpack_f32(b))
}

#[cfg(test)]
mod tests;
