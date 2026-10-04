//! A small, dependency-free software floating-point core, precise enough to be
//! *bit-exact* with real x86 hardware (SSE and x87).
//!
//! The x86 interpreter needs three things host `f32`/`f64` arithmetic can't
//! give it:
//!
//! * **true 80-bit x87 extended precision** — the x87 register stack computes
//!   in a 64-bit-significand format with a 15-bit exponent, optionally rounded
//!   to 24 or 53 significand bits (precision control) while keeping the wide
//!   exponent range;
//! * **directed rounding** — `MXCSR`/the x87 control word select
//!   round-toward-zero/±∞; re-rounding a host result double-rounds;
//! * **x86's exception semantics** — the IEEE flags, x86's tininess rule (after
//!   rounding), `DAZ`/`FTZ`, and the x86 NaN-propagation rules (which differ
//!   between SSE and x87, so they are left to the callers: every arithmetic
//!   core here takes non-NaN operands).
//!
//! Every operation is carried out on an *unpacked* value ([`Fp`]: `sign · sig ·
//! 2^exp`, `sig` an integer with all the bits the op produces) and rounded
//! exactly once, by [`round`], into the destination [`Fmt`] under the
//! requested [`Round`] mode.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
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
/// `OE`,`UE`,`PE`).
pub const INVALID: u32 = 0x01;
pub const DENORMAL: u32 = 0x02;
pub const DIVZERO: u32 = 0x04;
pub const OVERFLOW: u32 = 0x08;
pub const UNDERFLOW: u32 = 0x10;
pub const INEXACT: u32 = 0x20;

/// A binary floating-point format: significand precision (including the leading
/// bit) and the exponent of the leading bit for the largest/smallest normals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fmt {
    /// Significand bits, counting the implicit/explicit integer bit (24/53/64).
    pub prec: u32,
    /// Leading-bit exponent of the largest finite normal (127/1023/16383).
    pub emax: i32,
    /// Leading-bit exponent of the smallest normal (`1 - emax`).
    pub emin: i32,
}

pub const FMT32: Fmt = Fmt {
    prec: 24,
    emax: 127,
    emin: -126,
};
pub const FMT64: Fmt = Fmt {
    prec: 53,
    emax: 1023,
    emin: -1022,
};
/// x87 80-bit extended: 64-bit significand with an *explicit* integer bit,
/// 15-bit exponent (bias 16383).
pub const FMT80: Fmt = Fmt {
    prec: 64,
    emax: 16383,
    emin: -16382,
};

/// The x87 format for a precision-control setting (`CW[9:8]`): the 80-bit
/// exponent range with a 24-, 53- or 64-bit significand (`01`, reserved,
/// behaves as 64 on hardware).
#[must_use]
pub fn x87_fmt(pc: u16) -> Fmt {
    let prec = match pc & 3 {
        0 => 24,
        2 => 53,
        _ => 64,
    };
    Fmt { prec, ..FMT80 }
}

/// The category of a value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Class {
    Zero,
    Finite,
    Inf,
    Nan,
}

/// The quiet bit of a NaN fraction in [`Fp::sig`] (x87 significand bit 62).
pub const QUIET: u128 = 1 << 62;

/// An exact, unpacked value: `(-1)^sign · sig · 2^exp` for `Finite` (`sig`
/// nonzero, no assumed normalization). For `Nan`, `sig` holds the fraction
/// left-aligned as in the x87 format — bit 62 is the quiet bit, so converting
/// between formats shifts payloads exactly as x86 does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp {
    pub sign: bool,
    pub class: Class,
    pub sig: u128,
    pub exp: i32,
}

/// The x86 "real indefinite": the default QNaN an invalid operation yields
/// (negative sign, only the quiet bit set).
pub const INDEFINITE: Fp = Fp {
    sign: true,
    class: Class::Nan,
    sig: QUIET,
    exp: 0,
};

impl Fp {
    #[must_use]
    pub const fn zero(sign: bool) -> Self {
        Self {
            sign,
            class: Class::Zero,
            sig: 0,
            exp: 0,
        }
    }
    #[must_use]
    pub const fn inf(sign: bool) -> Self {
        Self {
            sign,
            class: Class::Inf,
            sig: 0,
            exp: 0,
        }
    }
    #[must_use]
    pub const fn finite(sign: bool, sig: u128, exp: i32) -> Self {
        if sig == 0 {
            Self::zero(sign)
        } else {
            Self {
                sign,
                class: Class::Finite,
                sig,
                exp,
            }
        }
    }
    #[must_use]
    pub fn is_nan(&self) -> bool {
        self.class == Class::Nan
    }
    #[must_use]
    pub fn is_snan(&self) -> bool {
        self.class == Class::Nan && self.sig & QUIET == 0
    }
    /// The NaN with its quiet bit set.
    #[must_use]
    pub fn quieted(self) -> Self {
        Self {
            sig: self.sig | QUIET,
            ..self
        }
    }
    /// Exponent of the leading set bit (finite only).
    #[must_use]
    pub fn lead_exp(&self) -> i32 {
        self.exp + msb(self.sig) as i32
    }
    /// Whether this finite value is below `fmt`'s normal range (a denormal
    /// *input* of that format).
    #[must_use]
    pub fn is_denormal(&self, fmt: Fmt) -> bool {
        self.class == Class::Finite && self.lead_exp() < fmt.emin
    }
    #[must_use]
    pub fn neg(self) -> Self {
        Self {
            sign: !self.sign,
            ..self
        }
    }
}

/// Position of the most-significant set bit (0-based); `sig` must be nonzero.
#[inline]
#[must_use]
pub fn msb(sig: u128) -> u32 {
    sig.ilog2()
}

// ---- unpack: format bits -> Fp (exact) ---------------------------------------

#[must_use]
pub fn unpack_f32(bits: u32) -> Fp {
    let sign = bits >> 31 != 0;
    let e = (bits >> 23) & 0xff;
    let mant = u128::from(bits & 0x7f_ffff);
    match e {
        0xff if mant == 0 => Fp::inf(sign),
        0xff => Fp {
            sign,
            class: Class::Nan,
            sig: mant << 40,
            exp: 0,
        },
        0 => Fp::finite(sign, mant, -149),
        _ => Fp::finite(sign, mant | (1 << 23), e as i32 - 150),
    }
}

#[must_use]
pub fn unpack_f64(bits: u64) -> Fp {
    let sign = bits >> 63 != 0;
    let e = (bits >> 52) & 0x7ff;
    let mant = u128::from(bits & 0xf_ffff_ffff_ffff);
    match e {
        0x7ff if mant == 0 => Fp::inf(sign),
        0x7ff => Fp {
            sign,
            class: Class::Nan,
            sig: mant << 11,
            exp: 0,
        },
        0 => Fp::finite(sign, mant, -1074),
        _ => Fp::finite(sign, mant | (1 << 52), e as i32 - 1075),
    }
}

/// Unpack an x87 80-bit value (low 80 bits of `bits`). The integer bit is
/// explicit, so the encodings x87 arithmetic rejects (unnormals, pseudo-NaN,
/// pseudo-infinity — see [`f80_unsupported`]) are representable; they unpack
/// by their face value here and the caller decides. Denormals and
/// pseudo-denormals (exponent field 0) both mean `sig · 2^-16445`.
#[must_use]
pub fn unpack_f80(bits: u128) -> Fp {
    let sign = (bits >> 79) & 1 != 0;
    let e = ((bits >> 64) & 0x7fff) as i32;
    let sig = bits & 0xffff_ffff_ffff_ffff;
    match e {
        0x7fff if sig & 0x7fff_ffff_ffff_ffff == 0 => Fp::inf(sign),
        0x7fff => Fp {
            sign,
            class: Class::Nan,
            sig: sig & 0x7fff_ffff_ffff_ffff,
            exp: 0,
        },
        0 => Fp::finite(sign, sig, -16445),
        _ => Fp::finite(sign, sig, e - 16446),
    }
}

/// Whether an 80-bit pattern is one of the encodings the x87 treats as an
/// invalid operand: an unnormal (nonzero exponent, integer bit clear) or a
/// pseudo-infinity/pseudo-NaN (maximal exponent, integer bit clear).
#[must_use]
pub fn f80_unsupported(bits: u128) -> bool {
    let e = (bits >> 64) & 0x7fff;
    e != 0 && (bits >> 63) & 1 == 0
}

// ---- pack: rounded Fp -> format bits -------------------------------------------
//
// A Finite value must already be representable in the destination format (the
// output of `round`): its leading bit at or above `emin` (normal) or, when
// below, its significand aligned at the format's denormal granularity.

/// Left-justify a finite significand so its top bit lands at `prec - 1`,
/// returning the significand and the leading-bit exponent.
fn justify(v: &Fp, prec: u32) -> (u128, i32) {
    let m = msb(v.sig);
    let e = v.exp + m as i32;
    let sig = if m < prec {
        v.sig << (prec - 1 - m)
    } else {
        v.sig >> (m - (prec - 1))
    };
    (sig, e)
}

#[must_use]
pub fn pack_f32(v: &Fp) -> u32 {
    let s = u32::from(v.sign) << 31;
    match v.class {
        Class::Zero => s,
        Class::Inf => s | 0x7f80_0000,
        Class::Nan => s | 0x7f80_0000 | ((v.sig >> 40) as u32 & 0x7f_ffff),
        Class::Finite => {
            let (sig, e) = justify(v, 24);
            if e < FMT32.emin {
                s | ((v.sig << (v.exp + 149)) as u32 & 0x7f_ffff)
            } else {
                s | (((e + 127) as u32) << 23) | (sig as u32 & 0x7f_ffff)
            }
        }
    }
}

#[must_use]
pub fn pack_f64(v: &Fp) -> u64 {
    let s = u64::from(v.sign) << 63;
    match v.class {
        Class::Zero => s,
        Class::Inf => s | 0x7ff0_0000_0000_0000,
        Class::Nan => s | 0x7ff0_0000_0000_0000 | ((v.sig >> 11) as u64 & 0xf_ffff_ffff_ffff),
        Class::Finite => {
            let (sig, e) = justify(v, 53);
            if e < FMT64.emin {
                s | ((v.sig << (v.exp + 1074)) as u64 & 0xf_ffff_ffff_ffff)
            } else {
                s | (((e + 1023) as u64) << 52) | (sig as u64 & 0xf_ffff_ffff_ffff)
            }
        }
    }
}

#[must_use]
pub fn pack_f80(v: &Fp) -> u128 {
    let s = u128::from(v.sign) << 79;
    match v.class {
        Class::Zero => s,
        Class::Inf => s | (0x7fff << 64) | (1 << 63),
        Class::Nan => s | (0x7fff << 64) | (1 << 63) | (v.sig & 0x7fff_ffff_ffff_ffff),
        Class::Finite => {
            let (sig, e) = justify(v, 64);
            if e < FMT80.emin {
                s | ((v.sig << (v.exp + 16445)) & 0xffff_ffff_ffff_ffff)
            } else {
                s | (((e + 16383) as u128) << 64) | sig
            }
        }
    }
}

// ---- the one rounding point --------------------------------------------------

/// The result of [`round`].
#[derive(Clone, Copy, Debug)]
pub struct Rounded {
    pub v: Fp,
    /// `INEXACT`/`UNDERFLOW`/`OVERFLOW` as IEEE (masked) reports them.
    pub flags: u32,
    /// The magnitude was rounded up (x87 reports this in `C1`).
    pub up: bool,
    /// The result is tiny — nonzero and below the normal range *after*
    /// rounding (x86 detects tininess after rounding). `FTZ` flushes such
    /// results.
    pub tiny: bool,
}

impl Rounded {
    const fn exact(v: Fp) -> Self {
        Self {
            v,
            flags: 0,
            up: false,
            tiny: false,
        }
    }
    const fn invalid() -> Self {
        Self {
            v: INDEFINITE,
            flags: INVALID,
            up: false,
            tiny: false,
        }
    }
}

/// Whether to round a magnitude up by one ulp, given the lsb of the kept part,
/// the round bit, the sticky bit, the sign, and the mode.
#[allow(clippy::fn_params_excessive_bools)] // lsb/round/sticky/sign are the IEEE rounding inputs
#[must_use]
pub fn round_decision(lsb: bool, round_bit: bool, sticky: bool, sign: bool, mode: Round) -> bool {
    match mode {
        Round::Nearest => round_bit && (sticky || lsb),
        Round::Zero => false,
        Round::Up => (round_bit || sticky) && !sign,
        Round::Down => (round_bit || sticky) && sign,
    }
}

/// Split `sig` at bit `sh` into (kept, round bit, sticky).
fn split(sig: u128, sh: i32) -> (u128, bool, bool) {
    if sh <= 0 {
        (sig << (-sh) as u32, false, false)
    } else if sh >= 129 {
        (0, false, sig != 0)
    } else if sh == 128 {
        (0, sig >> 127 != 0, sig & (u128::MAX >> 1) != 0)
    } else {
        let sh = sh as u32;
        (
            sig >> sh,
            (sig >> (sh - 1)) & 1 != 0,
            sig & ((1u128 << (sh - 1)) - 1) != 0,
        )
    }
}

/// Round the exact finite value `(-1)^sign · sig · 2^exp` (with `sticky`
/// recording nonzero bits already dropped below bit 0 of `sig`) into `fmt`
/// under `mode`.
///
/// This is where precision is lost — exactly once — so directed rounding is
/// correct: the true result's round/sticky bits and sign drive the decision,
/// never a previously-rounded intermediate.
#[must_use]
pub fn round(sign: bool, sig: u128, exp: i32, sticky: bool, fmt: Fmt, mode: Round) -> Rounded {
    if sig == 0 {
        // Only a sub-ulp remnant (sticky) is left: treat it as an
        // infinitesimal nonzero value.
        if !sticky {
            return Rounded::exact(Fp::zero(sign));
        }
        let up = round_decision(false, false, true, sign, mode);
        let ulp = fmt.emin - (fmt.prec as i32 - 1);
        let v = if up {
            Fp::finite(sign, 1, ulp)
        } else {
            Fp::zero(sign)
        };
        return Rounded {
            v,
            flags: INEXACT | UNDERFLOW,
            up,
            tiny: true,
        };
    }
    let p = fmt.prec as i32;
    let e = exp + msb(sig) as i32; // leading-bit exponent

    // Tininess after rounding: round to `prec` bits with an unbounded
    // exponent and see whether the result is below the normal range.
    let tiny = match e.cmp(&(fmt.emin - 1)) {
        core::cmp::Ordering::Less => true,
        core::cmp::Ordering::Equal => {
            let (q, r, s) = split(sig, (e - (p - 1)) - exp);
            let up = round_decision(q & 1 != 0, r, s || sticky, sign, mode);
            // Rounds up to 2^emin exactly when q is all ones and it rounds up.
            !(up && q + 1 == 1u128 << p)
        }
        core::cmp::Ordering::Greater => false,
    };

    let ulp = e.max(fmt.emin) - (p - 1);
    let (mut q, r, s) = split(sig, ulp - exp);
    let s = s || sticky;
    let inexact = r || s;
    let up = round_decision(q & 1 != 0, r, s, sign, mode);
    let mut ulp_exp = ulp;
    if up {
        q += 1;
        if q == 1u128 << p {
            // Carry out of the top: renormalize (the dropped bit is zero).
            q >>= 1;
            ulp_exp += 1;
        }
    }
    let mut flags = if inexact { INEXACT } else { 0 };
    if q != 0 && ulp_exp + msb(q) as i32 > fmt.emax {
        flags |= OVERFLOW | INEXACT;
        let to_inf = match mode {
            Round::Nearest => true,
            Round::Zero => false,
            Round::Up => !sign,
            Round::Down => sign,
        };
        let v = if to_inf {
            Fp::inf(sign)
        } else {
            Fp::finite(sign, (1u128 << p) - 1, fmt.emax - (p - 1))
        };
        return Rounded {
            v,
            flags,
            up: to_inf,
            tiny: false,
        };
    }
    if tiny && inexact {
        flags |= UNDERFLOW;
    }
    Rounded {
        v: Fp::finite(sign, q, ulp_exp),
        flags,
        up: up && inexact,
        tiny,
    }
}

/// Round an unpacked value of any class into `fmt` (NaNs/inf/zero pass
/// through unchanged).
#[must_use]
pub fn round_fp(v: Fp, fmt: Fmt, mode: Round) -> Rounded {
    match v.class {
        Class::Finite => round(v.sign, v.sig, v.exp, false, fmt, mode),
        _ => Rounded::exact(v),
    }
}

// ---- arithmetic cores (non-NaN operands) -------------------------------------

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

/// `a + b` for non-NaN operands (subtract by negating `b`).
#[must_use]
pub fn add(a: Fp, b: Fp, fmt: Fmt, mode: Round) -> Rounded {
    use Class::{Finite, Inf, Zero};
    match (a.class, b.class) {
        (Inf, Inf) => {
            if a.sign == b.sign {
                Rounded::exact(a)
            } else {
                Rounded::invalid() // (+∞) + (−∞)
            }
        }
        (Inf, _) => Rounded::exact(a),
        (_, Inf) => Rounded::exact(b),
        (Zero, Zero) => {
            // −0 + −0 = −0; every other zero-sum is +0 except toward −∞.
            let sign = if a.sign == b.sign {
                a.sign
            } else {
                mode == Round::Down
            };
            Rounded::exact(Fp::zero(sign))
        }
        (Zero, _) => round_fp(b, fmt, mode),
        (_, Zero) => round_fp(a, fmt, mode),
        (Finite, Finite) => add_finite(a, b, fmt, mode),
        _ => Rounded::invalid(),
    }
}

/// Core finite add/subtract. Both significands are left-justified to bit 63,
/// widened to bit 126 (leaving carry room above and 63 alignment bits below),
/// aligned by exponent with a sticky bit, then combined and rounded once.
fn add_finite(a: Fp, b: Fp, fmt: Fmt, mode: Round) -> Rounded {
    let (sa, ea) = norm63(a.sig, a.exp);
    let (sb, eb) = norm63(b.sig, b.exp);
    let (hs, he, hsign, ls, le, lsign) = if ea >= eb {
        (sa, ea, a.sign, sb, eb, b.sign)
    } else {
        (sb, eb, b.sign, sa, ea, a.sign)
    };
    let d = (he - le) as u32;
    let big = hs << 63; // MSB at bit 126
    let (small, sticky) = shr_sticky(ls << 63, d);
    if hsign == lsign {
        round(hsign, big + small, he - 126, sticky, fmt, mode)
    } else if d == 0 {
        match big.cmp(&small) {
            core::cmp::Ordering::Equal => Rounded::exact(Fp::zero(mode == Round::Down)),
            core::cmp::Ordering::Greater => round(hsign, big - small, he - 126, false, fmt, mode),
            core::cmp::Ordering::Less => round(lsign, small - big, he - 126, false, fmt, mode),
        }
    } else {
        // `big` dominates. The exact small operand is `small + frac` (frac < 1
        // unit, present iff sticky): subtract the borrow, keeping the sticky
        // for the remnant. Scale up one bit first so the borrowed unit stays
        // well below the rounding position.
        let diff = ((big - small) << 1) - u128::from(sticky);
        round(hsign, diff, he - 127, sticky, fmt, mode)
    }
}

/// `a · b` for non-NaN operands.
#[must_use]
pub fn mul(a: Fp, b: Fp, fmt: Fmt, mode: Round) -> Rounded {
    use Class::{Finite, Inf, Zero};
    let sign = a.sign ^ b.sign;
    match (a.class, b.class) {
        (Inf, Zero) | (Zero, Inf) => Rounded::invalid(),
        (Inf, _) | (_, Inf) => Rounded::exact(Fp::inf(sign)),
        (Zero, _) | (_, Zero) => Rounded::exact(Fp::zero(sign)),
        (Finite, Finite) => {
            // Significands ≤64 bits each → product ≤128 bits, exact in u128.
            round(sign, a.sig * b.sig, a.exp + b.exp, false, fmt, mode)
        }
        _ => Rounded::invalid(),
    }
}

/// `a / b` for non-NaN operands. A finite nonzero dividend over zero reports
/// `DIVZERO` (and an infinite result).
#[must_use]
pub fn div(a: Fp, b: Fp, fmt: Fmt, mode: Round) -> Rounded {
    use Class::{Finite, Inf, Zero};
    let sign = a.sign ^ b.sign;
    match (a.class, b.class) {
        (Inf, Inf) | (Zero, Zero) => Rounded::invalid(),
        (Inf, _) => Rounded::exact(Fp::inf(sign)),
        (_, Inf) | (Zero, _) => Rounded::exact(Fp::zero(sign)),
        (_, Zero) => Rounded {
            v: Fp::inf(sign),
            flags: DIVZERO,
            up: false,
            tiny: false,
        },
        (Finite, Finite) => {
            // Long division producing prec+3 quotient bits + sticky.
            let na = msb(a.sig);
            let nb = msb(b.sig);
            let da = a.sig << (126 - na);
            let db = b.sig << (126 - nb);
            let a_exp = a.exp + na as i32 - 126;
            let b_exp = b.exp + nb as i32 - 126;
            let nbits = fmt.prec + 3;
            let mut rem = da;
            let mut q: u128 = 0;
            for _ in 0..nbits {
                q <<= 1;
                if rem >= db {
                    rem -= db;
                    q |= 1;
                }
                rem <<= 1;
            }
            round(
                sign,
                q,
                a_exp - b_exp - (nbits as i32 - 1),
                rem != 0,
                fmt,
                mode,
            )
        }
        _ => Rounded::invalid(),
    }
}

/// Integer square root of a `u128` (floor).
fn isqrt128(n: u128) -> u128 {
    if n == 0 {
        return 0;
    }
    let mut x = 1u128 << (msb(n) / 2 + 1);
    loop {
        let nx = x.midpoint(n / x);
        if nx >= x {
            break;
        }
        x = nx;
    }
    while x > 0 && x > n / x {
        x -= 1;
    }
    while (x + 1) <= n / (x + 1) {
        x += 1;
    }
    x
}

/// `√a` for a non-NaN operand (negative nonzero → invalid; `√-0 = -0`).
#[must_use]
pub fn sqrt(a: Fp, fmt: Fmt, mode: Round) -> Rounded {
    match a.class {
        Class::Zero => Rounded::exact(a),
        Class::Inf if a.sign => Rounded::invalid(),
        Class::Inf => Rounded::exact(a),
        Class::Finite if a.sign => Rounded::invalid(),
        Class::Finite => {
            // value = sig · 2^exp. Make exp even, then scale sig (by an even
            // amount) so its integer square root q has exactly `prec` bits:
            // radicand in [2^(2·prec-2), 2^(2·prec)) — at most 128 bits.
            let mut sig = a.sig;
            let mut exp = a.exp;
            if exp & 1 != 0 {
                sig <<= 1;
                exp -= 1;
            }
            // Always take a 64-bit root (exact remainder included): rounding
            // that to a narrower `fmt` below is still a single rounding.
            let target = 126;
            let mut k = target - msb(sig) as i32;
            if k & 1 != 0 {
                k += 1;
            }
            let (sig, exp) = (sig << k, exp - k);
            let q = isqrt128(sig);
            let r = sig - q * q;
            // The exact root lies in [q, q+1) and is never exactly q + 1/2:
            // it is above the midpoint iff r > q.
            let round_bit = u128::from(r > q);
            round(false, (q << 1) | round_bit, exp / 2 - 1, r != 0, fmt, mode)
        }
        Class::Nan => Rounded::invalid(),
    }
}

/// Round a non-NaN value to an integral value (in the same format) per
/// `mode` (`FRNDINT`, `ROUNDSS`). Reports `INEXACT` and the round-up.
#[must_use]
pub fn round_to_int(a: Fp, mode: Round) -> Rounded {
    if a.class != Class::Finite || a.exp >= 0 {
        return Rounded::exact(a);
    }
    let (q, r, s) = split(a.sig, -a.exp);
    let up = round_decision(q & 1 != 0, r, s, a.sign, mode);
    let mag = q + u128::from(up);
    Rounded {
        v: Fp::finite(a.sign, mag, 0),
        flags: if r || s { INEXACT } else { 0 },
        up: up && (r || s),
        tiny: false,
    }
}

/// Convert a non-NaN value to a signed integer of `bits` bits (16/32/64),
/// rounding per `mode`; `None` when it doesn't fit (the caller produces the
/// "integer indefinite" with `INVALID`). The flags carry `INEXACT`.
#[must_use]
pub fn to_int(a: Fp, bits: u32, mode: Round) -> Option<(i64, u32, bool)> {
    match a.class {
        Class::Zero => Some((0, 0, false)),
        Class::Finite => {
            if a.lead_exp() > 64 {
                return None;
            }
            let r = round_to_int(a, mode);
            let mag = if r.v.class == Class::Zero {
                0
            } else {
                r.v.sig << r.v.exp.max(0)
            };
            let lim = 1u128 << (bits - 1);
            if (a.sign && mag > lim) || (!a.sign && mag >= lim) {
                return None;
            }
            let v = if a.sign {
                (mag as i64).wrapping_neg()
            } else {
                mag as i64
            };
            Some((v, r.flags, r.up))
        }
        _ => None,
    }
}

/// Exact unpacked value of a signed integer.
#[must_use]
pub fn from_int(v: i64) -> Fp {
    Fp::finite(v < 0, u128::from(v.unsigned_abs()), 0)
}

/// Ordered compare of two non-NaN values (`±0` equal).
#[must_use]
pub fn compare(a: &Fp, b: &Fp) -> core::cmp::Ordering {
    use core::cmp::Ordering;
    let neg = |v: &Fp| v.class != Class::Zero && v.sign;
    if a.class == Class::Zero && b.class == Class::Zero {
        return Ordering::Equal;
    }
    let (na, nb) = (neg(a), neg(b));
    if na != nb {
        return if na {
            Ordering::Less
        } else {
            Ordering::Greater
        };
    }
    let mag = magnitude_cmp(a, b);
    if na { mag.reverse() } else { mag }
}

fn magnitude_cmp(a: &Fp, b: &Fp) -> core::cmp::Ordering {
    use core::cmp::Ordering;
    let rank = |v: &Fp| match v.class {
        Class::Zero => 0,
        Class::Finite => 1,
        Class::Inf | Class::Nan => 2,
    };
    match rank(a).cmp(&rank(b)) {
        Ordering::Equal if a.class == Class::Finite => match a.lead_exp().cmp(&b.lead_exp()) {
            Ordering::Equal => {
                let (na, nb) = (msb(a.sig), msb(b.sig));
                let (sa, sb) = if na >= nb {
                    (a.sig, b.sig << (na - nb))
                } else {
                    (a.sig << (nb - na), b.sig)
                };
                sa.cmp(&sb)
            }
            o => o,
        },
        o => o,
    }
}

// ---- x87 80-bit value type -------------------------------------------------------

/// An x87 80-bit extended value, stored as its 80-bit encoding in the low bits.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct F80(pub u128);

impl core::fmt::Debug for F80 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "F80({:#022x})", self.0 & 0xffff_ffff_ffff_ffff_ffff)
    }
}

impl F80 {
    pub const ZERO: F80 = F80(0);
    pub const ONE: F80 = F80(0x3fff_8000_0000_0000_0000);
    /// The QNaN floating-point indefinite.
    pub const INDEFINITE: F80 = F80(0xffff_c000_0000_0000_0000);

    #[must_use]
    pub fn unpack(self) -> Fp {
        unpack_f80(self.0)
    }
    #[must_use]
    pub fn pack(v: &Fp) -> Self {
        F80(pack_f80(v))
    }
    /// The exact (widening) conversion of an `f64` bit pattern; NaNs keep their
    /// payload and quiet bit (callers decide about SNaN signaling).
    #[cfg(test)]
    #[must_use]
    pub fn from_f64(bits: u64) -> Self {
        Self::pack(&unpack_f64(bits))
    }
    #[must_use]
    pub fn sign(self) -> bool {
        (self.0 >> 79) & 1 != 0
    }
    #[must_use]
    pub fn exp_field(self) -> u32 {
        ((self.0 >> 64) & 0x7fff) as u32
    }
    #[must_use]
    pub fn mant(self) -> u64 {
        self.0 as u64
    }
    /// Test helper: the nearest `f64` (round-to-nearest).
    #[cfg(test)]
    #[must_use]
    pub fn to_f64(self) -> f64 {
        f64::from_bits(pack_f64(&round_fp(self.unpack(), FMT64, Round::Nearest).v))
    }
    /// Test helper: an `f64` value widened exactly.
    #[cfg(test)]
    #[must_use]
    pub fn from_f64_val(x: f64) -> Self {
        Self::from_f64(x.to_bits())
    }
    /// `FCHS`: flip the sign bit.
    #[must_use]
    pub fn neg(self) -> F80 {
        F80(self.0 ^ (1 << 79))
    }
    /// `FABS`: clear the sign bit.
    #[must_use]
    pub fn abs(self) -> F80 {
        F80(self.0 & !(1u128 << 79))
    }
}

// ---- SSE scalar helpers (f32/f64 bit patterns) ----------------------------------

/// The `MXCSR`-derived environment of an SSE floating-point operation.
#[derive(Clone, Copy, Debug)]
pub struct Mx {
    pub mode: Round,
    /// Denormals-are-zero: denormal inputs read as signed zeros (no `DE`).
    pub daz: bool,
    /// Flush-to-zero: tiny results become signed zeros (with `UE`/`PE`), when
    /// underflow is masked.
    pub ftz: bool,
}

impl Mx {
    #[must_use]
    pub fn from_mxcsr(mxcsr: u32) -> Self {
        Self {
            mode: Round::from_x86(mxcsr >> 13),
            daz: mxcsr & (1 << 6) != 0,
            ftz: mxcsr & (1 << 15) != 0 && mxcsr & (1 << 11) != 0,
        }
    }

    /// Apply `DAZ` to an unpacked input, returning it and its `DE` flag.
    #[must_use]
    pub fn input(self, v: Fp, fmt: Fmt) -> (Fp, u32) {
        if v.is_denormal(fmt) {
            if self.daz {
                (Fp::zero(v.sign), 0)
            } else {
                (v, DENORMAL)
            }
        } else {
            (v, 0)
        }
    }

    /// Apply `FTZ` to a rounded result.
    #[must_use]
    pub fn output(self, r: Rounded) -> (Fp, u32) {
        if self.ftz && r.tiny && r.v.class == Class::Finite {
            (Fp::zero(r.v.sign), r.flags | UNDERFLOW | INEXACT)
        } else {
            (r.v, r.flags)
        }
    }
}

/// An SSE binary arithmetic operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op {
    Add,
    Sub,
    Mul,
    Div,
}

/// SSE NaN propagation for a two-operand op: the first (destination) operand
/// if it is a NaN, else the second — quieted; `INVALID` if either is an SNaN.
#[must_use]
pub fn sse_nan2(a: Fp, b: Fp) -> Option<(Fp, u32)> {
    if !a.is_nan() && !b.is_nan() {
        return None;
    }
    let flags = if a.is_snan() || b.is_snan() {
        INVALID
    } else {
        0
    };
    let n = if a.is_nan() { a } else { b };
    Some((n.quieted(), flags))
}

/// One SSE arithmetic op on unpacked operands of format `fmt`, with x86 NaN
/// rules, `DAZ`/`FTZ` and the full flag set (`DE` included).
#[must_use]
pub fn sse_arith(op: Op, a: Fp, b: Fp, fmt: Fmt, mx: Mx) -> (Fp, u32) {
    if let Some(r) = sse_nan2(a, b) {
        return r;
    }
    let (a, da) = mx.input(a, fmt);
    let (b, db) = mx.input(b, fmt);
    let r = match op {
        Op::Add => add(a, b, fmt, mx.mode),
        Op::Sub => add(a, b.neg(), fmt, mx.mode),
        Op::Mul => mul(a, b, fmt, mx.mode),
        Op::Div => div(a, b, fmt, mx.mode),
    };
    let (v, f) = mx.output(r);
    // An invalid operation reports only IE (and DE), never PE.
    (v, f | da | db)
}

/// One SSE square root on an unpacked operand.
#[must_use]
pub fn sse_sqrt(a: Fp, fmt: Fmt, mx: Mx) -> (Fp, u32) {
    if a.is_nan() {
        return (a.quieted(), if a.is_snan() { INVALID } else { 0 });
    }
    let (a, da) = mx.input(a, fmt);
    let (v, f) = mx.output(sqrt(a, fmt, mx.mode));
    (v, f | da)
}

/// Scalar `f64` SSE op on bit patterns.
#[cfg(test)]
#[must_use]
pub fn f64_op(a: u64, b: u64, op: Op, mx: Mx) -> (u64, u32) {
    let (v, f) = sse_arith(op, unpack_f64(a), unpack_f64(b), FMT64, mx);
    (pack_f64(&v), f)
}

/// Scalar `f32` SSE op on bit patterns.
#[cfg(test)]
#[must_use]
pub fn f32_op(a: u32, b: u32, op: Op, mx: Mx) -> (u32, u32) {
    let (v, f) = sse_arith(op, unpack_f32(a), unpack_f32(b), FMT32, mx);
    (pack_f32(&v), f)
}

#[cfg(test)]
mod tests;
