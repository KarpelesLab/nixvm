//! AArch64 floating-point arithmetic, bit-exact with the ARM ARM pseudocode.
//!
//! Every operation takes raw IEEE bit patterns (`u64`, low `N` bits used) plus a
//! [`Fmt`] (half/single/double) and an [`Env`] carrying `FPCR` (rounding mode,
//! `FZ`/`FZ16` flush-to-zero, `DN` default-NaN, `AHP` alternative half
//! precision) and the `FPSR` cumulative exception flags it ORs into
//! (`IOC`/`DZC`/`OFC`/`UFC`/`IXC`/`IDC`). The structure mirrors the ARM ARM's
//! shared pseudocode: [`unpack`] is `FPUnpack`/`FPUnpackCV`, [`round`] is
//! `FPRoundBase` (tininess detected *before* rounding, as ARM does), and each
//! public op follows its namesake (`FPAdd`, `FPMulAdd`, `FPRecipEstimate`, …)
//! including the NaN-propagation priority of `FPProcessNaNs`/`FPProcessNaNs3`.
//!
//! The arithmetic core is exact: finite values unpack to `mant · 2^exp`, an
//! operation produces its exact result (or, for division/square root and
//! far-apart addends, the result truncated to ≥ 60 significant bits plus a
//! sticky bit) and [`round`] rounds once. That makes directed rounding, flush-
//! to-zero and the exception flags correct by construction.
//!
//! Speed: the common case — round-to-nearest, `FZ` off, a normal finite
//! result, and `FPSR.IXC` already set (it is sticky, and the very first
//! inexact operation a program performs sets it) — is computed with host
//! `f32`/`f64` arithmetic, which is correctly rounded and so bit-identical.
//! Everything else (NaNs, infinities, zero/subnormal results, other rounding
//! modes, flush-to-zero, or the first inexact op) takes the exact path.

// Exact-integer float emulation is dense with one-letter math names, compares
// host floats against known-exact boundaries, and mirrors pseudocode branches
// that happen to share bodies.
#![allow(
    clippy::float_cmp,
    clippy::match_same_arms,
    clippy::many_single_char_names
)]

/// `FPSR` cumulative exception bits.
pub(crate) const IOC: u32 = 1 << 0;
pub(crate) const DZC: u32 = 1 << 1;
pub(crate) const OFC: u32 = 1 << 2;
pub(crate) const UFC: u32 = 1 << 3;
pub(crate) const IXC: u32 = 1 << 4;
pub(crate) const IDC: u32 = 1 << 7;
/// `FPSR.QC`: cumulative saturation (set by saturating integer SIMD ops).
pub(crate) const QC: u32 = 1 << 27;

/// `FPCR` control bits.
const FPCR_AHP: u32 = 1 << 26;
const FPCR_DN: u32 = 1 << 25;
const FPCR_FZ: u32 = 1 << 24;
const FPCR_FZ16: u32 = 1 << 19;
/// `FPCR` bits that disqualify the host fast path: `FZ` and `RMode`.
const FPCR_SLOW: u32 = FPCR_FZ | (3 << 22);

/// An IEEE binary interchange format: `e` exponent bits, `f` fraction bits.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Fmt {
    pub e: u32,
    pub f: u32,
}

pub(crate) const H: Fmt = Fmt { e: 5, f: 10 };
pub(crate) const S: Fmt = Fmt { e: 8, f: 23 };
pub(crate) const D: Fmt = Fmt { e: 11, f: 52 };

impl Fmt {
    /// Total width in bits (16/32/64).
    pub(crate) const fn n(self) -> u32 {
        1 + self.e + self.f
    }
    const fn bias(self) -> i32 {
        (1 << (self.e - 1)) - 1
    }
    /// Unbiased exponent of the smallest normal number.
    const fn min_exp(self) -> i32 {
        1 - self.bias()
    }
    const fn max_biased(self) -> u64 {
        (1 << self.e) - 1
    }
    const fn frac_mask(self) -> u64 {
        (1 << self.f) - 1
    }
    const fn sign_bit(self) -> u64 {
        1 << (self.n() - 1)
    }
    /// All-ones mask of the format's width.
    pub(crate) const fn mask(self) -> u64 {
        if self.n() == 64 {
            u64::MAX
        } else {
            (1 << self.n()) - 1
        }
    }
    pub(crate) const fn default_nan(self) -> u64 {
        (self.max_biased() << self.f) | (1 << (self.f - 1))
    }
    pub(crate) const fn infinity(self, sign: bool) -> u64 {
        self.signed(sign, self.max_biased() << self.f)
    }
    pub(crate) const fn zero(self, sign: bool) -> u64 {
        self.signed(sign, 0)
    }
    const fn max_normal(self, sign: bool) -> u64 {
        self.signed(sign, ((self.max_biased() - 1) << self.f) | self.frac_mask())
    }
    const fn signed(self, sign: bool, mag: u64) -> u64 {
        if sign { mag | self.sign_bit() } else { mag }
    }
    /// The finite value `±(1 + frac/2^f) · 2^exp` (a normal number).
    const fn normal(self, sign: bool, exp: i32, frac: u64) -> u64 {
        self.signed(sign, (((exp + self.bias()) as u64) << self.f) | frac)
    }
    pub(crate) const fn is_nan(self, bits: u64) -> bool {
        (bits >> self.f) & self.max_biased() == self.max_biased() && bits & self.frac_mask() != 0
    }
}

/// An ARM rounding mode (`FPRounding`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Rounding {
    TieEven,
    PosInf,
    NegInf,
    Zero,
    TieAway,
    /// Von Neumann rounding (`FCVTXN`): truncate, then force the LSB to 1 if
    /// anything was discarded.
    Odd,
}

impl Rounding {
    /// Decode an `FPCR.RMode`-style 2-bit field.
    pub(crate) const fn from_rmode(rmode: u32) -> Self {
        match rmode & 3 {
            0 => Rounding::TieEven,
            1 => Rounding::PosInf,
            2 => Rounding::NegInf,
            _ => Rounding::Zero,
        }
    }
}

/// The floating-point environment an operation runs under: `FPCR` (read) and
/// `FPSR` (cumulative flags, OR-ed into).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Env {
    pub fpcr: u32,
    pub fpsr: u32,
}

impl Env {
    pub(crate) fn rounding(self) -> Rounding {
        Rounding::from_rmode(self.fpcr >> 22)
    }
    /// Whether arithmetic flushes subnormal inputs/outputs of `fmt` to zero.
    fn fz(self, fmt: Fmt) -> bool {
        if fmt == H {
            self.fpcr & FPCR_FZ16 != 0
        } else {
            self.fpcr & FPCR_FZ != 0
        }
    }
    /// `FZ` as the conversion instructions see it (`FPUnpackCV`/`FPRoundCV`
    /// ignore `FZ16`).
    fn fz_cv(self, fmt: Fmt) -> bool {
        fmt != H && self.fpcr & FPCR_FZ != 0
    }
    fn dn(self) -> bool {
        self.fpcr & FPCR_DN != 0
    }
    fn ahp(self) -> bool {
        self.fpcr & FPCR_AHP != 0
    }
    fn raise(&mut self, flags: u32) {
        self.fpsr |= flags;
    }
    /// Host arithmetic is bit-identical for a normal finite result when
    /// rounding to nearest without flush-to-zero, *and* `IXC` is already set
    /// (so the result's inexactness needn't be detected).
    #[inline]
    fn fast(self) -> bool {
        self.fpcr & FPCR_SLOW == 0 && self.fpsr & IXC != 0
    }
    /// Round-to-nearest without flush-to-zero: host results for *exact*
    /// operations (conversions that can't round, `FRINT*` of a normal) match.
    #[inline]
    fn plain(self) -> bool {
        self.fpcr & FPCR_SLOW == 0
    }
}

// ---- unpacking -------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Zero,
    Finite,
    Inf,
    QNaN,
    SNaN,
}

/// An unpacked operand: for `Finite`, the exact value `±mant · 2^exp`.
#[derive(Clone, Copy, Debug)]
struct Un {
    kind: Kind,
    sign: bool,
    mant: u64,
    exp: i32,
}

impl Un {
    fn is_nan(&self) -> bool {
        matches!(self.kind, Kind::QNaN | Kind::SNaN)
    }
}

/// `FPUnpackBase`. `fz` flushes a subnormal to (signed) zero, raising `IDC`
/// for single/double; `ahp` treats an all-ones half-precision exponent as an
/// ordinary number (`FPCR.AHP` alternative format).
fn unpack_with(bits: u64, fmt: Fmt, env: &mut Env, fz: bool, ahp: bool) -> Un {
    let sign = bits & fmt.sign_bit() != 0;
    let be = (bits >> fmt.f) & fmt.max_biased();
    let frac = bits & fmt.frac_mask();
    let (kind, mant, exp) = if be == 0 {
        if frac == 0 {
            (Kind::Zero, 0, 0)
        } else if fz {
            if fmt != H {
                env.raise(IDC);
            }
            (Kind::Zero, 0, 0)
        } else {
            (Kind::Finite, frac, fmt.min_exp() - fmt.f as i32)
        }
    } else if be == fmt.max_biased() && !(fmt == H && ahp) {
        if frac == 0 {
            (Kind::Inf, 0, 0)
        } else if frac >> (fmt.f - 1) & 1 == 1 {
            (Kind::QNaN, 0, 0)
        } else {
            (Kind::SNaN, 0, 0)
        }
    } else {
        (
            Kind::Finite,
            frac | (1 << fmt.f),
            be as i32 - fmt.bias() - fmt.f as i32,
        )
    };
    Un {
        kind,
        sign,
        mant,
        exp,
    }
}

/// `FPUnpack`: the arithmetic view (flush per `FZ`/`FZ16`, `AHP` ignored).
fn unpack(bits: u64, fmt: Fmt, env: &mut Env) -> Un {
    let fz = env.fz(fmt);
    unpack_with(bits, fmt, env, fz, false)
}

/// `FPUnpackCV`: the conversion view (`FZ16` ignored, `AHP` honoured).
fn unpack_cv(bits: u64, fmt: Fmt, env: &mut Env) -> Un {
    let fz = env.fz_cv(fmt);
    let ahp = env.ahp();
    unpack_with(bits, fmt, env, fz, ahp)
}

// ---- NaN handling ----------------------------------------------------------

/// `FPProcessNaN`: quiet a signaling NaN (raising `IOC`), or substitute the
/// default NaN under `FPCR.DN`.
fn process_nan(bits: u64, kind: Kind, fmt: Fmt, env: &mut Env) -> u64 {
    let mut r = bits & fmt.mask();
    if kind == Kind::SNaN {
        r |= 1 << (fmt.f - 1);
        env.raise(IOC);
    }
    if env.dn() { fmt.default_nan() } else { r }
}

/// `FPProcessNaNs`: signaling NaNs take priority over quiet ones, then
/// operand order.
fn process_nans2(a: u64, ua: &Un, b: u64, ub: &Un, fmt: Fmt, env: &mut Env) -> Option<u64> {
    if ua.kind == Kind::SNaN {
        Some(process_nan(a, ua.kind, fmt, env))
    } else if ub.kind == Kind::SNaN {
        Some(process_nan(b, ub.kind, fmt, env))
    } else if ua.kind == Kind::QNaN {
        Some(process_nan(a, ua.kind, fmt, env))
    } else if ub.kind == Kind::QNaN {
        Some(process_nan(b, ub.kind, fmt, env))
    } else {
        None
    }
}

/// `FPProcessNaNs3`.
#[allow(clippy::too_many_arguments)]
fn process_nans3(
    a: u64,
    ua: &Un,
    b: u64,
    ub: &Un,
    c: u64,
    uc: &Un,
    fmt: Fmt,
    env: &mut Env,
) -> Option<u64> {
    for (bits, u) in [(a, ua), (b, ub), (c, uc)] {
        if u.kind == Kind::SNaN {
            return Some(process_nan(bits, u.kind, fmt, env));
        }
    }
    for (bits, u) in [(a, ua), (b, ub), (c, uc)] {
        if u.kind == Kind::QNaN {
            return Some(process_nan(bits, u.kind, fmt, env));
        }
    }
    None
}

// ---- rounding --------------------------------------------------------------

/// How a truncation's discarded part compares with half an ULP.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Rest {
    Exact,
    Below,
    Half,
    Above,
}

/// Shift `sig` right by `shift` bits (`sticky`: nonzero bits already lost
/// below `sig`'s LSB), returning the kept integer and how the rest compares
/// with one half.
fn shift_rest(sig: u128, shift: i32, sticky: bool) -> (u128, Rest) {
    if shift <= 0 {
        debug_assert!(!sticky, "rounding lost precision");
        return (sig << (-shift) as u32, Rest::Exact);
    }
    if shift > 128 {
        return (
            0,
            if sig != 0 || sticky {
                Rest::Below
            } else {
                Rest::Exact
            },
        );
    }
    let (kept, rem, half) = if shift == 128 {
        (0, sig, 1u128 << 127)
    } else {
        let s = shift as u32;
        (sig >> s, sig & ((1u128 << s) - 1), 1u128 << (s - 1))
    };
    let rest = if rem == 0 && !sticky {
        Rest::Exact
    } else if rem > half || (rem == half && sticky) {
        Rest::Above
    } else if rem == half {
        Rest::Half
    } else {
        Rest::Below
    };
    (kept, rest)
}

/// Whether a magnitude truncated to `kept` (with discarded part `rest`) rounds
/// up under `mode`, for a value of sign `sign`.
fn rounds_up(mode: Rounding, sign: bool, kept_odd: bool, rest: Rest) -> bool {
    match mode {
        Rounding::TieEven => rest == Rest::Above || (rest == Rest::Half && kept_odd),
        Rounding::TieAway => matches!(rest, Rest::Above | Rest::Half),
        Rounding::PosInf => rest != Rest::Exact && !sign,
        Rounding::NegInf => rest != Rest::Exact && sign,
        Rounding::Zero | Rounding::Odd => false,
    }
}

/// `FPRoundBase`: round the exact value `±(sig + sticky·ε) · 2^exp` (`sig`
/// nonzero) into `fmt`. `fz` flushes a result that is tiny before rounding to
/// zero (raising `UFC` only); `ahp` selects the alternative half-precision
/// overflow behaviour (saturate, raise `IOC`).
#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
fn round_full(
    sign: bool,
    sig: u128,
    exp: i32,
    sticky: bool,
    fmt: Fmt,
    mode: Rounding,
    fz: bool,
    ahp: bool,
    env: &mut Env,
) -> u64 {
    debug_assert!(sig != 0);
    let f = fmt.f as i32;
    let e = exp + 127 - sig.leading_zeros() as i32; // value in [2^e, 2^(e+1))
    let min_exp = fmt.min_exp();
    if fz && e < min_exp {
        env.raise(UFC);
        return fmt.zero(sign);
    }
    let (mut biased, lsb_exp) = if e < min_exp {
        (0u64, min_exp - f)
    } else {
        ((e - min_exp + 1) as u64, e - f)
    };
    let (kept, rest) = shift_rest(sig, lsb_exp - exp, sticky);
    let mut m = kept as u64;
    let inexact = rest != Rest::Exact;
    if biased == 0 && inexact {
        env.raise(UFC);
    }
    if rounds_up(mode, sign, m & 1 == 1, rest) {
        m += 1;
        if m == 1 << f {
            biased = 1; // a subnormal rounded up to the smallest normal
        }
        if m == 1 << (f + 1) {
            biased += 1;
            m >>= 1;
        }
    }
    if mode == Rounding::Odd && inexact {
        m |= 1;
    }
    if ahp {
        if biased > fmt.max_biased() {
            env.raise(IOC);
            return fmt.signed(sign, fmt.mask() >> 1);
        }
    } else if biased >= fmt.max_biased() {
        let to_inf = match mode {
            Rounding::TieEven | Rounding::TieAway => true,
            Rounding::PosInf => !sign,
            Rounding::NegInf => sign,
            Rounding::Zero | Rounding::Odd => false,
        };
        env.raise(OFC | IXC);
        return if to_inf {
            fmt.infinity(sign)
        } else {
            fmt.max_normal(sign)
        };
    }
    if inexact {
        env.raise(IXC);
    }
    fmt.signed(sign, (biased << f) | (m & fmt.frac_mask()))
}

/// [`round_full`] for an arithmetic result under the environment's rounding
/// mode and flush-to-zero setting.
fn round(sign: bool, sig: u128, exp: i32, sticky: bool, fmt: Fmt, env: &mut Env) -> u64 {
    let mode = env.rounding();
    let fz = env.fz(fmt);
    round_full(sign, sig, exp, sticky, fmt, mode, fz, false, env)
}

/// The signed zero an exactly-cancelling sum produces: `-0` only when
/// rounding toward minus infinity.
fn cancel_zero(fmt: Fmt, env: Env) -> u64 {
    fmt.zero(env.rounding() == Rounding::NegInf)
}

// ---- exact arithmetic kernels ------------------------------------------------

/// Exact `±ma·2^ea ± mb·2^eb` (each mantissa nonzero and at most ~110 bits
/// wide). Returns `None` for an exact zero, else `(sign, sig, exp, sticky)`.
/// Both operands are normalized to bit 125; an operand shifted right past
/// those trailing zeros (exponents ≥ 17 apart) collapses into a sticky bit,
/// which can then cancel at most one leading bit — far above the rounding
/// point.
fn add_exact(
    sa: bool,
    ma: u128,
    ea: i32,
    sb: bool,
    mb: u128,
    eb: i32,
) -> Option<(bool, u128, i32, bool)> {
    let norm = |m: u128, e: i32| {
        let sh = m.leading_zeros() as i32 - 2;
        (m << sh as u32, e - sh)
    };
    let (mut ma, mut ea, mut sa) = {
        let (m, e) = norm(ma, ea);
        (m, e, sa)
    };
    let (mut mb, mut eb, mut sb) = {
        let (m, e) = norm(mb, eb);
        (m, e, sb)
    };
    if (eb, mb) > (ea, ma) {
        std::mem::swap(&mut ma, &mut mb);
        std::mem::swap(&mut ea, &mut eb);
        std::mem::swap(&mut sa, &mut sb);
    }
    let d = ea - eb;
    let (mb, sticky) = if d == 0 {
        (mb, false)
    } else if d >= 127 {
        (0, true)
    } else {
        (mb >> d as u32, mb & ((1u128 << d as u32) - 1) != 0)
    };
    if sa == sb {
        Some((sa, ma + mb, ea, sticky))
    } else {
        let mut r = ma - mb;
        if sticky {
            r -= 1;
        }
        if r == 0 && !sticky {
            None
        } else {
            Some((sa, r, ea, sticky))
        }
    }
}

/// Integer square root of `n` and whether it was inexact.
fn isqrt(n: u128) -> (u128, bool) {
    if n == 0 {
        return (0, false);
    }
    // Digit-by-digit (binary restoring) square root.
    let mut x = n;
    let mut r: u128 = 0;
    let mut bit: u128 = 1 << (n.ilog2() & !1);
    while bit != 0 {
        if x >= r + bit {
            x -= r + bit;
            r = (r >> 1) + bit;
        } else {
            r >>= 1;
        }
        bit >>= 2;
    }
    (r, x != 0)
}

// ---- host fast paths ---------------------------------------------------------

/// Whether a host result can be returned as-is: finite and strictly above the
/// smallest normal magnitude (so neither overflow nor tininess-before-rounding
/// can have occurred).
#[inline]
fn host_ok32(r: f32) -> bool {
    r.is_finite() && r.abs() > f32::MIN_POSITIVE
}
#[inline]
fn host_ok64(r: f64) -> bool {
    r.is_finite() && r.abs() > f64::MIN_POSITIVE
}

macro_rules! fast2 {
    ($fmt:expr, $env:expr, $a:expr, $b:expr, $op:expr, $allow_zero:expr) => {
        if $env.fast() {
            if $fmt == S {
                let r: f32 = ($op)(f32::from_bits($a as u32), f32::from_bits($b as u32));
                if host_ok32(r) || ($allow_zero && r == 0.0) {
                    return u64::from(r.to_bits());
                }
            } else if $fmt == D {
                let r: f64 = ($op)(f64::from_bits($a), f64::from_bits($b));
                if host_ok64(r) || ($allow_zero && r == 0.0) {
                    return r.to_bits();
                }
            }
        }
    };
}

// ---- arithmetic ---------------------------------------------------------------

/// `FPAdd` (`neg_b == false`) / `FPSub` (`neg_b == true`).
pub(crate) fn add(a: u64, b: u64, neg_b: bool, fmt: Fmt, env: &mut Env) -> u64 {
    if neg_b {
        fast2!(fmt, env, a, b, |x, y| x - y, true);
    } else {
        fast2!(fmt, env, a, b, |x, y| x + y, true);
    }
    let ua = unpack(a, fmt, env);
    let ub = unpack(b, fmt, env);
    if let Some(r) = process_nans2(a, &ua, b, &ub, fmt, env) {
        return r;
    }
    let sb = ub.sign ^ neg_b;
    match (ua.kind, ub.kind) {
        (Kind::Inf, Kind::Inf) if ua.sign != sb => {
            env.raise(IOC);
            fmt.default_nan()
        }
        (Kind::Inf, _) => fmt.infinity(ua.sign),
        (_, Kind::Inf) => fmt.infinity(sb),
        (Kind::Zero, Kind::Zero) if ua.sign == sb => fmt.zero(ua.sign),
        (Kind::Zero, Kind::Zero) => cancel_zero(fmt, *env),
        (Kind::Zero, _) => round(sb, u128::from(ub.mant), ub.exp, false, fmt, env),
        (_, Kind::Zero) => round(ua.sign, u128::from(ua.mant), ua.exp, false, fmt, env),
        _ => match add_exact(
            ua.sign,
            u128::from(ua.mant),
            ua.exp,
            sb,
            u128::from(ub.mant),
            ub.exp,
        ) {
            None => cancel_zero(fmt, *env),
            Some((s, m, e, st)) => round(s, m, e, st, fmt, env),
        },
    }
}

/// `FPMul` (`mulx == false`) / `FPMulX` (`∞ × 0 = ±2` instead of NaN).
pub(crate) fn mul(a: u64, b: u64, mulx: bool, fmt: Fmt, env: &mut Env) -> u64 {
    fast2!(fmt, env, a, b, |x, y| x * y, false);
    let ua = unpack(a, fmt, env);
    let ub = unpack(b, fmt, env);
    if let Some(r) = process_nans2(a, &ua, b, &ub, fmt, env) {
        return r;
    }
    let sign = ua.sign ^ ub.sign;
    match (ua.kind, ub.kind) {
        (Kind::Inf, Kind::Zero) | (Kind::Zero, Kind::Inf) => {
            if mulx {
                fmt.normal(sign, 1, 0)
            } else {
                env.raise(IOC);
                fmt.default_nan()
            }
        }
        (Kind::Inf, _) | (_, Kind::Inf) => fmt.infinity(sign),
        (Kind::Zero, _) | (_, Kind::Zero) => fmt.zero(sign),
        _ => round(
            sign,
            u128::from(ua.mant) * u128::from(ub.mant),
            ua.exp + ub.exp,
            false,
            fmt,
            env,
        ),
    }
}

/// `FPDiv`.
pub(crate) fn div(a: u64, b: u64, fmt: Fmt, env: &mut Env) -> u64 {
    fast2!(fmt, env, a, b, |x, y| x / y, false);
    let ua = unpack(a, fmt, env);
    let ub = unpack(b, fmt, env);
    if let Some(r) = process_nans2(a, &ua, b, &ub, fmt, env) {
        return r;
    }
    let sign = ua.sign ^ ub.sign;
    match (ua.kind, ub.kind) {
        (Kind::Inf, Kind::Inf) | (Kind::Zero, Kind::Zero) => {
            env.raise(IOC);
            fmt.default_nan()
        }
        (Kind::Inf, _) => fmt.infinity(sign),
        (_, Kind::Zero) => {
            env.raise(DZC);
            fmt.infinity(sign)
        }
        (Kind::Zero, _) | (_, Kind::Inf) => fmt.zero(sign),
        _ => {
            let n = u128::from(ua.mant);
            let sh = n.leading_zeros() - 1; // numerator MSB at bit 126
            let n = n << sh;
            let d = u128::from(ub.mant);
            let (q, r) = (n / d, n % d);
            round(sign, q, ua.exp - sh as i32 - ub.exp, r != 0, fmt, env)
        }
    }
}

/// `FPSqrt`.
pub(crate) fn sqrt(a: u64, fmt: Fmt, env: &mut Env) -> u64 {
    if env.fast() {
        if fmt == S {
            let x = f32::from_bits(a as u32);
            let r = x.sqrt();
            if host_ok32(r) {
                return u64::from(r.to_bits());
            }
        } else if fmt == D {
            let r = f64::from_bits(a).sqrt();
            if host_ok64(r) {
                return r.to_bits();
            }
        }
    }
    let ua = unpack(a, fmt, env);
    match ua.kind {
        Kind::QNaN | Kind::SNaN => process_nan(a, ua.kind, fmt, env),
        Kind::Zero => fmt.zero(ua.sign),
        Kind::Inf if !ua.sign => fmt.infinity(false),
        _ if ua.sign => {
            env.raise(IOC);
            fmt.default_nan()
        }
        _ => {
            let m = u128::from(ua.mant);
            let mut sh = m.leading_zeros() as i32 - 3; // MSB at bit 124
            if (ua.exp - sh) & 1 != 0 {
                sh += 1;
            }
            let (r, inexact) = isqrt(m << sh as u32);
            round(false, r, (ua.exp - sh) / 2, inexact, fmt, env)
        }
    }
}

/// `FPMulAdd`: `addend + op1·op2` with a single rounding. (`FMSUB`/`FNMADD`/
/// `FNMSUB` and the vector `FMLS` negate operands *before* calling this, as
/// the pseudocode does — so a propagated NaN carries the flipped sign.)
pub(crate) fn mul_add(addend: u64, op1: u64, op2: u64, fmt: Fmt, env: &mut Env) -> u64 {
    if env.fast() {
        if fmt == S {
            let r = f32::from_bits(op1 as u32)
                .mul_add(f32::from_bits(op2 as u32), f32::from_bits(addend as u32));
            if host_ok32(r) {
                return u64::from(r.to_bits());
            }
        } else if fmt == D {
            let r = f64::from_bits(op1).mul_add(f64::from_bits(op2), f64::from_bits(addend));
            if host_ok64(r) {
                return r.to_bits();
            }
        }
    }
    let ua = unpack(addend, fmt, env);
    let u1 = unpack(op1, fmt, env);
    let u2 = unpack(op2, fmt, env);
    let inf_zero = (u1.kind == Kind::Inf && u2.kind == Kind::Zero)
        || (u1.kind == Kind::Zero && u2.kind == Kind::Inf);
    let nan = process_nans3(addend, &ua, op1, &u1, op2, &u2, fmt, env);
    if ua.kind == Kind::QNaN && inf_zero {
        env.raise(IOC);
        return fmt.default_nan();
    }
    if let Some(r) = nan {
        return r;
    }
    let sp = u1.sign ^ u2.sign;
    let inf_p = u1.kind == Kind::Inf || u2.kind == Kind::Inf;
    let zero_p = u1.kind == Kind::Zero || u2.kind == Kind::Zero;
    let inf_a = ua.kind == Kind::Inf;
    if inf_zero || (inf_a && inf_p && ua.sign != sp) {
        env.raise(IOC);
        return fmt.default_nan();
    }
    if (inf_a && !ua.sign) || (inf_p && !sp) {
        return fmt.infinity(false);
    }
    if (inf_a && ua.sign) || (inf_p && sp) {
        return fmt.infinity(true);
    }
    let zero_a = ua.kind == Kind::Zero;
    if zero_a && zero_p {
        return if ua.sign == sp {
            fmt.zero(ua.sign)
        } else {
            cancel_zero(fmt, *env)
        };
    }
    if zero_p {
        return round(ua.sign, u128::from(ua.mant), ua.exp, false, fmt, env);
    }
    let pm = u128::from(u1.mant) * u128::from(u2.mant);
    let pe = u1.exp + u2.exp;
    if zero_a {
        return round(sp, pm, pe, false, fmt, env);
    }
    match add_exact(ua.sign, u128::from(ua.mant), ua.exp, sp, pm, pe) {
        None => cancel_zero(fmt, *env),
        Some((s, m, e, st)) => round(s, m, e, st, fmt, env),
    }
}

/// `FPRecipStepFused` (`FRECPS`: `2 − a·b`) and `FPRSqrtStepFused`
/// (`FRSQRTS`: `(3 − a·b) / 2`), each with one rounding.
pub(crate) fn step_fused(a: u64, b: u64, rsqrt: bool, fmt: Fmt, env: &mut Env) -> u64 {
    let a = a ^ fmt.sign_bit(); // FPNeg(op1), before NaN processing
    let ua = unpack(a, fmt, env);
    let ub = unpack(b, fmt, env);
    if let Some(r) = process_nans2(a, &ua, b, &ub, fmt, env) {
        return r;
    }
    let sign = ua.sign ^ ub.sign;
    match (ua.kind, ub.kind) {
        (Kind::Inf, Kind::Zero) | (Kind::Zero, Kind::Inf) => {
            if rsqrt {
                fmt.normal(false, 0, 1 << (fmt.f - 1)) // 1.5
            } else {
                fmt.normal(false, 1, 0) // 2.0
            }
        }
        (Kind::Inf, _) | (_, Kind::Inf) => fmt.infinity(sign),
        (Kind::Zero, _) | (_, Kind::Zero) => {
            if rsqrt {
                fmt.normal(false, 0, 1 << (fmt.f - 1))
            } else {
                fmt.normal(false, 1, 0)
            }
        }
        _ => {
            // 3 (or 2) + a·b, then halve for the rsqrt step (exact scaling).
            let pm = u128::from(ua.mant) * u128::from(ub.mant);
            let pe = ua.exp + ub.exp;
            let (k, ke) = if rsqrt { (3u128, -1) } else { (1u128, 1) };
            let pe = if rsqrt { pe - 1 } else { pe };
            match add_exact(false, k, ke, sign, pm, pe) {
                None => cancel_zero(fmt, *env),
                Some((s, m, e, st)) => round(s, m, e, st, fmt, env),
            }
        }
    }
}

/// `FPMax`/`FPMin` (`max` selects), and with `num` the IEEE 754-2008
/// `maxNum`/`minNum` variants (`FPMaxNum`/`FPMinNum`): a single quiet NaN
/// operand loses to a number.
pub(crate) fn max_min(a: u64, b: u64, max: bool, num: bool, fmt: Fmt, env: &mut Env) -> u64 {
    let (mut a, mut b) = (a, b);
    if num {
        let qa = is_qnan(a, fmt);
        let qb = is_qnan(b, fmt);
        if qa && !qb {
            a = fmt.infinity(max);
        } else if !qa && qb {
            b = fmt.infinity(max);
        }
    }
    let ua = unpack(a, fmt, env);
    let ub = unpack(b, fmt, env);
    if let Some(r) = process_nans2(a, &ua, b, &ub, fmt, env) {
        return r;
    }
    let ord = cmp_values(&ua, &ub);
    let pick_a = if max {
        ord == std::cmp::Ordering::Greater
    } else {
        ord == std::cmp::Ordering::Less
    };
    let u = if pick_a { ua } else { ub };
    match u.kind {
        Kind::Inf => fmt.infinity(u.sign),
        Kind::Zero => fmt.zero(if max {
            ua.sign && ub.sign
        } else {
            ua.sign || ub.sign
        }),
        _ => round(u.sign, u128::from(u.mant), u.exp, false, fmt, env),
    }
}

fn is_qnan(bits: u64, fmt: Fmt) -> bool {
    fmt.is_nan(bits) && bits >> (fmt.f - 1) & 1 == 1
}

/// Order two non-NaN unpacked values numerically.
fn cmp_values(a: &Un, b: &Un) -> std::cmp::Ordering {
    // Map to (class rank, magnitude) per sign.
    let key = |u: &Un| -> (i32, i32, u64) {
        match u.kind {
            Kind::Zero => (0, 0, 0),
            Kind::Inf => (if u.sign { -2 } else { 2 }, 0, 0),
            _ => {
                // Normalize magnitude for comparison: exponent of MSB, then mant
                // aligned to bit 63.
                let lz = u.mant.leading_zeros() as i32;
                let e = u.exp + 63 - lz;
                let m = u.mant << lz;
                (if u.sign { -1 } else { 1 }, e, m)
            }
        }
    };
    let (ka, kb) = (key(a), key(b));
    if ka.0 != kb.0 {
        return ka.0.cmp(&kb.0);
    }
    let ord = (ka.1, ka.2).cmp(&(kb.1, kb.2));
    if ka.0 < 0 { ord.reverse() } else { ord }
}

/// `FPCompare`: the NZCV result of `FCMP`/`FCMPE`/`FCCMP` (`signal`: the `E`
/// forms raise `IOC` on quiet NaNs too).
pub(crate) fn compare(a: u64, b: u64, signal: bool, fmt: Fmt, env: &mut Env) -> u32 {
    let ua = unpack(a, fmt, env);
    let ub = unpack(b, fmt, env);
    if ua.is_nan() || ub.is_nan() {
        if ua.kind == Kind::SNaN || ub.kind == Kind::SNaN || signal {
            env.raise(IOC);
        }
        return 0b0011;
    }
    match cmp_values(&ua, &ub) {
        std::cmp::Ordering::Equal => 0b0110,
        std::cmp::Ordering::Less => 0b1000,
        std::cmp::Ordering::Greater => 0b0010,
    }
}

/// The vector compares: `FPCompareEQ` (quiet) and `FPCompareGE`/`GT`
/// (signaling on any NaN).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Cmp {
    Eq,
    Ge,
    Gt,
}

pub(crate) fn compare_op(a: u64, b: u64, op: Cmp, fmt: Fmt, env: &mut Env) -> bool {
    let ua = unpack(a, fmt, env);
    let ub = unpack(b, fmt, env);
    if ua.is_nan() || ub.is_nan() {
        if op != Cmp::Eq || ua.kind == Kind::SNaN || ub.kind == Kind::SNaN {
            env.raise(IOC);
        }
        return false;
    }
    let ord = cmp_values(&ua, &ub);
    match op {
        Cmp::Eq => ord == std::cmp::Ordering::Equal,
        Cmp::Ge => ord != std::cmp::Ordering::Less,
        Cmp::Gt => ord == std::cmp::Ordering::Greater,
    }
}

// ---- rounding to integral, conversions ---------------------------------------

/// `FPRoundInt` (`FRINT*`): round to an integral value in the same format.
/// `exact` (`FRINTX`) raises `IXC` when the value changed.
pub(crate) fn round_int(a: u64, mode: Rounding, exact: bool, fmt: Fmt, env: &mut Env) -> u64 {
    let ua = unpack(a, fmt, env);
    match ua.kind {
        Kind::QNaN | Kind::SNaN => process_nan(a, ua.kind, fmt, env),
        Kind::Inf => fmt.infinity(ua.sign),
        Kind::Zero => fmt.zero(ua.sign),
        Kind::Finite => {
            if ua.exp >= 0 {
                return a & fmt.mask(); // already integral
            }
            let (kept, rest) = shift_rest(u128::from(ua.mant), -ua.exp, false);
            let mut m = kept;
            if rounds_up(mode, ua.sign, m & 1 == 1, rest) {
                m += 1;
            }
            if rest != Rest::Exact && exact {
                env.raise(IXC);
            }
            if m == 0 {
                fmt.zero(ua.sign)
            } else {
                let mut scratch = *env;
                round_full(
                    ua.sign,
                    m,
                    0,
                    false,
                    fmt,
                    Rounding::Zero,
                    false,
                    false,
                    &mut scratch,
                )
            }
        }
    }
}

/// `FPToFixed`: convert to a `bits`-wide (signed or unsigned) integer scaled
/// by `2^fbits`, saturating (with `IOC`) on overflow or NaN.
pub(crate) fn to_fixed(
    a: u64,
    fmt: Fmt,
    fbits: u32,
    unsigned: bool,
    bits: u32,
    mode: Rounding,
    env: &mut Env,
) -> u64 {
    let out_mask = if bits == 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    };
    let max_mag: u128 = if unsigned {
        u128::from(out_mask)
    } else {
        1u128 << (bits - 1)
    }; // positive limit is max_mag - 1 for signed
    let saturate = |sign: bool| -> u64 {
        if unsigned {
            if sign { 0 } else { out_mask }
        } else if sign {
            (1u64 << (bits - 1)) & out_mask
        } else {
            out_mask >> 1
        }
    };
    // Fast path: in-range, exactly integral or IXC already set.
    if env.plain() && fbits == 0 && mode == Rounding::Zero && fmt == D && bits >= 32 {
        let x = f64::from_bits(a);
        let lim = if bits == 64 { 9.2e18 } else { 2.1e9 };
        if x.is_finite() && x.abs() < lim && (x.abs() >= f64::MIN_POSITIVE || x == 0.0) {
            let t = x.trunc();
            if t == x || env.fpsr & IXC != 0 {
                if t != x {
                    env.raise(IXC);
                }
                if !unsigned {
                    return (t as i64 as u64) & out_mask;
                } else if t >= 0.0 {
                    return (t as u64) & out_mask;
                }
            }
        }
    }
    let ua = unpack(a, fmt, env);
    match ua.kind {
        Kind::QNaN | Kind::SNaN => {
            env.raise(IOC);
            0
        }
        Kind::Inf => {
            env.raise(IOC);
            saturate(ua.sign)
        }
        Kind::Zero => 0,
        Kind::Finite => {
            let e = ua.exp + fbits as i32;
            let m = u128::from(ua.mant);
            let (mag, rest) = if e >= 0 {
                if 128 - m.leading_zeros() as i32 + e > 72 {
                    env.raise(IOC);
                    return saturate(ua.sign);
                }
                (m << e as u32, Rest::Exact)
            } else {
                shift_rest(m, -e, false)
            };
            let mut mag = mag;
            if rounds_up(mode, ua.sign, mag & 1 == 1, rest) {
                mag += 1;
            }
            let overflow = if unsigned {
                (ua.sign && mag != 0) || mag > max_mag
            } else if ua.sign {
                mag > max_mag
            } else {
                mag >= max_mag
            };
            if overflow {
                env.raise(IOC);
                return saturate(ua.sign);
            }
            if rest != Rest::Exact {
                env.raise(IXC);
            }
            let v = mag as u64;
            (if ua.sign { v.wrapping_neg() } else { v }) & out_mask
        }
    }
}

/// `FixedToFP`: convert a `bits`-wide signed/unsigned integer scaled by
/// `2^-fbits` to `fmt`.
#[allow(clippy::cast_precision_loss)] // the host fast path is exact by construction
pub(crate) fn from_fixed(
    v: u64,
    bits: u32,
    signed: bool,
    fbits: u32,
    fmt: Fmt,
    mode: Rounding,
    env: &mut Env,
) -> u64 {
    let v = if bits == 64 {
        v
    } else {
        v & ((1u64 << bits) - 1)
    };
    let (sign, mag) = if signed && (v >> (bits - 1)) & 1 == 1 {
        let ext = if bits == 64 {
            v
        } else {
            v | !((1u64 << bits) - 1)
        };
        (true, ext.wrapping_neg())
    } else {
        (false, v)
    };
    if mag == 0 {
        return fmt.zero(false);
    }
    if fbits == 0 && mode == env.rounding() && env.plain() {
        // Exactly representable integers convert identically on the host.
        let lim = 1u64 << (fmt.f + 1);
        if mag <= lim {
            if fmt == D {
                let x = mag as f64;
                return (if sign { -x } else { x }).to_bits();
            } else if fmt == S {
                let x = mag as f32;
                return u64::from((if sign { -x } else { x }).to_bits());
            }
        }
    }
    let fz = env.fz(fmt);
    round_full(
        sign,
        u128::from(mag),
        -(fbits as i32),
        false,
        fmt,
        mode,
        fz,
        false,
        env,
    )
}

/// `FPConvert`: change precision (`FCVT`, `FCVTN`/`FCVTL`, `FCVTXN` with
/// [`Rounding::Odd`]).
pub(crate) fn convert(a: u64, from: Fmt, to: Fmt, mode: Rounding, env: &mut Env) -> u64 {
    if from == S && to == D && env.plain() {
        let x = f32::from_bits(a as u32);
        if x.is_normal() || x == 0.0 || x.is_infinite() {
            return f64::from(x).to_bits();
        }
    }
    let u = unpack_cv(a, from, env);
    let alt_hp = to == H && env.ahp();
    match u.kind {
        Kind::QNaN | Kind::SNaN => {
            let r = if alt_hp {
                to.zero(u.sign)
            } else if env.dn() {
                to.default_nan()
            } else {
                // FPConvertNaN: keep the sign and the top payload bits.
                let payload = a & from.frac_mask() & !(1 << (from.f - 1));
                let p = if to.f >= from.f {
                    payload << (to.f - from.f)
                } else {
                    payload >> (from.f - to.f)
                };
                to.signed(u.sign, (to.max_biased() << to.f) | (1 << (to.f - 1)) | p)
            };
            if u.kind == Kind::SNaN || alt_hp {
                env.raise(IOC);
            }
            r
        }
        Kind::Inf => {
            if alt_hp {
                env.raise(IOC);
                to.signed(u.sign, to.mask() >> 1)
            } else {
                to.infinity(u.sign)
            }
        }
        Kind::Zero => to.zero(u.sign),
        Kind::Finite => {
            let fz = env.fz_cv(to);
            round_full(
                u.sign,
                u128::from(u.mant),
                u.exp,
                false,
                to,
                mode,
                fz,
                alt_hp,
                env,
            )
        }
    }
}

// ---- estimates -------------------------------------------------------------------

/// `RecipEstimate` (8-bit): `a` in 256..512 is a fixed-point value in [0.5, 1).
fn recip_estimate(a: u64) -> u64 {
    let a = a * 2 + 1;
    let b = (1u64 << 19) / a;
    b.div_ceil(2) // round to nearest
}

/// `RecipSqrtEstimate` (8-bit): `a` in 128..512 is a fixed-point value in
/// [0.25, 1).
fn rsqrt_estimate(a: u64) -> u64 {
    let a = if a < 256 {
        a * 2 + 1 // units of 1/512, rounded to nearest
    } else {
        (((a >> 1) << 1) + 1) * 2 // drop the bottom bit; units of 1/256
    };
    let mut b: u64 = 512;
    while a * (b + 1) * (b + 1) < (1 << 28) {
        b += 1;
    }
    b.div_ceil(2) // round to nearest
}

/// `FPRecipEstimate` (`FRECPE`).
pub(crate) fn recip_est(a: u64, fmt: Fmt, env: &mut Env) -> u64 {
    let u = unpack(a, fmt, env);
    let f = fmt.f;
    match u.kind {
        Kind::QNaN | Kind::SNaN => return process_nan(a, u.kind, fmt, env),
        Kind::Inf => return fmt.zero(u.sign),
        Kind::Zero => {
            env.raise(DZC);
            return fmt.infinity(u.sign);
        }
        Kind::Finite => {}
    }
    let msb_exp = u.exp + 63 - u.mant.leading_zeros() as i32; // |value| in [2^msb_exp, ..)
    if msb_exp < -(fmt.bias() + 1) {
        let to_inf = match env.rounding() {
            Rounding::TieEven | Rounding::TieAway => true,
            Rounding::PosInf => !u.sign,
            Rounding::NegInf => u.sign,
            _ => false,
        };
        env.raise(OFC | IXC);
        return if to_inf {
            fmt.infinity(u.sign)
        } else {
            fmt.max_normal(u.sign)
        };
    }
    if env.fz(fmt) && msb_exp >= fmt.bias() - 1 {
        env.raise(UFC);
        return fmt.zero(u.sign);
    }
    let mask52 = (1u64 << 52) - 1;
    let mut fraction = (a & fmt.frac_mask()) << (52 - f);
    let mut exp = ((a >> f) & fmt.max_biased()) as i64;
    if exp == 0 {
        if fraction >> 51 & 1 == 0 {
            exp = -1;
            fraction = (fraction << 2) & mask52;
        } else {
            fraction = (fraction << 1) & mask52;
        }
    }
    let scaled = 0x100 | (fraction >> 44);
    let mut result_exp = 2 * i64::from(fmt.bias()) - 1 - exp;
    let estimate = recip_estimate(scaled);
    let mut fraction = (estimate & 0xff) << 44;
    if result_exp == 0 {
        fraction = (1 << 51) | (fraction >> 1);
    } else if result_exp == -1 {
        fraction = (1 << 50) | (fraction >> 2);
        result_exp = 0;
    }
    fmt.signed(u.sign, ((result_exp as u64) << f) | (fraction >> (52 - f)))
}

/// `FPRSqrtEstimate` (`FRSQRTE`).
pub(crate) fn rsqrt_est(a: u64, fmt: Fmt, env: &mut Env) -> u64 {
    let u = unpack(a, fmt, env);
    let f = fmt.f;
    match u.kind {
        Kind::QNaN | Kind::SNaN => return process_nan(a, u.kind, fmt, env),
        Kind::Zero => {
            env.raise(DZC);
            return fmt.infinity(u.sign);
        }
        _ if u.sign => {
            env.raise(IOC);
            return fmt.default_nan();
        }
        Kind::Inf => return fmt.zero(false),
        Kind::Finite => {}
    }
    let mask52 = (1u64 << 52) - 1;
    let mut fraction = (a & fmt.frac_mask()) << (52 - f);
    let mut exp = ((a >> f) & fmt.max_biased()) as i64;
    if exp == 0 {
        while fraction >> 51 & 1 == 0 {
            fraction = (fraction << 1) & mask52;
            exp -= 1;
        }
        fraction = (fraction << 1) & mask52;
    }
    let scaled = if exp & 1 == 0 {
        0x100 | (fraction >> 44)
    } else {
        0x80 | (fraction >> 45)
    };
    let result_exp = (3 * i64::from(fmt.bias()) - 1 - exp).div_euclid(2);
    let estimate = rsqrt_estimate(scaled);
    ((result_exp as u64) << f) | ((estimate & 0xff) << (f - 8))
}

/// `FPRecpX` (`FRECPX`): reciprocal exponent.
pub(crate) fn recpx(a: u64, fmt: Fmt, env: &mut Env) -> u64 {
    let u = unpack(a, fmt, env);
    if u.is_nan() {
        return process_nan(a, u.kind, fmt, env);
    }
    let exp = (a >> fmt.f) & fmt.max_biased();
    let e = if exp == 0 {
        fmt.max_biased() - 1
    } else {
        !exp & fmt.max_biased()
    };
    fmt.signed(u.sign, e << fmt.f)
}

/// `UnsignedRecipEstimate` (`URECPE`).
pub(crate) fn urecpe(a: u32) -> u32 {
    if a >> 31 == 0 {
        u32::MAX
    } else {
        (recip_estimate(u64::from(a >> 23)) as u32 & 0x1ff) << 23
    }
}

/// `UnsignedRSqrtEstimate` (`URSQRTE`).
pub(crate) fn ursqrte(a: u32) -> u32 {
    if a >> 30 == 0 {
        u32::MAX
    } else {
        (rsqrt_estimate(u64::from(a >> 23)) as u32 & 0x1ff) << 23
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env() -> Env {
        Env { fpcr: 0, fpsr: 0 }
    }

    #[test]
    fn add_matches_host_and_flags_inexact() {
        let mut e = env();
        let r = add(1.0f64.to_bits(), 1e-30f64.to_bits(), false, D, &mut e);
        assert_eq!(f64::from_bits(r), 1.0);
        assert_eq!(e.fpsr, IXC);
        let mut e = env();
        let r = add(0.1f64.to_bits(), 0.2f64.to_bits(), false, D, &mut e);
        assert_eq!(f64::from_bits(r), 0.1 + 0.2);
    }

    #[test]
    fn directed_rounding_differs_from_nearest() {
        // 1/3 rounds up to nearest (0x3EAAAAAB): RP agrees, RZ/RM truncate.
        let mut e = Env {
            fpcr: 1 << 22,
            fpsr: 0,
        }; // RP
        let r = div(1.0f32.to_bits().into(), 3.0f32.to_bits().into(), S, &mut e);
        assert_eq!(r as u32, (1.0f32 / 3.0).to_bits());
        let mut e = Env {
            fpcr: 3 << 22,
            fpsr: 0,
        }; // RZ
        let r = div(1.0f32.to_bits().into(), 3.0f32.to_bits().into(), S, &mut e);
        assert_eq!(r as u32, (1.0f32 / 3.0).to_bits() - 1);
    }

    #[test]
    fn nan_propagation_and_default_nan() {
        let mut e = env();
        let snan = 0x7FA0_0001u64;
        let r = add(1.0f32.to_bits().into(), snan, false, S, &mut e);
        assert_eq!(r, 0x7FE0_0001);
        assert_eq!(e.fpsr, IOC);
        let mut e = env();
        let r = mul(f32::INFINITY.to_bits().into(), 0, false, S, &mut e);
        assert_eq!(r, 0x7FC0_0000);
        assert_eq!(e.fpsr, IOC);
    }

    #[test]
    fn subnormal_flush_and_underflow() {
        // 2^-126 * 0.5 = 2^-127: subnormal, exact -> no UFC.
        let mut e = env();
        let r = mul(
            f32::MIN_POSITIVE.to_bits().into(),
            0.5f32.to_bits().into(),
            false,
            S,
            &mut e,
        );
        assert_eq!(f32::from_bits(r as u32), f32::MIN_POSITIVE / 2.0);
        assert_eq!(e.fpsr, 0);
        // Same under FZ: flushed to zero with UFC.
        let mut e = Env {
            fpcr: FPCR_FZ,
            fpsr: 0,
        };
        let r = mul(
            f32::MIN_POSITIVE.to_bits().into(),
            0.5f32.to_bits().into(),
            false,
            S,
            &mut e,
        );
        assert_eq!(r, 0);
        assert_eq!(e.fpsr, UFC);
    }

    #[test]
    fn fused_multiply_add_single_rounding() {
        let mut e = env();
        let a = 1.0f64 + f64::EPSILON;
        let r = mul_add((-1.0f64).to_bits(), a.to_bits(), a.to_bits(), D, &mut e);
        assert_eq!(f64::from_bits(r), a.mul_add(a, -1.0));
    }

    #[test]
    fn sqrt_and_conversions() {
        let mut e = env();
        assert_eq!(
            f64::from_bits(sqrt(2.0f64.to_bits(), D, &mut e)),
            2.0f64.sqrt()
        );
        let mut e = env();
        let h = convert(1.0f32.to_bits().into(), S, H, Rounding::TieEven, &mut e);
        assert_eq!(h, 0x3C00);
        let mut e = env();
        assert_eq!(
            to_fixed(
                (-1.5f64).to_bits(),
                D,
                0,
                false,
                32,
                Rounding::TieAway,
                &mut e
            ),
            u64::from((-2i32) as u32)
        );
        assert_eq!(e.fpsr, IXC);
    }

    #[test]
    fn recip_estimate_table_values() {
        let mut e = env();
        // FRECPE(1.0) = 0.998046875 (0x3F7F8000) on real hardware.
        assert_eq!(recip_est(1.0f32.to_bits().into(), S, &mut e), 0x3F7F_8000);
        // FRSQRTE(1.0) = 0.998046875, FRSQRTE(4.0) = 0.4990234375.
        assert_eq!(rsqrt_est(1.0f32.to_bits().into(), S, &mut e), 0x3F7F_8000);
        assert_eq!(rsqrt_est(4.0f32.to_bits().into(), S, &mut e), 0x3EFF_8000);
    }
}
