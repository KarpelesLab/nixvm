//! Soft-float correctness tests.
//!
//! The load-bearing check: in round-to-nearest, the soft `f32`/`f64` results
//! must be **bit-identical** to the host's native hardware arithmetic. Since
//! `f32`/`f64` and `f80` share one code path (only [`Fmt`] differs), agreement
//! on the two formats the host can check gives strong evidence for `f80`, which
//! the KVM differential harness pins against real x87 hardware separately.

// Oracle tests intentionally cast ints to floats and compare floats exactly
// (against native hardware); those lints fight the whole point here.
#![allow(clippy::cast_precision_loss, clippy::float_cmp)]

use super::*;

/// Deterministic xorshift64* PRNG (no `rand` dep, reproducible across runs).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

/// A random `f64` bit pattern with a distribution that stresses subnormals,
/// tiny/huge exponents, cancellation-prone neighbours, and (rarely) inf/NaN.
fn rand_f64(rng: &mut Rng) -> u64 {
    let r = rng.next();
    let sign = (r & 1) << 63;
    match r % 16 {
        0 => r,                                      // fully random: includes inf/NaN/subnormal
        1 => sign | ((r >> 8) & 0xf_ffff_ffff_ffff), // subnormal (exp field 0)
        2 => sign | (0x7fe << 52) | ((r >> 8) & 0xf_ffff_ffff_ffff), // near overflow
        3 => sign | (0x001 << 52) | ((r >> 8) & 0xf_ffff_ffff_ffff), // near underflow
        _ => {
            // Normal with a random but bounded exponent.
            let exp = ((r >> 12) % 0x7fd) + 1;
            sign | (exp << 52) | ((r >> 20) & 0xf_ffff_ffff_ffff)
        }
    }
}

fn rand_f32(rng: &mut Rng) -> u32 {
    let r = rng.next();
    let sign = ((r & 1) as u32) << 31;
    match r % 16 {
        0 => r as u32,
        1 => sign | ((r >> 8) as u32 & 0x7f_ffff),
        2 => sign | (0xfe << 23) | ((r >> 8) as u32 & 0x7f_ffff),
        3 => sign | (0x01 << 23) | ((r >> 8) as u32 & 0x7f_ffff),
        _ => {
            let exp = (((r >> 12) % 0xfd) + 1) as u32;
            sign | (exp << 23) | ((r >> 20) as u32 & 0x7f_ffff)
        }
    }
}

/// An SSE environment with no DAZ/FTZ.
fn mx(mode: Round) -> Mx {
    Mx {
        mode,
        daz: false,
        ftz: false,
    }
}

fn f64_sqrt(a: u64, mode: Round) -> (u64, u32) {
    let (v, f) = sse_sqrt(unpack_f64(a), FMT64, mx(mode));
    (pack_f64(&v), f)
}

fn f32_sqrt(a: u32, mode: Round) -> (u32, u32) {
    let (v, f) = sse_sqrt(unpack_f32(a), FMT32, mx(mode));
    (pack_f32(&v), f)
}

fn i64_to_f64(v: i64, mode: Round) -> (u64, u32) {
    let r = round_fp(from_int(v), FMT64, mode);
    (pack_f64(&r.v), r.flags)
}

fn i64_to_f32(v: i64, mode: Round) -> (u32, u32) {
    let r = round_fp(from_int(v), FMT32, mode);
    (pack_f32(&r.v), r.flags)
}

fn f64_to_f32(a: u64, mode: Round) -> (u32, u32) {
    let r = round_fp(unpack_f64(a), FMT32, mode);
    (pack_f32(&r.v), r.flags)
}

fn f32_to_f64(a: u32) -> u64 {
    pack_f64(&unpack_f32(a))
}

fn f64_to_i64(a: u64, mode: Round) -> i64 {
    let v = unpack_f64(a);
    if v.is_nan() {
        return i64::MIN;
    }
    to_int(v, 64, mode).map_or(i64::MIN, |(i, _, _)| i)
}

fn f80_from(x: f64) -> F80 {
    F80::from_f64(x.to_bits())
}

fn f80_to_f64(x: F80) -> u64 {
    pack_f64(&round_fp(x.unpack(), FMT64, Round::Nearest).v)
}

fn f80_op(a: F80, b: F80, op: Op) -> F80 {
    let (a, b) = (a.unpack(), b.unpack());
    let r = match op {
        Op::Add => add(a, b, FMT80, Round::Nearest),
        Op::Sub => add(a, b.neg(), FMT80, Round::Nearest),
        Op::Mul => mul(a, b, FMT80, Round::Nearest),
        Op::Div => div(a, b, FMT80, Round::Nearest),
    };
    F80::pack(&r.v)
}

/// Iterations for the random sweeps; bump via `NIXVM_SF_ITERS` for a heavy run.
fn iters() -> u64 {
    std::env::var("NIXVM_SF_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000)
}

fn same_f64(soft: u64, native: f64) -> bool {
    if native.is_nan() {
        f64::from_bits(soft).is_nan()
    } else {
        soft == native.to_bits()
    }
}
fn same_f32(soft: u32, native: f32) -> bool {
    if native.is_nan() {
        f32::from_bits(soft).is_nan()
    } else {
        soft == native.to_bits()
    }
}

#[test]
fn f64_arith_matches_native_rne() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for _ in 0..iters() {
        let (a, b) = (rand_f64(&mut rng), rand_f64(&mut rng));
        let (fa, fb) = (f64::from_bits(a), f64::from_bits(b));
        for (op, native) in [
            (Op::Add, fa + fb),
            (Op::Sub, fa - fb),
            (Op::Mul, fa * fb),
            (Op::Div, fa / fb),
        ] {
            let (soft, _) = f64_op(a, b, op, mx(Round::Nearest));
            assert!(
                same_f64(soft, native),
                "f64 {op:?}: a={a:#018x} b={b:#018x} soft={soft:#018x} native={:#018x}",
                native.to_bits()
            );
        }
        let (soft, _) = f64_sqrt(a, Round::Nearest);
        assert!(same_f64(soft, fa.sqrt()), "f64 sqrt: a={a:#018x}");
    }
}

#[test]
fn f32_arith_matches_native_rne() {
    let mut rng = Rng(0x2222_3333_4444_5555);
    for _ in 0..iters() {
        let (a, b) = (rand_f32(&mut rng), rand_f32(&mut rng));
        let (fa, fb) = (f32::from_bits(a), f32::from_bits(b));
        for (op, native) in [
            (Op::Add, fa + fb),
            (Op::Sub, fa - fb),
            (Op::Mul, fa * fb),
            (Op::Div, fa / fb),
        ] {
            let (soft, _) = f32_op(a, b, op, mx(Round::Nearest));
            assert!(
                same_f32(soft, native),
                "f32 {op:?}: a={a:#010x} b={b:#010x} soft={soft:#010x} native={:#010x}",
                native.to_bits()
            );
        }
        let (soft, _) = f32_sqrt(a, Round::Nearest);
        assert!(same_f32(soft, fa.sqrt()), "f32 sqrt: a={a:#010x}");
    }
}

#[test]
fn conversions_match_native() {
    let mut rng = Rng(0xdead_beef_cafe_babe);
    for _ in 0..iters() {
        let v = rng.next() as i64;
        assert!(
            same_f64(i64_to_f64(v, Round::Nearest).0, v as f64),
            "i64->f64 {v}"
        );
        assert!(
            same_f32(i64_to_f32(v, Round::Nearest).0, v as f32),
            "i64->f32 {v}"
        );

        let a = rand_f64(&mut rng);
        // f64->f32 narrowing (RNE) must match native `as f32`.
        assert!(
            same_f32(f64_to_f32(a, Round::Nearest).0, f64::from_bits(a) as f32),
            "f64->f32 {a:#018x}"
        );
        // f32->f64 widening is exact.
        let s = rand_f32(&mut rng);
        assert!(
            same_f64(f32_to_f64(s), f64::from(f32::from_bits(s))),
            "f32->f64 {s:#010x}"
        );
    }
}

#[test]
fn f64_to_int_truncates_like_x86() {
    // In-range finite values truncate toward zero; NaN/overflow -> i64::MIN.
    let cases: &[(f64, i64)] = &[
        (0.0, 0),
        (2.9, 2),
        (-2.9, -2),
        (1e18, 1_000_000_000_000_000_000),
        (-1e18, -1_000_000_000_000_000_000),
        (9.3e18, i64::MIN), // overflow -> indefinite
        (f64::NAN, i64::MIN),
        (f64::INFINITY, i64::MIN),
    ];
    for &(x, want) in cases {
        assert_eq!(f64_to_i64(x.to_bits(), Round::Zero), want, "trunc {x}");
    }
}

#[test]
fn directed_rounding_one_third() {
    // 1/3 in each mode, hand-verified against the exact f64 neighbours.
    let one = 1.0f64.to_bits();
    let three = 3.0f64.to_bits();
    let rne = f64_op(one, three, Op::Div, mx(Round::Nearest)).0;
    let down = f64_op(one, three, Op::Div, mx(Round::Down)).0;
    let up = f64_op(one, three, Op::Div, mx(Round::Up)).0;
    let zero = f64_op(one, three, Op::Div, mx(Round::Zero)).0;
    // 1/3's exact tail is < 0.5 ulp, so RNE rounds *down*: nearest == down ==
    // zero, and up is one ulp above.
    assert_eq!(rne, down);
    assert_eq!(zero, down);
    assert_eq!(up, down + 1, "up is exactly one ulp above down");
    // Sanity: the RNE result is the native one.
    assert_eq!(rne, (1.0f64 / 3.0).to_bits());
}

#[test]
fn directed_rounding_negative_and_flags() {
    // -1/3: signs flip which directed mode rounds away.
    let a = (-1.0f64).to_bits();
    let three = 3.0f64.to_bits();
    let up = f64_op(a, three, Op::Div, mx(Round::Up)).0; // toward +inf: toward zero for negatives
    let down = f64_op(a, three, Op::Div, mx(Round::Down)).0; // toward -inf: away from zero
    let zero = f64_op(a, three, Op::Div, mx(Round::Zero)).0;
    assert_eq!(up, zero, "negative: toward +inf == toward zero");
    assert_eq!(
        down,
        up.wrapping_add(1),
        "toward -inf is one ulp more negative (larger magnitude)"
    );

    // Inexact flag is set for 1/3; exact ops don't set it.
    let (_, f_inexact) = f64_op(1.0f64.to_bits(), three, Op::Div, mx(Round::Nearest));
    assert!(f_inexact & INEXACT != 0);
    let (_, f_exact) = f64_op(
        1.0f64.to_bits(),
        2.0f64.to_bits(),
        Op::Div,
        mx(Round::Nearest),
    );
    assert!(f_exact & INEXACT == 0, "0.5 is exact");
    // Division by zero flags DIVZERO and yields signed infinity.
    let (q, f) = f64_op(
        1.0f64.to_bits(),
        0.0f64.to_bits(),
        Op::Div,
        mx(Round::Nearest),
    );
    assert!(f & DIVZERO != 0);
    assert_eq!(q, f64::INFINITY.to_bits());
}

#[test]
fn overflow_respects_direction() {
    let big = f64::MAX.to_bits();
    // MAX + MAX overflows: RNE -> +inf; toward zero -> stays MAX.
    assert_eq!(
        f64_op(big, big, Op::Add, mx(Round::Nearest)).0,
        f64::INFINITY.to_bits()
    );
    assert_eq!(
        f64_op(big, big, Op::Add, mx(Round::Zero)).0,
        f64::MAX.to_bits()
    );
    // Toward -inf keeps MAX for a positive overflow.
    assert_eq!(
        f64_op(big, big, Op::Add, mx(Round::Down)).0,
        f64::MAX.to_bits()
    );
    let (_, f) = f64_op(big, big, Op::Add, mx(Round::Nearest));
    assert_eq!(f & (OVERFLOW | INEXACT), OVERFLOW | INEXACT);
}

#[test]
fn f32_directed_rounding_matches_reference() {
    // For f32 add/sub/mul the exact result fits an f64 (V), so the correctly
    // directed-rounded f32 can be derived independently from V and the hardware
    // round-to-nearest value — no soft-float in the reference.
    let mut rng = Rng(0x5151_a7a7_9090_3c3c);
    for _ in 0..iters() {
        let a = rand_f32(&mut rng);
        let b = rand_f32(&mut rng);
        let (fa, fb) = (f64::from(f32::from_bits(a)), f64::from(f32::from_bits(b)));
        for (op, v) in [(Op::Add, fa + fb), (Op::Sub, fa - fb), (Op::Mul, fa * fb)] {
            if !v.is_finite() {
                continue; // overflow/NaN direction rules are checked elsewhere
            }
            // The reference is only valid when `v` is the *exact* result. f32
            // products always fit f64 (48 ≤ 53 bits), but a sum/difference of
            // widely-separated exponents rounds — detect that via 2Sum and skip.
            if matches!(op, Op::Add | Op::Sub) {
                let y = if matches!(op, Op::Sub) { -fb } else { fb };
                let bb = v - fa;
                let err = (fa - (v - bb)) + (y - bb);
                if err != 0.0 {
                    continue;
                }
            }
            let rne = v as f32; // hardware round-to-nearest of the exact value
            // Independent down/up references bracketing the exact V.
            let (down_ref, up_ref) = if f64::from(rne) == v {
                (rne, rne) // exact
            } else if f64::from(rne) > v {
                (next_f32(rne, f32::NEG_INFINITY), rne)
            } else {
                (rne, next_f32(rne, f32::INFINITY))
            };
            let down = f32::from_bits(f32_op(a, b, op, mx(Round::Down)).0);
            let up = f32::from_bits(f32_op(a, b, op, mx(Round::Up)).0);
            let zero = f32::from_bits(f32_op(a, b, op, mx(Round::Zero)).0);
            assert_eq!(
                down.to_bits(),
                down_ref.to_bits(),
                "{op:?} down {a:#x} {b:#x}"
            );
            assert_eq!(up.to_bits(), up_ref.to_bits(), "{op:?} up {a:#x} {b:#x}");
            // Toward zero == toward the smaller magnitude neighbour.
            let zref = if v >= 0.0 { down_ref } else { up_ref };
            assert_eq!(zero.to_bits(), zref.to_bits(), "{op:?} zero {a:#x} {b:#x}");
        }
    }
}

/// Next representable f32 from `x` toward `dir` (a tiny `nextafter`, avoiding a
/// libm dep). Only used for finite, non-equal inputs in the test above.
fn next_f32(x: f32, dir: f32) -> f32 {
    if x == dir {
        return x;
    }
    let bits = x.to_bits();
    let toward_larger = (dir > x) == (x >= 0.0);
    let next = if x == 0.0 {
        (if dir > 0.0 { 0 } else { 1u32 << 31 }) | 1
    } else if toward_larger {
        bits + 1
    } else {
        bits - 1
    };
    f32::from_bits(next)
}

#[test]
fn f80_round_trips_and_computes() {
    // f64 -> f80 -> f64 is the identity for finite values.
    let mut rng = Rng(0x0f0f_0f0f_1111_2222);
    for _ in 0..iters() {
        let a = rand_f64(&mut rng);
        let back = f80_to_f64(F80::from_f64(a));
        assert!(
            same_f64(back, f64::from_bits(a)),
            "f80 round-trip {a:#018x}"
        );
    }
    // A computation carrying more than 53 bits: (1 + 2^-60) done at 80-bit and
    // narrowed back to f64 keeps the extra bit that pure-f64 would have lost.
    let one = f80_from(1.0);
    let tiny = f80_from(2f64.powi(-60));
    let sum = f80_op(one, tiny, Op::Add); // exact at 80-bit (64-bit sig)
    let sub = f80_op(sum, one, Op::Sub); // recovers 2^-60 exactly
    assert_eq!(f80_to_f64(sub), 2f64.powi(-60).to_bits());
}

#[test]
fn f80_matches_f64_for_exact_ops() {
    // Where the operation is *exact* in both formats there is no rounding — and
    // so no double rounding — and 80-bit must equal native f64 bit for bit.
    // (Where it is inexact, 80-bit deliberately differs by double-rounding, as
    // real x87 does; that path is pinned against hardware by the KVM harness.)
    let mut rng = Rng(0x9999_8888_7777_6666);
    for _ in 0..iters() {
        // Integers < 2^26: sum and product are exact in f64 (< 2^53) and f80.
        let a = (rng.next() % (1 << 26)) as i64 - (1 << 25);
        let b = (rng.next() % (1 << 26)) as i64 - (1 << 25);
        let (xa, xb) = (f80_from(a as f64), f80_from(b as f64));
        let mul = f80_to_f64(f80_op(xa, xb, Op::Mul));
        assert_eq!(mul, ((a * b) as f64).to_bits(), "f80 mul {a}*{b}");
        let add = f80_to_f64(f80_op(xa, xb, Op::Add));
        assert_eq!(add, ((a + b) as f64).to_bits(), "f80 add {a}+{b}");
        let sub = f80_to_f64(f80_op(xa, xb, Op::Sub));
        assert_eq!(sub, ((a - b) as f64).to_bits(), "f80 sub {a}-{b}");
    }
    // A division that is exact at 80-bit: x/1 and (a*b)/b recover a.
    let seven = f80_from(7.0);
    assert_eq!(f80_op(seven, f80_from(1.0), Op::Div), seven);
    // Denormal round trip through the 80-bit format and arithmetic on it.
    let d = f80_from(f64::from_bits(1));
    assert_eq!(f80_to_f64(f80_op(d, f80_from(2.0), Op::Mul)), 2);
}
