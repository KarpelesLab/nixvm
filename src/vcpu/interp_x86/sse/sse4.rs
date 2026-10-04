//! The x86-64-v2 SIMD extensions: SSE3 (in the two-byte map, see
//! [`X86Interp::exec_sse3`]), SSSE3 and SSE4.1/SSE4.2 (the three-byte maps
//! `0F 38` and `0F 3A`), including `PCMPxSTRx` and `CRC32`.

#![allow(clippy::too_many_lines)]

use super::{Ff, Mp, arith_lane, lane, map2, mp, pack, sat_s, set_lane, sx};
use crate::vcpu::GuestMemory;
use crate::vcpu::interp_x86::{Flags, Pfx, RAX, RCX, RDX, RmKind, Step, X86Interp, fetch};
use crate::vcpu::softfloat::{self as sf, Mx, Op, Round};

/// SSSE3 and SSE4.1/SSE4.2: `#UD` while a switch is off.
pub(in crate::vcpu::interp_x86) const SSSE3: bool = true;
pub(in crate::vcpu::interp_x86) const SSE41: bool = true;
pub(in crate::vcpu::interp_x86) const SSE42: bool = true;

/// The CRC-32C (Castagnoli, reflected polynomial `0x82F63B78`) byte table.
const CRC32C: [u32; 256] = {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                (c >> 1) ^ 0x82F6_3B78
            } else {
                c >> 1
            };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
};

/// SSSE3 binary ops over `n` bytes (8 for MMX, 16 for XMM).
pub(super) fn ssse3_op(op: u8, a: u128, b: u128, n: usize) -> Option<u128> {
    let half = |w: usize| n / w / 2; // lanes per operand half for horizontal ops
    let hz = |w: usize, f: &dyn Fn(u64, u64) -> u64| -> u128 {
        let bits = 8 * w;
        let m = (1u128 << bits) - 1;
        let mut out = 0u128;
        for (k, v) in [a, b].iter().enumerate() {
            for i in 0..half(w) {
                let r = f(lane(*v, w, 2 * i), lane(*v, w, 2 * i + 1));
                out |= (u128::from(r) & m) << (bits * (k * half(w) + i));
            }
        }
        out
    };
    Some(match op {
        0x00 => {
            // PSHUFB
            let mut out = 0u128;
            for i in 0..n {
                let s = lane(b, 1, i);
                if s & 0x80 == 0 {
                    out |= u128::from(lane(a, 1, (s as usize) & (n - 1))) << (8 * i);
                }
            }
            out
        }
        0x01 => hz(2, &|x, y| x.wrapping_add(y)),
        0x02 => hz(4, &|x, y| x.wrapping_add(y)),
        0x03 => hz(2, &|x, y| sat_s(sx(x, 2) + sx(y, 2), 2)),
        0x05 => hz(2, &|x, y| x.wrapping_sub(y)),
        0x06 => hz(4, &|x, y| x.wrapping_sub(y)),
        0x07 => hz(2, &|x, y| sat_s(sx(x, 2) - sx(y, 2), 2)),
        0x04 => map2(a, b, 2, n, |x, y| {
            // PMADDUBSW: unsigned bytes of a times signed bytes of b.
            let p0 = (x & 0xff) as i64 * sx(y & 0xff, 1);
            let p1 = ((x >> 8) & 0xff) as i64 * sx((y >> 8) & 0xff, 1);
            sat_s(p0 + p1, 2)
        }),
        0x08..=0x0A => {
            let w = 1 << (op - 8);
            map2(a, b, w, n, |x, y| match sx(y, w).cmp(&0) {
                core::cmp::Ordering::Less => (sx(x, w).wrapping_neg()) as u64,
                core::cmp::Ordering::Equal => 0,
                core::cmp::Ordering::Greater => x,
            })
        }
        0x0B => map2(a, b, 2, n, |x, y| {
            // PMULHRSW
            ((((sx(x, 2) * sx(y, 2)) >> 14) + 1) >> 1) as u64
        }),
        0x1C..=0x1E => {
            let w = 1 << (op - 0x1C);
            map2(b, 0, w, n, |x, _| sx(x, w).unsigned_abs())
        }
        _ => return None,
    })
}

/// `PALIGNR`: the `2n`-byte concatenation `a:b` shifted right by `imm` bytes.
pub(super) fn palignr(a: u128, b: u128, imm: u8, n: usize) -> u128 {
    let mut out = 0u128;
    for i in 0..n {
        let k = usize::from(imm) + i;
        let byte = if k < n {
            lane(b, 1, k)
        } else if k < 2 * n {
            lane(a, 1, k - n)
        } else {
            0
        };
        out |= u128::from(byte) << (8 * i);
    }
    out
}

/// `PMOVSX*`/`PMOVZX*`: widen the low lanes of `v` from `from` to `to` bytes.
pub(super) fn pmov_ext(v: u128, from: usize, to: usize, signed: bool) -> u128 {
    let mut out = 0u128;
    let m = if to == 8 {
        u128::from(u64::MAX)
    } else {
        (1u128 << (8 * to)) - 1
    };
    for i in 0..16 / to {
        let x = lane(v, from, i);
        let x = if signed { sx(x, from) as u64 } else { x };
        out |= (u128::from(x) & m) << (8 * to * i);
    }
    out
}

/// The SSE4.1/SSE4.2 binary integer ops of the `0F 38` map that are pure
/// functions of the two operands (`PMULDQ`, `PCMPEQQ`, `PACKUSDW`,
/// `PCMPGTQ`, `PMIN*`/`PMAX*`, `PMULLD`); `None` for other opcodes.
pub(super) fn sse41_op(op: u8, a: u128, b: u128) -> Option<u128> {
    Some(match op {
        0x28 => map2(a, b, 8, 16, |x, y| {
            (sx(x & 0xffff_ffff, 4).wrapping_mul(sx(y & 0xffff_ffff, 4))) as u64
        }),
        0x29 => map2(a, b, 8, 16, |x, y| if x == y { u64::MAX } else { 0 }),
        0x2B => pack(a, b, 4, 16, false),
        0x37 if SSE42 => map2(a, b, 8, 16, |x, y| {
            if (x as i64) > (y as i64) { u64::MAX } else { 0 }
        }),
        0x38 => map2(a, b, 1, 16, |x, y| if sx(x, 1) < sx(y, 1) { x } else { y }),
        0x39 => map2(a, b, 4, 16, |x, y| if sx(x, 4) < sx(y, 4) { x } else { y }),
        0x3A => map2(a, b, 2, 16, u64::min),
        0x3B => map2(a, b, 4, 16, u64::min),
        0x3C => map2(a, b, 1, 16, |x, y| if sx(x, 1) > sx(y, 1) { x } else { y }),
        0x3D => map2(a, b, 4, 16, |x, y| if sx(x, 4) > sx(y, 4) { x } else { y }),
        0x3E => map2(a, b, 2, 16, u64::max),
        0x3F => map2(a, b, 4, 16, u64::max),
        0x40 => map2(a, b, 4, 16, u64::wrapping_mul),
        _ => return None,
    })
}

/// `PHMINPOSUW`: the minimum unsigned word of `b` and its index.
pub(super) fn phminposuw(b: u128) -> u128 {
    let (mut min, mut idx) = (lane(b, 2, 0), 0u64);
    for i in 1..8 {
        let v = lane(b, 2, i);
        if v < min {
            min = v;
            idx = i as u64;
        }
    }
    u128::from(min | (idx << 16))
}

/// `BLENDPS`/`BLENDPD`/`PBLENDW` (`0F 3A 0C/0D/0E`): lane `i` from `b` when
/// `imm` bit `i` is set.
pub(super) fn blend_imm(op: u8, a: u128, b: u128, imm: u8) -> u128 {
    let w = match op {
        0x0C => 4,
        0x0D => 8,
        _ => 2,
    };
    let mut out = 0u128;
    for i in 0..16 / w {
        let s = if (imm >> i) & 1 != 0 { b } else { a };
        out |= u128::from(lane(s, w, i)) << (8 * w * i);
    }
    out
}

/// `PBLENDVB`/`BLENDVPS`/`BLENDVPD` (`w` = 1/4/8): lane from `b` where the
/// mask lane's sign bit is set.
pub(super) fn blendv(a: u128, b: u128, mask: u128, w: usize) -> u128 {
    let mut out = 0u128;
    for i in 0..16 / w {
        let s = if lane(mask, w, i) >> (8 * w - 1) != 0 {
            b
        } else {
            a
        };
        out = set_lane(out, w, i, lane(s, w, i));
    }
    out
}

/// `MPSADBW`: eight sums of absolute differences of 4-byte groups — `a` at
/// byte offsets `imm[2]·4 + i`, `b` at `imm[1:0]·4`.
pub(super) fn mpsadbw(a: u128, b: u128, imm: u8) -> u128 {
    let ao = usize::from((imm >> 2) & 1) * 4;
    let bo = usize::from(imm & 3) * 4;
    let mut out = 0u128;
    for i in 0..8 {
        let s: u64 = (0..4)
            .map(|k| lane(a, 1, ao + i + k).abs_diff(lane(b, 1, bo + k)))
            .sum();
        out |= u128::from(s) << (16 * i);
    }
    out
}

/// `ADDSUBPS/PD` (`kind` 0), `HADDPS/PD` (1), `HSUBPS/PD` (2).
pub(super) fn fp_horiz(ff: Ff, kind: u8, a: u128, b: u128, mx: Mx) -> (u128, u32) {
    let w = ff.bytes();
    let lanes = 16 / w;
    let mut out = 0u128;
    let mut flags = 0;
    for i in 0..lanes {
        let (x, y, o) = match kind {
            0 => (
                lane(a, w, i),
                lane(b, w, i),
                if i % 2 == 0 { Op::Sub } else { Op::Add },
            ),
            _ => {
                let src = if i < lanes / 2 { a } else { b };
                let k = (i % (lanes / 2)) * 2;
                (
                    lane(src, w, k),
                    lane(src, w, k + 1),
                    if kind == 1 { Op::Add } else { Op::Sub },
                )
            }
        };
        let (r, f) = arith_lane(o, ff, x, y, mx);
        out |= u128::from(r) << (8 * w * i);
        flags |= f;
    }
    (out, flags)
}

/// `ROUNDPS`/`ROUNDPD`/`ROUNDSS`/`ROUNDSD` (`op` = `0F 3A 08..0B`) of `b`;
/// the scalar forms keep `a`'s upper lanes.
pub(super) fn round_lanes(op: u8, a: u128, b: u128, imm: u8, mx: Mx) -> (u128, u32) {
    let (ff, lanes) = match op {
        0x08 => (Ff::S, 4),
        0x09 => (Ff::D, 2),
        0x0A => (Ff::S, 1),
        _ => (Ff::D, 1),
    };
    let mode = if imm & 4 != 0 {
        mx.mode
    } else {
        Round::from_x86(u32::from(imm & 3))
    };
    let w = ff.bytes();
    let mut out = a;
    let mut flags = 0;
    for i in 0..lanes {
        let (u, _) = mx.input(ff.unpack(lane(b, w, i)), ff.fmt());
        let (r, f) = if u.is_nan() {
            (
                ff.pack(&u.quieted()),
                if u.is_snan() { sf::INVALID } else { 0 },
            )
        } else {
            let rr = sf::round_to_int(u, mode);
            let fl = if imm & 8 != 0 { 0 } else { rr.flags };
            // Re-pack through the format (an integral value is exact).
            let v = sf::round_fp(rr.v, ff.fmt(), Round::Nearest).v;
            (ff.pack(&v), fl)
        };
        out = set_lane(out, w, i, r);
        flags |= f;
    }
    (out, flags)
}

/// `DPPS`/`DPPD`: the dot product of the lanes selected by `imm[7:4]`,
/// broadcast to the lanes selected by `imm[3:0]` (others zeroed).
pub(super) fn dpp(a: u128, b: u128, single: bool, imm: u8, mx: Mx) -> (u128, u32) {
    let (ff, lanes) = if single { (Ff::S, 4) } else { (Ff::D, 2) };
    let w = ff.bytes();
    let mut flags = 0;
    let zero = 0u64; // +0.0 in either format
    let mut prods = [zero; 4];
    for (i, pr) in prods.iter_mut().enumerate().take(lanes) {
        if (imm >> (4 + i)) & 1 != 0 {
            let (r, f) = arith_lane(Op::Mul, ff, lane(a, w, i), lane(b, w, i), mx);
            *pr = r;
            flags |= f;
        }
    }
    let mut add = |x: u64, y: u64| -> u64 {
        let (r, f) = arith_lane(Op::Add, ff, x, y, mx);
        flags |= f;
        r
    };
    let sum = if single {
        let t2 = add(prods[0], prods[1]);
        let t3 = add(prods[2], prods[3]);
        add(t2, t3)
    } else {
        add(prods[0], prods[1])
    };
    let mut out = 0u128;
    for i in 0..lanes {
        if (imm >> i) & 1 != 0 {
            out |= u128::from(sum) << (8 * w * i);
        }
    }
    (out, flags)
}

/// The element count, format and per-element accessor of a `PCMPxSTRx`.
struct StrFmt {
    n: usize,
    words: bool,
    signed: bool,
}

impl StrFmt {
    fn new(imm: u8) -> Self {
        let words = imm & 1 != 0;
        Self {
            n: if words { 8 } else { 16 },
            words,
            signed: imm & 2 != 0,
        }
    }
    fn get(&self, v: u128, i: usize) -> i64 {
        let w = if self.words { 2 } else { 1 };
        let x = lane(v, w, i);
        if self.signed { sx(x, w) } else { x as i64 }
    }
    /// The implicit length: the index of the first zero element.
    fn implicit_len(&self, v: u128) -> usize {
        (0..self.n).find(|&i| self.get(v, i) == 0).unwrap_or(self.n)
    }
}

/// Evaluate `PCMPxSTRx`: returns `IntRes2` and the flags.
fn pcmpstr(a: u128, b: u128, la: usize, lb: usize, imm: u8) -> (u32, Flags) {
    let f = StrFmt::new(imm);
    let n = f.n;
    let va = |i: usize| i < la;
    let vb = |j: usize| j < lb;
    let mut res1: u32 = 0;
    match (imm >> 2) & 3 {
        0 => {
            // equal any
            for j in 0..n {
                let hit = vb(j) && (0..la).any(|i| f.get(a, i) == f.get(b, j));
                res1 |= u32::from(hit) << j;
            }
        }
        1 => {
            // ranges
            for j in 0..n {
                let hit = vb(j)
                    && (0..n / 2).any(|k| {
                        let (lo, hi) = (2 * k, 2 * k + 1);
                        hi < la && f.get(a, lo) <= f.get(b, j) && f.get(b, j) <= f.get(a, hi)
                    });
                res1 |= u32::from(hit) << j;
            }
        }
        2 => {
            // equal each
            for i in 0..n {
                let hit = match (va(i), vb(i)) {
                    (true, true) => f.get(a, i) == f.get(b, i),
                    (false, false) => true,
                    _ => false,
                };
                res1 |= u32::from(hit) << i;
            }
        }
        _ => {
            // equal ordered
            for j in 0..n {
                let mut hit = true;
                for k in 0..n - j {
                    let (ia, ib) = (k, j + k);
                    let eq = match (va(ia), vb(ib)) {
                        (false, _) => true,
                        (true, false) => false,
                        (true, true) => f.get(a, ia) == f.get(b, ib),
                    };
                    if !eq {
                        hit = false;
                        break;
                    }
                    if !va(ia) {
                        break;
                    }
                }
                res1 |= u32::from(hit) << j;
            }
        }
    }
    let full: u32 = if n == 16 { 0xffff } else { 0xff };
    let valid_b: u32 = if lb >= 32 {
        full
    } else {
        ((1u32 << lb) - 1) & full
    };
    let res2 = match (imm >> 4) & 3 {
        1 => !res1 & full,
        3 => res1 ^ valid_b,
        _ => res1,
    } & full;
    let flags = Flags {
        cf: res2 != 0,
        pf: false,
        af: false,
        zf: lb < n,
        sf: la < n,
        of: res2 & 1 != 0,
    };
    (res2, flags)
}

impl X86Interp {
    /// The SSE3 opcodes of the two-byte map; `None` when `op`/`mp` isn't one.
    pub(in crate::vcpu::interp_x86) fn exec_sse3(
        &mut self,
        mem: &GuestMemory,
        pc: u64,
        p: Pfx,
        op: u8,
    ) -> Option<Step> {
        if !super::SSE3 {
            return None;
        }
        let mp = mp(p);
        let (ff, kind) = match (op, mp) {
            (0xD0, Mp::P66) => (Ff::D, 0),
            (0xD0, Mp::F2) => (Ff::S, 0),
            (0x7C, Mp::P66) => (Ff::D, 1),
            (0x7C, Mp::F2) => (Ff::S, 1),
            (0x7D, Mp::P66) => (Ff::D, 2),
            (0x7D, Mp::F2) => (Ff::S, 2),
            (0xF0, Mp::F2) => {
                // LDDQU xmm, m128 (unaligned; memory only).
                let (m, end) = match self.modrm(pc, p.rex) {
                    Ok(v) => v,
                    Err(s) => return Some(s),
                };
                let a = match self.mem_only(m.kind, end) {
                    Ok(a) => a,
                    Err(s) => return Some(s),
                };
                return Some(match Self::mem_read(mem, a, 16, 1) {
                    Ok(v) => {
                        self.xmm[m.reg] = v;
                        self.next(end)
                    }
                    Err(s) => s,
                });
            }
            _ => return None,
        };
        Some(self.sse3_arith(mem, pc, p, ff, kind))
    }

    /// `ADDSUBPS/PD` (`kind` 0), `HADDPS/PD` (1), `HSUBPS/PD` (2).
    fn sse3_arith(&mut self, mem: &GuestMemory, pc: u64, p: Pfx, ff: Ff, kind: u8) -> Step {
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        let b = fetch!(self.xsrc(mem, &m, end, 16, true));
        let (out, flags) = fp_horiz(ff, kind, self.xmm[m.reg], b, self.mx());
        fetch!(self.sse_flags(flags));
        self.xmm[m.reg] = out;
        self.next(end)
    }

    /// The three-byte map `0F 38` (`op` = third opcode byte; ModRM at `pc`).
    pub(in crate::vcpu::interp_x86) fn exec_0f38(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        p: Pfx,
        op: u8,
    ) -> Step {
        let mp = mp(p);
        // CRC32 (F2 0F 38 F0/F1): a general-purpose instruction.
        if SSE42 && matches!(op, 0xF0 | 0xF1) && p.rep == 2 {
            let (m, end) = fetch!(self.modrm(pc, p.rex));
            let w = if op == 0xF0 {
                8
            } else if p.rex.w {
                64
            } else if p.opsize {
                16
            } else {
                32
            };
            let src = self.opw_of(m.kind, end, w, p);
            let v = fetch!(self.read_operand(mem, src, w));
            let mut crc = self.gpr[m.reg] as u32;
            for k in 0..w / 8 {
                let byte = (v >> (8 * k)) as u8;
                crc = (crc >> 8) ^ CRC32C[usize::from((crc as u8) ^ byte)];
            }
            self.gpr[m.reg] = u64::from(crc);
            return self.next(end);
        }
        // SSSE3: MMX (no prefix) and XMM (66) forms.
        if SSSE3 && matches!(op, 0x00..=0x0B | 0x1C..=0x1E) && matches!(mp, Mp::None | Mp::P66) {
            let (m, end) = fetch!(self.modrm(pc, p.rex));
            if mp == Mp::None {
                let b = fetch!(self.msrc(mem, &m, end));
                fetch!(self.mmx_enter());
                let a = self.mm_get(m.reg);
                let r = ssse3_op(op, u128::from(a), u128::from(b), 8).unwrap_or(0);
                self.mm_set(m.reg, r as u64);
            } else {
                let b = fetch!(self.xsrc(mem, &m, end, 16, true));
                self.xmm[m.reg] = ssse3_op(op, self.xmm[m.reg], b, 16).unwrap_or(0);
            }
            return self.next(end);
        }
        if mp != Mp::P66 || !SSE41 {
            return Step::Illegal;
        }
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        let x0 = self.xmm[0];
        let a = self.xmm[m.reg];
        // The PMOVSX/ZX forms read a narrower memory operand.
        let src_n = match op {
            0x20 | 0x23 | 0x25 | 0x30 | 0x33 | 0x35 => 8,
            0x21 | 0x24 | 0x31 | 0x34 => 4,
            0x22 | 0x32 => 2,
            _ => 16,
        };
        let b = fetch!(self.xsrc(mem, &m, end, src_n, true));
        if let Some(r) = sse41_op(op, a, b) {
            self.xmm[m.reg] = r;
            return self.next(end);
        }
        let r = match op {
            // PBLENDVB / BLENDVPS / BLENDVPD (mask: XMM0 lane sign bits)
            0x10 => blendv(a, b, x0, 1),
            0x14 => blendv(a, b, x0, 4),
            0x15 => blendv(a, b, x0, 8),
            0x17 => {
                // PTEST
                self.flags = Flags {
                    zf: a & b == 0,
                    cf: !a & b == 0,
                    ..Flags::default()
                };
                return self.next(end);
            }
            0x20 => pmov_ext(b, 1, 2, true),
            0x21 => pmov_ext(b, 1, 4, true),
            0x22 => pmov_ext(b, 1, 8, true),
            0x23 => pmov_ext(b, 2, 4, true),
            0x24 => pmov_ext(b, 2, 8, true),
            0x25 => pmov_ext(b, 4, 8, true),
            0x30 => pmov_ext(b, 1, 2, false),
            0x31 => pmov_ext(b, 1, 4, false),
            0x32 => pmov_ext(b, 1, 8, false),
            0x33 => pmov_ext(b, 2, 4, false),
            0x34 => pmov_ext(b, 2, 8, false),
            0x35 => pmov_ext(b, 4, 8, false),
            0x2A => {
                // MOVNTDQA xmm, m128 (aligned load; memory only)
                let addr = fetch!(self.mem_only(m.kind, end));
                fetch!(Self::mem_read(mem, addr, 16, 16))
            }
            0x41 => phminposuw(b),
            _ => return Step::Illegal,
        };
        self.xmm[m.reg] = r;
        self.next(end)
    }

    /// The three-byte map `0F 3A` (immediate forms; ModRM at `pc`).
    pub(in crate::vcpu::interp_x86) fn exec_0f3a(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        p: Pfx,
        op: u8,
    ) -> Step {
        let mp = mp(p);
        if SSSE3 && op == 0x0F && matches!(mp, Mp::None | Mp::P66) {
            // PALIGNR
            let (m, imm, end) = fetch!(self.modrm_imm(pc, p, true));
            if mp == Mp::None {
                let b = fetch!(self.msrc(mem, &m, end));
                fetch!(self.mmx_enter());
                let a = self.mm_get(m.reg);
                let r = palignr(u128::from(a), u128::from(b), imm, 8);
                self.mm_set(m.reg, r as u64);
            } else {
                let b = fetch!(self.xsrc(mem, &m, end, 16, true));
                self.xmm[m.reg] = palignr(self.xmm[m.reg], b, imm, 16);
            }
            return self.next(end);
        }
        if mp != Mp::P66 || !SSE41 {
            return Step::Illegal;
        }
        let (m, imm, end) = fetch!(self.modrm_imm(pc, p, true));
        match op {
            0x0C..=0x0E => {
                // BLENDPS / BLENDPD / PBLENDW
                let b = fetch!(self.xsrc(mem, &m, end, 16, true));
                self.xmm[m.reg] = blend_imm(op, self.xmm[m.reg], b, imm);
                self.next(end)
            }
            0x14..=0x17 => {
                // PEXTRB / PEXTRW / PEXTRD|Q / EXTRACTPS → r/m
                let v = self.xmm[m.reg];
                let (w, val) = match op {
                    0x14 => (8, lane(v, 1, usize::from(imm & 15))),
                    0x15 => (16, lane(v, 2, usize::from(imm & 7))),
                    0x16 if p.rex.w => (64, lane(v, 8, usize::from(imm & 1))),
                    _ => (32, lane(v, 4, usize::from(imm & 3))),
                };
                match m.kind {
                    // A register destination is written whole (zero-extended).
                    RmKind::Reg(r) => {
                        self.gpr[r] = val;
                    }
                    _ => {
                        let a = self.lin(self.ea_of(m.kind, end).unwrap_or(0));
                        fetch!(self.write_mem(mem, a, val, w));
                    }
                }
                self.next(end)
            }
            0x20 | 0x22 => {
                // PINSRB / PINSRD|Q
                let (w, idx) = match op {
                    0x20 => (8, usize::from(imm & 15)),
                    _ if p.rex.w => (64, usize::from(imm & 1)),
                    _ => (32, usize::from(imm & 3)),
                };
                let v = match m.kind {
                    RmKind::Reg(r) => self.gpr[r],
                    _ => {
                        let a = self.lin(self.ea_of(m.kind, end).unwrap_or(0));
                        fetch!(Self::read_mem(mem, a, w))
                    }
                };
                let bits = w as usize;
                let lm = if bits == 64 {
                    u128::from(u64::MAX)
                } else {
                    (1u128 << bits) - 1
                };
                let sh = bits * idx;
                let x = &mut self.xmm[m.reg];
                *x = (*x & !(lm << sh)) | ((u128::from(v) & lm) << sh);
                self.next(end)
            }
            0x21 => {
                // INSERTPS
                let src = match m.kind {
                    RmKind::Reg(r) => lane(self.xmm[r], 4, usize::from(imm >> 6)),
                    _ => {
                        let a = self.lin(self.ea_of(m.kind, end).unwrap_or(0));
                        fetch!(Self::read_mem(mem, a, 32))
                    }
                };
                let d = usize::from((imm >> 4) & 3);
                let mut x = self.xmm[m.reg];
                x = (x & !(0xffff_ffffu128 << (32 * d))) | (u128::from(src) << (32 * d));
                for i in 0..4 {
                    if (imm >> i) & 1 != 0 {
                        x &= !(0xffff_ffffu128 << (32 * i));
                    }
                }
                self.xmm[m.reg] = x;
                self.next(end)
            }
            0x08..=0x0B | 0x40 | 0x41 => {
                // ROUNDPS/PD/SS/SD, DPPS/DPPD
                let n = match op {
                    0x0A => 4,
                    0x0B => 8,
                    _ => 16,
                };
                let b = fetch!(self.xsrc(mem, &m, end, n, true));
                let (a, mx) = (self.xmm[m.reg], self.mx());
                let (r, flags) = if op >= 0x40 {
                    dpp(a, b, op == 0x40, imm, mx)
                } else {
                    round_lanes(op, a, b, imm, mx)
                };
                fetch!(self.sse_flags(flags));
                self.xmm[m.reg] = r;
                self.next(end)
            }
            0x42 => {
                // MPSADBW
                let b = fetch!(self.xsrc(mem, &m, end, 16, true));
                self.xmm[m.reg] = mpsadbw(self.xmm[m.reg], b, imm);
                self.next(end)
            }
            0x60..=0x63 if SSE42 => {
                // PCMPESTRM / PCMPESTRI / PCMPISTRM / PCMPISTRI
                let b = fetch!(self.xsrc(mem, &m, end, 16, false));
                self.pcmpstr_run(op, imm, self.xmm[m.reg], b, p.rex.w);
                self.next(end)
            }
            _ => Step::Illegal,
        }
    }

    /// Run `PCMPxSTRx` (`op` = `0F 3A 60..63`) on `a`/`b` — explicit lengths
    /// from `EAX`/`EDX` (`RAX`/`RDX` when `wide`) or implicit (NUL-terminated):
    /// the flags, then the index into `ECX` (`op` odd) or the mask into
    /// `XMM0`.
    pub(super) fn pcmpstr_run(&mut self, op: u8, imm: u8, a: u128, b: u128, wide: bool) {
        let f = StrFmt::new(imm);
        let (la, lb) = if op <= 0x61 {
            let len = |r: usize| -> usize {
                let v = if wide {
                    self.gpr[r] as i64
                } else {
                    i64::from(self.gpr[r] as u32 as i32)
                };
                usize::try_from(v.unsigned_abs().min(f.n as u64)).unwrap_or(f.n)
            };
            (len(RAX), len(RDX))
        } else {
            (f.implicit_len(a), f.implicit_len(b))
        };
        let (res, flags) = pcmpstr(a, b, la, lb, imm);
        self.flags = flags;
        if op & 1 == 1 {
            // index → ECX
            let idx = if res == 0 {
                f.n as u32
            } else if imm & 0x40 != 0 {
                res.ilog2()
            } else {
                res.trailing_zeros()
            };
            self.gpr[RCX] = u64::from(idx);
        } else {
            // mask → XMM0
            self.xmm[0] = if imm & 0x40 == 0 {
                u128::from(res)
            } else {
                let w = if f.words { 2 } else { 1 };
                let ones = if w == 2 { 0xffffu128 } else { 0xff };
                (0..f.n).fold(0u128, |acc, i| {
                    acc | if (res >> i) & 1 != 0 {
                        ones << (8 * w * i)
                    } else {
                        0
                    }
                })
            };
        }
    }
}
