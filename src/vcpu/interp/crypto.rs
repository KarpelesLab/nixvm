//! The Armv8 Cryptographic Extension (`AESE`/`AESD`/`AESMC`/`AESIMC`,
//! `SHA1*`, `SHA256*`), `PMULL`'s carry-less multiply, and the `CRC32`/`CRC32C`
//! instructions' checksum step — pure functions over register values.

/// `CRC32*`/`CRC32C*`: fold the low `nbytes` bytes of `val` into `crc` with the
/// reflected polynomial `poly` (`0xEDB8_8320` for CRC-32, `0x82F6_3B78` for
/// CRC-32C — the bit-reflected 0x04C11DB7/0x1EDC6F41).
pub(super) fn crc32(mut crc: u32, val: u64, nbytes: u32, poly: u32) -> u32 {
    for i in 0..nbytes {
        crc ^= u32::from((val >> (8 * i)) as u8);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ poly
            } else {
                crc >> 1
            };
        }
    }
    crc
}

/// Carry-less (GF(2) polynomial) product of two 64-bit values.
pub(super) fn clmul64(a: u64, b: u64) -> u128 {
    let a = u128::from(a);
    (0..64)
        .filter(|i| (b >> i) & 1 == 1)
        .fold(0u128, |acc, i| acc ^ (a << i))
}

// ---- Cryptographic Extension: AES ----
//
// All of this is the plain FIPS-197 algorithm: `AES_SBOX` is generated (in
// the test module, `aes_sbox_matches_generated_table`) from the GF(2^8)
// multiplicative inverse plus the standard affine transform rather than
// typed in from memory, and every transform below was checked against
// native execution of the real `AESE`/`AESD`/`AESMC`/`AESIMC` instructions
// on this host's Apple Silicon CPU (which implements FEAT_AES) — see the
// test module. That check also confirmed the 16-byte vector maps to the
// FIPS-197 state array exactly as `bytes[r + 4c]` (byte 0 = the vector's
// least-significant byte), so no ARM-specific re-layout is needed here.

/// Forward AES S-box (`SubBytes`).
pub(super) const AES_SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

/// Inverse AES S-box (`InvSubBytes`): `AES_INV_SBOX[AES_SBOX[b]] == b`.
pub(super) const AES_INV_SBOX: [u8; 256] = [
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d,
];

/// `AESE`/`AESD`: XOR `vd`/`vn` as a 16-byte state, then apply `ShiftRows`
/// (or its inverse) and `SubBytes` (or its inverse). The ARM ARM's
/// pseudocode order is actually `ShiftRows` then `SubBytes` (`InvShiftRows`
/// then `InvSubBytes` for `AESD`), but the two commute — `SubBytes` acts on
/// each byte independently of its position, and `ShiftRows` only permutes
/// positions — so applying them in the other order here gives the same
/// result.
pub(super) fn aes_round(vd: u128, vn: u128, encrypt: bool) -> u128 {
    let state = (vd ^ vn).to_le_bytes();
    let shifted = if encrypt {
        aes_shift_rows(state)
    } else {
        aes_inv_shift_rows(state)
    };
    let sbox = if encrypt { &AES_SBOX } else { &AES_INV_SBOX };
    u128::from_le_bytes(shifted.map(|b| sbox[b as usize]))
}

/// `AESMC`/`AESIMC`: `MixColumns` (or its inverse) over `vn` alone — unlike
/// `AESE`/`AESD`, `Vd`'s prior value isn't read, only overwritten.
pub(super) fn aes_mix_columns(vn: u128, forward: bool) -> u128 {
    let state = vn.to_le_bytes();
    let mut out = [0u8; 16];
    for (out_col, in_col) in out
        .as_chunks_mut::<4>()
        .0
        .iter_mut()
        .zip(state.as_chunks::<4>().0)
    {
        let a = *in_col;
        let r = if forward {
            aes_mix_column(a)
        } else {
            aes_inv_mix_column(a)
        };
        out_col.copy_from_slice(&r);
    }
    u128::from_le_bytes(out)
}

/// FIPS-197 `ShiftRows`: state byte `r + 4c` (row `r`, column `c`) moves to
/// `r + 4*((c+r) mod 4)` — row `r` is cyclically shifted left by `r`.
pub(super) fn aes_shift_rows(state: [u8; 16]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for r in 0..4usize {
        for c in 0..4usize {
            out[r + 4 * c] = state[r + 4 * ((c + r) % 4)];
        }
    }
    out
}

/// `InvShiftRows`, the inverse permutation of [`aes_shift_rows`].
pub(super) fn aes_inv_shift_rows(state: [u8; 16]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for r in 0..4usize {
        for c in 0..4usize {
            out[r + 4 * ((c + r) % 4)] = state[r + 4 * c];
        }
    }
    out
}

/// GF(2^8) multiplication modulo the AES reduction polynomial
/// `x^8 + x^4 + x^3 + x + 1` (`0x11B`).
pub(super) fn gf_mul(mut a: u8, mut b: u8) -> u8 {
    let mut p = 0u8;
    for _ in 0..8 {
        if b & 1 != 0 {
            p ^= a;
        }
        let hi = a & 0x80;
        a <<= 1;
        if hi != 0 {
            a ^= 0x1b;
        }
        b >>= 1;
    }
    p
}

/// FIPS-197 `MixColumns` on one 4-byte state column.
pub(super) fn aes_mix_column(a: [u8; 4]) -> [u8; 4] {
    [
        gf_mul(2, a[0]) ^ gf_mul(3, a[1]) ^ a[2] ^ a[3],
        a[0] ^ gf_mul(2, a[1]) ^ gf_mul(3, a[2]) ^ a[3],
        a[0] ^ a[1] ^ gf_mul(2, a[2]) ^ gf_mul(3, a[3]),
        gf_mul(3, a[0]) ^ a[1] ^ a[2] ^ gf_mul(2, a[3]),
    ]
}

/// `InvMixColumns` on one 4-byte state column.
pub(super) fn aes_inv_mix_column(a: [u8; 4]) -> [u8; 4] {
    [
        gf_mul(14, a[0]) ^ gf_mul(11, a[1]) ^ gf_mul(13, a[2]) ^ gf_mul(9, a[3]),
        gf_mul(9, a[0]) ^ gf_mul(14, a[1]) ^ gf_mul(11, a[2]) ^ gf_mul(13, a[3]),
        gf_mul(13, a[0]) ^ gf_mul(9, a[1]) ^ gf_mul(14, a[2]) ^ gf_mul(11, a[3]),
        gf_mul(11, a[0]) ^ gf_mul(13, a[1]) ^ gf_mul(9, a[2]) ^ gf_mul(14, a[3]),
    ]
}

// ---- Cryptographic Extension: SHA-1 / SHA-256 ----
//
// These implement the standard FIPS 180-4 round functions and message
// schedule recurrence; the ARM-specific part is how each instruction packs
// 4 (SHA-1) or 8 (SHA-256) 32-bit working variables into one or two 128-bit
// vector registers and how many rounds/schedule words it advances per call.
// That packing isn't published in an easily-citable form, so it was derived
// empirically: probe the real `SHA1C`/`SHA1P`/`SHA1M`/`SHA1SU0`/`SHA1SU1`/
// `SHA256H`/`SHA256H2`/`SHA256SU0`/`SHA256SU1` instructions on this host's
// Apple Silicon CPU (which implements FEAT_SHA1/FEAT_SHA256) with
// distinguishable (non-repeating-nibble) inputs, then solve for the linear/
// round-function structure that reproduces the outputs — see the test
// module, which re-checks this against a full SHA-1 and SHA-256 block
// compression compared to a `sha1sum`/`sha256sum`-equivalent digest.

/// Unpack a 128-bit vector into 4 little-endian 32-bit lanes (lane 0 = the
/// vector's least-significant 32 bits, matching this file's `LD1`/`ldst_vec`
/// convention elsewhere).
pub(super) fn u32_lanes(v: u128) -> [u32; 4] {
    [
        v as u32,
        (v >> 32) as u32,
        (v >> 64) as u32,
        (v >> 96) as u32,
    ]
}

/// Read 32-bit lane `i` (0..=3) of `v`.
pub(super) fn lane32(v: u128, i: u32) -> u32 {
    (v >> (i * 32)) as u32
}

/// Pack 4 32-bit lanes (lane 0 first) into a 128-bit vector.
pub(super) fn pack_u32_lanes(l: [u32; 4]) -> u128 {
    u128::from(l[0])
        | (u128::from(l[1]) << 32)
        | (u128::from(l[2]) << 64)
        | (u128::from(l[3]) << 96)
}

/// Which SHA-1 nonlinear round function `SHA1C`/`SHA1P`/`SHA1M` runs.
#[derive(Clone, Copy)]
pub(super) enum Sha1Op {
    /// `SHA1C`: `Ch(b,c,d)` — rounds 0..19.
    Choose,
    /// `SHA1P`: `Parity(b,c,d)` — rounds 20..39 and 60..79.
    Parity,
    /// `SHA1M`: `Maj(b,c,d)` — rounds 40..59.
    Majority,
}

pub(super) fn sha1_f(op: Sha1Op, b: u32, c: u32, d: u32) -> u32 {
    match op {
        Sha1Op::Choose => (b & c) ^ (!b & d),
        Sha1Op::Parity => b ^ c ^ d,
        Sha1Op::Majority => (b & c) ^ (b & d) ^ (c & d),
    }
}

/// `SHA1C`/`SHA1P`/`SHA1M`: four rounds of the SHA-1 compression function,
/// folding scalar `e` and vector `abcd` (lanes 0..3 = `a,b,c,d`) against the
/// four pre-added `W[t]+K[t]` words in `wk` (lane 0 consumed first). Returns
/// the updated `{a,b,c,d}` packed the same way `abcd` was.
#[allow(clippy::many_single_char_names)]
pub(super) fn sha1_quad_round(abcd: u128, mut e: u32, wk: u128, op: Sha1Op) -> u128 {
    let [mut a, mut b, mut c, mut d] = u32_lanes(abcd);
    for i in 0..4u32 {
        let w = lane32(wk, i);
        let t = a
            .rotate_left(5)
            .wrapping_add(sha1_f(op, b, c, d))
            .wrapping_add(e)
            .wrapping_add(w);
        e = d;
        d = c;
        c = b.rotate_left(30);
        b = a;
        a = t;
    }
    pack_u32_lanes([a, b, c, d])
}

/// `SHA1SU0`: the XOR half of the SHA-1 message-schedule recurrence
/// `W[t] = ROL(W[t-3] ^ W[t-8] ^ W[t-14] ^ W[t-16], 1)` — computes
/// `W[t-16..t-13] ^ W[t-14..t-11] ^ W[t-8..t-5]`, missing the `W[t-3]` term
/// that `SHA1SU1` adds before rotating. `vd` = `W[t-16..t-13]`, `vn` =
/// `W[t-12..t-9]`, `vm` = `W[t-8..t-5]`.
pub(super) fn sha1_su0(vd: u128, vn: u128, vm: u128) -> u128 {
    let d = u32_lanes(vd);
    let n = u32_lanes(vn);
    let m = u32_lanes(vm);
    pack_u32_lanes([
        d[0] ^ d[2] ^ m[0],
        d[1] ^ d[3] ^ m[1],
        d[2] ^ n[0] ^ m[2],
        d[3] ^ n[1] ^ m[3],
    ])
}

/// `SHA1SU1`: finishes the recurrence `SHA1SU0` started, folding in the
/// `W[t-3]` term and rotating. `vd` = `SHA1SU0`'s output, `vn` =
/// `W[t-4..t-1]`. Lane 3 needs `W[t]` for its `W[t-3]` term — that's lane 0
/// of this very call's result, computed first.
pub(super) fn sha1_su1(vd: u128, vn: u128) -> u128 {
    let d = u32_lanes(vd);
    let n = u32_lanes(vn);
    let w0 = (d[0] ^ n[1]).rotate_left(1);
    let w1 = (d[1] ^ n[2]).rotate_left(1);
    let w2 = (d[2] ^ n[3]).rotate_left(1);
    let w3 = (d[3] ^ w0).rotate_left(1);
    pack_u32_lanes([w0, w1, w2, w3])
}

pub(super) fn sha256_ch(e: u32, f: u32, g: u32) -> u32 {
    (e & f) ^ (!e & g)
}
pub(super) fn sha256_maj(a: u32, b: u32, c: u32) -> u32 {
    (a & b) ^ (a & c) ^ (b & c)
}
pub(super) fn sha256_bsig0(a: u32) -> u32 {
    a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22)
}
pub(super) fn sha256_bsig1(e: u32) -> u32 {
    e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25)
}
pub(super) fn sha256_ssig0(x: u32) -> u32 {
    x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
}
pub(super) fn sha256_ssig1(x: u32) -> u32 {
    x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
}

/// `SHA256H`/`SHA256H2`: four rounds of the SHA-256 compression function
/// over working variables `{a,b,c,d}` (`abcd`, lanes 0..3) and `{e,f,g,h}`
/// (`efgh`), consuming the four pre-added `W[t]+K[t]` words in `wk` (lane 0
/// first). `SHA256H` wants the updated `{a,b,c,d}`; `SHA256H2` is called
/// with the *original* (pre-round) `abcd`/`efgh` and wants the updated
/// `{e,f,g,h}` from that same round — hence `want_efgh` picks which half of
/// one shared computation to return, rather than this being two unrelated
/// functions.
#[allow(clippy::many_single_char_names)]
pub(super) fn sha256_hash(abcd: u128, efgh: u128, wk: u128, want_efgh: bool) -> u128 {
    let [mut a, mut b, mut c, mut d] = u32_lanes(abcd);
    let [mut e, mut f, mut g, mut h] = u32_lanes(efgh);
    for i in 0..4u32 {
        let w = lane32(wk, i);
        let t1 = h
            .wrapping_add(sha256_bsig1(e))
            .wrapping_add(sha256_ch(e, f, g))
            .wrapping_add(w);
        let t2 = sha256_bsig0(a).wrapping_add(sha256_maj(a, b, c));
        h = g;
        g = f;
        f = e;
        e = d.wrapping_add(t1);
        d = c;
        c = b;
        b = a;
        a = t1.wrapping_add(t2);
    }
    if want_efgh {
        pack_u32_lanes([e, f, g, h])
    } else {
        pack_u32_lanes([a, b, c, d])
    }
}

/// `SHA256SU0`: the `ssig0` half of the SHA-256 message-schedule recurrence
/// `W[t] = ssig1(W[t-2]) + W[t-7] + ssig0(W[t-15]) + W[t-16]` — computes
/// `W[t-16..t-13] + ssig0(W[t-15..t-12])`. `vd` = `W[t-16..t-13]`, `vn` =
/// `W[t-12..t-9]` (only lane 0, `W[t-12]`, is used — as `W[t-15]` for the
/// fourth output word).
pub(super) fn sha256_su0(vd: u128, vn: u128) -> u128 {
    let d = u32_lanes(vd);
    let n = u32_lanes(vn);
    pack_u32_lanes([
        d[0].wrapping_add(sha256_ssig0(d[1])),
        d[1].wrapping_add(sha256_ssig0(d[2])),
        d[2].wrapping_add(sha256_ssig0(d[3])),
        d[3].wrapping_add(sha256_ssig0(n[0])),
    ])
}

/// `SHA256SU1`: finishes the recurrence `SHA256SU0` started, adding the
/// `ssig1(W[t-2])` and `W[t-7]` terms. `vd` = `SHA256SU0`'s output, `vn` =
/// `W[t-8..t-5]`, `vm` = `W[t-4..t-1]`. Words 2 and 3 need `W[t-2]` for a `t`
/// only one or two words in the future — that's this call's own lane 0 or 1,
/// computed first, since those source words don't exist as an earlier
/// instruction's output yet.
pub(super) fn sha256_su1(vd: u128, vn: u128, vm: u128) -> u128 {
    let d = u32_lanes(vd);
    let n = u32_lanes(vn);
    let m = u32_lanes(vm);
    let w0 = d[0].wrapping_add(sha256_ssig1(m[2])).wrapping_add(n[1]);
    let w1 = d[1].wrapping_add(sha256_ssig1(m[3])).wrapping_add(n[2]);
    let w2 = d[2].wrapping_add(sha256_ssig1(w0)).wrapping_add(n[3]);
    let w3 = d[3].wrapping_add(sha256_ssig1(w1)).wrapping_add(m[0]);
    pack_u32_lanes([w0, w1, w2, w3])
}
