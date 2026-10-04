//! A64 integer data processing: the "Data Processing -- Immediate" (`op0 =
//! 100x`) and "Data Processing -- Register" (`op0 = x101`) encoding groups.
//!
//! Each group is decoded by the sub-opcode fields the ARM ARM's encoding index
//! uses, and every unallocated combination (including the encodings later
//! architecture versions gave to extensions this CPU doesn't advertise —
//! MTE `ADDG`/`SUBP`, CSSC `SMAX`/`ABS`/`CTZ`, FlagM `RMIF`/`SETF`, PAuth
//! `PAC*`) returns [`Step::Illegal`], exactly as a v8.0 core would.

use super::{Aarch64Interp, Flags, Step, reg_field};

impl Aarch64Interp {
    /// Data Processing -- Immediate: bits 25:23 select the class.
    pub(super) fn exec_dp_imm(&mut self, instr: u32) -> Step {
        let rd = reg_field(instr, 0);
        let rn = reg_field(instr, 5);
        let sf = instr >> 31 == 1;
        match (instr >> 23) & 7 {
            // ADR / ADRP
            0b000 | 0b001 => {
                let immlo = u64::from((instr >> 29) & 3);
                let immhi = u64::from((instr >> 5) & 0x7ffff);
                let imm = sign_extend((immhi << 2) | immlo, 21);
                let r = if sf {
                    (self.pc & !0xfff).wrapping_add((imm << 12) as u64)
                } else {
                    self.pc.wrapping_add(imm as u64)
                };
                self.write_x(rd, r);
            }
            // ADD/ADDS/SUB/SUBS (immediate). Rn is SP-form; Rd is SP-form
            // unless flags are set.
            0b010 => {
                let sub = (instr >> 30) & 1 == 1;
                let s = (instr >> 29) & 1 == 1;
                let imm12 = u64::from((instr >> 10) & 0xfff);
                let imm = if (instr >> 22) & 1 == 1 {
                    imm12 << 12
                } else {
                    imm12
                };
                let a = self.read_sp(rn);
                if s {
                    let r = self.addsub_flags(a, imm, sub, sf);
                    self.write_x(rd, r);
                } else {
                    let r = if sub {
                        a.wrapping_sub(imm)
                    } else {
                        a.wrapping_add(imm)
                    };
                    self.write_sp(rd, mask_sf(r, sf));
                }
            }
            // AND/ORR/EOR/ANDS (immediate).
            0b100 => {
                let n = (instr >> 22) & 1;
                if !sf && n == 1 {
                    return Step::Illegal;
                }
                let immr = (instr >> 16) & 0x3f;
                let imms = (instr >> 10) & 0x3f;
                let width = if sf { 64 } else { 32 };
                let Some((imm, _)) = decode_bit_masks(n, imms, immr, width, true) else {
                    return Step::Illegal;
                };
                let a = self.read_x(rn);
                let opc = (instr >> 29) & 3;
                let r = mask_sf(
                    match opc {
                        0b00 | 0b11 => a & imm,
                        0b01 => a | imm,
                        _ => a ^ imm,
                    },
                    sf,
                );
                if opc == 0b11 {
                    self.set_nz(r, sf);
                    self.write_x(rd, r);
                } else {
                    self.write_sp(rd, r);
                }
            }
            // MOVN/MOVZ/MOVK.
            0b101 => {
                let opc = (instr >> 29) & 3;
                let hw = (instr >> 21) & 3;
                if opc == 0b01 || (!sf && hw > 1) {
                    return Step::Illegal;
                }
                let shift = hw * 16;
                let val = u64::from((instr >> 5) & 0xffff) << shift;
                let r = match opc {
                    0b00 => !val,
                    0b10 => val,
                    _ => (self.read_x(rd) & !(0xffff_u64 << shift)) | val,
                };
                self.write_x(rd, mask_sf(r, sf));
            }
            // SBFM/BFM/UBFM.
            0b110 => {
                let opc = (instr >> 29) & 3;
                let n = (instr >> 22) & 1;
                let immr = (instr >> 16) & 0x3f;
                let imms = (instr >> 10) & 0x3f;
                if opc == 0b11 || (n == 1) != sf || (!sf && (immr | imms) & 0x20 != 0) {
                    return Step::Illegal;
                }
                let width = if sf { 64 } else { 32 };
                let Some((wmask, tmask)) = decode_bit_masks(n, imms, immr, width, false) else {
                    return Step::Illegal;
                };
                let src = self.read_x(rn);
                let bot = ror_val(src, immr, width) & wmask;
                let r = match opc {
                    0b00 => {
                        // SBFM: replicate bit `imms` of the source above the field.
                        let top = if (src >> imms) & 1 == 1 {
                            ones(width)
                        } else {
                            0
                        };
                        (top & !tmask) | (bot & tmask)
                    }
                    0b01 => {
                        let dst = self.read_x(rd);
                        (dst & !tmask) | (((dst & !wmask) | bot) & tmask)
                    }
                    _ => bot & tmask,
                };
                self.write_x(rd, mask_sf(r, sf));
            }
            // ADDG/SUBG (MTE) and the CSSC min/max immediates.
            0b011 => return Step::Illegal,
            // EXTR (ROR immediate alias).
            _ => {
                let n = (instr >> 22) & 1;
                let op21 = (instr >> 29) & 3;
                let o0 = (instr >> 21) & 1;
                let imms = (instr >> 10) & 0x3f;
                if op21 != 0 || o0 != 0 || (n == 1) != sf || (!sf && imms >= 32) {
                    return Step::Illegal;
                }
                let rm = reg_field(instr, 16);
                let (hi, lo) = (self.read_x(rn), self.read_x(rm));
                let r = if sf {
                    if imms == 0 {
                        lo
                    } else {
                        (lo >> imms) | (hi << (64 - imms))
                    }
                } else {
                    let v = ((hi & 0xffff_ffff) << 32) | (lo & 0xffff_ffff);
                    (v >> imms) & 0xffff_ffff
                };
                self.write_x(rd, r);
            }
        }
        Step::Next
    }

    /// Data Processing -- Register.
    pub(super) fn exec_dp_reg(&mut self, instr: u32) -> Step {
        let sf = instr >> 31 == 1;
        let rd = reg_field(instr, 0);
        let rn = reg_field(instr, 5);
        let rm = reg_field(instr, 16);
        if (instr >> 28) & 1 == 0 {
            if (instr >> 24) & 1 == 0 {
                // Logical (shifted register): AND/BIC/ORR/ORN/EOR/EON/ANDS/BICS.
                let imm6 = (instr >> 10) & 0x3f;
                if !sf && imm6 >= 32 {
                    return Step::Illegal;
                }
                let mut b = shift_reg(self.read_x(rm), (instr >> 22) & 3, imm6, sf);
                if (instr >> 21) & 1 == 1 {
                    b = mask_sf(!b, sf);
                }
                let a = self.read_x(rn);
                let opc = (instr >> 29) & 3;
                let r = mask_sf(
                    match opc {
                        0b00 | 0b11 => a & b,
                        0b01 => a | b,
                        _ => a ^ b,
                    },
                    sf,
                );
                if opc == 0b11 {
                    self.set_nz(r, sf);
                }
                self.write_x(rd, r);
                return Step::Next;
            }
            let sub = (instr >> 30) & 1 == 1;
            let s = (instr >> 29) & 1 == 1;
            if (instr >> 21) & 1 == 0 {
                // ADD/SUB (shifted register). Rn/Rd are ZR-form.
                let shift = (instr >> 22) & 3;
                let imm6 = (instr >> 10) & 0x3f;
                if shift == 3 || (!sf && imm6 >= 32) {
                    return Step::Illegal;
                }
                let a = self.read_x(rn);
                let b = shift_reg(self.read_x(rm), shift, imm6, sf);
                let r = if s {
                    self.addsub_flags(a, b, sub, sf)
                } else if sub {
                    mask_sf(a.wrapping_sub(b), sf)
                } else {
                    mask_sf(a.wrapping_add(b), sf)
                };
                self.write_x(rd, r);
            } else {
                // ADD/SUB (extended register). Rn is SP-form; Rd is SP-form
                // unless flags are set.
                let imm3 = (instr >> 10) & 7;
                if (instr >> 22) & 3 != 0 || imm3 > 4 {
                    return Step::Illegal;
                }
                let a = self.read_sp(rn);
                let b = extend_reg(self.read_x(rm), (instr >> 13) & 7, imm3);
                if s {
                    let r = self.addsub_flags(a, b, sub, sf);
                    self.write_x(rd, r);
                } else {
                    let r = if sub {
                        a.wrapping_sub(b)
                    } else {
                        a.wrapping_add(b)
                    };
                    self.write_sp(rd, mask_sf(r, sf));
                }
            }
            return Step::Next;
        }
        match (instr >> 21) & 0xf {
            // ADC/ADCS/SBC/SBCS; RMIF and SETF8/SETF16 (FEAT_FlagM).
            0b0000 => {
                if (instr >> 10) & 0x3f != 0 {
                    return self.exec_flagm(instr);
                }
                let sub = (instr >> 30) & 1 == 1;
                let a = self.read_x(rn);
                let b = if sub {
                    !self.read_x(rm)
                } else {
                    self.read_x(rm)
                };
                let r = if (instr >> 29) & 1 == 1 {
                    self.add_with_carry_flags(a, b, self.flags.c, sf)
                } else {
                    mask_sf(a.wrapping_add(b).wrapping_add(u64::from(self.flags.c)), sf)
                };
                self.write_x(rd, r);
            }
            // CCMN/CCMP (register or immediate).
            0b0010 => {
                if (instr >> 29) & 1 == 0 || (instr >> 10) & 1 != 0 || (instr >> 4) & 1 != 0 {
                    return Step::Illegal;
                }
                let operand = if (instr >> 11) & 1 == 1 {
                    u64::from((instr >> 16) & 0x1f)
                } else {
                    self.read_x(rm)
                };
                if self.cond_holds((instr >> 12) & 0xf) {
                    let sub = (instr >> 30) & 1 == 1;
                    self.addsub_flags(self.read_x(rn), operand, sub, sf);
                } else {
                    self.flags = Flags::from_nzcv(instr & 0xf);
                }
            }
            // CSEL/CSINC/CSINV/CSNEG.
            0b0100 => {
                if (instr >> 29) & 1 != 0 || (instr >> 11) & 1 != 0 {
                    return Step::Illegal;
                }
                let r = if self.cond_holds((instr >> 12) & 0xf) {
                    self.read_x(rn)
                } else {
                    let m = self.read_x(rm);
                    match ((instr >> 30) & 1, (instr >> 10) & 1) {
                        (0, 0) => m,
                        (0, _) => m.wrapping_add(1),
                        (_, 0) => !m,
                        _ => m.wrapping_neg(),
                    }
                };
                self.write_x(rd, mask_sf(r, sf));
            }
            0b0110 => {
                if (instr >> 29) & 1 != 0 {
                    return Step::Illegal; // SUBPS and friends (MTE)
                }
                let opcode = (instr >> 10) & 0x3f;
                if (instr >> 30) & 1 == 1 {
                    // Data-processing (1 source).
                    if rm != 0 {
                        return Step::Illegal; // opcode2: PAuth
                    }
                    let width = if sf { 64 } else { 32 };
                    let x = mask_sf(self.read_x(rn), sf);
                    let r = match opcode {
                        0b000000 => rbit(x, width),
                        0b000001 => rev16(x, width),
                        0b000010 if sf => rev32(x),
                        0b000010 => u64::from((x as u32).swap_bytes()),
                        0b000011 if sf => x.swap_bytes(),
                        0b000100 => u64::from(x.leading_zeros() - (64 - width)),
                        0b000101 => u64::from(cls(x, width)),
                        _ => return Step::Illegal,
                    };
                    self.write_x(rd, r);
                } else {
                    // Data-processing (2 source).
                    let a = self.read_x(rn);
                    let b = self.read_x(rm);
                    let width: u64 = if sf { 64 } else { 32 };
                    let r = match opcode {
                        0b000010 => udiv(a, b, sf),
                        0b000011 => sdiv(a, b, sf),
                        0b001000..=0b001011 => shift_reg(a, opcode & 3, (b % width) as u32, sf),
                        0b010000..=0b010111 => {
                            // CRC32{B,H,W,X} / CRC32C{B,H,W,X}: only the X
                            // form takes sf=1.
                            let sz = opcode & 3;
                            if (sz == 3) != sf {
                                return Step::Illegal;
                            }
                            let poly = if opcode & 4 != 0 {
                                0x82F6_3B78
                            } else {
                                0xEDB8_8320
                            };
                            u64::from(super::crypto::crc32(a as u32, b, 1 << sz, poly))
                        }
                        _ => return Step::Illegal,
                    };
                    self.write_x(rd, mask_sf(r, sf));
                }
            }
            // Data-processing (3 source).
            0b1000..=0b1111 => {
                if (instr >> 29) & 3 != 0 {
                    return Step::Illegal;
                }
                let o0 = (instr >> 15) & 1 == 1;
                let ra = reg_field(instr, 10);
                let (n, m, a) = (self.read_x(rn), self.read_x(rm), self.read_x(ra));
                let acc = |prod: u64| {
                    if o0 {
                        a.wrapping_sub(prod)
                    } else {
                        a.wrapping_add(prod)
                    }
                };
                let r = match ((instr >> 21) & 7, sf) {
                    (0b000, _) => mask_sf(acc(n.wrapping_mul(m)), sf),
                    (0b001, true) => {
                        acc(i64::from(n as i32).wrapping_mul(i64::from(m as i32)) as u64)
                    }
                    (0b101, true) => acc(u64::from(n as u32) * u64::from(m as u32)),
                    (0b010, true) if !o0 => {
                        ((i128::from(n as i64) * i128::from(m as i64)) >> 64) as u64
                    }
                    (0b110, true) if !o0 => ((u128::from(n) * u128::from(m)) >> 64) as u64,
                    _ => return Step::Illegal,
                };
                self.write_x(rd, r);
            }
            _ => return Step::Illegal,
        }
        Step::Next
    }

    /// `RMIF` (rotate a register right, insert selected bits into NZCV) and
    /// `SETF8`/`SETF16` (flags from an 8/16-bit value), FEAT_FlagM.
    fn exec_flagm(&mut self, instr: u32) -> Step {
        let rn = reg_field(instr, 5);
        if instr & 0xFFE0_7C10 == 0xBA00_0400 {
            // RMIF Xn, #shift, #mask
            let v = self.read_x(rn).rotate_right((instr >> 15) & 0x3f);
            let mask = instr & 0xf;
            let cur = self.flags.nzcv();
            let new = (cur & !mask) | (v as u32 & mask);
            self.flags = Flags::from_nzcv(new);
            return Step::Next;
        }
        if instr & 0xFFFF_BC1F == 0x3A00_080D {
            // SETF8 / SETF16 Wn
            let bits = if (instr >> 14) & 1 == 1 { 16 } else { 8 };
            let v = self.read_x(rn);
            let top = (v >> (bits - 1)) & 1 == 1;
            self.flags.n = top;
            self.flags.z = v & ones(bits) == 0;
            self.flags.v = ((v >> bits) & 1 == 1) != top;
            return Step::Next;
        }
        Step::Illegal
    }

    /// Set N and Z from a logical result, clearing C and V (`ANDS`/`BICS`).
    fn set_nz(&mut self, r: u64, sf: bool) {
        let top = if sf { 63 } else { 31 };
        self.flags = Flags {
            n: (r >> top) & 1 == 1,
            z: r == 0,
            c: false,
            v: false,
        };
    }

    /// `AddWithCarry(a, b, carry)` setting NZCV; returns the (width-masked)
    /// sum. `SUB`/`SBC` pass `!b` (and carry 1 for `SUB`).
    pub(super) fn add_with_carry_flags(&mut self, a: u64, b: u64, carry: bool, sf: bool) -> u64 {
        let c = u128::from(carry);
        if sf {
            let sum = u128::from(a) + u128::from(b) + c;
            let r = sum as u64;
            self.flags = Flags {
                n: (r >> 63) & 1 == 1,
                z: r == 0,
                c: sum >> 64 != 0,
                v: ((a ^ r) & (b ^ r)) >> 63 == 1,
            };
            r
        } else {
            let (a, b) = (a as u32, b as u32);
            let sum = u64::from(a) + u64::from(b) + c as u64;
            let r = sum as u32;
            self.flags = Flags {
                n: (r >> 31) & 1 == 1,
                z: r == 0,
                c: sum >> 32 != 0,
                v: ((a ^ r) & (b ^ r)) >> 31 == 1,
            };
            u64::from(r)
        }
    }

    /// Compute `a - b` (if `sub`) or `a + b` at width `sf`, setting NZCV.
    pub(super) fn addsub_flags(&mut self, a: u64, b: u64, sub: bool, sf: bool) -> u64 {
        if sub {
            self.add_with_carry_flags(a, !b, true, sf)
        } else {
            self.add_with_carry_flags(a, b, false, sf)
        }
    }
}

/// Mask to 32 bits for a 32-bit (`sf == false`) operation.
#[inline]
pub(super) const fn mask_sf(v: u64, sf: bool) -> u64 {
    if sf { v } else { v & 0xffff_ffff }
}

/// Sign-extend the low `bits` bits of `v`.
#[inline]
pub(super) const fn sign_extend(v: u64, bits: u32) -> i64 {
    let shift = 64 - bits;
    ((v << shift) as i64) >> shift
}

/// `n` low bits set.
#[inline]
pub(super) const fn ones(n: u32) -> u64 {
    if n >= 64 { u64::MAX } else { (1u64 << n) - 1 }
}

/// Apply an A64 register shift (`LSL`/`LSR`/`ASR`/`ROR`) by `amount` at the
/// operation width.
pub(super) fn shift_reg(v: u64, shift_type: u32, amount: u32, sf: bool) -> u64 {
    if sf {
        match shift_type {
            0 => v << amount,
            1 => v >> amount,
            2 => ((v as i64) >> amount) as u64,
            _ => v.rotate_right(amount),
        }
    } else {
        let v = v as u32;
        u64::from(match shift_type {
            0 => v << amount,
            1 => v >> amount,
            2 => ((v as i32) >> amount) as u32,
            _ => v.rotate_right(amount),
        })
    }
}

/// Reverse the low `width` bits of `v`.
fn rbit(v: u64, width: u32) -> u64 {
    if width == 64 {
        v.reverse_bits()
    } else {
        u64::from((v as u32).reverse_bits())
    }
}

/// Reverse the bytes within each 16-bit halfword of the low `width` bits.
fn rev16(v: u64, width: u32) -> u64 {
    let r = ((v & 0x00ff_00ff_00ff_00ff) << 8) | ((v >> 8) & 0x00ff_00ff_00ff_00ff);
    if width == 64 { r } else { r & 0xffff_ffff }
}

/// Reverse the bytes within each 32-bit word (64-bit `REV32`).
fn rev32(v: u64) -> u64 {
    u64::from((v as u32).swap_bytes()) | (u64::from(((v >> 32) as u32).swap_bytes()) << 32)
}

/// `CLS`: leading sign bits (not counting the sign bit itself) over `width`.
fn cls(v: u64, width: u32) -> u32 {
    let x = if width == 64 {
        v
    } else {
        v << 32 | 0xffff_ffff
    };
    // XOR with the value shifted by one: the leading zeros of that, minus the
    // sign bit, count the repeated sign bits.
    let y = x ^ (x << 1);
    let y = if width == 64 { y } else { y | 1 };
    (y.leading_zeros()).min(width - 1)
}

/// Unsigned divide (division by zero yields 0).
fn udiv(a: u64, b: u64, sf: bool) -> u64 {
    if sf {
        a.checked_div(b).unwrap_or(0)
    } else {
        (a as u32).checked_div(b as u32).map_or(0, u64::from)
    }
}

/// Signed divide (division by zero yields 0; `INT_MIN / -1` wraps).
fn sdiv(a: u64, b: u64, sf: bool) -> u64 {
    if sf {
        let (a, b) = (a as i64, b as i64);
        if b == 0 { 0 } else { a.wrapping_div(b) as u64 }
    } else {
        let (a, b) = (a as i32, b as i32);
        if b == 0 {
            0
        } else {
            u64::from(a.wrapping_div(b) as u32)
        }
    }
}

/// Extend a register value per the `option` field (`UXTB/H/W/X`,
/// `SXTB/H/W/X`), then shift left by `shift` (0..=4).
pub(super) fn extend_reg(val: u64, option: u32, shift: u32) -> u64 {
    let extended = match option {
        0b000 => val & 0xff,
        0b001 => val & 0xffff,
        0b010 => val & 0xffff_ffff,
        0b100 => sign_extend(val, 8) as u64,
        0b101 => sign_extend(val, 16) as u64,
        0b110 => sign_extend(val, 32) as u64,
        _ => val,
    };
    extended << shift
}

/// Rotate the low `size` bits of `v` right by `r`.
fn ror_val(v: u64, r: u32, size: u32) -> u64 {
    let v = v & ones(size);
    let r = r % size;
    if r == 0 {
        v
    } else {
        ((v >> r) | (v << (size - r))) & ones(size)
    }
}

/// Replicate an `esize`-bit `pattern` across `width` bits.
fn replicate(pattern: u64, esize: u32, width: u32) -> u64 {
    let pat = pattern & ones(esize);
    let mut result = 0u64;
    let mut i = 0u32;
    while i < width {
        result |= pat << i;
        i += esize;
    }
    result
}

/// ARM `DecodeBitMasks`: `(wmask, tmask)` for the logical-immediate
/// (`immediate == true`) and bitfield instructions; `None` for a reserved
/// encoding.
pub(super) fn decode_bit_masks(
    n: u32,
    imms: u32,
    immr: u32,
    width: u32,
    immediate: bool,
) -> Option<(u64, u64)> {
    let x = (n << 6) | (!imms & 0x3f);
    if x == 0 {
        return None;
    }
    let len = x.ilog2();
    if len < 1 {
        return None;
    }
    let levels = (1u32 << len) - 1;
    if immediate && imms & levels == levels {
        return None;
    }
    let s = imms & levels;
    let r = immr & levels;
    let diff = s.wrapping_sub(r) & levels;
    let esize = 1u32 << len;
    let wmask = replicate(ror_val(ones(s + 1), r, esize), esize, width);
    let tmask = replicate(ones(diff + 1), esize, width);
    Some((wmask, tmask))
}
