//! The x87 FPU (`D8`-`DF` escape opcodes, plus `FWAIT`): an 8-register stack
//! of true 80-bit extended-precision values with tags, `TOP`, condition codes,
//! the control word's precision/rounding control and exception masks, and the
//! status word's sticky exception flags.
//!
//! Semantics follow the Intel SDM (Vol. 1 ch. 8, Vol. 2 instruction pages):
//!
//! * **Stack faults** — reading an empty register raises `IE`+`SF` with
//!   `C1 = 0`; pushing onto a full stack raises `IE`+`SF` with `C1 = 1`; the
//!   masked response substitutes the QNaN "real indefinite".
//! * **Operands** — denormals (and pseudo-denormals) raise `DE`; the
//!   encodings the x87 no longer supports (unnormals, pseudo-NaN/-infinity)
//!   raise `IE`; NaNs propagate per the x87 rule (SNaN quieted; of two NaNs,
//!   the QNaN, else the larger significand).
//! * **Results** — rounded once under `RC`, and for the basic arithmetic
//!   (`FADD`..`FDIVR`, `FSQRT`) at the `PC` significand width; `C1` reports a
//!   round-up. Masked exceptions produce the IEEE default results. An
//!   *unmasked* `IE`/`DE`/`ZE` leaves the destination and stack untouched; any
//!   unmasked exception sets `ES`/`B`, and the next waiting x87 instruction
//!   (or `FWAIT`) raises `#MF`.
//! * The last-instruction pointer/opcode/data pointer (`FIP`/`FOP`/`FDP`) read
//!   back as zero, as on CPUs that don't report them.

#![allow(
    clippy::match_same_arms,
    clippy::single_match_else,
    clippy::bool_to_int_with_if
)]

use super::x87math;
use super::{
    F80, Flags, GuestMemory, ModRm, Pfx, RAX, RmKind, Step, Trap, X86Interp, fetch, rd_fault,
    wr_fault,
};
use crate::vcpu::softfloat::{
    self as sf, Class, DENORMAL, DIVZERO, FMT32, FMT64, FMT80, Fp, INVALID, Round, Rounded,
    UNDERFLOW,
};

/// The stack-fault flag (`SF`, status word bit 6), raised with `INVALID`.
const STACK_FAULT: u32 = 0x40;

/// The x87 arithmetic group selected by the ModRM `reg` field of `D8`/`DA`/
/// `DC`/`DE` (and the `D8` register form): `/0..7`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FpuOp {
    Add,
    Mul,
    Com,
    Comp,
    Sub,
    SubR,
    Div,
    DivR,
}

impl FpuOp {
    fn from_ext(e: usize) -> Self {
        match e & 7 {
            0 => Self::Add,
            1 => Self::Mul,
            2 => Self::Com,
            3 => Self::Comp,
            4 => Self::Sub,
            5 => Self::SubR,
            6 => Self::Div,
            _ => Self::DivR,
        }
    }

    /// `DC`/`DE`'s register forms write `ST(i)` and read `ST(0)` as the other
    /// operand, so `SUB`/`SUBR` (and `DIV`/`DIVR`) trade places — `DC E0+i`
    /// is `FSUBR ST(i), ST(0)`. `/2`,`/3` are the `FCOM`/`FCOMP` aliases.
    fn from_ext_reversed(e: usize) -> Self {
        match e & 7 {
            4 => Self::SubR,
            5 => Self::Sub,
            6 => Self::DivR,
            7 => Self::Div,
            x => Self::from_ext(x),
        }
    }
}

/// A memory operand format of the x87 load/store/arithmetic instructions.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MemFmt {
    F32,
    F64,
    F80,
    I16,
    I32,
    I64,
}

/// Is this 80-bit value a denormal or pseudo-denormal (exponent field 0,
/// nonzero significand)?
fn f80_denormal(v: F80) -> bool {
    v.exp_field() == 0 && v.mant() != 0
}

/// The x87 NaN propagation rule for two operands (either may be a NaN): an
/// SNaN is quieted with `INVALID`; between two NaNs the QNaN wins over an
/// SNaN, else the larger significand (ties: the positive one).
pub(super) fn x87_nan2(a: &Fp, b: &Fp) -> Option<(Fp, u32)> {
    if !a.is_nan() && !b.is_nan() {
        return None;
    }
    let flags = if a.is_snan() || b.is_snan() {
        INVALID
    } else {
        0
    };
    let pick = if a.is_nan() && b.is_nan() {
        if a.is_snan() != b.is_snan() {
            if a.is_snan() { b } else { a }
        } else if a.sig != b.sig {
            if a.sig > b.sig { a } else { b }
        } else if a.sign {
            b
        } else {
            a
        }
    } else if a.is_nan() {
        a
    } else {
        b
    };
    Some((pick.quieted(), flags))
}

/// Unpack an x87 register operand for arithmetic: the value plus its operand
/// exceptions (`IE` for an unsupported encoding — in which case the value is
/// replaced by the indefinite — and `DE` for a denormal).
fn operand(v: F80) -> (Fp, u32) {
    if sf::f80_unsupported(v.0) {
        (sf::INDEFINITE, INVALID)
    } else {
        (v.unpack(), if f80_denormal(v) { DENORMAL } else { 0 })
    }
}

impl X86Interp {
    // ---- register stack ------------------------------------------------------

    pub(super) fn st_idx(&self, i: u8) -> usize {
        usize::from(self.fpu_top.wrapping_add(i) & 7)
    }

    /// `ST(i)`'s value (regardless of its tag).
    pub(super) fn st_get(&self, i: u8) -> F80 {
        self.st[self.st_idx(i)]
    }

    /// Write `ST(i)` and mark it valid.
    pub(super) fn st_set(&mut self, i: u8, v: F80) {
        let idx = self.st_idx(i);
        self.st[idx] = v;
        self.fpu_tag |= 1 << idx;
    }

    pub(super) fn st_empty(&self, i: u8) -> bool {
        self.fpu_tag & (1 << self.st_idx(i)) == 0
    }

    /// Pop: mark `ST(0)` empty and increment `TOP`.
    pub(super) fn fpu_pop(&mut self) {
        let idx = self.st_idx(0);
        self.fpu_tag &= !(1 << idx);
        self.fpu_top = (self.fpu_top + 1) & 7;
    }

    /// Push `v`. A full stack (the new `ST(0)` slot already valid) is a stack
    /// overflow: `IE`+`SF`, `C1 = 1`, and — masked — the indefinite is pushed
    /// instead. Returns `false` if an unmasked fault aborted the push.
    pub(super) fn fpu_push(&mut self, v: F80) -> bool {
        let new_top = self.fpu_top.wrapping_sub(1) & 7;
        if self.fpu_tag & (1 << new_top) != 0 {
            self.fpu_c1 = true;
            if !self.x87_flags(INVALID | STACK_FAULT) {
                return false;
            }
            self.fpu_top = new_top;
            self.st_set(0, F80::INDEFINITE);
            return true;
        }
        self.fpu_top = new_top;
        self.st_set(0, v);
        true
    }

    /// Read `ST(i)` as a source operand. An empty register is a stack
    /// underflow (`IE`+`SF`, `C1 = 0`); `None` means an unmasked fault aborted
    /// the instruction, otherwise the masked response reads the indefinite.
    pub(super) fn st_src(&mut self, i: u8) -> Option<F80> {
        if self.st_empty(i) {
            self.fpu_c1 = false;
            if !self.x87_flags(INVALID | STACK_FAULT) {
                return None;
            }
            return Some(F80::INDEFINITE);
        }
        Some(self.st_get(i))
    }

    /// Accumulate exception `flags` (softfloat layout, plus [`STACK_FAULT`])
    /// into the status word. Returns `false` when an *unmasked* invalid,
    /// denormal or zero-divide exception means the instruction must not
    /// deliver its result.
    fn x87_flags(&mut self, flags: u32) -> bool {
        let f = (flags & 0x7f) as u16;
        self.fpu_flags |= f;
        f & !self.fpu_cw & 0x07 == 0
    }

    /// A pending unmasked exception (`ES`): the next waiting x87 instruction
    /// raises `#MF`.
    pub(super) fn fpu_pending(&self) -> bool {
        self.fpu_flags & !self.fpu_cw & 0x3f != 0
    }

    /// The active x87 rounding mode (`CW.RC`).
    pub(super) fn fpu_round(&self) -> Round {
        Round::from_x86(u32::from(self.fpu_cw >> 10))
    }

    /// The status word: exception flags + `SF`, `ES` (bit 7) and `B` (bit
    /// 15) when an unmasked flag is pending, `C0..C3`, `TOP`.
    pub(super) fn fpu_sw(&self) -> u16 {
        let mut sw = self.fpu_flags & 0x7f;
        if self.fpu_pending() {
            sw |= 0x8080;
        }
        sw |= u16::from(self.fpu_top & 7) << 11;
        sw |= u16::from(self.fpu_c0) << 8;
        sw |= u16::from(self.fpu_c1) << 9;
        sw |= u16::from(self.fpu_c2) << 10;
        sw |= u16::from(self.fpu_c3) << 14;
        sw
    }

    /// Load the status word (`FRSTOR`/`FLDENV`/`FXRSTOR`): `TOP`, `C0..C3` and
    /// the exception flags (`IE..PE`, `SF`); `ES`/`B` are derived.
    pub(super) fn set_fpu_sw(&mut self, sw: u16) {
        self.fpu_flags = sw & 0x7f;
        self.fpu_top = ((sw >> 11) & 7) as u8;
        self.fpu_c0 = sw & (1 << 8) != 0;
        self.fpu_c1 = sw & (1 << 9) != 0;
        self.fpu_c2 = sw & (1 << 10) != 0;
        self.fpu_c3 = sw & (1 << 14) != 0;
    }

    /// Load the control word; bit 6 reads back set and bits 13-15 clear.
    pub(super) fn set_fpu_cw(&mut self, cw: u16) {
        self.fpu_cw = (cw & 0x1f3f) | 0x40;
    }

    /// `FNINIT`: the power-on FPU state (control word `037F`, empty stack).
    pub(super) fn fpu_init(&mut self) {
        self.fpu_top = 0;
        self.fpu_tag = 0;
        self.fpu_c0 = false;
        self.fpu_c1 = false;
        self.fpu_c2 = false;
        self.fpu_c3 = false;
        self.fpu_cw = 0x037F;
        self.fpu_flags = 0;
        self.fpu_fop = 0;
        self.fpu_fip = 0;
        self.fpu_fdp = 0;
    }

    /// The full 16-bit tag word (`FNSTENV`/`FNSAVE`): per *physical* register,
    /// `11` empty, `01` zero, `10` special (NaN, ∞, denormal, unsupported),
    /// `00` valid.
    pub(super) fn fpu_full_tag(&self) -> u16 {
        let mut tw = 0u16;
        for i in 0..8 {
            let t = if self.fpu_tag & (1 << i) == 0 {
                3
            } else {
                let v = self.st[i];
                let e = v.exp_field();
                if e == 0x7fff || sf::f80_unsupported(v.0) || (e == 0 && v.mant() != 0) {
                    2
                } else if e == 0 {
                    1
                } else {
                    0
                }
            };
            tw |= t << (2 * i);
        }
        tw
    }

    /// Set the tags from a full tag word: `11` → empty, anything else valid
    /// (the class is re-derived from the register contents when stored).
    fn set_full_tag(&mut self, tw: u16) {
        self.fpu_tag = 0;
        for i in 0..8 {
            if (tw >> (2 * i)) & 3 != 3 {
                self.fpu_tag |= 1 << i;
            }
        }
    }

    /// The format arithmetic results round to under precision control.
    fn fpu_fmt(&self) -> sf::Fmt {
        sf::x87_fmt(self.fpu_cw >> 8)
    }

    /// Round an exact/approximate result into an `F80` at full precision
    /// under `RC`, setting `C1` from the round-up and collecting the flags.
    fn fpu_round_result(&mut self, r: Rounded) -> (F80, u32) {
        self.fpu_c1 = r.up;
        (F80::pack(&r.v), r.flags)
    }

    // ---- arithmetic core -------------------------------------------------------

    /// `a OP b` for `FADD`..`FDIVR` (`op` must be arithmetic; the reversed
    /// variants are applied by the caller swapping operands). Returns the
    /// result and the exception flags; sets `C1`.
    fn fpu_binop(&mut self, op: FpuOp, a: F80, b: F80) -> (F80, u32) {
        let (a, b) = match op {
            FpuOp::SubR | FpuOp::DivR => (b, a),
            _ => (a, b),
        };
        let (ua, fa) = operand(a);
        let (ub, fb) = operand(b);
        self.fpu_c1 = false;
        if (fa | fb) & INVALID != 0 {
            return (F80::INDEFINITE, INVALID);
        }
        if let Some((n, f)) = x87_nan2(&ua, &ub) {
            return (F80::pack(&n), f);
        }
        let dflags = fa | fb;
        let fmt = self.fpu_fmt();
        let mode = self.fpu_round();
        let r = match op {
            FpuOp::Add => sf::add(ua, ub, fmt, mode),
            FpuOp::Sub | FpuOp::SubR => sf::add(ua, ub.neg(), fmt, mode),
            FpuOp::Mul => sf::mul(ua, ub, fmt, mode),
            _ => sf::div(ua, ub, fmt, mode),
        };
        if r.flags & INVALID != 0 {
            return (F80::INDEFINITE, INVALID | dflags);
        }
        let (v, f) = self.fpu_round_result(r);
        (v, f | dflags)
    }

    /// Apply an arithmetic op with destination `ST(dst)` and the other operand
    /// `src`, optionally popping. Compares (`Com`/`Comp`) are handled here too.
    fn fpu_arith(&mut self, op: FpuOp, dst: u8, src: F80, pop: bool) {
        let Some(d) = self.st_src(dst) else {
            return;
        };
        match op {
            FpuOp::Com | FpuOp::Comp => {
                if self.fpu_compare(d, src, true) && (op == FpuOp::Comp || pop) {
                    self.fpu_pop();
                }
            }
            _ => {
                let (v, f) = self.fpu_binop(op, d, src);
                if self.x87_flags(f) {
                    self.st_set(dst, v);
                    if pop {
                        self.fpu_pop();
                    }
                }
            }
        }
    }

    /// The compare core (`FCOM`/`FUCOM`/`FTST`/`FICOM`): `C3`/`C2`/`C0` =
    /// `000` greater, `001` less, `100` equal, `111` unordered; `C1` cleared.
    /// `signaling` compares raise `IE` on any NaN, quiet ones only on SNaN.
    /// Returns `false` if an unmasked exception aborted it.
    fn fpu_compare(&mut self, a: F80, b: F80, signaling: bool) -> bool {
        let (ord, flags) = Self::x87_cmp(a, b, signaling);
        if !self.x87_flags(flags) {
            return false;
        }
        let (c3, c2, c0) = match ord {
            None => (true, true, true),
            Some(core::cmp::Ordering::Less) => (false, false, true),
            Some(core::cmp::Ordering::Equal) => (true, false, false),
            Some(core::cmp::Ordering::Greater) => (false, false, false),
        };
        self.fpu_c3 = c3;
        self.fpu_c2 = c2;
        self.fpu_c0 = c0;
        self.fpu_c1 = false;
        true
    }

    /// Order two x87 values with the operand exceptions a compare raises.
    fn x87_cmp(a: F80, b: F80, signaling: bool) -> (Option<core::cmp::Ordering>, u32) {
        let (ua, fa) = operand(a);
        let (ub, fb) = operand(b);
        let mut flags = (fa | fb) & DENORMAL;
        if (fa | fb) & INVALID != 0 {
            return (None, INVALID | flags);
        }
        if ua.is_nan() || ub.is_nan() {
            if signaling || ua.is_snan() || ub.is_snan() {
                flags |= INVALID;
            }
            return (None, flags);
        }
        (Some(sf::compare(&ua, &ub)), flags)
    }

    /// `FCOMI`/`FUCOMI`[`P`]: compare `ST(0)` with `ST(i)` into `ZF`/`PF`/`CF`
    /// (`OF`/`SF`/`AF` cleared).
    fn fpu_comi(&mut self, i: u8, signaling: bool, pop: bool) {
        let (a, b) = if self.st_empty(0) || self.st_empty(i) {
            self.fpu_c1 = false;
            if !self.x87_flags(INVALID | STACK_FAULT) {
                return;
            }
            (F80::INDEFINITE, F80::INDEFINITE)
        } else {
            (self.st_get(0), self.st_get(i))
        };
        let (ord, flags) = Self::x87_cmp(a, b, signaling);
        if !self.x87_flags(flags) {
            return;
        }
        use core::cmp::Ordering;
        self.flags = Flags {
            cf: matches!(ord, None | Some(Ordering::Less)),
            pf: ord.is_none(),
            af: false,
            zf: matches!(ord, None | Some(Ordering::Equal)),
            sf: false,
            of: false,
        };
        self.fpu_c1 = false;
        if pop {
            self.fpu_pop();
        }
    }

    // ---- memory operands -------------------------------------------------------

    /// Load a memory operand as an `F80`, with the exceptions a load raises:
    /// `f32`/`f64` SNaNs are quieted with `IE`, their denormals raise `DE`;
    /// integers and `m80` load exactly and silently.
    fn x87_load(mem: &GuestMemory, addr: u64, fmt: MemFmt) -> Result<(F80, u32), Step> {
        let rd = |n: usize| -> Result<u128, Step> {
            let mut b = [0u8; 16];
            mem.read(addr, &mut b[..n]).map_err(|_| rd_fault(addr))?;
            Ok(u128::from_le_bytes(b))
        };
        Ok(match fmt {
            MemFmt::F32 => {
                let v = sf::unpack_f32(rd(4)? as u32);
                Self::widen(v, FMT32)
            }
            MemFmt::F64 => {
                let v = sf::unpack_f64(rd(8)? as u64);
                Self::widen(v, FMT64)
            }
            MemFmt::F80 => (F80(rd(10)?), 0),
            MemFmt::I16 => (F80::pack(&sf::from_int(i64::from(rd(2)? as u16 as i16))), 0),
            MemFmt::I32 => (F80::pack(&sf::from_int(i64::from(rd(4)? as u32 as i32))), 0),
            MemFmt::I64 => (F80::pack(&sf::from_int(rd(8)? as u64 as i64)), 0),
        })
    }

    /// Widen an `f32`/`f64` value to `F80` (exact), with its load exceptions.
    fn widen(v: Fp, fmt: sf::Fmt) -> (F80, u32) {
        if v.is_snan() {
            (F80::pack(&v.quieted()), INVALID)
        } else {
            let f = if v.is_denormal(fmt) { DENORMAL } else { 0 };
            (F80::pack(&v), f)
        }
    }

    /// Convert `v` for a store in `fmt`: the bytes to write and the flags
    /// (`IE` for SNaN/unsupported/out-of-range integers, `DE`, and the
    /// rounding flags; `C1` = round-up). Integer stores of NaN/∞/overflow
    /// write the integer indefinite.
    fn x87_store_bits(&mut self, v: F80, fmt: MemFmt, truncate: bool) -> (u128, u32) {
        let mode = if truncate {
            Round::Zero
        } else {
            self.fpu_round()
        };
        self.fpu_c1 = false;
        let (u, of) = operand(v);
        match fmt {
            MemFmt::F80 => (v.0, 0),
            MemFmt::F32 | MemFmt::F64 => {
                let ffmt = if fmt == MemFmt::F32 { FMT32 } else { FMT64 };
                let pack = |x: &Fp| -> u128 {
                    if fmt == MemFmt::F32 {
                        u128::from(sf::pack_f32(x))
                    } else {
                        u128::from(sf::pack_f64(x))
                    }
                };
                if of & INVALID != 0 {
                    return (pack(&sf::INDEFINITE), INVALID);
                }
                if u.is_nan() {
                    let f = if u.is_snan() { INVALID } else { 0 };
                    return (pack(&u.quieted()), f);
                }
                let r = sf::round_fp(u, ffmt, mode);
                self.fpu_c1 = r.up;
                (pack(&r.v), r.flags | of)
            }
            MemFmt::I16 | MemFmt::I32 | MemFmt::I64 => {
                let bits = match fmt {
                    MemFmt::I16 => 16,
                    MemFmt::I32 => 32,
                    _ => 64,
                };
                let indef = 1u128 << (bits - 1);
                if of & INVALID != 0 || u.is_nan() {
                    return (indef, INVALID);
                }
                match sf::to_int(u, bits, mode) {
                    Some((i, f, up)) => {
                        self.fpu_c1 = up;
                        (u128::from(i as u64) & ((1u128 << bits) - 1), f | of)
                    }
                    None => (indef, INVALID | of),
                }
            }
        }
    }

    fn mem_len(fmt: MemFmt) -> usize {
        match fmt {
            MemFmt::I16 => 2,
            MemFmt::F32 | MemFmt::I32 => 4,
            MemFmt::F64 | MemFmt::I64 => 8,
            MemFmt::F80 => 10,
        }
    }

    /// `FST`/`FSTP`/`FIST`/`FISTP`/`FISTTP` to memory (`pop` for the `P`
    /// forms). An empty `ST(0)` stores the indefinite (masked).
    fn fpu_store_mem(
        &mut self,
        mem: &mut GuestMemory,
        addr: u64,
        fmt: MemFmt,
        pop: bool,
        truncate: bool,
    ) -> Result<(), Step> {
        let n = Self::mem_len(fmt);
        let (bits, flags) = if self.st_empty(0) {
            self.fpu_c1 = false;
            let indef = match fmt {
                MemFmt::F32 => 0xffc0_0000,
                MemFmt::F64 => 0xfff8_0000_0000_0000,
                MemFmt::F80 => F80::INDEFINITE.0,
                MemFmt::I16 => 0x8000,
                MemFmt::I32 => 0x8000_0000,
                MemFmt::I64 => 0x8000_0000_0000_0000,
            };
            (indef, INVALID | STACK_FAULT)
        } else {
            self.x87_store_bits(self.st_get(0), fmt, truncate)
        };
        // Any unmasked exception suppresses the store (and the pop).
        if (flags as u16) & !self.fpu_cw & 0x3f != 0 {
            self.x87_flags(flags);
            return Ok(());
        }
        // A faulting store leaves the FPU state untouched (the instruction
        // restarts after the page fault).
        mem.write_trap(addr, &bits.to_le_bytes()[..n])
            .map_err(|e| wr_fault(&e))?;
        self.x87_flags(flags);
        if pop {
            self.fpu_pop();
        }
        Ok(())
    }

    /// The `m32/m64/m16int/m32int` arithmetic forms (`D8`/`DC`/`DA`/`DE`
    /// memory): `ST(0) op= [mem]`.
    fn fpu_arith_mem(&mut self, mem: &GuestMemory, addr: u64, op: FpuOp, fmt: MemFmt) -> Step {
        // The memory operand in its own format, so an m32/m64 SNaN signals and
        // its denormals raise DE.
        let Some(ub) = Self::mem_operand_fp(mem, addr, fmt) else {
            return rd_fault(addr);
        };
        match op {
            FpuOp::Com | FpuOp::Comp => {
                let Some(d) = self.st_src(0) else {
                    return Step::Next;
                };
                let (ua, fa) = operand(d);
                let mut flags = fa & (DENORMAL | INVALID);
                if fa & INVALID == 0 && ub.is_denormal(Self::fmt_of(fmt)) {
                    flags |= DENORMAL;
                }
                let ord = if flags & INVALID != 0 || ua.is_nan() || ub.is_nan() {
                    // FCOM/FICOM are signaling compares: any NaN raises IE.
                    flags |= INVALID;
                    None
                } else {
                    Some(sf::compare(&ua, &ub))
                };
                if self.x87_flags(flags) {
                    let (c3, c2, c0) = match ord {
                        None => (true, true, true),
                        Some(core::cmp::Ordering::Less) => (false, false, true),
                        Some(core::cmp::Ordering::Equal) => (true, false, false),
                        Some(core::cmp::Ordering::Greater) => (false, false, false),
                    };
                    self.fpu_c3 = c3;
                    self.fpu_c2 = c2;
                    self.fpu_c0 = c0;
                    self.fpu_c1 = false;
                    if op == FpuOp::Comp {
                        self.fpu_pop();
                    }
                }
            }
            _ => {
                let Some(d) = self.st_src(0) else {
                    return Step::Next;
                };
                let (v, f) = self.fpu_binop_fp(op, d, ub, Self::fmt_of(fmt));
                if self.x87_flags(f) {
                    self.st_set(0, v);
                }
            }
        }
        Step::Next
    }

    /// The unpacked value of a memory operand in its own format (for the
    /// arithmetic forms, which see SNaN/denormal operands as such).
    fn mem_operand_fp(mem: &GuestMemory, addr: u64, fmt: MemFmt) -> Option<Fp> {
        let mut b = [0u8; 16];
        let n = Self::mem_len(fmt);
        mem.read(addr, &mut b[..n]).ok()?;
        let v = u128::from_le_bytes(b);
        Some(match fmt {
            MemFmt::F32 => sf::unpack_f32(v as u32),
            MemFmt::F64 => sf::unpack_f64(v as u64),
            MemFmt::I16 => sf::from_int(i64::from(v as u16 as i16)),
            MemFmt::I32 => sf::from_int(i64::from(v as u32 as i32)),
            MemFmt::I64 => sf::from_int(v as u64 as i64),
            MemFmt::F80 => sf::unpack_f80(v),
        })
    }

    fn fmt_of(fmt: MemFmt) -> sf::Fmt {
        match fmt {
            MemFmt::F32 => FMT32,
            MemFmt::F64 => FMT64,
            _ => FMT80,
        }
    }

    /// [`X86Interp::fpu_binop`] with the second operand already unpacked
    /// (from memory, in format `bfmt`).
    fn fpu_binop_fp(&mut self, op: FpuOp, a: F80, b: Fp, bfmt: sf::Fmt) -> (F80, u32) {
        let (ua, fa) = operand(a);
        let fb = if b.is_denormal(bfmt) { DENORMAL } else { 0 };
        let (ua, ub) = match op {
            FpuOp::SubR | FpuOp::DivR => (b, ua),
            _ => (ua, b),
        };
        self.fpu_c1 = false;
        if fa & INVALID != 0 {
            return (F80::INDEFINITE, INVALID);
        }
        if let Some((n, f)) = x87_nan2(&ua, &ub) {
            return (F80::pack(&n), f);
        }
        let dflags = fa | fb;
        let fmt = self.fpu_fmt();
        let mode = self.fpu_round();
        let r = match op {
            FpuOp::Add => sf::add(ua, ub, fmt, mode),
            FpuOp::Sub | FpuOp::SubR => sf::add(ua, ub.neg(), fmt, mode),
            FpuOp::Mul => sf::mul(ua, ub, fmt, mode),
            _ => sf::div(ua, ub, fmt, mode),
        };
        if r.flags & INVALID != 0 {
            return (F80::INDEFINITE, INVALID | dflags);
        }
        let (v, f) = self.fpu_round_result(r);
        (v, f | dflags)
    }

    // ---- dispatch ------------------------------------------------------------------

    /// Execute an x87 escape opcode (`D8`..`DF`) whose ModRM starts at `pc`.
    pub(super) fn exec_x87(&mut self, mem: &mut GuestMemory, pc: u64, p: Pfx, esc: u8) -> Step {
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        // Every x87 instruction except the non-waiting control forms checks
        // for a pending unmasked exception first (#MF).
        let no_wait = match (esc, m.kind) {
            (0xD9, RmKind::Reg(_)) => false,
            (0xD9, _) => matches!(m.ext(), 6 | 7), // FNSTENV, FNSTCW
            (0xDB, RmKind::Reg(r)) => matches!(r & 7, 2 | 3) && m.ext() == 4, // FNCLEX, FNINIT
            (0xDD, RmKind::Reg(_)) => false,
            (0xDD, _) => matches!(m.ext(), 6 | 7), // FNSAVE, FNSTSW m
            (0xDF, RmKind::Reg(r)) => m.ext() == 4 && r & 7 == 0, // FNSTSW AX
            _ => false,
        };
        if !no_wait && self.fpu_pending() {
            return Step::Trap(Trap::X87);
        }
        let step = match m.kind {
            RmKind::Reg(r) => self.x87_reg(mem, esc, m.ext(), (r & 7) as u8),
            _ => {
                let addr = fetch!(self.mem_only(m.kind, end));
                self.x87_mem(mem, esc, m, addr, p)
            }
        };
        match step {
            Step::Next => self.next(end),
            other => other,
        }
    }

    /// The memory forms.
    #[allow(clippy::too_many_lines)]
    fn x87_mem(&mut self, mem: &mut GuestMemory, esc: u8, m: ModRm, addr: u64, p: Pfx) -> Step {
        let ext = m.ext();
        let load = |s: &mut Self, mem: &GuestMemory, fmt: MemFmt| -> Step {
            let (v, f) = match Self::x87_load(mem, addr, fmt) {
                Ok(x) => x,
                Err(e) => return e,
            };
            if s.x87_flags(f) {
                s.fpu_c1 = false;
                s.fpu_push(v);
            }
            Step::Next
        };
        let store =
            |s: &mut Self, mem: &mut GuestMemory, fmt: MemFmt, pop: bool, trunc: bool| match s
                .fpu_store_mem(mem, addr, fmt, pop, trunc)
            {
                Ok(()) => Step::Next,
                Err(e) => e,
            };
        match (esc, ext) {
            (0xD8, _) => self.fpu_arith_mem(mem, addr, FpuOp::from_ext(ext), MemFmt::F32),
            (0xDC, _) => self.fpu_arith_mem(mem, addr, FpuOp::from_ext(ext), MemFmt::F64),
            (0xDA, _) => self.fpu_arith_mem(mem, addr, FpuOp::from_ext(ext), MemFmt::I32),
            (0xDE, _) => self.fpu_arith_mem(mem, addr, FpuOp::from_ext(ext), MemFmt::I16),
            (0xD9, 0) => load(self, mem, MemFmt::F32),
            (0xD9, 2) => store(self, mem, MemFmt::F32, false, false),
            (0xD9, 3) => store(self, mem, MemFmt::F32, true, false),
            (0xD9, 4) => self
                .fldenv(mem, addr, p.opsize)
                .map_or_else(|e| e, |()| Step::Next),
            (0xD9, 5) => {
                let mut b = [0u8; 2];
                if mem.read(addr, &mut b).is_err() {
                    return rd_fault(addr);
                }
                self.set_fpu_cw(u16::from_le_bytes(b));
                Step::Next
            }
            (0xD9, 6) => {
                // FNSTENV, then mask all exceptions.
                if let Err(e) = self.fnstenv(mem, addr, p.opsize) {
                    return e;
                }
                self.fpu_cw |= 0x3f;
                Step::Next
            }
            (0xD9, 7) => match mem.write_trap(addr, &self.fpu_cw.to_le_bytes()) {
                Ok(()) => Step::Next,
                Err(e) => wr_fault(&e),
            },
            (0xDB, 0) => load(self, mem, MemFmt::I32),
            (0xDB, 1) if super::sse::SSE3 => store(self, mem, MemFmt::I32, true, true),
            (0xDB, 2) => store(self, mem, MemFmt::I32, false, false),
            (0xDB, 3) => store(self, mem, MemFmt::I32, true, false),
            (0xDB, 5) => load(self, mem, MemFmt::F80),
            (0xDB, 7) => store(self, mem, MemFmt::F80, true, false),
            (0xDD, 0) => load(self, mem, MemFmt::F64),
            (0xDD, 1) if super::sse::SSE3 => store(self, mem, MemFmt::I64, true, true),
            (0xDD, 2) => store(self, mem, MemFmt::F64, false, false),
            (0xDD, 3) => store(self, mem, MemFmt::F64, true, false),
            (0xDD, 4) => self
                .frstor(mem, addr, p.opsize)
                .map_or_else(|e| e, |()| Step::Next),
            (0xDD, 6) => match self.fnsave(mem, addr, p.opsize) {
                Ok(()) => {
                    self.fpu_init();
                    Step::Next
                }
                Err(e) => e,
            },
            (0xDD, 7) => match mem.write_trap(addr, &self.fpu_sw().to_le_bytes()) {
                Ok(()) => Step::Next,
                Err(e) => wr_fault(&e),
            },
            (0xDF, 0) => load(self, mem, MemFmt::I16),
            (0xDF, 1) if super::sse::SSE3 => store(self, mem, MemFmt::I16, true, true),
            (0xDF, 2) => store(self, mem, MemFmt::I16, false, false),
            (0xDF, 3) => store(self, mem, MemFmt::I16, true, false),
            (0xDF, 4) => self.fbld(mem, addr),
            (0xDF, 5) => load(self, mem, MemFmt::I64),
            (0xDF, 6) => self.fbstp(mem, addr),
            (0xDF, 7) => store(self, mem, MemFmt::I64, true, false),
            _ => Step::Illegal,
        }
    }

    /// The register forms.
    #[allow(clippy::too_many_lines)]
    fn x87_reg(&mut self, mem: &mut GuestMemory, esc: u8, ext: usize, i: u8) -> Step {
        let _ = mem;
        match (esc, ext) {
            (0xD8, _) => {
                let Some(src) = self.st_src(i) else {
                    return Step::Next;
                };
                self.fpu_arith(FpuOp::from_ext(ext), 0, src, false);
            }
            (0xDC | 0xDE, _) => {
                let op = FpuOp::from_ext_reversed(ext);
                let pop = esc == 0xDE;
                if pop && ext == 3 {
                    // DE D9: FCOMPP (other DE D8-DF encodings are #UD).
                    if i != 1 {
                        return Step::Illegal;
                    }
                    let (Some(a), Some(b)) = (self.st_src(0), self.st_src(1)) else {
                        return Step::Next;
                    };
                    if self.fpu_compare(a, b, true) {
                        self.fpu_pop();
                        self.fpu_pop();
                    }
                    return Step::Next;
                }
                if matches!(op, FpuOp::Com | FpuOp::Comp) {
                    // DC D0/D8 (FCOM2/FCOMP3) and DE D0 (FCOMP5) aliases:
                    // compare ST(0) with ST(i).
                    let (Some(a), Some(b)) = (self.st_src(0), self.st_src(i)) else {
                        return Step::Next;
                    };
                    if self.fpu_compare(a, b, true) && (op == FpuOp::Comp || pop) {
                        self.fpu_pop();
                    }
                    return Step::Next;
                }
                let Some(src) = self.st_src(0) else {
                    return Step::Next;
                };
                self.fpu_arith(op, i, src, pop);
            }
            (0xD9, 0) => {
                // FLD ST(i)
                let v = if self.st_empty(i) {
                    self.fpu_c1 = false;
                    if !self.x87_flags(INVALID | STACK_FAULT) {
                        return Step::Next;
                    }
                    F80::INDEFINITE
                } else {
                    self.st_get(i)
                };
                self.fpu_c1 = false;
                self.fpu_push(v);
            }
            (0xD9 | 0xDD | 0xDF, 1) => self.fxch(i),
            (0xD9, 2) => {
                if i != 0 {
                    return Step::Illegal;
                } // FNOP
            }
            (0xD9, 3) | (0xDF, 2 | 3) => self.fst_reg(i, true), // FSTP1/FSTP8/FSTP9 aliases
            (0xDD, 2) => self.fst_reg(i, false),
            (0xDD, 3) => self.fst_reg(i, true),
            (0xD9, 4) => return self.x87_d9_e0(i),
            (0xD9, 5) => {
                // Constants: FLD1, FLDL2T, FLDL2E, FLDPI, FLDLG2, FLDLN2, FLDZ.
                // The rounded-up/down forms of the inexact constants follow
                // RC, as on hardware.
                let Some(c) = x87math::constant(i, self.fpu_round()) else {
                    return Step::Illegal;
                };
                self.fpu_c1 = false;
                self.fpu_push(c);
            }
            (0xD9, 6 | 7) => return self.x87_transcendental(ext, i),
            (0xDA | 0xDB, 0..=3) => {
                // FCMOVcc: B, E, BE, U (DA) and their negations (DB).
                let f = self.flags;
                let cond = match ext {
                    0 => f.cf,
                    1 => f.zf,
                    2 => f.cf || f.zf,
                    _ => f.pf,
                };
                let cond = if esc == 0xDB { !cond } else { cond };
                // An empty source is a stack underflow (whether or not the
                // move happens); the masked response loads the indefinite.
                if self.st_empty(i) {
                    self.fpu_c1 = false;
                    if self.x87_flags(INVALID | STACK_FAULT) {
                        self.st_set(0, F80::INDEFINITE);
                    }
                    return Step::Next;
                }
                if cond {
                    let v = self.st_get(i);
                    self.st_set(0, v);
                }
                self.fpu_c1 = false;
            }
            (0xDA, 5) => {
                // DA E9: FUCOMPP
                if i != 1 {
                    return Step::Illegal;
                }
                let (Some(a), Some(b)) = (self.st_src(0), self.st_src(1)) else {
                    return Step::Next;
                };
                if self.fpu_compare(a, b, false) {
                    self.fpu_pop();
                    self.fpu_pop();
                }
            }
            (0xDB, 4) => match i {
                0 | 1 | 4 => {}          // FNENI, FNDISI, FNSETPM: no-ops since the 387
                2 => self.fpu_flags = 0, // FNCLEX
                3 => self.fpu_init(),    // FNINIT
                _ => return Step::Illegal,
            },
            (0xDB, 5) => self.fpu_comi(i, false, false), // FUCOMI
            (0xDB, 6) => self.fpu_comi(i, true, false),  // FCOMI
            (0xDD, 0) => {
                // FFREE ST(i)
                let idx = self.st_idx(i);
                self.fpu_tag &= !(1 << idx);
            }
            (0xDF, 0) => {
                // FFREEP ST(i) (undocumented): free, then pop.
                let idx = self.st_idx(i);
                self.fpu_tag &= !(1 << idx);
                self.fpu_pop();
            }
            (0xDD, 4 | 5) => {
                // FUCOM / FUCOMP ST(i)
                let (Some(a), Some(b)) = (self.st_src(0), self.st_src(i)) else {
                    return Step::Next;
                };
                if self.fpu_compare(a, b, false) && ext == 5 {
                    self.fpu_pop();
                }
            }
            (0xDF, 4) => {
                // DF E0: FNSTSW AX
                if i != 0 {
                    return Step::Illegal;
                }
                let sw = u64::from(self.fpu_sw());
                self.gpr[RAX] = (self.gpr[RAX] & !0xffff) | sw;
            }
            (0xDF, 5) => self.fpu_comi(i, false, true), // FUCOMIP
            (0xDF, 6) => self.fpu_comi(i, true, true),  // FCOMIP
            _ => return Step::Illegal,
        }
        Step::Next
    }

    /// `FXCH ST(i)` (and its `DD C8`/`DF C8` aliases): empty registers are
    /// first replaced by the indefinite (masked stack fault).
    fn fxch(&mut self, i: u8) {
        if self.st_empty(0) || self.st_empty(i) {
            self.fpu_c1 = false;
            if !self.x87_flags(INVALID | STACK_FAULT) {
                return;
            }
            for r in [0, i] {
                if self.st_empty(r) {
                    self.st_set(r, F80::INDEFINITE);
                }
            }
        }
        let a = self.st_get(0);
        let b = self.st_get(i);
        self.st_set(0, b);
        self.st_set(i, a);
        self.fpu_c1 = false;
    }

    /// `FST`/`FSTP ST(i)`.
    fn fst_reg(&mut self, i: u8, pop: bool) {
        let v = if self.st_empty(0) {
            self.fpu_c1 = false;
            if !self.x87_flags(INVALID | STACK_FAULT) {
                return;
            }
            F80::INDEFINITE
        } else {
            self.st_get(0)
        };
        self.st_set(i, v);
        self.fpu_c1 = false;
        if pop {
            self.fpu_pop();
        }
    }

    /// `D9 E0..E7`: `FCHS`, `FABS`, `FTST`, `FXAM`.
    fn x87_d9_e0(&mut self, i: u8) -> Step {
        match i {
            0 | 1 => {
                // FCHS / FABS. A masked stack underflow yields the plain
                // indefinite.
                if self.st_empty(0) {
                    self.fpu_c1 = false;
                    if self.x87_flags(INVALID | STACK_FAULT) {
                        self.st_set(0, F80::INDEFINITE);
                    }
                    return Step::Next;
                }
                let v = self.st_get(0);
                let r = if i == 0 { v.neg() } else { v.abs() };
                self.st_set(0, r);
                self.fpu_c1 = false;
            }
            4 => {
                // FTST: compare ST(0) with +0.0 (signaling).
                let Some(v) = self.st_src(0) else {
                    return Step::Next;
                };
                self.fpu_compare(v, F80::ZERO, true);
            }
            5 => {
                // FXAM: C1 = sign; C3,C2,C0 = class.
                let v = self.st_get(0);
                self.fpu_c1 = v.sign();
                let (c3, c2, c0) = if self.st_empty(0) {
                    (true, false, true)
                } else if sf::f80_unsupported(v.0) {
                    (false, false, false)
                } else {
                    let e = v.exp_field();
                    let frac = v.mant() << 1;
                    if e == 0x7fff {
                        if frac == 0 {
                            (false, true, true) // infinity
                        } else {
                            (false, false, true) // NaN
                        }
                    } else if e == 0 {
                        if v.mant() == 0 {
                            (true, false, false) // zero
                        } else {
                            (true, true, false) // denormal
                        }
                    } else {
                        (false, true, false) // normal
                    }
                };
                self.fpu_c3 = c3;
                self.fpu_c2 = c2;
                self.fpu_c0 = c0;
            }
            _ => return Step::Illegal,
        }
        Step::Next
    }

    // ---- environment / state ----------------------------------------------------

    /// The 28-byte (32-bit) or 14-byte (`0x66`, 16-bit) protected-mode
    /// environment: control/status/tag words, then the instruction and data
    /// pointers (zero here). Reserved halves read as zero.
    fn env_bytes(&self, small: bool) -> Vec<u8> {
        let words = [self.fpu_cw, self.fpu_sw(), self.fpu_full_tag()];
        let mut out = Vec::with_capacity(28);
        if small {
            for w in words {
                out.extend_from_slice(&w.to_le_bytes());
            }
            out.extend_from_slice(&[0u8; 8]);
        } else {
            for w in words {
                out.extend_from_slice(&u32::from(w).to_le_bytes());
            }
            out.extend_from_slice(&(self.fpu_fip as u32).to_le_bytes());
            out.extend_from_slice(&(u32::from(self.fpu_fop) << 16).to_le_bytes());
            out.extend_from_slice(&(self.fpu_fdp as u32).to_le_bytes());
            out.extend_from_slice(&[0u8; 4]);
        }
        out
    }

    fn load_env(&mut self, b: &[u8], small: bool) {
        let w = |i: usize| -> u16 {
            let o = if small { 2 * i } else { 4 * i };
            u16::from_le_bytes([b[o], b[o + 1]])
        };
        self.set_fpu_cw(w(0));
        self.set_fpu_sw(w(1));
        self.set_full_tag(w(2));
    }

    fn fnstenv(&self, mem: &mut GuestMemory, addr: u64, small: bool) -> Result<(), Step> {
        mem.write_trap(addr, &self.env_bytes(small))
            .map_err(|e| wr_fault(&e))
    }

    fn fldenv(&mut self, mem: &GuestMemory, addr: u64, small: bool) -> Result<(), Step> {
        let n = if small { 14 } else { 28 };
        let mut b = [0u8; 28];
        mem.read(addr, &mut b[..n]).map_err(|_| rd_fault(addr))?;
        self.load_env(&b, small);
        Ok(())
    }

    /// `FNSAVE`: the environment followed by `ST(0)..ST(7)` (10 bytes each).
    fn fnsave(&self, mem: &mut GuestMemory, addr: u64, small: bool) -> Result<(), Step> {
        let mut out = self.env_bytes(small);
        for i in 0..8 {
            out.extend_from_slice(&self.st_get(i).0.to_le_bytes()[..10]);
        }
        mem.write_trap(addr, &out).map_err(|e| wr_fault(&e))
    }

    fn frstor(&mut self, mem: &GuestMemory, addr: u64, small: bool) -> Result<(), Step> {
        let env = if small { 14 } else { 28 };
        let mut b = [0u8; 108];
        mem.read(addr, &mut b[..env + 80])
            .map_err(|_| rd_fault(addr))?;
        self.load_env(&b, small);
        for i in 0..8u8 {
            let o = env + 10 * usize::from(i);
            let mut r = [0u8; 16];
            r[..10].copy_from_slice(&b[o..o + 10]);
            let idx = self.st_idx(i);
            self.st[idx] = F80(u128::from_le_bytes(r));
        }
        Ok(())
    }

    // ---- BCD ---------------------------------------------------------------------

    /// `FBLD m80bcd`: push an 18-digit packed-BCD integer.
    fn fbld(&mut self, mem: &GuestMemory, addr: u64) -> Step {
        let mut b = [0u8; 10];
        if mem.read(addr, &mut b).is_err() {
            return rd_fault(addr);
        }
        let mut v: i64 = 0;
        for k in (0..9).rev() {
            v = v * 100 + i64::from(b[k] >> 4) * 10 + i64::from(b[k] & 15);
        }
        let mut fp = sf::from_int(v);
        if b[9] & 0x80 != 0 {
            fp = fp.neg();
        }
        self.fpu_c1 = false;
        self.fpu_push(F80::pack(&fp));
        Step::Next
    }

    /// `FBSTP m80bcd`: store `ST(0)` rounded per `RC` as packed BCD, then pop.
    /// NaN/∞/out-of-range values store the BCD indefinite with `IE`.
    fn fbstp(&mut self, mem: &mut GuestMemory, addr: u64) -> Step {
        const INDEF: [u8; 10] = [0, 0, 0, 0, 0, 0, 0, 0xC0, 0xFF, 0xFF];
        let (bytes, flags) = if self.st_empty(0) {
            self.fpu_c1 = false;
            (INDEF, INVALID | STACK_FAULT)
        } else {
            let (u, of) = operand(self.st_get(0));
            self.fpu_c1 = false;
            if of & INVALID != 0 || u.is_nan() {
                (INDEF, INVALID)
            } else {
                match sf::to_int(u, 64, self.fpu_round()) {
                    Some((i, f, up)) if i.unsigned_abs() < 1_000_000_000_000_000_000 => {
                        self.fpu_c1 = up;
                        let mut out = [0u8; 10];
                        let mut mag = i.unsigned_abs();
                        for byte in out.iter_mut().take(9) {
                            let lo = (mag % 10) as u8;
                            mag /= 10;
                            let hi = (mag % 10) as u8;
                            mag /= 10;
                            *byte = (hi << 4) | lo;
                        }
                        if u.sign {
                            out[9] = 0x80;
                        }
                        (out, f | of)
                    }
                    _ => (INDEF, INVALID | of),
                }
            }
        };
        if let Err(e) = mem.write_trap(addr, &bytes) {
            return wr_fault(&e);
        }
        if self.x87_flags(flags) {
            self.fpu_pop();
        }
        Step::Next
    }

    // ---- the D9 F0..FF group ------------------------------------------------------

    /// `D9 F0..FF`: `F2XM1`, `FYL2X`, `FPTAN`, `FPATAN`, `FXTRACT`, `FPREM1`,
    /// `FDECSTP`, `FINCSTP`, `FPREM`, `FYL2XP1`, `FSQRT`, `FSINCOS`,
    /// `FRNDINT`, `FSCALE`, `FSIN`, `FCOS`.
    #[allow(clippy::too_many_lines)]
    fn x87_transcendental(&mut self, ext: usize, i: u8) -> Step {
        let code = (ext - 6) * 8 + usize::from(i);
        match code {
            6 => {
                self.fpu_top = self.fpu_top.wrapping_sub(1) & 7; // FDECSTP
                self.fpu_c1 = false;
                return Step::Next;
            }
            7 => {
                self.fpu_top = (self.fpu_top + 1) & 7; // FINCSTP
                self.fpu_c1 = false;
                return Step::Next;
            }
            _ => {}
        }
        // FPTAN/FXTRACT/FSINCOS push a second result: with the stack full
        // that is an overflow, whose masked response leaves the indefinite
        // in both result registers.
        // (An out-of-range trig operand is reported first: C2, no change.)
        let out_of_range = matches!(code, 2 | 11) && !self.st_empty(0) && {
            let v = self.st_get(0);
            v.exp_field() != 0x7fff && v.exp_field() >= 0x3fff + 63
        };
        if out_of_range {
            self.fpu_c2 = true;
            return Step::Next;
        }
        if matches!(code, 2 | 4 | 11)
            && self.fpu_tag & (1 << (self.fpu_top.wrapping_sub(1) & 7)) != 0
        {
            // The function itself is still evaluated (its precision flag is
            // reported alongside the fault); C1 flags the overflow — unless
            // ST(0) was empty too, an underflow (C1 = 0).
            let (c1, f) = if code == 4 {
                (true, 0)
            } else if self.st_empty(0) {
                (false, 0)
            } else {
                let r = x87math::eval(code, self.st_get(0), F80::ZERO, self.fpu_round());
                (true, r.flags & !INVALID)
            };
            self.fpu_c1 = c1;
            if self.x87_flags(INVALID | STACK_FAULT | f) {
                self.st_set(0, F80::INDEFINITE);
                self.fpu_top = self.fpu_top.wrapping_sub(1) & 7;
                self.st_set(0, F80::INDEFINITE);
            }
            return Step::Next;
        }
        // Two-operand forms read ST(0) and ST(1).
        let two = matches!(code, 1 | 3 | 5 | 8 | 9 | 13);
        let Some(x) = self.st_src(0) else {
            return Step::Next;
        };
        let y = if two {
            let Some(y) = self.st_src(1) else {
                return Step::Next;
            };
            y
        } else {
            F80::ZERO
        };
        let mode = self.fpu_round();
        match code {
            10 => {
                // FSQRT (precision control applies)
                let (u, f) = operand(x);
                self.fpu_c1 = false;
                let (v, flags) = if f & INVALID != 0 {
                    (F80::INDEFINITE, INVALID)
                } else if u.is_nan() {
                    (
                        F80::pack(&u.quieted()),
                        if u.is_snan() { INVALID } else { 0 },
                    )
                } else {
                    let r = sf::sqrt(u, self.fpu_fmt(), mode);
                    if r.flags & INVALID != 0 {
                        (F80::INDEFINITE, INVALID | f)
                    } else {
                        let (v, fl) = self.fpu_round_result(r);
                        (v, fl | f)
                    }
                };
                if self.x87_flags(flags) {
                    self.st_set(0, v);
                }
            }
            12 => {
                // FRNDINT
                let (u, f) = operand(x);
                self.fpu_c1 = false;
                let (v, flags) = if f & INVALID != 0 {
                    (F80::INDEFINITE, INVALID)
                } else if u.is_nan() {
                    (
                        F80::pack(&u.quieted()),
                        if u.is_snan() { INVALID } else { 0 },
                    )
                } else {
                    let r = sf::round_to_int(u, mode);
                    let (v, fl) = self.fpu_round_result(r);
                    (v, fl | f)
                };
                if self.x87_flags(flags) {
                    self.st_set(0, v);
                }
            }
            4 => {
                // FXTRACT: ST(0) = exponent, push significand.
                let (u, f) = operand(x);
                self.fpu_c1 = false;
                let (e, s, flags) = match u.class {
                    _ if f & INVALID != 0 => (F80::INDEFINITE, F80::INDEFINITE, INVALID),
                    Class::Nan => {
                        let q = F80::pack(&u.quieted());
                        (q, q, if u.is_snan() { INVALID } else { 0 })
                    }
                    Class::Zero => (
                        F80::pack(&Fp::inf(true)),
                        F80::pack(&Fp::zero(u.sign)),
                        DIVZERO,
                    ),
                    Class::Inf => (F80::pack(&Fp::inf(false)), x, 0),
                    Class::Finite => {
                        let le = u.lead_exp();
                        let sig = Fp::finite(u.sign, u.sig, u.exp - le);
                        (F80::pack(&sf::from_int(i64::from(le))), F80::pack(&sig), f)
                    }
                };
                if self.x87_flags(flags) {
                    self.st_set(0, e);
                    self.fpu_push(s);
                }
            }
            8 | 5 => self.fprem(x, y, code == 5),
            13 => self.fscale(x, y),
            _ => {
                let r = x87math::eval(code, x, y, mode);
                self.fpu_c1 = r.c1;
                if matches!(code, 2 | 11 | 14 | 15) {
                    // Only the trigonometric forms define C2 (reduction
                    // incomplete); the others leave it.
                    self.fpu_c2 = r.c2;
                }
                if !self.x87_flags(r.flags) {
                    return Step::Next;
                }
                if r.incomplete {
                    return Step::Next;
                }
                match code {
                    // FYL2X, FPATAN, FYL2XP1: result into ST(1), pop.
                    1 | 3 | 9 => {
                        self.st_set(1, r.a);
                        self.fpu_pop();
                    }
                    // FPTAN: ST(0) = tan, push 1.0 (a NaN operand pushes the
                    // NaN again).
                    2 => {
                        self.st_set(0, r.a);
                        let one = if r.a.exp_field() == 0x7fff && r.a.mant() << 1 != 0 {
                            r.a
                        } else {
                            F80::ONE
                        };
                        self.fpu_push(one);
                    }
                    // FSINCOS: ST(0) = sin, push cos.
                    11 => {
                        self.st_set(0, r.a);
                        self.fpu_push(r.b);
                    }
                    // F2XM1, FSIN, FCOS.
                    _ => self.st_set(0, r.a),
                }
            }
        }
        Step::Next
    }

    /// `FSCALE`: `ST(0) = ST(0) · 2^trunc(ST(1))`.
    fn fscale(&mut self, x: F80, y: F80) {
        let (ux, fx) = operand(x);
        let (uy, fy) = operand(y);
        self.fpu_c1 = false;
        let mode = self.fpu_round();
        let (v, flags) = if (fx | fy) & INVALID != 0 {
            (F80::INDEFINITE, INVALID)
        } else if let Some((n, f)) = x87_nan2(&ux, &uy) {
            (F80::pack(&n), f)
        } else {
            let d = (fx | fy) & DENORMAL;
            match (ux.class, uy.class) {
                (Class::Zero, Class::Inf) if !uy.sign => (F80::INDEFINITE, INVALID | d),
                (Class::Inf, Class::Inf) if uy.sign => (F80::INDEFINITE, INVALID | d),
                (Class::Zero | Class::Inf, _) => (x, d),
                (Class::Finite, Class::Inf) => {
                    if uy.sign {
                        (F80::pack(&Fp::zero(ux.sign)), d)
                    } else {
                        (F80::pack(&Fp::inf(ux.sign)), d)
                    }
                }
                (Class::Finite, _) => {
                    // trunc(ST(1)), saturated far past the exponent range.
                    let n = match sf::round_to_int(uy, Round::Zero).v {
                        t if t.class == Class::Zero => 0i64,
                        t => {
                            // Far past the exponent range either way: saturate.
                            let mag = if t.lead_exp() > 20 {
                                1i64 << 20
                            } else {
                                (t.sig << t.exp) as i64
                            };
                            if t.sign { -mag } else { mag }
                        }
                    };
                    let r = sf::round(ux.sign, ux.sig, ux.exp + n as i32, false, FMT80, mode);
                    let (v, f) = self.fpu_round_result(r);
                    (v, f | d)
                }
                _ => (x, d),
            }
        };
        if self.x87_flags(flags) {
            self.st_set(0, v);
        }
    }

    /// `FPREM` (`nearest == false`: truncated quotient, the C `fmod`) and
    /// `FPREM1` (`true`: round-to-nearest-even quotient, the IEEE remainder):
    /// `ST(0) = ST(0) - ST(1)·Q`, exactly. When the exponents differ by 64 or
    /// more, only a partial reduction is done (`C2 = 1`, reduce again); else
    /// `C2 = 0` and the low three quotient bits land in `C0`/`C3`/`C1`.
    fn fprem(&mut self, x: F80, y: F80, nearest: bool) {
        let (ux, fx) = operand(x);
        let (uy, fy) = operand(y);
        let d = (fx | fy) & DENORMAL;
        self.fpu_c1 = false;
        self.fpu_c2 = false;
        let (v, flags) = if (fx | fy) & INVALID != 0 {
            (F80::INDEFINITE, INVALID)
        } else if let Some((n, f)) = x87_nan2(&ux, &uy) {
            (F80::pack(&n), f)
        } else {
            match (ux.class, uy.class) {
                (Class::Inf, _) | (_, Class::Zero) => (F80::INDEFINITE, INVALID | d),
                (Class::Zero, _) | (_, Class::Inf) => {
                    self.fpu_c0 = false;
                    self.fpu_c3 = false;
                    (x, d)
                }
                _ => {
                    let (r, q, partial) = x87math::fprem(&ux, &uy, nearest);
                    self.fpu_c2 = partial;
                    if !partial {
                        self.fpu_c0 = q & 4 != 0;
                        self.fpu_c3 = q & 2 != 0;
                        self.fpu_c1 = q & 1 != 0;
                    }
                    // An exact remainder; only a denormal result can be
                    // tiny (underflow is reported, it is never inexact).
                    let rr = sf::round_fp(r, FMT80, Round::Nearest);
                    let uf = if rr.tiny { UNDERFLOW } else { 0 };
                    let uf = uf & !(UNDERFLOW * u32::from(self.fpu_cw & 0x10 != 0));
                    (F80::pack(&rr.v), d | uf)
                }
            }
        };
        if self.x87_flags(flags) {
            self.st_set(0, v);
        }
    }
}
