//! Software CPU interpreter backend for x86-64 guests — the portable,
//! no-acceleration fallback for `Arch::X86_64` (mirrors [`super::interp`]'s
//! aarch64 interpreter, but decodes variable-length x86 instructions instead
//! of fixed 4-byte ones).
//!
//! This is a scaffold, not a full x86-64 implementation, but it now covers
//! enough of the instruction set to run a non-trivial statically-linked ELF
//! (arithmetic loops, byte/word/dword/qword memory traffic, string-copy
//! idioms). Coverage: REX-prefixed and non-REX `MOV` (reg/reg, imm→reg,
//! reg↔mem via ModRM+SIB+disp8/32, RIP-relative, `MOVABS`, 8-bit forms),
//! `MOVZX`/`MOVSX`/`MOVSXD`, `LEA`, the ALU group (`ADD`/`SUB`/`AND`/`OR`/
//! `XOR`/`CMP`/`TEST`) in register, immediate, and 8-bit forms with full flag
//! computation (CF/ZF/SF/OF/PF), `MUL`/`IMUL`/`DIV`/`IDIV`/`NOT`/`NEG`,
//! `CDQ`/`CQO`/`CWDE`/`CDQE`, `CMOVcc`/`SETcc` (all 16 conditions), `PUSH`/
//! `POP` (register, immediate, and r/m via Group 5), `CALL`/`JMP`
//! (`rel32` and r/m indirect)/`RET`/`LEAVE`, `Jcc rel8/rel32` (all 16
//! conditions), `INC`/`DEC` (Group 4/5), `SHL`/`SHR`/`SAR` by an immediate or
//! `CL`, `XCHG`, the `REP`/`REPE`/`REPNE`-prefixed string ops (`MOVS`/`STOS`/
//! `LODS`/`SCAS`/`CMPS`, honoring `DF` via `CLD`/`STD`), and `SYSCALL`. The
//! `0x66` operand-size prefix is decoded (16-bit width) even though most
//! flag/overflow edge cases are only exercised at 32/64-bit widths. Anything
//! else surfaces as [`Exit::IllegalInstruction`].

use crate::abi::Arch;

use super::{Backend, Exit, GuestMemory, Vcpu, VcpuError};

/// Upper bound on instructions executed per `run()` call before yielding —
/// mirrors [`super::interp`]'s guard against a runaway guest loop.
const MAX_STEPS: u64 = 50_000_000;

// ---- x86-64 GPR indices (the standard ModRM/REX numbering) ----
const RAX: usize = 0;
const RCX: usize = 1;
const RDX: usize = 2;
#[allow(dead_code)] // named for documentation of the register file layout
const RBX: usize = 3;
const RSP: usize = 4;
const RBP: usize = 5;
const RSI: usize = 6;
const RDI: usize = 7;
const R8: usize = 8;
const R9: usize = 9;
const R10: usize = 10;

#[derive(Debug)]
pub struct X86Backend {
    guest: Arch,
}

impl X86Backend {
    pub fn new(guest: Arch) -> Result<Self, VcpuError> {
        Ok(Self { guest })
    }
}

impl Backend for X86Backend {
    fn name(&self) -> &'static str {
        "interp-x86"
    }

    fn guest_arch(&self) -> Arch {
        self.guest
    }

    fn new_vcpu(&self, entry: u64, stack: u64) -> Result<Box<dyn Vcpu>, VcpuError> {
        match self.guest {
            Arch::X86_64 => Ok(Box::new(X86Interp::new(entry, stack))),
            Arch::Aarch64 => Err(VcpuError::Backend(
                "interp-x86 backend only supports x86-64 guests".into(),
            )),
        }
    }
}

/// Outcome of executing one instruction.
enum Step {
    /// Advance `rip` to the address just past the decoded instruction.
    Next,
    /// Instruction already set `rip` (branch/call/ret/jmp); do not auto-advance.
    Branched,
    /// `syscall` — hand control to the kernel. `rip` stays on the `syscall`
    /// opcode; the kernel advances it via [`Vcpu::set_syscall_ret`].
    Syscall,
    Illegal,
    /// A load/store/fetch touched bad guest memory.
    Fault { addr: u64, write: bool },
}

/// EFLAGS bits this interpreter tracks: CF/ZF/SF/OF/PF.
#[derive(Default, Clone, Copy)]
#[allow(clippy::struct_excessive_bools)]
struct Flags {
    cf: bool,
    zf: bool,
    sf: bool,
    of: bool,
    pf: bool,
}

/// Decoded REX prefix bits (all `false` when the instruction has none).
#[derive(Clone, Copy, Default)]
#[allow(clippy::struct_excessive_bools)]
struct Rex {
    w: bool,
    r: bool,
    x: bool,
    b: bool,
}

impl Rex {
    fn from_byte(byte: u8) -> Self {
        Self {
            w: byte & 0x08 != 0,
            r: byte & 0x04 != 0,
            x: byte & 0x02 != 0,
            b: byte & 0x01 != 0,
        }
    }
}

/// A decoded ModRM byte (plus any SIB/displacement that followed it).
struct ModRm {
    /// The `reg` field, extended by `REX.R`.
    reg: usize,
    kind: RmKind,
}

/// The r/m operand before RIP-relative addresses are resolved (resolving
/// requires knowing the address of the *end* of the instruction, which isn't
/// known until any trailing immediate has also been decoded).
#[derive(Clone, Copy)]
enum RmKind {
    Reg(usize),
    Mem(u64),
    /// `[rip + disp]`; resolved against the end-of-instruction address.
    MemRip(i64),
}

/// A fully-resolved operand.
#[derive(Clone, Copy)]
enum Operand {
    Reg(usize),
    /// The high byte (bits 15:8) of `gpr[r]` — `AH`/`CH`/`DH`/`BH`, only
    /// reachable for an 8-bit operand with `r` in `0..4` and no `REX` prefix.
    Reg8Hi(usize),
    Mem(u64),
}

fn resolve(kind: RmKind, end_pc: u64) -> Operand {
    match kind {
        RmKind::Reg(r) => Operand::Reg(r),
        RmKind::Mem(a) => Operand::Mem(a),
        RmKind::MemRip(disp) => Operand::Mem((end_pc as i64).wrapping_add(disp) as u64),
    }
}

/// Like [`resolve`], but for an 8-bit operand: without a `REX` prefix, ModRM
/// register indices 4..=7 name `AH`/`CH`/`DH`/`BH` (the high byte of
/// `RAX..RBX`) rather than the low byte of `RSP..RDI`.
fn resolve8(kind: RmKind, end_pc: u64, has_rex: bool) -> Operand {
    match kind {
        RmKind::Reg(r) => reg8_operand(r, has_rex),
        _ => resolve(kind, end_pc),
    }
}

/// Map a ModRM `reg` (or `rm` in register form) field to the 8-bit operand it
/// names — see [`resolve8`].
fn reg8_operand(r: usize, has_rex: bool) -> Operand {
    if !has_rex && (4..=7).contains(&r) {
        Operand::Reg8Hi(r - 4)
    } else {
        Operand::Reg(r)
    }
}

/// Arithmetic/logical operation selected by an ALU opcode or a group-1 `/r`
/// field.
#[derive(Clone, Copy, PartialEq, Eq)]
enum AluOp {
    Add,
    Or,
    And,
    Sub,
    Xor,
    Cmp,
    Test,
}

/// Mask `v` to `width` bits (8/16/32/64); a no-op at `width == 64`.
const fn mask_w(v: u64, width: u32) -> u64 {
    match width {
        8 => v & 0xff,
        16 => v & 0xffff,
        32 => v & 0xffff_ffff,
        _ => v,
    }
}

/// Sign-extend the low `bits` of `v` (`bits` in `1..=128`) to a full `i128`.
const fn sign_extend_128(v: u128, bits: u32) -> i128 {
    let shift = 128 - bits;
    ((v << shift) as i128) >> shift
}

/// Does signed `v` fit in a `width`-bit two's-complement integer?
const fn fits_signed(v: i128, width: u32) -> bool {
    let max = (1i128 << (width - 1)) - 1;
    let min = -(1i128 << (width - 1));
    v >= min && v <= max
}

/// Does unsigned `v` fit in a `width`-bit integer?
const fn fits_unsigned(v: u128, width: u32) -> bool {
    v < (1u128 << width)
}

/// Parity flag: `true` iff the low byte of `v` has an even number of 1 bits.
fn parity(v: u8) -> bool {
    v.count_ones().is_multiple_of(2)
}

/// The sign bit of `v` interpreted as a `width`-bit integer.
fn sign_bit(v: u64, width: u32) -> bool {
    (v >> (width - 1)) & 1 == 1
}

/// Sign-extend the low `width` bits of `v` (`width` in `1..=64`) to a full
/// 64-bit signed value.
const fn sign_extend_w(v: u64, width: u32) -> i64 {
    if width >= 64 {
        v as i64
    } else {
        let shift = 64 - width;
        ((v << shift) as i64) >> shift
    }
}

// ---- instruction-stream fetch helpers: thread `pc` through as a plain value
// so the borrow checker never has to reason about a `Fetcher` struct holding
// a live reference into `mem` across a later `&mut` use. ----

fn fetch_u8(mem: &GuestMemory, pc: u64) -> Result<(u8, u64), Step> {
    let mut b = [0u8; 1];
    mem.read(pc, &mut b)
        .map_err(|_| Step::Fault { addr: pc, write: false })?;
    Ok((b[0], pc + 1))
}

fn fetch_i8(mem: &GuestMemory, pc: u64) -> Result<(i8, u64), Step> {
    let (b, next) = fetch_u8(mem, pc)?;
    Ok((b as i8, next))
}

fn fetch_u16(mem: &GuestMemory, pc: u64) -> Result<(u16, u64), Step> {
    let mut b = [0u8; 2];
    mem.read(pc, &mut b)
        .map_err(|_| Step::Fault { addr: pc, write: false })?;
    Ok((u16::from_le_bytes(b), pc + 2))
}

fn fetch_i16(mem: &GuestMemory, pc: u64) -> Result<(i16, u64), Step> {
    let (v, next) = fetch_u16(mem, pc)?;
    Ok((v as i16, next))
}

fn fetch_u32(mem: &GuestMemory, pc: u64) -> Result<(u32, u64), Step> {
    let mut b = [0u8; 4];
    mem.read(pc, &mut b)
        .map_err(|_| Step::Fault { addr: pc, write: false })?;
    Ok((u32::from_le_bytes(b), pc + 4))
}

fn fetch_i32(mem: &GuestMemory, pc: u64) -> Result<(i32, u64), Step> {
    let (v, next) = fetch_u32(mem, pc)?;
    Ok((v as i32, next))
}

fn fetch_u64(mem: &GuestMemory, pc: u64) -> Result<(u64, u64), Step> {
    let mut b = [0u8; 8];
    mem.read(pc, &mut b)
        .map_err(|_| Step::Fault { addr: pc, write: false })?;
    Ok((u64::from_le_bytes(b), pc + 8))
}

/// Fetch an immediate sized to `width` the way the `0x81`/`0xF7`-family
/// opcodes do: `imm8` at 8-bit width, `imm16` at 16-bit width, otherwise a
/// sign-extended `imm32` (there is no `imm64` immediate form in x86-64).
fn imm_for_width(mem: &GuestMemory, pc: u64, width: u32) -> Result<(i64, u64), Step> {
    match width {
        8 => {
            let (v, p) = fetch_i8(mem, pc)?;
            Ok((i64::from(v), p))
        }
        16 => {
            let (v, p) = fetch_i16(mem, pc)?;
            Ok((i64::from(v), p))
        }
        _ => {
            let (v, p) = fetch_i32(mem, pc)?;
            Ok((i64::from(v), p))
        }
    }
}

/// Bail out of the enclosing `Step`-returning function on fetch/decode
/// failure, otherwise unwrap the `Ok` value. (`Step` isn't `Result`, so `?`
/// doesn't apply — this is the equivalent for our fetch/decode helpers.)
macro_rules! fetch {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(s) => return s,
        }
    };
}

/// A user-mode x86-64 interpreter.
#[derive(Clone)]
struct X86Interp {
    /// rax..r15, in the standard ModRM/REX numbering.
    gpr: [u64; 16],
    rip: u64,
    flags: Flags,
    /// The direction flag: `false` (`CLD`) advances string-op pointers
    /// upward, `true` (`STD`) advances them downward.
    df: bool,
    /// FS.base, set by `arch_prctl(ARCH_SET_FS, ...)` (thread pointer).
    fs_base: u64,
}

impl X86Interp {
    fn new(entry: u64, stack: u64) -> Self {
        let mut gpr = [0u64; 16];
        gpr[RSP] = stack;
        Self {
            gpr,
            rip: entry,
            flags: Flags::default(),
            df: false,
            fs_base: 0,
        }
    }

    fn next(&mut self, pc: u64) -> Step {
        self.rip = pc;
        Step::Next
    }

    fn jump(&mut self, target: u64) -> Step {
        self.rip = target;
        Step::Branched
    }

    fn push(&mut self, mem: &mut GuestMemory, val: u64) -> Result<(), Step> {
        let sp = self.gpr[RSP].wrapping_sub(8);
        mem.write(sp, &val.to_le_bytes())
            .map_err(|_| Step::Fault { addr: sp, write: true })?;
        self.gpr[RSP] = sp;
        Ok(())
    }

    fn pop(&mut self, mem: &GuestMemory) -> Result<u64, Step> {
        let sp = self.gpr[RSP];
        let mut b = [0u8; 8];
        mem.read(sp, &mut b)
            .map_err(|_| Step::Fault { addr: sp, write: false })?;
        self.gpr[RSP] = sp.wrapping_add(8);
        Ok(u64::from_le_bytes(b))
    }

    /// Decode a ModRM byte (and any SIB/displacement that follows it).
    /// Memory addresses that don't need the end-of-instruction address are
    /// resolved immediately; RIP-relative ones are deferred (see [`RmKind`]).
    fn decode_modrm(&self, mem: &GuestMemory, pc: u64, rex: Rex) -> Result<(ModRm, u64), Step> {
        let (byte, pc) = fetch_u8(mem, pc)?;
        let md = byte >> 6;
        let reg = usize::from((byte >> 3) & 7) | (usize::from(rex.r) << 3);
        let rm_field = byte & 7;

        if md == 0b11 {
            let rm = usize::from(rm_field) | (usize::from(rex.b) << 3);
            return Ok((ModRm { reg, kind: RmKind::Reg(rm) }, pc));
        }

        if rm_field == 0b100 {
            // SIB byte follows.
            let (sib, pc) = fetch_u8(mem, pc)?;
            let scale = 1u64 << (sib >> 6);
            let idx_field = (sib >> 3) & 7;
            let base_field = sib & 7;
            // index field == 0b100 (before REX.X extension) means "no index";
            // REX.X can turn it into r12, which *is* usable as an index.
            let index = if idx_field == 0b100 && !rex.x {
                None
            } else {
                Some(usize::from(idx_field) | (usize::from(rex.x) << 3))
            };
            let (base, disp, pc) = if base_field == 0b101 && md == 0b00 {
                let (d, pc) = fetch_i32(mem, pc)?;
                (None, i64::from(d), pc)
            } else {
                let b = usize::from(base_field) | (usize::from(rex.b) << 3);
                match md {
                    0b01 => {
                        let (d, pc) = fetch_i8(mem, pc)?;
                        (Some(b), i64::from(d), pc)
                    }
                    0b10 => {
                        let (d, pc) = fetch_i32(mem, pc)?;
                        (Some(b), i64::from(d), pc)
                    }
                    _ => (Some(b), 0i64, pc),
                }
            };
            let base_val = base.map_or(0, |b| self.gpr[b]);
            let index_val = index.map_or(0, |i| self.gpr[i]);
            let addr =
                (base_val.wrapping_add(index_val.wrapping_mul(scale)) as i64).wrapping_add(disp)
                    as u64;
            return Ok((ModRm { reg, kind: RmKind::Mem(addr) }, pc));
        }

        if rm_field == 0b101 && md == 0b00 {
            let (disp, pc) = fetch_i32(mem, pc)?;
            return Ok((ModRm { reg, kind: RmKind::MemRip(i64::from(disp)) }, pc));
        }

        let base = usize::from(rm_field) | (usize::from(rex.b) << 3);
        let (disp, pc) = match md {
            0b01 => {
                let (d, pc) = fetch_i8(mem, pc)?;
                (i64::from(d), pc)
            }
            0b10 => {
                let (d, pc) = fetch_i32(mem, pc)?;
                (i64::from(d), pc)
            }
            _ => (0i64, pc),
        };
        let addr = (self.gpr[base] as i64).wrapping_add(disp) as u64;
        Ok((ModRm { reg, kind: RmKind::Mem(addr) }, pc))
    }

    fn read_operand(&self, mem: &GuestMemory, op: Operand, width: u32) -> Result<u64, Step> {
        match op {
            Operand::Reg(r) => Ok(mask_w(self.gpr[r], width)),
            Operand::Reg8Hi(r) => Ok((self.gpr[r] >> 8) & 0xff),
            Operand::Mem(a) => {
                let n = (width / 8) as usize;
                let mut b = [0u8; 8];
                mem.read(a, &mut b[..n])
                    .map_err(|_| Step::Fault { addr: a, write: false })?;
                Ok(u64::from_le_bytes(b))
            }
        }
    }

    /// Write `val` (masked to `width`) into `op`. Register writes follow x86
    /// partial-write semantics: an 8/16-bit write preserves the untouched
    /// bits of the full 64-bit register, while a 32-bit write zero-extends
    /// (the standard "writing `eax` clears the top half of `rax`" rule) and a
    /// 64-bit write replaces it outright.
    fn write_operand(
        &mut self,
        mem: &mut GuestMemory,
        op: Operand,
        val: u64,
        width: u32,
    ) -> Result<(), Step> {
        match op {
            Operand::Reg(r) => {
                self.gpr[r] = match width {
                    8 => (self.gpr[r] & !0xffu64) | (val & 0xff),
                    16 => (self.gpr[r] & !0xffffu64) | (val & 0xffff),
                    _ => mask_w(val, width),
                };
                Ok(())
            }
            Operand::Reg8Hi(r) => {
                self.gpr[r] = (self.gpr[r] & !0xff00u64) | ((val & 0xff) << 8);
                Ok(())
            }
            Operand::Mem(a) => {
                let n = (width / 8) as usize;
                let bytes = val.to_le_bytes();
                mem.write(a, &bytes[..n])
                    .map_err(|_| Step::Fault { addr: a, write: true })
            }
        }
    }

    // ---- flags ----

    fn add_flags(&mut self, a: u64, b: u64, width: u32) -> u64 {
        if width == 64 {
            let (r, cf) = a.overflowing_add(b);
            self.flags = Flags {
                cf,
                zf: r == 0,
                sf: sign_bit(r, 64),
                of: (((a ^ r) & (b ^ r)) >> 63) & 1 == 1,
                pf: parity(r as u8),
            };
            r
        } else {
            let (a32, b32) = (a as u32, b as u32);
            let (r, cf) = a32.overflowing_add(b32);
            self.flags = Flags {
                cf,
                zf: r == 0,
                sf: sign_bit(u64::from(r), 32),
                of: (((a32 ^ r) & (b32 ^ r)) >> 31) & 1 == 1,
                pf: parity(r as u8),
            };
            u64::from(r)
        }
    }

    fn sub_flags(&mut self, a: u64, b: u64, width: u32) -> u64 {
        if width == 64 {
            let r = a.wrapping_sub(b);
            self.flags = Flags {
                cf: a < b,
                zf: r == 0,
                sf: sign_bit(r, 64),
                of: (((a ^ b) & (a ^ r)) >> 63) & 1 == 1,
                pf: parity(r as u8),
            };
            r
        } else {
            let (a32, b32) = (a as u32, b as u32);
            let r = a32.wrapping_sub(b32);
            self.flags = Flags {
                cf: a32 < b32,
                zf: r == 0,
                sf: sign_bit(u64::from(r), 32),
                of: (((a32 ^ b32) & (a32 ^ r)) >> 31) & 1 == 1,
                pf: parity(r as u8),
            };
            u64::from(r)
        }
    }

    fn logic_flags(&mut self, r: u64, width: u32) -> u64 {
        let r = mask_w(r, width);
        self.flags = Flags {
            cf: false,
            of: false,
            zf: r == 0,
            sf: sign_bit(r, width),
            pf: parity(r as u8),
        };
        r
    }

    fn apply_alu(&mut self, op: AluOp, a: u64, b: u64, width: u32) -> u64 {
        match op {
            AluOp::Add => self.add_flags(a, b, width),
            AluOp::Sub | AluOp::Cmp => self.sub_flags(a, b, width),
            AluOp::And | AluOp::Test => self.logic_flags(a & b, width),
            AluOp::Or => self.logic_flags(a | b, width),
            AluOp::Xor => self.logic_flags(a ^ b, width),
        }
    }

    /// `INC`/`DEC`: like `ADD`/`SUB` by 1, but CF is left untouched (an x86
    /// quirk, since `INC`/`DEC` must not disturb a carry chain).
    fn inc_dec_flags(&mut self, a: u64, sub: bool, width: u32) -> u64 {
        let saved_cf = self.flags.cf;
        let r = if sub {
            self.sub_flags(a, 1, width)
        } else {
            self.add_flags(a, 1, width)
        };
        self.flags.cf = saved_cf;
        r
    }

    fn shl_flags(&mut self, a: u64, amt: u8, width: u32) -> u64 {
        let a = mask_w(a, width);
        let amtu = u32::from(amt);
        let cf = (a >> (width - amtu)) & 1 == 1;
        let r = mask_w(a << amtu, width);
        self.flags.cf = cf;
        self.flags.zf = r == 0;
        self.flags.sf = sign_bit(r, width);
        self.flags.pf = parity(r as u8);
        if amt == 1 {
            self.flags.of = sign_bit(r, width) != cf;
        }
        r
    }

    fn shr_flags(&mut self, a: u64, amt: u8, width: u32) -> u64 {
        let a = mask_w(a, width);
        let amtu = u32::from(amt);
        let cf = (a >> (amtu - 1)) & 1 == 1;
        let r = mask_w(a >> amtu, width);
        self.flags.cf = cf;
        self.flags.zf = r == 0;
        self.flags.sf = sign_bit(r, width);
        self.flags.pf = parity(r as u8);
        if amt == 1 {
            self.flags.of = sign_bit(a, width);
        }
        r
    }

    fn sar_flags(&mut self, a: u64, amt: u8, width: u32) -> u64 {
        let amtu = u32::from(amt);
        let signed = sign_extend_w(mask_w(a, width), width);
        let cf = ((signed >> (amtu - 1)) & 1) == 1;
        let r = mask_w((signed >> amtu) as u64, width);
        self.flags.cf = cf;
        self.flags.zf = r == 0;
        self.flags.sf = sign_bit(r, width);
        self.flags.pf = parity(r as u8);
        if amt == 1 {
            self.flags.of = false;
        }
        r
    }

    fn cond_holds(&self, cc: u8) -> bool {
        let f = &self.flags;
        match cc & 0xf {
            0x0 => f.of,
            0x1 => !f.of,
            0x2 => f.cf,
            0x3 => !f.cf,
            0x4 => f.zf,
            0x5 => !f.zf,
            0x6 => f.cf || f.zf,
            0x7 => !f.cf && !f.zf,
            0x8 => f.sf,
            0x9 => !f.sf,
            0xA => f.pf,
            0xB => !f.pf,
            0xC => f.sf != f.of,
            0xD => f.sf == f.of,
            0xE => f.zf || (f.sf != f.of),
            _ => !f.zf && (f.sf == f.of), // 0xF
        }
    }

    // ---- instruction groups ----

    fn lea(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex, width: u32) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let addr = match resolve(modrm.kind, pc2) {
            Operand::Mem(a) => a,
            Operand::Reg(_) | Operand::Reg8Hi(_) => return Step::Illegal, // LEA requires a memory r/m
        };
        self.gpr[modrm.reg] = mask_w(addr, width);
        self.next(pc2)
    }

    fn mov_rm_gv(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex, width: u32) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let val = mask_w(self.gpr[modrm.reg], width);
        fetch!(self.write_operand(mem, rm_op, val, width));
        self.next(pc2)
    }

    fn mov_gv_rm(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex, width: u32) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let val = fetch!(self.read_operand(mem, rm_op, width));
        self.gpr[modrm.reg] = mask_w(val, width);
        self.next(pc2)
    }

    fn mov_imm(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex, width: u32) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        if modrm.reg != 0 {
            return Step::Illegal; // 0xC7 /0 only
        }
        let (imm, pc3) = fetch!(fetch_i32(mem, pc2));
        let rm_op = resolve(modrm.kind, pc3);
        let val = mask_w(i64::from(imm) as u64, width);
        fetch!(self.write_operand(mem, rm_op, val, width));
        self.next(pc3)
    }

    /// `op r/m, reg` (`Ev,Gv` encoding: destination is the r/m operand).
    fn alu_rm_gv(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        rex: Rex,
        width: u32,
        op: AluOp,
        store: bool,
    ) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let a = fetch!(self.read_operand(mem, rm_op, width));
        let b = mask_w(self.gpr[modrm.reg], width);
        let r = self.apply_alu(op, a, b, width);
        if store {
            fetch!(self.write_operand(mem, rm_op, r, width));
        }
        self.next(pc2)
    }

    /// `op reg, r/m` (`Gv,Ev` encoding: destination is the reg operand).
    fn alu_gv_rm(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex, width: u32, op: AluOp) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let b = fetch!(self.read_operand(mem, rm_op, width));
        let a = mask_w(self.gpr[modrm.reg], width);
        let r = self.apply_alu(op, a, b, width);
        if op != AluOp::Cmp {
            self.gpr[modrm.reg] = mask_w(r, width);
        }
        self.next(pc2)
    }

    /// Group 1: `0x81 /r id`, `0x83 /r ib` (16/32/64-bit r/m) and `0x80 /r ib`
    /// (8-bit r/m) — ALU op, r/m and an immediate.
    fn group1_imm(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        rex: Rex,
        has_rex: bool,
        width: u32,
        imm8: bool,
    ) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let (imm, pc3): (i64, u64) = if imm8 {
            let (v, p) = fetch!(fetch_i8(mem, pc2));
            (i64::from(v), p)
        } else {
            fetch!(imm_for_width(mem, pc2, width))
        };
        let op = match modrm.reg {
            0 => AluOp::Add,
            1 => AluOp::Or,
            4 => AluOp::And,
            5 => AluOp::Sub,
            6 => AluOp::Xor,
            7 => AluOp::Cmp,
            _ => return Step::Illegal, // ADC/SBB (2,3): not in our documented subset
        };
        let rm_op = if width == 8 {
            resolve8(modrm.kind, pc3, has_rex)
        } else {
            resolve(modrm.kind, pc3)
        };
        let a = fetch!(self.read_operand(mem, rm_op, width));
        let b = mask_w(imm as u64, width);
        let r = self.apply_alu(op, a, b, width);
        if op != AluOp::Cmp {
            fetch!(self.write_operand(mem, rm_op, r, width));
        }
        self.next(pc3)
    }

    /// `op r/m8, r8` (`Eb,Gb` encoding: destination is the r/m operand).
    fn alu_rm_gv8(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        rex: Rex,
        has_rex: bool,
        op: AluOp,
        store: bool,
    ) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve8(modrm.kind, pc2, has_rex);
        let reg_op = reg8_operand(modrm.reg, has_rex);
        let a = fetch!(self.read_operand(mem, rm_op, 8));
        let b = fetch!(self.read_operand(mem, reg_op, 8));
        let r = self.apply_alu(op, a, b, 8);
        if store {
            fetch!(self.write_operand(mem, rm_op, r, 8));
        }
        self.next(pc2)
    }

    /// `op r8, r/m8` (`Gb,Eb` encoding: destination is the reg operand).
    fn alu_gv_rm8(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        rex: Rex,
        has_rex: bool,
        op: AluOp,
    ) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve8(modrm.kind, pc2, has_rex);
        let reg_op = reg8_operand(modrm.reg, has_rex);
        let b = fetch!(self.read_operand(mem, rm_op, 8));
        let a = fetch!(self.read_operand(mem, reg_op, 8));
        let r = self.apply_alu(op, a, b, 8);
        if op != AluOp::Cmp {
            fetch!(self.write_operand(mem, reg_op, r, 8));
        }
        self.next(pc2)
    }

    /// `XCHG r/m, reg` (`0x86`/`0x87`) — swap the two operands' contents.
    fn xchg(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex, has_rex: bool, width: u32) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let (rm_op, reg_op) = if width == 8 {
            (resolve8(modrm.kind, pc2, has_rex), reg8_operand(modrm.reg, has_rex))
        } else {
            (resolve(modrm.kind, pc2), Operand::Reg(modrm.reg))
        };
        let a = fetch!(self.read_operand(mem, rm_op, width));
        let b = fetch!(self.read_operand(mem, reg_op, width));
        fetch!(self.write_operand(mem, rm_op, b, width));
        fetch!(self.write_operand(mem, reg_op, a, width));
        self.next(pc2)
    }

    /// Group 3: `0xF6`/`0xF7 /r` — `TEST r/m, imm` (/0, /1), `NOT r/m` (/2),
    /// `NEG r/m` (/3), `MUL r/m` (/4), `IMUL r/m` (/5, one-operand form),
    /// `DIV r/m` (/6) and `IDIV r/m` (/7). `0xF6` selects an 8-bit r/m;
    /// `0xF7` uses `width`.
    fn group3(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex, has_rex: bool, width: u32) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op_at = |end_pc| {
            if width == 8 {
                resolve8(modrm.kind, end_pc, has_rex)
            } else {
                resolve(modrm.kind, end_pc)
            }
        };
        match modrm.reg {
            0 | 1 => {
                let (imm, pc3) = fetch!(imm_for_width(mem, pc2, width));
                let rm_op = rm_op_at(pc3);
                let a = fetch!(self.read_operand(mem, rm_op, width));
                self.apply_alu(AluOp::Test, a, mask_w(imm as u64, width), width);
                self.next(pc3)
            }
            2 => {
                let rm_op = rm_op_at(pc2);
                let a = fetch!(self.read_operand(mem, rm_op, width));
                let r = mask_w(!a, width);
                fetch!(self.write_operand(mem, rm_op, r, width));
                self.next(pc2)
            }
            3 => {
                let rm_op = rm_op_at(pc2);
                let a = fetch!(self.read_operand(mem, rm_op, width));
                let r = self.sub_flags(0, a, width); // NEG = 0 - a; CF = (a != 0)
                fetch!(self.write_operand(mem, rm_op, r, width));
                self.next(pc2)
            }
            4 => self.mul_op(mem, rm_op_at(pc2), width, false, pc2),
            5 => self.mul_op(mem, rm_op_at(pc2), width, true, pc2),
            6 => self.div_op(mem, rm_op_at(pc2), width, false, pc2),
            _ => self.div_op(mem, rm_op_at(pc2), width, true, pc2), // 7 = IDIV
        }
    }

    /// Group 4: `0xFE /r` — `INC r/m8` (/0) and `DEC r/m8` (/1).
    fn group4(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex, has_rex: bool) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        match modrm.reg {
            0 | 1 => {
                let rm_op = resolve8(modrm.kind, pc2, has_rex);
                let a = fetch!(self.read_operand(mem, rm_op, 8));
                let r = self.inc_dec_flags(a, modrm.reg == 1, 8);
                fetch!(self.write_operand(mem, rm_op, r, 8));
                self.next(pc2)
            }
            _ => Step::Illegal,
        }
    }

    /// Group 5: `0xFF /r` — `INC r/m` (/0), `DEC r/m` (/1), `CALL r/m` (/2,
    /// near indirect), `JMP r/m` (/4, near indirect) and `PUSH r/m` (/6).
    /// `CALL`/`JMP`/`PUSH r/m` always use a 64-bit operand, matching the
    /// default (REX.W-independent) operand size these forms have in long
    /// mode.
    fn group5(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex, width: u32) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        match modrm.reg {
            0 | 1 => {
                let rm_op = resolve(modrm.kind, pc2);
                let a = fetch!(self.read_operand(mem, rm_op, width));
                let r = self.inc_dec_flags(a, modrm.reg == 1, width);
                fetch!(self.write_operand(mem, rm_op, r, width));
                self.next(pc2)
            }
            2 => {
                let rm_op = resolve(modrm.kind, pc2);
                let target = fetch!(self.read_operand(mem, rm_op, 64));
                fetch!(self.push(mem, pc2));
                self.jump(target)
            }
            4 => {
                let rm_op = resolve(modrm.kind, pc2);
                let target = fetch!(self.read_operand(mem, rm_op, 64));
                self.jump(target)
            }
            6 => {
                let rm_op = resolve(modrm.kind, pc2);
                let val = fetch!(self.read_operand(mem, rm_op, 64));
                fetch!(self.push(mem, val));
                self.next(pc2)
            }
            _ => Step::Illegal, // CALL far / JMP far (3,5): not in our documented subset
        }
    }

    /// Write a double-`width` product `p` (already reinterpreted as the
    /// unsigned bit pattern of the true signed or unsigned result) into the
    /// `MUL`/`IMUL` (one-operand) destination pair: `AX` for an 8-bit
    /// operand, otherwise `rDX:rAX`.
    fn write_wide_result(&mut self, width: u32, p: u128) {
        let lo = p as u64;
        let hi = (p >> width) as u64;
        match width {
            8 => self.gpr[RAX] = (self.gpr[RAX] & !0xffffu64) | (lo & 0xffff),
            16 => {
                self.gpr[RAX] = (self.gpr[RAX] & !0xffffu64) | (lo & 0xffff);
                self.gpr[RDX] = (self.gpr[RDX] & !0xffffu64) | (hi & 0xffff);
            }
            32 => {
                self.gpr[RAX] = lo & 0xffff_ffff;
                self.gpr[RDX] = hi & 0xffff_ffff;
            }
            _ => {
                self.gpr[RAX] = lo;
                self.gpr[RDX] = hi;
            }
        }
    }

    /// `MUL`/`IMUL` one-operand form: `rDX:rAX` (or just `AX` at 8-bit width)
    /// = `rAX` * `r/m`. Only `CF`/`OF` are defined by the ISA for this form;
    /// `ZF`/`SF`/`PF` are left untouched.
    fn mul_op(
        &mut self,
        mem: &GuestMemory,
        rm_op: Operand,
        width: u32,
        signed: bool,
        pc2: u64,
    ) -> Step {
        let src = fetch!(self.read_operand(mem, rm_op, width));
        let a = mask_w(self.gpr[RAX], width);
        let cf = if signed {
            let av = sign_extend_128(u128::from(a), width);
            let bv = sign_extend_128(u128::from(src), width);
            let p = av * bv;
            self.write_wide_result(width, p as u128);
            !fits_signed(p, width)
        } else {
            let p = u128::from(a) * u128::from(src);
            self.write_wide_result(width, p);
            !fits_unsigned(p, width)
        };
        self.flags.cf = cf;
        self.flags.of = cf;
        self.next(pc2)
    }

    /// Write the `width`-bit quotient/remainder pair from `DIV`/`IDIV`: `AL`/
    /// `AH` for an 8-bit divisor, otherwise `rAX`/`rDX`.
    fn write_div_result(&mut self, width: u32, q: u64, r: u64) {
        match width {
            8 => self.gpr[RAX] = (self.gpr[RAX] & !0xffffu64) | (q & 0xff) | ((r & 0xff) << 8),
            16 => {
                self.gpr[RAX] = (self.gpr[RAX] & !0xffffu64) | (q & 0xffff);
                self.gpr[RDX] = (self.gpr[RDX] & !0xffffu64) | (r & 0xffff);
            }
            32 => {
                self.gpr[RAX] = q & 0xffff_ffff;
                self.gpr[RDX] = r & 0xffff_ffff;
            }
            _ => {
                self.gpr[RAX] = q;
                self.gpr[RDX] = r;
            }
        }
    }

    /// `DIV`/`IDIV` one-operand form: the `2*width`-bit dividend in
    /// `rDX:rAX` (or `AX` at 8-bit width) is divided by `r/m`, leaving the
    /// quotient in `rAX`/`AL` and the remainder in `rDX`/`AH`. A zero divisor
    /// or an out-of-range quotient is a `#DE` on real hardware; we surface
    /// both as [`Step::Illegal`] rather than panicking on the division.
    fn div_op(
        &mut self,
        mem: &GuestMemory,
        rm_op: Operand,
        width: u32,
        signed: bool,
        pc2: u64,
    ) -> Step {
        let divisor = fetch!(self.read_operand(mem, rm_op, width));
        let bits = width * 2;
        let dividend: u128 = match width {
            8 => u128::from(self.gpr[RAX] & 0xffff),
            16 => u128::from(((self.gpr[RDX] & 0xffff) << 16) | (self.gpr[RAX] & 0xffff)),
            32 => u128::from(((self.gpr[RDX] & 0xffff_ffff) << 32) | (self.gpr[RAX] & 0xffff_ffff)),
            _ => (u128::from(self.gpr[RDX]) << 64) | u128::from(self.gpr[RAX]),
        };
        if signed {
            let dividend_s = sign_extend_128(dividend, bits);
            let divisor_s = sign_extend_128(u128::from(divisor), width);
            if divisor_s == 0 {
                return Step::Illegal;
            }
            let q = dividend_s / divisor_s;
            let r = dividend_s % divisor_s;
            if !fits_signed(q, width) {
                return Step::Illegal;
            }
            self.write_div_result(width, q as u64, r as u64);
        } else {
            let divisor_u = u128::from(divisor);
            if divisor_u == 0 {
                return Step::Illegal;
            }
            let q = dividend / divisor_u;
            let r = dividend % divisor_u;
            if !fits_unsigned(q, width) {
                return Step::Illegal;
            }
            self.write_div_result(width, q as u64, r as u64);
        }
        self.next(pc2)
    }

    /// `IMUL r, r/m, imm` (`0x69` imm32/imm16, `0x6B` imm8): `reg` = `r/m` *
    /// sign-extended immediate.
    fn imul_imm(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex, width: u32, imm8: bool) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let (imm, pc3): (i64, u64) = if imm8 {
            let (v, p) = fetch!(fetch_i8(mem, pc2));
            (i64::from(v), p)
        } else {
            fetch!(imm_for_width(mem, pc2, width))
        };
        let rm_op = resolve(modrm.kind, pc3);
        let b = fetch!(self.read_operand(mem, rm_op, width));
        let av = sign_extend_128(u128::from(b), width);
        let bv = i128::from(imm);
        let p = av * bv;
        let cf = !fits_signed(p, width);
        self.gpr[modrm.reg] = mask_w(p as u128 as u64, width);
        self.flags.cf = cf;
        self.flags.of = cf;
        self.next(pc3)
    }

    /// Group 2 shifts: `0xC1 /r ib` (by immediate) and `0xD3 /r` (by `CL`).
    fn group2(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        rex: Rex,
        width: u32,
        by_cl: bool,
    ) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        if !matches!(modrm.reg, 4..=7) {
            return Step::Illegal; // ROL/ROR/RCL/RCR: not in our documented subset
        }
        let (count, pc3) = if by_cl {
            (self.gpr[RCX] as u8, pc2)
        } else {
            fetch!(fetch_u8(mem, pc2))
        };
        let mask = if width == 64 { 63 } else { 31 };
        let amt = count & mask;
        let rm_op = resolve(modrm.kind, pc3);
        if amt == 0 {
            return self.next(pc3); // shift by 0 leaves flags and value unchanged
        }
        let a = fetch!(self.read_operand(mem, rm_op, width));
        let r = match modrm.reg {
            4 | 6 => self.shl_flags(a, amt, width), // SHL and its SAL alias
            5 => self.shr_flags(a, amt, width),
            _ => self.sar_flags(a, amt, width),
        };
        fetch!(self.write_operand(mem, rm_op, r, width));
        self.next(pc3)
    }

    // ---- REP-prefixed string ops. `rep` is `0` (no prefix, run once and
    // leave rCX alone), `1` (REP/REPE, `0xF3`) or `2` (REPNE, `0xF2`). Each
    // handler runs its whole repeat count in a single `Step` rather than
    // yielding to the caller between iterations — real hardware is
    // interruptible mid-string, but nothing here needs that granularity. ----

    /// `MOVS` (`0xA4`/`0xA5`): copy `[rSI]` to `[rDI]`, advancing both by
    /// `width` bytes per `DF`.
    fn movs(&mut self, mem: &mut GuestMemory, pc: u64, width: u32, rep: u8) -> Step {
        let step = u64::from(width / 8);
        let n = (width / 8) as usize;
        let mut count: u64 = if rep == 0 { 1 } else { self.gpr[RCX] };
        while count > 0 {
            let mut b = [0u8; 8];
            fetch!(mem
                .read(self.gpr[RSI], &mut b[..n])
                .map_err(|_| Step::Fault { addr: self.gpr[RSI], write: false }));
            fetch!(mem
                .write(self.gpr[RDI], &b[..n])
                .map_err(|_| Step::Fault { addr: self.gpr[RDI], write: true }));
            self.gpr[RSI] = self.advance_ptr(self.gpr[RSI], step);
            self.gpr[RDI] = self.advance_ptr(self.gpr[RDI], step);
            count -= 1;
            if rep != 0 {
                self.gpr[RCX] = count;
            }
        }
        self.next(pc)
    }

    /// `STOS` (`0xAA`/`0xAB`): store `AL`/`rAX` to `[rDI]`, advancing by
    /// `width` bytes per `DF`.
    fn stos(&mut self, mem: &mut GuestMemory, pc: u64, width: u32, rep: u8) -> Step {
        let step = u64::from(width / 8);
        let n = (width / 8) as usize;
        let val = mask_w(self.gpr[RAX], width);
        let mut count: u64 = if rep == 0 { 1 } else { self.gpr[RCX] };
        while count > 0 {
            let bytes = val.to_le_bytes();
            fetch!(mem
                .write(self.gpr[RDI], &bytes[..n])
                .map_err(|_| Step::Fault { addr: self.gpr[RDI], write: true }));
            self.gpr[RDI] = self.advance_ptr(self.gpr[RDI], step);
            count -= 1;
            if rep != 0 {
                self.gpr[RCX] = count;
            }
        }
        self.next(pc)
    }

    /// `LODS` (`0xAC`/`0xAD`): load `[rSI]` into `AL`/`rAX`, advancing by
    /// `width` bytes per `DF`.
    fn lods(&mut self, mem: &mut GuestMemory, pc: u64, width: u32, rep: u8) -> Step {
        let step = u64::from(width / 8);
        let n = (width / 8) as usize;
        let mut count: u64 = if rep == 0 { 1 } else { self.gpr[RCX] };
        while count > 0 {
            let mut b = [0u8; 8];
            fetch!(mem
                .read(self.gpr[RSI], &mut b[..n])
                .map_err(|_| Step::Fault { addr: self.gpr[RSI], write: false }));
            let v = u64::from_le_bytes(b);
            fetch!(self.write_operand(mem, Operand::Reg(RAX), v, width));
            self.gpr[RSI] = self.advance_ptr(self.gpr[RSI], step);
            count -= 1;
            if rep != 0 {
                self.gpr[RCX] = count;
            }
        }
        self.next(pc)
    }

    /// `SCAS` (`0xAE`/`0xAF`): compare `AL`/`rAX` against `[rDI]`, advancing
    /// by `width` bytes per `DF`; `REPE`/`REPNE` stop early on a `ZF`
    /// mismatch.
    fn scas(&mut self, mem: &mut GuestMemory, pc: u64, width: u32, rep: u8) -> Step {
        let step = u64::from(width / 8);
        let n = (width / 8) as usize;
        let a = mask_w(self.gpr[RAX], width);
        let mut count: u64 = if rep == 0 { 1 } else { self.gpr[RCX] };
        while count > 0 {
            let mut b = [0u8; 8];
            fetch!(mem
                .read(self.gpr[RDI], &mut b[..n])
                .map_err(|_| Step::Fault { addr: self.gpr[RDI], write: false }));
            self.sub_flags(a, mask_w(u64::from_le_bytes(b), width), width);
            self.gpr[RDI] = self.advance_ptr(self.gpr[RDI], step);
            count -= 1;
            if rep != 0 {
                self.gpr[RCX] = count;
            }
            if !self.rep_continues(rep, count) {
                break;
            }
        }
        self.next(pc)
    }

    /// `CMPS` (`0xA6`/`0xA7`): compare `[rSI]` against `[rDI]`, advancing
    /// both by `width` bytes per `DF`; `REPE`/`REPNE` stop early on a `ZF`
    /// mismatch.
    fn cmps(&mut self, mem: &mut GuestMemory, pc: u64, width: u32, rep: u8) -> Step {
        let step = u64::from(width / 8);
        let n = (width / 8) as usize;
        let mut count: u64 = if rep == 0 { 1 } else { self.gpr[RCX] };
        while count > 0 {
            let (mut bs, mut bd) = ([0u8; 8], [0u8; 8]);
            fetch!(mem
                .read(self.gpr[RSI], &mut bs[..n])
                .map_err(|_| Step::Fault { addr: self.gpr[RSI], write: false }));
            fetch!(mem
                .read(self.gpr[RDI], &mut bd[..n])
                .map_err(|_| Step::Fault { addr: self.gpr[RDI], write: false }));
            let (vs, vd) = (u64::from_le_bytes(bs), u64::from_le_bytes(bd));
            self.sub_flags(mask_w(vs, width), mask_w(vd, width), width);
            self.gpr[RSI] = self.advance_ptr(self.gpr[RSI], step);
            self.gpr[RDI] = self.advance_ptr(self.gpr[RDI], step);
            count -= 1;
            if rep != 0 {
                self.gpr[RCX] = count;
            }
            if !self.rep_continues(rep, count) {
                break;
            }
        }
        self.next(pc)
    }

    /// Advance a string-op pointer by `step` bytes, per `DF`.
    fn advance_ptr(&self, ptr: u64, step: u64) -> u64 {
        if self.df { ptr.wrapping_sub(step) } else { ptr.wrapping_add(step) }
    }

    /// Should a `REPE`/`REPNE`-prefixed `SCAS`/`CMPS` loop keep going after
    /// this iteration? `rep == 1` (`REPE`) continues while `ZF` is set;
    /// `rep == 2` (`REPNE`) continues while it's clear; `rep == 0` (no
    /// prefix, single iteration) and `count == 0` always stop.
    fn rep_continues(&self, rep: u8, count: u64) -> bool {
        if count == 0 {
            return false;
        }
        match rep {
            1 => self.flags.zf,
            2 => !self.flags.zf,
            _ => true,
        }
    }

    #[allow(clippy::too_many_lines)]
    fn exec_0f(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        rex: Rex,
        has_rex: bool,
        width: u32,
    ) -> Step {
        let (op2, pc) = fetch!(fetch_u8(mem, pc));
        match op2 {
            0x05 => Step::Syscall, // `rip` deliberately left on the `syscall` opcode
            0x40..=0x4F => {
                // CMOVcc Gv, Ev: only reads the r/m operand when the branch
                // is taken, mirroring how we skip the write when it isn't.
                let cc = op2 & 0x0f;
                let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
                if self.cond_holds(cc) {
                    let rm_op = resolve(modrm.kind, pc2);
                    let v = fetch!(self.read_operand(mem, rm_op, width));
                    self.gpr[modrm.reg] = mask_w(v, width);
                }
                self.next(pc2)
            }
            0x80..=0x8F => {
                let cc = op2 & 0x0f;
                let (rel, pc2) = fetch!(fetch_i32(mem, pc));
                if self.cond_holds(cc) {
                    self.jump((pc2 as i64).wrapping_add(i64::from(rel)) as u64)
                } else {
                    self.next(pc2)
                }
            }
            0x90..=0x9F => {
                // SETcc Eb: r/m8 = 1 if the condition holds, else 0.
                let cc = op2 & 0x0f;
                let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
                let rm_op = resolve8(modrm.kind, pc2, has_rex);
                let v = u64::from(self.cond_holds(cc));
                fetch!(self.write_operand(mem, rm_op, v, 8));
                self.next(pc2)
            }
            0xAF => {
                // IMUL Gv, Ev: reg *= r/m (signed), CF/OF set on overflow.
                let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
                let rm_op = resolve(modrm.kind, pc2);
                let b = fetch!(self.read_operand(mem, rm_op, width));
                let a = mask_w(self.gpr[modrm.reg], width);
                let av = sign_extend_128(u128::from(a), width);
                let bv = sign_extend_128(u128::from(b), width);
                let p = av * bv;
                let cf = !fits_signed(p, width);
                self.gpr[modrm.reg] = mask_w(p as u128 as u64, width);
                self.flags.cf = cf;
                self.flags.of = cf;
                self.next(pc2)
            }
            0xB6 | 0xB7 | 0xBE | 0xBF => {
                // MOVZX/MOVSX Gv, Eb/Ew.
                let src_width = if op2 == 0xB6 || op2 == 0xBE { 8 } else { 16 };
                let signed = op2 == 0xBE || op2 == 0xBF;
                let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
                let rm_op = if src_width == 8 {
                    resolve8(modrm.kind, pc2, has_rex)
                } else {
                    resolve(modrm.kind, pc2)
                };
                let raw = fetch!(self.read_operand(mem, rm_op, src_width));
                let val = if signed { sign_extend_w(raw, src_width) as u64 } else { raw };
                self.gpr[modrm.reg] = mask_w(val, width);
                self.next(pc2)
            }
            _ => Step::Illegal,
        }
    }

    #[allow(clippy::too_many_lines)]
    fn exec(&mut self, mem: &mut GuestMemory) -> Step {
        // Legacy prefixes (operand-size `0x66`, `REP`/`REPE` `0xF3`, `REPNE`
        // `0xF2`) precede any `REX` byte, which in turn must immediately
        // precede the opcode.
        let mut pc = self.rip;
        let mut opsize16 = false;
        let mut rep: u8 = 0; // 0 = none, 1 = REP/REPE (F3), 2 = REPNE (F2)
        loop {
            let (b, next) = fetch!(fetch_u8(mem, pc));
            match b {
                0x66 => opsize16 = true,
                0xF3 => rep = 1,
                0xF2 => rep = 2,
                _ => break,
            }
            pc = next;
        }
        let (b0, pc) = fetch!(fetch_u8(mem, pc));
        let (rex, has_rex, opcode, pc) = if (0x40..=0x4f).contains(&b0) {
            let (op, pc2) = fetch!(fetch_u8(mem, pc));
            (Rex::from_byte(b0), true, op, pc2)
        } else {
            (Rex::default(), false, b0, pc)
        };
        let width = if rex.w {
            64
        } else if opsize16 {
            16
        } else {
            32
        };

        match opcode {
            0x50..=0x57 => {
                let r = usize::from(opcode - 0x50) | (usize::from(rex.b) << 3);
                let val = self.gpr[r];
                fetch!(self.push(mem, val));
                self.next(pc)
            }
            0x58..=0x5F => {
                let r = usize::from(opcode - 0x58) | (usize::from(rex.b) << 3);
                let val = fetch!(self.pop(mem));
                self.gpr[r] = val;
                self.next(pc)
            }
            0x68 => {
                let (imm, pc2) = fetch!(fetch_i32(mem, pc));
                fetch!(self.push(mem, i64::from(imm) as u64));
                self.next(pc2)
            }
            0x6A => {
                let (imm, pc2) = fetch!(fetch_i8(mem, pc));
                fetch!(self.push(mem, i64::from(imm) as u64));
                self.next(pc2)
            }
            0x8D => self.lea(mem, pc, rex, width),
            0x89 => self.mov_rm_gv(mem, pc, rex, width),
            0x8B => self.mov_gv_rm(mem, pc, rex, width),
            0x88 => {
                // MOV r/m8, r8 (Eb,Gb).
                let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
                let rm_op = resolve8(modrm.kind, pc2, has_rex);
                let val = fetch!(self.read_operand(mem, reg8_operand(modrm.reg, has_rex), 8));
                fetch!(self.write_operand(mem, rm_op, val, 8));
                self.next(pc2)
            }
            0x8A => {
                // MOV r8, r/m8 (Gb,Eb).
                let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
                let rm_op = resolve8(modrm.kind, pc2, has_rex);
                let val = fetch!(self.read_operand(mem, rm_op, 8));
                fetch!(self.write_operand(mem, reg8_operand(modrm.reg, has_rex), val, 8));
                self.next(pc2)
            }
            0xB0..=0xB7 => {
                // MOV r8, imm8.
                let r = usize::from(opcode - 0xB0) | (usize::from(rex.b) << 3);
                let (imm, pc2) = fetch!(fetch_u8(mem, pc));
                fetch!(self.write_operand(mem, reg8_operand(r, has_rex), u64::from(imm), 8));
                self.next(pc2)
            }
            0xC6 => {
                // MOV r/m8, imm8 (/0 only).
                let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
                if modrm.reg != 0 {
                    return Step::Illegal;
                }
                let (imm, pc3) = fetch!(fetch_u8(mem, pc2));
                let rm_op = resolve8(modrm.kind, pc3, has_rex);
                fetch!(self.write_operand(mem, rm_op, u64::from(imm), 8));
                self.next(pc3)
            }
            0x63 => {
                // MOVSXD Gv, Ed (sign-extends to 64 bits under REX.W; a
                // plain 32-bit move otherwise).
                let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
                let rm_op = resolve(modrm.kind, pc2);
                let raw = fetch!(self.read_operand(mem, rm_op, 32));
                let val = if width == 64 { sign_extend_w(raw, 32) as u64 } else { raw };
                self.gpr[modrm.reg] = mask_w(val, width);
                self.next(pc2)
            }
            0x69 => self.imul_imm(mem, pc, rex, width, false),
            0x6B => self.imul_imm(mem, pc, rex, width, true),
            0x86 => self.xchg(mem, pc, rex, has_rex, 8),
            0x87 => self.xchg(mem, pc, rex, has_rex, width),
            0x91..=0x97 => {
                // XCHG rAX, r (0x90 itself is the plain NOP / XCHG eax,eax).
                let r = usize::from(opcode - 0x90) | (usize::from(rex.b) << 3);
                let a = fetch!(self.read_operand(mem, Operand::Reg(RAX), width));
                let b = fetch!(self.read_operand(mem, Operand::Reg(r), width));
                fetch!(self.write_operand(mem, Operand::Reg(RAX), b, width));
                fetch!(self.write_operand(mem, Operand::Reg(r), a, width));
                self.next(pc)
            }
            0x98 => {
                // CBW / CWDE / CDQE: sign-extend AL/AX/EAX into AX/EAX/RAX.
                match width {
                    64 => self.gpr[RAX] = sign_extend_w(mask_w(self.gpr[RAX], 32), 32) as u64,
                    16 => {
                        let v = sign_extend_w(mask_w(self.gpr[RAX], 8), 8) as u64;
                        self.gpr[RAX] = (self.gpr[RAX] & !0xffffu64) | (v & 0xffff);
                    }
                    _ => {
                        let v = sign_extend_w(mask_w(self.gpr[RAX], 16), 16) as u64;
                        self.gpr[RAX] = mask_w(v, 32);
                    }
                }
                self.next(pc)
            }
            0x99 => {
                // CWD / CDQ / CQO: sign-extend AX/EAX/RAX's sign bit into
                // DX/EDX/RDX.
                match width {
                    64 => self.gpr[RDX] = if sign_bit(self.gpr[RAX], 64) { u64::MAX } else { 0 },
                    16 => {
                        let d = if sign_bit(self.gpr[RAX] & 0xffff, 16) { 0xffffu64 } else { 0 };
                        self.gpr[RDX] = (self.gpr[RDX] & !0xffffu64) | d;
                    }
                    _ => {
                        self.gpr[RDX] =
                            if sign_bit(self.gpr[RAX] & 0xffff_ffff, 32) { 0xffff_ffff } else { 0 };
                    }
                }
                self.next(pc)
            }
            0xA4 => self.movs(mem, pc, 8, rep),
            0xA5 => self.movs(mem, pc, width, rep),
            0xA6 => self.cmps(mem, pc, 8, rep),
            0xA7 => self.cmps(mem, pc, width, rep),
            0xAA => self.stos(mem, pc, 8, rep),
            0xAB => self.stos(mem, pc, width, rep),
            0xAC => self.lods(mem, pc, 8, rep),
            0xAD => self.lods(mem, pc, width, rep),
            0xAE => self.scas(mem, pc, 8, rep),
            0xAF => self.scas(mem, pc, width, rep),
            0xFC => {
                self.df = false;
                self.next(pc)
            }
            0xFD => {
                self.df = true;
                self.next(pc)
            }
            0xC9 => {
                // LEAVE: rsp = rbp; rbp = pop().
                self.gpr[RSP] = self.gpr[RBP];
                let val = fetch!(self.pop(mem));
                self.gpr[RBP] = val;
                self.next(pc)
            }
            0xB8..=0xBF => {
                let r = usize::from(opcode - 0xB8) | (usize::from(rex.b) << 3);
                if rex.w {
                    let (imm, pc2) = fetch!(fetch_u64(mem, pc));
                    self.gpr[r] = imm; // MOVABS
                    self.next(pc2)
                } else {
                    let (imm, pc2) = fetch!(fetch_u32(mem, pc));
                    self.gpr[r] = u64::from(imm); // zero-extends to 64 bits
                    self.next(pc2)
                }
            }
            0xC7 => self.mov_imm(mem, pc, rex, width),
            0x00 => self.alu_rm_gv8(mem, pc, rex, has_rex, AluOp::Add, true),
            0x02 => self.alu_gv_rm8(mem, pc, rex, has_rex, AluOp::Add),
            0x01 => self.alu_rm_gv(mem, pc, rex, width, AluOp::Add, true),
            0x03 => self.alu_gv_rm(mem, pc, rex, width, AluOp::Add),
            0x09 => self.alu_rm_gv(mem, pc, rex, width, AluOp::Or, true),
            0x0B => self.alu_gv_rm(mem, pc, rex, width, AluOp::Or),
            0x21 => self.alu_rm_gv(mem, pc, rex, width, AluOp::And, true),
            0x23 => self.alu_gv_rm(mem, pc, rex, width, AluOp::And),
            0x28 => self.alu_rm_gv8(mem, pc, rex, has_rex, AluOp::Sub, true),
            0x2A => self.alu_gv_rm8(mem, pc, rex, has_rex, AluOp::Sub),
            0x29 => self.alu_rm_gv(mem, pc, rex, width, AluOp::Sub, true),
            0x2B => self.alu_gv_rm(mem, pc, rex, width, AluOp::Sub),
            0x30 => self.alu_rm_gv8(mem, pc, rex, has_rex, AluOp::Xor, true),
            0x32 => self.alu_gv_rm8(mem, pc, rex, has_rex, AluOp::Xor),
            0x31 => self.alu_rm_gv(mem, pc, rex, width, AluOp::Xor, true),
            0x33 => self.alu_gv_rm(mem, pc, rex, width, AluOp::Xor),
            0x38 => self.alu_rm_gv8(mem, pc, rex, has_rex, AluOp::Cmp, false),
            0x3A => self.alu_gv_rm8(mem, pc, rex, has_rex, AluOp::Cmp),
            0x39 => self.alu_rm_gv(mem, pc, rex, width, AluOp::Cmp, false),
            0x3B => self.alu_gv_rm(mem, pc, rex, width, AluOp::Cmp),
            0x84 => self.alu_rm_gv8(mem, pc, rex, has_rex, AluOp::Test, false),
            0x85 => self.alu_rm_gv(mem, pc, rex, width, AluOp::Test, false),
            0x80 => self.group1_imm(mem, pc, rex, has_rex, 8, true),
            0x81 => self.group1_imm(mem, pc, rex, has_rex, width, false),
            0x83 => self.group1_imm(mem, pc, rex, has_rex, width, true),
            0xF6 => self.group3(mem, pc, rex, has_rex, 8),
            0xF7 => self.group3(mem, pc, rex, has_rex, width),
            0xFE => self.group4(mem, pc, rex, has_rex),
            0xFF => self.group5(mem, pc, rex, width),
            0xC1 => self.group2(mem, pc, rex, width, false),
            0xD3 => self.group2(mem, pc, rex, width, true),
            0xE8 => {
                let (rel, pc2) = fetch!(fetch_i32(mem, pc));
                fetch!(self.push(mem, pc2));
                self.jump((pc2 as i64).wrapping_add(i64::from(rel)) as u64)
            }
            0xC3 => {
                let target = fetch!(self.pop(mem));
                self.jump(target)
            }
            0xE9 => {
                let (rel, pc2) = fetch!(fetch_i32(mem, pc));
                self.jump((pc2 as i64).wrapping_add(i64::from(rel)) as u64)
            }
            0xEB => {
                let (rel, pc2) = fetch!(fetch_i8(mem, pc));
                self.jump((pc2 as i64).wrapping_add(i64::from(rel)) as u64)
            }
            0x70..=0x7F => {
                let cc = opcode & 0x0f;
                let (rel, pc2) = fetch!(fetch_i8(mem, pc));
                if self.cond_holds(cc) {
                    self.jump((pc2 as i64).wrapping_add(i64::from(rel)) as u64)
                } else {
                    self.next(pc2)
                }
            }
            0x90 => self.next(pc), // NOP (also XCHG eax,eax, a no-op either way)
            0x0F => self.exec_0f(mem, pc, rex, has_rex, width),
            _ => Step::Illegal,
        }
    }
}

impl Vcpu for X86Interp {
    fn run(&mut self, mem: &mut GuestMemory) -> Result<Exit, VcpuError> {
        for _ in 0..MAX_STEPS {
            match self.exec(mem) {
                Step::Next | Step::Branched => {}
                Step::Syscall => return Ok(Exit::Syscall),
                Step::Illegal => return Ok(Exit::IllegalInstruction { pc: self.rip }),
                Step::Fault { addr, write } => return Ok(Exit::MemFault { addr, write }),
            }
        }
        Ok(Exit::Interrupted)
    }

    fn syscall_nr(&self) -> u64 {
        self.gpr[RAX]
    }

    fn syscall_args(&self) -> [u64; 6] {
        [
            self.gpr[RDI],
            self.gpr[RSI],
            self.gpr[RDX],
            self.gpr[R10],
            self.gpr[R8],
            self.gpr[R9],
        ]
    }

    fn set_syscall_ret(&mut self, value: u64) {
        self.gpr[RAX] = value;
        self.rip = self.rip.wrapping_add(2); // `syscall` is always the 2-byte 0F 05
    }

    fn reg(&self, idx: usize) -> u64 {
        if idx < 16 { self.gpr[idx] } else { 0 }
    }

    fn set_reg(&mut self, idx: usize, value: u64) {
        if idx < 16 {
            self.gpr[idx] = value;
        }
    }

    fn pc(&self) -> u64 {
        self.rip
    }

    fn set_pc(&mut self, pc: u64) {
        self.rip = pc;
    }

    fn sp(&self) -> u64 {
        self.gpr[RSP]
    }

    fn set_sp(&mut self, sp: u64) {
        self.gpr[RSP] = sp;
    }

    fn set_tls(&mut self, value: u64) {
        self.fs_base = value;
    }

    fn fork(&self) -> Box<dyn Vcpu> {
        Box::new(self.clone())
    }

    fn reset(&mut self, entry: u64, sp: u64) {
        self.gpr = [0; 16];
        self.gpr[RSP] = sp;
        self.rip = entry;
        self.flags = Flags::default();
        self.df = false;
        self.fs_base = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vcpu::Prot;

    /// A 64 KiB rwx region at `0x1_0000`, generous enough for code, a small
    /// data area, and a stack — this is a scaffold test harness, not a real
    /// loader, so we don't bother separating segments by permission.
    fn mem() -> GuestMemory {
        let mut m = GuestMemory::new(0x1_0000, 16 * crate::vcpu::mem::PAGE_SIZE);
        m.map(0x1_0000, 16 * crate::vcpu::mem::PAGE_SIZE, Prot::rwx())
            .unwrap();
        m
    }

    const CODE: u64 = 0x1_1000;
    const STACK: u64 = 0x1_F000;

    fn run_one(mem: &mut GuestMemory, code: &[u8]) -> X86Interp {
        mem.write_init(CODE, code).unwrap();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.exec(mem);
        cpu
    }

    #[test]
    fn mov_imm32_zero_extends() {
        let mut m = mem();
        // mov eax, 0x1234_5678
        let cpu = run_one(&mut m, &[0xB8, 0x78, 0x56, 0x34, 0x12]);
        assert_eq!(cpu.gpr[RAX], 0x1234_5678);
        assert_eq!(cpu.rip, CODE + 5);
    }

    #[test]
    fn movabs_imm64() {
        let mut m = mem();
        // movabs rax, 0x0102030405060708
        let mut code = vec![0x48, 0xB8];
        code.extend_from_slice(&0x0102_0304_0506_0708u64.to_le_bytes());
        let cpu = run_one(&mut m, &code);
        assert_eq!(cpu.gpr[RAX], 0x0102_0304_0506_0708);
    }

    #[test]
    fn mov_reg_reg() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 0xdead_beef;
        // mov rbx, rax  (REX.W 89 /r, modrm=11 000 011)
        m.write_init(CODE, &[0x48, 0x89, 0xC3]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RBX], 0xdead_beef);
    }

    #[test]
    fn mov_mem_roundtrip_with_disp_and_sib() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 0x1122_3344_5566_7788;
        cpu.gpr[RBX] = 0x1_2000; // base for [rbx+0x10]
        // mov [rbx+0x10], rax  (REX.W 89 /r, modrm=01 000 011, disp8=0x10)
        m.write_init(CODE, &[0x48, 0x89, 0x43, 0x10]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(m.read_u64(0x1_2010).unwrap(), 0x1122_3344_5566_7788);

        // mov rcx, [rbx+0x10]  (REX.W 8B /r, modrm=01 001 011, disp8=0x10)
        m.write_init(CODE, &[0x48, 0x8B, 0x4B, 0x10]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RCX], 0x1122_3344_5566_7788);
    }

    #[test]
    fn lea_computes_address_without_reading_memory() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RBX] = 0x1_2000;
        // lea rax, [rbx+0x20]  (REX.W 8D /r, modrm=01 000 011, disp8=0x20)
        m.write_init(CODE, &[0x48, 0x8D, 0x43, 0x20]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX], 0x1_2020);
    }

    #[test]
    fn lea_rip_relative() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        // lea rax, [rip+0x10]  (REX.W 8D /r, modrm=00 000 101, disp32=0x10)
        m.write_init(CODE, &[0x48, 0x8D, 0x05, 0x10, 0x00, 0x00, 0x00])
            .unwrap();
        cpu.exec(&mut m);
        // effective address = end-of-instruction rip (CODE+7) + 0x10
        assert_eq!(cpu.gpr[RAX], CODE + 7 + 0x10);
    }

    #[test]
    fn add_sets_overflow_and_carry() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 0x7fff_ffff;
        cpu.gpr[RCX] = 1;
        // add eax, ecx  (01 /r, modrm=11 001 000 -> Ev=eax,Gv=ecx)
        m.write_init(CODE, &[0x01, 0xC8]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX], 0x8000_0000);
        assert!(cpu.flags.of, "signed overflow must set OF");
        assert!(cpu.flags.sf);
        assert!(!cpu.flags.cf);
        assert!(!cpu.flags.zf);
    }

    #[test]
    fn sub_sets_carry_on_borrow() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 1;
        cpu.gpr[RCX] = 2;
        // sub eax, ecx  (29 /r, modrm=11 001 000 -> Ev=eax,Gv=ecx)
        m.write_init(CODE, &[0x29, 0xC8]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX], 0xffff_ffff); // -1 as u32, zero-extended
        assert!(cpu.flags.cf, "1 - 2 unsigned borrows");
        assert!(cpu.flags.sf);
        assert!(!cpu.flags.zf);
    }

    #[test]
    fn cmp_does_not_store() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 5;
        cpu.gpr[RCX] = 5;
        // cmp eax, ecx  (39 /r, modrm=11 001 000)
        m.write_init(CODE, &[0x39, 0xC8]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX], 5, "CMP must not write back");
        assert!(cpu.flags.zf);
    }

    #[test]
    fn and_or_xor_clear_cf_and_of() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 0xff;
        cpu.gpr[RCX] = 0x0f;
        // and eax, ecx  (21 /r, modrm=11 001 000)
        m.write_init(CODE, &[0x21, 0xC8]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX], 0x0f);
        assert!(!cpu.flags.cf);
        assert!(!cpu.flags.of);
    }

    #[test]
    fn test_instruction_is_and_without_store() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 0x0f;
        cpu.gpr[RCX] = 0xf0;
        // test eax, ecx  (85 /r, modrm=11 001 000)
        m.write_init(CODE, &[0x85, 0xC8]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX], 0x0f, "TEST must not write back");
        assert!(cpu.flags.zf, "0x0f & 0xf0 == 0");
    }

    #[test]
    fn group1_imm_add_and_cmp() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 10;
        // add eax, 5  (83 /0 ib, modrm=11 000 000)
        m.write_init(CODE, &[0x83, 0xC0, 0x05]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX], 15);

        // cmp eax, 15  (83 /7 ib, modrm=11 111 000)
        m.write_init(CODE, &[0x83, 0xF8, 0x0F]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX], 15, "CMP must not write back");
        assert!(cpu.flags.zf);
    }

    #[test]
    fn inc_dec_leave_carry_flag_alone() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 41;
        cpu.flags.cf = true; // pre-set CF to confirm INC leaves it untouched
        // inc eax  (FF /0, modrm=11 000 000)
        m.write_init(CODE, &[0xFF, 0xC0]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX], 42);
        assert!(cpu.flags.cf, "INC must not touch CF");

        // dec eax  (FF /1, modrm=11 001 000)
        m.write_init(CODE, &[0xFF, 0xC8]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX], 41);
        assert!(cpu.flags.cf, "DEC must not touch CF");
    }

    #[test]
    fn neg_sets_carry_unless_zero() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 5;
        // neg eax  (F7 /3, modrm=11 011 000)
        m.write_init(CODE, &[0xF7, 0xD8]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX], 0xffff_fffb); // -5 as u32
        assert!(cpu.flags.cf);

        cpu.gpr[RAX] = 0;
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX], 0);
        assert!(!cpu.flags.cf, "NEG 0 must clear CF");
    }

    #[test]
    fn shifts_by_immediate_and_cl() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 1;
        // shl eax, 4  (C1 /4 ib, modrm=11 100 000)
        m.write_init(CODE, &[0xC1, 0xE0, 0x04]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX], 0x10);

        // shr eax, cl  (D3 /5, modrm=11 101 000), cl = 2
        cpu.gpr[RCX] = 2;
        m.write_init(CODE, &[0xD3, 0xE8]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX], 0x4);

        // sar eax, 1  (C1 /7 ib, modrm=11 111 000) on a negative 32-bit value
        cpu.gpr[RAX] = 0xffff_fffe; // -2 as i32
        m.write_init(CODE, &[0xC1, 0xF8, 0x01]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX] as u32 as i32, -1);
    }

    #[test]
    fn push_pop_roundtrip() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 0x1234;
        // push rax ; pop rbx
        m.write_init(CODE, &[0x50, 0x5B]).unwrap();
        cpu.exec(&mut m); // push
        assert_eq!(cpu.gpr[RSP], STACK - 8);
        cpu.exec(&mut m); // pop
        assert_eq!(cpu.gpr[RBX], 0x1234);
        assert_eq!(cpu.gpr[RSP], STACK);
    }

    #[test]
    fn push_immediates() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        // push 0x7f (6A ib) ; push -1 as imm32 (68 id, sign-extended)
        let mut code = vec![0x6A, 0x7F, 0x68];
        code.extend_from_slice(&(-1i32).to_le_bytes());
        m.write_init(CODE, &code).unwrap();
        cpu.exec(&mut m);
        assert_eq!(m.read_u64(STACK - 8).unwrap(), 0x7f);
        cpu.exec(&mut m);
        assert_eq!(m.read_u64(STACK - 16).unwrap(), u64::MAX);
    }

    #[test]
    fn call_and_ret() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        // call +3 (jumps over the 3-byte filler, to CODE+5+3=CODE+8) ; at the
        // call target: ret.
        let mut code = vec![0xE8];
        code.extend_from_slice(&3i32.to_le_bytes()); // rel32
        code.push(0x90); // filler (skipped over)
        code.push(0x90);
        code.push(0x90);
        code.push(0xC3); // ret, at CODE+8
        m.write_init(CODE, &code).unwrap();
        cpu.exec(&mut m); // call
        assert_eq!(cpu.rip, CODE + 8);
        assert_eq!(m.read_u64(STACK - 8).unwrap(), CODE + 5); // return address
        cpu.exec(&mut m); // ret
        assert_eq!(cpu.rip, CODE + 5);
        assert_eq!(cpu.gpr[RSP], STACK);
    }

    #[test]
    fn jmp_rel8_and_rel32() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        // jmp +2 (rel8) -> CODE+2+2 = CODE+4
        m.write_init(CODE, &[0xEB, 0x02]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.rip, CODE + 4);

        // jmp -0x100 (rel32) from CODE
        let mut code = vec![0xE9];
        code.extend_from_slice(&(-0x100i32).to_le_bytes());
        m.write_init(CODE, &code).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.rip, (CODE + 5).wrapping_sub(0x100));
    }

    #[test]
    fn jcc_conditions() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);

        // JE taken: ZF set via `cmp eax,eax`, then `je +2`.
        cpu.gpr[RAX] = 7;
        m.write_init(CODE, &[0x39, 0xC0, 0x74, 0x02]).unwrap(); // cmp eax,eax; je +2
        cpu.exec(&mut m); // cmp
        assert!(cpu.flags.zf);
        cpu.exec(&mut m); // je
        assert_eq!(cpu.rip, CODE + 2 + 2 + 2);

        // JNE not taken (ZF still set): falls through.
        cpu.rip = CODE + 2;
        m.write_init(CODE + 2, &[0x75, 0x02]).unwrap(); // jne +2
        cpu.exec(&mut m);
        assert_eq!(cpu.rip, CODE + 4);

        // JB / JAE via unsigned CMP.
        cpu.gpr[RAX] = 1;
        cpu.gpr[RCX] = 2;
        m.write_init(CODE, &[0x39, 0xC8]).unwrap(); // cmp eax,ecx (1 vs 2)
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert!(cpu.flags.cf, "1 < 2 unsigned sets CF");
        m.write_init(CODE, &[0x72, 0x02]).unwrap(); // jb +2
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.rip, CODE + 2 + 2);
        m.write_init(CODE, &[0x73, 0x02]).unwrap(); // jae +2 (not taken, CF=1)
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.rip, CODE + 2);

        // JL / JGE via signed CMP.
        cpu.gpr[RAX] = 0xffff_ffff; // -1 as i32
        cpu.gpr[RCX] = 1;
        m.write_init(CODE, &[0x39, 0xC8]).unwrap(); // cmp eax,ecx (-1 vs 1)
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert!(cpu.flags.sf != cpu.flags.of, "-1 < 1 signed");
        m.write_init(CODE, &[0x7C, 0x02]).unwrap(); // jl +2
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.rip, CODE + 2 + 2);
        m.write_init(CODE, &[0x7D, 0x02]).unwrap(); // jge +2 (not taken)
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.rip, CODE + 2);
    }

    #[test]
    fn jcc_rel32_two_byte_opcode() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.flags.zf = true;
        // je rel32 (0F 84 <rel32>), taken.
        let mut code = vec![0x0F, 0x84];
        code.extend_from_slice(&0x100i32.to_le_bytes());
        m.write_init(CODE, &code).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.rip, CODE + 6 + 0x100);
    }

    /// End-to-end: a tiny statically-linked `write(1, msg, len)` +
    /// `exit_group(0)` sequence, entirely via `mov r32,imm32` (no memory
    /// operands needed) so the test stays focused on the `SYSCALL` trap path
    /// that the kernel's run/serve loop depends on.
    #[test]
    fn write_then_exit_group_traps_with_right_nr_and_args() {
        let mut m = mem();
        let msg_addr = 0x1_2000u64;
        m.write_init(msg_addr, b"hi\n").unwrap();

        let code: Vec<u8> = vec![
            0xBF, 0x01, 0x00, 0x00, 0x00, // mov edi, 1        (fd)
            0xBE, 0x00, 0x20, 0x01, 0x00, // mov esi, 0x1_2000 (buf)
            0xBA, 0x03, 0x00, 0x00, 0x00, // mov edx, 3        (len)
            0xB8, 0x01, 0x00, 0x00, 0x00, // mov eax, 1        (SYS_write)
            0x0F, 0x05, // syscall
            0xB8, 0xE7, 0x00, 0x00, 0x00, // mov eax, 231      (SYS_exit_group)
            0x31, 0xFF, // xor edi, edi
            0x0F, 0x05, // syscall
        ];
        m.write_init(CODE, &code).unwrap();

        let mut cpu = X86Interp::new(CODE, STACK);
        let syscall_pc = CODE + 20; // offset of the first `0F 05`

        match cpu.run(&mut m).unwrap() {
            Exit::Syscall => {}
            other => panic!("expected Exit::Syscall, got {other:?}"),
        }
        assert_eq!(cpu.pc(), syscall_pc, "rip must stay on the syscall opcode");
        assert_eq!(cpu.syscall_nr(), 1, "SYS_write");
        assert_eq!(cpu.syscall_args()[0], 1);
        assert_eq!(cpu.syscall_args()[1], msg_addr);
        assert_eq!(cpu.syscall_args()[2], 3);
        cpu.set_syscall_ret(3); // "wrote" 3 bytes
        assert_eq!(cpu.pc(), syscall_pc + 2);

        match cpu.run(&mut m).unwrap() {
            Exit::Syscall => {}
            other => panic!("expected Exit::Syscall, got {other:?}"),
        }
        assert_eq!(cpu.syscall_nr(), 231, "SYS_exit_group");
        assert_eq!(cpu.syscall_args()[0], 0);
    }

    #[test]
    fn illegal_opcode_surfaces_as_exit() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        // 0x0F 0xFF is not decoded by our subset.
        m.write_init(CODE, &[0x0F, 0xFF]).unwrap();
        match cpu.run(&mut m).unwrap() {
            Exit::IllegalInstruction { pc } => assert_eq!(pc, CODE),
            other => panic!("expected IllegalInstruction, got {other:?}"),
        }
    }

    #[test]
    fn fork_and_reset() {
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 42;
        let forked = cpu.fork();
        assert_eq!(forked.reg(RAX), 42);

        cpu.reset(0x2_0000, 0x3_0000);
        assert_eq!(cpu.pc(), 0x2_0000);
        assert_eq!(cpu.sp(), 0x3_0000);
        assert_eq!(cpu.gpr[RAX], 0);
    }

    #[test]
    fn mov_r8_imm8_and_high_byte_regs() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        // mov al, 0x12 ; mov ah, 0x34 ; mov cl, al (88 C1) ; mov bl, ah (8A DC)
        let code = [0xB0, 0x12, 0xB4, 0x34, 0x88, 0xC1, 0x8A, 0xDC];
        m.write_init(CODE, &code).unwrap();
        cpu.exec(&mut m); // mov al, 0x12
        assert_eq!(cpu.gpr[RAX] & 0xff, 0x12);
        cpu.exec(&mut m); // mov ah, 0x34
        assert_eq!((cpu.gpr[RAX] >> 8) & 0xff, 0x34, "AH is the high byte of RAX");
        cpu.exec(&mut m); // mov cl, al
        assert_eq!(cpu.gpr[RCX] & 0xff, 0x12);
        cpu.exec(&mut m); // mov bl, ah
        assert_eq!(cpu.gpr[RBX] & 0xff, 0x34);
    }

    #[test]
    fn mov_r8_via_rex_uses_low_byte_not_high_byte() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RDI] = 0xffff_ffff_ffff_ff00;
        // mov dil, 0x7f  (REX 40 B7 7F — REX present, so reg 7 is DIL, not BH)
        m.write_init(CODE, &[0x40, 0xB7, 0x7F]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RDI], 0xffff_ffff_ffff_ff7f);
    }

    #[test]
    fn alu_8bit_forms_add_sub_xor_cmp_and_group1() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        let code: Vec<u8> = vec![
            0xB0, 0x05, // mov al, 5
            0xB1, 0x03, // mov cl, 3
            0x00, 0xC8, // add al, cl  (Eb,Gb) -> al = 8
            0x28, 0xC8, // sub al, cl  (Eb,Gb) -> al = 5
            0x38, 0xC8, // cmp al, cl  (Eb,Gb): 5 vs 3, no borrow
            0x30, 0xC0, // xor al, al  -> al = 0, ZF set
            0x80, 0xC0, 0x0A, // add al, 0x0a (group1 /0 imm8) -> al = 10
            0x80, 0xF8, 0x0A, // cmp al, 0x0a (group1 /7 imm8) -> ZF set
        ];
        m.write_init(CODE, &code).unwrap();
        cpu.exec(&mut m); // mov al, 5
        cpu.exec(&mut m); // mov cl, 3
        cpu.exec(&mut m); // add al, cl
        assert_eq!(cpu.gpr[RAX] & 0xff, 8);
        cpu.exec(&mut m); // sub al, cl
        assert_eq!(cpu.gpr[RAX] & 0xff, 5);
        cpu.exec(&mut m); // cmp al, cl
        assert!(!cpu.flags.cf, "5 - 3 does not borrow");
        assert!(!cpu.flags.zf);
        cpu.exec(&mut m); // xor al, al
        assert_eq!(cpu.gpr[RAX] & 0xff, 0);
        assert!(cpu.flags.zf);
        cpu.exec(&mut m); // add al, 0x0a
        assert_eq!(cpu.gpr[RAX] & 0xff, 10);
        cpu.exec(&mut m); // cmp al, 0x0a
        assert!(cpu.flags.zf, "10 == 10");
        assert_eq!(cpu.gpr[RAX] & 0xff, 10, "CMP must not write back");
    }

    #[test]
    fn movzx_and_movsx_sign_and_zero_extend() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 0xdead_beef_dead_beef;
        m.write_init(CODE, &[0xB0, 0x80]).unwrap(); // mov al, 0x80
        cpu.exec(&mut m);

        // movzx rax, al  (48 0F B6 C0)
        m.write_init(CODE, &[0x48, 0x0F, 0xB6, 0xC0]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX], 0x80, "MOVZX zero-extends");

        // movsx rbx, al  (48 0F BE D8)
        m.write_init(CODE, &[0x48, 0x0F, 0xBE, 0xD8]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RBX], 0xffff_ffff_ffff_ff80, "MOVSX sign-extends");

        cpu.gpr[RCX] = 0x8000;
        // movzx eax, cx  (0F B7 C1)
        m.write_init(CODE, &[0x0F, 0xB7, 0xC1]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX], 0x8000);

        // movsx edx, cx  (0F BF D1)
        m.write_init(CODE, &[0x0F, 0xBF, 0xD1]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(
            cpu.gpr[RDX], 0xffff_8000,
            "MOVSX from 16-bit, 32-bit dest zero-extends the upper 32 bits of RDX"
        );
    }

    #[test]
    fn movsxd_sign_extends_dword_to_qword() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RCX] = 0xffff_ffff_8000_0000; // low 32 bits = 0x8000_0000 (negative)
        // movsxd rax, ecx  (REX.W 63 /r, modrm=11 000 001)
        m.write_init(CODE, &[0x48, 0x63, 0xC1]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX], 0xffff_ffff_8000_0000);
    }

    #[test]
    fn cmovcc_and_setcc_driven_by_cmp() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 1;
        cpu.gpr[RCX] = 1;
        cpu.gpr[RDX] = 0xffff_ffff_ffff_ffff;
        cpu.gpr[RBX] = 0;
        cpu.gpr[R8] = 0;
        // cmp eax, ecx (39 C8); cmove rdx, rax (48 0F 44 D0);
        // sete r8b (41 0F 94 C0, true — REX.B selects r8 for the rm field);
        // setne bl (0F 95 C3, false)
        let code = [
            0x39, 0xC8, 0x48, 0x0F, 0x44, 0xD0, 0x41, 0x0F, 0x94, 0xC0, 0x0F, 0x95, 0xC3,
        ];
        m.write_init(CODE, &code).unwrap();
        cpu.exec(&mut m); // cmp eax, ecx (equal -> ZF set)
        assert!(cpu.flags.zf);
        cpu.exec(&mut m); // cmove rdx, rax (condition true -> rdx = rax)
        assert_eq!(cpu.gpr[RDX], 1);
        cpu.exec(&mut m); // sete r8b (condition true -> r8b = 1)
        assert_eq!(cpu.gpr[R8] & 0xff, 1);
        cpu.exec(&mut m); // setne bl (condition false -> bl = 0)
        assert_eq!(cpu.gpr[RBX] & 0xff, 0);
    }

    #[test]
    fn cmovcc_does_not_write_when_condition_is_false() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 1;
        cpu.gpr[RCX] = 2;
        cpu.gpr[RDX] = 0x1234;
        // cmp eax, ecx (39 C8, not equal); cmove rdx, rax (48 0F 44 D0)
        let code = [0x39, 0xC8, 0x48, 0x0F, 0x44, 0xD0];
        m.write_init(CODE, &code).unwrap();
        cpu.exec(&mut m);
        assert!(!cpu.flags.zf);
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RDX], 0x1234, "CMOVcc must not write when the condition is false");
    }

    #[test]
    fn mul_and_div_pair() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 6;
        cpu.gpr[RCX] = 7;
        // mul ecx  (F7 /4, modrm=11 100 001)
        m.write_init(CODE, &[0xF7, 0xE1]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX] & 0xffff_ffff, 42);
        assert_eq!(cpu.gpr[RDX] & 0xffff_ffff, 0);
        assert!(!cpu.flags.cf, "42 fits in 32 bits, no overflow into edx");

        // div ecx  (F7 /6, modrm=11 110 001): 42 / 7 = 6 r 0
        m.write_init(CODE, &[0xF7, 0xF1]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX] & 0xffff_ffff, 6);
        assert_eq!(cpu.gpr[RDX] & 0xffff_ffff, 0);

        // idiv ecx: -20 / 7 = -2 r -6  (signed)
        cpu.gpr[RAX] = 0xffff_ffec; // -20 as i32, RDX:RAX dividend sign-extended below
        cpu.gpr[RDX] = 0xffff_ffff; // sign-extension of a negative EAX into EDX (as CDQ would do)
        cpu.gpr[RCX] = 7;
        // idiv ecx  (F7 /7, modrm=11 111 001)
        m.write_init(CODE, &[0xF7, 0xF9]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX] as u32 as i32, -2);
        assert_eq!(cpu.gpr[RDX] as u32 as i32, -6);
    }

    #[test]
    fn mul_8bit_and_div_by_zero_is_illegal() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 20; // AL = 20
        cpu.gpr[RCX] = 3; // CL = 3
        // mul cl  (F6 /4, modrm=11 100 001)
        m.write_init(CODE, &[0xF6, 0xE1]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX] & 0xffff, 60, "AX = AL * CL for an 8-bit operand");

        cpu.gpr[RCX] = 0;
        // div cl  (F6 /6, modrm=11 110 001): divide by zero
        m.write_init(CODE, &[0xF6, 0xF1]).unwrap();
        cpu.rip = CODE;
        match cpu.run(&mut m).unwrap() {
            Exit::IllegalInstruction { .. } => {}
            other => panic!("expected IllegalInstruction on divide-by-zero, got {other:?}"),
        }
    }

    #[test]
    fn imul_two_and_three_operand_forms() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 6;
        cpu.gpr[RCX] = 7;
        // imul eax, ecx  (0F AF /r, modrm=11 000 001)
        m.write_init(CODE, &[0x0F, 0xAF, 0xC1]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX] & 0xffff_ffff, 42);
        assert!(!cpu.flags.cf);

        // imul edx, ecx, 100  (69 /r id): edx = ecx * 100
        let mut code = vec![0x69, 0xD1];
        code.extend_from_slice(&100i32.to_le_bytes());
        m.write_init(CODE, &code).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RDX] & 0xffff_ffff, 700);

        // imul ebx, ecx, 5  (6B /r ib): ebx = ecx * 5
        m.write_init(CODE, &[0x6B, 0xD9, 0x05]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RBX] & 0xffff_ffff, 35);
    }

    #[test]
    fn cdq_cqo_and_cwde_cdqe() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 0xffff_ffff_8000_0000; // eax = 0x8000_0000 (negative)
        // cdq (99)
        m.write_init(CODE, &[0x99]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RDX] as u32, 0xffff_ffff);

        cpu.gpr[RAX] = 0x8000_0000; // eax negative
        // cwde (98): eax = sign_extend(ax) — ax's low bit pattern is 0x0000 here
        m.write_init(CODE, &[0x98]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX], 0, "AX=0 sign-extends to EAX=0, clearing the upper 32 bits");

        cpu.gpr[RAX] = 0xffff_ffff_ffff_8000;
        // cdqe (REX.W 98): rax = sign_extend(eax)
        m.write_init(CODE, &[0x48, 0x98]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX], 0xffff_ffff_ffff_8000);

        cpu.gpr[RAX] = 0xffff_ffff;
        // cqo (REX.W 99): rdx = sign_extend(sign bit of rax)
        m.write_init(CODE, &[0x48, 0x99]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RDX], 0, "rax's sign bit (bit 63) is 0 here");
    }

    #[test]
    fn not_and_group4_inc_dec_byte() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 0x0000_0000_ffff_0f0f;
        // not eax  (F7 /2, modrm=11 010 000)
        m.write_init(CODE, &[0xF7, 0xD0]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX], 0x0000_f0f0);

        cpu.gpr[RCX] = 0x7f;
        // inc cl  (FE /0, modrm=11 000 001)
        m.write_init(CODE, &[0xFE, 0xC1]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RCX] & 0xff, 0x80);

        // dec cl  (FE /1, modrm=11 001 001)
        m.write_init(CODE, &[0xFE, 0xC9]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RCX] & 0xff, 0x7f);
    }

    #[test]
    fn group5_call_jmp_and_push_indirect() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = CODE + 0x100;
        // call rax  (FF /2, modrm=11 010 000)
        m.write_init(CODE, &[0xFF, 0xD0]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.rip, CODE + 0x100);
        assert_eq!(m.read_u64(STACK - 8).unwrap(), CODE + 2, "return address pushed");

        cpu.gpr[RBX] = CODE + 0x200;
        // jmp rbx  (FF /4, modrm=11 100 011)
        m.write_init(CODE + 0x100, &[0xFF, 0xE3]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.rip, CODE + 0x200);

        cpu.gpr[RCX] = 0xdead_beef;
        // push rcx  (FF /6, modrm=11 110 001)
        m.write_init(CODE + 0x200, &[0xFF, 0xF1]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(m.read_u64(STACK - 16).unwrap(), 0xdead_beef);
    }

    #[test]
    fn leave_restores_rsp_from_rbp() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RBP] = STACK - 0x40;
        m.write_init(STACK - 0x40, &0x1122_3344u64.to_le_bytes()).unwrap();
        // leave (C9)
        m.write_init(CODE, &[0xC9]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RBP], 0x1122_3344);
        assert_eq!(cpu.gpr[RSP], STACK - 0x40 + 8);
    }

    #[test]
    fn xchg_swaps_registers_and_memory() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 1;
        cpu.gpr[RCX] = 2;
        // xchg ecx, eax  (91)
        m.write_init(CODE, &[0x91]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX], 2);
        assert_eq!(cpu.gpr[RCX], 1);

        cpu.gpr[RBX] = 0x1_8000;
        m.write_init(0x1_8000, &0x99u64.to_le_bytes()).unwrap();
        cpu.gpr[RDX] = 0x77;
        // xchg [rbx], edx  (87 /r, modrm=00 010 011)
        m.write_init(CODE, &[0x87, 0x13]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RDX] & 0xffff_ffff, 0x99);
        assert_eq!(m.read_u64(0x1_8000).unwrap() & 0xffff_ffff, 0x77);
    }

    #[test]
    fn rep_movsb_block_copy() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        let src = 0x1_2000u64;
        let dst = 0x1_3000u64;
        m.write_init(src, b"hello, nixvm!").unwrap();
        cpu.gpr[RSI] = src;
        cpu.gpr[RDI] = dst;
        cpu.gpr[RCX] = 13;
        // rep movsb  (F3 A4)
        m.write_init(CODE, &[0xF3, 0xA4]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RCX], 0);
        assert_eq!(cpu.gpr[RSI], src + 13);
        assert_eq!(cpu.gpr[RDI], dst + 13);
        let mut buf = [0u8; 13];
        m.read(dst, &mut buf).unwrap();
        assert_eq!(&buf, b"hello, nixvm!");
    }

    #[test]
    fn rep_stosb_and_repe_scasb_and_cmpsb() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        let dst = 0x1_2000u64;
        cpu.gpr[RAX] = 0x41; // 'A'
        cpu.gpr[RDI] = dst;
        cpu.gpr[RCX] = 8;
        // rep stosb  (F3 AA)
        m.write_init(CODE, &[0xF3, 0xAA]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RCX], 0);
        assert_eq!(cpu.gpr[RDI], dst + 8);
        let mut buf = [0u8; 8];
        m.read(dst, &mut buf).unwrap();
        assert_eq!(&buf, b"AAAAAAAA");

        // repe scasb: scan for a byte != 'A' (none here, so it runs to completion)
        cpu.gpr[RAX] = 0x41;
        cpu.gpr[RDI] = dst;
        cpu.gpr[RCX] = 8;
        m.write_init(CODE, &[0xF3, 0xAE]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RCX], 0);
        assert!(cpu.flags.zf);

        // repe cmpsb over two identical buffers.
        let dst2 = 0x1_3000u64;
        m.write_init(dst2, b"AAAAAAAA").unwrap();
        cpu.gpr[RSI] = dst;
        cpu.gpr[RDI] = dst2;
        cpu.gpr[RCX] = 8;
        m.write_init(CODE, &[0xF3, 0xA6]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RCX], 0);
        assert!(cpu.flags.zf, "all 8 bytes matched");
    }

    #[test]
    fn cld_std_control_string_op_direction() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        let base = 0x1_2000u64;
        m.write_init(base, b"XYZ").unwrap();
        cpu.gpr[RSI] = base + 2; // start at the last byte, walk backward
        cpu.gpr[RDI] = 0x1_3000 + 2;
        cpu.gpr[RCX] = 3;
        // std ; rep movsb
        m.write_init(CODE, &[0xFD, 0xF3, 0xA4]).unwrap();
        cpu.exec(&mut m); // std
        assert!(cpu.df);
        cpu.exec(&mut m); // rep movsb
        assert_eq!(cpu.gpr[RSI], base - 1);
        let mut buf = [0u8; 3];
        m.read(0x1_3000, &mut buf).unwrap();
        assert_eq!(&buf, b"XYZ");
    }

    /// A tiny assembled loop — `for (i = 5; i != 0; i--) sum += i;` — that
    /// exercises `MOV`, `ADD`, `DEC`, and `JNZ` together and leaves `15` in
    /// `ecx` (the sum of `1..=5`).
    #[test]
    fn assembled_loop_sums_one_to_five() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        let code: Vec<u8> = vec![
            0xB9, 0x05, 0x00, 0x00, 0x00, // mov ecx, 5      (loop counter)
            0x31, 0xD2, // xor edx, edx     (sum = 0)
            // loop:
            0x01, 0xCA, // add edx, ecx     (sum += counter)
            0xFF, 0xC9, // dec ecx
            0x75, 0xFA, // jnz loop  (rel8 = -6, back to `add edx, ecx`)
        ];
        m.write_init(CODE, &code).unwrap();
        cpu.rip = CODE;
        let exit = cpu.run(&mut m).unwrap();
        // The loop never syscalls or faults; it just runs off the end of the
        // buffer once ecx hits 0, which the harness treats as an illegal
        // fetch past the mapped code — that's fine, we only care about the
        // register state at that point.
        match exit {
            Exit::IllegalInstruction { .. } | Exit::MemFault { .. } => {}
            other => panic!("unexpected exit before the loop could fall through: {other:?}"),
        }
        assert_eq!(cpu.gpr[RDX] & 0xffff_ffff, 15, "1+2+3+4+5 == 15");
    }
}
