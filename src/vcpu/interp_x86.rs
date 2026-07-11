//! Software CPU interpreter backend for x86-64 guests — the portable,
//! no-acceleration fallback for `Arch::X86_64` (mirrors [`super::interp`]'s
//! aarch64 interpreter, but decodes variable-length x86 instructions instead
//! of fixed 4-byte ones).
//!
//! This is a scaffold, not a full x86-64 implementation: only enough of the
//! instruction set to run a trivial statically-linked ELF that does a couple
//! of syscalls. Coverage: REX-prefixed and non-REX `MOV` (reg/reg, imm→reg,
//! reg↔mem via ModRM+SIB+disp8/32, RIP-relative, `MOVABS`), `LEA`, the ALU
//! group (`ADD`/`SUB`/`AND`/`OR`/`XOR`/`CMP`/`TEST`) in register and
//! immediate forms with full flag computation (CF/ZF/SF/OF/PF), `PUSH`/`POP`
//! (register and immediate), `CALL rel32`/`RET`, `JMP rel8/rel32`, `Jcc
//! rel8/rel32` (all 16 conditions), `INC`/`DEC`/`NEG`, `SHL`/`SHR`/`SAR` by an
//! immediate or `CL`, and `SYSCALL`. Anything else surfaces as
//! [`Exit::IllegalInstruction`].

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
#[allow(dead_code)]
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
    Mem(u64),
}

fn resolve(kind: RmKind, end_pc: u64) -> Operand {
    match kind {
        RmKind::Reg(r) => Operand::Reg(r),
        RmKind::Mem(a) => Operand::Mem(a),
        RmKind::MemRip(disp) => Operand::Mem((end_pc as i64).wrapping_add(disp) as u64),
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

/// Mask `v` to the low 32 bits when `width == 32`; a no-op at `width == 64`.
const fn mask_w(v: u64, width: u32) -> u64 {
    if width == 64 { v } else { v & 0xffff_ffff }
}

/// Parity flag: `true` iff the low byte of `v` has an even number of 1 bits.
fn parity(v: u8) -> bool {
    v.count_ones().is_multiple_of(2)
}

/// The sign bit of `v` interpreted as a `width`-bit integer.
fn sign_bit(v: u64, width: u32) -> bool {
    (v >> (width - 1)) & 1 == 1
}

/// Sign-extend the low `width` bits of `v` to a full 64-bit signed value.
fn sign_extend_w(v: u64, width: u32) -> i64 {
    if width == 64 {
        v as i64
    } else {
        i64::from(v as u32 as i32)
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
            Operand::Mem(a) => {
                let n = (width / 8) as usize;
                let mut b = [0u8; 8];
                mem.read(a, &mut b[..n])
                    .map_err(|_| Step::Fault { addr: a, write: false })?;
                Ok(u64::from_le_bytes(b))
            }
        }
    }

    fn write_operand(
        &mut self,
        mem: &mut GuestMemory,
        op: Operand,
        val: u64,
        width: u32,
    ) -> Result<(), Step> {
        match op {
            Operand::Reg(r) => {
                self.gpr[r] = mask_w(val, width);
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
            Operand::Reg(_) => return Step::Illegal, // LEA requires a memory r/m
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

    /// Group 1: `0x81 /r id` and `0x83 /r ib` — ALU op, r/m and an immediate.
    fn group1_imm(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        rex: Rex,
        width: u32,
        imm8: bool,
    ) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let (imm, pc3): (i64, u64) = if imm8 {
            let (v, p) = fetch!(fetch_i8(mem, pc2));
            (i64::from(v), p)
        } else {
            let (v, p) = fetch!(fetch_i32(mem, pc2));
            (i64::from(v), p)
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
        let rm_op = resolve(modrm.kind, pc3);
        let a = fetch!(self.read_operand(mem, rm_op, width));
        let b = mask_w(imm as u64, width);
        let r = self.apply_alu(op, a, b, width);
        if op != AluOp::Cmp {
            fetch!(self.write_operand(mem, rm_op, r, width));
        }
        self.next(pc3)
    }

    /// Group 3: `0xF7 /r` — `TEST r/m, imm32` (/0) and `NEG r/m` (/3).
    fn group3(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex, width: u32) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        match modrm.reg {
            0 | 1 => {
                let (imm, pc3) = fetch!(fetch_i32(mem, pc2));
                let rm_op = resolve(modrm.kind, pc3);
                let a = fetch!(self.read_operand(mem, rm_op, width));
                self.apply_alu(AluOp::Test, a, mask_w(i64::from(imm) as u64, width), width);
                self.next(pc3)
            }
            3 => {
                let rm_op = resolve(modrm.kind, pc2);
                let a = fetch!(self.read_operand(mem, rm_op, width));
                let r = self.sub_flags(0, a, width); // NEG = 0 - a; CF = (a != 0)
                fetch!(self.write_operand(mem, rm_op, r, width));
                self.next(pc2)
            }
            _ => Step::Illegal, // NOT/MUL/IMUL/DIV/IDIV: not in our documented subset
        }
    }

    /// Group 5: `0xFF /r` — `INC r/m` (/0) and `DEC r/m` (/1).
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
            _ => Step::Illegal, // CALL/JMP/PUSH r/m (2,4,6): not in our documented subset
        }
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

    fn exec_0f(&mut self, mem: &mut GuestMemory, pc: u64) -> Step {
        let (op2, pc) = fetch!(fetch_u8(mem, pc));
        match op2 {
            0x05 => Step::Syscall, // `rip` deliberately left on the `syscall` opcode
            0x80..=0x8F => {
                let cc = op2 & 0x0f;
                let (rel, pc2) = fetch!(fetch_i32(mem, pc));
                if self.cond_holds(cc) {
                    self.jump((pc2 as i64).wrapping_add(i64::from(rel)) as u64)
                } else {
                    self.next(pc2)
                }
            }
            _ => Step::Illegal,
        }
    }

    #[allow(clippy::too_many_lines)]
    fn exec(&mut self, mem: &mut GuestMemory) -> Step {
        let (b0, pc) = fetch!(fetch_u8(mem, self.rip));
        let (rex, opcode, pc) = if (0x40..=0x4f).contains(&b0) {
            let (op, pc2) = fetch!(fetch_u8(mem, pc));
            (Rex::from_byte(b0), op, pc2)
        } else {
            (Rex::default(), b0, pc)
        };
        let width = if rex.w { 64 } else { 32 };

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
            0x01 => self.alu_rm_gv(mem, pc, rex, width, AluOp::Add, true),
            0x03 => self.alu_gv_rm(mem, pc, rex, width, AluOp::Add),
            0x09 => self.alu_rm_gv(mem, pc, rex, width, AluOp::Or, true),
            0x0B => self.alu_gv_rm(mem, pc, rex, width, AluOp::Or),
            0x21 => self.alu_rm_gv(mem, pc, rex, width, AluOp::And, true),
            0x23 => self.alu_gv_rm(mem, pc, rex, width, AluOp::And),
            0x29 => self.alu_rm_gv(mem, pc, rex, width, AluOp::Sub, true),
            0x2B => self.alu_gv_rm(mem, pc, rex, width, AluOp::Sub),
            0x31 => self.alu_rm_gv(mem, pc, rex, width, AluOp::Xor, true),
            0x33 => self.alu_gv_rm(mem, pc, rex, width, AluOp::Xor),
            0x39 => self.alu_rm_gv(mem, pc, rex, width, AluOp::Cmp, false),
            0x3B => self.alu_gv_rm(mem, pc, rex, width, AluOp::Cmp),
            0x85 => self.alu_rm_gv(mem, pc, rex, width, AluOp::Test, false),
            0x81 => self.group1_imm(mem, pc, rex, width, false),
            0x83 => self.group1_imm(mem, pc, rex, width, true),
            0xF7 => self.group3(mem, pc, rex, width),
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
            0x90 => self.next(pc),
            0x0F => self.exec_0f(mem, pc),
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
}
