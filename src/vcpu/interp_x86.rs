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
//! conditions), `INC`/`DEC` (Group 4/5), `SHL`/`SHR`/`SAR`/`ROL`/`ROR` by an
//! immediate, `CL`, or the implicit-1 `D1` form, `XCHG`, the
//! `REP`/`REPE`/`REPNE`-prefixed string ops (`MOVS`/`STOS`/
//! `LODS`/`SCAS`/`CMPS`, honoring `DF` via `CLD`/`STD`), and `SYSCALL`. The
//! `0x66` operand-size prefix is decoded (16-bit width) even though most
//! flag/overflow edge cases are only exercised at 32/64-bit widths; the
//! `0x67` address-size prefix truncates effective addresses to 32 bits; the
//! `fs:` segment override adds the `arch_prctl(ARCH_SET_FS)`-set base to
//! effective addresses (how x86-64 TLS and glibc's stack canary are reached),
//! while the long-mode zero-based overrides (`cs`/`ds`/`es`/`ss`/`gs`) are
//! consumed as no-ops; `ENDBR64`/`ENDBR32` and the multi-byte `0F 1F` NOP
//! (gcc's default function padding/CET landing pads) execute as NOPs.
//!
//! Also covers SSE/SSE2: a 16-entry `xmm` register file plus `MOVSS`/`MOVSD`/
//! `MOVAPS`/`MOVUPS`/`MOVAPD`/`MOVUPD`/`MOVDQA`/`MOVDQU`/`MOVD`/`MOVQ`,
//! scalar `ADDSD`/`SUBSD`/`MULSD`/`DIVSD`/`SQRTSD`/`MINSD`/`MAXSD` (and their
//! `SS` single-precision counterparts), packed `PS`/`PD` arithmetic sharing
//! the same opcodes, `CVTSI2SD`/`CVTSI2SS`/`CVTTSD2SI`/`CVTTSS2SI`/
//! `CVTSD2SI`/`CVTSS2SI`/`CVTSD2SS`/`CVTSS2SD`, `UCOMISD`/`COMISD`/
//! `UCOMISS`/`COMISS`, the packed-integer `PXOR`/`POR`/`PAND`/`PANDN`/
//! `PCMPEQB`/`PCMPEQD`/`PADDB`/`PSUBB`/`PADDD`/`PADDQ`/`PSUBD`/`PSUBQ`/
//! `PCMPGTB`/`PCMPGTD`/`PMINUB`/`PMAXUB`, `PMOVMSKB`, `MOVMSKPS`/
//! `MOVMSKPD`, and `XORPS`/`XORPD`/`ANDPS`/`ANDPD` (the mandatory
//! `0x66`/`0xF2`/`0xF3` prefixes are decoded as opcode-selectors here, not
//! as operand-size/`REP`). Also: the bit-scan/count group `BSF`/`BSR`/
//! `POPCNT`/`LZCNT`/`TZCNT`, the `BT`/`BTS`/`BTR`/`BTC` register and
//! immediate forms, `SHLD`/`SHRD`, `BSWAP`, and the SSE shuffle/unpack/
//! shift group `PSHUFD`/`PSHUFLW`/`PSHUFHW`/`PSHUFB`/`SHUFPS`/`SHUFPD`/
//! `UNPCKLPS`/`UNPCKHPS`/`PUNPCKL*`/`PUNPCKH*`/`PSLLDQ`/`PSRLDQ`/`PSLLD`/
//! `PSRLD`/`PSLLQ`/`PSRLQ`, and the 64-bit half-register moves
//! `MOVLPS`/`MOVHPS`/`MOVLPD`/`MOVHPD`/`MOVLHPS`/`MOVHLPS`/`MOVDDUP`/
//! `MOVSLDUP`/`MOVSHDUP`.
//!
//! Also a subset of the x87 FPU (the `D8-DF` ESC opcodes, needed since
//! musl/glibc use x87 for `long double` and some `printf`/`strtod` float
//! paths on x86-64 even though SSE2 is the default for `double`/`float`):
//! an 8-deep register stack (`FLD`/`FST`/`FSTP` for `m32`/`m64`/`m80` and
//! `ST(i)`, `FILD`/`FIST`/`FISTP` for `m16`/`m32`/`m64` integers, `FXCH`,
//! the constant loads `FLD1`/`FLDZ`/`FLDPI`/`FLDL2E`/`FLDL2T`/`FLDLG2`/
//! `FLDLN2`), arithmetic (`FADD`/`FADDP`/`FIADD`, `FSUB`/`FSUBP`/`FSUBR`/
//! `FSUBRP`, `FMUL`/`FMULP`/`FIMUL`, `FDIV`/`FDIVP`/`FDIVR`/`FDIVRP`,
//! `FABS`/`FCHS`/`FSQRT`/`FRNDINT`), compares (`FCOM`/`FCOMP`/`FCOMPP`/
//! `FUCOM`/`FUCOMP`/`FUCOMPP`/`FTST`, and `FCOMI`/`FCOMIP`/`FUCOMI`/
//! `FUCOMIP`, which set `EFLAGS` directly), and control (`FLDCW`/`FNSTCW`/
//! `FNSTSW`/`FNCLEX`/`FNINIT`/`FWAIT`/`FFREE`/`FINCSTP`/`FDECSTP`). Each
//! 80-bit `long double` register is modeled as an `f64` rather than true
//! extended precision — an accepted approximation for a software scaffold
//! (see `f80_to_f64`). Also: `CPUID`, `RDTSC`/`RDTSCP`, `RDRAND`/`RDSEED`,
//! `XGETBV`, the `LOCK` prefix (`0xF0`, decoded and otherwise ignored — this
//! interpreter is single-threaded, so every read-modify-write is already
//! atomic) alongside the `LOCK`-able ops it decorates (`XADD`, `CMPXCHG`,
//! `CMPXCHG8B`/`CMPXCHG16B`, and the existing `ADD`/`OR`/`AND`/`SUB`/`XOR`/
//! `BTS`/`BTR`/`INC`/`DEC`/`NEG`/`NOT`), `MOVNTI`, and the fence/cache-hint
//! group `LFENCE`/`SFENCE`/`MFENCE`/`CLFLUSH`/`PAUSE` (all no-ops). Anything
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
const RBX: usize = 3;
const RSP: usize = 4;
const RBP: usize = 5;
const RSI: usize = 6;
const RDI: usize = 7;
const R8: usize = 8;
const R9: usize = 9;
const R10: usize = 10;
const R11: usize = 11;

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
    Fault {
        addr: u64,
        write: bool,
    },
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

/// Where a group-2 shift/rotate takes its count from: an immediate byte
/// (`C0`/`C1`), an implicit 1 (`D0`/`D1`), or `CL` (`D2`/`D3`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum G2Count {
    Imm8,
    One,
    Cl,
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
///
/// `Copy` because the x87 dispatch (`fpu_d8`..`fpu_df`) matches on `.kind`
/// and then, in the memory-operand arm, re-resolves it via [`resolve`] —
/// cheaper to copy the two `usize`/[`RmKind`] fields than to thread a
/// reference through.
#[derive(Clone, Copy)]
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
    Adc,
    Or,
    And,
    Sub,
    Sbb,
    Xor,
    Cmp,
    Test,
}

/// SSE scalar/packed floating-point operation selected by the `0F 51/54..5F`
/// opcode group (`Sqrt` is unary; the rest read `dst op src`).
#[derive(Clone, Copy)]
enum SseOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Sqrt,
}

/// The 128-bit bitwise op selected by `ANDPS`/`XORPS`/`PAND`/`PANDN`/`POR`/
/// `PXOR` — all four opcodes (float-tagged or integer-tagged) compute the
/// same bit pattern, so one enum covers both opcode families.
#[derive(Clone, Copy)]
enum BitOp {
    And,
    Andn,
    Or,
    Xor,
}

/// The four `BT`/`BTS`/`BTR`/`BTC` variants (`0F A3/AB/B3/BB` register
/// forms, `0F BA /4../7` immediate group): all four start by copying the
/// tested bit into `CF`; only `BTS`/`BTR`/`BTC` then modify it.
#[derive(Clone, Copy)]
enum BitTestOp {
    Bt,
    Bts,
    Btr,
    Btc,
}

/// The x87 `D8-DF` ESC-opcode arithmetic/compare group, selected by the
/// ModRM `reg` field (`/0../7`) for both the memory forms (`FADD`/`FIADD`
/// m32/m64/m16/m32-int, ...) and `D8`'s `ST(0),ST(i)` register form — all
/// five opcode families (`D8`, `DA`, `DC` memory, `DE` memory) number their
/// eight sub-operations identically. `DC`/`DE`'s *register* forms (dest is
/// `ST(i)`, not `ST(0)`) instead use [`Self::from_reg_reversed`].
#[derive(Clone, Copy)]
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
    /// The straightforward `/0../7 -> Add/Mul/Com/Comp/Sub/SubR/Div/DivR`
    /// mapping used by `D8` (memory and `ST(0),ST(i)` register form), and
    /// the memory forms of `DA`/`DC`/`DE`.
    fn from_reg(reg: u8) -> Self {
        match reg & 7 {
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

    /// `DC`/`DE`'s register form writes `ST(i)` (not `ST(0)`) and reads
    /// `ST(0)` as the other operand, so the non-commutative pair `SUB`/
    /// `SUBR` (and `DIV`/`DIVR`) trade places relative to [`Self::from_reg`]
    /// — e.g. `DC E0+i` disassembles as `FSUBR ST(i), ST(0)`, not `FSUB`.
    /// `Com`/`Comp` (`/2`/`/3`) have no defined register-destination form in
    /// this range.
    fn from_reg_reversed(reg: u8) -> Option<Self> {
        Some(match reg & 7 {
            0 => Self::Add,
            1 => Self::Mul,
            4 => Self::SubR,
            5 => Self::Sub,
            6 => Self::DivR,
            7 => Self::Div,
            _ => return None,
        })
    }
}

/// Apply an arithmetic [`FpuOp`] (`Com`/`Comp` never reach here — callers
/// special-case compares before calling this) to `dst OP src`; the `R`
/// ("reversed") variants swap the operand order (`FSUBR`/`FDIVR` compute
/// `src - dst`/`src / dst`).
fn fpu_binop(op: FpuOp, dst: f64, src: f64) -> f64 {
    match op {
        FpuOp::Add => dst + src,
        FpuOp::Mul => dst * src,
        FpuOp::Sub => dst - src,
        FpuOp::SubR => src - dst,
        FpuOp::Div => dst / src,
        FpuOp::DivR => src / dst,
        FpuOp::Com | FpuOp::Comp => dst, // unreachable in practice; see doc comment above
    }
}

/// The memory operand width for the x87 arithmetic group's non-register
/// (`mod != 3`) forms: `D8` uses `F32`, `DC` uses `F64`, `DA` uses `I32`
/// (`FIADD`/... m32int), `DE` uses `I16` (m16int).
#[derive(Clone, Copy)]
enum MemWidth {
    F32,
    F64,
    I16,
    I32,
}

/// Decode an x87 80-bit extended-precision value (`m80fp`: 64-bit
/// significand with an explicit integer bit, then a 16-bit sign+exponent) to
/// the nearest `f64`. This interpreter models the x87 register stack with
/// `f64` throughout rather than true 80-bit precision — an accepted
/// approximation (see [`X86Interp::st`]) — so this conversion, and its
/// [`f64_to_f80_bytes`] inverse, are the only places 80-bit precision is
/// even nominally in play, and both narrow through `f64` immediately.
#[allow(clippy::cast_precision_loss)] // mantissa/2^63 is exactly the extended-precision significand formula
fn f80_to_f64(mantissa: u64, exp: u16, sign: bool) -> f64 {
    let value = if exp == 0 && mantissa == 0 {
        0.0
    } else if exp == 0x7fff {
        if mantissa == (1u64 << 63) {
            f64::INFINITY
        } else {
            f64::NAN
        }
    } else {
        let m = mantissa as f64 / (1u64 << 63) as f64;
        m * 2f64.powi(i32::from(exp) - 16383)
    };
    if sign { -value } else { value }
}

/// Encode an `f64` as the 10-byte x87 extended-precision (`m80fp`) format —
/// see `f80_to_f64`. Finite normal and subnormal `f64` values, `0`, `±inf`
/// and `NaN` are all handled exactly (mapping through the nearest `f64`, per
/// the same accepted approximation).
fn f64_to_f80_bytes(v: f64) -> [u8; 10] {
    let bits = v.to_bits();
    let sign = bits >> 63;
    let biased_exp = (bits >> 52) & 0x7ff;
    let frac = bits & 0x000f_ffff_ffff_ffff;
    let (mantissa, exp): (u64, u64) = if biased_exp == 0x7ff {
        if frac == 0 {
            (1u64 << 63, 0x7fff)
        } else {
            (0xC000_0000_0000_0000, 0x7fff)
        }
    } else if biased_exp == 0 {
        if frac == 0 {
            (0, 0)
        } else {
            // Subnormal f64: normalize by shifting the fraction so its
            // leading set bit lands at bit 63 (the extended format's
            // explicit integer bit), adjusting the exponent to match.
            let shift = frac.leading_zeros();
            let mantissa = frac << shift;
            let unbiased = -1011i64 - i64::from(shift);
            (mantissa, (unbiased + 16383) as u64)
        }
    } else {
        let mantissa = (1u64 << 63) | (frac << 11);
        let unbiased = biased_exp as i64 - 1023;
        (mantissa, (unbiased + 16383) as u64)
    };
    let mut out = [0u8; 10];
    out[0..8].copy_from_slice(&mantissa.to_le_bytes());
    let exp16 = (exp as u16) | ((sign as u16) << 15);
    out[8..10].copy_from_slice(&exp16.to_le_bytes());
    out
}

// ---- x87 memory operand read/write. Free functions (not `X86Interp`
// methods, unlike the GPR/XMM `read_operand`/`xmm_read128` family) since
// none of them touch FPU register-file state — only the ModRM/opcode
// dispatch in `fpu_d8`..`fpu_df` needs `self`. ----

fn fpu_read_f32(mem: &GuestMemory, addr: u64) -> Result<f64, Step> {
    let mut b = [0u8; 4];
    mem.read(addr, &mut b)
        .map_err(|_| Step::Fault { addr, write: false })?;
    Ok(f64::from(f32::from_le_bytes(b)))
}

#[allow(clippy::cast_possible_truncation)] // FST/FSTP m32fp is exactly this narrowing
fn fpu_write_f32(mem: &mut GuestMemory, addr: u64, v: f64) -> Result<(), Step> {
    let bytes = (v as f32).to_le_bytes();
    mem.write_trap(addr, &bytes).map_err(|e| Step::Fault {
        addr: e.fault_addr(),
        write: true,
    })
}

fn fpu_read_f64(mem: &GuestMemory, addr: u64) -> Result<f64, Step> {
    let mut b = [0u8; 8];
    mem.read(addr, &mut b)
        .map_err(|_| Step::Fault { addr, write: false })?;
    Ok(f64::from_le_bytes(b))
}

fn fpu_write_f64(mem: &mut GuestMemory, addr: u64, v: f64) -> Result<(), Step> {
    mem.write_trap(addr, &v.to_le_bytes())
        .map_err(|e| Step::Fault {
            addr: e.fault_addr(),
            write: true,
        })
}

fn fpu_read_f80(mem: &GuestMemory, addr: u64) -> Result<f64, Step> {
    let mut b = [0u8; 10];
    mem.read(addr, &mut b)
        .map_err(|_| Step::Fault { addr, write: false })?;
    let mantissa = u64::from_le_bytes(b[0..8].try_into().unwrap());
    let se = u16::from_le_bytes(b[8..10].try_into().unwrap());
    Ok(f80_to_f64(mantissa, se & 0x7fff, se & 0x8000 != 0))
}

fn fpu_write_f80(mem: &mut GuestMemory, addr: u64, v: f64) -> Result<(), Step> {
    mem.write_trap(addr, &f64_to_f80_bytes(v))
        .map_err(|e| Step::Fault {
            addr: e.fault_addr(),
            write: true,
        })
}

/// `FILD`/`FIADD`/`FICOM`/... source: a `width`-bit two's-complement
/// integer in memory, sign-extended.
fn fpu_read_int(mem: &GuestMemory, addr: u64, width: u32) -> Result<i64, Step> {
    let n = (width / 8) as usize;
    let mut b = [0u8; 8];
    mem.read(addr, &mut b[..n])
        .map_err(|_| Step::Fault { addr, write: false })?;
    Ok(sign_extend_w(u64::from_le_bytes(b), width))
}

/// `FIST`/`FISTP` destination: truncate `val` (already rounded per
/// [`X86Interp::round_per_cw`] by the caller) to `width` bits.
fn fpu_write_int(mem: &mut GuestMemory, addr: u64, val: i64, width: u32) -> Result<(), Step> {
    let n = (width / 8) as usize;
    let bytes = (val as u64).to_le_bytes();
    mem.write_trap(addr, &bytes[..n]).map_err(|e| Step::Fault {
        addr: e.fault_addr(),
        write: true,
    })
}

/// The `D8`/`DA`/`DC`/`DE` memory-form arithmetic source, at the width `w`
/// dictates ([`MemWidth`]).
#[allow(clippy::cast_precision_loss)] // FILD's int->f64 load is exactly this
fn fpu_read_src(mem: &GuestMemory, addr: u64, w: MemWidth) -> Result<f64, Step> {
    match w {
        MemWidth::F32 => fpu_read_f32(mem, addr),
        MemWidth::F64 => fpu_read_f64(mem, addr),
        MemWidth::I16 => fpu_read_int(mem, addr, 16).map(|v| v as f64),
        MemWidth::I32 => fpu_read_int(mem, addr, 32).map(|v| v as f64),
    }
}

/// Apply `f` lane-wise (`f64`) to the low `lanes` 8-byte lanes of `dst`/`src`,
/// leaving the untouched upper lanes of `dst` as-is — this is exactly the
/// "scalar op preserves the destination's upper bits" rule SSE arithmetic
/// follows (`lanes == 1`), generalized to the packed form (`lanes == 2`).
#[allow(clippy::many_single_char_names)] // dst/src/lanes/f is the natural naming for a lane op
fn f64_lane_binop(dst: u128, src: u128, lanes: usize, f: impl Fn(f64, f64) -> f64) -> u128 {
    let (d, s) = (dst.to_le_bytes(), src.to_le_bytes());
    let mut out = d;
    for i in 0..lanes {
        let o = i * 8;
        let a = f64::from_le_bytes(d[o..o + 8].try_into().unwrap());
        let b = f64::from_le_bytes(s[o..o + 8].try_into().unwrap());
        out[o..o + 8].copy_from_slice(&f(a, b).to_le_bytes());
    }
    u128::from_le_bytes(out)
}

/// Unary counterpart of [`f64_lane_binop`] (e.g. `SQRTSD`/`SQRTPD`): each
/// touched lane comes from `src`, not from combining it with `dst`.
#[allow(clippy::many_single_char_names)] // dst/src/lanes/f is the natural naming for a lane op
fn f64_lane_unop(dst: u128, src: u128, lanes: usize, f: impl Fn(f64) -> f64) -> u128 {
    let (d, s) = (dst.to_le_bytes(), src.to_le_bytes());
    let mut out = d;
    for i in 0..lanes {
        let o = i * 8;
        let b = f64::from_le_bytes(s[o..o + 8].try_into().unwrap());
        out[o..o + 8].copy_from_slice(&f(b).to_le_bytes());
    }
    u128::from_le_bytes(out)
}

/// `f32` counterpart of [`f64_lane_binop`] (4-byte lanes, up to 4 of them for
/// the packed `PS` forms).
#[allow(clippy::many_single_char_names)] // dst/src/lanes/f is the natural naming for a lane op
fn f32_lane_binop(dst: u128, src: u128, lanes: usize, f: impl Fn(f32, f32) -> f32) -> u128 {
    let (d, s) = (dst.to_le_bytes(), src.to_le_bytes());
    let mut out = d;
    for i in 0..lanes {
        let o = i * 4;
        let a = f32::from_le_bytes(d[o..o + 4].try_into().unwrap());
        let b = f32::from_le_bytes(s[o..o + 4].try_into().unwrap());
        out[o..o + 4].copy_from_slice(&f(a, b).to_le_bytes());
    }
    u128::from_le_bytes(out)
}

/// `f32` counterpart of [`f64_lane_unop`].
#[allow(clippy::many_single_char_names)] // dst/src/lanes/f is the natural naming for a lane op
fn f32_lane_unop(dst: u128, src: u128, lanes: usize, f: impl Fn(f32) -> f32) -> u128 {
    let (d, s) = (dst.to_le_bytes(), src.to_le_bytes());
    let mut out = d;
    for i in 0..lanes {
        let o = i * 4;
        let b = f32::from_le_bytes(s[o..o + 4].try_into().unwrap());
        out[o..o + 4].copy_from_slice(&f(b).to_le_bytes());
    }
    u128::from_le_bytes(out)
}

/// Whether an SSE compare predicate (`CMPPS`/`CMPSD`/… imm8, low 3 bits) holds
/// for a float pair whose ordering is `ord` (`None` = unordered, i.e. a NaN
/// operand — which the "not"-forms 4–6 and `UNORD` 3 treat as true).
fn cmp_pred_holds(pred: u8, ord: Option<std::cmp::Ordering>) -> bool {
    use std::cmp::Ordering::{Equal, Less};
    match pred & 7 {
        0 => ord == Some(Equal),
        1 => ord == Some(Less),
        2 => matches!(ord, Some(Less | Equal)),
        3 => ord.is_none(),
        4 => ord != Some(Equal),
        5 => ord != Some(Less),
        6 => !matches!(ord, Some(Less | Equal)),
        _ => ord.is_some(),
    }
}

/// `CMPPD`/`CMPSD` (`0F C2` with a `0x66`/`0xF2` prefix): per double lane, write
/// an all-ones mask when `pred` holds, else all-zeros. Lanes past `lanes` keep
/// `dst` (so the scalar `CMPSD` form preserves the high quadword).
fn f64_lane_cmp(dst: u128, src: u128, lanes: usize, pred: u8) -> u128 {
    let (d, s) = (dst.to_le_bytes(), src.to_le_bytes());
    let mut out = d;
    for i in 0..lanes {
        let o = i * 8;
        let a = f64::from_le_bytes(d[o..o + 8].try_into().unwrap());
        let b = f64::from_le_bytes(s[o..o + 8].try_into().unwrap());
        let hit = cmp_pred_holds(pred, a.partial_cmp(&b));
        out[o..o + 8].copy_from_slice(&(if hit { u64::MAX } else { 0 }).to_le_bytes());
    }
    u128::from_le_bytes(out)
}

/// `CMPPS`/`CMPSS` (`0F C2` with no prefix or `0xF3`): the single-precision
/// counterpart of [`f64_lane_cmp`].
fn f32_lane_cmp(dst: u128, src: u128, lanes: usize, pred: u8) -> u128 {
    let (d, s) = (dst.to_le_bytes(), src.to_le_bytes());
    let mut out = d;
    for i in 0..lanes {
        let o = i * 4;
        let a = f32::from_le_bytes(d[o..o + 4].try_into().unwrap());
        let b = f32::from_le_bytes(s[o..o + 4].try_into().unwrap());
        let hit = cmp_pred_holds(pred, a.partial_cmp(&b));
        out[o..o + 4].copy_from_slice(&(if hit { u32::MAX } else { 0 }).to_le_bytes());
    }
    u128::from_le_bytes(out)
}

/// Zero-extend up to 8 little-endian bytes into a `u64` — a lane-width-
/// generic byte read for the packed-integer lane ops below.
fn u64_from_le(bytes: &[u8]) -> u64 {
    let mut b = [0u8; 8];
    b[..bytes.len()].copy_from_slice(bytes);
    u64::from_le_bytes(b)
}

/// The interleave shared by `PUNPCKL*`/`PUNPCKH*` (`66 0F 60/61/62/68/69/
/// 6A/6C/6D`) and `UNPCKLPS`/`UNPCKHPS`/`UNPCKLPD`/`UNPCKHPD` (`0F 14/15`,
/// `66 0F 14/15`): merge alternating `lane_bytes`-wide lanes from the low
/// (`high == false`) or high (`high == true`) half of `dst`/`src` into one
/// interleaved result. The float-tagged and integer-tagged opcodes that
/// share a lane width compute an identical bit pattern, so one function
/// covers both.
fn unpck(dst: u128, src: u128, lane_bytes: usize, high: bool) -> u128 {
    let (d, s) = (dst.to_le_bytes(), src.to_le_bytes());
    let half = 16 / lane_bytes / 2;
    let base = if high { half } else { 0 };
    let mut out = [0u8; 16];
    for i in 0..half {
        let src_off = (base + i) * lane_bytes;
        let o0 = (2 * i) * lane_bytes;
        let o1 = (2 * i + 1) * lane_bytes;
        out[o0..o0 + lane_bytes].copy_from_slice(&d[src_off..src_off + lane_bytes]);
        out[o1..o1 + lane_bytes].copy_from_slice(&s[src_off..src_off + lane_bytes]);
    }
    u128::from_le_bytes(out)
}

/// `PACKSSWB`/`PACKUSWB`/`PACKSSDW` (`0F 63`/`0F 67`/`0F 6B`): narrow each
/// signed `in_bytes`-wide lane of `dst` then `src` to a saturated half-width
/// lane, writing `dst`'s lanes to the low half of the result and `src`'s to the
/// high half. `signed_out` selects signed saturation (`PACKSS`) vs unsigned
/// (`PACKUS`).
fn pack128(dst: u128, src: u128, in_bytes: usize, signed_out: bool) -> u128 {
    let lanes = 16 / in_bytes; // input lanes per operand
    let out_bytes = in_bytes / 2;
    let read_lane = |bytes: &[u8; 16], i: usize| -> i64 {
        let o = i * in_bytes;
        if in_bytes == 2 {
            i64::from(i16::from_le_bytes([bytes[o], bytes[o + 1]]))
        } else {
            i64::from(i32::from_le_bytes(bytes[o..o + 4].try_into().unwrap()))
        }
    };
    let clamp = |v: i64| -> i64 {
        match (out_bytes, signed_out) {
            (1, true) => v.clamp(-128, 127),
            (1, false) => v.clamp(0, 255),
            (_, true) => v.clamp(-32768, 32767),
            (_, false) => v.clamp(0, 65535),
        }
    };
    let (d, s) = (dst.to_le_bytes(), src.to_le_bytes());
    let mut out = [0u8; 16];
    for (half, base) in [(&d, 0usize), (&s, lanes)] {
        for i in 0..lanes {
            let c = clamp(read_lane(half, i));
            let oo = (base + i) * out_bytes;
            if out_bytes == 1 {
                out[oo] = c as u8;
            } else {
                out[oo..oo + 2].copy_from_slice(&(c as u16).to_le_bytes());
            }
        }
    }
    u128::from_le_bytes(out)
}

/// Logical right-shift each `lane_bits`-wide lane of `v` independently by
/// `count` bits (`PSRLD`/`PSRLQ`), zeroing a lane outright once `count`
/// reaches its width — packed shifts saturate rather than wrap, unlike a
/// scalar shift.
fn pack_shift_right(v: u128, lane_bits: u32, count: u32) -> u128 {
    if count >= lane_bits {
        return 0;
    }
    let mask = (1u128 << lane_bits) - 1;
    let lanes = 128 / lane_bits;
    let mut out = 0u128;
    for i in 0..lanes {
        let lane = (v >> (i * lane_bits)) & mask;
        out |= (lane >> count) << (i * lane_bits);
    }
    out
}

/// Left-shift counterpart of [`pack_shift_right`] (`PSLLD`/`PSLLQ`).
/// Arithmetic (sign-propagating) right shift of each `lane_bits`-wide lane —
/// `PSRAW`/`PSRAD`. A count at or past the lane width saturates to a lane full
/// of the sign bit, as hardware does.
fn pack_shift_arith_right(v: u128, lane_bits: u32, count: u32) -> u128 {
    let c = count.min(lane_bits - 1);
    let mask = (1u128 << lane_bits) - 1;
    let up = 128 - lane_bits;
    let lanes = 128 / lane_bits;
    let mut out = 0u128;
    for i in 0..lanes {
        let lane = (v >> (i * lane_bits)) & mask;
        // Sign-extend the lane to the full width, shift, then re-mask.
        let signed = ((lane << up) as i128) >> up;
        out |= ((signed >> c) as u128 & mask) << (i * lane_bits);
    }
    out
}

fn pack_shift_left(v: u128, lane_bits: u32, count: u32) -> u128 {
    if count >= lane_bits {
        return 0;
    }
    let mask = (1u128 << lane_bits) - 1;
    let lanes = 128 / lane_bits;
    let mut out = 0u128;
    for i in 0..lanes {
        let lane = (v >> (i * lane_bits)) & mask;
        out |= ((lane << count) & mask) << (i * lane_bits);
    }
    out
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
    mem.read(pc, &mut b).map_err(|_| Step::Fault {
        addr: pc,
        write: false,
    })?;
    Ok((b[0], pc + 1))
}

fn fetch_i8(mem: &GuestMemory, pc: u64) -> Result<(i8, u64), Step> {
    let (b, next) = fetch_u8(mem, pc)?;
    Ok((b as i8, next))
}

fn fetch_u16(mem: &GuestMemory, pc: u64) -> Result<(u16, u64), Step> {
    let mut b = [0u8; 2];
    mem.read(pc, &mut b).map_err(|_| Step::Fault {
        addr: pc,
        write: false,
    })?;
    Ok((u16::from_le_bytes(b), pc + 2))
}

fn fetch_i16(mem: &GuestMemory, pc: u64) -> Result<(i16, u64), Step> {
    let (v, next) = fetch_u16(mem, pc)?;
    Ok((v as i16, next))
}

fn fetch_u32(mem: &GuestMemory, pc: u64) -> Result<(u32, u64), Step> {
    let mut b = [0u8; 4];
    mem.read(pc, &mut b).map_err(|_| Step::Fault {
        addr: pc,
        write: false,
    })?;
    Ok((u32::from_le_bytes(b), pc + 4))
}

fn fetch_i32(mem: &GuestMemory, pc: u64) -> Result<(i32, u64), Step> {
    let (v, next) = fetch_u32(mem, pc)?;
    Ok((v as i32, next))
}

fn fetch_u64(mem: &GuestMemory, pc: u64) -> Result<(u64, u64), Step> {
    let mut b = [0u8; 8];
    mem.read(pc, &mut b).map_err(|_| Step::Fault {
        addr: pc,
        write: false,
    })?;
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
#[allow(clippy::struct_excessive_bools)] // df + the x87 C0-C3 condition codes are each independently meaningful flags, not a state machine
struct X86Interp {
    /// rax..r15, in the standard ModRM/REX numbering.
    gpr: [u64; 16],
    /// xmm0..xmm15, in the standard ModRM/REX numbering (extended the same
    /// way as `gpr` via `REX.R`/`REX.B`).
    xmm: [u128; 16],
    rip: u64,
    flags: Flags,
    /// The direction flag: `false` (`CLD`) advances string-op pointers
    /// upward, `true` (`STD`) advances them downward.
    df: bool,
    /// FS.base, set by `arch_prctl(ARCH_SET_FS, ...)` (thread pointer).
    fs_base: u64,
    /// The `0x67` address-size prefix on the instruction being executed:
    /// effective addresses truncate to 32 bits. Transient — reset at each
    /// `exec` and set by the prefix loop. (gcc also emits `0x67` as pure
    /// padding on `call` in glibc's `_start`, where it affects nothing.)
    addr32: bool,
    /// Segment base of the instruction being executed — nonzero only under an
    /// `fs:` override (`0x64`, how x86-64 reaches TLS: `mov %fs:0x28, ...` is
    /// every stack-canary check). Added to computed effective addresses in
    /// [`X86Interp::decode_modrm`]. Transient, like `addr32`. In long mode
    /// CS/DS/ES/SS (and our never-written GS) are zero-based, so their
    /// override prefixes are consumed with no effect.
    seg_base: u64,
    /// The x87 register stack, `ST(0)..ST(7)`, physically indexed (i.e. not
    /// yet rotated by `fpu_top`) — see [`X86Interp::st_get`]. Real x87
    /// registers are 80-bit extended precision; this interpreter models
    /// each as an `f64` instead (an accepted approximation for a software
    /// scaffold — see `f80_to_f64`), so values round-tripped through the
    /// register stack lose precision beyond `f64`'s ~15-17 significant
    /// digits relative to true 80-bit `long double`.
    st: [f64; 8],
    /// The status word's `TOP` field: `ST(i)` physically lives at
    /// `st[(fpu_top + i) & 7]`. `FLD`-family pushes decrement it (then
    /// write the new `ST(0)`); pops increment it.
    fpu_top: u8,
    /// Status-word condition codes `C0`/`C1`/`C2`/`C3`, set by compares
    /// (`FCOM`/`FUCOM`/`FTST`/...) and read back by `FNSTSW`.
    fpu_c0: bool,
    fpu_c1: bool,
    fpu_c2: bool,
    fpu_c3: bool,
    /// The control word (`FLDCW`/`FNSTCW`): only the rounding-control field
    /// (bits 10-11) is consulted, by [`X86Interp::round_per_cw`] (`FIST`/
    /// `FISTP`/`FRNDINT`); the precision-control and exception-mask fields
    /// are stored and read back verbatim but otherwise unused, since this
    /// interpreter doesn't model FPU exceptions at all.
    fpu_cw: u16,
    /// Free-running counter behind `RDTSC`/`RDTSCP` — see
    /// [`X86Interp::rdtsc_tick`].
    tsc: u64,
    /// PRNG state behind `RDRAND`/`RDSEED` — see
    /// [`X86Interp::rdrand_or_seed`].
    prng: u64,
    /// The SSE control/status register (`LDMXCSR`/`STMXCSR`). Stored and
    /// reloaded verbatim; this interpreter always computes in the default
    /// round-to-nearest, exceptions-masked mode, so the value only round-trips.
    mxcsr: u32,
}

impl X86Interp {
    fn new(entry: u64, stack: u64) -> Self {
        let mut gpr = [0u64; 16];
        gpr[RSP] = stack;
        Self {
            gpr,
            xmm: [0u128; 16],
            rip: entry,
            flags: Flags::default(),
            df: false,
            fs_base: 0,
            addr32: false,
            seg_base: 0,
            st: [0.0; 8],
            fpu_top: 0,
            fpu_c0: false,
            fpu_c1: false,
            fpu_c2: false,
            fpu_c3: false,
            fpu_cw: 0x037F, // the real x87's power-on/FNINIT default control word
            tsc: 0,
            prng: 0x9E37_79B9_7F4A_7C15, // arbitrary nonzero seed (golden-ratio constant)
            mxcsr: 0x1f80,               // the power-on default (all exceptions masked)
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
        mem.write_trap(sp, &val.to_le_bytes())
            .map_err(|e| Step::Fault {
                addr: e.fault_addr(),
                write: true,
            })?;
        self.gpr[RSP] = sp;
        Ok(())
    }

    fn pop(&mut self, mem: &GuestMemory) -> Result<u64, Step> {
        let sp = self.gpr[RSP];
        let mut b = [0u8; 8];
        mem.read(sp, &mut b).map_err(|_| Step::Fault {
            addr: sp,
            write: false,
        })?;
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
            return Ok((
                ModRm {
                    reg,
                    kind: RmKind::Reg(rm),
                },
                pc,
            ));
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
            let addr = self.mask_addr(
                (base_val.wrapping_add(index_val.wrapping_mul(scale)) as i64)
                    .wrapping_add(disp) as u64,
            );
            return Ok((
                ModRm {
                    reg,
                    kind: RmKind::Mem(addr),
                },
                pc,
            ));
        }

        if rm_field == 0b101 && md == 0b00 {
            // Under `addr32` this form is EIP-relative, and `resolve` (a free
            // function) can't see a pending segment base either; nothing real
            // emits these combinations (the prefixes only show up as padding
            // or with plain registers), so stay honest and fault rather than
            // resolve them wrong.
            if self.addr32 || self.seg_base != 0 {
                return Err(Step::Illegal);
            }
            let (disp, pc) = fetch_i32(mem, pc)?;
            return Ok((
                ModRm {
                    reg,
                    kind: RmKind::MemRip(i64::from(disp)),
                },
                pc,
            ));
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
        let addr = self.mask_addr((self.gpr[base] as i64).wrapping_add(disp) as u64);
        Ok((
            ModRm {
                reg,
                kind: RmKind::Mem(addr),
            },
            pc,
        ))
    }

    /// Effective address → linear address for the executing instruction:
    /// truncate to 32 bits under the `0x67` address-size prefix, then add the
    /// segment base (nonzero only under an `fs:` override — TLS access).
    fn mask_addr(&self, addr: u64) -> u64 {
        let ea = if self.addr32 { addr & 0xffff_ffff } else { addr };
        ea.wrapping_add(self.seg_base)
    }

    fn read_operand(&self, mem: &GuestMemory, op: Operand, width: u32) -> Result<u64, Step> {
        match op {
            Operand::Reg(r) => Ok(mask_w(self.gpr[r], width)),
            Operand::Reg8Hi(r) => Ok((self.gpr[r] >> 8) & 0xff),
            Operand::Mem(a) => {
                let n = (width / 8) as usize;
                let mut b = [0u8; 8];
                mem.read(a, &mut b[..n]).map_err(|_| Step::Fault {
                    addr: a,
                    write: false,
                })?;
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
                mem.write_trap(a, &bytes[..n]).map_err(|e| Step::Fault {
                    addr: e.fault_addr(),
                    write: true,
                })
            }
        }
    }

    // ---- XMM operand read/write. Separate from `read_operand`/
    // `write_operand` because those two address the GPR file (`self.gpr`);
    // an SSE `Operand::Reg(r)` instead names `self.xmm[r]`. Only the `Mem`
    // case is shared logic (re-derived here at the widths SSE needs: 32/64
    // scalar lanes and the full 128-bit `xmm/m128` forms). ----

    /// Read a 32- or 64-bit scalar lane from an SSE r/m operand (`xmm/m32`
    /// or `xmm/m64`) — the low bits of an `xmm` register, or a memory load.
    fn xmm_read_lo(&self, mem: &GuestMemory, op: Operand, width: u32) -> Result<u64, Step> {
        match op {
            Operand::Reg(r) => Ok(mask_w(self.xmm[r] as u64, width)),
            Operand::Mem(a) => {
                let n = (width / 8) as usize;
                let mut b = [0u8; 8];
                mem.read(a, &mut b[..n]).map_err(|_| Step::Fault {
                    addr: a,
                    write: false,
                })?;
                Ok(u64::from_le_bytes(b))
            }
            Operand::Reg8Hi(_) => unreachable!("SSE decode never yields an 8-bit-high operand"),
        }
    }

    /// Read a full 128-bit SSE r/m operand (`xmm/m128`).
    fn xmm_read128(&self, mem: &GuestMemory, op: Operand) -> Result<u128, Step> {
        match op {
            Operand::Reg(r) => Ok(self.xmm[r]),
            Operand::Mem(a) => {
                let mut b = [0u8; 16];
                mem.read(a, &mut b).map_err(|_| Step::Fault {
                    addr: a,
                    write: false,
                })?;
                Ok(u128::from_le_bytes(b))
            }
            Operand::Reg8Hi(_) => unreachable!("SSE decode never yields an 8-bit-high operand"),
        }
    }

    /// Write a full 128-bit value to an SSE r/m operand (`xmm/m128`).
    fn xmm_write128(&mut self, mem: &mut GuestMemory, op: Operand, val: u128) -> Result<(), Step> {
        match op {
            Operand::Reg(r) => {
                self.xmm[r] = val;
                Ok(())
            }
            Operand::Mem(a) => mem
                .write_trap(a, &val.to_le_bytes())
                .map_err(|e| Step::Fault {
                    addr: e.fault_addr(),
                    write: true,
                }),
            Operand::Reg8Hi(_) => unreachable!("SSE decode never yields an 8-bit-high operand"),
        }
    }

    // ---- flags ----

    /// `ADD` (and, with `carry_in`, `ADC`): result masked to `width`, all
    /// arithmetic flags computed *at that width* — an 8-bit `0xFF + 1` must
    /// set ZF and CF even though the value fits easily in a host integer.
    fn add_carry_flags(&mut self, a: u64, b: u64, carry_in: bool, width: u32) -> u64 {
        let m = mask_w(u64::MAX, width);
        let (a, b) = (a & m, b & m);
        let c = u64::from(carry_in);
        let full = u128::from(a) + u128::from(b) + u128::from(c);
        let r = (full as u64) & m;
        self.flags = Flags {
            cf: full > u128::from(m),
            zf: r == 0,
            sf: sign_bit(r, width),
            of: (((a ^ r) & (b ^ r)) >> (width - 1)) & 1 == 1,
            pf: parity(r as u8),
        };
        r
    }

    fn add_flags(&mut self, a: u64, b: u64, width: u32) -> u64 {
        self.add_carry_flags(a, b, false, width)
    }

    /// `SUB`/`CMP` (and, with `borrow_in`, `SBB`): width-masked result and
    /// width-accurate flags, like [`X86Interp::add_carry_flags`].
    fn sub_borrow_flags(&mut self, a: u64, b: u64, borrow_in: bool, width: u32) -> u64 {
        let m = mask_w(u64::MAX, width);
        let (a, b) = (a & m, b & m);
        let c = u64::from(borrow_in);
        let r = a.wrapping_sub(b).wrapping_sub(c) & m;
        self.flags = Flags {
            cf: u128::from(a) < u128::from(b) + u128::from(c),
            zf: r == 0,
            sf: sign_bit(r, width),
            of: (((a ^ b) & (a ^ r)) >> (width - 1)) & 1 == 1,
            pf: parity(r as u8),
        };
        r
    }

    fn sub_flags(&mut self, a: u64, b: u64, width: u32) -> u64 {
        self.sub_borrow_flags(a, b, false, width)
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
            AluOp::Adc => self.add_carry_flags(a, b, self.flags.cf, width),
            AluOp::Sub | AluOp::Cmp => self.sub_flags(a, b, width),
            AluOp::Sbb => self.sub_borrow_flags(a, b, self.flags.cf, width),
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
        // `OF` is architecturally defined only for 1-bit shifts, but real CPUs
        // (and the code V8 generates) compute it the same way for any nonzero
        // count — leaving it stale, as this interpreter used to, diverges a
        // later `jo`/`pushf`. `SHL`: sign of the result XOR the carried-out bit.
        self.flags.of = sign_bit(r, width) != cf;
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
        // `SHR`: the most-significant bit of the *original* operand, set for any
        // nonzero count (see the note in `shl_flags`).
        self.flags.of = sign_bit(a, width);
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
        // `SAR` always clears `OF` for any nonzero count (see `shl_flags`).
        self.flags.of = false;
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
        // LEA computes the *effective* address — hardware ignores a segment
        // override on it. `decode_modrm` bakes the segment base into the
        // linear address, so a nonzero base here would be silently wrong;
        // fault instead (no compiler emits `lea fs:...`).
        if self.seg_base != 0 {
            return Step::Illegal;
        }
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
        // The immediate follows the operand size: `imm16` under a `0x66`
        // prefix, else `imm32` (sign-extended to 64 bits for a `REX.W` store).
        // Reading a fixed `imm32` here mis-sized every 16-bit `mov word ptr,
        // imm16` — the two extra bytes desynced decoding and ran the CPU into
        // the middle of the next instruction.
        let (imm, pc3) = fetch!(imm_for_width(mem, pc2, width));
        let rm_op = resolve(modrm.kind, pc3);
        let val = mask_w(imm as u64, width);
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
    fn alu_gv_rm(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        rex: Rex,
        width: u32,
        op: AluOp,
    ) -> Step {
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
            2 => AluOp::Adc,
            3 => AluOp::Sbb,
            4 => AluOp::And,
            5 => AluOp::Sub,
            6 => AluOp::Xor,
            _ => AluOp::Cmp, // 7
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

    /// `op AL, imm8` / `op eAX, immz` — the accumulator-immediate short forms
    /// each ALU op reserves at `base+4`/`base+5` (plus `A8`/`A9` for TEST).
    fn alu_acc_imm(&mut self, mem: &mut GuestMemory, pc: u64, width: u32, op: AluOp) -> Step {
        let (imm, pc2): (i64, u64) = if width == 8 {
            let (v, p) = fetch!(fetch_u8(mem, pc));
            (i64::from(v), p)
        } else {
            fetch!(imm_for_width(mem, pc, width))
        };
        let a = fetch!(self.read_operand(mem, Operand::Reg(RAX), width));
        let b = mask_w(imm as u64, width);
        let r = self.apply_alu(op, a, b, width);
        if op != AluOp::Cmp && op != AluOp::Test {
            fetch!(self.write_operand(mem, Operand::Reg(RAX), r, width));
        }
        self.next(pc2)
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
    fn xchg(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        rex: Rex,
        has_rex: bool,
        width: u32,
    ) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let (rm_op, reg_op) = if width == 8 {
            (
                resolve8(modrm.kind, pc2, has_rex),
                reg8_operand(modrm.reg, has_rex),
            )
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
    fn group3(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        rex: Rex,
        has_rex: bool,
        width: u32,
    ) -> Step {
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
    /// = `rAX` * `r/m`. `CF`/`OF` flag a non-representable result; the ISA leaves
    /// `SF`/`ZF`/`PF` undefined, but real CPUs (unlike the two-operand `IMUL`,
    /// which clears `ZF`) set them from the low-half result — matching KVM so a
    /// dependent branch doesn't diverge.
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
        let lo = mask_w(self.gpr[RAX], width);
        self.flags.cf = cf;
        self.flags.of = cf;
        self.flags.zf = lo == 0;
        self.flags.sf = sign_bit(lo, width);
        self.flags.pf = parity(lo as u8);
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
    fn imul_imm(
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
            fetch!(imm_for_width(mem, pc2, width))
        };
        let rm_op = resolve(modrm.kind, pc3);
        let b = fetch!(self.read_operand(mem, rm_op, width));
        let av = sign_extend_128(u128::from(b), width);
        let bv = i128::from(imm);
        let p = av * bv;
        let cf = !fits_signed(p, width);
        let result = mask_w(p as u128 as u64, width);
        self.gpr[modrm.reg] = result;
        self.set_imul_flags(cf, result, width);
        self.next(pc3)
    }

    /// Set flags after an `IMUL`. `CF`/`OF` mark a truncated result; the Intel
    /// manual calls `SF`/`ZF`/`PF` *undefined*, but real CPUs (and thus the code
    /// V8 generates) set them deterministically — leaving them stale, as this
    /// interpreter used to, makes a `js`/`jns`/`jp` after an `imul` diverge.
    /// Matching the host CPUs KVM runs on: `SF` = the low-half result's sign,
    /// `PF` = its low byte's parity, and `ZF` is cleared even for a zero result
    /// (verified against KVM — IMUL does *not* set `ZF` from `result == 0`).
    fn set_imul_flags(&mut self, cf: bool, result: u64, width: u32) {
        self.flags.cf = cf;
        self.flags.of = cf;
        self.flags.zf = false;
        self.flags.sf = sign_bit(result, width);
        self.flags.pf = parity(result as u8);
    }

    /// Group 2 shifts: `0xC1 /r ib` (by immediate) and `0xD3 /r` (by `CL`).
    fn group2(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        rex: Rex,
        has_rex: bool,
        width: u32,
        by: G2Count,
    ) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        if matches!(modrm.reg, 2 | 3) {
            return Step::Illegal; // RCL/RCR (through-carry): not in our documented subset
        }
        let (count, pc3) = match by {
            G2Count::Cl => (self.gpr[RCX] as u8, pc2),
            G2Count::One => (1, pc2),
            G2Count::Imm8 => fetch!(fetch_u8(mem, pc2)),
        };
        let mask = if width == 64 { 63 } else { 31 };
        let amt = count & mask;
        let rm_op = if width == 8 {
            resolve8(modrm.kind, pc3, has_rex)
        } else {
            resolve(modrm.kind, pc3)
        };
        if amt == 0 {
            return self.next(pc3); // shift by 0 leaves flags and value unchanged
        }
        let a = fetch!(self.read_operand(mem, rm_op, width));
        let r = match modrm.reg {
            0 => self.rol_flags(a, amt, width),
            1 => self.ror_flags(a, amt, width),
            4 | 6 => self.shl_flags(a, amt, width), // SHL and its SAL alias
            5 => self.shr_flags(a, amt, width),
            _ => self.sar_flags(a, amt, width),
        };
        fetch!(self.write_operand(mem, rm_op, r, width));
        self.next(pc3)
    }

    /// `ROL`: rotate left within `width` bits. Unlike the shifts, rotates
    /// leave SF/ZF/PF untouched; CF gets the bit rotated across the boundary,
    /// and OF is defined only for 1-bit rotates. A count that is a multiple
    /// of the width leaves the value (and, as modeled here, the flags) alone.
    fn rol_flags(&mut self, a: u64, amt: u8, width: u32) -> u64 {
        let wa = mask_w(a, width);
        let k = u32::from(amt) % width;
        if k == 0 {
            return wa;
        }
        let r = mask_w((wa << k) | (wa >> (width - k)), width);
        self.flags.cf = r & 1 != 0;
        if amt == 1 {
            self.flags.of = ((r >> (width - 1)) & 1 != 0) ^ self.flags.cf;
        }
        r
    }

    /// `ROR`: rotate right within `width` bits (see [`X86Interp::rol_flags`]
    /// for the flag conventions).
    fn ror_flags(&mut self, a: u64, amt: u8, width: u32) -> u64 {
        let wa = mask_w(a, width);
        let k = u32::from(amt) % width;
        if k == 0 {
            return wa;
        }
        let r = mask_w((wa >> k) | (wa << (width - k)), width);
        self.flags.cf = (r >> (width - 1)) & 1 != 0;
        if amt == 1 {
            self.flags.of = self.flags.cf ^ ((r >> (width - 2)) & 1 != 0);
        }
        r
    }

    /// `BSF`/`BSR` (`0F BC`/`BD`, no mandatory prefix) and their `F3`-
    /// prefixed counterparts `TZCNT`/`LZCNT`: `Gv <- Ev`, finding the index
    /// of the least (`BSF`/`TZCNT`) or most (`BSR`/`LZCNT`) significant set
    /// bit. `BSF`/`BSR` leave `reg` unmodified when the source is zero
    /// (architecturally undefined; this matches common hardware behavior);
    /// `TZCNT`/`LZCNT` instead define the result as `width` and set `CF`.
    fn bit_scan(
        &mut self,
        mem: &GuestMemory,
        pc: u64,
        rex: Rex,
        width: u32,
        rep: u8,
        reverse: bool,
    ) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let src = fetch!(self.read_operand(mem, rm_op, width));
        let count_form = rep == 1; // F3-prefixed: TZCNT/LZCNT instead of BSF/BSR
        if src == 0 {
            self.flags = Flags {
                zf: true,
                cf: count_form,
                sf: false,
                of: false,
                pf: false,
            };
            if count_form {
                self.gpr[modrm.reg] = mask_w(u64::from(width), width);
            }
        } else {
            let lz_in_width = src.leading_zeros() - (64 - width);
            let result = if reverse {
                width - 1 - lz_in_width
            } else {
                src.trailing_zeros()
            };
            self.gpr[modrm.reg] = mask_w(u64::from(result), width);
            self.flags = Flags {
                zf: count_form && result == 0,
                cf: false,
                sf: false,
                of: false,
                pf: false,
            };
        }
        self.next(pc2)
    }

    /// `POPCNT Gv, Ev` (`F3 0F B8`, the `F3` mandatory): `reg` = the number
    /// of set bits in `r/m`; `ZF` = (result == 0), all other flags cleared.
    fn popcnt(&mut self, mem: &GuestMemory, pc: u64, rex: Rex, width: u32) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let src = fetch!(self.read_operand(mem, rm_op, width));
        let count = src.count_ones();
        self.gpr[modrm.reg] = u64::from(count);
        self.flags = Flags {
            zf: count == 0,
            cf: false,
            sf: false,
            of: false,
            pf: false,
        };
        self.next(pc2)
    }

    /// Shared body of `BT`/`BTS`/`BTR`/`BTC`: test bit `bit_idx % width` of
    /// `rm_op` into `CF`, then leave it (`Bt`), set it (`Bts`), clear it
    /// (`Btr`), or complement it (`Btc`). We always take the bit index
    /// modulo the operand width even for a memory destination (real
    /// hardware lets a register-index form address bits beyond the operand
    /// by adjusting the effective byte address; this scaffold doesn't model
    /// that).
    fn apply_bit_test(
        &mut self,
        mem: &mut GuestMemory,
        rm_op: Operand,
        width: u32,
        bit_idx: u64,
        op: BitTestOp,
    ) -> Result<(), Step> {
        let a = self.read_operand(mem, rm_op, width)?;
        let bit = (bit_idx % u64::from(width)) as u32;
        self.flags.cf = (a >> bit) & 1 == 1;
        let mask = 1u64 << bit;
        let r = match op {
            BitTestOp::Bt => return Ok(()),
            BitTestOp::Bts => a | mask,
            BitTestOp::Btr => a & !mask,
            BitTestOp::Btc => a ^ mask,
        };
        self.write_operand(mem, rm_op, r, width)
    }

    /// `BT`/`BTS`/`BTR`/`BTC Ev, Gv` (`0F A3/AB/B3/BB`): the bit index comes
    /// from a GPR.
    fn bt_ev_gv(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        rex: Rex,
        width: u32,
        op: BitTestOp,
    ) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let bit_idx = self.gpr[modrm.reg];
        fetch!(self.apply_bit_test(mem, rm_op, width, bit_idx, op));
        self.next(pc2)
    }

    /// Group 8: `BT`/`BTS`/`BTR`/`BTC Ev, ib` (`0F BA /4../7`): the bit
    /// index is an immediate byte.
    fn bt_group_imm(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex, width: u32) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let (imm, pc3) = fetch!(fetch_u8(mem, pc2));
        let op = match modrm.reg {
            4 => BitTestOp::Bt,
            5 => BitTestOp::Bts,
            6 => BitTestOp::Btr,
            7 => BitTestOp::Btc,
            _ => return Step::Illegal, // /0../3: not in our documented subset
        };
        let rm_op = resolve(modrm.kind, pc3);
        fetch!(self.apply_bit_test(mem, rm_op, width, u64::from(imm), op));
        self.next(pc3)
    }

    /// `SHLD`/`SHRD Ev, Gv, ib|CL` (`0F A4/A5`, `0F AC/AD`): a double-
    /// precision shift where the vacated bits of `dest` come from `src`
    /// rather than zeros/sign bits. `CF` is the last bit shifted out of
    /// `dest`; `OF` is only architecturally defined (a sign-change
    /// indicator) when the shift count is 1; `ZF`/`SF`/`PF` are set from the
    /// result like the ordinary shift group.
    fn shift_double(&mut self, dest: u64, src: u64, count: u32, width: u32, left: bool) -> u64 {
        let d = mask_w(dest, width);
        let s = mask_w(src, width);
        let (result, cf) = if left {
            let wide = (u128::from(d) << width) | u128::from(s);
            let result = mask_w(((wide << count) >> width) as u64, width);
            (result, (d >> (width - count)) & 1 == 1)
        } else {
            let wide = (u128::from(s) << width) | u128::from(d);
            let result = mask_w((wide >> count) as u64, width);
            (result, (d >> (count - 1)) & 1 == 1)
        };
        self.flags.cf = cf;
        self.flags.zf = result == 0;
        self.flags.sf = sign_bit(result, width);
        self.flags.pf = parity(result as u8);
        if count == 1 {
            self.flags.of = sign_bit(result, width) != sign_bit(d, width);
        }
        result
    }

    /// Decode-and-dispatch wrapper for [`Self::shift_double`]: fetches the
    /// count (`imm8` or `CL`, masked the same way as [`Self::group2`]) and
    /// leaves the r/m operand and flags untouched when it's zero.
    fn shld_shrd(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        rex: Rex,
        width: u32,
        left: bool,
        by_cl: bool,
    ) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let (count, pc3) = if by_cl {
            (self.gpr[RCX] as u8, pc2)
        } else {
            fetch!(fetch_u8(mem, pc2))
        };
        let mask = if width == 64 { 63 } else { 31 };
        let amt = count & mask;
        let rm_op = resolve(modrm.kind, pc3);
        if amt == 0 {
            return self.next(pc3);
        }
        let dest = fetch!(self.read_operand(mem, rm_op, width));
        let src = mask_w(self.gpr[modrm.reg], width);
        let r = self.shift_double(dest, src, u32::from(amt), width, left);
        fetch!(self.write_operand(mem, rm_op, r, width));
        self.next(pc3)
    }

    // ---- LOCK-prefixed atomics (XADD/CMPXCHG/CMPXCHG8B/16B — see also the
    // `0xF0` LOCK prefix itself, silently consumed alongside 0x66/0xF2/0xF3
    // in `exec`'s legacy-prefix loop) and the CPUID/RDTSC/RDRAND/XGETBV
    // family. Since this interpreter is single-threaded, a "LOCK"-prefixed
    // read-modify-write is automatically atomic — decoding the prefix and
    // running the plain op is the entire implementation; nothing here needs
    // a distinct locked/unlocked code path. ----

    /// `XADD Eb,Gb` / `Ev,Gv` (`0F C0`/`C1`): `reg` gets the *old* value of
    /// the destination, and the destination becomes `dest + reg` — flags are
    /// set exactly as for `ADD dest, reg`.
    fn xadd(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        rex: Rex,
        has_rex: bool,
        width: u32,
    ) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let (rm_op, reg_op) = if width == 8 {
            (
                resolve8(modrm.kind, pc2, has_rex),
                reg8_operand(modrm.reg, has_rex),
            )
        } else {
            (resolve(modrm.kind, pc2), Operand::Reg(modrm.reg))
        };
        let dest = fetch!(self.read_operand(mem, rm_op, width));
        let src = fetch!(self.read_operand(mem, reg_op, width));
        let sum = self.add_flags(dest, src, width);
        fetch!(self.write_operand(mem, reg_op, dest, width)); // reg <- old dest
        fetch!(self.write_operand(mem, rm_op, sum, width)); // dest <- dest + src
        self.next(pc2)
    }

    /// `CMPXCHG Eb,Gb` / `Ev,Gv` (`0F B0`/`B1`): compare `AL`/`rAX` against
    /// the destination (setting flags like `CMP`); on a match, `ZF=1` and
    /// `dest <- reg`, otherwise `ZF=0` and `AL`/`rAX <- dest`.
    fn cmpxchg(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        rex: Rex,
        has_rex: bool,
        width: u32,
    ) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let (rm_op, reg_op) = if width == 8 {
            (
                resolve8(modrm.kind, pc2, has_rex),
                reg8_operand(modrm.reg, has_rex),
            )
        } else {
            (resolve(modrm.kind, pc2), Operand::Reg(modrm.reg))
        };
        let dest = fetch!(self.read_operand(mem, rm_op, width));
        let acc = fetch!(self.read_operand(mem, Operand::Reg(RAX), width));
        self.sub_flags(acc, dest, width); // CMP acc, dest
        if acc == dest {
            let src = fetch!(self.read_operand(mem, reg_op, width));
            fetch!(self.write_operand(mem, rm_op, src, width));
        } else {
            fetch!(self.write_operand(mem, Operand::Reg(RAX), dest, width));
        }
        self.next(pc2)
    }

    /// `CMPXCHG8B`/`CMPXCHG16B m64/m128` (`0F C7 /1`; `REX.W` selects the
    /// 16-byte form). The r/m operand must be memory (the register form is
    /// `#UD` on real hardware). Only `ZF` is architecturally defined by this
    /// instruction; the other flags are left untouched.
    fn cmpxchg8b16b(&mut self, mem: &mut GuestMemory, modrm: ModRm, pc2: u64, rex: Rex) -> Step {
        let Operand::Mem(addr) = resolve(modrm.kind, pc2) else {
            return Step::Illegal; // register r/m: not a valid encoding
        };
        if rex.w {
            // CMPXCHG16B: compare RDX:RAX against the 128-bit value at
            // [addr] (two 64-bit halves, since `read_operand` tops out at 64 bits).
            let lo = fetch!(self.read_operand(mem, Operand::Mem(addr), 64));
            let hi = fetch!(self.read_operand(mem, Operand::Mem(addr.wrapping_add(8)), 64));
            let cur = (u128::from(hi) << 64) | u128::from(lo);
            let expect = (u128::from(self.gpr[RDX]) << 64) | u128::from(self.gpr[RAX]);
            if cur == expect {
                let new_lo = self.gpr[RBX];
                let new_hi = self.gpr[RCX];
                fetch!(self.write_operand(mem, Operand::Mem(addr), new_lo, 64));
                fetch!(self.write_operand(mem, Operand::Mem(addr.wrapping_add(8)), new_hi, 64));
                self.flags.zf = true;
            } else {
                self.gpr[RAX] = lo;
                self.gpr[RDX] = hi;
                self.flags.zf = false;
            }
        } else {
            // CMPXCHG8B: a single 64-bit memory read/write *is* the
            // EDX:EAX-shaped comparand (EAX low 32 bits, EDX high 32 bits).
            let cur = fetch!(self.read_operand(mem, Operand::Mem(addr), 64));
            let expect = (mask_w(self.gpr[RDX], 32) << 32) | mask_w(self.gpr[RAX], 32);
            if cur == expect {
                let new = (mask_w(self.gpr[RCX], 32) << 32) | mask_w(self.gpr[RBX], 32);
                fetch!(self.write_operand(mem, Operand::Mem(addr), new, 64));
                self.flags.zf = true;
            } else {
                self.gpr[RAX] = mask_w(cur, 32);
                self.gpr[RDX] = mask_w(cur >> 32, 32);
                self.flags.zf = false;
            }
        }
        self.next(pc2)
    }

    /// `RDRAND`/`RDSEED Rv` (`0F C7 /6`/`/7`, register-only — the memory
    /// form is a different, unrelated instruction under other prefixes and
    /// isn't implemented). Advances a simple deterministic PRNG (there's no
    /// host RNG in this scaffold) and writes the result to the destination,
    /// always reporting success (`CF=1`); `OF`/`SF`/`ZF`/`PF` are cleared,
    /// matching the real instructions' defined behavior.
    fn rdrand_or_seed(
        &mut self,
        mem: &mut GuestMemory,
        modrm: ModRm,
        pc2: u64,
        width: u32,
    ) -> Step {
        let RmKind::Reg(r) = modrm.kind else {
            return Step::Illegal;
        };
        // A splitmix64-style step: cheap, deterministic, and good enough to
        // not look like "always the same value" across successive calls.
        self.prng = self.prng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.prng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        fetch!(self.write_operand(mem, Operand::Reg(r), mask_w(z, width), width));
        self.flags = Flags {
            cf: true,
            zf: false,
            sf: false,
            of: false,
            pf: false,
        };
        self.next(pc2)
    }

    /// Group 9 (`0F C7 /r`): `CMPXCHG8B`/`CMPXCHG16B` (`/1`) and `RDRAND`/
    /// `RDSEED` (`/6`/`/7`).
    fn group9_c7(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex, width: u32) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        match modrm.reg {
            1 => self.cmpxchg8b16b(mem, modrm, pc2, rex),
            6 | 7 => self.rdrand_or_seed(mem, modrm, pc2, width),
            // VMPTRLD/VMCLEAR/VMXON/VMPTRST (/4, /6, /7 under other
            // mandatory prefixes) and /0,/2,/3,/5: not in our documented
            // subset. (`rdrand_or_seed` itself rejects a memory r/m, so a
            // 66/F3-prefixed VMX instruction that happens to hit /6 or /7
            // still correctly surfaces as illegal rather than misfiring.)
            _ => Step::Illegal,
        }
    }

    /// `MOVNTI Md,Gd`/`Mq,Gq` (`0F C3`, memory-only — the register form is
    /// `#UD`): a non-temporal store, modeled as an ordinary one (this
    /// scaffold has no cache to bypass).
    fn movnti(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex, width: u32) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op @ Operand::Mem(_) = resolve(modrm.kind, pc2) else {
            return Step::Illegal; // register r/m: not a valid encoding
        };
        let val = mask_w(self.gpr[modrm.reg], width);
        fetch!(self.write_operand(mem, rm_op, val, width));
        self.next(pc2)
    }

    /// Group 15 (`0F AE /r`): the fence/cache-hint forms are no-ops
    /// (`LFENCE`/`MFENCE`/`SFENCE` register `/5`/`/6`/`/7`, `CLFLUSH` memory
    /// `/7`), and `LDMXCSR`/`STMXCSR` (memory `/2`/`/3`) load/store the SSE
    /// control word — V8's JIT saves and restores it around float code.
    /// `FXSAVE`/`FXRSTOR`/`XSAVE*` still aren't modeled.
    fn group15_ae(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        match (modrm.kind, modrm.reg) {
            // LFENCE/MFENCE/SFENCE (register form) / CLFLUSH (memory form).
            (RmKind::Reg(_), 5..=7) | (RmKind::Mem(_) | RmKind::MemRip(_), 7) => self.next(pc2),
            // LDMXCSR (/2) / STMXCSR (/3): a 32-bit load/store to memory.
            (RmKind::Mem(_) | RmKind::MemRip(_), 2 | 3) => {
                let rm_op = resolve(modrm.kind, pc2);
                let Operand::Mem(addr) = rm_op else {
                    return Step::Illegal;
                };
                if modrm.reg == 2 {
                    match mem.read_u32(addr) {
                        Ok(v) => self.mxcsr = v,
                        Err(_) => return Step::Fault { addr, write: false },
                    }
                } else if mem.write(addr, &self.mxcsr.to_le_bytes()).is_err() {
                    return Step::Fault { addr, write: true };
                }
                self.next(pc2)
            }
            _ => Step::Illegal,
        }
    }

    /// `CPUID` (`0F A2`, no `ModRM`): dispatch on the leaf in `EAX` (and, for
    /// Pack the tracked arithmetic flags into an `RFLAGS` word, matching the
    /// bit layout the CPU writes to `R11` on `syscall`. Reserved bit 1 and the
    /// interrupt flag (bit 9, always set from a user task's view) are hardwired;
    /// `AF`/`TF` aren't modeled and read back as 0.
    fn rflags_word(&self) -> u64 {
        let mut f = 0x202u64; // bit 1 (reserved) | IF
        if self.flags.cf {
            f |= 1 << 0;
        }
        if self.flags.pf {
            f |= 1 << 2;
        }
        if self.flags.zf {
            f |= 1 << 6;
        }
        if self.flags.sf {
            f |= 1 << 7;
        }
        if self.df {
            f |= 1 << 10;
        }
        if self.flags.of {
            f |= 1 << 11;
        }
        f
    }

    /// leaf 7, the subleaf in `ECX` — this scaffold only implements subleaf
    /// 0) and write `EAX`/`EBX`/`ECX`/`EDX`. Feature bits are set *only* for
    /// what this interpreter actually executes, so glibc/musl's CPUID-gated
    /// dispatch never picks an unimplemented instruction path. Leaves this
    /// scaffold doesn't recognize return all-zero registers — a safe
    /// "nothing extra here" answer, rather than real hardware's leak-the-
    /// last-valid-leaf behavior.
    fn cpuid(&mut self) {
        let leaf = self.gpr[RAX] as u32;
        let (eax, ebx, ecx, edx): (u32, u32, u32, u32) = match leaf {
            // Leaf 0: max standard leaf (7) + the "GenuineIntel" vendor
            // string, split EBX/EDX/ECX = "Genu"/"ineI"/"ntel".
            0 => (7, 0x756E_6547, 0x6C65_746E, 0x4965_6E69),
            // Leaf 1: EAX = family/model/stepping (a plausible, unremarkable
            // identity — no real silicon behind it); EBX = 1 logical
            // processor, 64-byte CLFLUSH line size; ECX = CX16 (bit 13) |
            // POPCNT (bit 23) | RDRAND (bit 30); EDX = FPU (0) | TSC (4) |
            // CX8 (8) | CMOV (15) | CLFSH (19) | SSE (25) | SSE2 (26).
            1 => (0x0007_06A1, 0x0100_0800, 0x4080_2000, 0x0608_8111),
            0x8000_0000 => (0x8000_0004, 0, 0, 0), // max extended leaf
            // Extended feature bits: SYSCALL (11) | RDTSCP (27) | LM (29) —
            // this is a 64-bit ("long mode") interpreter with SYSCALL and
            // RDTSCP implemented.
            0x8000_0001 => (0, 0, 0, 0x2800_0800),
            0x8000_0002..=0x8000_0004 => Self::cpuid_brand_leaf(leaf),
            // Leaf 7 subleaf 0 (max subleaf 0; EBX's BMI1/BMI2/AVX2/... bits
            // are all zero — none of those are implemented) and any other
            // leaf this scaffold doesn't recognize: nothing extra.
            _ => (0, 0, 0, 0),
        };
        self.gpr[RAX] = u64::from(eax);
        self.gpr[RBX] = u64::from(ebx);
        self.gpr[RCX] = u64::from(ecx);
        self.gpr[RDX] = u64::from(edx);
    }

    /// The `EAX`/`EBX`/`ECX`/`EDX` quartet for one of `CPUID`'s three
    /// "processor brand string" leaves (`0x8000_0002..=0x8000_0004`): 16
    /// ASCII bytes per leaf, 48 total, from a fixed, null-padded identity
    /// string (there's no real silicon behind it).
    fn cpuid_brand_leaf(leaf: u32) -> (u32, u32, u32, u32) {
        const TEXT: &[u8] = b"nixvm software x86-64 CPU";
        let mut brand = [0u8; 48];
        brand[..TEXT.len()].copy_from_slice(TEXT);
        let base = ((leaf - 0x8000_0002) * 16) as usize;
        let word =
            |off: usize| u32::from_le_bytes(brand[base + off..base + off + 4].try_into().unwrap());
        (word(0), word(4), word(8), word(12))
    }

    /// Advance and return the free-running counter behind `RDTSC`/`RDTSCP`:
    /// incrementing on every read (rather than tracking real elapsed time,
    /// which this scaffold has no clock source for) guarantees a guest
    /// spin-loop that polls it for elapsed time always terminates.
    fn rdtsc_tick(&mut self) -> u64 {
        self.tsc = self.tsc.wrapping_add(1);
        self.tsc
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
            fetch!(
                mem.read(self.gpr[RSI], &mut b[..n])
                    .map_err(|_| Step::Fault {
                        addr: self.gpr[RSI],
                        write: false
                    })
            );
            fetch!(
                mem.write_trap(self.gpr[RDI], &b[..n])
                    .map_err(|e| Step::Fault {
                        addr: e.fault_addr(),
                        write: true
                    })
            );
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
            fetch!(
                mem.write_trap(self.gpr[RDI], &bytes[..n])
                    .map_err(|e| Step::Fault {
                        addr: e.fault_addr(),
                        write: true
                    })
            );
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
            fetch!(
                mem.read(self.gpr[RSI], &mut b[..n])
                    .map_err(|_| Step::Fault {
                        addr: self.gpr[RSI],
                        write: false
                    })
            );
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
            fetch!(
                mem.read(self.gpr[RDI], &mut b[..n])
                    .map_err(|_| Step::Fault {
                        addr: self.gpr[RDI],
                        write: false
                    })
            );
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
            fetch!(
                mem.read(self.gpr[RSI], &mut bs[..n])
                    .map_err(|_| Step::Fault {
                        addr: self.gpr[RSI],
                        write: false
                    })
            );
            fetch!(
                mem.read(self.gpr[RDI], &mut bd[..n])
                    .map_err(|_| Step::Fault {
                        addr: self.gpr[RDI],
                        write: false
                    })
            );
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
        if self.df {
            ptr.wrapping_sub(step)
        } else {
            ptr.wrapping_add(step)
        }
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

    // `opsize16`/`rep` are only consulted by the SSE fallback (see
    // `exec_0f_sse`); the two-byte opcode map is wide enough that threading
    // them through here is simpler than re-decoding prefixes twice.
    #[allow(clippy::too_many_lines, clippy::too_many_arguments)]
    fn exec_0f(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        rex: Rex,
        has_rex: bool,
        width: u32,
        opsize16: bool,
        rep: u8,
    ) -> Step {
        let (op2, pc) = fetch!(fetch_u8(mem, pc));
        match op2 {
            0x05 => {
                // `syscall` copies RIP→RCX and RFLAGS→R11 before entering the
                // kernel, exactly as hardware does. musl/V8 syscall trampolines
                // read RCX afterward (it holds the return address), so leaving
                // it stale silently corrupted their control flow. `rip` itself
                // stays on the opcode — the kernel advances it when it writes
                // the return value.
                self.gpr[RCX] = pc;
                self.gpr[R11] = self.rflags_word();
                Step::Syscall
            }
            // 0F 1F /0: the canonical multi-byte NOP (any prefix; the ModRM/SIB
            // is decoded only to consume the instruction's full length).
            // F3 0F 1E: CET instructions — ENDBR64/ENDBR32 landing pads (FA/FB)
            // and the RDSSP shadow-stack reads — all architecturally NOPs on a
            // CPU without CET, which is what this interpreter models. gcc emits
            // ENDBR64 at every function entry by default (-fcf-protection), so
            // stock distro binaries hit this on their very first instruction.
            0x1F => {
                let (_, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
                self.next(pc2)
            }
            0x1E if rep == 1 => {
                let (_, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
                self.next(pc2)
            }
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
                let result = mask_w(p as u128 as u64, width);
                self.gpr[modrm.reg] = result;
                self.set_imul_flags(cf, result, width);
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
                let val = if signed {
                    sign_extend_w(raw, src_width) as u64
                } else {
                    raw
                };
                self.gpr[modrm.reg] = mask_w(val, width);
                self.next(pc2)
            }
            0xA3 => self.bt_ev_gv(mem, pc, rex, width, BitTestOp::Bt),
            0xAB => self.bt_ev_gv(mem, pc, rex, width, BitTestOp::Bts),
            0xB3 => self.bt_ev_gv(mem, pc, rex, width, BitTestOp::Btr),
            0xBB => self.bt_ev_gv(mem, pc, rex, width, BitTestOp::Btc),
            0xBA => self.bt_group_imm(mem, pc, rex, width),
            0xA4 => self.shld_shrd(mem, pc, rex, width, true, false),
            0xA5 => self.shld_shrd(mem, pc, rex, width, true, true),
            0xAC => self.shld_shrd(mem, pc, rex, width, false, false),
            0xAD => self.shld_shrd(mem, pc, rex, width, false, true),
            0xBC => self.bit_scan(mem, pc, rex, width, rep, false), // BSF, or TZCNT under F3
            0xBD => self.bit_scan(mem, pc, rex, width, rep, true),  // BSR, or LZCNT under F3
            0xB8 if rep == 1 => self.popcnt(mem, pc, rex, width),
            0xC8..=0xCF => {
                // BSWAP r (register embedded in the low 3 bits of op2, no ModRM).
                let r = usize::from(op2 - 0xC8) | (usize::from(rex.b) << 3);
                let bs = match width {
                    64 => self.gpr[r].swap_bytes(),
                    16 => u64::from((self.gpr[r] as u16).swap_bytes()),
                    _ => u64::from((self.gpr[r] as u32).swap_bytes()),
                };
                fetch!(self.write_operand(mem, Operand::Reg(r), bs, width));
                self.next(pc)
            }
            0xA2 => {
                self.cpuid();
                self.next(pc)
            }
            0x31 => {
                // RDTSC: EDX:EAX = a free-running counter (see `rdtsc_tick`).
                let t = self.rdtsc_tick();
                self.gpr[RAX] = t & 0xffff_ffff;
                self.gpr[RDX] = t >> 32;
                self.next(pc)
            }
            0x01 => {
                // Group 7 is a large, mostly-privileged/system-instruction
                // opcode map; we only recognize the two register-form
                // (`mod == 11`) encodings a userspace program can actually
                // reach: `RDTSCP` (`F9`) and `XGETBV` (`D0`). Peeking at the
                // raw ModRM byte (rather than a full `decode_modrm`) is safe
                // here because every other sub-form we don't implement is
                // rejected outright, with no operand to resolve correctly.
                let (b, pc2) = fetch!(fetch_u8(mem, pc));
                match b {
                    0xF9 => {
                        // RDTSCP: like RDTSC, plus ECX = TSC_AUX (there's no
                        // real per-core/node id to report, so always 0).
                        let t = self.rdtsc_tick();
                        self.gpr[RAX] = t & 0xffff_ffff;
                        self.gpr[RDX] = t >> 32;
                        self.gpr[RCX] = 0;
                        self.next(pc2)
                    }
                    0xD0 => {
                        // XGETBV (ECX selects the XCR; only XCR0 is
                        // meaningful and we don't validate it): x87|SSE
                        // state only (bit 2, AVX/YMM state, is never
                        // advertised — no AVX support).
                        self.gpr[RAX] = 0x3;
                        self.gpr[RDX] = 0;
                        self.next(pc2)
                    }
                    // SGDT/SIDT/LGDT/LIDT/SMSW/LMSW/INVLPG (privileged or
                    // memory-system state), SWAPGS/MONITOR/MWAIT/XSETBV/
                    // VMCALL/VMFUNC/XEND/XTEST/RDPKRU/WRPKRU: not in our
                    // documented subset (either privileged, or no state to
                    // back them in a single-address-space scaffold).
                    _ => Step::Illegal,
                }
            }
            0xC0 => self.xadd(mem, pc, rex, has_rex, 8),
            0xC1 => self.xadd(mem, pc, rex, has_rex, width),
            0xB0 => self.cmpxchg(mem, pc, rex, has_rex, 8),
            0xB1 => self.cmpxchg(mem, pc, rex, has_rex, width),
            0xC3 => self.movnti(mem, pc, rex, width),
            0xAE => self.group15_ae(mem, pc, rex),
            0xC7 => self.group9_c7(mem, pc, rex, width),
            _ => self.exec_0f_sse(mem, pc, rex, opsize16, rep, op2),
        }
    }

    /// The SSE/SSE2 subset of the two-byte `0F` opcode map: everything
    /// [`exec_0f`] doesn't already claim. `opsize16` (`0x66`) and `rep`
    /// (`1` = `0xF3`, `2` = `0xF2`) are the *mandatory* prefixes that select
    /// among the `PS`/`PD`/`SS`/`SD` (or plain/`66` integer) variants each
    /// opcode packs together — not the operand-size/`REP` prefixes they'd be
    /// on a non-`0F` opcode.
    #[allow(clippy::too_many_lines)] // one flat opcode dispatch, same style as exec_0f
    fn exec_0f_sse(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        rex: Rex,
        opsize16: bool,
        rep: u8,
        op2: u8,
    ) -> Step {
        // REX.W selects a 64-bit GPR operand for the SSE<->GPR forms
        // (MOVD/MOVQ, CVTSI2S*, CVTS*2SI); the `0x66` mandatory prefix here
        // is *not* the 16-bit operand-size prefix, so it must not shrink it.
        let gw = if rex.w { 64 } else { 32 };
        match op2 {
            0x10 | 0x11 => self.sse_move(mem, pc, rex, rep, op2 == 0x11),
            0x12 | 0x13 | 0x16 | 0x17 => self.sse_mov_half(mem, pc, rex, opsize16, rep, op2),
            0x14 => self.sse_unpck(mem, pc, rex, if opsize16 { 8 } else { 4 }, false),
            0x15 => self.sse_unpck(mem, pc, rex, if opsize16 { 8 } else { 4 }, true),
            0x28 | 0x29 | 0x6F | 0x7F => self.sse_movaps(mem, pc, rex, matches!(op2, 0x29 | 0x7F)),
            0x38 => self.exec_0f_38(mem, pc, rex),
            0x3A => self.exec_0f_3a(mem, pc, rex),
            0x50 => self.sse_movmskp(mem, pc, rex, opsize16),
            0x63 => self.sse_pack(mem, pc, rex, 2, true),  // PACKSSWB
            0x67 => self.sse_pack(mem, pc, rex, 2, false), // PACKUSWB
            0x6B => self.sse_pack(mem, pc, rex, 4, true),  // PACKSSDW
            0x60 => self.sse_unpck(mem, pc, rex, 1, false), // PUNPCKLBW
            0x61 => self.sse_unpck(mem, pc, rex, 2, false), // PUNPCKLWD
            0x62 => self.sse_unpck(mem, pc, rex, 4, false), // PUNPCKLDQ
            0x64 => self.sse_pcmpgt(mem, pc, rex, 1),       // PCMPGTB
            0x66 => self.sse_pcmpgt(mem, pc, rex, 4),       // PCMPGTD
            0x68 => self.sse_unpck(mem, pc, rex, 1, true),  // PUNPCKHBW
            0x69 => self.sse_unpck(mem, pc, rex, 2, true),  // PUNPCKHWD
            0x6A => self.sse_unpck(mem, pc, rex, 4, true),  // PUNPCKHDQ
            0x6C => self.sse_unpck(mem, pc, rex, 8, false), // PUNPCKLQDQ
            0x6D => self.sse_unpck(mem, pc, rex, 8, true),  // PUNPCKHQDQ
            0x6E => self.sse_movd_load(mem, pc, rex, gw),
            0x7E if rep == 1 => self.sse_movq_xmm_load(mem, pc, rex),
            0x7E => self.sse_movd_store(mem, pc, rex, gw),
            0xD6 => self.sse_movq_store(mem, pc, rex),
            0xD7 => self.sse_pmovmskb(mem, pc, rex),
            0x2A => self.sse_cvtsi2sx(mem, pc, rex, gw, rep),
            0x2C => self.sse_cvt_sx2si(mem, pc, rex, gw, rep, false),
            0x2D => self.sse_cvt_sx2si(mem, pc, rex, gw, rep, true),
            0x2E | 0x2F => self.sse_comis(mem, pc, rex, opsize16),
            0x51 => self.sse_arith(mem, pc, rex, opsize16, rep, SseOp::Sqrt),
            0x54 | 0xDB => self.sse_bitwise(mem, pc, rex, BitOp::And),
            0x55 | 0xDF => self.sse_bitwise(mem, pc, rex, BitOp::Andn), // ANDNPS/ANDNPD, PANDN
            0x56 | 0xEB => self.sse_bitwise(mem, pc, rex, BitOp::Or),   // ORPS/ORPD, POR
            0xC2 => self.sse_cmp(mem, pc, rex, opsize16, rep),   // CMPPS/CMPSS/CMPPD/CMPSD
            0x57 | 0xEF => self.sse_bitwise(mem, pc, rex, BitOp::Xor),
            0x58 => self.sse_arith(mem, pc, rex, opsize16, rep, SseOp::Add),
            0x59 => self.sse_arith(mem, pc, rex, opsize16, rep, SseOp::Mul),
            0x5A => self.sse_cvt_ss_sd(mem, pc, rex, rep),
            0x5C => self.sse_arith(mem, pc, rex, opsize16, rep, SseOp::Sub),
            0x5D => self.sse_arith(mem, pc, rex, opsize16, rep, SseOp::Min),
            0x5E => self.sse_arith(mem, pc, rex, opsize16, rep, SseOp::Div),
            0x5F => self.sse_arith(mem, pc, rex, opsize16, rep, SseOp::Max),
            0x70 => self.sse_pshuf(mem, pc, rex, rep, opsize16),
            0x71 => self.sse_shift_imm_group(mem, pc, rex, 2),
            0x72 => self.sse_shift_imm_group(mem, pc, rex, 4),
            0x73 => self.sse_shift_imm_group(mem, pc, rex, 8),
            0x74 => self.sse_pcmpeq(mem, pc, rex, 1),
            0x76 => self.sse_pcmpeq(mem, pc, rex, 4),
            0xC6 => self.sse_shuf(mem, pc, rex, opsize16),
            0xD4 => self.sse_paddsub(mem, pc, rex, 8, true), // PADDQ
            0xDA => self.sse_pminmaxub(mem, pc, rex, true),  // PMINUB
            0xDE => self.sse_pminmaxub(mem, pc, rex, false), // PMAXUB
            0xFA => self.sse_paddsub(mem, pc, rex, 4, false), // PSUBD
            0xFB => self.sse_paddsub(mem, pc, rex, 8, false), // PSUBQ
            0xFC => self.sse_paddsubb(mem, pc, rex, true),
            0xF8 => self.sse_paddsubb(mem, pc, rex, false),
            0xFE => self.sse_paddsub(mem, pc, rex, 4, true), // PADDD
            _ => Step::Illegal,
        }
    }

    /// `MOVUPS`/`MOVUPD` (no mandatory prefix / `0x66`, full 128-bit) and
    /// `MOVSS`/`MOVSD` (`0xF3`/`0xF2`, 32-/64-bit scalar): load (`store ==
    /// false`) or store (`store == true`) between `xmm(reg)` and `xmm/m
    /// (r/m)`. The scalar forms only ever touch the low lane; a *register*
    /// destination keeps its upper bits, while a *memory* destination or
    /// source has none to preserve (mem-to-reg zeroes the upper lanes, per
    /// the `MOVSS`/`MOVSD` spec).
    /// `0F 12/13/16/17`: the 64-bit half-register moves — `MOVLPS`/`MOVLPD`
    /// (load/store the low half), `MOVHPS`/`MOVHPD` (the high half), the
    /// register forms `MOVHLPS`/`MOVLHPS` (cross-half reg-to-reg), and the
    /// `F3`/`F2`-selected dup forms sharing `12`/`16`: `MOVDDUP`,
    /// `MOVSLDUP`/`MOVSHDUP`. glibc's SSE `memcpy`/`strlen` lean on these.
    fn sse_mov_half(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        rex: Rex,
        opsize16: bool,
        rep: u8,
        op2: u8,
    ) -> Step {
        const LOW64: u128 = u64::MAX as u128;
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let high = op2 & 0x04 != 0; // 16/17 move the high half, 12/13 the low
        if op2 & 0x01 != 0 {
            // 13/17: store — memory destination only, no rep-selected forms.
            if rep != 0 {
                return Step::Illegal;
            }
            let Operand::Mem(a) = rm_op else {
                return Step::Illegal;
            };
            let v = self.xmm[modrm.reg];
            let half = if high { (v >> 64) as u64 } else { v as u64 };
            fetch!(mem.write_trap(a, &half.to_le_bytes()).map_err(|e| Step::Fault {
                addr: e.fault_addr(),
                write: true
            }));
            return self.next(pc2);
        }
        match rep {
            // F2 0F 12 MOVDDUP: both halves get the source's low 64 bits.
            2 => {
                if high || opsize16 {
                    return Step::Illegal;
                }
                let lo = fetch!(self.xmm_read_lo(mem, rm_op, 64));
                self.xmm[modrm.reg] = (u128::from(lo) << 64) | u128::from(lo);
            }
            // F3 0F 12/16 MOVSLDUP/MOVSHDUP: duplicate the even (SL) or odd
            // (SH) 32-bit lanes of the full 128-bit source.
            1 => {
                if opsize16 {
                    return Step::Illegal;
                }
                let src = fetch!(self.xmm_read128(mem, rm_op));
                let mut out = 0u128;
                for lane in 0..4u32 {
                    let pick = if high { lane | 1 } else { lane & !1 };
                    let v = (src >> (32 * pick)) as u32;
                    out |= u128::from(v) << (32 * lane);
                }
                self.xmm[modrm.reg] = out;
            }
            _ => match rm_op {
                // Register forms: MOVHLPS (12: low ← src's high half) and
                // MOVLHPS (16: high ← src's low half). No 66-prefixed
                // register encoding exists.
                Operand::Reg(r) => {
                    if opsize16 {
                        return Step::Illegal;
                    }
                    let src = self.xmm[r];
                    let dst = self.xmm[modrm.reg];
                    self.xmm[modrm.reg] = if high {
                        (dst & LOW64) | (u128::from(src as u64) << 64)
                    } else {
                        (dst & !LOW64) | u128::from((src >> 64) as u64)
                    };
                }
                // Memory forms: load 64 bits into one half, preserving the other.
                Operand::Mem(_) => {
                    let m = fetch!(self.xmm_read_lo(mem, rm_op, 64));
                    let dst = self.xmm[modrm.reg];
                    self.xmm[modrm.reg] = if high {
                        (dst & LOW64) | (u128::from(m) << 64)
                    } else {
                        (dst & !LOW64) | u128::from(m)
                    };
                }
                Operand::Reg8Hi(_) => return Step::Illegal,
            },
        }
        self.next(pc2)
    }

    fn sse_move(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex, rep: u8, store: bool) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let lane_bits = match rep {
            1 => Some(32u32),
            2 => Some(64u32),
            _ => None,
        };
        match (lane_bits, store) {
            (None, false) => {
                let v = fetch!(self.xmm_read128(mem, rm_op));
                self.xmm[modrm.reg] = v;
            }
            (None, true) => {
                let v = self.xmm[modrm.reg];
                fetch!(self.xmm_write128(mem, rm_op, v));
            }
            (Some(w), false) => {
                let is_reg = matches!(rm_op, Operand::Reg(_));
                let lo = fetch!(self.xmm_read_lo(mem, rm_op, w));
                self.xmm[modrm.reg] = if is_reg {
                    (self.xmm[modrm.reg] & !u128::from(mask_w(u64::MAX, w))) | u128::from(lo)
                } else {
                    u128::from(lo)
                };
            }
            (Some(w), true) => match rm_op {
                Operand::Reg(r) => {
                    let src = self.xmm[modrm.reg];
                    self.xmm[r] = (self.xmm[r] & !u128::from(mask_w(u64::MAX, w)))
                        | u128::from(mask_w(src as u64, w));
                }
                Operand::Mem(a) => {
                    let src = mask_w(self.xmm[modrm.reg] as u64, w);
                    let n = (w / 8) as usize;
                    let bytes = src.to_le_bytes();
                    fetch!(mem.write_trap(a, &bytes[..n]).map_err(|e| Step::Fault {
                        addr: e.fault_addr(),
                        write: true
                    }));
                }
                Operand::Reg8Hi(_) => unreachable!("SSE decode never yields an 8-bit-high operand"),
            },
        }
        self.next(pc2)
    }

    /// `MOVAPS`/`MOVAPD`/`MOVDQA`/`MOVDQU`: an unconditional full 128-bit
    /// load or store (alignment isn't enforced by this interpreter, so the
    /// aligned/unaligned and float/int-tagged variants all collapse to the
    /// same move).
    fn sse_movaps(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex, store: bool) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        if store {
            let v = self.xmm[modrm.reg];
            fetch!(self.xmm_write128(mem, rm_op, v));
        } else {
            let v = fetch!(self.xmm_read128(mem, rm_op));
            self.xmm[modrm.reg] = v;
        }
        self.next(pc2)
    }

    /// `MOVD`/`MOVQ` load (`66 0F 6E`): `xmm(reg) <- r/m32` (or `r/m64`
    /// under `REX.W`), zero-extended to 128 bits. The r/m side is a GPR or
    /// memory, so this reuses [`Self::read_operand`] (the GPR file), not the
    /// `xmm` helpers.
    fn sse_movd_load(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex, gw: u32) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let val = fetch!(self.read_operand(mem, rm_op, gw));
        self.xmm[modrm.reg] = u128::from(val);
        self.next(pc2)
    }

    /// `MOVD`/`MOVQ` store (`66 0F 7E`): `r/m32` (or `r/m64` under `REX.W`)
    /// `<- xmm(reg)`'s low lane.
    fn sse_movd_store(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex, gw: u32) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let val = mask_w(self.xmm[modrm.reg] as u64, gw);
        fetch!(self.write_operand(mem, rm_op, val, gw));
        self.next(pc2)
    }

    /// `MOVQ xmm1, xmm2/m64` (`F3 0F 7E`): load form — `xmm(reg) <- r/m64`,
    /// zeroing the upper 64 bits (unlike `MOVD`/`MOVQ`'s `66`-prefixed GPR
    /// form, the r/m side here is another `xmm`/memory).
    fn sse_movq_xmm_load(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let lo = fetch!(self.xmm_read_lo(mem, rm_op, 64));
        self.xmm[modrm.reg] = u128::from(lo);
        self.next(pc2)
    }

    /// `MOVQ xmm2/m64, xmm1` (`66 0F D6`): store form — `r/m64 <-
    /// xmm(reg)`'s low 64 bits; when the destination is itself an `xmm`
    /// register, its upper 64 bits are zeroed (not preserved).
    fn sse_movq_store(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let lo = self.xmm[modrm.reg] as u64;
        match rm_op {
            Operand::Reg(r) => self.xmm[r] = u128::from(lo),
            Operand::Mem(a) => {
                fetch!(
                    mem.write_trap(a, &lo.to_le_bytes())
                        .map_err(|e| Step::Fault {
                            addr: e.fault_addr(),
                            write: true
                        })
                );
            }
            Operand::Reg8Hi(_) => unreachable!("SSE decode never yields an 8-bit-high operand"),
        }
        self.next(pc2)
    }

    /// `PMOVMSKB Gd, xmm` (`66 0F D7`): each of the 16 bytes' sign bit packs
    /// into the corresponding bit of a GPR, zero-extended.
    fn sse_pmovmskb(&mut self, mem: &GuestMemory, pc: u64, rex: Rex) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let src = fetch!(self.xmm_read128(mem, rm_op));
        let bytes = src.to_le_bytes();
        let mut mask = 0u64;
        for (i, b) in bytes.iter().enumerate() {
            if b & 0x80 != 0 {
                mask |= 1 << i;
            }
        }
        self.gpr[modrm.reg] = mask;
        self.next(pc2)
    }

    /// `CVTSI2SD`/`CVTSI2SS` (`F2`/`F3` `0F 2A`): `xmm(reg)`'s low lane `<-
    /// (f64|f32) r/m` (a signed GPR or memory integer, `gw`-bits wide); the
    /// destination's upper bits are preserved (this is an arithmetic-style
    /// op, not a move).
    #[allow(clippy::cast_precision_loss)] // int->float is exactly what CVTSI2S* does
    fn sse_cvtsi2sx(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex, gw: u32, rep: u8) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let raw = fetch!(self.read_operand(mem, rm_op, gw));
        let ival = sign_extend_w(raw, gw);
        self.xmm[modrm.reg] = if rep == 2 {
            let bits = u128::from((ival as f64).to_bits());
            (self.xmm[modrm.reg] & !u128::from(u64::MAX)) | bits
        } else {
            let bits = u128::from((ival as f32).to_bits());
            (self.xmm[modrm.reg] & !u128::from(u32::MAX)) | bits
        };
        self.next(pc2)
    }

    /// `CVTTSD2SI`/`CVTTSS2SI` (`truncate == false` is misleading — see
    /// below) and `CVTSD2SI`/`CVTSS2SI`: `Gd/Gq(reg) <- (i64) xmm/m` (a
    /// `F2`/`F3`-selected `f64`/`f32` source), either truncated toward zero
    /// (`CVTT*`, `round == false`) or rounded to nearest-even (`CVT*`,
    /// `round == true`).
    fn sse_cvt_sx2si(
        &mut self,
        mem: &GuestMemory,
        pc: u64,
        rex: Rex,
        gw: u32,
        rep: u8,
        round: bool,
    ) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let result: i64 = if rep == 2 {
            let bits = fetch!(self.xmm_read_lo(mem, rm_op, 64));
            let f = f64::from_bits(bits);
            if round {
                f.round_ties_even() as i64
            } else {
                f.trunc() as i64
            }
        } else {
            let bits = fetch!(self.xmm_read_lo(mem, rm_op, 32));
            let f = f32::from_bits(bits as u32);
            if round {
                f.round_ties_even() as i64
            } else {
                f.trunc() as i64
            }
        };
        self.gpr[modrm.reg] = if gw == 64 {
            result as u64
        } else {
            mask_w(result as u64, 32)
        };
        self.next(pc2)
    }

    /// `CVTSD2SS`/`CVTSS2SD` (`F2`/`F3 0F 5A`): narrow or widen the low lane
    /// between `f64` and `f32`, preserving the destination's other bits.
    fn sse_cvt_ss_sd(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex, rep: u8) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        match rep {
            2 => {
                let bits = fetch!(self.xmm_read_lo(mem, rm_op, 64));
                let f = f64::from_bits(bits) as f32;
                self.xmm[modrm.reg] =
                    (self.xmm[modrm.reg] & !u128::from(u32::MAX)) | u128::from(f.to_bits());
            }
            1 => {
                let bits = fetch!(self.xmm_read_lo(mem, rm_op, 32));
                let f = f64::from(f32::from_bits(bits as u32));
                self.xmm[modrm.reg] =
                    (self.xmm[modrm.reg] & !u128::from(u64::MAX)) | u128::from(f.to_bits());
            }
            _ => return Step::Illegal, // CVTPS2PD/CVTPD2PS (packed): not in our documented subset
        }
        self.next(pc2)
    }

    /// `UCOMISD`/`COMISD` (`66 0F 2E`/`2F`) and `UCOMISS`/`COMISS` (`0F
    /// 2E`/`2F`): compare `xmm(reg)` against `xmm/m (r/m)` and set `ZF`/
    /// `PF`/`CF` per the IEEE-754 ordered-compare predicate table (unordered
    /// — either operand `NaN` — sets all three; otherwise exactly one of
    /// less-than/equal/greater-than holds). `OF`/`SF` are always cleared; we
    /// don't distinguish the signaling (`COMIS*`) and quiet (`UCOMIS*`)
    /// `#I` exception behavior since this interpreter doesn't model FP
    /// exceptions at all.
    fn sse_comis(&mut self, mem: &GuestMemory, pc: u64, rex: Rex, opsize16: bool) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let (unordered, gt, lt) = if opsize16 {
            let a = f64::from_bits(self.xmm[modrm.reg] as u64);
            let bits = fetch!(self.xmm_read_lo(mem, rm_op, 64));
            let b = f64::from_bits(bits);
            (a.is_nan() || b.is_nan(), a > b, a < b)
        } else {
            let a = f32::from_bits(self.xmm[modrm.reg] as u32);
            let bits = fetch!(self.xmm_read_lo(mem, rm_op, 32));
            let b = f32::from_bits(bits as u32);
            (a.is_nan() || b.is_nan(), a > b, a < b)
        };
        self.flags = Flags {
            cf: unordered || lt,
            zf: unordered || (!gt && !lt),
            pf: unordered,
            of: false,
            sf: false,
        };
        self.next(pc2)
    }

    /// The `0F 51`/`54..5F` scalar+packed arithmetic group: `ADD`/`SUB`/
    /// `MUL`/`DIV`/`MIN`/`MAX`/`SQRT`, each packing four variants into one
    /// opcode via the mandatory prefix — no prefix = packed `PS` (4x
    /// `f32`), `0x66` = packed `PD` (2x `f64`), `0xF3` = scalar `SS`,
    /// `0xF2` = scalar `SD`. The scalar/packed distinction is just the lane
    /// count; [`f64_lane_binop`]/[`f32_lane_binop`] (and their unary
    /// counterparts for `SQRT`) already leave a scalar op's upper lanes as
    /// `dst`'s original bits.
    fn sse_arith(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        rex: Rex,
        opsize16: bool,
        rep: u8,
        op: SseOp,
    ) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let dst = self.xmm[modrm.reg];
        let apply_f64 = |dst: u128, src: u128, lanes: usize| match op {
            SseOp::Add => f64_lane_binop(dst, src, lanes, |a, b| a + b),
            SseOp::Sub => f64_lane_binop(dst, src, lanes, |a, b| a - b),
            SseOp::Mul => f64_lane_binop(dst, src, lanes, |a, b| a * b),
            SseOp::Div => f64_lane_binop(dst, src, lanes, |a, b| a / b),
            SseOp::Min => f64_lane_binop(dst, src, lanes, |a, b| if a < b { a } else { b }),
            SseOp::Max => f64_lane_binop(dst, src, lanes, |a, b| if a > b { a } else { b }),
            SseOp::Sqrt => f64_lane_unop(dst, src, lanes, f64::sqrt),
        };
        let apply_f32 = |dst: u128, src: u128, lanes: usize| match op {
            SseOp::Add => f32_lane_binop(dst, src, lanes, |a, b| a + b),
            SseOp::Sub => f32_lane_binop(dst, src, lanes, |a, b| a - b),
            SseOp::Mul => f32_lane_binop(dst, src, lanes, |a, b| a * b),
            SseOp::Div => f32_lane_binop(dst, src, lanes, |a, b| a / b),
            SseOp::Min => f32_lane_binop(dst, src, lanes, |a, b| if a < b { a } else { b }),
            SseOp::Max => f32_lane_binop(dst, src, lanes, |a, b| if a > b { a } else { b }),
            SseOp::Sqrt => f32_lane_unop(dst, src, lanes, f32::sqrt),
        };
        let result = match rep {
            2 => {
                let bits = fetch!(self.xmm_read_lo(mem, rm_op, 64));
                apply_f64(dst, u128::from(bits), 1)
            }
            1 => {
                let bits = fetch!(self.xmm_read_lo(mem, rm_op, 32));
                apply_f32(dst, u128::from(bits), 1)
            }
            _ if opsize16 => {
                let src = fetch!(self.xmm_read128(mem, rm_op));
                apply_f64(dst, src, 2)
            }
            _ => {
                let src = fetch!(self.xmm_read128(mem, rm_op));
                apply_f32(dst, src, 4)
            }
        };
        self.xmm[modrm.reg] = result;
        self.next(pc2)
    }

    /// `ANDPS`/`ANDPD`/`PAND` (`0F 54`/`66 0F 54`/`66 0F DB`), `XORPS`/
    /// `XORPD`/`PXOR` (`0F 57`/`66 0F 57`/`66 0F EF`), `POR` (`66 0F EB`)
    /// and `PANDN` (`66 0F DF`): a plain 128-bit bitwise op, `dst = dst OP
    /// `CMPPS`/`CMPSS`/`CMPPD`/`CMPSD` (`0F C2 /r ib`): compare float lanes
    /// against the imm8 predicate, writing an all-ones or all-zeros mask per
    /// lane. The prefix selects the form — `0xF2` scalar-double, `0xF3`
    /// scalar-single, `0x66` packed-double, none packed-single — exactly as the
    /// arithmetic ops. The immediate follows the r/m operand, so it's fetched
    /// before resolving a RIP-relative address (which is relative to the end of
    /// the whole instruction). V8's JIT emits these for JavaScript's relational
    /// operators on doubles.
    fn sse_cmp(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        rex: Rex,
        opsize16: bool,
        rep: u8,
    ) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let (pred, pc3) = fetch!(fetch_u8(mem, pc2));
        let rm_op = resolve(modrm.kind, pc3);
        let dst = self.xmm[modrm.reg];
        let result = match rep {
            2 => {
                let b = fetch!(self.xmm_read_lo(mem, rm_op, 64));
                f64_lane_cmp(dst, u128::from(b), 1, pred)
            }
            1 => {
                let b = fetch!(self.xmm_read_lo(mem, rm_op, 32));
                f32_lane_cmp(dst, u128::from(b), 1, pred)
            }
            _ if opsize16 => {
                let src = fetch!(self.xmm_read128(mem, rm_op));
                f64_lane_cmp(dst, src, 2, pred)
            }
            _ => {
                let src = fetch!(self.xmm_read128(mem, rm_op));
                f32_lane_cmp(dst, src, 4, pred)
            }
        };
        self.xmm[modrm.reg] = result;
        self.next(pc3)
    }

    /// src`. The float-tagged (`ANDPS`/`XORPS`) and integer-tagged (`PAND`/
    /// `PXOR`) opcodes compute an identical bit pattern, so [`BitOp`]
    /// doesn't need to distinguish which opcode selected it.
    fn sse_bitwise(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex, op: BitOp) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let src = fetch!(self.xmm_read128(mem, rm_op));
        let dst = self.xmm[modrm.reg];
        self.xmm[modrm.reg] = match op {
            BitOp::And => dst & src,
            BitOp::Andn => !dst & src,
            BitOp::Or => dst | src,
            BitOp::Xor => dst ^ src,
        };
        self.next(pc2)
    }

    /// `PCMPEQB`/`PCMPEQD` (`66 0F 74`/`76`): compare `dst` and `src`
    /// lane-wise (`lane_bytes` = 1 for `PCMPEQB`, 4 for `PCMPEQD`), setting
    /// each equal lane to all-1s and each unequal lane to all-0s.
    fn sse_pcmpeq(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex, lane_bytes: usize) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let src = fetch!(self.xmm_read128(mem, rm_op));
        let dst = self.xmm[modrm.reg];
        let (d, s) = (dst.to_le_bytes(), src.to_le_bytes());
        let mut out = [0u8; 16];
        for lane in (0..16).step_by(lane_bytes) {
            let eq = d[lane..lane + lane_bytes] == s[lane..lane + lane_bytes];
            let fill = if eq { 0xffu8 } else { 0u8 };
            out[lane..lane + lane_bytes].fill(fill);
        }
        self.xmm[modrm.reg] = u128::from_le_bytes(out);
        self.next(pc2)
    }

    /// `PADDB`/`PSUBB` (`66 0F FC`/`F8`): wrapping byte-lane add/subtract.
    fn sse_paddsubb(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex, add: bool) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let src = fetch!(self.xmm_read128(mem, rm_op));
        let dst = self.xmm[modrm.reg];
        let (d, s) = (dst.to_le_bytes(), src.to_le_bytes());
        let mut out = [0u8; 16];
        for i in 0..16 {
            out[i] = if add {
                d[i].wrapping_add(s[i])
            } else {
                d[i].wrapping_sub(s[i])
            };
        }
        self.xmm[modrm.reg] = u128::from_le_bytes(out);
        self.next(pc2)
    }

    /// `MOVMSKPS`/`MOVMSKPD` (`0F 50`/`66 0F 50`): `Gd` gets each packed
    /// lane's sign bit (4 `f32` lanes, or 2 `f64` lanes under `66`), packed
    /// into consecutive low bits and zero-extended.
    fn sse_movmskp(&mut self, mem: &GuestMemory, pc: u64, rex: Rex, opsize16: bool) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let src = fetch!(self.xmm_read128(mem, rm_op));
        let bytes = src.to_le_bytes();
        let lane_bytes = if opsize16 { 8 } else { 4 };
        let mut result = 0u64;
        for (lane, chunk) in bytes.chunks(lane_bytes).enumerate() {
            if chunk[lane_bytes - 1] & 0x80 != 0 {
                result |= 1 << lane;
            }
        }
        self.gpr[modrm.reg] = result;
        self.next(pc2)
    }

    /// `UNPCKLPS`/`UNPCKHPS`/`UNPCKLPD`/`UNPCKHPD` (`0F 14/15`, `66 0F
    /// 14/15`) and `PUNPCKL*`/`PUNPCKH*` (`66 0F 60/61/62/68/69/6A/6C/6D`):
    /// see [`unpck`].
    fn sse_unpck(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        rex: Rex,
        lane_bytes: usize,
        high: bool,
    ) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let src = fetch!(self.xmm_read128(mem, rm_op));
        let dst = self.xmm[modrm.reg];
        self.xmm[modrm.reg] = unpck(dst, src, lane_bytes, high);
        self.next(pc2)
    }

    /// `PACKSSWB`/`PACKUSWB`/`PACKSSDW` (`0F 63`/`0F 67`/`0F 6B`): saturating
    /// narrow-and-pack of `dst`||`src`. See [`pack128`].
    fn sse_pack(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        rex: Rex,
        in_bytes: usize,
        signed_out: bool,
    ) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let src = fetch!(self.xmm_read128(mem, rm_op));
        let dst = self.xmm[modrm.reg];
        self.xmm[modrm.reg] = pack128(dst, src, in_bytes, signed_out);
        self.next(pc2)
    }

    /// `PSHUFD` (`66 0F 70`), `PSHUFHW` (`F3 0F 70`) and `PSHUFLW` (`F2 0F
    /// 70`): permute `src`'s dwords (`PSHUFD`, all four lanes) or words
    /// (`PSHUFHW`/`PSHUFLW`, only the high/low four) into `dst` per the
    /// two-bit lane selectors packed into `imm8`; `PSHUFHW`/`PSHUFLW` pass
    /// their untouched half through unchanged.
    fn sse_pshuf(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        rex: Rex,
        rep: u8,
        opsize16: bool,
    ) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let (imm, pc3) = fetch!(fetch_u8(mem, pc2));
        let rm_op = resolve(modrm.kind, pc3);
        let src = fetch!(self.xmm_read128(mem, rm_op));
        let s = src.to_le_bytes();
        let mut out = [0u8; 16];
        if opsize16 {
            for i in 0..4 {
                let sel = usize::from((imm >> (2 * i)) & 3);
                out[i * 4..i * 4 + 4].copy_from_slice(&s[sel * 4..sel * 4 + 4]);
            }
        } else if rep == 1 || rep == 2 {
            let shuf_base = if rep == 1 { 4 } else { 0 }; // F3 = PSHUFHW (high words)
            let pass_base = 4 - shuf_base;
            for w in 0..4 {
                let o = (pass_base + w) * 2;
                out[o..o + 2].copy_from_slice(&s[o..o + 2]);
            }
            for i in 0..4 {
                let sel = shuf_base + usize::from((imm >> (2 * i)) & 3);
                let o = (shuf_base + i) * 2;
                out[o..o + 2].copy_from_slice(&s[sel * 2..sel * 2 + 2]);
            }
        } else {
            return Step::Illegal; // plain PSHUFW (MMX): not in our documented subset
        }
        self.xmm[modrm.reg] = u128::from_le_bytes(out);
        self.next(pc3)
    }

    /// `SHUFPS`/`SHUFPD` (`0F C6`/`66 0F C6`): pick `dst`'s low half-lanes
    /// from `dst` and its high half-lanes from `src`, per the two-bit
    /// (`SHUFPS`) or one-bit (`SHUFPD`) selectors packed into `imm8`.
    fn sse_shuf(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex, opsize16: bool) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let (imm, pc3) = fetch!(fetch_u8(mem, pc2));
        let rm_op = resolve(modrm.kind, pc3);
        let src = fetch!(self.xmm_read128(mem, rm_op));
        let dst = self.xmm[modrm.reg];
        let (d, s) = (dst.to_le_bytes(), src.to_le_bytes());
        let mut out = [0u8; 16];
        if opsize16 {
            let sel0 = usize::from(imm & 1);
            let sel1 = usize::from((imm >> 1) & 1);
            out[0..8].copy_from_slice(&d[sel0 * 8..sel0 * 8 + 8]);
            out[8..16].copy_from_slice(&s[sel1 * 8..sel1 * 8 + 8]);
        } else {
            let sels = [imm & 3, (imm >> 2) & 3, (imm >> 4) & 3, (imm >> 6) & 3];
            for (i, &sel) in sels.iter().enumerate() {
                let sel = usize::from(sel);
                let o = i * 4;
                if i < 2 {
                    out[o..o + 4].copy_from_slice(&d[sel * 4..sel * 4 + 4]);
                } else {
                    out[o..o + 4].copy_from_slice(&s[sel * 4..sel * 4 + 4]);
                }
            }
        }
        self.xmm[modrm.reg] = u128::from_le_bytes(out);
        self.next(pc3)
    }

    /// `0F 72`/`73` (`lane_bytes` = 4 or 8): the packed-shift-by-immediate
    /// group — `PSRLD`/`PSLLD` (`/2`/`/6`, `lane_bytes == 4`) or `PSRLQ`/
    /// `PSLLQ`/`PSRLDQ`/`PSLLDQ` (`/2`/`/6`/`/3`/`/7`, `lane_bytes == 8`).
    /// `PSRLDQ`/`PSLLDQ` shift the whole 128-bit register by whole *bytes*
    /// (zero-filling); the others shift each `lane_bytes`-wide lane
    /// independently by *bits* (see [`pack_shift_right`]/
    /// [`pack_shift_left`]).
    fn sse_shift_imm_group(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        rex: Rex,
        lane_bytes: u32,
    ) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let (imm, pc3) = fetch!(fetch_u8(mem, pc2));
        let rm_op = resolve(modrm.kind, pc3);
        let dst = fetch!(self.xmm_read128(mem, rm_op));
        let count = u32::from(imm);
        let result = match (lane_bytes, modrm.reg) {
            (2, 2) => pack_shift_right(dst, 16, count), // PSRLW
            (2, 4) => pack_shift_arith_right(dst, 16, count), // PSRAW
            (2, 6) => pack_shift_left(dst, 16, count), // PSLLW
            (4, 2) => pack_shift_right(dst, 32, count),
            (4, 4) => pack_shift_arith_right(dst, 32, count), // PSRAD
            (4, 6) => pack_shift_left(dst, 32, count),
            (8, 2) => pack_shift_right(dst, 64, count),
            (8, 6) => pack_shift_left(dst, 64, count),
            (8, 3) => {
                if count >= 16 { 0 } else { dst >> (count * 8) } // PSRLDQ
            }
            (8, 7) => {
                if count >= 16 { 0 } else { dst << (count * 8) } // PSLLDQ
            }
            _ => return Step::Illegal, // other sub-ops: not in our documented subset
        };
        fetch!(self.xmm_write128(mem, rm_op, result));
        self.next(pc3)
    }

    /// The three-byte `0F 38` opcode map: only `PSHUFB` (`66 0F 38 00`) is
    /// in our documented subset.
    fn exec_0f_38(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex) -> Step {
        let (op3, pc) = fetch!(fetch_u8(mem, pc));
        match op3 {
            0x00 => self.sse_pshufb(mem, pc, rex),
            0x17 => self.sse_ptest(mem, pc, rex),
            _ => Step::Illegal,
        }
    }

    /// `PTEST xmm1, xmm2/m128` (`66 0F 38 17`): set `ZF` when `dst & src` is all
    /// zero and `CF` when `~dst & src` is all zero; clear the other arithmetic
    /// flags. V8 uses it to test SIMD bitmaps.
    fn sse_ptest(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let src = fetch!(self.xmm_read128(mem, rm_op));
        let dst = self.xmm[modrm.reg];
        self.flags = Flags {
            zf: (dst & src) == 0,
            cf: (!dst & src) == 0,
            sf: false,
            of: false,
            pf: false,
        };
        self.next(pc2)
    }

    /// The three-byte `0F 3A` opcode map (SSSE3/SSE4 immediate forms).
    fn exec_0f_3a(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex) -> Step {
        let (op3, pc) = fetch!(fetch_u8(mem, pc));
        match op3 {
            0x0F => self.sse_palignr(mem, pc, rex),
            _ => Step::Illegal,
        }
    }

    /// `PALIGNR xmm1, xmm2/m128, imm8` (`66 0F 3A 0F /r ib`): concatenate
    /// `dst:src` (dst high, src low) into 256 bits, shift right by `imm8` bytes,
    /// and keep the low 128. V8 emits it for `memmove`/`String` byte shuffles.
    fn sse_palignr(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let (imm, pc3) = fetch!(fetch_u8(mem, pc2));
        let rm_op = resolve(modrm.kind, pc3);
        let src = fetch!(self.xmm_read128(mem, rm_op));
        let dst = self.xmm[modrm.reg];
        let mut cat = [0u8; 32];
        cat[0..16].copy_from_slice(&src.to_le_bytes());
        cat[16..32].copy_from_slice(&dst.to_le_bytes());
        let sh = usize::from(imm);
        let mut out = [0u8; 16];
        for (i, b) in out.iter_mut().enumerate() {
            *b = cat.get(sh + i).copied().unwrap_or(0);
        }
        self.xmm[modrm.reg] = u128::from_le_bytes(out);
        self.next(pc3)
    }

    /// `PSHUFB xmm1, xmm2/m128` (`66 0F 38 00`): each byte of `dst` becomes
    /// `src`'s byte at the index given by the low nibble of the
    /// corresponding `dst` byte, or zero if that byte's high bit is set.
    fn sse_pshufb(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let src = fetch!(self.xmm_read128(mem, rm_op));
        let dst = self.xmm[modrm.reg];
        let (d, s) = (dst.to_le_bytes(), src.to_le_bytes());
        let mut out = [0u8; 16];
        for i in 0..16 {
            out[i] = if d[i] & 0x80 != 0 {
                0
            } else {
                s[usize::from(d[i] & 0x0f)]
            };
        }
        self.xmm[modrm.reg] = u128::from_le_bytes(out);
        self.next(pc2)
    }

    /// `PADDD`/`PADDQ`/`PSUBD`/`PSUBQ` (`66 0F FE`/`D4`/`FA`/`FB`): wrapping
    /// `lane_bytes`-wide lane add/subtract — a 32-/64-bit-lane
    /// generalization of [`Self::sse_paddsubb`]'s byte lanes.
    #[allow(clippy::many_single_char_names)] // dst/src/lane_bytes/add is the natural naming here
    fn sse_paddsub(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        rex: Rex,
        lane_bytes: usize,
        add: bool,
    ) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let src = fetch!(self.xmm_read128(mem, rm_op));
        let dst = self.xmm[modrm.reg];
        let (d, s) = (dst.to_le_bytes(), src.to_le_bytes());
        let mut out = [0u8; 16];
        for lane in (0..16).step_by(lane_bytes) {
            let a = u64_from_le(&d[lane..lane + lane_bytes]);
            let b = u64_from_le(&s[lane..lane + lane_bytes]);
            let r = if add {
                a.wrapping_add(b)
            } else {
                a.wrapping_sub(b)
            };
            out[lane..lane + lane_bytes].copy_from_slice(&r.to_le_bytes()[..lane_bytes]);
        }
        self.xmm[modrm.reg] = u128::from_le_bytes(out);
        self.next(pc2)
    }

    /// `PMINUB`/`PMAXUB` (`66 0F DA`/`DE`): unsigned byte-lane min/max.
    fn sse_pminmaxub(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex, is_min: bool) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let src = fetch!(self.xmm_read128(mem, rm_op));
        let dst = self.xmm[modrm.reg];
        let (d, s) = (dst.to_le_bytes(), src.to_le_bytes());
        let mut out = [0u8; 16];
        for i in 0..16 {
            out[i] = if is_min {
                d[i].min(s[i])
            } else {
                d[i].max(s[i])
            };
        }
        self.xmm[modrm.reg] = u128::from_le_bytes(out);
        self.next(pc2)
    }

    /// `PCMPGTB`/`PCMPGTD` (`66 0F 64`/`66`): signed `lane_bytes`-wide
    /// per-lane greater-than compare, filling each lane with all-1s (true)
    /// or all-0s (false) — the signed counterpart of [`Self::sse_pcmpeq`].
    fn sse_pcmpgt(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex, lane_bytes: usize) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        let rm_op = resolve(modrm.kind, pc2);
        let src = fetch!(self.xmm_read128(mem, rm_op));
        let dst = self.xmm[modrm.reg];
        let (d, s) = (dst.to_le_bytes(), src.to_le_bytes());
        let mut out = [0u8; 16];
        for lane in (0..16).step_by(lane_bytes) {
            let a = sign_extend_w(
                u64_from_le(&d[lane..lane + lane_bytes]),
                lane_bytes as u32 * 8,
            );
            let b = sign_extend_w(
                u64_from_le(&s[lane..lane + lane_bytes]),
                lane_bytes as u32 * 8,
            );
            let fill = if a > b { 0xffu8 } else { 0u8 };
            out[lane..lane + lane_bytes].fill(fill);
        }
        self.xmm[modrm.reg] = u128::from_le_bytes(out);
        self.next(pc2)
    }

    // ---- x87 FPU (the `D8-DF` ESC opcodes). Register-relative addressing:
    // `ST(i)` lives at `st[(fpu_top + i) & 7]`; `fpu_push` decrements
    // `fpu_top` then writes the new `ST(0)`, `fpu_pop` reads `ST(0)` then
    // increments `fpu_top`. See [`X86Interp::st`] for the `f64`-models-
    // 80-bit-`long-double` approximation this is all built on. ----

    fn st_idx(&self, i: u8) -> usize {
        ((self.fpu_top.wrapping_add(i)) & 7) as usize
    }

    fn st_get(&self, i: u8) -> f64 {
        self.st[self.st_idx(i)]
    }

    fn st_set(&mut self, i: u8, v: f64) {
        let idx = self.st_idx(i);
        self.st[idx] = v;
    }

    fn fpu_push(&mut self, v: f64) {
        self.fpu_top = self.fpu_top.wrapping_sub(1) & 7;
        self.st[self.fpu_top as usize] = v;
    }

    /// Pop and return the old `ST(0)`.
    fn fpu_pop(&mut self) -> f64 {
        let v = self.st[self.fpu_top as usize];
        self.fpu_top = (self.fpu_top + 1) & 7;
        v
    }

    /// `FNINIT`: the power-on-reset FPU state (used both by the `FNINIT`
    /// opcode and by [`Vcpu::reset`]).
    fn fpu_init(&mut self) {
        self.st = [0.0; 8];
        self.fpu_top = 0;
        self.fpu_c0 = false;
        self.fpu_c1 = false;
        self.fpu_c2 = false;
        self.fpu_c3 = false;
        self.fpu_cw = 0x037F;
    }

    /// The compare core shared by `FCOM`/`FCOMP`/`FCOMPP`/`FUCOM`/`FUCOMP`/
    /// `FUCOMPP`/`FTST`/`FICOM`/`FICOMP`: an unordered (either operand
    /// `NaN`) compare sets `C0`/`C2`/`C3` all `true` (mirroring the SSE
    /// `UCOMISx`/`COMISx` unordered predicate — see
    /// [`X86Interp::sse_comis`]); otherwise exactly one of less-than/equal/
    /// greater-than holds, with `C2` clear. `C1` is always cleared (this
    /// interpreter never raises the stack-fault/inexact conditions real
    /// hardware would report there).
    fn fpu_compare(&mut self, a: f64, b: f64) {
        let unordered = a.is_nan() || b.is_nan();
        let (lt, gt) = (a < b, a > b);
        self.fpu_c0 = unordered || lt;
        self.fpu_c2 = unordered;
        self.fpu_c3 = unordered || (!lt && !gt); // equal, without a direct `==` (see sse_comis)
        self.fpu_c1 = false;
    }

    /// `FCOMI`/`FUCOMI`/`FCOMIP`/`FUCOMIP`: compare `ST(0)` against `ST(i)`
    /// and write the result directly into `ZF`/`PF`/`CF` (`OF`/`SF` always
    /// cleared) instead of `C0`/`C2`/`C3` — the same unordered predicate as
    /// [`X86Interp::fpu_compare`], just routed to `EFLAGS`. This
    /// interpreter doesn't distinguish the signaling (`FCOMI`) and quiet
    /// (`FUCOMI`) `#IA` exception behavior (it doesn't model FP exceptions
    /// at all), so both share this one implementation; `pop` is set for the
    /// `...IP` forms.
    fn fpu_comi(&mut self, i: u8, pop: bool) {
        let a = self.st_get(0);
        let b = self.st_get(i);
        let unordered = a.is_nan() || b.is_nan();
        let (lt, gt) = (a < b, a > b);
        self.flags = Flags {
            cf: unordered || lt,
            zf: unordered || (!lt && !gt), // equal, without a direct `==` (see sse_comis)
            pf: unordered,
            of: false,
            sf: false,
        };
        if pop {
            self.fpu_pop();
        }
    }

    /// `FIST`/`FISTP`/`FRNDINT`'s rounding mode, per the control word's `RC`
    /// field (bits 10-11) — the default control word (`0x037F`) selects
    /// round-to-nearest-even.
    fn round_per_cw(&self, v: f64) -> f64 {
        match (self.fpu_cw >> 10) & 3 {
            1 => v.floor(),
            2 => v.ceil(),
            3 => v.trunc(),
            _ => v.round_ties_even(),
        }
    }

    /// The status word `FNSTSW`/`FSTSW` report: `TOP` (bits 11-13) and
    /// `C0`/`C1`/`C2`/`C3` (bits 8/9/10/14); the busy, exception-summary and
    /// exception-flag bits are always `0` (never modeled).
    fn fpu_sw(&self) -> u16 {
        let mut sw = (u16::from(self.fpu_top) & 7) << 11;
        if self.fpu_c0 {
            sw |= 1 << 8;
        }
        if self.fpu_c1 {
            sw |= 1 << 9;
        }
        if self.fpu_c2 {
            sw |= 1 << 10;
        }
        if self.fpu_c3 {
            sw |= 1 << 14;
        }
        sw
    }

    /// Shared body of `D8`'s register form and every arithmetic group's
    /// `dst == ST(0)` case: apply `op` (arithmetic) or compare (`Com`/
    /// `Comp`, the latter also popping) against `src`.
    fn fpu_arith_st0(&mut self, op: FpuOp, src: f64) {
        match op {
            FpuOp::Com => self.fpu_compare(self.st_get(0), src),
            FpuOp::Comp => {
                self.fpu_compare(self.st_get(0), src);
                self.fpu_pop();
            }
            _ => {
                let dst = self.st_get(0);
                self.st_set(0, fpu_binop(op, dst, src));
            }
        }
    }

    /// The `mod != 3` (memory) form shared by `D8`/`DA`/`DC`/`DE`: `ST(0) op=
    /// src`, where `src` is loaded from memory at width `w` and `reg`
    /// selects the operation via [`FpuOp::from_reg`].
    fn fpu_arith_mem(
        &mut self,
        mem: &mut GuestMemory,
        reg: usize,
        kind: RmKind,
        pc2: u64,
        w: MemWidth,
    ) -> Step {
        let op = FpuOp::from_reg((reg & 7) as u8);
        let addr = match resolve(kind, pc2) {
            Operand::Mem(a) => a,
            Operand::Reg(_) | Operand::Reg8Hi(_) => return Step::Illegal, // mod!=3 always resolves to memory
        };
        let src = fetch!(fpu_read_src(mem, addr, w));
        self.fpu_arith_st0(op, src);
        self.next(pc2)
    }

    /// `D8`: memory form is `ST(0) op= m32fp`; register form is `ST(0) op=
    /// ST(i)`, using the same `/0../7` operation numbering ([`FpuOp::from_reg`]).
    fn fpu_d8(&mut self, mem: &mut GuestMemory, modrm: ModRm, pc2: u64) -> Step {
        match modrm.kind {
            RmKind::Reg(r) => {
                let op = FpuOp::from_reg((modrm.reg & 7) as u8);
                let src = self.st_get((r & 7) as u8);
                self.fpu_arith_st0(op, src);
                self.next(pc2)
            }
            _ => self.fpu_arith_mem(mem, modrm.reg, modrm.kind, pc2, MemWidth::F32),
        }
    }

    /// `D9`: `FLD`/`FST`/`FSTP m32fp`, `FLDCW`/`FNSTCW`, `FLD ST(i)`,
    /// `FXCH`, `FNOP`, `FCHS`/`FABS`/`FTST`, the constant loads
    /// (`FLD1`/`FLDZ`/...), `FDECSTP`/`FINCSTP`, `FSQRT`, `FRNDINT`.
    #[allow(clippy::too_many_lines)] // one flat opcode dispatch, same style as exec_0f_sse
    #[allow(clippy::single_match_else)] // the register-vs-memory ModRM split is the real structure, not a single-pattern match
    fn fpu_d9(&mut self, mem: &mut GuestMemory, modrm: ModRm, pc2: u64) -> Step {
        match modrm.kind {
            RmKind::Reg(r) => {
                let reg = modrm.reg & 7;
                let rm = (r & 7) as u8;
                match reg {
                    0 => {
                        // FLD ST(i): push a copy of ST(i).
                        let v = self.st_get(rm);
                        self.fpu_push(v);
                        self.next(pc2)
                    }
                    1 => {
                        // FXCH ST(i): swap ST(0) and ST(i).
                        let a = self.st_get(0);
                        let b = self.st_get(rm);
                        self.st_set(0, b);
                        self.st_set(rm, a);
                        self.next(pc2)
                    }
                    2 if rm == 0 => self.next(pc2), // FNOP
                    4 => match rm {
                        0 => {
                            self.st_set(0, -self.st_get(0)); // FCHS
                            self.next(pc2)
                        }
                        1 => {
                            self.st_set(0, self.st_get(0).abs()); // FABS
                            self.next(pc2)
                        }
                        4 => {
                            self.fpu_compare(self.st_get(0), 0.0); // FTST
                            self.next(pc2)
                        }
                        _ => Step::Illegal, // FXAM (D9 E5): not in our documented subset
                    },
                    5 => {
                        let Some(c) = (match rm {
                            0 => Some(1.0),                       // FLD1
                            1 => Some(std::f64::consts::LOG2_10), // FLDL2T
                            2 => Some(std::f64::consts::LOG2_E),  // FLDL2E
                            3 => Some(std::f64::consts::PI),      // FLDPI
                            4 => Some(std::f64::consts::LOG10_2), // FLDLG2
                            5 => Some(std::f64::consts::LN_2),    // FLDLN2
                            6 => Some(0.0),                       // FLDZ
                            _ => None,
                        }) else {
                            return Step::Illegal;
                        };
                        self.fpu_push(c);
                        self.next(pc2)
                    }
                    6 => match rm {
                        6 => {
                            self.fpu_top = self.fpu_top.wrapping_sub(1) & 7; // FDECSTP
                            self.next(pc2)
                        }
                        7 => {
                            self.fpu_top = (self.fpu_top + 1) & 7; // FINCSTP
                            self.next(pc2)
                        }
                        // F2XM1/FYL2X/FPTAN/FPATAN/FXTRACT/FPREM1: not in our documented subset
                        _ => Step::Illegal,
                    },
                    7 => match rm {
                        2 => {
                            self.st_set(0, self.st_get(0).sqrt()); // FSQRT
                            self.next(pc2)
                        }
                        4 => {
                            let v = self.round_per_cw(self.st_get(0)); // FRNDINT
                            self.st_set(0, v);
                            self.next(pc2)
                        }
                        // FPREM/FYL2XP1/FSINCOS/FSCALE/FSIN/FCOS: not in our documented subset
                        _ => Step::Illegal,
                    },
                    _ => Step::Illegal, // D9 /1, /3 register forms: not in our documented subset
                }
            }
            _ => {
                let addr = match resolve(modrm.kind, pc2) {
                    Operand::Mem(a) => a,
                    Operand::Reg(_) | Operand::Reg8Hi(_) => return Step::Illegal,
                };
                match modrm.reg & 7 {
                    0 => {
                        let v = fetch!(fpu_read_f32(mem, addr)); // FLD m32fp
                        self.fpu_push(v);
                        self.next(pc2)
                    }
                    2 => {
                        let v = self.st_get(0); // FST m32fp
                        fetch!(fpu_write_f32(mem, addr, v));
                        self.next(pc2)
                    }
                    3 => {
                        let v = self.st_get(0); // FSTP m32fp
                        fetch!(fpu_write_f32(mem, addr, v));
                        self.fpu_pop();
                        self.next(pc2)
                    }
                    5 => {
                        // FLDCW m2byte
                        let mut b = [0u8; 2];
                        fetch!(
                            mem.read(addr, &mut b)
                                .map_err(|_| Step::Fault { addr, write: false })
                        );
                        self.fpu_cw = u16::from_le_bytes(b);
                        self.next(pc2)
                    }
                    7 => {
                        // FNSTCW m2byte
                        let bytes = self.fpu_cw.to_le_bytes();
                        fetch!(mem.write_trap(addr, &bytes).map_err(|e| Step::Fault {
                            addr: e.fault_addr(),
                            write: true,
                        }));
                        self.next(pc2)
                    }
                    // /1, FLDENV (/4), FNSTENV (/6): not in our documented subset
                    _ => Step::Illegal,
                }
            }
        }
    }

    /// `DA`: memory form is `ST(0) op= m32int` (`FIADD`/.../`FIDIVR`);
    /// the only register form we implement is `FUCOMPP` (`DA E9`).
    fn fpu_da(&mut self, mem: &mut GuestMemory, modrm: ModRm, pc2: u64) -> Step {
        match modrm.kind {
            RmKind::Reg(r) => {
                if (modrm.reg & 7) == 5 && (r & 7) == 1 {
                    self.fpu_compare(self.st_get(0), self.st_get(1)); // FUCOMPP
                    self.fpu_pop();
                    self.fpu_pop();
                    self.next(pc2)
                } else {
                    Step::Illegal // FCMOVcc: not in our documented subset
                }
            }
            _ => self.fpu_arith_mem(mem, modrm.reg, modrm.kind, pc2, MemWidth::I32),
        }
    }

    /// `DB`: `FILD`/`FIST`/`FISTP m32int`, `FLD`/`FSTP m80fp`, `FNCLEX`,
    /// `FNINIT`, `FUCOMI`/`FCOMI`.
    #[allow(clippy::single_match_else)] // the register-vs-memory ModRM split is the real structure, not a single-pattern match
    #[allow(clippy::cast_precision_loss)] // FILD's int->f64 load is exactly this
    fn fpu_db(&mut self, mem: &mut GuestMemory, modrm: ModRm, pc2: u64) -> Step {
        match modrm.kind {
            RmKind::Reg(r) => {
                let rm = (r & 7) as u8;
                match modrm.reg & 7 {
                    4 => match rm {
                        2 => self.next(pc2), // FNCLEX: no exception state is modeled, so a no-op
                        3 => {
                            self.fpu_init(); // FNINIT
                            self.next(pc2)
                        }
                        // FNENI/FNDISI/FNSETPM (obsolete 287 opcodes): not in our documented subset
                        _ => Step::Illegal,
                    },
                    5 | 6 => {
                        self.fpu_comi(rm, false); // FUCOMI (/5) / FCOMI (/6)
                        self.next(pc2)
                    }
                    _ => Step::Illegal, // FCMOVNcc: not in our documented subset
                }
            }
            _ => {
                let addr = match resolve(modrm.kind, pc2) {
                    Operand::Mem(a) => a,
                    Operand::Reg(_) | Operand::Reg8Hi(_) => return Step::Illegal,
                };
                match modrm.reg & 7 {
                    0 => {
                        let v = fetch!(fpu_read_int(mem, addr, 32)); // FILD m32int
                        self.fpu_push(v as f64);
                        self.next(pc2)
                    }
                    2 => {
                        let v = self.round_per_cw(self.st_get(0)); // FIST m32int
                        fetch!(fpu_write_int(mem, addr, v as i64, 32));
                        self.next(pc2)
                    }
                    3 => {
                        let v = self.round_per_cw(self.st_get(0)); // FISTP m32int
                        fetch!(fpu_write_int(mem, addr, v as i64, 32));
                        self.fpu_pop();
                        self.next(pc2)
                    }
                    5 => {
                        let v = fetch!(fpu_read_f80(mem, addr)); // FLD m80fp
                        self.fpu_push(v);
                        self.next(pc2)
                    }
                    7 => {
                        let v = self.st_get(0); // FSTP m80fp
                        fetch!(fpu_write_f80(mem, addr, v));
                        self.fpu_pop();
                        self.next(pc2)
                    }
                    // /1 FISTTP (SSE3), /4, /6: not in our documented subset
                    _ => Step::Illegal,
                }
            }
        }
    }

    /// `DC`: memory form is `ST(0) op= m64fp`; register form is `ST(i) op=
    /// ST(0)` with `SUB`/`SUBR` and `DIV`/`DIVR` swapped relative to `D8`
    /// (see [`FpuOp::from_reg_reversed`]).
    fn fpu_dc(&mut self, mem: &mut GuestMemory, modrm: ModRm, pc2: u64) -> Step {
        match modrm.kind {
            RmKind::Reg(r) => {
                let Some(op) = FpuOp::from_reg_reversed((modrm.reg & 7) as u8) else {
                    return Step::Illegal;
                };
                let i = (r & 7) as u8;
                let dst = self.st_get(i);
                let src = self.st_get(0);
                self.st_set(i, fpu_binop(op, dst, src));
                self.next(pc2)
            }
            _ => self.fpu_arith_mem(mem, modrm.reg, modrm.kind, pc2, MemWidth::F64),
        }
    }

    /// `DD`: `FLD`/`FST`/`FSTP m64fp`, `FNSTSW m2byte`, `FFREE`,
    /// register `FST`/`FSTP ST(i)`, `FUCOM`/`FUCOMP`.
    #[allow(clippy::single_match_else)] // the register-vs-memory ModRM split is the real structure, not a single-pattern match
    fn fpu_dd(&mut self, mem: &mut GuestMemory, modrm: ModRm, pc2: u64) -> Step {
        match modrm.kind {
            RmKind::Reg(r) => {
                let i = (r & 7) as u8;
                match modrm.reg & 7 {
                    0 => self.next(pc2), // FFREE ST(i): no tag word is modeled, so a no-op
                    2 => {
                        let v = self.st_get(0); // FST ST(i)
                        self.st_set(i, v);
                        self.next(pc2)
                    }
                    3 => {
                        let v = self.st_get(0); // FSTP ST(i)
                        self.st_set(i, v);
                        self.fpu_pop();
                        self.next(pc2)
                    }
                    4 => {
                        self.fpu_compare(self.st_get(0), self.st_get(i)); // FUCOM ST(i)
                        self.next(pc2)
                    }
                    5 => {
                        self.fpu_compare(self.st_get(0), self.st_get(i)); // FUCOMP ST(i)
                        self.fpu_pop();
                        self.next(pc2)
                    }
                    _ => Step::Illegal,
                }
            }
            _ => {
                let addr = match resolve(modrm.kind, pc2) {
                    Operand::Mem(a) => a,
                    Operand::Reg(_) | Operand::Reg8Hi(_) => return Step::Illegal,
                };
                match modrm.reg & 7 {
                    0 => {
                        let v = fetch!(fpu_read_f64(mem, addr)); // FLD m64fp
                        self.fpu_push(v);
                        self.next(pc2)
                    }
                    2 => {
                        let v = self.st_get(0); // FST m64fp
                        fetch!(fpu_write_f64(mem, addr, v));
                        self.next(pc2)
                    }
                    3 => {
                        let v = self.st_get(0); // FSTP m64fp
                        fetch!(fpu_write_f64(mem, addr, v));
                        self.fpu_pop();
                        self.next(pc2)
                    }
                    7 => {
                        let sw = self.fpu_sw(); // FNSTSW m2byte
                        fetch!(
                            mem.write_trap(addr, &sw.to_le_bytes())
                                .map_err(|e| Step::Fault {
                                    addr: e.fault_addr(),
                                    write: true,
                                })
                        );
                        self.next(pc2)
                    }
                    // /1 FISTTP (SSE3), FRSTOR (/4), FNSAVE (/6): not in our documented subset
                    _ => Step::Illegal,
                }
            }
        }
    }

    /// `DE`: memory form is `ST(0) op= m16int`; register form is `ST(i) op=
    /// ST(0)` then pop (the `P` mnemonics: `FADDP`/`FSUBRP`/...), using the
    /// same reversed numbering as `DC`; `DE D9` is the fixed opcode
    /// `FCOMPP`.
    fn fpu_de(&mut self, mem: &mut GuestMemory, modrm: ModRm, pc2: u64) -> Step {
        match modrm.kind {
            RmKind::Reg(r) => {
                let reg = modrm.reg & 7;
                let rm = (r & 7) as u8;
                if reg == 3 && rm == 1 {
                    self.fpu_compare(self.st_get(0), self.st_get(1)); // FCOMPP
                    self.fpu_pop();
                    self.fpu_pop();
                    return self.next(pc2);
                }
                let Some(op) = FpuOp::from_reg_reversed(reg as u8) else {
                    return Step::Illegal;
                };
                let dst = self.st_get(rm);
                let src = self.st_get(0);
                self.st_set(rm, fpu_binop(op, dst, src));
                self.fpu_pop();
                self.next(pc2)
            }
            _ => self.fpu_arith_mem(mem, modrm.reg, modrm.kind, pc2, MemWidth::I16),
        }
    }

    /// `DF`: `FILD`/`FIST`/`FISTP m16int`, `FILD m64int`, `FISTP m64int`,
    /// `FNSTSW AX` (`DF E0`), `FUCOMIP`/`FCOMIP`.
    #[allow(clippy::single_match_else)] // the register-vs-memory ModRM split is the real structure, not a single-pattern match
    #[allow(clippy::cast_precision_loss)] // FILD's int->f64 load is exactly this
    fn fpu_df(&mut self, mem: &mut GuestMemory, modrm: ModRm, pc2: u64) -> Step {
        match modrm.kind {
            RmKind::Reg(r) => {
                let reg = modrm.reg & 7;
                let rm = (r & 7) as u8;
                match reg {
                    4 if rm == 0 => {
                        let sw = u64::from(self.fpu_sw()); // FNSTSW AX
                        fetch!(self.write_operand(mem, Operand::Reg(RAX), sw, 16));
                        self.next(pc2)
                    }
                    5 | 6 => {
                        self.fpu_comi(rm, true); // FUCOMIP (/5) / FCOMIP (/6)
                        self.next(pc2)
                    }
                    _ => Step::Illegal, // other DF register forms: not in our documented subset
                }
            }
            _ => {
                let addr = match resolve(modrm.kind, pc2) {
                    Operand::Mem(a) => a,
                    Operand::Reg(_) | Operand::Reg8Hi(_) => return Step::Illegal,
                };
                match modrm.reg & 7 {
                    0 => {
                        let v = fetch!(fpu_read_int(mem, addr, 16)); // FILD m16int
                        self.fpu_push(v as f64);
                        self.next(pc2)
                    }
                    2 => {
                        let v = self.round_per_cw(self.st_get(0)); // FIST m16int
                        fetch!(fpu_write_int(mem, addr, v as i64, 16));
                        self.next(pc2)
                    }
                    3 => {
                        let v = self.round_per_cw(self.st_get(0)); // FISTP m16int
                        fetch!(fpu_write_int(mem, addr, v as i64, 16));
                        self.fpu_pop();
                        self.next(pc2)
                    }
                    5 => {
                        let v = fetch!(fpu_read_int(mem, addr, 64)); // FILD m64int
                        self.fpu_push(v as f64);
                        self.next(pc2)
                    }
                    7 => {
                        let v = self.round_per_cw(self.st_get(0)); // FISTP m64int
                        fetch!(fpu_write_int(mem, addr, v as i64, 64));
                        self.fpu_pop();
                        self.next(pc2)
                    }
                    // FBLD/FBSTP (packed BCD, /4 and /6): not in our documented subset
                    _ => Step::Illegal,
                }
            }
        }
    }

    /// Dispatch on the `D8-DF` ESC opcode byte after decoding its ModRM
    /// (shared by all eight, since the memory-vs-`ST(i)`-vs-fixed-opcode
    /// split always happens at the ModRM `mod`/`reg`/`rm` fields).
    fn exec_x87(&mut self, mem: &mut GuestMemory, pc: u64, rex: Rex, esc: u8) -> Step {
        let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
        match esc {
            0xD8 => self.fpu_d8(mem, modrm, pc2),
            0xD9 => self.fpu_d9(mem, modrm, pc2),
            0xDA => self.fpu_da(mem, modrm, pc2),
            0xDB => self.fpu_db(mem, modrm, pc2),
            0xDC => self.fpu_dc(mem, modrm, pc2),
            0xDD => self.fpu_dd(mem, modrm, pc2),
            0xDE => self.fpu_de(mem, modrm, pc2),
            _ => self.fpu_df(mem, modrm, pc2), // 0xDF
        }
    }

    #[allow(clippy::too_many_lines)]
    fn exec(&mut self, mem: &mut GuestMemory) -> Step {
        // Legacy prefixes (operand-size `0x66`, `REP`/`REPE` `0xF3`, `REPNE`
        // `0xF2`, `LOCK` `0xF0`) precede any `REX` byte, which in turn must
        // immediately precede the opcode. `LOCK` just decorates the
        // following read-modify-write with a hardware bus-lock guarantee;
        // since this interpreter is single-threaded every op is already
        // atomic, so the prefix is decoded and otherwise ignored (matching
        // real hardware, we don't validate that the opcode it precedes is
        // actually one of the lockable ones).
        let mut pc = self.rip;
        let mut opsize16 = false;
        let mut rep: u8 = 0; // 0 = none, 1 = REP/REPE (F3), 2 = REPNE (F2)
        self.addr32 = false;
        self.seg_base = 0;
        loop {
            let (b, next) = fetch!(fetch_u8(mem, pc));
            match b {
                0x66 => opsize16 = true,
                0xF3 => rep = 1,
                0xF2 => rep = 2,
                0x67 => self.addr32 = true, // address-size: 32-bit effective addresses
                // FS override (`0x64`) is the one segment with a settable base
                // (`arch_prctl(ARCH_SET_FS)` → TLS). LOCK (`0xF0`) just
                // decorates an already-atomic op here, and the remaining
                // segment overrides — CS/DS/ES/SS are architecturally
                // zero-based in long mode, GS stays zero until ARCH_SET_GS is
                // modeled — are no-ops (they mostly appear as padding).
                0x64 => self.seg_base = self.fs_base,
                0xF0 | 0x26 | 0x2E | 0x36 | 0x3E | 0x65 => {}
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
            // POP r/m64 (`8F /0`): pop into a register or memory slot. A memory
            // destination that uses RSP as a base is addressed with the RSP
            // value *after* the pop's `RSP += 8` (Intel SDM; verified against
            // KVM by lockstep). `decode_modrm` folds the base register into the
            // effective address, so it must be re-run *after* the pop to pick up
            // the new RSP — the first decode only validates the `/0` encoding.
            // Only /0 is a valid encoding.
            0x8F => {
                let (modrm, _) = fetch!(self.decode_modrm(mem, pc, rex));
                if modrm.reg != 0 {
                    return Step::Illegal;
                }
                let val = fetch!(self.pop(mem));
                let (modrm, pc2) = fetch!(self.decode_modrm(mem, pc, rex));
                let rm_op = resolve(modrm.kind, pc2);
                fetch!(self.write_operand(mem, rm_op, val, 64));
                self.next(pc2)
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
                let val = if width == 64 {
                    sign_extend_w(raw, 32) as u64
                } else {
                    raw
                };
                self.gpr[modrm.reg] = mask_w(val, width);
                self.next(pc2)
            }
            0x69 => self.imul_imm(mem, pc, rex, width, false),
            0x6B => self.imul_imm(mem, pc, rex, width, true),
            0x86 => self.xchg(mem, pc, rex, has_rex, 8),
            0x87 => self.xchg(mem, pc, rex, has_rex, width),
            // XCHG rAX, r. Plain 0x90 is the NOP (XCHG eax,eax), but REX.B
            // re-points it at r8 — `49 90` is a real `xchg rax, r8` and gcc
            // emits it (silently NOP-ing it loses a register's value).
            0x90..=0x97 if opcode != 0x90 || rex.b => {
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
                    64 => {
                        self.gpr[RDX] = if sign_bit(self.gpr[RAX], 64) {
                            u64::MAX
                        } else {
                            0
                        }
                    }
                    16 => {
                        let d = if sign_bit(self.gpr[RAX] & 0xffff, 16) {
                            0xffffu64
                        } else {
                            0
                        };
                        self.gpr[RDX] = (self.gpr[RDX] & !0xffffu64) | d;
                    }
                    _ => {
                        self.gpr[RDX] = if sign_bit(self.gpr[RAX] & 0xffff_ffff, 32) {
                            0xffff_ffff
                        } else {
                            0
                        };
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
            0x08 => self.alu_rm_gv8(mem, pc, rex, has_rex, AluOp::Or, true),
            0x0A => self.alu_gv_rm8(mem, pc, rex, has_rex, AluOp::Or),
            0x09 => self.alu_rm_gv(mem, pc, rex, width, AluOp::Or, true),
            0x0B => self.alu_gv_rm(mem, pc, rex, width, AluOp::Or),
            0x10 => self.alu_rm_gv8(mem, pc, rex, has_rex, AluOp::Adc, true),
            0x12 => self.alu_gv_rm8(mem, pc, rex, has_rex, AluOp::Adc),
            0x11 => self.alu_rm_gv(mem, pc, rex, width, AluOp::Adc, true),
            0x13 => self.alu_gv_rm(mem, pc, rex, width, AluOp::Adc),
            0x14 => self.alu_acc_imm(mem, pc, 8, AluOp::Adc),
            0x15 => self.alu_acc_imm(mem, pc, width, AluOp::Adc),
            0x18 => self.alu_rm_gv8(mem, pc, rex, has_rex, AluOp::Sbb, true),
            0x1A => self.alu_gv_rm8(mem, pc, rex, has_rex, AluOp::Sbb),
            0x19 => self.alu_rm_gv(mem, pc, rex, width, AluOp::Sbb, true),
            0x1B => self.alu_gv_rm(mem, pc, rex, width, AluOp::Sbb),
            0x1C => self.alu_acc_imm(mem, pc, 8, AluOp::Sbb),
            0x1D => self.alu_acc_imm(mem, pc, width, AluOp::Sbb),
            0x20 => self.alu_rm_gv8(mem, pc, rex, has_rex, AluOp::And, true),
            0x22 => self.alu_gv_rm8(mem, pc, rex, has_rex, AluOp::And),
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
            // Accumulator-immediate short forms (`op AL, imm8` / `op eAX, immz`).
            0x04 => self.alu_acc_imm(mem, pc, 8, AluOp::Add),
            0x05 => self.alu_acc_imm(mem, pc, width, AluOp::Add),
            0x0C => self.alu_acc_imm(mem, pc, 8, AluOp::Or),
            0x0D => self.alu_acc_imm(mem, pc, width, AluOp::Or),
            0x24 => self.alu_acc_imm(mem, pc, 8, AluOp::And),
            0x25 => self.alu_acc_imm(mem, pc, width, AluOp::And),
            0x2C => self.alu_acc_imm(mem, pc, 8, AluOp::Sub),
            0x2D => self.alu_acc_imm(mem, pc, width, AluOp::Sub),
            0x34 => self.alu_acc_imm(mem, pc, 8, AluOp::Xor),
            0x35 => self.alu_acc_imm(mem, pc, width, AluOp::Xor),
            0x3C => self.alu_acc_imm(mem, pc, 8, AluOp::Cmp),
            0x3D => self.alu_acc_imm(mem, pc, width, AluOp::Cmp),
            0xA8 => self.alu_acc_imm(mem, pc, 8, AluOp::Test),
            0xA9 => self.alu_acc_imm(mem, pc, width, AluOp::Test),
            0x84 => self.alu_rm_gv8(mem, pc, rex, has_rex, AluOp::Test, false),
            0x85 => self.alu_rm_gv(mem, pc, rex, width, AluOp::Test, false),
            0x80 => self.group1_imm(mem, pc, rex, has_rex, 8, true),
            0x81 => self.group1_imm(mem, pc, rex, has_rex, width, false),
            0x83 => self.group1_imm(mem, pc, rex, has_rex, width, true),
            0xF6 => self.group3(mem, pc, rex, has_rex, 8),
            0xF7 => self.group3(mem, pc, rex, has_rex, width),
            0xFE => self.group4(mem, pc, rex, has_rex),
            0xFF => self.group5(mem, pc, rex, width),
            0xC0 => self.group2(mem, pc, rex, has_rex, 8, G2Count::Imm8),
            0xC1 => self.group2(mem, pc, rex, has_rex, width, G2Count::Imm8),
            0xD0 => self.group2(mem, pc, rex, has_rex, 8, G2Count::One),
            0xD1 => self.group2(mem, pc, rex, has_rex, width, G2Count::One),
            0xD2 => self.group2(mem, pc, rex, has_rex, 8, G2Count::Cl),
            0xD3 => self.group2(mem, pc, rex, has_rex, width, G2Count::Cl),
            0xE8 => {
                let (rel, pc2) = fetch!(fetch_i32(mem, pc));
                fetch!(self.push(mem, pc2));
                self.jump((pc2 as i64).wrapping_add(i64::from(rel)) as u64)
            }
            0xC3 => {
                let target = fetch!(self.pop(mem));
                self.jump(target)
            }
            // RET imm16: pop the return address, then release `imm16` bytes of
            // caller-pushed arguments from the stack. gcc/V8 emit this for
            // stdcall-style callees that clean up their own stack slots.
            0xC2 => {
                let (imm, _) = fetch!(fetch_u16(mem, pc));
                let target = fetch!(self.pop(mem));
                self.gpr[RSP] = self.gpr[RSP].wrapping_add(u64::from(imm));
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
            // NOP (also XCHG eax,eax, a no-op either way) / FWAIT/WAIT (no
            // pending FPU exceptions are ever modeled, so also a no-op).
            0x90 | 0x9B => self.next(pc),
            0xD8..=0xDF => self.exec_x87(mem, pc, rex, opcode),
            0x0F => self.exec_0f(mem, pc, rex, has_rex, width, opsize16, rep),
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
        self.xmm = [0; 16];
        self.gpr[RSP] = sp;
        self.rip = entry;
        self.flags = Flags::default();
        self.df = false;
        self.fs_base = 0;
        self.fpu_init();
    }

}

#[cfg(test)]
mod tests {
    // The SSE arithmetic tests below compare against IEEE-754 values that
    // are exactly representable (integers, halves, and sqrt() of a value
    // computed the same way) and produced by the exact same deterministic
    // operation being tested, so an exact comparison is the right check.
    #![allow(clippy::float_cmp)]

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
        assert_eq!(
            (cpu.gpr[RAX] >> 8) & 0xff,
            0x34,
            "AH is the high byte of RAX"
        );
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
        assert_eq!(
            cpu.gpr[RDX], 0x1234,
            "CMOVcc must not write when the condition is false"
        );
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
        assert_eq!(
            cpu.gpr[RAX] & 0xffff,
            60,
            "AX = AL * CL for an 8-bit operand"
        );

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
        assert_eq!(
            cpu.gpr[RAX], 0,
            "AX=0 sign-extends to EAX=0, clearing the upper 32 bits"
        );

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
        assert_eq!(
            m.read_u64(STACK - 8).unwrap(),
            CODE + 2,
            "return address pushed"
        );

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
        m.write_init(STACK - 0x40, &0x1122_3344u64.to_le_bytes())
            .unwrap();
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
    fn cpuid_leaf0_reports_vendor_string_and_max_leaf() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 0;
        // cpuid (0F A2)
        m.write_init(CODE, &[0x0F, 0xA2]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX] as u32, 7, "max standard leaf");
        let mut vendor = Vec::new();
        vendor.extend_from_slice(&(cpu.gpr[RBX] as u32).to_le_bytes());
        vendor.extend_from_slice(&(cpu.gpr[RDX] as u32).to_le_bytes());
        vendor.extend_from_slice(&(cpu.gpr[RCX] as u32).to_le_bytes());
        assert_eq!(
            vendor, b"GenuineIntel",
            "EBX/EDX/ECX spell the vendor string"
        );
    }

    #[test]
    fn cpuid_leaf1_edx_has_sse2_bit() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 1;
        m.write_init(CODE, &[0x0F, 0xA2]).unwrap();
        cpu.exec(&mut m);
        assert_ne!(
            cpu.gpr[RDX] as u32 & (1 << 26),
            0,
            "SSE2 feature bit (EDX bit 26) is set"
        );
        assert_ne!(
            cpu.gpr[RDX] as u32 & (1 << 0),
            0,
            "FPU feature bit (EDX bit 0) is set"
        );
    }

    #[test]
    fn rdtsc_increases_across_reads() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        // rdtsc (0F 31)
        m.write_init(CODE, &[0x0F, 0x31]).unwrap();
        cpu.exec(&mut m);
        let first = (cpu.gpr[RDX] << 32) | (cpu.gpr[RAX] & 0xffff_ffff);
        cpu.rip = CODE;
        cpu.exec(&mut m);
        let second = (cpu.gpr[RDX] << 32) | (cpu.gpr[RAX] & 0xffff_ffff);
        assert!(
            second > first,
            "RDTSC must return a monotonically increasing counter"
        );
    }

    #[test]
    fn rdrand_sets_cf_and_a_nonzero_value() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        // rdrand eax  (0F C7 /6, modrm=11 110 000)
        m.write_init(CODE, &[0x0F, 0xC7, 0xF0]).unwrap();
        cpu.exec(&mut m);
        assert!(cpu.flags.cf, "RDRAND always reports success");
        assert_ne!(cpu.gpr[RAX] as u32, 0);
    }

    #[test]
    fn cmpxchg_success_sets_zf_and_stores_src_failure_loads_accumulator() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 5; // accumulator
        cpu.gpr[RCX] = 42; // src, stored into dest on a match
        cpu.gpr[RBX] = 5; // dest == accumulator -> match
        // cmpxchg ebx, ecx  (0F B1 /r, modrm=11 001 011: reg=ecx, rm=ebx)
        m.write_init(CODE, &[0x0F, 0xB1, 0xCB]).unwrap();
        cpu.exec(&mut m);
        assert!(cpu.flags.zf, "a match sets ZF");
        assert_eq!(cpu.gpr[RBX] & 0xffff_ffff, 42, "dest <- src on a match");

        // dest (ebx) is now 42; accumulator is still 5, so this mismatches.
        cpu.gpr[RCX] = 99;
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert!(!cpu.flags.zf, "a mismatch clears ZF");
        assert_eq!(
            cpu.gpr[RAX] & 0xffff_ffff,
            42,
            "accumulator <- dest on a mismatch"
        );
        assert_eq!(
            cpu.gpr[RBX] & 0xffff_ffff,
            42,
            "a mismatch leaves dest untouched"
        );
    }

    #[test]
    fn xadd_returns_old_dest_value_and_sums() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 10; // dest
        cpu.gpr[RCX] = 5; // src
        // xadd eax, ecx  (0F C1 /r, modrm=11 001 000: reg=ecx, rm=eax)
        m.write_init(CODE, &[0x0F, 0xC1, 0xC8]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(
            cpu.gpr[RCX] & 0xffff_ffff,
            10,
            "reg gets the old dest value"
        );
        assert_eq!(cpu.gpr[RAX] & 0xffff_ffff, 15, "dest becomes dest + src");
    }

    #[test]
    fn lock_add_updates_memory_and_sets_flags() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        let addr = 0x1_2000u64;
        m.write_init(addr, &10u64.to_le_bytes()).unwrap();
        cpu.gpr[RBX] = addr;
        cpu.gpr[RCX] = 5;
        // lock add [rbx], ecx  (F0 01 /r, modrm=00 001 011: reg=ecx, rm=[rbx])
        m.write_init(CODE, &[0xF0, 0x01, 0x0B]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(
            m.read_u32(addr).unwrap(),
            15,
            "LOCK ADD still performs the add on memory"
        );
        assert!(!cpu.flags.zf);
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

    #[test]
    fn sse_movsd_load_store_and_scalar_arith() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        let a_addr = 0x1_2000u64;
        let b_addr = 0x1_2008u64;
        let out_addr = 0x1_2010u64;
        m.write_init(a_addr, &3.0f64.to_le_bytes()).unwrap();
        m.write_init(b_addr, &4.0f64.to_le_bytes()).unwrap();
        cpu.gpr[RAX] = a_addr;
        cpu.gpr[RBX] = b_addr;
        cpu.gpr[RCX] = out_addr;

        // movsd xmm0, [rax]  (F2 0F 10 /r, modrm=00 000 000)
        m.write_init(CODE, &[0xF2, 0x0F, 0x10, 0x00]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.xmm[0] as u64, 3.0f64.to_bits(), "MOVSD load from [rax]");
        assert_eq!(
            cpu.xmm[0] >> 64,
            0,
            "MOVSD mem-load zeroes the upper 64 bits"
        );

        // movsd xmm1, [rbx]  (F2 0F 10 /r, modrm=00 001 011)
        m.write_init(CODE, &[0xF2, 0x0F, 0x10, 0x0B]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.xmm[1] as u64, 4.0f64.to_bits());

        // Reg-reg MOVSD must preserve the destination's upper 64 bits.
        cpu.xmm[3] = 0xdead_beefu128 << 64;
        // movsd xmm3, xmm1  (F2 0F 10 /r, modrm=11 011 001)
        m.write_init(CODE, &[0xF2, 0x0F, 0x10, 0xD9]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(
            cpu.xmm[3] >> 64,
            0xdead_beef,
            "reg-reg MOVSD preserves dest's upper bits"
        );
        assert_eq!(cpu.xmm[3] as u64, 4.0f64.to_bits());

        // addsd xmm0, xmm1  (F2 0F 58 /r, modrm=11 000 001) -> 3.0 + 4.0 = 7.0
        m.write_init(CODE, &[0xF2, 0x0F, 0x58, 0xC1]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(f64::from_bits(cpu.xmm[0] as u64), 7.0);

        // movsd [rcx], xmm0  (F2 0F 11 /r, modrm=00 000 001) -> store 7.0
        m.write_init(CODE, &[0xF2, 0x0F, 0x11, 0x01]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(f64::from_bits(m.read_u64(out_addr).unwrap()), 7.0);

        // mulsd xmm0, xmm1  (F2 0F 59 /r, modrm=11 000 001) -> 7.0 * 4.0 = 28.0
        m.write_init(CODE, &[0xF2, 0x0F, 0x59, 0xC1]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(f64::from_bits(cpu.xmm[0] as u64), 28.0);

        // divsd xmm0, xmm1  (F2 0F 5E /r, modrm=11 000 001) -> 28.0 / 4.0 = 7.0
        m.write_init(CODE, &[0xF2, 0x0F, 0x5E, 0xC1]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(f64::from_bits(cpu.xmm[0] as u64), 7.0);

        // sqrtsd xmm2, xmm0  (F2 0F 51 /r, modrm=11 010 000) -> sqrt(7.0)
        m.write_init(CODE, &[0xF2, 0x0F, 0x51, 0xD0]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(f64::from_bits(cpu.xmm[2] as u64), 7.0f64.sqrt());
    }

    #[test]
    fn sse_cvtsi2sd_and_cvttsd2si_round_trip() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = (-42i64) as u64;
        // cvtsi2sd xmm0, eax  (F2 0F 2A /r, modrm=11 000 000)
        m.write_init(CODE, &[0xF2, 0x0F, 0x2A, 0xC0]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(f64::from_bits(cpu.xmm[0] as u64), -42.0);

        // cvttsd2si ecx, xmm0  (F2 0F 2C /r, modrm=11 001 000)
        m.write_init(CODE, &[0xF2, 0x0F, 0x2C, 0xC8]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(
            cpu.gpr[RCX] as u32 as i32, -42,
            "CVTTSD2SI truncates back to the original int"
        );
    }

    #[test]
    fn sse_ucomisd_sets_zf_cf_pf() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);

        // xmm0 = 1.0 < xmm1 = 2.0
        cpu.xmm[0] = u128::from(1.0f64.to_bits());
        cpu.xmm[1] = u128::from(2.0f64.to_bits());
        // ucomisd xmm0, xmm1  (66 0F 2E /r, modrm=11 000 001)
        m.write_init(CODE, &[0x66, 0x0F, 0x2E, 0xC1]).unwrap();
        cpu.exec(&mut m);
        assert!(cpu.flags.cf, "1.0 < 2.0 sets CF");
        assert!(!cpu.flags.zf);
        assert!(!cpu.flags.pf);

        // equal
        cpu.xmm[1] = u128::from(1.0f64.to_bits());
        m.write_init(CODE, &[0x66, 0x0F, 0x2E, 0xC1]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert!(!cpu.flags.cf);
        assert!(cpu.flags.zf, "1.0 == 1.0 sets ZF");
        assert!(!cpu.flags.pf);

        // greater
        cpu.xmm[1] = u128::from(0.5f64.to_bits());
        m.write_init(CODE, &[0x66, 0x0F, 0x2E, 0xC1]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert!(!cpu.flags.cf);
        assert!(!cpu.flags.zf);
        assert!(!cpu.flags.pf, "1.0 > 0.5 clears CF/ZF/PF");

        // unordered (NaN)
        cpu.xmm[1] = u128::from(f64::NAN.to_bits());
        m.write_init(CODE, &[0x66, 0x0F, 0x2E, 0xC1]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert!(
            cpu.flags.cf && cpu.flags.zf && cpu.flags.pf,
            "an unordered compare sets CF/ZF/PF"
        );
        assert!(!cpu.flags.of && !cpu.flags.sf, "OF/SF are always cleared");
    }

    #[test]
    fn sse_pxor_zeroes_register() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.xmm[0] = 0xdead_beef_dead_beef_dead_beef_dead_beefu128;
        // pxor xmm0, xmm0  (66 0F EF /r, modrm=11 000 000)
        m.write_init(CODE, &[0x66, 0x0F, 0xEF, 0xC0]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.xmm[0], 0);
    }

    #[test]
    fn sse_pcmpeqb_and_pmovmskb() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        let a: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mut b = a;
        b[0] = 0xFF; // byte 0 differs
        b[15] = 0xFF; // byte 15 differs
        cpu.xmm[0] = u128::from_le_bytes(a);
        cpu.xmm[1] = u128::from_le_bytes(b);
        // pcmpeqb xmm0, xmm1  (66 0F 74 /r, modrm=11 000 001)
        m.write_init(CODE, &[0x66, 0x0F, 0x74, 0xC1]).unwrap();
        cpu.exec(&mut m);
        let mask_bytes = cpu.xmm[0].to_le_bytes();
        assert_eq!(mask_bytes[0], 0x00, "unequal byte 0 -> all-zero lane");
        assert_eq!(mask_bytes[1], 0xff, "equal byte 1 -> all-one lane");
        assert_eq!(mask_bytes[15], 0x00, "unequal byte 15 -> all-zero lane");

        // pmovmskb eax, xmm0  (66 0F D7 /r, modrm=11 000 000)
        m.write_init(CODE, &[0x66, 0x0F, 0xD7, 0xC0]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX], 0x7ffe, "bits 1..=14 set, bits 0 and 15 clear");
    }

    #[test]
    fn bsf_bsr_and_zero_source_sets_zf() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 0b0101_0000; // lowest set bit at index 4, highest at index 6
        // bsf ecx, eax  (0F BC /r, modrm=11 001 000)
        m.write_init(CODE, &[0x0F, 0xBC, 0xC8]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RCX] & 0xffff_ffff, 4);
        assert!(!cpu.flags.zf);

        // bsr edx, eax  (0F BD /r, modrm=11 010 000)
        m.write_init(CODE, &[0x0F, 0xBD, 0xD0]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RDX] & 0xffff_ffff, 6);
        assert!(!cpu.flags.zf);

        // bsf ebx, esi with esi == 0: ZF set, ebx left unmodified.
        cpu.gpr[RSI] = 0;
        cpu.gpr[RBX] = 0x1234;
        // bsf ebx, esi  (0F BC /r, modrm=11 011 110)
        m.write_init(CODE, &[0x0F, 0xBC, 0xDE]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert!(cpu.flags.zf, "BSF of a zero source sets ZF");
        assert_eq!(
            cpu.gpr[RBX], 0x1234,
            "BSF must not modify the destination when the source is zero"
        );
    }

    #[test]
    fn popcnt_counts_bits_and_sets_zf() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 0b1011_0110; // 5 set bits
        // popcnt ecx, eax  (F3 0F B8 /r, modrm=11 001 000)
        m.write_init(CODE, &[0xF3, 0x0F, 0xB8, 0xC8]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RCX], 5);
        assert!(!cpu.flags.zf);

        cpu.gpr[RAX] = 0;
        m.write_init(CODE, &[0xF3, 0x0F, 0xB8, 0xC8]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RCX], 0);
        assert!(cpu.flags.zf);
    }

    #[test]
    fn bt_register_and_immediate_forms_set_cf() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 0b0000_0100; // bit 2 set
        cpu.gpr[RCX] = 2;
        // bt eax, ecx  (0F A3 /r, modrm=11 001 000)
        m.write_init(CODE, &[0x0F, 0xA3, 0xC8]).unwrap();
        cpu.exec(&mut m);
        assert!(cpu.flags.cf, "bit 2 of 0b100 is set");
        assert_eq!(cpu.gpr[RAX], 0b0000_0100, "BT must not modify the operand");

        cpu.gpr[RCX] = 1;
        // bt eax, ecx (bit 1, clear)
        m.write_init(CODE, &[0x0F, 0xA3, 0xC8]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert!(!cpu.flags.cf, "bit 1 of 0b100 is clear");

        // bts eax, 0  (0F BA /5 ib, modrm=11 101 000)
        m.write_init(CODE, &[0x0F, 0xBA, 0xE8, 0x00]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert!(!cpu.flags.cf, "bit 0 of 0b100 was clear before the set");
        assert_eq!(cpu.gpr[RAX] & 0xff, 0b0000_0101, "BTS sets bit 0");
    }

    #[test]
    fn shld_shrd_numeric_results() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 0x0000_0001; // dest
        cpu.gpr[RCX] = 0x8000_0000; // src (top bit feeds dest's vacated low bits)
        // shld eax, ecx, 4  (0F A4 /r ib, modrm=11 001 000)
        m.write_init(CODE, &[0x0F, 0xA4, 0xC8, 0x04]).unwrap();
        cpu.exec(&mut m);
        // (0x1 << 4) | (0x8000_0000 >> 28) = 0x10 | 0x8 = 0x18
        assert_eq!(cpu.gpr[RAX] & 0xffff_ffff, 0x18);

        cpu.gpr[RAX] = 0x8000_0000; // dest
        cpu.gpr[RCX] = 0x0000_000f; // src (low bits feed dest's vacated high bits)
        // shrd eax, ecx, 4  (0F AC /r ib, modrm=11 001 000)
        m.write_init(CODE, &[0x0F, 0xAC, 0xC8, 0x04]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        // (0x8000_0000 >> 4) | (0xf << 28) = 0x0800_0000 | 0xf000_0000 = 0xf800_0000
        assert_eq!(cpu.gpr[RAX] & 0xffff_ffff, 0xf800_0000);
    }

    #[test]
    fn bswap_reverses_byte_order() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 0x1122_3344;
        // bswap eax  (0F C8)
        m.write_init(CODE, &[0x0F, 0xC8]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX], 0x4433_2211);

        cpu.gpr[RCX] = 0x0102_0304_0506_0708;
        // bswap rcx  (REX.W 0F C9)
        m.write_init(CODE, &[0x48, 0x0F, 0xC9]).unwrap();
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RCX], 0x0807_0605_0403_0201);
    }

    #[test]
    fn pshufd_permutes_dword_lanes() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        let lanes: [u32; 4] = [0x1111_1111, 0x2222_2222, 0x3333_3333, 0x4444_4444];
        let mut bytes = [0u8; 16];
        for (i, v) in lanes.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        cpu.xmm[1] = u128::from_le_bytes(bytes);
        // pshufd xmm0, xmm1, 0x1B  (66 0F 70 /r ib, modrm=11 000 001):
        // imm=0b00_01_10_11 reverses the four lanes.
        m.write_init(CODE, &[0x66, 0x0F, 0x70, 0xC1, 0x1B]).unwrap();
        cpu.exec(&mut m);
        let out = cpu.xmm[0].to_le_bytes();
        assert_eq!(
            u32::from_le_bytes(out[0..4].try_into().unwrap()),
            0x4444_4444
        );
        assert_eq!(
            u32::from_le_bytes(out[4..8].try_into().unwrap()),
            0x3333_3333
        );
        assert_eq!(
            u32::from_le_bytes(out[8..12].try_into().unwrap()),
            0x2222_2222
        );
        assert_eq!(
            u32::from_le_bytes(out[12..16].try_into().unwrap()),
            0x1111_1111
        );
    }

    #[test]
    fn punpcklbw_interleaves_bytes() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.xmm[0] = u128::from_le_bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        cpu.xmm[1] = u128::from_le_bytes([
            101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
        ]);
        // punpcklbw xmm0, xmm1  (66 0F 60 /r, modrm=11 000 001)
        m.write_init(CODE, &[0x66, 0x0F, 0x60, 0xC1]).unwrap();
        cpu.exec(&mut m);
        let out = cpu.xmm[0].to_le_bytes();
        assert_eq!(
            out,
            [
                1, 101, 2, 102, 3, 103, 4, 104, 5, 105, 6, 106, 7, 107, 8, 108
            ]
        );
    }

    #[test]
    fn shufps_selects_lanes() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        let d: [u32; 4] = [10, 20, 30, 40];
        let s: [u32; 4] = [50, 60, 70, 80];
        let (mut db, mut sb) = ([0u8; 16], [0u8; 16]);
        for i in 0..4 {
            db[i * 4..i * 4 + 4].copy_from_slice(&d[i].to_le_bytes());
            sb[i * 4..i * 4 + 4].copy_from_slice(&s[i].to_le_bytes());
        }
        cpu.xmm[0] = u128::from_le_bytes(db);
        cpu.xmm[1] = u128::from_le_bytes(sb);
        // shufps xmm0, xmm1, imm  (0F C6 /r ib, modrm=11 000 001):
        // lane0<-dst[2], lane1<-dst[3], lane2<-src[0], lane3<-src[1]
        let imm = 0b01_00_11_10u8;
        m.write_init(CODE, &[0x0F, 0xC6, 0xC1, imm]).unwrap();
        cpu.exec(&mut m);
        let out = cpu.xmm[0].to_le_bytes();
        assert_eq!(
            u32::from_le_bytes(out[0..4].try_into().unwrap()),
            30,
            "lane0 <- dst[2]"
        );
        assert_eq!(
            u32::from_le_bytes(out[4..8].try_into().unwrap()),
            40,
            "lane1 <- dst[3]"
        );
        assert_eq!(
            u32::from_le_bytes(out[8..12].try_into().unwrap()),
            50,
            "lane2 <- src[0]"
        );
        assert_eq!(
            u32::from_le_bytes(out[12..16].try_into().unwrap()),
            60,
            "lane3 <- src[1]"
        );
    }

    #[test]
    fn pslldq_shifts_whole_register_by_bytes() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.xmm[0] = u128::from_le_bytes([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]);
        // pslldq xmm0, 2  (66 0F 73 /7 ib, modrm=11 111 000)
        m.write_init(CODE, &[0x66, 0x0F, 0x73, 0xF8, 0x02]).unwrap();
        cpu.exec(&mut m);
        let out = cpu.xmm[0].to_le_bytes();
        assert_eq!(&out[0..2], &[0, 0], "low 2 bytes are zero-filled");
        assert_eq!(
            &out[2..16],
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            "bytes shifted up by 2, top 2 bytes dropped"
        );
    }

    // ---- x87 FPU ----

    #[test]
    fn fld_fadd_fstp_m64_roundtrip() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        let (a, b, c) = (0x1_2000u64, 0x1_2008u64, 0x1_2010u64);
        m.write_init(a, &2.5f64.to_le_bytes()).unwrap();
        m.write_init(b, &4.0f64.to_le_bytes()).unwrap();
        cpu.gpr[RBX] = a;
        cpu.gpr[RCX] = b;
        cpu.gpr[RDX] = c;
        // fld qword [rbx]  (DD /0, modrm=00 000 011)
        // fadd qword [rcx] (DC /0, modrm=00 000 001)
        // fstp qword [rdx] (DD /3, modrm=00 011 010)
        m.write_init(CODE, &[0xDD, 0x03, 0xDC, 0x01, 0xDD, 0x1A])
            .unwrap();
        cpu.exec(&mut m); // fld
        cpu.exec(&mut m); // fadd
        cpu.exec(&mut m); // fstp
        assert_eq!(m.read_u64(c).unwrap(), 6.5f64.to_bits());
        assert_eq!(cpu.fpu_top, 0, "FLD's push and FSTP's pop must cancel out");
    }

    #[test]
    fn fmulp_multiplies_and_pops() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        let (p1, p2) = (0x1_2000u64, 0x1_2008u64);
        m.write_init(p1, &2.0f64.to_le_bytes()).unwrap();
        m.write_init(p2, &3.0f64.to_le_bytes()).unwrap();
        cpu.gpr[RBX] = p1;
        cpu.gpr[RCX] = p2;
        // fld qword [rbx] ; fld qword [rcx] ; fmulp st(1), st(0)  (DE C9)
        m.write_init(CODE, &[0xDD, 0x03, 0xDD, 0x01, 0xDE, 0xC9])
            .unwrap();
        cpu.exec(&mut m);
        cpu.exec(&mut m);
        cpu.exec(&mut m);
        assert_eq!(cpu.st_get(0), 6.0);
        assert_eq!(cpu.fpu_top, 7, "FMULP pops one value off the stack");
    }

    #[test]
    fn fild_fsqrt_int_to_float() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        let p = 0x1_2000u64;
        m.write_init(p, &16i32.to_le_bytes()).unwrap();
        cpu.gpr[RBX] = p;
        // fild dword [rbx]  (DB /0, modrm=00 000 011) ; fsqrt  (D9 FA)
        m.write_init(CODE, &[0xDB, 0x03, 0xD9, 0xFA]).unwrap();
        cpu.exec(&mut m); // fild
        cpu.exec(&mut m); // fsqrt
        assert_eq!(cpu.st_get(0), 4.0);
    }

    #[test]
    fn fld1_fldz_constants() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        // fld1 (D9 E8) ; fldz (D9 EE)
        m.write_init(CODE, &[0xD9, 0xE8, 0xD9, 0xEE]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.st_get(0), 1.0);
        cpu.exec(&mut m);
        assert_eq!(cpu.st_get(0), 0.0, "FLDZ pushes 0.0 as the new ST(0)");
        assert_eq!(cpu.st_get(1), 1.0, "FLD1's value is still ST(1)");
    }

    #[test]
    fn fcomi_sets_eflags_for_less_greater_equal() {
        let mut m = mem();
        let (p_one, p_two) = (0x1_2000u64, 0x1_2008u64);
        m.write_init(p_one, &1.0f64.to_le_bytes()).unwrap();
        m.write_init(p_two, &2.0f64.to_le_bytes()).unwrap();
        // fld qword [rbx] (-> ST(1) once the second fld runs)
        // fld qword [rcx] (-> ST(0))
        // fcomi st(0), st(1)  (DB F1)
        let code = [0xDD, 0x03, 0xDD, 0x01, 0xDB, 0xF1];

        // ST(0) = 1.0, ST(1) = 2.0: ST(0) < ST(1) sets CF, clears ZF.
        let mut less = X86Interp::new(CODE, STACK);
        less.gpr[RBX] = p_two;
        less.gpr[RCX] = p_one;
        m.write_init(CODE, &code).unwrap();
        less.exec(&mut m);
        less.exec(&mut m);
        less.exec(&mut m);
        assert!(less.flags.cf, "ST(0)=1.0 < ST(1)=2.0 sets CF");
        assert!(!less.flags.zf);

        // ST(0) = 2.0, ST(1) = 1.0: ST(0) > ST(1) clears both CF and ZF.
        let mut greater = X86Interp::new(CODE, STACK);
        greater.gpr[RBX] = p_one;
        greater.gpr[RCX] = p_two;
        m.write_init(CODE, &code).unwrap();
        greater.exec(&mut m);
        greater.exec(&mut m);
        greater.exec(&mut m);
        assert!(!greater.flags.cf, "ST(0)=2.0 > ST(1)=1.0 clears CF");
        assert!(!greater.flags.zf);

        // ST(0) = ST(1) = 1.0: equal operands clear CF and set ZF.
        let mut equal = X86Interp::new(CODE, STACK);
        equal.gpr[RBX] = p_one;
        equal.gpr[RCX] = p_one;
        m.write_init(CODE, &code).unwrap();
        equal.exec(&mut m);
        equal.exec(&mut m);
        equal.exec(&mut m);
        assert!(!equal.flags.cf);
        assert!(equal.flags.zf, "equal operands set ZF");
    }

    #[test]
    fn fistp_rounds_per_control_word_truncate_mode() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        let (cw_addr, val_addr, out_addr) = (0x1_2000u64, 0x1_2008u64, 0x1_2010u64);
        m.write_init(cw_addr, &0x0F7Fu16.to_le_bytes()).unwrap(); // default 0x037F with RC=11 (truncate)
        m.write_init(val_addr, &3.75f64.to_le_bytes()).unwrap();
        cpu.gpr[RBX] = cw_addr;
        cpu.gpr[RCX] = val_addr;
        cpu.gpr[RDX] = out_addr;
        // fldcw [rbx]        (D9 /5, modrm=00 101 011)
        // fld qword [rcx]    (DD /0, modrm=00 000 001)
        // fistp dword [rdx]  (DB /3, modrm=00 011 010)
        m.write_init(CODE, &[0xD9, 0x2B, 0xDD, 0x01, 0xDB, 0x1A])
            .unwrap();
        cpu.exec(&mut m); // fldcw
        cpu.exec(&mut m); // fld
        cpu.exec(&mut m); // fistp
        assert_eq!(
            m.read_u32(out_addr).unwrap() as i32,
            3,
            "round-toward-zero (FLDCW RC=11) truncates 3.75 to 3"
        );
    }

    #[test]
    fn fxch_swaps_st0_and_st1() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        let (p1, p2) = (0x1_2000u64, 0x1_2008u64);
        m.write_init(p1, &1.0f64.to_le_bytes()).unwrap();
        m.write_init(p2, &2.0f64.to_le_bytes()).unwrap();
        cpu.gpr[RBX] = p1;
        cpu.gpr[RCX] = p2;
        // fld qword [rbx] ; fld qword [rcx] ; fxch st(1)  (D9 C9)
        m.write_init(CODE, &[0xDD, 0x03, 0xDD, 0x01, 0xD9, 0xC9])
            .unwrap();
        cpu.exec(&mut m); // ST(0) = 1.0
        cpu.exec(&mut m); // ST(0) = 2.0, ST(1) = 1.0
        cpu.exec(&mut m); // fxch
        assert_eq!(cpu.st_get(0), 1.0);
        assert_eq!(cpu.st_get(1), 2.0);
    }

    #[test]
    fn endbr64_and_long_nop_are_nops() {
        let mut m = mem();
        // endbr64 (gcc emits it at every function entry under -fcf-protection).
        let cpu = run_one(&mut m, &[0xF3, 0x0F, 0x1E, 0xFA]);
        assert_eq!(cpu.rip, CODE + 4);
        // The canonical data16 cs-prefixed 10-byte NOP from gcc's padding.
        let mut m = mem();
        let cpu = run_one(
            &mut m,
            &[0x66, 0x66, 0x2E, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00],
        );
        assert_eq!(cpu.rip, CODE + 11);
    }

    #[test]
    fn fs_segment_override_adds_fs_base() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.fs_base = 0x1_8000; // TLS block inside the test region
        m.write_init(0x1_8028, &0xfeed_face_cafe_f00du64.to_le_bytes())
            .unwrap();
        // mov rax, fs:[0x28] — the stack-protector canary load.
        m.write_init(
            CODE,
            &[0x64, 0x48, 0x8B, 0x04, 0x25, 0x28, 0x00, 0x00, 0x00],
        )
        .unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX], 0xfeed_face_cafe_f00d);
        // The override is transient: the next instruction is fs-free.
        m.write_init(0x1_2000, &42u64.to_le_bytes()).unwrap();
        // mov rbx, [0x12000]
        m.write_init(
            CODE + 9,
            &[0x48, 0x8B, 0x1C, 0x25, 0x00, 0x20, 0x01, 0x00],
        )
        .unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RBX], 42, "seg base must not leak across instructions");
    }

    #[test]
    fn alu_accumulator_imm_forms() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 0x3d;
        // cmp al, 0x3d — sets ZF, leaves AL alone.
        m.write_init(CODE, &[0x3C, 0x3D]).unwrap();
        cpu.exec(&mut m);
        assert!(cpu.flags.zf);
        assert_eq!(cpu.gpr[RAX], 0x3d);
        // add eax, 0x100 — writes back, zero-extending to 64 bits.
        m.write_init(CODE + 2, &[0x05, 0x00, 0x01, 0x00, 0x00]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX], 0x13d);
        // test al, 0x80 — flags only.
        m.write_init(CODE + 7, &[0xA8, 0x80]).unwrap();
        cpu.exec(&mut m);
        assert!(cpu.flags.zf, "0x3d & 0x80 == 0");
        assert_eq!(cpu.gpr[RAX], 0x13d, "TEST must not write back");
    }

    #[test]
    fn rotates_and_shift_by_one() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RDX] = 0x8000_0000_0000_0001;
        // rol rdx, 0x11 (glibc's PTR_MANGLE uses exactly this)
        m.write_init(CODE, &[0x48, 0xC1, 0xC2, 0x11]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RDX], 0x8000_0000_0000_0001u64.rotate_left(0x11));
        // ror rdx, 0x11 undoes it.
        m.write_init(CODE + 4, &[0x48, 0xC1, 0xCA, 0x11]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RDX], 0x8000_0000_0000_0001);
        // sar rsi, 1 (the D1 shift-by-one form).
        cpu.gpr[RSI] = 0x8000_0000_0000_0002;
        m.write_init(CODE + 8, &[0x48, 0xD1, 0xFE]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RSI], 0xC000_0000_0000_0001, "arithmetic: sign fills");
    }

    #[test]
    fn xchg_rax_r8_is_not_a_nop() {
        // `49 90` is XCHG rax,r8 (REX.B re-points the "NOP" encoding at r8);
        // treating it as a NOP silently loses a register (found booting
        // Alpine's busybox, which returns values through exactly this).
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 1;
        cpu.gpr[R8] = 2;
        m.write_init(CODE, &[0x49, 0x90]).unwrap();
        cpu.exec(&mut m);
        assert_eq!((cpu.gpr[RAX], cpu.gpr[R8]), (2, 1));
        // Plain 0x90 stays a NOP.
        m.write_init(CODE + 2, &[0x90]).unwrap();
        cpu.exec(&mut m);
        assert_eq!((cpu.gpr[RAX], cpu.gpr[R8]), (2, 1));
    }

    #[test]
    fn alu_8bit_and_or_forms() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = 0xF0;
        cpu.gpr[RCX] = 0x3C;
        // and al, cl (20 C8) ; or al, cl (08 C8)
        m.write_init(CODE, &[0x20, 0xC8, 0x08, 0xC8]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX], 0x30);
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX], 0x3C);
    }

    #[test]
    fn adc_sbb_carry_chains() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RAX] = u64::MAX;
        cpu.gpr[RDX] = 5;
        // add rax, 1 (sets CF) ; adc rdx, 0 (consumes it)
        m.write_init(CODE, &[0x48, 0x83, 0xC0, 0x01, 0x48, 0x83, 0xD2, 0x00])
            .unwrap();
        cpu.exec(&mut m);
        assert!(cpu.flags.cf, "add wrapped");
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RDX], 6, "adc added the carry");
        // sub rax, 1 (0 - 1 borrows) ; sbb rdx, 0 (consumes the borrow)
        cpu.gpr[RAX] = 0;
        m.write_init(CODE + 8, &[0x48, 0x83, 0xE8, 0x01, 0x48, 0x83, 0xDA, 0x00])
            .unwrap();
        cpu.exec(&mut m);
        assert!(cpu.flags.cf, "sub borrowed");
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RDX], 5, "sbb subtracted the borrow");
    }

    #[test]
    fn eight_bit_flags_are_width_accurate() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        // add al, 1 with AL = 0xFF: the 8-bit result is 0 → ZF and CF set.
        cpu.gpr[RAX] = 0xFF;
        m.write_init(CODE, &[0x04, 0x01]).unwrap();
        cpu.exec(&mut m);
        assert!(cpu.flags.zf, "8-bit wrap to zero sets ZF");
        assert!(cpu.flags.cf, "8-bit carry out sets CF");
        assert_eq!(cpu.gpr[RAX] & 0xff, 0);
        // cmp al, 1 with AL = 0x81: result 0x80 → SF at bit 7.
        cpu.gpr[RAX] = 0x81;
        m.write_init(CODE + 2, &[0x3C, 0x01]).unwrap();
        cpu.exec(&mut m);
        assert!(cpu.flags.sf, "8-bit SF comes from bit 7");
    }

    #[test]
    fn group2_8bit_shift() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RDI] = 0xAB;
        // shr dil, 4 (40 C0 EF 04 — REX-extended 8-bit register)
        m.write_init(CODE, &[0x40, 0xC0, 0xEF, 0x04]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RDI], 0x0A);
    }

    #[test]
    fn sse_half_moves() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        m.write_init(0x1_2000, &0x1111_2222_3333_4444u64.to_le_bytes())
            .unwrap();
        cpu.xmm[0] = 0xAAAA_BBBB_CCCC_DDDD_0123_4567_89AB_CDEF;
        // movhps xmm0, [0x12000]: high half loaded, low preserved.
        m.write_init(CODE, &[0x0F, 0x16, 0x04, 0x25, 0x00, 0x20, 0x01, 0x00])
            .unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.xmm[0], 0x1111_2222_3333_4444_0123_4567_89AB_CDEF);
        // movlps [0x12008], xmm0: stores the (preserved) low half.
        m.write_init(CODE + 8, &[0x0F, 0x13, 0x04, 0x25, 0x08, 0x20, 0x01, 0x00])
            .unwrap();
        cpu.exec(&mut m);
        let mut b = [0u8; 8];
        m.read(0x1_2008, &mut b).unwrap();
        assert_eq!(u64::from_le_bytes(b), 0x0123_4567_89AB_CDEF);
        // movhlps xmm1, xmm0 (reg form): xmm1.low <- xmm0.high.
        cpu.xmm[1] = u128::MAX;
        m.write_init(CODE + 16, &[0x0F, 0x12, 0xC8]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(
            cpu.xmm[1],
            0xFFFF_FFFF_FFFF_FFFF_1111_2222_3333_4444,
            "low half replaced, high preserved"
        );
    }

    #[test]
    fn mov_mem_imm16_consumes_exactly_two_immediate_bytes() {
        // `66 C7 /0` is `mov word ptr, imm16`. Reading a fixed imm32 here
        // over-consumed by two bytes and desynced every following instruction
        // (the bug that crashed V8's JIT). Verify the length and the value.
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        let data = 0x1_2000u64;
        cpu.gpr[RBP] = data;
        // mov word [rbp+0], 0x1234  (66 C7 45 00 34 12) — exactly 6 bytes.
        m.write_init(CODE, &[0x66, 0xC7, 0x45, 0x00, 0x34, 0x12]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.rip, CODE + 6, "imm16 form is 6 bytes, not 8");
        assert_eq!(m.read_vec(data, 2).unwrap(), vec![0x34, 0x12]);
    }

    #[test]
    fn andnpd_and_orpd() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        // andnpd xmm0, xmm1  (66 0F 55 C1): xmm0 <- ~xmm0 & xmm1.
        cpu.xmm[0] = 0x0F;
        cpu.xmm[1] = 0xFF;
        m.write_init(CODE, &[0x66, 0x0F, 0x55, 0xC1]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.xmm[0], (!0x0Fu128) & 0xFF);
        // orpd xmm0, xmm1  (66 0F 56 C1): xmm0 <- xmm0 | xmm1.
        cpu.xmm[0] = 0x0F;
        cpu.xmm[1] = 0xF0;
        cpu.rip = CODE;
        m.write_init(CODE, &[0x66, 0x0F, 0x56, 0xC1]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.xmm[0], 0xFF);
    }

    #[test]
    fn cmpsd_predicate_masks() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.xmm[0] = u128::from(3.0f64.to_bits());
        cpu.xmm[1] = u128::from(3.0f64.to_bits());
        // cmpsd xmm0, xmm1, 0 (EQ): equal → low quadword all ones, high kept.
        m.write_init(CODE, &[0xF2, 0x0F, 0xC2, 0xC1, 0x00]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.xmm[0] as u64, u64::MAX, "3==3 is true");
        assert_eq!(cpu.xmm[0] >> 64, 0, "high quadword preserved");
        // cmpsd xmm0, xmm1, 1 (LT): 3<3 false → all zeros.
        cpu.xmm[0] = u128::from(3.0f64.to_bits());
        cpu.rip = CODE;
        m.write_init(CODE, &[0xF2, 0x0F, 0xC2, 0xC1, 0x01]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.xmm[0] as u64, 0, "3<3 is false");
    }

    #[test]
    fn packuswb_saturates() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        // dst words: [0x0005, 0x1234, 0, …]; 5 stays, 0x1234 saturates to 255.
        cpu.xmm[0] = 0x1234_0005;
        // src word0 = 0x8000 (negative i16) saturates to 0 (unsigned).
        cpu.xmm[1] = 0x8000;
        // packuswb xmm0, xmm1  (66 0F 67 C1).
        m.write_init(CODE, &[0x66, 0x0F, 0x67, 0xC1]).unwrap();
        cpu.exec(&mut m);
        // low bytes from dst: 0x05, 0xFF, then zeros; src half all zero.
        assert_eq!(cpu.xmm[0], 0xFF05);
    }

    #[test]
    fn ret_imm16_pops_and_releases_args() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RSP] = STACK;
        m.write_init(STACK, &0x1_3000u64.to_le_bytes()).unwrap();
        // ret 0x10  (C2 10 00): pop target, then rsp += 0x10.
        m.write_init(CODE, &[0xC2, 0x10, 0x00]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.rip, 0x1_3000);
        assert_eq!(cpu.gpr[RSP], STACK + 8 + 0x10);
    }

    #[test]
    fn pop_rm_into_register() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RSP] = STACK;
        m.write_init(STACK, &0xDEAD_BEEFu64.to_le_bytes()).unwrap();
        // pop rax  (8F C0): 8F /0 with a register operand.
        m.write_init(CODE, &[0x8F, 0xC0]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RAX], 0xDEAD_BEEF);
        assert_eq!(cpu.gpr[RSP], STACK + 8);
    }

    #[test]
    fn psrlw_psraw_psllw_word_shifts() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        // psrlw xmm0, 4  (66 0F 71 D0 04): logical word shift right.
        cpu.xmm[0] = 0x8000_0010_0000_ffff_u128 << 64 | 0xffff_0080_0010_8000;
        m.write_init(CODE, &[0x66, 0x0F, 0x71, 0xD0, 0x04]).unwrap();
        cpu.exec(&mut m);
        // each 16-bit lane >> 4, zero-filled.
        assert_eq!(cpu.xmm[0], 0x0800_0001_0000_0fff_u128 << 64 | 0x0fff_0008_0001_0800);
        // psraw xmm0, 4  (66 0F 71 E0 04): arithmetic — 0x8000 → 0xF800.
        cpu.xmm[0] = 0x8000;
        cpu.rip = CODE;
        m.write_init(CODE, &[0x66, 0x0F, 0x71, 0xE0, 0x04]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.xmm[0], 0xF800);
        // psllw xmm0, 4  (66 0F 71 F0 04).
        cpu.xmm[0] = 0x0011;
        cpu.rip = CODE;
        m.write_init(CODE, &[0x66, 0x0F, 0x71, 0xF0, 0x04]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.xmm[0], 0x0110);
    }

    #[test]
    fn ldmxcsr_stmxcsr_roundtrip() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        let addr = 0x1_2000u64;
        m.write_init(addr, &0x0000_1f80u32.to_le_bytes()).unwrap();
        cpu.gpr[RAX] = addr;
        // ldmxcsr [rax]  (0F AE 10): load MXCSR from memory.
        m.write_init(CODE, &[0x0F, 0xAE, 0x10]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.mxcsr, 0x1f80);
        // stmxcsr [rax+8]  (0F AE 58 08): store it back.
        cpu.mxcsr = 0x9fc0;
        cpu.rip = CODE;
        m.write_init(CODE, &[0x0F, 0xAE, 0x58, 0x08]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(m.read_vec(addr + 8, 4).unwrap(), 0x9fc0u32.to_le_bytes());
    }

    #[test]
    fn palignr_concatenates_and_shifts() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.xmm[0] = 0x03; // dst byte 0 = 3
        cpu.xmm[1] = 0x01; // src byte 0 = 1
        // palignr xmm0, xmm1, 15  (66 0F 3A 0F C1 0F): result[1] = dst[0].
        m.write_init(CODE, &[0x66, 0x0F, 0x3A, 0x0F, 0xC1, 0x0F]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.xmm[0], 0x0300);
    }

    #[test]
    fn ptest_sets_zf_and_cf() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.xmm[0] = 0x0f;
        cpu.xmm[1] = 0xf0;
        // ptest xmm0, xmm1  (66 0F 38 17 C1): dst&src=0 → ZF; ~dst&src≠0 → !CF.
        m.write_init(CODE, &[0x66, 0x0F, 0x38, 0x17, 0xC1]).unwrap();
        cpu.exec(&mut m);
        assert!(cpu.flags.zf && !cpu.flags.cf);
        // dst covers all of src's bits → ~dst&src=0 → CF; dst&src≠0 → !ZF.
        cpu.xmm[0] = 0xff;
        cpu.xmm[1] = 0x0f;
        cpu.rip = CODE;
        cpu.exec(&mut m);
        assert!(!cpu.flags.zf && cpu.flags.cf);
    }

    #[test]
    fn pop_rm_rsp_relative_uses_post_pop_rsp() {
        // `pop [rsp+disp]` addresses the destination with RSP *after* the pop's
        // `RSP += 8` — the bug that let node's saved return address be written
        // one slot too low and later `ret` into a heap pointer.
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RSP] = STACK;
        m.write_init(STACK, &0xCAFEu64.to_le_bytes()).unwrap();
        // pop qword [rsp+8]  (8F 44 24 08): pop [STACK] (rsp→STACK+8), then store
        // to [new_rsp + 8] = [STACK+16], not [old_rsp + 8] = [STACK+8].
        m.write_init(CODE, &[0x8F, 0x44, 0x24, 0x08]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RSP], STACK + 8);
        assert_eq!(m.read_vec(STACK + 16, 8).unwrap(), 0xCAFEu64.to_le_bytes());
    }

    #[test]
    fn imul_clears_zf_and_sets_sf_pf() {
        // Two/three-operand IMUL sets SF/PF from the result and *clears* ZF even
        // for a zero result (verified against KVM), rather than leaving flags
        // stale — a `jz`/`js` after it would otherwise diverge from hardware.
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.flags.zf = true; // stale ZF must be cleared
        cpu.flags.sf = true; // stale SF must be recomputed to 0
        cpu.gpr[RAX] = 0;
        // imul rcx, rax, 5  (48 6B C8 05): result 0.
        m.write_init(CODE, &[0x48, 0x6B, 0xC8, 0x05]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.gpr[RCX], 0);
        assert!(!cpu.flags.zf && !cpu.flags.sf);
    }

    #[test]
    fn shr_multibit_sets_of_from_original_msb() {
        // `OF` is set for any nonzero shift count, not only 1-bit shifts.
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.gpr[RCX] = 0xa0; // CL, bit 7 set
        // shr cl, 5  (C0 E9 05): OF = MSB of the original operand = 1.
        m.write_init(CODE, &[0xC0, 0xE9, 0x05]).unwrap();
        cpu.exec(&mut m);
        assert!(cpu.flags.of);
    }

    #[test]
    fn syscall_sets_rcx_and_r11_like_hardware() {
        // `syscall` copies RIP→RCX and RFLAGS→R11. Leaving RCX stale silently
        // broke V8/musl trampolines that read it after the call.
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.flags.cf = true; // CF should show up in R11 (bit 0).
        m.write_init(CODE, &[0x0F, 0x05]).unwrap();
        let step = cpu.exec(&mut m);
        assert!(matches!(step, Step::Syscall));
        assert_eq!(cpu.gpr[RCX], CODE + 2, "RCX holds the post-syscall RIP");
        assert_eq!(cpu.gpr[R11] & 0x203, 0x203, "R11 = RFLAGS (reserved|IF|CF)");
    }
}
