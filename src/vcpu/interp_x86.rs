//! Software CPU interpreter backend for x86-64 guests — the portable,
//! no-acceleration fallback for `Arch::X86_64` (mirrors [`super::interp`]'s
//! aarch64 interpreter, but decodes variable-length x86 instructions instead
//! of fixed 4-byte ones).
//!
//! It implements the complete user-mode (CPL 3) instruction set of the CPU it
//! advertises through `CPUID` (see [`X86Interp::cpuid`]) and the loader's
//! `AT_HWCAP` —
//!
//! * **general purpose**: every integer instruction valid in 64-bit mode, in
//!   all operand sizes (8/16/32/64) and addressing forms (ModRM/SIB/disp,
//!   RIP-relative, `0x67` 32-bit addressing, `fs:`/`gs:` segment bases), with
//!   exact `CF`/`PF`/`AF`/`ZF`/`SF`/`OF` — including the architecturally
//!   "undefined" flag results, which follow what real hardware deterministically
//!   produces (verified against real x86-64 execution, see
//!   `tests/x86_diff.rs`). Shifts/rotates (incl. `RCL`/`RCR`, `SHLD`/`SHRD`)
//!   honor the count-masking and count-0 rules; `BT*` with a register bit
//!   offset address the full bit string; `DIV`/`IDIV` raise `#DE`; `LOCK`
//!   is validated (`#UD` on a non-lockable form). Privileged/IO instructions
//!   raise `#GP` (`SIGSEGV`), `INT3` `#BP`, `UD2`/invalid encodings `#UD`.
//! * **x87** (`interp_x86/x87.rs`): the full FPU with true 80-bit extended
//!   precision, precision and rounding control, exception flags, the tag word
//!   and stack faults, the environment/state save/restore instructions, BCD,
//!   and transcendentals computed in extended precision
//!   (`interp_x86/x87math.rs`).
//! * **MMX, SSE, SSE2** (`interp_x86/sse.rs`), with `MXCSR` rounding,
//!   `DAZ`/`FTZ` and exception flags, and the x86-64-v2 extensions **SSE3,
//!   SSSE3, SSE4.1, SSE4.2** (`interp_x86/sse/sse4.rs`: incl. `PCMPxSTRx`,
//!   `CRC32`, `ROUND*`, `DPPS`, …).
//! * `CPUID`, `RDTSC`/`RDTSCP`, `RDRAND`, `CMPXCHG16B`, `POPCNT`,
//!   `FXSAVE`/`FXRSTOR`, `LAHF`/`SAHF`, fences/prefetches/`CLFLUSH`, and
//!   `SYSCALL`.
//!
//! Instructions outside the advertised feature set (AVX, BMI, `MOVBE`, …)
//! decode as `#UD`, exactly as on a CPU without them. The `0x66`/`0xF2`/
//! `0xF3` prefixes select among SIMD opcode variants ("mandatory prefixes")
//! when an SSE opcode follows, and operand size / `REP` otherwise.

// Opcode dispatch: many arms legitimately share a body (aliases, groups),
// and the register-vs-memory ModRM split reads best as a `match`.
#![allow(clippy::match_same_arms, clippy::single_match_else)]

use crate::abi::Arch;

use std::time::{Duration, Instant};

use super::softfloat::F80;
use super::{Backend, Exit, GuestMemory, Vcpu, VcpuError};

mod sse;
#[doc(hidden)]
pub mod testing;
mod x87;
mod x87math;

/// Upper bound on instructions executed per `run()` call before yielding —
/// mirrors [`super::interp`]'s guard against a runaway guest loop.
const MAX_STEPS: u64 = 50_000_000;

/// How often (in instructions) [`X86Interp::run`] polls the wall clock for a
/// time-quantum expiry. A clock read per instruction would dominate the
/// interpreter's cost, so we amortize it: a power of two makes the test a cheap
/// mask. The residual overrun (up to this many instructions past the deadline)
/// is negligible next to a millisecond quantum.
const QUANTUM_STRIDE: u64 = 4096;

/// The architectural maximum instruction length; anything longer is `#GP`.
const MAX_INSN_LEN: usize = 15;

/// Guest page size (for instruction fetches that straddle a page boundary).
const PAGE: u64 = super::mem::PAGE_SIZE;

/// [`X86Interp::code_page`] when no page is cached (not page-aligned, so it
/// never matches a real page).
const NO_PAGE: u64 = u64::MAX;

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

/// The `MXCSR` bits this CPU implements (`FXSAVE`'s `MXCSR_MASK` field): every
/// flag/mask/rounding bit of the low 16, including `DAZ` (bit 6). Loading a
/// set bit outside it (`LDMXCSR`/`FXRSTOR`) is a `#GP`.
const MXCSR_MASK: u32 = 0xffff;

/// The `RFLAGS` system bits user code may toggle with `POPF`: `NT` (14), `AC`
/// (18) and `ID` (21 — the classic "CPUID supported" probe).
const RFLAGS_SYS_MASK: u32 = (1 << 14) | (1 << 18) | (1 << 21);

/// Segment selectors a Linux x86-64 user task observes (`MOV r/m, Sreg`,
/// `PUSH FS/GS`): the flat 64-bit user code segment and user data segment;
/// `DS`/`ES`/`FS`/`GS` hold the null selector (their bases come from MSRs).
const USER_CS: u16 = 0x33;
const USER_SS: u16 = 0x2b;

/// `CPUID` leaf 1 `ECX`: SSE3 (0), SSSE3 (9), CX16 (13), SSE4.1 (19), SSE4.2
/// (20), POPCNT (23), RDRAND (30) — with LAHF-SAHF (`0x8000_0001` `ECX`) the
/// whole x86-64-v2 level.
const CPUID1_ECX: u32 = 1 | (1 << 9) | (1 << 13) | (1 << 19) | (1 << 20) | (1 << 23) | (1 << 30);

/// `CPUID` leaf 1 `EDX`: FPU (0), PSE (3), TSC (4), MSR (5), PAE (6), CX8
/// (8), PGE (13), CMOV (15), CLFSH (19), MMX (23), FXSR (24), SSE (25), SSE2
/// (26) — the same bits the loader reports in `AT_HWCAP`.
const CPUID1_EDX: u32 = (1 << 0)
    | (1 << 3)
    | (1 << 4)
    | (1 << 5)
    | (1 << 6)
    | (1 << 8)
    | (1 << 13)
    | (1 << 15)
    | (1 << 19)
    | (1 << 23)
    | (1 << 24)
    | (1 << 25)
    | (1 << 26);

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
#[derive(Debug)]
enum Step {
    /// Advance `rip` to the address just past the decoded instruction.
    Next,
    /// Instruction already set `rip` (branch/call/ret/jmp); do not auto-advance.
    Branched,
    /// `syscall` — hand control to the kernel. `rip` stays on the `syscall`
    /// opcode; the kernel advances it via [`Vcpu::set_syscall_ret`].
    Syscall,
    /// `#UD`.
    Illegal,
    /// A load/store/fetch touched bad guest memory.
    Fault { addr: u64, write: bool },
    /// A non-memory CPU exception (see [`Trap`]).
    Trap(Trap),
}

/// The non-memory exceptions a user-mode instruction can raise, each of which
/// Linux turns into a specific signal ([`Trap::signal`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Trap {
    /// `#DE`: `DIV`/`IDIV` by zero or with a quotient that doesn't fit.
    Divide,
    /// `#BP`/`#DB`: `INT3` / `INT1`.
    Breakpoint,
    /// `#GP`: a privileged instruction (`HLT`, `CLI`, `IN`/`OUT`, `INT n`, …)
    /// at CPL 3, or a `#GP`-raising operand (`LDMXCSR` reserved bits, a
    /// misaligned `MOVAPS`, an over-long instruction, …). Linux delivers
    /// `SIGSEGV` with `si_addr == 0`.
    Protection,
    /// `#MF`: a pending unmasked x87 exception, raised by the next waiting x87
    /// instruction.
    X87,
    /// `#XM`: an unmasked SSE floating-point exception.
    Simd,
}

impl Trap {
    /// The Linux signal number this exception is delivered as.
    const fn signal(self) -> i32 {
        match self {
            Self::Divide | Self::X87 | Self::Simd => 8, // SIGFPE
            Self::Breakpoint => 5,                      // SIGTRAP
            Self::Protection => 11,                     // SIGSEGV
        }
    }
}

/// The six arithmetic status flags (`CF`/`PF`/`AF`/`ZF`/`SF`/`OF`), kept as
/// separate bools so the common "compute, then test one flag" path never has
/// to pack/unpack an `RFLAGS` word.
#[derive(Default, Clone, Copy, Debug)]
#[allow(clippy::struct_excessive_bools)]
struct Flags {
    cf: bool,
    pf: bool,
    af: bool,
    zf: bool,
    sf: bool,
    of: bool,
}

/// Decoded REX prefix bits (all `false` when the instruction has none).
#[derive(Clone, Copy, Default, Debug)]
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

/// The prefixes decoded in front of an opcode. (The address-size prefix and
/// the segment base live on [`X86Interp`] itself, because effective-address
/// computation needs them deep inside ModRM decoding.)
#[derive(Clone, Copy, Default, Debug)]
#[allow(clippy::struct_excessive_bools)]
struct Pfx {
    rex: Rex,
    /// A REX prefix immediately precedes the opcode (changes the 8-bit
    /// register names `SPL`/`BPL`/`SIL`/`DIL` vs `AH`/`CH`/`DH`/`BH`).
    has_rex: bool,
    /// `0x66`: operand size 16 (or the `66` SIMD mandatory prefix).
    opsize: bool,
    /// `0` none, `1` = `0xF3` (`REP`/`REPE`), `2` = `0xF2` (`REPNE`); the last
    /// one wins.
    rep: u8,
    /// `0xF0`.
    lock: bool,
}

impl Pfx {
    /// The operand size for a non-byte operation: `REX.W` → 64, `0x66` → 16,
    /// else 32.
    const fn width(self) -> u32 {
        if self.rex.w {
            64
        } else if self.opsize {
            16
        } else {
            32
        }
    }

    /// The operand size of a default-64 operation (stack ops, near branches):
    /// 16 under `0x66` (without `REX.W`), else 64.
    const fn stack_width(self) -> u32 {
        if self.opsize && !self.rex.w { 16 } else { 64 }
    }
}

/// A decoded ModRM byte (plus any SIB/displacement that followed it).
#[derive(Clone, Copy, Debug)]
struct ModRm {
    /// The `reg` field, extended by `REX.R`.
    reg: usize,
    kind: RmKind,
}

impl ModRm {
    /// The raw 3-bit `reg` field (opcode extension of group encodings).
    const fn ext(&self) -> usize {
        self.reg & 7
    }
}

/// The r/m operand before RIP-relative addresses are resolved (resolving
/// requires knowing the address of the *end* of the instruction, which isn't
/// known until any trailing immediate has also been decoded).
#[derive(Clone, Copy, Debug)]
enum RmKind {
    Reg(usize),
    /// An effective address (offset within the segment; already truncated to
    /// 32 bits under the `0x67` prefix). The segment base is added when it is
    /// turned into an [`Operand`].
    Mem(u64),
    /// `[rip + disp]`; resolved against the end-of-instruction address.
    MemRip(i64),
}

/// A fully-resolved operand.
#[derive(Clone, Copy, Debug)]
enum Operand {
    Reg(usize),
    /// The high byte (bits 15:8) of `gpr[r]` — `AH`/`CH`/`DH`/`BH`, only
    /// reachable for an 8-bit operand with `r` in `0..4` and no `REX` prefix.
    Reg8Hi(usize),
    /// A linear (segment-based) address.
    Mem(u64),
}

/// Map a ModRM `reg` (or `rm` in register form) field to the 8-bit operand it
/// names: without a `REX` prefix, indices 4..=7 are `AH`/`CH`/`DH`/`BH` (the
/// high byte of `RAX..RBX`) rather than the low byte of `RSP..RDI`.
fn reg8_operand(r: usize, has_rex: bool) -> Operand {
    if !has_rex && (4..=7).contains(&r) {
        Operand::Reg8Hi(r - 4)
    } else {
        Operand::Reg(r)
    }
}

/// Arithmetic/logical operation selected by an ALU opcode or a group-1 `/r`
/// field (in that field's `/0../7` order).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AluOp {
    Add,
    Or,
    Adc,
    Sbb,
    And,
    Sub,
    Xor,
    Cmp,
    Test,
}

impl AluOp {
    const fn from_ext(e: usize) -> Self {
        match e & 7 {
            0 => Self::Add,
            1 => Self::Or,
            2 => Self::Adc,
            3 => Self::Sbb,
            4 => Self::And,
            5 => Self::Sub,
            6 => Self::Xor,
            _ => Self::Cmp,
        }
    }

    /// Whether the result is written back (`CMP`/`TEST` only set flags).
    const fn stores(self) -> bool {
        !matches!(self, Self::Cmp | Self::Test)
    }
}

/// The four `BT`/`BTS`/`BTR`/`BTC` variants.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BitTestOp {
    Bt,
    Bts,
    Btr,
    Btc,
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

/// Parity flag: `true` iff the low byte of `v` has an even number of 1 bits.
fn parity(v: u64) -> bool {
    (v as u8).count_ones().is_multiple_of(2)
}

/// The sign bit of `v` interpreted as a `width`-bit integer.
const fn sign_bit(v: u64, width: u32) -> bool {
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
use fetch;

/// Map a guest memory read error to the [`Step`] it raises.
fn rd_fault(addr: u64) -> Step {
    Step::Fault { addr, write: false }
}

/// Map a guest memory write error to the [`Step`] it raises.
fn wr_fault(e: &super::MemError) -> Step {
    Step::Fault {
        addr: e.fault_addr(),
        write: true,
    }
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
    /// `RFLAGS` system bits a CPL-3 `POPF` may change and `PUSHF` reports
    /// verbatim: `NT` (14), `AC` (18), `ID` (21). Nothing here acts on them
    /// (alignment checking is not modeled).
    rflags_sys: u32,
    /// FS.base, set by `arch_prctl(ARCH_SET_FS, ...)` (thread pointer).
    fs_base: u64,
    /// GS.base (`arch_prctl(ARCH_SET_GS)`); zero unless the kernel sets it.
    gs_base: u64,
    /// The `0x67` address-size prefix on the instruction being executed:
    /// effective addresses truncate to 32 bits (and string ops/`LOOP` use
    /// `ECX`/`ESI`/`EDI`). Transient — reset at each `exec`.
    addr32: bool,
    /// Segment base of the instruction being executed — nonzero only under an
    /// `fs:`/`gs:` override (`0x64`/`0x65`, how x86-64 reaches TLS: `mov
    /// %fs:0x28, ...` is every stack-canary check). Added to effective
    /// addresses when they become linear addresses. Transient, like `addr32`.
    /// In long mode CS/DS/ES/SS are zero-based, so their override prefixes
    /// select a zero base.
    seg_base: u64,
    /// The bytes of the instruction being executed (up to [`MAX_INSN_LEN`]),
    /// fetched once per instruction from `rip`, and how many of them are
    /// readable+executable (`ilen`); decoding reads from here instead of
    /// translating every byte.
    ibuf: [u8; 16],
    ilen: u8,
    /// A copy of the executable page instructions are currently being fetched
    /// from (its base in `code_page`, [`NO_PAGE`] when none): straight-line
    /// code then decodes without a permission check and page walk per
    /// instruction. Valid only within one [`Vcpu::run`] (the kernel may
    /// remap/rewrite memory between runs); every guest store through
    /// [`X86Interp::store`] that touches the page drops it, so self-modifying
    /// code stays coherent.
    code_page: u64,
    code: Box<[u8; PAGE as usize]>,
    /// The x87 register stack, physically indexed (`R0..R7`; `ST(i)` lives at
    /// `st[(fpu_top + i) & 7]` — see [`X86Interp::st_get`]). Each register
    /// holds a true 80-bit extended-precision value (its `m80` encoding); the
    /// MMX registers alias their low 64 bits.
    st: [F80; 8],
    /// The status word's `TOP` field.
    fpu_top: u8,
    /// Status-word condition codes `C0`/`C1`/`C2`/`C3`.
    fpu_c0: bool,
    fpu_c1: bool,
    fpu_c2: bool,
    fpu_c3: bool,
    /// The control word (`FLDCW`/`FNSTCW`): exception masks (bits 0-5),
    /// precision control (8-9) and rounding control (10-11).
    fpu_cw: u16,
    /// Accumulated x87 exception flags (`IE`/`DE`/`ZE`/`OE`/`UE`/`PE`, bits 0-5
    /// — the [`super::softfloat`] flag layout) plus the stack-fault flag `SF`
    /// (bit 6), as the status word reports them.
    fpu_flags: u16,
    /// The physical x87 registers holding a value (bit `i` set = `R_i` is not
    /// empty) — the "abridged" tag `FXSAVE` stores. The full 2-bit-per-register
    /// tag word `FNSTENV`/`FNSAVE` report is derived from it and the register
    /// contents.
    fpu_tag: u8,
    /// Last x87 instruction's opcode (11 bits), instruction pointer and data
    /// pointer, as `FNSTENV`/`FNSAVE`/`FXSAVE` report them.
    fpu_fop: u16,
    fpu_fip: u64,
    fpu_fdp: u64,
    /// Free-running counter behind `RDTSC`/`RDTSCP` — see
    /// [`X86Interp::rdtsc_tick`].
    tsc: u64,
    /// PRNG state behind `RDRAND` — see [`X86Interp::rdrand`].
    prng: u64,
    /// The SSE control/status register (`LDMXCSR`/`STMXCSR`): rounding control,
    /// `FTZ`/`DAZ`, exception masks and sticky exception flags.
    mxcsr: u32,
    /// Wall-clock preemption quantum: [`X86Interp::run`] returns
    /// [`Exit::Interrupted`] once this much time has elapsed since it started,
    /// so a compute-bound guest that never syscalls still yields the CPU. `None`
    /// disables it. Cached from `NIXVM_QUANTUM_MS` at construction (see
    /// [`super::preempt_quantum`]); a `fork` clone inherits it.
    quantum: Option<Duration>,
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
            rflags_sys: 0,
            fs_base: 0,
            gs_base: 0,
            addr32: false,
            seg_base: 0,
            ibuf: [0; 16],
            ilen: 0,
            code_page: NO_PAGE,
            code: Box::new([0; PAGE as usize]),
            st: [F80(0); 8],
            fpu_top: 0,
            fpu_c0: false,
            fpu_c1: false,
            fpu_c2: false,
            fpu_c3: false,
            fpu_cw: 0x037F, // the real x87's power-on/FNINIT default control word
            fpu_flags: 0,
            fpu_tag: 0,
            fpu_fop: 0,
            fpu_fip: 0,
            fpu_fdp: 0,
            tsc: 0,
            prng: 0x9E37_79B9_7F4A_7C15, // arbitrary nonzero seed (golden-ratio constant)
            mxcsr: 0x1f80,               // the power-on default (all exceptions masked)
            quantum: super::preempt_quantum(),
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

    // ---- instruction fetch -------------------------------------------------

    /// Fetch the instruction at `rip` into [`X86Interp::ibuf`]: up to 15 bytes,
    /// stopping early at a following page that isn't executable (an
    /// instruction that actually extends into it then faults there, precisely
    /// like hardware).
    fn fill_ibuf(&mut self, mem: &GuestMemory) -> Result<(), Step> {
        let rip = self.rip;
        let page = rip & !(PAGE - 1);
        let off = (rip - page) as usize;
        if page != self.code_page {
            // NX: an instruction fetch requires EXEC on the page at rip.
            if !mem.can_exec(rip) {
                return Err(rd_fault(rip));
            }
            if mem.read(page, &mut self.code[..]).is_ok() {
                self.code_page = page;
            }
        }
        if page == self.code_page && off + MAX_INSN_LEN <= PAGE as usize {
            self.ibuf[..MAX_INSN_LEN].copy_from_slice(&self.code[off..off + MAX_INSN_LEN]);
            self.ilen = MAX_INSN_LEN as u8;
            return Ok(());
        }
        // NX: an instruction fetch requires EXEC on the page at rip. Jumping
        // to a non-executable page (the stack, a data buffer) faults here
        // rather than running whatever bytes are there.
        if !mem.can_exec(rip) {
            return Err(rd_fault(rip));
        }
        let in_page = (PAGE - (rip & (PAGE - 1))) as usize;
        let n1 = in_page.min(MAX_INSN_LEN);
        mem.read(rip, &mut self.ibuf[..n1])
            .map_err(|_| rd_fault(rip))?;
        let mut n = n1;
        if n1 < MAX_INSN_LEN {
            let next = rip.wrapping_add(n1 as u64);
            if mem.can_exec(next) && mem.read(next, &mut self.ibuf[n1..MAX_INSN_LEN]).is_ok() {
                n = MAX_INSN_LEN;
            }
        }
        self.ilen = n as u8;
        Ok(())
    }

    /// The instruction byte at `pc` (which must lie within the instruction
    /// being executed). Running off the fetched bytes is a fetch fault, or a
    /// `#GP` past the 15-byte architectural limit.
    #[inline]
    fn fetch8(&self, pc: u64) -> Result<(u8, u64), Step> {
        let off = pc.wrapping_sub(self.rip) as usize;
        if off < usize::from(self.ilen) {
            Ok((self.ibuf[off], pc + 1))
        } else if off >= MAX_INSN_LEN {
            Err(Step::Trap(Trap::Protection))
        } else {
            Err(rd_fault(pc))
        }
    }

    #[inline]
    fn fetch_n<const N: usize>(&self, pc: u64) -> Result<([u8; N], u64), Step> {
        let off = pc.wrapping_sub(self.rip) as usize;
        if off + N <= usize::from(self.ilen) {
            let mut b = [0u8; N];
            b.copy_from_slice(&self.ibuf[off..off + N]);
            Ok((b, pc + N as u64))
        } else if off + N > MAX_INSN_LEN {
            Err(Step::Trap(Trap::Protection))
        } else {
            Err(rd_fault(self.rip.wrapping_add(u64::from(self.ilen))))
        }
    }

    fn fetch_i8(&self, pc: u64) -> Result<(i8, u64), Step> {
        let (b, p) = self.fetch8(pc)?;
        Ok((b as i8, p))
    }

    fn fetch16(&self, pc: u64) -> Result<(u16, u64), Step> {
        let (b, p) = self.fetch_n::<2>(pc)?;
        Ok((u16::from_le_bytes(b), p))
    }

    fn fetch32(&self, pc: u64) -> Result<(u32, u64), Step> {
        let (b, p) = self.fetch_n::<4>(pc)?;
        Ok((u32::from_le_bytes(b), p))
    }

    fn fetch_i32(&self, pc: u64) -> Result<(i32, u64), Step> {
        let (v, p) = self.fetch32(pc)?;
        Ok((v as i32, p))
    }

    fn fetch64(&self, pc: u64) -> Result<(u64, u64), Step> {
        let (b, p) = self.fetch_n::<8>(pc)?;
        Ok((u64::from_le_bytes(b), p))
    }

    /// Fetch an immediate sized to `width` the way the `0x81`/`0xF7`-family
    /// opcodes do: `imm8` at 8-bit width, `imm16` at 16-bit width, otherwise a
    /// sign-extended `imm32` (there is no `imm64` immediate form in x86-64
    /// except `MOV r64, imm64`).
    fn fetch_imm(&self, pc: u64, width: u32) -> Result<(i64, u64), Step> {
        match width {
            8 => {
                let (v, p) = self.fetch_i8(pc)?;
                Ok((i64::from(v), p))
            }
            16 => {
                let (v, p) = self.fetch16(pc)?;
                Ok((i64::from(v as i16), p))
            }
            _ => {
                let (v, p) = self.fetch_i32(pc)?;
                Ok((i64::from(v), p))
            }
        }
    }

    // ---- ModRM / effective addresses ---------------------------------------

    /// Decode a ModRM byte (and any SIB/displacement that follows it).
    /// Memory addresses that don't need the end-of-instruction address are
    /// resolved immediately; RIP-relative ones are deferred (see [`RmKind`]).
    fn modrm(&self, pc: u64, rex: Rex) -> Result<(ModRm, u64), Step> {
        let (byte, pc) = self.fetch8(pc)?;
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

        let (ea, pc) = if rm_field == 0b100 {
            // SIB byte follows.
            let (sib, pc) = self.fetch8(pc)?;
            let scale = sib >> 6;
            let idx_field = (sib >> 3) & 7;
            let base_field = sib & 7;
            // index field 0b100 (before REX.X extension) means "no index";
            // REX.X turns it into r12, which *is* usable as an index.
            let index_val = if idx_field == 0b100 && !rex.x {
                0
            } else {
                self.gpr[usize::from(idx_field) | (usize::from(rex.x) << 3)] << scale
            };
            let (base_val, disp, pc) = if base_field == 0b101 && md == 0b00 {
                let (d, pc) = self.fetch_i32(pc)?;
                (0, i64::from(d), pc)
            } else {
                let b = self.gpr[usize::from(base_field) | (usize::from(rex.b) << 3)];
                match md {
                    0b01 => {
                        let (d, pc) = self.fetch_i8(pc)?;
                        (b, i64::from(d), pc)
                    }
                    0b10 => {
                        let (d, pc) = self.fetch_i32(pc)?;
                        (b, i64::from(d), pc)
                    }
                    _ => (b, 0, pc),
                }
            };
            (
                base_val.wrapping_add(index_val).wrapping_add(disp as u64),
                pc,
            )
        } else if rm_field == 0b101 && md == 0b00 {
            let (disp, pc) = self.fetch_i32(pc)?;
            return Ok((
                ModRm {
                    reg,
                    kind: RmKind::MemRip(i64::from(disp)),
                },
                pc,
            ));
        } else {
            let base = self.gpr[usize::from(rm_field) | (usize::from(rex.b) << 3)];
            match md {
                0b01 => {
                    let (d, pc) = self.fetch_i8(pc)?;
                    (base.wrapping_add(i64::from(d) as u64), pc)
                }
                0b10 => {
                    let (d, pc) = self.fetch_i32(pc)?;
                    (base.wrapping_add(i64::from(d) as u64), pc)
                }
                _ => (base, pc),
            }
        };
        let ea = if self.addr32 { ea & 0xffff_ffff } else { ea };
        Ok((
            ModRm {
                reg,
                kind: RmKind::Mem(ea),
            },
            pc,
        ))
    }

    /// The effective address (segment offset) of a memory r/m, `None` for a
    /// register r/m. `end` is the address just past the instruction (for
    /// RIP-relative forms; `EIP`-relative, i.e. truncated, under `0x67`).
    fn ea_of(&self, kind: RmKind, end: u64) -> Option<u64> {
        match kind {
            RmKind::Reg(_) => None,
            RmKind::Mem(a) => Some(a),
            RmKind::MemRip(d) => {
                let a = end.wrapping_add(d as u64);
                Some(if self.addr32 { a & 0xffff_ffff } else { a })
            }
        }
    }

    /// Effective address → linear address (adds the `fs:`/`gs:` base).
    fn lin(&self, ea: u64) -> u64 {
        ea.wrapping_add(self.seg_base)
    }

    /// Resolve a (non-8-bit) r/m operand.
    fn op_of(&self, kind: RmKind, end: u64) -> Operand {
        match kind {
            RmKind::Reg(r) => Operand::Reg(r),
            _ => Operand::Mem(self.lin(self.ea_of(kind, end).unwrap_or(0))),
        }
    }

    /// Resolve an 8-bit r/m operand (see [`reg8_operand`]).
    fn op8_of(&self, kind: RmKind, end: u64, has_rex: bool) -> Operand {
        match kind {
            RmKind::Reg(r) => reg8_operand(r, has_rex),
            _ => self.op_of(kind, end),
        }
    }

    /// Resolve an r/m operand of `width` bits.
    fn opw_of(&self, kind: RmKind, end: u64, width: u32, p: Pfx) -> Operand {
        if width == 8 {
            self.op8_of(kind, end, p.has_rex)
        } else {
            self.op_of(kind, end)
        }
    }

    /// The linear address of a memory-only operand; `#UD` for a register
    /// r/m (`LEA`, `CMPXCHG8B`, `LDMXCSR`, …).
    fn mem_only(&self, kind: RmKind, end: u64) -> Result<u64, Step> {
        match kind {
            RmKind::Reg(_) => Err(Step::Illegal),
            _ => Ok(self.lin(self.ea_of(kind, end).unwrap_or(0))),
        }
    }

    // ---- memory and register access -----------------------------------------

    fn read_mem(mem: &GuestMemory, a: u64, width: u32) -> Result<u64, Step> {
        let n = (width / 8) as usize;
        let mut b = [0u8; 8];
        mem.read(a, &mut b[..n]).map_err(|_| rd_fault(a))?;
        Ok(u64::from_le_bytes(b))
    }

    fn write_mem(
        &mut self,
        mem: &mut GuestMemory,
        a: u64,
        val: u64,
        width: u32,
    ) -> Result<(), Step> {
        let n = (width / 8) as usize;
        self.store(mem, a, &val.to_le_bytes()[..n])
    }

    /// Every guest store goes through here: it keeps the decoded-code page
    /// cache ([`X86Interp::code_page`]) coherent with self-modifying code.
    fn store(&mut self, mem: &mut GuestMemory, a: u64, bytes: &[u8]) -> Result<(), Step> {
        let last = a.wrapping_add(bytes.len().max(1) as u64 - 1);
        if a.wrapping_sub(self.code_page) < PAGE || last.wrapping_sub(self.code_page) < PAGE {
            self.code_page = NO_PAGE;
        }
        mem.write_trap(a, bytes).map_err(|e| wr_fault(&e))
    }

    fn read_operand(&self, mem: &GuestMemory, op: Operand, width: u32) -> Result<u64, Step> {
        match op {
            Operand::Reg(r) => Ok(mask_w(self.gpr[r], width)),
            Operand::Reg8Hi(r) => Ok((self.gpr[r] >> 8) & 0xff),
            Operand::Mem(a) => Self::read_mem(mem, a, width),
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
                self.set_reg(r, val, width);
                Ok(())
            }
            Operand::Reg8Hi(r) => {
                self.gpr[r] = (self.gpr[r] & !0xff00u64) | ((val & 0xff) << 8);
                Ok(())
            }
            Operand::Mem(a) => self.write_mem(mem, a, val, width),
        }
    }

    /// Write a GPR with x86 partial-register semantics (see
    /// [`X86Interp::write_operand`]).
    fn set_reg(&mut self, r: usize, val: u64, width: u32) {
        self.gpr[r] = match width {
            8 => (self.gpr[r] & !0xffu64) | (val & 0xff),
            16 => (self.gpr[r] & !0xffffu64) | (val & 0xffff),
            32 => val & 0xffff_ffff,
            _ => val,
        };
    }

    /// Push a `width`-bit value (16 or 64).
    fn push_w(&mut self, mem: &mut GuestMemory, val: u64, width: u32) -> Result<(), Step> {
        let sp = self.gpr[RSP].wrapping_sub(u64::from(width / 8));
        self.write_mem(mem, sp, val, width)?;
        self.gpr[RSP] = sp;
        Ok(())
    }

    /// Pop a `width`-bit value (16 or 64).
    fn pop_w(&mut self, mem: &GuestMemory, width: u32) -> Result<u64, Step> {
        let sp = self.gpr[RSP];
        let v = Self::read_mem(mem, sp, width)?;
        self.gpr[RSP] = sp.wrapping_add(u64::from(width / 8));
        Ok(v)
    }

    fn push(&mut self, mem: &mut GuestMemory, val: u64) -> Result<(), Step> {
        self.push_w(mem, val, 64)
    }

    // ---- flags ----------------------------------------------------------------

    /// `ADD` (and, with `carry_in`, `ADC`): result masked to `width`, all
    /// arithmetic flags computed *at that width* — an 8-bit `0xFF + 1` must
    /// set ZF and CF even though the value fits easily in a host integer.
    fn add_flags(&mut self, a: u64, b: u64, carry_in: bool, width: u32) -> u64 {
        let m = mask_w(u64::MAX, width);
        let (a, b) = (a & m, b & m);
        let full = u128::from(a) + u128::from(b) + u128::from(carry_in);
        let r = (full as u64) & m;
        self.flags = Flags {
            cf: full > u128::from(m),
            pf: parity(r),
            af: (a ^ b ^ r) & 0x10 != 0,
            zf: r == 0,
            sf: sign_bit(r, width),
            of: sign_bit((a ^ r) & (b ^ r), width),
        };
        r
    }

    /// `SUB`/`CMP` (and, with `borrow_in`, `SBB`): width-masked result and
    /// width-accurate flags, like [`X86Interp::add_flags`].
    fn sub_flags(&mut self, a: u64, b: u64, borrow_in: bool, width: u32) -> u64 {
        let m = mask_w(u64::MAX, width);
        let (a, b) = (a & m, b & m);
        let r = a.wrapping_sub(b).wrapping_sub(u64::from(borrow_in)) & m;
        self.flags = Flags {
            cf: u128::from(a) < u128::from(b) + u128::from(borrow_in),
            pf: parity(r),
            af: (a ^ b ^ r) & 0x10 != 0,
            zf: r == 0,
            sf: sign_bit(r, width),
            of: sign_bit((a ^ b) & (a ^ r), width),
        };
        r
    }

    /// Flags for a result whose `CF`/`OF`/`AF` are cleared (the logical ops
    /// `AND`/`OR`/`XOR`/`TEST`): `ZF`/`SF`/`PF` from `r`.
    fn logic_flags(&mut self, r: u64, width: u32) -> u64 {
        let r = mask_w(r, width);
        self.flags = Flags {
            cf: false,
            pf: parity(r),
            af: false,
            zf: r == 0,
            sf: sign_bit(r, width),
            of: false,
        };
        r
    }

    /// Set `ZF`/`SF`/`PF` from a `width`-bit result, leaving the others.
    fn szp(&mut self, r: u64, width: u32) {
        self.flags.zf = mask_w(r, width) == 0;
        self.flags.sf = sign_bit(r, width);
        self.flags.pf = parity(r);
    }

    fn apply_alu(&mut self, op: AluOp, a: u64, b: u64, width: u32) -> u64 {
        match op {
            AluOp::Add => self.add_flags(a, b, false, width),
            AluOp::Adc => self.add_flags(a, b, self.flags.cf, width),
            AluOp::Sub | AluOp::Cmp => self.sub_flags(a, b, false, width),
            AluOp::Sbb => self.sub_flags(a, b, self.flags.cf, width),
            AluOp::And | AluOp::Test => self.logic_flags(a & b, width),
            AluOp::Or => self.logic_flags(a | b, width),
            AluOp::Xor => self.logic_flags(a ^ b, width),
        }
    }

    /// `INC`/`DEC`: like `ADD`/`SUB` by 1, but CF is left untouched (an x86
    /// quirk, since `INC`/`DEC` must not disturb a carry chain).
    fn inc_dec_flags(&mut self, a: u64, dec: bool, width: u32) -> u64 {
        let saved_cf = self.flags.cf;
        let r = if dec {
            self.sub_flags(a, 1, false, width)
        } else {
            self.add_flags(a, 1, false, width)
        };
        self.flags.cf = saved_cf;
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

    /// Pack the flags into an `RFLAGS` word, as `PUSHF`/`SYSCALL` (into
    /// `R11`)/a signal frame see it: the six status flags, `DF`, the
    /// user-writable system bits `NT`/`AC`/`ID` (stored verbatim — see
    /// [`X86Interp::set_rflags_user`]), reserved bit 1 and `IF` (always set from
    /// a user task's view). `TF` and `RF` read back as 0.
    fn rflags_word(&self) -> u64 {
        let f = &self.flags;
        0x202
            | u64::from(f.cf)
            | (u64::from(f.pf) << 2)
            | (u64::from(f.af) << 4)
            | (u64::from(f.zf) << 6)
            | (u64::from(f.sf) << 7)
            | (u64::from(self.df) << 10)
            | (u64::from(f.of) << 11)
            | u64::from(self.rflags_sys)
    }

    /// Load `RFLAGS` the way `POPF` does at CPL 3 with `IOPL == 0`: the
    /// status flags, `DF`, `NT`, `AC` and `ID` take the new value;
    /// `IF`/`IOPL`/`VM`/`VIF`/`VIP`/`RF` are silently left alone (and are
    /// fixed in this model anyway). `TF` (single-step) is accepted but not
    /// modeled — no debug trap is raised.
    fn set_rflags_user(&mut self, v: u64) {
        self.flags = Flags {
            cf: v & (1 << 0) != 0,
            pf: v & (1 << 2) != 0,
            af: v & (1 << 4) != 0,
            zf: v & (1 << 6) != 0,
            sf: v & (1 << 7) != 0,
            of: v & (1 << 11) != 0,
        };
        self.df = v & (1 << 10) != 0;
        self.rflags_sys = (v as u32) & RFLAGS_SYS_MASK;
    }

    // ---- instruction groups ---------------------------------------------------

    /// `op r/m, reg` (`reg_dst == false`: the `00`/`01` forms) or `op reg,
    /// r/m` (`reg_dst == true`: the `02`/`03` forms).
    fn alu_modrm(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        p: Pfx,
        op: AluOp,
        width: u32,
        reg_dst: bool,
    ) -> Step {
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        let rm = self.opw_of(m.kind, end, width, p);
        let rg = if width == 8 {
            reg8_operand(m.reg, p.has_rex)
        } else {
            Operand::Reg(m.reg)
        };
        let (dst, src) = if reg_dst { (rg, rm) } else { (rm, rg) };
        let a = fetch!(self.read_operand(mem, dst, width));
        let b = fetch!(self.read_operand(mem, src, width));
        let r = self.apply_alu(op, a, b, width);
        if op.stores() {
            fetch!(self.write_operand(mem, dst, r, width));
        }
        self.next(end)
    }

    /// `op AL, imm8` / `op eAX, immz` — the accumulator-immediate short forms
    /// each ALU op reserves at `base+4`/`base+5` (plus `A8`/`A9` for TEST).
    fn alu_acc_imm(&mut self, pc: u64, width: u32, op: AluOp) -> Step {
        let (imm, end) = fetch!(self.fetch_imm(pc, width));
        let a = mask_w(self.gpr[RAX], width);
        let r = self.apply_alu(op, a, mask_w(imm as u64, width), width);
        if op.stores() {
            self.set_reg(RAX, r, width);
        }
        self.next(end)
    }

    /// Group 1: `0x80 /r ib` (8-bit), `0x81 /r iz`, `0x83 /r ib`
    /// (sign-extended) — ALU op, r/m and an immediate.
    fn group1(&mut self, mem: &mut GuestMemory, pc: u64, p: Pfx, width: u32, imm8: bool) -> Step {
        let (m, pc2) = fetch!(self.modrm(pc, p.rex));
        let (imm, end) = fetch!(self.fetch_imm(pc2, if imm8 { 8 } else { width }));
        let op = AluOp::from_ext(m.ext());
        let rm = self.opw_of(m.kind, end, width, p);
        let a = fetch!(self.read_operand(mem, rm, width));
        let r = self.apply_alu(op, a, mask_w(imm as u64, width), width);
        if op.stores() {
            fetch!(self.write_operand(mem, rm, r, width));
        }
        self.next(end)
    }

    /// `XCHG r/m, reg` (`0x86`/`0x87`) — swap the two operands' contents.
    fn xchg(&mut self, mem: &mut GuestMemory, pc: u64, p: Pfx, width: u32) -> Step {
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        let rm = self.opw_of(m.kind, end, width, p);
        let rg = if width == 8 {
            reg8_operand(m.reg, p.has_rex)
        } else {
            Operand::Reg(m.reg)
        };
        let a = fetch!(self.read_operand(mem, rm, width));
        let b = fetch!(self.read_operand(mem, rg, width));
        fetch!(self.write_operand(mem, rm, b, width));
        fetch!(self.write_operand(mem, rg, a, width));
        self.next(end)
    }

    /// `MOV r/m, reg` / `MOV reg, r/m` (`88`-`8B`).
    fn mov_modrm(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        p: Pfx,
        width: u32,
        to_reg: bool,
    ) -> Step {
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        let rm = self.opw_of(m.kind, end, width, p);
        let rg = if width == 8 {
            reg8_operand(m.reg, p.has_rex)
        } else {
            Operand::Reg(m.reg)
        };
        let (dst, src) = if to_reg { (rg, rm) } else { (rm, rg) };
        let v = fetch!(self.read_operand(mem, src, width));
        fetch!(self.write_operand(mem, dst, v, width));
        self.next(end)
    }

    /// `MOV r/m, imm` (`C6 /0 ib`, `C7 /0 iz`). The immediate follows the
    /// operand size: `imm16` under `0x66`, else `imm32` (sign-extended for a
    /// 64-bit store). `/1../7` are `#UD` (`C6 F8`/`C7 F8` are the RTM
    /// `XABORT`/`XBEGIN`, not advertised).
    fn mov_imm(&mut self, mem: &mut GuestMemory, pc: u64, p: Pfx, width: u32) -> Step {
        let (m, pc2) = fetch!(self.modrm(pc, p.rex));
        if m.ext() != 0 {
            return Step::Illegal;
        }
        let (imm, end) = fetch!(self.fetch_imm(pc2, width));
        let rm = self.opw_of(m.kind, end, width, p);
        fetch!(self.write_operand(mem, rm, imm as u64, width));
        self.next(end)
    }

    /// Group 3: `0xF6`/`0xF7 /r` — `TEST r/m, imm` (/0, /1), `NOT` (/2),
    /// `NEG` (/3), `MUL` (/4), `IMUL` (/5, one-operand), `DIV` (/6), `IDIV`
    /// (/7).
    fn group3(&mut self, mem: &mut GuestMemory, pc: u64, p: Pfx, width: u32) -> Step {
        let (m, pc2) = fetch!(self.modrm(pc, p.rex));
        if m.ext() < 2 {
            let (imm, end) = fetch!(self.fetch_imm(pc2, width));
            let rm = self.opw_of(m.kind, end, width, p);
            let a = fetch!(self.read_operand(mem, rm, width));
            self.logic_flags(a & imm as u64, width);
            return self.next(end);
        }
        let rm = self.opw_of(m.kind, pc2, width, p);
        let a = fetch!(self.read_operand(mem, rm, width));
        match m.ext() {
            2 => fetch!(self.write_operand(mem, rm, !a, width)),
            3 => {
                let r = self.sub_flags(0, a, false, width); // NEG = 0 - a; CF = (a != 0)
                fetch!(self.write_operand(mem, rm, r, width));
            }
            4 => self.mul1(a, width, false),
            5 => self.mul1(a, width, true),
            e => {
                if let Err(s) = self.div1(a, width, e == 7) {
                    return s;
                }
            }
        }
        self.next(pc2)
    }

    /// `MUL`/`IMUL` one-operand form: `rDX:rAX` (or `AX` at 8-bit width) =
    /// `rAX` * `src`. `CF`/`OF` flag a result that doesn't fit the low half;
    /// the architecturally undefined `SF`/`ZF`/`AF`/`PF` follow hardware
    /// (see [`X86Interp::mul_flags`]).
    fn mul1(&mut self, src: u64, width: u32, signed: bool) {
        let a = mask_w(self.gpr[RAX], width);
        let (p, cf) = if signed {
            let p = sign_extend_128(u128::from(a), width) * sign_extend_128(u128::from(src), width);
            (p as u128, !fits_signed(p, width))
        } else {
            let p = u128::from(a) * u128::from(src);
            (p, (p >> width) != 0)
        };
        let lo = mask_w(p as u64, width);
        let hi = mask_w((p >> width) as u64, width);
        if width == 8 {
            self.set_reg(RAX, (hi << 8) | lo, 16);
        } else {
            self.set_reg(RAX, lo, width);
            self.set_reg(RDX, hi, width);
        }
        self.mul_flags(cf, lo, width);
    }

    /// Flags after any `MUL`/`IMUL`: `CF = OF = overflow`; `SF`/`PF` from the
    /// low half of the product, `ZF`/`AF` cleared.
    fn mul_flags(&mut self, cf: bool, lo: u64, width: u32) {
        self.flags = Flags {
            cf,
            pf: parity(lo),
            af: false,
            zf: false,
            sf: sign_bit(lo, width),
            of: cf,
        };
    }

    /// `DIV`/`IDIV`: the `2*width`-bit dividend in `rDX:rAX` (or `AX` at
    /// 8-bit width) is divided by `src`, leaving the quotient in `rAX`/`AL`
    /// and the remainder in `rDX`/`AH`. A zero divisor or an out-of-range
    /// quotient raises `#DE`. The flags (all architecturally undefined) are
    /// left unchanged.
    fn div1(&mut self, src: u64, width: u32, signed: bool) -> Result<(), Step> {
        if src == 0 {
            return Err(Step::Trap(Trap::Divide));
        }
        let dividend: u128 = if width == 8 {
            u128::from(self.gpr[RAX] & 0xffff)
        } else {
            (u128::from(mask_w(self.gpr[RDX], width)) << width)
                | u128::from(mask_w(self.gpr[RAX], width))
        };
        let (q, r) = if signed {
            let n = sign_extend_128(dividend, width * 2);
            let d = sign_extend_128(u128::from(src), width);
            let q = n.wrapping_div(d);
            if !fits_signed(q, width) {
                return Err(Step::Trap(Trap::Divide));
            }
            (q as u64, n.wrapping_rem(d) as u64)
        } else {
            let d = u128::from(src);
            let q = dividend / d;
            if q >> width != 0 {
                return Err(Step::Trap(Trap::Divide));
            }
            (q as u64, (dividend % d) as u64)
        };
        if width == 8 {
            self.set_reg(RAX, (mask_w(r, 8) << 8) | mask_w(q, 8), 16);
        } else {
            self.set_reg(RAX, q, width);
            self.set_reg(RDX, r, width);
        }
        Ok(())
    }

    /// `IMUL r, r/m, imm` (`69` with `imm_w` = the operand size, `6B` with
    /// `imm_w == 8`) and `IMUL r, r/m` (`0F AF`, `imm_w == 0`).
    fn imul_rm(&mut self, mem: &GuestMemory, pc: u64, p: Pfx, imm_w: u32) -> Step {
        let width = p.width();
        let (m, pc2) = fetch!(self.modrm(pc, p.rex));
        let (b, end) = if imm_w == 0 {
            (self.gpr[m.reg], pc2)
        } else {
            let (v, e) = fetch!(self.fetch_imm(pc2, imm_w));
            (v as u64, e)
        };
        let src = self.op_of(m.kind, end);
        let a = fetch!(self.read_operand(mem, src, width));
        let prod = sign_extend_128(u128::from(a), width)
            * sign_extend_128(u128::from(mask_w(b, width)), width);
        let lo = mask_w(prod as u64, width);
        self.set_reg(m.reg, lo, width);
        self.mul_flags(!fits_signed(prod, width), lo, width);
        self.next(end)
    }

    /// Group 4: `0xFE /r` — `INC r/m8` (/0) and `DEC r/m8` (/1).
    fn group4(&mut self, mem: &mut GuestMemory, pc: u64, p: Pfx) -> Step {
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        if m.ext() > 1 {
            return Step::Illegal;
        }
        let rm = self.op8_of(m.kind, end, p.has_rex);
        let a = fetch!(self.read_operand(mem, rm, 8));
        let r = self.inc_dec_flags(a, m.ext() == 1, 8);
        fetch!(self.write_operand(mem, rm, r, 8));
        self.next(end)
    }

    /// Group 5: `0xFF /r` — `INC`/`DEC r/m` (/0, /1), `CALL`/`JMP r/m` (/2,
    /// /4, near indirect, always a 64-bit target in long mode), `CALL`/`JMP
    /// m16:64` (/3, /5, far — `#GP` here: no far code segments for a user
    /// task) and `PUSH r/m` (/6).
    fn group5(&mut self, mem: &mut GuestMemory, pc: u64, p: Pfx) -> Step {
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        let width = p.width();
        match m.ext() {
            0 | 1 => {
                let rm = self.op_of(m.kind, end);
                let a = fetch!(self.read_operand(mem, rm, width));
                let r = self.inc_dec_flags(a, m.ext() == 1, width);
                fetch!(self.write_operand(mem, rm, r, width));
                self.next(end)
            }
            2 | 4 => {
                let rm = self.op_of(m.kind, end);
                let target = fetch!(self.read_operand(mem, rm, 64));
                if m.ext() == 2 {
                    fetch!(self.push(mem, end));
                }
                self.jump(target)
            }
            3 | 5 => match m.kind {
                RmKind::Reg(_) => Step::Illegal,
                _ => Step::Trap(Trap::Protection),
            },
            6 => {
                let w = p.stack_width();
                let rm = self.op_of(m.kind, end);
                let v = fetch!(self.read_operand(mem, rm, w));
                fetch!(self.push_w(mem, v, w));
                self.next(end)
            }
            _ => Step::Illegal,
        }
    }

    /// Group 2 shifts and rotates: `C0`/`C1 /r ib` (`count == None`: fetch
    /// the immediate), `D0`/`D1 /r` (by 1), `D2`/`D3 /r` (by `CL`).
    fn group2(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        p: Pfx,
        width: u32,
        count: Option<u8>,
    ) -> Step {
        let (m, pc2) = fetch!(self.modrm(pc, p.rex));
        let (cnt, end) = match count {
            Some(c) => (c, pc2),
            None => fetch!(self.fetch8(pc2)),
        };
        let rm = self.opw_of(m.kind, end, width, p);
        let a = fetch!(self.read_operand(mem, rm, width));
        let masked = u32::from(cnt) & if width == 64 { 63 } else { 31 };
        if masked == 0 {
            // Flags untouched, but the destination is still written: a
            // 32-bit register is zero-extended even by a zero count.
            fetch!(self.write_operand(mem, rm, a, width));
            return self.next(end);
        }
        let r = self.shift_rotate(m.ext(), a, masked, width);
        fetch!(self.write_operand(mem, rm, r, width));
        self.next(end)
    }

    /// One shift/rotate of the `width`-bit value `a` by the already-masked,
    /// nonzero `cnt` (`ext` is the group-2 `/r`: ROL, ROR, RCL, RCR, SHL, SHR,
    /// SAL (= SHL), SAR), setting the flags it defines. Where the manual leaves
    /// `OF` (multi-bit counts) or `AF` undefined this follows hardware.
    fn shift_rotate(&mut self, ext: usize, a: u64, cnt: u32, width: u32) -> u64 {
        let a = mask_w(a, width);
        let msb = |v: u64| sign_bit(v, width);
        match ext {
            0 => {
                // ROL
                let k = cnt % width;
                let r = if k == 0 {
                    a
                } else {
                    mask_w((a << k) | (a >> (width - k)), width)
                };
                self.flags.cf = r & 1 != 0;
                self.flags.of = msb(r) ^ self.flags.cf;
                r
            }
            1 => {
                // ROR
                let k = cnt % width;
                let r = if k == 0 {
                    a
                } else {
                    mask_w((a >> k) | (a << (width - k)), width)
                };
                self.flags.cf = msb(r);
                self.flags.of = msb(r) ^ sign_bit(r << 1, width);
                r
            }
            2 | 3 => {
                // RCL / RCR: rotate the (width+1)-bit value CF:a.
                let n = width + 1;
                let k = match width {
                    8 => cnt % 9,
                    16 => cnt % 17,
                    _ => cnt,
                };
                if k == 0 {
                    // A whole-ring rotate: value and CF unchanged; OF is
                    // recomputed as for a 1-bit rotate.
                    self.flags.of = if ext == 2 {
                        msb(a) ^ self.flags.cf
                    } else {
                        msb(a) ^ sign_bit(a << 1, width)
                    };
                    return a;
                }
                let v = (u128::from(self.flags.cf) << width) | u128::from(a);
                let ring = (1u128 << n) - 1;
                let rot = if ext == 2 {
                    ((v << k) | (v >> (n - k))) & ring
                } else {
                    ((v >> k) | (v << (n - k))) & ring
                };
                let r = mask_w(rot as u64, width);
                self.flags.cf = (rot >> width) & 1 != 0;
                self.flags.of = if ext == 2 {
                    msb(r) ^ self.flags.cf
                } else {
                    msb(r) ^ sign_bit(r << 1, width)
                };
                r
            }
            4 | 6 => {
                // SHL/SAL
                let wide = u128::from(a) << cnt;
                let r = mask_w(wide as u64, width);
                self.flags.cf = (wide >> width) & 1 != 0;
                self.flags.of = msb(r) ^ self.flags.cf;
                self.flags.af = false;
                self.szp(r, width);
                r
            }
            5 => {
                // SHR
                let r = if cnt >= 64 { 0 } else { a >> cnt };
                self.flags.cf = cnt <= 64 && (a >> (cnt - 1)) & 1 != 0;
                self.flags.of = msb(a);
                self.flags.af = false;
                self.szp(r, width);
                r
            }
            _ => {
                // SAR
                let s = sign_extend_w(a, width);
                let r = mask_w((s >> cnt.min(63)) as u64, width);
                self.flags.cf = (s >> (cnt - 1).min(63)) & 1 != 0;
                self.flags.of = false;
                self.flags.af = false;
                self.szp(r, width);
                r
            }
        }
    }

    /// `SHLD`/`SHRD Ev, Gv, ib|CL` (`0F A4/A5`, `0F AC/AD`): a double-
    /// precision shift where the vacated bits of the destination come from
    /// `src`. A masked count of 0 changes nothing.
    fn shld_shrd(
        &mut self,
        mem: &mut GuestMemory,
        pc: u64,
        p: Pfx,
        left: bool,
        by_cl: bool,
    ) -> Step {
        let width = p.width();
        let (m, pc2) = fetch!(self.modrm(pc, p.rex));
        let (count, end) = if by_cl {
            (self.gpr[RCX] as u8, pc2)
        } else {
            fetch!(self.fetch8(pc2))
        };
        let cnt = u32::from(count) & if width == 64 { 63 } else { 31 };
        let rm = self.op_of(m.kind, end);
        let d = fetch!(self.read_operand(mem, rm, width));
        if cnt == 0 {
            fetch!(self.write_operand(mem, rm, d, width));
            return self.next(end);
        }
        let s = mask_w(self.gpr[m.reg], width);
        // Concatenate into a 128-bit (for 16-bit operands, a repeating
        // dest:src:dest 48-bit) pattern so counts past a 16-bit width behave
        // as on hardware.
        let (r, cf) = if left {
            let (wide, total): (u128, u32) = if width == 16 {
                (
                    (u128::from(d) << 32) | (u128::from(s) << 16) | u128::from(d),
                    48,
                )
            } else {
                ((u128::from(d) << width) | u128::from(s), 2 * width)
            };
            let r = mask_w((wide >> (total - width - cnt)) as u64, width);
            (r, (wide >> (total - cnt)) & 1 != 0)
        } else {
            let wide: u128 = if width == 16 {
                (u128::from(d) << 32) | (u128::from(s) << 16) | u128::from(d)
            } else {
                (u128::from(s) << width) | u128::from(d)
            };
            let r = mask_w((wide >> cnt) as u64, width);
            (r, (wide >> (cnt - 1)) & 1 != 0)
        };
        self.flags.cf = cf;
        self.flags.of = sign_bit(r ^ d, width);
        self.flags.af = false;
        self.szp(r, width);
        fetch!(self.write_operand(mem, rm, r, width));
        self.next(end)
    }

    /// `BT`/`BTS`/`BTR`/`BTC`. With a register bit offset and a memory
    /// operand the offset is a signed index into a bit string starting at the
    /// effective address (so it can reach far outside the addressed word);
    /// with an immediate offset, or a register operand, it is taken modulo the
    /// operand width.
    #[allow(clippy::too_many_arguments)]
    fn bit_test(
        &mut self,
        mem: &mut GuestMemory,
        m: ModRm,
        end: u64,
        width: u32,
        offset: u64,
        from_reg: bool,
        op: BitTestOp,
    ) -> Step {
        let target = match m.kind {
            RmKind::Reg(r) => Operand::Reg(r),
            _ => {
                let ea = self.ea_of(m.kind, end).unwrap_or(0);
                let ea = if from_reg {
                    let off = sign_extend_w(offset, width);
                    let words = off >> width.trailing_zeros();
                    let a = ea.wrapping_add(words.wrapping_mul(i64::from(width / 8)) as u64);
                    if self.addr32 { a & 0xffff_ffff } else { a }
                } else {
                    ea
                };
                Operand::Mem(self.lin(ea))
            }
        };
        let bit = (offset & u64::from(width - 1)) as u32;
        let a = fetch!(self.read_operand(mem, target, width));
        self.flags.cf = (a >> bit) & 1 != 0;
        let mask = 1u64 << bit;
        let r = match op {
            BitTestOp::Bt => return self.next(end),
            BitTestOp::Bts => a | mask,
            BitTestOp::Btr => a & !mask,
            BitTestOp::Btc => a ^ mask,
        };
        fetch!(self.write_operand(mem, target, r, width));
        self.next(end)
    }

    /// `BSF`/`BSR` (`0F BC`/`BD`): the index of the lowest/highest set bit. A
    /// zero source sets `ZF` and leaves the destination unchanged.
    fn bit_scan(&mut self, mem: &GuestMemory, pc: u64, p: Pfx, reverse: bool) -> Step {
        let width = p.width();
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        let src = fetch!(self.read_operand(mem, self.op_of(m.kind, end), width));
        self.flags.zf = src == 0;
        if src != 0 {
            let idx = if reverse {
                src.ilog2()
            } else {
                src.trailing_zeros()
            };
            self.set_reg(m.reg, u64::from(idx), width);
        }
        self.next(end)
    }

    /// `POPCNT Gv, Ev` (`F3 0F B8`): `ZF` = (source == 0), other flags cleared.
    fn popcnt(&mut self, mem: &GuestMemory, pc: u64, p: Pfx) -> Step {
        let width = p.width();
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        let src = fetch!(self.read_operand(mem, self.op_of(m.kind, end), width));
        self.set_reg(m.reg, u64::from(src.count_ones()), width);
        self.flags = Flags {
            zf: src == 0,
            ..Flags::default()
        };
        self.next(end)
    }

    /// `XADD Eb,Gb` / `Ev,Gv` (`0F C0`/`C1`): the register gets the old
    /// destination; the destination becomes the sum (flags as `ADD`).
    fn xadd(&mut self, mem: &mut GuestMemory, pc: u64, p: Pfx, width: u32) -> Step {
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        let rm = self.opw_of(m.kind, end, width, p);
        let rg = if width == 8 {
            reg8_operand(m.reg, p.has_rex)
        } else {
            Operand::Reg(m.reg)
        };
        let d = fetch!(self.read_operand(mem, rm, width));
        let s = fetch!(self.read_operand(mem, rg, width));
        let sum = self.add_flags(d, s, false, width);
        fetch!(self.write_operand(mem, rg, d, width));
        fetch!(self.write_operand(mem, rm, sum, width));
        self.next(end)
    }

    /// `CMPXCHG Eb,Gb` / `Ev,Gv` (`0F B0`/`B1`): compare the accumulator with
    /// the destination (flags as `CMP`); equal → destination = source, else
    /// accumulator = destination.
    fn cmpxchg(&mut self, mem: &mut GuestMemory, pc: u64, p: Pfx, width: u32) -> Step {
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        let rm = self.opw_of(m.kind, end, width, p);
        let rg = if width == 8 {
            reg8_operand(m.reg, p.has_rex)
        } else {
            Operand::Reg(m.reg)
        };
        let d = fetch!(self.read_operand(mem, rm, width));
        let acc = mask_w(self.gpr[RAX], width);
        self.sub_flags(acc, d, false, width);
        if acc == d {
            let s = fetch!(self.read_operand(mem, rg, width));
            fetch!(self.write_operand(mem, rm, s, width));
        } else {
            // The destination is written back unchanged: a memory operand
            // must be writable even on a mismatch, and a 32-bit register
            // destination is zero-extended like any 32-bit write.
            fetch!(self.write_operand(mem, rm, d, width));
            self.set_reg(RAX, d, width);
        }
        self.next(end)
    }

    /// Group 9 (`0F C7`): `CMPXCHG8B`/`CMPXCHG16B m64/m128` (`/1`; `REX.W`
    /// selects the 16-byte form, which `#GP`s on a misaligned operand) and
    /// `RDRAND r` (`/6`, register form). `RDSEED` (`/7`) isn't advertised.
    fn group9(&mut self, mem: &mut GuestMemory, pc: u64, p: Pfx) -> Step {
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        match (m.ext(), m.kind) {
            (1, RmKind::Mem(_) | RmKind::MemRip(_)) => {
                let addr = fetch!(self.mem_only(m.kind, end));
                if p.rex.w {
                    if addr & 15 != 0 {
                        return Step::Trap(Trap::Protection);
                    }
                    let mut b = [0u8; 16];
                    fetch!(mem.read(addr, &mut b).map_err(|_| rd_fault(addr)));
                    let cur = u128::from_le_bytes(b);
                    let expect = (u128::from(self.gpr[RDX]) << 64) | u128::from(self.gpr[RAX]);
                    let eq = cur == expect;
                    let new = if eq {
                        (u128::from(self.gpr[RCX]) << 64) | u128::from(self.gpr[RBX])
                    } else {
                        cur
                    };
                    fetch!(self.store(mem, addr, &new.to_le_bytes()));
                    if !eq {
                        self.gpr[RAX] = cur as u64;
                        self.gpr[RDX] = (cur >> 64) as u64;
                    }
                    self.flags.zf = eq;
                } else {
                    let cur = fetch!(Self::read_mem(mem, addr, 64));
                    let expect = (mask_w(self.gpr[RDX], 32) << 32) | mask_w(self.gpr[RAX], 32);
                    let eq = cur == expect;
                    let new = if eq {
                        (mask_w(self.gpr[RCX], 32) << 32) | mask_w(self.gpr[RBX], 32)
                    } else {
                        cur
                    };
                    fetch!(self.write_mem(mem, addr, new, 64));
                    if !eq {
                        self.gpr[RAX] = mask_w(cur, 32);
                        self.gpr[RDX] = cur >> 32;
                    }
                    self.flags.zf = eq;
                }
                self.next(end)
            }
            (6, RmKind::Reg(r)) if p.rep == 0 => {
                let v = self.rdrand();
                self.set_reg(r, v, p.width());
                self.flags = Flags {
                    cf: true,
                    ..Flags::default()
                };
                self.next(end)
            }
            _ => Step::Illegal,
        }
    }

    /// The `RDRAND` value source: a splitmix64 step over a private state —
    /// deterministic, never "not ready" (`CF` is always set).
    fn rdrand(&mut self) -> u64 {
        self.prng = self.prng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.prng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// `ENTER imm16, imm8`: push `rBP`, copy `level - 1` outer frame pointers,
    /// push the new frame pointer, set `rBP` and reserve `imm16` bytes.
    fn enter(&mut self, mem: &mut GuestMemory, pc: u64, p: Pfx) -> Step {
        let (size, pc2) = fetch!(self.fetch16(pc));
        let (level, end) = fetch!(self.fetch8(pc2));
        let level = level & 31;
        let w = p.stack_width();
        let step = u64::from(w / 8);
        let rbp = self.gpr[RBP];
        // Work on a local stack pointer so a fault leaves the architectural
        // state untouched.
        let mut sp = self.gpr[RSP].wrapping_sub(step);
        fetch!(self.write_mem(mem, sp, rbp, w));
        let frame = sp;
        if level > 0 {
            let mut bp = rbp;
            for _ in 1..level {
                bp = bp.wrapping_sub(step);
                let addr = if w == 16 { bp & 0xffff } else { bp };
                let v = fetch!(Self::read_mem(mem, addr, w));
                sp = sp.wrapping_sub(step);
                fetch!(self.write_mem(mem, sp, v, w));
            }
            sp = sp.wrapping_sub(step);
            fetch!(self.write_mem(mem, sp, frame, w));
        }
        self.set_reg(RBP, frame, w);
        self.gpr[RSP] = sp.wrapping_sub(u64::from(size));
        self.next(end)
    }

    /// `LEAVE`: `rSP = rBP`, then pop `rBP` (16-bit under `0x66`).
    fn leave(&mut self, mem: &GuestMemory, pc: u64, p: Pfx) -> Step {
        let w = p.stack_width();
        let sp = self.gpr[RBP];
        let v = fetch!(Self::read_mem(mem, sp, w));
        self.gpr[RSP] = sp.wrapping_add(u64::from(w / 8));
        self.set_reg(RBP, v, w);
        self.next(pc)
    }

    /// `CPUID` (`0F A2`): dispatch on the leaf in `EAX` (and the subleaf in
    /// `ECX`) and write `EAX`/`EBX`/`ECX`/`EDX`. Feature bits are set *only*
    /// for what this interpreter executes, so glibc/musl's CPUID-gated dispatch
    /// never picks an unimplemented instruction path; they mirror the loader's
    /// `AT_HWCAP` (leaf 1 `EDX`). Unrecognized leaves return zeros.
    fn cpuid(&mut self) {
        let leaf = self.gpr[RAX] as u32;
        let sub = self.gpr[RCX] as u32;
        let (eax, ebx, ecx, edx): (u32, u32, u32, u32) = match leaf {
            // Max standard leaf + the "GenuineIntel" vendor string, split
            // EBX/EDX/ECX = "Genu"/"ineI"/"ntel".
            0 => (7, 0x756E_6547, 0x6C65_746E, 0x4965_6E69),
            // EAX = family 6 signature; EBX = 1 logical processor, 64-byte
            // CLFLUSH line; ECX/EDX = the feature words.
            1 => (0x0007_06A1, 0x0001_0800, CPUID1_ECX, CPUID1_EDX),
            // Deterministic cache parameters: a plausible hierarchy so libc
            // cache-size probes (memcpy non-temporal thresholds) see sane
            // values.
            4 => cache_leaf(sub),
            // Structured extended features (leaf 7): none.
            0x8000_0000 => (0x8000_0008, 0, 0, 0),
            // LAHF/SAHF in 64-bit mode (ECX bit 0); SYSCALL (EDX 11), NX (20),
            // RDTSCP (27), LM (29).
            0x8000_0001 => (0, 0, 0x1, 0x2810_0800),
            0x8000_0002..=0x8000_0004 => Self::cpuid_brand_leaf(leaf),
            // 48-bit virtual / 46-bit physical address sizes.
            0x8000_0008 => (0x0000_302e, 0, 0, 0),
            _ => (0, 0, 0, 0),
        };
        self.gpr[RAX] = u64::from(eax);
        self.gpr[RBX] = u64::from(ebx);
        self.gpr[RCX] = u64::from(ecx);
        self.gpr[RDX] = u64::from(edx);
    }

    /// The `EAX`/`EBX`/`ECX`/`EDX` quartet for one of `CPUID`'s three
    /// "processor brand string" leaves (`0x8000_0002..=0x8000_0004`).
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
    /// incrementing on every read (rather than tracking real elapsed time)
    /// guarantees a guest spin-loop that polls it for elapsed time terminates.
    fn rdtsc_tick(&mut self) -> u64 {
        self.tsc = self.tsc.wrapping_add(1);
        self.tsc
    }

    // ---- string instructions --------------------------------------------------
    //
    // `rep` is `0` (no prefix: run once, leave rCX alone), `1` (`REP`/`REPE`,
    // `0xF3`) or `2` (`REPNE`, `0xF2`; on `MOVS`/`STOS`/`LODS` it acts as a
    // plain `REP`). The address-size prefix selects `ECX`/`ESI`/`EDI` (with
    // 32-bit wraparound); a segment override applies to the `rSI` source. A
    // whole repeat runs in one `Step`, but the registers are updated per
    // element, so a fault mid-string leaves the precise resume state.

    /// The string-op pointer/count register value (`rSI`/`rDI`/`rCX`, or the
    /// 32-bit form under `0x67`).
    fn sreg(&self, r: usize) -> u64 {
        if self.addr32 {
            self.gpr[r] & 0xffff_ffff
        } else {
            self.gpr[r]
        }
    }

    fn set_sreg(&mut self, r: usize, v: u64) {
        if self.addr32 {
            self.gpr[r] = v & 0xffff_ffff;
        } else {
            self.gpr[r] = v;
        }
    }

    /// Step a string-op pointer register by one element of `bytes`.
    fn advance(&mut self, r: usize, bytes: u64) {
        let v = self.sreg(r);
        let n = if self.df {
            v.wrapping_sub(bytes)
        } else {
            v.wrapping_add(bytes)
        };
        self.set_sreg(r, n);
    }

    /// Run a string op: `body` performs one element; the repeat prefix and
    /// `rCX` drive the loop (`compares` makes `REPE`/`REPNE` also test `ZF`).
    fn string_op(
        &mut self,
        mem: &mut GuestMemory,
        end: u64,
        rep: u8,
        compares: bool,
        mut body: impl FnMut(&mut Self, &mut GuestMemory) -> Result<(), Step>,
    ) -> Step {
        if rep == 0 {
            fetch!(body(self, mem));
            return self.next(end);
        }
        loop {
            if self.sreg(RCX) == 0 {
                return self.next(end);
            }
            fetch!(body(self, mem));
            let c = self.sreg(RCX).wrapping_sub(1);
            self.set_sreg(RCX, c);
            if compares {
                let go_on = if rep == 1 {
                    self.flags.zf
                } else {
                    !self.flags.zf
                };
                if !go_on {
                    return self.next(end);
                }
            }
        }
    }

    /// `MOVS`/`CMPS`/`STOS`/`LODS`/`SCAS` (`A4`-`A7`, `AA`-`AF`).
    fn string_insn(&mut self, mem: &mut GuestMemory, end: u64, p: Pfx, op: u8) -> Step {
        let width = if op & 1 == 0 { 8 } else { p.width() };
        let n = u64::from(width / 8);
        let src_seg = self.seg_base;
        match op {
            0xA4 | 0xA5 => {
                // MOVS: [rDI] = [seg:rSI]
                if p.rep != 0
                    && !self.df
                    && width == 8
                    && let Some(s) = self.rep_movsb_fast(mem, end, src_seg)
                {
                    return s;
                }
                self.string_op(mem, end, p.rep, false, |c, mem| {
                    let s = src_seg.wrapping_add(c.sreg(RSI));
                    let v = Self::read_mem(mem, s, width)?;
                    c.write_mem(mem, c.sreg(RDI), v, width)?;
                    c.advance(RSI, n);
                    c.advance(RDI, n);
                    Ok(())
                })
            }
            0xAA | 0xAB => {
                // STOS: [rDI] = rAX
                if p.rep != 0
                    && !self.df
                    && width == 8
                    && let Some(s) = self.rep_stosb_fast(mem, end)
                {
                    return s;
                }
                let v = mask_w(self.gpr[RAX], width);
                self.string_op(mem, end, p.rep, false, |c, mem| {
                    c.write_mem(mem, c.sreg(RDI), v, width)?;
                    c.advance(RDI, n);
                    Ok(())
                })
            }
            0xAC | 0xAD => self.string_op(mem, end, p.rep, false, |c, mem| {
                // LODS: rAX = [seg:rSI]
                let v = Self::read_mem(mem, src_seg.wrapping_add(c.sreg(RSI)), width)?;
                c.set_reg(RAX, v, width);
                c.advance(RSI, n);
                Ok(())
            }),
            0xAE | 0xAF => self.string_op(mem, end, p.rep, true, |c, mem| {
                // SCAS: compare rAX with [rDI]
                let v = Self::read_mem(mem, c.sreg(RDI), width)?;
                c.sub_flags(c.gpr[RAX], v, false, width);
                c.advance(RDI, n);
                Ok(())
            }),
            _ => self.string_op(mem, end, p.rep, true, |c, mem| {
                // CMPS (A6/A7): compare [seg:rSI] with [rDI]
                let a = Self::read_mem(mem, src_seg.wrapping_add(c.sreg(RSI)), width)?;
                let b = Self::read_mem(mem, c.sreg(RDI), width)?;
                c.sub_flags(a, b, false, width);
                c.advance(RSI, n);
                c.advance(RDI, n);
                Ok(())
            }),
        }
    }

    /// Bulk `REP MOVSB` (forward): copy page-bounded chunks instead of one
    /// byte per iteration. Returns `None` to fall back to the element loop —
    /// when the destination overlaps the source chunk ahead of it (the classic
    /// overlapping forward copy, which must re-read written bytes) or a chunk
    /// faults (so the per-element loop stops at the precise element).
    fn rep_movsb_fast(&mut self, mem: &mut GuestMemory, end: u64, src_seg: u64) -> Option<Step> {
        if self.addr32 {
            return None;
        }
        let mut buf = [0u8; PAGE as usize];
        while self.gpr[RCX] != 0 {
            let s = src_seg.wrapping_add(self.gpr[RSI]);
            let d = self.gpr[RDI];
            let n = self.gpr[RCX]
                .min(PAGE - (s & (PAGE - 1)))
                .min(PAGE - (d & (PAGE - 1)));
            if d > s && d - s < n {
                return None;
            }
            let k = n as usize;
            if mem.read(s, &mut buf[..k]).is_err() || self.store(mem, d, &buf[..k]).is_err() {
                return None;
            }
            self.gpr[RSI] = self.gpr[RSI].wrapping_add(n);
            self.gpr[RDI] = d.wrapping_add(n);
            self.gpr[RCX] -= n;
        }
        Some(self.next(end))
    }

    /// Bulk `REP STOSB` (forward), page-bounded like [`Self::rep_movsb_fast`].
    fn rep_stosb_fast(&mut self, mem: &mut GuestMemory, end: u64) -> Option<Step> {
        if self.addr32 {
            return None;
        }
        let buf = [self.gpr[RAX] as u8; PAGE as usize];
        while self.gpr[RCX] != 0 {
            let d = self.gpr[RDI];
            let n = self.gpr[RCX].min(PAGE - (d & (PAGE - 1)));
            if self.store(mem, d, &buf[..n as usize]).is_err() {
                return None;
            }
            self.gpr[RDI] = d.wrapping_add(n);
            self.gpr[RCX] -= n;
        }
        Some(self.next(end))
    }

    /// Whether `LOCK` may prefix the instruction whose ModRM byte is at `pc`
    /// (opcode `op`, `two` for the `0F` map): only the read-modify-write forms
    /// with a memory destination are lockable; anything else is `#UD`.
    fn lock_ok(&self, op: u8, two: bool, pc: u64) -> bool {
        let Ok((modrm, _)) = self.fetch8(pc) else {
            return false;
        };
        let mem_dst = modrm >> 6 != 3;
        let ext = (modrm >> 3) & 7;
        mem_dst
            && if two {
                match op {
                    0xAB | 0xB3 | 0xBB | 0xB0 | 0xB1 | 0xC0 | 0xC1 => true,
                    0xBA => ext >= 5,
                    0xC7 => ext == 1,
                    _ => false,
                }
            } else {
                match op {
                    0x00 | 0x01 | 0x08 | 0x09 | 0x10 | 0x11 | 0x18 | 0x19 | 0x20 | 0x21 | 0x28
                    | 0x29 | 0x30 | 0x31 | 0x86 | 0x87 => true,
                    0x80 | 0x81 | 0x83 => ext != 7,
                    0xF6 | 0xF7 => ext == 2 || ext == 3,
                    0xFE | 0xFF => ext < 2,
                    _ => false,
                }
            }
    }

    /// Execute one instruction, outside a [`Vcpu::run`] loop (unit tests, the
    /// differential-testing hook): memory may have changed behind our back, so
    /// the code-page cache is dropped first.
    fn exec(&mut self, mem: &mut GuestMemory) -> Step {
        self.code_page = NO_PAGE;
        self.step(mem)
    }

    /// Execute one instruction.
    #[allow(clippy::too_many_lines)]
    fn step(&mut self, mem: &mut GuestMemory) -> Step {
        if let Err(s) = self.fill_ibuf(mem) {
            return s;
        }
        self.addr32 = false;
        self.seg_base = 0;
        let mut p = Pfx::default();
        let mut pc = self.rip;
        // Legacy prefixes in any order, then an optional REX immediately
        // before the opcode (a REX followed by another prefix is ignored).
        let op = loop {
            let (b, next) = fetch!(self.fetch8(pc));
            pc = next;
            match b {
                0x40..=0x4F => {
                    p.rex = Rex::from_byte(b);
                    p.has_rex = true;
                    continue;
                }
                0x66 => p.opsize = true,
                0x67 => self.addr32 = true,
                0xF0 => p.lock = true,
                0xF2 => p.rep = 2,
                0xF3 => p.rep = 1,
                0x26 | 0x2E | 0x36 | 0x3E => self.seg_base = 0,
                0x64 => self.seg_base = self.fs_base,
                0x65 => self.seg_base = self.gs_base,
                _ => break b,
            }
            p.rex = Rex::default();
            p.has_rex = false;
        };
        if p.lock && op != 0x0F && !self.lock_ok(op, false, pc) {
            return Step::Illegal;
        }
        let width = p.width();
        match op {
            // ---- ALU: 00-3D ----
            0x00..=0x3F if op & 7 < 6 => {
                let alu = AluOp::from_ext(usize::from(op >> 3));
                match op & 7 {
                    0 => self.alu_modrm(mem, pc, p, alu, 8, false),
                    1 => self.alu_modrm(mem, pc, p, alu, width, false),
                    2 => self.alu_modrm(mem, pc, p, alu, 8, true),
                    3 => self.alu_modrm(mem, pc, p, alu, width, true),
                    4 => self.alu_acc_imm(pc, 8, alu),
                    _ => self.alu_acc_imm(pc, width, alu),
                }
            }
            // PUSH/POP r (16-bit under 0x66).
            0x50..=0x57 => {
                let r = usize::from(op & 7) | (usize::from(p.rex.b) << 3);
                let v = self.gpr[r];
                fetch!(self.push_w(mem, v, p.stack_width()));
                self.next(pc)
            }
            0x58..=0x5F => {
                let r = usize::from(op & 7) | (usize::from(p.rex.b) << 3);
                let w = p.stack_width();
                let v = fetch!(self.pop_w(mem, w));
                self.set_reg(r, v, w);
                self.next(pc)
            }
            // MOVSXD Gv, Ed (sign-extends under REX.W; a plain move otherwise).
            0x63 => {
                let (m, end) = fetch!(self.modrm(pc, p.rex));
                let src_w = width.min(32);
                let raw = fetch!(self.read_operand(mem, self.op_of(m.kind, end), src_w));
                self.set_reg(m.reg, sign_extend_w(raw, src_w) as u64, width);
                self.next(end)
            }
            0x68 | 0x6A => {
                let w = p.stack_width();
                let imm_w = if op == 0x6A {
                    8
                } else if w == 16 {
                    16
                } else {
                    32
                };
                let (imm, end) = fetch!(self.fetch_imm(pc, imm_w));
                fetch!(self.push_w(mem, imm as u64, w));
                self.next(end)
            }
            0x69 => self.imul_rm(mem, pc, p, width.min(32)),
            0x6B => self.imul_rm(mem, pc, p, 8),
            // INS/OUTS: I/O at CPL 3 with IOPL 0.
            0x6C..=0x6F => Step::Trap(Trap::Protection),
            0x70..=0x7F => {
                let (rel, end) = fetch!(self.fetch_i8(pc));
                if self.cond_holds(op) {
                    self.jump(end.wrapping_add(i64::from(rel) as u64))
                } else {
                    self.next(end)
                }
            }
            0x80 => self.group1(mem, pc, p, 8, true),
            0x81 => self.group1(mem, pc, p, width, false),
            0x83 => self.group1(mem, pc, p, width, true),
            0x84 => self.alu_modrm(mem, pc, p, AluOp::Test, 8, false),
            0x85 => self.alu_modrm(mem, pc, p, AluOp::Test, width, false),
            0x86 => self.xchg(mem, pc, p, 8),
            0x87 => self.xchg(mem, pc, p, width),
            0x88 => self.mov_modrm(mem, pc, p, 8, false),
            0x89 => self.mov_modrm(mem, pc, p, width, false),
            0x8A => self.mov_modrm(mem, pc, p, 8, true),
            0x8B => self.mov_modrm(mem, pc, p, width, true),
            0x8C => {
                // MOV r/m, Sreg: a register destination is written at the
                // operand size (zero-extended); memory always gets 16 bits.
                let (m, end) = fetch!(self.modrm(pc, p.rex));
                let sel = match m.ext() {
                    1 => USER_CS,
                    2 => USER_SS,
                    0 | 3..=5 => 0,
                    _ => return Step::Illegal,
                };
                match m.kind {
                    RmKind::Reg(r) => self.set_reg(r, u64::from(sel), width),
                    _ => {
                        fetch!(self.write_operand(
                            mem,
                            self.op_of(m.kind, end),
                            u64::from(sel),
                            16
                        ));
                    }
                }
                self.next(end)
            }
            0x8D => {
                // LEA computes the effective address (no segment base).
                let (m, end) = fetch!(self.modrm(pc, p.rex));
                let Some(ea) = self.ea_of(m.kind, end) else {
                    return Step::Illegal;
                };
                self.set_reg(m.reg, ea, width);
                self.next(end)
            }
            0x8E => {
                // MOV Sreg, r/m: only null selectors (DS/ES/FS/GS) and the
                // user data selector are loadable; CS is #UD.
                let (m, end) = fetch!(self.modrm(pc, p.rex));
                let v = fetch!(self.read_operand(mem, self.op_of(m.kind, end), 16)) as u16;
                match m.ext() {
                    1 | 6 | 7 => Step::Illegal,
                    2 if v != USER_SS => Step::Trap(Trap::Protection),
                    _ if v != 0 && v != USER_SS => Step::Trap(Trap::Protection),
                    _ => self.next(end),
                }
            }
            // POP r/m (8F /0). A memory destination that uses RSP as a base is
            // addressed with RSP *after* the pop's increment, so decode again
            // once RSP has moved.
            0x8F => {
                let (m, _) = fetch!(self.modrm(pc, p.rex));
                if m.ext() != 0 {
                    return Step::Illegal;
                }
                let w = p.stack_width();
                let saved = self.gpr[RSP];
                let v = fetch!(self.pop_w(mem, w));
                let (m, end) = fetch!(self.modrm(pc, p.rex));
                if let Err(s) = self.write_operand(mem, self.op_of(m.kind, end), v, w) {
                    self.gpr[RSP] = saved;
                    return s;
                }
                self.next(end)
            }
            // NOP / PAUSE (F3 90). With REX.B, 90 is `xchg rax, r8`.
            0x90 if !p.rex.b => self.next(pc),
            0x90..=0x97 => {
                let r = usize::from(op & 7) | (usize::from(p.rex.b) << 3);
                let a = self.gpr[RAX];
                let b = self.gpr[r];
                self.set_reg(RAX, b, width);
                self.set_reg(r, a, width);
                self.next(pc)
            }
            0x98 => {
                // CBW / CWDE / CDQE
                let half = width / 2;
                let v = sign_extend_w(mask_w(self.gpr[RAX], half), half) as u64;
                self.set_reg(RAX, v, width);
                self.next(pc)
            }
            0x99 => {
                // CWD / CDQ / CQO
                let v = if sign_bit(self.gpr[RAX], width) {
                    u64::MAX
                } else {
                    0
                };
                self.set_reg(RDX, v, width);
                self.next(pc)
            }
            // FWAIT: raises a pending unmasked x87 exception.
            0x9B => {
                if self.fpu_pending() {
                    return Step::Trap(Trap::X87);
                }
                self.next(pc)
            }
            0x9C => {
                // PUSHF (RF/VM read as 0 in the image).
                let v = self.rflags_word();
                fetch!(self.push_w(mem, v, p.stack_width()));
                self.next(pc)
            }
            0x9D => {
                let w = p.stack_width();
                let v = fetch!(self.pop_w(mem, w));
                let v = if w == 16 {
                    (self.rflags_word() & !0xffff) | v
                } else {
                    v
                };
                self.set_rflags_user(v);
                self.next(pc)
            }
            0x9E => {
                // SAHF: SF:ZF:x:AF:x:PF:x:CF <- AH
                let ah = self.gpr[RAX] >> 8;
                self.flags.cf = ah & 1 != 0;
                self.flags.pf = ah & 4 != 0;
                self.flags.af = ah & 0x10 != 0;
                self.flags.zf = ah & 0x40 != 0;
                self.flags.sf = ah & 0x80 != 0;
                self.next(pc)
            }
            0x9F => {
                // LAHF
                let ah = (self.rflags_word() & 0xd5) | 2;
                self.gpr[RAX] = (self.gpr[RAX] & !0xff00) | (ah << 8);
                self.next(pc)
            }
            0xA0..=0xA3 => {
                // MOV AL/eAX <-> moffs: a 64-bit absolute offset (32-bit
                // under 0x67).
                let (off, end) = if self.addr32 {
                    let (v, e) = fetch!(self.fetch32(pc));
                    (u64::from(v), e)
                } else {
                    fetch!(self.fetch64(pc))
                };
                let w = if op & 1 == 0 { 8 } else { width };
                let a = self.lin(off);
                if op < 0xA2 {
                    let v = fetch!(Self::read_mem(mem, a, w));
                    self.set_reg(RAX, v, w);
                } else {
                    fetch!(self.write_mem(mem, a, self.gpr[RAX], w));
                }
                self.next(end)
            }
            0xA4..=0xA7 | 0xAA..=0xAF => self.string_insn(mem, pc, p, op),
            0xA8 => self.alu_acc_imm(pc, 8, AluOp::Test),
            0xA9 => self.alu_acc_imm(pc, width, AluOp::Test),
            0xB0..=0xB7 => {
                let r = usize::from(op & 7) | (usize::from(p.rex.b) << 3);
                let (imm, end) = fetch!(self.fetch8(pc));
                fetch!(self.write_operand(mem, reg8_operand(r, p.has_rex), u64::from(imm), 8));
                self.next(end)
            }
            0xB8..=0xBF => {
                let r = usize::from(op & 7) | (usize::from(p.rex.b) << 3);
                let (imm, end) = match width {
                    64 => fetch!(self.fetch64(pc)),
                    16 => {
                        let (v, e) = fetch!(self.fetch16(pc));
                        (u64::from(v), e)
                    }
                    _ => {
                        let (v, e) = fetch!(self.fetch32(pc));
                        (u64::from(v), e)
                    }
                };
                self.set_reg(r, imm, width);
                self.next(end)
            }
            0xC0 => self.group2(mem, pc, p, 8, None),
            0xC1 => self.group2(mem, pc, p, width, None),
            0xC2 | 0xC3 => {
                // RET [imm16]: pop the return address (16-bit under 0x66),
                // then release imm16 bytes of arguments.
                let imm = if op == 0xC2 {
                    fetch!(self.fetch16(pc)).0
                } else {
                    0
                };
                let target = fetch!(self.pop_w(mem, p.stack_width()));
                self.gpr[RSP] = self.gpr[RSP].wrapping_add(u64::from(imm));
                self.jump(target)
            }
            0xC6 => self.mov_imm(mem, pc, p, 8),
            0xC7 => self.mov_imm(mem, pc, p, width),
            0xC8 => self.enter(mem, pc, p),
            0xC9 => self.leave(mem, pc, p),
            // Far returns/interrupt returns and software interrupts: no far
            // code segments or IDT gates are reachable from a user task.
            0xCA | 0xCB | 0xCD | 0xCF => Step::Trap(Trap::Protection),
            0xCC => Step::Trap(Trap::Breakpoint),
            0xD0 => self.group2(mem, pc, p, 8, Some(1)),
            0xD1 => self.group2(mem, pc, p, width, Some(1)),
            0xD2 => self.group2(mem, pc, p, 8, Some(self.gpr[RCX] as u8)),
            0xD3 => self.group2(mem, pc, p, width, Some(self.gpr[RCX] as u8)),
            0xD7 => {
                // XLAT: AL = [seg:rBX + AL]
                let ea = self.sreg(RBX).wrapping_add(self.gpr[RAX] & 0xff);
                let ea = if self.addr32 { ea & 0xffff_ffff } else { ea };
                let v = fetch!(Self::read_mem(mem, self.lin(ea), 8));
                self.set_reg(RAX, v, 8);
                self.next(pc)
            }
            0xD8..=0xDF => self.exec_x87(mem, pc, p, op),
            0xE0..=0xE3 => {
                // LOOPNE/LOOPE/LOOP/JrCXZ rel8 (ECX under 0x67).
                let (rel, end) = fetch!(self.fetch_i8(pc));
                let take = if op == 0xE3 {
                    self.sreg(RCX) == 0
                } else {
                    let c = self.sreg(RCX).wrapping_sub(1);
                    self.set_sreg(RCX, c);
                    c != 0
                        && match op {
                            0xE0 => !self.flags.zf,
                            0xE1 => self.flags.zf,
                            _ => true,
                        }
                };
                if take {
                    self.jump(end.wrapping_add(i64::from(rel) as u64))
                } else {
                    self.next(end)
                }
            }
            // IN/OUT.
            0xE4..=0xE7 | 0xEC..=0xEF => Step::Trap(Trap::Protection),
            0xE8 => {
                let (rel, end) = fetch!(self.fetch_i32(pc));
                fetch!(self.push(mem, end));
                self.jump(end.wrapping_add(i64::from(rel) as u64))
            }
            0xE9 => {
                let (rel, end) = fetch!(self.fetch_i32(pc));
                self.jump(end.wrapping_add(i64::from(rel) as u64))
            }
            0xEB => {
                let (rel, end) = fetch!(self.fetch_i8(pc));
                self.jump(end.wrapping_add(i64::from(rel) as u64))
            }
            0xF1 => Step::Trap(Trap::Breakpoint), // INT1 / ICEBP
            // HLT, CLI, STI: privileged at CPL 3.
            0xF4 | 0xFA | 0xFB => Step::Trap(Trap::Protection),
            0xF5 => {
                self.flags.cf = !self.flags.cf;
                self.next(pc)
            }
            0xF6 => self.group3(mem, pc, p, 8),
            0xF7 => self.group3(mem, pc, p, width),
            0xF8 | 0xF9 => {
                self.flags.cf = op == 0xF9;
                self.next(pc)
            }
            0xFC | 0xFD => {
                self.df = op == 0xFD;
                self.next(pc)
            }
            0xFE => self.group4(mem, pc, p),
            0xFF => self.group5(mem, pc, p),
            0x0F => self.exec_0f(mem, pc, p),
            // 06/07/0E/16/17/1E/1F/27/2F/37/3F/60-62/82/9A/C4/C5/CE/D4-D6/EA:
            // invalid in 64-bit mode.
            _ => Step::Illegal,
        }
    }

    /// The two-byte (`0F xx`) opcode map: general-purpose and system
    /// instructions here; SIMD opcodes go to [`X86Interp::exec_simd`].
    #[allow(clippy::too_many_lines)]
    fn exec_0f(&mut self, mem: &mut GuestMemory, pc: u64, p: Pfx) -> Step {
        let (op, pc) = fetch!(self.fetch8(pc));
        if p.lock && !self.lock_ok(op, true, pc) {
            return Step::Illegal;
        }
        let width = p.width();
        match op {
            // SLDT/STR/LLDT/LTR/VERR/VERW (group 6): system instructions
            // (UMIP blocks the stores at CPL 3).
            0x00 => Step::Trap(Trap::Protection),
            0x01 => self.group7(mem, pc, p),
            // LAR/LSL, CLTS, INVD, WBINVD, MOV CR/DR, WRMSR, RDMSR, RDPMC,
            // SYSENTER/SYSEXIT, SYSRET: privileged/descriptor-table access.
            0x02 | 0x03 | 0x06 | 0x07 | 0x08 | 0x09 | 0x20..=0x23 | 0x30 | 0x32..=0x35 => {
                Step::Trap(Trap::Protection)
            }
            0x05 => {
                // `syscall` copies RIP→RCX and RFLAGS→R11 before entering the
                // kernel, exactly as hardware does; `rip` stays on the opcode —
                // the kernel advances it when it writes the return value.
                self.gpr[RCX] = pc;
                self.gpr[R11] = self.rflags_word();
                Step::Syscall
            }
            // PREFETCH/PREFETCHW (0F 0D) and the hint-NOP space 0F 18-1F
            // (prefetchT0/1/2/NTA, the multi-byte NOP, ENDBR64/32 and other
            // reserved NOPs): all execute as NOPs; the ModRM is decoded only
            // to consume the instruction's length.
            0x0D | 0x18..=0x1F => {
                let (_, end) = fetch!(self.modrm(pc, p.rex));
                self.next(end)
            }
            0x31 => {
                // RDTSC
                let t = self.rdtsc_tick();
                self.gpr[RAX] = t & 0xffff_ffff;
                self.gpr[RDX] = t >> 32;
                self.next(pc)
            }
            0x40..=0x4F => {
                // CMOVcc: the source is read even when the condition is false
                // (a faulting operand faults), and a 32-bit destination is
                // zero-extended either way.
                let (m, end) = fetch!(self.modrm(pc, p.rex));
                let v = fetch!(self.read_operand(mem, self.op_of(m.kind, end), width));
                if self.cond_holds(op) {
                    self.set_reg(m.reg, v, width);
                } else if width == 32 {
                    self.gpr[m.reg] &= 0xffff_ffff;
                }
                self.next(end)
            }
            0x80..=0x8F => {
                let (rel, end) = fetch!(self.fetch_i32(pc));
                if self.cond_holds(op) {
                    self.jump(end.wrapping_add(i64::from(rel) as u64))
                } else {
                    self.next(end)
                }
            }
            0x90..=0x9F => {
                let (m, end) = fetch!(self.modrm(pc, p.rex));
                let v = u64::from(self.cond_holds(op));
                fetch!(self.write_operand(mem, self.op8_of(m.kind, end, p.has_rex), v, 8));
                self.next(end)
            }
            0xA0 | 0xA8 => {
                // PUSH FS/GS: the (null) selector.
                fetch!(self.push_w(mem, 0, p.stack_width()));
                self.next(pc)
            }
            0xA1 | 0xA9 => {
                // POP FS/GS: only the null selector loads (base unchanged).
                let w = p.stack_width();
                let sp = self.gpr[RSP];
                let v = fetch!(Self::read_mem(mem, sp, w));
                if v & 0xffff != 0 {
                    return Step::Trap(Trap::Protection);
                }
                self.gpr[RSP] = sp.wrapping_add(u64::from(w / 8));
                self.next(pc)
            }
            0xA2 => {
                self.cpuid();
                self.next(pc)
            }
            0xA3 | 0xAB | 0xB3 | 0xBB => {
                let (m, end) = fetch!(self.modrm(pc, p.rex));
                let off = self.gpr[m.reg];
                let bt = match op {
                    0xA3 => BitTestOp::Bt,
                    0xAB => BitTestOp::Bts,
                    0xB3 => BitTestOp::Btr,
                    _ => BitTestOp::Btc,
                };
                self.bit_test(mem, m, end, width, off, true, bt)
            }
            0xBA => {
                let (m, pc2) = fetch!(self.modrm(pc, p.rex));
                let (imm, end) = fetch!(self.fetch8(pc2));
                let bt = match m.ext() {
                    4 => BitTestOp::Bt,
                    5 => BitTestOp::Bts,
                    6 => BitTestOp::Btr,
                    7 => BitTestOp::Btc,
                    _ => return Step::Illegal,
                };
                self.bit_test(mem, m, end, width, u64::from(imm), false, bt)
            }
            0xA4 => self.shld_shrd(mem, pc, p, true, false),
            0xA5 => self.shld_shrd(mem, pc, p, true, true),
            0xAC => self.shld_shrd(mem, pc, p, false, false),
            0xAD => self.shld_shrd(mem, pc, p, false, true),
            0xAE => self.group15(mem, pc, p),
            0xAF => self.imul_rm(mem, pc, p, 0),
            0xB0 => self.cmpxchg(mem, pc, p, 8),
            0xB1 => self.cmpxchg(mem, pc, p, width),
            // LSS/LFS/LGS: far-pointer loads need descriptor tables.
            0xB2 | 0xB4 | 0xB5 => {
                let (m, end) = fetch!(self.modrm(pc, p.rex));
                if let Err(s) = self.mem_only(m.kind, end) {
                    return s;
                }
                Step::Trap(Trap::Protection)
            }
            0xB6 | 0xB7 | 0xBE | 0xBF => {
                // MOVZX/MOVSX Gv, Eb/Ew.
                let src_w = if op & 1 == 0 { 8 } else { 16 };
                let (m, end) = fetch!(self.modrm(pc, p.rex));
                let src = self.opw_of(m.kind, end, src_w, p);
                let raw = fetch!(self.read_operand(mem, src, src_w));
                let v = if op >= 0xBE {
                    sign_extend_w(raw, src_w) as u64
                } else {
                    raw
                };
                self.set_reg(m.reg, v, width);
                self.next(end)
            }
            0xB8 if p.rep == 1 => self.popcnt(mem, pc, p),
            // BSF/BSR — and, under F3, TZCNT/LZCNT on CPUs with BMI1/ABM,
            // which this one doesn't advertise: there the F3 is ignored.
            0xBC => self.bit_scan(mem, pc, p, false),
            0xBD => self.bit_scan(mem, pc, p, true),
            0xC0 => self.xadd(mem, pc, p, 8),
            0xC1 => self.xadd(mem, pc, p, width),
            0xC3 if p.rep == 0 => {
                // MOVNTI Md/q, Gd/q (memory only): an ordinary store here.
                let (m, end) = fetch!(self.modrm(pc, p.rex));
                let a = fetch!(self.mem_only(m.kind, end));
                let w = if p.rex.w { 64 } else { 32 };
                fetch!(self.write_mem(mem, a, self.gpr[m.reg], w));
                self.next(end)
            }
            0xC7 => self.group9(mem, pc, p),
            0xC8..=0xCF => {
                // BSWAP r (a 16-bit BSWAP zeroes the low word, as hardware).
                let r = usize::from(op & 7) | (usize::from(p.rex.b) << 3);
                let v = match width {
                    64 => self.gpr[r].swap_bytes(),
                    16 => 0,
                    _ => u64::from((self.gpr[r] as u32).swap_bytes()),
                };
                self.set_reg(r, v, width);
                self.next(pc)
            }
            0x38 => {
                let (op3, pc) = fetch!(self.fetch8(pc));
                self.exec_0f38(mem, pc, p, op3)
            }
            0x3A => {
                let (op3, pc) = fetch!(self.fetch8(pc));
                self.exec_0f3a(mem, pc, p, op3)
            }
            0x10..=0x17 | 0x28..=0x2F | 0x50..=0x7F | 0xC2 | 0xC4..=0xC6 | 0xD0..=0xFE => {
                self.exec_simd(mem, pc, p, op)
            }
            // UD2, UD1, UD0, JMPE, RSM, FEMMS/3DNow!, and the unassigned rest.
            _ => Step::Illegal,
        }
    }

    /// Group 7 (`0F 01`). From user mode: `RDTSCP`; `SGDT`/`SIDT`/`SMSW`,
    /// which Linux's UMIP emulation answers with fixed dummy values (a zero
    /// limit and a kernel-half base for the tables, the usual CR0 bits for
    /// `SMSW`); the privileged forms (`LGDT`/`LIDT`/`LMSW`/`INVLPG`/
    /// `SWAPGS`) raise `#GP`. `XGETBV` needs CR4.OSXSAVE, which a CPU that
    /// doesn't advertise XSAVE never sets, and `MONITOR`/`MWAIT`, `CLAC`/
    /// `STAC`, `XTEST`, `RDPKRU`, … aren't available: `#UD`.
    fn group7(&mut self, mem: &mut GuestMemory, pc: u64, p: Pfx) -> Step {
        const UMIP_GDT_BASE: u64 = 0xffff_ffff_fffe_0000;
        const UMIP_IDT_BASE: u64 = 0xffff_ffff_ffff_0000;
        const UMIP_CR0: u64 = 0x8005_0033;
        let (b, _) = fetch!(self.fetch8(pc));
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        match (m.ext(), m.kind) {
            (0 | 1, RmKind::Mem(_) | RmKind::MemRip(_)) => {
                // SGDT/SIDT m: 2-byte limit, 8-byte base.
                let a = fetch!(self.mem_only(m.kind, end));
                let base = if m.ext() == 0 {
                    UMIP_GDT_BASE
                } else {
                    UMIP_IDT_BASE
                };
                let mut img = [0u8; 10];
                img[2..].copy_from_slice(&base.to_le_bytes());
                fetch!(self.store(mem, a, &img));
                self.next(end)
            }
            (4, RmKind::Reg(r)) => {
                self.set_reg(r, UMIP_CR0, p.width());
                self.next(end)
            }
            (4, _) => {
                let a = fetch!(self.mem_only(m.kind, end));
                fetch!(self.write_mem(mem, a, UMIP_CR0, 16));
                self.next(end)
            }
            (7, RmKind::Reg(_)) if b == 0xF9 => {
                // RDTSCP: like RDTSC, plus ECX = TSC_AUX (cpu 0).
                let t = self.rdtsc_tick();
                self.gpr[RAX] = t & 0xffff_ffff;
                self.gpr[RDX] = t >> 32;
                self.gpr[RCX] = 0;
                self.next(end)
            }
            (6, _) | (2 | 3 | 7, RmKind::Mem(_) | RmKind::MemRip(_)) => {
                Step::Trap(Trap::Protection)
            }
            (7, RmKind::Reg(_)) if b == 0xF8 => Step::Trap(Trap::Protection), // SWAPGS
            _ => Step::Illegal,
        }
    }

    /// Group 15 (`0F AE`): `FXSAVE`/`FXRSTOR` (`/0`/`/1`), `LDMXCSR`/`STMXCSR`
    /// (`/2`/`/3`), `CLFLUSH` (`/7` memory), and the fences `LFENCE`/
    /// `MFENCE`/`SFENCE` (`/5`/`/6`/`/7` register). `XSAVE*`/`FSGSBASE` and
    /// the other forms aren't advertised and are `#UD`.
    fn group15(&mut self, mem: &mut GuestMemory, pc: u64, p: Pfx) -> Step {
        let (m, end) = fetch!(self.modrm(pc, p.rex));
        // (F3 0F AE selects the FSGSBASE group, not advertised; a 66 is
        // ignored, as on hardware — 66 0F AE /7 is CLFLUSHOPT.)
        if p.rep != 0 {
            return Step::Illegal;
        }
        match (m.kind, m.ext()) {
            (RmKind::Reg(_), 5..=7) => self.next(end),
            (RmKind::Reg(_), _) => Step::Illegal,
            (_, 0 | 1) => {
                let a = fetch!(self.mem_only(m.kind, end));
                if a & 15 != 0 {
                    return Step::Trap(Trap::Protection);
                }
                if m.ext() == 0 {
                    // Bytes 464..512 belong to software: FXSAVE leaves them.
                    let img = self.fxsave_image(p.rex.w);
                    fetch!(self.store(mem, a, &img[..464]));
                } else {
                    let mut img = [0u8; 512];
                    fetch!(mem.read(a, &mut img).map_err(|_| rd_fault(a)));
                    if !self.fxrstor_image(&img, p.rex.w) {
                        return Step::Trap(Trap::Protection);
                    }
                }
                self.next(end)
            }
            (_, 2) => {
                let a = fetch!(self.mem_only(m.kind, end));
                let v = fetch!(Self::read_mem(mem, a, 32)) as u32;
                if v & !MXCSR_MASK != 0 {
                    return Step::Trap(Trap::Protection);
                }
                self.mxcsr = v;
                self.next(end)
            }
            (_, 3) => {
                let a = fetch!(self.mem_only(m.kind, end));
                fetch!(self.write_mem(mem, a, u64::from(self.mxcsr), 32));
                self.next(end)
            }
            (_, 7) => {
                // CLFLUSH: the line must be addressable (a fault otherwise).
                let a = fetch!(self.mem_only(m.kind, end));
                fetch!(Self::read_mem(mem, a, 8));
                self.next(end)
            }
            _ => Step::Illegal,
        }
    }

    /// The 512-byte `FXSAVE` image of the x87/MMX/SSE state (Intel SDM Vol. 1
    /// §10.5.1): control/status words, abridged tag, last opcode/pointers
    /// (64-bit `FIP`/`FDP` for the `REX.W` form, else 32-bit offsets with a
    /// zero selector), `MXCSR` (+ its mask), the eight `ST(i)`/`MMi` slots in
    /// stack order, and `XMM0..15`. Reserved bytes are zero.
    fn fxsave_image(&self, rex_w: bool) -> [u8; 512] {
        let mut img = [0u8; 512];
        img[0..2].copy_from_slice(&self.fpu_cw.to_le_bytes());
        img[2..4].copy_from_slice(&self.fpu_sw().to_le_bytes());
        img[4] = self.fpu_tag;
        img[6..8].copy_from_slice(&self.fpu_fop.to_le_bytes());
        if rex_w {
            img[8..16].copy_from_slice(&self.fpu_fip.to_le_bytes());
            img[16..24].copy_from_slice(&self.fpu_fdp.to_le_bytes());
        } else {
            img[8..12].copy_from_slice(&(self.fpu_fip as u32).to_le_bytes());
            img[16..20].copy_from_slice(&(self.fpu_fdp as u32).to_le_bytes());
        }
        img[24..28].copy_from_slice(&self.mxcsr.to_le_bytes());
        img[28..32].copy_from_slice(&MXCSR_MASK.to_le_bytes());
        for i in 0..8u8 {
            let o = 32 + 16 * usize::from(i);
            img[o..o + 10].copy_from_slice(&self.st_get(i).0.to_le_bytes()[..10]);
        }
        for (i, x) in self.xmm.iter().enumerate() {
            img[160 + 16 * i..176 + 16 * i].copy_from_slice(&x.to_le_bytes());
        }
        img
    }

    /// Load an `FXSAVE` image (see [`X86Interp::fxsave_image`]). Returns
    /// `false` — the `#GP` `FXRSTOR` raises — when the image's `MXCSR` sets a
    /// bit outside [`MXCSR_MASK`], leaving the state untouched.
    fn fxrstor_image(&mut self, img: &[u8; 512], rex_w: bool) -> bool {
        let mxcsr = u32::from_le_bytes(img[24..28].try_into().unwrap());
        if mxcsr & !MXCSR_MASK != 0 {
            return false;
        }
        self.mxcsr = mxcsr;
        self.set_fpu_cw(u16::from_le_bytes([img[0], img[1]]));
        self.set_fpu_sw(u16::from_le_bytes([img[2], img[3]]));
        self.fpu_fop = u16::from_le_bytes([img[6], img[7]]) & 0x7ff;
        if rex_w {
            self.fpu_fip = u64::from_le_bytes(img[8..16].try_into().unwrap());
            self.fpu_fdp = u64::from_le_bytes(img[16..24].try_into().unwrap());
        } else {
            self.fpu_fip = u64::from(u32::from_le_bytes(img[8..12].try_into().unwrap()));
            self.fpu_fdp = u64::from(u32::from_le_bytes(img[16..20].try_into().unwrap()));
        }
        for i in 0..8u8 {
            let o = 32 + 16 * usize::from(i);
            let mut b = [0u8; 16];
            b[..10].copy_from_slice(&img[o..o + 10]);
            self.st_set(i, F80(u128::from_le_bytes(b)));
        }
        self.fpu_tag = img[4];
        for (i, x) in self.xmm.iter_mut().enumerate() {
            *x = u128::from_le_bytes(img[160 + 16 * i..176 + 16 * i].try_into().unwrap());
        }
        true
    }
}

/// `CPUID` leaf 4 (deterministic cache parameters), subleaf `sub`: L1d, L1i,
/// L2, L3, then the terminating null entry. `EBX` = (ways-1) << 22 | (line
/// size-1); `ECX` = sets-1.
fn cache_leaf(sub: u32) -> (u32, u32, u32, u32) {
    // (type: 1 data / 2 instruction / 3 unified, level, ways, sets)
    let (ty, level, ways, sets): (u32, u32, u32, u32) = match sub {
        0 => (1, 1, 8, 64),    // 32 KiB L1d
        1 => (2, 1, 8, 64),    // 32 KiB L1i
        2 => (3, 2, 4, 1024),  // 256 KiB L2
        3 => (3, 3, 16, 8192), // 8 MiB L3
        _ => return (0, 0, 0, 0),
    };
    let eax = ty | (level << 5) | (1 << 8); // self-initializing
    let ebx = ((ways - 1) << 22) | 63;
    (eax, ebx, sets - 1, 0)
}

impl Vcpu for X86Interp {
    fn run(&mut self, mem: &mut GuestMemory) -> Result<Exit, VcpuError> {
        // Time-based preemption deadline, computed once. A compute-bound guest
        // that never syscalls would otherwise run the whole MAX_STEPS budget and
        // starve its siblings; expiring the quantum ends the slice as
        // Interrupted (the scheduler keeps the task runnable and resumes it).
        let deadline = self.quantum.map(|q| Instant::now() + q);
        // The kernel may have remapped or rewritten memory since the last run.
        self.code_page = NO_PAGE;
        for i in 0..MAX_STEPS {
            match self.step(mem) {
                Step::Next | Step::Branched => {}
                Step::Syscall => return Ok(Exit::Syscall),
                Step::Illegal => return Ok(Exit::IllegalInstruction { pc: self.rip }),
                Step::Fault { addr, write } => return Ok(Exit::MemFault { addr, write }),
                // `#GP` is a `SIGSEGV` with `si_addr == 0` on Linux — exactly
                // what an unresolvable fault at address 0 becomes. The
                // `SIGFPE`/`SIGTRAP` exceptions have no `Exit` of their own
                // yet, so they surface as an illegal instruction (`SIGILL`).
                Step::Trap(Trap::Protection) => {
                    return Ok(Exit::MemFault {
                        addr: 0,
                        write: false,
                    });
                }
                Step::Trap(_) => return Ok(Exit::IllegalInstruction { pc: self.rip }),
            }
            // Poll the wall clock only every QUANTUM_STRIDE instructions — a read
            // per instruction would swamp the interpreter's per-op cost.
            if i & (QUANTUM_STRIDE - 1) == 0
                && (deadline.is_some_and(|d| Instant::now() >= d) || super::yield_due())
            {
                return Ok(Exit::Interrupted);
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

    fn rflags(&self) -> u64 {
        self.rflags_word()
    }

    fn set_rflags(&mut self, v: u64) {
        self.set_rflags_user(v);
    }

    fn simd_state(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(256);
        for x in &self.xmm {
            out.extend_from_slice(&x.to_le_bytes());
        }
        out
    }

    fn set_simd_state(&mut self, bytes: &[u8]) {
        for (i, chunk) in bytes.as_chunks::<16>().0.iter().take(16).enumerate() {
            self.xmm[i] = u128::from_le_bytes(*chunk);
        }
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
        self.rflags_sys = 0;
        self.fs_base = 0;
        self.gs_base = 0;
        self.mxcsr = 0x1f80;
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
    fn executing_a_non_exec_page_faults() {
        // A page mapped RW (no EXEC) holds a valid `nop`; executing from it must
        // fault at rip rather than run the byte — NX enforcement.
        let mut m = GuestMemory::new(0x1_0000, 16 * crate::vcpu::mem::PAGE_SIZE);
        m.map(0x1_0000, crate::vcpu::mem::PAGE_SIZE, Prot::rw())
            .unwrap(); // data page
        m.map(0x1_1000, crate::vcpu::mem::PAGE_SIZE, Prot::rx())
            .unwrap(); // code page
        m.write_init(0x1_0000, &[0x90]).unwrap(); // nop on the data page
        m.write_init(0x1_1000, &[0x90]).unwrap(); // nop on the code page

        // Executing the data page faults at its address.
        let mut cpu = X86Interp::new(0x1_0000, STACK);
        assert!(matches!(
            cpu.exec(&mut m),
            Step::Fault {
                addr: 0x1_0000,
                write: false
            }
        ));
        // Executing the code page runs the nop.
        let mut cpu = X86Interp::new(0x1_1000, STACK);
        assert!(matches!(cpu.exec(&mut m), Step::Next));
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

    #[test]
    fn divsd_honors_mxcsr_rounding_and_flags() {
        // 1/10 rounds *up* under round-to-nearest, so round-toward-zero yields a
        // distinct (one ulp lower) result — proving MXCSR's RC field is honored.
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.xmm[0] = u128::from(1.0f64.to_bits());
        cpu.xmm[1] = u128::from(10.0f64.to_bits());
        cpu.mxcsr = 0x7f80; // exceptions masked, RC = toward zero
        m.write_init(CODE, &[0xF2, 0x0F, 0x5E, 0xC1]).unwrap(); // divsd xmm0, xmm1
        cpu.exec(&mut m);
        let got = cpu.xmm[0] as u64;
        let want = crate::vcpu::softfloat::f64_op(
            1.0f64.to_bits(),
            10.0f64.to_bits(),
            crate::vcpu::softfloat::Op::Div,
            crate::vcpu::softfloat::Mx {
                mode: crate::vcpu::softfloat::Round::Zero,
                daz: false,
                ftz: false,
            },
        )
        .0;
        assert_eq!(got, want, "divsd rounded per MXCSR");
        assert_eq!(
            got,
            (1.0f64 / 10.0).to_bits() - 1,
            "one ulp below round-to-nearest"
        );
        assert!(cpu.mxcsr & 0x20 != 0, "PE (inexact) flag accumulated");
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
        assert_eq!(cpu.st_get(0).to_f64(), 6.0);
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
        assert_eq!(cpu.st_get(0).to_f64(), 4.0);
    }

    #[test]
    fn fld1_fldz_constants() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        // fld1 (D9 E8) ; fldz (D9 EE)
        m.write_init(CODE, &[0xD9, 0xE8, 0xD9, 0xEE]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.st_get(0).to_f64(), 1.0);
        cpu.exec(&mut m);
        assert_eq!(
            cpu.st_get(0).to_f64(),
            0.0,
            "FLDZ pushes 0.0 as the new ST(0)"
        );
        assert_eq!(cpu.st_get(1).to_f64(), 1.0, "FLD1's value is still ST(1)");
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
    fn fprem_computes_fmod_and_reports_complete() {
        // The `FPREM; FNSTSW; TEST AH,4; JNZ` reduction loop musl/libm emit for
        // `fmod`: one FPREM must produce the full remainder and clear C2 so the
        // loop runs exactly once. (This exact sequence SIGILL'd node's
        // `Number.toString(16)` before FPREM was implemented.)
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.fpu_push(F80::from_f64_val(2.0)); // ST(1) divisor
        cpu.fpu_push(F80::from_f64_val(5.3)); // ST(0) dividend
        m.write_init(CODE, &[0xD9, 0xF8]).unwrap(); // FPREM
        cpu.exec(&mut m);
        assert!(
            (cpu.st_get(0).to_f64() - (5.3f64 % 2.0)).abs() < 1e-12,
            "{}",
            cpu.st_get(0).to_f64()
        );
        assert!(!cpu.fpu_c2, "single-step reduction is always complete");
        // trunc(5.3/2.0) = 2 = 0b010 → Q0=0 (C1), Q1=1 (C3), Q2=0 (C0).
        assert!(
            !cpu.fpu_c1 && cpu.fpu_c3 && !cpu.fpu_c0,
            "quotient bits in C1/C3/C0"
        );
    }

    #[test]
    fn fprem1_uses_the_nearest_even_quotient() {
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.fpu_push(F80::from_f64_val(2.0)); // ST(1)
        cpu.fpu_push(F80::from_f64_val(5.3)); // ST(0)
        m.write_init(CODE, &[0xD9, 0xF5]).unwrap(); // FPREM1
        cpu.exec(&mut m);
        // IEEE remainder: 5.3 - 2.0*round(2.65) = 5.3 - 6.0 = -0.7.
        assert!(
            (cpu.st_get(0).to_f64() + 0.7).abs() < 1e-12,
            "{}",
            cpu.st_get(0).to_f64()
        );
        assert!(!cpu.fpu_c2);
    }

    #[test]
    fn x87_transcendentals_match_f64_math() {
        // FSIN(π/2) = 1.
        let mut m = mem();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.fpu_push(F80::from_f64_val(std::f64::consts::FRAC_PI_2));
        m.write_init(CODE, &[0xD9, 0xFE]).unwrap();
        cpu.exec(&mut m);
        assert!((cpu.st_get(0).to_f64() - 1.0).abs() < 1e-12);

        // FSCALE: 3.0 * 2^trunc(4.7) = 3 * 16 = 48.
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.fpu_push(F80::from_f64_val(4.7)); // ST(1)
        cpu.fpu_push(F80::from_f64_val(3.0)); // ST(0)
        m.write_init(CODE, &[0xD9, 0xFD]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.st_get(0).to_f64(), 48.0);

        // F2XM1(0.5) = 2^0.5 - 1.
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.fpu_push(F80::from_f64_val(0.5));
        m.write_init(CODE, &[0xD9, 0xF0]).unwrap();
        cpu.exec(&mut m);
        assert!((cpu.st_get(0).to_f64() - (2f64.sqrt() - 1.0)).abs() < 1e-12);

        // FYL2X: ST(1)*log2(ST(0)), then pop → 3 * log2(8) = 9.
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.fpu_push(F80::from_f64_val(3.0)); // ST(1) = y
        cpu.fpu_push(F80::from_f64_val(8.0)); // ST(0) = x
        m.write_init(CODE, &[0xD9, 0xF1]).unwrap();
        cpu.exec(&mut m);
        assert!((cpu.st_get(0).to_f64() - 9.0).abs() < 1e-12);
        assert_eq!(cpu.fpu_top, 7, "FYL2X pops one operand");

        // FPATAN: atan2(ST(1), ST(0)), then pop → atan2(1,1) = π/4.
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.fpu_push(F80::from_f64_val(1.0)); // ST(1) = y
        cpu.fpu_push(F80::from_f64_val(1.0)); // ST(0) = x
        m.write_init(CODE, &[0xD9, 0xF3]).unwrap();
        cpu.exec(&mut m);
        assert!((cpu.st_get(0).to_f64() - std::f64::consts::FRAC_PI_4).abs() < 1e-12);

        // FXTRACT: 12.0 = 1.5 * 2^3 → exponent 3 in ST(1), significand 1.5 in ST(0).
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.fpu_push(F80::from_f64_val(12.0));
        m.write_init(CODE, &[0xD9, 0xF4]).unwrap();
        cpu.exec(&mut m);
        assert_eq!(cpu.st_get(0).to_f64(), 1.5, "significand in [1,2)");
        assert_eq!(cpu.st_get(1).to_f64(), 3.0, "unbiased exponent");
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
        assert_eq!(cpu.st_get(0).to_f64(), 1.0);
        assert_eq!(cpu.st_get(1).to_f64(), 2.0);
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
            &[
                0x66, 0x66, 0x2E, 0x0F, 0x1F, 0x84, 0x00, 0x00, 0x00, 0x00, 0x00,
            ],
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
        m.write_init(CODE + 9, &[0x48, 0x8B, 0x1C, 0x25, 0x00, 0x20, 0x01, 0x00])
            .unwrap();
        cpu.exec(&mut m);
        assert_eq!(
            cpu.gpr[RBX], 42,
            "seg base must not leak across instructions"
        );
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
        m.write_init(CODE + 2, &[0x05, 0x00, 0x01, 0x00, 0x00])
            .unwrap();
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
        assert_eq!(
            cpu.gpr[RSI], 0xC000_0000_0000_0001,
            "arithmetic: sign fills"
        );
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
            cpu.xmm[1], 0xFFFF_FFFF_FFFF_FFFF_1111_2222_3333_4444,
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
        m.write_init(CODE, &[0x66, 0xC7, 0x45, 0x00, 0x34, 0x12])
            .unwrap();
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
        assert_eq!(
            cpu.xmm[0],
            0x0800_0001_0000_0fff_u128 << 64 | 0x0fff_0008_0001_0800
        );
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
        m.write_init(CODE, &[0x66, 0x0F, 0x3A, 0x0F, 0xC1, 0x0F])
            .unwrap();
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

    #[test]
    fn time_quantum_preempts_a_syscall_free_hot_loop() {
        // A guest that spins forever with no syscall (`jmp $`, the archetypal
        // JIT hot loop / GC sweep) must still hand the CPU back on the wall-clock
        // quantum. Without time-based preemption `run` would spin the whole
        // MAX_STEPS budget (tens of millions of iterations) before yielding.
        let mut m = mem();
        m.write_init(CODE, &[0xEB, 0xFE]).unwrap(); // jmp $ (rel8 = -2)
        let mut cpu = X86Interp::new(CODE, STACK);
        let quantum = Duration::from_millis(20);
        cpu.quantum = Some(quantum);

        let started = Instant::now();
        let exit = cpu.run(&mut m).unwrap();
        let elapsed = started.elapsed();

        assert_eq!(exit, Exit::Interrupted, "the quantum ends the slice");
        assert_eq!(cpu.rip, CODE, "still parked on the self-loop, resumable");
        // It ran until (roughly) the quantum, not instantly and not for the
        // whole MAX_STEPS budget: the upper bound is far below the time tens of
        // millions of interpreted `jmp`s take, so a broken quantum fails here.
        assert!(
            elapsed >= quantum,
            "slice lasted at least the quantum: {elapsed:?}"
        );
        assert!(
            elapsed < quantum * 15,
            "slice ended near the quantum, not at MAX_STEPS: {elapsed:?}"
        );
    }

    #[test]
    fn no_quantum_runs_to_max_steps_on_a_hot_loop() {
        // With time-based preemption disabled the same self-loop still
        // terminates — via the MAX_STEPS runaway guard — so `run` never hangs.
        let mut m = mem();
        m.write_init(CODE, &[0xEB, 0xFE]).unwrap();
        let mut cpu = X86Interp::new(CODE, STACK);
        cpu.quantum = None;
        assert_eq!(cpu.run(&mut m).unwrap(), Exit::Interrupted);
    }

    /// Coverage scan: execute every instruction encoding listed in the file
    /// named by `NIXVM_SCAN_X86` (lines of `hexbytes mnemonic operands`, e.g.
    /// from `llvm-objdump -d -M intel` over a corpus) once, from a sane state
    /// with every GPR pointing into mapped memory, and report the encodings
    /// that decode as `#UD` (grouped by mnemonic). `v`-prefixed (VEX/EVEX)
    /// mnemonics and other above-baseline extensions are listed separately.
    #[test]
    #[ignore = "corpus coverage report; run with NIXVM_SCAN_X86=<words file>"]
    fn scan_instruction_coverage() {
        let Ok(path) = std::env::var("NIXVM_SCAN_X86") else {
            return;
        };
        let text = std::fs::read_to_string(path).unwrap();
        let base = 0x1_0000u64;
        let page = crate::vcpu::mem::PAGE_SIZE;
        let mut m = GuestMemory::new(base, 64 * page);
        m.map(base, 64 * page, Prot::rwx()).unwrap();
        let code = base + 32 * page;
        let mut by_mnemonic: std::collections::BTreeMap<String, (usize, String, bool)> =
            std::collections::BTreeMap::new();
        let mut total = 0usize;
        std::panic::set_hook(Box::new(|_| {}));
        for line in text.lines() {
            let mut it = line.splitn(2, ' ');
            let (Some(hex), Some(asm)) = (it.next(), it.next()) else {
                continue;
            };
            let Ok(bytes) = (0..hex.len() / 2)
                .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16))
                .collect::<Result<Vec<u8>, _>>()
            else {
                continue;
            };
            let mnemonic = asm.split_whitespace().next().unwrap_or("").to_string();
            if mnemonic.is_empty() || mnemonic.starts_with('<') || mnemonic == "ud2" {
                continue; // data in .text, or a deliberate trap
            }
            total += 1;
            m.write_init(code, &bytes).unwrap();
            let mut c = X86Interp::new(code, base + 16 * page);
            for r in 0..16 {
                if r != RSP {
                    c.gpr[r] = base + 8 * page;
                }
            }
            c.gpr[RCX] = 4;
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                matches!(c.exec(&mut m), Step::Illegal)
            }));
            let (bad, panicked) = match res {
                Ok(illegal) => (illegal, false),
                Err(_) => (true, true),
            };
            if bad {
                let e = by_mnemonic
                    .entry(mnemonic)
                    .or_insert((0, line.to_string(), false));
                e.0 += 1;
                e.2 |= panicked;
            }
        }
        let _ = std::panic::take_hook();
        let above = |mn: &str| {
            mn.starts_with('v') || mn.starts_with("kmov") || mn.starts_with('k') && mn.len() > 4
        };
        let mut n_base = 0;
        for (mn, (n, example, panicked)) in &by_mnemonic {
            if above(mn) {
                continue;
            }
            n_base += 1;
            let p = if *panicked { " PANIC" } else { "" };
            println!("{n:6} {mn:14}{p}  e.g. {example}");
        }
        let above_list: Vec<&String> = by_mnemonic.keys().filter(|m| above(m)).collect();
        println!(
            "{total} encodings scanned; {n_base} non-VEX mnemonics with #UD encodings; {} VEX/EVEX mnemonics (not advertised)",
            above_list.len()
        );
    }

    // ---- behaviours pinned by the differential tester / the SDM ----

    /// Run `code` once from a fresh CPU prepared by `setup`.
    fn run_with(
        mem: &mut GuestMemory,
        code: &[u8],
        setup: impl FnOnce(&mut X86Interp),
    ) -> (X86Interp, Step) {
        mem.write_init(CODE, code).unwrap();
        let mut cpu = X86Interp::new(CODE, STACK);
        setup(&mut cpu);
        let s = cpu.exec(mem);
        (cpu, s)
    }

    #[test]
    fn traps_map_to_linux_signals() {
        let mut m = mem();
        for (code, sig) in [
            (&[0xCC][..], 5),        // int3 -> SIGTRAP
            (&[0xF4][..], 11),       // hlt -> SIGSEGV
            (&[0xFA][..], 11),       // cli
            (&[0xCD, 0x80][..], 11), // int 0x80
            (&[0xE4, 0x60][..], 11), // in al, 0x60
        ] {
            let (_, s) = run_with(&mut m, code, |_| {});
            assert!(
                matches!(s, Step::Trap(t) if t.signal() == sig),
                "{code:02x?}: {s:?}"
            );
        }
        // div by zero and quotient overflow -> #DE (SIGFPE)
        let (_, s) = run_with(&mut m, &[0xF7, 0xF1], |c| c.gpr[RCX] = 0);
        assert!(matches!(s, Step::Trap(Trap::Divide)));
        let (_, s) = run_with(&mut m, &[0xF7, 0xF9], |c| {
            c.gpr[RAX] = 0x8000_0000;
            c.gpr[RDX] = 0xffff_ffff;
            c.gpr[RCX] = 0xffff_ffff; // idiv ecx: INT_MIN / -1
        });
        assert!(matches!(s, Step::Trap(Trap::Divide)));
        // ud2 -> #UD
        let (_, s) = run_with(&mut m, &[0x0F, 0x0B], |_| {});
        assert!(matches!(s, Step::Illegal));
    }

    #[test]
    fn lock_is_ud_on_non_lockable_forms() {
        let mut m = mem();
        // lock add eax, ebx (register destination)
        let (_, s) = run_with(&mut m, &[0xF0, 0x01, 0xD8], |_| {});
        assert!(matches!(s, Step::Illegal));
        // lock mov [rax], ebx
        let (_, s) = run_with(&mut m, &[0xF0, 0x89, 0x18], |c| c.gpr[RAX] = 0x1_2000);
        assert!(matches!(s, Step::Illegal));
        // lock add [rax], ebx is fine
        let (_, s) = run_with(&mut m, &[0xF0, 0x01, 0x18], |c| c.gpr[RAX] = 0x1_2000);
        assert!(matches!(s, Step::Next));
    }

    #[test]
    fn bt_register_offset_addresses_the_bit_string() {
        let mut m = mem();
        let base = 0x1_2000u64;
        m.write_init(base, &[0u8; 64]).unwrap();
        m.write_init(base + 12, &[0x08]).unwrap(); // bit 99 = byte 12, bit 3
        // bt [rax], ecx with ecx = 99
        let (c, _) = run_with(&mut m, &[0x0F, 0xA3, 0x08], |c| {
            c.gpr[RAX] = base;
            c.gpr[RCX] = 99;
        });
        assert!(c.flags.cf);
        // bts [rax+16], ecx with ecx = -1 sets bit 7 of byte 15
        let (_, _) = run_with(&mut m, &[0x0F, 0xAB, 0x48, 0x10], |c| {
            c.gpr[RAX] = base;
            c.gpr[RCX] = 0xffff_ffff;
        });
        assert_eq!(m.read_vec(base + 15, 1).unwrap(), vec![0x80]);
        // the immediate form takes the offset modulo the width
        let (c, _) = run_with(&mut m, &[0x0F, 0xBA, 0x20, 99], |c| c.gpr[RAX] = base + 12);
        assert!(c.flags.cf, "99 mod 32 = bit 3 of the dword at base+12");
    }

    #[test]
    fn rotate_through_carry_and_counts() {
        let mut m = mem();
        // rcl al, 1 with CF=1: 0x80 -> 0x01, CF=1
        let (c, _) = run_with(&mut m, &[0xD0, 0xD0], |c| {
            c.gpr[RAX] = 0x80;
            c.flags.cf = true;
        });
        assert_eq!(c.gpr[RAX] & 0xff, 0x01);
        assert!(c.flags.cf);
        // rcr al, 9 is a full 9-bit ring rotation: unchanged
        let (c, _) = run_with(&mut m, &[0xC0, 0xD8, 9], |c| {
            c.gpr[RAX] = 0x5a;
            c.flags.cf = true;
        });
        assert_eq!(c.gpr[RAX] & 0xff, 0x5a);
        assert!(c.flags.cf);
        // a zero shift count still zero-extends a 32-bit register
        let (c, _) = run_with(&mut m, &[0xC1, 0xE0, 0x20], |c| {
            c.gpr[RAX] = 0xdead_beef_0000_0001;
        });
        assert_eq!(c.gpr[RAX], 1);
    }

    #[test]
    fn popf_keeps_system_flags_user_writable_only() {
        let mut m = mem();
        let (c, _) = run_with(&mut m, &[0x9D], |c| {
            c.gpr[RSP] = 0x1_3000;
            c.flags.cf = true;
        });
        let _ = c;
        m.write_init(0x1_3000, &(0x0024_4ed5u64).to_le_bytes())
            .unwrap(); // ID AC NT OF DF ... + IOPL 3
        let (c, _) = run_with(&mut m, &[0x9D, 0x9C], |c| c.gpr[RSP] = 0x1_3000);
        let w = c.rflags_word();
        assert_eq!(w & 0x8d5, 0x8d5 & 0x0024_4ed5);
        assert_ne!(w & (1 << 21), 0, "ID toggles");
        assert_ne!(w & (1 << 18), 0, "AC toggles");
        assert_eq!(w & 0x3000, 0, "IOPL is not user-writable");
        assert_ne!(w & 0x200, 0, "IF stays set");
    }

    #[test]
    fn sahf_lahf_roundtrip() {
        let mut m = mem();
        let (c, _) = run_with(&mut m, &[0x9E, 0x9F], |c| c.gpr[RAX] = 0xd500);
        assert!(c.flags.sf && c.flags.zf && c.flags.af && c.flags.pf && c.flags.cf);
        let (c, _) = run_with(&mut m, &[0x9F], |c| {
            c.flags.zf = true;
            c.gpr[RAX] = 0;
        });
        assert_eq!(c.gpr[RAX], 0x4200);
    }

    #[test]
    fn enter_with_nesting_level() {
        let mut m = mem();
        // enter 0x10, 2
        let (c, s) = run_with(&mut m, &[0xC8, 0x10, 0x00, 0x02], |c| {
            c.gpr[RSP] = 0x1_8000;
            c.gpr[RBP] = 0x1_9000;
        });
        assert!(matches!(s, Step::Next));
        assert_eq!(c.gpr[RBP], 0x1_7ff8);
        // pushes: rbp, [rbp-8], frame -> rsp = 0x18000 - 24 - 0x10
        assert_eq!(c.gpr[RSP], 0x1_8000 - 24 - 0x10);
    }

    #[test]
    fn movsxd_without_rex_w_zero_extends() {
        let mut m = mem();
        let (c, _) = run_with(&mut m, &[0x63, 0xC1], |c| c.gpr[RCX] = 0xffff_fff0);
        assert_eq!(c.gpr[RAX], 0xffff_fff0);
        let (c, _) = run_with(&mut m, &[0x48, 0x63, 0xC1], |c| c.gpr[RCX] = 0xffff_fff0);
        assert_eq!(c.gpr[RAX], 0xffff_ffff_ffff_fff0);
    }

    #[test]
    fn cmpxchg_flags_are_accumulator_minus_destination() {
        let mut m = mem();
        // cmpxchg ecx, edx with eax=1, ecx=2: mismatch, flags of 1-2 (CF/SF)
        let (c, _) = run_with(&mut m, &[0x0F, 0xB1, 0xD1], |c| {
            c.gpr[RAX] = 1;
            c.gpr[RCX] = 0xffff_ffff_0000_0002;
        });
        assert!(c.flags.cf && c.flags.sf && !c.flags.zf);
        assert_eq!(c.gpr[RAX], 2);
        assert_eq!(
            c.gpr[RCX], 2,
            "the destination is written back (zero-extended)"
        );
    }

    #[test]
    fn ldmxcsr_reserved_bits_gp_and_fxsave_fxrstor_roundtrip() {
        let mut m = mem();
        let p = 0x1_2000u64;
        m.write_init(p, &0x0001_1f80u32.to_le_bytes()).unwrap();
        let (_, s) = run_with(&mut m, &[0x0F, 0xAE, 0x10], |c| c.gpr[RAX] = p);
        assert!(matches!(s, Step::Trap(Trap::Protection)));
        // fxsave [rax]; then fxrstor it into a fresh CPU
        let (c, s) = run_with(&mut m, &[0x0F, 0xAE, 0x00], |c| {
            c.gpr[RAX] = p;
            c.xmm[3] = 0x1234_5678_9abc_def0;
            c.mxcsr = 0x3f80;
            c.fpu_cw = 0x027f;
        });
        assert!(matches!(s, Step::Next));
        let _ = c;
        let (c, _) = run_with(&mut m, &[0x0F, 0xAE, 0x08], |c| c.gpr[RAX] = p);
        assert_eq!(c.xmm[3], 0x1234_5678_9abc_def0);
        assert_eq!(c.mxcsr, 0x3f80);
        assert_eq!(c.fpu_cw, 0x027f);
        // misaligned fxsave is #GP
        let (_, s) = run_with(&mut m, &[0x0F, 0xAE, 0x00], |c| c.gpr[RAX] = p + 8);
        assert!(matches!(s, Step::Trap(Trap::Protection)));
    }

    #[test]
    fn hint_nops_and_prefetches_execute() {
        let mut m = mem();
        for code in [
            &[0x0F, 0x18, 0x08][..],             // prefetcht0 [rax]
            &[0x0F, 0x0D, 0x08][..],             // prefetchw [rax]
            &[0x0F, 0x1F, 0x44, 0x00, 0x00][..], // nop dword [rax+rax]
            &[0xF3, 0x0F, 0x1E, 0xFA][..],       // endbr64
            &[0xF3, 0x90][..],                   // pause
        ] {
            let (c, s) = run_with(&mut m, code, |c| c.gpr[RAX] = 0x1_2000);
            assert!(matches!(s, Step::Next), "{code:02x?}");
            assert_eq!(c.rip, CODE + code.len() as u64);
        }
    }

    #[test]
    fn x87_stack_faults_and_tags() {
        let mut m = mem();
        // fadd st0, st1 on an empty stack: IE|SF, C1=0, ST0 = indefinite
        let (c, _) = run_with(&mut m, &[0xD8, 0xC1], |_| {});
        assert_eq!(c.fpu_sw() & 0x241, 0x41);
        // eight fld1 then a ninth: overflow with C1=1
        let mut code = vec![];
        for _ in 0..9 {
            code.extend_from_slice(&[0xD9, 0xE8]);
        }
        m.write_init(CODE, &code).unwrap();
        let mut c = X86Interp::new(CODE, STACK);
        for _ in 0..9 {
            c.exec(&mut m);
        }
        assert_eq!(c.fpu_sw() & 0x241, 0x241);
        assert_eq!(c.st_get(0), F80::INDEFINITE);
        // fxam on an empty register: C3=1, C0=1
        let (c, _) = run_with(&mut m, &[0xD9, 0xE5], |_| {});
        assert_eq!(c.fpu_sw() & 0x4700, 0x4100);
    }

    #[test]
    fn x87_precision_control_rounds_to_24_bits() {
        let mut m = mem();
        // fld1; fldpi... with PC=00 (24 bits): fdiv st0, st1 rounds the
        // significand to 24 bits.
        m.write_init(CODE, &[0xD9, 0xE8, 0xD9, 0xEB, 0xD8, 0xF1])
            .unwrap();
        let mut c = X86Interp::new(CODE, STACK);
        c.fpu_cw = 0x007f;
        for _ in 0..3 {
            c.exec(&mut m);
        }
        // pi rounded to 24 bits: 0xC90FDB << 40
        assert_eq!(c.st_get(0).0, 0x4000_c90f_db00_0000_0000);
    }

    #[test]
    fn fsin_uses_the_hardware_66_bit_pi() {
        // sin(π₈₀) — the 80-bit π — is the reduction residue against π₆₆.
        let mut m = mem();
        m.write_init(CODE, &[0xD9, 0xEB, 0xD9, 0xFE]).unwrap();
        let mut c = X86Interp::new(CODE, STACK);
        c.exec(&mut m);
        c.exec(&mut m);
        // π₈₀ = …C235 overshoots π₆₆ = …C234.C by a quarter unit (2^-64):
        // sin(π₆₆ + 2^-64) = -2^-64 exactly as the hardware reports it.
        let v = c.st_get(0);
        assert!(v.sign());
        assert_eq!(v.exp_field(), 0x3fff - 64);
        assert_eq!(v.mant(), 1 << 63);
    }

    #[test]
    fn mmx_aliases_the_x87_registers() {
        let mut m = mem();
        // movd mm1, eax: TOP=0, all tags valid, R1 = 0xffff:value
        let (c, _) = run_with(&mut m, &[0x0F, 0x6E, 0xC8], |c| {
            c.gpr[RAX] = 0x1234_5678;
            c.fpu_top = 5;
        });
        assert_eq!(c.fpu_top, 0);
        assert_eq!(c.fpu_tag, 0xff);
        assert_eq!(c.st[1].0, (0xffffu128 << 64) | 0x1234_5678);
        // emms empties the tags
        let (c, _) = run_with(&mut m, &[0x0F, 0x77], |c| c.fpu_tag = 0xff);
        assert_eq!(c.fpu_tag, 0);
    }

    #[test]
    fn sse_unmasked_exception_raises_xm_without_writing() {
        let mut m = mem();
        // divss xmm0, xmm1 with ZM unmasked and xmm1 = 0
        let (c, s) = run_with(&mut m, &[0xF3, 0x0F, 0x5E, 0xC1], |c| {
            c.xmm[0] = u128::from(1.0f32.to_bits());
            c.xmm[1] = 0;
            c.mxcsr = 0x1f80 & !(1 << 9);
        });
        assert!(matches!(s, Step::Trap(Trap::Simd)));
        assert_eq!(c.xmm[0], u128::from(1.0f32.to_bits()));
        assert_ne!(c.mxcsr & 4, 0, "ZE is recorded");
    }

    #[test]
    fn sse_misaligned_m128_is_gp_but_movups_is_fine() {
        let mut m = mem();
        let (_, s) = run_with(&mut m, &[0x0F, 0x58, 0x00], |c| c.gpr[RAX] = 0x1_2008); // addps
        assert!(matches!(s, Step::Trap(Trap::Protection)));
        let (_, s) = run_with(&mut m, &[0x0F, 0x10, 0x00], |c| c.gpr[RAX] = 0x1_2008); // movups
        assert!(matches!(s, Step::Next));
        let (_, s) = run_with(&mut m, &[0xF3, 0x0F, 0x58, 0x00], |c| c.gpr[RAX] = 0x1_2002); // addss m32
        assert!(matches!(s, Step::Next));
    }

    #[test]
    fn self_modifying_code_is_seen_within_a_run() {
        // mov byte [rip+0], 0x90 rewrites the following int3 into a nop
        // before it executes; the code-page cache must not serve the stale
        // byte. Then syscall ends the run.
        let mut m = mem();
        m.write_init(CODE, &[0xC6, 0x05, 0, 0, 0, 0, 0x90, 0xCC, 0x0F, 0x05])
            .unwrap();
        let mut c = X86Interp::new(CODE, STACK);
        c.quantum = None;
        assert_eq!(c.run(&mut m).unwrap(), Exit::Syscall);
        assert_eq!(c.rip, CODE + 8);
    }

    #[test]
    fn crc32c_and_pcmpistri() {
        let mut m = mem();
        // crc32 eax, byte ptr [rbx] over "123456789" = 0xE3069283 (CRC-32C).
        let p = 0x1_2000u64;
        m.write_init(p, b"123456789").unwrap();
        let mut code = Vec::new();
        for _ in 0..9 {
            code.extend_from_slice(&[0xF2, 0x0F, 0x38, 0xF0, 0x03, 0x48, 0xFF, 0xC3]); // crc32 eax,[rbx]; inc rbx
        }
        m.write_init(CODE, &code).unwrap();
        let mut c = X86Interp::new(CODE, STACK);
        c.gpr[RAX] = 0xffff_ffff;
        c.gpr[RBX] = p;
        for _ in 0..18 {
            c.exec(&mut m);
        }
        assert_eq!(c.gpr[RAX] as u32 ^ 0xffff_ffff, 0xE306_9283);
        // pcmpistri xmm0, xmm1, 0x0C (unsigned bytes, equal ordered): find
        // "lo" in "hello world".
        let mut needle = [0u8; 16];
        needle[..2].copy_from_slice(b"lo");
        let mut hay = [0u8; 16];
        hay[..11].copy_from_slice(b"hello world");
        let (c, _) = run_with(&mut m, &[0x66, 0x0F, 0x3A, 0x63, 0xC1, 0x0C], |c| {
            c.xmm[0] = u128::from_le_bytes(needle);
            c.xmm[1] = u128::from_le_bytes(hay);
        });
        assert_eq!(c.gpr[RCX], 3);
        assert!(c.flags.cf && c.flags.zf && c.flags.sf);
    }

    #[test]
    fn rcpps_matches_the_hardware_table() {
        let mut m = mem();
        let (c, _) = run_with(&mut m, &[0x0F, 0x53, 0xC1], |c| {
            c.xmm[1] = u128::from(1.0f32.to_bits()) | (u128::from(2.0f32.to_bits()) << 32);
        });
        assert_eq!(c.xmm[0] as u32, 0x3f7f_f000);
        assert_eq!((c.xmm[0] >> 32) as u32, 0x3eff_f000);
    }
}
