//! Software CPU interpreter backend — the portable, no-acceleration fallback.
//!
//! Decodes and executes guest instructions against [`GuestMemory`] in a loop,
//! returning [`Exit::Syscall`] when it decodes a syscall instruction (`svc #0`
//! on arm64). Slower than the hardware backends but runs anywhere and on any
//! guest arch — this is the path the browser (wasm) demo uses, and it makes the
//! syscall engine testable in CI with no hypervisor.
//!
//! The aarch64 interpreter implements the complete Armv8.0-A A64 instruction
//! set reachable from EL0 — integer, load/store (every addressing mode,
//! exclusives, acquire/release), branches and system instructions, scalar
//! floating point and Advanced SIMD — plus exactly the extensions advertised in
//! `AT_HWCAP` and the ID registers: AES + PMULL, SHA-1, SHA-256, CRC32 and the
//! LSE atomics. Anything else is UNDEFINED, as it would be on such a core, and
//! surfaces as [`Exit::IllegalInstruction`] (`SIGILL`).
//!
//! `Aarch64Interp::exec` dispatches on the top-level `op0` field (bits 28:25)
//! to one module per ARM ARM encoding group:
//!
//! * `alu` — data processing (immediate and register)
//! * `branch` — branches, exception generation and system instructions
//! * `ldst` — loads and stores
//! * `fp` — scalar floating point
//! * `simd` — Advanced SIMD (vector and scalar) and the crypto extension
//!
//! with floating-point arithmetic in `fpu` (bit-exact with the ARM ARM
//! pseudocode, including `FPCR` rounding modes, flush-to-zero, default NaN and
//! the `FPSR` exception flags) and the crypto/CRC primitives in `crypto`.
//! Correctness is pinned by `tests/aarch64_diff.rs`, which runs instructions
//! natively on an arm64 host and under this interpreter from identical random
//! state and requires identical results.

mod alu;
mod branch;
mod crypto;
mod fp;
mod fpu;
mod ldst;
mod simd;
mod simd_ext;

use crate::abi::Arch;

use super::{Backend, Exit, GuestMemory, Vcpu, VcpuError};

/// Upper bound on instructions executed per `run()` call before yielding, so a
/// runaway guest loop can't wedge the host. (Real deadlines land in Phase 9.)
const MAX_STEPS: u64 = 50_000_000;

#[derive(Debug)]
pub struct InterpBackend {
    guest: Arch,
}

impl InterpBackend {
    pub fn new(guest: Arch) -> Result<Self, VcpuError> {
        Ok(Self { guest })
    }
}

impl Backend for InterpBackend {
    fn name(&self) -> &'static str {
        "interp"
    }

    fn guest_arch(&self) -> Arch {
        self.guest
    }

    fn new_vcpu(&self, entry: u64, stack: u64) -> Result<Box<dyn Vcpu>, VcpuError> {
        match self.guest {
            Arch::Aarch64 => Ok(Box::new(Aarch64Interp::new(entry, stack))),
            Arch::X86_64 => Err(VcpuError::Backend(
                "interp x86-64 not implemented yet (ROADMAP Phase 10)".into(),
            )),
        }
    }
}

/// Outcome of executing one instruction.
enum Step {
    /// Advance to the next instruction (`pc += 4`).
    Next,
    /// Instruction already set `pc` (branch); do not auto-advance.
    Branched,
    /// `svc` — hand control to the kernel. `pc` stays on the `svc`; the kernel
    /// advances it via [`Vcpu::set_syscall_ret`].
    Syscall,
    Illegal,
    /// `BRK #imm` — a software breakpoint (`SIGTRAP`); `pc` stays on it.
    Breakpoint {
        imm: u16,
    },
    /// An alignment fault (`SIGBUS`/`BUS_ADRALN`): misaligned SP base, or an
    /// atomic/ordered access crossing its 16-byte granule.
    Misaligned {
        addr: u64,
        write: bool,
    },
    /// A load/store touched bad guest memory.
    Fault {
        addr: u64,
        write: bool,
    },
}

/// NZCV condition flags. (Four is the architectural count, not a smell.)
#[derive(Default, Clone, Copy)]
#[allow(clippy::struct_excessive_bools)]
struct Flags {
    n: bool,
    z: bool,
    c: bool,
    v: bool,
}

impl Flags {
    /// From a 4-bit `NZCV` value (N in bit 3).
    fn from_nzcv(nzcv: u32) -> Self {
        Flags {
            n: nzcv & 8 != 0,
            z: nzcv & 4 != 0,
            c: nzcv & 2 != 0,
            v: nzcv & 1 != 0,
        }
    }
    /// As a 4-bit `NZCV` value.
    fn nzcv(self) -> u32 {
        (u32::from(self.n) << 3)
            | (u32::from(self.z) << 2)
            | (u32::from(self.c) << 1)
            | u32::from(self.v)
    }
}

/// A user-mode aarch64 interpreter.
#[derive(Clone)]
struct Aarch64Interp {
    /// x0..x30. x31 is the zero register (reads 0) or SP depending on encoding.
    x: [u64; 31],
    sp: u64,
    pc: u64,
    tpidr: u64,
    /// Backing state for `CNTVCT_EL0`: incremented on every read so a guest
    /// spin loop waiting for the counter to advance terminates, even though
    /// this interpreter has no wall-clock timer to drive a real one.
    cntvct: u64,
    /// `FPCR` (control: rounding mode, `FZ`, `DN`, `AHP`) and `FPSR`
    /// (cumulative exception flags and `QC`), honoured/updated by every FP and
    /// saturating SIMD instruction.
    fpcr: u64,
    fpsr: u64,
    flags: Flags,
    /// SIMD/FP registers v0..v31 (128-bit; D/S/H/B views are the low bits).
    v: [u128; 32],
    /// Debug: print stores to this guest address (from `NIXVM_WATCH`).
    watch: Option<u64>,
    /// Local exclusive monitor for `LDXR`/`STXR` and friends: opened (on
    /// `excl_addr`) by a load-exclusive, consumed by the matching
    /// store-exclusive, and cleared by any intervening store (`note_store`),
    /// `CLREX`, or a return to the kernel (every `run()` starts with it
    /// closed, as an exception return does on real hardware). The SMP layer
    /// runs one vcpu of an address space at a time, so "nothing else ran
    /// since the load-exclusive" is exactly what a still-open monitor means.
    excl_monitor: bool,
    excl_addr: u64,
    /// `PSTATE.DIT` (data-independent timing; FEAT_DIT). Only stored — this
    /// interpreter's timing doesn't depend on data either way.
    dit: bool,
    /// Decoded-fetch cache: copies of recently executed code pages.
    fetch: FetchCache,
}

/// A tiny direct-mapped cache of guest code pages, so instruction fetch is an
/// array index instead of a page-table walk per instruction. Coherent by
/// construction: every store this vcpu performs goes through `note_store`,
/// which drops a cached copy of the page it writes; `ISB`/`IC IVAU` drop
/// everything; and each `run()` starts empty, since the kernel or another
/// vcpu may have remapped or modified memory in between.
#[derive(Clone)]
struct FetchCache {
    tags: [u64; FETCH_WAYS],
    pages: Box<[[u8; 4096]]>,
}

const FETCH_WAYS: usize = 16;
/// Tag of an empty cache way (no page is aligned like this).
const NO_PAGE: u64 = 1;

impl FetchCache {
    fn new() -> Self {
        FetchCache {
            tags: [NO_PAGE; FETCH_WAYS],
            pages: vec![[0; 4096]; FETCH_WAYS].into_boxed_slice(),
        }
    }
    #[inline]
    fn way(page: u64) -> usize {
        ((page >> 12) as usize) % FETCH_WAYS
    }
    fn clear(&mut self) {
        self.tags = [NO_PAGE; FETCH_WAYS];
    }
    /// Drop any cached copy of the pages `[addr, addr + nbytes)` touches.
    #[inline]
    fn invalidate(&mut self, addr: u64, nbytes: usize) {
        let first = addr & !0xfff;
        let last = addr.wrapping_add(nbytes as u64 - 1) & !0xfff;
        for p in [first, last] {
            let w = Self::way(p);
            if self.tags[w] == p {
                self.tags[w] = NO_PAGE;
            }
        }
    }
    /// The instruction at `pc` (4-byte aligned), filling its page on a miss.
    #[inline]
    fn fetch(&mut self, pc: u64, mem: &GuestMemory) -> Option<u32> {
        let page = pc & !0xfff;
        let w = Self::way(page);
        if self.tags[w] != page {
            if mem.read(page, &mut self.pages[w]).is_err() {
                // An unreadable page: fault (or fetch just this word).
                return mem.read_u32(pc).ok();
            }
            self.tags[w] = page;
        }
        let off = (pc & 0xfff) as usize;
        let b = &self.pages[w][off..off + 4];
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

impl Aarch64Interp {
    fn new(entry: u64, stack: u64) -> Self {
        Self {
            x: [0; 31],
            sp: stack,
            pc: entry,
            tpidr: 0,
            cntvct: 0,
            fpcr: 0,
            fpsr: 0,
            flags: Flags::default(),
            v: [0; 32],
            watch: std::env::var("NIXVM_WATCH")
                .ok()
                .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok()),
            excl_monitor: false,
            excl_addr: 0,
            dit: false,
            fetch: FetchCache::new(),
        }
    }

    /// Read a register with zero-register semantics (index 31 → 0).
    #[inline]
    fn read_x(&self, i: usize) -> u64 {
        if i == 31 { 0 } else { self.x[i] }
    }
    /// Write a register with zero-register semantics (index 31 → discard).
    #[inline]
    fn write_x(&mut self, i: usize, v: u64) {
        if i != 31 {
            self.x[i] = v;
        }
    }
    /// Read a register with stack-pointer semantics (index 31 → SP).
    #[inline]
    fn read_sp(&self, i: usize) -> u64 {
        if i == 31 { self.sp } else { self.x[i] }
    }
    /// Write a register with stack-pointer semantics (index 31 → SP).
    #[inline]
    fn write_sp(&mut self, i: usize, v: u64) {
        if i == 31 {
            self.sp = v;
        } else {
            self.x[i] = v;
        }
    }

    #[inline]
    fn branch(&mut self, offset: i64) -> Step {
        self.pc = self.pc.wrapping_add(offset as u64);
        Step::Branched
    }

    /// Every store funnels through here: it clears the exclusive monitor (a
    /// store between `LDXR` and `STXR` must make the latter fail) and reports
    /// stores overlapping the `NIXVM_WATCH` debug address.
    #[inline]
    fn note_store(&mut self, addr: u64, value: u128, nbytes: usize) {
        self.excl_monitor = false;
        self.fetch.invalidate(addr, nbytes);
        if let Some(w) = self.watch
            && w >= addr
            && w < addr + nbytes as u64
        {
            eprintln!(
                "[watch] pc={:#x} store {value:#x} ({nbytes}B) -> {addr:#x}",
                self.pc
            );
        }
    }

    /// Instruction-fetch invalidation (`ISB`, `IC IVAU`): drop every cached
    /// code page.
    #[inline]
    fn invalidate_fetch(&mut self) {
        self.fetch.clear();
    }

    #[inline]
    fn cond_holds(&self, cond: u32) -> bool {
        let f = &self.flags;
        let r = match cond >> 1 {
            0b000 => f.z,
            0b001 => f.c,
            0b010 => f.n,
            0b011 => f.v,
            0b100 => f.c && !f.z,
            0b101 => f.n == f.v,
            0b110 => f.n == f.v && !f.z,
            _ => true,
        };
        // The low bit inverts, except for AL/NV (both "always").
        if cond & 1 == 1 && cond != 0b1111 {
            !r
        } else {
            r
        }
    }

    /// Execute one instruction: dispatch on `op0` (bits 28:25).
    #[inline]
    fn exec(&mut self, instr: u32, mem: &mut GuestMemory) -> Step {
        match (instr >> 25) & 0xf {
            0b1000 | 0b1001 => self.exec_dp_imm(instr),
            0b0100 | 0b0110 | 0b1100 | 0b1110 => self.exec_ldst(instr, mem),
            0b0101 | 0b1101 => self.exec_dp_reg(instr),
            0b1010 | 0b1011 => self.exec_branch_sys(instr, mem),
            0b0111 | 0b1111 => self.exec_simd_fp(instr),
            // 0000 reserved (UDF), 0001/0011 unallocated, 0010 SVE.
            _ => Step::Illegal,
        }
    }
}

impl Vcpu for Aarch64Interp {
    fn run(&mut self, mem: &mut GuestMemory) -> Result<Exit, VcpuError> {
        // Entering from the kernel is an exception return: the local
        // exclusive monitor is cleared.
        self.excl_monitor = false;
        // Memory may have changed while the kernel (or another vcpu) ran.
        self.fetch.clear();
        // Time-based preemption (`NIXVM_QUANTUM_MS`): a syscall-free hot loop
        // still hands control back, so siblings run and pending signals
        // (`alarm`, Go's SIGURG preemption) reach it.
        let deadline = super::preempt_quantum().map(|q| crate::clock::now_monotonic() + q);
        for i in 0..MAX_STEPS {
            // The quantum and the embedder's yield deadline
            // (`Kernel::pump_for`), polled every 4096 instructions: a clock
            // read per instruction would dominate.
            if i & 4095 == 4095
                && (super::yield_due()
                    || deadline.is_some_and(|d| crate::clock::now_monotonic() >= d))
            {
                return Ok(Exit::Interrupted);
            }
            // A misaligned PC is a PC alignment fault (SIGBUS on Linux).
            let fetched = if self.pc & 3 == 0 {
                self.fetch.fetch(self.pc, mem)
            } else {
                None
            };
            let Some(instr) = fetched else {
                // A misaligned PC is a PC alignment fault (SIGBUS).
                return Ok(if self.pc & 3 == 0 {
                    Exit::MemFault {
                        addr: self.pc,
                        write: false,
                    }
                } else {
                    Exit::Misaligned {
                        addr: self.pc,
                        write: false,
                    }
                });
            };
            match self.exec(instr, mem) {
                Step::Next => self.pc = self.pc.wrapping_add(4),
                Step::Branched => {}
                Step::Syscall => return Ok(Exit::Syscall),
                Step::Illegal => return Ok(Exit::IllegalInstruction { pc: self.pc }),
                Step::Fault { addr, write } => return Ok(Exit::MemFault { addr, write }),
                Step::Breakpoint { imm } => {
                    return Ok(Exit::Breakpoint {
                        pc: self.pc,
                        code: u64::from(imm),
                    });
                }
                Step::Misaligned { addr, write } => return Ok(Exit::Misaligned { addr, write }),
            }
        }
        Ok(Exit::Interrupted)
    }

    fn syscall_nr(&self) -> u64 {
        self.x[8]
    }
    fn syscall_args(&self) -> [u64; 6] {
        [
            self.x[0], self.x[1], self.x[2], self.x[3], self.x[4], self.x[5],
        ]
    }
    fn set_syscall_ret(&mut self, value: u64) {
        self.x[0] = value;
        self.pc = self.pc.wrapping_add(4);
    }
    fn reg(&self, idx: usize) -> u64 {
        if idx < 31 { self.x[idx] } else { self.sp }
    }
    fn set_reg(&mut self, idx: usize, value: u64) {
        if idx < 31 {
            self.x[idx] = value;
        } else {
            self.sp = value;
        }
    }
    fn pc(&self) -> u64 {
        self.pc
    }
    fn set_pc(&mut self, pc: u64) {
        self.pc = pc;
    }
    fn sp(&self) -> u64 {
        self.sp
    }
    fn set_sp(&mut self, sp: u64) {
        self.sp = sp;
    }
    /// PSTATE, packed with the condition flags in their architectural bits
    /// (`N`=31, `Z`=30, `C`=29, `V`=28) — the only fields the interpreter models.
    /// Signal delivery saves this into `uc_mcontext.pstate`; `rt_sigreturn`
    /// restores it via [`Self::set_rflags`].
    fn rflags(&self) -> u64 {
        (u64::from(self.flags.nzcv()) << 28) | (u64::from(self.dit) << 24)
    }
    fn set_rflags(&mut self, value: u64) {
        self.flags = Flags::from_nzcv((value >> 28) as u32);
        self.dit = (value >> 24) & 1 == 1;
    }
    /// The FP/SIMD state in the layout of Linux's `struct fpsimd_context`
    /// minus its 8-byte header: `fpsr` (u32), `fpcr` (u32), then `vregs[32]`
    /// (16 bytes each, little-endian) — 520 bytes.
    fn simd_state(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + 32 * 16);
        out.extend_from_slice(&(self.fpsr as u32).to_le_bytes());
        out.extend_from_slice(&(self.fpcr as u32).to_le_bytes());
        for v in &self.v {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }
    fn set_simd_state(&mut self, bytes: &[u8]) {
        if bytes.len() < 8 + 32 * 16 {
            return;
        }
        let word = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        self.fpsr = u64::from(word(0)) & 0x0800_009F;
        self.fpcr = u64::from(word(4)) & 0x07C8_0000;
        for (i, v) in self.v.iter_mut().enumerate() {
            let off = 8 + i * 16;
            *v = u128::from_le_bytes(bytes[off..off + 16].try_into().unwrap());
        }
    }
    fn set_tls(&mut self, value: u64) {
        self.tpidr = value;
    }

    fn fork(&self) -> Box<dyn Vcpu> {
        Box::new(self.clone())
    }

    fn reset(&mut self, entry: u64, sp: u64) {
        self.x = [0; 31];
        self.v = [0; 32];
        self.sp = sp;
        self.pc = entry;
        self.tpidr = 0;
        self.fpcr = 0;
        self.fpsr = 0;
        self.flags = Flags::default();
        self.excl_monitor = false;
        self.dit = false;
    }
}

/// Extract a 5-bit register field starting at bit `lsb`.
#[inline]
fn reg_field(instr: u32, lsb: u32) -> usize {
    ((instr >> lsb) & 0x1f) as usize
}

/// Architectural state of an aarch64 vcpu, for the differential test harness
/// (`tests/aarch64_diff.rs`) — not part of the embedding API.
#[doc(hidden)]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct A64State {
    pub x: [u64; 31],
    pub sp: u64,
    pub pc: u64,
    /// `NZCV` in bits 31:28.
    pub nzcv: u64,
    pub fpcr: u64,
    pub fpsr: u64,
    pub tpidr: u64,
    pub v: [u128; 32],
}

/// How [`a64_step`] ended.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum A64Step {
    Next,
    Branched,
    Syscall,
    Illegal,
    Fault { addr: u64, write: bool },
}

/// Execute one instruction word on `state` (test harness hook): `pc` advances
/// by 4 unless the instruction branched.
#[doc(hidden)]
pub fn a64_step(state: &mut A64State, instr: u32, mem: &mut GuestMemory) -> A64Step {
    let mut c = Aarch64Interp::new(state.pc, state.sp);
    c.x = state.x;
    c.fpcr = state.fpcr;
    c.fpsr = state.fpsr;
    c.tpidr = state.tpidr;
    c.flags = Flags::from_nzcv((state.nzcv >> 28) as u32);
    c.v = state.v;
    let r = match c.exec(instr, mem) {
        Step::Next => {
            c.pc = c.pc.wrapping_add(4);
            A64Step::Next
        }
        Step::Branched => A64Step::Branched,
        Step::Syscall => A64Step::Syscall,
        // The harness compares against native signals only coarsely: a
        // breakpoint is "not executed", an alignment fault is a fault.
        Step::Illegal | Step::Breakpoint { .. } => A64Step::Illegal,
        Step::Fault { addr, write } | Step::Misaligned { addr, write } => {
            A64Step::Fault { addr, write }
        }
    };
    state.x = c.x;
    state.sp = c.sp;
    state.pc = c.pc;
    state.nzcv = u64::from(c.flags.nzcv()) << 28;
    state.fpcr = c.fpcr;
    state.fpsr = c.fpsr;
    state.tpidr = c.tpidr;
    state.v = c.v;
    r
}

#[cfg(test)]
mod tests {
    // FP tests compare exactly-representable results (integers, halves) by value.
    #![allow(clippy::float_cmp)]
    use super::crypto::{AES_INV_SBOX, AES_SBOX, pack_u32_lanes, u32_lanes};
    use super::*;
    use crate::vcpu::mem::{PAGE_SIZE, Prot};

    const CTR_EL0_VAL: u64 = 0x8444_C004;
    const DC_ZVA_BLOCK_BYTES: u64 = 64;

    fn cpu() -> Aarch64Interp {
        Aarch64Interp::new(0x1_0000, 0x2_0000)
    }
    /// A scratch memory for instructions that don't touch it.
    fn scratch() -> GuestMemory {
        GuestMemory::new(0x1_0000, PAGE_SIZE)
    }

    #[test]
    fn fmov_ins_sshll() {
        let (mut c, mut m) = (cpu(), scratch());
        // fmov s0, w1  (GP -> FP low 32); fmov w3, s0 (back)
        c.x[1] = 0x1234_5678;
        c.exec(0x1E27_0020, &mut m);
        assert_eq!(c.v[0], 0x1234_5678);
        c.exec(0x1E26_0003, &mut m);
        assert_eq!(c.x[3], 0x1234_5678);
        // mov v0.s[1], w2  (insert GP into element 1)
        c.v[0] = 0;
        c.x[2] = 0xAABB;
        c.exec(0x4E0C_1C40, &mut m);
        assert_eq!(c.v[0], 0xAABB_u128 << 32);
        // sshll v0.2d, v0.2s, #0  ([-1, 2] -> [-1, 2] widened & sign-extended)
        c.v[0] = (2u128 << 32) | 0xFFFF_FFFF;
        c.exec(0x0F20_A400, &mut m);
        assert_eq!(c.v[0], (2u128 << 64) | u128::from(u64::MAX));
    }

    #[test]
    fn simd_modified_immediate_movi_mvni() {
        let (mut c, mut m) = (cpu(), scratch());
        c.exec(0x4F00_0400, &mut m); // movi v0.4s, #0
        assert_eq!(c.v[0], 0);
        c.v[3] = 0xdead; // must not be treated as an LDP/STP pair
        c.exec(0x2F00_0403, &mut m); // mvni v3.2s, #0  -> low 64 bits all ones
        assert_eq!(c.v[3], 0xFFFF_FFFF_FFFF_FFFF);
    }

    #[test]
    fn movz_movk_build_64bit_immediate() {
        let (mut c, mut m) = (cpu(), scratch());
        assert!(matches!(c.exec(0xD282_0001, &mut m), Step::Next)); // movz x1,#0x1000
        assert_eq!(c.x[1], 0x1000);
        assert!(matches!(c.exec(0xF2A0_0021, &mut m), Step::Next)); // movk x1,#1,lsl#16
        assert_eq!(c.x[1], 0x1_1000);
    }

    #[test]
    fn add_sub_immediate() {
        let (mut c, mut m) = (cpu(), scratch());
        c.x[0] = 100;
        c.exec(0x9100_1401, &mut m); // add x1,x0,#5
        assert_eq!(c.x[1], 105);
        c.exec(0xD100_2802, &mut m); // sub x2,x0,#10
        assert_eq!(c.x[2], 90);
    }

    #[test]
    fn add_extended_register() {
        let (mut c, mut m) = (cpu(), scratch());
        c.x[1] = 0x1000;
        c.x[2] = 0x1FF;
        c.exec(0x8B22_0020, &mut m); // add x0,x1,w2,uxtb -> 0x1000 + 0xFF
        assert_eq!(c.x[0], 0x10FF);
    }

    #[test]
    fn add_shifted_register() {
        let (mut c, mut m) = (cpu(), scratch());
        c.x[0] = 10;
        c.x[1] = 20;
        c.exec(0x8B01_0002, &mut m); // add x2,x0,x1
        assert_eq!(c.x[2], 30);
    }

    #[test]
    fn cmp_sets_flags_for_branch() {
        let (mut c, mut m) = (cpu(), scratch());
        c.x[1] = 6;
        c.exec(0xF100_183F, &mut m); // cmp x1,#6  (subs xzr,x1,#6)
        assert!(c.flags.z, "6 == 6 sets Z");
        assert!(c.cond_holds(0b0000), "EQ holds");
        assert!(!c.cond_holds(0b0001), "NE does not hold");
    }

    #[test]
    fn bitfield_shifts_and_extends() {
        let (mut c, mut m) = (cpu(), scratch());
        c.x[1] = 0x1234;
        c.exec(0xD37C_EC20, &mut m); // lsl x0,x1,#4
        assert_eq!(c.x[0], 0x1234 << 4);
        c.exec(0xD344_FC20, &mut m); // lsr x0,x1,#4
        assert_eq!(c.x[0], 0x1234 >> 4);

        c.x[1] = (-16i64) as u64;
        c.exec(0x9344_FC20, &mut m); // asr x0,x1,#4
        assert_eq!(c.x[0] as i64, -1);

        c.x[1] = 0x1234_5678_9abc_def0;
        c.exec(0x5300_1C20, &mut m); // uxtb w0,w1  -> 0xf0
        assert_eq!(c.x[0], 0xf0);
        c.x[1] = 0x80; // high bit of the byte set
        c.exec(0x9340_1C20, &mut m); // sxtb x0,x1  -> sign-extended
        assert_eq!(c.x[0] as i64, -128);
    }

    #[test]
    fn logical_immediate() {
        let (mut c, mut m) = (cpu(), scratch());
        c.x[1] = 0x1_2345;
        c.exec(0x9240_1C20, &mut m); // and x0,x1,#0xff
        assert_eq!(c.x[0], 0x45);
    }

    #[test]
    fn mul_and_madd() {
        let (mut c, mut m) = (cpu(), scratch());
        c.x[1] = 6;
        c.x[2] = 7;
        c.exec(0x9B02_7C20, &mut m); // mul x0,x1,x2
        assert_eq!(c.x[0], 42);
        c.x[3] = 1;
        c.exec(0x9B02_0C20, &mut m); // madd x0,x1,x2,x3
        assert_eq!(c.x[0], 43);
    }

    #[test]
    fn udiv_sdiv_and_div_by_zero() {
        let (mut c, mut m) = (cpu(), scratch());
        c.x[1] = 100;
        c.x[2] = 7;
        c.exec(0x9AC2_0820, &mut m); // udiv x0,x1,x2
        assert_eq!(c.x[0], 14);
        c.x[1] = (-100i64) as u64;
        c.exec(0x9AC2_0C20, &mut m); // sdiv x0,x1,x2
        assert_eq!(c.x[0] as i64, -14);
        c.x[2] = 0;
        c.exec(0x9AC2_0820, &mut m); // udiv by zero -> 0
        assert_eq!(c.x[0], 0);
    }

    #[test]
    fn lslv_variable_shift() {
        let (mut c, mut m) = (cpu(), scratch());
        c.x[1] = 1;
        c.x[2] = 4;
        c.exec(0x9AC2_2020, &mut m); // lslv x0,x1,x2
        assert_eq!(c.x[0], 16);
    }

    #[test]
    fn csel_and_csinc_use_flags() {
        let (mut c, mut m) = (cpu(), scratch());
        c.x[1] = 111;
        c.x[2] = 222;
        c.flags.z = true; // EQ holds
        c.exec(0x9A82_0020, &mut m); // csel x0,x1,x2,eq -> x1
        assert_eq!(c.x[0], 111);
        c.flags.z = false; // EQ fails
        c.exec(0x9A82_0020, &mut m); // csel -> x2
        assert_eq!(c.x[0], 222);
        // csinc x0,x1,x2,eq with EQ false -> x2 + 1
        c.exec(0x9A82_0420, &mut m);
        assert_eq!(c.x[0], 223);
    }

    #[test]
    fn mov_via_orr() {
        let (mut c, mut m) = (cpu(), scratch());
        c.x[5] = 0xabcd;
        c.exec(0xAA05_03E0, &mut m); // mov x0,x5  (orr x0,xzr,x5)
        assert_eq!(c.x[0], 0xabcd);
    }

    #[test]
    fn ldr_str_roundtrip() {
        let mut c = cpu();
        let mut m = GuestMemory::new(0x1_0000, 4 * PAGE_SIZE);
        m.map(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        c.x[1] = 0x1_0040; // base address (mapped)
        c.x[0] = 0x1122_3344_5566_7788;
        assert!(matches!(c.exec(0xF900_0020, &mut m), Step::Next)); // str x0,[x1]
        c.x[0] = 0;
        assert!(matches!(c.exec(0xF940_0022, &mut m), Step::Next)); // ldr x2,[x1]
        assert_eq!(c.x[2], 0x1122_3344_5566_7788);
    }

    #[test]
    fn store_to_unmapped_faults() {
        let mut c = cpu();
        let mut m = GuestMemory::new(0x1_0000, PAGE_SIZE);
        c.x[1] = 0x1_0000; // not mapped
        assert!(matches!(
            c.exec(0xF900_0020, &mut m),
            Step::Fault { write: true, .. }
        ));
    }

    /// A summation loop exercises add(reg), add(imm), cmp, and b.ne.
    #[test]
    fn sum_loop_runs_control_flow() {
        let base = 0x1_0000u64;
        let program: [u32; 8] = [
            0xD280_0000, // movz x0,#0      ; sum
            0xD280_0021, // movz x1,#1      ; i
            0x8B01_0000, // add  x0,x0,x1   ; loop:
            0x9100_0421, // add  x1,x1,#1
            0xF100_183F, // cmp  x1,#6
            0x54FF_FFA1, // b.ne loop  (-12)
            0xD280_0BA8, // movz x8,#93     ; __NR_exit
            0xD400_0001, // svc
        ];
        let mut mem = GuestMemory::new(base, 4 * PAGE_SIZE);
        mem.map(base, PAGE_SIZE, Prot::rx()).unwrap();
        let mut bytes = Vec::new();
        for w in program {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        mem.write_init(base, &bytes).unwrap();

        let mut c = Aarch64Interp::new(base, base + 3 * PAGE_SIZE);
        assert_eq!(c.run(&mut mem).unwrap(), Exit::Syscall);
        assert_eq!(c.x[8], 93, "exit syscall");
        assert_eq!(c.x[0], 15, "sum of 1..=5");
    }

    /// BL saves the return address; RET restores it.
    #[test]
    fn bl_ret_calls_subroutine() {
        let base = 0x1_0000u64;
        let program: [u32; 5] = [
            0x9400_0003, // bl  +12  -> subroutine
            0xD280_0BA8, // movz x8,#93
            0xD400_0001, // svc
            0xD280_00E0, // movz x0,#7   ; subroutine
            0xD65F_03C0, // ret
        ];
        let mut mem = GuestMemory::new(base, 4 * PAGE_SIZE);
        mem.map(base, PAGE_SIZE, Prot::rx()).unwrap();
        let mut bytes = Vec::new();
        for w in program {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        mem.write_init(base, &bytes).unwrap();

        let mut c = Aarch64Interp::new(base, base + 3 * PAGE_SIZE);
        assert_eq!(c.run(&mut mem).unwrap(), Exit::Syscall);
        assert_eq!(c.x[0], 7, "subroutine set x0");
        assert_eq!(c.x[8], 93);
    }

    /// STP pre-index pushes a register pair; LDP post-index pops it and
    /// restores SP — the shape of every function prologue/epilogue.
    #[test]
    fn stp_ldp_push_pop_roundtrip() {
        let base = 0x1_0000u64;
        let mut mem = GuestMemory::new(base, 8 * PAGE_SIZE);
        mem.map(base, PAGE_SIZE, Prot::rx()).unwrap();
        mem.map(base + 4 * PAGE_SIZE, PAGE_SIZE, Prot::rw())
            .unwrap();
        let sp = base + 5 * PAGE_SIZE;

        let program: [u32; 7] = [
            0xD282_4680, // movz x0,#0x1234
            0xD28A_CF01, // movz x1,#0x5678
            0xA9BF_07E0, // stp x0,x1,[sp,#-16]!
            0xD280_0000, // movz x0,#0    (clobber)
            0xD280_0001, // movz x1,#0
            0xA8C1_07E0, // ldp x0,x1,[sp],#16
            0xD400_0001, // svc
        ];
        let mut bytes = Vec::new();
        for w in program {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        mem.write_init(base, &bytes).unwrap();

        let mut c = Aarch64Interp::new(base, sp);
        assert_eq!(c.run(&mut mem).unwrap(), Exit::Syscall);
        assert_eq!(c.x[0], 0x1234, "x0 restored from stack");
        assert_eq!(c.x[1], 0x5678, "x1 restored from stack");
        assert_eq!(c.sp, sp, "sp restored to its original value");
    }

    #[test]
    fn indexed_and_register_offset_load_store() {
        let base = 0x1_0000u64;
        let mut m = GuestMemory::new(base, 4 * PAGE_SIZE);
        m.map(base, PAGE_SIZE, Prot::rw()).unwrap();
        let mut c = cpu();

        // str x0,[x1,#-8]!  (pre-index, writes back x1)
        c.x[1] = base + 0x100;
        c.x[0] = 0xAABB_CCDD;
        assert!(matches!(c.exec(0xF81F_8C20, &mut m), Step::Next));
        assert_eq!(c.x[1], base + 0xF8, "pre-index writeback");

        // ldur x2,[x1]  (unscaled offset 0)
        assert!(matches!(c.exec(0xF840_0022, &mut m), Step::Next));
        assert_eq!(c.x[2], 0xAABB_CCDD);

        // ldr x3,[x5,x6]  (register offset)
        c.x[5] = base;
        c.x[6] = 0xF8;
        assert!(matches!(c.exec(0xF866_68A3, &mut m), Step::Next));
        assert_eq!(c.x[3], 0xAABB_CCDD);
    }

    #[test]
    fn ldrsb_sign_extends() {
        let base = 0x1_0000u64;
        let mut m = GuestMemory::new(base, 4 * PAGE_SIZE);
        m.map(base, PAGE_SIZE, Prot::rw()).unwrap();
        let mut c = cpu();
        c.x[1] = base + 0x200;
        c.x[0] = 0x80;
        c.exec(0x3900_0020, &mut m); // strb w0,[x1]
        c.exec(0x3880_0022, &mut m); // ldrsb x2,[x1]
        assert_eq!(c.x[2] as i64, -128, "signed byte load sign-extends");
    }

    #[test]
    fn exclusive_store_load_roundtrip() {
        let base = 0x1_0000u64;
        let mut m = GuestMemory::new(base, 4 * PAGE_SIZE);
        m.map(base, PAGE_SIZE, Prot::rw()).unwrap();
        let mut c = cpu();
        c.x[1] = base + 0x40;
        c.x[3] = 0x42;
        // stxr w2,x3,[x1] with no open monitor fails (status 1), no store
        assert!(matches!(c.exec(0xC802_7C23, &mut m), Step::Next));
        assert_eq!(c.x[2], 1, "store-exclusive without a monitor fails");
        assert_eq!(m.read_u64(base + 0x40).unwrap(), 0);
        // ldxr x0,[x1] ; stxr w2,x3,[x1] succeeds (status 0)
        assert!(matches!(c.exec(0xC85F_7C20, &mut m), Step::Next));
        assert_eq!(c.x[0], 0);
        assert!(matches!(c.exec(0xC802_7C23, &mut m), Step::Next));
        assert_eq!(c.x[2], 0, "store-exclusive after load-exclusive succeeds");
        assert_eq!(m.read_u64(base + 0x40).unwrap(), 0x42);
    }

    #[test]
    fn ldxr_stxr_sequence_reports_success() {
        // The real usage order (LDXR opens the monitor, then STXR consumes
        // it), unlike `exclusive_store_load_roundtrip` above which only
        // exercises the always-succeeds default.
        let base = 0x1_0000u64;
        let mut m = GuestMemory::new(base, 4 * PAGE_SIZE);
        m.map(base, PAGE_SIZE, Prot::rw()).unwrap();
        let mut c = cpu();
        let addr = base + 0x40;
        m.write(addr, &0x42u64.to_le_bytes()).unwrap();
        c.x[1] = addr;
        // ldxr x0,[x1]
        assert!(matches!(c.exec(0xC85F_7C20, &mut m), Step::Next));
        assert_eq!(c.x[0], 0x42);
        c.x[3] = 0x99;
        // stxr w2,x3,[x1] — monitor is open, so this succeeds (status 0).
        assert!(matches!(c.exec(0xC802_7C23, &mut m), Step::Next));
        assert_eq!(c.x[2], 0, "STXR reports success while the monitor is open");
        let mut buf = [0u8; 8];
        m.read(addr, &mut buf).unwrap();
        assert_eq!(u64::from_le_bytes(buf), 0x99, "STXR's store took effect");
    }

    #[test]
    fn intervening_store_clears_exclusive_monitor() {
        let base = 0x1_0000u64;
        let mut m = GuestMemory::new(base, 4 * PAGE_SIZE);
        m.map(base, PAGE_SIZE, Prot::rw()).unwrap();
        let mut c = cpu();
        let addr = base + 0x40;
        m.write(addr, &0x42u64.to_le_bytes()).unwrap();
        c.x[1] = addr;
        assert!(matches!(c.exec(0xC85F_7C20, &mut m), Step::Next)); // ldxr x0,[x1]
        c.x[4] = 0x1234;
        assert!(matches!(c.exec(0xB900_0024, &mut m), Step::Next)); // str w4,[x1] (unrelated store)
        c.x[3] = 0x99;
        // stxr w2,x3,[x1] — monitor was cleared by the plain store above.
        assert!(matches!(c.exec(0xC802_7C23, &mut m), Step::Next));
        assert_eq!(
            c.x[2], 1,
            "STXR reports failure once the monitor is cleared"
        );
        let mut buf = [0u8; 8];
        m.read(addr, &mut buf).unwrap();
        assert_eq!(
            u64::from_le_bytes(buf) & 0xffff_ffff,
            0x1234,
            "failed STXR must not have written memory"
        );
    }

    #[test]
    fn cas_success_and_failure_paths() {
        let base = 0x1_0000u64;
        let mut m = GuestMemory::new(base, 4 * PAGE_SIZE);
        m.map(base, PAGE_SIZE, Prot::rw()).unwrap();
        let mut c = cpu();
        let addr = base + 0x300;
        m.write(addr, &0x1111_1111u32.to_le_bytes()).unwrap();
        c.x[0] = addr;

        // cas w1,w2,[x0] with a mismatching compare value: no swap, but the
        // original memory value is still returned in w1.
        c.x[1] = 0xdead_beef;
        c.x[2] = 0x2222_2222;
        assert!(matches!(c.exec(0x88a1_7c02, &mut m), Step::Next));
        assert_eq!(
            c.x[1], 0x1111_1111,
            "CAS returns the original value even on mismatch"
        );
        let mut buf = [0u8; 4];
        m.read(addr, &mut buf).unwrap();
        assert_eq!(
            u32::from_le_bytes(buf),
            0x1111_1111,
            "no swap on a failed compare"
        );

        // cas w1,w2,[x0] with a matching compare value: swap happens.
        c.x[1] = 0x1111_1111;
        c.x[2] = 0x2222_2222;
        assert!(matches!(c.exec(0x88a1_7c02, &mut m), Step::Next));
        assert_eq!(c.x[1], 0x1111_1111, "CAS still returns the pre-swap value");
        m.read(addr, &mut buf).unwrap();
        assert_eq!(
            u32::from_le_bytes(buf),
            0x2222_2222,
            "swap happens on a matching compare"
        );
    }

    #[test]
    fn swp_round_trip() {
        let base = 0x1_0000u64;
        let mut m = GuestMemory::new(base, 4 * PAGE_SIZE);
        m.map(base, PAGE_SIZE, Prot::rw()).unwrap();
        let mut c = cpu();
        let addr = base + 0x300;
        m.write(addr, &0x1234_5678u32.to_le_bytes()).unwrap();
        c.x[0] = addr;
        c.x[1] = 0xAAAA_BBBB; // new value (Ws)
        // swp w1,w2,[x0]
        assert!(matches!(c.exec(0xb821_8002, &mut m), Step::Next));
        assert_eq!(c.x[2], 0x1234_5678, "SWP returns the original value");
        let mut buf = [0u8; 4];
        m.read(addr, &mut buf).unwrap();
        assert_eq!(
            u32::from_le_bytes(buf),
            0xAAAA_BBBB,
            "SWP stores the new value"
        );
    }

    #[test]
    fn ldadd_ldset_ldclr_return_old_value_and_update_memory() {
        let base = 0x1_0000u64;
        let mut m = GuestMemory::new(base, 4 * PAGE_SIZE);
        m.map(base, PAGE_SIZE, Prot::rw()).unwrap();
        let mut c = cpu();
        let addr = base + 0x300;
        m.write(addr, &0x0000_000fu32.to_le_bytes()).unwrap();
        c.x[0] = addr;
        let mut buf = [0u8; 4];

        // ldadd w1,w2,[x0]: w2 = old; mem = old + w1.
        c.x[1] = 0x10;
        assert!(matches!(c.exec(0xb821_0002, &mut m), Step::Next));
        assert_eq!(c.x[2], 0x0f, "LDADD returns the original value");
        m.read(addr, &mut buf).unwrap();
        assert_eq!(
            u32::from_le_bytes(buf),
            0x1f,
            "LDADD writes old+rs back to memory"
        );

        // ldset w1,w2,[x0]: w2 = old; mem = old | w1.
        c.x[1] = 0xf0;
        assert!(matches!(c.exec(0xb821_3002, &mut m), Step::Next));
        assert_eq!(c.x[2], 0x1f, "LDSET returns the original value");
        m.read(addr, &mut buf).unwrap();
        assert_eq!(
            u32::from_le_bytes(buf),
            0xff,
            "LDSET writes old|rs back to memory"
        );

        // ldclr w1,w2,[x0]: w2 = old; mem = old & !w1.
        c.x[1] = 0x0f;
        assert!(matches!(c.exec(0xb821_1002, &mut m), Step::Next));
        assert_eq!(c.x[2], 0xff, "LDCLR returns the original value");
        m.read(addr, &mut buf).unwrap();
        assert_eq!(
            u32::from_le_bytes(buf),
            0xf0,
            "LDCLR writes old&!rs back to memory"
        );
    }

    #[test]
    fn stadd_updates_memory_without_register_writeback() {
        let base = 0x1_0000u64;
        let mut m = GuestMemory::new(base, 4 * PAGE_SIZE);
        m.map(base, PAGE_SIZE, Prot::rw()).unwrap();
        let mut c = cpu();
        let addr = base + 0x300;
        m.write(addr, &0x5u32.to_le_bytes()).unwrap();
        c.x[0] = addr;
        c.x[1] = 0x3;
        // stadd w1,[x0] — same encoding as LDADD with Rt == 31 (no result).
        assert!(matches!(c.exec(0xb821_001f, &mut m), Step::Next));
        let mut buf = [0u8; 4];
        m.read(addr, &mut buf).unwrap();
        assert_eq!(u32::from_le_bytes(buf), 0x8, "STADD still updates memory");
    }

    #[test]
    fn msr_mrs_tpidr_roundtrip() {
        let (mut c, mut m) = (cpu(), scratch());
        c.x[0] = 0x1234_5678;
        c.exec(0xD51B_D040, &mut m); // msr tpidr_el0, x0
        assert_eq!(c.tpidr, 0x1234_5678);
        c.exec(0xD53B_D041, &mut m); // mrs x1, tpidr_el0
        assert_eq!(c.x[1], 0x1234_5678);
    }

    #[test]
    fn tbz_tests_a_bit() {
        let (mut c, mut m) = (cpu(), scratch());
        c.pc = 0x1000;
        c.x[0] = 0; // bit 3 clear -> TBZ taken
        assert!(matches!(c.exec(0x3618_0040, &mut m), Step::Branched));
        assert_eq!(c.pc, 0x1008);
        c.pc = 0x1000;
        c.x[0] = 8; // bit 3 set -> TBZ not taken
        assert!(matches!(c.exec(0x3618_0040, &mut m), Step::Next));
    }

    #[test]
    fn adc_sbc_use_carry() {
        let (mut c, mut m) = (cpu(), scratch());
        c.x[0] = 10;
        c.x[1] = 3;
        c.flags.c = true;
        c.exec(0x9A01_0002, &mut m); // adc x2,x0,x1 -> 10+3+1
        assert_eq!(c.x[2], 14);
        c.flags.c = false;
        c.exec(0xDA01_0002, &mut m); // sbc x2,x0,x1 -> 10-3-1
        assert_eq!(c.x[2], 6);
    }

    #[test]
    fn shl_scalar_and_vector_mov() {
        let (mut c, mut m) = (cpu(), scratch());
        c.v[1] = 0xff;
        c.exec(0x5F74_5421, &mut m); // shl d1,d1,#52
        assert_eq!(c.v[1], 0xff << 52);
        c.v[0] = 0x1234_5678_9abc_def0;
        c.exec(0x4EA0_1C02, &mut m); // mov v2.16b, v0.16b (orr)
        assert_eq!(c.v[2], c.v[0]);
    }

    #[test]
    fn svc_traps_without_advancing() {
        let (mut c, mut m) = (cpu(), scratch());
        c.pc = 0x1_0004;
        assert!(matches!(c.exec(0xD400_0001, &mut m), Step::Syscall));
        assert_eq!(c.pc, 0x1_0004);
        c.set_syscall_ret(0);
        assert_eq!(c.pc, 0x1_0008);
    }

    /// Load a scalar `f64` into `v[n]`'s low 64 bits.
    fn setd(c: &mut Aarch64Interp, n: usize, x: f64) {
        c.v[n] = u128::from(x.to_bits());
    }
    /// Load a scalar `f32` into `v[n]`'s low 32 bits.
    fn sets(c: &mut Aarch64Interp, n: usize, x: f32) {
        c.v[n] = u128::from(x.to_bits());
    }
    fn getd(c: &Aarch64Interp, n: usize) -> f64 {
        f64::from_bits(c.v[n] as u64)
    }
    fn gets(c: &Aarch64Interp, n: usize) -> f32 {
        f32::from_bits(c.v[n] as u32)
    }

    #[test]
    fn fp_arithmetic_double() {
        let (mut c, mut m) = (cpu(), scratch());
        setd(&mut c, 1, 3.5);
        setd(&mut c, 2, 2.0);
        c.v[0] = u128::MAX; // ensure upper bits get cleared
        c.exec(0x1E62_2820, &mut m); // fadd d0,d1,d2
        assert_eq!(getd(&c, 0), 5.5);
        assert_eq!(c.v[0] >> 64, 0, "upper bits cleared");
        c.exec(0x1E62_3820, &mut m); // fsub d0,d1,d2
        assert_eq!(getd(&c, 0), 1.5);
        c.exec(0x1E62_0820, &mut m); // fmul d0,d1,d2
        assert_eq!(getd(&c, 0), 7.0);
        c.exec(0x1E62_1820, &mut m); // fdiv d0,d1,d2
        assert_eq!(getd(&c, 0), 1.75);
        c.exec(0x1E62_4820, &mut m); // fmax d0,d1,d2
        assert_eq!(getd(&c, 0), 3.5);
        c.exec(0x1E62_5820, &mut m); // fmin d0,d1,d2
        assert_eq!(getd(&c, 0), 2.0);
        c.exec(0x1E62_8820, &mut m); // fnmul d0,d1,d2 -> -(3.5*2)
        assert_eq!(getd(&c, 0), -7.0);
    }

    #[test]
    fn fp_arithmetic_single() {
        let (mut c, mut m) = (cpu(), scratch());
        sets(&mut c, 1, 1.5);
        sets(&mut c, 2, 4.0);
        c.exec(0x1E22_2820, &mut m); // fadd s0,s1,s2
        assert_eq!(gets(&c, 0), 5.5);
        c.exec(0x1E22_0820, &mut m); // fmul s0,s1,s2
        assert_eq!(gets(&c, 0), 6.0);
    }

    #[test]
    fn fp_one_source_ops() {
        let (mut c, mut m) = (cpu(), scratch());
        setd(&mut c, 1, -3.25);
        c.exec(0x1E60_C020, &mut m); // fabs d0,d1
        assert_eq!(getd(&c, 0), 3.25);
        c.exec(0x1E61_4020, &mut m); // fneg d0,d1
        assert_eq!(getd(&c, 0), 3.25);
        setd(&mut c, 1, 9.0);
        c.exec(0x1E61_C020, &mut m); // fsqrt d0,d1
        assert_eq!(getd(&c, 0), 3.0);
        setd(&mut c, 1, 2.7);
        c.exec(0x1E65_C020, &mut m); // frintz d0,d1 (toward zero)
        assert_eq!(getd(&c, 0), 2.0);
        c.exec(0x1E65_4020, &mut m); // frintm d0,d1 (floor)
        assert_eq!(getd(&c, 0), 2.0);
        c.exec(0x1E64_C020, &mut m); // frintp d0,d1 (ceil)
        assert_eq!(getd(&c, 0), 3.0);
        setd(&mut c, 1, 2.5);
        c.exec(0x1E64_4020, &mut m); // frintn d0,d1 (ties to even -> 2)
        assert_eq!(getd(&c, 0), 2.0);
        c.exec(0x1E66_4020, &mut m); // frinta d0,d1 (ties away -> 3)
        assert_eq!(getd(&c, 0), 3.0);
    }

    #[test]
    fn fcvt_single_double_roundtrip() {
        let (mut c, mut m) = (cpu(), scratch());
        sets(&mut c, 1, 1.5);
        c.exec(0x1E22_C020, &mut m); // fcvt d0,s1  (single -> double)
        assert_eq!(getd(&c, 0), 1.5);
        setd(&mut c, 1, 1.5);
        c.exec(0x1E62_4020, &mut m); // fcvt s0,d1  (double -> single)
        assert_eq!(gets(&c, 0), 1.5);
    }

    #[test]
    fn fmov_immediate() {
        let (mut c, mut m) = (cpu(), scratch());
        c.exec(0x1E6E_1000, &mut m); // fmov d0,#1.0
        assert_eq!(getd(&c, 0), 1.0);
        c.exec(0x1E60_1000, &mut m); // fmov d0,#2.0
        assert_eq!(getd(&c, 0), 2.0);
        c.exec(0x1E2E_1000, &mut m); // fmov s0,#1.0
        assert_eq!(gets(&c, 0), 1.0);
    }

    #[test]
    fn scvtf_fcvtzs_roundtrip_integer() {
        let (mut c, mut m) = (cpu(), scratch());
        c.x[1] = (-42i64) as u64;
        c.exec(0x9E62_0020, &mut m); // scvtf d0,x1
        assert_eq!(getd(&c, 0), -42.0);
        // round-trip back to an integer with FCVTZS
        c.exec(0x9E78_0000, &mut m); // fcvtzs x0,d0
        assert_eq!(c.x[0] as i64, -42);
        // unsigned conversion round-trip
        c.x[3] = 300;
        c.exec(0x9E63_0060, &mut m); // ucvtf d0,x3
        assert_eq!(getd(&c, 0), 300.0);
        setd(&mut c, 0, 300.0);
        c.exec(0x9E79_0000, &mut m); // fcvtzu x0,d0
        assert_eq!(c.x[0], 300);
        // saturation: NaN -> 0, +inf -> i64::MAX
        setd(&mut c, 0, f64::NAN);
        c.exec(0x9E78_0000, &mut m); // fcvtzs x0,d0
        assert_eq!(c.x[0], 0, "NaN converts to 0");
        setd(&mut c, 0, f64::INFINITY);
        c.exec(0x9E78_0000, &mut m); // fcvtzs x0,d0
        assert_eq!(c.x[0] as i64, i64::MAX, "+inf saturates");
        // W-form: fcvtzs w0,s1 truncates toward zero
        sets(&mut c, 1, -2.9);
        c.exec(0x1E38_0020, &mut m); // fcvtzs w0,s1
        assert_eq!(c.x[0] as i32, -2);
    }

    #[test]
    fn fcmp_sets_flags() {
        let (mut c, mut m) = (cpu(), scratch());
        setd(&mut c, 1, 1.0);
        setd(&mut c, 2, 2.0);
        c.exec(0x1E62_2020, &mut m); // fcmp d1,d2  (1 < 2)
        assert!(
            c.flags.n && !c.flags.z && !c.flags.c && !c.flags.v,
            "less-than"
        );
        setd(&mut c, 2, 1.0);
        c.exec(0x1E62_2020, &mut m); // fcmp d1,d2  (equal)
        assert!(!c.flags.n && c.flags.z && c.flags.c && !c.flags.v, "equal");
        setd(&mut c, 2, 0.5);
        c.exec(0x1E62_2020, &mut m); // fcmp d1,d2  (1 > 0.5)
        assert!(
            !c.flags.n && !c.flags.z && c.flags.c && !c.flags.v,
            "greater-than"
        );
        setd(&mut c, 1, f64::NAN);
        c.exec(0x1E62_2020, &mut m); // fcmp d1,d2  (unordered)
        assert!(
            !c.flags.n && !c.flags.z && c.flags.c && c.flags.v,
            "unordered"
        );
        // compare against #0.0
        setd(&mut c, 1, 0.0);
        c.exec(0x1E60_2028, &mut m); // fcmp d1,#0.0
        assert!(c.flags.z && c.flags.c, "d1 == 0.0");
    }

    #[test]
    fn fcsel_picks_by_condition() {
        let (mut c, mut m) = (cpu(), scratch());
        setd(&mut c, 1, 11.0);
        setd(&mut c, 2, 22.0);
        c.flags.z = true; // EQ holds
        c.exec(0x1E62_0C20, &mut m); // fcsel d0,d1,d2,eq
        assert_eq!(getd(&c, 0), 11.0);
        c.flags.z = false; // EQ fails
        c.exec(0x1E62_0C20, &mut m);
        assert_eq!(getd(&c, 0), 22.0);
    }

    #[test]
    fn fccmp_conditional() {
        let (mut c, mut m) = (cpu(), scratch());
        setd(&mut c, 1, 1.0);
        setd(&mut c, 2, 1.0);
        c.flags.z = true; // EQ holds -> perform the compare (1.0 == 1.0)
        c.exec(0x1E62_0420, &mut m); // fccmp d1,d2,#0,eq
        assert!(c.flags.z && c.flags.c, "compare taken: equal");
        c.flags = Flags::default();
        c.flags.z = false; // EQ fails -> load nzcv = 0xF (all set)
        c.exec(0x1E62_042F, &mut m); // fccmp d1,d2,#0xf,eq
        assert!(
            c.flags.n && c.flags.z && c.flags.c && c.flags.v,
            "nzcv loaded"
        );
    }

    #[test]
    fn vector_add_sub_cmeq() {
        let (mut c, mut m) = (cpu(), scratch());
        c.v[1] = (5u128 << 64) | 3;
        c.v[2] = (7u128 << 64) | 4;
        c.exec(0x4EE2_8420, &mut m); // add v0.2d,v1.2d,v2.2d
        assert_eq!(c.v[0], (12u128 << 64) | 7);
        c.exec(0x6EA2_8420, &mut m); // sub v0.4s,v1.4s,v2.4s
        // low 32: 3-4 = -1 (0xFFFFFFFF); next 32: 0; high 64: 5-7=-2 lane, 0 lane
        assert_eq!(c.v[0] & 0xffff_ffff, 0xffff_ffff);
        c.v[1] = 0x1111_2222;
        c.v[2] = 0x1111_9999;
        c.exec(0x6EA2_8C20, &mut m); // cmeq v0.4s,v1.4s,v2.4s
        assert_eq!(c.v[0] & 0xffff_ffff, 0, "low lane differs -> 0");
        assert_eq!(
            (c.v[0] >> 32) & 0xffff_ffff,
            0xffff_ffff,
            "high lane equal -> all ones"
        );
    }

    #[test]
    fn sbfx_ubfx_extract_bitfield() {
        let (mut c, mut m) = (cpu(), scratch());
        c.x[1] = 0x1234_5678_9abc_def0;
        // sbfx x0,x1,#4,#8 -> bits[11:4] = 0xef, sign-extended (top bit set)
        c.exec(0x9344_2C20, &mut m);
        assert_eq!(c.x[0] as i64, -17);
        // ubfx x0,x1,#4,#8 -> same bits, zero-extended
        c.exec(0xD344_2C20, &mut m);
        assert_eq!(c.x[0], 0xef);
    }

    #[test]
    fn ccmp_feeds_csel() {
        let (mut c, mut m) = (cpu(), scratch());
        c.x[1] = 5;
        c.x[3] = 111;
        c.x[4] = 222;
        // ccmp x1,#5,#0,eq with EQ holding -> real compare: 5-5=0 -> Z set
        c.flags.z = true;
        assert!(matches!(c.exec(0xFA45_0820, &mut m), Step::Next));
        assert!(c.flags.z && c.flags.c, "5 == 5 sets Z and C");
        // csel x0,x3,x4,eq now sees Z set -> picks x3
        assert!(matches!(c.exec(0x9A84_0060, &mut m), Step::Next));
        assert_eq!(c.x[0], 111);

        // ccmp x1,#5,#0xf,ne with NE failing -> flags loaded from nzcv=0xf
        c.flags = Flags::default();
        c.flags.z = true; // NE fails since Z is set
        assert!(matches!(c.exec(0xFA45_182F, &mut m), Step::Next));
        assert!(
            c.flags.n && c.flags.z && c.flags.c && c.flags.v,
            "nzcv literal loaded when outer cond fails"
        );
        // csel x0,x3,x4,eq still sees Z set -> picks x3 again
        assert!(matches!(c.exec(0x9A84_0060, &mut m), Step::Next));
        assert_eq!(c.x[0], 111);
        // flip Z off directly and re-run csel -> now picks x4
        c.flags.z = false;
        assert!(matches!(c.exec(0x9A84_0060, &mut m), Step::Next));
        assert_eq!(c.x[0], 222);
    }

    #[test]
    fn rev_rbit_clz_ops() {
        let (mut c, mut m) = (cpu(), scratch());
        c.x[1] = 0x1122_3344_5566_7788;
        c.exec(0xDAC0_0C20, &mut m); // rev x0,x1
        assert_eq!(c.x[0], 0x8877_6655_4433_2211);
        c.exec(0xDAC0_0020, &mut m); // rbit x0,x1
        assert_eq!(c.x[0], 0x11ee_66aa_22cc_4488);
        c.exec(0xDAC0_1020, &mut m); // clz x0,x1
        assert_eq!(c.x[0], 3);
    }

    #[test]
    fn ldr_literal_and_ldpsw() {
        let base = 0x1_0000u64;
        let mut m = GuestMemory::new(base, 4 * PAGE_SIZE);
        m.map(base, PAGE_SIZE, Prot::rw()).unwrap();
        let mut c = cpu();
        c.pc = base;
        m.write(base + 16, &0x1122_3344_5566_7788u64.to_le_bytes())
            .unwrap();
        // ldr x2, .+16  (imm19=4 -> byte offset 16)
        assert!(matches!(c.exec(0x5800_0082, &mut m), Step::Next));
        assert_eq!(c.x[2], 0x1122_3344_5566_7788);

        // ldpsw x0,x1,[x2]: two consecutive words, first sign-extended
        c.x[2] = base + 0x100;
        m.write(base + 0x100, &(-1i32).to_le_bytes()).unwrap();
        m.write(base + 0x104, &5i32.to_le_bytes()).unwrap();
        assert!(matches!(c.exec(0x6940_0440, &mut m), Step::Next));
        assert_eq!(c.x[0] as i64, -1, "LDPSW sign-extends the first word");
        assert_eq!(c.x[1], 5);
    }

    #[test]
    fn neon_dup_umov_roundtrip() {
        let (mut c, mut m) = (cpu(), scratch());
        c.x[1] = 0xDEAD_BEEF;
        c.exec(0x4E04_0C20, &mut m); // dup v0.4s, w1
        let expect = (0xDEAD_BEEFu128 << 96)
            | (0xDEAD_BEEFu128 << 64)
            | (0xDEAD_BEEFu128 << 32)
            | 0xDEAD_BEEFu128;
        assert_eq!(c.v[0], expect);
        c.exec(0x0E0C_3C00, &mut m); // umov w0, v0.s[1]
        assert_eq!(c.x[0], 0xDEAD_BEEF);
    }

    #[test]
    fn neon_dup_element_and_smov() {
        let (mut c, mut m) = (cpu(), scratch());
        c.v[1] = 0x8442_0201;
        c.exec(0x0E07_0420, &mut m); // dup v0.8b, v1.b[3]  (byte 3 = 0x84)
        assert_eq!(c.v[0], 0x8484_8484_8484_8484);
        c.exec(0x4E07_2C20, &mut m); // smov x0, v1.b[3]  (sign-extend 0x84)
        assert_eq!(c.x[0] as i64, -124);
    }

    #[test]
    fn neon_vector_compares() {
        let (mut c, mut m) = (cpu(), scratch());
        c.v[1] = 0x0000_0000_0000_000A_FFFF_FFFD_0000_0005u128;
        c.v[2] = 0x0000_0001_0000_000A_FFFF_FFFD_0000_0003u128;
        c.exec(0x4EA2_3420, &mut m); // cmgt v0.4s,v1.4s,v2.4s
        assert_eq!(c.v[0], 0xffff_ffffu128);
        c.exec(0x4EA2_3C20, &mut m); // cmge v0.4s,v1.4s,v2.4s
        assert_eq!(c.v[0], 0xffff_ffff_ffff_ffff_ffff_ffffu128);
        c.exec(0x6EA2_3420, &mut m); // cmhi v0.4s,v1.4s,v2.4s
        assert_eq!(c.v[0], 0xffff_ffffu128);
        c.exec(0x6EA2_3C20, &mut m); // cmhs v0.4s,v1.4s,v2.4s
        assert_eq!(c.v[0], 0xffff_ffff_ffff_ffff_ffff_ffffu128);
    }

    #[test]
    fn neon_sshl_ushl_signed_shift_amount() {
        let (mut c, mut m) = (cpu(), scratch());
        // lanes: [1, 0x8000_0000, 0x1000_0000, 0xFFFF_FFFF]
        c.v[1] = 0xFFFF_FFFF_1000_0000_8000_0000_0000_0001u128;
        // shift amounts (low byte, signed): [4, -4, 31, -1]
        c.v[2] = 0xFFFF_FFFF_0000_001F_FFFF_FFFC_0000_0004u128;
        c.exec(0x4EA2_4420, &mut m); // sshl v0.4s,v1.4s,v2.4s
        assert_eq!(c.v[0], 0xFFFF_FFFF_0000_0000_F800_0000_0000_0010u128);
        c.exec(0x6EA2_4420, &mut m); // ushl v0.4s,v1.4s,v2.4s
        assert_eq!(c.v[0], 0x7FFF_FFFF_0000_0000_0800_0000_0000_0010u128);
    }

    #[test]
    fn neon_cnt_rbit() {
        let (mut c, mut m) = (cpu(), scratch());
        c.v[0] = u128::from_le_bytes([
            0x00, 0x01, 0x03, 0xff, 0x80, 0x0f, 0xf0, 0x55, 1, 1, 1, 1, 1, 1, 1, 1,
        ]);
        // cnt v1.8b, v0.8b ; addv b2, v1.8b  (popcount, as nginx uses it)
        c.exec(0x0E20_5801, &mut m);
        assert_eq!(
            c.v[1],
            u128::from(u64::from_le_bytes([0, 1, 2, 8, 1, 4, 4, 4]))
        );
        c.exec(0x0E31_B822, &mut m);
        assert_eq!(c.v[2], 24);
        c.exec(0x4E20_5803, &mut m); // cnt v3.16b, v0.16b
        assert_eq!(c.v[3] >> 64, u128::from(u64::from_le_bytes([1; 8])));
        c.exec(0x2E60_5804, &mut m); // rbit v4.8b, v0.8b
        assert_eq!(
            c.v[4],
            u128::from(u64::from_le_bytes([
                0x00, 0x80, 0xc0, 0xff, 0x01, 0xf0, 0x0f, 0xaa
            ]))
        );
    }

    #[test]
    fn simd_compare_zero_and_fp_int_conversions() {
        let (mut c, mut m) = (cpu(), scratch());
        // cmeq d0, d0, #0 (scalar) on zero and nonzero
        c.v[0] = 0;
        c.exec(0x5EE0_9800, &mut m);
        assert_eq!(c.v[0], u128::from(u64::MAX));
        c.exec(0x5EE0_9800, &mut m);
        assert_eq!(c.v[0], 0);
        // cmlt v1.4s, v2.4s, #0
        c.v[2] = u128::from_le_bytes([
            1, 0, 0, 0, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0x80,
        ]);
        c.exec(0x4EA0_A841, &mut m);
        assert_eq!(c.v[1], 0xffff_ffff_0000_0000_ffff_ffff_0000_0000);
        // scvtf d0, d0 ; ucvtf d1, d1
        c.v[0] = u128::from((-3i64) as u64);
        c.exec(0x5E61_D800, &mut m);
        assert_eq!(f64::from_bits(c.v[0] as u64), -3.0);
        c.v[1] = u128::from(u64::MAX);
        c.exec(0x7E61_D821, &mut m);
        assert_eq!(f64::from_bits(c.v[1] as u64), 18_446_744_073_709_551_615.0);
        // fcvtzs d2, d2 (toward zero) ; fcvtas x0, d0 (ties away)
        c.v[2] = u128::from((-2.7f64).to_bits());
        c.exec(0x5EE1_B842, &mut m);
        assert_eq!(c.v[2] as u64 as i64, -2);
        c.v[0] = u128::from(2.5f64.to_bits());
        c.exec(0x9E64_0000, &mut m);
        assert_eq!(c.x[0], 3);
        // fcvtms x0, d0 (floor) ; fcvtns x0, d0 (ties to even)
        c.v[0] = u128::from((-2.5f64).to_bits());
        c.exec(0x9E70_0000, &mut m);
        assert_eq!(c.x[0] as i64, -3);
        c.exec(0x9E60_0000, &mut m);
        assert_eq!(c.x[0] as i64, -2);
        // scvtf d0, w0, #2 ; fcvtzs w0, d0, #2  (fixed point, 2 fraction bits)
        c.x[0] = 10;
        c.exec(0x1E42_F800, &mut m);
        assert_eq!(f64::from_bits(c.v[0] as u64), 2.5);
        c.v[0] = u128::from((-1.3f64).to_bits());
        c.exec(0x1E58_F800, &mut m);
        assert_eq!(c.x[0], u64::from((-5i32) as u32));
        // frintx / frinti d0, d0 use FPCR.RMode (default: nearest even)
        c.v[0] = u128::from(2.5f64.to_bits());
        c.exec(0x1E67_4000, &mut m);
        assert_eq!(f64::from_bits(c.v[0] as u64), 2.0);
        c.v[0] = u128::from(2.5f64.to_bits());
        c.fpcr = 1 << 22; // round toward +inf
        c.exec(0x1E67_C000, &mut m);
        assert_eq!(f64::from_bits(c.v[0] as u64), 3.0);
    }

    #[test]
    fn neon_not_addv_uaddlv() {
        let (mut c, mut m) = (cpu(), scratch());
        c.v[1] = 0x1234_5678_9ABC_DEF0_1122_3344_5566_7788u128;
        c.exec(0x6E20_5820, &mut m); // not v0.16b, v1.16b (mvn)
        assert_eq!(c.v[0], !c.v[1]);

        // bytes 1..=16 (little-endian lane order)
        c.v[1] = 0x100F_0E0D_0C0B_0A09_0807_0605_0403_0201u128;
        c.exec(0x4E31_B820, &mut m); // addv b0, v1.16b
        assert_eq!(c.v[0], 136);
        c.exec(0x6E30_3820, &mut m); // uaddlv h0, v1.16b
        assert_eq!(c.v[0], 136);
    }

    #[test]
    fn neon_ld1_st1_multiple_structures() {
        let base = 0x1_0000u64;
        let mut m = GuestMemory::new(base, 4 * PAGE_SIZE);
        m.map(base, PAGE_SIZE, Prot::rw()).unwrap();
        let mut c = cpu();
        c.x[1] = base + 0x40;
        c.v[0] = 0x1122_3344_5566_7788_99AA_BBCC_DDEE_FF00u128;
        // st1 {v0.16b},[x1]
        assert!(matches!(c.exec(0x4C00_7020, &mut m), Step::Next));
        // ld1 {v0.16b},[x1],#16 (post-index, clobber v0 first)
        c.v[0] = 0;
        assert!(matches!(c.exec(0x4CDF_7020, &mut m), Step::Next));
        assert_eq!(c.v[0], 0x1122_3344_5566_7788_99AA_BBCC_DDEE_FF00u128);
        assert_eq!(c.x[1], base + 0x50, "post-index advanced by 16 bytes");
    }

    #[test]
    fn neon_ld1_st1_two_registers() {
        // The 2-register form (`{v0.16b, v1.16b}`) aarch64 memcpy uses — the
        // gap that stopped apk on aarch64. Store the pair, reload it, and
        // check the post-index advances by 2 * 16 bytes.
        let base = 0x1_0000u64;
        let mut m = GuestMemory::new(base, 4 * PAGE_SIZE);
        m.map(base, PAGE_SIZE, Prot::rw()).unwrap();
        let mut c = cpu();
        c.x[2] = base + 0x40;
        c.v[0] = 0x0101_0101_0101_0101_0101_0101_0101_0101u128;
        c.v[1] = 0x0202_0202_0202_0202_0202_0202_0202_0202u128;
        // st1 {v0.16b, v1.16b}, [x2]   (opcode 0b1010)
        assert!(matches!(c.exec(0x4C00_A040, &mut m), Step::Next));
        c.v[0] = 0;
        c.v[1] = 0;
        // ld1 {v0.16b, v1.16b}, [x2], #32  (post-index by the 2-reg total)
        assert!(matches!(c.exec(0x4CDF_A040, &mut m), Step::Next));
        assert_eq!(c.v[0], 0x0101_0101_0101_0101_0101_0101_0101_0101u128);
        assert_eq!(c.v[1], 0x0202_0202_0202_0202_0202_0202_0202_0202u128);
        assert_eq!(c.x[2], base + 0x60, "post-index advanced by 2*16 bytes");
    }

    /// Coverage scan: execute every instruction word listed in
    /// `NIXVM_SCAN_WORDS` (lines of `<hex word> <mnemonic …>`, e.g. from
    /// `objdump -d` of a distro's binaries) on a scratch CPU and report the
    /// mnemonics the decoder rejects or panics on. A way to find the
    /// instructions real programs use that the interpreter lacks, all at
    /// once. Run with `--ignored --nocapture`.
    #[test]
    #[ignore = "needs NIXVM_SCAN_WORDS"]
    fn scan_instruction_coverage() {
        let Ok(path) = std::env::var("NIXVM_SCAN_WORDS") else {
            return;
        };
        let text = std::fs::read_to_string(path).unwrap();
        let base = 0x1_0000u64;
        let mut m = GuestMemory::new(base, 4 * PAGE_SIZE);
        m.map(base, 2 * PAGE_SIZE, Prot::rw()).unwrap();
        let mut by_mnemonic: std::collections::BTreeMap<String, (usize, String, bool)> =
            std::collections::BTreeMap::new();
        std::panic::set_hook(Box::new(|_| {}));
        for line in text.lines() {
            let mut it = line.splitn(2, ' ');
            let (Some(hex), Some(asm)) = (it.next(), it.next()) else {
                continue;
            };
            let Ok(word) = u32::from_str_radix(hex, 16) else {
                continue;
            };
            let mnemonic = asm.split_whitespace().next().unwrap_or("").to_string();
            if mnemonic.is_empty() || mnemonic.starts_with('<') || mnemonic == "udf" {
                continue; // data in .text, or a deliberate trap
            }
            let mut c = cpu();
            for r in 0..31 {
                c.x[r] = base + 0x100;
            }
            c.sp = base + 0x1000;
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                matches!(c.exec(word, &mut m), Step::Illegal)
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
        for (mn, (n, example, panicked)) in &by_mnemonic {
            let p = if *panicked { " PANIC" } else { "" };
            println!("{n:6} {mn:12}{p}  e.g. {example}");
        }
        println!("{} mnemonics unsupported", by_mnemonic.len());
    }

    #[test]
    fn neon_dup_element_scalar() {
        let (mut c, mut m) = (cpu(), scratch());
        c.v[5] = (0xDEAD_BEEF_0000_0001u128 << 64) | 7;
        c.v[30] = u128::MAX;
        c.exec(0x5e18_04be, &mut m); // mov d30, v5.d[1]
        assert_eq!(c.v[30], 0xDEAD_BEEF_0000_0001);
    }

    #[test]
    fn simd_ldst_register_offset() {
        let base = 0x1_0000u64;
        let mut m = GuestMemory::new(base, 4 * PAGE_SIZE);
        m.map(base, PAGE_SIZE, Prot::rw()).unwrap();
        let mut c = cpu();
        c.x[1] = base;
        c.x[3] = 0x40;
        c.v[0] = 0x0011_2233_4455_6677_8899_AABB_CCDD_EEFFu128;
        assert!(matches!(c.exec(0x3ca3_6820, &mut m), Step::Next)); // str q0,[x1,x3]
        c.x[2] = base;
        c.x[4] = 4; // lsl #4 -> +0x40
        assert!(matches!(c.exec(0x3ce4_7841, &mut m), Step::Next)); // ldr q1,[x2,x4,lsl #4]
        assert_eq!(c.v[1], c.v[0]);
        c.x[5] = base + 0x48;
        c.x[6] = u64::from((-1i32) as u32); // sxtw #3 -> -8
        assert!(matches!(c.exec(0xfc66_d8a3, &mut m), Step::Next)); // ldr d3,[x5,w6,sxtw #3]
        assert_eq!(c.v[3], 0x8899_AABB_CCDD_EEFF);
        c.x[11] = base + 0x80;
        c.x[0] = 1;
        c.v[2] = 0x5a;
        assert!(matches!(c.exec(0x3c20_6962, &mut m), Step::Next)); // str b2,[x11,x0]
        let mut b = [0u8; 1];
        m.read(base + 0x81, &mut b).unwrap();
        assert_eq!(b[0], 0x5a);
    }

    #[test]
    fn neon_ins_element() {
        let (mut c, mut m) = (cpu(), scratch());
        c.v[1] = (0xAAAA_AAAA_AAAA_AAAAu128 << 64) | 0x1111;
        c.v[2] = 0x2222;
        c.exec(0x6e08_4422, &mut m); // mov v2.d[0], v1.d[1]
        assert_eq!(c.v[2], 0xAAAA_AAAA_AAAA_AAAA);
        c.v[0] = 0x0123_4567_89AB_CDEF;
        c.exec(0x6e18_0401, &mut m); // mov v1.d[1], v0.d[0]
        assert_eq!(c.v[1], (0x0123_4567_89AB_CDEFu128 << 64) | 0x1111);
        c.v[3] = 0;
        c.v[4] = 0x5555_5555u128 << 32;
        c.exec(0x6e1c_2483, &mut m); // mov v3.s[3], v4.s[1]
        assert_eq!(c.v[3], 0x5555_5555u128 << 96);
        c.v[5] = 0;
        c.v[6] = 0x7f;
        c.exec(0x6e1f_04c5, &mut m); // mov v5.b[15], v6.b[0]
        assert_eq!(c.v[5], 0x7fu128 << 120);
        c.v[7] = u128::MAX;
        c.v[8] = 0x1234u128 << 112;
        c.exec(0x6e0a_7507, &mut m); // mov v7.h[2], v8.h[7]
        assert_eq!(c.v[7], !(0xffffu128 << 32) | (0x1234u128 << 32));
    }

    #[test]
    fn neon_pmull_carryless_multiply() {
        let (mut c, mut m) = (cpu(), scratch());
        // 64x64: x^63 * x^63 = x^126, and (x+1)^2 = x^2+1 (no carries).
        c.v[20] = (3u128 << 64) | (1u128 << 63);
        c.exec(0x0ef4_e280, &mut m); // pmull v0.1q, v20.1d, v20.1d
        assert_eq!(c.v[0], 1u128 << 126);
        c.exec(0x4ef4_e282, &mut m); // pmull2 v2.1q, v20.2d, v20.2d
        assert_eq!(c.v[2], 5);
        // A GHASH-sized check against a reference carry-less multiply.
        let (a, b) = (0x8765_4321_0fed_cba9u64, 0xdead_beef_cafe_f00du64);
        let mut want = 0u128;
        for i in 0..64 {
            if (b >> i) & 1 == 1 {
                want ^= u128::from(a) << i;
            }
        }
        c.v[20] = u128::from(a);
        c.v[21] = u128::from(b);
        c.exec(0x0ef5_e280, &mut m); // pmull v0.1q, v20.1d, v21.1d
        assert_eq!(c.v[0], want);
        // 8x8 lanes: 0x03*0x03 = 0x05, 0xff*0x02 = 0x1fe.
        c.v[2] = 0x0000_0000_0000_ff03u128 | (0x0000_0000_0000_0003u128 << 64);
        c.v[3] = 0x0000_0000_0000_0203u128 | (0x0000_0000_0000_0003u128 << 64);
        c.exec(0x0e23_e041, &mut m); // pmull v1.8h, v2.8b, v3.8b
        assert_eq!(c.v[1], 0x01fe_0005u128);
        c.exec(0x4e23_e041, &mut m); // pmull2 v1.8h, v2.16b, v3.16b
        assert_eq!(c.v[1], 0x0005u128);
    }

    #[test]
    fn neon_single_structure_lanes_and_replicate() {
        // `ld1 {v1.s}[0],[x0]` stopped apk's TLS on aarch64 ("Illegal
        // instruction"). Encodings from clang; memory holds bytes 0x10.. at x0.
        let base = 0x1_0000u64;
        let mut m = GuestMemory::new(base, 4 * PAGE_SIZE);
        m.map(base, PAGE_SIZE, Prot::rw()).unwrap();
        let data: Vec<u8> = (0x10..0x30).collect();
        m.write(base, &data).unwrap();
        let mut c = cpu();
        let ones = u128::MAX;

        // ld1 {v1.s}[0],[x0]: lane 0 loaded, lanes 1..3 kept.
        c.x[0] = base;
        c.v[1] = ones;
        assert!(matches!(c.exec(0x0d40_8001, &mut m), Step::Next));
        assert_eq!(c.v[1], (ones << 32) | 0x1312_1110);
        // ld1 {v1.s}[3],[x0],#4: lane 3, post-index by 4.
        assert!(matches!(c.exec(0x4ddf_9001, &mut m), Step::Next));
        assert_eq!(c.v[1] >> 96, 0x1312_1110);
        assert_eq!(c.x[0], base + 4);
        // ld1 {v3.b}[9],[x0],x4: byte lane 9, post-index by register.
        c.x[4] = 3;
        c.v[3] = 0;
        assert!(matches!(c.exec(0x4dc4_0403, &mut m), Step::Next));
        assert_eq!(c.v[3], 0x14u128 << 72);
        assert_eq!(c.x[0], base + 7);
        // ld1 {v4.d}[1],[x0]: 64-bit lane 1.
        c.x[0] = base;
        c.v[4] = 0xAAAA;
        assert!(matches!(c.exec(0x4d40_8404, &mut m), Step::Next));
        assert_eq!(c.v[4], (0x1716_1514_1312_1110u128 << 64) | 0xAAAA);
        // st1 {v2.h}[5],[x1]: stores halfword lane 5 only.
        c.x[1] = base + 0x100;
        c.v[2] = 0xBEEFu128 << 80;
        assert!(matches!(c.exec(0x4d00_4822, &mut m), Step::Next));
        let mut out = [0u8; 4];
        m.read(base + 0x100, &mut out).unwrap();
        assert_eq!(out, [0xEF, 0xBE, 0, 0]);
        // ld1r {v5.4s},[x0]: replicate across all four lanes.
        assert!(matches!(c.exec(0x4d40_c805, &mut m), Step::Next));
        assert_eq!(c.v[5], 0x1312_1110_1312_1110_1312_1110_1312_1110u128);
        // ld1r {v6.8b},[x0],#1: 64-bit replicate zeroes the upper half.
        c.v[6] = ones;
        assert!(matches!(c.exec(0x0ddf_c006, &mut m), Step::Next));
        assert_eq!(c.v[6], 0x1010_1010_1010_1010u128);
        assert_eq!(c.x[0], base + 1);
        // ld2 {v7.s,v8.s}[1],[x0]: consecutive elements to consecutive regs.
        c.x[0] = base;
        c.v[7] = 0;
        c.v[8] = 0;
        assert!(matches!(c.exec(0x0d60_9007, &mut m), Step::Next));
        assert_eq!(c.v[7], 0x1312_1110u128 << 32);
        assert_eq!(c.v[8], 0x1716_1514u128 << 32);
        // st2 {v7.s,v8.s}[1],[x1],#8: writes both lanes back, post-index 8.
        c.x[1] = base + 0x200;
        assert!(matches!(c.exec(0x0dbf_9027, &mut m), Step::Next));
        let mut out = [0u8; 8];
        m.read(base + 0x200, &mut out).unwrap();
        assert_eq!(out, [0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17]);
        assert_eq!(c.x[1], base + 0x208);
        // ld4r {v10-v13.16b},[x0]: four bytes, each replicated into its reg.
        assert!(matches!(c.exec(0x4d60_e00a, &mut m), Step::Next));
        for (i, b) in [0x10u8, 0x11, 0x12, 0x13].iter().enumerate() {
            assert_eq!(c.v[10 + i], u128::from_le_bytes([*b; 16]));
        }
    }

    /// Pack four `f32` lanes into a 128-bit vector register value (lane 0 low).
    fn quad_f32(a: f32, b: f32, c: f32, d: f32) -> u128 {
        (u128::from(d.to_bits()) << 96)
            | (u128::from(c.to_bits()) << 64)
            | (u128::from(b.to_bits()) << 32)
            | u128::from(a.to_bits())
    }
    /// Pack four 32-bit lanes into a 128-bit vector register value.
    fn quad_u32(a: u32, b: u32, c: u32, d: u32) -> u128 {
        (u128::from(d) << 96) | (u128::from(c) << 64) | (u128::from(b) << 32) | u128::from(a)
    }
    /// Pack four 16-bit lanes into the low 64 bits of a vector register value.
    fn quad_u16(a: u16, b: u16, c: u16, d: u16) -> u128 {
        (u128::from(d) << 48) | (u128::from(c) << 32) | (u128::from(b) << 16) | u128::from(a)
    }

    #[test]
    fn fmadd_fmsub_fnmadd_fnmsub() {
        let (mut c, mut m) = (cpu(), scratch());
        sets(&mut c, 1, 2.0);
        sets(&mut c, 2, 3.0);
        sets(&mut c, 3, 1.0);
        c.exec(0x1F02_0C20, &mut m); // fmadd s0,s1,s2,s3 -> 1.0 + 2.0*3.0
        assert_eq!(gets(&c, 0), 7.0);

        setd(&mut c, 1, 2.0);
        setd(&mut c, 2, 3.0);
        setd(&mut c, 3, 1.0);
        c.exec(0x1F42_8C20, &mut m); // fmsub d0,d1,d2,d3 -> 1.0 - 2.0*3.0
        assert_eq!(getd(&c, 0), -5.0);
        c.exec(0x1F62_0C20, &mut m); // fnmadd d0,d1,d2,d3 -> -1.0 - 2.0*3.0
        assert_eq!(getd(&c, 0), -7.0);
        c.exec(0x1F62_8C20, &mut m); // fnmsub d0,d1,d2,d3 -> 2.0*3.0 - 1.0
        assert_eq!(getd(&c, 0), 5.0);
    }

    #[test]
    fn fcvt_half_precision_roundtrip() {
        let (mut c, mut m) = (cpu(), scratch());
        sets(&mut c, 1, 1.5);
        c.exec(0x1E23_C020, &mut m); // fcvt h0,s1  (single -> half)
        assert_eq!(c.v[0] as u16, 0x3E00, "1.5 as f16");
        c.v[1] = c.v[0]; // fcvt s0,h1 reads h1, so move the half result there first
        c.exec(0x1EE2_4020, &mut m); // fcvt s0,h1  (half -> single)
        assert_eq!(gets(&c, 0), 1.5);

        setd(&mut c, 1, 0.5);
        c.exec(0x1E63_C020, &mut m); // fcvt h0,d1  (double -> half)
        assert_eq!(c.v[0] as u16, 0x3800, "0.5 as f16");
        c.v[1] = c.v[0];
        c.exec(0x1EE2_C020, &mut m); // fcvt d0,h1  (half -> double)
        assert_eq!(getd(&c, 0), 0.5);
    }

    #[test]
    fn neon_fp_vector_arithmetic_4s() {
        let (mut c, mut m) = (cpu(), scratch());
        c.v[1] = quad_f32(1.0, 2.0, 3.0, 4.0);
        c.v[2] = quad_f32(1.0, 1.0, 1.0, 2.0);
        c.exec(0x4E22_D420, &mut m); // fadd v0.4s, v1.4s, v2.4s
        assert_eq!(c.v[0], quad_f32(2.0, 3.0, 4.0, 6.0));
        c.exec(0x6E22_DC20, &mut m); // fmul v0.4s, v1.4s, v2.4s
        assert_eq!(c.v[0], quad_f32(1.0, 2.0, 3.0, 8.0));

        c.v[0] = quad_f32(10.0, 10.0, 10.0, 10.0);
        c.exec(0x4E22_CC20, &mut m); // fmla v0.4s, v1.4s, v2.4s  (v0 += v1*v2)
        assert_eq!(c.v[0], quad_f32(11.0, 12.0, 13.0, 18.0));

        c.v[1] = quad_f32(-1.0, -2.0, 3.0, -4.0);
        c.exec(0x4EA0_F820, &mut m); // fabs v0.4s, v1.4s
        assert_eq!(c.v[0], quad_f32(1.0, 2.0, 3.0, 4.0));

        c.v[1] = quad_f32(4.0, 9.0, 16.0, 25.0);
        c.exec(0x6EA1_F820, &mut m); // fsqrt v0.4s, v1.4s
        assert_eq!(c.v[0], quad_f32(2.0, 3.0, 4.0, 5.0));
    }

    #[test]
    fn neon_integer_mul_mla_abs_neg_minmax() {
        let (mut c, mut m) = (cpu(), scratch());
        c.v[1] = quad_u32(2, 3, 4, 5);
        c.v[2] = quad_u32(10, 10, 10, 10);
        c.exec(0x4EA2_9C20, &mut m); // mul v0.4s, v1.4s, v2.4s
        assert_eq!(c.v[0], quad_u32(20, 30, 40, 50));

        c.v[0] = quad_u32(1, 1, 1, 1);
        c.exec(0x4EA2_9420, &mut m); // mla v0.4s, v1.4s, v2.4s  (v0 += v1*v2)
        assert_eq!(c.v[0], quad_u32(21, 31, 41, 51));

        c.v[1] = quad_u32((-1i32) as u32, (-2i32) as u32, 3, (-4i32) as u32);
        c.exec(0x4EA0_B820, &mut m); // abs v0.4s, v1.4s
        assert_eq!(c.v[0], quad_u32(1, 2, 3, 4));
        c.exec(0x6EA0_B820, &mut m); // neg v0.4s, v1.4s  (v1 is still [-1,-2,3,-4])
        assert_eq!(c.v[0], quad_u32(1, 2, (-3i32) as u32, 4));

        c.v[1] = quad_u32(5, 5, (-1i32) as u32, 100);
        c.v[2] = quad_u32(3, 3, 1, 200);
        c.exec(0x4EA2_6420, &mut m); // smax v0.4s, v1.4s, v2.4s (signed: -1 < 1)
        assert_eq!(c.v[0], quad_u32(5, 5, 1, 200));
        c.exec(0x6EA2_6C20, &mut m); // umin v0.4s, v1.4s, v2.4s (unsigned: -1 is huge)
        assert_eq!(c.v[0], quad_u32(3, 3, 1, 100));
    }

    #[test]
    fn neon_saddl_uaddl_widening_add() {
        let (mut c, mut m) = (cpu(), scratch());
        c.v[1] = quad_u16(1, 2, 3, 0xFFFF); // last lane is -1 as i16
        c.v[2] = quad_u16(10, 20, 30, 1);
        c.exec(0x0E62_0020, &mut m); // saddl v0.4s, v1.4h, v2.4h
        assert_eq!(c.v[0], quad_u32(11, 22, 33, 0), "signed: -1 + 1 == 0");
        c.exec(0x2E62_0020, &mut m); // uaddl v0.4s, v1.4h, v2.4h
        assert_eq!(
            c.v[0],
            quad_u32(11, 22, 33, 0x1_0000),
            "unsigned: 0xFFFF + 1"
        );
    }

    #[test]
    fn fpcr_fpsr_read_zero() {
        let (mut c, mut m) = (cpu(), scratch());
        c.x[0] = 0xdead;
        c.exec(0xD53B_4400, &mut m); // mrs x0, fpcr
        assert_eq!(c.x[0], 0, "FPCR reads 0");
        c.exec(0xD51B_4400, &mut m); // msr fpcr, x0  (ignored, no panic)
    }

    // The expected values in the NEON/CRC tests below were cross-checked
    // against native execution of the same instructions on aarch64 hardware
    // (Apple Silicon, via inline asm), not just hand-derived from the ARM
    // ARM pseudocode — see the interp NEON-widening task notes.

    #[test]
    fn tbl_tbx_and_ext() {
        let (mut c, mut m) = (cpu(), scratch());
        // tbl v0.8b, {v1.16b}, v2.8b: 1-register table lookup; indices past
        // the 16-byte table (16, 255, ...) read as 0, and the 8B form zeroes
        // the upper 64 bits of Vd.
        c.v[1] = u128::from_le_bytes([
            100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
        ]);
        c.v[2] = u128::from_le_bytes([0, 5, 15, 16, 255, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        c.exec(0x0E02_0020, &mut m);
        assert_eq!(
            c.v[0].to_le_bytes(),
            [100, 105, 115, 0, 0, 103, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0]
        );

        // tbx v0.8b, {v1.16b}, v2.8b: same indices, but out-of-range leaves
        // the destination byte unchanged instead of zeroing it.
        c.v[0] = u128::from_le_bytes([9, 9, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0]);
        c.exec(0x0E02_1020, &mut m);
        assert_eq!(
            c.v[0].to_le_bytes(),
            [100, 105, 115, 9, 9, 103, 100, 100, 0, 0, 0, 0, 0, 0, 0, 0]
        );

        // ext v0.16b, v1.16b, v2.16b, #4: 16 bytes of Vn:Vm starting at 4.
        c.v[1] = u128::from_le_bytes([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]);
        c.v[2] = u128::from_le_bytes([
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        ]);
        c.exec(0x6E02_2020, &mut m);
        assert_eq!(
            c.v[0].to_le_bytes(),
            [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        );
    }

    #[test]
    fn zip_uzp_trn_permute() {
        let (mut c, mut m) = (cpu(), scratch());
        c.v[1] = quad_u32(0x10, 0x11, 0x12, 0x13);
        c.v[2] = quad_u32(0x20, 0x21, 0x22, 0x23);
        c.exec(0x4E82_3820, &mut m); // zip1 v0.4s, v1.4s, v2.4s
        assert_eq!(c.v[0], quad_u32(0x10, 0x20, 0x11, 0x21));
        c.exec(0x4E82_7820, &mut m); // zip2 v0.4s, v1.4s, v2.4s
        assert_eq!(c.v[0], quad_u32(0x12, 0x22, 0x13, 0x23));
        c.exec(0x4E82_1820, &mut m); // uzp1 v0.4s, v1.4s, v2.4s
        assert_eq!(c.v[0], quad_u32(0x10, 0x12, 0x20, 0x22));
        c.exec(0x4E82_5820, &mut m); // uzp2 v0.4s, v1.4s, v2.4s
        assert_eq!(c.v[0], quad_u32(0x11, 0x13, 0x21, 0x23));
        c.exec(0x4E82_2820, &mut m); // trn1 v0.4s, v1.4s, v2.4s
        assert_eq!(c.v[0], quad_u32(0x10, 0x20, 0x12, 0x22));
        c.exec(0x4E82_6820, &mut m); // trn2 v0.4s, v1.4s, v2.4s
        assert_eq!(c.v[0], quad_u32(0x11, 0x21, 0x13, 0x23));
    }

    #[test]
    fn rev64_rev32_rev16_vector() {
        let (mut c, mut m) = (cpu(), scratch());
        let bytes: [u8; 16] = core::array::from_fn(|i| i as u8);
        c.v[1] = u128::from_le_bytes(bytes);
        c.exec(0x4EA0_0820, &mut m); // rev64 v0.4s, v1.4s
        assert_eq!(
            c.v[0].to_le_bytes(),
            [4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11]
        );
        c.exec(0x6E60_0820, &mut m); // rev32 v0.8h, v1.8h
        assert_eq!(
            c.v[0].to_le_bytes(),
            [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13]
        );
        c.exec(0x4E20_1820, &mut m); // rev16 v0.16b, v1.16b
        assert_eq!(
            c.v[0].to_le_bytes(),
            [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14]
        );
    }

    #[test]
    fn xtn_sqxtn_uqxtn_sqxtun_narrow() {
        let (mut c, mut m) = (cpu(), scratch());
        c.v[1] = quad_u32(0x0001_FFFF, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF);
        c.exec(0x0E61_2820, &mut m); // xtn v0.4h, v1.4s: plain truncation.
        assert_eq!(c.v[0], quad_u16(0xFFFF, 0xFFFF, 0x0000, 0xFFFF));
        c.exec(0x0E61_4820, &mut m); // sqxtn v0.4h, v1.4s: signed-saturate.
        assert_eq!(c.v[0], quad_u16(0x7FFF, 0x7FFF, 0x8000, 0xFFFF));
        c.exec(0x2E61_4820, &mut m); // uqxtn v0.4h, v1.4s: unsigned-saturate.
        assert_eq!(c.v[0], quad_u16(0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF));
        c.exec(0x2E61_2820, &mut m); // sqxtun v0.4h, v1.4s: signed->unsigned saturate.
        assert_eq!(c.v[0], quad_u16(0xFFFF, 0xFFFF, 0x0000, 0x0000));
    }

    #[test]
    fn addhn_uaddw_saddw_wide() {
        let (mut c, mut m) = (cpu(), scratch());
        c.v[1] = quad_u32(0xFFFF_FFFF, 0x0001_0000, 0x8000_0000, 1);
        c.v[2] = quad_u32(1, 0x0000_FFFF, 0x8000_0000, 1);
        c.exec(0x0E62_4020, &mut m); // addhn v0.4h, v1.4s, v2.4s
        assert_eq!(c.v[0], quad_u16(0, 1, 0, 0));

        c.v[1] = quad_u32(1, 2, 3, 4);
        c.v[2] = quad_u16(0xFFFF, 1, 2, 3);
        c.exec(0x2E62_1020, &mut m); // uaddw v0.4s, v1.4s, v2.4h
        assert_eq!(c.v[0], quad_u32(0x1_0000, 3, 5, 7));
        c.exec(0x0E62_1020, &mut m); // saddw v0.4s, v1.4s, v2.4h
        assert_eq!(c.v[0], quad_u32(0, 3, 5, 7));
    }

    #[test]
    fn sqadd_uqadd_sqsub_uqsub_saturate() {
        let (mut c, mut m) = (cpu(), scratch());
        c.v[1] = quad_u32(0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF, 5);
        c.v[2] = quad_u32(1, 0xFFFF_FFFF, 1, 3);
        c.exec(0x4EA2_0C20, &mut m); // sqadd v0.4s, v1.4s, v2.4s
        assert_eq!(
            c.v[0],
            quad_u32(0x7FFF_FFFF, 0x8000_0000, 0, 8),
            "SQADD clamps at INT32_MAX/INT32_MIN instead of wrapping"
        );
        c.exec(0x6EA2_0C20, &mut m); // uqadd v0.4s, v1.4s, v2.4s
        assert_eq!(c.v[0], quad_u32(0x8000_0000, 0xFFFF_FFFF, 0xFFFF_FFFF, 8));
        c.exec(0x4EA2_2C20, &mut m); // sqsub v0.4s, v1.4s, v2.4s
        assert_eq!(c.v[0], quad_u32(0x7FFF_FFFE, 0x8000_0001, 0xFFFF_FFFE, 2));
        c.exec(0x6EA2_2C20, &mut m); // uqsub v0.4s, v1.4s, v2.4s
        assert_eq!(c.v[0], quad_u32(0x7FFF_FFFE, 0, 0xFFFF_FFFE, 2));
    }

    #[test]
    fn sqshl_uqshl_sqrshl_uqrshl_register() {
        let (mut c, mut m) = (cpu(), scratch());
        c.v[1] = quad_u32(1, 0x4000_0000, 0xFFFF_FFFF, 0x8000_0000);
        // Shift amounts (low signed byte of each Vm lane): 4, 2, -1, -4.
        c.v[2] = quad_u32(4, 2, 0xFFFF_FFFF, 0xFFFF_FFFC);
        c.exec(0x4EA2_4C20, &mut m); // sqshl v0.4s, v1.4s, v2.4s
        assert_eq!(
            c.v[0],
            quad_u32(0x10, 0x7FFF_FFFF, 0xFFFF_FFFF, 0xF800_0000)
        );
        c.exec(0x6EA2_4C20, &mut m); // uqshl v0.4s, v1.4s, v2.4s
        assert_eq!(
            c.v[0],
            quad_u32(0x10, 0xFFFF_FFFF, 0x7FFF_FFFF, 0x0800_0000)
        );
        c.exec(0x4EA2_5C20, &mut m); // sqrshl v0.4s, v1.4s, v2.4s (rounding right shift)
        assert_eq!(c.v[0], quad_u32(0x10, 0x7FFF_FFFF, 0, 0xF800_0000));
        c.exec(0x6EA2_5C20, &mut m); // uqrshl v0.4s, v1.4s, v2.4s
        assert_eq!(
            c.v[0],
            quad_u32(0x10, 0xFFFF_FFFF, 0x8000_0000, 0x0800_0000)
        );
    }

    #[test]
    fn suqadd_usqadd_saturating_accumulate() {
        let (mut c, mut m) = (cpu(), scratch());
        c.v[0] = quad_u32(5, (-5i32) as u32, 0x7FFF_FFFF, (-2_000_000_000i32) as u32);
        c.v[1] = quad_u32(10, 3, 10, 0xFFFF_FFFF);
        c.exec(0x4EA0_3820, &mut m); // suqadd v0.4s, v1.4s (signed acc + unsigned addend)
        assert_eq!(
            c.v[0],
            quad_u32(15, (-2i32) as u32, 0x7FFF_FFFF, 0x7FFF_FFFF)
        );

        c.v[0] = quad_u32(5, 3, 0xFFFF_FFF0, 0);
        c.v[1] = quad_u32(10, (-5i32) as u32, 10, (-1i32) as u32);
        c.exec(0x6EA0_3820, &mut m); // usqadd v0.4s, v1.4s (unsigned acc + signed addend)
        assert_eq!(c.v[0], quad_u32(15, 0, 0xFFFF_FFFA, 0));
    }

    #[test]
    fn addp_smaxp_sminp_umaxp_uminp_pairwise() {
        let (mut c, mut m) = (cpu(), scratch());
        c.v[1] = quad_u32(1, 2, 3, 4);
        c.v[2] = quad_u32(10, 20, 30, 40);
        c.exec(0x4EA2_BC20, &mut m); // addp v0.4s, v1.4s, v2.4s
        assert_eq!(c.v[0], quad_u32(3, 7, 30, 70));
        c.exec(0x4EA2_A420, &mut m); // smaxp v0.4s, v1.4s, v2.4s
        assert_eq!(c.v[0], quad_u32(2, 4, 20, 40));
        c.exec(0x4EA2_AC20, &mut m); // sminp v0.4s, v1.4s, v2.4s
        assert_eq!(c.v[0], quad_u32(1, 3, 10, 30));
        c.exec(0x6EA2_A420, &mut m); // umaxp v0.4s, v1.4s, v2.4s
        assert_eq!(c.v[0], quad_u32(2, 4, 20, 40));
        c.exec(0x6EA2_AC20, &mut m); // uminp v0.4s, v1.4s, v2.4s
        assert_eq!(c.v[0], quad_u32(1, 3, 10, 30));

        // addp d0, v1.2d (scalar pairwise: the two 64-bit halves of one register)
        c.v[1] = (200u128 << 64) | 0x64;
        c.exec(0x5EF1_B820, &mut m);
        assert_eq!(c.v[0], 300);
    }

    #[test]
    fn faddp_fmaxp_vector_and_scalar() {
        let (mut c, mut m) = (cpu(), scratch());
        c.v[1] = quad_f32(1.0, 2.0, 3.0, 4.0);
        c.v[2] = quad_f32(10.0, 20.0, 30.0, 40.0);
        c.exec(0x6E22_D420, &mut m); // faddp v0.4s, v1.4s, v2.4s
        assert_eq!(c.v[0], quad_f32(3.0, 7.0, 30.0, 70.0));
        c.exec(0x6E22_F420, &mut m); // fmaxp v0.4s, v1.4s, v2.4s
        assert_eq!(c.v[0], quad_f32(2.0, 4.0, 20.0, 40.0));

        // faddp s0, v1.2s (scalar pairwise)
        c.v[1] = quad_f32(3.5, 4.5, 0.0, 0.0);
        c.exec(0x7E30_D820, &mut m);
        assert_eq!(f32::from_bits(c.v[0] as u32), 8.0);
    }

    #[test]
    fn smaxv_sminv_umaxv_uminv_across_lanes() {
        let (mut c, mut m) = (cpu(), scratch());
        c.v[1] = quad_u32((-5i32) as u32, 10, (-20i32) as u32, 3);
        c.exec(0x4EB0_A820, &mut m); // smaxv s0, v1.4s
        assert_eq!(c.v[0] as u32, 10);
        c.exec(0x4EB1_A820, &mut m); // sminv s0, v1.4s
        assert_eq!(c.v[0] as u32 as i32, -20);
        c.exec(0x6EB0_A820, &mut m); // umaxv s0, v1.4s (unsigned: -5's bit pattern is huge)
        assert_eq!(c.v[0] as u32, (-5i32) as u32);
        c.exec(0x6EB1_A820, &mut m); // uminv s0, v1.4s
        assert_eq!(c.v[0] as u32, 3);
    }

    #[test]
    fn crc32x_and_crc32cx_known_vectors() {
        let (mut c, mut m) = (cpu(), scratch());
        // CRC32X/CRC32CX over the 8-byte ASCII chunk "01234567", seeded with
        // 0xFFFFFFFF — cross-checked against native `crc32x`/`crc32cx`
        // execution on real aarch64 hardware.
        c.x[1] = 0xFFFF_FFFF;
        c.x[2] = 0x3736_3534_3332_3130;
        c.exec(0x9AC2_4C20, &mut m); // crc32x w0, w1, x2
        assert_eq!(c.x[0] as u32, 0xd27f_c50a);
        c.exec(0x9AC2_5C20, &mut m); // crc32cx w0, w1, x2
        assert_eq!(c.x[0] as u32, 0x53dd_dcdf);

        // Byte-at-a-time CRC32B of "123456789" (seeded 0xFFFFFFFF, then
        // bit-complemented) must match the standard CRC-32 check value.
        c.x[1] = 0xFFFF_FFFF;
        for &byte in b"123456789" {
            c.x[2] = u64::from(byte);
            c.exec(0x1AC2_4020, &mut m); // crc32b w0, w1, w2
            c.x[1] = c.x[0];
        }
        assert_eq!(!(c.x[1] as u32), 0xcbf4_3926);
    }

    #[test]
    fn unknown_instruction_is_illegal() {
        let (mut c, mut m) = (cpu(), scratch());
        assert!(matches!(c.exec(0x0000_0000, &mut m), Step::Illegal));
    }

    #[test]
    fn run_faults_on_unmapped_pc() {
        let mut mem = GuestMemory::new(0x1_0000, PAGE_SIZE);
        let mut c = Aarch64Interp::new(0x1_0000, 0x1_0000);
        assert_eq!(
            c.run(&mut mem).unwrap(),
            Exit::MemFault {
                addr: 0x1_0000,
                write: false
            }
        );
    }

    #[test]
    fn fmul_by_element() {
        let (mut c, mut m) = (cpu(), scratch());
        // fmul s0, s1, v2.s[0]  (encoding cross-checked via clang+objdump)
        c.v[1] = u128::from(3.0f32.to_bits());
        c.v[2] = quad_f32(2.0, 100.0, 100.0, 100.0); // index 0 -> 2.0
        c.exec(0x5F82_9020, &mut m);
        assert_eq!(f32::from_bits(c.v[0] as u32), 6.0);

        // fmul v0.4s, v1.4s, v2.s[1] (vector by-element, index 1 -> 5.0)
        c.v[1] = quad_f32(1.0, 2.0, 3.0, 4.0);
        c.v[2] = quad_f32(10.0, 5.0, 10.0, 10.0);
        c.exec(0x4FA2_9020, &mut m);
        assert_eq!(
            c.v[0],
            quad_f32(5.0, 10.0, 15.0, 20.0),
            "each lane of v1 * v2.s[1] (5.0)"
        );
    }

    #[test]
    fn fmla_by_element_accumulates() {
        let (mut c, mut m) = (cpu(), scratch());
        // fmla v0.4s, v1.4s, v2.s[1]  (Vd += Vn * Vm[index])
        c.v[0] = quad_f32(1.0, 1.0, 1.0, 1.0); // pre-existing accumulator
        c.v[1] = quad_f32(1.0, 2.0, 3.0, 4.0);
        c.v[2] = quad_f32(10.0, 5.0, 10.0, 10.0); // index 1 -> 5.0
        c.exec(0x4FA2_1020, &mut m);
        assert_eq!(c.v[0], quad_f32(6.0, 11.0, 16.0, 21.0));
    }

    #[test]
    fn frecpe_then_frecps_converges_toward_reciprocal() {
        let (mut c, mut m) = (cpu(), scratch());
        let x = 4.0f32;
        // frecpe s0, s1  (initial estimate of 1/x)
        c.v[1] = u128::from(x.to_bits());
        c.exec(0x5EA1_D820, &mut m);
        let estimate = f32::from_bits(c.v[0] as u32);
        // The architected 8-bit estimate (RecipEstimate table): 511/2048.
        assert_eq!(estimate, 0.249_511_72, "FRECPE(4.0) is the 8-bit estimate");

        // One Newton-Raphson refinement step: y1 = y0 * frecps(x, y0) roughly
        // doubles the estimate's precision.
        c.v[1] = u128::from(x.to_bits()); // s1 = x
        c.v[2] = u128::from(estimate.to_bits()); // s2 = y0
        c.exec(0x5E22_FC20, &mut m); // frecps s0, s1, s2 -> 2.0 - x*y0
        let step = f32::from_bits(c.v[0] as u32);
        let refined = estimate * step;
        assert!(
            (refined - 1.0 / x).abs() < 1e-6,
            "refined={refined} should converge to 1/x={}",
            1.0 / x
        );
    }

    #[test]
    fn fcmeq_vector_vs_zero_mask() {
        let (mut c, mut m) = (cpu(), scratch());
        // fcmeq v0.4s, v1.4s, #0.0
        c.v[1] = quad_f32(0.0, -0.0, 1.0, f32::NAN);
        c.exec(0x4EA0_D820, &mut m);
        assert_eq!(
            c.v[0],
            quad_u32(u32::MAX, u32::MAX, 0, 0),
            "lanes 0/1 (+0.0/-0.0) compare equal to zero, lane 2 (1.0) and \
             lane 3 (NaN, unordered) do not"
        );
    }

    #[test]
    fn fabd_scalar_and_vector() {
        let (mut c, mut m) = (cpu(), scratch());
        // fabd s0, s1, s2 -> |5.0 - 8.0| = 3.0
        c.v[1] = u128::from(5.0f32.to_bits());
        c.v[2] = u128::from(8.0f32.to_bits());
        c.exec(0x7EA2_D420, &mut m);
        assert_eq!(f32::from_bits(c.v[0] as u32), 3.0);

        // fabd v0.4s, v1.4s, v2.4s
        c.v[1] = quad_f32(1.0, -2.0, 10.0, 0.0);
        c.v[2] = quad_f32(4.0, 2.0, 3.0, -5.0);
        c.exec(0x6EA2_D420, &mut m);
        assert_eq!(c.v[0], quad_f32(3.0, 4.0, 7.0, 5.0));
    }

    #[test]
    fn ldnp_stnp_pair_roundtrip() {
        // LDNP/STNP share the plain LDP/STP decode path in this
        // interpreter (both addressing modes are identical; only the
        // non-temporal cache hint differs, which this model doesn't need to
        // simulate), so this doubles as regression coverage for that reuse.
        let mut c = cpu();
        let mut m = GuestMemory::new(0x1_0000, 4 * PAGE_SIZE);
        m.map(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        c.x[2] = 0x1_0040; // base (mapped)
        c.x[0] = 0x1111_1111_1111_1111;
        c.x[1] = 0x2222_2222_2222_2222;
        assert!(matches!(
            c.exec(0xA800_0440, &mut m), // stnp x0, x1, [x2]
            Step::Next
        ));
        c.x[0] = 0;
        c.x[1] = 0;
        assert!(matches!(
            c.exec(0xA840_0440, &mut m), // ldnp x0, x1, [x2]
            Step::Next
        ));
        assert_eq!(c.x[0], 0x1111_1111_1111_1111);
        assert_eq!(c.x[1], 0x2222_2222_2222_2222);
    }

    #[test]
    fn prfm_decodes_as_noop() {
        let (mut c, mut m) = (cpu(), scratch());
        // prfm pldl1keep, [x0] — point at an address with no mapping at
        // all; a real load would fault here, but a prefetch hint must not.
        c.x[0] = 0xDEAD_0000;
        let pc_before = c.pc;
        assert!(matches!(c.exec(0xF980_0000, &mut m), Step::Next));
        assert_eq!(c.x[0], 0xDEAD_0000, "PRFM must not touch any register");
        assert_eq!(
            c.pc, pc_before,
            "exec() itself doesn't advance pc; run() does"
        );

        // prfm pldl1keep, [x0, x1] (register-offset form) and the unscaled
        // immediate form must equally be no-ops, not faults.
        c.x[1] = 8;
        assert!(matches!(c.exec(0xF8A1_6800, &mut m), Step::Next));
    }

    #[test]
    fn mrs_ctr_el0_returns_constant() {
        let (mut c, mut m) = (cpu(), scratch());
        // mrs x0, ctr_el0
        assert!(matches!(c.exec(0xD53B_0020, &mut m), Step::Next));
        assert_eq!(c.x[0], CTR_EL0_VAL);
    }

    #[test]
    fn mrs_cntvct_el0_increases_across_reads() {
        let (mut c, mut m) = (cpu(), scratch());
        // mrs x0, cntvct_el0 (twice)
        assert!(matches!(c.exec(0xD53B_E040, &mut m), Step::Next));
        let first = c.x[0];
        assert!(matches!(c.exec(0xD53B_E040, &mut m), Step::Next));
        let second = c.x[0];
        assert!(
            second > first,
            "a spin on CNTVCT_EL0 must observe it advance"
        );
    }

    #[test]
    fn dc_zva_zeroes_a_64_byte_block() {
        let base = 0x1_0000u64;
        let mut m = GuestMemory::new(base, 4 * PAGE_SIZE);
        m.map(base, PAGE_SIZE, Prot::rw()).unwrap();
        // Fill a region spanning the target block (and its neighbours) with
        // a recognizable non-zero pattern first.
        let region = base + 0x100;
        m.write(region, &[0xAAu8; 256]).unwrap();
        let mut c = cpu();
        // DC ZVA's address isn't block-aligned; the real block touched must
        // still be exactly the DCZID_EL0_VAL-sized, block-aligned one.
        let unaligned_off = 0x72usize;
        c.x[0] = region + unaligned_off as u64;
        // dc zva, x0
        assert!(matches!(c.exec(0xD50B_7420, &mut m), Step::Next));
        let mut buf = [0u8; 256];
        m.read(region, &mut buf).unwrap();
        let block = DC_ZVA_BLOCK_BYTES as usize;
        let aligned_off = unaligned_off & !(block - 1);
        assert!(
            buf[..aligned_off].iter().all(|&b| b == 0xAA),
            "before the block"
        );
        assert!(
            buf[aligned_off..aligned_off + block]
                .iter()
                .all(|&b| b == 0),
            "the DC ZVA block itself"
        );
        assert!(
            buf[aligned_off + block..].iter().all(|&b| b == 0xAA),
            "after the block"
        );
    }

    #[test]
    fn dmb_isb_decode_as_noops() {
        let (mut c, mut m) = (cpu(), scratch());
        c.pc = 0x1000;
        // dmb sy — exec() itself never advances pc (run() does), so a
        // Step::Next with pc unchanged is exactly "this was a no-op".
        assert!(matches!(c.exec(0xD503_3FBF, &mut m), Step::Next));
        assert_eq!(c.pc, 0x1000);
        // isb (sy)
        assert!(matches!(c.exec(0xD503_3FDF, &mut m), Step::Next));
        assert_eq!(c.pc, 0x1000);
    }

    /// `AESE v0.16b, v1.16b` with `Vd=0`, `Vn = 00 01 02 .. 0f` (byte `i` ==
    /// `i`). Expected result captured from native execution of the real
    /// `AESE` instruction on this host's Apple Silicon CPU (`FEAT_AES`) —
    /// see the module-level comment above `AES_SBOX`.
    #[test]
    fn aese_matches_hardware_vector() {
        let (mut c, mut m) = (cpu(), scratch());
        c.v[0] = 0; // Vd
        c.v[1] = 0x0f0e_0d0c_0b0a_0908_0706_0504_0302_0100; // Vn = 00..0f
        // aese v0.16b, v1.16b
        assert!(matches!(c.exec(0x4E28_4820, &mut m), Step::Next));
        assert_eq!(c.v[0], 0x2b6f_7cfe_c577_d730_7bab_01f2_7667_6b63);
    }

    /// AES S-box generated from the GF(2^8) multiplicative inverse plus the
    /// standard affine transform, independently of the hardcoded
    /// `AES_SBOX`/`AES_INV_SBOX` tables — catches a transcription error in
    /// either table that a self-consistency check alone couldn't.
    #[test]
    fn aes_sbox_matches_generated_table() {
        fn gf_mul(mut a: u8, mut b: u8) -> u8 {
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
        let mut inv = [0u8; 256];
        for a in 1..=255u16 {
            for b in 1..=255u16 {
                if gf_mul(a as u8, b as u8) == 1 {
                    inv[a as usize] = b as u8;
                    break;
                }
            }
        }
        let rol = |x: u8, n: u32| x.rotate_left(n);
        for i in 0..256usize {
            let b = inv[i];
            let generated = b ^ rol(b, 1) ^ rol(b, 2) ^ rol(b, 3) ^ rol(b, 4) ^ 0x63;
            assert_eq!(AES_SBOX[i], generated, "AES_SBOX[{i:#04x}]");
            assert_eq!(AES_INV_SBOX[generated as usize], i as u8, "AES_INV_SBOX");
        }
    }

    /// `SHA256H q0, q1, v2.4s` with distinguishable (non-repeating-nibble)
    /// inputs. Expected result captured from native execution of the real
    /// `SHA256H` instruction on this host's Apple Silicon CPU
    /// (`FEAT_SHA256`) — see the module-level comment above `sha1_f`.
    #[test]
    fn sha256h_matches_hardware_vector() {
        let (mut c, mut m) = (cpu(), scratch());
        c.v[0] = 0xfeed_face_0def_aced_8bad_f00d_dead_beef; // Qd = abcd
        c.v[1] = 0xba5e_ba11_5ca1_ab1e_1337_c0de_cafe_babe; // Qn = efgh
        c.v[2] = 0xc0ff_ee11_ba1d_2323_0fac_ade1_000f_f1ce; // Vm.4S = W+K
        // sha256h q0, q1, v2.4s
        assert!(matches!(c.exec(0x5E02_4020, &mut m), Step::Next));
        assert_eq!(c.v[0], 0xea32_0a6e_642e_88ee_9b73_03d0_dd3d_d598);
    }

    /// A full SHA-256 block compression, built only from `SHA256SU0`/
    /// `SHA256SU1` (message schedule) and `SHA256H`/`SHA256H2` (compression
    /// rounds) plus `ADD`/`INS` for the surrounding bookkeeping a real
    /// SHA-256 implementation does in registers, checked against the
    /// well-known `SHA-256("abc")` digest. This is the strongest available
    /// check that the ARM-specific packing derived empirically for these
    /// four instructions (see the module-level comment above `sha1_f`) is
    /// actually self-consistent end to end, not just matching one captured
    /// vector.
    #[test]
    fn sha256_block_compression_matches_known_digest() {
        const K: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];
        const H0: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];
        // SHA-256("abc"), from the FIPS 180-4 example / any `sha256sum`.
        const EXPECT: [u32; 8] = [
            0xba7816bf, 0x8f01cfea, 0x414140de, 0x5dae2223, 0xb00361a3, 0x96177a9c, 0xb410ff61,
            0xf20015ad,
        ];

        // SHA-256("abc"), padded to one 64-byte block.
        let mut block = [0u8; 64];
        block[0..3].copy_from_slice(b"abc");
        block[3] = 0x80;
        block[63] = 0x18; // bit length 24, big-endian in the last byte
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(block[4 * i..4 * i + 4].try_into().unwrap());
        }
        // Message schedule extension via SHA256SU0 + SHA256SU1, 4 words at a
        // time: W[16..19] needs W[0..15], SU0(W[i-16..i-13], W[i-12..i-9])
        // then SU1(<su0 result>, W[i-8..i-5], W[i-4..i-1]).
        let (mut c, mut m) = (cpu(), scratch());
        for i in (16..64).step_by(4) {
            let wd = pack4(&w, i - 16);
            let wn = pack4(&w, i - 12);
            let wm8 = pack4(&w, i - 8);
            let wm4 = pack4(&w, i - 4);
            c.v[0] = wd; // Vd = W[i-16..i-13]
            c.v[1] = wn;
            // sha256su0 v0.4s, v1.4s
            assert!(matches!(c.exec(0x5E28_2820, &mut m), Step::Next));
            c.v[2] = wm8;
            c.v[3] = wm4;
            // sha256su1 v0.4s, v2.4s, v3.4s
            assert!(matches!(c.exec(0x5E03_6040, &mut m), Step::Next));
            for (j, word) in u32_lanes(c.v[0]).into_iter().enumerate() {
                w[i + j] = word;
            }
        }

        let abcd0 = pack_u32_lanes([H0[0], H0[1], H0[2], H0[3]]);
        let efgh0 = pack_u32_lanes([H0[4], H0[5], H0[6], H0[7]]);
        c.v[10] = abcd0; // ABCD_SAVED
        c.v[11] = efgh0; // EFGH_SAVED
        c.v[0] = abcd0;
        c.v[1] = efgh0;
        for i in (0..64).step_by(4) {
            let wk = pack_u32_lanes([
                w[i].wrapping_add(K[i]),
                w[i + 1].wrapping_add(K[i + 1]),
                w[i + 2].wrapping_add(K[i + 2]),
                w[i + 3].wrapping_add(K[i + 3]),
            ]);
            c.v[2] = wk;
            c.v[10] = c.v[0]; // save pre-round ABCD for SHA256H2
            // sha256h q0, q1, v2.4s
            assert!(matches!(c.exec(0x5E02_4020, &mut m), Step::Next));
            // sha256h2 q1, q10, v2.4s
            assert!(matches!(c.exec(0x5E02_5141, &mut m), Step::Next));
        }
        let final_a = u32_lanes(c.v[0]);
        let final_e = u32_lanes(c.v[1]);
        let digest = [
            H0[0].wrapping_add(final_a[0]),
            H0[1].wrapping_add(final_a[1]),
            H0[2].wrapping_add(final_a[2]),
            H0[3].wrapping_add(final_a[3]),
            H0[4].wrapping_add(final_e[0]),
            H0[5].wrapping_add(final_e[1]),
            H0[6].wrapping_add(final_e[2]),
            H0[7].wrapping_add(final_e[3]),
        ];
        assert_eq!(digest, EXPECT);
    }

    /// Pack `w[i..i+4]` into a `V.4S` register value, lane 0 = `w[i]` — the
    /// words are already plain `u32` values (`sha256_block_compression_
    /// matches_known_digest` converts the input bytes with
    /// `u32::from_be_bytes`, since SHA message words are big-endian), so
    /// packing them into lanes is just `pack_u32_lanes`, no further
    /// byte-order fixup needed.
    fn pack4(w: &[u32; 64], i: usize) -> u128 {
        pack_u32_lanes([w[i], w[i + 1], w[i + 2], w[i + 3]])
    }

    /// What EL0 code under Linux may not touch must SIGILL, as on hardware:
    /// `DAIF` (UMA clear), `MSR` (immediate), set/way-free EL1 cache ops,
    /// debug/hypervisor calls, the PMU and physical counter, unimplemented
    /// feature registers, and non-emulated parts of the ID space.
    #[test]
    fn el0_inaccessible_system_instructions_are_undefined() {
        let (mut c, mut m) = (cpu(), scratch());
        for word in [
            0xD53B_4220u32, // mrs x0, daif
            0xD503_42DF,    // msr daifset, #2
            0xD508_751F,    // ic iallu
            0xD508_7620,    // dc ivac, x0
            0xD400_0002,    // hvc #0
            0xD53B_9D00,    // mrs x0, pmccntr_el0
            0xD53B_2400,    // mrs x0, rndr (FEAT_RNG not advertised)
            0xD53B_E020,    // mrs x0, cntpct_el0
            0xD538_0020,    // mrs x0, s3_0_c0_c0_1 (not emulated)
            0xD538_4240,    // mrs x0, currentel
            0xD538_0800,    // mrs x0, s3_0_c0_c8_0 (outside the ID space)
            0xD51B_D060,    // msr tpidrro_el0, x0 (read-only at EL0)
            0xD503_31FF,    // sb with CRm != 0 (unallocated)
            0x0000_0001,    // udf #1
        ] {
            assert!(
                matches!(c.exec(word, &mut m), Step::Illegal),
                "{word:#010x} should be UNDEFINED at EL0"
            );
        }
    }

    /// `BRK #imm` is a breakpoint (SIGTRAP), not an undefined instruction;
    /// its immediate is reported and the pc stays on it.
    #[test]
    fn brk_is_a_breakpoint_with_its_immediate() {
        let (mut c, mut m) = (cpu(), scratch());
        assert!(matches!(
            c.exec(0xD420_0000, &mut m),
            Step::Breakpoint { imm: 0 }
        ));
        assert!(matches!(
            c.exec(0xD43E_8000, &mut m), // brk #0xf400
            Step::Breakpoint { imm: 0xf400 }
        ));
    }

    /// The ID registers Linux emulates for EL0 agree with `AT_HWCAP`.
    #[test]
    fn id_registers_match_advertised_features() {
        let (mut c, mut m) = (cpu(), scratch());
        let mut mrs = |word: u32| {
            assert!(matches!(c.exec(word, &mut m), Step::Next));
            c.x[0]
        };
        // AES=2 (with PMULL), SHA1=1, SHA2=2, CRC32=1, Atomic=2, RDM, SHA3,
        // DP, FHM, TS=2.
        assert_eq!(mrs(0xD538_0600), 0x0021_1001_1021_2120); // id_aa64isar0_el1
        // DPB=2, JSCVT, FCMA, LRCPC=2, FRINTTS, SB, BF16, I8MM.
        assert_eq!(mrs(0xD538_0620), 0x0010_1011_0021_1002); // id_aa64isar1_el1
        assert_eq!(mrs(0xD538_0400), 0x0001_0000_0011_0011); // id_aa64pfr0_el1
        assert_eq!(mrs(0xD538_0740), 1 << 32); // id_aa64mmfr2_el1: AT (LSE2)
        assert_eq!(mrs(0xD538_0000), 0x410F_D0C0); // midr_el1
        assert_eq!(mrs(0xD538_0700), 0xFF00_0000); // id_aa64mmfr0_el1
        assert_eq!(mrs(0xD53B_00E0), 4); // dczid_el0: 64-byte DC ZVA
    }

    /// Hint-space instructions from extensions this CPU lacks (PAuth, BTI)
    /// are architecturally NOPs and must leave every register alone.
    #[test]
    fn unimplemented_hints_are_nops() {
        let (mut c, mut m) = (cpu(), scratch());
        c.x[30] = 0x1234_5678;
        c.x[16] = 0xabcd;
        for word in [
            0xD503_233Fu32,
            0xD503_23BF,
            0xD503_245F,
            0xD503_20FF,
            0xD503_201F,
        ] {
            assert!(matches!(c.exec(word, &mut m), Step::Next));
        }
        assert_eq!((c.x[30], c.x[16]), (0x1234_5678, 0xabcd));
    }

    #[test]
    fn nzcv_fpcr_fpsr_roundtrip_and_masking() {
        let (mut c, mut m) = (cpu(), scratch());
        c.x[1] = 0xA000_0000; // N and C
        c.exec(0xD51B_4201, &mut m); // msr nzcv, x1
        assert!(c.flags.n && !c.flags.z && c.flags.c && !c.flags.v);
        c.exec(0xD53B_4202, &mut m); // mrs x2, nzcv
        assert_eq!(c.x[2], 0xA000_0000);
        // FPCR keeps only AHP/DN/FZ/RMode; FPSR only QC/IDC/IXC..IOC.
        c.x[1] = u64::MAX;
        c.exec(0xD51B_4401, &mut m); // msr fpcr, x1
        c.exec(0xD53B_4402, &mut m); // mrs x2, fpcr
        assert_eq!(c.x[2], 0x07C8_0000);
        c.exec(0xD51B_4421, &mut m); // msr fpsr, x1
        c.exec(0xD53B_4422, &mut m); // mrs x2, fpsr
        assert_eq!(c.x[2], 0x0800_009F);
    }

    /// `CLREX` closes the exclusive monitor, so a following store-exclusive
    /// fails.
    #[test]
    fn clrex_breaks_an_exclusive_pair() {
        let base = 0x1_0000u64;
        let mut m = GuestMemory::new(base, 4 * PAGE_SIZE);
        m.map(base, PAGE_SIZE, Prot::rw()).unwrap();
        let mut c = cpu();
        c.x[1] = base + 0x40;
        c.exec(0xC85F_7C20, &mut m); // ldxr x0, [x1]
        c.exec(0xD503_3F5F, &mut m); // clrex
        c.exec(0xC802_7C23, &mut m); // stxr w2, x3, [x1]
        assert_eq!(c.x[2], 1);
    }

    /// Regressions from real programs: perl's `add d1, d1, d2` (scalar
    /// integer ADD on a D register) and ripgrep's `uaddlp v5.8h, v5.16b`.
    #[test]
    fn scalar_add_and_uaddlp_from_real_programs() {
        let (mut c, mut m) = (cpu(), scratch());
        c.v[1] = (0xdead_u128 << 64) | u128::from(u64::MAX);
        c.v[2] = 2;
        assert!(matches!(c.exec(0x5EE2_8421, &mut m), Step::Next)); // add d1, d1, d2
        assert_eq!(c.v[1], 1, "wraps, and the upper half is zeroed");
        c.v[5] = u128::from_le_bytes([1, 2, 255, 255, 0, 0, 7, 9, 10, 20, 30, 40, 128, 128, 1, 0]);
        assert!(matches!(c.exec(0x6E20_28A5, &mut m), Step::Next)); // uaddlp v5.8h, v5.16b
        let lanes: Vec<u16> = (0..8).map(|i| (c.v[5] >> (16 * i)) as u16).collect();
        assert_eq!(lanes, [3, 510, 0, 16, 30, 70, 256, 1]);
    }

    /// Regression from a Go binary's startup (yq): `sri v4.4s, v30.4s, #20`.
    #[test]
    fn sri_from_go_runtime() {
        let (mut c, mut m) = (cpu(), scratch());
        c.v[4] = 0xFFFF_FFFF_1234_5678_0000_0000_AAAA_AAAA;
        c.v[30] = 0x8000_0000_FFFF_FFFF_1234_5678_0000_0001;
        assert!(matches!(c.exec(0x6F2C_47C4, &mut m), Step::Next));
        // Each lane keeps its top 20 bits and takes n >> 20 below them.
        let expect = |d: u32, n: u32| (d & !(u32::MAX >> 20)) | (n >> 20);
        let lanes: Vec<u32> = (0..4).map(|i| (c.v[4] >> (32 * i)) as u32).collect();
        assert_eq!(
            lanes,
            [
                expect(0xAAAA_AAAA, 1),
                expect(0, 0x1234_5678),
                expect(0x1234_5678, 0xFFFF_FFFF),
                expect(0xFFFF_FFFF, 0x8000_0000)
            ]
        );
    }
}
