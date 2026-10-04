//! Differential test of the aarch64 interpreter against real hardware.
//!
//! On an arm64 macOS host, each instruction word is executed twice from the
//! *same* random architectural state — once natively (JIT-assembled into a
//! `MAP_JIT` page, wrapped in a trampoline that loads every GPR, SP, NZCV,
//! FPCR, FPSR and all 32 V registers from a context block, runs the word, and
//! stores everything back) and once by the interpreter (`a64_step`) — and the
//! resulting states must match bit for bit: every GPR, SP, NZCV, V0-V31, the
//! FPSR exception/saturation flags, and all of scratch memory. Loads and
//! stores work identically in both worlds because the interpreter's
//! `GuestMemory` is placed at the *same virtual addresses* as the host scratch
//! buffer, and base registers are pointed into it. Native faults (SIGSEGV/
//! SIGBUS/SIGILL) are caught and must correspond to an interpreter fault or
//! UNDEFINED.
//!
//! By default it runs a curated sample of encodings (`tests/data/
//! aarch64_diff_words.txt`). For a full sweep, point `NIXVM_DIFF_WORDS` at a
//! file of `<hex word> <disassembly>` lines (e.g. random words filtered
//! through `llvm-mc --disassemble -mattr=+v8a,+neon,+fp-armv8,+aes,+sha2,
//! +crc,+lse`) and set `NIXVM_DIFF_ITERS` (random states per word, default
//! 24) and optionally `NIXVM_DIFF_SEED`.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]
#![allow(
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    clippy::cast_ptr_alignment
)]

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::fmt::Write as _;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};

use nixvm::vcpu::GuestMemory;
use nixvm::vcpu::interp::{A64State, A64Step, a64_step};
use nixvm::vcpu::mem::Prot;

// ---- host FFI ------------------------------------------------------------------

const PROT_READ: i32 = 1;
const PROT_WRITE: i32 = 2;
const PROT_EXEC: i32 = 4;
const MAP_PRIVATE: i32 = 0x0002;
const MAP_ANON: i32 = 0x1000;
const MAP_JIT: i32 = 0x0800;
const SIGILL: i32 = 4;
const SIGTRAP: i32 = 5;
const SIGFPE: i32 = 8;
const SIGBUS: i32 = 10;
const SIGSEGV: i32 = 11;
const SA_ONSTACK: i32 = 0x0001;
const SA_SIGINFO: i32 = 0x0040;

#[repr(C)]
struct SigAction {
    handler: usize,
    mask: u32,
    flags: i32,
}

#[repr(C)]
struct StackT {
    sp: *mut c_void,
    size: usize,
    flags: i32,
}

unsafe extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64)
    -> *mut c_void;
    fn pthread_jit_write_protect_np(enabled: i32);
    fn sys_icache_invalidate(start: *mut c_void, len: usize);
    fn sigaction(sig: i32, act: *const SigAction, old: *mut SigAction) -> i32;
    fn sigaltstack(ss: *const StackT, old: *mut StackT) -> i32;
}

/// Signal number caught during the native run (0 = none) and where to resume.
static FAULT_SIG: AtomicI32 = AtomicI32::new(0);
static FAULT_ADDR: AtomicU64 = AtomicU64::new(0);
static RESUME_PC: AtomicU64 = AtomicU64::new(0);

/// macOS arm64: `siginfo_t.si_addr` is at offset 24; `ucontext_t.uc_mcontext`
/// at 48; `__darwin_mcontext64.__ss.__pc` at 272.
extern "C" fn on_signal(sig: i32, info: *mut u8, uctx: *mut u8) {
    unsafe {
        let addr = *(info.add(24).cast::<u64>());
        FAULT_SIG.store(sig, Ordering::SeqCst);
        FAULT_ADDR.store(addr, Ordering::SeqCst);
        let mctx = *(uctx.add(48).cast::<*mut u8>());
        let pc = mctx.add(272).cast::<u64>();
        let resume = RESUME_PC.load(Ordering::SeqCst);
        if *pc != resume - 4 {
            // A fault outside the instruction under test: the trampoline
            // itself broke (a clobbered context register). Resuming would
            // loop forever.
            eprintln!("fault (signal {sig}) at {:#x} outside the test slot", *pc);
            std::process::abort();
        }
        *pc = resume;
    }
}

// ---- context block shared with the trampoline -----------------------------------

#[repr(C, align(16))]
#[derive(Clone, Copy)]
struct Ctx {
    x: [u64; 32],
    sp: u64,
    nzcv: u64,
    fpcr: u64,
    fpsr: u64,
    host_sp: u64,
    taken: u64,
    _pad: [u64; 2],
    v: [u128; 32],
}
const OFF_X: u32 = 0;
const OFF_SP: u32 = 256;
const OFF_NZCV: u32 = 264;
const OFF_FPCR: u32 = 272;
const OFF_FPSR: u32 = 280;
const OFF_HOSTSP: u32 = 288;
const OFF_TAKEN: u32 = 296;
const OFF_V: u32 = 320;

// ---- tiny assembler for the trampoline ---------------------------------------------

fn stp_x(a: u32, b: u32, n: u32, off: u32) -> u32 {
    0xA900_0000 | (((off / 8) & 0x7f) << 15) | (b << 10) | (n << 5) | a
}
fn ldp_x(a: u32, b: u32, n: u32, off: u32) -> u32 {
    0xA940_0000 | (((off / 8) & 0x7f) << 15) | (b << 10) | (n << 5) | a
}
fn stp_d(a: u32, b: u32, n: u32, off: u32) -> u32 {
    0x6D00_0000 | (((off / 8) & 0x7f) << 15) | (b << 10) | (n << 5) | a
}
fn ldp_d(a: u32, b: u32, n: u32, off: u32) -> u32 {
    0x6D40_0000 | (((off / 8) & 0x7f) << 15) | (b << 10) | (n << 5) | a
}
fn stp_q(a: u32, b: u32, n: u32, off: u32) -> u32 {
    0xAD00_0000 | (((off / 16) & 0x7f) << 15) | (b << 10) | (n << 5) | a
}
fn ldp_q(a: u32, b: u32, n: u32, off: u32) -> u32 {
    0xAD40_0000 | (((off / 16) & 0x7f) << 15) | (b << 10) | (n << 5) | a
}
fn str_x(t: u32, n: u32, off: u32) -> u32 {
    0xF900_0000 | ((off / 8) << 10) | (n << 5) | t
}
fn ldr_x(t: u32, n: u32, off: u32) -> u32 {
    0xF940_0000 | ((off / 8) << 10) | (n << 5) | t
}
fn mov_x(d: u32, s: u32) -> u32 {
    0xAA00_03E0 | (s << 16) | d
}
fn mov_from_sp(d: u32) -> u32 {
    0x9100_03E0 | d
}
fn mov_to_sp(s: u32) -> u32 {
    0x9100_001F | (s << 5)
}
const MSR_NZCV: u32 = 0xD51B_4200;
const MRS_NZCV: u32 = 0xD53B_4200;
const MSR_FPCR: u32 = 0xD51B_4400;
const MSR_FPSR: u32 = 0xD51B_4420;
const MRS_FPSR: u32 = 0xD53B_4420;
const RET: u32 = 0xD65F_03C0;

/// The native executor: a `MAP_JIT` code page plus the scratch data region
/// (immediately after it, so PC-relative literal loads can reach data).
struct Native {
    code: *mut u32,
    code_len: usize,
    data: *mut u8,
}

const CODE_LEN: usize = 16 * 1024;
const DATA_LEN: usize = 128 * 1024;
/// Base registers of memory instructions point here (leaves room below for
/// negative offsets, and above for scaled 12-bit offsets up to 64 KiB).
const DATA_BASE_OFF: u64 = 8 * 1024;

impl Native {
    fn new() -> Native {
        unsafe {
            let code = mmap(
                std::ptr::null_mut(),
                CODE_LEN,
                PROT_READ | PROT_WRITE | PROT_EXEC,
                MAP_PRIVATE | MAP_ANON | MAP_JIT,
                -1,
                0,
            );
            assert!(code as isize != -1, "mmap(MAP_JIT) failed");
            let hint = code.cast::<u8>().add(CODE_LEN).cast::<c_void>();
            let data = mmap(
                hint,
                DATA_LEN,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANON,
                -1,
                0,
            );
            assert!(data as isize != -1, "mmap(data) failed");
            // Alternate signal stack: the test's SP may point anywhere.
            let alt = mmap(
                std::ptr::null_mut(),
                256 * 1024,
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANON,
                -1,
                0,
            );
            let ss = StackT {
                sp: alt,
                size: 256 * 1024,
                flags: 0,
            };
            assert_eq!(sigaltstack(&raw const ss, std::ptr::null_mut()), 0);
            let act = SigAction {
                handler: on_signal as *const () as usize,
                mask: 0,
                flags: SA_SIGINFO | SA_ONSTACK,
            };
            for sig in [SIGILL, SIGSEGV, SIGBUS, SIGTRAP, SIGFPE] {
                assert_eq!(sigaction(sig, &raw const act, std::ptr::null_mut()), 0);
            }
            Native {
                code: code.cast(),
                code_len: CODE_LEN,
                data: data.cast(),
            }
        }
    }

    fn code_addr(&self) -> u64 {
        self.code as u64
    }
    fn data_addr(&self) -> u64 {
        self.data as u64
    }
    fn adjacent(&self) -> bool {
        self.data_addr() == self.code_addr() + self.code_len as u64
    }
}

/// Build the trampoline for `word` using `r` as the context-pointer register
/// and `t` as a scratch register (both unused by `word`). Returns the code and
/// the word index of the instruction under test.
fn build(word: u32, r: u32, t: u32, branchy: bool) -> (Vec<u32>, usize) {
    let mut c = vec![
        0xD102_C3FF, // sub sp, sp, #176
        stp_x(19, 20, 31, 0),
        stp_x(21, 22, 31, 16),
        stp_x(23, 24, 31, 32),
        stp_x(25, 26, 31, 48),
        stp_x(27, 28, 31, 64),
        stp_x(29, 30, 31, 80),
        stp_d(8, 9, 31, 96),
        stp_d(10, 11, 31, 112),
        stp_d(12, 13, 31, 128),
        stp_d(14, 15, 31, 144),
        str_x(18, 31, 160),
    ];
    if r != 0 {
        c.push(mov_x(r, 0));
    }
    c.push(mov_from_sp(t));
    c.push(str_x(t, r, OFF_HOSTSP));
    for i in (0..32).step_by(2) {
        c.push(ldp_q(i, i + 1, r, OFF_V + 16 * i));
    }
    c.push(ldr_x(t, r, OFF_FPCR));
    c.push(MSR_FPCR | t);
    c.push(ldr_x(t, r, OFF_FPSR));
    c.push(MSR_FPSR | t);
    c.push(ldr_x(t, r, OFF_NZCV));
    c.push(MSR_NZCV | t);
    c.push(ldr_x(t, r, OFF_SP));
    c.push(mov_to_sp(t));
    for i in 0..31 {
        if i != r {
            c.push(ldr_x(i, r, OFF_X + 8 * i));
        }
    }
    let slot = c.len();
    c.push(word);
    let epilogue = |c: &mut Vec<u32>, taken: u32| {
        for i in 0..31 {
            if i != r {
                c.push(str_x(i, r, OFF_X + 8 * i));
            }
        }
        c.push(mov_from_sp(t));
        c.push(str_x(t, r, OFF_SP));
        c.push(MRS_NZCV | t);
        c.push(str_x(t, r, OFF_NZCV));
        c.push(MRS_FPSR | t);
        c.push(str_x(t, r, OFF_FPSR));
        c.push(0xD280_0000 | (taken << 5) | t); // movz xt, #taken
        c.push(str_x(t, r, OFF_TAKEN));
        for i in (0..32).step_by(2) {
            c.push(stp_q(i, i + 1, r, OFF_V + 16 * i));
        }
        c.push(ldr_x(t, r, OFF_HOSTSP));
        c.push(mov_to_sp(t));
        c.push(MSR_FPCR | 31);
        c.push(MSR_FPSR | 31);
        c.push(ldr_x(18, 31, 160));
        c.push(ldp_x(19, 20, 31, 0));
        c.push(ldp_x(21, 22, 31, 16));
        c.push(ldp_x(23, 24, 31, 32));
        c.push(ldp_x(25, 26, 31, 48));
        c.push(ldp_x(27, 28, 31, 64));
        c.push(ldp_x(29, 30, 31, 80));
        c.push(ldp_d(8, 9, 31, 96));
        c.push(ldp_d(10, 11, 31, 112));
        c.push(ldp_d(12, 13, 31, 128));
        c.push(ldp_d(14, 15, 31, 144));
        c.push(0x9102_C3FF); // add sp, sp, #176
        c.push(RET);
    };
    if branchy {
        // slot: branch (to slot+8 if taken); slot+4: b not_taken;
        // slot+8: taken epilogue; not_taken: epilogue.
        let b_at = c.len();
        c.push(0); // patched below
        epilogue(&mut c, 1);
        let not_taken = c.len();
        c[b_at] = 0x1400_0000 | ((not_taken - b_at) as u32 & 0x03ff_ffff);
        epilogue(&mut c, 0);
    } else {
        epilogue(&mut c, 0);
    }
    (c, slot)
}

impl Native {
    /// Run `code` natively on `ctx`; returns the caught signal (0 if none).
    fn run(&self, code: &[u32], slot: usize, ctx: &mut Ctx) -> (i32, u64) {
        assert!(code.len() * 4 <= self.code_len);
        unsafe {
            pthread_jit_write_protect_np(0);
            std::ptr::copy_nonoverlapping(code.as_ptr(), self.code, code.len());
            pthread_jit_write_protect_np(1);
            sys_icache_invalidate(self.code.cast(), code.len() * 4);
            FAULT_SIG.store(0, Ordering::SeqCst);
            RESUME_PC.store(self.code_addr() + (slot as u64 + 1) * 4, Ordering::SeqCst);
            let f: extern "C" fn(*mut Ctx) = std::mem::transmute(self.code);
            f(ctx);
        }
        (
            FAULT_SIG.load(Ordering::SeqCst),
            FAULT_ADDR.load(Ordering::SeqCst),
        )
    }
}

// ---- random state ----------------------------------------------------------------

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn chance(&mut self, pct: u64) -> bool {
        self.below(100) < pct
    }
    fn pick<T: Copy>(&mut self, xs: &[T]) -> T {
        xs[self.below(xs.len() as u64) as usize]
    }
}

const F32_SPECIAL: &[u32] = &[
    0x0000_0000,
    0x8000_0000,
    0x3F80_0000,
    0xBF80_0000,
    0x7F80_0000,
    0xFF80_0000,
    0x7FC0_0000,
    0xFFC0_0000,
    0x7FC1_2345,
    0x7F80_0001,
    0xFFA0_0000,
    0x7FBF_FFFF,
    0x0000_0001,
    0x007F_FFFF,
    0x8040_0000,
    0x0080_0000,
    0x8080_0000,
    0x7F7F_FFFF,
    0xFF7F_FFFF,
    0x3F00_0000,
    0x3FC0_0000,
    0x4B80_0001,
    0x4040_0000,
    0x4F00_0000,
    0x4F80_0000,
    0x5F00_0000,
    0x5F80_0000,
    0xCF00_0000,
    0xDF00_0000,
    0x3EAA_AAAB,
    0x0080_0001,
    0x00FF_FFFF,
    0x3F7F_FFFF,
    0x4EFF_FFFF,
    0x2000_0000,
    0x1F80_0000,
    0x7E80_0000,
    0x0040_0000,
];
const F64_SPECIAL: &[u64] = &[
    0x0000_0000_0000_0000,
    0x8000_0000_0000_0000,
    0x3FF0_0000_0000_0000,
    0xBFF0_0000_0000_0000,
    0x7FF0_0000_0000_0000,
    0xFFF0_0000_0000_0000,
    0x7FF8_0000_0000_0000,
    0xFFF8_0000_0000_0000,
    0x7FF8_0000_1234_5678,
    0x7FF0_0000_0000_0001,
    0xFFF4_0000_0000_0000,
    0x0000_0000_0000_0001,
    0x000F_FFFF_FFFF_FFFF,
    0x8008_0000_0000_0000,
    0x0010_0000_0000_0000,
    0x7FEF_FFFF_FFFF_FFFF,
    0x3FE0_0000_0000_0000,
    0x3FF8_0000_0000_0000,
    0x4340_0000_0000_0001,
    0x41E0_0000_0000_0000,
    0x41F0_0000_0000_0000,
    0x43E0_0000_0000_0000,
    0x43F0_0000_0000_0000,
    0xC1E0_0000_0000_0000,
    0xC3E0_0000_0000_0000,
    0x3FD5_5555_5555_5555,
    0x380F_FFFF_FFFF_FFFF,
    0x3810_0000_0000_0000,
    0x47EF_FFFF_E000_0000,
    0x47EF_FFFF_F000_0000,
    0x36A0_0000_0000_0000,
    0x3E70_0000_0000_0000,
    0x40F0_0000_0000_0000,
];
const F16_SPECIAL: &[u16] = &[
    0x0000, 0x8000, 0x3C00, 0xBC00, 0x7C00, 0xFC00, 0x7E00, 0x7D00, 0xFE01, 0x0001, 0x03FF, 0x0400,
    0x7BFF, 0x3800, 0x3E00, 0x7FFF,
];
const I_SPECIAL: &[u64] = &[
    0,
    1,
    2,
    u64::MAX,
    0x7FFF_FFFF,
    0x8000_0000,
    0xFFFF_FFFF,
    0x1_0000_0000,
    0x7FFF_FFFF_FFFF_FFFF,
    0x8000_0000_0000_0000,
    0x8000_0000_0000_0001,
    0xFFFF_FFFF_8000_0000,
    0x7F,
    0x80,
    0xFF,
    0x7FFF,
    0x8000,
    0xFFFF,
    0xFFFF_FFFF_FFFF_FF80,
    63,
    64,
    32,
    31,
];

fn rand_gpr(rng: &mut Rng) -> u64 {
    match rng.below(10) {
        0..=3 => rng.next(),
        4 | 5 => rng.below(256),
        6 => rng.below(256).wrapping_neg(),
        7 | 8 => rng.pick(I_SPECIAL),
        _ => rng.next() & 0xffff_ffff,
    }
}

fn rand_half(rng: &mut Rng) -> u64 {
    match rng.below(9) {
        0 | 1 => rng.next(),
        2 => (0..2).fold(0u64, |a, i| {
            a | (u64::from(rng.pick(F32_SPECIAL)) << (32 * i))
        }),
        3 => rng.pick(F64_SPECIAL),
        4 => (0..4).fold(0u64, |a, i| {
            a | (u64::from(rng.pick(F16_SPECIAL)) << (16 * i))
        }),
        5 => {
            // random normal-ish floats
            let f = |rng: &mut Rng| {
                let e = 0x3F00_0000 + ((rng.below(64) as u32) << 23);
                u64::from(e | (rng.next() as u32 & 0x807F_FFFF))
            };
            f(rng) | (f(rng) << 32)
        }
        6 => {
            let e = (0x3F0u64 + rng.below(32)) << 52;
            e | (rng.next() & 0x800F_FFFF_FFFF_FFFF)
        }
        7 => (0..8).fold(0u64, |a, i| a | (rng.pick(I_SPECIAL) & 0xff) << (8 * i)),
        _ => (0..4).fold(0u64, |a, i| {
            a | ((rng.pick(I_SPECIAL) & 0xffff) << (16 * i))
        }),
    }
}

fn rand_state(rng: &mut Rng, pc: u64) -> A64State {
    let mut s = A64State::default();
    for x in &mut s.x {
        *x = rand_gpr(rng);
    }
    s.sp = rand_gpr(rng);
    for v in &mut s.v {
        *v = u128::from(rand_half(rng)) | (u128::from(rand_half(rng)) << 64);
    }
    s.nzcv = rng.below(16) << 28;
    let rmode = if rng.chance(50) { 0 } else { rng.below(4) };
    let mut fpcr = rmode << 22;
    if rng.chance(20) {
        fpcr |= 1 << 24; // FZ
    }
    if rng.chance(20) {
        fpcr |= 1 << 25; // DN
    }
    if rng.chance(20) {
        fpcr |= 1 << 26; // AHP
    }
    if rng.chance(20) {
        fpcr |= 1 << 19; // FZ16
    }
    s.fpcr = fpcr;
    s.fpsr = match rng.below(4) {
        0 | 1 => 0,
        2 => 0x10,
        _ => rng.next() & 0x0800_009F,
    };
    s.pc = pc;
    s
}

// ---- classification -----------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    Plain,
    Mem,
    /// B.cond/CBZ/CBNZ/TBZ/TBNZ/B/BL, offset rewritten to +8.
    Branch,
    /// BR/BLR/RET: Rn set to the taken target.
    BranchReg,
    Skip,
}

fn classify(word: u32) -> (Class, u32) {
    let op0 = (word >> 25) & 0xf;
    if op0 & 0b0101 == 0b0100 {
        return (Class::Mem, word);
    }
    if op0 & 0b1110 == 0b1010 {
        // Branches, exception generation, system.
        return match word >> 29 {
            0b000 | 0b100 => (Class::Branch, (word & 0xFC00_0000) | 2),
            0b001 | 0b101 if (word >> 25) & 1 == 0 => {
                (Class::Branch, (word & 0xFF00_001F) | (2 << 5))
            }
            0b001 | 0b101 => (Class::Branch, (word & 0xFFF8_001F) | (2 << 5)),
            0b010 if (word >> 24) & 3 == 0 && (word >> 4) & 1 == 0 => {
                (Class::Branch, (word & 0xFF00_001F) | (2 << 5))
            }
            0b110 if word & 0xFE1F_FC1F == 0xD61F_0000 && (word >> 21) & 0xf <= 2 => {
                (Class::BranchReg, word)
            }
            // System instructions safe to run natively: CFINV/XAFLAG/AXFLAG,
            // SB/DSB/DMB/ISB, and MRS/MSR of NZCV/FPSR plus MRS FPCR.
            0b110
                if matches!(word, 0xD500_401F | 0xD500_403F | 0xD500_405F | 0xD503_30FF)
                    || word & 0xFFFF_F0DF == 0xD503_309F =>
            {
                (Class::Plain, word)
            }
            0b110
                if matches!(
                    word & 0xFFFF_FFE0,
                    0xD53B_4200 | 0xD51B_4200 | 0xD53B_4420 | 0xD51B_4420 | 0xD53B_4400
                ) =>
            {
                (Class::Plain, word)
            }
            _ => (Class::Skip, word),
        };
    }
    (Class::Plain, word)
}

/// Register fields a word may reference (conservatively: every 5-bit field
/// position, plus pair successors).
fn used_regs(word: u32) -> u32 {
    let mut used = 0u32;
    for lsb in [0, 5, 10, 16] {
        let r = (word >> lsb) & 0x1f;
        used |= 1 << r;
        used |= 1 << ((r + 1) & 0x1f);
    }
    used
}

/// Constrained-unpredictable load/store forms (writeback with Rn == Rt, LDP
/// with Rt == Rt2, store-exclusive status overlapping data/address) whose
/// architectural result is UNKNOWN — skipped.
fn unpredictable(word: u32) -> bool {
    let op0 = (word >> 25) & 0xf;
    if op0 & 0b0101 != 0b0100 {
        return false;
    }
    let rt = word & 0x1f;
    let rn = (word >> 5) & 0x1f;
    let rt2 = (word >> 10) & 0x1f;
    let rs = (word >> 16) & 0x1f;
    let vector = (word >> 26) & 1 == 1;
    match (word >> 28) & 3 {
        0b10 => {
            let wback = (word >> 23) & 3 == 0b01 || (word >> 23) & 3 == 0b11;
            let load = (word >> 22) & 1 == 1;
            (load && rt == rt2) || (!vector && wback && rn != 31 && (rn == rt || rn == rt2))
        }
        0b11 if (word >> 24) & 1 == 0 && (word >> 21) & 1 == 0 => {
            let wback = (word >> 10) & 1 == 1;
            !vector && wback && rn != 31 && rn == rt
        }
        0b00 if !vector => {
            // Exclusives: Rs overlapping Rt/Rt2/Rn; LDXP with Rt == Rt2.
            let o2 = (word >> 23) & 1;
            let load = (word >> 22) & 1 == 1;
            let o1 = (word >> 21) & 1;
            if o2 == 0 && !load && (o1 == 0 || (word >> 30) >= 2) {
                rs == rt || rs == rn || (o1 == 1 && rs == rt2)
            } else {
                o2 == 0 && o1 == 1 && load && (word >> 30) >= 2 && rt == rt2
            }
        }
        // (Structure loads write V registers, so their post-index writeback
        // can't collide with a destination.)
        _ => false,
    }
}

/// Register-offset and post-index-register forms whose index register is
/// also the base: the address would be twice a host pointer — possibly mapped
/// host memory natively, and a store there would corrupt the test process.
fn wild_address(word: u32) -> bool {
    let op0 = (word >> 25) & 0xf;
    if op0 & 0b0101 != 0b0100 {
        return false;
    }
    let rn = (word >> 5) & 0x1f;
    let rm = (word >> 16) & 0x1f;
    let structure = (word >> 28) & 3 == 0 && (word >> 26) & 1 == 1;
    let regoff = (word >> 28) & 3 == 3
        && (word >> 24) & 1 == 0
        && (word >> 21) & 1 == 1
        && (word >> 10) & 3 == 2;
    (structure || regoff) && rm == rn
}

// ---- comparison -----------------------------------------------------------------

struct Report {
    by_mnemonic: BTreeMap<String, (u64, Vec<String>)>,
    cases: u64,
    words: u64,
    skipped: u64,
}

impl Report {
    fn add(&mut self, mnemonic: &str, msg: String) {
        let e = self
            .by_mnemonic
            .entry(mnemonic.to_string())
            .or_insert((0, Vec::new()));
        e.0 += 1;
        if e.1.len() < 3 {
            e.1.push(msg);
        }
    }
}

fn diff_states(
    word: u32,
    asm: &str,
    input: &A64State,
    nat: &A64State,
    emu: &A64State,
    r: u32,
) -> Option<String> {
    let mut out = String::new();
    for i in 0..31 {
        if i as u32 != r && i != 18 && nat.x[i] != emu.x[i] {
            let _ = writeln!(
                out,
                "  x{i}: in={:#x} native={:#x} interp={:#x}",
                input.x[i], nat.x[i], emu.x[i]
            );
        }
    }
    if nat.sp != emu.sp {
        let _ = writeln!(
            out,
            "  sp: in={:#x} native={:#x} interp={:#x}",
            input.sp, nat.sp, emu.sp
        );
    }
    if nat.nzcv != emu.nzcv {
        let _ = writeln!(
            out,
            "  nzcv: in={:x} native={:x} interp={:x}",
            input.nzcv >> 28,
            nat.nzcv >> 28,
            emu.nzcv >> 28
        );
    }
    if nat.fpsr != emu.fpsr {
        let _ = writeln!(
            out,
            "  fpsr: in={:#x} native={:#x} interp={:#x} (fpcr={:#x})",
            input.fpsr, nat.fpsr, emu.fpsr, input.fpcr
        );
    }
    for i in 0..32 {
        if nat.v[i] != emu.v[i] {
            let _ = writeln!(
                out,
                "  v{i}: in={:#034x} native={:#034x} interp={:#034x}",
                input.v[i], nat.v[i], emu.v[i]
            );
        }
    }
    if out.is_empty() {
        None
    } else {
        // Show the likely source operands too.
        let (rn, rm) = (
            ((word >> 5) & 0x1f) as usize,
            ((word >> 16) & 0x1f) as usize,
        );
        let _ = writeln!(
            out,
            "  (inputs: v{rn}={:#034x} v{rm}={:#034x} x{rn}={:#x} x{rm}={:#x} fpcr={:#x})",
            input.v[rn],
            input.v[rm],
            input.x[rn.min(30)],
            input.x[rm.min(30)],
            input.fpcr
        );
        Some(format!("{word:08x} {asm}\n{out}"))
    }
}

fn run_words(words: &[(u32, String)], iters: u64, seed: u64) -> Report {
    let native = Native::new();
    let mut rng = Rng(seed | 1);
    let code_base = native.code_addr();
    let data_base = native.data_addr();
    // Interpreter memory at the same addresses: the code page (for literal
    // loads) and the data region.
    let (gm_base, gm_len) = if native.adjacent() {
        (code_base, (CODE_LEN + DATA_LEN) as u64)
    } else {
        (data_base, DATA_LEN as u64)
    };
    let mut mem = GuestMemory::new(gm_base, gm_len);
    mem.map(data_base, DATA_LEN as u64, Prot::rw()).unwrap();
    if native.adjacent() {
        mem.map(code_base, CODE_LEN as u64, Prot::rx()).unwrap();
    }
    let mut report = Report {
        by_mnemonic: BTreeMap::new(),
        cases: 0,
        words: 0,
        skipped: 0,
    };
    let mut data = vec![0u8; DATA_LEN];
    // Optionally misalign base registers (exercises alignment faults).
    let misalign = std::env::var_os("NIXVM_DIFF_MISALIGN").is_some();
    for (word, asm) in words {
        let mnemonic = asm.split_whitespace().next().unwrap_or("?").to_string();
        let (class, word) = classify(*word);
        if class == Class::Skip || unpredictable(word) || wild_address(word) {
            report.skipped += 1;
            continue;
        }
        let is_literal = class == Class::Mem && (word >> 28) & 3 == 1 && (word >> 24) & 1 == 0;
        let word = if is_literal {
            if !native.adjacent() {
                report.skipped += 1;
                continue;
            }
            // Point the literal at the data region: offset from the slot,
            // patched once the slot address is known (below).
            word
        } else {
            word
        };
        let mut used = used_regs(word);
        if class == Class::Branch || class == Class::BranchReg {
            used |= 1 << 30; // BL/BLR write the link register
        }
        let free: Vec<u32> = (0..31)
            .filter(|&i| used & (1 << i) == 0 && i != 18)
            .collect();
        if free.len() < 2 {
            report.skipped += 1;
            continue;
        }
        let r = free[rng.below(free.len() as u64) as usize];
        let t = *free.iter().find(|&&x| x != r).unwrap();
        let branchy = class == Class::Branch || class == Class::BranchReg;
        let (mut code, slot) = build(word, r, t, branchy);
        let slot_addr = code_base + slot as u64 * 4;
        if is_literal {
            let target = data_base + DATA_BASE_OFF + 16 * rng.below(64);
            let off = (target - slot_addr) / 4;
            code[slot] = (word & 0xFF00_001F) | (((off as u32) & 0x7ffff) << 5);
        }
        let word = code[slot];
        report.words += 1;
        if native.adjacent() {
            let bytes: Vec<u8> = code.iter().flat_map(|w| w.to_le_bytes()).collect();
            mem.write_init(code_base, &bytes).unwrap();
        }
        for _ in 0..iters {
            let mut st = rand_state(&mut rng, slot_addr);
            if class == Class::Mem {
                // Base register into the data region (16-byte aligned for
                // exclusives/atomics), index register small.
                let rn = ((word >> 5) & 0x1f) as usize;
                let mut base = data_base + DATA_BASE_OFF + 16 * rng.below(64);
                if misalign && rng.chance(50) {
                    base += rng.below(16);
                }
                if rn == 31 {
                    st.sp = base;
                } else {
                    st.x[rn] = base;
                }
                let rm = ((word >> 16) & 0x1f) as usize;
                let structure = (word >> 28) & 3 == 0 && (word >> 26) & 1 == 1;
                let regoff = (word >> 28) & 3 == 3
                    && (word >> 24) & 1 == 0
                    && (word >> 21) & 1 == 1
                    && (word >> 10) & 3 == 2;
                if (structure || regoff) && rm != 31 && rm != rn {
                    st.x[rm] = rng.below(512);
                }
            } else {
                // SP-relative non-memory ops are fine with any SP, but keep it
                // 16-byte aligned for realism.
                st.sp &= !0xf;
            }
            if class == Class::BranchReg {
                st.x[((word >> 5) & 0x1f) as usize] = slot_addr + 8;
            }
            for chunk in data.chunks_mut(8) {
                let v = rng.next().to_le_bytes();
                chunk.copy_from_slice(&v[..chunk.len()]);
            }
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), native.data, DATA_LEN);
            }
            mem.write_init(data_base, &data).unwrap();
            // Native run.
            let mut ctx = Ctx {
                x: [0; 32],
                sp: st.sp,
                nzcv: st.nzcv,
                fpcr: st.fpcr,
                fpsr: st.fpsr,
                host_sp: 0,
                taken: 0,
                _pad: [0; 2],
                v: st.v,
            };
            ctx.x[..31].copy_from_slice(&st.x);
            ctx.x[r as usize] = 0; // overwritten by the trampoline
            let (sig, fault_addr) = native.run(&code, slot, &mut ctx);
            // Interpreter run.
            let mut emu = st.clone();
            let step = a64_step(&mut emu, word, &mut mem);
            report.cases += 1;
            if sig != 0 {
                let ok = match sig {
                    SIGILL => step == A64Step::Illegal,
                    SIGSEGV | SIGBUS => matches!(step, A64Step::Fault { .. }),
                    _ => false,
                };
                if !ok {
                    report.add(
                        &mnemonic,
                        format!("{word:08x} {asm}: native signal {sig} (addr {fault_addr:#x}), interp {step:?}"),
                    );
                }
                continue;
            }
            let expect_step = if branchy {
                A64Step::Branched
            } else {
                A64Step::Next
            };
            let branch_ok = !branchy
                || (ctx.taken == 1 && step == A64Step::Branched && emu.pc == slot_addr + 8)
                || (ctx.taken == 0 && step == A64Step::Next && emu.pc == slot_addr + 4);
            if (!branchy && step != expect_step) || !branch_ok {
                report.add(
                    &mnemonic,
                    format!(
                        "{word:08x} {asm}: native ok (taken={}), interp {step:?} pc={:#x} (slot {slot_addr:#x})",
                        ctx.taken, emu.pc
                    ),
                );
                continue;
            }
            let mut nat = st.clone();
            nat.x.copy_from_slice(&ctx.x[..31]);
            nat.x[r as usize] = emu.x[r as usize];
            nat.sp = ctx.sp;
            nat.nzcv = ctx.nzcv & 0xF000_0000;
            nat.fpsr = ctx.fpsr;
            nat.v = ctx.v;
            if let Some(msg) = diff_states(word, asm, &st, &nat, &emu, r) {
                report.add(&mnemonic, msg);
                continue;
            }
            // Memory.
            let native_mem = unsafe { std::slice::from_raw_parts(native.data, DATA_LEN) };
            let emu_mem = mem.read_vec(data_base, DATA_LEN).unwrap();
            if native_mem != emu_mem.as_slice() {
                let first = native_mem
                    .iter()
                    .zip(&emu_mem)
                    .position(|(a, b)| a != b)
                    .unwrap();
                report.add(
                    &mnemonic,
                    format!(
                        "{word:08x} {asm}: memory differs at {:#x} (native {:#04x}, interp {:#04x})",
                        data_base + first as u64,
                        native_mem[first],
                        emu_mem[first]
                    ),
                );
            }
        }
    }
    report
}

fn load_words(text: &str) -> Vec<(u32, String)> {
    text.lines()
        .filter_map(|l| {
            let l = l.trim();
            if l.is_empty() || l.starts_with('#') {
                return None;
            }
            let (hex, asm) = l.split_once(' ').unwrap_or((l, "?"));
            Some((u32::from_str_radix(hex, 16).ok()?, asm.trim().to_string()))
        })
        .collect()
}

#[test]
fn interpreter_matches_native_execution() {
    let words = match std::env::var("NIXVM_DIFF_WORDS") {
        Ok(path) => load_words(&std::fs::read_to_string(path).unwrap()),
        Err(_) => load_words(include_str!("data/aarch64_diff_words.txt")),
    };
    let iters = std::env::var("NIXVM_DIFF_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let seed = std::env::var("NIXVM_DIFF_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x9E37_79B9_7F4A_7C15);
    let report = run_words(&words, iters, seed);
    let mut total = 0;
    for (mn, (n, examples)) in &report.by_mnemonic {
        total += n;
        eprintln!("== {mn}: {n} mismatches");
        for e in examples {
            eprintln!("{e}");
        }
    }
    eprintln!(
        "{} words, {} cases, {} skipped, {} mismatching mnemonics, {total} mismatches",
        report.words,
        report.cases,
        report.skipped,
        report.by_mnemonic.len()
    );
    assert_eq!(total, 0, "interpreter differs from native execution");
}

/// The converse check: words `llvm-mc` rejects for the advertised feature set
/// (`NIXVM_UNDEF_WORDS`, one hex word per line) must be UNDEFINED in the
/// interpreter. Reports the ones that aren't, grouped by their top 11 bits.
#[test]
fn rejected_encodings_are_undefined() {
    let Ok(path) = std::env::var("NIXVM_UNDEF_WORDS") else {
        return;
    };
    let text = std::fs::read_to_string(path).unwrap();
    let base = 0x10_0000u64;
    let mut mem = GuestMemory::new(base, 0x10_0000);
    mem.map(base, 0x10_0000, Prot::rw()).unwrap();
    let mut groups: BTreeMap<u32, (u64, Vec<u32>)> = BTreeMap::new();
    let mut total = 0u64;
    for line in text.lines() {
        let Ok(word) = u32::from_str_radix(line.split_whitespace().next().unwrap_or(""), 16) else {
            continue;
        };
        let mut st = A64State::default();
        for x in &mut st.x {
            *x = base + 0x8000;
        }
        st.sp = base + 0x8000;
        st.pc = base;
        if a64_step(&mut st, word, &mut mem) != A64Step::Illegal {
            total += 1;
            if let Ok(out) = std::env::var("NIXVM_UNDEF_OUT") {
                use std::io::Write;
                let mut f = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(out)
                    .unwrap();
                writeln!(f, "{word:08x} accepted-undef").unwrap();
            }
            let e = groups.entry(word >> 21).or_insert((0, Vec::new()));
            e.0 += 1;
            if e.1.len() < 4 {
                e.1.push(word);
            }
        }
    }
    for (top, (n, ex)) in &groups {
        let ex: Vec<String> = ex.iter().map(|w| format!("{w:08x}")).collect();
        eprintln!("{:011b}: {n} accepted, e.g. {}", top, ex.join(" "));
    }
    eprintln!("{total} rejected-by-llvm words accepted by the interpreter");
}

/// Coverage scan: write every word of `NIXVM_SCAN_WORDS` (`<hex> <asm>`
/// lines) that the interpreter treats as UNDEFINED to `NIXVM_SCAN_OUT`.
#[test]
fn scan_undefined_words() {
    let (Ok(path), Ok(out)) = (
        std::env::var("NIXVM_SCAN_WORDS"),
        std::env::var("NIXVM_SCAN_OUT"),
    ) else {
        return;
    };
    let text = std::fs::read_to_string(path).unwrap();
    let base = 0x10_0000u64;
    let mut mem = GuestMemory::new(base, 0x10_0000);
    mem.map(base, 0x10_0000, Prot::rw()).unwrap();
    let mut undefined = String::new();
    for line in text.lines() {
        let Some((hex, _)) = line.split_once(' ') else {
            continue;
        };
        let Ok(word) = u32::from_str_radix(hex, 16) else {
            continue;
        };
        let mut st = A64State::default();
        for x in &mut st.x {
            *x = base + 0x8000;
        }
        st.sp = base + 0x8000;
        st.pc = base;
        if a64_step(&mut st, word, &mut mem) == A64Step::Illegal {
            undefined.push_str(line);
            undefined.push('\n');
        }
    }
    std::fs::write(out, undefined).unwrap();
}
