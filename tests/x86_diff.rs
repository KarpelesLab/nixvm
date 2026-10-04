//! Differential test: random x86-64 instructions executed by nixvm's software
//! interpreter (`interp_x86`) and by a real x86-64 execution environment, with
//! the complete architectural state compared afterwards — GPRs, `RFLAGS`, the
//! x87/MMX/SSE state (as an `FXSAVE64` image), the upper halves of the YMM
//! registers and an 8 KiB memory window.
//!
//! The oracle is `tests/x86_oracle/harness.c`, compiled with
//! `clang -arch x86_64` and run under Rosetta 2 (`arch -x86_64`) on Apple
//! silicon. Each case's instruction bytes are copied between a prologue that
//! loads the exact initial state and an epilogue that captures the result; the
//! instruction sits at the same address in both worlds, so RIP-relative
//! operands and pushed return addresses agree. Faults are reported as the
//! Linux signal they would raise.
//!
//! Skips (passing) when the oracle can't be built or run — non-macOS hosts,
//! no clang, no Rosetta. A quick sample runs by default; heavier runs are
//! opt-in:
//!
//! ```text
//! NIXVM_X86_DIFF_CASES=200000 NIXVM_X86_DIFF_SEED=7 NIXVM_X86_DIFF_ONLY=x87 \
//!     cargo test --release --test x86_diff -- --nocapture
//! ```
//!
//! `NIXVM_X86_DIFF_ORACLE=native` runs the harness directly (an x86-64 host);
//! `NIXVM_X86_DIFF_REPORT=1` prints every mismatch group instead of failing.

// A test generator: index loops over opcode tables, lossy int->float for
// test values, and string building are all deliberate here.
#![allow(
    clippy::needless_range_loop,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::format_push_string,
    clippy::manual_range_patterns,
    clippy::match_same_arms
)]

use std::collections::BTreeMap;
use std::io::{BufReader, BufWriter, Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use nixvm::vcpu::GuestMemory;
use nixvm::vcpu::interp_x86::testing::{self, CpuState, Outcome};
use nixvm::vcpu::mem::Prot;

const BASE: u64 = 0x6_0000_0000;
const CODE_PAGE: u64 = BASE;
const DATA: u64 = BASE + 0x8000;
const DATA_LEN: usize = 0x2000;

// ---------------------------------------------------------------------------
// RNG

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // splitmix64
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn chance(&mut self, num: u64, den: u64) -> bool {
        self.below(den) < num
    }
    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len() as u64) as usize]
    }
    fn byte(&mut self) -> u8 {
        self.next() as u8
    }
}

// ---------------------------------------------------------------------------
// Oracle process

struct Oracle {
    child: Child,
    /// When the in-flight case started (`None` when idle), polled by a
    /// watchdog thread that kills the oracle if Rosetta livelocks on a case.
    busy: std::sync::Arc<std::sync::Mutex<Option<std::time::Instant>>>,
    tx: BufWriter<ChildStdin>,
    rx: BufReader<ChildStdout>,
    insn_addr: u64,
}

struct HwResult {
    signo: i32,
    gpr: [u64; 16],
    rflags: u64,
    fx: [u8; 512],
    ymm_hi: [u128; 16],
    data: Vec<u8>,
}

impl Oracle {
    fn start() -> Option<Self> {
        let mode = std::env::var("NIXVM_X86_DIFF_ORACLE").unwrap_or_default();
        let native = mode == "native";
        if !native && !cfg!(target_os = "macos") {
            eprintln!("x86_diff: no Rosetta oracle on this host; skipping");
            return None;
        }
        let src = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/x86_oracle/harness.c");
        let out = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("x86_oracle");
        static BUILT: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let built = *BUILT.get_or_init(|| {
            let mut cc = Command::new(if native { "cc" } else { "clang" });
            if !native {
                cc.args(["-arch", "x86_64"]);
            }
            cc.args(["-O1", "-w", "-o"])
                .arg(&out)
                .arg(src)
                .status()
                .is_ok_and(|s| s.success())
        });
        if !built {
            eprintln!("x86_diff: could not build the oracle harness; skipping");
            return None;
        }
        let mut cmd = if native {
            Command::new(&out)
        } else {
            let mut c = Command::new("arch");
            c.arg("-x86_64").arg(&out);
            c
        };
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .ok()?;
        let tx = BufWriter::new(child.stdin.take()?);
        let mut rx = BufReader::new(child.stdout.take()?);
        let mut hello = [0u8; 32];
        if rx.read_exact(&mut hello).is_err() {
            eprintln!("x86_diff: oracle did not start (no Rosetta?); skipping");
            return None;
        }
        let word = |i: usize| u64::from_le_bytes(hello[i * 8..i * 8 + 8].try_into().unwrap());
        assert_eq!(word(0), 0x6E69786F72616C65, "oracle handshake");
        assert_eq!(word(2), DATA);
        assert_eq!(word(3), DATA_LEN as u64);
        let busy = std::sync::Arc::new(std::sync::Mutex::new(None::<std::time::Instant>));
        let watch = busy.clone();
        let pid = child.id();
        std::thread::spawn(move || {
            loop {
                std::thread::sleep(std::time::Duration::from_millis(100));
                if std::sync::Arc::strong_count(&watch) == 1 {
                    return; // the oracle is gone
                }
                let started = *watch.lock().unwrap();
                if started.is_some_and(|t| t.elapsed() > std::time::Duration::from_secs(3)) {
                    let _ = Command::new("kill").arg("-9").arg(pid.to_string()).status();
                    return;
                }
            }
        });
        Some(Self {
            busy,
            child,
            tx,
            rx,
            insn_addr: word(1),
        })
    }

    /// Run one case; `None` if the oracle process itself died on it (Rosetta
    /// aborts on a few exotic faulting encodings) — it is restarted for the
    /// next case.
    fn run(&mut self, code: &[u8], st: &CpuState, data: &[u8], mmx: bool) -> Option<HwResult> {
        *self.busy.lock().unwrap() = Some(std::time::Instant::now());
        let r = self.try_run(code, st, data, mmx);
        *self.busy.lock().unwrap() = None;
        if r.is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
            *self = Self::start().expect("oracle restart");
        }
        r
    }

    fn try_run(&mut self, code: &[u8], st: &CpuState, data: &[u8], mmx: bool) -> Option<HwResult> {
        let w = &mut self.tx;
        let len = code.len() as u32 | (u32::from(mmx) << 31);
        w.write_all(&len.to_le_bytes()).ok()?;
        w.write_all(code).ok()?;
        for r in st.gpr {
            w.write_all(&r.to_le_bytes()).ok()?;
        }
        w.write_all(&st.rflags.to_le_bytes()).ok()?;
        w.write_all(&st.fxsave).ok()?;
        for v in st.ymm_hi {
            w.write_all(&v.to_le_bytes()).ok()?;
        }
        w.write_all(data).ok()?;
        w.flush().ok()?;
        let mut hdr = [0u8; 16];
        self.rx.read_exact(&mut hdr).ok()?;
        let mut g = [0u8; 128];
        self.rx.read_exact(&mut g).ok()?;
        let mut f = [0u8; 8];
        self.rx.read_exact(&mut f).ok()?;
        let mut fx = [0u8; 512];
        self.rx.read_exact(&mut fx).ok()?;
        let mut y = [0u8; 256];
        self.rx.read_exact(&mut y).ok()?;
        let mut ymm_hi = [0u128; 16];
        for (i, v) in ymm_hi.iter_mut().enumerate() {
            *v = u128::from_le_bytes(y[16 * i..16 * i + 16].try_into().unwrap());
        }
        let mut d = vec![0u8; DATA_LEN];
        self.rx.read_exact(&mut d).ok()?;
        let mut gpr = [0u64; 16];
        for (i, r) in gpr.iter_mut().enumerate() {
            *r = u64::from_le_bytes(g[i * 8..i * 8 + 8].try_into().unwrap());
        }
        Some(HwResult {
            signo: i32::from_le_bytes(hdr[0..4].try_into().unwrap()),
            gpr,
            rflags: u64::from_le_bytes(f),
            fx,
            ymm_hi,
            data: d,
        })
    }
}

// ---------------------------------------------------------------------------
// Instruction generator

/// Immediate operand following the opcode/ModRM.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Imm {
    None,
    /// imm8
    B,
    /// imm16/32 by operand size (imm32 sign-extended for 64-bit).
    Z,
    /// imm16/32/64 by operand size (`MOV r, imm`).
    V,
    /// `ENTER`'s imm16 + imm8.
    Enter,
    /// Only for `F6`/`F7`: `/0` and `/1` (TEST) take imm8/immZ.
    Grp3,
    /// A forward branch displacement (8/32-bit) into filler instructions
    /// appended after the branch, so taken and not-taken are distinguishable.
    Rel8,
    Rel32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Cat {
    Int,
    X87,
    Mmx,
    Sse,
    /// VEX-encoded SIMD (AVX, AVX2, FMA, F16C).
    Avx,
    /// VEX-encoded general-purpose (BMI1/BMI2), plus LZCNT/TZCNT/MOVBE.
    Bmi,
}

/// How a VEX instruction is encoded: `VEX.W`/`VEX.L` fixed or random, and
/// whether `VEX.vvvv` names a register (else it must be `1111`).
#[derive(Clone, Copy, Debug)]
struct VexSpec {
    w: Option<bool>,
    l: Option<bool>,
    nds: bool,
}

#[derive(Clone, Copy, Debug)]
struct OpSpec {
    cat: Cat,
    /// Escape bytes + opcode (`[0x0F, 0x58]`, `[0x01]`, …).
    op: &'static [u8],
    modrm: bool,
    imm: Imm,
    /// Mandatory prefix to emit (`0`, `0x66`, `0xF2`, `0xF3`).
    mand: u8,
    /// Restrict ModRM: `Some(true)` memory only, `Some(false)` register only.
    mem: Option<bool>,
    /// Restrict ModRM.reg to this value (opcode extension).
    reg: Option<u8>,
    /// VEX-encoded: `op` is then the logical opcode (`0F xx`, `0F 38 xx`,
    /// `0F 3A xx`, selecting the map) and `mand` the implied prefix.
    vex: Option<VexSpec>,
}

const fn op(cat: Cat, op: &'static [u8], modrm: bool, imm: Imm) -> OpSpec {
    OpSpec {
        cat,
        op,
        modrm,
        imm,
        mand: 0,
        mem: None,
        reg: None,
        vex: None,
    }
}

/// A VEX spec: `nds` = `vvvv` is a source, `l`/`w` fixed or random, `mem`
/// restricts the r/m form.
#[allow(clippy::too_many_arguments)]
fn vx(
    cat: Cat,
    mand: u8,
    opc: &'static [u8],
    imm: bool,
    nds: bool,
    l: Option<bool>,
    w: Option<bool>,
    mem: Option<bool>,
) -> OpSpec {
    OpSpec {
        mand,
        mem,
        vex: Some(VexSpec { w, l, nds }),
        ..op(cat, opc, true, if imm { Imm::B } else { Imm::None })
    }
}

fn specs() -> Vec<OpSpec> {
    use Cat::*;
    use Imm::*;
    let mut v = Vec::new();
    // ---- one-byte map, general-purpose ----
    static ONE: [[u8; 1]; 256] = {
        let mut t = [[0u8; 1]; 256];
        let mut i = 0;
        while i < 256 {
            t[i][0] = i as u8;
            i += 1;
        }
        t
    };
    for base in [0x00u8, 0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38] {
        for k in 0..4 {
            v.push(op(Int, &ONE[(base + k) as usize], true, None));
        }
        v.push(op(Int, &ONE[(base + 4) as usize], false, B));
        v.push(op(Int, &ONE[(base + 5) as usize], false, Z));
    }
    // Conditional branches, LOOP/LOOPE/LOOPNE/JrCXZ, short/near JMP.
    for o in 0x70..=0x7F {
        v.push(op(Int, &ONE[o], false, Rel8));
    }
    for o in 0xE0..=0xE3 {
        v.push(op(Int, &ONE[o], false, Rel8));
    }
    v.push(op(Int, &ONE[0xEB], false, Rel8));
    v.push(op(Int, &ONE[0xE9], false, Rel32));
    for o in 0x80..=0x8F {
        v.push(op(Int, &TWO[o], false, Rel32));
    }
    for o in 0x50..=0x5F {
        v.push(op(Int, &ONE[o], false, None));
    }
    // MOVSXD without REX.W: Rosetta sign-extends into the upper half; the SDM
    // (and hardware) zero-extend the 32-bit destination. Only REX.W forms.
    v.push(op(Int, &ONE[0x63], true, None));
    v.push(op(Int, &ONE[0x68], false, Z));
    v.push(op(Int, &ONE[0x69], true, Z));
    v.push(op(Int, &ONE[0x6A], false, B));
    v.push(op(Int, &ONE[0x6B], true, B));
    v.push(op(Int, &ONE[0x80], true, B));
    v.push(op(Int, &ONE[0x81], true, Z));
    v.push(op(Int, &ONE[0x83], true, B));
    for o in 0x84..=0x8B {
        v.push(op(Int, &ONE[o], true, None));
    }
    // MOV r/m, Sreg: only DS/ES/FS/GS (CS/SS hold macOS selectors under
    // Rosetta).
    for r in [0u8, 3, 4, 5] {
        v.push(OpSpec {
            reg: Some(r),
            ..op(Int, &ONE[0x8C], true, None)
        });
    }
    v.push(op(Int, &ONE[0x8D], true, None));
    v.push(op(Int, &ONE[0x8F], true, None));
    for o in 0x90..=0x99 {
        v.push(op(Int, &ONE[o], false, None));
    }
    // (POPF is unit-tested: a random popped TF would single-step.)
    for o in [0x9C, 0x9E, 0x9F] {
        v.push(op(Int, &ONE[o], false, None));
    }
    for o in 0xA4..=0xA7 {
        v.push(op(Int, &ONE[o], false, None));
    }
    v.push(op(Int, &ONE[0xA8], false, B));
    v.push(op(Int, &ONE[0xA9], false, Z));
    for o in 0xAA..=0xAF {
        v.push(op(Int, &ONE[o], false, None));
    }
    for o in 0xB0..=0xB7 {
        v.push(op(Int, &ONE[o], false, B));
    }
    for o in 0xB8..=0xBF {
        v.push(op(Int, &ONE[o], false, V));
    }
    v.push(op(Int, &ONE[0xC0], true, B));
    v.push(op(Int, &ONE[0xC1], true, B));
    v.push(op(Int, &ONE[0xC6], true, B));
    v.push(op(Int, &ONE[0xC7], true, Z));
    v.push(op(Int, &ONE[0xC8], false, Enter));
    v.push(op(Int, &ONE[0xC9], false, None));
    v.push(op(Int, &ONE[0xCC], false, None));
    for o in 0xD0..=0xD3 {
        v.push(op(Int, &ONE[o], true, None));
    }
    v.push(op(Int, &ONE[0xD7], false, None));
    v.push(op(Int, &ONE[0xF4], false, None));
    v.push(op(Int, &ONE[0xF5], false, None));
    v.push(op(Int, &ONE[0xF6], true, Grp3));
    v.push(op(Int, &ONE[0xF7], true, Grp3));
    for o in 0xF8..=0xFD {
        v.push(op(Int, &ONE[o], false, None));
    }
    v.push(op(Int, &ONE[0xFE], true, None));
    for r in [0u8, 1, 6] {
        v.push(OpSpec {
            reg: Some(r),
            ..op(Int, &ONE[0xFF], true, None)
        });
    }
    // ---- two-byte map, general-purpose ----
    static TWO: [[u8; 2]; 256] = {
        let mut t = [[0u8; 2]; 256];
        let mut i = 0;
        while i < 256 {
            t[i][0] = 0x0F;
            t[i][1] = i as u8;
            i += 1;
        }
        t
    };
    v.push(op(Int, &TWO[0x0B], false, None)); // UD2
    // (The hint-NOP space 0F 0D/18-1F is unit-tested: Rosetta raises #UD
    // for several reserved-NOP encodings real CPUs execute.)
    for o in 0x40..=0x4F {
        v.push(op(Int, &TWO[o], true, None));
    }
    for o in 0x90..=0x9F {
        v.push(op(Int, &TWO[o], true, None));
    }
    v.push(op(Int, &TWO[0xA0], false, None)); // push fs
    v.push(op(Int, &TWO[0xA8], false, None)); // push gs
    for o in [
        0xA3, 0xAB, 0xB3, 0xBB, 0xAF, 0xB0, 0xB1, 0xB6, 0xB7, 0xBE, 0xBF, 0xBC, 0xBD,
    ] {
        v.push(op(Int, &TWO[o], true, None));
    }
    v.push(op(Int, &TWO[0xA4], true, B));
    v.push(op(Int, &TWO[0xA5], true, None));
    v.push(op(Int, &TWO[0xAC], true, B));
    v.push(op(Int, &TWO[0xAD], true, None));
    v.push(op(Int, &TWO[0xBA], true, B));
    v.push(op(Int, &TWO[0xC0], true, None));
    v.push(op(Int, &TWO[0xC1], true, None));
    v.push(OpSpec {
        mem: Some(true),
        ..op(Int, &TWO[0xC3], true, None)
    });
    v.push(OpSpec {
        reg: Some(1),
        ..op(Int, &TWO[0xC7], true, None)
    });
    for o in 0xC8..=0xCF {
        v.push(op(Int, &TWO[o], false, None));
    }
    v.push(OpSpec {
        mand: 0xF3,
        ..op(Int, &TWO[0xB8], true, None)
    });
    // FXSAVE/FXRSTOR, STMXCSR, CLFLUSH (memory) and the fences (register).
    // (LDMXCSR is unit-tested: Rosetta doesn't #GP on reserved bits and
    // refuses to unmask exceptions; XSAVE/XRSTOR are generated with valid
    // areas by gen_xstate.)
    // (FXRSTOR from random memory would mostly #GP or unmask exceptions,
    // which Rosetta doesn't model; it is unit-tested.)
    for r in [0u8, 3, 7] {
        v.push(OpSpec {
            reg: Some(r),
            mem: Some(true),
            ..op(Int, &TWO[0xAE], true, None)
        });
    }
    for r in [5u8, 6, 7] {
        v.push(OpSpec {
            reg: Some(r),
            mem: Some(false),
            ..op(Int, &TWO[0xAE], true, None)
        });
    }

    // ---- x87 ----
    for o in 0xD8..=0xDF {
        v.push(op(X87, &ONE[o], true, None));
    }
    v.push(op(X87, &ONE[0x9B], false, None));

    // ---- MMX (no mandatory prefix) / SSE / SSE2 ----
    let mmx_ops: &[usize] = &[
        0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x6B, 0x6E, 0x6F, 0x71,
        0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x7E, 0x7F, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD8, 0xD9,
        0xDB, 0xDC, 0xDD, 0xDF, 0xE1, 0xE2, 0xE5, 0xE8, 0xE9, 0xEB, 0xEC, 0xED, 0xEF, 0xF1, 0xF2,
        0xF3, 0xF4, 0xF5, 0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE,
        // SSE integer extensions on MMX registers
        0x70, 0xC4, 0xC5, 0xD7, 0xDA, 0xDE, 0xE0, 0xE3, 0xE4, 0xE7, 0xEA, 0xEE, 0xF6, 0xF7,
        // cvt between MMX and packed float
        0x2A, 0x2C, 0x2D,
    ];
    for &o in mmx_ops {
        let imm = if matches!(o, 0x70 | 0x71 | 0x72 | 0x73 | 0xC4 | 0xC5) {
            B
        } else {
            None
        };
        v.push(op(Mmx, &TWO[o], o != 0x77, imm));
    }
    // SSE/SSE2 two-byte opcodes with each mandatory prefix.
    let sse_ops: &[usize] = &[
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x28, 0x29, 0x2A, 0x2B, 0x2C, 0x2D, 0x2E,
        0x2F, 0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x5B, 0x5C, 0x5D,
        0x5E, 0x5F, 0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x6B, 0x6C,
        0x6D, 0x6E, 0x6F, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x7E, 0x7F, 0xC2, 0xC4, 0xC5,
        0xC6, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xDB, 0xDC, 0xDD, 0xDE,
        0xDF, 0xE0, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xEB, 0xEC, 0xED,
        0xEE, 0xEF, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD,
        0xFE,
    ];
    for &o in sse_ops {
        let imm = if matches!(o, 0x70 | 0x71 | 0x72 | 0x73 | 0xC2 | 0xC4 | 0xC5 | 0xC6) {
            B
        } else {
            None
        };
        for mand in [0u8, 0x66, 0xF2, 0xF3] {
            if mand == 0 && mmx_ops.contains(&o) {
                continue; // an MMX instruction: the Mmx category runs it in MMX mode
            }
            // (The SSE3 MOVSLDUP/MOVSHDUP/MOVDDUP forms are listed with the v2
            // extensions below.)
            if matches!((mand, o), (0xF3 | 0xF2, 0x12) | (0xF3, 0x16)) {
                continue;
            }
            // SSE2 forms that read or write MMX registers.
            let touches_mmx = matches!((mand, o), (0x66, 0x2A | 0x2C | 0x2D) | (0xF2 | 0xF3, 0xD6));
            v.push(OpSpec {
                mand,
                ..op(if touches_mmx { Mmx } else { Sse }, &TWO[o], true, imm)
            });
        }
    }
    v.push(OpSpec {
        mand: 0x66,
        mem: Some(false),
        ..op(Sse, &TWO[0xF7], true, None)
    }); // MASKMOVDQU

    // ---- x86-64-v2: SSE3, SSSE3, SSE4.1, SSE4.2 ----
    static T38: [[u8; 3]; 256] = {
        let mut t = [[0u8; 3]; 256];
        let mut i = 0;
        while i < 256 {
            t[i] = [0x0F, 0x38, i as u8];
            i += 1;
        }
        t
    };
    static T3A: [[u8; 3]; 256] = {
        let mut t = [[0u8; 3]; 256];
        let mut i = 0;
        while i < 256 {
            t[i] = [0x0F, 0x3A, i as u8];
            i += 1;
        }
        t
    };
    for (mand, o) in [
        (0x66u8, 0xD0usize),
        (0xF2, 0xD0),
        (0x66, 0x7C),
        (0xF2, 0x7C),
        (0x66, 0x7D),
        (0xF2, 0x7D),
        (0xF3, 0x12),
        (0xF3, 0x16),
        (0xF2, 0x12),
    ] {
        v.push(OpSpec {
            mand,
            ..op(Sse, &TWO[o], true, None)
        });
    }
    v.push(OpSpec {
        mand: 0xF2,
        mem: Some(true),
        ..op(Sse, &TWO[0xF0], true, None)
    }); // LDDQU
    for o in (0x00..=0x0B).chain(0x1C..=0x1E) {
        v.push(op(Mmx, &T38[o], true, None));
        v.push(OpSpec {
            mand: 0x66,
            ..op(Sse, &T38[o], true, None)
        });
    }
    v.push(op(Mmx, &T3A[0x0F], true, B));
    v.push(OpSpec {
        mand: 0x66,
        ..op(Sse, &T3A[0x0F], true, B)
    });
    for o in [
        0x10usize, 0x14, 0x15, 0x17, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x28, 0x29, 0x2B, 0x30,
        0x31, 0x32, 0x33, 0x34, 0x35, 0x37, 0x38, 0x39, 0x3A, 0x3B, 0x3C, 0x3D, 0x3E, 0x3F, 0x40,
        0x41,
    ] {
        v.push(OpSpec {
            mand: 0x66,
            ..op(Sse, &T38[o], true, None)
        });
    }
    v.push(OpSpec {
        mand: 0x66,
        mem: Some(true),
        ..op(Sse, &T38[0x2A], true, None)
    }); // MOVNTDQA
    for o in [
        0x08usize, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x14, 0x15, 0x16, 0x17, 0x20, 0x21, 0x22,
        0x40, 0x41, 0x42, 0x60, 0x61, 0x62, 0x63,
    ] {
        v.push(OpSpec {
            mand: 0x66,
            ..op(Sse, &T3A[o], true, B)
        });
    }
    for o in [0xF0usize, 0xF1] {
        v.push(OpSpec {
            mand: 0xF2,
            ..op(Int, &T38[o], true, None)
        });
    } // CRC32

    // ---- x86-64-v3: AVX, AVX2, FMA, F16C (VEX) ----
    let (l0, l1, w0, w1) = (Some(false), Some(true), Some(false), Some(true));
    let (memo, rego) = (Some(true), Some(false));
    let any = Option::<bool>::None;
    // 0F map, 66 integer ops (NDS).
    for o in (0x60..=0x6D)
        .chain(0x74..=0x76)
        .chain(0xD1..=0xD5)
        .chain(0xD8..=0xE5)
        .chain(0xE8..=0xEF)
        .chain(0xF1..=0xF6)
        .chain(0xF8..=0xFE)
    {
        v.push(vx(Avx, 0x66, &TWO[o], false, true, any, any, any));
    }
    for mand in [0u8, 0x66] {
        for o in [
            0x14, 0x15, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5C, 0x5D, 0x5E, 0x5F,
        ] {
            v.push(vx(Avx, mand, &TWO[o], false, true, any, any, any));
        }
        v.push(vx(Avx, mand, &TWO[0xC6], true, true, any, any, any));
        v.push(vx(Avx, mand, &TWO[0xC2], true, true, any, any, any));
        for o in [0x10, 0x11, 0x28, 0x29, 0x51, 0x2E, 0x2F] {
            v.push(vx(Avx, mand, &TWO[o], false, false, any, any, any));
        }
        v.push(vx(Avx, mand, &TWO[0x2B], false, false, any, any, memo));
        v.push(vx(Avx, mand, &TWO[0x50], false, false, any, any, rego));
        v.push(vx(Avx, mand, &TWO[0x5B], false, false, any, any, any));
        v.push(vx(Avx, mand, &TWO[0x5A], false, false, any, any, any));
        for o in [0x12, 0x16] {
            v.push(vx(
                Avx,
                mand,
                &TWO[o],
                false,
                true,
                l0,
                any,
                if mand == 0 { any } else { memo },
            ));
        }
        for o in [0x13, 0x17] {
            v.push(vx(Avx, mand, &TWO[o], false, false, l0, any, memo));
        }
    }
    for mand in [0xF3u8, 0xF2] {
        for o in [
            0x51, 0x58, 0x59, 0x5A, 0x5C, 0x5D, 0x5E, 0x5F, 0x2A, 0x10, 0x11,
        ] {
            v.push(vx(Avx, mand, &TWO[o], false, true, any, any, any));
        }
        v.push(vx(Avx, mand, &TWO[0xC2], true, true, any, any, any));
        for o in [0x2C, 0x2D] {
            v.push(vx(Avx, mand, &TWO[o], false, false, any, any, any));
        }
    }
    v.push(vx(Avx, 0xF3, &TWO[0x5B], false, false, any, any, any));
    v.push(vx(Avx, 0, &TWO[0x52], false, false, any, any, any));
    v.push(vx(Avx, 0, &TWO[0x53], false, false, any, any, any));
    v.push(vx(Avx, 0xF3, &TWO[0x52], false, true, any, any, any));
    v.push(vx(Avx, 0xF3, &TWO[0x53], false, true, any, any, any));
    for mand in [0x66u8, 0xF2] {
        for o in [0xD0, 0x7C, 0x7D] {
            v.push(vx(Avx, mand, &TWO[o], false, true, any, any, any));
        }
    }
    for mand in [0x66u8, 0xF3] {
        v.push(vx(Avx, mand, &TWO[0x6F], false, false, any, any, any));
        v.push(vx(Avx, mand, &TWO[0x7F], false, false, any, any, any));
    }
    for mand in [0x66u8, 0xF3, 0xF2] {
        v.push(vx(Avx, mand, &TWO[0x70], true, false, any, any, any));
        v.push(vx(Avx, mand, &TWO[0xE6], false, false, any, any, any));
    }
    v.push(vx(Avx, 0xF3, &TWO[0x12], false, false, any, any, any));
    v.push(vx(Avx, 0xF3, &TWO[0x16], false, false, any, any, any));
    v.push(vx(Avx, 0xF2, &TWO[0x12], false, false, any, any, any));
    for o in [0x6E, 0x7E, 0xD6] {
        v.push(vx(Avx, 0x66, &TWO[o], false, false, l0, any, any));
    }
    v.push(vx(Avx, 0xF3, &TWO[0x7E], false, false, l0, any, any));
    for (o, exts) in [
        (0x71usize, &[2u8, 4, 6][..]),
        (0x72, &[2, 4, 6]),
        (0x73, &[2, 3, 6, 7]),
    ] {
        for &r in exts {
            v.push(OpSpec {
                reg: Some(r),
                ..vx(Avx, 0x66, &TWO[o], true, true, any, any, rego)
            });
        }
    }
    v.push(vx(Avx, 0x66, &TWO[0xC4], true, true, l0, any, any));
    v.push(vx(Avx, 0x66, &TWO[0xC5], true, false, l0, any, rego));
    v.push(vx(Avx, 0x66, &TWO[0xD7], false, false, any, any, rego));
    v.push(vx(Avx, 0x66, &TWO[0xE7], false, false, any, any, memo));
    v.push(vx(Avx, 0xF2, &TWO[0xF0], false, false, any, any, memo));
    v.push(vx(Avx, 0x66, &TWO[0xF7], false, false, l0, any, rego));
    v.push(OpSpec {
        reg: Some(3),
        ..vx(Avx, 0, &TWO[0xAE], false, false, l0, any, memo)
    });
    v.push(OpSpec {
        modrm: false,
        ..vx(Avx, 0, &TWO[0x77], false, false, any, any, any)
    });
    // 0F 38 map (66).
    for o in (0x00..=0x0B).chain([0x28, 0x29, 0x2B]).chain(0x37..=0x40) {
        v.push(vx(Avx, 0x66, &T38[o], false, true, any, any, any));
    }
    for o in (0x1C..=0x1E)
        .chain(0x20..=0x25)
        .chain(0x30..=0x35)
        .chain([0x17])
    {
        v.push(vx(Avx, 0x66, &T38[o], false, false, any, any, any));
    }
    for o in [0x0C, 0x0D] {
        v.push(vx(Avx, 0x66, &T38[o], false, true, any, w0, any));
    }
    for o in [0x0E, 0x0F, 0x13, 0x18, 0x58, 0x59, 0x78, 0x79] {
        v.push(vx(Avx, 0x66, &T38[o], false, false, any, w0, any));
    }
    v.push(vx(Avx, 0x66, &T38[0x19], false, false, l1, w0, any));
    v.push(vx(Avx, 0x66, &T38[0x1A], false, false, l1, w0, memo));
    v.push(vx(Avx, 0x66, &T38[0x5A], false, false, l1, w0, memo));
    v.push(vx(Avx, 0x66, &T38[0x16], false, true, l1, w0, any));
    v.push(vx(Avx, 0x66, &T38[0x36], false, true, l1, w0, any));
    v.push(vx(Avx, 0x66, &T38[0x2A], false, false, any, any, memo));
    for o in [0x2C, 0x2D, 0x2E, 0x2F] {
        v.push(vx(Avx, 0x66, &T38[o], false, true, any, w0, memo));
    }
    for o in [0x8C, 0x8E] {
        v.push(vx(Avx, 0x66, &T38[o], false, true, any, any, memo));
    }
    v.push(vx(Avx, 0x66, &T38[0x41], false, false, l0, any, any));
    v.push(vx(Avx, 0x66, &T38[0x45], false, true, any, any, any));
    v.push(vx(Avx, 0x66, &T38[0x46], false, true, any, w0, any));
    v.push(vx(Avx, 0x66, &T38[0x47], false, true, any, any, any));
    for o in 0x90..=0x93 {
        v.push(vx(Avx, 0x66, &T38[o], false, true, any, any, memo));
    }
    for o in (0x96..=0x9F).chain(0xA6..=0xAF).chain(0xB6..=0xBF) {
        v.push(vx(Avx, 0x66, &T38[o], false, true, any, any, any));
    }
    // 0F 3A map (66, all with imm8).
    for o in [0x00, 0x01] {
        v.push(vx(Avx, 0x66, &T3A[o], true, false, l1, w1, any));
    }
    for o in [0x02, 0x4A, 0x4B, 0x4C] {
        v.push(vx(Avx, 0x66, &T3A[o], true, true, any, w0, any));
    }
    for o in [0x04, 0x05, 0x1D] {
        v.push(vx(Avx, 0x66, &T3A[o], true, false, any, w0, any));
    }
    for o in [0x06, 0x18, 0x38, 0x46] {
        v.push(vx(Avx, 0x66, &T3A[o], true, true, l1, w0, any));
    }
    for o in [0x19, 0x39] {
        v.push(vx(Avx, 0x66, &T3A[o], true, false, l1, w0, any));
    }
    for o in [0x08, 0x09] {
        v.push(vx(Avx, 0x66, &T3A[o], true, false, any, any, any));
    }
    for o in [0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x40, 0x42] {
        v.push(vx(Avx, 0x66, &T3A[o], true, true, any, any, any));
    }
    for o in [0x14, 0x15, 0x16, 0x17, 0x60, 0x61, 0x62, 0x63] {
        v.push(vx(Avx, 0x66, &T3A[o], true, false, l0, any, any));
    }
    for o in [0x20, 0x21, 0x22, 0x41] {
        v.push(vx(Avx, 0x66, &T3A[o], true, true, l0, any, any));
    }
    // XSAVE/XRSTOR and XGETBV (see gen_xstate).
    for r in [4u8, 5] {
        v.push(OpSpec {
            reg: Some(r),
            mem: Some(true),
            ..op(Avx, &TWO[0xAE], true, None)
        });
    }
    v.push(OpSpec {
        reg: Some(2),
        ..op(Avx, &TWO[0x01], true, None)
    });
    // ---- BMI1/BMI2 (VEX, general purpose), LZCNT/TZCNT, MOVBE ----
    v.push(vx(Bmi, 0, &T38[0xF2], false, true, l0, any, any));
    for r in [1u8, 2, 3] {
        v.push(OpSpec {
            reg: Some(r),
            ..vx(Bmi, 0, &T38[0xF3], false, true, l0, any, any)
        });
    }
    for mand in [0u8, 0xF3, 0xF2] {
        v.push(vx(Bmi, mand, &T38[0xF5], false, true, l0, any, any));
    }
    v.push(vx(Bmi, 0xF2, &T38[0xF6], false, true, l0, any, any));
    for mand in [0u8, 0x66, 0xF3, 0xF2] {
        v.push(vx(Bmi, mand, &T38[0xF7], false, true, l0, any, any));
    }
    v.push(vx(Bmi, 0xF2, &T3A[0xF0], true, false, l0, any, any));
    for o in [0xBC, 0xBD] {
        v.push(OpSpec {
            mand: 0xF3,
            ..op(Bmi, &TWO[o], true, None)
        });
    }
    for o in [0xF0, 0xF1] {
        v.push(OpSpec {
            mem: Some(true),
            ..op(Bmi, &T38[o], true, None)
        });
    }
    v
}

/// "Interesting" 64-bit integers: boundaries at every operand width.
fn int_value(rng: &mut Rng) -> u64 {
    match rng.below(10) {
        0 => *rng.pick(&[
            0,
            1,
            2,
            u64::MAX,
            0x7f,
            0x80,
            0xff,
            0x7fff,
            0x8000,
            0xffff,
            0x7fff_ffff,
            0x8000_0000,
            0xffff_ffff,
            0x7fff_ffff_ffff_ffff,
            0x8000_0000_0000_0000,
        ]),
        1 => rng.below(64),
        2 => rng.below(0x400),
        3 => (rng.next() as i8) as i64 as u64,
        4 => (rng.next() as i16) as i64 as u64,
        5 => (rng.next() as i32) as i64 as u64,
        6 => rng.next() & 0xffff_ffff,
        _ => rng.next(),
    }
}

/// An "interesting" 32-bit float bit pattern.
fn f32_bits(rng: &mut Rng) -> u32 {
    match rng.below(12) {
        0 => *rng.pick(&[
            0x0000_0000,
            0x8000_0000,
            0x3f80_0000,
            0xbf80_0000,
            0x7f80_0000,
            0xff80_0000,
            0x7fc0_0000,
            0xffc0_0000,
            0x7fa0_0000,
            0x7f80_0001,
            0x0000_0001,
            0x807f_ffff,
            0x0080_0000,
            0x7f7f_ffff,
            0x4f00_0000,
            0xcf00_0000,
            0x5f00_0000,
            0x3f00_0000,
            0x4b00_0000,
        ]),
        1 => rng.next() as u32 & 0x807f_ffff, // denormal
        2 => (rng.next() as u32 & 0x807f_ffff) | 0x7f80_0000, // inf/nan
        3..=6 => {
            // moderate magnitude
            let e = 100 + rng.below(60) as u32;
            (rng.next() as u32 & 0x807f_ffff) | (e << 23)
        }
        7 => (rng.below(1 << 20) as f32).to_bits() ^ ((rng.next() as u32) & 0x8000_0000),
        _ => rng.next() as u32,
    }
}

/// An "interesting" 64-bit float bit pattern.
fn f64_bits(rng: &mut Rng) -> u64 {
    match rng.below(12) {
        0 => *rng.pick(&[
            0,
            0x8000_0000_0000_0000,
            0x3ff0_0000_0000_0000,
            0xbff0_0000_0000_0000,
            0x7ff0_0000_0000_0000,
            0xfff0_0000_0000_0000,
            0x7ff8_0000_0000_0000,
            0xfff8_0000_0000_0000,
            0x7ff4_0000_0000_0000,
            0x7ff0_0000_0000_0001,
            1,
            0x800f_ffff_ffff_ffff,
            0x0010_0000_0000_0000,
            0x7fef_ffff_ffff_ffff,
            0x43e0_0000_0000_0000,
            0xc3e0_0000_0000_0000,
            0x41e0_0000_0000_0000,
            0xc1e0_0000_0000_0000,
            0x3fe0_0000_0000_0000,
            0x4330_0000_0000_0000,
        ]),
        1 => rng.next() & 0x800f_ffff_ffff_ffff,
        2 => (rng.next() & 0x800f_ffff_ffff_ffff) | 0x7ff0_0000_0000_0000,
        3..=6 => {
            let e = 990 + rng.below(80);
            (rng.next() & 0x800f_ffff_ffff_ffff) | (e << 52)
        }
        7 => (rng.below(1 << 40) as f64).to_bits() ^ (rng.next() & 0x8000_0000_0000_0000),
        8 => f64::from(f32::from_bits(f32_bits(rng))).to_bits(),
        _ => rng.next(),
    }
}

/// An "interesting" 80-bit extended value.
fn f80_bits(rng: &mut Rng) -> u128 {
    let sign = u128::from(rng.next() & 1) << 79;
    let v = match rng.below(14) {
        0 => *rng.pick(&[
            0u128,
            0x3fff_8000_0000_0000_0000,
            0x7fff_8000_0000_0000_0000,
            0x7fff_c000_0000_0000_0000,
            0x7fff_a000_0000_0000_0000,
            0x7fff_8000_0000_0000_0001,
            0x0000_0000_0000_0000_0001,
            0x0001_8000_0000_0000_0000,
            0x7ffe_ffff_ffff_ffff_ffff,
            0x403e_8000_0000_0000_0000,
            0x403d_ffff_ffff_ffff_fffe,
            0x4000_c90f_daa2_2168_c235,
            0x3ffe_8000_0000_0000_0000,
        ]),
        // denormal / pseudo-denormal
        1 => u128::from(rng.next() & 0x7fff_ffff_ffff_ffff) | (u128::from(rng.below(2)) << 63),
        // (Unnormals and pseudo-NaN/-infinity — invalid operands per the SDM
        // — are unit-tested only: Rosetta treats them as ordinary numbers.)
        2 | 3 | 4 => (0x7fffu128 << 64) | u128::from(rng.next() | (1 << 63)),
        // from a double
        5 | 6 => {
            let d = f64::from_bits(f64_bits(rng));
            f64_to_f80(d)
        }
        // integers
        7 => f64_to_f80((rng.below(1 << 40) as f64) - (1u64 << 39) as f64),
        _ => {
            let e = 0x3fff - 70 + rng.below(140);
            (u128::from(e) << 64) | u128::from(rng.next() | (1 << 63))
        }
    };
    v | sign
}

fn f64_to_f80(d: f64) -> u128 {
    let b = d.to_bits();
    let sign = u128::from(b >> 63) << 79;
    let e = ((b >> 52) & 0x7ff) as i32;
    let m = b & 0xf_ffff_ffff_ffff;
    let body = if e == 0x7ff {
        (0x7fffu128 << 64) | u128::from((1 << 63) | (m << 11))
    } else if e == 0 {
        if m == 0 {
            0
        } else {
            let lz = m.leading_zeros() - 11; // shift to make bit 52 the msb
            let mm = (m << lz) << 11;
            let ee = 1 - 1023 - lz as i32 + 16383;
            ((ee as u128) << 64) | u128::from(mm)
        }
    } else {
        (((e - 1023 + 16383) as u128) << 64) | u128::from((1 << 63) | (m << 11))
    };
    sign | body
}

/// A random 128-bit XMM value made of float lanes of one kind.
fn xmm_value(rng: &mut Rng) -> u128 {
    match rng.below(4) {
        0 => {
            let mut v = 0u128;
            for i in 0..4 {
                v |= u128::from(f32_bits(rng)) << (32 * i);
            }
            v
        }
        1 => u128::from(f64_bits(rng)) | (u128::from(f64_bits(rng)) << 64),
        2 => {
            let mut v = 0u128;
            for i in 0..2 {
                v |= u128::from(int_value(rng)) << (64 * i);
            }
            v
        }
        _ => u128::from(rng.next()) | (u128::from(rng.next()) << 64),
    }
}

struct Case {
    code: Vec<u8>,
    st: CpuState,
    data: Vec<u8>,
    spec: OpSpec,
    /// Where the ModRM byte is in `code` (when the opcode has one).
    modrm_at: Option<usize>,
    /// Memory bytes (address, length) not compared.
    mem_ignore: Vec<(u64, usize)>,
}

/// Build the FXSAVE image for a random but valid x87/SSE state.
fn random_fx(rng: &mut Rng, cat: Cat) -> [u8; 512] {
    let mut fx = [0u8; 512];
    // FCW: all exceptions masked, random PC and RC.
    let pc = *rng.pick(&[0u16, 2, 3, 3]);
    let rc = rng.below(4) as u16;
    let fcw = 0x7f | 0x40 | (pc << 8) | (rc << 10);
    fx[0..2].copy_from_slice(&fcw.to_le_bytes());
    // MMX code runs in MMX mode (TOP = 0, all tags valid): Rosetta does
    // not model the x87→MMX transition, so start there.
    let top = if cat == Cat::Mmx {
        0
    } else {
        rng.below(8) as u16
    };
    let flags = if rng.chance(1, 2) {
        0
    } else {
        rng.below(0x40) as u16
    };
    // Condition codes start clear: Rosetta leaves several codes the SDM
    // defines as cleared (or undefined) untouched.
    let cc = 0u16;
    let fsw = flags | (top << 11) | ((cc & 7) << 8) | ((cc >> 3) << 14);
    fx[2..4].copy_from_slice(&fsw.to_le_bytes());
    let tags = match cat {
        Cat::X87 => match rng.below(4) {
            0 => 0xff,
            1 => 0,
            _ => rng.byte(),
        },
        Cat::Mmx => 0xff,
        _ => rng.byte(),
    };
    fx[4] = tags;
    // MXCSR: masked exceptions, random RC/FTZ/DAZ, sometimes sticky flags.
    let mut mx: u32 = 0x1f80 | ((rng.below(4) as u32) << 13);
    if rng.chance(1, 4) {
        mx |= 1 << 15; // FTZ
    }
    if rng.chance(1, 4) {
        mx |= 1 << 6; // DAZ
    }
    if rng.chance(1, 3) {
        mx |= rng.below(0x40) as u32;
    }
    fx[24..28].copy_from_slice(&mx.to_le_bytes());
    for i in 0..8 {
        let v = if cat == Cat::Mmx {
            u128::from(int_value(rng)) | (0xffffu128 << 64)
        } else {
            f80_bits(rng)
        };
        fx[32 + 16 * i..42 + 16 * i].copy_from_slice(&v.to_le_bytes()[..10]);
    }
    for i in 0..16 {
        let v = xmm_value(rng);
        fx[160 + 16 * i..176 + 16 * i].copy_from_slice(&v.to_le_bytes());
    }
    fx
}

/// Opcodes LOCK may legally prefix (with a memory destination). Rosetta
/// executes some illegal LOCK forms that real CPUs reject with #UD, so only
/// these get a LOCK in generated cases.
fn lockable(spec: &OpSpec) -> bool {
    match spec.op {
        [0x0F, o] => matches!(
            o,
            0xAB | 0xB3 | 0xBB | 0xB0 | 0xB1 | 0xC0 | 0xC1 | 0xBA | 0xC7
        ),
        [o] => matches!(
            o,
            0x00 | 0x01
                | 0x08
                | 0x09
                | 0x10
                | 0x11
                | 0x18
                | 0x19
                | 0x20
                | 0x21
                | 0x28
                | 0x29
                | 0x30
                | 0x31
                | 0x86
                | 0x87
                | 0x80
                | 0x81
                | 0x83
                | 0xF6
                | 0xF7
                | 0xFE
                | 0xFF
        ),
        _ => false,
    }
}

fn gen_case(rng: &mut Rng, spec: &OpSpec, insn_addr: u64) -> Case {
    if spec.cat == Cat::Avx && spec.vex.is_none() {
        return gen_xstate(rng, spec, insn_addr);
    }
    let mut st = CpuState::default();
    for r in &mut st.gpr {
        *r = int_value(rng);
    }
    // RSP always points into the data window (push/pop/enter/leave).
    st.gpr[4] = DATA + 0x1000 + rng.below(0x40) * 8;
    if rng.chance(1, 3) {
        st.gpr[5] = DATA + 0x1400 + rng.below(0x40) * 8; // RBP for LEAVE/ENTER
    }
    // RFLAGS: random status flags + DF.
    let status = rng.next() & 0x8d5;
    let df = if rng.chance(1, 4) { 1 << 10 } else { 0 };
    st.rflags = 0x202 | status | df;
    st.fxsave = random_fx(rng, spec.cat);
    for v in &mut st.ymm_hi {
        *v = xmm_value(rng);
    }

    let mut code = Vec::new();
    // Prefixes.
    let mut has66 = false;
    let no66 = matches!(spec.op, [0xC8 | 0xC9] | [0x0F, 0xC3 | 0xAE])
        || matches!(spec.imm, Imm::Rel8 | Imm::Rel32);
    if spec.cat == Cat::Int {
        if !no66 && rng.chance(1, 5) {
            code.push(0x66);
            has66 = true;
        }
        // REP/REPNE only where they mean something: Rosetta raises #UD for a
        // stray F2/F3 that real CPUs ignore.
        let string_op = spec.op.len() == 1 && matches!(spec.op[0], 0xA4..=0xA7 | 0xAA..=0xAF);
        if string_op && rng.chance(1, 2) {
            code.push(*rng.pick(&[0xF2u8, 0xF3]));
        }
        if rng.chance(1, 16) && lockable(spec) {
            code.push(0xF0);
        }
    }
    if rng.chance(1, 20) {
        code.push(*rng.pick(&[0x2Eu8, 0x3E, 0x26, 0x36]));
    }
    let (rex_w, rex_x, rex_b);
    if let Some(vs) = spec.vex {
        // VEX: R/X/B random (REX-like), W/L as the spec says or random,
        // vvvv a random register or unused (1111).
        let (map, opc) = match spec.op {
            [0x0F, 0x38, o] => (2u8, *o),
            [0x0F, 0x3A, o] => (3, *o),
            [0x0F, o] => (1, *o),
            _ => unreachable!("VEX spec opcode"),
        };
        let (r, x, b) = (rng.chance(1, 2), rng.chance(1, 2), rng.chance(1, 2));
        let w = vs.w.unwrap_or_else(|| rng.chance(1, 2));
        let l = vs.l.unwrap_or_else(|| rng.chance(1, 2));
        let vvvv = if vs.nds { rng.below(16) as u8 } else { 0 };
        let pp = match spec.mand {
            0 => 0u8,
            0x66 => 1,
            0xF3 => 2,
            _ => 3,
        };
        let tail = ((!vvvv & 15) << 3) | (u8::from(l) << 2) | pp;
        if map == 1 && !w && !x && !b && rng.chance(1, 2) {
            code.extend_from_slice(&[0xC5, (u8::from(!r) << 7) | tail]);
        } else {
            code.extend_from_slice(&[
                0xC4,
                (u8::from(!r) << 7) | (u8::from(!x) << 6) | (u8::from(!b) << 5) | map,
                (u8::from(w) << 7) | tail,
            ]);
        }
        code.push(opc);
        (rex_w, rex_x, rex_b) = (w, x, b);
        if spec.cat == Cat::Avx && map == 2 && (0x90..=0x93).contains(&opc) {
            return gen_gather(rng, spec, code, st, (r, x, b, w, l), vvvv, insn_addr);
        }
    } else {
        if spec.mand != 0 {
            if spec.mand != 0x66 && rng.chance(1, 8) {
                code.push(0x66); // a 66 alongside F2/F3 is ignored
            }
            code.push(spec.mand);
        }
        let rex = if rng.chance(1, 2) {
            0x40 | rng.below(16) as u8
        } else {
            0
        };
        let rex = if spec.op == [0x63] { rex | 0x48 } else { rex };
        if rex != 0 {
            code.push(rex);
        }
        rex_w = rex & 8 != 0;
        (rex_x, rex_b) = (rex & 2 != 0, rex & 1 != 0);
        code.extend_from_slice(spec.op);
    }
    let mut rip_disp_at = None;
    // (VPSRLVQ/VPSLLVQ counts must stay quadword-aligned: see below.)
    let align = (matches!(spec.cat, Cat::Sse | Cat::Avx) && rng.chance(1, 2))
        || (spec.vex.is_some() && matches!(spec.op, [0x0F, 0x38, 0x45 | 0x47]));
    let mut modrm_reg = 0u8;
    let mut modrm_at = None;
    if spec.modrm {
        let mut modrm = rng.byte();
        if let Some(r) = spec.reg {
            modrm = (modrm & !0x38) | (r << 3);
        }
        match spec.mem {
            Some(true) if modrm >> 6 == 3 => modrm &= 0x3f,
            Some(false) => modrm |= 0xc0,
            _ => {}
        }
        if !matches!(spec.cat, Cat::Int | Cat::Bmi) && spec.mem.is_none() && rng.chance(1, 2) {
            modrm |= 0xc0; // favour register forms for vector/x87 ops
        }
        modrm_reg = (modrm >> 3) & 7;
        modrm_at = Some(code.len());
        code.push(modrm);
        let md = modrm >> 6;
        let rm = modrm & 7;
        if md != 3 {
            // Half the SSE/AVX cases use 16/32-byte-aligned addresses, so
            // aligned forms are compared too (Rosetta never raises #GP on
            // misalignment).
            let al: u64 = match (align, spec.cat) {
                (false, _) => !0,
                (true, Cat::Avx) => !31,
                _ => !15,
            };
            let ptr = |rng: &mut Rng| (DATA + 0x800 + rng.below(0x800)) & al;
            if rm == 4 {
                let sib = rng.byte();
                code.push(sib);
                let idx = ((sib >> 3) & 7) | if rex_x { 8 } else { 0 };
                let base = (sib & 7) | if rex_b { 8 } else { 0 };
                if idx != 4 {
                    st.gpr[usize::from(idx)] = rng.below(0x80) & al;
                }
                if sib & 7 == 5 && md == 0 {
                    // no base: disp32 absolute is out of reach; make it so
                    // index*scale + disp32 lands in the window.
                    let scale = 1u64 << (sib >> 6);
                    let iv = if idx == 4 {
                        0
                    } else {
                        st.gpr[usize::from(idx)]
                    };
                    let target = ptr(rng);
                    let d = target.wrapping_sub(iv.wrapping_mul(scale)) as i64;
                    // An absolute disp32 can't reach 0x6_0000_0000: let it
                    // fault (both sides must agree on the fault).
                    code.extend_from_slice(&(d as i32).to_le_bytes());
                } else {
                    st.gpr[usize::from(base)] = ptr(rng);
                    if base == 4 && idx != 4 {
                        st.gpr[4] = ptr(rng) & !7;
                    }
                }
            } else if rm == 5 && md == 0 {
                rip_disp_at = Some(code.len());
                code.extend_from_slice(&[0; 4]);
            } else {
                let base = rm | if rex_b { 8 } else { 0 };
                st.gpr[usize::from(base)] = ptr(rng);
            }
            match md {
                1 => code.push(rng.byte() & al as u8),
                2 => code.extend_from_slice(
                    &(((rng.below(0x800) as i32) - 0x400) & al as i32).to_le_bytes(),
                ),
                _ => {}
            }
        }
    }
    let osz = if rex_w {
        64
    } else if has66 {
        16
    } else {
        32
    };
    let imm_len = match spec.imm {
        Imm::None => 0,
        Imm::B => 1,
        Imm::Z => {
            if osz == 16 {
                2
            } else {
                4
            }
        }
        Imm::V => osz / 8,
        Imm::Enter => 3,
        Imm::Rel8 => 1,
        Imm::Rel32 => 4,
        Imm::Grp3 => {
            if modrm_reg < 2 {
                if spec.op[0] == 0xF6 {
                    1
                } else if osz == 16 {
                    2
                } else {
                    4
                }
            } else {
                0
            }
        }
    };
    for _ in 0..imm_len {
        code.push(rng.byte());
    }
    if spec.op == [0x0F, 0xC2] {
        // Legacy CMPccPS/PD: predicates 0..7 (Rosetta reads 5 bits); VEX: 0..31.
        let n = code.len();
        code[n - 1] &= if spec.vex.is_some() { 31 } else { 7 };
    }
    if matches!(spec.imm, Imm::Rel8 | Imm::Rel32) {
        // K fillers of `inc r15` (3 bytes); jump over j of them.
        const FILL: [u8; 3] = [0x49, 0xFF, 0xC7];
        let k = rng.below(4);
        let j = rng.below(k + 1) as u32;
        let n = code.len();
        let w = if spec.imm == Imm::Rel8 { 1 } else { 4 };
        code[n - w..].copy_from_slice(&(3 * j).to_le_bytes()[..w]);
        for _ in 0..k {
            code.extend_from_slice(&FILL);
        }
        st.gpr[1] = rng.below(3); // LOOP counts 0, 1, 2
    }
    if spec.imm == Imm::Enter {
        // Keep the nesting level small: level copies that many frame words.
        let n = code.len();
        code[n - 1] &= 3;
    }
    if let Some(at) = rip_disp_at {
        let end = insn_addr + code.len() as u64;
        let target = DATA + 0x800 + rng.below(0x800);
        let target = match (align, spec.cat) {
            (false, _) => target,
            (true, Cat::Avx) => target & !31,
            _ => target & !15,
        };
        let d = target.wrapping_sub(end) as i64 as i32;
        code[at..at + 4].copy_from_slice(&d.to_le_bytes());
    }
    // String ops / XLAT: pointers into the window, small counts.
    let opc = *spec.op.last().unwrap();
    if spec.op == [0x0F, 0xF7] {
        // MASKMOVQ/MASKMOVDQU store to [rDI].
        st.gpr[7] = DATA + 0x800 + rng.below(0x800);
    }
    if spec.op.len() == 1 && matches!(opc, 0xA4..=0xA7 | 0xAA..=0xAF | 0xD7) {
        st.gpr[6] = DATA + 0x800 + rng.below(0x800);
        st.gpr[7] = DATA + 0x800 + rng.below(0x800);
        st.gpr[1] = rng.below(12);
        st.gpr[3] = DATA + 0x800 + rng.below(0x700);
    }

    let mut data = random_data(rng);
    if spec.vex.is_some() && matches!(spec.op, [0x0F, 0x38, 0x45 | 0x47]) && vex_w(&code) {
        // VPSRLVQ/VPSLLVQ: Rosetta takes only the low 32 bits of each count
        // (a count of 2^32 shifts by 0; the SDM zeroes the lane): keep every
        // quadword's high half clear.
        for i in 0..16 {
            let o = 160 + 16 * i;
            let x = u128::from_le_bytes(st.fxsave[o..o + 16].try_into().unwrap());
            let m = 0x0000_0000_ffff_ffff_0000_0000_ffff_ffffu128;
            st.fxsave[o..o + 16].copy_from_slice(&(x & m).to_le_bytes());
            st.ymm_hi[i] &= m;
        }
        for q in data.chunks_mut(8) {
            q[4..].fill(0);
        }
    }
    st.rip = insn_addr;
    Case {
        code,
        st,
        data,
        spec: *spec,
        modrm_at,
        mem_ignore: Vec::new(),
    }
}

/// `XSAVE`/`XRSTOR` (`[rbx + disp8]`, usually 64-byte aligned, `REX.W`
/// random) and `XGETBV` (`ECX` mostly 0). An `XRSTOR` area holds a valid
/// image: masked exceptions, a random `XSTATE_BV`, a clean header.
fn gen_xstate(rng: &mut Rng, spec: &OpSpec, insn_addr: u64) -> Case {
    let mut st = CpuState::default();
    for r in &mut st.gpr {
        *r = int_value(rng);
    }
    st.gpr[4] = DATA + 0x1000 + rng.below(0x40) * 8;
    st.rflags = 0x202 | (rng.next() & 0x8d5);
    st.fxsave = random_fx(rng, Cat::X87);
    // Rosetta's XSAVE area holds the x87 registers in physical order (the
    // SDM: stack order, as FXSAVE): they agree at TOP = 0.
    st.fxsave[3] &= !0x38;
    for v in &mut st.ymm_hi {
        *v = xmm_value(rng);
    }
    let mut data = random_data(rng);
    let mut code = Vec::new();
    let mut mem_ignore = Vec::new();
    let ext = spec.reg.unwrap_or(0);
    let modrm_at;
    if ext == 2 {
        code.extend_from_slice(&[0x0F, 0x01]);
        modrm_at = Some(code.len());
        code.push(0xD0);
        if rng.chance(3, 4) {
            st.gpr[1] = 0;
        }
    } else {
        if rng.chance(1, 2) {
            code.push(0x48);
        }
        code.extend_from_slice(&[0x0F, 0xAE]);
        modrm_at = Some(code.len());
        code.push(0x43 | (ext << 3)); // [rbx + disp8]
        let disp = rng.below(2) * 64 + if rng.chance(1, 8) { rng.below(64) } else { 0 };
        code.push(disp as u8);
        let base = DATA + 0x800 + rng.below(0x10) * 64;
        st.gpr[3] = base;
        let addr = base + disp;
        st.gpr[0] = if rng.chance(3, 4) {
            rng.below(8)
        } else {
            int_value(rng)
        };
        st.gpr[2] = if rng.chance(3, 4) { 0 } else { int_value(rng) };
        let off = (addr - DATA) as usize;
        if ext == 5 {
            let mut fx = random_fx(rng, Cat::X87);
            fx[3] &= !0x38;
            data[off..off + 512].copy_from_slice(&fx);
            data[off + 512..off + 576].fill(0);
            data[off + 512] = rng.below(8) as u8;
        } else {
            // Rosetta writes XSTATE_BV whole (RFBM | AVX) instead of
            // merging RFBM's bits into the old value, and always writes
            // MXCSR (the SDM: only with SSE or AVX in RFBM).
            mem_ignore.push((addr + 512, 8));
            if st.gpr[0] & 6 == 0 {
                mem_ignore.push((addr + 24, 8));
            }
        }
    }
    st.rip = insn_addr;
    Case {
        code,
        st,
        data,
        spec: *spec,
        modrm_at,
        mem_ignore,
    }
}

/// `VEX.W` of a VEX-encoded case (the 2-byte form implies 0). (the 2-byte form implies 0).
fn vex_w(code: &[u8]) -> bool {
    let i = code.iter().position(|&b| b == 0xC4 || b == 0xC5).unwrap();
    code[i] == 0xC4 && code[i + 2] & 0x80 != 0
}

/// Random data-window contents: float, x87 and integer chunks.
fn random_data(rng: &mut Rng) -> Vec<u8> {
    let mut data = vec![0u8; DATA_LEN];
    let mut i = 0;
    while i < DATA_LEN {
        let chunk: u128 = match rng.below(3) {
            0 => xmm_value(rng),
            1 => f80_bits(rng),
            _ => u128::from(int_value(rng)) | (u128::from(int_value(rng)) << 64),
        };
        data[i..i + 16].copy_from_slice(&chunk.to_le_bytes());
        i += 16;
    }
    data
}

/// A gather (`VEX 0F 38 90..93`) with a VSIB operand: a base register into
/// the data window, an index vector of small offsets, and mostly distinct
/// destination/index/mask registers (equal ones are `#UD`).
fn gen_gather(
    rng: &mut Rng,
    spec: &OpSpec,
    mut code: Vec<u8>,
    mut st: CpuState,
    (r, x, b, w, l): (bool, bool, bool, bool, bool),
    vvvv: u8,
    insn_addr: u64,
) -> Case {
    // (Equal destination/index/mask registers are #UD per the SDM, which
    // Rosetta doesn't check: unit-tested instead.)
    let mut dst = (rng.below(8) as u8) | (u8::from(r) << 3);
    let mut idx = (rng.below(8) as u8) | (u8::from(x) << 3);
    while idx == vvvv {
        idx = ((idx + 1) & 7) | (u8::from(x) << 3);
    }
    while dst == vvvv || dst == idx {
        dst = ((dst + 1) & 7) | (u8::from(r) << 3);
    }
    let md = rng.below(3) as u8;
    let mut base = rng.below(8) as u8;
    if base == 5 && md == 0 {
        base = 3; // (no-base disp32 can't reach the window)
    }
    let base_r = base | (u8::from(b) << 3);
    let scale = rng.below(4) as u8;
    let modrm_at = Some(code.len());
    code.push((md << 6) | ((dst & 7) << 3) | 4);
    code.push((scale << 6) | ((idx & 7) << 3) | base);
    match md {
        1 => code.push(rng.byte() & 0x3f),
        2 => code.extend_from_slice(&(rng.below(0x100) as i32).to_le_bytes()),
        _ => {}
    }
    if base_r == 4 {
        st.gpr[4] = DATA + 0x800 + rng.below(0x80) * 8;
    } else {
        st.gpr[usize::from(base_r)] = DATA + 0x800 + rng.below(0x400);
    }
    let _ = (w, l);
    // Index lanes: small non-negative offsets (dword or qword lanes).
    let qidx = spec.op[2] & 1 == 1;
    let mut iv = [0u128; 2];
    for (h, half) in iv.iter_mut().enumerate() {
        let _ = h;
        if qidx {
            *half = u128::from(rng.below(0x40)) | (u128::from(rng.below(0x40)) << 64);
        } else {
            for k in 0..4 {
                *half |= u128::from(rng.below(0x40)) << (32 * k);
            }
        }
    }
    let i = usize::from(idx);
    st.fxsave[160 + 16 * i..176 + 16 * i].copy_from_slice(&iv[0].to_le_bytes());
    st.ymm_hi[i] = iv[1];
    let data = random_data(rng);
    st.rip = insn_addr;
    Case {
        code,
        st,
        data,
        spec: *spec,
        modrm_at,
        mem_ignore: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Interpreter side

fn interp_mem(code_page: &[u8], data: &[u8]) -> GuestMemory {
    let mut m = GuestMemory::new(BASE, 0x10000);
    m.map(CODE_PAGE, 0x1000, Prot::rwx()).unwrap();
    m.map(DATA, DATA_LEN as u64, Prot::rw()).unwrap();
    m.write_init(CODE_PAGE, code_page).unwrap();
    m.write_init(DATA, data).unwrap();
    m
}

// ---------------------------------------------------------------------------
// Comparison

/// Fields that may legitimately differ, per case.
#[derive(Clone, Copy)]
struct Ignore {
    /// RFLAGS bits not compared.
    flags: u64,
    /// x87 status-word bits not compared.
    fsw: u16,
    /// Accept x87 register results within one unit in the last place
    /// (Rosetta's transcendentals are occasionally 1 ulp off near halfway
    /// cases — checked against exact rational arithmetic).
    ulp: bool,
    /// MXCSR exception-flag bits not compared.
    mxcsr: u8,
}

fn describe_fx_diff(a: &[u8; 512], b: &[u8; 512]) -> Vec<String> {
    // Rosetta: no DE (denormal-operand) flag, no ES/B summary (it never
    // models unmasked exceptions), and empty registers read back as the
    // indefinite (hardware keeps their stale contents).
    let mut a = *a;
    let mut b = *b;
    for x in [&mut a, &mut b] {
        x[2] &= !0x82;
        x[3] &= !0x80;
    }
    let top = usize::from((b[3] >> 3) & 7);
    for i in 0..8 {
        let phys = (top + i) & 7;
        if a[4] & (1 << phys) == 0 && b[4] & (1 << phys) == 0 {
            let o = 32 + 16 * i;
            a[o..o + 10].fill(0);
            b[o..o + 10].fill(0);
        }
    }
    let (a, b) = (&a, &b);
    let mut d = Vec::new();
    let u16at = |x: &[u8; 512], o: usize| u16::from_le_bytes([x[o], x[o + 1]]);
    if u16at(a, 0) != u16at(b, 0) {
        d.push(format!("fcw {:04x} vs {:04x}", u16at(a, 0), u16at(b, 0)));
    }
    if u16at(a, 2) != u16at(b, 2) {
        d.push(format!("fsw {:04x} vs {:04x}", u16at(a, 2), u16at(b, 2)));
    }
    if a[4] != b[4] {
        d.push(format!("ftw {:02x} vs {:02x}", a[4], b[4]));
    }
    // (DE, MXCSR bit 1, is never reported by Rosetta.)
    let mx = |x: &[u8; 512]| u32::from_le_bytes(x[24..28].try_into().unwrap()) & !2;
    if mx(a) != mx(b) {
        d.push(format!("mxcsr {:08x} vs {:08x}", mx(a), mx(b)));
    }
    for i in 0..8 {
        let o = 32 + 16 * i;
        if a[o..o + 10] != b[o..o + 10] {
            let v = |x: &[u8; 512]| {
                let mut t = [0u8; 16];
                t[..10].copy_from_slice(&x[o..o + 10]);
                u128::from_le_bytes(t)
            };
            d.push(format!("st{i} {:020x} vs {:020x}", v(a), v(b)));
        }
    }
    for i in 0..16 {
        let o = 160 + 16 * i;
        if a[o..o + 16] != b[o..o + 16] {
            let v = |x: &[u8; 512]| u128::from_le_bytes(x[o..o + 16].try_into().unwrap());
            d.push(format!("xmm{i} {:032x} vs {:032x}", v(a), v(b)));
        }
    }
    d
}

/// Valid encodings Rosetta raises #UD for: the undocumented x87 aliases real
/// CPUs execute, and the 287-era no-ops.
fn rosetta_unsupported(case: &Case) -> bool {
    let Some(m) = modrm_of(case) else {
        return false;
    };
    let reg_form = m >> 6 == 3;
    let ext = (m >> 3) & 7;
    match case.spec.op {
        [0xD9] => reg_form && ext == 3,                               // FSTP1
        [0xDC] => reg_form && (ext == 2 || ext == 3),                 // FCOM2/FCOMP3
        [0xDD] => reg_form && ext == 1,                               // FXCH4
        [0xDE] => reg_form && ext == 2,                               // FCOMP5
        [0xDF] => reg_form && ext <= 3,                               // FFREEP/FXCH7/FSTP8/9
        [0xDB] => reg_form && ext == 4 && matches!(m & 7, 0 | 1 | 4), // FNENI/FNDISI/FNSETPM
        _ => false,
    }
}

fn signal_of(o: Outcome) -> i32 {
    match o {
        Outcome::Done => 0,
        Outcome::Syscall => -1,
        Outcome::Illegal => 4,
        Outcome::Fault { .. } => 11,
        Outcome::Signal(s) => s,
    }
}

/// The ModRM byte of a case (if its opcode has one) and the byte after the
/// whole instruction's last byte position (for an imm8 count).
fn modrm_of(case: &Case) -> Option<u8> {
    case.code.get(case.modrm_at?).copied()
}

const CF: u64 = 1;
const PF: u64 = 4;
const AF: u64 = 0x10;
const ZF: u64 = 0x40;
const SF: u64 = 0x80;
const OF: u64 = 0x800;

/// Where the oracle can't be compared bit for bit: architecturally undefined
/// flags (the interpreter follows real-hardware behavior where it is known,
/// which Rosetta doesn't always share), and documented Rosetta deviations
/// from the Intel SDM. `None` skips the case entirely.
fn known_differences(case: &Case, sw_sig: i32, hw_sig: i32) -> Option<Ignore> {
    let op = case.spec.op;
    let modrm = modrm_of(case);
    let ext = modrm.map_or(0, |m| (m >> 3) & 7);
    let mem = modrm.is_some_and(|m| m >> 6 != 3);
    // Rosetta doesn't enforce alignment (#GP) for CMPXCHG16B, FXSAVE/FXRSTOR
    // and aligned SSE operands, and doesn't #GP on LDMXCSR reserved bits.
    if sw_sig == 11
        && hw_sig == 0
        && (matches!(case.spec.cat, Cat::Sse | Cat::Mmx | Cat::Avx)
            || matches!(op, [0x0F, 0xC7 | 0xAE]))
    {
        return None;
    }
    // Rosetta takes BT* register bit offsets modulo the operand size even for
    // memory operands (the SDM's bit-string addressing reaches outside).
    if matches!(op, [0x0F, 0xA3 | 0xAB | 0xB3 | 0xBB]) && mem && sw_sig != hw_sig {
        return None;
    }
    // x87 cases Rosetta handles outside the SDM: FBLD doesn't check for stack
    // overflow, and F2XM1 outside its [-1, 1] domain returns 0.
    if case.spec.cat == Cat::X87 {
        let fx = &case.st.fxsave;
        let top = (fx[3] >> 3) & 7;
        let st0 = {
            let mut t = [0u8; 16];
            t[..10].copy_from_slice(&fx[32..42]);
            u128::from_le_bytes(t)
        };
        if op == [0xDF] && mem && ext == 4 && fx[4] & (1 << (top.wrapping_sub(1) & 7)) != 0 {
            return None;
        }
        // The transcendentals (F2XM1, FYL2X, FPTAN, FPATAN, FYL2XP1, FSINCOS,
        // FSIN, FCOS): Rosetta rounds them correctly only to nearest and
        // mis-rounds denormal results; F2XM1 and FYL2XP1 outside their
        // domains are undefined.
        let transcendental =
            op == [0xD9] && matches!(modrm, Some(0xF0..=0xF3 | 0xF9 | 0xFB | 0xFE | 0xFF));
        let exp0 = (st0 >> 64) & 0x7fff;
        if transcendental
            && (fx[1] & 0x0c != 0
                || exp0 == 0
                || (modrm == Some(0xF0) && exp0 >= 0x3fff)
                || (modrm == Some(0xF9) && !(0x3fbf..0x3ffd).contains(&exp0)))
        {
            return None;
        }
        if transcendental {
            // Rosetta's round-up indicator (C1) for these is unreliable, and
            // its F2XM1 sometimes omits PE.
            let pe = if modrm == Some(0xF0) { 0x20 } else { 0 };
            return Some(Ignore {
                flags: 0,
                fsw: 0x200 | pe,
                ulp: true,
                mxcsr: 0,
            });
        }
    }
    // VCVTPS2PH with an immediate rounding mode: Rosetta swaps round-down
    // (01) and round-up (10).
    if case.spec.vex.is_some()
        && op == [0x0F, 0x3A, 0x1D]
        && case
            .code
            .last()
            .is_some_and(|&i| i & 4 == 0 && matches!(i & 3, 1 | 2))
    {
        return None;
    }
    // MINSS/MAXSS…: Rosetta flushes a denormal result under FTZ; FTZ only
    // applies to results that underflow, which MIN/MAX never produce.
    if matches!(case.spec.cat, Cat::Sse | Cat::Avx)
        && matches!(op, [0x0F, 0x5D | 0x5F])
        && u32::from_le_bytes(case.st.fxsave[24..28].try_into().unwrap()) & 0x8000 != 0
    {
        return None;
    }
    let count_is_one = || -> bool {
        match op {
            [0xD0 | 0xD1] => true,
            [0xD2 | 0xD3] | [0x0F, 0xA5 | 0xAD] => {
                let mask = if case.code.iter().any(|&b| (0x48..=0x4F).contains(&b)) {
                    63
                } else {
                    31
                };
                case.st.gpr[1] & mask == 1
            }
            _ => case.code.last() == Some(&1),
        }
    };
    let flags = match op {
        // MUL/IMUL: SF/ZF/AF/PF undefined (hardware sets SF/PF from the low
        // half, Rosetta clears them).
        [0xF6 | 0xF7] if ext == 4 || ext == 5 => SF | ZF | AF | PF,
        [0x69 | 0x6B] | [0x0F, 0xAF] => SF | ZF | AF | PF,
        // DIV/IDIV: all flags undefined.
        [0xF6 | 0xF7] if ext >= 6 => CF | PF | AF | ZF | SF | OF,
        // BSF/BSR: all but ZF undefined.
        [0x0F, 0xBC | 0xBD] if case.spec.mand != 0xF3 => CF | PF | AF | SF | OF,
        // BT*: OF/SF/AF/PF undefined.
        [0x0F, 0xA3 | 0xAB | 0xB3 | 0xBB | 0xBA] => PF | AF | SF | OF,
        // Shifts/rotates: AF undefined; OF undefined for counts other than 1.
        [0xC0 | 0xC1 | 0xD0 | 0xD1 | 0xD2 | 0xD3] => {
            if count_is_one() {
                AF
            } else {
                AF | OF
            }
        }
        // SHLD/SHRD: also CF is undefined for a 16-bit count above 16.
        [0x0F, 0xA4 | 0xA5 | 0xAC | 0xAD] => {
            let big16 = case.code.contains(&0x66) && {
                let c = if matches!(op, [0x0F, 0xA5 | 0xAD]) {
                    case.st.gpr[1] as u8
                } else {
                    *case.code.last().unwrap()
                };
                c & 31 > 16
            };
            AF | if count_is_one() { 0 } else { OF } | if big16 { CF } else { 0 }
        }
        // CMPXCHG: Rosetta computes the comparison as dest - accumulator; the
        // SDM (and real CPUs) use accumulator - dest. Only ZF agrees.
        [0x0F, 0xB0 | 0xB1] => CF | PF | AF | SF | OF,
        // PTEST (VPTEST, VTESTPS/PD): Rosetta leaves AF/OF/PF/SF; the SDM
        // clears them.
        [0x0F, 0x38, 0x17 | 0x0E | 0x0F] => PF | AF | SF | OF,
        // LZCNT/TZCNT: only CF/ZF are defined.
        [0x0F, 0xBC | 0xBD] if case.spec.mand == 0xF3 => PF | AF | SF | OF,
        // BMI: AF/PF undefined (BEXTR: SF too).
        [0x0F, 0x38, 0xF2 | 0xF3 | 0xF5] if case.spec.vex.is_some() => PF | AF,
        [0x0F, 0x38, 0xF7] if case.spec.vex.is_some() => PF | AF | SF,
        _ => 0,
    };
    // VFMADDSUB/VFMSUBADD: Rosetta raises the flags of both the adding and
    // the subtracting computation in every lane.
    let mxcsr = if case.spec.vex.is_some()
        && matches!(op, [0x0F, 0x38, 0x96 | 0x97 | 0xA6 | 0xA7 | 0xB6 | 0xB7])
    {
        0x3d
    } else {
        0
    };
    Some(Ignore {
        flags,
        fsw: 0,
        ulp: false,
        mxcsr,
    })
}

/// Whether the host CPU has `FEAT_AFP`. Without it (M1/M2, e.g. GitHub's
/// macOS runners), Rosetta can't produce x86's negative default NaN (it
/// returns Arm's positive one) and its `RCPPS`/`RSQRTPS` estimates differ
/// from a newer host's.
fn host_has_afp() -> bool {
    static AFP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AFP.get_or_init(|| {
        std::process::Command::new("sysctl")
            .args(["-n", "hw.optional.arm.FEAT_AFP"])
            .output()
            .is_ok_and(|o| o.stdout.starts_with(b"1"))
    })
}

/// On a host without `FEAT_AFP` (see [`host_has_afp`]), accept in the
/// interpreter's register `a` what such a Rosetta can't match in `b`: a NaN
/// lane differing only in its sign, and an `RCP`/`RSQRT` estimate within the
/// SDM's error bound (`1.5 * 2^-12` relative, so two compliant estimates are
/// within `2^-11` of each other).
fn older_rosetta_lanes(case: &Case, a: &mut u128, b: u128) {
    let estimate = matches!(case.spec.op, [0x0F, 0x52 | 0x53]);
    for lane in 0..2 {
        let sh = 64 * lane;
        let (x, y) = ((*a >> sh) as u64, (b >> sh) as u64);
        if f64::from_bits(x).is_nan() && f64::from_bits(y).is_nan() && x ^ y == 1 << 63 {
            *a = (*a & !(u128::from(u64::MAX) << sh)) | (u128::from(y) << sh);
        }
    }
    for lane in 0..4 {
        let sh = 32 * lane;
        let (x, y) = ((*a >> sh) as u32, (b >> sh) as u32);
        let (fx, fy) = (f32::from_bits(x), f32::from_bits(y));
        let nan_sign = fx.is_nan() && fy.is_nan() && x ^ y == 1 << 31;
        let close = estimate
            && fx.is_finite()
            && fy.is_finite()
            && (fx - fy).abs() <= fy.abs() * (2.0f32).powi(-11);
        if nan_sign || close {
            *a = (*a & !(u128::from(u32::MAX) << sh)) | (u128::from(y) << sh);
        }
    }
}

fn compare(
    case: &Case,
    sw_out: Outcome,
    sw: &CpuState,
    sw_data: &[u8],
    hw: &HwResult,
    ign: Ignore,
) -> Vec<String> {
    let mut diffs = Vec::new();
    let sw_sig = match sw_out {
        Outcome::Done => 0,
        Outcome::Syscall => -1,
        Outcome::Illegal => 4,
        Outcome::Fault { .. } => 11,
        Outcome::Signal(s) => s,
    };
    // Rosetta reports some faults as SIGBUS where Linux would say SIGSEGV.
    let hw_sig = if hw.signo == 10 { 11 } else { hw.signo };
    if hw_sig == 14 {
        // The oracle's watchdog fired: hardware never finished the
        // instruction (a Rosetta livelock); not comparable.
        return diffs;
    }
    if sw_sig != hw_sig {
        diffs.push(format!("signal interp={sw_sig} hw={hw_sig}"));
        return diffs;
    }
    if hw_sig != 0 {
        return diffs; // faulted identically; state is the pre-fault state
    }
    for r in 0..16 {
        // CMPXCHG r/m32 that succeeds: the SDM writes no accumulator (RAX's
        // upper half survives); Rosetta writes EAX (zero-extending).
        let cmpxchg32_ok = r == 0
            && case.spec.op == [0x0F, 0xB1]
            && hw.rflags & ZF != 0
            && sw.gpr[0] & 0xffff_ffff == hw.gpr[0]
            && hw.gpr[0] >> 32 == 0;
        if sw.gpr[r] != hw.gpr[r] && !cmpxchg32_ok {
            diffs.push(format!(
                "{} {:#x} vs {:#x} (was {:#x})",
                GPR_NAMES[r], sw.gpr[r], hw.gpr[r], case.st.gpr[r]
            ));
        }
    }
    let fmask = 0xcd5 | (1 << 14) | (1 << 18) | (1 << 21);
    let (a, b) = (
        sw.rflags & fmask & !ign.flags,
        hw.rflags & fmask & !ign.flags,
    );
    if a != b {
        diffs.push(format!(
            "rflags {a:#x} vs {b:#x} (diff {:#x}, was {:#x})",
            a ^ b,
            case.st.rflags
        ));
    }
    let (mut sfx, mut hfx) = (sw.fxsave, hw.fx);
    for x in [&mut sfx, &mut hfx] {
        x[2] &= !(ign.fsw as u8);
        x[3] &= !((ign.fsw >> 8) as u8);
        x[24] &= !ign.mxcsr;
    }
    if ign.ulp {
        for i in 0..8 {
            let o = 32 + 16 * i;
            let v = |x: &[u8; 512]| {
                let mut t = [0u8; 16];
                t[..10].copy_from_slice(&x[o..o + 10]);
                u128::from_le_bytes(t)
            };
            if v(&sfx).abs_diff(v(&hfx)) <= 1 {
                sfx[o..o + 10].copy_from_slice(&hfx[o..o + 10]);
            }
        }
    }
    let mut sw_ymm = sw.ymm_hi;
    if !host_has_afp() {
        for i in 0..16 {
            let o = 160 + 16 * i;
            let mut a = u128::from_le_bytes(sfx[o..o + 16].try_into().unwrap());
            let b = u128::from_le_bytes(hfx[o..o + 16].try_into().unwrap());
            older_rosetta_lanes(case, &mut a, b);
            sfx[o..o + 16].copy_from_slice(&a.to_le_bytes());
            older_rosetta_lanes(case, &mut sw_ymm[i], hw.ymm_hi[i]);
        }
    }
    diffs.extend(describe_fx_diff(&sfx, &hfx));
    for i in 0..16 {
        if sw_ymm[i] != hw.ymm_hi[i] {
            diffs.push(format!(
                "ymm{i}.hi {:032x} vs {:032x}",
                sw_ymm[i], hw.ymm_hi[i]
            ));
        }
    }
    // FXSAVE: the pointer fields (FOP/FIP/FDP, written as zero here) and the
    // reserved bytes are left unwritten by Rosetta.
    let fxsave = case.spec.op == [0x0F, 0xAE]
        && case.spec.vex.is_none()
        && modrm_of(case).is_some_and(|m| matches!((m >> 3) & 7, 0 | 4));
    let ignored = |i: usize| {
        case.mem_ignore
            .iter()
            .any(|&(a, n)| (a..a + n as u64).contains(&(DATA + i as u64)))
    };
    // FNSTENV/FNSAVE: Rosetta tags every non-empty register "01" (zero)
    // instead of classifying it; only emptiness is comparable.
    let env = matches!(case.spec.op, [0xD9 | 0xDD])
        && modrm_of(case).is_some_and(|m| m >> 6 != 3 && (m >> 3) & 7 == 6);
    let tag_like =
        |x: u8, y: u8| (0..4).all(|k| ((x >> (2 * k)) & 3 == 3) == ((y >> (2 * k)) & 3 == 3));
    let bad: Vec<usize> = sw_data
        .iter()
        .zip(&hw.data)
        .enumerate()
        .filter(|&(i, (x, y))| {
            !(x == y || ignored(i) || (fxsave && *x == 0) || (env && tag_like(*x, *y)))
        })
        .map(|(i, _)| i)
        .collect();
    if let (Some(&first), Some(&last)) = (bad.first(), bad.last()) {
        diffs.push(format!(
            "mem[{:#x}..={:#x}] interp {:02x?} hw {:02x?}",
            DATA + first as u64,
            DATA + last as u64,
            &sw_data[first..=last.min(first + 15)],
            &hw.data[first..=last.min(first + 15)]
        ));
    }
    diffs
}

const GPR_NAMES: [&str; 16] = [
    "rax", "rcx", "rdx", "rbx", "rsp", "rbp", "rsi", "rdi", "r8", "r9", "r10", "r11", "r12", "r13",
    "r14", "r15",
];

/// The mismatch-grouping key: the opcode bytes plus ModRM.reg (the opcode
/// extension of group encodings) and the register-vs-memory form.
fn group_key(case: &Case) -> String {
    if !std::env::var("NIXVM_X86_DIFF_HEX")
        .unwrap_or_default()
        .is_empty()
    {
        return hex(&case.code);
    }
    let mut k = String::new();
    k.push_str(&group_key_spec(&case.spec));
    if let Some(m) = modrm_of(case) {
        k.push_str(&format!(
            "/{} {}",
            (m >> 3) & 7,
            if m >> 6 == 3 { "reg" } else { "mem" }
        ));
    }
    k.trim_end().to_string()
}

fn hex(b: &[u8]) -> String {
    b.iter()
        .map(|x| format!("{x:02x}"))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn interpreter_matches_real_x86() {
    let Some(mut oracle) = Oracle::start() else {
        return;
    };
    let cases: u64 = std::env::var("NIXVM_X86_DIFF_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);
    let seed: u64 = std::env::var("NIXVM_X86_DIFF_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let only = std::env::var("NIXVM_X86_DIFF_ONLY").unwrap_or_default();
    let report = std::env::var_os("NIXVM_X86_DIFF_REPORT").is_some();
    let mut rng = Rng(seed);
    let all = specs();
    let chosen: Vec<OpSpec> = all
        .into_iter()
        .filter(|s| {
            only.is_empty()
                || only.split(',').any(|o| {
                    let cat = format!("{:?}", s.cat).to_lowercase();
                    cat == o || group_key_spec(s).starts_with(&o.to_uppercase())
                })
        })
        .collect();
    assert!(
        !chosen.is_empty(),
        "no specs match NIXVM_X86_DIFF_ONLY={only}"
    );
    let insn_off = (oracle.insn_addr - CODE_PAGE) as usize;
    let mut groups: BTreeMap<String, (u64, Vec<String>)> = BTreeMap::new();
    let mut total_bad = 0u64;
    let mut crashes = 0u64;
    let mut skipped = 0u64;
    // Compared cases by the signal real execution raised (0 = completed).
    let mut signals: BTreeMap<i32, u64> = BTreeMap::new();
    // NIXVM_X86_DIFF_HEX=0f77,660f77,…: run just these encodings.
    let fixed: Vec<Vec<u8>> = std::env::var("NIXVM_X86_DIFF_HEX")
        .unwrap_or_default()
        .split(',')
        .filter(|h| !h.is_empty())
        .map(|h| {
            (0..h.len() / 2)
                .map(|i| u8::from_str_radix(&h[2 * i..2 * i + 2], 16).unwrap())
                .collect()
        })
        .collect();
    for case_no in 0..cases {
        let spec = rng.pick(&chosen);
        let mut case = gen_case(&mut rng, spec, oracle.insn_addr);
        while rosetta_unsupported(&case) {
            case = gen_case(&mut rng, spec, oracle.insn_addr);
        }
        if !fixed.is_empty() {
            // Probe mode: a listed encoding, with every GPR but RSP pointing
            // into the data window (so any memory form is addressable).
            case.code = fixed[(case_no % fixed.len() as u64) as usize].clone();
            for (r, v) in case.st.gpr.iter_mut().enumerate() {
                if r != 4 {
                    *v = DATA + 0x800 + rng.below(0x40) * 16;
                }
            }
        }
        let Some(hw) = oracle.run(&case.code, &case.st, &case.data, case.spec.cat == Cat::Mmx)
        else {
            crashes += 1;
            println!("oracle crashed on [{}]", hex(&case.code));
            continue;
        };
        let mut page = vec![0u8; 0x1000];
        page[insn_off..insn_off + case.code.len()].copy_from_slice(&case.code);
        let mut mem = interp_mem(&page, &case.data);
        let mut sw = case.st.clone();
        // Run until execution reaches the end of the case's code (one
        // instruction, or a branch plus the filler it may skip).
        let end = oracle.insn_addr + case.code.len() as u64;
        let mut out = testing::step(&mut mem, &mut sw);
        for _ in 0..8 {
            if out != Outcome::Done || sw.rip == end {
                break;
            }
            out = testing::step(&mut mem, &mut sw);
        }
        let sw_data = mem.read_vec(DATA, DATA_LEN).unwrap();
        let sw_sig = signal_of(out);
        let Some(ign) = known_differences(&case, sw_sig, hw.signo) else {
            skipped += 1;
            continue;
        };
        *signals.entry(hw.signo).or_insert(0u64) += 1;
        let diffs = compare(&case, out, &sw, &sw_data, &hw, ign);
        if !diffs.is_empty() {
            total_bad += 1;
            let e = groups.entry(group_key(&case)).or_insert((0, Vec::new()));
            e.0 += 1;
            if e.1.len() < 3 {
                let mut line = format!("[{}] {}", hex(&case.code), diffs.join("; "));
                if case.spec.cat == Cat::Avx && std::env::var_os("NIXVM_X86_DIFF_VERBOSE").is_some()
                {
                    for i in 0..16 {
                        let o = 160 + 16 * i;
                        let lo = u128::from_le_bytes(case.st.fxsave[o..o + 16].try_into().unwrap());
                        line.push_str(&format!(
                            "\n             ymm{i:<2} {:032x}_{lo:032x}",
                            case.st.ymm_hi[i]
                        ));
                    }
                }
                if matches!(case.spec.cat, Cat::Sse | Cat::Avx) {
                    let fx = &case.st.fxsave;
                    line.push_str(&format!(
                        "
             init mxcsr {:08x}",
                        u32::from_le_bytes(fx[24..28].try_into().unwrap())
                    ));
                }
                if case.spec.cat == Cat::X87 {
                    let fx = &case.st.fxsave;
                    let st = |i: usize| {
                        let mut t = [0u8; 16];
                        t[..10].copy_from_slice(&fx[32 + 16 * i..42 + 16 * i]);
                        u128::from_le_bytes(t)
                    };
                    line.push_str(&format!(
                        "\n             init fcw {:02x}{:02x} fsw {:02x}{:02x} ftw {:02x} st0 {:020x} st1 {:020x}",
                        fx[1], fx[0], fx[3], fx[2], fx[4], st(0), st(1)
                    ));
                }
                e.1.push(line);
            }
        }
    }
    for (k, (n, ex)) in &groups {
        println!("{n:7}  {k}");
        for e in ex {
            println!("           {e}");
        }
    }
    println!(
        "{total_bad} of {cases} cases mismatched ({} groups); oracle crashed on {crashes}; \
         {skipped} skipped as known Rosetta deviations; outcomes by signal {signals:?}",
        groups.len()
    );
    assert!(
        report || total_bad == 0,
        "interpreter diverges from real x86 execution"
    );
}

fn group_key_spec(s: &OpSpec) -> String {
    let mut k = String::new();
    if s.vex.is_some() {
        k.push_str("V ");
    }
    if s.mand != 0 {
        k.push_str(&format!("{:02X} ", s.mand));
    }
    for b in s.op {
        k.push_str(&format!("{b:02X} "));
    }
    k
}
