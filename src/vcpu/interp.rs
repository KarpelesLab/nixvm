//! Software CPU interpreter backend — the portable, no-acceleration fallback.
//!
//! Decodes and executes guest instructions against [`GuestMemory`] in a loop,
//! returning [`Exit::Syscall`] when it decodes a syscall instruction (`svc #0`
//! on arm64). Slower than the hardware backends but runs anywhere and on any
//! guest arch — this is the path the browser (wasm) demo uses, and it makes the
//! syscall engine testable in CI with no hypervisor.
//!
//! Coverage grows toward full user-mode aarch64 (ROADMAP Phase 10). Implemented
//! so far: move-wide immediates, PC-relative addressing (`ADR`/`ADRP` and
//! `LDR` literal), add/sub (immediate, shifted, and extended register, with
//! flags), logical (immediate and shifted register), bitfield move
//! (`SBFM`/`UBFM`/`BFM` and their `LSL`/`LSR`/`ASR`/`SXT*`/`UXT*`/`*BFX`/`BFI`
//! aliases), conditional compare/select (`CCMP`/`CCMN`, `CSEL`/`CSINC`/
//! `CSINV`/`CSNEG`), bit manipulation (`REV`/`REV16`/`REV32`/`RBIT`/`CLZ`/
//! `CLS`), compares, conditional/unconditional branches, `BL`/`BLR`/`RET`,
//! load/store (unsigned immediate, unscaled/pre/post-index, register offset,
//! pair including `LDPSW`, and exclusive/acquire-release backed by a
//! best-effort local monitor), ARMv8.1 LSE atomics (`CAS`/`CASP` and their
//! `A`/`L`/`AL` forms, `SWP`, and the `LD<op>`/`ST<op>` atomic memory ops:
//! `ADD`/`CLR`/`EOR`/`SET`/`SMAX`/`SMIN`/`UMAX`/`UMIN`, implemented as plain
//! read-modify-write since a per-address-space lock in the SMP layer
//! serializes guest memory access), and a growing slice of NEON/SIMD
//! (`DUP`/`INS`/`UMOV`/`SMOV`, `LD1`/`ST1`, vector `ADD`/`SUB`/`MUL`/`MLA`/
//! `MLS`/`ABS`/`NEG`/`SMAX`/`SMIN`/`UMAX`/`UMIN`/compares/`SSHL`/`USHL`/
//! `SHL`/`SSHR`/`USHR`/`NOT`/`SADDL`/`UADDL`, `ADDV`/`UADDLV`, and vector
//! floating-point `FADD`/`FSUB`/`FMUL`/`FDIV`/`FMLA`/`FMLS`/`FABS`/`FNEG`/
//! `FSQRT`/`FCMEQ`/`FCMGE`/`FCMGT`) plus scalar FP (`FMADD`/`FMSUB`/
//! `FNMADD`/`FNMSUB`, `FABS`/`FNEG`/`FSQRT`/`FRINT*`/`FCVT` including half
//! precision, `FMAX`/`FMIN`/`FMAXNM`/`FMINNM`/`FNMUL`, `FCMP`/`FCCMP`/
//! `FCSEL`, `SCVTF`/`UCVTF`/`FCVTZS`/`FCVTZU`, `FMOV` in its GPR/immediate/
//! vector forms), NEON permute (`TBL`/`TBX`, `EXT`, `ZIP1`/`ZIP2`/`UZP1`/
//! `UZP2`/`TRN1`/`TRN2`, vector `REV64`/`REV32`/`REV16`), widening/narrowing
//! (`XTN`/`XTN2`, `SQXTN`/`UQXTN`/`SQXTUN` and their `2` forms, `SSHLL`/
//! `USHLL`, `UADDW`/`SADDW`, `ADDHN`), saturating integer arithmetic
//! (`SQADD`/`UQADD`/`SQSUB`/`UQSUB`, `SQSHL`/`UQSHL`/`SQRSHL`/`UQRSHL`,
//! `SUQADD`/`USQADD`), pairwise (`ADDP`/`UMAXP`/`UMINP`/`SMAXP`/`SMINP`/
//! `FADDP`/`FMAXP`, vector and scalar — including `LDNP`/`STNP`, which reuse
//! the `LDP`/`STP` decode paths since they share an addressing mode),
//! across-lanes (`SMAXV`/`SMINV`/`UMAXV`/`UMINV` alongside `ADDV`/`UADDLV`,
//! plus float `FMAXNMV`/`FMINNMV`/`FMAXV`/`FMINV`), scalar `CRC32B`/`H`/`W`/
//! `X` + `CRC32CB`/`H`/`W`/`X`, by-element FP/int multiply (`FMUL`/`FMLA`/
//! `FMLS`/`FMULX`, `MUL`/`MLA`/`MLS`, `SMULL`/`UMULL`/`SQDMULL`, scalar and
//! vector as applicable), reciprocal/rsqrt estimates and their
//! Newton-Raphson refinement steps (`FRECPE`/`FRSQRTE`/`FRECPS`/`FRSQRTS`
//! scalar and vector, `URECPE`/`URSQRTE` vector), scalar/vector FP
//! compare-to-zero (`FCMEQ`/`FCMGT`/`FCMGE`/`FCMLT`/`FCMLE`) and `FABD`, and
//! `PRFM` (decoded as a no-op in all its addressing forms). System-register
//! access (`MRS`/`MSR` for `TPIDR_EL0`/`TPIDRRO_EL0`, `FPCR`/`FPSR`, the
//! `MIDR_EL1`/`CTR_EL0`/`DCZID_EL0`/`ID_AA64ISAR0_EL1`/`ID_AA64PFR0_EL1`
//! ID/feature registers, and a free-running `CNTVCT_EL0`/`CNTVCTSS_EL0`/
//! `CNTFRQ_EL0`), barriers and hints (`DMB`/`DSB`/`ISB`/`SB`/`ESB`/…, all
//! no-ops), cache maintenance (`DC ZVA` — which really zeroes guest memory
//! — plus `DC CVAC`/`CVAU`/`CIVAC`/`IVAC` and `IC IALLU`/`IALLUIS`/`IVAU` as
//! no-ops), and the Armv8 Cryptographic Extension (`AESE`/`AESD`/`AESMC`/
//! `AESIMC` and `SHA1C`/`P`/`M`/`H`/`SU0`/`SU1`, `SHA256H`/`H2`/`SU0`/`SU1`)
//! round out what musl startup, `getauxval`-driven feature dispatch, and
//! hashing/crypto code need. Anything else surfaces as
//! [`Exit::IllegalInstruction`].

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

/// A user-mode aarch64 interpreter.
#[derive(Clone)]
struct Aarch64Interp {
    /// x0..x30. x31 is the zero register (reads 0) or SP depending on encoding.
    x: [u64; 31],
    sp: u64,
    pc: u64,
    tpidr: u64,
    /// Backing state for the `CNTVCT_EL0`/`CNTVCTSS_EL0` virtual counter:
    /// incremented on every read (see `read_sysreg`) so a guest spin loop
    /// waiting for the counter to advance terminates, even though this
    /// interpreter has no wall-clock timer to drive a real one.
    cntvct: u64,
    /// Modeled `FPCR`/`FPSR`. This interpreter doesn't consult the rounding
    /// mode or raise FP exception bits, but `MRS`/`MSR` round-trip through
    /// these so save/restore code (e.g. a signal handler prologue) and
    /// feature-probing code that reads them back see consistent state.
    fpcr: u64,
    fpsr: u64,
    flags: Flags,
    /// SIMD/FP registers v0..v31 (128-bit; D/S/H/B views are the low bits).
    v: [u128; 32],
    /// Debug: print stores to this guest address (from `NIXVM_WATCH`).
    watch: Option<u64>,
    /// Local exclusive monitor for LDXR/LDAXR + STXR/STLXR: opened by a
    /// load-exclusive, checked (and consumed) by the matching
    /// store-exclusive, and cleared by any intervening store (see
    /// `note_store`). Best-effort single-flag model — real hardware tracks a
    /// monitored address range, but this interpreter is single-threaded per
    /// slice and memory access is already serialized by the SMP layer, so a
    /// flag is enough to give musl-style lock code the pass/fail behavior it
    /// expects. Starts `true` so a lone STXR (no preceding LDXR) still
    /// succeeds, matching the exclusive-always-succeeds behavior this
    /// interpreter had before LSE support landed.
    excl_monitor: bool,
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
            excl_monitor: true,
        }
    }

    /// Load or store `1 << scale` bytes of SIMD register `rt` at `addr`.
    fn ldst_vec(
        &mut self,
        addr: u64,
        scale: u32,
        is_load: bool,
        rt: usize,
        mem: &mut GuestMemory,
    ) -> Step {
        let nbytes = 1usize << scale; // 1..=16
        if is_load {
            let mut buf = [0u8; 16];
            if mem.read(addr, &mut buf[..nbytes]).is_err() {
                return Step::Fault { addr, write: false };
            }
            self.v[rt] = u128::from_le_bytes(buf); // sub-128 loads zero-extend
        } else {
            self.note_store(addr, self.v[rt], nbytes);
            let bytes = self.v[rt].to_le_bytes();
            if let Err(e) = mem.write_trap(addr, &bytes[..nbytes]) {
                return Step::Fault {
                    addr: e.fault_addr(),
                    write: true,
                };
            }
        }
        Step::Next
    }

    /// Read a register with zero-register semantics (index 31 → 0).
    fn read_x(&self, i: usize) -> u64 {
        if i == 31 { 0 } else { self.x[i] }
    }
    /// Write a register with zero-register semantics (index 31 → discard).
    fn write_x(&mut self, i: usize, v: u64) {
        if i != 31 {
            self.x[i] = v;
        }
    }
    /// Read a register with stack-pointer semantics (index 31 → SP).
    fn read_sp(&self, i: usize) -> u64 {
        if i == 31 { self.sp } else { self.x[i] }
    }
    /// Write a register with stack-pointer semantics (index 31 → SP).
    fn write_sp(&mut self, i: usize, v: u64) {
        if i == 31 {
            self.sp = v;
        } else {
            self.x[i] = v;
        }
    }

    fn branch(&mut self, offset: i64) -> Step {
        self.pc = (self.pc as i64).wrapping_add(offset) as u64;
        Step::Branched
    }

    /// Read a system register named by the `MRS` encoding. Backs the
    /// registers musl startup, `getauxval`-driven feature dispatch, and
    /// hashing/crypto code actually probe from EL0: the user thread
    /// pointers, `FPCR`/`FPSR`, the cache-geometry/DCZID registers backing
    /// `DC ZVA`, a free-running virtual counter, and a handful of ID/feature
    /// registers advertising exactly what this interpreter implements.
    /// Everything else reads as 0 — a legal reading for any system register
    /// this interpreter doesn't model (equivalent to an EL1 that never wrote
    /// it), rather than an illegal-instruction trap.
    // `TPIDRRO_EL0`/`MPIDR_EL1`/`REVIDR_EL1` read the same as the wildcard
    // (0) — kept as explicit, named arms anyway (rather than folded into
    // `_`) because they're specific registers real startup/feature-dispatch
    // code names, not "whatever this interpreter didn't get around to".
    #[allow(clippy::match_same_arms)]
    fn read_sysreg(&mut self, instr: u32) -> u64 {
        match (instr >> 5) & 0x7fff {
            TPIDR_EL0 => self.tpidr,
            TPIDRRO_EL0 => 0,
            FPCR => self.fpcr,
            FPSR => self.fpsr,
            MIDR_EL1 => MIDR_EL1_VAL,
            MPIDR_EL1 => 0,
            REVIDR_EL1 => 0,
            CTR_EL0 => CTR_EL0_VAL,
            DCZID_EL0 => DCZID_EL0_VAL,
            CNTFRQ_EL0 => CNTFRQ_EL0_VAL,
            CNTVCT_EL0 | CNTVCTSS_EL0 => {
                // Advance on every read so `while (cntvct == cntvct) {}`-style
                // spins (and real "did time pass" checks) always terminate.
                self.cntvct = self.cntvct.wrapping_add(1);
                self.cntvct
            }
            ID_AA64ISAR0_EL1 => ID_AA64ISAR0_EL1_VAL,
            ID_AA64PFR0_EL1 => ID_AA64PFR0_EL1_VAL,
            _ => 0,
        }
    }

    /// Write a system register named by the `MSR` encoding. Only
    /// `TPIDR_EL0`/`FPCR`/`FPSR` are backed by state; writes to the
    /// read-only ID/counter/cache registers `read_sysreg` models above are
    /// architecturally UNDEFINED from EL0 on real hardware, but harmless to
    /// silently accept here, and everything else this interpreter doesn't
    /// model is likewise a no-op.
    fn write_sysreg(&mut self, instr: u32, value: u64) {
        match (instr >> 5) & 0x7fff {
            TPIDR_EL0 => self.tpidr = value,
            FPCR => self.fpcr = value,
            FPSR => self.fpsr = value,
            _ => {}
        }
    }

    /// Read the low 32 bits of `v[n]` as an `f32` (scalar single-precision).
    fn fp32(&self, n: usize) -> f32 {
        f32::from_bits(self.v[n] as u32)
    }
    /// Read the low 64 bits of `v[n]` as an `f64` (scalar double-precision).
    fn fp64(&self, n: usize) -> f64 {
        f64::from_bits(self.v[n] as u64)
    }
    /// Write `x` to `v[d]`, zeroing the upper bits (scalar single-precision).
    fn set_fp32(&mut self, d: usize, x: f32) {
        self.v[d] = u128::from(x.to_bits());
    }
    /// Write `x` to `v[d]`, zeroing the upper bits (scalar double-precision).
    fn set_fp64(&mut self, d: usize, x: f64) {
        self.v[d] = u128::from(x.to_bits());
    }
    /// Read the low 16 bits of `v[n]` as IEEE-754 half-precision bits.
    fn fp16(&self, n: usize) -> u16 {
        self.v[n] as u16
    }
    /// Write `bits` to `v[d]`, zeroing the upper bits (scalar half-precision).
    fn set_fp16(&mut self, d: usize, bits: u16) {
        self.v[d] = u128::from(bits);
    }
    /// Set NZCV per the ARM floating-point comparison rules: unordered (a NaN)
    /// sets C and V; otherwise Z=equal, C=(a>=b), N=(a<b).
    #[allow(clippy::float_cmp)]
    fn set_fcmp_flags(&mut self, a: f64, b: f64) {
        self.flags = if a.is_nan() || b.is_nan() {
            Flags {
                n: false,
                z: false,
                c: true,
                v: true,
            }
        } else if a == b {
            Flags {
                n: false,
                z: true,
                c: true,
                v: false,
            }
        } else if a < b {
            Flags {
                n: true,
                z: false,
                c: false,
                v: false,
            }
        } else {
            Flags {
                n: false,
                z: false,
                c: true,
                v: false,
            }
        };
    }

    /// Debug: report a store overlapping the `NIXVM_WATCH` address. Also
    /// clears the local exclusive monitor (`excl_monitor`) — every store in
    /// this interpreter funnels through here or through
    /// `mem_write_sized`/`ldst`/`ldst_vec` (which call this), so this is the
    /// one place that needs to know "a store just happened".
    fn note_store(&mut self, addr: u64, value: u128, nbytes: usize) {
        self.excl_monitor = false;
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

    /// Perform a single load or store of `1 << size` bytes at `addr`. `opc`
    /// selects: 00 store, 01 load (zero-extend), 10 load (sign-extend to 64),
    /// 11 load (sign-extend to 32).
    fn ldst(&mut self, addr: u64, size: u32, opc: u32, rt: usize, mem: &mut GuestMemory) -> Step {
        let nbytes = 1usize << size;
        if opc == 0b00 {
            let value = self.read_x(rt);
            self.note_store(addr, u128::from(value), nbytes);
            let val = value.to_le_bytes();
            if let Err(e) = mem.write_trap(addr, &val[..nbytes]) {
                return Step::Fault {
                    addr: e.fault_addr(),
                    write: true,
                };
            }
            return Step::Next;
        }
        let mut buf = [0u8; 8];
        if mem.read(addr, &mut buf[..nbytes]).is_err() {
            return Step::Fault { addr, write: false };
        }
        let raw = u64::from_le_bytes(buf);
        let val = match opc {
            0b01 => raw,                                                     // zero-extend
            0b10 => sign_extend(raw, (nbytes * 8) as u32) as u64,            // sign-extend to 64
            _ => sign_extend(raw, (nbytes * 8) as u32) as u64 & 0xffff_ffff, // to 32
        };
        self.write_x(rt, val);
        Step::Next
    }

    /// Write the low `nbytes` bytes of `val` to `addr`, going through
    /// `note_store` (watch logging + exclusive-monitor clear) like every
    /// other store path. Shared by the LSE atomic helpers below.
    fn mem_write_sized(
        &mut self,
        mem: &mut GuestMemory,
        addr: u64,
        nbytes: usize,
        val: u64,
    ) -> bool {
        self.note_store(addr, u128::from(val), nbytes);
        // `write_trap` so a copy-on-write page faults (the atomic is retried
        // after the kernel privatizes it). LSE atomics are naturally aligned, so
        // the access lies in one page and the caller's `addr` names that page.
        mem.write_trap(addr, &val.to_le_bytes()[..nbytes]).is_ok()
    }

    /// CAS/CASA/CASL/CASAL (and the CASB/CASH byte/halfword forms): compare
    /// `rs` against the `nbytes`-wide value at `addr` and, on a match, swap
    /// in `rt`. Always returns the *original* memory value in `rs`,
    /// regardless of whether the swap happened — that's the real CAS
    /// contract, not a simplification. Implemented as a plain
    /// read-modify-write: this interpreter is single-threaded per vcpu slice
    /// and the SMP layer serializes memory access across vcpus, so no host
    /// atomics are needed to be atomic from the guest's perspective.
    fn cas_single(
        &mut self,
        addr: u64,
        nbytes: usize,
        rs: usize,
        rt: usize,
        mem: &mut GuestMemory,
    ) -> Step {
        let Some(old) = mem_read_sized(mem, addr, nbytes) else {
            return Step::Fault { addr, write: false };
        };
        if old == self.read_x(rs) & ones((nbytes * 8) as u32) {
            let new = self.read_x(rt);
            if !self.mem_write_sized(mem, addr, nbytes, new) {
                return Step::Fault { addr, write: true };
            }
        }
        self.write_x(rs, old);
        Step::Next
    }

    /// CASP/CASPA/CASPL/CASPAL: compare-and-swap a register pair. Compares
    /// `rs`/`rs+1` against the two `nbytes`-wide values at `addr`/
    /// `addr+nbytes`; on a full-pair match, swaps in `rt`/`rt+1`. Like
    /// `cas_single`, the original pair is always written back to `rs`/`rs+1`.
    fn cas_pair(
        &mut self,
        addr: u64,
        nbytes: usize,
        rs: usize,
        rt: usize,
        mem: &mut GuestMemory,
    ) -> Step {
        // Real encodings require Rs/Rt even and < 31; mask defensively so a
        // malformed instruction word can't index out of the register file.
        let rs2 = (rs + 1) & 0x1f;
        let rt2 = (rt + 1) & 0x1f;
        let addr2 = addr.wrapping_add(nbytes as u64);
        let Some(old0) = mem_read_sized(mem, addr, nbytes) else {
            return Step::Fault { addr, write: false };
        };
        let Some(old1) = mem_read_sized(mem, addr2, nbytes) else {
            return Step::Fault {
                addr: addr2,
                write: false,
            };
        };
        let mask = ones((nbytes * 8) as u32);
        if old0 == self.read_x(rs) & mask && old1 == self.read_x(rs2) & mask {
            let (new0, new1) = (self.read_x(rt), self.read_x(rt2));
            if !self.mem_write_sized(mem, addr, nbytes, new0) {
                return Step::Fault { addr, write: true };
            }
            if !self.mem_write_sized(mem, addr2, nbytes, new1) {
                return Step::Fault {
                    addr: addr2,
                    write: true,
                };
            }
        }
        self.write_x(rs, old0);
        self.write_x(rs2, old1);
        Step::Next
    }

    /// SWP/SWPA/SWPL/SWPAL (+ SWPB/SWPH): atomic swap — store `rs` to
    /// `addr`, return the original `nbytes`-wide value there in `rt`.
    fn swp(
        &mut self,
        addr: u64,
        nbytes: usize,
        rs: usize,
        rt: usize,
        mem: &mut GuestMemory,
    ) -> Step {
        let Some(old) = mem_read_sized(mem, addr, nbytes) else {
            return Step::Fault { addr, write: false };
        };
        let new = self.read_x(rs);
        if !self.mem_write_sized(mem, addr, nbytes, new) {
            return Step::Fault { addr, write: true };
        }
        self.write_x(rt, old);
        Step::Next
    }

    /// LD<op>/LD<op>A/LD<op>L/LD<op>AL and their ST<op> aliases (same
    /// encoding with `rt == 31`): atomic read-modify-write. Returns the
    /// *original* `nbytes`-wide value in `rt` (silently discarded for the
    /// ST<op> aliases via `write_x`'s usual zero-register semantics) and
    /// writes `old <op> rs` back to memory. `op` is the 3-bit LSE opcode:
    /// 000 ADD, 001 CLR (AND NOT), 010 EOR, 011 SET (OR), 100 SMAX,
    /// 101 SMIN, 110 UMAX, 111 UMIN.
    fn ld_op(
        &mut self,
        addr: u64,
        nbytes: usize,
        rs: usize,
        rt: usize,
        op: u32,
        mem: &mut GuestMemory,
    ) -> Step {
        let Some(old) = mem_read_sized(mem, addr, nbytes) else {
            return Step::Fault { addr, write: false };
        };
        let bits = (nbytes * 8) as u32;
        let mask = ones(bits);
        let s = self.read_x(rs) & mask;
        let new = match op {
            0b000 => old.wrapping_add(s) & mask, // ADD
            0b001 => old & !s & mask,            // CLR: old AND NOT rs
            0b010 => (old ^ s) & mask,           // EOR
            0b011 => (old | s) & mask,           // SET: old OR rs
            0b100 => {
                // SMAX
                if sign_extend(old, bits) >= sign_extend(s, bits) {
                    old
                } else {
                    s
                }
            }
            0b101 => {
                // SMIN
                if sign_extend(old, bits) <= sign_extend(s, bits) {
                    old
                } else {
                    s
                }
            }
            0b110 => {
                if old >= s { old } else { s } // UMAX
            }
            _ => {
                if old <= s { old } else { s } // UMIN (op == 0b111)
            }
        };
        if !self.mem_write_sized(mem, addr, nbytes, new) {
            return Step::Fault { addr, write: true };
        }
        self.write_x(rt, old);
        Step::Next
    }

    /// Compute `a - b` (if `sub`) or `a + b`, setting NZCV. Returns the result.
    fn addsub_flags(&mut self, a: u64, b: u64, sub: bool, sf: bool) -> u64 {
        let (operand, carry_in) = if sub { (!b, 1u128) } else { (b, 0u128) };
        if sf {
            let sum = u128::from(a) + u128::from(operand) + carry_in;
            let r = sum as u64;
            self.flags = Flags {
                n: (r >> 63) & 1 == 1,
                z: r == 0,
                c: (sum >> 64) & 1 == 1,
                v: (((a ^ r) & (operand ^ r)) >> 63) & 1 == 1,
            };
            r
        } else {
            let (a, operand) = (a as u32, operand as u32);
            let sum = u64::from(a) + u64::from(operand) + carry_in as u64;
            let r = sum as u32;
            self.flags = Flags {
                n: (r >> 31) & 1 == 1,
                z: r == 0,
                c: (sum >> 32) & 1 == 1,
                v: (((a ^ r) & (operand ^ r)) >> 31) & 1 == 1,
            };
            u64::from(r)
        }
    }

    fn cond_holds(&self, cond: u32) -> bool {
        let f = &self.flags;
        match cond {
            0b0000 => f.z,
            0b0001 => !f.z,
            0b0010 => f.c,
            0b0011 => !f.c,
            0b0100 => f.n,
            0b0101 => !f.n,
            0b0110 => f.v,
            0b0111 => !f.v,
            0b1000 => f.c && !f.z,          // HI
            0b1001 => !f.c || f.z,          // LS  (not HI)
            0b1010 => f.n == f.v,           // GE
            0b1011 => f.n != f.v,           // LT
            0b1100 => !f.z && (f.n == f.v), // GT
            0b1101 => f.z || (f.n != f.v),  // LE  (not GT)
            _ => true,                      // AL / NV
        }
    }

    #[allow(clippy::too_many_lines)]
    // The by-element-multiply arm binds `q`/`u`/`l`/`m`/`h` together in one
    // scope — these are the ARM ARM's own one-letter names for those
    // instruction-encoding fields (Q, U, L, M, H), and spelling them out
    // would make that arm harder, not easier, to cross-check against the
    // manual.
    #[allow(clippy::many_single_char_names)]
    fn exec(&mut self, instr: u32, mem: &mut GuestMemory) -> Step {
        // ---- exact-match control flow ----
        if instr & 0xFFE0_001F == 0xD400_0001 {
            return Step::Syscall; // svc #imm
        }
        // System hints (NOP/YIELD/WFE/…) and barriers (DMB/DSB/ISB) — no-ops on
        // our single-core, in-order model.
        if instr & 0xFFFF_F000 == 0xD503_2000 || instr & 0xFFFF_F000 == 0xD503_3000 {
            return Step::Next;
        }
        // MRS Xt, <sysreg> / MSR <sysreg>, Xt
        if (instr >> 20) & 0xfff == 0xD53 {
            let rt = reg_field(instr, 0);
            let val = self.read_sysreg(instr);
            self.write_x(rt, val);
            return Step::Next;
        }
        if (instr >> 20) & 0xfff == 0xD51 {
            let rt = reg_field(instr, 0);
            self.write_sysreg(instr, self.read_x(rt));
            return Step::Next;
        }

        // ---- SYS (register): cache maintenance — DC ZVA/CVAC/CVAU/CIVAC/
        // IVAC, IC IALLU/IALLUIS/IVAU ----
        // Shares the System-instructions base encoding with MRS (0xD53) and
        // MSR (0xD51) above, but with op0==1 (a named sysreg needs op0 in
        // {2,3}) and L==0: bits[31:19] == 0x1AA1, objdump-verified against
        // `clang -target aarch64-linux-gnu` (e.g. `dc zva, x0` assembles to
        // 0xd50b7420). That's disjoint from those two exact-match checks
        // above and from the hint/barrier no-op arm below (which fixes
        // CRn in {2,3}; every DC/IC op here fixes CRn==7), so this can't
        // shadow, or be shadowed by, any of them. AT/TLBI and anything else
        // in this instruction class we don't implement falls through to
        // `Illegal`, matching a real EL0 trap.
        if (instr >> 19) & 0x1fff == 0x1AA1 {
            let key = (instr >> 5) & 0x3fff; // op1:CRn:CRm:op2
            let rt = reg_field(instr, 0);
            return match key {
                SYS_DC_ZVA => {
                    // DC ZVA must really zero guest memory — memset/bzero
                    // fast paths (and musl's) rely on it, not just PC
                    // advancing. The address is block-aligned per the
                    // architecture; DCZID_EL0_VAL says the block is
                    // DC_ZVA_BLOCK_BYTES bytes (see that const).
                    let addr = self.read_x(rt) & !(DC_ZVA_BLOCK_BYTES - 1);
                    self.note_store(addr, 0u128, DC_ZVA_BLOCK_BYTES as usize);
                    // Block-aligned and ≤ one page, so `addr` names the faulting
                    // (COW) page directly.
                    if mem
                        .write_trap(addr, &[0u8; DC_ZVA_BLOCK_BYTES as usize])
                        .is_err()
                    {
                        Step::Fault { addr, write: true }
                    } else {
                        Step::Next
                    }
                }
                SYS_DC_CVAC | SYS_DC_CVAU | SYS_DC_CIVAC | SYS_DC_IVAC | SYS_IC_IALLU
                | SYS_IC_IALLUIS | SYS_IC_IVAU => Step::Next,
                _ => Step::Illegal,
            };
        }

        // ---- Cryptographic AES / SHA (2-register): AESE/AESD/AESMC/
        // AESIMC, SHA1H/SHA1SU1/SHA256SU0 ----
        // Fixed bits verified against `clang -target aarch64-linux-gnu` +
        // `llvm-objdump` (e.g. `aese v0.16b,v1.16b` -> 0x4e284820), and the
        // transforms themselves against native execution of the real ARMv8
        // Crypto Extension instructions (this host's Apple Silicon CPU
        // implements FEAT_AES/FEAT_SHA1/FEAT_SHA256) — see the test module.
        // bit28 (U) selects AES (0) vs SHA-1/256 (1); bits[16:12] select the
        // specific op. bits[31:24] == 0x4E/0x5E don't appear as a fixed
        // pattern in any earlier arm, and every SIMD class below that shares
        // a *partial* top-byte mask with this one requires bit10==1, while
        // this class always has bit10==0 — so this can't shadow, or be
        // shadowed by, anything else in this function.
        if instr & 0xEFFE_0C00 == 0x4E28_0800 {
            let u = (instr >> 28) & 1;
            let opcode = (instr >> 12) & 0x1f;
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            self.v[rd] = match (u, opcode) {
                (0, 0b00100) => aes_round(self.v[rd], self.v[rn], true), // AESE
                (0, 0b00101) => aes_round(self.v[rd], self.v[rn], false), // AESD
                (0, 0b00110) => aes_mix_columns(self.v[rn], true),       // AESMC
                (0, 0b00111) => aes_mix_columns(self.v[rn], false),      // AESIMC
                // SHA1H: Sd = ROL(Sn, 30), scalar (zero-extends to 128 bits).
                (1, 0b00000) => u128::from((self.v[rn] as u32).rotate_left(30)),
                (1, 0b00001) => sha1_su1(self.v[rd], self.v[rn]), // SHA1SU1
                (1, 0b00010) => sha256_su0(self.v[rd], self.v[rn]), // SHA256SU0
                _ => return Step::Illegal,
            };
            return Step::Next;
        }

        // ---- Cryptographic SHA (3-register): SHA1C/P/M/SU0, SHA256H/H2/
        // SU1 ----
        // Same verification approach as the 2-register crypto class above
        // (e.g. `sha256h q0,q1,v2.4s` -> 0x5e024020); disjoint from it (and
        // everything else) the same way: bit21==0 here (vs ==1 above), and
        // bit10==0 always, again distinct from every SIMD class requiring
        // bit10==1.
        if instr & 0xFFE0_0C00 == 0x5E00_0000 {
            let opcode = (instr >> 12) & 7;
            let rm = reg_field(instr, 16);
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let e = self.v[rn] as u32;
            self.v[rd] = match opcode {
                0b000 => sha1_quad_round(self.v[rd], e, self.v[rm], Sha1Op::Choose), // SHA1C
                0b001 => sha1_quad_round(self.v[rd], e, self.v[rm], Sha1Op::Parity), // SHA1P
                0b010 => sha1_quad_round(self.v[rd], e, self.v[rm], Sha1Op::Majority), // SHA1M
                0b011 => sha1_su0(self.v[rd], self.v[rn], self.v[rm]),               // SHA1SU0
                // SHA256H2 is called with the pre-round `abcd`/`efgh` swapped
                // into `Vn`/`Vd` (matching the real usage pattern of saving
                // `abcd` before `SHA256H` overwrites it), so it re-derives
                // the same round `SHA256H` computed and keeps the other half.
                0b100 => sha256_hash(self.v[rd], self.v[rn], self.v[rm], false), // SHA256H
                0b101 => sha256_hash(self.v[rn], self.v[rd], self.v[rm], true),  // SHA256H2
                0b110 => sha256_su1(self.v[rd], self.v[rn], self.v[rm]),         // SHA256SU1
                _ => return Step::Illegal,
            };
            return Step::Next;
        }

        if instr & 0xFFFF_FC1F == 0xD65F_0000 {
            self.pc = self.read_x(reg_field(instr, 5)); // ret
            return Step::Branched;
        }
        if instr & 0xFFFF_FC1F == 0xD61F_0000 {
            self.pc = self.read_x(reg_field(instr, 5)); // br
            return Step::Branched;
        }
        if instr & 0xFFFF_FC1F == 0xD63F_0000 {
            let target = self.read_x(reg_field(instr, 5)); // blr
            self.x[30] = self.pc.wrapping_add(4);
            self.pc = target;
            return Step::Branched;
        }

        // ---- branches ----
        if (instr >> 26) & 0x3f == 0b00_0101 {
            let off = sign_extend(u64::from(instr & 0x03ff_ffff), 26) << 2; // b
            return self.branch(off);
        }
        if (instr >> 26) & 0x3f == 0b10_0101 {
            self.x[30] = self.pc.wrapping_add(4); // bl
            let off = sign_extend(u64::from(instr & 0x03ff_ffff), 26) << 2;
            return self.branch(off);
        }
        if instr & 0xFF00_0010 == 0x5400_0000 {
            let cond = instr & 0xf; // b.cond
            if self.cond_holds(cond) {
                let off = sign_extend(u64::from((instr >> 5) & 0x7ffff), 19) << 2;
                return self.branch(off);
            }
            return Step::Next;
        }
        if (instr >> 25) & 0x3f == 0b01_1010 {
            let sf = (instr >> 31) & 1; // cbz / cbnz
            let op = (instr >> 24) & 1;
            let rt = reg_field(instr, 0);
            let mut val = self.read_x(rt);
            if sf == 0 {
                val &= 0xffff_ffff;
            }
            let take = if op == 0 { val == 0 } else { val != 0 };
            if take {
                let off = sign_extend(u64::from((instr >> 5) & 0x7ffff), 19) << 2;
                return self.branch(off);
            }
            return Step::Next;
        }

        // ---- move wide immediate: MOVN/MOVZ/MOVK ----
        if (instr >> 23) & 0x3f == 0b1_00101 {
            let sf = (instr >> 31) & 1;
            let opc = (instr >> 29) & 3;
            let hw = (instr >> 21) & 3;
            let imm16 = u64::from((instr >> 5) & 0xffff);
            let rd = reg_field(instr, 0);
            if sf == 0 && hw > 1 {
                return Step::Illegal;
            }
            let shift = hw * 16;
            let val = imm16 << shift;
            let result = match opc {
                0b10 => val,
                0b00 => !val,
                0b11 => (self.read_x(rd) & !(0xffff_u64 << shift)) | val,
                _ => return Step::Illegal,
            };
            self.write_x(rd, mask_sf(result, sf));
            return Step::Next;
        }

        // ---- PC-relative addressing: ADR / ADRP ----
        if (instr >> 24) & 0x1f == 0b1_0000 {
            let op = (instr >> 31) & 1;
            let immlo = u64::from((instr >> 29) & 3);
            let immhi = u64::from((instr >> 5) & 0x7ffff);
            let rd = reg_field(instr, 0);
            let imm = sign_extend((immhi << 2) | immlo, 21);
            let result = if op == 0 {
                (self.pc as i64).wrapping_add(imm) as u64
            } else {
                ((self.pc & !0xfff) as i64).wrapping_add(imm << 12) as u64
            };
            self.write_x(rd, result);
            return Step::Next;
        }

        // ---- logical immediate: AND/ORR/EOR/ANDS (bitmask immediate) ----
        if (instr >> 23) & 0x3f == 0b1_00100 {
            let sf = (instr >> 31) & 1;
            let opc = (instr >> 29) & 3;
            let n = (instr >> 22) & 1;
            let immr = (instr >> 16) & 0x3f;
            let imms = (instr >> 10) & 0x3f;
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let width = if sf == 1 { 64 } else { 32 };
            if sf == 0 && n == 1 {
                return Step::Illegal;
            }
            let Some((imm, _)) = decode_bit_masks(n, imms, immr, width) else {
                return Step::Illegal;
            };
            let a = self.read_x(rn);
            let r = mask_sf(
                match opc {
                    0b00 | 0b11 => a & imm, // AND / ANDS
                    0b01 => a | imm,        // ORR
                    _ => a ^ imm,           // EOR (0b10)
                },
                sf,
            );
            if opc == 0b11 {
                self.flags = Flags {
                    n: (r >> if sf == 1 { 63 } else { 31 }) & 1 == 1,
                    z: r == 0,
                    c: false,
                    v: false,
                };
                self.write_x(rd, r);
            } else {
                self.write_sp(rd, r);
            }
            return Step::Next;
        }

        // ---- EXTR (extract from a register pair; ROR is an alias) ----
        if (instr >> 23) & 0xff == 0b00100111 {
            let sf = (instr >> 31) & 1;
            let rm = reg_field(instr, 16);
            let imms = (instr >> 10) & 0x3f;
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let (n, m) = (self.read_x(rn), self.read_x(rm));
            let r = if sf == 1 {
                if imms == 0 {
                    m
                } else {
                    (n << (64 - imms)) | (m >> imms)
                }
            } else {
                let (n, m) = (n & 0xffff_ffff, m & 0xffff_ffff);
                if imms == 0 {
                    m
                } else {
                    (n << (32 - imms)) | (m >> imms)
                }
            };
            self.write_x(rd, mask_sf(r, sf));
            return Step::Next;
        }

        // ---- bitfield: SBFM/BFM/UBFM (LSL/LSR/ASR/xtend/xbfx aliases) ----
        if (instr >> 23) & 0x3f == 0b1_00110 {
            let sf = (instr >> 31) & 1;
            let opc = (instr >> 29) & 3;
            let n = (instr >> 22) & 1;
            let immr = (instr >> 16) & 0x3f;
            let imms = (instr >> 10) & 0x3f;
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let width = if sf == 1 { 64u32 } else { 32 };
            if (sf == 1) != (n == 1) {
                return Step::Illegal; // N must match sf
            }
            let Some((wmask, tmask)) = decode_bit_masks(n, imms, immr, width) else {
                return Step::Illegal;
            };
            let src = self.read_x(rn);
            let rotated = ror_val(src, immr, width) & wmask;
            let (bot, top) = match opc {
                0b10 => (rotated, 0u64), // UBFM
                0b00 => {
                    // SBFM: sign-fill from bit `imms`.
                    let top = if (src >> imms) & 1 == 1 {
                        ones(width)
                    } else {
                        0
                    };
                    (rotated, top)
                }
                0b01 => {
                    // BFM: merge with the destination register.
                    let dst = self.read_x(rd);
                    ((dst & !wmask) | rotated, dst)
                }
                _ => return Step::Illegal,
            };
            let result = mask_sf((top & !tmask) | (bot & tmask), sf);
            self.write_x(rd, result);
            return Step::Next;
        }

        // ---- add/subtract immediate (incl. ADDS/SUBS/CMP/CMN) ----
        if (instr >> 23) & 0x3f == 0b1_00010 {
            let sf = (instr >> 31) & 1;
            let op = (instr >> 30) & 1;
            let s = (instr >> 29) & 1;
            let sh = (instr >> 22) & 1;
            let imm12 = u64::from((instr >> 10) & 0xfff);
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let imm = if sh == 1 { imm12 << 12 } else { imm12 };
            let a = self.read_sp(rn);
            if s == 1 {
                let r = self.addsub_flags(a, imm, op == 1, sf == 1);
                self.write_x(rd, r); // Rd is ZR-form for the flag-setting variant
            } else {
                let r = if op == 0 {
                    a.wrapping_add(imm)
                } else {
                    a.wrapping_sub(imm)
                };
                self.write_sp(rd, mask_sf(r, sf));
            }
            return Step::Next;
        }

        // ---- add/subtract shifted register (incl. ADDS/SUBS/CMP) ----
        if (instr >> 24) & 0x1f == 0b0_1011 && (instr >> 21) & 1 == 0 {
            let sf = (instr >> 31) & 1;
            let op = (instr >> 30) & 1;
            let s = (instr >> 29) & 1;
            let shift_type = (instr >> 22) & 3;
            let rm = reg_field(instr, 16);
            let imm6 = (instr >> 10) & 0x3f;
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let a = self.read_x(rn);
            let b = shift_reg(self.read_x(rm), shift_type, imm6, sf == 1);
            let r = if s == 1 {
                self.addsub_flags(a, b, op == 1, sf == 1)
            } else {
                let r = if op == 0 {
                    a.wrapping_add(b)
                } else {
                    a.wrapping_sub(b)
                };
                mask_sf(r, sf)
            };
            self.write_x(rd, r);
            return Step::Next;
        }

        // ---- add/subtract extended register (SP arithmetic) ----
        if (instr >> 24) & 0x1f == 0b0_1011 && (instr >> 21) & 0x7 == 0b001 {
            let sf = (instr >> 31) & 1;
            let op = (instr >> 30) & 1;
            let s = (instr >> 29) & 1;
            let rm = reg_field(instr, 16);
            let option = (instr >> 13) & 7;
            let imm3 = (instr >> 10) & 7;
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let a = self.read_sp(rn);
            let b = extend_reg(self.read_x(rm), option, imm3);
            if s == 1 {
                let r = self.addsub_flags(a, b, op == 1, sf == 1);
                self.write_x(rd, r);
            } else {
                let r = if op == 0 {
                    a.wrapping_add(b)
                } else {
                    a.wrapping_sub(b)
                };
                self.write_sp(rd, mask_sf(r, sf));
            }
            return Step::Next;
        }

        // ---- logical shifted register: AND/ORR/EOR/ANDS (+ BIC via N bit) ----
        if (instr >> 24) & 0x1f == 0b0_1010 {
            let sf = (instr >> 31) & 1;
            let opc = (instr >> 29) & 3;
            let shift_type = (instr >> 22) & 3;
            let n_bit = (instr >> 21) & 1;
            let rm = reg_field(instr, 16);
            let imm6 = (instr >> 10) & 0x3f;
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let a = self.read_x(rn);
            let mut b = shift_reg(self.read_x(rm), shift_type, imm6, sf == 1);
            if n_bit == 1 {
                b = mask_sf(!b, sf);
            }
            let r = match opc {
                0b00 | 0b11 => a & b, // AND / ANDS
                0b01 => a | b,        // ORR (MOV Xd,Xm == ORR Xd,XZR,Xm)
                0b10 => a ^ b,        // EOR
                _ => return Step::Illegal,
            };
            let r = mask_sf(r, sf);
            if opc == 0b11 {
                self.flags = Flags {
                    n: (r >> if sf == 1 { 63 } else { 31 }) & 1 == 1,
                    z: r == 0,
                    c: false,
                    v: false,
                };
            }
            self.write_x(rd, r);
            return Step::Next;
        }

        // ---- 3-source: MADD/MSUB, S/UMADDL, S/UMSUBL, S/UMULH ----
        if (instr >> 24) & 0x1f == 0b1_1011 {
            let sf = (instr >> 31) & 1;
            let op31 = (instr >> 21) & 0x7;
            let o0 = (instr >> 15) & 1;
            let rm = reg_field(instr, 16);
            let ra = reg_field(instr, 10);
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let (n, m, a) = (self.read_x(rn), self.read_x(rm), self.read_x(ra));
            let r = match op31 {
                0b000 => {
                    // MADD/MSUB
                    let prod = n.wrapping_mul(m);
                    let r = if o0 == 0 {
                        a.wrapping_add(prod)
                    } else {
                        a.wrapping_sub(prod)
                    };
                    mask_sf(r, sf)
                }
                0b001 => {
                    // SMADDL/SMSUBL: 64 += sext(Wn) * sext(Wm)
                    let prod = i64::from(n as i32).wrapping_mul(i64::from(m as i32)) as u64;
                    if o0 == 0 {
                        a.wrapping_add(prod)
                    } else {
                        a.wrapping_sub(prod)
                    }
                }
                0b101 => {
                    // UMADDL/UMSUBL: 64 += zext(Wn) * zext(Wm)
                    let prod = u64::from(n as u32).wrapping_mul(u64::from(m as u32));
                    if o0 == 0 {
                        a.wrapping_add(prod)
                    } else {
                        a.wrapping_sub(prod)
                    }
                }
                0b010 => ((i128::from(n as i64).wrapping_mul(i128::from(m as i64))) >> 64) as u64, // SMULH
                0b110 => ((u128::from(n).wrapping_mul(u128::from(m))) >> 64) as u64, // UMULH
                _ => return Step::Illegal,
            };
            self.write_x(rd, r);
            return Step::Next;
        }

        // ---- 1-source: RBIT/REV16/REV32/REV/CLZ/CLS ----
        if (instr >> 21) & 0x3ff == 0b10_1101_0110 {
            let sf = (instr >> 31) & 1;
            let opcode = (instr >> 10) & 0x3f;
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let width = if sf == 1 { 64u32 } else { 32 };
            let x = if sf == 1 {
                self.read_x(rn)
            } else {
                self.read_x(rn) & 0xffff_ffff
            };
            let r = match opcode {
                0b000000 => rbit(x, width),
                0b000001 => rev16(x, width),
                0b000010 => {
                    if sf == 1 {
                        rev32(x)
                    } else {
                        u64::from((x as u32).swap_bytes())
                    }
                }
                0b000011 => x.swap_bytes(), // REV (64-bit)
                0b000100 => u64::from(if sf == 1 {
                    x.leading_zeros()
                } else {
                    (x as u32).leading_zeros()
                }),
                0b000101 => u64::from(cls(x, width)),
                _ => return Step::Illegal,
            };
            self.write_x(rd, mask_sf(r, sf));
            return Step::Next;
        }

        // ---- 2-source: UDIV/SDIV, variable shifts LSLV/LSRV/ASRV/RORV, and
        // CRC32B/H/W/X + CRC32CB/H/W/X (scalar CRC) ----
        // CRC32/CRC32C share this exact 10-bit outer class with UDIV/SDIV/the
        // shifts (opcode 0b01_0000..=0b01_0111, vs their 0b00_0010..=
        // 0b00_1011), so they're handled as more `opcode` cases here rather
        // than a separate arm — a separate arm placed *after* this one would
        // never be reached, since this arm's catch-all already claims every
        // instruction matching the outer mask.
        if (instr >> 21) & 0x3ff == 0b00_1101_0110 {
            let sf = (instr >> 31) & 1;
            let opcode = (instr >> 10) & 0x3f;
            let rm = reg_field(instr, 16);
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let a = self.read_x(rn);
            let b = self.read_x(rm);
            let width = if sf == 1 { 64u32 } else { 32 };
            let amount = (b % u64::from(width)) as u32;
            let r = match opcode {
                0b00_0010 => udiv(a, b, sf == 1),
                0b00_0011 => sdiv(a, b, sf == 1),
                0b00_1000 => shift_reg(a, 0, amount, sf == 1),
                0b00_1001 => shift_reg(a, 1, amount, sf == 1),
                0b00_1010 => shift_reg(a, 2, amount, sf == 1),
                0b00_1011 => shift_reg(a, 3, amount, sf == 1),
                0b01_0000..=0b01_0111 => {
                    // CRC32<B/H/W/X> (bit12=0) / CRC32C<B/H/W/X> (bit12=1);
                    // the low 2 bits of opcode select B/H/W/X (0/1/2/3).
                    let sz = opcode & 3;
                    if (sz == 3) != (sf == 1) {
                        return Step::Illegal; // the X form requires sf=1
                    }
                    let is_c = (opcode >> 2) & 1 == 1;
                    let nbytes = 1usize << sz;
                    let val = b & ones((nbytes * 8) as u32);
                    let poly = if is_c { 0x82F6_3B78u32 } else { 0xEDB8_8320u32 };
                    let mut crc = a as u32;
                    for i in 0..nbytes {
                        crc = crc32_step(crc, (val >> (i * 8)) as u8, poly);
                    }
                    u64::from(crc)
                }
                _ => return Step::Illegal,
            };
            self.write_x(rd, mask_sf(r, sf));
            return Step::Next;
        }

        // ---- add/subtract with carry: ADC/SBC (+ flag-setting S forms) ----
        if (instr >> 21) & 0xff == 0b1101_0000 {
            let sf = (instr >> 31) & 1;
            let op = (instr >> 30) & 1; // 0 = ADC, 1 = SBC
            let s = (instr >> 29) & 1;
            let rm = reg_field(instr, 16);
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let a = self.read_x(rn);
            let b = if op == 1 {
                !self.read_x(rm)
            } else {
                self.read_x(rm)
            };
            let carry = u128::from(self.flags.c);
            let r = if sf == 1 {
                let sum = u128::from(a) + u128::from(b) + carry;
                let r = sum as u64;
                if s == 1 {
                    self.flags = Flags {
                        n: (r >> 63) & 1 == 1,
                        z: r == 0,
                        c: (sum >> 64) & 1 == 1,
                        v: (((a ^ r) & (b ^ r)) >> 63) & 1 == 1,
                    };
                }
                r
            } else {
                let (a32, b32) = (a as u32, b as u32);
                let sum = u64::from(a32) + u64::from(b32) + carry as u64;
                let r = sum as u32;
                if s == 1 {
                    self.flags = Flags {
                        n: (r >> 31) & 1 == 1,
                        z: r == 0,
                        c: (sum >> 32) & 1 == 1,
                        v: (((a32 ^ r) & (b32 ^ r)) >> 31) & 1 == 1,
                    };
                }
                u64::from(r)
            };
            self.write_x(rd, r);
            return Step::Next;
        }

        // ---- conditional compare: CCMP/CCMN (immediate or register) ----
        if (instr >> 21) & 0xff == 0b1101_0010 {
            let sf = (instr >> 31) & 1;
            let op = (instr >> 30) & 1; // 1 = CCMP (subtract), 0 = CCMN (add)
            let cond = (instr >> 12) & 0xf;
            let rn = reg_field(instr, 5);
            let nzcv = instr & 0xf;
            let operand = if (instr >> 11) & 1 == 1 {
                u64::from((instr >> 16) & 0x1f) // immediate
            } else {
                self.read_x(reg_field(instr, 16)) // register
            };
            if self.cond_holds(cond) {
                self.addsub_flags(self.read_x(rn), operand, op == 1, sf == 1);
            } else {
                self.flags = Flags {
                    n: nzcv & 8 != 0,
                    z: nzcv & 4 != 0,
                    c: nzcv & 2 != 0,
                    v: nzcv & 1 != 0,
                };
            }
            return Step::Next;
        }

        // ---- conditional select: CSEL/CSINC/CSINV/CSNEG ----
        if (instr >> 21) & 0xff == 0b1101_0100 && (instr >> 29) & 1 == 0 {
            let sf = (instr >> 31) & 1;
            let op = (instr >> 30) & 1;
            let op2 = (instr >> 10) & 3;
            if op2 > 1 {
                return Step::Illegal;
            }
            let cond = (instr >> 12) & 0xf;
            let rm = reg_field(instr, 16);
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let r = if self.cond_holds(cond) {
                self.read_x(rn)
            } else {
                let m = self.read_x(rm);
                match (op, op2) {
                    (0, 0) => m,                 // CSEL
                    (0, 1) => m.wrapping_add(1), // CSINC
                    (1, 0) => !m,                // CSINV
                    _ => m.wrapping_neg(),       // CSNEG (1,1)
                }
            };
            self.write_x(rd, mask_sf(r, sf));
            return Step::Next;
        }

        // ---- load/store register (literal, PC-relative): LDR/LDRSW/PRFM ----
        if (instr >> 24) & 0x3f == 0b01_1000 {
            let opc = (instr >> 30) & 3;
            let v = (instr >> 26) & 1;
            let imm19 = sign_extend(u64::from((instr >> 5) & 0x7ffff), 19) << 2;
            let rt = reg_field(instr, 0);
            let addr = (self.pc as i64).wrapping_add(imm19) as u64;
            if v == 1 {
                // SIMD&FP: opc 00=S(4B) 01=D(8B) 10=Q(16B) 11=reserved.
                if opc == 3 {
                    return Step::Illegal;
                }
                return self.ldst_vec(addr, opc + 2, true, rt, mem);
            }
            return match opc {
                0 => self.ldst(addr, 2, 0b01, rt, mem), // LDR Wt (zero-extend)
                1 => self.ldst(addr, 3, 0b01, rt, mem), // LDR Xt
                2 => self.ldst(addr, 2, 0b10, rt, mem), // LDRSW Xt (sign-extend)
                _ => Step::Next,                        // PRFM: prefetch hint, no-op
            };
        }

        // ---- SIMD/FP load/store pair: LDP/STP q/d/s ----
        // bit25 == 0 distinguishes real pairs from SIMD modified-immediate
        // (MOVI/MVNI), which also has bits[29:27]==101 with V==1.
        if (instr >> 27) & 0x7 == 0b101 && (instr >> 26) & 1 == 1 && (instr >> 25) & 1 == 0 {
            let opc = (instr >> 30) & 3;
            if opc == 3 {
                return Step::Illegal;
            }
            let nbytes = 4usize << opc; // S=4, D=8, Q=16
            let class = (instr >> 23) & 3; // 1 post, 2 offset, 3 pre
            let is_load = (instr >> 22) & 1 == 1;
            let imm7 = sign_extend(u64::from((instr >> 15) & 0x7f), 7);
            let rt2 = reg_field(instr, 10);
            let rn = reg_field(instr, 5);
            let rt = reg_field(instr, 0);
            let offset = imm7 * nbytes as i64;
            let base = self.read_sp(rn);
            let addr = if class == 1 {
                base
            } else {
                (base as i64).wrapping_add(offset) as u64
            };
            for (i, r) in [rt, rt2].into_iter().enumerate() {
                let a = addr.wrapping_add((i * nbytes) as u64);
                if is_load {
                    let mut buf = [0u8; 16];
                    if mem.read(a, &mut buf[..nbytes]).is_err() {
                        return Step::Fault {
                            addr: a,
                            write: false,
                        };
                    }
                    self.v[r] = u128::from_le_bytes(buf);
                } else {
                    self.note_store(a, self.v[r], nbytes);
                    let bytes = self.v[r].to_le_bytes();
                    if let Err(e) = mem.write_trap(a, &bytes[..nbytes]) {
                        return Step::Fault {
                            addr: e.fault_addr(),
                            write: true,
                        };
                    }
                }
            }
            if class == 1 || class == 3 {
                self.write_sp(rn, (base as i64).wrapping_add(offset) as u64);
            }
            return Step::Next;
        }

        // ---- CAS/CASP: LSE compare-and-swap (single and register-pair) ----
        // Shares its outer (instr>>24)&0x3f == 0b00_1000 bits with the
        // exclusive-class block below, so — per this file's ordering
        // discipline — the more specific mask (fixed opc/b11:10 = 0b1_1111,
        // and o1 == 1) must be, and is, checked first.
        if (instr >> 24) & 0x3f == 0b00_1000
            && (instr >> 21) & 1 == 1
            && (instr >> 10) & 0x1f == 0b1_1111
        {
            let rs = reg_field(instr, 16);
            let rn = reg_field(instr, 5);
            let rt = reg_field(instr, 0);
            let addr = self.read_sp(rn);
            if (instr >> 23) & 1 == 1 {
                // CAS/CASA/CASL/CASAL/CASB/CASH: size selects B/H/W/X.
                let size = (instr >> 30) & 3;
                return self.cas_single(addr, 1usize << size, rs, rt, mem);
            }
            // CASP/CASPA/CASPL/CASPAL: bit30 selects a 32- or 64-bit pair
            // (bit31 is not part of `size` here, unlike every other class).
            let nbytes = if (instr >> 30) & 1 == 1 { 8 } else { 4 };
            return self.cas_pair(addr, nbytes, rs, rt, mem);
        }

        // ---- LSE atomic memory operations: SWP and LD<op>/ST<op> ----
        // (ADD/CLR/EOR/SET/SMAX/SMIN/UMAX/UMIN, each with +A/+L/+AL
        // acquire/release forms; ST<op> is the same encoding with Rt == 31).
        // Outer bits (111000) are disjoint from every other class in this
        // function (verified against the exclusive/CAS class above and the
        // load/store-pair and register-offset classes elsewhere), so this
        // arm's position relative to them doesn't matter.
        if (instr >> 24) & 0x3f == 0b11_1000 && (instr >> 21) & 1 == 1 && (instr >> 10) & 3 == 0 {
            let size = (instr >> 30) & 3;
            let nbytes = 1usize << size;
            let rs = reg_field(instr, 16);
            let rn = reg_field(instr, 5);
            let rt = reg_field(instr, 0);
            let addr = self.read_sp(rn);
            let o3 = (instr >> 15) & 1; // 1 = SWP, 0 = LD<op>/ST<op>
            let opc = (instr >> 12) & 7;
            if o3 == 1 {
                return if opc == 0 {
                    self.swp(addr, nbytes, rs, rt, mem)
                } else {
                    Step::Illegal // reserved
                };
            }
            return self.ld_op(addr, nbytes, rs, rt, opc, mem);
        }

        // ---- load/store exclusive & acquire/release (LDXR/STXR/LDAR/STLR) ----
        if (instr >> 24) & 0x3f == 0b00_1000 {
            let size = (instr >> 30) & 3;
            let o2 = (instr >> 23) & 1; // 0 = exclusive, 1 = ordered (LDAR/STLR)
            let l = (instr >> 22) & 1; // 1 = load
            let o1 = (instr >> 21) & 1; // 1 = pair: LDXP/STXP, or CAS/CASP above
            if o1 == 1 {
                return Step::Illegal; // LDXP/STXP: Phase 10 (CAS/CASP handled above)
            }
            let rs = reg_field(instr, 16);
            let rn = reg_field(instr, 5);
            let rt = reg_field(instr, 0);
            let addr = self.read_sp(rn);
            if l == 1 {
                let step = self.ldst(addr, size, 0b01, rt, mem);
                if o2 == 0 && matches!(step, Step::Next) {
                    // LDXR/LDAXR opens the local exclusive monitor.
                    self.excl_monitor = true;
                }
                return step;
            }
            if o2 == 0 {
                // STXR/STLXR: only stores — and only succeeds — while the
                // monitor opened by a prior LDXR/LDAXR is still set; any
                // intervening store clears it (see `note_store`).
                if !self.excl_monitor {
                    self.write_x(rs, 1); // status = 1: exclusive access failed
                    return Step::Next;
                }
                let step = self.ldst(addr, size, 0b00, rt, mem);
                if let Step::Fault { .. } = step {
                    // The store faulted (e.g. a copy-on-write page not yet
                    // privatized) and will be retried from the same PC — it did
                    // not architecturally complete. `ldst` already cleared the
                    // monitor via `note_store`, so restore it, or the retry would
                    // see a closed monitor and spuriously fail the store (losing
                    // the write for a lone STXR with no retry loop).
                    self.excl_monitor = true;
                    return step;
                }
                self.excl_monitor = false; // consumed by a completed attempt
                if matches!(step, Step::Next) {
                    self.write_x(rs, 0); // status = 0: success
                }
                return step;
            }
            // STLR: plain ordered store, no exclusive-monitor status register.
            return self.ldst(addr, size, 0b00, rt, mem);
        }

        // ---- load/store pair: LDP/STP (signed offset / pre / post index) ----
        if (instr >> 27) & 0x7 == 0b101 && (instr >> 26) & 1 == 0 && (instr >> 25) & 1 == 0 {
            let opc = (instr >> 30) & 3;
            if opc == 0b11 {
                return Step::Illegal; // reserved
            }
            let is64 = opc == 0b10;
            let is_load = (instr >> 22) & 1 == 1;
            if opc == 0b01 && !is_load {
                return Step::Illegal; // STP has no signed-word form
            }
            let class = (instr >> 23) & 3; // 1 = post, 2 = signed offset, 3 = pre
            let imm7 = sign_extend(u64::from((instr >> 15) & 0x7f), 7);
            let rt2 = reg_field(instr, 10);
            let rn = reg_field(instr, 5);
            let rt = reg_field(instr, 0);
            let nbytes = if is64 { 8usize } else { 4 };
            let offset = imm7 * nbytes as i64;
            let base = self.read_sp(rn);
            let addr = if class == 1 {
                base
            } else {
                (base as i64).wrapping_add(offset) as u64
            };
            for (i, r) in [rt, rt2].into_iter().enumerate() {
                let a = addr.wrapping_add((i * nbytes) as u64);
                if is_load {
                    let mut buf = [0u8; 8];
                    if mem.read(a, &mut buf[..nbytes]).is_err() {
                        return Step::Fault {
                            addr: a,
                            write: false,
                        };
                    }
                    let raw = u64::from_le_bytes(buf);
                    let val = if opc == 0b01 {
                        sign_extend(raw, 32) as u64 // LDPSW
                    } else {
                        raw
                    };
                    self.write_x(r, val);
                } else {
                    self.note_store(a, u128::from(self.read_x(r)), nbytes);
                    let val = self.read_x(r).to_le_bytes();
                    if let Err(e) = mem.write_trap(a, &val[..nbytes]) {
                        return Step::Fault {
                            addr: e.fault_addr(),
                            write: true,
                        };
                    }
                }
            }
            if class == 1 || class == 3 {
                // post/pre index write the updated base back.
                self.write_sp(rn, (base as i64).wrapping_add(offset) as u64);
            }
            return Step::Next;
        }

        // ---- PRFM (immediate unsigned offset / unscaled / register offset): prefetch, no-op ----
        // Shares its outer format bits with the three general GP load/store
        // classes immediately below (unsigned immediate, unscaled/pre/
        // post-index, and register offset) — within each of those, `opc`
        // (bits[23:22]) is a full 2-bit field, and `opc == 0b10` with
        // `size == 0b11` (64-bit) is exactly the encoding space PRFM claims
        // (their generic `opc` only ever uses 00/01 for size 0b11: STR/LDR).
        // Checked first, per this file's ordering discipline, so those
        // classes' `ldst` dispatch doesn't misdecode a prefetch hint's
        // "Rt" (really a prefetch-operation selector, not a register) as a
        // real load destination — which could needlessly fault on memory a
        // prefetch is allowed to ignore.
        if (instr >> 27) & 0x7 == 0b111
            && (instr >> 26) & 1 == 0
            && (instr >> 30) & 3 == 0b11
            && (instr >> 22) & 3 == 0b10
            && matches!((instr >> 24) & 0x3, 0b00 | 0b01)
        {
            return Step::Next;
        }

        // ---- load/store register, unsigned immediate offset ----
        if (instr >> 27) & 0x7 == 0b111 && (instr >> 24) & 0x3 == 0b01 && (instr >> 26) & 1 == 0 {
            let size = (instr >> 30) & 3;
            let opc = (instr >> 22) & 3;
            let imm12 = u64::from((instr >> 10) & 0xfff);
            let rn = reg_field(instr, 5);
            let rt = reg_field(instr, 0);
            let addr = self.read_sp(rn).wrapping_add(imm12 << size);
            return self.ldst(addr, size, opc, rt, mem);
        }

        // ---- load/store register, immediate pre/post-index and unscaled ----
        if (instr >> 27) & 0x7 == 0b111
            && (instr >> 24) & 0x3 == 0b00
            && (instr >> 26) & 1 == 0
            && (instr >> 21) & 1 == 0
        {
            let size = (instr >> 30) & 3;
            let opc = (instr >> 22) & 3;
            let imm9 = sign_extend(u64::from((instr >> 12) & 0x1ff), 9);
            let idx = (instr >> 10) & 3; // 00 unscaled, 01 post, 11 pre
            let rn = reg_field(instr, 5);
            let rt = reg_field(instr, 0);
            let base = self.read_sp(rn);
            let addr = if idx == 0b01 {
                base
            } else {
                (base as i64).wrapping_add(imm9) as u64
            };
            let step = self.ldst(addr, size, opc, rt, mem);
            if matches!(step, Step::Next) && (idx == 0b01 || idx == 0b11) {
                self.write_sp(rn, (base as i64).wrapping_add(imm9) as u64);
            }
            return step;
        }

        // ---- load/store register, register offset ----
        if (instr >> 27) & 0x7 == 0b111
            && (instr >> 24) & 0x3 == 0b00
            && (instr >> 26) & 1 == 0
            && (instr >> 21) & 1 == 1
            && (instr >> 10) & 3 == 0b10
        {
            let size = (instr >> 30) & 3;
            let opc = (instr >> 22) & 3;
            let rm = reg_field(instr, 16);
            let option = (instr >> 13) & 7;
            let s = (instr >> 12) & 1;
            let rn = reg_field(instr, 5);
            let rt = reg_field(instr, 0);
            let shift = if s == 1 { size } else { 0 };
            let addr = self
                .read_sp(rn)
                .wrapping_add(extend_reg(self.read_x(rm), option, shift));
            return self.ldst(addr, size, opc, rt, mem);
        }

        // ---- SIMD/FP load/store register, unsigned immediate offset ----
        if (instr >> 27) & 0x7 == 0b111 && (instr >> 24) & 0x3 == 0b01 && (instr >> 26) & 1 == 1 {
            let size = (instr >> 30) & 3;
            let opc = (instr >> 22) & 3;
            let scale = if opc & 2 != 0 { 4 } else { size }; // opc<1> set => 128-bit
            let imm12 = u64::from((instr >> 10) & 0xfff);
            let rn = reg_field(instr, 5);
            let rt = reg_field(instr, 0);
            let addr = self.read_sp(rn).wrapping_add(imm12 << scale);
            return self.ldst_vec(addr, scale, opc & 1 == 1, rt, mem);
        }

        // ---- SIMD/FP load/store register, immediate pre/post/unscaled ----
        if (instr >> 27) & 0x7 == 0b111
            && (instr >> 24) & 0x3 == 0b00
            && (instr >> 26) & 1 == 1
            && (instr >> 21) & 1 == 0
        {
            let size = (instr >> 30) & 3;
            let opc = (instr >> 22) & 3;
            let scale = if opc & 2 != 0 { 4 } else { size };
            let imm9 = sign_extend(u64::from((instr >> 12) & 0x1ff), 9);
            let idx = (instr >> 10) & 3;
            let rn = reg_field(instr, 5);
            let rt = reg_field(instr, 0);
            let base = self.read_sp(rn);
            let addr = if idx == 0b01 {
                base
            } else {
                (base as i64).wrapping_add(imm9) as u64
            };
            let step = self.ldst_vec(addr, scale, opc & 1 == 1, rt, mem);
            if matches!(step, Step::Next) && (idx == 0b01 || idx == 0b11) {
                self.write_sp(rn, (base as i64).wrapping_add(imm9) as u64);
            }
            return step;
        }

        // ---- LD1/ST1 (single register, multiple structures: 8B/16B) ----
        // Covers `ld1 {Vt.16b},[Xn]` and the post-indexed (reg or #16/#8)
        // forms used heavily by memcpy-style code; element arrangement
        // (`size`) is irrelevant to us since we move the raw bytes as-is.
        if (instr >> 31) & 1 == 0
            && (instr >> 29) & 1 == 0
            && (instr >> 24) & 0x1f == 0b0_1100
            && (instr >> 21) & 1 == 0
            && (instr >> 12) & 0xf == 0b0111
        {
            let q = (instr >> 30) & 1;
            let post = (instr >> 23) & 1 == 1;
            let l = (instr >> 22) & 1;
            let rm = reg_field(instr, 16);
            let rn = reg_field(instr, 5);
            let rt = reg_field(instr, 0);
            let (nbytes, scale) = if q == 1 { (16u64, 4) } else { (8u64, 3) };
            let addr = self.read_sp(rn);
            let step = self.ldst_vec(addr, scale, l == 1, rt, mem);
            if matches!(step, Step::Next) && post {
                let inc = if rm == 31 { nbytes } else { self.read_x(rm) };
                self.write_sp(rn, addr.wrapping_add(inc));
            }
            return step;
        }

        // ---- SIMD modified immediate: MOVI/MVNI/ORR/BIC (vector immediate) ----
        if (instr >> 19) & 0x3ff == 0b0111100000 && (instr >> 10) & 1 == 1 {
            let q = (instr >> 30) & 1;
            let op = (instr >> 29) & 1;
            let cmode = (instr >> 12) & 0xf;
            let imm8 = u64::from((((instr >> 16) & 0x7) << 5) | ((instr >> 5) & 0x1f));
            let rd = reg_field(instr, 0);
            let imm64 = adv_simd_expand_imm(cmode, op, imm8);
            let to_q = |x: u64| {
                if q == 1 {
                    (u128::from(x) << 64) | u128::from(x)
                } else {
                    u128::from(x)
                }
            };
            self.v[rd] = if cmode == 0b1110 || cmode == 0b1111 {
                to_q(imm64) // MOVI (byte/bytemask/fp)
            } else if cmode & 1 == 0 {
                to_q(if op == 0 { imm64 } else { !imm64 }) // MOVI / MVNI
            } else {
                // ORR / BIC immediate: modify the existing register.
                let m = to_q(imm64);
                if op == 0 {
                    self.v[rd] | m
                } else {
                    self.v[rd] & !m
                }
            };
            return Step::Next;
        }

        // ---- DUP Vd.T, Vn.Ts[index] (replicate a vector lane across lanes) ----
        if (instr >> 21) & 0x1ff == 0b0_0111_0000 && (instr >> 10) & 0x3f == 0b00_0001 {
            let q = (instr >> 30) & 1;
            let imm5 = (instr >> 16) & 0x1f;
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let esize = elem_bits(imm5);
            let index = imm5 >> (esize / 8).trailing_zeros().wrapping_add(1);
            let elem = (self.v[rn] >> (u128::from(index) * u128::from(esize))) & ones_u128(esize);
            let width = if q == 1 { 128 } else { 64 };
            let mut val = 0u128;
            let mut shift = 0;
            while shift < width {
                val |= elem << shift;
                shift += esize;
            }
            self.v[rd] = val;
            return Step::Next;
        }

        // ---- DUP Vd.T, Rn (replicate a GP register across lanes) ----
        if (instr >> 21) & 0x1ff == 0b0_0111_0000 && (instr >> 10) & 0x3f == 0b00_0011 {
            let q = (instr >> 30) & 1;
            let imm5 = (instr >> 16) & 0x1f;
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let esize = elem_bits(imm5);
            let elem = u128::from(self.read_x(rn)) & ones_u128(esize);
            let width = if q == 1 { 128 } else { 64 };
            let mut val = 0u128;
            let mut shift = 0;
            while shift < width {
                val |= elem << shift;
                shift += esize;
            }
            self.v[rd] = val;
            return Step::Next;
        }

        // ---- UMOV Rd, Vn.Ts[index] (extract a lane to a GP register) ----
        if (instr >> 21) & 0x1ff == 0b0_0111_0000 && (instr >> 10) & 0x3f == 0b00_1111 {
            let imm5 = (instr >> 16) & 0x1f;
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let esize = elem_bits(imm5);
            let index = imm5 >> (esize / 8).trailing_zeros().wrapping_add(1);
            let elem = (self.v[rn] >> (u128::from(index) * u128::from(esize))) & ones_u128(esize);
            self.write_x(rd, elem as u64);
            return Step::Next;
        }

        // ---- SMOV Rd, Vn.Ts[index] (extract a lane, sign-extended) ----
        if (instr >> 21) & 0x1ff == 0b0_0111_0000 && (instr >> 10) & 0x3f == 0b00_1011 {
            let q = (instr >> 30) & 1;
            let imm5 = (instr >> 16) & 0x1f;
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let esize = elem_bits(imm5);
            let index = imm5 >> (esize / 8).trailing_zeros().wrapping_add(1);
            let elem = (self.v[rn] >> (u128::from(index) * u128::from(esize))) & ones_u128(esize);
            let se = sign_extend(elem as u64, esize);
            let result = if q == 1 {
                se as u64
            } else {
                u64::from(se as u32)
            };
            self.write_x(rd, result);
            return Step::Next;
        }

        // ---- INS Vd.Ts[index], Rn (insert a GP register into a lane) ----
        if (instr >> 21) & 0x1ff == 0b0_0111_0000 && (instr >> 10) & 0x3f == 0b00_0111 {
            let imm5 = (instr >> 16) & 0x1f;
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let esize = elem_bits(imm5);
            let index = imm5 >> (esize / 8).trailing_zeros().wrapping_add(1);
            let shift = u128::from(index) * u128::from(esize);
            let mask = ones_u128(esize) << shift;
            let val = (u128::from(self.read_x(rn)) & ones_u128(esize)) << shift;
            self.v[rd] = (self.v[rd] & !mask) | val;
            return Step::Next;
        }

        // ---- TBL/TBX Vd.Ta, {Vn..Vn+len}, Vm.Ta (table vector lookup) ----
        // `0 Q 0 01110 000 Rm 0 len op 00 Rn Rd`; len (1-4) selects how many
        // consecutive 16B table registers (Vn, Vn+1, ... mod 32) participate;
        // op selects TBL (out-of-range index -> 0) vs TBX (out-of-range index
        // leaves the destination lane unchanged). Distinct opcode-field bit10
        // (0 here, 1 throughout the DUP/INS/UMOV/SMOV family above) keeps
        // this from ever shadowing — or being shadowed by — those arms.
        if (instr >> 24) & 0x1f == 0b0_1110
            && (instr >> 29) & 1 == 0
            && (instr >> 21) & 0x7 == 0
            && (instr >> 15) & 1 == 0
            && (instr >> 10) & 0x3 == 0
        {
            let q = (instr >> 30) & 1;
            let len = ((instr >> 13) & 3) + 1; // number of 16B table registers
            let is_tbx = (instr >> 12) & 1 == 1;
            let rm = reg_field(instr, 16);
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let table_bytes = (len as usize) * 16;
            let idx_bytes = self.v[rm].to_le_bytes();
            let nbytes = if q == 1 { 16usize } else { 8 };
            let mut out = self.v[rd].to_le_bytes();
            for i in 0..nbytes {
                let idx = idx_bytes[i] as usize;
                if idx < table_bytes {
                    let reg = (rn + idx / 16) & 0x1f;
                    out[i] = self.v[reg].to_le_bytes()[idx % 16];
                } else if !is_tbx {
                    out[i] = 0;
                } // TBX: out-of-range leaves the destination byte unchanged.
            }
            for b in out.iter_mut().skip(nbytes) {
                *b = 0;
            }
            self.v[rd] = u128::from_le_bytes(out);
            return Step::Next;
        }

        // ---- EXT Vd.T, Vn.T, Vm.T, #imm4 (extract from a register pair) ----
        if (instr >> 24) & 0x3f == 0b10_1110
            && (instr >> 22) & 3 == 0
            && (instr >> 21) & 1 == 0
            && (instr >> 15) & 1 == 0
            && (instr >> 10) & 1 == 0
        {
            let q = (instr >> 30) & 1;
            let imm4 = (instr >> 11) & 0xf;
            let rm = reg_field(instr, 16);
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let size = if q == 1 { 16usize } else { 8 };
            let mut concat = [0u8; 32];
            concat[..16].copy_from_slice(&self.v[rn].to_le_bytes());
            concat[16..].copy_from_slice(&self.v[rm].to_le_bytes());
            let off = imm4 as usize;
            let mut out = [0u8; 16];
            out[..size].copy_from_slice(&concat[off..off + size]);
            self.v[rd] = u128::from_le_bytes(out);
            return Step::Next;
        }

        // ---- ZIP1/ZIP2/UZP1/UZP2/TRN1/TRN2 (vector permute) ----
        if (instr >> 24) & 0x1f == 0b0_1110
            && (instr >> 29) & 1 == 0
            && (instr >> 21) & 1 == 0
            && (instr >> 15) & 1 == 0
            && (instr >> 11) & 1 == 1
            && (instr >> 10) & 1 == 0
        {
            let q = (instr >> 30) & 1;
            let size = (instr >> 22) & 3;
            let opcode = (instr >> 12) & 0x7;
            let rm = reg_field(instr, 16);
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let esize = 8u32 << size;
            let width = if q == 1 { 128 } else { 64 };
            let lanes = width / esize;
            let half = lanes / 2;
            let mask = ones_u128(esize);
            let lane = |v: u128, i: u32| (v >> (i * esize)) & mask;
            let second = opcode & 0b100 != 0; // "2" (vs "1") variant
            let mut result = 0u128;
            match opcode & 0b011 {
                0b11 => {
                    // ZIP1/ZIP2: interleave one half of Vn with the same half of Vm.
                    let base = if second { half } else { 0 };
                    for i in 0..half {
                        result |= lane(self.v[rn], base + i) << (2 * i * esize);
                        result |= lane(self.v[rm], base + i) << ((2 * i + 1) * esize);
                    }
                }
                0b01 => {
                    // UZP1/UZP2: gather every-other element (even/odd) from Vn then Vm.
                    let start = u32::from(second);
                    for i in 0..half {
                        result |= lane(self.v[rn], 2 * i + start) << (i * esize);
                        result |= lane(self.v[rm], 2 * i + start) << ((half + i) * esize);
                    }
                }
                _ => {
                    // TRN1/TRN2: interleave even/odd elements from Vn and Vm.
                    let start = u32::from(second);
                    for i in 0..half {
                        result |= lane(self.v[rn], 2 * i + start) << (2 * i * esize);
                        result |= lane(self.v[rm], 2 * i + start) << ((2 * i + 1) * esize);
                    }
                }
            }
            self.v[rd] = result & (if q == 1 { u128::MAX } else { ones_u128(64) });
            return Step::Next;
        }

        // ---- FMOV between GP and SIMD/FP registers (bit-exact, no convert) ----
        if (instr >> 24) & 0x7f == 0b0011110 && (instr >> 21) & 1 == 1 && (instr >> 10) & 0x3f == 0
        {
            let ftype = (instr >> 22) & 3; // 00 = S (32-bit), 01 = D (64-bit)
            let sf = (instr >> 31) & 1; // 0 = W (32-bit), 1 = X (64-bit) int register
            let rmode = (instr >> 19) & 3;
            let opcode = (instr >> 16) & 7;
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            match (rmode, opcode) {
                (0, 0b010 | 0b011) => {
                    // SCVTF / UCVTF: integer register -> floating-point.
                    let signed = opcode == 0b010;
                    let val = self.read_x(rn);
                    match ftype {
                        0 => self.set_fp32(rd, int_to_f32(val, signed, sf == 1)),
                        1 => self.set_fp64(rd, int_to_f64(val, signed, sf == 1)),
                        _ => return Step::Illegal,
                    }
                }
                (3, 0b000 | 0b001) => {
                    // FCVTZS / FCVTZU: floating-point -> integer, round toward zero.
                    let signed = opcode == 0b000;
                    let x = match ftype {
                        0 => f64::from(self.fp32(rn)),
                        1 => self.fp64(rn),
                        _ => return Step::Illegal,
                    };
                    self.write_x(rd, fp_to_int(x, signed, sf == 1));
                }
                (0, 0b111) => {
                    // GP -> FP: Vd = Rn (32 or 64 bits), upper bits cleared.
                    let val = self.read_x(rn);
                    self.v[rd] = if ftype == 0 {
                        u128::from(val & 0xffff_ffff)
                    } else {
                        u128::from(val)
                    };
                }
                (0, 0b110) => {
                    // FP -> GP: Rd = Vn low 32/64 bits.
                    let val = if ftype == 0 {
                        (self.v[rn] as u64) & 0xffff_ffff
                    } else {
                        self.v[rn] as u64
                    };
                    self.write_x(rd, val);
                }
                (1, 0b111) => {
                    // GP -> Vd.D[1] (insert into the high 64 bits).
                    let val = u128::from(self.read_x(rn));
                    self.v[rd] = (self.v[rd] & u128::from(u64::MAX)) | (val << 64);
                }
                (1, 0b110) => self.write_x(rd, (self.v[rn] >> 64) as u64), // Vn.D[1] -> GP
                _ => return Step::Illegal, // FCVT*/SCVTF/etc: not implemented
            }
            return Step::Next;
        }

        // ---- scalar FP data-processing (1 source) ----
        if (instr >> 24) & 0x7f == 0b0011110
            && (instr >> 21) & 1 == 1
            && (instr >> 10) & 0x1f == 0b10000
        {
            let ftype = (instr >> 22) & 3;
            let opcode = (instr >> 15) & 0x3f;
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            // FCVT (precision change): opcode = 0001<dst>, dst 00=S 01=D 11=H.
            if opcode & 0b11_1100 == 0b00_0100 {
                match (ftype, opcode & 3) {
                    (0, 1) => self.set_fp64(rd, f64::from(self.fp32(rn))), // S -> D
                    (1, 0) => self.set_fp32(rd, self.fp64(rn) as f32),     // D -> S
                    (0, 3) => self.set_fp16(rd, f32_to_f16(self.fp32(rn))), // S -> H
                    (1, 3) => self.set_fp16(rd, f32_to_f16(self.fp64(rn) as f32)), // D -> H
                    (3, 0) => self.set_fp32(rd, f16_to_f32(self.fp16(rn))), // H -> S
                    (3, 1) => self.set_fp64(rd, f64::from(f16_to_f32(self.fp16(rn)))), // H -> D
                    _ => return Step::Illegal,
                }
                return Step::Next;
            }
            match ftype {
                0 => {
                    let a = self.fp32(rn);
                    let r = match opcode {
                        0b000000 => a,                   // FMOV
                        0b000001 => a.abs(),             // FABS
                        0b000010 => -a,                  // FNEG
                        0b000011 => a.sqrt(),            // FSQRT
                        0b001000 => a.round_ties_even(), // FRINTN
                        0b001001 => a.ceil(),            // FRINTP
                        0b001010 => a.floor(),           // FRINTM
                        0b001011 => a.trunc(),           // FRINTZ
                        0b001100 => a.round(),           // FRINTA
                        _ => return Step::Illegal,
                    };
                    self.set_fp32(rd, r);
                }
                1 => {
                    let a = self.fp64(rn);
                    let r = match opcode {
                        0b000000 => a,
                        0b000001 => a.abs(),
                        0b000010 => -a,
                        0b000011 => a.sqrt(),
                        0b001000 => a.round_ties_even(),
                        0b001001 => a.ceil(),
                        0b001010 => a.floor(),
                        0b001011 => a.trunc(),
                        0b001100 => a.round(),
                        _ => return Step::Illegal,
                    };
                    self.set_fp64(rd, r);
                }
                _ => return Step::Illegal,
            }
            return Step::Next;
        }

        // ---- scalar FP data-processing (2 source) ----
        if (instr >> 24) & 0x7f == 0b0011110
            && (instr >> 21) & 1 == 1
            && (instr >> 10) & 0x3 == 0b10
        {
            let ftype = (instr >> 22) & 3;
            let opcode = (instr >> 12) & 0xf;
            let rm = reg_field(instr, 16);
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            match ftype {
                0 => {
                    let (a, b) = (self.fp32(rn), self.fp32(rm));
                    let r = match opcode {
                        0b0000 => a * b,        // FMUL
                        0b0001 => a / b,        // FDIV
                        0b0010 => a + b,        // FADD
                        0b0011 => a - b,        // FSUB
                        0b0100 => fmax32(a, b), // FMAX
                        0b0101 => fmin32(a, b), // FMIN
                        0b0110 => a.max(b),     // FMAXNM
                        0b0111 => a.min(b),     // FMINNM
                        0b1000 => -(a * b),     // FNMUL
                        _ => return Step::Illegal,
                    };
                    self.set_fp32(rd, r);
                }
                1 => {
                    let (a, b) = (self.fp64(rn), self.fp64(rm));
                    let r = match opcode {
                        0b0000 => a * b,
                        0b0001 => a / b,
                        0b0010 => a + b,
                        0b0011 => a - b,
                        0b0100 => fmax64(a, b),
                        0b0101 => fmin64(a, b),
                        0b0110 => a.max(b),
                        0b0111 => a.min(b),
                        0b1000 => -(a * b),
                        _ => return Step::Illegal,
                    };
                    self.set_fp64(rd, r);
                }
                _ => return Step::Illegal,
            }
            return Step::Next;
        }

        // ---- scalar FP data-processing (3 source): FMADD/FMSUB/FNMADD/FNMSUB ----
        // Distinct top-level mask (0b0011111) from the 1-/2-source classes above
        // (0b0011110), so this can't shadow or be shadowed by them.
        if (instr >> 24) & 0x7f == 0b0011111 {
            let ftype = (instr >> 22) & 3;
            let o1 = (instr >> 21) & 1;
            let o0 = (instr >> 15) & 1;
            let rm = reg_field(instr, 16);
            let ra = reg_field(instr, 10);
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            match ftype {
                0 => {
                    let (n, m, a) = (self.fp32(rn), self.fp32(rm), self.fp32(ra));
                    let r = match (o1, o0) {
                        (0, 0) => n.mul_add(m, a),     // FMADD:  a + n*m
                        (0, 1) => (-n).mul_add(m, a),  // FMSUB:  a - n*m
                        (1, 0) => (-n).mul_add(m, -a), // FNMADD: -a - n*m
                        _ => n.mul_add(m, -a),         // FNMSUB: n*m - a
                    };
                    self.set_fp32(rd, r);
                }
                1 => {
                    let (n, m, a) = (self.fp64(rn), self.fp64(rm), self.fp64(ra));
                    let r = match (o1, o0) {
                        (0, 0) => n.mul_add(m, a),
                        (0, 1) => (-n).mul_add(m, a),
                        (1, 0) => (-n).mul_add(m, -a),
                        _ => n.mul_add(m, -a),
                    };
                    self.set_fp64(rd, r);
                }
                _ => return Step::Illegal,
            }
            return Step::Next;
        }

        // ---- scalar FP compare: FCMP/FCMPE (register and #0.0) ----
        if (instr >> 24) & 0x7f == 0b0011110
            && (instr >> 21) & 1 == 1
            && (instr >> 14) & 0x3 == 0b00
            && (instr >> 10) & 0xf == 0b1000
        {
            let ftype = (instr >> 22) & 3;
            let rm = reg_field(instr, 16);
            let rn = reg_field(instr, 5);
            let cmp_zero = (instr >> 3) & 1 == 1; // opcode2<3>: compare against +0.0
            let (a, b) = match ftype {
                0 => (
                    f64::from(self.fp32(rn)),
                    if cmp_zero {
                        0.0
                    } else {
                        f64::from(self.fp32(rm))
                    },
                ),
                1 => (self.fp64(rn), if cmp_zero { 0.0 } else { self.fp64(rm) }),
                _ => return Step::Illegal,
            };
            self.set_fcmp_flags(a, b);
            return Step::Next;
        }

        // ---- scalar FP conditional compare: FCCMP/FCCMPE ----
        if (instr >> 24) & 0x7f == 0b0011110
            && (instr >> 21) & 1 == 1
            && (instr >> 10) & 0x3 == 0b01
        {
            let ftype = (instr >> 22) & 3;
            let rm = reg_field(instr, 16);
            let cond = (instr >> 12) & 0xf;
            let rn = reg_field(instr, 5);
            let nzcv = instr & 0xf;
            if self.cond_holds(cond) {
                let (a, b) = match ftype {
                    0 => (f64::from(self.fp32(rn)), f64::from(self.fp32(rm))),
                    1 => (self.fp64(rn), self.fp64(rm)),
                    _ => return Step::Illegal,
                };
                self.set_fcmp_flags(a, b);
            } else {
                self.flags = Flags {
                    n: nzcv & 8 != 0,
                    z: nzcv & 4 != 0,
                    c: nzcv & 2 != 0,
                    v: nzcv & 1 != 0,
                };
            }
            return Step::Next;
        }

        // ---- scalar FP conditional select: FCSEL ----
        if (instr >> 24) & 0x7f == 0b0011110
            && (instr >> 21) & 1 == 1
            && (instr >> 10) & 0x3 == 0b11
        {
            let ftype = (instr >> 22) & 3;
            let rm = reg_field(instr, 16);
            let cond = (instr >> 12) & 0xf;
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let src = if self.cond_holds(cond) { rn } else { rm };
            match ftype {
                0 => self.set_fp32(rd, self.fp32(src)),
                1 => self.set_fp64(rd, self.fp64(src)),
                _ => return Step::Illegal,
            }
            return Step::Next;
        }

        // ---- scalar FP immediate: FMOV (scalar, immediate) via VFPExpandImm ----
        if (instr >> 24) & 0x7f == 0b0011110
            && (instr >> 21) & 1 == 1
            && (instr >> 10) & 0x7 == 0b100
            && (instr >> 5) & 0x1f == 0
        {
            let ftype = (instr >> 22) & 3;
            let imm8 = (instr >> 13) & 0xff;
            let rd = reg_field(instr, 0);
            match ftype {
                0 => self.v[rd] = u128::from(vfp_expand_imm32(imm8)),
                1 => self.v[rd] = u128::from(vfp_expand_imm64(imm8)),
                _ => return Step::Illegal,
            }
            return Step::Next;
        }

        // ---- SIMD three-same logical: AND/BIC/ORR(MOV)/ORN/EOR/BSL/BIT/BIF ----
        // Checked before the integer three-same arm (which uses a looser mask).
        if (instr >> 24) & 0x1f == 0b0_1110
            && (instr >> 21) & 1 == 1
            && (instr >> 10) & 0x3f == 0b000111
        {
            let q = (instr >> 30) & 1;
            let u = (instr >> 29) & 1;
            let size = (instr >> 22) & 3;
            let rm = reg_field(instr, 16);
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let (vn, vm, vd) = (self.v[rn], self.v[rm], self.v[rd]);
            let out = match (u, size) {
                (0, 0b00) => vn & vm,                // AND
                (0, 0b01) => vn & !vm,               // BIC
                (0, 0b10) => vn | vm,                // ORR (MOV when Rn==Rm)
                (0, 0b11) => vn | !vm,               // ORN
                (1, 0b00) => vn ^ vm,                // EOR
                (1, 0b01) => (vd & vn) | (!vd & vm), // BSL
                (1, 0b10) => (vm & vn) | (!vm & vd), // BIT
                _ => (!vm & vn) | (vm & vd),         // BIF (1,0b11)
            };
            let mask = if q == 1 { u128::MAX } else { ones_u128(64) };
            self.v[rd] = out & mask;
            return Step::Next;
        }

        // ---- SIMD three-same (floating-point): FADD/FSUB/FMUL/FDIV/FMLA/FMLS/
        // FABD/FRECPS/FRSQRTS/FCMEQ/FCMGE/FCMGT (2S/4S/2D) ----
        // Shares its outer bit pattern with the integer three-same class below,
        // but is disambiguated (and thus must be checked first) by the exact
        // (U, a, opcode) combination baked into the match guard, so it never
        // shadows — and is never shadowed by — the integer arm.
        if (instr >> 24) & 0x9f == 0b0_0001110
            && (instr >> 21) & 1 == 1
            && (instr >> 10) & 1 == 1
            && matches!(
                ((instr >> 29) & 1, (instr >> 23) & 1, (instr >> 11) & 0x1f),
                (0, 0 | 1, 0b11010 | 0b11001 | 0b11111) // FADD/FSUB/FRECPS/FRSQRTS, FMLA/FMLS
                    | (1, 0, 0b11011 | 0b11111) // FMUL / FDIV
                    | (1, 1, 0b11010) // FABD
                    | (0, 0, 0b11100) // FCMEQ
                    | (1, 0 | 1, 0b11100) // FCMGE / FCMGT
            )
        {
            let q = (instr >> 30) & 1;
            let uns = (instr >> 29) & 1;
            let asub = (instr >> 23) & 1; // selects SUB/MLS/GT within the U-selected op
            let dbl = (instr >> 22) & 1;
            let opcode = (instr >> 11) & 0x1f;
            let rm = reg_field(instr, 16);
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            if dbl == 1 && q == 0 {
                return Step::Illegal; // 1D is not a valid vector arrangement here
            }
            self.v[rd] = if dbl == 0 {
                let lanes = if q == 1 { 4 } else { 2 };
                let mut result = 0u128;
                for i in 0..lanes {
                    let sh = i * 32;
                    let lhs = f32::from_bits(((self.v[rn] >> sh) & 0xffff_ffff) as u32);
                    let rhs = f32::from_bits(((self.v[rm] >> sh) & 0xffff_ffff) as u32);
                    let acc = || f32::from_bits(((self.v[rd] >> sh) & 0xffff_ffff) as u32);
                    let bits = match (uns, asub, opcode) {
                        (0, 0, 0b11010) => (lhs + rhs).to_bits(),
                        (0, 1, 0b11010) => (lhs - rhs).to_bits(),
                        (1, 0, 0b11011) => (lhs * rhs).to_bits(),
                        (1, 0, 0b11111) => (lhs / rhs).to_bits(),
                        (0, 0, 0b11001) => lhs.mul_add(rhs, acc()).to_bits(),
                        (0, 1, 0b11001) => (-lhs).mul_add(rhs, acc()).to_bits(),
                        (1, 1, 0b11010) => (lhs - rhs).abs().to_bits(), // FABD
                        (0, 0, 0b11111) => (2.0 - lhs * rhs).to_bits(), // FRECPS
                        (0, 1, 0b11111) => ((3.0 - lhs * rhs) / 2.0).to_bits(), // FRSQRTS
                        (0, 0, 0b11100) => fbits_bool32(fp_eq(f64::from(lhs), f64::from(rhs))),
                        (1, 0, 0b11100) => fbits_bool32(f64::from(lhs) >= f64::from(rhs)),
                        _ => fbits_bool32(f64::from(lhs) > f64::from(rhs)), // (1,1,0b11100) FCMGT
                    };
                    result |= u128::from(bits) << sh;
                }
                result
            } else {
                let mut result = 0u128;
                for i in 0..2u32 {
                    let sh = i * 64;
                    let lhs = f64::from_bits(((self.v[rn] >> sh) & u128::from(u64::MAX)) as u64);
                    let rhs = f64::from_bits(((self.v[rm] >> sh) & u128::from(u64::MAX)) as u64);
                    let acc = || f64::from_bits(((self.v[rd] >> sh) & u128::from(u64::MAX)) as u64);
                    let bits = match (uns, asub, opcode) {
                        (0, 0, 0b11010) => (lhs + rhs).to_bits(),
                        (0, 1, 0b11010) => (lhs - rhs).to_bits(),
                        (1, 0, 0b11011) => (lhs * rhs).to_bits(),
                        (1, 0, 0b11111) => (lhs / rhs).to_bits(),
                        (0, 0, 0b11001) => lhs.mul_add(rhs, acc()).to_bits(),
                        (0, 1, 0b11001) => (-lhs).mul_add(rhs, acc()).to_bits(),
                        (1, 1, 0b11010) => (lhs - rhs).abs().to_bits(), // FABD
                        (0, 0, 0b11111) => (2.0 - lhs * rhs).to_bits(), // FRECPS
                        (0, 1, 0b11111) => ((3.0 - lhs * rhs) / 2.0).to_bits(), // FRSQRTS
                        (0, 0, 0b11100) => fbits_bool64(fp_eq(lhs, rhs)),
                        (1, 0, 0b11100) => fbits_bool64(lhs >= rhs),
                        _ => fbits_bool64(lhs > rhs), // (1,1,0b11100) FCMGT
                    };
                    result |= u128::from(bits) << sh;
                }
                result
            };
            return Step::Next;
        }

        // ---- SIMD scalar three-same (floating-point): FABD/FRECPS/FRSQRTS ----
        // The scalar (bit28 == 1) counterpart of the vector three-same FP arm
        // above. No other scalar three-same FP op is implemented here: plain
        // scalar FADD/FSUB/FMUL/FDIV go through the unrelated, non-SIMD "FP
        // data-processing (2 source)" encoding instead (see below), and
        // scalar FMLA/FMLS/FMUL only exist in their by-element form — so only
        // these three opcodes are handled by this arm.
        if (instr >> 24) & 0x9f == 0b0_0011110
            && (instr >> 21) & 1 == 1
            && (instr >> 10) & 1 == 1
            && matches!(
                ((instr >> 29) & 1, (instr >> 23) & 1, (instr >> 11) & 0x1f),
                (1, 1, 0b11010) // FABD
                    | (0, 0 | 1, 0b11111) // FRECPS / FRSQRTS
            )
        {
            let uns = (instr >> 29) & 1;
            let asub = (instr >> 23) & 1;
            let dbl = (instr >> 22) & 1;
            let opcode = (instr >> 11) & 0x1f;
            let rm = reg_field(instr, 16);
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            if dbl == 0 {
                let (a, b) = (self.fp32(rn), self.fp32(rm));
                let r = match (uns, asub, opcode) {
                    (1, 1, 0b11010) => (a - b).abs(), // FABD
                    (0, 0, 0b11111) => 2.0 - a * b,   // FRECPS
                    _ => (3.0 - a * b) / 2.0,         // (0,1,0b11111) FRSQRTS
                };
                self.set_fp32(rd, r);
            } else {
                let (a, b) = (self.fp64(rn), self.fp64(rm));
                let r = match (uns, asub, opcode) {
                    (1, 1, 0b11010) => (a - b).abs(),
                    (0, 0, 0b11111) => 2.0 - a * b,
                    _ => (3.0 - a * b) / 2.0,
                };
                self.set_fp64(rd, r);
            }
            return Step::Next;
        }

        // ---- SIMD three-same pairwise (integer): ADDP/SMAXP/SMINP/UMAXP/
        // UMINP (vector) ----
        // Shares its outer bits with the general integer three-same arm below
        // but these opcodes (0b10111/0b10100/0b10101) aren't in that arm's op
        // table, so — per this file's ordering discipline — this
        // specific-opcode arm must come first, or the general arm's
        // catch-all would swallow these as Illegal.
        if (instr >> 24) & 0x9f == 0b0_0001110
            && (instr >> 21) & 1 == 1
            && (instr >> 10) & 1 == 1
            && matches!((instr >> 11) & 0x1f, 0b10111 | 0b10100 | 0b10101)
        {
            let q = (instr >> 30) & 1;
            let uns = (instr >> 29) & 1;
            let size = (instr >> 22) & 3;
            let opcode = (instr >> 11) & 0x1f;
            let rm = reg_field(instr, 16);
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            if opcode == 0b10111 && uns == 1 {
                return Step::Illegal; // no unsigned ADDP
            }
            let esize = 8u32 << size;
            let lanes = (if q == 1 { 128 } else { 64 }) / esize;
            let mask = ones_u128(esize);
            let lane = |v: u128, i: u32| (v >> (i * esize)) & mask;
            // concat[j] = Vn[j] for j < lanes, else Vm[j - lanes] — pairwise
            // ops act on adjacent elements of Vn:Vm concatenated together.
            let concat = |v0: u128, v1: u128, j: u32| {
                if j < lanes {
                    lane(v0, j)
                } else {
                    lane(v1, j - lanes)
                }
            };
            let mut result = 0u128;
            for i in 0..lanes {
                let (lhs, rhs) = (
                    concat(self.v[rn], self.v[rm], 2 * i),
                    concat(self.v[rn], self.v[rm], 2 * i + 1),
                );
                let pair = match (opcode, uns) {
                    (0b10111, _) => lhs.wrapping_add(rhs) & mask, // ADDP
                    (0b10100, 0) => {
                        // SMAXP
                        if sign_extend(lhs as u64, esize) >= sign_extend(rhs as u64, esize) {
                            lhs
                        } else {
                            rhs
                        }
                    }
                    (0b10100, _) => {
                        if lhs >= rhs { lhs } else { rhs } // UMAXP
                    }
                    (0b10101, 0) => {
                        // SMINP
                        if sign_extend(lhs as u64, esize) <= sign_extend(rhs as u64, esize) {
                            lhs
                        } else {
                            rhs
                        }
                    }
                    _ => {
                        if lhs <= rhs { lhs } else { rhs } // UMINP
                    }
                };
                result |= pair << (i * esize);
            }
            self.v[rd] = result;
            return Step::Next;
        }

        // ---- FADDP/FMAXP (vector, pairwise floating-point) ----
        // Same outer bits as the FP three-same arm above, but the (U=1, a=0,
        // opcode=0b11010/0b11110) combination isn't in that arm's `matches!`
        // list, so — per the ordering discipline — this must be checked
        // before the general integer three-same arm's catch-all below.
        if (instr >> 24) & 0x9f == 0b0_0001110
            && (instr >> 21) & 1 == 1
            && (instr >> 10) & 1 == 1
            && (instr >> 29) & 1 == 1
            && (instr >> 23) & 1 == 0
            && matches!((instr >> 11) & 0x1f, 0b11010 | 0b11110)
        {
            let q = (instr >> 30) & 1;
            let dbl = (instr >> 22) & 1;
            let is_max = (instr >> 11) & 0x1f == 0b11110;
            let rm = reg_field(instr, 16);
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            if dbl == 1 && q == 0 {
                return Step::Illegal;
            }
            self.v[rd] = if dbl == 0 {
                let lanes = if q == 1 { 4 } else { 2 };
                let lane = |v: u128, i: u32| f32::from_bits(((v >> (i * 32)) & 0xffff_ffff) as u32);
                let concat = |v0: u128, v1: u128, j: u32| {
                    if j < lanes {
                        lane(v0, j)
                    } else {
                        lane(v1, j - lanes)
                    }
                };
                let mut result = 0u128;
                for i in 0..lanes {
                    let (a, b) = (
                        concat(self.v[rn], self.v[rm], 2 * i),
                        concat(self.v[rn], self.v[rm], 2 * i + 1),
                    );
                    let r = if is_max { fmax32(a, b) } else { a + b };
                    result |= u128::from(r.to_bits()) << (i * 32);
                }
                result
            } else {
                let lane = |v: u128, i: u32| {
                    f64::from_bits(((v >> (i * 64)) & u128::from(u64::MAX)) as u64)
                };
                let concat = |v0: u128, v1: u128, j: u32| {
                    if j < 2 { lane(v0, j) } else { lane(v1, j - 2) }
                };
                let mut result = 0u128;
                for i in 0..2u32 {
                    let (a, b) = (
                        concat(self.v[rn], self.v[rm], 2 * i),
                        concat(self.v[rn], self.v[rm], 2 * i + 1),
                    );
                    let r = if is_max { fmax64(a, b) } else { a + b };
                    result |= u128::from(r.to_bits()) << (i * 64);
                }
                result
            };
            return Step::Next;
        }

        // ---- SIMD three-same (integer): ADD/SUB/compares/SSHL/USHL (vector) ----
        if (instr >> 24) & 0x9f == 0b0_0001110 && (instr >> 21) & 1 == 1 && (instr >> 10) & 1 == 1 {
            let q = (instr >> 30) & 1;
            let u = (instr >> 29) & 1;
            let size = (instr >> 22) & 3;
            let opcode = (instr >> 11) & 0x1f;
            let rm = reg_field(instr, 16);
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            // 0=ADD 1=SUB 2=CMEQ 3=CMGT 4=CMGE 5=CMHI 6=CMHS 7=SSHL 8=USHL 9=MUL
            // 10=MLA 11=MLS 12=SMAX 13=SMIN 14=UMAX 15=UMIN 16=SQADD 17=UQADD
            // 18=SQSUB 19=UQSUB 20=SQSHL 21=UQSHL 22=SQRSHL 23=UQRSHL.
            let op = match (u, opcode) {
                (0, 0b10000) => 0u8,
                (1, 0b10000) => 1,
                (1, 0b10001) => 2,
                (0, 0b00110) => 3,
                (0, 0b00111) => 4,
                (1, 0b00110) => 5,
                (1, 0b00111) => 6,
                (0, 0b01000) => 7,
                (1, 0b01000) => 8,
                (0, 0b10011) => 9,
                (0, 0b10010) => 10,
                (1, 0b10010) => 11,
                (0, 0b01100) => 12,
                (0, 0b01101) => 13,
                (1, 0b01100) => 14,
                (1, 0b01101) => 15,
                (0, 0b00001) => 16,
                (1, 0b00001) => 17,
                (0, 0b00101) => 18,
                (1, 0b00101) => 19,
                (0, 0b01001) => 20,
                (1, 0b01001) => 21,
                (0, 0b01011) => 22,
                (1, 0b01011) => 23,
                _ => return Step::Illegal,
            };
            let esize = 8u32 << size;
            let lanes = (if q == 1 { 128 } else { 64 }) / esize;
            let mask = ones_u128(esize);
            let mut result = 0u128;
            for i in 0..lanes {
                let sh = i * esize;
                let x = (self.v[rn] >> sh) & mask;
                let y = (self.v[rm] >> sh) & mask;
                let lane = match op {
                    0 => x.wrapping_add(y) & mask,
                    1 => x.wrapping_sub(y) & mask,
                    2 => {
                        if x == y {
                            mask
                        } else {
                            0
                        }
                    }
                    3 => {
                        if sign_extend(x as u64, esize) > sign_extend(y as u64, esize) {
                            mask
                        } else {
                            0
                        }
                    }
                    4 => {
                        if sign_extend(x as u64, esize) >= sign_extend(y as u64, esize) {
                            mask
                        } else {
                            0
                        }
                    }
                    5 => {
                        if x > y {
                            mask
                        } else {
                            0
                        }
                    }
                    6 => {
                        if x >= y {
                            mask
                        } else {
                            0
                        }
                    }
                    7 | 8 => {
                        // SSHL (op==7, signed) / USHL (op==8, unsigned): shift amount is
                        // the low signed byte of the corresponding y-lane.
                        let amt = sign_extend(y as u64 & 0xff, 8);
                        simd_shl(x, amt, esize, op == 7)
                    }
                    9 => x.wrapping_mul(y) & mask, // MUL
                    10 => {
                        // MLA: Vd += Vn * Vm (per lane, using the original Vd).
                        let acc = (self.v[rd] >> sh) & mask;
                        acc.wrapping_add(x.wrapping_mul(y)) & mask
                    }
                    11 => {
                        // MLS: Vd -= Vn * Vm (per lane, using the original Vd).
                        let acc = (self.v[rd] >> sh) & mask;
                        acc.wrapping_sub(x.wrapping_mul(y)) & mask
                    }
                    12 => {
                        // SMAX
                        if sign_extend(x as u64, esize) > sign_extend(y as u64, esize) {
                            x
                        } else {
                            y
                        }
                    }
                    13 => {
                        // SMIN
                        if sign_extend(x as u64, esize) < sign_extend(y as u64, esize) {
                            x
                        } else {
                            y
                        }
                    }
                    14 => {
                        if x > y { x } else { y } // UMAX
                    }
                    15 => {
                        if x < y { x } else { y } // UMIN
                    }
                    16 => signed_sat(
                        i128::from(sign_extend(x as u64, esize))
                            + i128::from(sign_extend(y as u64, esize)),
                        esize,
                    ), // SQADD
                    17 => unsigned_sat(x as i128 + y as i128, esize), // UQADD
                    18 => signed_sat(
                        i128::from(sign_extend(x as u64, esize))
                            - i128::from(sign_extend(y as u64, esize)),
                        esize,
                    ), // SQSUB
                    19 => unsigned_sat(x as i128 - y as i128, esize), // UQSUB
                    20 => sat_shl(x, sign_extend(y as u64 & 0xff, 8), esize, true, false), // SQSHL
                    21 => sat_shl(x, sign_extend(y as u64 & 0xff, 8), esize, false, false), // UQSHL
                    22 => sat_shl(x, sign_extend(y as u64 & 0xff, 8), esize, true, true), // SQRSHL
                    _ => sat_shl(x, sign_extend(y as u64 & 0xff, 8), esize, false, true), // UQRSHL (23)
                };
                result |= lane << sh;
            }
            self.v[rd] = result;
            return Step::Next;
        }

        // ---- NOT/MVN (vector): bitwise complement ----
        if (instr >> 24) & 0x1f == 0b0_1110
            && (instr >> 29) & 1 == 1
            && (instr >> 22) & 3 == 0
            && (instr >> 17) & 0x1f == 0b1_0000
            && (instr >> 12) & 0x1f == 0b0_0101
            && (instr >> 10) & 3 == 0b10
        {
            let q = (instr >> 30) & 1;
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let mask = if q == 1 { u128::MAX } else { ones_u128(64) };
            self.v[rd] = !self.v[rn] & mask;
            return Step::Next;
        }

        // ---- REV64/REV32/REV16 (vector): reverse element order within a
        // 64-/32-/16-bit container ----
        // Same outer two-reg-misc class as NOT/MVN above and ABS/NEG/FABS
        // below but a disjoint opcode (0b00000/0b00001 vs their 0b00101/
        // 0b01011/0b01111/0b11111); still, per this file's ordering
        // discipline, checked before the looser ABS/NEG/FABS arm so its
        // catch-all can't swallow these first.
        if (instr >> 24) & 0x1f == 0b0_1110
            && (instr >> 17) & 0x1f == 0b1_0000
            && (instr >> 10) & 3 == 0b10
            && (instr >> 12) & 0x1f <= 1
        {
            let u = (instr >> 29) & 1;
            let size = (instr >> 22) & 3;
            let opcode = (instr >> 12) & 0x1f;
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let container = match (u, opcode) {
                (0, 0b00000) => 64u32, // REV64
                (1, 0b00000) => 32,    // REV32
                (0, 0b00001) => 16,    // REV16
                _ => return Step::Illegal,
            };
            let esize = 8u32 << size;
            if esize >= container {
                return Step::Illegal;
            }
            let q = (instr >> 30) & 1;
            let width = if q == 1 { 128 } else { 64 };
            let per = container / esize; // elements per container
            let mask = ones_u128(esize);
            let mut result = 0u128;
            let mut i = 0u32;
            while i < width / esize {
                let c = i / per; // which container
                let within = i % per;
                let src = c * per + (per - 1 - within);
                let lane = (self.v[rn] >> (src * esize)) & mask;
                result |= lane << (i * esize);
                i += 1;
            }
            self.v[rd] = result;
            return Step::Next;
        }

        // ---- XTN/XTN2, SQXTN/SQXTN2, UQXTN/UQXTN2, SQXTUN/SQXTUN2 (narrow) ----
        if (instr >> 24) & 0x1f == 0b0_1110
            && (instr >> 17) & 0x1f == 0b1_0000
            && (instr >> 10) & 3 == 0b10
            && matches!((instr >> 12) & 0x1f, 0b10010 | 0b10100)
        {
            let q = (instr >> 30) & 1;
            let u = (instr >> 29) & 1;
            let size = (instr >> 22) & 3;
            let opcode = (instr >> 12) & 0x1f;
            if size == 3 {
                return Step::Illegal;
            }
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let esize = 8u32 << size; // destination (narrow) element size
            let lanes = 64 / esize;
            let mask = ones_u128(esize);
            let src_mask = ones_u128(esize * 2);
            let mut narrow = 0u128;
            for i in 0..lanes {
                let wide = (self.v[rn] >> (i * esize * 2)) & src_mask;
                let lane = match (u, opcode) {
                    (0, 0b10010) => wide & mask, // XTN/XTN2: plain truncate
                    (0, 0b10100) => {
                        signed_sat(i128::from(sign_extend(wide as u64, esize * 2)), esize)
                    } // SQXTN/SQXTN2
                    (1, 0b10100) => unsigned_sat(wide as i128, esize), // UQXTN/UQXTN2
                    _ => unsigned_sat(i128::from(sign_extend(wide as u64, esize * 2)), esize), // SQXTUN/SQXTUN2 (1, 0b10010)
                };
                narrow |= lane << (i * esize);
            }
            self.v[rd] = if q == 1 {
                (self.v[rd] & ones_u128(64)) | (narrow << 64)
            } else {
                narrow
            };
            return Step::Next;
        }

        // ---- SUQADD/USQADD (vector): saturating accumulate of the other
        // signedness into the existing Vd ----
        if (instr >> 24) & 0x1f == 0b0_1110
            && (instr >> 17) & 0x1f == 0b1_0000
            && (instr >> 10) & 3 == 0b10
            && (instr >> 12) & 0x1f == 0b0_0011
        {
            let q = (instr >> 30) & 1;
            let u = (instr >> 29) & 1;
            let size = (instr >> 22) & 3;
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let esize = 8u32 << size;
            let width = if q == 1 { 128 } else { 64 };
            let lanes = width / esize;
            let mask = ones_u128(esize);
            let mut result = 0u128;
            for i in 0..lanes {
                let sh = i * esize;
                let vd = (self.v[rd] >> sh) & mask;
                let vn = (self.v[rn] >> sh) & mask;
                let lane = if u == 0 {
                    // SUQADD: signed accumulator + unsigned addend, signed-saturate.
                    let sum = i128::from(sign_extend(vd as u64, esize)) + vn as i128;
                    signed_sat(sum, esize)
                } else {
                    // USQADD: unsigned accumulator + signed addend, unsigned-saturate.
                    let sum = vd as i128 + i128::from(sign_extend(vn as u64, esize));
                    unsigned_sat(sum, esize)
                };
                result |= lane << sh;
            }
            self.v[rd] = result;
            return Step::Next;
        }

        // ---- SIMD two-register misc (scalar + vector): FCMEQ/FCMGT/FCMGE/
        // FCMLT/FCMLE against #0.0, FRECPE/FRSQRTE, URECPE/URSQRTE ----
        // Same "two-register miscellaneous" outer format as the FABS/FNEG/
        // FSQRT/ABS/NEG arm below, whose own opcode guard is broad enough
        // (any `(uns, hi, opcode)` not in its table falls straight to
        // `Illegal`) that it would swallow the vector (bit28 == 0) form of
        // these opcodes before they got a chance to run — so, per this
        // file's ordering discipline, this more-specific arm is checked
        // first. This is also the first arm in this file to decode the
        // *scalar* (bit28 == 1) two-register-misc encoding at all — nothing
        // below handles it, so there's no shadowing risk on that side.
        if ((instr >> 24) & 0x1f == 0b0_1110 || (instr >> 24) & 0x1f == 0b1_1110)
            && (instr >> 17) & 0x1f == 0b1_0000
            && (instr >> 10) & 3 == 0b10
            && matches!(
                (instr >> 12) & 0x1f,
                0b0_1101 | 0b0_1100 | 0b0_1110 | 0b1_1101 | 0b1_1100
            )
        {
            let scalar = (instr >> 24) & 0x1f == 0b1_1110;
            let q = (instr >> 30) & 1;
            let uns = (instr >> 29) & 1;
            let size = (instr >> 22) & 3;
            let opcode = (instr >> 12) & 0x1f;
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            if size >> 1 == 0 {
                return Step::Illegal; // reserved: needs S/D (float) or 32-bit (int)
            }
            let dbl = size & 1 == 1;

            if opcode == 0b1_1100 {
                // URECPE/URSQRTE: integer, vector-only, 32-bit lanes only.
                if scalar || dbl {
                    return Step::Illegal;
                }
                let lanes = if q == 1 { 4 } else { 2 };
                let mut result = 0u128;
                for i in 0..lanes {
                    let sh = i * 32;
                    let x = ((self.v[rn] >> sh) & 0xffff_ffff) as u32;
                    let r = if uns == 0 {
                        u32_recip_estimate(x)
                    } else {
                        u32_rsqrt_estimate(x)
                    };
                    result |= u128::from(r) << sh;
                }
                self.v[rd] = result;
                return Step::Next;
            }

            if dbl && !scalar && q == 0 {
                return Step::Illegal; // 1D is not a valid vector arrangement here
            }
            let lanes: u32 = if scalar {
                1
            } else if dbl {
                2
            } else if q == 1 {
                4
            } else {
                2
            };
            let esize = if dbl { 64u32 } else { 32 };
            let mask = ones_u128(esize);
            let mut result = self.v[rd];
            for i in 0..lanes {
                let sh = i * esize;
                // FRECPE/FRSQRTE "estimate": this interpreter has no hardware
                // pipeline to approximate, so it uses the exact reciprocal /
                // reciprocal square root here rather than the ARM ARM's
                // 8-bit lookup table. Real guest code that relies on the
                // *_ESTIMATE contract always follows up with a
                // Newton-Raphson refinement step (FRECPS/FRSQRTS) and
                // converges either way.
                let bits: u128 = if dbl {
                    let x = f64::from_bits(((self.v[rn] >> sh) & mask) as u64);
                    let r = match (opcode, uns) {
                        (0b0_1101, 0) => fbits_bool64(fp_eq(x, 0.0)), // FCMEQ #0.0
                        (0b0_1101, 1) => fbits_bool64(x <= 0.0),      // FCMLE #0.0
                        (0b0_1100, 0) => fbits_bool64(x > 0.0),       // FCMGT #0.0
                        (0b0_1100, 1) => fbits_bool64(x >= 0.0),      // FCMGE #0.0
                        (0b0_1110, 0) => fbits_bool64(x < 0.0),       // FCMLT #0.0
                        (0b1_1101, 0) => (1.0f64 / x).to_bits(),      // FRECPE
                        (0b1_1101, 1) => (1.0f64 / x.sqrt()).to_bits(), // FRSQRTE
                        _ => return Step::Illegal,
                    };
                    u128::from(r)
                } else {
                    let x = f32::from_bits(((self.v[rn] >> sh) & mask) as u32);
                    let r = match (opcode, uns) {
                        (0b0_1101, 0) => fbits_bool32(fp_eq(f64::from(x), 0.0)),
                        (0b0_1101, 1) => fbits_bool32(x <= 0.0),
                        (0b0_1100, 0) => fbits_bool32(x > 0.0),
                        (0b0_1100, 1) => fbits_bool32(x >= 0.0),
                        (0b0_1110, 0) => fbits_bool32(x < 0.0),
                        (0b1_1101, 0) => (1.0f32 / x).to_bits(),
                        (0b1_1101, 1) => (1.0f32 / x.sqrt()).to_bits(),
                        _ => return Step::Illegal,
                    };
                    u128::from(r)
                };
                result = (result & !(mask << sh)) | (bits << sh);
            }
            self.v[rd] = if scalar {
                result & mask
            } else if q == 1 {
                result
            } else {
                result & ones_u128(64)
            };
            return Step::Next;
        }

        // ---- SIMD two-reg-misc: FABS/FNEG/FSQRT and ABS/NEG (vector) ----
        // Same outer class as NOT/MVN above but a disjoint opcode field
        // (0b01111/0b11111 vs NOT's 0b00101), so ordering relative to it is
        // immaterial; still placed after it for readability.
        if (instr >> 24) & 0x1f == 0b0_1110
            && (instr >> 17) & 0x1f == 0b1_0000
            && (instr >> 10) & 3 == 0b10
        {
            let q = (instr >> 30) & 1;
            let uns = (instr >> 29) & 1;
            let size = (instr >> 22) & 3;
            let hi = size >> 1;
            let dbl = size & 1;
            let opcode = (instr >> 12) & 0x1f;
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            match (uns, hi, opcode) {
                (0 | 1, 1, 0b01111) | (1, 1, 0b11111) => {
                    // FABS / FNEG / FSQRT
                    if dbl == 1 && q == 0 {
                        return Step::Illegal;
                    }
                    self.v[rd] = if dbl == 0 {
                        let lanes = if q == 1 { 4 } else { 2 };
                        let mut result = 0u128;
                        for i in 0..lanes {
                            let sh = i * 32;
                            let val = f32::from_bits(((self.v[rn] >> sh) & 0xffff_ffff) as u32);
                            let res = match (uns, opcode) {
                                (0, 0b01111) => val.abs(),
                                (1, 0b01111) => -val,
                                _ => val.sqrt(),
                            };
                            result |= u128::from(res.to_bits()) << sh;
                        }
                        result
                    } else {
                        let mut result = 0u128;
                        for i in 0..2u32 {
                            let sh = i * 64;
                            let val =
                                f64::from_bits(((self.v[rn] >> sh) & u128::from(u64::MAX)) as u64);
                            let res = match (uns, opcode) {
                                (0, 0b01111) => val.abs(),
                                (1, 0b01111) => -val,
                                _ => val.sqrt(),
                            };
                            result |= u128::from(res.to_bits()) << sh;
                        }
                        result
                    };
                }
                (0 | 1, _, 0b01011) => {
                    // ABS / NEG (integer, any element size)
                    let esize = 8u32 << size;
                    let lanes = (if q == 1 { 128 } else { 64 }) / esize;
                    let mask = ones_u128(esize);
                    let mut result = 0u128;
                    for i in 0..lanes {
                        let sh = i * esize;
                        let val = (self.v[rn] >> sh) & mask;
                        let signed = sign_extend(val as u64, esize);
                        let res = if uns == 0 {
                            signed.wrapping_abs()
                        } else {
                            0i64.wrapping_sub(signed)
                        };
                        result |= (u128::from(res as u64) & mask) << sh;
                    }
                    self.v[rd] = result;
                }
                _ => return Step::Illegal,
            }
            return Step::Next;
        }

        // ---- SADDL/UADDL (vector long add: widen then add adjacent-size lanes) ----
        if (instr >> 24) & 0x1f == 0b0_1110
            && (instr >> 21) & 1 == 1
            && (instr >> 12) & 0xf == 0b0000
            && (instr >> 10) & 3 == 0b00
        {
            let q = (instr >> 30) & 1;
            let unsigned = (instr >> 29) & 1 == 1;
            let size = (instr >> 22) & 3;
            if size == 3 {
                return Step::Illegal; // no 64-bit source element
            }
            let rm = reg_field(instr, 16);
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let esize = 8u32 << size;
            let mask = ones_u128(esize);
            // Q selects which half (low 64 bits, or high 64 bits) of each
            // 128-bit source register supplies the 64 bits of narrow lanes.
            let src_shift = if q == 1 { 64 } else { 0 };
            let lanes = 64 / esize;
            let mut result = 0u128;
            for i in 0..lanes {
                let sh = i * esize;
                let xn = (self.v[rn] >> (src_shift + sh)) & mask;
                let xm = (self.v[rm] >> (src_shift + sh)) & mask;
                let (wn, wm) = if unsigned {
                    (xn, xm)
                } else {
                    (
                        (sign_extend(xn as u64, esize) as u128) & ones_u128(2 * esize),
                        (sign_extend(xm as u64, esize) as u128) & ones_u128(2 * esize),
                    )
                };
                let sum = (wn.wrapping_add(wm)) & ones_u128(2 * esize);
                result |= sum << (i * 2 * esize);
            }
            self.v[rd] = result;
            return Step::Next;
        }

        // ---- UADDW/UADDW2/SADDW/SADDW2 (vector wide add: Vn is already
        // wide, Vm is narrow and gets widened first) ----
        if (instr >> 24) & 0x1f == 0b0_1110
            && (instr >> 21) & 1 == 1
            && (instr >> 12) & 0xf == 0b0001
            && (instr >> 10) & 3 == 0b00
        {
            let q = (instr >> 30) & 1;
            let unsigned = (instr >> 29) & 1 == 1;
            let size = (instr >> 22) & 3;
            if size == 3 {
                return Step::Illegal;
            }
            let rm = reg_field(instr, 16);
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let esize = 8u32 << size;
            let mask = ones_u128(esize);
            // Q selects which half of the narrow Vm supplies the addend
            // (UADDW/SADDW use the low half, the "2" forms the high half).
            let src_shift = if q == 1 { 64 } else { 0 };
            let lanes = 64 / esize;
            let wide_mask = ones_u128(2 * esize);
            let mut result = 0u128;
            for i in 0..lanes {
                let sh = i * 2 * esize;
                let xn = (self.v[rn] >> sh) & wide_mask;
                let xm = (self.v[rm] >> (src_shift + i * esize)) & mask;
                let wm = if unsigned {
                    xm
                } else {
                    (sign_extend(xm as u64, esize) as u128) & wide_mask
                };
                result |= (xn.wrapping_add(wm) & wide_mask) << sh;
            }
            self.v[rd] = result;
            return Step::Next;
        }

        // ---- ADDHN/ADDHN2 (vector add, high narrow) ----
        if (instr >> 24) & 0x1f == 0b0_1110
            && (instr >> 29) & 1 == 0
            && (instr >> 21) & 1 == 1
            && (instr >> 12) & 0xf == 0b0100
            && (instr >> 10) & 3 == 0b00
        {
            let q = (instr >> 30) & 1;
            let size = (instr >> 22) & 3;
            if size == 3 {
                return Step::Illegal;
            }
            let rm = reg_field(instr, 16);
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let esize = 8u32 << size; // destination (narrow) element size
            let src_mask = ones_u128(esize * 2);
            let lanes = 64 / esize;
            let mut narrow = 0u128;
            for i in 0..lanes {
                let sh = i * esize * 2;
                let xn = (self.v[rn] >> sh) & src_mask;
                let xm = (self.v[rm] >> sh) & src_mask;
                let sum = xn.wrapping_add(xm) & src_mask;
                narrow |= (sum >> esize) << (i * esize);
            }
            self.v[rd] = if q == 1 {
                (self.v[rd] & ones_u128(64)) | (narrow << 64)
            } else {
                narrow
            };
            return Step::Next;
        }

        // ---- FMAXNMV/FMINNMV/FMAXV/FMINV (across-lanes float reduction, .4S only) ----
        // Shares its outer bits with the ADDV/UADDLV/S|UMAXV/S|UMINV arm
        // below, whose own opcode guard is broad enough (any `(u, opcode)`
        // not in its table falls straight to `Illegal`) that it would
        // swallow these two opcodes (0b01100/0b01111) before they got a
        // chance to run — so, per this file's ordering discipline, this
        // more-specific arm is checked first.
        if (instr >> 24) & 0x1f == 0b0_1110
            && (instr >> 17) & 0x1f == 0b1_1000
            && (instr >> 10) & 3 == 0b10
            && matches!((instr >> 12) & 0x1f, 0b0_1100 | 0b0_1111)
        {
            let q = (instr >> 30) & 1;
            let is_min = (instr >> 22) & 1 == 1;
            let is_nm = (instr >> 12) & 0x1f == 0b0_1100;
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            if q == 0 {
                return Step::Illegal; // only the 4S arrangement is defined
            }
            let lane = |i: u32| f32::from_bits(((self.v[rn] >> (i * 32)) & 0xffff_ffff) as u32);
            let mut acc = lane(0);
            for i in 1..4 {
                let v = lane(i);
                acc = match (is_nm, is_min) {
                    (true, true) => acc.min(v),       // FMINNMV
                    (true, false) => acc.max(v),      // FMAXNMV
                    (false, true) => fmin32(acc, v),  // FMINV
                    (false, false) => fmax32(acc, v), // FMAXV
                };
            }
            self.set_fp32(rd, acc);
            return Step::Next;
        }

        // ---- ADDV / UADDLV (across-lanes reduction) ----
        if (instr >> 24) & 0x1f == 0b0_1110
            && (instr >> 17) & 0x1f == 0b1_1000
            && (instr >> 10) & 3 == 0b10
        {
            let q = (instr >> 30) & 1;
            let u = (instr >> 29) & 1;
            let size = (instr >> 22) & 3;
            let opcode = (instr >> 12) & 0x1f;
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            if size == 3 {
                return Step::Illegal;
            }
            let esize = 8u32 << size;
            let width = if q == 1 { 128 } else { 64 };
            let lanes = width / esize;
            let mask = ones_u128(esize);
            match (u, opcode) {
                (0, 0b1_1011) => {
                    // ADDV: sum all lanes, esize-bit result.
                    let mut sum = 0u128;
                    for i in 0..lanes {
                        sum = sum.wrapping_add((self.v[rn] >> (i * esize)) & mask);
                    }
                    self.v[rd] = sum & mask;
                }
                (1, 0b0_0011) => {
                    // UADDLV: sum all lanes zero-extended, 2*esize-bit result.
                    let mut sum = 0u128;
                    for i in 0..lanes {
                        sum = sum.wrapping_add((self.v[rn] >> (i * esize)) & mask);
                    }
                    self.v[rd] = sum & ones_u128(esize * 2);
                }
                (0 | 1, 0b0_1010 | 0b1_1010) => {
                    // SMAXV/UMAXV/SMINV/UMINV: extremum across all lanes.
                    let is_min = opcode == 0b1_1010;
                    let lane_at = |i: u32| (self.v[rn] >> (i * esize)) & mask;
                    let mut acc = lane_at(0);
                    for i in 1..lanes {
                        let v = lane_at(i);
                        let better = if u == 0 {
                            let (sv, sacc) =
                                (sign_extend(v as u64, esize), sign_extend(acc as u64, esize));
                            if is_min { sv < sacc } else { sv > sacc }
                        } else if is_min {
                            v < acc
                        } else {
                            v > acc
                        };
                        if better {
                            acc = v;
                        }
                    }
                    self.v[rd] = acc;
                }
                _ => return Step::Illegal,
            }
            return Step::Next;
        }

        // ---- ADDP (scalar, D-form) / FADDP (scalar, S/D-form): pairwise-add
        // the two lanes of a single source register into a scalar result ----
        if (instr >> 24) & 0x1f == 0b1_1110
            && (instr >> 30) & 1 == 1
            && (instr >> 17) & 0x1f == 0b1_1000
            && (instr >> 10) & 3 == 0b10
        {
            let u = (instr >> 29) & 1;
            let sz = (instr >> 22) & 1;
            let opcode = (instr >> 12) & 0x1f;
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            match (u, opcode) {
                (0, 0b1_1011) => {
                    // ADDP d, Vn.2D
                    let lo = self.v[rn] as u64;
                    let hi = (self.v[rn] >> 64) as u64;
                    self.v[rd] = u128::from(lo.wrapping_add(hi));
                }
                (1, 0b0_1101) => {
                    // FADDP (scalar): S (sz=0) or D (sz=1).
                    if sz == 0 {
                        let a = f32::from_bits(self.v[rn] as u32);
                        let b = f32::from_bits((self.v[rn] >> 32) as u32);
                        self.set_fp32(rd, a + b);
                    } else {
                        let a = f64::from_bits(self.v[rn] as u64);
                        let b = f64::from_bits((self.v[rn] >> 64) as u64);
                        self.set_fp64(rd, a + b);
                    }
                }
                _ => return Step::Illegal,
            }
            return Step::Next;
        }

        // ---- SSHLL/USHLL (vector shift-left-long, widening) ----
        if (instr >> 23) & 0x3f == 0b0_11110 && (instr >> 10) & 0x3f == 0b10_1001 {
            let q = (instr >> 30) & 1;
            let unsigned = (instr >> 29) & 1 == 1;
            let immh = (instr >> 19) & 0xf;
            let immb = (instr >> 16) & 7;
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let esize = if immh & 0b1000 != 0 {
                return Step::Illegal;
            } else if immh & 0b0100 != 0 {
                32
            } else if immh & 0b0010 != 0 {
                16
            } else if immh & 0b0001 != 0 {
                8
            } else {
                return Step::Illegal;
            };
            let shift = ((immh << 3) | immb) - esize;
            let src = if q == 0 {
                self.v[rn] as u64
            } else {
                (self.v[rn] >> 64) as u64
            };
            let lanes = 64 / esize;
            let mut result = 0u128;
            for i in 0..lanes {
                let e = (src >> (i * esize)) & ones(esize);
                let ext = if unsigned {
                    u128::from(e)
                } else {
                    (sign_extend(e, esize) as u128) & ones_u128(2 * esize)
                };
                let widened = (ext << shift) & ones_u128(2 * esize);
                result |= widened << (i * 2 * esize);
            }
            self.v[rd] = result;
            return Step::Next;
        }

        // ---- SIMD shift by immediate: SHL / SSHR / USHR (scalar + vector) ----
        if ((instr >> 23) & 0x3f == 0b0_11110 || (instr >> 23) & 0x3f == 0b1_11110)
            && (instr >> 10) & 1 == 1
        {
            let scalar = (instr >> 23) & 0x3f == 0b1_11110;
            let q = (instr >> 30) & 1;
            let u = (instr >> 29) & 1;
            let immh = (instr >> 19) & 0xf;
            let immb = (instr >> 16) & 7;
            let opcode = (instr >> 11) & 0x1f;
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            let esize: u32 = if immh & 0b1000 != 0 {
                64
            } else if immh & 0b0100 != 0 {
                32
            } else if immh & 0b0010 != 0 {
                16
            } else if immh & 0b0001 != 0 {
                8
            } else {
                return Step::Illegal;
            };
            let immhb = (immh << 3) | immb;
            let lanes_bits = if scalar {
                esize
            } else if q == 1 {
                128
            } else {
                64
            };
            let mut result = 0u128;
            let mut i = 0;
            while i < lanes_bits / esize {
                let e = ((self.v[rn] >> (i * esize)) & ones_u128(esize)) as u64;
                let lane = match opcode {
                    0b01010 => e << (immhb - esize), // SHL
                    0b00000 => {
                        // SSHR (u==0) / USHR (u==1)
                        let sh = (2 * esize - immhb).min(63);
                        if u == 1 {
                            e >> sh
                        } else {
                            (sign_extend(e, esize) >> sh) as u64
                        }
                    }
                    _ => return Step::Illegal,
                };
                result |= u128::from(lane & ones(esize)) << (i * esize);
                i += 1;
            }
            self.v[rd] = result;
            return Step::Next;
        }

        // ---- Advanced SIMD (scalar and vector) x indexed element: FMUL/
        // FMLA/FMLS/FMULX, MUL/MLA/MLS, SMULL/UMULL/SQDMULL (by element) ----
        // A previously-unhandled top-level class: bits[28:24] == 0b01111
        // (vector, Q at bit30) or 0b11111 (scalar, bit30 fixed 1) with
        // bit10 == 0 — disjoint from every "three-same" class above (which
        // fixes the same bit range to `...1110` and always has bit10 == 1),
        // so this can't shadow, or be shadowed by, anything already
        // handled; placement relative to other arms is immaterial.
        if ((instr >> 24) & 0x1f == 0b0_1111 || (instr >> 24) & 0x1f == 0b1_1111)
            && (instr >> 10) & 1 == 0
        {
            let scalar = (instr >> 24) & 0x1f == 0b1_1111;
            let q = (instr >> 30) & 1;
            let u = (instr >> 29) & 1;
            let size = (instr >> 22) & 3;
            let l = (instr >> 21) & 1;
            let m = (instr >> 20) & 1;
            let rm_lo4 = (instr >> 16) & 0xf; // Rm[3:0]; M (above) supplies bit 4 for S/D sizes
            let opcode = (instr >> 12) & 0xf;
            let h = (instr >> 11) & 1;
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);

            // The indexed element's register and lane index; layout depends
            // on the source element size (ARM `Rm`/`index` tables for this
            // encoding — H restricts Vm to V0-V15 and folds M into the
            // index, S/D use the full V0-V31 range via M:Rm).
            let (esize, rm, index): (u32, usize, u32) = match size {
                0b01 => (16, rm_lo4 as usize, (h << 2) | (l << 1) | m),
                0b10 => (32, (((m << 4) | rm_lo4) & 0x1f) as usize, (h << 1) | l),
                0b11 => (64, (((m << 4) | rm_lo4) & 0x1f) as usize, h),
                _ => return Step::Illegal,
            };
            let mask = ones_u128(esize);
            let m_elem = (self.v[rm] >> (index * esize)) & mask;

            if matches!(opcode, 0b0001 | 0b0101 | 0b1001) {
                // FMLA / FMLS / FMUL / FMULX (float; S or D elements only).
                if esize != 32 && esize != 64 {
                    return Step::Illegal;
                }
                if esize == 64 && !scalar && q == 0 {
                    return Step::Illegal; // 1D is not a valid vector arrangement
                }
                let lanes: u32 = if scalar {
                    1
                } else if esize == 32 {
                    if q == 1 { 4 } else { 2 }
                } else {
                    2
                };
                let mut result = self.v[rd];
                for i in 0..lanes {
                    let sh = i * esize;
                    let bits: u128 = if esize == 32 {
                        let mv = f32::from_bits(m_elem as u32);
                        let nv = f32::from_bits(((self.v[rn] >> sh) & mask) as u32);
                        let acc = f32::from_bits(((self.v[rd] >> sh) & mask) as u32);
                        let r = match (opcode, u) {
                            (0b0001, 0) => nv.mul_add(mv, acc),    // FMLA
                            (0b0101, 0) => (-nv).mul_add(mv, acc), // FMLS
                            (0b1001, 0) => nv * mv,                // FMUL
                            (0b1001, 1) => fmulx32(nv, mv),        // FMULX
                            _ => return Step::Illegal,
                        };
                        u128::from(r.to_bits())
                    } else {
                        let mv = f64::from_bits(m_elem as u64);
                        let nv = f64::from_bits(((self.v[rn] >> sh) & mask) as u64);
                        let acc = f64::from_bits(((self.v[rd] >> sh) & mask) as u64);
                        let r = match (opcode, u) {
                            (0b0001, 0) => nv.mul_add(mv, acc),
                            (0b0101, 0) => (-nv).mul_add(mv, acc),
                            (0b1001, 0) => nv * mv,
                            (0b1001, 1) => fmulx64(nv, mv),
                            _ => return Step::Illegal,
                        };
                        u128::from(r.to_bits())
                    };
                    result = (result & !(mask << sh)) | (bits << sh);
                }
                self.v[rd] = if scalar {
                    result & mask
                } else if esize == 32 && q == 0 {
                    result & ones_u128(64)
                } else {
                    result
                };
                return Step::Next;
            }

            // Integer by-element: MUL/MLA/MLS (non-widening, H/S) and
            // SMULL/UMULL/SQDMULL (widening H->S or S->D). Vector-only —
            // none of these have a defined scalar-register form.
            if scalar || esize == 64 {
                return Step::Illegal;
            }
            match (opcode, u) {
                (0b1000, 0) | (0b0000 | 0b0100, 1) => {
                    // MUL / MLA / MLS (non-widening).
                    let lanes = (if q == 1 { 128 } else { 64 }) / esize;
                    let mut result = 0u128;
                    for i in 0..lanes {
                        let sh = i * esize;
                        let n_elem = (self.v[rn] >> sh) & mask;
                        let prod = n_elem.wrapping_mul(m_elem) & mask;
                        let lane = match (opcode, u) {
                            (0b1000, 0) => prod,                                                  // MUL
                            (0b0000, 1) => ((self.v[rd] >> sh) & mask).wrapping_add(prod) & mask, // MLA
                            _ => ((self.v[rd] >> sh) & mask).wrapping_sub(prod) & mask, // MLS
                        };
                        result |= lane << sh;
                    }
                    self.v[rd] = result;
                }
                (0b1010, _) | (0b1011, 0) => {
                    // SMULL / UMULL / SQDMULL (widening H->S or S->D). Q
                    // selects which half of the narrow source registers
                    // supplies the lanes, same convention as SADDL/UADDL.
                    let src_shift = if q == 1 { 64 } else { 0 };
                    let lanes = 64 / esize;
                    let wide_mask = ones_u128(esize * 2);
                    let mut result = 0u128;
                    for i in 0..lanes {
                        let n_elem = (self.v[rn] >> (src_shift + i * esize)) & mask;
                        let lane = if opcode == 0b1011 {
                            // SQDMULL: 2 * n * m, signed-saturated to 2*esize bits.
                            signed_sat(
                                2 * i128::from(sign_extend(n_elem as u64, esize))
                                    * i128::from(sign_extend(m_elem as u64, esize)),
                                esize * 2,
                            )
                        } else if u == 0 {
                            // SMULL
                            ((i128::from(sign_extend(n_elem as u64, esize))
                                * i128::from(sign_extend(m_elem as u64, esize)))
                                as u128)
                                & wide_mask
                        } else {
                            // UMULL
                            n_elem.wrapping_mul(m_elem) & wide_mask
                        };
                        result |= lane << (i * esize * 2);
                    }
                    self.v[rd] = result;
                }
                _ => return Step::Illegal,
            }
            return Step::Next;
        }

        // ---- TBZ / TBNZ (test bit and branch) ----
        if (instr >> 25) & 0x3f == 0b01_1011 {
            let op = (instr >> 24) & 1;
            let bitpos = (((instr >> 31) & 1) << 5) | ((instr >> 19) & 0x1f);
            let rt = reg_field(instr, 0);
            let bit = (self.read_x(rt) >> bitpos) & 1;
            let take = if op == 0 { bit == 0 } else { bit == 1 };
            if take {
                let off = sign_extend(u64::from((instr >> 5) & 0x3fff), 14) << 2;
                return self.branch(off);
            }
            return Step::Next;
        }

        Step::Illegal
    }
}

impl Vcpu for Aarch64Interp {
    fn run(&mut self, mem: &mut GuestMemory) -> Result<Exit, VcpuError> {
        for _ in 0..MAX_STEPS {
            let Ok(instr) = mem.read_u32(self.pc) else {
                return Ok(Exit::MemFault {
                    addr: self.pc,
                    write: false,
                });
            };
            match self.exec(instr, mem) {
                Step::Next => self.pc = self.pc.wrapping_add(4),
                Step::Branched => {}
                Step::Syscall => return Ok(Exit::Syscall),
                Step::Illegal => return Ok(Exit::IllegalInstruction { pc: self.pc }),
                Step::Fault { addr, write } => return Ok(Exit::MemFault { addr, write }),
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
        self.excl_monitor = true;
    }
}

/// Encoded `op0<0>:op1:CRn:CRm:op2` field (bits 19:5) used by `MRS`/`MSR` to
/// name a system register — see `Aarch64Interp::read_sysreg`/`write_sysreg`.
/// Every value below was cross-checked by assembling the named register with
/// `clang -target aarch64-linux-gnu` and disassembling with `llvm-objdump`
/// (e.g. `mrs x0, tpidr_el0` -> `0xd53bd040`, giving `(0xd53bd040 >> 5) &
/// 0x7fff == 0x5e82`).
const TPIDR_EL0: u32 = 0x5E82;
const TPIDRRO_EL0: u32 = 0x5E83;
const FPCR: u32 = 0x5A20;
const FPSR: u32 = 0x5A21;
const MIDR_EL1: u32 = 0x4000;
const MPIDR_EL1: u32 = 0x4005;
const REVIDR_EL1: u32 = 0x4006;
const CTR_EL0: u32 = 0x5801;
const DCZID_EL0: u32 = 0x5807;
const CNTFRQ_EL0: u32 = 0x5F00;
const CNTVCT_EL0: u32 = 0x5F02;
const CNTVCTSS_EL0: u32 = 0x5F06;
const ID_AA64ISAR0_EL1: u32 = 0x4030;
const ID_AA64PFR0_EL1: u32 = 0x4020;

/// Plausible `MIDR_EL1` value (ARM-implementer, architecture=0xf meaning "see
/// `ID_AA64*`", an arbitrary but real-looking Cortex-ish part/revision).
/// Nothing in this interpreter's target userland (musl, openssl) branches on
/// the exact value, just that reading it doesn't trap.
const MIDR_EL1_VAL: u64 = 0x410F_D0C0;
/// `CTR_EL0`: 64-byte I-cache and D-cache lines, unified L1, `IDC`/`DIC` set
/// (matches `DCZID_EL0_VAL`'s 64-byte `DC ZVA` block below).
const CTR_EL0_VAL: u64 = 0x8444_c004;
/// `DCZID_EL0`: `BS` (bits 3:0) = 4 -> `DC ZVA` zeroes `4 << 4` =
/// [`DC_ZVA_BLOCK_BYTES`] bytes; `DZP` (bit 4) = 0, i.e. `DC ZVA` is not
/// disabled.
const DCZID_EL0_VAL: u64 = 0x4;
/// `DC ZVA`'s zeroed block size in bytes. Must stay in sync with
/// `DCZID_EL0_VAL`'s `BS` field above (`4 << 4 == 64`).
const DC_ZVA_BLOCK_BYTES: u64 = 64;
const CNTFRQ_EL0_VAL: u64 = 1_000_000_000;
/// `ID_AA64ISAR0_EL1`: advertise exactly the optional features this
/// interpreter implements — `AES` (=1: AES, no `PMULL`), `SHA1` (=1),
/// `SHA2` (=1: SHA-256 only, no SHA-512), `CRC32` (=1), `Atomic` (=2: full
/// LSE including `CASP`) — all other fields (`RDM`, `SM3/4`, `DP`, …) stay 0.
const ID_AA64ISAR0_EL1_VAL: u64 = 0x0021_1110;
/// `ID_AA64PFR0_EL1`: `EL0`/`EL1` = 1 (AArch64-only, no AArch32), `FP`/
/// `AdvSIMD` = 0 (implemented, no FP16), everything else (`EL2`, `EL3`,
/// `GIC`, `RAS`, `SVE`, …) unimplemented (0xf or 0, per field).
const ID_AA64PFR0_EL1_VAL: u64 = 0x0000_0011;

/// Encoded `op1:CRn:CRm:op2` field (bits 18:5) for the cache-maintenance
/// `SYS` instructions below — same objdump-verification method as the MRS/
/// MSR codes above (e.g. `dc zva, x0` -> `0xd50b7420`, `(0xd50b7420 >> 5) &
/// 0x3fff == 0x1ba1`).
const SYS_DC_ZVA: u32 = 0x1BA1;
const SYS_DC_CVAC: u32 = 0x1BD1;
const SYS_DC_CVAU: u32 = 0x1BD9;
const SYS_DC_CIVAC: u32 = 0x1BF1;
const SYS_DC_IVAC: u32 = 0x03B1;
const SYS_IC_IALLU: u32 = 0x03A8;
const SYS_IC_IALLUIS: u32 = 0x0388;
const SYS_IC_IVAU: u32 = 0x1BA9;

/// Extract a 5-bit register field starting at bit `lsb`.
fn reg_field(instr: u32, lsb: u32) -> usize {
    ((instr >> lsb) & 0x1f) as usize
}

/// Mask to 32 bits when `sf == 0` (32-bit operation).
const fn mask_sf(v: u64, sf: u32) -> u64 {
    if sf == 0 { v & 0xffff_ffff } else { v }
}

/// Apply an aarch64 register shift (LSL/LSR/ASR/ROR) by `amount`.
fn shift_reg(v: u64, shift_type: u32, amount: u32, sf: bool) -> u64 {
    let width = if sf { 64 } else { 32 };
    let amt = amount % width;
    let v = if sf { v } else { v & 0xffff_ffff };
    let r = match shift_type {
        0 => v << amt,
        1 => v >> amt,
        2 => {
            if sf {
                ((v as i64) >> amt) as u64
            } else {
                u64::from(((v as u32 as i32) >> amt) as u32)
            }
        }
        _ => {
            if sf {
                v.rotate_right(amt)
            } else {
                u64::from((v as u32).rotate_right(amt))
            }
        }
    };
    if sf { r } else { r & 0xffff_ffff }
}

/// Reverse the low `width` bits of `v`.
fn rbit(v: u64, width: u32) -> u64 {
    if width == 64 {
        v.reverse_bits()
    } else {
        u64::from((v as u32).reverse_bits())
    }
}

/// Reverse the byte order within each 16-bit halfword across `width` bits.
fn rev16(v: u64, width: u32) -> u64 {
    let mut r = 0u64;
    let mut i = 0;
    while i < width {
        let h = ((v >> i) & 0xffff) as u16;
        r |= u64::from(h.swap_bytes()) << i;
        i += 16;
    }
    r
}

/// Reverse the byte order within each 32-bit word (64-bit REV32).
fn rev32(v: u64) -> u64 {
    u64::from((v as u32).swap_bytes()) | (u64::from(((v >> 32) as u32).swap_bytes()) << 32)
}

/// Count leading sign bits minus one (CLS) over the low `width` bits.
fn cls(v: u64, width: u32) -> u32 {
    let sign = (v >> (width - 1)) & 1;
    let mut count = 0;
    let mut i = width - 1;
    while i > 0 {
        i -= 1;
        if (v >> i) & 1 == sign {
            count += 1;
        } else {
            break;
        }
    }
    count
}

/// Unsigned divide with aarch64 semantics (division by zero yields 0).
fn udiv(a: u64, b: u64, sf: bool) -> u64 {
    if sf {
        a.checked_div(b).unwrap_or(0)
    } else {
        (a as u32).checked_div(b as u32).map_or(0, u64::from)
    }
}

/// Signed divide with aarch64 semantics (division by zero yields 0;
/// `INT_MIN / -1` wraps to `INT_MIN`).
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

/// Extend a register value per the `option` field (UXTB/H/W/X, SXTB/H/W/X),
/// then shift left by `shift`.
fn extend_reg(val: u64, option: u32, shift: u32) -> u64 {
    let extended = match option {
        0b000 => val & 0xff,                                // UXTB
        0b001 => val & 0xffff,                              // UXTH
        0b010 => val & 0xffff_ffff,                         // UXTW
        0b100 => sign_extend(val & 0xff, 8) as u64,         // SXTB
        0b101 => sign_extend(val & 0xffff, 16) as u64,      // SXTH
        0b110 => sign_extend(val & 0xffff_ffff, 32) as u64, // SXTW
        _ => val,                                           // UXTX / SXTX (LSL)
    };
    extended << shift
}

/// `n` low bits set.
fn ones(n: u32) -> u64 {
    if n >= 64 { u64::MAX } else { (1u64 << n) - 1 }
}

/// Read `nbytes` (1, 2, 4, or 8) little-endian bytes from `addr`,
/// zero-extended to 64 bits. `None` on a memory fault. A plain integer read
/// (no register write-back), unlike `Aarch64Interp::ldst` — shared by the LSE
/// atomic (CAS/CASP/SWP/LD<op>) helpers, which all need the raw old value
/// before deciding what (if anything) to write back.
fn mem_read_sized(mem: &mut GuestMemory, addr: u64, nbytes: usize) -> Option<u64> {
    let mut buf = [0u8; 8];
    mem.read(addr, &mut buf[..nbytes]).ok()?;
    Some(u64::from_le_bytes(buf))
}

/// ARM `AdvSIMDExpandImm`: build the 64-bit lane value for a SIMD modified
/// immediate from `cmode`/`op`/`imm8`.
fn adv_simd_expand_imm(cmode: u32, op: u32, imm8: u64) -> u64 {
    let rep32 = |x: u64| (x & 0xffff_ffff) | ((x & 0xffff_ffff) << 32);
    let rep16 = |x: u64| {
        let x = x & 0xffff;
        x | (x << 16) | (x << 32) | (x << 48)
    };
    match cmode >> 1 {
        0b000 => rep32(imm8),
        0b001 => rep32(imm8 << 8),
        0b010 => rep32(imm8 << 16),
        0b011 => rep32(imm8 << 24),
        0b100 => rep16(imm8),
        0b101 => rep16(imm8 << 8),
        0b110 => {
            if cmode & 1 == 0 {
                rep32((imm8 << 8) | 0xff)
            } else {
                rep32((imm8 << 16) | 0xffff)
            }
        }
        _ => {
            if cmode == 0b1110 {
                if op == 0 {
                    (imm8 & 0xff) * 0x0101_0101_0101_0101 // byte replicate
                } else {
                    // bytemask: each bit of imm8 expands to a full byte.
                    let mut r = 0u64;
                    for i in 0..8 {
                        if (imm8 >> i) & 1 == 1 {
                            r |= 0xffu64 << (i * 8);
                        }
                    }
                    r
                }
            } else if op == 0 {
                rep32(u64::from(vfp_expand_imm32(imm8 as u32)))
            } else {
                vfp_expand_imm64(imm8 as u32)
            }
        }
    }
}

/// VFP 32-bit immediate expansion (`FMOV`/`MOVI` fp forms).
fn vfp_expand_imm32(imm8: u32) -> u32 {
    let sign = (imm8 >> 7) & 1;
    let b6 = (imm8 >> 6) & 1;
    let exp = ((1 - b6) << 7) | (if b6 == 1 { 0x1f } else { 0 } << 2) | ((imm8 >> 4) & 3);
    let frac = (imm8 & 0xf) << 19;
    (sign << 31) | (exp << 23) | frac
}

/// VFP 64-bit immediate expansion.
fn vfp_expand_imm64(imm8: u32) -> u64 {
    let sign = u64::from((imm8 >> 7) & 1);
    let b6 = u64::from((imm8 >> 6) & 1);
    let exp = ((1 - b6) << 10) | (if b6 == 1 { 0xff } else { 0 } << 2) | u64::from((imm8 >> 4) & 3);
    let frac = u64::from(imm8 & 0xf) << 48;
    (sign << 63) | (exp << 52) | frac
}

/// ARM `SSHL`/`USHL` per-lane shift: `x` is an `esize`-bit lane, `amt` is a
/// signed shift amount (positive = left, negative = right, saturating at
/// `esize` in either direction). `signed` selects an arithmetic (vs logical)
/// right shift.
fn simd_shl(x: u128, amt: i64, esize: u32, signed: bool) -> u128 {
    let mask = ones_u128(esize);
    if amt >= 0 {
        let sh = amt.min(i64::from(esize)) as u32;
        if sh >= esize { 0 } else { (x << sh) & mask }
    } else {
        let sh = (-amt).min(i64::from(esize)) as u32;
        if signed {
            let se = sign_extend(x as u64, esize);
            let shifted = if sh >= 64 {
                if se < 0 { -1i64 } else { 0 }
            } else {
                se >> sh.min(63)
            };
            u128::from(shifted as u64) & mask
        } else if sh >= esize {
            0
        } else {
            (x >> sh) & mask
        }
    }
}

/// Clamp `v` to the signed range representable in `esize` bits (1..=64),
/// returning the clamped value as its `esize`-bit two's-complement bit
/// pattern. Shared by the saturating NEON integer arithmetic (SQADD/SQSUB/
/// SQSHL/SQRSHL/SUQADD) and the SQXTN/SQXTUN narrowing family.
fn signed_sat(v: i128, esize: u32) -> u128 {
    let max = (1i128 << (esize - 1)) - 1;
    let min = -(1i128 << (esize - 1));
    (v.clamp(min, max) as u128) & ones_u128(esize)
}

/// Clamp `v` to the unsigned range representable in `esize` bits (1..=64),
/// i.e. `[0, 2^esize - 1]` (negative inputs saturate to 0). Shared by
/// UQADD/UQSUB/UQSHL/UQRSHL/USQADD and the UQXTN/SQXTUN narrowing family.
fn unsigned_sat(v: i128, esize: u32) -> u128 {
    let max = (1i128 << esize) - 1;
    v.clamp(0, max) as u128
}

/// ARM `SQSHL`/`UQSHL`/`SQRSHL`/`UQRSHL` per-lane saturating shift: `x` is an
/// `esize`-bit lane, `amt` is a signed shift amount (positive = left,
/// negative = right, matching the `SSHL`/`USHL` convention above). Left
/// shifts saturate at the `esize`-bit signed/unsigned bound; right shifts
/// never saturate but, when `rounding`, add a half-ULP before shifting
/// (verified against native `sqrshl`/`uqrshl` execution on aarch64 hardware).
fn sat_shl(x: u128, amt: i64, esize: u32, signed: bool, rounding: bool) -> u128 {
    if amt >= 0 {
        let sh = amt.min(i64::from(esize)) as u32;
        if signed {
            let se = i128::from(sign_extend(x as u64, esize));
            signed_sat(se << sh, esize)
        } else {
            // `x < 2^esize` and `sh <= esize <= 64`, so `x << sh` fits in a
            // `u128` (< 2^128) without overflow — unlike the signed path,
            // this deliberately stays in `u128` rather than widening to
            // `i128`, which (being one bit narrower) could wrap here.
            let shifted = x << sh;
            let max = ones_u128(esize);
            if shifted > max { max } else { shifted }
        }
    } else {
        let sh = (-amt) as u32; // 1..=128
        if signed {
            let se = i128::from(sign_extend(x as u64, esize));
            let r = if sh >= 127 {
                if se < 0 { -1 } else { 0 }
            } else if rounding {
                (se + (1i128 << (sh - 1))) >> sh
            } else {
                se >> sh
            };
            (r as u128) & ones_u128(esize)
        } else {
            let r = if sh >= 128 {
                0
            } else if rounding {
                (x + (1u128 << (sh - 1))) >> sh
            } else {
                x >> sh
            };
            r & ones_u128(esize)
        }
    }
}

/// One byte of the reflected CRC32/CRC32C update used by the `CRC32*`
/// instructions: XOR the byte in, then run 8 rounds of the reflected LFSR
/// with `poly` (`0xEDB8_8320` for CRC32, `0x82F6_3B78` for CRC32C — the
/// bit-reflected forms of the architectural 0x04C11DB7/0x1EDC6F41
/// polynomials). Verified against native `crc32b`/`crc32cb` execution and the
/// standard `"123456789"` CRC-32/CRC-32C check vectors.
fn crc32_step(crc: u32, byte: u8, poly: u32) -> u32 {
    let mut crc = crc ^ u32::from(byte);
    for _ in 0..8 {
        crc = if crc & 1 != 0 {
            (crc >> 1) ^ poly
        } else {
            crc >> 1
        };
    }
    crc
}

/// `n` low bits set, as a `u128`.
fn ones_u128(n: u32) -> u128 {
    if n >= 128 {
        u128::MAX
    } else {
        (1u128 << n) - 1
    }
}

/// SIMD element size in bits from a `Q`/`DUP`/`UMOV` `imm5` field.
fn elem_bits(imm5: u32) -> u32 {
    if imm5 & 1 != 0 {
        8
    } else if imm5 & 2 != 0 {
        16
    } else if imm5 & 4 != 0 {
        32
    } else {
        64
    }
}

/// Rotate the low `size` bits of `v` right by `r`.
fn ror_val(v: u64, r: u32, size: u32) -> u64 {
    if size == 0 {
        return v;
    }
    let r = r % size;
    let v = v & ones(size);
    if r == 0 {
        return v;
    }
    ((v >> r) | (v << (size - r))) & ones(size)
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
    result & ones(width)
}

/// ARM `DecodeBitMasks`: turn `(N, imms, immr)` into the `(wmask, tmask)` pair
/// used by the logical-immediate and bitfield instructions. Returns `None` for
/// reserved encodings.
fn decode_bit_masks(n: u32, imms: u32, immr: u32, width: u32) -> Option<(u64, u64)> {
    let x = (n << 6) | ((!imms) & 0x3f);
    if x == 0 {
        return None;
    }
    let len = x.ilog2();
    if len < 1 {
        return None;
    }
    let levels = (1u32 << len) - 1;
    let s = imms & levels;
    let r = immr & levels;
    let diff = s.wrapping_sub(r) & levels;
    let esize = 1u32 << len;
    let wmask = replicate(ror_val(ones(s + 1), r, esize), esize, width);
    let tmask = replicate(ones(diff + 1), esize, width);
    Some((wmask, tmask))
}

/// Convert an integer register value to `f32` (SCVTF/UCVTF single). `sf`
/// selects the 64-bit (X) source width; otherwise the low 32 bits (W) are used.
#[allow(clippy::cast_precision_loss)]
fn int_to_f32(v: u64, signed: bool, sf: bool) -> f32 {
    match (signed, sf) {
        (true, true) => v as i64 as f32,
        (true, false) => v as i32 as f32,
        (false, true) => v as f32,
        (false, false) => v as u32 as f32,
    }
}

/// Convert an integer register value to `f64` (SCVTF/UCVTF double).
#[allow(clippy::cast_precision_loss)]
fn int_to_f64(v: u64, signed: bool, sf: bool) -> f64 {
    match (signed, sf) {
        (true, true) => v as i64 as f64,
        (true, false) => f64::from(v as i32),
        (false, true) => v as f64,
        (false, false) => f64::from(v as u32),
    }
}

/// Convert a floating-point value to an integer register value, rounding toward
/// zero (FCVTZS/FCVTZU). Rust's saturating float-to-int casts give the ARM
/// behaviour: NaN maps to 0 and out-of-range values saturate to the min/max.
fn fp_to_int(x: f64, signed: bool, sf: bool) -> u64 {
    match (signed, sf) {
        (true, true) => x as i64 as u64,
        (true, false) => u64::from(x as i32 as u32),
        (false, true) => x as u64,
        (false, false) => u64::from(x as u32),
    }
}

/// FMAX (single): larger of the two, with NaN propagated.
fn fmax32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.max(b)
    }
}

/// FMIN (single): smaller of the two, with NaN propagated.
fn fmin32(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() {
        f32::NAN
    } else {
        a.min(b)
    }
}

/// FMAX (double): larger of the two, with NaN propagated.
fn fmax64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// IEEE-754 equality for NEON `FCMEQ` lanes (unordered, e.g. either operand a
/// `NaN`, compares false — exactly the ARM semantics, not a "should these
/// have been equal" bug, hence the `_eq` name clippy's `float_cmp` exempts).
fn fp_eq(a: f64, b: f64) -> bool {
    a == b
}

/// Expand a NEON vector-compare result to an all-ones (true) or all-zero
/// (false) 32-bit lane, per the ARM `FCMEQ`/`FCMGE`/`FCMGT` convention.
fn fbits_bool32(cond: bool) -> u32 {
    if cond { u32::MAX } else { 0 }
}

/// As [`fbits_bool32`], for a 64-bit (double-precision) lane.
fn fbits_bool64(cond: bool) -> u64 {
    if cond { u64::MAX } else { 0 }
}

/// FMIN (double): smaller of the two, with NaN propagated.
fn fmin64(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

/// `FMULX` (single): like plain multiplication, except the IEEE-754-awkward
/// `0.0 * infinity` (or `infinity * 0.0`) case is redefined to `±2.0` (sign
/// from the XOR of the two operands' signs) instead of the default `NaN` —
/// every other input is a plain `a * b`. Zero-ness is tested via the bit
/// pattern (masking off the sign bit) rather than `==` so this doesn't need
/// a `clippy::float_cmp` exemption.
fn fmulx32(a: f32, b: f32) -> f32 {
    let a_zero = a.to_bits() & 0x7fff_ffff == 0;
    let b_zero = b.to_bits() & 0x7fff_ffff == 0;
    if (a_zero && b.is_infinite()) || (a.is_infinite() && b_zero) {
        if a.is_sign_negative() ^ b.is_sign_negative() {
            -2.0
        } else {
            2.0
        }
    } else {
        a * b
    }
}

/// As [`fmulx32`], for double precision.
fn fmulx64(a: f64, b: f64) -> f64 {
    let a_zero = a.to_bits() & 0x7fff_ffff_ffff_ffff == 0;
    let b_zero = b.to_bits() & 0x7fff_ffff_ffff_ffff == 0;
    if (a_zero && b.is_infinite()) || (a.is_infinite() && b_zero) {
        if a.is_sign_negative() ^ b.is_sign_negative() {
            -2.0
        } else {
            2.0
        }
    } else {
        a * b
    }
}

/// `URECPE` unsigned integer reciprocal estimate: `operand` and the result
/// are both unsigned 0.32 fixed-point fractions (implicitly scaled by
/// `2^32`) with `operand * result ≈ 2^64` — i.e. the result approximates
/// `1/operand` in that fixed-point system. The real `URECPE` instead
/// consults the ARM ARM's 8-bit reciprocal lookup table; this is a plainer
/// (but reasonably-behaved and monotonic) stand-in, in the same spirit as
/// using the exact reciprocal for `FRECPE` above.
fn u32_recip_estimate(operand: u32) -> u32 {
    let x = operand.max(0x8000_0000); // architecturally clamped to >= 0.5
    ((1u64 << 63) / u64::from(x)).min(u64::from(u32::MAX)) as u32
}

/// `URSQRTE` unsigned integer reciprocal-square-root estimate, in the same
/// 0.32 fixed-point system as [`u32_recip_estimate`] (`sqrt(operand) *
/// result ≈ 2^47`), using an exact `f64` square root rather than the ARM
/// ARM's lookup table.
fn u32_rsqrt_estimate(operand: u32) -> u32 {
    let x = operand.max(0x4000_0000); // architecturally clamped to >= 0.25
    let r = 2f64.powi(47) / f64::from(x).sqrt();
    r.min(f64::from(u32::MAX)) as u32
}

/// Convert IEEE-754 half-precision bits to `f32` (used by `FCVT`). Rust has no
/// stable `f16` type, so half-precision values are carried as raw `u16` bits
/// and converted through `f32`/`f64` at the point of use.
fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = u32::from((h >> 10) & 0x1f);
    let frac = u32::from(h & 0x3ff);
    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal half: normalize the mantissa into a normal f32.
        let mut e = 0i32;
        let mut f = frac;
        while f & 0x400 == 0 {
            f <<= 1;
            e -= 1;
        }
        f &= 0x3ff;
        let exp32 = (127 - 15 + 1 + e) as u32;
        return f32::from_bits(sign | (exp32 << 23) | (f << 13));
    }
    if exp == 0x1f {
        return f32::from_bits(sign | 0xff80_0000 | (frac << 13)); // inf / NaN
    }
    let exp32 = exp + (127 - 15);
    f32::from_bits(sign | (exp32 << 23) | (frac << 13))
}

/// Convert `f32` to IEEE-754 half-precision bits (used by `FCVT`), rounding to
/// nearest-even and flushing under/overflow to zero/infinity.
#[allow(clippy::cast_sign_loss)]
fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = i32::try_from((bits >> 23) & 0xff).unwrap_or(0);
    let frac = bits & 0x007f_ffff;
    if exp == 0xff {
        // Infinity, or NaN (force a nonzero mantissa so it stays a NaN).
        let payload = if frac != 0 {
            (frac >> 13) as u16 | 0x200
        } else {
            0
        };
        return sign | 0x7c00 | payload;
    }
    let hexp = exp - 127 + 15;
    if hexp >= 0x1f {
        return sign | 0x7c00; // overflow -> infinity
    }
    if hexp <= 0 {
        if hexp < -10 {
            return sign; // underflow -> zero
        }
        let frac_full = frac | 0x0080_0000; // restore the implicit leading 1
        let shift = (14 - hexp) as u32;
        let mut hfrac = (frac_full >> shift) as u16;
        let round_bit = 1u32 << (shift - 1);
        let rem = frac_full & ((round_bit << 1) - 1);
        if rem > round_bit || (rem == round_bit && hfrac & 1 == 1) {
            hfrac += 1;
        }
        return sign | hfrac;
    }
    let mut hexp16 = hexp as u16;
    let mut hfrac = (frac >> 13) as u16;
    let round_bit = frac & 0x1000;
    let sticky = frac & 0x0fff;
    if round_bit != 0 && (sticky != 0 || hfrac & 1 == 1) {
        hfrac += 1;
        if hfrac == 0x400 {
            hfrac = 0;
            hexp16 += 1;
            if hexp16 >= 0x1f {
                return sign | 0x7c00;
            }
        }
    }
    sign | (hexp16 << 10) | hfrac
}

/// Sign-extend the low `bits` of `v` to a full `i64`.
const fn sign_extend(v: u64, bits: u32) -> i64 {
    let shift = 64 - bits;
    ((v << shift) as i64) >> shift
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
const AES_SBOX: [u8; 256] = [
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
const AES_INV_SBOX: [u8; 256] = [
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
fn aes_round(vd: u128, vn: u128, encrypt: bool) -> u128 {
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
fn aes_mix_columns(vn: u128, forward: bool) -> u128 {
    let state = vn.to_le_bytes();
    let mut out = [0u8; 16];
    for (out_col, in_col) in out.chunks_exact_mut(4).zip(state.chunks_exact(4)) {
        let a = [in_col[0], in_col[1], in_col[2], in_col[3]];
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
fn aes_shift_rows(state: [u8; 16]) -> [u8; 16] {
    let mut out = [0u8; 16];
    for r in 0..4usize {
        for c in 0..4usize {
            out[r + 4 * c] = state[r + 4 * ((c + r) % 4)];
        }
    }
    out
}

/// `InvShiftRows`, the inverse permutation of [`aes_shift_rows`].
fn aes_inv_shift_rows(state: [u8; 16]) -> [u8; 16] {
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

/// FIPS-197 `MixColumns` on one 4-byte state column.
fn aes_mix_column(a: [u8; 4]) -> [u8; 4] {
    [
        gf_mul(2, a[0]) ^ gf_mul(3, a[1]) ^ a[2] ^ a[3],
        a[0] ^ gf_mul(2, a[1]) ^ gf_mul(3, a[2]) ^ a[3],
        a[0] ^ a[1] ^ gf_mul(2, a[2]) ^ gf_mul(3, a[3]),
        gf_mul(3, a[0]) ^ a[1] ^ a[2] ^ gf_mul(2, a[3]),
    ]
}

/// `InvMixColumns` on one 4-byte state column.
fn aes_inv_mix_column(a: [u8; 4]) -> [u8; 4] {
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
fn u32_lanes(v: u128) -> [u32; 4] {
    [
        v as u32,
        (v >> 32) as u32,
        (v >> 64) as u32,
        (v >> 96) as u32,
    ]
}

/// Read 32-bit lane `i` (0..=3) of `v`.
fn lane32(v: u128, i: u32) -> u32 {
    (v >> (i * 32)) as u32
}

/// Pack 4 32-bit lanes (lane 0 first) into a 128-bit vector.
fn pack_u32_lanes(l: [u32; 4]) -> u128 {
    u128::from(l[0])
        | (u128::from(l[1]) << 32)
        | (u128::from(l[2]) << 64)
        | (u128::from(l[3]) << 96)
}

/// Which SHA-1 nonlinear round function `SHA1C`/`SHA1P`/`SHA1M` runs.
#[derive(Clone, Copy)]
enum Sha1Op {
    /// `SHA1C`: `Ch(b,c,d)` — rounds 0..19.
    Choose,
    /// `SHA1P`: `Parity(b,c,d)` — rounds 20..39 and 60..79.
    Parity,
    /// `SHA1M`: `Maj(b,c,d)` — rounds 40..59.
    Majority,
}

fn sha1_f(op: Sha1Op, b: u32, c: u32, d: u32) -> u32 {
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
fn sha1_quad_round(abcd: u128, mut e: u32, wk: u128, op: Sha1Op) -> u128 {
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
fn sha1_su0(vd: u128, vn: u128, vm: u128) -> u128 {
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
fn sha1_su1(vd: u128, vn: u128) -> u128 {
    let d = u32_lanes(vd);
    let n = u32_lanes(vn);
    let w0 = (d[0] ^ n[1]).rotate_left(1);
    let w1 = (d[1] ^ n[2]).rotate_left(1);
    let w2 = (d[2] ^ n[3]).rotate_left(1);
    let w3 = (d[3] ^ w0).rotate_left(1);
    pack_u32_lanes([w0, w1, w2, w3])
}

fn sha256_ch(e: u32, f: u32, g: u32) -> u32 {
    (e & f) ^ (!e & g)
}
fn sha256_maj(a: u32, b: u32, c: u32) -> u32 {
    (a & b) ^ (a & c) ^ (b & c)
}
fn sha256_bsig0(a: u32) -> u32 {
    a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22)
}
fn sha256_bsig1(e: u32) -> u32 {
    e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25)
}
fn sha256_ssig0(x: u32) -> u32 {
    x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
}
fn sha256_ssig1(x: u32) -> u32 {
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
fn sha256_hash(abcd: u128, efgh: u128, wk: u128, want_efgh: bool) -> u128 {
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
fn sha256_su0(vd: u128, vn: u128) -> u128 {
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
fn sha256_su1(vd: u128, vn: u128, vm: u128) -> u128 {
    let d = u32_lanes(vd);
    let n = u32_lanes(vn);
    let m = u32_lanes(vm);
    let w0 = d[0].wrapping_add(sha256_ssig1(m[2])).wrapping_add(n[1]);
    let w1 = d[1].wrapping_add(sha256_ssig1(m[3])).wrapping_add(n[2]);
    let w2 = d[2].wrapping_add(sha256_ssig1(w0)).wrapping_add(n[3]);
    let w3 = d[3].wrapping_add(sha256_ssig1(w1)).wrapping_add(m[0]);
    pack_u32_lanes([w0, w1, w2, w3])
}

#[cfg(test)]
mod tests {
    // FP tests compare exactly-representable results (integers, halves) by value.
    #![allow(clippy::float_cmp)]
    use super::*;
    use crate::vcpu::mem::{PAGE_SIZE, Prot};

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
        // stxr w2,x3,[x1] — succeeds on our single core (status 0)
        assert!(matches!(c.exec(0xC802_7C23, &mut m), Step::Next));
        assert_eq!(c.x[2], 0, "store-exclusive reports success");
        // ldxr x0,[x1]
        assert!(matches!(c.exec(0xC85F_7C20, &mut m), Step::Next));
        assert_eq!(c.x[0], 0x42);
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
        assert_eq!(
            estimate, 0.25,
            "this interpreter's FRECPE is the exact reciprocal"
        );

        // One Newton-Raphson refinement step: y1 = y0 * frecps(x, y0), which
        // should already have converged to 1/x given an exact-reciprocal
        // FRECPE estimate.
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
}
