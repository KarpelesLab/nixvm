//! Software CPU interpreter backend — the portable, no-acceleration fallback.
//!
//! Decodes and executes guest instructions against [`GuestMemory`] in a loop,
//! returning [`Exit::Syscall`] when it decodes a syscall instruction (`svc #0`
//! on arm64). Slower than the hardware backends but runs anywhere and on any
//! guest arch — this is the path the browser (wasm) demo uses, and it makes the
//! syscall engine testable in CI with no hypervisor.
//!
//! Coverage grows toward full user-mode aarch64 (ROADMAP Phase 10). Implemented
//! so far: move-wide immediates, PC-relative addressing, add/sub (immediate and
//! shifted register, with flags), logical shifted register, compares,
//! conditional/unconditional branches, `BL`/`BLR`/`RET`, and load/store with an
//! unsigned immediate offset. Anything else surfaces as
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
    flags: Flags,
    /// SIMD/FP registers v0..v31 (128-bit; D/S/H/B views are the low bits).
    v: [u128; 32],
    /// Debug: print stores to this guest address (from `NIXVM_WATCH`).
    watch: Option<u64>,
}

impl Aarch64Interp {
    fn new(entry: u64, stack: u64) -> Self {
        Self {
            x: [0; 31],
            sp: stack,
            pc: entry,
            tpidr: 0,
            flags: Flags::default(),
            v: [0; 32],
            watch: std::env::var("NIXVM_WATCH")
                .ok()
                .and_then(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).ok()),
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
            if mem.write(addr, &bytes[..nbytes]).is_err() {
                return Step::Fault { addr, write: true };
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

    /// Read a system register named by the `MRS` encoding. Only `TPIDR_EL0`
    /// (the user thread pointer) is backed; everything else reads as 0.
    fn read_sysreg(&self, instr: u32) -> u64 {
        if (instr >> 5) & 0x7fff == TPIDR_EL0 {
            self.tpidr
        } else {
            0
        }
    }

    /// Write a system register named by the `MSR` encoding. Only `TPIDR_EL0` is
    /// backed; writes to other registers are ignored.
    fn write_sysreg(&mut self, instr: u32, value: u64) {
        if (instr >> 5) & 0x7fff == TPIDR_EL0 {
            self.tpidr = value;
        }
    }

    /// Debug: report a store overlapping the `NIXVM_WATCH` address.
    fn note_store(&self, addr: u64, value: u128, nbytes: usize) {
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
            if mem.write(addr, &val[..nbytes]).is_err() {
                return Step::Fault { addr, write: true };
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

        // ---- 2-source: UDIV/SDIV and variable shifts LSLV/LSRV/ASRV/RORV ----
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
                _ => return Step::Illegal,
            };
            self.write_x(rd, mask_sf(r, sf));
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
                    if mem.write(a, &bytes[..nbytes]).is_err() {
                        return Step::Fault {
                            addr: a,
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

        // ---- load/store exclusive & acquire/release (LDXR/STXR/LDAR/STLR) ----
        if (instr >> 24) & 0x3f == 0b00_1000 {
            let size = (instr >> 30) & 3;
            let o2 = (instr >> 23) & 1; // 0 = exclusive, 1 = ordered (LDAR/STLR)
            let l = (instr >> 22) & 1; // 1 = load
            let o1 = (instr >> 21) & 1; // 1 = pair (LDXP/STXP)
            if o1 == 1 {
                return Step::Illegal; // exclusive pair: Phase 10
            }
            let rs = reg_field(instr, 16);
            let rn = reg_field(instr, 5);
            let rt = reg_field(instr, 0);
            let addr = self.read_sp(rn);
            if l == 1 {
                return self.ldst(addr, size, 0b01, rt, mem);
            }
            let step = self.ldst(addr, size, 0b00, rt, mem);
            if matches!(step, Step::Next) && o2 == 0 {
                // Store-exclusive always succeeds on our single core: status = 0.
                self.write_x(rs, 0);
            }
            return step;
        }

        // ---- load/store pair: LDP/STP (signed offset / pre / post index) ----
        if (instr >> 27) & 0x7 == 0b101 && (instr >> 26) & 1 == 0 && (instr >> 25) & 1 == 0 {
            let opc = (instr >> 30) & 3;
            let is64 = match opc {
                0b00 => false,
                0b10 => true,
                _ => return Step::Illegal, // SIMD/other pairs: Phase 10
            };
            let class = (instr >> 23) & 3; // 1 = post, 2 = signed offset, 3 = pre
            let is_load = (instr >> 22) & 1 == 1;
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
                    self.write_x(r, u64::from_le_bytes(buf));
                } else {
                    self.note_store(a, u128::from(self.read_x(r)), nbytes);
                    let val = self.read_x(r).to_le_bytes();
                    if mem.write(a, &val[..nbytes]).is_err() {
                        return Step::Fault {
                            addr: a,
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

        // ---- FMOV between GP and SIMD/FP registers (bit-exact, no convert) ----
        if (instr >> 24) & 0x7f == 0b0011110 && (instr >> 21) & 1 == 1 && (instr >> 10) & 0x3f == 0
        {
            let ftype = (instr >> 22) & 3; // 00 = S (32-bit), 01 = D (64-bit)
            let rmode = (instr >> 19) & 3;
            let opcode = (instr >> 16) & 7;
            let rn = reg_field(instr, 5);
            let rd = reg_field(instr, 0);
            match (rmode, opcode) {
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
        self.flags = Flags::default();
    }
}

/// Encoded `op0<0>:op1:CRn:CRm:op2` field (bits 19:5) for `TPIDR_EL0`.
const TPIDR_EL0: u32 = 0x5E82;

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

/// Sign-extend the low `bits` of `v` to a full `i64`.
const fn sign_extend(v: u64, bits: u32) -> i64 {
    let shift = 64 - bits;
    ((v << shift) as i64) >> shift
}

#[cfg(test)]
mod tests {
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
    fn svc_traps_without_advancing() {
        let (mut c, mut m) = (cpu(), scratch());
        c.pc = 0x1_0004;
        assert!(matches!(c.exec(0xD400_0001, &mut m), Step::Syscall));
        assert_eq!(c.pc, 0x1_0004);
        c.set_syscall_ret(0);
        assert_eq!(c.pc, 0x1_0008);
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
}
