//! Software CPU interpreter backend — the portable, no-acceleration fallback.
//!
//! Decodes and executes guest instructions against [`GuestMemory`] in a loop,
//! returning [`Exit::Syscall`] when it decodes a syscall instruction (`svc #0`
//! on arm64). Slower than the hardware backends but runs anywhere and on any
//! guest arch — this is the path the browser (wasm) demo uses, and it makes the
//! syscall engine testable in CI with no hypervisor.
//!
//! The instruction set starts minimal (enough to run a hand-assembled
//! `write`/`exit` program) and grows toward full user-mode coverage in ROADMAP
//! Phase 10. Anything not yet decoded surfaces as [`Exit::IllegalInstruction`].

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
    // Used once branch instructions land (Phase 10).
    #[allow(dead_code)]
    Branched,
    /// `svc` — hand control to the kernel. `pc` stays on the `svc`; the kernel
    /// advances it via [`Vcpu::set_syscall_ret`].
    Syscall,
    Illegal,
    /// A load/store touched bad guest memory. (Used once load/store land.)
    #[allow(dead_code)]
    Fault { addr: u64, write: bool },
}

/// A minimal aarch64 user-mode interpreter.
struct Aarch64Interp {
    /// x0..x30. x31 is the zero register (reads 0) or SP depending on encoding.
    x: [u64; 31],
    sp: u64,
    pc: u64,
    tpidr: u64,
}

impl Aarch64Interp {
    fn new(entry: u64, stack: u64) -> Self {
        Self {
            x: [0; 31],
            sp: stack,
            pc: entry,
            tpidr: 0,
        }
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

    fn exec(&mut self, instr: u32) -> Step {
        // svc #imm  — the syscall gate.
        if instr & 0xFFE0_001F == 0xD400_0001 {
            return Step::Syscall;
        }
        // nop
        if instr == 0xD503_201F {
            return Step::Next;
        }
        // Move wide immediate: MOVN/MOVZ/MOVK.
        if (instr >> 23) & 0x3f == 0b1_00101 {
            let sf = (instr >> 31) & 1;
            let opc = (instr >> 29) & 3;
            let hw = (instr >> 21) & 3;
            let imm16 = u64::from((instr >> 5) & 0xffff);
            let rd = (instr & 0x1f) as usize;
            if sf == 0 && hw > 1 {
                return Step::Illegal; // 32-bit form only allows hw 0/1
            }
            let shift = hw * 16;
            let val = imm16 << shift;
            let result = match opc {
                0b10 => val,  // MOVZ
                0b00 => !val, // MOVN
                0b11 => {
                    // MOVK: keep the other 48 bits of Rd.
                    let cur = self.read_x(rd);
                    (cur & !(0xffff_u64 << shift)) | val
                }
                _ => return Step::Illegal,
            };
            let result = if sf == 0 { result & 0xffff_ffff } else { result };
            self.write_x(rd, result);
            return Step::Next;
        }
        // PC-relative addressing: ADR / ADRP.
        if (instr >> 24) & 0x1f == 0b1_0000 {
            let op = (instr >> 31) & 1;
            let immlo = u64::from((instr >> 29) & 3);
            let immhi = u64::from((instr >> 5) & 0x7ffff);
            let rd = (instr & 0x1f) as usize;
            let imm = sign_extend((immhi << 2) | immlo, 21);
            let result = if op == 0 {
                (self.pc as i64).wrapping_add(imm) as u64
            } else {
                ((self.pc & !0xfff) as i64).wrapping_add(imm << 12) as u64
            };
            self.write_x(rd, result);
            return Step::Next;
        }
        // Add/subtract immediate (also ADDS/SUBS; flags not modeled yet).
        if (instr >> 23) & 0x3f == 0b1_00010 {
            let sf = (instr >> 31) & 1;
            let op = (instr >> 30) & 1;
            let sh = (instr >> 22) & 1;
            let imm12 = u64::from((instr >> 10) & 0xfff);
            let rn = ((instr >> 5) & 0x1f) as usize;
            let rd = (instr & 0x1f) as usize;
            let imm = if sh == 1 { imm12 << 12 } else { imm12 };
            let a = self.read_sp(rn);
            let result = if op == 0 {
                a.wrapping_add(imm)
            } else {
                a.wrapping_sub(imm)
            };
            let result = if sf == 0 { result & 0xffff_ffff } else { result };
            self.write_sp(rd, result);
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
            match self.exec(instr) {
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
        [self.x[0], self.x[1], self.x[2], self.x[3], self.x[4], self.x[5]]
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

    #[test]
    fn movz_movk_build_64bit_immediate() {
        let mut c = cpu();
        // movz x1, #0x1000
        assert!(matches!(c.exec(0xD282_0001), Step::Next));
        assert_eq!(c.x[1], 0x1000);
        // movk x1, #0x1, lsl #16  -> x1 = 0x1_1000
        assert!(matches!(c.exec(0xF2A0_0021), Step::Next));
        assert_eq!(c.x[1], 0x1_1000);
    }

    #[test]
    fn movz_into_various_regs() {
        let mut c = cpu();
        c.exec(0xD280_0020); // movz x0, #1
        c.exec(0xD280_0062); // movz x2, #3
        c.exec(0xD280_0808); // movz x8, #64
        assert_eq!(c.x[0], 1);
        assert_eq!(c.x[2], 3);
        assert_eq!(c.x[8], 64);
    }

    #[test]
    fn add_sub_immediate() {
        let mut c = cpu();
        c.x[0] = 100;
        // add x1, x0, #5
        assert!(matches!(c.exec(0x9100_1401), Step::Next));
        assert_eq!(c.x[1], 105);
        // sub x2, x0, #10
        assert!(matches!(c.exec(0xD100_2802), Step::Next));
        assert_eq!(c.x[2], 90);
    }

    #[test]
    fn adr_is_pc_relative() {
        let mut c = cpu();
        c.pc = 0x1_0000;
        // adr x0, .+8  (imm=8)  encoding: immlo=(8&3)=0, immhi=8>>2=2
        // 0x10000000 | (immlo<<29) | (immhi<<5) | Rd
        let instr = 0x1000_0000 | (2 << 5);
        assert!(matches!(c.exec(instr), Step::Next));
        assert_eq!(c.x[0], 0x1_0008);
    }

    #[test]
    fn svc_traps_without_advancing() {
        let mut c = cpu();
        c.pc = 0x1_0004;
        assert!(matches!(c.exec(0xD400_0001), Step::Syscall));
        assert_eq!(c.pc, 0x1_0004, "svc must not advance pc itself");
        c.set_syscall_ret(0);
        assert_eq!(c.pc, 0x1_0008, "kernel advances pc after servicing");
    }

    #[test]
    fn unknown_instruction_is_illegal() {
        let mut c = cpu();
        assert!(matches!(c.exec(0x0000_0000), Step::Illegal));
    }

    #[test]
    fn run_faults_on_unmapped_pc() {
        let mut mem = GuestMemory::new(0x1_0000, PAGE_SIZE);
        let mut c = Aarch64Interp::new(0x1_0000, 0x1_0000);
        // pc page is not mapped → fetch fault
        assert_eq!(
            c.run(&mut mem).unwrap(),
            Exit::MemFault {
                addr: 0x1_0000,
                write: false
            }
        );
        // Map it and place a single svc: run stops with Syscall.
        mem.map(0x1_0000, PAGE_SIZE, Prot::rx()).unwrap();
        mem.write_init(0x1_0000, &0xD400_0001u32.to_le_bytes()).unwrap();
        assert_eq!(c.run(&mut mem).unwrap(), Exit::Syscall);
    }
}
