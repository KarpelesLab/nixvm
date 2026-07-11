//! End-to-end SMP determinism test: a small aarch64 program forks four
//! children, each doing a bit of arithmetic and exiting with the result; the
//! parent `wait4`s all four and exits with their sum. The test runs this
//! program repeatedly under `Kernel::set_ncpus(1)` (the cooperative serial
//! scheduler) and `Kernel::set_ncpus(4)` (the SMP scheduler, which runs guest
//! compute on parallel host worker threads while servicing syscalls serially —
//! see `src/kernel/mod.rs`'s `schedule_smp`) and asserts they agree, several
//! times over, to catch any scheduler-dependent nondeterminism.
//!
//! Self-contained: its own ELF builder + instruction encoders, duplicated from
//! `tests/aarch64_smoke.rs` / `tests/proc_ipc.rs` rather than shared (see
//! `tests/proc_ipc.rs`'s module doc for why, and for the two-pass
//! address-patching trick used below for the forward `CBZ` branches).

use std::io::Write;
use std::sync::{Arc, Mutex};

use nixvm::abi::Arch;
use nixvm::fs::MountTable;
use nixvm::kernel::Kernel;
use nixvm::loader::{ProcessSpec, load_static};
use nixvm::vcpu::Backend;
use nixvm::vcpu::GuestMemory;
use nixvm::vcpu::interp::InterpBackend;
use nixvm::vcpu::mem::PAGE_SIZE;

const EHDR_LEN: usize = 64;
const PHDR_LEN: usize = 56;
const EM_AARCH64: u16 = 183;
const BODY_OFF: u64 = (EHDR_LEN + PHDR_LEN) as u64;

// aarch64 syscall numbers used below (see `src/abi/arch/aarch64.rs`).
const NR_CLONE: u32 = 220;
const NR_EXIT: u32 = 93;
const NR_WAIT4: u32 = 260;

// ---- instruction encoders (A64) --------------------------------------------
//
// See `tests/proc_ipc.rs` for the same helpers with fuller commentary; kept
// identical here so the two files' encoded instructions are trivially
// comparable.

/// `MOVZ Xd, #imm16` (64-bit, shift 0).
fn movz(rd: u32, imm16: u32) -> u32 {
    0xD280_0000 | (imm16 << 5) | rd
}

/// `MOVK Xd, #imm16, LSL #16` (64-bit).
fn movk16(rd: u32, imm16: u32) -> u32 {
    0xF2A0_0000 | (imm16 << 5) | rd
}

/// `SVC #0`.
fn svc0() -> u32 {
    0xD400_0001
}

/// One or two instructions materializing a compile-time-known 32-bit-range
/// immediate into `Xd` (`MOVZ`, plus `MOVK` if the high half is nonzero).
fn mov_imm32(rd: u32, val: u32) -> Vec<u32> {
    let lo = val & 0xffff;
    let hi = (val >> 16) & 0xffff;
    let mut words = vec![movz(rd, lo)];
    if hi != 0 {
        words.push(movk16(rd, hi));
    }
    words
}

/// Exactly two instructions (`MOVZ` + `MOVK`) materializing `val` into `Xd`,
/// even when the high half is zero — used for the one address in this
/// program (`wstatus`) whose value depends on the code's own length; see the
/// module doc for the two-pass build this enables.
fn mov_addr(rd: u32, val: u32) -> [u32; 2] {
    [movz(rd, val & 0xffff), movk16(rd, (val >> 16) & 0xffff)]
}

/// `LDRB Wt, [Xn, #imm12]` (unsigned immediate offset, zero-extending byte
/// load).
fn ldrb(rt: u32, rn: u32, imm12: u32) -> u32 {
    0x3940_0000 | (imm12 << 10) | (rn << 5) | rt
}

/// `MUL Xd, Xn, Xm` (alias of `MADD Xd, Xn, Xm, XZR`).
fn mul(rd: u32, rn: u32, rm: u32) -> u32 {
    0x9B00_7C00 | (rm << 16) | (rn << 5) | rd
}

/// `ADD Xd, Xn, Xm` (shifted register, no shift).
fn add_reg(rd: u32, rn: u32, rm: u32) -> u32 {
    0x8B00_0000 | (rm << 16) | (rn << 5) | rd
}

/// `MOV Xd, Xm` (alias of `ORR Xd, XZR, Xm`).
fn mov_reg(rd: u32, rm: u32) -> u32 {
    0xAA00_03E0 | (rm << 16) | rd
}

/// `CBZ Xt, #byte_offset` (64-bit): branch (relative to this instruction's
/// own address, must be a multiple of 4) if `Xt == 0`.
fn cbz(rt: u32, byte_offset: i32) -> u32 {
    let imm19 = ((byte_offset / 4) as u32) & 0x7_ffff;
    0xB400_0000 | (imm19 << 5) | rt
}

// ---- ELF + run harness ------------------------------------------------------

#[derive(Clone)]
struct SharedBuf(Arc<Mutex<Vec<u8>>>);
impl Write for SharedBuf {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Assemble a minimal ET_EXEC aarch64 ELF: one RWX PT_LOAD at `vaddr` covering
/// the headers + `body`, entry at the start of `body`.
fn build_elf(vaddr: u64, body: &[u8]) -> Vec<u8> {
    let mut f = vec![0u8; EHDR_LEN + PHDR_LEN];
    f[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
    f[4] = 2; // ELFCLASS64
    f[5] = 1; // ELFDATA2LSB
    f[6] = 1;
    f[16..18].copy_from_slice(&2u16.to_le_bytes()); // ET_EXEC
    f[18..20].copy_from_slice(&EM_AARCH64.to_le_bytes());
    f[20..24].copy_from_slice(&1u32.to_le_bytes());
    f[24..32].copy_from_slice(&(vaddr + BODY_OFF).to_le_bytes()); // e_entry
    f[32..40].copy_from_slice(&(EHDR_LEN as u64).to_le_bytes()); // e_phoff
    f[52..54].copy_from_slice(&(EHDR_LEN as u16).to_le_bytes());
    f[54..56].copy_from_slice(&(PHDR_LEN as u16).to_le_bytes());
    f[56..58].copy_from_slice(&1u16.to_le_bytes()); // e_phnum

    let total = BODY_OFF + body.len() as u64;
    let p = EHDR_LEN;
    f[p..p + 4].copy_from_slice(&1u32.to_le_bytes()); // PT_LOAD
    f[p + 4..p + 8].copy_from_slice(&7u32.to_le_bytes()); // R|W|X
    f[p + 16..p + 24].copy_from_slice(&vaddr.to_le_bytes()); // p_vaddr
    f[p + 24..p + 32].copy_from_slice(&vaddr.to_le_bytes());
    f[p + 32..p + 40].copy_from_slice(&total.to_le_bytes()); // p_filesz
    f[p + 40..p + 48].copy_from_slice(&total.to_le_bytes()); // p_memsz
    f[p + 48..p + 56].copy_from_slice(&PAGE_SIZE.to_le_bytes());

    f.extend_from_slice(body);
    f
}

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    bytes
}

/// Load `body` at `vaddr` and run it to completion on the software
/// interpreter with `ncpus` virtual CPUs, capturing stdout. Returns
/// `(exit_code, stdout, kernel)`.
fn run_program(vaddr: u64, body: &[u8], ncpus: usize) -> (i32, Vec<u8>, Kernel) {
    let elf = build_elf(vaddr, body);

    let mut mem = GuestMemory::new(vaddr, 256 * PAGE_SIZE);
    let spec = ProcessSpec {
        argv: vec!["prog".into()],
        envp: vec![],
    };
    let img = load_static(&mut mem, &elf, &spec).unwrap();

    let backend = InterpBackend::new(Arch::Aarch64).unwrap();
    let vcpu = backend.new_vcpu(img.entry, img.stack_pointer).unwrap();

    let captured = Arc::new(Mutex::new(Vec::new()));
    let mut kernel = Kernel::new(Arch::Aarch64, MountTable::new());
    kernel.set_stdout(Box::new(SharedBuf(captured.clone())));
    kernel.set_ncpus(ncpus);

    let exit_code = kernel.run(vcpu, mem).unwrap();
    let out = captured.lock().unwrap().clone();
    (exit_code, out, kernel)
}

/// Build the fork-4-children-and-sum program:
///
///   for i in 1..=4 { if clone(0, 0) == 0 { exit(i * 7) } }  // 4 direct children
///   let mut sum = 0;
///   for _ in 1..=4 { sum += wait4_any_childs_exit_code() }   // reap all 4
///   exit(sum)
///
/// laid out as: 4x (clone + forward `CBZ` to that child's block), then the
/// accumulator init + 4x unrolled `wait4`-and-accumulate + final `exit`, then
/// the 4 child blocks themselves. The `CBZ` targets are resolved by recording
/// each branch's position and each child block's start position (both are
/// just `Vec` lengths at the time), then patching the branch words in a final
/// pass — no manual instruction-counting.
fn build_code(wstatus_addr: u32) -> Vec<u32> {
    let mut code = Vec::new();
    let mut cbz_positions = Vec::new();
    for _ in 0..4u32 {
        code.extend(mov_imm32(0, 0)); // clone flags = 0
        code.extend(mov_imm32(1, 0)); // clone stack = 0 (share parent's)
        code.extend(mov_imm32(8, NR_CLONE));
        code.push(svc0()); // x0 = child pid (parent) or 0 (child)
        cbz_positions.push(code.len());
        code.push(0); // CBZ placeholder, patched below once child_starts is known
    }

    code.extend(mov_imm32(21, 0)); // x21 = accumulator

    for _ in 0..4u32 {
        code.extend(mov_imm32(0, 0)); // pid arg (ignored by this kernel's wait4)
        code.extend(mov_addr(1, wstatus_addr)); // x1 = &wstatus
        code.extend(mov_imm32(2, 0)); // options = 0
        code.extend(mov_imm32(8, NR_WAIT4));
        code.push(svc0());
        code.push(ldrb(2, 1, 1)); // x2 = reaped child's exit code (wstatus byte 1)
        code.push(add_reg(21, 21, 2)); // x21 += x2
    }

    code.push(mov_reg(0, 21));
    code.extend(mov_imm32(8, NR_EXIT));
    code.push(svc0());

    let mut child_starts = Vec::new();
    for i in 1..=4u32 {
        child_starts.push(code.len());
        code.extend(mov_imm32(0, i)); // x0 = i
        code.extend(mov_imm32(1, 7)); // x1 = 7
        code.push(mul(0, 0, 1)); // x0 = i * 7
        code.extend(mov_imm32(8, NR_EXIT));
        code.push(svc0());
    }

    for (k, pos) in cbz_positions.into_iter().enumerate() {
        let off_bytes = ((child_starts[k] as i64 - pos as i64) * 4) as i32;
        code[pos] = cbz(0, off_bytes);
    }
    code
}

/// `1*7 + 2*7 + 3*7 + 4*7`: the deterministic result every run must produce,
/// regardless of scheduling order (sum is commutative, so it doesn't matter
/// which of the 4 children `wait4` happens to reap first).
const EXPECTED_EXIT: i32 = 7 + 14 + 21 + 28;

#[test]
fn smp_scheduling_matches_serial_fork_sum() {
    let vaddr = 0x1_0000u64;

    let code_words = build_code(0).len() as u64;
    let wstatus_addr = u32::try_from(vaddr + BODY_OFF + code_words * 4).unwrap();
    let instrs = build_code(wstatus_addr);
    assert_eq!(
        instrs.len() as u64,
        code_words,
        "two-pass build must be length-stable"
    );

    let mut body = words_to_bytes(&instrs);
    body.extend_from_slice(&[0u8; 4]); // wstatus scratch word, filled by wait4

    // Run several times under each scheduler to catch scheduler-dependent
    // nondeterminism, not just a single lucky interleaving.
    for iter in 0..15 {
        let (serial_code, serial_out, serial_kernel) = run_program(vaddr, &body, 1);
        assert_eq!(serial_code, EXPECTED_EXIT, "serial (ncpus=1) run {iter}");
        assert!(serial_out.is_empty());
        assert!(
            serial_kernel.unsupported().is_empty(),
            "{:?}",
            serial_kernel.unsupported()
        );

        let (smp_code, smp_out, smp_kernel) = run_program(vaddr, &body, 4);
        assert_eq!(
            smp_code, EXPECTED_EXIT,
            "SMP (ncpus=4) run {iter} must agree with the serial scheduler"
        );
        assert!(smp_out.is_empty());
        assert!(
            smp_kernel.unsupported().is_empty(),
            "{:?}",
            smp_kernel.unsupported()
        );
    }
}
