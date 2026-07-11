//! End-to-end aarch64 *compute* integration tests: each program below does
//! real arithmetic / floating-point / NEON / atomic work in registers and
//! memory, then folds the result into its process exit code, so a wrong
//! instruction implementation makes the test fail rather than merely
//! producing different (unchecked) output. Complements `tests/aarch64_smoke.rs`
//! (loader/syscall plumbing) and `tests/proc_ipc.rs` (process/IPC control
//! flow) by exercising instruction *breadth*: integer ALU ops incl. a
//! SUBS/CBNZ loop, bitfield extraction + conditional-select/-compare,
//! scalar floating point, NEON vectors, and LSE atomic read-modify-write.
//!
//! Like those two files, this one hand-assembles minimal static ELF64
//! images, loads them with the real loader (`nixvm::loader::load_static`),
//! and runs them on the software interpreter
//! (`InterpBackend::new(Arch::Aarch64)`) with a captured-stdout `Kernel` —
//! the same wiring as `src/bin/run-elf.rs`. It is self-contained (its own
//! ELF builder + instruction encoders): integration test binaries in
//! `tests/` don't share code cleanly, so the handful of encoder helpers also
//! present in `tests/aarch64_smoke.rs` / `tests/proc_ipc.rs` are duplicated
//! here rather than factored out.
//!
//! ## Encoding methodology
//!
//! Every encoder below is a small formula (register/immediate fields packed
//! into a 32-bit word), not a copied hex literal, and every formula was
//! cross-checked two ways before use:
//!
//!   1. Against this crate's own `src/vcpu/interp.rs` unit tests, which
//!      already exercise many of these exact instructions with known-good
//!      hex literals (e.g. `sbfx_ubfx_extract_bitfield`, `ccmp_feeds_csel`,
//!      `csel_and_csinc_use_flags`, `scvtf_fcvtzs_roundtrip_integer`,
//!      `neon_dup_umov_roundtrip`, `ldadd_ldset_ldclr_return_old_value_and_update_memory`,
//!      `cas_success_and_failure_paths`, `fp_arithmetic_double`).
//!   2. By assembling the same mnemonics natively with the host's own
//!      toolchain (this dev host is `arm64`, so `as -arch arm64` / `objdump
//!      -d` assemble and disassemble real aarch64 instructions) and solving
//!      for each field by varying one register/immediate at a time and
//!      differencing the resulting words — e.g. `sub x4,x5,x6` assembles to
//!      `0xcb0600a4`, matching `0xCB00_0000 | (rm<<16)|(rn<<5)|rd` with
//!      `rd=4,rn=5,rm=6`.
//!
//! Both checks agreed on every formula used here.

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

const NR_EXIT: u32 = 93;
const NR_MMAP: u32 = 222;

// AArch64 condition-code encodings (`cond` field of B.cond/CSEL/CCMP/…).
const COND_EQ: u32 = 0b0000;
const COND_MI: u32 = 0b0100;
const COND_PL: u32 = 0b0101;

// ---- instruction encoders (A64) --------------------------------------------
//
// `rd`/`rn`/`rm`/`rt`/`rs` are `0..=30` for `x0..=x30`/`v0..=v30` (31 means
// `xzr`/`wzr`, per the ISA, in the fields these helpers place it in).

/// `MOVZ Xd, #imm16` (64-bit, shift 0): loads `imm16` into the low 16 bits,
/// zeroing the rest.
fn movz(rd: u32, imm16: u32) -> u32 {
    0xD280_0000 | (imm16 << 5) | rd
}

/// `MOVK Xd, #imm16, LSL #(16*hw)` (64-bit): merges `imm16` into bits
/// `[16*hw+15 : 16*hw]`, leaving the rest of `Xd` untouched. `hw` is `0..=3`.
fn movk(rd: u32, imm16: u32, hw: u32) -> u32 {
    0xF280_0000 | (hw << 21) | (imm16 << 5) | rd
}

/// `SVC #0`: trap to the kernel (syscall number in `x8`, args in `x0..x5`).
fn svc0() -> u32 {
    0xD400_0001
}

/// Materialize a 32-bit-range immediate into `Xd` as one or two instructions
/// (`MOVZ`, plus `MOVK ,LSL #16` if the high half is nonzero).
fn mov_imm32(rd: u32, val: u32) -> Vec<u32> {
    let lo = val & 0xffff;
    let hi = (val >> 16) & 0xffff;
    let mut words = vec![movz(rd, lo)];
    if hi != 0 {
        words.push(movk(rd, hi, 1));
    }
    words
}

/// Materialize an arbitrary 64-bit immediate into `Xd` as exactly four
/// instructions (`MOVZ` + three `MOVK`s), used for the full 64-bit bit
/// pattern the bitfield test needs (rather than the two-instruction, 32-bit
/// range `mov_imm32` above).
fn mov_imm64(rd: u32, val: u64) -> [u32; 4] {
    [
        movz(rd, (val & 0xffff) as u32),
        movk(rd, ((val >> 16) & 0xffff) as u32, 1),
        movk(rd, ((val >> 32) & 0xffff) as u32, 2),
        movk(rd, ((val >> 48) & 0xffff) as u32, 3),
    ]
}

/// Materialize a value into `Xd` as exactly two instructions (`MOVZ` +
/// `MOVK`), even when the high half is zero — used for a guest address whose
/// numeric value isn't known until the surrounding code's length is (two-pass
/// build: measure with a placeholder, then re-emit with the real address).
fn mov_addr(rd: u32, val: u32) -> [u32; 2] {
    [movz(rd, val & 0xffff), movk(rd, (val >> 16) & 0xffff, 1)]
}

/// `STRB Wt, [Xn, #imm12]` (unsigned immediate offset, byte store).
fn strb(rt: u32, rn: u32, imm12: u32) -> u32 {
    0x3900_0000 | (imm12 << 10) | (rn << 5) | rt
}

/// `LDRB Wt, [Xn, #imm12]` (unsigned immediate offset, zero-extending byte
/// load).
fn ldrb(rt: u32, rn: u32, imm12: u32) -> u32 {
    0x3940_0000 | (imm12 << 10) | (rn << 5) | rt
}

/// `LDRH Wt, [Xn, #(imm12*2)]` (unsigned immediate offset, zero-extending
/// halfword load).
fn ldrh(rt: u32, rn: u32, imm12: u32) -> u32 {
    0x7940_0000 | (imm12 << 10) | (rn << 5) | rt
}

/// `LDR Wt, [Xn, #(imm12*4)]` (unsigned immediate offset, 32-bit load,
/// zero-extended into `Xt`).
fn ldr_w(rt: u32, rn: u32, imm12: u32) -> u32 {
    0xB940_0000 | (imm12 << 10) | (rn << 5) | rt
}

/// `LDR Xt, [Xn, #(imm12*8)]` (unsigned immediate offset, 64-bit load).
fn ldr_x(rt: u32, rn: u32, imm12: u32) -> u32 {
    0xF940_0000 | (imm12 << 10) | (rn << 5) | rt
}

/// `MOV Xd, Xm` (alias of `ORR Xd, XZR, Xm`).
fn mov_reg(rd: u32, rm: u32) -> u32 {
    0xAA00_03E0 | (rm << 16) | rd
}

/// `ADD Xd, Xn, Xm` (shifted register, no shift).
fn add_reg(rd: u32, rn: u32, rm: u32) -> u32 {
    0x8B00_0000 | (rm << 16) | (rn << 5) | rd
}

/// `SUB Xd, Xn, Xm` (shifted register, no shift).
fn sub_reg(rd: u32, rn: u32, rm: u32) -> u32 {
    0xCB00_0000 | (rm << 16) | (rn << 5) | rd
}

/// `SUBS Xd, Xn, #imm12` (`Xd == xzr`, i.e. `rd == 31`, gives the `CMP`
/// alias: flags are set, the subtraction result is discarded).
fn subs_imm(rd: u32, rn: u32, imm12: u32) -> u32 {
    0xF100_0000 | (imm12 << 10) | (rn << 5) | rd
}

/// `AND Xd, Xn, Xm` (shifted register, no shift).
fn and_reg(rd: u32, rn: u32, rm: u32) -> u32 {
    0x8A00_0000 | (rm << 16) | (rn << 5) | rd
}

/// `ORR Xd, Xn, Xm` (shifted register, no shift).
fn orr_reg(rd: u32, rn: u32, rm: u32) -> u32 {
    0xAA00_0000 | (rm << 16) | (rn << 5) | rd
}

/// `EOR Xd, Xn, Xm` (shifted register, no shift).
fn eor_reg(rd: u32, rn: u32, rm: u32) -> u32 {
    0xCA00_0000 | (rm << 16) | (rn << 5) | rd
}

/// `MUL Xd, Xn, Xm` (alias of `MADD Xd, Xn, Xm, XZR`).
fn mul(rd: u32, rn: u32, rm: u32) -> u32 {
    0x9B00_7C00 | (rm << 16) | (rn << 5) | rd
}

/// `UDIV Xd, Xn, Xm` (unsigned integer division, truncating).
fn udiv(rd: u32, rn: u32, rm: u32) -> u32 {
    0x9AC0_0800 | (rm << 16) | (rn << 5) | rd
}

/// `LSL Xd, Xn, Xm` (alias of `LSLV`, register-controlled shift).
fn lslv(rd: u32, rn: u32, rm: u32) -> u32 {
    0x9AC0_2000 | (rm << 16) | (rn << 5) | rd
}

/// `LSR Xd, Xn, Xm` (alias of `LSRV`, register-controlled shift).
fn lsrv(rd: u32, rn: u32, rm: u32) -> u32 {
    0x9AC0_2400 | (rm << 16) | (rn << 5) | rd
}

/// `CBNZ Xt, #byte_offset` (64-bit): branch (relative to this instruction's
/// own address, must be a multiple of 4) if `Xt != 0`.
fn cbnz(rt: u32, byte_offset: i32) -> u32 {
    let imm19 = ((byte_offset / 4) as u32) & 0x7_ffff;
    0xB500_0000 | (imm19 << 5) | rt
}

/// `SBFX Xd, Xn, #lsb, #width` (signed bitfield extract).
fn sbfx(rd: u32, rn: u32, lsb: u32, width: u32) -> u32 {
    0x9340_0000 | (lsb << 16) | ((lsb + width - 1) << 10) | (rn << 5) | rd
}

/// `UBFX Xd, Xn, #lsb, #width` (unsigned bitfield extract).
fn ubfx(rd: u32, rn: u32, lsb: u32, width: u32) -> u32 {
    0xD340_0000 | (lsb << 16) | ((lsb + width - 1) << 10) | (rn << 5) | rd
}

/// `CSEL Xd, Xn, Xm, cond`: `Xd = cond ? Xn : Xm`.
fn csel(rd: u32, rn: u32, rm: u32, cond: u32) -> u32 {
    0x9A80_0000 | (rm << 16) | (cond << 12) | (rn << 5) | rd
}

/// `CSINC Xd, Xn, Xm, cond`: `Xd = cond ? Xn : (Xm + 1)`.
fn csinc(rd: u32, rn: u32, rm: u32, cond: u32) -> u32 {
    0x9A80_0400 | (rm << 16) | (cond << 12) | (rn << 5) | rd
}

/// `CCMP Xn, #imm5, #nzcv, cond`: if `cond` holds, compares `Xn` against
/// `imm5` (setting NZCV as `CMP` would); otherwise loads NZCV directly from
/// the literal `nzcv` (4-bit `{N,Z,C,V}`).
fn ccmp_imm(rn: u32, imm5: u32, nzcv: u32, cond: u32) -> u32 {
    0xFA40_0800 | (imm5 << 16) | (cond << 12) | (rn << 5) | nzcv
}

/// `SCVTF Dd, Xn` (signed 64-bit integer -> double, round to nearest).
fn scvtf_d(rd: u32, rn: u32) -> u32 {
    0x9E62_0000 | (rn << 5) | rd
}

/// `FCVTZS Xd, Dn` (double -> signed 64-bit integer, round toward zero).
fn fcvtzs_xd(rd: u32, rn: u32) -> u32 {
    0x9E78_0000 | (rn << 5) | rd
}

/// `FADD Dd, Dn, Dm` (double precision).
fn fadd_d(rd: u32, rn: u32, rm: u32) -> u32 {
    0x1E60_2800 | (rm << 16) | (rn << 5) | rd
}

/// `FMUL Dd, Dn, Dm` (double precision).
fn fmul_d(rd: u32, rn: u32, rm: u32) -> u32 {
    0x1E60_0800 | (rm << 16) | (rn << 5) | rd
}

/// `FDIV Dd, Dn, Dm` (double precision).
fn fdiv_d(rd: u32, rn: u32, rm: u32) -> u32 {
    0x1E60_1800 | (rm << 16) | (rn << 5) | rd
}

/// `FSQRT Dd, Dn` (double precision).
fn fsqrt_d(rd: u32, rn: u32) -> u32 {
    0x1E61_C000 | (rn << 5) | rd
}

/// `DUP Vd.4S, Wn` (replicate a 32-bit GP register across all 4 lanes).
fn dup_4s(rd: u32, rn: u32) -> u32 {
    0x4E04_0C00 | (rn << 5) | rd
}

/// `INS Vd.S[index], Wn` (insert a GP register into one 32-bit lane).
fn ins_s(rd: u32, index: u32, rn: u32) -> u32 {
    let imm5 = ((index << 3) | 0b100) & 0x1f;
    0x4E00_1C00 | (imm5 << 16) | (rn << 5) | rd
}

/// `MOVI Vd.4S, #imm8` (replicate an 8-bit immediate into all 4 32-bit
/// lanes, shift amount 0).
fn movi_4s(rd: u32, imm8: u32) -> u32 {
    0x4F00_0400 | (((imm8 >> 5) & 0x7) << 16) | ((imm8 & 0x1f) << 5) | rd
}

/// `ADD Vd.4S, Vn.4S, Vm.4S` (vector, 32-bit lanes).
fn add_4s(rd: u32, rn: u32, rm: u32) -> u32 {
    0x4EA0_8400 | (rm << 16) | (rn << 5) | rd
}

/// `MUL Vd.4S, Vn.4S, Vm.4S` (vector, 32-bit lanes).
fn mul_4s(rd: u32, rn: u32, rm: u32) -> u32 {
    0x4EA0_9C00 | (rm << 16) | (rn << 5) | rd
}

/// `ADDV Sd, Vn.4S` (across-lanes add reduction of four 32-bit lanes).
fn addv_4s(rd: u32, rn: u32) -> u32 {
    0x4EB1_B800 | (rn << 5) | rd
}

/// `UMOV Wd, Vn.S[index]` (move one 32-bit lane to a GP register).
fn umov_s(rd: u32, rn: u32, index: u32) -> u32 {
    let imm5 = ((index << 3) | 0b100) & 0x1f;
    0x0E00_3C00 | (imm5 << 16) | (rn << 5) | rd
}

/// `LDADD Ws, Wt, [Xn]` (LSE atomic: `Wt = *Xn; *Xn += Ws`, non-tearing).
fn ldadd_w(rs: u32, rt: u32, rn: u32) -> u32 {
    0xB820_0000 | (rs << 16) | (rn << 5) | rt
}

/// `CAS Ws, Wt, [Xn]` (LSE atomic compare-and-swap: if `*Xn == Ws`, `*Xn =
/// Wt`; either way `Ws` is overwritten with the pre-swap value of `*Xn`).
fn cas_w(rs: u32, rt: u32, rn: u32) -> u32 {
    0x88A0_7C00 | (rs << 16) | (rn << 5) | rt
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

/// Assemble a minimal ET_EXEC aarch64 ELF: one RWX PT_LOAD at `vaddr`
/// covering the headers + `body` (code followed by any trailing scratch
/// data), entry at the start of `body`. Mirrors `tests/aarch64_smoke.rs`'s
/// `build_elf`.
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
/// interpreter, capturing stdout. Returns `(exit_code, stdout, kernel)`.
/// Mirrors `tests/aarch64_smoke.rs`'s `run_program`; used by every test here
/// that doesn't need an explicit heap/mmap arena.
fn run_program(vaddr: u64, body: &[u8]) -> (i32, Vec<u8>, Kernel) {
    let elf = build_elf(vaddr, body);

    let mut mem = GuestMemory::new(vaddr, 256 * PAGE_SIZE);
    let spec = ProcessSpec {
        argv: vec!["prog".into()],
        envp: vec![],
    };
    let img = load_static(&mut mem, &elf, &spec).unwrap();

    // Force the portable software interpreter (as `src/bin/run-elf.rs` does),
    // not `vcpu::select`, which would pick a hardware backend (e.g. HVF) on a
    // capable host and bypass the interpreter this test targets.
    let backend = InterpBackend::new(Arch::Aarch64).unwrap();
    let vcpu = backend.new_vcpu(img.entry, img.stack_pointer).unwrap();

    let captured = Arc::new(Mutex::new(Vec::new()));
    let mut kernel = Kernel::new(Arch::Aarch64, MountTable::new());
    kernel.set_stdout(Box::new(SharedBuf(captured.clone())));

    let code = kernel.run(vcpu, mem).unwrap();
    let out = captured.lock().unwrap().clone();
    (code, out, kernel)
}

// ---- 1. integer arithmetic chain + a SUBS/CBNZ summation loop --------------
//
// Loop: `sum(1..=10)` via `ADD` (accumulate) / `SUBS #1` (decrement +
// flags) / `CBNZ` (loop-while-nonzero) -> 55. Then a chain of
// SUB/MUL/UDIV/AND/ORR/EOR/LSL/LSR (register forms) walks that 55 through a
// fixed sequence of operations to a single small, order-sensitive result: if
// any one of these instructions computes the wrong value, the final exit
// code changes.
//
//   55 -6=49  *2=98  /7=14  &31=14  |64=78  ^8=70  <<1=140  >>3=17

#[test]
fn integer_chain_with_summation_loop_exit_code() {
    let vaddr = 0x1_0000u64;

    let mut code = Vec::new();
    code.push(movz(0, 0)); // x0 = 0  (sum)
    code.push(movz(1, 10)); // x1 = 10 (counter)
    let loop_start = code.len(); // words
    code.push(add_reg(0, 0, 1)); // loop: x0 += x1
    code.push(subs_imm(1, 1, 1)); // x1 -= 1 (sets flags)
    let cbnz_idx = code.len();
    code.push(0); // placeholder, patched below
    let back_off = ((loop_start as i64 - cbnz_idx as i64) * 4) as i32;
    code[cbnz_idx] = cbnz(1, back_off); // cbnz x1, loop
    // x0 == 55 (sum of 1..=10) here.

    code.extend(mov_imm32(2, 6));
    code.push(sub_reg(0, 0, 2)); // 55-6 = 49
    code.extend(mov_imm32(2, 2));
    code.push(mul(0, 0, 2)); // 49*2 = 98
    code.extend(mov_imm32(2, 7));
    code.push(udiv(0, 0, 2)); // 98/7 = 14
    code.extend(mov_imm32(2, 31));
    code.push(and_reg(0, 0, 2)); // 14 & 31 = 14
    code.extend(mov_imm32(2, 64));
    code.push(orr_reg(0, 0, 2)); // 14 | 64 = 78
    code.extend(mov_imm32(2, 8));
    code.push(eor_reg(0, 0, 2)); // 78 ^ 8 = 70
    code.extend(mov_imm32(2, 1));
    code.push(lslv(0, 0, 2)); // 70 << 1 = 140
    code.extend(mov_imm32(2, 3));
    code.push(lsrv(0, 0, 2)); // 140 >> 3 = 17

    code.extend(mov_imm32(8, NR_EXIT));
    code.push(svc0());

    let body = words_to_bytes(&code);
    let (exit_code, out, kernel) = run_program(vaddr, &body);

    assert_eq!(
        exit_code, 17,
        "loop must sum 1..=10 to 55, then the SUB/MUL/UDIV/AND/ORR/EOR/LSL/LSR chain must fold it to 17"
    );
    assert!(out.is_empty());
    assert!(kernel.unsupported().is_empty(), "{:?}", kernel.unsupported());
}

// ---- 2. bitfield extract + conditional select/increment/compare ------------
//
// `x1 = 0x1234_5678_9abc_def0` (the same pattern `sbfx_ubfx_extract_bitfield`
// uses in `src/vcpu/interp.rs`'s unit tests, so the extracted values are
// already known-good): `UBFX x0,x1,#4,#8` = 0xef = 239, `SBFX x2,x1,#4,#8` =
// -17. `CMP x0,#239` sets EQ (so it's really `SUBS` under the hood).
// `CCMP x2,#17,#0,eq`: since EQ holds, this performs a *real* compare of
// -17 (as an unsigned 64-bit value, i.e. huge) against 17, which sets N=1
// (result's top bit) among other flags. `CSEL ...,mi` (N==1) then picks its
// true operand (200); `CSINC ...,pl` (N==0, false here) picks its false
// operand incremented (10+1=11). `200-11=189` is the exit code.

#[test]
fn bitfield_extract_and_conditional_chain_exit_code() {
    let vaddr = 0x1_0000u64;

    let mut code = Vec::new();
    code.extend(mov_imm64(1, 0x1234_5678_9abc_def0));
    code.push(ubfx(0, 1, 4, 8)); // x0 = 0xef = 239
    code.push(sbfx(2, 1, 4, 8)); // x2 = -17 (sign-extended)
    code.push(subs_imm(31, 0, 239)); // cmp x0,#239 -> EQ holds (Z=1)
    code.push(ccmp_imm(2, 17, 0, COND_EQ)); // real compare: x2(-17 as huge u64) vs 17 -> N=1
    code.extend(mov_imm32(9, 200)); // x9 = 200 (csel/csinc "true" operand)
    code.extend(mov_imm32(10, 10)); // x10 = 10 (csel/csinc "false" operand)
    code.push(csel(3, 9, 10, COND_MI)); // N==1 holds -> x3 = x9 = 200
    code.push(csinc(4, 9, 10, COND_PL)); // N==0 fails -> x4 = x10+1 = 11
    code.push(sub_reg(0, 3, 4)); // x0 = 200-11 = 189

    code.extend(mov_imm32(8, NR_EXIT));
    code.push(svc0());

    let body = words_to_bytes(&code);
    let (exit_code, out, kernel) = run_program(vaddr, &body);

    assert_eq!(
        exit_code, 189,
        "UBFX/SBFX extraction feeding CMP/CCMP/CSEL/CSINC must fold to 189"
    );
    assert!(out.is_empty());
    assert!(kernel.unsupported().is_empty(), "{:?}", kernel.unsupported());
}

// ---- 3. scalar floating point -----------------------------------------------
//
// `sqrt(16.0) = 4.0`; `4.0 + 3.0 = 7.0`; `21.0 / 7.0 = 3.0`; `3.0 * 4.0 =
// 12.0`; `FCVTZS` truncates that back to the integer exit code 12. Exercises
// SCVTF, FSQRT, FADD, FDIV, FMUL, and FCVTZS (all double precision).

#[test]
fn scalar_fp_sqrt_chain_exit_code() {
    let vaddr = 0x1_0000u64;

    let mut code = Vec::new();
    code.extend(mov_imm32(1, 16));
    code.push(scvtf_d(0, 1)); // d0 = 16.0
    code.push(fsqrt_d(1, 0)); // d1 = sqrt(16.0) = 4.0
    code.extend(mov_imm32(2, 3));
    code.push(scvtf_d(2, 2)); // d2 = 3.0
    code.push(fadd_d(3, 1, 2)); // d3 = 4.0 + 3.0 = 7.0
    code.extend(mov_imm32(4, 21));
    code.push(scvtf_d(4, 4)); // d4 = 21.0
    code.push(fdiv_d(5, 4, 3)); // d5 = 21.0 / 7.0 = 3.0
    code.push(fmul_d(6, 5, 1)); // d6 = 3.0 * 4.0 = 12.0
    code.push(fcvtzs_xd(0, 6)); // x0 = 12

    code.extend(mov_imm32(8, NR_EXIT));
    code.push(svc0());

    let body = words_to_bytes(&code);
    let (exit_code, out, kernel) = run_program(vaddr, &body);

    assert_eq!(
        exit_code, 12,
        "sqrt(16)+3=7, 21/7=3, 3*4=12 must round-trip through SCVTF/FSQRT/FADD/FDIV/FMUL/FCVTZS"
    );
    assert!(out.is_empty());
    assert!(kernel.unsupported().is_empty(), "{:?}", kernel.unsupported());
}

// ---- 4. NEON vector build + arithmetic + across-lanes reduction ------------
//
// `DUP v0.4s,w1` / `DUP v1.4s,w2` build two all-5s / all-3s vectors;
// `MUL v2.4s,v0.4s,v1.4s` -> [15,15,15,15]; `INS v2.s[0],w3` overwrites lane
// 0 with 39 -> [39,15,15,15]; `MOVI v5.4s,#1` builds an all-1s vector and
// `ADD v2.4s,v2.4s,v5.4s` bumps every lane -> [40,16,16,16]; `ADDV s4,v2.4s`
// reduces across lanes to 40+16+16+16=88; `UMOV w0,v4.s[0]` moves that back
// to a GP register as the exit code.

#[test]
fn neon_build_and_reduce_exit_code() {
    let vaddr = 0x1_0000u64;

    let mut code = Vec::new();
    code.extend(mov_imm32(1, 5));
    code.push(dup_4s(0, 1)); // v0.4s = [5,5,5,5]
    code.extend(mov_imm32(2, 3));
    code.push(dup_4s(1, 2)); // v1.4s = [3,3,3,3]
    code.push(mul_4s(2, 0, 1)); // v2.4s = [15,15,15,15]
    code.extend(mov_imm32(3, 39));
    code.push(ins_s(2, 0, 3)); // v2.4s = [39,15,15,15]
    code.push(movi_4s(5, 1)); // v5.4s = [1,1,1,1]
    code.push(add_4s(2, 2, 5)); // v2.4s = [40,16,16,16]
    code.push(addv_4s(4, 2)); // s4 = 40+16+16+16 = 88
    code.push(umov_s(0, 4, 0)); // x0 = 88

    code.extend(mov_imm32(8, NR_EXIT));
    code.push(svc0());

    let body = words_to_bytes(&code);
    let (exit_code, out, kernel) = run_program(vaddr, &body);

    assert_eq!(
        exit_code, 88,
        "DUP/MUL/INS/MOVI/ADD (vector) reduced via ADDV/UMOV must yield 88"
    );
    assert!(out.is_empty());
    assert!(kernel.unsupported().is_empty(), "{:?}", kernel.unsupported());
}

// ---- 5. LSE atomics: LDADD then CAS on a stack/code-segment word ------------
//
// A scratch word trailing the code (inside the same RWX `PT_LOAD`, like
// `tests/aarch64_smoke.rs`'s buffers) starts at 10. `LDADD w1,w2,[x0]` with
// `w1=5` returns the old value (10) in `w2` and leaves 15 in memory.
// `CAS w4,w5,[x0]` with compare value `w4=15` (matching) and new value
// `w5=77` swaps 77 into memory (and returns the pre-swap 15 into `w4`).
// `LDR w6,[x0]` reloads memory to confirm the swap really happened, and that
// value (77) becomes the exit code.

#[test]
fn lse_atomics_ldadd_then_cas_exit_code() {
    fn build(word_addr: u32) -> Vec<u32> {
        let mut code = Vec::new();
        code.extend(mov_addr(0, word_addr)); // x0 = &word (2 words, fixed length)
        code.extend(mov_imm32(1, 5)); // x1 = 5 (ldadd's rs)
        code.push(ldadd_w(1, 2, 0)); // w2 = old (10); mem = 10+5 = 15
        code.extend(mov_imm32(4, 15)); // x4 = 15 (cas compare value, matches mem)
        code.extend(mov_imm32(5, 77)); // x5 = 77 (cas new value)
        code.push(cas_w(4, 5, 0)); // mem == 15 -> swap: mem = 77
        code.push(ldr_w(6, 0, 0)); // w6 = mem (confirm the swap)
        code.push(mov_reg(0, 6)); // x0 = 77
        code.extend(mov_imm32(8, NR_EXIT));
        code.push(svc0());
        code
    }

    let vaddr = 0x1_0000u64;
    let code_words = build(0).len() as u64;
    let word_addr = u32::try_from(vaddr + BODY_OFF + code_words * 4).unwrap();
    let instrs = build(word_addr);
    assert_eq!(instrs.len() as u64, code_words, "two-pass build must be length-stable");

    let mut body = words_to_bytes(&instrs);
    body.extend_from_slice(&10u32.to_le_bytes()); // scratch word, starts at 10

    let (exit_code, out, kernel) = run_program(vaddr, &body);

    assert_eq!(
        exit_code, 77,
        "LDADD(10,+5)=15 then CAS(==15 -> 77) must leave 77 in memory"
    );
    assert!(out.is_empty());
    assert!(kernel.unsupported().is_empty(), "{:?}", kernel.unsupported());
}

// ---- 6. mmap'd page: store a multi-byte pattern, load it back at three ------
//        widths, fold with EOR
//
// `mmap(NULL,PAGE_SIZE,PROT_RW,MAP_ANONYMOUS,...)` gives a fresh page (needs
// the explicit host-side mmap arena from `Kernel::set_mmap_area`, mirroring
// `src/bin/run-elf.rs`). Eight `STRB`s write the bytes `0x10..=0x17` at
// offsets `0..=7`. `LDR x1,[base]` reads all 8 bytes back as one 64-bit
// word, `LDRH w2,[base,#2]` reads a 16-bit slice, `LDRB w3,[base,#5]` reads
// a single byte — three different load widths over the same bytes. Their low
// bytes are `0x10`, `0x12`, and `0x15` respectively; `EOR`ing all three
// together (`0x10 ^ 0x12 ^ 0x15 = 0x17 = 23`) is only right if every store
// landed at the right offset *and* every load width sliced the right bytes.

#[test]
fn mmap_pattern_multi_width_load_exit_code() {
    const MAP_ANONYMOUS: u32 = 0x20;
    const PROT_RW: u32 = 0x3; // PROT_READ | PROT_WRITE
    let vaddr = 0x1_0000u64;

    let mut code = Vec::new();
    code.extend(mov_imm32(0, 0)); // addr = NULL (let the kernel place it)
    code.extend(mov_imm32(1, PAGE_SIZE as u32)); // len = one page
    code.extend(mov_imm32(2, PROT_RW));
    code.extend(mov_imm32(3, MAP_ANONYMOUS));
    code.extend(mov_imm32(4, 0)); // fd (unused: anonymous mapping)
    code.extend(mov_imm32(5, 0)); // offset
    code.extend(mov_imm32(8, NR_MMAP));
    code.push(svc0()); // x0 = mapped base address
    code.push(mov_reg(9, 0)); // x9 = mapped base (survives further syscalls/ops)

    for off in 0u32..8 {
        code.extend(mov_imm32(2, 0x10 + off));
        code.push(strb(2, 9, off));
    }

    code.push(ldr_x(1, 9, 0)); // x1 = all 8 bytes, offset 0 (imm12 unit = 8 bytes)
    code.push(ldrh(2, 9, 1)); // x2 = halfword at byte offset 2 (imm12 unit = 2 bytes)
    code.push(ldrb(3, 9, 5)); // x3 = byte at offset 5
    code.push(eor_reg(4, 1, 2));
    code.push(eor_reg(0, 4, 3)); // x0 = x1 ^ x2 ^ x3 (low byte: 0x10^0x12^0x15=0x17=23)

    code.extend(mov_imm32(8, NR_EXIT));
    code.push(svc0());

    let body = words_to_bytes(&code);
    let elf = build_elf(vaddr, &body);

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
    // mmap needs an explicit arena (host-side setup), mirroring
    // `src/bin/run-elf.rs`'s `kernel.set_mmap_area` call after `load_static`.
    kernel.set_mmap_area(img.stack_bottom, img.program_break);

    let exit_code = kernel.run(vcpu, mem).unwrap();

    assert_eq!(
        exit_code, 23,
        "LDR x/LDRH/LDRB reading the same stored bytes at three widths, EORed, must yield 23"
    );
    assert!(captured.lock().unwrap().is_empty());
    assert!(kernel.unsupported().is_empty(), "{:?}", kernel.unsupported());
}
