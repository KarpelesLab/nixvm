//! End-to-end aarch64 process/IPC/memory integration tests: multi-process
//! (`clone` + `wait4`), pipe IPC (`pipe2` + `read`/`write`), the two combined
//! (fork then talk over a pipe), and anonymous `mmap`. Each program is hand
//! assembled, loaded by the real loader (`nixvm::loader::load_static`), then
//! executed on the software interpreter (`InterpBackend::new(Arch::Aarch64)`)
//! with a captured stdout sink — the same loader + `interp` + `Kernel` wiring
//! as `tests/aarch64_smoke.rs`, extended to multi-task control flow.
//!
//! This file is self-contained (its own ELF builder + instruction encoders),
//! deliberately duplicating the small helpers already in `tests/aarch64_smoke.rs`
//! rather than sharing a module — integration test binaries in `tests/` don't
//! share code cleanly, and duplicating a few dozen lines is simpler and more
//! robust than wiring up a `tests/common/` module.
//!
//! ## A note on assembling programs with runtime-computed addresses
//!
//! Several programs here need to bake a *guest address* (of a scratch buffer
//! trailing the code) into the code itself, but that address depends on the
//! code's own length. Two tricks make this tractable without hand-counting
//! instruction words:
//!
//!   1. [`mov_addr`] always emits exactly two instructions (`MOVZ` + `MOVK`),
//!      regardless of the value — unlike [`mov_imm32`], which emits one
//!      instruction when the high half happens to be zero. So a program's
//!      instruction *count* never depends on the numeric value of any address
//!      loaded via `mov_addr`.
//!   2. Programs needing such an address are built by a local `fn build(..)`
//!      taking the address(es) as parameters. It is called once with a
//!      placeholder (`0`) to measure the resulting code length (and hence
//!      compute the real address, which lands right after the code), then
//!      called again with the real address to produce the final program. Both
//!      calls produce the same length by (1), so this is exact, not a guess.
//!
//! Forward branches (`CBZ`) to a not-yet-laid-out block are resolved the same
//! way: the branch target's position falls out of the lengths of the
//! instruction vectors placed between it and the branch, computed in Rust
//! before encoding the branch word — no manual word-counting.

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
const NR_READ: u32 = 63;
const NR_WRITE: u32 = 64;
const NR_PIPE2: u32 = 59;
const NR_CLONE: u32 = 220;
const NR_EXIT: u32 = 93;
const NR_WAIT4: u32 = 260;
const NR_MMAP: u32 = 222;

// ---- instruction encoders (A64) --------------------------------------------
//
// Each helper builds one 32-bit instruction word from a formula (cross-checked
// against the encodings already exercised by `tests/aarch64_smoke.rs`), not a
// copied literal. `rd`/`rn`/`rm`/`rt` are `0..=30` for `x0..=x30` throughout.

/// `MOVZ Xd, #imm16` (64-bit, shift 0): loads `imm16` into the low 16 bits,
/// zeroing the rest.
fn movz(rd: u32, imm16: u32) -> u32 {
    0xD280_0000 | (imm16 << 5) | rd
}

/// `MOVK Xd, #imm16, LSL #16` (64-bit): merges `imm16` into bits `[31:16]`,
/// leaving the rest of `Xd` untouched.
fn movk16(rd: u32, imm16: u32) -> u32 {
    0xF2A0_0000 | (imm16 << 5) | rd
}

/// `SVC #0`: trap to the kernel (syscall number in `x8`, args in `x0..x5`).
fn svc0() -> u32 {
    0xD400_0001
}

/// Materialize a 32-bit-range immediate into `Xd` as one or two instructions
/// (`MOVZ`, plus `MOVK ,LSL #16` if the high half is nonzero). Only used here
/// for values known at Rust compile time (syscall numbers, flags, small
/// constants), so the emitted instruction count is always statically known to
/// the caller.
fn mov_imm32(rd: u32, val: u32) -> Vec<u32> {
    let lo = val & 0xffff;
    let hi = (val >> 16) & 0xffff;
    let mut words = vec![movz(rd, lo)];
    if hi != 0 {
        words.push(movk16(rd, hi));
    }
    words
}

/// Materialize a value into `Xd` as exactly two instructions (`MOVZ` +
/// `MOVK`), even when the high half is zero. Used for guest addresses whose
/// numeric value isn't known until the surrounding code's length is (see the
/// module doc's two-pass build pattern) — the fixed two-word cost keeps a
/// program's total length independent of the address's actual value.
fn mov_addr(rd: u32, val: u32) -> [u32; 2] {
    [movz(rd, val & 0xffff), movk16(rd, (val >> 16) & 0xffff)]
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

/// `LDR Wt, [Xn, #(imm12*4)]` (unsigned immediate offset, 32-bit load,
/// zero-extended into `Xt`).
fn ldr_w(rt: u32, rn: u32, imm12: u32) -> u32 {
    0xB940_0000 | (imm12 << 10) | (rn << 5) | rt
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
/// the headers + `body` (code followed by any trailing data), entry at the
/// start of `body`. Mirrors `tests/aarch64_smoke.rs`'s `build_elf`.
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
/// interpreter (single vCPU / `ncpus == 1`), capturing stdout. Returns
/// `(exit_code, stdout, kernel)` so callers can also inspect the
/// unsupported-syscall ledger.
fn run_program(vaddr: u64, body: &[u8]) -> (i32, Vec<u8>, Kernel) {
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

    let exit_code = kernel.run(vcpu, mem).unwrap();
    let out = captured.lock().unwrap().clone();
    (exit_code, out, kernel)
}

// ---- 1. fork + exit code propagation ----------------------------------------
//
// Program: `clone(0, 0)`; the child exits with a fixed code, the parent
// `wait4`s it and re-exits with the same code (extracted from the `wstatus`
// word `wait4` fills in — `WIFEXITED`'s encoding is `(code & 0xff) << 8`, so
// the exit code is byte 1 of that word).

#[test]
fn fork_propagates_child_exit_code() {
    const CHILD_EXIT_CODE: u32 = 55;

    fn build(wstatus_addr: u32) -> Vec<u32> {
        let mut code = Vec::new();
        code.extend(mov_imm32(0, 0)); // x0 = 0 (clone flags)
        code.extend(mov_imm32(1, 0)); // x1 = 0 (clone stack: none, share parent's)
        code.extend(mov_imm32(8, NR_CLONE));
        code.push(svc0()); // x0 = child pid (parent) or 0 (child)

        // Parent path: x0 (still the clone() result) is already wait4's pid
        // arg (its actual value doesn't matter to this kernel's `wait4`,
        // which reaps *a* zombie child regardless — see `sys_wait4`).
        let mut parent = Vec::new();
        parent.extend(mov_addr(1, wstatus_addr)); // x1 = &wstatus
        parent.extend(mov_imm32(2, 0)); // x2 = 0 (options)
        parent.extend(mov_imm32(8, NR_WAIT4));
        parent.push(svc0());
        parent.push(ldrb(0, 1, 1)); // x0 = wstatus byte 1 = child's exit code
        parent.extend(mov_imm32(8, NR_EXIT));
        parent.push(svc0());

        let mut child = Vec::new();
        child.extend(mov_imm32(0, CHILD_EXIT_CODE));
        child.extend(mov_imm32(8, NR_EXIT));
        child.push(svc0());

        // If x0 == 0 (child), skip over `parent` straight to `child`.
        let cbz_off_bytes = ((1 + parent.len()) * 4) as i32;
        code.push(cbz(0, cbz_off_bytes));
        code.extend(parent);
        code.extend(child);
        code
    }

    let vaddr = 0x1_0000u64;
    let code_words = build(0).len() as u64;
    let wstatus_addr = u32::try_from(vaddr + BODY_OFF + code_words * 4).unwrap();
    let instrs = build(wstatus_addr);
    assert_eq!(instrs.len() as u64, code_words, "two-pass build must be length-stable");

    let mut body = words_to_bytes(&instrs);
    body.extend_from_slice(&[0, 0, 0, 0]); // wstatus scratch word, filled by wait4

    let (exit_code, out, kernel) = run_program(vaddr, &body);
    assert_eq!(exit_code, CHILD_EXIT_CODE as i32, "parent's exit code must be the child's, via wait4");
    assert!(out.is_empty());
    assert!(kernel.unsupported().is_empty(), "{:?}", kernel.unsupported());
}

// ---- 2. pipe IPC -------------------------------------------------------------
//
// Program: `pipe2(&fds, 0)`; write a message to the write end; read it back
// from the read end; write the bytes read back to stdout. Single task — pure
// IPC plumbing, no `clone`.

#[test]
fn pipe_write_read_roundtrips_to_stdout() {
    const MSG: &[u8] = b"hi!\n";
    const MSG_LEN: u32 = MSG.len() as u32;

    fn build(fds_addr: u32, msg_addr: u32, rbuf_addr: u32) -> Vec<u32> {
        let mut code = Vec::new();
        // pipe2(&fds, 0) -- x9 keeps &fds alive across the syscall (which
        // clobbers x0 with the return value).
        code.extend(mov_addr(9, fds_addr));
        code.push(mov_reg(0, 9));
        code.extend(mov_imm32(1, 0)); // flags = 0
        code.extend(mov_imm32(8, NR_PIPE2));
        code.push(svc0());
        code.push(ldr_w(19, 9, 0)); // x19 = fds[0] (read fd)
        code.push(ldr_w(20, 9, 1)); // x20 = fds[1] (write fd)

        // write(writefd, &msg, MSG_LEN)
        code.push(mov_reg(0, 20));
        code.extend(mov_addr(1, msg_addr));
        code.extend(mov_imm32(2, MSG_LEN));
        code.extend(mov_imm32(8, NR_WRITE));
        code.push(svc0());

        // read(readfd, &rbuf, MSG_LEN)
        code.push(mov_reg(0, 19));
        code.extend(mov_addr(1, rbuf_addr));
        code.extend(mov_imm32(2, MSG_LEN));
        code.extend(mov_imm32(8, NR_READ));
        code.push(svc0());

        // write(stdout, &rbuf, MSG_LEN) -- echo what came back through the pipe
        code.extend(mov_imm32(0, 1));
        code.extend(mov_addr(1, rbuf_addr));
        code.extend(mov_imm32(2, MSG_LEN));
        code.extend(mov_imm32(8, NR_WRITE));
        code.push(svc0());

        code.extend(mov_imm32(0, 0));
        code.extend(mov_imm32(8, NR_EXIT));
        code.push(svc0());
        code
    }

    let vaddr = 0x1_0000u64;
    let code_words = build(0, 0, 0).len() as u64;
    let fds_addr = u32::try_from(vaddr + BODY_OFF + code_words * 4).unwrap();
    let msg_addr = fds_addr + 8; // past the 8-byte `int[2]` fds buffer
    let rbuf_addr = msg_addr + MSG_LEN;
    let instrs = build(fds_addr, msg_addr, rbuf_addr);
    assert_eq!(instrs.len() as u64, code_words);

    let mut body = words_to_bytes(&instrs);
    body.extend_from_slice(&[0u8; 8]); // fds[2] scratch, filled by pipe2
    body.extend_from_slice(MSG);
    body.extend(std::iter::repeat_n(0u8, MSG_LEN as usize)); // rbuf scratch

    let (exit_code, out, kernel) = run_program(vaddr, &body);
    assert_eq!(exit_code, 0);
    assert_eq!(&out, MSG, "bytes written into the pipe should read back unchanged and echo to stdout");
    assert!(kernel.unsupported().is_empty(), "{:?}", kernel.unsupported());
}

// ---- 3. fork + pipe -----------------------------------------------------------
//
// Program: create a pipe, then `clone(0, 0)`. The child writes a message to
// the write end and exits; the parent reads it from the read end, echoes it
// to stdout, `wait4`s the child, and exits 0. `read_pipe` (see
// `src/kernel/mod.rs`) blocks cooperatively on an empty pipe with an open
// writer, so this is correct regardless of whether the scheduler happens to
// run the child's write or the parent's read first.

#[test]
fn fork_pipe_parent_reads_child_writes_and_exits() {
    const MSG: &[u8] = b"ok\n";
    const MSG_LEN: u32 = MSG.len() as u32;

    fn build(fds_addr: u32, msg_addr: u32, rbuf_addr: u32) -> Vec<u32> {
        let mut code = Vec::new();
        // pipe2(&fds, 0)
        code.extend(mov_addr(9, fds_addr));
        code.push(mov_reg(0, 9));
        code.extend(mov_imm32(1, 0));
        code.extend(mov_imm32(8, NR_PIPE2));
        code.push(svc0());
        code.push(ldr_w(19, 9, 0)); // x19 = readfd
        code.push(ldr_w(20, 9, 1)); // x20 = writefd

        // clone(0, 0)
        code.extend(mov_imm32(0, 0));
        code.extend(mov_imm32(1, 0));
        code.extend(mov_imm32(8, NR_CLONE));
        code.push(svc0()); // x0 = child pid (parent) or 0 (child)

        let mut parent = Vec::new();
        parent.push(mov_reg(21, 0)); // x21 = child pid (survives the read/write below)
        parent.push(mov_reg(0, 19)); // read(readfd, &rbuf, MSG_LEN)
        parent.extend(mov_addr(1, rbuf_addr));
        parent.extend(mov_imm32(2, MSG_LEN));
        parent.extend(mov_imm32(8, NR_READ));
        parent.push(svc0());
        parent.extend(mov_imm32(0, 1)); // write(stdout, &rbuf, MSG_LEN)
        parent.extend(mov_addr(1, rbuf_addr));
        parent.extend(mov_imm32(2, MSG_LEN));
        parent.extend(mov_imm32(8, NR_WRITE));
        parent.push(svc0());
        parent.push(mov_reg(0, 21)); // wait4(child pid, NULL, 0)
        parent.extend(mov_imm32(1, 0));
        parent.extend(mov_imm32(2, 0));
        parent.extend(mov_imm32(8, NR_WAIT4));
        parent.push(svc0());
        parent.extend(mov_imm32(0, 0));
        parent.extend(mov_imm32(8, NR_EXIT));
        parent.push(svc0());

        let mut child = Vec::new();
        child.push(mov_reg(0, 20)); // write(writefd, &msg, MSG_LEN)
        child.extend(mov_addr(1, msg_addr));
        child.extend(mov_imm32(2, MSG_LEN));
        child.extend(mov_imm32(8, NR_WRITE));
        child.push(svc0());
        child.extend(mov_imm32(0, 0));
        child.extend(mov_imm32(8, NR_EXIT));
        child.push(svc0());

        let cbz_off_bytes = ((1 + parent.len()) * 4) as i32;
        code.push(cbz(0, cbz_off_bytes));
        code.extend(parent);
        code.extend(child);
        code
    }

    let vaddr = 0x1_0000u64;
    let code_words = build(0, 0, 0).len() as u64;
    let fds_addr = u32::try_from(vaddr + BODY_OFF + code_words * 4).unwrap();
    let msg_addr = fds_addr + 8;
    let rbuf_addr = msg_addr + MSG_LEN;
    let instrs = build(fds_addr, msg_addr, rbuf_addr);
    assert_eq!(instrs.len() as u64, code_words);

    let mut body = words_to_bytes(&instrs);
    body.extend_from_slice(&[0u8; 8]);
    body.extend_from_slice(MSG);
    body.extend(std::iter::repeat_n(0u8, MSG_LEN as usize));

    let (exit_code, out, kernel) = run_program(vaddr, &body);
    assert_eq!(exit_code, 0);
    assert_eq!(&out, MSG, "the parent should read exactly what the child wrote into the pipe");
    assert!(kernel.unsupported().is_empty(), "{:?}", kernel.unsupported());
}

// ---- 4. mmap anonymous --------------------------------------------------------
//
// Program: `mmap(NULL, PAGE_SIZE, PROT_READ|PROT_WRITE, MAP_ANONYMOUS, -1, 0)`,
// store a byte into the mapping, load it back, exit with that byte. Needs an
// explicit mmap arena (host-side setup via `Kernel::set_mmap_area`, mirroring
// `src/bin/run-elf.rs`), so this test doesn't use the shared `run_program`.

#[test]
fn mmap_anonymous_store_load_roundtrip() {
    const MAP_ANONYMOUS: u32 = 0x20;
    const PROT_RW: u32 = 0x3; // PROT_READ | PROT_WRITE
    const VALUE: u32 = 0x2A; // 42
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
    code.push(mov_reg(9, 0)); // x9 = mapped base (survives further syscalls)
    code.push(movz(2, VALUE));
    code.push(strb(2, 9, 0)); // [x9] = VALUE
    code.push(ldrb(1, 9, 0)); // x1 = [x9] (round-trip through the mapping)
    code.push(mov_reg(0, 1));
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

    assert_eq!(exit_code, VALUE as i32, "byte stored into the mmap'd page should round-trip through a load");
    assert!(captured.lock().unwrap().is_empty());
    assert!(kernel.unsupported().is_empty(), "{:?}", kernel.unsupported());
}
