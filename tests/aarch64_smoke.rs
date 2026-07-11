//! End-to-end aarch64 smoke tests: minimal static ELF64 images are hand
//! assembled, loaded by the real loader (`nixvm::loader::load_static`), then
//! executed on the software interpreter (`InterpBackend::new(Arch::Aarch64)`,
//! mirroring `src/bin/run-elf.rs`'s wiring) with a captured stdout sink,
//! exercising loader + `interp` + `Kernel` together for the aarch64 guest
//! path. Analogous to `tests/x86_smoke.rs` for the x86-64 path.
//!
//! Instructions are hand-encoded via small formula helpers below (rather than
//! copied hex literals) so each program's bytes are derived, not guessed; the
//! encodings are cross-checked against known-good words already exercised by
//! `tests/hello_elf.rs` / `tests/hello_interp.rs` / `tests/mm_brk.rs` /
//! `tests/mm_mmap.rs` (e.g. `movz x8,#93` / `svc #0` / `mov x19,x0`) where
//! they overlap.

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

// ---- instruction encoders (A64) --------------------------------------------
//
// Each helper builds one 32-bit instruction word from a formula, not a copied
// literal, so the constants below (register numbers, syscall numbers) are the
// only thing a reader has to trust. `rd`/`rn`/`rm`/`rt` are `0..=30` for
// `x0..=x30` (31 means `xzr`, per the ISA, in the fields these helpers use).

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
/// (`MOVZ`, plus `MOVK ,LSL #16` if the high half is nonzero). Every guest
/// address used by these tests fits in 32 bits, so `MOVZ`/`MOVK` at shift 0/16
/// is all that's needed (no shift 32/48 forms).
fn mov_imm32(rd: u32, val: u32) -> Vec<u32> {
    let lo = val & 0xffff;
    let hi = (val >> 16) & 0xffff;
    let mut words = vec![movz(rd, lo)];
    if hi != 0 {
        words.push(movk16(rd, hi));
    }
    words
}

/// `STRB Wt, [Xn, #imm12]` (unsigned immediate offset, byte store): writes the
/// low byte of `Xt` to `[Xn + imm12]`.
fn strb(rt: u32, rn: u32, imm12: u32) -> u32 {
    0x3900_0000 | (imm12 << 10) | (rn << 5) | rt
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

fn round_up(v: u64, align: u64) -> u64 {
    v.div_ceil(align) * align
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
/// start of `body`. Mirrors `tests/x86_smoke.rs`'s `build_elf`.
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

/// Load `body` at `vaddr` and run it to completion on the software
/// interpreter, capturing stdout. Returns `(exit_code, stdout, kernel)` so
/// callers can also inspect the unsupported-syscall ledger.
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

fn words_to_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for w in words {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    bytes
}

// ---- 1. write + exit --------------------------------------------------------

#[test]
fn write_and_exit() {
    // x0=1(1 word) ; x1=msg_addr(2 words) ; x2=3(1) ; x8=64(1) ; svc(1) ;
    // x0=0(1) ; x8=93(1) ; svc(1)
    const CODE_WORDS: u64 = 9;
    let vaddr = 0x1_0000u64;
    let msg_addr = vaddr + BODY_OFF + CODE_WORDS * 4;

    let mut code = Vec::new();
    code.extend(mov_imm32(0, 1)); // x0 = 1 (fd = stdout)
    let x1 = mov_imm32(1, u32::try_from(msg_addr).unwrap());
    assert_eq!(
        x1.len(),
        2,
        "msg_addr's high half must be nonzero for this vaddr"
    );
    code.extend(x1);
    code.extend(mov_imm32(2, 3)); // x2 = 3 (len)
    code.extend(mov_imm32(8, 64)); // x8 = __NR_write
    code.push(svc0());
    code.extend(mov_imm32(0, 0)); // x0 = 0 (exit status)
    code.extend(mov_imm32(8, 93)); // x8 = __NR_exit
    code.push(svc0());
    assert_eq!(
        code.len() as u64,
        CODE_WORDS,
        "CODE_WORDS must match the assembled program"
    );

    let mut body = words_to_bytes(&code);
    body.extend_from_slice(b"ok\n");

    let (code, out, kernel) = run_program(vaddr, &body);
    assert_eq!(code, 0);
    assert_eq!(&out, b"ok\n");
    assert!(
        kernel.unsupported().is_empty(),
        "{:?}",
        kernel.unsupported()
    );
}

// ---- 2. arithmetic -> exit code ---------------------------------------------

#[test]
fn arithmetic_into_exit_code() {
    let vaddr = 0x1_0000u64;

    let mut code = Vec::new();
    code.extend(mov_imm32(0, 7)); // x0 = 7
    code.extend(mov_imm32(1, 6)); // x1 = 6
    code.push(mul(0, 0, 1)); // x0 = x0 * x1 = 42
    code.extend(mov_imm32(8, 93)); // x8 = __NR_exit
    code.push(svc0());

    let body = words_to_bytes(&code);
    let (code, out, kernel) = run_program(vaddr, &body);

    assert_eq!(code, 42, "7*6 should exit with status 42");
    assert!(out.is_empty());
    assert!(
        kernel.unsupported().is_empty(),
        "{:?}",
        kernel.unsupported()
    );
}

// ---- 3. getpid / gettid ------------------------------------------------------

#[test]
fn getpid_and_gettid_reflect_pid_one() {
    let vaddr = 0x1_0000u64;

    let mut code = Vec::new();
    code.extend(mov_imm32(8, 172)); // __NR_getpid
    code.push(svc0()); // x0 = tgid (1 for the run() root process)
    code.push(mov_reg(19, 0)); // x19 = x0
    code.extend(mov_imm32(8, 178)); // __NR_gettid
    code.push(svc0()); // x0 = tid (1 for the run() root process)
    code.push(add_reg(0, 0, 19)); // x0 = gettid() + getpid() = 2
    code.extend(mov_imm32(8, 93)); // __NR_exit
    code.push(svc0());

    let body = words_to_bytes(&code);
    let (code, out, kernel) = run_program(vaddr, &body);

    assert_eq!(
        code, 2,
        "getpid() (tgid=1) + gettid() (pid=1) should exit with status 2"
    );
    assert!(out.is_empty());
    assert!(
        kernel.unsupported().is_empty(),
        "{:?}",
        kernel.unsupported()
    );
}

// ---- 4. write from a computed buffer (loads/stores through guest memory) ---

#[test]
fn write_from_computed_buffer() {
    // x1=buf_addr(2 words) + 3x(movz+strb) + x0=1(1) + x2=3(1) + x8=64(1) +
    // svc(1) + x0=0(1) + x8=93(1) + svc(1) = 2+6+7 = 15
    const CODE_WORDS: u64 = 15;
    let vaddr = 0x1_0000u64;
    let buf_addr = vaddr + BODY_OFF + CODE_WORDS * 4;

    let mut code = Vec::new();
    let x1 = mov_imm32(1, u32::try_from(buf_addr).unwrap());
    assert_eq!(
        x1.len(),
        2,
        "buf_addr's high half must be nonzero for this vaddr"
    );
    code.extend(x1);
    code.push(movz(2, 0x6f)); // x2 = 'o'
    code.push(strb(2, 1, 0)); // [x1+0] = 'o'
    code.push(movz(2, 0x6b)); // x2 = 'k'
    code.push(strb(2, 1, 1)); // [x1+1] = 'k'
    code.push(movz(2, 0x0a)); // x2 = '\n'
    code.push(strb(2, 1, 2)); // [x1+2] = '\n'
    code.extend(mov_imm32(0, 1)); // x0 = 1 (fd = stdout)
    code.extend(mov_imm32(2, 3)); // x2 = 3 (len)
    code.extend(mov_imm32(8, 64)); // x8 = __NR_write
    code.push(svc0());
    code.extend(mov_imm32(0, 0)); // x0 = 0 (exit status)
    code.extend(mov_imm32(8, 93)); // x8 = __NR_exit
    code.push(svc0());
    assert_eq!(
        code.len() as u64,
        CODE_WORDS,
        "CODE_WORDS must match the assembled program"
    );

    let mut body = words_to_bytes(&code);
    body.extend_from_slice(&[0, 0, 0]); // scratch buffer, filled by STRB at runtime

    let (code, out, kernel) = run_program(vaddr, &body);
    assert_eq!(code, 0);
    assert_eq!(
        &out, b"ok\n",
        "bytes stored via STRB should round-trip through write()"
    );
    assert!(
        kernel.unsupported().is_empty(),
        "{:?}",
        kernel.unsupported()
    );
}

// ---- 5. brk + write (memory syscalls) ---------------------------------------

#[test]
fn brk_then_write() {
    // Code length is fixed (nothing trails it), so `program_break` can be
    // predicted with the same `round_up` the loader uses, then baked into the
    // guest code as constants ahead of assembling it.
    const CODE_WORDS: u64 = 15;
    let vaddr = 0x1_0000u64;
    let program_break = round_up(vaddr + BODY_OFF + CODE_WORDS * 4, PAGE_SIZE);
    let brk_target = program_break + PAGE_SIZE;
    let buf_addr = program_break; // first byte of the freshly brk'd page

    let mut code = Vec::new();
    code.extend(mov_imm32(0, u32::try_from(brk_target).unwrap())); // x0 = new break
    code.extend(mov_imm32(8, 214)); // __NR_brk
    code.push(svc0());
    code.extend(mov_imm32(1, u32::try_from(buf_addr).unwrap())); // x1 = buf
    code.push(movz(2, 0x42)); // x2 = 'B'
    code.push(strb(2, 1, 0)); // [x1] = 'B'
    code.extend(mov_imm32(0, 1)); // x0 = 1 (fd)
    code.extend(mov_imm32(2, 1)); // x2 = 1 (len)
    code.extend(mov_imm32(8, 64)); // __NR_write
    code.push(svc0());
    code.extend(mov_imm32(0, 0));
    code.extend(mov_imm32(8, 93)); // __NR_exit
    code.push(svc0());
    assert_eq!(
        code.len() as u64,
        CODE_WORDS,
        "CODE_WORDS must match the assembled program"
    );

    let body = words_to_bytes(&code);
    let elf = build_elf(vaddr, &body);

    let mut mem = GuestMemory::new(vaddr, 256 * PAGE_SIZE);
    let spec = ProcessSpec {
        argv: vec!["prog".into()],
        envp: vec![],
    };
    let img = load_static(&mut mem, &elf, &spec).unwrap();
    assert_eq!(
        img.program_break, program_break,
        "predicted program_break must match the loader's"
    );

    let backend = InterpBackend::new(Arch::Aarch64).unwrap();
    let vcpu = backend.new_vcpu(img.entry, img.stack_pointer).unwrap();

    let captured = Arc::new(Mutex::new(Vec::new()));
    let mut kernel = Kernel::new(Arch::Aarch64, MountTable::new());
    kernel.set_stdout(Box::new(SharedBuf(captured.clone())));
    // `brk` needs an explicit heap window (host-side setup, mirroring
    // `src/bin/run-elf.rs`'s `kernel.set_heap` call after `load_static`).
    kernel.set_heap(img.program_break, img.program_break + 16 * PAGE_SIZE);

    let code = kernel.run(vcpu, mem).unwrap();

    assert_eq!(code, 0);
    assert_eq!(
        &*captured.lock().unwrap(),
        b"B",
        "byte stored into the brk-grown page should round-trip through write()"
    );
    assert!(
        kernel.unsupported().is_empty(),
        "{:?}",
        kernel.unsupported()
    );
}
