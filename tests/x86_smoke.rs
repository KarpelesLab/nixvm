//! End-to-end x86-64 smoke test: a minimal *static ELF64* is loaded by the
//! loader, then executed on the x86-64 software interpreter; its `write`
//! reaches a captured sink and `exit` returns the status. Exercises loader +
//! `interp_x86` + kernel together for the x86-64 guest path, mirroring
//! `tests/hello_elf.rs` (the aarch64 equivalent).

use std::io::Write;
use std::sync::{Arc, Mutex};

use nixvm::abi::Arch;
use nixvm::fs::MountTable;
use nixvm::kernel::Kernel;
use nixvm::loader::{ProcessSpec, load_static};
use nixvm::vcpu::GuestMemory;
use nixvm::vcpu::mem::PAGE_SIZE;

const EHDR_LEN: usize = 64;
const PHDR_LEN: usize = 56;
const EM_X86_64: u16 = 62;

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

/// Assemble a minimal ET_EXEC x86-64 ELF: one RWX PT_LOAD at `vaddr` covering
/// the headers + `body` (code followed by data), entry at the start of `body`.
fn build_elf(vaddr: u64, body: &[u8]) -> Vec<u8> {
    let mut f = vec![0u8; EHDR_LEN + PHDR_LEN];
    f[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
    f[4] = 2; // ELFCLASS64
    f[5] = 1; // ELFDATA2LSB
    f[6] = 1;
    f[16..18].copy_from_slice(&2u16.to_le_bytes()); // ET_EXEC
    f[18..20].copy_from_slice(&EM_X86_64.to_le_bytes());
    f[20..24].copy_from_slice(&1u32.to_le_bytes());
    let body_off = (EHDR_LEN + PHDR_LEN) as u64;
    f[24..32].copy_from_slice(&(vaddr + body_off).to_le_bytes()); // e_entry
    f[32..40].copy_from_slice(&(EHDR_LEN as u64).to_le_bytes()); // e_phoff
    f[52..54].copy_from_slice(&(EHDR_LEN as u16).to_le_bytes());
    f[54..56].copy_from_slice(&(PHDR_LEN as u16).to_le_bytes());
    f[56..58].copy_from_slice(&1u16.to_le_bytes()); // e_phnum

    let total = body_off + body.len() as u64;
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

/// Length in bytes of the assembled program below (checked by an assertion
/// in the test, since a hand-tweak to the instructions must keep this in
/// sync with where the "ok\n" source buffer lands).
const PROG_LEN: u64 = 31;

#[test]
fn static_x86_64_elf_prints_and_exits() {
    let vaddr = 0x1_0000u64;
    let body_off = (EHDR_LEN + PHDR_LEN) as u64;

    // Instructions known-good in `interp_x86`'s own unit tests (`mov r32,
    // imm32`, `xor r32,r32`, `syscall`) — kept to that minimal, proven subset
    // so this smoke test is robust to interpreter coverage gaps.
    //
    //   mov edi, 1            ; fd = stdout
    //   mov esi, <msg_addr>   ; buf (zero-extends to 64 bits)
    //   mov edx, 3            ; len
    //   mov eax, 1            ; SYS_write
    //   syscall
    //   xor edi, edi          ; status = 0
    //   mov eax, 60           ; SYS_exit
    //   syscall
    //
    // The program is exactly PROG_LEN bytes, so "ok\n" (the write's source
    // buffer) sits right after it.
    let msg_addr = vaddr + body_off + PROG_LEN;

    let mut body = Vec::new();
    body.extend_from_slice(&[0xBF, 0x01, 0x00, 0x00, 0x00]); // mov edi, 1
    body.push(0xBE); // mov esi, imm32
    body.extend_from_slice(&(msg_addr as u32).to_le_bytes());
    body.extend_from_slice(&[0xBA, 0x03, 0x00, 0x00, 0x00]); // mov edx, 3
    body.extend_from_slice(&[0xB8, 0x01, 0x00, 0x00, 0x00]); // mov eax, 1
    body.extend_from_slice(&[0x0F, 0x05]); // syscall
    body.extend_from_slice(&[0x31, 0xFF]); // xor edi, edi
    body.extend_from_slice(&[0xB8, 0x3C, 0x00, 0x00, 0x00]); // mov eax, 60
    body.extend_from_slice(&[0x0F, 0x05]); // syscall
    assert_eq!(body.len() as u64, PROG_LEN, "PROG_LEN must match the assembled program");
    body.extend_from_slice(b"ok\n");

    let elf = build_elf(vaddr, &body);

    let mut mem = GuestMemory::new(vaddr, 256 * PAGE_SIZE);
    let spec = ProcessSpec {
        argv: vec!["prog".into()],
        envp: vec![],
    };
    let img = load_static(&mut mem, &elf, &spec).unwrap();

    // `vcpu::select` always routes an `X86_64` guest to the dedicated
    // software interpreter (`interp_x86`) — there is no hardware backend for
    // it yet, on any host.
    let backend = nixvm::vcpu::select(Arch::X86_64).unwrap();
    let vcpu = backend.new_vcpu(img.entry, img.stack_pointer).unwrap();

    let captured = Arc::new(Mutex::new(Vec::new()));
    let mut kernel = Kernel::new(Arch::X86_64, MountTable::new());
    kernel.set_stdout(Box::new(SharedBuf(captured.clone())));

    let code = kernel.run(vcpu, mem).unwrap();

    assert_eq!(code, 0);
    assert_eq!(&*captured.lock().unwrap(), b"ok\n");
}
