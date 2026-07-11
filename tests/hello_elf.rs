//! End-to-end: a minimal *static ELF64* is loaded by the loader, then executed
//! on the interpreter; its `write` reaches a captured sink and `exit` returns
//! the status. Exercises loader + interp + kernel together (ROADMAP Phase 2).

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
/// the headers + `body` (code followed by data), entry at the start of `body`.
fn build_elf(vaddr: u64, body: &[u8]) -> Vec<u8> {
    let mut f = vec![0u8; EHDR_LEN + PHDR_LEN];
    f[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
    f[4] = 2; // ELFCLASS64
    f[5] = 1; // ELFDATA2LSB
    f[6] = 1;
    f[16..18].copy_from_slice(&2u16.to_le_bytes()); // ET_EXEC
    f[18..20].copy_from_slice(&183u16.to_le_bytes()); // EM_AARCH64
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

#[test]
fn static_elf_prints_and_exits() {
    let vaddr = 0x1_0000u64;
    // "hi\n" sits right after the 9-instruction (36-byte) program, so its guest
    // address is vaddr + (EHDR+PHDR) + 36 = 0x1_0000 + 156 = 0x1_009C.
    //   movz x0,#1 ; movz x1,#0x9C ; movk x1,#1,lsl#16 ; movz x2,#3
    //   movz x8,#64 ; svc ; movz x0,#0 ; movz x8,#93 ; svc
    let program: [u32; 9] = [
        0xD280_0020,
        0xD280_1381,
        0xF2A0_0021,
        0xD280_0062,
        0xD280_0808,
        0xD400_0001,
        0xD280_0000,
        0xD280_0BA8,
        0xD400_0001,
    ];
    let mut body = Vec::new();
    for w in program {
        body.extend_from_slice(&w.to_le_bytes());
    }
    body.extend_from_slice(b"hi\n");

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

    let code = kernel.run(vcpu, mem).unwrap();

    assert_eq!(code, 0);
    assert_eq!(&*captured.lock().unwrap(), b"hi\n");
}
