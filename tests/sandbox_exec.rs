//! The public `Sandbox` API runs a static ELF end-to-end (loader → interpreter
//! → kernel), computing the heap/mmap/stack layout itself. This is the whole
//! pipeline exercised through the embeddable entry point (ROADMAP Phases 1-2).

use nixvm::Sandbox;

const EHDR_LEN: usize = 64;
const PHDR_LEN: usize = 56;
const PAGE: u64 = 4096;

/// Minimal ET_EXEC aarch64 ELF: one RWX PT_LOAD at `vaddr` covering headers +
/// `body`, entry at the start of `body`.
fn build_elf(vaddr: u64, body: &[u8]) -> Vec<u8> {
    let mut f = vec![0u8; EHDR_LEN + PHDR_LEN];
    f[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
    f[4] = 2;
    f[5] = 1;
    f[6] = 1;
    f[16..18].copy_from_slice(&2u16.to_le_bytes());
    f[18..20].copy_from_slice(&183u16.to_le_bytes()); // EM_AARCH64
    f[20..24].copy_from_slice(&1u32.to_le_bytes());
    let body_off = (EHDR_LEN + PHDR_LEN) as u64;
    f[24..32].copy_from_slice(&(vaddr + body_off).to_le_bytes());
    f[32..40].copy_from_slice(&(EHDR_LEN as u64).to_le_bytes());
    f[52..54].copy_from_slice(&(EHDR_LEN as u16).to_le_bytes());
    f[54..56].copy_from_slice(&(PHDR_LEN as u16).to_le_bytes());
    f[56..58].copy_from_slice(&1u16.to_le_bytes());
    let total = body_off + body.len() as u64;
    let p = EHDR_LEN;
    f[p..p + 4].copy_from_slice(&1u32.to_le_bytes()); // PT_LOAD
    f[p + 4..p + 8].copy_from_slice(&7u32.to_le_bytes()); // RWX
    f[p + 16..p + 24].copy_from_slice(&vaddr.to_le_bytes());
    f[p + 24..p + 32].copy_from_slice(&vaddr.to_le_bytes());
    f[p + 32..p + 40].copy_from_slice(&total.to_le_bytes());
    f[p + 40..p + 48].copy_from_slice(&total.to_le_bytes());
    f[p + 48..p + 56].copy_from_slice(&PAGE.to_le_bytes());
    f.extend_from_slice(body);
    f
}

#[test]
fn sandbox_runs_static_elf_and_returns_exit_code() {
    // movz x0,#42 ; movz x8,#93 ; svc   -> exit(42)
    let program: [u32; 3] = [0xD280_0540, 0xD280_0BA8, 0xD400_0001];
    let mut body = Vec::new();
    for w in program {
        body.extend_from_slice(&w.to_le_bytes());
    }
    let elf = build_elf(0x1_0000, &body);

    let code = Sandbox::builder()
        .command(["prog"])
        .prefer_interp(true)
        .mem_bytes(8 * 1024 * 1024)
        .build()
        .exec_elf(&elf)
        .expect("sandbox should run the ELF");

    assert_eq!(code, 42);
}
