//! Identity/time syscalls: getpid returns 1, uname fills utsname (ROADMAP
//! Phase 3 groundwork).

use nixvm::abi::Arch;
use nixvm::fs::MountTable;
use nixvm::kernel::Kernel;
use nixvm::vcpu::mem::{PAGE_SIZE, Prot};
use nixvm::vcpu::Backend;
use nixvm::vcpu::interp::InterpBackend;
use nixvm::vcpu::GuestMemory;

/// Load `program` at `base`, optionally map an extra rw page at `rw_page`, run
/// to exit, and return the guest exit code.
fn run(program: &[u32], rw_page: Option<u64>) -> i32 {
    let base = 0x1_0000u64;
    let mut mem = GuestMemory::new(base, 256 * PAGE_SIZE);
    mem.map(base, PAGE_SIZE, Prot::rx()).unwrap();
    if let Some(p) = rw_page {
        mem.map(p, PAGE_SIZE, Prot::rw()).unwrap();
    }
    let mut bytes = Vec::new();
    for w in program {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    mem.write_init(base, &bytes).unwrap();

    let backend = InterpBackend::new(Arch::Aarch64).unwrap();
    let mut vcpu = backend.new_vcpu(base, base + 250 * PAGE_SIZE).unwrap();
    let mut kernel = Kernel::new(Arch::Aarch64, MountTable::new());
    let code = kernel.run(vcpu.as_mut(), &mut mem).unwrap();
    assert!(kernel.unsupported().is_empty(), "{:?}", kernel.unsupported());
    code
}

#[test]
fn getpid_returns_one() {
    // movz x8,#172 ; svc ; movz x8,#93 ; svc   -> exit(getpid())
    let program = [0xD280_1588, 0xD400_0001, 0xD280_0BA8, 0xD400_0001];
    assert_eq!(run(&program, None), 1);
}

#[test]
fn uname_fills_sysname() {
    // x0 = 0x1_2000 (an rw scratch page); uname; exit(sysname[0]) == 'L' (76)
    let program = [
        0xD284_0000, // movz x0,#0x2000
        0xF2A0_0020, // movk x0,#1,lsl#16   -> x0 = 0x1_2000
        0xAA00_03E9, // mov  x9,x0
        0xD280_1408, // movz x8,#160         ; uname
        0xD400_0001, // svc
        0x3940_0120, // ldrb w0,[x9]         ; sysname[0]
        0xD280_0BA8, // movz x8,#93
        0xD400_0001, // svc                  ; exit('L')
    ];
    assert_eq!(run(&program, Some(0x1_2000)), i32::from(b'L'));
}
