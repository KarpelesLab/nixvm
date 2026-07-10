//! End-to-end: anonymous `mmap` returns a usable page; the program writes and
//! reads it back, then exits with the value (ROADMAP Phase 2).

use nixvm::abi::Arch;
use nixvm::fs::MountTable;
use nixvm::kernel::Kernel;
use nixvm::vcpu::mem::{PAGE_SIZE, Prot};
use nixvm::vcpu::Backend;
use nixvm::vcpu::interp::InterpBackend;
use nixvm::vcpu::GuestMemory;

#[test]
fn anonymous_mmap_returns_usable_memory() {
    let base = 0x1_0000u64;

    //   mov  x0,#0            ; addr = NULL (let the kernel choose)
    //   movz x1,#0x1000       ; len = 4096
    //   movz x2,#3            ; prot = READ|WRITE
    //   movz x3,#0x22         ; flags = MAP_PRIVATE|MAP_ANONYMOUS
    //   movn x4,#0            ; fd = -1
    //   mov  x5,#0            ; offset = 0
    //   movz x8,#222 ; svc    ; x0 = mapped address
    //   mov  x9,x0
    //   movz x2,#99 ; strb w2,[x9] ; ldrb w0,[x9]
    //   movz x8,#93 ; svc     ; exit(99)
    let program: [u32; 15] = [
        0xD280_0000,
        0xD282_0001,
        0xD280_0062,
        0xD280_0443,
        0x9280_0004,
        0xD280_0005,
        0xD280_1BC8,
        0xD400_0001,
        0xAA00_03E9,
        0xD280_0C62,
        0x3900_0122,
        0x3940_0120,
        0xD280_0BA8,
        0xD400_0001,
        0xD503_201F, // nop (padding)
    ];

    let mut mem = GuestMemory::new(base, 256 * PAGE_SIZE);
    mem.map(base, PAGE_SIZE, Prot::rx()).unwrap();
    let mut bytes = Vec::new();
    for w in program {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    mem.write_init(base, &bytes).unwrap();

    let backend = InterpBackend::new(Arch::Aarch64).unwrap();
    let mut vcpu = backend.new_vcpu(base, base + 250 * PAGE_SIZE).unwrap();

    let mut kernel = Kernel::new(Arch::Aarch64, MountTable::new());
    kernel.set_mmap_area(base + 240 * PAGE_SIZE, base + 200 * PAGE_SIZE);

    let code = kernel.run(vcpu.as_mut(), &mut mem).unwrap();
    assert_eq!(code, 99, "value written to the mmap'd page round-trips");
    assert!(kernel.unsupported().is_empty(), "{:?}", kernel.unsupported());
}
