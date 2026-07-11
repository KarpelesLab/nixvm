//! End-to-end: `brk` grows the guest heap, the program stores into the freshly
//! mapped page and reads it back, then exits with that value (ROADMAP Phase 2).

use nixvm::abi::Arch;
use nixvm::fs::MountTable;
use nixvm::kernel::Kernel;
use nixvm::vcpu::mem::{PAGE_SIZE, Prot};
use nixvm::vcpu::Backend;
use nixvm::vcpu::interp::InterpBackend;
use nixvm::vcpu::GuestMemory;

#[test]
fn brk_grows_heap_and_memory_is_usable() {
    let base = 0x1_0000u64;
    let heap_start = base + PAGE_SIZE; // 0x1_1000
    let heap_limit = base + 200 * PAGE_SIZE;

    //   mov x8,#214 ; mov x0,#0 ; svc          ; x0 = current break (heap_start)
    //   mov x19,x0
    //   add x0,x19,#2,lsl#12                    ; request +0x2000
    //   mov x8,#214 ; svc                       ; grow the heap
    //   movz x2,#42 ; strb w2,[x19]             ; write into the new page
    //   ldrb w0,[x19]                           ; read it back
    //   movz x8,#93 ; svc                       ; exit(42)
    let program: [u32; 12] = [
        0xD280_1AC8,
        0xD280_0000,
        0xD400_0001,
        0xAA00_03F3,
        0x9140_0A60,
        0xD280_1AC8,
        0xD400_0001,
        0xD280_0542,
        0x3900_0262,
        0x3940_0260,
        0xD280_0BA8,
        0xD400_0001,
    ];

    let mut mem = GuestMemory::new(base, 256 * PAGE_SIZE);
    mem.map(base, PAGE_SIZE, Prot::rx()).unwrap();
    let mut bytes = Vec::new();
    for w in program {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    mem.write_init(base, &bytes).unwrap();

    let backend = InterpBackend::new(Arch::Aarch64).unwrap();
    let vcpu = backend.new_vcpu(base, base + 250 * PAGE_SIZE).unwrap();

    let mut kernel = Kernel::new(Arch::Aarch64, MountTable::new());
    kernel.set_heap(heap_start, heap_limit);

    let code = kernel.run(vcpu, mem).unwrap();
    assert_eq!(code, 42, "value written to the brk-grown heap round-trips");
    assert!(kernel.unsupported().is_empty(), "{:?}", kernel.unsupported());
}
