//! End-to-end: a hand-assembled aarch64 program runs on the software
//! interpreter, its `write` syscall reaches a captured sink, and `exit_group`
//! returns the guest's status through the kernel run/serve loop.
//!
//! This is ROADMAP Phase 1's observable outcome (a guest makes a syscall and
//! exits) on the CI-testable interpreter path — no hypervisor required.

use std::io::Write;
use std::sync::{Arc, Mutex};

use nixvm::abi::Arch;
use nixvm::fs::MountTable;
use nixvm::kernel::Kernel;
use nixvm::vcpu::Backend;
use nixvm::vcpu::interp::InterpBackend;
use nixvm::vcpu::mem::{PAGE_SIZE, Prot};
use nixvm::vcpu::GuestMemory;

/// A `Write` sink that appends into a shared buffer the test can inspect.
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

#[test]
fn hello_world_runs_and_exits() {
    let base = 0x1_0000u64;
    let code = base;
    let data = base + PAGE_SIZE; // 0x1_1000
    let stack = base + 60 * PAGE_SIZE;

    // aarch64:
    //   movz x0, #1              ; fd = stdout
    //   movz x1, #0x1000         ; }
    //   movk x1, #0x1, lsl #16   ; } x1 = 0x1_1000 (address of "hi\n")
    //   movz x2, #3              ; len
    //   movz x8, #64             ; __NR_write
    //   svc  #0
    //   movz x0, #0              ; status = 0
    //   movz x8, #93             ; __NR_exit
    //   svc  #0
    let program: [u32; 9] = [
        0xD280_0020,
        0xD282_0001,
        0xF2A0_0021,
        0xD280_0062,
        0xD280_0808,
        0xD400_0001,
        0xD280_0000,
        0xD280_0BA8,
        0xD400_0001,
    ];

    let mut mem = GuestMemory::new(base, 64 * PAGE_SIZE);
    mem.map(code, PAGE_SIZE, Prot::rx()).unwrap();
    mem.map(data, PAGE_SIZE, Prot::READ).unwrap();

    let mut bytes = Vec::new();
    for w in program {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    mem.write_init(code, &bytes).unwrap();
    mem.write_init(data, b"hi\n").unwrap();

    let backend = InterpBackend::new(Arch::Aarch64).unwrap();
    let vcpu = backend.new_vcpu(code, stack).unwrap();

    let captured = Arc::new(Mutex::new(Vec::new()));
    let mut kernel = Kernel::new(Arch::Aarch64, MountTable::new());
    kernel.set_stdout(Box::new(SharedBuf(captured.clone())));

    let exit_code = kernel.run(vcpu, mem).unwrap();

    assert_eq!(exit_code, 0, "guest should exit with status 0");
    assert_eq!(&*captured.lock().unwrap(), b"hi\n", "guest should print hi");
    // Nothing should have hit the unsupported ledger.
    assert!(kernel.unsupported().is_empty(), "unexpected: {:?}", kernel.unsupported());
}
