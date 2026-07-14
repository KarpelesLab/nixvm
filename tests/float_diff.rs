//! Differential test: run the *same* float machine code under the software
//! interpreter and under real hardware (KVM), across all four rounding modes,
//! and require bit-identical results — SSE `DIVSD`/`SQRTSD`, the resulting
//! MXCSR exception flags, and an x87 80-bit `FMULP` stored as `m80`.
//!
//! Skips cleanly when `/dev/kvm` is unavailable (CI without the device), where
//! there is no hardware oracle to diff against.

#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use nixvm::abi::Arch;
use nixvm::vcpu::mem::{PAGE_SIZE, Prot};
use nixvm::vcpu::{Backend, Exit, GuestMemory};

const BASE: u64 = 0x10_0000;
const DATA: u64 = 0x11_0000;

/// Assemble the snippet once; it addresses its data area through `rax = DATA`.
fn program() -> Vec<u8> {
    let mut c = Vec::new();
    c.extend_from_slice(&[0x48, 0xB8]); // mov rax, imm64
    c.extend_from_slice(&DATA.to_le_bytes());
    c.extend_from_slice(&[0x0F, 0xAE, 0x10]); // ldmxcsr [rax]        (rounding mode in)
    c.extend_from_slice(&[0xF2, 0x0F, 0x10, 0x40, 0x08]); // movsd xmm0,[rax+8]  (a)
    c.extend_from_slice(&[0xF2, 0x0F, 0x5E, 0x40, 0x10]); // divsd xmm0,[rax+16] (a/b)
    c.extend_from_slice(&[0xF2, 0x0F, 0x11, 0x40, 0x18]); // movsd [rax+24],xmm0
    c.extend_from_slice(&[0xF2, 0x0F, 0x51, 0x48, 0x08]); // sqrtsd xmm1,[rax+8] (√a)
    c.extend_from_slice(&[0xF2, 0x0F, 0x11, 0x48, 0x20]); // movsd [rax+32],xmm1
    c.extend_from_slice(&[0xD9, 0xEB]); // fldpi
    c.extend_from_slice(&[0xDD, 0x40, 0x08]); // fld qword [rax+8]
    c.extend_from_slice(&[0xDE, 0xC9]); // fmulp st(1),st(0)   (π·a at 80-bit)
    c.extend_from_slice(&[0xDB, 0x78, 0x28]); // fstp tbyte [rax+40]
    c.extend_from_slice(&[0x0F, 0xAE, 0x58, 0x38]); // stmxcsr [rax+56]    (flags out)
    c.extend_from_slice(&[0x31, 0xFF]); // xor edi, edi
    c.extend_from_slice(&[0xB8, 0x3C, 0x00, 0x00, 0x00]); // mov eax, 60 (exit)
    c.extend_from_slice(&[0x0F, 0x05]); // syscall
    c
}

/// Run the snippet on `backend` with inputs `(mxcsr, a, b)`; return the 48-byte
/// result window `[DATA+24 .. DATA+72]` (divsd, sqrtsd, m80, stmxcsr).
fn run(backend: &dyn Backend, code: &[u8], mxcsr: u32, a: f64, b: f64) -> Vec<u8> {
    let mut mem = GuestMemory::new(BASE, 512 * PAGE_SIZE);
    mem.map(BASE, 512 * PAGE_SIZE, Prot::rwx()).unwrap();
    mem.write_init(BASE, code).unwrap();
    mem.write_init(DATA, &mxcsr.to_le_bytes()).unwrap();
    mem.write_init(DATA + 8, &a.to_bits().to_le_bytes()).unwrap();
    mem.write_init(DATA + 16, &b.to_bits().to_le_bytes()).unwrap();

    let mut vcpu = backend.new_vcpu(BASE, BASE + 400 * PAGE_SIZE).unwrap();
    // The snippet ends in an exit `syscall`; run until it traps out.
    for _ in 0..1000 {
        match vcpu.run(&mut mem).unwrap() {
            Exit::Syscall => break,
            Exit::Interrupted => {} // rescheduled; keep running
            other => panic!("unexpected exit {other:?}"),
        }
    }
    mem.read_vec(DATA + 24, 48).unwrap()
}

#[test]
fn sse_and_x87_match_hardware_across_rounding_modes() {
    let Ok(kvm) = nixvm::vcpu::kvm::KvmBackend::new() else {
        eprintln!("/dev/kvm unavailable; skipping hardware differential");
        return;
    };
    let interp = nixvm::vcpu::interp_x86::X86Backend::new(Arch::X86_64).unwrap();
    let code = program();

    // Operand pairs that make a/b, √a and π·a inexact, so directed rounding
    // actually bites; plus a subnormal and a large value.
    let cases: &[(f64, f64)] = &[
        (1.0, 3.0),
        (2.0, 7.0),
        (10.0, 3.0),
        (1.0, 0.0),          // divide by zero -> inf + ZE flag
        (2.0, 1.0),          // √2 inexact
        (1e300, 7.0),
        (5e-324, 3.0),       // smallest subnormal
        (-1.0, 3.0),
    ];
    // MXCSR values selecting each rounding mode (exceptions masked, bits set).
    let modes = [0x1f80u32, 0x3f80, 0x5f80, 0x7f80]; // RNE, down, up, zero

    for &mx in &modes {
        for &(a, b) in cases {
            let hw = run(&kvm, &code, mx, a, b);
            let sw = run(&interp, &code, mx, a, b);
            assert_eq!(
                sw, hw,
                "mismatch mxcsr={mx:#06x} a={a} b={b}\n interp={sw:02x?}\n kvm   ={hw:02x?}"
            );
        }
    }
}
