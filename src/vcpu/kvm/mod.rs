//! KVM backend (Linux) — hardware virtualization for x86-64 guests.
//!
//! Runs guest code on a real CPU core via `/dev/kvm`, reusing the exact seam
//! the HVF backend proved: one VM per backend whose guest-physical space holds
//! the guest's contiguous [`crate::vcpu::GuestMemory`] region (identity-mapped
//! so guest virtual == guest physical), with each guest thread on its own KVM
//! vcpu. A guest `syscall` vectors to a `hlt; sysretq` trampoline whose `hlt`
//! exits to the host as `KVM_EXIT_HLT` → [`crate::vcpu::Exit::Syscall`];
//! accesses to unbacked guest-physical addresses become
//! [`crate::vcpu::Exit::MemFault`]. All `unsafe` FFI lives in [`sys`]; the VM,
//! its guest-physical layout, and the control block (page tables, GDT,
//! trampoline) in [`vm`].
//!
//! Availability: creating the VM needs a readable+writable `/dev/kvm`, so a
//! host without KVM (or a CI runner without the device) gets a graceful error
//! from [`KvmBackend::new`] and the caller falls back to the interpreter.
//! Unlike HVF there is no entitlement/codesigning step — the tests below run
//! under plain `cargo test` wherever `/dev/kvm` exists, and skip themselves
//! where it doesn't.

mod sys;
mod vcpu;
mod vm;

use crate::abi::Arch;
use std::sync::Arc;

use super::{Backend, Vcpu, VcpuError};

/// Backend handle owning one KVM virtual machine.
#[derive(Debug)]
pub struct KvmBackend {
    vm: Arc<vm::Vm>,
}

impl KvmBackend {
    /// Probe KVM by opening `/dev/kvm` and building the VM (control block
    /// mapped, CPUID snapshot taken). Returns an error — which
    /// [`crate::vcpu::select`] turns into an interpreter fallback — when KVM
    /// is unavailable or inaccessible.
    pub fn new() -> Result<Self, VcpuError> {
        Ok(Self {
            vm: Arc::new(vm::Vm::new()?),
        })
    }
}

impl Backend for KvmBackend {
    fn name(&self) -> &'static str {
        "kvm"
    }

    fn guest_arch(&self) -> Arch {
        Arch::X86_64
    }

    fn new_vcpu(&self, entry: u64, stack: u64) -> Result<Box<dyn Vcpu>, VcpuError> {
        vcpu::KvmVcpu::new(self.vm.clone(), entry, stack)
    }
}

#[cfg(test)]
mod tests {
    use super::KvmBackend;
    use crate::vcpu::mem::PAGE_SIZE;
    use crate::vcpu::{Backend, Exit, GuestMemory, Prot};

    /// A skip-not-fail probe: build the backend, or report why the host can't
    /// (no `/dev/kvm` in a container/CI). Every test below starts here, so the
    /// suite stays green on KVM-less hosts while running for real elsewhere.
    fn backend_or_skip() -> Option<KvmBackend> {
        match KvmBackend::new() {
            Ok(b) => Some(b),
            Err(e) => {
                eprintln!("skipping KVM test: {e}");
                None
            }
        }
    }

    /// Drive a guest CPL3 program through the full `Vcpu` surface: two
    /// `syscall`s with a `set_syscall_ret` in between. Proves the
    /// syscall→LSTAR-trampoline→HLT trap, the number/argument reads off the
    /// hardware registers, and that the resumed `sysretq` returns to user code
    /// after the `syscall` (the guest makes progress to the second one).
    #[test]
    fn cpl3_syscall_trap_and_resume() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let base = 0x1_0000u64;
        let mut mem = GuestMemory::new(base, 64 * 1024);
        mem.map(base, PAGE_SIZE, Prot::rwx()).unwrap();
        let mut program: Vec<u8> = Vec::new();
        program.extend_from_slice(&[0xB8, 0xAC, 0x00, 0x00, 0x00]); // mov eax, 172
        program.extend_from_slice(&[0x0F, 0x05]); // syscall
        program.extend_from_slice(&[0xBF, 0x07, 0x00, 0x00, 0x00]); // mov edi, 7
        program.extend_from_slice(&[0xB8, 0x3C, 0x00, 0x00, 0x00]); // mov eax, 60
        program.extend_from_slice(&[0x0F, 0x05]); // syscall
        mem.write_init(base, &program).unwrap();

        let mut v = backend.new_vcpu(base, base + 0x8000).expect("create KVM vcpu");

        assert_eq!(
            v.run(&mut mem).unwrap(),
            Exit::Syscall,
            "first syscall traps"
        );
        assert_eq!(v.syscall_nr(), 172, "rax read as the syscall number");
        v.set_syscall_ret(1234);

        assert_eq!(
            v.run(&mut mem).unwrap(),
            Exit::Syscall,
            "resumed to 2nd syscall"
        );
        assert_eq!(v.syscall_nr(), 60, "resumed past the first syscall");
        assert_eq!(v.syscall_args()[0], 7, "rdi read as arg0");
    }

    /// End-to-end through the real kernel: a guest program does
    /// `write(1,"hi\n",3)` then `exit(0)`, run on KVM and driven by the actual
    /// `Kernel` run/serve loop — the "static program runs entirely through
    /// KVM" deliverable, mirroring the HVF milestone test.
    #[test]
    fn program_write_exit_through_kernel() {
        use crate::abi::Arch;
        use crate::fs::MountTable;
        use crate::kernel::Kernel;
        use std::io::Write;
        use std::sync::{Arc, Mutex};

        #[derive(Clone)]
        struct Sink(Arc<Mutex<Vec<u8>>>);
        impl Write for Sink {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        const PROG_LEN: usize = 31; // "hi\n" (the write's buffer) sits right after

        let Some(backend) = backend_or_skip() else {
            return;
        };

        let base = 0x1_0000u64;
        let mut program: Vec<u8> = Vec::new();
        program.extend_from_slice(&[0xBF, 0x01, 0x00, 0x00, 0x00]); // mov edi, 1 (fd)
        program.push(0xBE); // mov esi, imm32 (buf)
        program.extend_from_slice(&(base as u32 + PROG_LEN as u32).to_le_bytes());
        program.extend_from_slice(&[0xBA, 0x03, 0x00, 0x00, 0x00]); // mov edx, 3 (len)
        program.extend_from_slice(&[0xB8, 0x01, 0x00, 0x00, 0x00]); // mov eax, 1 (write)
        program.extend_from_slice(&[0x0F, 0x05]); // syscall
        program.extend_from_slice(&[0x31, 0xFF]); // xor edi, edi (status 0)
        program.extend_from_slice(&[0xB8, 0x3C, 0x00, 0x00, 0x00]); // mov eax, 60 (exit)
        program.extend_from_slice(&[0x0F, 0x05]); // syscall
        assert_eq!(program.len(), PROG_LEN, "PROG_LEN must match the assembly");
        program.extend_from_slice(b"hi\n");

        let mut mem = GuestMemory::new(base, 256 * PAGE_SIZE);
        mem.map(base, PAGE_SIZE, Prot::rwx()).unwrap();
        mem.write_init(base, &program).unwrap();

        let vcpu = backend.new_vcpu(base, base + 0x1_0000).expect("new_vcpu");

        let captured = Arc::new(Mutex::new(Vec::new()));
        let mut kernel = Kernel::new(Arch::X86_64, MountTable::new());
        kernel.set_ncpus(1); // hardware vcpus pair with the serial scheduler
        kernel.set_stdout(Box::new(Sink(captured.clone())));

        let code = kernel.run(vcpu, mem).expect("kernel run");
        assert_eq!(code, 0, "exit code");
        assert_eq!(&*captured.lock().unwrap(), b"hi\n", "stdout via KVM write()");
    }

    /// A guest touching a guest-physical hole (below the mapped region) must
    /// surface as a memory fault, not a hang or a backend error.
    #[test]
    fn stray_access_is_a_memfault() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let base = 0x1_0000u64;
        let mut mem = GuestMemory::new(base, 64 * 1024);
        mem.map(base, PAGE_SIZE, Prot::rwx()).unwrap();
        // mov byte ptr [0x1000], 1 — a write to unbacked guest-physical space
        // (below the region): C6 04 25 00 10 00 00 01
        mem.write_init(base, &[0xC6, 0x04, 0x25, 0x00, 0x10, 0x00, 0x00, 0x01])
            .unwrap();
        let mut v = backend.new_vcpu(base, base + 0x8000).expect("create KVM vcpu");
        match v.run(&mut mem).unwrap() {
            Exit::MemFault { addr, .. } => {
                // The access is to an unmapped page (below the guest region), so
                // it faults through the guest page tables now enforcing
                // protection — `cr2` carries the address; the read/write
                // direction isn't recovered from the triple-fault path.
                assert_eq!(addr, 0x1000, "faulting guest-physical address");
            }
            other => panic!("expected MemFault, got {other:?}"),
        }
    }

    /// W^X: the page tables must enforce protection. A store to a read-only
    /// (executable) code page faults, and a jump into a writable (`NX`) data
    /// page faults — neither silently succeeds as it did under the old
    /// uniformly-RWX identity map.
    #[test]
    fn write_to_code_and_exec_of_data_both_fault() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let base = 0x1_0000u64;

        // (a) code page is RX; a store into it must fault.
        {
            let mut mem = GuestMemory::new(base, 64 * 1024);
            mem.map(base, PAGE_SIZE, Prot::rx()).unwrap(); // read + execute, no write
            // mov byte ptr [rip-relative self], 1 → write into the code page:
            //   C6 05 00 00 00 00 01  (mov byte [rip+0], 1) then it faults on the store.
            mem.write_init(base, &[0xC6, 0x05, 0x00, 0x00, 0x00, 0x00, 0x01]).unwrap();
            let mut v = backend.new_vcpu(base, base + 0x8000).unwrap();
            assert!(
                matches!(v.run(&mut mem).unwrap(), Exit::MemFault { .. }),
                "store into a read-only code page must fault"
            );
        }

        // (b) data page is RW (NX); jumping to it must fault on the fetch.
        {
            let mut mem = GuestMemory::new(base, 128 * 1024);
            mem.map(base, PAGE_SIZE, Prot::rx()).unwrap(); // a tiny code page
            let data = base + PAGE_SIZE;
            mem.map(data, PAGE_SIZE, Prot::rw()).unwrap(); // NX data page
            mem.write_init(data, &[0x90]).unwrap(); // a valid `nop` sits there
            // mov rax, data ; jmp rax
            let mut code = vec![0x48, 0xB8];
            code.extend_from_slice(&data.to_le_bytes());
            code.extend_from_slice(&[0xFF, 0xE0]); // jmp rax
            mem.write_init(base, &code).unwrap();
            let mut v = backend.new_vcpu(base, base + 0x1_0000).unwrap();
            match v.run(&mut mem).unwrap() {
                Exit::MemFault { addr, .. } => {
                    assert_eq!(addr, data, "instruction fetch from the NX page faults at it");
                }
                other => panic!("expected an NX fetch fault, got {other:?}"),
            }
        }
    }

    /// Real intra-process parallelism: two KVM vcpus in the *same* address space
    /// (one shared VM, one `GuestMemory`) must execute `KVM_RUN` on two host cores
    /// at once — the whole point of the lockless SMP split. Each thread mirrors the
    /// scheduler: reconcile under the memory lock, then run with it dropped. A
    /// `jmp $` guest is compute-bound, so each `run_bare` spans a full preemption
    /// quantum; an instrument around it records the peak number of vcpus inside
    /// `KVM_RUN` simultaneously. If the run still held the memory lock (the bug
    /// this milestone fixes), the two threads would serialize on that one Mutex and
    /// the peak would never exceed 1. Also checks each vcpu's independent write
    /// landed in its own slot — no corruption from the concurrent runs + the
    /// concurrent shadow-page-table sync. Skips cleanly without `/dev/kvm`.
    #[test]
    fn two_vcpus_run_concurrently_in_one_address_space() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::{Arc, Barrier, Mutex};

        let Some(backend) = backend_or_skip() else {
            return;
        };
        let base = 0x1_0000u64;
        let code = base;
        let data = base + PAGE_SIZE; // each vcpu's result slot lives here
        let mut mem = GuestMemory::new(base, 64 * 1024);
        mem.map(code, PAGE_SIZE, Prot::rwx()).unwrap();
        mem.map(data, PAGE_SIZE, Prot::rw()).unwrap();
        // mov [rsi], eax ; jmp $  — store this vcpu's value to its slot, then spin.
        mem.write_init(code, &[0x89, 0x06, 0xEB, 0xFE]).unwrap();
        let mem = Arc::new(Mutex::new(mem));

        // Two vcpus over the one shared VM (backend), each with a distinct value
        // (eax) and slot (rsi) so their writes must not collide.
        let magic = [0x1111_1111u32, 0x2222_2222u32];
        let slots = [data, data + 8];
        let mut vcpus = Vec::new();
        for i in 0..2usize {
            let mut v = backend
                .new_vcpu(code, base + 0x8000 + (i as u64) * 0x1000)
                .expect("create KVM vcpu");
            v.set_reg(0, u64::from(magic[i])); // eax
            v.set_reg(6, slots[i]); // rsi
            vcpus.push(v);
        }

        let in_flight = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));
        let barrier = Arc::new(Barrier::new(2));
        let handles: Vec<_> = vcpus
            .into_iter()
            .map(|mut v| {
                let mem = Arc::clone(&mem);
                let in_flight = Arc::clone(&in_flight);
                let max_seen = Arc::clone(&max_seen);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    for _ in 0..8 {
                        {
                            let mut m = mem.lock().unwrap();
                            v.reconcile(&mut m).unwrap();
                        }
                        let n = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                        max_seen.fetch_max(n, Ordering::SeqCst);
                        let exit = v.run_bare().expect("KVM run");
                        in_flight.fetch_sub(1, Ordering::SeqCst);
                        assert_eq!(exit, Exit::Interrupted, "compute-bound guest is preempted");
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }

        assert!(
            max_seen.load(Ordering::SeqCst) >= 2,
            "two vcpus in one address space must be inside KVM_RUN at once (peak {})",
            max_seen.load(Ordering::SeqCst)
        );
        let m = mem.lock().unwrap();
        assert_eq!(m.read_u32(slots[0]).unwrap(), magic[0], "vcpu 0 wrote its slot");
        assert_eq!(m.read_u32(slots[1]).unwrap(), magic[1], "vcpu 1 wrote its slot");
    }

    /// Time-based preemption on the hardware path: a guest spinning forever in a
    /// `jmp $` self-loop makes no syscall and never faults, so without the
    /// per-thread preemption timer `KVM_RUN` would never return and `run` would
    /// hang. The armed timer's signal must break it out as `Exit::Interrupted`,
    /// leaving the guest resumable (rip still on the loop). Uses the default
    /// `NIXVM_QUANTUM_MS` quantum, so a broken timer hangs the test rather than
    /// passing.
    #[test]
    fn time_quantum_preempts_a_compute_bound_guest() {
        let Some(backend) = backend_or_skip() else {
            return;
        };
        let base = 0x1_0000u64;
        let mut mem = GuestMemory::new(base, 64 * 1024);
        mem.map(base, PAGE_SIZE, Prot::rwx()).unwrap();
        mem.write_init(base, &[0xEB, 0xFE]).unwrap(); // jmp $ (spin forever)

        let mut v = backend.new_vcpu(base, base + 0x8000).expect("create KVM vcpu");
        let exit = v.run(&mut mem).expect("KVM run");
        assert_eq!(exit, Exit::Interrupted, "the quantum timer preempts the guest");
        assert_eq!(v.pc(), base, "still on the self-loop, resumable");
    }
}
