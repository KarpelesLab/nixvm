//! Hypervisor.framework backend (macOS / arm64) — the primary target.
//!
//! Runs aarch64 guest code on a real CPU core via Apple's Hypervisor.framework:
//! one process-global VM (`vm`) whose guest-physical (IPA) space holds each
//! guest's physical RAM pool at that pool's own window, and guest threads at
//! EL0 with the MMU on, translated by the guest's real arm64 page tables (one
//! tree per process, in the pool — the same tables the kernel and interpreter
//! use, see [`crate::vcpu::GuestMemory::set_guest_arch`]). Every exception the
//! guest takes to EL1 lands in a vector page of `hvc`s and comes straight out to
//! the host, which turns it into an [`crate::vcpu::Exit`]. See `vcpu.rs` for the
//! execution model, and how thread-bound `hv_vcpu`s meet a scheduler that moves
//! tasks between threads. All `unsafe` FFI lives in `sys.rs`.
//!
//! Entitlement: creating the VM needs `com.apple.security.hypervisor`, so an
//! un-codesigned binary (CI, `cargo test`) gets a graceful error from
//! [`HvfBackend::new`] and the caller falls back to the interpreter.
//! `scripts/hvf-test.sh` signs the test binary and runs the tests below.

mod sys;
mod vcpu;
mod vm;

use crate::abi::Arch;

use super::{Backend, Vcpu, VcpuError};

/// Backend handle. Its existence means the process VM was created successfully.
#[derive(Debug)]
pub struct HvfBackend {
    _private: (),
}

impl HvfBackend {
    /// Probe the hypervisor by creating (once) the process VM. Returns an error
    /// — which [`crate::vcpu::select`] turns into an interpreter fallback — when
    /// the hypervisor is unavailable or the process lacks the
    /// `com.apple.security.hypervisor` entitlement (an unsigned binary / CI).
    pub fn new() -> Result<Self, VcpuError> {
        vm::vm()?;
        Ok(Self { _private: () })
    }
}

impl Backend for HvfBackend {
    fn name(&self) -> &'static str {
        "hvf"
    }

    fn guest_arch(&self) -> Arch {
        Arch::Aarch64
    }

    fn new_vcpu(&self, entry: u64, stack: u64) -> Result<Box<dyn Vcpu>, VcpuError> {
        vcpu::HvfVcpu::new(entry, stack)
    }
}

#[cfg(test)]
mod tests {
    use super::HvfBackend;
    use crate::abi::Arch;
    use crate::fs::MountTable;
    use crate::kernel::Kernel;
    use crate::vcpu::mem::PAGE_SIZE;
    use crate::vcpu::{Backend, Exit, GuestMemory, Prot};
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    const BASE: u64 = 0x1_0000;

    /// The backend, or `None` (the test passes vacuously) unless `NIXVM_HVF` is
    /// set — these need a binary codesigned with the hypervisor entitlement
    /// (`scripts/hvf-test.sh`).
    fn backend() -> Option<HvfBackend> {
        std::env::var_os("NIXVM_HVF")?;
        Some(HvfBackend::new().expect("HVF backend (codesigned + entitled?)"))
    }

    /// `movz`/`movk` loading the 64-bit `imm` into `Xd`.
    fn mov(d: u32, imm: u64) -> Vec<u32> {
        let mut out = vec![0xD280_0000 | (((imm & 0xffff) as u32) << 5) | d];
        for hw in 1..4u32 {
            let part = ((imm >> (16 * hw)) & 0xffff) as u32;
            if part != 0 {
                out.push(0xF280_0000 | (hw << 21) | (part << 5) | d);
            }
        }
        out
    }

    const SVC: u32 = 0xD400_0001;
    const SPIN: u32 = 0x1400_0000; // b .

    fn bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// A guest whose page at `BASE` holds `code`, mapped `prot`.
    fn guest(code: &[u32], prot: Prot) -> GuestMemory {
        let mut mem = GuestMemory::new(BASE, 256 * PAGE_SIZE);
        mem.set_guest_arch(Arch::Aarch64).unwrap();
        mem.map(BASE, PAGE_SIZE, prot).unwrap();
        mem.write_init(BASE, &bytes(code)).unwrap();
        // A stack, for the kernel's signal frames.
        mem.map(BASE + 112 * PAGE_SIZE, 16 * PAGE_SIZE, Prot::rw())
            .unwrap();
        mem
    }

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

    /// Run `mem` from `BASE` under the real kernel; its exit code and stdout.
    fn run_kernel(backend: &dyn Backend, mem: GuestMemory, ncpus: usize) -> (i32, Vec<u8>) {
        let vcpu = backend.new_vcpu(BASE, BASE + 128 * PAGE_SIZE).unwrap();
        let captured = Arc::new(Mutex::new(Vec::new()));
        let mut kernel = Kernel::new(Arch::Aarch64, MountTable::new());
        kernel.set_ncpus(ncpus);
        kernel.set_stdout(Box::new(Sink(captured.clone())));
        kernel.set_heap(BASE + 16 * PAGE_SIZE, BASE + 32 * PAGE_SIZE);
        kernel.set_mmap_area(BASE + 64 * PAGE_SIZE, BASE + 32 * PAGE_SIZE);
        let code = kernel.run(vcpu, mem).expect("kernel run");
        let out = captured.lock().unwrap().clone();
        (code, out)
    }

    /// Two `svc`s with a `set_syscall_ret` in between: the EL0 → vector →
    /// `hvc` trap, the number/argument reads, and the emulated `eret`.
    #[test]
    #[ignore = "requires the hypervisor entitlement + codesign; run scripts/hvf-test.sh"]
    fn el0_syscall_trap_and_resume() {
        let Some(backend) = backend() else { return };
        // movz x8,#172 ; svc #0 ; movz x8,#93 ; movz x0,#7 ; svc #0
        let mut mem = guest(
            &[0xD280_1588, SVC, 0xD280_0BA8, 0xD280_00E0, SVC],
            Prot::rx(),
        );
        let mut v = backend.new_vcpu(BASE, BASE + 0x8000).unwrap();
        assert_eq!(v.run(&mut mem).unwrap(), Exit::Syscall, "first svc traps");
        assert_eq!(v.syscall_nr(), 172, "x8 read as the syscall number");
        assert_eq!(v.pc(), BASE + 4, "pc on the svc");
        assert_eq!(
            v.run(&mut mem).unwrap(),
            Exit::Syscall,
            "an unanswered syscall is re-delivered"
        );
        v.set_syscall_ret(1234);
        assert_eq!(
            v.run(&mut mem).unwrap(),
            Exit::Syscall,
            "resumed to 2nd svc"
        );
        assert_eq!(v.syscall_nr(), 93, "resumed past the first svc");
        assert_eq!(v.syscall_args()[0], 7, "x0 read as arg0");
    }

    /// `write(1, "hi\n", 3); exit(0)` through the real kernel.
    #[test]
    #[ignore = "requires the hypervisor entitlement + codesign; run scripts/hvf-test.sh"]
    fn program_write_exit_through_kernel() {
        let Some(backend) = backend() else { return };
        let msg = BASE + 0x800;
        let mut code = mov(0, 1);
        code.extend(mov(1, msg));
        code.extend(mov(2, 3));
        code.extend(mov(8, 64)); // write
        code.push(SVC);
        code.extend(mov(0, 0));
        code.extend(mov(8, 93)); // exit
        code.push(SVC);
        let mut mem = guest(&code, Prot::rx());
        mem.write_init(msg, b"hi\n").unwrap();
        let (status, out) = run_kernel(&backend, mem, 1);
        assert_eq!(status, 0, "exit code");
        assert_eq!(out, b"hi\n", "stdout via HVF write()");
    }

    /// A load from an unmapped page is a memory fault at that address.
    #[test]
    #[ignore = "requires the hypervisor entitlement + codesign; run scripts/hvf-test.sh"]
    fn stray_access_is_a_memfault() {
        let Some(backend) = backend() else { return };
        let hole = BASE + 64 * PAGE_SIZE;
        let mut code = mov(1, hole);
        code.push(0xF940_0020); // ldr x0, [x1]
        let mut mem = guest(&code, Prot::rx());
        let mut v = backend.new_vcpu(BASE, BASE + 0x8000).unwrap();
        assert_eq!(
            v.run(&mut mem).unwrap(),
            Exit::MemFault {
                addr: hole,
                write: false
            }
        );
        assert_eq!(v.pc(), BASE + 4 * (code.len() as u64 - 1), "pc on the load");
    }

    /// The page tables enforce protection: a store into code faults, and so
    /// does a jump into data.
    #[test]
    #[ignore = "requires the hypervisor entitlement + codesign; run scripts/hvf-test.sh"]
    fn write_to_code_and_exec_of_data_both_fault() {
        let Some(backend) = backend() else { return };
        let mut code = mov(1, BASE);
        code.push(0xF900_0020); // str x0, [x1]
        let mut mem = guest(&code, Prot::rx());
        let mut v = backend.new_vcpu(BASE, BASE + 0x8000).unwrap();
        assert_eq!(
            v.run(&mut mem).unwrap(),
            Exit::MemFault {
                addr: BASE,
                write: true
            },
            "a store into a read-only code page faults"
        );

        let data = BASE + PAGE_SIZE;
        let mut code = mov(1, data);
        code.push(0xD61F_0020); // br x1
        let mut mem = guest(&code, Prot::rx());
        mem.map(data, PAGE_SIZE, Prot::rw()).unwrap();
        mem.write_init(data, &bytes(&[SPIN])).unwrap();
        let mut v = backend.new_vcpu(BASE, BASE + 0x8000).unwrap();
        assert_eq!(
            v.run(&mut mem).unwrap(),
            Exit::MemFault {
                addr: data,
                write: false
            },
            "a fetch from a non-executable page faults"
        );
        assert_eq!(v.pc(), data);
    }

    /// A guest spinning in `b .` never exits on its own; the quantum timer
    /// must break it out at an instruction boundary, resumable.
    #[test]
    #[ignore = "requires the hypervisor entitlement + codesign; run scripts/hvf-test.sh"]
    fn time_quantum_preempts_a_compute_bound_guest() {
        let Some(backend) = backend() else { return };
        let mut mem = guest(&[SPIN], Prot::rx());
        let mut v = backend.new_vcpu(BASE, BASE + 0x8000).unwrap();
        for _ in 0..3 {
            assert_eq!(v.run(&mut mem).unwrap(), Exit::Interrupted);
            assert_eq!(v.pc(), BASE, "still on the loop");
            assert!(v.async_signal_boundary(), "preempted in user mode");
        }
    }

    /// A program that arms `ITIMER_REAL` and spins without another syscall: the
    /// `SIGALRM` handler must still run (it exits 42).
    #[test]
    #[ignore = "requires the hypervisor entitlement + codesign; run scripts/hvf-test.sh"]
    fn a_signal_reaches_a_guest_that_never_makes_a_syscall() {
        let Some(backend) = backend() else { return };
        let (status, _) = run_kernel(&backend, alarm_then_spin(), 1);
        assert_eq!(status, 42, "the handler ran and exited");
        let (status, _) = run_kernel(&backend, alarm_then_spin(), 2);
        assert_eq!(status, 42, "the handler ran and exited, on SMP");
    }

    /// The same program on the interpreter, which runs everywhere: the program
    /// itself is right.
    #[test]
    fn the_alarm_program_works_on_the_interpreter() {
        let backend = crate::vcpu::interp::InterpBackend::new(Arch::Aarch64).unwrap();
        let (status, _) = run_kernel(&backend, alarm_then_spin(), 1);
        assert_eq!(status, 42);
    }

    /// `rt_sigaction(SIGALRM, handler)`, `setitimer(ITIMER_REAL, 20 ms)`, `b .`;
    /// the handler does `exit_group(42)`.
    fn alarm_then_spin() -> GuestMemory {
        let act = BASE + 0x800; // struct sigaction
        let itv = BASE + 0x900; // struct itimerval
        let handler = BASE + 0x400;
        let mut code = mov(0, 14); // SIGALRM
        code.extend(mov(1, act));
        code.extend(mov(2, 0));
        code.extend(mov(3, 8));
        code.extend(mov(8, 134)); // rt_sigaction
        code.push(SVC);
        code.extend(mov(0, 0)); // ITIMER_REAL
        code.extend(mov(1, itv));
        code.extend(mov(2, 0));
        code.extend(mov(8, 103)); // setitimer
        code.push(SVC);
        code.push(SPIN);
        let mut mem = guest(&code, Prot::rwx());
        let mut h = mov(0, 42);
        h.extend(mov(8, 94)); // exit_group
        h.push(SVC);
        mem.write_init(handler, &bytes(&h)).unwrap();
        let mut sa = Vec::new();
        for w in [handler, 0, 0, 0] {
            sa.extend_from_slice(&w.to_le_bytes());
        }
        mem.write_init(act, &sa).unwrap();
        let mut it = Vec::new();
        for w in [0u64, 0, 0, 20_000] {
            it.extend_from_slice(&w.to_le_bytes());
        }
        mem.write_init(itv, &it).unwrap();
        mem
    }

    /// `mrs x0, midr_el1` traps to EL1 and reads what Linux would report;
    /// the guest carries on past it.
    #[test]
    #[ignore = "requires the hypervisor entitlement + codesign; run scripts/hvf-test.sh"]
    fn id_register_reads_are_emulated() {
        let Some(backend) = backend() else { return };
        let mut mem = guest(&[0xD538_0000, SVC], Prot::rx()); // mrs x0, midr_el1
        let mut v = backend.new_vcpu(BASE, BASE + 0x8000).unwrap();
        assert_eq!(v.run(&mut mem).unwrap(), Exit::Syscall);
        assert_eq!(
            Some(v.reg(0)),
            crate::vcpu::interp::el0_id_register(0xC000),
            "MIDR_EL1"
        );
    }

    /// SIMD state set by the host is what the guest computes with, and what
    /// the guest leaves is what the host reads back.
    #[test]
    #[ignore = "requires the hypervisor entitlement + codesign; run scripts/hvf-test.sh"]
    fn simd_state_round_trips() {
        let Some(backend) = backend() else { return };
        // fmov x0, d0 ; mov v1.16b, v0.16b (orr) ; svc
        let mut mem = guest(&[0x9E66_0000, 0x4EA0_1C01, SVC], Prot::rx());
        let mut v = backend.new_vcpu(BASE, BASE + 0x8000).unwrap();
        let mut st = v.simd_state();
        let q0 = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210u128;
        st[8..24].copy_from_slice(&q0.to_le_bytes());
        v.set_simd_state(&st);
        assert_eq!(v.run(&mut mem).unwrap(), Exit::Syscall);
        assert_eq!(v.reg(0), q0 as u64, "the guest saw q0");
        let back = v.simd_state();
        assert_eq!(&back[24..40], &q0.to_le_bytes(), "q1 = q0 read back");
    }

    /// A child forked from a vcpu runs its own copy of the registers on
    /// another thread (another `hv_vcpu`), in its own copy-on-write address
    /// space: a store there leaves the parent's memory alone.
    #[test]
    #[ignore = "requires the hypervisor entitlement + codesign; run scripts/hvf-test.sh"]
    fn fork_runs_elsewhere_copy_on_write() {
        let Some(backend) = backend() else { return };
        let data = BASE + PAGE_SIZE;
        // str x0, [x1] ; svc
        let mut mem = guest(&[0xF900_0020, SVC], Prot::rx());
        mem.map(data, PAGE_SIZE, Prot::rw()).unwrap();
        mem.write_init(data, &7u64.to_le_bytes()).unwrap();
        let mut parent = backend.new_vcpu(BASE, BASE + 0x8000).unwrap();
        parent.set_reg(1, data);
        parent.set_reg(0, 1111);
        let mut child = parent.fork().unwrap();
        child.set_reg(0, 2222);
        let mut child_mem = mem.fork();
        let handle = std::thread::spawn(move || {
            // Its page is copy-on-write too: resolve that as the kernel would.
            let mut exit = child.run(&mut child_mem).unwrap();
            if exit
                == (Exit::MemFault {
                    addr: data,
                    write: true,
                })
            {
                assert!(child_mem.cow_fault(data, true));
                child.flush_tlb();
                exit = child.run(&mut child_mem).unwrap();
            }
            (exit, child_mem.read_u64(data).unwrap())
        });
        let (exit, child_saw) = handle.join().unwrap();
        assert_eq!(exit, Exit::Syscall);
        assert_eq!(child_saw, 2222, "the child's store, in its copy");
        parent.flush_tlb(); // the fork made the parent's pages copy-on-write
        let exit = parent.run(&mut mem).unwrap();
        assert_eq!(
            exit,
            Exit::MemFault {
                addr: data,
                write: true
            },
            "the parent's store hits its now copy-on-write page"
        );
        assert!(mem.cow_fault(data, true));
        parent.flush_tlb();
        assert_eq!(parent.run(&mut mem).unwrap(), Exit::Syscall);
        assert_eq!(mem.read_u64(data).unwrap(), 1111, "the parent's own store");
        assert_eq!(child_saw, 2222);
    }

    /// Two vcpus of one address space on two host threads run at once.
    #[test]
    #[ignore = "requires the hypervisor entitlement + codesign; run scripts/hvf-test.sh"]
    fn two_vcpus_run_concurrently() {
        let Some(backend) = backend() else { return };
        let data = BASE + PAGE_SIZE;
        // str x0, [x1] ; b .
        let mut mem = guest(&[0xF900_0020, SPIN], Prot::rx());
        mem.map(data, PAGE_SIZE, Prot::rw()).unwrap();
        mem.write_init(data, &[0; 16]).unwrap();
        let mem = Arc::new(Mutex::new(mem));
        let handles: Vec<_> = (0..2u64)
            .map(|i| {
                let mut v = backend.new_vcpu(BASE, BASE + 0x8000).unwrap();
                v.set_reg(0, 0x1111 * (i + 1));
                v.set_reg(1, data + 8 * i);
                let mem = Arc::clone(&mem);
                std::thread::spawn(move || {
                    for _ in 0..4 {
                        v.reconcile(&mut mem.lock().unwrap()).unwrap();
                        assert_eq!(v.run_bare().unwrap(), Exit::Interrupted);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let m = mem.lock().unwrap();
        assert_eq!(m.read_u64(data).unwrap(), 0x1111);
        assert_eq!(m.read_u64(data + 8).unwrap(), 0x2222);
    }

    /// A vcpu spinning on another thread with no quantum to end its run stops
    /// when kicked — how a signal reaches a task running on another worker.
    #[test]
    #[ignore = "requires the hypervisor entitlement + codesign; run scripts/hvf-test.sh"]
    fn a_kick_stops_a_vcpu_running_on_another_thread() {
        let Some(_backend) = backend() else { return };
        let mut mem = guest(&[SPIN], Prot::rx());
        let mut v = super::vcpu::HvfVcpu::unpreempted(BASE, BASE + 0x8000);
        let kick = v.kicker().expect("HVF can kick");
        let (tx, rx) = std::sync::mpsc::channel();
        let runner = std::thread::spawn(move || {
            let exit = v.run(&mut mem).unwrap();
            tx.send(()).unwrap();
            (exit, v.pc())
        });
        // Kick until it stops (a kick before it is inside the run may be lost).
        let mut stopped = false;
        for _ in 0..200 {
            std::thread::sleep(std::time::Duration::from_millis(10));
            kick();
            if rx.try_recv().is_ok() {
                stopped = true;
                break;
            }
        }
        assert!(stopped, "the kick stopped the run");
        let (exit, pc) = runner.join().unwrap();
        assert_eq!(exit, Exit::Interrupted);
        assert_eq!(pc, BASE, "at a user instruction boundary");
    }
}
