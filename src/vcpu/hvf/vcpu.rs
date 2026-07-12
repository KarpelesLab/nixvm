//! The HVF virtual CPU: one guest thread on a thread-bound `hv_vcpu`.
//!
//! Execution model (MMU off, guest at EL0):
//! * Guest RAM ([`GuestMemory`]) is one contiguous region `hv_vm_map`ped into
//!   the VM's single IPA space at `[base, base+size)`; with the MMU off the
//!   guest's virtual addresses *are* those IPAs, so the flat address space the
//!   interpreter models maps 1:1 onto hardware. The backend re-issues the map
//!   whenever [`GuestMemory::backing_generation`] changes (fork/execve).
//! * A guest `svc` traps to EL1, whose vector base ([`stub`]) is a page full of
//!   `hvc #0` — so the syscall bounces straight out to the host as an HVC exit.
//!   [`HvfVcpu::set_syscall_ret`] then emulates the `eret` back to EL0 from the
//!   `ELR_EL1`/`SPSR_EL1` captured at the trap.
//! * A guest access outside the mapped region is a stage-2 abort that exits to
//!   the host directly (no EL1 stub) as [`Exit::MemFault`], driving the kernel's
//!   existing fault path.

use super::stub;
use super::sys::{
    self, HV_REG_CPSR, HV_REG_PC, HV_REG_X0, HV_SUCCESS, HV_SYS_REG_CPACR_EL1, HV_SYS_REG_ELR_EL1,
    HV_SYS_REG_ESR_EL1, HV_SYS_REG_SCTLR_EL1, HV_SYS_REG_SP_EL0, HV_SYS_REG_SPSR_EL1,
    HV_SYS_REG_TPIDR_EL0, HV_SYS_REG_VBAR_EL1, hv_reg_t, hv_sys_reg_t, hv_vcpu_exit_t, hv_vcpu_t,
};
use super::vm::vm;
use crate::vcpu::{Exit, GuestMemory, Vcpu, VcpuError};

/// `CPSR`/`SPSR` value for EL0t with `DAIF` masked (no interrupts): mode
/// `EL0t` (`M[3:0]=0`) plus `D`,`A`,`I`,`F` set (`0x3C0`).
const CPSR_EL0T: u64 = 0x3C0;
/// `SCTLR_EL1` with the MMU and caches off and all `RES1` bits set (the ARMv8
/// reset shape) — guest VA == IPA, host stores to the mapped region are seen.
const SCTLR_EL1_MMU_OFF: u64 = 0x30D0_0800;
/// `CPACR_EL1.FPEN = 0b11`: don't trap FP/SIMD at EL0/EL1 (musl uses SIMD).
const CPACR_EL1_FP_ON: u64 = 0x30_0000;

/// Stage-2 permissions for the whole guest region in this milestone (per-page
/// W^X is a later milestone; here the region is uniformly RWX).
const RWX: u64 = sys::HV_MEMORY_READ | sys::HV_MEMORY_WRITE | sys::HV_MEMORY_EXEC;

pub struct HvfVcpu {
    vcpu: hv_vcpu_t,
    /// Framework-owned exit record, updated by each `hv_vcpu_run`.
    exit: *mut hv_vcpu_exit_t,
    /// Backing generation currently mapped into the VM (`None` = nothing mapped
    /// by this vcpu yet).
    mapped_gen: Option<u64>,
    /// The `(ipa, size)` window this vcpu currently has mapped, so it can unmap
    /// on remap and on drop (the single IPA space is shared, so leaving a stale
    /// mapping would collide with the next vcpu/process to use that window).
    mapped_window: Option<(u64, usize)>,
    /// EL0 resume context captured at the last syscall trap, applied by
    /// `set_syscall_ret` to emulate the `eret` the stub would have done.
    el0_pc: u64,
    el0_cpsr: u64,
}

// SAFETY: `hv_vcpu` handles are thread-bound and this milestone runs the guest
// only on the serial scheduler (one thread), so the vcpu never actually moves
// threads. The `Send` bound is required by the `Vcpu` trait; honoring the
// single-thread invariant is what makes it sound. (SMP thread-affinity is a
// later milestone.)
unsafe impl Send for HvfVcpu {}

impl HvfVcpu {
    /// Create a vcpu for a fresh thread, entering at `entry` with stack `stack`.
    /// Returns a boxed trait object (a factory, not a `Self` constructor).
    #[allow(clippy::new_ret_no_self)]
    pub fn new(entry: u64, stack: u64) -> Result<Box<dyn Vcpu>, VcpuError> {
        let vm = vm()?;
        let stub_ipa = stub::ensure_mapped(vm)?;

        let mut vcpu: hv_vcpu_t = 0;
        let mut exit: *mut hv_vcpu_exit_t = std::ptr::null_mut();
        // SAFETY: valid out-pointers; null config selects defaults.
        let ret = unsafe { sys::hv_vcpu_create(&raw mut vcpu, &raw mut exit, std::ptr::null()) };
        if ret != HV_SUCCESS {
            return Err(VcpuError::Backend(format!(
                "hv_vcpu_create failed (status {ret:#x})"
            )));
        }
        let mut v = Self {
            vcpu,
            exit,
            mapped_gen: None,
            mapped_window: None,
            el0_pc: 0,
            el0_cpsr: 0,
        };
        v.init_regs(entry, stack, stub_ipa);
        Ok(Box::new(v))
    }

    /// Establish the fixed run configuration and the entry PC/SP.
    fn init_regs(&mut self, entry: u64, stack: u64, stub_ipa: u64) {
        self.set_reg_raw(HV_REG_PC, entry);
        self.set_reg_raw(HV_REG_CPSR, CPSR_EL0T);
        self.set_sys(HV_SYS_REG_SP_EL0, stack);
        self.set_sys(HV_SYS_REG_VBAR_EL1, stub_ipa);
        self.set_sys(HV_SYS_REG_SCTLR_EL1, SCTLR_EL1_MMU_OFF);
        self.set_sys(HV_SYS_REG_CPACR_EL1, CPACR_EL1_FP_ON);
        self.set_sys(HV_SYS_REG_TPIDR_EL0, 0);
    }

    fn get_reg(&self, reg: hv_reg_t) -> u64 {
        let mut v = 0u64;
        // SAFETY: `self.vcpu` is a live handle for this thread; `reg` is valid.
        let ret = unsafe { sys::hv_vcpu_get_reg(self.vcpu, reg, &raw mut v) };
        debug_assert_eq!(ret, HV_SUCCESS, "hv_vcpu_get_reg({reg})");
        v
    }
    fn set_reg_raw(&self, reg: hv_reg_t, val: u64) {
        // SAFETY: as `get_reg`.
        let ret = unsafe { sys::hv_vcpu_set_reg(self.vcpu, reg, val) };
        debug_assert_eq!(ret, HV_SUCCESS, "hv_vcpu_set_reg({reg})");
    }
    fn get_sys(&self, reg: hv_sys_reg_t) -> u64 {
        let mut v = 0u64;
        // SAFETY: as `get_reg`.
        let ret = unsafe { sys::hv_vcpu_get_sys_reg(self.vcpu, reg, &raw mut v) };
        debug_assert_eq!(ret, HV_SUCCESS, "hv_vcpu_get_sys_reg({reg})");
        v
    }
    fn set_sys(&self, reg: hv_sys_reg_t, val: u64) {
        // SAFETY: as `get_reg`.
        let ret = unsafe { sys::hv_vcpu_set_sys_reg(self.vcpu, reg, val) };
        debug_assert_eq!(ret, HV_SUCCESS, "hv_vcpu_set_sys_reg({reg})");
    }

    /// (Re)map the guest region into the VM if the backing changed since the
    /// last run — the seam that makes `fork`/`execve` (which hand `run` a new
    /// backing) and a future context switch just work.
    fn reconcile(&mut self, mem: &GuestMemory) -> Result<(), VcpuError> {
        let generation = mem.backing_generation();
        if self.mapped_gen == Some(generation) {
            return Ok(());
        }
        let vm = vm()?;
        let (ipa, size) = (mem.base(), mem.size() as usize);
        if let Some((old_ipa, old_size)) = self.mapped_window.take() {
            // A prior backing occupies its IPA window; drop it first.
            vm.unmap(old_ipa, old_size)?;
        }
        vm.map(mem.host_base(), ipa, size, RWX)?;
        // The host just wrote guest code into this region; make it fetch-coherent.
        // SAFETY: `[host_base, host_base+size)` is the live region we mapped.
        unsafe { sys::sys_icache_invalidate(mem.host_base().cast(), size) };
        self.mapped_gen = Some(generation);
        self.mapped_window = Some((ipa, size));
        Ok(())
    }

    /// Decode one guest exit into an [`Exit`].
    fn decode_exit(&mut self) -> Exit {
        // SAFETY: `self.exit` points at framework-owned storage, valid and freshly
        // written by the `hv_vcpu_run` that just returned.
        let ex = unsafe { *self.exit };
        match ex.reason {
            sys::HV_EXIT_REASON_EXCEPTION => {
                let ec = (ex.exception.syndrome >> 26) & 0x3f;
                match ec {
                    // HVC from the EL1 stub — a guest EL0 `svc`. Confirm via the
                    // EL1 syndrome and capture the EL0 return context.
                    0x16 => {
                        self.el0_pc = self.get_sys(HV_SYS_REG_ELR_EL1);
                        self.el0_cpsr = self.get_sys(HV_SYS_REG_SPSR_EL1);
                        let el1_ec = (self.get_sys(HV_SYS_REG_ESR_EL1) >> 26) & 0x3f;
                        if el1_ec == 0x15 {
                            Exit::Syscall
                        } else {
                            // Some other EL1 exception reached the stub.
                            Exit::IllegalInstruction {
                                pc: self.el0_pc.wrapping_sub(4),
                            }
                        }
                    }
                    // Stage-2 abort (guest touched an unmapped IPA): data (0x24)
                    // or instruction (0x20/0x21) abort from a lower EL.
                    0x24 | 0x20 | 0x21 => {
                        let write = ec == 0x24 && (ex.exception.syndrome >> 6) & 1 == 1;
                        Exit::MemFault {
                            addr: ex.exception.physical_address,
                            write,
                        }
                    }
                    _ => Exit::IllegalInstruction {
                        pc: self.get_reg(HV_REG_PC),
                    },
                }
            }
            sys::HV_EXIT_REASON_CANCELED | sys::HV_EXIT_REASON_VTIMER => Exit::Interrupted,
            _ => Exit::IllegalInstruction {
                pc: self.get_reg(HV_REG_PC),
            },
        }
    }
}

impl Vcpu for HvfVcpu {
    fn run(&mut self, mem: &mut GuestMemory) -> Result<Exit, VcpuError> {
        self.reconcile(mem)?;
        // SAFETY: created on and run from the same (serial-scheduler) thread.
        let ret = unsafe { sys::hv_vcpu_run(self.vcpu) };
        if ret != HV_SUCCESS {
            return Err(VcpuError::Backend(format!(
                "hv_vcpu_run failed (status {ret:#x})"
            )));
        }
        Ok(self.decode_exit())
    }

    fn syscall_nr(&self) -> u64 {
        self.get_reg(HV_REG_X0 + 8)
    }

    fn syscall_args(&self) -> [u64; 6] {
        [
            self.get_reg(HV_REG_X0),
            self.get_reg(HV_REG_X0 + 1),
            self.get_reg(HV_REG_X0 + 2),
            self.get_reg(HV_REG_X0 + 3),
            self.get_reg(HV_REG_X0 + 4),
            self.get_reg(HV_REG_X0 + 5),
        ]
    }

    fn set_syscall_ret(&mut self, value: u64) {
        // Emulate the stub's `eret`: return value in x0, resume the guest at the
        // instruction after the `svc` (ELR_EL1) back in EL0 (SPSR_EL1).
        self.set_reg_raw(HV_REG_X0, value);
        self.set_reg_raw(HV_REG_PC, self.el0_pc);
        self.set_reg_raw(HV_REG_CPSR, self.el0_cpsr);
    }

    fn reg(&self, idx: usize) -> u64 {
        self.get_reg(HV_REG_X0 + idx as u32)
    }
    fn set_reg(&mut self, idx: usize, value: u64) {
        self.set_reg_raw(HV_REG_X0 + idx as u32, value);
    }

    fn pc(&self) -> u64 {
        self.get_reg(HV_REG_PC)
    }
    fn set_pc(&mut self, pc: u64) {
        self.set_reg_raw(HV_REG_PC, pc);
    }
    fn sp(&self) -> u64 {
        self.get_sys(HV_SYS_REG_SP_EL0)
    }
    fn set_sp(&mut self, sp: u64) {
        self.set_sys(HV_SYS_REG_SP_EL0, sp);
    }
    fn set_tls(&mut self, value: u64) {
        self.set_sys(HV_SYS_REG_TPIDR_EL0, value);
    }

    fn fork(&self) -> Box<dyn Vcpu> {
        // Create a sibling vcpu and copy the full guest register context. The
        // child re-maps its own (forked) memory on its first `run`.
        let vm = vm().expect("process VM exists (parent already created one)");
        let stub_ipa = stub::ensure_mapped(vm).expect("stub already mapped");
        let mut vcpu: hv_vcpu_t = 0;
        let mut exit: *mut hv_vcpu_exit_t = std::ptr::null_mut();
        // SAFETY: valid out-pointers; null config selects defaults.
        let ret = unsafe { sys::hv_vcpu_create(&raw mut vcpu, &raw mut exit, std::ptr::null()) };
        assert_eq!(ret, HV_SUCCESS, "hv_vcpu_create (fork)");
        let child = Self {
            vcpu,
            exit,
            mapped_gen: None,
            mapped_window: None,
            el0_pc: self.el0_pc,
            el0_cpsr: self.el0_cpsr,
        };
        // General-purpose x0..x30 plus PC/CPSR.
        for r in 0..=30 {
            child.set_reg_raw(HV_REG_X0 + r, self.get_reg(HV_REG_X0 + r));
        }
        child.set_reg_raw(HV_REG_PC, self.get_reg(HV_REG_PC));
        child.set_reg_raw(HV_REG_CPSR, self.get_reg(HV_REG_CPSR));
        // System state that configures execution.
        child.set_sys(HV_SYS_REG_SP_EL0, self.get_sys(HV_SYS_REG_SP_EL0));
        child.set_sys(HV_SYS_REG_TPIDR_EL0, self.get_sys(HV_SYS_REG_TPIDR_EL0));
        child.set_sys(HV_SYS_REG_VBAR_EL1, stub_ipa);
        child.set_sys(HV_SYS_REG_SCTLR_EL1, SCTLR_EL1_MMU_OFF);
        child.set_sys(HV_SYS_REG_CPACR_EL1, CPACR_EL1_FP_ON);
        Box::new(child)
    }

    fn reset(&mut self, entry: u64, sp: u64) {
        for r in 0..=30 {
            self.set_reg_raw(HV_REG_X0 + r, 0);
        }
        let stub_ipa = stub::ensure_mapped(vm().expect("process VM exists")).expect("stub mapped");
        self.init_regs(entry, sp, stub_ipa);
        self.el0_pc = 0;
        self.el0_cpsr = 0;
    }
}

impl Drop for HvfVcpu {
    fn drop(&mut self) {
        // Release this vcpu's IPA window so the shared address space doesn't
        // accumulate stale mappings (which would collide with the next user of
        // that window).
        if let Some((ipa, size)) = self.mapped_window.take()
            && let Ok(vm) = vm()
        {
            let _ = vm.unmap(ipa, size);
        }
        // SAFETY: destroying the vcpu on its owning thread; the handle is valid
        // and used nowhere after this.
        unsafe { sys::hv_vcpu_destroy(self.vcpu) };
    }
}
