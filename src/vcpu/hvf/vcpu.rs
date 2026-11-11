//! The HVF virtual CPU: one guest thread's EL0 state, run on whichever host
//! thread the scheduler hands it to.
//!
//! Execution model (MMU on, guest at EL0):
//! * Guest RAM is the shared physical pool, mapped once into the VM's IPA space
//!   at the pool's window ([`super::vm::ensure_pool`]). Each process has real
//!   arm64 stage-1 page tables in that pool; `TTBR0_EL1` names the running
//!   process's root, so a context switch is one register write (plus a TLB
//!   flush, below). Stage 2 maps the pool RWX; stage 1 carries every
//!   protection, so a store to a read-only page or a fetch from a
//!   non-executable one faults exactly as on Linux.
//! * Every exception the guest takes to EL1 vectors (`VBAR_EL1`) to the control
//!   block's vector page ([`crate::vcpu::ctrl::VECTORS_VA`], in the upper half:
//!   `TTBR1_EL1`, tables shared by every process and never changed), where each
//!   slot is an `hvc` that hands it straight to the host. The host reads
//!   `ESR_EL1`/`ELR_EL1`/`FAR_EL1` and turns it into an [`Exit`]: `svc` is
//!   [`Exit::Syscall`], a translation or permission fault [`Exit::MemFault`],
//!   and so on. Returning to EL0 is an `eret` the host does itself: it writes
//!   the user `PC`/`CPSR` back, so nothing ever runs at EL1 but those `hvc`s
//!   and the flush stub.
//! * When the host has changed page tables under the guest (unmap, protect,
//!   copy-on-write, a new address space on this vcpu) or put code in its memory,
//!   the vcpu re-enters at EL1 through the flush stub, which invalidates every
//!   TLB and instruction cache entry of the VM and `eret`s to the user context.
//! * A compute-bound guest is preempted by its own virtual timer: armed for the
//!   quantum before each entry, it ends the run with
//!   `HV_EXIT_REASON_VTIMER_ACTIVATED` → [`Exit::Interrupted`], at an EL0
//!   instruction boundary where the kernel can deliver a pending signal.
//!
//! `hv_vcpu`s are bound to the thread that created them, while the SMP
//! scheduler moves tasks between worker threads (and `fork` makes a vcpu on the
//! parent's thread for a child that runs elsewhere). So an [`HvfVcpu`] holds no
//! `hv_vcpu`: it is the guest thread's register state, cached host-side, and
//! each host thread keeps one `hv_vcpu` of its own ([`Slot`]) that runs
//! whichever `HvfVcpu` is handed to it, reloading the registers when the last
//! one it ran was another.

use super::sys::{
    self, HV_REG_CPSR, HV_REG_FPCR, HV_REG_FPSR, HV_REG_PC, HV_REG_X0, HV_SIMD_FP_REG_Q0,
    HV_SUCCESS, HV_SYS_REG_CNTKCTL_EL1, HV_SYS_REG_CNTV_CTL_EL0, HV_SYS_REG_CNTV_CVAL_EL0,
    HV_SYS_REG_CPACR_EL1, HV_SYS_REG_ELR_EL1, HV_SYS_REG_ESR_EL1, HV_SYS_REG_FAR_EL1,
    HV_SYS_REG_MAIR_EL1, HV_SYS_REG_MDSCR_EL1, HV_SYS_REG_SCTLR_EL1, HV_SYS_REG_SP_EL0,
    HV_SYS_REG_SP_EL1, HV_SYS_REG_SPSR_EL1, HV_SYS_REG_TCR_EL1, HV_SYS_REG_TPIDR_EL0,
    HV_SYS_REG_TPIDRRO_EL0, HV_SYS_REG_TTBR0_EL1, HV_SYS_REG_TTBR1_EL1, HV_SYS_REG_VBAR_EL1,
    Simd16, hv_reg_t, hv_sys_reg_t, hv_vcpu_exit_t, hv_vcpu_t,
};
use super::vm::{ensure_pool, vm};
use crate::vcpu::ctrl::{VECTORS_VA, vectors};
use crate::vcpu::{Exit, GuestMemory, PtFormat, Vcpu, VcpuError};
use std::cell::RefCell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// `SCTLR_EL1`: the `RES1` bits plus `M` (MMU on), `C`/`I` (caches), `SA`/`SA0`
/// (SP alignment checks, as Linux), and EL0 permissions Linux grants: `DZE`
/// (`DC ZVA`), `UCT` (`CTR_EL0`), `UCI` (cache maintenance by VA), `nTWI`/`nTWE`
/// (`WFI`/`WFE` don't trap to EL1).
const SCTLR_EL1: u64 = 0x30D0_0800 // RES1
    | 1 // M
    | 1 << 2 // C
    | 1 << 3 // SA
    | 1 << 4 // SA0
    | 1 << 12 // I
    | 1 << 14 // DZE
    | 1 << 15 // UCT
    | 1 << 16 // nTWI
    | 1 << 18 // nTWE
    | 1 << 26; // UCI
/// `TCR_EL1`: `TTBR0` (the process) walks 48-bit VAs (`T0SZ` = 16, 4 levels
/// from level 0) and `TTBR1` (the vector page) 39-bit ones, both with a 4 KiB
/// granule, tables inner-shareable write-back cached; `IPS` = 48 bits (the
/// hardware clamps it to what it implements).
const TCR_EL1: u64 = 16 // T0SZ
    | 0b01 << 8 // IRGN0: write-back, write-allocate
    | 0b01 << 10 // ORGN0
    | 0b11 << 12 // SH0: inner shareable
    | crate::vcpu::ctrl::ARM_T1SZ << 16 // T1SZ
    | 0b01 << 24 // IRGN1
    | 0b01 << 26 // ORGN1
    | 0b11 << 28 // SH1
    | 0b10 << 30 // TG1: 4 KiB
    | 0b101 << 32; // IPS
/// `MAIR_EL1`: attribute 0 (every descriptor's `AttrIndx`) is normal memory,
/// inner and outer write-back non-transient, read- and write-allocate.
const MAIR_EL1: u64 = 0xFF;
/// `CPACR_EL1.FPEN = 0b11`: don't trap FP/SIMD at EL0/EL1.
const CPACR_EL1: u64 = 0b11 << 20;
/// `CNTKCTL_EL1.EL0VCTEN`: EL0 reads the virtual counter (and `CNTFRQ_EL0`), as
/// on Linux; its timer registers stay out of reach.
const CNTKCTL_EL1: u64 = 1 << 1;
/// `CPSR` for EL1h with `DAIF` masked: how the flush stub is entered.
const CPSR_EL1H: u64 = 0x3C5;
/// `CNTV_CTL_EL0.ENABLE` with `IMASK` clear.
const VTIMER_ON: u64 = 1;

/// The architectural user state of one guest thread.
#[derive(Clone)]
struct Regs {
    x: [u64; 31],
    pc: u64,
    /// `PSTATE` as EL0 sees it (`SPSR` layout): `NZCV`, `DIT`, … mode EL0t.
    pstate: u64,
    sp: u64,
    tpidr: u64,
    fpcr: u64,
    fpsr: u64,
    v: [u128; 32],
}

impl Regs {
    fn new(entry: u64, sp: u64) -> Self {
        Self {
            x: [0; 31],
            pc: entry,
            pstate: 0,
            sp,
            tpidr: 0,
            fpcr: 0,
            fpsr: 0,
            v: [0; 32],
        }
    }
}

/// Which parts of [`Regs`] the host changed since they were loaded into the
/// thread's `hv_vcpu`.
#[derive(Clone, Copy)]
struct Dirty {
    /// Bit `n`: `Xn`.
    x: u32,
    pc: bool,
    sys: bool,
    simd: bool,
}

impl Dirty {
    const ALL: Self = Self {
        x: (1 << 31) - 1,
        pc: true,
        sys: true,
        simd: true,
    };
    const NONE: Self = Self {
        x: 0,
        pc: false,
        sys: false,
        simd: false,
    };
}

/// The `hv_vcpu` a host thread runs guest threads on.
struct Slot {
    vcpu: hv_vcpu_t,
    /// Framework-owned exit record, updated by each `hv_vcpu_run`.
    exit: *const hv_vcpu_exit_t,
    /// The [`HvfVcpu::id`] whose registers are loaded, 0 for none.
    owner: u64,
    /// The `TTBR0_EL1` loaded.
    ttbr0: u64,
    /// The `TTBR1_EL1` loaded (one per pool).
    ttbr1: u64,
    /// The [`FLUSH_GEN`] its last flush covered.
    flushed: u64,
}

impl Slot {
    /// Create this thread's `hv_vcpu` with the fixed EL1 configuration.
    fn new() -> Result<Self, VcpuError> {
        vm()?;
        let mut vcpu: hv_vcpu_t = 0;
        let mut exit: *mut hv_vcpu_exit_t = std::ptr::null_mut();
        // SAFETY: valid out-pointers; null config selects defaults. The vcpu is
        // bound to this thread, which is the only one that will use it.
        let ret = unsafe { sys::hv_vcpu_create(&raw mut vcpu, &raw mut exit, std::ptr::null()) };
        if ret != HV_SUCCESS {
            return Err(VcpuError::Backend(format!(
                "hv_vcpu_create failed (status {ret:#x})"
            )));
        }
        let slot = Self {
            vcpu,
            exit,
            owner: 0,
            ttbr0: 0,
            ttbr1: 0,
            flushed: 0,
        };
        for (reg, val) in [
            (HV_SYS_REG_SCTLR_EL1, SCTLR_EL1),
            (HV_SYS_REG_TCR_EL1, TCR_EL1),
            (HV_SYS_REG_MAIR_EL1, MAIR_EL1),
            (HV_SYS_REG_CPACR_EL1, CPACR_EL1),
            (HV_SYS_REG_VBAR_EL1, VECTORS_VA),
            (HV_SYS_REG_CNTKCTL_EL1, CNTKCTL_EL1),
            (HV_SYS_REG_TTBR0_EL1, 0),
            (HV_SYS_REG_TTBR1_EL1, 0),
            (HV_SYS_REG_SP_EL1, 0),
            (HV_SYS_REG_MDSCR_EL1, 0),
            (HV_SYS_REG_TPIDRRO_EL0, 0),
            (HV_SYS_REG_CNTV_CTL_EL0, 0),
        ] {
            slot.set_sys(reg, val)?;
        }
        // The guest's virtual counter reads `mach_absolute_time()` itself, so a
        // timer deadline is computed straight from the host clock.
        // SAFETY: a live vcpu owned by this thread.
        check(
            unsafe { sys::hv_vcpu_set_vtimer_offset(vcpu, 0) },
            "hv_vcpu_set_vtimer_offset",
        )?;
        Ok(slot)
    }

    fn get(&self, reg: hv_reg_t) -> u64 {
        let mut v = 0u64;
        // SAFETY: `self.vcpu` is this thread's live vcpu; `reg` is valid.
        let ret = unsafe { sys::hv_vcpu_get_reg(self.vcpu, reg, &raw mut v) };
        debug_assert_eq!(ret, HV_SUCCESS, "hv_vcpu_get_reg({reg})");
        v
    }
    fn set(&self, reg: hv_reg_t, val: u64) {
        // SAFETY: as `get`.
        let ret = unsafe { sys::hv_vcpu_set_reg(self.vcpu, reg, val) };
        debug_assert_eq!(ret, HV_SUCCESS, "hv_vcpu_set_reg({reg})");
    }
    fn get_sys(&self, reg: hv_sys_reg_t) -> u64 {
        let mut v = 0u64;
        // SAFETY: as `get`.
        let ret = unsafe { sys::hv_vcpu_get_sys_reg(self.vcpu, reg, &raw mut v) };
        debug_assert_eq!(ret, HV_SUCCESS, "hv_vcpu_get_sys_reg({reg:#x})");
        v
    }
    fn set_sys(&self, reg: hv_sys_reg_t, val: u64) -> Result<(), VcpuError> {
        // SAFETY: as `get`.
        let ret = unsafe { sys::hv_vcpu_set_sys_reg(self.vcpu, reg, val) };
        check(ret, "hv_vcpu_set_sys_reg")
    }
    fn get_q(&self, n: u32) -> u128 {
        let mut v = Simd16::default();
        // SAFETY: as `get`; `v` is a 16-aligned 16-byte buffer.
        let ret =
            unsafe { sys::hv_vcpu_get_simd_fp_reg(self.vcpu, HV_SIMD_FP_REG_Q0 + n, &raw mut v) };
        debug_assert_eq!(ret, HV_SUCCESS, "hv_vcpu_get_simd_fp_reg");
        u128::from_le_bytes(v.0)
    }
    fn set_q(&self, n: u32, val: u128) {
        // SAFETY: as `get`.
        let ret =
            unsafe { sys::set_simd_fp_reg(self.vcpu, HV_SIMD_FP_REG_Q0 + n, val.to_le_bytes()) };
        debug_assert_eq!(ret, HV_SUCCESS, "hv_vcpu_set_simd_fp_reg");
    }

    /// Load the parts of `r` that `dirty` names.
    fn push(&self, r: &Regs, dirty: Dirty) {
        for n in 0..31u32 {
            if dirty.x & (1 << n) != 0 {
                self.set(HV_REG_X0 + n, r.x[n as usize]);
            }
        }
        if dirty.pc {
            self.set(HV_REG_PC, r.pc);
            self.set(HV_REG_CPSR, r.pstate);
        }
        if dirty.sys {
            let _ = self.set_sys(HV_SYS_REG_SP_EL0, r.sp);
            let _ = self.set_sys(HV_SYS_REG_TPIDR_EL0, r.tpidr);
        }
        if dirty.simd {
            self.set(HV_REG_FPCR, r.fpcr);
            self.set(HV_REG_FPSR, r.fpsr);
            for n in 0..32u32 {
                self.set_q(n, r.v[n as usize]);
            }
        }
    }

    /// Read the guest's user state back after an exit. `el1` says the exit
    /// came from a vector slot at EL1, where `ELR_EL1`/`SPSR_EL1` hold the user
    /// `PC`/`PSTATE` (the vector code touches nothing else).
    fn pull(&self, r: &mut Regs, el1: bool) {
        for n in 0..31u32 {
            r.x[n as usize] = self.get(HV_REG_X0 + n);
        }
        if el1 {
            r.pc = self.get_sys(HV_SYS_REG_ELR_EL1);
            r.pstate = self.get_sys(HV_SYS_REG_SPSR_EL1);
        } else {
            r.pc = self.get(HV_REG_PC);
            r.pstate = self.get(HV_REG_CPSR);
        }
        r.sp = self.get_sys(HV_SYS_REG_SP_EL0);
        r.tpidr = self.get_sys(HV_SYS_REG_TPIDR_EL0);
        r.fpcr = self.get(HV_REG_FPCR);
        r.fpsr = self.get(HV_REG_FPSR);
        for n in 0..32u32 {
            r.v[n as usize] = self.get_q(n);
        }
    }

    /// Arm the guest's virtual timer to end the run after `quantum`.
    fn arm_timer(&self, quantum: Duration) {
        let deadline = now_ticks().wrapping_add(ns_to_ticks(quantum));
        let _ = self.set_sys(HV_SYS_REG_CNTV_CVAL_EL0, deadline);
        let _ = self.set_sys(HV_SYS_REG_CNTV_CTL_EL0, VTIMER_ON);
        // SAFETY: a live vcpu owned by this thread.
        unsafe { sys::hv_vcpu_set_vtimer_mask(self.vcpu, false) };
    }
}

impl Drop for Slot {
    fn drop(&mut self) {
        // SAFETY: thread-local destructors run on the owning thread; the handle
        // is valid and used nowhere after this.
        unsafe { sys::hv_vcpu_destroy(self.vcpu) };
    }
}

thread_local! {
    /// This thread's `hv_vcpu`, created on its first guest run.
    static SLOT: RefCell<Option<Slot>> = const { RefCell::new(None) };
}

fn check(ret: i32, what: &str) -> Result<(), VcpuError> {
    if ret == HV_SUCCESS {
        Ok(())
    } else {
        Err(VcpuError::Backend(format!(
            "{what} failed (status {ret:#x})"
        )))
    }
}

/// The host tick count the guest's virtual counter follows.
fn now_ticks() -> u64 {
    // SAFETY: no arguments, no side effects.
    unsafe { sys::mach_absolute_time() }
}

/// `d` in `mach_absolute_time` ticks (at least one).
fn ns_to_ticks(d: Duration) -> u64 {
    static TB: OnceLock<(u64, u64)> = OnceLock::new();
    let &(numer, denom) = TB.get_or_init(|| {
        let mut tb = sys::mach_timebase_info_data_t::default();
        // SAFETY: a valid out-pointer.
        unsafe { sys::mach_timebase_info(&raw mut tb) };
        if tb.numer == 0 || tb.denom == 0 {
            (1, 1)
        } else {
            (u64::from(tb.numer), u64::from(tb.denom))
        }
    });
    let ticks = d.as_nanos() * u128::from(denom) / u128::from(numer);
    (ticks as u64).max(1)
}

/// Distinguishes [`HvfVcpu`]s for [`Slot::owner`].
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Bumped whenever the host changes page tables or code under some guest. A
/// thread's `hv_vcpu` that hasn't flushed since enters through the flush stub:
/// the vcpu that asked may never run again (a thread that unmaps and exits),
/// while a sibling elsewhere still holds the old translations.
static FLUSH_GEN: AtomicU64 = AtomicU64::new(1);

/// Ask every thread's `hv_vcpu` to flush before it next enters the guest.
fn request_flush() {
    FLUSH_GEN.fetch_add(1, Ordering::Release);
}

pub struct HvfVcpu {
    /// Unique among this process's vcpus; never 0.
    id: u64,
    regs: Regs,
    /// What changed since the registers were loaded into a slot.
    dirty: Dirty,
    /// Trapped in a syscall whose return value has not been written yet. While
    /// set, [`Vcpu::run`] re-delivers [`Exit::Syscall`] without entering the
    /// guest (a blocked syscall is retried by running again), and `pc` sits on
    /// the `svc`, as the interpreter has it.
    in_syscall: bool,
    /// Invalidate the VM's TLBs and instruction caches before the next entry.
    flush: bool,
    /// The `TTBR0_EL1` of the address space to run, set by `reconcile`.
    ttbr0: u64,
    /// Its pool's `TTBR1_EL1` (the vector page).
    ttbr1: u64,
    /// The preemption quantum ([`crate::vcpu::preempt_quantum`]).
    quantum: Option<Duration>,
}

impl HvfVcpu {
    /// Create a vcpu for a fresh thread, entering at `entry` with stack `stack`.
    /// Returns a boxed trait object (a factory, not a `Self` constructor).
    #[allow(clippy::new_ret_no_self)]
    pub fn new(entry: u64, stack: u64) -> Result<Box<dyn Vcpu>, VcpuError> {
        vm()?;
        Ok(Box::new(Self::with_regs(Regs::new(entry, stack))))
    }

    fn with_regs(regs: Regs) -> Self {
        Self {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            regs,
            dirty: Dirty::ALL,
            in_syscall: false,
            flush: true,
            ttbr0: 0,
            ttbr1: 0,
            quantum: crate::vcpu::preempt_quantum(),
        }
    }

    /// Run on this thread's slot until an [`Exit`].
    fn run_slot(&mut self, slot: &mut Slot) -> Result<Exit, VcpuError> {
        if slot.owner != self.id {
            slot.owner = self.id;
            self.dirty = Dirty::ALL;
        }
        if slot.ttbr1 != self.ttbr1 {
            slot.set_sys(HV_SYS_REG_TTBR1_EL1, self.ttbr1)?;
            slot.ttbr1 = self.ttbr1;
            self.flush = true;
        }
        if slot.ttbr0 != self.ttbr0 {
            slot.set_sys(HV_SYS_REG_TTBR0_EL1, self.ttbr0)?;
            slot.ttbr0 = self.ttbr0;
            // Another address space's translations may be cached.
            self.flush = true;
        }
        loop {
            // Load what the host changed (all of it for a vcpu new to this
            // slot); after an exit handled here, that's the user `PC`/`CPSR` —
            // the `eret` back from the vector page.
            let mut dirty = std::mem::replace(&mut self.dirty, Dirty::NONE);
            let generation = FLUSH_GEN.load(Ordering::Acquire);
            if slot.flushed != generation {
                self.flush = true;
            }
            if self.flush {
                slot.flushed = generation;
                // Enter through the flush stub, which `eret`s to the user context.
                dirty.pc = false;
                slot.set_sys(HV_SYS_REG_ELR_EL1, self.regs.pc)?;
                slot.set_sys(HV_SYS_REG_SPSR_EL1, self.regs.pstate)?;
                slot.set(HV_REG_PC, VECTORS_VA + vectors::FLUSH);
                slot.set(HV_REG_CPSR, CPSR_EL1H);
                self.flush = false;
            }
            slot.push(&self.regs, dirty);
            if let Some(q) = self.quantum {
                slot.arm_timer(q);
            }
            // SAFETY: the slot's vcpu belongs to this thread.
            let ret = unsafe { sys::hv_vcpu_run(slot.vcpu) };
            check(ret, "hv_vcpu_run")?;
            // SAFETY: framework-owned storage, written by the run that returned.
            let ex = unsafe { *slot.exit };
            match ex.reason {
                sys::HV_EXIT_REASON_EXCEPTION => {
                    if let Some(exit) = self.exception(slot, ex.exception)? {
                        return Ok(exit);
                    }
                }
                sys::HV_EXIT_REASON_VTIMER_ACTIVATED | sys::HV_EXIT_REASON_CANCELED => {
                    // Stopped in the vector page (between a trap and its `hvc`,
                    // or in the flush stub): carry on — it's a few instructions
                    // from EL0 or the host.
                    if slot.get(HV_REG_CPSR) & 0xF != 0 {
                        continue;
                    }
                    slot.pull(&mut self.regs, false);
                    return Ok(Exit::Interrupted);
                }
                other => {
                    return Err(VcpuError::Backend(format!(
                        "unexpected hv_vcpu_run exit reason {other}"
                    )));
                }
            }
        }
    }

    /// Decode an exception exit: `Some(exit)` for the kernel, `None` to resume
    /// (handled here).
    fn exception(
        &mut self,
        slot: &Slot,
        ex: sys::hv_vcpu_exit_exception_t,
    ) -> Result<Option<Exit>, VcpuError> {
        let ec = (ex.syndrome >> 26) & 0x3f;
        // Only the vector page's `hvc`s leave EL1 for the host; anything else
        // there would be a fault in it, not user state to hand the kernel.
        if ec != 0x16 && slot.get(HV_REG_CPSR) & 0xF != 0 {
            return Err(VcpuError::Backend(format!(
                "exception at EL1 (syndrome {:#x}, PC {:#x}, VA {:#x})",
                ex.syndrome,
                slot.get(HV_REG_PC),
                ex.virtual_address
            )));
        }
        match ec {
            // An `hvc` from the vector page: the guest took an exception to EL1.
            0x16 => {
                let slot_no = (ex.syndrome & 0xffff) as u16;
                slot.pull(&mut self.regs, true);
                // The vcpu sits at EL1 in the vector page; the next entry
                // returns to the user context instead.
                self.dirty.pc = true;
                match slot_no {
                    vectors::LOWER_SYNC => {
                        let esr = slot.get_sys(HV_SYS_REG_ESR_EL1);
                        let far = slot.get_sys(HV_SYS_REG_FAR_EL1);
                        Ok(self.user_exception(esr, far))
                    }
                    // IRQ/FIQ from EL0: nothing is ever injected; resume.
                    9 | 10 => Ok(None),
                    _ => Err(VcpuError::Backend(format!(
                        "guest exception in vector slot {slot_no} (ESR_EL1 {:#x}, ELR_EL1 {:#x}, FAR_EL1 {:#x})",
                        slot.get_sys(HV_SYS_REG_ESR_EL1),
                        self.regs.pc,
                        slot.get_sys(HV_SYS_REG_FAR_EL1)
                    ))),
                }
            }
            // `WFI`/`WFE` trapped to the host: the guest is waiting, so give
            // the CPU up.
            0x01 => {
                slot.pull(&mut self.regs, false);
                self.regs.pc += 4;
                self.dirty.pc = true;
                Ok(Some(Exit::Interrupted))
            }
            // A system-register access the hypervisor traps.
            0x18 => {
                slot.pull(&mut self.regs, false);
                Ok(self.sysreg_access(ex.syndrome))
            }
            // A stage-2 abort: the guest's tables point outside its pool.
            0x20 | 0x24 => {
                slot.pull(&mut self.regs, false);
                Ok(Some(Exit::MemFault {
                    addr: ex.virtual_address,
                    write: ec == 0x24 && (ex.syndrome >> 6) & 1 == 1,
                }))
            }
            _ => {
                slot.pull(&mut self.regs, false);
                Ok(Some(Exit::IllegalInstruction { pc: self.regs.pc }))
            }
        }
    }

    /// What a synchronous exception the guest took at EL0 (`ESR_EL1`, with
    /// `FAR_EL1` = `far`) means. `self.regs.pc` holds `ELR_EL1`.
    fn user_exception(&mut self, esr: u64, far: u64) -> Option<Exit> {
        let ec = (esr >> 26) & 0x3f;
        let iss = esr & 0x1ff_ffff;
        let pc = self.regs.pc;
        Some(match ec {
            // `svc`: ELR is past it; the interpreter keeps pc on it until the
            // return value is written.
            0x15 => {
                self.regs.pc = pc.wrapping_sub(4);
                self.in_syscall = true;
                Exit::Syscall
            }
            // Data abort.
            0x24 => {
                let write = iss & (1 << 6) != 0;
                if iss & 0x3f == 0x21 {
                    Exit::Misaligned { addr: far, write }
                } else {
                    Exit::MemFault { addr: far, write }
                }
            }
            // Instruction abort.
            0x20 => Exit::MemFault {
                addr: far,
                write: false,
            },
            // PC / SP alignment.
            0x22 => Exit::Misaligned {
                addr: far,
                write: false,
            },
            0x26 => Exit::Misaligned {
                addr: self.regs.sp,
                write: false,
            },
            // `brk #imm` (ELR is on it).
            0x3C => Exit::Breakpoint {
                pc,
                code: iss & 0xffff,
            },
            // `mrs`/`msr` of a register EL0 can't reach.
            0x18 => return self.sysreg_access(esr),
            _ => Exit::IllegalInstruction { pc },
        })
    }

    /// Emulate a trapped `mrs` the way Linux does for EL0 (the CPU
    /// identification registers, `HWCAP_CPUID`): write the value and step past
    /// it (`None`: resume). Anything else is undefined (`SIGILL`).
    fn sysreg_access(&mut self, esr: u64) -> Option<Exit> {
        let iss = esr & 0x1ff_ffff;
        let read = iss & 1 == 1;
        let rt = ((iss >> 5) & 0x1f) as usize;
        let key = (((iss >> 20) & 3) << 14) // op0
            | (((iss >> 14) & 7) << 11) // op1
            | (((iss >> 10) & 0xf) << 7) // CRn
            | (((iss >> 1) & 0xf) << 3) // CRm
            | ((iss >> 17) & 7); // op2
        if read && let Some(v) = crate::vcpu::interp::el0_id_register(key as u32) {
            if rt != 31 {
                self.regs.x[rt] = v;
                self.dirty.x |= 1 << rt;
            }
            self.regs.pc += 4;
            self.dirty.pc = true;
            return None;
        }
        Some(Exit::IllegalInstruction { pc: self.regs.pc })
    }

    /// Point the vcpu at `mem`'s address space, mapping its pool if this is the
    /// first run on it, and note what the host changed under the guest.
    fn sync(&mut self, mem: &mut GuestMemory) -> Result<(), VcpuError> {
        if mem.page_table_format() != PtFormat::Aarch64 {
            mem.set_guest_arch(crate::abi::Arch::Aarch64)
                .map_err(VcpuError::Mem)?;
        }
        let base = ensure_pool(mem)?;
        self.ttbr0 = base + mem.cr3();
        self.ttbr1 = mem.arm_ttbr1();
        if mem.take_tlb_dirty() | mem.take_icache_dirty() {
            self.flush = true;
            request_flush();
        }
        Ok(())
    }
}

impl Vcpu for HvfVcpu {
    fn run(&mut self, mem: &mut GuestMemory) -> Result<Exit, VcpuError> {
        if self.in_syscall {
            return Ok(Exit::Syscall);
        }
        self.sync(mem)?;
        self.run_bare()
    }

    fn needs_locked_run(&self) -> bool {
        // The guest runs against the mapped pool; nothing in the run borrows
        // the `GuestMemory`.
        false
    }

    fn reconcile(&mut self, mem: &mut GuestMemory) -> Result<(), VcpuError> {
        if self.in_syscall {
            return Ok(());
        }
        self.sync(mem)
    }

    fn run_bare(&mut self) -> Result<Exit, VcpuError> {
        if self.in_syscall {
            return Ok(Exit::Syscall);
        }
        SLOT.with(|cell| {
            let mut cell = cell.borrow_mut();
            if cell.is_none() {
                *cell = Some(Slot::new()?);
            }
            self.run_slot(cell.as_mut().expect("slot just created"))
        })
    }

    fn syscall_nr(&self) -> u64 {
        self.regs.x[8]
    }

    fn syscall_args(&self) -> [u64; 6] {
        let x = &self.regs.x;
        [x[0], x[1], x[2], x[3], x[4], x[5]]
    }

    fn set_syscall_ret(&mut self, value: u64) {
        self.regs.x[0] = value;
        self.regs.pc = self.regs.pc.wrapping_add(4);
        self.dirty.x |= 1;
        self.dirty.pc = true;
        self.in_syscall = false;
    }

    fn reg(&self, idx: usize) -> u64 {
        if idx < 31 {
            self.regs.x[idx]
        } else {
            self.regs.sp
        }
    }
    fn set_reg(&mut self, idx: usize, value: u64) {
        if idx < 31 {
            self.regs.x[idx] = value;
            self.dirty.x |= 1 << idx;
        } else {
            self.regs.sp = value;
            self.dirty.sys = true;
        }
    }

    fn pc(&self) -> u64 {
        self.regs.pc
    }
    fn set_pc(&mut self, pc: u64) {
        self.regs.pc = pc;
        self.dirty.pc = true;
        // Redirected: whatever syscall was pending is over.
        self.in_syscall = false;
    }
    fn sp(&self) -> u64 {
        self.regs.sp
    }
    fn set_sp(&mut self, sp: u64) {
        self.regs.sp = sp;
        self.dirty.sys = true;
    }

    fn rflags(&self) -> u64 {
        // NZCV and DIT, as the interpreter models them.
        self.regs.pstate & 0xF100_0000
    }
    fn set_rflags(&mut self, value: u64) {
        // Only the condition flags and DIT are the guest's to set: the mode
        // stays EL0t.
        self.regs.pstate = value & 0xF100_0000;
        self.dirty.pc = true;
    }

    fn simd_state(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 + 32 * 16);
        out.extend_from_slice(&(self.regs.fpsr as u32).to_le_bytes());
        out.extend_from_slice(&(self.regs.fpcr as u32).to_le_bytes());
        for v in &self.regs.v {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }
    fn set_simd_state(&mut self, bytes: &[u8]) {
        if bytes.len() < 8 + 32 * 16 {
            return;
        }
        let word = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
        self.regs.fpsr = u64::from(word(0));
        self.regs.fpcr = u64::from(word(4));
        for (i, v) in self.regs.v.iter_mut().enumerate() {
            let off = 8 + i * 16;
            *v = u128::from_le_bytes(bytes[off..off + 16].try_into().unwrap());
        }
        self.dirty.simd = true;
    }

    fn set_tls(&mut self, value: u64) {
        self.regs.tpidr = value;
        self.dirty.sys = true;
    }

    fn fork(&self) -> Result<Box<dyn Vcpu>, VcpuError> {
        // Just the registers: the child gets an `hv_vcpu` from whichever thread
        // runs it, and its own address space at its first `reconcile`.
        let mut child = Self::with_regs(self.regs.clone());
        child.in_syscall = self.in_syscall;
        Ok(Box::new(child))
    }

    fn reset(&mut self, entry: u64, sp: u64) {
        self.regs = Regs::new(entry, sp);
        self.dirty = Dirty::ALL;
        self.in_syscall = false;
        // execve freed the old image's tables, which may be cached anywhere.
        self.flush = true;
        request_flush();
    }

    fn settle_syscall_return(&mut self) {
        // The host is redirecting the guest (a signal frame, `rt_sigreturn`):
        // the syscall is over. `pc` stays where it is — on the `svc` if the
        // syscall is to be restarted, past it if its result was written.
        self.in_syscall = false;
    }

    /// Every [`Exit::Interrupted`] leaves the vcpu between two user
    /// instructions (a stop inside the vector page is run on to EL0 or to the
    /// host first), so a signal can be delivered there — unless a syscall is
    /// pending.
    fn async_signal_boundary(&self) -> bool {
        !self.in_syscall
    }

    fn flush_tlb(&mut self) {
        self.flush = true;
        request_flush();
    }
}
