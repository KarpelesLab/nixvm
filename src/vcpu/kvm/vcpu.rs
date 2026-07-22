//! The KVM virtual CPU: one guest thread on a KVM vcpu fd.
//!
//! Execution model (long mode, guest at CPL3):
//! * Guest RAM ([`GuestMemory`]) is one contiguous region mapped as slot 0 of
//!   the backend's VM at `guest_phys == base`; the fixed identity page tables
//!   in the control block (see [`super::vm`]) make guest virtual == guest
//!   physical, so the flat address space the interpreter models maps 1:1 onto
//!   hardware. The VM re-issues the slot whenever
//!   [`GuestMemory::backing_generation`] changes (fork/execve).
//! * A guest `syscall` vectors to the `hlt; sysretq` trampoline at CPL0; the
//!   `hlt` exits to the host as `KVM_EXIT_HLT` → [`Exit::Syscall`]. The kernel
//!   writes the return value into `rax` ([`KvmVcpu::set_syscall_ret`]) and the
//!   resumed `sysretq` returns to CPL3 at the user `rip`/`rflags` that
//!   `syscall` saved in `rcx`/`r11`.
//! * A guest access to a guest-physical hole (no memory slot) exits as
//!   `KVM_EXIT_MMIO` → [`Exit::MemFault`]. A CPU exception (the IDT is empty)
//!   escalates to a triple fault → `KVM_EXIT_SHUTDOWN`, decoded as a fault at
//!   `cr2` when one is recorded.
//!
//! Register state is cached host-side: refreshed from the vcpu after every
//! `KVM_RUN`, flushed back (when dirty) before the next. That makes the
//! accessors cheap, and — because the refresh includes the special registers —
//! `fork` clones the *true* mid-trap state (CPL0, in the trampoline), so the
//! child's first `sysretq` returns to user mode exactly like the parent's.

use super::sys;
use super::vm::{
    FAULT_TRAMP_VA, GDT_BASE, GDT_LIMIT, LSTAR_VA, SEL_UCODE, SEL_UDATA, STAR_VALUE, Vm, check,
};
use crate::vcpu::mem::PAGE_SIZE;
use crate::vcpu::phys::PhysMem;
use crate::vcpu::{Exit, GuestMemory, Vcpu, VcpuError};
use std::sync::Arc;
use std::time::Duration;

/// Per-thread wall-clock preemption for `KVM_RUN`.
///
/// `KVM_RUN` only returns on a vmexit or a delivered signal, so a compute-bound
/// guest (a JIT hot loop, a GC sweep) that never faults or syscalls would run
/// its whole time slice on hardware. To bound that, each host thread arms a
/// POSIX timer just before entering the guest; when it fires it raises a
/// dedicated signal *at that very thread* (`SIGEV_THREAD_ID`), whose no-op
/// handler — installed without `SA_RESTART` — makes `KVM_RUN` return `-EINTR`,
/// which [`KvmVcpu::run_once`] already turns into [`Exit::Interrupted`].
///
/// A `KvmVcpu` can migrate between the SMP worker threads, so the timer is a
/// thread-local keyed to the thread that actually runs the guest, created lazily
/// and re-armed around each `KVM_RUN`.
mod preempt {
    use super::sys;
    use std::cell::Cell;
    use std::sync::Once;
    use std::time::Duration;

    /// The signal the preemption timer raises. A real-time signal above the two
    /// (`SIGRTMIN`, `SIGRTMIN+1`) that glibc's NPTL reserves for thread
    /// cancellation and `setxid`, so it collides with nothing the host runtime
    /// or the guest-signal emulation (which is entirely in-VM and touches no
    /// host signal) uses. The number is the kernel's raw signal, not glibc's
    /// remapped `SIGRTMIN`.
    const PREEMPT_SIG: i32 = 40;

    /// The signal handler: it does nothing. Its entire purpose is to be a
    /// non-`SA_RESTART` handler so that its delivery interrupts the `KVM_RUN`
    /// ioctl (returning `-EINTR`) rather than being ignored or restarting it.
    extern "C" fn on_preempt(_sig: i32) {}

    fn install_handler() {
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            let act = sys::sigaction {
                sa_handler: on_preempt as *const () as usize,
                sa_mask: [0; 16],
                // No SA_RESTART: the delivery must break KVM_RUN out with EINTR.
                sa_flags: 0,
                sa_restorer: 0,
            };
            // SAFETY: `act` is a valid, fully-initialized sigaction; we install
            // our own no-op handler for a signal nothing else uses. glibc's
            // wrapper supplies the SA_RESTORER trampoline.
            unsafe {
                sys::sigaction(PREEMPT_SIG, std::ptr::from_ref(&act), std::ptr::null_mut());
            }
        });
    }

    thread_local! {
        /// This thread's one-shot preemption timer, created on first use and
        /// targeting this thread's tid. `None` until created (or if creation
        /// failed — then preemption silently degrades to the syscall-count cap).
        static TIMER: Cell<Option<sys::timer_t>> = const { Cell::new(None) };
    }

    /// This thread's timer id, creating it (and installing the handler) on first
    /// use. The timer notifies via `SIGEV_THREAD_ID` at *this* thread, so the
    /// signal always lands on whoever is inside `KVM_RUN`.
    fn thread_timer() -> Option<sys::timer_t> {
        install_handler();
        TIMER.with(|slot| {
            if let Some(id) = slot.get() {
                return Some(id);
            }
            // SAFETY: `gettid` takes no arguments and only reads the caller's tid.
            let tid = unsafe { sys::gettid() };
            let mut ev = sys::sigevent {
                sigev_signo: PREEMPT_SIG,
                sigev_notify: sys::SIGEV_THREAD_ID,
                sigev_notify_thread_id: tid,
                ..sys::sigevent::default()
            };
            let mut id: sys::timer_t = std::ptr::null_mut();
            // SAFETY: valid out-pointers; creates a per-thread CLOCK_MONOTONIC
            // timer bound to this thread via SIGEV_THREAD_ID.
            let ret = unsafe {
                sys::timer_create(
                    sys::CLOCK_MONOTONIC,
                    std::ptr::from_mut(&mut ev),
                    std::ptr::from_mut(&mut id),
                )
            };
            if ret != 0 {
                return None;
            }
            slot.set(Some(id));
            Some(id)
        })
    }

    /// Arm this thread's timer to fire once after `d` (a zero `it_interval`
    /// makes it one-shot). Call immediately before `KVM_RUN`.
    pub fn arm(d: Duration) {
        let Some(id) = thread_timer() else {
            return;
        };
        let spec = sys::itimerspec {
            it_interval: sys::timespec::default(),
            it_value: sys::timespec {
                // A zero it_value would *disarm* the timer, so a sub-nanosecond
                // quantum still asks for at least one nanosecond.
                tv_sec: d.as_secs() as i64,
                tv_nsec: i64::from(d.subsec_nanos().max(1)),
            },
        };
        // SAFETY: valid timer id and itimerspec; one-shot relative arming.
        unsafe {
            sys::timer_settime(id, 0, std::ptr::from_ref(&spec), std::ptr::null_mut());
        }
    }

    /// Disarm this thread's timer (zero `it_value`). Call immediately after
    /// `KVM_RUN` so a late expiry doesn't fire into the next, unrelated slice.
    ///
    /// v1 accepts two benign races: the signal landing between `arm` and
    /// `KVM_RUN` (the run returns EINTR at once — a harmlessly short slice), and
    /// landing after `KVM_RUN` returns but before `disarm` (the no-op handler
    /// runs, doing nothing). The fully race-free upgrade is
    /// `KVM_SET_SIGNAL_MASK` — unblock the signal only for the duration of
    /// `KVM_RUN` — which we deliberately skip for v1.
    pub fn disarm() {
        TIMER.with(|slot| {
            if let Some(id) = slot.get() {
                let spec = sys::itimerspec::default();
                // SAFETY: valid timer id; a zero it_value disarms the timer.
                unsafe {
                    sys::timer_settime(id, 0, std::ptr::from_ref(&spec), std::ptr::null_mut());
                }
            }
        });
    }
}

/// `RFLAGS` for fresh user code: just the always-one bit and IF (interrupts
/// are never injected, but a clear IF would make a guest `pushf` look odd).
const RFLAGS_USER: u64 = 0x202;
/// `IA32_FMASK`: `syscall` clears TF|IF|DF for the trampoline.
const FMASK_VALUE: u64 = 0x700;
/// `CR0`: PE|MP|ET|NE|WP|AM|PG — protected, paged, natural FPU error handling.
const CR0_LONG: u64 = 0x8005_0033;
/// `CR4`: PAE (long mode requires it) + OSFXSR|OSXMMEXCPT (SSE enabled) +
/// OSXSAVE (bit 18) so `xgetbv`/AVX state is usable — paired with `XCR0` set to
/// x87|SSE|AVX via `KVM_SET_XCRS`. Without OSXSAVE a runtime's AVX probe (which
/// checks OSXSAVE then `xgetbv`) reports AVX unavailable and may fall back to a
/// slower/rarer path.
const CR4_LONG: u64 = 0x620 | 0x4_0000;

/// `XCR0` value: enable x87 (bit 0), SSE (bit 1) and AVX (bit 2) xsave state.
const XCR0_AVX: u64 = 0x7;
/// `EFER`: SCE (`syscall`) | LME | LMA | **NXE** (bit 11) — NXE lets the page
/// tables' `NX` bit take effect, so non-executable pages actually fault.
const EFER_LONG: u64 = 0xD01;

pub struct KvmVcpu {
    vm: Arc<Vm>,
    fd: super::vm::Fd,
    /// Kernel-shared run page, mapped for the fd's lifetime.
    run: *mut sys::kvm_run,
    regs: sys::kvm_regs,
    regs_dirty: bool,
    sregs: sys::kvm_sregs,
    sregs_dirty: bool,
    /// Trapped in a syscall whose return value has not been written yet. A
    /// *blocked* syscall (kernel parks the task, sets no return) is retried
    /// by simply running the vcpu again — the interpreter re-executes the
    /// `syscall` instruction because its pc still points at it, but this vcpu
    /// is already past the trampoline's `hlt`, so "running" it would `sysretq`
    /// back to user code with a stale rax. Instead, while this flag is set,
    /// [`KvmVcpu::run`] re-delivers [`Exit::Syscall`] from the cached
    /// registers without entering the guest; [`KvmVcpu::set_syscall_ret`]
    /// (and `reset`) clear it.
    in_syscall: bool,
    /// Debug: linear addresses of armed 4-byte write watchpoints
    /// (`NIXVM_WATCHPOINT`), used to log every write to them. Empty when unarmed.
    watchpoint: Vec<u64>,
    /// Debug: an execute breakpoint (`NIXVM_EBP`); logs `r8`/`rdi` state each
    /// time the address is reached. `None` when unarmed.
    ebp: Option<u64>,
    /// Wall-clock preemption quantum: a per-thread timer armed around each
    /// `KVM_RUN` breaks the guest out with `Exit::Interrupted` after this long,
    /// so a compute-bound guest that never exits still yields the CPU. `None`
    /// disables it. Cached from `NIXVM_QUANTUM_MS` (see [`preempt`] and
    /// [`crate::vcpu::preempt_quantum`]).
    quantum: Option<Duration>,
    /// The `CR3` this vcpu's `sregs` currently name, so a context switch (execve
    /// installs fresh page tables; a forked child runs a different address space)
    /// is picked up as a single `KVM_SET_SREGS` before the next run. `None` until
    /// the first run sets it from the guest memory's `cr3()`.
    cur_cr3: Option<u64>,
    /// Set by [`Vcpu::flush_tlb`]: force a guest-TLB flush before the next run
    /// because the host edited this address space's page tables in place (a fork's
    /// copy-on-write downgrade of the parent) without the guest reloading cr3.
    tlb_flush_pending: bool,
    /// The shared physical pool, cached from the running `GuestMemory` so the
    /// fault path can read the pushed `#PF` exception frame lock-free (no
    /// `GuestMemory` borrow — the SMP path holds none). Set on the first run.
    phys: Option<Arc<PhysMem>>,
    /// Physical address of the running address space's private kernel-stack frame
    /// (see [`GuestMemory::kstack_pa`]): where this vcpu reads the exception frame
    /// the CPU pushed, so sibling vcpus faulting at once never read each other's.
    /// Refreshed from `mem` on every reconcile (it changes on execve/fork).
    kstack_pa: u64,
}

// The register caches and run page are noise in a debug dump; the fd
// identifies the vcpu.
#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for KvmVcpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvmVcpu").field("fd", &self.fd).finish()
    }
}

// SAFETY: a `KvmVcpu` is owned and run by one thread at a time — the SMP
// scheduler hands each vcpu to a single worker for the duration of a run and
// only moves it between workers between runs (never issuing its ioctls
// concurrently), and KVM permits that migration. Distinct vcpus of one VM *do*
// run concurrently on different workers, which their per-vcpu fds and the
// shared VM's internally-serialized state (the `guest_slot` Mutex, atomic
// page-table stores) support. The raw `run` pointer is a per-vcpu mapping owned
// by this value.
unsafe impl Send for KvmVcpu {}

impl KvmVcpu {
    /// Create a vcpu for a fresh thread, entering user code at `entry` with
    /// stack `stack`. Returns a boxed trait object (a factory, not `Self`).
    #[allow(clippy::new_ret_no_self)]
    pub fn new(vm: Arc<Vm>, entry: u64, stack: u64) -> Result<Box<dyn Vcpu>, VcpuError> {
        let mut v = Self::create(vm)?;
        v.regs = sys::kvm_regs {
            rip: entry,
            rsp: stack,
            rflags: RFLAGS_USER,
            ..Default::default()
        };
        v.regs_dirty = true;
        Ok(Box::new(v))
    }

    /// Create and fully configure a vcpu (CPUID, MSRs, long-mode sregs) with
    /// zeroed general registers; the caller sets rip/rsp.
    fn create(vm: Arc<Vm>) -> Result<Self, VcpuError> {
        let (fd, run) = vm.create_vcpu()?;

        // SAFETY: `fd` is a live vcpu fd; the cpuid2 struct is valid for the
        // call and the kernel only reads it.
        let ret = unsafe {
            sys::ioctl(
                fd.0,
                sys::KVM_SET_CPUID2,
                std::ptr::from_ref(vm.cpuid()),
            )
        };
        check(ret, "KVM_SET_CPUID2")?;

        // The syscall trampoline wiring (see `super::vm`).
        let entries = [
            (sys::MSR_IA32_STAR, STAR_VALUE),
            (sys::MSR_IA32_LSTAR, LSTAR_VA),
            (sys::MSR_IA32_CSTAR, 0),
            (sys::MSR_IA32_FMASK, FMASK_VALUE),
            (sys::MSR_KERNEL_GS_BASE, 0),
        ];
        let mut msrs = sys::kvm_msrs {
            nmsrs: entries.len() as u32,
            ..Default::default()
        };
        for (i, (index, data)) in entries.into_iter().enumerate() {
            msrs.entries[i] = sys::kvm_msr_entry {
                index,
                reserved: 0,
                data,
            };
        }
        // SAFETY: `msrs` is a valid struct whose `nmsrs` bounds the entries.
        let ret = unsafe { sys::ioctl(fd.0, sys::KVM_SET_MSRS, std::ptr::from_ref(&msrs)) };
        // KVM_SET_MSRS returns the number of MSRs set; a short count is a failure.
        if check(ret, "KVM_SET_MSRS")? != entries.len() as i32 {
            return Err(VcpuError::Backend("KVM_SET_MSRS set fewer MSRs than requested".into()));
        }

        // Start from the vcpu's current sregs (sane reset values for the
        // fields we keep, e.g. apic_base) and switch it into flat long mode.
        let mut sregs = sys::kvm_sregs::default();
        // SAFETY: valid out-pointer; the fd is live.
        let ret = unsafe { sys::ioctl(fd.0, sys::KVM_GET_SREGS, std::ptr::from_mut(&mut sregs)) };
        check(ret, "KVM_GET_SREGS")?;
        init_user_sregs(&mut sregs);

        // Enable AVX xsave state (XCR0 = x87|SSE|AVX). CR4.OSXSAVE is set in
        // sregs above; a CPL3 guest can't run `xsetbv`, so the host sets XCR0.
        let mut xcrs = sys::kvm_xcrs {
            nr_xcrs: 1,
            ..sys::kvm_xcrs::default()
        };
        xcrs.xcrs[0] = sys::kvm_xcr {
            xcr: 0,
            reserved: 0,
            value: XCR0_AVX,
        };
        // SAFETY: valid struct pointer; the fd is live.
        let ret = unsafe { sys::ioctl(fd.0, sys::KVM_SET_XCRS, std::ptr::from_ref(&xcrs)) };
        check(ret, "KVM_SET_XCRS")?;

        // Debug: NIXVM_EBP=0xADDR arms an execute breakpoint on DR0 (fires
        // *before* the instruction); otherwise NIXVM_WATCHPOINT arms up to four
        // 4-byte write watchpoints (DR0..3). Both exit with KVM_EXIT_DEBUG.
        let ebp = std::env::var("NIXVM_EBP")
            .ok()
            .and_then(|s| u64::from_str_radix(s.trim().trim_start_matches("0x"), 16).ok());
        let watchpoints: Vec<u64> = std::env::var("NIXVM_WATCHPOINT")
            .ok()
            .map(|s| {
                s.split(',')
                    .filter_map(|p| u64::from_str_radix(p.trim().trim_start_matches("0x"), 16).ok())
                    .take(4)
                    .collect()
            })
            .unwrap_or_default();
        if let Some(addr) = ebp {
            // DR0 execute breakpoint: L0, R/W0=00 (execute), LEN0=00 (1 byte).
            let dbg = sys::kvm_guest_debug {
                control: sys::KVM_GUESTDBG_ENABLE | sys::KVM_GUESTDBG_USE_HW_BP,
                pad: 0,
                debugreg: [addr, 0, 0, 0, 0, 0, 0, (1 << 10) | 1],
            };
            // SAFETY: valid struct pointer; the fd is live.
            let ret =
                unsafe { sys::ioctl(fd.0, sys::KVM_SET_GUEST_DEBUG, std::ptr::from_ref(&dbg)) };
            check(ret, "KVM_SET_GUEST_DEBUG")?;
        } else if !watchpoints.is_empty() {
            let mut debugreg = [0u64; 8];
            let mut dr7 = 1u64 << 10; // reserved bit 10
            for (i, &addr) in watchpoints.iter().enumerate() {
                debugreg[i] = addr;
                // DRi: Li (bit 2i), R/Wi=01 write (bits 16+4i..), LENi=11 4-byte.
                dr7 |= 1 << (2 * i); // local enable
                dr7 |= 0b01 << (16 + 4 * i); // write
                dr7 |= 0b11 << (18 + 4 * i); // 4-byte
            }
            debugreg[7] = dr7;
            let dbg = sys::kvm_guest_debug {
                control: sys::KVM_GUESTDBG_ENABLE | sys::KVM_GUESTDBG_USE_HW_BP,
                pad: 0,
                debugreg,
            };
            // SAFETY: valid struct pointer; the fd is live.
            let ret =
                unsafe { sys::ioctl(fd.0, sys::KVM_SET_GUEST_DEBUG, std::ptr::from_ref(&dbg)) };
            check(ret, "KVM_SET_GUEST_DEBUG")?;
        }
        let watchpoint = watchpoints;

        Ok(Self {
            vm,
            fd,
            run,
            regs: sys::kvm_regs::default(),
            regs_dirty: true,
            sregs,
            sregs_dirty: true,
            in_syscall: false,
            watchpoint,
            ebp,
            quantum: crate::vcpu::preempt_quantum(),
            cur_cr3: None,
            tlb_flush_pending: false,
            phys: None,
            kstack_pa: 0,
        })
    }

    /// Point the vcpu's `sregs.cr3` at `mem`'s address space if it isn't already —
    /// the per-process page-table switch. One `KVM_SET_SREGS` (deferred via the
    /// dirty flag) on a change, nothing on the common same-process run.
    fn sync_cr3(&mut self, mem: &GuestMemory) {
        let cr3 = mem.cr3();
        if self.cur_cr3 != Some(cr3) {
            self.sregs.cr3 = cr3;
            self.sregs_dirty = true;
            self.cur_cr3 = Some(cr3);
        }
        // Cache the pool handle (once) and this address space's private kstack
        // frame so the fault path reads the pushed exception frame from *this*
        // vcpu's kstack — never a sibling's — without a `GuestMemory` borrow.
        if self.phys.is_none() {
            self.phys = Some(mem.phys_arc());
        }
        self.kstack_pa = mem.kstack_pa();
    }

    /// Flush dirty caches to the vcpu, run it once, and refresh the caches.
    fn run_once(&mut self) -> Result<u32, VcpuError> {
        if self.regs_dirty {
            // SAFETY: valid struct pointer; the fd is live.
            let ret =
                unsafe { sys::ioctl(self.fd.0, sys::KVM_SET_REGS, std::ptr::from_ref(&self.regs)) };
            check(ret, "KVM_SET_REGS")?;
            self.regs_dirty = false;
        }
        if self.tlb_flush_pending {
            // Force the guest TLB to be flushed before this run. KVM skips the
            // flush when `KVM_SET_SREGS` leaves cr3 unchanged, so first push a cr3
            // that differs only in an ignored bit (PCD, bit 4 — CR4.PCIDE is off,
            // so it does not select a page table), which KVM treats as a pgd
            // switch and flushes. The real cr3 is restored by the `sregs_dirty`
            // write below (also a change → flushed), and the run uses it.
            let real = self.sregs.cr3;
            self.sregs.cr3 = real ^ (1 << 4);
            // SAFETY: valid struct pointer; the fd is live.
            let ret = unsafe {
                sys::ioctl(self.fd.0, sys::KVM_SET_SREGS, std::ptr::from_ref(&self.sregs))
            };
            check(ret, "KVM_SET_SREGS")?;
            self.sregs.cr3 = real;
            self.sregs_dirty = true;
            self.tlb_flush_pending = false;
        }
        if self.sregs_dirty {
            // SAFETY: valid struct pointer; the fd is live.
            let ret = unsafe {
                sys::ioctl(self.fd.0, sys::KVM_SET_SREGS, std::ptr::from_ref(&self.sregs))
            };
            check(ret, "KVM_SET_SREGS")?;
            self.sregs_dirty = false;
        }

        // Arm this thread's preemption timer so a compute-bound guest that never
        // exits is broken out after the quantum: the timer signal interrupts
        // KVM_RUN (EINTR), which becomes Exit::Interrupted below. Disarmed right
        // after so a late expiry can't fire into an unrelated later slice.
        if let Some(q) = self.quantum {
            preempt::arm(q);
        }
        // SAFETY: `KVM_RUN` takes no argument; the fd is live and not run
        // concurrently (see the `Send` note).
        let ret = unsafe { sys::ioctl(self.fd.0, sys::KVM_RUN, 0) };
        if self.quantum.is_some() {
            preempt::disarm();
        }
        let interrupted = if ret < 0 {
            let err = std::io::Error::last_os_error();
            // EINTR/EAGAIN: a host signal broke us out of the guest.
            if !matches!(err.raw_os_error(), Some(4 | 11)) {
                return Err(VcpuError::Backend(format!("KVM_RUN failed: {err}")));
            }
            true
        } else {
            false
        };

        // Refresh both caches so accessors (and `fork`) see the true state.
        // SAFETY: valid out-pointers; the fd is live.
        let ret =
            unsafe { sys::ioctl(self.fd.0, sys::KVM_GET_REGS, std::ptr::from_mut(&mut self.regs)) };
        check(ret, "KVM_GET_REGS")?;
        // SAFETY: as above.
        let ret = unsafe {
            sys::ioctl(self.fd.0, sys::KVM_GET_SREGS, std::ptr::from_mut(&mut self.sregs))
        };
        check(ret, "KVM_GET_SREGS")?;

        if interrupted {
            return Ok(sys::KVM_EXIT_INTR);
        }
        // SAFETY: the run page is a live mapping; the kernel just wrote the
        // exit reason for the KVM_RUN that returned.
        Ok(unsafe { (*self.run).exit_reason })
    }

    /// Drive `KVM_RUN` until a real [`Exit`], having already reconciled memory.
    ///
    /// `mem` is `Some` only on the serial path (a locked run), where it is used
    /// solely to read the value written at a debug watchpoint; the lockless SMP
    /// path passes `None` (no `GuestMemory` borrow is held) and logs the hit
    /// without the value. Everything else — fault-frame recovery, exit decode —
    /// reads the control block via `self.vm`, never guest memory, so it works
    /// identically with or without `mem`.
    fn run_loop(&mut self, mem: Option<&GuestMemory>) -> Result<Exit, VcpuError> {
        loop {
            let reason = self.run_once()?;
            // Debug watchpoint hit: log the write (value + faulting pc) and
            // resume — the write already completed (a data #DB is a trap).
            if reason == sys::KVM_EXIT_DEBUG {
                // SAFETY: the run page is live; `debug` is the arm the kernel
                // wrote for KVM_EXIT_DEBUG. dr6 bits 0..3 say which DRn matched.
                let (pc, dr6) = unsafe {
                    let d = (*self.run).exit.debug;
                    (d.pc, d.dr6)
                };
                if self.ebp.is_some() {
                    // Execute breakpoint: log the register state each time the
                    // address is reached (for tracing a specific instruction).
                    let r = &self.regs;
                    eprintln!(
                        "[ebp] pc={pc:#x} rax={:#x} rdi={:#x} rsi={:#x} rdx={:#x} rcx={:#x} r8={:#x} rsp={:#x}",
                        r.rax, r.rdi, r.rsi, r.rdx, r.rcx, r.r8, r.rsp
                    );
                    // An execute #DB is a fault (rip unchanged); set RF so the
                    // resumed instruction runs once without re-triggering.
                    self.regs.rflags |= 0x1_0000;
                    self.regs_dirty = true;
                    continue;
                }
                for (i, &addr) in self.watchpoint.iter().enumerate() {
                    if dr6 & (1 << i) != 0 {
                        let val = mem.and_then(|m| m.read_u64(addr).ok()).unwrap_or(0);
                        eprintln!(
                            "[wp] {addr:#x} <- {val:#018x} (pc={pc:#x}) rsp={:#x}",
                            self.regs.rsp
                        );
                    }
                }
                continue;
            }
            let exit = self.decode_exit(reason)?;
            if exit == Exit::Syscall {
                self.in_syscall = true;
            }
            return Ok(exit);
        }
    }

    /// Decode one exit reason into an [`Exit`].
    fn decode_exit(&mut self, reason: u32) -> Result<Exit, VcpuError> {
        match reason {
            // A `hlt` exit is either the syscall trampoline or the #PF
            // trampoline; the vcpu `rip` (just past the executed `hlt`)
            // distinguishes them.
            sys::KVM_EXIT_HLT if self.regs.rip == FAULT_TRAMP_VA + 1 => {
                // A page fault vectored here at CPL0. The CPU pushed
                // [error_code, RIP, CS, RFLAGS, RSP, SS] onto the kernel stack
                // (now `rsp`). Recover the faulting user state so accessors and
                // signal delivery see it, and report the fault at cr2.
                let ksp = self.regs.rsp;
                // The frame sits in this address space's private kstack page; read
                // it from that frame's pool physical address (cached in `sync_cr3`)
                // so concurrent sibling faults never read one another's frame.
                let frame = self.phys.as_ref().map(|ph| {
                    let at = |va: u64| ph.read_u64(self.kstack_pa + (va & (PAGE_SIZE - 1)));
                    (at(ksp), at(ksp + 8), at(ksp + 24), at(ksp + 32))
                });
                if let Some((err, fault_rip, fault_rflags, fault_rsp)) = frame {
                    self.regs.rip = fault_rip;
                    self.regs.rsp = fault_rsp;
                    self.regs.rflags = fault_rflags;
                    self.regs_dirty = true; // resume re-runs the faulting instruction
                    // The exception entered CPL0; restore the user segments so
                    // the resumed guest runs at CPL3, not with kernel privilege
                    // (and so its next fault switches to RSP0 again).
                    self.sregs.cs = user_cs();
                    self.sregs.ss = user_ss();
                    self.sregs_dirty = true;
                    Ok(Exit::MemFault {
                        addr: self.sregs.cr2,
                        write: err & 0x2 != 0, // #PF error-code bit 1 = write
                    })
                } else {
                    // The frame wasn't on the kernel stack (unexpected); fall
                    // back to a bare fault at cr2 rather than crashing.
                    Ok(Exit::MemFault { addr: self.sregs.cr2, write: false })
                }
            }
            // The trampoline's `hlt` — a guest `syscall`. (With no in-kernel
            // irqchip, `hlt` exits straight to userspace and the next KVM_RUN
            // resumes after it, at the `sysretq`.)
            sys::KVM_EXIT_HLT => Ok(Exit::Syscall),
            sys::KVM_EXIT_MMIO => {
                // SAFETY: the union's `mmio` arm is the one the kernel wrote
                // for this exit reason; all fields are plain old data.
                let mmio = unsafe { (*self.run).exit.mmio };
                Ok(Exit::MemFault {
                    addr: mmio.phys_addr,
                    write: mmio.is_write != 0,
                })
            }
            // Triple fault. The IDT has a #PF gate, so genuine page faults are
            // delivered through the trampoline above and never reach here (the
            // kstack is a pinned control-block frame, so #PF delivery cannot
            // itself fault). A SHUTDOWN is therefore an exception we don't handle
            // — #UD, #GP, #BP, … — reported as an illegal instruction at the
            // faulting `rip`. (cr2 is deliberately *not* consulted: it is sticky
            // from the last real #PF, so with demand paging it is almost always
            // non-zero and would misattribute a #UD as a bogus memory fault.)
            sys::KVM_EXIT_SHUTDOWN => Ok(Exit::IllegalInstruction { pc: self.regs.rip }),
            sys::KVM_EXIT_INTR => Ok(Exit::Interrupted),
            sys::KVM_EXIT_FAIL_ENTRY => {
                // SAFETY: as `mmio` — the arm matching the exit reason.
                let fail = unsafe { (*self.run).exit.fail_entry };
                Err(VcpuError::Backend(format!(
                    "KVM_EXIT_FAIL_ENTRY (hardware reason {:#x})",
                    fail.hardware_entry_failure_reason
                )))
            }
            sys::KVM_EXIT_INTERNAL_ERROR => {
                // SAFETY: as `mmio` — the arm matching the exit reason.
                let internal = unsafe { (*self.run).exit.internal };
                Err(VcpuError::Backend(format!(
                    "KVM_EXIT_INTERNAL_ERROR (suberror {})",
                    internal.suberror
                )))
            }
            other => Err(VcpuError::Backend(format!(
                "unexpected KVM exit reason {other}"
            ))),
        }
    }

    /// Architectural x86 register index → cached `kvm_regs` field. The trait
    /// uses encoding order (rax, rcx, rdx, rbx, rsp, rbp, rsi, rdi, r8–r15),
    /// matching `interp_x86`; `kvm_regs` declares them in a different order.
    fn gpr(&self, idx: usize) -> u64 {
        let r = &self.regs;
        match idx {
            0 => r.rax,
            1 => r.rcx,
            2 => r.rdx,
            3 => r.rbx,
            4 => r.rsp,
            5 => r.rbp,
            6 => r.rsi,
            7 => r.rdi,
            8 => r.r8,
            9 => r.r9,
            10 => r.r10,
            11 => r.r11,
            12 => r.r12,
            13 => r.r13,
            14 => r.r14,
            15 => r.r15,
            _ => 0,
        }
    }

    fn set_gpr(&mut self, idx: usize, value: u64) {
        let r = &mut self.regs;
        match idx {
            0 => r.rax = value,
            1 => r.rcx = value,
            2 => r.rdx = value,
            3 => r.rbx = value,
            4 => r.rsp = value,
            5 => r.rbp = value,
            6 => r.rsi = value,
            7 => r.rdi = value,
            8 => r.r8 = value,
            9 => r.r9 = value,
            10 => r.r10 = value,
            11 => r.r11 = value,
            12 => r.r12 = value,
            13 => r.r13 = value,
            14 => r.r14 = value,
            15 => r.r15 = value,
            _ => return,
        }
        self.regs_dirty = true;
    }
}

/// The flat 64-bit user code segment (CPL3).
fn user_cs() -> sys::kvm_segment {
    sys::kvm_segment {
        base: 0,
        limit: 0xFFFF_FFFF,
        selector: SEL_UCODE,
        type_: 0xB, // execute/read, accessed
        present: 1,
        dpl: 3,
        db: 0,
        s: 1,
        l: 1,
        g: 1,
        avl: 0,
        unusable: 0,
        padding: 0,
    }
}

/// The flat 64-bit user data segment (CPL3).
fn user_ss() -> sys::kvm_segment {
    sys::kvm_segment {
        limit: 0xFFFF_FFFF,
        selector: SEL_UDATA,
        type_: 0x3, // read/write, accessed
        dpl: 3,
        db: 1,
        l: 0,
        ..user_cs()
    }
}

/// Switch `sregs` into flat 64-bit user mode: the control block's GDT and
/// page tables, user code/data segments, long mode enabled. Everything not
/// set here (apic_base, interrupt_bitmap) keeps its current value.
fn init_user_sregs(sregs: &mut sys::kvm_sregs) {
    let code = user_cs();
    let data = user_ss();
    sregs.cs = code;
    sregs.ss = data;
    sregs.ds = data;
    sregs.es = data;
    sregs.fs = data;
    sregs.gs = data;
    // The real TSS (base = TSS_BASE, RSP0 set): the CPU switches to RSP0 on a
    // CPL3→CPL0 exception (a #PF vectoring through the IDT), where it pushes the
    // exception frame the host reads back.
    sregs.tr = sys::kvm_segment {
        base: super::vm::TSS_BASE,
        limit: 0x67,
        selector: super::vm::SEL_TSS,
        type_: 0xB, // busy 64-bit TSS
        present: 1,
        dpl: 0,
        db: 0,
        s: 0,
        l: 0,
        g: 0,
        avl: 0,
        unusable: 0,
        padding: 0,
    };
    sregs.ldt = sys::kvm_segment {
        unusable: 1,
        ..sys::kvm_segment::default()
    };
    sregs.gdt = sys::kvm_dtable {
        base: GDT_BASE,
        limit: GDT_LIMIT,
        padding: [0; 3],
    };
    // IDT with a #PF gate (see the control block); a page fault vectors to the
    // fault trampoline and exits cleanly with the real register state, instead
    // of triple-faulting (which loses it). Other exceptions still triple-fault.
    sregs.idt = sys::kvm_dtable {
        base: super::vm::IDT_BASE,
        limit: 0xFFF,
        padding: [0; 3],
    };
    sregs.cr0 = CR0_LONG;
    sregs.cr2 = 0;
    // cr3 is this process's real PML4, loaded per-run from the guest memory's
    // `cr3()` (see `sync_cr3`); a placeholder here, overwritten before the first
    // `KVM_RUN`.
    sregs.cr3 = 0;
    sregs.cr4 = CR4_LONG;
    sregs.efer = EFER_LONG;
}

impl Vcpu for KvmVcpu {
    fn run(&mut self, mem: &mut GuestMemory) -> Result<Exit, VcpuError> {
        // A trapped syscall whose return was never written is being retried
        // (it blocked) — re-deliver it without touching the guest.
        if self.in_syscall {
            return Ok(Exit::Syscall);
        }
        // Register the pool memslot (once) and point cr3 at this process's page
        // tables, then run under the caller's held memory lock.
        self.vm.ensure_memslot(mem)?;
        self.sync_cr3(mem);
        self.run_loop(Some(mem))
    }

    fn needs_locked_run(&self) -> bool {
        // KVM executes against the mapped memslot: `KVM_RUN` needs no Rust borrow
        // of GuestMemory, so the SMP scheduler runs it with the lock dropped.
        false
    }

    fn reconcile(&mut self, mem: &mut GuestMemory) -> Result<(), VcpuError> {
        // Re-delivering a blocked syscall: `run_bare` returns Syscall without a
        // KVM_RUN, so nothing to prepare.
        if self.in_syscall {
            return Ok(());
        }
        // The one memory-touching step before a lockless run: register the pool
        // memslot (once) and load this process's cr3. With one always-mapped
        // memslot and real per-process page tables there is nothing else to sync —
        // sibling vcpus of any address space run concurrently against the same slot.
        self.vm.ensure_memslot(mem)?;
        self.sync_cr3(mem);
        Ok(())
    }

    fn run_bare(&mut self) -> Result<Exit, VcpuError> {
        if self.in_syscall {
            return Ok(Exit::Syscall);
        }
        self.run_loop(None)
    }

    fn syscall_nr(&self) -> u64 {
        self.regs.rax
    }

    fn syscall_args(&self) -> [u64; 6] {
        [
            self.regs.rdi,
            self.regs.rsi,
            self.regs.rdx,
            self.regs.r10,
            self.regs.r8,
            self.regs.r9,
        ]
    }

    fn set_syscall_ret(&mut self, value: u64) {
        // rip already points past the trampoline's `hlt`, at the `sysretq`
        // that returns to user code — only the return register needs writing.
        self.regs.rax = value;
        self.regs_dirty = true;
        self.in_syscall = false;
    }

    fn reg(&self, idx: usize) -> u64 {
        self.gpr(idx)
    }
    fn set_reg(&mut self, idx: usize, value: u64) {
        self.set_gpr(idx, value);
    }

    fn pc(&self) -> u64 {
        self.regs.rip
    }
    fn set_pc(&mut self, pc: u64) {
        self.regs.rip = pc;
        self.regs_dirty = true;
    }
    fn sp(&self) -> u64 {
        self.regs.rsp
    }
    fn set_sp(&mut self, sp: u64) {
        self.regs.rsp = sp;
        self.regs_dirty = true;
    }
    fn rflags(&self) -> u64 {
        self.regs.rflags
    }
    fn set_rflags(&mut self, value: u64) {
        // Force the always-set reserved bit and keep interrupts enabled at CPL3;
        // clearing IF or the reserved bit would make the vcpu unrunnable.
        self.regs.rflags = (value & 0x00dd_5dd5) | 0x0000_0202;
        self.regs_dirty = true;
    }

    fn set_tls(&mut self, value: u64) {
        self.sregs.fs.base = value;
        self.sregs_dirty = true;
    }

    fn fork(&self) -> Box<dyn Vcpu> {
        // A sibling vcpu in the same VM, cloned from the cached (just
        // refreshed) register state — including the mid-trap CPL0 segment
        // state, so the child's resume path is identical to the parent's. The
        // child re-issues the guest slot for its own (forked) memory on its
        // first `run` via the backing-generation seam.
        let mut child = Self::create(self.vm.clone()).expect("create KVM vcpu (fork)");
        child.regs = self.regs;
        child.regs_dirty = true;
        child.sregs = self.sregs;
        child.sregs_dirty = true;
        child.in_syscall = self.in_syscall;
        // The child runs the *child's* address space (a different cr3); force a
        // re-sync on its first run rather than inheriting the parent's cached one.
        child.cur_cr3 = None;
        // FPU/SSE state is not cached host-side; copy it fd-to-fd.
        let mut fpu = sys::kvm_fpu::default();
        // SAFETY: valid out-pointer; both fds are live vcpus of this VM.
        let ret = unsafe { sys::ioctl(self.fd.0, sys::KVM_GET_FPU, std::ptr::from_mut(&mut fpu)) };
        check(ret, "KVM_GET_FPU").expect("read FPU state (fork)");
        // SAFETY: as above; the kernel only reads the struct.
        let ret = unsafe { sys::ioctl(child.fd.0, sys::KVM_SET_FPU, std::ptr::from_ref(&fpu)) };
        check(ret, "KVM_SET_FPU").expect("write FPU state (fork)");
        Box::new(child)
    }

    fn reset(&mut self, entry: u64, sp: u64) {
        self.regs = sys::kvm_regs {
            rip: entry,
            rsp: sp,
            rflags: RFLAGS_USER,
            ..Default::default()
        };
        self.regs_dirty = true;
        // Back to flat user mode (an execve can land mid-trap, with the vcpu
        // still in the trampoline's CPL0 state) and clear TLS.
        init_user_sregs(&mut self.sregs);
        self.sregs.fs.base = 0;
        self.sregs_dirty = true;
        self.in_syscall = false;
        // execve installed fresh page tables in the same GuestMemory: its `cr3()`
        // changed, so drop the cached value and re-sync on the next run.
        self.cur_cr3 = None;
    }

    fn vdso_calibration(&self) -> Option<crate::vcpu::VdsoCal> {
        // Guest TSC frequency (kHz) — the ioctl returns it directly.
        // SAFETY: `KVM_GET_TSC_KHZ` takes no argument; the fd is a live vcpu.
        let khz = unsafe { sys::ioctl(self.fd.0, sys::KVM_GET_TSC_KHZ, 0) };
        let tsc_khz = u64::try_from(khz).ok().filter(|&k| k > 0)?;
        // ns ≈ tsc_delta * 1e6 / tsc_khz. As a fixed-point multiply-shift with
        // shift = 32: mult = round(1e6 * 2^32 / tsc_khz). For 1–5 GHz this is
        // ~1.7e9 down to ~8.6e8, comfortably within u64, and the vDSO's 128-bit
        // `mul` + `shrd` keeps full precision.
        let shift: u64 = 32;
        let mult = ((1_000_000u128 << shift) / u128::from(tsc_khz)) as u64;

        // Read the guest TSC (IA32_TSC) and the host wall clock as close together
        // as possible so the vDSO's TSC-derived time matches the syscall clock
        // (which reads the same host wall clock). nixvm returns wall time for
        // every clock id, so mono and wall share the base.
        let mut msrs = sys::kvm_msrs {
            nmsrs: 1,
            pad: 0,
            entries: [sys::kvm_msr_entry::default(); sys::MSRS_CAP],
        };
        msrs.entries[0].index = 0x10; // IA32_TSC
        // SAFETY: valid struct; `nmsrs = 1` bounds the entries read/written.
        let got = unsafe { sys::ioctl(self.fd.0, sys::KVM_GET_MSRS, std::ptr::from_mut(&mut msrs)) };
        if got != 1 {
            return None;
        }
        let base_tsc = msrs.entries[0].data;
        let now = crate::clock::now_unix().as_nanos() as u64;
        Some(crate::vcpu::VdsoCal {
            mult,
            shift,
            base_tsc,
            base_mono_ns: now,
            base_wall_ns: now,
        })
    }

    fn settle_syscall_return(&mut self) {
        // Software `sysretq`: what the trampoline's `sysretq` would have done to
        // return the guest from CPL0 to CPL3 — rip←rcx, rflags←r11, and the flat
        // user segments. Called when the host is about to redirect the guest away
        // from that pending `sysretq` (signal delivery / `rt_sigreturn`), so the
        // handler and the resumed user code run at CPL3 rather than supervisor.
        self.regs.rip = self.regs.rcx;
        self.regs.rflags = self.regs.r11;
        self.regs_dirty = true;
        self.sregs.cs = user_cs();
        self.sregs.ss = user_ss();
        self.sregs.ds = user_ss();
        self.sregs.es = user_ss();
        self.sregs_dirty = true;
        // The trampoline `sysretq` will not run; the syscall is logically done.
        self.in_syscall = false;
    }

    fn flush_tlb(&mut self) {
        // Arm the cr3-reload dance in `run_once` (a plain same-cr3 `KVM_SET_SREGS`
        // does not flush), so a host-side page-table change (fork's parent CoW
        // downgrade) is seen by the vcpu instead of shadowed by a stale TLB entry.
        self.tlb_flush_pending = true;
    }
}

impl Drop for KvmVcpu {
    fn drop(&mut self) {
        // SAFETY: `run` is the live mapping created in `Vm::create_vcpu`,
        // unmapped exactly once here; the fd closes via its own Drop.
        unsafe { sys::munmap(self.run.cast(), self.vm.vcpu_mmap_size()) };
    }
}
