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
use crate::vcpu::{Exit, GuestMemory, Vcpu, VcpuError};
use std::sync::Arc;

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
}

// The register caches and run page are noise in a debug dump; the fd
// identifies the vcpu.
#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for KvmVcpu {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KvmVcpu").field("fd", &self.fd).finish()
    }
}

// SAFETY: the vcpu is only ever *run* by one thread at a time (the serial
// scheduler this milestone pairs hardware backends with — same policy as HVF),
// and KVM permits moving a vcpu between threads as long as its ioctls are not
// issued concurrently. The raw `run` pointer is a per-vcpu mapping owned by
// this value.
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
        })
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
        if self.sregs_dirty {
            // SAFETY: valid struct pointer; the fd is live.
            let ret = unsafe {
                sys::ioctl(self.fd.0, sys::KVM_SET_SREGS, std::ptr::from_ref(&self.sregs))
            };
            check(ret, "KVM_SET_SREGS")?;
            self.sregs_dirty = false;
        }

        // SAFETY: `KVM_RUN` takes no argument; the fd is live and not run
        // concurrently (see the `Send` note).
        let ret = unsafe { sys::ioctl(self.fd.0, sys::KVM_RUN, 0) };
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
                let frame = (|| {
                    Some((
                        self.vm.read_ctrl_u64(ksp)?,
                        self.vm.read_ctrl_u64(ksp + 8)?,
                        self.vm.read_ctrl_u64(ksp + 24)?,
                        self.vm.read_ctrl_u64(ksp + 32)?,
                    ))
                })();
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
            // Triple fault: with an empty IDT every guest exception lands
            // here. A page fault records the address in cr2; anything else
            // (e.g. #UD) leaves it clear and is reported as illegal.
            sys::KVM_EXIT_SHUTDOWN => {
                if self.sregs.cr2 != 0 {
                    Ok(Exit::MemFault {
                        addr: self.sregs.cr2,
                        write: false,
                    })
                } else {
                    Ok(Exit::IllegalInstruction { pc: self.regs.rip })
                }
            }
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
    // The protection-enforcing page tables (super::paging), not the control
    // block's old uniformly-RWX identity map. Their PML4 is at the region base.
    sregs.cr3 = super::paging::PT_AREA_GPA;
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
        self.vm.reconcile_guest(mem)?;
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
                        let val = mem.read_u64(addr).unwrap_or(0);
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
    }
}

impl Drop for KvmVcpu {
    fn drop(&mut self) {
        // SAFETY: `run` is the live mapping created in `Vm::create_vcpu`,
        // unmapped exactly once here; the fd closes via its own Drop.
        unsafe { sys::munmap(self.run.cast(), self.vm.vcpu_mmap_size()) };
    }
}
