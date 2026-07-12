//! Raw FFI to macOS Hypervisor.framework (arm64) and the libSystem cache-
//! maintenance call. This is the crate's hardware-virtualization unsafe site:
//! every declaration here is an `extern "C"` binding, and the safe wrappers that
//! uphold their contracts live in [`super::vm`] and [`super`].
//!
//! Constants are the stable Apple ABI. The `hv_sys_reg_t` values are the ARM
//! `MRS`/`MSR` system-register encodings `(op0<<14)|(op1<<11)|(CRn<<7)|
//! (CRm<<3)|op2`; each is annotated with the register it selects.
#![allow(non_camel_case_types)]
// The full binding surface is declared up front; items not yet used by the vcpu
// implementation are allowed dead until the remaining milestone steps land.
#![allow(dead_code)]

use std::ffi::c_void;

/// `hv_return_t` — a `kern_return_t`; `HV_SUCCESS` is 0, everything else is an
/// error (e.g. `HV_DENIED` when the process lacks the hypervisor entitlement).
pub type hv_return_t = i32;
pub const HV_SUCCESS: hv_return_t = 0;

/// Opaque per-thread vcpu id.
pub type hv_vcpu_t = u64;
/// Guest intermediate physical address.
pub type hv_ipa_t = u64;

/// The exception detail of an [`hv_vcpu_exit_t`].
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct hv_vcpu_exit_exception_t {
    /// `ESR_EL2`-style syndrome: exception class in bits [31:26].
    pub syndrome: u64,
    /// Faulting virtual address (`FAR`), when applicable.
    pub virtual_address: u64,
    /// Faulting guest physical address (IPA), when applicable.
    pub physical_address: u64,
}

/// What `hv_vcpu_run` returned through. The framework writes this at the pointer
/// handed back by `hv_vcpu_create`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct hv_vcpu_exit_t {
    pub reason: u32,
    pub exception: hv_vcpu_exit_exception_t,
}

// hv_exit_reason_t
pub const HV_EXIT_REASON_CANCELED: u32 = 0;
pub const HV_EXIT_REASON_EXCEPTION: u32 = 1;
pub const HV_EXIT_REASON_VTIMER: u32 = 2;
pub const HV_EXIT_REASON_UNKNOWN: u32 = 3;

// hv_memory_flags_t (stage-2 permissions for hv_vm_map / hv_vm_protect)
pub const HV_MEMORY_READ: u64 = 1 << 0;
pub const HV_MEMORY_WRITE: u64 = 1 << 1;
pub const HV_MEMORY_EXEC: u64 = 1 << 2;

// hv_reg_t — general-purpose / PC / flags.
pub type hv_reg_t = u32;
/// `HV_REG_X0`; `Xn` is `HV_REG_X0 + n` for `n` in `0..=30`.
pub const HV_REG_X0: hv_reg_t = 0;
pub const HV_REG_PC: hv_reg_t = 31;
pub const HV_REG_FPCR: hv_reg_t = 32;
pub const HV_REG_FPSR: hv_reg_t = 33;
pub const HV_REG_CPSR: hv_reg_t = 34;

// hv_sys_reg_t — system registers (MRS/MSR encodings, see module doc).
pub type hv_sys_reg_t = u16;
pub const HV_SYS_REG_SP_EL0: hv_sys_reg_t = 0xc208; // op0 3 op1 0 CRn 4 CRm 1 op2 0
pub const HV_SYS_REG_SPSR_EL1: hv_sys_reg_t = 0xc200; // 3 0 4 0 0
pub const HV_SYS_REG_ELR_EL1: hv_sys_reg_t = 0xc201; // 3 0 4 0 1
pub const HV_SYS_REG_SCTLR_EL1: hv_sys_reg_t = 0xc080; // 3 0 1 0 0
pub const HV_SYS_REG_CPACR_EL1: hv_sys_reg_t = 0xc082; // 3 0 1 0 2
pub const HV_SYS_REG_ESR_EL1: hv_sys_reg_t = 0xc290; // 3 0 5 2 0
pub const HV_SYS_REG_FAR_EL1: hv_sys_reg_t = 0xc300; // 3 0 6 0 0
pub const HV_SYS_REG_VBAR_EL1: hv_sys_reg_t = 0xc600; // 3 0 12 0 0
pub const HV_SYS_REG_TPIDR_EL0: hv_sys_reg_t = 0xde82; // 3 3 13 0 2

#[link(name = "Hypervisor", kind = "framework")]
unsafe extern "C" {
    /// Create the process's VM. `config` may be null for defaults. Fails with a
    /// non-zero `HV_DENIED` if the process lacks `com.apple.security.hypervisor`.
    pub fn hv_vm_create(config: *const c_void) -> hv_return_t;
    pub fn hv_vm_destroy() -> hv_return_t;
    /// Map host memory `addr` into the guest at IPA `ipa` (all of `addr`, `ipa`,
    /// `size` must be host-page (16 KiB) aligned) with `flags` permissions.
    pub fn hv_vm_map(addr: *mut c_void, ipa: hv_ipa_t, size: usize, flags: u64) -> hv_return_t;
    pub fn hv_vm_unmap(ipa: hv_ipa_t, size: usize) -> hv_return_t;
    pub fn hv_vm_protect(ipa: hv_ipa_t, size: usize, flags: u64) -> hv_return_t;
    /// Create a vcpu bound to the calling thread; `*exit` receives a pointer to
    /// the framework-owned [`hv_vcpu_exit_t`] updated by each `hv_vcpu_run`.
    pub fn hv_vcpu_create(
        vcpu: *mut hv_vcpu_t,
        exit: *mut *mut hv_vcpu_exit_t,
        config: *const c_void,
    ) -> hv_return_t;
    pub fn hv_vcpu_destroy(vcpu: hv_vcpu_t) -> hv_return_t;
    /// Run the vcpu until it exits. Must be called on the creating thread.
    pub fn hv_vcpu_run(vcpu: hv_vcpu_t) -> hv_return_t;
    pub fn hv_vcpu_get_reg(vcpu: hv_vcpu_t, reg: hv_reg_t, value: *mut u64) -> hv_return_t;
    pub fn hv_vcpu_set_reg(vcpu: hv_vcpu_t, reg: hv_reg_t, value: u64) -> hv_return_t;
    pub fn hv_vcpu_get_sys_reg(vcpu: hv_vcpu_t, reg: hv_sys_reg_t, value: *mut u64) -> hv_return_t;
    pub fn hv_vcpu_set_sys_reg(vcpu: hv_vcpu_t, reg: hv_sys_reg_t, value: u64) -> hv_return_t;
    /// Force the listed vcpus out of `hv_vcpu_run` (for preemption from another
    /// thread).
    pub fn hv_vcpus_exit(vcpus: *const hv_vcpu_t, count: u32) -> hv_return_t;
}

unsafe extern "C" {
    /// libSystem: make the instruction cache coherent with host stores to
    /// `[start, start+len)` — required after loading guest code into mapped RAM.
    pub fn sys_icache_invalidate(start: *mut c_void, len: usize);
}
