//! Hand-rolled KVM FFI: the ioctls, structs, and constants `vcpu::kvm` needs.
//!
//! Mirrors `vcpu::hvf::sys` in spirit — no third-party crate, just the small
//! slice of `<linux/kvm.h>` this backend actually uses, declared by hand. Every
//! constant and struct size below was verified against the system headers
//! (`ioctl` request numbers encode the struct size, so a wrong layout fails
//! loudly with `EINVAL`/`EFAULT` rather than corrupting memory).
//!
//! This module only *declares*; the callers own the `unsafe` (with SAFETY
//! notes) since correctness depends on call-site invariants (live fds, valid
//! pointers, struct layouts matching the running kernel's ABI).

#![allow(non_camel_case_types)]
#![allow(dead_code)]

use std::ffi::{c_char, c_int, c_ulong, c_void};

unsafe extern "C" {
    pub fn open(path: *const c_char, flags: c_int, ...) -> c_int;
    pub fn close(fd: c_int) -> c_int;
    pub fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
    pub fn mmap(
        addr: *mut c_void,
        len: usize,
        prot: c_int,
        flags: c_int,
        fd: c_int,
        offset: i64,
    ) -> *mut c_void;
    pub fn munmap(addr: *mut c_void, len: usize) -> c_int;
}

pub const O_RDWR: c_int = 2;
pub const O_CLOEXEC: c_int = 0o2000000;
pub const PROT_READ: c_int = 1;
pub const PROT_WRITE: c_int = 2;
pub const MAP_SHARED: c_int = 1;
pub const MAP_FAILED: *mut c_void = usize::MAX as *mut c_void;

/// The KVM API version this backend understands (`KVM_GET_API_VERSION` must
/// return it; it has been 12 since Linux 2.6.22 and is documented stable).
pub const KVM_API_VERSION: c_int = 12;

// ioctl request numbers (x86-64 Linux). The `_IO*` encodings below match the
// values printed from `<linux/kvm.h>` on this host — see the module doc.
pub const KVM_GET_API_VERSION: c_ulong = 0xAE00;
pub const KVM_CREATE_VM: c_ulong = 0xAE01;
pub const KVM_CHECK_EXTENSION: c_ulong = 0xAE03;
pub const KVM_GET_VCPU_MMAP_SIZE: c_ulong = 0xAE04;
pub const KVM_GET_SUPPORTED_CPUID: c_ulong = 0xC008_AE05;
pub const KVM_CREATE_VCPU: c_ulong = 0xAE41;
pub const KVM_SET_USER_MEMORY_REGION: c_ulong = 0x4020_AE46;
pub const KVM_SET_TSS_ADDR: c_ulong = 0xAE47;
pub const KVM_RUN: c_ulong = 0xAE80;
pub const KVM_GET_REGS: c_ulong = 0x8090_AE81;
pub const KVM_SET_REGS: c_ulong = 0x4090_AE82;
pub const KVM_GET_SREGS: c_ulong = 0x8138_AE83;
pub const KVM_SET_SREGS: c_ulong = 0x4138_AE84;
pub const KVM_SET_MSRS: c_ulong = 0x4008_AE89;
pub const KVM_GET_FPU: c_ulong = 0x81A0_AE8C;
pub const KVM_SET_FPU: c_ulong = 0x41A0_AE8D;
pub const KVM_SET_CPUID2: c_ulong = 0x4008_AE90;

// `kvm_run.exit_reason` values this backend decodes.
pub const KVM_EXIT_IO: u32 = 2;
pub const KVM_EXIT_HLT: u32 = 5;
pub const KVM_EXIT_MMIO: u32 = 6;
pub const KVM_EXIT_SHUTDOWN: u32 = 8;
pub const KVM_EXIT_FAIL_ENTRY: u32 = 9;
pub const KVM_EXIT_INTR: u32 = 10;
pub const KVM_EXIT_INTERNAL_ERROR: u32 = 17;

// MSR indices the trampoline setup writes.
pub const MSR_IA32_STAR: u32 = 0xC000_0081;
pub const MSR_IA32_LSTAR: u32 = 0xC000_0082;
pub const MSR_IA32_CSTAR: u32 = 0xC000_0083;
pub const MSR_IA32_FMASK: u32 = 0xC000_0084;
pub const MSR_KERNEL_GS_BASE: u32 = 0xC000_0102;

/// General-purpose registers (`KVM_GET_REGS`/`KVM_SET_REGS`). 144 bytes.
/// Field order is the kernel's (rax, rbx, rcx, rdx, …), *not* the x86 register
/// encoding order — the trait-facing index mapping lives in `vcpu.rs`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct kvm_regs {
    pub rax: u64,
    pub rbx: u64,
    pub rcx: u64,
    pub rdx: u64,
    pub rsi: u64,
    pub rdi: u64,
    pub rsp: u64,
    pub rbp: u64,
    pub r8: u64,
    pub r9: u64,
    pub r10: u64,
    pub r11: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
    pub rip: u64,
    pub rflags: u64,
}

/// One segment register's hidden-state descriptor cache. 24 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct kvm_segment {
    pub base: u64,
    pub limit: u32,
    pub selector: u16,
    pub type_: u8,
    pub present: u8,
    pub dpl: u8,
    pub db: u8,
    pub s: u8,
    pub l: u8,
    pub g: u8,
    pub avl: u8,
    pub unusable: u8,
    pub padding: u8,
}

/// GDTR/IDTR. 16 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct kvm_dtable {
    pub base: u64,
    pub limit: u16,
    pub padding: [u16; 3],
}

/// Special registers (`KVM_GET_SREGS`/`KVM_SET_SREGS`). 312 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct kvm_sregs {
    pub cs: kvm_segment,
    pub ds: kvm_segment,
    pub es: kvm_segment,
    pub fs: kvm_segment,
    pub gs: kvm_segment,
    pub ss: kvm_segment,
    pub tr: kvm_segment,
    pub ldt: kvm_segment,
    pub gdt: kvm_dtable,
    pub idt: kvm_dtable,
    pub cr0: u64,
    pub cr2: u64,
    pub cr3: u64,
    pub cr4: u64,
    pub cr8: u64,
    pub efer: u64,
    pub apic_base: u64,
    pub interrupt_bitmap: [u64; 4],
}

/// FPU/SSE state (`KVM_GET_FPU`/`KVM_SET_FPU`), 416 bytes. Copied opaquely for
/// `fork` — nothing in nixvm inspects the fields, so the layout is a sized blob.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct kvm_fpu(pub [u64; 52]);

impl Default for kvm_fpu {
    fn default() -> Self {
        Self([0; 52])
    }
}

/// A guest-physical memory slot (`KVM_SET_USER_MEMORY_REGION`). 32 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct kvm_userspace_memory_region {
    pub slot: u32,
    pub flags: u32,
    pub guest_phys_addr: u64,
    pub memory_size: u64,
    pub userspace_addr: u64,
}

/// One MSR write. 16 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct kvm_msr_entry {
    pub index: u32,
    pub reserved: u32,
    pub data: u64,
}

/// Fixed-capacity `struct kvm_msrs` (`KVM_SET_MSRS`): a header followed by
/// `nmsrs` entries. The trampoline setup writes at most [`MSRS_CAP`].
pub const MSRS_CAP: usize = 8;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct kvm_msrs {
    pub nmsrs: u32,
    pub pad: u32,
    pub entries: [kvm_msr_entry; MSRS_CAP],
}

/// One CPUID leaf. 40 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct kvm_cpuid_entry2 {
    pub function: u32,
    pub index: u32,
    pub flags: u32,
    pub eax: u32,
    pub ebx: u32,
    pub ecx: u32,
    pub edx: u32,
    pub padding: [u32; 3],
}

/// Fixed-capacity `struct kvm_cpuid2` (`KVM_GET_SUPPORTED_CPUID` /
/// `KVM_SET_CPUID2`): a header followed by `nent` entries. 128 leaves is
/// comfortably above what current kernels report (~40).
pub const CPUID_CAP: usize = 128;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct kvm_cpuid2 {
    pub nent: u32,
    pub padding: u32,
    pub entries: [kvm_cpuid_entry2; CPUID_CAP],
}

impl Default for kvm_cpuid2 {
    fn default() -> Self {
        Self {
            nent: 0,
            padding: 0,
            entries: [kvm_cpuid_entry2::default(); CPUID_CAP],
        }
    }
}

// The entries array is deliberately elided — 128 leaves of raw CPUID data.
#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for kvm_cpuid2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("kvm_cpuid2").field("nent", &self.nent).finish()
    }
}

/// `KVM_EXIT_MMIO` payload: the guest touched a guest-physical address no
/// memory slot covers.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct kvm_run_mmio {
    pub phys_addr: u64,
    pub data: [u8; 8],
    pub len: u32,
    pub is_write: u8,
}

/// `KVM_EXIT_FAIL_ENTRY` payload.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct kvm_run_fail_entry {
    pub hardware_entry_failure_reason: u64,
    pub cpu: u32,
}

/// `KVM_EXIT_INTERNAL_ERROR` payload.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct kvm_run_internal {
    pub suberror: u32,
    pub ndata: u32,
    pub data: [u64; 16],
}

/// The per-exit payload union at byte offset 32 of `struct kvm_run`.
#[repr(C)]
#[derive(Clone, Copy)]
pub union kvm_run_exit {
    pub mmio: kvm_run_mmio,
    pub fail_entry: kvm_run_fail_entry,
    pub internal: kvm_run_internal,
    /// Sized to the kernel's 256-byte exit union so the header layout is exact.
    pub raw: [u64; 32],
}

/// The leading, layout-stable slice of `struct kvm_run` (the shared
/// kernel↔userspace vcpu page mapped by `mmap` on the vcpu fd). Only the
/// header fields and the exit union are declared; the page's trailing shared
/// register blocks (`kvm_sync_regs` etc.) are unused here and left opaque —
/// the mmap length comes from `KVM_GET_VCPU_MMAP_SIZE`, not this type.
#[repr(C)]
pub struct kvm_run {
    pub request_interrupt_window: u8,
    pub immediate_exit: u8,
    pub padding1: [u8; 6],
    pub exit_reason: u32,
    pub ready_for_interrupt_injection: u8,
    pub if_flag: u8,
    pub flags: u16,
    pub cr8: u64,
    pub apic_base: u64,
    pub exit: kvm_run_exit,
}

// The exit union can't be printed without knowing which arm is live.
#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for kvm_run {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("kvm_run")
            .field("exit_reason", &self.exit_reason)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ioctl request numbers above encode each struct's size; a layout
    /// drift would make the kernel reject the call. Pin the sizes here so a
    /// bad edit fails at `cargo test` instead of at first KVM use.
    #[test]
    fn abi_struct_sizes_match_linux() {
        assert_eq!(size_of::<kvm_regs>(), 144);
        assert_eq!(size_of::<kvm_segment>(), 24);
        assert_eq!(size_of::<kvm_dtable>(), 16);
        assert_eq!(size_of::<kvm_sregs>(), 312);
        assert_eq!(size_of::<kvm_fpu>(), 416);
        assert_eq!(size_of::<kvm_userspace_memory_region>(), 32);
        assert_eq!(size_of::<kvm_msr_entry>(), 16);
        assert_eq!(size_of::<kvm_cpuid_entry2>(), 40);
        assert_eq!(std::mem::offset_of!(kvm_run, exit_reason), 8);
        assert_eq!(std::mem::offset_of!(kvm_run, exit), 32);
    }
}
