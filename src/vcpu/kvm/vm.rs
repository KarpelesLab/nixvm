//! The KVM virtual machine: fds, the single guest-RAM memslot, and CPUID.
//!
//! One [`Vm`] per [`super::KvmBackend`] (KVM has no per-process VM quota, so —
//! unlike HVF's process-global VM — each backend owns its own, which keeps
//! parallel `cargo test` kernels from sharing one guest-physical space).
//!
//! Since the Phase-3 MMU refactor the guest-physical space is trivial: **one**
//! RAM memslot mapping the whole shared physical pool
//! ([`crate::vcpu::phys::PhysMem`]) at `guest_phys == 0`, registered once (on the
//! first run) and never re-pointed. Every process has real page tables over that
//! pool and its own `CR3`; a context switch is just a `KVM_SET_SREGS` of `cr3`,
//! not a memslot swap. The supervisor scaffold (GDT/IDT/TSS/trampolines) lives in
//! pinned pool frames mapped into every address space by [`crate::vcpu::ctrl`];
//! there is no separate control-block slot and no shadow page-table builder.
//!
//! A guest `syscall` at CPL3 vectors to the `hlt; sysretq` trampoline at CPL0;
//! `hlt` exits to the host (`KVM_EXIT_HLT`), the kernel services the syscall, and
//! the resumed `sysretq` drops back to CPL3. A `#PF` vectors (via the IDT's only
//! gate) to the `#PF` trampoline (`hlt`); the host reads the exception frame the
//! CPU pushed onto the kernel stack out of the pool.

use super::sys;
use crate::vcpu::VcpuError;
use crate::vcpu::mem::GuestMemory;
use crate::vcpu::phys::PhysMem;
use std::ffi::c_int;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

// Re-export the control-block virtual addresses and selectors the vcpu wires into
// its sregs/MSRs, so vcpu.rs keeps importing them from `super::vm`.
pub use crate::vcpu::ctrl::{
    FAULT_TRAMP_VA, GDT_BASE, GDT_LIMIT, IDT_BASE, LSTAR_VA, SEL_TSS, SEL_UCODE, SEL_UDATA,
    STAR_VALUE, TSS_BASE,
};

/// An owned Unix fd that closes on drop.
#[derive(Debug)]
pub struct Fd(pub c_int);

impl Drop for Fd {
    fn drop(&mut self) {
        // SAFETY: `self.0` is a live fd owned by this value, closed only here.
        unsafe { sys::close(self.0) };
    }
}

/// Result of `ioctl` calls: `Ok(ret)` for `ret >= 0`, else the OS error.
pub fn check(ret: c_int, what: &str) -> Result<c_int, VcpuError> {
    if ret >= 0 {
        Ok(ret)
    } else {
        Err(VcpuError::Backend(format!(
            "{what} failed: {}",
            std::io::Error::last_os_error()
        )))
    }
}

/// One KVM virtual machine. Holds `/dev/kvm`, the VM fd, the supported-CPUID
/// snapshot, and — once the first vcpu runs — a handle on the shared pool whose
/// single memslot it registered.
#[derive(Debug)]
pub struct Vm {
    _sys_fd: Fd,
    fd: Fd,
    vcpu_mmap_size: usize,
    cpuid: Box<sys::kvm_cpuid2>,
    /// The shared physical pool, registered as the one RAM memslot on the first
    /// run. `None` until then. Kept so the fault path can read the exception frame
    /// out of the control block without a `GuestMemory` borrow (the SMP path holds
    /// none). All address spaces share this pool, so it is registered exactly once.
    pool: Mutex<Option<std::sync::Arc<PhysMem>>>,
    next_vcpu_id: AtomicU32,
}

// SAFETY: `Vm` is shared (`Arc`) across the vcpus of one kernel. The pool handle
// lives behind a `Mutex`; the fds/ids are ioctl handles the kernel serializes
// internally. `PhysMem`'s bytes are the shared physical RAM — concurrent access
// to distinct frames is the model, serialized where it matters by the kernel lock.
unsafe impl Send for Vm {}
// SAFETY: see the `Send` note above.
unsafe impl Sync for Vm {}

impl Vm {
    /// Open `/dev/kvm` and build a VM. No memslot is registered yet — the pool
    /// isn't known until the first `GuestMemory` runs.
    pub fn new() -> Result<Self, VcpuError> {
        // SAFETY: a NUL-terminated path literal; `Fd` owns the returned fd.
        let raw = unsafe { sys::open(c"/dev/kvm".as_ptr(), sys::O_RDWR | sys::O_CLOEXEC) };
        let sys_fd = Fd(check(raw, "open(/dev/kvm)")?);

        // SAFETY: `KVM_GET_API_VERSION` takes no argument; the fd is live.
        let version = unsafe { sys::ioctl(sys_fd.0, sys::KVM_GET_API_VERSION, 0) };
        if version != sys::KVM_API_VERSION {
            return Err(VcpuError::Backend(format!(
                "KVM API version {version} (want {})",
                sys::KVM_API_VERSION
            )));
        }

        // SAFETY: `KVM_CREATE_VM` takes the machine type (0 = default).
        let raw = unsafe { sys::ioctl(sys_fd.0, sys::KVM_CREATE_VM, 0) };
        let fd = Fd(check(raw, "KVM_CREATE_VM")?);

        // Intel without "unrestricted guest" needs a TSS window; harmless else.
        // SAFETY: takes a plain GPA argument; the fd is live.
        unsafe { sys::ioctl(fd.0, sys::KVM_SET_TSS_ADDR, 0xFFFB_D000u64) };

        // SAFETY: `KVM_GET_VCPU_MMAP_SIZE` takes no argument.
        let raw = unsafe { sys::ioctl(sys_fd.0, sys::KVM_GET_VCPU_MMAP_SIZE, 0) };
        let vcpu_mmap_size = check(raw, "KVM_GET_VCPU_MMAP_SIZE")? as usize;

        // Snapshot the host's supported CPUID so guest `cpuid` reports real
        // features (musl/libgcc probe SSE levels at startup).
        let mut cpuid = Box::new(sys::kvm_cpuid2 {
            nent: sys::CPUID_CAP as u32,
            ..Default::default()
        });
        // SAFETY: `cpuid` is a live, writable allocation whose `nent` bounds the
        // entry array; the kernel fills entries and shrinks `nent`.
        let raw = unsafe {
            sys::ioctl(
                sys_fd.0,
                sys::KVM_GET_SUPPORTED_CPUID,
                std::ptr::from_mut(cpuid.as_mut()),
            )
        };
        check(raw, "KVM_GET_SUPPORTED_CPUID")?;

        Ok(Self {
            _sys_fd: sys_fd,
            fd,
            vcpu_mmap_size,
            cpuid,
            pool: Mutex::new(None),
            next_vcpu_id: AtomicU32::new(0),
        })
    }

    /// Issue `KVM_SET_USER_MEMORY_REGION` for `slot`.
    fn set_slot(&self, slot: u32, gpa: u64, size: u64, host: *mut u8) -> Result<(), VcpuError> {
        let region = sys::kvm_userspace_memory_region {
            slot,
            flags: 0,
            guest_phys_addr: gpa,
            memory_size: size,
            userspace_addr: host as u64,
        };
        // SAFETY: `region` is valid for the ioctl's duration; `host` points at a
        // live allocation of at least `size` bytes owned by the pool for its life.
        let ret = unsafe {
            sys::ioctl(
                self.fd.0,
                sys::KVM_SET_USER_MEMORY_REGION,
                std::ptr::from_ref(&region),
            )
        };
        check(ret, "KVM_SET_USER_MEMORY_REGION").map(drop)
    }

    /// Register the shared pool as the one RAM memslot. In production this fires
    /// exactly once: every address space (fork child, execve image) shares the one
    /// pool minted for pid 1, so the memslot is never re-pointed. The pool-changed
    /// branch exists only for tests that drive one backend across several
    /// independent `GuestMemory::new` pools; a serial run makes re-pointing safe.
    pub fn ensure_memslot(&self, mem: &GuestMemory) -> Result<(), VcpuError> {
        let mut pool = self.pool.lock().unwrap();
        let want = mem.phys_ptr();
        if pool.as_ref().map(|p| p.as_ptr()) != Some(want) {
            if pool.is_some() {
                self.set_slot(0, 0, 0, std::ptr::null_mut())?; // delete the stale slot
            }
            self.set_slot(0, 0, mem.phys_len(), want)?;
            *pool = Some(mem.phys_arc());
        }
        Ok(())
    }

    /// Create a vcpu: its fd and the shared `kvm_run` page.
    pub fn create_vcpu(&self) -> Result<(Fd, *mut sys::kvm_run), VcpuError> {
        let id = self.next_vcpu_id.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `KVM_CREATE_VCPU` takes the vcpu id; the fd is owned below.
        let raw = unsafe { sys::ioctl(self.fd.0, sys::KVM_CREATE_VCPU, u64::from(id)) };
        let fd = Fd(check(raw, "KVM_CREATE_VCPU")?);
        // SAFETY: mapping `vcpu_mmap_size` bytes of the vcpu fd as documented; the
        // result is checked against MAP_FAILED.
        let run = unsafe {
            sys::mmap(
                std::ptr::null_mut(),
                self.vcpu_mmap_size,
                sys::PROT_READ | sys::PROT_WRITE,
                sys::MAP_SHARED,
                fd.0,
                0,
            )
        };
        if run == sys::MAP_FAILED {
            return Err(VcpuError::Backend(format!(
                "mmap(kvm_run) failed: {}",
                std::io::Error::last_os_error()
            )));
        }
        Ok((fd, run.cast()))
    }

    pub fn vcpu_mmap_size(&self) -> usize {
        self.vcpu_mmap_size
    }

    pub fn cpuid(&self) -> &sys::kvm_cpuid2 {
        &self.cpuid
    }
}
