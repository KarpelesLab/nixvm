//! The KVM virtual machine: fds, guest-physical layout, and the control block.
//!
//! One [`Vm`] per [`super::KvmBackend`] (KVM has no per-process VM quota, so —
//! unlike HVF's process-global VM — each backend owns its own, which keeps
//! parallel `cargo test` kernels from sharing one guest-physical space). Its
//! GPA space holds two memory slots:
//!
//! * **slot 0** — the guest's contiguous [`GuestMemory`] region, re-issued
//!   whenever [`GuestMemory::backing_generation`] changes (fork/execve/context
//!   switch), mapped at `guest_phys == guest base` so guest VA == GPA.
//! * **slot 1** — the **control block** at [`CTRL_GPA`]: the identity page
//!   tables (guest VA == PA over the low 4 GiB, 2 MiB pages, user-accessible),
//!   the GDT, and the one-instruction-pair syscall trampoline
//!   (`hlt; sysretq`) that `IA32_LSTAR` points at. x86-64 long mode cannot run
//!   with paging off (unlike the arm64 MMU-off trick HVF uses), so the flat
//!   address space the interpreter models is reproduced with these fixed
//!   tables instead.
//!
//! A guest `syscall` at CPL3 vectors to the trampoline at CPL0; `hlt` there
//! exits to the host (`KVM_EXIT_HLT` — with no in-kernel irqchip the exit
//! comes straight to userspace and `KVM_RUN` resumes after the `hlt`), the
//! kernel services the syscall, and the resumed `sysretq` drops back to CPL3
//! at the saved user `rip` (`rcx`) and `rflags` (`r11`).

use super::sys;
use crate::vcpu::VcpuError;
use crate::vcpu::mem::GuestMemory;
use crate::vcpu::region::Region;
use std::ffi::c_int;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};

/// Guest-physical base of the control block (slot 1) — just under 4 GiB, the
/// top of the identity-mapped window, far above any guest region in use
/// (`run-elf*` and the sandbox place guests at `0x1_0000` with ≤ 512 MiB).
pub const CTRL_GPA: u64 = 0xFF80_0000;

// Byte offsets of the control block's pieces within its region.
const PML4_OFF: usize = 0x0000;
const PDPT_OFF: usize = 0x1000;
const PD_OFF: usize = 0x2000; // 4 pages, one per GiB mapped
const GDT_OFF: usize = 0x6000;
const TRAMP_OFF: usize = 0x7000;
const CTRL_SIZE: usize = 0x8000;

/// How much of the guest-physical space the identity map covers (2 MiB pages,
/// present/writable/user). Everything a guest can architecturally address must
/// be below this; the control block sits at its top.
const IDENTITY_TOP: u64 = 4 << 30;

/// The virtual address `IA32_LSTAR` points at: the `hlt; sysretq` trampoline
/// (identity-mapped, so VA == GPA).
pub const LSTAR_VA: u64 = CTRL_GPA + TRAMP_OFF as u64;

/// Linear address of the GDT (identity-mapped).
pub const GDT_BASE: u64 = CTRL_GPA + GDT_OFF as u64;

/// `CR3` for every vcpu: the PML4's guest-physical address.
pub const CR3_PML4: u64 = CTRL_GPA + PML4_OFF as u64;

// GDT layout (selectors). `syscall` loads CS/SS from `STAR[47:32]` (= 0x08,
// +8 for SS); `sysretq` loads them from `STAR[63:48]` (= 0x13: +16 → CS
// 0x20|RPL3 = 0x23, +8 → SS 0x18|RPL3 = 0x1B). The GDT entries themselves are
// the classic flat 64-bit descriptors.
pub const SEL_KCODE: u16 = 0x08;
#[allow(dead_code)] // syscall's SS is implicitly SEL_KCODE + 8; named for the GDT map above
pub const SEL_KDATA: u16 = 0x10;
pub const SEL_UDATA: u16 = 0x18 | 3;
pub const SEL_UCODE: u16 = 0x20 | 3;
/// `IA32_STAR`: `sysretq` base selector in [63:48], `syscall` CS in [47:32].
pub const STAR_VALUE: u64 =
    (((SEL_UDATA as u64 & !3) - 8) | 3) << 48 | (SEL_KCODE as u64) << 32;

/// Page-table entry bits: present, writable, user-accessible; PS marks a
/// 2 MiB leaf in a PD entry.
const PTE_P_RW_US: u64 = 0b111;
const PTE_PS: u64 = 1 << 7;

/// The number of GDT bytes in use (5 descriptors), for the GDTR limit.
pub const GDT_LIMIT: u16 = 5 * 8 - 1;

/// An owned Unix fd that closes on drop (std's `OwnedFd` would work too, but
/// the raw `c_int` keeps this module symmetric with the hand-rolled FFI).
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

/// The state of guest slot 0, guarded so concurrent vcpus (SMP is a later
/// milestone, but `fork` already means several vcpus share one VM) reconcile
/// consistently.
#[derive(Debug, Default)]
struct GuestSlot {
    /// `backing_generation` of the [`GuestMemory`] currently mapped.
    generation: Option<u64>,
}

/// One KVM virtual machine. Holds `/dev/kvm`, the VM fd, the control-block
/// memory, and the supported-CPUID snapshot each vcpu is initialized with.
#[derive(Debug)]
pub struct Vm {
    /// `/dev/kvm`, retained so the system fd outlives the VM (closed on drop;
    /// future system ioctls — `KVM_CHECK_EXTENSION` — go through it).
    _sys_fd: Fd,
    fd: Fd,
    /// Size of the per-vcpu `kvm_run` mmap.
    vcpu_mmap_size: usize,
    /// Backing for slot 1 (page tables, GDT, trampoline). Owned so it outlives
    /// the kernel's mapping of it; never written after construction.
    ctrl: Region,
    /// `KVM_GET_SUPPORTED_CPUID` snapshot handed to each vcpu.
    cpuid: Box<sys::kvm_cpuid2>,
    guest_slot: Mutex<GuestSlot>,
    next_vcpu_id: AtomicU32,
}

// SAFETY: `Vm` is shared (`Arc`) across the vcpus of one kernel. Its only
// non-`Sync` member is the `Region` behind `ctrl` (a raw allocation), which is
// written exclusively during `Vm::new` and only read (its pointer) afterwards;
// the mutable slot state lives behind a `Mutex` and the fds/ids are ioctl
// handles the kernel serializes internally.
unsafe impl Send for Vm {}
// SAFETY: see the `Send` note above.
unsafe impl Sync for Vm {}

impl Vm {
    /// Open `/dev/kvm` and build a VM with the control block mapped. Any
    /// failure (no `/dev/kvm`, no permission, wrong API version) is returned
    /// as a backend error — which [`crate::vcpu::select`] turns into an
    /// interpreter fallback.
    pub fn new() -> Result<Self, VcpuError> {
        // SAFETY: a NUL-terminated path literal; `open` allocates nothing we
        // must free besides the fd, which `Fd` owns.
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

        // SAFETY: `KVM_CREATE_VM` takes the machine type (0 = default); the
        // returned fd is owned by `Fd`.
        let raw = unsafe { sys::ioctl(sys_fd.0, sys::KVM_CREATE_VM, 0) };
        let fd = Fd(check(raw, "KVM_CREATE_VM")?);

        // Intel without "unrestricted guest" needs a TSS window for real-mode
        // emulation; harmless elsewhere. Failure is non-fatal by design.
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
        // SAFETY: `cpuid` is a live, writable allocation whose `nent` bounds
        // the entry array; the kernel fills entries and shrinks `nent`.
        let raw = unsafe {
            sys::ioctl(
                sys_fd.0,
                sys::KVM_GET_SUPPORTED_CPUID,
                std::ptr::from_mut(cpuid.as_mut()),
            )
        };
        check(raw, "KVM_GET_SUPPORTED_CPUID")?;

        let ctrl = build_control_block();
        let vm = Self {
            _sys_fd: sys_fd,
            fd,
            vcpu_mmap_size,
            ctrl,
            cpuid,
            guest_slot: Mutex::new(GuestSlot::default()),
            next_vcpu_id: AtomicU32::new(0),
        };
        vm.set_slot(1, CTRL_GPA, CTRL_SIZE as u64, vm.ctrl.as_ptr())?;
        Ok(vm)
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
        // SAFETY: `region` is a valid struct for the ioctl's duration;
        // `host` points at a live allocation of at least `size` bytes owned by
        // the caller for the mapping's lifetime (a `Region`).
        let ret = unsafe {
            sys::ioctl(
                self.fd.0,
                sys::KVM_SET_USER_MEMORY_REGION,
                std::ptr::from_ref(&region),
            )
        };
        check(ret, "KVM_SET_USER_MEMORY_REGION").map(drop)
    }

    /// (Re)issue guest slot 0 if `mem`'s backing changed since the last run —
    /// the seam that makes fork/execve (a new backing) and a future context
    /// switch just work. The mapping is `guest_phys == mem.base()`, so the
    /// guest's flat virtual addresses hit the identity map and land on the
    /// same bytes the kernel's copy-in/out reads.
    pub fn reconcile_guest(&self, mem: &GuestMemory) -> Result<(), VcpuError> {
        let generation = mem.backing_generation();
        let mut slot = self.guest_slot.lock().unwrap();
        if slot.generation == Some(generation) {
            return Ok(());
        }
        let end = mem.base() + mem.size();
        if end > CTRL_GPA {
            return Err(VcpuError::Backend(format!(
                "guest region [{:#x}, {end:#x}) reaches past the control block at {CTRL_GPA:#x}",
                mem.base()
            )));
        }
        if slot.generation.is_some() {
            // Delete the stale slot (size 0) before mapping the new backing.
            self.set_slot(0, 0, 0, std::ptr::null_mut())?;
        }
        self.set_slot(0, mem.base(), mem.size(), mem.host_base())?;
        slot.generation = Some(generation);
        Ok(())
    }

    /// Create a vcpu: its fd and the shared `kvm_run` page. vcpu ids are never
    /// reused (KVM keeps a vcpu alive until the VM dies, even after its fd
    /// closes), so a pathological fork storm eventually hits the kernel's
    /// max-vcpu cap — an accepted limit of this milestone.
    pub fn create_vcpu(&self) -> Result<(Fd, *mut sys::kvm_run), VcpuError> {
        let id = self.next_vcpu_id.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `KVM_CREATE_VCPU` takes the vcpu id; the fd is owned below.
        let raw = unsafe { sys::ioctl(self.fd.0, sys::KVM_CREATE_VCPU, u64::from(id)) };
        let fd = Fd(check(raw, "KVM_CREATE_VCPU")?);
        // SAFETY: mapping `vcpu_mmap_size` bytes of the vcpu fd as the kernel
        // documents; the resulting pointer is checked against MAP_FAILED.
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

/// Build the control block: identity page tables over the low 4 GiB (2 MiB
/// pages, present/writable/user — per-page W^X is a later milestone, matching
/// HVF's uniformly-RWX stage 2), the flat GDT, and the syscall trampoline.
fn build_control_block() -> Region {
    let mut ctrl = Region::new(CTRL_SIZE);

    // PML4[0] → PDPT.
    let pml4e = (CTRL_GPA + PDPT_OFF as u64) | PTE_P_RW_US;
    ctrl.write(PML4_OFF, &pml4e.to_le_bytes());

    // PDPT[0..4] → one PD per GiB.
    for g in 0..4usize {
        let pdpte = (CTRL_GPA + (PD_OFF + g * 0x1000) as u64) | PTE_P_RW_US;
        ctrl.write(PDPT_OFF + g * 8, &pdpte.to_le_bytes());
    }

    // PD entries: 2 MiB identity leaves covering [0, IDENTITY_TOP).
    let mut gpa = 0u64;
    let mut off = PD_OFF;
    while gpa < IDENTITY_TOP {
        let pde = gpa | PTE_P_RW_US | PTE_PS;
        ctrl.write(off, &pde.to_le_bytes());
        gpa += 2 << 20;
        off += 8;
    }

    // GDT: null, kernel code64, kernel data, user data, user code64 — the
    // classic flat descriptors, in the order `syscall`/`sysretq` expect.
    let gdt: [u64; 5] = [
        0,
        0x00AF_9A00_0000_FFFF, // 0x08 kernel code (L=1, DPL0)
        0x00CF_9200_0000_FFFF, // 0x10 kernel data
        0x00CF_F200_0000_FFFF, // 0x18 user data  (DPL3)
        0x00AF_FA00_0000_FFFF, // 0x20 user code  (L=1, DPL3)
    ];
    for (i, d) in gdt.iter().enumerate() {
        ctrl.write(GDT_OFF + i * 8, &d.to_le_bytes());
    }

    // The trampoline `IA32_LSTAR` points at: exit to the host, then return to
    // the guest. `syscall` leaves the user rip in rcx and rflags in r11, which
    // `sysretq` restores — so between the two, the host services the syscall
    // and writes the return value into rax.
    ctrl.write(TRAMP_OFF, &[0xF4, 0x48, 0x0F, 0x07]); // hlt ; sysretq

    ctrl
}
