//! Execution backends: run guest code until it needs the kernel.
//!
//! nixvm never emulates devices or interrupts. Each backend runs guest
//! instructions at the lowest privilege it can (EL0/ring 3) and hands control
//! back the moment the guest makes a syscall (or faults). The kernel services
//! the request, writes the return value into guest registers, and calls
//! [`Vcpu::run`] again.
//!
//! Three backends implement one [`Vcpu`] trait:
//!
//! * **hvf** — Hypervisor.framework, macOS/arm64. (Phase 1)
//! * **kvm** — KVM, Linux/arm64 and Linux/x86-64. (Phase 10)
//! * **interp** — a software CPU, works everywhere with no acceleration. (Phase 10)
//!
//! The kernel is written against the trait and never names a concrete backend;
//! [`select`] picks the best available one for the host + guest arch.

use crate::abi::Arch;

pub mod mem;
pub(crate) mod region;

pub use mem::{GuestMemory, MemError, Prot};

/// Why [`Vcpu::run`] returned control to the kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    /// The guest executed a syscall instruction (`svc #0` / `syscall`). Read the
    /// number/args via the `syscall_*` accessors, service it, write the result
    /// with [`Vcpu::set_syscall_ret`], then run again.
    Syscall,
    /// The guest hit an unmapped/forbidden address.
    MemFault { addr: u64, write: bool },
    /// The guest executed an illegal/undefined instruction.
    IllegalInstruction { pc: u64 },
    /// The host asked the vcpu to stop (another thread wants to run, or a
    /// deadline/signal fired). The kernel decides what to do next.
    Interrupted,
    /// The guest halted the whole machine (shouldn't happen for a userspace
    /// guest; treated as a fault).
    Halt,
}

/// Backend-level failure (never a guest-visible errno — those don't surface here).
#[derive(Debug)]
pub enum VcpuError {
    /// No backend supports this host/guest combination.
    Unsupported {
        host: &'static str,
        guest: Arch,
    },
    /// The hypervisor rejected an operation.
    Backend(String),
    Mem(MemError),
}

impl core::fmt::Display for VcpuError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Unsupported { host, guest } => {
                write!(
                    f,
                    "no vcpu backend for host {host} + guest {}",
                    guest.as_str()
                )
            }
            Self::Backend(m) => write!(f, "vcpu backend error: {m}"),
            Self::Mem(e) => write!(f, "guest memory error: {e:?}"),
        }
    }
}

impl std::error::Error for VcpuError {}

impl From<MemError> for VcpuError {
    fn from(e: MemError) -> Self {
        Self::Mem(e)
    }
}

/// A single virtual CPU executing one guest thread.
///
/// The kernel owns one `Vcpu` per running guest thread and drives it with a
/// run/serve loop. Register accessors are expressed in the guest's *syscall
/// ABI* so the kernel never needs to know which backend or arch is underneath.
pub trait Vcpu: Send {
    /// Run guest code until the next [`Exit`].
    ///
    /// `mem` is the guest address space. The software interpreter reads and
    /// writes it directly as it executes; hardware backends map it into the
    /// hypervisor (once) and the guest writes through to the same buffer.
    fn run(&mut self, mem: &mut GuestMemory) -> Result<Exit, VcpuError>;

    /// The syscall number the guest requested (arm64 `x8` / x86-64 `rax`).
    fn syscall_nr(&self) -> u64;

    /// The six syscall argument registers, in ABI order.
    fn syscall_args(&self) -> [u64; 6];

    /// Write the syscall return value into the ABI return register and advance
    /// the program counter past the syscall instruction.
    fn set_syscall_ret(&mut self, value: u64);

    /// Read a general-purpose register by its architectural index.
    fn reg(&self, idx: usize) -> u64;
    /// Write a general-purpose register by its architectural index.
    fn set_reg(&mut self, idx: usize, value: u64);

    fn pc(&self) -> u64;
    fn set_pc(&mut self, pc: u64);
    fn sp(&self) -> u64;
    fn set_sp(&mut self, sp: u64);

    /// The condition/status flags register (`RFLAGS`/`PSTATE`), read and written
    /// as one word so signal delivery can save it into the guest's `ucontext`
    /// and `rt_sigreturn` can restore it. Backends that don't model the full
    /// word return a sane default; the low condition bits are what matter.
    fn rflags(&self) -> u64 {
        0
    }
    fn set_rflags(&mut self, _value: u64) {}

    /// Read/replace the SIMD/FP register file as raw bytes (x86 `XMM0..15` as a
    /// 256-byte little-endian blob), so a signal frame can save and restore it.
    /// Backends without SIMD return an empty vector and ignore a set.
    fn simd_state(&self) -> Vec<u8> {
        Vec::new()
    }
    fn set_simd_state(&mut self, _bytes: &[u8]) {}

    /// Thread pointer (arm64 `TPIDR_EL0` / x86-64 `FS.base`), set by TLS syscalls.
    fn set_tls(&mut self, value: u64);

    /// Duplicate this vcpu's full register state (for `fork`/`clone`). The copy
    /// resumes at the same point; the kernel then sets the child's syscall
    /// return value and advances its PC.
    fn fork(&self) -> Box<dyn Vcpu>;

    /// Reset every register for `execve`: clear general/SIMD registers, flags,
    /// and TLS, then set the entry PC and initial SP for the new image.
    fn reset(&mut self, entry: u64, sp: u64);
}

/// Constructs [`Vcpu`]s that share one guest address space.
pub trait Backend {
    fn name(&self) -> &'static str;
    fn guest_arch(&self) -> Arch;

    /// Create a vcpu with its PC and SP set for a fresh thread. The guest
    /// address space is provided to [`Vcpu::run`], not here, so one backend can
    /// spawn several vcpus over the same memory.
    fn new_vcpu(&self, entry: u64, stack: u64) -> Result<Box<dyn Vcpu>, VcpuError>;
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub mod hvf;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
pub mod kvm;

pub mod interp;
pub mod interp_x86;
pub(crate) mod softfloat;

/// Pick the best backend available for the host, targeting `guest`.
///
/// Prefers hardware virtualization when the guest arch matches the host and the
/// process can create a VM; otherwise falls back to the software interpreter.
/// The fallback is what keeps an unentitled/unsigned binary (CI, plain
/// `cargo test`) working — [`hvf::HvfBackend::new`] fails there, and we drop to
/// the interpreter instead of erroring. `NIXVM_INTERP=1` skips the hardware
/// probes entirely (a debugging/parity escape hatch, the env twin of
/// `SandboxBuilder::prefer_interp`).
pub fn select(guest: Arch) -> Result<Box<dyn Backend>, VcpuError> {
    let force_interp = std::env::var_os("NIXVM_INTERP").is_some_and(|v| v == "1");
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        // When the hypervisor is unavailable/unentitled, `new` fails and we fall
        // through to the interpreter.
        if !force_interp
            && guest == Arch::Aarch64
            && let Ok(backend) = hvf::HvfBackend::new()
        {
            return Ok(Box::new(backend) as Box<dyn Backend>);
        }
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        // When `/dev/kvm` is unavailable/inaccessible (a container, CI without
        // the device), `new` fails and we fall through to the interpreter.
        if !force_interp
            && guest == Arch::X86_64
            && let Ok(backend) = kvm::KvmBackend::new()
        {
            return Ok(Box::new(backend) as Box<dyn Backend>);
        }
    }
    let _ = force_interp;
    // TODO(Phase 10): KVM on Linux/arm64.
    // x86-64 guests run on the dedicated x86 software interpreter; other guest
    // arches fall back to `interp::InterpBackend`.
    if guest == Arch::X86_64 {
        return interp_x86::X86Backend::new(guest).map(|b| Box::new(b) as Box<dyn Backend>);
    }
    interp::InterpBackend::new(guest).map(|b| Box::new(b) as Box<dyn Backend>)
}
