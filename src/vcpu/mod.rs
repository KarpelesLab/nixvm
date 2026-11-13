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

pub(crate) mod ctrl;
pub mod filemap;
pub mod mem;
mod pagelocks;
pub(crate) mod pagetable;
mod pagevec;
pub(crate) mod phys;
pub(crate) mod region;
pub(crate) mod vdso;

pub use mem::{FileFault, GuestMemory, MemError, Prot};
pub use pagetable::PtFormat;

/// The wall-clock preemption quantum shared by every backend.
///
/// Syscall-count preemption (the scheduler's `slice_cap`) has a hole: a
/// compute-bound guest thread that issues *no* syscalls — a JIT'd hot loop, a
/// GC sweep, a crypto kernel — never hands control back, so it monopolizes its
/// CPU for the whole run. The time quantum closes that hole: after roughly this
/// long, a running vcpu's [`Vcpu::run`] returns [`Exit::Interrupted`] even mid
/// computation, and the scheduler resumes the (still-runnable) task on its next
/// turn. The interpreter polls the wall clock in its instruction loop; the KVM
/// backend arms a per-thread timer whose signal breaks `KVM_RUN` out.
///
/// Configured once by `NIXVM_QUANTUM_MS` (default 10 ms); `0` disables
/// time-based preemption, leaving only the syscall-count cap. Read and cached on
/// first use.
#[must_use]
pub(crate) fn preempt_quantum() -> Option<std::time::Duration> {
    // wasm32-unknown-unknown has no monotonic clock: `Instant::now()` panics
    // ("time not implemented on this platform"). Time-based preemption is a
    // scheduling nicety, not a correctness requirement — the per-run `MAX_STEPS`
    // budget still bounds a compute-bound slice — so disable it there rather
    // than crash the browser demo the first time the x86 interpreter runs.
    #[cfg(target_arch = "wasm32")]
    {
        return None;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::OnceLock;
        static Q: OnceLock<Option<std::time::Duration>> = OnceLock::new();
        *Q.get_or_init(|| {
            let ms = std::env::var("NIXVM_QUANTUM_MS")
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
                .unwrap_or(10);
            (ms > 0).then(|| std::time::Duration::from_millis(ms))
        })
    }
}

thread_local! {
    /// When this thread's embedder wants control back (monotonic ns; 0 = no
    /// deadline). Set by [`crate::kernel::Kernel::pump_for`] so a single-
    /// threaded embedder (the browser tab) gets its event loop back while the
    /// guest is still computing.
    static YIELD_AT_NS: std::cell::Cell<u128> = const { std::cell::Cell::new(0) };
}

/// Ask this thread's running vcpu to hand control back once the monotonic
/// clock passes `at` (`None` clears it).
pub(crate) fn set_yield_deadline(at: Option<std::time::Duration>) {
    YIELD_AT_NS.with(|c| c.set(at.map_or(0, |d| d.as_nanos().max(1))));
}

/// Whether this thread's yield deadline (if any) has passed. Interpreters
/// poll it every few thousand instructions and return [`Exit::Interrupted`].
pub(crate) fn yield_due() -> bool {
    let at = YIELD_AT_NS.with(std::cell::Cell::get);
    at != 0 && crate::clock::now_monotonic().as_nanos() >= at
}

/// Size of the x86-64 `XSAVE` area [`Vcpu::simd_state`] uses (x87, SSE and
/// AVX components, standard format: `CPUID.(EAX=0xD,ECX=0).EBX`).
pub const X86_XSAVE_SIZE: usize = 0x340;

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
    /// The guest executed a software breakpoint: arm64 `BRK #imm` (`code` =
    /// the 16-bit immediate) or x86 `int3` (`code` = 0). Linux answers with
    /// `SIGTRAP` (`TRAP_BRKPT` on arm64, `SI_KERNEL` on x86-64) and leaves
    /// the pc on the arm64 `BRK` (after the x86 `int3`).
    Breakpoint { pc: u64, code: u64 },
    /// An alignment fault: a misaligned SP-based access or PC (arm64's SP/PC
    /// alignment checks), or an atomic/ordered access crossing its
    /// single-copy-atomicity granule. Linux answers with `SIGBUS`
    /// (`BUS_ADRALN`, `si_addr` = `addr`), not `SIGSEGV`.
    Misaligned { addr: u64, write: bool },
    /// An arithmetic exception at the instruction at `pc` — x86 `#DE`
    /// (`vector` 0), x87 `#MF` (16), SIMD `#XM` (19) — that Linux delivers as
    /// `SIGFPE` with `si_code` = `code` (`FPE_INTDIV`, `FPE_FLTDIV`, …, as the
    /// backend derived it from the unmasked status/`MXCSR` bits).
    ArithmeticFault { pc: u64, code: u64, vector: u64 },
    /// x86 `#GP` (a privileged instruction at CPL 3, a misaligned `MOVAPS`,
    /// reserved `MXCSR` bits, …) at `pc`: Linux's `SIGSEGV` with `SI_KERNEL`
    /// and no address.
    ProtectionFault { pc: u64 },
    /// A debug trap (`#DB` on x86, a software-step or watchpoint exception
    /// on arm64), with the task stopped at `pc`, where it resumes: a single
    /// step armed by [`Vcpu::set_single_step`] completed (`dr6` has bit 14,
    /// `BS`), or a hardware breakpoint/watchpoint armed by
    /// [`Vcpu::set_debug_regs`] matched (`dr6` bits 0-3, which slots). Linux
    /// answers with `SIGTRAP` — `TRAP_TRACE` for a step, `TRAP_HWBKPT` for a
    /// match.
    DebugTrap { pc: u64, dr6: u64 },
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

/// vDSO clock calibration produced by [`Vcpu::vdso_calibration`] and written to
/// the guest's vvar page by `ctrl::write_vvar`. The guest's
/// `__vdso_clock_gettime` computes `base_*_ns + ((rdtsc() - base_tsc) * mult) >>
/// shift`.
#[derive(Clone, Copy, Debug)]
pub struct VdsoCal {
    pub mult: u64,
    pub shift: u64,
    pub base_tsc: u64,
    pub base_mono_ns: u64,
    pub base_wall_ns: u64,
}

/// Something that interrupts a vcpu running on another host thread, so it
/// returns [`Exit::Interrupted`] promptly (see [`Vcpu::kicker`]). Calling it
/// when the vcpu isn't running is harmless: at worst its next run returns at
/// once.
pub type Kick = std::sync::Arc<dyn Fn() + Send + Sync>;

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

    /// Whether [`Vcpu::run`] must hold the `GuestMemory` lock for its whole
    /// duration. The software interpreter reads and writes guest memory *through*
    /// the `GuestMemory` struct as it executes, so it genuinely needs the lock the
    /// entire time — `true`. Hardware backends (KVM) execute against a memory slot
    /// the hypervisor maps once; guest loads/stores go straight to the shared
    /// backing with no Rust borrow, so the SMP scheduler can [`Vcpu::reconcile`]
    /// under the lock, drop it, and then run the guest in parallel with siblings
    /// of the same address space — `false`.
    fn needs_locked_run(&self) -> bool {
        true
    }

    /// Prepare the backend's view of `mem` for a subsequent lockless
    /// [`Vcpu::run_bare`] (the SMP path). Hardware backends sync their memory
    /// slots and shadow page tables here — the one step that must see `mem` — so
    /// that `run_bare` needs no `GuestMemory` borrow. A no-op for backends whose
    /// `needs_locked_run` is `true` (the interpreter), which are never split this
    /// way.
    fn reconcile(&mut self, _mem: &mut GuestMemory) -> Result<(), VcpuError> {
        Ok(())
    }

    /// Run guest code after a prior [`Vcpu::reconcile`], without a `GuestMemory`
    /// borrow — the lockless half of the SMP split. Only called on backends whose
    /// [`Vcpu::needs_locked_run`] is `false`; the default panics so a mis-paired
    /// backend fails loudly instead of silently running unreconciled.
    fn run_bare(&mut self) -> Result<Exit, VcpuError> {
        unreachable!("run_bare on a backend that requires a locked run")
    }

    /// Whether a memory fault at `addr` is only a *stale shadow* fault: the page
    /// is mapped/accessible in `mem`, but this backend's page tables — synced at
    /// the vcpu's last dispatch — do not yet reflect a mapping change a sibling
    /// made mid-run. The SMP scheduler retries such a fault (re-dispatch
    /// reconciles the tables) rather than turning it into a signal. Always `false`
    /// for backends without lazily-synced shadow tables (the interpreter, which
    /// consults `mem` directly, and serial KVM, which reconciles before every run).
    fn shadow_stale(&self, _mem: &GuestMemory, _addr: u64) -> bool {
        false
    }

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

    /// At a syscall exit: the user address the syscall returns to — the
    /// instruction after `syscall`/`svc`, Linux's `instruction_pointer()`
    /// there (what seccomp's `seccomp_data.instruction_pointer` and a
    /// `SIGSYS`'s `si_call_addr` report).
    fn syscall_ip(&self) -> u64 {
        self.pc()
    }
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

    /// Read/replace the SIMD/FP register file as raw bytes, so a signal frame
    /// can save and restore it: on arm64 the `fpsimd_context` payload (FPSR,
    /// FPCR, V0–V31; 520 bytes); on x86-64 the standard-format `XSAVE` area
    /// of every enabled component (legacy x87/SSE region, header, AVX upper
    /// halves; [`X86_XSAVE_SIZE`] bytes). An x86 set also accepts a bare
    /// 512-byte `FXSAVE` image (the `fxrstor`-only restore: the extended
    /// components are reset). Backends without SIMD state return an empty
    /// vector and ignore a set.
    fn simd_state(&self) -> Vec<u8> {
        Vec::new()
    }
    fn set_simd_state(&mut self, _bytes: &[u8]) {}

    /// The FP/SIMD state for a debugger (`PTRACE_GETFPREGS`, the FP regsets),
    /// in the [`Vcpu::simd_state`] layout. The same thing unless a backend
    /// leaves that empty for signal frames but can still read it on demand.
    fn fp_state(&self) -> Vec<u8> {
        self.simd_state()
    }
    fn set_fp_state(&mut self, bytes: &[u8]) {
        self.set_simd_state(bytes);
    }

    /// Arm (or disarm) a single step: once the next user instruction
    /// completes, [`Vcpu::run`] returns [`Exit::DebugTrap`] with `BS` set
    /// instead of carrying on. One-shot: the trap disarms it, and so does a
    /// syscall instruction (the kernel reports a step over a syscall itself,
    /// at the syscall's return). A fault doesn't: the instruction hasn't
    /// completed, and re-running it still traps after it. Called with the
    /// vcpu in user mode, or (KVM) with a syscall's result already written
    /// back, which then returns to user mode first. Backends without it
    /// ignore it.
    fn set_single_step(&mut self, _on: bool) {}

    /// Load the x86 debug registers a debugger set for this task: `dr[0..4]`
    /// the addresses, `dr[7]` the control word (local/global enables, R/W and
    /// LEN fields); `dr[4..7]` are ignored. A matching access stops the task
    /// with [`Exit::DebugTrap`] (an execute breakpoint before the
    /// instruction, a data watchpoint after it). Whether the backend has
    /// them.
    fn set_debug_regs(&mut self, _dr: &[u64; 8]) -> bool {
        false
    }

    /// Thread pointer (arm64 `TPIDR_EL0` / x86-64 `FS.base`), set by TLS syscalls.
    fn set_tls(&mut self, value: u64);

    /// Duplicate this vcpu's full register state (for `fork`/`clone`). The copy
    /// resumes at the same point; the kernel then sets the child's syscall
    /// return value and advances its PC.
    ///
    /// # Errors
    /// The backend can't provide another vcpu right now, as with KVM's limit on
    /// live vcpus; the guest's `fork` fails rather than the host process.
    fn fork(&self) -> Result<Box<dyn Vcpu>, VcpuError>;

    /// Reset every register for `execve`: clear general/SIMD registers, flags,
    /// and TLS, then set the entry PC and initial SP for the new image.
    fn reset(&mut self, entry: u64, sp: u64);

    /// Collapse the syscall-return trampoline into the logical user state — a
    /// software `sysretq`. A KVM guest servicing a syscall is parked at CPL0 on
    /// the trampoline's `sysretq` (rip on the trampoline, the user return in
    /// `rcx` and user flags in `r11`); the host normally lets the guest execute
    /// that `sysretq` to drop back to CPL3. When the host instead redirects the
    /// guest elsewhere — into a signal handler, or back out of `rt_sigreturn` —
    /// `sysretq` never runs, so the vcpu would keep executing user code at CPL0
    /// (supervisor). This adopts the user context (rip←`rcx`, rflags←`r11`) and
    /// switches the segments to flat CPL3, exactly as `sysretq` would, so the
    /// subsequent redirect lands at user privilege. TLS (`fs.base`) is untouched.
    /// A no-op for the interpreter, which has no trampoline — its pc already sits
    /// on the post-syscall user instruction at no privilege level.
    fn settle_syscall_return(&mut self) {}

    /// Whether an [`Exit::Interrupted`] left the vcpu at a user-mode
    /// instruction boundary, where the kernel may deliver a pending
    /// asynchronous signal (build a frame from the live registers and redirect
    /// the pc) as Linux does on any return to user mode — not only at syscall
    /// returns. A compute loop that never makes a syscall (a `SIGALRM`-driven
    /// benchmark, Go's `SIGURG` goroutine preemption) depends on it. True for
    /// the interpreters, which only stop between guest instructions; the
    /// hardware backends may be interrupted inside their trampolines and
    /// answer `false`, deferring delivery to the next syscall.
    fn async_signal_boundary(&self) -> bool {
        true
    }

    /// One-time vDSO clock calibration: read the guest TSC frequency and its
    /// current value and correlate it with the host wall clock, so the guest's
    /// `vdso` can compute `clock_gettime` from `rdtsc` with no syscall. `None`
    /// when the backend can't (the interpreter — no real TSC) or calibration
    /// failed; the caller then leaves the vvar page zeroed and the vDSO falls
    /// back to the syscall. Read once, before the guest runs.
    fn vdso_calibration(&self) -> Option<VdsoCal> {
        None
    }

    /// Flush this vcpu's TLB before its next run, because the host modified the
    /// page tables of the address space it is running *without* the guest issuing
    /// an `invlpg`/`mov cr3`. The canonical case is `fork`: making the parent's
    /// pages copy-on-write read-only would otherwise be masked by stale writable
    /// TLB entries, letting the parent write straight through a shared frame.
    /// A no-op for the interpreter, which re-walks the page tables on every access
    /// and so has no TLB to invalidate.
    fn flush_tlb(&mut self) {}

    /// The vcpu may next be used — run, or its registers read — on another
    /// host thread: bring all of its state into host memory. A hardware backend
    /// that reads registers back lazily after an exit (HVF) needs this before
    /// the task leaves the thread it ran on; the scheduler calls it as each
    /// slice ends. A no-op for the others.
    fn detach(&mut self) {}

    /// A handle another thread can use to cut this vcpu's current run short
    /// (a signal was posted to its task, which runs on another SMP worker), so
    /// the signal is delivered now rather than at the end of the quantum.
    /// `None` when the backend can't: its runs end on their own soon enough
    /// (the interpreter) or at the next quantum.
    fn kicker(&self) -> Option<Kick> {
        None
    }
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

// Always built on macOS/arm64; running it needs a binary codesigned with the
// hypervisor entitlement, which `select` probes for at runtime.
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
/// `cargo test`) working — `hvf::HvfBackend::new` fails there, and we drop to
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
