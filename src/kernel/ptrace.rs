//! `ptrace(2)` between guest tasks, as `strace` and the like use it: `TRACEME`,
//! `ATTACH`/`SEIZE`, syscall-entry/exit stops, signal-delivery stops, fork/
//! clone/exec/exit events, `INTERRUPT`, register and memory access, and
//! seccomp's `RET_TRACE`.
//!
//! # How a stop works here
//! A tracee stops *itself*, on its own worker, while it services an exit: it
//! snapshots its registers into [`PtraceState`] (in the ptrace user layout),
//! becomes [`RunState::Traced`] (which keeps it off every run queue), and tells
//! its tracer. The tracer only ever reads and writes that snapshot, under the
//! `sh` lock, and never touches another task's vcpu. When the tracer resumes
//! it, the tracee takes the changes back into its own vcpu.
//!
//! Most stops happen in the middle of a syscall — before it runs (entry,
//! seccomp, an `INTERRUPT`) or after its result is known but before it is
//! written back (exit, an event, a signal). The vcpu is then still "in" the
//! syscall, and resuming the task simply re-traps the same syscall (exactly as
//! a blocked one does); [`PtraceState`] remembers how far the call got, so the
//! re-trap picks up there: it runs the call with the number the tracer left,
//! or skips it, or writes back the result the tracer left. A stop with the
//! vcpu back in user mode (after `execve`, at a preemption, a new child's
//! first stop) just resumes the user code.

use super::{Kernel, ProcInfo, QueuedSig, RunState, ServiceCtx, Serviced, Shared, err};
use crate::abi::Arch;
use crate::abi::errno::Errno;
use crate::vcpu::{GuestMemory, Vcpu};

// Requests (`linux/ptrace.h`, plus the x86-64 register ones).
const PTRACE_TRACEME: u64 = 0;
const PTRACE_PEEKTEXT: u64 = 1;
const PTRACE_PEEKDATA: u64 = 2;
const PTRACE_PEEKUSER: u64 = 3;
const PTRACE_POKETEXT: u64 = 4;
const PTRACE_POKEDATA: u64 = 5;
const PTRACE_POKEUSER: u64 = 6;
const PTRACE_CONT: u64 = 7;
const PTRACE_KILL: u64 = 8;
const PTRACE_SINGLESTEP: u64 = 9;
const PTRACE_GETREGS: u64 = 12;
const PTRACE_SETREGS: u64 = 13;
const PTRACE_GETFPREGS: u64 = 14;
const PTRACE_SETFPREGS: u64 = 15;
const PTRACE_GETFPXREGS: u64 = 18;
const PTRACE_SETFPXREGS: u64 = 19;
const PTRACE_GET_THREAD_AREA: u64 = 25;
const PTRACE_SET_THREAD_AREA: u64 = 26;
const PTRACE_ATTACH: u64 = 16;
const PTRACE_DETACH: u64 = 17;
const PTRACE_SYSCALL: u64 = 24;
const PTRACE_SETOPTIONS: u64 = 0x4200;
const PTRACE_GETEVENTMSG: u64 = 0x4201;
const PTRACE_GETSIGINFO: u64 = 0x4202;
const PTRACE_SETSIGINFO: u64 = 0x4203;
const PTRACE_GETREGSET: u64 = 0x4204;
const PTRACE_SETREGSET: u64 = 0x4205;
const PTRACE_SEIZE: u64 = 0x4206;
const PTRACE_INTERRUPT: u64 = 0x4207;
const PTRACE_GETSIGMASK: u64 = 0x420a;
const PTRACE_SETSIGMASK: u64 = 0x420b;
const PTRACE_GET_SYSCALL_INFO: u64 = 0x420e;

// Options.
const O_TRACESYSGOOD: u32 = 0x1;
const O_TRACEEXEC: u32 = 0x10;
const O_TRACEEXIT: u32 = 0x40;
const O_TRACESECCOMP: u32 = 0x80;
const O_EXITKILL: u32 = 0x10_0000;
const O_SUSPEND_SECCOMP: u32 = 0x20_0000;
const O_MASK: u32 = 0xff | O_EXITKILL | O_SUSPEND_SECCOMP;

// Events (the status word's bits 16 and up).
pub(super) const EVENT_FORK: u32 = 1;
pub(super) const EVENT_VFORK: u32 = 2;
pub(super) const EVENT_CLONE: u32 = 3;
const EVENT_EXEC: u32 = 4;
const EVENT_EXIT: u32 = 6;
const EVENT_SECCOMP: u32 = 7;
const EVENT_STOP: u32 = 128;

// Regsets.
const NT_PRSTATUS: u64 = 1;
const NT_PRFPREG: u64 = 2;
/// The i386 `fxsave` image (`user32_fxsr_struct`).
const NT_PRXFPREG: u64 = 0x46e6_2b7f;
/// The i386 GDT TLS entries, as three `struct user_desc`s.
const NT_386_TLS: u64 = 0x200;
const NT_X86_XSTATE: u64 = 0x202;
const NT_ARM_SYSTEM_CALL: u64 = 0x404;

const SIGKILL: u64 = 9;
const SIGTRAP: u32 = 5;
const SIGSTOP: u32 = 19;
const SIGCHLD: u64 = 17;
/// `si_code` of the `SIGCHLD` a tracer gets for a stop.
const CLD_TRAPPED: i32 = 4;
/// What a syscall-entry stop shows in x86-64's `rax`.
const ENOSYS: u64 = (-38i64) as u64;

/// Registers kept per stop: x86-64's `user_regs_struct` (27 words) or arm64's
/// `user_pt_regs` (31 + sp + pc + pstate).
const NREGS: usize = 34;
/// `user_regs_struct` slot ↔ vcpu register index (`rax`=0, `rcx`=1, …),
/// for the 15 general registers besides `rsp`.
const X86_SLOTS: [(usize, usize); 15] = [
    (0, 15),
    (1, 14),
    (2, 13),
    (3, 12),
    (4, 5),
    (5, 3),
    (6, 11),
    (7, 10),
    (8, 9),
    (9, 8),
    (10, 0),
    (11, 1),
    (12, 2),
    (13, 6),
    (14, 7),
];
const X86_RAX: usize = 10;
const X86_RCX: usize = 11;
const X86_R11: usize = 6;
const X86_ORIG_RAX: usize = 15;
const X86_RIP: usize = 16;
const X86_EFLAGS: usize = 18;
const X86_RSP: usize = 19;
/// `sizeof(struct user_regs_struct)`.
const X86_REGS_SIZE: usize = 27 * 8;
const X86_CS: usize = 17;
const X86_SS: usize = 20;
/// `ds`, `es`, `fs`, `gs`, and the vcpu's segment numbers for them.
const X86_DATA_SEGS: [(usize, usize); 4] = [(23, 3), (24, 0), (25, 4), (26, 5)];
/// `user_regs_struct32` word ↔ `user_regs_struct` slot: `ebx`, `ecx`,
/// `edx`, `esi`, `edi`, `ebp`, `eax`, `ds`, `es`, `fs`, `gs`, `orig_eax`,
/// `eip`, `cs`, `eflags`, `esp`, `ss`.
const X86_32_SLOTS: [usize; 17] = [
    5, 11, 12, 13, 14, 4, 10, 23, 24, 25, 26, 15, 16, 17, 18, 19, 20,
];
/// `sizeof(struct user_regs_struct32)`.
const X86_32_REGS_SIZE: usize = 17 * 4;
/// `sizeof(struct user32)`, the `PEEKUSR` area of a 32-bit tracer.
const USER32_SIZE: u64 = 284;
/// `sizeof(struct user_i387_ia32_struct)`, the `fsave` image.
const I387_IA32_SIZE: usize = 108;
const AUDIT_ARCH_I386: u32 = 0x4000_0003;

/// What kind of stop a tracee is in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum StopKind {
    /// Signal-delivery stop: the signal waits on the tracer's verdict.
    Signal,
    /// Syscall-entry stop: the call hasn't run; the tracer may change it.
    Entry,
    /// Syscall-exit stop: the result may still be changed.
    Exit,
    /// `PTRACE_EVENT_SECCOMP`: a `RET_TRACE` filter handed the call over.
    Seccomp,
    /// Any other `PTRACE_EVENT_*` stop.
    Event,
}

/// Where the vcpu is while stopped, which decides what resuming means.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum Phase {
    /// Inside a syscall that hasn't run (or has to run again).
    Before,
    /// Inside a syscall whose result is known but not yet written back.
    Done,
    /// In user mode.
    User,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Stop {
    kind: StopKind,
    sig: u32,
    event: u32,
    phase: Phase,
    /// Its tracer's `wait` has reported it.
    reported: bool,
}

impl Stop {
    /// The `wait` status word: `WIFSTOPPED`, the signal, the event above it.
    fn status(self) -> u32 {
        (self.event << 16) | (self.sig << 8) | 0x7f
    }
}

/// A task's ptrace state: who traces it, and the stop it's in. Boxed into
/// [`ProcInfo::ptrace`] only for a task that is (or was just) traced, so the
/// common untraced task pays one null pointer.
#[derive(Clone)]
#[allow(clippy::struct_excessive_bools)] // independent latches, not a state enum
pub(super) struct PtraceState {
    /// The tracing task, 0 once detached.
    pub(super) tracer: i32,
    options: u32,
    /// Attached by `PTRACE_SEIZE` (new-style stops, `INTERRUPT`).
    seized: bool,
    /// Resumed by `PTRACE_SYSCALL`: stop at syscall entry and exit.
    syscall_trace: bool,
    stop: Option<Stop>,
    /// A stop to enter as soon as the current one is resumed (an exec event
    /// is followed by its syscall-exit stop).
    queued: Option<Stop>,
    /// The registers at the stop, as the tracer sees and edits them.
    regs: [u64; NREGS],
    /// arm64's syscall number (`NT_ARM_SYSTEM_CALL`); x86-64 keeps it in
    /// `orig_rax`.
    a64_nr: u64,
    regs_dirty: bool,
    /// Take the tracer's register changes at the next re-trap (in a syscall).
    apply_gprs: bool,
    /// Take them before running user code again.
    apply_user: bool,
    /// x86: the `rip`, `eflags`, `rcx` and `r11` the tracer left at a stop
    /// inside a syscall, which the call returns with (Linux returns through
    /// `iret` to whatever `pt_regs` hold then, not where `syscall` came
    /// from).
    ret_regs: Option<[u64; 4]>,
    event_msg: u64,
    /// The signal of a signal-delivery stop, as `PTRACE_GETSIGINFO` shows it.
    siginfo: [u8; 128],
    /// The FP/SIMD registers at the stop, in the vcpu's own layout (see
    /// [`Vcpu::fp_state`]).
    fp: Vec<u8>,
    /// The tracer changed them: put them back before user code runs again.
    apply_fp: bool,
    /// x86's three GDT TLS descriptors at the stop
    /// (`PTRACE_GET_THREAD_AREA`).
    tls: [u64; 3],
    /// The tracer changed them: put them back before the task runs again.
    apply_tls: bool,
    /// The stop is in a call through the i386 entry (Linux's `TS_COMPAT`):
    /// `orig_rax` is its i386 number, its arguments `%ebx`..`%ebp`.
    compat: bool,
    /// Resumed by `PTRACE_SINGLESTEP` from a user-mode stop: arm a step
    /// before user code runs again.
    step: bool,
    /// Resumed by `PTRACE_SINGLESTEP` from a signal or event stop inside a
    /// syscall: arm the step once the syscall has returned.
    step_at_return: bool,
    /// `PTRACE_INTERRUPT` asked for a stop.
    interrupt: bool,
    /// A signal the tracer passed on, delivered without stopping again.
    pub(super) inject: Option<(u64, QueuedSig)>,
    // ---- how far the current syscall got ----
    /// Past its entry stop (or the point where there would be one).
    sc_entered: bool,
    /// The syscall number the tracer left at entry (`-1`: skip the call).
    sc_nr: Option<u64>,
    /// What a skipped call returns.
    skip_ret: u64,
    /// The call ran; this is its result, not yet written back.
    sc_result: Option<i64>,
    /// Its seccomp filters already ran (and stopped for the tracer).
    sc_seccomp_done: bool,
    /// Its syscall-exit stop (if any) is behind it.
    sc_exit_done: bool,
    /// `PTRACE_EVENT_EXIT` was reported for this exit call.
    exit_event_done: bool,
    /// A fork/clone event to report once the call is done: `(event, child)`.
    pub(super) pending_event: Option<(u32, u64)>,
    /// A seccomp `RET_TRACE` with this data wants a stop.
    seccomp_trace: Option<u32>,
    /// `execve` succeeded; the thread id it had before.
    pub(super) execed: Option<i32>,
    /// As a zombie: the tracer has been told of the exit.
    exit_reported: bool,
}

impl PtraceState {
    fn new(tracer: i32) -> Self {
        Self {
            tracer,
            options: 0,
            seized: false,
            syscall_trace: false,
            stop: None,
            queued: None,
            regs: [0; NREGS],
            a64_nr: 0,
            regs_dirty: false,
            apply_gprs: false,
            apply_user: false,
            ret_regs: None,
            event_msg: 0,
            siginfo: [0; 128],
            fp: Vec::new(),
            apply_fp: false,
            tls: [0; 3],
            apply_tls: false,
            compat: false,
            step: false,
            step_at_return: false,
            interrupt: false,
            inject: None,
            sc_entered: false,
            sc_nr: None,
            skip_ret: ENOSYS,
            sc_result: None,
            sc_seccomp_done: false,
            sc_exit_done: false,
            exit_event_done: false,
            pending_event: None,
            seccomp_trace: None,
            execed: None,
            exit_reported: false,
        }
    }

    /// What the tracer side owns: changed by the tracer while the task may be
    /// out running on another CPU (see `smp_service_step`'s fold).
    pub(super) fn link(&self) -> (i32, u32, bool, bool) {
        (self.tracer, self.options, self.seized, self.interrupt)
    }

    pub(super) fn set_link(&mut self, link: (i32, u32, bool, bool)) {
        (self.tracer, self.options, self.seized, self.interrupt) = link;
    }

    /// Before the task runs user code again: what its tracer changed while
    /// it was stopped there (registers, FP state, TLS entries) goes back
    /// into its vcpu, and a single step is armed if one was asked for.
    pub(super) fn resume_user(&mut self, arch: Arch, vcpu: &mut dyn Vcpu) {
        if std::mem::take(&mut self.apply_user) {
            apply_user(arch, vcpu, &self.regs);
        }
        if std::mem::take(&mut self.apply_fp) {
            vcpu.set_fp_state(&self.fp);
        }
        self.put_tls(vcpu);
        if std::mem::take(&mut self.step) {
            vcpu.set_single_step(true);
        }
    }

    /// The TLS entries the tracer set (`PTRACE_SET_THREAD_AREA`), into the
    /// vcpu (reloading a segment register that holds one).
    fn put_tls(&mut self, vcpu: &mut dyn Vcpu) {
        if std::mem::take(&mut self.apply_tls) {
            for (i, &d) in self.tls.iter().enumerate() {
                if vcpu.tls_desc(i) != d {
                    vcpu.set_tls_desc(i, d);
                }
            }
        }
    }

    /// The syscall's result is in the vcpu: return where the tracer said,
    /// with the flags, `rcx` and `r11` it left, if it changed them at a stop
    /// in the call.
    pub(super) fn return_to(&mut self, vcpu: &mut dyn Vcpu) {
        if let Some([rip, flags, rcx, r11]) = self.ret_regs.take() {
            vcpu.settle_syscall_return();
            vcpu.set_reg(1, rcx);
            vcpu.set_reg(11, r11);
            vcpu.set_rflags(flags);
            vcpu.set_pc(rip);
        }
    }

    /// Whether to arm a single step now that the syscall has returned.
    pub(super) fn take_step_at_return(&mut self) -> bool {
        std::mem::take(&mut self.step_at_return)
    }

    /// The current syscall is over: forget how far it got.
    fn end_syscall(&mut self) {
        self.sc_entered = false;
        self.sc_nr = None;
        self.skip_ret = ENOSYS;
        self.sc_result = None;
        self.sc_seccomp_done = false;
        self.sc_exit_done = false;
        self.exit_event_done = false;
        self.pending_event = None;
        self.seccomp_trace = None;
    }

    /// Nothing left to do: not traced, nothing in progress.
    fn idle(&self) -> bool {
        self.tracer == 0
            && self.stop.is_none()
            && self.inject.is_none()
            && !self.apply_user
            && self.ret_regs.is_none()
            && !self.apply_fp
            && !self.apply_tls
            && !self.step
            && !self.step_at_return
            && !self.apply_gprs
            && !self.sc_entered
            && self.sc_result.is_none()
    }

    fn sys_nr(&self, arch: Arch) -> u64 {
        match arch {
            Arch::X86_64 => self.regs[X86_ORIG_RAX],
            Arch::Aarch64 => self.a64_nr,
        }
    }

    fn ret_reg(&self, arch: Arch) -> u64 {
        match arch {
            Arch::X86_64 => self.regs[X86_RAX],
            Arch::Aarch64 => self.regs[0],
        }
    }

    fn sig_status(&self, sig: u32) -> u32 {
        if self.options & O_TRACESYSGOOD != 0 {
            sig | 0x80
        } else {
            sig
        }
    }
}

impl ProcInfo {
    /// The task tracing this one, or 0.
    pub(super) fn tracer(&self) -> i32 {
        self.ptrace.as_ref().map_or(0, |t| t.tracer)
    }

    /// A zombie whose tracer hasn't yet been told it exited: nobody else may
    /// reap it until then. (A tracer that is also the parent of a process
    /// just reaps it, as a parent.)
    pub(super) fn exit_awaits_tracer(&self) -> bool {
        self.ptrace.as_ref().is_some_and(|t| {
            t.tracer != 0 && !t.exit_reported && (t.tracer != self.ppid || self.is_thread)
        })
    }

    /// Take a tracee out of a stop to die: `SIGKILL` reaches it even there.
    pub(super) fn wake_traced_for_kill(&mut self) {
        if self.run == RunState::Traced {
            self.run = RunState::Running;
            self.parked = false;
            if let Some(t) = self.ptrace.as_mut() {
                t.stop = None;
                t.queued = None;
            }
        }
    }
}

/// The registers of a stopped task, in the arch's ptrace layout.
fn snapshot(arch: Arch, vcpu: &mut dyn Vcpu, phase: Phase, fs_base: u64) -> [u64; NREGS] {
    let mut r = [0; NREGS];
    if phase == Phase::User {
        // Collapse a pending syscall return (KVM's trampoline) into the
        // user-mode state it would return to.
        vcpu.settle_syscall_return();
    }
    match arch {
        Arch::X86_64 => {
            for (slot, gpr) in X86_SLOTS {
                r[slot] = vcpu.reg(gpr);
            }
            r[X86_RSP] = vcpu.sp();
            if phase == Phase::User {
                r[X86_RIP] = vcpu.pc();
                r[X86_EFLAGS] = vcpu.rflags();
                r[X86_ORIG_RAX] = u64::MAX;
            } else if vcpu.syscall_ia32() {
                // `int $0x80` keeps the user's flags, and returns where
                // `syscall_ip` says; `orig_rax` is the i386 number.
                r[X86_RIP] = vcpu.syscall_ip();
                r[X86_EFLAGS] = vcpu.rflags();
                r[X86_ORIG_RAX] = vcpu.syscall_nr();
            } else {
                // In a syscall the user rip and flags are where `syscall` put
                // them (and where `sysret` takes them from).
                r[X86_RIP] = vcpu.reg(1);
                r[X86_EFLAGS] = vcpu.reg(11);
                r[X86_ORIG_RAX] = vcpu.syscall_nr();
            }
            let (cs, ss) = vcpu.user_segments();
            r[X86_CS] = u64::from(cs);
            r[X86_SS] = u64::from(ss);
            r[21] = fs_base;
            r[22] = vcpu.gs_base();
            for (slot, seg) in X86_DATA_SEGS {
                r[slot] = u64::from(vcpu.segment_selector(seg));
            }
        }
        Arch::Aarch64 => {
            for (i, slot) in r.iter_mut().enumerate().take(31) {
                *slot = vcpu.reg(i);
            }
            r[31] = vcpu.sp();
            // In a syscall the pc is still on the `svc`; Linux shows the
            // address after it.
            r[32] = vcpu.pc() + if phase == Phase::User { 0 } else { 4 };
            r[33] = vcpu.rflags();
        }
    }
    r
}

/// Put the tracer's register changes back into a vcpu that is in a syscall:
/// the general registers (the syscall number and result are carried
/// separately, and after x86-64's `syscall` `rcx`/`r11` hold the return rip
/// and flags; `int $0x80` leaves them the user's).
fn apply_in_syscall(arch: Arch, vcpu: &mut dyn Vcpu, r: &[u64; NREGS]) {
    match arch {
        Arch::X86_64 => {
            let ia32 = vcpu.syscall_ia32();
            for (slot, gpr) in X86_SLOTS {
                if slot != X86_RAX && (ia32 || !matches!(slot, X86_RCX | X86_R11)) {
                    vcpu.set_reg(gpr, r[slot]);
                }
            }
            vcpu.set_sp(r[X86_RSP]);
            put_data_segments(vcpu, r);
        }
        Arch::Aarch64 => {
            for (i, &v) in r.iter().enumerate().take(31) {
                vcpu.set_reg(i, v);
            }
            vcpu.set_sp(r[31]);
        }
    }
}

/// The data segment registers the tracer changed (`set_segment_reg`); one
/// the task can't load is left as it was.
fn put_data_segments(vcpu: &mut dyn Vcpu, r: &[u64; NREGS]) {
    for (slot, seg) in X86_DATA_SEGS {
        let sel = r[slot] as u16;
        if sel != vcpu.segment_selector(seg) {
            vcpu.load_data_segment(seg, sel);
        }
    }
}

/// Put them back into a vcpu in user mode: everything.
fn apply_user(arch: Arch, vcpu: &mut dyn Vcpu, r: &[u64; NREGS]) {
    match arch {
        Arch::X86_64 => {
            for (slot, gpr) in X86_SLOTS {
                vcpu.set_reg(gpr, r[slot]);
            }
            vcpu.set_sp(r[X86_RSP]);
            vcpu.set_rflags(r[X86_EFLAGS]);
            // A code or stack segment the tracer changed (to 32-bit code,
            // say); a pair the task can't use is left as it was.
            let (cs, ss) = (r[X86_CS] as u16, r[X86_SS] as u16);
            if (cs, ss) != vcpu.user_segments() && cs != 0 && ss != 0 {
                vcpu.set_user_segments(cs, ss);
            }
            put_data_segments(vcpu, r);
            vcpu.set_pc(r[X86_RIP]);
        }
        Arch::Aarch64 => {
            for (i, &v) in r.iter().enumerate().take(31) {
                vcpu.set_reg(i, v);
            }
            vcpu.set_sp(r[31]);
            vcpu.set_rflags(r[33]);
            vcpu.set_pc(r[32]);
        }
    }
}

/// FP state (the vcpu's layout) as the ptrace user layout: x86-64's
/// `user_fpregs_struct` (the 512-byte `FXSAVE` image that opens the `XSAVE`
/// area); arm64's `user_fpsimd_state` (`V0..V31`, then `FPSR`, `FPCR`, and
/// padding: 528 bytes) from the vcpu's `FPSR`, `FPCR`, `V0..V31`.
fn fp_user(arch: Arch, fp: &[u8]) -> Vec<u8> {
    match arch {
        Arch::X86_64 => {
            let mut out = vec![0u8; 512];
            let n = fp.len().min(512);
            out[..n].copy_from_slice(&fp[..n]);
            if n < 512 {
                // Nothing captured: the state a fresh process has.
                out[0..2].copy_from_slice(&0x37fu16.to_le_bytes()); // fcw
                out[24..28].copy_from_slice(&0x1f80u32.to_le_bytes()); // mxcsr
            }
            out
        }
        Arch::Aarch64 => {
            let mut out = vec![0u8; 528];
            if fp.len() >= 8 + 512 {
                out[..512].copy_from_slice(&fp[8..520]);
                out[512..520].copy_from_slice(&fp[..8]);
            }
            out
        }
    }
}

impl PtraceState {
    /// Take the tracer's FP registers (ptrace user layout, possibly short)
    /// into the captured state, to be put back before user code runs.
    fn set_fp_user(&mut self, arch: Arch, user: &[u8]) {
        let mut cur = fp_user(arch, &self.fp);
        let n = user.len().min(cur.len());
        cur[..n].copy_from_slice(&user[..n]);
        match arch {
            Arch::X86_64 => {
                // Keep the extended state the stop captured; replace the
                // legacy region, which is in use now (`XFEATURE_MASK_FPSSE`
                // in the header).
                if self.fp.len() >= 520 {
                    self.fp[..512].copy_from_slice(&cur);
                    self.fp[512] |= 3;
                } else {
                    self.fp = cur;
                }
            }
            Arch::Aarch64 => {
                let mut fp = Vec::with_capacity(520);
                fp.extend_from_slice(&cur[512..520]);
                fp.extend_from_slice(&cur[..512]);
                self.fp = fp;
            }
        }
        self.apply_fp = true;
    }

    /// `getreg32`: the word at `off` in a 32-bit tracer's `struct user32`
    /// (its registers, from the 64-bit ones; 0 past them).
    fn getreg32(&self, off: u64) -> u32 {
        let Some(&slot) = X86_32_SLOTS.get(off as usize / 4) else {
            return 0;
        };
        let v = self.regs[slot];
        if is_segment(slot) {
            u32::from(v as u16)
        } else {
            v as u32
        }
    }

    /// `putreg32`: a 32-bit tracer's write of the word at `off` (zero-
    /// extended into the 64-bit register; a selector must be a user one,
    /// and `%cs` and `%ss` not null). Past the registers it is ignored.
    fn putreg32(&mut self, off: u64, v: u32) -> Result<(), i64> {
        let Some(&slot) = X86_32_SLOTS.get(off as usize / 4) else {
            return Ok(());
        };
        if is_segment(slot) {
            let sel = v as u16;
            if (sel != 0 && sel & 3 != 3) || (sel == 0 && matches!(slot, X86_CS | X86_SS)) {
                return Err(err(Errno::EIO));
            }
            self.regs[slot] = u64::from(sel);
        } else {
            self.regs[slot] = u64::from(v);
        }
        self.regs_dirty = true;
        Ok(())
    }

    /// `struct user_regs_struct32`.
    fn regs32(&self) -> Vec<u8> {
        (0..X86_32_REGS_SIZE as u64)
            .step_by(4)
            .flat_map(|off| self.getreg32(off).to_le_bytes())
            .collect()
    }

    /// Write (a prefix of) `struct user_regs_struct32`, word by word.
    fn set_regs32(&mut self, b: &[u8]) -> Result<(), i64> {
        for (i, w) in b.as_chunks::<4>().0.iter().enumerate() {
            self.putreg32(i as u64 * 4, u32::from_le_bytes(*w))?;
        }
        Ok(())
    }

    /// `struct user_i387_ia32_struct`, the `fsave` image a 32-bit tracer
    /// sees (`convert_from_fxsr`).
    fn fsave(&self) -> Vec<u8> {
        let fx = fp_user(Arch::X86_64, &self.fp);
        let (cs, ds) = (self.regs[X86_CS] as u16, self.regs[23] as u16);
        super::signal::fsave_header(&fx, cs, ds)[..I387_IA32_SIZE].to_vec()
    }

    /// Write (a prefix of) the `fsave` image (`convert_to_fxsr`): the x87
    /// state changes, the SSE state stays.
    fn set_fsave(&mut self, user: &[u8]) {
        let mut env = self.fsave();
        let n = user.len().min(env.len());
        env[..n].copy_from_slice(&user[..n]);
        let word = |o: usize| u32::from_le_bytes(env[o..o + 4].try_into().unwrap());
        let mut fx = fp_user(Arch::X86_64, &self.fp);
        fx[0..2].copy_from_slice(&env[0..2]); // cwd
        fx[2..4].copy_from_slice(&env[4..6]); // swd
        fx[4..6].copy_from_slice(&twd_i387_to_fxsr(word(8) as u16).to_le_bytes());
        fx[6..8].copy_from_slice(&((word(16) >> 16) as u16).to_le_bytes()); // fop
        fx[8..16].copy_from_slice(&u64::from(word(12)).to_le_bytes()); // rip
        fx[16..24].copy_from_slice(&u64::from(word(20)).to_le_bytes()); // rdp
        for i in 0..8 {
            fx[32 + i * 16..42 + i * 16].copy_from_slice(&env[28 + i * 10..38 + i * 10]);
        }
        self.set_fp_user(Arch::X86_64, &fx);
    }

    /// `NT_X86_XSTATE`: the `XSAVE` area in its standard form, with the
    /// `XCR0` it was saved under in the software-reserved bytes
    /// (`xstate_fx_sw_bytes`).
    fn xstate(&self) -> Vec<u8> {
        let size = crate::vcpu::X86_XSAVE_SIZE;
        let mut x = vec![0u8; size];
        if self.fp.len() >= size {
            x.copy_from_slice(&self.fp[..size]);
        } else {
            x[..512].copy_from_slice(&fp_user(Arch::X86_64, &self.fp));
        }
        x[464..512].fill(0);
        x[464..472].copy_from_slice(&XCR0.to_le_bytes());
        x
    }

    /// Replace the `XSAVE` area (`copy_uabi_from_kernel_to_xstate`): all of
    /// it at once (`EFAULT` otherwise), with no feature beyond `XCR0`, a
    /// standard-form header and a valid `MXCSR` (`EINVAL` otherwise).
    fn set_xstate(&mut self, x: &[u8]) -> Result<(), i64> {
        if x.len() != crate::vcpu::X86_XSAVE_SIZE {
            return Err(err(Errno::EFAULT));
        }
        let q = |o: usize| u64::from_le_bytes(x[o..o + 8].try_into().unwrap());
        let mxcsr = u32::from_le_bytes(x[24..28].try_into().unwrap());
        if q(512) & !XCR0 != 0 || q(520) != 0 || (528..576).step_by(8).any(|o| q(o) != 0) {
            return Err(err(Errno::EINVAL));
        }
        if mxcsr & !0xffff != 0 {
            return Err(err(Errno::EINVAL));
        }
        self.fp = x.to_vec();
        self.apply_fp = true;
        Ok(())
    }
}

/// The `XSAVE` features the vcpus have (x87, SSE, AVX), as `XCR0`.
const XCR0: u64 = 7;

/// A `user_regs_struct` slot holding a segment selector.
const fn is_segment(slot: usize) -> bool {
    matches!(slot, X86_CS | X86_SS | 23..=26)
}

/// The abridged `fxsave` tag word (a bit a register: valid) from the
/// `i387` one (`twd_i387_to_fxsr`).
fn twd_i387_to_fxsr(twd: u16) -> u16 {
    let mut tmp = u32::from(!twd);
    tmp = (tmp | (tmp >> 1)) & 0x5555;
    tmp = (tmp | (tmp >> 1)) & 0x3333;
    tmp = (tmp | (tmp >> 2)) & 0x0f0f;
    tmp = (tmp | (tmp >> 4)) & 0x00ff;
    tmp as u16
}

/// A signal's `siginfo_t` as `PTRACE_GETSIGINFO` returns it.
fn siginfo_bytes(sig: u64, q: Option<QueuedSig>) -> [u8; 128] {
    let mut si = [0u8; 128];
    si[0..4].copy_from_slice(&(sig as i32).to_le_bytes());
    if let Some(q) = q {
        si[4..8].copy_from_slice(&q.errno.to_le_bytes());
        si[8..12].copy_from_slice(&q.code.to_le_bytes());
        si[16..20].copy_from_slice(&q.pid.to_le_bytes());
        si[20..24].copy_from_slice(&q.uid.to_le_bytes());
        si[24..32].copy_from_slice(&q.value.to_le_bytes());
    }
    si
}

fn siginfo_queued(si: &[u8; 128]) -> QueuedSig {
    let word = |o: usize| u32::from_le_bytes(si[o..o + 4].try_into().unwrap());
    QueuedSig {
        code: word(8) as i32,
        errno: word(4) as i32,
        pid: word(16) as i32,
        uid: word(20),
        value: u64::from_le_bytes(si[24..32].try_into().unwrap()),
    }
}

/// How the tracer lets a stopped tracee go.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Resume {
    Cont,
    Syscall,
    Step,
    Detach,
}

/// A tracee event for its tracer's `wait`.
pub(super) enum TraceReport {
    /// A ptrace stop: the task and its status word.
    Stop(i32, u32),
    /// A traced task (not the tracer's child) exited.
    Exit(i32, super::ExitCause),
}

impl Kernel {
    /// `ptrace(request, pid, addr, data)`.
    #[allow(clippy::too_many_lines)]
    pub(super) fn sys_ptrace(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        args: &[u64; 6],
        mem: &mut GuestMemory,
    ) -> i64 {
        let (req, pid, addr, data) = (args[0], args[1] as i32, args[2], args[3]);
        match req {
            PTRACE_TRACEME => {
                if cx.cur.tracer() != 0 {
                    return err(Errno::EPERM);
                }
                let parent = cx.cur.ppid;
                if !sh.procs.iter().flatten().any(|p| p.info.pid == parent) {
                    return err(Errno::EPERM);
                }
                let mut t = cx
                    .cur
                    .ptrace
                    .take()
                    .unwrap_or_else(|| Box::new(PtraceState::new(0)));
                t.tracer = parent;
                cx.cur.ptrace = Some(t);
                return 0;
            }
            PTRACE_ATTACH | PTRACE_SEIZE => {
                return self.ptrace_attach(sh, cx, req, pid, addr, data);
            }
            _ => {}
        }
        let me = cx.cur.pid;
        let Some(idx) = sh.procs.iter().position(|s| {
            s.as_ref()
                .is_some_and(|p| p.info.pid == pid && p.info.tracer() == me && pid != me)
        }) else {
            return err(Errno::ESRCH);
        };
        if req == PTRACE_KILL {
            let p = &mut sh.procs[idx].as_mut().unwrap().info;
            if !matches!(p.run, RunState::Zombie(_)) {
                p.raise(SIGKILL);
                p.parked = false;
            }
            return 0;
        }
        if req == PTRACE_INTERRUPT {
            let p = &mut sh.procs[idx].as_mut().unwrap().info;
            let t = p.ptrace.as_mut().unwrap();
            if !t.seized {
                return err(Errno::EIO);
            }
            if p.run != RunState::Traced {
                t.interrupt = true;
                p.parked = false;
                p.kick.kick();
            }
            return 0;
        }
        // Everything else needs the tracee stopped.
        let stopped = sh.procs[idx]
            .as_ref()
            .is_some_and(|p| p.info.run == RunState::Traced && !p.info.in_flight);
        if !stopped {
            return err(Errno::ESRCH);
        }
        let arch = self.arch;
        // Through the i386 entry (`compat_sys_ptrace`): words and the
        // `user` area are 32-bit, and so are `siginfo`, the event message
        // and the regset `iovec`.
        let compat = cx.ia32 && arch == Arch::X86_64;
        if arch == Arch::X86_64
            && let Some(r) = Self::ptrace_x86(sh, idx, req, addr, data, compat, mem)
        {
            return r;
        }
        let word = if compat { 4 } else { 8 };
        match req {
            PTRACE_PEEKTEXT | PTRACE_PEEKDATA => {
                let mut buf = [0u8; 8];
                let w = &mut buf[..word];
                match Self::tracee_mem(sh, cx, idx, mem, |m| m.read(addr, w).is_ok()) {
                    Some(true) => {}
                    Some(false) => return err(Errno::EIO),
                    None => {
                        cx.retry_busy();
                        return 0;
                    }
                }
                if mem.write(data, &buf[..word]).is_err() {
                    return err(Errno::EFAULT);
                }
                0
            }
            PTRACE_POKETEXT | PTRACE_POKEDATA => {
                let bytes = data.to_le_bytes();
                let w = &bytes[..word];
                match Self::tracee_mem(sh, cx, idx, mem, |m| m.write_init(addr, w).is_ok()) {
                    Some(true) => 0,
                    Some(false) => err(Errno::EIO),
                    None => {
                        cx.retry_busy();
                        0
                    }
                }
            }
            PTRACE_CONT | PTRACE_SYSCALL | PTRACE_SINGLESTEP | PTRACE_DETACH => {
                if data > 64 {
                    return err(Errno::EIO);
                }
                let mode = match req {
                    PTRACE_CONT => Resume::Cont,
                    PTRACE_SYSCALL => Resume::Syscall,
                    PTRACE_SINGLESTEP => Resume::Step,
                    _ => Resume::Detach,
                };
                let p = &mut sh.procs[idx].as_mut().unwrap().info;
                Self::ptrace_resume(arch, p, data, mode);
                if mode == Resume::Detach {
                    // Its parent may be waiting for it to be free again.
                    let ppid = p.ppid;
                    for q in sh.procs.iter_mut().flatten() {
                        if q.info.pid == ppid {
                            q.info.parked = false;
                        }
                    }
                }
                0
            }
            PTRACE_SETOPTIONS => {
                let opts = data as u32;
                if data > u64::from(u32::MAX) || opts & !O_MASK != 0 {
                    return err(Errno::EINVAL);
                }
                Self::tracee_state(sh, idx).options = opts;
                0
            }
            PTRACE_GETEVENTMSG => {
                let t = sh.procs[idx]
                    .as_ref()
                    .unwrap()
                    .info
                    .ptrace
                    .as_ref()
                    .unwrap();
                let mut msg = t.event_msg;
                let is_pid = t.stop.is_some_and(|s| {
                    matches!(s.event, EVENT_FORK | EVENT_VFORK | EVENT_CLONE | EVENT_EXEC)
                });
                if is_pid {
                    msg = self.visible_pid(cx, msg as i32) as u64;
                }
                if mem.write(data, &msg.to_le_bytes()[..word]).is_err() {
                    return err(Errno::EFAULT);
                }
                0
            }
            PTRACE_GETSIGINFO => {
                let p = &sh.procs[idx].as_ref().unwrap().info;
                let t = p.ptrace.as_ref().unwrap();
                let stop = t.stop.unwrap();
                let mut si = if stop.kind == StopKind::Signal {
                    t.siginfo
                } else {
                    // A trap: what the kernel's own SIGTRAP carries.
                    let code = (stop.status() >> 8) as i32;
                    let q = QueuedSig {
                        code,
                        errno: 0,
                        pid: p.tgid,
                        uid: p.creds.ruid,
                        value: 0,
                    };
                    siginfo_bytes(u64::from(SIGTRAP), Some(q))
                };
                if compat {
                    si = super::compat::siginfo_narrow(&si);
                }
                if mem.write(data, &si).is_err() {
                    return err(Errno::EFAULT);
                }
                0
            }
            PTRACE_SETSIGINFO => {
                let mut si = [0u8; 128];
                if mem.read(data, &mut si).is_err() {
                    return err(Errno::EFAULT);
                }
                if compat {
                    si = super::compat::siginfo_widen(&si);
                }
                let t = Self::tracee_state(sh, idx);
                if t.stop.is_some_and(|s| s.kind == StopKind::Signal) {
                    t.siginfo = si;
                }
                0
            }
            PTRACE_GETSIGMASK | PTRACE_SETSIGMASK => {
                if addr != 8 {
                    return err(Errno::EINVAL);
                }
                let p = &mut sh.procs[idx].as_mut().unwrap().info;
                if req == PTRACE_GETSIGMASK {
                    if mem.write(data, &p.blocked.to_le_bytes()).is_err() {
                        return err(Errno::EFAULT);
                    }
                } else {
                    let Ok(mask) = mem.read_u64(data) else {
                        return err(Errno::EFAULT);
                    };
                    p.blocked = mask & !((1 << (SIGKILL - 1)) | (1 << (u64::from(SIGSTOP) - 1)));
                }
                0
            }
            PTRACE_GETREGSET | PTRACE_SETREGSET => {
                self.ptrace_regset(sh, idx, req, addr, data, compat, mem)
            }
            PTRACE_GET_SYSCALL_INFO => {
                let info = Self::syscall_info(arch, Self::tracee_state(sh, idx));
                let n = (addr as usize).min(info.len());
                if mem.write(data, &info[..n]).is_err() {
                    return err(Errno::EFAULT);
                }
                info.len() as i64
            }
            _ => err(Errno::EIO),
        }
    }

    /// The x86 register and thread-area requests, for a 64-bit tracer or
    /// (`compat`) a 32-bit one (`ia32_arch_ptrace`); `None` for the others.
    fn ptrace_x86(
        sh: &mut Shared,
        idx: usize,
        req: u64,
        addr: u64,
        data: u64,
        compat: bool,
        mem: &mut GuestMemory,
    ) -> Option<i64> {
        let arch = Arch::X86_64;
        let t = Self::tracee_state(sh, idx);
        let put = |mem: &mut GuestMemory, bytes: &[u8]| {
            if mem.write(data, bytes).is_ok() {
                0
            } else {
                err(Errno::EFAULT)
            }
        };
        Some(match req {
            PTRACE_PEEKUSER if compat => {
                if addr & 3 != 0 || addr > USER32_SIZE {
                    return Some(err(Errno::EIO));
                }
                put(mem, &t.getreg32(addr).to_le_bytes())
            }
            PTRACE_POKEUSER if compat => {
                if addr & 3 != 0 || addr > USER32_SIZE {
                    return Some(err(Errno::EIO));
                }
                match t.putreg32(addr, data as u32) {
                    Ok(()) => 0,
                    Err(e) => e,
                }
            }
            PTRACE_PEEKUSER => {
                if !addr.is_multiple_of(8) || addr >= 1024 {
                    return Some(err(Errno::EIO));
                }
                let v = if (addr as usize) < X86_REGS_SIZE {
                    t.regs[addr as usize / 8]
                } else {
                    0
                };
                put(mem, &v.to_le_bytes())
            }
            PTRACE_POKEUSER => {
                if !addr.is_multiple_of(8) || addr >= 1024 {
                    return Some(err(Errno::EIO));
                }
                if (addr as usize) < X86_REGS_SIZE {
                    t.regs[addr as usize / 8] = data;
                    t.regs_dirty = true;
                }
                0
            }
            PTRACE_GETREGS if compat => put(mem, &t.regs32()),
            PTRACE_SETREGS if compat => {
                let Ok(bytes) = mem.read_vec(data, X86_32_REGS_SIZE) else {
                    return Some(err(Errno::EFAULT));
                };
                match t.set_regs32(&bytes) {
                    Ok(()) => 0,
                    Err(e) => e,
                }
            }
            PTRACE_GETREGS => {
                let bytes: Vec<u8> = t.regs[..27].iter().flat_map(|v| v.to_le_bytes()).collect();
                put(mem, &bytes)
            }
            PTRACE_SETREGS => {
                let Ok(bytes) = mem.read_vec(data, X86_REGS_SIZE) else {
                    return Some(err(Errno::EFAULT));
                };
                for (i, c) in bytes.as_chunks::<8>().0.iter().enumerate() {
                    t.regs[i] = u64::from_le_bytes(*c);
                }
                t.regs_dirty = true;
                0
            }
            // The `fsave` image for a 32-bit tracer, the `fxsave` one for a
            // 64-bit one (and `GETFPXREGS`, which only a 32-bit one has).
            PTRACE_GETFPREGS if compat => put(mem, &t.fsave()),
            PTRACE_SETFPREGS if compat => {
                let Ok(env) = mem.read_vec(data, I387_IA32_SIZE) else {
                    return Some(err(Errno::EFAULT));
                };
                t.set_fsave(&env);
                0
            }
            PTRACE_GETFPREGS => put(mem, &fp_user(arch, &t.fp)),
            PTRACE_GETFPXREGS if compat => put(mem, &fp_user(arch, &t.fp)),
            PTRACE_SETFPREGS => {
                let mut buf = [0u8; 512];
                if mem.read(data, &mut buf).is_err() {
                    return Some(err(Errno::EFAULT));
                }
                t.set_fp_user(arch, &buf);
                0
            }
            PTRACE_SETFPXREGS if compat => {
                let mut buf = [0u8; 512];
                if mem.read(data, &mut buf).is_err() {
                    return Some(err(Errno::EFAULT));
                }
                t.set_fp_user(arch, &buf);
                0
            }
            // `addr` is the GDT entry (12-14), `data` a `struct user_desc`.
            PTRACE_GET_THREAD_AREA | PTRACE_SET_THREAD_AREA => {
                if addr as u32 > i32::MAX as u32 {
                    return Some(err(Errno::EIO));
                }
                let entry = u64::from(addr as u32);
                if req == PTRACE_GET_THREAD_AREA {
                    match super::sys_misc::thread_area_get(&t.tls, entry) {
                        Ok(desc) => put(mem, &desc),
                        Err(e) => e,
                    }
                } else {
                    let mut raw = [0u8; 16];
                    if mem.read(data, &mut raw).is_err() {
                        return Some(err(Errno::EFAULT));
                    }
                    match super::sys_misc::thread_area_set(&mut t.tls, entry, &raw) {
                        Ok(()) => {
                            t.apply_tls = true;
                            0
                        }
                        Err(e) => e,
                    }
                }
            }
            _ => return None,
        })
    }

    fn tracee_state(sh: &mut Shared, idx: usize) -> &mut PtraceState {
        sh.procs[idx]
            .as_mut()
            .unwrap()
            .info
            .ptrace
            .as_mut()
            .unwrap()
    }

    /// Run `f` on tracee `idx`'s memory: `None` when another CPU has it now
    /// (the caller retries). The target's lock is only tried, as
    /// `process_vm_readv` does: memory is the outermost lock and the caller
    /// already holds its own.
    fn tracee_mem<R>(
        sh: &Shared,
        cx: &ServiceCtx,
        idx: usize,
        mem: &mut GuestMemory,
        f: impl FnOnce(&mut GuestMemory) -> R,
    ) -> Option<R> {
        let mm = sh.procs[idx].as_ref()?.info.mm;
        if mm == cx.cur.mm {
            return Some(f(mem));
        }
        let space = sh.spaces.get(mm)?;
        let Ok(mut other) = space.try_lock() else {
            return None;
        };
        Some(f(&mut other))
    }

    /// `PTRACE_ATTACH`/`PTRACE_SEIZE`.
    fn ptrace_attach(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        req: u64,
        pid: i32,
        addr: u64,
        data: u64,
    ) -> i64 {
        if req == PTRACE_SEIZE && (addr != 0 || data & !u64::from(O_MASK) != 0) {
            return if addr != 0 {
                err(Errno::EIO)
            } else {
                err(Errno::EINVAL)
            };
        }
        if pid == cx.cur.pid {
            return err(Errno::EPERM);
        }
        let Some(idx) = sh
            .procs
            .iter()
            .position(|s| s.as_ref().is_some_and(|p| p.info.pid == pid))
        else {
            return err(Errno::ESRCH);
        };
        let target = &sh.procs[idx].as_ref().unwrap().info;
        if target.tgid == cx.cur.tgid
            || matches!(target.run, RunState::Zombie(_))
            || target.tracer() != 0
            || !self.may_ptrace(cx, target)
        {
            return err(Errno::EPERM);
        }
        let me = cx.cur.pid;
        let p = &mut sh.procs[idx].as_mut().unwrap().info;
        let mut t = p
            .ptrace
            .take()
            .unwrap_or_else(|| Box::new(PtraceState::new(0)));
        t.tracer = me;
        t.exit_reported = false;
        if req == PTRACE_SEIZE {
            t.seized = true;
            t.options = data as u32;
        }
        p.ptrace = Some(t);
        if req == PTRACE_ATTACH {
            let qs = QueuedSig {
                code: 0,
                errno: 0,
                pid: cx.cur.tgid,
                uid: cx.cur.creds.ruid,
                value: 0,
            };
            p.raise_with(u64::from(SIGSTOP), qs);
            p.parked = false;
        }
        0
    }

    /// Linux's `__ptrace_may_access` for an attach: the same user (real ids
    /// against the target's real, effective and saved ones) and a dumpable
    /// target, or `CAP_SYS_PTRACE` over the target's user namespace.
    pub(super) fn may_ptrace(&self, cx: &ServiceCtx, target: &ProcInfo) -> bool {
        const CAP_SYS_PTRACE: u32 = 19;
        let (c, t) = (&cx.cur.creds, &target.creds);
        let same_user = [t.euid, t.suid, t.ruid].iter().all(|&u| u == c.ruid)
            && [t.egid, t.sgid, t.rgid].iter().all(|&g| g == c.rgid);
        let capable = self.ns_capable(cx, target.ns.user, CAP_SYS_PTRACE);
        capable || (same_user && target.dumpable == 1)
    }

    /// Let tracee `p` go from its stop, passing `sig` on (0: none).
    fn ptrace_resume(arch: Arch, p: &mut ProcInfo, sig: u64, mode: Resume) {
        let Some(t) = p.ptrace.as_mut() else {
            return;
        };
        let Some(stop) = t.stop.take() else {
            return;
        };
        t.syscall_trace = mode == Resume::Syscall;
        // A step from user mode is one instruction; from a stop inside a
        // syscall it ends as the syscall returns (Linux reports a step over
        // a syscall at its exit).
        let step_in_syscall = mode == Resume::Step
            && stop.phase != Phase::User
            && matches!(
                stop.kind,
                StopKind::Entry | StopKind::Exit | StopKind::Seccomp
            );
        t.step = mode == Resume::Step && stop.phase == Phase::User;
        t.step_at_return = mode == Resume::Step && stop.phase != Phase::User && !step_in_syscall;
        match stop.phase {
            Phase::Before => {
                if matches!(stop.kind, StopKind::Entry | StopKind::Seccomp) {
                    t.sc_nr = Some(t.sys_nr(arch));
                    t.skip_ret = t.ret_reg(arch);
                }
                t.apply_gprs = t.regs_dirty;
            }
            Phase::Done => {
                t.sc_result = Some(t.ret_reg(arch) as i64);
                t.apply_gprs = t.regs_dirty;
            }
            Phase::User => t.apply_user = t.regs_dirty,
        }
        if stop.phase != Phase::User && t.regs_dirty && arch == Arch::X86_64 {
            let r = &t.regs;
            t.ret_regs = Some([r[X86_RIP], r[X86_EFLAGS], r[X86_RCX], r[X86_R11]]);
        }
        t.regs_dirty = false;
        let mut raise = None;
        if sig != 0 {
            match stop.kind {
                // The signal it stopped for, or the one the tracer swapped in.
                StopKind::Signal => {
                    let mut qs = siginfo_queued(&t.siginfo);
                    let signo = u64::from(u32::from_le_bytes(t.siginfo[0..4].try_into().unwrap()));
                    if signo != sig {
                        // A different signal: from the tracer, as `kill` would.
                        qs = QueuedSig {
                            code: 0,
                            errno: 0,
                            pid: t.tracer,
                            uid: 0,
                            value: 0,
                        };
                    }
                    t.inject = Some((sig, qs));
                }
                StopKind::Entry | StopKind::Exit => raise = Some(sig),
                // An event stop drops it, as Linux's does.
                _ => {}
            }
        }
        if mode == Resume::Detach {
            t.tracer = 0;
            t.options = 0;
            t.seized = false;
            t.syscall_trace = false;
            t.queued = None;
            t.interrupt = false;
        } else if let Some(q) = t.queued.take()
            && (q.kind != StopKind::Exit || mode == Resume::Syscall)
        {
            // Straight into the next stop (its tracer will wait for it next).
            t.stop = Some(q);
            return;
        }
        p.run = RunState::Running;
        p.parked = false;
        if let Some(sig) = raise {
            p.raise(sig);
        }
        if step_in_syscall {
            p.raise(u64::from(SIGTRAP));
        }
    }

    /// `PTRACE_GETREGSET`/`PTRACE_SETREGSET` (`addr` = the note type, `data` = an
    /// iovec whose length is updated to what was transferred): a 32-bit one
    /// for a `compat` tracer. The regsets are those of the tracee's view
    /// (`task_user_regset_view`): the i386 ones while it runs 32-bit code
    /// (a `%cs` other than `__USER_CS`), whoever asks.
    #[allow(clippy::too_many_arguments)]
    fn ptrace_regset(
        &self,
        sh: &mut Shared,
        idx: usize,
        req: u64,
        note: u64,
        iov: u64,
        compat: bool,
        mem: &mut GuestMemory,
    ) -> i64 {
        let arch = self.arch;
        let iovec = if compat {
            mem.read_u32(iov)
                .and_then(|b| Ok((u64::from(b), u64::from(mem.read_u32(iov + 4)?))))
        } else {
            mem.read_u64(iov)
                .and_then(|b| Ok((b, mem.read_u64(iov + 8)?)))
        };
        let Ok((base, len)) = iovec else {
            return err(Errno::EFAULT);
        };
        let t = Self::tracee_state(sh, idx);
        let view32 = arch == Arch::X86_64 && t.regs[X86_CS] != 0x33;
        // The regset and the size of its unit, which the length must be a
        // multiple of.
        let (mut buf, unit): (Vec<u8>, usize) = match (note, arch) {
            (NT_PRSTATUS, Arch::X86_64) if view32 => (t.regs32(), 4),
            (NT_PRSTATUS, Arch::X86_64) => {
                let regs = t.regs[..27].iter().flat_map(|v| v.to_le_bytes()).collect();
                (regs, 8)
            }
            (NT_PRFPREG, Arch::X86_64) if view32 => (t.fsave(), 4),
            (NT_PRXFPREG, Arch::X86_64) if view32 => (fp_user(arch, &t.fp), 4),
            (NT_X86_XSTATE, Arch::X86_64) => (t.xstate(), 8),
            (NT_386_TLS, Arch::X86_64) if view32 => {
                let descs = (0..3u64).flat_map(|i| {
                    super::sys_misc::thread_area_get(&t.tls, 12 + i).unwrap_or([0; 16])
                });
                (descs.collect(), 16)
            }
            (NT_PRSTATUS, Arch::Aarch64) => {
                (t.regs.iter().flat_map(|v| v.to_le_bytes()).collect(), 1)
            }
            (NT_PRFPREG, Arch::X86_64) => (fp_user(arch, &t.fp), 8),
            (NT_PRFPREG, _) => (fp_user(arch, &t.fp), 1),
            (NT_ARM_SYSTEM_CALL, Arch::Aarch64) => ((t.a64_nr as i32).to_le_bytes().to_vec(), 1),
            _ => return err(Errno::EINVAL),
        };
        if !(len as usize).is_multiple_of(unit) {
            return err(Errno::EINVAL);
        }
        let n = (len as usize).min(buf.len());
        if req == PTRACE_GETREGSET {
            if mem.write(base, &buf[..n]).is_err() {
                return err(Errno::EFAULT);
            }
        } else {
            if mem.read(base, &mut buf[..n]).is_err() {
                return err(Errno::EFAULT);
            }
            let r = match (note, arch) {
                (NT_PRSTATUS, Arch::X86_64) if view32 => t.set_regs32(&buf[..n]),
                (NT_PRSTATUS, _) => {
                    for (i, c) in buf[..n].as_chunks::<8>().0.iter().enumerate() {
                        t.regs[i] = u64::from_le_bytes(*c);
                    }
                    t.regs_dirty = true;
                    Ok(())
                }
                (NT_PRFPREG, _) if view32 => {
                    t.set_fsave(&buf[..n]);
                    Ok(())
                }
                (NT_PRFPREG | NT_PRXFPREG, _) => {
                    t.set_fp_user(arch, &buf[..n]);
                    Ok(())
                }
                (NT_X86_XSTATE, _) => t.set_xstate(&buf[..n]),
                (NT_386_TLS, _) => {
                    let mut tls = t.tls;
                    let set = buf[..n]
                        .as_chunks::<16>()
                        .0
                        .iter()
                        .enumerate()
                        .try_for_each(|(i, d)| {
                            super::sys_misc::thread_area_set(&mut tls, 12 + i as u64, d)
                        });
                    set.map(|()| {
                        t.tls = tls;
                        t.apply_tls = true;
                    })
                }
                (NT_ARM_SYSTEM_CALL, _) if n == 4 => {
                    t.a64_nr = i64::from(i32::from_le_bytes(buf[..4].try_into().unwrap())) as u64;
                    t.regs_dirty = true;
                    Ok(())
                }
                _ => Ok(()),
            };
            if let Err(e) = r {
                return e;
            }
        }
        let wrote = if compat {
            mem.write(iov + 4, &(n as u32).to_le_bytes())
        } else {
            mem.write(iov + 8, &(n as u64).to_le_bytes())
        };
        if wrote.is_err() {
            return err(Errno::EFAULT);
        }
        0
    }

    /// `struct ptrace_syscall_info` for the stop `t` is in: an i386 call's
    /// is `AUDIT_ARCH_I386`, its arguments `%ebx`..`%ebp`.
    fn syscall_info(arch: Arch, t: &PtraceState) -> Vec<u8> {
        const AUDIT_ARCH_X86_64: u32 = 0xc000_003e;
        const AUDIT_ARCH_AARCH64: u32 = 0xc000_00b7;
        let r = &t.regs;
        let (audit, ip, sp, args) = match arch {
            Arch::X86_64 if t.compat => (
                AUDIT_ARCH_I386,
                r[X86_RIP],
                r[X86_RSP],
                [r[5], r[11], r[12], r[13], r[14], r[4]],
            ),
            Arch::X86_64 => (
                AUDIT_ARCH_X86_64,
                r[X86_RIP],
                r[X86_RSP],
                [r[14], r[13], r[12], r[7], r[9], r[8]],
            ),
            Arch::Aarch64 => (
                AUDIT_ARCH_AARCH64,
                r[32],
                r[31],
                [r[0], r[1], r[2], r[3], r[4], r[5]],
            ),
        };
        let mut info = vec![0u8; 88];
        info[4..8].copy_from_slice(&audit.to_le_bytes());
        info[8..16].copy_from_slice(&ip.to_le_bytes());
        info[16..24].copy_from_slice(&sp.to_le_bytes());
        let kind = t.stop.map(|s| s.kind);
        let put_call = |info: &mut Vec<u8>| {
            info[24..32].copy_from_slice(&t.sys_nr(arch).to_le_bytes());
            for (i, a) in args.iter().enumerate() {
                info[32 + i * 8..40 + i * 8].copy_from_slice(&a.to_le_bytes());
            }
        };
        let size = match kind {
            Some(StopKind::Entry) => {
                info[0] = 1;
                put_call(&mut info);
                80
            }
            Some(StopKind::Exit) => {
                info[0] = 2;
                // `syscall_get_error`: an i386 call's error is its `%eax`,
                // sign-extended.
                let ret = t.ret_reg(arch);
                let error = if t.compat {
                    i64::from(ret as i32)
                } else {
                    ret as i64
                };
                let is_error = (-4095..0).contains(&error);
                let rval = if is_error { error } else { ret as i64 };
                info[24..32].copy_from_slice(&rval.to_le_bytes());
                info[32] = u8::from(is_error);
                33
            }
            Some(StopKind::Seccomp) => {
                info[0] = 3;
                put_call(&mut info);
                info[80..84].copy_from_slice(&(t.event_msg as u32).to_le_bytes());
                84
            }
            _ => 24,
        };
        info.truncate(size);
        info
    }

    // ---- the tracee's side ----------------------------------------------

    /// Stop the current task for its tracer: snapshot its registers, take it
    /// off the CPU ([`RunState::Traced`]) and tell the tracer. Must be called
    /// without `sh` held.
    #[allow(clippy::too_many_arguments)]
    fn ptrace_stop(
        &self,
        cx: &mut ServiceCtx,
        vcpu: &mut dyn Vcpu,
        kind: StopKind,
        sig: u32,
        event: u32,
        phase: Phase,
        ret: Option<u64>,
    ) {
        let arch = self.arch;
        let fs_base = cx.cur.fs_base;
        let Some(t) = cx.cur.ptrace.as_mut() else {
            return;
        };
        let mut regs = snapshot(arch, vcpu, phase, fs_base);
        let nr = if phase == Phase::User {
            u64::MAX
        } else {
            t.sc_nr.unwrap_or_else(|| vcpu.syscall_nr())
        };
        if arch == Arch::X86_64 {
            regs[X86_ORIG_RAX] = nr;
        }
        if let Some(v) = ret {
            match arch {
                Arch::X86_64 => regs[X86_RAX] = v,
                Arch::Aarch64 => regs[0] = v,
            }
        }
        t.regs = regs;
        t.a64_nr = nr;
        t.regs_dirty = false;
        t.apply_fp = false;
        t.compat = arch == Arch::X86_64 && phase != Phase::User && vcpu.syscall_ia32();
        if arch == Arch::X86_64 {
            t.tls = std::array::from_fn(|i| vcpu.tls_desc(i));
            t.apply_tls = false;
        }
        t.fp = vcpu.fp_state();
        // A return the tracer redirected at an earlier stop of this call
        // shows as it will happen.
        if phase != Phase::User
            && let Some([rip, flags, rcx, r11]) = t.ret_regs
        {
            t.regs[X86_RIP] = rip;
            t.regs[X86_EFLAGS] = flags;
            t.regs[X86_RCX] = rcx;
            t.regs[X86_R11] = r11;
        }
        t.stop = Some(Stop {
            kind,
            sig,
            event,
            phase,
            reported: false,
        });
        cx.cur.run = RunState::Traced;
        cx.cur.parked = false;
        let tracer = t.tracer;
        let mut sh = self.shared.lock().unwrap();
        Self::notify_tracer(&mut sh, &cx.cur, tracer, CLD_TRAPPED, sig as i32);
    }

    /// A synchronous fault in a tracee (`SIGSEGV`, `SIGILL`, a breakpoint, a
    /// finished single step…): it stops for its tracer before anything is
    /// delivered, as Linux's signal-delivery stop. `Some` is how the exit
    /// ends (stopped); `None` lets the fault be delivered as usual — untraced,
    /// or the tracer passed this very signal back (the instruction ran again
    /// and faulted again: now it is delivered, to a handler or fatally).
    pub(super) fn ptrace_fault(
        &self,
        cx: &mut ServiceCtx,
        fault: &super::signal::Fault,
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> Option<Serviced> {
        let t = cx.cur.ptrace.as_ref()?;
        if t.tracer == 0 {
            return None;
        }
        // Killed while stopped (`PTRACE_KILL`, `kill -9`): the instruction
        // faults again on the way out, but the kill comes first.
        if cx.cur.pending & (1 << (SIGKILL - 1)) != 0 {
            self.die_of_signal(cx, SIGKILL as u32, mem);
            return Some(Serviced::Ended);
        }
        let t = cx.cur.ptrace.as_mut()?;
        if let Some((sig, _)) = t.inject
            && sig == fault.sig
        {
            t.inject = None;
            return None;
        }
        let mut si = [0u8; 128];
        si[0..4].copy_from_slice(&(fault.sig as i32).to_le_bytes());
        si[8..12].copy_from_slice(&(fault.code as i32).to_le_bytes());
        si[16..24].copy_from_slice(&fault.addr.to_le_bytes());
        t.siginfo = si;
        self.ptrace_stop(
            cx,
            vcpu,
            StopKind::Signal,
            fault.sig as u32,
            0,
            Phase::User,
            None,
        );
        Some(Serviced::Blocked)
    }

    /// Tell `tracer` that `tracee` changed state: `SIGCHLD` with the given
    /// `si_code`/status, and a wake-up for its `wait`.
    pub(super) fn notify_tracer(
        sh: &mut Shared,
        tracee: &ProcInfo,
        tracer: i32,
        code: i32,
        status: i32,
    ) {
        let qs = QueuedSig {
            code,
            errno: 0,
            pid: tracee.tgid,
            uid: tracee.creds.ruid,
            value: u64::from(status as u32),
        };
        for slot in sh.procs.iter_mut().flatten() {
            if slot.info.pid == tracer {
                slot.info.raise_with(SIGCHLD, qs);
                slot.info.parked = false;
            }
        }
    }

    /// A traced task's next pending signal (all but `SIGKILL` stop for the
    /// tracer first), dequeued, with its info recorded for `GETSIGINFO`.
    fn ptrace_take_signal(cx: &mut ServiceCtx) -> Option<u64> {
        let t = cx.cur.ptrace.as_ref()?;
        if t.tracer == 0 || t.inject.is_some() || cx.cur.pending & (1 << (SIGKILL - 1)) != 0 {
            return None;
        }
        let deliverable = cx.cur.pending & !cx.cur.blocked;
        if deliverable == 0 {
            return None;
        }
        let sig = u64::from(deliverable.trailing_zeros()) + 1;
        cx.cur.pending &= !(1 << (sig - 1));
        let (info, more) = cx.cur.take_siginfo(sig);
        if more {
            cx.cur.pending |= 1 << (sig - 1);
        }
        cx.cur.ptrace.as_mut()?.siginfo = siginfo_bytes(sig, info);
        Some(sig)
    }

    /// Signal delivery for a tracee, from [`Kernel::deliver_pending_signals`]:
    /// a signal the tracer passed on is delivered here (returned, with its
    /// info posted, for the caller to act on); otherwise the next pending one
    /// stops the task for the tracer. `Err(())` means it stopped.
    pub(super) fn ptrace_signal_delivery(
        &self,
        cx: &mut ServiceCtx,
        vcpu: &mut dyn Vcpu,
    ) -> Result<Option<u64>, ()> {
        let Some(t) = cx.cur.ptrace.as_mut() else {
            return Ok(None);
        };
        if let Some((sig, qs)) = t.inject.take() {
            if cx.cur.blocked & (1 << (sig - 1)) != 0 {
                // Blocked now: it waits like any other.
                cx.cur.raise_with(sig, qs);
                return Ok(None);
            }
            cx.cur.post_siginfo(sig, qs);
            return Ok(Some(sig));
        }
        let Some(sig) = Self::ptrace_take_signal(cx) else {
            return Ok(None);
        };
        // Still in the syscall (it blocked, or restarts): resuming re-traps it.
        let phase = if cx.block || cx.restart_syscall {
            Phase::Before
        } else {
            Phase::User
        };
        self.ptrace_stop(cx, vcpu, StopKind::Signal, sig as u32, 0, phase, None);
        Err(())
    }

    /// The traced part of a syscall's start. `Err(flow)` ends the step there
    /// (a stop, a death, an injected signal's handler); otherwise the syscall
    /// number to use instead of the vcpu's, and the result if the call needn't
    /// (or mustn't again) run.
    pub(super) fn ptrace_syscall_enter(
        &self,
        cx: &mut ServiceCtx,
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> Result<(Option<u64>, Option<i64>), Serviced> {
        let arch = self.arch;
        // Killed while stopped: die now.
        if cx.cur.pending & (1 << (SIGKILL - 1)) != 0 {
            self.die_of_signal(cx, SIGKILL as u32, mem);
            return Err(Serviced::Ended);
        }
        let Some(t) = cx.cur.ptrace.as_mut() else {
            return Ok((None, None));
        };
        if std::mem::take(&mut t.apply_gprs) {
            let regs = t.regs;
            apply_in_syscall(arch, vcpu, &regs);
        }
        t.put_tls(vcpu);
        // A signal the tracer passed on, before the call starts: delivered now,
        // and the call made again after its handler.
        if t.sc_result.is_none() && t.inject.is_some() {
            cx.restart_syscall = true;
            let delivered = self.deliver_pending_signals(cx, vcpu, mem);
            if matches!(cx.cur.run, RunState::Zombie(_)) {
                return Err(Serviced::Ended);
            }
            if delivered {
                if let Some(t) = cx.cur.ptrace.as_mut() {
                    t.end_syscall();
                }
                if mem.take_tlb_dirty() {
                    vcpu.flush_tlb();
                }
                return Err(Serviced::Resume);
            }
            if cx.cur.run != RunState::Running {
                cx.block = true;
                return Err(Serviced::Blocked);
            }
            cx.restart_syscall = false;
        }
        let t = cx.cur.ptrace.as_mut().unwrap();
        if t.tracer != 0 && std::mem::take(&mut t.interrupt) {
            self.ptrace_stop(
                cx,
                vcpu,
                StopKind::Event,
                SIGTRAP,
                EVENT_STOP,
                Phase::Before,
                None,
            );
            cx.block = true;
            return Err(Serviced::Blocked);
        }
        let t = cx.cur.ptrace.as_mut().unwrap();
        if let Some(r) = t.sc_result {
            return Ok((t.sc_nr, Some(r)));
        }
        if !t.sc_entered {
            t.sc_entered = true;
            if t.tracer != 0 && t.syscall_trace {
                let sig = t.sig_status(SIGTRAP);
                let ret = (arch == Arch::X86_64).then_some(ENOSYS);
                self.ptrace_stop(cx, vcpu, StopKind::Entry, sig, 0, Phase::Before, ret);
                cx.block = true;
                return Err(Serviced::Blocked);
            }
        }
        let t = cx.cur.ptrace.as_mut().unwrap();
        let nr = t.sc_nr.unwrap_or_else(|| vcpu.syscall_nr());
        // PTRACE_O_TRACEEXIT: the exit is reported before it happens.
        let nr = if vcpu.syscall_ia32() {
            crate::abi::arch::i386::entry(nr & 0xffff_ffff)
                .nr64()
                .unwrap_or(u64::MAX)
        } else {
            nr
        };
        let sys = crate::abi::arch::decode(arch, nr);
        if t.tracer != 0
            && t.options & O_TRACEEXIT != 0
            && !t.exit_event_done
            && matches!(
                sys,
                crate::abi::arch::Sysno::Exit | crate::abi::arch::Sysno::ExitGroup
            )
        {
            t.exit_event_done = true;
            t.event_msg = (vcpu.syscall_args()[0] & 0xff) << 8;
            self.ptrace_stop(
                cx,
                vcpu,
                StopKind::Event,
                SIGTRAP,
                EVENT_EXIT,
                Phase::Before,
                None,
            );
            cx.block = true;
            return Err(Serviced::Blocked);
        }
        let t = cx.cur.ptrace.as_ref().unwrap();
        if t.sc_nr.is_some_and(|n| n as u32 == u32::MAX) {
            // The tracer skipped it (the number is an `int`, so a 32-bit
            // tracer's `orig_eax` of -1 does too).
            return Ok((t.sc_nr, Some(t.skip_ret as i64)));
        }
        Ok((t.sc_nr, None))
    }

    /// Whether the seccomp filters still have to run for this call.
    pub(super) fn ptrace_seccomp_pending(cx: &ServiceCtx) -> bool {
        !cx.cur.ptrace.as_ref().is_some_and(|t| t.sc_seccomp_done)
    }

    /// seccomp's `SECCOMP_RET_TRACE` with `data`: with a tracer that asked
    /// for it (`PTRACE_O_TRACESECCOMP`), the call stops for it (the caller
    /// sees [`Kernel::ptrace_seccomp_stop`] and lets the call proceed once
    /// resumed); without one it fails with `ENOSYS`, as on Linux.
    pub(super) fn seccomp_ret_trace(cx: &mut ServiceCtx, data: u32) -> Option<i64> {
        match cx.cur.ptrace.as_mut() {
            Some(t) if t.tracer != 0 && t.options & O_TRACESECCOMP != 0 => {
                t.seccomp_trace = Some(data);
                None
            }
            _ => Some(err(Errno::ENOSYS)),
        }
    }

    /// Stop for a seccomp `RET_TRACE` the filters just asked for, if any.
    pub(super) fn ptrace_seccomp_stop(&self, cx: &mut ServiceCtx, vcpu: &mut dyn Vcpu) -> bool {
        let Some(t) = cx.cur.ptrace.as_mut() else {
            return false;
        };
        let Some(data) = t.seccomp_trace.take() else {
            return false;
        };
        t.sc_seccomp_done = true;
        t.event_msg = u64::from(data);
        let ret = (self.arch == Arch::X86_64).then_some(ENOSYS);
        self.ptrace_stop(
            cx,
            vcpu,
            StopKind::Seccomp,
            SIGTRAP,
            EVENT_SECCOMP,
            Phase::Before,
            ret,
        );
        cx.block = true;
        true
    }

    /// The traced part of a syscall's end, once its result `ret` is known and
    /// before it's written back: events, the exit stop and signal stops, each
    /// a stop the call is re-trapped from. `Some(flow)` ends the step there;
    /// otherwise `ret` is what to write back (the tracer may have changed it).
    pub(super) fn ptrace_syscall_exit(
        &self,
        cx: &mut ServiceCtx,
        ret: &mut i64,
        vcpu: &mut dyn Vcpu,
    ) -> Option<Serviced> {
        let t = cx.cur.ptrace.as_mut()?;
        if cx.exec_ok {
            // A new image (or `rt_sigreturn`'s restored context): the vcpu is
            // in user mode now, so these stops just resume it.
            let execed = t.execed.take();
            t.end_syscall();
            t.ret_regs = None;
            let traced = t.tracer != 0;
            let exit_stop = traced && t.syscall_trace;
            let (options, seized) = (t.options, t.seized);
            let exit = Stop {
                kind: StopKind::Exit,
                sig: t.sig_status(SIGTRAP),
                event: 0,
                phase: Phase::User,
                reported: false,
            };
            match execed {
                Some(old) if traced && options & O_TRACEEXEC != 0 => {
                    t.event_msg = old as u64;
                    if exit_stop {
                        t.queued = Some(exit);
                    }
                    self.ptrace_stop(
                        cx,
                        vcpu,
                        StopKind::Event,
                        SIGTRAP,
                        EVENT_EXEC,
                        Phase::User,
                        Some(0),
                    );
                }
                Some(_) if traced => {
                    if !seized {
                        // The old-style exec trap: a plain SIGTRAP.
                        let qs = QueuedSig {
                            code: 0,
                            errno: 0,
                            pid: cx.cur.tgid,
                            uid: cx.cur.creds.ruid,
                            value: 0,
                        };
                        cx.cur.raise_with(u64::from(SIGTRAP), qs);
                    }
                    if exit_stop {
                        self.ptrace_stop(
                            cx,
                            vcpu,
                            StopKind::Exit,
                            exit.sig,
                            0,
                            Phase::User,
                            Some(0),
                        );
                    }
                }
                _ if exit_stop => {
                    self.ptrace_stop(cx, vcpu, StopKind::Exit, exit.sig, 0, Phase::User, None);
                }
                _ => {}
            }
            if cx.cur.run == RunState::Traced {
                return Some(Serviced::Resume);
            }
            Self::ptrace_forget_if_idle(cx);
            return None;
        }
        if cx.restart_syscall {
            // It will run again from the start.
            t.end_syscall();
            t.ret_regs = None;
            return None;
        }
        t.sc_result = Some(*ret);
        let traced = t.tracer != 0;
        if let Some((event, child)) = t.pending_event.take()
            && traced
        {
            t.event_msg = child;
            self.ptrace_stop(
                cx,
                vcpu,
                StopKind::Event,
                SIGTRAP,
                event,
                Phase::Done,
                Some(*ret as u64),
            );
            cx.block = true;
            return Some(Serviced::Blocked);
        }
        let t = cx.cur.ptrace.as_mut().unwrap();
        if !std::mem::replace(&mut t.sc_exit_done, true) && traced && t.syscall_trace {
            let sig = t.sig_status(SIGTRAP);
            self.ptrace_stop(
                cx,
                vcpu,
                StopKind::Exit,
                sig,
                0,
                Phase::Done,
                Some(*ret as u64),
            );
            cx.block = true;
            return Some(Serviced::Blocked);
        }
        if let Some(sig) = Self::ptrace_take_signal(cx) {
            self.ptrace_stop(
                cx,
                vcpu,
                StopKind::Signal,
                sig as u32,
                0,
                Phase::Done,
                Some(*ret as u64),
            );
            cx.block = true;
            return Some(Serviced::Blocked);
        }
        let t = cx.cur.ptrace.as_mut().unwrap();
        *ret = t.sc_result.take().unwrap_or(*ret);
        t.end_syscall();
        Self::ptrace_forget_if_idle(cx);
        None
    }

    /// Drop the ptrace state of a task no longer traced and with nothing left
    /// in progress.
    fn ptrace_forget_if_idle(cx: &mut ServiceCtx) {
        if cx.cur.ptrace.as_ref().is_some_and(|t| t.idle()) {
            cx.cur.ptrace = None;
        }
    }

    /// A stop at a preemption point (`PTRACE_INTERRUPT` of a task computing in
    /// user mode). Whether it stopped.
    pub(super) fn ptrace_interrupt_user(&self, cx: &mut ServiceCtx, vcpu: &mut dyn Vcpu) -> bool {
        let Some(t) = cx.cur.ptrace.as_mut() else {
            return false;
        };
        if t.tracer == 0 || !std::mem::take(&mut t.interrupt) {
            return false;
        }
        self.ptrace_stop(
            cx,
            vcpu,
            StopKind::Event,
            SIGTRAP,
            EVENT_STOP,
            Phase::User,
            None,
        );
        true
    }

    /// Whether the task has something for the tracer at a preemption point.
    pub(super) fn ptrace_wants_boundary(p: &ProcInfo) -> bool {
        p.ptrace
            .as_ref()
            .is_some_and(|t| t.interrupt || t.inject.is_some())
    }

    /// Ptrace state for a new task made by `parent` (traced) with `flags`:
    /// whether the tracer gets an event for it (and which), and the child's
    /// own state if it starts traced. `child_vcpu` is the child's vcpu, for
    /// its first stop's registers.
    pub(super) fn ptrace_clone(
        &self,
        sh: &mut Shared,
        parent: &mut ProcInfo,
        child: &mut ProcInfo,
        flags: u64,
        exit_signal: u64,
        child_vcpu: &mut dyn Vcpu,
    ) {
        child.ptrace = None;
        let Some(t) = parent.ptrace.as_mut() else {
            return;
        };
        if t.tracer == 0 || flags & super::CLONE_UNTRACED != 0 {
            return;
        }
        let event = if flags & super::CLONE_VFORK != 0 {
            EVENT_VFORK
        } else if exit_signal != SIGCHLD {
            EVENT_CLONE
        } else {
            EVENT_FORK
        };
        let enabled = t.options & (1 << event) != 0;
        if !enabled && flags & super::CLONE_PTRACE == 0 {
            return;
        }
        if enabled {
            t.pending_event = Some((event, child.pid as u64));
        }
        // The child starts traced, in a stop: `SIGSTOP` for an old-style
        // tracer, `PTRACE_EVENT_STOP` for a seizing one.
        let mut c = PtraceState::new(t.tracer);
        c.options = t.options;
        c.seized = t.seized;
        let (sig, event, kind) = if c.seized {
            (SIGTRAP, EVENT_STOP, StopKind::Event)
        } else {
            (SIGSTOP, 0, StopKind::Signal)
        };
        c.regs = snapshot(self.arch, child_vcpu, Phase::User, child.fs_base);
        c.a64_nr = u64::MAX;
        if kind == StopKind::Signal {
            let qs = QueuedSig {
                code: 0,
                errno: 0,
                pid: parent.tgid,
                uid: parent.creds.ruid,
                value: 0,
            };
            c.siginfo = siginfo_bytes(u64::from(SIGSTOP), Some(qs));
        }
        c.stop = Some(Stop {
            kind,
            sig,
            event,
            phase: Phase::User,
            reported: false,
        });
        let tracer = c.tracer;
        child.ptrace = Some(Box::new(c));
        child.run = RunState::Traced;
        Self::notify_tracer(sh, child, tracer, CLD_TRAPPED, sig as i32);
    }

    /// For tracer `cur`'s `wait`: whether it traces any task `matches`
    /// accepts, and the first event of one waiting to be reported (a stop,
    /// or the exit of a task that isn't its own child).
    pub(super) fn ptrace_wait_scan(
        sh: &Shared,
        cur: i32,
        matches: impl Fn(&ProcInfo) -> bool,
    ) -> (bool, Option<TraceReport>) {
        let mut any = false;
        for p in sh.procs.iter().flatten().map(|p| &p.info) {
            let Some(t) = p.ptrace.as_ref() else {
                continue;
            };
            if t.tracer != cur || !matches(p) {
                continue;
            }
            any = true;
            match p.run {
                RunState::Traced => {
                    if let Some(s) = t.stop.filter(|s| !s.reported) {
                        return (true, Some(TraceReport::Stop(p.pid, s.status())));
                    }
                }
                RunState::Zombie(cause)
                    if !p.in_flight && !t.exit_reported && (p.ppid != cur || p.is_thread) =>
                {
                    return (true, Some(TraceReport::Exit(p.pid, cause)));
                }
                _ => {}
            }
        }
        (any, None)
    }

    /// The tracer's `wait` reported `r`: a stop isn't reported twice; an exit
    /// detaches the task, which its parent may now reap.
    pub(super) fn ptrace_consume(&self, sh: &mut Shared, cx: &ServiceCtx, r: &TraceReport) {
        let mut ppid = None;
        match *r {
            TraceReport::Stop(pid, _) => {
                for p in sh.procs.iter_mut().flatten() {
                    if p.info.pid == pid
                        && let Some(s) = p.info.ptrace.as_mut().and_then(|t| t.stop.as_mut())
                    {
                        s.reported = true;
                    }
                }
            }
            TraceReport::Exit(pid, _) => {
                for p in sh.procs.iter_mut().flatten() {
                    if p.info.pid == pid
                        && let Some(t) = p.info.ptrace.as_mut()
                    {
                        t.exit_reported = true;
                        t.tracer = 0;
                        ppid = Some(pid);
                    }
                }
            }
        }
        if let Some(pid) = ppid {
            self.notify_parent_of_exit(sh, pid);
            sh.reap_unwaited_zombies(cx);
        }
    }

    /// Zombie `pid`'s tracer is done with it: now its parent gets the exit
    /// signal it was held back from (see `exit_task`), and may reap it.
    fn notify_parent_of_exit(&self, sh: &mut Shared, pid: i32) {
        let Some(z) = sh.procs.iter().flatten().find(|p| p.info.pid == pid) else {
            return;
        };
        let (ppid, sig) = (
            z.info.ppid,
            if z.info.is_thread {
                0
            } else {
                u64::from(z.info.exit_signal)
            },
        );
        let (code, status) = match z.info.run {
            RunState::Zombie(super::ExitCause::Exited(c)) => (1, c & 0xff),
            RunState::Zombie(super::ExitCause::Signaled(s)) => (2, s),
            _ => return,
        };
        let qs = QueuedSig {
            code,
            errno: 0,
            pid: z.info.tgid,
            uid: z.info.creds.ruid,
            value: u64::from(status as u32),
        };
        for p in sh.procs.iter_mut().flatten() {
            if p.info.pid == ppid {
                if (1..=64).contains(&sig) {
                    let qs = self.seen_by(p.info.ns.pid, qs);
                    p.info.raise_with(sig, qs);
                }
                p.info.parked = false;
            }
        }
    }

    /// Tracer `me` is exiting: its tracees are detached (and resumed), or with
    /// `PTRACE_O_EXITKILL` killed; the parents of those already dead hear of
    /// it now.
    pub(super) fn ptrace_tracer_exit(&self, sh: &mut Shared, me: i32) {
        let arch = self.arch;
        let mut parents = Vec::new();
        for p in sh.procs.iter_mut().flatten() {
            let info = &mut p.info;
            let Some(t) = info.ptrace.as_mut() else {
                continue;
            };
            if t.tracer != me {
                continue;
            }
            let kill = t.options & O_EXITKILL != 0;
            t.tracer = 0;
            t.options = 0;
            t.seized = false;
            t.syscall_trace = false;
            t.interrupt = false;
            t.queued = None;
            if let Some(stop) = t.stop.take() {
                // Let it go as a plain resume would.
                match stop.phase {
                    Phase::Before => {
                        if matches!(stop.kind, StopKind::Entry | StopKind::Seccomp) {
                            t.sc_nr = None;
                        }
                    }
                    Phase::Done => t.sc_result = Some(t.ret_reg(arch) as i64),
                    Phase::User => {}
                }
            }
            if info.run == RunState::Traced {
                info.run = RunState::Running;
                info.parked = false;
            }
            if matches!(info.run, RunState::Zombie(_)) {
                // (Its parent was told already if that's the tracer itself.)
                if info.ppid != me {
                    parents.push(info.pid);
                }
            } else if kill {
                info.raise(SIGKILL);
                info.parked = false;
            }
        }
        for pid in parents {
            self.notify_parent_of_exit(sh, pid);
        }
    }
}
