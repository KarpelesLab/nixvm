//! Signal handling: per-process handler tables, masks, `kill`/`tgkill`
//! delivery, the DEFAULT dispositions, and — for *synchronous* (fault) signals —
//! full custom-handler invocation.
//!
//! A `SIGSEGV`/`SIGILL`/`SIGBUS` raised by the running instruction is delivered
//! to the guest's handler if one is installed: [`Kernel::deliver_fault_signal`]
//! builds the x86-64 `rt_sigframe` on the (alternate or interrupted) stack,
//! points the vcpu at the handler, and [`Kernel::sys_rt_sigreturn`] restores the
//! saved context when it returns. This is what lets a JIT that faults on purpose
//! (JSC/V8 use `SIGSEGV` for stack-limit and null checks) run.
//!
//! *Asynchronous* signals (from `kill`/`tgkill`) still only take their default
//! action — interrupting the guest at an arbitrary instruction to run a handler
//! is not modeled — and a pending async signal with a real handler is dropped
//! rather than left to spin the scheduler.

use super::{Kernel, RunState, SA_ONSTACK, SIGSEGV, SS_DISABLE, err};
use crate::abi::errno::Errno;
use crate::vcpu::GuestMemory;

/// `SIG_DFL`: take the default action for the signal.
const SIG_DFL: u64 = 0;
/// `SIG_IGN`: ignore the signal.
const SIG_IGN: u64 = 1;

/// Highest supported signal number (`_NSIG - 1` on Linux).
const NSIG: u64 = 64;
const SIGKILL: u64 = 9;
const SIGSTOP: u64 = 19;

impl Kernel {
    /// `rt_sigaction(sig, act, oldact, sigsetsize)` — store the disposition for
    /// `sig`. `sigsetsize` is accepted but ignored. Changing `SIGKILL`/`SIGSTOP`
    /// is rejected with `EINVAL`.
    pub(super) fn sys_rt_sigaction(
        &mut self,
        sig: u64,
        act: u64,
        oldact: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        if !(1..=NSIG).contains(&sig) {
            return err(Errno::EINVAL);
        }
        if act != 0 && (sig == SIGKILL || sig == SIGSTOP) {
            return err(Errno::EINVAL);
        }
        let idx = sig as usize;
        if oldact != 0 {
            // struct sigaction: handler u64, flags u64, restorer u64, mask u64.
            let old = self.cur.handlers[idx];
            let mut buf = [0u8; 32];
            buf[0..8].copy_from_slice(&old.handler.to_le_bytes());
            buf[8..16].copy_from_slice(&old.flags.to_le_bytes());
            buf[16..24].copy_from_slice(&old.restorer.to_le_bytes());
            buf[24..32].copy_from_slice(&old.mask.to_le_bytes());
            if mem.write(oldact, &buf).is_err() {
                return err(Errno::EFAULT);
            }
        }
        if act != 0 {
            let mut buf = [0u8; 32];
            if mem.read(act, &mut buf).is_err() {
                return err(Errno::EFAULT);
            }
            let word = |o: usize| u64::from_le_bytes(buf[o..o + 8].try_into().unwrap());
            self.cur.handlers[idx] = super::SigAction {
                handler: word(0),
                flags: word(8),
                restorer: word(16),
                mask: word(24),
            };
        }
        0
    }

    /// `sigaltstack(ss, old_ss)` — get/set the alternate signal stack a handler
    /// registered `SA_ONSTACK` runs on. `stack_t` is `{ void *ss_sp; int
    /// ss_flags; size_t ss_size }`.
    pub(super) fn sys_sigaltstack(&mut self, ss: u64, old_ss: u64, mem: &mut GuestMemory) -> i64 {
        let (sp, size, flags) = self.cur.altstack;
        if old_ss != 0 {
            let mut buf = [0u8; 24];
            buf[0..8].copy_from_slice(&sp.to_le_bytes());
            buf[8..12].copy_from_slice(&(flags as u32).to_le_bytes());
            buf[16..24].copy_from_slice(&size.to_le_bytes());
            if mem.write(old_ss, &buf).is_err() {
                return err(Errno::EFAULT);
            }
        }
        if ss != 0 {
            let mut buf = [0u8; 24];
            if mem.read(ss, &mut buf).is_err() {
                return err(Errno::EFAULT);
            }
            let new_sp = u64::from_le_bytes(buf[0..8].try_into().unwrap());
            let new_flags = u64::from(u32::from_le_bytes(buf[8..12].try_into().unwrap()));
            let new_size = u64::from_le_bytes(buf[16..24].try_into().unwrap());
            self.cur.altstack = (new_sp, new_size, new_flags);
        }
        0
    }

    /// `rt_sigprocmask(how, set, oldset, sigsetsize)` — read/modify the blocked
    /// mask. `sigsetsize` is accepted but ignored.
    pub(super) fn sys_rt_sigprocmask(
        &mut self,
        how: u64,
        set: u64,
        oldset: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const SIG_BLOCK: u64 = 0;
        const SIG_UNBLOCK: u64 = 1;
        const SIG_SETMASK: u64 = 2;

        if oldset != 0 && mem.write(oldset, &self.cur.blocked.to_le_bytes()).is_err() {
            return err(Errno::EFAULT);
        }
        if set != 0 {
            let Ok(mask) = mem.read_u64(set) else {
                return err(Errno::EFAULT);
            };
            match how {
                SIG_BLOCK => self.cur.blocked |= mask,
                SIG_UNBLOCK => self.cur.blocked &= !mask,
                SIG_SETMASK => self.cur.blocked = mask,
                _ => return err(Errno::EINVAL),
            }
        }
        0
    }

    /// `rt_sigpending(set, sigsetsize)` — report the pending-signal mask.
    pub(super) fn sys_rt_sigpending(&mut self, set: u64, mem: &mut GuestMemory) -> i64 {
        if set != 0 && mem.write(set, &self.cur.pending.to_le_bytes()).is_err() {
            return err(Errno::EFAULT);
        }
        0
    }

    /// `kill`/`tkill(pid, sig)` — post `sig` to the target process. `sig == 0`
    /// is an existence check. `pid <= 0` or `pid == self` targets the caller.
    pub(super) fn sys_kill(&mut self, pid: i64, sig: u64) -> i64 {
        if sig == 0 {
            return if pid <= 0 || self.pid_exists(pid) {
                0
            } else {
                err(Errno::ESRCH)
            };
        }
        if sig > NSIG {
            return err(Errno::EINVAL);
        }
        let bit = 1u64 << (sig - 1);
        if pid <= 0 || pid == i64::from(self.cur.pid) {
            self.cur.pending |= bit;
            return 0;
        }
        for slot in self.procs.iter_mut().flatten() {
            if i64::from(slot.info.pid) == pid {
                slot.info.pending |= bit;
                return 0;
            }
        }
        err(Errno::ESRCH)
    }

    /// Whether a process with `pid` exists (the running process is held in
    /// `self.cur`, out of the table, during its slice).
    fn pid_exists(&self, pid: i64) -> bool {
        pid == i64::from(self.cur.pid)
            || self
                .procs
                .iter()
                .flatten()
                .any(|p| i64::from(p.info.pid) == pid)
    }

    /// Apply the DEFAULT disposition of every deliverable pending signal for the
    /// current process. Runs once after each serviced syscall; it never loops
    /// waiting for a signal and never invokes a real handler.
    pub(super) fn deliver_pending_signals(&mut self) {
        if !matches!(self.cur.run, RunState::Running) {
            return;
        }
        let deliverable = self.cur.pending & !self.cur.blocked;
        if deliverable == 0 {
            return;
        }
        for sig in 1..=NSIG {
            let bit = 1u64 << (sig - 1);
            if deliverable & bit == 0 {
                continue;
            }
            // We are about to act on this signal in every branch below.
            self.cur.pending &= !bit;
            match self.cur.handlers[sig as usize].handler {
                SIG_IGN => {}
                // An asynchronously-posted signal (kill/tgkill) with a real
                // handler: delivering it would need to interrupt the guest at an
                // arbitrary point, which this kernel doesn't do — only
                // *synchronous* (fault) signals are delivered to handlers, via
                // `deliver_fault_signal`. Drop it rather than deadlock.
                h if h != SIG_DFL => {}
                // SIG_DFL: ignore the "ignored-by-default" set, else terminate.
                _ if is_default_ignored(sig) => {}
                _ => {
                    self.cur.run = RunState::Zombie(128 + sig as i32);
                    return;
                }
            }
        }
    }
}

// ---- synchronous (fault) signal delivery ----------------------------------
//
// x86-64 `rt_sigframe` the kernel pushes on delivery, and `rt_sigreturn`
// restores. Offsets are into the frame (which starts at the new `rsp`):
//
//   +0    pretcode (return address = sa_restorer)
//   +8    ucontext: uc_flags(+8) uc_link(+16) uc_stack(+24: sp,flags,size)
//         uc_mcontext(+48): the 23 gregs r8..cr2, then fpstate ptr, reserved
//         uc_sigmask(+296)
//   +8 + sizeof(ucontext)   siginfo (128 bytes)
//
// The gregs order matches glibc's `REG_*` indices, so a handler reading
// `uc_mcontext.gregs[REG_RIP]` (JSC does, to inspect/skip its own traps) sees
// the right value.
const UC_OFF: u64 = 8; // ucontext within the frame
const MCTX_OFF: u64 = UC_OFF + 40; // uc_mcontext within the frame
const GREG_COUNT: usize = 23;
/// Byte offset of each greg within `uc_mcontext`, in `REG_*` order.
const REG_RIP: usize = 16;
const REG_EFL: usize = 17;
const REG_CSGSFS: usize = 18;
const REG_RSP: usize = 15;
/// Total ucontext size the kernel writes (gregs + fpstate ptr + reserved[8] +
/// the 8-byte kernel sigmask), rounded so the frame stays laid out like Linux's.
const UCONTEXT_SIZE: u64 = 40 + (GREG_COUNT as u64 * 8) + 8 + 64 + 8;
const SIGINFO_SIZE: u64 = 128;

/// The `ucontext` size the signal frame reserves — exposed for the round-trip
/// test to locate the siginfo that follows it.
#[cfg(test)]
pub(super) fn signal_ucontext_size() -> u64 {
    UCONTEXT_SIZE
}

impl Kernel {
    /// The order the gregs are stored in `uc_mcontext`, expressed as guest
    /// register indices (`RAX=0`,`RCX=1`,…) — `REG_*` on x86-64. `RSP`/`RIP`/
    /// flags are handled separately by the caller.
    const GREG_TO_GPR: [usize; GREG_COUNT] = [
        8, 9, 10, 11, 12, 13, 14, 15, // r8..r15
        7, 6, 5, 3, 2, 0, 1, // rdi rsi rbp rbx rdx rax rcx
        4,  // rsp (index 15)
        0,  // rip (index 16) — placeholder, written from vcpu.pc()
        0,  // eflags (17)
        0, 0, 0, 0, 0, // csgsfs, err, trapno, oldmask, cr2
    ];

    /// Deliver a *synchronous* fault signal to the guest's handler, if one is
    /// installed: build the `rt_sigframe`, block the handler's mask, and point
    /// the vcpu at the handler. Returns `true` when the handler was set up (the
    /// caller resumes the guest into it); `false` when there is no catchable
    /// handler and the fault should stay fatal.
    ///
    /// This is what lets a JIT that deliberately faults — JSC/V8 use `SIGSEGV`
    /// for stack-limit and null checks and to poll for VM interrupts — run at
    /// all: without it every such trap is a hard crash.
    pub(super) fn deliver_fault_signal(
        &mut self,
        sig: u64,
        fault_addr: u64,
        vcpu: &mut dyn crate::vcpu::Vcpu,
        mem: &mut GuestMemory,
    ) -> bool {
        let act = self.cur.handlers[sig as usize];
        // Only a real, non-default, non-ignore handler is deliverable.
        if act.handler == SIG_DFL || act.handler == SIG_IGN {
            return false;
        }
        // A synchronous fault whose signal is already blocked — e.g. a second
        // fault *inside* the handler — is unrecoverable; Linux forces the
        // default action (terminate). This also stops an infinite deliver→
        // fault→deliver cascade when the handler itself faults.
        if self.cur.blocked & (1u64 << (sig - 1)) != 0 {
            return false;
        }

        // Choose the stack: the alternate stack if the handler asked for it and
        // one is configured, else just below the current rsp (with the ABI red
        // zone skipped).
        let cur_sp = vcpu.sp();
        let (alt_sp, alt_size, alt_flags) = self.cur.altstack;
        let base = if act.flags & SA_ONSTACK != 0 && alt_flags & SS_DISABLE == 0 && alt_size != 0 {
            alt_sp + alt_size
        } else {
            cur_sp - 128 // red zone
        };

        // Frame layout: reserve the whole frame, then 16-align so that at the
        // handler's first instruction rsp+8 is 16-aligned (as after a `call`).
        let frame_size = UC_OFF + UCONTEXT_SIZE + SIGINFO_SIZE;
        let frame = ((base - frame_size) & !15) - 8;

        // Saved register file → uc_mcontext.gregs.
        let mut wrote_ok = true;
        let mut put = |off: u64, v: u64| {
            wrote_ok &= mem.write(frame + off, &v.to_le_bytes()).is_ok();
        };
        put(0, act.restorer); // pretcode
        put(UC_OFF, 0); // uc_flags
        put(UC_OFF + 8, 0); // uc_link
        put(UC_OFF + 16, alt_sp); // uc_stack.ss_sp
        put(UC_OFF + 24, alt_flags); // ss_flags (+ padded size)
        put(UC_OFF + 32, alt_size); // ss_size
        for (i, &gpr) in Self::GREG_TO_GPR.iter().enumerate() {
            #[allow(clippy::match_same_arms)] // each greg is a distinct field that happens to share a value

            let v = match i {
                REG_RSP => cur_sp,
                REG_RIP => vcpu.pc(),
                REG_EFL => vcpu.rflags(),
                REG_CSGSFS => 0x0033, // CS=0x33 (user code); gs/fs 0
                19 => 0,              // err
                20 => if sig == SIGSEGV { 14 } else { 6 }, // trapno (#PF / #UD)
                21 => 0,              // oldmask
                22 => fault_addr,     // cr2 — the faulting address
                _ => vcpu.reg(gpr),
            };
            put(MCTX_OFF + (i as u64) * 8, v);
        }
        // uc_mcontext.fpstate pointer: none saved (0) — handlers that only
        // inspect the fault don't touch it.
        put(MCTX_OFF + (GREG_COUNT as u64) * 8, 0);
        put(UC_OFF + 296, self.cur.blocked); // uc_sigmask (kernel 8-byte)

        // siginfo: si_signo, si_errno, si_code, then si_addr for SIGSEGV/SIGILL.
        let si = frame + UC_OFF + UCONTEXT_SIZE;
        put(si - frame, sig & 0xffff_ffff); // si_signo (si_errno = 0)
        put(si - frame + 8, 1); // si_code = SI_KERNEL(0x80)? use 1 (SEGV_MAPERR)
        put(si - frame + 16, fault_addr); // si_addr

        if !wrote_ok {
            return false; // couldn't build the frame (guest stack unusable)
        }

        if std::env::var_os("NIXVM_SIGTRACE").is_some() {
            let hb = mem.read_vec(act.handler, 8).unwrap_or_default();
            eprintln!(
                "[sig] deliver sig={sig} fault={fault_addr:#x} pc={:#x} -> handler={:#x} restorer={:#x} frame={frame:#x} onstack={} handler_bytes={:02x?}",
                vcpu.pc(),
                act.handler,
                act.restorer,
                act.flags & SA_ONSTACK != 0,
                hb,
            );
        }
        // Enter the handler: SysV entry regs, masked signals, redirected pc/sp.
        vcpu.set_reg(7, sig); // rdi = signum
        vcpu.set_reg(6, si); // rsi = &siginfo
        vcpu.set_reg(2, frame + UC_OFF); // rdx = &ucontext
        vcpu.set_reg(0, 0); // rax cleared, per the SysV entry convention
        vcpu.set_sp(frame);
        vcpu.set_pc(act.handler);
        // Block this signal (unless SA_NODEFER) plus the handler's mask.
        self.cur.blocked |= act.mask | (1u64 << (sig - 1));
        true
    }

    /// `rt_sigreturn` — restore the context the handler was entered with. The
    /// frame is at `rsp - 8` (the handler's trampoline `ret`'d off `pretcode`),
    /// so `uc_mcontext` is at a fixed offset below the current `rsp`.
    pub(super) fn sys_rt_sigreturn(&mut self, vcpu: &mut dyn crate::vcpu::Vcpu, mem: &GuestMemory) {
        // On entry to the restorer, rsp pointed at pretcode; its `ret` popped 8,
        // so uc_mcontext is at rsp + (MCTX_OFF - 8).
        let mctx = vcpu.sp().wrapping_add(MCTX_OFF - 8);
        let read = |i: usize| mem.read_u64(mctx + (i as u64) * 8).unwrap_or(0);
        for (i, &gpr) in Self::GREG_TO_GPR.iter().enumerate() {
            match i {
                REG_RSP | REG_RIP | REG_EFL | REG_CSGSFS | 19 | 20 | 21 | 22 => {}
                _ => vcpu.set_reg(gpr, read(i)),
            }
        }
        vcpu.set_sp(read(REG_RSP));
        vcpu.set_rflags(read(REG_EFL));
        vcpu.set_pc(read(REG_RIP));
        // Restore the signal mask the handler ran under (uc_sigmask).
        let uc = mctx.wrapping_sub(MCTX_OFF - UC_OFF);
        if let Ok(mask) = mem.read_u64(uc + 296) {
            self.cur.blocked = mask;
        }
    }
}

/// Signals whose default disposition is to be ignored.
fn is_default_ignored(sig: u64) -> bool {
    const SIGCHLD: u64 = 17;
    const SIGCONT: u64 = 18;
    const SIGURG: u64 = 23;
    const SIGWINCH: u64 = 28;
    matches!(sig, SIGCHLD | SIGCONT | SIGURG | SIGWINCH)
}
