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
//! *Asynchronous* signals (from `kill`/`tgkill`/on-exit SIGCHLD) are now also
//! delivered to a real handler: [`Kernel::deliver_pending_signals`] runs at each
//! syscall boundary and, for the first deliverable pending signal with a
//! handler, calls [`Kernel::deliver_async_signal`] (which shares the frame
//! builder with the fault path). This is what lets a shell's `wait` — blocked in
//! [`Kernel::sys_rt_sigsuspend`] for SIGCHLD — wake, run its handler, and reap.
//! A signal left at its default disposition still takes the default action.

use super::{ExitCause, Kernel, QueuedSig, RunState, SA_NODEFER, SA_ONSTACK, SA_RESETHAND, SIGSEGV, SS_DISABLE, ServiceCtx, Shared, err, pgid_of};
use crate::abi::errno::Errno;
use crate::vcpu::GuestMemory;

/// The mode-specific `siginfo_t` fields [`Kernel::push_sigframe`] writes past
/// the common `si_signo`/`si_errno`/`si_code` header. A fault sets `addr`
/// (`si_addr`); an async/queued signal sets `pid`/`uid`/`value` (the `_rt`
/// union arm) — the two overlap in the C union, so only one is written.
#[derive(Clone, Copy, Default)]
struct SiFields {
    /// `si_code` (SI_USER=0, SI_QUEUE=-1, SEGV_MAPERR=1, …); low 32 bits used.
    code: u64,
    /// `si_addr` for a fault (SIGSEGV/SIGILL).
    addr: u64,
    /// `si_pid` — the sending process (async signals).
    pid: u64,
    /// `si_uid` — the sending user (async signals).
    uid: u64,
    /// `si_value` — the `sigqueue` payload (async signals).
    value: u64,
}

/// `SIG_DFL`: take the default action for the signal.
const SIG_DFL: u64 = 0;
/// `SIG_IGN`: ignore the signal.
const SIG_IGN: u64 = 1;

/// Highest supported signal number (`_NSIG - 1` on Linux).
pub(super) const NSIG: u64 = 64;
const SIGKILL: u64 = 9;
/// SIGCHLD — posted to a parent when a child stops or continues (to wake its
/// `wait4`/`waitid`), on top of the usual on-exit delivery.
const SIGCHLD: u64 = 17;
/// SIGCONT — resumes a stopped process (job control's "fg"/"bg").
const SIGCONT: u64 = 18;
const SIGSTOP: u64 = 19;
/// SIGTSTP/SIGTTIN/SIGTTOU — the *catchable* stop signals: they stop the target
/// only at their default disposition (a caught handler runs instead; SIG_IGN
/// drops them). SIGSTOP (19) always stops and can't be caught or ignored.
const SIGTSTP: u64 = 20;
const SIGTTIN: u64 = 21;
const SIGTTOU: u64 = 22;

/// The stop signals (`SIGSTOP`/`SIGTSTP`/`SIGTTIN`/`SIGTTOU`), whose default
/// action is to job-control-stop the target.
pub(super) fn is_stop_signal(sig: u64) -> bool {
    matches!(sig, SIGSTOP | SIGTSTP | SIGTTIN | SIGTTOU)
}

/// Bit-mask of the four stop signals' pending bits.
pub(super) const STOP_SIG_BITS: u64 =
    (1 << (SIGSTOP - 1)) | (1 << (SIGTSTP - 1)) | (1 << (SIGTTIN - 1)) | (1 << (SIGTTOU - 1));
/// The SIGCONT pending bit.
pub(super) const CONT_SIG_BIT: u64 = 1 << (SIGCONT - 1);
/// The SIGCHLD pending bit.
const CHLD_SIG_BIT: u64 = 1 << (SIGCHLD - 1);

impl Kernel {
    /// `rt_sigaction(sig, act, oldact, sigsetsize)` — store the disposition for
    /// `sig`. `sigsetsize` is accepted but ignored. Changing `SIGKILL`/`SIGSTOP`
    /// is rejected with `EINVAL`.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_rt_sigaction(
        &self, cx: &mut ServiceCtx,
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
            let old = cx.cur.handlers[idx];
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
            cx.cur.handlers[idx] = super::SigAction {
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
    #[allow(clippy::unused_self)]
    pub(super) fn sys_sigaltstack(&self, cx: &mut ServiceCtx, ss: u64, old_ss: u64, mem: &mut GuestMemory) -> i64 {
        let (sp, size, flags) = cx.cur.altstack;
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
            cx.cur.altstack = (new_sp, new_size, new_flags);
        }
        0
    }

    /// `rt_sigprocmask(how, set, oldset, sigsetsize)` — read/modify the blocked
    /// mask. `sigsetsize` is accepted but ignored.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_rt_sigprocmask(
        &self, cx: &mut ServiceCtx,
        how: u64,
        set: u64,
        oldset: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const SIG_BLOCK: u64 = 0;
        const SIG_UNBLOCK: u64 = 1;
        const SIG_SETMASK: u64 = 2;

        if oldset != 0 && mem.write(oldset, &cx.cur.blocked.to_le_bytes()).is_err() {
            return err(Errno::EFAULT);
        }
        if set != 0 {
            let Ok(mask) = mem.read_u64(set) else {
                return err(Errno::EFAULT);
            };
            match how {
                SIG_BLOCK => cx.cur.blocked |= mask,
                SIG_UNBLOCK => cx.cur.blocked &= !mask,
                SIG_SETMASK => cx.cur.blocked = mask,
                _ => return err(Errno::EINVAL),
            }
        }
        0
    }

    /// `rt_sigpending(set, sigsetsize)` — report the pending-signal mask.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_rt_sigpending(&self, cx: &mut ServiceCtx, set: u64, mem: &mut GuestMemory) -> i64 {
        if set != 0 && mem.write(set, &cx.cur.pending.to_le_bytes()).is_err() {
            return err(Errno::EFAULT);
        }
        0
    }

    /// `rt_sigsuspend(mask)` — atomically install `mask` as the blocked set and
    /// suspend until a signal not in it is delivered, then restore the pre-call
    /// mask and return `-EINTR`. This is how a shell's `wait` sleeps for SIGCHLD.
    ///
    /// We record the pre-call mask in `sigsuspend_prev` (first entry only — a
    /// parked suspend re-traps the same syscall), install the temporary mask, and
    /// either park (`cx.block`) when nothing is deliverable, or return `-EINTR`
    /// when a deliverable signal is already pending. On the `-EINTR` return the
    /// post-dispatch `deliver_pending_signals` delivers that signal: a real
    /// handler consumes `sigsuspend_prev` (restoring the pre-call mask as its
    /// `uc_sigmask`); an ignored one is cleaned up by that fn's post-loop restore.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_rt_sigsuspend(&self, cx: &mut ServiceCtx, mask_ptr: u64, mem: &GuestMemory) -> i64 {
        let Ok(new_mask) = mem.read_u64(mask_ptr) else {
            return err(Errno::EFAULT);
        };
        // First entry only: a parked suspend re-traps this same syscall, and its
        // pre-call mask was already saved — don't overwrite it with the temp mask.
        if cx.cur.sigsuspend_prev.is_none() {
            cx.cur.sigsuspend_prev = Some(cx.cur.blocked);
        }
        cx.cur.blocked = new_mask; // temporary mask for the suspend
        let deliverable = cx.cur.pending & !cx.cur.blocked;
        if deliverable == 0 {
            // Nothing to deliver: park. The re-trap after an unpark (e.g. the
            // child posting SIGCHLD) finds `deliverable != 0` and falls through.
            cx.block = true;
            return 0;
        }
        // A signal is deliverable — return -EINTR; the post-dispatch
        // `deliver_pending_signals` delivers it.
        err(Errno::EINTR)
    }

    /// `rt_sigtimedwait(set, info, timeout, sigsetsize)` — synchronously accept a
    /// pending signal in `set` *without* running its handler, the primitive behind
    /// `sigwait`/`sigwaitinfo`/`sigtimedwait`. The caller is expected to have
    /// blocked the signals in `set` (else they'd reach a handler first, staying out
    /// of `pending`); a blocked pending signal in `set` is dequeued and its number
    /// returned. With none pending it waits — until one arrives (re-trap after an
    /// unpark), until `timeout` elapses (`EAGAIN`), or until an unblocked caught
    /// signal interrupts it (`EINTR`, via the dispatcher). A zero `timeout` polls.
    pub(super) fn sys_rt_sigtimedwait(&self, cx: &mut ServiceCtx, set_ptr: u64, info: u64, timeout: u64, mem: &mut GuestMemory) -> i64 {
        let Ok(set) = mem.read_u64(set_ptr) else {
            return err(Errno::EFAULT);
        };
        // SIGKILL/SIGSTOP can't be waited for.
        let waitable = set & !((1 << (SIGKILL - 1)) | (1 << (SIGSTOP - 1)));
        // Dequeue the lowest pending signal in the set.
        let ready = cx.cur.pending & waitable;
        if ready != 0 {
            let sig = u64::from(ready.trailing_zeros()) + 1;
            cx.cur.pending &= !(1u64 << (sig - 1));
            cx.cur.wake_deadline = None;
            // Consume one queued instance so sigwaitinfo reports the same
            // si_code/si_value a handler would have seen. A real-time signal
            // dequeues FIFO; if more remain, keep its pending bit set.
            let (q, more) = cx.cur.take_siginfo(sig);
            if more {
                cx.cur.pending |= 1u64 << (sig - 1);
            }
            if info != 0 {
                let mut si = [0u8; 128];
                si[0..4].copy_from_slice(&(sig as i32).to_le_bytes()); // si_signo
                if let Some(q) = q {
                    si[8..12].copy_from_slice(&q.code.to_le_bytes()); // si_code
                    si[16..20].copy_from_slice(&q.pid.to_le_bytes()); // si_pid
                    si[20..24].copy_from_slice(&q.uid.to_le_bytes()); // si_uid
                    si[24..32].copy_from_slice(&q.value.to_le_bytes()); // si_value
                }
                let _ = mem.write(info, &si);
            }
            return sig as i64;
        }
        // Nothing pending: honor a finite/zero timeout via the scheduler's timed
        // wait (seeded once, reused across re-traps — like nanosleep).
        if timeout != 0 {
            let deadline = match cx.cur.wake_deadline {
                Some(dl) => dl,
                None => {
                    let (Ok(sec), Ok(nsec)) = (mem.read_u64(timeout), mem.read_u64(timeout + 8)) else {
                        return err(Errno::EFAULT);
                    };
                    if nsec >= 1_000_000_000 {
                        return err(Errno::EINVAL);
                    }
                    if sec == 0 && nsec == 0 {
                        return err(Errno::EAGAIN); // {0,0}: a non-blocking poll
                    }
                    let dl = super::poll::now_ns() + u128::from(sec) * 1_000_000_000 + u128::from(nsec);
                    cx.cur.wake_deadline = Some(dl);
                    dl
                }
            };
            if super::poll::now_ns() >= deadline {
                cx.cur.wake_deadline = None;
                return err(Errno::EAGAIN); // timed out
            }
        }
        cx.block = true;
        cx.restartable = false; // sigtimedwait is not restarted on SA_RESTART
        0
    }

    /// `kill(pid, sig)` — post `sig` to the target(s), with POSIX targeting:
    /// `pid > 0` a single process, `pid == 0` the caller's process group,
    /// `pid == -1` every process (sparing init), `pid < -1` process group `-pid`.
    /// `sig == 0` sends nothing but still reports `ESRCH` when no target exists.
    /// (`tkill`/`tgkill` reach here too — they always pass a positive tid, so only
    /// the single-target branch fires.)
    pub(super) fn sys_kill(&self, sh: &mut Shared, cx: &mut ServiceCtx, pid: i64, sig: u64) -> i64 {
        // A bare kill carries SI_USER (code 0) with the sender's pid.
        let sender = super::QueuedSig { code: 0, pid: cx.cur.pid, uid: 0, value: 0 };
        self.post_signal(sh, cx, pid, sig, sender)
    }

    /// The shared core of `kill`/`tkill`/`tgkill`/`rt_sigqueueinfo`: post `sig`
    /// (with its accompanying `info`) to the POSIX target(s). `sender`-vs-queued
    /// siginfo differs only in the `info` the caller supplies.
    pub(super) fn post_signal(&self, sh: &mut Shared, cx: &mut ServiceCtx, pid: i64, sig: u64, sender: super::QueuedSig) -> i64 {
        if sig > NSIG {
            return err(Errno::EINVAL);
        }
        let deliver = sig != 0; // sig == 0 is an existence/permission probe
        let bit = if deliver { 1u64 << (sig - 1) } else { 0 };
        let cur_pid = i64::from(cx.cur.pid);

        // Single process (or a tkill/tgkill tid): the common path.
        if pid > 0 {
            if pid == cur_pid {
                cx.cur.pending |= bit;
                if deliver {
                    cx.cur.post_siginfo(sig, sender);
                    // The current task is Running (it's executing this syscall),
                    // so this never resumes it, but it still applies the
                    // stop/cont bit cancellation.
                    if let Some(ppid) = apply_stop_cont(&mut cx.cur, sig) {
                        notify_parent(sh, cx, ppid);
                    }
                }
                return 0;
            }
            let mut resumed_ppid = None;
            let mut found = false;
            for slot in sh.procs.iter_mut().flatten() {
                if i64::from(slot.info.pid) == pid {
                    slot.info.pending |= bit;
                    slot.info.parked = false;
                    if deliver {
                        slot.info.post_siginfo(sig, sender);
                        resumed_ppid = apply_stop_cont(&mut slot.info, sig);
                    }
                    found = true;
                    break;
                }
            }
            if !found {
                return err(Errno::ESRCH);
            }
            if let Some(ppid) = resumed_ppid {
                notify_parent(sh, cx, ppid);
            }
            return 0;
        }

        // Group / broadcast. `None` = broadcast (pid == -1); otherwise the target
        // process group. The caller is out of `sh.procs` for its slice, so test it
        // separately.
        let target_pgrp = match pid {
            0 => Some(pgid_of(&cx.cur)), // the caller's own group
            -1 => None,                  // every process
            _ => Some((-pid) as i32),    // group -pid
        };
        let mut hit = false;
        let mut resumed_ppids: Vec<i32> = Vec::new();
        if target_pgrp.is_none_or(|pg| pgid_of(&cx.cur) == pg) {
            cx.cur.pending |= bit;
            if deliver {
                cx.cur.post_siginfo(sig, sender);
                if let Some(ppid) = apply_stop_cont(&mut cx.cur, sig) {
                    resumed_ppids.push(ppid);
                }
            }
            hit = true;
        }
        for slot in sh.procs.iter_mut().flatten() {
            let is_target = match target_pgrp {
                None => slot.info.pid != 1, // broadcast spares init (pid 1)
                Some(pg) => pgid_of(&slot.info) == pg,
            };
            if is_target {
                slot.info.pending |= bit;
                slot.info.parked = false;
                if deliver {
                    slot.info.post_siginfo(sig, sender);
                    if let Some(ppid) = apply_stop_cont(&mut slot.info, sig) {
                        resumed_ppids.push(ppid);
                    }
                }
                hit = true;
            }
        }
        for ppid in resumed_ppids {
            notify_parent(sh, cx, ppid);
        }
        if hit { 0 } else { err(Errno::ESRCH) }
    }

    /// `rt_sigqueueinfo(pid, sig, uinfo)` / `rt_tgsigqueueinfo(tgid, tid, sig,
    /// uinfo)` — like `kill`, but carries the sender's `siginfo_t` so an
    /// `SA_SIGINFO` handler (or `sigwaitinfo`) sees the real `si_code`/`si_value`
    /// (`sigqueue`'s payload). Delivery/targeting reuse [`Self::sys_kill`]; this
    /// just records the accompanying info for the target's pending signal.
    pub(super) fn sys_rt_sigqueueinfo(
        &self, sh: &mut Shared, cx: &mut ServiceCtx,
        pid: i64,
        sig: u64,
        uinfo: u64,
        mem: &GuestMemory,
    ) -> i64 {
        // Read the guest's siginfo: si_code (offset 8) and the sigqueue value
        // (offset 24, the 8-byte `_rt` union). si_pid/si_uid are the sender's.
        let code = mem.read_u32(uinfo + 8).map_or(-1, |v| v as i32);
        let value = mem.read_u64(uinfo + 24).unwrap_or(0);
        let info = QueuedSig { code, pid: cx.cur.pid, uid: 0, value };
        // Post with the caller's siginfo directly (a real-time signal thereby
        // queues this exact value; a standard one records it).
        self.post_signal(sh, cx, pid, sig, info)
    }

    /// The signal (`1..=NSIG`) whose *real handler* [`Self::deliver_pending_signals`]
    /// will run at this syscall boundary, or `None` when the first actionable
    /// pending signal isn't a caught one. Used by the syscall dispatcher to decide
    /// whether a blocking syscall must be interrupted (a handler exists to run) and
    /// how (`SA_RESTART` restart vs `EINTR`). The scan mirrors `deliver_pending_signals`
    /// exactly: `SIG_IGN` and default-ignored signals are skipped (they don't
    /// interrupt), and a default-*terminate* signal returns `None` (the process is
    /// about to die — the dispatcher's `Zombie` check handles it, not the block path).
    pub(super) fn first_handled_signal(&self, cx: &ServiceCtx) -> Option<usize> {
        let deliverable = cx.cur.pending & !cx.cur.blocked;
        for sig in 1..=NSIG {
            if deliverable & (1u64 << (sig - 1)) == 0 {
                continue;
            }
            match cx.cur.handlers[sig as usize].handler {
                SIG_IGN => {}                              // dropped — no interrupt
                h if h != SIG_DFL => return Some(sig as usize), // real handler runs
                _ if is_default_ignored(sig) => {}         // dropped — no interrupt
                _ => return None,                          // default terminate/stop
            }
        }
        None
    }

    /// Act on the first deliverable pending signal for the current process. Runs
    /// once after each serviced syscall. Unlike before, a signal with a real
    /// handler is now *delivered* (an `rt_sigframe` is pushed and the vcpu points
    /// at the handler) — so `kill`/`tgkill`/on-exit-SIGCHLD reach their handler,
    /// which is what lets a shell's `wait` (blocked in `sigsuspend` for SIGCHLD)
    /// wake and reap. At most one handler is entered per call: the guest must run
    /// it (and `rt_sigreturn`) before the next pending signal is considered, so
    /// any others deliver at the following syscall boundary.
    pub(super) fn deliver_pending_signals(
        &self, cx: &mut ServiceCtx,
        vcpu: &mut dyn crate::vcpu::Vcpu,
        mem: &mut GuestMemory,
    ) -> bool {
        if !matches!(cx.cur.run, RunState::Running) {
            return false;
        }
        let deliverable = cx.cur.pending & !cx.cur.blocked;
        for sig in 1..=NSIG {
            let bit = 1u64 << (sig - 1);
            if deliverable & bit == 0 {
                continue;
            }
            // Clear the pending bit: every branch below acts on this signal.
            cx.cur.pending &= !bit;
            // Job-control stop: SIGSTOP always stops (uncatchable), and the
            // other stop signals stop only at their default disposition. A
            // caught stop signal falls through to run its handler; SIG_IGN
            // falls through and is dropped. Stopping parks the task and notifies
            // the parent so its `wait4(WUNTRACED)`/`waitid(WSTOPPED)` wakes.
            if sig == SIGSTOP
                || (is_stop_signal(sig) && cx.cur.handlers[sig as usize].handler == SIG_DFL)
            {
                cx.cur.run = RunState::Stopped(sig as i32);
                cx.cur.stop_reported = false;
                // A pending SIGCONT is cancelled by an actual stop.
                cx.cur.pending &= !CONT_SIG_BIT;
                self.notify_parent_of_child_event(cx.cur.ppid);
                return false;
            }
            match cx.cur.handlers[sig as usize].handler {
                // Ignored explicitly: drop it (and any queued RT instances).
                SIG_IGN => cx.cur.drain_rt(sig),
                // A real handler: push the frame + redirect the vcpu, then STOP —
                // the guest runs the handler now; any remaining pendings deliver
                // at the next syscall boundary. Returns whether the frame built.
                h if h != SIG_DFL => {
                    return self.deliver_async_signal(cx, sig, vcpu, mem);
                }
                // SIG_DFL: ignore the "ignored-by-default" set, else terminate.
                _ if is_default_ignored(sig) => cx.cur.drain_rt(sig),
                _ => {
                    cx.cur.run = RunState::Zombie(ExitCause::Signaled(sig as i32));
                    return false;
                }
            }
        }
        // No handler was entered and we didn't zombie. If a `sigsuspend` woke on
        // an ignored signal, its temporary mask is still installed and its
        // pre-call mask un-restored — restore it now (when a handler WAS entered,
        // `deliver_async_signal` already took `sigsuspend_prev` as the restore
        // mask, so this is skipped).
        if let Some(prev) = cx.cur.sigsuspend_prev.take() {
            cx.cur.blocked = prev;
        }
        false
    }

    /// Notify a parent that a child changed job-control state (stopped or
    /// continued): post SIGCHLD and unpark it, so a `wait4`/`waitid` blocked for
    /// the event wakes and reports it. Called from
    /// [`Self::deliver_pending_signals`], which runs with **no kernel lock held**
    /// (see `service` in `mod.rs`), so it takes `self.shared` itself. The child
    /// itself is checked out into `cx.cur` and never in `sh.procs`, so this only
    /// scans the table for the parent — no aliasing with the current task.
    fn notify_parent_of_child_event(&self, ppid: i32) {
        let mut sh = self.shared.lock().unwrap();
        for slot in sh.procs.iter_mut().flatten() {
            if slot.info.pid == ppid {
                slot.info.pending |= CHLD_SIG_BIT;
                slot.info.parked = false;
                break;
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
        &self, cx: &mut ServiceCtx,
        sig: u64,
        fault_addr: u64,
        vcpu: &mut dyn crate::vcpu::Vcpu,
        mem: &mut GuestMemory,
    ) -> bool {
        // Debug escape hatch: skip delivery so a fault is fatal and the kernel
        // dumps its context (used to inspect a stack overflow that a guest
        // handler would otherwise catch and hide).
        if std::env::var_os("NIXVM_NOSIG").is_some() {
            return false;
        }
        let act = cx.cur.handlers[sig as usize];
        // Only a real, non-default, non-ignore handler is deliverable.
        if act.handler == SIG_DFL || act.handler == SIG_IGN {
            return false;
        }
        // A synchronous fault whose signal is already blocked — e.g. a second
        // fault *inside* the handler — is unrecoverable; Linux forces the
        // default action (terminate). This also stops an infinite deliver→
        // fault→deliver cascade when the handler itself faults.
        if cx.cur.blocked & (1u64 << (sig - 1)) != 0 {
            return false;
        }
        // trapno #PF(14)/#UD(6), si_code SEGV_MAPERR(1), si_addr = fault_addr,
        // and the handler's uc_sigmask is the *current* blocked mask (restored
        // by rt_sigreturn) — the fault path's original behavior, unchanged.
        let si = SiFields { code: 1, addr: fault_addr, ..SiFields::default() }; // SEGV_MAPERR
        self.push_sigframe(cx, sig, if sig == SIGSEGV { 14 } else { 6 }, si, cx.cur.blocked, vcpu, mem)
    }

    /// Deliver an *asynchronous* signal (posted by `kill`/`tgkill`/on-exit
    /// SIGCHLD) to the guest's handler. Returns `false` when there is no real
    /// handler (SIG_DFL/SIG_IGN) so the caller falls back to the default action.
    ///
    /// Unlike a fault, an async signal carries no faulting address; `trapno`,
    /// `si_code` (SI_USER = 0), and `si_addr` are all 0. When a `sigsuspend` is
    /// in progress its saved pre-call mask is the mask to restore on
    /// `rt_sigreturn`; otherwise the current blocked mask is restored.
    pub(super) fn deliver_async_signal(
        &self, cx: &mut ServiceCtx,
        sig: u64,
        vcpu: &mut dyn crate::vcpu::Vcpu,
        mem: &mut GuestMemory,
    ) -> bool {
        let act = cx.cur.handlers[sig as usize];
        if act.handler == SIG_DFL || act.handler == SIG_IGN {
            return false;
        }
        // A sigsuspend restores its pre-call mask when the handler returns; take
        // it so it isn't double-restored by `deliver_pending_signals`.
        let restore = cx.cur.sigsuspend_prev.take().unwrap_or(cx.cur.blocked);
        // Async delivery happens at a syscall boundary: the (KVM) vcpu is parked
        // at CPL0 on the return trampoline. Collapse that to the logical user
        // state first, so the frame saves the real user rip/rflags (not the
        // trampoline's `sysretq`) and the handler runs at CPL3, not supervisor.
        vcpu.settle_syscall_return();
        // Restarting an `SA_RESTART`-interrupted blocking syscall: rewind the saved
        // rip to the 2-byte `syscall` instruction so the handler's `rt_sigreturn`
        // re-executes it. `rcx` holds the syscall's return address on both x86-64
        // backends (KVM's `settle` set rip←rcx just above; the interpreter set rcx
        // at `syscall` entry), so the instruction is at `rcx − 2`. RAX still holds
        // the syscall number — a restarted syscall skips `set_syscall_ret`. (This,
        // like the whole `rt_sigframe` path, is x86-64-specific.)
        if cx.restart_syscall {
            let syscall_pc = vcpu.reg(1).wrapping_sub(2); // rcx − len(`syscall`)
            vcpu.set_pc(syscall_pc);
        }
        // Carry the siginfo the sender queued (sigqueue's si_value/si_code, or a
        // sender pid); a bare `kill`/on-exit SIGCHLD leaves it SI_USER (all 0).
        // A real-time signal dequeues one instance FIFO; if more remain queued,
        // re-arm its pending bit so the next boundary delivers the next one.
        let (info, more) = cx.cur.take_siginfo(sig);
        if more {
            cx.cur.pending |= 1u64 << (sig - 1);
        }
        let si = info.map_or(SiFields::default(), |q| SiFields {
            code: q.code as u32 as u64,
            addr: 0,
            pid: u64::from(q.pid as u32),
            uid: u64::from(q.uid),
            value: q.value,
        });
        self.push_sigframe(cx, sig, 0, si, restore, vcpu, mem)
    }

    /// Build the x86-64 `rt_sigframe` for `sig` on the (alternate or interrupted)
    /// stack, block the handler's mask, and point the vcpu at the handler. Shared
    /// by fault and async delivery; the two differ only in `trapno`/`si_code`/
    /// `si_addr` (the `uc_mcontext` #PF fields and siginfo) and in `restore_mask`
    /// (the `uc_sigmask` a later `rt_sigreturn` restores). Returns `true` when the
    /// frame was built and the vcpu redirected.
    #[allow(clippy::unused_self, clippy::too_many_arguments)]
    fn push_sigframe(
        &self, cx: &mut ServiceCtx,
        sig: u64,
        trapno: u64,
        si: SiFields,
        restore_mask: u64,
        vcpu: &mut dyn crate::vcpu::Vcpu,
        mem: &mut GuestMemory,
    ) -> bool {
        let si_code = si.code;
        let si_addr = si.addr;
        let act = cx.cur.handlers[sig as usize];
        // Choose the stack: the alternate stack if the handler asked for it and
        // one is configured, else just below the current rsp (with the ABI red
        // zone skipped).
        let cur_sp = vcpu.sp();
        let (alt_sp, alt_size, alt_flags) = cx.cur.altstack;
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
                20 => trapno,         // trapno (#PF / #UD; 0 for async)
                21 => 0,              // oldmask
                22 => si_addr,        // cr2 — the faulting address (0 for async)
                _ => vcpu.reg(gpr),
            };
            put(MCTX_OFF + (i as u64) * 8, v);
        }
        // uc_mcontext.fpstate pointer: none saved (0) — handlers that only
        // inspect the fault don't touch it.
        put(MCTX_OFF + (GREG_COUNT as u64) * 8, 0);
        put(UC_OFF + 296, restore_mask); // uc_sigmask (kernel 8-byte)

        // siginfo: si_signo, si_errno, si_code, then the mode-specific union at
        // offset 16. A fault carries si_addr there (SIGSEGV/SIGILL); an async
        // signal carries the sending pid/uid and the sigqueue value instead
        // (the `_sigfault` and `_rt` arms of the `_sifields` union overlap).
        let si_base = frame + UC_OFF + UCONTEXT_SIZE;
        put(si_base - frame, sig & 0xffff_ffff); // si_signo (si_errno = 0)
        put(si_base - frame + 8, si_code & 0xffff_ffff); // si_code
        if trapno != 0 {
            put(si_base - frame + 16, si_addr); // _sigfault: si_addr
        } else {
            // _rt: si_pid @16, si_uid @20, si_value @24 (8-byte union).
            put(si_base - frame + 16, (si.pid & 0xffff_ffff) | (si.uid << 32));
            put(si_base - frame + 24, si.value);
        }

        if !wrote_ok {
            return false; // couldn't build the frame (guest stack unusable)
        }

        if std::env::var_os("NIXVM_SIGTRACE").is_some() {
            let hb = mem.read_vec(act.handler, 8).unwrap_or_default();
            eprintln!(
                "[sig] deliver sig={sig} fault={si_addr:#x} pc={:#x} -> handler={:#x} restorer={:#x} frame={frame:#x} onstack={} handler_bytes={:02x?}",
                vcpu.pc(),
                act.handler,
                act.restorer,
                act.flags & SA_ONSTACK != 0,
                hb,
            );
        }
        // Enter the handler: SysV entry regs, masked signals, redirected pc/sp.
        vcpu.set_reg(7, sig); // rdi = signum
        vcpu.set_reg(6, si_base); // rsi = &siginfo
        vcpu.set_reg(2, frame + UC_OFF); // rdx = &ucontext
        vcpu.set_reg(0, 0); // rax cleared, per the SysV entry convention
        vcpu.set_sp(frame);
        vcpu.set_pc(act.handler);
        // Block the handler's mask, and this signal too *unless* SA_NODEFER —
        // without honoring SA_NODEFER a handler that re-raises its own signal
        // could never re-enter, and (with our redelivery loop) deadlocks.
        cx.cur.blocked |= act.mask;
        if act.flags & SA_NODEFER == 0 {
            cx.cur.blocked |= 1u64 << (sig - 1);
        }
        // SA_RESETHAND (one-shot): reset the disposition to SIG_DFL on entry, so
        // a second delivery takes the default action (classic `signal()`).
        if act.flags & SA_RESETHAND != 0 {
            cx.cur.handlers[sig as usize] = super::SigAction::default();
        }
        true
    }

    /// `rt_sigreturn` — restore the context the handler was entered with. The
    /// frame is at `rsp - 8` (the handler's trampoline `ret`'d off `pretcode`),
    /// so `uc_mcontext` is at a fixed offset below the current `rsp`.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_rt_sigreturn(&self, cx: &mut ServiceCtx, vcpu: &mut dyn crate::vcpu::Vcpu, mem: &GuestMemory) {
        // `rt_sigreturn` arrives as a syscall: the (KVM) vcpu is parked at CPL0 on
        // the return trampoline. Collapse it to CPL3 user mode now; the explicit
        // rip/sp/rflags restores below then overwrite the rip/rflags this set from
        // `rcx`/`r11` with the handler's saved context, leaving the guest resuming
        // the interrupted user code at user privilege rather than supervisor.
        vcpu.settle_syscall_return();
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
            cx.cur.blocked = mask;
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

/// Apply the *job-control* side effects of posting `sig` to `info`, on top of
/// the pending bit the caller already set. Posting SIGCONT cancels any pending
/// (not-yet-delivered) stop and, if the task is `Stopped`, resumes it — flipping
/// it back to `Running`, latching a "continued" event for its parent's
/// `wait4(WCONTINUED)`, and unparking it. Posting a stop signal conversely
/// cancels a pending SIGCONT (they annihilate). Returns `Some(ppid)` when the
/// task was resumed from `Stopped`, so the caller can notify that parent.
fn apply_stop_cont(info: &mut super::ProcInfo, sig: u64) -> Option<i32> {
    if sig == SIGCONT {
        info.pending &= !STOP_SIG_BITS; // a pending stop is cancelled by SIGCONT
        if matches!(info.run, RunState::Stopped(_)) {
            info.run = RunState::Running;
            info.continued = true;
            info.stop_reported = false;
            info.parked = false;
            return Some(info.ppid);
        }
    } else if is_stop_signal(sig) {
        info.pending &= !CONT_SIG_BIT; // a pending SIGCONT is cancelled by a stop
    }
    None
}

/// Post SIGCHLD to the parent `ppid` and unpark it (so its `wait4`/`waitid`
/// wakes to report a child's stop/continue). Used by [`Kernel::post_signal`],
/// which already holds the kernel lock (`sh`) and has the caller checked out
/// into `cx.cur` — the parent may be either, so both are searched.
fn notify_parent(sh: &mut Shared, cx: &mut ServiceCtx, ppid: i32) {
    if cx.cur.pid == ppid {
        cx.cur.pending |= CHLD_SIG_BIT;
        return;
    }
    for slot in sh.procs.iter_mut().flatten() {
        if slot.info.pid == ppid {
            slot.info.pending |= CHLD_SIG_BIT;
            slot.info.parked = false;
            return;
        }
    }
}
