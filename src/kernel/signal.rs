//! Signal handling: per-process handler tables, masks, `kill`/`tgkill`
//! delivery, and the DEFAULT dispositions.
//!
//! Custom-handler invocation (pushing a signal frame onto the guest stack and
//! diverting the PC) is out of scope: handlers are *stored* by `rt_sigaction`
//! but delivery only performs default actions — TERMINATE or IGNORE. A pending
//! signal whose disposition is a real handler address is cleared (rather than
//! left set) so the scheduler never spins on an undeliverable signal.

use super::{Kernel, RunState, err};
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
            let mut buf = [0u8; 32];
            buf[0..8].copy_from_slice(&self.cur.handlers[idx].to_le_bytes());
            if mem.write(oldact, &buf).is_err() {
                return err(Errno::EFAULT);
            }
        }
        if act != 0 {
            let Ok(handler) = mem.read_u64(act) else {
                return err(Errno::EFAULT);
            };
            self.cur.handlers[idx] = handler;
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
            match self.cur.handlers[sig as usize] {
                SIG_IGN => {}
                // A real handler address: we cannot push a frame yet, so drop
                // the signal to avoid deadlocking the scheduler.
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

/// Signals whose default disposition is to be ignored.
fn is_default_ignored(sig: u64) -> bool {
    const SIGCHLD: u64 = 17;
    const SIGCONT: u64 = 18;
    const SIGURG: u64 = 23;
    const SIGWINCH: u64 = 28;
    matches!(sig, SIGCHLD | SIGCONT | SIGURG | SIGWINCH)
}
