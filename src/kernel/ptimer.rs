//! POSIX per-process timers: `timer_create`, `timer_settime`, `timer_gettime`,
//! `timer_getoverrun`, `timer_delete`.
//!
//! Each timer lives on the task its expirations are delivered to (the creating
//! task for `SIGEV_SIGNAL`, the named thread for `SIGEV_THREAD_ID`), next to the
//! `ITIMER_REAL` state it generalizes: an absolute wall-clock deadline that the
//! schedulers already wake parked tasks for ([`super::ProcInfo::timer_deadline`]),
//! fired from the same per-step hook as `alarm` ([`fire_timers_if_due`]). Any
//! thread of the group may operate on a timer by id (Linux timers are
//! per-process), so lookups search the caller and then its thread-group
//! siblings.
//!
//! Delivery follows Linux's queueing rule: an expiration posts the timer's
//! signal with `si_code = SI_TIMER`, `si_timerid`, `si_overrun` and the
//! `sigev_value`; while that signal is still pending, further expirations don't
//! queue another one but count as *overruns* (reported by `timer_getoverrun`
//! and in the next delivery's `si_overrun`). `SIGEV_NONE` timers never signal —
//! `timer_gettime` is their whole interface. musl's `SIGEV_THREAD` emulation
//! (a helper thread `sigwaitinfo`-ing `SIGEV_THREAD_ID` deliveries) and glibc's
//! work unchanged on top.
//!
//! A timer on a CPU clock (`CLOCK_PROCESS_CPUTIME_ID`, `CLOCK_THREAD_CPUTIME_ID`
//! or one by pid) expires on that clock's CPU time instead: it is queued on
//! the clock's [`cputimer::CpuTimers`], which count its expirations as the
//! scheduler charges CPU time, and the task holding it signals them here.

use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;

use super::time::CpuOf;
use super::{
    Kernel, ProcInfo, QueuedSig, RLimits, RunState, ServiceCtx, Shared, cputimer, err, poll,
};
use crate::abi::errno::Errno;
use crate::vcpu::GuestMemory;

/// `si_code` of a timer expiration signal.
const SI_TIMER: i32 = -2;
const SIGEV_SIGNAL: i32 = 0;
const SIGEV_NONE: i32 = 1;
const SIGEV_THREAD_ID: i32 = 4;
const SIGALRM: u32 = 14;
const TIMER_ABSTIME: u64 = 1;
const NS: u128 = 1_000_000_000;

/// One POSIX timer.
#[derive(Clone, Debug)]
pub(super) struct PosixTimer {
    /// The id `timer_create` returned (unique within the thread group).
    pub(super) id: i32,
    /// The thread group that owns the timer (ids are per-process).
    pub(super) tgid: i32,
    /// `CLOCK_REALTIME` timers re-anchor on a wall-clock jump on Linux; with no
    /// settable clock here that never happens, but the clock still selects how
    /// an absolute `it_value` converts to a wall deadline.
    clock: i64,
    /// `SIGEV_SIGNAL`/`SIGEV_NONE`/`SIGEV_THREAD_ID`.
    notify: i32,
    signo: u32,
    /// `sigev_value`, echoed as `si_value`.
    value: u64,
    /// Next expiration, as an absolute wall-clock time (ns since the epoch).
    pub(super) deadline: Option<u128>,
    interval_ns: u128,
    /// Overruns reported by `timer_getoverrun`: the expirations folded into
    /// the most recently delivered signal beyond the first.
    overrun_last: i32,
    /// Expirations that occurred while this timer's signal was still pending.
    overrun_pending: i32,
    /// On a CPU clock: where it is queued. `deadline` is then unused.
    cpu: Option<CpuTimer>,
    /// A `SIGEV_THREAD_ID` timer whose thread has exited: it expires, but
    /// signals no one (Linux finds no task to send to).
    orphaned: bool,
}

/// A POSIX timer's place on a CPU clock.
#[derive(Clone, Debug)]
struct CpuTimer {
    /// Whose CPU time it counts.
    of: CpuOf,
    /// That process's limits and CPU-time timers, where it is queued.
    base: RLimits,
    link: Arc<cputimer::Link>,
}

impl PosixTimer {
    /// What `timer_settime` and the timer's expiries change.
    fn setting(&self) -> (Option<u128>, u128, i32, i32) {
        (
            self.deadline,
            self.interval_ns,
            self.overrun_last,
            self.overrun_pending,
        )
    }

    /// The timer is deleted (or its process gone): take it off its CPU clock.
    pub(super) fn forget(self) {
        if let Some(c) = self.cpu {
            c.base.cpu_timers().delete(&c.link);
        }
    }
}

impl CpuTimer {
    fn target(&self) -> cputimer::Target {
        match self.of {
            CpuOf::Thread(tid) => cputimer::Target::Thread(tid),
            CpuOf::Group(_) => cputimer::Target::Group,
        }
    }
}

/// Whether `info` still has `t`'s expiration signal queued (undelivered).
fn signal_queued(info: &ProcInfo, t: &PosixTimer) -> bool {
    let sig = u64::from(t.signo);
    if info.pending & (1u64 << (sig - 1)) == 0 {
        return false;
    }
    let mine = |q: &QueuedSig| q.code == SI_TIMER && q.pid == t.id;
    if sig >= super::SIGRTMIN {
        info.rt_queue
            .get(&t.signo)
            .is_some_and(|q| q.iter().any(mine))
    } else {
        info.queued_siginfo[sig as usize].as_ref().is_some_and(mine)
    }
}

/// Fire every POSIX timer on `info` whose deadline has passed at `now` (wall
/// ns), and signal the expirations a CPU clock counted: post its signal (or
/// count an overrun if one is still pending), re-arm a periodic timer past
/// `now`, disarm a one-shot. Un-parks the task when a signal was posted so a
/// blocking syscall wakes to take it.
pub(super) fn fire_ptimers(info: &mut ProcInfo, now: u128) {
    for i in 0..info.ptimers.len() {
        let t = &mut info.ptimers[i];
        let count = if let Some(c) = &t.cpu {
            match c.link.fired.swap(0, Relaxed) {
                0 => continue,
                n => i32::try_from(n).unwrap_or(i32::MAX),
            }
        } else {
            let Some(dl) = t.deadline else {
                continue;
            };
            if now < dl {
                continue;
            }
            // How many expirations elapsed (a periodic timer may have missed
            // some while the task was descheduled).
            let (count, next) = match (now - dl).checked_div(t.interval_ns) {
                Some(missed) => (missed + 1, Some(dl + (missed + 1) * t.interval_ns)),
                None => (1, None), // one-shot (interval 0)
            };
            t.deadline = next;
            i32::try_from(count).unwrap_or(i32::MAX)
        };
        if t.notify != SIGEV_NONE && !t.orphaned {
            signal_expiry(info, i, count);
        }
    }
}

/// Signal `count` expirations of `info.ptimers[i]`: one signal, the rest
/// overruns, or all overruns while its last signal is still pending.
fn signal_expiry(info: &mut ProcInfo, i: usize, count: i32) {
    let queued = signal_queued(info, &info.ptimers[i]);
    let t = &mut info.ptimers[i];
    if queued {
        t.overrun_pending = t.overrun_pending.saturating_add(count);
        return;
    }
    let overrun = t.overrun_pending.saturating_add(count - 1);
    t.overrun_pending = 0;
    let (sig, qs) = (
        u64::from(t.signo),
        QueuedSig {
            code: SI_TIMER,
            errno: 0,
            pid: t.id,
            uid: overrun as u32,
            value: t.value,
        },
    );
    if info.raise_with(sig, qs) {
        info.parked = false;
    }
}

/// Task `cx` is exiting. The timers it holds are its process's, so they go
/// to another of its threads (Linux deletes them only with the process, as
/// `exit_itimers`); a `SIGEV_THREAD_ID` one, whose thread this was, signals
/// no more. With no thread left, they go.
pub(super) fn exit_timers(sh: &mut Shared, cx: &mut ServiceCtx) {
    let timers = std::mem::take(&mut cx.cur.ptimers);
    if timers.is_empty() {
        return;
    }
    let (pid, tgid) = (cx.cur.pid, cx.cur.tgid);
    let heir = sh.procs.iter_mut().flatten().find(|p| {
        p.info.tgid == tgid && p.info.pid != pid && !matches!(p.info.run, RunState::Zombie(_))
    });
    let Some(heir) = heir else {
        for t in timers {
            t.forget();
        }
        return;
    };
    for mut t in timers {
        t.orphaned |= t.notify == SIGEV_THREAD_ID;
        if let Some(c) = &t.cpu {
            c.link.set_owner(heir.info.pid);
        }
        heir.info.ptimers.push(t);
    }
}

/// Task `mine` ran a step on another CPU while others saw a copy of it
/// (the SMP stand-in, as it was `before`): carry over what they did to its
/// timers there — created one for it (`SIGEV_THREAD_ID`) or handed it an
/// exited thread's, deleted one, or set one.
pub(super) fn fold_stand_in(
    mine: &mut Vec<PosixTimer>,
    before: &[PosixTimer],
    stand_in: &[PosixTimer],
) {
    let had = |list: &[PosixTimer], id: i32| list.iter().any(|t| t.id == id);
    mine.retain(|t| !had(before, t.id) || had(stand_in, t.id));
    for s in stand_in {
        match before.iter().find(|b| b.id == s.id) {
            None if !had(mine, s.id) => mine.push(s.clone()),
            Some(b) if b.setting() != s.setting() => {
                if let Some(m) = mine.iter_mut().find(|m| m.id == s.id) {
                    m.deadline = s.deadline;
                    m.interval_ns = s.interval_ns;
                    m.overrun_last = s.overrun_last;
                    m.overrun_pending = s.overrun_pending;
                }
            }
            _ => {}
        }
    }
}

/// A timer's signal `q` is being taken (delivered to a handler, or
/// dequeued by `sigwaitinfo` and the like): as Linux's
/// `posixtimer_deliver_signal`, it carries the expirations that came while it
/// was pending as `si_overrun`, which is then what `timer_getoverrun` says.
pub(super) fn delivered(info: &mut ProcInfo, q: &mut QueuedSig) {
    if q.code != SI_TIMER {
        return;
    }
    if let Some(t) = info.ptimers.iter_mut().find(|t| t.id == q.pid) {
        let overrun = (q.uid as i32).saturating_add(t.overrun_pending);
        t.overrun_last = overrun;
        t.overrun_pending = 0;
        q.uid = overrun as u32;
    }
}

/// Encode `ns` as a `struct timespec` into `b[off..off + 16]`.
fn put_ts(b: &mut [u8], off: usize, ns: u128) {
    b[off..off + 8].copy_from_slice(&((ns / NS) as i64).to_le_bytes());
    b[off + 8..off + 16].copy_from_slice(&((ns % NS) as i64).to_le_bytes());
}

/// A `struct itimerspec` (`it_interval`, then `it_value`) holding `s`.
fn itimerspec(s: cputimer::Setting) -> [u8; 32] {
    let mut b = [0u8; 32];
    put_ts(&mut b, 0, u128::from(s.interval));
    put_ts(&mut b, 16, u128::from(s.value));
    b
}

/// Decode the `struct timespec` at `addr`: `EFAULT` if unreadable, `EINVAL`
/// for a negative second or an out-of-range nanosecond field.
fn read_ts(mem: &GuestMemory, addr: u64) -> Result<u128, i64> {
    let (Ok(s), Ok(n)) = (mem.read_u64(addr), mem.read_u64(addr + 8)) else {
        return Err(err(Errno::EFAULT));
    };
    let (s, n) = (s as i64, n as i64);
    if s < 0 || !(0..1_000_000_000).contains(&n) {
        return Err(err(Errno::EINVAL));
    }
    Ok(s as u128 * NS + n as u128)
}

impl Kernel {
    /// Find timer `id` of the caller's thread group: on the caller itself, or
    /// on a sibling thread it was delivered to. Returns `None` if no such timer.
    fn ptimer_mut<'a>(
        sh: &'a mut Shared,
        cx: &'a mut ServiceCtx,
        id: i32,
    ) -> Option<&'a mut PosixTimer> {
        let tgid = cx.cur.tgid;
        if let Some(t) = cx.cur.ptimers.iter_mut().find(|t| t.id == id) {
            return Some(t);
        }
        sh.procs
            .iter_mut()
            .flatten()
            .filter(|p| p.info.tgid == tgid && !matches!(p.info.run, RunState::Zombie(_)))
            .flat_map(|p| p.info.ptimers.iter_mut())
            .find(|t| t.id == id && t.tgid == tgid)
    }

    /// `timer_create(clockid, struct sigevent *sevp, timer_t *timerid)`.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_timer_create(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        clock: u64,
        sevp: u64,
        out: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        // The sigevent is read first, then the clock checked: EINVAL for an
        // id Linux doesn't have, EOPNOTSUPP for MONOTONIC_RAW, the COARSE
        // clocks and a clock device, which have no timers.
        let sev = if sevp == 0 {
            None
        } else {
            match mem.read_vec(sevp, 20) {
                Ok(raw) => Some(raw),
                Err(_) => return err(Errno::EFAULT),
            }
        };
        let kind = match Self::clock_for(clock, true) {
            Ok(c) => c,
            Err(e) => return err(e),
        };
        let clock = i64::from(clock as i32);
        let tgid = cx.cur.tgid;
        // Ids are per thread group: the lowest one not in use.
        let mut used: Vec<i32> = cx.cur.ptimers.iter().map(|t| t.id).collect();
        used.extend(
            sh.procs
                .iter()
                .flatten()
                .filter(|p| p.info.tgid == tgid)
                .flat_map(|p| p.info.ptimers.iter().map(|t| t.id)),
        );
        let id = (0..i32::MAX).find(|i| !used.contains(i)).unwrap_or(0);
        // struct sigevent: sigev_value@0 (8), sigev_signo@8, sigev_notify@12,
        // sigev_notify_thread_id@16.
        let (value, signo, notify, tid) = if let Some(raw) = sev {
            let w32 = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
            (
                u64::from_le_bytes(raw[0..8].try_into().unwrap()),
                w32(8),
                w32(12) as i32,
                w32(16) as i32,
            )
        } else {
            // NULL sevp: SIGEV_SIGNAL with SIGALRM and the timer id as value.
            (id as u64, SIGALRM, SIGEV_SIGNAL, 0)
        };
        match notify {
            SIGEV_NONE => {}
            SIGEV_SIGNAL | SIGEV_THREAD_ID if (1..=64).contains(&signo) => {}
            // A bad signal, or SIGEV_THREAD (a libc construct the kernel never
            // sees) / anything else.
            _ => return err(Errno::EINVAL),
        }
        let mut timer = PosixTimer {
            id,
            tgid,
            clock,
            notify,
            signo,
            value,
            deadline: None,
            interval_ns: 0,
            overrun_last: 0,
            overrun_pending: 0,
            cpu: None,
            orphaned: false,
        };
        // SIGEV_THREAD_ID must name a thread of the caller's own group
        // (`good_sigevent`); the timer lives on (and signals) that thread.
        let elsewhere = notify == SIGEV_THREAD_ID && tid != cx.cur.pid;
        if elsewhere
            && !sh
                .procs
                .iter()
                .flatten()
                .any(|p| p.info.pid == tid && p.info.tgid == tgid)
        {
            return err(Errno::EINVAL);
        }
        if mem.write(out, &id.to_le_bytes()).is_err() {
            return err(Errno::EFAULT);
        }
        // The clock's own `timer_create`: an alarm clock wants
        // CAP_WAKE_ALARM, a CPU clock someone to be the clock of; its timer
        // is queued with that process's CPU time.
        if kind.is_alarm() && !Self::may_wake_alarm(cx) {
            return err(Errno::EPERM);
        }
        if let Some(c) = kind.cpu() {
            let of = match self.cpu_clock_of(sh, cx, c, false) {
                Ok(of) => of,
                Err(e) => return err(e),
            };
            let base = match of {
                CpuOf::Group(g) if g != tgid => sh
                    .procs
                    .iter()
                    .flatten()
                    .find(|p| p.info.tgid == g)
                    .map(|p| p.info.rlimits.clone()),
                _ => Some(cx.cur.rlimits.clone()),
            };
            let Some(base) = base else {
                return err(Errno::EINVAL);
            };
            let owner = if elsewhere { tid } else { cx.cur.pid };
            timer.cpu = Some(CpuTimer {
                of,
                base,
                link: cputimer::Link::new(owner),
            });
        }
        if elsewhere
            && let Some(p) = sh
                .procs
                .iter_mut()
                .flatten()
                .find(|p| p.info.pid == tid && p.info.tgid == tgid)
        {
            p.info.ptimers.push(timer);
        } else {
            cx.cur.ptimers.push(timer);
        }
        0
    }

    /// `timer_settime(timerid, flags, new, old)`: arm (or, with a zero
    /// `it_value`, disarm) the timer, reporting the previous setting in `old`.
    #[allow(clippy::unused_self, clippy::too_many_arguments)]
    pub(super) fn sys_timer_settime(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        id: u64,
        flags: u64,
        new: u64,
        old: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        if new == 0 {
            return err(Errno::EFAULT);
        }
        let interval = match read_ts(mem, new) {
            Ok(v) => v,
            Err(e) => return e,
        };
        let value = match read_ts(mem, new + 16) {
            Ok(v) => v,
            Err(e) => return e,
        };
        let now = poll::now_ns();
        // What the timer's clock reads, for an absolute expiry.
        let Some(t) = Self::ptimer_mut(sh, cx, id as i32) else {
            return err(Errno::EINVAL);
        };
        // A new setting starts the overrun count over.
        t.overrun_last = 0;
        t.overrun_pending = 0;
        if let Some(c) = t.cpu.clone() {
            let sigev_none = t.notify == SIGEV_NONE;
            let new = cputimer::Setting {
                value: u64::try_from(value).unwrap_or(u64::MAX),
                interval: u64::try_from(interval).unwrap_or(u64::MAX),
            };
            return Self::cpu_timer_settime(sh, cx, &c, sigev_none, flags, new, old, mem);
        }
        let clock = t.clock;
        let clock_now = super::time::Clock::decode(clock as u64)
            .and_then(|c| self.clock_now_ns(sh, cx, c).ok())
            .unwrap_or(now);
        let Some(t) = Self::ptimer_mut(sh, cx, id as i32) else {
            return err(Errno::EINVAL);
        };
        if old != 0 {
            let mut b = [0u8; 32];
            put_ts(&mut b, 0, t.interval_ns);
            put_ts(&mut b, 16, t.deadline.map_or(0, |d| d.saturating_sub(now)));
            if mem.write(old, &b).is_err() {
                return err(Errno::EFAULT);
            }
        }
        // A disarmed timer has no interval either (`posix_timer_set_common`).
        t.interval_ns = if value == 0 { 0 } else { interval };
        t.deadline = if value == 0 {
            None
        } else if flags & TIMER_ABSTIME != 0 {
            // An absolute expiry on the timer's clock, converted to the wall
            // deadline the scheduler tracks. A past time fires at once.
            Some(now + value.saturating_sub(clock_now))
        } else {
            Some(now + value)
        };
        0
    }

    /// `timer_settime` of CPU timer `c`, as Linux's `posix_cpu_timer_set`:
    /// `ESRCH` once its clock's task is gone (reaped); an expiry on the
    /// clock's CPU time, absolute or from now; and one already past fires at
    /// once, the task holding the timer told to signal it.
    #[allow(clippy::too_many_arguments)]
    fn cpu_timer_settime(
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        c: &CpuTimer,
        sigev_none: bool,
        flags: u64,
        new: cputimer::Setting,
        old: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some(now) = Self::cpu_time_of(sh, cx, c.of) else {
            return err(Errno::ESRCH);
        };
        let now = u64::try_from(now).unwrap_or(u64::MAX);
        let abstime = flags & TIMER_ABSTIME != 0;
        let (was, fired) =
            c.base
                .cpu_timers()
                .settime(&c.link, c.target(), now, new, abstime, sigev_none);
        if fired {
            let owner = c.link.owner();
            if owner == cx.cur.pid {
                fire_ptimers(&mut cx.cur, poll::now_ns());
            } else if let Some(p) = sh.procs.iter_mut().flatten().find(|p| p.info.pid == owner) {
                p.info.parked = false;
                p.info.kick.kick();
            }
        }
        if old != 0 && mem.write(old, &itimerspec(was)).is_err() {
            return err(Errno::EFAULT);
        }
        0
    }

    /// `timer_gettime(timerid, curr)`: the interval and the time left until
    /// the next expiration (zero when disarmed).
    #[allow(clippy::unused_self)]
    pub(super) fn sys_timer_gettime(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        id: u64,
        curr: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let now = poll::now_ns();
        let Some(t) = Self::ptimer_mut(sh, cx, id as i32) else {
            return err(Errno::EINVAL);
        };
        if let Some(c) = t.cpu.clone() {
            // Nothing to read once the clock's task is gone.
            let now = Self::cpu_time_of(sh, cx, c.of);
            let got = now.map_or_else(cputimer::Setting::default, |now| {
                let now = u64::try_from(now).unwrap_or(u64::MAX);
                c.base.cpu_timers().gettime(&c.link, now)
            });
            if mem.write(curr, &itimerspec(got)).is_err() {
                return err(Errno::EFAULT);
            }
            return 0;
        }
        let mut b = [0u8; 32];
        put_ts(&mut b, 0, t.interval_ns);
        // An armed timer whose deadline just passed reports the minimum
        // non-zero remainder (it is about to fire), never "disarmed".
        put_ts(
            &mut b,
            16,
            t.deadline.map_or(0, |d| d.saturating_sub(now).max(1)),
        );
        if mem.write(curr, &b).is_err() {
            return err(Errno::EFAULT);
        }
        0
    }

    /// `timer_getoverrun(timerid)`: overruns of the last delivered expiration.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_timer_getoverrun(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        id: u64,
    ) -> i64 {
        match Self::ptimer_mut(sh, cx, id as i32) {
            Some(t) => i64::from(t.overrun_last),
            None => err(Errno::EINVAL),
        }
    }

    /// `timer_delete(timerid)`. A signal already queued stays queued (Linux
    /// leaves it to be delivered or discarded).
    #[allow(clippy::unused_self)]
    pub(super) fn sys_timer_delete(&self, sh: &mut Shared, cx: &mut ServiceCtx, id: u64) -> i64 {
        let id = id as i32;
        let tgid = cx.cur.tgid;
        let gone = if let Some(i) = cx.cur.ptimers.iter().position(|t| t.id == id) {
            cx.cur.ptimers.remove(i)
        } else {
            let found = sh
                .procs
                .iter_mut()
                .flatten()
                .filter(|p| p.info.tgid == tgid)
                .find_map(|p| {
                    let i = p.info.ptimers.iter().position(|t| t.id == id)?;
                    Some(p.info.ptimers.remove(i))
                });
            let Some(t) = found else {
                return err(Errno::EINVAL);
            };
            t
        };
        gone.forget();
        0
    }
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{BASE, call, e, setup};
    use super::fire_ptimers;
    use crate::abi::arch::Sysno;
    use crate::abi::errno::Errno;

    #[test]
    fn timer_lifecycle_signals_and_overruns() {
        let (k, mut mem, mut v, mut cx) = setup();
        let (sev, idp, its, cur) = (BASE, BASE + 0x100, BASE + 0x200, BASE + 0x300);
        // SIGEV_SIGNAL, SIGUSR1 (10), value 0x55.
        let mut s = [0u8; 64];
        s[0..8].copy_from_slice(&0x55u64.to_le_bytes());
        s[8..12].copy_from_slice(&10u32.to_le_bytes());
        mem.write(sev, &s).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::TimerCreate,
                [1, sev, idp, 0, 0, 0]
            ),
            0
        );
        let id = u64::from(mem.read_u32(idp).unwrap());
        // A clock with no timers (MONOTONIC_RAW) is EOPNOTSUPP; an unknown one EINVAL.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::TimerCreate,
                [4, sev, idp, 0, 0, 0]
            ),
            e(Errno::EOPNOTSUPP)
        );
        // Arm: 1 ms period, first expiry in 1 s.
        let mut t = [0u8; 32];
        t[8..16].copy_from_slice(&1_000_000u64.to_le_bytes());
        t[16..24].copy_from_slice(&1u64.to_le_bytes());
        mem.write(its, &t).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::TimerSettime,
                [id, 0, its, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::TimerGettime,
                [id, cur, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_u64(cur + 8).unwrap(), 1_000_000, "interval");
        let left =
            mem.read_u64(cur + 16).unwrap() * 1_000_000_000 + mem.read_u64(cur + 24).unwrap();
        assert!(left > 900_000_000 && left <= 1_000_000_000, "{left}");
        // Expire it 3 periods late: one signal with 3 overruns.
        let dl = cx.cur.ptimers[0].deadline.unwrap();
        fire_ptimers(&mut cx.cur, dl + 3_000_000);
        assert_ne!(cx.cur.pending & (1 << 9), 0, "SIGUSR1 pending");
        let q = cx.cur.queued_siginfo[10].unwrap();
        assert_eq!((q.code, q.pid, q.uid, q.value), (-2, id as i32, 3, 0x55));
        // While still pending, a further expiry only counts an overrun.
        let dl = cx.cur.ptimers[0].deadline.unwrap();
        fire_ptimers(&mut cx.cur, dl);
        assert_eq!(cx.cur.ptimers[0].overrun_pending, 1);
        // Taken, the signal carries them all, and timer_getoverrun says so.
        let (q, _) = cx.cur.take_siginfo(10);
        assert_eq!(q.unwrap().uid, 4);
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::TimerGetoverrun,
                [id, 0, 0, 0, 0, 0]
            ),
            4
        );
        // Disarm, delete, and a second delete fails.
        mem.write(its, &[0u8; 32]).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::TimerSettime,
                [id, 0, its, 0, 0, 0]
            ),
            0
        );
        assert_eq!(cx.cur.ptimers[0].deadline, None);
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::TimerDelete,
                [id, 0, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::TimerDelete,
                [id, 0, 0, 0, 0, 0]
            ),
            e(Errno::EINVAL)
        );
    }

    #[test]
    fn sigev_none_never_signals_and_null_sevp_is_sigalrm() {
        let (k, mut mem, mut v, mut cx) = setup();
        let (sev, idp, its) = (BASE, BASE + 0x100, BASE + 0x200);
        let mut s = [0u8; 64];
        s[12..16].copy_from_slice(&1i32.to_le_bytes()); // SIGEV_NONE
        mem.write(sev, &s).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::TimerCreate,
                [0, sev, idp, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::TimerCreate,
                [1, 0, idp + 4, 0, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_u32(idp + 4).unwrap(), 1, "lowest free id");
        let mut t = [0u8; 32];
        t[24..32].copy_from_slice(&1u64.to_le_bytes()); // 1 ns
        mem.write(its, &t).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::TimerSettime,
                [0, 0, its, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::TimerSettime,
                [1, 0, its, 0, 0, 0]
            ),
            0
        );
        fire_ptimers(&mut cx.cur, u128::MAX / 2);
        assert_eq!(
            cx.cur.pending,
            1 << 13,
            "only the NULL-sevp timer's SIGALRM"
        );
        assert_eq!(
            cx.cur.queued_siginfo[14].unwrap().value,
            1,
            "value = timer id"
        );
    }
}
