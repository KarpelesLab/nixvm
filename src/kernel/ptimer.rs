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
//! The CPU-time clocks (`CLOCK_PROCESS_CPUTIME_ID`/`CLOCK_THREAD_CPUTIME_ID`)
//! are approximated by wall time: a task that is mostly running consumes CPU at
//! wall rate, which is what profilers arming these timers assume.

use super::{Kernel, ProcInfo, QueuedSig, RunState, ServiceCtx, Shared, err, poll};
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
/// ns): post its signal (or count an overrun if one is still pending), re-arm a
/// periodic timer past `now`, disarm a one-shot. Un-parks the task when a
/// signal was posted so a blocking syscall wakes to take it.
pub(super) fn fire_ptimers(info: &mut ProcInfo, now: u128) {
    for i in 0..info.ptimers.len() {
        let Some(dl) = info.ptimers[i].deadline else {
            continue;
        };
        if now < dl {
            continue;
        }
        let t = &info.ptimers[i];
        // How many expirations elapsed (a periodic timer may have missed some
        // while the task was descheduled).
        let (count, next) = match (now - dl).checked_div(t.interval_ns) {
            Some(missed) => (missed + 1, Some(dl + (missed + 1) * t.interval_ns)),
            None => (1, None), // one-shot (interval 0)
        };
        let count = i32::try_from(count).unwrap_or(i32::MAX);
        let queued = t.notify != SIGEV_NONE && signal_queued(info, t);
        let t = &mut info.ptimers[i];
        t.deadline = next;
        if t.notify == SIGEV_NONE {
            continue;
        }
        if queued {
            t.overrun_pending = t.overrun_pending.saturating_add(count);
            continue;
        }
        let overrun = t.overrun_pending.saturating_add(count - 1);
        t.overrun_last = overrun;
        t.overrun_pending = 0;
        let (sig, qs) = (
            u64::from(t.signo),
            QueuedSig {
                code: SI_TIMER,
                pid: t.id,
                uid: overrun as u32,
                value: t.value,
            },
        );
        info.pending |= 1u64 << (sig - 1);
        info.post_siginfo(sig, qs);
        info.parked = false;
    }
}

/// Encode `ns` as a `struct timespec` into `b[off..off + 16]`.
fn put_ts(b: &mut [u8], off: usize, ns: u128) {
    b[off..off + 8].copy_from_slice(&((ns / NS) as i64).to_le_bytes());
    b[off + 8..off + 16].copy_from_slice(&((ns % NS) as i64).to_le_bytes());
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
        let clock = i64::from(clock as i32);
        // REALTIME, MONOTONIC, the two CPU clocks, BOOTTIME, the *_ALARM
        // clocks and TAI have timer support; MONOTONIC_RAW and the COARSE
        // clocks don't. Negative ids are another task's CPU clock.
        if clock >= 0 && !matches!(clock, 0..=3 | 7..=9 | 11) {
            return err(Errno::EINVAL);
        }
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
        let (value, signo, notify, tid) = if sevp == 0 {
            // NULL sevp: SIGEV_SIGNAL with SIGALRM and the timer id as value.
            (id as u64, SIGALRM, SIGEV_SIGNAL, 0)
        } else {
            let Ok(raw) = mem.read_vec(sevp, 20) else {
                return err(Errno::EFAULT);
            };
            let w32 = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
            (
                u64::from_le_bytes(raw[0..8].try_into().unwrap()),
                w32(8),
                w32(12) as i32,
                w32(16) as i32,
            )
        };
        match notify {
            SIGEV_NONE => {}
            SIGEV_SIGNAL | SIGEV_THREAD_ID if (1..=64).contains(&signo) => {}
            // A bad signal, or SIGEV_THREAD (a libc construct the kernel never
            // sees) / anything else.
            _ => return err(Errno::EINVAL),
        }
        let timer = PosixTimer {
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
        };
        if mem.write(out, &id.to_le_bytes()).is_err() {
            return err(Errno::EFAULT);
        }
        // SIGEV_THREAD_ID must name a thread of the caller's own group; the
        // timer lives on (and signals) that thread.
        if notify == SIGEV_THREAD_ID && tid != cx.cur.pid {
            let Some(p) = sh
                .procs
                .iter_mut()
                .flatten()
                .find(|p| p.info.pid == tid && p.info.tgid == tgid)
            else {
                return err(Errno::EINVAL);
            };
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
        t.interval_ns = interval;
        t.overrun_pending = 0;
        t.deadline = if value == 0 {
            None
        } else if flags & TIMER_ABSTIME != 0 {
            // An absolute expiry on the timer's clock, converted to the wall
            // deadline the scheduler tracks. A past time fires at once.
            let clock_now = match t.clock {
                0 | 8 | 11 => crate::clock::now_unix(),
                _ => crate::clock::now_monotonic(),
            }
            .as_nanos();
            Some(now + value.saturating_sub(clock_now))
        } else {
            Some(now + value)
        };
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
        if let Some(i) = cx.cur.ptimers.iter().position(|t| t.id == id) {
            cx.cur.ptimers.remove(i);
            return 0;
        }
        for p in sh.procs.iter_mut().flatten() {
            if p.info.tgid != tgid {
                continue;
            }
            if let Some(i) = p.info.ptimers.iter().position(|t| t.id == id) {
                p.info.ptimers.remove(i);
                return 0;
            }
        }
        err(Errno::EINVAL)
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
        // Unsupported clock / bad notify are EINVAL.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::TimerCreate,
                [4, sev, idp, 0, 0, 0]
            ),
            e(Errno::EINVAL)
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
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::TimerGetoverrun,
                [id, 0, 0, 0, 0, 0]
            ),
            3
        );
        // While still pending, a further expiry only counts an overrun.
        let dl = cx.cur.ptimers[0].deadline.unwrap();
        fire_ptimers(&mut cx.cur, dl);
        assert_eq!(cx.cur.ptimers[0].overrun_pending, 1);
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
