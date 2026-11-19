//! Time and sleep syscalls, and the clock ids they take.
//!
//! nixvm reads the host clocks for the "get" calls. Clock/time *setters* are
//! refused (`EPERM`) since the guest does not own the host clock.
//!
//! A clock id is checked as Linux's `clockid_to_kclock` checks it ([`Clock`]):
//! the fixed ids, the CPU clocks `clock_getcpuclockid` and
//! `pthread_getcpuclockid` make (negative, naming a task by pid), and the
//! `CLOCKFD` ids of a clock device, which nixvm has none of. Each call then
//! refuses what its clock can't do, as Linux's `struct k_clock` has it: no
//! sleep on `CLOCK_THREAD_CPUTIME_ID` or the raw and coarse clocks, no timers
//! on the latter, `CAP_WAKE_ALARM` for the `*_ALARM` clocks' timers and sleeps.

use std::time::Duration;

use super::userns::TimeOffsets;
use super::{Kernel, ServiceCtx, Shared, err};
use crate::abi::errno::Errno;
use crate::vcpu::GuestMemory;

/// Time since the UNIX epoch on the host clock (see [`crate::clock`] — the
/// platform-safe source that also works on wasm32).
fn host_now() -> Duration {
    crate::clock::now_unix()
}

/// `gettimeofday(tv, tz)` — write `struct timeval { i64 tv_sec; i64 tv_usec }`
/// (16 bytes) from the host clock. `tz` is obsolete: it gets the kernel's
/// `sys_tz`, which nothing here sets (`settimeofday` can't), so zeros — what
/// the vDSOs write too.
pub(super) fn sys_gettimeofday(tv: u64, tz: u64, mem: &mut GuestMemory) -> i64 {
    if tz != 0 && mem.write(tz, &[0u8; 8]).is_err() {
        return err(Errno::EFAULT);
    }
    if tv == 0 {
        return 0;
    }
    let now = host_now();
    let mut b = [0u8; 16];
    b[0..8].copy_from_slice(&(now.as_secs() as i64).to_le_bytes());
    b[8..16].copy_from_slice(&i64::from(now.subsec_micros()).to_le_bytes());
    match mem.write(tv, &b) {
        Ok(()) => 0,
        Err(_) => err(Errno::EFAULT),
    }
}

// `nanosleep`/`clock_nanosleep` live in the kernel module ([`Kernel::sys_nanosleep`]):
// unlike the pure time getters here, they suspend the caller on the scheduler's
// timed-wait, so they need the task context.

/// `time(tloc)` — return host unix seconds; if `tloc` is non-null also write it
/// there as an 8-byte value.
pub(super) fn sys_time(tloc: u64, mem: &mut GuestMemory) -> i64 {
    let secs = host_now().as_secs() as i64;
    if tloc != 0 && mem.write(tloc, &secs.to_le_bytes()).is_err() {
        return err(Errno::EFAULT);
    }
    secs
}

// ---- clock ids ----------------------------------------------------------

/// How far `CLOCK_TAI` runs ahead of `CLOCK_REALTIME`: the TAI offset
/// `adjtimex` reports ([`Kernel::sys_adjtimex`]), so the two agree.
pub(super) const TAI_OFFSET_S: u64 = 37;

/// `CPUCLOCK_SCHED`, the run-time clock `CLOCK_*_CPUTIME_ID` are.
const CPUCLOCK_SCHED: u32 = 2;
/// `CPUCLOCK_MAX`: `PROF`, `VIRT` and `SCHED` are the CPU clocks there are.
const CPUCLOCK_MAX: u32 = 3;
/// `CAP_WAKE_ALARM`: what the `*_ALARM` clocks' timers and sleeps need.
const CAP_WAKE_ALARM: u32 = 35;

/// A clock id, as Linux's `clockid_to_kclock` tells them apart: the fixed
/// ids (`CLOCK_SGI_CYCLE`, 10, was removed, and nixvm has no auxiliary
/// clocks), the CPU clocks `clock_getcpuclockid` and `pthread_getcpuclockid`
/// make (`MAKE_PROCESS_CPUCLOCK`/`MAKE_THREAD_CPUCLOCK`, negative), and the
/// `CLOCKFD` ids of a PTP-style clock device, which nixvm has none of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Clock {
    Realtime,
    Monotonic,
    ProcessCpu,
    ThreadCpu,
    MonotonicRaw,
    RealtimeCoarse,
    MonotonicCoarse,
    Boottime,
    RealtimeAlarm,
    BoottimeAlarm,
    Tai,
    /// A CPU clock by pid (`clock_posix_cpu`).
    Cpu(CpuClock),
    /// `CLOCKFD`: a clock device's descriptor.
    Fd,
}

/// A CPU clock: whose (`pid`, 0 for the caller; a thread with `thread`) and
/// which of `PROF`, `VIRT` and `SCHED` (they all read the same here: the
/// guest's CPU time is all user time).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CpuClock {
    pub(super) pid: i32,
    pub(super) thread: bool,
    pub(super) which: u32,
}

impl Clock {
    /// The clock `id` (a 32-bit `clockid_t`) names, or `None` for one Linux
    /// doesn't have (`EINVAL`).
    pub(super) fn decode(id: u64) -> Option<Self> {
        let id = id as i32;
        if id < 0 {
            // CLOCKFD_MASK is the per-thread bit and the clock bits.
            if id & 7 == 3 {
                return Some(Self::Fd);
            }
            return Some(Self::Cpu(CpuClock {
                pid: !(id >> 3),
                thread: id & 4 != 0,
                which: (id & 3) as u32,
            }));
        }
        Some(match id {
            0 => Self::Realtime,
            1 => Self::Monotonic,
            2 => Self::ProcessCpu,
            3 => Self::ThreadCpu,
            4 => Self::MonotonicRaw,
            5 => Self::RealtimeCoarse,
            6 => Self::MonotonicCoarse,
            7 => Self::Boottime,
            8 => Self::RealtimeAlarm,
            9 => Self::BoottimeAlarm,
            11 => Self::Tai,
            _ => return None,
        })
    }

    /// The CPU clock this is, if it is one (`CLOCK_PROCESS_CPUTIME_ID` and
    /// `CLOCK_THREAD_CPUTIME_ID` are the caller's run-time clocks).
    pub(super) fn cpu(self) -> Option<CpuClock> {
        let own = |thread| CpuClock {
            pid: 0,
            thread,
            which: CPUCLOCK_SCHED,
        };
        match self {
            Self::ProcessCpu => Some(own(false)),
            Self::ThreadCpu => Some(own(true)),
            Self::Cpu(c) => Some(c),
            _ => None,
        }
    }

    /// Whether `clock_nanosleep` can sleep on it (the clock's `nsleep`):
    /// not the thread clock, the raw and coarse clocks, or a clock device.
    fn sleeps(self) -> bool {
        !matches!(
            self,
            Self::ThreadCpu
                | Self::MonotonicRaw
                | Self::RealtimeCoarse
                | Self::MonotonicCoarse
                | Self::Fd
        )
    }

    /// Whether `timer_create` can make a timer on it (`timer_create`).
    fn has_timers(self) -> bool {
        !matches!(
            self,
            Self::MonotonicRaw | Self::RealtimeCoarse | Self::MonotonicCoarse | Self::Fd
        )
    }

    /// One of the `*_ALARM` clocks, which wake a suspended machine.
    pub(super) fn is_alarm(self) -> bool {
        matches!(self, Self::RealtimeAlarm | Self::BoottimeAlarm)
    }

    /// What a clock that isn't a CPU clock reads now, in a time namespace
    /// with `offsets` (which move `CLOCK_MONOTONIC`'s three and
    /// `CLOCK_BOOTTIME`'s two). The machine never suspends, so boot time is
    /// monotonic time. `None` for a CPU clock or a clock device.
    pub(super) fn now(self, offsets: TimeOffsets) -> Option<Duration> {
        let mono = crate::clock::now_monotonic;
        Some(match self {
            Self::Realtime | Self::RealtimeCoarse | Self::RealtimeAlarm => host_now(),
            Self::Tai => host_now() + Duration::from_secs(TAI_OFFSET_S),
            Self::Monotonic | Self::MonotonicRaw | Self::MonotonicCoarse => {
                super::offset_clock(mono(), offsets.monotonic)
            }
            Self::Boottime | Self::BoottimeAlarm => super::offset_clock(mono(), offsets.boottime),
            Self::ProcessCpu | Self::ThreadCpu | Self::Cpu(_) | Self::Fd => return None,
        })
    }
}

/// Whose CPU time a CPU clock reads: a thread's, or a thread group's (by
/// global id).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum CpuOf {
    Thread(i32),
    Group(i32),
}

/// Write `d` as a `struct timespec` at `addr`.
fn put_timespec(mem: &mut GuestMemory, addr: u64, d: Duration) -> i64 {
    let mut b = [0u8; 16];
    b[0..8].copy_from_slice(&d.as_secs().to_le_bytes());
    b[8..16].copy_from_slice(&u64::from(d.subsec_nanos()).to_le_bytes());
    match mem.write(addr, &b) {
        Ok(()) => 0,
        Err(_) => err(Errno::EFAULT),
    }
}

/// The nanoseconds `ns` as a `Duration`.
fn from_ns(ns: u128) -> Duration {
    Duration::from_nanos(u64::try_from(ns).unwrap_or(u64::MAX))
}

impl Kernel {
    /// Whose time CPU clock `c` reads, as Linux's `pid_for_clock` finds it:
    /// the caller (pid 0), a thread of the caller's own group, or a process
    /// by its leader's pid — or by the caller's own thread id, for
    /// `clock_gettime` (`gettime`). `EINVAL` for anything else: a pid with
    /// no task, another group's thread, a non-leader's pid, or no such
    /// clock (`which` past `SCHED`).
    pub(super) fn cpu_clock_of(
        &self,
        sh: &Shared,
        cx: &ServiceCtx,
        c: CpuClock,
        gettime: bool,
    ) -> Result<CpuOf, Errno> {
        if c.which >= CPUCLOCK_MAX {
            return Err(Errno::EINVAL);
        }
        if c.pid == 0 {
            return Ok(if c.thread {
                CpuOf::Thread(cx.cur.pid)
            } else {
                CpuOf::Group(cx.cur.tgid)
            });
        }
        let pid = self.global_pid(cx, c.pid).ok_or(Errno::EINVAL)?;
        let task = |pid: i32| {
            (cx.cur.pid == pid).then_some(&cx.cur).or_else(|| {
                sh.procs
                    .iter()
                    .flatten()
                    .map(|p| &p.info)
                    .find(|p| p.pid == pid)
            })
        };
        if c.thread {
            return match task(pid) {
                Some(t) if t.tgid == cx.cur.tgid => Ok(CpuOf::Thread(pid)),
                _ => Err(Errno::EINVAL),
            };
        }
        if gettime && pid == cx.cur.pid {
            return Ok(CpuOf::Group(cx.cur.tgid));
        }
        match task(pid) {
            Some(t) if t.tgid == pid => Ok(CpuOf::Group(pid)),
            _ => Err(Errno::EINVAL),
        }
    }

    /// The CPU time (ns) of `of`: a thread's own, or its whole group's.
    /// `None` once it has gone.
    pub(super) fn cpu_time_of(sh: &Shared, cx: &ServiceCtx, of: CpuOf) -> Option<u128> {
        match of {
            CpuOf::Thread(tid) if tid == cx.cur.pid => Some(cx.cur.cpu_ns),
            CpuOf::Thread(tid) => sh
                .procs
                .iter()
                .flatten()
                .find(|p| p.info.pid == tid)
                .map(|p| p.info.cpu_ns),
            CpuOf::Group(tgid) if tgid == cx.cur.tgid => Some(super::process_cpu_ns(sh, cx)),
            CpuOf::Group(tgid) => {
                let mut members = sh.procs.iter().flatten().filter(|p| p.info.tgid == tgid);
                let first = members.next()?;
                Some(first.info.cpu_ns + members.map(|p| p.info.cpu_ns).sum::<u128>())
            }
        }
    }

    /// What `clock` reads now (ns), CPU clocks included; `Err` as
    /// `clock_gettime` fails (`EINVAL` for a CPU clock of no one, or a
    /// clock device).
    pub(super) fn clock_now_ns(
        &self,
        sh: &Shared,
        cx: &ServiceCtx,
        clock: Clock,
    ) -> Result<u128, Errno> {
        if let Some(c) = clock.cpu() {
            let of = self.cpu_clock_of(sh, cx, c, true)?;
            return Self::cpu_time_of(sh, cx, of).ok_or(Errno::EINVAL);
        }
        clock
            .now(self.time_offsets_for(cx))
            .map(|d| d.as_nanos())
            .ok_or(Errno::EINVAL)
    }

    /// `clock_gettime(clk_id, timespec)`. Each Linux clock id is served from
    /// the matching source, so the guest sees genuine wall, monotonic and CPU
    /// time: `CLOCK_MONOTONIC` agrees with the vDSO fast path (both use
    /// [`crate::clock::now_monotonic`]), and the CPU clocks, which the vDSO
    /// never fast-paths, read the scheduler's per-task accounting
    /// ([`super::ProcInfo::cpu_ns`]): this task's, its whole group's, or
    /// another task's by a `clock_getcpuclockid` id. An id Linux doesn't
    /// have, or a CPU clock of nobody, is `EINVAL`. (The `*_ALARM` clocks
    /// read as Linux's do with an RTC, which the machine has.)
    pub(super) fn sys_clock_gettime(
        &self,
        cx: &ServiceCtx,
        clk_id: u64,
        ts: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some(clock) = Clock::decode(clk_id) else {
            return err(Errno::EINVAL);
        };
        let now = if clock.cpu().is_some() {
            // The CPU clocks read the task table. No `sh` is held on this
            // call path (the lock-free dispatch reaches here), so lock it.
            let sh = self.shared.lock().unwrap();
            match self.clock_now_ns(&sh, cx, clock) {
                Ok(ns) => from_ns(ns),
                Err(e) => return err(e),
            }
        } else {
            match clock.now(self.time_offsets_for(cx)) {
                Some(d) => d,
                None => return err(Errno::EINVAL),
            }
        };
        put_timespec(mem, ts, now)
    }

    /// `clock_getres(clk_id, res)`: 1 ns for every clock (the hrtimer clocks
    /// are high-resolution, and the CPU clocks are `CPUCLOCK_SCHED`'s or
    /// read from it), once the id is checked as `clock_gettime` checks it.
    /// A null `res` only checks.
    pub(super) fn sys_clock_getres(
        &self,
        cx: &ServiceCtx,
        clk_id: u64,
        res: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some(clock) = Clock::decode(clk_id) else {
            return err(Errno::EINVAL);
        };
        if clock == Clock::Fd {
            return err(Errno::EINVAL);
        }
        if let Some(c) = clock.cpu() {
            let sh = self.shared.lock().unwrap();
            if let Err(e) = self.cpu_clock_of(&sh, cx, c, false) {
                return err(e);
            }
        }
        if res == 0 {
            return 0;
        }
        put_timespec(mem, res, Duration::from_nanos(1))
    }

    /// `clock_settime(clk_id, tp)`: only `CLOCK_REALTIME` and the CPU clocks
    /// can be set at all (`EINVAL` for the others), and a CPU clock never
    /// (`EPERM`, once its pid checks out); the guest doesn't own the host's
    /// wall clock either, so that one is `EPERM` too, after the checks
    /// Linux makes first: the time is read (`EFAULT`) and must be valid.
    pub(super) fn sys_clock_settime(
        &self,
        sh: &Shared,
        cx: &ServiceCtx,
        clk_id: u64,
        tp: u64,
        mem: &GuestMemory,
    ) -> i64 {
        // `TIME_SETTOD_SEC_MAX`: `KTIME_SEC_MAX` less 30 years.
        const SETTOD_SEC_MAX: i64 = i64::MAX / 1_000_000_000 - 30 * 365 * 86_400;
        let Some(clock @ (Clock::Realtime | Clock::Cpu(_) | Clock::Fd)) = Clock::decode(clk_id)
        else {
            return err(Errno::EINVAL);
        };
        let (Ok(sec), Ok(nsec)) = (mem.read_u64(tp), mem.read_u64(tp + 8)) else {
            return err(Errno::EFAULT);
        };
        match clock {
            Clock::Cpu(c) => match self.cpu_clock_of(sh, cx, c, false) {
                Ok(_) => err(Errno::EPERM),
                Err(e) => err(e),
            },
            Clock::Realtime => {
                let (sec, nsec) = (sec as i64, nsec as i64);
                if !(0..SETTOD_SEC_MAX).contains(&sec) || !(0..1_000_000_000).contains(&nsec) {
                    err(Errno::EINVAL)
                } else {
                    err(Errno::EPERM)
                }
            }
            _ => err(Errno::EINVAL),
        }
    }

    /// `clock_adjtime`'s check of the clock (`do_clock_adjtime`): only
    /// `CLOCK_REALTIME` can be adjusted; another clock Linux has is
    /// `EOPNOTSUPP`, and an unknown id or a clock device (nixvm has none)
    /// `EINVAL`.
    pub(super) fn adjtime_clock(clk_id: u64) -> Result<(), Errno> {
        match Clock::decode(clk_id) {
            Some(Clock::Realtime) => Ok(()),
            None | Some(Clock::Fd) => Err(Errno::EINVAL),
            Some(_) => Err(Errno::EOPNOTSUPP),
        }
    }

    /// `clock_nanosleep`'s and `timer_create`'s first checks of the clock,
    /// in Linux's order: `EINVAL` for an unknown id, then `EOPNOTSUPP` for
    /// a clock with no sleep (`timer`: with no timers). The `*_ALARM`
    /// clocks' `CAP_WAKE_ALARM` ([`Self::may_wake_alarm`]) and the CPU
    /// clocks' pid come later.
    pub(super) fn clock_for(clk_id: u64, timer: bool) -> Result<Clock, Errno> {
        let clock = Clock::decode(clk_id).ok_or(Errno::EINVAL)?;
        let can = if timer {
            clock.has_timers()
        } else {
            clock.sleeps()
        };
        if !can {
            return Err(Errno::EOPNOTSUPP);
        }
        Ok(clock)
    }

    /// Whether the caller may set an `*_ALARM` clock's timer or sleep on it
    /// (`CAP_WAKE_ALARM`).
    pub(super) fn may_wake_alarm(cx: &ServiceCtx) -> bool {
        Self::capable(cx, CAP_WAKE_ALARM)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_ids_decode_as_linux_tells_them_apart() {
        assert_eq!(Clock::decode(0), Some(Clock::Realtime));
        assert_eq!(Clock::decode(11), Some(Clock::Tai));
        assert_eq!(Clock::decode(10), None); // CLOCK_SGI_CYCLE, removed
        assert_eq!(Clock::decode(12), None);
        assert_eq!(Clock::decode(16), None); // no auxiliary clocks
        // MAKE_PROCESS_CPUCLOCK(0, CPUCLOCK_SCHED) = -6: the caller's.
        let me = CpuClock {
            pid: 0,
            thread: false,
            which: 2,
        };
        assert_eq!(Clock::decode(-6i64 as u64), Some(Clock::Cpu(me)));
        // MAKE_THREAD_CPUCLOCK(1234, CPUCLOCK_SCHED).
        let id = ((!1234i32) << 3) | 4 | 2;
        let t = Clock::decode(u64::from(id as u32)).unwrap();
        assert_eq!(
            t,
            Clock::Cpu(CpuClock {
                pid: 1234,
                thread: true,
                which: 2
            })
        );
        // FD_TO_CLOCKID(5) = (~5 << 3) | 3.
        assert_eq!(
            Clock::decode(u64::from(((!5i32 << 3) | 3) as u32)),
            Some(Clock::Fd)
        );
        assert!(!Clock::ThreadCpu.sleeps() && Clock::ThreadCpu.has_timers());
        assert!(!Clock::MonotonicCoarse.has_timers() && Clock::ProcessCpu.sleeps());
    }
}
