//! CPU-time timers: the POSIX timers on a CPU clock (`timer_create` on
//! `CLOCK_PROCESS_CPUTIME_ID`, `CLOCK_THREAD_CPUTIME_ID` or a CPU clock by
//! pid), and `ITIMER_VIRTUAL`/`ITIMER_PROF`. They expire on the CPU time their
//! clock's task has used, as Linux's `posix-cpu-timers.c` has them: a process
//! clock on the sum over its threads (exited ones included), a thread clock
//! on that thread's alone.
//!
//! Each thread group has one [`CpuTimers`], kept with its CPU time and
//! `RLIMIT_CPU` in [`super::RLimits`] (Linux's `signal_struct`), holding every
//! timer on its process clock and on its threads' clocks. The scheduler adds a
//! step's CPU time to the group and asks [`CpuTimers::charge`] whether anything
//! is due: two loads and compares against the earliest expiry (Linux's
//! `nextevt` expiry cache), the lock only when something is.
//!
//! A POSIX timer's signal goes to the task holding it (see [`super::ptimer`]),
//! which may be another thread or process than the clock's: the expiry only
//! counts on the timer's [`Link`], and that task turns the count into its
//! signal (and overruns) on its next step, woken for it if it was asleep.
//! The itimers signal the process whose CPU time ran out.
//!
//! The guest's CPU time is all user time here, so the `PROF`, `VIRT` and
//! `SCHED` clocks read the same, and `ITIMER_VIRTUAL` counts as `ITIMER_PROF`.

use std::sync::atomic::{AtomicI32, AtomicU32, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

/// `ITIMER_VIRTUAL`'s signal.
pub(super) const SIGVTALRM: u64 = 26;
/// `ITIMER_PROF`'s signal.
pub(super) const SIGPROF: u64 = 27;
/// Linux's `TICK_NSEC` (at `CONFIG_HZ=250`): what `setitimer` adds to a CPU
/// itimer's first expiry, and what one about to fire reads as left.
pub(super) const TICK_NSEC: u64 = 4_000_000;

/// The clock a POSIX CPU timer is on: its process's, or one of its threads'.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Target {
    Group,
    Thread(i32),
}

/// What a POSIX timer on a CPU clock shares with the clock's [`CpuTimers`].
#[derive(Debug)]
pub(super) struct Link {
    /// Expirations the task holding the timer hasn't turned into a signal yet.
    pub(super) fired: AtomicU32,
    /// That task (another thread of its process, once it has exited).
    owner: AtomicI32,
}

impl Link {
    pub(super) fn new(owner: i32) -> Arc<Self> {
        Arc::new(Self {
            fired: AtomicU32::new(0),
            owner: AtomicI32::new(owner),
        })
    }

    /// The task holding the timer.
    pub(super) fn owner(&self) -> i32 {
        self.owner.load(Relaxed)
    }

    /// The timer is handed to task `pid`.
    pub(super) fn set_owner(&self, pid: i32) {
        self.owner.store(pid, Relaxed);
    }
}

/// An `itimerspec`, in nanoseconds: the time left (or an absolute expiry, as
/// [`CpuTimers::settime`] takes it) and the interval.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Setting {
    pub(super) value: u64,
    pub(super) interval: u64,
}

#[derive(Debug)]
struct Entry {
    link: Arc<Link>,
    target: Target,
    /// The absolute expiry on the target's clock; 0 for none.
    expires: u64,
    interval: u64,
    /// Queued to expire: set, not yet expired (or re-armed by its interval),
    /// and not `SIGEV_NONE`, which never is (`timer_gettime` is all it has).
    armed: bool,
    sigev_none: bool,
}

impl Entry {
    /// Expire at `now`: count it (and the intervals missed) on the link, and
    /// re-arm a periodic timer past `now` or disarm a one-shot one.
    fn fire(&mut self, now: u64) {
        let count = if let Some(missed) = (now - self.expires).checked_div(self.interval) {
            let n = missed + 1;
            self.expires = self.expires.saturating_add(n.saturating_mul(self.interval));
            n
        } else {
            self.expires = 0;
            self.armed = false;
            1
        };
        // The holder takes the count (as an `int` of overruns) from another
        // CPU, so it's added to, not replaced.
        let count = u32::try_from(count.min(i32::MAX as u64)).unwrap_or(u32::MAX);
        self.link.fired.fetch_add(count, Relaxed);
    }

    /// The setting `timer_gettime` reads at `now`, as Linux's
    /// `__posix_cpu_timer_get`: a periodic timer not queued (`SIGEV_NONE`)
    /// moves on past `now`; one that expired and hasn't signalled yet has
    /// 1 ns left, a `SIGEV_NONE` one none.
    fn get(&mut self, now: u64) -> Setting {
        if self.expires == 0 {
            return Setting::default();
        }
        if self.interval != 0 && !self.armed && now >= self.expires {
            let n = (now - self.expires) / self.interval + 1;
            self.expires = self.expires.saturating_add(n.saturating_mul(self.interval));
        }
        let value = if now < self.expires {
            self.expires - now
        } else {
            u64::from(!self.sigev_none)
        };
        Setting {
            value,
            interval: self.interval,
        }
    }
}

/// `ITIMER_VIRTUAL` or `ITIMER_PROF`: an absolute expiry on the process's
/// CPU clock (0 for none) and the interval it re-arms by.
#[derive(Clone, Copy, Debug, Default)]
struct ITimer {
    expires: u64,
    incr: u64,
}

#[derive(Debug, Default)]
struct Inner {
    entries: Vec<Entry>,
    /// `ITIMER_VIRTUAL`, `ITIMER_PROF`.
    itimers: [ITimer; 2],
}

/// A thread group's CPU-time timers.
#[derive(Debug)]
pub(super) struct CpuTimers {
    /// The process CPU time at which a timer on the process clock or an
    /// itimer is next due; `u64::MAX` for none.
    next_group: AtomicU64,
    /// The least thread CPU time at which a timer on any of its threads'
    /// clocks is due; `u64::MAX` for none.
    next_thread: AtomicU64,
    inner: Mutex<Inner>,
}

impl Default for CpuTimers {
    fn default() -> Self {
        Self {
            next_group: AtomicU64::new(u64::MAX),
            next_thread: AtomicU64::new(u64::MAX),
            inner: Mutex::new(Inner::default()),
        }
    }
}

/// What CPU time just used calls for ([`CpuTimers::charge`]).
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct Due {
    /// Signals for the process: its itimers'.
    pub(super) signals: Vec<u64>,
    /// The tasks holding a POSIX timer that expired.
    pub(super) owners: Vec<i32>,
}

impl CpuTimers {
    /// Forget timers whose holder deleted them or is gone (the link is
    /// all that is left of them), and publish the earliest expiries. Called
    /// with the lock held, so a store can't overtake a later one.
    fn publish(&self, inner: &mut Inner) {
        inner.entries.retain(|e| Arc::strong_count(&e.link) > 1);
        let mut group = inner
            .itimers
            .iter()
            .filter(|it| it.expires != 0)
            .map(|it| it.expires)
            .min()
            .unwrap_or(u64::MAX);
        let mut thread = u64::MAX;
        for e in inner.entries.iter().filter(|e| e.armed) {
            match e.target {
                Target::Group => group = group.min(e.expires),
                Target::Thread(_) => thread = thread.min(e.expires),
            }
        }
        self.next_group.store(group, Relaxed);
        self.next_thread.store(thread, Relaxed);
    }

    /// The process has used `group_ns` of CPU time and its thread `tid`
    /// `thread_ns`: expire what that makes due (Linux's
    /// `run_posix_cpu_timers`). `None`, without the lock, when nothing is.
    pub(super) fn charge(&self, group_ns: u64, tid: i32, thread_ns: u64) -> Option<Due> {
        if group_ns < self.next_group.load(Relaxed) && thread_ns < self.next_thread.load(Relaxed) {
            return None;
        }
        let mut inner = self.inner.lock().unwrap();
        let mut due = Due::default();
        // `check_cpu_itimer`: one signal, and the expiry moves on by one
        // interval (the next tick looks again).
        for (it, sig) in inner.itimers.iter_mut().zip([SIGVTALRM, SIGPROF]) {
            if it.expires != 0 && group_ns >= it.expires {
                it.expires = if it.incr == 0 {
                    0
                } else {
                    it.expires.saturating_add(it.incr)
                };
                due.signals.push(sig);
            }
        }
        for e in &mut inner.entries {
            let now = match e.target {
                Target::Group => group_ns,
                Target::Thread(t) if t == tid => thread_ns,
                Target::Thread(_) => continue,
            };
            if e.armed && now >= e.expires && Arc::strong_count(&e.link) > 1 {
                e.fire(now);
                let owner = e.link.owner();
                if !due.owners.contains(&owner) {
                    due.owners.push(owner);
                }
            }
        }
        self.publish(&mut inner);
        Some(due)
    }

    /// `timer_settime` on the timer `link` names, whose clock (`target`)
    /// reads `now`: `new.value` is when it expires, after `now` or
    /// absolutely (`abstime`), or 0 to disarm it. Returns the setting it had,
    /// and whether it expired at once (an expiry already past), as Linux's
    /// `posix_cpu_timer_set` fires one. A new setting drops expirations of
    /// the old one not yet signalled.
    pub(super) fn settime(
        &self,
        link: &Arc<Link>,
        target: Target,
        now: u64,
        new: Setting,
        abstime: bool,
        sigev_none: bool,
    ) -> (Setting, bool) {
        let mut inner = self.inner.lock().unwrap();
        let i = if let Some(i) = inner
            .entries
            .iter()
            .position(|e| Arc::ptr_eq(&e.link, link))
        {
            i
        } else {
            inner.entries.push(Entry {
                link: Arc::clone(link),
                target,
                expires: 0,
                interval: 0,
                armed: false,
                sigev_none,
            });
            inner.entries.len() - 1
        };
        let e = &mut inner.entries[i];
        let old = e.get(now);
        link.fired.store(0, Relaxed);
        let expires = match new.value {
            0 => 0,
            v if abstime => v,
            v => v.saturating_add(now),
        };
        e.expires = expires;
        e.interval = if expires == 0 { 0 } else { new.interval };
        e.armed = !sigev_none && expires != 0 && now < expires;
        let fire_now = !sigev_none && expires != 0 && now >= expires;
        if fire_now {
            e.fire(now);
        }
        self.publish(&mut inner);
        (old, fire_now)
    }

    /// `timer_gettime` on the timer `link` names, whose clock reads `now`.
    pub(super) fn gettime(&self, link: &Arc<Link>, now: u64) -> Setting {
        let mut inner = self.inner.lock().unwrap();
        inner
            .entries
            .iter_mut()
            .find(|e| Arc::ptr_eq(&e.link, link))
            .map_or_else(Setting::default, |e| e.get(now))
    }

    /// `timer_delete` of the timer `link` names.
    pub(super) fn delete(&self, link: &Arc<Link>) {
        let mut inner = self.inner.lock().unwrap();
        inner.entries.retain(|e| !Arc::ptr_eq(&e.link, link));
        self.publish(&mut inner);
    }

    /// Thread `tid` has exited: the timers on its clock stay, but won't
    /// expire (Linux's `posix_cpu_timers_exit`).
    pub(super) fn thread_gone(&self, tid: i32) {
        let mut inner = self.inner.lock().unwrap();
        for e in &mut inner.entries {
            if e.target == Target::Thread(tid) {
                e.armed = false;
            }
        }
        self.publish(&mut inner);
    }

    /// `setitimer(ITIMER_VIRTUAL or ITIMER_PROF)` (`which` 0 or 1), with
    /// the process's CPU clock reading `now`, as Linux's `set_cpu_itimer`:
    /// a tick more than asked for, and the old setting with a timer about to
    /// fire reading a tick left.
    pub(super) fn set_itimer(&self, which: usize, now: u64, new: Setting) -> Setting {
        let mut inner = self.inner.lock().unwrap();
        let it = &mut inner.itimers[which];
        let mut old = it.expires;
        let mut value = new.value;
        if old != 0 || value != 0 {
            if value > 0 {
                value = value.saturating_add(TICK_NSEC).saturating_add(now);
            }
            if old != 0 {
                old = if old <= now { TICK_NSEC } else { old - now };
            }
        }
        let was = Setting {
            value: old,
            interval: it.incr,
        };
        *it = ITimer {
            expires: value,
            incr: new.interval,
        };
        self.publish(&mut inner);
        was
    }

    /// `getitimer(ITIMER_VIRTUAL or ITIMER_PROF)`, as Linux's
    /// `get_cpu_itimer`.
    pub(super) fn get_itimer(&self, which: usize, now: u64) -> Setting {
        let it = self.inner.lock().unwrap().itimers[which];
        let value = match it.expires {
            0 => 0,
            e if e < now => TICK_NSEC,
            e => e - now,
        };
        Setting {
            value,
            interval: it.incr,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setting(value: u64, interval: u64) -> Setting {
        Setting { value, interval }
    }

    #[test]
    fn a_process_timer_expires_on_the_group_clock_and_rearms() {
        let t = CpuTimers::default();
        let link = Link::new(7);
        assert_eq!(t.charge(1_000, 1, 1_000), None, "nothing armed");
        let (old, now) = t.settime(&link, Target::Group, 100, setting(50, 20), false, false);
        assert_eq!((old, now), (Setting::default(), false));
        assert_eq!(t.gettime(&link, 120), setting(30, 20));
        assert_eq!(t.charge(149, 1, 0), None, "not yet");
        // 150, 170, 190: three expirations at 195, next at 210.
        let due = t.charge(195, 1, 0).unwrap();
        assert_eq!(due.owners, vec![7]);
        assert_eq!(link.fired.load(Relaxed), 3);
        assert_eq!(t.gettime(&link, 200), setting(10, 20));
        // Deleted, it's forgotten.
        t.delete(&link);
        assert_eq!(t.charge(10_000, 1, 0), None);
    }

    #[test]
    fn a_thread_timer_counts_only_its_threads_time() {
        let t = CpuTimers::default();
        let link = Link::new(3);
        t.settime(&link, Target::Thread(5), 0, setting(100, 0), false, false);
        assert_eq!(t.charge(0, 5, 99), None);
        // Another thread's time, however much, isn't this clock's.
        assert_eq!(t.charge(1_000, 6, 1_000), Some(Due::default()));
        assert_eq!(link.fired.load(Relaxed), 0);
        assert_eq!(t.charge(1_000, 5, 100).unwrap().owners, vec![3]);
        assert_eq!(link.fired.load(Relaxed), 1);
        // A one-shot timer is disarmed once expired.
        assert_eq!(t.gettime(&link, 200), Setting::default());
        // Its thread gone, a re-armed timer never expires.
        t.settime(&link, Target::Thread(5), 0, setting(10, 0), false, false);
        t.thread_gone(5);
        assert_eq!(t.charge(0, 5, 1_000), None);
    }

    #[test]
    fn an_expiry_already_past_fires_at_once_and_sigev_none_never_does() {
        let t = CpuTimers::default();
        let link = Link::new(1);
        let (_, now) = t.settime(&link, Target::Group, 500, setting(400, 0), true, false);
        assert!(now);
        assert_eq!(link.fired.load(Relaxed), 1);
        let none = Link::new(1);
        let (_, now) = t.settime(&none, Target::Group, 500, setting(100, 0), false, true);
        assert!(!now);
        assert_eq!(t.charge(10_000, 1, 0), None, "SIGEV_NONE is never queued");
        // Expired, a one-shot SIGEV_NONE timer reads 0 left; a periodic one
        // moves on.
        assert_eq!(t.gettime(&none, 700), setting(0, 0));
        t.settime(&none, Target::Group, 500, setting(100, 30), false, true);
        assert_eq!(t.gettime(&none, 650), setting(10, 30));
        // The old setting comes back from a new one.
        let (old, _) = t.settime(&none, Target::Group, 655, setting(0, 0), false, true);
        assert_eq!(old, setting(5, 30));
    }

    #[test]
    fn itimers_signal_the_process_a_tick_late() {
        let t = CpuTimers::default();
        let old = t.set_itimer(1, 1_000, setting(10_000_000, 5_000_000));
        assert_eq!(old, Setting::default());
        assert_eq!(
            t.get_itimer(1, 1_000),
            setting(10_000_000 + TICK_NSEC, 5_000_000)
        );
        let at = 1_000 + 10_000_000 + TICK_NSEC;
        assert_eq!(t.charge(at - 1, 1, 0), None);
        assert_eq!(t.charge(at, 1, 0).unwrap().signals, vec![SIGPROF]);
        assert_eq!(t.get_itimer(1, at), setting(5_000_000, 5_000_000));
        // About to fire reads a tick; setting none disarms it.
        assert_eq!(t.get_itimer(1, at + 6_000_000).value, TICK_NSEC);
        let old = t.set_itimer(1, at + 6_000_000, Setting::default());
        assert_eq!(old.value, TICK_NSEC);
        assert_eq!(t.charge(u64::MAX - 1, 1, 0), None);
    }
}
