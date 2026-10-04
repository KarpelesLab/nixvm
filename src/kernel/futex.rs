//! Futexes: `futex(2)` (wait/wake/requeue/wake-op, bitsets, timeouts, and the
//! priority-inheritance lock ops), `futex_waitv`, and the futex2 calls
//! (`futex_wake`/`futex_wait`/`futex_requeue`).
//!
//! A waiter parks on a [`FutexKey`] and re-traps its syscall until a wake flips
//! its `futex_woken` flag (the classic re-trap convention of this kernel). Each
//! re-run re-reads the futex word — a changed word ends the wait even if the
//! wake itself was "lost" — and honors the wait's timeout through the
//! scheduler's timed-wait deadline, so `pthread_cond_timedwait`,
//! `sem_timedwait` and friends return `ETIMEDOUT` instead of sleeping forever.
//!
//! Keys follow Linux: a futex word on a page shared across address spaces
//! (`MAP_SHARED`, SysV shm) is keyed by its *physical* address unless the call
//! says `FUTEX_PRIVATE_FLAG`, so a process-shared mutex/semaphore/barrier works
//! between processes that map it at different addresses; everything else is
//! keyed by `(address space, address)`.

use super::{Kernel, RunState, ServiceCtx, Shared, err, poll};
use crate::abi::errno::Errno;
use crate::vcpu::GuestMemory;

const FUTEX_WAIT: u64 = 0;
const FUTEX_WAKE: u64 = 1;
const FUTEX_REQUEUE: u64 = 3;
const FUTEX_CMP_REQUEUE: u64 = 4;
const FUTEX_WAKE_OP: u64 = 5;
const FUTEX_LOCK_PI: u64 = 6;
const FUTEX_UNLOCK_PI: u64 = 7;
const FUTEX_TRYLOCK_PI: u64 = 8;
const FUTEX_WAIT_BITSET: u64 = 9;
const FUTEX_WAKE_BITSET: u64 = 10;
const FUTEX_WAIT_REQUEUE_PI: u64 = 11;
const FUTEX_LOCK_PI2: u64 = 13;
const FUTEX_PRIVATE_FLAG: u64 = 128;
const FUTEX_CLOCK_REALTIME: u64 = 256;
const BITSET_ALL: u32 = u32::MAX;
/// The PI futex word: owner tid in the low 30 bits, plus two state bits.
const FUTEX_TID_MASK: u32 = 0x3fff_ffff;
const FUTEX_WAITERS: u32 = 0x8000_0000;
const FUTEX_OWNER_DIED: u32 = 0x4000_0000;
/// futex2 flags: the word size (only 32-bit is supported, as on Linux) and the
/// private bit.
const FUTEX2_SIZE_U32: u64 = 0x02;
const FUTEX2_SIZE_MASK: u64 = 0x03;
const FUTEX2_PRIVATE: u64 = 128;
/// `FUTEX_WAITV_MAX`.
const WAITV_MAX: u64 = 128;

/// What a futex waiter is parked on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FutexKey {
    /// A word private to one address space: `(mm, address)`.
    Private(usize, u64),
    /// A word on a page shared between address spaces: its physical address.
    Shared(u64),
}

/// How a futex wait's timeout is expressed.
#[derive(Clone, Copy)]
enum Timeout {
    /// `struct timespec *` relative to now (`FUTEX_WAIT`); NULL (0) = none.
    Relative(u64),
    /// `struct timespec *` absolute on `CLOCK_REALTIME` (`true`) or
    /// `CLOCK_MONOTONIC` (`false`).
    Absolute(u64, bool),
}

impl Kernel {
    /// The key of the futex word at `uaddr` for the caller.
    pub(super) fn futex_key(
        cx: &ServiceCtx,
        mem: &GuestMemory,
        uaddr: u64,
        private: bool,
    ) -> FutexKey {
        match (private, mem.shared_phys(uaddr)) {
            (false, Some(pa)) => FutexKey::Shared(pa),
            _ => FutexKey::Private(cx.cur.mm, uaddr),
        }
    }

    /// Whether any *other* live task could wake a waiter on `key`: a sibling
    /// sharing the address space for a private word, any other process for a
    /// shared one. With nobody, an untimed wait would be a false deadlock.
    fn futex_could_wake(sh: &Shared, key: FutexKey) -> bool {
        sh.procs.iter().flatten().any(|p| {
            !matches!(p.info.run, RunState::Zombie(_))
                && match key {
                    FutexKey::Private(mm, _) => p.info.mm == mm,
                    FutexKey::Shared(_) => true,
                }
        })
    }

    /// Whether any *other* live task shares address space `mm` (`cx.cur` is
    /// out of the table during its slice, so this scans only the siblings).
    #[allow(clippy::unused_self)]
    pub(super) fn has_cowaiter(&self, sh: &Shared, mm: usize) -> bool {
        Self::futex_could_wake(sh, FutexKey::Private(mm, 0))
    }

    /// Seed (on the first run of a timed wait) or fetch the wall-clock
    /// deadline of the caller's futex wait. `Err` for a malformed timespec.
    fn futex_deadline(
        cx: &mut ServiceCtx,
        t: Timeout,
        mem: &GuestMemory,
    ) -> Result<Option<u128>, i64> {
        let (ptr, abs, realtime) = match t {
            Timeout::Relative(0) | Timeout::Absolute(0, _) => return Ok(None),
            Timeout::Relative(p) => (p, false, false),
            Timeout::Absolute(p, rt) => (p, true, rt),
        };
        if let Some(dl) = cx.cur.wake_deadline {
            return Ok(Some(dl));
        }
        let (Ok(s), Ok(n)) = (mem.read_u64(ptr), mem.read_u64(ptr + 8)) else {
            return Err(err(Errno::EFAULT));
        };
        if (s as i64) < 0 || n >= 1_000_000_000 {
            return Err(err(Errno::EINVAL));
        }
        let want = u128::from(s) * 1_000_000_000 + u128::from(n);
        let now = poll::now_ns();
        let dl = if !abs {
            now + want
        } else if realtime {
            want
        } else {
            let mono = crate::clock::now_monotonic().as_nanos();
            now + want.saturating_sub(mono)
        };
        cx.cur.wake_deadline = Some(dl);
        Ok(Some(dl))
    }

    /// Forget the caller's futex wait (it returned).
    fn futex_done(cx: &mut ServiceCtx) {
        cx.cur.futex_wait = None;
        cx.cur.futex_waitv.clear();
        cx.cur.futex_woken = false;
        cx.cur.futex_pi = false;
    }

    /// `futex(uaddr, op, val, timeout | val2, uaddr2, val3)`.
    #[allow(clippy::unused_self, clippy::too_many_lines)]
    pub(super) fn sys_futex(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        a: &[u64; 6],
        mem: &mut GuestMemory,
    ) -> i64 {
        let (uaddr, op, val) = (a[0], a[1], a[2] as u32);
        let cmd = op & 0x7f;
        let private = op & FUTEX_PRIVATE_FLAG != 0;
        let realtime = op & FUTEX_CLOCK_REALTIME != 0;
        // CLOCK_REALTIME is only meaningful for the absolute-timeout waits.
        if realtime
            && !matches!(
                cmd,
                FUTEX_WAIT_BITSET | FUTEX_WAIT_REQUEUE_PI | FUTEX_LOCK_PI2
            )
        {
            return err(Errno::ENOSYS);
        }
        if !uaddr.is_multiple_of(4) {
            return err(Errno::EINVAL);
        }
        let key = Self::futex_key(cx, mem, uaddr, private);
        match cmd {
            FUTEX_WAIT => self.futex_wait(
                sh,
                cx,
                mem,
                key,
                uaddr,
                val,
                BITSET_ALL,
                Timeout::Relative(a[3]),
            ),
            FUTEX_WAIT_BITSET => {
                let bitset = a[5] as u32;
                if bitset == 0 {
                    return err(Errno::EINVAL);
                }
                self.futex_wait(
                    sh,
                    cx,
                    mem,
                    key,
                    uaddr,
                    val,
                    bitset,
                    Timeout::Absolute(a[3], realtime),
                )
            }
            FUTEX_WAKE => Self::futex_wake_key(sh, key, i64::from(val as i32), BITSET_ALL),
            FUTEX_WAKE_BITSET => {
                let bitset = a[5] as u32;
                if bitset == 0 {
                    return err(Errno::EINVAL);
                }
                Self::futex_wake_key(sh, key, i64::from(val as i32), bitset)
            }
            // Requeue: wake up to `val` waiters on `uaddr`, then move up to
            // `val2` of the rest to wait on `uaddr2` instead — how
            // pthread_cond_signal/broadcast hand woken threads to the mutex.
            FUTEX_REQUEUE | FUTEX_CMP_REQUEUE => {
                if cmd == FUTEX_CMP_REQUEUE {
                    match mem.read_u32(uaddr) {
                        Ok(cur) if cur != a[5] as u32 => return err(Errno::EAGAIN),
                        Err(_) => return err(Errno::EFAULT),
                        Ok(_) => {}
                    }
                }
                let to = Self::futex_key(cx, mem, a[4], private);
                Self::futex_requeue_keys(sh, key, to, i64::from(val as i32), a[3] as i64)
            }
            FUTEX_WAKE_OP => self.futex_wake_op(sh, cx, a, key, mem),
            FUTEX_LOCK_PI | FUTEX_LOCK_PI2 | FUTEX_TRYLOCK_PI => {
                let timeout = if cmd == FUTEX_TRYLOCK_PI {
                    None
                } else {
                    // LOCK_PI's timeout is absolute CLOCK_REALTIME; LOCK_PI2's
                    // is CLOCK_MONOTONIC unless FUTEX_CLOCK_REALTIME.
                    Some(Timeout::Absolute(a[3], cmd == FUTEX_LOCK_PI || realtime))
                };
                self.futex_lock_pi(sh, cx, mem, key, uaddr, timeout)
            }
            FUTEX_UNLOCK_PI => self.futex_unlock_pi(sh, cx, mem, key, uaddr),
            // Requeue-to-PI (condvars over PI mutexes — glibc's condvar no
            // longer uses it) is not provided; neither is anything unknown.
            _ => {
                self.note_unsupported("futex", cmd);
                err(Errno::ENOSYS)
            }
        }
    }

    /// The common wait: return 0 once woken, `EAGAIN` if the word no longer
    /// holds `val`, `ETIMEDOUT` past the deadline; otherwise park on `key`.
    #[allow(clippy::unused_self, clippy::too_many_arguments)]
    fn futex_wait(
        &self,
        sh: &Shared,
        cx: &mut ServiceCtx,
        mem: &GuestMemory,
        key: FutexKey,
        uaddr: u64,
        val: u32,
        bitset: u32,
        timeout: Timeout,
    ) -> i64 {
        // Woken by an explicit wake (directly, or after being requeued to
        // another word by a condvar signal): consume it.
        if cx.cur.futex_woken {
            Self::futex_done(cx);
            return 0;
        }
        let deadline = match Self::futex_deadline(cx, timeout, mem) {
            Ok(d) => d,
            Err(e) => return e,
        };
        if deadline.is_some_and(|d| poll::now_ns() >= d) {
            Self::futex_done(cx);
            return err(Errno::ETIMEDOUT);
        }
        // Parked on a *different* word than this call names: requeued. Stay
        // parked; only a wake on the requeue target releases us (re-comparing
        // this word would spuriously EAGAIN and desync the condvar).
        if cx.cur.futex_wait.is_some_and(|w| w != key) {
            cx.block = true;
            return 0;
        }
        match mem.read_u32(uaddr) {
            Err(_) => {
                Self::futex_done(cx);
                err(Errno::EFAULT)
            }
            // The word moved on: the wait is over (this also catches a wake
            // that raced ahead of the park).
            Ok(cur) if cur != val => {
                Self::futex_done(cx);
                err(Errno::EAGAIN)
            }
            // Nobody could ever wake an untimed wait: report a spurious wake
            // (the futex contract permits it; callers re-check and loop)
            // instead of a false deadlock — the single-threaded case.
            Ok(_) if deadline.is_none() && !Self::futex_could_wake(sh, key) => {
                Self::futex_done(cx);
                0
            }
            Ok(_) => {
                cx.cur.futex_wait = Some(key);
                cx.cur.futex_bitset = bitset;
                cx.block = true;
                0
            }
        }
    }

    /// Wake up to `n` waiters parked on `key` whose bitset intersects
    /// `bitset` (plain waits, and `futex_waitv` waits listing the key).
    /// Returns how many were woken.
    pub(super) fn futex_wake_key(sh: &mut Shared, key: FutexKey, n: i64, bitset: u32) -> i64 {
        let mut woken = 0i64;
        for p in sh.procs.iter_mut().flatten() {
            if woken >= n {
                break;
            }
            let i = &mut p.info;
            if i.futex_woken || i.futex_pi {
                continue;
            }
            if i.futex_wait == Some(key) && i.futex_bitset & bitset != 0 {
                i.futex_woken = true;
            } else if let Some(idx) = i.futex_waitv.iter().position(|k| *k == key) {
                i.futex_woken = true;
                i.futex_waitv_idx = idx;
            } else {
                continue;
            }
            i.parked = false; // make it runnable so the sweep re-runs it
            woken += 1;
        }
        woken
    }

    /// Wake up to `nr_wake` waiters on `from`, then requeue up to
    /// `nr_requeue` of the rest onto `to`. Returns the number woken.
    fn futex_requeue_keys(
        sh: &mut Shared,
        from: FutexKey,
        to: FutexKey,
        nr_wake: i64,
        nr_requeue: i64,
    ) -> i64 {
        let mut woken = 0i64;
        let mut requeued = 0i64;
        for p in sh.procs.iter_mut().flatten() {
            if p.info.futex_wait != Some(from) || p.info.futex_woken {
                continue;
            }
            if woken < nr_wake {
                p.info.futex_woken = true;
                p.info.parked = false;
                woken += 1;
            } else if requeued < nr_requeue {
                // Move it to the new word; it stays parked until a wake there.
                p.info.futex_wait = Some(to);
                requeued += 1;
            } else {
                break;
            }
        }
        woken
    }

    /// `FUTEX_WAKE_OP`: atomically apply the encoded operation to `*uaddr2`,
    /// wake up to `val` waiters on `uaddr`, and — if the old `*uaddr2` passes
    /// the encoded comparison — up to `val2` waiters on `uaddr2` too.
    #[allow(clippy::unused_self)]
    fn futex_wake_op(
        &self,
        sh: &mut Shared,
        cx: &ServiceCtx,
        a: &[u64; 6],
        key: FutexKey,
        mem: &mut GuestMemory,
    ) -> i64 {
        let (val, val2, uaddr2, enc) = (a[2] as i32, a[3] as i32, a[4], a[5] as u32);
        let private = a[1] & FUTEX_PRIVATE_FLAG != 0;
        let mut op = (enc >> 28) & 0xf;
        let cmp = (enc >> 24) & 0xf;
        let mut oparg = ((enc << 8) as i32) >> 20; // sign-extended 12 bits
        let cmparg = ((enc << 20) as i32) >> 20;
        if op & 8 != 0 {
            // FUTEX_OP_OPARG_SHIFT: the argument is a shift count.
            op &= 7;
            if !(0..32).contains(&oparg) {
                return err(Errno::EINVAL);
            }
            oparg = 1 << oparg;
        }
        let Ok(old) = mem.read_u32(uaddr2) else {
            return err(Errno::EFAULT);
        };
        let old_i = old as i32;
        let new = match op {
            0 => oparg,                     // SET
            1 => old_i.wrapping_add(oparg), // ADD
            2 => old_i | oparg,             // OR
            3 => old_i & !oparg,            // ANDN
            4 => old_i ^ oparg,             // XOR
            _ => return err(Errno::ENOSYS),
        };
        let pass = match cmp {
            0 => old_i == cmparg,
            1 => old_i != cmparg,
            2 => old_i < cmparg,
            3 => old_i <= cmparg,
            4 => old_i > cmparg,
            5 => old_i >= cmparg,
            _ => return err(Errno::ENOSYS),
        };
        if mem.write(uaddr2, &(new as u32).to_le_bytes()).is_err() {
            return err(Errno::EFAULT);
        }
        let mut n = Self::futex_wake_key(sh, key, i64::from(val), BITSET_ALL);
        if pass {
            let key2 = Self::futex_key(cx, mem, uaddr2, private);
            n += Self::futex_wake_key(sh, key2, i64::from(val2), BITSET_ALL);
        }
        n
    }

    /// Whether a task with tid `tid` is alive (a PI owner that died leaves the
    /// lock to be taken over with `FUTEX_OWNER_DIED`).
    fn tid_alive(sh: &Shared, cx: &ServiceCtx, tid: i32) -> bool {
        cx.cur.pid == tid
            || sh
                .procs
                .iter()
                .flatten()
                .any(|p| p.info.pid == tid && !matches!(p.info.run, RunState::Zombie(_)))
    }

    /// `FUTEX_LOCK_PI`/`LOCK_PI2`/`TRYLOCK_PI`: take the PI lock word at
    /// `uaddr` (owner tid in the low bits). Free → ours; held by us →
    /// `EDEADLK`; held by a dead task → ours with `FUTEX_OWNER_DIED`; held
    /// otherwise → `EAGAIN` for trylock, else mark `FUTEX_WAITERS` and park
    /// until `FUTEX_UNLOCK_PI` hands the lock over (or the timeout passes).
    /// There is no priority to inherit under the cooperative scheduler, so
    /// "PI" is just the hand-off protocol.
    #[allow(clippy::unused_self)]
    fn futex_lock_pi(
        &self,
        sh: &Shared,
        cx: &mut ServiceCtx,
        mem: &mut GuestMemory,
        key: FutexKey,
        uaddr: u64,
        timeout: Option<Timeout>,
    ) -> i64 {
        let tid = cx.cur.pid as u32;
        if cx.cur.futex_woken {
            // The unlocker already wrote our tid into the word.
            Self::futex_done(cx);
            return 0;
        }
        let Ok(word) = mem.read_u32(uaddr) else {
            Self::futex_done(cx);
            return err(Errno::EFAULT);
        };
        let owner = word & FUTEX_TID_MASK;
        let waiters_left = sh
            .procs
            .iter()
            .flatten()
            .any(|p| p.info.futex_pi && p.info.futex_wait == Some(key) && !p.info.futex_woken);
        let take = |extra: u32| tid | extra | if waiters_left { FUTEX_WAITERS } else { 0 };
        let acquired = if owner == 0 {
            Some(take(0))
        } else if owner == tid {
            Self::futex_done(cx);
            return err(Errno::EDEADLK);
        } else if !Self::tid_alive(sh, cx, owner as i32) {
            Some(take(FUTEX_OWNER_DIED))
        } else {
            None
        };
        if let Some(w) = acquired {
            Self::futex_done(cx);
            return if mem.write(uaddr, &w.to_le_bytes()).is_ok() {
                0
            } else {
                err(Errno::EFAULT)
            };
        }
        let Some(timeout) = timeout else {
            return err(Errno::EAGAIN); // trylock on a held lock
        };
        let deadline = match Self::futex_deadline(cx, timeout, mem) {
            Ok(d) => d,
            Err(e) => return e,
        };
        if deadline.is_some_and(|d| poll::now_ns() >= d) {
            Self::futex_done(cx);
            return err(Errno::ETIMEDOUT);
        }
        if word & FUTEX_WAITERS == 0
            && mem
                .write(uaddr, &(word | FUTEX_WAITERS).to_le_bytes())
                .is_err()
        {
            return err(Errno::EFAULT);
        }
        cx.cur.futex_wait = Some(key);
        cx.cur.futex_pi = true;
        cx.block = true;
        0
    }

    /// `FUTEX_UNLOCK_PI`: release a PI lock the caller owns (`EPERM`
    /// otherwise), handing it directly to the first waiter (its tid, plus
    /// `FUTEX_WAITERS` while more remain) or clearing the word.
    #[allow(clippy::unused_self)]
    fn futex_unlock_pi(
        &self,
        sh: &mut Shared,
        cx: &ServiceCtx,
        mem: &mut GuestMemory,
        key: FutexKey,
        uaddr: u64,
    ) -> i64 {
        let Ok(word) = mem.read_u32(uaddr) else {
            return err(Errno::EFAULT);
        };
        if word & FUTEX_TID_MASK != cx.cur.pid as u32 {
            return err(Errno::EPERM);
        }
        let waiting: Vec<usize> = sh
            .procs
            .iter()
            .enumerate()
            .filter_map(|(i, p)| {
                let p = p.as_ref()?;
                (p.info.futex_pi && p.info.futex_wait == Some(key) && !p.info.futex_woken)
                    .then_some(i)
            })
            .collect();
        let new = match waiting.first() {
            Some(&i) => {
                let p = &mut sh.procs[i].as_mut().expect("present").info;
                p.futex_woken = true;
                p.parked = false;
                p.pid as u32 | if waiting.len() > 1 { FUTEX_WAITERS } else { 0 }
            }
            None => 0,
        };
        if mem.write(uaddr, &new.to_le_bytes()).is_err() {
            return err(Errno::EFAULT);
        }
        0
    }

    /// Decode one `struct futex_waitv { u64 val; u64 uaddr; u32 flags; u32
    /// __reserved; }` (or a futex2 call's own flags): only 32-bit words, an
    /// aligned address, no reserved bits. Returns `(val, uaddr, private)`.
    fn futex2_word(val: u64, uaddr: u64, flags: u64) -> Result<(u32, u64, bool), i64> {
        if flags & !(FUTEX2_SIZE_MASK | FUTEX2_PRIVATE) != 0
            || flags & FUTEX2_SIZE_MASK != FUTEX2_SIZE_U32
            || !uaddr.is_multiple_of(4)
            || val > u64::from(u32::MAX)
        {
            return Err(err(Errno::EINVAL));
        }
        Ok((val as u32, uaddr, flags & FUTEX2_PRIVATE != 0))
    }

    /// `futex_waitv(waiters, nr_futexes, flags, timeout, clockid)`: wait on up
    /// to 128 futexes at once; returns the index of the one that woke us,
    /// `EAGAIN` if any word already differs from its expected value, or
    /// `ETIMEDOUT` at the absolute `clockid` (`REALTIME`/`MONOTONIC`)
    /// deadline.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_futex_waitv(
        &self,
        sh: &Shared,
        cx: &mut ServiceCtx,
        a: &[u64; 6],
        mem: &GuestMemory,
    ) -> i64 {
        let (waiters, nr, flags, timeout, clock) = (a[0], a[1], a[2], a[3], a[4]);
        if flags != 0 || nr == 0 || nr > WAITV_MAX {
            return err(Errno::EINVAL);
        }
        if timeout != 0 && clock > 1 {
            return err(Errno::EINVAL);
        }
        if cx.cur.futex_woken {
            let idx = cx.cur.futex_waitv_idx;
            Self::futex_done(cx);
            return idx as i64;
        }
        let Ok(raw) = mem.read_vec(waiters, (nr * 24) as usize) else {
            return err(Errno::EFAULT);
        };
        let mut words = Vec::with_capacity(nr as usize);
        for c in raw.chunks(24) {
            let u = |o: usize| u64::from_le_bytes(c[o..o + 8].try_into().unwrap());
            let f = u64::from(u32::from_le_bytes(c[16..20].try_into().unwrap()));
            if c[20..24] != [0; 4] {
                return err(Errno::EINVAL);
            }
            match Self::futex2_word(u(0), u(8), f) {
                Ok(w) => words.push(w),
                Err(e) => return e,
            }
        }
        let deadline = match Self::futex_deadline(cx, Timeout::Absolute(timeout, clock == 0), mem) {
            Ok(d) => d,
            Err(e) => return e,
        };
        if deadline.is_some_and(|d| poll::now_ns() >= d) {
            Self::futex_done(cx);
            return err(Errno::ETIMEDOUT);
        }
        let mut keys = Vec::with_capacity(words.len());
        for &(val, uaddr, private) in &words {
            match mem.read_u32(uaddr) {
                Ok(cur) if cur == val => keys.push(Self::futex_key(cx, mem, uaddr, private)),
                Ok(_) => {
                    Self::futex_done(cx);
                    return err(Errno::EAGAIN);
                }
                Err(_) => {
                    Self::futex_done(cx);
                    return err(Errno::EFAULT);
                }
            }
        }
        if deadline.is_none() && !keys.iter().any(|&k| Self::futex_could_wake(sh, k)) {
            Self::futex_done(cx);
            return 0; // spurious wake of index 0, rather than a false deadlock
        }
        cx.cur.futex_waitv = keys;
        cx.block = true;
        0
    }

    /// futex2 `futex_wake(uaddr, mask, nr, flags)`.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_futex2_wake(
        &self,
        sh: &mut Shared,
        cx: &ServiceCtx,
        a: &[u64; 6],
        mem: &GuestMemory,
    ) -> i64 {
        let (uaddr, mask, nr, flags) = (a[0], a[1] as u32, a[2] as i64, a[3]);
        let (_, uaddr, private) = match Self::futex2_word(0, uaddr, flags) {
            Ok(w) => w,
            Err(e) => return e,
        };
        if mask == 0 || nr < 0 {
            return err(Errno::EINVAL);
        }
        let key = Self::futex_key(cx, mem, uaddr, private);
        Self::futex_wake_key(sh, key, nr.min(i64::from(i32::MAX)), mask)
    }

    /// futex2 `futex_wait(uaddr, val, mask, flags, timeout, clockid)`.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_futex2_wait(
        &self,
        sh: &Shared,
        cx: &mut ServiceCtx,
        a: &[u64; 6],
        mem: &GuestMemory,
    ) -> i64 {
        let (uaddr, val, mask, flags, timeout, clock) = (a[0], a[1], a[2] as u32, a[3], a[4], a[5]);
        let (val, uaddr, private) = match Self::futex2_word(val, uaddr, flags) {
            Ok(w) => w,
            Err(e) => return e,
        };
        if mask == 0 || (timeout != 0 && clock > 1) {
            return err(Errno::EINVAL);
        }
        let key = Self::futex_key(cx, mem, uaddr, private);
        self.futex_wait(
            sh,
            cx,
            mem,
            key,
            uaddr,
            val,
            mask,
            Timeout::Absolute(timeout, clock == 0),
        )
    }

    /// futex2 `futex_requeue(waiters[2], flags, nr_wake, nr_requeue)`: the
    /// first `futex_waitv` names the source (its value is compared, as
    /// `FUTEX_CMP_REQUEUE` does), the second the target.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_futex2_requeue(
        &self,
        sh: &mut Shared,
        cx: &ServiceCtx,
        a: &[u64; 6],
        mem: &GuestMemory,
    ) -> i64 {
        let (waiters, flags, nr_wake, nr_requeue) = (a[0], a[1], a[2] as i64, a[3] as i64);
        if flags != 0 || nr_wake < 0 || nr_requeue < 0 {
            return err(Errno::EINVAL);
        }
        let Ok(raw) = mem.read_vec(waiters, 48) else {
            return err(Errno::EFAULT);
        };
        let mut w = Vec::with_capacity(2);
        for c in raw.chunks(24) {
            let u = |o: usize| u64::from_le_bytes(c[o..o + 8].try_into().unwrap());
            let f = u64::from(u32::from_le_bytes(c[16..20].try_into().unwrap()));
            match Self::futex2_word(u(0), u(8), f) {
                Ok(x) => w.push(x),
                Err(e) => return e,
            }
        }
        match mem.read_u32(w[0].1) {
            Ok(cur) if cur != w[0].0 => return err(Errno::EAGAIN),
            Err(_) => return err(Errno::EFAULT),
            Ok(_) => {}
        }
        let from = Self::futex_key(cx, mem, w[0].1, w[0].2);
        let to = Self::futex_key(cx, mem, w[1].1, w[1].2);
        Self::futex_requeue_keys(sh, from, to, nr_wake, nr_requeue)
    }
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{BASE, call, e, setup};
    use super::super::{ProcInfo, Process, RunState};
    use super::FutexKey;
    use crate::abi::arch::Sysno;
    use crate::abi::errno::Errno;

    /// A sibling thread (pid 2, same mm) so waits can park.
    fn add_thread(k: &super::Kernel) {
        let info = ProcInfo {
            pid: 2,
            tgid: 1,
            mm: 0,
            run: RunState::Running,
            ..ProcInfo::default()
        };
        k.shared
            .lock()
            .unwrap()
            .procs
            .push(Some(Process { vcpu: None, info }));
    }

    #[test]
    fn timed_wait_times_out_and_untimed_parks() {
        let (k, mut mem, mut v, mut cx) = setup();
        let (word, ts) = (BASE, BASE + 0x100);
        mem.write(word, &5u32.to_le_bytes()).unwrap();
        // Relative timeout of 0 ns: already expired.
        mem.write(ts, &[0u8; 16]).unwrap();
        mem.write(ts + 8, &1u64.to_le_bytes()).unwrap();
        // First run seeds the deadline and parks (a sibling could wake it)…
        add_thread(&k);
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Futex,
                [word, 0x80, 5, ts, 0, 0]
            ),
            0
        );
        assert!(cx.block);
        std::thread::sleep(std::time::Duration::from_millis(1));
        // …and the re-trap after the deadline reports ETIMEDOUT.
        cx.block = false;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Futex,
                [word, 0x80, 5, ts, 0, 0]
            ),
            e(Errno::ETIMEDOUT)
        );
        assert_eq!(cx.cur.futex_wait, None);
        // A value mismatch is EAGAIN; an unaligned word EINVAL; unknown op ENOSYS.
        cx.cur.wake_deadline = None;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Futex,
                [word, 0x80, 4, 0, 0, 0]
            ),
            e(Errno::EAGAIN)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Futex,
                [word + 1, 0x80, 5, 0, 0, 0]
            ),
            e(Errno::EINVAL)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Futex,
                [word, 0x80 | 0x63, 5, 0, 0, 0]
            ),
            e(Errno::ENOSYS)
        );
    }

    #[test]
    fn wake_op_modifies_and_conditionally_wakes() {
        let (k, mut mem, mut v, mut cx) = setup();
        let (w1, w2) = (BASE, BASE + 4);
        mem.write(w2, &3u32.to_le_bytes()).unwrap();
        // Park the sibling on w2.
        {
            let mut sh = k.shared.lock().unwrap();
            let info = ProcInfo {
                pid: 2,
                tgid: 1,
                mm: 0,
                run: RunState::Running,
                futex_wait: Some(FutexKey::Private(0, w2)),
                futex_bitset: u32::MAX,
                ..ProcInfo::default()
            };
            sh.procs.push(Some(Process { vcpu: None, info }));
        }
        // op = ADD 1, cmp = EQ 3: *w2 becomes 4, and since old == 3, wake 1 on w2.
        let enc: u64 = (1 << 28) | (1 << 12) | 3; // cmp EQ (0) in bits 24..28
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Futex,
                [w1, 0x80 | 5, 1, 1, w2, enc]
            ),
            1
        );
        assert_eq!(mem.read_u32(w2).unwrap(), 4);
        let sh = k.shared.lock().unwrap();
        assert!(
            sh.procs
                .iter()
                .flatten()
                .any(|p| p.info.pid == 2 && p.info.futex_woken)
        );
    }

    #[test]
    fn pi_lock_trylock_unlock_handoff() {
        let (k, mut mem, mut v, mut cx) = setup();
        let w = BASE;
        mem.write(w, &0u32.to_le_bytes()).unwrap();
        // Lock a free word: it becomes our tid (1).
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Futex,
                [w, 6, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_u32(w).unwrap(), 1);
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Futex,
                [w, 8, 0, 0, 0, 0]
            ),
            e(Errno::EDEADLK)
        );
        // A sibling (pid 2) parked as a PI waiter; unlock hands it the lock.
        {
            let mut sh = k.shared.lock().unwrap();
            let info = ProcInfo {
                pid: 2,
                tgid: 1,
                mm: 0,
                run: RunState::Running,
                futex_wait: Some(FutexKey::Private(0, w)),
                futex_pi: true,
                ..ProcInfo::default()
            };
            sh.procs.push(Some(Process { vcpu: None, info }));
        }
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Futex,
                [w, 7, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_u32(w).unwrap(), 2, "owner is now the waiter");
        // We no longer own it: unlock is EPERM, trylock EAGAIN.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Futex,
                [w, 7, 0, 0, 0, 0]
            ),
            e(Errno::EPERM)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Futex,
                [w, 8, 0, 0, 0, 0]
            ),
            e(Errno::EAGAIN)
        );
        // An owner tid that isn't alive: taken over with OWNER_DIED.
        mem.write(w, &77u32.to_le_bytes()).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Futex,
                [w, 8, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_u32(w).unwrap(), 1 | 0x4000_0000);
    }

    #[test]
    fn waitv_and_futex2() {
        let (k, mut mem, mut v, mut cx) = setup();
        add_thread(&k);
        let (wv, a, b) = (BASE, BASE + 0x100, BASE + 0x104);
        mem.write(a, &1u32.to_le_bytes()).unwrap();
        mem.write(b, &2u32.to_le_bytes()).unwrap();
        let mut ent = [0u8; 48];
        ent[0..8].copy_from_slice(&1u64.to_le_bytes());
        ent[8..16].copy_from_slice(&a.to_le_bytes());
        ent[16..20].copy_from_slice(&(2u32 | 128).to_le_bytes());
        ent[24..32].copy_from_slice(&2u64.to_le_bytes());
        ent[32..40].copy_from_slice(&b.to_le_bytes());
        ent[40..44].copy_from_slice(&(2u32 | 128).to_le_bytes());
        mem.write(wv, &ent).unwrap();
        // Bad flags / sizes.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::FutexWaitv,
                [wv, 2, 1, 0, 0, 0]
            ),
            e(Errno::EINVAL)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::FutexWaitv,
                [wv, 0, 0, 0, 0, 0]
            ),
            e(Errno::EINVAL)
        );
        // Park on both; a futex2 wake on `b` releases index 1.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::FutexWaitv,
                [wv, 2, 0, 0, 0, 0]
            ),
            0
        );
        assert!(cx.block);
        cx.block = false;
        // Simulate the waker by putting ourselves in the table's view: use
        // the helper directly against our own parked state.
        {
            let mut sh = k.shared.lock().unwrap();
            let me = ProcInfo {
                pid: 3,
                tgid: 1,
                mm: 0,
                run: RunState::Running,
                futex_waitv: cx.cur.futex_waitv.clone(),
                ..ProcInfo::default()
            };
            sh.procs.push(Some(Process {
                vcpu: None,
                info: me,
            }));
        }
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::FutexWake,
                [b, u64::from(u32::MAX), 1, 2 | 128, 0, 0]
            ),
            1
        );
        let sh = k.shared.lock().unwrap();
        let p = sh.procs.iter().flatten().find(|p| p.info.pid == 3).unwrap();
        assert!(p.info.futex_woken);
        assert_eq!(p.info.futex_waitv_idx, 1);
        drop(sh);
        // A mismatching value: EAGAIN. futex2 wait with a non-u32 size: EINVAL.
        mem.write(a, &9u32.to_le_bytes()).unwrap();
        cx.cur.futex_waitv.clear();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::FutexWaitv,
                [wv, 2, 0, 0, 0, 0]
            ),
            e(Errno::EAGAIN)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::FutexWait,
                [a, 9, u64::from(u32::MAX), 3, 0, 0]
            ),
            e(Errno::EINVAL)
        );
    }
}
