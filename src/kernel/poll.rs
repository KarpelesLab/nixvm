//! Event-notification / readiness syscalls: `poll`, `ppoll`, `select`,
//! `pselect6`, `epoll_create1`/`epoll_ctl`/`epoll_wait`/`epoll_pwait`,
//! `eventfd2`, and `timerfd_create`/`timerfd_settime`/`timerfd_gettime`.
//!
//! Readiness is computed synchronously against the existing pollable fd kinds
//! (pipes, sockets, eventfds, timerfds) via [`Kernel::fd_ready`]. A caller with
//! a zero timeout always gets an immediate answer; a caller willing to wait
//! (nonzero or infinite timeout) that finds nothing ready sets `self.block`
//! and re-traps, exactly like `read_pipe`/`accept4`/`wait4` — the scheduler
//! retries the same syscall once another process makes progress. There is no
//! wall-clock-driven wakeup (nixvm's cooperative scheduler has none), so a
//! finite timeout behaves like an infinite one when nothing else in the VM
//! ever changes the awaited state; this mirrors the rest of the kernel's
//! blocking primitives.
//!
//! Socket readiness is necessarily best-effort: `net.rs` is a sibling module
//! and its connection state is private to it, so a connected socket fd is
//! reported as always read/write ready rather than risk incorrectly blocking
//! forever on state this module cannot observe.

use std::collections::BTreeMap;

use super::{Fd, Kernel, ServiceCtx, err};
use crate::abi::Arch;
use crate::abi::errno::Errno;
use crate::vcpu::GuestMemory;

const POLLIN: u32 = 0x0001;
const POLLOUT: u32 = 0x0004;
const POLLERR: u32 = 0x0008;
const POLLHUP: u32 = 0x0010;
const POLLNVAL: u32 = 0x0020;

/// One `eventfd2` counter.
#[derive(Debug, Default)]
pub(super) struct EventFdInst {
    count: u64,
    semaphore: bool,
    nonblock: bool,
}

/// One `timerfd_create` timer. `expiry_ns` is the absolute host-clock deadline
/// (nanoseconds since the UNIX epoch) of the next expiration, or `None` while
/// disarmed; `expirations` accumulates until drained by `read`.
#[derive(Debug, Default)]
pub(super) struct TimerFdInst {
    expiry_ns: Option<u128>,
    interval_ns: u128,
    expirations: u64,
    nonblock: bool,
}

/// One registered `epoll_ctl` interest: the requested event mask and the
/// opaque `data` the kernel echoes back on `epoll_wait`.
#[derive(Debug, Clone, Copy)]
struct EpollWatch {
    events: u32,
    data: u64,
}

/// One `epoll_create1` instance: fd -> interest, keyed by the watched fd
/// number (fd numbers are per-process, but an epoll instance only ever
/// watches fds from the process that created it, so this is unambiguous).
#[derive(Debug, Default)]
pub(super) struct EpollInst {
    interest: BTreeMap<i32, EpollWatch>,
}

/// Nanoseconds since the UNIX epoch on the host wall clock.
pub(super) fn now_ns() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
}

fn read_u16(mem: &GuestMemory, addr: u64) -> Option<u16> {
    let b = mem.read_vec(addr, 2).ok()?;
    Some(u16::from_le_bytes([b[0], b[1]]))
}

fn write_u16(mem: &mut GuestMemory, addr: u64, v: u16) -> bool {
    mem.write(addr, &v.to_le_bytes()).is_ok()
}

/// Read a `struct timespec { i64 tv_sec; i64 tv_nsec; }` (16 bytes).
fn read_timespec(mem: &GuestMemory, ptr: u64) -> Option<(u64, u64)> {
    let sec = mem.read_u64(ptr).ok()?;
    let nsec = mem.read_u64(ptr + 8).ok()?;
    Some((sec, nsec))
}

/// Convert a `timespec` duration to whole milliseconds, rounding a non-zero
/// sub-millisecond remainder up to 1 (so a tiny timeout still parks briefly
/// rather than collapsing to the zero-timeout "poll and return" case).
fn timespec_to_ms(sec: u64, nsec: u64) -> i64 {
    let ms = sec.saturating_mul(1000) + (nsec / 1_000_000);
    if ms == 0 && nsec > 0 {
        1
    } else {
        ms.min(i64::MAX as u64) as i64
    }
}

/// Write a `struct timespec` (16 bytes).
fn write_timespec(mem: &mut GuestMemory, ptr: u64, sec: u64, nsec: u64) -> bool {
    mem.write(ptr, &sec.to_le_bytes()).is_ok() && mem.write(ptr + 8, &nsec.to_le_bytes()).is_ok()
}

/// Number of `u64` words a `fd_set` covering `nfds` descriptors spans.
fn fdset_word_count(nfds: u64) -> u64 {
    nfds.div_ceil(64)
}

/// Read an `fd_set` into one bool per descriptor `0..nfds`. A null pointer
/// means "no fds requested" (all `false`), matching `select`/`pselect6`.
fn read_fdset(mem: &GuestMemory, ptr: u64, nfds: u64) -> Option<Vec<bool>> {
    if ptr == 0 {
        return Some(vec![false; nfds as usize]);
    }
    let mut bits = Vec::with_capacity(nfds as usize);
    for i in 0..nfds {
        let word = mem.read_u64(ptr + (i / 64) * 8).ok()?;
        bits.push(word & (1u64 << (i % 64)) != 0);
    }
    Some(bits)
}

/// Write an `fd_set` from one bool per descriptor. A null pointer is a no-op.
fn write_fdset(mem: &mut GuestMemory, ptr: u64, nfds: u64, bits: &[bool]) -> bool {
    if ptr == 0 {
        return true;
    }
    let words = fdset_word_count(nfds) as usize;
    let mut out = vec![0u64; words];
    for (i, &b) in bits.iter().enumerate() {
        if b {
            out[i / 64] |= 1u64 << (i % 64);
        }
    }
    for (w, val) in out.iter().enumerate() {
        if mem.write(ptr + (w as u64) * 8, &val.to_le_bytes()).is_err() {
            return false;
        }
    }
    true
}

impl Kernel {
    /// The readiness mask (`POLLIN`/`POLLOUT`/`POLLERR`/`POLLHUP`/`POLLNVAL`)
    /// of guest fd `fd_num` right now. `POLLNVAL` if the fd is not open.
    fn fd_ready(&mut self, cx: &mut ServiceCtx, fd_num: i32) -> u32 {
        let Some(fd) = cx.cur.fds.get(fd_num).cloned() else {
            return POLLNVAL;
        };
        match fd {
            // Host stdin readiness isn't tracked; assume data may be waiting.
            Fd::Stdin => POLLIN,
            Fd::Stdout | Fd::Stderr => POLLOUT,
            // Regular files/dirs never block in this kernel.
            Fd::File { .. } | Fd::Dir { .. } => POLLIN | POLLOUT,
            // A host-bridged socket gets a precise readable answer (a peek);
            // in-VM loopback sockets stay best-effort always-ready, since
            // their queues aren't observable from here.
            Fd::Socket { sock, .. } => self.host_socket_readiness(sock).unwrap_or(POLLIN | POLLOUT),
            Fd::PipeRead(i) => {
                let p = &self.pipes[i];
                if !p.buf.is_empty() {
                    POLLIN
                } else if p.writers == 0 {
                    POLLIN | POLLHUP
                } else {
                    0
                }
            }
            Fd::PipeWrite(i) => {
                if self.pipes[i].readers == 0 {
                    POLLERR
                } else {
                    POLLOUT
                }
            }
            Fd::Eventfd(i) => {
                let mut m = POLLOUT;
                if self.eventfds[i].count > 0 {
                    m |= POLLIN;
                }
                m
            }
            Fd::Timerfd(i) => {
                self.update_timerfd(i);
                if self.timerfds[i].expirations > 0 {
                    POLLIN
                } else {
                    0
                }
            }
            // Nested epoll readiness is not modeled.
            Fd::Epoll(_) => 0,
        }
    }

    /// Decide what a timed wait with no ready fd should do. Returns `true` if
    /// the caller should return "timed out" (0 ready) — either because the
    /// deadline has now passed, or (defensively) `timeout_ms == 0`. Returns
    /// `false` after arranging to park (`self.block = true`); a negative
    /// `timeout_ms` means "wait forever" and always parks.
    ///
    /// The absolute deadline lives on `cur.wake_deadline`, seeded on the first
    /// call and re-checked on every re-trap of the same blocking syscall (the
    /// guest PC never advanced past the `svc`). [`Kernel::service`] clears it
    /// once the syscall finally completes.
    #[allow(clippy::unused_self)]
    fn block_or_timeout(&mut self, cx: &mut ServiceCtx, timeout_ms: i64) -> bool {
        if timeout_ms == 0 {
            return true;
        }
        if timeout_ms < 0 {
            cx.block = true;
            return false;
        }
        let now = now_ns();
        match cx.cur.wake_deadline {
            Some(dl) if now >= dl => {
                cx.cur.wake_deadline = None;
                true
            }
            Some(_) => {
                cx.block = true;
                false
            }
            None => {
                cx.cur.wake_deadline = Some(now + (timeout_ms as u128) * 1_000_000);
                cx.block = true;
                false
            }
        }
    }

    // ---- poll / ppoll -------------------------------------------------

    /// `poll(fds, nfds, timeout_ms)`.
    pub(super) fn sys_poll(
        &mut self, cx: &mut ServiceCtx,
        fds_ptr: u64,
        nfds: u64,
        timeout_ms: i64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const STRIDE: u64 = 8; // struct pollfd { int fd; short events; short revents; }
        let mut entries = Vec::with_capacity(nfds as usize);
        for i in 0..nfds {
            let addr = fds_ptr + i * STRIDE;
            let Ok(fd_raw) = mem.read_u32(addr) else {
                return err(Errno::EFAULT);
            };
            let Some(events) = read_u16(mem, addr + 4) else {
                return err(Errno::EFAULT);
            };
            entries.push((addr, fd_raw as i32, events));
        }

        let mut ready_count = 0i64;
        for (addr, fd, events) in entries {
            let revents = if fd < 0 {
                0
            } else {
                self.fd_ready(cx, fd) & (u32::from(events) | POLLERR | POLLHUP | POLLNVAL)
            };
            if !write_u16(mem, addr + 6, revents as u16) {
                return err(Errno::EFAULT);
            }
            if revents != 0 {
                ready_count += 1;
            }
        }

        if ready_count > 0 {
            return ready_count;
        }
        self.block_or_timeout(cx, timeout_ms);
        0
    }

    /// `ppoll(fds, nfds, timeout, sigmask, sigsetsize)`. The sigmask is
    /// accepted but not honored (no signal-frame delivery exists to restore
    /// into).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_ppoll(
        &mut self, cx: &mut ServiceCtx,
        fds_ptr: u64,
        nfds: u64,
        timeout_ts: u64,
        _sigmask: u64,
        _sigsetsize: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let timeout_ms = if timeout_ts == 0 {
            -1
        } else {
            let Some((sec, nsec)) = read_timespec(mem, timeout_ts) else {
                return err(Errno::EFAULT);
            };
            timespec_to_ms(sec, nsec)
        };
        self.sys_poll(cx, fds_ptr, nfds, timeout_ms, mem)
    }

    // ---- select / pselect6 ---------------------------------------------

    /// Shared `select`/`pselect6` body: compute readiness for every fd named
    /// in `r`/`w`/`e`, write the ready subsets back, and return the total
    /// count set across all three sets. `immediate` is the zero-timeout case.
    #[allow(clippy::too_many_arguments)]
    fn sys_select_core(
        &mut self, cx: &mut ServiceCtx,
        nfds: u64,
        r: u64,
        w: u64,
        e: u64,
        immediate: bool,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some(rbits) = read_fdset(mem, r, nfds) else {
            return err(Errno::EFAULT);
        };
        let Some(wbits) = read_fdset(mem, w, nfds) else {
            return err(Errno::EFAULT);
        };
        let Some(ebits) = read_fdset(mem, e, nfds) else {
            return err(Errno::EFAULT);
        };

        let mut rout = vec![false; nfds as usize];
        let mut wout = vec![false; nfds as usize];
        let mut eout = vec![false; nfds as usize];
        let mut total = 0i64;
        for fd in 0..nfds as usize {
            if !rbits[fd] && !wbits[fd] && !ebits[fd] {
                continue;
            }
            let ready = self.fd_ready(cx, fd as i32);
            if rbits[fd] && ready & (POLLIN | POLLHUP) != 0 {
                rout[fd] = true;
                total += 1;
            }
            if wbits[fd] && ready & POLLOUT != 0 {
                wout[fd] = true;
                total += 1;
            }
            if ebits[fd] && ready & POLLERR != 0 {
                eout[fd] = true;
                total += 1;
            }
        }

        if total > 0 || immediate {
            if !write_fdset(mem, r, nfds, &rout)
                || !write_fdset(mem, w, nfds, &wout)
                || !write_fdset(mem, e, nfds, &eout)
            {
                return err(Errno::EFAULT);
            }
            return total;
        }
        cx.block = true;
        0
    }

    /// `pselect6(nfds, readfds, writefds, exceptfds, timeout, sigmask)`. The
    /// sigmask argument is accepted but not honored.
    #[allow(clippy::too_many_arguments)] // one parameter per syscall argument
    pub(super) fn sys_pselect6(
        &mut self, cx: &mut ServiceCtx,
        nfds: u64,
        r: u64,
        w: u64,
        e: u64,
        timeout_ts: u64,
        _sigmask: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let immediate = if timeout_ts == 0 {
            false
        } else {
            let Some((sec, nsec)) = read_timespec(mem, timeout_ts) else {
                return err(Errno::EFAULT);
            };
            sec == 0 && nsec == 0
        };
        self.sys_select_core(cx, nfds, r, w, e, immediate, mem)
    }

    /// The legacy `select(nfds, readfds, writefds, exceptfds, timeout)`
    /// (x86-64 only); `timeout` is a `struct timeval { i64 sec; i64 usec; }`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_select(
        &mut self, cx: &mut ServiceCtx,
        nfds: u64,
        r: u64,
        w: u64,
        e: u64,
        timeout_tv: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let immediate = if timeout_tv == 0 {
            false
        } else {
            let (Ok(sec), Ok(usec)) = (mem.read_u64(timeout_tv), mem.read_u64(timeout_tv + 8))
            else {
                return err(Errno::EFAULT);
            };
            sec == 0 && usec == 0
        };
        self.sys_select_core(cx, nfds, r, w, e, immediate, mem)
    }

    // ---- epoll ------------------------------------------------------------

    /// `(data_offset, struct_size)` of `struct epoll_event` for the guest
    /// arch: x86-64's is `__attribute__((packed))` (12 bytes); every other
    /// arch (aarch64 included) naturally aligns the 8-byte `data` union to a
    /// 16-byte struct.
    fn epoll_event_layout(&self) -> (u64, u64) {
        match self.arch {
            Arch::X86_64 => (4, 12),
            Arch::Aarch64 => (8, 16),
        }
    }

    fn read_epoll_event(&self, mem: &GuestMemory, ptr: u64) -> Option<(u32, u64)> {
        let events = mem.read_u32(ptr).ok()?;
        let (data_off, _) = self.epoll_event_layout();
        let data = mem.read_u64(ptr + data_off).ok()?;
        Some((events, data))
    }

    fn write_epoll_event(&self, mem: &mut GuestMemory, ptr: u64, events: u32, data: u64) -> bool {
        let (data_off, _) = self.epoll_event_layout();
        mem.write(ptr, &events.to_le_bytes()).is_ok()
            && mem.write(ptr + data_off, &data.to_le_bytes()).is_ok()
    }

    /// `epoll_create`/`epoll_create1(flags)` — a fresh, empty interest set.
    pub(super) fn sys_epoll_create1(&mut self, cx: &mut ServiceCtx, _flags: u64) -> i64 {
        let idx = self.epolls.len();
        self.epolls.push(EpollInst::default());
        i64::from(cx.cur.fds.alloc(Fd::Epoll(idx)))
    }

    /// `epoll_ctl(epfd, op, fd, event)`.
    pub(super) fn sys_epoll_ctl(
        &mut self, cx: &mut ServiceCtx,
        epfd: u64,
        op: u64,
        fd: u64,
        event_ptr: u64,
        mem: &GuestMemory,
    ) -> i64 {
        const EPOLL_CTL_ADD: u64 = 1;
        const EPOLL_CTL_DEL: u64 = 2;
        const EPOLL_CTL_MOD: u64 = 3;

        let Some(Fd::Epoll(idx)) = cx.cur.fds.get(epfd as i32).cloned() else {
            return err(Errno::EBADF);
        };
        if cx.cur.fds.get(fd as i32).is_none() {
            return err(Errno::EBADF);
        }
        let target = fd as i32;
        match op {
            EPOLL_CTL_ADD => {
                if self.epolls[idx].interest.contains_key(&target) {
                    return err(Errno::EEXIST);
                }
                let Some((events, data)) = self.read_epoll_event(mem, event_ptr) else {
                    return err(Errno::EFAULT);
                };
                self.epolls[idx]
                    .interest
                    .insert(target, EpollWatch { events, data });
                0
            }
            EPOLL_CTL_MOD => {
                if !self.epolls[idx].interest.contains_key(&target) {
                    return err(Errno::ENOENT);
                }
                let Some((events, data)) = self.read_epoll_event(mem, event_ptr) else {
                    return err(Errno::EFAULT);
                };
                self.epolls[idx]
                    .interest
                    .insert(target, EpollWatch { events, data });
                0
            }
            EPOLL_CTL_DEL => {
                if self.epolls[idx].interest.remove(&target).is_none() {
                    return err(Errno::ENOENT);
                }
                0
            }
            _ => err(Errno::EINVAL),
        }
    }

    /// `epoll_wait`/`epoll_pwait(epfd, events, maxevents, timeout_ms, ...)`.
    /// The `epoll_pwait` sigmask argument is accepted but not honored.
    pub(super) fn sys_epoll_wait(
        &mut self, cx: &mut ServiceCtx,
        epfd: u64,
        events_ptr: u64,
        maxevents: u64,
        timeout_ms: i64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some(Fd::Epoll(idx)) = cx.cur.fds.get(epfd as i32).cloned() else {
            return err(Errno::EBADF);
        };
        if maxevents == 0 {
            return err(Errno::EINVAL);
        }

        let watches: Vec<(i32, EpollWatch)> = self.epolls[idx]
            .interest
            .iter()
            .map(|(&fd, &w)| (fd, w))
            .collect();

        let (_, stride) = self.epoll_event_layout();
        let mut n = 0u64;
        for (fd, w) in watches {
            if n >= maxevents {
                break;
            }
            let ready = self.fd_ready(cx, fd) & (w.events | POLLERR | POLLHUP);
            if ready == 0 {
                continue;
            }
            let addr = events_ptr + n * stride;
            if !self.write_epoll_event(mem, addr, ready, w.data) {
                return err(Errno::EFAULT);
            }
            n += 1;
        }

        if n > 0 {
            return n as i64;
        }
        self.block_or_timeout(cx, timeout_ms);
        0
    }

    /// `epoll_pwait2(epfd, events, maxevents, timeout, sigmask, sigsetsize)` —
    /// like `epoll_pwait` but the timeout is a `struct timespec*`.
    pub(super) fn sys_epoll_pwait2(
        &mut self, cx: &mut ServiceCtx,
        epfd: u64,
        events_ptr: u64,
        maxevents: u64,
        timeout_ts: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let timeout_ms = if timeout_ts == 0 {
            -1
        } else {
            let Some((sec, nsec)) = read_timespec(mem, timeout_ts) else {
                return err(Errno::EFAULT);
            };
            timespec_to_ms(sec, nsec)
        };
        self.sys_epoll_wait(cx, epfd, events_ptr, maxevents, timeout_ms, mem)
    }

    // ---- eventfd ------------------------------------------------------------

    /// `eventfd`/`eventfd2(initval, flags)`.
    pub(super) fn sys_eventfd2(&mut self, cx: &mut ServiceCtx, initval: u64, flags: u64) -> i64 {
        const EFD_SEMAPHORE: u64 = 1;
        const EFD_NONBLOCK: u64 = 0o4000;
        let idx = self.eventfds.len();
        self.eventfds.push(EventFdInst {
            count: initval,
            semaphore: flags & EFD_SEMAPHORE != 0,
            nonblock: flags & EFD_NONBLOCK != 0,
        });
        i64::from(cx.cur.fds.alloc(Fd::Eventfd(idx)))
    }

    /// `read(eventfd_fd, buf, count)` — drain the counter (called from
    /// [`Kernel::sys_read`](super::Kernel)).
    pub(super) fn read_eventfd(
        &mut self, cx: &mut ServiceCtx,
        i: usize,
        buf: u64,
        count: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        if count < 8 {
            return err(Errno::EINVAL);
        }
        if self.eventfds[i].count == 0 {
            if self.eventfds[i].nonblock {
                return err(Errno::EAGAIN);
            }
            cx.block = true;
            return 0;
        }
        let value = if self.eventfds[i].semaphore {
            self.eventfds[i].count -= 1;
            1u64
        } else {
            std::mem::replace(&mut self.eventfds[i].count, 0)
        };
        if mem.write(buf, &value.to_le_bytes()).is_err() {
            return err(Errno::EFAULT);
        }
        8
    }

    /// `write(eventfd_fd, buf, count)` — add to the counter (called from
    /// [`Kernel::sys_write`](super::Kernel)).
    pub(super) fn write_eventfd(&mut self, cx: &mut ServiceCtx, i: usize, data: &[u8]) -> i64 {
        if data.len() < 8 {
            return err(Errno::EINVAL);
        }
        let mut b = [0u8; 8];
        b.copy_from_slice(&data[..8]);
        let value = u64::from_le_bytes(b);
        if value == u64::MAX {
            return err(Errno::EINVAL);
        }
        let cur = self.eventfds[i].count;
        match cur.checked_add(value) {
            Some(sum) if sum < u64::MAX => {
                self.eventfds[i].count = sum;
                8
            }
            _ if self.eventfds[i].nonblock => err(Errno::EAGAIN),
            _ => {
                cx.block = true;
                0
            }
        }
    }

    // ---- timerfd ------------------------------------------------------------

    /// `timerfd_create(clockid, flags)` — `clockid` is accepted but ignored
    /// (there is only the one host wall clock).
    pub(super) fn sys_timerfd_create(&mut self, cx: &mut ServiceCtx, _clockid: u64, flags: u64) -> i64 {
        const TFD_NONBLOCK: u64 = 0o4000;
        let idx = self.timerfds.len();
        self.timerfds.push(TimerFdInst {
            expiry_ns: None,
            interval_ns: 0,
            expirations: 0,
            nonblock: flags & TFD_NONBLOCK != 0,
        });
        i64::from(cx.cur.fds.alloc(Fd::Timerfd(idx)))
    }

    /// Advance timer `i` to the current time: if its deadline has passed,
    /// accumulate the elapsed expiration count and (for a periodic timer)
    /// rearm the deadline, or (for a one-shot) disarm it.
    fn update_timerfd(&mut self, i: usize) {
        let Some(expiry) = self.timerfds[i].expiry_ns else {
            return;
        };
        let now = now_ns();
        if now < expiry {
            return;
        }
        let interval = self.timerfds[i].interval_ns;
        if let Some(periods) = (now - expiry).checked_div(interval) {
            // Periodic: rearm `elapsed` periods past the missed deadline.
            let elapsed = periods + 1;
            self.timerfds[i].expirations = self.timerfds[i]
                .expirations
                .saturating_add(u64::try_from(elapsed).unwrap_or(u64::MAX));
            self.timerfds[i].expiry_ns = Some(expiry + elapsed * interval);
        } else {
            // `interval == 0`: a one-shot timer, disarmed after firing once.
            self.timerfds[i].expirations = self.timerfds[i].expirations.saturating_add(1);
            self.timerfds[i].expiry_ns = None;
        }
    }

    /// The `(sec, nsec)` remaining until timer `i`'s next expiration (after
    /// bringing its state up to date), or `(0, 0)` while disarmed.
    fn timerfd_remaining(&mut self, i: usize) -> (u64, u64) {
        self.update_timerfd(i);
        match self.timerfds[i].expiry_ns {
            None => (0, 0),
            Some(exp) => {
                let now = now_ns();
                let rem = exp.saturating_sub(now);
                (
                    u64::try_from(rem / 1_000_000_000).unwrap_or(u64::MAX),
                    (rem % 1_000_000_000) as u64,
                )
            }
        }
    }

    /// `timerfd_settime(fd, flags, new_value, old_value)`.
    pub(super) fn sys_timerfd_settime(
        &mut self, cx: &mut ServiceCtx,
        fd: u64,
        flags: u64,
        new_value: u64,
        old_value: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const TFD_TIMER_ABSTIME: u64 = 1;

        let Some(Fd::Timerfd(i)) = cx.cur.fds.get(fd as i32).cloned() else {
            return err(Errno::EBADF);
        };
        let Some((int_sec, int_nsec)) = read_timespec(mem, new_value) else {
            return err(Errno::EFAULT);
        };
        let Some((val_sec, val_nsec)) = read_timespec(mem, new_value + 16) else {
            return err(Errno::EFAULT);
        };
        if int_nsec >= 1_000_000_000 || val_nsec >= 1_000_000_000 {
            return err(Errno::EINVAL);
        }

        if old_value != 0 {
            let interval_ns = self.timerfds[i].interval_ns;
            let (rem_sec, rem_nsec) = self.timerfd_remaining(i);
            if !write_timespec(
                mem,
                old_value,
                u64::try_from(interval_ns / 1_000_000_000).unwrap_or(u64::MAX),
                (interval_ns % 1_000_000_000) as u64,
            ) || !write_timespec(mem, old_value + 16, rem_sec, rem_nsec)
            {
                return err(Errno::EFAULT);
            }
        }

        let interval_ns = u128::from(int_sec) * 1_000_000_000 + u128::from(int_nsec);
        let value_ns = u128::from(val_sec) * 1_000_000_000 + u128::from(val_nsec);
        self.timerfds[i].expiry_ns = if value_ns == 0 {
            None
        } else if flags & TFD_TIMER_ABSTIME != 0 {
            Some(value_ns)
        } else {
            Some(now_ns() + value_ns)
        };
        self.timerfds[i].interval_ns = interval_ns;
        self.timerfds[i].expirations = 0;
        0
    }

    /// `timerfd_gettime(fd, curr_value)`.
    pub(super) fn sys_timerfd_gettime(
        &mut self, cx: &mut ServiceCtx,
        fd: u64,
        curr_value: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some(Fd::Timerfd(i)) = cx.cur.fds.get(fd as i32).cloned() else {
            return err(Errno::EBADF);
        };
        let interval_ns = self.timerfds[i].interval_ns;
        let (sec, nsec) = self.timerfd_remaining(i);
        if !write_timespec(
            mem,
            curr_value,
            u64::try_from(interval_ns / 1_000_000_000).unwrap_or(u64::MAX),
            (interval_ns % 1_000_000_000) as u64,
        ) || !write_timespec(mem, curr_value + 16, sec, nsec)
        {
            return err(Errno::EFAULT);
        }
        0
    }

    /// `read(timerfd_fd, buf, count)` — drain the accumulated expiration
    /// count (called from [`Kernel::sys_read`](super::Kernel)).
    pub(super) fn read_timerfd(
        &mut self, cx: &mut ServiceCtx,
        i: usize,
        buf: u64,
        count: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        if count < 8 {
            return err(Errno::EINVAL);
        }
        self.update_timerfd(i);
        if self.timerfds[i].expirations == 0 {
            if self.timerfds[i].nonblock {
                return err(Errno::EAGAIN);
            }
            cx.block = true;
            return 0;
        }
        let val = self.timerfds[i].expirations;
        self.timerfds[i].expirations = 0;
        if mem.write(buf, &val.to_le_bytes()).is_err() {
            return err(Errno::EFAULT);
        }
        8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::Arch;
    use crate::abi::arch::Sysno;
    use crate::fs::{MountTable, TmpFs};
    use crate::vcpu::mem::Prot;
    use crate::vcpu::{Exit, Vcpu, VcpuError};

    /// A no-op vcpu, matching the one in the `kernel` module's tests.
    #[derive(Clone)]
    struct DummyVcpu;
    impl Vcpu for DummyVcpu {
        fn run(&mut self, _m: &mut GuestMemory) -> Result<Exit, VcpuError> {
            Ok(Exit::Halt)
        }
        fn syscall_nr(&self) -> u64 {
            0
        }
        fn syscall_args(&self) -> [u64; 6] {
            [0; 6]
        }
        fn set_syscall_ret(&mut self, _v: u64) {}
        fn reg(&self, _i: usize) -> u64 {
            0
        }
        fn set_reg(&mut self, _i: usize, _v: u64) {}
        fn pc(&self) -> u64 {
            0
        }
        fn set_pc(&mut self, _v: u64) {}
        fn sp(&self) -> u64 {
            0
        }
        fn set_sp(&mut self, _v: u64) {}
        fn set_tls(&mut self, _v: u64) {}
        fn fork(&self) -> Box<dyn Vcpu> {
            Box::new(self.clone())
        }
        fn reset(&mut self, _e: u64, _s: u64) {}
    }

    const PAGE: u64 = 4096;

    fn setup() -> (Kernel, GuestMemory, DummyVcpu, ServiceCtx) {
        let mut mounts = MountTable::new();
        mounts.mount("/", Box::new(TmpFs::new()));
        let kernel = Kernel::new(Arch::Aarch64, mounts);
        let mut cx = ServiceCtx::default();
        cx.cur.pid = 1;
        let mut mem = GuestMemory::new(0x1_0000, 16 * PAGE);
        mem.map(0x1_0000, 4 * PAGE, Prot::rw()).unwrap();
        (kernel, mem, DummyVcpu, cx)
    }

    fn call(
        k: &mut Kernel,
        cx: &mut ServiceCtx,
        mem: &mut GuestMemory,
        v: &mut DummyVcpu,
        s: Sysno,
        a: [u64; 6],
    ) -> i64 {
        k.dispatch(cx, s, 0, &a, v, mem)
    }

    #[test]
    fn eventfd_write_then_read_counter() {
        let (mut k, mut mem, mut v, mut cx) = setup();
        let fd = call(
            &mut k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Eventfd2,
            [0, 0, 0, 0, 0, 0],
        );
        assert!(fd >= 3);
        let fd = fd as u64;

        let buf = 0x1_0000;
        mem.write_init(buf, &3u64.to_le_bytes()).unwrap();
        assert_eq!(
            call(
                &mut k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [fd, buf, 8, 0, 0, 0]
            ),
            8
        );
        mem.write_init(buf, &4u64.to_le_bytes()).unwrap();
        assert_eq!(
            call(
                &mut k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [fd, buf, 8, 0, 0, 0]
            ),
            8
        );

        let out = 0x1_1000;
        assert_eq!(
            call(&mut k, &mut cx, &mut mem, &mut v, Sysno::Read, [fd, out, 8, 0, 0, 0]),
            8
        );
        assert_eq!(mem.read_u64(out).unwrap(), 7);

        // Drained: read again with count == 0 (non-blocking check via the
        // `block` flag, mirroring the pipe test's convention).
        assert_eq!(
            call(&mut k, &mut cx, &mut mem, &mut v, Sysno::Read, [fd, out, 8, 0, 0, 0]),
            0
        );
        assert!(cx.block);
    }

    #[test]
    fn eventfd_semaphore_mode_decrements_by_one() {
        const EFD_SEMAPHORE: u64 = 1;
        let (mut k, mut mem, mut v, mut cx) = setup();
        let fd = call(
            &mut k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Eventfd2,
            [5, EFD_SEMAPHORE, 0, 0, 0, 0],
        ) as u64;

        let out = 0x1_0000;
        assert_eq!(
            call(&mut k, &mut cx, &mut mem, &mut v, Sysno::Read, [fd, out, 8, 0, 0, 0]),
            8
        );
        assert_eq!(mem.read_u64(out).unwrap(), 1);
    }

    #[test]
    fn poll_reports_pollin_when_pipe_has_data() {
        let (mut k, mut mem, mut v, mut cx) = setup();
        let fds = 0x1_0000;
        call(&mut k, &mut cx, &mut mem, &mut v, Sysno::Pipe2, [fds, 0, 0, 0, 0, 0]);
        let rfd = u64::from(mem.read_u32(fds).unwrap());
        let wfd = u64::from(mem.read_u32(fds + 4).unwrap());

        let msg = 0x1_1000;
        mem.write_init(msg, b"hi").unwrap();
        call(
            &mut k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Write,
            [wfd, msg, 2, 0, 0, 0],
        );

        // A zero-timeout poll on the read end: no data yet on an *empty*
        // pipe would report 0; here data is buffered so it must report ready.
        let pollfds = 0x1_2000;
        mem.write_init(pollfds, &(rfd as u32).to_le_bytes())
            .unwrap();
        mem.write_init(pollfds + 4, &1u16.to_le_bytes()).unwrap(); // POLLIN
        mem.write_init(pollfds + 6, &0u16.to_le_bytes()).unwrap();

        let n = call(
            &mut k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Poll,
            [pollfds, 1, 0, 0, 0, 0],
        );
        assert_eq!(n, 1);
        assert_eq!(mem.read_vec(pollfds + 6, 2).unwrap(), 1u16.to_le_bytes());
    }

    #[test]
    fn poll_zero_timeout_on_empty_pipe_returns_immediately() {
        let (mut k, mut mem, mut v, mut cx) = setup();
        let fds = 0x1_0000;
        call(&mut k, &mut cx, &mut mem, &mut v, Sysno::Pipe2, [fds, 0, 0, 0, 0, 0]);
        let rfd = u64::from(mem.read_u32(fds).unwrap());

        let pollfds = 0x1_2000;
        mem.write_init(pollfds, &(rfd as u32).to_le_bytes())
            .unwrap();
        mem.write_init(pollfds + 4, &1u16.to_le_bytes()).unwrap();

        let n = call(
            &mut k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Poll,
            [pollfds, 1, 0, 0, 0, 0],
        );
        assert_eq!(n, 0, "no data and writer still open, but timeout=0");
        assert!(!cx.block, "zero timeout must not block");
    }

    #[test]
    fn epoll_create_ctl_wait_on_ready_pipe() {
        let (mut k, mut mem, mut v, mut cx) = setup();
        let fds = 0x1_0000;
        call(&mut k, &mut cx, &mut mem, &mut v, Sysno::Pipe2, [fds, 0, 0, 0, 0, 0]);
        let rfd = u64::from(mem.read_u32(fds).unwrap());
        let wfd = u64::from(mem.read_u32(fds + 4).unwrap());

        let msg = 0x1_1000;
        mem.write_init(msg, b"yo").unwrap();
        call(
            &mut k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Write,
            [wfd, msg, 2, 0, 0, 0],
        );

        let epfd = call(
            &mut k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::EpollCreate1,
            [0, 0, 0, 0, 0, 0],
        );
        assert!(epfd >= 3);
        let epfd = epfd as u64;

        let ev = 0x1_2000;
        mem.write_init(ev, &1u32.to_le_bytes()).unwrap(); // EPOLLIN
        mem.write_init(ev + 8, &0x1234_5678u64.to_le_bytes())
            .unwrap(); // data (aarch64 offset)
        assert_eq!(
            call(
                &mut k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::EpollCtl,
                [epfd, 1, rfd, ev, 0, 0]
            ),
            0
        );

        let out = 0x1_3000;
        let n = call(
            &mut k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::EpollPwait,
            [epfd, out, 4, 0, 0, 0],
        );
        assert_eq!(n, 1);
        assert_eq!(mem.read_u32(out).unwrap() & 1, 1, "POLLIN reported");
        assert_eq!(mem.read_u64(out + 8).unwrap(), 0x1234_5678);
    }

    #[test]
    fn timerfd_create_settime_gettime() {
        let (mut k, mut mem, mut v, mut cx) = setup();
        let fd = call(
            &mut k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::TimerfdCreate,
            [0, 0, 0, 0, 0, 0],
        );
        assert!(fd >= 3);
        let fd = fd as u64;

        // Arm with a 10-second one-shot value.
        let newval = 0x1_0000;
        mem.write_init(newval, &0u64.to_le_bytes()).unwrap(); // interval sec
        mem.write_init(newval + 8, &0u64.to_le_bytes()).unwrap(); // interval nsec
        mem.write_init(newval + 16, &10u64.to_le_bytes()).unwrap(); // value sec
        mem.write_init(newval + 24, &0u64.to_le_bytes()).unwrap(); // value nsec
        assert_eq!(
            call(
                &mut k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::TimerfdSettime,
                [fd, 0, newval, 0, 0, 0]
            ),
            0
        );

        let curval = 0x1_1000;
        assert_eq!(
            call(
                &mut k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::TimerfdGettime,
                [fd, curval, 0, 0, 0, 0]
            ),
            0
        );
        // Interval is 0; remaining value should be <= 10s and > 0.
        assert_eq!(mem.read_u64(curval).unwrap(), 0);
        assert_eq!(mem.read_u64(curval + 8).unwrap(), 0);
        let rem_sec = mem.read_u64(curval + 16).unwrap();
        assert!(rem_sec <= 10, "remaining seconds should be at most 10");

        // A zero-timeout read must not block forever (armed but not yet
        // expired -> EAGAIN via O_NONBLOCK-equivalent isn't set, so it would
        // normally block; assert it sets the block flag instead of hanging).
        let out = 0x1_2000;
        let ret = call(&mut k, &mut cx, &mut mem, &mut v, Sysno::Read, [fd, out, 8, 0, 0, 0]);
        assert_eq!(ret, 0);
        assert!(cx.block);
    }

    #[test]
    fn timerfd_disarm_with_zero_value() {
        let (mut k, mut mem, mut v, mut cx) = setup();
        let fd = call(
            &mut k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::TimerfdCreate,
            [0, 0, 0, 0, 0, 0],
        ) as u64;

        let newval = 0x1_0000;
        mem.write_init(newval, &[0u8; 32]).unwrap(); // all-zero itimerspec disarms
        assert_eq!(
            call(
                &mut k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::TimerfdSettime,
                [fd, 0, newval, 0, 0, 0]
            ),
            0
        );

        let curval = 0x1_1000;
        call(
            &mut k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::TimerfdGettime,
            [fd, curval, 0, 0, 0, 0],
        );
        assert_eq!(mem.read_u64(curval + 16).unwrap(), 0);
        assert_eq!(mem.read_u64(curval + 24).unwrap(), 0);
    }
}
