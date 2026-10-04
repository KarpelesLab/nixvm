//! POSIX message queues: `mq_open`, `mq_unlink`, `mq_timedsend`,
//! `mq_timedreceive`, `mq_notify`, `mq_getsetattr`.
//!
//! A queue descriptor is a real fd ([`Fd::Mqueue`]) — pollable (`POLLIN` when
//! a message waits, `POLLOUT` when there is room), so it plugs into
//! `poll`/`select`/`epoll` like on Linux, where libc's `mqd_t` *is* an fd.
//! Queues live in the poll subsystem's table ([`PollFds::mqueues`]) under a
//! flat namespace (the mqueue filesystem's root); `mq_unlink` removes the name
//! while open descriptors keep the queue. Messages are kept by priority, FIFO
//! within one; full/empty queues block (re-trap) until space/data appears or
//! the absolute `CLOCK_REALTIME` timeout passes (`ETIMEDOUT`), or fail
//! `EAGAIN` on an `O_NONBLOCK` descriptor. `mq_notify` with `SIGEV_SIGNAL`
//! signals the registered process when a message lands in an empty queue
//! nobody is waiting on, then deregisters — POSIX's one-shot rule.

use super::poll::PollFds;
use super::{Fd, Kernel, QueuedSig, ServiceCtx, Shared, err, poll};
use crate::abi::errno::Errno;
use crate::vcpu::GuestMemory;

const O_ACCMODE: u64 = 0o3;
const O_CREAT: u64 = 0o100;
const O_EXCL: u64 = 0o200;
const O_NONBLOCK: u64 = 0o4000;
const O_CLOEXEC: u64 = 0o2000000;
/// `MQ_PRIO_MAX`: priorities are `0..32768`.
const MQ_PRIO_MAX: u64 = 32768;
/// Defaults and ceilings (`/proc/sys/fs/mqueue/{msg,msgsize}_{default,max}`,
/// and the hard limits root may go up to).
const DFL_MAXMSG: u64 = 10;
const DFL_MSGSIZE: u64 = 8192;
const HARD_MAXMSG: u64 = 65536;
const HARD_MSGSIZE: u64 = 16 * 1024 * 1024;
/// `si_code` of an `mq_notify` signal.
const SI_MESGQ: i32 = -3;
const SIGEV_SIGNAL: i32 = 0;
const SIGEV_NONE: i32 = 1;

/// One message queue.
#[derive(Debug)]
pub(super) struct MqInst {
    /// Messages, highest priority first, FIFO within a priority.
    msgs: Vec<(u32, Vec<u8>)>,
    maxmsg: u64,
    msgsize: u64,
    mode: u32,
    /// `mq_notify` registration: `(pid, signal, sigev_value)`; signal 0 =
    /// `SIGEV_NONE` (registered, but nothing is sent).
    notify: Option<(i32, u32, u64)>,
}

impl MqInst {
    fn full(&self) -> bool {
        self.msgs.len() as u64 >= self.maxmsg
    }
}

impl PollFds {
    /// `poll` readiness of queue `q`.
    pub(super) fn mq_ready(&self, q: usize) -> u32 {
        const POLLIN: u32 = 0x0001;
        const POLLOUT: u32 = 0x0004;
        let Some(m) = self.mqueues.get(q) else {
            return 0;
        };
        let mut r = 0;
        if !m.msgs.is_empty() {
            r |= POLLIN;
        }
        if !m.full() {
            r |= POLLOUT;
        }
        r
    }
}

/// Validate a queue name as the kernel receives it (libc strips the leading
/// `/`): non-empty, no further `/`, at most `NAME_MAX` bytes.
fn check_name(mem: &GuestMemory, ptr: u64) -> Result<String, i64> {
    let Ok(raw) = mem.read_cstr(ptr, 4096) else {
        return Err(err(Errno::EFAULT));
    };
    if raw.len() > 255 {
        return Err(err(Errno::ENAMETOOLONG));
    }
    if raw.is_empty() {
        return Err(err(Errno::ENOENT));
    }
    if raw.contains(&b'/') {
        return Err(err(Errno::EACCES));
    }
    Ok(String::from_utf8_lossy(&raw).into_owned())
}

/// The descriptor's queue and access: `(queue, readable, writable,
/// nonblock)`, or `EBADF` if `fd` is not a message queue.
fn mq_fd(cx: &ServiceCtx, fd: u64) -> Result<(usize, bool, bool, bool), i64> {
    match cx.cur.fds.get(fd as i32) {
        Some(Fd::Mqueue { q, flags }) => {
            let acc = flags & O_ACCMODE;
            Ok((*q, acc != 1, acc != 0, flags & O_NONBLOCK != 0))
        }
        _ => Err(err(Errno::EBADF)),
    }
}

/// Turn an absolute `CLOCK_REALTIME` timeout (`struct timespec *`, NULL =
/// none) into the wall deadline the scheduler tracks, seeding it once per
/// blocking call.
fn deadline(cx: &mut ServiceCtx, ts: u64, mem: &GuestMemory) -> Result<Option<u128>, i64> {
    if ts == 0 {
        return Ok(None);
    }
    if let Some(dl) = cx.cur.wake_deadline {
        return Ok(Some(dl));
    }
    let (Ok(s), Ok(n)) = (mem.read_u64(ts), mem.read_u64(ts + 8)) else {
        return Err(err(Errno::EFAULT));
    };
    if (s as i64) < 0 || n >= 1_000_000_000 {
        return Err(err(Errno::EINVAL));
    }
    let dl = u128::from(s) * 1_000_000_000 + u128::from(n);
    cx.cur.wake_deadline = Some(dl);
    Ok(Some(dl))
}

impl Kernel {
    /// `mq_open(name, oflag, mode, attr)`.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_mq_open(
        &self,
        pf: &mut PollFds,
        cx: &mut ServiceCtx,
        a: &[u64; 6],
        mem: &GuestMemory,
    ) -> i64 {
        let (name, oflag, mode, attr) = (a[0], a[1], a[2], a[3]);
        let name = match check_name(mem, name) {
            Ok(n) => n,
            Err(e) => return e,
        };
        if oflag & O_ACCMODE == 3 {
            return err(Errno::EINVAL);
        }
        let q = match pf.mq_names.get(&name) {
            Some(&q) => {
                if oflag & O_CREAT != 0 && oflag & O_EXCL != 0 {
                    return err(Errno::EEXIST);
                }
                let m = &pf.mqueues[q];
                let want = match oflag & O_ACCMODE {
                    0 => 0o4,
                    1 => 0o2,
                    _ => 0o6,
                };
                if cx.cur.creds.euid != 0 && (m.mode >> 6) & want != want {
                    return err(Errno::EACCES);
                }
                q
            }
            None if oflag & O_CREAT == 0 => return err(Errno::ENOENT),
            None => {
                let root = cx.cur.creds.euid == 0;
                let (maxmsg, msgsize) = if attr == 0 {
                    (DFL_MAXMSG, DFL_MSGSIZE)
                } else {
                    let (Ok(mx), Ok(sz)) = (mem.read_u64(attr + 8), mem.read_u64(attr + 16)) else {
                        return err(Errno::EFAULT);
                    };
                    let (mx_lim, sz_lim) = if root {
                        (HARD_MAXMSG, HARD_MSGSIZE)
                    } else {
                        (DFL_MAXMSG, DFL_MSGSIZE)
                    };
                    if (mx as i64) <= 0 || (sz as i64) <= 0 || mx > mx_lim || sz > sz_lim {
                        return err(Errno::EINVAL);
                    }
                    (mx, sz)
                };
                pf.mqueues.push(MqInst {
                    msgs: Vec::new(),
                    maxmsg,
                    msgsize,
                    mode: (mode & 0o777) as u32,
                    notify: None,
                });
                let q = pf.mqueues.len() - 1;
                pf.mq_names.insert(name, q);
                q
            }
        };
        let fd = cx.cur.fds.alloc(Fd::Mqueue {
            q,
            flags: oflag & (O_ACCMODE | O_NONBLOCK),
        });
        cx.cur.fds.set_cloexec(fd, oflag & O_CLOEXEC != 0);
        i64::from(fd)
    }

    /// `mq_unlink(name)`: remove the name; open descriptors keep the queue.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_mq_unlink(&self, pf: &mut PollFds, name: u64, mem: &GuestMemory) -> i64 {
        let name = match check_name(mem, name) {
            Ok(n) => n,
            Err(e) => return e,
        };
        if pf.mq_names.remove(&name).is_some() {
            0
        } else {
            err(Errno::ENOENT)
        }
    }

    /// `mq_timedsend(mqd, msg_ptr, msg_len, msg_prio, abs_timeout)`.
    pub(super) fn sys_mq_timedsend(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        a: &[u64; 6],
        mem: &GuestMemory,
    ) -> i64 {
        let (fd, ptr, len, prio, ts) = (a[0], a[1], a[2], a[3], a[4]);
        let (q, _, writable, nonblock) = match mq_fd(cx, fd) {
            Ok(v) => v,
            Err(e) => return e,
        };
        if !writable {
            return err(Errno::EBADF);
        }
        if prio >= MQ_PRIO_MAX {
            return err(Errno::EINVAL);
        }
        let dl = match deadline(cx, ts, mem) {
            Ok(d) => d,
            Err(e) => return e,
        };
        let mut pf = self.pollfds.lock().unwrap();
        let m = &mut pf.mqueues[q];
        if len > m.msgsize {
            return err(Errno::EMSGSIZE);
        }
        if m.full() {
            if nonblock {
                return err(Errno::EAGAIN);
            }
            if dl.is_some_and(|d| poll::now_ns() >= d) {
                cx.cur.wake_deadline = None;
                return err(Errno::ETIMEDOUT);
            }
            cx.block = true;
            return 0;
        }
        let Ok(data) = mem.read_vec(ptr, len as usize) else {
            return err(Errno::EFAULT);
        };
        cx.cur.wake_deadline = None;
        let was_empty = m.msgs.is_empty();
        let at = m
            .msgs
            .iter()
            .position(|e| e.0 < prio as u32)
            .unwrap_or(m.msgs.len());
        m.msgs.insert(at, (prio as u32, data));
        // Notify on the empty → non-empty transition (one-shot).
        let note = if was_empty { m.notify.take() } else { None };
        drop(pf);
        if let Some((pid, sig, value)) = note
            && sig != 0
        {
            let info = QueuedSig {
                code: SI_MESGQ,
                pid: cx.cur.tgid,
                uid: cx.cur.creds.ruid,
                value,
            };
            let _ = self.post_signal(sh, cx, i64::from(pid), u64::from(sig), info);
        }
        sh.unpark_all();
        0
    }

    /// `mq_timedreceive(mqd, msg_ptr, msg_len, msg_prio*, abs_timeout)`: the
    /// oldest message of the highest priority.
    pub(super) fn sys_mq_timedreceive(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        a: &[u64; 6],
        mem: &mut GuestMemory,
    ) -> i64 {
        let (fd, ptr, len, prio_ptr, ts) = (a[0], a[1], a[2], a[3], a[4]);
        let (q, readable, _, nonblock) = match mq_fd(cx, fd) {
            Ok(v) => v,
            Err(e) => return e,
        };
        if !readable {
            return err(Errno::EBADF);
        }
        let dl = match deadline(cx, ts, mem) {
            Ok(d) => d,
            Err(e) => return e,
        };
        let mut pf = self.pollfds.lock().unwrap();
        let m = &mut pf.mqueues[q];
        if len < m.msgsize {
            return err(Errno::EMSGSIZE);
        }
        if m.msgs.is_empty() {
            if nonblock {
                return err(Errno::EAGAIN);
            }
            if dl.is_some_and(|d| poll::now_ns() >= d) {
                cx.cur.wake_deadline = None;
                return err(Errno::ETIMEDOUT);
            }
            cx.block = true;
            return 0;
        }
        cx.cur.wake_deadline = None;
        let (prio, data) = &m.msgs[0];
        if mem.write(ptr, data).is_err()
            || (prio_ptr != 0 && mem.write(prio_ptr, &prio.to_le_bytes()).is_err())
        {
            return err(Errno::EFAULT);
        }
        let n = data.len() as i64;
        m.msgs.remove(0);
        drop(pf);
        sh.unpark_all();
        n
    }

    /// `mq_notify(mqd, sevp)`: register (or, with NULL, cancel) the caller's
    /// one-shot arrival notification. Only one process may be registered
    /// (`EBUSY`). `SIGEV_THREAD` is a libc construct built on a netlink socket
    /// the kernel would hand the cookie to; that channel isn't modeled, so it
    /// is refused (`EINVAL`).
    #[allow(clippy::unused_self)]
    pub(super) fn sys_mq_notify(
        &self,
        pf: &mut PollFds,
        cx: &ServiceCtx,
        fd: u64,
        sevp: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let q = match mq_fd(cx, fd) {
            Ok(v) => v.0,
            Err(e) => return e,
        };
        let me = cx.cur.tgid;
        let m = &mut pf.mqueues[q];
        if sevp == 0 {
            if m.notify.is_some_and(|n| n.0 == me) {
                m.notify = None;
            }
            return 0;
        }
        let Ok(raw) = mem.read_vec(sevp, 16) else {
            return err(Errno::EFAULT);
        };
        let value = u64::from_le_bytes(raw[0..8].try_into().unwrap());
        let signo = u32::from_le_bytes(raw[8..12].try_into().unwrap());
        let notify = i32::from_le_bytes(raw[12..16].try_into().unwrap());
        let sig = match notify {
            SIGEV_NONE => 0,
            SIGEV_SIGNAL if (1..=64).contains(&signo) => signo,
            _ => return err(Errno::EINVAL),
        };
        if m.notify.is_some() {
            return err(Errno::EBUSY);
        }
        m.notify = Some((me, sig, value));
        0
    }

    /// `mq_getsetattr(mqd, newattr, oldattr)`: report `struct mq_attr` (flags,
    /// maxmsg, msgsize, curmsgs) and/or change the descriptor's `O_NONBLOCK`.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_mq_getsetattr(
        &self,
        pf: &mut PollFds,
        cx: &mut ServiceCtx,
        fd: u64,
        new: u64,
        old: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let (q, _, _, nonblock) = match mq_fd(cx, fd) {
            Ok(v) => v,
            Err(e) => return e,
        };
        let new_flags = if new == 0 {
            None
        } else {
            match mem.read_u64(new) {
                Ok(f) if f & !O_NONBLOCK == 0 => Some(f),
                Ok(_) => return err(Errno::EINVAL),
                Err(_) => return err(Errno::EFAULT),
            }
        };
        if old != 0 {
            let m = &pf.mqueues[q];
            let mut b = [0u8; 64];
            let flags = if nonblock { O_NONBLOCK } else { 0 };
            b[0..8].copy_from_slice(&flags.to_le_bytes());
            b[8..16].copy_from_slice(&m.maxmsg.to_le_bytes());
            b[16..24].copy_from_slice(&m.msgsize.to_le_bytes());
            b[24..32].copy_from_slice(&(m.msgs.len() as u64).to_le_bytes());
            if mem.write(old, &b).is_err() {
                return err(Errno::EFAULT);
            }
        }
        if let Some(nf) = new_flags
            && let Some(Fd::Mqueue { flags, .. }) = cx.cur.fds.get_mut(fd as i32)
        {
            *flags = (*flags & !O_NONBLOCK) | nf;
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{BASE, call, e, put_str, setup};
    use crate::abi::arch::Sysno;
    use crate::abi::errno::Errno;

    #[test]
    fn open_send_receive_by_priority_and_attrs() {
        let (k, mut mem, mut v, mut cx) = setup();
        let (name, attr, msg, prio) = (BASE, BASE + 0x100, BASE + 0x200, BASE + 0x300);
        put_str(&mut mem, name, "q1");
        // Missing without O_CREAT; a bad name.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::MqOpen,
                [name, 2, 0, 0, 0, 0]
            ),
            e(Errno::ENOENT)
        );
        put_str(&mut mem, BASE + 0x80, "a/b");
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::MqOpen,
                [BASE + 0x80, 0o102, 0o600, 0, 0, 0]
            ),
            e(Errno::EACCES)
        );
        // maxmsg 2, msgsize 16.
        let mut a = [0u8; 64];
        a[8..16].copy_from_slice(&2u64.to_le_bytes());
        a[16..24].copy_from_slice(&16u64.to_le_bytes());
        mem.write(attr, &a).unwrap();
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::MqOpen,
            [name, 0o4102, 0o600, attr, 0, 0],
        );
        assert!(fd >= 0);
        let fd = fd as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::MqOpen,
                [name, 0o302, 0o600, 0, 0, 0]
            ),
            e(Errno::EEXIST)
        );
        mem.write(msg, b"low").unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::MqTimedsend,
                [fd, msg, 3, 1, 0, 0]
            ),
            0
        );
        mem.write(msg, b"high").unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::MqTimedsend,
                [fd, msg, 4, 9, 0, 0]
            ),
            0
        );
        // Full and O_NONBLOCK: EAGAIN. Oversized: EMSGSIZE.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::MqTimedsend,
                [fd, msg, 4, 0, 0, 0]
            ),
            e(Errno::EAGAIN)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::MqTimedsend,
                [fd, msg, 17, 0, 0, 0]
            ),
            e(Errno::EMSGSIZE)
        );
        // getattr: curmsgs 2, O_NONBLOCK set.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::MqGetsetattr,
                [fd, 0, attr, 0, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_u64(attr).unwrap(), 0o4000);
        assert_eq!(mem.read_u64(attr + 24).unwrap(), 2);
        // Receive buffer smaller than msgsize: EMSGSIZE; then highest prio first.
        let buf = BASE + 0x1000;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::MqTimedreceive,
                [fd, buf, 8, prio, 0, 0]
            ),
            e(Errno::EMSGSIZE)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::MqTimedreceive,
                [fd, buf, 16, prio, 0, 0]
            ),
            4
        );
        assert_eq!(mem.read_vec(buf, 4).unwrap(), b"high");
        assert_eq!(mem.read_u32(prio).unwrap(), 9);
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::MqTimedreceive,
                [fd, buf, 16, prio, 0, 0]
            ),
            3
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::MqTimedreceive,
                [fd, buf, 16, prio, 0, 0]
            ),
            e(Errno::EAGAIN)
        );
        // Clear O_NONBLOCK; an already-past timeout then gives ETIMEDOUT.
        mem.write_u64(attr, 0).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::MqGetsetattr,
                [fd, attr, 0, 0, 0, 0]
            ),
            0
        );
        let ts = BASE + 0x400;
        mem.write(ts, &[0u8; 16]).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::MqTimedreceive,
                [fd, buf, 16, 0, ts, 0]
            ),
            e(Errno::ETIMEDOUT)
        );
        // Unlink: the name goes, the descriptor still works.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::MqUnlink,
                [name, 0, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::MqUnlink,
                [name, 0, 0, 0, 0, 0]
            ),
            e(Errno::ENOENT)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::MqTimedsend,
                [fd, msg, 1, 0, 0, 0]
            ),
            0
        );
    }

    #[test]
    fn notify_fires_once_on_empty_to_nonempty() {
        let (k, mut mem, mut v, mut cx) = setup();
        let (name, sev, msg) = (BASE, BASE + 0x100, BASE + 0x200);
        put_str(&mut mem, name, "nq");
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::MqOpen,
            [name, 0o4102, 0o600, 0, 0, 0],
        ) as u64;
        let mut s = [0u8; 64];
        s[0..8].copy_from_slice(&0x77u64.to_le_bytes());
        s[8..12].copy_from_slice(&12u32.to_le_bytes()); // SIGUSR2
        mem.write(sev, &s).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::MqNotify,
                [fd, sev, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::MqNotify,
                [fd, sev, 0, 0, 0, 0]
            ),
            e(Errno::EBUSY)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::MqTimedsend,
                [fd, msg, 1, 0, 0, 0]
            ),
            0
        );
        assert_ne!(cx.cur.pending & (1 << 11), 0, "SIGUSR2 posted");
        assert_eq!(cx.cur.queued_siginfo[12].unwrap().value, 0x77);
        // One-shot: re-registering is allowed again.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::MqNotify,
                [fd, sev, 0, 0, 0, 0]
            ),
            0
        );
        // A non-mq fd is EBADF.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::MqNotify,
                [1, 0, 0, 0, 0, 0]
            ),
            e(Errno::EBADF)
        );
    }
}
