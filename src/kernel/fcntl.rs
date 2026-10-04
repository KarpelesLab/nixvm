//! `fcntl(2)` and the file-locking family: POSIX record locks (`F_SETLK`/
//! `F_SETLKW`/`F_GETLK`), open-file-description locks (`F_OFD_*`), `flock(2)`,
//! memfd seals (`F_ADD_SEALS`/`F_GET_SEALS`), and the descriptor attributes
//! (`F_SETOWN[_EX]`, `F_SETSIG`, leases, `F_NOTIFY`, write-life hints, …).
//!
//! Locks are real and machine-wide — the VM can run several processes that
//! coordinate through them (two SQLite connections in different processes, a
//! `flock -n` single-instance guard, a daemon's pid file). They follow
//! Linux's ownership rules:
//! - POSIX record locks belong to the *process* (thread group): a conflicting
//!   request from another process fails `EAGAIN` (`F_SETLK`) or waits
//!   (`F_SETLKW`, re-trapping until the range frees); the process's own locks
//!   on a file are converted/split in place; and closing *any* descriptor of
//!   the file drops all of the process's locks on it (the infamous POSIX
//!   rule), as does exit.
//! - OFD locks and `flock` locks belong to the *open file description*
//!   ([`super::FileOffset`] identity), shared by dups and fork children, and
//!   go away when its last descriptor closes.
//!
//! Files are identified by `(mount point, inode)` so every path to a file
//! shares its locks. Lock state sits behind a leaf lock ([`Kernel::locks`]).

use std::collections::BTreeMap;

use super::{Fd, Kernel, ServiceCtx, Shared, err};
use crate::abi::errno::Errno;
use crate::fs::MountTable;
use crate::vcpu::GuestMemory;

const F_RDLCK: i16 = 0;
const F_WRLCK: i16 = 1;
const F_UNLCK: i16 = 2;
const SEEK_SET: i16 = 0;
const SEEK_CUR: i16 = 1;
const SEEK_END: i16 = 2;
/// memfd seals.
const F_SEAL_SEAL: u32 = 0x01;
const F_SEAL_SHRINK: u32 = 0x02;
const F_SEAL_GROW: u32 = 0x04;
const F_SEAL_WRITE: u32 = 0x08;
const F_SEAL_FUTURE_WRITE: u32 = 0x10;
const F_SEAL_EXEC: u32 = 0x20;

/// A file's identity (as in the page cache).
type FileKey = (String, u64);

/// Who holds a record lock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Owner {
    /// A POSIX lock: the thread group.
    Process(i32),
    /// An OFD lock: the open file description.
    Ofd(usize),
}

/// One record lock: `[start, end)` (`end == u64::MAX` = to EOF and beyond).
#[derive(Clone, Copy, Debug)]
struct RecLock {
    owner: Owner,
    write: bool,
    start: u64,
    end: u64,
    /// The pid `F_GETLK` reports (`-1` for an OFD lock).
    pid: i32,
}

/// One `flock` lock, held by an open file description.
#[derive(Clone, Copy, Debug)]
struct Flock {
    ofd: usize,
    exclusive: bool,
}

/// Machine-wide lock and seal state.
#[derive(Debug, Default)]
pub(super) struct FileLocks {
    records: BTreeMap<FileKey, Vec<RecLock>>,
    flocks: BTreeMap<FileKey, Vec<Flock>>,
    /// memfd seals by the memfd's (hidden) path.
    seals: BTreeMap<String, u32>,
}

impl FileLocks {
    fn is_empty(&self) -> bool {
        self.records.is_empty() && self.flocks.is_empty()
    }

    /// The first lock (by another owner) that conflicts with a `write`/read
    /// request over `[start, end)` by `owner`.
    fn conflict(
        &self,
        key: &FileKey,
        owner: Owner,
        write: bool,
        start: u64,
        end: u64,
    ) -> Option<RecLock> {
        self.records
            .get(key)?
            .iter()
            .copied()
            .find(|l| l.owner != owner && l.start < end && start < l.end && (write || l.write))
    }

    /// Set (or, `typ == F_UNLCK`, clear) `owner`'s lock over `[start, end)`:
    /// the owner's existing locks are trimmed/split around the range first,
    /// so a new lock *replaces* whatever the owner held there (POSIX lock
    /// conversion).
    fn set(&mut self, key: &FileKey, owner: Owner, typ: i16, start: u64, end: u64, pid: i32) {
        let v = self.records.entry(key.clone()).or_default();
        let mut out = Vec::with_capacity(v.len() + 2);
        for l in v.drain(..) {
            if l.owner != owner || l.end <= start || end <= l.start {
                out.push(l);
                continue;
            }
            if l.start < start {
                out.push(RecLock { end: start, ..l });
            }
            if end < l.end {
                out.push(RecLock { start: end, ..l });
            }
        }
        if typ != F_UNLCK {
            out.push(RecLock {
                owner,
                write: typ == F_WRLCK,
                start,
                end,
                pid,
            });
        }
        if out.is_empty() {
            self.records.remove(key);
        } else {
            *self.records.get_mut(key).expect("present") = out;
        }
    }

    /// Drop every lock `pred` matches; returns whether any went.
    fn drop_where(
        &mut self,
        key: Option<&FileKey>,
        rec: impl Fn(&RecLock) -> bool,
        fl: impl Fn(&Flock) -> bool,
    ) -> bool {
        let mut any = false;
        for (k, v) in &mut self.records {
            if key.is_none_or(|key| key == k) {
                let n = v.len();
                v.retain(|l| !rec(l));
                any |= v.len() != n;
            }
        }
        for (k, v) in &mut self.flocks {
            if key.is_none_or(|key| key == k) {
                let n = v.len();
                v.retain(|l| !fl(l));
                any |= v.len() != n;
            }
        }
        self.records.retain(|_, v| !v.is_empty());
        self.flocks.retain(|_, v| !v.is_empty());
        any
    }
}

/// Decode a `struct flock { i16 l_type; i16 l_whence; i64 l_start; i64
/// l_len; i32 l_pid; }` into `(type, start, end, pid)` given the descriptor's
/// position and the file's size (for `SEEK_CUR`/`SEEK_END`).
fn read_flock(
    mem: &GuestMemory,
    arg: u64,
    pos: u64,
    size: u64,
) -> Result<(i16, u64, u64, i32), i64> {
    let Ok(raw) = mem.read_vec(arg, 32) else {
        return Err(err(Errno::EFAULT));
    };
    let typ = i16::from_le_bytes([raw[0], raw[1]]);
    let whence = i16::from_le_bytes([raw[2], raw[3]]);
    let start = i64::from_le_bytes(raw[8..16].try_into().unwrap());
    let len = i64::from_le_bytes(raw[16..24].try_into().unwrap());
    let pid = i32::from_le_bytes(raw[24..28].try_into().unwrap());
    if !matches!(typ, F_RDLCK | F_WRLCK | F_UNLCK) {
        return Err(err(Errno::EINVAL));
    }
    let base: i64 = match whence {
        SEEK_SET => 0,
        SEEK_CUR => pos as i64,
        SEEK_END => size as i64,
        _ => return Err(err(Errno::EINVAL)),
    };
    let mut s = base
        .checked_add(start)
        .ok_or_else(|| err(Errno::EOVERFLOW))?;
    // l_len 0 = to EOF and beyond; a negative length covers the bytes
    // *before* l_start.
    let e: i64 = match len.cmp(&0) {
        std::cmp::Ordering::Equal => i64::MAX,
        std::cmp::Ordering::Greater => s.checked_add(len).ok_or_else(|| err(Errno::EOVERFLOW))?,
        std::cmp::Ordering::Less => {
            let e = s;
            s += len;
            e
        }
    };
    if s < 0 {
        return Err(err(Errno::EINVAL));
    }
    let end = if e == i64::MAX { u64::MAX } else { e as u64 };
    Ok((typ, s as u64, end, pid))
}

impl Kernel {
    /// The lock key of the file at `path`.
    fn lock_key(vfs: &mut MountTable, path: &str) -> Option<FileKey> {
        Self::pc_key(vfs, path)
    }

    /// Release the locks closing descriptor `f` drops: the process's POSIX
    /// locks on the file (any close does that), and — when `f` was the open
    /// file description's last descriptor — its OFD and `flock` locks. Wakes
    /// waiters if anything was released. `f` is the value just removed from a
    /// table (still alive here, so `refs() == 1` means "last").
    pub(super) fn release_fd_locks(&self, sh: &mut Shared, cx: &ServiceCtx, f: &Fd) {
        if !matches!(f, Fd::File { .. }) || self.locks.lock().unwrap().is_empty() {
            return;
        }
        let mut vfs = self.vfs.lock().unwrap();
        self.release_fd_locks_with(sh, cx, &mut vfs, f);
    }

    /// [`Self::release_fd_locks`] for a caller already holding `vfs`.
    pub(super) fn release_fd_locks_with(
        &self,
        sh: &mut Shared,
        cx: &ServiceCtx,
        vfs: &mut MountTable,
        f: &Fd,
    ) {
        let Fd::File { path, offset, .. } = f else {
            return;
        };
        if self.locks.lock().unwrap().is_empty() {
            return;
        }
        let key = Self::lock_key(vfs, path);
        let tgid = cx.cur.tgid;
        let last = offset.refs() == 1;
        let ofd = offset.id();
        let released = self.locks.lock().unwrap().drop_where(
            key.as_ref(),
            |l| l.owner == Owner::Process(tgid) || (last && l.owner == Owner::Ofd(ofd)),
            |l| last && l.ofd == ofd,
        );
        if released {
            sh.unpark_all();
        }
    }

    /// The thread group `tgid` is gone: drop its POSIX locks everywhere.
    pub(super) fn release_process_locks(&self, sh: &mut Shared, tgid: i32) {
        let mut locks = self.locks.lock().unwrap();
        if locks.is_empty() {
            return;
        }
        if locks.drop_where(None, |l| l.owner == Owner::Process(tgid), |_| false) {
            drop(locks);
            sh.unpark_all();
        }
    }

    /// The memfd seals for `path` (memfds only).
    pub(super) fn seals_of(&self, path: &str) -> Option<u32> {
        let l = self.locks.lock().unwrap();
        if l.seals.is_empty() {
            return None;
        }
        l.seals.get(path).copied()
    }

    /// Register a new memfd's seal state: sealable ones start with none, the
    /// rest start sealed against further sealing (`F_SEAL_SEAL`), as Linux.
    pub(super) fn memfd_register(&self, path: &str, allow_sealing: bool) {
        self.locks.lock().unwrap().seals.insert(
            path.to_string(),
            if allow_sealing { 0 } else { F_SEAL_SEAL },
        );
    }

    /// Whether the seals on `path` forbid writing `[off, off + len)` (`EPERM`).
    /// Costs a map probe; the file is only stat'ed when it is actually sealed.
    pub(super) fn seal_blocks_write(
        &self,
        vfs: &mut MountTable,
        path: &str,
        off: u64,
        len: u64,
    ) -> bool {
        self.seals_of(path).is_some_and(|s| {
            s & (F_SEAL_WRITE | F_SEAL_FUTURE_WRITE) != 0
                || (s & F_SEAL_GROW != 0 && off + len > vfs.stat(path).map_or(0, |a| a.size))
        })
    }

    /// Whether the seals on `path` forbid resizing it to `len`.
    pub(super) fn seal_blocks_truncate(&self, vfs: &mut MountTable, path: &str, len: u64) -> bool {
        self.seals_of(path).is_some_and(|s| {
            let size = vfs.stat(path).map_or(0, |a| a.size);
            (s & F_SEAL_SHRINK != 0 && len < size) || (s & F_SEAL_GROW != 0 && len > size)
        })
    }

    /// `fcntl(fd, cmd, arg)`.
    #[allow(clippy::too_many_lines)]
    pub(super) fn sys_fcntl(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        fd: u64,
        cmd: u64,
        arg: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const F_DUPFD: u64 = 0;
        const F_GETFD: u64 = 1;
        const F_SETFD: u64 = 2;
        const F_GETFL: u64 = 3;
        const F_SETFL: u64 = 4;
        const F_GETLK: u64 = 5;
        const F_SETLK: u64 = 6;
        const F_SETLKW: u64 = 7;
        const F_SETOWN: u64 = 8;
        const F_GETOWN: u64 = 9;
        const F_SETSIG: u64 = 10;
        const F_GETSIG: u64 = 11;
        const F_SETOWN_EX: u64 = 15;
        const F_GETOWN_EX: u64 = 16;
        const F_GETOWNER_UIDS: u64 = 17;
        const F_OFD_GETLK: u64 = 36;
        const F_OFD_SETLK: u64 = 37;
        const F_OFD_SETLKW: u64 = 38;
        const F_SETLEASE: u64 = 1024;
        const F_GETLEASE: u64 = 1025;
        const F_NOTIFY: u64 = 1026;
        const F_DUPFD_QUERY: u64 = 1027;
        const F_CREATED_QUERY: u64 = 1028;
        const F_CANCELLK: u64 = 1029;
        const F_DUPFD_CLOEXEC: u64 = 1030;
        const F_SETPIPE_SZ: u64 = 1031;
        const F_GETPIPE_SZ: u64 = 1032;
        const F_ADD_SEALS: u64 = 1033;
        const F_GET_SEALS: u64 = 1034;
        const F_GET_RW_HINT: u64 = 1035;
        const F_SET_RW_HINT: u64 = 1036;
        const F_GET_FILE_RW_HINT: u64 = 1037;
        const F_SET_FILE_RW_HINT: u64 = 1038;
        const FD_CLOEXEC: u64 = 1;
        const O_WRONLY: i64 = 1;
        const O_RDWR: i64 = 2;
        const O_APPEND: u64 = 0o2000;
        const O_NONBLOCK: u64 = 0o4000;
        let (o_directory, o_largefile): (i64, i64) = match self.arch {
            crate::abi::Arch::X86_64 => (0o200000, 0o100000),
            crate::abi::Arch::Aarch64 => (0o40000, 0o400000),
        };
        // Every fcntl command operates on an open fd. Returning success for a
        // closed fd breaks the common "mark every fd from 3 up cloexec until
        // EBADF" loop (node/libuv do this at startup) into an unbounded spin —
        // it must see EBADF to stop.
        let n = fd as i32;
        let Some(f) = cx.cur.fds.get(n).cloned() else {
            return err(Errno::EBADF);
        };
        match cmd {
            // Duplicate to the lowest free fd `>= arg`, optionally close-on-exec.
            F_DUPFD | F_DUPFD_CLOEXEC => {
                if !(0..4096).contains(&(arg as i32)) {
                    return err(Errno::EINVAL);
                }
                self.bump_pipe(&f, true);
                let m = cx.cur.fds.alloc_from(f, arg as i32);
                cx.cur.fds.set_cloexec(m, cmd == F_DUPFD_CLOEXEC);
                i64::from(m)
            }
            // Pipe capacity: nixvm's pipes are unbounded; report/accept the
            // Linux default (64 KiB). Only pipes have one.
            F_GETPIPE_SZ | F_SETPIPE_SZ => match f {
                Fd::PipeRead(_) | Fd::PipeWrite(_) if cmd == F_GETPIPE_SZ => 65536,
                Fd::PipeRead(_) | Fd::PipeWrite(_) => {
                    if arg > 1 << 20 && cx.cur.creds.euid != 0 {
                        err(Errno::EPERM)
                    } else {
                        (arg.max(4096).next_power_of_two()) as i64
                    }
                }
                _ => err(Errno::EBADF),
            },
            // The close-on-exec flag (`FD_CLOEXEC`) — the only `F_*FD` bit.
            F_GETFD => i64::from(cx.cur.fds.is_cloexec(n)),
            F_SETFD => {
                cx.cur.fds.set_cloexec(n, arg & FD_CLOEXEC != 0);
                0
            }
            // Status flags: O_NONBLOCK (wired to every subsystem's blocking
            // mode — libuv/c-ares rely on it) and O_APPEND (shared by the open
            // file description); O_ASYNC/O_DIRECT/O_NOATIME are accepted. The
            // access mode and creation flags can't change.
            F_SETFL => {
                self.fd_set_nonblock(&f, arg & O_NONBLOCK != 0);
                if let Some(Fd::Mqueue { flags, .. }) = cx.cur.fds.get_mut(n) {
                    *flags = (*flags & !O_NONBLOCK) | (arg & O_NONBLOCK);
                }
                cx.cur.fds.set_append(n, arg & O_APPEND != 0);
                0
            }
            F_GETFL => {
                let access = match &f {
                    Fd::File {
                        readable, writable, ..
                    } => {
                        let acc = match (readable, writable) {
                            (true, true) => O_RDWR,
                            (false, true) => O_WRONLY,
                            _ => 0,
                        };
                        acc | o_largefile
                    }
                    Fd::Dir { .. } => o_directory | o_largefile,
                    Fd::PipeRead(_) => 0,
                    Fd::PipeWrite(_) => O_WRONLY,
                    Fd::Mqueue { flags, .. } => (*flags & 3) as i64,
                    _ => O_RDWR,
                };
                let nb = if self.fd_is_nonblock(&f)
                    || matches!(f, Fd::Mqueue { flags, .. } if flags & O_NONBLOCK != 0)
                {
                    O_NONBLOCK as i64
                } else {
                    0
                };
                let ap = if cx.cur.fds.is_append(n) {
                    O_APPEND as i64
                } else {
                    0
                };
                access | nb | ap
            }
            F_SETLK | F_SETLKW | F_OFD_SETLK | F_OFD_SETLKW | F_GETLK | F_OFD_GETLK => {
                self.fcntl_lock(sh, cx, &f, cmd, arg, mem)
            }
            F_CANCELLK => err(Errno::EINVAL),
            // Signal-driven I/O ownership: recorded and reported (no SIGIO is
            // ever raised — readiness is only observable through poll/epoll).
            F_SETOWN => {
                let who = arg as i32;
                cx.cur.fds.meta_mut(n).owner = if who < 0 { (2, -who) } else { (1, who) };
                0
            }
            F_GETOWN => {
                let (typ, who) = cx.cur.fds.meta(n).owner;
                i64::from(if typ == 2 { -who } else { who })
            }
            F_SETOWN_EX => {
                let (Ok(typ), Ok(who)) = (mem.read_u32(arg), mem.read_u32(arg + 4)) else {
                    return err(Errno::EFAULT);
                };
                if typ > 2 {
                    return err(Errno::EINVAL);
                }
                cx.cur.fds.meta_mut(n).owner = (typ as i32, who as i32);
                0
            }
            F_GETOWN_EX => {
                let (typ, who) = cx.cur.fds.meta(n).owner;
                let mut b = [0u8; 8];
                b[0..4].copy_from_slice(&typ.to_le_bytes());
                b[4..8].copy_from_slice(&who.to_le_bytes());
                if mem.write(arg, &b).is_err() {
                    return err(Errno::EFAULT);
                }
                0
            }
            F_GETOWNER_UIDS => {
                let uid = cx.cur.creds.ruid;
                let euid = cx.cur.creds.euid;
                let mut b = [0u8; 8];
                b[0..4].copy_from_slice(&uid.to_le_bytes());
                b[4..8].copy_from_slice(&euid.to_le_bytes());
                if mem.write(arg, &b).is_err() {
                    return err(Errno::EFAULT);
                }
                0
            }
            F_SETSIG => {
                if arg > 64 {
                    return err(Errno::EINVAL);
                }
                cx.cur.fds.meta_mut(n).sig = arg as i32;
                0
            }
            F_GETSIG => i64::from(cx.cur.fds.meta(n).sig),
            // Leases: only on regular files; a write lease needs the only
            // open descriptor. Recorded; there is nobody else to break them.
            F_SETLEASE => {
                if !matches!(f, Fd::File { .. }) {
                    return err(Errno::EINVAL);
                }
                let typ = arg as i16;
                if !matches!(typ, F_RDLCK | F_WRLCK | F_UNLCK) {
                    return err(Errno::EINVAL);
                }
                if typ == F_WRLCK
                    && let Fd::File { offset, .. } = &f
                    && offset.refs() > 2
                {
                    return err(Errno::EAGAIN);
                }
                cx.cur.fds.meta_mut(n).lease = i32::from(typ);
                0
            }
            F_GETLEASE => i64::from(cx.cur.fds.meta(n).lease),
            // dnotify: directories only; recorded (no events are delivered).
            F_NOTIFY => {
                if !matches!(f, Fd::Dir { .. }) {
                    return err(Errno::ENOTDIR);
                }
                cx.cur.fds.meta_mut(n).notify = arg;
                0
            }
            // Is `arg` the same open file as `fd`?
            F_DUPFD_QUERY => match cx.cur.fds.get(arg as i32) {
                None => err(Errno::EBADF),
                Some(Fd::File { offset: a, .. }) => {
                    i64::from(matches!(&f, Fd::File { offset: b, .. } if a.same(b)))
                }
                Some(g) => i64::from(format!("{g:?}") == format!("{f:?}")),
            },
            // Whether this open created the file — not tracked: "no".
            F_CREATED_QUERY => 0,
            F_ADD_SEALS | F_GET_SEALS => self.fcntl_seals(&f, cmd == F_ADD_SEALS, arg),
            F_GET_RW_HINT | F_GET_FILE_RW_HINT => {
                let h = cx.cur.fds.meta(n).rw_hint;
                if mem.write_u64(arg, h).is_err() {
                    return err(Errno::EFAULT);
                }
                0
            }
            F_SET_RW_HINT | F_SET_FILE_RW_HINT => match mem.read_u64(arg) {
                Ok(h) if h <= 5 => {
                    cx.cur.fds.meta_mut(n).rw_hint = h;
                    0
                }
                Ok(_) => err(Errno::EINVAL),
                Err(_) => err(Errno::EFAULT),
            },
            _ => {
                self.note_unsupported("fcntl", cmd);
                err(Errno::EINVAL)
            }
        }
    }

    /// `F_ADD_SEALS`/`F_GET_SEALS` on a memfd (`EINVAL` on anything else).
    fn fcntl_seals(&self, f: &Fd, add: bool, arg: u64) -> i64 {
        let Fd::File { path, writable, .. } = f else {
            return err(Errno::EINVAL);
        };
        let mut l = self.locks.lock().unwrap();
        let Some(cur) = l.seals.get(path).copied() else {
            return err(Errno::EINVAL);
        };
        if !add {
            return i64::from(cur);
        }
        let new = arg as u32;
        if new
            & !(F_SEAL_SEAL
                | F_SEAL_SHRINK
                | F_SEAL_GROW
                | F_SEAL_WRITE
                | F_SEAL_FUTURE_WRITE
                | F_SEAL_EXEC)
            != 0
        {
            return err(Errno::EINVAL);
        }
        if !writable {
            return err(Errno::EPERM);
        }
        if cur & F_SEAL_SEAL != 0 {
            return err(Errno::EPERM);
        }
        l.seals.insert(path.clone(), cur | new);
        0
    }

    /// The record-lock commands.
    #[allow(clippy::too_many_arguments)]
    fn fcntl_lock(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        f: &Fd,
        cmd: u64,
        arg: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const F_GETLK: u64 = 5;
        const F_SETLKW: u64 = 7;
        const F_OFD_GETLK: u64 = 36;
        const F_OFD_SETLKW: u64 = 38;
        let ofd_cmd = cmd >= 36;
        // Locks only apply to regular files; other descriptors have nothing
        // to contend over (Linux accepts them too), so grant/report "free".
        let Fd::File {
            path,
            offset,
            readable,
            writable,
        } = f
        else {
            if matches!(cmd, F_GETLK | F_OFD_GETLK)
                && mem.write(arg, &F_UNLCK.to_le_bytes()).is_err()
            {
                return err(Errno::EFAULT);
            }
            return 0;
        };
        let (key, size) = {
            let mut vfs = self.vfs.lock().unwrap();
            let size = vfs.stat(path).map_or(0, |a| a.size);
            (Self::lock_key(&mut vfs, path), size)
        };
        let Some(key) = key else {
            return err(Errno::EBADF);
        };
        let (typ, start, end, l_pid) = match read_flock(mem, arg, offset.get(), size) {
            Ok(v) => v,
            Err(e) => return e,
        };
        if ofd_cmd && l_pid != 0 {
            return err(Errno::EINVAL);
        }
        let owner = if ofd_cmd {
            Owner::Ofd(offset.id())
        } else {
            Owner::Process(cx.cur.tgid)
        };
        let mut locks = self.locks.lock().unwrap();
        if matches!(cmd, F_GETLK | F_OFD_GETLK) {
            if typ == F_UNLCK {
                return err(Errno::EINVAL);
            }
            let mut b = [0u8; 32];
            match locks.conflict(&key, owner, typ == F_WRLCK, start, end) {
                None => b[0..2].copy_from_slice(&F_UNLCK.to_le_bytes()),
                Some(l) => {
                    b[0..2]
                        .copy_from_slice(&(if l.write { F_WRLCK } else { F_RDLCK }).to_le_bytes());
                    b[8..16].copy_from_slice(&(l.start as i64).to_le_bytes());
                    let len = if l.end == u64::MAX {
                        0
                    } else {
                        (l.end - l.start) as i64
                    };
                    b[16..24].copy_from_slice(&len.to_le_bytes());
                    b[24..28].copy_from_slice(&l.pid.to_le_bytes());
                }
            }
            return if mem.write(arg, &b[..28]).is_ok() {
                0
            } else {
                err(Errno::EFAULT)
            };
        }
        // A read lock needs a readable descriptor, a write lock a writable one.
        if (typ == F_RDLCK && !readable) || (typ == F_WRLCK && !writable) {
            return err(Errno::EBADF);
        }
        if typ != F_UNLCK
            && locks
                .conflict(&key, owner, typ == F_WRLCK, start, end)
                .is_some()
        {
            return if matches!(cmd, F_SETLKW | F_OFD_SETLKW) {
                cx.block = true;
                0
            } else {
                err(Errno::EAGAIN)
            };
        }
        let pid = if ofd_cmd { -1 } else { cx.cur.tgid };
        locks.set(&key, owner, typ, start, end, pid);
        drop(locks);
        if typ != F_WRLCK {
            // Unlocking or downgrading may let a waiter in.
            sh.unpark_all();
        }
        0
    }

    /// `flock(fd, operation)`: whole-file advisory locks owned by the open
    /// file description. `LOCK_SH`/`LOCK_EX` convert any lock the description
    /// already holds; a conflict blocks (or `EWOULDBLOCK` with `LOCK_NB`).
    pub(super) fn sys_flock(&self, sh: &mut Shared, cx: &mut ServiceCtx, fd: u64, op: u64) -> i64 {
        const LOCK_SH: u64 = 1;
        const LOCK_EX: u64 = 2;
        const LOCK_NB: u64 = 4;
        const LOCK_UN: u64 = 8;
        let Some(f) = cx.cur.fds.get(fd as i32).cloned() else {
            return err(Errno::EBADF);
        };
        let kind = op & !LOCK_NB;
        if !matches!(kind, LOCK_SH | LOCK_EX | LOCK_UN) {
            return err(Errno::EINVAL);
        }
        let Fd::File { path, offset, .. } = &f else {
            return 0; // nothing contends over a non-file descriptor
        };
        let key = {
            let mut vfs = self.vfs.lock().unwrap();
            Self::lock_key(&mut vfs, path)
        };
        let Some(key) = key else {
            return err(Errno::EBADF);
        };
        let ofd = offset.id();
        let mut locks = self.locks.lock().unwrap();
        let list = locks.flocks.entry(key.clone()).or_default();
        if kind == LOCK_UN {
            list.retain(|l| l.ofd != ofd);
            if list.is_empty() {
                locks.flocks.remove(&key);
            }
            drop(locks);
            sh.unpark_all();
            return 0;
        }
        let exclusive = kind == LOCK_EX;
        let blocked = list
            .iter()
            .any(|l| l.ofd != ofd && (exclusive || l.exclusive));
        if blocked {
            if list.is_empty() {
                locks.flocks.remove(&key);
            }
            return if op & LOCK_NB != 0 {
                err(Errno::EAGAIN)
            } else {
                cx.block = true;
                0
            };
        }
        list.retain(|l| l.ofd != ofd);
        list.push(Flock { ofd, exclusive });
        if !exclusive {
            drop(locks);
            sh.unpark_all();
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{BASE, call, e, put_str, setup};
    use crate::abi::arch::Sysno;
    use crate::abi::errno::Errno;

    const AT_FDCWD: u64 = (-100i64) as u64;

    fn flock_struct(mem: &mut crate::vcpu::GuestMemory, at: u64, typ: i16, start: i64, len: i64) {
        let mut b = [0u8; 32];
        b[0..2].copy_from_slice(&typ.to_le_bytes());
        b[8..16].copy_from_slice(&start.to_le_bytes());
        b[16..24].copy_from_slice(&len.to_le_bytes());
        mem.write(at, &b).unwrap();
    }

    #[test]
    fn posix_locks_conflict_across_processes_and_drop_on_close() {
        let (k, mut mem, mut v, mut cx) = setup();
        let (path, fl) = (BASE, BASE + 0x100);
        put_str(&mut mem, path, "/db");
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_FDCWD, path, 0o102, 0o600, 0, 0],
        ) as u64;
        flock_struct(&mut mem, fl, 1, 0, 100); // write-lock [0,100)
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Fcntl,
                [fd, 6, fl, 0, 0, 0]
            ),
            0
        );
        // Another process (tgid 2) can't take an overlapping read lock…
        let me = cx.cur.tgid;
        cx.cur.tgid = 2;
        flock_struct(&mut mem, fl, 0, 50, 10);
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Fcntl,
                [fd, 6, fl, 0, 0, 0]
            ),
            e(Errno::EAGAIN)
        );
        // …F_GETLK reports the holder…
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Fcntl,
                [fd, 5, fl, 0, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_vec(fl, 2).unwrap(), 1i16.to_le_bytes());
        assert_eq!(mem.read_u32(fl + 24).unwrap(), me as u32);
        // …F_SETLKW parks, and a disjoint range is fine.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Fcntl,
                [fd, 7, fl, 0, 0, 0]
            ),
            0
        );
        assert!(cx.block);
        cx.block = false;
        flock_struct(&mut mem, fl, 0, 100, 10);
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Fcntl,
                [fd, 6, fl, 0, 0, 0]
            ),
            0
        );
        // Back as the owner: closing *a* descriptor of the file drops its locks.
        cx.cur.tgid = me;
        let dup = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Dup,
            [fd, 0, 0, 0, 0, 0],
        ) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Close,
                [dup, 0, 0, 0, 0, 0]
            ),
            0
        );
        cx.cur.tgid = 2;
        flock_struct(&mut mem, fl, 1, 0, 50);
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Fcntl,
                [fd, 6, fl, 0, 0, 0]
            ),
            0
        );
    }

    #[test]
    fn flock_is_per_open_file_description() {
        let (k, mut mem, mut v, mut cx) = setup();
        let path = BASE;
        put_str(&mut mem, path, "/lk");
        let a = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_FDCWD, path, 0o102, 0o600, 0, 0],
        ) as u64;
        let b = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_FDCWD, path, 2, 0, 0, 0],
        ) as u64;
        let a2 = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Dup,
            [a, 0, 0, 0, 0, 0],
        ) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Flock,
                [a, 2, 0, 0, 0, 0]
            ),
            0
        );
        // A dup shares the lock; a separate open conflicts.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Flock,
                [a2, 2 | 4, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Flock,
                [b, 1 | 4, 0, 0, 0, 0]
            ),
            e(Errno::EAGAIN)
        );
        // Closing one of two descriptors keeps it; closing the last frees it.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Close,
                [a, 0, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Flock,
                [b, 1 | 4, 0, 0, 0, 0]
            ),
            e(Errno::EAGAIN)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Close,
                [a2, 0, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Flock,
                [b, 1 | 4, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Flock,
                [b, 16, 0, 0, 0, 0]
            ),
            e(Errno::EINVAL)
        );
    }

    #[test]
    fn getfl_owner_and_seals() {
        let (k, mut mem, mut v, mut cx) = setup();
        let (path, buf) = (BASE, BASE + 0x100);
        put_str(&mut mem, path, "/ro");
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_FDCWD, path, 0o101, 0o600, 0, 0],
        ) as u64;
        // O_WRONLY | O_LARGEFILE (arm64 value).
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Fcntl,
                [fd, 3, 0, 0, 0, 0]
            ),
            1 | 0o400000
        );
        // O_APPEND set through one descriptor shows through its dup.
        let d = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Dup,
            [fd, 0, 0, 0, 0, 0],
        ) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Fcntl,
                [fd, 4, 0o2000, 0, 0, 0]
            ),
            0
        );
        assert_ne!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Fcntl,
                [d, 3, 0, 0, 0, 0]
            ) & 0o2000,
            0
        );
        // F_SETOWN with a negative id is a process group.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Fcntl,
                [fd, 8, (-7i64) as u64, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Fcntl,
                [fd, 9, 0, 0, 0, 0]
            ),
            -7
        );
        // Unknown command: EINVAL. Seals on a non-memfd: EINVAL.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Fcntl,
                [fd, 999, 0, 0, 0, 0]
            ),
            e(Errno::EINVAL)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Fcntl,
                [fd, 1034, 0, 0, 0, 0]
            ),
            e(Errno::EINVAL)
        );
        // A sealable memfd: seal against writes, then writes fail EPERM.
        put_str(&mut mem, path, "m");
        let m = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::MemfdCreate,
            [path, 2, 0, 0, 0, 0],
        ) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Fcntl,
                [m, 1034, 0, 0, 0, 0]
            ),
            0
        );
        mem.write(buf, b"abc").unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [m, buf, 3, 0, 0, 0]
            ),
            3
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Fcntl,
                [m, 1033, 0x8 | 0x2, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [m, buf, 3, 0, 0, 0]
            ),
            e(Errno::EPERM)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Ftruncate,
                [m, 0, 0, 0, 0, 0]
            ),
            e(Errno::EPERM)
        );
        // A memfd without MFD_ALLOW_SEALING can't be sealed.
        let m2 = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::MemfdCreate,
            [path, 0, 0, 0, 0, 0],
        ) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Fcntl,
                [m2, 1034, 0, 0, 0, 0]
            ),
            1
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Fcntl,
                [m2, 1033, 0x8, 0, 0, 0]
            ),
            e(Errno::EPERM)
        );
    }
}
