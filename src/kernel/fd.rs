//! The per-process file-descriptor table.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// The shared state of an *open file description*: its file position and
/// its `O_APPEND` status flag. Shared — not copied — by every descriptor that
/// refers to the same open file: a `dup`, a `fork` child's inherited
/// descriptor, an `SCM_RIGHTS` copy. So when a shell runs `{ cmd1; cmd2; } >
/// out`, each child's writes advance the one position its siblings and
/// parent continue from (instead of every writer starting over at offset 0
/// and clobbering the previous output), and `>> log` followed by
/// `dup2(fd, 1)` keeps appending through fd 1. A fresh `open` makes a new
/// one. The pointer identity doubles as the description's identity
/// (`kcmp(KCMP_FILE)`, `F_DUPFD_QUERY`, OFD/`flock` lock ownership), and the
/// reference count says when its last descriptor closed.
#[derive(Debug, Clone)]
pub struct FileOffset(Arc<OfdState>);

#[derive(Debug)]
struct OfdState {
    pos: AtomicU64,
    append: AtomicBool,
}

impl FileOffset {
    /// A new open file description positioned at `pos`.
    #[must_use]
    pub fn new(pos: u64) -> Self {
        Self(Arc::new(OfdState {
            pos: AtomicU64::new(pos),
            append: AtomicBool::new(false),
        }))
    }
    #[must_use]
    pub fn get(&self) -> u64 {
        self.0.pos.load(Ordering::Relaxed)
    }
    pub fn set(&self, pos: u64) {
        self.0.pos.store(pos, Ordering::Relaxed);
    }
    pub fn add(&self, n: u64) {
        self.0.pos.fetch_add(n, Ordering::Relaxed);
    }
    /// The description's `O_APPEND` flag.
    #[must_use]
    pub fn append(&self) -> bool {
        self.0.append.load(Ordering::Relaxed)
    }
    pub fn set_append(&self, on: bool) {
        self.0.append.store(on, Ordering::Relaxed);
    }
    /// Whether two descriptors share this open file description.
    #[must_use]
    pub fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    /// A stable identity for the open file description.
    #[must_use]
    pub fn id(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }
    /// How many descriptors (across every task) refer to the description —
    /// 1 means the one about to be dropped is the last.
    #[must_use]
    pub fn refs(&self) -> usize {
        Arc::strong_count(&self.0)
    }
}

/// Per-descriptor attributes `fcntl` records and reports back: the
/// `F_SETOWN[_EX]` owner (`(F_OWNER_* type, id)`), the `F_SETSIG` signal, the
/// `F_SETLEASE` lease, the `F_NOTIFY` mask, and the `F_SET_RW_HINT` hint.
/// (Linux keeps several of these per open file description; per descriptor is
/// the approximation, and nothing is delivered through them — no `SIGIO`,
/// lease breaks or dnotify events.)
#[derive(Debug, Clone, Copy)]
pub struct FdMeta {
    pub owner: (i32, i32),
    pub sig: i32,
    pub lease: i32,
    pub notify: u64,
    pub rw_hint: u64,
}

impl Default for FdMeta {
    fn default() -> Self {
        Self {
            // F_OWNER_PID with no owner; no signal; F_UNLCK; no notify; NOT_SET.
            owner: (1, 0),
            sig: 0,
            lease: 2,
            notify: 0,
            rw_hint: 0,
        }
    }
}

/// What a guest file descriptor points at.
///
/// Expanded as backends land: pipes (Phase 7), sockets (Phase 8), epoll/timerfd
/// (Phase 7).
#[derive(Debug, Clone)]
pub enum Fd {
    Stdin,
    Stdout,
    Stderr,
    /// An open path in the [`crate::fs::MountTable`], with the current offset.
    File {
        path: String,
        /// The (shared) position of the open file description.
        offset: FileOffset,
        /// Whether the open access mode permits reading (`O_RDONLY`/`O_RDWR`).
        /// A `read`/`pread` on a write-only fd must fail `EBADF`.
        readable: bool,
        /// Whether the open access mode permits writing (`O_WRONLY`/`O_RDWR`).
        /// A `write`/`pwrite`/`ftruncate` on a read-only fd must fail `EBADF`.
        writable: bool,
    },
    /// An open directory being walked by `getdents64`.
    Dir {
        path: String,
        pos: usize,
    },
    /// Read end of pipe `index` in the kernel's pipe table.
    PipeRead(usize),
    /// Write end of pipe `index` in the kernel's pipe table.
    PipeWrite(usize),
    /// An endpoint of socket `sock` in the kernel's socket table. `end` is 0 or
    /// 1, selecting which side of a connected pair (and thus which direction is
    /// read vs. written). Unconnected/listening sockets always use `end == 0`.
    Socket {
        sock: usize,
        end: usize,
    },
    /// An `eventfd2` counter: index into the kernel's eventfd table.
    Eventfd(usize),
    /// A `signalfd4`: index into the kernel's signalfd table.
    Signalfd(usize),
    /// A `timerfd_create` timer: index into the kernel's timerfd table.
    Timerfd(usize),
    /// A `CLONE_PIDFD` process descriptor: index into the kernel's pidfd table.
    /// Becomes `POLLIN`-readable when the referenced process exits; `read` on it
    /// is `EINVAL` (matching Linux — a pidfd carries no data).
    Pidfd(usize),
    /// An `epoll_create1` instance: index into the kernel's epoll table.
    Epoll(usize),
    /// The master end of pseudo-terminal `index` (`/dev/ptmx`).
    PtyMaster(usize),
    /// A slave end of pseudo-terminal `index` (`/dev/pts/index`).
    PtySlave(usize),
    /// A POSIX message-queue descriptor (`mq_open`): queue `q` in the poll
    /// subsystem's table, opened with `flags` (`O_ACCMODE | O_NONBLOCK`).
    Mqueue {
        q: usize,
        flags: u64,
    },
}

/// Maps small integer descriptors to [`Fd`]s, allocating the lowest free number.
#[derive(Debug, Clone, Default)]
pub struct FdTable {
    map: BTreeMap<i32, Fd>,
    /// Descriptors with `FD_CLOEXEC` set (`O_CLOEXEC`/`SOCK_CLOEXEC`/
    /// `F_DUPFD_CLOEXEC`/`fcntl(F_SETFD)`): closed on `execve`, inherited on
    /// `fork` (the whole table is cloned).
    cloexec: BTreeSet<i32>,
    /// `fcntl`-recorded per-descriptor attributes (see [`FdMeta`]); absent =
    /// the defaults.
    meta: BTreeMap<i32, FdMeta>,
}

impl FdTable {
    /// A fresh table with 0/1/2 wired to the host stdio.
    #[must_use]
    pub fn with_standard_streams() -> Self {
        let mut map = BTreeMap::new();
        map.insert(0, Fd::Stdin);
        map.insert(1, Fd::Stdout);
        map.insert(2, Fd::Stderr);
        Self {
            map,
            cloexec: BTreeSet::new(),
            meta: BTreeMap::new(),
        }
    }

    /// Insert `fd` at the lowest available descriptor, as POSIX `open` requires.
    /// This is normally 3 (0/1/2 hold stdio), but a program that closes one of
    /// the standard streams and reopens gets it back at that number — busybox
    /// ash relies on exactly this for background jobs: it does `close(0);
    /// open("/dev/null")` and *dies* unless the reopen lands on fd 0.
    pub fn alloc(&mut self, fd: Fd) -> i32 {
        self.alloc_from(fd, 0)
    }

    /// Allocate the lowest free descriptor `>= min` — POSIX `dup`/`fcntl(F_DUPFD)`
    /// semantics (also the base of [`Self::alloc`], with `min == 0`).
    pub fn alloc_from(&mut self, fd: Fd, min: i32) -> i32 {
        let mut n = min.max(0);
        while self.map.contains_key(&n) {
            n += 1;
        }
        self.map.insert(n, fd);
        n
    }

    /// Place `fd` at a specific descriptor number, replacing any existing entry
    /// (which is returned). Used by `dup2`/`dup3`. The new descriptor starts
    /// without `FD_CLOEXEC` (dup2 clears it; dup3 sets it afterward if asked).
    pub fn insert(&mut self, n: i32, fd: Fd) -> Option<Fd> {
        self.cloexec.remove(&n);
        self.meta.remove(&n);
        self.map.insert(n, fd)
    }

    /// Set or clear `FD_CLOEXEC` on `n` (a no-op if `n` isn't open).
    pub fn set_cloexec(&mut self, n: i32, on: bool) {
        if !self.map.contains_key(&n) {
            return;
        }
        if on {
            self.cloexec.insert(n);
        } else {
            self.cloexec.remove(&n);
        }
    }

    #[must_use]
    pub fn is_cloexec(&self, n: i32) -> bool {
        self.cloexec.contains(&n)
    }

    /// Set `O_APPEND` on the open file description behind `n` (shared with
    /// its dups; a no-op for anything but a regular file).
    pub fn set_append(&mut self, n: i32, on: bool) {
        if let Some(Fd::File { offset, .. }) = self.map.get(&n) {
            offset.set_append(on);
        }
    }

    /// Whether the open file description behind `n` is `O_APPEND`.
    #[must_use]
    pub fn is_append(&self, n: i32) -> bool {
        matches!(self.map.get(&n), Some(Fd::File { offset, .. }) if offset.append())
    }

    /// The `fcntl` attributes recorded for `n`.
    #[must_use]
    pub fn meta(&self, n: i32) -> FdMeta {
        self.meta.get(&n).copied().unwrap_or_default()
    }

    /// Mutable `fcntl` attributes of `n` (created with the defaults).
    pub fn meta_mut(&mut self, n: i32) -> &mut FdMeta {
        self.meta.entry(n).or_default()
    }

    /// Close every `FD_CLOEXEC` descriptor, returning the removed [`Fd`]s so the
    /// caller can drop backing refcounts (pipes/sockets). Runs on `execve`.
    pub fn close_cloexec(&mut self) -> Vec<Fd> {
        let fds: Vec<i32> = std::mem::take(&mut self.cloexec).into_iter().collect();
        fds.into_iter()
            .filter_map(|n| {
                self.meta.remove(&n);
                self.map.remove(&n)
            })
            .collect()
    }

    #[must_use]
    pub fn get(&self, fd: i32) -> Option<&Fd> {
        self.map.get(&fd)
    }

    pub fn get_mut(&mut self, fd: i32) -> Option<&mut Fd> {
        self.map.get_mut(&fd)
    }

    pub fn close(&mut self, fd: i32) -> Option<Fd> {
        self.cloexec.remove(&fd);
        self.meta.remove(&fd);
        self.map.remove(&fd)
    }

    /// Iterate over the open descriptors (used to adjust pipe refcounts on
    /// `fork` and `exit`).
    pub fn values(&self) -> impl Iterator<Item = &Fd> {
        self.map.values()
    }

    /// Iterate over `(fd number, descriptor)` pairs, ascending (backs the live
    /// `/proc/self/fd/` listing).
    pub fn iter(&self) -> impl Iterator<Item = (i32, &Fd)> {
        self.map.iter().map(|(&n, fd)| (n, fd))
    }

    /// Remove every descriptor, returning them (used on process exit).
    pub fn drain(&mut self) -> Vec<Fd> {
        self.cloexec.clear();
        self.meta.clear();
        std::mem::take(&mut self.map).into_values().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_starts_at_three_and_fills_gaps() {
        let mut t = FdTable::with_standard_streams();
        assert_eq!(t.alloc(Fd::Stdin), 3);
        assert_eq!(t.alloc(Fd::Stdin), 4);
        t.close(3);
        assert_eq!(t.alloc(Fd::Stdin), 3);
    }

    #[test]
    fn alloc_reuses_a_closed_standard_stream() {
        // POSIX `open` returns the lowest free fd — including 0/1/2 once closed.
        // busybox ash's background-job setup (`close(0); open("/dev/null")`)
        // dies unless the reopen lands back on fd 0.
        let mut t = FdTable::with_standard_streams();
        t.close(0);
        assert_eq!(t.alloc(Fd::Stdin), 0);
    }

    #[test]
    fn alloc_from_honors_the_minimum() {
        let mut t = FdTable::with_standard_streams();
        assert_eq!(t.alloc_from(Fd::Stdin, 10), 10, "lowest free >= 10");
        assert_eq!(t.alloc_from(Fd::Stdin, 10), 11, "then the next free");
    }

    #[test]
    fn cloexec_tracked_inherited_and_closed_on_exec() {
        let mut t = FdTable::with_standard_streams();
        let c = t.alloc(Fd::Stdin);
        let keep = t.alloc(Fd::Stdout);
        t.set_cloexec(c, true);
        assert!(t.is_cloexec(c) && !t.is_cloexec(keep));
        // fork inherits the flag (the whole table is cloned).
        let forked = t.clone();
        assert!(forked.is_cloexec(c));
        // execve closes the cloexec fd, keeps the rest.
        let closed = t.close_cloexec();
        assert_eq!(closed.len(), 1);
        assert!(t.get(c).is_none(), "cloexec fd closed on exec");
        assert!(t.get(keep).is_some(), "plain fd survives exec");
        assert!(!t.is_cloexec(c));
    }
}
