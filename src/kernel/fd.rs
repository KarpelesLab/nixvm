//! The per-process file-descriptor table.

use std::collections::{BTreeMap, BTreeSet};

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
        offset: u64,
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
    /// An `epoll_create1` instance: index into the kernel's epoll table.
    Epoll(usize),
    /// The master end of pseudo-terminal `index` (`/dev/ptmx`).
    PtyMaster(usize),
    /// A slave end of pseudo-terminal `index` (`/dev/pts/index`).
    PtySlave(usize),
}

/// Maps small integer descriptors to [`Fd`]s, allocating the lowest free number.
#[derive(Debug, Clone, Default)]
pub struct FdTable {
    map: BTreeMap<i32, Fd>,
    /// Descriptors with `FD_CLOEXEC` set (`O_CLOEXEC`/`SOCK_CLOEXEC`/
    /// `F_DUPFD_CLOEXEC`/`fcntl(F_SETFD)`): closed on `execve`, inherited on
    /// `fork` (the whole table is cloned).
    cloexec: BTreeSet<i32>,
    /// Descriptors opened `O_APPEND`: every write seeks to end-of-file first.
    append: BTreeSet<i32>,
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
            append: BTreeSet::new(),
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

    /// Mark `n` as `O_APPEND` (a no-op if not open).
    pub fn set_append(&mut self, n: i32, on: bool) {
        if !self.map.contains_key(&n) {
            return;
        }
        if on {
            self.append.insert(n);
        } else {
            self.append.remove(&n);
        }
    }

    #[must_use]
    pub fn is_append(&self, n: i32) -> bool {
        self.append.contains(&n)
    }

    /// Close every `FD_CLOEXEC` descriptor, returning the removed [`Fd`]s so the
    /// caller can drop backing refcounts (pipes/sockets). Runs on `execve`.
    pub fn close_cloexec(&mut self) -> Vec<Fd> {
        let fds: Vec<i32> = std::mem::take(&mut self.cloexec).into_iter().collect();
        fds.into_iter().filter_map(|n| self.map.remove(&n)).collect()
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
        self.append.remove(&fd);
        self.map.remove(&fd)
    }

    /// Iterate over the open descriptors (used to adjust pipe refcounts on
    /// `fork` and `exit`).
    pub fn values(&self) -> impl Iterator<Item = &Fd> {
        self.map.values()
    }

    /// Remove every descriptor, returning them (used on process exit).
    pub fn drain(&mut self) -> Vec<Fd> {
        self.cloexec.clear();
        self.append.clear();
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
