//! The per-process file-descriptor table.

use std::collections::BTreeMap;

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
    /// A `timerfd_create` timer: index into the kernel's timerfd table.
    Timerfd(usize),
    /// An `epoll_create1` instance: index into the kernel's epoll table.
    Epoll(usize),
}

/// Maps small integer descriptors to [`Fd`]s, allocating the lowest free number.
#[derive(Debug, Clone, Default)]
pub struct FdTable {
    map: BTreeMap<i32, Fd>,
}

impl FdTable {
    /// A fresh table with 0/1/2 wired to the host stdio.
    #[must_use]
    pub fn with_standard_streams() -> Self {
        let mut map = BTreeMap::new();
        map.insert(0, Fd::Stdin);
        map.insert(1, Fd::Stdout);
        map.insert(2, Fd::Stderr);
        Self { map }
    }

    /// Insert `fd` at the lowest available descriptor >= 3.
    pub fn alloc(&mut self, fd: Fd) -> i32 {
        let mut n = 3;
        while self.map.contains_key(&n) {
            n += 1;
        }
        self.map.insert(n, fd);
        n
    }

    /// Place `fd` at a specific descriptor number, replacing any existing entry
    /// (which is returned). Used by `dup2`/`dup3`.
    pub fn insert(&mut self, n: i32, fd: Fd) -> Option<Fd> {
        self.map.insert(n, fd)
    }

    #[must_use]
    pub fn get(&self, fd: i32) -> Option<&Fd> {
        self.map.get(&fd)
    }

    pub fn get_mut(&mut self, fd: i32) -> Option<&mut Fd> {
        self.map.get_mut(&fd)
    }

    pub fn close(&mut self, fd: i32) -> Option<Fd> {
        self.map.remove(&fd)
    }

    /// Iterate over the open descriptors (used to adjust pipe refcounts on
    /// `fork` and `exit`).
    pub fn values(&self) -> impl Iterator<Item = &Fd> {
        self.map.values()
    }

    /// Remove every descriptor, returning them (used on process exit).
    pub fn drain(&mut self) -> Vec<Fd> {
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
}
