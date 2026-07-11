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
    File { path: String, offset: u64 },
    /// An open directory being walked by `getdents64`.
    Dir { path: String, pos: usize },
    /// Read end of pipe `index` in the kernel's pipe table.
    PipeRead(usize),
    /// Write end of pipe `index` in the kernel's pipe table.
    PipeWrite(usize),
}

/// Maps small integer descriptors to [`Fd`]s, allocating the lowest free number.
#[derive(Debug)]
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
