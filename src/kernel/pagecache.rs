//! A page cache for `MAP_SHARED` file mappings, so a shared file mapping is
//! *shared*: every process (and every `mmap` within one) that maps the same
//! page of the same file maps the same physical frame, a `fork` child aliases
//! it, and `read`/`write` on the file stay coherent with the mappings.
//!
//! Before this, a `MAP_SHARED` file mapping was a private copy written back on
//! `munmap`, so two mappers never saw each other's stores — which broke every
//! process-shared object living in a file: musl/glibc named semaphores and
//! POSIX shared memory in `/dev/shm` (Python `multiprocessing` locks, queues,
//! `shared_memory`), SQLite's WAL index, LMDB. Pages are keyed by the file's
//! identity — `(mount point, inode)` — not its path, so a rename (or an unlink
//! of a still-mapped file) keeps the pages. The path-keyed in-memory
//! filesystems copy on `link`, so a hard link is a different file here; that
//! remains a limitation.
//!
//! Frames are allocated from the shared pool, filled from the file, and owned
//! by the cache (one reference) while also mapped (one reference per mapping,
//! tagged shared so `fork` aliases instead of copying — see
//! [`GuestMemory::map_frames`]). Coherency rules: file writes (`write`,
//! `pwrite`, `splice`, …) are written through into cached pages; file reads
//! overlay cached pages (a mapping's stores are visible to `read` before any
//! flush); `truncate` zeroes cached bytes past the new end. The existing
//! write-back of writable shared mappings (`munmap`/`msync`/exit) still
//! persists the contents into the backend. When no address space maps an
//! entry's frames any more, [`Kernel::pc_gc`] drops it.
//!
//! The cache sits behind its own leaf lock ([`Kernel::page_cache`]): it is
//! taken after `vfs` (the file paths call it with `vfs` held) and nothing is
//! acquired while holding it.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::{Kernel, err};
use crate::abi::errno::Errno;
use crate::fs::MountTable;
use crate::vcpu::GuestMemory;
use crate::vcpu::mem::{PAGE_SIZE, Prot};
use crate::vcpu::phys::PhysMem;

/// A file's identity: the mount point serving it and its inode there.
type FileKey = (String, u64);

/// The shared-file page cache.
#[derive(Debug, Default)]
pub(super) struct PageCache {
    /// The frame pool (captured from the first mapping), for reading and
    /// writing cached pages without an address space.
    phys: Option<Arc<PhysMem>>,
    /// File → page index → frame.
    files: BTreeMap<FileKey, BTreeMap<u64, u64>>,
}

impl PageCache {
    /// Copy every cached page overlapping `[off, off + buf.len())` of `pages`
    /// over `buf` (the cached bytes are the newest).
    fn overlay(&self, pages: &BTreeMap<u64, u64>, off: u64, buf: &mut [u8]) {
        let Some(phys) = &self.phys else { return };
        let end = off + buf.len() as u64;
        for (&idx, &frame) in pages.range(off / PAGE_SIZE..=end.saturating_sub(1) / PAGE_SIZE) {
            let (pstart, pend) = (idx * PAGE_SIZE, (idx + 1) * PAGE_SIZE);
            let (lo, hi) = (pstart.max(off), pend.min(end));
            if lo >= hi {
                continue;
            }
            phys.read(
                frame + (lo - pstart),
                &mut buf[(lo - off) as usize..(hi - off) as usize],
            );
        }
    }

    /// Write `data` at `off` into every cached page it overlaps.
    fn write_through(&self, pages: &BTreeMap<u64, u64>, off: u64, data: &[u8]) {
        let Some(phys) = &self.phys else { return };
        let end = off + data.len() as u64;
        for (&idx, &frame) in pages.range(off / PAGE_SIZE..=end.saturating_sub(1) / PAGE_SIZE) {
            let (pstart, pend) = (idx * PAGE_SIZE, (idx + 1) * PAGE_SIZE);
            let (lo, hi) = (pstart.max(off), pend.min(end));
            if lo >= hi {
                continue;
            }
            phys.write(
                frame + (lo - pstart),
                &data[(lo - off) as usize..(hi - off) as usize],
            );
        }
    }
}

impl Kernel {
    /// `vfs.read_at` with the page cache's view laid over the result: every
    /// file read the syscalls do goes through this, so a mapping's stores are
    /// visible to `read` without waiting for a flush.
    pub(super) fn vfs_read(
        &self,
        vfs: &mut MountTable,
        path: &str,
        off: u64,
        buf: &mut [u8],
    ) -> std::io::Result<usize> {
        let n = vfs.read_at(path, off, buf)?;
        self.pc_after_read(vfs, path, off, &mut buf[..n]);
        Ok(n)
    }

    /// `vfs.write_at`, written through into any cached pages of the file.
    pub(super) fn vfs_write(
        &self,
        vfs: &mut MountTable,
        path: &str,
        off: u64,
        data: &[u8],
    ) -> std::io::Result<usize> {
        // memfd seals (F_SEAL_WRITE/F_SEAL_GROW) refuse the write outright.
        if self.seal_blocks_write(vfs, path, off, data.len() as u64) {
            return Err(std::io::Error::from_raw_os_error(1)); // EPERM
        }
        let n = vfs.write_at(path, off, data)?;
        self.pc_after_write(vfs, path, off, &data[..n]);
        Ok(n)
    }

    /// `vfs.truncate`, keeping cached pages in step.
    pub(super) fn vfs_truncate(
        &self,
        vfs: &mut MountTable,
        path: &str,
        len: u64,
    ) -> std::io::Result<()> {
        if self.seal_blocks_truncate(vfs, path, len) {
            return Err(std::io::Error::from_raw_os_error(1)); // EPERM
        }
        vfs.truncate(path, len)?;
        self.pc_after_truncate(vfs, path, len);
        Ok(())
    }

    /// The cache key of the file at `path`, if it exists.
    pub(super) fn pc_key(vfs: &mut MountTable, path: &str) -> Option<FileKey> {
        let ino = vfs.stat(path)?.inode;
        Some((vfs.mount_point_of(path)?, ino))
    }

    /// Map `[base, base + len)` as a `MAP_SHARED` view of `path` from byte
    /// `offset` (page-aligned) with `prot`, through the page cache: pages
    /// already cached are shared, missing ones are read in.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn pc_map_shared(
        &self,
        vfs: &mut MountTable,
        mem: &mut GuestMemory,
        path: &str,
        offset: u64,
        base: u64,
        len: u64,
        prot: Prot,
    ) -> Result<(), i64> {
        let Some(key) = Self::pc_key(vfs, path) else {
            return Err(err(Errno::ENOENT));
        };
        let mut pc = self.page_cache.lock().unwrap();
        if pc.phys.is_none() {
            pc.phys = Some(mem.phys_arc());
        }
        let first = offset / PAGE_SIZE;
        let n = len / PAGE_SIZE;
        let mut frames = Vec::with_capacity(n as usize);
        let mut fresh = Vec::new();
        {
            let pages = pc.files.entry(key).or_default();
            for idx in first..first + n {
                if let Some(&f) = pages.get(&idx) {
                    frames.push(f);
                } else {
                    let Some(f) = mem.alloc_frames(1) else {
                        return Err(err(Errno::ENOMEM));
                    };
                    pages.insert(idx, f[0]);
                    frames.push(f[0]);
                    fresh.push((idx, f[0]));
                }
            }
        }
        // Fill the new pages from the file (past EOF they stay zero).
        if let Some(phys) = pc.phys.clone() {
            let mut buf = vec![0u8; PAGE_SIZE as usize];
            for (idx, frame) in fresh {
                buf.fill(0);
                let mut got = 0;
                while got < buf.len() {
                    match vfs.read_at(path, idx * PAGE_SIZE + got as u64, &mut buf[got..]) {
                        Ok(k) if k > 0 => got += k,
                        _ => break,
                    }
                }
                phys.write(frame, &buf);
            }
        }
        drop(pc);
        mem.map_frames(base, &frames, prot)
            .map_err(|_| err(Errno::ENOMEM))
    }

    /// After `read` of `buf.len()` bytes at `off` from `path`: lay any cached
    /// (possibly newer, mapped-and-stored) pages over the bytes read.
    pub(super) fn pc_after_read(&self, vfs: &mut MountTable, path: &str, off: u64, buf: &mut [u8]) {
        let pc = self.page_cache.lock().unwrap();
        if pc.files.is_empty() || buf.is_empty() {
            return;
        }
        drop(pc);
        let Some(key) = Self::pc_key(vfs, path) else {
            return;
        };
        let pc = self.page_cache.lock().unwrap();
        if let Some(pages) = pc.files.get(&key) {
            pc.overlay(pages, off, buf);
        }
    }

    /// After writing `data` at `off` of `path`: write it through into cached
    /// pages so every mapping sees it.
    pub(super) fn pc_after_write(&self, vfs: &mut MountTable, path: &str, off: u64, data: &[u8]) {
        let pc = self.page_cache.lock().unwrap();
        if pc.files.is_empty() || data.is_empty() {
            return;
        }
        drop(pc);
        let Some(key) = Self::pc_key(vfs, path) else {
            return;
        };
        let pc = self.page_cache.lock().unwrap();
        if let Some(pages) = pc.files.get(&key) {
            pc.write_through(pages, off, data);
        }
    }

    /// After `path` was truncated to `len`: cached bytes past the new end read
    /// as zero (as the file now does).
    pub(super) fn pc_after_truncate(&self, vfs: &mut MountTable, path: &str, len: u64) {
        let pc = self.page_cache.lock().unwrap();
        if pc.files.is_empty() {
            return;
        }
        drop(pc);
        let Some(key) = Self::pc_key(vfs, path) else {
            return;
        };
        let pc = self.page_cache.lock().unwrap();
        let (Some(pages), Some(phys)) = (pc.files.get(&key), &pc.phys) else {
            return;
        };
        for (&idx, &frame) in pages.range(len / PAGE_SIZE..) {
            let start = idx * PAGE_SIZE;
            let from = len.saturating_sub(start).min(PAGE_SIZE);
            phys.write(frame + from, &vec![0u8; (PAGE_SIZE - from) as usize]);
        }
    }

    /// Drop every cached page no address space maps any more (only the
    /// cache's own reference is left), and every file left with no pages.
    /// Called after mappings go away (`munmap`, `execve`, exit).
    pub(super) fn pc_gc(&self, mem: &GuestMemory) {
        let mut pc = self.page_cache.lock().unwrap();
        if pc.files.is_empty() {
            return;
        }
        let mut dead = Vec::new();
        for pages in pc.files.values_mut() {
            pages.retain(|_, &mut f| {
                let live = mem.frame_refcount(f) > 1;
                if !live {
                    dead.push(f);
                }
                live
            });
        }
        pc.files.retain(|_, pages| !pages.is_empty());
        drop(pc);
        mem.release_frames(&dead);
    }
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{BASE, call, put_str, setup};
    use crate::abi::arch::Sysno;

    const AT_FDCWD: u64 = (-100i64) as u64;
    const MAP_SHARED: u64 = 1;

    #[test]
    fn shared_file_mappings_share_pages_and_stay_coherent_with_io() {
        let (k, mut mem, mut v, mut cx) = setup();
        let (path, buf) = (BASE, BASE + 0x1000);
        put_str(&mut mem, path, "/f");
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_FDCWD, path, 0o102, 0o644, 0, 0],
        ) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Ftruncate,
                [fd, 8192, 0, 0, 0, 0]
            ),
            0
        );
        mem.write(buf, b"seed").unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Pwrite64,
                [fd, buf, 4, 0, 0, 0]
            ),
            4
        );
        // Two independent shared mappings of the same file.
        let a = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Mmap,
            [0, 8192, 3, MAP_SHARED, fd, 0],
        ) as u64;
        let b = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Mmap,
            [0, 8192, 3, MAP_SHARED, fd, 0],
        ) as u64;
        assert_ne!(a, b);
        assert_eq!(mem.read_vec(b, 4).unwrap(), b"seed");
        // A store through one is visible through the other…
        mem.write(a + 4096, b"page two").unwrap();
        assert_eq!(mem.read_vec(b + 4096, 8).unwrap(), b"page two");
        // …and to read() before any flush.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Pread64,
                [fd, buf, 8, 4096, 0, 0]
            ),
            8
        );
        assert_eq!(mem.read_vec(buf, 8).unwrap(), b"page two");
        // write() is visible through the mappings.
        mem.write(buf, b"WRITE").unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Pwrite64,
                [fd, buf, 5, 0, 0, 0]
            ),
            5
        );
        assert_eq!(mem.read_vec(a, 5).unwrap(), b"WRITE");
        // A fork child aliases the shared pages.
        let mut child = mem.fork();
        child.write(b, b"kid").unwrap();
        assert_eq!(mem.read_vec(a, 3).unwrap(), b"kid");
        child.release(); // the child exits: its mappings go away
        // Unmapping both releases the cache entry.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Munmap,
                [a, 8192, 0, 0, 0, 0]
            ),
            0
        );
        assert!(
            !k.page_cache.lock().unwrap().files.is_empty(),
            "still mapped at b"
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Munmap,
                [b, 8192, 0, 0, 0, 0]
            ),
            0
        );
        assert!(k.page_cache.lock().unwrap().files.is_empty(), "collected");
        // The stores reached the file.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Pread64,
                [fd, buf, 3, 0, 0, 0]
            ),
            3
        );
        assert_eq!(mem.read_vec(buf, 3).unwrap(), b"kid");
    }
}
