//! A page cache for file mappings, so a shared file mapping is *shared*:
//! every process (and every `mmap` within one) that maps the same page of the
//! same file maps the same physical frame, a `fork` child aliases it, and
//! `read`/`write` on the file stay coherent with the mappings. A private
//! mapping is copy-on-write over the same frames, so it reads the file as it
//! is until it writes a page of its own.
//!
//! A file's pages are a [`FileObj`] (see [`crate::vcpu::filemap`]), which the
//! mappings reference; the cache holds one per mapped regular file, keyed by
//! the file's identity. Shared-memory files — memfds and `/dev/shm` — need no
//! cache: their contents *are* such pages ([`ShmStore`]), so they are sparse,
//! never written back, and mapped straight from there.
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
//! Frames are allocated from the shared pool, filled from the file before a
//! mapping maps them, and owned by the [`FileObj`] (one reference) while also
//! mapped (one reference per page table entry). Coherency rules: file writes
//! (`write`, `pwrite`, `splice`, …) are written through into cached pages;
//! file reads overlay cached pages (a mapping's stores are visible to `read`
//! before any flush); `truncate` zeroes cached bytes past the new end and
//! takes the pages past it from every mapping (touching them is `SIGBUS`).
//! The write-back of writable shared mappings (`munmap`/`msync`/exit)
//! persists the contents into the backend. When no address space maps a
//! file any more, [`Kernel::pc_gc`] drops its pages.
//!
//! The cache sits behind its own leaf lock ([`Kernel::page_cache`]): it is
//! taken after `vfs` (the file paths call it with `vfs` held) and nothing is
//! acquired while holding it.

use std::collections::BTreeMap;
use std::sync::Arc;

use std::sync::Mutex;

use super::{Kernel, err};
use crate::abi::errno::Errno;
use crate::fs::{MountTable, PageStore};
use crate::vcpu::GuestMemory;
use crate::vcpu::filemap::FileObj;
use crate::vcpu::mem::PAGE_SIZE;
use crate::vcpu::phys::{FrameAllocator, PhysMem};

/// The task whose syscall this host thread is servicing, as file I/O needs
/// it: set by [`IoTask::enter`] for the length of a step.
pub(super) struct IoTask {
    pid: i32,
    memcg: u16,
    fa: Arc<std::sync::Mutex<crate::vcpu::phys::FrameAllocator>>,
    /// Until when its I/O has it wait, by its cgroups' `io.max`.
    delay_until: Option<u128>,
}

thread_local! {
    static IO_TASK: std::cell::RefCell<Option<IoTask>> = const { std::cell::RefCell::new(None) };
}

impl IoTask {
    /// File I/O on this thread is task `pid`'s (memory cgroup `memcg`, in
    /// `mem`'s frame pool) until [`IoTask::leave`].
    pub(super) fn enter(pid: i32, memcg: u16, mem: &GuestMemory) {
        let fa = mem.allocator();
        IO_TASK.with(|t| {
            *t.borrow_mut() = Some(Self {
                pid,
                memcg,
                fa,
                delay_until: None,
            });
        });
    }

    /// The step is over: until when its I/O has the task wait, if it does.
    pub(super) fn leave() -> Option<u128> {
        IO_TASK.with(|t| t.borrow_mut().take().and_then(|t| t.delay_until))
    }

    /// The memory cgroup the task's I/O allocates in (0 outside a step).
    pub(super) fn memcg() -> u16 {
        IO_TASK.with(|t| t.borrow().as_ref().map_or(0, |t| t.memcg))
    }
}

/// A file's identity: the mount point serving it and its inode there.
type FileKey = (String, u64);

/// The shared-file page cache.
#[derive(Debug, Default)]
pub(super) struct PageCache {
    /// File → its pages.
    files: BTreeMap<FileKey, Arc<FileObj>>,
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
        self.disk_io(vfs, path, off, n as u64, false);
        Ok(n)
    }

    /// After `n` bytes at `off` of `path` were read (or with `write`
    /// written): if they're on one of the VM's disks, run them through the
    /// page cache's accounting (charged to the task's memory cgroup), count
    /// what had to go to the disk — what a read didn't find in memory, all of
    /// a write — in `/proc/diskstats` and the task's `io.stat`, and hold it to
    /// its cgroups' `io.max` (see [`IoTask`]).
    fn disk_io(&self, vfs: &mut MountTable, path: &str, off: u64, n: u64, write: bool) {
        if n == 0 {
            return;
        }
        let Some(dev) = vfs.data_dev(path) else {
            return;
        };
        let Some(inode) = vfs.stat(path).map(|a| a.inode) else {
            return;
        };
        IO_TASK.with(|t| {
            let mut t = t.borrow_mut();
            let Some(task) = t.as_mut() else { return };
            let first = off / PAGE_SIZE;
            let pages = (off + n).div_ceil(PAGE_SIZE) - first;
            let missed =
                task.fa
                    .lock()
                    .unwrap()
                    .cache_access((dev, inode), first, pages, task.memcg);
            let bytes = if write { n } else { missed * PAGE_SIZE };
            if bytes == 0 {
                return;
            }
            vfs.disks().lock().unwrap().account(dev, write, bytes);
            let mut tree = self.cgroups.lock().unwrap();
            let node = tree.node_of(task.pid);
            if let Some(until) = tree.io_account(node, dev, write, bytes) {
                task.delay_until = task.delay_until.max(Some(until));
            }
        });
    }

    /// Drop pages `[first, end)` of the file at `path` from the page cache,
    /// if it's on a disk.
    pub(super) fn cache_drop(vfs: &mut MountTable, path: &str, first: u64, end: u64) {
        let (Some(dev), Some(a)) = (vfs.data_dev(path), vfs.stat(path)) else {
            return;
        };
        IO_TASK.with(|t| {
            if let Some(task) = t.borrow().as_ref() {
                task.fa
                    .lock()
                    .unwrap()
                    .cache_drop((dev, a.inode), first, end);
            }
        });
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
        self.disk_io(vfs, path, off, n as u64, true);
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
        Self::cache_drop(vfs, path, len.div_ceil(PAGE_SIZE), u64::MAX);
        Ok(())
    }

    /// The cache key of the file at `path`, if it exists.
    pub(super) fn pc_key(vfs: &mut MountTable, path: &str) -> Option<FileKey> {
        let ino = vfs.stat(path)?.inode;
        Some((vfs.mount_point_of(path)?, ino))
    }

    /// The pages of the file at `path`, as mappings map them: a
    /// shared-memory file's own (its contents), or the page cache's for any
    /// other regular file (made now if the file has none).
    pub(super) fn file_obj(
        &self,
        vfs: &mut MountTable,
        mem: &GuestMemory,
        path: &str,
    ) -> Result<Arc<FileObj>, i64> {
        if let Some(id) = vfs.page_id(path) {
            self.shm.attach_pool(mem);
            return self.shm.obj(id).ok_or_else(|| err(Errno::ENODEV));
        }
        let size = vfs.stat(path).ok_or_else(|| err(Errno::ENOENT))?.size;
        let key = Self::pc_key(vfs, path).ok_or_else(|| err(Errno::ENOENT))?;
        let mut pc = self.page_cache.lock().unwrap();
        let obj = pc
            .files
            .entry(key)
            .or_insert_with(|| FileObj::new(mem.phys_arc(), mem.allocator(), size, false));
        obj.set_origin(path);
        Ok(Arc::clone(obj))
    }

    /// Fill the page cache's pages `[first, first + n)` of the (non
    /// shared-memory) file at `path` from the file, where they're inside it
    /// and not cached yet: what a mapping of them is about to map.
    pub(super) fn pc_populate(
        vfs: &mut MountTable,
        obj: &FileObj,
        path: &str,
        first: u64,
        n: u64,
        memcg: u16,
    ) -> Result<(), i64> {
        if obj.is_shmem() {
            return Ok(());
        }
        let end = (first + n).min(obj.size().div_ceil(PAGE_SIZE));
        let mut buf = vec![0u8; PAGE_SIZE as usize];
        for idx in first..end {
            if obj.has_page(idx) {
                continue;
            }
            buf.fill(0);
            let mut got = 0;
            while got < buf.len() {
                match vfs.read_at(path, idx * PAGE_SIZE + got as u64, &mut buf[got..]) {
                    Ok(k) if k > 0 => got += k,
                    _ => break,
                }
            }
            obj.fill(idx, &buf, memcg).map_err(|_| err(Errno::ENOMEM))?;
        }
        Ok(())
    }

    /// The page cache's pages of the (non shared-memory) file at `path`, if
    /// it has any.
    fn cached(&self, vfs: &mut MountTable, path: &str) -> Option<Arc<FileObj>> {
        if self.page_cache.lock().unwrap().files.is_empty() {
            return None;
        }
        let key = Self::pc_key(vfs, path)?;
        self.page_cache.lock().unwrap().files.get(&key).cloned()
    }

    /// After `read` of `buf.len()` bytes at `off` from `path`: lay any cached
    /// (possibly newer, mapped-and-stored) pages over the bytes read.
    pub(super) fn pc_after_read(&self, vfs: &mut MountTable, path: &str, off: u64, buf: &mut [u8]) {
        if buf.is_empty() {
            return;
        }
        if let Some(obj) = self.cached(vfs, path) {
            obj.overlay(off, buf);
        }
    }

    /// After writing `data` at `off` of `path`: write it through into cached
    /// pages so every mapping sees it, growing the cached file with it and
    /// caching the pages it wrote that weren't (a mapping past the old end of
    /// the file maps them next).
    pub(super) fn pc_after_write(&self, vfs: &mut MountTable, path: &str, off: u64, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        let Some(obj) = self.cached(vfs, path) else {
            return;
        };
        obj.write_through(off, data);
        let end = off + data.len() as u64;
        if end > obj.size() {
            obj.set_size(end);
        }
        let first = off / PAGE_SIZE;
        let pages = end.div_ceil(PAGE_SIZE) - first;
        let memcg = IoTask::memcg();
        let _ = Self::pc_populate(vfs, &obj, path, first, pages, memcg);
    }

    /// After `path` was truncated to `len`: cached bytes past the new end read
    /// as zero, and pages past it are gone from every mapping (`SIGBUS`).
    pub(super) fn pc_after_truncate(&self, vfs: &mut MountTable, path: &str, len: u64) {
        if let Some(obj) = self.cached(vfs, path) {
            obj.set_size(len);
        }
    }

    /// Drop the cached files no address space maps any more.
    /// Called after mappings go away (`munmap`, `execve`, exit).
    pub(super) fn pc_gc(&self, _mem: &GuestMemory) {
        let mut pc = self.page_cache.lock().unwrap();
        if pc.files.is_empty() {
            return;
        }
        let dead: Vec<Arc<FileObj>> = {
            let mut dead = Vec::new();
            pc.files.retain(|_, obj| {
                let live = Arc::strong_count(obj) > 1;
                if !live {
                    dead.push(Arc::clone(obj));
                }
                live
            });
            dead
        };
        drop(pc);
        // Their frames go back to the pool as the last references drop.
        drop(dead);
    }
}

/// Where shared-memory files (memfds, `/dev/shm`) keep their contents: one
/// [`FileObj`] per file, in guest memory. See [`crate::fs::PageStore`].
#[derive(Default)]
pub(super) struct ShmStore {
    /// The frame pool (the first address space's, which every other shares).
    pool: std::sync::OnceLock<(Arc<PhysMem>, Arc<Mutex<FrameAllocator>>)>,
    objs: Mutex<BTreeMap<u64, Arc<FileObj>>>,
    next: std::sync::atomic::AtomicU64,
}

impl std::fmt::Debug for ShmStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ShmStore")
            .field("files", &self.objs.lock().unwrap().len())
            .finish_non_exhaustive()
    }
}

impl ShmStore {
    /// Learn the frame pool from an address space in it, once.
    pub(super) fn attach_pool(&self, mem: &GuestMemory) {
        if self.pool.get().is_none() {
            let _ = self.pool.set((mem.phys_arc(), mem.allocator()));
        }
    }

    /// The pages of shared-memory file `id`.
    pub(super) fn obj(&self, id: u64) -> Option<Arc<FileObj>> {
        self.objs.lock().unwrap().get(&id).cloned()
    }

    fn with<T>(&self, id: u64, f: impl FnOnce(&FileObj) -> T) -> Option<T> {
        let obj = self.obj(id)?;
        Some(f(&obj))
    }
}

impl PageStore for ShmStore {
    fn create(&self) -> Option<u64> {
        let (phys, fa) = self.pool.get()?;
        let id = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        let obj = FileObj::new(Arc::clone(phys), Arc::clone(fa), 0, true);
        self.objs.lock().unwrap().insert(id, obj);
        Some(id)
    }

    fn release(&self, id: u64) {
        let obj = self.objs.lock().unwrap().remove(&id);
        drop(obj);
    }

    fn len(&self, id: u64) -> u64 {
        self.with(id, FileObj::size).unwrap_or(0)
    }

    fn read(&self, id: u64, off: u64, buf: &mut [u8]) -> usize {
        self.with(id, |o| o.read(off, buf)).unwrap_or(0)
    }

    fn write(&self, id: u64, off: u64, data: &[u8]) -> std::io::Result<()> {
        let memcg = IoTask::memcg();
        match self.with(id, |o| o.write(off, data, memcg)) {
            Some(Ok(())) => Ok(()),
            Some(Err(_)) => Err(std::io::Error::from_raw_os_error(28)), // ENOSPC
            None => Err(std::io::Error::from_raw_os_error(2)),
        }
    }

    fn set_len(&self, id: u64, len: u64) -> std::io::Result<()> {
        self.with(id, |o| o.set_size(len))
            .ok_or_else(|| std::io::Error::from_raw_os_error(2))
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
        child.retire(); // the child exits: its mappings go away
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
