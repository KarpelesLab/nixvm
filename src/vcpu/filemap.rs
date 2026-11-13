//! The pages of a file, as the address spaces that map it see them.
//!
//! A [`FileObj`] is one file's pages in the shared frame pool: what every
//! `mmap` of the file maps, a fork child inherits, and `read`/`write` stay
//! coherent with. For a shared-memory file (a memfd, `/dev/shm`) the pages
//! *are* the file — it has no other contents — and a page nobody has written
//! is a hole, read as zero and given a frame on first touch, so a sparse
//! terabyte costs nothing until used. For any other file the kernel's page
//! cache fills the pages from the file before they're mapped.
//!
//! Mappings reach the pages lazily: a page is put in an address space's page
//! tables on its first touch ([`GuestMemory::file_fault`]), a touch past the
//! end of the file is `SIGBUS`, and a file shrinking under its mappings
//! ([`FileObj::set_size`]) bumps [`SHRINK_EPOCH`] so every address space drops
//! the pages it lost before it next runs ([`GuestMemory::revalidate_files`]).
//!
//! Locking: an object's own lock is taken before the frame allocator's, never
//! after — a caller holding the allocator must not touch an object.
//!
//! [`GuestMemory::file_fault`]: super::GuestMemory::file_fault
//! [`GuestMemory::revalidate_files`]: super::GuestMemory::revalidate_files

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use super::mem::PAGE_SIZE;
use super::phys::{FrameAllocator, PhysMem};

/// Bumped whenever a mapped file loses pages (a shrinking `ftruncate`, a hole
/// punched): every address space compares it with what it saw last.
pub static SHRINK_EPOCH: AtomicU64 = AtomicU64::new(0);

/// Why a page of a file can't be had.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageError {
    /// It's past the end of the file: `SIGBUS` for a mapping, as on Linux.
    Beyond,
    /// The frame pool (or the memory cgroup) has no frame for it.
    NoMemory,
}

/// One file's pages. See the module docs.
pub struct FileObj {
    phys: Arc<PhysMem>,
    fa: Arc<Mutex<FrameAllocator>>,
    inner: Mutex<Inner>,
    /// Shared mappings that could store into the file (`VM_SHARED` with
    /// `VM_MAYWRITE`): what `F_SEAL_WRITE` waits for to be none.
    writable_maps: AtomicUsize,
    /// Set once a mapping could have stored into the pages: they may differ
    /// from the file's, and are written back before being dropped.
    dirty: AtomicBool,
    /// The [`SHRINK_EPOCH`] of the last time this file lost pages.
    shrunk_at: AtomicU64,
    /// A shared-memory file: these pages are its only contents.
    shmem: bool,
    /// Where the kernel last found the file (another file's pages come from
    /// it), for filling pages a mapping grows over.
    origin: Mutex<String>,
}

#[derive(Default)]
struct Inner {
    /// Page index → frame (the object's own reference).
    pages: BTreeMap<u64, u64>,
    /// The file's size in bytes.
    size: u64,
}

impl std::fmt::Debug for FileObj {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.inner.lock().unwrap();
        f.debug_struct("FileObj")
            .field("size", &inner.size)
            .field("pages", &inner.pages.len())
            .field("shmem", &self.shmem)
            .finish_non_exhaustive()
    }
}

const PS: usize = PAGE_SIZE as usize;

impl FileObj {
    /// A file of `size` bytes with no pages yet, in the pool of `phys`/`fa`.
    #[must_use]
    pub fn new(
        phys: Arc<PhysMem>,
        fa: Arc<Mutex<FrameAllocator>>,
        size: u64,
        shmem: bool,
    ) -> Arc<Self> {
        Arc::new(Self {
            phys,
            fa,
            inner: Mutex::new(Inner {
                pages: BTreeMap::new(),
                size,
            }),
            writable_maps: AtomicUsize::new(0),
            dirty: AtomicBool::new(false),
            shrunk_at: AtomicU64::new(0),
            shmem,
            origin: Mutex::new(String::new()),
        })
    }

    /// Where the file was last found.
    #[must_use]
    pub fn origin(&self) -> String {
        self.origin.lock().unwrap().clone()
    }

    pub fn set_origin(&self, path: &str) {
        let mut o = self.origin.lock().unwrap();
        if *o != path {
            path.clone_into(&mut o);
        }
    }

    /// Whether the pages are the file (a memfd, `/dev/shm`).
    #[must_use]
    pub fn is_shmem(&self) -> bool {
        self.shmem
    }

    #[must_use]
    pub fn size(&self) -> u64 {
        self.inner.lock().unwrap().size
    }

    /// How many pages hold a frame (`st_blocks`, in pages).
    #[must_use]
    pub fn resident(&self) -> u64 {
        self.inner.lock().unwrap().pages.len() as u64
    }

    /// Whether the page is in the file (pages up to its size rounded up).
    fn within(inner: &Inner, idx: u64) -> bool {
        idx < inner.size.div_ceil(PAGE_SIZE)
    }

    /// A frame for page `idx`, holding a new reference for the caller (a
    /// mapping): the page's own, or a fresh zeroed one for a hole, charged to
    /// memory cgroup `memcg`.
    ///
    /// # Errors
    /// [`PageError::Beyond`] past the end of the file, [`PageError::NoMemory`]
    /// with no frame to be had.
    pub fn get(&self, idx: u64, memcg: u16) -> Result<u64, PageError> {
        let mut inner = self.inner.lock().unwrap();
        if !Self::within(&inner, idx) {
            return Err(PageError::Beyond);
        }
        let mut fa = self.fa.lock().unwrap();
        if let Some(&f) = inner.pages.get(&idx) {
            fa.incref(f);
            return Ok(f);
        }
        fa.charge_to = memcg;
        let f = fa.alloc(&self.phys).ok_or(PageError::NoMemory)?;
        fa.incref(f);
        drop(fa);
        inner.pages.insert(idx, f);
        Ok(f)
    }

    /// Page `idx`'s frame, given one now if it's a hole, taking no reference:
    /// what names the page while it's the file's (a process-shared futex).
    ///
    /// # Errors
    /// As [`Self::get`].
    pub fn frame(&self, idx: u64, memcg: u16) -> Result<u64, PageError> {
        let f = self.get(idx, memcg)?;
        self.fa.lock().unwrap().decref(f);
        Ok(f)
    }

    /// Page `idx`'s frame if it has one, taking no reference: `None` for a
    /// hole (it reads as zero).
    ///
    /// # Errors
    /// [`PageError::Beyond`] past the end of the file.
    pub fn peek(&self, idx: u64) -> Result<Option<u64>, PageError> {
        let inner = self.inner.lock().unwrap();
        if !Self::within(&inner, idx) {
            return Err(PageError::Beyond);
        }
        Ok(inner.pages.get(&idx).copied())
    }

    /// Whether page `idx` has a frame.
    #[must_use]
    pub fn has_page(&self, idx: u64) -> bool {
        self.inner.lock().unwrap().pages.contains_key(&idx)
    }

    /// The pages in `[first, end)` that have frames, in order.
    #[must_use]
    pub fn pages_in(&self, first: u64, end: u64) -> Vec<(u64, u64)> {
        let inner = self.inner.lock().unwrap();
        inner
            .pages
            .range(first..end)
            .map(|(&i, &f)| (i, f))
            .collect()
    }

    /// Read from byte `off` into `buf`, up to the end of the file (holes read
    /// as zero); how many bytes that was.
    pub fn read(&self, off: u64, buf: &mut [u8]) -> usize {
        let inner = self.inner.lock().unwrap();
        let n = inner.size.saturating_sub(off).min(buf.len() as u64) as usize;
        self.copy_out(&inner, off, &mut buf[..n]);
        n
    }

    /// Lay the pages that have frames over `buf`, which holds the file's
    /// bytes from `off` as another layer had them.
    pub fn overlay(&self, off: u64, buf: &mut [u8]) {
        let inner = self.inner.lock().unwrap();
        let end = off + buf.len() as u64;
        for (&idx, &frame) in inner.pages.range(off / PAGE_SIZE..end.div_ceil(PAGE_SIZE)) {
            let (lo, hi) = (off.max(idx * PAGE_SIZE), end.min((idx + 1) * PAGE_SIZE));
            if lo < hi {
                self.phys.read(
                    frame + lo % PAGE_SIZE,
                    &mut buf[(lo - off) as usize..(hi - off) as usize],
                );
            }
        }
    }

    fn copy_out(&self, inner: &Inner, off: u64, buf: &mut [u8]) {
        let mut done = 0usize;
        while done < buf.len() {
            let pos = off + done as u64;
            let in_page = (pos % PAGE_SIZE) as usize;
            let n = (buf.len() - done).min(PS - in_page);
            match inner.pages.get(&(pos / PAGE_SIZE)) {
                Some(&f) => self.phys.read(f + in_page as u64, &mut buf[done..done + n]),
                None => buf[done..done + n].fill(0),
            }
            done += n;
        }
    }

    /// Write `data` at byte `off`, giving holes frames as needed and growing
    /// the file if it ends past it (a shared-memory file's `write`).
    ///
    /// # Errors
    /// [`PageError::NoMemory`] with no frame for a page; what was written
    /// before it stays.
    pub fn write(&self, off: u64, data: &[u8], memcg: u16) -> Result<(), PageError> {
        let mut inner = self.inner.lock().unwrap();
        let mut done = 0usize;
        while done < data.len() {
            let pos = off + done as u64;
            let idx = pos / PAGE_SIZE;
            let in_page = (pos % PAGE_SIZE) as usize;
            let n = (data.len() - done).min(PS - in_page);
            let frame = if let Some(&f) = inner.pages.get(&idx) {
                f
            } else {
                let mut fa = self.fa.lock().unwrap();
                fa.charge_to = memcg;
                let f = fa.alloc(&self.phys).ok_or(PageError::NoMemory)?;
                drop(fa);
                inner.pages.insert(idx, f);
                f
            };
            self.phys
                .write(frame + in_page as u64, &data[done..done + n]);
            done += n;
            inner.size = inner.size.max(pos + n as u64);
        }
        Ok(())
    }

    /// Write `data` at byte `off` into the pages that have frames only (a
    /// write to the file another layer holds, kept coherent here).
    pub fn write_through(&self, off: u64, data: &[u8]) {
        let inner = self.inner.lock().unwrap();
        let end = off + data.len() as u64;
        for (&idx, &frame) in inner.pages.range(off / PAGE_SIZE..end.div_ceil(PAGE_SIZE)) {
            let (lo, hi) = (off.max(idx * PAGE_SIZE), end.min((idx + 1) * PAGE_SIZE));
            if lo < hi {
                self.phys.write(
                    frame + lo % PAGE_SIZE,
                    &data[(lo - off) as usize..(hi - off) as usize],
                );
            }
        }
    }

    /// Give page `idx` a frame holding `bytes` (the rest zero) unless it has
    /// one already: filling the page cache from the file.
    ///
    /// # Errors
    /// [`PageError::NoMemory`] with no frame for it.
    pub fn fill(&self, idx: u64, bytes: &[u8], memcg: u16) -> Result<(), PageError> {
        let mut inner = self.inner.lock().unwrap();
        if inner.pages.contains_key(&idx) {
            return Ok(());
        }
        let mut fa = self.fa.lock().unwrap();
        fa.charge_to = memcg;
        let f = fa.alloc(&self.phys).ok_or(PageError::NoMemory)?;
        drop(fa);
        self.phys.write(f, &bytes[..bytes.len().min(PS)]);
        inner.pages.insert(idx, f);
        Ok(())
    }

    /// The file is now `len` bytes: pages wholly past the end go (their
    /// frames live on in mappings until those drop them), the part of the
    /// last page past it reads zero, and a file that grew has holes to its
    /// new end. Mappings are told about lost pages through [`SHRINK_EPOCH`].
    pub fn set_size(&self, len: u64) {
        let mut inner = self.inner.lock().unwrap();
        let shrinking = len < inner.size;
        inner.size = len;
        if !shrinking {
            return;
        }
        let keep = len.div_ceil(PAGE_SIZE);
        let gone = inner.pages.split_off(&keep);
        if !len.is_multiple_of(PAGE_SIZE)
            && let Some(&f) = inner.pages.get(&(len / PAGE_SIZE))
        {
            let from = len % PAGE_SIZE;
            self.phys
                .write(f + from, &vec![0u8; (PAGE_SIZE - from) as usize]);
        }
        drop(inner);
        self.lost_pages();
        if !gone.is_empty() {
            let mut fa = self.fa.lock().unwrap();
            for f in gone.into_values() {
                fa.decref(f);
            }
        }
    }

    /// Zero `[off, off + len)` (within the file), dropping the frames of the
    /// pages wholly inside it — a punched hole (`FALLOC_FL_PUNCH_HOLE`,
    /// `MADV_REMOVE`).
    pub fn punch(&self, off: u64, len: u64) {
        let mut inner = self.inner.lock().unwrap();
        let end = off.saturating_add(len).min(inner.size);
        if end <= off {
            return;
        }
        let (first_whole, end_whole) = (off.div_ceil(PAGE_SIZE), end / PAGE_SIZE);
        let mut gone = Vec::new();
        if first_whole < end_whole {
            let tail = inner.pages.split_off(&end_whole);
            gone = inner.pages.split_off(&first_whole).into_values().collect();
            inner.pages.extend(tail);
        }
        for (&idx, &frame) in inner.pages.range(off / PAGE_SIZE..end.div_ceil(PAGE_SIZE)) {
            let (lo, hi) = (off.max(idx * PAGE_SIZE), end.min((idx + 1) * PAGE_SIZE));
            if lo < hi {
                self.phys
                    .write(frame + lo % PAGE_SIZE, &vec![0u8; (hi - lo) as usize]);
            }
        }
        drop(inner);
        if !gone.is_empty() {
            self.lost_pages();
            let mut fa = self.fa.lock().unwrap();
            for f in gone {
                fa.decref(f);
            }
        }
    }

    fn lost_pages(&self) {
        let epoch = SHRINK_EPOCH.fetch_add(1, Ordering::AcqRel) + 1;
        self.shrunk_at.store(epoch, Ordering::Release);
    }

    /// The [`SHRINK_EPOCH`] of the last time this file lost pages.
    #[must_use]
    pub fn shrunk_at(&self) -> u64 {
        self.shrunk_at.load(Ordering::Acquire)
    }

    /// How many shared mappings could store into the file.
    #[must_use]
    pub fn writable_maps(&self) -> usize {
        self.writable_maps.load(Ordering::Acquire)
    }

    /// Whether a mapping could have stored into the pages since they were
    /// filled (they may differ from the file's), clearing the mark when `clear`.
    pub fn take_dirty(&self, clear: bool) -> bool {
        if clear {
            self.dirty.swap(false, Ordering::AcqRel)
        } else {
            self.dirty.load(Ordering::Acquire)
        }
    }

    /// Note that a mapping may store into the pages.
    pub fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }
}

impl Drop for FileObj {
    fn drop(&mut self) {
        let pages = std::mem::take(&mut self.inner.get_mut().unwrap().pages);
        if pages.is_empty() {
            return;
        }
        let mut fa = self.fa.lock().unwrap();
        for f in pages.into_values() {
            fa.decref(f);
        }
    }
}

/// A shared mapping that could store into its file, counted in
/// [`FileObj::writable_maps`] for as long as it lives (a copy — a fork
/// child's, a split mapping's — counts again).
#[derive(Debug)]
pub struct WriteRef(Arc<FileObj>);

impl WriteRef {
    #[must_use]
    pub fn new(obj: &Arc<FileObj>) -> Self {
        obj.writable_maps.fetch_add(1, Ordering::AcqRel);
        obj.mark_dirty();
        Self(Arc::clone(obj))
    }
}

impl Clone for WriteRef {
    fn clone(&self) -> Self {
        Self::new(&self.0)
    }
}

impl Drop for WriteRef {
    fn drop(&mut self) {
        self.0.writable_maps.fetch_sub(1, Ordering::AcqRel);
    }
}

/// What a file mapping maps: which file, from which page, how, and what
/// `/proc/<pid>/maps` calls it.
#[derive(Debug, Clone)]
pub struct FileMap {
    /// The file's pages; `None` for a named mapping of frames installed
    /// otherwise (a SysV shared-memory attachment).
    pub obj: Option<Arc<FileObj>>,
    /// The file page the mapping starts at.
    pub pgoff: u64,
    /// `MAP_SHARED`: stores reach the file and every other mapper.
    pub shared: bool,
    /// Could be made writable (`VM_MAYWRITE`): `mprotect` to add `PROT_WRITE`
    /// is `EACCES` otherwise.
    pub maywrite: bool,
    /// What `/proc/<pid>/maps` names it, and its device and inode there.
    pub name: Arc<str>,
    pub dev: u64,
    pub ino: u64,
}

/// A file mapping in an address space: [`FileMap`] over `pages` pages.
#[derive(Debug, Clone)]
pub(crate) struct FileVma {
    pub(crate) pages: usize,
    pub(crate) map: FileMap,
    /// Held while the mapping counts as writable.
    pub(crate) wref: Option<WriteRef>,
}

impl FileVma {
    pub(crate) fn new(pages: usize, map: FileMap) -> Self {
        let wref = match &map.obj {
            Some(obj) if map.shared && map.maywrite => Some(WriteRef::new(obj)),
            _ => None,
        };
        Self { pages, map, wref }
    }

    /// The part of this mapping from `skip` pages in, `pages` long.
    pub(crate) fn slice(&self, skip: usize, pages: usize) -> Self {
        let mut map = self.map.clone();
        map.pgoff += skip as u64;
        Self {
            pages,
            map,
            wref: self.wref.clone(),
        }
    }
}

/// One line of `/proc/<pid>/maps`: `[start, end)` with one protection, one
/// kind of sharing and one backing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Region {
    pub start: u64,
    pub end: u64,
    pub prot: super::Prot,
    pub shared: bool,
    /// File offset in bytes (0 for anonymous memory).
    pub offset: u64,
    pub dev: u64,
    pub ino: u64,
    /// The file's name, or empty for anonymous memory.
    pub name: Arc<str>,
}
