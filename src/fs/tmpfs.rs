//! In-memory read-write filesystem.
//!
//! Backs `/tmp` and serves as the writable upper layer of the copy-on-write
//! overlay (Phase 4). The namespace is stored flat in a `BTreeMap` keyed by the
//! mount-relative path (`""` is the backend root, then `"a"`, `"a/b"`, …); this
//! keeps `readdir` (children of a directory) and subtree `rename` simple.
//!
//! Regular files are *inodes*: a path entry names a file by inode number and
//! the data, mode, owner, timestamps and xattrs live in `TmpFs::files`, so
//! several names can share one file — real hard links (`link(2)`), with
//! `st_nlink` counting the names and the file freed when the last name goes.
//! (`git clone` of a local repository verifies that its hard-linked objects
//! report the source's inode; a copy-on-"link" fails that check.)

use std::collections::BTreeMap;
use std::io;
use std::sync::Arc;

use super::{Attrs, DirEntry, MountFs, NodeKind, PageStore, SetTime};
use crate::limits::Quota;

/// Unix mode type bits.
const S_IFDIR: u32 = 0o040_000;
const S_IFREG: u32 = 0o100_000;
const S_IFLNK: u32 = 0o120_000;
const S_IFIFO: u32 = 0o010_000;
const S_IFSOCK: u32 = 0o140_000;

/// Per-node metadata common to every node kind: identity, owner, the
/// access/modification timestamps `utimensat`/`chown` mutate, and the node's
/// extended attributes. Living with the node (or, for a regular file, its
/// inode), the attributes follow it through a `rename` and vanish with it —
/// the inode semantics a path-keyed side table would get wrong.
#[derive(Debug)]
struct Meta {
    inode: u64,
    uid: u32,
    gid: u32,
    atime: i64,
    mtime: i64,
    xattrs: BTreeMap<String, Vec<u8>>,
}

/// A regular file's inode: shared by every name hard-linked to it.
#[derive(Debug)]
struct FileData {
    meta: Meta,
    data: Contents,
    /// Where the contents are instead, for a shared-memory filesystem's file:
    /// its id in the page store, which holds them in guest memory.
    paged: Option<(u64, Arc<dyn PageStore>)>,
    mode: u32,
    /// How many path entries name this inode (`st_nlink`).
    links: u32,
    /// Charged for `data`'s pages and `meta`'s attributes until dropped.
    quota: Arc<Quota>,
}

impl FileData {
    /// What this inode holds charged against its quota.
    fn cost(&self) -> u64 {
        self.data.held() + xattr_cost(&self.meta)
    }
}

/// The unit a regular file's contents are held in, as Linux tmpfs holds them:
/// a page.
const PAGE: u64 = 4096;

/// One page of a file's contents.
type Page = Box<[u8; PAGE as usize]>;

/// A regular file's contents: its size and the pages written so far. A page
/// never written (a hole, as `ftruncate` past the end or a write beyond it
/// leaves) holds no memory and reads as zeros, so a sparse file costs only
/// what's in it, as on Linux. Bytes of a held page past `len` are always zero,
/// so growing the file again shows zeros there.
#[derive(Debug, Default)]
struct Contents {
    len: u64,
    pages: BTreeMap<u64, Page>,
}

/// A new page holding `fill` (a whole page), or zeros; `None` if there's no
/// memory for it.
fn new_page(fill: Option<&[u8]>) -> Option<Page> {
    let mut v = Vec::new();
    v.try_reserve_exact(PAGE as usize).ok()?;
    match fill {
        Some(bytes) => v.extend_from_slice(bytes),
        None => v.resize(PAGE as usize, 0),
    }
    v.into_boxed_slice().try_into().ok()
}

impl Contents {
    /// The memory the pages hold, which is what the file is charged.
    fn held(&self) -> u64 {
        self.pages.len() as u64 * PAGE
    }

    /// The pages covering `[off, end)` that aren't held yet.
    fn missing(&self, off: u64, end: u64) -> u64 {
        let (first, last) = (off / PAGE, end.div_ceil(PAGE));
        (last - first) - self.pages.range(first..last).count() as u64
    }

    /// Read from `off` into `buf`: how many bytes, up to the end of the file.
    fn read(&self, off: u64, buf: &mut [u8]) -> usize {
        if off >= self.len {
            return 0;
        }
        let n = (buf.len() as u64).min(self.len - off) as usize;
        let end = off + n as u64;
        // One walk over the held pages in range; the gaps between are holes.
        let mut done = 0;
        for (&idx, page) in self.pages.range(off / PAGE..end.div_ceil(PAGE)) {
            let start = (idx * PAGE).max(off);
            let hole = (start - off) as usize;
            buf[done..hole].fill(0);
            let at = (start % PAGE) as usize;
            let chunk = (PAGE as usize - at).min(n - hole);
            buf[hole..hole + chunk].copy_from_slice(&page[at..at + chunk]);
            done = hole + chunk;
        }
        buf[done..n].fill(0);
        n
    }

    /// Write `buf` at `off` (`off + buf.len()` already checked), charging the
    /// pages it adds to `quota`: all of it, or (`ENOSPC`) none of it.
    fn write(&mut self, off: u64, buf: &[u8], quota: &Quota) -> io::Result<()> {
        if buf.is_empty() {
            return Ok(());
        }
        let end = off + buf.len() as u64;
        let missing = self.missing(off, end);
        if !quota.try_charge(missing * PAGE) {
            return Err(enospc());
        }
        // Allocate every new page first, so a failure changes nothing.
        let mut fresh = Vec::new();
        if missing > 0 {
            if fresh.try_reserve_exact(missing as usize).is_err() {
                quota.release(missing * PAGE);
                return Err(enospc());
            }
            for idx in off / PAGE..end.div_ceil(PAGE) {
                if self.pages.contains_key(&idx) {
                    continue;
                }
                // A page the write covers whole is made from it directly.
                let start = idx * PAGE;
                let whole = (start >= off && start + PAGE <= end).then(|| {
                    let at = (start - off) as usize;
                    &buf[at..at + PAGE as usize]
                });
                let Some(page) = new_page(whole) else {
                    quota.release(missing * PAGE);
                    return Err(enospc());
                };
                fresh.push((idx, page, whole.is_some()));
            }
        }
        // The pages made from the write already hold their part of it.
        let mut filled = Vec::new();
        for (idx, page, whole) in fresh {
            if whole {
                filled.push(idx);
            }
            self.pages.insert(idx, page);
        }
        let mut filled = filled.into_iter().peekable();
        for (&idx, page) in self.pages.range_mut(off / PAGE..end.div_ceil(PAGE)) {
            if filled.next_if_eq(&idx).is_some() {
                continue;
            }
            let start = (idx * PAGE).max(off);
            let at = (start % PAGE) as usize;
            let from = (start - off) as usize;
            let chunk = (PAGE as usize - at).min(buf.len() - from);
            page[at..at + chunk].copy_from_slice(&buf[from..from + chunk]);
        }
        self.len = self.len.max(end);
        Ok(())
    }

    /// Give `[off, end)` pages now (`fallocate`), charged to `quota`: all of
    /// them, or (`ENOSPC`) none. The size stays; pages past it read as zeros
    /// when the file grows over them.
    fn allocate(&mut self, off: u64, end: u64, quota: &Quota) -> io::Result<()> {
        let missing = self.missing(off, end);
        let bytes = missing.checked_mul(PAGE).ok_or_else(enospc)?;
        if !quota.try_charge(bytes) {
            return Err(enospc());
        }
        let mut fresh = Vec::new();
        if fresh.try_reserve_exact(missing as usize).is_err() {
            quota.release(bytes);
            return Err(enospc());
        }
        for idx in off / PAGE..end.div_ceil(PAGE) {
            if self.pages.contains_key(&idx) {
                continue;
            }
            let Some(page) = new_page(None) else {
                quota.release(bytes);
                return Err(enospc());
            };
            fresh.push((idx, page));
        }
        self.pages.extend(fresh);
        Ok(())
    }

    /// Make `[off, end)` within the file a hole (`FALLOC_FL_PUNCH_HOLE`): the
    /// pages wholly inside it are freed (released from `quota`), and the
    /// parts of the pages at its edges zeroed.
    fn punch(&mut self, off: u64, end: u64, quota: &Quota) {
        let end = end.min(self.len);
        if end <= off {
            return;
        }
        let (first_whole, end_whole) = (off.div_ceil(PAGE), end / PAGE);
        if first_whole < end_whole {
            let mut tail = self.pages.split_off(&end_whole);
            let gone = self.pages.split_off(&first_whole);
            quota.release(gone.len() as u64 * PAGE);
            self.pages.append(&mut tail);
        }
        for (&idx, page) in self.pages.range_mut(off / PAGE..end.div_ceil(PAGE)) {
            let (lo, hi) = (off.max(idx * PAGE), end.min((idx + 1) * PAGE));
            if lo < hi {
                page[(lo % PAGE) as usize..][..(hi - lo) as usize].fill(0);
            }
        }
    }

    /// Make the file `len` bytes long. Growing adds a hole, holding nothing;
    /// shrinking frees the pages wholly past the end (released from `quota`)
    /// and zeroes the rest of the last one.
    fn set_len(&mut self, len: u64, quota: &Quota) {
        if len < self.len {
            let gone = self.pages.split_off(&len.div_ceil(PAGE));
            quota.release(gone.len() as u64 * PAGE);
            let at = (len % PAGE) as usize;
            if at != 0
                && let Some(page) = self.pages.get_mut(&(len / PAGE))
            {
                page[at..].fill(0);
            }
        }
        self.len = len;
    }

    /// `lseek`'s `SEEK_DATA` (`hole == false`) or `SEEK_HOLE` from `off`, which
    /// is below the size: where the next data or hole starts, page by page
    /// (the end of the file counts as a hole); `None` with no data after `off`.
    fn seek(&self, off: u64, hole: bool) -> Option<u64> {
        let idx = off / PAGE;
        let found = if hole {
            let mut next = idx;
            for &held in self.pages.range(idx..).map(|(i, _)| i) {
                if held != next {
                    break;
                }
                next += 1;
            }
            (next * PAGE).max(off).min(self.len)
        } else {
            let (&held, _) = self.pages.range(idx..).next()?;
            (held * PAGE).max(off)
        };
        (found < self.len || hole).then_some(found)
    }
}

impl Drop for FileData {
    fn drop(&mut self) {
        self.quota.release(self.cost());
        if let Some((id, store)) = &self.paged {
            store.release(*id);
        }
    }
}

/// The largest file offset, Linux's `s_maxbytes` for tmpfs.
const MAX_FILE_SIZE: u64 = i64::MAX as u64;

/// Roughly what a path entry costs on top of its name and contents.
const NODE_OVERHEAD: u64 = 256;

/// Host memory held by `meta`'s extended attributes.
fn xattr_cost(meta: &Meta) -> u64 {
    meta.xattrs
        .iter()
        .map(|(name, value)| (name.len() + value.len()) as u64 + NODE_OVERHEAD)
        .sum()
}

#[derive(Debug)]
enum Node {
    Dir {
        meta: Meta,
        /// The permission bits (and set-group-ID / sticky).
        mode: u32,
    },
    /// A regular file: its inode in `TmpFs::files`.
    File {
        ino: u64,
    },
    /// A FIFO, or (`socket`) the name a bound `AF_UNIX` socket made.
    Fifo {
        meta: Meta,
        mode: u32,
        socket: bool,
    },
    Symlink {
        meta: Meta,
        target: String,
    },
}

#[derive(Debug)]
pub struct TmpFs {
    /// Shared memory (`/dev/shm`, memfds): regular files keep their contents
    /// in [`Self::pages`], once the kernel has given it one.
    shmem: bool,
    pages: Option<Arc<dyn PageStore>>,
    nodes: BTreeMap<String, Node>,
    /// Regular-file inodes by inode number.
    files: BTreeMap<u64, FileData>,
    next_inode: u64,
    /// What file data, names and attributes are charged to.
    quota: Arc<Quota>,
}

impl Default for TmpFs {
    fn default() -> Self {
        Self::new()
    }
}

impl TmpFs {
    #[must_use]
    pub fn new() -> Self {
        let mut nodes = BTreeMap::new();
        nodes.insert(
            String::new(),
            Node::Dir {
                meta: Meta {
                    inode: 1,
                    uid: 0,
                    gid: 0,
                    atime: 0,
                    mtime: 0,
                    xattrs: BTreeMap::new(),
                },
                // A fresh tmpfs's root is world-writable and sticky, as Linux's.
                mode: 0o1777,
            },
        );
        Self {
            shmem: false,
            pages: None,
            nodes,
            files: BTreeMap::new(),
            next_inode: 2,
            quota: Quota::unlimited(),
        }
    }

    /// A shared-memory tmpfs: its files' contents live in the page store the
    /// kernel attaches ([`MountFs::attach_pages`]) — guest memory, sparse,
    /// and what mappings of them map.
    #[must_use]
    pub fn shmem() -> Self {
        Self {
            shmem: true,
            ..Self::new()
        }
    }

    fn alloc_inode(&mut self) -> u64 {
        let i = self.next_inode;
        self.next_inode += 1;
        i
    }

    /// A fresh `Meta` for a newly created node: owned by root, timestamps now.
    fn new_meta(&mut self) -> Meta {
        let now = now_ts();
        Meta {
            inode: self.alloc_inode(),
            uid: 0,
            gid: 0,
            atime: now,
            mtime: now,
            xattrs: BTreeMap::new(),
        }
    }

    /// Create a fresh regular-file inode with one link and return its number.
    fn new_file(&mut self, mode: u32) -> u64 {
        let meta = self.new_meta();
        let ino = meta.inode;
        let paged = self
            .pages
            .as_ref()
            .and_then(|store| Some((store.create()?, Arc::clone(store))));
        self.files.insert(
            ino,
            FileData {
                meta,
                data: Contents::default(),
                paged,
                mode,
                links: 1,
                quota: self.quota.clone(),
            },
        );
        ino
    }

    /// The metadata of the node at `rel` (a regular file's lives in its inode).
    fn meta_of(&self, rel: &str) -> Option<&Meta> {
        match self.nodes.get(rel)? {
            Node::Dir { meta, .. } | Node::Fifo { meta, .. } | Node::Symlink { meta, .. } => {
                Some(meta)
            }
            Node::File { ino } => self.files.get(ino).map(|f| &f.meta),
        }
    }

    fn meta_of_mut(&mut self, rel: &str) -> Option<&mut Meta> {
        match self.nodes.get_mut(rel)? {
            Node::Dir { meta, .. } | Node::Fifo { meta, .. } | Node::Symlink { meta, .. } => {
                Some(meta)
            }
            Node::File { ino } => self.files.get_mut(ino).map(|f| &mut f.meta),
        }
    }

    /// The regular file named `rel`, if it is one.
    fn file_mut(&mut self, rel: &str) -> Option<&mut FileData> {
        match self.nodes.get(rel)? {
            Node::File { ino } => self.files.get_mut(ino),
            _ => None,
        }
    }

    /// What the path entry `rel` holding `node` costs. A regular file's data and
    /// attributes belong to its inode (see [`FileData::cost`]).
    fn node_cost(rel: &str, node: &Node) -> u64 {
        let contents = match node {
            Node::File { .. } => 0,
            Node::Dir { meta, .. } | Node::Fifo { meta, .. } => xattr_cost(meta),
            Node::Symlink { meta, target } => target.len() as u64 + xattr_cost(meta),
        };
        NODE_OVERHEAD + rel.len() as u64 + contents
    }

    /// Add the path entry `rel`, if the quota has room for it.
    fn insert_node(&mut self, rel: &str, node: Node) -> io::Result<()> {
        if !self.quota.try_charge(Self::node_cost(rel, &node)) {
            return Err(enospc());
        }
        self.nodes.insert(rel.to_string(), node);
        Ok(())
    }

    /// Drop the path entry `rel`, releasing a regular file's inode when it was
    /// the last name.
    fn remove_entry(&mut self, rel: &str) {
        let Some(node) = self.nodes.remove(rel) else {
            return;
        };
        self.quota.release(Self::node_cost(rel, &node));
        if let Node::File { ino } = node
            && let Some(f) = self.files.get_mut(&ino)
        {
            f.links -= 1;
            if f.links == 0 {
                self.files.remove(&ino);
            }
        }
    }

    /// The mount-relative parent path of `rel` (`""` for a top-level entry).
    fn parent_of(rel: &str) -> &str {
        match rel.rfind('/') {
            Some(i) => &rel[..i],
            None => "",
        }
    }

    fn base_name(rel: &str) -> &str {
        match rel.rfind('/') {
            Some(i) => &rel[i + 1..],
            None => rel,
        }
    }

    /// `ENOTDIR` if any proper ancestor of `rel` exists but is not a directory
    /// (e.g. resolving `a/b/c` where `a` is a regular file). Missing ancestors
    /// are not an error here — the caller decides whether that's `ENOENT`.
    fn ancestors_are_dirs(&self, rel: &str) -> io::Result<()> {
        let parent = Self::parent_of(rel);
        if parent.is_empty() {
            return Ok(());
        }
        let mut prefix = String::new();
        for c in parent.split('/') {
            if !prefix.is_empty() {
                prefix.push('/');
            }
            prefix.push_str(c);
            match self.nodes.get(&prefix) {
                Some(Node::Dir { .. }) | None => {}
                Some(_) => return Err(enotdir()),
            }
        }
        Ok(())
    }

    /// Fail unless the parent directory of `rel` exists and is a directory:
    /// `ENOTDIR` if a component along the way is a non-directory, otherwise
    /// `ENOENT` if the immediate parent is missing.
    fn require_parent(&self, rel: &str) -> io::Result<()> {
        self.ancestors_are_dirs(rel)?;
        match self.nodes.get(Self::parent_of(rel)) {
            Some(Node::Dir { .. }) => Ok(()),
            Some(_) => Err(enotdir()),
            None => Err(enoent()),
        }
    }
}

fn enoent() -> io::Error {
    io::Error::from_raw_os_error(2)
}
fn enospc() -> io::Error {
    io::Error::from_raw_os_error(28)
}
fn efbig() -> io::Error {
    io::Error::from_raw_os_error(27)
}
fn eperm() -> io::Error {
    io::Error::from_raw_os_error(1)
}
fn eexist() -> io::Error {
    io::Error::from_raw_os_error(17)
}
fn enotdir() -> io::Error {
    io::Error::from_raw_os_error(20)
}
fn eisdir() -> io::Error {
    io::Error::from_raw_os_error(21)
}
fn einval() -> io::Error {
    io::Error::from_raw_os_error(22)
}
fn enotempty() -> io::Error {
    io::Error::from_raw_os_error(39)
}
fn eopnotsupp() -> io::Error {
    io::Error::from_raw_os_error(95)
}
fn enxio() -> io::Error {
    io::Error::from_raw_os_error(6)
}
fn enodata() -> io::Error {
    io::Error::from_raw_os_error(61)
}

/// Current wall-clock time as Unix seconds, for `mtime` on write
/// (wasm32-safe — see [`crate::clock`]).
fn now_ts() -> i64 {
    crate::clock::now_unix().as_secs() as i64
}

impl MountFs for TmpFs {
    fn attach_pages(&mut self, store: &Arc<dyn PageStore>) {
        if self.shmem && self.pages.is_none() {
            self.pages = Some(Arc::clone(store));
        }
    }

    fn page_id(&mut self, rel: &str) -> Option<u64> {
        match self.nodes.get(rel)? {
            Node::File { ino } => self.files.get(ino)?.paged.as_ref().map(|(id, _)| *id),
            _ => None,
        }
    }

    fn attach_quota(&mut self, quota: &Arc<Quota>) {
        let nodes: u64 = self.nodes.iter().map(|(k, n)| Self::node_cost(k, n)).sum();
        self.quota.release(nodes);
        quota.force_charge(nodes);
        for f in self.files.values_mut() {
            f.quota.release(f.cost());
            quota.force_charge(f.cost());
            f.quota = quota.clone();
        }
        self.quota = quota.clone();
    }

    fn read_only(&self) -> bool {
        false
    }

    fn stat(&mut self, rel: &str) -> Option<Attrs> {
        let node = self.nodes.get(rel)?;
        let mut blocks = None;
        let (kind, mode, size, nlink) = match node {
            Node::Dir { mode, .. } => {
                // Directory hard-link count: "." plus ".." plus one ".." per
                // immediate subdirectory (the standard Unix accounting).
                let subdirs = self
                    .nodes
                    .iter()
                    .filter(|(k, n)| {
                        Self::parent_of(k) == rel && !k.is_empty() && matches!(n, Node::Dir { .. })
                    })
                    .count();
                (
                    NodeKind::Dir,
                    S_IFDIR | mode,
                    0,
                    2 + u32::try_from(subdirs).unwrap_or(u32::MAX),
                )
            }
            Node::File { ino } => {
                let f = self.files.get(ino)?;
                let (size, pages) = match &f.paged {
                    Some((id, store)) => (store.len(*id), store.resident(*id)),
                    None => (f.data.len, f.data.pages.len() as u64),
                };
                blocks = Some(pages * (PAGE / 512));
                (NodeKind::File, S_IFREG | (f.mode & 0o7777), size, f.links)
            }
            Node::Fifo {
                mode, socket: true, ..
            } => (NodeKind::Socket, S_IFSOCK | (mode & 0o7777), 0, 1),
            Node::Fifo { mode, .. } => (NodeKind::Fifo, S_IFIFO | (mode & 0o7777), 0, 1),
            Node::Symlink { target, .. } => {
                (NodeKind::Symlink, S_IFLNK | 0o777, target.len() as u64, 1)
            }
        };
        let meta = self.meta_of(rel)?;
        Some(Attrs {
            kind,
            size,
            mode,
            uid: meta.uid,
            gid: meta.gid,
            atime: meta.atime,
            mtime: meta.mtime,
            inode: meta.inode,
            nlink,
            rdev: 0,
            dev: 0,
            blocks,
        })
    }

    fn seek_data(&mut self, rel: &str, off: u64, size: u64, hole: bool) -> io::Result<u64> {
        let f = match self.nodes.get(rel) {
            Some(Node::File { ino }) => self.files.get(ino).ok_or_else(enoent)?,
            Some(_) => return Err(einval()),
            None => return Err(enoent()),
        };
        if f.paged.is_some() {
            // Shared memory: its pages are the store's, which keeps no map
            // of its holes to ask; everything is data.
            return Ok(if hole { size } else { off });
        }
        f.data.seek(off, hole).ok_or_else(enxio)
    }

    fn read_at(&mut self, rel: &str, off: u64, buf: &mut [u8]) -> io::Result<usize> {
        match self.nodes.get(rel) {
            Some(Node::File { ino }) => {
                let f = self.files.get(ino).ok_or_else(enoent)?;
                if let Some((id, store)) = &f.paged {
                    return Ok(store.read(*id, off, buf));
                }
                Ok(f.data.read(off, buf))
            }
            Some(_) => Err(io::Error::from_raw_os_error(21)), // EISDIR
            None => Err(enoent()),
        }
    }

    fn readdir(&mut self, rel: &str) -> io::Result<Vec<DirEntry>> {
        match self.nodes.get(rel) {
            Some(Node::Dir { .. }) => {}
            Some(_) => return Err(enotdir()),
            None => return Err(enoent()),
        }
        let mut out = Vec::new();
        for (path, node) in &self.nodes {
            if path.is_empty() || Self::parent_of(path) != rel {
                continue;
            }
            let (kind, inode) = match node {
                Node::Dir { meta, .. } => (NodeKind::Dir, meta.inode),
                Node::File { ino } => (NodeKind::File, *ino),
                Node::Fifo {
                    meta, socket: true, ..
                } => (NodeKind::Socket, meta.inode),
                Node::Fifo { meta, .. } => (NodeKind::Fifo, meta.inode),
                Node::Symlink { meta, .. } => (NodeKind::Symlink, meta.inode),
            };
            out.push(DirEntry {
                name: Self::base_name(path).to_string(),
                kind,
                inode,
            });
        }
        Ok(out)
    }

    fn write_at(&mut self, rel: &str, off: u64, buf: &[u8]) -> io::Result<usize> {
        match self.nodes.get(rel) {
            Some(Node::File { .. }) => {
                let f = self.file_mut(rel).ok_or_else(enoent)?;
                if let Some((id, store)) = &f.paged {
                    off.checked_add(buf.len() as u64)
                        .filter(|&end| end <= MAX_FILE_SIZE)
                        .ok_or_else(efbig)?;
                    store.write(*id, off, buf)?;
                    f.meta.mtime = now_ts();
                    return Ok(buf.len());
                }
                off.checked_add(buf.len() as u64)
                    .filter(|&end| end <= MAX_FILE_SIZE)
                    .ok_or_else(efbig)?;
                f.data.write(off, buf, &f.quota)?;
                f.meta.mtime = now_ts();
                Ok(buf.len())
            }
            Some(_) => Err(eisdir()),
            None => Err(enoent()),
        }
    }

    fn create(&mut self, rel: &str, mode: u32) -> io::Result<()> {
        self.require_parent(rel)?;
        if self.nodes.contains_key(rel) {
            return Err(eexist());
        }
        let ino = self.new_file(mode);
        if let Err(e) = self.insert_node(rel, Node::File { ino }) {
            self.files.remove(&ino);
            return Err(e);
        }
        Ok(())
    }

    fn mkdir(&mut self, rel: &str, mode: u32) -> io::Result<()> {
        self.require_parent(rel)?;
        if self.nodes.contains_key(rel) {
            return Err(eexist());
        }
        let meta = self.new_meta();
        let mode = mode & 0o7777;
        self.insert_node(rel, Node::Dir { meta, mode })?;
        Ok(())
    }

    fn mknod(&mut self, rel: &str, mode: u32) -> io::Result<()> {
        // Only regular files and FIFOs are representable in this in-memory
        // backend. A type of 0 means a regular file (mknod(2) semantics);
        // device/socket nodes report EPERM, matching an unprivileged mknod.
        let typ = mode & 0o170_000;
        let (is_fifo, socket) = match typ {
            0 | S_IFREG => (false, false),
            S_IFIFO => (true, false),
            S_IFSOCK => (true, true),
            _ => return Err(eperm()),
        };
        self.require_parent(rel)?;
        if self.nodes.contains_key(rel) {
            return Err(eexist());
        }
        let node = if is_fifo {
            let meta = self.new_meta();
            Node::Fifo {
                meta,
                mode: mode & 0o7777,
                socket,
            }
        } else {
            Node::File {
                ino: self.new_file(mode & 0o7777),
            }
        };
        let ino = match node {
            Node::File { ino } => Some(ino),
            _ => None,
        };
        let inserted = self.insert_node(rel, node);
        if let (Err(_), Some(ino)) = (&inserted, ino) {
            self.files.remove(&ino);
        }
        inserted
    }

    fn unlink(&mut self, rel: &str) -> io::Result<()> {
        self.ancestors_are_dirs(rel)?;
        match self.nodes.get(rel) {
            Some(Node::Dir { .. }) => Err(eisdir()),
            Some(_) => {
                self.remove_entry(rel);
                Ok(())
            }
            None => Err(enoent()),
        }
    }

    fn rmdir(&mut self, rel: &str) -> io::Result<()> {
        self.ancestors_are_dirs(rel)?;
        match self.nodes.get(rel) {
            Some(Node::Dir { .. }) => {}
            Some(_) => return Err(enotdir()),
            None => return Err(enoent()),
        }
        if self
            .nodes
            .keys()
            .any(|k| Self::parent_of(k) == rel && !k.is_empty())
        {
            return Err(io::Error::from_raw_os_error(39)); // ENOTEMPTY
        }
        self.remove_entry(rel);
        Ok(())
    }

    fn truncate(&mut self, rel: &str, len: u64) -> io::Result<()> {
        match self.nodes.get(rel) {
            Some(Node::File { .. }) => {
                let f = self.file_mut(rel).ok_or_else(enoent)?;
                if let Some((id, store)) = &f.paged {
                    if len > MAX_FILE_SIZE {
                        return Err(efbig());
                    }
                    store.set_len(*id, len)?;
                    f.meta.mtime = now_ts();
                    return Ok(());
                }
                if len > MAX_FILE_SIZE {
                    return Err(efbig());
                }
                f.data.set_len(len, &f.quota);
                f.meta.mtime = now_ts();
                Ok(())
            }
            Some(Node::Dir { .. }) => Err(eisdir()),
            Some(_) => Err(einval()),
            None => Err(enoent()),
        }
    }

    fn allocate(&mut self, rel: &str, off: u64, len: u64) -> io::Result<()> {
        let f = self.file_mut(rel).ok_or_else(enoent)?;
        if f.paged.is_some() {
            return Err(eopnotsupp());
        }
        let end = off
            .checked_add(len)
            .filter(|&end| end <= MAX_FILE_SIZE)
            .ok_or_else(efbig)?;
        f.data.allocate(off, end, &f.quota)
    }

    fn punch_hole(&mut self, rel: &str, off: u64, len: u64) -> io::Result<()> {
        let f = self.file_mut(rel).ok_or_else(enoent)?;
        if f.paged.is_some() {
            return Err(eopnotsupp());
        }
        f.data.punch(off, off.saturating_add(len), &f.quota);
        f.meta.mtime = now_ts();
        Ok(())
    }

    fn set_mode(&mut self, rel: &str, mode: u32) -> io::Result<()> {
        match self.nodes.get_mut(rel) {
            // Symlinks have no mode of their own.
            Some(Node::File { ino }) => {
                let ino = *ino;
                if let Some(f) = self.files.get_mut(&ino) {
                    f.mode = (f.mode & !0o7777) | (mode & 0o7777);
                }
                Ok(())
            }
            Some(Node::Fifo { mode: m, .. } | Node::Dir { mode: m, .. }) => {
                *m = (*m & !0o7777) | (mode & 0o7777);
                Ok(())
            }
            Some(_) => Ok(()),
            None => Err(enoent()),
        }
    }

    fn set_times(&mut self, rel: &str, atime: SetTime, mtime: SetTime) -> io::Result<()> {
        let now = now_ts();
        let apply = |slot: &mut i64, t: SetTime| match t {
            SetTime::Omit => {}
            SetTime::Now => *slot = now,
            SetTime::Set { sec, .. } => *slot = sec,
        };
        let meta = self.meta_of_mut(rel).ok_or_else(enoent)?;
        apply(&mut meta.atime, atime);
        apply(&mut meta.mtime, mtime);
        Ok(())
    }

    fn set_owner(&mut self, rel: &str, uid: Option<u32>, gid: Option<u32>) -> io::Result<()> {
        let meta = self.meta_of_mut(rel).ok_or_else(enoent)?;
        if let Some(u) = uid {
            meta.uid = u;
        }
        if let Some(g) = gid {
            meta.gid = g;
        }
        Ok(())
    }

    fn getxattr(&mut self, rel: &str, name: &str) -> io::Result<Vec<u8>> {
        let meta = self.meta_of(rel).ok_or_else(enoent)?;
        meta.xattrs.get(name).cloned().ok_or_else(enodata)
    }

    fn setxattr(&mut self, rel: &str, name: &str, value: &[u8]) -> io::Result<()> {
        let quota = self.quota.clone();
        let meta = self.meta_of_mut(rel).ok_or_else(enoent)?;
        let new_cost = (name.len() + value.len()) as u64 + NODE_OVERHEAD;
        if !quota.try_charge(new_cost) {
            return Err(enospc());
        }
        if let Some(old) = meta.xattrs.insert(name.to_string(), value.to_vec()) {
            quota.release((name.len() + old.len()) as u64 + NODE_OVERHEAD);
        }
        Ok(())
    }

    fn listxattr(&mut self, rel: &str) -> io::Result<Vec<String>> {
        let meta = self.meta_of(rel).ok_or_else(enoent)?;
        Ok(meta.xattrs.keys().cloned().collect())
    }

    fn removexattr(&mut self, rel: &str, name: &str) -> io::Result<()> {
        let quota = self.quota.clone();
        let meta = self.meta_of_mut(rel).ok_or_else(enoent)?;
        let old = meta.xattrs.remove(name).ok_or_else(enodata)?;
        quota.release((name.len() + old.len()) as u64 + NODE_OVERHEAD);
        Ok(())
    }

    fn symlink(&mut self, target: &str, linkpath: &str) -> io::Result<()> {
        self.require_parent(linkpath)?;
        if self.nodes.contains_key(linkpath) {
            return Err(eexist());
        }
        let meta = self.new_meta();
        self.insert_node(
            linkpath,
            Node::Symlink {
                meta,
                target: target.to_string(),
            },
        )
    }

    fn readlink(&mut self, rel: &str) -> io::Result<String> {
        match self.nodes.get(rel) {
            Some(Node::Symlink { target, .. }) => Ok(target.clone()),
            Some(_) => Err(io::Error::from_raw_os_error(22)), // EINVAL
            None => Err(enoent()),
        }
    }

    /// A real hard link: `new_rel` names the same inode as `old_rel`. Only
    /// regular files are inodes here; a directory is `EPERM` (as on Linux),
    /// and a FIFO or symlink is `EOPNOTSUPP` (the kernel then recreates it).
    fn link(&mut self, old_rel: &str, new_rel: &str) -> io::Result<()> {
        let ino = match self.nodes.get(old_rel) {
            Some(Node::File { ino }) => *ino,
            Some(Node::Dir { .. }) => return Err(eperm()),
            Some(_) => return Err(io::Error::from_raw_os_error(95)), // EOPNOTSUPP
            None => return Err(enoent()),
        };
        self.require_parent(new_rel)?;
        if self.nodes.contains_key(new_rel) {
            return Err(eexist());
        }
        self.insert_node(new_rel, Node::File { ino })?;
        if let Some(f) = self.files.get_mut(&ino) {
            f.links += 1;
        }
        Ok(())
    }

    fn rename(&mut self, from: &str, to: &str) -> io::Result<()> {
        if from == to {
            return if self.nodes.contains_key(from) {
                Ok(())
            } else {
                Err(enoent())
            };
        }
        let from_is_dir = match self.nodes.get(from) {
            Some(Node::Dir { .. }) => true,
            Some(_) => false,
            None => return Err(enoent()),
        };
        // Renaming one name of a file onto another name of the same file is
        // a successful no-op (POSIX), leaving both names.
        if let (Some(Node::File { ino: a }), Some(Node::File { ino: b })) =
            (self.nodes.get(from), self.nodes.get(to))
            && a == b
        {
            return Ok(());
        }
        // Refuse to move a directory into itself or one of its own
        // descendants (mirrors Linux `rename(2)`'s EINVAL).
        if from_is_dir && (to == from || to.starts_with(&format!("{from}/"))) {
            return Err(einval());
        }
        self.require_parent(to)?;
        if let Some(to_node) = self.nodes.get(to) {
            match (from_is_dir, to_node) {
                // Directory onto directory: only if the destination is empty.
                (true, Node::Dir { .. }) => {
                    if self.nodes.keys().any(|k| Self::parent_of(k) == to) {
                        return Err(enotempty());
                    }
                }
                // Directory onto a non-directory, or vice versa: cross-type
                // rename is rejected.
                (true, _) => return Err(enotdir()),
                (false, Node::Dir { .. }) => return Err(eisdir()),
                // File/symlink onto an existing file/symlink: replaces it.
                (false, _) => {}
            }
            self.remove_entry(to);
        }
        // Move the node itself and, for a directory, every descendant, by
        // rewriting the path prefix.
        let prefix = format!("{from}/");
        let moved: Vec<String> = self
            .nodes
            .keys()
            .filter(|k| *k == from || k.starts_with(&prefix))
            .cloned()
            .collect();
        for key in moved {
            let node = self.nodes.remove(&key).expect("just collected from nodes");
            let new_key = if key == from {
                to.to_string()
            } else {
                format!("{to}{}", &key[from.len()..])
            };
            // A rename must not fail halfway through a subtree: a longer name
            // is charged even if it goes over, and released when removed.
            self.quota.force_charge(new_key.len() as u64);
            self.quota.release(key.len() as u64);
            self.nodes.insert(new_key, node);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limited(limit: u64) -> (TmpFs, Arc<Quota>) {
        let mut fs = TmpFs::new();
        let quota = Quota::new(limit);
        fs.attach_quota(&quota);
        (fs, quota)
    }

    fn errno(e: &io::Error) -> i32 {
        e.raw_os_error().unwrap()
    }

    #[test]
    fn a_write_past_the_quota_is_enospc_and_leaves_the_file_alone() {
        let (mut fs, _quota) = limited(1024 * 1024);
        fs.create("f", 0o644).unwrap();
        fs.write_at("f", 0, b"keep").unwrap();
        let big = vec![0u8; 2 * 1024 * 1024];
        assert_eq!(errno(&fs.write_at("f", 0, &big).unwrap_err()), 28);
        assert_eq!(fs.stat("f").unwrap().size, 4);
    }

    #[test]
    fn freeing_a_file_returns_its_quota() {
        let (mut fs, quota) = limited(u64::MAX);
        let baseline = quota.used();
        fs.create("f", 0o644).unwrap();
        fs.write_at("f", 0, &vec![0; 100_000]).unwrap();
        fs.setxattr("f", "user.a", b"value").unwrap();
        fs.link("f", "g").unwrap();
        assert!(quota.used() >= baseline + 100_000);
        fs.unlink("f").unwrap();
        assert!(quota.used() >= baseline + 100_000, "still linked as g");
        fs.unlink("g").unwrap();
        assert_eq!(quota.used(), baseline);
    }

    #[test]
    fn extending_past_the_largest_offset_is_efbig() {
        let (mut fs, _quota) = limited(1024 * 1024);
        fs.create("f", 0o644).unwrap();
        assert_eq!(errno(&fs.truncate("f", u64::MAX).unwrap_err()), 27);
        assert_eq!(errno(&fs.write_at("f", u64::MAX, b"x").unwrap_err()), 27);
        assert_eq!(
            errno(&fs.write_at("f", i64::MAX as u64, b"x").unwrap_err()),
            27
        );
        assert_eq!(fs.stat("f").unwrap().size, 0);
        // Up to it, a byte far out is one page.
        fs.write_at("f", 1 << 62, b"x").unwrap();
        assert_eq!(fs.stat("f").unwrap().size, (1 << 62) + 1);
        assert_eq!(fs.stat("f").unwrap().blocks, Some(8));
    }

    #[test]
    fn a_sparse_file_costs_only_the_pages_written() {
        let (mut fs, quota) = limited(64 * 1024);
        fs.create("f", 0o644).unwrap();
        let start = quota.used();
        // Far past the quota, as a hole: nothing held, as on Linux.
        fs.truncate("f", 8 << 30).unwrap();
        assert_eq!(quota.used(), start);
        let a = fs.stat("f").unwrap();
        assert_eq!((a.size, a.blocks), (8 << 30, Some(0)));
        // Four bytes in the middle: one page; across a page boundary, two.
        fs.write_at("f", 4 << 30, b"abcd").unwrap();
        assert_eq!(quota.used(), start + PAGE);
        fs.write_at("f", (5 << 30) - 2, b"abcd").unwrap();
        assert_eq!(quota.used(), start + 3 * PAGE);
        assert_eq!(fs.stat("f").unwrap().blocks, Some(3 * 8));
        // A hole reads as zeros, around and between what was written.
        let mut buf = [0xffu8; 12];
        assert_eq!(fs.read_at("f", (4 << 30) - 4, &mut buf).unwrap(), 12);
        assert_eq!(&buf, b"\0\0\0\0abcd\0\0\0\0");
        let mut buf = vec![0xffu8; 3 * PAGE as usize];
        fs.read_at("f", 6 << 30, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0));
        // Rewriting in place holds nothing more.
        fs.write_at("f", 4 << 30, b"ABCD").unwrap();
        assert_eq!(quota.used(), start + 3 * PAGE);
    }

    #[test]
    fn truncating_down_frees_pages_and_zeroes_the_tail() {
        let (mut fs, quota) = limited(u64::MAX);
        fs.create("f", 0o644).unwrap();
        let start = quota.used();
        fs.write_at("f", 0, &vec![7; 1_000_000]).unwrap();
        assert_eq!(quota.used(), start + 1_000_000u64.div_ceil(PAGE) * PAGE);
        fs.truncate("f", 10).unwrap();
        assert_eq!(quota.used(), start + PAGE);
        // Growing again shows zeros past the old end, not the old bytes.
        fs.truncate("f", 100).unwrap();
        assert_eq!(quota.used(), start + PAGE);
        let mut buf = [0xffu8; 100];
        assert_eq!(fs.read_at("f", 0, &mut buf).unwrap(), 100);
        assert_eq!(&buf[..10], &[7; 10]);
        assert!(buf[10..].iter().all(|&b| b == 0));
        fs.truncate("f", 0).unwrap();
        assert_eq!(quota.used(), start);
    }

    #[test]
    fn seek_data_and_hole_find_the_written_pages() {
        let mut fs = TmpFs::new();
        fs.create("f", 0o644).unwrap();
        let size = 10 * PAGE + 100;
        fs.truncate("f", size).unwrap();
        let seek = |fs: &mut TmpFs, off, hole| fs.seek_data("f", off, size, hole);
        // All hole: no data, and the hole is where we are.
        assert_eq!(errno(&seek(&mut fs, 0, false).unwrap_err()), 6);
        assert_eq!(seek(&mut fs, 5, true).unwrap(), 5);
        fs.write_at("f", 3 * PAGE + 1, &[1; 2 * PAGE as usize])
            .unwrap(); // pages 3..=5
        assert_eq!(seek(&mut fs, 0, false).unwrap(), 3 * PAGE);
        assert_eq!(seek(&mut fs, 4 * PAGE + 9, false).unwrap(), 4 * PAGE + 9);
        assert_eq!(seek(&mut fs, 3 * PAGE, true).unwrap(), 6 * PAGE);
        assert_eq!(errno(&seek(&mut fs, 6 * PAGE, false).unwrap_err()), 6);
        // Data in the last page: the next hole is the end of the file.
        fs.write_at("f", size - 1, b"z").unwrap();
        assert_eq!(seek(&mut fs, 6 * PAGE, false).unwrap(), 10 * PAGE);
        assert_eq!(seek(&mut fs, 10 * PAGE, true).unwrap(), size);
    }

    #[test]
    fn a_write_with_room_for_only_part_of_it_changes_nothing() {
        let (mut fs, quota) = limited(64 * 1024);
        fs.create("f", 0o644).unwrap();
        fs.write_at("f", 0, &[1; 100]).unwrap();
        let held = quota.used();
        assert_eq!(
            errno(&fs.write_at("f", 50, &vec![2; 128 * 1024]).unwrap_err()),
            28
        );
        assert_eq!(quota.used(), held);
        let mut buf = [0u8; 101];
        assert_eq!(fs.read_at("f", 0, &mut buf).unwrap(), 100);
        assert!(buf[..100].iter().all(|&b| b == 1));
    }

    #[test]
    fn fallocate_holds_pages_and_punching_frees_them() {
        let (mut fs, quota) = limited(64 * 1024);
        fs.create("f", 0o644).unwrap();
        let start = quota.used();
        // Allocating holds the pages, without changing the size.
        fs.allocate("f", 100, 3 * PAGE).unwrap();
        assert_eq!(quota.used(), start + 4 * PAGE);
        assert_eq!(fs.stat("f").unwrap().size, 0);
        // More than there's room for is ENOSPC, holding nothing more.
        assert_eq!(errno(&fs.allocate("f", 0, 1 << 62).unwrap_err()), 28);
        assert_eq!(errno(&fs.allocate("f", 0, 1 << 63).unwrap_err()), 27);
        assert_eq!(quota.used(), start + 4 * PAGE);
        // A hole punched in written data frees the pages wholly inside it
        // and zeroes the edges of the others.
        fs.write_at("f", 0, &[9; 4 * PAGE as usize]).unwrap();
        fs.punch_hole("f", 10, 2 * PAGE + 20).unwrap();
        assert_eq!(quota.used(), start + 3 * PAGE);
        let mut buf = vec![0u8; 4 * PAGE as usize];
        fs.read_at("f", 0, &mut buf).unwrap();
        let hole = 10..(2 * PAGE + 30) as usize;
        for (i, &b) in buf.iter().enumerate() {
            assert_eq!(b, if hole.contains(&i) { 0 } else { 9 }, "byte {i}");
        }
        assert_eq!(fs.stat("f").unwrap().size, 4 * PAGE);
    }

    #[test]
    fn empty_files_and_directories_cannot_grow_without_bound() {
        let (mut fs, quota) = limited(10_000);
        let mut made = 0;
        let failure = loop {
            match fs.create(&format!("f{made}"), 0o644) {
                Ok(()) => made += 1,
                Err(e) => break e,
            }
        };
        assert_eq!(errno(&failure), 28);
        assert!(made > 0 && quota.used() <= 10_000);
        assert_eq!(errno(&fs.mkdir("d", 0o755).unwrap_err()), 28);
        assert_eq!(errno(&fs.symlink("t", "l").unwrap_err()), 28);
        // Deleting makes room again, and a failed create left no stray inode.
        fs.unlink("f0").unwrap();
        fs.create("again", 0o644).unwrap();
        assert_eq!(fs.files.len(), made);
    }

    #[test]
    fn xattrs_are_charged_and_replacing_one_does_not_leak() {
        let (mut fs, quota) = limited(u64::MAX);
        fs.create("f", 0o644).unwrap();
        let start = quota.used();
        fs.setxattr("f", "user.a", &[0; 1000]).unwrap();
        fs.setxattr("f", "user.a", &[0; 1000]).unwrap();
        assert!(quota.used() >= start + 1000 && quota.used() < start + 2000);
        fs.removexattr("f", "user.a").unwrap();
        assert_eq!(quota.used(), start);
    }

    #[test]
    fn create_write_read_roundtrip() {
        let mut fs = TmpFs::new();
        fs.create("hello.txt", 0o644).unwrap();
        assert_eq!(fs.write_at("hello.txt", 0, b"hi there").unwrap(), 8);
        let mut buf = [0u8; 8];
        assert_eq!(fs.read_at("hello.txt", 0, &mut buf).unwrap(), 8);
        assert_eq!(&buf, b"hi there");
        assert_eq!(fs.stat("hello.txt").unwrap().size, 8);
    }

    #[test]
    fn mkdir_and_readdir() {
        let mut fs = TmpFs::new();
        fs.mkdir("d", 0o755).unwrap();
        fs.create("d/a", 0o644).unwrap();
        fs.create("d/b", 0o644).unwrap();
        fs.create("top", 0o644).unwrap();

        let mut names: Vec<_> = fs
            .readdir("d")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);

        let root: Vec<_> = fs
            .readdir("")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert!(root.contains(&"d".to_string()) && root.contains(&"top".to_string()));
    }

    #[test]
    fn create_needs_parent_and_rejects_dup() {
        let mut fs = TmpFs::new();
        assert!(fs.create("missing/f", 0o644).is_err());
        fs.create("f", 0o644).unwrap();
        assert!(fs.create("f", 0o644).is_err());
    }

    #[test]
    fn rmdir_requires_empty() {
        let mut fs = TmpFs::new();
        fs.mkdir("d", 0o755).unwrap();
        fs.create("d/x", 0o644).unwrap();
        assert!(fs.rmdir("d").is_err());
        fs.unlink("d/x").unwrap();
        fs.rmdir("d").unwrap();
        assert!(fs.stat("d").is_none());
    }

    #[test]
    fn rename_moves_subtree() {
        let mut fs = TmpFs::new();
        fs.mkdir("a", 0o755).unwrap();
        fs.create("a/f", 0o644).unwrap();
        fs.write_at("a/f", 0, b"data").unwrap();
        fs.rename("a", "b").unwrap();
        assert!(fs.stat("a").is_none());
        assert!(fs.stat("b").is_some());
        let mut buf = [0u8; 4];
        fs.read_at("b/f", 0, &mut buf).unwrap();
        assert_eq!(&buf, b"data");
    }

    #[test]
    fn symlink_readlink() {
        let mut fs = TmpFs::new();
        fs.symlink("/target", "link").unwrap();
        assert_eq!(fs.readlink("link").unwrap(), "/target");
        assert_eq!(fs.stat("link").unwrap().kind, NodeKind::Symlink);
    }

    #[test]
    fn rename_over_existing_file_replaces_it() {
        let mut fs = TmpFs::new();
        fs.create("a", 0o644).unwrap();
        fs.write_at("a", 0, b"AAA").unwrap();
        fs.create("b", 0o644).unwrap();
        fs.write_at("b", 0, b"B").unwrap();
        fs.rename("a", "b").unwrap();
        assert!(fs.stat("a").is_none());
        let mut buf = [0u8; 3];
        fs.read_at("b", 0, &mut buf).unwrap();
        assert_eq!(&buf, b"AAA");
    }

    #[test]
    fn rename_dir_onto_empty_dir_succeeds() {
        let mut fs = TmpFs::new();
        fs.mkdir("a", 0o755).unwrap();
        fs.create("a/f", 0o644).unwrap();
        fs.mkdir("b", 0o755).unwrap();
        fs.rename("a", "b").unwrap();
        assert!(fs.stat("a").is_none());
        assert!(fs.stat("b/f").is_some());
    }

    #[test]
    fn rename_dir_onto_nonempty_dir_fails_enotempty() {
        let mut fs = TmpFs::new();
        fs.mkdir("a", 0o755).unwrap();
        fs.mkdir("b", 0o755).unwrap();
        fs.create("b/f", 0o644).unwrap();
        let err = fs.rename("a", "b").unwrap_err();
        assert_eq!(err.raw_os_error(), Some(39)); // ENOTEMPTY
        // Nothing moved.
        assert!(fs.stat("a").is_some());
        assert!(fs.stat("b/f").is_some());
    }

    #[test]
    fn rename_dir_into_own_subtree_fails_einval() {
        let mut fs = TmpFs::new();
        fs.mkdir("a", 0o755).unwrap();
        fs.mkdir("a/b", 0o755).unwrap();
        let err = fs.rename("a", "a/b/c").unwrap_err();
        assert_eq!(err.raw_os_error(), Some(22)); // EINVAL
    }

    #[test]
    fn rename_cross_type_rejected() {
        let mut fs = TmpFs::new();
        fs.mkdir("d", 0o755).unwrap();
        fs.create("f", 0o644).unwrap();
        // Directory onto a file: ENOTDIR.
        assert_eq!(fs.rename("d", "f").unwrap_err().raw_os_error(), Some(20));
        // File onto a directory: EISDIR.
        assert_eq!(fs.rename("f", "d").unwrap_err().raw_os_error(), Some(21));
    }

    #[test]
    fn rmdir_nonempty_fails_enotempty() {
        let mut fs = TmpFs::new();
        fs.mkdir("d", 0o755).unwrap();
        fs.create("d/x", 0o644).unwrap();
        let err = fs.rmdir("d").unwrap_err();
        assert_eq!(err.raw_os_error(), Some(39)); // ENOTEMPTY
    }

    #[test]
    fn mkdir_existing_path_fails_eexist() {
        let mut fs = TmpFs::new();
        fs.mkdir("d", 0o755).unwrap();
        let err = fs.mkdir("d", 0o755).unwrap_err();
        assert_eq!(err.raw_os_error(), Some(17)); // EEXIST
        // Also EEXIST when the existing path is a file, not a dir.
        fs.create("f", 0o644).unwrap();
        assert_eq!(fs.mkdir("f", 0o755).unwrap_err().raw_os_error(), Some(17));
    }

    #[test]
    fn write_updates_mtime() {
        let mut fs = TmpFs::new();
        fs.create("f", 0o644).unwrap();
        let before = fs.stat("f").unwrap().mtime;
        fs.write_at("f", 0, b"x").unwrap();
        let after = fs.stat("f").unwrap().mtime;
        assert!(after >= before);
    }

    #[test]
    fn set_mode_updates_permission_bits() {
        let mut fs = TmpFs::new();
        fs.create("f", 0o644).unwrap();
        fs.set_mode("f", 0o755).unwrap();
        assert_eq!(fs.stat("f").unwrap().mode & 0o777, 0o755);
        assert!(fs.set_mode("nope", 0o755).is_err());
    }

    #[test]
    fn set_times_honors_each_field_and_omit() {
        let mut fs = TmpFs::new();
        fs.create("f", 0o644).unwrap();
        let set = |sec| SetTime::Set { sec, nsec: 0 };
        // Both fields stored independently.
        fs.set_times("f", set(111), set(222)).unwrap();
        let a = fs.stat("f").unwrap();
        assert_eq!((a.atime, a.mtime), (111, 222));
        // UTIME_OMIT on mtime leaves it, sets atime only.
        fs.set_times("f", set(333), SetTime::Omit).unwrap();
        let a = fs.stat("f").unwrap();
        assert_eq!((a.atime, a.mtime), (333, 222));
        // UTIME_OMIT on atime leaves it, sets mtime only.
        fs.set_times("f", SetTime::Omit, set(444)).unwrap();
        let a = fs.stat("f").unwrap();
        assert_eq!((a.atime, a.mtime), (333, 444));
        // A missing node is ENOENT.
        assert!(fs.set_times("nope", set(1), set(1)).is_err());
    }

    #[test]
    fn set_owner_stores_uid_gid() {
        let mut fs = TmpFs::new();
        fs.create("f", 0o644).unwrap();
        fs.set_owner("f", Some(1000), Some(1001)).unwrap();
        let a = fs.stat("f").unwrap();
        assert_eq!((a.uid, a.gid), (1000, 1001));
        // -1/None leaves a field unchanged.
        fs.set_owner("f", None, Some(2002)).unwrap();
        let a = fs.stat("f").unwrap();
        assert_eq!((a.uid, a.gid), (1000, 2002));
        assert!(fs.set_owner("nope", Some(1), None).is_err());
    }

    #[test]
    fn mknod_fifo_reports_as_fifo() {
        let mut fs = TmpFs::new();
        fs.mknod("p", S_IFIFO | 0o644).unwrap();
        assert_eq!(fs.stat("p").unwrap().kind, NodeKind::Fifo);
        assert_eq!(fs.stat("p").unwrap().mode & 0o170_000, S_IFIFO);
        // A plain (S_IFREG) mknod makes a regular file.
        fs.mknod("r", S_IFREG | 0o600).unwrap();
        assert_eq!(fs.stat("r").unwrap().kind, NodeKind::File);
    }

    #[test]
    fn dir_nlink_counts_subdirectories() {
        let mut fs = TmpFs::new();
        fs.mkdir("d", 0o755).unwrap();
        assert_eq!(fs.stat("d").unwrap().nlink, 2);
        fs.mkdir("d/sub1", 0o755).unwrap();
        fs.mkdir("d/sub2", 0o755).unwrap();
        fs.create("d/file", 0o644).unwrap();
        assert_eq!(fs.stat("d").unwrap().nlink, 4);
    }

    #[test]
    fn hard_links_share_one_inode() {
        let mut fs = TmpFs::new();
        fs.create("a", 0o644).unwrap();
        fs.write_at("a", 0, b"one").unwrap();
        fs.link("a", "b").unwrap();
        let (sa, sb) = (fs.stat("a").unwrap(), fs.stat("b").unwrap());
        assert_eq!(sa.inode, sb.inode);
        assert_eq!((sa.nlink, sb.nlink), (2, 2));
        // A write through one name shows through the other.
        fs.write_at("b", 0, b"TWO").unwrap();
        let mut buf = [0u8; 3];
        fs.read_at("a", 0, &mut buf).unwrap();
        assert_eq!(&buf, b"TWO");
        // Unlinking one name keeps the file under the other.
        fs.unlink("a").unwrap();
        assert_eq!(fs.stat("b").unwrap().nlink, 1);
        // Renaming a name onto another name of the same file is a no-op.
        fs.link("b", "c").unwrap();
        fs.rename("b", "c").unwrap();
        assert!(fs.stat("b").is_some() && fs.stat("c").is_some());
        // Linking a directory is EPERM; over an existing name, EEXIST.
        fs.mkdir("d", 0o755).unwrap();
        assert_eq!(fs.link("d", "e").unwrap_err().raw_os_error(), Some(1));
        assert_eq!(fs.link("b", "c").unwrap_err().raw_os_error(), Some(17));
        // The inode goes with its last name.
        fs.unlink("b").unwrap();
        fs.unlink("c").unwrap();
        assert!(fs.files.is_empty());
    }
}
