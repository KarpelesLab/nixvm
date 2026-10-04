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

use super::{Attrs, DirEntry, MountFs, NodeKind, SetTime};

/// Unix mode type bits.
const S_IFDIR: u32 = 0o040_000;
const S_IFREG: u32 = 0o100_000;
const S_IFLNK: u32 = 0o120_000;
const S_IFIFO: u32 = 0o010_000;

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
    data: Vec<u8>,
    mode: u32,
    /// How many path entries name this inode (`st_nlink`).
    links: u32,
}

#[derive(Debug)]
enum Node {
    Dir {
        meta: Meta,
    },
    /// A regular file: its inode in `TmpFs::files`.
    File {
        ino: u64,
    },
    Fifo {
        meta: Meta,
        mode: u32,
    },
    Symlink {
        meta: Meta,
        target: String,
    },
}

#[derive(Debug)]
pub struct TmpFs {
    nodes: BTreeMap<String, Node>,
    /// Regular-file inodes by inode number.
    files: BTreeMap<u64, FileData>,
    next_inode: u64,
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
            },
        );
        Self {
            nodes,
            files: BTreeMap::new(),
            next_inode: 2,
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
        self.files.insert(
            ino,
            FileData {
                meta,
                data: Vec::new(),
                mode,
                links: 1,
            },
        );
        ino
    }

    /// The metadata of the node at `rel` (a regular file's lives in its inode).
    fn meta_of(&self, rel: &str) -> Option<&Meta> {
        match self.nodes.get(rel)? {
            Node::Dir { meta } | Node::Fifo { meta, .. } | Node::Symlink { meta, .. } => Some(meta),
            Node::File { ino } => self.files.get(ino).map(|f| &f.meta),
        }
    }

    fn meta_of_mut(&mut self, rel: &str) -> Option<&mut Meta> {
        match self.nodes.get_mut(rel)? {
            Node::Dir { meta } | Node::Fifo { meta, .. } | Node::Symlink { meta, .. } => Some(meta),
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

    /// Drop the path entry `rel`, releasing a regular file's inode when it was
    /// the last name.
    fn remove_entry(&mut self, rel: &str) {
        if let Some(Node::File { ino }) = self.nodes.remove(rel)
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
fn enodata() -> io::Error {
    io::Error::from_raw_os_error(61)
}

/// Current wall-clock time as Unix seconds, for `mtime` on write
/// (wasm32-safe — see [`crate::clock`]).
fn now_ts() -> i64 {
    crate::clock::now_unix().as_secs() as i64
}

impl MountFs for TmpFs {
    fn read_only(&self) -> bool {
        false
    }

    fn stat(&mut self, rel: &str) -> Option<Attrs> {
        let node = self.nodes.get(rel)?;
        let (kind, mode, size, nlink) = match node {
            Node::Dir { .. } => {
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
                    S_IFDIR | 0o755,
                    0,
                    2 + u32::try_from(subdirs).unwrap_or(u32::MAX),
                )
            }
            Node::File { ino } => {
                let f = self.files.get(ino)?;
                (
                    NodeKind::File,
                    S_IFREG | (f.mode & 0o7777),
                    f.data.len() as u64,
                    f.links,
                )
            }
            Node::Fifo { mode, .. } => (NodeKind::Fifo, S_IFIFO | (mode & 0o777), 0, 1),
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
        })
    }

    fn read_at(&mut self, rel: &str, off: u64, buf: &mut [u8]) -> io::Result<usize> {
        match self.nodes.get(rel) {
            Some(Node::File { ino }) => {
                let data = &self.files.get(ino).ok_or_else(enoent)?.data;
                let off = off as usize;
                if off >= data.len() {
                    return Ok(0);
                }
                let n = buf.len().min(data.len() - off);
                buf[..n].copy_from_slice(&data[off..off + n]);
                Ok(n)
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
                Node::Dir { meta } => (NodeKind::Dir, meta.inode),
                Node::File { ino } => (NodeKind::File, *ino),
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
                let end = off as usize + buf.len();
                if f.data.len() < end {
                    f.data.resize(end, 0);
                }
                f.data[off as usize..end].copy_from_slice(buf);
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
        self.nodes.insert(rel.to_string(), Node::File { ino });
        Ok(())
    }

    fn mkdir(&mut self, rel: &str, _mode: u32) -> io::Result<()> {
        self.require_parent(rel)?;
        if self.nodes.contains_key(rel) {
            return Err(eexist());
        }
        let meta = self.new_meta();
        self.nodes.insert(rel.to_string(), Node::Dir { meta });
        Ok(())
    }

    fn mknod(&mut self, rel: &str, mode: u32) -> io::Result<()> {
        // Only regular files and FIFOs are representable in this in-memory
        // backend. A type of 0 means a regular file (mknod(2) semantics);
        // device/socket nodes report EPERM, matching an unprivileged mknod.
        let typ = mode & 0o170_000;
        let is_fifo = match typ {
            0 | S_IFREG => false,
            S_IFIFO => true,
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
            }
        } else {
            Node::File {
                ino: self.new_file(mode & 0o7777),
            }
        };
        self.nodes.insert(rel.to_string(), node);
        Ok(())
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
        self.nodes.remove(rel);
        Ok(())
    }

    fn truncate(&mut self, rel: &str, len: u64) -> io::Result<()> {
        match self.nodes.get(rel) {
            Some(Node::File { .. }) => {
                let f = self.file_mut(rel).ok_or_else(enoent)?;
                f.data.resize(len as usize, 0);
                f.meta.mtime = now_ts();
                Ok(())
            }
            Some(Node::Dir { .. }) => Err(eisdir()),
            Some(_) => Err(einval()),
            None => Err(enoent()),
        }
    }

    fn set_mode(&mut self, rel: &str, mode: u32) -> io::Result<()> {
        match self.nodes.get_mut(rel) {
            // Files and fifos store their mode; dirs/symlinks don't model one.
            Some(Node::File { ino }) => {
                let ino = *ino;
                if let Some(f) = self.files.get_mut(&ino) {
                    f.mode = (f.mode & !0o7777) | (mode & 0o7777);
                }
                Ok(())
            }
            Some(Node::Fifo { mode: m, .. }) => {
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
        let meta = self.meta_of_mut(rel).ok_or_else(enoent)?;
        meta.xattrs.insert(name.to_string(), value.to_vec());
        Ok(())
    }

    fn listxattr(&mut self, rel: &str) -> io::Result<Vec<String>> {
        let meta = self.meta_of(rel).ok_or_else(enoent)?;
        Ok(meta.xattrs.keys().cloned().collect())
    }

    fn removexattr(&mut self, rel: &str, name: &str) -> io::Result<()> {
        let meta = self.meta_of_mut(rel).ok_or_else(enoent)?;
        meta.xattrs.remove(name).map(drop).ok_or_else(enodata)
    }

    fn symlink(&mut self, target: &str, linkpath: &str) -> io::Result<()> {
        self.require_parent(linkpath)?;
        if self.nodes.contains_key(linkpath) {
            return Err(eexist());
        }
        let meta = self.new_meta();
        self.nodes.insert(
            linkpath.to_string(),
            Node::Symlink {
                meta,
                target: target.to_string(),
            },
        );
        Ok(())
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
        if let Some(f) = self.files.get_mut(&ino) {
            f.links += 1;
        }
        self.nodes.insert(new_rel.to_string(), Node::File { ino });
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
            self.nodes.insert(new_key, node);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
