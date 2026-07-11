//! Synthesized `/sys` (sysfs) filesystem.
//!
//! Presents the skeleton of a Linux `sysfs` tree — the top-level directories a
//! userland expects (`kernel`, `devices`, `class`, `fs`, `block`, `bus`,
//! `module`, `firmware`) plus a handful of fixed-content attribute files
//! (`kernel/ostype`, `devices/system/cpu/online`, …). Nothing here reflects
//! real hardware; the tree is entirely static and read-only, so every mutating
//! method falls through to the `EROFS` default.
//!
//! The tree is a `BTreeMap` keyed by mount-relative path (`""` is the `/sys`
//! root, then `"kernel"`, `"kernel/ostype"`, …); this keeps `readdir`
//! (children of a directory) a simple parent-path scan, mirroring `tmpfs`.

use std::collections::BTreeMap;
use std::io;
use std::sync::OnceLock;

use super::{Attrs, DirEntry, MountFs, NodeKind};

/// Unix mode type bit for a directory.
const S_IFDIR: u32 = 0o040_000;
/// Unix mode type bit for a regular file.
const S_IFREG: u32 = 0o100_000;

/// One node in the static `/sys` tree.
#[derive(Debug)]
enum Node {
    Dir { inode: u64 },
    File { inode: u64, data: &'static [u8] },
}

impl Node {
    fn inode(&self) -> u64 {
        match self {
            Node::Dir { inode } | Node::File { inode, .. } => *inode,
        }
    }
}

/// The synthesized `/sys` backend. The tree it serves is fixed at construction.
#[derive(Debug)]
pub struct SysFs;

impl Default for SysFs {
    fn default() -> Self {
        Self::new()
    }
}

impl SysFs {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// The shared static tree, built once on first access. Directories carry
    /// distinct small inodes; files carry their fixed contents.
    fn tree() -> &'static BTreeMap<&'static str, Node> {
        static TREE: OnceLock<BTreeMap<&'static str, Node>> = OnceLock::new();
        TREE.get_or_init(|| {
            let mut t = BTreeMap::new();
            // Directories: (path, inode). Root ("") reserves inode 1.
            let dirs: &[(&str, u64)] = &[
                ("", 1),
                ("kernel", 2),
                ("devices", 3),
                ("devices/system", 4),
                ("devices/system/cpu", 5),
                ("class", 6),
                ("fs", 7),
                ("block", 8),
                ("bus", 9),
                ("module", 10),
                ("firmware", 11),
            ];
            for &(path, inode) in dirs {
                t.insert(path, Node::Dir { inode });
            }
            // Files: (path, inode, contents).
            let files: &[(&str, u64, &[u8])] = &[
                ("kernel/ostype", 12, b"Linux\n"),
                ("kernel/osrelease", 13, b"6.1.0-nixvm\n"),
                ("kernel/hostname", 14, b"nixvm\n"),
                ("devices/system/cpu/online", 15, b"0\n"),
                ("devices/system/cpu/possible", 16, b"0\n"),
                ("devices/system/cpu/present", 17, b"0\n"),
            ];
            for &(path, inode, data) in files {
                t.insert(path, Node::File { inode, data });
            }
            t
        })
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
}

fn enoent() -> io::Error {
    io::Error::from_raw_os_error(2) // ENOENT
}
fn enotdir() -> io::Error {
    io::Error::from_raw_os_error(20) // ENOTDIR
}

impl MountFs for SysFs {
    fn stat(&mut self, rel: &str) -> Option<Attrs> {
        let node = Self::tree().get(rel)?;
        let (kind, mode, size) = match node {
            Node::Dir { .. } => (NodeKind::Dir, S_IFDIR | 0o755, 0),
            Node::File { data, .. } => (NodeKind::File, S_IFREG | 0o444, data.len() as u64),
        };
        Some(Attrs {
            kind,
            size,
            mode,
            uid: 0,
            gid: 0,
            mtime: 0,
            inode: node.inode(),
            nlink: 1,
            rdev: 0,
        })
    }

    fn read_at(&mut self, rel: &str, off: u64, buf: &mut [u8]) -> io::Result<usize> {
        match Self::tree().get(rel) {
            Some(Node::File { data, .. }) => {
                let off = off as usize;
                if off >= data.len() {
                    return Ok(0);
                }
                let n = buf.len().min(data.len() - off);
                buf[..n].copy_from_slice(&data[off..off + n]);
                Ok(n)
            }
            Some(Node::Dir { .. }) => Err(io::Error::from_raw_os_error(21)), // EISDIR
            None => Err(enoent()),
        }
    }

    fn readdir(&mut self, rel: &str) -> io::Result<Vec<DirEntry>> {
        let tree = Self::tree();
        match tree.get(rel) {
            Some(Node::Dir { .. }) => {}
            Some(_) => return Err(enotdir()),
            None => return Err(enoent()),
        }
        let mut out = Vec::new();
        for (&path, node) in tree {
            if path.is_empty() || Self::parent_of(path) != rel {
                continue;
            }
            let kind = match node {
                Node::Dir { .. } => NodeKind::Dir,
                Node::File { .. } => NodeKind::File,
            };
            out.push(DirEntry {
                name: Self::base_name(path).to_string(),
                kind,
                inode: node.inode(),
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn readdir_root_lists_top_level_dirs() {
        let mut fs = SysFs::new();
        let mut names: Vec<_> = fs.readdir("").unwrap().into_iter().map(|e| e.name).collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                "block", "bus", "class", "devices", "firmware", "fs", "kernel", "module",
            ]
        );
        // Every top-level entry is a directory.
        assert!(fs.readdir("").unwrap().iter().all(|e| e.kind == NodeKind::Dir));
    }

    #[test]
    fn read_kernel_ostype() {
        let mut fs = SysFs::new();
        let mut buf = [0u8; 32];
        let n = fs.read_at("kernel/ostype", 0, &mut buf).unwrap();
        assert_eq!(&buf[..n], b"Linux\n");
    }

    #[test]
    fn stat_dir_is_dir() {
        let mut fs = SysFs::new();
        let a = fs.stat("devices/system/cpu").unwrap();
        assert_eq!(a.kind, NodeKind::Dir);
        assert_eq!(a.mode, S_IFDIR | 0o755);
        assert_eq!(a.nlink, 1);
    }

    #[test]
    fn stat_file_is_read_only_reg() {
        let mut fs = SysFs::new();
        let a = fs.stat("kernel/osrelease").unwrap();
        assert_eq!(a.kind, NodeKind::File);
        assert_eq!(a.mode, S_IFREG | 0o444);
        assert_eq!(a.size, "6.1.0-nixvm\n".len() as u64);
    }

    #[test]
    fn nested_cpu_files_present() {
        let mut fs = SysFs::new();
        for name in ["online", "possible", "present"] {
            let path = format!("devices/system/cpu/{name}");
            let mut buf = [0u8; 8];
            let n = fs.read_at(&path, 0, &mut buf).unwrap();
            assert_eq!(&buf[..n], b"0\n");
        }
    }

    #[test]
    fn unknown_path_errors() {
        let mut fs = SysFs::new();
        assert!(fs.stat("nope").is_none());
        let mut buf = [0u8; 4];
        assert_eq!(
            fs.read_at("nope", 0, &mut buf).unwrap_err().raw_os_error(),
            Some(2)
        );
        assert_eq!(fs.readdir("nope").unwrap_err().raw_os_error(), Some(2));
    }

    #[test]
    fn readdir_on_file_is_enotdir() {
        let mut fs = SysFs::new();
        assert_eq!(
            fs.readdir("kernel/ostype").unwrap_err().raw_os_error(),
            Some(20)
        );
    }

    #[test]
    fn read_only_by_default() {
        assert!(SysFs::new().read_only());
    }
}
