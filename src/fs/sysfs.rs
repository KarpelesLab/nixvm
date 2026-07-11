//! Synthesized `/sys` (sysfs) filesystem.
//!
//! Presents the skeleton of a Linux `sysfs` tree — the top-level directories a
//! userland expects (`kernel`, `devices`, `class`, `fs`, `block`, `bus`,
//! `module`, `firmware`) plus a handful of fixed-content attribute files
//! (`kernel/ostype`, `devices/system/cpu/online`, …). Nothing here reflects
//! real hardware; the tree is entirely static and read-only, so every mutating
//! method falls through to the `EROFS` default.
//!
//! The CPU topology (`devices/system/cpu/cpu0`, `cpu1`, …) is sized from
//! [`std::thread::available_parallelism`] at first access, so it matches the
//! host's reported core count (the guest's "nproc"). Everything else is a
//! fixed, plausible placeholder.
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
    File { inode: u64, data: Vec<u8> },
}

impl Node {
    fn inode(&self) -> u64 {
        match self {
            Node::Dir { inode } | Node::File { inode, .. } => *inode,
        }
    }
}

/// Number of logical CPUs to report, mirroring what the guest's `nproc` would
/// see. Falls back to `1` if the host declines to say.
fn nproc() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get)
}

/// Render a CPU index set the way `sysfs` bitmap-list files do: `"0\n"` for a
/// single CPU, `"0-N\n"` for a contiguous range starting at 0.
fn cpu_range(ncpu: usize) -> String {
    if ncpu <= 1 {
        "0\n".to_string()
    } else {
        format!("0-{}\n", ncpu - 1)
    }
}

/// Incrementally builds the static tree, handing out distinct small inodes in
/// insertion order.
struct Builder {
    map: BTreeMap<String, Node>,
    next_inode: u64,
}

impl Builder {
    fn new() -> Self {
        Self {
            map: BTreeMap::new(),
            next_inode: 0,
        }
    }

    fn dir(&mut self, path: impl Into<String>) {
        self.next_inode += 1;
        self.map.insert(path.into(), Node::Dir { inode: self.next_inode });
    }

    fn file(&mut self, path: impl Into<String>, data: impl Into<Vec<u8>>) {
        self.next_inode += 1;
        self.map.insert(
            path.into(),
            Node::File {
                inode: self.next_inode,
                data: data.into(),
            },
        );
    }

    fn build(self) -> BTreeMap<String, Node> {
        self.map
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
    fn tree() -> &'static BTreeMap<String, Node> {
        static TREE: OnceLock<BTreeMap<String, Node>> = OnceLock::new();
        TREE.get_or_init(Self::build_tree)
    }

    fn build_tree() -> BTreeMap<String, Node> {
        let ncpu = nproc();
        let online = cpu_range(ncpu);
        let mut b = Builder::new();

        // ---- top level ----
        b.dir("");
        b.dir("kernel");
        b.dir("devices");
        b.dir("class");
        b.dir("fs");
        b.dir("block");
        b.dir("bus");
        b.dir("module");
        b.dir("firmware");

        // ---- kernel/ ----
        b.file("kernel/ostype", *b"Linux\n");
        b.file("kernel/osrelease", *b"6.1.0-nixvm\n");
        b.file("kernel/hostname", *b"nixvm\n");
        b.dir("kernel/mm");
        b.dir("kernel/mm/transparent_hugepage");
        b.file(
            "kernel/mm/transparent_hugepage/enabled",
            *b"always [madvise] never\n",
        );

        // ---- devices/system/cpu ----
        b.dir("devices/system");
        b.dir("devices/system/cpu");
        b.file("devices/system/cpu/online", online.clone());
        b.file("devices/system/cpu/possible", online.clone());
        b.file("devices/system/cpu/present", online.clone());
        // The compile-time max CPU index the (synthetic) kernel was built
        // for; real kernels report a value well above the online count.
        b.file("devices/system/cpu/kernel_max", *b"255\n");
        for i in 0..ncpu {
            let dir = format!("devices/system/cpu/cpu{i}");
            let online_path = format!("{dir}/online");
            b.dir(dir);
            b.file(online_path, *b"1\n");
        }

        // ---- devices/system/node (single-node NUMA topology) ----
        b.dir("devices/system/node");
        b.dir("devices/system/node/node0");
        b.file("devices/system/node/node0/cpulist", online);
        b.file(
            "devices/system/node/node0/meminfo",
            *b"Node 0 MemTotal:        2048000 kB\n\
               Node 0 MemFree:         1024000 kB\n\
               Node 0 MemUsed:         1024000 kB\n",
        );

        // ---- class/ (sparse — just enough dirs to exist and be listable) ----
        b.dir("class/net");
        b.dir("class/tty");
        b.dir("class/mem");

        // ---- fs/cgroup (minimal presence) ----
        b.dir("fs/cgroup");

        b.build()
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
        for (path, node) in tree {
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
    fn cpu_online_matches_nproc() {
        let mut fs = SysFs::new();
        let ncpu = nproc();
        let mut buf = [0u8; 64];
        for name in ["online", "possible", "present"] {
            let path = format!("devices/system/cpu/{name}");
            let n = fs.read_at(&path, 0, &mut buf).unwrap();
            let text = std::str::from_utf8(&buf[..n]).unwrap();
            assert_eq!(text, cpu_range(ncpu));
        }
    }

    #[test]
    fn per_cpu_dirs_match_nproc() {
        let mut fs = SysFs::new();
        let ncpu = nproc();
        let mut names: Vec<_> = fs
            .readdir("devices/system/cpu")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        names.sort();
        let mut expected: Vec<String> = (0..ncpu).map(|i| format!("cpu{i}")).collect();
        expected.extend(
            ["kernel_max", "online", "possible", "present"]
                .iter()
                .map(|s| (*s).to_string()),
        );
        expected.sort();
        assert_eq!(names, expected);

        // Every per-cpu dir has an "online" file reporting "1".
        for i in 0..ncpu {
            let path = format!("devices/system/cpu/cpu{i}/online");
            let mut buf = [0u8; 4];
            let n = fs.read_at(&path, 0, &mut buf).unwrap();
            assert_eq!(&buf[..n], b"1\n");
        }
    }

    #[test]
    fn ls_sys_devices_system_cpu_lists_expected_entries() {
        let mut fs = SysFs::new();
        let names: Vec<_> = fs
            .readdir("devices/system")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert!(names.contains(&"cpu".to_string()));
        assert!(names.contains(&"node".to_string()));
    }

    #[test]
    fn node0_present_with_meminfo_and_cpulist() {
        let mut fs = SysFs::new();
        let a = fs.stat("devices/system/node/node0").unwrap();
        assert_eq!(a.kind, NodeKind::Dir);
        let mut buf = [0u8; 256];
        let n = fs
            .read_at("devices/system/node/node0/meminfo", 0, &mut buf)
            .unwrap();
        assert!(String::from_utf8_lossy(&buf[..n]).contains("Node 0 MemTotal"));
    }

    #[test]
    fn class_and_block_and_cgroup_present() {
        let mut fs = SysFs::new();
        for path in ["class", "class/net", "class/tty", "class/mem", "block", "fs/cgroup"] {
            assert_eq!(fs.stat(path).unwrap().kind, NodeKind::Dir, "{path}");
        }
    }

    #[test]
    fn transparent_hugepage_enabled_present() {
        let mut fs = SysFs::new();
        let mut buf = [0u8; 64];
        let n = fs
            .read_at("kernel/mm/transparent_hugepage/enabled", 0, &mut buf)
            .unwrap();
        assert!(String::from_utf8_lossy(&buf[..n]).contains("madvise"));
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
