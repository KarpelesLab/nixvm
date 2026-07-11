//! Synthesized `/proc` filesystem.
//!
//! Presents the small set of pseudo-files a Linux userland pokes at during
//! startup and everyday operation. There are two flavours of content:
//!
//! * **Static** files — `cpuinfo`, `meminfo`, `version`, `stat`, … — carry
//!   fixed, plausible bytes baked in at compile time.
//! * **Per-process** files under `self/` — `cmdline`, `status`, `maps`, the
//!   `exe` symlink, … — are rendered from a [`ProcData`] value. The backend
//!   ships sensible placeholders; the kernel injects the real running-process
//!   data later through [`ProcFs::set_self`], keeping this file free of any
//!   dependency on the process model.
//!
//! Layout is a small fixed tree (`""`, `self`, `sys`, `sys/kernel`); paths are
//! matched directly rather than through a stored map. The backend is read-only,
//! so every mutating method keeps the trait's `EROFS` default.

use std::io;

use super::{Attrs, DirEntry, MountFs, NodeKind};

/// Unix mode type bit for a directory.
const S_IFDIR: u32 = 0o040_000;
/// Unix mode type bit for a regular file.
const S_IFREG: u32 = 0o100_000;
/// Unix mode type bit for a symbolic link.
const S_IFLNK: u32 = 0o120_000;

/// Every path this backend knows, in a fixed order. The 1-based index into this
/// table is the node's inode, which keeps every inode distinct for free.
const PATHS: &[&str] = &[
    "", // root directory — inode 1
    "self",
    "sys",
    "sys/kernel",
    "cpuinfo",
    "meminfo",
    "version",
    "uptime",
    "loadavg",
    "stat",
    "filesystems",
    "mounts",
    "cmdline",
    "sys/kernel/ostype",
    "sys/kernel/osrelease",
    "sys/kernel/hostname",
    "sys/kernel/pid_max",
    "self/cmdline",
    "self/status",
    "self/stat",
    "self/comm",
    "self/maps",
    "self/exe",
];

/// Static top-level file names, in `readdir("")` order.
const ROOT_FILES: &[&str] = &[
    "cpuinfo",
    "meminfo",
    "version",
    "uptime",
    "loadavg",
    "stat",
    "filesystems",
    "mounts",
    "cmdline",
];

/// Per-process file names under `self/`. `exe` is a symlink; the rest are files.
const SELF_FILES: &[&str] = &["cmdline", "status", "stat", "comm", "maps", "exe"];

/// Tunables exposed under `sys/kernel/`.
const SYS_KERNEL_FILES: &[&str] = &["ostype", "osrelease", "hostname", "pid_max"];

// ---- static file bodies (each ends with a newline) ----

const CPUINFO: &str = "processor\t: 0\n\
BogoMIPS\t: 100.00\n\
Features\t: fp asimd\n\
CPU implementer\t: 0x41\n\
CPU architecture: 8\n\
CPU variant\t: 0x0\n\
CPU part\t: 0xd08\n\
CPU revision\t: 0\n";

const MEMINFO: &str = "MemTotal:        2048000 kB\n\
MemFree:         1024000 kB\n\
MemAvailable:    1536000 kB\n\
Buffers:               0 kB\n\
Cached:           512000 kB\n\
SwapTotal:             0 kB\n\
SwapFree:              0 kB\n";

const VERSION: &str = "Linux version 6.1.0-nixvm (nixvm@nixvm) (gcc) #1 SMP nixvm\n";

const UPTIME: &str = "0.00 0.00\n";

const LOADAVG: &str = "0.00 0.00 0.00 1/1 1\n";

const STAT: &str = "cpu  0 0 0 0 0 0 0 0 0 0\n\
cpu0 0 0 0 0 0 0 0 0 0 0\n\
intr 0\n\
ctxt 0\n\
btime 0\n\
processes 1\n\
procs_running 1\n\
procs_blocked 0\n";

const FILESYSTEMS: &str = "nodev\ttmpfs\n\
nodev\tproc\n\
nodev\tsysfs\n\
nodev\tdevtmpfs\n";

const MOUNTS: &str = "tmpfs / tmpfs rw 0 0\n\
proc /proc proc rw 0 0\n\
sysfs /sys sysfs rw 0 0\n";

const CMDLINE: &str = "\n";

const OSTYPE: &str = "Linux\n";
const OSRELEASE: &str = "6.1.0-nixvm\n";
const HOSTNAME: &str = "nixvm\n";
const PID_MAX: &str = "32768\n";

/// Running-process data backing the `self/` files.
///
/// The kernel builds one of these from the real process it is executing and
/// installs it with [`ProcFs::set_self`]; until then [`ProcData::default`]
/// supplies placeholders so the files still render.
#[derive(Debug, Clone)]
pub struct ProcData {
    /// Raw `argv` for `self/cmdline`, NUL-separated as the kernel presents it.
    pub cmdline: Vec<u8>,
    /// Absolute path the `self/exe` symlink resolves to.
    pub exe: String,
    /// Body of `self/maps`.
    pub maps: String,
    /// Short command name for `self/comm` (and the parenthesised field of
    /// `self/stat`).
    pub comm: String,
    /// The zeroth argument the process was launched with.
    pub argv0: String,
}

impl Default for ProcData {
    fn default() -> Self {
        Self {
            cmdline: Vec::new(),
            exe: "/bin/busybox".to_string(),
            maps: String::new(),
            comm: "nixvm".to_string(),
            argv0: "busybox".to_string(),
        }
    }
}

/// The synthesized `/proc` backend.
#[derive(Debug)]
pub struct ProcFs {
    data: ProcData,
}

impl Default for ProcFs {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcFs {
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: ProcData::default(),
        }
    }

    /// Install the running-process data backing the `self/` files.
    pub fn set_self(&mut self, data: ProcData) {
        self.data = data;
    }

    /// The rendered bytes of a readable file, or `None` if `rel` is not a
    /// regular file (a directory, the `self/exe` symlink, or unknown).
    fn content(&self, rel: &str) -> Option<Vec<u8>> {
        if let Some(bytes) = static_content(rel) {
            return Some(bytes.to_vec());
        }
        self.self_content(rel)
    }

    /// Render a per-process file. `self/exe` is deliberately excluded — it is a
    /// symlink, not a readable file.
    fn self_content(&self, rel: &str) -> Option<Vec<u8>> {
        let comm = &self.data.comm;
        let body = match rel {
            "self/cmdline" => return Some(self.data.cmdline.clone()),
            "self/maps" => return Some(self.data.maps.clone().into_bytes()),
            "self/comm" => format!("{comm}\n"),
            "self/stat" => {
                format!("1 ({comm}) R 0 1 1 0 -1 0 0 0 0 0 0 0 0 20 0 1 0 0\n")
            }
            "self/status" => format!(
                "Name:\t{comm}\n\
                 State:\tR (running)\n\
                 Pid:\t1\n\
                 PPid:\t0\n\
                 Uid:\t0\t0\t0\t0\n\
                 Gid:\t0\t0\t0\t0\n"
            ),
            _ => return None,
        };
        Some(body.into_bytes())
    }
}

/// Byte body of a static file, or `None` if `rel` is not one.
fn static_content(rel: &str) -> Option<&'static [u8]> {
    let text = match rel {
        "cpuinfo" => CPUINFO,
        "meminfo" => MEMINFO,
        "version" => VERSION,
        "uptime" => UPTIME,
        "loadavg" => LOADAVG,
        "stat" => STAT,
        "filesystems" => FILESYSTEMS,
        "mounts" => MOUNTS,
        "cmdline" => CMDLINE,
        "sys/kernel/ostype" => OSTYPE,
        "sys/kernel/osrelease" => OSRELEASE,
        "sys/kernel/hostname" => HOSTNAME,
        "sys/kernel/pid_max" => PID_MAX,
        _ => return None,
    };
    Some(text.as_bytes())
}

/// The inode for a known path (its 1-based position in [`PATHS`]).
fn inode_of(rel: &str) -> Option<u64> {
    PATHS.iter().position(|p| *p == rel).map(|i| i as u64 + 1)
}

/// Whether `rel` names one of the fixed directories.
fn is_dir(rel: &str) -> bool {
    matches!(rel, "" | "self" | "sys" | "sys/kernel")
}

/// Build a directory entry, looking the inode up from the full path.
fn entry(name: &str, path: &str, kind: NodeKind) -> DirEntry {
    DirEntry {
        name: name.to_string(),
        kind,
        inode: inode_of(path).unwrap_or(0),
    }
}

/// Copy `data[off..]` into `buf`, returning the byte count (0 at or past EOF).
fn read_slice(data: &[u8], off: u64, buf: &mut [u8]) -> usize {
    let off = off as usize;
    if off >= data.len() {
        return 0;
    }
    let n = buf.len().min(data.len() - off);
    buf[..n].copy_from_slice(&data[off..off + n]);
    n
}

fn enoent() -> io::Error {
    io::Error::from_raw_os_error(2) // ENOENT
}
fn eisdir() -> io::Error {
    io::Error::from_raw_os_error(21) // EISDIR
}
fn enotdir() -> io::Error {
    io::Error::from_raw_os_error(20) // ENOTDIR
}
fn einval() -> io::Error {
    io::Error::from_raw_os_error(22) // EINVAL
}

impl MountFs for ProcFs {
    fn stat(&mut self, rel: &str) -> Option<Attrs> {
        let inode = inode_of(rel)?;
        let (kind, mode, size) = if is_dir(rel) {
            (NodeKind::Dir, S_IFDIR | 0o555, 0)
        } else if rel == "self/exe" {
            (NodeKind::Symlink, S_IFLNK | 0o777, self.data.exe.len() as u64)
        } else {
            let data = self.content(rel)?;
            (NodeKind::File, S_IFREG | 0o444, data.len() as u64)
        };
        Some(Attrs {
            kind,
            size,
            mode,
            uid: 0,
            gid: 0,
            mtime: 0,
            inode,
            nlink: 1,
            rdev: 0,
        })
    }

    fn read_at(&mut self, rel: &str, off: u64, buf: &mut [u8]) -> io::Result<usize> {
        if is_dir(rel) {
            return Err(eisdir());
        }
        if rel == "self/exe" {
            return Err(einval()); // read on a symlink; use readlink
        }
        match self.content(rel) {
            Some(data) => Ok(read_slice(&data, off, buf)),
            None => Err(enoent()),
        }
    }

    fn readdir(&mut self, rel: &str) -> io::Result<Vec<DirEntry>> {
        match rel {
            "" => {
                let mut out: Vec<DirEntry> = ROOT_FILES
                    .iter()
                    .map(|n| entry(n, n, NodeKind::File))
                    .collect();
                out.push(entry("self", "self", NodeKind::Dir));
                out.push(entry("sys", "sys", NodeKind::Dir));
                Ok(out)
            }
            "self" => Ok(SELF_FILES
                .iter()
                .map(|n| {
                    let path = format!("self/{n}");
                    let kind = if *n == "exe" {
                        NodeKind::Symlink
                    } else {
                        NodeKind::File
                    };
                    entry(n, &path, kind)
                })
                .collect()),
            "sys" => Ok(vec![entry("kernel", "sys/kernel", NodeKind::Dir)]),
            "sys/kernel" => Ok(SYS_KERNEL_FILES
                .iter()
                .map(|n| {
                    let path = format!("sys/kernel/{n}");
                    entry(n, &path, NodeKind::File)
                })
                .collect()),
            _ if inode_of(rel).is_some() => Err(enotdir()),
            _ => Err(enoent()),
        }
    }

    fn readlink(&mut self, rel: &str) -> io::Result<String> {
        if rel == "self/exe" {
            return Ok(self.data.exe.clone());
        }
        if inode_of(rel).is_some() {
            return Err(einval()); // not a symlink
        }
        Err(enoent())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read an entire file by looping `read_at` to EOF.
    fn read_all(fs: &mut ProcFs, path: &str) -> Vec<u8> {
        let mut out = Vec::new();
        let mut off = 0u64;
        let mut buf = [0u8; 64];
        loop {
            let n = fs.read_at(path, off, &mut buf).unwrap();
            if n == 0 {
                break;
            }
            out.extend_from_slice(&buf[..n]);
            off += n as u64;
        }
        out
    }

    #[test]
    fn version_contains_linux() {
        let mut fs = ProcFs::new();
        let data = read_all(&mut fs, "version");
        assert!(String::from_utf8_lossy(&data).contains("Linux"));
    }

    #[test]
    fn root_readdir_lists_files_and_dirs() {
        let mut fs = ProcFs::new();
        let names: Vec<String> = fs.readdir("").unwrap().into_iter().map(|e| e.name).collect();
        assert!(names.contains(&"cpuinfo".to_string()));
        assert!(names.contains(&"self".to_string()));
        assert!(names.contains(&"sys".to_string()));
    }

    #[test]
    fn self_exe_is_a_symlink() {
        let mut fs = ProcFs::new();
        let attrs = fs.stat("self/exe").unwrap();
        assert_eq!(attrs.kind, NodeKind::Symlink);
        assert_eq!(attrs.mode, S_IFLNK | 0o777);
        // readlink resolves to the exe path.
        assert_eq!(fs.readlink("self/exe").unwrap(), "/bin/busybox");
    }

    #[test]
    fn set_self_changes_cmdline() {
        let mut fs = ProcFs::new();
        // Placeholder cmdline is empty.
        assert!(read_all(&mut fs, "self/cmdline").is_empty());
        fs.set_self(ProcData {
            cmdline: b"prog\0--flag\0".to_vec(),
            exe: "/usr/bin/prog".to_string(),
            maps: String::new(),
            comm: "prog".to_string(),
            argv0: "prog".to_string(),
        });
        assert_eq!(read_all(&mut fs, "self/cmdline"), b"prog\0--flag\0");
        assert_eq!(fs.readlink("self/exe").unwrap(), "/usr/bin/prog");
        let comm = read_all(&mut fs, "self/comm");
        assert_eq!(comm, b"prog\n");
    }

    #[test]
    fn directories_and_inodes() {
        let mut fs = ProcFs::new();
        assert_eq!(fs.stat("").unwrap().kind, NodeKind::Dir);
        assert_eq!(fs.stat("sys/kernel").unwrap().kind, NodeKind::Dir);
        assert_eq!(fs.stat("sys/kernel").unwrap().mode, S_IFDIR | 0o555);
        // Distinct inodes across entries.
        assert_ne!(
            fs.stat("cpuinfo").unwrap().inode,
            fs.stat("meminfo").unwrap().inode
        );
        // A regular file reports its rendered size.
        let meminfo = read_all(&mut fs, "meminfo");
        assert_eq!(fs.stat("meminfo").unwrap().size, meminfo.len() as u64);
        assert_eq!(fs.stat("meminfo").unwrap().mode, S_IFREG | 0o444);
    }

    #[test]
    fn unknown_path_errors() {
        let mut fs = ProcFs::new();
        assert!(fs.stat("nope").is_none());
        let mut buf = [0u8; 8];
        assert_eq!(
            fs.read_at("nope", 0, &mut buf).unwrap_err().raw_os_error(),
            Some(2)
        );
        assert_eq!(
            fs.readdir("nope").unwrap_err().raw_os_error(),
            Some(2)
        );
    }

    #[test]
    fn read_only_backend() {
        assert!(ProcFs::new().read_only());
    }
}
