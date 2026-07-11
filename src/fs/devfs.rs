//! Synthesized `/dev` filesystem.
//!
//! Provides the handful of pseudo-device nodes a POSIX userland expects
//! (`null`, `zero`, `full`, `random`, `urandom`, `tty`, `console`, `ptmx`,
//! `kmsg`) plus a handful of extra character devices (`loop-control`, `rtc0`,
//! `hpet`, `fuse`, `vga_arbiter`) and block devices (`loop0`..`loop7`,
//! `ram0`), without any real hardware behind them. The backend is otherwise
//! stateless apart from a small PRNG used to feed `/dev/random` and
//! `/dev/urandom`.
//!
//! Also provides the standard `/dev/fd`, `/dev/std{in,out,err}`, `/dev/rtc`
//! symlinks and the `/dev/pts`, `/dev/shm`, `/dev/net`, `/dev/mapper`,
//! `/dev/disk`, `/dev/char`, `/dev/block`, `/dev/hugepages` directories
//! (present and listable). Most of those directories are empty — nothing
//! allocates entries under them yet — except `/dev/net`, which holds
//! `/dev/net/tun`.
//!
//! The set of nodes is fixed, so there is no path map: `stat`/`read_at`/
//! `write_at` dispatch on the mount-relative name directly (the one nested
//! path, `net/tun`, is matched as a literal string like everything else). The
//! root (`""`) is the `/dev` directory itself and `readdir("")` enumerates the
//! device names, directories, and symlinks.

use std::io;
use std::time::{SystemTime, UNIX_EPOCH};

use super::{Attrs, DirEntry, MountFs, NodeKind};

/// Unix mode type bit for a character device.
const S_IFCHR: u32 = 0o020_000;
/// Unix mode type bit for a directory.
const S_IFDIR: u32 = 0o040_000;
/// Unix mode type bit for a block device.
const S_IFBLK: u32 = 0o060_000;
/// Unix mode type bit for a symbolic link.
const S_IFLNK: u32 = 0o120_000;
/// Mode reported for every char device node: char device, `rw` for everyone.
const CHR_MODE: u32 = S_IFCHR | 0o666;
/// Mode reported for every block device node: block device, `rw` for
/// everyone.
const BLK_MODE: u32 = S_IFBLK | 0o666;

/// Inode of the `/dev` root directory.
const ROOT_INODE: u64 = 1;

/// Encode a device number from `major`/`minor` using the Linux glibc `makedev`
/// layout (matching `crate::kernel::stat::makedev`). For the small in-range
/// values in [`DEVICES`] this is simply `major * 256 + minor`.
const fn makedev(major: u64, minor: u64) -> u64 {
    ((major & 0xfff) << 8) | (minor & 0xff) | ((minor & !0xff) << 12) | ((major & !0xfff) << 32)
}

/// The fixed char-device table at the `/dev` root: `(name, inode, major,
/// minor)`. Inodes are small distinct constants; the root reserves inode 1.
/// Majors/minors are the canonical Linux numbers so `/dev/null` reports
/// `1:3`, etc.
const DEVICES: &[(&str, u64, u64, u64)] = &[
    ("null", 2, 1, 3),
    ("zero", 3, 1, 5),
    ("full", 4, 1, 7),
    ("random", 5, 1, 8),
    ("urandom", 6, 1, 9),
    ("tty", 7, 5, 0),
    ("console", 8, 5, 1),
    ("ptmx", 9, 5, 2),
    ("kmsg", 10, 1, 11),
    // Misc char devices (major 10), plus the RTC (major 254).
    ("loop-control", 17, 10, 237),
    ("rtc0", 18, 254, 0),
    ("hpet", 19, 10, 228),
    ("fuse", 20, 10, 229),
    ("vga_arbiter", 21, 10, 63),
];

/// The one char device that lives under `/dev/net/` rather than the `/dev`
/// root: `(name, inode, major, minor)`, matched as the literal relative path
/// `"net/<name>"`.
const NET_DEVICES: &[(&str, u64, u64, u64)] = &[("tun", 22, 10, 200)];

/// The fixed block-device table at the `/dev` root: `(name, inode, major,
/// minor)`. `loop0`..`loop7` and `ram0` have no backing store; reads report
/// EOF and writes are discarded (see [`DevFs::read_at`]/[`DevFs::write_at`]).
const BLOCK_DEVICES: &[(&str, u64, u64, u64)] = &[
    ("loop0", 23, 7, 0),
    ("loop1", 24, 7, 1),
    ("loop2", 25, 7, 2),
    ("loop3", 26, 7, 3),
    ("loop4", 27, 7, 4),
    ("loop5", 28, 7, 5),
    ("loop6", 29, 7, 6),
    ("loop7", 30, 7, 7),
    ("ram0", 31, 1, 0),
];

/// Plain subdirectories under `/dev`: `(name, inode)`. Present and listable;
/// all but `net` (which holds `tun`, see [`NET_DEVICES`]) are always empty —
/// nothing allocates pty, shm-segment, or `by-id`-style entries yet.
const DIRS: &[(&str, u64)] = &[
    ("pts", 11),
    ("shm", 12),
    ("net", 32),
    ("mapper", 33),
    ("disk", 34),
    ("char", 35),
    ("block", 36),
    ("hugepages", 37),
];

/// Fixed symlinks under `/dev`: `(name, inode, target)`.
const SYMLINKS: &[(&str, u64, &str)] = &[
    ("fd", 13, "/proc/self/fd"),
    ("stdin", 14, "/proc/self/fd/0"),
    ("stdout", 15, "/proc/self/fd/1"),
    ("stderr", 16, "/proc/self/fd/2"),
    // Mirrors udev's usual `rtc -> rtc0` alias.
    ("rtc", 38, "rtc0"),
];

/// The synthesized `/dev` backend.
#[derive(Debug)]
pub struct DevFs {
    /// xorshift64 state for `/dev/random` and `/dev/urandom`. `0` means
    /// "not yet seeded"; seeding happens lazily on first use.
    rng: u64,
}

impl Default for DevFs {
    fn default() -> Self {
        Self::new()
    }
}

impl DevFs {
    #[must_use]
    pub fn new() -> Self {
        Self { rng: 0 }
    }

    /// Look up a root-level char-device name, returning its `(inode, rdev)`.
    fn lookup(name: &str) -> Option<(u64, u64)> {
        DEVICES
            .iter()
            .find(|(n, ..)| *n == name)
            .map(|(_, inode, major, minor)| (*inode, makedev(*major, *minor)))
    }

    /// Look up a root-level block-device name, returning its `(inode, rdev)`.
    fn block_lookup(name: &str) -> Option<(u64, u64)> {
        BLOCK_DEVICES
            .iter()
            .find(|(n, ..)| *n == name)
            .map(|(_, inode, major, minor)| (*inode, makedev(*major, *minor)))
    }

    /// Look up a char device nested under `/dev/net/`. `rel` must be the
    /// literal relative path `"net/<name>"`.
    fn net_lookup(rel: &str) -> Option<(u64, u64)> {
        let name = rel.strip_prefix("net/")?;
        NET_DEVICES
            .iter()
            .find(|(n, ..)| *n == name)
            .map(|(_, inode, major, minor)| (*inode, makedev(*major, *minor)))
    }

    /// Look up a plain subdirectory name, returning its inode.
    fn dir_lookup(name: &str) -> Option<u64> {
        DIRS.iter()
            .find(|(n, _)| *n == name)
            .map(|(_, inode)| *inode)
    }

    /// Look up a symlink name, returning its `(inode, target)`.
    fn symlink_lookup(name: &str) -> Option<(u64, &'static str)> {
        SYMLINKS
            .iter()
            .find(|(n, ..)| *n == name)
            .map(|(_, inode, target)| (*inode, *target))
    }

    /// Char-device attributes for a device with the given inode and device
    /// number.
    fn dev_attrs((inode, rdev): (u64, u64)) -> Attrs {
        Attrs {
            kind: NodeKind::CharDevice,
            size: 0,
            mode: CHR_MODE,
            uid: 0,
            gid: 0,
            mtime: 0,
            inode,
            nlink: 1,
            rdev,
        }
    }

    /// Block-device attributes for a device with the given inode and device
    /// number.
    fn blk_attrs((inode, rdev): (u64, u64)) -> Attrs {
        Attrs {
            kind: NodeKind::BlockDevice,
            size: 0,
            mode: BLK_MODE,
            uid: 0,
            gid: 0,
            mtime: 0,
            inode,
            nlink: 1,
            rdev,
        }
    }

    /// Next pseudo-random `u64` from the internal xorshift64 generator, seeding
    /// lazily from the wall clock on first use.
    fn next_u64(&mut self) -> u64 {
        if self.rng == 0 {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |d| d.as_nanos() as u64);
            // `| 1` guarantees a non-zero seed (xorshift is stuck at 0).
            self.rng = nanos | 1;
        }
        let mut x = self.rng;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng = x;
        x
    }

    /// Fill `buf` with pseudo-random bytes.
    fn fill_random(&mut self, buf: &mut [u8]) {
        for chunk in buf.chunks_mut(8) {
            let bytes = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
    }
}

fn enoent() -> io::Error {
    io::Error::from_raw_os_error(2) // ENOENT
}
fn enospc() -> io::Error {
    io::Error::from_raw_os_error(28) // ENOSPC
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
fn eagain() -> io::Error {
    io::Error::from_raw_os_error(11) // EAGAIN
}

impl MountFs for DevFs {
    fn read_only(&self) -> bool {
        false
    }

    fn stat(&mut self, rel: &str) -> Option<Attrs> {
        if rel.is_empty() {
            // The `/dev` directory itself.
            return Some(Attrs {
                kind: NodeKind::Dir,
                size: 0,
                mode: S_IFDIR | 0o755,
                uid: 0,
                gid: 0,
                mtime: 0,
                inode: ROOT_INODE,
                nlink: 1,
                rdev: 0,
            });
        }
        if let Some(attrs) = Self::lookup(rel).map(Self::dev_attrs) {
            return Some(attrs);
        }
        if let Some(attrs) = Self::block_lookup(rel).map(Self::blk_attrs) {
            return Some(attrs);
        }
        if let Some(attrs) = Self::net_lookup(rel).map(Self::dev_attrs) {
            return Some(attrs);
        }
        if let Some(inode) = Self::dir_lookup(rel) {
            return Some(Attrs {
                kind: NodeKind::Dir,
                size: 0,
                mode: S_IFDIR | 0o755,
                uid: 0,
                gid: 0,
                mtime: 0,
                inode,
                nlink: 1,
                rdev: 0,
            });
        }
        Self::symlink_lookup(rel).map(|(inode, target)| Attrs {
            kind: NodeKind::Symlink,
            size: target.len() as u64,
            mode: S_IFLNK | 0o777,
            uid: 0,
            gid: 0,
            mtime: 0,
            inode,
            nlink: 1,
            rdev: 0,
        })
    }

    fn read_at(&mut self, rel: &str, _off: u64, buf: &mut [u8]) -> io::Result<usize> {
        match rel {
            // Always EOF: no data ever arrives on these control-only nodes.
            "null" | "tty" | "console" | "kmsg" | "ptmx" | "loop-control" | "rtc0" | "hpet"
            | "fuse" | "vga_arbiter" => Ok(0),
            "zero" | "full" => {
                buf.fill(0);
                Ok(buf.len())
            }
            "random" | "urandom" => {
                self.fill_random(buf);
                Ok(buf.len())
            }
            // /dev/net/tun: open succeeds, but there is no queued traffic and
            // never will be — a nonblocking reader spins on EAGAIN forever.
            "net/tun" => Err(eagain()),
            // loop0..loop7, ram0: no backing store, so every read is
            // immediately EOF.
            _ if Self::block_lookup(rel).is_some() => Ok(0),
            _ if Self::dir_lookup(rel).is_some() => Err(eisdir()),
            _ if Self::symlink_lookup(rel).is_some() => Err(einval()),
            _ => Err(enoent()),
        }
    }

    fn readdir(&mut self, rel: &str) -> io::Result<Vec<DirEntry>> {
        match rel {
            "" => {
                let mut out: Vec<DirEntry> = DEVICES
                    .iter()
                    .map(|(name, inode, ..)| DirEntry {
                        name: (*name).to_string(),
                        kind: NodeKind::CharDevice,
                        inode: *inode,
                    })
                    .collect();
                out.extend(BLOCK_DEVICES.iter().map(|(name, inode, ..)| DirEntry {
                    name: (*name).to_string(),
                    kind: NodeKind::BlockDevice,
                    inode: *inode,
                }));
                out.extend(DIRS.iter().map(|(name, inode)| DirEntry {
                    name: (*name).to_string(),
                    kind: NodeKind::Dir,
                    inode: *inode,
                }));
                out.extend(SYMLINKS.iter().map(|(name, inode, _)| DirEntry {
                    name: (*name).to_string(),
                    kind: NodeKind::Symlink,
                    inode: *inode,
                }));
                Ok(out)
            }
            "net" => Ok(NET_DEVICES
                .iter()
                .map(|(name, inode, ..)| DirEntry {
                    name: (*name).to_string(),
                    kind: NodeKind::CharDevice,
                    inode: *inode,
                })
                .collect()),
            // `pts`, `shm`, `mapper`, `disk`, `char`, `block`, `hugepages`
            // exist and are listable, but nothing populates entries under
            // them yet.
            "pts" | "shm" | "mapper" | "disk" | "char" | "block" | "hugepages" => Ok(Vec::new()),
            _ if Self::lookup(rel).is_some()
                || Self::block_lookup(rel).is_some()
                || Self::net_lookup(rel).is_some()
                || Self::symlink_lookup(rel).is_some() =>
            {
                Err(enotdir())
            }
            _ => Err(enoent()),
        }
    }

    fn write_at(&mut self, rel: &str, _off: u64, buf: &[u8]) -> io::Result<usize> {
        match rel {
            // Discard writes, reporting the whole buffer consumed.
            "null" | "zero" | "random" | "urandom" | "tty" | "console" | "kmsg" | "ptmx"
            | "loop-control" | "rtc0" | "hpet" | "fuse" | "vga_arbiter" => Ok(buf.len()),
            // `/dev/full` is always out of space.
            "full" => Err(enospc()),
            // /dev/net/tun: nothing is ever transmitted (stub interface).
            "net/tun" => Ok(0),
            // loop0..loop7, ram0: no backing store, writes are discarded.
            _ if Self::block_lookup(rel).is_some() => Ok(buf.len()),
            _ if Self::dir_lookup(rel).is_some() => Err(eisdir()),
            _ => Err(enoent()),
        }
    }

    fn readlink(&mut self, rel: &str) -> io::Result<String> {
        if let Some((_, target)) = Self::symlink_lookup(rel) {
            return Ok(target.to_string());
        }
        if rel.is_empty()
            || Self::lookup(rel).is_some()
            || Self::block_lookup(rel).is_some()
            || Self::net_lookup(rel).is_some()
            || Self::dir_lookup(rel).is_some()
        {
            return Err(einval()); // exists, but is not a symlink
        }
        Err(enoent())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_reads_zeros() {
        let mut fs = DevFs::new();
        let mut buf = [0xffu8; 16];
        assert_eq!(fs.read_at("zero", 0, &mut buf).unwrap(), 16);
        assert!(buf.iter().all(|&b| b == 0));
    }

    #[test]
    fn null_reads_eof_and_accepts_writes() {
        let mut fs = DevFs::new();
        let mut buf = [0u8; 8];
        assert_eq!(fs.read_at("null", 0, &mut buf).unwrap(), 0);
        assert_eq!(fs.write_at("null", 0, b"discarded").unwrap(), 9);
    }

    #[test]
    fn full_write_errors_with_enospc() {
        let mut fs = DevFs::new();
        let err = fs.write_at("full", 0, b"x").unwrap_err();
        assert_eq!(err.raw_os_error(), Some(28));
        // Reads from /dev/full still give zeros.
        let mut buf = [0xffu8; 4];
        assert_eq!(fs.read_at("full", 0, &mut buf).unwrap(), 4);
        assert!(buf.iter().all(|&b| b == 0));
    }

    #[test]
    fn urandom_fills_non_zero_bytes() {
        let mut fs = DevFs::new();
        let mut buf = [0u8; 64];
        assert_eq!(fs.read_at("urandom", 0, &mut buf).unwrap(), 64);
        // Overwhelmingly likely at least one byte is non-zero.
        assert!(buf.iter().any(|&b| b != 0));
    }

    #[test]
    fn random_accepts_writes() {
        let mut fs = DevFs::new();
        assert_eq!(fs.write_at("random", 0, b"seed").unwrap(), 4);
        assert_eq!(fs.write_at("urandom", 0, b"seed").unwrap(), 4);
    }

    #[test]
    fn readdir_lists_all_nodes() {
        let mut fs = DevFs::new();
        let mut names: Vec<_> = fs
            .readdir("")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        names.sort();
        assert_eq!(
            names,
            vec![
                "block",
                "char",
                "console",
                "disk",
                "fd",
                "full",
                "fuse",
                "hpet",
                "hugepages",
                "kmsg",
                "loop-control",
                "loop0",
                "loop1",
                "loop2",
                "loop3",
                "loop4",
                "loop5",
                "loop6",
                "loop7",
                "mapper",
                "net",
                "null",
                "ptmx",
                "pts",
                "ram0",
                "random",
                "rtc",
                "rtc0",
                "shm",
                "stderr",
                "stdin",
                "stdout",
                "tty",
                "urandom",
                "vga_arbiter",
                "zero",
            ]
        );
        // Every char-device entry is reported as such.
        assert!(
            fs.readdir("")
                .unwrap()
                .iter()
                .filter(|e| DEVICES.iter().any(|(n, ..)| *n == e.name))
                .all(|e| e.kind == NodeKind::CharDevice)
        );
        // Every block-device entry is reported as such.
        assert!(
            fs.readdir("")
                .unwrap()
                .iter()
                .filter(|e| BLOCK_DEVICES.iter().any(|(n, ..)| *n == e.name))
                .all(|e| e.kind == NodeKind::BlockDevice)
        );
    }

    #[test]
    fn stat_reports_char_devices() {
        let mut fs = DevFs::new();
        let a = fs.stat("null").unwrap();
        assert_eq!(a.kind, NodeKind::CharDevice);
        assert_eq!(a.mode, S_IFCHR | 0o666);
        assert_eq!(a.size, 0);
        assert_eq!(a.nlink, 1);
        // Distinct inodes per device.
        assert_ne!(
            fs.stat("null").unwrap().inode,
            fs.stat("zero").unwrap().inode
        );
        // The root is a directory.
        assert_eq!(fs.stat("").unwrap().kind, NodeKind::Dir);
    }

    #[test]
    fn stat_reports_canonical_device_numbers() {
        let mut fs = DevFs::new();
        // /dev/null is the canonical char device 1:3.
        assert_eq!(fs.stat("null").unwrap().rdev, makedev(1, 3));
        assert_eq!(fs.stat("zero").unwrap().rdev, makedev(1, 5));
        assert_eq!(fs.stat("full").unwrap().rdev, makedev(1, 7));
        assert_eq!(fs.stat("random").unwrap().rdev, makedev(1, 8));
        assert_eq!(fs.stat("urandom").unwrap().rdev, makedev(1, 9));
        assert_eq!(fs.stat("tty").unwrap().rdev, makedev(5, 0));
        // Small values collapse to major*256 + minor.
        assert_eq!(fs.stat("null").unwrap().rdev, 256 + 3);
        // The /dev root directory carries no device number.
        assert_eq!(fs.stat("").unwrap().rdev, 0);
    }

    #[test]
    fn unknown_name_is_enoent() {
        let mut fs = DevFs::new();
        assert!(fs.stat("nope").is_none());
        let mut buf = [0u8; 4];
        assert_eq!(
            fs.read_at("nope", 0, &mut buf).unwrap_err().raw_os_error(),
            Some(2)
        );
        assert_eq!(
            fs.write_at("nope", 0, b"x").unwrap_err().raw_os_error(),
            Some(2)
        );
    }

    #[test]
    fn not_read_only() {
        assert!(!DevFs::new().read_only());
    }

    #[test]
    fn console_ptmx_kmsg_have_canonical_numbers_and_behave_like_devices() {
        let mut fs = DevFs::new();
        assert_eq!(fs.stat("console").unwrap().rdev, makedev(5, 1));
        assert_eq!(fs.stat("ptmx").unwrap().rdev, makedev(5, 2));
        assert_eq!(fs.stat("kmsg").unwrap().rdev, makedev(1, 11));
        for name in ["console", "ptmx", "kmsg"] {
            let a = fs.stat(name).unwrap();
            assert_eq!(a.kind, NodeKind::CharDevice);
            assert_eq!(a.mode, S_IFCHR | 0o666);
            // Always EOF on read, writes are swallowed.
            let mut buf = [0u8; 4];
            assert_eq!(fs.read_at(name, 0, &mut buf).unwrap(), 0);
            assert_eq!(fs.write_at(name, 0, b"x").unwrap(), 1);
        }
    }

    #[test]
    fn symlinks_resolve_to_proc_self_fd() {
        let mut fs = DevFs::new();
        assert_eq!(fs.readlink("fd").unwrap(), "/proc/self/fd");
        assert_eq!(fs.readlink("stdin").unwrap(), "/proc/self/fd/0");
        assert_eq!(fs.readlink("stdout").unwrap(), "/proc/self/fd/1");
        assert_eq!(fs.readlink("stderr").unwrap(), "/proc/self/fd/2");
        for name in ["fd", "stdin", "stdout", "stderr"] {
            let a = fs.stat(name).unwrap();
            assert_eq!(a.kind, NodeKind::Symlink);
            assert_eq!(a.mode, S_IFLNK | 0o777);
        }
    }

    #[test]
    fn pts_and_shm_dirs_present_and_listable() {
        let mut fs = DevFs::new();
        for name in ["pts", "shm"] {
            assert_eq!(fs.stat(name).unwrap().kind, NodeKind::Dir);
            assert_eq!(fs.readdir(name).unwrap().len(), 0);
        }
    }

    #[test]
    fn readlink_on_non_symlink_is_einval() {
        let mut fs = DevFs::new();
        assert_eq!(fs.readlink("null").unwrap_err().raw_os_error(), Some(22));
        assert_eq!(fs.readlink("pts").unwrap_err().raw_os_error(), Some(22));
        assert_eq!(fs.readlink("").unwrap_err().raw_os_error(), Some(22));
    }

    #[test]
    fn loop0_is_block_device_with_correct_rdev() {
        let mut fs = DevFs::new();
        let a = fs.stat("loop0").unwrap();
        assert_eq!(a.kind, NodeKind::BlockDevice);
        assert_eq!(a.mode, S_IFBLK | 0o666);
        assert_eq!(a.rdev, makedev(7, 0));
        // Reads report EOF, writes are discarded (no backing store).
        let mut buf = [0xffu8; 8];
        assert_eq!(fs.read_at("loop0", 0, &mut buf).unwrap(), 0);
        assert_eq!(fs.write_at("loop0", 0, b"data").unwrap(), 4);
    }

    #[test]
    fn loop_devices_and_ram0_have_canonical_numbers() {
        let mut fs = DevFs::new();
        for i in 0..8u64 {
            let name = format!("loop{i}");
            assert_eq!(fs.stat(&name).unwrap().rdev, makedev(7, i));
            assert_eq!(fs.stat(&name).unwrap().kind, NodeKind::BlockDevice);
        }
        assert_eq!(fs.stat("ram0").unwrap().rdev, makedev(1, 0));
        assert_eq!(fs.stat("ram0").unwrap().kind, NodeKind::BlockDevice);
    }

    #[test]
    fn net_tun_is_char_device_with_correct_rdev_and_stub_io() {
        let mut fs = DevFs::new();
        let a = fs.stat("net/tun").unwrap();
        assert_eq!(a.kind, NodeKind::CharDevice);
        assert_eq!(a.mode, S_IFCHR | 0o666);
        assert_eq!(a.rdev, makedev(10, 200));
        // Open succeeds (stat above), but read/write are stubs.
        let mut buf = [0u8; 8];
        assert_eq!(
            fs.read_at("net/tun", 0, &mut buf)
                .unwrap_err()
                .raw_os_error(),
            Some(11) // EAGAIN
        );
        assert_eq!(fs.write_at("net/tun", 0, b"packet").unwrap(), 0);
    }

    #[test]
    fn ls_dev_net_lists_tun() {
        let mut fs = DevFs::new();
        assert_eq!(fs.stat("net").unwrap().kind, NodeKind::Dir);
        let names: Vec<_> = fs
            .readdir("net")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert_eq!(names, vec!["tun"]);
    }

    #[test]
    fn new_char_devices_have_canonical_numbers() {
        let mut fs = DevFs::new();
        assert_eq!(fs.stat("loop-control").unwrap().rdev, makedev(10, 237));
        assert_eq!(fs.stat("rtc0").unwrap().rdev, makedev(254, 0));
        assert_eq!(fs.stat("hpet").unwrap().rdev, makedev(10, 228));
        assert_eq!(fs.stat("fuse").unwrap().rdev, makedev(10, 229));
        assert_eq!(fs.stat("vga_arbiter").unwrap().rdev, makedev(10, 63));
        for name in ["loop-control", "rtc0", "hpet", "fuse", "vga_arbiter"] {
            let a = fs.stat(name).unwrap();
            assert_eq!(a.kind, NodeKind::CharDevice);
            assert_eq!(a.mode, S_IFCHR | 0o666);
        }
    }

    #[test]
    fn rtc_symlink_resolves_to_rtc0() {
        let mut fs = DevFs::new();
        assert_eq!(fs.readlink("rtc").unwrap(), "rtc0");
        assert_eq!(fs.stat("rtc").unwrap().kind, NodeKind::Symlink);
        // The target itself is the real rtc0 character device.
        assert_eq!(fs.stat("rtc0").unwrap().kind, NodeKind::CharDevice);
    }

    #[test]
    fn new_dirs_present_and_listable() {
        let mut fs = DevFs::new();
        for name in ["net", "mapper", "disk", "char", "block", "hugepages"] {
            assert_eq!(fs.stat(name).unwrap().kind, NodeKind::Dir, "{name}");
        }
        // Only "net" has entries (tun); the rest are empty.
        for name in ["mapper", "disk", "char", "block", "hugepages"] {
            assert_eq!(fs.readdir(name).unwrap().len(), 0, "{name}");
        }
    }

    #[test]
    fn existing_symlinks_still_resolve() {
        // Re-verifies fd/stdin/stdout/stderr weren't disturbed by the new
        // nested "net/tun" node or the new "rtc" symlink.
        let mut fs = DevFs::new();
        assert_eq!(fs.readlink("fd").unwrap(), "/proc/self/fd");
        assert_eq!(fs.readlink("stdin").unwrap(), "/proc/self/fd/0");
        assert_eq!(fs.readlink("stdout").unwrap(), "/proc/self/fd/1");
        assert_eq!(fs.readlink("stderr").unwrap(), "/proc/self/fd/2");
    }
}
