//! Synthesized `/dev` filesystem.
//!
//! Provides the handful of pseudo-device nodes a POSIX userland expects
//! (`null`, `zero`, `full`, `random`, `urandom`, `tty`) without any real
//! hardware behind them. Every node is a character device (`S_IFCHR`); the
//! backend is otherwise stateless apart from a small PRNG used to feed
//! `/dev/random` and `/dev/urandom`.
//!
//! The set of nodes is fixed, so there is no path map: `stat`/`read_at`/
//! `write_at` dispatch on the mount-relative name directly. The root (`""`) is
//! the `/dev` directory itself and `readdir("")` enumerates the device names.

use std::io;
use std::time::{SystemTime, UNIX_EPOCH};

use super::{Attrs, DirEntry, MountFs, NodeKind};

/// Unix mode type bit for a character device.
const S_IFCHR: u32 = 0o020_000;
/// Unix mode type bit for a directory.
const S_IFDIR: u32 = 0o040_000;
/// Mode reported for every device node: char device, `rw` for everyone.
const CHR_MODE: u32 = S_IFCHR | 0o666;

/// Inode of the `/dev` root directory.
const ROOT_INODE: u64 = 1;

/// Encode a device number from `major`/`minor` using the Linux glibc `makedev`
/// layout (matching `crate::kernel::stat::makedev`). For the small in-range
/// values in [`DEVICES`] this is simply `major * 256 + minor`.
const fn makedev(major: u64, minor: u64) -> u64 {
    ((major & 0xfff) << 8) | (minor & 0xff) | ((minor & !0xff) << 12) | ((major & !0xfff) << 32)
}

/// The fixed device table: `(name, inode, major, minor)`. Inodes are small
/// distinct constants; the root reserves inode 1. Majors/minors are the
/// canonical Linux numbers so `/dev/null` reports `1:3`, etc.
const DEVICES: &[(&str, u64, u64, u64)] = &[
    ("null", 2, 1, 3),
    ("zero", 3, 1, 5),
    ("full", 4, 1, 7),
    ("random", 5, 1, 8),
    ("urandom", 6, 1, 9),
    ("tty", 7, 5, 0),
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

    /// Look up a device name, returning its `(inode, rdev)`.
    fn lookup(name: &str) -> Option<(u64, u64)> {
        DEVICES
            .iter()
            .find(|(n, ..)| *n == name)
            .map(|(_, inode, major, minor)| (*inode, makedev(*major, *minor)))
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
        Self::lookup(rel).map(Self::dev_attrs)
    }

    fn read_at(&mut self, rel: &str, _off: u64, buf: &mut [u8]) -> io::Result<usize> {
        match rel {
            "null" | "tty" => Ok(0), // always EOF
            "zero" | "full" => {
                buf.fill(0);
                Ok(buf.len())
            }
            "random" | "urandom" => {
                self.fill_random(buf);
                Ok(buf.len())
            }
            _ => Err(enoent()),
        }
    }

    fn readdir(&mut self, rel: &str) -> io::Result<Vec<DirEntry>> {
        if !rel.is_empty() {
            // Device nodes are not directories.
            return Err(io::Error::from_raw_os_error(20)); // ENOTDIR
        }
        Ok(DEVICES
            .iter()
            .map(|(name, inode, ..)| DirEntry {
                name: (*name).to_string(),
                kind: NodeKind::CharDevice,
                inode: *inode,
            })
            .collect())
    }

    fn write_at(&mut self, rel: &str, _off: u64, buf: &[u8]) -> io::Result<usize> {
        match rel {
            // Discard writes, reporting the whole buffer consumed.
            "null" | "zero" | "random" | "urandom" | "tty" => Ok(buf.len()),
            // `/dev/full` is always out of space.
            "full" => Err(enospc()),
            _ => Err(enoent()),
        }
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
            vec!["full", "null", "random", "tty", "urandom", "zero"]
        );
        // Every entry is a character device.
        assert!(
            fs.readdir("")
                .unwrap()
                .iter()
                .all(|e| e.kind == NodeKind::CharDevice)
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
}
