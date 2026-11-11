//! The VM's disks and device numbers.
//!
//! Every filesystem whose files live on the host's disk — a host directory
//! mounted with [`super::Passthrough`], a filesystem image — is a block device
//! of the VM's, a virtio disk (`vda`, `vdb`, …, major [`VIRTBLK_MAJOR`]) the
//! size of what holds it. Its files report it as their `st_dev`, it's listed in
//! `/proc/partitions` and `/sys/block`, and the I/O to it is what the `io`
//! cgroup controller counts and limits. Every other filesystem (tmpfs, proc,
//! an overlay) gets an anonymous device number, `0:N`, as Linux gives one to
//! each superblock with no device behind it.

use std::fmt::Write as _;
use std::sync::{Arc, Mutex};

/// The major number of the VM's disks: the one virtio-blk is usually given.
pub const VIRTBLK_MAJOR: u32 = 254;
/// Minor numbers a disk takes up (a whole disk and its 15 partitions).
const MINORS_PER_DISK: u32 = 16;
/// The first minor number handed out for an anonymous device: below it are
/// the ones nixvm's pipes, sockets and other nodes without a filesystem use.
const FIRST_ANON_MINOR: u32 = 16;

/// A device number in the encoding `st_dev` uses (glibc's `makedev`).
#[must_use]
pub const fn makedev(major: u32, minor: u32) -> u64 {
    let (major, minor) = (major as u64, minor as u64);
    ((major & 0xfff) << 8) | ((major & !0xfff) << 32) | (minor & 0xff) | ((minor & !0xff) << 12)
}

/// The major number of an encoded device number.
#[must_use]
pub const fn major(dev: u64) -> u32 {
    (((dev >> 8) & 0xfff) | ((dev >> 32) & !0xfff)) as u32
}

/// The minor number of an encoded device number.
#[must_use]
pub const fn minor(dev: u64) -> u32 {
    ((dev & 0xff) | ((dev >> 12) & !0xff)) as u32
}

/// One of the VM's disks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Disk {
    /// Its device number.
    pub dev: u64,
    /// Its name: `vda`, `vdb`, ….
    pub name: String,
    /// How big it is, in bytes: what holds it on the host.
    pub bytes: u64,
    pub read_only: bool,
    /// The I/O done to it: reads and bytes read, writes and bytes written.
    pub reads: u64,
    pub read_bytes: u64,
    pub writes: u64,
    pub write_bytes: u64,
}

/// The VM's disks, and the anonymous device numbers handed out so far.
#[derive(Debug, Default)]
pub struct Disks {
    list: Vec<Disk>,
    anon: u32,
}

/// [`Disks`] as the mount table and the filesystems showing them share it.
pub type SharedDisks = Arc<Mutex<Disks>>;

impl Disks {
    /// Add a disk of `bytes` and return its device number.
    pub fn add(&mut self, bytes: u64, read_only: bool) -> u64 {
        let i = self.list.len() as u32;
        let dev = makedev(VIRTBLK_MAJOR, i * MINORS_PER_DISK);
        self.list.push(Disk {
            dev,
            name: disk_name(i),
            bytes,
            read_only,
            reads: 0,
            read_bytes: 0,
            writes: 0,
            write_bytes: 0,
        });
        dev
    }

    /// Count an I/O of `bytes` to disk `dev` (`/proc/diskstats`).
    pub fn account(&mut self, dev: u64, write: bool, bytes: u64) {
        if let Some(d) = self.list.iter_mut().find(|d| d.dev == dev) {
            if write {
                d.writes += 1;
                d.write_bytes += bytes;
            } else {
                d.reads += 1;
                d.read_bytes += bytes;
            }
        }
    }

    /// `/proc/partitions`.
    #[must_use]
    pub fn partitions(&self) -> String {
        let mut out = String::from("major minor  #blocks  name\n\n");
        for d in &self.list {
            let _ = writeln!(
                out,
                "{:4} {:7} {:10} {}",
                major(d.dev),
                minor(d.dev),
                d.bytes / 1024,
                d.name
            );
        }
        out
    }

    /// `/proc/diskstats`, in the 20-field form of Linux 5.5 and later.
    #[must_use]
    pub fn diskstats(&self) -> String {
        let mut out = String::new();
        for d in &self.list {
            let _ = writeln!(
                out,
                "{:4} {:7} {} {} 0 {} 0 {} 0 {} 0 0 0 0 0 0 0 0 0 0",
                major(d.dev),
                minor(d.dev),
                d.name,
                d.reads,
                d.read_bytes / 512,
                d.writes,
                d.write_bytes / 512
            );
        }
        out
    }

    /// A new anonymous device number.
    pub fn anon(&mut self) -> u64 {
        self.anon += 1;
        makedev(0, FIRST_ANON_MINOR + self.anon)
    }

    /// The disks, in the order they were added.
    #[must_use]
    pub fn list(&self) -> &[Disk] {
        &self.list
    }

    /// The disk with device number `major:minor`, if there's one.
    #[must_use]
    pub fn find(&self, major_no: u32, minor_no: u32) -> Option<&Disk> {
        self.list
            .iter()
            .find(|d| major(d.dev) == major_no && minor(d.dev) == minor_no)
    }
}

/// The name of disk number `i`: `vda`…`vdz`, then `vdaa`, `vdab`, … as Linux's
/// `sd_format_disk_name` goes on.
fn disk_name(i: u32) -> String {
    let mut suffix = Vec::new();
    let mut n = i + 1;
    while n > 0 {
        n -= 1;
        suffix.push(b'a' + (n % 26) as u8);
        n /= 26;
    }
    suffix.reverse();
    format!("vd{}", String::from_utf8(suffix).unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_numbers_round_trip_and_disks_are_named_like_linux() {
        for (ma, mi) in [(0, 0), (8, 1), (254, 16), (259, 300), (4095, 1 << 19)] {
            let d = makedev(ma, mi);
            assert_eq!((major(d), minor(d)), (ma, mi));
        }
        assert_eq!(makedev(1, 3), 259);
        let mut disks = Disks::default();
        let a = disks.add(1 << 30, false);
        let b = disks.add(1 << 20, true);
        assert_eq!((major(a), minor(a)), (254, 0));
        assert_eq!((major(b), minor(b)), (254, 16));
        assert_eq!(disks.find(254, 16).unwrap().name, "vdb");
        assert!(disks.find(254, 1).is_none(), "no partitions");
        assert_eq!(disk_name(25), "vdz");
        assert_eq!(disk_name(26), "vdaa");
        assert_eq!(disk_name(27), "vdab");
        let anon = disks.anon();
        assert_eq!((major(anon), minor(anon)), (0, 17));
    }
}
