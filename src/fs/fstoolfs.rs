//! A real on-disk filesystem mounted into the table, backed by the `fstool`
//! crate. Built only with the `fstool` cargo feature.
//!
//! The primary use is the **read-only squashfs root**: many nixvm instances can
//! mount the same immutable image cheaply, layering a per-instance writable
//! upper (RAM tmpfs, or an ext4 image) on top via [`Overlay`](super::Overlay).
//! An in-memory ext4 constructor ([`FsToolMount::format_ram`]) backs the
//! writable-overlay option and the hermetic tests.
//!
//! fstool's open handles borrow the filesystem *and* the device for their
//! lifetime, so every [`MountFs`] operation opens, does the I/O, and drops the
//! handle within the call.

use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

use fstool::block::{BlockDevice, FileBackend, MemoryBackend};
use fstool::fs::{EntryKind, FileMeta, FileSource, Filesystem, OpenFlags, ext, squashfs};

use super::{Attrs, DirEntry, MountFs, NodeKind};

/// A mounted real filesystem: an fstool [`Filesystem`] over a block device.
pub struct FsToolMount {
    fs: Box<dyn Filesystem>,
    dev: Box<dyn BlockDevice>,
    read_only: bool,
}

impl std::fmt::Debug for FsToolMount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FsToolMount(read_only={})", self.read_only)
    }
}

#[allow(clippy::needless_pass_by_value)] // used as a `map_err` function pointer
fn to_io(e: fstool::Error) -> io::Error {
    io::Error::other(e.to_string())
}

/// fstool paths are absolute; the mount table hands us relative paths.
fn abs(rel: &str) -> String {
    if rel.is_empty() {
        "/".to_string()
    } else {
        format!("/{rel}")
    }
}

fn map_kind(k: EntryKind) -> NodeKind {
    match k {
        EntryKind::Dir => NodeKind::Dir,
        EntryKind::Symlink => NodeKind::Symlink,
        EntryKind::Char => NodeKind::CharDevice,
        EntryKind::Block => NodeKind::BlockDevice,
        EntryKind::Fifo => NodeKind::Fifo,
        EntryKind::Socket => NodeKind::Socket,
        _ => NodeKind::File,
    }
}

impl FsToolMount {
    /// Open a **squashfs** image read-only — the shared, immutable guest root.
    ///
    /// # Errors
    /// I/O error opening the file, or an fstool error parsing the image.
    pub fn open_squashfs(path: &Path) -> io::Result<Self> {
        let mut dev = FileBackend::open_read_only(path).map_err(to_io)?;
        let fs = squashfs::Squashfs::open(&mut dev).map_err(to_io)?;
        Ok(Self {
            fs: Box::new(fs),
            dev: Box::new(dev),
            read_only: true,
        })
    }

    /// Format a fresh writable **ext4** filesystem in RAM (`size` bytes, min
    /// 4 MiB) — the writable-overlay-as-ext option and the test fixture.
    ///
    /// # Errors
    /// fstool error formatting the filesystem.
    pub fn format_ram(size: u64) -> io::Result<Self> {
        let size = size.max(4 << 20);
        let mut dev = MemoryBackend::new(size);
        let block_size = 4096u32;
        let blocks_count = (size / u64::from(block_size)).max(64) as u32;
        let opts = ext::FormatOpts {
            kind: ext::FsKind::Ext4,
            block_size,
            blocks_count,
            inodes_count: (blocks_count / 4).max(16),
            ..Default::default()
        };
        let fs = ext::Ext::format_with(&mut dev, &opts).map_err(to_io)?;
        Ok(Self {
            fs: Box::new(fs),
            dev: Box::new(dev),
            read_only: false,
        })
    }
}

fn erofs() -> io::Error {
    io::Error::from_raw_os_error(30)
}

impl MountFs for FsToolMount {
    fn read_only(&self) -> bool {
        self.read_only
    }

    fn stat(&mut self, rel: &str) -> Option<Attrs> {
        let a = self.fs.getattr(&mut *self.dev, Path::new(&abs(rel))).ok()?;
        Some(Attrs {
            kind: map_kind(a.kind),
            size: a.size,
            mode: u32::from(a.mode),
            uid: a.uid,
            gid: a.gid,
            mtime: i64::from(a.mtime),
            inode: u64::from(a.inode),
            nlink: a.nlink,
        })
    }

    fn read_at(&mut self, rel: &str, off: u64, buf: &mut [u8]) -> io::Result<usize> {
        let mut h = self
            .fs
            .open_file_ro(&mut *self.dev, Path::new(&abs(rel)))
            .map_err(to_io)?;
        h.seek(SeekFrom::Start(off))?;
        h.read(buf)
    }

    fn readdir(&mut self, rel: &str) -> io::Result<Vec<DirEntry>> {
        let entries = self
            .fs
            .list(&mut *self.dev, Path::new(&abs(rel)))
            .map_err(to_io)?;
        Ok(entries
            .into_iter()
            .map(|e| DirEntry {
                name: e.name,
                kind: map_kind(e.kind),
                inode: u64::from(e.inode),
            })
            .collect())
    }

    fn write_at(&mut self, rel: &str, off: u64, buf: &[u8]) -> io::Result<usize> {
        if self.read_only {
            return Err(erofs());
        }
        let mut h = self
            .fs
            .open_file_rw(
                &mut *self.dev,
                Path::new(&abs(rel)),
                OpenFlags {
                    create: false,
                    truncate: false,
                    append: false,
                },
                None,
            )
            .map_err(to_io)?;
        h.seek(SeekFrom::Start(off))?;
        h.write(buf)
    }

    fn create(&mut self, rel: &str, mode: u32) -> io::Result<()> {
        if self.read_only {
            return Err(erofs());
        }
        self.fs
            .create_file(
                &mut *self.dev,
                Path::new(&abs(rel)),
                FileSource::Zero(0),
                FileMeta::with_mode(mode as u16),
            )
            .map_err(to_io)
    }

    fn mkdir(&mut self, rel: &str, mode: u32) -> io::Result<()> {
        if self.read_only {
            return Err(erofs());
        }
        self.fs
            .create_dir(
                &mut *self.dev,
                Path::new(&abs(rel)),
                FileMeta::with_mode(mode as u16),
            )
            .map_err(to_io)
    }

    fn unlink(&mut self, rel: &str) -> io::Result<()> {
        if self.read_only {
            return Err(erofs());
        }
        self.fs
            .remove(&mut *self.dev, Path::new(&abs(rel)))
            .map_err(to_io)
    }

    fn rmdir(&mut self, rel: &str) -> io::Result<()> {
        self.unlink(rel)
    }

    fn truncate(&mut self, rel: &str, len: u64) -> io::Result<()> {
        if self.read_only {
            return Err(erofs());
        }
        self.fs
            .truncate(&mut *self.dev, Path::new(&abs(rel)), len)
            .map_err(to_io)
    }

    fn readlink(&mut self, rel: &str) -> io::Result<String> {
        let target = self
            .fs
            .read_symlink(&mut *self.dev, Path::new(&abs(rel)))
            .map_err(to_io)?;
        Ok(target.to_string_lossy().into_owned())
    }

    fn symlink(&mut self, target: &str, linkpath: &str) -> io::Result<()> {
        if self.read_only {
            return Err(erofs());
        }
        self.fs
            .create_symlink(
                &mut *self.dev,
                Path::new(&abs(linkpath)),
                Path::new(target),
                FileMeta::with_mode(0o777),
            )
            .map_err(to_io)
    }

    fn rename(&mut self, from: &str, to: &str) -> io::Result<()> {
        if self.read_only {
            return Err(erofs());
        }
        self.fs
            .rename(&mut *self.dev, Path::new(&abs(from)), Path::new(&abs(to)))
            .map_err(to_io)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ram_ext4_create_write_read() {
        let mut fs = FsToolMount::format_ram(8 << 20).unwrap();
        assert!(!fs.read_only());
        fs.create("hello", 0o644).unwrap();
        assert_eq!(fs.write_at("hello", 0, b"fstool!").unwrap(), 7);
        let mut buf = [0u8; 7];
        assert_eq!(fs.read_at("hello", 0, &mut buf).unwrap(), 7);
        assert_eq!(&buf, b"fstool!");
        assert_eq!(fs.stat("hello").unwrap().size, 7);

        fs.mkdir("d", 0o755).unwrap();
        let names: Vec<_> = fs
            .readdir("")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert!(names.contains(&"hello".to_string()) && names.contains(&"d".to_string()));
    }
}
