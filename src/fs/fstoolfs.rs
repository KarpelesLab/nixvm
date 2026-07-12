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
//!
//! ## Error mapping
//!
//! fstool's [`fstool::Error`] is coarser than POSIX errno: in particular
//! `Error::InvalidArgument` covers "no such path component", "wrong entry
//! type" (e.g. reading a directory as a file), and genuine bad arguments all
//! at once — the crate has no dedicated not-found variant. `to_io` maps it
//! to `ENOENT` (the dominant case for a path-based lookup), and the read
//! paths ([`FsToolMount::read_at`], [`FsToolMount::readdir`],
//! [`FsToolMount::readlink`]) run one extra `getattr` on failure to
//! disambiguate the POSIX-significant cases (`EISDIR`, `ENOTDIR`, `EINVAL`)
//! from a plain miss — see `FsToolMount::classify`.

use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::Path;

use fstool::block::{BlockDevice, FileBackend, MemoryBackend};
use fstool::fs::{EntryKind, FileMeta, FileSource, Filesystem, OpenFlags, ext, squashfs};
use fstool::repack;

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

/// Map an fstool crate error to the nearest POSIX errno. `Io` passes the
/// underlying [`io::Error`] through unchanged (preserving its kind/raw errno);
/// every other variant is a fstool-level condition with no `io::Error`
/// behind it, so we pick the closest sane errno rather than propagating the
/// message as an opaque `Other` error (which previously hid corruption,
/// unsupported-feature, and not-found cases behind one undifferentiated
/// kind).
fn to_io(e: fstool::Error) -> io::Error {
    match e {
        fstool::Error::Io(e) => e,
        // The crate's catch-all for "path didn't resolve" as well as some
        // genuine bad-argument cases; ENOENT is the correct call for the
        // former and a defensible default for the latter (callers that care
        // about the EISDIR/ENOTDIR/EINVAL distinction use `classify` first).
        fstool::Error::InvalidArgument(_) => enoent(),
        // On-disk structure failed validation, or a read/write reached past
        // the device's logical extent: both mean the image itself can't be
        // trusted for this operation.
        fstool::Error::InvalidImage(_) | fstool::Error::OutOfBounds { .. } => eio(),
        // The format lacks a feature this build of fstool implements.
        fstool::Error::Unsupported(_) => enosys(),
        // Attempted mutation of a write-once (SquashFS/ISO) or streaming
        // (tar) format. `FsToolMount` already gates every mutator on
        // `self.read_only` before reaching fstool, so this is a defensive
        // fallback rather than the common path.
        fstool::Error::Streaming { .. } | fstool::Error::Immutable { .. } => erofs(),
    }
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
        // `Regular` and the catch-all `Unknown` (a type fstool couldn't
        // classify) both fold to a plain file — matching fstool's own
        // `FileAttrs::defaults_for` fallback for kinds it can't otherwise
        // describe.
        EntryKind::Regular | EntryKind::Unknown => NodeKind::File,
    }
}

// Unix `S_IFMT` type bits. fstool's `FileAttrs::mode` is permission-only —
// every backend (ext, squashfs) masks it to `0o7777` before returning it —
// so `Attrs::mode`, which per the sibling backends' convention carries both
// the type *and* permission bits, has to have the type bits OR'd in here.
const S_IFIFO: u32 = 0o010_000;
const S_IFCHR: u32 = 0o020_000;
const S_IFDIR: u32 = 0o040_000;
const S_IFBLK: u32 = 0o060_000;
const S_IFREG: u32 = 0o100_000;
const S_IFLNK: u32 = 0o120_000;
const S_IFSOCK: u32 = 0o140_000;

/// The `S_IFMT` bits for a [`NodeKind`] — see the module-level note on why
/// these have to be reattached to fstool's (permission-only) mode.
const fn type_bits(kind: NodeKind) -> u32 {
    match kind {
        NodeKind::File => S_IFREG,
        NodeKind::Dir => S_IFDIR,
        NodeKind::Symlink => S_IFLNK,
        NodeKind::CharDevice => S_IFCHR,
        NodeKind::BlockDevice => S_IFBLK,
        NodeKind::Fifo => S_IFIFO,
        NodeKind::Socket => S_IFSOCK,
    }
}

/// The node type an [`FsToolMount`] call site expects at a path, used by
/// `FsToolMount::classify` to turn an ambiguous fstool error into the
/// POSIX-correct errno.
#[derive(Clone, Copy)]
enum WantKind {
    /// The call needs a directory (`readdir`): a real entry of any other
    /// kind is `ENOTDIR`.
    Dir,
    /// The call needs a non-directory (`read_at`/`open_file_ro`): a real
    /// directory is `EISDIR`.
    NotDir,
    /// The call needs a symlink (`readlink`): a real entry of any other
    /// kind is `EINVAL`.
    Symlink,
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

    /// Open a **squashfs** image held entirely in memory, read-only. The image
    /// is copied into an in-memory block device — no filesystem required, so
    /// this works on wasm32.
    ///
    /// # Errors
    /// fstool error parsing the image.
    pub fn open_squashfs_bytes(bytes: &[u8]) -> io::Result<Self> {
        let mut mem = MemoryBackend::new(bytes.len() as u64);
        io::Write::write_all(&mut mem, bytes)?;
        let mut dev: Box<dyn BlockDevice> = Box::new(mem);
        let fs = squashfs::Squashfs::open(&mut *dev).map_err(to_io)?;
        Ok(Self {
            fs: Box::new(fs),
            dev,
            read_only: true,
        })
    }

    /// Build a **squashfs** in memory from an (uncompressed) tar archive and
    /// open it read-only — the browser demo's "unpack a rootfs image on demand"
    /// path. The tar is streamed straight into a squashfs writer over an
    /// in-memory block device (no temp files, no filesystem), then re-opened
    /// for reading. Used as the read-only lower of a copy-on-write overlay.
    ///
    /// # Errors
    /// fstool error building or re-opening the squashfs.
    pub fn from_tar(tar: &[u8]) -> io::Result<Self> {
        // A squashfs of a tar is smaller than the tar's uncompressed content;
        // size the device at the tar length plus slack for metadata.
        let cap = tar.len() as u64 + (8 << 20);
        let mut dev: Box<dyn BlockDevice> = Box::new(MemoryBackend::new(cap));
        let opts = squashfs::FormatOpts::default();
        let mut writer = squashfs::Squashfs::format(&mut *dev, &opts).map_err(to_io)?;
        {
            let mut sink = repack::FsSink::new(&mut writer, &mut *dev).lossy();
            let mut stream = fstool::fs::tar::stream::TarArchiveStream::new(io::Cursor::new(tar));
            repack::walk_stream(&mut stream, &mut sink).map_err(to_io)?;
        }
        Filesystem::flush(&mut writer, &mut *dev).map_err(to_io)?;
        drop(writer);
        let fs = squashfs::Squashfs::open(&mut *dev).map_err(to_io)?;
        Ok(Self {
            fs: Box::new(fs),
            dev,
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

    /// Disambiguate a failed operation at `abs_path` into a POSIX-correct
    /// errno: an extra `getattr` tells us whether the path actually exists
    /// and, if so, what kind of node it is, so a real directory read as a
    /// file comes back `EISDIR` (not `ENOENT`) and so on. When the extra
    /// `getattr` also fails, `path` genuinely doesn't resolve (or the image
    /// is unreadable), and the generic `to_io` mapping of the original
    /// error applies.
    fn classify(&mut self, abs_path: &str, want: WantKind, err: fstool::Error) -> io::Error {
        let Ok(a) = self.fs.getattr(&mut *self.dev, Path::new(abs_path)) else {
            return to_io(err);
        };
        match (want, a.kind == EntryKind::Dir, a.kind == EntryKind::Symlink) {
            (WantKind::Dir, true, _)
            | (WantKind::NotDir, false, _)
            | (WantKind::Symlink, _, true) => to_io(err),
            (WantKind::Dir, false, _) => enotdir(),
            (WantKind::NotDir, true, _) => eisdir(),
            (WantKind::Symlink, _, false) => einval(),
        }
    }
}

fn erofs() -> io::Error {
    io::Error::from_raw_os_error(30) // EROFS
}
fn enoent() -> io::Error {
    io::Error::from_raw_os_error(2) // ENOENT
}
fn eio() -> io::Error {
    io::Error::from_raw_os_error(5) // EIO
}
fn enosys() -> io::Error {
    io::Error::from_raw_os_error(38) // ENOSYS
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

impl MountFs for FsToolMount {
    fn read_only(&self) -> bool {
        self.read_only
    }

    fn stat(&mut self, rel: &str) -> Option<Attrs> {
        let a = self.fs.getattr(&mut *self.dev, Path::new(&abs(rel))).ok()?;
        let kind = map_kind(a.kind);
        Some(Attrs {
            kind,
            size: a.size,
            // fstool's `mode` is permission-only (see the module doc); OR in
            // the type bits so guest `stat()` sees a complete `st_mode`.
            mode: type_bits(kind) | (u32::from(a.mode) & 0o7_777),
            uid: a.uid,
            gid: a.gid,
            mtime: i64::from(a.mtime),
            inode: u64::from(a.inode),
            nlink: a.nlink,
            rdev: u64::from(a.rdev),
        })
    }

    fn read_at(&mut self, rel: &str, off: u64, buf: &mut [u8]) -> io::Result<usize> {
        let path = abs(rel);
        // The `Ok` arm returns eagerly so the open handle (which borrows
        // `self.fs`/`self.dev` for its lifetime) never needs to outlive this
        // match — only an owned `fstool::Error` does, letting the `Err` arm
        // call back into `self.classify` below without a borrow conflict.
        let open_err = match self.fs.open_file_ro(&mut *self.dev, Path::new(&path)) {
            Ok(mut h) => {
                // The handle clamps an out-of-range seek to `len()` (verified
                // against both backends), so a read at/past EOF falls
                // straight through to `Ok(0)` — no separate EOF check needed.
                h.seek(SeekFrom::Start(off))?;
                return h.read(buf);
            }
            Err(e) => e,
        };
        Err(self.classify(&path, WantKind::NotDir, open_err))
    }

    fn readdir(&mut self, rel: &str) -> io::Result<Vec<DirEntry>> {
        let path = abs(rel);
        let entries = match self.fs.list(&mut *self.dev, Path::new(&path)) {
            Ok(e) => e,
            Err(e) => return Err(self.classify(&path, WantKind::Dir, e)),
        };
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
        let path = abs(rel);
        let target = match self.fs.read_symlink(&mut *self.dev, Path::new(&path)) {
            Ok(t) => t,
            Err(e) => return Err(self.classify(&path, WantKind::Symlink, e)),
        };
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
    //! `FsToolMount::format_ram` builds a real in-memory ext4 filesystem
    //! through fstool's own writer, so the tests below exercise the actual
    //! `fstool` read/write/getattr/list/symlink code paths end-to-end — no
    //! fabricated image bytes.
    //!
    //! There's no squashfs-specific test here: fstool has no in-memory
    //! squashfs *writer* (it's write-once, built by `mksquashfs`-style
    //! tooling), so exercising `open_squashfs` needs a real image on disk.
    //! To test against one locally, point an image at this backend directly,
    //! e.g. in a scratch integration test:
    //! ```ignore
    //! let path = std::path::Path::new(&std::env::var("NIXVM_TEST_SQUASHFS_IMAGE").unwrap());
    //! let mut fs = FsToolMount::open_squashfs(path).unwrap();
    //! ```
    //! No such fixture is available in this checkout or CI, so it's not
    //! wired into the suite — the ext4-in-RAM tests below already cover the
    //! same `Filesystem`/`MountFs` bridge code (`stat`, `read_at`, `readdir`,
    //! `readlink`, error mapping) that `open_squashfs` would exercise.
    use super::*;

    #[test]
    fn type_bits_cover_every_kind() {
        assert_eq!(type_bits(NodeKind::File), 0o100_000);
        assert_eq!(type_bits(NodeKind::Dir), 0o040_000);
        assert_eq!(type_bits(NodeKind::Symlink), 0o120_000);
        assert_eq!(type_bits(NodeKind::CharDevice), 0o020_000);
        assert_eq!(type_bits(NodeKind::BlockDevice), 0o060_000);
        assert_eq!(type_bits(NodeKind::Fifo), 0o010_000);
        assert_eq!(type_bits(NodeKind::Socket), 0o140_000);
    }

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

    #[test]
    fn stat_reports_type_bits_for_file_and_dir() {
        let mut fs = FsToolMount::format_ram(8 << 20).unwrap();
        fs.create("f", 0o640).unwrap();
        fs.mkdir("d", 0o750).unwrap();

        let f = fs.stat("f").unwrap();
        assert_eq!(f.kind, NodeKind::File);
        assert_eq!(f.mode & 0o170_000, S_IFREG);
        assert_eq!(f.mode & 0o7_777, 0o640);

        let d = fs.stat("d").unwrap();
        assert_eq!(d.kind, NodeKind::Dir);
        assert_eq!(d.mode & 0o170_000, S_IFDIR);
        assert_eq!(d.mode & 0o7_777, 0o750);
    }

    #[test]
    fn symlink_readlink_and_stat_report_lnk_bits() {
        let mut fs = FsToolMount::format_ram(8 << 20).unwrap();
        fs.create("target", 0o644).unwrap();
        fs.symlink("target", "link").unwrap();

        assert_eq!(fs.readlink("link").unwrap(), "target");
        let a = fs.stat("link").unwrap();
        assert_eq!(a.kind, NodeKind::Symlink);
        assert_eq!(a.mode & 0o170_000, S_IFLNK);

        let names_and_kinds: Vec<_> = fs
            .readdir("")
            .unwrap()
            .into_iter()
            .map(|e| (e.name, e.kind))
            .collect();
        assert!(names_and_kinds.contains(&("link".to_string(), NodeKind::Symlink)));
        assert!(names_and_kinds.contains(&("target".to_string(), NodeKind::File)));
    }

    #[test]
    fn missing_path_is_enoent_not_panic() {
        let mut fs = FsToolMount::format_ram(8 << 20).unwrap();
        assert!(fs.stat("nope").is_none());

        let mut buf = [0u8; 4];
        assert_eq!(
            fs.read_at("nope", 0, &mut buf).unwrap_err().raw_os_error(),
            Some(2)
        );
        assert_eq!(fs.readdir("nope").unwrap_err().raw_os_error(), Some(2));
        assert_eq!(fs.readlink("nope").unwrap_err().raw_os_error(), Some(2));
    }

    #[test]
    fn reading_a_directory_as_a_file_is_eisdir() {
        let mut fs = FsToolMount::format_ram(8 << 20).unwrap();
        fs.mkdir("d", 0o755).unwrap();
        let mut buf = [0u8; 4];
        assert_eq!(
            fs.read_at("d", 0, &mut buf).unwrap_err().raw_os_error(),
            Some(21) // EISDIR
        );
    }

    #[test]
    fn listing_a_file_as_a_directory_is_enotdir() {
        let mut fs = FsToolMount::format_ram(8 << 20).unwrap();
        fs.create("f", 0o644).unwrap();
        assert_eq!(
            fs.readdir("f").unwrap_err().raw_os_error(),
            Some(20) // ENOTDIR
        );
    }

    #[test]
    fn readlink_on_non_symlink_is_einval() {
        let mut fs = FsToolMount::format_ram(8 << 20).unwrap();
        fs.create("f", 0o644).unwrap();
        assert_eq!(
            fs.readlink("f").unwrap_err().raw_os_error(),
            Some(22) // EINVAL
        );
    }

    #[test]
    fn read_past_eof_returns_zero() {
        let mut fs = FsToolMount::format_ram(8 << 20).unwrap();
        fs.create("f", 0o644).unwrap();
        fs.write_at("f", 0, b"hi").unwrap();
        let mut buf = [0u8; 8];
        assert_eq!(fs.read_at("f", 100, &mut buf).unwrap(), 0);
        // Exactly at EOF is also zero, not an error.
        assert_eq!(fs.read_at("f", 2, &mut buf).unwrap(), 0);
    }
}
