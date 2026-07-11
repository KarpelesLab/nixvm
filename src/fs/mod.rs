//! The nixvm virtual filesystem.
//!
//! A [`MountTable`] resolves an absolute guest path to `(backend, relative
//! path)` by longest-prefix match, then delegates to a [`MountFs`] backend. The
//! kernel's file syscalls (`openat`, `read`, `getdents64`, `stat`, …) are
//! written entirely against these traits.
//!
//! Backends (added over Phases 4, 7):
//!
//! * **squashfs** — the read-only Alpine root image.
//! * **overlay** — copy-on-write: a writable upper (tmpfs) over a read-only
//!   lower (squashfs), giving `/` its mutable-but-ephemeral behavior.
//! * **passthrough** — maps a host directory (the cwd) to `/work`, read-write.
//! * **tmpfs** — in-memory, for `/tmp` and overlay uppers.
//! * **procfs / devfs** — synthesized `/proc`, `/sys`, `/dev`.
//!
//! The trait is *path + offset* rather than open-handle based, so read-only
//! backends implement just three methods and everything else defaults to
//! `EROFS`. (Pattern proven in univdreams' `fsmount`.)

use std::io;

#[cfg(feature = "fstool")]
pub mod fstoolfs;
pub mod devfs;
pub mod mount;
pub mod overlay;
#[cfg(unix)]
pub mod passthrough;
pub mod procfs;
pub mod sysfs;
pub mod tmpfs;

#[cfg(feature = "fstool")]
pub use fstoolfs::FsToolMount;
pub use devfs::DevFs;
pub use mount::MountTable;
pub use overlay::Overlay;
#[cfg(unix)]
pub use passthrough::Passthrough;
pub use procfs::ProcFs;
pub use sysfs::SysFs;
pub use tmpfs::TmpFs;

/// The kind of a filesystem node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeKind {
    File,
    Dir,
    Symlink,
    CharDevice,
    BlockDevice,
    Fifo,
    Socket,
}

/// Metadata for one node (the subset the kernel maps into guest `stat`).
#[derive(Debug, Clone)]
pub struct Attrs {
    pub kind: NodeKind,
    pub size: u64,
    /// Unix mode bits (permission + type).
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
    pub mtime: i64,
    pub inode: u64,
    pub nlink: u32,
}

/// One entry returned by [`MountFs::readdir`].
#[derive(Debug, Clone)]
pub struct DirEntry {
    pub name: String,
    pub kind: NodeKind,
    pub inode: u64,
}

/// A filesystem backend mounted at some point in the [`MountTable`].
///
/// Only `stat`, `read_at`, and `readdir` are required; every mutating method
/// defaults to `EROFS`, so a read-only backend (like squashfs) implements the
/// three required methods and nothing else.
pub trait MountFs: std::fmt::Debug {
    // ---- required (read side) ----
    fn stat(&mut self, rel: &str) -> Option<Attrs>;
    fn read_at(&mut self, rel: &str, off: u64, buf: &mut [u8]) -> io::Result<usize>;
    fn readdir(&mut self, rel: &str) -> io::Result<Vec<DirEntry>>;

    // ---- optional (write side); default read-only ----
    fn read_only(&self) -> bool {
        true
    }
    fn write_at(&mut self, _rel: &str, _off: u64, _buf: &[u8]) -> io::Result<usize> {
        Err(erofs())
    }
    fn create(&mut self, _rel: &str, _mode: u32) -> io::Result<()> {
        Err(erofs())
    }
    fn mkdir(&mut self, _rel: &str, _mode: u32) -> io::Result<()> {
        Err(erofs())
    }
    fn unlink(&mut self, _rel: &str) -> io::Result<()> {
        Err(erofs())
    }
    fn rmdir(&mut self, _rel: &str) -> io::Result<()> {
        Err(erofs())
    }
    fn truncate(&mut self, _rel: &str, _len: u64) -> io::Result<()> {
        Err(erofs())
    }
    fn symlink(&mut self, _target: &str, _linkpath: &str) -> io::Result<()> {
        Err(erofs())
    }
    fn readlink(&mut self, _rel: &str) -> io::Result<String> {
        Err(io::Error::from_raw_os_error(22)) // EINVAL: not a symlink
    }
    fn rename(&mut self, _from: &str, _to: &str) -> io::Result<()> {
        Err(erofs())
    }
}

fn erofs() -> io::Error {
    io::Error::from_raw_os_error(30) // EROFS
}
