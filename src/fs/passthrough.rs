//! Host-directory passthrough — the mechanism for "punching holes" in the
//! sealed root.
//!
//! Mounting a `Passthrough` at a guest path (e.g. `/work`, the user's home)
//! makes that subtree map directly to a host directory, read-write, bypassing
//! the read-only squashfs root and its overlay. The mount table's
//! longest-prefix resolution means a passthrough at `/work` naturally overrides
//! the overlay `/` for everything under it.
//!
//! Paths are contained to the mapped root: a `..` component is rejected rather
//! than allowed to escape (the kernel also normalizes paths before they reach
//! here, so this is defense in depth). Host-only; not built for wasm.
//!
//! # Security: symlink containment is NOT complete yet
//!
//! Today's containment is purely *lexical* (`..` rejection). It does **not** yet
//! stop a host **symlink** inside the mapped directory from redirecting a lookup
//! outside it — and it is vulnerable to the **TOCTOU** race where a concurrent
//! actor swaps a component for a symlink between a `stat` and the `open`. A
//! passthrough must never become an escape hatch out of the shared path.
//!
//! The race-free fix (tracked in ROADMAP §4): resolve every path **beneath the
//! root** with the kernel holding the mount root as a directory fd and walking
//! components with `openat(.., O_NOFOLLOW)` — `openat2(RESOLVE_BENEATH |
//! RESOLVE_NO_MAGICLINKS)` on Linux, and a per-component `O_NOFOLLOW` walk +
//! `fstatat` on macOS (no `openat2`). Symlinks are then resolved by *our* VFS
//! *within the sandbox root* (so a `/work` symlink to `/etc` hits the sealed
//! squashfs `/etc`, never the host's). Until that lands, treat writable
//! passthroughs of attacker-influenced directories as unsafe.

use std::fs;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::PathBuf;

use super::{Attrs, DirEntry, MountFs, NodeKind};

#[derive(Debug)]
pub struct Passthrough {
    root: PathBuf,
    read_only: bool,
}

impl Passthrough {
    /// Map host directory `root` at the mount point, read-write.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            read_only: false,
        }
    }

    /// Map host directory `root` read-only.
    #[must_use]
    pub fn read_only(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            read_only: true,
        }
    }

    /// Resolve a mount-relative path to a host path, rejecting escapes.
    fn host_path(&self, rel: &str) -> io::Result<PathBuf> {
        let mut p = self.root.clone();
        for comp in rel.split('/') {
            match comp {
                "" | "." => {}
                ".." => return Err(io::Error::from_raw_os_error(13)), // EACCES
                c => p.push(c),
            }
        }
        Ok(p)
    }

    fn deny_if_ro(&self) -> io::Result<()> {
        if self.read_only {
            Err(io::Error::from_raw_os_error(30)) // EROFS
        } else {
            Ok(())
        }
    }
}

fn kind_of(m: &fs::Metadata) -> NodeKind {
    let t = m.file_type();
    if t.is_dir() {
        NodeKind::Dir
    } else if t.is_symlink() {
        NodeKind::Symlink
    } else if t.is_file() {
        NodeKind::File
    } else {
        // FIFO/socket/device — classify by mode bits.
        match m.mode() & 0o170_000 {
            0o010_000 => NodeKind::Fifo,
            0o140_000 => NodeKind::Socket,
            0o060_000 => NodeKind::BlockDevice,
            _ => NodeKind::CharDevice,
        }
    }
}

impl MountFs for Passthrough {
    fn read_only(&self) -> bool {
        self.read_only
    }

    fn stat(&mut self, rel: &str) -> Option<Attrs> {
        let host = self.host_path(rel).ok()?;
        let m = fs::symlink_metadata(&host).ok()?;
        Some(Attrs {
            kind: kind_of(&m),
            size: m.len(),
            mode: m.mode(),
            uid: m.uid(),
            gid: m.gid(),
            mtime: m.mtime(),
            inode: m.ino(),
            nlink: m.nlink() as u32,
        })
    }

    fn read_at(&mut self, rel: &str, off: u64, buf: &mut [u8]) -> io::Result<usize> {
        let mut f = fs::File::open(self.host_path(rel)?)?;
        f.seek(SeekFrom::Start(off))?;
        f.read(buf)
    }

    fn readdir(&mut self, rel: &str) -> io::Result<Vec<DirEntry>> {
        let mut out = Vec::new();
        for entry in fs::read_dir(self.host_path(rel)?)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            let (kind, inode) = match entry.metadata() {
                Ok(m) => (kind_of(&m), m.ino()),
                Err(_) => (NodeKind::File, 0),
            };
            out.push(DirEntry { name, kind, inode });
        }
        Ok(out)
    }

    fn write_at(&mut self, rel: &str, off: u64, buf: &[u8]) -> io::Result<usize> {
        self.deny_if_ro()?;
        let mut f = fs::OpenOptions::new().write(true).open(self.host_path(rel)?)?;
        f.seek(SeekFrom::Start(off))?;
        f.write(buf)
    }

    fn create(&mut self, rel: &str, mode: u32) -> io::Result<()> {
        self.deny_if_ro()?;
        let f = fs::File::create(self.host_path(rel)?)?;
        f.set_permissions(fs::Permissions::from_mode(mode & 0o777))
    }

    fn mkdir(&mut self, rel: &str, mode: u32) -> io::Result<()> {
        self.deny_if_ro()?;
        let host = self.host_path(rel)?;
        fs::create_dir(&host)?;
        fs::set_permissions(&host, fs::Permissions::from_mode(mode & 0o777))
    }

    fn unlink(&mut self, rel: &str) -> io::Result<()> {
        self.deny_if_ro()?;
        fs::remove_file(self.host_path(rel)?)
    }

    fn rmdir(&mut self, rel: &str) -> io::Result<()> {
        self.deny_if_ro()?;
        fs::remove_dir(self.host_path(rel)?)
    }

    fn truncate(&mut self, rel: &str, len: u64) -> io::Result<()> {
        self.deny_if_ro()?;
        fs::OpenOptions::new()
            .write(true)
            .open(self.host_path(rel)?)?
            .set_len(len)
    }

    fn symlink(&mut self, target: &str, linkpath: &str) -> io::Result<()> {
        self.deny_if_ro()?;
        std::os::unix::fs::symlink(target, self.host_path(linkpath)?)
    }

    fn readlink(&mut self, rel: &str) -> io::Result<String> {
        Ok(fs::read_link(self.host_path(rel)?)?
            .to_string_lossy()
            .into_owned())
    }

    fn rename(&mut self, from: &str, to: &str) -> io::Result<()> {
        self.deny_if_ro()?;
        fs::rename(self.host_path(from)?, self.host_path(to)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A throwaway host directory for a single test.
    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!("nixvm-pt-{}-{tag}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn write_read_and_list_host_dir() {
        let tmp = TempDir::new("rw");
        let mut pt = Passthrough::new(tmp.0.clone());
        pt.create("f.txt", 0o644).unwrap();
        assert_eq!(pt.write_at("f.txt", 0, b"host data").unwrap(), 9);
        let mut buf = [0u8; 9];
        assert_eq!(pt.read_at("f.txt", 0, &mut buf).unwrap(), 9);
        assert_eq!(&buf, b"host data");

        // The file is really on the host.
        assert_eq!(fs::read(tmp.0.join("f.txt")).unwrap(), b"host data");

        pt.mkdir("sub", 0o755).unwrap();
        let names: Vec<_> = pt.readdir("").unwrap().into_iter().map(|e| e.name).collect();
        assert!(names.contains(&"f.txt".to_string()) && names.contains(&"sub".to_string()));
        assert_eq!(pt.stat("f.txt").unwrap().size, 9);
    }

    #[test]
    fn rejects_parent_escape() {
        let tmp = TempDir::new("esc");
        let mut pt = Passthrough::new(tmp.0.clone());
        assert!(pt.stat("../etc/passwd").is_none());
        assert!(pt.read_at("../../secret", 0, &mut [0u8; 1]).is_err());
    }

    #[test]
    fn read_only_denies_writes() {
        let tmp = TempDir::new("ro");
        fs::write(tmp.0.join("x"), b"hi").unwrap();
        let mut pt = Passthrough::read_only(tmp.0.clone());
        assert!(pt.write_at("x", 0, b"no").is_err());
        assert!(pt.create("new", 0o644).is_err());
        // reads still work
        let mut buf = [0u8; 2];
        assert_eq!(pt.read_at("x", 0, &mut buf).unwrap(), 2);
    }
}
