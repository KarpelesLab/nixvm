//! Host-directory passthrough — the mechanism for "punching holes" in the
//! sealed root.
//!
//! Mounting a `Passthrough` at a guest path (e.g. `/work`, the user's home)
//! makes that subtree map directly to a host directory, read-write, bypassing
//! the read-only squashfs root and its overlay. The mount table's
//! longest-prefix resolution means a passthrough at `/work` naturally overrides
//! the overlay `/` for everything under it.
//!
//! Host-only; not built for wasm.
//!
//! This module is the one deliberate exception to the project's "`unsafe`
//! lives only in `vcpu::hvf`" rule: safe, TOCTOU-free confinement genuinely
//! requires the dirfd-relative `*at(2)` syscalls, which `std` does not expose,
//! so the `host` submodule provides them (see the security note): raw system
//! calls on a Linux-ABI host — libc-free ones included — and hand-declared
//! libc functions elsewhere (macOS). The `unsafe` is confined to those shims.
//!
//! # Security: confined, symlink/TOCTOU-safe path resolution
//!
//! A passthrough must never become an escape hatch out of the shared host
//! directory — not even when a host symlink inside the mapped directory
//! points outside it, and not even when a concurrent actor swaps a path
//! component for such a symlink *between* a check and the actual I/O (the
//! classic TOCTOU race).
//!
//! Resolution here never joins a guest-relative path onto the host root as a
//! string and hands it to a path-based syscall. Instead every lookup walks
//! the path **one component at a time**, starting from a directory fd opened
//! on the mount root, using the `openat(2)` family with `O_NOFOLLOW` on each
//! component (`std` has no dirfd-relative open, so the handful of `openat`/
//! `unlinkat`/`mkdirat`/`symlinkat`/`renameat`/`readlinkat` calls needed are
//! declared by hand below — this crate has no direct `libc` dependency and
//! none is added here). A symlink encountered mid-walk is never handed to
//! the kernel to auto-follow; instead *we* read its target and continue the
//! walk ourselves, re-anchored: an absolute target is re-rooted at the mount
//! root (so `/work/link -> /etc/passwd` resolves under the sealed mount, not
//! the host's real `/etc`), and a `..` in a relative target can pop our own
//! directory-fd stack but never past the root frame — so no chain of
//! symlinks, however constructed, can walk the resolution above the mount
//! root. Legitimate in-tree symlinks (pointing at another in-root path) are
//! still followed and work exactly as before.
//!
//! The *final* path component is resolved the same way, and the actual
//! mutating/reading syscall (`openat`/`unlinkat`/`mkdirat`/`symlinkat`/
//! `renameat`) is always issued directly against `(parent_dirfd, name)` with
//! `O_NOFOLLOW` (or an inherently non-following `*at` call) as the *only*
//! operation that touches the node — so even if a concurrent swap manages to
//! replace that exact name with a symlink in between our resolution and that
//! call, the kernel rejects following it (`ELOOP`) rather than silently
//! redirecting the I/O. There is no check-then-open window: the check *is*
//! the open (or the open fails safely).
//!
//! `readdir` lists an already-opened (and thus already-confined) directory
//! fd directly via `fdopendir`/`readdir`/`closedir` (POSIX functions `std`
//! doesn't bind) rather than by re-opening any path — `/proc/self/fd/<n>`
//! works for this on Linux, but macOS's `/dev/fd/<n>` does not support
//! directories (verified empirically: `opendir("/dev/fd/N")` on a real
//! directory fd fails `ENOTDIR`), so the fd-native functions are used
//! uniformly on both platforms instead.

use std::collections::VecDeque;
use std::ffi::{CStr, CString, c_int};
use std::fs;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::protect::{Op, Protected};
use super::{Attrs, DirEntry, MountFs, NodeKind, SetTime};
use crate::limits::Quota;

mod host;
use host::Timespec;

/// `openat(2)`-family flag and errno values `std` doesn't expose. Declared
/// by hand (rather than depending on the `libc` crate, which is not a
/// dependency of this crate) directly from each platform's headers
/// (`<fcntl.h>` / `<sys/errno.h>`); these are stable, documented ABI
/// constants, not implementation details.
#[cfg(any(target_os = "linux", nixvm_linux))]
mod sys {
    pub const O_RDONLY: c_int = 0o0;
    pub const O_WRONLY: c_int = 0o1;
    pub const O_CREAT: c_int = 0o100;
    pub const O_TRUNC: c_int = 0o1000;
    pub const O_NONBLOCK: c_int = 0o4000;
    pub const O_DIRECTORY: c_int = 0o200_000;
    pub const O_NOFOLLOW: c_int = 0o400_000;
    pub const O_CLOEXEC: c_int = 0o2_000_000;
    /// Linux-only: an fd that identifies a location in the filesystem tree
    /// without opening the underlying file for I/O — usable with `fstat`
    /// even when the target is a symlink and `O_NOFOLLOW` is set, which is
    /// exactly what's needed to `lstat` a symlink via an fd instead of a
    /// racy path.
    pub const O_PATH: c_int = 0o10_000_000;
    pub const AT_REMOVEDIR: c_int = 0x200;
    pub const AT_SYMLINK_NOFOLLOW: c_int = 0x100;
    pub const ELOOP: i32 = 40;
    /// `utimensat` `tv_nsec` sentinels (Linux `<bits/stat.h>`): use the current
    /// time / leave this field unchanged.
    pub const UTIME_NOW: i64 = 0x3fff_ffff;
    pub const UTIME_OMIT: i64 = 0x3fff_fffe;

    use std::ffi::c_int;
}

#[cfg(target_os = "macos")]
mod sys {
    pub const O_RDONLY: c_int = 0x0000;
    pub const O_WRONLY: c_int = 0x0001;
    pub const O_CREAT: c_int = 0x0200;
    pub const O_TRUNC: c_int = 0x0400;
    pub const O_NONBLOCK: c_int = 0x0004;
    pub const O_DIRECTORY: c_int = 0x0010_0000;
    pub const O_NOFOLLOW: c_int = 0x0000_0100;
    pub const O_CLOEXEC: c_int = 0x0100_0000;
    /// macOS-only: open the symlink itself rather than its target (there is
    /// no `O_PATH` on macOS, so this is the equivalent for getting an fd we
    /// can `fstat` to `lstat` a symlink).
    pub const O_SYMLINK: c_int = 0x0020_0000;
    pub const AT_REMOVEDIR: c_int = 0x0080;
    pub const AT_SYMLINK_NOFOLLOW: c_int = 0x0020;
    pub const ELOOP: i32 = 62;
    /// `utimensat` `tv_nsec` sentinels (macOS `<sys/stat.h>`).
    pub const UTIME_NOW: i64 = -1;
    pub const UTIME_OMIT: i64 = -2;

    use std::ffi::c_int;
}

#[cfg(not(any(target_os = "linux", nixvm_linux, target_os = "macos")))]
compile_error!("passthrough: openat(2) flag/errno constants are only defined for linux and macos");

/// Bound on symlink hops resolved per lookup (matches Linux's own
/// `MAXSYMLINKS`), so a symlink cycle fails with `ELOOP` instead of looping
/// forever.
const MAX_SYMLINKS: u32 = 40;

/// Bound on total path components processed per lookup (original path plus
/// every symlink target spliced in), so a maliciously long chain of short
/// symlink targets can't force unbounded work.
const MAX_COMPONENTS: u32 = 4096;

fn cstr(component: &str) -> io::Result<CString> {
    CString::new(component).map_err(|_| io::Error::from_raw_os_error(22)) // EINVAL
}

/// Canonical (Linux) `ELOOP`, used only when *constructing* a fresh
/// guest-facing "too many symlinks" error. Never used for *comparing*
/// against a host-native errno just returned by `openat`/`readlinkat` —
/// those comparisons must use the platform's own `sys::ELOOP`, since that's
/// the value the host syscall actually reports.
const LINUX_ELOOP: i32 = 40;

/// `ENOTDIR`, numbered the same on every supported host.
const ENOTDIR: i32 = 20;

fn eloop() -> io::Error {
    io::Error::from_raw_os_error(LINUX_ELOOP)
}

/// Map a *host*-native `errno` (as produced by the raw `openat`/`unlinkat`/
/// `mkdirat`/`symlinkat`/`renameat`/`readlinkat` calls below, or by the
/// plain `open` of the mount root) to the canonical Linux errno the rest of
/// the kernel expects: guest syscalls hand `io::Error::raw_os_error()`
/// straight through to the guest as a Linux errno (see
/// `crate::kernel::io_errno`), so on a Linux host this is the identity map —
/// but on macOS/BSD a handful of codes that mean the same thing are numbered
/// differently (`ENOTEMPTY`, `ELOOP`, `ENAMETOOLONG`, the `EAGAIN`/`EDEADLK`
/// pair, ...), and must be translated so a passthrough mount behaves
/// identically for the guest regardless of which host OS backs it.
///
/// Only ever applied at a point where the raw errno is about to become a
/// *terminal*, guest-visible result — never at a point where a caller still
/// needs to compare the untranslated value against the host's own
/// `sys::ELOOP` to detect "this path component is a symlink" mid-walk.
#[cfg(target_os = "macos")]
fn normalize_errno(err: io::Error) -> io::Error {
    let Some(raw) = err.raw_os_error() else {
        return err;
    };
    let linux = match raw {
        35 => 11,          // EAGAIN/EWOULDBLOCK
        11 => 35,          // EDEADLK
        63 => 36,          // ENAMETOOLONG
        77 => 37,          // ENOLCK
        78 => 38,          // ENOSYS
        66 => 39,          // ENOTEMPTY
        62 => LINUX_ELOOP, // ELOOP
        91 => 42,          // ENOMSG
        90 => 43,          // EIDRM
        92 => 84,          // EILSEQ
        94 => 74,          // EBADMSG
        84 => 75,          // EOVERFLOW
        70 => 116,         // ESTALE
        69 => 122,         // EDQUOT
        89 => 125,         // ECANCELED
        105 => 130,        // EOWNERDEAD
        104 => 131,        // ENOTRECOVERABLE
        45 | 102 => 95,    // ENOTSUP / EOPNOTSUPP (Linux aliases both to 95)
        other => other,
    };
    io::Error::from_raw_os_error(linux)
}

#[cfg(not(target_os = "macos"))]
fn normalize_errno(err: io::Error) -> io::Error {
    err
}

/// Translate one [`SetTime`] into a host `timespec` for `utimensat`.
fn to_timespec(t: SetTime) -> Timespec {
    match t {
        SetTime::Omit => Timespec {
            tv_sec: 0,
            tv_nsec: sys::UTIME_OMIT,
        },
        SetTime::Now => Timespec {
            tv_sec: 0,
            tv_nsec: sys::UTIME_NOW,
        },
        SetTime::Set { sec, nsec } => Timespec {
            tv_sec: sec,
            tv_nsec: nsec,
        },
    }
}

/// `utimensat(parent, name, [atime, mtime], AT_SYMLINK_NOFOLLOW)` — set both
/// timestamps on the node named by `(dirfd, name)` without following a final
/// symlink (the confined walk has already done any following the guest asked
/// for).
fn raw_utimensat(dirfd: RawFd, name: &CStr, atime: SetTime, mtime: SetTime) -> io::Result<()> {
    let times = [to_timespec(atime), to_timespec(mtime)];
    host::utimensat(dirfd, name, &times, sys::AT_SYMLINK_NOFOLLOW).map_err(normalize_errno)
}

fn raw_fchmodat(dirfd: RawFd, name: &CStr, mode: u32) -> io::Result<()> {
    // The confined resolution already followed any symlink the guest wanted
    // followed, so the final component is a concrete node; flags = 0.
    host::fchmodat(dirfd, name, mode & 0o7777).map_err(normalize_errno)
}

fn raw_fchownat(dirfd: RawFd, name: &CStr, uid: Option<u32>, gid: Option<u32>) -> io::Result<()> {
    // A `-1` (u32::MAX) owner/group means "leave unchanged", matching chown(2).
    let owner = uid.unwrap_or(u32::MAX);
    let group = gid.unwrap_or(u32::MAX);
    host::fchownat(dirfd, name, owner, group, sys::AT_SYMLINK_NOFOLLOW).map_err(normalize_errno)
}

fn raw_mknodat(dirfd: RawFd, name: &CStr, mode: u32) -> io::Result<()> {
    host::mknodat(dirfd, name, mode).map_err(normalize_errno)
}

fn raw_openat(dirfd: RawFd, name: &CStr, flags: c_int, mode: u32) -> io::Result<OwnedFd> {
    host::openat(dirfd, name, flags | sys::O_CLOEXEC, mode)
}

/// `Ok(Some(target))` if `name` is a symlink, `Ok(None)` if it exists but
/// isn't one (`EINVAL`), `Err` for any other failure (including not found).
fn raw_readlinkat(dirfd: RawFd, name: &CStr) -> io::Result<Option<String>> {
    let mut buf = [0u8; 4096];
    match host::readlinkat(dirfd, name, &mut buf) {
        Ok(n) => Ok(Some(String::from_utf8_lossy(&buf[..n]).into_owned())),
        // EINVAL: exists, but isn't a symlink
        Err(err) if err.raw_os_error() == Some(22) => Ok(None),
        Err(err) => Err(normalize_errno(err)),
    }
}

fn raw_unlinkat(dirfd: RawFd, name: &CStr, flags: c_int) -> io::Result<()> {
    host::unlinkat(dirfd, name, flags).map_err(normalize_errno)
}

fn raw_mkdirat(dirfd: RawFd, name: &CStr, mode: u32) -> io::Result<()> {
    host::mkdirat(dirfd, name, mode).map_err(normalize_errno)
}

fn raw_symlinkat(target: &CStr, dirfd: RawFd, name: &CStr) -> io::Result<()> {
    host::symlinkat(target, dirfd, name).map_err(normalize_errno)
}

fn raw_renameat(from_dirfd: RawFd, from: &CStr, to_dirfd: RawFd, to: &CStr) -> io::Result<()> {
    host::renameat(from_dirfd, from, to_dirfd, to).map_err(normalize_errno)
}

/// Hard-link `to` (under `to_dirfd`) to the existing `from` (under `from_dirfd`).
/// `flags = 0`: the final component of `from` is not followed, so linking a
/// symlink links the link itself — matching `link(2)`.
fn raw_linkat(from_dirfd: RawFd, from: &CStr, to_dirfd: RawFd, to: &CStr) -> io::Result<()> {
    host::linkat(from_dirfd, from, to_dirfd, to).map_err(normalize_errno)
}

/// Open an fd on a symlink *itself* (not its target) — the fd-based
/// equivalent of `lstat`, needed because `stat()` must describe the symlink
/// even though the confined walk otherwise only ever opens non-symlinks.
#[cfg(not(target_os = "macos"))]
fn open_symlink_itself(parent: RawFd, name: &CStr) -> io::Result<OwnedFd> {
    raw_openat(parent, name, sys::O_PATH | sys::O_NOFOLLOW, 0)
}

#[cfg(target_os = "macos")]
fn open_symlink_itself(parent: RawFd, name: &CStr) -> io::Result<OwnedFd> {
    raw_openat(parent, name, sys::O_SYMLINK, 0)
}

/// List a directory we already hold open (and have therefore already
/// confined/verified via [`Passthrough::resolve`]) by its fd — never by
/// re-opening any path, which would reintroduce a symlink-swap TOCTOU
/// window. Consumes `fd`.
fn list_dir_fd(fd: OwnedFd) -> io::Result<Vec<DirEntry>> {
    // `dir` keeps the descriptor open for the whole function, so it's still
    // valid for the per-entry `openat`/`fstat` lookups below. Confinement/
    // TOCTOU-safety doesn't depend on the listing: each name is only ever
    // looked up again relative to this same fd.
    let (dir, raw_names) = host::read_dir(fd)?;
    let raw = dir.as_raw_fd();
    let names: Vec<String> = raw_names
        .iter()
        .map(|n| String::from_utf8_lossy(n).into_owned())
        .filter(|n| n != "." && n != "..")
        .collect();

    // Per-entry metadata via our own confined `stat`-style lookup (an
    // `openat(O_NOFOLLOW)` + `fstat`, falling back to the symlink-itself fd
    // for symlinks) — never a further path-based syscall.
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let cname = cstr(&name)?;
        // O_NONBLOCK: a FIFO entry must not block this listing's per-entry fstat.
        let (kind, inode) = match raw_openat(
            raw,
            &cname,
            sys::O_RDONLY | sys::O_NOFOLLOW | sys::O_NONBLOCK,
            0,
        ) {
            Ok(entry_fd) => match fs::File::from(entry_fd).metadata() {
                Ok(m) => (kind_of(&m), m.ino()),
                Err(_) => (NodeKind::File, 0),
            },
            Err(e) if e.raw_os_error() == Some(sys::ELOOP) => {
                match open_symlink_itself(raw, &cname).and_then(|f| fs::File::from(f).metadata()) {
                    Ok(m) => (kind_of(&m), m.ino()),
                    Err(_) => (NodeKind::Symlink, 0),
                }
            }
            Err(_) => (NodeKind::File, 0),
        };
        out.push(DirEntry { name, kind, inode });
    }

    drop(dir); // closes `raw`.
    Ok(out)
}

#[derive(Debug)]
pub struct Passthrough {
    root: PathBuf,
    read_only: bool,
    /// What growth on the host's disk is charged to.
    disk: Arc<Quota>,
    /// Bytes charged since the host's free space was last checked; `None` before
    /// the first check.
    since_host_check: Option<u64>,
    /// The VM's disk this is (see [`super::blockdev`]), once mounted.
    dev: Option<u64>,
    /// Paths kept read-only inside the tree (see [`Self::protect`]).
    protected: Protected,
}

/// What a new directory entry is charged, since it takes host disk too.
const ENTRY_COST: u64 = 4096;
/// How much may be written between checks of the host's free space.
const HOST_CHECK_INTERVAL: u64 = 64 << 20;

impl Passthrough {
    /// Map host directory `root` at the mount point, read-write.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            read_only: false,
            disk: Quota::unlimited(),
            since_host_check: None,
            dev: None,
            protected: Protected::default(),
        }
    }

    /// Map host directory `root` read-only.
    #[must_use]
    pub fn read_only(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            read_only: true,
            disk: Quota::unlimited(),
            since_host_check: None,
            dev: None,
            protected: Protected::default(),
        }
    }

    /// Keep `paths` (relative to the mapped directory: directories, which
    /// protects everything under them, or single files) read-only for the
    /// guest, as the host nodes they are rather than by the names they're
    /// reached through: renaming a parent doesn't move them out of reach, a
    /// symlink into one doesn't get past it, and one that doesn't exist yet
    /// can't be made. Writes, creation, removal, renames, links and metadata
    /// changes under a protected path fail `EROFS`; renaming or removing a
    /// directory leading to one, or putting anything but a directory at its
    /// name, fails `EPERM`, as does hard-linking a protected file. Names
    /// compare case-insensitively. See [`super::protect`] for the details.
    ///
    /// ```no_run
    /// # use nixvm::fs::Passthrough;
    /// let project = Passthrough::new("/home/me/project").protect([".git/hooks", "atelier.toml"]);
    /// ```
    #[must_use]
    pub fn protect<I, S>(mut self, paths: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        for p in paths {
            self.protected.add(p.as_ref());
        }
        self
    }

    /// Refuse `op` on `rel` if it touches a protected path (see
    /// [`Self::protect`]). `rel` is resolved exactly as the operation itself
    /// resolves it (`follow_final`, `allow_missing_final` as for
    /// [`Self::resolve`]), symlinks and all, and the check is made on where
    /// that lands. The backend is behind the mount table's lock for the whole
    /// call, so nothing in the guest can change the tree between this walk
    /// and the operation's own.
    fn guard(
        &self,
        op: Op,
        rel: &str,
        follow_final: bool,
        allow_missing_final: bool,
    ) -> io::Result<()> {
        if self.protected.is_empty() {
            return Ok(());
        }
        if Self::split_rel(rel)?.is_empty() {
            // The mount root: an ancestor of everything, never itself made,
            // removed or renamed (resolution refuses it).
            let none: [&str; 0] = [];
            return self.protected.check(op, &none);
        }
        let (_, name, mut path) = self.resolve_traced(rel, follow_final, allow_missing_final)?;
        path.push(name);
        self.protected.check(op, &path)
    }

    /// Charge `bytes` of new disk use, or fail with `ENOSPC` if that's over the
    /// VM's quota or the host is running out. The host always keeps some room,
    /// whatever the quota says: a guest must not be able to fill its disk.
    fn charge_disk(&mut self, bytes: u64) -> io::Result<()> {
        if !self.disk.try_charge(bytes) {
            return Err(enospc());
        }
        let since = self
            .since_host_check
            .map_or(HOST_CHECK_INTERVAL, |s| s + bytes);
        if since < HOST_CHECK_INTERVAL {
            self.since_host_check = Some(since);
            return Ok(());
        }
        self.since_host_check = Some(0);
        if host_is_low_on_space(&self.root) {
            self.disk.release(bytes);
            // Check again next time rather than after another interval.
            self.since_host_check = None;
            return Err(enospc());
        }
        Ok(())
    }

    /// Hold the charge for a new directory entry while `make` creates it.
    fn with_entry_charge(
        &mut self,
        make: impl FnOnce(&mut Self) -> io::Result<()>,
    ) -> io::Result<()> {
        self.charge_disk(ENTRY_COST)?;
        let made = make(self);
        if made.is_err() {
            self.disk.release(ENTRY_COST);
        }
        made
    }

    fn create_entry(&mut self, rel: &str, mode: u32) -> io::Result<()> {
        let (parent, name) = self.resolve(rel, true, true)?;
        let cname = cstr(&name)?;
        let fd = raw_openat(
            parent.as_raw_fd(),
            &cname,
            sys::O_WRONLY | sys::O_CREAT | sys::O_TRUNC | sys::O_NOFOLLOW,
            mode & 0o777,
        )
        .map_err(normalize_errno)?;
        // openat's mode is subject to umask; chmod explicitly afterward to
        // match the exact requested bits, as the original `fs::File::create`
        // + `set_permissions` pair did.
        fs::File::from(fd).set_permissions(fs::Permissions::from_mode(mode & 0o777))
    }

    fn mkdir_entry(&mut self, rel: &str, mode: u32) -> io::Result<()> {
        self.deny_if_ro()?;
        let (parent, name) = self.resolve(rel, false, false)?;
        let cname = cstr(&name)?;
        raw_mkdirat(parent.as_raw_fd(), &cname, mode & 0o777)?;
        let fd = raw_openat(
            parent.as_raw_fd(),
            &cname,
            sys::O_DIRECTORY | sys::O_RDONLY | sys::O_NOFOLLOW,
            0,
        )
        .map_err(normalize_errno)?;
        fs::File::from(fd).set_permissions(fs::Permissions::from_mode(mode & 0o777))
    }

    fn mknod_entry(&mut self, rel: &str, mode: u32) -> io::Result<()> {
        self.deny_if_ro()?;
        let (parent, name) = self.resolve(rel, false, false)?;
        let cname = cstr(&name)?;
        raw_mknodat(parent.as_raw_fd(), &cname, mode)?;
        // Match the exact requested permission bits (mknod's mode is subject to
        // the umask), mirroring `create`/`mkdir`.
        let _ = raw_fchmodat(parent.as_raw_fd(), &cname, mode & 0o7777);
        Ok(())
    }

    fn symlink_entry(&mut self, target: &str, linkpath: &str) -> io::Result<()> {
        self.deny_if_ro()?;
        let (parent, name) = self.resolve(linkpath, false, false)?;
        let cname = cstr(&name)?;
        let ctarget = cstr(target)?;
        raw_symlinkat(&ctarget, parent.as_raw_fd(), &cname)
    }

    fn link_entry(&mut self, old_rel: &str, new_rel: &str) -> io::Result<()> {
        self.deny_if_ro()?;
        // Resolve both without following the final component (link(2) links the
        // symlink itself), then make a real host hard link so st_nlink, the
        // shared inode, and write-through all behave.
        let (old_parent, old_name) = self.resolve(old_rel, false, false)?;
        let (new_parent, new_name) = self.resolve(new_rel, false, false)?;
        let cold = cstr(&old_name)?;
        let cnew = cstr(&new_name)?;
        raw_linkat(old_parent.as_raw_fd(), &cold, new_parent.as_raw_fd(), &cnew)
    }

    fn deny_if_ro(&self) -> io::Result<()> {
        if self.read_only {
            Err(io::Error::from_raw_os_error(30)) // EROFS
        } else {
            Ok(())
        }
    }

    /// Open a dirfd on the configured mount root. The root path itself is
    /// trusted, operator-configured state (not attacker-influenced on a
    /// per-call basis), so it alone is opened directly by path; every path
    /// *beneath* it is then walked component-by-component via `openat` and
    /// never rejoined into a string for the kernel to re-resolve.
    fn root_fd(&self) -> io::Result<OwnedFd> {
        let f = fs::OpenOptions::new()
            .read(true)
            .custom_flags(sys::O_DIRECTORY)
            .open(&self.root)
            .map_err(normalize_errno)?;
        Ok(OwnedFd::from(f))
    }

    /// Split a mount-relative guest path into components, rejecting any
    /// literal `..` outright. This is the original lexical guard, kept
    /// as-is: it applies only to the *literal* guest-supplied path, not to
    /// symlink targets encountered while walking it (see `split_target`).
    fn split_rel(rel: &str) -> io::Result<VecDeque<String>> {
        let mut out = VecDeque::new();
        for c in rel.split('/') {
            match c {
                "" | "." => {}
                ".." => return Err(io::Error::from_raw_os_error(13)), // EACCES
                c => out.push_back(c.to_string()),
            }
        }
        Ok(out)
    }

    /// Split a symlink *target* into components. Unlike `split_rel`, `..`
    /// is kept (relative symlinks legitimately use it) — but every `..`
    /// encountered while walking is resolved against our own dirfd stack,
    /// which refuses to pop past the root frame, so it can only ever walk
    /// back up as far as the mount root, never above it.
    fn split_target(target: &str) -> (bool, VecDeque<String>) {
        let absolute = target.starts_with('/');
        let mut out = VecDeque::new();
        for c in target.split('/') {
            if !c.is_empty() && c != "." {
                out.push_back(c.to_string());
            }
        }
        (absolute, out)
    }

    /// Confined resolution of a mount-relative path to `(parent_dirfd,
    /// final_name)`. Every intermediate directory component — and, when
    /// `follow_final` is set, the final component too — is resolved
    /// component-by-component from a dirfd stack rooted at the mount root:
    /// each is opened with `O_NOFOLLOW`, so a symlink (whether legitimately
    /// present or swapped in by a concurrent race) is never silently
    /// followed by the kernel. When one is found, *we* read its target and
    /// splice the resulting components back into the walk, re-anchored
    /// (absolute targets reset to the root; `..` is bounded by the stack).
    ///
    /// The caller performs the real, operation-specific syscall itself
    /// against the returned `(parent_dirfd, name)` with `O_NOFOLLOW` (or an
    /// inherently non-following `*at` call) — that call, not this
    /// resolution, is what actually touches the node, so a last-instant
    /// symlink swap can only make that final syscall fail safely, never
    /// redirect it.
    ///
    /// When `follow_final` is false, the final component is returned as-is,
    /// without even a probe — appropriate for operations that must act on
    /// the name itself, not whatever it might point to (`lstat`, `unlink`,
    /// `rmdir`, `mkdir`, `readlink`, `symlink` creation, `rename`).
    ///
    /// `allow_missing_final`, only meaningful with `follow_final`, tolerates
    /// the final component not existing yet (for `create`, which must be
    /// able to make a brand new name).
    fn resolve(
        &self,
        rel: &str,
        follow_final: bool,
        allow_missing_final: bool,
    ) -> io::Result<(OwnedFd, String)> {
        self.resolve_traced(rel, follow_final, allow_missing_final)
            .map(|(parent, name, _)| (parent, name))
    }

    /// [`Self::resolve`], also giving the path from the mount root to the
    /// returned parent directory as it was actually walked — one name per
    /// directory fd opened, symlinks already followed and `..` applied — which
    /// is what [`Self::guard`] checks.
    fn resolve_traced(
        &self,
        rel: &str,
        follow_final: bool,
        allow_missing_final: bool,
    ) -> io::Result<(OwnedFd, String, Vec<String>)> {
        let mut queue = Self::split_rel(rel)?;
        if queue.is_empty() {
            // The mount root itself: none of the (parent, name) operations
            // apply meaningfully to it (there is no in-confinement parent
            // above the root).
            return Err(io::Error::from_raw_os_error(21)); // EISDIR
        }

        let mut stack: Vec<OwnedFd> = vec![self.root_fd()?];
        // The name of each frame above the root: `names[i]` is `stack[i + 1]`.
        let mut names: Vec<String> = Vec::new();
        let mut budget = MAX_SYMLINKS;
        let mut seen = 0u32;

        loop {
            let Some(name) = queue.pop_front() else {
                // The walk fully consumed itself — e.g. a symlink target of
                // `..`, `.`, or `/` that collapses back to a bare directory
                // — leaving no filename component for the caller's
                // operation to act on.
                return Err(io::Error::from_raw_os_error(21)); // EISDIR
            };
            seen += 1;
            if seen > MAX_COMPONENTS {
                return Err(eloop());
            }

            if name == ".." {
                if stack.len() > 1 {
                    stack.pop();
                    names.pop();
                }
                continue;
            }

            let is_last = queue.is_empty();
            let cname = cstr(&name)?;
            let parent_raw = stack.last().expect("root frame always present").as_raw_fd();

            if is_last && !follow_final {
                let parent = stack.pop().expect("root frame always present");
                return Ok((parent, name, names));
            }

            let open_flags = if is_last {
                sys::O_RDONLY | sys::O_NOFOLLOW
            } else {
                sys::O_DIRECTORY | sys::O_RDONLY | sys::O_NOFOLLOW
            };

            match raw_openat(parent_raw, &cname, open_flags, 0) {
                Ok(fd) => {
                    if is_last {
                        // Confirmed not a symlink; drop this probe fd and
                        // let the caller perform its own, operation-specific
                        // O_NOFOLLOW open as the one call that actually
                        // touches the node.
                        drop(fd);
                        let parent = stack.pop().expect("root frame always present");
                        return Ok((parent, name, names));
                    }
                    stack.push(fd);
                    names.push(name);
                }
                Err(e)
                    if is_last
                        && follow_final
                        && allow_missing_final
                        && e.raw_os_error() == Some(2) =>
                {
                    // ENOENT on the final component with a caller that's
                    // fine creating it (e.g. `create`): hand back the name
                    // for the caller's own O_CREAT open.
                    let parent = stack.pop().expect("root frame always present");
                    return Ok((parent, name, names));
                }
                // A symlink: ELOOP from `O_NOFOLLOW` — or, for a directory
                // component, ENOTDIR, which Linux reports for `O_DIRECTORY`
                // on a symlink before it looks at `O_NOFOLLOW`. (A real
                // non-directory there is still ENOTDIR: readlink says so.)
                Err(e)
                    if e.raw_os_error() == Some(sys::ELOOP)
                        || (!is_last && e.raw_os_error() == Some(ENOTDIR)) =>
                {
                    let target = match raw_readlinkat(parent_raw, &cname)? {
                        Some(t) => t,
                        None if e.raw_os_error() == Some(ENOTDIR) => return Err(e),
                        None => return Err(eloop()),
                    };
                    budget = budget.checked_sub(1).ok_or_else(eloop)?;
                    let (absolute, mut target_comps) = Self::split_target(&target);
                    if absolute {
                        stack.truncate(1);
                        names.clear();
                    }
                    while let Some(c) = target_comps.pop_back() {
                        queue.push_front(c);
                    }
                }
                Err(e) => return Err(normalize_errno(e)),
            }
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

fn attrs_of(m: &fs::Metadata) -> Attrs {
    Attrs {
        kind: kind_of(m),
        size: m.len(),
        mode: m.mode(),
        uid: m.uid(),
        gid: m.gid(),
        atime: m.atime(),
        mtime: m.mtime(),
        inode: m.ino(),
        nlink: m.nlink() as u32,
        rdev: m.rdev(),
        dev: 0,
        blocks: None,
    }
}

impl MountFs for Passthrough {
    fn fs_type(&self) -> &'static str {
        "bind"
    }

    fn attach_disk_quota(&mut self, quota: &Arc<Quota>) {
        self.disk = quota.clone();
    }

    fn attach_disks(&mut self, disks: &super::blockdev::SharedDisks) {
        if self.dev.is_none() {
            let bytes = host_space(&self.root).map_or(0, |s| s.total);
            self.dev = Some(disks.lock().unwrap().add(bytes, self.read_only));
        }
    }

    fn own_dev(&self) -> Option<u64> {
        self.dev
    }

    fn read_only(&self) -> bool {
        self.read_only
    }

    fn stat(&mut self, rel: &str) -> Option<Attrs> {
        if rel.is_empty() || rel == "." {
            let fd = self.root_fd().ok()?;
            let m = fs::File::from(fd).metadata().ok()?;
            return Some(attrs_of(&m));
        }
        let (parent, name) = self.resolve(rel, false, false).ok()?;
        let cname = cstr(&name).ok()?;
        // O_NONBLOCK so a FIFO (or other special file) opens immediately for the
        // fstat instead of blocking forever waiting for a peer.
        match raw_openat(
            parent.as_raw_fd(),
            &cname,
            sys::O_RDONLY | sys::O_NOFOLLOW | sys::O_NONBLOCK,
            0,
        ) {
            Ok(fd) => {
                let m = fs::File::from(fd).metadata().ok()?;
                Some(attrs_of(&m))
            }
            Err(e) if e.raw_os_error() == Some(sys::ELOOP) => {
                // The final component is itself a symlink: describe the
                // link (matches `symlink_metadata`/`lstat`), not its target.
                let fd = open_symlink_itself(parent.as_raw_fd(), &cname).ok()?;
                let m = fs::File::from(fd).metadata().ok()?;
                Some(attrs_of(&m))
            }
            Err(_) => None,
        }
    }

    fn read_at(&mut self, rel: &str, off: u64, buf: &mut [u8]) -> io::Result<usize> {
        let (parent, name) = self.resolve(rel, true, false)?;
        let cname = cstr(&name)?;
        let fd = raw_openat(
            parent.as_raw_fd(),
            &cname,
            sys::O_RDONLY | sys::O_NOFOLLOW,
            0,
        )
        .map_err(normalize_errno)?;
        let mut f = fs::File::from(fd);
        f.seek(SeekFrom::Start(off))?;
        f.read(buf)
    }

    fn readdir(&mut self, rel: &str) -> io::Result<Vec<DirEntry>> {
        let fd = if rel.is_empty() || rel == "." {
            self.root_fd()?
        } else {
            let (parent, name) = self.resolve(rel, true, false)?;
            let cname = cstr(&name)?;
            raw_openat(
                parent.as_raw_fd(),
                &cname,
                sys::O_DIRECTORY | sys::O_RDONLY | sys::O_NOFOLLOW,
                0,
            )
            .map_err(normalize_errno)?
        };

        list_dir_fd(fd)
    }

    fn write_at(&mut self, rel: &str, off: u64, buf: &[u8]) -> io::Result<usize> {
        self.deny_if_ro()?;
        self.guard(Op::Write, rel, true, false)?;
        let (parent, name) = self.resolve(rel, true, false)?;
        let cname = cstr(&name)?;
        let fd = raw_openat(
            parent.as_raw_fd(),
            &cname,
            sys::O_WRONLY | sys::O_NOFOLLOW,
            0,
        )
        .map_err(normalize_errno)?;
        let mut f = fs::File::from(fd);
        // Disk use is the blocks a file has, as `du` counts it, not its size:
        // a sparse file costs only what's written. Charge the most the write
        // could allocate, then give back what it didn't.
        let before = allocated(&f.metadata()?);
        let most =
            (buf.len() as u64).div_ceil(BLOCK) * BLOCK + if buf.is_empty() { 0 } else { BLOCK };
        self.charge_disk(most)?;
        let written = f
            .seek(SeekFrom::Start(off))
            .and_then(|_| f.write(buf))
            .inspect_err(|_| self.disk.release(most))?;
        let after = f.metadata().map_or(before + most, |m| allocated(&m));
        let used = after.saturating_sub(before);
        if used > most {
            // Filesystem metadata (indirect blocks) beyond the estimate.
            self.disk.force_charge(used - most);
        } else {
            self.disk.release(most - used);
        }
        Ok(written)
    }

    fn create(&mut self, rel: &str, mode: u32) -> io::Result<()> {
        self.deny_if_ro()?;
        self.guard(Op::Make, rel, true, true)?;
        self.with_entry_charge(|fs| fs.create_entry(rel, mode))
    }

    fn mkdir(&mut self, rel: &str, mode: u32) -> io::Result<()> {
        self.deny_if_ro()?;
        self.guard(Op::MakeDir, rel, false, false)?;
        self.with_entry_charge(|fs| fs.mkdir_entry(rel, mode))
    }

    fn mknod(&mut self, rel: &str, mode: u32) -> io::Result<()> {
        self.deny_if_ro()?;
        self.guard(Op::Make, rel, false, false)?;
        self.with_entry_charge(|fs| fs.mknod_entry(rel, mode))
    }

    fn symlink(&mut self, target: &str, linkpath: &str) -> io::Result<()> {
        self.deny_if_ro()?;
        self.guard(Op::Make, linkpath, false, false)?;
        self.with_entry_charge(|fs| fs.symlink_entry(target, linkpath))
    }

    fn link(&mut self, old_rel: &str, new_rel: &str) -> io::Result<()> {
        self.deny_if_ro()?;
        self.guard(Op::LinkSource, old_rel, false, false)?;
        self.guard(Op::Make, new_rel, false, false)?;
        self.with_entry_charge(|fs| fs.link_entry(old_rel, new_rel))
    }

    fn set_times(&mut self, rel: &str, atime: SetTime, mtime: SetTime) -> io::Result<()> {
        self.deny_if_ro()?;
        self.guard(Op::Attr, rel, false, false)?;
        let (parent, name) = self.resolve(rel, false, false)?;
        let cname = cstr(&name)?;
        raw_utimensat(parent.as_raw_fd(), &cname, atime, mtime)
    }

    fn set_mode(&mut self, rel: &str, mode: u32) -> io::Result<()> {
        self.deny_if_ro()?;
        self.guard(Op::Attr, rel, false, false)?;
        let (parent, name) = self.resolve(rel, false, false)?;
        let cname = cstr(&name)?;
        raw_fchmodat(parent.as_raw_fd(), &cname, mode)
    }

    fn set_owner(&mut self, rel: &str, uid: Option<u32>, gid: Option<u32>) -> io::Result<()> {
        self.deny_if_ro()?;
        self.guard(Op::Attr, rel, false, false)?;
        let (parent, name) = self.resolve(rel, false, false)?;
        let cname = cstr(&name)?;
        raw_fchownat(parent.as_raw_fd(), &cname, uid, gid)
    }

    fn unlink(&mut self, rel: &str) -> io::Result<()> {
        self.deny_if_ro()?;
        self.guard(Op::Remove, rel, false, false)?;
        let (parent, name) = self.resolve(rel, false, false)?;
        let cname = cstr(&name)?;
        // unlink(2)/unlinkat(2) on a directory is nonstandard across
        // platforms — Linux reports EISDIR, but macOS/BSD reports EPERM
        // (verified empirically). Guests here model Linux, so probe first
        // (an O_NOFOLLOW-confined lookup, exactly like every other lookup in
        // this file) and report EISDIR consistently regardless of host OS.
        // The probe never follows a symlink, so a symlink *to* a directory
        // is correctly left unlink-able (it unlinks the link, not the
        // target) — only a real directory entry short-circuits here. A
        // benign race (the entry changing between this probe and the actual
        // `unlinkat` below) can only make the outcome match whatever is
        // really there at unlink time, never escape confinement.
        if let Ok(fd) = raw_openat(
            parent.as_raw_fd(),
            &cname,
            sys::O_RDONLY | sys::O_NOFOLLOW,
            0,
        ) && fs::File::from(fd).metadata().is_ok_and(|m| m.is_dir())
        {
            return Err(io::Error::from_raw_os_error(21)); // EISDIR
        }
        // Only the last name frees the file's data.
        let freed = if let Ok(fd) = raw_openat(
            parent.as_raw_fd(),
            &cname,
            sys::O_RDONLY | sys::O_NOFOLLOW | sys::O_NONBLOCK,
            0,
        ) {
            let meta = fs::File::from(fd).metadata();
            meta.map_or(0, |m| if m.nlink() == 1 { allocated(&m) } else { 0 })
        } else {
            0
        };
        raw_unlinkat(parent.as_raw_fd(), &cname, 0)?;
        self.disk.release(freed + ENTRY_COST);
        Ok(())
    }

    fn rmdir(&mut self, rel: &str) -> io::Result<()> {
        self.deny_if_ro()?;
        self.guard(Op::Remove, rel, false, false)?;
        let (parent, name) = self.resolve(rel, false, false)?;
        let cname = cstr(&name)?;
        raw_unlinkat(parent.as_raw_fd(), &cname, sys::AT_REMOVEDIR)?;
        self.disk.release(ENTRY_COST);
        Ok(())
    }

    fn truncate(&mut self, rel: &str, len: u64) -> io::Result<()> {
        self.deny_if_ro()?;
        self.guard(Op::Write, rel, true, false)?;
        let (parent, name) = self.resolve(rel, true, false)?;
        let cname = cstr(&name)?;
        // O_NONBLOCK guards against a FIFO's write-open blocking forever
        // waiting for a reader — with it, a FIFO either opens immediately or
        // fails ENXIO, and the type check below then turns any such
        // non-regular target into a deterministic EINVAL instead of a hang.
        let fd = raw_openat(
            parent.as_raw_fd(),
            &cname,
            sys::O_WRONLY | sys::O_NOFOLLOW | sys::O_NONBLOCK,
            0,
        )
        .map_err(normalize_errno)?;
        let file = fs::File::from(fd);
        let meta = file.metadata()?;
        if meta.is_dir() {
            // Defensive: opening a directory O_WRONLY already fails EISDIR
            // on both supported platforms (verified empirically), so this
            // is never actually reached, but it keeps the invariant
            // explicit rather than implicit in open()'s behavior.
            return Err(io::Error::from_raw_os_error(21)); // EISDIR
        }
        if !meta.file_type().is_file() {
            return Err(io::Error::from_raw_os_error(22)); // EINVAL: ftruncate is only meaningful on a regular file
        }
        // Growing leaves a hole, which uses no disk; shrinking frees blocks.
        let before = allocated(&meta);
        file.set_len(len)?;
        let after = file.metadata().map_or(before, |m| allocated(&m));
        self.disk.release(before.saturating_sub(after));
        Ok(())
    }

    fn check_write(&mut self, rel: &str) -> io::Result<()> {
        self.guard(Op::Write, rel, true, true)
    }

    fn readlink(&mut self, rel: &str) -> io::Result<String> {
        // The mount root is a directory, and `readlink` on any non-symlink is
        // EINVAL ("not a symbolic link") — never EISDIR. `resolve` reports the
        // rootless case as EISDIR (right for unlink/rename, wrong here), which
        // glibc's `realpath` treats as fatal: it readlinks every path prefix
        // to detect symlinks, so an EISDIR on `/work` breaks realpath for
        // everything under the passthrough. Return EINVAL directly for the root.
        if Self::split_rel(rel)?.is_empty() {
            return Err(io::Error::from_raw_os_error(22)); // EINVAL
        }
        let (parent, name) = self.resolve(rel, false, false)?;
        let cname = cstr(&name)?;
        raw_readlinkat(parent.as_raw_fd(), &cname)?.ok_or_else(|| {
            io::Error::from_raw_os_error(22) // EINVAL: not a symlink
        })
    }

    fn rename(&mut self, from: &str, to: &str) -> io::Result<()> {
        self.deny_if_ro()?;
        self.guard(Op::Remove, from, false, false)?;
        self.guard(Op::Make, to, false, false)?;
        let (from_parent, from_name) = self.resolve(from, false, false)?;
        let (to_parent, to_name) = self.resolve(to, false, false)?;
        let cfrom = cstr(&from_name)?;
        let cto = cstr(&to_name)?;
        raw_renameat(from_parent.as_raw_fd(), &cfrom, to_parent.as_raw_fd(), &cto)
    }
}

/// The unit disk use is counted in: a filesystem block.
const BLOCK: u64 = 4096;

/// The disk a file uses: its allocated blocks (`st_blocks` is in 512-byte
/// units whatever the filesystem's block size).
fn allocated(m: &fs::Metadata) -> u64 {
    m.blocks().saturating_mul(512)
}

/// `ENOSPC`.
fn enospc() -> io::Error {
    io::Error::from_raw_os_error(28)
}

/// Whether the host filesystem holding `path` is down to its last 5% of space
/// (at most 1 GiB of that) or inodes. If it can't be told, it's assumed fine: the
/// host will still report a truly full disk itself.
fn host_is_low_on_space(path: &Path) -> bool {
    let Some(HostSpace {
        total,
        avail,
        files,
        favail,
    }) = host_space(path)
    else {
        return false;
    };
    let reserve = (total / 20).min(1 << 30);
    avail < reserve || (files > 0 && favail < files / 20)
}

/// What `statvfs` says of the host filesystem holding a path.
#[derive(Debug, Clone, Copy)]
pub struct HostSpace {
    /// Its size and what an unprivileged user may still use of it, in bytes.
    pub total: u64,
    pub avail: u64,
    /// Its inodes, and those still free to such a user.
    pub files: u64,
    pub favail: u64,
}

/// [`HostSpace`] for the host filesystem holding `path`, if it can be told.
#[must_use]
pub fn host_space(path: &Path) -> Option<HostSpace> {
    let cpath = CString::new(path.as_os_str().as_encoded_bytes()).ok()?;
    let host::VfsStat {
        frsize,
        blocks,
        bavail,
        files,
        favail,
    } = host::statvfs(&cpath).ok()?;
    Some(HostSpace {
        total: blocks.saturating_mul(frsize),
        avail: bavail.saturating_mul(frsize),
        files,
        favail,
    })
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

    fn with_quota(tag: &str, limit: u64) -> (TempDir, Passthrough, Arc<Quota>) {
        let tmp = TempDir::new(tag);
        let mut pt = Passthrough::new(tmp.0.clone());
        let quota = Quota::new(limit);
        pt.attach_disk_quota(&quota);
        (tmp, pt, quota)
    }

    #[test]
    fn writes_past_the_disk_quota_fail_and_nothing_reaches_the_host() {
        let (_tmp, mut pt, quota) = with_quota("quota-write", 64 * 1024);
        pt.create("f", 0o644).unwrap();
        pt.write_at("f", 0, &vec![0; 1024]).unwrap();
        let err = pt.write_at("f", 1024, &vec![0; 1 << 20]).unwrap_err();
        assert_eq!(err.raw_os_error(), Some(28));
        assert_eq!(pt.stat("f").unwrap().size, 1024);
        let held = quota.used();
        assert!(
            (ENTRY_COST + 1..=ENTRY_COST + 2 * BLOCK).contains(&held),
            "{held}"
        );
        // Overwriting in place uses no more disk.
        pt.write_at("f", 0, &vec![1; 1024]).unwrap();
        assert_eq!(quota.used(), held);
    }

    #[test]
    fn a_sparse_file_costs_only_what_is_written() {
        let (_tmp, mut pt, quota) = with_quota("quota-truncate", 64 * 1024);
        pt.create("f", 0o644).unwrap();
        // Far past the quota, as a hole: no disk used, as on Linux.
        pt.truncate("f", 1 << 33).unwrap();
        assert_eq!(pt.stat("f").unwrap().size, 1 << 33);
        assert_eq!(quota.used(), ENTRY_COST);
        pt.write_at("f", 1 << 32, &[7; 8192]).unwrap();
        let held = quota.used();
        assert!(
            (ENTRY_COST + 8192..=ENTRY_COST + 32 * 1024).contains(&held),
            "{held}"
        );
        // Shrinking frees the written blocks.
        pt.truncate("f", 0).unwrap();
        assert_eq!(quota.used(), ENTRY_COST);
    }

    #[test]
    fn deleting_refunds_only_when_the_last_name_goes() {
        let (_tmp, mut pt, quota) = with_quota("quota-unlink", 1 << 20);
        pt.create("f", 0o644).unwrap();
        pt.write_at("f", 0, &vec![0; 4096]).unwrap();
        pt.link("f", "g").unwrap();
        let held = quota.used();
        pt.unlink("f").unwrap();
        assert_eq!(quota.used(), held - ENTRY_COST, "the data is still there");
        pt.unlink("g").unwrap();
        assert_eq!(quota.used(), 0);
    }

    #[test]
    fn entries_cannot_be_created_without_bound() {
        let (_tmp, mut pt, quota) = with_quota("quota-entries", 3 * ENTRY_COST);
        pt.mkdir("d", 0o755).unwrap();
        pt.create("f", 0o644).unwrap();
        pt.symlink("t", "l").unwrap();
        let full = |r: io::Result<()>| r.unwrap_err().raw_os_error() == Some(28);
        assert!(full(pt.create("g", 0o644)));
        assert!(full(pt.mkdir("e", 0o755)));
        assert!(full(pt.link("f", "h")));
        pt.rmdir("d").unwrap();
        pt.mkdir("e", 0o755).unwrap();
        // A failed create gives its charge back.
        pt.rmdir("e").unwrap();
        assert_eq!(pt.mkdir("f", 0o755).unwrap_err().raw_os_error(), Some(17));
        assert_eq!(quota.used(), 2 * ENTRY_COST);
    }

    #[test]
    fn the_host_free_space_check_runs_here() {
        // This filesystem has room: the guard must not refuse.
        let tmp = TempDir::new("low-space");
        assert!(!host_is_low_on_space(&tmp.0));
        assert!(!host_is_low_on_space(Path::new("/no/such/path")));
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
        let names: Vec<_> = pt
            .readdir("")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
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

    #[test]
    fn legit_file_resolves_and_reads() {
        let tmp = TempDir::new("legit");
        fs::write(tmp.0.join("real.txt"), b"hello world").unwrap();
        let mut pt = Passthrough::new(tmp.0.clone());
        let mut buf = [0u8; 11];
        assert_eq!(pt.read_at("real.txt", 0, &mut buf).unwrap(), 11);
        assert_eq!(&buf, b"hello world");
        assert_eq!(pt.stat("real.txt").unwrap().size, 11);
    }

    #[test]
    fn in_tree_relative_symlink_resolves() {
        let tmp = TempDir::new("intree");
        fs::write(tmp.0.join("target.txt"), b"symlinked data").unwrap();
        std::os::unix::fs::symlink("target.txt", tmp.0.join("link.txt")).unwrap();

        let mut pt = Passthrough::new(tmp.0.clone());
        let mut buf = [0u8; 14];
        assert_eq!(pt.read_at("link.txt", 0, &mut buf).unwrap(), 14);
        assert_eq!(&buf, b"symlinked data");

        // `stat` on the link itself reports a symlink (lstat semantics);
        // `readlink` returns the stored target text.
        assert_eq!(pt.stat("link.txt").unwrap().kind, NodeKind::Symlink);
        assert_eq!(pt.readlink("link.txt").unwrap(), "target.txt");
    }

    #[test]
    fn a_symlink_mid_path_is_followed_and_kept_confined() {
        let tmp = TempDir::new("mid-path");
        fs::create_dir(tmp.0.join("real")).unwrap();
        fs::write(tmp.0.join("real/f"), b"in tree").unwrap();
        fs::write(tmp.0.join("plain"), b"").unwrap();
        std::os::unix::fs::symlink("real", tmp.0.join("rel")).unwrap();
        std::os::unix::fs::symlink("/real", tmp.0.join("abs")).unwrap();
        std::os::unix::fs::symlink("/usr", tmp.0.join("sys")).unwrap();

        let mut pt = Passthrough::new(tmp.0.clone());
        let mut buf = [0u8; 7];
        assert_eq!(pt.read_at("rel/f", 0, &mut buf).unwrap(), 7);
        assert_eq!(&buf, b"in tree");
        pt.write_at("abs/f", 0, b"IN").unwrap();
        assert_eq!(fs::read(tmp.0.join("real/f")).unwrap(), b"IN tree");
        assert_eq!(pt.readdir("rel").unwrap().len(), 1);
        // Re-rooted under the mount: no `usr/bin/env` there (only `sys`).
        assert_eq!(
            pt.read_at("sys/bin/env", 0, &mut buf)
                .unwrap_err()
                .raw_os_error(),
            Some(2)
        );
        // A real non-directory mid-path is still ENOTDIR.
        assert_eq!(
            pt.read_at("plain/x", 0, &mut buf)
                .unwrap_err()
                .raw_os_error(),
            Some(20)
        );
    }

    #[test]
    fn absolute_symlink_target_is_not_followed_out_of_root() {
        let tmp = TempDir::new("abs-escape");
        std::os::unix::fs::symlink("/etc/passwd", tmp.0.join("evil")).unwrap();

        let mut pt = Passthrough::new(tmp.0.clone());
        // Confined resolution re-anchors an absolute symlink target under
        // the mount root, so "etc/passwd" is looked up *inside* the empty
        // temp root (where it doesn't exist), never the host's real
        // `/etc/passwd`.
        let mut buf = [0u8; 16];
        let err = pt.read_at("evil", 0, &mut buf).unwrap_err();
        assert_eq!(err.raw_os_error(), Some(2)); // ENOENT under the root, not host /etc

        // Belt and suspenders: if it ever did somehow open something, it
        // must not be the real host file's contents.
        assert!(pt.read_at("evil", 0, &mut buf).is_err());
    }

    #[test]
    fn symlink_ascending_above_root_does_not_escape() {
        let tmp = TempDir::new("ascend-escape");
        // Many levels of ".." followed by a real host path — if resolution
        // let a symlink target's ".." walk above the dirfd stack's root
        // frame, this would read the host's real /etc/passwd.
        std::os::unix::fs::symlink("../../../../../../../../../../etc/passwd", tmp.0.join("up"))
            .unwrap();

        let mut pt = Passthrough::new(tmp.0.clone());
        let mut buf = [0u8; 16];
        let err = pt.read_at("up", 0, &mut buf).unwrap_err();
        // ".." is clamped at the root frame, so this resolves to
        // "etc/passwd" *inside* the temp root, which doesn't exist.
        assert_eq!(err.raw_os_error(), Some(2)); // ENOENT under the root

        // A bare "../.." with nothing past it also must not escape (it
        // clamps to the root itself, which has no filename component).
        std::os::unix::fs::symlink("../..", tmp.0.join("up2")).unwrap();
        assert!(pt.read_at("up2", 0, &mut buf).is_err());
    }

    #[test]
    fn dotdot_ascending_above_root_is_rejected() {
        let tmp = TempDir::new("dotdot");
        let mut pt = Passthrough::new(tmp.0.clone());
        assert!(pt.stat("../etc/passwd").is_none());
        assert!(pt.read_at("../../secret", 0, &mut [0u8; 1]).is_err());
        assert!(pt.stat("sub/../../escape").is_none());
    }

    #[test]
    fn unlink_rmdir_rename_and_symlink_roundtrip() {
        let tmp = TempDir::new("ops");
        let mut pt = Passthrough::new(tmp.0.clone());
        pt.create("a.txt", 0o644).unwrap();
        pt.write_at("a.txt", 0, b"x").unwrap();
        pt.rename("a.txt", "b.txt").unwrap();
        assert!(pt.stat("a.txt").is_none());
        assert_eq!(pt.stat("b.txt").unwrap().size, 1);

        pt.mkdir("d", 0o755).unwrap();
        pt.rmdir("d").unwrap();
        assert!(pt.stat("d").is_none());

        pt.symlink("b.txt", "c.txt").unwrap();
        assert_eq!(pt.readlink("c.txt").unwrap(), "b.txt");
        pt.unlink("c.txt").unwrap();
        assert!(pt.stat("c.txt").is_none());
        assert!(pt.stat("b.txt").is_some());
    }

    #[test]
    fn link_makes_a_real_hard_link() {
        let tmp = TempDir::new("link");
        let mut pt = Passthrough::new(tmp.0.clone());
        pt.create("orig", 0o644).unwrap();
        pt.write_at("orig", 0, b"data").unwrap();
        pt.link("orig", "hard").unwrap();
        // Same inode, nlink == 2 (a real hard link, not a copy).
        let a = pt.stat("orig").unwrap();
        let b = pt.stat("hard").unwrap();
        assert_eq!(a.inode, b.inode, "shared inode");
        assert_eq!(a.nlink, 2, "st_nlink reflects the extra link");
        // Write-through: a write via one link is visible via the other.
        pt.write_at("hard", 0, b"XXXX").unwrap();
        let mut buf = [0u8; 4];
        pt.read_at("orig", 0, &mut buf).unwrap();
        assert_eq!(&buf, b"XXXX");
        // Linking onto an existing name fails EEXIST; a missing source ENOENT.
        assert_eq!(
            pt.link("orig", "hard").unwrap_err().raw_os_error(),
            Some(17)
        );
        assert_eq!(pt.link("nope", "x").unwrap_err().raw_os_error(), Some(2));
    }

    #[test]
    fn readlink_on_non_symlink_and_root_is_einval() {
        let tmp = TempDir::new("readlink");
        let mut pt = Passthrough::new(tmp.0.clone());
        pt.create("file", 0o644).unwrap();
        pt.mkdir("dir", 0o755).unwrap();
        // A regular file, a directory, and the mount root are all "not a
        // symbolic link" → EINVAL (22), never EISDIR — glibc's realpath
        // readlinks every path prefix and treats a non-EINVAL error as fatal.
        for p in ["file", "dir", "", "/"] {
            let e = pt.readlink(p).unwrap_err();
            assert_eq!(
                e.raw_os_error(),
                Some(22),
                "readlink({p:?}) should be EINVAL"
            );
        }
    }

    #[test]
    fn full_rw_roundtrip_create_write_read_stat_truncate_unlink() {
        let tmp = TempDir::new("full-rw");
        let mut pt = Passthrough::new(tmp.0.clone());

        pt.create("f.txt", 0o640).unwrap();
        assert_eq!(pt.stat("f.txt").unwrap().size, 0);
        assert_eq!(pt.stat("f.txt").unwrap().mode & 0o777, 0o640);

        assert_eq!(pt.write_at("f.txt", 0, b"hello world").unwrap(), 11);
        let mut buf = [0u8; 11];
        assert_eq!(pt.read_at("f.txt", 0, &mut buf).unwrap(), 11);
        assert_eq!(&buf, b"hello world");
        let attrs = pt.stat("f.txt").unwrap();
        assert_eq!(attrs.size, 11);
        assert_eq!(attrs.kind, NodeKind::File);

        pt.truncate("f.txt", 5).unwrap();
        assert_eq!(pt.stat("f.txt").unwrap().size, 5);
        let mut buf5 = [0u8; 5];
        assert_eq!(pt.read_at("f.txt", 0, &mut buf5).unwrap(), 5);
        assert_eq!(&buf5, b"hello");

        pt.unlink("f.txt").unwrap();
        assert!(pt.stat("f.txt").is_none());

        // The host reflects every step directly.
        assert!(!tmp.0.join("f.txt").exists());
    }

    #[test]
    fn truncate_shrinks_and_grows() {
        let tmp = TempDir::new("trunc");
        let mut pt = Passthrough::new(tmp.0.clone());
        pt.create("g.txt", 0o644).unwrap();
        pt.write_at("g.txt", 0, b"abcdefghij").unwrap();
        assert_eq!(pt.stat("g.txt").unwrap().size, 10);

        // Shrink.
        pt.truncate("g.txt", 4).unwrap();
        assert_eq!(pt.stat("g.txt").unwrap().size, 4);
        let mut buf = [0u8; 4];
        assert_eq!(pt.read_at("g.txt", 0, &mut buf).unwrap(), 4);
        assert_eq!(&buf, b"abcd");

        // Grow: the extension must read back as zero bytes (a hole), and
        // the host file must really be that long.
        pt.truncate("g.txt", 10).unwrap();
        assert_eq!(pt.stat("g.txt").unwrap().size, 10);
        let mut tail = [0xffu8; 6];
        assert_eq!(pt.read_at("g.txt", 4, &mut tail).unwrap(), 6);
        assert_eq!(&tail, &[0u8; 6]);
        assert_eq!(fs::metadata(tmp.0.join("g.txt")).unwrap().len(), 10);
    }

    #[test]
    fn write_at_offset_beyond_eof_extends_file() {
        let tmp = TempDir::new("write-extend");
        let mut pt = Passthrough::new(tmp.0.clone());
        pt.create("h.txt", 0o644).unwrap();
        pt.write_at("h.txt", 0, b"abc").unwrap();
        assert_eq!(pt.write_at("h.txt", 8, b"xyz").unwrap(), 3);
        assert_eq!(pt.stat("h.txt").unwrap().size, 11);
        let mut buf = [0u8; 11];
        assert_eq!(pt.read_at("h.txt", 0, &mut buf).unwrap(), 11);
        assert_eq!(&buf[..3], b"abc");
        assert_eq!(&buf[3..8], &[0u8; 5]);
        assert_eq!(&buf[8..], b"xyz");
    }

    #[test]
    fn truncate_on_directory_fails_eisdir() {
        let tmp = TempDir::new("trunc-dir");
        let mut pt = Passthrough::new(tmp.0.clone());
        pt.mkdir("d", 0o755).unwrap();
        let err = pt.truncate("d", 0).unwrap_err();
        assert_eq!(err.raw_os_error(), Some(21)); // EISDIR
    }

    #[test]
    fn read_only_rejects_all_mutators_with_erofs() {
        let tmp = TempDir::new("ro-full");
        fs::write(tmp.0.join("x"), b"hi").unwrap();
        fs::create_dir(tmp.0.join("d")).unwrap();
        let mut pt = Passthrough::read_only(tmp.0.clone());

        assert_eq!(
            pt.create("new.txt", 0o644).unwrap_err().raw_os_error(),
            Some(30) // EROFS
        );
        assert_eq!(
            pt.write_at("x", 0, b"no").unwrap_err().raw_os_error(),
            Some(30)
        );
        assert_eq!(pt.truncate("x", 0).unwrap_err().raw_os_error(), Some(30));
        assert_eq!(pt.unlink("x").unwrap_err().raw_os_error(), Some(30));
        assert_eq!(pt.rmdir("d").unwrap_err().raw_os_error(), Some(30));
        assert_eq!(pt.mkdir("d2", 0o755).unwrap_err().raw_os_error(), Some(30));
        assert_eq!(
            pt.symlink("x", "link").unwrap_err().raw_os_error(),
            Some(30)
        );
        assert_eq!(pt.rename("x", "y").unwrap_err().raw_os_error(), Some(30));

        // Nothing was actually touched on the host.
        assert_eq!(fs::read(tmp.0.join("x")).unwrap(), b"hi");
        assert!(tmp.0.join("d").is_dir());

        // Reads still work.
        let mut buf = [0u8; 2];
        assert_eq!(pt.read_at("x", 0, &mut buf).unwrap(), 2);
        assert_eq!(pt.stat("x").unwrap().size, 2);
    }

    #[test]
    fn rmdir_nonempty_fails_enotempty() {
        let tmp = TempDir::new("rmdir-nonempty");
        let mut pt = Passthrough::new(tmp.0.clone());
        pt.mkdir("d", 0o755).unwrap();
        pt.create("d/child.txt", 0o644).unwrap();

        let err = pt.rmdir("d").unwrap_err();
        assert_eq!(err.raw_os_error(), Some(39)); // ENOTEMPTY (host-normalized to the Linux value)

        // Still there, still a dir with its child.
        assert_eq!(pt.stat("d").unwrap().kind, NodeKind::Dir);
        assert!(pt.stat("d/child.txt").is_some());

        // Emptying it makes rmdir succeed.
        pt.unlink("d/child.txt").unwrap();
        pt.rmdir("d").unwrap();
        assert!(pt.stat("d").is_none());
    }

    #[test]
    fn unlink_on_directory_fails_eisdir() {
        let tmp = TempDir::new("unlink-dir");
        let mut pt = Passthrough::new(tmp.0.clone());
        pt.mkdir("d", 0o755).unwrap();

        let err = pt.unlink("d").unwrap_err();
        assert_eq!(err.raw_os_error(), Some(21)); // EISDIR (normalized; host may report EPERM)

        // Still there.
        assert_eq!(pt.stat("d").unwrap().kind, NodeKind::Dir);

        // A symlink *to* a directory is a different node — unlinking it
        // removes the link, not the target, and must still succeed.
        pt.symlink("d", "link_to_d").unwrap();
        pt.unlink("link_to_d").unwrap();
        assert!(pt.stat("link_to_d").is_none());
        assert!(pt.stat("d").is_some());
    }

    #[test]
    fn mkdir_existing_path_fails_eexist() {
        let tmp = TempDir::new("mkdir-eexist");
        let mut pt = Passthrough::new(tmp.0.clone());
        pt.mkdir("d", 0o755).unwrap();
        let err = pt.mkdir("d", 0o755).unwrap_err();
        assert_eq!(err.raw_os_error(), Some(17)); // EEXIST
    }

    #[test]
    fn absolute_symlink_escape_is_not_followed_on_write_or_truncate() {
        let tmp = TempDir::new("write-escape");
        std::os::unix::fs::symlink("/etc/passwd", tmp.0.join("evil")).unwrap();
        let mut pt = Passthrough::new(tmp.0.clone());

        // Confined resolution re-anchors "/etc/passwd" under the mount
        // root, so it's looked up inside the (otherwise empty) temp root,
        // where it doesn't exist — never the host's real /etc/passwd.
        let err = pt.write_at("evil", 0, b"pwned").unwrap_err();
        assert_eq!(err.raw_os_error(), Some(2)); // ENOENT under the root
        let err = pt.truncate("evil", 0).unwrap_err();
        assert_eq!(err.raw_os_error(), Some(2));

        // The real host file must be untouched.
        assert_ne!(fs::read_to_string("/etc/passwd").unwrap().len(), 0);
    }

    #[test]
    fn ascending_symlink_escape_is_not_followed_on_write_or_truncate() {
        let tmp = TempDir::new("write-ascend-escape");
        std::os::unix::fs::symlink("../../../../../../../../../../etc/passwd", tmp.0.join("up"))
            .unwrap();
        let mut pt = Passthrough::new(tmp.0.clone());

        let err = pt.write_at("up", 0, b"pwned").unwrap_err();
        assert_eq!(err.raw_os_error(), Some(2)); // resolves under the root, not host /etc
        let err = pt.truncate("up", 0).unwrap_err();
        assert_eq!(err.raw_os_error(), Some(2));
    }
}
