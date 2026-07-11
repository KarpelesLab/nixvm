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
//! so they are declared by hand below (see the security note). The `unsafe` is
//! confined to those thin FFI shims.
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
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::fs;
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd, OwnedFd, RawFd};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::PathBuf;

use super::{Attrs, DirEntry, MountFs, NodeKind};

/// `openat(2)`-family flag and errno values `std` doesn't expose. Declared
/// by hand (rather than depending on the `libc` crate, which is not a
/// dependency of this crate) directly from each platform's headers
/// (`<fcntl.h>` / `<sys/errno.h>`); these are stable, documented ABI
/// constants, not implementation details.
#[cfg(target_os = "linux")]
mod sys {
    pub const O_RDONLY: c_int = 0o0;
    pub const O_WRONLY: c_int = 0o1;
    pub const O_CREAT: c_int = 0o100;
    pub const O_TRUNC: c_int = 0o1000;
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
    pub const ELOOP: i32 = 40;

    use std::ffi::c_int;
}

#[cfg(target_os = "macos")]
mod sys {
    pub const O_RDONLY: c_int = 0x0000;
    pub const O_WRONLY: c_int = 0x0001;
    pub const O_CREAT: c_int = 0x0200;
    pub const O_TRUNC: c_int = 0x0400;
    pub const O_DIRECTORY: c_int = 0x0010_0000;
    pub const O_NOFOLLOW: c_int = 0x0000_0100;
    pub const O_CLOEXEC: c_int = 0x0100_0000;
    /// macOS-only: open the symlink itself rather than its target (there is
    /// no `O_PATH` on macOS, so this is the equivalent for getting an fd we
    /// can `fstat` to `lstat` a symlink).
    pub const O_SYMLINK: c_int = 0x0020_0000;
    pub const AT_REMOVEDIR: c_int = 0x0080;
    pub const ELOOP: i32 = 62;

    use std::ffi::c_int;
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
compile_error!(
    "passthrough: openat(2) flag/errno constants are only defined for linux and macos"
);

// The handful of dirfd-relative syscalls confined resolution needs. `std`
// exposes none of these; declaring exactly these six by hand keeps this
// crate free of a `libc` dependency. Argument types mirror each function's
// POSIX prototype; `mode` is always passed (even where the callee ignores
// it, e.g. `openat` without `O_CREAT`) since C's variadic argument-promotion
// rules mean a real caller would pass a promoted (register-width) value
// there too.
unsafe extern "C" {
    fn openat(dirfd: c_int, path: *const c_char, flags: c_int, mode: u32) -> c_int;
    fn unlinkat(dirfd: c_int, path: *const c_char, flags: c_int) -> c_int;
    fn mkdirat(dirfd: c_int, path: *const c_char, mode: u32) -> c_int;
    fn symlinkat(target: *const c_char, dirfd: c_int, linkpath: *const c_char) -> c_int;
    fn renameat(olddirfd: c_int, oldpath: *const c_char, newdirfd: c_int, newpath: *const c_char)
    -> c_int;
    fn readlinkat(dirfd: c_int, path: *const c_char, buf: *mut c_char, bufsiz: usize) -> isize;
    /// Takes ownership of `fd` (POSIX: on success the fd must not be used or
    /// closed independently afterward — only via `closedir`).
    fn fdopendir(fd: c_int) -> *mut c_void;
    /// Returns `NULL` both at end-of-stream and on error; see
    /// [`list_dir_fd`] for how that ambiguity is handled here.
    fn readdir(dirp: *mut c_void) -> *mut c_void;
    fn closedir(dirp: *mut c_void) -> c_int;
}

/// Mirrors the *leading* fields of the platform's real `struct dirent`
/// (verified against `<sys/dirent.h>` for macOS; the standard glibc/musl
/// `dirent64` layout for Linux — stable, undocumented-but-unchanged-in-decades
/// ABI). Only used to compute the byte offset of `d_name` via `offset_of!`;
/// the trailing array's declared length is irrelevant to that offset (it's
/// the last field), so a 1-element stub is deliberately used instead of the
/// real (1024/256-byte) array size.
#[cfg(target_os = "macos")]
#[repr(C)]
#[allow(clippy::struct_field_names)] // names mirror the C `struct dirent` fields verbatim
struct RawDirentHeader {
    d_ino: u64,
    d_seekoff: u64,
    d_reclen: u16,
    d_namlen: u16,
    d_type: u8,
    d_name: [c_char; 1],
}

#[cfg(target_os = "linux")]
#[repr(C)]
#[allow(clippy::struct_field_names)] // names mirror the C `struct dirent64` fields verbatim
struct RawDirentHeader {
    d_ino: u64,
    d_off: i64,
    d_reclen: u16,
    d_type: u8,
    d_name: [c_char; 1],
}

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

fn raw_openat(dirfd: RawFd, name: &CStr, flags: c_int, mode: u32) -> io::Result<OwnedFd> {
    let fd = unsafe { openat(dirfd, name.as_ptr(), flags | sys::O_CLOEXEC, mode) };
    if fd < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(unsafe { OwnedFd::from_raw_fd(fd) })
    }
}

/// `Ok(Some(target))` if `name` is a symlink, `Ok(None)` if it exists but
/// isn't one (`EINVAL`), `Err` for any other failure (including not found).
fn raw_readlinkat(dirfd: RawFd, name: &CStr) -> io::Result<Option<String>> {
    let mut buf = [0u8; 4096];
    let n = unsafe { readlinkat(dirfd, name.as_ptr(), buf.as_mut_ptr().cast::<c_char>(), buf.len()) };
    if n < 0 {
        let err = io::Error::last_os_error();
        if err.raw_os_error() == Some(22) {
            return Ok(None); // EINVAL: exists, but isn't a symlink
        }
        return Err(err);
    }
    Ok(Some(String::from_utf8_lossy(&buf[..n as usize]).into_owned()))
}

fn raw_unlinkat(dirfd: RawFd, name: &CStr, flags: c_int) -> io::Result<()> {
    let r = unsafe { unlinkat(dirfd, name.as_ptr(), flags) };
    if r < 0 { Err(io::Error::last_os_error()) } else { Ok(()) }
}

fn raw_mkdirat(dirfd: RawFd, name: &CStr, mode: u32) -> io::Result<()> {
    let r = unsafe { mkdirat(dirfd, name.as_ptr(), mode) };
    if r < 0 { Err(io::Error::last_os_error()) } else { Ok(()) }
}

fn raw_symlinkat(target: &CStr, dirfd: RawFd, name: &CStr) -> io::Result<()> {
    let r = unsafe { symlinkat(target.as_ptr(), dirfd, name.as_ptr()) };
    if r < 0 { Err(io::Error::last_os_error()) } else { Ok(()) }
}

fn raw_renameat(from_dirfd: RawFd, from: &CStr, to_dirfd: RawFd, to: &CStr) -> io::Result<()> {
    let r = unsafe { renameat(from_dirfd, from.as_ptr(), to_dirfd, to.as_ptr()) };
    if r < 0 { Err(io::Error::last_os_error()) } else { Ok(()) }
}

/// Open an fd on a symlink *itself* (not its target) — the fd-based
/// equivalent of `lstat`, needed because `stat()` must describe the symlink
/// even though the confined walk otherwise only ever opens non-symlinks.
#[cfg(target_os = "linux")]
fn open_symlink_itself(parent: RawFd, name: &CStr) -> io::Result<OwnedFd> {
    raw_openat(parent, name, sys::O_PATH | sys::O_NOFOLLOW, 0)
}

#[cfg(target_os = "macos")]
fn open_symlink_itself(parent: RawFd, name: &CStr) -> io::Result<OwnedFd> {
    raw_openat(parent, name, sys::O_SYMLINK, 0)
}

/// RAII guard closing a `DIR *` (and, transitively, its underlying fd) on
/// every exit path, including early returns on error.
struct DirGuard(*mut c_void);
impl Drop for DirGuard {
    fn drop(&mut self) {
        unsafe {
            closedir(self.0);
        }
    }
}

/// List a directory we already hold open (and have therefore already
/// confined/verified via [`Passthrough::resolve`]) by its fd — never by
/// re-opening any path, which would reintroduce a symlink-swap TOCTOU
/// window. Consumes `fd`: `fdopendir` takes ownership of it on success.
fn list_dir_fd(fd: OwnedFd) -> io::Result<Vec<DirEntry>> {
    let raw = fd.into_raw_fd();
    let dirp = unsafe { fdopendir(raw) };
    if dirp.is_null() {
        let err = io::Error::last_os_error();
        // `fdopendir` failing leaves the fd unowned by it; reclaim it as an
        // `OwnedFd` purely so it gets closed on drop.
        drop(unsafe { OwnedFd::from_raw_fd(raw) });
        return Err(err);
    }
    // Keeps `raw` open (via `closedir` on drop) for the whole function, so
    // it's still valid for the per-entry `openat`/`fstat` lookups below —
    // `openat`/`fstat` on `raw` don't touch the `readdir` cursor, so using
    // it directly alongside the `DIR *` stream is safe.
    let guard = DirGuard(dirp);

    let name_offset = std::mem::offset_of!(RawDirentHeader, d_name);
    let mut names = Vec::new();
    loop {
        // POSIX: `readdir` returns NULL at end-of-stream *and* on error,
        // distinguished only by whether it left `errno` set. `std` has no
        // portable way to reset/inspect `errno` directly, so — like most
        // hand-rolled `readdir` loops — end-of-stream is assumed on every
        // NULL; a genuine mid-stream I/O error just truncates the listing
        // rather than surfacing as an `Err`. Confinement/TOCTOU-safety is
        // unaffected either way (`d_name` extraction below never touches
        // the actual filesystem).
        let entry = unsafe { readdir(dirp) };
        if entry.is_null() {
            break;
        }
        // Safety: `entry` is a non-null `dirent`/`dirent64` pointer just
        // returned by `readdir`; `d_name` is NUL-terminated within it.
        let name_ptr = unsafe { entry.cast::<u8>().add(name_offset).cast::<c_char>() };
        let name = unsafe { CStr::from_ptr(name_ptr) }
            .to_string_lossy()
            .into_owned();
        if name != "." && name != ".." {
            names.push(name);
        }
    }

    // Per-entry metadata via our own confined `stat`-style lookup (an
    // `openat(O_NOFOLLOW)` + `fstat`, falling back to the symlink-itself fd
    // for symlinks) — never a further path-based syscall.
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let cname = cstr(&name)?;
        let (kind, inode) = match raw_openat(raw, &cname, sys::O_RDONLY | sys::O_NOFOLLOW, 0) {
            Ok(entry_fd) => match fs::File::from(entry_fd).metadata() {
                Ok(m) => (kind_of(&m), m.ino()),
                Err(_) => (NodeKind::File, 0),
            },
            Err(e) if e.raw_os_error() == Some(sys::ELOOP) => match open_symlink_itself(raw, &cname)
                .and_then(|f| fs::File::from(f).metadata())
            {
                Ok(m) => (kind_of(&m), m.ino()),
                Err(_) => (NodeKind::Symlink, 0),
            },
            Err(_) => (NodeKind::File, 0),
        };
        out.push(DirEntry { name, kind, inode });
    }

    drop(guard); // closedir(dirp) — also closes `raw`.
    Ok(out)
}

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
            .open(&self.root)?;
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
        let mut queue = Self::split_rel(rel)?;
        if queue.is_empty() {
            // The mount root itself: none of the (parent, name) operations
            // apply meaningfully to it (there is no in-confinement parent
            // above the root).
            return Err(io::Error::from_raw_os_error(21)); // EISDIR
        }

        let mut stack: Vec<OwnedFd> = vec![self.root_fd()?];
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
                return Err(io::Error::from_raw_os_error(sys::ELOOP));
            }

            if name == ".." {
                if stack.len() > 1 {
                    stack.pop();
                }
                continue;
            }

            let is_last = queue.is_empty();
            let cname = cstr(&name)?;
            let parent_raw = stack.last().expect("root frame always present").as_raw_fd();

            if is_last && !follow_final {
                let parent = stack.pop().expect("root frame always present");
                return Ok((parent, name));
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
                        return Ok((parent, name));
                    }
                    stack.push(fd);
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
                    return Ok((parent, name));
                }
                Err(e) if e.raw_os_error() == Some(sys::ELOOP) => {
                    let target = raw_readlinkat(parent_raw, &cname)?
                        .ok_or_else(|| io::Error::from_raw_os_error(sys::ELOOP))?;
                    budget = budget
                        .checked_sub(1)
                        .ok_or_else(|| io::Error::from_raw_os_error(sys::ELOOP))?;
                    let (absolute, mut target_comps) = Self::split_target(&target);
                    if absolute {
                        stack.truncate(1);
                    }
                    while let Some(c) = target_comps.pop_back() {
                        queue.push_front(c);
                    }
                }
                Err(e) => return Err(e),
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
        mtime: m.mtime(),
        inode: m.ino(),
        nlink: m.nlink() as u32,
        rdev: m.rdev(),
    }
}

impl MountFs for Passthrough {
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
        match raw_openat(parent.as_raw_fd(), &cname, sys::O_RDONLY | sys::O_NOFOLLOW, 0) {
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
        let fd = raw_openat(parent.as_raw_fd(), &cname, sys::O_RDONLY | sys::O_NOFOLLOW, 0)?;
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
            )?
        };

        list_dir_fd(fd)
    }

    fn write_at(&mut self, rel: &str, off: u64, buf: &[u8]) -> io::Result<usize> {
        self.deny_if_ro()?;
        let (parent, name) = self.resolve(rel, true, false)?;
        let cname = cstr(&name)?;
        let fd = raw_openat(parent.as_raw_fd(), &cname, sys::O_WRONLY | sys::O_NOFOLLOW, 0)?;
        let mut f = fs::File::from(fd);
        f.seek(SeekFrom::Start(off))?;
        f.write(buf)
    }

    fn create(&mut self, rel: &str, mode: u32) -> io::Result<()> {
        self.deny_if_ro()?;
        let (parent, name) = self.resolve(rel, true, true)?;
        let cname = cstr(&name)?;
        let fd = raw_openat(
            parent.as_raw_fd(),
            &cname,
            sys::O_WRONLY | sys::O_CREAT | sys::O_TRUNC | sys::O_NOFOLLOW,
            mode & 0o777,
        )?;
        // openat's mode is subject to umask; chmod explicitly afterward to
        // match the exact requested bits, as the original `fs::File::create`
        // + `set_permissions` pair did.
        fs::File::from(fd).set_permissions(fs::Permissions::from_mode(mode & 0o777))
    }

    fn mkdir(&mut self, rel: &str, mode: u32) -> io::Result<()> {
        self.deny_if_ro()?;
        let (parent, name) = self.resolve(rel, false, false)?;
        let cname = cstr(&name)?;
        raw_mkdirat(parent.as_raw_fd(), &cname, mode & 0o777)?;
        let fd = raw_openat(
            parent.as_raw_fd(),
            &cname,
            sys::O_DIRECTORY | sys::O_RDONLY | sys::O_NOFOLLOW,
            0,
        )?;
        fs::File::from(fd).set_permissions(fs::Permissions::from_mode(mode & 0o777))
    }

    fn unlink(&mut self, rel: &str) -> io::Result<()> {
        self.deny_if_ro()?;
        let (parent, name) = self.resolve(rel, false, false)?;
        let cname = cstr(&name)?;
        raw_unlinkat(parent.as_raw_fd(), &cname, 0)
    }

    fn rmdir(&mut self, rel: &str) -> io::Result<()> {
        self.deny_if_ro()?;
        let (parent, name) = self.resolve(rel, false, false)?;
        let cname = cstr(&name)?;
        raw_unlinkat(parent.as_raw_fd(), &cname, sys::AT_REMOVEDIR)
    }

    fn truncate(&mut self, rel: &str, len: u64) -> io::Result<()> {
        self.deny_if_ro()?;
        let (parent, name) = self.resolve(rel, true, false)?;
        let cname = cstr(&name)?;
        let fd = raw_openat(parent.as_raw_fd(), &cname, sys::O_WRONLY | sys::O_NOFOLLOW, 0)?;
        fs::File::from(fd).set_len(len)
    }

    fn symlink(&mut self, target: &str, linkpath: &str) -> io::Result<()> {
        self.deny_if_ro()?;
        let (parent, name) = self.resolve(linkpath, false, false)?;
        let cname = cstr(&name)?;
        let ctarget = cstr(target)?;
        raw_symlinkat(&ctarget, parent.as_raw_fd(), &cname)
    }

    fn readlink(&mut self, rel: &str) -> io::Result<String> {
        let (parent, name) = self.resolve(rel, false, false)?;
        let cname = cstr(&name)?;
        raw_readlinkat(parent.as_raw_fd(), &cname)?.ok_or_else(|| {
            io::Error::from_raw_os_error(22) // EINVAL: not a symlink
        })
    }

    fn rename(&mut self, from: &str, to: &str) -> io::Result<()> {
        self.deny_if_ro()?;
        let (from_parent, from_name) = self.resolve(from, false, false)?;
        let (to_parent, to_name) = self.resolve(to, false, false)?;
        let cfrom = cstr(&from_name)?;
        let cto = cstr(&to_name)?;
        raw_renameat(
            from_parent.as_raw_fd(),
            &cfrom,
            to_parent.as_raw_fd(),
            &cto,
        )
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
        std::os::unix::fs::symlink(
            "../../../../../../../../../../etc/passwd",
            tmp.0.join("up"),
        )
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
}
