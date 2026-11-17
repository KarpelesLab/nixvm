//! The host calls behind [`super`]'s confined resolution: the dirfd-relative
//! `*at(2)` family, directory listing by fd, and `statvfs`.
//!
//! `std` exposes none of these. On a Linux-ABI host (`cfg(nixvm_linux)`:
//! glibc, musl, or the libc-free fullrust target) they are raw system calls
//! ([`crate::linux_raw`]); on other unixes (macOS) they are the libc functions,
//! declared by hand since the crate has no `libc` dependency.
//!
//! Each returns the host's own errno on failure (not yet normalized to Linux's —
//! callers compare some against the host's `sys::ELOOP` first).

use std::ffi::{CStr, c_int};
use std::io;
use std::os::fd::{OwnedFd, RawFd};

pub use imp::*;

/// What `statvfs` reports, in the units [`super::host_space`] needs.
#[derive(Debug, Clone, Copy)]
pub struct VfsStat {
    pub frsize: u64,
    pub blocks: u64,
    pub bavail: u64,
    pub files: u64,
    pub favail: u64,
}

#[cfg(nixvm_linux)]
mod imp {
    use super::{CStr, OwnedFd, RawFd, VfsStat, c_int, io};
    use crate::linux_raw as raw;
    use std::os::fd::AsRawFd;

    pub use raw::Timespec;

    pub fn openat(dirfd: RawFd, path: &CStr, flags: c_int, mode: u32) -> io::Result<OwnedFd> {
        raw::openat(dirfd, path, flags, mode)
    }
    pub fn unlinkat(dirfd: RawFd, path: &CStr, flags: c_int) -> io::Result<()> {
        raw::unlinkat(dirfd, path, flags)
    }
    pub fn mkdirat(dirfd: RawFd, path: &CStr, mode: u32) -> io::Result<()> {
        raw::mkdirat(dirfd, path, mode)
    }
    pub fn mknodat(dirfd: RawFd, path: &CStr, mode: u32) -> io::Result<()> {
        raw::mknodat(dirfd, path, mode, 0)
    }
    pub fn symlinkat(target: &CStr, dirfd: RawFd, path: &CStr) -> io::Result<()> {
        raw::symlinkat(target, dirfd, path)
    }
    pub fn renameat(olddir: RawFd, old: &CStr, newdir: RawFd, new: &CStr) -> io::Result<()> {
        raw::renameat(olddir, old, newdir, new)
    }
    pub fn linkat(olddir: RawFd, old: &CStr, newdir: RawFd, new: &CStr) -> io::Result<()> {
        raw::linkat(olddir, old, newdir, new, 0)
    }
    pub fn readlinkat(dirfd: RawFd, path: &CStr, buf: &mut [u8]) -> io::Result<usize> {
        raw::readlinkat(dirfd, path, buf)
    }
    pub fn utimensat(
        dirfd: RawFd,
        path: &CStr,
        times: &[Timespec; 2],
        flags: c_int,
    ) -> io::Result<()> {
        raw::utimensat(dirfd, path, times, flags)
    }
    pub fn fchmodat(dirfd: RawFd, path: &CStr, mode: u32) -> io::Result<()> {
        raw::fchmodat(dirfd, path, mode)
    }
    pub fn fchownat(dirfd: RawFd, path: &CStr, uid: u32, gid: u32, flags: c_int) -> io::Result<()> {
        raw::fchownat(dirfd, path, uid, gid, flags)
    }

    /// The names in the directory open on `fd` (`.` and `..` included), with a
    /// handle that keeps the descriptor open for per-entry lookups after.
    pub fn read_dir(fd: OwnedFd) -> io::Result<(OwnedFd, Vec<Vec<u8>>)> {
        let names = raw::dir_names(fd.as_raw_fd())?;
        Ok((fd, names))
    }

    /// `statvfs(3)` as glibc and musl compute it from `statfs(2)`.
    pub fn statvfs(path: &CStr) -> io::Result<VfsStat> {
        let s = raw::statfs(path)?;
        Ok(VfsStat {
            frsize: if s.frsize == 0 { s.bsize } else { s.frsize },
            blocks: s.blocks,
            bavail: s.bavail,
            files: s.files,
            favail: s.ffree,
        })
    }
}

#[cfg(not(nixvm_linux))]
mod imp {
    use super::{CStr, OwnedFd, RawFd, VfsStat, c_int, io};
    use std::ffi::{c_char, c_void};
    use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd};

    /// The kernel `struct timespec` (64-bit `tv_sec`/`tv_nsec`) passed to
    /// `utimensat`. Same layout on Linux and macOS for 64-bit targets.
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct Timespec {
        pub tv_sec: i64,
        pub tv_nsec: i64,
    }

    /// The libc functions, by their C names. Argument types mirror each
    /// function's POSIX prototype; `mode` is always passed (even where the
    /// callee ignores it, e.g. `openat` without `O_CREAT`) since C's variadic
    /// argument-promotion rules mean a real caller would pass a promoted
    /// (register-width) value there too.
    mod c {
        use super::Timespec;
        use std::ffi::{c_char, c_int, c_void};
        unsafe extern "C" {
            pub fn openat(dirfd: c_int, path: *const c_char, flags: c_int, mode: u32) -> c_int;
            pub fn unlinkat(dirfd: c_int, path: *const c_char, flags: c_int) -> c_int;
            pub fn statvfs(path: *const c_char, buf: *mut c_void) -> c_int;
            pub fn mkdirat(dirfd: c_int, path: *const c_char, mode: u32) -> c_int;
            pub fn symlinkat(target: *const c_char, dirfd: c_int, linkpath: *const c_char)
            -> c_int;
            pub fn renameat(
                olddirfd: c_int,
                oldpath: *const c_char,
                newdirfd: c_int,
                newpath: *const c_char,
            ) -> c_int;
            pub fn linkat(
                olddirfd: c_int,
                oldpath: *const c_char,
                newdirfd: c_int,
                newpath: *const c_char,
                flags: c_int,
            ) -> c_int;
            pub fn readlinkat(
                dirfd: c_int,
                path: *const c_char,
                buf: *mut c_char,
                bufsiz: usize,
            ) -> isize;
            pub fn utimensat(
                dirfd: c_int,
                path: *const c_char,
                times: *const Timespec,
                flags: c_int,
            ) -> c_int;
            pub fn fchmodat(dirfd: c_int, path: *const c_char, mode: u32, flags: c_int) -> c_int;
            pub fn fchownat(
                dirfd: c_int,
                path: *const c_char,
                owner: u32,
                group: u32,
                flags: c_int,
            ) -> c_int;
            pub fn mknodat(dirfd: c_int, path: *const c_char, mode: u32, dev: u64) -> c_int;
            /// Takes ownership of `fd` (POSIX: on success the fd must not be
            /// used or closed independently afterward — only via `closedir`).
            pub fn fdopendir(fd: c_int) -> *mut c_void;
            /// Returns `NULL` both at end-of-stream and on error; see
            /// [`super::read_dir`].
            pub fn readdir(dirp: *mut c_void) -> *mut c_void;
            pub fn closedir(dirp: *mut c_void) -> c_int;
        }
    }

    fn check(r: c_int) -> io::Result<()> {
        if r < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }

    // SAFETY (every call below): each `&CStr` is NUL-terminated and borrowed
    // for the call; each buffer is valid for the length passed with it.

    pub fn openat(dirfd: RawFd, path: &CStr, flags: c_int, mode: u32) -> io::Result<OwnedFd> {
        let fd = unsafe { c::openat(dirfd, path.as_ptr(), flags, mode) };
        if fd < 0 {
            Err(io::Error::last_os_error())
        } else {
            // SAFETY: a fresh descriptor owned by no one else.
            Ok(unsafe { OwnedFd::from_raw_fd(fd) })
        }
    }
    pub fn unlinkat(dirfd: RawFd, path: &CStr, flags: c_int) -> io::Result<()> {
        check(unsafe { c::unlinkat(dirfd, path.as_ptr(), flags) })
    }
    pub fn mkdirat(dirfd: RawFd, path: &CStr, mode: u32) -> io::Result<()> {
        check(unsafe { c::mkdirat(dirfd, path.as_ptr(), mode) })
    }
    pub fn mknodat(dirfd: RawFd, path: &CStr, mode: u32) -> io::Result<()> {
        check(unsafe { c::mknodat(dirfd, path.as_ptr(), mode, 0) })
    }
    pub fn symlinkat(target: &CStr, dirfd: RawFd, path: &CStr) -> io::Result<()> {
        check(unsafe { c::symlinkat(target.as_ptr(), dirfd, path.as_ptr()) })
    }
    pub fn renameat(olddir: RawFd, old: &CStr, newdir: RawFd, new: &CStr) -> io::Result<()> {
        check(unsafe { c::renameat(olddir, old.as_ptr(), newdir, new.as_ptr()) })
    }
    pub fn linkat(olddir: RawFd, old: &CStr, newdir: RawFd, new: &CStr) -> io::Result<()> {
        check(unsafe { c::linkat(olddir, old.as_ptr(), newdir, new.as_ptr(), 0) })
    }
    pub fn readlinkat(dirfd: RawFd, path: &CStr, buf: &mut [u8]) -> io::Result<usize> {
        let n = unsafe {
            c::readlinkat(
                dirfd,
                path.as_ptr(),
                buf.as_mut_ptr().cast::<c_char>(),
                buf.len(),
            )
        };
        if n < 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(n as usize)
        }
    }
    pub fn utimensat(
        dirfd: RawFd,
        path: &CStr,
        times: &[Timespec; 2],
        flags: c_int,
    ) -> io::Result<()> {
        check(unsafe { c::utimensat(dirfd, path.as_ptr(), times.as_ptr(), flags) })
    }
    pub fn fchmodat(dirfd: RawFd, path: &CStr, mode: u32) -> io::Result<()> {
        check(unsafe { c::fchmodat(dirfd, path.as_ptr(), mode, 0) })
    }
    pub fn fchownat(dirfd: RawFd, path: &CStr, uid: u32, gid: u32, flags: c_int) -> io::Result<()> {
        check(unsafe { c::fchownat(dirfd, path.as_ptr(), uid, gid, flags) })
    }

    /// Mirrors the *leading* fields of the platform's real `struct dirent`
    /// (verified against `<sys/dirent.h>` for macOS; the standard glibc/musl
    /// `dirent64` layout for Linux — stable, undocumented-but-unchanged-in-
    /// decades ABI). Only used to compute the byte offset of `d_name` via
    /// `offset_of!`; the trailing array's declared length is irrelevant to that
    /// offset (it's the last field), so a 1-element stub is deliberately used
    /// instead of the real (1024/256-byte) array size.
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

    #[cfg(not(target_os = "macos"))]
    #[repr(C)]
    #[allow(clippy::struct_field_names)] // names mirror the C `struct dirent64` fields verbatim
    struct RawDirentHeader {
        d_ino: u64,
        d_off: i64,
        d_reclen: u16,
        d_type: u8,
        d_name: [c_char; 1],
    }

    /// RAII guard closing a `DIR *` (and, transitively, its underlying fd) on
    /// every exit path, including early returns on error.
    #[derive(Debug)]
    pub struct DirGuard {
        dirp: *mut c_void,
        fd: RawFd,
    }
    impl AsRawFd for DirGuard {
        fn as_raw_fd(&self) -> RawFd {
            self.fd
        }
    }
    impl Drop for DirGuard {
        fn drop(&mut self) {
            // SAFETY: `dirp` came from `fdopendir` and is closed exactly once.
            unsafe {
                c::closedir(self.dirp);
            }
        }
    }

    /// The names in the directory open on `fd` (`.` and `..` included), with a
    /// handle that keeps the descriptor open for per-entry lookups after:
    /// `fdopendir` takes ownership of it, and `openat`/`fstat` on it don't
    /// touch the `readdir` cursor, so using it alongside the `DIR *` is safe.
    pub fn read_dir(fd: OwnedFd) -> io::Result<(DirGuard, Vec<Vec<u8>>)> {
        let raw = fd.into_raw_fd();
        // SAFETY: `raw` is an open directory descriptor we own.
        let dirp = unsafe { c::fdopendir(raw) };
        if dirp.is_null() {
            let err = io::Error::last_os_error();
            // `fdopendir` failing leaves the fd unowned by it; reclaim it as an
            // `OwnedFd` purely so it gets closed on drop.
            // SAFETY: still ours, as above.
            drop(unsafe { OwnedFd::from_raw_fd(raw) });
            return Err(err);
        }
        let guard = DirGuard { dirp, fd: raw };
        let name_offset = std::mem::offset_of!(RawDirentHeader, d_name);
        let mut names = Vec::new();
        loop {
            // POSIX: `readdir` returns NULL at end-of-stream *and* on error,
            // distinguished only by whether it left `errno` set. `std` has no
            // portable way to reset/inspect `errno` directly, so — like most
            // hand-rolled `readdir` loops — end-of-stream is assumed on every
            // NULL; a genuine mid-stream I/O error just truncates the listing
            // rather than surfacing as an `Err`.
            // SAFETY: `dirp` is a live `DIR *`.
            let entry = unsafe { c::readdir(dirp) };
            if entry.is_null() {
                break;
            }
            // SAFETY: `entry` is a non-null `dirent`/`dirent64` pointer just
            // returned by `readdir`; `d_name` is NUL-terminated within it.
            let name =
                unsafe { CStr::from_ptr(entry.cast::<u8>().add(name_offset).cast::<c_char>()) };
            names.push(name.to_bytes().to_vec());
        }
        Ok((guard, names))
    }

    pub fn statvfs(path: &CStr) -> io::Result<VfsStat> {
        // `struct statvfs`, whose start is all we read: f_bsize, f_frsize, then
        // f_blocks, f_bfree, f_bavail, f_files, f_ffree, f_favail. Those last six
        // are 64-bit on Linux and 32-bit on macOS. Oversized for either's whole
        // struct.
        let mut buf = [0u64; 64];
        // SAFETY: `path` is NUL-terminated, and `buf` is larger than `struct
        // statvfs` on every target here; the callee only writes within it.
        check(unsafe { c::statvfs(path.as_ptr(), buf.as_mut_ptr().cast()) })?;
        let frsize = buf[1];
        #[cfg(not(target_os = "macos"))]
        let (blocks, bavail, files, favail) = (buf[2], buf[4], buf[5], buf[7]);
        #[cfg(target_os = "macos")]
        let (blocks, bavail, files, favail) = {
            let field = |i: usize| {
                let word = buf[2 + i / 2];
                if i.is_multiple_of(2) {
                    word & 0xffff_ffff
                } else {
                    word >> 32
                }
            };
            (field(0), field(2), field(3), field(5))
        };
        Ok(VfsStat {
            frsize,
            blocks,
            bavail,
            files,
            favail,
        })
    }
}
