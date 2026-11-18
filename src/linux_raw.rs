//! Raw Linux system calls — the host OS layer on a Linux-ABI host, libc or not.
//!
//! nixvm declares no libc dependency, and what `std` doesn't wrap (the
//! dirfd-relative `*at(2)` family, `ioctl`, `setsockopt`, `mmap`, POSIX timers,
//! `rt_sigaction`) used to be reached through hand-declared `extern "C"` libc
//! symbols. That links on `*-linux-gnu`/`-musl`, but not on a libc-free target
//! such as `x86_64-unknown-linux-fullrust` — whose std is itself built on raw
//! syscalls and is neither `cfg(unix)` nor `target_os = "linux"`. So on any
//! host that speaks the Linux kernel ABI (`cfg(nixvm_linux)`, see `build.rs`)
//! those calls are made here, directly, with the `syscall` instruction on
//! x86-64 and `svc #0` on arm64. A normal Linux build takes this path too: the
//! kernel does exactly what the libc wrapper would have asked it to, so the
//! behaviour is the same, and there is one code path to test.
//!
//! Every wrapper returns `io::Result`, the error being the kernel's `-errno`
//! (never the thread's libc `errno`, which a raw call doesn't set — so no caller
//! may use `io::Error::last_os_error()` after one of these).
//!
//! This is one of the crate's documented `unsafe` sites; the `unsafe` is the
//! system call itself, with each wrapper's pointer arguments borrowed from safe
//! references for the call's duration.

#![allow(dead_code)] // each consumer is cfg-gated; not every build uses every call

use std::ffi::{CStr, c_int};
use std::io;
use std::os::fd::{FromRawFd, OwnedFd, RawFd};

/// System call numbers for the host arch.
#[cfg(target_arch = "x86_64")]
pub mod nr {
    pub const READ: usize = 0;
    pub const CLOSE: usize = 3;
    pub const LSEEK: usize = 8;
    pub const MMAP: usize = 9;
    pub const MUNMAP: usize = 11;
    pub const RT_SIGACTION: usize = 13;
    pub const RT_SIGRETURN: usize = 15;
    pub const IOCTL: usize = 16;
    pub const MADVISE: usize = 28;
    pub const SETSOCKOPT: usize = 54;
    pub const GETSOCKOPT: usize = 55;
    pub const STATFS: usize = 137;
    pub const GETTID: usize = 186;
    pub const GETDENTS64: usize = 217;
    pub const TIMER_CREATE: usize = 222;
    pub const TIMER_SETTIME: usize = 223;
    pub const TIMER_DELETE: usize = 226;
    pub const CLOCK_GETTIME: usize = 228;
    pub const OPENAT: usize = 257;
    pub const MKDIRAT: usize = 258;
    pub const MKNODAT: usize = 259;
    pub const FCHOWNAT: usize = 260;
    pub const UNLINKAT: usize = 263;
    pub const RENAMEAT: usize = 264;
    pub const LINKAT: usize = 265;
    pub const SYMLINKAT: usize = 266;
    pub const READLINKAT: usize = 267;
    pub const FCHMODAT: usize = 268;
    pub const UTIMENSAT: usize = 280;
}

/// System call numbers for the host arch (the asm-generic table).
#[cfg(target_arch = "aarch64")]
pub mod nr {
    pub const IOCTL: usize = 29;
    pub const MKNODAT: usize = 33;
    pub const MKDIRAT: usize = 34;
    pub const UNLINKAT: usize = 35;
    pub const SYMLINKAT: usize = 36;
    pub const LINKAT: usize = 37;
    pub const RENAMEAT: usize = 38;
    pub const STATFS: usize = 43;
    pub const FCHMODAT: usize = 53;
    pub const FCHOWNAT: usize = 54;
    pub const OPENAT: usize = 56;
    pub const CLOSE: usize = 57;
    pub const GETDENTS64: usize = 61;
    pub const LSEEK: usize = 62;
    pub const READ: usize = 63;
    pub const READLINKAT: usize = 78;
    pub const UTIMENSAT: usize = 88;
    pub const TIMER_CREATE: usize = 107;
    pub const TIMER_SETTIME: usize = 110;
    pub const TIMER_DELETE: usize = 111;
    pub const CLOCK_GETTIME: usize = 113;
    pub const RT_SIGACTION: usize = 134;
    pub const RT_SIGRETURN: usize = 139;
    pub const GETTID: usize = 178;
    pub const SETSOCKOPT: usize = 208;
    pub const GETSOCKOPT: usize = 209;
    pub const MUNMAP: usize = 215;
    pub const MMAP: usize = 222;
    pub const MADVISE: usize = 233;
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
compile_error!("linux_raw: raw system calls are only implemented for x86_64 and aarch64");

/// `AT_FDCWD`: resolve a relative path against the working directory.
pub const AT_FDCWD: c_int = -100;

/// Issue system call `n` with six arguments (unused ones are ignored by the
/// kernel), returning the raw result: a value, or `-errno` in `-4095..=-1`.
///
/// # Safety
/// The arguments must be valid for system call `n`: any pointer among them must
/// be live and point at memory of the size and mutability the call expects.
#[inline]
pub unsafe fn syscall6(n: usize, a: [usize; 6]) -> isize {
    let ret: isize;
    #[cfg(target_arch = "x86_64")]
    // SAFETY: the caller vouches for the arguments; the instruction clobbers only
    // rcx/r11 (declared) and returns in rax.
    unsafe {
        std::arch::asm!(
            "syscall",
            inlateout("rax") n as isize => ret,
            in("rdi") a[0],
            in("rsi") a[1],
            in("rdx") a[2],
            in("r10") a[3],
            in("r8") a[4],
            in("r9") a[5],
            lateout("rcx") _,
            lateout("r11") _,
            options(nostack),
        );
    }
    #[cfg(target_arch = "aarch64")]
    // SAFETY: as above; `svc #0` takes the number in x8 and returns in x0.
    unsafe {
        std::arch::asm!(
            "svc #0",
            in("x8") n,
            inlateout("x0") a[0] as isize => ret,
            in("x1") a[1],
            in("x2") a[2],
            in("x3") a[3],
            in("x4") a[4],
            in("x5") a[5],
            options(nostack),
        );
    }
    ret
}

/// A raw result as `io::Result`: `-errno` is the error, anything else the value.
pub fn cvt(ret: isize) -> io::Result<usize> {
    if (-4095..0).contains(&ret) {
        Err(io::Error::from_raw_os_error(-ret as i32))
    } else {
        Ok(ret as usize)
    }
}

/// [`syscall6`] with fewer arguments, mapped through [`cvt`].
///
/// # Safety
/// As [`syscall6`].
#[inline]
pub unsafe fn call(n: usize, args: &[usize]) -> io::Result<usize> {
    let mut a = [0usize; 6];
    a[..args.len()].copy_from_slice(args);
    // SAFETY: forwarded from the caller.
    cvt(unsafe { syscall6(n, a) })
}

fn fd_arg(fd: RawFd) -> usize {
    fd as isize as usize
}

fn path_arg(p: &CStr) -> usize {
    p.as_ptr() as usize
}

// ---- files -------------------------------------------------------------------

/// `openat(dirfd, path, flags, mode)`, owning the new descriptor.
pub fn openat(dirfd: RawFd, path: &CStr, flags: c_int, mode: u32) -> io::Result<OwnedFd> {
    // SAFETY: `path` is NUL-terminated and borrowed for the call.
    let fd = unsafe {
        call(
            nr::OPENAT,
            &[fd_arg(dirfd), path_arg(path), flags as usize, mode as usize],
        )
    }?;
    // SAFETY: a fresh descriptor the kernel just returned, owned by no one else.
    Ok(unsafe { OwnedFd::from_raw_fd(fd as RawFd) })
}

/// `close(fd)` of a descriptor not held by an `OwnedFd`.
pub fn close(fd: RawFd) -> io::Result<()> {
    // SAFETY: no pointers.
    unsafe { call(nr::CLOSE, &[fd_arg(fd)]) }.map(drop)
}

/// `lseek(fd, offset, whence)`: the new offset.
pub fn lseek(fd: RawFd, offset: u64, whence: c_int) -> io::Result<u64> {
    // SAFETY: no pointers.
    unsafe { call(nr::LSEEK, &[fd_arg(fd), offset as usize, whence as usize]) }.map(|r| r as u64)
}

pub fn unlinkat(dirfd: RawFd, path: &CStr, flags: c_int) -> io::Result<()> {
    // SAFETY: `path` is NUL-terminated and borrowed for the call.
    unsafe {
        call(
            nr::UNLINKAT,
            &[fd_arg(dirfd), path_arg(path), flags as usize],
        )
    }
    .map(drop)
}

pub fn mkdirat(dirfd: RawFd, path: &CStr, mode: u32) -> io::Result<()> {
    // SAFETY: `path` is NUL-terminated and borrowed for the call.
    unsafe { call(nr::MKDIRAT, &[fd_arg(dirfd), path_arg(path), mode as usize]) }.map(drop)
}

pub fn mknodat(dirfd: RawFd, path: &CStr, mode: u32, dev: u32) -> io::Result<()> {
    // SAFETY: `path` is NUL-terminated and borrowed for the call.
    unsafe {
        call(
            nr::MKNODAT,
            &[fd_arg(dirfd), path_arg(path), mode as usize, dev as usize],
        )
    }
    .map(drop)
}

pub fn symlinkat(target: &CStr, dirfd: RawFd, path: &CStr) -> io::Result<()> {
    // SAFETY: both strings are NUL-terminated and borrowed for the call.
    unsafe {
        call(
            nr::SYMLINKAT,
            &[path_arg(target), fd_arg(dirfd), path_arg(path)],
        )
    }
    .map(drop)
}

pub fn renameat(olddir: RawFd, old: &CStr, newdir: RawFd, new: &CStr) -> io::Result<()> {
    // SAFETY: both strings are NUL-terminated and borrowed for the call.
    unsafe {
        call(
            nr::RENAMEAT,
            &[fd_arg(olddir), path_arg(old), fd_arg(newdir), path_arg(new)],
        )
    }
    .map(drop)
}

pub fn linkat(
    olddir: RawFd,
    old: &CStr,
    newdir: RawFd,
    new: &CStr,
    flags: c_int,
) -> io::Result<()> {
    // SAFETY: both strings are NUL-terminated and borrowed for the call.
    unsafe {
        call(
            nr::LINKAT,
            &[
                fd_arg(olddir),
                path_arg(old),
                fd_arg(newdir),
                path_arg(new),
                flags as usize,
            ],
        )
    }
    .map(drop)
}

/// `readlinkat` into `buf`, returning the target's length (not NUL-terminated).
pub fn readlinkat(dirfd: RawFd, path: &CStr, buf: &mut [u8]) -> io::Result<usize> {
    // SAFETY: `path` is NUL-terminated; `buf` is `buf.len()` writable bytes.
    unsafe {
        call(
            nr::READLINKAT,
            &[
                fd_arg(dirfd),
                path_arg(path),
                buf.as_mut_ptr() as usize,
                buf.len(),
            ],
        )
    }
}

/// The kernel's `struct timespec` (64-bit fields on both arches).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Timespec {
    pub tv_sec: i64,
    pub tv_nsec: i64,
}

/// `utimensat(dirfd, path, [atime, mtime], flags)`.
pub fn utimensat(dirfd: RawFd, path: &CStr, times: &[Timespec; 2], flags: c_int) -> io::Result<()> {
    // SAFETY: `path` is NUL-terminated; `times` is two readable timespecs.
    unsafe {
        call(
            nr::UTIMENSAT,
            &[
                fd_arg(dirfd),
                path_arg(path),
                times.as_ptr() as usize,
                flags as usize,
            ],
        )
    }
    .map(drop)
}

/// `fchmodat(dirfd, path, mode)` (the system call has no flags argument; the
/// libc `flags = 0` call is exactly this).
pub fn fchmodat(dirfd: RawFd, path: &CStr, mode: u32) -> io::Result<()> {
    // SAFETY: `path` is NUL-terminated and borrowed for the call.
    unsafe {
        call(
            nr::FCHMODAT,
            &[fd_arg(dirfd), path_arg(path), mode as usize],
        )
    }
    .map(drop)
}

pub fn fchownat(dirfd: RawFd, path: &CStr, uid: u32, gid: u32, flags: c_int) -> io::Result<()> {
    // SAFETY: `path` is NUL-terminated and borrowed for the call.
    unsafe {
        call(
            nr::FCHOWNAT,
            &[
                fd_arg(dirfd),
                path_arg(path),
                uid as usize,
                gid as usize,
                flags as usize,
            ],
        )
    }
    .map(drop)
}

/// The names in the directory open on `fd`, read with `getdents64` from the
/// descriptor's current offset to the end (`.` and `..` included). An error
/// part-way through ends the listing early with what was read, as a libc
/// `readdir` loop (which can't tell end-of-stream from error) does.
pub fn dir_names(fd: RawFd) -> io::Result<Vec<Vec<u8>>> {
    // `struct linux_dirent64`: d_ino u64, d_off i64, d_reclen u16, d_type u8,
    // then the NUL-terminated name.
    const NAME_OFF: usize = 19;
    let mut buf = vec![0u8; 32 * 1024];
    let mut names = Vec::new();
    let mut first = true;
    loop {
        // SAFETY: `buf` is `buf.len()` writable bytes for the call.
        let n = match unsafe {
            call(
                nr::GETDENTS64,
                &[fd_arg(fd), buf.as_mut_ptr() as usize, buf.len()],
            )
        } {
            Ok(0) => break,
            Ok(n) => n.min(buf.len()),
            Err(e) if first => return Err(e),
            Err(_) => break,
        };
        first = false;
        let mut off = 0;
        while off + NAME_OFF <= n {
            let reclen = usize::from(u16::from_ne_bytes([buf[off + 16], buf[off + 17]]));
            if reclen == 0 || off + reclen > n {
                break;
            }
            let name = &buf[off + NAME_OFF..off + reclen];
            let len = name.iter().position(|&b| b == 0).unwrap_or(name.len());
            names.push(name[..len].to_vec());
            off += reclen;
        }
    }
    Ok(names)
}

/// The leading fields of `statfs(2)`'s `struct statfs` (the same 64-bit layout
/// on x86-64 and arm64) that `statvfs(3)` is computed from.
#[derive(Debug, Clone, Copy)]
pub struct StatFs {
    pub bsize: u64,
    pub blocks: u64,
    pub bfree: u64,
    pub bavail: u64,
    pub files: u64,
    pub ffree: u64,
    /// Fragment size; 0 from kernels that don't report it (then `bsize`).
    pub frsize: u64,
}

pub fn statfs(path: &CStr) -> io::Result<StatFs> {
    // f_type, f_bsize, f_blocks, f_bfree, f_bavail, f_files, f_ffree, f_fsid,
    // f_namelen, f_frsize, f_flags, f_spare[4]: 15 words, oversized here.
    let mut b = [0u64; 32];
    // SAFETY: `path` is NUL-terminated; `b` is larger than `struct statfs`.
    unsafe { call(nr::STATFS, &[path_arg(path), b.as_mut_ptr() as usize]) }?;
    Ok(StatFs {
        bsize: b[1],
        blocks: b[2],
        bfree: b[3],
        bavail: b[4],
        files: b[5],
        ffree: b[6],
        frsize: b[9],
    })
}

/// `ioctl(fd, request, arg)` with a pointer or integer argument.
///
/// # Safety
/// `arg` must be what `request` expects of it: for a pointer, live memory of
/// the size and mutability the request reads or writes.
pub unsafe fn ioctl(fd: RawFd, request: u64, arg: usize) -> io::Result<usize> {
    // SAFETY: forwarded from the caller.
    unsafe { call(nr::IOCTL, &[fd_arg(fd), request as usize, arg]) }
}

// ---- sockets -----------------------------------------------------------------

pub fn setsockopt(fd: RawFd, level: c_int, name: c_int, val: &[u8]) -> io::Result<()> {
    // SAFETY: `val` is `val.len()` readable bytes for the call.
    unsafe {
        call(
            nr::SETSOCKOPT,
            &[
                fd_arg(fd),
                level as usize,
                name as usize,
                val.as_ptr() as usize,
                val.len(),
            ],
        )
    }
    .map(drop)
}

/// `getsockopt` into `val`, returning the length the kernel wrote.
pub fn getsockopt(fd: RawFd, level: c_int, name: c_int, val: &mut [u8]) -> io::Result<usize> {
    let mut len = val.len() as u32;
    // SAFETY: `val` is `len` writable bytes, `len` a writable u32.
    unsafe {
        call(
            nr::GETSOCKOPT,
            &[
                fd_arg(fd),
                level as usize,
                name as usize,
                val.as_mut_ptr() as usize,
                (&raw mut len) as usize,
            ],
        )
    }?;
    Ok(len as usize)
}

// ---- memory ------------------------------------------------------------------

/// `mmap`, returning the mapping's address.
///
/// # Safety
/// As `mmap(2)`: a fixed `addr` replaces whatever was mapped there.
pub unsafe fn mmap(
    addr: usize,
    len: usize,
    prot: c_int,
    flags: c_int,
    fd: RawFd,
    off: i64,
) -> io::Result<usize> {
    // SAFETY: forwarded from the caller.
    unsafe {
        call(
            nr::MMAP,
            &[
                addr,
                len,
                prot as usize,
                flags as usize,
                fd_arg(fd),
                off as usize,
            ],
        )
    }
}

/// # Safety
/// `[addr, addr+len)` must be a mapping nothing still uses.
pub unsafe fn munmap(addr: usize, len: usize) -> io::Result<()> {
    // SAFETY: forwarded from the caller.
    unsafe { call(nr::MUNMAP, &[addr, len]) }.map(drop)
}

/// # Safety
/// As `madvise(2)`; a destructive advice (`MADV_DONTNEED`) zeroes the range.
pub unsafe fn madvise(addr: usize, len: usize, advice: c_int) -> io::Result<()> {
    // SAFETY: forwarded from the caller.
    unsafe { call(nr::MADVISE, &[addr, len, advice as usize]) }.map(drop)
}

// ---- time, threads, signals --------------------------------------------------

/// `clock_gettime(clk)` (the system call, not the vDSO).
pub fn clock_gettime(clk: c_int) -> io::Result<Timespec> {
    let mut ts = Timespec::default();
    // SAFETY: `ts` is a writable timespec.
    unsafe { call(nr::CLOCK_GETTIME, &[clk as usize, (&raw mut ts) as usize]) }?;
    Ok(ts)
}

/// The calling thread's id.
pub fn gettid() -> c_int {
    // SAFETY: no arguments; cannot fail.
    unsafe { syscall6(nr::GETTID, [0; 6]) as c_int }
}

/// `SA_RESTORER`: `sa_restorer` holds the signal-return trampoline.
const SA_RESTORER: u64 = 0x0400_0000;

/// The kernel's `struct sigaction` (`rt_sigaction`'s layout, which is not
/// libc's): handler, flags, restorer, then a 64-bit mask.
#[repr(C)]
struct KernelSigaction {
    handler: usize,
    flags: u64,
    restorer: usize,
    mask: u64,
}

/// The signal-return trampoline a handler returns into: `rt_sigreturn`. libc
/// supplies its own; a raw `rt_sigaction` on x86-64 must name one (arm64 falls
/// back to the vDSO's).
#[cfg(target_arch = "x86_64")]
#[unsafe(naked)]
unsafe extern "C" fn restore_rt() {
    std::arch::naked_asm!("mov eax, 15", "syscall", "ud2");
}

/// Install `handler` for signal `sig` with `flags` (no `SA_SIGINFO`), an empty
/// mask, and the return trampoline set up as libc's `sigaction` would.
pub fn set_signal_handler(sig: c_int, handler: extern "C" fn(c_int), flags: u64) -> io::Result<()> {
    #[cfg(target_arch = "x86_64")]
    let (flags, restorer) = (flags | SA_RESTORER, restore_rt as *const () as usize);
    #[cfg(target_arch = "aarch64")]
    let restorer = 0;
    let act = KernelSigaction {
        handler: handler as *const () as usize,
        flags,
        restorer,
        mask: 0,
    };
    // SAFETY: `act` is a readable kernel sigaction; no old action is asked for;
    // the mask is the kernel's 8 bytes.
    unsafe {
        call(
            nr::RT_SIGACTION,
            &[sig as usize, (&raw const act) as usize, 0, 8],
        )
    }
    .map(drop)
}

/// `struct sigevent` (64 bytes), for a timer that signals one thread.
#[repr(C)]
struct Sigevent {
    value: usize,
    signo: c_int,
    notify: c_int,
    tid: c_int,
    pad: [c_int; 11],
}

/// `SIGEV_THREAD_ID`: deliver to the thread `sigev_notify_thread_id` names.
const SIGEV_THREAD_ID: c_int = 4;

/// A POSIX timer on `clock` that raises `sig` at thread `tid` when it expires;
/// returns the kernel's timer id.
pub fn timer_create_thread(clock: c_int, sig: c_int, tid: c_int) -> io::Result<c_int> {
    let mut ev = Sigevent {
        value: 0,
        signo: sig,
        notify: SIGEV_THREAD_ID,
        tid,
        pad: [0; 11],
    };
    let mut id: c_int = 0;
    // SAFETY: `ev` is a readable sigevent, `id` a writable kernel timer id.
    unsafe {
        call(
            nr::TIMER_CREATE,
            &[
                clock as usize,
                (&raw mut ev) as usize,
                (&raw mut id) as usize,
            ],
        )
    }?;
    Ok(id)
}

/// Arm timer `id` to expire once after `value` (`interval` zero), or every
/// `interval` after that; an all-zero `value` disarms it.
pub fn timer_settime(id: c_int, interval: Timespec, value: Timespec) -> io::Result<()> {
    let spec = [interval, value];
    // SAFETY: `spec` is a readable itimerspec; no old value is asked for.
    unsafe {
        call(
            nr::TIMER_SETTIME,
            &[id as usize, 0, spec.as_ptr() as usize, 0],
        )
    }
    .map(drop)
}

pub fn timer_delete(id: c_int) -> io::Result<()> {
    // SAFETY: no pointers.
    unsafe { call(nr::TIMER_DELETE, &[id as usize]) }.map(drop)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::fd::AsRawFd;

    fn scratch(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("nixvm-raw-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn errors_are_the_kernels_errno() {
        let e = openat(AT_FDCWD, c"/nonexistent/nixvm", 0, 0).unwrap_err();
        assert_eq!(e.raw_os_error(), Some(2)); // ENOENT
        assert_eq!(close(-1).unwrap_err().raw_os_error(), Some(9)); // EBADF
    }

    #[test]
    fn at_calls_and_dir_names() {
        let d = scratch("at");
        let dir = std::fs::File::open(&d).unwrap();
        let fd = dir.as_raw_fd();
        mkdirat(fd, c"sub", 0o755).unwrap();
        // O_WRONLY | O_CREAT
        let f = openat(fd, c"f", 0o1 | 0o100, 0o644).unwrap();
        drop(f);
        symlinkat(c"f", fd, c"l").unwrap();
        let mut buf = [0u8; 64];
        let n = readlinkat(fd, c"l", &mut buf).unwrap();
        assert_eq!(&buf[..n], b"f");
        linkat(fd, c"f", fd, c"h", 0).unwrap();
        renameat(fd, c"h", fd, c"h2").unwrap();
        fchmodat(fd, c"f", 0o600).unwrap();
        let t = Timespec {
            tv_sec: 1_000_000,
            tv_nsec: 0,
        };
        utimensat(fd, c"f", &[t, t], 0).unwrap();
        use std::os::unix::fs::MetadataExt;
        let m = std::fs::metadata(d.join("f")).unwrap();
        assert_eq!(
            (m.mode() & 0o777, m.mtime(), m.nlink()),
            (0o600, 1_000_000, 2)
        );

        let dfd = openat(
            AT_FDCWD,
            &std::ffi::CString::new(d.to_str().unwrap()).unwrap(),
            0o200_000,
            0,
        )
        .unwrap();
        let mut names = dir_names(dfd.as_raw_fd()).unwrap();
        names.sort();
        let names: Vec<_> = names
            .iter()
            .map(|n| String::from_utf8_lossy(n).into_owned())
            .collect();
        assert_eq!(names, [".", "..", "f", "h2", "l", "sub"]);

        unlinkat(fd, c"sub", 0x200).unwrap(); // AT_REMOVEDIR
        assert!(!d.join("sub").exists());
        let s = statfs(&std::ffi::CString::new(d.to_str().unwrap()).unwrap()).unwrap();
        assert!(s.bsize > 0 && s.blocks > 0);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn clocks_and_mappings() {
        let a = clock_gettime(1).unwrap();
        let b = clock_gettime(1).unwrap();
        assert!((b.tv_sec, b.tv_nsec) >= (a.tv_sec, a.tv_nsec));
        assert!(gettid() > 0);
        // PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS
        let p = unsafe { mmap(0, 8192, 3, 0x2 | 0x20, -1, 0) }.unwrap();
        // SAFETY: a fresh, writable private mapping of 8 KiB.
        unsafe {
            (p as *mut u8).write(7);
            madvise(p, 8192, 4).unwrap(); // MADV_DONTNEED zeroes it
            assert_eq!((p as *const u8).read(), 0);
            munmap(p, 8192).unwrap();
        }
    }
}
