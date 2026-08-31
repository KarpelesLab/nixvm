//! Breadth for the filesystem syscalls: the `*at` family, `statfs`, xattr
//! stubs, and permission/ownership/timestamp no-ops.
//!
//! These mirror the core handlers in the parent module (`sys_openat`,
//! `sys_newfstatat`, …): read the guest path with [`read_path`], resolve it
//! with [`Kernel::resolve_path`] (honoring `AT_FDCWD`), and delegate to the
//! [`crate::fs::MountTable`]. Handlers for state nixvm does not model yet
//! (permissions, ownership, timestamps, extended attributes) accept the call
//! and either succeed or report the benign "unset" error.

use crate::abi::errno::Errno;
use crate::fs::{MountTable, NodeKind};
use crate::vcpu::GuestMemory;

use super::{AT_FDCWD, Fd, Kernel, ServiceCtx, Shared, err, io_errno, read_path, stat};

/// `unlinkat` flag: remove a directory, like `rmdir(2)`.
const AT_REMOVEDIR: u64 = 0x200;

impl Kernel {
    /// `statfs(path, buf)` — write a plausible `struct statfs` for the
    /// filesystem containing `path`.
    pub(super) fn sys_statfs(&self, vfs: &mut MountTable, cx: &mut ServiceCtx, pathptr: u64, buf: u64, mem: &mut GuestMemory) -> i64 {
        let Some(rel) = read_path(mem, pathptr) else {
            return err(Errno::EFAULT);
        };
        let abs = self.resolve_path(cx, AT_FDCWD, &rel);
        let abs = match self.follow_or_eloop(vfs, &abs) {
            Ok(p) => p,
            Err(e) => return e,
        };
        if vfs.stat(&abs).is_none() {
            return err(Errno::ENOENT);
        }
        write_statfs_or_fault(mem, buf)
    }

    /// `fstatfs(fd, buf)` — as `statfs`, keyed by an open fd.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_fstatfs(&self, cx: &mut ServiceCtx, fd: u64, buf: u64, mem: &mut GuestMemory) -> i64 {
        if cx.cur.fds.get(fd as i32).is_none() {
            return err(Errno::EBADF);
        }
        write_statfs_or_fault(mem, buf)
    }

    /// `readlinkat(dirfd, path, buf, bufsz)` — copy the link target (truncated
    /// to `bufsz`, not NUL-terminated) and return the byte count.
    /// If `abs` is `/proc/self/fd/<n>` or `/proc/<this-pid>/fd/<n>`, return the
    /// symlink target for descriptor `n` from the running task's live fd table
    /// (the path for a file/dir, an `anon_inode:`/`pipe:`/`socket:` name
    /// otherwise). `None` if the path isn't such a link or the fd is closed.
    #[allow(clippy::unused_self)]
    fn proc_fd_link(&self, cx: &ServiceCtx, abs: &str) -> Option<String> {
        let rest = abs.strip_prefix("/proc/")?;
        let (who, tail) = rest.split_once('/')?;
        if who != "self" && who != cx.cur.pid.to_string() {
            return None;
        }
        // `/proc/self/{exe,cwd,root}` resolve to *live* per-task state, not
        // procfs's static snapshot — programs read these to locate themselves.
        match tail {
            "exe" => return (!cx.cur.exe.is_empty()).then(|| cx.cur.exe.clone()),
            "cwd" => return Some(cx.cur.cwd.clone()),
            "root" => return Some("/".to_string()),
            _ => {}
        }
        let n: i32 = tail.strip_prefix("fd/")?.parse().ok()?;
        Some(fd_link_target(cx.cur.fds.get(n)?))
    }

    /// Build the live `/proc/self` view (comm, cmdline, exe, cwd, pid/ppid, open
    /// fds) for the running task, so procfs renders the real running program
    /// instead of the boot-time placeholder. Called just before a `/proc` read.
    pub(super) fn proc_self_live(&self, cx: &ServiceCtx) -> crate::fs::ProcSelf {
        let fds = cx
            .cur
            .fds
            .iter()
            .map(|(n, fd)| (n as u32, fd_link_target(fd)))
            .collect();
        crate::fs::ProcSelf {
            comm: cx.cur.comm.clone(),
            cmdline: cx.cur.cmdline.clone(),
            exe: cx.cur.exe.clone(),
            cwd: cx.cur.cwd.clone(),
            pid: cx.cur.pid as u32,
            ppid: cx.cur.ppid as u32,
            fds,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_readlinkat(
        &self, vfs: &mut MountTable, cx: &mut ServiceCtx,
        dirfd: i64,
        pathptr: u64,
        buf: u64,
        bufsz: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some(rel) = read_path(mem, pathptr) else {
            return err(Errno::EFAULT);
        };
        let abs = self.resolve_path(cx, dirfd, &rel);
        // /proc/self/fd/<n> (and /proc/<pid>/fd/<n> for this task) must resolve
        // against the *live* fd table, not procfs's static snapshot — programs
        // canonicalize a path by opening it and reading this link (realpath).
        let target = if let Some(t) = self.proc_fd_link(cx, &abs) {
            t
        } else {
            match vfs.readlink(&abs) {
                Ok(t) => t,
                Err(e) => return io_errno(&e),
            }
        };
        let bytes = target.as_bytes();
        let n = bytes.len().min(bufsz as usize);
        if mem.write(buf, &bytes[..n]).is_err() {
            return err(Errno::EFAULT);
        }
        n as i64
    }

    /// `symlinkat(target, newdirfd, linkpath)` — the target is stored verbatim.
    pub(super) fn sys_symlinkat(
        &self, vfs: &mut MountTable, cx: &mut ServiceCtx,
        targetptr: u64,
        newdirfd: i64,
        linkptr: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let (Some(target), Some(link)) = (read_path(mem, targetptr), read_path(mem, linkptr))
        else {
            return err(Errno::EFAULT);
        };
        let abs = self.resolve_path(cx, newdirfd, &link);
        match vfs.symlink(&target, &abs) {
            Ok(()) => 0,
            Err(e) => io_errno(&e),
        }
    }

    /// `mkdirat(dirfd, path, mode)`.
    pub(super) fn sys_mkdirat(
        &self, vfs: &mut MountTable, cx: &mut ServiceCtx,
        dirfd: i64,
        pathptr: u64,
        mode: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let Some(rel) = read_path(mem, pathptr) else {
            return err(Errno::EFAULT);
        };
        let abs = self.resolve_path(cx, dirfd, &rel);
        match vfs.mkdir(&abs, (mode & 0o777) as u32) {
            Ok(()) => 0,
            Err(e) => io_errno(&e),
        }
    }

    /// `unlinkat(dirfd, path, flags)` — `rmdir` when `AT_REMOVEDIR` is set,
    /// otherwise `unlink`.
    pub(super) fn sys_unlinkat(
        &self, vfs: &mut MountTable, cx: &mut ServiceCtx,
        dirfd: i64,
        pathptr: u64,
        flags: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let Some(rel) = read_path(mem, pathptr) else {
            return err(Errno::EFAULT);
        };
        let abs = self.resolve_path(cx, dirfd, &rel);
        let r = if flags & AT_REMOVEDIR != 0 {
            vfs.rmdir(&abs)
        } else {
            vfs.unlink(&abs)
        };
        match r {
            Ok(()) => 0,
            Err(e) => io_errno(&e),
        }
    }

    /// `utimensat(dirfd, path, times, flags)` — set a node's access and
    /// modification times. `times` is `[atime, mtime]`, each a `struct timespec`
    /// (`{ tv_sec, tv_nsec }`); a `tv_nsec` of `UTIME_NOW` uses the current time,
    /// `UTIME_OMIT` leaves that field unchanged, and a NULL `times` sets both to
    /// now. A NULL `path` targets `dirfd` itself (`futimens`);
    /// `AT_SYMLINK_NOFOLLOW` acts on the link rather than its target.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_utimensat(
        &self, vfs: &mut MountTable, cx: &mut ServiceCtx,
        dirfd: i64,
        pathptr: u64,
        times: u64,
        flags: u64,
        mem: &GuestMemory,
    ) -> i64 {
        const AT_SYMLINK_NOFOLLOW: u64 = 0x100;
        // Decode the timespec at `times + off` into a `SetTime`.
        let read_field = |off: u64| -> Option<crate::fs::SetTime> {
            const UTIME_NOW: u64 = 0x3fff_ffff;
            const UTIME_OMIT: u64 = 0x3fff_fffe;
            let (sec, nsec) = (mem.read_u64(times + off).ok()?, mem.read_u64(times + off + 8).ok()?);
            Some(match nsec {
                UTIME_OMIT => crate::fs::SetTime::Omit,
                UTIME_NOW => crate::fs::SetTime::Now,
                _ => crate::fs::SetTime::Set { sec: sec as i64, nsec: nsec as i64 },
            })
        };
        // atime is the first timespec (offset 0), mtime the second (offset 16).
        let (atime, mtime) = if times == 0 {
            (crate::fs::SetTime::Now, crate::fs::SetTime::Now)
        } else {
            let (Some(a), Some(m)) = (read_field(0), read_field(16)) else {
                return err(Errno::EFAULT);
            };
            (a, m)
        };
        // A NULL path means dirfd *is* the file (futimens); otherwise resolve.
        let abs = if pathptr == 0 {
            match cx.cur.fds.get(dirfd as i32) {
                Some(Fd::File { path, .. } | Fd::Dir { path, .. }) => path.clone(),
                _ => return err(Errno::EBADF),
            }
        } else {
            let Some(rel) = read_path(mem, pathptr) else {
                return err(Errno::EFAULT);
            };
            let abs = self.resolve_path(cx, dirfd, &rel);
            if flags & AT_SYMLINK_NOFOLLOW == 0 {
                match self.follow_or_eloop(vfs, &abs) {
                    Ok(p) => p,
                    Err(e) => return e,
                }
            } else {
                abs
            }
        };
        match vfs.set_times(&abs, atime, mtime) {
            Ok(()) => 0,
            Err(e) => io_errno(&e),
        }
    }

    /// `fchmodat(dirfd, path, mode, flags)` / `chmod(path, mode)` — set a file's
    /// permission bits. `fchmod(fd, mode)` shares the store via the fd's path.
    pub(super) fn sys_fchmodat(&self, vfs: &mut MountTable, cx: &mut ServiceCtx, dirfd: i64, pathptr: u64, mode: u64, mem: &GuestMemory) -> i64 {
        let Some(rel) = read_path(mem, pathptr) else {
            return err(Errno::EFAULT);
        };
        let abs = self.resolve_path(cx, dirfd, &rel);
        let abs = match self.follow_or_eloop(vfs, &abs) {
            Ok(p) => p,
            Err(e) => return e,
        };
        match vfs.set_mode(&abs, mode as u32) {
            Ok(()) => 0,
            Err(e) => io_errno(&e),
        }
    }

    /// `fchmod(fd, mode)` — chmod on an open file, resolved via its path.
    pub(super) fn sys_fchmod(&self, vfs: &mut MountTable, cx: &mut ServiceCtx, fd: u64, mode: u64) -> i64 {
        let path = match cx.cur.fds.get(fd as i32) {
            Some(Fd::File { path, .. } | Fd::Dir { path, .. }) => path.clone(),
            Some(_) => return 0, // non-file fds: accept (nothing to chmod)
            None => return err(Errno::EBADF),
        };
        match vfs.set_mode(&path, mode as u32) {
            Ok(()) => 0,
            Err(e) => io_errno(&e),
        }
    }

    /// `fchownat(dirfd, path, uid, gid, flags)` — and the `chown`/`lchown`
    /// spellings, which route here with `AT_FDCWD` and the appropriate
    /// follow/no-follow flag. A `uid`/`gid` of `(uid_t)-1` leaves that id
    /// unchanged. Symlinks are followed unless `AT_SYMLINK_NOFOLLOW` is set.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_fchownat(
        &self, vfs: &mut MountTable, cx: &mut ServiceCtx,
        dirfd: i64,
        pathptr: u64,
        uid: u64,
        gid: u64,
        flags: u64,
        mem: &GuestMemory,
    ) -> i64 {
        const AT_SYMLINK_NOFOLLOW: u64 = 0x100;
        let Some(rel) = read_path(mem, pathptr) else {
            return err(Errno::EFAULT);
        };
        let abs = self.resolve_path(cx, dirfd, &rel);
        let abs = if flags & AT_SYMLINK_NOFOLLOW == 0 {
            self.follow_symlinks(vfs, &abs).unwrap_or(abs)
        } else {
            abs
        };
        match vfs.set_owner(&abs, decode_id(uid), decode_id(gid)) {
            Ok(()) => 0,
            Err(e) => io_errno(&e),
        }
    }

    /// `fchown(fd, uid, gid)` — chown on an open file, resolved via its path.
    pub(super) fn sys_fchown(&self, vfs: &mut MountTable, cx: &mut ServiceCtx, fd: u64, uid: u64, gid: u64) -> i64 {
        let path = match cx.cur.fds.get(fd as i32) {
            Some(Fd::File { path, .. } | Fd::Dir { path, .. }) => path.clone(),
            Some(_) => return 0, // non-file fds: accept (nothing to chown)
            None => return err(Errno::EBADF),
        };
        match vfs.set_owner(&path, decode_id(uid), decode_id(gid)) {
            Ok(()) => 0,
            Err(e) => io_errno(&e),
        }
    }

    /// `mknodat(dirfd, path, mode, dev)` / `mknod(path, mode, dev)` /
    /// `mkfifo` (glibc issues `mknodat` with `S_IFIFO`). Creates a regular file
    /// or FIFO; device/socket nodes are left for the backend to reject
    /// (`EPERM` on the in-memory backends).
    pub(super) fn sys_mknodat(
        &self, vfs: &mut MountTable, cx: &mut ServiceCtx,
        dirfd: i64,
        pathptr: u64,
        mode: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let Some(rel) = read_path(mem, pathptr) else {
            return err(Errno::EFAULT);
        };
        let abs = self.resolve_path(cx, dirfd, &rel);
        match vfs.mknod(&abs, mode as u32) {
            Ok(()) => 0,
            Err(e) => io_errno(&e),
        }
    }

    /// `renameat(olddirfd, old, newdirfd, new)` / `renameat2(..., flags)`.
    /// `RENAME_NOREPLACE` fails with `EEXIST` if the target exists (atomic
    /// create-if-absent); `RENAME_EXCHANGE` atomically swaps the two paths.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_renameat(
        &self, vfs: &mut MountTable, cx: &mut ServiceCtx,
        olddirfd: i64,
        oldptr: u64,
        newdirfd: i64,
        newptr: u64,
        flags: u64,
        mem: &GuestMemory,
    ) -> i64 {
        const RENAME_NOREPLACE: u64 = 1;
        const RENAME_EXCHANGE: u64 = 2;
        let (Some(old), Some(new)) = (read_path(mem, oldptr), read_path(mem, newptr)) else {
            return err(Errno::EFAULT);
        };
        let from = self.resolve_path(cx, olddirfd, &old);
        let to = self.resolve_path(cx, newdirfd, &new);
        if flags & RENAME_NOREPLACE != 0 && vfs.stat(&to).is_some() {
            return err(Errno::EEXIST);
        }
        if flags & RENAME_EXCHANGE != 0 {
            // Both must exist; swap via a temporary name (in-memory, single-
            // threaded per syscall, so the temp path is safe to reuse).
            if vfs.stat(&from).is_none() || vfs.stat(&to).is_none() {
                return err(Errno::ENOENT);
            }
            let tmp = format!("{to}.__nixvm_xchg__");
            if vfs.rename(&from, &tmp).is_err()
                || vfs.rename(&to, &from).is_err()
                || vfs.rename(&tmp, &to).is_err()
            {
                return err(Errno::EIO);
            }
            return 0;
        }
        match vfs.rename(&from, &to) {
            Ok(()) => 0,
            Err(e) => io_errno(&e),
        }
    }

    /// `faccessat(dirfd, path, ...)` / `access(path, ...)` — existence check
    /// only; there is no permission model yet.
    pub(super) fn sys_faccessat(&self, vfs: &mut MountTable, cx: &mut ServiceCtx, dirfd: i64, pathptr: u64, mode: u64, mem: &GuestMemory) -> i64 {
        const X_OK: u64 = 1;
        let Some(rel) = read_path(mem, pathptr) else {
            return err(Errno::EFAULT);
        };
        let had_slash = rel.len() > 1 && rel.ends_with('/');
        let abs = self.resolve_path(cx, dirfd, &rel);
        let abs = match self.follow_or_eloop(vfs, &abs) {
            Ok(p) => p,
            Err(e) => return e,
        };
        let Some(attrs) = vfs.stat(&abs) else {
            if self.has_nondir_component(vfs, &abs) {
                return err(Errno::ENOTDIR);
            }
            return err(Errno::ENOENT);
        };
        if had_slash && attrs.kind != NodeKind::Dir {
            return err(Errno::ENOTDIR);
        }
        // The VM runs as root, so read/write are always granted; execute still
        // requires at least one execute bit (even root can't exec a plain file).
        if mode & X_OK != 0 && attrs.mode & 0o111 == 0 {
            return err(Errno::EACCES);
        }
        0
    }

    /// `umask(mask)` — set the file-creation mask, returning the previous one.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_umask(&self, sh: &mut Shared, mask: u64) -> i64 {
        let old = sh.umask;
        sh.umask = (mask & 0o777) as u32;
        i64::from(old)
    }
}

/// Decode a `chown` uid/gid argument: `(uid_t)-1` (`0xFFFF_FFFF`, how the guest
/// zero-extends a 32-bit `-1`) means "leave unchanged" → `None`.
fn decode_id(v: u64) -> Option<u32> {
    let id = v as u32;
    if id == u32::MAX { None } else { Some(id) }
}

/// Write a `struct statfs` at `addr`, or return `-EFAULT`.
fn write_statfs_or_fault(mem: &mut GuestMemory, addr: u64) -> i64 {
    let buf = stat::encode_statfs();
    if mem.write(addr, &buf).is_err() {
        err(Errno::EFAULT)
    } else {
        0
    }
}

/// The `/proc/self/fd/<n>` symlink target for a descriptor: the path for a
/// file/dir, an `anon_inode:`/`pipe:`/`socket:` name otherwise (matching the
/// kernel's spellings).
fn fd_link_target(fd: &Fd) -> String {
    match fd {
        Fd::File { path, .. } | Fd::Dir { path, .. } => path.clone(),
        Fd::Stdin | Fd::Stdout | Fd::Stderr => "/dev/null".to_string(),
        Fd::PipeRead(i) | Fd::PipeWrite(i) => format!("pipe:[{i}]"),
        Fd::Socket { sock, .. } => format!("socket:[{sock}]"),
        Fd::Eventfd(_) => "anon_inode:[eventfd]".to_string(),
        Fd::Signalfd(_) => "anon_inode:[signalfd]".to_string(),
        Fd::Timerfd(_) => "anon_inode:[timerfd]".to_string(),
        Fd::Pidfd(_) => "anon_inode:[pidfd]".to_string(),
        Fd::Epoll(_) => "anon_inode:[eventpoll]".to_string(),
        Fd::PtyMaster(_) => "/dev/ptmx".to_string(),
        Fd::PtySlave(i) => format!("/dev/pts/{i}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::Arch;
    use crate::fs::{MountTable, NodeKind, TmpFs};
    use crate::vcpu::GuestMemory;
    use crate::vcpu::mem::Prot;

    const PAGE: u64 = 4096;

    fn setup() -> (Kernel, GuestMemory, ServiceCtx) {
        let mut mounts = MountTable::new();
        mounts.mount("/", Box::new(TmpFs::new()));
        let kernel = Kernel::new(Arch::Aarch64, mounts);
        let mut cx = ServiceCtx::default();
        cx.cur.pid = 1;
        let mut mem = GuestMemory::new(0x1_0000, 16 * PAGE);
        mem.map(0x1_0000, 4 * PAGE, Prot::rw()).unwrap();
        (kernel, mem, cx)
    }

    #[test]
    fn renameat2_noreplace_and_exchange() {
        let (k, mut mem, mut cx) = setup();
        let (pa, pb) = (0x1_0000, 0x1_0010);
        mem.write_init(pa, b"/a\0").unwrap();
        mem.write_init(pb, b"/b\0").unwrap();
        {
            let mut vfs = k.vfs.lock().unwrap();
            vfs.create("/a", 0o644).unwrap();
            vfs.write_at("/a", 0, b"AAA").unwrap();
            vfs.create("/b", 0o644).unwrap();
            vfs.write_at("/b", 0, b"BBB").unwrap();
        }
        // RENAME_NOREPLACE onto an existing target → EEXIST, both untouched.
        assert_eq!(
            k.sys_renameat(&mut k.vfs.lock().unwrap(), &mut cx, AT_FDCWD, pa, AT_FDCWD, pb, 1, &mem),
            err(Errno::EEXIST)
        );
        assert!(k.vfs.lock().unwrap().stat("/a").is_some());
        // RENAME_EXCHANGE swaps the two files' contents.
        assert_eq!(
            k.sys_renameat(&mut k.vfs.lock().unwrap(), &mut cx, AT_FDCWD, pa, AT_FDCWD, pb, 2, &mem),
            0
        );
        let mut buf = [0u8; 3];
        k.vfs.lock().unwrap().read_at("/a", 0, &mut buf).unwrap();
        assert_eq!(&buf, b"BBB", "/a now holds what /b had");
        k.vfs.lock().unwrap().read_at("/b", 0, &mut buf).unwrap();
        assert_eq!(&buf, b"AAA");
    }

    #[test]
    fn mkdirat_then_faccessat_and_stat() {
        let (k, mut mem, mut cx) = setup();
        let path = 0x1_0000;
        mem.write_init(path, b"/d\0").unwrap();
        assert_eq!(k.sys_mkdirat(&mut k.vfs.lock().unwrap(), &mut cx, AT_FDCWD, path, 0o755, &mem), 0);
        // A directory (0o755) is searchable (X_OK).
        assert_eq!(k.sys_faccessat(&mut k.vfs.lock().unwrap(), &mut cx, AT_FDCWD, path, 1, &mem), 0);
        assert_eq!(k.vfs.lock().unwrap().stat("/d").unwrap().kind, NodeKind::Dir);
    }

    #[test]
    fn symlinkat_then_readlinkat() {
        let (k, mut mem, mut cx) = setup();
        let target = 0x1_0000;
        let link = 0x1_0100;
        let buf = 0x1_1000;
        mem.write_init(target, b"/target\0").unwrap();
        mem.write_init(link, b"/l\0").unwrap();
        assert_eq!(k.sys_symlinkat(&mut k.vfs.lock().unwrap(), &mut cx, target, AT_FDCWD, link, &mem), 0);
        assert_eq!(k.sys_readlinkat(&mut k.vfs.lock().unwrap(), &mut cx, AT_FDCWD, link, buf, 64, &mut mem), 7);
        assert_eq!(mem.read_vec(buf, 7).unwrap(), b"/target");
    }

    #[test]
    fn statfs_writes_bsize() {
        let (k, mut mem, mut cx) = setup();
        let path = 0x1_0000;
        let buf = 0x1_1000;
        mem.write_init(path, b"/\0").unwrap();
        assert_eq!(k.sys_statfs(&mut k.vfs.lock().unwrap(), &mut cx, path, buf, &mut mem), 0);
        assert_eq!(mem.read_u64(buf + 8).unwrap(), 4096); // f_bsize
    }

    #[test]
    fn unlinkat_removes_file() {
        let (k, mut mem, mut cx) = setup();
        k.vfs.lock().unwrap().create("/f", 0o644).unwrap();
        let path = 0x1_0000;
        mem.write_init(path, b"/f\0").unwrap();
        assert_eq!(k.sys_unlinkat(&mut k.vfs.lock().unwrap(), &mut cx, AT_FDCWD, path, 0, &mem), 0);
        assert!(k.vfs.lock().unwrap().stat("/f").is_none());
    }

    #[test]
    fn umask_returns_previous() {
        let (k, _mem, _cx) = setup();
        assert_eq!(k.sys_umask(&mut k.shared.lock().unwrap(), 0o077), 0o022);
        assert_eq!(k.sys_umask(&mut k.shared.lock().unwrap(), 0o022), 0o077);
    }
}
