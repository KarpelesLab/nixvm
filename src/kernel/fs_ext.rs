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
use crate::vcpu::GuestMemory;

use super::{AT_FDCWD, Fd, Kernel, ServiceCtx, Shared, err, io_errno, read_path, stat};

/// `unlinkat` flag: remove a directory, like `rmdir(2)`.
const AT_REMOVEDIR: u64 = 0x200;

impl Kernel {
    /// `statfs(path, buf)` — write a plausible `struct statfs` for the
    /// filesystem containing `path`.
    pub(super) fn sys_statfs(&self, sh: &mut Shared, cx: &mut ServiceCtx, pathptr: u64, buf: u64, mem: &mut GuestMemory) -> i64 {
        let Some(rel) = read_path(mem, pathptr) else {
            return err(Errno::EFAULT);
        };
        let abs = self.resolve_path(cx, AT_FDCWD, &rel);
        let abs = self.follow_symlinks(sh, &abs).unwrap_or(abs);
        if sh.mounts.stat(&abs).is_none() {
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
        let (who, tail) = rest.split_once("/fd/")?;
        if who != "self" && who != cx.cur.pid.to_string() {
            return None;
        }
        let n: i32 = tail.parse().ok()?;
        Some(match cx.cur.fds.get(n)? {
            Fd::File { path, .. } | Fd::Dir { path, .. } => path.clone(),
            Fd::Stdin | Fd::Stdout | Fd::Stderr => "/dev/null".to_string(),
            Fd::PipeRead(i) | Fd::PipeWrite(i) => format!("pipe:[{i}]"),
            Fd::Socket { sock, .. } => format!("socket:[{sock}]"),
            Fd::Eventfd(_) => "anon_inode:[eventfd]".to_string(),
            Fd::Timerfd(_) => "anon_inode:[timerfd]".to_string(),
            Fd::Epoll(_) => "anon_inode:[eventpoll]".to_string(),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_readlinkat(
        &self, sh: &mut Shared, cx: &mut ServiceCtx,
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
            match sh.mounts.readlink(&abs) {
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
        &self, sh: &mut Shared, cx: &mut ServiceCtx,
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
        match sh.mounts.symlink(&target, &abs) {
            Ok(()) => 0,
            Err(e) => io_errno(&e),
        }
    }

    /// `mkdirat(dirfd, path, mode)`.
    pub(super) fn sys_mkdirat(
        &self, sh: &mut Shared, cx: &mut ServiceCtx,
        dirfd: i64,
        pathptr: u64,
        mode: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let Some(rel) = read_path(mem, pathptr) else {
            return err(Errno::EFAULT);
        };
        let abs = self.resolve_path(cx, dirfd, &rel);
        match sh.mounts.mkdir(&abs, (mode & 0o777) as u32) {
            Ok(()) => 0,
            Err(e) => io_errno(&e),
        }
    }

    /// `unlinkat(dirfd, path, flags)` — `rmdir` when `AT_REMOVEDIR` is set,
    /// otherwise `unlink`.
    pub(super) fn sys_unlinkat(
        &self, sh: &mut Shared, cx: &mut ServiceCtx,
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
            sh.mounts.rmdir(&abs)
        } else {
            sh.mounts.unlink(&abs)
        };
        match r {
            Ok(()) => 0,
            Err(e) => io_errno(&e),
        }
    }

    /// `renameat(olddirfd, old, newdirfd, new)` / `renameat2(..., flags)` — the
    /// flags argument is accepted but not honored.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_renameat(
        &self, sh: &mut Shared, cx: &mut ServiceCtx,
        olddirfd: i64,
        oldptr: u64,
        newdirfd: i64,
        newptr: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let (Some(old), Some(new)) = (read_path(mem, oldptr), read_path(mem, newptr)) else {
            return err(Errno::EFAULT);
        };
        let from = self.resolve_path(cx, olddirfd, &old);
        let to = self.resolve_path(cx, newdirfd, &new);
        match sh.mounts.rename(&from, &to) {
            Ok(()) => 0,
            Err(e) => io_errno(&e),
        }
    }

    /// `faccessat(dirfd, path, ...)` / `access(path, ...)` — existence check
    /// only; there is no permission model yet.
    pub(super) fn sys_faccessat(&self, sh: &mut Shared, cx: &mut ServiceCtx, dirfd: i64, pathptr: u64, mem: &GuestMemory) -> i64 {
        let Some(rel) = read_path(mem, pathptr) else {
            return err(Errno::EFAULT);
        };
        let abs = self.resolve_path(cx, dirfd, &rel);
        let abs = self.follow_symlinks(sh, &abs).unwrap_or(abs);
        if sh.mounts.stat(&abs).is_some() {
            0
        } else {
            err(Errno::ENOENT)
        }
    }

    /// `umask(mask)` — set the file-creation mask, returning the previous one.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_umask(&self, sh: &mut Shared, mask: u64) -> i64 {
        let old = sh.umask;
        sh.umask = (mask & 0o777) as u32;
        i64::from(old)
    }
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
    fn mkdirat_then_faccessat_and_stat() {
        let (k, mut mem, mut cx) = setup();
        let path = 0x1_0000;
        mem.write_init(path, b"/d\0").unwrap();
        assert_eq!(k.sys_mkdirat(&mut k.shared.lock().unwrap(), &mut cx, AT_FDCWD, path, 0o755, &mem), 0);
        assert_eq!(k.sys_faccessat(&mut k.shared.lock().unwrap(), &mut cx, AT_FDCWD, path, &mem), 0);
        assert_eq!(k.shared.lock().unwrap().mounts.stat("/d").unwrap().kind, NodeKind::Dir);
    }

    #[test]
    fn symlinkat_then_readlinkat() {
        let (k, mut mem, mut cx) = setup();
        let target = 0x1_0000;
        let link = 0x1_0100;
        let buf = 0x1_1000;
        mem.write_init(target, b"/target\0").unwrap();
        mem.write_init(link, b"/l\0").unwrap();
        assert_eq!(k.sys_symlinkat(&mut k.shared.lock().unwrap(), &mut cx, target, AT_FDCWD, link, &mem), 0);
        assert_eq!(k.sys_readlinkat(&mut k.shared.lock().unwrap(), &mut cx, AT_FDCWD, link, buf, 64, &mut mem), 7);
        assert_eq!(mem.read_vec(buf, 7).unwrap(), b"/target");
    }

    #[test]
    fn statfs_writes_bsize() {
        let (k, mut mem, mut cx) = setup();
        let path = 0x1_0000;
        let buf = 0x1_1000;
        mem.write_init(path, b"/\0").unwrap();
        assert_eq!(k.sys_statfs(&mut k.shared.lock().unwrap(), &mut cx, path, buf, &mut mem), 0);
        assert_eq!(mem.read_u64(buf + 8).unwrap(), 4096); // f_bsize
    }

    #[test]
    fn unlinkat_removes_file() {
        let (k, mut mem, mut cx) = setup();
        k.shared.lock().unwrap().mounts.create("/f", 0o644).unwrap();
        let path = 0x1_0000;
        mem.write_init(path, b"/f\0").unwrap();
        assert_eq!(k.sys_unlinkat(&mut k.shared.lock().unwrap(), &mut cx, AT_FDCWD, path, 0, &mem), 0);
        assert!(k.shared.lock().unwrap().mounts.stat("/f").is_none());
    }

    #[test]
    fn umask_returns_previous() {
        let (k, _mem, _cx) = setup();
        assert_eq!(k.sys_umask(&mut k.shared.lock().unwrap(), 0o077), 0o022);
        assert_eq!(k.sys_umask(&mut k.shared.lock().unwrap(), 0o022), 0o077);
    }
}
