//! Zero-copy-in-name I/O and the newer open/read/write spellings: `splice`,
//! `tee`, `vmsplice`, `preadv2`/`pwritev2`, `openat2`, and `cachestat`.
//!
//! nixvm has no page cache to move pages between, so `splice`/`tee` copy —
//! but with Linux's *contract*: one end must be a pipe (`EINVAL` otherwise),
//! a pipe end takes no offset (`ESPIPE`), an empty pipe blocks / `EAGAIN`s /
//! reports EOF exactly like `read`, a reader-less destination pipe raises
//! `SIGPIPE`/`EPIPE`, and a non-pipe end's offset (pointer or fd position) is
//! advanced. A single splice into a pipe moves at most one pipe's worth
//! ([`PIPE_BUF_CAP`]), as Linux bounds it by the pipe's capacity — callers
//! loop. Each handler takes its own locks in the global order (the non-pipe
//! side's `sh`/`vfs`/`net` first, then `pipes`), so the move is atomic with
//! respect to other tasks touching the same pipe.

use super::{AT_FDCWD, Fd, Kernel, ServiceCtx, err, io_errno, read_path};
use crate::abi::Arch;
use crate::abi::errno::Errno;
use crate::fs::{MountTable, NodeKind};
use crate::vcpu::GuestMemory;
use std::io::Write;

/// The most one `splice`/`tee` moves into a pipe: the default pipe capacity.
const PIPE_BUF_CAP: usize = 65536;
const SPLICE_F_NONBLOCK: u64 = 0x02;
/// `SPLICE_F_MOVE | SPLICE_F_NONBLOCK | SPLICE_F_MORE | SPLICE_F_GIFT`.
const SPLICE_F_ALL: u64 = 0x0f;
/// `UIO_MAXIOV`: the most iovecs a vectored call accepts.
const UIO_MAXIOV: u64 = 1024;
/// `RWF_APPEND`: `pwritev2` appends regardless of the offset.
const RWF_APPEND: u64 = 0x10;
/// `RWF_NOWAIT`: fail `EAGAIN` instead of blocking.
const RWF_NOWAIT: u64 = 0x08;
/// Every `RWF_*` flag Linux defines (`HIPRI` … `DONTCACHE`). An unknown bit is
/// `EOPNOTSUPP`, the answer callers probe new flags with.
const RWF_SUPPORTED: u64 = 0xff;

/// The outcome of a pipe-side precheck: either a final syscall result or the
/// go-ahead to move bytes.
enum PipeState {
    Done(i64),
    Ready,
}

impl Kernel {
    /// Decide what an empty-source read of pipe `i` returns: EOF (0) once every
    /// writer is gone, `EAGAIN` when non-blocking, else park (re-trap).
    fn pipe_src_state(
        pipes: &[super::Pipe],
        cx: &mut ServiceCtx,
        i: usize,
        nonblock: bool,
    ) -> PipeState {
        let p = &pipes[i];
        if !p.buf.is_empty() {
            return PipeState::Ready;
        }
        if p.writers == 0 {
            return PipeState::Done(0);
        }
        if nonblock || p.read_nonblock {
            return PipeState::Done(err(Errno::EAGAIN));
        }
        cx.block = true;
        PipeState::Done(0)
    }

    /// `splice(fd_in, off_in, fd_out, off_out, len, flags)`.
    #[allow(clippy::too_many_lines)]
    pub(super) fn sys_splice(
        &self,
        cx: &mut ServiceCtx,
        a: &[u64; 6],
        mem: &mut GuestMemory,
    ) -> i64 {
        let (fd_in, off_in, fd_out, off_out, len, flags) = (a[0], a[1], a[2], a[3], a[4], a[5]);
        if flags & !SPLICE_F_ALL != 0 {
            return err(Errno::EINVAL);
        }
        let nonblock = flags & SPLICE_F_NONBLOCK != 0;
        let (Some(fin), Some(fout)) = (
            cx.cur.fds.get(fd_in as i32).cloned(),
            cx.cur.fds.get(fd_out as i32).cloned(),
        ) else {
            return err(Errno::EBADF);
        };
        // The wrong end of a pipe is simply not open for that direction.
        if matches!(fin, Fd::PipeWrite(_)) || matches!(fout, Fd::PipeRead(_)) {
            return err(Errno::EBADF);
        }
        let in_pipe = if let Fd::PipeRead(i) = fin {
            Some(i)
        } else {
            None
        };
        let out_pipe = if let Fd::PipeWrite(j) = fout {
            Some(j)
        } else {
            None
        };
        if in_pipe.is_none() && out_pipe.is_none() {
            return err(Errno::EINVAL);
        }
        if (in_pipe.is_some() && off_in != 0) || (out_pipe.is_some() && off_out != 0) {
            return err(Errno::ESPIPE);
        }
        if len == 0 {
            return 0;
        }
        let read_off = |ptr: u64, fallback: u64| -> Result<u64, i64> {
            if ptr == 0 {
                return Ok(fallback);
            }
            match mem.read_u64(ptr) {
                Ok(v) if (v as i64) >= 0 => Ok(v),
                Ok(_) => Err(err(Errno::EINVAL)),
                Err(_) => Err(err(Errno::EFAULT)),
            }
        };
        match (in_pipe, out_pipe) {
            // pipe -> pipe: move within the pipe table.
            (Some(i), Some(j)) => {
                if i == j {
                    return err(Errno::EINVAL);
                }
                let mut pipes = self.pipes.lock().unwrap();
                if let PipeState::Done(r) = Self::pipe_src_state(&pipes, cx, i, nonblock) {
                    return r;
                }
                if pipes[j].readers == 0 {
                    self.raise_sigpipe(cx);
                    return err(Errno::EPIPE);
                }
                let n = (len as usize).min(pipes[i].buf.len()).min(PIPE_BUF_CAP);
                let data: Vec<u8> = pipes[i].buf.drain(..n).collect();
                pipes[j].buf.extend(data);
                n as i64
            }
            // pipe -> file / socket / stdio.
            (Some(i), None) => match fout {
                Fd::File {
                    path,
                    offset,
                    writable,
                    ..
                } => {
                    if !writable {
                        return err(Errno::EBADF);
                    }
                    // Linux refuses to splice into an O_APPEND file.
                    if cx.cur.fds.is_append(fd_out as i32) {
                        return err(Errno::EINVAL);
                    }
                    let pos = match read_off(off_out, offset.get()) {
                        Ok(p) => p,
                        Err(e) => return e,
                    };
                    let mut vfs = self.vfs.lock().unwrap();
                    let mut pipes = self.pipes.lock().unwrap();
                    if let PipeState::Done(r) = Self::pipe_src_state(&pipes, cx, i, nonblock) {
                        return r;
                    }
                    let n = (len as usize).min(pipes[i].buf.len());
                    let data: Vec<u8> = pipes[i].buf.iter().take(n).copied().collect();
                    let w = match self.vfs_write(&mut vfs, &path, pos, &data) {
                        Ok(w) => w,
                        Err(e) => return io_errno(&e),
                    };
                    pipes[i].buf.drain(..w);
                    drop(pipes);
                    drop(vfs);
                    self.advance_off(cx, fd_out, off_out, pos, w as u64, mem);
                    w as i64
                }
                Fd::Socket { sock, end } => {
                    let mut net = self.net.lock().unwrap();
                    let mut pipes = self.pipes.lock().unwrap();
                    if let PipeState::Done(r) = Self::pipe_src_state(&pipes, cx, i, nonblock) {
                        return r;
                    }
                    let n = (len as usize).min(pipes[i].buf.len());
                    let data: Vec<u8> = pipes[i].buf.iter().take(n).copied().collect();
                    let w = self.write_socket(&mut net, cx, sock, end, &data, false);
                    if w > 0 {
                        pipes[i].buf.drain(..w as usize);
                    }
                    w
                }
                Fd::Stdout | Fd::Stderr | Fd::Tty => {
                    let mut sh = self.shared.lock().unwrap();
                    let mut pipes = self.pipes.lock().unwrap();
                    if let PipeState::Done(r) = Self::pipe_src_state(&pipes, cx, i, nonblock) {
                        return r;
                    }
                    let n = (len as usize).min(pipes[i].buf.len());
                    let data: Vec<u8> = pipes[i].buf.drain(..n).collect();
                    let sink: &mut dyn Write = if matches!(fout, Fd::Stdout | Fd::Tty) {
                        &mut sh.stdout
                    } else {
                        &mut sh.stderr
                    };
                    match sink.write_all(&data) {
                        Ok(()) => n as i64,
                        Err(_) => err(Errno::EIO),
                    }
                }
                // A tty/eventfd/… destination has no splice support here.
                _ => err(Errno::EINVAL),
            },
            // file / socket -> pipe.
            (None, Some(j)) => {
                let cap = (len as usize).min(PIPE_BUF_CAP);
                match fin {
                    Fd::File {
                        path,
                        offset,
                        readable,
                        ..
                    } => {
                        if !readable {
                            return err(Errno::EBADF);
                        }
                        let pos = match read_off(off_in, offset.get()) {
                            Ok(p) => p,
                            Err(e) => return e,
                        };
                        let mut vfs = self.vfs.lock().unwrap();
                        let mut pipes = self.pipes.lock().unwrap();
                        if pipes[j].readers == 0 {
                            self.raise_sigpipe(cx);
                            return err(Errno::EPIPE);
                        }
                        let mut buf = vec![0u8; cap];
                        let n = match self.vfs_read(&mut vfs, &path, pos, &mut buf) {
                            Ok(n) => n,
                            Err(e) => return io_errno(&e),
                        };
                        pipes[j].buf.extend(&buf[..n]);
                        drop(pipes);
                        drop(vfs);
                        self.advance_off(cx, fd_in, off_in, pos, n as u64, mem);
                        n as i64
                    }
                    Fd::Socket { .. } => {
                        let mut net = self.net.lock().unwrap();
                        let mut pipes = self.pipes.lock().unwrap();
                        if pipes[j].readers == 0 {
                            self.raise_sigpipe(cx);
                            return err(Errno::EPIPE);
                        }
                        let rflags = if nonblock { 0x40 } else { 0 }; // MSG_DONTWAIT
                        match self.recv_fd_bytes(&mut net, cx, fd_in, cap as u64, rflags) {
                            Ok(data) => {
                                pipes[j].buf.extend(&data);
                                data.len() as i64
                            }
                            Err(e) => e,
                        }
                    }
                    _ => err(Errno::EINVAL),
                }
            }
            (None, None) => unreachable!("checked above"),
        }
    }

    /// Advance a non-pipe splice end by `n` bytes from `pos`: through the
    /// caller's offset pointer when one was given (the fd position is then
    /// untouched), else the fd's own position.
    #[allow(clippy::unused_self)]
    fn advance_off(
        &self,
        cx: &mut ServiceCtx,
        fd: u64,
        ptr: u64,
        pos: u64,
        n: u64,
        mem: &mut GuestMemory,
    ) {
        if ptr != 0 {
            let _ = mem.write_u64(ptr, pos + n);
        } else if let Some(Fd::File { offset, .. }) = cx.cur.fds.get_mut(fd as i32) {
            offset.set(pos + n);
        }
    }

    /// `tee(fd_in, fd_out, len, flags)` — duplicate up to `len` bytes from one
    /// pipe into another *without* consuming them from the source.
    pub(super) fn sys_tee(
        &self,
        cx: &mut ServiceCtx,
        fd_in: u64,
        fd_out: u64,
        len: u64,
        flags: u64,
    ) -> i64 {
        if flags & !SPLICE_F_ALL != 0 {
            return err(Errno::EINVAL);
        }
        let (Some(fin), Some(fout)) = (
            cx.cur.fds.get(fd_in as i32).cloned(),
            cx.cur.fds.get(fd_out as i32).cloned(),
        ) else {
            return err(Errno::EBADF);
        };
        let (Fd::PipeRead(i), Fd::PipeWrite(j)) = (fin, fout) else {
            return err(Errno::EINVAL);
        };
        if i == j {
            return err(Errno::EINVAL);
        }
        if len == 0 {
            return 0;
        }
        let mut pipes = self.pipes.lock().unwrap();
        if let PipeState::Done(r) =
            Self::pipe_src_state(&pipes, cx, i, flags & SPLICE_F_NONBLOCK != 0)
        {
            return r;
        }
        if pipes[j].readers == 0 {
            self.raise_sigpipe(cx);
            return err(Errno::EPIPE);
        }
        let n = (len as usize).min(pipes[i].buf.len()).min(PIPE_BUF_CAP);
        let data: Vec<u8> = pipes[i].buf.iter().take(n).copied().collect();
        pipes[j].buf.extend(data);
        n as i64
    }

    /// `vmsplice(fd, iov, nr_segs, flags)` — user memory into a pipe (write
    /// end: a `writev`) or, the rarer direction, a pipe into user memory (read
    /// end: a `readv`). A non-pipe fd is `EBADF`.
    pub(super) fn sys_vmsplice(
        &self,
        cx: &mut ServiceCtx,
        fd: u64,
        iov: u64,
        nr_segs: u64,
        flags: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        if flags & !SPLICE_F_ALL != 0 || nr_segs > UIO_MAXIOV {
            return err(Errno::EINVAL);
        }
        match cx.cur.fds.get(fd as i32).cloned() {
            Some(Fd::PipeWrite(_)) => self.sys_writev(cx, fd, iov, nr_segs, mem),
            Some(Fd::PipeRead(i)) => {
                if flags & SPLICE_F_NONBLOCK != 0 {
                    let pipes = self.pipes.lock().unwrap();
                    if pipes[i].buf.is_empty() && pipes[i].writers > 0 {
                        return err(Errno::EAGAIN);
                    }
                }
                self.sys_readv(cx, fd, iov, nr_segs, mem)
            }
            _ => err(Errno::EBADF),
        }
    }

    /// `preadv2`/`pwritev2(fd, iov, iovcnt, pos_l, pos_h, flags)`. On 64-bit,
    /// `pos_l` is the whole offset; `-1` means "the current file position"
    /// (plain `readv`/`writev` semantics, valid on any fd), anything else a
    /// positioned transfer (files only — `ESPIPE` otherwise). `RWF_APPEND`
    /// makes a write append whatever the offset; `RWF_NOWAIT` turns a would-
    /// block pipe read into `EAGAIN`; the hint flags (`HIPRI`/`DSYNC`/`SYNC`/
    /// `DONTCACHE`/`ATOMIC`) have nothing to act on in memory and are
    /// accepted; unknown flags are `EOPNOTSUPP`.
    pub(super) fn sys_rwv2(
        &self,
        cx: &mut ServiceCtx,
        a: &[u64; 6],
        write: bool,
        mem: &mut GuestMemory,
    ) -> i64 {
        let (fd, iov, iovcnt, off, flags) = (a[0], a[1], a[2], a[3] as i64, a[5]);
        if flags & !RWF_SUPPORTED != 0 {
            return err(Errno::EOPNOTSUPP);
        }
        if iovcnt > UIO_MAXIOV {
            return err(Errno::EINVAL);
        }
        if off < -1 {
            return err(Errno::EINVAL);
        }
        let Some(f) = cx.cur.fds.get(fd as i32).cloned() else {
            return err(Errno::EBADF);
        };
        if write && flags & RWF_APPEND != 0 {
            // Append: write at EOF; the file position moves only for the
            // "current position" (-1) spelling, as with an O_APPEND writev.
            let Fd::File { path, .. } = &f else {
                return self.sys_writev(cx, fd, iov, iovcnt, mem);
            };
            let mut vfs = self.vfs.lock().unwrap();
            let eof = vfs.stat(path).map_or(0, |at| at.size);
            let n = self.sys_pwritev(&mut vfs, cx, fd, iov, iovcnt, eof, mem);
            if n >= 0
                && off == -1
                && let Some(Fd::File { offset, .. }) = cx.cur.fds.get_mut(fd as i32)
            {
                offset.set(eof + n as u64);
            }
            return n;
        }
        if off == -1 {
            if !write && flags & RWF_NOWAIT != 0 && matches!(f, Fd::PipeRead(_)) {
                let Fd::PipeRead(i) = f else { unreachable!() };
                let pipes = self.pipes.lock().unwrap();
                if pipes[i].buf.is_empty() && pipes[i].writers > 0 {
                    return err(Errno::EAGAIN);
                }
            }
            return if write {
                self.sys_writev(cx, fd, iov, iovcnt, mem)
            } else {
                self.sys_readv(cx, fd, iov, iovcnt, mem)
            };
        }
        let mut vfs = self.vfs.lock().unwrap();
        if write {
            self.sys_pwritev(&mut vfs, cx, fd, iov, iovcnt, off as u64, mem)
        } else {
            self.sys_preadv(&mut vfs, cx, fd, iov, iovcnt, off as u64, mem)
        }
    }

    /// `openat2(dirfd, path, struct open_how *how, size)`: `openat` with
    /// strict flag validation and the `RESOLVE_*` path-walk restrictions.
    /// `struct open_how { u64 flags; u64 mode; u64 resolve; }` is extensible:
    /// `size` below the v0 size is `EINVAL`, above it the tail must be zero
    /// (`E2BIG`). Unlike `openat`, unknown `O_*` bits, a `mode` without
    /// `O_CREAT`/`O_TMPFILE`, and stray bits beside `O_PATH` are all `EINVAL`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_openat2(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        dirfd: i64,
        pathptr: u64,
        how: u64,
        size: u64,
        mem: &GuestMemory,
    ) -> i64 {
        const OPEN_HOW_SIZE_VER0: u64 = 24;
        const O_CREAT: u64 = 0o100;
        const O_TRUNC: u64 = 0o1000;
        const O_CLOEXEC: u64 = 0o2000000;
        const O_PATH: u64 = 0o10000000;
        const O_TMPFILE: u64 = 0o20000000;
        // Every O_* bit either arch defines (the arch-specific four —
        // O_DIRECT/O_LARGEFILE/O_DIRECTORY/O_NOFOLLOW — occupy the same four
        // bit positions on both, just permuted).
        const VALID_O: u64 = 0o3 | 0o37_777_700;
        const RESOLVE_NO_XDEV: u64 = 0x01;
        const RESOLVE_NO_MAGICLINKS: u64 = 0x02;
        const RESOLVE_NO_SYMLINKS: u64 = 0x04;
        const RESOLVE_BENEATH: u64 = 0x08;
        const RESOLVE_IN_ROOT: u64 = 0x10;
        const RESOLVE_CACHED: u64 = 0x20;
        const VALID_RESOLVE: u64 = 0x3f;
        let (o_directory, o_nofollow): (u64, u64) = match self.arch {
            Arch::X86_64 => (0o200000, 0o400000),
            Arch::Aarch64 => (0o40000, 0o100000),
        };
        if size < OPEN_HOW_SIZE_VER0 {
            return err(Errno::EINVAL);
        }
        if size > 4096 {
            return err(Errno::E2BIG);
        }
        let Ok(raw) = mem.read_vec(how, size as usize) else {
            return err(Errno::EFAULT);
        };
        if raw[24..].iter().any(|&b| b != 0) {
            return err(Errno::E2BIG);
        }
        let word = |i: usize| u64::from_le_bytes(raw[i * 8..i * 8 + 8].try_into().unwrap());
        let (flags, mode, resolve) = (word(0), word(1), word(2));
        if flags & !VALID_O != 0 || resolve & !VALID_RESOLVE != 0 {
            return err(Errno::EINVAL);
        }
        if flags & (O_CREAT | O_TMPFILE) == 0 && mode != 0 {
            return err(Errno::EINVAL);
        }
        if mode & !0o7777 != 0 {
            return err(Errno::EINVAL);
        }
        if flags & O_PATH != 0 && flags & !(O_PATH | o_directory | o_nofollow | O_CLOEXEC) != 0 {
            return err(Errno::EINVAL);
        }
        if resolve & RESOLVE_BENEATH != 0 && resolve & RESOLVE_IN_ROOT != 0 {
            return err(Errno::EINVAL);
        }
        // RESOLVE_CACHED promises a lookup that needs no I/O; anything that
        // would modify the filesystem can't honor that.
        if resolve & RESOLVE_CACHED != 0 && flags & (O_CREAT | O_TRUNC | O_TMPFILE) != 0 {
            return err(Errno::EAGAIN);
        }
        let Some(rel) = read_path(mem, pathptr) else {
            return err(Errno::EFAULT);
        };
        if rel.is_empty() {
            return err(Errno::ENOENT);
        }
        let restrict = RESOLVE_NO_XDEV
            | RESOLVE_NO_MAGICLINKS
            | RESOLVE_NO_SYMLINKS
            | RESOLVE_BENEATH
            | RESOLVE_IN_ROOT;
        if resolve & restrict == 0 {
            return self.open_path(vfs, cx, dirfd, &rel, flags, mode);
        }
        let base = if dirfd == AT_FDCWD {
            cx.cur.cwd.clone()
        } else {
            match cx.cur.fds.get(dirfd as i32) {
                Some(Fd::Dir { path, .. }) => path.clone(),
                Some(_) => return err(Errno::ENOTDIR),
                None => return err(Errno::EBADF),
            }
        };
        let walk = RestrictedWalk {
            beneath: resolve & RESOLVE_BENEATH != 0,
            in_root: resolve & RESOLVE_IN_ROOT != 0,
            no_symlinks: resolve & RESOLVE_NO_SYMLINKS != 0,
            no_magiclinks: resolve & RESOLVE_NO_MAGICLINKS != 0,
            no_xdev: resolve & RESOLVE_NO_XDEV != 0,
            follow_final: flags & o_nofollow == 0,
        };
        match walk.resolve(vfs, &base, &rel) {
            Ok(abs) => self.open_path(vfs, cx, AT_FDCWD, &abs, flags, mode),
            Err(e) => e,
        }
    }

    /// `cachestat(fd, struct cachestat_range *, struct cachestat *, flags)`:
    /// page-cache residency of a file range. Every byte of an in-memory file
    /// is "cached" and nothing is ever dirty, under writeback, or evicted.
    #[allow(clippy::unused_self, clippy::too_many_arguments)]
    pub(super) fn sys_cachestat(
        &self,
        vfs: &mut MountTable,
        cx: &ServiceCtx,
        fd: u64,
        range: u64,
        out: u64,
        flags: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const PAGE: u64 = 4096;
        if flags != 0 {
            return err(Errno::EINVAL);
        }
        let path = match cx.cur.fds.get(fd as i32) {
            Some(Fd::File { path, .. } | Fd::Dir { path, .. }) => path.clone(),
            Some(_) => return err(Errno::EOPNOTSUPP),
            None => return err(Errno::EBADF),
        };
        let (Ok(off), Ok(len)) = (mem.read_u64(range), mem.read_u64(range + 8)) else {
            return err(Errno::EFAULT);
        };
        let size = vfs.stat(&path).map_or(0, |a| a.size);
        let end = if len == 0 {
            size
        } else {
            off.saturating_add(len).min(size)
        };
        let first = off / PAGE;
        let last = end.div_ceil(PAGE);
        let nr_cache = last.saturating_sub(first);
        let mut b = [0u8; 40];
        b[0..8].copy_from_slice(&nr_cache.to_le_bytes());
        if mem.write(out, &b).is_err() {
            return err(Errno::EFAULT);
        }
        0
    }
}

/// The `RESOLVE_*` restrictions of one `openat2` path walk.
#[allow(clippy::struct_excessive_bools)]
struct RestrictedWalk {
    beneath: bool,
    in_root: bool,
    no_symlinks: bool,
    no_magiclinks: bool,
    no_xdev: bool,
    follow_final: bool,
}

impl RestrictedWalk {
    /// Walk `rel` from directory `base` component by component, expanding
    /// symlinks under the restrictions, and return the absolute result.
    /// `RESOLVE_BENEATH`: any escape above `base` (an absolute path or link, or
    /// a `..` past it) is `EXDEV`. `RESOLVE_IN_ROOT`: `base` acts as `/` —
    /// absolute paths/links restart there and `..` clamps at it.
    /// `RESOLVE_NO_SYMLINKS`: any symlink is `ELOOP` (the final one too, unless
    /// it isn't followed). `RESOLVE_NO_MAGICLINKS`: a `/proc` symlink (the
    /// `fd/N`, `exe`, `cwd` magic links) is `ELOOP`. `RESOLVE_NO_XDEV`:
    /// crossing a mount point is `EXDEV`.
    fn resolve(&self, vfs: &mut MountTable, base: &str, rel: &str) -> Result<String, i64> {
        const MAX_HOPS: u32 = 40;
        let comps = |p: &str| -> Vec<String> {
            p.split('/')
                .filter(|c| !c.is_empty())
                .map(str::to_string)
                .collect()
        };
        let anchor = comps(base);
        let floor = if self.beneath || self.in_root {
            anchor.len()
        } else {
            0
        };
        let join = |c: &[String]| format!("/{}", c.join("/"));
        let base_mount = vfs.mount_point_of(base);
        let mut cur: Vec<String> = if rel.starts_with('/') {
            if self.beneath {
                return Err(err(Errno::EXDEV));
            }
            if self.in_root {
                anchor.clone()
            } else {
                Vec::new()
            }
        } else {
            anchor.clone()
        };
        let mut todo: Vec<String> = comps(rel).into_iter().rev().collect();
        let mut hops = 0;
        while let Some(c) = todo.pop() {
            match c.as_str() {
                "." => continue,
                ".." => {
                    if cur.len() > floor {
                        cur.pop();
                    } else if self.beneath {
                        return Err(err(Errno::EXDEV));
                    } else if !self.in_root {
                        cur.pop();
                    }
                    continue;
                }
                _ => {}
            }
            let mut cand = cur.clone();
            cand.push(c);
            let cand_path = join(&cand);
            let is_final = todo.is_empty();
            match vfs.stat(&cand_path) {
                Some(a) if a.kind == NodeKind::Symlink && (!is_final || self.follow_final) => {
                    if self.no_symlinks || (self.no_magiclinks && cand_path.starts_with("/proc/")) {
                        return Err(err(Errno::ELOOP));
                    }
                    hops += 1;
                    if hops > MAX_HOPS {
                        return Err(err(Errno::ELOOP));
                    }
                    let target = vfs.readlink(&cand_path).map_err(|e| io_errno(&e))?;
                    if target.starts_with('/') {
                        if self.beneath {
                            return Err(err(Errno::EXDEV));
                        }
                        cur = if self.in_root {
                            anchor.clone()
                        } else {
                            Vec::new()
                        };
                    }
                    todo.extend(comps(&target).into_iter().rev());
                }
                Some(a) if a.kind == NodeKind::Symlink && self.no_symlinks => {
                    // An unfollowed final link is still a link under
                    // RESOLVE_NO_SYMLINKS (O_PATH|O_NOFOLLOW is the exception).
                    return Err(err(Errno::ELOOP));
                }
                _ => cur = cand,
            }
            if self.no_xdev && vfs.mount_point_of(&join(&cur)) != base_mount {
                return Err(err(Errno::EXDEV));
            }
        }
        let mut out = join(&cur);
        if rel.ends_with('/') && out != "/" {
            out.push('/');
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{BASE, call, e, put_str, setup};
    use crate::abi::arch::Sysno;
    use crate::abi::errno::Errno;

    const AT_FDCWD: u64 = (-100i64) as u64;

    #[test]
    fn splice_file_to_pipe_to_file_and_tee() {
        let (k, mut mem, mut v, mut cx) = setup();
        let (p_in, p_out, fds, fds2, offp, buf) = (
            BASE,
            BASE + 0x100,
            BASE + 0x200,
            BASE + 0x210,
            BASE + 0x300,
            BASE + 0x1000,
        );
        put_str(&mut mem, p_in, "/in");
        put_str(&mut mem, p_out, "/out");
        let fin = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_FDCWD, p_in, 0o102, 0o644, 0, 0],
        ) as u64;
        mem.write(buf, b"hello splice").unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Pwrite64,
                [fin, buf, 12, 0, 0, 0]
            ),
            12
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Pipe2,
                [fds, 0, 0, 0, 0, 0]
            ),
            0
        );
        let (r, w) = (
            u64::from(mem.read_u32(fds).unwrap()),
            u64::from(mem.read_u32(fds + 4).unwrap()),
        );
        // Neither end a pipe → EINVAL; an offset on the pipe end → ESPIPE.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Splice,
                [fin, 0, fin, 0, 4, 0]
            ),
            e(Errno::EINVAL)
        );
        mem.write_u64(offp, 6).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Splice,
                [fin, 0, w, offp, 4, 0]
            ),
            e(Errno::ESPIPE)
        );
        // file (from offset 6 via pointer) → pipe; the pointer advances, the
        // fd position doesn't.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Splice,
                [fin, offp, w, 0, 100, 0]
            ),
            6
        );
        assert_eq!(mem.read_u64(offp).unwrap(), 12);
        // tee into a second pipe leaves the first intact.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Pipe2,
                [fds2, 0, 0, 0, 0, 0]
            ),
            0
        );
        let (r2, w2) = (
            u64::from(mem.read_u32(fds2).unwrap()),
            u64::from(mem.read_u32(fds2 + 4).unwrap()),
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Tee,
                [r, w2, 100, 0, 0, 0]
            ),
            6
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Read,
                [r2, buf, 100, 0, 0, 0]
            ),
            6
        );
        assert_eq!(mem.read_vec(buf, 6).unwrap(), b"splice");
        // pipe → file at the file's own position.
        let fout = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_FDCWD, p_out, 0o102, 0o644, 0, 0],
        ) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Splice,
                [r, 0, fout, 0, 100, 0]
            ),
            6
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Pread64,
                [fout, buf, 100, 0, 0, 0]
            ),
            6
        );
        assert_eq!(mem.read_vec(buf, 6).unwrap(), b"splice");
        // The source pipe is now empty with a writer open: NONBLOCK → EAGAIN.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Splice,
                [r, 0, fout, 0, 100, 2]
            ),
            e(Errno::EAGAIN)
        );
        // vmsplice user memory into the pipe, then read it back.
        let iov = BASE + 0x400;
        mem.write_u64(iov, buf).unwrap();
        mem.write_u64(iov + 8, 3).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Vmsplice,
                [w, iov, 1, 0, 0, 0]
            ),
            3
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Vmsplice,
                [fin, iov, 1, 0, 0, 0]
            ),
            e(Errno::EBADF)
        );
    }

    #[test]
    fn preadv2_and_pwritev2_offsets_and_flags() {
        let (k, mut mem, mut v, mut cx) = setup();
        let (path, iov, buf) = (BASE, BASE + 0x100, BASE + 0x1000);
        put_str(&mut mem, path, "/f");
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_FDCWD, path, 0o102, 0o644, 0, 0],
        ) as u64;
        mem.write(buf, b"abcdef").unwrap();
        mem.write_u64(iov, buf).unwrap();
        mem.write_u64(iov + 8, 3).unwrap();
        let cur = u64::MAX; // -1: current position
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Pwritev2,
                [fd, iov, 1, cur, 0, 0]
            ),
            3
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Pwritev2,
                [fd, iov, 1, cur, 0, 0]
            ),
            3
        );
        // A positioned write doesn't move the file position…
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Pwritev2,
                [fd, iov, 1, 0, 0, 0]
            ),
            3
        );
        // …and RWF_APPEND ignores the offset.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Pwritev2,
                [fd, iov, 1, 0, 0, 0x10]
            ),
            3
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Lseek,
                [fd, 0, 2, 0, 0, 0]
            ),
            9
        );
        // Unknown RWF flag: EOPNOTSUPP.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Preadv2,
                [fd, iov, 1, 0, 0, 0x1000]
            ),
            e(Errno::EOPNOTSUPP)
        );
        mem.write_u64(iov + 8, 9).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Preadv2,
                [fd, iov, 1, 0, 0, 0]
            ),
            9
        );
        assert_eq!(mem.read_vec(buf, 9).unwrap(), b"abcabcabc");
    }

    #[test]
    fn openat2_validates_and_restricts_resolution() {
        let (k, mut mem, mut v, mut cx) = setup();
        let (path, how) = (BASE, BASE + 0x200);
        let set_how = |mem: &mut crate::vcpu::GuestMemory, flags: u64, mode: u64, resolve: u64| {
            mem.write_u64(how, flags).unwrap();
            mem.write_u64(how + 8, mode).unwrap();
            mem.write_u64(how + 16, resolve).unwrap();
            mem.write_u64(how + 24, 0).unwrap();
        };
        put_str(&mut mem, path, "/d");
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Mkdirat,
                [AT_FDCWD, path, 0o755, 0, 0, 0]
            ),
            0
        );
        put_str(&mut mem, path, "/d/f");
        set_how(&mut mem, 0o102, 0o644, 0);
        let f = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat2,
            [AT_FDCWD, path, how, 24, 0, 0],
        );
        assert!(f >= 0, "{f}");
        // Undersized struct; non-zero extension; mode without O_CREAT.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Openat2,
                [AT_FDCWD, path, how, 16, 0, 0]
            ),
            e(Errno::EINVAL)
        );
        mem.write_u64(how + 24, 1).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Openat2,
                [AT_FDCWD, path, how, 32, 0, 0]
            ),
            e(Errno::E2BIG)
        );
        set_how(&mut mem, 0, 0o644, 0);
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Openat2,
                [AT_FDCWD, path, how, 24, 0, 0]
            ),
            e(Errno::EINVAL)
        );
        // RESOLVE_BENEATH from /d: "f" is fine, "../d/f" and "/d/f" escape.
        put_str(&mut mem, path, "/d");
        let dfd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_FDCWD, path, 0, 0, 0, 0],
        ) as u64;
        set_how(&mut mem, 0, 0, 0x08);
        put_str(&mut mem, path, "f");
        assert!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Openat2,
                [dfd, path, how, 24, 0, 0]
            ) >= 0
        );
        put_str(&mut mem, path, "../d/f");
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Openat2,
                [dfd, path, how, 24, 0, 0]
            ),
            e(Errno::EXDEV)
        );
        put_str(&mut mem, path, "/d/f");
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Openat2,
                [dfd, path, how, 24, 0, 0]
            ),
            e(Errno::EXDEV)
        );
        // RESOLVE_IN_ROOT: "/f" is dirfd-relative.
        set_how(&mut mem, 0, 0, 0x10);
        put_str(&mut mem, path, "/../f");
        assert!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Openat2,
                [dfd, path, how, 24, 0, 0]
            ) >= 0
        );
        // RESOLVE_NO_SYMLINKS refuses a link.
        put_str(&mut mem, path, "f");
        put_str(&mut mem, BASE + 0x100, "/d/l");
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Symlinkat,
                [path, AT_FDCWD, BASE + 0x100, 0, 0, 0]
            ),
            0
        );
        set_how(&mut mem, 0, 0, 0x04);
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Openat2,
                [AT_FDCWD, BASE + 0x100, how, 24, 0, 0]
            ),
            e(Errno::ELOOP)
        );
        set_how(&mut mem, 0, 0, 0);
        assert!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Openat2,
                [AT_FDCWD, BASE + 0x100, how, 24, 0, 0]
            ) >= 0
        );
    }

    #[test]
    fn cachestat_reports_every_page_resident() {
        let (k, mut mem, mut v, mut cx) = setup();
        let (path, range, out) = (BASE, BASE + 0x100, BASE + 0x200);
        put_str(&mut mem, path, "/f");
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_FDCWD, path, 0o102, 0o644, 0, 0],
        ) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Ftruncate,
                [fd, 3 * 4096 + 1, 0, 0, 0, 0]
            ),
            0
        );
        mem.write_u64(range, 0).unwrap();
        mem.write_u64(range + 8, 0).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Cachestat,
                [fd, range, out, 0, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_u64(out).unwrap(), 4);
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Cachestat,
                [fd, range, out, 1, 0, 0]
            ),
            e(Errno::EINVAL)
        );
    }

    #[test]
    fn huge_counts_are_bounded_not_allocated() {
        let (k, mut mem, mut v, mut cx) = setup();
        let (path, buf) = (BASE, BASE + 0x1000);
        put_str(&mut mem, path, "/small");
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_FDCWD, path, 0o102, 0o644, 0, 0],
        ) as u64;
        mem.write(buf, b"tiny").unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Pwrite64,
                [fd, buf, 4, 0, 0, 0]
            ),
            4
        );
        // read(fd, buf, SSIZE_MAX) on a 4-byte file: 4, without trying to
        // allocate SSIZE_MAX host bytes.
        let huge = i64::MAX as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Read,
                [fd, buf, huge, 0, 0, 0]
            ),
            4
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Pread64,
                [fd, buf, huge, 0, 0, 0]
            ),
            4
        );
        // write of a huge count from memory that isn't there: EFAULT.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [fd, buf, huge, 0, 0, 0]
            ),
            e(Errno::EFAULT)
        );
        // copy_file_range with SIZE_MAX copies what there is.
        put_str(&mut mem, path, "/dst");
        let d = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_FDCWD, path, 0o102, 0o644, 0, 0],
        ) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Lseek,
                [fd, 0, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::CopyFileRange,
                [fd, 0, d, 0, u64::MAX >> 1, 0]
            ),
            4
        );
    }
}
