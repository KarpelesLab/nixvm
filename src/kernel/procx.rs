//! Syscalls that reach *another* process: the pidfd family (`pidfd_open`,
//! `pidfd_send_signal`, `pidfd_getfd`), `kcmp`, `process_vm_readv`/`writev`,
//! `process_madvise`, and `process_mrelease`.
//!
//! A pidfd is the same pollable descriptor `CLONE_PIDFD` hands out
//! ([`super::poll::PidfdInst`]): it turns `POLLIN` when its process exits,
//! `waitid(P_PIDFD)` waits on it, and here it also addresses signals and fd
//! theft. Python's asyncio (`PidfdChildWatcher`), systemd, and glibc's
//! `pidfd_spawn` build on exactly these.
//!
//! The VM is a single-user root machine, so the `PTRACE_MODE_ATTACH` checks
//! these calls make on Linux always pass; what remains are the existence and
//! argument checks, which are kept faithful.

use super::{Fd, Kernel, ProcInfo, RunState, ServiceCtx, Shared, err};
use crate::abi::errno::Errno;
use crate::vcpu::GuestMemory;
use poll::PidfdInst;

use super::poll;

/// `PIDFD_NONBLOCK` (= `O_NONBLOCK`).
const PIDFD_NONBLOCK: u64 = 0o4000;
/// `PIDFD_THREAD` (= `O_EXCL`, 6.9): the pid may name a non-leader thread.
const PIDFD_THREAD: u64 = 0o200;
/// `UIO_MAXIOV`.
const UIO_MAXIOV: u64 = 1024;

/// The task `pid` names: the caller itself, or a live table entry (a zombie
/// still counts until reaped, as on Linux).
fn find_proc<'a>(sh: &'a Shared, cx: &'a ServiceCtx, pid: i32) -> Option<&'a ProcInfo> {
    if cx.cur.pid == pid {
        return Some(&cx.cur);
    }
    sh.procs
        .iter()
        .flatten()
        .map(|p| &p.info)
        .find(|p| p.pid == pid)
}

/// Read a `struct iovec[cnt]` array at `iov`: `(base, len)` pairs.
fn read_iovs(mem: &GuestMemory, iov: u64, cnt: u64) -> Result<Vec<(u64, u64)>, i64> {
    if cnt > UIO_MAXIOV {
        return Err(err(Errno::EINVAL));
    }
    let mut v = Vec::with_capacity(cnt as usize);
    for i in 0..cnt {
        let (Ok(b), Ok(l)) = (mem.read_u64(iov + i * 16), mem.read_u64(iov + i * 16 + 8)) else {
            return Err(err(Errno::EFAULT));
        };
        if (l as i64) < 0 {
            return Err(err(Errno::EINVAL));
        }
        v.push((b, l));
    }
    Ok(v)
}

impl Kernel {
    /// The pid a pidfd refers to and its `O_NONBLOCK` state: `EBADF` for a
    /// closed fd or one that is not a pidfd.
    pub(super) fn pidfd_target(&self, cx: &ServiceCtx, fd: i32) -> Result<(i32, bool), i64> {
        match cx.cur.fds.get(fd) {
            Some(Fd::Pidfd(i)) => {
                let pf = self.pollfds.lock().unwrap();
                let p = &pf.pidfds[*i];
                Ok((p.target_pid, p.nonblock))
            }
            _ => Err(err(Errno::EBADF)),
        }
    }

    /// `pidfd_open(pid, flags)`: a close-on-exec pidfd for process `pid`
    /// (already readable if it has exited). Only a thread-group leader may be
    /// named unless `PIDFD_THREAD` is given.
    pub(super) fn sys_pidfd_open(
        &self,
        sh: &Shared,
        cx: &mut ServiceCtx,
        pid: u64,
        flags: u64,
    ) -> i64 {
        if flags & !(PIDFD_NONBLOCK | PIDFD_THREAD) != 0 {
            return err(Errno::EINVAL);
        }
        let pid = pid as i32;
        if pid <= 0 {
            return err(Errno::EINVAL);
        }
        let Some(p) = find_proc(sh, cx, pid) else {
            return err(Errno::ESRCH);
        };
        if p.is_thread && flags & PIDFD_THREAD == 0 {
            return err(Errno::EINVAL);
        }
        let exited = matches!(p.run, RunState::Zombie(_));
        let idx = {
            let mut pf = self.pollfds.lock().unwrap();
            pf.pidfds.push(PidfdInst {
                target_pid: pid,
                exited,
                nonblock: flags & PIDFD_NONBLOCK != 0,
            });
            pf.pidfds.len() - 1
        };
        let fd = cx.cur.fds.alloc(Fd::Pidfd(idx));
        cx.cur.fds.set_cloexec(fd, true);
        i64::from(fd)
    }

    /// `pidfd_send_signal(pidfd, sig, info, flags)`: `kill`/`rt_sigqueueinfo`
    /// addressed by pidfd. `ESRCH` once the process has been reaped. The scope
    /// flags pick the thread, the thread group (default), or its process group.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_pidfd_send_signal(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        pidfd: u64,
        sig: u64,
        info: u64,
        flags: u64,
        mem: &GuestMemory,
    ) -> i64 {
        const PIDFD_SIGNAL_THREAD: u64 = 1;
        const PIDFD_SIGNAL_THREAD_GROUP: u64 = 2;
        const PIDFD_SIGNAL_PROCESS_GROUP: u64 = 4;
        let scope =
            flags & (PIDFD_SIGNAL_THREAD | PIDFD_SIGNAL_THREAD_GROUP | PIDFD_SIGNAL_PROCESS_GROUP);
        if flags != scope || scope.count_ones() > 1 {
            return err(Errno::EINVAL);
        }
        if sig > 64 {
            return err(Errno::EINVAL);
        }
        let pid = match self.pidfd_target(cx, pidfd as i32) {
            Ok((pid, _)) => pid,
            Err(e) => return e,
        };
        let Some(p) = find_proc(sh, cx, pid) else {
            return err(Errno::ESRCH);
        };
        let target = if scope == PIDFD_SIGNAL_PROCESS_GROUP {
            -i64::from(super::pgid_of(p))
        } else {
            i64::from(pid)
        };
        if info != 0 {
            // The caller-supplied siginfo must describe this very signal, and
            // may only forge a kernel si_code (>= 0) to itself.
            let (Ok(signo), Ok(code)) = (mem.read_u32(info), mem.read_u32(info + 8)) else {
                return err(Errno::EFAULT);
            };
            if u64::from(signo) != sig {
                return err(Errno::EINVAL);
            }
            if (code as i32) >= 0 && p.tgid != cx.cur.tgid {
                return err(Errno::EPERM);
            }
            return self.sys_rt_sigqueueinfo(sh, cx, target, sig, info, mem);
        }
        self.sys_kill(sh, cx, target, sig)
    }

    /// `pidfd_getfd(pidfd, targetfd, flags)`: duplicate another process's
    /// descriptor into the caller (close-on-exec), sharing the open file.
    pub(super) fn sys_pidfd_getfd(
        &self,
        sh: &Shared,
        cx: &mut ServiceCtx,
        pidfd: u64,
        targetfd: u64,
        flags: u64,
    ) -> i64 {
        if flags != 0 {
            return err(Errno::EINVAL);
        }
        let pid = match self.pidfd_target(cx, pidfd as i32) {
            Ok((pid, _)) => pid,
            Err(e) => return e,
        };
        let Some(p) = find_proc(sh, cx, pid) else {
            return err(Errno::ESRCH);
        };
        if matches!(p.run, RunState::Zombie(_)) {
            return err(Errno::ESRCH);
        }
        let files = p.files;
        let got = if files == cx.cur.files {
            cx.cur.fds.get(targetfd as i32).cloned()
        } else {
            match sh.file_tables.get(files) {
                Some(Some(t)) => t.get(targetfd as i32).cloned(),
                // The target is mid-slice on another CPU with its table checked
                // out: try again shortly.
                Some(None) => return err(Errno::EAGAIN),
                None => None,
            }
        };
        let Some(f) = got else {
            return err(Errno::EBADF);
        };
        self.bump_pipe(&f, true);
        let n = cx.cur.fds.alloc(f);
        cx.cur.fds.set_cloexec(n, true);
        i64::from(n)
    }

    /// `kcmp(pid1, pid2, type, idx1, idx2)`: whether two processes share a
    /// kernel resource — 0 if equal, else 1/2 for a consistent ordering.
    #[allow(clippy::too_many_arguments, clippy::unused_self)]
    pub(super) fn sys_kcmp(
        &self,
        sh: &Shared,
        cx: &ServiceCtx,
        pid1: u64,
        pid2: u64,
        kind: u64,
        idx1: u64,
        idx2: u64,
    ) -> i64 {
        const KCMP_FILE: u64 = 0;
        const KCMP_VM: u64 = 1;
        const KCMP_FILES: u64 = 2;
        const KCMP_FS: u64 = 3;
        const KCMP_SIGHAND: u64 = 4;
        const KCMP_IO: u64 = 5;
        const KCMP_SYSVSEM: u64 = 6;
        let (Some(a), Some(b)) = (
            find_proc(sh, cx, pid1 as i32),
            find_proc(sh, cx, pid2 as i32),
        ) else {
            return err(Errno::ESRCH);
        };
        // An fd's identity: a regular file by its open file description (shared
        // by dups and fork children), anything else by the object it names.
        let fd_key = |p: &ProcInfo, fd: u64| -> Option<String> {
            let table = if p.pid == cx.cur.pid || p.files == cx.cur.files {
                Some(&cx.cur.fds)
            } else {
                sh.file_tables.get(p.files).and_then(Option::as_ref)
            }?;
            Some(match table.get(fd as i32)? {
                Fd::File { offset, .. } => format!("file:{:x}", offset.id()),
                Fd::Dir { path, .. } => format!("dir:{path}"),
                other => format!("{other:?}"),
            })
        };
        let (ka, kb): (String, String) = match kind {
            KCMP_FILE => match (fd_key(a, idx1), fd_key(b, idx2)) {
                (Some(x), Some(y)) => (x, y),
                _ => return err(Errno::EBADF),
            },
            KCMP_VM => (a.mm.to_string(), b.mm.to_string()),
            KCMP_FILES => (a.files.to_string(), b.files.to_string()),
            KCMP_FS => (a.fs.to_string(), b.fs.to_string()),
            // Handlers, I/O context and SysV undo lists are per thread group.
            KCMP_SIGHAND | KCMP_IO | KCMP_SYSVSEM => (a.tgid.to_string(), b.tgid.to_string()),
            _ => return err(Errno::EINVAL),
        };
        match ka.cmp(&kb) {
            std::cmp::Ordering::Equal => 0,
            std::cmp::Ordering::Less => 1,
            std::cmp::Ordering::Greater => 2,
        }
    }

    /// `process_vm_readv`/`process_vm_writev(pid, local_iov, liovcnt,
    /// remote_iov, riovcnt, flags)`: copy between the caller's memory and
    /// another process's, as one stream across both iovec lists. Returns the
    /// bytes moved; a fault part-way returns the partial count (`EFAULT` only
    /// if nothing moved).
    ///
    /// The target's address space is a separate lock; it is only *tried*
    /// (memory is the outermost lock and the caller already holds its own, so
    /// waiting could deadlock against a target doing the same). A target busy
    /// on another CPU makes the call re-trap until it's free.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_process_vm(
        &self,
        sh: &Shared,
        cx: &mut ServiceCtx,
        a: &[u64; 6],
        write: bool,
        mem: &mut GuestMemory,
    ) -> i64 {
        let (pid, liov, lcnt, riov, rcnt, flags) = (a[0] as i32, a[1], a[2], a[3], a[4], a[5]);
        if flags != 0 {
            return err(Errno::EINVAL);
        }
        let (local, remote) = match (read_iovs(mem, liov, lcnt), read_iovs(mem, riov, rcnt)) {
            (Ok(l), Ok(r)) => (l, r),
            (Err(e), _) | (_, Err(e)) => return e,
        };
        let Some(p) = find_proc(sh, cx, pid) else {
            return err(Errno::ESRCH);
        };
        if matches!(p.run, RunState::Zombie(_)) {
            return err(Errno::ESRCH);
        }
        let mm = p.mm;
        // Same address space: both sides are `mem`.
        if mm == cx.cur.mm {
            return copy_streams(mem, None, &local, &remote, write);
        }
        let Some(space) = sh.spaces.get(mm) else {
            return err(Errno::ESRCH);
        };
        let Ok(mut other) = space.try_lock() else {
            cx.block = true;
            return 0;
        };
        copy_streams(mem, Some(&mut other), &local, &remote, write)
    }

    /// `process_madvise(pidfd, iovec, vlen, advice, flags)`: advise another
    /// process's memory. The advice Linux allows here (`COLD`, `PAGEOUT`,
    /// `WILLNEED`, `COLLAPSE`) is purely a reclaim/readahead hint with nothing
    /// to act on in a VM without swap, so it is validated and reported as
    /// fully applied.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_process_madvise(
        &self,
        sh: &Shared,
        cx: &ServiceCtx,
        pidfd: u64,
        iov: u64,
        vlen: u64,
        advice: u64,
        flags: u64,
        mem: &GuestMemory,
    ) -> i64 {
        const MADV_WILLNEED: u64 = 3;
        const MADV_COLD: u64 = 20;
        const MADV_PAGEOUT: u64 = 21;
        const MADV_COLLAPSE: u64 = 25;
        if flags != 0
            || !matches!(
                advice,
                MADV_WILLNEED | MADV_COLD | MADV_PAGEOUT | MADV_COLLAPSE
            )
        {
            return err(Errno::EINVAL);
        }
        let pid = match self.pidfd_target(cx, pidfd as i32) {
            Ok((pid, _)) => pid,
            Err(e) => return e,
        };
        if find_proc(sh, cx, pid).is_none_or(|p| matches!(p.run, RunState::Zombie(_))) {
            return err(Errno::ESRCH);
        }
        match read_iovs(mem, iov, vlen) {
            Ok(v) => v.iter().map(|&(_, l)| l as i64).sum(),
            Err(e) => e,
        }
    }

    /// `process_mrelease(pidfd, flags)`: reap a dying process's memory early.
    /// Only valid on a process that is already exiting (`EINVAL` otherwise);
    /// its memory is released at exit here anyway, so that is all there is.
    pub(super) fn sys_process_mrelease(
        &self,
        sh: &Shared,
        cx: &ServiceCtx,
        pidfd: u64,
        flags: u64,
    ) -> i64 {
        if flags != 0 {
            return err(Errno::EINVAL);
        }
        let pid = match self.pidfd_target(cx, pidfd as i32) {
            Ok((pid, _)) => pid,
            Err(e) => return e,
        };
        match find_proc(sh, cx, pid) {
            None => err(Errno::ESRCH),
            Some(p) if matches!(p.run, RunState::Zombie(_)) => 0,
            Some(_) => err(Errno::EINVAL),
        }
    }

    /// Answer a syscall on `/proc/<pid>/mem`, if this is one: its open, and
    /// reads, writes and seeks on the descriptor. The file is the task's
    /// address space, each offset an address: a debugger's or a seccomp
    /// supervisor's way into another process's memory. Opening it takes
    /// `PTRACE_MODE_ATTACH` rights (the caller's own process, the same user
    /// with the target dumpable, or privilege; `EACCES` otherwise) and binds
    /// the descriptor to the address space the task has then — after the
    /// task exits or `execve`s, it reads as empty. Reads and writes go a page
    /// at a time and stop at the first one that can't be accessed (`EIO` if
    /// that's the first); writes reach read-only pages too, as Linux's
    /// `FOLL_FORCE` does.
    pub(super) fn proc_mem_syscall(
        &self,
        cx: &mut ServiceCtx,
        sys: crate::abi::arch::Sysno,
        args: &[u64; 6],
        mem: &mut GuestMemory,
    ) -> Option<i64> {
        use crate::abi::arch::Sysno;
        let on_mem_fd = || matches!(cx.cur.fds.get(args[0] as i32), Some(Fd::ProcMem { .. }));
        match sys {
            Sysno::Open | Sysno::Openat | Sysno::Openat2 => {
                const AT_FDCWD: i64 = -100;
                let (dirfd, ptr) = if sys == Sysno::Open {
                    (AT_FDCWD, args[0])
                } else {
                    (args[0] as i64, args[1])
                };
                let raw = super::read_path(mem, ptr)?;
                if !raw.ends_with("/mem") {
                    return None;
                }
                let flags = match sys {
                    Sysno::Open => args[1],
                    Sysno::Openat => args[2],
                    _ => mem.read_u64(args[2]).ok()?,
                };
                let abs = self.resolve_path(cx, dirfd, &raw);
                let inside = Self::as_seen_from_root(cx, &abs);
                let rest = &inside[inside.rfind("/proc/")? + "/proc/".len()..];
                let who = rest.strip_suffix("/mem")?;
                Some(self.open_proc_mem(cx, who, flags))
            }
            Sysno::Read | Sysno::Pread64 | Sysno::Write | Sysno::Pwrite64 if on_mem_fd() => {
                let Some(Fd::ProcMem {
                    pid,
                    mm,
                    offset,
                    readable,
                    writable,
                }) = cx.cur.fds.get(args[0] as i32).cloned()
                else {
                    return None;
                };
                let write = matches!(sys, Sysno::Write | Sysno::Pwrite64);
                if (write && !writable) || (!write && !readable) {
                    return Some(err(Errno::EBADF));
                }
                let positional = matches!(sys, Sysno::Pread64 | Sysno::Pwrite64);
                let at = if positional { args[3] } else { offset.get() };
                let n = self.proc_mem_rw(cx, pid, mm, at, args[1], args[2], write, mem);
                if n > 0 && !positional {
                    offset.set(at.wrapping_add(n as u64));
                }
                Some(n)
            }
            // `mem_lseek`: SEEK_SET and SEEK_CUR only, any address.
            Sysno::Lseek if on_mem_fd() => {
                let Some(Fd::ProcMem { offset, .. }) = cx.cur.fds.get(args[0] as i32) else {
                    return None;
                };
                let pos = match args[2] {
                    0 => args[1],
                    1 => offset.get().wrapping_add(args[1]),
                    _ => return Some(err(Errno::EINVAL)),
                };
                offset.set(pos);
                Some(pos as i64)
            }
            _ => None,
        }
    }

    /// Open `/proc/<who>/mem`.
    fn open_proc_mem(&self, cx: &mut ServiceCtx, who: &str, flags: u64) -> i64 {
        const O_ACCMODE: u64 = 3;
        const O_CLOEXEC: u64 = 0o2000000;
        // `<pid>`, `<pid>/task/<tid>`, `self`, `thread-self`.
        let local = who.rsplit('/').next().unwrap_or(who);
        let global = match local {
            "self" => Some(cx.cur.tgid),
            "thread-self" => Some(cx.cur.pid),
            n => n.parse().ok().and_then(|l| self.global_pid(cx, l)),
        };
        let Some(global) = global else {
            return err(Errno::ENOENT);
        };
        let target = {
            let sh = self.shared.lock().unwrap();
            let Some(p) = find_proc(&sh, cx, global) else {
                return err(Errno::ENOENT);
            };
            (
                p.tgid,
                p.mm,
                p.creds,
                p.dumpable,
                matches!(p.run, RunState::Zombie(_)),
                p.ns.user,
            )
        };
        let (tgid, mm, creds, dumpable, dead, userns) = target;
        // CAP_SYS_PTRACE over the target's user namespace.
        const CAP_SYS_PTRACE: u32 = 19;
        if tgid != cx.cur.tgid && !self.ns_capable(cx, userns, CAP_SYS_PTRACE) {
            let me = &cx.cur.creds;
            let same_user = [creds.ruid, creds.euid, creds.suid]
                .iter()
                .all(|&u| u == me.fsuid)
                && [creds.rgid, creds.egid, creds.sgid]
                    .iter()
                    .all(|&g| g == me.fsgid);
            if !same_user || dumpable != 1 {
                return err(Errno::EACCES);
            }
        }
        let fd = Fd::ProcMem {
            // A zombie has no memory: its descriptor reads nothing.
            pid: if dead { 0 } else { tgid },
            mm,
            offset: super::FileOffset::new(0),
            readable: flags & O_ACCMODE != 1,
            writable: flags & O_ACCMODE != 0,
        };
        let n = cx.cur.fds.alloc(fd);
        cx.cur.fds.set_cloexec(n, flags & O_CLOEXEC != 0);
        i64::from(n)
    }

    /// Move `count` bytes between the caller's `buf` and address `at` of
    /// address space `mm` (if process `pid` still has it).
    #[allow(clippy::too_many_arguments)]
    fn proc_mem_rw(
        &self,
        cx: &mut ServiceCtx,
        pid: i32,
        mm: usize,
        at: u64,
        buf: u64,
        count: u64,
        write: bool,
        mem: &mut GuestMemory,
    ) -> i64 {
        if count == 0 {
            return 0;
        }
        if mm == cx.cur.mm && pid == cx.cur.tgid {
            return mem_copy(mem, None, at, buf, count, write);
        }
        let sh = self.shared.lock().unwrap();
        let alive = sh.procs.iter().flatten().any(|p| {
            p.info.tgid == pid && p.info.mm == mm && !matches!(p.info.run, RunState::Zombie(_))
        });
        let Some(space) = sh.spaces.get(mm).filter(|_| alive && pid != 0) else {
            return 0;
        };
        // Memory is the outermost lock and the caller holds its own: only try
        // the target's, and come back if it's busy on another CPU.
        let Ok(mut other) = space.try_lock() else {
            cx.block = true;
            return 0;
        };
        mem_copy(mem, Some(&mut other), at, buf, count, write)
    }
}

/// `mem_rw`: copy `count` bytes between the caller's `buf` (in `local`) and
/// address `at` of `remote` (or `local` itself), a page at a time, stopping at
/// the first page that can't be accessed. Writes to `remote` ignore its page
/// protection (`FOLL_FORCE`).
fn mem_copy(
    local: &mut GuestMemory,
    mut remote: Option<&mut GuestMemory>,
    at: u64,
    buf: u64,
    count: u64,
    write: bool,
) -> i64 {
    const PAGE: u64 = crate::vcpu::mem::PAGE_SIZE;
    let mut done = 0u64;
    while done < count {
        let addr = at.wrapping_add(done);
        let n = (count - done).min(PAGE - addr % PAGE);
        let (local_ok, remote_ok) = if write {
            match local.read_vec(buf + done, n as usize) {
                Ok(data) => {
                    let target = remote.as_deref_mut().unwrap_or(local);
                    (true, target.write_init(addr, &data).is_ok())
                }
                Err(_) => (false, true),
            }
        } else {
            match remote
                .as_deref()
                .unwrap_or(local)
                .read_vec(addr, n as usize)
            {
                Ok(data) => (local.write(buf + done, &data).is_ok(), true),
                Err(_) => (true, false),
            }
        };
        if !(local_ok && remote_ok) {
            if done > 0 {
                break;
            }
            return err(if local_ok { Errno::EIO } else { Errno::EFAULT });
        }
        done += n;
    }
    done as i64
}

/// Copy between the local iovecs (in `local_mem`) and the remote ones (in
/// `remote_mem`, or `local_mem` itself when `None`), in order, until either
/// list runs out. `write` copies local → remote.
fn copy_streams(
    local_mem: &mut GuestMemory,
    mut remote_mem: Option<&mut GuestMemory>,
    local: &[(u64, u64)],
    remote: &[(u64, u64)],
    write: bool,
) -> i64 {
    let (mut li, mut loff, mut ri, mut roff) = (0usize, 0u64, 0usize, 0u64);
    let mut total = 0i64;
    while li < local.len() && ri < remote.len() {
        let (lb, ll) = local[li];
        let (rb, rl) = remote[ri];
        if loff >= ll {
            li += 1;
            loff = 0;
            continue;
        }
        if roff >= rl {
            ri += 1;
            roff = 0;
            continue;
        }
        let n = (ll - loff).min(rl - roff).min(64 * 1024) as usize;
        let (src_addr, dst_addr) = if write {
            (lb + loff, rb + roff)
        } else {
            (rb + roff, lb + loff)
        };
        // Read the source side.
        let data = if write {
            local_mem.read_vec(src_addr, n)
        } else {
            match remote_mem.as_deref() {
                Some(r) => r.read_vec(src_addr, n),
                None => local_mem.read_vec(src_addr, n),
            }
        };
        let Ok(data) = data else {
            return if total > 0 { total } else { err(Errno::EFAULT) };
        };
        let wrote = if write {
            match remote_mem.as_deref_mut() {
                Some(r) => r.write(dst_addr, &data),
                None => local_mem.write(dst_addr, &data),
            }
        } else {
            local_mem.write(dst_addr, &data)
        };
        if wrote.is_err() {
            return if total > 0 { total } else { err(Errno::EFAULT) };
        }
        total += n as i64;
        loff += n as u64;
        roff += n as u64;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{BASE, call, e, setup};
    use super::super::{ProcInfo, Process, RunState};
    use crate::abi::arch::Sysno;
    use crate::abi::errno::Errno;
    use crate::vcpu::GuestMemory;
    use crate::vcpu::mem::{PAGE_SIZE, Prot};
    use std::sync::{Arc, Mutex};

    /// Add a child process (pid 2, its own address space mm 1 with one mapped
    /// page at BASE) to the kernel's table.
    fn add_child(k: &super::Kernel, mem_page: &[u8]) {
        let mut sh = k.shared.lock().unwrap();
        while sh.spaces.is_empty() {
            sh.spaces
                .push(Arc::new(Mutex::new(GuestMemory::new(BASE, PAGE_SIZE))));
        }
        let mut m = GuestMemory::new(BASE, 16 * PAGE_SIZE);
        m.map(BASE, PAGE_SIZE, Prot::rw()).unwrap();
        m.write(BASE, mem_page).unwrap();
        sh.spaces.push(Arc::new(Mutex::new(m)));
        let info = ProcInfo {
            pid: 2,
            tgid: 2,
            ppid: 1,
            mm: 1,
            files: 7,
            run: RunState::Running,
            ..ProcInfo::default()
        };
        sh.procs.push(Some(Process { vcpu: None, info }));
    }

    #[test]
    fn pidfd_open_signal_and_wait() {
        let (k, mut mem, mut v, mut cx) = setup();
        add_child(&k, b"x");
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::PidfdOpen,
                [2, 0x1, 0, 0, 0, 0]
            ),
            e(Errno::EINVAL)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::PidfdOpen,
                [99, 0, 0, 0, 0, 0]
            ),
            e(Errno::ESRCH)
        );
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::PidfdOpen,
            [2, 0o4000, 0, 0, 0, 0],
        );
        assert!(fd >= 0);
        assert!(cx.cur.fds.is_cloexec(fd as i32), "pidfds are close-on-exec");
        // SIGUSR1 via the pidfd lands on the child.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::PidfdSendSignal,
                [fd as u64, 10, 0, 0, 0, 0]
            ),
            0
        );
        {
            let sh = k.shared.lock().unwrap();
            let child = sh.procs.iter().flatten().find(|p| p.info.pid == 2).unwrap();
            assert_ne!(child.info.pending & (1 << 9), 0);
        }
        // waitid(P_PIDFD) on a live child through an O_NONBLOCK pidfd: EAGAIN.
        let info = BASE + 0x800;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Waitid,
                [3, fd as u64, info, 4, 0, 0]
            ),
            e(Errno::EAGAIN)
        );
        // WNOHANG instead: 0 with si_pid zeroed.
        mem.write(info, &[0xffu8; 128]).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Waitid,
                [3, fd as u64, info, 5, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_u32(info + 16).unwrap(), 0);
        // No WEXITED/WSTOPPED/WCONTINUED at all is EINVAL.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Waitid,
                [0, 0, info, 1, 0, 0]
            ),
            e(Errno::EINVAL)
        );
        // A non-pidfd is EBADF.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::PidfdSendSignal,
                [1, 10, 0, 0, 0, 0]
            ),
            e(Errno::EBADF)
        );
        // process_mrelease on a live process is EINVAL.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::ProcessMrelease,
                [fd as u64, 0, 0, 0, 0, 0]
            ),
            e(Errno::EINVAL)
        );
    }

    #[test]
    fn process_vm_readv_and_writev_cross_address_spaces() {
        let (k, mut mem, mut v, mut cx) = setup();
        add_child(&k, b"remote bytes");
        let (liov, riov, buf) = (BASE + 0x100, BASE + 0x200, BASE + 0x1000);
        mem.write_u64(liov, buf).unwrap();
        mem.write_u64(liov + 8, 6).unwrap();
        mem.write_u64(riov, BASE).unwrap();
        mem.write_u64(riov + 8, 12).unwrap();
        // Read 6 bytes (the local iovec is the shorter list).
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::ProcessVmReadv,
                [2, liov, 1, riov, 1, 0]
            ),
            6
        );
        assert_eq!(mem.read_vec(buf, 6).unwrap(), b"remote");
        // Write them back over the tail of the child's buffer.
        mem.write(buf, b"LOCAL!").unwrap();
        mem.write_u64(riov, BASE + 6).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::ProcessVmWritev,
                [2, liov, 1, riov, 1, 0]
            ),
            6
        );
        let sh = k.shared.lock().unwrap();
        assert_eq!(
            sh.spaces[1].lock().unwrap().read_vec(BASE, 12).unwrap(),
            b"remoteLOCAL!"
        );
        drop(sh);
        // An unmapped remote address faults with nothing moved.
        mem.write_u64(riov, BASE + 0x8000).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::ProcessVmReadv,
                [2, liov, 1, riov, 1, 0]
            ),
            e(Errno::EFAULT)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::ProcessVmReadv,
                [2, liov, 1, riov, 1, 1]
            ),
            e(Errno::EINVAL)
        );
    }

    #[test]
    fn kcmp_compares_shared_resources() {
        let (k, mut mem, mut v, mut cx) = setup();
        add_child(&k, b"");
        // Same process: everything equal.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Kcmp,
                [1, 1, 1, 0, 0, 0]
            ),
            0
        );
        // Different address spaces / fd tables: unequal (1 or 2).
        let r = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Kcmp,
            [1, 2, 1, 0, 0, 0],
        );
        assert!(r == 1 || r == 2);
        // KCMP_FILE: a dup of fd 1 is the same file; fd 1 vs 0 is not.
        let d = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Dup,
            [1, 0, 0, 0, 0, 0],
        ) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Kcmp,
                [1, 1, 0, 1, d, 0]
            ),
            0
        );
        assert_ne!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Kcmp,
                [1, 1, 0, 1, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Kcmp,
                [1, 1, 0, 1, 77, 0]
            ),
            e(Errno::EBADF)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Kcmp,
                [1, 9, 1, 0, 0, 0]
            ),
            e(Errno::ESRCH)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Kcmp,
                [1, 1, 99, 0, 0, 0]
            ),
            e(Errno::EINVAL)
        );
    }
}
