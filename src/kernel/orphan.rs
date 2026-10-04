//! Open-but-unlinked files, renames under open descriptors, and `O_TMPFILE`.
//!
//! Descriptors here name their file by path ([`Fd::File`]), so without help a
//! file vanished from under its open descriptors the moment it was unlinked or
//! renamed — breaking one of POSIX's most used idioms: create a temp file,
//! unlink it, keep using the fd (`tmpfile()`, Python's `TemporaryFile`,
//! `multiprocessing`'s shared-memory arenas, sqlite's temp databases, every
//! "delete on close" scheme).
//!
//! The fix keeps the descriptor's view intact:
//! - unlinking a file that some descriptor still has open *moves* it to a
//!   hidden name at the root of its mount (an "orphan") and repoints the open
//!   descriptors there; the name is gone (`ENOENT` for a new open), the data
//!   lives on, and once no descriptor names the orphan it is really deleted;
//! - renaming a file or directory repoints every descriptor under the old
//!   name, and a rename *over* an open file orphans the replaced file first;
//! - `O_TMPFILE` creates an orphan directly, and `linkat` of an orphan gives
//!   it a name (moving it, so the descriptor keeps writing the named file).
//!
//! Orphans are hidden from directory listings. Descriptors of tasks that are
//! mid-slice on another CPU (their table checked out) aren't repointed — a
//! narrow SMP window where the old behavior remains.

use super::{Fd, Kernel, ServiceCtx, Shared, err, io_errno, read_path};
use crate::abi::errno::Errno;
use crate::fs::{MountTable, NodeKind};
use crate::vcpu::GuestMemory;

/// The name prefix of a hidden orphan (filtered out of `getdents`).
pub(super) const ORPHAN_PREFIX: &str = ".nixvm-orphan-";

/// Whether a descriptor names `path` itself or (for a directory rename)
/// something beneath it.
fn names(fd_path: &str, path: &str) -> bool {
    fd_path == path
        || fd_path
            .strip_prefix(path)
            .is_some_and(|rest| rest.starts_with('/'))
}

impl Kernel {
    /// A fresh hidden name in directory `dir`.
    fn orphan_in(&self, dir: &str) -> String {
        let n = self
            .orphan_seq
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let sep = if dir.ends_with('/') { "" } else { "/" };
        format!("{dir}{sep}{ORPHAN_PREFIX}{n}")
    }

    /// Candidate homes for an orphan of `path` (a file, or for `O_TMPFILE` the
    /// directory itself): the root of its mount first — so the directory it
    /// came from can still be removed — then the directory itself, for
    /// backends with a writable subtree only (`/dev/shm` inside `/dev`).
    fn orphan_homes(&self, vfs: &mut MountTable, dir: &str) -> [String; 2] {
        let mp = vfs.mount_point_of(dir).unwrap_or_else(|| "/".to_string());
        [self.orphan_in(&mp), self.orphan_in(dir)]
    }

    /// Move file `path` to a fresh orphan name, returning it.
    fn orphan_move(&self, vfs: &mut MountTable, path: &str) -> std::io::Result<String> {
        let parent = super::parent_of(path).to_string();
        let [root, local] = self.orphan_homes(vfs, &parent);
        match vfs.rename(path, &root) {
            Ok(()) => Ok(root),
            Err(_) => vfs.rename(path, &local).map(|()| local),
        }
    }

    /// Does any descriptor — the caller's or a checked-in table's — name
    /// `path` (or a path under it)?
    fn path_is_open(sh: &Shared, cx: &ServiceCtx, path: &str) -> bool {
        let hit = |f: &Fd| matches!(f, Fd::File { path: p, .. } | Fd::Dir { path: p, .. } if names(p, path));
        cx.cur.fds.values().any(hit) || sh.file_tables.iter().flatten().any(|t| t.values().any(hit))
    }

    /// Repoint every descriptor naming `from` (or a path under it) at `to`.
    fn repoint(sh: &mut Shared, cx: &mut ServiceCtx, from: &str, to: &str) {
        let fix = |p: &mut String| {
            if names(p, from) {
                *p = format!("{to}{}", &p[from.len()..]);
            }
        };
        let apply = |t: &mut super::FdTable| {
            let open: Vec<i32> = t.iter().map(|(n, _)| n).collect();
            for n in open {
                if let Some(Fd::File { path, .. } | Fd::Dir { path, .. }) = t.get_mut(n) {
                    fix(path);
                }
            }
        };
        apply(&mut cx.cur.fds);
        for t in sh.file_tables.iter_mut().flatten() {
            apply(t);
        }
    }

    /// Really delete every orphan no descriptor names any more.
    pub(super) fn reap_orphans(&self, sh: &Shared, cx: &ServiceCtx, vfs: &mut MountTable) {
        let mut orphans = self.orphans.lock().unwrap();
        if orphans.is_empty() {
            return;
        }
        orphans.retain(|o| {
            if Self::path_is_open(sh, cx, o) {
                true
            } else {
                let _ = vfs.unlink(o);
                false
            }
        });
    }

    /// [`Self::reap_orphans`], taking `vfs` itself (the caller holds only
    /// `sh`, which sorts first) — and only when there is anything to reap.
    pub(super) fn reap_orphans_locked(&self, sh: &Shared, cx: &ServiceCtx) {
        if self.orphans.lock().unwrap().is_empty() {
            return;
        }
        let mut vfs = self.vfs.lock().unwrap();
        self.reap_orphans(sh, cx, &mut vfs);
    }

    /// `unlinkat`/`unlink`/`rmdir` with open-file semantics: a file some
    /// descriptor still has open becomes an orphan instead of disappearing.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_unlink_keep_open(
        &self,
        sh: &mut Shared,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        dirfd: i64,
        pathptr: u64,
        flags: u64,
        mem: &GuestMemory,
    ) -> i64 {
        const AT_REMOVEDIR: u64 = 0x200;
        if flags & AT_REMOVEDIR != 0 {
            return self.sys_unlinkat(vfs, cx, dirfd, pathptr, flags, mem);
        }
        let Some(rel) = read_path(mem, pathptr) else {
            return err(Errno::EFAULT);
        };
        let abs = self.resolve_path(cx, dirfd, &rel);
        let is_file = vfs.stat(&abs).is_some_and(|a| a.kind != NodeKind::Dir);
        if is_file && Self::path_is_open(sh, cx, &abs) {
            return match self.orphan_move(vfs, &abs) {
                Ok(hidden) => {
                    Self::repoint(sh, cx, &abs, &hidden);
                    self.orphans.lock().unwrap().insert(hidden);
                    0
                }
                Err(e) => io_errno(&e),
            };
        }
        self.sys_unlinkat(vfs, cx, dirfd, pathptr, flags, mem)
    }

    /// `rename`/`renameat`/`renameat2` with open-file semantics: descriptors
    /// follow the renamed node, and an open file the rename replaces lives on
    /// as an orphan.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_rename_keep_open(
        &self,
        sh: &mut Shared,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        olddirfd: i64,
        oldptr: u64,
        newdirfd: i64,
        newptr: u64,
        flags: u64,
        mem: &GuestMemory,
    ) -> i64 {
        const RENAME_NOREPLACE: u64 = 1;
        const RENAME_EXCHANGE: u64 = 2;
        const RENAME_WHITEOUT: u64 = 4;
        if flags & !(RENAME_NOREPLACE | RENAME_EXCHANGE | RENAME_WHITEOUT) != 0
            || (flags & RENAME_EXCHANGE != 0 && flags & (RENAME_NOREPLACE | RENAME_WHITEOUT) != 0)
        {
            return err(Errno::EINVAL);
        }
        let (Some(old), Some(new)) = (read_path(mem, oldptr), read_path(mem, newptr)) else {
            return err(Errno::EFAULT);
        };
        let from = self.resolve_path(cx, olddirfd, &old);
        let to = self.resolve_path(cx, newdirfd, &new);
        if from == to || flags & RENAME_EXCHANGE != 0 {
            let r = self.sys_renameat(vfs, cx, olddirfd, oldptr, newdirfd, newptr, flags, mem);
            if r == 0 && flags & RENAME_EXCHANGE != 0 {
                // Swap the two names under the descriptors too.
                let tmp = format!("{to}\0xchg");
                Self::repoint(sh, cx, &from, &tmp);
                Self::repoint(sh, cx, &to, &from);
                Self::repoint(sh, cx, &tmp, &to);
            }
            return r;
        }
        // An open, non-directory target being replaced: keep it as an orphan.
        if flags & RENAME_NOREPLACE == 0
            && vfs.stat(&to).is_some_and(|a| a.kind != NodeKind::Dir)
            && vfs.stat(&from).is_some_and(|a| a.kind != NodeKind::Dir)
            && Self::path_is_open(sh, cx, &to)
            && let Ok(hidden) = self.orphan_move(vfs, &to)
        {
            Self::repoint(sh, cx, &to, &hidden);
            self.orphans.lock().unwrap().insert(hidden);
        }
        let r = self.sys_renameat(vfs, cx, olddirfd, oldptr, newdirfd, newptr, flags, mem);
        if r == 0 {
            Self::repoint(sh, cx, &from, &to);
        }
        r
    }

    /// `linkat` of an orphan (an `O_TMPFILE` or unlinked file, named through
    /// `AT_EMPTY_PATH` or `/proc/self/fd/N`): give it the new name by moving
    /// it there, so the descriptor keeps writing the now-named file. `None`
    /// when `old_abs` is not an orphan.
    pub(super) fn link_orphan(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        old_abs: &str,
        new_abs: &str,
    ) -> Option<i64> {
        let mut orphans = self.orphans.lock().unwrap();
        if !orphans.contains(old_abs) {
            return None;
        }
        if vfs.stat(new_abs).is_some() {
            return Some(err(Errno::EEXIST));
        }
        Some(match vfs.rename(old_abs, new_abs) {
            Ok(()) => {
                orphans.remove(old_abs);
                // Only the caller's own table is reachable on this (vfs-only)
                // path; the orphan was this process's to begin with.
                let open: Vec<i32> = cx.cur.fds.iter().map(|(n, _)| n).collect();
                for n in open {
                    if let Some(Fd::File { path, .. }) = cx.cur.fds.get_mut(n)
                        && path == old_abs
                    {
                        new_abs.clone_into(path);
                    }
                }
                0
            }
            Err(e) => io_errno(&e),
        })
    }

    /// `open(dir, O_TMPFILE | …, mode)`: an unnamed regular file on `dir`'s
    /// filesystem, reachable only through the returned descriptor until
    /// `linkat` names it. Must be opened for writing (`EINVAL` otherwise).
    pub(super) fn open_tmpfile(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        dir: &str,
        flags: u64,
        mode: u64,
    ) -> i64 {
        const O_ACCMODE: u64 = 0o3;
        const O_CLOEXEC: u64 = 0o2000000;
        if flags & O_ACCMODE == 0 || flags & O_ACCMODE == 3 {
            return err(Errno::EINVAL);
        }
        match vfs.stat(dir) {
            Some(a) if a.kind == NodeKind::Dir => {}
            Some(_) => return err(Errno::ENOTDIR),
            None => return err(Errno::ENOENT),
        }
        let [root, local] = self.orphan_homes(vfs, dir);
        let hidden = match vfs.create(&root, (mode & 0o7777) as u32) {
            Ok(()) => root,
            Err(_) => match vfs.create(&local, (mode & 0o7777) as u32) {
                Ok(()) => local,
                Err(e) => return io_errno(&e),
            },
        };
        self.orphans.lock().unwrap().insert(hidden.clone());
        let fd = cx.cur.fds.alloc(Fd::File {
            path: hidden,
            offset: super::FileOffset::new(0),
            readable: flags & O_ACCMODE == 2,
            writable: true,
        });
        cx.cur.fds.set_cloexec(fd, flags & O_CLOEXEC != 0);
        i64::from(fd)
    }
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{BASE, call, e, put_str, setup};
    use crate::abi::arch::Sysno;
    use crate::abi::errno::Errno;

    const AT_FDCWD: u64 = (-100i64) as u64;

    #[test]
    fn unlinked_open_file_keeps_working_and_is_reaped_on_close() {
        let (k, mut mem, mut v, mut cx) = setup();
        let (path, buf) = (BASE, BASE + 0x1000);
        put_str(&mut mem, path, "/t");
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_FDCWD, path, 0o102, 0o600, 0, 0],
        ) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Unlinkat,
                [AT_FDCWD, path, 0, 0, 0, 0]
            ),
            0
        );
        // The name is gone…
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Openat,
                [AT_FDCWD, path, 0, 0, 0, 0]
            ),
            e(Errno::ENOENT)
        );
        // …but the descriptor still works (ftruncate, write, read).
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Ftruncate,
                [fd, 4, 0, 0, 0, 0]
            ),
            0
        );
        mem.write(buf, b"data").unwrap();
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
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Pread64,
                [fd, buf + 8, 4, 0, 0, 0]
            ),
            4
        );
        assert_eq!(mem.read_vec(buf + 8, 4).unwrap(), b"data");
        // The orphan is invisible to getdents of the root.
        put_str(&mut mem, BASE + 0x100, "/");
        let d = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_FDCWD, BASE + 0x100, 0, 0, 0, 0],
        ) as u64;
        let n = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Getdents64,
            [d, buf, 4096, 0, 0, 0],
        ) as usize;
        let listing = mem.read_vec(buf, n).unwrap();
        assert!(!listing.windows(6).any(|w| w == b"nixvm-"), "orphan hidden");
        assert_eq!(k.orphans.lock().unwrap().len(), 1);
        // Closing the last descriptor deletes it.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Close,
                [fd, 0, 0, 0, 0, 0]
            ),
            0
        );
        assert!(k.orphans.lock().unwrap().is_empty());
    }

    #[test]
    fn renames_carry_descriptors_and_tmpfile_links() {
        let (k, mut mem, mut v, mut cx) = setup();
        let (a, b, buf) = (BASE, BASE + 0x100, BASE + 0x1000);
        put_str(&mut mem, a, "/a");
        put_str(&mut mem, b, "/b");
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_FDCWD, a, 0o102, 0o600, 0, 0],
        ) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Renameat,
                [AT_FDCWD, a, AT_FDCWD, b, 0, 0]
            ),
            0
        );
        mem.write(buf, b"xy").unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [fd, buf, 2, 0, 0, 0]
            ),
            2
        );
        // O_TMPFILE (0o20200000 with O_RDWR) in /, then link it as /c.
        put_str(&mut mem, a, "/");
        let t = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_FDCWD, a, 0o20_200_002, 0o600, 0, 0],
        );
        assert!(t >= 0, "{t}");
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [t as u64, buf, 2, 0, 0, 0]
            ),
            2
        );
        put_str(&mut mem, a, "");
        put_str(&mut mem, b, "/c");
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Linkat,
                [t as u64, a, AT_FDCWD, b, 0x1000, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [t as u64, buf, 2, 0, 0, 0]
            ),
            2
        );
        let c = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_FDCWD, b, 0, 0, 0, 0],
        ) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Read,
                [c, buf + 16, 8, 0, 0, 0]
            ),
            4
        );
        // O_TMPFILE read-only is EINVAL.
        put_str(&mut mem, a, "/");
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Openat,
                [AT_FDCWD, a, 0o20_200_000, 0o600, 0, 0]
            ),
            e(Errno::EINVAL)
        );
    }

    #[test]
    fn orphans_inside_a_writable_subtree_stay_in_it() {
        // /dev/shm is a writable tmpfs inside the read-only /dev mount: the
        // orphan can't go to /dev's root, so it stays in /dev/shm.
        let (k, mut mem, mut v, mut cx) = setup();
        k.vfs
            .lock()
            .unwrap()
            .mount("/dev", Box::new(crate::fs::DevFs::new()));
        let path = BASE;
        put_str(&mut mem, path, "/dev/shm/pym-1");
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_FDCWD, path, 0o102, 0o600, 0, 0],
        ) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Unlinkat,
                [AT_FDCWD, path, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Ftruncate,
                [fd, 4096, 0, 0, 0, 0]
            ),
            0
        );
        let o = k.orphans.lock().unwrap().iter().next().cloned().unwrap();
        assert!(o.starts_with("/dev/shm/.nixvm-orphan-"), "{o}");
    }

    #[test]
    fn dup_and_fork_copies_share_the_file_offset() {
        let (k, mut mem, mut v, mut cx) = setup();
        let (path, buf) = (BASE, BASE + 0x1000);
        put_str(&mut mem, path, "/o");
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_FDCWD, path, 0o102, 0o600, 0, 0],
        ) as u64;
        let d = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Dup,
            [fd, 0, 0, 0, 0, 0],
        ) as u64;
        mem.write(buf, b"abc").unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [fd, buf, 3, 0, 0, 0]
            ),
            3
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [d, buf, 3, 0, 0, 0]
            ),
            3
        );
        // Both wrote at the shared position: 6 bytes, and both see it.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Lseek,
                [fd, 0, 1, 0, 0, 0]
            ),
            6
        );
        // A forked table's copy shares it too (the table is cloned).
        let child = cx.cur.fds.clone();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Lseek,
                [d, 1, 0, 0, 0, 0]
            ),
            1
        );
        let Some(super::Fd::File { offset, .. }) = child.get(fd as i32) else {
            panic!("not a file");
        };
        assert_eq!(offset.get(), 1);
        // kcmp(KCMP_FILE) sees the dup as the same file, a fresh open as not.
        let other = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_FDCWD, path, 2, 0, 0, 0],
        ) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Kcmp,
                [1, 1, 0, fd, d, 0]
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
                [1, 1, 0, fd, other, 0]
            ),
            0
        );
    }
}
