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

use super::fd::FdTable;
use super::{AT_FDCWD, Fd, Kernel, ServiceCtx, Shared, err, io_errno, read_path, stat};

/// `unlinkat` flag: remove a directory, like `rmdir(2)`.
const AT_REMOVEDIR: u64 = 0x200;

impl Kernel {
    /// `mount(source, target, fstype, flags, data)`. The in-memory pseudo
    /// filesystems really mount (`tmpfs`/`ramfs`, `proc`, `sysfs`,
    /// `devtmpfs`), stacking over the target directory until `umount`; the
    /// remount, bind and move variants work, as does propagation (shared, slave,
    /// private and unbindable mounts); the kernel's other virtual filesystems (`devpts`, `mqueue`, `cgroup2`, `debugfs`, `binfmt_misc`,
    /// …, which nixvm already provides or doesn't model) are accepted as
    /// no-ops; a filesystem that needs a block device (`ext4`, `vfat`, …) is
    /// `ENODEV` — there are none. Root only (`EPERM`); the target must be an
    /// existing directory.
    pub(super) fn sys_mount(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        a: &[u64; 6],
        mem: &GuestMemory,
    ) -> i64 {
        const MS_REMOUNT: u64 = 32;
        const MS_BIND: u64 = 4096;
        const MS_MOVE: u64 = 8192;
        const MS_PROPAGATION: u64 = (1 << 17) | (1 << 18) | (1 << 19) | (1 << 20);
        let (target, fstype, flags) = (a[1], a[2], a[3]);
        if !self.may_administer(cx, cx.cur.ns.mnt) {
            return err(Errno::EPERM);
        }
        let Some(rel) = read_path(mem, target) else {
            return err(Errno::EFAULT);
        };
        let abs = match self.follow_or_eloop(vfs, cx, &self.resolve_path(cx, AT_FDCWD, &rel)) {
            Ok(p) => p,
            Err(e) => return e,
        };
        if self.trace {
            eprintln!(
                "[mount] pid={} target {rel:?} -> {abs:?} flags={flags:#x}",
                cx.cur.pid
            );
        }
        match vfs.stat(&abs) {
            Some(st) if st.kind == NodeKind::Dir => {}
            Some(_) if flags & MS_BIND != 0 => {}
            Some(_) => return err(Errno::ENOTDIR),
            None => return err(Errno::ENOENT),
        }
        const MS_RDONLY: u64 = 1;
        const MS_REC: u64 = 0x4000;
        if flags & MS_REMOUNT != 0 {
            // A remount changes how the existing mount behaves: here, read-only.
            return match vfs.set_read_only(&abs, flags & MS_RDONLY != 0) {
                Ok(()) => 0,
                Err(e) => io_errno(&e),
            };
        }
        if flags & (MS_BIND | MS_MOVE) != 0 {
            let Some(source) = read_path(mem, a[0]) else {
                return err(Errno::EFAULT);
            };
            let source =
                match self.follow_or_eloop(vfs, cx, &self.resolve_path(cx, AT_FDCWD, &source)) {
                    Ok(p) => p,
                    Err(e) => return e,
                };
            if vfs.stat(&source).is_none() {
                return err(Errno::ENOENT);
            }
            let moved = if flags & MS_BIND != 0 {
                vfs.bind(&source, &abs, flags & MS_REC != 0)
            } else {
                vfs.move_mount(&source, &abs)
            };
            return match moved {
                Ok(()) => 0,
                Err(e) => io_errno(&e),
            };
        }
        // Propagation: exactly one type, on a mount point, and with MS_REC on
        // everything below it too.
        if flags & MS_PROPAGATION != 0 {
            use crate::fs::Propagation;
            let kind = match flags & MS_PROPAGATION {
                f if f == 1 << 20 => Propagation::Shared,
                f if f == 1 << 19 => Propagation::Slave,
                f if f == 1 << 18 => Propagation::Private,
                f if f == 1 << 17 => Propagation::Unbindable,
                _ => return err(Errno::EINVAL),
            };
            return match vfs.set_propagation(&abs, kind, flags & MS_REC != 0) {
                Ok(()) => 0,
                Err(e) => io_errno(&e),
            };
        }
        let fstype = if fstype == 0 {
            String::new()
        } else {
            match mem.read_cstr(fstype, 64) {
                Ok(b) => String::from_utf8_lossy(&b).into_owned(),
                Err(_) => return err(Errno::EFAULT),
            }
        };
        let fs: Box<dyn crate::fs::MountFs> = match fstype.as_str() {
            "tmpfs" | "ramfs" => Box::new(crate::fs::TmpFs::new()),
            "proc" => Box::new(crate::fs::ProcFs::new(self.ncpus)),
            "sysfs" => Box::new(crate::fs::SysFs::new(self.ncpus)),
            "devtmpfs" => Box::new(crate::fs::DevFs::new()),
            "cgroup2" => {
                self.cgroup_mounted
                    .store(true, std::sync::atomic::Ordering::Relaxed);
                Box::new(crate::fs::CgroupFs::new(
                    self.cgroups.clone(),
                    self.cgroup_mount_root(cx),
                ))
            }
            "devpts" | "mqueue" | "securityfs" | "debugfs" | "tracefs" | "bpf" | "fusectl"
            | "configfs" | "pstore" | "efivarfs" | "binfmt_misc" | "hugetlbfs" | "autofs"
            | "rpc_pipefs" | "nsfs" => return 0,
            // cgroup v1 isn't offered either: there's one v2 hierarchy, as on a
            // system booted with cgroup_no_v1=all.
            _ => return err(Errno::ENODEV),
        };
        vfs.mount_guest(&abs, fs);
        0
    }

    /// `chroot(path)`: make `path` this task's root. Later paths resolve below it,
    /// including absolute symlinks, and `..` stops there. Root only.
    pub(super) fn sys_chroot(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        path: u64,
        mem: &GuestMemory,
    ) -> i64 {
        if !cx.cur.caps.has(super::caps::CAP_SYS_CHROOT) {
            return err(Errno::EPERM);
        }
        let Some(rel) = read_path(mem, path) else {
            return err(Errno::EFAULT);
        };
        let abs = match self.follow_or_eloop(vfs, cx, &self.resolve_path(cx, AT_FDCWD, &rel)) {
            Ok(p) => p,
            Err(e) => return e,
        };
        match vfs.stat(&abs) {
            Some(st) if st.kind == NodeKind::Dir => {
                cx.cur.root = abs;
                0
            }
            Some(_) => err(Errno::ENOTDIR),
            None => err(Errno::ENOENT),
        }
    }

    /// `pivot_root(new_root, put_old)`: swap the namespace's root. Paths the kernel
    /// holds on the namespace's tasks' behalf (working directories, open files) are
    /// names, not references, so they're rewritten to match.
    pub(super) fn sys_pivot_root(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        new_root: u64,
        put_old: u64,
        mem: &GuestMemory,
    ) -> i64 {
        if !self.may_administer(cx, cx.cur.ns.mnt) {
            return err(Errno::EPERM);
        }
        // From inside a chroot this isn't allowed, as on Linux.
        if cx.cur.root != "/" || cx.cur.ns.mnt == 0 {
            return err(Errno::EINVAL);
        }
        let (Some(new_root), Some(put_old)) = (read_path(mem, new_root), read_path(mem, put_old))
        else {
            return err(Errno::EFAULT);
        };
        let mut vfs = self.vfs_for(cx);
        let follow = |this: &Self, vfs: &mut MountTable, p: &str| {
            this.follow_or_eloop(vfs, cx, &this.resolve_path(cx, AT_FDCWD, p))
        };
        let new_root = match follow(self, &mut vfs, &new_root) {
            Ok(p) => p,
            Err(e) => return e,
        };
        let put_old = match follow(self, &mut vfs, &put_old) {
            Ok(p) => p,
            Err(e) => return e,
        };
        for dir in [&new_root, &put_old] {
            match vfs.stat(dir) {
                Some(st) if st.kind == NodeKind::Dir => {}
                Some(_) => return err(Errno::ENOTDIR),
                None => return err(Errno::ENOENT),
            }
        }
        let map = match vfs.pivot_root(&new_root, &put_old) {
            Ok(map) => map,
            Err(e) => return io_errno(&e),
        };
        drop(vfs);
        let ns = cx.cur.ns.mnt;
        // A working directory at the old root moves to the new one rather than with
        // the old root, which is what makes `pivot_root(".", ".")` usable.
        let moved = |path: &str| {
            if path == "/" {
                "/".to_string()
            } else {
                map.translate(path)
            }
        };
        cx.cur.cwd = moved(&cx.cur.cwd);
        cx.cur.exe = map.translate(&cx.cur.exe);
        cx.cur.fds.rewrite_paths(&|p| map.translate(p));
        let (mut files, mut cwds) = (Vec::new(), Vec::new());
        for p in sh.procs.iter_mut().flatten() {
            if p.info.ns.mnt == ns {
                p.info.exe = map.translate(&p.info.exe);
                files.push(p.info.files);
                cwds.push(p.info.fs);
            }
        }
        for f in files {
            if let Some(Some(table)) = sh.file_tables.get_mut(f) {
                table.rewrite_paths(&|p| map.translate(p));
            }
        }
        for c in cwds {
            if let Some(Some(cwd)) = sh.cwd_tables.get_mut(c) {
                *cwd = moved(cwd);
            }
        }
        0
    }

    /// `umount2(target, flags)`: remove a mount. The VM's own in its first namespace
    /// are busy (`EBUSY`), and a non-mount-point `EINVAL`. With `MNT_DETACH` what's
    /// below it goes too, and files open there keep working.
    pub(super) fn sys_umount2(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        target: u64,
        flags: u64,
        mem: &GuestMemory,
    ) -> i64 {
        const MNT_DETACH: u64 = 2;
        // MNT_FORCE | MNT_DETACH | MNT_EXPIRE | UMOUNT_NOFOLLOW.
        if flags & !0xf != 0 || (flags & 4 != 0 && flags & 3 != 0) {
            return err(Errno::EINVAL);
        }
        if !self.may_administer(cx, cx.cur.ns.mnt) {
            return err(Errno::EPERM);
        }
        let Some(rel) = read_path(mem, target) else {
            return err(Errno::EFAULT);
        };
        let abs = self.resolve_path(cx, AT_FDCWD, &rel);
        let mut vfs = self.vfs_for(cx);
        let moved = match vfs.unmount(&abs, flags & MNT_DETACH != 0) {
            Ok(Some(moved)) => moved,
            Ok(None) => return 0,
            Err(e) => return io_errno(&e),
        };
        let ns = cx.cur.ns.mnt;
        cx.cur.cwd = moved.translate(&cx.cur.cwd);
        cx.cur.fds.rewrite_paths(&|p| moved.translate(p));
        let (mut files, mut cwds) = (Vec::new(), Vec::new());
        for p in sh.procs.iter().flatten() {
            if p.info.ns.mnt == ns {
                files.push(p.info.files);
                cwds.push(p.info.fs);
            }
        }
        for f in files {
            if let Some(Some(table)) = sh.file_tables.get_mut(f) {
                table.rewrite_paths(&|p| moved.translate(p));
            }
        }
        for c in cwds {
            if let Some(Some(cwd)) = sh.cwd_tables.get_mut(c) {
                *cwd = moved.translate(cwd);
            }
        }
        Self::collect_unused_hidden_mounts(sh, cx, &mut vfs);
        0
    }

    /// Drop the hidden mounts (detached, or carried across `setns`) that no open
    /// file, mapping or working directory refers to any more.
    pub(super) fn collect_unused_hidden_mounts(sh: &Shared, cx: &ServiceCtx, vfs: &mut MountTable) {
        let tables = sh.file_tables.iter().flatten().chain([&cx.cur.fds]);
        let open = tables
            .flat_map(FdTable::iter)
            .filter_map(|(_, fd)| match fd {
                Fd::File { path, .. } | Fd::Dir { path, .. } => Some(path.as_str()),
                _ => None,
            });
        let dirs = sh.cwd_tables.iter().flatten().map(String::as_str);
        let maps = sh
            .procs
            .iter()
            .flatten()
            .flat_map(|p| p.info.shared_maps.iter())
            .chain(cx.cur.shared_maps.iter())
            .map(|m| m.path.as_str());
        let in_use: Vec<&str> = open
            .chain(dirs)
            .chain(maps)
            .chain([cx.cur.cwd.as_str()])
            .collect();
        vfs.collect_detached(&in_use);
    }

    /// The task just joined mount namespace `cx.cur.ns.mnt` from `old`: keep its
    /// open files and shared file mappings on the files they were opened on,
    /// which the same paths may not name here.
    pub(super) fn carry_open_files(&self, sh: &Shared, cx: &mut ServiceCtx, old: u32) {
        if old == cx.cur.ns.mnt {
            return;
        }
        let mut paths: Vec<String> = cx
            .cur
            .fds
            .iter()
            .filter_map(|(_, fd)| match fd {
                Fd::File { path, .. } | Fd::Dir { path, .. } => Some(path.clone()),
                _ => None,
            })
            .chain(cx.cur.shared_maps.iter().map(|m| m.path.clone()))
            .collect();
        paths.sort();
        paths.dedup();
        let mut vfs = self.vfs_for(cx);
        let carried = vfs.carry_from(old, &paths);
        let moved: std::collections::BTreeMap<String, String> = paths
            .into_iter()
            .zip(carried)
            .filter_map(|(from, to)| Some((from, to?)))
            .collect();
        let translate = |p: &str| moved.get(p).cloned().unwrap_or_else(|| p.to_string());
        cx.cur.fds.rewrite_paths(&translate);
        for m in &mut cx.cur.shared_maps {
            m.path = translate(&m.path);
        }
        Self::collect_unused_hidden_mounts(sh, cx, &mut vfs);
    }

    /// `statfs(path, buf)` — write a plausible `struct statfs` for the
    /// filesystem containing `path`.
    pub(super) fn sys_statfs(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        pathptr: u64,
        buf: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some(rel) = read_path(mem, pathptr) else {
            return err(Errno::EFAULT);
        };
        let abs = self.resolve_path(cx, AT_FDCWD, &rel);
        let abs = match self.follow_or_eloop(vfs, cx, &abs) {
            Ok(p) => p,
            Err(e) => return e,
        };
        if vfs.stat(&abs).is_none() {
            return err(Errno::ENOENT);
        }
        let magic = stat::fs_magic(vfs.fs_type_at(&abs).unwrap_or("tmpfs"));
        write_statfs_or_fault(mem, buf, magic)
    }

    /// `fstatfs(fd, buf)` — as `statfs`, keyed by an open fd.
    pub(super) fn sys_fstatfs(
        &self,
        cx: &mut ServiceCtx,
        fd: u64,
        buf: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let path = match cx.cur.fds.get(fd as i32) {
            None => return err(Errno::EBADF),
            Some(super::Fd::File { path, .. }) => Some(path.clone()),
            Some(_) => None,
        };
        let fs_type = path.and_then(|p| self.vfs_for(cx).fs_type_at(&p));
        write_statfs_or_fault(mem, buf, stat::fs_magic(fs_type.unwrap_or("tmpfs")))
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
        let me = self.visible_pid(cx, cx.cur.tgid).to_string();
        if who != "self" && who != "thread-self" && who != me {
            return None;
        }
        // `/proc/self/{exe,cwd,root}` resolve to *live* per-task state, not
        // procfs's static snapshot — programs read these to locate themselves.
        match tail {
            "exe" => {
                return (!cx.cur.exe.is_empty()).then(|| Self::as_seen_from_root(cx, &cx.cur.exe));
            }
            "cwd" => return Some(Self::as_seen_from_root(cx, &cx.cur.cwd)),
            "root" => return Some("/".to_string()),
            _ => {}
        }
        let n: i32 = tail.strip_prefix("fd/")?.parse().ok()?;
        Some(Self::as_seen_from_root(
            cx,
            &fd_link_target(cx.cur.fds.get(n)?),
        ))
    }

    /// Everything visible from this task, with the numbers it knows them by. Threads
    /// don't get an entry of their own at the top level, on Linux or here.
    pub(super) fn visible_tasks(&self, cx: &ServiceCtx) -> Vec<crate::fs::ProcTask> {
        use super::RunState;
        let sh = self.shared.lock().unwrap();
        let describe = |info: &super::ProcInfo| {
            let (owner, status_ids, id_files) = {
                let users = self.users.lock().unwrap();
                let (owner, ids) = Self::task_ids(&users, cx.cur.ns.user, info);
                let (u, g, s) = users.identity_files(info.ns.user);
                (owner, ids, [u, g, s])
            };
            let pid = self.visible_pid(cx, info.tgid);
            (pid != 0).then(|| crate::fs::ProcTask {
                pid: pid as u32,
                ppid: self.visible_pid(cx, info.ppid) as u32,
                tracer: self.visible_pid(cx, info.tracer()) as u32,
                comm: info.comm.clone(),
                cmdline: info.cmdline.clone(),
                state: match info.run {
                    // A killable wait (TASK_KILLABLE) reads as D, as Linux shows it.
                    RunState::Running if info.parked && info.seccomp.killable_wait() => 'D',
                    RunState::Running if info.parked => 'S',
                    RunState::Running => 'R',
                    RunState::Stopped(_) => 'T',
                    RunState::Traced => 't',
                    RunState::Zombie(_) => 'Z',
                },
                uid: owner.0,
                gid: owner.1,
                status_ids,
                id_files,
                cgroup: self.cgroup_line(cx, info.pid),
                sandbox: sandbox_of(info),
                syscall: if info.parked || info.run != RunState::Running {
                    syscall_line(info)
                } else {
                    "running\n".to_string()
                },
            })
        };
        let others = sh.procs.iter().flatten().map(|p| &p.info);
        others
            .chain([&cx.cur])
            .filter(|info| !info.is_thread)
            .filter_map(describe)
            .collect()
    }

    /// The threads of this task's process, as it numbers them.
    pub(super) fn visible_thread_ids(&self, cx: &ServiceCtx) -> Vec<u32> {
        use super::RunState;
        let sh = self.shared.lock().unwrap();
        let mut tids: Vec<u32> = sh
            .procs
            .iter()
            .flatten()
            .map(|p| &p.info)
            .filter(|i| i.tgid == cx.cur.tgid && !matches!(i.run, RunState::Zombie(_)))
            .chain([&cx.cur])
            .map(|i| self.visible_pid(cx, i.pid) as u32)
            .filter(|&t| t != 0)
            .collect();
        tids.sort_unstable();
        tids.dedup();
        tids
    }

    /// Build the live `/proc/self` view (comm, cmdline, exe, cwd, pid/ppid, open
    /// fds) for the running task, so procfs renders the real running program
    /// instead of the boot-time placeholder. Called just before a `/proc` read.
    #[allow(clippy::unused_self)]
    pub(super) fn proc_self_live(&self, cx: &ServiceCtx) -> crate::fs::ProcSelf {
        let fds = cx
            .cur
            .fds
            .iter()
            .map(|(n, fd)| (n as u32, Self::as_seen_from_root(cx, &fd_link_target(fd))))
            .collect();
        let (uid_map, gid_map, setgroups) = self.proc_identity_files(cx);
        crate::fs::ProcSelf {
            tasks: Vec::new(),
            uid_map,
            gid_map,
            setgroups,
            timens_offsets: super::userns::render_time_offsets(self.time_offsets_for(cx)),
            limits: render_limits(&cx.cur.rlimits),
            hostname: String::new(),
            domainname: String::new(),
            cgroup: self.cgroup_line(cx, cx.cur.pid),
            mounts: Vec::new(),
            comm: cx.cur.comm.clone(),
            cmdline: cx.cur.cmdline.clone(),
            exe: Self::as_seen_from_root(cx, &cx.cur.exe),
            cwd: Self::as_seen_from_root(cx, &cx.cur.cwd),
            pid: self.visible_pid(cx, cx.cur.tgid) as u32,
            ppid: self.visible_pid(cx, cx.cur.ppid) as u32,
            owner: {
                let users = self.users.lock().unwrap();
                Self::task_ids(&users, cx.cur.ns.user, &cx.cur).0
            },
            tid: self.visible_pid(cx, cx.cur.pid) as u32,
            tids: Vec::new(),
            status_ids: self.status_id_lines(cx),
            status_security: format!(
                "{}{}",
                cx.cur.caps.status_lines(),
                sandbox_of(&cx.cur).lines()
            ),
            tracer: self.visible_pid(cx, cx.cur.tracer()) as u32,
            fds,
            auxv: cx.cur.auxv.clone(),
            arch: Some(self.arch),
            syscall: syscall_line(&cx.cur),
            seccomp_logged: Some(super::seccomp::logged_names(
                self.seccomp_logged
                    .load(std::sync::atomic::Ordering::Relaxed),
            )),
        }
    }

    /// A task's effective ids as the VM sees them, which own its `/proc/<pid>`,
    /// and its `status` id lines as user namespace `reader` sees them.
    fn task_ids(
        users: &super::userns::UserNamespaces,
        reader: u32,
        info: &super::ProcInfo,
    ) -> ((u32, u32), String) {
        use super::userns::IdKind;
        let ns = info.ns.user;
        let global = |kind, id| users.kernel_id(ns, kind, id);
        let seen = |kind, id| users.seen_from_global(reader, kind, global(kind, id));
        let c = info.creds;
        let line = |kind, ids: [u32; 4]| ids.map(|id| seen(kind, id).to_string()).join("\t");
        let ids = format!(
            "Uid:\t{}\nGid:\t{}\n",
            line(IdKind::Uid, [c.ruid, c.euid, c.suid, c.fsuid]),
            line(IdKind::Gid, [c.rgid, c.egid, c.sgid, c.fsgid])
        );
        (
            (global(IdKind::Uid, c.euid), global(IdKind::Gid, c.egid)),
            ids,
        )
    }

    /// `status`'s `Uid:`, `Gid:` and `Groups:` lines, in the reader's user
    /// namespace.
    fn status_id_lines(&self, cx: &ServiceCtx) -> String {
        use std::fmt::Write as _;
        let c = cx.cur.creds;
        let mut out = format!(
            "Uid:\t{}\t{}\t{}\t{}\nGid:\t{}\t{}\t{}\t{}\nGroups:\t",
            c.ruid, c.euid, c.suid, c.fsuid, c.rgid, c.egid, c.sgid, c.fsgid
        );
        for g in self.visible_groups(cx) {
            let _ = write!(out, "{g} ");
        }
        out.push('\n');
        out
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_readlinkat(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
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
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
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
            Ok(()) => {
                self.own_new_node(vfs, cx, &abs);
                0
            }
            Err(e) => io_errno(&e),
        }
    }

    /// Make the node just created at `abs` its creator's (Linux's
    /// `inode_init_owner`): owned by its fsuid, and its fsgid or, in a
    /// set-group-ID directory, the directory's group. Seen from a user
    /// namespace these are the ids it maps; a backend makes everything root's.
    pub(super) fn own_new_node(&self, vfs: &mut MountTable, cx: &ServiceCtx, abs: &str) {
        use super::userns::IdKind;
        const S_ISGID: u32 = 0o2000;
        let (uid, gid) = {
            let users = self.users.lock().unwrap();
            let ns = cx.cur.ns.user;
            (
                users.kernel_id(ns, IdKind::Uid, cx.cur.creds.fsuid),
                users.kernel_id(ns, IdKind::Gid, cx.cur.creds.fsgid),
            )
        };
        let parent = match abs.rsplit_once('/') {
            Some(("", _)) | None => "/",
            Some((dir, _)) => dir,
        };
        let gid = vfs
            .stat(parent)
            .filter(|a| a.mode & S_ISGID != 0)
            .map_or(gid, |a| a.gid);
        if uid != 0 || gid != 0 {
            let _ = vfs.set_owner(abs, Some(uid), Some(gid));
        }
    }

    /// `mkdirat(dirfd, path, mode)`.
    pub(super) fn sys_mkdirat(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
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
            Ok(()) => {
                self.own_new_node(vfs, cx, &abs);
                // A new cgroup belongs to whoever made it, so a delegate can
                // manage what it creates.
                if self
                    .cgroup_mounted
                    .load(std::sync::atomic::Ordering::Relaxed)
                    && vfs.cgroup_at(&abs).is_some()
                {
                    let _ = vfs.set_owner(&abs, Some(cx.cur.creds.euid), Some(cx.cur.creds.egid));
                }
                0
            }
            Err(e) => io_errno(&e),
        }
    }

    /// `unlinkat(dirfd, path, flags)` — `rmdir` when `AT_REMOVEDIR` is set,
    /// otherwise `unlink`.
    pub(super) fn sys_unlinkat(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
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
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        dirfd: i64,
        pathptr: u64,
        times: u64,
        flags: u64,
        mem: &GuestMemory,
    ) -> i64 {
        // Decode the timespec at `times + off` into a `SetTime`.
        let read_field = |off: u64| -> Option<crate::fs::SetTime> {
            const UTIME_NOW: u64 = 0x3fff_ffff;
            const UTIME_OMIT: u64 = 0x3fff_fffe;
            let (sec, nsec) = (
                mem.read_u64(times + off).ok()?,
                mem.read_u64(times + off + 8).ok()?,
            );
            Some(match nsec {
                UTIME_OMIT => crate::fs::SetTime::Omit,
                UTIME_NOW => crate::fs::SetTime::Now,
                _ => crate::fs::SetTime::Set {
                    sec: sec as i64,
                    nsec: nsec as i64,
                },
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
        self.set_times_at(vfs, cx, dirfd, pathptr, atime, mtime, flags, mem)
    }

    /// The legacy x86-64 timestamp setters, lowered onto [`Self::set_times_at`]:
    /// `utime(path, struct utimbuf *)` (`utimbuf` = `{ actime, modtime }` in
    /// whole seconds; `utimbuf == true`) and `utimes`/`futimesat(dirfd, path,
    /// struct timeval[2])`. A NULL times pointer means "now" for both, and a
    /// `tv_usec` outside `0..1_000_000` is `EINVAL`, as Linux checks. Both
    /// follow symlinks (there is no `AT_SYMLINK_NOFOLLOW` spelling).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_utimes_legacy(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        dirfd: i64,
        pathptr: u64,
        times: u64,
        utimbuf: bool,
        mem: &GuestMemory,
    ) -> i64 {
        use crate::fs::SetTime;
        let (atime, mtime) = if times == 0 {
            (SetTime::Now, SetTime::Now)
        } else if utimbuf {
            let (Ok(a), Ok(m)) = (mem.read_u64(times), mem.read_u64(times + 8)) else {
                return err(Errno::EFAULT);
            };
            (
                SetTime::Set {
                    sec: a as i64,
                    nsec: 0,
                },
                SetTime::Set {
                    sec: m as i64,
                    nsec: 0,
                },
            )
        } else {
            let mut tv = [0i64; 4];
            for (i, slot) in tv.iter_mut().enumerate() {
                let Ok(v) = mem.read_u64(times + 8 * i as u64) else {
                    return err(Errno::EFAULT);
                };
                *slot = v as i64;
            }
            if !(0..1_000_000).contains(&tv[1]) || !(0..1_000_000).contains(&tv[3]) {
                return err(Errno::EINVAL);
            }
            (
                SetTime::Set {
                    sec: tv[0],
                    nsec: tv[1] * 1000,
                },
                SetTime::Set {
                    sec: tv[2],
                    nsec: tv[3] * 1000,
                },
            )
        };
        if pathptr == 0 {
            return err(Errno::EFAULT); // only utimensat accepts a NULL path
        }
        self.set_times_at(vfs, cx, dirfd, pathptr, atime, mtime, 0, mem)
    }

    /// Resolve `(dirfd, pathptr, flags)` like `utimensat` does and store the
    /// decoded timestamps. A NULL `pathptr` targets `dirfd` itself (`futimens`).
    #[allow(clippy::too_many_arguments)]
    fn set_times_at(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        dirfd: i64,
        pathptr: u64,
        atime: crate::fs::SetTime,
        mtime: crate::fs::SetTime,
        flags: u64,
        mem: &GuestMemory,
    ) -> i64 {
        const AT_SYMLINK_NOFOLLOW: u64 = 0x100;
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
                match self.follow_or_eloop(vfs, cx, &abs) {
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
    pub(super) fn sys_fchmodat(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        dirfd: i64,
        pathptr: u64,
        mode: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let Some(rel) = read_path(mem, pathptr) else {
            return err(Errno::EFAULT);
        };
        let abs = self.resolve_path(cx, dirfd, &rel);
        let abs = match self.follow_or_eloop(vfs, cx, &abs) {
            Ok(p) => p,
            Err(e) => return e,
        };
        match vfs.set_mode(&abs, mode as u32) {
            Ok(()) => 0,
            Err(e) => io_errno(&e),
        }
    }

    /// `fchmodat2(dirfd, path, mode, flags)` — `fchmodat` with the flags the
    /// original syscall ignored. `AT_SYMLINK_NOFOLLOW` acts on the final
    /// component itself: a symlink's own mode can't be changed on Linux, so that
    /// is `EOPNOTSUPP`, while a non-link is chmod'ed normally. `AT_EMPTY_PATH`
    /// with an empty path targets `dirfd` (like `fchmod`). Any other flag bit is
    /// `EINVAL`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_fchmodat2(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        dirfd: i64,
        pathptr: u64,
        mode: u64,
        flags: u64,
        mem: &GuestMemory,
    ) -> i64 {
        const AT_SYMLINK_NOFOLLOW: u64 = 0x100;
        const AT_EMPTY_PATH: u64 = 0x1000;
        if flags & !(AT_SYMLINK_NOFOLLOW | AT_EMPTY_PATH) != 0 {
            return err(Errno::EINVAL);
        }
        let Some(rel) = read_path(mem, pathptr) else {
            return err(Errno::EFAULT);
        };
        if rel.is_empty() {
            return if flags & AT_EMPTY_PATH != 0 {
                self.sys_fchmod(vfs, cx, dirfd as u64, mode)
            } else {
                err(Errno::ENOENT)
            };
        }
        if flags & AT_SYMLINK_NOFOLLOW == 0 {
            return self.sys_fchmodat(vfs, cx, dirfd, pathptr, mode, mem);
        }
        let abs = self.resolve_path(cx, dirfd, &rel);
        match vfs.stat(&abs) {
            None => err(Errno::ENOENT),
            Some(a) if a.kind == crate::fs::NodeKind::Symlink => err(Errno::EOPNOTSUPP),
            Some(_) => match vfs.set_mode(&abs, mode as u32) {
                Ok(()) => 0,
                Err(e) => io_errno(&e),
            },
        }
    }

    /// `fchmod(fd, mode)` — chmod on an open file, resolved via its path.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_fchmod(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        fd: u64,
        mode: u64,
    ) -> i64 {
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
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
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
            Ok(()) => {
                Self::chown_kills_caps(vfs, &abs, uid, gid);
                0
            }
            Err(e) => io_errno(&e),
        }
    }

    /// A file given an owner loses its capabilities (Linux's killpriv).
    fn chown_kills_caps(vfs: &mut MountTable, path: &str, uid: u64, gid: u64) {
        if decode_id(uid).is_some() || decode_id(gid).is_some() {
            let _ = vfs.removexattr(path, super::xattr::FILE_CAPS);
        }
    }

    /// `fchown(fd, uid, gid)` — chown on an open file, resolved via its path.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_fchown(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        fd: u64,
        uid: u64,
        gid: u64,
    ) -> i64 {
        let path = match cx.cur.fds.get(fd as i32) {
            Some(Fd::File { path, .. } | Fd::Dir { path, .. }) => path.clone(),
            Some(_) => return 0, // non-file fds: accept (nothing to chown)
            None => return err(Errno::EBADF),
        };
        match vfs.set_owner(&path, decode_id(uid), decode_id(gid)) {
            Ok(()) => {
                Self::chown_kills_caps(vfs, &path, uid, gid);
                0
            }
            Err(e) => io_errno(&e),
        }
    }

    /// `mknodat(dirfd, path, mode, dev)` / `mknod(path, mode, dev)` /
    /// `mkfifo` (glibc issues `mknodat` with `S_IFIFO`). Creates a regular file
    /// or FIFO; device/socket nodes are left for the backend to reject
    /// (`EPERM` on the in-memory backends).
    pub(super) fn sys_mknodat(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
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
            Ok(()) => {
                self.own_new_node(vfs, cx, &abs);
                0
            }
            Err(e) => io_errno(&e),
        }
    }

    /// `renameat(olddirfd, old, newdirfd, new)` / `renameat2(..., flags)`.
    /// `RENAME_NOREPLACE` fails with `EEXIST` if the target exists (atomic
    /// create-if-absent); `RENAME_EXCHANGE` atomically swaps the two paths.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_renameat(
        &self,
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
    pub(super) fn sys_faccessat(
        &self,
        vfs: &mut MountTable,
        cx: &mut ServiceCtx,
        dirfd: i64,
        pathptr: u64,
        mode: u64,
        mem: &GuestMemory,
    ) -> i64 {
        const X_OK: u64 = 1;
        let Some(rel) = read_path(mem, pathptr) else {
            return err(Errno::EFAULT);
        };
        let had_slash = rel.len() > 1 && rel.ends_with('/');
        let abs = self.resolve_path(cx, dirfd, &rel);
        let abs = match self.follow_or_eloop(vfs, cx, &abs) {
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
fn write_statfs_or_fault(mem: &mut GuestMemory, addr: u64, magic: u64) -> i64 {
    let buf = stat::encode_statfs(magic);
    if mem.write(addr, &buf).is_err() {
        err(Errno::EFAULT)
    } else {
        0
    }
}

/// `/proc/<pid>/syscall` for a task in a syscall: its number, arguments, and
/// stack and return addresses.
fn syscall_line(info: &super::ProcInfo) -> String {
    let (nr, a, sp, pc) = info.last_syscall;
    format!(
        "{} {:#x} {:#x} {:#x} {:#x} {:#x} {:#x} {sp:#x} {pc:#x}\n",
        nr as i64, a[0], a[1], a[2], a[3], a[4], a[5]
    )
}

/// A task's `NoNewPrivs:`/`Seccomp:`/`Seccomp_filters:` status lines.
fn sandbox_of(info: &super::ProcInfo) -> crate::fs::Sandbox {
    crate::fs::Sandbox {
        no_new_privs: info.no_new_privs,
        seccomp: info.seccomp.mode(),
        filters: info.seccomp.filter_count(),
    }
}

/// The `/proc/self/fd/<n>` symlink target for a descriptor: the path for a
/// file/dir, an `anon_inode:`/`pipe:`/`socket:` name otherwise (matching the
/// kernel's spellings).
fn fd_link_target(fd: &Fd) -> String {
    match fd {
        // A file on a hidden mount (detached, or kept from another mount
        // namespace) reads as its path within that mount, as Linux shows it.
        Fd::File { path, .. } | Fd::Dir { path, .. } => match path.strip_prefix("/\0") {
            Some(rest) => rest
                .find('/')
                .map_or_else(|| "/".to_string(), |i| rest[i..].to_string()),
            None => path.clone(),
        },
        Fd::Stdin | Fd::Stdout | Fd::Stderr => "/dev/null".to_string(),
        Fd::Tty => "/dev/tty".to_string(),
        Fd::PipeRead(i, ..) | Fd::PipeWrite(i, ..) => format!("pipe:[{i}]"),
        Fd::Socket { sock, .. } => format!("socket:[{sock}]"),
        Fd::Eventfd(_) => "anon_inode:[eventfd]".to_string(),
        Fd::Signalfd(_) => "anon_inode:[signalfd]".to_string(),
        Fd::Timerfd(_) => "anon_inode:[timerfd]".to_string(),
        Fd::Pidfd(_) => "anon_inode:[pidfd]".to_string(),
        Fd::Namespace { kind, id } => kind.link(*id),
        Fd::Epoll(_) => "anon_inode:[eventpoll]".to_string(),
        Fd::PtyMaster(_) => "/dev/ptmx".to_string(),
        Fd::PtySlave(i) => format!("/dev/pts/{i}"),
        Fd::Mqueue { q, .. } => format!("mqueue:[{q}]"),
        Fd::SeccompNotify(_) => "anon_inode:seccomp notify".to_string(),
        Fd::ProcMem { pid, .. } => format!("/proc/{pid}/mem"),
    }
}

/// `/proc/<pid>/limits`, laid out exactly as Linux prints it.
fn render_limits(limits: &super::RLimits) -> String {
    const ROWS: [(&str, &str); super::sys_misc::RLIM_NLIMITS] = [
        ("Max cpu time", "seconds"),
        ("Max file size", "bytes"),
        ("Max data size", "bytes"),
        ("Max stack size", "bytes"),
        ("Max core file size", "bytes"),
        ("Max resident set", "bytes"),
        ("Max processes", "processes"),
        ("Max open files", "files"),
        ("Max locked memory", "bytes"),
        ("Max address space", "bytes"),
        ("Max file locks", "locks"),
        ("Max pending signals", "signals"),
        ("Max msgqueue size", "bytes"),
        ("Max nice priority", ""),
        ("Max realtime priority", ""),
        ("Max realtime timeout", "us"),
    ];
    use std::fmt::Write as _;
    let show = |v: u64| {
        if v == super::sys_misc::RLIM_INFINITY {
            "unlimited".to_string()
        } else {
            v.to_string()
        }
    };
    let mut out = format!(
        "{:<25} {:<20} {:<20} {:<10}\n",
        "Limit", "Soft Limit", "Hard Limit", "Units"
    );
    for (i, (name, unit)) in ROWS.iter().enumerate() {
        let (soft, hard) = limits.get(i as u64);
        let _ = write!(out, "{name:<25} {:<20} {:<20} ", show(soft), show(hard));
        if !unit.is_empty() {
            let _ = write!(out, "{unit:<10}");
        }
        out.push('\n');
    }
    out
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
            k.sys_renameat(
                &mut k.vfs.lock().unwrap(),
                &mut cx,
                AT_FDCWD,
                pa,
                AT_FDCWD,
                pb,
                1,
                &mem
            ),
            err(Errno::EEXIST)
        );
        assert!(k.vfs.lock().unwrap().stat("/a").is_some());
        // RENAME_EXCHANGE swaps the two files' contents.
        assert_eq!(
            k.sys_renameat(
                &mut k.vfs.lock().unwrap(),
                &mut cx,
                AT_FDCWD,
                pa,
                AT_FDCWD,
                pb,
                2,
                &mem
            ),
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
        assert_eq!(
            k.sys_mkdirat(
                &mut k.vfs.lock().unwrap(),
                &mut cx,
                AT_FDCWD,
                path,
                0o755,
                &mem
            ),
            0
        );
        // A directory (0o755) is searchable (X_OK).
        assert_eq!(
            k.sys_faccessat(&mut k.vfs.lock().unwrap(), &mut cx, AT_FDCWD, path, 1, &mem),
            0
        );
        assert_eq!(
            k.vfs.lock().unwrap().stat("/d").unwrap().kind,
            NodeKind::Dir
        );
    }

    #[test]
    fn symlinkat_then_readlinkat() {
        let (k, mut mem, mut cx) = setup();
        let target = 0x1_0000;
        let link = 0x1_0100;
        let buf = 0x1_1000;
        mem.write_init(target, b"/target\0").unwrap();
        mem.write_init(link, b"/l\0").unwrap();
        assert_eq!(
            k.sys_symlinkat(
                &mut k.vfs.lock().unwrap(),
                &mut cx,
                target,
                AT_FDCWD,
                link,
                &mem
            ),
            0
        );
        assert_eq!(
            k.sys_readlinkat(
                &mut k.vfs.lock().unwrap(),
                &mut cx,
                AT_FDCWD,
                link,
                buf,
                64,
                &mut mem
            ),
            7
        );
        assert_eq!(mem.read_vec(buf, 7).unwrap(), b"/target");
    }

    #[test]
    fn statfs_writes_bsize() {
        let (k, mut mem, mut cx) = setup();
        let path = 0x1_0000;
        let buf = 0x1_1000;
        mem.write_init(path, b"/\0").unwrap();
        assert_eq!(
            k.sys_statfs(&mut k.vfs.lock().unwrap(), &mut cx, path, buf, &mut mem),
            0
        );
        assert_eq!(mem.read_u64(buf + 8).unwrap(), 4096); // f_bsize
    }

    #[test]
    fn unlinkat_removes_file() {
        let (k, mut mem, mut cx) = setup();
        k.vfs.lock().unwrap().create("/f", 0o644).unwrap();
        let path = 0x1_0000;
        mem.write_init(path, b"/f\0").unwrap();
        assert_eq!(
            k.sys_unlinkat(&mut k.vfs.lock().unwrap(), &mut cx, AT_FDCWD, path, 0, &mem),
            0
        );
        assert!(k.vfs.lock().unwrap().stat("/f").is_none());
    }

    #[test]
    fn umask_returns_previous() {
        let (k, _mem, _cx) = setup();
        assert_eq!(k.sys_umask(&mut k.shared.lock().unwrap(), 0o077), 0o022);
        assert_eq!(k.sys_umask(&mut k.shared.lock().unwrap(), 0o022), 0o077);
    }
}
