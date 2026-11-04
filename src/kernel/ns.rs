//! Namespaces: what a task sees of the VM's global resources.
//!
//! Each task holds one id per namespace kind in an [`NsProxy`], which it inherits
//! and which `clone`, `unshare` and `setns` change. Id 0 of every kind is the VM's
//! initial namespace.
//!
//! A namespace is mostly a label on the things it isolates. SysV IPC objects record
//! the namespace that made them and are invisible from any other, and hostnames
//! live in a table by id, rather than each kind needing a whole copy of its
//! subsystem and every call site having to pick the right one.

use crate::abi::errno::Errno;

use super::{Fd, Kernel, RunState, ServiceCtx, Shared, err};
use crate::vcpu::GuestMemory;

/// The `CLONE_NEW*` flags.
pub(super) const CLONE_NEWTIME: u64 = 0x80;
pub(super) const CLONE_NEWNS: u64 = 0x2_0000;
pub(super) const CLONE_NEWCGROUP: u64 = 0x200_0000;
pub(super) const CLONE_NEWUTS: u64 = 0x400_0000;
pub(super) const CLONE_NEWIPC: u64 = 0x800_0000;
pub(super) const CLONE_NEWUSER: u64 = 0x1000_0000;
pub(super) const CLONE_NEWPID: u64 = 0x2000_0000;
pub(super) const CLONE_NEWNET: u64 = 0x4000_0000;

/// One kind of namespace, as named under `/proc/<pid>/ns`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NsKind {
    Cgroup,
    Ipc,
    Mnt,
    Net,
    Pid,
    PidForChildren,
    Time,
    TimeForChildren,
    User,
    Uts,
}

impl NsKind {
    /// Every kind, in the order `/proc/<pid>/ns` lists them.
    pub(super) const ALL: [Self; 10] = [
        Self::Cgroup,
        Self::Ipc,
        Self::Mnt,
        Self::Net,
        Self::Pid,
        Self::PidForChildren,
        Self::Time,
        Self::TimeForChildren,
        Self::User,
        Self::Uts,
    ];

    /// Its file name under `/proc/<pid>/ns`.
    pub(super) fn file_name(self) -> &'static str {
        match self {
            Self::Cgroup => "cgroup",
            Self::Ipc => "ipc",
            Self::Mnt => "mnt",
            Self::Net => "net",
            Self::Pid => "pid",
            Self::PidForChildren => "pid_for_children",
            Self::Time => "time",
            Self::TimeForChildren => "time_for_children",
            Self::User => "user",
            Self::Uts => "uts",
        }
    }

    fn from_file_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.file_name() == name)
    }

    /// The name inside a namespace link's target, such as `pid` in `pid:[4026...]`.
    fn link_name(self) -> &'static str {
        match self {
            Self::PidForChildren => "pid",
            Self::TimeForChildren => "time",
            other => other.file_name(),
        }
    }

    /// The `CLONE_NEW*` flag `setns` names this kind by.
    pub(super) fn flag(self) -> u64 {
        match self {
            Self::Cgroup => CLONE_NEWCGROUP,
            Self::Ipc => CLONE_NEWIPC,
            Self::Mnt => CLONE_NEWNS,
            Self::Net => CLONE_NEWNET,
            Self::Pid | Self::PidForChildren => CLONE_NEWPID,
            Self::Time | Self::TimeForChildren => CLONE_NEWTIME,
            Self::User => CLONE_NEWUSER,
            Self::Uts => CLONE_NEWUTS,
        }
    }

    /// This kind's namespaces share an inode range, as on Linux, where comparing
    /// two links or two `st_ino`s is how a program tells if it shares a namespace.
    pub(crate) fn inode(self, id: u32) -> u64 {
        let base = match self {
            Self::Cgroup => 0xF000_0000_u64,
            Self::Ipc => 0xF100_0000,
            Self::Mnt => 0xF200_0000,
            Self::Net => 0xF300_0000,
            Self::Pid | Self::PidForChildren => 0xF400_0000,
            Self::Time | Self::TimeForChildren => 0xF500_0000,
            Self::User => 0xF600_0000,
            Self::Uts => 0xF700_0000,
        };
        base + u64::from(id)
    }

    /// The target of a `/proc/<pid>/ns` link: `uts:[4026531838]`.
    pub(crate) fn link(self, id: u32) -> String {
        format!("{}:[{}]", self.link_name(), self.inode(id))
    }
}

/// The namespaces one task is in.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct NsProxy {
    pub(super) cgroup: u32,
    pub(super) ipc: u32,
    pub(super) mnt: u32,
    pub(super) net: u32,
    pub(super) pid: u32,
    /// What this task's children will join. Not the same as [`Self::pid`] after
    /// `unshare(CLONE_NEWPID)`, which never moves the caller itself.
    pub(super) pid_for_children: u32,
    pub(super) time: u32,
    /// As [`Self::pid_for_children`], for `unshare(CLONE_NEWTIME)`.
    pub(super) time_for_children: u32,
    pub(super) user: u32,
    pub(super) uts: u32,
}

impl NsProxy {
    /// The id this task has for `kind`.
    pub(super) fn get(&self, kind: NsKind) -> u32 {
        match kind {
            NsKind::Cgroup => self.cgroup,
            NsKind::Ipc => self.ipc,
            NsKind::Mnt => self.mnt,
            NsKind::Net => self.net,
            NsKind::Pid => self.pid,
            NsKind::PidForChildren => self.pid_for_children,
            NsKind::Time => self.time,
            NsKind::TimeForChildren => self.time_for_children,
            NsKind::User => self.user,
            NsKind::Uts => self.uts,
        }
    }
}

/// A UTS namespace's names, what `uname` reports.
#[derive(Clone, Debug)]
pub(super) struct UtsNs {
    pub(super) hostname: String,
    pub(super) domainname: String,
}

impl Default for UtsNs {
    fn default() -> Self {
        Self {
            hostname: "nixvm".to_string(),
            domainname: "(none)".to_string(),
        }
    }
}

/// How many mount namespaces may exist (`user.max_mnt_namespaces`, about).
pub(super) const MAX_MOUNT_NAMESPACES: usize = 1 << 16;

/// How many UTS namespaces may exist. Linux has a limit of about this size too
/// (`user.max_uts_namespaces`); each one here is a little host memory.
const MAX_UTS_NAMESPACES: usize = 1 << 16;

impl Shared {
    fn fresh_ns_id(&mut self) -> u32 {
        self.next_ns_id += 1;
        self.next_ns_id
    }

    /// The names in UTS namespace `id`.
    pub(super) fn uts(&mut self, id: u32) -> &mut UtsNs {
        self.uts_namespaces.entry(id).or_default()
    }

    /// Put `ns` in new namespaces for each `CLONE_NEW*` in `flags`. A new PID or
    /// time namespace is only for its children when `moves_self` is false (the
    /// `unshare` rule); a new task (`clone`) is itself inside it.
    fn enter_new_namespaces(
        &mut self,
        cx: &ServiceCtx,
        ns: &mut NsProxy,
        flags: u64,
        moves_self: bool,
    ) -> Result<(), i64> {
        // First, since it owns whatever else this call creates.
        if flags & CLONE_NEWUSER != 0 {
            ns.user = self.fresh_ns_id();
        }
        let before = *ns;
        if flags & CLONE_NEWUTS != 0 {
            let in_use: Vec<u32> = self
                .procs
                .iter()
                .flatten()
                .map(|p| p.info.ns.uts)
                .chain([cx.cur.ns.uts])
                .collect();
            self.uts_namespaces
                .retain(|id, _| *id == 0 || in_use.contains(id));
            if self.uts_namespaces.len() >= MAX_UTS_NAMESPACES {
                return Err(err(Errno::ENOSPC));
            }
            let copy = self.uts(ns.uts).clone();
            ns.uts = self.fresh_ns_id();
            self.uts_namespaces.insert(ns.uts, copy);
        }
        if flags & CLONE_NEWIPC != 0 {
            ns.ipc = self.fresh_ns_id();
        }
        if flags & CLONE_NEWNET != 0 {
            ns.net = self.fresh_ns_id();
        }
        if flags & CLONE_NEWNS != 0 {
            ns.mnt = self.fresh_ns_id();
        }
        if flags & CLONE_NEWCGROUP != 0 {
            ns.cgroup = self.fresh_ns_id();
        }
        if flags & CLONE_NEWPID != 0 {
            ns.pid_for_children = self.fresh_ns_id();
            if moves_self {
                ns.pid = ns.pid_for_children;
            }
        }
        if flags & CLONE_NEWTIME != 0 {
            ns.time_for_children = self.fresh_ns_id();
            if moves_self {
                ns.time = ns.time_for_children;
            }
        }
        let mut users = self.users.lock().unwrap();
        if users.owned_count() >= super::userns::COLLECT_ABOVE {
            let tasks = self.procs.iter().flatten().map(|p| p.info.ns);
            let in_use: Vec<u32> = tasks
                .chain([cx.cur.ns, *ns])
                .flat_map(|n| NsKind::ALL.into_iter().map(move |kind| n.get(kind)))
                .collect();
            users.collect_garbage(&in_use);
        }
        for kind in [
            NsKind::Cgroup,
            NsKind::Ipc,
            NsKind::Mnt,
            NsKind::Net,
            NsKind::PidForChildren,
            NsKind::TimeForChildren,
            NsKind::Uts,
        ] {
            if ns.get(kind) != before.get(kind) {
                users.set_owner(ns.get(kind), ns.user).map_err(err)?;
            }
        }
        Ok(())
    }

    /// The namespaces of a task `clone` is creating: the parent's, except that it
    /// joins what the parent set aside for children, and new ones as asked.
    pub(super) fn namespaces_for_clone(
        &mut self,
        cx: &ServiceCtx,
        flags: u64,
    ) -> Result<NsProxy, i64> {
        let mut ns = cx.cur.ns;
        ns.pid = ns.pid_for_children;
        ns.time = ns.time_for_children;
        self.enter_new_namespaces(cx, &mut ns, flags, true)?;
        Ok(ns)
    }

    /// Move the calling task into new namespaces (`unshare`).
    pub(super) fn unshare_namespaces(
        &mut self,
        cx: &mut ServiceCtx,
        flags: u64,
    ) -> Result<(), i64> {
        let mut ns = cx.cur.ns;
        self.enter_new_namespaces(cx, &mut ns, flags, false)?;
        cx.cur.ns = ns;
        Ok(())
    }
}

/// What a guest path is, if it names a namespace link: `/proc/<pid>/ns/<kind>`.
///
/// A proc mount can be anywhere, as `/oldroot/proc` is partway through setting up a
/// container, so what counts is the last `/proc/` in the path.
pub(super) fn parse_ns_path(path: &str) -> Option<(NsTarget, NsKind)> {
    let rest = &path[path.rfind("/proc/")? + "/proc/".len()..];
    let (who, rest) = rest.split_once('/')?;
    let name = rest.strip_prefix("ns/")?;
    let kind = NsKind::from_file_name(name)?;
    let target = match who {
        "self" | "thread-self" => NsTarget::Current,
        pid => NsTarget::Pid(pid.parse().ok()?),
    };
    Some((target, kind))
}

/// Whose namespace link a path names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum NsTarget {
    Current,
    Pid(i32),
}

impl Kernel {
    /// The namespace id of `kind` for `target`, or `ENOENT` if there's no such
    /// process. A process that has exited has no namespaces left to open.
    fn ns_of_target(&self, cx: &ServiceCtx, target: NsTarget, kind: NsKind) -> Result<u32, i64> {
        let pid = match target {
            NsTarget::Current => return Ok(cx.cur.ns.get(kind)),
            NsTarget::Pid(local) => self
                .global_pid(cx, local)
                .ok_or_else(|| err(Errno::ENOENT))?,
        };
        if pid == cx.cur.pid || pid == cx.cur.tgid {
            return Ok(cx.cur.ns.get(kind));
        }
        let sh = self.shared.lock().unwrap();
        let task = sh
            .procs
            .iter()
            .flatten()
            .find(|p| p.info.pid == pid && !matches!(p.info.run, RunState::Zombie(_)));
        task.map(|p| p.info.ns.get(kind))
            .ok_or_else(|| err(Errno::ENOENT))
    }

    /// Bring procfs up to date for this task, ahead of a lookup in it. The other
    /// tasks have to be read before taking `vfs`, which sorts after `shared`.
    fn refresh_procfs(&self, cx: &ServiceCtx) {
        let tasks = self.visible_tasks(cx);
        let mut live = self.proc_self_live(cx);
        live.tasks = tasks;
        {
            let mut sh = self.shared.lock().unwrap();
            let uts = sh.uts(cx.cur.ns.uts);
            live.hostname.clone_from(&uts.hostname);
            live.domainname.clone_from(&uts.domainname);
        }
        let mut vfs = self.vfs_for(cx);
        live.mounts = vfs.listing();
        vfs.with_procfs(|procfs| procfs.update_self(live.clone()));
    }

    /// Answer a syscall aimed at a `/proc/<pid>/ns/<kind>` link, if this is one.
    /// These don't resolve through the filesystem: a link's target only names the
    /// namespace, and opening it gives a descriptor for the namespace itself.
    pub(super) fn ns_path_syscall(
        &self,
        cx: &mut ServiceCtx,
        sys: crate::abi::arch::Sysno,
        args: &[u64; 6],
        mem: &mut GuestMemory,
    ) -> Option<i64> {
        use crate::abi::arch::Sysno;
        const AT_FDCWD: i64 = -100;
        const AT_SYMLINK_NOFOLLOW: u64 = 0x100;
        let (dirfd, ptr) = match sys {
            Sysno::Open | Sysno::Stat | Sysno::Lstat | Sysno::Readlink | Sysno::Access => {
                (AT_FDCWD, args[0])
            }
            Sysno::Openat
            | Sysno::Openat2
            | Sysno::Readlinkat
            | Sysno::Newfstatat
            | Sysno::Statx
            | Sysno::Faccessat
            | Sysno::Faccessat2 => (args[0] as i64, args[1]),
            _ => return None,
        };
        let raw = super::read_path(mem, ptr)?;
        let abs = self.resolve_path(cx, dirfd, &raw);
        if !abs.contains("/proc") {
            return None;
        }
        self.refresh_procfs(cx);
        let inside = Self::as_seen_from_root(cx, &abs);
        let (target, kind) = parse_ns_path(&inside)?;
        let id = match self.ns_of_target(cx, target, kind) {
            Ok(id) => id,
            Err(e) => return Some(e),
        };
        let attrs = || crate::fs::Attrs {
            inode: kind.inode(id),
            ..super::stat::char_device_attrs()
        };
        Some(match sys {
            Sysno::Open | Sysno::Openat | Sysno::Openat2 => {
                i64::from(cx.cur.fds.alloc(Fd::Namespace { kind, id }))
            }
            Sysno::Readlink | Sysno::Readlinkat => {
                let (buf, size) = if sys == Sysno::Readlink {
                    (args[1], args[2])
                } else {
                    (args[2], args[3])
                };
                let link = kind.link(id);
                let n = link.len().min(size as usize);
                if mem.write(buf, &link.as_bytes()[..n]).is_err() {
                    return Some(err(Errno::EFAULT));
                }
                n as i64
            }
            // These see the link itself, which is the filesystem's.
            Sysno::Newfstatat if args[3] & AT_SYMLINK_NOFOLLOW != 0 => return None,
            Sysno::Statx if args[2] & AT_SYMLINK_NOFOLLOW != 0 => return None,
            Sysno::Stat | Sysno::Newfstatat => {
                let buf = if sys == Sysno::Stat { args[1] } else { args[2] };
                super::write_stat_or_fault(mem, buf, &attrs(), self.arch)
            }
            Sysno::Statx => {
                let bytes = super::stat::encode_statx(&attrs());
                if mem.write(args[4], &bytes).is_err() {
                    return Some(err(Errno::EFAULT));
                }
                0
            }
            _ => return None,
        })
    }

    /// `setns(fd, nstype)`: join the namespace `fd` refers to, or with a pidfd,
    /// those of the kinds in `nstype` that the process it names is in.
    pub(super) fn sys_setns(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        fd: u64,
        nstype: u64,
    ) -> i64 {
        const ALL: u64 = CLONE_NEWTIME
            | CLONE_NEWNS
            | CLONE_NEWCGROUP
            | CLONE_NEWUTS
            | CLONE_NEWIPC
            | CLONE_NEWUSER
            | CLONE_NEWPID
            | CLONE_NEWNET;
        if nstype & !ALL != 0 {
            return err(Errno::EINVAL);
        }
        let joins_user = matches!(
            cx.cur.fds.get(fd as i32),
            Some(Fd::Namespace {
                kind: NsKind::User,
                ..
            })
        );
        // Joining a user namespace has its own rule, in `join_user_namespace`.
        if !cx.cur.privileged() && !joins_user {
            return err(Errno::EPERM);
        }
        let others_share_tgid = sh.procs.iter().flatten().any(|p| {
            p.info.tgid == cx.cur.tgid
                && p.info.pid != cx.cur.pid
                && !matches!(p.info.run, RunState::Zombie(_))
        });
        let threaded_refused =
            |flags: u64| others_share_tgid && flags & (CLONE_NEWNS | CLONE_NEWUSER) != 0;
        match cx.cur.fds.get(fd as i32).cloned() {
            Some(Fd::Namespace { kind, id }) => {
                if nstype != 0 && nstype != kind.flag() {
                    return err(Errno::EINVAL);
                }
                if threaded_refused(kind.flag()) {
                    return err(Errno::EINVAL);
                }
                if kind == NsKind::User {
                    return self.join_user_namespace(cx, id);
                }
                if !self.may_administer(cx, id) {
                    return err(Errno::EPERM);
                }
                if matches!(kind, NsKind::Pid | NsKind::PidForChildren)
                    && !self.may_join_pid_namespace(cx, id)
                {
                    return err(Errno::EINVAL);
                }
                let old_mnt = cx.cur.ns.mnt;
                Self::join_namespace(&mut cx.cur.ns, kind, id);
                if kind == NsKind::Mnt {
                    Self::reset_to_new_root(cx);
                    self.carry_open_files(sh, cx, old_mnt);
                }
                0
            }
            Some(Fd::Pidfd(i)) => {
                if nstype == 0 || threaded_refused(nstype) {
                    return err(Errno::EINVAL);
                }
                let pid = self.pollfds.lock().unwrap().pidfds[i].target_pid;
                let target = if pid == cx.cur.pid {
                    cx.cur.ns
                } else {
                    let task = sh.procs.iter().flatten().find(|p| p.info.pid == pid);
                    match task {
                        Some(p) if !matches!(p.info.run, RunState::Zombie(_)) => p.info.ns,
                        _ => return err(Errno::ESRCH),
                    }
                };
                for kind in [
                    NsKind::Cgroup,
                    NsKind::Ipc,
                    NsKind::Mnt,
                    NsKind::Net,
                    NsKind::Pid,
                    NsKind::Time,
                    NsKind::User,
                    NsKind::Uts,
                ] {
                    if kind == NsKind::Pid
                        && nstype & kind.flag() != 0
                        && !self.may_join_pid_namespace(cx, target.pid)
                    {
                        return err(Errno::EINVAL);
                    }
                }
                let old_mnt = cx.cur.ns.mnt;
                for kind in [
                    NsKind::Cgroup,
                    NsKind::Ipc,
                    NsKind::Mnt,
                    NsKind::Net,
                    NsKind::Pid,
                    NsKind::Time,
                    NsKind::User,
                    NsKind::Uts,
                ] {
                    if nstype & kind.flag() != 0 {
                        Self::join_namespace(&mut cx.cur.ns, kind, target.get(kind));
                    }
                }
                if nstype & CLONE_NEWNS != 0 {
                    Self::reset_to_new_root(cx);
                    self.carry_open_files(sh, cx, old_mnt);
                }
                0
            }
            Some(_) => err(Errno::EINVAL),
            None => err(Errno::EBADF),
        }
    }

    /// `setns` into a user namespace. This grants its capabilities, so it needs a
    /// namespace nested in the caller's, where the caller is privileged already or
    /// created it: a namespace's creator has every capability in it from outside.
    /// The caller's ids stay the same ids, and are seen through the new maps.
    fn join_user_namespace(&self, cx: &mut ServiceCtx, id: u32) -> i64 {
        use super::userns::IdKind;
        let current = cx.cur.ns.user;
        let users = self.users.lock().unwrap();
        if id == current {
            return err(Errno::EINVAL);
        }
        let allowed = users.is_within(id, current)
            && (cx.cur.privileged() || users.created_by(id, current, cx.cur.creds.euid));
        if !allowed {
            return err(Errno::EPERM);
        }
        let c = cx.cur.creds;
        let view = |kind, value| {
            let global = users.to_global(current, kind, value).unwrap_or(value);
            users.seen_from_global(id, kind, global)
        };
        cx.cur.creds = super::Creds {
            ruid: view(IdKind::Uid, c.ruid),
            euid: view(IdKind::Uid, c.euid),
            suid: view(IdKind::Uid, c.suid),
            fsuid: view(IdKind::Uid, c.fsuid),
            rgid: view(IdKind::Gid, c.rgid),
            egid: view(IdKind::Gid, c.egid),
            sgid: view(IdKind::Gid, c.sgid),
            fsgid: view(IdKind::Gid, c.fsgid),
        };
        drop(users);
        cx.cur.ns.user = id;
        cx.cur.caps = None;
        cx.cur.ns_caps = true;
        0
    }

    /// A task entering a mount namespace starts at its root, as paths from where
    /// it was don't mean anything there.
    fn reset_to_new_root(cx: &mut ServiceCtx) {
        "/".clone_into(&mut cx.cur.root);
        "/".clone_into(&mut cx.cur.cwd);
    }

    /// Join one namespace. Joining a PID or time namespace affects only the
    /// children the task goes on to create, as on Linux.
    fn join_namespace(ns: &mut NsProxy, kind: NsKind, id: u32) {
        match kind {
            NsKind::Cgroup => ns.cgroup = id,
            NsKind::Ipc => ns.ipc = id,
            NsKind::Mnt => ns.mnt = id,
            NsKind::Net => ns.net = id,
            NsKind::Pid | NsKind::PidForChildren => ns.pid_for_children = id,
            NsKind::Time | NsKind::TimeForChildren => ns.time_for_children = id,
            NsKind::User => ns.user = id,
            NsKind::Uts => ns.uts = id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_naming_a_namespace_link_are_recognized() {
        let cur = NsTarget::Current;
        assert_eq!(parse_ns_path("/proc/self/ns/uts"), Some((cur, NsKind::Uts)));
        assert_eq!(
            parse_ns_path("/proc/thread-self/ns/mnt"),
            Some((cur, NsKind::Mnt))
        );
        assert_eq!(
            parse_ns_path("/proc/42/ns/pid_for_children"),
            Some((NsTarget::Pid(42), NsKind::PidForChildren))
        );
        assert_eq!(
            parse_ns_path("/oldroot/proc/self/ns/uts"),
            Some((cur, NsKind::Uts))
        );
        for other in [
            "/proc/self/ns",
            "/proc/self/ns/nope",
            "/proc/x/ns/uts",
            "/tmp/ns/uts",
        ] {
            assert_eq!(parse_ns_path(other), None, "{other}");
        }
    }

    #[test]
    fn links_look_like_linux_and_differ_by_namespace() {
        assert!(NsKind::Uts.link(0).starts_with("uts:["));
        assert!(NsKind::PidForChildren.link(5).starts_with("pid:["));
        assert_eq!(NsKind::Pid.link(5), NsKind::PidForChildren.link(5));
        assert_ne!(NsKind::Uts.inode(1), NsKind::Uts.inode(2));
        assert_ne!(NsKind::Ipc.inode(1), NsKind::Uts.inode(1));
    }
}
