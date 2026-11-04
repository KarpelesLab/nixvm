//! User namespaces: whose ids mean what, and whose privilege counts where.
//!
//! A task's credentials hold the ids as it sees them. Inside a user namespace these
//! are translated at the syscall boundary against the namespace's `uid_map` and
//! `gid_map`, and file ownership is shown and set through the same maps. A process
//! gets full capabilities in a namespace it creates, but they only count for
//! resources that namespace owns: root in a user namespace can't rename the VM or
//! touch its mounts, only those of namespaces created from inside.
//!
//! Like the PID table, this has its own leaf lock, as checks are needed under any of
//! the kernel's other locks.

use std::collections::BTreeMap;

use super::{Kernel, RunState, ServiceCtx, Shared, Vcpu, err};
use crate::abi::arch::Sysno;
use crate::abi::errno::Errno;
use crate::vcpu::GuestMemory;

/// What an unmapped id reads as (`/proc/sys/kernel/overflowuid`).
pub(super) const OVERFLOW_ID: u32 = 65534;
/// How deeply user namespaces nest, as on Linux.
const MAX_DEPTH: usize = 32;
/// The most lines a map may have (`UID_GID_MAP_MAX_EXTENTS`).
const MAX_EXTENTS: usize = 340;
/// How many namespaces of any kind may have a recorded owner.
const MAX_OWNED: usize = 1 << 17;
/// When to look for namespaces that can be forgotten.
pub(super) const COLLECT_ABOVE: usize = 1 << 16;

/// Which id space a map translates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum IdKind {
    Uid,
    Gid,
}

/// One line of a map: `count` ids starting at `inside` here are `outside` in the
/// parent namespace.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Extent {
    inside: u32,
    outside: u32,
    count: u32,
}

#[derive(Debug, Default)]
struct IdMap(Option<Vec<Extent>>);

impl IdMap {
    fn to_outside(&self, id: u32) -> Option<u32> {
        self.0.as_ref()?.iter().find_map(|e| {
            let off = id.checked_sub(e.inside)?;
            (off < e.count).then(|| e.outside + off)
        })
    }

    fn to_inside(&self, id: u32) -> Option<u32> {
        self.0.as_ref()?.iter().find_map(|e| {
            let off = id.checked_sub(e.outside)?;
            (off < e.count).then(|| e.inside + off)
        })
    }

    /// The text of `/proc/<pid>/uid_map`.
    fn render(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        for e in self.0.iter().flatten() {
            let _ = writeln!(out, "{:>10} {:>10} {:>10}", e.inside, e.outside, e.count);
        }
        out
    }
}

#[derive(Debug)]
struct UserNs {
    parent: u32,
    uids: IdMap,
    gids: IdMap,
    setgroups_denied: bool,
    /// The creator's ids in the parent namespace, which an unprivileged creator is
    /// the only thing it may map.
    creator_uid: u32,
    creator_gid: u32,
}

/// Every user namespace but the initial one, and who owns the other namespaces.
#[derive(Debug, Default)]
pub(super) struct UserNamespaces {
    table: BTreeMap<u32, UserNs>,
    /// The user namespace that owns each namespace of another kind, by its id.
    owners: BTreeMap<u32, u32>,
}

impl UserNamespaces {
    fn chain(&self, mut ns: u32) -> Vec<u32> {
        let mut out = Vec::new();
        while let Some(entry) = self.table.get(&ns) {
            out.push(ns);
            ns = entry.parent;
        }
        out
    }

    /// Record a new namespace nested in `parent`, unless that's too deep.
    pub(super) fn create(
        &mut self,
        ns: u32,
        parent: u32,
        creator_uid: u32,
        creator_gid: u32,
    ) -> Result<(), Errno> {
        if self.chain(parent).len() + 1 >= MAX_DEPTH {
            return Err(Errno::ENOSPC);
        }
        // Once denied, always denied below: a child can't get setgroups back.
        let setgroups_denied = self.table.get(&parent).is_some_and(|p| p.setgroups_denied);
        self.table.insert(
            ns,
            UserNs {
                parent,
                uids: IdMap::default(),
                gids: IdMap::default(),
                setgroups_denied,
                creator_uid,
                creator_gid,
            },
        );
        Ok(())
    }

    /// Whether `inner` is `outer` or nested inside it.
    pub(super) fn is_within(&self, inner: u32, outer: u32) -> bool {
        outer == 0 || inner == outer || self.chain(inner).contains(&outer)
    }

    /// Record that user namespace `user` owns namespace `ns` of some other kind.
    pub(super) fn set_owner(&mut self, ns: u32, user: u32) -> Result<(), Errno> {
        if self.owners.len() >= MAX_OWNED {
            return Err(Errno::ENOSPC);
        }
        self.owners.insert(ns, user);
        Ok(())
    }

    /// The user namespace that owns `ns`. Everything not recorded is the VM's own.
    pub(super) fn owner(&self, ns: u32) -> u32 {
        self.owners.get(&ns).copied().unwrap_or(0)
    }

    /// Whether `ns` is a direct child of `parent` created by `uid` there.
    pub(super) fn created_by(&self, ns: u32, parent: u32, uid: u32) -> bool {
        self.table
            .get(&ns)
            .is_some_and(|n| n.parent == parent && n.creator_uid == uid)
    }

    /// Whether `id` has a mapping in `ns`, which is all that lets a task take it on.
    pub(super) fn is_mapped(&self, ns: u32, kind: IdKind, id: u32) -> bool {
        self.to_global(ns, kind, id).is_some()
    }

    fn map(&self, ns: u32, kind: IdKind) -> Option<&IdMap> {
        let entry = self.table.get(&ns)?;
        Some(match kind {
            IdKind::Uid => &entry.uids,
            IdKind::Gid => &entry.gids,
        })
    }

    /// `id` in `ns`, translated up to the initial namespace.
    pub(super) fn to_global(&self, ns: u32, kind: IdKind, id: u32) -> Option<u32> {
        let mut id = id;
        for level in self.chain(ns) {
            id = self.map(level, kind)?.to_outside(id)?;
        }
        Some(id)
    }

    /// Global `id` as seen in `ns`, or the overflow id if it isn't mapped there.
    pub(super) fn seen_from_global(&self, ns: u32, kind: IdKind, id: u32) -> u32 {
        let mut id = id;
        for level in self.chain(ns).into_iter().rev() {
            match self.map(level, kind).and_then(|m| m.to_inside(id)) {
                Some(inside) => id = inside,
                None => return OVERFLOW_ID,
            }
        }
        id
    }

    /// The text of a `uid_map` or `gid_map` in `ns`.
    pub(super) fn render(&self, ns: u32, kind: IdKind) -> Option<String> {
        self.map(ns, kind).map(IdMap::render)
    }

    /// The text of `setgroups` in `ns`.
    pub(super) fn setgroups_state(&self, ns: u32) -> &'static str {
        if self.table.get(&ns).is_some_and(|n| n.setgroups_denied) {
            "deny\n"
        } else {
            "allow\n"
        }
    }

    /// Whether `setgroups(2)` is refused in `ns`.
    pub(super) fn setgroups_refused(&self, ns: u32) -> bool {
        self.chain(ns)
            .iter()
            .any(|n| self.table.get(n).is_some_and(|e| e.setgroups_denied))
    }

    /// Write `/proc/<pid>/setgroups`.
    fn write_setgroups(&mut self, ns: u32, text: &str) -> Result<(), Errno> {
        let entry = self.table.get_mut(&ns).ok_or(Errno::EPERM)?;
        match text.trim() {
            "deny" => entry.setgroups_denied = true,
            // Once a gid_map exists, or after denying, it can't be allowed again.
            "allow" if !entry.setgroups_denied && entry.gids.0.is_none() => {}
            "allow" => return Err(Errno::EPERM),
            _ => return Err(Errno::EINVAL),
        }
        Ok(())
    }

    /// Write a `uid_map` or `gid_map`. It can be written once. A writer with
    /// privilege in the parent namespace can map anything mapped there; otherwise
    /// only its own id, one id long, and for a `gid_map` only once `setgroups` is
    /// denied, or dropping a group through `setgroups` could gain access.
    fn write_map(
        &mut self,
        ns: u32,
        kind: IdKind,
        text: &str,
        privileged_in_parent: bool,
    ) -> Result<(), Errno> {
        let extents = parse_extents(text)?;
        let entry = self.table.get(&ns).ok_or(Errno::EPERM)?;
        let (existing, own) = match kind {
            IdKind::Uid => (&entry.uids, entry.creator_uid),
            IdKind::Gid => (&entry.gids, entry.creator_gid),
        };
        if existing.0.is_some() {
            return Err(Errno::EPERM);
        }
        let parent = entry.parent;
        if !privileged_in_parent {
            let only_own = matches!(extents.as_slice(), [e] if e.count == 1 && e.outside == own);
            let groups_ok = kind == IdKind::Uid || entry.setgroups_denied;
            if !only_own || !groups_ok {
                return Err(Errno::EPERM);
            }
        }
        // Everything mapped must already mean something in the parent.
        for e in &extents {
            let last = e.outside.checked_add(e.count - 1).ok_or(Errno::EINVAL)?;
            let mapped = |id| self.to_global(parent, kind, id).is_some();
            if !mapped(e.outside) || !mapped(last) {
                return Err(Errno::EPERM);
            }
        }
        let entry = self.table.get_mut(&ns).ok_or(Errno::EPERM)?;
        let map = match kind {
            IdKind::Uid => &mut entry.uids,
            IdKind::Gid => &mut entry.gids,
        };
        map.0 = Some(extents);
        Ok(())
    }

    /// How many namespaces have a recorded owner.
    pub(super) fn owned_count(&self) -> usize {
        self.owners.len()
    }

    /// Forget namespaces nothing refers to any more.
    pub(super) fn collect_garbage(&mut self, in_use: &[u32]) {
        let parents: Vec<u32> = self.table.values().map(|n| n.parent).collect();
        self.table
            .retain(|id, _| in_use.contains(id) || parents.contains(id));
        self.owners.retain(|ns, _| in_use.contains(ns));
    }
}

/// The lines of a map, `inside outside count`, checked for overlap.
fn parse_extents(text: &str) -> Result<Vec<Extent>, Errno> {
    let mut extents = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let mut fields = line.split_whitespace().map(str::parse::<u32>);
        let mut next = || {
            fields
                .next()
                .ok_or(Errno::EINVAL)?
                .map_err(|_| Errno::EINVAL)
        };
        let (inside, outside, count) = (next()?, next()?, next()?);
        if count == 0 || fields.next().is_some() {
            return Err(Errno::EINVAL);
        }
        inside.checked_add(count - 1).ok_or(Errno::EINVAL)?;
        extents.push(Extent {
            inside,
            outside,
            count,
        });
    }
    if extents.is_empty() || extents.len() > MAX_EXTENTS {
        return Err(Errno::EINVAL);
    }
    let overlaps =
        |a: u32, n: u32, b: u32, m: u32| a < b.saturating_add(m) && b < a.saturating_add(n);
    for (i, e) in extents.iter().enumerate() {
        for f in &extents[..i] {
            if overlaps(e.inside, e.count, f.inside, f.count)
                || overlaps(e.outside, e.count, f.outside, f.count)
            {
                return Err(Errno::EINVAL);
            }
        }
    }
    Ok(extents)
}

/// Offsets of a time namespace, `(monotonic, boottime)` in nanoseconds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct TimeOffsets {
    pub(super) monotonic: i64,
    pub(super) boottime: i64,
}

/// A time namespace.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct TimeNs {
    pub(super) offsets: TimeOffsets,
    /// A task is in it, so its clocks are fixed from here on.
    pub(super) entered: bool,
}

/// `/proc/self/timens_offsets` as text.
pub(super) fn render_time_offsets(o: TimeOffsets) -> String {
    const NS: i64 = 1_000_000_000;
    format!(
        "monotonic {} {}\nboottime {} {}\n",
        o.monotonic.div_euclid(NS),
        o.monotonic.rem_euclid(NS),
        o.boottime.div_euclid(NS),
        o.boottime.rem_euclid(NS)
    )
}

/// Apply written `/proc/self/timens_offsets` text to `offsets`.
fn parse_time_offsets(text: &str, offsets: &mut TimeOffsets) -> Result<(), Errno> {
    const NS: i64 = 1_000_000_000;
    let mut parsed = *offsets;
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let mut fields = line.split_whitespace();
        let clock = fields.next().ok_or(Errno::EINVAL)?;
        let secs: i64 = fields
            .next()
            .ok_or(Errno::EINVAL)?
            .parse()
            .map_err(|_| Errno::EINVAL)?;
        let nanos: i64 = fields
            .next()
            .ok_or(Errno::EINVAL)?
            .parse()
            .map_err(|_| Errno::EINVAL)?;
        if !(0..NS).contains(&nanos) || fields.next().is_some() {
            return Err(Errno::EINVAL);
        }
        let total = secs.checked_mul(NS).and_then(|s| s.checked_add(nanos));
        let total = total.ok_or(Errno::ERANGE)?;
        match clock {
            "monotonic" | "1" => parsed.monotonic = total,
            "boottime" | "7" => parsed.boottime = total,
            _ => return Err(Errno::EINVAL),
        }
    }
    *offsets = parsed;
    Ok(())
}

impl Kernel {
    /// Whether this task has privilege over namespace `ns`: it's privileged, and
    /// the user namespace that owns `ns` is its own or one nested inside it.
    pub(super) fn may_administer(&self, cx: &ServiceCtx, ns: u32) -> bool {
        if !cx.cur.privileged() {
            return false;
        }
        if cx.cur.ns.user == 0 {
            return true;
        }
        let users = self.users.lock().unwrap();
        users.is_within(users.owner(ns), cx.cur.ns.user)
    }

    /// Move `info` into new user namespace `ns`: until a map says otherwise its ids
    /// are unmapped, and it has every capability, there.
    pub(super) fn enter_user_namespace(
        &self,
        info: &mut super::ProcInfo,
        ns: u32,
        parent: u32,
    ) -> Result<(), i64> {
        let c = info.creds;
        self.users
            .lock()
            .unwrap()
            .create(ns, parent, c.euid, c.egid)
            .map_err(err)?;
        info.creds = super::Creds {
            ruid: OVERFLOW_ID,
            euid: OVERFLOW_ID,
            suid: OVERFLOW_ID,
            fsuid: OVERFLOW_ID,
            rgid: OVERFLOW_ID,
            egid: OVERFLOW_ID,
            sgid: OVERFLOW_ID,
            fsgid: OVERFLOW_ID,
        };
        info.caps = None;
        info.ns_caps = true;
        Ok(())
    }

    /// Run a syscall for a task in a nested user namespace: requested ids must be
    /// mapped, file owners are set and read through the maps.
    pub(super) fn dispatch_in_user_ns(
        &self,
        cx: &mut ServiceCtx,
        sys: Sysno,
        raw: u64,
        args: &[u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        const UNCHANGED: u32 = u32::MAX;
        let ns = cx.cur.ns.user;
        let check = |this: &Self, kind: IdKind, id: u64| {
            let id = id as u32;
            id == UNCHANGED || this.users.lock().unwrap().is_mapped(ns, kind, id)
        };
        let mut a = *args;
        match sys {
            Sysno::Setuid | Sysno::Setfsuid if !check(self, IdKind::Uid, args[0]) => {
                return err(Errno::EINVAL);
            }
            Sysno::Setgid | Sysno::Setfsgid if !check(self, IdKind::Gid, args[0]) => {
                return err(Errno::EINVAL);
            }
            Sysno::Setreuid | Sysno::Setresuid
                if !(0..3).all(|i| {
                    (i == 2 && sys == Sysno::Setreuid) || check(self, IdKind::Uid, args[i])
                }) =>
            {
                return err(Errno::EINVAL);
            }
            Sysno::Setregid | Sysno::Setresgid
                if !(0..3).all(|i| {
                    (i == 2 && sys == Sysno::Setregid) || check(self, IdKind::Gid, args[i])
                }) =>
            {
                return err(Errno::EINVAL);
            }
            Sysno::Setgroups if self.users.lock().unwrap().setgroups_refused(ns) => {
                return err(Errno::EPERM);
            }
            Sysno::Chown | Sysno::Lchown | Sysno::Fchown | Sysno::Fchownat => {
                let (u, g) = if sys == Sysno::Fchownat {
                    (2, 3)
                } else {
                    (1, 2)
                };
                let users = self.users.lock().unwrap();
                for (i, kind) in [(u, IdKind::Uid), (g, IdKind::Gid)] {
                    let id = args[i] as u32;
                    if id != UNCHANGED {
                        let Some(global) = users.to_global(ns, kind, id) else {
                            return err(Errno::EINVAL);
                        };
                        a[i] = u64::from(global);
                    }
                }
            }
            _ => {}
        }
        let ret = self.dispatch_ids_translated(cx, sys, raw, &a, vcpu, mem);
        if ret == 0 {
            self.show_owner_through_maps(cx, sys, args, mem);
        }
        ret
    }

    /// Rewrite the owner in a `struct stat` or `statx` a syscall just filled in.
    fn show_owner_through_maps(
        &self,
        cx: &ServiceCtx,
        sys: Sysno,
        args: &[u64; 6],
        mem: &mut GuestMemory,
    ) {
        const STATX_UID: u64 = 20;
        const STATX_GID: u64 = 24;
        let (x86_uid, x86_gid) = (28, 32);
        let (arm_uid, arm_gid) = (24, 28);
        let is_x86 = self.arch == crate::abi::Arch::X86_64;
        let (buf, uid_at, gid_at) = match sys {
            Sysno::Statx => (args[4], STATX_UID, STATX_GID),
            Sysno::Fstat | Sysno::Stat | Sysno::Lstat | Sysno::Newfstatat => {
                let buf = if sys == Sysno::Newfstatat {
                    args[2]
                } else {
                    args[1]
                };
                if is_x86 {
                    (buf, x86_uid, x86_gid)
                } else {
                    (buf, arm_uid, arm_gid)
                }
            }
            _ => return,
        };
        let users = self.users.lock().unwrap();
        let ns = cx.cur.ns.user;
        for (at, kind) in [(uid_at, IdKind::Uid), (gid_at, IdKind::Gid)] {
            if let Ok(global) = mem.read_u32(buf + at) {
                let seen = users.seen_from_global(ns, kind, global);
                let _ = mem.write(buf + at, &seen.to_le_bytes());
            }
        }
    }

    /// A write to one of `/proc/<pid>`'s control files: the id maps, `setgroups`,
    /// `timens_offsets`. These change kernel state rather than a file.
    pub(super) fn proc_control_write(
        &self,
        cx: &mut ServiceCtx,
        fd: u64,
        buf: u64,
        count: u64,
        mem: &GuestMemory,
    ) -> Option<i64> {
        let Some(super::Fd::File { path, .. }) = cx.cur.fds.get(fd as i32) else {
            return None;
        };
        let inside = Self::as_seen_from_root(cx, path);
        // The proc mount can be anywhere; a container mid-setup has it at /oldroot/proc.
        let rest = &inside[inside.rfind("/proc/")? + "/proc/".len()..];
        if let Some(name @ ("hostname" | "domainname")) = rest.strip_prefix("sys/kernel/") {
            return Some(self.write_uts_file(cx, name == "domainname", buf, count, mem));
        }
        let (who, file) = rest.split_once('/')?;
        if !matches!(file, "uid_map" | "gid_map" | "setgroups" | "timens_offsets") {
            return None;
        }
        let Ok(bytes) = mem.read_vec(buf, count.min(32 * 1024) as usize) else {
            return Some(err(Errno::EFAULT));
        };
        let Ok(text) = String::from_utf8(bytes) else {
            return Some(err(Errno::EINVAL));
        };
        let target = if who == "self" || who == "thread-self" {
            None
        } else {
            let Some(global) = who.parse().ok().and_then(|l| self.global_pid(cx, l)) else {
                return Some(err(Errno::ESRCH));
            };
            (global != cx.cur.pid && global != cx.cur.tgid).then_some(global)
        };
        let mut sh = self.shared.lock().unwrap();
        let ns = match target {
            None => cx.cur.ns,
            Some(pid) => {
                let task = sh.procs.iter().flatten().find(|p| p.info.pid == pid);
                match task {
                    Some(p) => p.info.ns,
                    None => return Some(err(Errno::ESRCH)),
                }
            }
        };
        let written = match file {
            "timens_offsets" => self.write_time_offsets(cx, &text, mem),
            "setgroups" => self.users.lock().unwrap().write_setgroups(ns.user, &text),
            _ => {
                let kind = if file == "uid_map" {
                    IdKind::Uid
                } else {
                    IdKind::Gid
                };
                self.write_id_map(&mut sh, cx, ns.user, kind, &text)
            }
        };
        Some(written.map_or_else(err, |()| count as i64))
    }

    /// A write to `/proc/sys/kernel/hostname` or `domainname`: `sethostname` /
    /// `setdomainname` on the writer's UTS namespace, taking the text up to the
    /// first newline, at most 64 bytes. Sysctls are writable only by root of the
    /// first user namespace (`EACCES` otherwise), as on Linux.
    fn write_uts_file(
        &self,
        cx: &ServiceCtx,
        domain: bool,
        buf: u64,
        count: u64,
        mem: &GuestMemory,
    ) -> i64 {
        if cx.cur.ns.user != 0 || cx.cur.creds.euid != 0 {
            return err(Errno::EACCES);
        }
        let Ok(bytes) = mem.read_vec(buf, count.min(4096) as usize) else {
            return err(Errno::EFAULT);
        };
        let end = bytes
            .iter()
            .position(|&b| b == b'\n' || b == 0)
            .unwrap_or(bytes.len())
            .min(64);
        let name = String::from_utf8_lossy(&bytes[..end]).into_owned();
        let mut sh = self.shared.lock().unwrap();
        let uts = sh.uts(cx.cur.ns.uts);
        if domain {
            uts.domainname = name;
        } else {
            uts.hostname = name;
        }
        count as i64
    }

    /// `uid_map`/`gid_map`: record the map, then give every task in the namespace
    /// the ids it now has. It was showing the overflow id until now.
    fn write_id_map(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        ns: u32,
        kind: IdKind,
        text: &str,
    ) -> Result<(), Errno> {
        // A map is written with privilege in the namespace's parent, or by the
        // creator itself, for its own id. Only the first of those can map others.
        let (parent, creator) = {
            let users = self.users.lock().unwrap();
            let entry = users.table.get(&ns).ok_or(Errno::EPERM)?;
            let creator = match kind {
                IdKind::Uid => entry.creator_uid,
                IdKind::Gid => entry.creator_gid,
            };
            (entry.parent, creator)
        };
        let privileged_in_parent = cx.cur.ns.user == parent && cx.cur.privileged();
        if !privileged_in_parent && cx.cur.ns.user != ns {
            return Err(Errno::EPERM);
        }
        let mut users = self.users.lock().unwrap();
        users.write_map(ns, kind, text, privileged_in_parent)?;
        let seen = users
            .map(ns, kind)
            .and_then(|m| m.to_inside(creator))
            .unwrap_or(OVERFLOW_ID);
        drop(users);
        let apply = |c: &mut super::Creds| match kind {
            IdKind::Uid => {
                (c.ruid, c.euid, c.suid, c.fsuid) = (seen, seen, seen, seen);
            }
            IdKind::Gid => {
                (c.rgid, c.egid, c.sgid, c.fsgid) = (seen, seen, seen, seen);
            }
        };
        if cx.cur.ns.user == ns {
            apply(&mut cx.cur.creds);
        }
        for p in sh.procs.iter_mut().flatten() {
            if p.info.ns.user == ns && !matches!(p.info.run, RunState::Zombie(_)) {
                apply(&mut p.info.creds);
            }
        }
        Ok(())
    }

    /// `timens_offsets`: only for a namespace nothing has entered yet, which is
    /// one the caller set aside for its children with `unshare(CLONE_NEWTIME)`.
    fn write_time_offsets(
        &self,
        cx: &ServiceCtx,
        text: &str,
        mem: &GuestMemory,
    ) -> Result<(), Errno> {
        let ns = cx.cur.ns.time_for_children;
        if ns == cx.cur.ns.time {
            return Err(Errno::EACCES);
        }
        let mut times = self.time_offsets.lock().unwrap();
        let entry = times.entry(ns).or_default();
        if entry.entered {
            return Err(Errno::EACCES);
        }
        parse_time_offsets(text, &mut entry.offsets)?;
        // The vDSO reads the clock without entering the kernel, so it can't apply
        // an offset. Sending every clock read to the kernel from here on can.
        if entry.offsets != TimeOffsets::default() {
            mem.disable_vdso_clocks();
        }
        Ok(())
    }

    /// The offsets of the time namespace this task is in.
    pub(super) fn time_offsets_for(&self, cx: &ServiceCtx) -> TimeOffsets {
        if cx.cur.ns.time == 0 {
            return TimeOffsets::default();
        }
        let times = self.time_offsets.lock().unwrap();
        times
            .get(&cx.cur.ns.time)
            .map(|t| t.offsets)
            .unwrap_or_default()
    }

    /// The maps and related files as `/proc/self` shows them to this task.
    pub(super) fn proc_identity_files(&self, cx: &ServiceCtx) -> (String, String, String) {
        const IDENTITY: &str = "         0          0 4294967295\n";
        if cx.cur.ns.user == 0 {
            return (IDENTITY.into(), IDENTITY.into(), "allow\n".into());
        }
        let users = self.users.lock().unwrap();
        let ns = cx.cur.ns.user;
        (
            users.render(ns, IdKind::Uid).unwrap_or_default(),
            users.render(ns, IdKind::Gid).unwrap_or_default(),
            users.setgroups_state(ns).to_string(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with_map(uid_line: &str) -> UserNamespaces {
        let mut t = UserNamespaces::default();
        t.create(1, 0, 1000, 1000).unwrap();
        t.write_map(1, IdKind::Uid, uid_line, false).unwrap();
        t
    }

    #[test]
    fn a_map_translates_both_ways_and_unmapped_ids_overflow() {
        let t = with_map("0 1000 1\n");
        assert_eq!(t.to_global(1, IdKind::Uid, 0), Some(1000));
        assert_eq!(t.to_global(1, IdKind::Uid, 1), None);
        assert_eq!(t.seen_from_global(1, IdKind::Uid, 1000), 0);
        assert_eq!(t.seen_from_global(1, IdKind::Uid, 0), OVERFLOW_ID);
        assert_eq!(
            t.to_global(0, IdKind::Uid, 7),
            Some(7),
            "the initial map is identity"
        );
    }

    #[test]
    fn maps_chain_through_nested_namespaces() {
        let mut t = UserNamespaces::default();
        t.create(1, 0, 0, 0).unwrap();
        t.write_map(1, IdKind::Uid, "0 100000 65536\n", true)
            .unwrap();
        t.create(2, 1, 5, 5).unwrap();
        t.write_map(2, IdKind::Uid, "0 5 1\n", false).unwrap();
        assert_eq!(t.to_global(2, IdKind::Uid, 0), Some(100_005));
        assert_eq!(t.seen_from_global(2, IdKind::Uid, 100_005), 0);
        assert_eq!(t.seen_from_global(2, IdKind::Uid, 100_006), OVERFLOW_ID);
    }

    #[test]
    fn an_unprivileged_writer_can_only_map_itself_once() {
        let mut t = UserNamespaces::default();
        t.create(1, 0, 1000, 1000).unwrap();
        let wrote = |t: &mut UserNamespaces, line: &str| t.write_map(1, IdKind::Uid, line, false);
        assert_eq!(
            wrote(&mut t, "0 0 1\n"),
            Err(Errno::EPERM),
            "someone else's id"
        );
        assert_eq!(
            wrote(&mut t, "0 1000 2\n"),
            Err(Errno::EPERM),
            "more than one"
        );
        assert_eq!(wrote(&mut t, "0 1000 1\n"), Ok(()));
        assert_eq!(wrote(&mut t, "0 1000 1\n"), Err(Errno::EPERM), "only once");
    }

    #[test]
    fn a_gid_map_needs_setgroups_denied_when_unprivileged() {
        let mut t = UserNamespaces::default();
        t.create(1, 0, 1000, 1000).unwrap();
        assert_eq!(
            t.write_map(1, IdKind::Gid, "0 1000 1\n", false),
            Err(Errno::EPERM)
        );
        t.write_setgroups(1, "deny\n").unwrap();
        assert!(t.setgroups_refused(1));
        assert_eq!(t.write_map(1, IdKind::Gid, "0 1000 1\n", false), Ok(()));
        assert_eq!(t.write_setgroups(1, "allow"), Err(Errno::EPERM));
    }

    #[test]
    fn a_child_inherits_setgroups_being_denied() {
        let mut t = UserNamespaces::default();
        t.create(1, 0, 1000, 1000).unwrap();
        t.write_setgroups(1, "deny").unwrap();
        t.create(2, 1, 0, 0).unwrap();
        assert_eq!(t.setgroups_state(2), "deny\n");
        t.write_map(1, IdKind::Gid, "0 1000 1\n", false).unwrap();
        assert_eq!(t.write_map(2, IdKind::Gid, "0 0 1\n", false), Ok(()));
    }

    #[test]
    fn malformed_and_overlapping_maps_are_refused() {
        for bad in [
            "",
            "0 0",
            "0 0 0",
            "a b c",
            "0 0 1 9",
            "0 0 5\n3 9 1",
            "0 0 5\n9 3 1",
        ] {
            assert_eq!(parse_extents(bad), Err(Errno::EINVAL), "{bad:?}");
        }
        assert_eq!(parse_extents("0 0 5\n5 5 5").unwrap().len(), 2);
    }

    #[test]
    fn ownership_decides_who_may_administer() {
        let mut t = UserNamespaces::default();
        t.create(1, 0, 0, 0).unwrap();
        t.set_owner(50, 1).unwrap();
        assert_eq!(t.owner(50), 1);
        assert_eq!(t.owner(0), 0, "the VM's own");
        assert!(t.is_within(t.owner(50), 1));
        assert!(!t.is_within(t.owner(0), 1));
        assert!(t.is_within(t.owner(0), 0));
    }

    #[test]
    fn time_offsets_parse_and_render() {
        let mut o = TimeOffsets::default();
        parse_time_offsets("monotonic 5 10\nboottime -1 0\n", &mut o).unwrap();
        assert_eq!(o.monotonic, 5_000_000_010);
        assert_eq!(o.boottime, -1_000_000_000);
        assert_eq!(render_time_offsets(o), "monotonic 5 10\nboottime -1 0\n");
        assert_eq!(parse_time_offsets("nope 1 1", &mut o), Err(Errno::EINVAL));
        assert_eq!(
            parse_time_offsets("monotonic 1 -1", &mut o),
            Err(Errno::EINVAL)
        );
    }
}
