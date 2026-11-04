//! PID namespaces: the numbers a task sees for itself and others.
//!
//! Inside the kernel every task has one global id, used everywhere. A task in a
//! nested PID namespace sees different numbers, so ids are translated as they cross
//! the syscall boundary, here. The initial namespace's numbers are the global ones,
//! so a task there pays nothing, and a task can only name what its namespace
//! contains, since nothing else has a number there.
//!
//! The table has a lock of its own, a leaf nothing else is taken under, because
//! translation is needed from syscalls holding any of the kernel's other locks.

use std::collections::BTreeMap;

use super::{Kernel, RunState, ServiceCtx, Shared, Vcpu, err, pgid_of};
use crate::abi::arch::Sysno;
use crate::abi::errno::Errno;
use crate::vcpu::GuestMemory;

/// How deeply PID namespaces nest (Linux's `MAX_PID_NS_LEVEL`).
const MAX_DEPTH: usize = 32;

/// One PID namespace below the initial one.
#[derive(Debug)]
struct PidNs {
    parent: u32,
    /// Global id to the number it has here, for every task in this namespace and
    /// those nested in it.
    local: BTreeMap<i32, i32>,
    /// The reverse of [`Self::local`].
    global: BTreeMap<i32, i32>,
    next: i32,
    /// The global id of its first process, its PID 1.
    init: Option<i32>,
    /// Its init has exited: nothing can be created in it again.
    dead: bool,
}

/// Every PID namespace but the initial one, by namespace id.
#[derive(Debug, Default)]
pub(super) struct PidNamespaces {
    table: BTreeMap<u32, PidNs>,
}

impl PidNamespaces {
    /// `ns`, then its ancestors, up to but not including the initial namespace.
    fn chain(&self, mut ns: u32) -> Vec<u32> {
        let mut out = Vec::new();
        while let Some(entry) = self.table.get(&ns) {
            out.push(ns);
            ns = entry.parent;
        }
        out
    }

    /// Record a new namespace nested in `parent`, unless that's too deep.
    pub(super) fn create(&mut self, ns: u32, parent: u32) -> Result<(), Errno> {
        if self.chain(parent).len() + 1 >= MAX_DEPTH {
            return Err(Errno::ENOSPC);
        }
        self.table.insert(
            ns,
            PidNs {
                parent,
                local: BTreeMap::new(),
                global: BTreeMap::new(),
                next: 1,
                init: None,
                dead: false,
            },
        );
        Ok(())
    }

    /// Whether `ns` still takes new tasks. A namespace whose init has gone doesn't.
    pub(super) fn accepts_tasks(&self, ns: u32) -> bool {
        ns == 0 || self.table.get(&ns).is_some_and(|n| !n.dead)
    }

    /// Forget finished namespaces no task can refer to any more: not in use by any
    /// task, directly or as a parent. The record of a dead one has to outlast its
    /// tasks, since a process that unshared can still try to fork into it.
    pub(super) fn collect_garbage(&mut self, in_use: &[u32]) {
        let parents: Vec<u32> = self.table.values().map(|n| n.parent).collect();
        self.table
            .retain(|id, n| !n.local.is_empty() || in_use.contains(id) || parents.contains(id));
    }

    /// How many namespaces exist below the initial one.
    pub(super) fn count(&self) -> usize {
        self.table.len()
    }

    /// Give a new task its number in `ns` and in each namespace above it.
    pub(super) fn add_task(&mut self, global: i32, ns: u32) {
        for id in self.chain(ns) {
            let entry = self.table.get_mut(&id).expect("from chain");
            let local = entry.next;
            entry.next += 1;
            entry.local.insert(global, local);
            entry.global.insert(local, global);
            if id == ns && entry.init.is_none() {
                entry.init = Some(global);
            }
        }
    }

    /// Forget a task that has gone for good.
    pub(super) fn remove_task(&mut self, global: i32, ns: u32) {
        for id in self.chain(ns) {
            let entry = self.table.get_mut(&id).expect("from chain");
            if let Some(local) = entry.local.remove(&global) {
                entry.global.remove(&local);
            }
        }
    }

    /// The number task `global` has in `ns`, or 0 if it isn't visible there.
    pub(super) fn to_local(&self, global: i32, ns: u32) -> i32 {
        if ns == 0 {
            return global;
        }
        let entry = self.table.get(&ns);
        entry
            .and_then(|n| n.local.get(&global))
            .copied()
            .unwrap_or(0)
    }

    /// The global id of whatever has the number `local` in `ns`.
    pub(super) fn to_global(&self, local: i32, ns: u32) -> Option<i32> {
        if ns == 0 {
            return Some(local);
        }
        self.table.get(&ns)?.global.get(&local).copied()
    }

    /// The init of `ns`, or `None` for the initial namespace.
    pub(super) fn init_of(&self, ns: u32) -> Option<i32> {
        self.table.get(&ns).and_then(|n| n.init)
    }

    /// Whether `inner` is `outer` or nested inside it. Everything is inside the
    /// initial namespace, which isn't in the table.
    pub(super) fn is_within(&self, inner: u32, outer: u32) -> bool {
        outer == 0 || self.chain(inner).contains(&outer)
    }

    /// Mark `ns` as finished and list everything in it, nested namespaces included.
    pub(super) fn close(&mut self, ns: u32) -> Vec<i32> {
        let Some(entry) = self.table.get_mut(&ns) else {
            return Vec::new();
        };
        entry.dead = true;
        entry.local.keys().copied().collect()
    }

    /// Everything visible in `ns`.
    pub(super) fn members(&self, ns: u32) -> Vec<i32> {
        self.table
            .get(&ns)
            .map_or_else(Vec::new, |n| n.local.keys().copied().collect())
    }
}

/// How many PID namespaces may exist at once (`user.max_pid_namespaces`, about).
const MAX_NAMESPACES: usize = 1 << 16;

impl Kernel {
    /// Record new PID namespace `ns`, nested in `parent`.
    pub(super) fn new_pid_namespace(
        &self,
        sh: &Shared,
        cx: &ServiceCtx,
        ns: u32,
        parent: u32,
    ) -> Result<(), i64> {
        let mut pids = self.pids.lock().unwrap();
        if pids.count() >= MAX_NAMESPACES / 2 {
            let tasks = sh.procs.iter().flatten().map(|p| &p.info);
            let in_use: Vec<u32> = tasks
                .chain([&cx.cur])
                .flat_map(|info| [info.ns.pid, info.ns.pid_for_children])
                .collect();
            pids.collect_garbage(&in_use);
        }
        if pids.count() >= MAX_NAMESPACES {
            return Err(err(Errno::ENOSPC));
        }
        pids.create(ns, parent).map_err(err)
    }

    /// The number the current task sees for task `global`, 0 if it can't see it.
    pub(super) fn visible_pid(&self, cx: &ServiceCtx, global: i32) -> i32 {
        if cx.cur.ns.pid == 0 {
            return global;
        }
        self.pids.lock().unwrap().to_local(global, cx.cur.ns.pid)
    }

    /// The global id of what the current task calls `local`.
    pub(super) fn global_pid(&self, cx: &ServiceCtx, local: i32) -> Option<i32> {
        if cx.cur.ns.pid == 0 {
            return Some(local);
        }
        self.pids.lock().unwrap().to_global(local, cx.cur.ns.pid)
    }

    /// Whether a signal may reach `target`: a namespace's init is shielded from
    /// signals it hasn't installed a handler for or blocked. Only `SIGKILL` and
    /// `SIGSTOP` from an outer namespace get through regardless.
    pub(super) fn signal_reaches_init(
        &self,
        cx: &ServiceCtx,
        target: &super::ProcInfo,
        sig: u64,
    ) -> bool {
        const SIGKILL: u64 = 9;
        const SIGSTOP: u64 = 19;
        const SIG_DFL: u64 = 0;
        if sig == 0 || target.ns.pid == 0 {
            return true;
        }
        let pids = self.pids.lock().unwrap();
        if pids.init_of(target.ns.pid) != Some(target.tgid) {
            return true;
        }
        if cx.cur.ns.pid != target.ns.pid
            && pids.is_within(target.ns.pid, cx.cur.ns.pid)
            && matches!(sig, SIGKILL | SIGSTOP)
        {
            return true;
        }
        // A signal init has blocked still gets queued, for sigwaitinfo or a
        // signalfd to collect.
        let blocked = sig <= 64 && target.blocked & (1u64 << (sig - 1)) != 0;
        !matches!(sig, SIGKILL | SIGSTOP)
            && (blocked || target.handlers.get(sig as usize).handler != SIG_DFL)
    }

    /// Run a syscall for a task in a nested PID namespace: the ids in its arguments
    /// and result are numbers there, and the kernel works in global ids.
    #[allow(clippy::match_same_arms)] // a table: one arm per group of syscalls
    pub(super) fn dispatch_in_pid_ns(
        &self,
        cx: &mut ServiceCtx,
        sys: Sysno,
        raw: u64,
        args: &[u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        // A pid argument: 0 means "me" (or "my group") and stays 0.
        let positive = |this: &Self, cx: &ServiceCtx, v: u64| -> Result<u64, i64> {
            let local = v as i32;
            if local == 0 {
                return Ok(0);
            }
            if local < 0 {
                return Err(err(Errno::EINVAL));
            }
            this.global_pid(cx, local)
                .map(|g| g as u64)
                .ok_or_else(|| err(Errno::ESRCH))
        };
        // A wait or kill target: >0 a task, -1 everything, <-1 a process group.
        let target = |this: &Self, cx: &ServiceCtx, v: u64| -> Result<u64, i64> {
            let local = v as i32;
            match local {
                0 | -1 => Ok(v),
                l if l > 0 => this
                    .global_pid(cx, l)
                    .map(|g| g as u64)
                    .ok_or_else(|| err(Errno::ESRCH)),
                l => this
                    .global_pid(cx, -l)
                    .map(|g| i64::from(-g) as u64)
                    .ok_or_else(|| err(Errno::ESRCH)),
            }
        };
        let mut a = *args;
        macro_rules! map_arg {
            ($map:ident, $i:expr) => {
                match $map(self, cx, a[$i]) {
                    Ok(v) => a[$i] = v,
                    Err(e) => return e,
                }
            };
        }
        let view = |this: &Self, cx: &ServiceCtx, ret: i64| -> i64 {
            if ret > 0 {
                i64::from(this.visible_pid(cx, ret as i32))
            } else {
                ret
            }
        };
        match sys {
            Sysno::Getpid => return i64::from(self.visible_pid(cx, cx.cur.tgid)),
            Sysno::Gettid => return i64::from(self.visible_pid(cx, cx.cur.pid)),
            Sysno::Getppid => return i64::from(self.visible_pid(cx, cx.cur.ppid)),
            Sysno::Getpgrp => return i64::from(self.visible_pid(cx, pgid_of(&cx.cur))),
            Sysno::Setsid | Sysno::Getpgid | Sysno::Getsid => {
                if sys != Sysno::Setsid {
                    map_arg!(positive, 0);
                }
                let ret = self.dispatch_impl(cx, sys, raw, &a, vcpu, mem);
                return view(self, cx, ret);
            }
            Sysno::Setpgid => {
                map_arg!(positive, 0);
                map_arg!(positive, 1);
            }
            Sysno::Kill | Sysno::Tkill | Sysno::RtSigqueueinfo => {
                if sys == Sysno::Kill && args[0] as u32 == u32::MAX {
                    return self.kill_every_visible(cx, raw, args, vcpu, mem);
                }
                map_arg!(target, 0);
            }
            Sysno::Tgkill | Sysno::RtTgsigqueueinfo => map_arg!(positive, 1),
            // What wait returns is translated inside it: a reaped child no longer
            // has a number by the time this would see it.
            Sysno::Wait4 => map_arg!(target, 0),
            Sysno::Waitid => {
                const P_PID: u64 = 1;
                const P_PGID: u64 = 2;
                if matches!(args[0], P_PID | P_PGID) {
                    map_arg!(positive, 1);
                }
            }
            Sysno::PidfdOpen
            | Sysno::Prlimit64
            | Sysno::SchedGetaffinity
            | Sysno::SchedSetaffinity
            | Sysno::SchedGetscheduler
            | Sysno::SchedSetscheduler
            | Sysno::SchedGetparam
            | Sysno::SchedSetparam
            | Sysno::SchedRrGetInterval
            | Sysno::ProcessVmReadv
            | Sysno::ProcessVmWritev => map_arg!(positive, 0),
            Sysno::Kcmp => {
                map_arg!(positive, 0);
                map_arg!(positive, 1);
            }
            Sysno::Getpriority | Sysno::Setpriority | Sysno::IoprioGet | Sysno::IoprioSet => {
                // `who` is a pid for PRIO_PROCESS / IOPRIO_WHO_PROCESS, and a
                // process group for PRIO_PGRP / IOPRIO_WHO_PGRP.
                let by_process = u64::from(!matches!(sys, Sysno::Getpriority | Sysno::Setpriority));
                if args[0] == by_process || args[0] == by_process + 1 {
                    map_arg!(positive, 1);
                }
            }
            // The pids these report back are numbers in the caller's namespace.
            Sysno::Msgctl | Sysno::Shmctl | Sysno::Semctl | Sysno::Fcntl => {
                return self.ipc_and_fcntl_in_pid_ns(cx, sys, raw, &a, vcpu, mem);
            }
            _ => {}
        }
        self.dispatch_impl(cx, sys, raw, &a, vcpu, mem)
    }

    /// The SysV IPC and `fcntl` commands that carry pids, for a task in a nested
    /// PID namespace: `IPC_STAT`'s last-sender/creator pids, `semctl(GETPID)`,
    /// `F_GETLK`'s lock holder, and the `F_SETOWN`/`F_GETOWN` owner. A pid that
    /// isn't in the namespace reads as 0, as on Linux.
    fn ipc_and_fcntl_in_pid_ns(
        &self,
        cx: &mut ServiceCtx,
        sys: Sysno,
        raw: u64,
        args: &[u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        const IPC_STAT: u64 = 2;
        const MSG_STAT: u64 = 11;
        const MSG_STAT_ANY: u64 = 13;
        const SHM_STAT: u64 = 13;
        const SHM_STAT_ANY: u64 = 15;
        const SEM_GETPID: u64 = 11;
        const F_GETLK: u64 = 5;
        const F_SETOWN: u64 = 8;
        const F_GETOWN: u64 = 9;
        const F_SETOWN_EX: u64 = 15;
        const F_GETOWN_EX: u64 = 16;
        const F_UNLCK: i16 = 2;
        let seen = |this: &Self, cx: &ServiceCtx, global: i32| -> i32 {
            match global.cmp(&0) {
                std::cmp::Ordering::Greater => this.visible_pid(cx, global),
                std::cmp::Ordering::Less => -this.visible_pid(cx, -global),
                std::cmp::Ordering::Equal => 0,
            }
        };
        // Rewrite the pids (`i32`s) at `offsets` from `base` in guest memory.
        let rewrite =
            |this: &Self, cx: &ServiceCtx, mem: &mut GuestMemory, base: u64, offsets: &[u64]| {
                for &o in offsets {
                    if let Ok(v) = mem.read_u32(base + o) {
                        let _ = mem.write(base + o, &seen(this, cx, v as i32).to_le_bytes());
                    }
                }
            };
        // A local pid (or negated process group) to the global one; one that
        // isn't in the namespace names nothing.
        let global = |this: &Self, cx: &ServiceCtx, local: i32| -> i32 {
            let g = |l: i32| this.global_pid(cx, l).unwrap_or(0);
            match local {
                0 => 0,
                l if l > 0 => g(l),
                l => -g(-l),
            }
        };
        // IPC commands may carry IPC_64 (0x100); fcntl's are whole numbers.
        let cmd = match sys {
            Sysno::Semctl => args[2] & !0x100,
            Sysno::Fcntl => args[1],
            _ => args[1] & !0x100,
        };
        let mut a = *args;
        match (sys, cmd) {
            (Sysno::Fcntl, F_SETOWN) => a[2] = i64::from(global(self, cx, args[2] as i32)) as u64,
            (Sysno::Fcntl, F_SETOWN_EX) => {
                // struct f_owner_ex { int type; pid_t pid; }: translated in place
                // for the call, then put back as the caller wrote it.
                let Ok(local) = mem.read_u32(args[2] + 4) else {
                    return err(Errno::EFAULT);
                };
                let g = global(self, cx, (local as i32).abs());
                let _ = mem.write(args[2] + 4, &g.to_le_bytes());
                let ret = self.dispatch_impl(cx, sys, raw, &a, vcpu, mem);
                let _ = mem.write(args[2] + 4, &local.to_le_bytes());
                return ret;
            }
            _ => {}
        }
        let ret = self.dispatch_impl(cx, sys, raw, &a, vcpu, mem);
        if ret < 0 {
            return ret;
        }
        match (sys, cmd) {
            (Sysno::Msgctl, IPC_STAT | MSG_STAT | MSG_STAT_ANY) => {
                rewrite(self, cx, mem, args[2], &[96, 100]); // msg_lspid, msg_lrpid
            }
            (Sysno::Shmctl, IPC_STAT | SHM_STAT | SHM_STAT_ANY) => {
                rewrite(self, cx, mem, args[2], &[80, 84]); // shm_cpid, shm_lpid
            }
            // Both return a pid.
            (Sysno::Semctl, SEM_GETPID) | (Sysno::Fcntl, F_GETOWN) => {
                return i64::from(seen(self, cx, ret as i32));
            }
            (Sysno::Fcntl, F_GETLK) => {
                // l_pid, when a conflicting lock was found.
                if mem.read_u32(args[2]).is_ok_and(|t| t as i16 != F_UNLCK) {
                    rewrite(self, cx, mem, args[2], &[24]);
                }
            }
            (Sysno::Fcntl, F_GETOWN_EX) => rewrite(self, cx, mem, args[2], &[4]),
            _ => {}
        }
        ret
    }

    /// `kill(-1, sig)` inside a PID namespace: everything visible in it, except the
    /// caller and the namespace's init.
    fn kill_every_visible(
        &self,
        cx: &mut ServiceCtx,
        raw: u64,
        args: &[u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        let (members, init) = {
            let pids = self.pids.lock().unwrap();
            (pids.members(cx.cur.ns.pid), pids.init_of(cx.cur.ns.pid))
        };
        let mut hit = false;
        for global in members {
            if global == cx.cur.pid || Some(global) == init {
                continue;
            }
            let one = [global as u64, args[1], 0, 0, 0, 0];
            hit |= self.dispatch_impl(cx, Sysno::Kill, raw, &one, vcpu, mem) == 0;
        }
        if hit { 0 } else { err(Errno::ESRCH) }
    }

    /// The last thread of a namespace's init has exited: everything in that
    /// namespace dies with it, and nothing new can be created there.
    pub(super) fn close_pid_namespace_if_init(&self, sh: &mut Shared, cx: &ServiceCtx) {
        const SIGKILL_BIT: u64 = 1 << (9 - 1);
        let ns = cx.cur.ns.pid;
        if ns == 0 || self.pids.lock().unwrap().init_of(ns) != Some(cx.cur.tgid) {
            return;
        }
        let tgid = cx.cur.tgid;
        let threads_left = sh.procs.iter().flatten().any(|p| {
            p.info.tgid == tgid
                && p.info.pid != cx.cur.pid
                && !matches!(p.info.run, RunState::Zombie(_))
        });
        if threads_left {
            return;
        }
        let doomed = self.pids.lock().unwrap().close(ns);
        for p in sh.procs.iter_mut().flatten() {
            if doomed.contains(&p.info.pid) && !matches!(p.info.run, RunState::Zombie(_)) {
                p.info.pending |= SIGKILL_BIT;
                p.info.parked = false;
                p.info.run = RunState::Running;
            }
        }
    }

    /// Whether `fd` names a PID namespace the caller may join: its own, or one
    /// nested inside it. A task can't move itself outward or sideways.
    pub(super) fn may_join_pid_namespace(&self, cx: &ServiceCtx, target: u32) -> bool {
        self.pids.lock().unwrap().is_within(target, cx.cur.ns.pid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> PidNamespaces {
        let mut t = PidNamespaces::default();
        t.create(1, 0).unwrap();
        t.create(2, 1).unwrap();
        t
    }

    #[test]
    fn a_task_has_a_number_in_each_namespace_above_it() {
        let mut t = table();
        t.add_task(100, 2);
        assert_eq!(t.to_local(100, 2), 1, "PID 1 where it was created");
        assert_eq!(t.to_local(100, 1), 1, "first seen in the parent too");
        assert_eq!(
            t.to_local(100, 0),
            100,
            "the initial namespace uses global ids"
        );
        t.add_task(101, 1);
        assert_eq!(t.to_local(101, 1), 2);
        assert_eq!(t.to_local(101, 2), 0, "not visible from the nested one");
        assert_eq!(t.to_global(2, 1), Some(101));
        assert_eq!(t.to_global(2, 2), None);
    }

    #[test]
    fn the_first_task_is_init_and_its_death_closes_the_namespace() {
        let mut t = table();
        t.add_task(100, 2);
        t.add_task(101, 2);
        assert_eq!(t.init_of(2), Some(100));
        assert!(t.accepts_tasks(2));
        let mut doomed = t.close(2);
        doomed.sort_unstable();
        assert_eq!(doomed, vec![100, 101]);
        assert!(!t.accepts_tasks(2));
        assert!(t.accepts_tasks(0), "the initial namespace never closes");
    }

    #[test]
    fn nesting_is_limited_and_within_follows_the_chain() {
        let mut t = table();
        assert!(t.is_within(2, 1));
        assert!(t.is_within(2, 2));
        assert!(t.is_within(2, 0));
        assert!(!t.is_within(1, 2));
        let mut parent = 2;
        let mut failed = false;
        for id in 3..64 {
            if t.create(id, parent).is_err() {
                failed = true;
                break;
            }
            parent = id;
        }
        assert!(failed);
    }

    #[test]
    fn a_dead_namespace_is_remembered_until_nothing_refers_to_it() {
        let mut t = PidNamespaces::default();
        t.create(1, 0).unwrap();
        t.add_task(100, 1);
        t.close(1);
        t.remove_task(100, 1);
        assert!(
            !t.accepts_tasks(1),
            "still refused after its tasks are gone"
        );
        t.collect_garbage(&[1]);
        assert!(!t.accepts_tasks(1), "a task still names it");
        t.collect_garbage(&[]);
        assert_eq!(t.count(), 0);
    }

    #[test]
    fn a_namespace_nobody_has_entered_yet_survives_collection_while_named() {
        let mut t = PidNamespaces::default();
        t.create(1, 0).unwrap();
        t.collect_garbage(&[1]);
        assert!(t.accepts_tasks(1));
    }
}
