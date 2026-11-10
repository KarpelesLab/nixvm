//! The kernel's side of cgroups (see [`crate::fs::cgroupfs`]): putting new
//! tasks in their cgroup and enforcing `pids.max`, charging CPU time to
//! `cpu.max` and handing tasks their cgroup's `cpu` settings, keeping the
//! frame allocator's memory cgroups in step and killing a task when one is out
//! of memory, the writes that act on tasks
//! (`cgroup.procs`, `cgroup.threads`, `cgroup.kill`, `cgroup.freeze`), the pids
//! a reader of `cgroup.procs` sees, and `/proc/<pid>/cgroup`.
//!
//! The tree's lock is a leaf, taken after `shared` when both are needed.

use super::{Fd, GuestMemory, Kernel, ProcInfo, RunState, ServiceCtx, Shared, err, poll};
use crate::abi::errno::Errno;
use crate::fs::cgroupfs::{CgroupTree, KERNEL_FILES, ROOT, SCHED_FILES, SchedState};

const SIGKILL_BIT: u64 = 1 << (9 - 1);

/// What a write to `memory.max` is to do next (see [`Kernel::memcg_shrink`]).
pub(super) enum Shrink {
    /// The cgroup fits, or there's no one left to kill.
    Done,
    /// A task was killed to make room: try again once it's gone.
    Wait,
    /// The writer itself was picked.
    KillSelf,
}

fn live(p: &ProcInfo) -> bool {
    !matches!(p.run, RunState::Zombie(_))
}

/// Kill `victim` for memory cgroup `slot`'s `memory.max` (and, with
/// `memory.oom.group`, the cgroup asking for that), counting it in
/// `memory.events`. Returns whether the caller is among them.
fn oom_kill(
    sh: &mut Shared,
    tree: &mut CgroupTree,
    cx: &ServiceCtx,
    victim: i32,
    slot: u16,
) -> bool {
    let node = tree.node_of(victim);
    let group = tree.record_oom(node, slot);
    let doomed =
        |pid: i32| pid == victim || group.is_some_and(|g| tree.is_within(tree.node_of(pid), g));
    for p in sh.procs.iter_mut().flatten() {
        if doomed(p.info.pid) && live(&p.info) && p.info.pid != cx.cur.pid {
            p.info.pending |= SIGKILL_BIT;
            p.info.parked = false;
            p.info.frozen = false;
            p.info.run = RunState::Running;
        }
    }
    doomed(cx.cur.pid)
}

/// What the cgroup code needs of a task.
struct TaskState {
    pid: i32,
    tgid: i32,
    zombie: bool,
    cpu_ns: u64,
}

fn tasks_of(sh: &Shared, cx: &ServiceCtx) -> Vec<TaskState> {
    sh.procs
        .iter()
        .flatten()
        .map(|p| &p.info)
        .chain([&cx.cur])
        .map(|i| TaskState {
            pid: i.pid,
            tgid: i.tgid,
            zombie: matches!(i.run, RunState::Zombie(_)),
            cpu_ns: i.cpu_ns as u64,
        })
        .collect()
}

/// Mark the tasks that have exited since the tree last looked: they no longer
/// make their cgroup populated, and their CPU time is the cgroup's to keep.
fn sync_exits(tree: &mut CgroupTree, tasks: &[TaskState]) {
    for tid in tree.live_members() {
        match tasks.iter().find(|t| t.pid == tid) {
            Some(t) if !t.zombie => {}
            Some(t) => tree.exit(tid, t.cpu_ns),
            None => tree.exit(tid, 0),
        }
    }
}

impl Kernel {
    /// Whether a new task forked by the caller may go into `into` (or the
    /// caller's own cgroup): `EAGAIN` past a `pids.max` on the way up. A new
    /// thread goes elsewhere only within the caller's resource domain.
    pub(super) fn cgroup_can_fork(
        &self,
        cx: &ServiceCtx,
        into: Option<u32>,
        thread: bool,
    ) -> Result<(), i64> {
        let mut tree = self.cgroups.lock().unwrap();
        if let Some(node) = into
            && thread
        {
            tree.check_thread_attach(cx.cur.pid, node)
                .map_err(|e| err(Errno(e)))?;
        }
        let node = into.unwrap_or_else(|| tree.node_of(cx.cur.pid));
        tree.can_fork(node, into.is_some())
            .map_err(|e| err(Errno(e)))
    }

    /// Put new task `tid` in its cgroup. Returns how it's to be scheduled.
    pub(super) fn cgroup_fork(&self, cx: &ServiceCtx, tid: i32, into: Option<u32>) -> SchedState {
        let mut tree = self.cgroups.lock().unwrap();
        let node = into.unwrap_or_else(|| tree.node_of(cx.cur.pid));
        tree.attach_new(tid, node);
        tree.sched_state(node, poll::now_ns())
    }

    /// Charge `ns` of CPU time the running task used to the `cpu.max` quotas
    /// of its cgroup and those above. One that runs out holds every task below
    /// it until its next period.
    pub(super) fn cgroup_charge_cpu(&self, cx: &mut ServiceCtx, ns: u64) {
        let now = poll::now_ns();
        let (until, newly) = {
            let mut tree = self.cgroups.lock().unwrap();
            let node = tree.node_of(cx.cur.pid);
            tree.charge_cpu(node, ns, now)
        };
        cx.cur.throttled_until = until;
        if let Some(node) = newly {
            let mut sh = self.shared.lock().unwrap();
            let tree = self.cgroups.lock().unwrap();
            let below = tree.subtree(node);
            for p in sh.procs.iter_mut().flatten() {
                if below.contains(&tree.node_of(p.info.pid)) {
                    p.info.throttled_until = p.info.throttled_until.max(until);
                }
            }
        }
    }

    /// Hand every task its cgroup's current settings (frozen, `cpu.weight`,
    /// `cpu.max`, memory cgroup), after a write that may have changed them,
    /// first giving the frame allocator the memory cgroups and limits there
    /// now are.
    pub(super) fn cgroup_resync(&self, cx: &mut ServiceCtx, mem: &GuestMemory) {
        let mut sh = self.shared.lock().unwrap();
        let mut tree = self.cgroups.lock().unwrap();
        mem.with_allocator(|fa| tree.sync_memcgs(fa));
        let now = poll::now_ns();
        let apply = |p: &mut ProcInfo| {
            let st = tree.sched_state(tree.node_of(p.pid), now);
            p.frozen = st.frozen;
            p.cg_scale = st.scale;
            p.throttled_until = st.throttled_until;
            p.memcg = st.memcg;
        };
        for p in sh.procs.iter_mut().flatten() {
            apply(&mut p.info);
        }
        apply(&mut cx.cur);
    }

    /// Memory cgroup `slot` refused memory at its `memory.max`, while the
    /// caller ran: kill a task in it, as Linux's out-of-memory killer does —
    /// the caller, if it's in there (it asked for the memory), or else another
    /// task of that cgroup — and, with `memory.oom.group`, every task of the
    /// cgroup that asks for it. Returns whether the caller is to die.
    pub(super) fn memcg_oom(&self, cx: &mut ServiceCtx, slot: u16) -> bool {
        let mut sh = self.shared.lock().unwrap();
        let mut tree = self.cgroups.lock().unwrap();
        let Some(domain) = tree.node_of_memcg(slot) else {
            return false;
        };
        let me = tree.node_of(cx.cur.pid);
        let victim = if tree.is_within(me, domain) {
            Some(cx.cur.pid)
        } else {
            sh.procs
                .iter()
                .flatten()
                .map(|p| &p.info)
                .find(|p| live(p) && tree.is_within(tree.node_of(p.pid), domain))
                .map(|p| p.pid)
        };
        victim.is_some_and(|v| oom_kill(&mut sh, &mut tree, cx, v, slot))
    }

    /// After a write to `memory.max` of cgroup `node`: while what it's
    /// charged is over the new limit, kill its tasks, biggest first, as
    /// Linux's `memory_max_write` does (there's no reclaim to try first).
    /// One victim at a time: while one is dying, the writer waits for it.
    pub(super) fn memcg_shrink(&self, cx: &mut ServiceCtx, node: u32, mem: &GuestMemory) -> Shrink {
        let mut sh = self.shared.lock().unwrap();
        let mut tree = self.cgroups.lock().unwrap();
        let Some(slot) = tree.memcg_slot(node) else {
            return Shrink::Done;
        };
        let over = mem.with_allocator(|fa| fa.memcg(slot).is_some_and(|m| m.frames > m.max));
        if !over {
            return Shrink::Done;
        }
        let tasks: Vec<&ProcInfo> = sh
            .procs
            .iter()
            .flatten()
            .map(|p| &p.info)
            .chain([&cx.cur])
            .filter(|p| live(p) && tree.is_within(tree.node_of(p.pid), node))
            .collect();
        if tasks.iter().any(|p| p.pending & SIGKILL_BIT != 0) {
            return Shrink::Wait;
        }
        // Linux's `oom_badness` ranks processes by what they hold; here that's
        // what their address space maps (one that's busy elsewhere counts 0).
        let size = |p: &ProcInfo| {
            if p.mm == cx.cur.mm {
                mem.mapped_bytes()
            } else {
                sh.spaces
                    .get(p.mm)
                    .and_then(|s| s.try_lock().ok().map(|m| m.mapped_bytes()))
                    .unwrap_or(0)
            }
        };
        let Some(victim) = tasks.iter().max_by_key(|p| (size(p), p.pid)).map(|p| p.pid) else {
            tree.record_oom_attempt(slot);
            return Shrink::Done;
        };
        if oom_kill(&mut sh, &mut tree, cx, victim, slot) {
            Shrink::KillSelf
        } else {
            Shrink::Wait
        }
    }

    /// If `fd` is a cgroup file whose writes change how tasks are scheduled,
    /// so [`Self::cgroup_resync`] is due after one: its cgroup and name.
    pub(super) fn cgroup_sched_file(&self, cx: &ServiceCtx, fd: u64) -> Option<(u32, String)> {
        let Some(Fd::File { path, .. }) = cx.cur.fds.get(fd as i32) else {
            return None;
        };
        let path = path.clone();
        match self.vfs_for(cx).cgroup_at(&path) {
            Some((_, node, Some(f))) if SCHED_FILES.contains(&f.as_str()) => Some((node, f)),
            _ => None,
        }
    }

    /// The cgroup a `clone3` `CLONE_INTO_CGROUP` descriptor names.
    pub(super) fn cgroup_of_fd(&self, cx: &ServiceCtx, fd: u64) -> Result<u32, i64> {
        let path = match cx.cur.fds.get(fd as i32) {
            Some(Fd::File { path, .. } | Fd::Dir { path, .. }) => path.clone(),
            _ => return Err(err(Errno::EBADF)),
        };
        match self.vfs_for(cx).cgroup_at(&path) {
            Some((_, node, None)) => Ok(node),
            _ => Err(err(Errno::EBADF)),
        }
    }

    /// Work out what readers of the cgroup files see, from the caller's point
    /// of view: the pids in each cgroup by its numbers, and CPU time used.
    /// Done when the caller opens a file in a `cgroup2` mount.
    pub(super) fn cgroup_refresh(&self, cx: &ServiceCtx, mem: &GuestMemory) {
        let (tasks, threads) = {
            let sh = self.shared.lock().unwrap();
            let threads: Vec<bool> = sh
                .procs
                .iter()
                .flatten()
                .map(|p| p.info.is_thread)
                .chain([cx.cur.is_thread])
                .collect();
            (tasks_of(&sh, cx), threads)
        };
        let mut tree = self.cgroups.lock().unwrap();
        sync_exits(&mut tree, &tasks);
        let mut view = crate::fs::cgroupfs::View::default();
        for (t, _) in tasks.iter().zip(&threads).filter(|(t, _)| !t.zombie) {
            let node = tree.node_of(t.pid);
            let tid = self.visible_pid(cx, t.pid);
            let tgid = self.visible_pid(cx, t.tgid);
            view.threads.entry(node).or_default().push(tid);
            // A process in a threaded subtree is listed in its thread root.
            let procs = view.procs.entry(tree.domain_of(node)).or_default();
            if !procs.contains(&tgid) {
                procs.push(tgid);
            }
            for n in tree.chain(node) {
                *view.cpu_ns.entry(n).or_default() += t.cpu_ns;
            }
        }
        for list in view.procs.values_mut().chain(view.threads.values_mut()) {
            list.sort_unstable();
        }
        for (node, cpu) in &mut view.cpu_ns {
            *cpu += tree.dead_cpu_ns(*node);
        }
        tree.view = view;
        mem.with_allocator(|fa| {
            tree.sync_memcgs(fa);
            tree.refresh_memory(fa);
        });
    }

    /// Before an open: if it's of a file in a `cgroup2` mount, bring what it
    /// will read up to date (see [`Self::cgroup_refresh`]).
    pub(super) fn cgroup_open_hook(
        &self,
        cx: &ServiceCtx,
        sys: crate::abi::arch::Sysno,
        args: &[u64; 6],
        mem: &GuestMemory,
    ) {
        use crate::abi::arch::Sysno;
        const AT_FDCWD: i64 = -100;
        let (dirfd, ptr) = match sys {
            Sysno::Open => (AT_FDCWD, args[0]),
            Sysno::Openat | Sysno::Openat2 => (args[0] as i64, args[1]),
            _ => return,
        };
        let Some(raw) = super::read_path(mem, ptr) else {
            return;
        };
        let abs = self.resolve_path(cx, dirfd, &raw);
        let in_cgroup = self.vfs_for(cx).cgroup_at(&abs).is_some();
        if in_cgroup {
            self.cgroup_refresh(cx, mem);
        }
    }

    /// `/proc/<pid>/cgroup` for task `tid`, as the caller sees it: its path
    /// below the caller's cgroup namespace root.
    pub(super) fn cgroup_line(&self, cx: &ServiceCtx, tid: i32) -> String {
        let tree = self.cgroups.lock().unwrap();
        let root = tree.ns_root(cx.cur.ns.cgroup);
        format!("0::{}\n", tree.path_from(root, tree.node_of(tid)))
    }

    /// A write to `cgroup.procs`, `cgroup.threads`, `cgroup.kill` or
    /// `cgroup.freeze`, which act on tasks; `None` for any other write.
    pub(super) fn cgroup_control_write(
        &self,
        cx: &mut ServiceCtx,
        fd: u64,
        buf: u64,
        count: u64,
        mem: &GuestMemory,
    ) -> Option<i64> {
        let Some(Fd::File { path, .. }) = cx.cur.fds.get(fd as i32) else {
            return None;
        };
        let path = path.clone();
        let (_, node, file) = self.vfs_for(cx).cgroup_at(&path)?;
        let file = file?;
        if !KERNEL_FILES.contains(&file.as_str()) {
            return None;
        }
        let Ok(bytes) = mem.read_vec(buf, count.min(4096) as usize) else {
            return Some(err(Errno::EFAULT));
        };
        let text = String::from_utf8_lossy(&bytes).trim().to_string();
        let result = match file.as_str() {
            "cgroup.procs" | "cgroup.threads" => {
                self.cgroup_move(cx, node, &text, file == "cgroup.threads")
            }
            "cgroup.kill" => self.cgroup_kill(cx, node, &text),
            _ => self.cgroup_freeze(cx, node, &text),
        };
        Some(result.map_or_else(|e| e, |()| count as i64))
    }

    /// Move the process (or, for `cgroup.threads`, the thread) whose id is
    /// `text` (0 for the caller) into cgroup `node`.
    fn cgroup_move(
        &self,
        cx: &mut ServiceCtx,
        node: u32,
        text: &str,
        thread: bool,
    ) -> Result<(), i64> {
        let local: i32 = text.parse().map_err(|_| err(Errno::EINVAL))?;
        let target = if local == 0 {
            cx.cur.pid
        } else {
            self.global_pid(cx, local)
                .ok_or_else(|| err(Errno::ESRCH))?
        };
        let mut sh = self.shared.lock().unwrap();
        let tasks = tasks_of(&sh, cx);
        let Some(task) = tasks.iter().find(|t| t.pid == target && !t.zombie) else {
            return Err(err(Errno::ESRCH));
        };
        let tgid = task.tgid;
        let mut tree = self.cgroups.lock().unwrap();
        sync_exits(&mut tree, &tasks);
        let from = tree.node_of(target);
        // Both ends must be in the writer's cgroup namespace.
        let root = tree.ns_root(cx.cur.ns.cgroup);
        if !tree.is_within(node, root) || !tree.is_within(from, root) {
            return Err(err(Errno::ENOENT));
        }
        if thread {
            // A thread moves on its own only within its resource domain.
            tree.check_thread_attach(target, node)
                .map_err(|e| err(Errno(e)))?;
        } else {
            tree.check_attach(node).map_err(|e| err(Errno(e)))?;
        }
        // The writer needs the right to move tasks out of where they are too:
        // write access to the common ancestor's `cgroup.procs` (cgroup v2's
        // delegation rule), which a delegated cgroup's owner has below it.
        let ancestor = tree.common_ancestor(from, node);
        if !cx.cur.privileged() && tree.owner(ancestor).0 != cx.cur.creds.euid {
            return Err(err(Errno::EACCES));
        }
        let moving: Vec<(i32, bool)> = tasks
            .iter()
            .filter(|t| {
                if thread {
                    t.pid == target
                } else {
                    t.tgid == tgid
                }
            })
            .map(|t| (t.pid, !t.zombie))
            .collect();
        tree.migrate(&moving, node);
        let st = tree.sched_state(node, poll::now_ns());
        drop(tree);
        let apply = |p: &mut ProcInfo| {
            if moving.iter().any(|&(t, _)| t == p.pid) {
                p.frozen = st.frozen;
                p.cg_scale = st.scale;
                p.throttled_until = st.throttled_until;
                p.memcg = st.memcg;
            }
        };
        for p in sh.procs.iter_mut().flatten() {
            apply(&mut p.info);
        }
        apply(&mut cx.cur);
        Ok(())
    }

    /// `cgroup.kill`: `1` kills every task in the cgroup and below.
    fn cgroup_kill(&self, cx: &mut ServiceCtx, node: u32, text: &str) -> Result<(), i64> {
        if text != "1" {
            return Err(err(Errno::EINVAL));
        }
        let mut sh = self.shared.lock().unwrap();
        let tree = self.cgroups.lock().unwrap();
        // Killing is per process, so not something a threaded cgroup does.
        if tree.is_threaded(node) {
            return Err(err(Errno::EOPNOTSUPP));
        }
        let doomed = tree.subtree(node);
        let hit = |tid: i32| doomed.contains(&tree.node_of(tid));
        for p in sh.procs.iter_mut().flatten() {
            if hit(p.info.pid) && !matches!(p.info.run, RunState::Zombie(_)) {
                p.info.pending |= SIGKILL_BIT;
                p.info.parked = false;
                p.info.run = RunState::Running;
            }
        }
        if hit(cx.cur.pid) {
            cx.cur.pending |= SIGKILL_BIT;
        }
        Ok(())
    }

    /// `cgroup.freeze`: `1` stops every task in the cgroup and below from
    /// running until it's `0` again (and no cgroup above is frozen).
    fn cgroup_freeze(&self, cx: &mut ServiceCtx, node: u32, text: &str) -> Result<(), i64> {
        let on = match text {
            "0" => false,
            "1" => true,
            _ => return Err(err(Errno::EINVAL)),
        };
        let mut sh = self.shared.lock().unwrap();
        let mut tree = self.cgroups.lock().unwrap();
        if !tree.set_freeze(node, on) {
            return Ok(());
        }
        let below = tree.subtree(node);
        for p in sh.procs.iter_mut().flatten() {
            let n = tree.node_of(p.info.pid);
            if below.contains(&n) {
                p.info.frozen = tree.frozen(n);
                p.info.parked = false;
            }
        }
        let n = tree.node_of(cx.cur.pid);
        if below.contains(&n) {
            cx.cur.frozen = tree.frozen(n);
        }
        Ok(())
    }

    /// The root a new `cgroup2` mount shows: the mounter's namespace's.
    pub(super) fn cgroup_mount_root(&self, cx: &ServiceCtx) -> u32 {
        let tree = self.cgroups.lock().unwrap();
        let root = tree.ns_root(cx.cur.ns.cgroup);
        if tree.exists(root) { root } else { ROOT }
    }
}
