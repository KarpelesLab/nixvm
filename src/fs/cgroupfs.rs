//! cgroup v2: the control group hierarchy and the `cgroup2` filesystem that
//! shows it.
//!
//! [`CgroupTree`] is the VM's one hierarchy, shared by every `cgroup2` mount
//! ([`CgroupFs`]) and the kernel. A mount shows the tree from a node down: the
//! root for the first cgroup namespace, a namespace's own root otherwise.
//!
//! Only controllers nixvm enforces are offered. That is `pids` (a `pids.max`
//! that `fork` and `clone` honour), next to the core files every cgroup has:
//! moving processes (`cgroup.procs`), `cgroup.kill`, `cgroup.freeze`, the
//! `cgroup.max.*` limits and `cpu.stat`. Writes that act on tasks (moving,
//! killing, freezing) need the writer's identity and the task table, so the
//! kernel takes them before they reach [`CgroupFs::write_at`]; the rest are
//! handled here.
//!
//! The tree's lock is a leaf: nothing is locked while holding it.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io;
use std::sync::{Arc, Mutex};

use super::{Attrs, DirEntry, MountFs, NodeKind};

/// The root cgroup's id.
pub const ROOT: u32 = 0;

/// The controllers there are, as `cgroup.controllers` lists them.
const CONTROLLERS: &[&str] = &["pids"];

/// Files every cgroup has, and whether a write is accepted.
const CORE_FILES: &[(&str, bool)] = &[
    ("cgroup.controllers", false),
    ("cgroup.events", false),
    ("cgroup.freeze", true),
    ("cgroup.kill", true),
    ("cgroup.max.depth", true),
    ("cgroup.max.descendants", true),
    ("cgroup.procs", true),
    ("cgroup.stat", false),
    ("cgroup.subtree_control", true),
    ("cgroup.threads", true),
    ("cgroup.type", true),
    ("cpu.stat", false),
];

/// Core files the root doesn't have.
const NOT_AT_ROOT: &[&str] = &[
    "cgroup.events",
    "cgroup.freeze",
    "cgroup.kill",
    "cgroup.type",
];

/// The `pids` controller's files, in a cgroup whose parent enables it.
const PIDS_FILES: &[(&str, bool)] = &[
    ("pids.current", false),
    ("pids.events", false),
    ("pids.max", true),
];

/// Writes the kernel handles, since they act on tasks.
pub const KERNEL_FILES: &[&str] = &[
    "cgroup.procs",
    "cgroup.threads",
    "cgroup.kill",
    "cgroup.freeze",
];

fn errno(e: i32) -> io::Error {
    io::Error::from_raw_os_error(e)
}
const EPERM: i32 = 1;
const ENOENT: i32 = 2;
const EAGAIN: i32 = 11;
const EACCES: i32 = 13;
const EBUSY: i32 = 16;
const EEXIST: i32 = 17;
const ENOTDIR: i32 = 20;
const EINVAL: i32 = 22;
const EOPNOTSUPP: i32 = 95;

/// One cgroup.
#[derive(Debug, Clone)]
struct Node {
    name: String,
    parent: Option<u32>,
    children: BTreeMap<String, u32>,
    /// `cgroup.subtree_control` has `pids`.
    pids_below: bool,
    /// `pids.max`; `None` is "max".
    pids_max: Option<u64>,
    /// How often a fork failed on this cgroup's `pids.max` (`pids.events`).
    pids_events: u64,
    max_depth: Option<u64>,
    max_descendants: Option<u64>,
    /// `cgroup.freeze` was set here (a task is frozen if any ancestor's is).
    freeze: bool,
    /// Tasks in this cgroup and below, zombies included (`pids.current`).
    tasks: u64,
    /// Tasks in this cgroup and below that haven't exited (`populated`).
    live: u64,
    /// CPU time of the tasks that have exited in this cgroup and below.
    dead_cpu_ns: u64,
    uid: u32,
    gid: u32,
    mtime: i64,
}

impl Node {
    fn new(name: &str, parent: Option<u32>, uid: u32, gid: u32) -> Self {
        Self {
            name: name.to_string(),
            parent,
            children: BTreeMap::new(),
            pids_below: false,
            pids_max: None,
            pids_events: 0,
            max_depth: None,
            max_descendants: None,
            freeze: false,
            tasks: 0,
            live: 0,
            dead_cpu_ns: 0,
            uid,
            gid,
            mtime: now(),
        }
    }
}

fn now() -> i64 {
    crate::clock::now_unix().as_secs() as i64
}

/// A task with a cgroup other than the root.
#[derive(Debug, Clone, Copy)]
struct Member {
    node: u32,
    /// Hasn't exited yet.
    live: bool,
}

/// What the kernel last worked out for a reader: the tasks in each cgroup by
/// the numbers the reader knows them by, and CPU time used.
#[derive(Debug, Clone, Default)]
pub struct View {
    /// Processes (thread group ids) directly in each cgroup.
    pub procs: BTreeMap<u32, Vec<i32>>,
    /// Threads directly in each cgroup.
    pub threads: BTreeMap<u32, Vec<i32>>,
    /// CPU time of each cgroup and below, live tasks and exited ones.
    pub cpu_ns: BTreeMap<u32, u64>,
}

/// The VM's cgroup hierarchy.
#[derive(Debug)]
pub struct CgroupTree {
    nodes: BTreeMap<u32, Node>,
    next_id: u32,
    /// Every task outside the root cgroup, by global thread id.
    members: BTreeMap<i32, Member>,
    /// Each cgroup namespace's root (the first namespace's is [`ROOT`]).
    ns_roots: BTreeMap<u32, u32>,
    /// What [`CgroupFs`] renders (see [`View`]).
    pub view: View,
}

impl Default for CgroupTree {
    fn default() -> Self {
        let mut nodes = BTreeMap::new();
        nodes.insert(ROOT, Node::new("", None, 0, 0));
        Self {
            nodes,
            next_id: 1,
            members: BTreeMap::new(),
            ns_roots: BTreeMap::new(),
            view: View::default(),
        }
    }
}

/// The tree as the kernel and every mount share it.
pub type SharedTree = Arc<Mutex<CgroupTree>>;

impl CgroupTree {
    /// A new hierarchy holding only the root.
    #[must_use]
    pub fn shared() -> SharedTree {
        Arc::new(Mutex::new(Self::default()))
    }

    /// The cgroup task `tid` is in.
    #[must_use]
    pub fn node_of(&self, tid: i32) -> u32 {
        self.members.get(&tid).map_or(ROOT, |m| m.node)
    }

    /// Whether cgroup `node` still exists.
    #[must_use]
    pub fn exists(&self, node: u32) -> bool {
        self.nodes.contains_key(&node)
    }

    /// `node`, then its ancestors up to the root.
    #[must_use]
    pub fn chain(&self, node: u32) -> Vec<u32> {
        let mut out = Vec::new();
        let mut at = Some(node);
        while let Some(n) = at {
            out.push(n);
            at = self.nodes.get(&n).and_then(|x| x.parent);
        }
        out
    }

    /// Whether `inner` is `outer` or below it.
    #[must_use]
    pub fn is_within(&self, inner: u32, outer: u32) -> bool {
        self.chain(inner).contains(&outer)
    }

    /// The nearest cgroup both `a` and `b` are within.
    #[must_use]
    pub fn common_ancestor(&self, a: u32, b: u32) -> u32 {
        let up = self.chain(b);
        self.chain(a)
            .into_iter()
            .find(|n| up.contains(n))
            .unwrap_or(ROOT)
    }

    /// Whether tasks in `node` are frozen: it, or a cgroup above it, is.
    #[must_use]
    pub fn frozen(&self, node: u32) -> bool {
        self.chain(node)
            .iter()
            .any(|n| self.nodes.get(n).is_some_and(|x| x.freeze))
    }

    /// Who may write `node`'s files: its owner (a delegated cgroup), or root.
    #[must_use]
    pub fn owner(&self, node: u32) -> (u32, u32) {
        self.nodes.get(&node).map_or((0, 0), |n| (n.uid, n.gid))
    }

    /// Whether a task can be put in `node`: one whose `cgroup.subtree_control`
    /// enables a controller has children for its tasks, not tasks of its own
    /// (cgroup v2's "no internal processes" rule; the root is exempt).
    pub fn check_attach(&self, node: u32) -> Result<(), i32> {
        match self.nodes.get(&node) {
            None => Err(ENOENT),
            Some(n) if node != ROOT && n.pids_below => Err(EBUSY),
            Some(_) => Ok(()),
        }
    }

    /// Whether a new task may go into `node` (`explicit`: asked for by
    /// `CLONE_INTO_CGROUP`, rather than inherited). Fails `EAGAIN` past a
    /// `pids.max` on the way up, counting it in that cgroup's `pids.events`.
    pub fn can_fork(&mut self, node: u32, explicit: bool) -> Result<(), i32> {
        if explicit {
            self.check_attach(node)?;
        }
        for n in self.chain(node) {
            let x = self.nodes.get_mut(&n).ok_or(ENOENT)?;
            if x.pids_max.is_some_and(|max| x.tasks + 1 > max) {
                x.pids_events += 1;
                return Err(EAGAIN);
            }
        }
        Ok(())
    }

    /// Put new task `tid` in `node`, which [`Self::can_fork`] allowed.
    /// Returns whether it starts frozen.
    pub fn attach_new(&mut self, tid: i32, node: u32) -> bool {
        let chain = self.chain(node);
        self.charge(&chain, 1, 1);
        if node != ROOT {
            self.members.insert(tid, Member { node, live: true });
        }
        self.frozen(node)
    }

    /// [`Self::can_fork`] then [`Self::attach_new`], for task `tid` forked by
    /// `parent`, into `into` or the parent's cgroup.
    pub fn fork(&mut self, parent: i32, tid: i32, into: Option<u32>) -> Result<bool, i32> {
        let node = into.unwrap_or_else(|| self.node_of(parent));
        self.can_fork(node, into.is_some())?;
        Ok(self.attach_new(tid, node))
    }

    /// Add `tasks` and `live` to each cgroup in `chain` (negative to remove).
    fn charge(&mut self, chain: &[u32], tasks: i64, live: i64) {
        for n in chain {
            if let Some(x) = self.nodes.get_mut(n) {
                x.tasks = x.tasks.saturating_add_signed(tasks);
                x.live = x.live.saturating_add_signed(live);
            }
        }
    }

    /// Task `tid` has exited, after using `cpu_ns` of CPU time: it no longer
    /// makes its cgroup populated, though `pids.current` counts it until it's
    /// reaped. Does nothing for a task already marked.
    pub fn exit(&mut self, tid: i32, cpu_ns: u64) {
        let node = match self.members.get_mut(&tid) {
            Some(m) if m.live => {
                m.live = false;
                m.node
            }
            Some(_) => return,
            None => ROOT,
        };
        let chain = self.chain(node);
        self.charge(&chain, 0, -1);
        for n in chain {
            if let Some(x) = self.nodes.get_mut(&n) {
                x.dead_cpu_ns = x.dead_cpu_ns.saturating_add(cpu_ns);
            }
        }
    }

    /// Task `tid` is gone for good (reaped). `cpu_ns` is used if it wasn't
    /// marked as exited first.
    pub fn release(&mut self, tid: i32, cpu_ns: u64) {
        if self.members.get(&tid).is_some_and(|m| m.live) {
            self.exit(tid, cpu_ns);
        }
        let node = self.members.remove(&tid).map_or(ROOT, |m| m.node);
        let chain = self.chain(node);
        self.charge(&chain, -1, 0);
    }

    /// Move each of `tids` (with whether it's still alive) to `node`. Returns
    /// whether tasks there are frozen. `pids.max` doesn't stop a move, as on
    /// Linux; it only stops new tasks.
    pub fn migrate(&mut self, tids: &[(i32, bool)], node: u32) -> bool {
        for &(tid, alive) in tids {
            let from = self.node_of(tid);
            if from == node {
                continue;
            }
            let live = i64::from(alive && self.members.get(&tid).is_none_or(|m| m.live));
            let old = self.chain(from);
            self.charge(&old, -1, -live);
            let new = self.chain(node);
            self.charge(&new, 1, live);
            if node == ROOT {
                self.members.remove(&tid);
            } else {
                self.members.insert(
                    tid,
                    Member {
                        node,
                        live: live != 0,
                    },
                );
            }
        }
        self.frozen(node)
    }

    /// Every cgroup from `node` down.
    #[must_use]
    pub fn subtree(&self, node: u32) -> Vec<u32> {
        let mut out = vec![node];
        let mut i = 0;
        while i < out.len() {
            if let Some(n) = self.nodes.get(&out[i]) {
                out.extend(n.children.values().copied());
            }
            i += 1;
        }
        out
    }

    /// Set `node`'s `cgroup.freeze`. Returns whether that changed anything.
    pub fn set_freeze(&mut self, node: u32, on: bool) -> bool {
        match self.nodes.get_mut(&node) {
            Some(n) if n.freeze != on => {
                n.freeze = on;
                true
            }
            _ => false,
        }
    }

    /// Exited CPU time of `node` and below.
    #[must_use]
    pub fn dead_cpu_ns(&self, node: u32) -> u64 {
        self.nodes.get(&node).map_or(0, |n| n.dead_cpu_ns)
    }

    /// Each live task outside the root, with its cgroup.
    #[must_use]
    pub fn live_members(&self) -> Vec<i32> {
        self.members
            .iter()
            .filter(|(_, m)| m.live)
            .map(|(&t, _)| t)
            .collect()
    }

    /// Cgroup namespace `ns`'s root.
    #[must_use]
    pub fn ns_root(&self, ns: u32) -> u32 {
        self.ns_roots
            .get(&ns)
            .copied()
            .filter(|n| self.nodes.contains_key(n))
            .unwrap_or(ROOT)
    }

    /// Record a new cgroup namespace rooted at `node`.
    pub fn set_ns_root(&mut self, ns: u32, node: u32) {
        self.ns_roots.insert(ns, node);
    }

    /// `node`'s path below `root`, as `/proc/<pid>/cgroup` shows it: `/a/b`, or
    /// with `..` steps when `node` is outside `root`, as Linux shows it.
    #[must_use]
    pub fn path_from(&self, root: u32, node: u32) -> String {
        let up = self.chain(root);
        let mut down = Vec::new();
        let mut at = node;
        while !up.contains(&at) {
            let Some(n) = self.nodes.get(&at) else {
                break;
            };
            down.push(n.name.clone());
            at = n.parent.unwrap_or(ROOT);
        }
        let common = at;
        let mut parts: Vec<String> = up
            .iter()
            .take_while(|&&n| n != common)
            .map(|_| "..".to_string())
            .collect();
        parts.extend(down.into_iter().rev());
        format!("/{}", parts.join("/"))
    }

    /// The cgroup at `rel` below `root`, and the file named, if any.
    fn lookup(&self, root: u32, rel: &str) -> Option<(u32, Option<String>)> {
        let mut node = root;
        let mut parts = rel.split('/').filter(|p| !p.is_empty()).peekable();
        while let Some(part) = parts.next() {
            let n = self.nodes.get(&node)?;
            if let Some(&child) = n.children.get(part) {
                node = child;
                continue;
            }
            // A file: the last component, and one this cgroup has.
            if parts.peek().is_none() && self.files(node).iter().any(|(f, _)| *f == part) {
                return Some((node, Some(part.to_string())));
            }
            return None;
        }
        Some((node, None))
    }

    /// The interface files of `node`, and which take writes.
    fn files(&self, node: u32) -> Vec<(&'static str, bool)> {
        let mut out: Vec<(&'static str, bool)> = CORE_FILES
            .iter()
            .filter(|(f, _)| node != ROOT || !NOT_AT_ROOT.contains(f))
            .copied()
            .collect();
        let parent = self.nodes.get(&node).and_then(|n| n.parent);
        if parent
            .and_then(|p| self.nodes.get(&p))
            .is_some_and(|p| p.pids_below)
        {
            out.extend_from_slice(PIDS_FILES);
        }
        out.sort_unstable();
        out
    }

    /// The controllers `node` may enable for its children.
    fn controllers(&self, node: u32) -> Vec<&'static str> {
        match self.nodes.get(&node).and_then(|n| n.parent) {
            None => CONTROLLERS.to_vec(),
            Some(p) if self.nodes.get(&p).is_some_and(|p| p.pids_below) => vec!["pids"],
            Some(_) => Vec::new(),
        }
    }

    fn depth(&self, node: u32) -> u64 {
        self.chain(node).len() as u64 - 1
    }

    fn descendants(&self, node: u32) -> u64 {
        self.subtree(node).len() as u64 - 1
    }

    /// `mkdir`: a new cgroup `name` in `parent`, owned by `uid`/`gid`.
    pub fn mkdir(&mut self, parent: u32, name: &str, uid: u32, gid: u32) -> Result<u32, i32> {
        let Some(p) = self.nodes.get(&parent) else {
            return Err(ENOENT);
        };
        if p.children.contains_key(name) || self.files(parent).iter().any(|(f, _)| *f == name) {
            return Err(EEXIST);
        }
        if name.contains('\n') || name == "." || name == ".." {
            return Err(EINVAL);
        }
        // Every ancestor's limits on depth and number of descendants.
        let new_depth = self.depth(parent) + 1;
        for n in self.chain(parent) {
            let x = &self.nodes[&n];
            let below = new_depth - self.depth(n);
            if x.max_depth.is_some_and(|d| below > d)
                || x.max_descendants
                    .is_some_and(|d| self.descendants(n) + 1 > d)
            {
                return Err(EAGAIN);
            }
        }
        let id = self.next_id;
        self.next_id += 1;
        self.nodes
            .insert(id, Node::new(name, Some(parent), uid, gid));
        let p = self.nodes.get_mut(&parent).expect("checked");
        p.children.insert(name.to_string(), id);
        p.mtime = now();
        Ok(id)
    }

    /// `rmdir`: remove cgroup `node`, which must have no children and no live
    /// tasks.
    pub fn rmdir(&mut self, node: u32) -> Result<(), i32> {
        let Some(n) = self.nodes.get(&node) else {
            return Err(ENOENT);
        };
        let Some(parent) = n.parent else {
            return Err(EBUSY);
        };
        if !n.children.is_empty() || n.live > 0 {
            return Err(EBUSY);
        }
        let name = n.name.clone();
        // Zombies still charged here move up, as their cgroup goes away.
        let zombies: Vec<i32> = self
            .members
            .iter()
            .filter(|(_, m)| m.node == node)
            .map(|(&t, _)| t)
            .collect();
        for t in zombies {
            if let Some(m) = self.members.get_mut(&t) {
                m.node = parent;
            }
        }
        let tasks = n.tasks;
        self.nodes.remove(&node);
        if let Some(p) = self.nodes.get_mut(&parent) {
            p.children.remove(&name);
            p.mtime = now();
            // Its zombies are now directly in the parent, which counted them
            // already; nothing to re-charge.
            let _ = tasks;
        }
        self.ns_roots.retain(|_, r| *r != node);
        Ok(())
    }

    /// A write to one of the files handled here.
    fn write_file(&mut self, node: u32, file: &str, text: &str) -> Result<(), i32> {
        let text = text.trim();
        let limit = |t: &str| -> Result<Option<u64>, i32> {
            if t == "max" {
                Ok(None)
            } else {
                t.parse().map(Some).map_err(|_| EINVAL)
            }
        };
        match file {
            "cgroup.subtree_control" => self.write_subtree_control(node, text),
            "cgroup.max.depth" => {
                let v = limit(text)?;
                self.nodes.get_mut(&node).ok_or(ENOENT)?.max_depth = v;
                Ok(())
            }
            "cgroup.max.descendants" => {
                let v = limit(text)?;
                self.nodes.get_mut(&node).ok_or(ENOENT)?.max_descendants = v;
                Ok(())
            }
            "pids.max" => {
                let v = limit(text)?;
                self.nodes.get_mut(&node).ok_or(ENOENT)?.pids_max = v;
                Ok(())
            }
            // Threaded subtrees aren't supported; "domain" isn't something to
            // write, as on Linux.
            "cgroup.type" if text == "threaded" => Err(EOPNOTSUPP),
            "cgroup.type" => Err(EINVAL),
            f if KERNEL_FILES.contains(&f) => Err(EOPNOTSUPP),
            _ => Err(EACCES),
        }
    }

    /// `+pids`/`-pids`, space separated, all or nothing.
    fn write_subtree_control(&mut self, node: u32, text: &str) -> Result<(), i32> {
        let available = self.controllers(node);
        let mut enable = None;
        for word in text.split_whitespace() {
            let (on, name) = match word.split_at(1) {
                ("+", n) => (true, n),
                ("-", n) => (false, n),
                _ => return Err(EINVAL),
            };
            if !available.contains(&name) {
                return Err(ENOENT);
            }
            enable = Some(on);
        }
        let Some(on) = enable else {
            return Ok(());
        };
        let n = &self.nodes[&node];
        if on == n.pids_below {
            return Ok(());
        }
        if on {
            // No internal processes: a non-root cgroup with tasks of its own
            // can't hand controllers to children.
            let own = self.members.values().any(|m| m.node == node && m.live);
            if node != ROOT && own {
                return Err(EBUSY);
            }
        } else if n
            .children
            .values()
            .any(|c| self.nodes.get(c).is_some_and(|c| c.pids_below))
        {
            // A child still hands it on.
            return Err(EBUSY);
        }
        self.nodes.get_mut(&node).expect("checked").pids_below = on;
        Ok(())
    }

    /// The contents of `node`'s `file`.
    fn render(&self, node: u32, file: &str) -> String {
        let n = &self.nodes[&node];
        let list = |m: &BTreeMap<u32, Vec<i32>>| {
            m.get(&node)
                .map(|v| {
                    v.iter().fold(String::new(), |mut out, p| {
                        let _ = writeln!(out, "{p}");
                        out
                    })
                })
                .unwrap_or_default()
        };
        let limit = |v: Option<u64>| v.map_or_else(|| "max\n".to_string(), |v| format!("{v}\n"));
        match file {
            "cgroup.controllers" => format!("{}\n", self.controllers(node).join(" ")),
            "cgroup.subtree_control" => {
                format!("{}\n", if n.pids_below { "pids" } else { "" })
            }
            "cgroup.procs" => list(&self.view.procs),
            "cgroup.threads" => list(&self.view.threads),
            "cgroup.events" => format!(
                "populated {}\nfrozen {}\n",
                u8::from(n.live > 0),
                u8::from(self.frozen(node))
            ),
            "cgroup.type" => "domain\n".to_string(),
            "cgroup.max.depth" => limit(n.max_depth),
            "cgroup.max.descendants" => limit(n.max_descendants),
            "cgroup.stat" => format!(
                "nr_descendants {}\nnr_dying_descendants 0\n",
                self.descendants(node)
            ),
            "cgroup.freeze" => format!("{}\n", u8::from(n.freeze)),
            "cpu.stat" => {
                let usec = self
                    .view
                    .cpu_ns
                    .get(&node)
                    .copied()
                    .unwrap_or(n.dead_cpu_ns)
                    / 1000;
                format!("usage_usec {usec}\nuser_usec {usec}\nsystem_usec 0\n")
            }
            "pids.max" => limit(n.pids_max),
            "pids.current" => format!("{}\n", n.tasks),
            "pids.events" => format!("max {}\n", n.pids_events),
            _ => String::new(),
        }
    }

    fn inode(node: u32, file: Option<&str>) -> u64 {
        const BASE: u64 = 0x6367_0000_0000;
        let slot = file.map_or(0, |f| {
            CORE_FILES
                .iter()
                .chain(PIDS_FILES)
                .position(|(n, _)| *n == f)
                .map_or(0, |i| i as u64 + 1)
        });
        BASE + (u64::from(node) << 8) + slot
    }
}

/// A `cgroup2` mount: the hierarchy from one cgroup down.
#[derive(Debug)]
pub struct CgroupFs {
    tree: SharedTree,
    root: u32,
}

impl CgroupFs {
    /// Show `tree` from cgroup `root` down.
    #[must_use]
    pub fn new(tree: SharedTree, root: u32) -> Self {
        Self { tree, root }
    }

    /// The hierarchy this shows.
    #[must_use]
    pub fn tree(&self) -> &SharedTree {
        &self.tree
    }

    /// The cgroup at `rel`, and the file named in it, if any.
    #[must_use]
    pub fn resolve(&self, rel: &str) -> Option<(u32, Option<String>)> {
        self.tree.lock().unwrap().lookup(self.root, rel)
    }
}

/// `rel`'s last component and the directory holding it.
fn split_last(rel: &str) -> (&str, &str) {
    let rel = rel.trim_matches('/');
    rel.rsplit_once('/').unwrap_or(("", rel))
}

impl MountFs for CgroupFs {
    fn stat(&mut self, rel: &str) -> Option<Attrs> {
        let tree = self.tree.lock().unwrap();
        let (node, file) = tree.lookup(self.root, rel)?;
        let n = &tree.nodes[&node];
        Some(match file {
            None => Attrs {
                kind: NodeKind::Dir,
                size: 0,
                mode: 0o040_755,
                uid: n.uid,
                gid: n.gid,
                atime: n.mtime,
                mtime: n.mtime,
                inode: CgroupTree::inode(node, None),
                nlink: 2 + n.children.len() as u32,
                rdev: 0,
            },
            Some(f) => {
                let writable = tree.files(node).iter().any(|(x, w)| *x == f && *w);
                let mode = match f.as_str() {
                    "cgroup.kill" => 0o100_200,
                    _ if writable => 0o100_644,
                    _ => 0o100_444,
                };
                // The files a delegation hands over share the directory's owner.
                let delegated = matches!(
                    f.as_str(),
                    "cgroup.procs" | "cgroup.threads" | "cgroup.subtree_control"
                );
                let (uid, gid) = if delegated { (n.uid, n.gid) } else { (0, 0) };
                Attrs {
                    kind: NodeKind::File,
                    size: 0,
                    mode,
                    uid,
                    gid,
                    atime: n.mtime,
                    mtime: n.mtime,
                    inode: CgroupTree::inode(node, Some(&f)),
                    nlink: 1,
                    rdev: 0,
                }
            }
        })
    }

    fn read_at(&mut self, rel: &str, off: u64, buf: &mut [u8]) -> io::Result<usize> {
        let tree = self.tree.lock().unwrap();
        let (node, file) = tree.lookup(self.root, rel).ok_or_else(|| errno(ENOENT))?;
        let Some(file) = file else {
            return Err(errno(21)); // EISDIR
        };
        let text = tree.render(node, &file);
        let bytes = text.as_bytes();
        let start = (off as usize).min(bytes.len());
        let n = buf.len().min(bytes.len() - start);
        buf[..n].copy_from_slice(&bytes[start..start + n]);
        Ok(n)
    }

    fn readdir(&mut self, rel: &str) -> io::Result<Vec<DirEntry>> {
        let tree = self.tree.lock().unwrap();
        let (node, file) = tree.lookup(self.root, rel).ok_or_else(|| errno(ENOENT))?;
        if file.is_some() {
            return Err(errno(ENOTDIR));
        }
        let mut out: Vec<DirEntry> = tree
            .files(node)
            .into_iter()
            .map(|(f, _)| DirEntry {
                name: f.to_string(),
                kind: NodeKind::File,
                inode: CgroupTree::inode(node, Some(f)),
            })
            .collect();
        for (name, &child) in &tree.nodes[&node].children {
            out.push(DirEntry {
                name: name.clone(),
                kind: NodeKind::Dir,
                inode: CgroupTree::inode(child, None),
            });
        }
        Ok(out)
    }

    fn fs_type(&self) -> &'static str {
        "cgroup2"
    }

    fn read_only(&self) -> bool {
        false
    }

    fn write_at(&mut self, rel: &str, _off: u64, buf: &[u8]) -> io::Result<usize> {
        let mut tree = self.tree.lock().unwrap();
        let (node, file) = tree.lookup(self.root, rel).ok_or_else(|| errno(ENOENT))?;
        let Some(file) = file else {
            return Err(errno(21)); // EISDIR
        };
        let text = std::str::from_utf8(buf).map_err(|_| errno(EINVAL))?;
        tree.write_file(node, &file, text).map_err(errno)?;
        Ok(buf.len())
    }

    // A control file is never cut short.
    fn truncate(&mut self, rel: &str, _len: u64) -> io::Result<()> {
        self.stat(rel).map(|_| ()).ok_or_else(|| errno(ENOENT))
    }

    fn create(&mut self, rel: &str, _mode: u32) -> io::Result<()> {
        if self.stat(rel).is_some() {
            return Ok(());
        }
        Err(errno(EPERM))
    }

    fn mkdir(&mut self, rel: &str, _mode: u32) -> io::Result<()> {
        let (dir, name) = split_last(rel);
        let mut tree = self.tree.lock().unwrap();
        let (parent, file) = tree.lookup(self.root, dir).ok_or_else(|| errno(ENOENT))?;
        if file.is_some() {
            return Err(errno(ENOTDIR));
        }
        tree.mkdir(parent, name, 0, 0).map(|_| ()).map_err(errno)
    }

    fn rmdir(&mut self, rel: &str) -> io::Result<()> {
        let mut tree = self.tree.lock().unwrap();
        let (node, file) = tree.lookup(self.root, rel).ok_or_else(|| errno(ENOENT))?;
        if file.is_some() {
            return Err(errno(ENOTDIR));
        }
        if node == self.root {
            return Err(errno(EBUSY));
        }
        tree.rmdir(node).map_err(errno)
    }

    fn unlink(&mut self, rel: &str) -> io::Result<()> {
        match self.stat(rel) {
            Some(a) if a.kind == NodeKind::Dir => Err(errno(21)), // EISDIR
            Some(_) => Err(errno(EPERM)),
            None => Err(errno(ENOENT)),
        }
    }

    fn rename(&mut self, from: &str, to: &str) -> io::Result<()> {
        // Linux renames a cgroup only within its parent.
        let (from_dir, from_name) = split_last(from);
        let (to_dir, to_name) = split_last(to);
        if from_dir != to_dir {
            return Err(errno(18)); // EXDEV
        }
        let mut tree = self.tree.lock().unwrap();
        let (node, file) = tree.lookup(self.root, from).ok_or_else(|| errno(ENOENT))?;
        if file.is_some() || node == self.root {
            return Err(errno(EPERM));
        }
        let (parent, _) = tree
            .lookup(self.root, from_dir)
            .ok_or_else(|| errno(ENOENT))?;
        if tree.lookup(self.root, to).is_some() {
            return Err(errno(EEXIST));
        }
        let p = tree.nodes.get_mut(&parent).expect("found");
        p.children.remove(from_name);
        p.children.insert(to_name.to_string(), node);
        tree.nodes.get_mut(&node).expect("found").name = to_name.to_string();
        Ok(())
    }

    fn set_owner(&mut self, rel: &str, uid: Option<u32>, gid: Option<u32>) -> io::Result<()> {
        let mut tree = self.tree.lock().unwrap();
        let (node, file) = tree.lookup(self.root, rel).ok_or_else(|| errno(ENOENT))?;
        // A directory's owner is its delegate, and the files a delegation
        // hands over follow it; other files stay root's.
        let hands_over = file.as_deref().is_none_or(|f| {
            matches!(
                f,
                "cgroup.procs" | "cgroup.threads" | "cgroup.subtree_control"
            )
        });
        if hands_over {
            let n = tree.nodes.get_mut(&node).expect("found");
            if let Some(u) = uid {
                n.uid = u;
            }
            if let Some(g) = gid {
                n.gid = g;
            }
        }
        Ok(())
    }

    fn as_cgroupfs_mut(&mut self) -> Option<&mut CgroupFs> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(fs: &mut CgroupFs, rel: &str) -> String {
        let mut buf = [0u8; 512];
        let n = fs.read_at(rel, 0, &mut buf).unwrap();
        String::from_utf8_lossy(&buf[..n]).into_owned()
    }

    #[test]
    fn a_hierarchy_with_the_pids_controller() {
        let tree = CgroupTree::shared();
        let mut fs = CgroupFs::new(tree.clone(), ROOT);
        assert_eq!(read(&mut fs, "cgroup.controllers"), "pids\n");
        assert!(fs.stat("cgroup.events").is_none(), "the root has none");
        fs.mkdir("a", 0o755).unwrap();
        assert!(fs.stat("a/pids.max").is_none(), "not enabled yet");
        fs.write_at("cgroup.subtree_control", 0, b"+pids").unwrap();
        assert_eq!(read(&mut fs, "a/pids.max"), "max\n");
        fs.write_at("a/pids.max", 0, b"2\n").unwrap();
        let a = fs.resolve("a").unwrap().0;
        let mut t = tree.lock().unwrap();
        t.migrate(&[(10, true)], a);
        assert_eq!(t.fork(10, 11, None), Ok(false));
        assert_eq!(t.fork(10, 12, None), Err(EAGAIN), "pids.max");
        assert_eq!(t.render(a, "pids.current"), "2\n");
        assert_eq!(t.render(a, "pids.events"), "max 1\n");
        assert_eq!(t.rmdir(a), Err(EBUSY), "populated");
        t.exit(11, 5_000);
        t.release(11, 0);
        t.exit(10, 1_000);
        assert_eq!(t.render(a, "cgroup.events"), "populated 0\nfrozen 0\n");
        assert_eq!(
            t.render(a, "pids.current"),
            "1\n",
            "the zombie still counts"
        );
        assert_eq!(t.dead_cpu_ns(ROOT), 6_000);
        t.rmdir(a).unwrap();
        assert_eq!(t.node_of(10), ROOT, "the zombie moved up");
        t.release(10, 0);
    }

    #[test]
    fn no_internal_processes_and_limits_on_the_tree() {
        let tree = CgroupTree::shared();
        let mut t = tree.lock().unwrap();
        let a = t.mkdir(ROOT, "a", 0, 0).unwrap();
        t.migrate(&[(5, true)], a);
        t.write_file(ROOT, "cgroup.subtree_control", "+pids")
            .unwrap();
        assert_eq!(
            t.write_file(a, "cgroup.subtree_control", "+pids"),
            Err(EBUSY),
            "a has a task of its own"
        );
        assert_eq!(
            t.write_file(a, "cgroup.subtree_control", "+cpu"),
            Err(ENOENT)
        );
        t.write_file(a, "cgroup.max.depth", "1").unwrap();
        let b = t.mkdir(a, "b", 0, 0).unwrap();
        assert_eq!(t.mkdir(b, "c", 0, 0), Err(EAGAIN), "max.depth");
        assert_eq!(t.mkdir(a, "b", 0, 0), Err(EEXIST));
        assert_eq!(t.path_from(ROOT, b), "/a/b");
        assert_eq!(t.path_from(b, a), "/..");
        assert_eq!(t.path_from(a, b), "/b");
        let c = t.mkdir(ROOT, "c", 0, 0).unwrap();
        assert_eq!(t.path_from(b, c), "/../../c");
        assert_eq!(t.common_ancestor(b, c), ROOT);
    }
}
