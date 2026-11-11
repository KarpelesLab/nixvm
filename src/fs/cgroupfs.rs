//! cgroup v2: the control group hierarchy and the `cgroup2` filesystem that
//! shows it.
//!
//! [`CgroupTree`] is the VM's one hierarchy, shared by every `cgroup2` mount
//! ([`CgroupFs`]) and the kernel. A mount shows the tree from a node down: the
//! root for the first cgroup namespace, a namespace's own root otherwise.
//!
//! Only controllers nixvm enforces are offered: `cpu` (a `cpu.max` bandwidth
//! limit, which holds a cgroup's tasks off the CPU once its quota for the
//! period is used, and `cpu.weight`, which scales its tasks' share of the CPU),
//! `memory` (pages charged to the cgroup by the frame allocator, and a
//! `memory.max` past which a task is killed) and `pids` (a `pids.max` that
//! `fork` and `clone` honour), next to the core files every cgroup has:
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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use super::{Attrs, DirEntry, MountFs, NodeKind};

/// The root cgroup's id.
pub const ROOT: u32 = 0;

/// The controllers there are, as `cgroup.controllers` lists them, with the
/// bit each has in a cgroup's set of enabled controllers.
const CONTROLLERS: &[(&str, u8)] = &[("cpu", CPU), ("io", IO), ("memory", MEMORY), ("pids", PIDS)];
const CPU: u8 = 1;
const PIDS: u8 = 2;
const MEMORY: u8 = 4;
const IO: u8 = 8;
/// The threaded controllers, which may share a cgroup with tasks; the others
/// (`memory`) are domain controllers, held to the no-internal-process rule.
const THREADED: u8 = CPU | PIDS;

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
    ("cpu.stat.local", false),
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

/// The `cpu` controller's files, in a cgroup whose parent enables it.
const CPU_FILES: &[(&str, bool)] = &[
    ("cpu.idle", true),
    ("cpu.max", true),
    ("cpu.max.burst", true),
    ("cpu.weight", true),
    ("cpu.weight.nice", true),
];

/// The `io` controller's files, in a cgroup whose parent enables it.
const IO_FILES: &[(&str, bool)] = &[("io.max", true), ("io.stat", false), ("io.weight", true)];

/// How far back `io.max` lets a cgroup's unused allowance reach, so a burst
/// after a quiet spell gets through at once: Linux's `throtl_slice`.
const IO_SLICE_NS: u128 = 100_000_000;

/// The `memory` controller's files, in a cgroup whose parent enables it.
const MEMORY_FILES: &[(&str, bool)] = &[
    ("memory.current", false),
    ("memory.events", false),
    ("memory.events.local", false),
    ("memory.high", true),
    ("memory.low", true),
    ("memory.max", true),
    ("memory.min", true),
    ("memory.oom.group", true),
    ("memory.peak", false),
    ("memory.reclaim", true),
    ("memory.stat", false),
];

/// Writes that change how the cgroup's tasks are scheduled or charged, which
/// the kernel passes on to them once the write is done.
pub const SCHED_FILES: &[&str] = &[
    "cgroup.subtree_control",
    "cgroup.type",
    "memory.high",
    "memory.low",
    "memory.max",
    "memory.min",
    "io.max",
    "cpu.idle",
    "cpu.max",
    "cpu.max.burst",
    "cpu.weight",
    "cpu.weight.nice",
];

/// Writes the kernel handles, since they act on tasks.
pub const KERNEL_FILES: &[&str] = &[
    "memory.reclaim",
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
const ERANGE: i32 = 34;
const EOPNOTSUPP: i32 = 95;

/// The default scheduling weight (a nice-0 task's), which `cpu.weight` 100 is.
const NICE0_SHARES: u64 = 1024;
/// An idle cgroup's weight (Linux's `WEIGHT_IDLEPRIO`).
const IDLE_SHARES: u64 = 3;
const NS_PER_US: u128 = 1000;

/// `cpu.max`, `cpu.max.burst` and the state of the bandwidth they allow, as
/// Linux's CFS bandwidth control keeps it: each period the cgroup gets its
/// quota of CPU time (plus what it saved, up to the burst); used up, its tasks
/// wait for the next period.
#[derive(Debug, Clone)]
struct Bandwidth {
    /// Microseconds of CPU time per period; `None` is "max".
    quota_us: Option<u64>,
    period_us: u64,
    burst_us: u64,
    /// When the current period started (ns, as `poll::now_ns`), once the
    /// cgroup has used CPU time under a quota.
    period_start: u128,
    /// CPU time left this period (ns); negative when overdrawn.
    runtime: i128,
    /// `runtime` as the period began, to tell how much of a burst was used.
    snap: i128,
    /// Throttled since this time, until the period ends.
    throttled_since: Option<u128>,
    nr_periods: u64,
    nr_throttled: u64,
    throttled_ns: u128,
    nr_bursts: u64,
    burst_ns: u128,
}

impl Default for Bandwidth {
    fn default() -> Self {
        Self {
            quota_us: None,
            period_us: 100_000,
            burst_us: 0,
            period_start: 0,
            runtime: 0,
            snap: 0,
            throttled_since: None,
            nr_periods: 0,
            nr_throttled: 0,
            throttled_ns: 0,
            nr_bursts: 0,
            burst_ns: 0,
        }
    }
}

impl Bandwidth {
    fn period_ns(&self) -> u128 {
        u128::from(self.period_us) * NS_PER_US
    }

    fn end(&self) -> u128 {
        self.period_start + self.period_ns()
    }

    /// Move on to the period `now` is in, refilling the quota for it.
    fn roll(&mut self, now: u128) {
        let Some(quota) = self.quota_us else {
            return;
        };
        let quota = i128::from(quota) * NS_PER_US as i128;
        let cap = quota + i128::from(self.burst_us) * NS_PER_US as i128;
        if self.period_start == 0 {
            self.period_start = now;
            self.runtime = quota;
            self.snap = quota;
            self.nr_periods = 1;
            return;
        }
        if now < self.end() {
            return;
        }
        if let Some(since) = self.throttled_since.take() {
            self.throttled_ns += self.end().saturating_sub(since);
        }
        let periods = (now - self.period_start) / self.period_ns();
        self.period_start += periods * self.period_ns();
        // Linux counts the periods its timer ran for, which stops once the
        // cgroup goes idle: a long idle gap counts once.
        self.nr_periods += 1;
        // A burst is the saved runtime the last period drew on. Overdrawing
        // (a task runs a little past the end of the quota) isn't one: Linux
        // keeps that debt per CPU, apart from the pool, and repays it from
        // the next period's quota, as it is here.
        let used_beyond_quota = self.snap - (self.runtime.max(0) + quota);
        self.runtime += quota;
        if used_beyond_quota > 0 {
            self.nr_bursts += 1;
            self.burst_ns += used_beyond_quota.cast_unsigned();
        }
        self.runtime = self.runtime.min(cap);
        self.snap = self.runtime;
    }

    /// Throttled time, the current throttle's elapsed part included.
    fn throttled_ns(&self, now: u128) -> u128 {
        self.throttled_ns
            + self
                .throttled_since
                .map_or(0, |s| now.min(self.end()).saturating_sub(s))
    }

    /// Until when the cgroup is throttled, if it is at `now`.
    fn throttled_until(&self, now: u128) -> Option<u128> {
        (self.throttled_since.is_some() && now < self.end()).then(|| self.end())
    }
}

/// What the tree needs of the frame allocator to keep its memory cgroups (see
/// [`CgroupTree::sync_memcgs`]).
pub trait MemcgHost {
    /// A new memory cgroup below `parent`; `None` if there's no room.
    fn memcg_create(&mut self, parent: u16) -> Option<u16>;
    fn memcg_drop(&mut self, slot: u16);
    /// `memory.max` and `memory.high`, in pages (`u64::MAX` for none), and
    /// `memory.low` and `memory.min` (0 for none).
    fn memcg_limits(&mut self, slot: u16, max: u64, high: u64, low: u64, min: u64);
    /// What's charged to `slot` and below.
    fn memcg_use(&self, slot: u16) -> MemUse;
}

/// The guest's page size, the unit memory is charged in.
const PAGE: u64 = 4096;

/// A memory size as Linux's `memparse` takes one: a number (decimal, `0x`
/// hex, `0` octal) with an optional `K`, `M`, `G`, `T`, `P` or `E` suffix, or
/// "max". Rounded down to whole pages, as `page_counter_memparse` does.
/// A memory size as `memory.max` takes one, for the kernel: in pages, `None` for "max".
///
/// # Errors
/// `EINVAL` (as an errno) when `text` isn't a size.
pub fn parse_mem_pages(text: &str) -> Result<Option<u64>, i32> {
    mem_limit(text)
}

fn mem_limit(text: &str) -> Result<Option<u64>, i32> {
    if text == "max" {
        return Ok(None);
    }
    let hex = text.starts_with("0x") || text.starts_with("0X");
    let digits = if hex { 2 } else { 0 };
    let split = text[digits..]
        .find(|c: char| {
            if hex {
                !c.is_ascii_hexdigit()
            } else {
                !c.is_ascii_digit()
            }
        })
        .map_or(text.len(), |i| i + digits);
    let (num, suffix) = text.split_at(split);
    let n: u64 = parse_int(num)?;
    let shift = match suffix {
        "" => 0,
        "k" | "K" => 10,
        "m" | "M" => 20,
        "g" | "G" => 30,
        "t" | "T" => 40,
        "p" | "P" => 50,
        "e" | "E" => 60,
        _ => return Err(EINVAL),
    };
    Ok(Some(
        n.checked_shl(shift)
            .filter(|v| v >> shift == n)
            .unwrap_or(u64::MAX)
            / PAGE,
    ))
}

/// An integer as a kernfs file takes one: decimal, or `0x` hex or `0` octal.
fn parse_int<T: TryFrom<i128>>(text: &str) -> Result<T, i32> {
    let (neg, t) = match text.strip_prefix('-') {
        Some(t) => (true, t),
        None => (false, text.strip_prefix('+').unwrap_or(text)),
    };
    let v = if let Some(h) = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        i128::from_str_radix(h, 16)
    } else if t.len() > 1 && t.starts_with('0') {
        i128::from_str_radix(&t[1..], 8)
    } else {
        t.parse()
    }
    .map_err(|_| EINVAL)?;
    T::try_from(if neg { -v } else { v }).map_err(|_| ERANGE)
}

/// Set a cgroup's bandwidth, with the bounds Linux's `tg_set_bandwidth` has:
/// a period from 1 ms to 1 s, a quota of at least 1 ms, and no more burst than
/// quota. The period starts over.
fn set_bandwidth(
    bw: &mut Bandwidth,
    quota: Option<u64>,
    period: u64,
    burst: u64,
) -> Result<(), i32> {
    const MAX_RUNTIME_US: u64 = (1 << 44) - 1;
    if !(1000..=1_000_000).contains(&period)
        || quota.is_some_and(|q| !(1000..=MAX_RUNTIME_US).contains(&q))
        || quota.is_some_and(|q| burst > q || burst + q > MAX_RUNTIME_US)
        || burst > u64::MAX / 1000
    {
        return Err(EINVAL);
    }
    *bw = Bandwidth {
        quota_us: quota,
        period_us: period,
        burst_us: burst,
        nr_periods: bw.nr_periods,
        nr_throttled: bw.nr_throttled,
        throttled_ns: bw.throttled_ns,
        nr_bursts: bw.nr_bursts,
        burst_ns: bw.burst_ns,
        ..Bandwidth::default()
    };
    Ok(())
}

/// A memory limit in pages as its file shows it: bytes, or "max".
fn mem_bytes(pages: Option<u64>) -> String {
    pages.map_or_else(|| "max\n".to_string(), |p| format!("{}\n", p * PAGE))
}

/// The weights Linux gives each nice level, for `cpu.weight.nice`: nice 0 is
/// 1024, and each level is about 1.25 times the next.
fn nice_shares(nice: i64) -> u64 {
    (1024.0 / 1.25f64.powi(nice.clamp(-20, 19) as i32)).round() as u64
}

/// One cgroup.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)] // independent per-cgroup settings
struct Node {
    name: String,
    parent: Option<u32>,
    children: BTreeMap<String, u32>,
    /// The controllers `cgroup.subtree_control` enables for the children.
    subtree: u8,
    /// `cgroup.type` "threaded": the cgroup is part of the resource domain of
    /// its nearest non-threaded ancestor, its thread root (Linux's `dom_cgrp`).
    threaded: bool,
    /// `pids.max`; `None` is "max".
    pids_max: Option<u64>,
    /// How often a fork failed on this cgroup's `pids.max` (`pids.events`).
    pids_events: u64,
    /// The scheduling weight `cpu.weight` sets (in Linux's units, 1024 being
    /// `cpu.weight` 100).
    shares: u64,
    /// `cpu.idle`.
    idle: bool,
    bandwidth: Bandwidth,
    /// `memory.max`, `memory.high` in bytes (`None` is "max"), and
    /// `memory.low`, `memory.min`: what reclaim leaves to it (see
    /// `FrameAllocator::reclaim`).
    mem_max: Option<u64>,
    mem_high: Option<u64>,
    mem_low: Option<u64>,
    mem_min: Option<u64>,
    /// `memory.oom.group`: an out-of-memory kill takes every task in it.
    oom_group: bool,
    /// `io.max`: per disk, `rbps`, `wbps`, `riops`, `wiops` (`None`: max).
    io_max: BTreeMap<u64, [Option<u64>; 4]>,
    /// For each of those, when the I/O allowed so far is used up (see
    /// [`CgroupTree::io_account`]).
    io_next: BTreeMap<(u64, usize), u128>,
    /// `io.weight`: the default and per disk. Linux's `io.cost` model, which
    /// weights would share disk time by, is off unless `io.cost.qos` is set
    /// up (there's none here), so they're kept but don't change anything.
    io_weight: u64,
    io_weight_dev: BTreeMap<u64, u64>,
    /// `io.stat`: per disk, the bytes and I/Os read and written here and
    /// below.
    io_stat: BTreeMap<u64, [u64; 4]>,
    /// The memory cgroup the frame allocator charges for this one, once
    /// the kernel has made one (see [`CgroupTree::memcg_of`]).
    memcg: Option<u16>,
    /// Out-of-memory events here: `oom`, `oom_kill`, `oom_group_kill`.
    oom_events: [u64; 3],
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
    /// Back to what a controller's files start with, as when Linux makes the
    /// controller's state for the cgroup anew.
    fn reset(&mut self, controllers: u8) {
        if controllers & CPU != 0 {
            self.shares = NICE0_SHARES;
            self.idle = false;
            self.bandwidth = Bandwidth::default();
        }
        if controllers & PIDS != 0 {
            self.pids_max = None;
            self.pids_events = 0;
        }
        if controllers & MEMORY != 0 {
            self.mem_max = None;
            self.mem_high = None;
            self.mem_low = Some(0);
            self.mem_min = Some(0);
            self.oom_group = false;
            self.oom_events = [0; 3];
        }
        if controllers & IO != 0 {
            self.io_max.clear();
            self.io_next.clear();
            self.io_weight = 100;
            self.io_weight_dev.clear();
        }
    }

    fn new(name: &str, parent: Option<u32>, uid: u32, gid: u32) -> Self {
        Self {
            name: name.to_string(),
            parent,
            children: BTreeMap::new(),
            subtree: 0,
            threaded: false,
            pids_max: None,
            pids_events: 0,
            shares: NICE0_SHARES,
            idle: false,
            bandwidth: Bandwidth::default(),
            mem_max: None,
            mem_high: None,
            mem_low: Some(0),
            mem_min: Some(0),
            oom_group: false,
            io_max: BTreeMap::new(),
            io_next: BTreeMap::new(),
            io_weight: 100,
            io_weight_dev: BTreeMap::new(),
            io_stat: BTreeMap::new(),
            memcg: None,
            oom_events: [0; 3],
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

/// The time the bandwidth periods run on: the kernel's `poll::now_ns` clock.
fn now_ns() -> u128 {
    crate::clock::now_unix().as_nanos()
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
    /// Each memory cgroup's use as the frame allocator has it.
    pub memory: BTreeMap<u32, MemUse>,
}

/// A memory cgroup's use, as the frame allocator counts it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MemUse {
    /// Bytes charged here and below, and the most there have been.
    pub current: u64,
    pub peak: u64,
    /// Times an allocation here went past `memory.high`, or was refused at
    /// `memory.max`.
    pub high_events: u64,
    pub max_events: u64,
    /// Bytes of it that are file pages (the page cache).
    pub file: u64,
    /// Times reclaim took from it below its `memory.low`.
    pub low_events: u64,
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
    /// How many cgroups have a `cpu.max` quota: while none do, the kernel
    /// needn't charge CPU time to cgroups as it goes.
    quotas: Arc<AtomicUsize>,
    /// Memory cgroups of cgroups that went away, for the kernel to drop.
    dead_memcgs: Vec<u16>,
    /// The VM's disks, which `io.max` and `io.weight` name.
    disks: Option<super::blockdev::SharedDisks>,
    /// Whether any cgroup has an `io.max` limit.
    io_limited: bool,
}

/// How a cgroup's tasks are to be scheduled, from its own settings and its
/// ancestors'.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchedState {
    /// `cgroup.freeze` holds them.
    pub frozen: bool,
    /// What each nanosecond they run counts for in fair scheduling, against a
    /// task in the root, in units of 1/65536: from `cpu.weight` on the way up.
    pub scale: u64,
    /// A `cpu.max` holds them until then.
    pub throttled_until: Option<u128>,
    /// The memory cgroup what they allocate is charged to.
    pub memcg: u16,
}

/// [`SchedState::scale`] for a cgroup with the default weight.
pub const UNSCALED: u64 = 1 << 16;

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
            quotas: Arc::new(AtomicUsize::new(0)),
            dead_memcgs: Vec::new(),
            disks: None,
            io_limited: false,
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

    /// A count, kept up to date, of the cgroups with a `cpu.max` quota.
    #[must_use]
    pub fn quota_gate(&self) -> Arc<AtomicUsize> {
        self.quotas.clone()
    }

    fn update_quota_gate(&self) {
        let n = self
            .nodes
            .iter()
            .filter(|&(&id, x)| x.bandwidth.quota_us.is_some() && self.has(id, CPU))
            .count();
        self.quotas.store(n, Ordering::Relaxed);
    }

    /// The cgroups from `node` up whose `cpu` settings apply.
    fn cpu_chain(&self, node: u32) -> Vec<u32> {
        self.chain(node)
            .into_iter()
            .filter(|&n| n != ROOT && self.has(n, CPU))
            .collect()
    }

    /// Whether any cgroup has the `cpu` controller, so its weights matter.
    #[must_use]
    pub fn weights_active(&self) -> bool {
        self.nodes.get(&ROOT).is_some_and(|r| r.subtree & CPU != 0)
    }

    /// The [`SchedState::scale`] of each of `tasks` (task id, its own
    /// weight from its nice value), the tasks that are runnable now, as
    /// Linux's group scheduling shares out the CPU: a cgroup gets its
    /// `cpu.weight`'s part of what's runnable beside it, and splits that among
    /// what's runnable in it. That's a task getting
    /// `Π shares(G) / Σ weights runnable in G` over its cgroups `G` with the
    /// controller, against a task in the root; the scale is the inverse, so
    /// its virtual runtime moves as fast as everyone's at its fair share.
    #[must_use]
    pub fn group_scales(&self, tasks: &[(i32, u128)]) -> Vec<u64> {
        let chains: Vec<Vec<u32>> = tasks
            .iter()
            .map(|&(tid, _)| self.cpu_chain(self.node_of(tid)))
            .collect();
        // What's runnable in each cgroup: its own tasks' weights, and the
        // shares of each child cgroup with something runnable below it.
        let mut inside: BTreeMap<u32, u128> = BTreeMap::new();
        let mut active = std::collections::BTreeSet::new();
        for (chain, &(_, w)) in chains.iter().zip(tasks) {
            // `cpu_chain` runs from the cgroup up.
            if let Some(&g) = chain.first() {
                *inside.entry(g).or_default() += w;
            }
            for (k, &g) in chain.iter().enumerate() {
                if active.insert(g)
                    && let Some(&parent) = chain.get(k + 1)
                {
                    *inside.entry(parent).or_default() += u128::from(self.nodes[&g].shares.max(1));
                }
            }
        }
        chains
            .iter()
            .map(|chain| {
                let scale = chain.iter().fold(u128::from(UNSCALED), |scale, g| {
                    (scale * inside.get(g).copied().unwrap_or(1)
                        / u128::from(self.nodes[g].shares.max(1)))
                    .min(u128::from(u64::MAX))
                });
                u64::try_from(scale).unwrap_or(u64::MAX)
            })
            .collect()
    }

    /// How tasks in `node` are to be scheduled at `now`.
    #[must_use]
    pub fn sched_state(&self, node: u32, now: u128) -> SchedState {
        let mut scale = u128::from(UNSCALED);
        let mut throttled_until = None;
        for n in self.cpu_chain(node) {
            let x = &self.nodes[&n];
            scale = (scale * u128::from(NICE0_SHARES) / u128::from(x.shares.max(1)))
                .min(u128::from(u64::MAX));
            throttled_until = throttled_until.max(x.bandwidth.throttled_until(now));
        }
        SchedState {
            frozen: self.frozen(node),
            scale: u64::try_from(scale).unwrap_or(u64::MAX),
            throttled_until,
            memcg: self.memcg_of(node),
        }
    }

    /// Charge `ns` of CPU time a task in `node` used to each `cpu.max` on the
    /// way up, at `now`. Returns until when the task is now throttled, if it
    /// is, and the highest cgroup this throttled anew, whose other tasks are
    /// to be held as well.
    pub fn charge_cpu(&mut self, node: u32, ns: u64, now: u128) -> (Option<u128>, Option<u32>) {
        let mut until = None;
        let mut newly = None;
        for n in self.cpu_chain(node) {
            let b = &mut self.nodes.get_mut(&n).expect("in the chain").bandwidth;
            if b.quota_us.is_none() {
                continue;
            }
            b.roll(now);
            b.runtime -= i128::from(ns);
            if b.throttled_since.is_none() && b.runtime <= 0 {
                b.throttled_since = Some(now);
                b.nr_throttled += 1;
                newly = Some(n);
            }
            until = until.max(b.throttled_until(now));
        }
        (until, newly)
    }

    /// The memory cgroup tasks in `node` are charged to: its own, or the
    /// nearest one above it (0, the root's, if none).
    #[must_use]
    pub fn memcg_of(&self, node: u32) -> u16 {
        self.chain(node)
            .into_iter()
            .find_map(|n| self.nodes.get(&n).and_then(|x| x.memcg))
            .unwrap_or(0)
    }

    /// Cgroup `node`'s own memory cgroup, if it has the controller.
    #[must_use]
    pub fn memcg_slot(&self, node: u32) -> Option<u16> {
        self.nodes.get(&node).and_then(|n| n.memcg)
    }

    /// Memory cgroup `slot` was over its `memory.max` with no task left to
    /// kill: still an `oom` event.
    pub fn record_oom_attempt(&mut self, slot: u16) {
        if let Some(d) = self
            .node_of_memcg(slot)
            .and_then(|n| self.nodes.get_mut(&n))
        {
            d.oom_events[0] += 1;
        }
    }

    /// The cgroup memory cgroup `slot` belongs to.
    #[must_use]
    pub fn node_of_memcg(&self, slot: u16) -> Option<u32> {
        self.nodes
            .iter()
            .find(|(_, n)| n.memcg == Some(slot))
            .map(|(&id, _)| id)
    }

    /// Give each cgroup with the `memory` controller a memory cgroup in the
    /// frame allocator, with its limits, and drop those of cgroups that lost
    /// it or are gone.
    pub fn sync_memcgs(&mut self, host: &mut dyn MemcgHost) {
        for slot in std::mem::take(&mut self.dead_memcgs) {
            host.memcg_drop(slot);
        }
        for node in self.subtree(ROOT) {
            let want = node != ROOT && self.has(node, MEMORY);
            let parent = self.nodes[&node].parent.map_or(0, |p| self.memcg_of(p));
            let n = self.nodes.get_mut(&node).expect("in the tree");
            match (want, n.memcg) {
                (true, None) => n.memcg = host.memcg_create(parent),
                (false, Some(slot)) => {
                    host.memcg_drop(slot);
                    n.memcg = None;
                }
                _ => {}
            }
            if let Some(slot) = n.memcg {
                host.memcg_limits(
                    slot,
                    n.mem_max.unwrap_or(u64::MAX),
                    n.mem_high.unwrap_or(u64::MAX),
                    n.mem_low.unwrap_or(u64::MAX),
                    n.mem_min.unwrap_or(u64::MAX),
                );
            }
        }
    }

    /// Bring [`View::memory`] up to date from the frame allocator.
    pub fn refresh_memory(&mut self, host: &dyn MemcgHost) {
        self.view.memory = self
            .nodes
            .iter()
            .filter_map(|(&id, n)| Some((id, host.memcg_use(n.memcg?))))
            .collect();
    }

    /// A task in `victim` is being killed because memory cgroup `slot`'s
    /// `memory.max` was reached: count it, and say which cgroup to kill
    /// whole, if `memory.oom.group` asks for that (the highest cgroup from
    /// the victim's up to the one out of memory that has it set).
    pub fn record_oom(&mut self, victim: u32, slot: u16) -> Option<u32> {
        let domain = self.node_of_memcg(slot)?;
        let group = self
            .chain(victim)
            .into_iter()
            .take_while(|&n| self.is_within(n, domain))
            .filter(|n| self.nodes.get(n).is_some_and(|x| x.oom_group))
            .last();
        let d = self.nodes.get_mut(&domain)?;
        d.oom_events[0] += 1; // oom
        d.oom_events[1] += 1; // oom_kill
        if group.is_some() {
            d.oom_events[2] += 1; // oom_group_kill
        }
        group
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

    /// Whether one of `node`'s children that isn't threaded has live tasks
    /// in it or below (Linux's `nr_populated_domain_children`).
    fn has_populated_domain_child(&self, node: u32) -> bool {
        self.nodes.get(&node).is_some_and(|n| {
            n.children
                .values()
                .any(|c| self.nodes.get(c).is_some_and(|c| c.live > 0 && !c.threaded))
        })
    }

    /// Whether `node` has live tasks directly in it.
    fn has_own_tasks(&self, node: u32) -> bool {
        self.members.values().any(|m| m.node == node && m.live)
    }

    /// Whether `node`'s `cgroup.type` is "threaded".
    #[must_use]
    pub fn is_threaded(&self, node: u32) -> bool {
        self.nodes.get(&node).is_some_and(|n| n.threaded)
    }

    /// The resource domain `node` is part of: itself, or for a threaded
    /// cgroup its thread root, the nearest ancestor that isn't threaded.
    #[must_use]
    pub fn domain_of(&self, node: u32) -> u32 {
        self.chain(node)
            .into_iter()
            .find(|&n| !self.is_threaded(n))
            .unwrap_or(ROOT)
    }

    /// Whether `node` is the root of a threaded subtree: a domain with
    /// threaded children, or with tasks and threaded controllers enabled
    /// (Linux's `cgroup_is_thread_root`).
    fn is_thread_root(&self, node: u32) -> bool {
        let Some(n) = self.nodes.get(&node) else {
            return false;
        };
        !n.threaded
            && (n.children.values().any(|&c| self.is_threaded(c))
                || (self.has_own_tasks(node) && n.subtree & THREADED != 0))
    }

    /// Whether `node` could serve as a thread root (Linux's
    /// `cgroup_can_be_thread_root`): the root always can; otherwise a domain
    /// with no populated domain children and no domain controllers enabled.
    fn can_be_thread_root(&self, node: u32) -> bool {
        if node == ROOT {
            return true;
        }
        let Some(n) = self.nodes.get(&node) else {
            return false;
        };
        !n.threaded && !self.has_populated_domain_child(node) && n.subtree & !THREADED == 0
    }

    /// Whether `node` is a domain that can hold resources (Linux's
    /// `cgroup_is_valid_domain`): not threaded, and no ancestor below the
    /// root is threaded or a thread root. One that isn't is "domain invalid".
    fn is_valid_domain(&self, node: u32) -> bool {
        !self.is_threaded(node)
            && self
                .chain(node)
                .into_iter()
                .skip(1)
                .all(|a| !self.is_threaded(a) && (a == ROOT || !self.is_thread_root(a)))
    }

    /// Whether a task can be put in `node` (Linux's `cgroup_migrate_vet_dst`):
    /// its domain must be a valid one (`EOPNOTSUPP`), and, by cgroup v2's "no
    /// internal processes" rule, a cgroup that hands controllers to its
    /// children takes no tasks (`EBUSY`), unless it is or could be a thread
    /// root, or is threaded: threaded controllers (`cpu`, `pids`) may share
    /// a cgroup with tasks.
    pub fn check_attach(&self, node: u32) -> Result<(), i32> {
        let Some(n) = self.nodes.get(&node) else {
            return Err(ENOENT);
        };
        if !self.is_valid_domain(self.domain_of(node)) {
            return Err(EOPNOTSUPP);
        }
        if self.can_be_thread_root(node) || n.threaded || n.subtree == 0 {
            return Ok(());
        }
        Err(EBUSY)
    }

    /// Whether thread `tid` may move to `node` on its own (`cgroup.threads`,
    /// or `CLONE_INTO_CGROUP` with `CLONE_THREAD`): only within its resource
    /// domain (`EOPNOTSUPP` otherwise), on top of [`Self::check_attach`].
    pub fn check_thread_attach(&self, tid: i32, node: u32) -> Result<(), i32> {
        self.check_attach(node)?;
        if self.domain_of(self.node_of(tid)) == self.domain_of(node) {
            Ok(())
        } else {
            Err(EOPNOTSUPP)
        }
    }

    /// `cgroup.type`: "threaded" makes `node` join its parent's resource
    /// domain (Linux's `cgroup_enable_threaded`); there's no going back.
    fn enable_threaded(&mut self, node: u32) -> Result<(), i32> {
        let n = self.nodes.get(&node).ok_or(ENOENT)?;
        if n.threaded {
            return Ok(());
        }
        // Populated, or with domain controllers enabled: it can't be switched.
        if n.live > 0 || n.subtree & !THREADED != 0 {
            return Err(EOPNOTSUPP);
        }
        let dom = self.domain_of(n.parent.ok_or(EINVAL)?);
        if !self.is_valid_domain(dom) || !self.can_be_thread_root(dom) {
            return Err(EOPNOTSUPP);
        }
        let n = self.nodes.get_mut(&node).expect("checked");
        n.threaded = true;
        // Only threaded controllers are left to it; a domain one's state
        // goes, as Linux drops it.
        n.reset(!THREADED);
        n.subtree &= THREADED;
        self.update_quota_gate();
        Ok(())
    }

    /// The "cgroup.type" of `node`.
    fn type_name(&self, node: u32) -> &'static str {
        if self.is_threaded(node) {
            "threaded"
        } else if !self.is_valid_domain(node) {
            "domain invalid"
        } else if self.is_thread_root(node) {
            "domain threaded"
        } else {
            "domain"
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
        if node != ROOT && self.has(node, CPU) {
            out.extend_from_slice(CPU_FILES);
        }
        if node != ROOT && self.has(node, PIDS) {
            out.extend_from_slice(PIDS_FILES);
        }
        if node != ROOT && self.has(node, MEMORY) {
            out.extend_from_slice(MEMORY_FILES);
        }
        if node == ROOT {
            out.push(("io.stat", false));
        } else if self.has(node, IO) {
            out.extend_from_slice(IO_FILES);
        }
        out.sort_unstable();
        out
    }

    /// The controllers `node` has: all of them at the root, elsewhere those
    /// its parent enables.
    fn controller_bits(&self, node: u32) -> u8 {
        match self.nodes.get(&node).and_then(|n| n.parent) {
            None => CONTROLLERS.iter().fold(0, |a, (_, b)| a | b),
            Some(p) => {
                let bits = self.nodes.get(&p).map_or(0, |p| p.subtree);
                if self.is_threaded(node) {
                    bits & THREADED
                } else {
                    bits
                }
            }
        }
    }

    /// Whether `node` has `controller`.
    fn has(&self, node: u32, controller: u8) -> bool {
        self.controller_bits(node) & controller != 0
    }

    /// The names of the controllers in `bits`.
    fn names(bits: u8) -> String {
        CONTROLLERS
            .iter()
            .filter(|(_, b)| bits & b != 0)
            .map(|(n, _)| *n)
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// The controllers `node` may enable for its children.
    fn controllers(&self, node: u32) -> String {
        Self::names(self.controller_bits(node))
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
        let n_memcg = n.memcg;
        self.nodes.remove(&node);
        if let Some(p) = self.nodes.get_mut(&parent) {
            p.children.remove(&name);
            p.mtime = now();
            // Its zombies are now directly in the parent, which counted them
            // already; nothing to re-charge.
            let _ = tasks;
        }
        self.ns_roots.retain(|_, r| *r != node);
        self.dead_memcgs.extend(n_memcg);
        self.update_quota_gate();
        Ok(())
    }

    /// A write to one of the files handled here.
    pub fn write_file(&mut self, node: u32, file: &str, text: &str) -> Result<(), i32> {
        let text = text.trim();
        let limit = |t: &str| -> Result<Option<u64>, i32> {
            if t == "max" {
                Ok(None)
            } else {
                t.parse().map(Some).map_err(|_| EINVAL)
            }
        };
        match self.files(node).iter().find(|(f, _)| *f == file) {
            None => return Err(ENOENT),
            Some((_, false)) => return Err(EACCES),
            Some(_) => {}
        }
        let n = self.nodes.get_mut(&node).ok_or(ENOENT)?;
        match file {
            "cgroup.subtree_control" => return self.write_subtree_control(node, text),
            "cgroup.max.depth" => n.max_depth = limit(text)?,
            "cgroup.max.descendants" => n.max_descendants = limit(text)?,
            "pids.max" => n.pids_max = limit(text)?,
            "cpu.weight" => {
                let w: u64 = parse_int(text)?;
                if !(1..=10_000).contains(&w) {
                    return Err(ERANGE);
                }
                if n.idle {
                    return Err(EINVAL);
                }
                n.shares = (w * NICE0_SHARES + 50) / 100;
            }
            "cpu.weight.nice" => {
                let nice: i64 = parse_int(text)?;
                if !(-20..=19).contains(&nice) {
                    return Err(ERANGE);
                }
                if n.idle {
                    return Err(EINVAL);
                }
                n.shares = nice_shares(nice);
            }
            "cpu.idle" => {
                let idle = match parse_int::<i64>(text)? {
                    0 => false,
                    1 => true,
                    _ => return Err(EINVAL),
                };
                if idle != n.idle {
                    // Leaving idle goes back to the default weight, as on Linux.
                    n.idle = idle;
                    n.shares = if idle { IDLE_SHARES } else { NICE0_SHARES };
                }
            }
            "cpu.max" => {
                // "$MAX [$PERIOD]": the period stays as it was if not given.
                let mut words = text.split_whitespace();
                let quota = match words.next().ok_or(EINVAL)? {
                    "max" => None,
                    q => Some(q.parse::<u64>().map_err(|_| EINVAL)?),
                };
                let period = match words.next() {
                    Some(p) => p.parse::<u64>().map_err(|_| EINVAL)?,
                    None => n.bandwidth.period_us,
                };
                let burst = n.bandwidth.burst_us;
                set_bandwidth(&mut n.bandwidth, quota, period, burst)?;
                self.update_quota_gate();
            }
            "io.max" => return self.write_io_max(node, text),
            "io.weight" => return self.write_io_weight(node, text),
            "memory.max" => n.mem_max = mem_limit(text)?,
            "memory.high" => n.mem_high = mem_limit(text)?,
            "memory.low" => n.mem_low = mem_limit(text)?,
            "memory.min" => n.mem_min = mem_limit(text)?,
            "memory.oom.group" => {
                n.oom_group = match parse_int::<i32>(text)? {
                    0 => false,
                    1 => true,
                    _ => return Err(EINVAL),
                };
            }
            "cpu.max.burst" => {
                let burst: u64 = parse_int(text)?;
                let (quota, period) = (n.bandwidth.quota_us, n.bandwidth.period_us);
                set_bandwidth(&mut n.bandwidth, quota, period, burst)?;
            }
            // "domain" isn't something to write, as on Linux.
            "cgroup.type" if text == "threaded" => return self.enable_threaded(node),
            "cgroup.type" => return Err(EINVAL),
            f if KERNEL_FILES.contains(&f) => return Err(EOPNOTSUPP),
            _ => return Err(EACCES),
        }
        Ok(())
    }

    /// The disk `text` (`MAJ:MIN`) names: `EINVAL` if it isn't a device
    /// number, `ENODEV` if there's no such disk (a partition isn't one).
    fn disk_named(&self, text: &str) -> Result<u64, i32> {
        const ENODEV: i32 = 19;
        let (ma, mi) = text.split_once(':').ok_or(EINVAL)?;
        let ma: u32 = ma.parse().map_err(|_| EINVAL)?;
        let mi: u32 = mi.parse().map_err(|_| EINVAL)?;
        let disks = self.disks.as_ref().ok_or(ENODEV)?.lock().unwrap();
        disks.find(ma, mi).map(|d| d.dev).ok_or(ENODEV)
    }

    /// `io.max`: `MAJ:MIN` and `key=value` words (`rbps`, `wbps`, `riops`,
    /// `wiops`; a number or "max"), as Linux's `tg_set_limit` takes them.
    fn write_io_max(&mut self, node: u32, text: &str) -> Result<(), i32> {
        let mut words = text.split_whitespace();
        let dev = self.disk_named(words.next().ok_or(EINVAL)?)?;
        let n = self.nodes.get_mut(&node).ok_or(ENOENT)?;
        let mut v = n.io_max.get(&dev).copied().unwrap_or([None; 4]);
        for w in words {
            let (key, val) = w.split_once('=').ok_or(EINVAL)?;
            let val = match val {
                "max" => None,
                x => Some(x.parse::<u64>().map_err(|_| EINVAL)?),
            };
            if val == Some(0) {
                return Err(ERANGE);
            }
            let i = ["rbps", "wbps", "riops", "wiops"]
                .iter()
                .position(|k| *k == key)
                .ok_or(EINVAL)?;
            v[i] = if i >= 2 {
                val.map(|x| x.min(u64::from(u32::MAX)))
            } else {
                val
            };
        }
        if v.iter().all(Option::is_none) {
            n.io_max.remove(&dev);
        } else {
            n.io_max.insert(dev, v);
        }
        n.io_next.retain(|(d, _), _| *d != dev);
        self.io_limited = self.nodes.values().any(|n| !n.io_max.is_empty());
        Ok(())
    }

    /// `io.weight`: `[default] N` for the default, `MAJ:MIN N|default` for a
    /// disk, as Linux's `ioc_weight_write` takes it.
    fn write_io_weight(&mut self, node: u32, text: &str) -> Result<(), i32> {
        let weight = |t: &str| -> Result<u64, i32> {
            let w: u64 = t.parse().map_err(|_| EINVAL)?;
            if (1..=10_000).contains(&w) {
                Ok(w)
            } else {
                Err(EINVAL)
            }
        };
        let mut words = text.split_whitespace();
        let first = words.next().ok_or(EINVAL)?;
        if !first.contains(':') {
            let w = if first == "default" {
                weight(words.next().ok_or(EINVAL)?)?
            } else {
                weight(first)?
            };
            self.nodes.get_mut(&node).ok_or(ENOENT)?.io_weight = w;
            return Ok(());
        }
        let dev = self.disk_named(first)?;
        let w = match words.next().ok_or(EINVAL)? {
            "default" => None,
            t => Some(weight(t)?),
        };
        let n = self.nodes.get_mut(&node).ok_or(ENOENT)?;
        match w {
            Some(w) => n.io_weight_dev.insert(dev, w),
            None => n.io_weight_dev.remove(&dev),
        };
        Ok(())
    }

    /// Count an I/O of `bytes` by a task of cgroup `node` to disk `dev` (a
    /// read, or with `write` a write) in `io.stat` there and above, and hold
    /// it to the `io.max` limits on the way up, as Linux's `blk-throttle`
    /// does: each limit lets through what its rate allows since its I/O last
    /// caught up (reaching back at most Linux's `throtl_slice`, 100 ms). Returns until when
    /// the task is to wait for it, if at all.
    pub fn io_account(&mut self, node: u32, dev: u64, write: bool, bytes: u64) -> Option<u128> {
        let now = crate::clock::now_monotonic().as_nanos();
        let mut until: Option<u128> = None;
        let (bps, iops) = if write { (1, 3) } else { (0, 2) };
        for id in self.chain(node) {
            let Some(n) = self.nodes.get_mut(&id) else {
                continue;
            };
            let st = n.io_stat.entry(dev).or_default();
            st[usize::from(write)] += bytes;
            st[2 + usize::from(write)] += 1;
            if id == ROOT || !self.io_limited {
                continue;
            }
            let Some(limits) = n.io_max.get(&dev).copied() else {
                continue;
            };
            for (k, amount) in [(bps, u128::from(bytes)), (iops, 1)] {
                let Some(rate) = limits[k] else {
                    continue;
                };
                let next = n.io_next.entry((dev, k)).or_insert(0);
                let start = (*next).max(now.saturating_sub(IO_SLICE_NS));
                *next = start + amount * 1_000_000_000 / u128::from(rate);
                if *next > now {
                    until = until.max(Some(*next));
                }
            }
        }
        until.map(|t| t - now + crate::clock::now_unix().as_nanos())
    }

    /// Whether any cgroup has an `io.max` limit.
    #[must_use]
    pub fn io_limited(&self) -> bool {
        self.io_limited
    }

    /// Give the tree the VM's disks, which `io.max` names.
    pub fn attach_disks(&mut self, disks: &super::blockdev::SharedDisks) {
        self.disks = Some(disks.clone());
    }

    /// `io.stat`, `io.max` or `io.weight` of `n`.
    fn render_io(n: &Node, file: &str) -> String {
        use super::blockdev::{major, minor};
        let mut out = String::new();
        let mm = |d: u64| format!("{}:{}", major(d), minor(d));
        match file {
            "io.stat" => {
                for (&d, st) in &n.io_stat {
                    let _ = writeln!(
                        out,
                        "{} rbytes={} wbytes={} rios={} wios={} dbytes=0 dios=0",
                        mm(d),
                        st[0],
                        st[1],
                        st[2],
                        st[3]
                    );
                }
            }
            "io.max" => {
                for (&d, v) in &n.io_max {
                    let f = |x: Option<u64>| x.map_or_else(|| "max".to_string(), |x| x.to_string());
                    let _ = writeln!(
                        out,
                        "{} rbps={} wbps={} riops={} wiops={}",
                        mm(d),
                        f(v[0]),
                        f(v[1]),
                        f(v[2]),
                        f(v[3])
                    );
                }
            }
            _ => {
                let _ = writeln!(out, "default {}", n.io_weight);
                for (&d, w) in &n.io_weight_dev {
                    let _ = writeln!(out, "{} {w}", mm(d));
                }
            }
        }
        out
    }

    /// `+name`/`-name` words, space separated, all or nothing.
    fn write_subtree_control(&mut self, node: u32, text: &str) -> Result<(), i32> {
        let available = self.controller_bits(node);
        let old = self.nodes[&node].subtree;
        let (mut on, mut off) = (0u8, 0u8);
        for word in text.split_whitespace() {
            let (enable, name) = match word.split_at(1) {
                ("+", n) => (true, n),
                ("-", n) => (false, n),
                _ => return Err(EINVAL),
            };
            // A controller the kernel doesn't have is EINVAL; one it has, but
            // that the parent doesn't hand down, is ENOENT.
            let bit = CONTROLLERS
                .iter()
                .find(|(n, _)| *n == name)
                .ok_or(EINVAL)?
                .1;
            if enable {
                if available & bit == 0 {
                    return Err(ENOENT);
                }
                on |= bit;
                off &= !bit;
            } else {
                off |= bit;
                on &= !bit;
            }
        }
        let enabling = on & !old;
        let disabling = off & old;
        // Linux's `cgroup_vet_subtree_control_enable`: nothing in a domain
        // that can't hold resources; no domain controller inside a threaded
        // subtree; and no internal processes: a domain controller can't be
        // enabled where there are tasks; a threaded one may be, where the
        // cgroup is or could be a thread root.
        if enabling != 0 {
            if !self.is_valid_domain(self.domain_of(node)) {
                return Err(EOPNOTSUPP);
            }
            if node != ROOT {
                let threaded = self.is_threaded(node);
                let exempt = if enabling & !THREADED != 0 {
                    if threaded || self.is_thread_root(node) {
                        return Err(EOPNOTSUPP);
                    }
                    false
                } else {
                    threaded || self.can_be_thread_root(node)
                };
                if !exempt && self.has_own_tasks(node) {
                    return Err(EBUSY);
                }
            }
        }
        // A child still hands a controller on to its own children.
        let children: Vec<u32> = self.nodes[&node].children.values().copied().collect();
        if children.iter().any(|c| {
            self.nodes
                .get(c)
                .is_some_and(|c| c.subtree & disabling != 0)
        }) {
            return Err(EBUSY);
        }
        // The children's state for a controller that comes or goes starts
        // over, as Linux makes it anew.
        for c in children {
            if let Some(c) = self.nodes.get_mut(&c) {
                c.reset(enabling | disabling);
            }
        }
        self.nodes.get_mut(&node).expect("checked").subtree = (old | enabling) & !disabling;
        self.update_quota_gate();
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
            "cgroup.controllers" => format!("{}\n", self.controllers(node)),
            "cgroup.subtree_control" => format!("{}\n", Self::names(n.subtree)),
            "cgroup.procs" => list(&self.view.procs),
            "cgroup.threads" => list(&self.view.threads),
            "cgroup.events" => format!(
                "populated {}\nfrozen {}\n",
                u8::from(n.live > 0),
                u8::from(self.frozen(node))
            ),
            "cgroup.type" => format!("{}\n", self.type_name(node)),
            "cgroup.max.depth" => limit(n.max_depth),
            "cgroup.max.descendants" => limit(n.max_descendants),
            // Linux's layout: the controller counts are how many cgroups from
            // here down have that controller (`css`), this one included.
            "cgroup.stat" => {
                let mut out = format!("nr_descendants {}\n", self.descendants(node));
                for &(name, bit) in CONTROLLERS {
                    let count = if self.has(node, bit) {
                        self.subtree(node)
                            .iter()
                            .filter(|&&c| self.has(c, bit))
                            .count()
                    } else {
                        0
                    };
                    let _ = writeln!(out, "nr_subsys_{name} {count}");
                }
                out.push_str("nr_dying_descendants 0\n");
                for (name, _) in CONTROLLERS {
                    let _ = writeln!(out, "nr_dying_subsys_{name} 0");
                }
                out
            }
            "cgroup.freeze" => format!("{}\n", u8::from(n.freeze)),
            "cpu.stat" => {
                let usec = self
                    .view
                    .cpu_ns
                    .get(&node)
                    .copied()
                    .unwrap_or(n.dead_cpu_ns)
                    / 1000;
                let mut out = format!("usage_usec {usec}\nuser_usec {usec}\nsystem_usec 0\n");
                if self.has(node, CPU) {
                    let b = &n.bandwidth;
                    let _ = write!(
                        out,
                        "nr_periods {}\nnr_throttled {}\nthrottled_usec {}\nnr_bursts {}\n\
                         burst_usec {}\n",
                        b.nr_periods,
                        b.nr_throttled,
                        b.throttled_ns(now_ns()) / NS_PER_US,
                        b.nr_bursts,
                        b.burst_ns / NS_PER_US
                    );
                }
                out
            }
            "cpu.stat.local" if self.has(node, CPU) => format!(
                "throttled_usec {}\n",
                n.bandwidth.throttled_ns(now_ns()) / NS_PER_US
            ),
            "cpu.weight" => format!(
                "{}\n",
                ((n.shares * 100 + NICE0_SHARES / 2) / NICE0_SHARES).clamp(1, 10_000)
            ),
            "cpu.weight.nice" => {
                // The nice level whose weight is closest.
                let nice = (-20..=19)
                    .min_by_key(|&l| nice_shares(l).abs_diff(n.shares))
                    .unwrap_or(0);
                format!("{nice}\n")
            }
            "cpu.idle" => format!("{}\n", u8::from(n.idle)),
            "cpu.max" => {
                let b = &n.bandwidth;
                match b.quota_us {
                    Some(q) => format!("{q} {}\n", b.period_us),
                    None => format!("max {}\n", b.period_us),
                }
            }
            "cpu.max.burst" => format!("{}\n", n.bandwidth.burst_us),
            "io.stat" | "io.max" | "io.weight" => Self::render_io(n, file),
            "memory.current" => format!("{}\n", self.mem_use(node).current),
            "memory.peak" => format!("{}\n", self.mem_use(node).peak),
            "memory.max" => mem_bytes(n.mem_max),
            "memory.high" => mem_bytes(n.mem_high),
            "memory.low" => mem_bytes(n.mem_low),
            "memory.min" => mem_bytes(n.mem_min),
            "memory.oom.group" => format!("{}\n", u8::from(n.oom_group)),
            "memory.events" => self.mem_events(&self.subtree(node)),
            "memory.events.local" => self.mem_events(&[node]),
            "memory.stat" => {
                // What's charged here is the page cache's file pages and a
                // process's own memory (and its page tables).
                let u = self.mem_use(node);
                let anon = u.current - u.file;
                let mut out = format!("anon {anon}\nfile {}\n", u.file);
                for key in [
                    "kernel",
                    "kernel_stack",
                    "pagetables",
                    "sec_pagetables",
                    "percpu",
                    "sock",
                    "vmalloc",
                    "shmem",
                    "file_mapped",
                    "file_dirty",
                    "file_writeback",
                    "swapcached",
                    "anon_thp",
                    "file_thp",
                    "shmem_thp",
                    "inactive_anon",
                ] {
                    let _ = writeln!(out, "{key} 0");
                }
                let _ = writeln!(out, "active_anon {anon}");
                let _ = writeln!(out, "inactive_file {}", u.file);
                let _ = writeln!(out, "active_file 0");
                out
            }
            "pids.max" => limit(n.pids_max),
            "pids.current" => format!("{}\n", n.tasks),
            "pids.events" => format!("max {}\n", n.pids_events),
            _ => String::new(),
        }
    }

    fn mem_use(&self, node: u32) -> MemUse {
        self.view.memory.get(&node).copied().unwrap_or_default()
    }

    /// `memory.events` for `nodes` together: one cgroup's own events, or a
    /// subtree's.
    fn mem_events(&self, nodes: &[u32]) -> String {
        let (mut low, mut high, mut max, mut oom) = (0, 0, 0, [0u64; 3]);
        for n in nodes {
            let u = self.mem_use(*n);
            low += u.low_events;
            high += u.high_events;
            max += u.max_events;
            if let Some(x) = self.nodes.get(n) {
                for (total, e) in oom.iter_mut().zip(x.oom_events) {
                    *total += e;
                }
            }
        }
        format!(
            "low {low}\nhigh {high}\nmax {max}\noom {}\noom_kill {}\noom_group_kill {}\n",
            oom[0], oom[1], oom[2]
        )
    }

    fn inode(node: u32, file: Option<&str>) -> u64 {
        const BASE: u64 = 0x6367_0000_0000;
        let slot = file.map_or(0, |f| {
            CORE_FILES
                .iter()
                .chain(PIDS_FILES)
                .chain(CPU_FILES)
                .chain(MEMORY_FILES)
                .chain(IO_FILES)
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
    fn attach_disks(&mut self, disks: &super::blockdev::SharedDisks) {
        self.tree.lock().unwrap().attach_disks(disks);
    }

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
                dev: 0,
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
                    dev: 0,
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
        // All of a threaded subtree's processes belong to its thread root.
        if file == "cgroup.procs" && tree.is_threaded(node) {
            return Err(errno(EOPNOTSUPP));
        }
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

    // cgroup v2 has no rename (only v1 does): kernfs refuses it.
    fn rename(&mut self, _from: &str, _to: &str) -> io::Result<()> {
        Err(errno(EPERM))
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
        assert_eq!(read(&mut fs, "cgroup.controllers"), "cpu io memory pids\n");
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
    fn the_cpu_controller_weights_and_bandwidth() {
        let tree = CgroupTree::shared();
        let mut t = tree.lock().unwrap();
        let gate = t.quota_gate();
        let a = t.mkdir(ROOT, "a", 0, 0).unwrap();
        assert!(!t.files(a).iter().any(|(f, _)| *f == "cpu.max"));
        t.write_file(ROOT, "cgroup.subtree_control", "+cpu +pids")
            .unwrap();
        assert_eq!(t.render(ROOT, "cgroup.subtree_control"), "cpu pids\n");
        assert_eq!(t.render(a, "cpu.weight"), "100\n");
        assert_eq!(t.render(a, "cpu.max"), "max 100000\n");
        // Weights: the scale is relative to a task in the root.
        t.write_file(a, "cpu.weight", "200").unwrap();
        assert_eq!(t.sched_state(a, 0).scale, UNSCALED / 2);
        assert_eq!(t.write_file(a, "cpu.weight", "0"), Err(ERANGE));
        assert_eq!(t.write_file(a, "cpu.weight", "x"), Err(EINVAL));
        t.write_file(a, "cpu.weight.nice", "0").unwrap();
        assert_eq!(t.render(a, "cpu.weight"), "100\n");
        t.write_file(a, "cpu.weight.nice", "5").unwrap();
        assert_eq!(t.render(a, "cpu.weight.nice"), "5\n");
        t.write_file(a, "cpu.idle", "1").unwrap();
        assert_eq!(t.render(a, "cpu.weight"), "1\n");
        assert_eq!(t.write_file(a, "cpu.weight", "50"), Err(EINVAL), "idle");
        t.write_file(a, "cpu.idle", "0").unwrap();
        assert_eq!(t.render(a, "cpu.weight"), "100\n", "back to the default");
        // Bandwidth: 10 ms every 50 ms.
        assert_eq!(t.write_file(a, "cpu.max", "500 100000"), Err(EINVAL));
        assert_eq!(t.write_file(a, "cpu.max", "10000 2000000"), Err(EINVAL));
        assert_eq!(t.write_file(ROOT, "cpu.max", "10000"), Err(ENOENT));
        t.write_file(a, "cpu.max", "10000 50000").unwrap();
        assert_eq!(gate.load(Ordering::Relaxed), 1);
        assert_eq!(t.render(a, "cpu.max"), "10000 50000\n");
        assert_eq!(t.write_file(a, "cpu.max.burst", "20000"), Err(EINVAL));
        let ms = 1_000_000u128;
        let start = 1000 * ms;
        assert_eq!(t.charge_cpu(a, 4_000_000, start), (None, None));
        let (until, newly) = t.charge_cpu(a, 6_000_000, start + 5 * ms);
        assert_eq!((until, newly), (Some(start + 50 * ms), Some(a)));
        assert_eq!(t.sched_state(a, start + 6 * ms).throttled_until, until);
        assert_eq!(t.sched_state(a, start + 50 * ms).throttled_until, None);
        // The next period has its quota again.
        assert_eq!(t.charge_cpu(a, 1_000_000, start + 51 * ms), (None, None));
        let stat = t.render(a, "cpu.stat");
        assert!(
            stat.contains("nr_periods 2\nnr_throttled 1\nthrottled_usec 45000\n"),
            "{stat}"
        );
        // A child's time counts against its parent's quota.
        let b = t.mkdir(a, "b", 0, 0).unwrap();
        t.write_file(a, "cgroup.subtree_control", "+cpu").unwrap();
        let (until, newly) = t.charge_cpu(b, 20_000_000, start + 52 * ms);
        assert_eq!((until, newly), (Some(start + 100 * ms), Some(a)));
        // Taking the controller away from the children resets them.
        t.write_file(a, "cgroup.subtree_control", "-cpu").unwrap();
        t.write_file(ROOT, "cgroup.subtree_control", "-cpu")
            .unwrap();
        assert_eq!(gate.load(Ordering::Relaxed), 0);
        t.write_file(ROOT, "cgroup.subtree_control", "+cpu")
            .unwrap();
        assert_eq!(t.render(a, "cpu.max"), "max 100000\n");
    }

    #[test]
    fn the_memory_controller_is_a_domain_one() {
        let tree = CgroupTree::shared();
        let mut t = tree.lock().unwrap();
        let a = t.mkdir(ROOT, "a", 0, 0).unwrap();
        let b = t.mkdir(a, "b", 0, 0).unwrap();
        t.write_file(ROOT, "cgroup.subtree_control", "+memory +pids")
            .unwrap();
        t.migrate(&[(7, true)], a);
        // Not where there are tasks, unlike pids...
        assert_eq!(
            t.write_file(a, "cgroup.subtree_control", "+memory"),
            Err(EBUSY)
        );
        t.write_file(a, "cgroup.subtree_control", "+pids").unwrap();
        t.migrate(&[(7, true)], ROOT);
        t.write_file(a, "cgroup.subtree_control", "+memory")
            .unwrap();
        // ...and then no task goes in, even with no child populated.
        assert_eq!(t.check_attach(a), Err(EBUSY));
        assert_eq!(t.check_attach(b), Ok(()));
        // Sizes: memparse suffixes, whole pages.
        assert_eq!(mem_limit("16M"), Ok(Some(4096)));
        assert_eq!(mem_limit("0x2000"), Ok(Some(2)));
        assert_eq!(mem_limit("4097"), Ok(Some(1)));
        assert_eq!(mem_limit("max"), Ok(None));
        assert_eq!(mem_limit("1Q"), Err(EINVAL));
        // A kill in b, out of a's memory: oom.group on b takes b whole.
        t.nodes.get_mut(&a).unwrap().memcg = Some(1);
        t.nodes.get_mut(&b).unwrap().memcg = Some(2);
        assert_eq!(t.memcg_of(b), 2);
        assert_eq!(t.record_oom(b, 1), None);
        t.write_file(b, "memory.oom.group", "1").unwrap();
        assert_eq!(t.record_oom(b, 1), Some(b));
        assert_eq!(
            t.render(a, "memory.events.local"),
            "low 0\nhigh 0\nmax 0\noom 2\noom_kill 2\noom_group_kill 1\n"
        );
    }

    #[test]
    fn cpu_weights_are_shared_by_the_group() {
        let tree = CgroupTree::shared();
        let mut t = tree.lock().unwrap();
        t.write_file(ROOT, "cgroup.subtree_control", "+cpu")
            .unwrap();
        assert!(t.weights_active());
        let a = t.mkdir(ROOT, "a", 0, 0).unwrap();
        let b = t.mkdir(ROOT, "b", 0, 0).unwrap();
        t.write_file(a, "cpu.weight", "200").unwrap();
        t.migrate(&[(1, true), (2, true)], a);
        t.migrate(&[(3, true)], b);
        let w = u128::from(NICE0_SHARES);
        // Two tasks splitting weight 200 run as fast as one at 100: the
        // scheduler gives all three the same third.
        assert_eq!(
            t.group_scales(&[(1, w), (2, w), (3, w), (4, w)]),
            [UNSCALED; 4]
        );
        // Alone in a, a task has all of a's 200: half the virtual runtime.
        assert_eq!(t.group_scales(&[(1, w), (3, w)]), [UNSCALED / 2, UNSCALED]);
        // Nested: b's share is split again by its child.
        t.write_file(b, "cgroup.subtree_control", "+cpu").unwrap();
        let c = t.mkdir(b, "c", 0, 0).unwrap();
        t.migrate(&[(3, true)], c);
        t.migrate(&[(5, true)], b);
        t.write_file(c, "cpu.weight", "300").unwrap();
        // In b: task 5 (1024) against c (3072); c's 3072 holds task 3 (1024).
        let s = t.group_scales(&[(3, w), (5, w)]);
        assert_eq!(
            s,
            [UNSCALED / 3 * 4, UNSCALED * 4],
            "4/3, rounded down step by step"
        );
    }

    #[test]
    fn threaded_subtrees_follow_linux() {
        let tree = CgroupTree::shared();
        let mut t = tree.lock().unwrap();
        t.write_file(ROOT, "cgroup.subtree_control", "+cpu +pids +memory")
            .unwrap();
        let d = t.mkdir(ROOT, "d", 0, 0).unwrap();
        t.write_file(d, "cgroup.subtree_control", "+memory")
            .unwrap();
        let a = t.mkdir(d, "a", 0, 0).unwrap();
        assert_eq!(
            t.write_file(a, "cgroup.type", "threaded"),
            Err(EOPNOTSUPP),
            "d has a domain controller enabled"
        );
        t.write_file(d, "cgroup.subtree_control", "-memory")
            .unwrap();
        let b = t.mkdir(d, "b", 0, 0).unwrap();
        t.write_file(a, "cgroup.type", "threaded").unwrap();
        assert_eq!(t.type_name(a), "threaded");
        assert_eq!(t.type_name(d), "domain threaded");
        assert_eq!(t.type_name(b), "domain invalid");
        assert_eq!(t.check_attach(b), Err(EOPNOTSUPP));
        assert_eq!(t.domain_of(a), d);
        assert_eq!(
            t.write_file(d, "cgroup.subtree_control", "+memory"),
            Err(EOPNOTSUPP)
        );
        t.write_file(d, "cgroup.subtree_control", "+cpu").unwrap();
        assert_eq!(t.controllers(a), "cpu");
        // Threads move within the domain, not out of it.
        t.migrate(&[(7, true), (8, true)], d);
        assert_eq!(t.check_thread_attach(8, a), Ok(()));
        t.migrate(&[(8, true)], a);
        assert_eq!(t.check_thread_attach(8, ROOT), Err(EOPNOTSUPP));
        assert_eq!(t.check_attach(d), Ok(()), "a thread root takes tasks");
        // Populated, it can't become threaded itself.
        assert_eq!(t.write_file(d, "cgroup.type", "threaded"), Err(EOPNOTSUPP));
        t.migrate(&[(7, true), (8, true)], ROOT);
        t.rmdir(a).unwrap();
        assert_eq!(t.type_name(d), "domain");
        assert_eq!(t.type_name(b), "domain");
    }

    #[test]
    fn no_internal_processes_and_limits_on_the_tree() {
        let tree = CgroupTree::shared();
        let mut t = tree.lock().unwrap();
        let a = t.mkdir(ROOT, "a", 0, 0).unwrap();
        t.migrate(&[(5, true)], a);
        assert_eq!(
            t.write_file(a, "cgroup.subtree_control", "+pids"),
            Err(ENOENT),
            "the root doesn't hand it down yet"
        );
        t.write_file(ROOT, "cgroup.subtree_control", "+pids")
            .unwrap();
        // pids is threaded: it may be enabled where there are tasks...
        t.write_file(a, "cgroup.subtree_control", "+pids").unwrap();
        assert_eq!(
            t.write_file(a, "cgroup.subtree_control", "+rdma"),
            Err(EINVAL)
        );
        assert_eq!(
            t.write_file(a, "cgroup.subtree_control", "+cpu"),
            Err(ENOENT),
            "not handed down"
        );
        // ...but then nothing more goes into a once a child is populated.
        let kid = t.mkdir(a, "kid", 0, 0).unwrap();
        t.migrate(&[(6, true)], kid);
        assert_eq!(t.check_attach(a), Err(EBUSY));
        // With tasks and a threaded controller enabled, a is a thread root,
        // under which a domain child can't hold processes.
        assert_eq!(t.type_name(a), "domain threaded");
        assert_eq!(t.check_attach(kid), Err(EOPNOTSUPP));
        t.write_file(a, "cgroup.subtree_control", "-pids").unwrap();
        assert_eq!(t.check_attach(a), Ok(()));
        assert_eq!(t.check_attach(kid), Ok(()));
        t.migrate(&[(6, true)], ROOT);
        t.rmdir(kid).unwrap();
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
