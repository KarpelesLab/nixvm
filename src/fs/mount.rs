//! The mount table: composes [`MountFs`] backends by longest-prefix match.

use std::collections::BTreeMap;
use std::io;
use std::sync::{Arc, Mutex, MutexGuard};

use super::{Attrs, DirEntry, MountFs, NodeKind};

/// Most symlinks followed while resolving one path's directories (Linux's
/// `MAXSYMLINKS`).
const SYMLINK_MAX: usize = 40;

#[derive(Clone)]
// The flags are independent properties of a mount, as Linux's are.
#[allow(clippy::struct_excessive_bools)]
struct Mount {
    /// Absolute mount point, e.g. "/", "/work", "/proc".
    point: String,
    /// The filesystem, which a bind mount or another namespace's copy of this
    /// mount shares.
    fs: Arc<Mutex<Box<dyn MountFs>>>,
    /// Where in that filesystem this mount starts: empty for its root, and the
    /// source directory (or file) of a bind mount.
    subdir: String,
    /// Mounted at runtime by the guest (`mount(2)`), so `umount(2)` can
    /// remove it; the embedder's mounts are permanent.
    guest: bool,
    /// Writes through this mount fail with `EROFS`.
    read_only: bool,
    /// Came from a more privileged user namespace: can't be undone individually.
    locked: bool,
    /// Which pivot this mount was the old root's side of, 0 if none. After
    /// `pivot_root(".", ".")` an old root is stacked on the new one, and detaching
    /// it must take only its own mounts.
    pivot_side: u32,
    /// Its peer group if shared (mounts and unmounts below it happen on every
    /// peer, in any namespace), 0 if not.
    peer: u32,
    /// The peer group it's a slave of, 0 if none: what happens below that
    /// group's mounts happens below this one, but not the other way round.
    master: u32,
    /// Can't be bind mounted (`MS_UNBINDABLE`).
    unbindable: bool,
}

impl std::fmt::Debug for Mount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mount")
            .field("point", &self.point)
            .field("subdir", &self.subdir)
            .field("guest", &self.guest)
            .field("read_only", &self.read_only)
            .finish_non_exhaustive()
    }
}

impl Mount {
    fn new(point: String, fs: Box<dyn MountFs>, guest: bool) -> Self {
        Self {
            point,
            fs: Arc::new(Mutex::new(fs)),
            subdir: String::new(),
            guest,
            read_only: false,
            locked: false,
            pivot_side: 0,
            peer: 0,
            master: 0,
            unbindable: false,
        }
    }

    /// The path in this mount's filesystem of `rel`, a path below its mount point.
    fn fs_path(&self, rel: &str) -> String {
        match (self.subdir.is_empty(), rel.is_empty()) {
            (true, _) => rel.to_string(),
            (false, true) => self.subdir.clone(),
            (false, false) => format!("{}/{rel}", self.subdir),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Box<dyn MountFs>> {
        self.fs.lock().unwrap()
    }
}

/// Resolves guest paths across a set of mounted backends.
#[derive(Debug, Default)]
pub struct MountTable {
    /// The mounts of the namespace in [`Self::current`]. Another namespace's are
    /// parked in [`Self::parked`], and swapped in by [`Self::enter`].
    mounts: Vec<Mount>,
    current: u32,
    parked: BTreeMap<u32, Vec<Mount>>,
    /// The caller's root directory (`chroot`), which absolute symlinks resolve
    /// against and `..` can't climb above. Empty means `/`.
    root: String,
    /// How many times `pivot_root` has run, to label each one's old mounts.
    pivots: u32,
    /// How many mounts have been detached, to give each a place of its own.
    detached: u32,
    /// The last peer group number handed out.
    groups: u32,
    /// What every mounted filesystem charges its in-memory contents to.
    quota: Option<std::sync::Arc<crate::limits::Quota>>,
    /// What every mounted filesystem charges growth on the host's disk to.
    disk_quota: Option<std::sync::Arc<crate::limits::Quota>>,
    /// The VM's disks (see [`super::blockdev`]).
    disks: super::blockdev::SharedDisks,
    /// The device each mounted filesystem is (keyed by [`fs_key`]): its disk,
    /// or an anonymous device.
    devs: BTreeMap<usize, u64>,
}

/// What tells one mounted filesystem from another: every mount of it (a bind
/// mount, a copy in another namespace) shares the one instance.
fn fs_key(m: &Mount) -> usize {
    Arc::as_ptr(&m.fs).cast::<()>() as usize
}

impl MountTable {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Answer calls from here on for a task in mount namespace `ns` whose root
    /// directory is `root`. Cheap when nothing changes, as it almost never does.
    pub fn enter(&mut self, ns: u32, root: &str) {
        if self.current != ns {
            let incoming = self.parked.remove(&ns).unwrap_or_default();
            let outgoing = std::mem::replace(&mut self.mounts, incoming);
            self.parked.insert(self.current, outgoing);
            self.current = ns;
        }
        let root = if root == "/" { "" } else { root };
        if self.root != root {
            root.clone_into(&mut self.root);
        }
    }

    /// Start mount namespace `new` as a copy of `from`. The two share filesystems,
    /// but mounting or unmounting in one no longer shows in the other.
    ///
    /// With `lock_inherited`, the copy is going to a less privileged user namespace,
    /// which gets these mounts as they are: it can stack on them, but can't uncover
    /// what's beneath one or make a read-only one writable.
    pub fn copy_namespace(&mut self, from: u32, new: u32, lock_inherited: bool) {
        let mut copy = if from == self.current {
            self.mounts.clone()
        } else {
            self.parked.get(&from).cloned().unwrap_or_default()
        };
        if lock_inherited {
            // The root itself isn't locked, so that it can be pivoted away from.
            for m in copy.iter_mut().filter(|m| m.point != "/") {
                m.locked = true;
            }
            // And what was shared reaches it only one way: it's a slave there.
            for m in &mut copy {
                if m.peer != 0 {
                    m.master = m.peer;
                    m.peer = 0;
                }
            }
        }
        self.parked.insert(new, copy);
    }

    /// Forget the mounts of namespaces not in `in_use`.
    pub fn collect_namespaces(&mut self, in_use: &[u32]) {
        self.parked.retain(|ns, _| *ns == 0 || in_use.contains(ns));
    }

    /// How many mount namespaces exist.
    #[must_use]
    pub fn namespace_count(&self) -> usize {
        self.parked.len() + 1
    }

    /// Bind `source` (a directory or file, as seen now) onto `target`. With
    /// `recursive`, mounts below `source` come along, as `MS_REC` asks.
    pub fn bind(&mut self, source: &str, target: &str, recursive: bool) -> io::Result<()> {
        let source = self.canonical(source);
        let target = normalize(target);
        let i = self.best_mount(&source).ok_or_else(enoent)?;
        if self.mounts[i].unbindable {
            return Err(einval());
        }
        // A bind mount of a shared mount is its peer, and of a slave a slave of
        // the same master (Linux's clone_mnt), so it keeps getting what they get.
        let mut new = vec![Mount {
            point: target.clone(),
            subdir: self.mounts[i].fs_path(&relative_to(&source, &self.mounts[i].point)),
            guest: true,
            locked: false,
            ..self.mounts[i].clone()
        }];
        if recursive {
            for m in &self.mounts {
                if m.point != source && path_has_prefix(&m.point, &source) && !m.unbindable {
                    new.push(Mount {
                        point: normalize(&format!("{target}/{}", relative_to(&m.point, &source))),
                        guest: true,
                        locked: false,
                        ..m.clone()
                    });
                }
            }
        }
        self.attach(new);
        Ok(())
    }

    /// Move the mount at `from`, and any below it, to `to` (`MS_MOVE`).
    pub fn move_mount(&mut self, from: &str, to: &str) -> io::Result<()> {
        let from = normalize(from);
        let to = normalize(to);
        if !self.mounts.iter().any(|m| m.point == from && !m.locked) {
            return Err(einval());
        }
        for m in &mut self.mounts {
            if path_has_prefix(&m.point, &from) {
                m.point = normalize(&format!("{to}/{}", relative_to(&m.point, &from)));
            }
        }
        Ok(())
    }

    /// Set whether writes through the mount at `point` are refused.
    pub fn set_read_only(&mut self, point: &str, read_only: bool) -> io::Result<()> {
        let point = normalize(point);
        let m = self.mounts.iter_mut().rfind(|m| m.point == point);
        let m = m.ok_or_else(einval)?;
        if m.locked && m.read_only && !read_only {
            return Err(io::Error::from_raw_os_error(1)); // EPERM
        }
        m.read_only = read_only;
        Ok(())
    }

    /// Whether a mount sits at exactly `point`.
    #[must_use]
    pub fn is_mount_point(&self, point: &str) -> bool {
        let point = normalize(point);
        self.mounts.iter().any(|m| m.point == point)
    }

    /// `pivot_root`: make `new_root` the root, moving the old root to `put_old`
    /// (which must be at or below `new_root`). Every path in this namespace's view
    /// changes; returns how to translate one, for the kernel's own stored paths.
    pub fn pivot_root(&mut self, new_root: &str, put_old: &str) -> io::Result<PivotMap> {
        let new_root = normalize(new_root);
        let put_old = normalize(put_old);
        if !self.is_mount_point(&new_root)
            || !path_has_prefix(&put_old, &new_root)
            || new_root == "/"
        {
            return Err(einval());
        }
        let map = PivotMap {
            new_root: new_root.clone(),
            put_old: normalize(&format!("/{}", relative_to(&put_old, &new_root))),
        };
        self.pivots += 1;
        // The new root is the mount on top at that path, with what's been mounted
        // on it since. Anything earlier at or below the path is hidden beneath it,
        // and goes with the old root.
        let top = self
            .mounts
            .iter()
            .rposition(|m| m.point == new_root)
            .ok_or_else(einval)?;
        let (mut kept, mut old) = (Vec::new(), Vec::new());
        for (i, mut m) in std::mem::take(&mut self.mounts).into_iter().enumerate() {
            if i >= top && path_has_prefix(&m.point, &new_root) {
                m.point = normalize(&format!("/{}", relative_to(&m.point, &new_root)));
                kept.push(m);
            } else {
                m.point = normalize(&format!("{}/{}", map.put_old, m.point));
                m.pivot_side = self.pivots;
                old.push(m);
            }
        }
        // The old root goes on top, which is what "pivot_root(".", ".")" relies on.
        kept.append(&mut old);
        self.mounts = kept;
        Ok(map)
    }

    /// The mounts as `/proc/self/mountinfo` shows them to a task with the current
    /// root.
    #[must_use]
    pub fn listing(&self) -> Vec<MountEntry> {
        self.mounts
            .iter()
            .filter(|m| detached_root(&m.point).is_none())
            .filter(|m| self.root.is_empty() || path_has_prefix(&m.point, &self.root))
            .map(|m| (m, m.lock().fs_type().to_string()))
            .map(|(m, fs_type)| MountEntry {
                point: if self.root.is_empty() {
                    m.point.clone()
                } else {
                    normalize(&format!("/{}", relative_to(&m.point, &self.root)))
                },
                subdir: m.subdir.clone(),
                read_only: m.read_only,
                fs_type,
                dev: self.devs.get(&fs_key(m)).copied().unwrap_or(1),
                source: self.source_of(m),
                peer: m.peer,
                master: m.master,
                unbindable: m.unbindable,
            })
            .collect()
    }

    /// A name for what `abs_path` refers to whichever namespace is looking: the
    /// filesystem and the path inside it. Two paths with the same identity are the
    /// same file, however they're mounted.
    pub fn identity(&mut self, abs_path: &str) -> String {
        let abs_path = self.canonical(abs_path);
        match self.best_mount(&abs_path) {
            Some(i) => {
                let m = &self.mounts[i];
                let fs = Arc::as_ptr(&m.fs).cast::<()>() as usize;
                format!("{fs:x}:{}", m.fs_path(&relative_to(&abs_path, &m.point)))
            }
            None => abs_path,
        }
    }

    /// The cgroup `abs_path` names in a `cgroup2` mount, with the file in it
    /// if it names one, and the hierarchy it's in; `None` anywhere else.
    pub fn cgroup_at(
        &mut self,
        abs_path: &str,
    ) -> Option<(super::cgroupfs::SharedTree, u32, Option<String>)> {
        let abs_path = self.canonical(abs_path);
        let i = self.best_mount(&abs_path)?;
        let m = &self.mounts[i];
        let rel = m.fs_path(&relative_to(&abs_path, &m.point));
        let mut fs = m.lock();
        let cg = fs.as_cgroupfs_mut()?;
        let (node, file) = cg.resolve(&rel)?;
        Some((cg.tree().clone(), node, file))
    }

    /// The type of the filesystem `abs_path` is on (`"tmpfs"`, `"proc"`, ...).
    pub fn fs_type_at(&mut self, abs_path: &str) -> Option<&'static str> {
        let abs_path = self.canonical(abs_path);
        let i = self.best_mount(&abs_path)?;
        Some(self.mounts[i].lock().fs_type())
    }

    /// The hierarchy a `cgroup2` mount shows, if there is one.
    #[must_use]
    pub fn cgroup_tree(&self) -> Option<super::cgroupfs::SharedTree> {
        self.mounts
            .iter()
            .find_map(|m| m.lock().as_cgroupfs_mut().map(|c| c.tree().clone()))
    }

    /// Run `f` on each mounted procfs, so the kernel can refresh its live
    /// per-process data before serving a `/proc` read.
    pub fn with_procfs(&mut self, mut f: impl FnMut(&mut super::ProcFs)) {
        for m in &self.mounts {
            if let Some(procfs) = m.lock().as_procfs_mut() {
                f(procfs);
            }
        }
    }

    /// Relative symlink `target`, found in directory `dir`. Worked out from inside
    /// the caller's root, so however many `..` it has, it can't climb out.
    #[must_use]
    pub fn join_relative(&self, dir: &str, target: &str) -> String {
        match dir.strip_prefix(self.root.as_str()) {
            Some(inside) if !self.root.is_empty() => self.join_root(&format!("{inside}/{target}")),
            _ => normalize(&format!("{dir}/{target}")),
        }
    }

    /// `abs_path`, read from the caller's root (`chroot`), in the namespace's terms.
    #[must_use]
    pub fn join_root(&self, abs_path: &str) -> String {
        normalize(&format!("{}/{}", self.root, normalize(abs_path)))
    }

    /// The mount owning `abs_path` and its path in that mount's filesystem, for a
    /// call that changes something: fails `EROFS` on a read-only mount.
    fn resolve_for_write(
        &mut self,
        abs_path: &str,
    ) -> io::Result<(MutexGuard<'_, Box<dyn MountFs>>, String)> {
        let abs_path = self.canonical(abs_path);
        let i = self.best_mount(&abs_path).ok_or_else(enoent)?;
        if self.mounts[i].read_only {
            return Err(io::Error::from_raw_os_error(30)); // EROFS
        }
        let rel = self.mounts[i].fs_path(&relative_to(&abs_path, &self.mounts[i].point));
        Ok((self.mounts[i].lock(), rel))
    }

    /// Charge every filesystem's in-memory contents to `quota`, now and for
    /// anything mounted later.
    pub fn attach_quota(&mut self, quota: &std::sync::Arc<crate::limits::Quota>) {
        for mount in &self.mounts {
            mount.lock().attach_quota(quota);
        }
        self.quota = Some(quota.clone());
    }

    /// Charge every filesystem's growth on the host's disk to `quota`, now and for
    /// anything mounted later.
    pub fn attach_disk_quota(&mut self, quota: &std::sync::Arc<crate::limits::Quota>) {
        for mount in &self.mounts {
            mount.lock().attach_disk_quota(quota);
        }
        self.disk_quota = Some(quota.clone());
    }

    /// Mount `fs` at absolute `point`. Deeper mount points win via
    /// longest-prefix resolution.
    ///
    /// `point` is normalized (see `normalize`) before being stored, so
    /// callers may pass e.g. `"/dev/"` or `"/dev/../dev"` and still get the
    /// expected `"/dev"` mount point.
    pub fn mount(&mut self, point: impl Into<String>, mut fs: Box<dyn MountFs>) {
        if let Some(quota) = &self.quota {
            fs.attach_quota(quota);
        }
        if let Some(quota) = &self.disk_quota {
            fs.attach_disk_quota(quota);
        }
        let m = Mount::new(normalize(&point.into()), fs, false);
        self.register(&m);
        self.mounts.push(m);
    }

    /// Give a newly mounted filesystem its device: the disk it registers, if
    /// it's on one, or a fresh anonymous device.
    fn register(&mut self, m: &Mount) {
        if self.devs.contains_key(&fs_key(m)) {
            return;
        }
        let mut fs = m.lock();
        fs.attach_disks(&self.disks);
        let dev = fs
            .own_dev()
            .unwrap_or_else(|| self.disks.lock().unwrap().anon());
        drop(fs);
        self.devs.insert(fs_key(m), dev);
    }

    /// Where mount `m` is mounted from, as `mountinfo` says.
    fn source_of(&self, m: &Mount) -> String {
        let dev = self.devs.get(&fs_key(m)).copied().unwrap_or(1);
        let disks = self.disks.lock().unwrap();
        disks.list().iter().find(|d| d.dev == dev).map_or_else(
            || m.lock().fs_type().to_string(),
            |d| format!("/dev/{}", d.name),
        )
    }

    /// The VM's disks.
    #[must_use]
    pub fn disks(&self) -> super::blockdev::SharedDisks {
        self.disks.clone()
    }

    /// The device of mount `i`'s filesystem.
    fn dev_of(&self, i: usize) -> u64 {
        self.devs
            .get(&fs_key(&self.mounts[i]))
            .copied()
            .unwrap_or(1)
    }

    /// The disk holding `abs_path`'s contents, if they're on one (see
    /// [`MountFs::data_dev`]).
    pub fn data_dev(&mut self, abs_path: &str) -> Option<u64> {
        let (mut fs, rel) = self.resolve(abs_path)?;
        fs.data_dev(&rel)
    }

    /// Mount `fs` at `point` on the guest's behalf (`mount(2)`); it stacks
    /// over whatever was there and can be removed with [`Self::unmount`].
    pub fn mount_guest(&mut self, point: &str, mut fs: Box<dyn MountFs>) {
        if let Some(quota) = &self.quota {
            fs.attach_quota(quota);
        }
        if let Some(quota) = &self.disk_quota {
            fs.attach_disk_quota(quota);
        }
        let m = Mount::new(normalize(point), fs, true);
        self.register(&m);
        self.attach(vec![m]);
    }

    /// Add `tree` (a mount, then any below it) where its first mount says, and
    /// propagate it: under a shared mount the tree becomes shared, and a copy
    /// of it goes under each of that mount's peers and their slaves, in every
    /// namespace (Linux's `propagate_mnt`).
    fn attach(&mut self, mut tree: Vec<Mount>) {
        let Some(top_point) = tree.first().map(|m| m.point.clone()) else {
            return;
        };
        let parent = self.best_mount(&top_point).map(|i| self.mounts[i].clone());
        let Some(parent) = parent.filter(|p| p.peer != 0) else {
            self.mounts.extend(tree);
            return;
        };
        for m in &mut tree {
            if m.peer == 0 {
                m.peer = self.fresh_group();
            }
        }
        // Where the tree sits in the parent's filesystem.
        let at = parent.fs_path(&relative_to(&top_point, &parent.point));
        // Each step: a peer group to copy into, and the peer group and master
        // each tree mount's copies there get.
        let groups: Vec<u32> = tree.iter().map(|m| m.peer).collect();
        let mut queue = vec![(parent.peer, groups, vec![0u32; tree.len()])];
        let mut seen = std::collections::BTreeSet::new();
        let mut copies: Vec<(u32, Mount)> = Vec::new();
        while let Some((group, peers, masters)) = queue.pop() {
            if !seen.insert(group) {
                continue;
            }
            for (ns, target) in self.all_mounts() {
                let is_source = ns == self.current && target.point == parent.point;
                let place = |copies: &mut Vec<(u32, Mount)>, peers: &[u32], masters: &[u32]| {
                    let Some(at_target) = path_strip(&at, &target.subdir) else {
                        return;
                    };
                    let point = normalize(&format!("{}/{at_target}", target.point));
                    for (k, m) in tree.iter().enumerate() {
                        copies.push((
                            ns,
                            Mount {
                                point: normalize(&format!(
                                    "{point}/{}",
                                    relative_to(&m.point, &top_point)
                                )),
                                guest: true,
                                locked: false,
                                pivot_side: target.pivot_side,
                                peer: peers[k],
                                master: masters[k],
                                ..m.clone()
                            },
                        ));
                    }
                };
                if target.peer == group && !is_source && Arc::ptr_eq(&target.fs, &parent.fs) {
                    place(&mut copies, &peers, &masters);
                } else if target.master == group && Arc::ptr_eq(&target.fs, &parent.fs) {
                    if target.peer == 0 {
                        place(&mut copies, &vec![0; tree.len()], &peers);
                    } else {
                        // A slave that's shared too: its peers get copies that
                        // are peers of each other and slaves of these.
                        let fresh: Vec<u32> = (0..tree.len()).map(|_| 0).collect();
                        queue.push((target.peer, fresh, peers.clone()));
                    }
                }
            }
            // Peer groups for a level whose copies are shared among themselves.
            for (_, peers, _) in &mut queue {
                for g in peers.iter_mut() {
                    if *g == 0 {
                        self.groups += 1;
                        *g = self.groups;
                    }
                }
            }
        }
        self.mounts.extend(tree);
        for (ns, m) in copies {
            self.list_mut(ns).push(m);
        }
    }

    /// Every mount, in any namespace, that gets what happens below the mounts
    /// of peer group `group`: its peers, their slaves, and so on down.
    fn receivers(&self, group: u32) -> Vec<(u32, Mount)> {
        let all = self.all_mounts();
        let mut groups = vec![group];
        let mut out = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        while let Some(g) = groups.pop() {
            if !seen.insert(g) {
                continue;
            }
            for (ns, m) in &all {
                if m.peer == g || m.master == g {
                    out.push((*ns, m.clone()));
                    if m.master == g && m.peer != 0 {
                        groups.push(m.peer);
                    }
                }
            }
        }
        out
    }

    /// The mount at `self.mounts[i]` is about to go: under a shared parent,
    /// its copies under the parent's peers and slaves go too, where nothing is
    /// mounted below them (Linux's `propagate_umount`).
    fn propagate_unmount(&mut self, i: usize) {
        let gone = self.mounts[i].clone();
        let parent = self
            .mounts
            .iter()
            .enumerate()
            .filter(|&(j, m)| {
                j != i && path_has_prefix(&gone.point, &m.point) && m.point != gone.point
            })
            .max_by_key(|&(j, m)| (m.point.len(), j))
            .map(|(_, m)| m.clone());
        let Some(parent) = parent.filter(|p| p.peer != 0) else {
            return;
        };
        let at = parent.fs_path(&relative_to(&gone.point, &parent.point));
        let current = self.current;
        for (ns, target) in self.receivers(parent.peer) {
            if (ns == current && target.point == parent.point)
                || !Arc::ptr_eq(&target.fs, &parent.fs)
            {
                continue;
            }
            let Some(below) = path_strip(&at, &target.subdir) else {
                continue;
            };
            let point = normalize(&format!("{}/{below}", target.point));
            let list = self.list_mut(ns);
            let busy = list
                .iter()
                .any(|m| m.point != point && path_has_prefix(&m.point, &point));
            if busy {
                continue;
            }
            if let Some(k) = list
                .iter()
                .rposition(|m| m.point == point && Arc::ptr_eq(&m.fs, &gone.fs))
            {
                list.remove(k);
            }
        }
    }

    fn fresh_group(&mut self) -> u32 {
        self.groups += 1;
        self.groups
    }

    /// Every mount of every namespace, with its namespace.
    fn all_mounts(&self) -> Vec<(u32, Mount)> {
        let current = self.mounts.iter().map(|m| (self.current, m.clone()));
        let parked = self
            .parked
            .iter()
            .flat_map(|(&ns, list)| list.iter().map(move |m| (ns, m.clone())));
        current.chain(parked).collect()
    }

    /// Namespace `ns`'s mount list.
    fn list_mut(&mut self, ns: u32) -> &mut Vec<Mount> {
        if ns == self.current {
            &mut self.mounts
        } else {
            self.parked.entry(ns).or_default()
        }
    }

    /// Change how the mount at `point` (and with `recursive`, every mount below
    /// it) propagates: `mount --make-shared`, `--make-slave`, `--make-private`,
    /// `--make-unbindable`. `EINVAL` if `point` isn't a mount point.
    pub fn set_propagation(
        &mut self,
        point: &str,
        kind: Propagation,
        recursive: bool,
    ) -> io::Result<()> {
        let point = normalize(point);
        let Some(top) = self.mounts.iter().rposition(|m| m.point == point) else {
            return Err(einval());
        };
        let targets: Vec<usize> = (0..self.mounts.len())
            .filter(|&i| {
                i == top
                    || (recursive
                        && path_has_prefix(&self.mounts[i].point, &point)
                        && self.mounts[i].point != point)
            })
            .collect();
        for i in targets {
            match kind {
                Propagation::Shared => {
                    if self.mounts[i].peer == 0 {
                        self.mounts[i].peer = self.fresh_group();
                    }
                    self.mounts[i].unbindable = false;
                }
                Propagation::Private => {
                    let m = &mut self.mounts[i];
                    (m.peer, m.master, m.unbindable) = (0, 0, false);
                }
                Propagation::Unbindable => {
                    let m = &mut self.mounts[i];
                    (m.peer, m.master, m.unbindable) = (0, 0, true);
                }
                Propagation::Slave => {
                    // A shared mount becomes a slave of the group it leaves, if
                    // anything is left in it; one that's neither stays private.
                    let group = self.mounts[i].peer;
                    if group != 0 {
                        let others = self
                            .all_mounts()
                            .iter()
                            .filter(|(_, m)| m.peer == group)
                            .count()
                            > 1;
                        let m = &mut self.mounts[i];
                        if others {
                            m.master = group;
                        }
                        m.peer = 0;
                    }
                    self.mounts[i].unbindable = false;
                }
            }
        }
        Ok(())
    }

    /// Remove the most recent mount at exactly `point`: `Ok` if a guest mount
    /// went, `EBUSY` for one of the embedder's (the VM's own /proc, /dev, …),
    /// `EINVAL` if nothing is mounted there.
    pub fn unmount(&mut self, point: &str, detach: bool) -> io::Result<Option<PathMove>> {
        let point = normalize(point);
        let Some(i) = self.mounts.iter().rposition(|m| m.point == point) else {
            return Err(einval());
        };
        if self.mounts[i].locked {
            return Err(einval());
        }
        // Outside the initial namespace nothing is the VM's own: a container that
        // pivoted away from its old root drops those mounts like any other.
        if !self.mounts[i].guest && self.current == 0 {
            return Err(io::Error::from_raw_os_error(16)); // EBUSY
        }
        self.propagate_unmount(i);
        let i = self
            .mounts
            .iter()
            .rposition(|m| m.point == point)
            .ok_or_else(einval)?;
        let side = self.mounts[i].pivot_side;
        let has_children = self
            .mounts
            .iter()
            .any(|m| m.point != point && m.pivot_side == side && path_has_prefix(&m.point, &point));
        if has_children && !detach {
            return Err(io::Error::from_raw_os_error(16)); // EBUSY
        }
        let side = self.mounts[i].pivot_side;
        let gone = self.mounts.remove(i);
        if !detach {
            return Ok(None);
        }
        // Linux keeps a detached mount alive for anything still using it. Paths are
        // what's held here, so it moves somewhere no guest path can name (a path from
        // the guest has no NUL in it), for the caller to send those users after it.
        let hidden = format!("/{DETACHED_MARK}{}", self.detached);
        self.detached += 1;
        for m in &mut self.mounts {
            if m.point != point && m.pivot_side == side && path_has_prefix(&m.point, &point) {
                m.point = normalize(&format!("{hidden}/{}", relative_to(&m.point, &point)));
            }
        }
        self.mounts.push(Mount {
            point: hidden.clone(),
            ..gone
        });
        Ok(Some(PathMove {
            from: point,
            to: hidden,
        }))
    }

    /// Keep `paths`, as mount namespace `from` resolves them, reachable from the
    /// current namespace: what a task's open files need when it joins another
    /// one, since Linux's still name the file they were opened on. Each path's
    /// mount is shared under a hidden name, like a detached mount, and the path
    /// to use from now on is returned in its place (`None`: nothing there to
    /// carry). [`Self::collect_detached`] drops them once unused.
    pub fn carry_from(&mut self, from: u32, paths: &[String]) -> Vec<Option<String>> {
        if from == self.current {
            return paths.iter().cloned().map(Some).collect();
        }
        let Some(source) = self.parked.get(&from) else {
            return vec![None; paths.len()];
        };
        // The longest-prefix mount for each path, among `from`'s.
        let owner = |path: &str| {
            source
                .iter()
                .enumerate()
                .filter(|(_, m)| path_has_prefix(path, &m.point))
                .max_by_key(|(i, m)| (m.point.len(), *i))
                .map(|(i, _)| i)
        };
        let mut carried: BTreeMap<usize, String> = BTreeMap::new();
        let mut added = Vec::new();
        let mut out = Vec::with_capacity(paths.len());
        for path in paths {
            let path = normalize(path);
            let Some(i) = owner(&path) else {
                out.push(None);
                continue;
            };
            let hidden = carried.entry(i).or_insert_with(|| {
                let hidden = format!("/{DETACHED_MARK}{}", self.detached);
                self.detached += 1;
                added.push(Mount {
                    point: hidden.clone(),
                    ..source[i].clone()
                });
                hidden
            });
            out.push(Some(normalize(&format!(
                "{hidden}/{}",
                relative_to(&path, &source[i].point)
            ))));
        }
        self.mounts.extend(added);
        out
    }

    /// Drop detached mounts nothing refers to any more. `in_use` is every path held
    /// on behalf of a task.
    pub fn collect_detached(&mut self, in_use: &[&str]) {
        let is_used = |root: &str| in_use.iter().any(|p| path_has_prefix(p, root));
        let roots: Vec<String> = self
            .mounts
            .iter()
            .filter_map(|m| detached_root(&m.point))
            .map(str::to_string)
            .collect();
        let dead: Vec<String> = roots.into_iter().filter(|r| !is_used(r)).collect();
        self.mounts
            .retain(|m| detached_root(&m.point).is_none_or(|r| !dead.iter().any(|d| d == r)));
    }

    /// Index of the longest-prefix mount owning `abs_path`.
    ///
    /// `abs_path` must already be normalized (callers go through [`resolve`]
    /// or normalize explicitly, e.g. [`rename`]) so that prefix matching
    /// happens at path-component boundaries only: `/dev` matches `/dev` and
    /// `/dev/null` but never `/devfoo`.
    ///
    /// [`resolve`]: Self::resolve
    /// [`rename`]: Self::rename
    fn best_mount(&self, abs_path: &str) -> Option<usize> {
        let mut best: Option<usize> = None;
        let mut best_len = 0usize;
        for (i, m) in self.mounts.iter().enumerate() {
            if path_has_prefix(abs_path, &m.point) && m.point.len() >= best_len {
                best = Some(i);
                best_len = m.point.len();
            }
        }
        best
    }

    /// Find the backend owning `abs_path` and the path relative to its mount
    /// point. Returns the longest matching mount.
    ///
    /// `abs_path` is defensively normalized here (collapsing `//`, resolving
    /// `.`/`..` lexically without escaping `/`, and dropping any trailing
    /// slash) before being matched against mount points. The kernel layer
    /// already normalizes guest paths before calling into the mount table,
    /// but `MountTable` doesn't rely on that: an exact mount point (e.g.
    /// `stat("/dev")`) resolves to the backend's root (empty relative path),
    /// and equivalent spellings of a path (`/a/./b`, `/a//b`, `/a/x/../b`)
    /// all resolve to the same `(backend, relative path)` pair.
    fn resolve(&mut self, abs_path: &str) -> Option<(MutexGuard<'_, Box<dyn MountFs>>, String)> {
        let abs_path = self.canonical(abs_path);
        let i = self.best_mount(&abs_path)?;
        let rel = self.mounts[i].fs_path(&relative_to(&abs_path, &self.mounts[i].point));
        Some((self.mounts[i].lock(), rel))
    }

    /// `abs_path` with its final component's symlink chain followed — inside
    /// this mount table, so an absolute target (`/bin/sh -> /bin/busybox`)
    /// resolves against the guest root, never the host's. `None` for a loop
    /// (more than 40 hops) or a dangling intermediate link.
    pub fn follow(&mut self, abs_path: &str) -> Option<String> {
        let mut p = normalize(abs_path);
        for _ in 0..SYMLINK_MAX {
            // The link's directories resolved first: a relative target is read
            // from where the link really is. Debian's `/lib64 -> usr/lib64`
            // holds `ld-linux-x86-64.so.2 -> ../lib/x86_64-linux-gnu/…`, whose
            // `..` is `/usr`, not `/`.
            p = self.canonical(&p);
            match self.stat_raw(&p) {
                Some(a) if a.kind == NodeKind::Symlink => {
                    let target = self.readlink(&p).ok()?;
                    p = if target.starts_with('/') {
                        self.join_root(&target)
                    } else {
                        let dir = p.rfind('/').map_or("", |i| &p[..i]);
                        self.join_relative(dir, &target)
                    };
                }
                _ => return Some(p),
            }
        }
        None
    }

    /// Read a whole file, following symlinks as [`Self::follow`] does — how
    /// the host side loads the initial program and its dynamic linker.
    pub fn read_file(&mut self, abs_path: &str) -> Option<Vec<u8>> {
        let path = self.follow(abs_path)?;
        let size = self.stat(&path)?.size as usize;
        // A sparse file can claim any size at all.
        let mut buf = Vec::new();
        buf.try_reserve_exact(size).ok()?;
        buf.resize(size, 0);
        let mut off = 0;
        while off < size {
            match self.read_at(&path, off as u64, &mut buf[off..]) {
                Ok(0) => break,
                Ok(n) => off += n,
                Err(_) => return None,
            }
        }
        buf.truncate(off);
        Some(buf)
    }

    pub fn stat(&mut self, abs_path: &str) -> Option<Attrs> {
        let abs_path = self.canonical(abs_path);
        self.stat_raw(&abs_path)
    }

    /// `stat` of an already-canonical path, without resolving symlinks.
    fn stat_raw(&mut self, abs_path: &str) -> Option<Attrs> {
        let i = self.best_mount(abs_path)?;
        let rel = self.mounts[i].fs_path(&relative_to(abs_path, &self.mounts[i].point));
        let mut a = self.mounts[i].lock().stat(&rel)?;
        a.dev = self.dev_of(i);
        Some(a)
    }

    /// `abs_path`, normalized, with every symlink among its directories
    /// followed (the final component is left alone: whether to follow it is
    /// the caller's choice, e.g. `stat` vs `lstat`). Backends only know their
    /// own paths, so a link to a directory — possibly on another mount, like
    /// Alpine's `/var/lib/nginx/logs -> /var/log/nginx` — is resolved here.
    /// A symlink loop leaves the path as is, for the backend to fail on.
    fn canonical(&mut self, abs_path: &str) -> String {
        let path = normalize(abs_path);
        // Fast path: the parent is a real directory, so no symlink is involved.
        let parent = match path.rfind('/') {
            Some(0) | None => return path,
            Some(i) => &path[..i],
        };
        if self
            .stat_raw(parent)
            .is_some_and(|a| a.kind == NodeKind::Dir)
        {
            return path;
        }
        let mut done = String::new();
        let mut rest: Vec<String> = path
            .split('/')
            .rev()
            .filter(|c| !c.is_empty())
            .map(str::to_string)
            .collect();
        let mut hops = 0;
        while let Some(c) = rest.pop() {
            if c == "." {
                continue;
            }
            if c == ".." {
                // `..` stops at the root, wherever that's been moved to.
                if done.len() > self.root.len() {
                    done.truncate(done.rfind('/').unwrap_or(0));
                }
                continue;
            }
            let cand = format!("{done}/{c}");
            if rest.is_empty() {
                done = cand;
                break;
            }
            match self.stat_raw(&cand) {
                Some(a) if a.kind == NodeKind::Symlink => {
                    hops += 1;
                    let Some(target) = (hops <= SYMLINK_MAX)
                        .then(|| {
                            let i = self.best_mount(&cand)?;
                            let m = &self.mounts[i];
                            let rel = m.fs_path(&relative_to(&cand, &m.point));
                            m.lock().readlink(&rel).ok()
                        })
                        .flatten()
                    else {
                        return path;
                    };
                    if target.starts_with('/') {
                        // Absolute: from the caller's root, not the namespace's.
                        done.clone_from(&self.root);
                    }
                    rest.extend(
                        target
                            .split('/')
                            .rev()
                            .filter(|c| !c.is_empty())
                            .map(str::to_string),
                    );
                }
                // A missing or non-directory component: the rest can't
                // resolve; let the backend report it.
                None => {
                    done = cand;
                    for c in rest.drain(..).rev() {
                        done.push('/');
                        done.push_str(&c);
                    }
                    return normalize(&done);
                }
                Some(_) => done = cand,
            }
        }
        if done.is_empty() {
            "/".to_string()
        } else {
            done
        }
    }

    pub fn read_at(&mut self, abs_path: &str, off: u64, buf: &mut [u8]) -> io::Result<usize> {
        let (mut fs, rel) = self.resolve(abs_path).ok_or_else(enoent)?;
        fs.read_at(&rel, off, buf)
    }

    /// List the contents of the directory at `abs_path`.
    ///
    /// If `abs_path` is itself a mount point, this lists the mounted
    /// backend's root (not the parent backend's view of the mount-point
    /// directory). Note that `MountTable` does not synthesize entries for
    /// child mount points nested under `abs_path`: a `readdir` of `/dev`
    /// returns exactly what the `/dev` backend reports, even if e.g. `/dev/pts`
    /// is a separate mount — cross-mount directory-listing merge isn't
    /// modeled at this layer.
    pub fn readdir(&mut self, abs_path: &str) -> io::Result<Vec<DirEntry>> {
        let (mut fs, rel) = self.resolve(abs_path).ok_or_else(enoent)?;
        fs.readdir(&rel)
    }

    pub fn write_at(&mut self, abs_path: &str, off: u64, buf: &[u8]) -> io::Result<usize> {
        let (mut fs, rel) = self.resolve_for_write(abs_path)?;
        fs.write_at(&rel, off, buf)
    }

    pub fn create(&mut self, abs_path: &str, mode: u32) -> io::Result<()> {
        let (mut fs, rel) = self.resolve_for_write(abs_path)?;
        fs.create(&rel, mode)
    }

    pub fn mkdir(&mut self, abs_path: &str, mode: u32) -> io::Result<()> {
        let (mut fs, rel) = self.resolve_for_write(abs_path)?;
        fs.mkdir(&rel, mode)
    }

    pub fn mknod(&mut self, abs_path: &str, mode: u32) -> io::Result<()> {
        let (mut fs, rel) = self.resolve_for_write(abs_path)?;
        fs.mknod(&rel, mode)
    }

    pub fn unlink(&mut self, abs_path: &str) -> io::Result<()> {
        let (mut fs, rel) = self.resolve_for_write(abs_path)?;
        fs.unlink(&rel)
    }

    pub fn rmdir(&mut self, abs_path: &str) -> io::Result<()> {
        let (mut fs, rel) = self.resolve_for_write(abs_path)?;
        fs.rmdir(&rel)
    }

    pub fn truncate(&mut self, abs_path: &str, len: u64) -> io::Result<()> {
        let (mut fs, rel) = self.resolve_for_write(abs_path)?;
        fs.truncate(&rel, len)
    }

    pub fn set_times(
        &mut self,
        abs_path: &str,
        atime: super::SetTime,
        mtime: super::SetTime,
    ) -> io::Result<()> {
        let (mut fs, rel) = self.resolve_for_write(abs_path)?;
        fs.set_times(&rel, atime, mtime)
    }

    pub fn set_owner(
        &mut self,
        abs_path: &str,
        uid: Option<u32>,
        gid: Option<u32>,
    ) -> io::Result<()> {
        let (mut fs, rel) = self.resolve_for_write(abs_path)?;
        fs.set_owner(&rel, uid, gid)
    }

    pub fn set_mode(&mut self, abs_path: &str, mode: u32) -> io::Result<()> {
        let (mut fs, rel) = self.resolve_for_write(abs_path)?;
        fs.set_mode(&rel, mode)
    }

    pub fn readlink(&mut self, abs_path: &str) -> io::Result<String> {
        let (mut fs, rel) = self.resolve(abs_path).ok_or_else(enoent)?;
        fs.readlink(&rel)
    }

    /// Create a symlink at `abs_link` pointing at `target` (stored verbatim).
    pub fn symlink(&mut self, target: &str, abs_link: &str) -> io::Result<()> {
        let (mut fs, rel) = self.resolve_for_write(abs_link)?;
        fs.symlink(target, &rel)
    }

    /// The mount point whose backend serves `abs_path` (after resolving the
    /// symlinks among its directories) — what `RESOLVE_NO_XDEV` and
    /// `st_dev`-style "same filesystem?" checks compare.
    pub fn mount_point_of(&mut self, abs_path: &str) -> Option<String> {
        let abs = self.canonical(abs_path);
        let i = self.best_mount(&abs)?;
        Some(self.mounts[i].point.clone())
    }

    /// Extended attribute `name` of the node at `abs_path` (see
    /// [`MountFs::getxattr`]). Like the other per-node operations the final
    /// component is not followed — the kernel decides follow vs. no-follow.
    pub fn getxattr(&mut self, abs_path: &str, name: &str) -> io::Result<Vec<u8>> {
        let (mut fs, rel) = self.resolve(abs_path).ok_or_else(enoent)?;
        fs.getxattr(&rel, name)
    }

    /// Set extended attribute `name` on the node at `abs_path`.
    pub fn setxattr(&mut self, abs_path: &str, name: &str, value: &[u8]) -> io::Result<()> {
        let (mut fs, rel) = self.resolve_for_write(abs_path)?;
        fs.setxattr(&rel, name, value)
    }

    /// Every extended-attribute name on the node at `abs_path`.
    pub fn listxattr(&mut self, abs_path: &str) -> io::Result<Vec<String>> {
        let (mut fs, rel) = self.resolve(abs_path).ok_or_else(enoent)?;
        fs.listxattr(&rel)
    }

    /// Remove extended attribute `name` from the node at `abs_path`.
    pub fn removexattr(&mut self, abs_path: &str, name: &str) -> io::Result<()> {
        let (mut fs, rel) = self.resolve_for_write(abs_path)?;
        fs.removexattr(&rel, name)
    }

    /// Rename within a single backend. Cross-mount renames return `EXDEV`.
    pub fn rename(&mut self, from: &str, to: &str) -> io::Result<()> {
        let from = self.canonical(from);
        let to = self.canonical(to);
        let from_idx = self.best_mount(&from).ok_or_else(enoent)?;
        let to_idx = self.best_mount(&to).ok_or_else(enoent)?;
        if from_idx != to_idx {
            return Err(io::Error::from_raw_os_error(18)); // EXDEV
        }
        let m = &self.mounts[from_idx];
        if m.read_only {
            return Err(io::Error::from_raw_os_error(30)); // EROFS
        }
        let from_rel = m.fs_path(&relative_to(&from, &m.point));
        let to_rel = m.fs_path(&relative_to(&to, &m.point));
        m.lock().rename(&from_rel, &to_rel)
    }

    /// Hard-link `new` to the existing `old` within a single backend. Returns
    /// `EOPNOTSUPP` for a cross-mount request (leaving the kernel to fall back
    /// to copying, its historical behavior) and otherwise delegates to the
    /// backend — which either makes a real hard link (host-backed) or itself
    /// reports `EOPNOTSUPP` (path-keyed in-memory backends → copy).
    pub fn link(&mut self, old: &str, new: &str) -> io::Result<()> {
        let old = self.canonical(old);
        let new = self.canonical(new);
        let old_idx = self.best_mount(&old).ok_or_else(enoent)?;
        let new_idx = self.best_mount(&new).ok_or_else(enoent)?;
        if old_idx != new_idx {
            return Err(io::Error::from_raw_os_error(18)); // EXDEV, as Linux
        }
        let m = &self.mounts[old_idx];
        if m.read_only {
            return Err(io::Error::from_raw_os_error(30)); // EROFS
        }
        let old_rel = m.fs_path(&relative_to(&old, &m.point));
        let new_rel = m.fs_path(&relative_to(&new, &m.point));
        m.lock().link(&old_rel, &new_rel)
    }
}

fn enoent() -> io::Error {
    io::Error::from_raw_os_error(2)
}

fn einval() -> io::Error {
    io::Error::from_raw_os_error(22)
}

/// What marks a detached mount's path. No path from the guest can contain it.
const DETACHED_MARK: char = '\0';

/// The top of the detached mount `path` is in, if it's in one.
fn detached_root(path: &str) -> Option<&str> {
    let rest = path.strip_prefix('/')?;
    if !rest.starts_with(DETACHED_MARK) {
        return None;
    }
    let end = rest.find('/').map_or(path.len(), |i| i + 1);
    Some(&path[..end])
}

/// Where paths at or below `from` have gone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathMove {
    from: String,
    to: String,
}

impl PathMove {
    /// Where `path` is now.
    #[must_use]
    pub fn translate(&self, path: &str) -> String {
        if path_has_prefix(path, &self.from) {
            normalize(&format!("{}/{}", self.to, relative_to(path, &self.from)))
        } else {
            path.to_string()
        }
    }
}

/// One mount, for `/proc/self/mountinfo`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MountEntry {
    pub point: String,
    pub subdir: String,
    pub read_only: bool,
    pub fs_type: String,
    /// The device its filesystem is (`st_dev` of its files).
    pub dev: u64,
    /// What `mountinfo` says it's mounted from: its disk (`/dev/vda`), or for
    /// a filesystem not on one its type.
    pub source: String,
    /// `shared:N`, `master:N` and `unbindable` in `mountinfo`; 0 for none.
    pub peer: u32,
    pub master: u32,
    pub unbindable: bool,
}

/// A mount's propagation type, as `mount(2)`'s `MS_SHARED`, `MS_SLAVE`,
/// `MS_PRIVATE` and `MS_UNBINDABLE` set it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Propagation {
    Shared,
    Slave,
    Private,
    Unbindable,
}

/// How paths change across a [`MountTable::pivot_root`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PivotMap {
    new_root: String,
    put_old: String,
}

impl PivotMap {
    /// Where `path`, from before the pivot, is now.
    #[must_use]
    pub fn translate(&self, path: &str) -> String {
        if path_has_prefix(path, &self.new_root) {
            normalize(&format!("/{}", relative_to(path, &self.new_root)))
        } else {
            normalize(&format!("{}/{}", self.put_old, path))
        }
    }
}

/// Normalize an absolute guest path into a clean `/a/b/c` form.
///
/// Collapses duplicate slashes, drops `.` components, resolves `..`
/// lexically (never escaping above `/`), and strips any trailing slash
/// (the root itself normalizes to `/`, not the empty string). This mirrors
/// `kernel::path::normalize`'s behavior; it is duplicated here (rather than
/// imported) because that module is private to `kernel` and `MountTable`
/// must not assume every caller has already normalized its input — mount
/// registration and every path-taking method normalize defensively so
/// prefix matching is always done on a canonical path.
fn normalize(p: &str) -> String {
    let mut stack: Vec<&str> = Vec::new();
    for comp in p.split('/') {
        match comp {
            "" | "." => {}
            ".." => {
                stack.pop();
            }
            other => stack.push(other),
        }
    }
    if stack.is_empty() {
        "/".to_string()
    } else {
        let mut out = String::with_capacity(p.len());
        for c in stack {
            out.push('/');
            out.push_str(c);
        }
        out
    }
}

/// Does `path` fall under mount point `mp`? "/" matches everything.
///
/// Matching happens at path-component boundaries: `mp` must be either the
/// whole of `path` or a proper ancestor of it, so `/dev` matches `/dev` and
/// `/dev/null` but never `/devfoo` (the naive `str::starts_with` check would
/// wrongly match the latter). Both `path` and `mp` are expected to already
/// be normalized (no trailing slash, no `.`/`..`, no duplicate slashes).
fn path_has_prefix(path: &str, mp: &str) -> bool {
    if mp == "/" {
        return true;
    }
    if let Some(rest) = path.strip_prefix(mp) {
        rest.is_empty() || rest.starts_with('/')
    } else {
        false
    }
}

/// Strip the mount point, yielding a backend-relative path without the leading
/// slash (so backends see e.g. "bin/sh" for "/bin/sh"). When `path` is
/// exactly the mount point `mp`, the result is the empty string — the
/// backend's own root — so `stat`/`readdir` of a bare mount point reaches
/// the backend instead of failing.
/// `fs_path` (a path inside a filesystem, as [`Mount::fs_path`] gives it) below
/// `subdir`, relative to it; `None` if it isn't below it.
fn path_strip(fs_path: &str, subdir: &str) -> Option<String> {
    if subdir.is_empty() {
        return Some(fs_path.to_string());
    }
    if fs_path == subdir {
        return Some(String::new());
    }
    fs_path
        .strip_prefix(subdir)
        .and_then(|r| r.strip_prefix('/'))
        .map(str::to_string)
}

fn relative_to(path: &str, mp: &str) -> String {
    let rest = if mp == "/" {
        path
    } else {
        path.strip_prefix(mp).unwrap_or(path)
    };
    rest.trim_start_matches('/').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fs::NodeKind;

    fn table_with_tmpfs_root() -> MountTable {
        let mut t = MountTable::new();
        t.mount("/", Box::new(crate::fs::TmpFs::new()));
        t.mkdir("/a", 0o755).unwrap();
        t.mkdir("/b", 0o755).unwrap();
        t
    }

    #[test]
    fn a_copied_namespace_shares_files_but_not_mounts() {
        let mut t = table_with_tmpfs_root();
        t.copy_namespace(0, 1, false);
        t.enter(1, "/");
        t.mount_guest("/a", Box::new(crate::fs::TmpFs::new()));
        t.create("/a/inner", 0o644).unwrap();
        t.create("/shared", 0o644).unwrap();
        t.enter(0, "/");
        assert!(
            t.stat("/a/inner").is_none(),
            "the mount stayed in namespace 1"
        );
        assert!(
            t.stat("/shared").is_some(),
            "the root filesystem is the same one"
        );
        t.enter(1, "/");
        assert!(t.stat("/a/inner").is_some());
    }

    #[test]
    fn a_bind_mount_is_the_same_files_and_can_be_read_only() {
        let mut t = table_with_tmpfs_root();
        t.create("/a/f", 0o644).unwrap();
        t.bind("/a", "/b", false).unwrap();
        assert!(t.stat("/b/f").is_some());
        t.create("/b/g", 0o644).unwrap();
        assert!(t.stat("/a/g").is_some());
        t.set_read_only("/b", true).unwrap();
        let erofs = |r: io::Result<()>| r.unwrap_err().raw_os_error() == Some(30);
        assert!(erofs(t.create("/b/h", 0o644)));
        assert!(erofs(t.unlink("/b/f")));
        assert!(erofs(t.mkdir("/b/d", 0o755)));
        assert!(erofs(t.rename("/b/f", "/b/f2")));
        t.create("/a/h", 0o644).unwrap();
        assert!(
            t.stat("/b/h").is_some(),
            "read-only is the mount's, not the file's"
        );
    }

    #[test]
    fn identity_sees_through_bind_mounts_and_tells_mounts_apart() {
        let mut t = table_with_tmpfs_root();
        t.bind("/a", "/b", false).unwrap();
        assert_eq!(t.identity("/a/sock"), t.identity("/b/sock"));
        t.mount_guest("/a", Box::new(crate::fs::TmpFs::new()));
        assert_ne!(t.identity("/a/sock"), t.identity("/b/sock"));
    }

    #[test]
    fn pivot_root_takes_the_top_mount_and_leaves_what_it_was_hiding() {
        let mut t = table_with_tmpfs_root();
        t.mount("/a", Box::new(crate::fs::TmpFs::new()));
        t.create("/a/hidden", 0o644).unwrap();
        t.copy_namespace(0, 1, false);
        t.enter(1, "/");
        // A tmpfs over the existing mount at /a becomes the new root.
        t.mount_guest("/a", Box::new(crate::fs::TmpFs::new()));
        t.mkdir("/a/old", 0o755).unwrap();
        t.create("/a/visible", 0o644).unwrap();
        t.pivot_root("/a", "/a/old").unwrap();
        assert!(t.stat("/visible").is_some());
        assert!(
            t.stat("/hidden").is_none(),
            "what it hid isn't in the new root"
        );
        assert!(
            t.stat("/old/a/hidden").is_some(),
            "it went with the old root"
        );
        // And dropping the old root leaves the new one in charge at /.
        t.unmount("/old", true).unwrap();
        assert!(t.stat("/visible").is_some());
        assert!(t.stat("/old/a/hidden").is_none());
    }

    #[test]
    fn pivot_root_rewrites_the_view_and_detaching_takes_the_old_root() {
        let mut t = table_with_tmpfs_root();
        t.copy_namespace(0, 1, false);
        t.enter(1, "/");
        t.mount_guest("/a", Box::new(crate::fs::TmpFs::new()));
        t.mkdir("/a/old", 0o755).unwrap();
        t.create("/a/mark", 0o644).unwrap();
        assert!(t.pivot_root("/b", "/b").is_err(), "not a mount point");
        let map = t.pivot_root("/a", "/a/old").unwrap();
        assert_eq!(map.translate("/a/mark"), "/mark");
        assert_eq!(map.translate("/b"), "/old/b");
        assert!(t.stat("/mark").is_some());
        assert!(t.stat("/old/b").is_some());
        t.unmount("/old", true).unwrap();
        assert!(t.stat("/old/b").is_none());
    }

    #[test]
    fn a_relative_link_is_read_from_where_it_really_is() {
        // Debian's merged /usr: the loader's path goes through a linked
        // directory, then ends in a link whose `..` is /usr, not /.
        let mut t = table_with_tmpfs_root();
        for d in [
            "/usr",
            "/usr/lib",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
        ] {
            t.mkdir(d, 0o755).unwrap();
        }
        t.symlink("usr/lib64", "/lib64").unwrap();
        t.symlink(
            "../lib/x86_64-linux-gnu/ld-linux-x86-64.so.2",
            "/usr/lib64/ld-linux-x86-64.so.2",
        )
        .unwrap();
        let real = "/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2";
        t.create(real, 0o755).unwrap();
        t.write_at(real, 0, b"\x7fELF").unwrap();
        assert_eq!(t.follow("/lib64/ld-linux-x86-64.so.2").unwrap(), real);
        assert_eq!(
            t.read_file("/lib64/ld-linux-x86-64.so.2").unwrap(),
            b"\x7fELF"
        );
    }

    #[test]
    fn a_chroot_holds_for_symlinks_and_dot_dot() {
        let mut t = table_with_tmpfs_root();
        t.mkdir("/a/in", 0o755).unwrap();
        t.create("/a/in/f", 0o644).unwrap();
        t.create("/secret", 0o644).unwrap();
        t.symlink("/in", "/a/abs").unwrap();
        t.symlink("../../secret", "/a/in/esc").unwrap();
        t.enter(0, "/a");
        assert_eq!(t.join_root("/in"), "/a/in");
        assert_eq!(t.follow("/a/abs").unwrap(), "/a/in");
        assert!(
            t.stat("/a/abs/f").is_some(),
            "an absolute link stays inside"
        );
        let climbed = t.follow("/a/in/esc").unwrap();
        assert_ne!(climbed, "/secret");
    }

    #[test]
    fn a_detached_mount_lives_on_for_paths_that_follow_it_until_they_let_go() {
        let mut t = table_with_tmpfs_root();
        t.mount_guest("/a", Box::new(crate::fs::TmpFs::new()));
        t.mount_guest("/a/sub", Box::new(crate::fs::TmpFs::new()));
        t.create("/a/f", 0o644).unwrap();
        t.create("/a/sub/g", 0o644).unwrap();
        let moved = t.unmount("/a", true).unwrap().unwrap();
        assert!(t.stat("/a/f").is_none(), "gone from the tree");
        assert!(t.stat("/a/sub/g").is_none());
        let held = moved.translate("/a/sub");
        assert!(held.contains(DETACHED_MARK));
        assert!(
            t.stat(&format!("{held}/g")).is_some(),
            "still there for its user"
        );
        assert!(t.listing().iter().all(|m| !m.point.contains(DETACHED_MARK)));
        assert_eq!(moved.translate("/other"), "/other");
        t.collect_detached(&[&held]);
        assert!(t.stat(&format!("{held}/g")).is_some(), "still in use");
        t.collect_detached(&[]);
        assert!(
            t.stat(&format!("{held}/g")).is_none(),
            "nothing holds it now"
        );
        assert_eq!(t.mounts.len(), 1);
    }

    #[test]
    fn only_the_initial_namespace_keeps_the_vms_own_mounts() {
        let mut t = table_with_tmpfs_root();
        assert_eq!(t.unmount("/", false).unwrap_err().raw_os_error(), Some(16));
        t.copy_namespace(0, 1, false);
        t.enter(1, "/");
        t.unmount("/", false).unwrap();
        t.enter(0, "/");
        assert!(t.stat("/a").is_some());
    }

    #[test]
    fn longest_prefix_and_relative() {
        assert!(path_has_prefix("/work/src/a.rs", "/work"));
        assert!(path_has_prefix("/anything", "/"));
        assert!(!path_has_prefix("/worker", "/work"));
        assert_eq!(relative_to("/work/src/a.rs", "/work"), "src/a.rs");
        assert_eq!(relative_to("/bin/sh", "/"), "bin/sh");
    }

    #[test]
    fn normalize_collapses_dot_dotdot_and_slashes() {
        assert_eq!(normalize("/"), "/");
        assert_eq!(normalize(""), "/");
        assert_eq!(normalize("/a/./b"), "/a/b");
        assert_eq!(normalize("/a//b"), "/a/b");
        assert_eq!(normalize("/a/../b"), "/b");
        assert_eq!(normalize("/a/b/"), "/a/b");
        assert_eq!(normalize("/../a"), "/a");
    }

    /// Minimal `MountFs` stub for exercising [`MountTable`] routing without a
    /// real backend. `stat`/`readdir` always succeed (never `ENOENT`) and
    /// report the exact relative path they were asked about, so tests can
    /// assert not just *which* backend was chosen but *what sub-path* it saw.
    #[derive(Debug)]
    struct TagFs {
        tag: &'static str,
    }

    impl MountFs for TagFs {
        fn stat(&mut self, rel: &str) -> Option<Attrs> {
            Some(Attrs {
                kind: NodeKind::Dir,
                // Encode the rel-path length so callers can tell an exact
                // mount-point hit (rel == "", size 0) from a sub-path.
                size: rel.len() as u64,
                mode: 0o755,
                uid: 0,
                gid: 0,
                atime: 0,
                mtime: 0,
                inode: 0,
                nlink: 1,
                rdev: 0,
                dev: 0,
            })
        }

        fn read_at(&mut self, _rel: &str, _off: u64, _buf: &mut [u8]) -> io::Result<usize> {
            Ok(0)
        }

        fn readdir(&mut self, rel: &str) -> io::Result<Vec<DirEntry>> {
            Ok(vec![DirEntry {
                name: format!("{}:{rel}", self.tag),
                kind: NodeKind::File,
                inode: 0,
            }])
        }
    }

    fn tagged(tag: &'static str) -> Box<dyn MountFs> {
        Box::new(TagFs { tag })
    }

    /// Reads back which `(tag, rel)` pair a path routed to, via `readdir`'s
    /// synthesized entry name (`"tag:rel"`).
    fn route(table: &mut MountTable, path: &str) -> String {
        table.readdir(path).unwrap()[0].name.clone()
    }

    #[test]
    fn dev_does_not_match_devfoo() {
        let mut table = MountTable::new();
        table.mount("/", tagged("root"));
        table.mount("/dev", tagged("dev"));

        assert_eq!(route(&mut table, "/devfoo"), "root:devfoo");
        assert_eq!(route(&mut table, "/dev/null"), "dev:null");
        assert_eq!(route(&mut table, "/dev"), "dev:");
    }

    #[test]
    fn longest_prefix_picks_deepest_mount() {
        let mut table = MountTable::new();
        table.mount("/", tagged("root"));
        table.mount("/dev", tagged("dev"));
        table.mount("/dev/pts", tagged("pts"));

        assert_eq!(route(&mut table, "/etc/passwd"), "root:etc/passwd");
        assert_eq!(route(&mut table, "/dev/console"), "dev:console");
        assert_eq!(route(&mut table, "/dev/pts/0"), "pts:0");
        // Mount order shouldn't matter for the longest-prefix outcome.
        assert_eq!(route(&mut table, "/dev/pts"), "pts:");
    }

    #[test]
    fn stat_of_bare_mount_point_works() {
        let mut table = MountTable::new();
        table.mount("/", tagged("root"));
        table.mount("/dev", tagged("dev"));

        // Must return Some (the backend's root attrs), never None/ENOENT.
        let attrs = table.stat("/dev").expect("stat of mount point itself");
        assert_eq!(attrs.size, 0); // rel == "" -> len 0
        assert_eq!(route(&mut table, "/dev"), "dev:");
    }

    #[test]
    fn dot_dotdot_and_double_slash_route_identically() {
        let mut table = MountTable::new();
        table.mount("/", tagged("root"));
        table.mount("/work", tagged("work"));

        let a = route(&mut table, "/work/x/./y");
        let b = route(&mut table, "/work/x//y");
        let c = route(&mut table, "/work/x/sub/../y");
        assert_eq!(a, "work:x/y");
        assert_eq!(a, b);
        assert_eq!(b, c);
    }

    #[test]
    fn directory_symlinks_resolve_across_mounts() {
        let mut t = MountTable::new();
        t.mount("/", Box::new(crate::fs::TmpFs::new()));
        t.mount("/run", Box::new(crate::fs::TmpFs::new()));
        for d in [
            "/var",
            "/var/log",
            "/var/log/nginx",
            "/var/lib",
            "/var/lib/nginx",
        ] {
            t.mkdir(d, 0o755).unwrap();
        }
        t.mkdir("/run/nginx", 0o755).unwrap();
        // Absolute, relative, and cross-mount links to directories.
        t.symlink("/var/log/nginx", "/var/lib/nginx/logs").unwrap();
        t.symlink("../log", "/var/lib/oldlog").unwrap();
        t.symlink("/run/nginx", "/var/lib/nginx/run").unwrap();
        t.create("/var/lib/nginx/logs/error.log", 0o644).unwrap();
        assert!(t.stat("/var/log/nginx/error.log").is_some());
        assert!(t.stat("/var/lib/oldlog/nginx/error.log").is_some());
        t.create("/var/lib/nginx/run/nginx.pid", 0o644).unwrap();
        assert!(
            t.readdir("/run/nginx")
                .unwrap()
                .iter()
                .any(|e| e.name == "nginx.pid")
        );
        // The final component itself is not followed.
        assert_eq!(
            t.stat("/var/lib/nginx/logs").unwrap().kind,
            NodeKind::Symlink
        );
        // A loop fails instead of hanging.
        t.symlink("/var/loop", "/var/loop").unwrap();
        assert!(t.stat("/var/loop/x").is_none());
    }

    #[test]
    fn trailing_slash_resolves() {
        let mut table = MountTable::new();
        table.mount("/", tagged("root"));
        table.mount("/work", tagged("work"));

        assert_eq!(route(&mut table, "/work/"), "work:");
        assert_eq!(route(&mut table, "/work"), "work:");
        assert_eq!(route(&mut table, "/work/sub/"), "work:sub");
    }

    #[test]
    fn root_fallback_catches_unmounted_paths() {
        let mut table = MountTable::new();
        table.mount("/", tagged("root"));
        table.mount("/work", tagged("work"));

        assert_eq!(route(&mut table, "/etc/hosts"), "root:etc/hosts");
        assert_eq!(route(&mut table, "/"), "root:");
    }

    #[test]
    fn mount_point_itself_is_normalized() {
        let mut table = MountTable::new();
        table.mount("/", tagged("root"));
        table.mount("/dev/", tagged("dev")); // trailing slash at registration

        assert_eq!(route(&mut table, "/dev/null"), "dev:null");
        assert_eq!(route(&mut table, "/dev"), "dev:");
    }
}
