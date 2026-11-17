//! Copy-on-write overlay filesystem.
//!
//! Presents a merged view of a read-only `lower` (e.g. the squashfs root) and a
//! writable `upper`. Reads prefer the upper and fall through to the lower; the
//! first write to a lower-only path *copies it up* into the upper; deletions
//! record a *whiteout* that hides the lower entry. The lower is never modified.
//!
//! With a [`TmpFs`](super::TmpFs) upper, `/` is mutable but ephemeral. With an
//! upper that outlives the guest — a host directory, see
//! [`Overlay::persistent`] — what one guest installs is there for the next.
//!
//! ## What the upper layer holds
//!
//! The upper is an ordinary filesystem tree: files written or copied up, at
//! their paths. What the overlay itself needs to remember is kept in the same
//! tree as empty marker files, in the AUFS / OCI-layer convention (no device
//! nodes, no extended attributes, so any host directory can hold them):
//!
//! * **whiteout** — `dir/.wh.<name>` says that the lower layer's `dir/<name>`
//!   was deleted and is hidden. It is written only when there is a lower entry
//!   to hide.
//! * **opaque directory** — `dir/.wh..wh..opq` says that nothing of the lower
//!   layer's `dir` shows through: `dir` was deleted and made again, or a
//!   directory was renamed onto it.
//! * **lock** — `.wh..wh..lock` at the top of a persistent upper, held while a
//!   guest uses it (see [`Overlay::persistent`]).
//!
//! Every name starting with `.wh.` belongs to the overlay: the guest sees none
//! of them in the upper, and cannot create one (`EPERM`). An upper written this
//! way reads the same as a Docker image layer, and can be inspected — or
//! repaired — with ordinary tools on the host.

use std::collections::{BTreeMap, BTreeSet};
use std::io;

use super::{Attrs, DirEntry, MountFs, NodeKind};

/// High bit set on an upper-layer inode so it can never collide with a
/// lower-layer one. The upper (tmpfs, inodes from 2) and the lower (squashfs,
/// its own image inodes) each start numbering low, so without this an
/// apk-installed file and a base-image file share an inode number — and since
/// every node reports the same `st_dev`, `(dev, ino)` collides. musl's dynamic
/// linker dedups already-loaded libraries by `(dev, ino)`, so a collision made
/// it skip loading a library (e.g. libbrotlicommon), and the symbols it
/// exported went "not found" — which is what stopped `node` from starting.
const UPPER_INODE_TAG: u64 = 1 << 63;

/// The prefix of every name the overlay keeps for itself in the upper layer.
pub const MARKER_PREFIX: &str = ".wh.";
/// The marker that makes a directory opaque.
pub const OPAQUE_MARKER: &str = ".wh..wh..opq";
/// The lock file at the top of a persistent upper.
pub const LOCK_MARKER: &str = ".wh..wh..lock";

/// Tag an upper-layer node's inode as distinct from the lower layer's.
fn tag_upper(mut a: Attrs) -> Attrs {
    a.inode |= UPPER_INODE_TAG;
    a
}

/// What the upper layer's markers say about one directory.
#[derive(Debug, Default)]
struct DirMarks {
    /// Names of the lower layer's entries hidden by a whiteout.
    whiteouts: BTreeSet<String>,
    /// Nothing of the lower layer's directory shows.
    opaque: bool,
}

/// Which layer a path's node is in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layer {
    Upper,
    Lower,
}

#[derive(Debug)]
pub struct Overlay {
    upper: Box<dyn MountFs>,
    lower: Box<dyn MountFs>,
    /// The upper layer's markers, read per directory on first use and kept in
    /// step with every change the overlay makes. The overlay is the upper's
    /// only writer while it is mounted, so this never goes stale.
    marks: BTreeMap<String, DirMarks>,
    /// A persistent upper's lock, held for as long as this overlay exists.
    _lock: Option<std::fs::File>,
}

impl Overlay {
    /// Build an overlay with `upper` (writable) over `lower` (read-only).
    ///
    /// `upper` may already hold a previous overlay's work — its files,
    /// whiteouts and opaque markers (see the [module docs](self)) — which is
    /// then what the merged view starts from.
    #[must_use]
    pub fn new(lower: Box<dyn MountFs>, upper: Box<dyn MountFs>) -> Self {
        Self::with_lock(lower, upper, None)
    }

    fn with_lock(
        lower: Box<dyn MountFs>,
        upper: Box<dyn MountFs>,
        lock: Option<std::fs::File>,
    ) -> Self {
        let mut o = Self {
            upper,
            lower,
            marks: BTreeMap::new(),
            _lock: lock,
        };
        // The merged root is the upper's, which must look like the lower's —
        // its mode and times (it stays root's: a host directory's owner isn't
        // the guest's business).
        if let Some(a) = o.lower.stat("") {
            let _ = o.upper.set_mode("", a.mode & 0o7777);
            let t = |sec| super::SetTime::Set { sec, nsec: 0 };
            let _ = o.upper.set_times("", t(a.atime), t(a.mtime));
        }
        o
    }

    /// An overlay whose upper layer is the host directory `upper_dir` (made if
    /// missing), so the guest's changes to `/` — `apk add`, a config edit, a
    /// deleted file — are on the host's disk, and the next overlay made over
    /// the same directory starts from them.
    ///
    /// The directory holds the changed files at their guest paths and the
    /// overlay's markers (see the [module docs](self)). Keep one per lower
    /// image: over a different image, the files copied up from the old one
    /// shadow the new one's.
    ///
    /// One guest at a time: the overlay holds a lock on
    /// `upper_dir/.wh..wh..lock` until it is dropped, and a second overlay on
    /// the same directory meanwhile fails with `EBUSY` (where the host has no
    /// file locks, this isn't checked).
    ///
    /// # Errors
    /// The directory can't be made, or another overlay holds it (`EBUSY`).
    #[cfg(unix)]
    pub fn persistent(
        lower: Box<dyn MountFs>,
        upper_dir: impl AsRef<std::path::Path>,
    ) -> io::Result<Self> {
        let dir = upper_dir.as_ref();
        std::fs::create_dir_all(dir)?;
        let lock = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(dir.join(LOCK_MARKER))?;
        match lock.try_lock() {
            Ok(()) => {}
            Err(std::fs::TryLockError::WouldBlock) => {
                return Err(io::Error::from_raw_os_error(16)); // EBUSY
            }
            Err(std::fs::TryLockError::Error(e)) if e.kind() == io::ErrorKind::Unsupported => {}
            Err(std::fs::TryLockError::Error(e)) => return Err(e),
        }
        Ok(Self::with_lock(
            lower,
            Box::new(super::Passthrough::new(dir)),
            Some(lock),
        ))
    }

    /// The upper layer, once the overlay is done with it (e.g. to mount it
    /// under another overlay later).
    #[must_use]
    pub fn into_upper(self) -> Box<dyn MountFs> {
        self.upper
    }

    /// The markers of upper directory `dir`, read from the upper on first use.
    fn marks(&mut self, dir: &str) -> &mut DirMarks {
        if !self.marks.contains_key(dir) {
            let mut m = DirMarks::default();
            match self.upper.stat(dir).map(|a| a.kind) {
                // A non-directory in the upper hides whatever the lower has
                // under that name.
                Some(k) if k != NodeKind::Dir => m.opaque = true,
                _ => {}
            }
            if !m.opaque
                && let Ok(entries) = self.upper.readdir(dir)
            {
                for e in entries {
                    if e.name == OPAQUE_MARKER {
                        m.opaque = true;
                    } else if let Some(name) = e.name.strip_prefix(MARKER_PREFIX)
                        && !name.starts_with(MARKER_PREFIX)
                        && !name.is_empty()
                    {
                        m.whiteouts.insert(name.to_string());
                    }
                }
            }
            self.marks.insert(dir.to_string(), m);
        }
        self.marks.get_mut(dir).expect("just inserted")
    }

    /// Forget the markers read for `dir` and everything under it (it was
    /// removed or moved); they are read again on next use.
    fn forget_marks(&mut self, dir: &str) {
        if dir.is_empty() {
            self.marks.clear();
            return;
        }
        let sub = format!("{dir}/");
        self.marks.retain(|k, _| k != dir && !k.starts_with(&sub));
    }

    /// Is `rel` (or an ancestor) hidden by a whiteout?
    fn is_whited(&mut self, rel: &str) -> bool {
        let mut p = rel;
        while !p.is_empty() {
            let (dir, name) = split(p);
            if self.marks(dir).whiteouts.contains(name) {
                return true;
            }
            p = dir;
        }
        false
    }

    /// Is the lower layer's `rel` out of the merged view — whited out, or under
    /// an opaque directory?
    fn lower_hidden(&mut self, rel: &str) -> bool {
        if self.is_whited(rel) {
            return true;
        }
        let mut p = rel;
        while !p.is_empty() {
            p = parent_of(p);
            if self.marks(p).opaque {
                return true;
            }
        }
        false
    }

    /// Does the merged view show the lower layer's `rel`?
    fn lower_visible(&mut self, rel: &str) -> bool {
        self.lower.stat(rel).is_some() && !self.lower_hidden(rel)
    }

    /// Does the merged view of directory `rel` show the lower's entries?
    fn lower_dir_shows(&mut self, rel: &str) -> bool {
        !self.marks(rel).opaque && self.lower_visible(rel)
    }

    /// The layer `rel` is read from, if it's in the merged view.
    fn layer(&mut self, rel: &str) -> Option<Layer> {
        if is_marker(rel) || self.is_whited(rel) {
            return None;
        }
        if self.upper.stat(rel).is_some() {
            Some(Layer::Upper)
        } else if self.lower_visible(rel) {
            Some(Layer::Lower)
        } else {
            None
        }
    }

    /// Hide the lower layer's `rel`, if there is one to hide.
    fn add_whiteout(&mut self, rel: &str, lower_was_visible: bool) -> io::Result<()> {
        if !lower_was_visible {
            return Ok(());
        }
        let (dir, name) = split(rel);
        self.ensure_dir_in_upper(dir);
        let marker = join(dir, &format!("{MARKER_PREFIX}{name}"));
        match self.upper.create(&marker, 0o600) {
            Ok(()) => {}
            Err(e) if e.raw_os_error() == Some(17) => {} // EEXIST: already hidden
            Err(e) => return Err(e),
        }
        self.marks(dir).whiteouts.insert(name.to_string());
        Ok(())
    }

    /// Drop the whiteout on `rel`, if any; whether there was one.
    fn clear_whiteout(&mut self, rel: &str) -> io::Result<bool> {
        let (dir, name) = split(rel);
        if !self.marks(dir).whiteouts.contains(name) {
            return Ok(false);
        }
        let marker = join(dir, &format!("{MARKER_PREFIX}{name}"));
        match self.upper.unlink(&marker) {
            Ok(()) => {}
            Err(e) if e.raw_os_error() == Some(2) => {}
            Err(e) => return Err(e),
        }
        self.marks(dir).whiteouts.remove(name);
        Ok(true)
    }

    /// Make upper directory `dir` opaque: none of the lower's `dir` shows.
    fn set_opaque(&mut self, dir: &str) -> io::Result<()> {
        match self.upper.create(&join(dir, OPAQUE_MARKER), 0o600) {
            Ok(()) => {}
            Err(e) if e.raw_os_error() == Some(17) => {}
            Err(e) => return Err(e),
        }
        self.marks(dir).opaque = true;
        Ok(())
    }

    /// Remove the markers in upper directory `dir`, so that it can be removed.
    fn clear_markers(&mut self, dir: &str) -> io::Result<()> {
        if let Ok(entries) = self.upper.readdir(dir) {
            for e in entries {
                if e.name.starts_with(MARKER_PREFIX) {
                    self.upper.unlink(&join(dir, &e.name))?;
                }
            }
        }
        self.forget_marks(dir);
        Ok(())
    }

    /// Refuse a name the overlay keeps for itself.
    fn check_name(rel: &str) -> io::Result<()> {
        if is_marker(rel) {
            Err(io::Error::from_raw_os_error(1)) // EPERM
        } else {
            Ok(())
        }
    }

    /// Give `rel`'s fresh upper copy the lower node's permissions, owner and
    /// times, as overlayfs's copy-up does.
    fn copy_attrs_up(&mut self, rel: &str) {
        let Some(a) = self.lower.stat(rel) else {
            return;
        };
        if a.kind != NodeKind::Symlink {
            let _ = self.upper.set_mode(rel, a.mode & 0o7777);
        }
        let _ = self.upper.set_owner(rel, Some(a.uid), Some(a.gid));
        let t = |sec| super::SetTime::Set { sec, nsec: 0 };
        let _ = self.upper.set_times(rel, t(a.atime), t(a.mtime));
    }

    /// Ensure directory `dir` exists in the upper layer, creating ancestors.
    fn ensure_dir_in_upper(&mut self, dir: &str) {
        if dir.is_empty() || self.upper.stat(dir).is_some() {
            return;
        }
        self.ensure_dir_in_upper(parent_of(dir));
        if self.upper.mkdir(dir, 0o755).is_ok() {
            self.copy_attrs_up(dir);
        }
    }

    /// Read a lower file's full contents.
    fn read_all_lower(&mut self, rel: &str) -> Vec<u8> {
        let size = self.lower.stat(rel).map_or(0, |a| a.size) as usize;
        let mut data = vec![0u8; size];
        let mut off = 0;
        while off < size {
            match self.lower.read_at(rel, off as u64, &mut data[off..]) {
                Ok(0) | Err(_) => break,
                Ok(n) => off += n,
            }
        }
        data.truncate(off);
        data
    }

    /// Carry `rel`'s lower-layer extended attributes onto its fresh upper copy
    /// (Linux overlayfs copies xattrs up with the data), so e.g. a `security.
    /// capability` on a base-image binary survives the first write to it.
    fn copy_up_xattrs(&mut self, rel: &str) {
        let Ok(names) = self.lower.listxattr(rel) else {
            return;
        };
        for name in names {
            if let Ok(v) = self.lower.getxattr(rel, &name) {
                let _ = self.upper.setxattr(rel, &name, &v);
            }
        }
    }

    /// Make `rel` itself writable in the upper layer: a file or symlink with
    /// its contents, a directory as an empty one (the merged `readdir` still
    /// shows the lower children beneath it). The caller has checked that `rel`
    /// is in the merged view.
    fn copy_up_node(&mut self, rel: &str) -> io::Result<()> {
        if self.upper.stat(rel).is_some() {
            return Ok(());
        }
        let Some(attrs) = self.lower.stat(rel) else {
            return Ok(()); // nothing to copy; caller creates fresh
        };
        self.ensure_dir_in_upper(parent_of(rel));
        match attrs.kind {
            NodeKind::Dir => self.ensure_dir_in_upper(rel),
            NodeKind::Symlink => {
                let target = self.lower.readlink(rel)?;
                self.upper.symlink(&target, rel)?;
                self.copy_attrs_up(rel);
            }
            _ => {
                let data = self.read_all_lower(rel);
                self.upper.create(rel, attrs.mode & 0o7777)?;
                self.upper.write_at(rel, 0, &data)?;
                self.copy_attrs_up(rel);
            }
        }
        self.copy_up_xattrs(rel);
        Ok(())
    }

    /// Copy `rel` and, for a directory, everything of the lower layer's under
    /// it that the merged view shows, into the upper — so that renaming it
    /// moves its whole subtree rather than an empty shell.
    fn copy_up_tree(&mut self, rel: &str) -> io::Result<()> {
        self.copy_up_node(rel)?;
        let is_dir = self
            .upper
            .stat(rel)
            .is_some_and(|a| a.kind == NodeKind::Dir);
        if !is_dir || !self.lower_dir_shows(rel) {
            return Ok(());
        }
        if let Ok(entries) = self.lower.readdir(rel) {
            for e in entries {
                let child = join(rel, &e.name);
                if !self.is_whited(&child) {
                    self.copy_up_tree(&child)?;
                }
            }
        }
        Ok(())
    }

    /// Clear the way for a new node at `rel`: lift its whiteout and make sure
    /// its parent is in the upper. Whether `rel` was whited out.
    fn prepare_new(&mut self, rel: &str) -> io::Result<bool> {
        Self::check_name(rel)?;
        if self.stat(rel).is_some() {
            return Err(eexist());
        }
        let was_whited = self.clear_whiteout(rel)?;
        self.ensure_dir_in_upper(parent_of(rel));
        self.forget_marks(rel);
        Ok(was_whited)
    }
}

/// Is `rel`'s last component one of the overlay's own names?
fn is_marker(rel: &str) -> bool {
    split(rel).1.starts_with(MARKER_PREFIX)
}

/// `rel` as (parent directory, last component).
fn split(rel: &str) -> (&str, &str) {
    match rel.rfind('/') {
        Some(i) => (&rel[..i], &rel[i + 1..]),
        None => ("", rel),
    }
}

fn parent_of(rel: &str) -> &str {
    split(rel).0
}

fn join(dir: &str, name: &str) -> String {
    if dir.is_empty() {
        name.to_string()
    } else {
        format!("{dir}/{name}")
    }
}

fn enoent() -> io::Error {
    io::Error::from_raw_os_error(2)
}
fn eexist() -> io::Error {
    io::Error::from_raw_os_error(17)
}
fn enotdir() -> io::Error {
    io::Error::from_raw_os_error(20)
}
fn eisdir() -> io::Error {
    io::Error::from_raw_os_error(21)
}
fn einval() -> io::Error {
    io::Error::from_raw_os_error(22)
}
fn enodata() -> io::Error {
    io::Error::from_raw_os_error(61)
}
fn enotempty() -> io::Error {
    io::Error::from_raw_os_error(39)
}

impl MountFs for Overlay {
    fn fs_type(&self) -> &'static str {
        "overlay"
    }

    fn attach_quota(&mut self, quota: &std::sync::Arc<crate::limits::Quota>) {
        self.upper.attach_quota(quota);
    }

    fn attach_disk_quota(&mut self, quota: &std::sync::Arc<crate::limits::Quota>) {
        self.upper.attach_disk_quota(quota);
    }

    fn attach_disks(&mut self, disks: &super::blockdev::SharedDisks) {
        self.lower.attach_disks(disks);
        self.upper.attach_disks(disks);
    }

    fn data_dev(&mut self, rel: &str) -> Option<u64> {
        match self.layer(rel) {
            Some(Layer::Upper) => self.upper.data_dev(rel),
            Some(Layer::Lower) => self.lower.data_dev(rel),
            None => None,
        }
    }

    fn read_only(&self) -> bool {
        false
    }

    fn stat(&mut self, rel: &str) -> Option<Attrs> {
        match self.layer(rel)? {
            Layer::Upper => self.upper.stat(rel).map(tag_upper),
            Layer::Lower => self.lower.stat(rel),
        }
    }

    fn read_at(&mut self, rel: &str, off: u64, buf: &mut [u8]) -> io::Result<usize> {
        match self.layer(rel) {
            Some(Layer::Upper) => self.upper.read_at(rel, off, buf),
            Some(Layer::Lower) => self.lower.read_at(rel, off, buf),
            None => Err(enoent()),
        }
    }

    fn readdir(&mut self, rel: &str) -> io::Result<Vec<DirEntry>> {
        let Some(layer) = self.layer(rel) else {
            return Err(enoent());
        };
        let mut seen: BTreeMap<String, DirEntry> = BTreeMap::new();
        // The lower's entries, unless the directory is opaque (or only the
        // upper's).
        if self.lower_dir_shows(rel)
            && let Ok(entries) = self.lower.readdir(rel)
        {
            let marks = self.marks(rel);
            for e in entries {
                if !marks.whiteouts.contains(&e.name) {
                    seen.insert(e.name.clone(), e);
                }
            }
        }
        if layer == Layer::Upper
            && let Ok(entries) = self.upper.readdir(rel)
        {
            for mut e in entries {
                if e.name.starts_with(MARKER_PREFIX) {
                    continue;
                }
                e.inode |= UPPER_INODE_TAG; // match `stat`'s upper-layer tag
                seen.insert(e.name.clone(), e); // upper shadows lower
            }
        }
        Ok(seen.into_values().collect())
    }

    fn write_at(&mut self, rel: &str, off: u64, buf: &[u8]) -> io::Result<usize> {
        if self.layer(rel).is_none() {
            return Err(enoent());
        }
        self.copy_up_node(rel)?;
        self.upper.write_at(rel, off, buf)
    }

    fn create(&mut self, rel: &str, mode: u32) -> io::Result<()> {
        self.prepare_new(rel)?;
        self.upper.create(rel, mode)
    }

    fn mkdir(&mut self, rel: &str, mode: u32) -> io::Result<()> {
        let was_whited = self.prepare_new(rel)?;
        self.upper.mkdir(rel, mode)?;
        // A directory made where a lower one was deleted starts empty.
        if was_whited {
            self.set_opaque(rel)?;
        }
        Ok(())
    }

    fn mknod(&mut self, rel: &str, mode: u32) -> io::Result<()> {
        self.prepare_new(rel)?;
        self.upper.mknod(rel, mode)
    }

    fn unlink(&mut self, rel: &str) -> io::Result<()> {
        match self.stat(rel) {
            None => return Err(enoent()),
            Some(a) if a.kind == NodeKind::Dir => return Err(eisdir()),
            Some(_) => {}
        }
        let lower_was_visible = self.lower_visible(rel);
        if self.upper.stat(rel).is_some() {
            self.upper.unlink(rel)?;
        }
        self.forget_marks(rel);
        self.add_whiteout(rel, lower_was_visible)
    }

    fn rmdir(&mut self, rel: &str) -> io::Result<()> {
        match self.stat(rel) {
            Some(a) if a.kind == NodeKind::Dir => {}
            Some(_) => return Err(enotdir()),
            None => return Err(enoent()),
        }
        if !self.readdir(rel)?.is_empty() {
            return Err(enotempty());
        }
        let lower_was_visible = self.lower_visible(rel);
        if self.upper.stat(rel).is_some() {
            self.clear_markers(rel)?;
            self.upper.rmdir(rel)?;
        }
        self.add_whiteout(rel, lower_was_visible)
    }

    fn truncate(&mut self, rel: &str, len: u64) -> io::Result<()> {
        if self.layer(rel).is_none() {
            return Err(enoent());
        }
        self.copy_up_node(rel)?;
        self.upper.truncate(rel, len)
    }

    fn set_times(
        &mut self,
        rel: &str,
        atime: super::SetTime,
        mtime: super::SetTime,
    ) -> io::Result<()> {
        if self.layer(rel).is_none() {
            return Err(enoent());
        }
        // Copy the node up so the timestamp change lands on a writable node
        // (the lower layer is immutable), then apply it there.
        self.copy_up_node(rel)?;
        self.upper.set_times(rel, atime, mtime)
    }

    fn set_mode(&mut self, rel: &str, mode: u32) -> io::Result<()> {
        if self.layer(rel).is_none() {
            return Err(enoent());
        }
        self.copy_up_node(rel)?;
        self.upper.set_mode(rel, mode)
    }

    fn set_owner(&mut self, rel: &str, uid: Option<u32>, gid: Option<u32>) -> io::Result<()> {
        if self.layer(rel).is_none() {
            return Err(enoent());
        }
        self.copy_up_node(rel)?;
        self.upper.set_owner(rel, uid, gid)
    }

    fn getxattr(&mut self, rel: &str, name: &str) -> io::Result<Vec<u8>> {
        match self.layer(rel) {
            Some(Layer::Upper) => self.upper.getxattr(rel, name),
            // The merged filesystem supports xattrs even where the lower layer
            // doesn't: an attribute it can't produce is simply absent.
            Some(Layer::Lower) => match self.lower.getxattr(rel, name) {
                Err(e) if e.raw_os_error() == Some(95) => Err(enodata()),
                r => r,
            },
            None => Err(enoent()),
        }
    }

    fn setxattr(&mut self, rel: &str, name: &str, value: &[u8]) -> io::Result<()> {
        if self.layer(rel).is_none() {
            return Err(enoent());
        }
        self.copy_up_node(rel)?;
        self.upper.setxattr(rel, name, value)
    }

    fn listxattr(&mut self, rel: &str) -> io::Result<Vec<String>> {
        match self.layer(rel) {
            Some(Layer::Upper) => self.upper.listxattr(rel),
            Some(Layer::Lower) => Ok(self.lower.listxattr(rel).unwrap_or_default()),
            None => Err(enoent()),
        }
    }

    fn removexattr(&mut self, rel: &str, name: &str) -> io::Result<()> {
        match self.layer(rel) {
            None => return Err(enoent()),
            Some(Layer::Lower) => {
                // Lower-only: nothing to remove unless the lower node carries it.
                if self.lower.getxattr(rel, name).is_err() {
                    return Err(enodata());
                }
                self.copy_up_node(rel)?;
            }
            Some(Layer::Upper) => {}
        }
        self.upper.removexattr(rel, name)
    }

    fn symlink(&mut self, target: &str, linkpath: &str) -> io::Result<()> {
        self.prepare_new(linkpath)?;
        self.upper.symlink(target, linkpath)
    }

    /// A hard link: the source is copied up (if it lives only in the lower
    /// layer) and linked within the upper layer, so both names share one
    /// upper inode.
    fn link(&mut self, old_rel: &str, new_rel: &str) -> io::Result<()> {
        match self.stat(old_rel) {
            None => return Err(enoent()),
            Some(a) if a.kind == NodeKind::Dir => return Err(io::Error::from_raw_os_error(1)),
            Some(_) => {}
        }
        Self::check_name(new_rel)?;
        if self.stat(new_rel).is_some() {
            return Err(eexist());
        }
        self.copy_up_node(old_rel)?;
        self.prepare_new(new_rel)?;
        self.upper.link(old_rel, new_rel)
    }

    fn readlink(&mut self, rel: &str) -> io::Result<String> {
        match self.layer(rel) {
            Some(Layer::Upper) => self.upper.readlink(rel),
            Some(Layer::Lower) => self.lower.readlink(rel),
            None => Err(enoent()),
        }
    }

    fn rename(&mut self, from: &str, to: &str) -> io::Result<()> {
        if from == to {
            return if self.stat(from).is_some() {
                Ok(())
            } else {
                Err(enoent())
            };
        }
        Self::check_name(to)?;
        let Some(from_attrs) = self.stat(from) else {
            return Err(enoent());
        };
        let from_is_dir = from_attrs.kind == NodeKind::Dir;
        // Refuse to move a directory into itself or one of its own
        // descendants.
        if from_is_dir && to.starts_with(&format!("{from}/")) {
            return Err(einval());
        }
        if let Some(to_attrs) = self.stat(to) {
            let to_is_dir = to_attrs.kind == NodeKind::Dir;
            match (from_is_dir, to_is_dir) {
                // Directory onto directory: only if the destination is empty
                // in the merged view.
                (true, true) => {
                    if !self.readdir(to)?.is_empty() {
                        return Err(enotempty());
                    }
                }
                (true, false) => return Err(enotdir()),
                (false, true) => return Err(eisdir()),
                (false, false) => {}
            }
            // Clear the destination from the upper layer (if present there);
            // any lower counterpart is superseded once `to` reappears in the
            // upper below.
            if self.upper.stat(to).is_some() {
                if to_is_dir {
                    self.clear_markers(to)?;
                    self.upper.rmdir(to)?;
                } else {
                    self.upper.unlink(to)?;
                }
            }
        }
        let lower_had_from = self.lower_visible(from);

        // Copy the whole `from` subtree into the upper layer first (a no-op
        // for anything already upper-resident), then rename within the
        // upper — the lower layer is never mutated.
        self.copy_up_tree(from)?;
        self.ensure_dir_in_upper(parent_of(to));
        self.clear_whiteout(to)?;
        self.upper.rename(from, to)?;
        self.forget_marks(from);
        self.forget_marks(to);
        if from_is_dir {
            // The moved directory has all of its contents in the upper now: a
            // lower directory at `to` must not show through it.
            if self.lower.stat(to).is_some() {
                self.set_opaque(to)?;
            }
        }
        self.add_whiteout(from, lower_had_from) // hide any lower `from`
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fs::TmpFs;

    /// A lower layer with `/f` = "low" and directory `/d` containing `/d/x`.
    fn seeded_lower() -> Box<dyn MountFs> {
        let mut fs = TmpFs::new();
        fs.create("f", 0o644).unwrap();
        fs.write_at("f", 0, b"low").unwrap();
        fs.mkdir("d", 0o755).unwrap();
        fs.create("d/x", 0o644).unwrap();
        Box::new(fs)
    }

    fn overlay() -> Overlay {
        Overlay::new(seeded_lower(), Box::new(TmpFs::new()))
    }

    #[test]
    fn reads_fall_through_to_lower() {
        let mut o = overlay();
        let mut buf = [0u8; 3];
        assert_eq!(o.read_at("f", 0, &mut buf).unwrap(), 3);
        assert_eq!(&buf, b"low");
    }

    #[test]
    fn write_copies_up_without_touching_lower() {
        let mut lower = seeded_lower();
        let mut o = Overlay::new(
            {
                // fresh identical lower we can inspect separately
                let mut fs = TmpFs::new();
                fs.create("f", 0o644).unwrap();
                fs.write_at("f", 0, b"low").unwrap();
                Box::new(fs)
            },
            Box::new(TmpFs::new()),
        );
        o.write_at("f", 0, b"UP!").unwrap();
        let mut buf = [0u8; 3];
        o.read_at("f", 0, &mut buf).unwrap();
        assert_eq!(&buf, b"UP!", "overlay sees the new contents");
        // The original lower is untouched.
        let mut lb = [0u8; 3];
        lower.read_at("f", 0, &mut lb).unwrap();
        assert_eq!(&lb, b"low");
    }

    #[test]
    fn whiteout_hides_lower_entry() {
        let mut o = overlay();
        assert!(o.stat("f").is_some());
        o.unlink("f").unwrap();
        assert!(o.stat("f").is_none(), "unlinked entry hidden");
        assert!(o.read_at("f", 0, &mut [0u8; 3]).is_err());
    }

    #[test]
    fn readdir_merges_layers_minus_whiteouts() {
        let mut o = overlay();
        o.create("newfile", 0o644).unwrap();
        let mut names: Vec<_> = o.readdir("").unwrap().into_iter().map(|e| e.name).collect();
        names.sort();
        assert_eq!(names, vec!["d", "f", "newfile"]);

        o.unlink("f").unwrap();
        let mut names: Vec<_> = o.readdir("").unwrap().into_iter().map(|e| e.name).collect();
        names.sort();
        assert_eq!(names, vec!["d", "newfile"]);
    }

    #[test]
    fn create_in_lower_only_subdir_copies_parent_up() {
        let mut o = overlay();
        // Write into a file under the lower-only dir /d.
        o.write_at("d/x", 0, b"z").unwrap();
        let mut buf = [0u8; 1];
        o.read_at("d/x", 0, &mut buf).unwrap();
        assert_eq!(&buf, b"z");
    }

    #[test]
    fn unlink_lower_file_removes_it_from_readdir_and_stat() {
        let mut o = overlay();
        o.unlink("f").unwrap();
        assert!(o.stat("f").is_none());
        assert_eq!(
            o.read_at("f", 0, &mut [0u8; 3]).unwrap_err().raw_os_error(),
            Some(2) // ENOENT
        );
        let names: Vec<_> = o.readdir("").unwrap().into_iter().map(|e| e.name).collect();
        assert!(!names.contains(&"f".to_string()));
    }

    #[test]
    fn recreating_after_unlink_clears_the_whiteout() {
        let mut o = overlay();
        o.unlink("f").unwrap();
        assert!(o.stat("f").is_none());
        o.create("f", 0o644).unwrap();
        o.write_at("f", 0, b"new").unwrap();
        assert!(o.stat("f").is_some());
        let mut buf = [0u8; 3];
        o.read_at("f", 0, &mut buf).unwrap();
        assert_eq!(&buf, b"new");
        let names: Vec<_> = o.readdir("").unwrap().into_iter().map(|e| e.name).collect();
        assert!(names.contains(&"f".to_string()));
    }

    #[test]
    fn readdir_dedups_name_present_in_both_layers() {
        let mut o = overlay();
        // After a write, "f" is copied up: it now exists in both upper and
        // lower under the same name.
        o.write_at("f", 0, b"UP!").unwrap();
        let names: Vec<_> = o.readdir("").unwrap().into_iter().map(|e| e.name).collect();
        assert_eq!(names.iter().filter(|n| n.as_str() == "f").count(), 1);
    }

    #[test]
    fn recreating_dir_after_rmdir_does_not_leak_lower_children() {
        let mut o = overlay();
        // "d" (lower) contains only "d/x"; empty it out, then rmdir it.
        o.unlink("d/x").unwrap();
        o.rmdir("d").unwrap();
        assert!(o.stat("d").is_none());

        // Recreate "d" fresh and populate it differently.
        o.mkdir("d", 0o755).unwrap();
        o.create("d/y", 0o644).unwrap();

        let names: Vec<_> = o
            .readdir("d")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert_eq!(names, vec!["y".to_string()]);
        assert!(o.stat("d/x").is_none(), "stale lower child must not leak");
    }

    #[test]
    fn mkdir_and_create_reject_existing_lower_entries() {
        let mut o = overlay();
        assert_eq!(o.mkdir("d", 0o755).unwrap_err().raw_os_error(), Some(17)); // EEXIST
        assert_eq!(o.create("f", 0o644).unwrap_err().raw_os_error(), Some(17)); // EEXIST
    }

    #[test]
    fn rename_copies_up_lower_file_then_moves_it() {
        let mut lower_check = seeded_lower();
        let mut o = overlay();
        o.rename("f", "g").unwrap();
        assert!(o.stat("f").is_none());
        let mut buf = [0u8; 3];
        o.read_at("g", 0, &mut buf).unwrap();
        assert_eq!(&buf, b"low");
        // The lower layer was never touched.
        let mut lb = [0u8; 3];
        lower_check.read_at("f", 0, &mut lb).unwrap();
        assert_eq!(&lb, b"low");
    }

    #[test]
    fn upper_and_lower_inodes_never_collide() {
        // The lower's `/f` and a fresh upper file can have the same raw inode
        // (both tmpfs, both counting from low numbers). Without the layer tag
        // they'd report the same inode — which broke musl's `(dev,ino)` library
        // dedup and stopped node from loading. They must differ here.
        let mut o = overlay();
        o.create("g", 0o644).unwrap(); // upper-only file
        let lower_ino = o.stat("f").unwrap().inode; // lower `/f`
        let upper_ino = o.stat("g").unwrap().inode; // upper `/g`
        assert_ne!(
            lower_ino, upper_ino,
            "upper and lower inodes must be distinct"
        );
        assert_eq!(
            upper_ino & UPPER_INODE_TAG,
            UPPER_INODE_TAG,
            "upper is tagged"
        );
        assert_eq!(lower_ino & UPPER_INODE_TAG, 0, "lower is untagged");
        // readdir agrees with stat on the upper tag.
        let g = o
            .readdir("")
            .unwrap()
            .into_iter()
            .find(|e| e.name == "g")
            .unwrap();
        assert_eq!(g.inode, upper_ino, "readdir and stat agree on the inode");
    }

    #[test]
    fn rename_lower_only_dir_brings_its_subtree() {
        let mut o = overlay();
        o.rename("d", "e").unwrap();
        assert!(o.stat("d").is_none());
        let mut buf = [0u8; 1];
        o.read_at("e/x", 0, &mut buf).unwrap();
        let names: Vec<_> = o
            .readdir("e")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert_eq!(names, vec!["x".to_string()]);
    }

    #[test]
    fn xattrs_copy_up_only_the_node_and_carry_lower_attributes() {
        let mut lower = TmpFs::new();
        lower.create("f", 0o644).unwrap();
        lower.setxattr("f", "user.a", b"1").unwrap();
        lower.mkdir("d", 0o755).unwrap();
        lower.create("d/x", 0o644).unwrap();
        let mut o = Overlay::new(Box::new(lower), Box::new(TmpFs::new()));
        // A lower attribute is visible; an absent one is ENODATA.
        assert_eq!(o.getxattr("f", "user.a").unwrap(), b"1");
        assert_eq!(
            o.getxattr("f", "user.b").unwrap_err().raw_os_error(),
            Some(61)
        );
        // Setting a second attribute copies the file up *with* the first.
        o.setxattr("f", "user.b", b"2").unwrap();
        let mut names = o.listxattr("f").unwrap();
        names.sort();
        assert_eq!(names, vec!["user.a".to_string(), "user.b".to_string()]);
        // Tagging a lower directory doesn't copy its children up.
        o.setxattr("d", "user.c", b"3").unwrap();
        assert!(o.upper.stat("d").is_some());
        assert!(o.upper.stat("d/x").is_none(), "child stays lower-only");
        assert_eq!(o.readdir("d").unwrap().len(), 1, "merged view keeps it");
        o.removexattr("d", "user.c").unwrap();
        assert_eq!(
            o.removexattr("d", "user.c").unwrap_err().raw_os_error(),
            Some(61)
        );
    }

    /// Names in the merged directory `dir`, sorted.
    fn names(o: &mut Overlay, dir: &str) -> Vec<String> {
        let mut v: Vec<_> = o
            .readdir(dir)
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        v.sort();
        v
    }

    #[test]
    fn whiteouts_are_marker_files_and_only_for_lower_entries() {
        let mut o = overlay();
        o.create("g", 0o644).unwrap();
        o.unlink("g").unwrap(); // upper-only: nothing to hide
        o.unlink("f").unwrap(); // lower: hidden by `.wh.f`
        assert!(o.upper.stat(".wh.g").is_none());
        assert!(o.upper.stat(".wh.f").is_some());
        assert_eq!(names(&mut o, ""), vec!["d"]);
    }

    #[test]
    fn marker_names_are_reserved_and_hidden() {
        let mut o = overlay();
        o.unlink("f").unwrap();
        for r in [
            o.create(".wh.f", 0o644),
            o.mkdir("d/.wh.x", 0o755),
            o.symlink("f", ".wh..wh..opq"),
            o.rename("d/x", ".wh.y"),
        ] {
            assert_eq!(r.unwrap_err().raw_os_error(), Some(1)); // EPERM
        }
        assert!(o.stat(".wh.f").is_none());
        assert!(o.read_at(".wh.f", 0, &mut [0u8; 1]).is_err());
        assert_eq!(names(&mut o, ""), vec!["d"]);
    }

    #[test]
    fn a_directory_made_again_is_opaque() {
        let mut o = overlay();
        o.unlink("d/x").unwrap();
        o.rmdir("d").unwrap();
        o.mkdir("d", 0o755).unwrap();
        assert!(o.upper.stat("d/.wh..wh..opq").is_some());
        assert!(o.readdir("d").unwrap().is_empty());
        assert!(o.stat("d/x").is_none());
        assert!(o.read_at("d/x", 0, &mut [0u8; 1]).is_err());
    }

    #[test]
    fn a_renamed_directory_leaves_nothing_behind() {
        let mut o = overlay();
        o.rename("d", "e").unwrap();
        // Making `d` again must not bring back the lower `d/x`, which moved.
        o.mkdir("d", 0o755).unwrap();
        assert!(o.readdir("d").unwrap().is_empty());
        assert_eq!(names(&mut o, "e"), vec!["x"]);
    }

    #[test]
    fn renaming_a_merged_directory_moves_both_layers() {
        let mut o = overlay();
        // `d` is in the upper (for `d/new`) and the lower (`d/x`).
        o.create("d/new", 0o644).unwrap();
        o.rename("d", "e").unwrap();
        assert_eq!(names(&mut o, "e"), vec!["new", "x"]);
        assert!(o.stat("d").is_none());
    }

    #[test]
    fn a_directory_renamed_onto_a_lower_one_hides_it() {
        let mut lower = TmpFs::new();
        lower.mkdir("a", 0o755).unwrap();
        lower.create("a/old", 0o644).unwrap();
        lower.mkdir("b", 0o755).unwrap();
        lower.create("b/mine", 0o644).unwrap();
        let mut o = Overlay::new(Box::new(lower), Box::new(TmpFs::new()));
        o.unlink("a/old").unwrap();
        o.rename("b", "a").unwrap();
        assert_eq!(names(&mut o, "a"), vec!["mine"]);
        // Even with the whiteout of `a/old` gone with the old upper `a`.
        assert!(o.upper.stat("a/.wh.old").is_none());
        assert!(o.upper.stat("a/.wh..wh..opq").is_some());
    }

    #[test]
    fn metadata_changes_to_a_directory_copy_up_only_the_directory() {
        let mut o = overlay();
        o.set_mode("d", 0o700).unwrap();
        assert_eq!(o.stat("d").unwrap().mode & 0o777, 0o700);
        assert!(o.upper.stat("d/x").is_none());
        assert_eq!(names(&mut o, "d"), vec!["x"]);
    }

    /// Changes a guest makes to `o`: writes `f`, deletes `d/x`, adds `n`, and
    /// remakes `d` empty with a new `d/y`.
    fn change(o: &mut Overlay) {
        o.write_at("f", 0, b"UP!").unwrap();
        o.create("n", 0o644).unwrap();
        o.write_at("n", 0, b"new").unwrap();
        o.unlink("d/x").unwrap();
        o.rmdir("d").unwrap();
        o.mkdir("d", 0o755).unwrap();
        o.create("d/y", 0o644).unwrap();
    }

    /// What [`change`] did, seen through a fresh overlay over the same upper.
    fn check_changed(o: &mut Overlay) {
        let mut buf = [0u8; 3];
        o.read_at("f", 0, &mut buf).unwrap();
        assert_eq!(&buf, b"UP!");
        o.read_at("n", 0, &mut buf).unwrap();
        assert_eq!(&buf, b"new");
        assert_eq!(names(o, ""), vec!["d", "f", "n"]);
        assert_eq!(names(o, "d"), vec!["y"]);
        assert!(o.stat("d/x").is_none());
    }

    #[test]
    fn an_upper_carries_the_changes_to_the_next_overlay() {
        let mut o = overlay();
        change(&mut o);
        let upper = o.into_upper();
        let mut o = Overlay::new(seeded_lower(), upper);
        check_changed(&mut o);
        // And deleting there works on what was carried over.
        o.unlink("f").unwrap();
        let mut o = Overlay::new(seeded_lower(), o.into_upper());
        assert_eq!(names(&mut o, ""), vec!["d", "n"]);
    }

    #[cfg(unix)]
    struct TempDir(std::path::PathBuf);

    #[cfg(unix)]
    impl TempDir {
        fn new(tag: &str) -> Self {
            use std::sync::atomic::{AtomicU32, Ordering};
            static N: AtomicU32 = AtomicU32::new(0);
            let p = std::env::temp_dir().join(format!(
                "nixvm-overlay-{tag}-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = std::fs::remove_dir_all(&p);
            Self(p)
        }
    }

    #[cfg(unix)]
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[cfg(unix)]
    #[test]
    fn a_host_directory_upper_persists_across_overlays() {
        let dir = TempDir::new("persist");
        let mut o = Overlay::persistent(seeded_lower(), &dir.0).unwrap();
        change(&mut o);
        // One overlay at a time on a directory.
        let busy = Overlay::persistent(seeded_lower(), &dir.0).unwrap_err();
        assert_eq!(busy.raw_os_error(), Some(16));
        drop(o);

        // On the host: the changed files at their paths, and the markers.
        let host = &dir.0;
        assert_eq!(std::fs::read(host.join("f")).unwrap(), b"UP!");
        assert!(host.join("d/y").is_file());
        assert!(host.join("d/.wh..wh..opq").is_file());
        assert!(!host.join("d/.wh.x").exists(), "the opaque d needs none");

        let mut o = Overlay::persistent(seeded_lower(), &dir.0).unwrap();
        check_changed(&mut o);
        o.unlink("f").unwrap();
        assert!(host.join(".wh.f").is_file());
        drop(o);
        let mut o = Overlay::persistent(seeded_lower(), &dir.0).unwrap();
        assert_eq!(names(&mut o, ""), vec!["d", "n"]);
    }
}
