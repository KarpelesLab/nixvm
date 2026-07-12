//! Copy-on-write overlay filesystem.
//!
//! Presents a merged view of a read-only `lower` (e.g. the squashfs root) and a
//! writable `upper` (a [`TmpFs`](super::TmpFs)). Reads prefer the upper and fall
//! through to the lower; the first write to a lower-only path *copies it up*
//! into the upper; deletions record a *whiteout* that hides the lower entry.
//! The lower is never modified — so `/` is mutable but ephemeral.

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

/// Tag an upper-layer node's inode as distinct from the lower layer's.
fn tag_upper(mut a: Attrs) -> Attrs {
    a.inode |= UPPER_INODE_TAG;
    a
}

#[derive(Debug)]
pub struct Overlay {
    upper: Box<dyn MountFs>,
    lower: Box<dyn MountFs>,
    /// Paths deleted from the merged view (hide the lower entry).
    whiteouts: BTreeSet<String>,
}

impl Overlay {
    /// Build an overlay with `upper` (writable) over `lower` (read-only).
    #[must_use]
    pub fn new(lower: Box<dyn MountFs>, upper: Box<dyn MountFs>) -> Self {
        Self {
            upper,
            lower,
            whiteouts: BTreeSet::new(),
        }
    }

    /// Is `rel` (or an ancestor) hidden by a whiteout?
    fn is_whited(&self, rel: &str) -> bool {
        if self.whiteouts.contains(rel) {
            return true;
        }
        let mut p = rel;
        while let Some(i) = p.rfind('/') {
            p = &p[..i];
            if self.whiteouts.contains(p) {
                return true;
            }
        }
        false
    }

    /// Ensure directory `dir` exists in the upper layer, creating ancestors.
    fn ensure_dir_in_upper(&mut self, dir: &str) {
        if dir.is_empty() || self.upper.stat(dir).is_some() {
            return;
        }
        self.ensure_dir_in_upper(parent_of(dir));
        let _ = self.upper.mkdir(dir, 0o755);
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

    /// Materialize `rel` into the upper layer if it lives only in the lower.
    /// For a directory, this recurses into every non-whited-out lower child
    /// so that renaming (or otherwise moving) a lower-only directory brings
    /// its whole subtree along rather than an empty shell.
    fn copy_up(&mut self, rel: &str) -> io::Result<()> {
        if self.upper.stat(rel).is_some() {
            return Ok(());
        }
        let Some(attrs) = self.lower.stat(rel) else {
            return Ok(()); // nothing to copy; caller creates fresh
        };
        self.ensure_dir_in_upper(parent_of(rel));
        match attrs.kind {
            NodeKind::Dir => {
                self.ensure_dir_in_upper(rel);
                if let Ok(entries) = self.lower.readdir(rel) {
                    for e in entries {
                        let child = join(rel, &e.name);
                        if !self.is_whited(&child) {
                            self.copy_up(&child)?;
                        }
                    }
                }
            }
            NodeKind::Symlink => {
                if let Ok(target) = self.lower.readlink(rel) {
                    self.upper.symlink(&target, rel)?;
                }
            }
            _ => {
                let data = self.read_all_lower(rel);
                self.upper.create(rel, attrs.mode & 0o777)?;
                self.upper.write_at(rel, 0, &data)?;
            }
        }
        Ok(())
    }
}

fn parent_of(rel: &str) -> &str {
    match rel.rfind('/') {
        Some(i) => &rel[..i],
        None => "",
    }
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
fn enotempty() -> io::Error {
    io::Error::from_raw_os_error(39)
}

impl MountFs for Overlay {
    fn read_only(&self) -> bool {
        false
    }

    fn stat(&mut self, rel: &str) -> Option<Attrs> {
        if self.is_whited(rel) {
            return None;
        }
        self.upper
            .stat(rel)
            .map(tag_upper)
            .or_else(|| self.lower.stat(rel))
    }

    fn read_at(&mut self, rel: &str, off: u64, buf: &mut [u8]) -> io::Result<usize> {
        if self.is_whited(rel) {
            return Err(enoent());
        }
        if self.upper.stat(rel).is_some() {
            self.upper.read_at(rel, off, buf)
        } else {
            self.lower.read_at(rel, off, buf)
        }
    }

    fn readdir(&mut self, rel: &str) -> io::Result<Vec<DirEntry>> {
        if self.is_whited(rel) || self.stat(rel).is_none() {
            return Err(enoent());
        }
        let mut seen: BTreeMap<String, DirEntry> = BTreeMap::new();
        if let Ok(entries) = self.lower.readdir(rel) {
            for e in entries {
                if !self.is_whited(&join(rel, &e.name)) {
                    seen.insert(e.name.clone(), e);
                }
            }
        }
        if let Ok(entries) = self.upper.readdir(rel) {
            for mut e in entries {
                e.inode |= UPPER_INODE_TAG; // match `stat`'s upper-layer tag
                seen.insert(e.name.clone(), e); // upper shadows lower
            }
        }
        Ok(seen.into_values().collect())
    }

    fn write_at(&mut self, rel: &str, off: u64, buf: &[u8]) -> io::Result<usize> {
        self.copy_up(rel)?;
        self.upper.write_at(rel, off, buf)
    }

    fn create(&mut self, rel: &str, mode: u32) -> io::Result<()> {
        if self.stat(rel).is_some() {
            return Err(eexist());
        }
        self.whiteouts.remove(rel);
        self.ensure_dir_in_upper(parent_of(rel));
        self.upper.create(rel, mode)
    }

    fn mkdir(&mut self, rel: &str, mode: u32) -> io::Result<()> {
        if self.stat(rel).is_some() {
            return Err(eexist());
        }
        self.whiteouts.remove(rel);
        self.ensure_dir_in_upper(parent_of(rel));
        self.upper.mkdir(rel, mode)
    }

    fn unlink(&mut self, rel: &str) -> io::Result<()> {
        match self.stat(rel) {
            None => return Err(enoent()),
            Some(a) if a.kind == NodeKind::Dir => return Err(eisdir()),
            Some(_) => {}
        }
        let _ = self.upper.unlink(rel);
        self.whiteouts.insert(rel.to_string());
        Ok(())
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
        let _ = self.upper.rmdir(rel);
        self.whiteouts.insert(rel.to_string());
        Ok(())
    }

    fn truncate(&mut self, rel: &str, len: u64) -> io::Result<()> {
        self.copy_up(rel)?;
        self.upper.truncate(rel, len)
    }

    fn symlink(&mut self, target: &str, linkpath: &str) -> io::Result<()> {
        if self.stat(linkpath).is_some() {
            return Err(eexist());
        }
        self.whiteouts.remove(linkpath);
        self.ensure_dir_in_upper(parent_of(linkpath));
        self.upper.symlink(target, linkpath)
    }

    fn readlink(&mut self, rel: &str) -> io::Result<String> {
        if self.is_whited(rel) {
            return Err(enoent());
        }
        if self.upper.stat(rel).is_some() {
            self.upper.readlink(rel)
        } else {
            self.lower.readlink(rel)
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
        let Some(from_attrs) = self.stat(from) else {
            return Err(enoent());
        };
        let from_is_dir = from_attrs.kind == NodeKind::Dir;
        // Refuse to move a directory into itself or one of its own
        // descendants.
        if from_is_dir && (to == from || to.starts_with(&format!("{from}/"))) {
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
            if to_is_dir {
                let _ = self.upper.rmdir(to);
            } else {
                let _ = self.upper.unlink(to);
            }
        }

        // Copy the whole `from` subtree into the upper layer first (a no-op
        // for anything already upper-resident), then rename within the
        // upper — the lower layer is never mutated.
        self.copy_up(from)?;
        self.ensure_dir_in_upper(parent_of(to));
        self.whiteouts.remove(to);
        self.upper.rename(from, to)?;
        self.whiteouts.insert(from.to_string()); // hide any lower `from`
        Ok(())
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
        assert_ne!(lower_ino, upper_ino, "upper and lower inodes must be distinct");
        assert_eq!(upper_ino & UPPER_INODE_TAG, UPPER_INODE_TAG, "upper is tagged");
        assert_eq!(lower_ino & UPPER_INODE_TAG, 0, "lower is untagged");
        // readdir agrees with stat on the upper tag.
        let g = o.readdir("").unwrap().into_iter().find(|e| e.name == "g").unwrap();
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
}
