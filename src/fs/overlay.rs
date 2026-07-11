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
    fn copy_up(&mut self, rel: &str) -> io::Result<()> {
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

impl MountFs for Overlay {
    fn read_only(&self) -> bool {
        false
    }

    fn stat(&mut self, rel: &str) -> Option<Attrs> {
        if self.is_whited(rel) {
            return None;
        }
        self.upper.stat(rel).or_else(|| self.lower.stat(rel))
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
            for e in entries {
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
        self.whiteouts.remove(rel);
        self.ensure_dir_in_upper(parent_of(rel));
        self.upper.create(rel, mode)
    }

    fn mkdir(&mut self, rel: &str, mode: u32) -> io::Result<()> {
        self.whiteouts.remove(rel);
        self.ensure_dir_in_upper(parent_of(rel));
        self.upper.mkdir(rel, mode)
    }

    fn unlink(&mut self, rel: &str) -> io::Result<()> {
        if self.stat(rel).is_none() {
            return Err(enoent());
        }
        let _ = self.upper.unlink(rel);
        self.whiteouts.insert(rel.to_string());
        Ok(())
    }

    fn rmdir(&mut self, rel: &str) -> io::Result<()> {
        match self.stat(rel) {
            Some(a) if a.kind == NodeKind::Dir => {}
            Some(_) => return Err(io::Error::from_raw_os_error(20)), // ENOTDIR
            None => return Err(enoent()),
        }
        if !self.readdir(rel)?.is_empty() {
            return Err(io::Error::from_raw_os_error(39)); // ENOTEMPTY
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
        if self.stat(from).is_none() {
            return Err(enoent());
        }
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
}
