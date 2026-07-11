//! The mount table: composes [`MountFs`] backends by longest-prefix match.

use std::io;

use super::{Attrs, DirEntry, MountFs};

struct Mount {
    /// Absolute mount point, e.g. "/", "/work", "/proc".
    point: String,
    fs: Box<dyn MountFs>,
}

impl std::fmt::Debug for Mount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mount")
            .field("point", &self.point)
            .finish_non_exhaustive()
    }
}

/// Resolves guest paths across a set of mounted backends.
#[derive(Debug, Default)]
pub struct MountTable {
    mounts: Vec<Mount>,
}

impl MountTable {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Mount `fs` at absolute `point`. Deeper mount points win via
    /// longest-prefix resolution.
    pub fn mount(&mut self, point: impl Into<String>, fs: Box<dyn MountFs>) {
        self.mounts.push(Mount {
            point: point.into(),
            fs,
        });
    }

    /// Index of the longest-prefix mount owning `abs_path`.
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
    fn resolve(&mut self, abs_path: &str) -> Option<(&mut dyn MountFs, String)> {
        let i = self.best_mount(abs_path)?;
        let rel = relative_to(abs_path, &self.mounts[i].point);
        Some((self.mounts[i].fs.as_mut(), rel))
    }

    pub fn stat(&mut self, abs_path: &str) -> Option<Attrs> {
        let (fs, rel) = self.resolve(abs_path)?;
        fs.stat(&rel)
    }

    pub fn read_at(&mut self, abs_path: &str, off: u64, buf: &mut [u8]) -> io::Result<usize> {
        let (fs, rel) = self.resolve(abs_path).ok_or_else(enoent)?;
        fs.read_at(&rel, off, buf)
    }

    pub fn readdir(&mut self, abs_path: &str) -> io::Result<Vec<DirEntry>> {
        let (fs, rel) = self.resolve(abs_path).ok_or_else(enoent)?;
        fs.readdir(&rel)
    }

    pub fn write_at(&mut self, abs_path: &str, off: u64, buf: &[u8]) -> io::Result<usize> {
        let (fs, rel) = self.resolve(abs_path).ok_or_else(enoent)?;
        fs.write_at(&rel, off, buf)
    }

    pub fn create(&mut self, abs_path: &str, mode: u32) -> io::Result<()> {
        let (fs, rel) = self.resolve(abs_path).ok_or_else(enoent)?;
        fs.create(&rel, mode)
    }

    pub fn mkdir(&mut self, abs_path: &str, mode: u32) -> io::Result<()> {
        let (fs, rel) = self.resolve(abs_path).ok_or_else(enoent)?;
        fs.mkdir(&rel, mode)
    }

    pub fn unlink(&mut self, abs_path: &str) -> io::Result<()> {
        let (fs, rel) = self.resolve(abs_path).ok_or_else(enoent)?;
        fs.unlink(&rel)
    }

    pub fn rmdir(&mut self, abs_path: &str) -> io::Result<()> {
        let (fs, rel) = self.resolve(abs_path).ok_or_else(enoent)?;
        fs.rmdir(&rel)
    }

    pub fn truncate(&mut self, abs_path: &str, len: u64) -> io::Result<()> {
        let (fs, rel) = self.resolve(abs_path).ok_or_else(enoent)?;
        fs.truncate(&rel, len)
    }

    pub fn readlink(&mut self, abs_path: &str) -> io::Result<String> {
        let (fs, rel) = self.resolve(abs_path).ok_or_else(enoent)?;
        fs.readlink(&rel)
    }

    /// Create a symlink at `abs_link` pointing at `target` (stored verbatim).
    pub fn symlink(&mut self, target: &str, abs_link: &str) -> io::Result<()> {
        let (fs, rel) = self.resolve(abs_link).ok_or_else(enoent)?;
        fs.symlink(target, &rel)
    }

    /// Rename within a single backend. Cross-mount renames return `EXDEV`.
    pub fn rename(&mut self, from: &str, to: &str) -> io::Result<()> {
        let from_idx = self.best_mount(from).ok_or_else(enoent)?;
        let to_idx = self.best_mount(to).ok_or_else(enoent)?;
        if from_idx != to_idx {
            return Err(io::Error::from_raw_os_error(18)); // EXDEV
        }
        let point = self.mounts[from_idx].point.clone();
        let from_rel = relative_to(from, &point);
        let to_rel = relative_to(to, &point);
        self.mounts[from_idx].fs.rename(&from_rel, &to_rel)
    }
}

fn enoent() -> io::Error {
    io::Error::from_raw_os_error(2)
}

/// Does `path` fall under mount point `mp`? "/" matches everything.
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
/// slash (so backends see e.g. "bin/sh" for "/bin/sh").
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

    #[test]
    fn longest_prefix_and_relative() {
        assert!(path_has_prefix("/work/src/a.rs", "/work"));
        assert!(path_has_prefix("/anything", "/"));
        assert!(!path_has_prefix("/worker", "/work"));
        assert_eq!(relative_to("/work/src/a.rs", "/work"), "src/a.rs");
        assert_eq!(relative_to("/bin/sh", "/"), "bin/sh");
    }
}
