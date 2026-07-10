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

    /// Find the backend owning `abs_path` and the path relative to its mount
    /// point. Returns the longest matching mount.
    fn resolve(&mut self, abs_path: &str) -> Option<(&mut dyn MountFs, String)> {
        let mut best: Option<usize> = None;
        let mut best_len = 0usize;
        for (i, m) in self.mounts.iter().enumerate() {
            if path_has_prefix(abs_path, &m.point) && m.point.len() >= best_len {
                best = Some(i);
                best_len = m.point.len();
            }
        }
        let i = best?;
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
