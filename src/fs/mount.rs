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
    ///
    /// `point` is normalized (see `normalize`) before being stored, so
    /// callers may pass e.g. `"/dev/"` or `"/dev/../dev"` and still get the
    /// expected `"/dev"` mount point.
    pub fn mount(&mut self, point: impl Into<String>, fs: Box<dyn MountFs>) {
        self.mounts.push(Mount {
            point: normalize(&point.into()),
            fs,
        });
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
    fn resolve(&mut self, abs_path: &str) -> Option<(&mut dyn MountFs, String)> {
        let abs_path = normalize(abs_path);
        let i = self.best_mount(&abs_path)?;
        let rel = relative_to(&abs_path, &self.mounts[i].point);
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
        let from = normalize(from);
        let to = normalize(to);
        let from_idx = self.best_mount(&from).ok_or_else(enoent)?;
        let to_idx = self.best_mount(&to).ok_or_else(enoent)?;
        if from_idx != to_idx {
            return Err(io::Error::from_raw_os_error(18)); // EXDEV
        }
        let point = self.mounts[from_idx].point.clone();
        let from_rel = relative_to(&from, &point);
        let to_rel = relative_to(&to, &point);
        self.mounts[from_idx].fs.rename(&from_rel, &to_rel)
    }
}

fn enoent() -> io::Error {
    io::Error::from_raw_os_error(2)
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
                mtime: 0,
                inode: 0,
                nlink: 1,
                rdev: 0,
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
