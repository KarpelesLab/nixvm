//! Read-only paths inside a writable [`Passthrough`](super::Passthrough).
//!
//! An embedder that binds a host directory read-write often needs a few paths
//! in it to stay out of the guest's reach for writing — a project's
//! `.git/hooks` or a tool's own config file, either of which would let a
//! sandboxed command arrange for code to run on the host later. A read-only
//! mount on top doesn't do it: mounts are matched by guest path, so renaming a
//! parent (`mv .git g && echo evil > g/hooks/pre-commit && mv g .git`) walks
//! around it. [`Protected`] instead keeps those paths read-only *as the host
//! nodes they are*, however they're reached:
//!
//! * The check runs on the path the passthrough actually resolved, symlink by
//!   symlink, from its own root (see [`super::passthrough`]'s confined walk),
//!   so a symlink anywhere in the mount that leads into a protected path is
//!   caught, and so is a `..` or a case variant (names are compared
//!   case-insensitively, so a case-insensitive host can't alias them; on a
//!   case-sensitive host that protects the variants as well, which is
//!   harmless).
//! * A protected path, and every directory leading to it (its *ancestors*),
//!   can't be renamed or removed, and nothing can be renamed onto them — so a
//!   protected name keeps naming the same node for as long as the VM runs, and
//!   checking names is checking identity. Nothing protected can be hard-linked
//!   elsewhere either, so there is no second name to write it through.
//! * A protected path that doesn't exist yet can't be made, by any means
//!   (creation, `mkdir`, rename, link, symlink, `mknod`). Its missing
//!   ancestors can only be made as real directories (`mkdir`): a file, symlink
//!   or node at an ancestor's name could point the host somewhere else — a
//!   `.git` *file* names a git directory of its choosing, a `.git` symlink a
//!   directory the guest controls.
//!
//! # Errors
//!
//! * `EROFS` for anything that would change a protected path or what's under
//!   it — its contents, its metadata (`chmod`/`chown`/`utimes`), entries made
//!   or removed in it, opening it for writing, renaming it: it behaves as a
//!   read-only mount would.
//! * `EPERM` for what's refused around one rather than in it: removing or
//!   renaming an ancestor, renaming onto one, making an ancestor as anything
//!   but a directory, writing to an ancestor that exists as a file, and
//!   hard-linking a protected file (as Linux's `protected_hardlinks` does).
//!
//! An ancestor's own metadata (`chmod .git`) is left alone: that changes
//! nothing about what's under it.
//!
//! What this doesn't cover: hard links to protected files that already exist
//! on the host when the VM starts (the guest can't make new ones, but can
//! write through an old one), host-side mounts inside the directory that alias
//! a protected path, and Unicode-normalization aliases of non-ASCII names on a
//! host filesystem that ignores normalization. The protected paths themselves
//! are host configuration, trusted as is.

use std::io;

/// What an operation does to the path it's checked against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Op {
    /// Change the node's contents: write, truncate, open for writing, map
    /// shared-writable.
    Write,
    /// Change the node's metadata: `chmod`, `chown`, `utimes`.
    Attr,
    /// Make a directory at the path.
    MakeDir,
    /// Make anything else there: a file, a node, a symlink, a hard link, or
    /// the destination of a rename.
    Make,
    /// Remove the name: `unlink`, `rmdir`, or the source of a rename.
    Remove,
    /// Give the node a second name (the source of `link`).
    LinkSource,
}

/// Where a path stands relative to the protected ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Place {
    /// Unrelated to any protected path.
    Outside,
    /// A directory on the way to a protected path (the mount root is one).
    Ancestor,
    /// A protected path, or something beneath one.
    Inside,
}

/// A set of protected paths, relative to a mount's root.
#[derive(Debug, Clone, Default)]
pub struct Protected {
    /// Each path's components, case-folded.
    paths: Vec<Vec<String>>,
}

impl Protected {
    /// Protect `path`, relative to the mount root. A leading `/`, empty and
    /// `.` components are ignored and `..` is applied lexically (never above
    /// the root), so `a/./b/../c` protects `a/c`. An empty path (or `/`)
    /// protects the whole mount.
    pub fn add(&mut self, path: &str) {
        let mut comps: Vec<String> = Vec::new();
        for c in path.split('/') {
            match c {
                "" | "." => {}
                ".." => {
                    comps.pop();
                }
                c => comps.push(fold(c)),
            }
        }
        if !self.paths.contains(&comps) {
            self.paths.push(comps);
        }
    }

    /// Whether nothing is protected.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    /// Where the node at `comps` (the resolved path from the mount root, one
    /// name per component) stands.
    pub(super) fn place<S: AsRef<str>>(&self, comps: &[S]) -> Place {
        let mut place = Place::Outside;
        for p in &self.paths {
            let common = p
                .iter()
                .zip(comps)
                .take_while(|(a, b)| **a == fold(b.as_ref()))
                .count();
            if common == p.len() {
                return Place::Inside;
            }
            if common == comps.len() {
                place = Place::Ancestor;
            }
        }
        place
    }

    /// Whether `op` may be done to the node at `comps`; the error to fail it
    /// with if not (see the [module docs](self) for which is which).
    pub(super) fn check<S: AsRef<str>>(&self, op: Op, comps: &[S]) -> io::Result<()> {
        match (self.place(comps), op) {
            (Place::Outside, _) | (Place::Ancestor, Op::Attr | Op::MakeDir) => Ok(()),
            (Place::Inside, Op::LinkSource) | (Place::Ancestor, _) => Err(eperm()),
            (Place::Inside, _) => Err(erofs()),
        }
    }
}

/// A name as it's compared: case-folded, so that a case-insensitive host
/// can't reach a protected node under another spelling.
fn fold(name: &str) -> String {
    name.to_lowercase()
}

fn erofs() -> io::Error {
    io::Error::from_raw_os_error(30)
}

fn eperm() -> io::Error {
    io::Error::from_raw_os_error(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(paths: &[&str]) -> Protected {
        let mut p = Protected::default();
        for path in paths {
            p.add(path);
        }
        p
    }

    fn comps(path: &str) -> Vec<&str> {
        path.split('/').filter(|c| !c.is_empty()).collect()
    }

    fn errno(r: io::Result<()>) -> Option<i32> {
        r.err().and_then(|e| e.raw_os_error())
    }

    #[test]
    fn places_paths_against_the_protected_ones() {
        let p = set(&[".git/hooks", "atelier.toml"]);
        assert_eq!(p.place(&comps("")), Place::Ancestor);
        assert_eq!(p.place(&comps(".git")), Place::Ancestor);
        assert_eq!(p.place(&comps(".git/hooks")), Place::Inside);
        assert_eq!(p.place(&comps(".git/hooks/pre-commit")), Place::Inside);
        assert_eq!(p.place(&comps(".git/hooks/a/b")), Place::Inside);
        assert_eq!(p.place(&comps("atelier.toml")), Place::Inside);
        assert_eq!(p.place(&comps(".git/config")), Place::Outside);
        assert_eq!(p.place(&comps(".git/hooksx")), Place::Outside);
        assert_eq!(p.place(&comps("src/main.rs")), Place::Outside);
        assert_eq!(p.place(&comps("atelier.toml.bak")), Place::Outside);
    }

    #[test]
    fn names_compare_case_insensitively() {
        let p = set(&[".git/hooks"]);
        assert_eq!(p.place(&comps(".GIT/Hooks/pre-commit")), Place::Inside);
        assert_eq!(p.place(&comps(".Git")), Place::Ancestor);
        let p = set(&[".GIT/HOOKS"]);
        assert_eq!(p.place(&comps(".git/hooks")), Place::Inside);
    }

    #[test]
    fn protected_paths_are_normalized() {
        let p = set(&["/./.git//x/../hooks/", "../../top"]);
        assert_eq!(p.place(&comps(".git/hooks/pre-commit")), Place::Inside);
        assert_eq!(p.place(&comps(".git/x")), Place::Outside);
        assert_eq!(p.place(&comps("top")), Place::Inside);
        assert!(set(&[]).is_empty());
        assert!(!set(&["a"]).is_empty());
    }

    #[test]
    fn an_empty_path_protects_everything() {
        let p = set(&[""]);
        assert_eq!(p.place(&comps("")), Place::Inside);
        assert_eq!(p.place(&comps("any/thing")), Place::Inside);
    }

    #[test]
    fn errors_by_operation() {
        let p = set(&[".git/hooks"]);
        let inside = comps(".git/hooks/pre-commit");
        for op in [Op::Write, Op::Attr, Op::MakeDir, Op::Make, Op::Remove] {
            assert_eq!(errno(p.check(op, &inside)), Some(30), "{op:?}");
        }
        assert_eq!(errno(p.check(Op::LinkSource, &inside)), Some(1));
        // The protected directory itself is inside.
        assert_eq!(errno(p.check(Op::MakeDir, &comps(".git/hooks"))), Some(30));
        assert_eq!(errno(p.check(Op::Remove, &comps(".git/hooks"))), Some(30));

        let ancestor = comps(".git");
        assert_eq!(errno(p.check(Op::MakeDir, &ancestor)), None);
        assert_eq!(errno(p.check(Op::Attr, &ancestor)), None);
        for op in [Op::Write, Op::Make, Op::Remove, Op::LinkSource] {
            assert_eq!(errno(p.check(op, &ancestor)), Some(1), "{op:?}");
        }

        let outside = comps(".git/config");
        for op in [
            Op::Write,
            Op::Attr,
            Op::MakeDir,
            Op::Make,
            Op::Remove,
            Op::LinkSource,
        ] {
            assert_eq!(errno(p.check(op, &outside)), None, "{op:?}");
        }
    }
}

/// The protection as a [`Passthrough`](super::Passthrough) applies it, on a
/// real host directory.
#[cfg(test)]
mod passthrough_tests {
    use super::super::{MountFs, Passthrough, SetTime};
    use std::fs;
    use std::os::unix::fs::symlink;
    use std::path::PathBuf;

    /// A throwaway host directory laid out as a repository: `.git/hooks`
    /// with a hook in it, `.git/config`, `atelier.toml` and a `src` file.
    struct Repo(PathBuf);
    impl Repo {
        fn new(tag: &str) -> Self {
            let dir =
                std::env::temp_dir().join(format!("nixvm-protect-{}-{tag}", std::process::id()));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(dir.join(".git/hooks")).unwrap();
            fs::create_dir_all(dir.join("src")).unwrap();
            fs::write(dir.join(".git/hooks/pre-commit"), b"good\n").unwrap();
            fs::write(dir.join(".git/config"), b"[core]\n").unwrap();
            fs::write(dir.join("atelier.toml"), b"safe = true\n").unwrap();
            fs::write(dir.join("src/main.rs"), b"fn main() {}\n").unwrap();
            Self(dir)
        }
        fn fs(&self) -> Passthrough {
            Passthrough::new(self.0.clone()).protect([".git/hooks", "atelier.toml"])
        }
        fn read(&self, rel: &str) -> Option<Vec<u8>> {
            fs::read(self.0.join(rel)).ok()
        }
        fn exists(&self, rel: &str) -> bool {
            fs::symlink_metadata(self.0.join(rel)).is_ok()
        }
        /// Nothing protected changed on the host.
        fn intact(&self) {
            assert_eq!(
                self.read(".git/hooks/pre-commit").as_deref(),
                Some(&b"good\n"[..])
            );
            assert_eq!(
                fs::read_dir(self.0.join(".git/hooks")).unwrap().count(),
                1,
                "nothing new in .git/hooks"
            );
            assert_eq!(
                self.read("atelier.toml").as_deref(),
                Some(&b"safe = true\n"[..])
            );
        }
    }
    impl Drop for Repo {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn errno<T: std::fmt::Debug>(r: std::io::Result<T>) -> i32 {
        r.expect_err("should be refused").raw_os_error().unwrap()
    }

    const EPERM: i32 = 1;
    const EROFS: i32 = 30;

    #[test]
    fn the_rename_a_parent_attack_is_refused() {
        // mv .git g && echo evil > g/hooks/pre-commit && mv g .git
        let repo = Repo::new("mv-parent");
        let mut pt = repo.fs();
        assert_eq!(errno(pt.rename(".git", "g")), EPERM);
        assert!(repo.exists(".git/hooks") && !repo.exists("g"));
        assert_eq!(
            errno(pt.write_at(".git/hooks/pre-commit", 0, b"evil")),
            EROFS
        );
        // Nor can the protected directory itself be moved, or replaced.
        assert_eq!(errno(pt.rename(".git/hooks", ".git/h")), EROFS);
        pt.mkdir("h", 0o755).unwrap();
        assert_eq!(errno(pt.rename("h", ".git/hooks")), EROFS);
        assert_eq!(errno(pt.rename("h", ".git")), EPERM);
        repo.intact();
    }

    #[test]
    fn nothing_under_a_protected_path_changes() {
        let repo = Repo::new("writes");
        let mut pt = repo.fs();
        let hook = ".git/hooks/pre-commit";
        assert_eq!(errno(pt.write_at(hook, 0, b"evil")), EROFS);
        assert_eq!(errno(pt.truncate(hook, 0)), EROFS);
        assert_eq!(errno(pt.create(hook, 0o755)), EROFS);
        assert_eq!(errno(pt.create(".git/hooks/post-checkout", 0o755)), EROFS);
        assert_eq!(errno(pt.mkdir(".git/hooks/d", 0o755)), EROFS);
        assert_eq!(errno(pt.mknod(".git/hooks/f", 0o010_644)), EROFS);
        assert_eq!(errno(pt.symlink("/bin/sh", ".git/hooks/s")), EROFS);
        assert_eq!(errno(pt.unlink(hook)), EROFS);
        assert_eq!(errno(pt.rmdir(".git/hooks")), EROFS);
        assert_eq!(errno(pt.set_mode(hook, 0o777)), EROFS);
        assert_eq!(errno(pt.set_owner(hook, Some(0), None)), EROFS);
        assert_eq!(errno(pt.set_times(hook, SetTime::Now, SetTime::Now)), EROFS);
        assert_eq!(errno(pt.rename(hook, "stolen")), EROFS);
        assert_eq!(errno(pt.check_write(hook)), EROFS);
        assert_eq!(errno(pt.check_write(".git/hooks/new")), EROFS);
        // A protected single file is the same.
        assert_eq!(errno(pt.write_at("atelier.toml", 0, b"x")), EROFS);
        assert_eq!(errno(pt.unlink("atelier.toml")), EROFS);
        assert_eq!(errno(pt.rename("src/main.rs", "atelier.toml")), EROFS);
        // Reading is fine.
        let mut buf = [0u8; 5];
        assert_eq!(pt.read_at(hook, 0, &mut buf).unwrap(), 5);
        assert_eq!(pt.readdir(".git/hooks").unwrap().len(), 1);
        assert!(pt.stat(hook).is_some());
        repo.intact();
    }

    #[test]
    fn everything_else_stays_writable() {
        let repo = Repo::new("rest");
        let mut pt = repo.fs();
        pt.write_at(".git/config", 0, b"[user]\n").unwrap();
        pt.create(".git/HEAD", 0o644).unwrap();
        pt.rename(".git/HEAD", ".git/ORIG_HEAD").unwrap();
        pt.unlink(".git/ORIG_HEAD").unwrap();
        pt.mkdir(".git/objects", 0o755).unwrap();
        pt.set_mode(".git", 0o755).unwrap();
        pt.create("atelier.toml.bak", 0o644).unwrap();
        pt.rename("src", "lib").unwrap();
        pt.write_at("lib/main.rs", 0, b"//").unwrap();
        pt.check_write("lib/main.rs").unwrap();
        pt.check_write("lib/new.rs").unwrap();
        assert_eq!(repo.read(".git/config").as_deref(), Some(&b"[user]\n"[..]));
        repo.intact();
    }

    #[test]
    fn a_missing_protected_path_cannot_be_made() {
        // A repository without .git/hooks must not gain one, however it goes.
        let repo = Repo::new("missing");
        fs::remove_dir_all(repo.0.join(".git")).unwrap();
        let mut pt = repo.fs();
        // `.git` itself may be made, as a directory, but hooks may not.
        pt.mkdir(".git", 0o755).unwrap();
        assert_eq!(errno(pt.mkdir(".git/hooks", 0o755)), EROFS);
        assert_eq!(errno(pt.create(".git/hooks", 0o644)), EROFS);
        assert_eq!(errno(pt.symlink("../src", ".git/hooks")), EROFS);
        pt.mkdir("h", 0o755).unwrap();
        assert_eq!(errno(pt.rename("h", ".git/hooks")), EROFS);
        assert_eq!(errno(pt.link("src/main.rs", ".git/hooks")), EROFS);
        assert!(!repo.exists(".git/hooks"));
        // Nor may `.git` be anything but a directory: a .git file or symlink
        // would point the host at hooks somewhere else.
        pt.rmdir("h").unwrap();
        assert_eq!(errno(pt.rmdir(".git")), EPERM);
        assert_eq!(errno(pt.rename(".git", "g")), EPERM);
        fs::remove_dir(repo.0.join(".git")).unwrap();
        assert_eq!(errno(pt.create(".git", 0o644)), EPERM);
        assert_eq!(errno(pt.symlink("elsewhere", ".git")), EPERM);
        assert_eq!(errno(pt.mknod(".git", 0o010_644)), EPERM);
        assert_eq!(errno(pt.link("src/main.rs", ".git")), EPERM);
        pt.mkdir("g", 0o755).unwrap();
        pt.mkdir("g/hooks", 0o755).unwrap();
        assert_eq!(errno(pt.rename("g", ".git")), EPERM);
        assert!(!repo.exists(".git"));
        // A missing protected file likewise.
        fs::remove_file(repo.0.join("atelier.toml")).unwrap();
        assert_eq!(errno(pt.create("atelier.toml", 0o644)), EROFS);
        assert_eq!(errno(pt.check_write("atelier.toml")), EROFS);
        assert!(!repo.exists("atelier.toml"));
    }

    #[test]
    fn symlinks_into_a_protected_path_are_followed_to_it() {
        let repo = Repo::new("symlinks");
        // Made on the host, as the guest could make them: links aren't
        // refused, writing through them is.
        symlink(".git/hooks", repo.0.join("h")).unwrap();
        symlink("/.git/hooks/pre-commit", repo.0.join("abs")).unwrap();
        symlink("src/../.git", repo.0.join("g")).unwrap();
        symlink("atelier.toml", repo.0.join("cfg")).unwrap();
        let mut pt = repo.fs();
        assert_eq!(errno(pt.write_at("h/pre-commit", 0, b"evil")), EROFS);
        assert_eq!(errno(pt.create("h/post-merge", 0o755)), EROFS);
        assert_eq!(errno(pt.write_at("abs", 0, b"evil")), EROFS);
        assert_eq!(errno(pt.truncate("abs", 0)), EROFS);
        assert_eq!(errno(pt.check_write("abs")), EROFS);
        assert_eq!(errno(pt.write_at("g/hooks/pre-commit", 0, b"evil")), EROFS);
        assert_eq!(errno(pt.rename("g/hooks", "x")), EROFS);
        assert_eq!(errno(pt.rename("src", "g/hooks/src")), EROFS);
        assert_eq!(errno(pt.write_at("cfg", 0, b"evil")), EROFS);
        // A symlink pointing at a missing protected file doesn't make it.
        fs::remove_file(repo.0.join("atelier.toml")).unwrap();
        assert_eq!(errno(pt.create("cfg", 0o644)), EROFS);
        assert!(!repo.exists("atelier.toml"));
        fs::write(repo.0.join("atelier.toml"), b"safe = true\n").unwrap();
        // The links themselves are the guest's, outside: removable.
        pt.unlink("h").unwrap();
        pt.rename("g", "g2").unwrap();
        repo.intact();
    }

    #[test]
    fn a_symlink_swapped_in_for_a_parent_is_caught() {
        // A descriptor opened as `d/f` keeps writing to `d/f`; turning `d`
        // into a link to .git/hooks must not turn that into a hook.
        let repo = Repo::new("swap");
        let mut pt = repo.fs();
        pt.mkdir("d", 0o755).unwrap();
        pt.create("d/pre-commit", 0o755).unwrap();
        pt.rename("d", "d2").unwrap();
        pt.symlink(".git/hooks", "d").unwrap();
        assert_eq!(errno(pt.write_at("d/pre-commit", 0, b"evil")), EROFS);
        assert_eq!(errno(pt.check_write("d/pre-commit")), EROFS);
        repo.intact();
    }

    #[test]
    fn protected_files_cannot_gain_a_second_name() {
        let repo = Repo::new("links");
        let mut pt = repo.fs();
        assert_eq!(errno(pt.link(".git/hooks/pre-commit", "x")), EPERM);
        assert_eq!(errno(pt.link("atelier.toml", "y")), EPERM);
        assert!(!repo.exists("x") && !repo.exists("y"));
        // Linking unprotected files still works.
        pt.link("src/main.rs", "src/alias.rs").unwrap();
        repo.intact();
    }

    #[test]
    fn other_spellings_of_a_protected_name_are_protected() {
        let repo = Repo::new("case");
        let mut pt = repo.fs();
        // On a case-insensitive host these are the protected nodes; on a
        // case-sensitive one they're protected anyway.
        assert_eq!(errno(pt.rename(".GIT", "g")), EPERM);
        assert_eq!(errno(pt.create("ATELIER.TOML", 0o644)), EROFS);
        // `.` and empty components don't change the path.
        assert_eq!(
            errno(pt.write_at("./.git//hooks/./pre-commit", 0, b"x")),
            EROFS
        );
        // A literal `..` is refused outright by the passthrough (EACCES).
        assert_eq!(
            errno(pt.write_at("src/../.git/hooks/pre-commit", 0, b"x")),
            13
        );
        repo.intact();
    }

    #[test]
    fn protecting_nothing_changes_nothing() {
        let repo = Repo::new("none");
        let mut pt = Passthrough::new(repo.0.clone()).protect(Vec::<String>::new());
        pt.rename(".git", "g").unwrap();
        pt.write_at("g/hooks/pre-commit", 0, b"evil\n").unwrap();
        pt.rename("g", ".git").unwrap();
        assert_eq!(repo.read(".git/hooks/pre-commit").unwrap(), b"evil\n");
    }

    #[test]
    fn protecting_the_root_makes_the_mount_read_only() {
        let repo = Repo::new("root");
        let mut pt = Passthrough::new(repo.0.clone()).protect(["/"]);
        assert_eq!(errno(pt.write_at("src/main.rs", 0, b"x")), EROFS);
        assert_eq!(errno(pt.create("new", 0o644)), EROFS);
        assert_eq!(errno(pt.mkdir("new", 0o755)), EROFS);
        assert!(pt.stat("src/main.rs").is_some());
    }
}
