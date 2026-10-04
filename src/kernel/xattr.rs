//! Extended attributes: `{,l,f}{get,set,list,remove}xattr` and the 6.13
//! `*xattrat` family.
//!
//! The storage lives in the filesystem backends ([`crate::fs::MountFs`]'s xattr
//! methods — per node in tmpfs, copied up through the overlay); this module is
//! the syscall layer on top, and owns everything Linux decides *before* a
//! filesystem is consulted (`fs/xattr.c`): name and size limits, the namespace
//! prefix check, the `XATTR_CREATE`/`XATTR_REPLACE` contract, the "size 0 is a
//! length query / too small is `ERANGE`" buffer protocol, and the per-namespace
//! permission rules (`trusted.*` needs `CAP_SYS_ADMIN`; `user.*` lives only on
//! regular files and directories). apk, GNU/busybox tar `--xattrs`, `cp -a`,
//! `rsync -X`, pip and Python's `os.*xattr` all lean on exactly these answers.

use super::{AT_FDCWD, Fd, Kernel, ServiceCtx, err, io_errno, read_path};
use crate::abi::errno::Errno;
use crate::fs::{MountTable, NodeKind};
use crate::vcpu::GuestMemory;

/// Longest attribute name, excluding the NUL (`XATTR_NAME_MAX`).
const XATTR_NAME_MAX: usize = 255;
/// Largest attribute value / name list (`XATTR_SIZE_MAX`, `XATTR_LIST_MAX`).
const XATTR_SIZE_MAX: usize = 65536;
const XATTR_CREATE: u64 = 1;
const XATTR_REPLACE: u64 = 2;
const AT_SYMLINK_NOFOLLOW: u64 = 0x100;
const AT_EMPTY_PATH: u64 = 0x1000;

/// Which node an xattr syscall names.
pub(super) enum XattrTarget {
    /// A path (`dirfd`-relative), following a final symlink unless `nofollow`.
    Path {
        dirfd: i64,
        path: u64,
        nofollow: bool,
        /// `AT_EMPTY_PATH`: an empty `path` means `dirfd` itself.
        empty_ok: bool,
    },
    /// An open descriptor (`f*xattr`).
    Fd(i32),
}

impl XattrTarget {
    /// The plain path spellings: `getxattr`/`lgetxattr`/… on `path`.
    pub(super) fn path(path: u64, nofollow: bool) -> Self {
        Self::Path {
            dirfd: AT_FDCWD,
            path,
            nofollow,
            empty_ok: false,
        }
    }

    /// The `*xattrat(dirfd, path, at_flags, …)` spellings. Unknown `at_flags`
    /// bits are `EINVAL`.
    pub(super) fn at(dirfd: u64, path: u64, at_flags: u64) -> Result<Self, i64> {
        if at_flags & !(AT_SYMLINK_NOFOLLOW | AT_EMPTY_PATH) != 0 {
            return Err(err(Errno::EINVAL));
        }
        Ok(Self::Path {
            dirfd: i64::from(dirfd as i32),
            path,
            nofollow: at_flags & AT_SYMLINK_NOFOLLOW != 0,
            empty_ok: at_flags & AT_EMPTY_PATH != 0,
        })
    }
}

/// Where a resolved target lives: a node in the mount table, or a descriptor
/// with no filesystem behind it (pipe, socket, eventfd, tty, …) — which, like
/// Linux's pipefs/sockfs/anon-inode, supports no attributes.
enum Node {
    Path(String, NodeKind),
    Anon,
}

/// Validate an attribute name the way `fs/xattr.c` does before any
/// filesystem sees it: 1..=255 bytes (`ERANGE` otherwise) under a namespace
/// some handler claims (`EOPNOTSUPP` otherwise), and not the bare prefix
/// (`EINVAL`). `system.*` is only the two POSIX ACL names.
fn check_name(mem: &GuestMemory, ptr: u64) -> Result<String, i64> {
    let Ok(raw) = mem.read_cstr(ptr, XATTR_NAME_MAX + 1) else {
        return Err(err(Errno::EFAULT));
    };
    if raw.is_empty() || raw.len() > XATTR_NAME_MAX {
        return Err(err(Errno::ERANGE));
    }
    let name = String::from_utf8_lossy(&raw).into_owned();
    for prefix in ["user.", "trusted.", "security."] {
        if let Some(rest) = name.strip_prefix(prefix) {
            return if rest.is_empty() {
                Err(err(Errno::EINVAL))
            } else {
                Ok(name)
            };
        }
    }
    if matches!(
        name.as_str(),
        "system.posix_acl_access" | "system.posix_acl_default"
    ) {
        return Ok(name);
    }
    Err(err(Errno::EOPNOTSUPP))
}

/// Copy `bytes` out under the getxattr/listxattr buffer protocol: a zero
/// `size` asks only for the length; a buffer too small is `ERANGE`.
fn copy_out(mem: &mut GuestMemory, buf: u64, size: u64, bytes: &[u8]) -> i64 {
    if size == 0 {
        return bytes.len() as i64;
    }
    if (size as usize) < bytes.len() {
        return err(Errno::ERANGE);
    }
    if mem.write(buf, bytes).is_err() {
        return err(Errno::EFAULT);
    }
    bytes.len() as i64
}

impl Kernel {
    /// Resolve an [`XattrTarget`] to the node it names: `ENOENT` for a missing
    /// path (or an empty one without `AT_EMPTY_PATH`), `EBADF` for a closed fd.
    fn xattr_node(
        &self,
        vfs: &mut MountTable,
        cx: &ServiceCtx,
        t: &XattrTarget,
        mem: &GuestMemory,
    ) -> Result<Node, i64> {
        let from_fd = |fd: i32| -> Result<Node, i64> {
            match cx.cur.fds.get(fd) {
                Some(Fd::File { path, .. } | Fd::Dir { path, .. }) => Ok(Node::Path(
                    path.clone(),
                    NodeKind::File, // refined by the stat below
                )),
                Some(_) => Ok(Node::Anon),
                None => Err(err(Errno::EBADF)),
            }
        };
        let node = match *t {
            XattrTarget::Fd(fd) => from_fd(fd)?,
            XattrTarget::Path {
                dirfd,
                path,
                nofollow,
                empty_ok,
            } => {
                let Some(rel) = read_path(mem, path) else {
                    return Err(err(Errno::EFAULT));
                };
                if rel.is_empty() {
                    if !empty_ok {
                        return Err(err(Errno::ENOENT));
                    }
                    if dirfd == AT_FDCWD {
                        Node::Path(cx.cur.cwd.clone(), NodeKind::Dir)
                    } else {
                        from_fd(dirfd as i32)?
                    }
                } else {
                    let abs = self.resolve_path(cx, dirfd, &rel);
                    let abs = if nofollow {
                        abs
                    } else {
                        self.follow_or_eloop(vfs, &abs)?
                    };
                    Node::Path(abs, NodeKind::File)
                }
            }
        };
        Ok(match node {
            Node::Path(p, _) => match vfs.stat(&p) {
                Some(a) => Node::Path(p, a.kind),
                None => return Err(err(Errno::ENOENT)),
            },
            Node::Anon => Node::Anon,
        })
    }

    /// Whether the caller holds `CAP_SYS_ADMIN` (the `trusted.*` gate). The VM
    /// grants root every capability, so this is "is effectively root".
    fn xattr_admin(cx: &ServiceCtx) -> bool {
        cx.cur.creds.euid == 0
    }

    /// `getxattr`/`lgetxattr`/`fgetxattr`/`getxattrat`: copy attribute `name`'s
    /// value into `(buf, size)`, or report its length when `size == 0`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_getxattr(
        &self,
        vfs: &mut MountTable,
        cx: &ServiceCtx,
        t: &XattrTarget,
        name: u64,
        buf: u64,
        size: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let node = match self.xattr_node(vfs, cx, t, mem) {
            Ok(n) => n,
            Err(e) => return e,
        };
        let name = match check_name(mem, name) {
            Ok(n) => n,
            Err(e) => return e,
        };
        let Node::Path(path, kind) = node else {
            return err(Errno::EOPNOTSUPP);
        };
        if name.starts_with("trusted.") && !Self::xattr_admin(cx) {
            return err(Errno::ENODATA); // invisible without CAP_SYS_ADMIN
        }
        if name.starts_with("user.") && !matches!(kind, NodeKind::File | NodeKind::Dir) {
            return err(Errno::ENODATA); // user.* exists only on files/dirs
        }
        match vfs.getxattr(&path, &name) {
            Ok(v) => copy_out(mem, buf, size.min(XATTR_SIZE_MAX as u64), &v),
            Err(e) => io_errno(&e),
        }
    }

    /// `setxattr`/`lsetxattr`/`fsetxattr`/`setxattrat`: store `(value, size)`
    /// as attribute `name`, honoring `XATTR_CREATE` (fail `EEXIST` if present)
    /// and `XATTR_REPLACE` (fail `ENODATA` if absent).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_setxattr(
        &self,
        vfs: &mut MountTable,
        cx: &ServiceCtx,
        t: &XattrTarget,
        name: u64,
        value: u64,
        size: u64,
        flags: u64,
        mem: &GuestMemory,
    ) -> i64 {
        if flags & !(XATTR_CREATE | XATTR_REPLACE) != 0 || flags == XATTR_CREATE | XATTR_REPLACE {
            return err(Errno::EINVAL);
        }
        let name = match check_name(mem, name) {
            Ok(n) => n,
            Err(e) => return e,
        };
        if size as usize > XATTR_SIZE_MAX {
            return err(Errno::E2BIG);
        }
        let bytes = if size == 0 {
            Vec::new()
        } else {
            match mem.read_vec(value, size as usize) {
                Ok(b) => b,
                Err(_) => return err(Errno::EFAULT),
            }
        };
        let node = match self.xattr_node(vfs, cx, t, mem) {
            Ok(n) => n,
            Err(e) => return e,
        };
        let Node::Path(path, kind) = node else {
            return err(Errno::EOPNOTSUPP);
        };
        if name.starts_with("trusted.") && !Self::xattr_admin(cx) {
            return err(Errno::EPERM);
        }
        if name.starts_with("user.") && !matches!(kind, NodeKind::File | NodeKind::Dir) {
            return err(Errno::EPERM);
        }
        let exists = vfs.getxattr(&path, &name).is_ok();
        if flags == XATTR_CREATE && exists {
            return err(Errno::EEXIST);
        }
        if flags == XATTR_REPLACE && !exists {
            return err(Errno::ENODATA);
        }
        match vfs.setxattr(&path, &name, &bytes) {
            Ok(()) => 0,
            Err(e) => io_errno(&e),
        }
    }

    /// `listxattr`/`llistxattr`/`flistxattr`/`listxattrat`: the attribute names
    /// as a NUL-separated list (length only when `size == 0`). Names the caller
    /// may not see (`trusted.*` without `CAP_SYS_ADMIN`) are omitted.
    pub(super) fn sys_listxattr(
        &self,
        vfs: &mut MountTable,
        cx: &ServiceCtx,
        t: &XattrTarget,
        buf: u64,
        size: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let node = match self.xattr_node(vfs, cx, t, mem) {
            Ok(n) => n,
            Err(e) => return e,
        };
        let Node::Path(path, _) = node else {
            return 0; // no attribute support: an empty list
        };
        let names = match vfs.listxattr(&path) {
            Ok(n) => n,
            Err(e) => return io_errno(&e),
        };
        let admin = Self::xattr_admin(cx);
        let mut list = Vec::new();
        for n in names {
            if n.starts_with("trusted.") && !admin {
                continue;
            }
            list.extend_from_slice(n.as_bytes());
            list.push(0);
        }
        if list.len() > XATTR_SIZE_MAX {
            return err(Errno::E2BIG);
        }
        copy_out(mem, buf, size, &list)
    }

    /// `removexattr`/`lremovexattr`/`fremovexattr`/`removexattrat`.
    pub(super) fn sys_removexattr(
        &self,
        vfs: &mut MountTable,
        cx: &ServiceCtx,
        t: &XattrTarget,
        name: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let name = match check_name(mem, name) {
            Ok(n) => n,
            Err(e) => return e,
        };
        let node = match self.xattr_node(vfs, cx, t, mem) {
            Ok(n) => n,
            Err(e) => return e,
        };
        let Node::Path(path, kind) = node else {
            return err(Errno::EOPNOTSUPP);
        };
        if name.starts_with("trusted.") && !Self::xattr_admin(cx) {
            return err(Errno::EPERM);
        }
        if name.starts_with("user.") && !matches!(kind, NodeKind::File | NodeKind::Dir) {
            return err(Errno::EPERM);
        }
        match vfs.removexattr(&path, &name) {
            Ok(()) => 0,
            Err(e) => io_errno(&e),
        }
    }

    /// Decode the `struct xattr_args { u64 value; u32 size; u32 flags; }` the
    /// `setxattrat`/`getxattrat` syscalls take, checking the `usize` the caller
    /// declared (the struct is extensible: a larger one must be zero-padded,
    /// `E2BIG` otherwise; anything under the v0 size is `EINVAL`).
    pub(super) fn read_xattr_args(
        mem: &GuestMemory,
        uargs: u64,
        usize_: u64,
    ) -> Result<(u64, u64, u64), i64> {
        const XATTR_ARGS_SIZE_VER0: u64 = 16;
        if usize_ < XATTR_ARGS_SIZE_VER0 {
            return Err(err(Errno::EINVAL));
        }
        if usize_ > 4096 {
            return Err(err(Errno::E2BIG));
        }
        let Ok(raw) = mem.read_vec(uargs, usize_ as usize) else {
            return Err(err(Errno::EFAULT));
        };
        if raw[16..].iter().any(|&b| b != 0) {
            return Err(err(Errno::E2BIG));
        }
        let value = u64::from_le_bytes(raw[0..8].try_into().unwrap());
        let size = u64::from(u32::from_le_bytes(raw[8..12].try_into().unwrap()));
        let flags = u64::from(u32::from_le_bytes(raw[12..16].try_into().unwrap()));
        Ok((value, size, flags))
    }
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{BASE, call, e, put_str, setup};
    use crate::abi::arch::Sysno;
    use crate::abi::errno::Errno;

    const AT_FDCWD: u64 = (-100i64) as u64;

    #[test]
    fn set_get_list_remove_roundtrip_on_tmpfs() {
        let (k, mut mem, mut v, mut cx) = setup();
        let (path, name, val, buf) = (BASE, BASE + 0x100, BASE + 0x200, BASE + 0x1000);
        put_str(&mut mem, path, "/f");
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [AT_FDCWD, path, 0o102, 0o644, 0, 0],
        );
        assert!(fd >= 0);
        put_str(&mut mem, name, "user.mime");
        mem.write(val, b"text/plain").unwrap();
        // Nothing set yet.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Getxattr,
                [path, name, buf, 64, 0, 0]
            ),
            e(Errno::ENODATA)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Listxattr,
                [path, buf, 0, 0, 0, 0]
            ),
            0
        );
        // XATTR_REPLACE on a missing attribute fails; a plain set works.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Setxattr,
                [path, name, val, 10, 2, 0]
            ),
            e(Errno::ENODATA)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Setxattr,
                [path, name, val, 10, 0, 0]
            ),
            0
        );
        // XATTR_CREATE on an existing one fails.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Setxattr,
                [path, name, val, 10, 1, 0]
            ),
            e(Errno::EEXIST)
        );
        // Size query, too-small buffer, then the value (via the fd spelling).
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Getxattr,
                [path, name, buf, 0, 0, 0]
            ),
            10
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Getxattr,
                [path, name, buf, 4, 0, 0]
            ),
            e(Errno::ERANGE)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Fgetxattr,
                [fd as u64, name, buf, 64, 0, 0]
            ),
            10
        );
        assert_eq!(mem.read_vec(buf, 10).unwrap(), b"text/plain");
        // The list is NUL-separated names.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Flistxattr,
                [fd as u64, buf, 64, 0, 0, 0]
            ),
            10
        );
        assert_eq!(mem.read_vec(buf, 10).unwrap(), b"user.mime\0");
        // The attribute follows a rename (it lives in the node).
        put_str(&mut mem, BASE + 0x300, "/g");
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Renameat,
                [AT_FDCWD, path, AT_FDCWD, BASE + 0x300, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Getxattr,
                [BASE + 0x300, name, buf, 64, 0, 0]
            ),
            10
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Removexattr,
                [BASE + 0x300, name, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Removexattr,
                [BASE + 0x300, name, 0, 0, 0, 0]
            ),
            e(Errno::ENODATA)
        );
    }

    #[test]
    fn names_are_validated_before_the_filesystem() {
        let (k, mut mem, mut v, mut cx) = setup();
        let (path, name, buf) = (BASE, BASE + 0x100, BASE + 0x1000);
        put_str(&mut mem, path, "/");
        put_str(&mut mem, name, "bogus.ns");
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Getxattr,
                [path, name, buf, 64, 0, 0]
            ),
            e(Errno::EOPNOTSUPP)
        );
        put_str(&mut mem, name, "user.");
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Setxattr,
                [path, name, buf, 1, 0, 0]
            ),
            e(Errno::EINVAL)
        );
        put_str(&mut mem, name, "user.x");
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Setxattr,
                [path, name, buf, 1, 3, 0]
            ),
            e(Errno::EINVAL)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Setxattr,
                [path, name, buf, 70_000, 0, 0]
            ),
            e(Errno::E2BIG)
        );
        put_str(&mut mem, BASE + 0x200, "/missing");
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Getxattr,
                [BASE + 0x200, name, buf, 64, 0, 0]
            ),
            e(Errno::ENOENT)
        );
        // A pipe has no xattr support: get is EOPNOTSUPP, list is empty.
        let fds = BASE + 0x400;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Pipe2,
                [fds, 0, 0, 0, 0, 0]
            ),
            0
        );
        let rfd = u64::from(mem.read_u32(fds).unwrap());
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Fgetxattr,
                [rfd, name, buf, 64, 0, 0]
            ),
            e(Errno::EOPNOTSUPP)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Flistxattr,
                [rfd, buf, 64, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Fgetxattr,
                [99, name, buf, 64, 0, 0]
            ),
            e(Errno::EBADF)
        );
    }

    #[test]
    fn xattrat_takes_struct_args_and_at_flags() {
        let (k, mut mem, mut v, mut cx) = setup();
        let (path, name, val, args, buf) = (
            BASE,
            BASE + 0x100,
            BASE + 0x200,
            BASE + 0x300,
            BASE + 0x1000,
        );
        put_str(&mut mem, path, "/");
        put_str(&mut mem, name, "trusted.k");
        mem.write(val, b"vv").unwrap();
        let mut a = [0u8; 16];
        a[0..8].copy_from_slice(&val.to_le_bytes());
        a[8..12].copy_from_slice(&2u32.to_le_bytes());
        mem.write(args, &a).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Setxattrat,
                [AT_FDCWD, path, 0, name, args, 16]
            ),
            0
        );
        // Undersized args struct and unknown at_flags are EINVAL.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Setxattrat,
                [AT_FDCWD, path, 0, name, args, 8]
            ),
            e(Errno::EINVAL)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Listxattrat,
                [AT_FDCWD, path, 0x4, buf, 64, 0]
            ),
            e(Errno::EINVAL)
        );
        // getxattrat reads into the struct's buffer.
        a[0..8].copy_from_slice(&buf.to_le_bytes());
        a[8..12].copy_from_slice(&64u32.to_le_bytes());
        mem.write(args, &a).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Getxattrat,
                [AT_FDCWD, path, 0, name, args, 16]
            ),
            2
        );
        assert_eq!(mem.read_vec(buf, 2).unwrap(), b"vv");
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Removexattrat,
                [AT_FDCWD, path, 0, name, 0, 0]
            ),
            0
        );
        // trusted.* is hidden from a non-root caller.
        cx.cur.creds.euid = 1000;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Setxattrat,
                [AT_FDCWD, path, 0, name, args, 16]
            ),
            e(Errno::EPERM)
        );
    }
}
