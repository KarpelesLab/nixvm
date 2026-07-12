//! Encoding guest ABI structures: `struct stat` and `linux_dirent64`.
//!
//! `linux_dirent64` and `struct statfs` follow the shared 64-bit definitions,
//! but `struct stat` is genuinely **per-arch**: arm64 uses the asm-generic
//! layout (u32 `st_mode` at offset 16, u32 `st_nlink` at 20, 128 bytes) while
//! x86-64 predates it (u64 `st_nlink` at 16, u32 `st_mode` at 24, 144 bytes).
//! Writing the arm64 layout to an x86-64 guest puts zeros where it reads
//! `st_mode` — every file looks unexecutable ("Permission denied" from a
//! shell's PATH walk), which is how the mismatch was found.

use crate::abi::Arch;
use crate::fs::{Attrs, NodeKind};

const S_IFCHR: u32 = 0o020_000;
const S_IFIFO: u32 = 0o010_000;
const S_IFSOCK: u32 = 0o140_000;

/// Encode a device number from its `major`/`minor` parts using the Linux glibc
/// `makedev` layout (the encoding of `st_rdev`). For the small all-in-range
/// values used by `/dev` nodes this reduces to `major * 256 + minor`.
///
/// Exposed for the kernel's `mknod`/`stat` path to build `st_rdev` values; the
/// `allow` keeps the lib build warning-free until that call site lands.
#[must_use]
#[allow(dead_code)]
pub fn makedev(major: u64, minor: u64) -> u64 {
    ((major & 0xfff) << 8) | (minor & 0xff) | ((minor & !0xff) << 12) | ((major & !0xfff) << 32)
}

/// The guest's `struct stat` for `attrs`, in `arch`'s layout (arm64: 128
/// bytes; x86-64: 144). The tail — `st_size`(48), `st_blksize`(56),
/// `st_blocks`(64) and the three timestamps (72/88/104) — happens to coincide
/// between the two; only the `st_nlink`/`st_mode`/`st_uid`/`st_gid`/`st_rdev`
/// block differs.
pub fn encode_stat(attrs: &Attrs, arch: Arch) -> Vec<u8> {
    let len = match arch {
        Arch::Aarch64 => 128,
        Arch::X86_64 => 144,
    };
    let mut b = vec![0u8; len];
    let put64 =
        |b: &mut [u8], off: usize, v: u64| b[off..off + 8].copy_from_slice(&v.to_le_bytes());
    let put32 =
        |b: &mut [u8], off: usize, v: u32| b[off..off + 4].copy_from_slice(&v.to_le_bytes());

    put64(&mut b, 0, 1); // st_dev
    put64(&mut b, 8, attrs.inode); // st_ino
    match arch {
        Arch::Aarch64 => {
            put32(&mut b, 16, attrs.mode); // st_mode (type + perms)
            put32(&mut b, 20, attrs.nlink); // st_nlink
            put32(&mut b, 24, attrs.uid);
            put32(&mut b, 28, attrs.gid);
            put64(&mut b, 32, attrs.rdev); // st_rdev
            // 40: __pad1
        }
        Arch::X86_64 => {
            put64(&mut b, 16, u64::from(attrs.nlink)); // st_nlink (u64 here)
            put32(&mut b, 24, attrs.mode); // st_mode
            put32(&mut b, 28, attrs.uid);
            put32(&mut b, 32, attrs.gid);
            // 36: __pad0
            put64(&mut b, 40, attrs.rdev); // st_rdev
        }
    }
    put64(&mut b, 48, attrs.size); // st_size
    put32(&mut b, 56, 4096); // st_blksize
    put64(&mut b, 64, attrs.size.div_ceil(512)); // st_blocks (512-byte units)
    let t = attrs.mtime as u64;
    put64(&mut b, 72, t); // st_atime
    put64(&mut b, 88, t); // st_mtime
    put64(&mut b, 104, t); // st_ctime
    b
}

/// The 120-byte `struct statfs` (arm64 / x86-64 layout). Reports a large,
/// mostly-empty in-memory filesystem with 4 KiB blocks.
pub fn encode_statfs() -> [u8; 120] {
    let mut b = [0u8; 120];
    let put64 =
        |b: &mut [u8; 120], off: usize, v: u64| b[off..off + 8].copy_from_slice(&v.to_le_bytes());

    put64(&mut b, 0, 0x0102_1994); // f_type (TMPFS_MAGIC)
    put64(&mut b, 8, 4096); // f_bsize
    put64(&mut b, 16, 1 << 20); // f_blocks
    put64(&mut b, 24, 1 << 19); // f_bfree
    put64(&mut b, 32, 1 << 19); // f_bavail
    put64(&mut b, 40, 1 << 16); // f_files
    put64(&mut b, 48, 1 << 15); // f_ffree
    // 56: f_fsid[2] left zero
    put64(&mut b, 64, 255); // f_namelen
    put64(&mut b, 72, 4096); // f_frsize
    // 80: f_flags, 88..120: f_spare[4] left zero
    b
}

/// Attributes for a stdio character device (fd 0/1/2 under `fstat`).
pub fn char_device_attrs() -> Attrs {
    Attrs {
        kind: NodeKind::CharDevice,
        size: 0,
        mode: S_IFCHR | 0o620,
        uid: 0,
        gid: 0,
        mtime: 0,
        inode: 0,
        nlink: 1,
        rdev: 0,
    }
}

/// Attributes for a pipe end (`fstat` on a pipe fd).
pub fn fifo_attrs() -> Attrs {
    Attrs {
        kind: NodeKind::Fifo,
        size: 0,
        mode: S_IFIFO | 0o600,
        uid: 0,
        gid: 0,
        mtime: 0,
        inode: 0,
        nlink: 1,
        rdev: 0,
    }
}

/// Attributes for a socket endpoint (`fstat` on a socket fd).
pub fn socket_attrs() -> Attrs {
    Attrs {
        kind: NodeKind::Socket,
        size: 0,
        mode: S_IFSOCK | 0o777,
        uid: 0,
        gid: 0,
        mtime: 0,
        inode: 0,
        nlink: 1,
        rdev: 0,
    }
}

/// `DT_*` value for a node kind.
fn d_type(kind: NodeKind) -> u8 {
    match kind {
        NodeKind::Fifo => 1,
        NodeKind::CharDevice => 2,
        NodeKind::Dir => 4,
        NodeKind::BlockDevice => 6,
        NodeKind::File => 8,
        NodeKind::Symlink => 10,
        NodeKind::Socket => 12,
    }
}

/// Encode `linux_dirent64` records for `entries[pos..]` into at most `cap`
/// bytes. Returns the encoded bytes and the index of the first unencoded entry.
pub fn encode_dirents(
    entries: &[(String, NodeKind, u64)],
    pos: usize,
    cap: usize,
) -> (Vec<u8>, usize) {
    let mut out = Vec::new();
    let mut i = pos;
    while i < entries.len() {
        let (name, kind, ino) = &entries[i];
        // d_ino(8) d_off(8) d_reclen(2) d_type(1) name(len+1), padded to 8.
        let reclen = (19 + name.len() + 1).div_ceil(8) * 8;
        if out.len() + reclen > cap {
            break;
        }
        let start = out.len();
        out.resize(start + reclen, 0);
        out[start..start + 8].copy_from_slice(&ino.to_le_bytes());
        out[start + 8..start + 16].copy_from_slice(&((i + 1) as i64).to_le_bytes());
        out[start + 16..start + 18].copy_from_slice(&(reclen as u16).to_le_bytes());
        out[start + 18] = d_type(*kind);
        out[start + 19..start + 19 + name.len()].copy_from_slice(name.as_bytes());
        i += 1;
    }
    (out, i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stat_size_and_mode() {
        let attrs = Attrs {
            kind: NodeKind::File,
            size: 1234,
            mode: 0o100_644,
            uid: 0,
            gid: 0,
            mtime: 0,
            inode: 42,
            nlink: 1,
            rdev: 0,
        };
        let b = encode_stat(&attrs, Arch::Aarch64);
        assert_eq!(u64::from_le_bytes(b[8..16].try_into().unwrap()), 42); // st_ino
        assert_eq!(u32::from_le_bytes(b[16..20].try_into().unwrap()), 0o100_644);
        assert_eq!(u64::from_le_bytes(b[48..56].try_into().unwrap()), 1234); // st_size
    }

    #[test]
    fn stat_encodes_rdev() {
        let attrs = Attrs {
            kind: NodeKind::CharDevice,
            size: 0,
            mode: S_IFCHR | 0o666,
            uid: 0,
            gid: 0,
            mtime: 0,
            inode: 7,
            nlink: 1,
            rdev: makedev(1, 3), // /dev/null
        };
        let b = encode_stat(&attrs, Arch::Aarch64);
        // st_rdev sits at offset 32 and must round-trip the attrs value.
        assert_eq!(
            u64::from_le_bytes(b[32..40].try_into().unwrap()),
            attrs.rdev
        );
        assert_eq!(attrs.rdev, makedev(1, 3));
    }

    #[test]
    fn stat_x86_64_layout_places_mode_at_24() {
        let attrs = Attrs {
            kind: NodeKind::File,
            size: 1234,
            mode: 0o100_755,
            uid: 3,
            gid: 4,
            mtime: 0,
            inode: 42,
            nlink: 2,
            rdev: 0,
        };
        let b = encode_stat(&attrs, Arch::X86_64);
        assert_eq!(b.len(), 144, "x86-64 struct stat is 144 bytes");
        assert_eq!(u64::from_le_bytes(b[16..24].try_into().unwrap()), 2); // st_nlink (u64)
        assert_eq!(u32::from_le_bytes(b[24..28].try_into().unwrap()), 0o100_755); // st_mode
        assert_eq!(u32::from_le_bytes(b[28..32].try_into().unwrap()), 3); // st_uid
        assert_eq!(u32::from_le_bytes(b[32..36].try_into().unwrap()), 4); // st_gid
        assert_eq!(u64::from_le_bytes(b[48..56].try_into().unwrap()), 1234); // st_size
    }

    #[test]
    fn makedev_small_values_are_major_times_256_plus_minor() {
        assert_eq!(makedev(1, 3), 256 + 3);
        assert_eq!(makedev(5, 0), 5 * 256);
        assert_eq!(makedev(1, 9), 256 + 9);
    }

    #[test]
    fn dirents_paginate() {
        let entries = vec![
            (".".to_string(), NodeKind::Dir, 1),
            ("file".to_string(), NodeKind::File, 2),
        ];
        let (bytes, next) = encode_dirents(&entries, 0, 4096);
        assert_eq!(next, 2);
        // First record: reclen at offset 16, d_type '.' is dir(4) at 18.
        assert_eq!(bytes[18], 4);
        // Too-small buffer encodes nothing.
        let (empty, n) = encode_dirents(&entries, 0, 8);
        assert!(empty.is_empty());
        assert_eq!(n, 0);
    }
}
