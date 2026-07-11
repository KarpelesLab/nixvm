//! Encoding guest ABI structures: `struct stat` and `linux_dirent64`.
//!
//! Layouts follow the arm64 / asm-generic 64-bit definitions. When x86-64 guest
//! support lands its `stat` layout is identical for these fields, so this is
//! shared.

use crate::fs::{Attrs, NodeKind};

const S_IFCHR: u32 = 0o020_000;

/// The 128-byte `struct stat` for `attrs`.
pub fn encode_stat(attrs: &Attrs) -> [u8; 128] {
    let mut b = [0u8; 128];
    let put64 = |b: &mut [u8; 128], off: usize, v: u64| b[off..off + 8].copy_from_slice(&v.to_le_bytes());
    let put32 = |b: &mut [u8; 128], off: usize, v: u32| b[off..off + 4].copy_from_slice(&v.to_le_bytes());

    put64(&mut b, 0, 1); // st_dev
    put64(&mut b, 8, attrs.inode); // st_ino
    put32(&mut b, 16, attrs.mode); // st_mode (type + perms)
    put32(&mut b, 20, attrs.nlink); // st_nlink
    put32(&mut b, 24, attrs.uid);
    put32(&mut b, 28, attrs.gid);
    put64(&mut b, 32, 0); // st_rdev
    // 40: __pad1
    put64(&mut b, 48, attrs.size); // st_size
    put32(&mut b, 56, 4096); // st_blksize
    put64(&mut b, 64, attrs.size.div_ceil(512)); // st_blocks (512-byte units)
    let t = attrs.mtime as u64;
    put64(&mut b, 72, t); // st_atime
    put64(&mut b, 88, t); // st_mtime
    put64(&mut b, 104, t); // st_ctime
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
        };
        let b = encode_stat(&attrs);
        assert_eq!(u64::from_le_bytes(b[8..16].try_into().unwrap()), 42); // st_ino
        assert_eq!(u32::from_le_bytes(b[16..20].try_into().unwrap()), 0o100_644);
        assert_eq!(u64::from_le_bytes(b[48..56].try_into().unwrap()), 1234); // st_size
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
