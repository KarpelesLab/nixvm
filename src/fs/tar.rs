//! Minimal tar (ustar / GNU) reader — enough to unpack a distro root image
//! (e.g. an Alpine minirootfs `.tar`) into a [`MountTable`]'s writable root.
//!
//! Only what a rootfs needs: regular files, directories, and symlinks, plus
//! GNU long-name (`L`) records and the ustar `prefix` field for long paths.
//! Hardlinks and device/fifo nodes are skipped (a rootfs boots without them).
//! No compression: the caller decompresses first (the browser does gzip via
//! `DecompressionStream`), so this takes a plain, uncompressed tar.

use crate::fs::MountTable;

const BLOCK: usize = 512;

/// Read a NUL-terminated string from a fixed-width header field.
fn field_str(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}

/// Parse an octal numeric header field (size/mode), ignoring NUL/space padding.
fn field_octal(bytes: &[u8]) -> u64 {
    let mut v = 0u64;
    for &b in bytes {
        if (b'0'..=b'7').contains(&b) {
            v = v * 8 + u64::from(b - b'0');
        } else if b == 0 || b == b' ' {
            if v != 0 {
                break;
            }
        } else {
            break;
        }
    }
    v
}

/// Normalize a tar entry path to an absolute guest path: strip a leading `./`,
/// drop a trailing slash, and anchor it at `/`. Returns `None` for the archive
/// root (`.` / empty).
fn norm_path(name: &str) -> Option<String> {
    let n = name.strip_prefix("./").unwrap_or(name);
    let n = n.strip_suffix('/').unwrap_or(n);
    if n.is_empty() || n == "." {
        return None;
    }
    Some(format!("/{n}"))
}

/// Ensure every parent directory of `path` exists (best-effort `mkdir -p`).
fn ensure_parents(mounts: &mut MountTable, path: &str) {
    let mut acc = String::new();
    let comps: Vec<&str> = path.trim_start_matches('/').split('/').collect();
    // All but the last component are directories to create.
    for comp in &comps[..comps.len().saturating_sub(1)] {
        if comp.is_empty() {
            continue;
        }
        acc.push('/');
        acc.push_str(comp);
        if mounts.stat(&acc).is_none() {
            let _ = mounts.mkdir(&acc, 0o755);
        }
    }
}

/// Unpack an uncompressed tar archive into `mounts` (rooted at `/`). Returns
/// the number of entries created. Malformed/truncated tails stop extraction
/// cleanly rather than panicking.
pub fn extract_into(mounts: &mut MountTable, tar: &[u8]) -> usize {
    let mut off = 0usize;
    let mut created = 0usize;
    let mut long_name: Option<String> = None;

    while off + BLOCK <= tar.len() {
        let hdr = &tar[off..off + BLOCK];
        // Two all-zero blocks mark the end; one is enough to stop.
        if hdr.iter().all(|&b| b == 0) {
            break;
        }

        let size = field_octal(&hdr[124..136]);
        let mode = (field_octal(&hdr[100..108]) as u32) & 0o7777;
        let typeflag = hdr[156];
        let data_off = off + BLOCK;
        let data_blocks = (size as usize).div_ceil(BLOCK);
        let next = data_off + data_blocks * BLOCK;
        let data_end = (data_off + size as usize).min(tar.len());
        let data = if data_off <= tar.len() {
            &tar[data_off..data_end]
        } else {
            &[][..]
        };

        // Resolve the entry name: a pending GNU long name wins; otherwise the
        // ustar `prefix` field is prepended to `name` when set.
        let name = if let Some(ln) = long_name.take() {
            ln
        } else {
            let name = field_str(&hdr[0..100]);
            let prefix = field_str(&hdr[345..500]);
            if prefix.is_empty() {
                name
            } else {
                format!("{prefix}/{name}")
            }
        };

        match typeflag {
            b'L' => {
                // GNU long name: the real name is this record's data; applies to
                // the next header.
                long_name = Some(field_str(data));
            }
            b'5' => {
                if let Some(p) = norm_path(&name) {
                    ensure_parents(mounts, &p);
                    if mounts.stat(&p).is_none() {
                        let _ = mounts.mkdir(&p, mode);
                        created += 1;
                    }
                }
            }
            b'2' => {
                if let Some(p) = norm_path(&name) {
                    let target = field_str(&hdr[157..257]);
                    ensure_parents(mounts, &p);
                    let _ = mounts.symlink(&target, &p);
                    created += 1;
                }
            }
            b'0' | 0 => {
                if let Some(p) = norm_path(&name) {
                    ensure_parents(mounts, &p);
                    if mounts.create(&p, mode).is_ok() {
                        let _ = mounts.write_at(&p, 0, data);
                        created += 1;
                    }
                }
            }
            _ => {} // hardlink / device / fifo: skipped
        }

        off = next.max(off + BLOCK);
    }
    created
}

/// Decompress a gzip stream (e.g. a `.tar.gz` root image) with `compcol`,
/// refusing to produce more than `max_output` bytes of plaintext (a
/// decompression-bomb guard). Lets an embedder unpack a `.tar.gz` in-process —
/// the browser demo uses this instead of the browser's `DecompressionStream`.
///
/// # Errors
/// Returns a message on malformed gzip data or if the output would exceed
/// `max_output`.
#[cfg(feature = "targz")]
pub fn gunzip(gz: &[u8], max_output: u64) -> Result<Vec<u8>, String> {
    compcol::vec::decompress_to_vec_capped::<compcol::gzip::Gzip>(gz, max_output)
        .map_err(|e| format!("gzip decompression failed: {e:?}"))
}

/// Decompress a `.tar.gz` and unpack it into `mounts` (rooted at `/`), capping
/// the decompressed size at `max_output`. Returns the number of entries created.
///
/// # Errors
/// Returns a message if the gzip stream is malformed or exceeds `max_output`.
#[cfg(feature = "targz")]
pub fn extract_targz_into(
    mounts: &mut MountTable,
    targz: &[u8],
    max_output: u64,
) -> Result<usize, String> {
    let tar = gunzip(targz, max_output)?;
    Ok(extract_into(mounts, &tar))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fs::TmpFs;

    #[cfg(feature = "targz")]
    #[test]
    fn gunzip_decompresses_a_gzip_stream() {
        // `printf 'hello nixvm from compcol' | gzip -n`
        let gz: &[u8] = &[
            0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0xcb, 0x48, 0xcd, 0xc9,
            0xc9, 0x57, 0xc8, 0xcb, 0xac, 0x28, 0xcb, 0x55, 0x48, 0x2b, 0xca, 0xcf, 0x55, 0x48,
            0xce, 0xcf, 0x2d, 0x48, 0xce, 0xcf, 0x01, 0x00, 0x84, 0x0a, 0xf1, 0xef, 0x18, 0x00,
            0x00, 0x00,
        ];
        assert_eq!(gunzip(gz, 1 << 20).unwrap(), b"hello nixvm from compcol");
        // The bomb guard trips when the cap is below the real output size.
        assert!(gunzip(gz, 4).is_err());
    }

    /// Build one 512-byte ustar header + padded data for an entry.
    fn entry(name: &str, typeflag: u8, link: &str, data: &[u8]) -> Vec<u8> {
        let mut h = vec![0u8; BLOCK];
        let nb = name.as_bytes();
        h[..nb.len().min(100)].copy_from_slice(&nb[..nb.len().min(100)]);
        // mode 0644, octal, NUL-terminated
        h[100..107].copy_from_slice(b"0000644");
        // size octal in [124..136]
        let sz = format!("{:011o}", data.len());
        h[124..135].copy_from_slice(sz.as_bytes());
        h[156] = typeflag;
        let lb = link.as_bytes();
        h[157..157 + lb.len().min(100)].copy_from_slice(&lb[..lb.len().min(100)]);
        h[257..262].copy_from_slice(b"ustar");
        let mut out = h;
        out.extend_from_slice(data);
        let pad = (BLOCK - data.len() % BLOCK) % BLOCK;
        out.extend(std::iter::repeat_n(0u8, pad));
        out
    }

    #[test]
    fn extracts_files_dirs_and_symlinks() {
        let mut tar = Vec::new();
        tar.extend(entry("bin/", b'5', "", &[]));
        tar.extend(entry("bin/busybox", b'0', "", b"ELFDATA"));
        tar.extend(entry("bin/sh", b'2', "/bin/busybox", &[]));
        tar.extend(entry("etc/hostname", b'0', "", b"alpine\n"));
        tar.extend(vec![0u8; BLOCK * 2]); // end marker

        let mut mounts = MountTable::new();
        mounts.mount("/", Box::new(TmpFs::new()));
        let n = extract_into(&mut mounts, &tar);
        assert_eq!(n, 4);

        assert!(mounts.stat("/bin").is_some());
        let mut buf = vec![0u8; 7];
        assert_eq!(mounts.read_at("/bin/busybox", 0, &mut buf).unwrap(), 7);
        assert_eq!(&buf, b"ELFDATA");
        assert_eq!(mounts.readlink("/bin/sh").unwrap(), "/bin/busybox");
        // Parent /etc was auto-created for the file entry.
        assert!(mounts.stat("/etc").is_some());
    }

    #[test]
    fn auto_creates_missing_parents() {
        let mut tar = Vec::new();
        tar.extend(entry("a/b/c/file", b'0', "", b"x"));
        tar.extend(vec![0u8; BLOCK * 2]);
        let mut mounts = MountTable::new();
        mounts.mount("/", Box::new(TmpFs::new()));
        extract_into(&mut mounts, &tar);
        assert!(mounts.stat("/a/b/c").is_some());
        assert!(mounts.stat("/a/b/c/file").is_some());
    }
}
