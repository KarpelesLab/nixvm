//! Layers → one root filesystem tar, by the OCI image layer rules.
//!
//! Each layer is a tar applied over the ones below it: an entry replaces
//! whatever was at its path (a non-directory replacing a directory takes the
//! directory's contents with it), `.wh.NAME` deletes `NAME` from the layers
//! below, and `.wh..wh..opq` in a directory empties it of everything the
//! layers below put there. What's left is written out as one tar, each entry's
//! records copied as they were (PAX and GNU long-name headers included), in
//! the order they were laid down, so a hard link always follows its target.

use std::collections::BTreeMap;

use super::OciError;

/// One tar entry: its path and type, and the bytes of all its records.
struct Entry<'a> {
    path: String,
    typeflag: u8,
    /// A hard link's target path.
    link: String,
    raw: &'a [u8],
    data: &'a [u8],
}

const BLOCK: usize = 512;

fn octal(field: &[u8]) -> Option<u64> {
    // GNU base-256: the high bit of the first byte set.
    if field.first().is_some_and(|b| b & 0x80 != 0) {
        let mut v = u64::from(field[0] & 0x7f);
        for &b in &field[1..] {
            v = v.checked_mul(256)?.checked_add(u64::from(b))?;
        }
        return Some(v);
    }
    let s = std::str::from_utf8(field).ok()?;
    let s = s.trim_matches(|c: char| c == '\0' || c == ' ');
    if s.is_empty() {
        return Some(0);
    }
    u64::from_str_radix(s, 8).ok()
}

fn cstr(field: &[u8]) -> String {
    let end = field.iter().position(|&b| b == 0).unwrap_or(field.len());
    String::from_utf8_lossy(&field[..end]).into_owned()
}

/// `path` as a key: no leading `./` or `/`, no trailing `/`.
fn clean(path: &str) -> String {
    let mut p = path;
    loop {
        if let Some(rest) = p.strip_prefix("./") {
            p = rest;
        } else if let Some(rest) = p.strip_prefix('/') {
            p = rest;
        } else {
            break;
        }
    }
    let p = p.trim_end_matches('/');
    if p == "." {
        String::new()
    } else {
        p.to_string()
    }
}

/// The PAX records `path` and `linkpath` (and `size`) of an extended header.
fn pax(data: &[u8]) -> (Option<String>, Option<String>, Option<u64>) {
    let (mut path, mut link, mut size) = (None, None, None);
    let mut rest = data;
    while !rest.is_empty() {
        let Some(sp) = rest.iter().position(|&b| b == b' ') else {
            break;
        };
        let Some(len) = std::str::from_utf8(&rest[..sp])
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
        else {
            break;
        };
        if len <= sp || len > rest.len() {
            break;
        }
        let rec = &rest[sp + 1..len];
        let rec = rec.strip_suffix(b"\n").unwrap_or(rec);
        if let Some(eq) = rec.iter().position(|&b| b == b'=') {
            let (k, v) = (
                &rec[..eq],
                String::from_utf8_lossy(&rec[eq + 1..]).into_owned(),
            );
            match k {
                b"path" => path = Some(v),
                b"linkpath" => link = Some(v),
                b"size" => size = v.parse().ok(),
                _ => {}
            }
        }
        rest = &rest[len..];
    }
    (path, link, size)
}

/// The entries of `tar`.
fn entries(tar: &[u8]) -> Result<Vec<Entry<'_>>, OciError> {
    let bad = |why: &str, at: usize| OciError::Layer(format!("{why} at byte {at}"));
    let mut out = Vec::new();
    let mut off = 0;
    // Where the records of the entry being read began, and what its prefix
    // records said.
    let mut start = 0;
    let (mut long_name, mut long_link, mut pax_path, mut pax_link, mut pax_size) =
        (None, None, None, None, None);
    while off + BLOCK <= tar.len() {
        let h = &tar[off..off + BLOCK];
        if h.iter().all(|&b| b == 0) {
            break;
        }
        let size = match pax_size.take() {
            Some(s) => s,
            None => octal(&h[124..136]).ok_or_else(|| bad("bad size", off))?,
        };
        let data_start = off + BLOCK;
        let data_end = usize::try_from(size)
            .ok()
            .and_then(|s| data_start.checked_add(s))
            .filter(|&e| e <= tar.len())
            .ok_or_else(|| bad("entry runs past the end", off))?;
        let next = data_start + (data_end - data_start).div_ceil(BLOCK) * BLOCK;
        let data = &tar[data_start..data_end];
        let typeflag = h[156];
        match typeflag {
            b'L' => long_name = Some(cstr(data)),
            b'K' => long_link = Some(cstr(data)),
            b'x' => {
                let (p, l, s) = pax(data);
                pax_path = p.or(pax_path);
                pax_link = l.or(pax_link);
                pax_size = s;
            }
            // A global header applies to the whole archive; nothing here uses it.
            b'g' => {}
            _ => {
                let name = {
                    let prefix = if &h[257..262] == b"ustar" {
                        cstr(&h[345..500])
                    } else {
                        String::new()
                    };
                    let n = cstr(&h[0..100]);
                    if prefix.is_empty() {
                        n
                    } else {
                        format!("{prefix}/{n}")
                    }
                };
                let path = pax_path.take().or(long_name.take()).unwrap_or(name);
                let link = pax_link
                    .take()
                    .or(long_link.take())
                    .unwrap_or_else(|| cstr(&h[157..257]));
                out.push(Entry {
                    path: clean(&path),
                    typeflag,
                    link: clean(&link),
                    raw: &tar[start..next.min(tar.len())],
                    data,
                });
                start = next;
                off = next;
                continue;
            }
        }
        off = next;
    }
    Ok(out)
}

/// Where an entry goes in the output and which entry's records it is: a
/// directory a later layer declares again keeps its first place (ahead of what
/// lower layers put in it) but takes the newer metadata.
#[derive(Clone, Copy)]
struct Slot {
    order: (usize, usize),
    src: (usize, usize),
}

/// The regular files of `tar`, by path (a later entry for a path wins) — how
/// a `docker save` or OCI layout archive is read.
///
/// # Errors
/// [`OciError::Layer`] if `tar` isn't a readable tar.
pub fn files(tar: &[u8]) -> Result<BTreeMap<String, &[u8]>, OciError> {
    Ok(entries(tar)?
        .into_iter()
        .filter(|e| matches!(e.typeflag, b'0' | 0 | b'7'))
        .map(|e| (e.path, e.data))
        .collect())
}

/// Remove `dir`'s contents from `live` (and `dir` itself when `with_dir`),
/// keeping what the current layer (`layer`) put there.
fn remove_tree(live: &mut BTreeMap<String, Slot>, dir: &str, with_dir: bool, layer: usize) {
    let prefix = if dir.is_empty() {
        String::new()
    } else {
        format!("{dir}/")
    };
    let doomed: Vec<String> = live
        .range(prefix.clone()..)
        .take_while(|(k, _)| k.starts_with(&prefix))
        .filter(|(_, s)| s.src.0 < layer)
        .map(|(k, _)| k.clone())
        .collect();
    for k in doomed {
        live.remove(&k);
    }
    if with_dir && live.get(dir).is_some_and(|s| s.src.0 < layer) {
        live.remove(dir);
    }
}

/// Write hard link `link` as a regular file holding `data`: its records as
/// they were but for its own header, which becomes a regular file's.
fn copy_of(out: &mut Vec<u8>, link: &Entry<'_>, data: &[u8]) {
    let own_data = link.data.len().div_ceil(BLOCK) * BLOCK;
    let at = link.raw.len() - own_data - BLOCK;
    out.extend_from_slice(&link.raw[..at]);
    let mut h = [0u8; BLOCK];
    h.copy_from_slice(&link.raw[at..at + BLOCK]);
    h[156] = b'0';
    h[157..257].fill(0);
    let size = data.len() as u64;
    if size < 0o77_777_777_777 {
        h[124..136].copy_from_slice(format!("{size:011o}\0").as_bytes());
    } else {
        // GNU base-256 for 8 GiB and up.
        h[124..136].fill(0);
        h[124] = 0x80;
        h[128..136].copy_from_slice(&size.to_be_bytes());
    }
    h[148..156].fill(b' ');
    let sum: u32 = h.iter().map(|&b| u32::from(b)).sum();
    h[148..156].copy_from_slice(format!("{sum:06o}\0 ").as_bytes());
    out.extend_from_slice(&h);
    out.extend_from_slice(data);
    out.resize(out.len().div_ceil(BLOCK) * BLOCK, 0);
}

/// Apply `layers` (uncompressed tars, bottom first) and write the result as
/// one tar.
///
/// # Errors
/// [`OciError::Layer`] for a layer that isn't a readable tar.
pub fn flatten(layers: &[Vec<u8>]) -> Result<Vec<u8>, OciError> {
    let parsed: Vec<Vec<Entry<'_>>> = layers
        .iter()
        .map(|l| entries(l))
        .collect::<Result<_, _>>()?;
    // Path → the entry now there.
    let mut live: BTreeMap<String, Slot> = BTreeMap::new();
    for (li, layer) in parsed.iter().enumerate() {
        // Whiteouts first: they speak of the layers below, not of this one.
        for e in layer {
            let (dir, base) = match e.path.rsplit_once('/') {
                Some((d, b)) => (d, b),
                None => ("", e.path.as_str()),
            };
            if base == ".wh..wh..opq" {
                remove_tree(&mut live, dir, false, li);
            } else if let Some(name) = base.strip_prefix(".wh.") {
                let target = if dir.is_empty() {
                    name.to_string()
                } else {
                    format!("{dir}/{name}")
                };
                remove_tree(&mut live, &target, true, li);
            }
        }
        for (ei, e) in layer.iter().enumerate() {
            let base = e.path.rsplit('/').next().unwrap_or("");
            if base.starts_with(".wh.") {
                continue;
            }
            let mut order = (li, ei);
            if let Some(old) = live.get(&e.path).copied() {
                // A directory over a directory merges; anything else replaces
                // the old entry, and a directory it replaces goes with its
                // contents.
                let was_dir = parsed[old.src.0][old.src.1].typeflag == b'5';
                if was_dir && e.typeflag == b'5' {
                    order = old.order;
                } else if was_dir {
                    remove_tree(&mut live, &e.path, false, li + 1);
                }
            }
            live.insert(
                e.path.clone(),
                Slot {
                    order,
                    src: (li, ei),
                },
            );
        }
    }
    let mut keep: Vec<Slot> = live.values().copied().collect();
    keep.sort_unstable_by_key(|s| s.order);
    let mut out = Vec::with_capacity(layers.iter().map(Vec::len).sum::<usize>() / 2 + 2 * BLOCK);
    for slot in keep {
        let e = &parsed[slot.src.0][slot.src.1];
        if e.typeflag == b'1' {
            // A hard link to something a later layer removed has nothing to
            // point at: leave it out rather than write a broken archive.
            let Some(t) = live.get(&e.link) else { continue };
            let target = &parsed[t.src.0][t.src.1];
            // The squashfs writer the root goes to can't take a hard link
            // from a stream, so it's written as a copy of its target.
            if matches!(target.typeflag, b'0' | 0 | b'7') {
                copy_of(&mut out, e, target.data);
                continue;
            }
        }
        out.extend_from_slice(e.raw);
    }
    out.resize(out.len() + 2 * BLOCK, 0);
    Ok(out)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// One ustar entry: header plus padded data.
    pub(crate) fn entry(name: &str, typeflag: u8, link: &str, data: &[u8]) -> Vec<u8> {
        let mut h = [0u8; BLOCK];
        h[..name.len()].copy_from_slice(name.as_bytes());
        h[100..107].copy_from_slice(b"0000755");
        h[108..115].copy_from_slice(b"0000000");
        h[116..123].copy_from_slice(b"0000000");
        h[124..135].copy_from_slice(format!("{:011o}", data.len()).as_bytes());
        h[136..147].copy_from_slice(b"00000000000");
        h[156] = typeflag;
        h[157..157 + link.len()].copy_from_slice(link.as_bytes());
        h[257..263].copy_from_slice(b"ustar\0");
        h[263..265].copy_from_slice(b"00");
        h[148..156].copy_from_slice(b"        ");
        let sum: u32 = h.iter().map(|&b| u32::from(b)).sum();
        h[148..155].copy_from_slice(format!("{sum:06o}\0").as_bytes());
        let mut v = h.to_vec();
        v.extend_from_slice(data);
        v.resize(v.len().div_ceil(BLOCK) * BLOCK, 0);
        v
    }

    pub(crate) fn tar(entries: &[Vec<u8>]) -> Vec<u8> {
        let mut v: Vec<u8> = entries.concat();
        v.resize(v.len() + 2 * BLOCK, 0);
        v
    }

    fn paths(t: &[u8]) -> Vec<String> {
        entries(t).unwrap().into_iter().map(|e| e.path).collect()
    }

    #[test]
    fn layers_merge_with_whiteouts_and_opaque_directories() {
        let pax_name = format!("usr/{}", "x".repeat(150));
        let pax_rec = {
            let body = format!("path={pax_name}\n");
            let mut len = body.len() + 3;
            if format!("{len} {body}").len() != len {
                len += 1;
            }
            format!("{len} {body}")
        };
        let base = tar(&[
            entry("./", b'5', "", b""),
            entry("./etc/", b'5', "", b""),
            entry("./etc/passwd", b'0', "", b"root"),
            entry("./etc/gone", b'0', "", b"x"),
            entry("./var/", b'5', "", b""),
            entry("./var/cache/", b'5', "", b""),
            entry("./var/cache/a", b'0', "", b"a"),
            entry("./bin/", b'5', "", b""),
            entry("./bin/sh", b'0', "", b"sh"),
            entry("./bin/ash", b'1', "bin/sh", b""),
            entry("./bin/sh2", b'0', "", b"sh2"),
            entry("./bin/ash2", b'1', "./bin/sh2", b""),
            entry("./opt/", b'5', "", b""),
            entry("./opt/old", b'0', "", b"o"),
            entry("./lib", b'5', "", b""),
            entry("./lib/f", b'0', "", b"f"),
            [
                entry("PaxHeaders/x", b'x', "", pax_rec.as_bytes()),
                entry("usr/short", b'0', "", b"long"),
            ]
            .concat(),
        ]);
        let top = tar(&[
            entry("etc/passwd", b'0', "", b"root:x:0:0"),
            entry("etc/.wh.gone", b'0', "", b""),
            entry("var/cache/.wh..wh..opq", b'0', "", b""),
            entry("var/cache/b", b'0', "", b"b"),
            entry("bin/.wh.sh", b'0', "", b""),
            entry("lib", b'2', "usr/lib", b""),
            entry("opt/new", b'0', "", b"n"),
            entry("etc/", b'5', "", b""),
        ]);
        let merged = flatten(&[base, top]).unwrap();
        let got = paths(&merged);
        assert!(got.contains(&"etc/passwd".to_string()));
        assert!(!got.contains(&"etc/gone".to_string()), "whited out");
        assert!(
            !got.contains(&"var/cache/a".to_string()),
            "opaque dir emptied"
        );
        assert!(
            got.contains(&"var/cache/b".to_string()),
            "but keeps its own layer's"
        );
        assert!(got.contains(&"var/cache".to_string()));
        assert!(!got.contains(&"bin/sh".to_string()));
        assert!(
            !got.contains(&"bin/ash".to_string()),
            "a link to a removed file goes"
        );
        let ash2 = entries(&merged)
            .unwrap()
            .into_iter()
            .find(|e| e.path == "bin/ash2")
            .unwrap();
        assert_eq!(
            (ash2.typeflag, ash2.data),
            (b'0', &b"sh2"[..]),
            "a hard link becomes a copy"
        );
        assert!(
            !got.contains(&"lib/f".to_string()),
            "a replaced dir takes its contents"
        );
        assert!(got.contains(&"opt/old".to_string()) && got.contains(&"opt/new".to_string()));
        assert!(got.contains(&pax_name), "a PAX path survives: {got:?}");
        let passwd = entries(&merged)
            .unwrap()
            .into_iter()
            .find(|e| e.path == "etc/passwd")
            .unwrap();
        assert!(passwd.raw.windows(10).any(|w| w == b"root:x:0:0"));
        // Directories come before what's in them.
        let pos = |p: &str| got.iter().position(|g| g == p).unwrap();
        assert!(pos("etc") < pos("etc/passwd"));
        assert!(entries(&[1u8; 600]).is_err());
    }
}
