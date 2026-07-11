//! Guest root images: resolve a name → a local squashfs file, downloading and
//! verifying it into a content-addressed cache on first use.
//!
//! Default image is a minimal Alpine root as a squashfs. Images are identified
//! by `(distro, version, arch)` and pinned by digest so runs are reproducible
//! and the download is verified.
//!
//! Resolution order for a not-yet-cached image, see [`ImageStore::ensure`]:
//! 1. `NIXVM_IMAGE_<NAME>` environment override (a local path or URL).
//! 2. [`ImageRef::source`] (a local path, `file://` URL, or `http(s)://` URL).
//! 3. Otherwise, [`ImageError::NotCached`].
//!
//! A local path or `file://` source is always supported. An `http(s)://`
//! source needs the `fetch` cargo feature (off by default, so the core stays
//! dependency-free); without it, `ensure()` returns a clear error explaining
//! how to enable it. SHA-256 verification (when [`ImageRef::sha256`] is
//! pinned) uses a small hand-rolled implementation below rather than an extra
//! dependency, so it's available in every build.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::abi::Arch;

/// A request for a guest root image.
#[derive(Debug, Clone)]
pub struct ImageRef {
    pub distro: String,
    pub version: String,
    pub arch: Arch,
    /// Pinned sha256 digest of the image file. When set, [`ImageStore::ensure`]
    /// verifies any cached-or-freshly-staged file against it and refuses to
    /// hand back (or cache) a mismatching file.
    pub sha256: Option<[u8; 32]>,
    /// Where to fetch the image from if it isn't already cached: a local
    /// filesystem path, a `file://` URL, or an `http(s)://` URL (the latter
    /// needs the `fetch` feature). `None` means "cache-only, unless the
    /// `NIXVM_IMAGE_<NAME>` environment override is set".
    pub source: Option<String>,
}

impl ImageRef {
    /// The default shipped image for `arch`: minimal Alpine.
    #[must_use]
    pub fn default_for(arch: Arch) -> Self {
        Self {
            distro: "alpine".into(),
            version: "3.20".into(),
            arch,
            sha256: None,
            source: None,
        }
    }

    /// Sets the source to fetch this image from, if it isn't already cached.
    #[must_use]
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Pins the expected sha256 digest of the image file.
    #[must_use]
    pub fn with_sha256(mut self, digest: [u8; 32]) -> Self {
        self.sha256 = Some(digest);
        self
    }

    /// Cache-relative filename, e.g. `alpine-3.20-aarch64.sqfs`.
    #[must_use]
    pub fn file_name(&self) -> String {
        format!(
            "{}-{}-{}.sqfs",
            self.distro,
            self.version,
            self.arch.as_str()
        )
    }

    /// The name of the `NIXVM_IMAGE_<NAME>` environment override that points
    /// at a local source for this image, e.g. `NIXVM_IMAGE_ALPINE_3_20_AARCH64`.
    #[must_use]
    pub fn env_override_name(&self) -> String {
        let file_name = self.file_name();
        let stem = file_name.strip_suffix(".sqfs").unwrap_or(&file_name);
        let mut out = String::with_capacity(stem.len() + 12);
        out.push_str("NIXVM_IMAGE_");
        for c in stem.chars() {
            out.push(if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            });
        }
        out
    }
}

#[derive(Debug)]
pub enum ImageError {
    /// Not cached, and no source (env override or [`ImageRef::source`]) was
    /// available to resolve it from.
    NotCached(Box<ImageRef>),
    /// A staged or cached file's sha256 didn't match the pinned digest. The
    /// offending file is never left in (or moved into) the cache.
    DigestMismatch {
        expected: [u8; 32],
        actual: [u8; 32],
    },
    /// Fetching the image failed: a network/HTTP error, or (without the
    /// `fetch` feature) an `http(s)://` source that can't be fetched at all.
    Fetch(String),
    /// A filesystem operation (read, write, rename, mkdir, …) failed.
    Io(std::io::Error),
}

impl core::fmt::Display for ImageError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotCached(r) => write!(
                f,
                "image {} not in cache and has no source (set ImageRef::source, or export {}=<path-or-url>)",
                r.file_name(),
                r.env_override_name()
            ),
            Self::DigestMismatch { expected, actual } => write!(
                f,
                "image digest mismatch: expected {}, got {}",
                hex(expected),
                hex(actual)
            ),
            Self::Fetch(msg) => write!(f, "image fetch failed: {msg}"),
            Self::Io(e) => write!(f, "image store I/O error: {e}"),
        }
    }
}

impl std::error::Error for ImageError {}

/// A local, content-addressed image cache (default: `~/.nixvm/images`).
#[derive(Debug)]
pub struct ImageStore {
    root: PathBuf,
}

impl ImageStore {
    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// The default per-user cache location.
    #[must_use]
    pub fn default_location() -> Self {
        let root = std::env::var_os("NIXVM_CACHE")
            .map(PathBuf::from)
            .or_else(|| std::env::home_dir().map(|h| h.join(".nixvm")))
            .unwrap_or_else(|| PathBuf::from(".nixvm"));
        Self::new(root.join("images"))
    }

    /// The path an image would live at once cached.
    #[must_use]
    pub fn path_for(&self, image: &ImageRef) -> PathBuf {
        self.root.join(image.file_name())
    }

    /// Ensure `image` is present locally, resolving and verifying it if
    /// needed; return its cached path.
    ///
    /// If a pinned digest is present and the cached file already matches it,
    /// this does no work beyond a hash of the existing file. Otherwise the
    /// image is resolved from its source (a local path, `file://` URL, or,
    /// with the `fetch` feature, an `http(s)://` URL), staged into a temp
    /// file next to the cache, digest-verified if pinned, and atomically
    /// renamed into place. A digest mismatch is refused and never cached.
    pub fn ensure(&self, image: &ImageRef) -> Result<PathBuf, ImageError> {
        let path = self.path_for(image);

        if path.exists() {
            match &image.sha256 {
                None => return Ok(path),
                Some(expected) => {
                    if hash_file(&path)? == *expected {
                        return Ok(path);
                    }
                    // Cached file doesn't match a pinned digest (stale or
                    // corrupt); fall through and re-stage from source.
                }
            }
        }

        let source = resolve_source(image)
            .ok_or_else(|| ImageError::NotCached(Box::new(image.clone())))?;

        std::fs::create_dir_all(&self.root).map_err(ImageError::Io)?;

        let tmp_path = self.root.join(format!(
            ".{}.{}.tmp",
            image.file_name(),
            std::process::id()
        ));
        let _cleanup = TempGuard(&tmp_path);

        let digest = match source {
            Source::Local(src) => stage_local(&src, &tmp_path)?,
            Source::Http(url) => stage_http(&url, &tmp_path)?,
        };

        if let Some(expected) = &image.sha256
            && digest != *expected
        {
            return Err(ImageError::DigestMismatch {
                expected: *expected,
                actual: digest,
            });
        }

        std::fs::rename(&tmp_path, &path).map_err(ImageError::Io)?;

        Ok(path)
    }
}

/// A resolved, not-yet-fetched image source.
enum Source {
    Local(PathBuf),
    Http(String),
}

fn classify_source(s: &str) -> Source {
    if s.starts_with("http://") || s.starts_with("https://") {
        Source::Http(s.to_string())
    } else if let Some(rest) = s.strip_prefix("file://") {
        Source::Local(PathBuf::from(rest))
    } else {
        Source::Local(PathBuf::from(s))
    }
}

/// Resolves where to fetch `image` from: the `NIXVM_IMAGE_<NAME>` environment
/// override takes priority over [`ImageRef::source`].
fn resolve_source(image: &ImageRef) -> Option<Source> {
    if let Ok(v) = std::env::var(image.env_override_name())
        && !v.is_empty()
    {
        return Some(classify_source(&v));
    }
    image.source.as_deref().map(classify_source)
}

/// Removes the file at the wrapped path when dropped (best-effort cleanup for
/// a temp file that either failed to stage or was already moved away by a
/// successful rename).
struct TempGuard<'a>(&'a Path);

impl Drop for TempGuard<'_> {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(self.0);
    }
}

/// Copies all of `src` into `dst`, hashing the bytes as they're copied so
/// callers don't need a second pass over the file to verify a digest.
fn copy_with_hash(
    src: &mut dyn Read,
    dst: &mut dyn Write,
    hasher: &mut Sha256,
) -> std::io::Result<()> {
    let mut buf = [0u8; 16384];
    loop {
        let n = src.read(&mut buf)?;
        if n == 0 {
            break;
        }
        dst.write_all(&buf[..n])?;
        hasher.update(&buf[..n]);
    }
    Ok(())
}

fn stage_local(src: &Path, tmp: &Path) -> Result<[u8; 32], ImageError> {
    let mut input = std::fs::File::open(src).map_err(ImageError::Io)?;
    let mut output = std::fs::File::create(tmp).map_err(ImageError::Io)?;
    let mut hasher = Sha256::new();
    copy_with_hash(&mut input, &mut output, &mut hasher).map_err(ImageError::Io)?;
    Ok(hasher.finalize())
}

#[cfg(feature = "fetch")]
fn stage_http(url: &str, tmp: &Path) -> Result<[u8; 32], ImageError> {
    let resp = ureq::get(url)
        .call()
        .map_err(|e| ImageError::Fetch(format!("GET {url}: {e}")))?;
    let mut reader = resp.into_reader();
    let mut output = std::fs::File::create(tmp).map_err(ImageError::Io)?;
    let mut hasher = Sha256::new();
    copy_with_hash(&mut reader, &mut output, &mut hasher).map_err(ImageError::Io)?;
    Ok(hasher.finalize())
}

#[cfg(not(feature = "fetch"))]
fn stage_http(url: &str, _tmp: &Path) -> Result<[u8; 32], ImageError> {
    Err(ImageError::Fetch(format!(
        "cannot fetch {url}: nixvm was built without the `fetch` feature \
         (rebuild with `--features fetch`, or set a local ImageRef::source / \
         NIXVM_IMAGE_<NAME> override instead)"
    )))
}

fn hash_file(path: &Path) -> Result<[u8; 32], ImageError> {
    let mut file = std::fs::File::open(path).map_err(ImageError::Io)?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 16384];
    loop {
        let n = file.read(&mut buf).map_err(ImageError::Io)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher.finalize())
}

#[cfg(test)]
fn sha256_bytes(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize()
}

fn hex(bytes: &[u8; 32]) -> String {
    use core::fmt::Write as _;
    let mut s = String::with_capacity(64);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

// --- A small, dependency-free SHA-256 (FIPS 180-4). ---
//
// Kept here rather than pulling in `sha2` so the default build (and the
// wasm32 build) stay third-party-crypto-free; digest verification of a
// pinned image is the only thing that needs it.

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

const SHA256_H0: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

struct Sha256 {
    state: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    /// Total number of *message* bytes fed via [`Sha256::update`] (excludes
    /// padding bytes appended internally by [`Sha256::finalize`]).
    msg_len: u64,
}

impl Sha256 {
    fn new() -> Self {
        Self {
            state: SHA256_H0,
            buf: [0; 64],
            buf_len: 0,
            msg_len: 0,
        }
    }

    fn update(&mut self, data: &[u8]) {
        self.msg_len = self.msg_len.wrapping_add(data.len() as u64);
        self.absorb(data);
    }

    fn absorb(&mut self, mut data: &[u8]) {
        if self.buf_len > 0 {
            let need = 64 - self.buf_len;
            let take = need.min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let block = self.buf;
                self.process_block(&block);
                self.buf_len = 0;
            }
        }
        while data.len() >= 64 {
            let mut block = [0u8; 64];
            block.copy_from_slice(&data[..64]);
            self.process_block(&block);
            data = &data[64..];
        }
        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    fn finalize(mut self) -> [u8; 32] {
        let bit_len = self.msg_len.wrapping_mul(8);
        let mut pad = [0u8; 64];
        pad[0] = 0x80;
        let pad_len = if self.buf_len < 56 {
            56 - self.buf_len
        } else {
            120 - self.buf_len
        };
        self.absorb(&pad[..pad_len]);
        self.absorb(&bit_len.to_be_bytes());
        debug_assert_eq!(self.buf_len, 0);

        let mut out = [0u8; 32];
        for (i, word) in self.state.iter().enumerate() {
            out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
        }
        out
    }

    #[allow(clippy::many_single_char_names)]
    fn process_block(&mut self, block: &[u8; 64]) {
        let mut w = [0u32; 64];
        for (i, chunk) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.state;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = h
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        self.state[0] = self.state[0].wrapping_add(a);
        self.state[1] = self.state[1].wrapping_add(b);
        self.state[2] = self.state[2].wrapping_add(c);
        self.state[3] = self.state[3].wrapping_add(d);
        self.state[4] = self.state[4].wrapping_add(e);
        self.state[5] = self.state[5].wrapping_add(f);
        self.state[6] = self.state[6].wrapping_add(g);
        self.state[7] = self.state[7].wrapping_add(h);
    }
}

#[cfg(test)]
mod tests {
    use super::{sha256_bytes, ImageError, ImageRef, ImageStore};
    use crate::abi::Arch;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static COUNTER: AtomicU64 = AtomicU64::new(0);

    /// A fresh, uniquely-named directory under the system temp dir, not yet
    /// created on disk (callers/`ensure()` create it as needed).
    fn unique_temp_path(tag: &str) -> PathBuf {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!("nixvm-image-test-{}-{tag}-{n}", std::process::id()))
    }

    struct TempFile(PathBuf);
    impl TempFile {
        fn with_contents(tag: &str, contents: &[u8]) -> Self {
            let path = unique_temp_path(tag);
            std::fs::write(&path, contents).expect("write temp source file");
            Self(path)
        }
    }
    impl Drop for TempFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    struct TempCache(PathBuf);
    impl TempCache {
        fn new(tag: &str) -> Self {
            Self(unique_temp_path(tag))
        }
    }
    impl Drop for TempCache {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn sha256_matches_known_vector() {
        // NIST/FIPS test vector: SHA-256("abc").
        let digest = sha256_bytes(b"abc");
        assert_eq!(
            super::hex(&digest),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );

        // Empty-input vector.
        let digest = sha256_bytes(b"");
        assert_eq!(
            super::hex(&digest),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn local_source_populates_cache() {
        let src = TempFile::with_contents("local-src", b"local squashfs bytes");
        let cache = TempCache::new("local-cache");
        let store = ImageStore::new(cache.0.clone());
        let image = ImageRef::default_for(Arch::Aarch64)
            .with_source(src.0.to_str().expect("utf8 temp path"));

        let path = store.ensure(&image).expect("ensure succeeds");
        assert_eq!(path, store.path_for(&image));
        assert_eq!(
            std::fs::read(&path).expect("read cached file"),
            b"local squashfs bytes"
        );

        // Second call hits the cache without needing the source anymore.
        std::fs::remove_file(&src.0).expect("remove source");
        let path2 = store.ensure(&image).expect("ensure hits cache");
        assert_eq!(path, path2);
    }

    #[test]
    fn pinned_sha256_matching_succeeds() {
        let contents = b"pinned content example";
        let digest = sha256_bytes(contents);
        let src = TempFile::with_contents("pin-ok-src", contents);
        let cache = TempCache::new("pin-ok-cache");
        let store = ImageStore::new(cache.0.clone());
        let image = ImageRef::default_for(Arch::X86_64)
            .with_source(format!("file://{}", src.0.display()))
            .with_sha256(digest);

        let path = store.ensure(&image).expect("digest matches, ensure succeeds");
        assert!(path.exists());
        assert_eq!(std::fs::read(&path).expect("read"), contents);
    }

    #[test]
    fn mismatched_sha256_rejected_and_not_cached() {
        let contents = b"some content that will not match";
        let src = TempFile::with_contents("pin-bad-src", contents);
        let cache = TempCache::new("pin-bad-cache");
        let store = ImageStore::new(cache.0.clone());
        let wrong_digest = [0u8; 32];
        let image = ImageRef::default_for(Arch::Aarch64)
            .with_source(src.0.to_str().expect("utf8 temp path"))
            .with_sha256(wrong_digest);

        let err = store.ensure(&image).expect_err("digest mismatch must error");
        assert!(
            matches!(err, ImageError::DigestMismatch { .. }),
            "expected DigestMismatch, got {err:?}"
        );
        assert!(
            !store.path_for(&image).exists(),
            "mismatching content must not be cached"
        );
    }

    #[test]
    #[cfg(not(feature = "fetch"))]
    fn http_source_without_fetch_feature_errors() {
        let cache = TempCache::new("http-no-fetch-cache");
        let store = ImageStore::new(cache.0.clone());
        let image = ImageRef::default_for(Arch::Aarch64)
            .with_source("http://example.invalid/does-not-matter.sqfs");

        let err = store.ensure(&image).expect_err("no fetch feature must error");
        assert!(
            matches!(err, ImageError::Fetch(_)),
            "expected Fetch error, got {err:?}"
        );
        assert!(!store.path_for(&image).exists());
    }
}
