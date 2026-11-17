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
//! source needs the `images` cargo feature (the pure-Rust client below) or
//! `fetch` (ureq), both off by default so the core stays dependency-free;
//! without either, `ensure()` returns a clear error explaining
//! how to enable it. SHA-256 verification (when [`ImageRef::sha256`] is
//! pinned) uses a small hand-rolled implementation below rather than an extra
//! dependency, so it's available in every build.
//!
//! ## Released images
//!
//! nixvm publishes root images as squashfs files on its GitHub releases
//! (`images-*` tags, built by `.github/workflows/images.yml`), for x86-64 and
//! arm64:
//!
//! * **`alpine`** — the Alpine minirootfs: busybox and `apk`.
//! * **`alpine-dev`** — the same with git, make, gcc and the musl headers.
//!
//! Each nixvm version pins one image release by digest (`src/image/pinned.txt`,
//! listed by [`released`]); [`ImageRef::released`] names a pinned image, and
//! `fetch` (the `images` feature) downloads it once into a cache directory,
//! verifies it, and opens it as a filesystem to mount — read-only, straight
//! from the file, no unpacking:
//!
//! ```no_run
//! # #[cfg(feature = "images")] {
//! use nixvm::{Arch, fs::Overlay, image};
//! let img = image::ImageRef::released("alpine-dev", Arch::X86_64).expect("pinned");
//! let lower = image::fetch(&img, "/var/cache/me/images".as_ref())?;
//! // `apk add` in this guest stays in ~/.local/share/me/upper for the next one.
//! let root = Overlay::persistent(Box::new(lower), "/home/me/.local/share/me/upper")?;
//! # }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! The download is HTTPS in pure Rust (purecrypto's TLS and CA roots, no C,
//! no libc of its own) over the host's TCP, or the embedder's `Connector`.
//!
//! **Updating the pins** — run the "Root images" workflow (Actions → Run
//! workflow). It builds the four images from the current Alpine release,
//! publishes them under a new `images-<alpine version>-<n>` release with their
//! `SHA256SUMS`, and attaches `pinned.txt`: copy that over
//! `src/image/pinned.txt` (`scripts/pin-images.sh <tag>`), commit, and the next
//! nixvm release pins the new images. An image release is never changed in
//! place: the pins in a published nixvm keep naming exactly what they named.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use crate::abi::Arch;

#[cfg(feature = "images")]
mod http;
#[cfg(feature = "images")]
pub use http::{Connector, TcpConnector, Transport};

/// Where the image releases are downloaded from.
pub const RELEASE_BASE: &str = "https://github.com/KarpelesLab/nixvm/releases/download";

/// The pinned image releases, as `.github/workflows/images.yml` writes them.
const PINNED: &str = include_str!("image/pinned.txt");

/// A released root image this version of nixvm pins (see the
/// [module docs](self)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Released {
    /// `alpine` or `alpine-dev`.
    pub name: String,
    /// The Alpine release it's built from, e.g. `3.22.2`.
    pub version: String,
    pub arch: Arch,
    /// The image file's SHA-256.
    pub sha256: [u8; 32],
    /// The image file's size in bytes.
    pub size: u64,
    /// Where to download it.
    pub url: String,
}

/// The released images this version of nixvm pins.
#[must_use]
pub fn released() -> Vec<Released> {
    parse_pinned(PINNED).expect("src/image/pinned.txt is well-formed (a unit test checks)")
}

/// Parse a `pinned.txt` (see the file for its format).
fn parse_pinned(text: &str) -> Result<Vec<Released>, String> {
    let mut tag = None;
    let mut out = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let bad = || format!("pinned.txt line {}: {line:?}", n + 1);
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.as_slice() {
            ["release", t] => tag = Some((*t).to_string()),
            [name, version, arch, sha, size] => {
                let tag = tag.as_deref().ok_or_else(bad)?;
                let arch = match *arch {
                    "x86_64" => Arch::X86_64,
                    "aarch64" => Arch::Aarch64,
                    _ => return Err(bad()),
                };
                let ok = |s: &str| {
                    !s.is_empty()
                        && s.bytes()
                            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
                };
                if !ok(name) || !ok(version) {
                    return Err(bad());
                }
                out.push(Released {
                    url: format!(
                        "{RELEASE_BASE}/{tag}/{name}-{version}-{}.sqfs",
                        arch.as_str()
                    ),
                    name: (*name).to_string(),
                    version: (*version).to_string(),
                    arch,
                    sha256: parse_hex(sha).ok_or_else(bad)?,
                    size: size.parse().map_err(|_| bad())?,
                });
            }
            _ => return Err(bad()),
        }
    }
    Ok(out)
}

/// A 64-digit hex SHA-256.
#[must_use]
pub fn parse_hex(s: &str) -> Option<[u8; 32]> {
    if s.len() != 64 || !s.is_ascii() {
        return None;
    }
    let mut out = [0u8; 32];
    for (i, b) in out.iter_mut().enumerate() {
        *b = u8::from_str_radix(&s[i * 2..i * 2 + 2], 16).ok()?;
    }
    Some(out)
}

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
    /// needs the `images` or `fetch` feature). `None` means "cache-only, unless the
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

    /// The pinned released image `name` (`alpine`, `alpine-dev`) for `arch`:
    /// its download URL as the source and its digest pinned. `None` if this
    /// version of nixvm pins no such image (see [`released`]).
    #[must_use]
    pub fn released(name: &str, arch: Arch) -> Option<Self> {
        released()
            .into_iter()
            .find(|r| r.name == name && r.arch == arch)
            .map(|r| Self {
                distro: r.name,
                version: r.version,
                arch,
                sha256: Some(r.sha256),
                source: Some(r.url),
            })
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
    /// The cache directory.
    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    #[must_use]
    pub fn new(root: PathBuf) -> Self {
        Self { root }
    }

    /// The default per-user cache location.
    #[must_use]
    pub fn default_location() -> Self {
        let root = crate::env::var("NIXVM_CACHE")
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

        let source =
            resolve_source(image).ok_or_else(|| ImageError::NotCached(Box::new(image.clone())))?;

        std::fs::create_dir_all(&self.root).map_err(ImageError::Io)?;

        let tmp_path = temp_path(&self.root, &image.file_name());
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

/// How [`fetch_with`] downloads.
#[cfg(feature = "images")]
#[derive(Default)]
pub struct FetchOptions<'a> {
    /// Opens the connections: the host's TCP ([`TcpConnector`]) when `None`.
    pub connector: Option<&'a dyn Connector>,
    /// CA certificates (DER) to trust besides the embedded roots — for a
    /// mirror behind a private CA.
    pub extra_roots: Vec<Vec<u8>>,
    /// Hears (bytes so far, total when known) as the download goes.
    pub progress: Option<&'a mut dyn FnMut(u64, Option<u64>)>,
    /// Hash an already-cached image again before using it. By default the
    /// cache is trusted: a file is only ever put there once its digest checked
    /// out, under a name that carries the digest.
    pub reverify: bool,
}

#[cfg(feature = "images")]
impl core::fmt::Debug for FetchOptions<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FetchOptions")
            .field("connector", &self.connector.is_some())
            .field("extra_roots", &self.extra_roots.len())
            .field("progress", &self.progress.is_some())
            .field("reverify", &self.reverify)
            .finish()
    }
}

/// Download `image` into `cache_dir` unless it's already there, verify it
/// against its pinned digest, and open it to mount: a read-only squashfs
/// served straight from the cached file. See the [module docs](self).
///
/// The image's source is its `NIXVM_IMAGE_<NAME>` override, else
/// [`ImageRef::source`]: an `http(s)://` URL (a [`ImageRef::released`] image's
/// is its release asset), a `file://` URL or a path.
///
/// # Errors
/// The image has no pinned digest or no source, the download fails, what
/// arrived doesn't match the digest (nothing is cached then), or the file
/// isn't a squashfs.
#[cfg(feature = "images")]
pub fn fetch(image: &ImageRef, cache_dir: &Path) -> Result<crate::fs::FsToolMount, ImageError> {
    fetch_with(image, cache_dir, FetchOptions::default())
}

/// [`fetch`], downloading as `opts` says.
///
/// # Errors
/// As [`fetch`].
#[cfg(feature = "images")]
pub fn fetch_with(
    image: &ImageRef,
    cache_dir: &Path,
    opts: FetchOptions<'_>,
) -> Result<crate::fs::FsToolMount, ImageError> {
    let path = fetch_file(image, cache_dir, opts)?;
    crate::fs::FsToolMount::open_squashfs(&path).map_err(ImageError::Io)
}

/// [`fetch_with`] without opening the image: the verified file's path in
/// `cache_dir`, `<name>-<version>-<arch>.<first 16 digits of its digest>.sqfs`.
///
/// # Errors
/// As [`fetch`], but for opening the image.
#[cfg(feature = "images")]
pub fn fetch_file(
    image: &ImageRef,
    cache_dir: &Path,
    opts: FetchOptions<'_>,
) -> Result<PathBuf, ImageError> {
    let Some(expected) = image.sha256 else {
        return Err(ImageError::Fetch(format!(
            "{} has no pinned digest to verify it against",
            image.file_name()
        )));
    };
    let stem = image.file_name();
    let stem = stem.strip_suffix(".sqfs").unwrap_or(&stem);
    let path = cache_dir.join(format!("{stem}.{}.sqfs", &hex(&expected)[..16]));
    if path.is_file() {
        if !opts.reverify || hash_file(&path)? == expected {
            return Ok(path);
        }
        std::fs::remove_file(&path).map_err(ImageError::Io)?;
    }
    let source =
        resolve_source(image).ok_or_else(|| ImageError::NotCached(Box::new(image.clone())))?;
    std::fs::create_dir_all(cache_dir).map_err(ImageError::Io)?;
    let tmp = temp_path(cache_dir, &image.file_name());
    let _cleanup = TempGuard(&tmp);
    let digest = match source {
        Source::Local(src) => stage_local(&src, &tmp)?,
        Source::Http(url) => {
            let tcp = TcpConnector::default();
            let mut quiet = |_: u64, _: Option<u64>| {};
            download(
                &url,
                &tmp,
                opts.connector.unwrap_or(&tcp),
                &opts.extra_roots,
                opts.progress.unwrap_or(&mut quiet),
            )?
        }
    };
    if digest != expected {
        return Err(ImageError::DigestMismatch {
            expected,
            actual: digest,
        });
    }
    std::fs::rename(&tmp, &path).map_err(ImageError::Io)?;
    Ok(path)
}

/// Download `url` into the new file `tmp` with the pure-Rust client; the
/// SHA-256 of what arrived.
#[cfg(feature = "images")]
fn download(
    url: &str,
    tmp: &Path,
    connector: &dyn Connector,
    extra_roots: &[Vec<u8>],
    progress: &mut dyn FnMut(u64, Option<u64>),
) -> Result<[u8; 32], ImageError> {
    let mut roots = purecrypto::tls::RootCertStore::with_embedded_roots();
    for der in extra_roots {
        roots
            .add_der(der.clone())
            .map_err(|e| ImageError::Fetch(format!("an extra CA certificate: {e:?}")))?;
    }
    let client = http::Client { connector, roots };
    let mut file = std::io::BufWriter::with_capacity(
        256 << 10,
        std::fs::File::create(tmp).map_err(ImageError::Io)?,
    );
    let mut hasher = Sha256::new();
    client
        .get(
            url,
            &mut |b| {
                hasher.update(b);
                file.write_all(b)
            },
            progress,
        )
        .map_err(ImageError::Fetch)?;
    let file = file
        .into_inner()
        .map_err(|e| ImageError::Io(e.into_error()))?;
    file.sync_all().map_err(ImageError::Io)?;
    Ok(hasher.finalize())
}

/// A fresh name for a file being staged into `dir`.
fn temp_path(dir: &Path, name: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    dir.join(format!(
        ".{name}.{}.{}.part",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ))
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
    if let Some(v) = crate::env::var(&image.env_override_name())
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

#[cfg(feature = "images")]
fn stage_http(url: &str, tmp: &Path) -> Result<[u8; 32], ImageError> {
    download(url, tmp, &TcpConnector::default(), &[], &mut |_, _| {})
}

#[cfg(all(feature = "fetch", not(feature = "images")))]
fn stage_http(url: &str, tmp: &Path) -> Result<[u8; 32], ImageError> {
    let resp = ureq::get(url)
        .call()
        .map_err(|e| ImageError::Fetch(format!("GET {url}: {e}")))?;
    let mut reader = resp.into_body().into_reader();
    let mut output = std::fs::File::create(tmp).map_err(ImageError::Io)?;
    let mut hasher = Sha256::new();
    copy_with_hash(&mut reader, &mut output, &mut hasher).map_err(ImageError::Io)?;
    Ok(hasher.finalize())
}

#[cfg(not(any(feature = "fetch", feature = "images")))]
fn stage_http(url: &str, _tmp: &Path) -> Result<[u8; 32], ImageError> {
    Err(ImageError::Fetch(format!(
        "cannot fetch {url}: nixvm was built without the `images` feature \
         (rebuild with `--features images`, or set a local ImageRef::source / \
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

/// The SHA-256 of `data` (also what OCI blob digests are).
#[cfg(any(test, feature = "oci"))]
pub(crate) fn sha256_bytes(data: &[u8]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(data);
    h.finalize()
}

pub(crate) fn hex(bytes: &[u8; 32]) -> String {
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
        for (i, chunk) in block.as_chunks::<4>().0.iter().enumerate() {
            w[i] = u32::from_be_bytes(*chunk);
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
    use super::{ImageError, ImageRef, ImageStore, sha256_bytes};
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

        let path = store
            .ensure(&image)
            .expect("digest matches, ensure succeeds");
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

        let err = store
            .ensure(&image)
            .expect_err("digest mismatch must error");
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
    #[cfg(not(any(feature = "fetch", feature = "images")))]
    fn http_source_without_fetch_feature_errors() {
        let cache = TempCache::new("http-no-fetch-cache");
        let store = ImageStore::new(cache.0.clone());
        let image = ImageRef::default_for(Arch::Aarch64)
            .with_source("http://example.invalid/does-not-matter.sqfs");

        let err = store
            .ensure(&image)
            .expect_err("no fetch feature must error");
        assert!(
            matches!(err, ImageError::Fetch(_)),
            "expected Fetch error, got {err:?}"
        );
        assert!(!store.path_for(&image).exists());
    }

    #[test]
    fn pinned_file_parses() {
        // Whatever is committed must parse (`released()` relies on it).
        let pins = super::parse_pinned(super::PINNED).expect("src/image/pinned.txt parses");
        for p in &pins {
            assert!(p.url.starts_with(super::RELEASE_BASE), "{p:?}");
        }
        let text = format!(
            "# comment\nrelease images-3.22.2-1\nalpine-dev 3.22.2 aarch64 {} 1234\n",
            "ab".repeat(32)
        );
        let pins = super::parse_pinned(&text).unwrap();
        assert_eq!(pins.len(), 1);
        assert_eq!(pins[0].arch, Arch::Aarch64);
        assert_eq!(pins[0].sha256, [0xab; 32]);
        assert_eq!(pins[0].size, 1234);
        assert_eq!(
            pins[0].url,
            "https://github.com/KarpelesLab/nixvm/releases/download/images-3.22.2-1/alpine-dev-3.22.2-aarch64.sqfs"
        );
        for bad in [
            "alpine 3.22.2 x86_64 {d} 1\n", // no release line
            "release t\nalpine 3.22.2 x86_64 00 1\n",
            "release t\nalpine 3.22.2 riscv64 {d} 1\n",
            "release t\nal/pine 3.22.2 x86_64 {d} 1\n",
            "release t\nalpine 3.22.2 x86_64 {d}\n",
        ] {
            let bad = bad.replace("{d}", &"00".repeat(32));
            assert!(super::parse_pinned(&bad).is_err(), "{bad:?}");
        }
    }
}

/// [`fetch`] against a fixture image: from a file, over HTTP and over HTTPS
/// from a server on loopback.
#[cfg(all(test, feature = "images"))]
mod fetch_tests {
    use super::{Connector, FetchOptions, ImageError, ImageRef, Transport, fetch_with, hex};
    use crate::abi::Arch;
    use crate::fs::MountFs;
    use std::io::{self, BufRead, BufReader, Read, Write};
    use std::net::{SocketAddr, TcpListener, TcpStream};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A small squashfs made by `mksquashfs -comp gzip`, as the images
    /// workflow makes them.
    const TINY: &[u8] = include_bytes!("../tests/data/tiny.sqfs");

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            static N: AtomicUsize = AtomicUsize::new(0);
            let p = std::env::temp_dir().join(format!(
                "nixvm-fetch-{tag}-{}-{}",
                std::process::id(),
                N.fetch_add(1, Ordering::Relaxed)
            ));
            let _ = std::fs::remove_dir_all(&p);
            Self(p)
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn tiny_ref(source: String) -> ImageRef {
        ImageRef {
            distro: "tiny".into(),
            version: "1".into(),
            arch: Arch::X86_64,
            sha256: Some(super::sha256_bytes(TINY)),
            source: Some(source),
        }
    }

    fn check_mounts(mut fs: crate::fs::FsToolMount) {
        let mut buf = [0u8; 64];
        let n = fs.read_at("etc/alpine-release", 0, &mut buf).unwrap();
        assert_eq!(&buf[..n], b"3.99.0\n");
        assert_eq!(fs.readlink("bin/sh").unwrap(), "busybox");
    }

    /// Cache files (not the in-progress ones) in `dir`.
    fn cached(dir: &Path) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(dir)
            .map(|d| {
                d.filter_map(|e| e.ok()?.file_name().into_string().ok())
                    .collect()
            })
            .unwrap_or_default();
        v.sort();
        v
    }

    #[test]
    fn a_local_image_is_verified_cached_and_opened() {
        let src = TempDir::new("src");
        std::fs::create_dir_all(&src.0).unwrap();
        let file = src.0.join("tiny.sqfs");
        std::fs::write(&file, TINY).unwrap();
        let cache = TempDir::new("cache");
        let img = tiny_ref(file.display().to_string());
        check_mounts(fetch_with(&img, &cache.0, FetchOptions::default()).unwrap());
        let name = format!(
            "tiny-1-x86_64.{}.sqfs",
            &hex(&super::sha256_bytes(TINY))[..16]
        );
        assert_eq!(cached(&cache.0), vec![name]);
        // From the cache: the source can go.
        std::fs::remove_file(&file).unwrap();
        check_mounts(fetch_with(&img, &cache.0, FetchOptions::default()).unwrap());
        // A damaged cache is caught when asked to look (and then, with the
        // source gone, there's nothing to fetch it from).
        let path = cache.0.join(&cached(&cache.0)[0]);
        std::fs::write(&path, b"not a squashfs").unwrap();
        let opts = FetchOptions {
            reverify: true,
            ..FetchOptions::default()
        };
        let err = fetch_with(&img, &cache.0, opts).unwrap_err();
        assert!(matches!(err, ImageError::Io(_)), "{err:?}");
        assert_eq!(cached(&cache.0), Vec::<String>::new());
        // An unpinned image isn't fetched at all.
        let mut unpinned = img.clone();
        unpinned.sha256 = None;
        assert!(matches!(
            fetch_with(&unpinned, &cache.0, FetchOptions::default()),
            Err(ImageError::Fetch(_))
        ));
    }

    /// Serve `body` at `/dir/tiny.sqfs?sig=x` (behind a redirect from `/r`),
    /// chunked when `chunked`, over TLS with `tls`'s identity when given; how
    /// many connections it took.
    fn serve(
        listener: TcpListener,
        tls: Option<(Vec<u8>, purecrypto::tls::SigningKey)>,
        chunked: bool,
        body: &'static [u8],
    ) -> Arc<AtomicUsize> {
        let count = Arc::new(AtomicUsize::new(0));
        let c = count.clone();
        std::thread::spawn(move || {
            for conn in listener.incoming() {
                let Ok(conn) = conn else { return };
                c.fetch_add(1, Ordering::Relaxed);
                let tls = tls.clone();
                std::thread::spawn(move || {
                    let mut stream: Box<dyn Transport> = match tls {
                        Some((cert, key)) => match tls_accept(conn, cert, key) {
                            Ok(s) => Box::new(s),
                            Err(_) => return,
                        },
                        None => Box::new(conn),
                    };
                    let _ = answer(&mut *stream, chunked, body);
                });
            }
        });
        count
    }

    fn answer(stream: &mut dyn Transport, chunked: bool, body: &[u8]) -> io::Result<()> {
        let mut r = BufReader::new(&mut *stream);
        let mut first = String::new();
        r.read_line(&mut first)?;
        loop {
            let mut l = String::new();
            r.read_line(&mut l)?;
            if l.trim().is_empty() {
                break;
            }
        }
        drop(r);
        let path = first.split(' ').nth(1).unwrap_or("");
        let mut out = Vec::new();
        match path {
            "/r" => out.extend_from_slice(
                b"HTTP/1.1 302 Found\r\nLocation: /dir/tiny.sqfs?sig=x\r\nContent-Length: 0\r\n\r\n",
            ),
            "/dir/tiny.sqfs?sig=x" if chunked => {
                out.extend_from_slice(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n");
                for c in body.chunks(1000) {
                    out.extend_from_slice(format!("{:x}\r\n", c.len()).as_bytes());
                    out.extend_from_slice(c);
                    out.extend_from_slice(b"\r\n");
                }
                out.extend_from_slice(b"0\r\n\r\n");
            }
            "/dir/tiny.sqfs?sig=x" => {
                out.extend_from_slice(
                    format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n", body.len())
                        .as_bytes(),
                );
                out.extend_from_slice(body);
            }
            _ => out.extend_from_slice(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n"),
        }
        stream.write_all(&out)?;
        stream.flush()
    }

    /// The server side of TLS over a connection, as a blocking stream.
    struct ServerTls {
        io: TcpStream,
        tls: purecrypto::tls::Connection,
        plain: Vec<u8>,
        at: usize,
    }

    fn tls_err(e: &purecrypto::tls::Error) -> io::Error {
        io::Error::other(format!("{e:?}"))
    }

    fn tls_accept(
        io: TcpStream,
        cert: Vec<u8>,
        key: purecrypto::tls::SigningKey,
    ) -> io::Result<ServerTls> {
        use purecrypto::tls::{Config, Connection};
        let cfg = Config::builder().os_rng().identity(vec![cert], key).build();
        let tls = Connection::server(&cfg).map_err(|e| tls_err(&e))?;
        let mut s = ServerTls {
            io,
            tls,
            plain: Vec::new(),
            at: 0,
        };
        let mut buf = [0u8; 16384];
        while !s.tls.is_handshake_complete() {
            let n = s.io.read(&mut buf)?;
            if n == 0 {
                return Err(io::ErrorKind::UnexpectedEof.into());
            }
            s.feed(&buf[..n])?;
        }
        Ok(s)
    }

    impl ServerTls {
        fn feed(&mut self, mut wire: &[u8]) -> io::Result<()> {
            while !wire.is_empty() {
                let n = self.tls.feed(wire).map_err(|e| tls_err(&e))?;
                if n == 0 {
                    break;
                }
                wire = &wire[n..];
            }
            if !self.tls.is_handshake_complete() {
                while let purecrypto::tls::Step::WantWrite =
                    self.tls.drive().map_err(|e| tls_err(&e))?
                {
                    let out = self.tls.pop().map_err(|e| tls_err(&e))?;
                    self.io.write_all(&out)?;
                }
            }
            self.out()
        }

        fn out(&mut self) -> io::Result<()> {
            loop {
                let out = self.tls.pop().map_err(|e| tls_err(&e))?;
                if out.is_empty() {
                    return Ok(());
                }
                self.io.write_all(&out)?;
            }
        }
    }

    impl Read for ServerTls {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            let mut buf = [0u8; 16384];
            while self.at == self.plain.len() {
                self.plain.clear();
                self.at = 0;
                if self
                    .tls
                    .recv_into(&mut self.plain)
                    .map_err(|e| tls_err(&e))?
                    > 0
                {
                    break;
                }
                let n = self.io.read(&mut buf)?;
                if n == 0 {
                    return Ok(0);
                }
                self.feed(&buf[..n])?;
            }
            let n = out.len().min(self.plain.len() - self.at);
            out[..n].copy_from_slice(&self.plain[self.at..self.at + n]);
            self.at += n;
            Ok(n)
        }
    }

    impl Write for ServerTls {
        fn write(&mut self, data: &[u8]) -> io::Result<usize> {
            self.tls.send(data).map_err(|e| tls_err(&e))?;
            self.out()?;
            Ok(data.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            self.out()
        }
    }

    /// Connects every host name to one address: the test server's.
    struct To(SocketAddr);
    impl Connector for To {
        fn connect(&self, _host: &str, _port: u16) -> io::Result<Box<dyn Transport>> {
            Ok(Box::new(TcpStream::connect(self.0)?))
        }
    }

    #[test]
    fn an_image_downloads_over_http_through_a_redirect() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        let hits = serve(l, None, true, TINY);
        let cache = TempDir::new("http");
        let img = tiny_ref(format!("http://{addr}/r"));
        let mut seen = Vec::new();
        let mut progress = |n: u64, total: Option<u64>| seen.push((n, total));
        let opts = FetchOptions {
            progress: Some(&mut progress),
            ..FetchOptions::default()
        };
        check_mounts(fetch_with(&img, &cache.0, opts).unwrap());
        assert_eq!(
            hits.load(Ordering::Relaxed),
            2,
            "the redirect, then the file"
        );
        assert_eq!(seen.last(), Some(&(TINY.len() as u64, None)));
        // Cached: no more requests.
        check_mounts(fetch_with(&img, &cache.0, FetchOptions::default()).unwrap());
        assert_eq!(hits.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn a_download_that_does_not_match_its_digest_is_not_kept() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap();
        serve(l, None, false, b"something else entirely");
        let cache = TempDir::new("mismatch");
        let img = tiny_ref(format!("http://{addr}/r"));
        let err = fetch_with(&img, &cache.0, FetchOptions::default()).unwrap_err();
        assert!(matches!(err, ImageError::DigestMismatch { .. }), "{err:?}");
        assert_eq!(cached(&cache.0), Vec::<String>::new());
        let missing = tiny_ref(format!("http://{addr}/nope"));
        let err = fetch_with(&missing, &cache.0, FetchOptions::default()).unwrap_err();
        assert!(err.to_string().contains("HTTP 404"), "{err}");
    }

    #[test]
    fn an_image_downloads_over_https() {
        use purecrypto::ec::{BoxedEcdsaPrivateKey, CurveId};
        use purecrypto::x509::{CertSigner, Certificate, DistinguishedName, Time, Validity};
        let key = BoxedEcdsaPrivateKey::generate(CurveId::P256, &mut purecrypto::rng::OsRng);
        let now = crate::clock::now_unix().as_secs();
        let validity = Validity::new(Time::from_unix(now - 3600), Time::from_unix(now + 86400));
        let cert = Certificate::self_signed_general(
            &CertSigner::Ecdsa(&key),
            &DistinguishedName::common_name("images.test"),
            &validity,
            1,
            false,
            &["images.test"],
        )
        .unwrap()
        .to_der()
        .to_vec();
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let to = To(l.local_addr().unwrap());
        let tls = Some((cert.clone(), purecrypto::tls::SigningKey::Ecdsa(key)));
        serve(l, tls, false, TINY);
        let img = tiny_ref("https://images.test/r".into());

        // Not trusted by the embedded roots alone.
        let cache = TempDir::new("https-untrusted");
        let opts = FetchOptions {
            connector: Some(&to),
            ..FetchOptions::default()
        };
        let err = fetch_with(&img, &cache.0, opts).unwrap_err();
        assert!(err.to_string().contains("TLS"), "{err}");
        assert_eq!(cached(&cache.0), Vec::<String>::new());

        // Trusted once its certificate is added.
        let cache = TempDir::new("https");
        let opts = FetchOptions {
            connector: Some(&to),
            extra_roots: vec![cert],
            ..FetchOptions::default()
        };
        check_mounts(fetch_with(&img, &cache.0, opts).unwrap());
    }
}
