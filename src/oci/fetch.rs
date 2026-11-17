//! Pulling natively: [`super::Pull`] driven over rsurl (pure Rust, TLS on
//! purecrypto), with blobs cached by digest under the image cache, so a second
//! pull of the same image (or one sharing layers) fetches only its manifests.

use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};

use super::{Image, OciError, Pull, Response};
use crate::abi::Arch;
use crate::http::Http;

/// A manifest, config or token response larger than this is refused.
const MAX_DOC: usize = 16 << 20;

/// Pull `name` for `arch`, using (and filling) the default cache. Credentials
/// come from `NIXVM_REGISTRY_USER`/`NIXVM_REGISTRY_PASSWORD` when set.
///
/// # Errors
/// What the registry or the network said, or what's wrong with the image.
pub fn pull(name: &str, arch: Arch) -> Result<Image, OciError> {
    let mut p = Pull::new(name, arch)?;
    if let (Some(u), Some(pw)) = (
        crate::env::var("NIXVM_REGISTRY_USER"),
        crate::env::var("NIXVM_REGISTRY_PASSWORD"),
    ) {
        p = p.with_credentials(&u, &pw);
    }
    let cache = crate::image::ImageStore::default_location()
        .root()
        .join("blobs");
    pull_with(p, Some(&cache), &mut |_| {})
}

/// Drive `pull` to the end, caching blobs under `cache` (when given), and
/// telling `progress` how far it's got after each response.
///
/// # Errors
/// As [`pull`].
pub fn pull_with(
    mut pull: Pull,
    cache: Option<&Path>,
    progress: &mut dyn FnMut(&Pull),
) -> Result<Image, OciError> {
    let mut on = Driver {
        http: &Http::default(),
        cache,
        layers_to_disk: false,
        progress: &mut |p, _| progress(p),
        stop: &|_| false,
    };
    on.drive(&mut pull)?;
    pull.finish()
}

/// How [`Driver::drive`] runs a pull.
pub(crate) struct Driver<'a> {
    pub http: &'a Http,
    /// Where blobs are cached: `<cache>/sha256/<hex>`.
    pub cache: Option<&'a Path>,
    /// Stream layers into `cache` (which must be given) and keep them there
    /// ([`Pull::layer_kept`]) rather than feed them through memory.
    pub layers_to_disk: bool,
    /// After each step, and as a layer streams to disk: the pull, and the
    /// bytes of that layer so far (0 between steps).
    pub progress: &'a mut dyn FnMut(&Pull, u64),
    /// Checked after each step: stop there (the caller has what it needs).
    pub stop: &'a dyn Fn(&Pull) -> bool,
}

impl Driver<'_> {
    /// Run `pull` until it's done or [`Driver::stop`] says so.
    pub(crate) fn drive(&mut self, pull: &mut Pull) -> Result<(), OciError> {
        while !(self.stop)(pull)
            && let Some(req) = pull.next_request()
        {
            let cached = req
                .digest
                .as_deref()
                .and_then(|d| self.cache.map(|c| blob_path(c, d)));
            if self.layers_to_disk
                && pull.wanted_layer().is_some()
                && let Some(path) = &cached
            {
                let digest = req.digest.as_deref().unwrap_or_default();
                let headers = req.headers.clone();
                let progress = &mut *self.progress;
                match layer_to_disk(self.http, &req.url, &headers, digest, path, &mut |n| {
                    progress(pull, n);
                })? {
                    None => pull.layer_kept(),
                    // Not the blob: a 401 for a fresh token, or an error.
                    Some(resp) => pull.feed(resp)?,
                }
                (self.progress)(pull, 0);
                continue;
            }
            if let Some(path) = &cached
                && let Ok(body) = std::fs::read(path)
            {
                // A cached blob is checked like a fetched one; a bad one is
                // dropped and fetched again.
                if super::verify(req.digest.as_deref().unwrap_or(""), &body).is_ok() {
                    pull.feed(Response {
                        status: 200,
                        headers: Vec::new(),
                        body,
                    })?;
                    (self.progress)(pull, 0);
                    continue;
                }
                let _ = std::fs::remove_file(path);
            }
            let limit = if pull.wanted_layer().is_some() {
                usize::MAX
            } else {
                MAX_DOC
            };
            let r = self
                .http
                .get(&req.url, &req.headers, limit)
                .map_err(OciError::Fetch)?;
            let resp = Response {
                status: r.status,
                headers: r.headers,
                body: r.body,
            };
            if resp.status == 200
                && let Some(path) = &cached
                && req
                    .digest
                    .as_deref()
                    .is_some_and(|d| super::verify(d, &resp.body).is_ok())
            {
                store(path, &resp.body);
            }
            pull.feed(resp)?;
            (self.progress)(pull, 0);
        }
        Ok(())
    }
}

/// Make sure the blob `digest` is at `path`: there already and intact, or
/// fetched from `url`, hashed as it streams to a temporary file, and moved
/// into place only if it matches. `None` when it's there; the response when
/// the server didn't send the blob (a `401` wanting a token, an error).
fn layer_to_disk(
    http: &Http,
    url: &str,
    headers: &[(String, String)],
    digest: &str,
    path: &Path,
    progress: &mut dyn FnMut(u64),
) -> Result<Option<Response>, OciError> {
    let want = digest.strip_prefix("sha256:").unwrap_or(digest);
    if path.is_file() {
        if hash_file(path)?.eq_ignore_ascii_case(want) {
            return Ok(None);
        }
        let _ = std::fs::remove_file(path);
    }
    let dir = path
        .parent()
        .ok_or_else(|| OciError::Fetch(format!("no directory for {}", path.display())))?;
    std::fs::create_dir_all(dir)?;
    let tmp = temp_next_to(path);
    let _cleanup = Remove(&tmp);
    let mut file = std::io::BufWriter::with_capacity(256 << 10, std::fs::File::create(&tmp)?);
    let mut hasher = crate::image::Sha256::new();
    let reply = http
        .get_to(
            url,
            headers,
            &mut |b| {
                hasher.update(b);
                file.write_all(b)
            },
            &mut |n, _| progress(n),
        )
        .map_err(OciError::Fetch)?;
    if reply.status != 200 {
        return Ok(Some(Response {
            status: reply.status,
            headers: reply.headers,
            body: reply.body,
        }));
    }
    let file = file
        .into_inner()
        .map_err(|e| OciError::Io(e.into_error()))?;
    file.sync_all()?;
    drop(file);
    let got = crate::image::hex(&hasher.finalize());
    if !got.eq_ignore_ascii_case(want) {
        return Err(OciError::Digest {
            expected: digest.to_string(),
            got: format!("sha256:{got}"),
        });
    }
    std::fs::rename(&tmp, path)?;
    Ok(None)
}

/// The hex SHA-256 of the file at `path`.
pub(crate) fn hash_file(path: &Path) -> Result<String, OciError> {
    let mut f = std::fs::File::open(path)?;
    let mut h = crate::image::Sha256::new();
    let mut buf = vec![0u8; 256 << 10];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(crate::image::hex(&h.finalize()))
}

/// Where the blob `digest` lives in `cache`.
pub(crate) fn blob_path(cache: &Path, digest: &str) -> PathBuf {
    let (algo, hex) = digest.split_once(':').unwrap_or(("sha256", digest));
    cache.join(algo).join(hex)
}

/// A fresh name next to `path` to write it under before renaming it there.
pub(crate) fn temp_next_to(path: &Path) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    path.with_file_name(format!(
        ".{name}.{}.{}.part",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ))
}

/// Removes a file when dropped (a temporary one, gone already if it was moved
/// into place).
pub(crate) struct Remove<'a>(pub &'a Path);

impl Drop for Remove<'_> {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(self.0);
    }
}

/// Write a blob into the cache, atomically; a cache that can't be written is
/// only a slower next pull.
fn store(path: &Path, data: &[u8]) {
    let Some(dir) = path.parent() else { return };
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let tmp = temp_next_to(path);
    if std::fs::write(&tmp, data).is_ok() && std::fs::rename(&tmp, path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}
