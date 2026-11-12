//! Pulling natively: [`super::Pull`] driven over a blocking HTTP client, with
//! blobs cached by digest under the image cache, so a second pull of the same
//! image (or one sharing layers) fetches only its manifests.

use std::io::Read as _;
use std::path::{Path, PathBuf};

use super::{Image, OciError, Pull, Request, Response};
use crate::abi::Arch;

/// Pull `name` for `arch`, using (and filling) the default cache. Credentials
/// come from `NIXVM_REGISTRY_USER`/`NIXVM_REGISTRY_PASSWORD` when set.
///
/// # Errors
/// What the registry or the network said, or what's wrong with the image.
pub fn pull(name: &str, arch: Arch) -> Result<Image, OciError> {
    let mut p = Pull::new(name, arch)?;
    if let (Ok(u), Ok(pw)) = (
        std::env::var("NIXVM_REGISTRY_USER"),
        std::env::var("NIXVM_REGISTRY_PASSWORD"),
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
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .max_redirects(10)
        .build()
        .into();
    while let Some(req) = pull.next_request() {
        let cached = req
            .digest
            .as_deref()
            .and_then(|d| cache.map(|c| blob_path(c, d)));
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
                progress(&pull);
                continue;
            }
            let _ = std::fs::remove_file(path);
        }
        let resp = get(&agent, &req)?;
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
        progress(&pull);
    }
    pull.finish()
}

fn get(agent: &ureq::Agent, req: &Request) -> Result<Response, OciError> {
    let mut r = agent.get(&req.url);
    for (k, v) in &req.headers {
        r = r.header(k.as_str(), v.as_str());
    }
    let mut resp = r
        .call()
        .map_err(|e| OciError::Fetch(format!("GET {}: {e}", req.url)))?;
    let status = resp.status().as_u16();
    let headers = resp
        .headers()
        .iter()
        .filter_map(|(k, v)| Some((k.as_str().to_string(), v.to_str().ok()?.to_string())))
        .collect();
    let mut body = Vec::new();
    resp.body_mut()
        .as_reader()
        .read_to_end(&mut body)
        .map_err(|e| OciError::Fetch(format!("GET {}: {e}", req.url)))?;
    Ok(Response {
        status,
        headers,
        body,
    })
}

fn blob_path(cache: &Path, digest: &str) -> PathBuf {
    let (algo, hex) = digest.split_once(':').unwrap_or(("sha256", digest));
    cache.join(algo).join(hex)
}

/// Write a blob into the cache, atomically; a cache that can't be written is
/// only a slower next pull.
fn store(path: &Path, data: &[u8]) {
    let Some(dir) = path.parent() else { return };
    if std::fs::create_dir_all(dir).is_err() {
        return;
    }
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    if std::fs::write(&tmp, data).is_ok() && std::fs::rename(&tmp, path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}
