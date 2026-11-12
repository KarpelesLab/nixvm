//! Pulling an image from a registry, as a state machine that does no I/O.
//!
//! [`Pull::next_request`] says what to fetch; the caller fetches it however
//! it can (following redirects) and hands the answer to [`Pull::feed`]. The
//! registry protocol is the OCI distribution API that Docker Hub, GHCR, Quay
//! and the rest speak: the manifest for the tag (an index of platforms, or one
//! image), the image manifest for the wanted platform, its config, its layers.
//! A `401` with a `Bearer` challenge sends it for a token first, anonymous or
//! with credentials. Every blob, and every manifest named by digest, is
//! checked against its digest, so a cache may answer a [`Request`] that names
//! a digest from what it has, and a tampered byte is caught either way.

use std::fmt::Write as _;

use super::json::{self, Value};
use super::{
    Image, ImageConfig, OciError, Reference, check_arch, decompress, platform_arch, verify,
};
use crate::abi::Arch;

/// What every manifest request accepts.
const MANIFEST_TYPES: &str = "application/vnd.oci.image.index.v1+json, \
    application/vnd.docker.distribution.manifest.list.v2+json, \
    application/vnd.oci.image.manifest.v1+json, \
    application/vnd.docker.distribution.manifest.v2+json";

/// One HTTP `GET` the pull wants made.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    pub url: String,
    /// Header names are lowercase.
    pub headers: Vec<(String, String)>,
    /// The digest the response must have, when the request names one (a
    /// blob, or a manifest by digest): a cache keyed by digest may answer it.
    pub digest: Option<String>,
}

/// The answer to a [`Request`], after redirects.
#[derive(Debug, Clone, Default)]
pub struct Response {
    pub status: u16,
    /// Header names in any case; `www-authenticate` and `content-type` are
    /// the ones read.
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Response {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

#[derive(Debug, Clone)]
enum Step {
    /// The manifest (or index) for a tag or digest.
    Manifest(String),
    /// A token from the registry's auth server, then back to `then`.
    Token {
        url: String,
        then: Box<Step>,
    },
    Config,
    Layer(usize),
    Done,
}

/// A pull in progress. See the module docs.
#[derive(Debug)]
pub struct Pull {
    reference: Reference,
    arch: Arch,
    base: String,
    /// `Authorization` for registry requests, once there is one.
    auth: Option<String>,
    /// `Basic …` from the credentials given, for the token server.
    basic: Option<String>,
    step: Step,
    /// Tokens fetched, to stop a registry that keeps refusing them.
    tokens: u32,
    manifest_digest: Option<String>,
    config_digest: String,
    config: Option<ImageConfig>,
    layer_digests: Vec<String>,
    layer_sizes: Vec<u64>,
    layers: Vec<Vec<u8>>,
    bytes: u64,
}

impl Pull {
    /// Start pulling `name` (`alpine:3.20`, `ghcr.io/o/i@sha256:…`) for a
    /// guest of `arch`.
    ///
    /// # Errors
    /// [`OciError::Reference`] if `name` doesn't parse.
    pub fn new(name: &str, arch: Arch) -> Result<Self, OciError> {
        let reference = Reference::parse(name)?;
        let base = reference.api_base();
        let step = Step::Manifest(reference.reference.clone());
        Ok(Self {
            reference,
            arch,
            base,
            auth: None,
            basic: None,
            step,
            tokens: 0,
            manifest_digest: None,
            config_digest: String::new(),
            config: None,
            layer_digests: Vec::new(),
            layer_sizes: Vec::new(),
            layers: Vec::new(),
            bytes: 0,
        })
    }

    /// Log in with `user` and `password` (or a token as the password), for a
    /// private image or a higher rate limit.
    #[must_use]
    pub fn with_credentials(mut self, user: &str, password: &str) -> Self {
        self.set_credentials(user, password);
        self
    }

    /// [`Pull::with_credentials`], in place.
    pub fn set_credentials(&mut self, user: &str, password: &str) {
        self.basic = Some(format!(
            "Basic {}",
            base64(format!("{user}:{password}").as_bytes())
        ));
    }

    /// Talk to `base` (`https://mirror.example`) instead of the registry the
    /// name says: a mirror or a pull-through cache.
    #[must_use]
    pub fn with_registry_base(mut self, base: &str) -> Self {
        self.set_registry_base(base);
        self
    }

    /// [`Pull::with_registry_base`], in place.
    pub fn set_registry_base(&mut self, base: &str) {
        self.base = base.trim_end_matches('/').to_string();
    }

    /// The image being pulled.
    #[must_use]
    pub fn reference(&self) -> &Reference {
        &self.reference
    }

    /// The next thing to fetch; `None` once the pull is done.
    #[must_use]
    pub fn next_request(&self) -> Option<Request> {
        let repo = &self.reference.repository;
        let mut headers = Vec::new();
        let (url, digest) = match &self.step {
            Step::Done => return None,
            Step::Token { url, .. } => {
                if let Some(b) = &self.basic {
                    headers.push(("authorization".to_string(), b.clone()));
                }
                return Some(Request {
                    url: url.clone(),
                    headers,
                    digest: None,
                });
            }
            Step::Manifest(r) => {
                headers.push(("accept".to_string(), MANIFEST_TYPES.to_string()));
                let digest = r.starts_with("sha256:").then(|| r.clone());
                (format!("{}/v2/{repo}/manifests/{r}", self.base), digest)
            }
            Step::Config => (
                format!("{}/v2/{repo}/blobs/{}", self.base, self.config_digest),
                Some(self.config_digest.clone()),
            ),
            Step::Layer(i) => (
                format!("{}/v2/{repo}/blobs/{}", self.base, self.layer_digests[*i]),
                Some(self.layer_digests[*i].clone()),
            ),
        };
        if let Some(a) = &self.auth {
            headers.push(("authorization".to_string(), a.clone()));
        }
        Some(Request {
            url,
            headers,
            digest,
        })
    }

    /// Take the response to the last [`Pull::next_request`].
    ///
    /// # Errors
    /// What the registry said, or what's wrong with what it sent.
    pub fn feed(&mut self, resp: Response) -> Result<(), OciError> {
        let step = std::mem::replace(&mut self.step, Step::Done);
        match self.advance(step, resp) {
            Ok(next) => {
                self.step = next;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn advance(&mut self, step: Step, resp: Response) -> Result<Step, OciError> {
        if let Step::Token { then, .. } = step {
            if resp.status != 200 {
                return Err(OciError::Registry(format!(
                    "the token server refused ({}): {}",
                    resp.status,
                    snippet(&resp.body)
                )));
            }
            let v =
                json::parse(&resp.body).map_err(|e| OciError::Registry(format!("token: {e}")))?;
            let token = v
                .str_at("token")
                .or_else(|| v.str_at("access_token"))
                .ok_or_else(|| OciError::Registry("the token server sent no token".into()))?;
            self.auth = Some(format!("Bearer {token}"));
            return Ok(*then);
        }
        if resp.status == 401 {
            let challenge = resp
                .header("www-authenticate")
                .unwrap_or_default()
                .to_string();
            self.tokens += 1;
            if self.tokens > 3 {
                return Err(OciError::Registry(format!(
                    "{} refused access: {}",
                    self.reference,
                    registry_error(&resp.body)
                )));
            }
            return self.challenge(&challenge, step);
        }
        if resp.status != 200 {
            let what = match &step {
                Step::Manifest(_) => format!("{}", self.reference),
                _ => "a blob".to_string(),
            };
            let hint = if resp.status == 404 {
                " (no such image or tag)"
            } else {
                ""
            };
            return Err(OciError::Registry(format!(
                "GET {what}: HTTP {}{hint}: {}",
                resp.status,
                registry_error(&resp.body)
            )));
        }
        match step {
            Step::Manifest(r) => {
                if r.starts_with("sha256:") {
                    verify(&r, &resp.body)?;
                }
                let v = json::parse(&resp.body)
                    .map_err(|e| OciError::Manifest(format!("{}: {e}", self.reference)))?;
                let media = v
                    .str_at("mediaType")
                    .or_else(|| resp.header("content-type"))
                    .unwrap_or_default()
                    .to_string();
                if v.get("manifests").is_some() || media.contains("index") || media.contains("list")
                {
                    return Ok(Step::Manifest(pick_manifest(&v, self.arch)?));
                }
                if v.get("schemaVersion").and_then(Value::as_u64) == Some(1) {
                    return Err(OciError::Manifest(
                        "a schema 1 manifest: too old, push the image again with a current tool"
                            .into(),
                    ));
                }
                self.manifest_digest = Some(if r.starts_with("sha256:") {
                    r
                } else {
                    super::digest_of(&resp.body)
                });
                self.config_digest = v
                    .get("config")
                    .and_then(|c| c.str_at("digest"))
                    .ok_or_else(|| OciError::Manifest("a manifest without a config".into()))?
                    .to_string();
                for l in v.get("layers").map(Value::as_array).unwrap_or_default() {
                    let d = l
                        .str_at("digest")
                        .ok_or_else(|| OciError::Manifest("a layer without a digest".into()))?;
                    if !d.starts_with("sha256:") {
                        return Err(OciError::Manifest(format!("unsupported digest {d}")));
                    }
                    self.layer_digests.push(d.to_string());
                    self.layer_sizes
                        .push(l.get("size").and_then(Value::as_u64).unwrap_or(0));
                }
                Ok(Step::Config)
            }
            Step::Config => {
                verify(&self.config_digest, &resp.body)?;
                let config = ImageConfig::parse(&resp.body)?;
                check_arch(&config, self.arch)?;
                self.config = Some(config);
                Ok(if self.layer_digests.is_empty() {
                    Step::Done
                } else {
                    Step::Layer(0)
                })
            }
            Step::Layer(i) => {
                verify(&self.layer_digests[i], &resp.body)?;
                self.bytes += resp.body.len() as u64;
                self.layers.push(resp.body);
                Ok(if i + 1 < self.layer_digests.len() {
                    Step::Layer(i + 1)
                } else {
                    Step::Done
                })
            }
            Step::Token { .. } | Step::Done => Ok(Step::Done),
        }
    }

    /// Answer a `401`: fetch a `Bearer` token, or use the credentials for
    /// `Basic`, then try `step` again.
    fn challenge(&mut self, challenge: &str, step: Step) -> Result<Step, OciError> {
        let (scheme, params) = challenge.split_once(' ').unwrap_or((challenge, ""));
        if scheme.eq_ignore_ascii_case("basic") {
            return match &self.basic {
                Some(b) => {
                    self.auth = Some(b.clone());
                    Ok(step)
                }
                None => Err(OciError::Registry(format!(
                    "{} needs a login (Basic auth)",
                    self.reference.registry
                ))),
            };
        }
        if !scheme.eq_ignore_ascii_case("bearer") {
            return Err(OciError::Registry(format!(
                "unsupported auth challenge {challenge:?}"
            )));
        }
        let params = challenge_params(params);
        let get = |k: &str| {
            params
                .iter()
                .find(|(n, _)| n.eq_ignore_ascii_case(k))
                .map(|(_, v)| v.as_str())
        };
        let realm = get("realm")
            .ok_or_else(|| OciError::Registry("a Bearer challenge without a realm".into()))?;
        let mut url = realm.to_string();
        let mut sep = if url.contains('?') { '&' } else { '?' };
        if let Some(service) = get("service") {
            let _ = write!(url, "{sep}service={}", urlencode(service));
            sep = '&';
        }
        let scope = get("scope").map_or_else(
            || format!("repository:{}:pull", self.reference.repository),
            str::to_string,
        );
        let _ = write!(url, "{sep}scope={}", urlencode(&scope));
        Ok(Step::Token {
            url,
            then: Box::new(step),
        })
    }

    /// Whether everything has been fetched.
    #[must_use]
    pub fn is_done(&self) -> bool {
        matches!(self.step, Step::Done) && self.config.is_some()
    }

    /// `(layers fetched, layers in all, bytes fetched, bytes in all)`: the
    /// totals are known once the manifest is in.
    #[must_use]
    pub fn progress(&self) -> (usize, usize, u64, u64) {
        (
            self.layers.len(),
            self.layer_digests.len(),
            self.bytes,
            self.layer_sizes.iter().sum(),
        )
    }

    /// The pulled image, its layers uncompressed.
    ///
    /// # Errors
    /// [`OciError::Fetch`] if the pull isn't done; [`OciError::Layer`] for a
    /// layer that doesn't uncompress.
    pub fn finish(self) -> Result<Image, OciError> {
        if !self.is_done() {
            return Err(OciError::Fetch("the pull isn't finished".into()));
        }
        let layers = self
            .layers
            .into_iter()
            .map(|b| decompress(&b))
            .collect::<Result<_, _>>()?;
        Ok(Image {
            config: self.config.unwrap_or_default(),
            layers,
            digest: self.manifest_digest,
        })
    }
}

/// The digest of `arch`'s image in an index (a manifest list).
pub(super) fn pick_manifest(index: &Value, arch: Arch) -> Result<String, OciError> {
    let want = platform_arch(arch);
    let entries = index
        .get("manifests")
        .map(Value::as_array)
        .unwrap_or_default();
    let mut seen = Vec::new();
    let mut fallback = None;
    for m in entries {
        let Some(d) = m.str_at("digest") else {
            continue;
        };
        match m.get("platform") {
            Some(p) => {
                let (os, a) = (
                    p.str_at("os").unwrap_or(""),
                    p.str_at("architecture").unwrap_or(""),
                );
                // Build attestations are listed as unknown/unknown.
                if os == "unknown" {
                    continue;
                }
                if os == "linux" && a == want {
                    // arm64 images are v8 (or unmarked).
                    if p.str_at("variant").is_none_or(|v| v == "v8") || arch != Arch::Aarch64 {
                        return Ok(d.to_string());
                    }
                }
                seen.push(format!("{os}/{a}"));
            }
            // An entry without a platform (a plain OCI layout's): take it if
            // it's the only one.
            None => fallback = Some(d.to_string()),
        }
    }
    if seen.is_empty()
        && let Some(d) = fallback
    {
        return Ok(d);
    }
    Err(OciError::Platform(format!(
        "no linux/{want} image (it has {})",
        if seen.is_empty() {
            "none".to_string()
        } else {
            seen.join(", ")
        }
    )))
}

/// `realm="…",service="…"` → pairs.
fn challenge_params(s: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut rest = s.trim();
    while !rest.is_empty() {
        let Some(eq) = rest.find('=') else { break };
        let key = rest[..eq].trim().trim_start_matches(',').trim().to_string();
        rest = &rest[eq + 1..];
        let value;
        if let Some(r) = rest.strip_prefix('"') {
            let end = r.find('"').unwrap_or(r.len());
            value = r[..end].to_string();
            rest = r.get(end + 1..).unwrap_or("");
        } else {
            let end = rest.find(',').unwrap_or(rest.len());
            value = rest[..end].trim().to_string();
            rest = &rest[end..];
        }
        rest = rest.trim_start_matches([',', ' ']);
        out.push((key, value));
    }
    out
}

fn urlencode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) {
            out.push(b as char);
        } else {
            let _ = write!(out, "%{b:02X}");
        }
    }
    out
}

fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for c in data.chunks(3) {
        let n = (u32::from(c[0]) << 16)
            | (u32::from(*c.get(1).unwrap_or(&0)) << 8)
            | u32::from(*c.get(2).unwrap_or(&0));
        for i in 0..4 {
            if i <= c.len() {
                out.push(T[(n >> (18 - 6 * i)) as usize & 63] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// The start of a body, for an error message.
fn snippet(body: &[u8]) -> String {
    let s = String::from_utf8_lossy(&body[..body.len().min(200)]);
    s.trim().to_string()
}

/// A registry's `{"errors":[{"code","message"}]}`, or the body's start.
fn registry_error(body: &[u8]) -> String {
    if let Ok(v) = json::parse(body)
        && let Some(e) = v.get("errors").and_then(|e| e.as_array().first().cloned())
    {
        return format!(
            "{} {}",
            e.str_at("code").unwrap_or(""),
            e.str_at("message").unwrap_or("")
        )
        .trim()
        .to_string();
    }
    snippet(body)
}

#[cfg(test)]
mod tests {
    use super::super::digest_of;
    use super::super::flatten::tests::{entry, tar};
    use super::*;

    fn ok(body: &[u8]) -> Response {
        Response {
            status: 200,
            headers: vec![],
            body: body.to_vec(),
        }
    }

    #[test]
    fn a_pull_follows_the_registry_protocol() {
        let layer = tar(&[
            entry("bin/", b'5', "", b""),
            entry("bin/sh", b'0', "", b"x"),
        ]);
        let mut gz = Vec::new();
        {
            // A stored (uncompressed) deflate stream inside gzip framing is
            // still gzip; the plain tar is fine too — use the tar as is.
            gz.extend_from_slice(&layer);
        }
        let config = br#"{"architecture":"arm64","os":"linux","config":{"Cmd":["/bin/sh"]}}"#;
        let manifest = format!(
            r#"{{"schemaVersion":2,"mediaType":"application/vnd.oci.image.manifest.v1+json",
            "config":{{"digest":"{}"}},"layers":[{{"digest":"{}","size":{}}}]}}"#,
            digest_of(config),
            digest_of(&gz),
            gz.len()
        );
        let index = format!(
            r#"{{"schemaVersion":2,"mediaType":"application/vnd.oci.image.index.v1+json","manifests":[
            {{"digest":"sha256:{}","platform":{{"architecture":"unknown","os":"unknown"}}}},
            {{"digest":"{}","platform":{{"architecture":"arm64","os":"linux","variant":"v8"}}}}]}}"#,
            "0".repeat(64),
            digest_of(manifest.as_bytes())
        );
        let mut p = Pull::new("alpine:3.20", Arch::Aarch64).unwrap();
        let r = p.next_request().unwrap();
        assert_eq!(
            r.url,
            "https://registry-1.docker.io/v2/library/alpine/manifests/3.20"
        );
        assert!(
            r.headers
                .iter()
                .any(|(k, v)| k == "accept" && v.contains("oci.image.index"))
        );
        p.feed(Response {
            status: 401,
            headers: vec![(
                "WWW-Authenticate".into(),
                r#"Bearer realm="https://auth.docker.io/token",service="registry.docker.io",scope="repository:library/alpine:pull""#.into(),
            )],
            body: vec![],
        })
        .unwrap();
        let r = p.next_request().unwrap();
        assert_eq!(
            r.url,
            "https://auth.docker.io/token?service=registry.docker.io&scope=repository%3Alibrary%2Falpine%3Apull"
        );
        p.feed(ok(br#"{"token":"T"}"#)).unwrap();
        let r = p.next_request().unwrap();
        assert!(
            r.headers
                .contains(&("authorization".into(), "Bearer T".into()))
        );
        p.feed(ok(index.as_bytes())).unwrap();
        let r = p.next_request().unwrap();
        assert_eq!(
            r.digest.as_deref(),
            Some(digest_of(manifest.as_bytes()).as_str())
        );
        // A manifest that doesn't match the digest it was asked for by.
        let mut bad = Pull::new("alpine", Arch::Aarch64).unwrap();
        bad.feed(ok(index.as_bytes())).unwrap();
        assert!(matches!(bad.feed(ok(b"{}")), Err(OciError::Digest { .. })));
        p.feed(ok(manifest.as_bytes())).unwrap();
        assert_eq!(p.progress(), (0, 1, 0, gz.len() as u64));
        let r = p.next_request().unwrap();
        assert!(r.url.ends_with(&format!("/blobs/{}", digest_of(config))));
        p.feed(ok(config)).unwrap();
        p.feed(ok(&gz)).unwrap();
        assert!(p.next_request().is_none());
        let img = p.finish().unwrap();
        assert_eq!(img.config.cmd, vec!["/bin/sh"]);
        assert_eq!(
            img.digest.as_deref(),
            Some(digest_of(manifest.as_bytes()).as_str())
        );
        assert!(img.rootfs_tar().unwrap().windows(6).any(|w| w == b"bin/sh"));

        // No amd64 image in that index.
        let mut p = Pull::new("alpine", Arch::X86_64).unwrap();
        let e = p.feed(ok(index.as_bytes())).unwrap_err().to_string();
        assert!(
            e.contains("no linux/amd64 image (it has linux/arm64)"),
            "{e}"
        );
        // An error the registry explains.
        let mut p = Pull::new("nope/nope", Arch::X86_64).unwrap();
        let e = p
            .feed(Response {
                status: 404,
                headers: vec![],
                body: br#"{"errors":[{"code":"MANIFEST_UNKNOWN","message":"manifest unknown"}]}"#
                    .to_vec(),
            })
            .unwrap_err()
            .to_string();
        assert!(e.contains("MANIFEST_UNKNOWN manifest unknown"), "{e}");
    }

    #[test]
    fn credentials_and_challenges() {
        assert_eq!(base64(b"user:pass"), "dXNlcjpwYXNz");
        assert_eq!(base64(b"ab"), "YWI=");
        let p = challenge_params(
            r#"realm="https://ghcr.io/token",service="ghcr.io",scope="repository:a/b:pull""#,
        );
        assert_eq!(p[1], ("service".into(), "ghcr.io".into()));
        let mut p = Pull::new("ghcr.io/a/b", Arch::X86_64)
            .unwrap()
            .with_credentials("u", "p");
        p.feed(Response {
            status: 401,
            headers: vec![("www-authenticate".into(), r#"Basic realm="x""#.into())],
            body: vec![],
        })
        .unwrap();
        let r = p.next_request().unwrap();
        assert!(
            r.headers
                .contains(&("authorization".into(), "Basic dTpw".into()))
        );
        assert!(r.url.starts_with("https://ghcr.io/v2/a/b/manifests/latest"));
    }
}
