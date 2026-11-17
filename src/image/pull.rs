//! Root images from OCI registries: [`pull`] an image for one architecture,
//! keep it on disk, and mount it read-only from there (see [`pull`]).

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::Connector;
use crate::abi::Arch;
use crate::fs::FsToolMount;
use crate::http::Http;
use crate::oci::fetch::{Driver, Remove, blob_path, temp_next_to};
use crate::oci::{ImageConfig, OciError, Pull, Reference, decompress, flatten};

/// The official Alpine Linux image, 3.22, by the digest of its multi-platform
/// index on Docker Hub (as `docker buildx imagetools inspect alpine:3.22`
/// showed it): x86-64 and arm64 among its platforms. The same content is on
/// Google's Docker Hub mirror as
/// `mirror.gcr.io/library/alpine@sha256:49340b34…`, for when Docker Hub's
/// anonymous rate limit bites.
pub const ALPINE: &str = "docker.io/library/alpine:3.22@sha256:49340b346798875b5592e5d420241f2f9cfba5f47e62ebb1446a1b34734e898f";

/// How far a [`pull`] has got.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PullProgress {
    /// Layers fetched (or found in the cache), and how many there are.
    pub layers_done: usize,
    pub layers: usize,
    /// Compressed layer bytes fetched or found, and the total the manifest
    /// gives.
    pub bytes: u64,
    pub total_bytes: u64,
}

/// How [`pull`] reaches the registry.
#[derive(Default)]
pub struct PullOptions<'a> {
    /// Opens the connections: the host's TCP when `None`.
    pub connector: Option<Arc<dyn Connector>>,
    /// A PEM file of CA certificates to trust *instead of* the embedded
    /// roots — a registry behind a private CA.
    pub ca_bundle: Option<String>,
    /// A directory of PEM CA certificates to trust *besides* them.
    pub ca_path: Option<String>,
    /// A proxy, curl-style: `http://host:port`, `socks5h://host:port`.
    pub proxy: Option<String>,
    /// `(user, password or token)`, for a private image or a higher rate
    /// limit.
    pub credentials: Option<(String, String)>,
    /// Talk to this registry (`https://mirror.example`) instead of the one
    /// the reference names: a mirror or a pull-through cache. The digests
    /// are checked all the same.
    pub registry_base: Option<String>,
    /// Hears how far the pull has got, as it goes.
    pub progress: Option<&'a mut dyn FnMut(PullProgress)>,
}

impl core::fmt::Debug for PullOptions<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PullOptions")
            .field("connector", &self.connector.is_some())
            .field("ca_bundle", &self.ca_bundle)
            .field("ca_path", &self.ca_path)
            .field("proxy", &self.proxy)
            .field("credentials", &self.credentials.as_ref().map(|c| &c.0))
            .field("registry_base", &self.registry_base)
            .field("progress", &self.progress.is_some())
            .finish()
    }
}

/// A pulled image, flattened into a root on disk.
#[derive(Debug, Clone)]
pub struct Pulled {
    /// The image as named.
    pub reference: Reference,
    /// The digest of the image manifest for the architecture pulled.
    pub manifest_digest: String,
    /// The image's config: its environment, entrypoint, user….
    pub config: ImageConfig,
    /// The root filesystem: a squashfs file in the cache.
    pub path: PathBuf,
}

impl Pulled {
    /// Open the root to mount: read-only, served from its file.
    ///
    /// # Errors
    /// The file can't be opened or isn't a squashfs.
    pub fn open(&self) -> std::io::Result<FsToolMount> {
        FsToolMount::open_squashfs(&self.path)
    }
}

/// Pull `reference` (`alpine:3.22@sha256:…`, `ghcr.io/o/i@sha256:…`, see
/// [`ALPINE`]) for a guest of `arch` into `cache_dir`, unless it's there
/// already, and open its root to mount read-only — the lower layer of an
/// [`Overlay`](crate::fs::Overlay), whose upper may be a host directory that
/// outlives the guest ([`Overlay::persistent`](crate::fs::Overlay::persistent)).
///
/// The pull is [`crate::oci::Pull`] driven over rsurl (pure Rust), so every
/// manifest and blob is checked against the digest that names it — starting
/// from the one in the reference, when it has one:
/// `alpine:3.22@sha256:<index digest>`
/// pins the platform index, the index pins each platform's manifest, which
/// pins the config and layers. A reference by tag alone (`alpine:3.22`) works
/// too, but only a digest pins it: a tag can be moved to other content, and
/// asking what it names takes a request every time.
///
/// The cache directory holds:
///
/// * `blobs/sha256/<hex>` — every manifest named by digest, config and layer,
///   as the registry sent it (layers still compressed). A layer streams to
///   disk as it downloads, hashed on the way, and only takes its name once it
///   matches its digest.
/// * `roots/<manifest hex>.sqfs` — the layers flattened into one root, as a
///   squashfs file, keyed by the platform manifest's digest.
///
/// A pull whose root is already there stops as soon as it knows the manifest
/// (and config): for a reference by digest that's all from `blobs/`, so a
/// later boot makes no request at all, re-flattens nothing, and mounts the
/// root from its file without reading it into memory. The first pull holds
/// the uncompressed layers and the flattened tar in memory while it builds
/// that file.
///
/// # Errors
/// The reference doesn't parse, the registry or the network fails, content
/// doesn't match its digest (nothing is cached then), the image has no build
/// for `arch`, or the cache can't be written.
pub fn pull(
    reference: &str,
    arch: Arch,
    cache_dir: &Path,
    opts: PullOptions<'_>,
) -> Result<FsToolMount, OciError> {
    Ok(pull_image(reference, arch, cache_dir, opts)?.open()?)
}

/// [`pull`] without opening the root: where it is, and what the image is.
///
/// # Errors
/// As [`pull`], but for opening the root.
pub fn pull_image(
    reference: &str,
    arch: Arch,
    cache_dir: &Path,
    opts: PullOptions<'_>,
) -> Result<Pulled, OciError> {
    let mut p = Pull::new(reference, arch)?;
    if let Some((u, pw)) = &opts.credentials {
        p.set_credentials(u, pw);
    }
    if let Some(b) = &opts.registry_base {
        p.set_registry_base(b);
    }
    let http = Http {
        connector: opts.connector,
        ca_bundle: opts.ca_bundle,
        ca_path: opts.ca_path,
        proxy: opts.proxy,
    };
    let blobs = cache_dir.join("blobs");
    let roots = cache_dir.join("roots");
    let root_of = |digest: &str| {
        let hex = digest.strip_prefix("sha256:").unwrap_or(digest);
        roots.join(format!("{hex}.sqfs"))
    };
    let mut quiet = |_: PullProgress| {};
    let report = opts.progress.unwrap_or(&mut quiet);
    let mut progress = |p: &Pull, now: u64| {
        let (layers_done, layers, bytes, total_bytes) = p.progress();
        report(PullProgress {
            layers_done,
            layers,
            bytes: bytes + now,
            total_bytes,
        });
    };
    // Enough once the config is in and the root is already built.
    let stop = |p: &Pull| {
        p.config().is_some() && p.manifest_digest().is_some_and(|d| root_of(d).is_file())
    };
    Driver {
        http: &http,
        cache: Some(&blobs),
        layers_to_disk: true,
        progress: &mut progress,
        stop: &stop,
    }
    .drive(&mut p)?;
    let (Some(digest), Some(config)) = (p.manifest_digest(), p.config()) else {
        return Err(OciError::Fetch("the pull ended before the manifest".into()));
    };
    let path = root_of(digest);
    if !path.is_file() {
        if !p.is_done() {
            return Err(OciError::Fetch("the pull ended before its layers".into()));
        }
        // Every layer is in `blobs/`, checked against its digest just now.
        let layers = p
            .layer_digests()
            .iter()
            .map(|d| decompress(&std::fs::read(blob_path(&blobs, d))?))
            .collect::<Result<Vec<_>, _>>()?;
        let tar = flatten(&layers)?;
        drop(layers);
        std::fs::create_dir_all(&roots)?;
        let tmp = temp_next_to(&path);
        let _cleanup = Remove(&tmp);
        FsToolMount::write_squashfs(&tar, &tmp)?;
        std::fs::rename(&tmp, &path)?;
    }
    Ok(Pulled {
        reference: p.reference().clone(),
        manifest_digest: digest.to_string(),
        config: config.clone(),
        path,
    })
}

#[cfg(test)]
mod tests {
    use super::super::test_server::{Req, Resp, Server, To, pem, self_signed, serve_later};
    use super::*;
    use crate::fs::MountFs;
    use crate::oci::digest_of;
    use crate::oci::flatten::tests::{entry, tar};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    struct TempDir(PathBuf);
    impl TempDir {
        fn new(tag: &str) -> Self {
            static N: AtomicUsize = AtomicUsize::new(0);
            let p = std::env::temp_dir().join(format!(
                "nixvm-pull-{tag}-{}-{}",
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

    /// A two-platform image, as a registry holds it.
    struct Image {
        index: Vec<u8>,
        /// Digest → content: manifests, configs, layers.
        blobs: Blobs,
        amd64_manifest: String,
        amd64_layers: Vec<String>,
    }

    /// Digest → content.
    type Blobs = Vec<(String, Vec<u8>)>;

    fn platform(arch: &str) -> (Vec<u8>, Blobs, Vec<String>) {
        let base = tar(&[
            entry("etc/", b'5', "", b""),
            entry(
                "etc/alpine-release",
                b'0',
                "",
                format!("{arch}\n").as_bytes(),
            ),
            entry("etc/motd", b'0', "", b"hello\n"),
            entry("bin/", b'5', "", b""),
            entry("bin/busybox", b'0', "", b"\x7fELF"),
            entry("bin/sh", b'2', "busybox", b""),
        ]);
        // The top layer deletes /etc/motd and adds a file.
        let top = tar(&[
            entry("etc/", b'5', "", b""),
            entry("etc/.wh.motd", b'0', "", b""),
            entry("etc/top", b'0', "", b"top\n"),
        ]);
        let config = format!(
            r#"{{"architecture":"{arch}","os":"linux","config":{{"Cmd":["/bin/sh"],"Env":["A=1"]}}}}"#
        )
        .into_bytes();
        let manifest = format!(
            r#"{{"schemaVersion":2,"mediaType":"application/vnd.oci.image.manifest.v1+json",
            "config":{{"digest":"{}","size":{}}},"layers":[{{"digest":"{}","size":{}}},{{"digest":"{}","size":{}}}]}}"#,
            digest_of(&config),
            config.len(),
            digest_of(&base),
            base.len(),
            digest_of(&top),
            top.len()
        )
        .into_bytes();
        let layers = vec![digest_of(&base), digest_of(&top)];
        let blobs = vec![
            (digest_of(&config), config),
            (digest_of(&base), base),
            (digest_of(&top), top),
        ];
        (manifest, blobs, layers)
    }

    fn image() -> Image {
        let (amd, mut blobs, amd64_layers) = platform("amd64");
        let (arm, arm_blobs, _) = platform("arm64");
        blobs.extend(arm_blobs);
        let index = format!(
            r#"{{"schemaVersion":2,"mediaType":"application/vnd.oci.image.index.v1+json","manifests":[
            {{"digest":"{}","platform":{{"architecture":"arm64","os":"linux","variant":"v8"}}}},
            {{"digest":"sha256:{}","platform":{{"architecture":"unknown","os":"unknown"}}}},
            {{"digest":"{}","platform":{{"architecture":"amd64","os":"linux"}}}}]}}"#,
            digest_of(&arm),
            "0".repeat(64),
            digest_of(&amd)
        )
        .into_bytes();
        let amd64_manifest = digest_of(&amd);
        blobs.push((digest_of(&arm), arm));
        blobs.push((amd64_manifest.clone(), amd));
        blobs.push((digest_of(&index), index.clone()));
        Image {
            index,
            blobs,
            amd64_manifest,
            amd64_layers,
        }
    }

    /// A registry for `img` at `/v2/test/img`: token auth on everything under
    /// `/v2/`, blobs redirected to `<cdn>/cdn/<digest>` on another host (no
    /// auth there), tag `1.0`
    /// naming the index. With `tamper` set, the CDN corrupts layers.
    fn registry(
        img: Image,
        tamper: Arc<AtomicBool>,
        base: String,
        cdn: String,
    ) -> Arc<dyn Fn(&Req) -> Resp + Send + Sync> {
        Arc::new(move |r: &Req| {
            let path = r.path.split('?').next().unwrap_or("");
            if path == "/token" {
                assert!(
                    r.path.contains("scope=repository%3Atest%2Fimg%3Apull"),
                    "{}",
                    r.path
                );
                return Resp::ok(br#"{"token":"T0K3N"}"#.to_vec());
            }
            if let Some(d) = path.strip_prefix("/cdn/") {
                assert!(r.header("authorization").is_none(), "no auth to the CDN");
                let Some((_, b)) = img.blobs.iter().find(|(k, _)| k == d) else {
                    return Resp::status(404);
                };
                let mut b = b.clone();
                if tamper.load(Ordering::Relaxed) && img.amd64_layers.iter().any(|l| l == d) {
                    b[600] ^= 1;
                }
                return Resp {
                    chunked: true,
                    ..Resp::ok(b)
                };
            }
            if r.header("authorization") != Some("Bearer T0K3N") {
                return Resp::status(401)
                    .with(
                        "WWW-Authenticate",
                        &format!(r#"Bearer realm="{base}/token",service="fake""#),
                    )
                    .with("Content-Type", "application/json");
            }
            if let Some(rf) = path.strip_prefix("/v2/test/img/manifests/") {
                let body = if rf == "1.0" {
                    img.index.clone()
                } else if let Some((_, b)) = img.blobs.iter().find(|(k, _)| k == rf) {
                    b.clone()
                } else {
                    return Resp::status(404);
                };
                return Resp::ok(body)
                    .with("Content-Type", "application/vnd.oci.image.index.v1+json");
            }
            if let Some(d) = path.strip_prefix("/v2/test/img/blobs/") {
                return Resp::status(307).with("Location", &format!("{cdn}/cdn/{d}"));
            }
            Resp::status(404)
        })
    }

    fn http_registry() -> (Server, Arc<AtomicBool>, String) {
        // The handler needs the server's address.
        let tamper = Arc::new(AtomicBool::new(false));
        let (srv, slot) = serve_later(None);
        let base = format!("http://{}", srv.addr);
        let cdn = format!("http://localhost:{}", srv.addr.port());
        let _ = slot.set(registry(image(), tamper.clone(), base, cdn));
        let index = digest_of(&image().index);
        let name = srv_name(&srv);
        (srv, tamper, format!("{name}/test/img@{index}"))
    }

    fn srv_name(srv: &Server) -> String {
        format!("127.0.0.1:{}", srv.addr.port())
    }

    fn check_root(mut root: FsToolMount, arch: &str) {
        let mut buf = [0u8; 64];
        let n = root.read_at("etc/alpine-release", 0, &mut buf).unwrap();
        assert_eq!(&buf[..n], format!("{arch}\n").as_bytes());
        let n = root.read_at("etc/top", 0, &mut buf).unwrap();
        assert_eq!(&buf[..n], b"top\n");
        assert!(root.read_at("etc/motd", 0, &mut buf).is_err(), "whited out");
        assert_eq!(root.readlink("bin/sh").unwrap(), "busybox");
    }

    fn files(dir: &Path) -> Vec<String> {
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
    fn a_pull_by_digest_authenticates_picks_the_platform_and_is_reused() {
        let (srv, _tamper, reference) = http_registry();
        let cache = TempDir::new("http");
        let mut seen = Vec::new();
        let mut progress = |p: PullProgress| seen.push(p);
        let opts = PullOptions {
            progress: Some(&mut progress),
            ..PullOptions::default()
        };
        let pulled = pull_image(&reference, Arch::X86_64, &cache.0, opts).unwrap();
        let img = image();
        assert_eq!(pulled.manifest_digest, img.amd64_manifest);
        assert_eq!(pulled.config.cmd, vec!["/bin/sh"]);
        check_root(pulled.open().unwrap(), "amd64");
        let last = seen.last().copied().unwrap();
        assert_eq!((last.layers_done, last.layers), (2, 2));
        assert_eq!(last.bytes, last.total_bytes);
        let log = srv.take_log();
        assert!(
            log[0].starts_with("/v2/test/img/manifests/sha256:"),
            "{log:?}"
        );
        assert!(log[1].starts_with("/token?service=fake&scope="), "{log:?}");
        assert!(log.iter().any(|p| p.starts_with("/cdn/")), "{log:?}");
        assert!(!log.iter().any(|p| p.contains("unknown")), "{log:?}");
        // Cached: the root under its manifest digest, the blobs by digest.
        assert_eq!(
            files(&cache.0.join("roots")),
            [format!("{}.sqfs", &img.amd64_manifest[7..])]
        );
        let blobs = files(&cache.0.join("blobs/sha256"));
        for l in &img.amd64_layers {
            assert!(blobs.contains(&l[7..].to_string()), "{blobs:?}");
        }
        // The next pull asks nothing of the registry.
        check_root(
            pull(&reference, Arch::X86_64, &cache.0, PullOptions::default()).unwrap(),
            "amd64",
        );
        assert_eq!(srv.take_log(), Vec::<String>::new());
        // The other platform shares the index, and gets its own root.
        check_root(
            pull(&reference, Arch::Aarch64, &cache.0, PullOptions::default()).unwrap(),
            "arm64",
        );
        assert_eq!(files(&cache.0.join("roots")).len(), 2);
        assert!(
            !srv.take_log()
                .iter()
                .any(|p| p.contains("/manifests/sha256:")
                    && p.ends_with(&digest_of(&img.index)[7..]))
        );
        // A lost root is rebuilt from the cached blobs, without the network.
        std::fs::remove_dir_all(cache.0.join("roots")).unwrap();
        check_root(
            pull(&reference, Arch::X86_64, &cache.0, PullOptions::default()).unwrap(),
            "amd64",
        );
        assert_eq!(srv.take_log(), Vec::<String>::new());
    }

    #[test]
    fn a_tag_is_asked_for_every_time_but_its_root_is_reused() {
        let (srv, _tamper, by_digest) = http_registry();
        let reference = by_digest.split('@').next().unwrap().to_string() + ":1.0";
        let cache = TempDir::new("tag");
        check_root(
            pull(&reference, Arch::X86_64, &cache.0, PullOptions::default()).unwrap(),
            "amd64",
        );
        srv.take_log();
        check_root(
            pull(&reference, Arch::X86_64, &cache.0, PullOptions::default()).unwrap(),
            "amd64",
        );
        let log = srv.take_log();
        assert!(log.iter().any(|p| p.ends_with("/manifests/1.0")), "{log:?}");
        assert!(
            !log.iter().any(|p| p.starts_with("/cdn/")),
            "no blob fetched again: {log:?}"
        );
    }

    #[test]
    fn content_that_does_not_match_its_digest_is_refused_and_not_cached() {
        let (srv, tamper, reference) = http_registry();
        let img = image();
        let cache = TempDir::new("tamper");
        tamper.store(true, Ordering::Relaxed);
        let err = pull(&reference, Arch::X86_64, &cache.0, PullOptions::default()).unwrap_err();
        assert!(matches!(err, OciError::Digest { .. }), "{err}");
        let blobs = files(&cache.0.join("blobs/sha256"));
        for l in &img.amd64_layers {
            assert!(!blobs.contains(&l[7..].to_string()), "{blobs:?}");
        }
        assert!(
            !blobs.iter().any(|b| b.starts_with('.')),
            "no partial file left: {blobs:?}"
        );
        assert_eq!(files(&cache.0.join("roots")), Vec::<String>::new());
        // A reference whose digest the index doesn't have.
        let wrong = format!(
            "{}@sha256:{}",
            reference.split('@').next().unwrap(),
            "1".repeat(64)
        );
        assert!(pull(&wrong, Arch::X86_64, &cache.0, PullOptions::default()).is_err());
        // The registry serving the right bytes again: the pull completes.
        tamper.store(false, Ordering::Relaxed);
        srv.take_log();
        check_root(
            pull(&reference, Arch::X86_64, &cache.0, PullOptions::default()).unwrap(),
            "amd64",
        );
    }

    #[test]
    fn a_pull_over_https_trusts_what_it_is_told_to() {
        let (cert, key) = self_signed(&["registry.test", "cdn.registry.test"]);
        let tamper = Arc::new(AtomicBool::new(false));
        let (srv, slot) = serve_later(Some((cert.clone(), key)));
        let base = format!("https://registry.test:{}", srv.addr.port());
        let cdn = format!("https://cdn.registry.test:{}", srv.addr.port());
        let _ = slot.set(registry(image(), tamper, base, cdn));
        let reference = format!(
            "registry.test:{}/test/img@{}",
            srv.addr.port(),
            digest_of(&image().index)
        );
        let to: Arc<dyn Connector> = Arc::new(To(srv.addr));
        let cache = TempDir::new("https");
        let opts = PullOptions {
            connector: Some(to.clone()),
            ..PullOptions::default()
        };
        let err = pull(&reference, Arch::X86_64, &cache.0, opts).unwrap_err();
        assert!(matches!(err, OciError::Fetch(_)), "untrusted: {err}");
        std::fs::create_dir_all(&cache.0).unwrap();
        let ca = cache.0.join("ca.pem");
        std::fs::write(&ca, pem(&cert)).unwrap();
        let opts = PullOptions {
            connector: Some(to),
            ca_bundle: Some(ca.display().to_string()),
            ..PullOptions::default()
        };
        check_root(
            pull(&reference, Arch::X86_64, &cache.0, opts).unwrap(),
            "amd64",
        );
    }
}
