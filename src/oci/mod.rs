//! Docker/OCI images, run without a daemon.
//!
//! An image comes from a registry ([`Pull`], or [`pull`] natively with the
//! `fetch` feature) or from an archive (`docker save` output or an OCI image
//! layout, [`Image::from_archive`]). Its layers are flattened into one root
//! filesystem ([`Image::rootfs_tar`]), which [`crate::vm::Vm::boot_image`] and
//! [`crate::Sandbox`] mount read-only under a writable layer and run the
//! image's entrypoint in, with its environment, working directory and user
//! ([`Image::launch`]).
//!
//! [`Pull`] does no I/O of its own: it says which URL to fetch next and is
//! handed the response. The same pull runs natively over a blocking HTTP
//! client and in a browser over `fetch()` — where, since registries send no
//! CORS headers, the embedder routes the requests through a proxy.

mod flatten;
pub mod json;
mod pull;
mod reference;

#[cfg(all(feature = "fetch", not(target_arch = "wasm32")))]
mod fetch;
#[cfg(feature = "oci-net")]
pub mod net;

#[cfg(all(feature = "fetch", not(target_arch = "wasm32")))]
pub use fetch::{pull, pull_with};
pub use flatten::flatten;
pub use pull::{Pull, Request, Response};
pub use reference::Reference;

use crate::abi::Arch;
use crate::fs::{MountTable, NodeKind};

/// What went wrong getting or running an image.
#[derive(Debug)]
pub enum OciError {
    /// The image name doesn't parse.
    Reference(String),
    /// The registry refused or failed a request.
    Registry(String),
    /// A manifest, index or config that isn't what the format says.
    Manifest(String),
    /// No image for the wanted platform.
    Platform(String),
    /// Content didn't match its digest.
    Digest { expected: String, got: String },
    /// A layer that isn't a readable (compressed) tar.
    Layer(String),
    /// Reading a local archive or cache.
    Io(std::io::Error),
    /// The network, or a build without a way to reach it.
    Fetch(String),
    /// The image can't be started as asked.
    Launch(String),
}

impl core::fmt::Display for OciError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Reference(m) => write!(f, "bad image name {m}"),
            Self::Registry(m) => write!(f, "registry: {m}"),
            Self::Manifest(m) => write!(f, "image manifest: {m}"),
            Self::Digest { expected, got } => {
                write!(
                    f,
                    "content doesn't match its digest: expected {expected}, got {got}"
                )
            }
            Self::Layer(m) => write!(f, "image layer: {m}"),
            Self::Io(e) => write!(f, "{e}"),
            Self::Platform(m) | Self::Fetch(m) | Self::Launch(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for OciError {}

impl From<std::io::Error> for OciError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// The largest a layer may decompress to: a guard against a decompression
/// bomb, well above any real image's.
#[cfg(target_arch = "wasm32")]
const MAX_LAYER_BYTES: u64 = 2 << 30;
#[cfg(not(target_arch = "wasm32"))]
const MAX_LAYER_BYTES: u64 = 64 << 30;

/// The architecture name images use for `arch`.
#[must_use]
pub fn platform_arch(arch: Arch) -> &'static str {
    match arch {
        Arch::X86_64 => "amd64",
        Arch::Aarch64 => "arm64",
    }
}

/// `sha256:` + the hex SHA-256 of `data`.
#[must_use]
pub fn digest_of(data: &[u8]) -> String {
    format!(
        "sha256:{}",
        crate::image::hex(&crate::image::sha256_bytes(data))
    )
}

/// Check `data` against `digest` (only `sha256:` digests are accepted).
///
/// # Errors
/// [`OciError::Digest`] on a mismatch or another algorithm.
pub fn verify(digest: &str, data: &[u8]) -> Result<(), OciError> {
    let got = digest_of(data);
    if got.eq_ignore_ascii_case(digest) {
        Ok(())
    } else {
        Err(OciError::Digest {
            expected: digest.to_string(),
            got,
        })
    }
}

/// Uncompress a layer blob: gzip and zstd by their magic, else a plain tar.
///
/// # Errors
/// [`OciError::Layer`] for a corrupt stream, or one that would uncompress past
/// a decompression-bomb guard (64 GiB natively, 2 GiB in the browser).
pub fn decompress(blob: &[u8]) -> Result<Vec<u8>, OciError> {
    if blob.starts_with(&[0x1f, 0x8b]) {
        compcol::vec::decompress_to_vec_capped::<compcol::gzip::Gzip>(blob, MAX_LAYER_BYTES)
            .map_err(|e| OciError::Layer(format!("gzip: {e:?}")))
    } else if blob.starts_with(&[0x28, 0xb5, 0x2f, 0xfd]) {
        compcol::vec::decompress_to_vec_capped::<compcol::zstd::Zstd>(blob, MAX_LAYER_BYTES)
            .map_err(|e| OciError::Layer(format!("zstd: {e:?}")))
    } else {
        Ok(blob.to_vec())
    }
}

/// The parts of an image's config that say how it runs.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ImageConfig {
    /// `amd64`, `arm64`, ….
    pub architecture: String,
    pub os: String,
    pub entrypoint: Vec<String>,
    pub cmd: Vec<String>,
    /// `KEY=VALUE` entries.
    pub env: Vec<String>,
    /// Empty means `/`.
    pub working_dir: String,
    /// `user`, `uid`, `user:group` or `uid:gid`; empty means root.
    pub user: String,
    /// `80/tcp`, ….
    pub exposed_ports: Vec<String>,
}

impl ImageConfig {
    /// Read an image config blob.
    ///
    /// # Errors
    /// [`OciError::Manifest`] if it isn't JSON.
    pub fn parse(blob: &[u8]) -> Result<Self, OciError> {
        let v = json::parse(blob).map_err(|e| OciError::Manifest(format!("image config: {e}")))?;
        let c = v.get("config").cloned().unwrap_or(json::Value::Null);
        let ports = match c.get("ExposedPorts") {
            Some(json::Value::Obj(m)) => m.iter().map(|(k, _)| k.clone()).collect(),
            _ => Vec::new(),
        };
        Ok(Self {
            architecture: v.str_at("architecture").unwrap_or_default().to_string(),
            os: v.str_at("os").unwrap_or_default().to_string(),
            entrypoint: c.strings_at("Entrypoint"),
            cmd: c.strings_at("Cmd"),
            env: c.strings_at("Env"),
            working_dir: c.str_at("WorkingDir").unwrap_or_default().to_string(),
            user: c.str_at("User").unwrap_or_default().to_string(),
            exposed_ports: ports,
        })
    }

    /// The guest architecture the image is built for, if nixvm runs it.
    #[must_use]
    pub fn arch(&self) -> Option<Arch> {
        match self.architecture.as_str() {
            "amd64" | "x86_64" => Some(Arch::X86_64),
            "arm64" | "aarch64" => Some(Arch::Aarch64),
            _ => None,
        }
    }
}

/// How to start an image, beyond what its config says: `docker run`'s knobs.
#[derive(Debug, Clone, Default)]
pub struct RunOptions {
    /// Replaces the image's entrypoint (`--entrypoint`), and its `Cmd` too
    /// unless [`RunOptions::cmd`] is given.
    pub entrypoint: Option<Vec<String>>,
    /// Replaces the image's `Cmd` when not empty.
    pub cmd: Vec<String>,
    /// `KEY=VALUE` entries added over the image's.
    pub env: Vec<String>,
    /// Replaces the image's user (`--user`).
    pub user: Option<String>,
    /// Replaces the image's working directory (`--workdir`).
    pub working_dir: Option<String>,
}

/// What [`Image::launch`] works out: the first process, as it starts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Launch {
    /// `argv[0]` is the program's path in the image.
    pub argv: Vec<String>,
    pub env: Vec<String>,
    pub cwd: String,
    pub uid: u32,
    pub gid: u32,
    /// Supplementary groups: those `/etc/group` lists the user in.
    pub groups: Vec<u32>,
}

/// A pulled or loaded image: its config and its layers, uncompressed.
#[derive(Debug, Clone)]
pub struct Image {
    pub config: ImageConfig,
    /// Bottom first.
    pub layers: Vec<Vec<u8>>,
    /// The manifest's digest, when known.
    pub digest: Option<String>,
}

impl Image {
    /// The root filesystem: the layers flattened into one tar.
    ///
    /// # Errors
    /// [`OciError::Layer`] for a layer that isn't a readable tar.
    pub fn rootfs_tar(&self) -> Result<Vec<u8>, OciError> {
        flatten(&self.layers)
    }

    /// Load an image from an archive: `docker save` output (optionally
    /// gzipped) or an OCI image layout as a tar. Where it holds several
    /// platforms, `arch`'s is taken.
    ///
    /// # Errors
    /// What's missing or malformed in the archive.
    pub fn from_archive(bytes: &[u8], arch: Arch) -> Result<Self, OciError> {
        let tar = decompress(bytes)?;
        let files = flatten::files(&tar)?;
        load(
            &|p: &str| files.get(p.trim_start_matches("./")).map(|d| d.to_vec()),
            arch,
        )
    }

    /// Load an image from an OCI image layout directory (or an unpacked
    /// `docker save`).
    ///
    /// # Errors
    /// As [`Image::from_archive`].
    pub fn from_dir(dir: &std::path::Path, arch: Arch) -> Result<Self, OciError> {
        load(&|p: &str| std::fs::read(dir.join(p)).ok(), arch)
    }

    /// Work out the first process: argv (`argv[0]` found on the image's
    /// `PATH` when it has no `/`), environment, working directory and ids,
    /// looking names up in the image's `/etc/passwd` and `/etc/group`, read
    /// from `root`.
    ///
    /// # Errors
    /// [`OciError::Launch`] when there's nothing to run, the program isn't in
    /// the image, or the user or group isn't known.
    pub fn launch(&self, opts: &RunOptions, root: &mut MountTable) -> Result<Launch, OciError> {
        let c = &self.config;
        let entry = opts
            .entrypoint
            .clone()
            .unwrap_or_else(|| c.entrypoint.clone());
        let args = if !opts.cmd.is_empty() {
            opts.cmd.clone()
        } else if opts.entrypoint.is_some() {
            Vec::new()
        } else {
            c.cmd.clone()
        };
        let mut argv: Vec<String> = entry.into_iter().chain(args).collect();
        if argv.is_empty() {
            return Err(OciError::Launch(
                "the image names no command to run (no Entrypoint or Cmd): give one".into(),
            ));
        }
        let mut env = c.env.clone();
        let set = |env: &mut Vec<String>, kv: &str| {
            let key = kv.split('=').next().unwrap_or(kv);
            env.retain(|e| e.split('=').next() != Some(key));
            env.push(kv.to_string());
        };
        for kv in &opts.env {
            set(&mut env, kv);
        }
        let has = |env: &[String], key: &str| env.iter().any(|e| e.split('=').next() == Some(key));
        if !has(&env, "PATH") {
            env.insert(
                0,
                "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin".into(),
            );
        }
        if !has(&env, "HOSTNAME") {
            env.push("HOSTNAME=nixvm".into());
        }
        let user = opts.user.clone().unwrap_or_else(|| c.user.clone());
        let (uid, gid, home, groups) = resolve_user(&user, root)?;
        if !has(&env, "HOME") {
            env.push(format!("HOME={home}"));
        }
        let cwd = opts
            .working_dir
            .clone()
            .unwrap_or_else(|| c.working_dir.clone());
        let cwd = if cwd.is_empty() { "/".to_string() } else { cwd };
        if !argv[0].contains('/') {
            let path = env
                .iter()
                .find_map(|e| e.strip_prefix("PATH="))
                .unwrap_or("")
                .to_string();
            let found = path.split(':').filter(|d| !d.is_empty()).find_map(|d| {
                let p = format!("{}/{}", d.trim_end_matches('/'), argv[0]);
                let p = if p.starts_with('/') {
                    p
                } else {
                    format!("{cwd}/{p}")
                };
                let target = root.follow(&p)?;
                root.stat(&target)
                    .filter(|a| a.kind == NodeKind::File && a.mode & 0o111 != 0)
                    .map(|_| p)
            });
            match found {
                Some(p) => argv[0] = p,
                None => {
                    return Err(OciError::Launch(format!(
                        "{}: not found on the image's PATH ({path})",
                        argv[0]
                    )));
                }
            }
        } else if !argv[0].starts_with('/') {
            argv[0] = crate::kernel::path::normalize(&format!("{cwd}/{}", argv[0]));
        }
        Ok(Launch {
            argv,
            env,
            cwd,
            uid,
            gid,
            groups,
        })
    }
}

/// `user[:group]` → (uid, gid, home, supplementary groups), names looked up
/// in the image's `/etc/passwd` and `/etc/group`. A numeric uid needn't be
/// listed; its gid is then its passwd entry's, or 0.
fn resolve_user(
    spec: &str,
    root: &mut MountTable,
) -> Result<(u32, u32, String, Vec<u32>), OciError> {
    let read = |root: &mut MountTable, p: &str| {
        root.read_file(p)
            .map(|b| String::from_utf8_lossy(&b).into_owned())
            .unwrap_or_default()
    };
    let passwd = read(root, "/etc/passwd");
    let group = read(root, "/etc/group");
    let (u, g) = match spec.split_once(':') {
        Some((u, g)) => (u, Some(g)),
        None => (spec, None),
    };
    let u = if u.is_empty() { "0" } else { u };
    // name:x:uid:gid:gecos:home:shell
    let entry = passwd
        .lines()
        .map(|l| l.split(':').collect::<Vec<_>>())
        .find(|f| f.len() >= 4 && (f[0] == u || (u.parse::<u32>().is_ok() && f[2] == u)));
    let uid = match (u.parse::<u32>(), &entry) {
        (Ok(n), _) => n,
        (Err(_), Some(f)) => f[2].parse().unwrap_or(0),
        (Err(_), None) => {
            return Err(OciError::Launch(format!(
                "user {u:?} isn't in the image's /etc/passwd"
            )));
        }
    };
    let name = entry.as_ref().map(|f| f[0].to_string());
    let home = entry
        .as_ref()
        .and_then(|f| f.get(5).filter(|h| !h.is_empty()).map(|h| (*h).to_string()))
        .unwrap_or_else(|| if uid == 0 { "/root".into() } else { "/".into() });
    let primary = entry.as_ref().and_then(|f| f[3].parse().ok()).unwrap_or(0);
    let gid = match g {
        None | Some("") => primary,
        Some(g) => match g.parse::<u32>() {
            Ok(n) => n,
            Err(_) => group
                .lines()
                .map(|l| l.split(':').collect::<Vec<_>>())
                .find(|f| f.len() >= 3 && f[0] == g)
                .and_then(|f| f[2].parse().ok())
                .ok_or_else(|| {
                    OciError::Launch(format!("group {g:?} isn't in the image's /etc/group"))
                })?,
        },
    };
    // name:x:gid:member,member
    let mut groups: Vec<u32> = match &name {
        Some(n) => group
            .lines()
            .map(|l| l.split(':').collect::<Vec<_>>())
            .filter(|f| f.len() >= 4 && f[3].split(',').any(|m| m == n))
            .filter_map(|f| f[2].parse().ok())
            .collect(),
        None => Vec::new(),
    };
    if !groups.contains(&gid) {
        groups.insert(0, gid);
    }
    Ok((uid, gid, home, groups))
}

/// Read an image through `read` (a path in the archive → its bytes).
fn load(read: &dyn Fn(&str) -> Option<Vec<u8>>, arch: Arch) -> Result<Image, OciError> {
    let need =
        |p: &str| read(p).ok_or_else(|| OciError::Manifest(format!("the archive has no {p}")));
    let blob = |digest: &str| -> Result<Vec<u8>, OciError> {
        let (algo, hex) = digest
            .split_once(':')
            .ok_or_else(|| OciError::Manifest(format!("bad digest {digest:?}")))?;
        let data = need(&format!("blobs/{algo}/{hex}"))?;
        verify(digest, &data)?;
        Ok(data)
    };
    // `docker save`'s own manifest names the files directly.
    if let Some(m) = read("manifest.json") {
        let v = json::parse(&m).map_err(|e| OciError::Manifest(format!("manifest.json: {e}")))?;
        let first = v
            .as_array()
            .first()
            .ok_or_else(|| OciError::Manifest("manifest.json lists no image".into()))?;
        let config = ImageConfig::parse(&need(first.str_at("Config").unwrap_or_default())?)?;
        check_arch(&config, arch)?;
        let layers = first
            .strings_at("Layers")
            .iter()
            .map(|p| need(p).and_then(|b| decompress(&b)))
            .collect::<Result<_, _>>()?;
        return Ok(Image {
            config,
            layers,
            digest: None,
        });
    }
    // An OCI image layout: index.json → (index →)* manifest → blobs.
    let index = need("index.json")?;
    let mut doc =
        json::parse(&index).map_err(|e| OciError::Manifest(format!("index.json: {e}")))?;
    let mut digest = None;
    for _ in 0..8 {
        if doc.get("manifests").is_some() {
            let d = pull::pick_manifest(&doc, arch)?;
            let data = blob(&d)?;
            doc = json::parse(&data).map_err(|e| OciError::Manifest(format!("{d}: {e}")))?;
            digest = Some(d);
            continue;
        }
        let config_digest = doc
            .get("config")
            .and_then(|c| c.str_at("digest"))
            .ok_or_else(|| OciError::Manifest("a manifest without a config".into()))?;
        let config = ImageConfig::parse(&blob(config_digest)?)?;
        check_arch(&config, arch)?;
        let layers = doc
            .get("layers")
            .map(|l| l.as_array().to_vec())
            .unwrap_or_default()
            .iter()
            .map(|l| {
                let d = l.str_at("digest").unwrap_or_default();
                blob(d).and_then(|b| decompress(&b))
            })
            .collect::<Result<_, _>>()?;
        return Ok(Image {
            config,
            layers,
            digest,
        });
    }
    Err(OciError::Manifest("index.json nests too deep".into()))
}

/// An image built for another architecture can't run as `arch`.
fn check_arch(config: &ImageConfig, arch: Arch) -> Result<(), OciError> {
    match config.arch() {
        Some(a) if a != arch => Err(OciError::Platform(format!(
            "the image is for {}, not {}",
            config.architecture,
            platform_arch(arch)
        ))),
        None if !config.architecture.is_empty() => Err(OciError::Platform(format!(
            "the image is for {}, which nixvm doesn't run",
            config.architecture
        ))),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::flatten::tests::{entry, tar};
    use super::*;
    use crate::fs::TmpFs;

    fn config(entrypoint: &str, cmd: &str, user: &str) -> Vec<u8> {
        format!(
            r#"{{"architecture":"amd64","os":"linux","config":{{"Env":["PATH=/usr/bin:/bin","A=1"],
            "Entrypoint":{entrypoint},"Cmd":{cmd},"WorkingDir":"/srv","User":"{user}",
            "ExposedPorts":{{"80/tcp":{{}}}}}}}}"#
        )
        .into_bytes()
    }

    fn rootfs() -> Vec<u8> {
        tar(&[
            entry("etc/", b'5', "", b""),
            entry(
                "etc/passwd",
                b'0',
                "",
                b"root:x:0:0:root:/root:/bin/sh\napp:x:1000:1000::/home/app:/bin/sh\n",
            ),
            entry(
                "etc/group",
                b'0',
                "",
                b"root:x:0:\napp:x:1000:\nwheel:x:10:app\n",
            ),
            entry("bin/", b'5', "", b""),
            entry("bin/sh", b'0', "", b"\x7fELF"),
            entry("usr/", b'5', "", b""),
            entry("usr/bin/", b'5', "", b""),
            entry("usr/bin/env", b'2', "/bin/sh", b""),
        ])
    }

    fn root_of(img: &Image) -> MountTable {
        let mut t = MountTable::new();
        t.mount("/", Box::new(TmpFs::new()));
        crate::fs::tar::extract_into(&mut t, &img.rootfs_tar().unwrap());
        t
    }

    #[test]
    fn a_docker_save_archive_loads_and_launches_as_docker_would() {
        let cfg = config(r#"["env"]"#, r#"["sh","-c","x"]"#, "app");
        let layer = rootfs();
        let archive = tar(&[
            entry(
                "manifest.json",
                b'0',
                "",
                br#"[{"Config":"cfg.json","RepoTags":["t:1"],"Layers":["l1/layer.tar"]}]"#,
            ),
            entry("cfg.json", b'0', "", &cfg),
            entry("l1/layer.tar", b'0', "", &layer),
        ]);
        let img = Image::from_archive(&archive, Arch::X86_64).unwrap();
        assert_eq!(img.config.exposed_ports, vec!["80/tcp"]);
        assert!(Image::from_archive(&archive, Arch::Aarch64).is_err());
        let mut root = root_of(&img);
        let l = img.launch(&RunOptions::default(), &mut root).unwrap();
        assert_eq!(l.argv, vec!["/usr/bin/env", "sh", "-c", "x"]);
        assert_eq!((l.uid, l.gid, l.cwd.as_str()), (1000, 1000, "/srv"));
        assert_eq!(l.groups, vec![1000, 10]);
        assert!(l.env.contains(&"HOME=/home/app".to_string()));
        assert!(l.env.contains(&"A=1".to_string()));
        // docker run IMAGE args: Cmd replaced, Entrypoint kept.
        let o = RunOptions {
            cmd: vec!["ls".into()],
            env: vec!["A=2".into()],
            user: Some("0:wheel".into()),
            ..RunOptions::default()
        };
        let l = img.launch(&o, &mut root).unwrap();
        assert_eq!(l.argv, vec!["/usr/bin/env", "ls"]);
        assert_eq!((l.uid, l.gid), (0, 10));
        assert!(l.env.contains(&"A=2".to_string()) && !l.env.contains(&"A=1".to_string()));
        // --entrypoint drops the image's Cmd.
        let o = RunOptions {
            entrypoint: Some(vec!["sh".into()]),
            ..RunOptions::default()
        };
        assert_eq!(img.launch(&o, &mut root).unwrap().argv, vec!["/bin/sh"]);
        let o = RunOptions {
            entrypoint: Some(vec!["nope".into()]),
            ..RunOptions::default()
        };
        assert!(img.launch(&o, &mut root).is_err());
        let o = RunOptions {
            user: Some("nobody".into()),
            ..RunOptions::default()
        };
        assert!(img.launch(&o, &mut root).is_err());
    }

    #[test]
    fn an_oci_layout_picks_the_wanted_platform_and_checks_digests() {
        let layer = rootfs();
        let arm_cfg = String::from_utf8(config("null", r#"["sh"]"#, ""))
            .unwrap()
            .replace("amd64", "arm64")
            .into_bytes();
        let amd_cfg = config("null", r#"["sh"]"#, "");
        let manifest = |cfg: &[u8]| {
            format!(
                r#"{{"schemaVersion":2,"mediaType":"application/vnd.oci.image.manifest.v1+json",
                "config":{{"digest":"{}"}},"layers":[{{"digest":"{}"}}]}}"#,
                digest_of(cfg),
                digest_of(&layer)
            )
            .into_bytes()
        };
        let (m_arm, m_amd) = (manifest(&arm_cfg), manifest(&amd_cfg));
        let index = format!(
            r#"{{"schemaVersion":2,"manifests":[
              {{"digest":"{}","platform":{{"architecture":"arm64","os":"linux"}}}},
              {{"digest":"{}","platform":{{"architecture":"amd64","os":"linux"}}}}]}}"#,
            digest_of(&m_arm),
            digest_of(&m_amd)
        );
        let b = |d: &[u8]| format!("blobs/sha256/{}", &digest_of(d)[7..]);
        let mut files = vec![
            entry("oci-layout", b'0', "", br#"{"imageLayoutVersion":"1.0.0"}"#),
            entry("index.json", b'0', "", index.as_bytes()),
        ];
        for blob in [&arm_cfg, &amd_cfg, &m_arm, &m_amd, &layer] {
            files.push(entry(&b(blob), b'0', "", blob));
        }
        let archive = tar(&files);
        let img = Image::from_archive(&archive, Arch::Aarch64).unwrap();
        assert_eq!(img.config.architecture, "arm64");
        assert_eq!(img.digest.as_deref(), Some(digest_of(&m_arm).as_str()));
        let img = Image::from_archive(&archive, Arch::X86_64).unwrap();
        assert_eq!(img.config.architecture, "amd64");
        let mut root = root_of(&img);
        assert_eq!(
            img.launch(&RunOptions::default(), &mut root).unwrap().argv,
            vec!["/bin/sh"]
        );
        // A blob that doesn't match its digest is refused.
        let mut bad = files.clone();
        bad.pop();
        bad.push(entry(&b(&layer), b'0', "", b"tampered"));
        assert!(matches!(
            Image::from_archive(&tar(&bad), Arch::X86_64),
            Err(OciError::Digest { .. })
        ));
    }
}
