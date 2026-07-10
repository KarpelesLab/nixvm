//! Guest root images: resolve a name → a local squashfs file, downloading and
//! verifying it into a content-addressed cache on first use.
//!
//! Default image is a minimal Alpine root as a squashfs. Images are identified
//! by `(distro, version, arch)` and pinned by digest so runs are reproducible
//! and the download is verified.
//!
//! Fetching lands in Phase 11; the resolver API is defined here.

use std::path::PathBuf;

use crate::abi::Arch;

/// A request for a guest root image.
#[derive(Debug, Clone)]
pub struct ImageRef {
    pub distro: String,
    pub version: String,
    pub arch: Arch,
}

impl ImageRef {
    /// The default shipped image for `arch`: minimal Alpine.
    #[must_use]
    pub fn default_for(arch: Arch) -> Self {
        Self {
            distro: "alpine".into(),
            version: "3.20".into(),
            arch,
        }
    }

    /// Cache-relative filename, e.g. `alpine-3.20-aarch64.sqfs`.
    #[must_use]
    pub fn file_name(&self) -> String {
        format!("{}-{}-{}.sqfs", self.distro, self.version, self.arch.as_str())
    }
}

#[derive(Debug)]
pub enum ImageError {
    NotCached(ImageRef),
    Unimplemented(&'static str),
}

impl core::fmt::Display for ImageError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotCached(r) => write!(f, "image {} not in cache", r.file_name()),
            Self::Unimplemented(w) => write!(f, "image store: {w} not implemented"),
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

    /// Ensure `image` is present locally, downloading if needed; return its path.
    ///
    /// Stub: Phase 11 implements fetch + digest verification.
    pub fn ensure(&self, image: &ImageRef) -> Result<PathBuf, ImageError> {
        let path = self.path_for(image);
        if path.exists() {
            return Ok(path);
        }
        Err(ImageError::Unimplemented("image download (Phase 11)"))
    }
}
