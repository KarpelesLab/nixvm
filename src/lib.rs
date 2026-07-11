//! nixvm — a portable, VM-style sandbox that runs a real Linux userland by
//! emulating Linux syscalls directly.
//!
//! There is no guest kernel and no device emulation. Guest processes run on the
//! host CPU (via Hypervisor.framework or KVM) or a software interpreter; when a
//! process makes a syscall the CPU traps out and [`kernel::Kernel`] services it
//! entirely in userspace, under nixvm's resource limits.
//!
//! # Module map
//!
//! * [`abi`]    — the Linux ABI as data: errno, per-arch syscall tables.
//! * [`vcpu`]   — execution backends (`hvf`, `interp`) behind one trait.
//! * [`loader`] — ELF loading, stack/auxv setup.
//! * [`fs`]     — the VFS mount table (squashfs, overlay, passthrough, …).
//! * [`kernel`] — the arch-agnostic syscall engine + process state.
//! * [`image`]  — guest root-image resolve/download/cache.
//! * [`sandbox`]— the public [`Sandbox`] builder that wires it all together.
//!
//! `unsafe` is confined to the hardware backend (`vcpu::hvf`); everything else
//! is safe Rust.
//!
//! ```no_run
//! use nixvm::Sandbox;
//!
//! // Run `npm install` in a sandbox with the cwd mounted at /work.
//! let status = Sandbox::builder()
//!     .command(["npm", "install"])
//!     .run()?;
//! std::process::exit(status);
//! # Ok::<(), nixvm::Error>(())
//! ```

pub mod abi;
pub mod fs;
pub mod image;
pub mod kernel;
pub mod loader;
pub mod sandbox;
pub mod vcpu;
pub mod vm;
#[cfg(feature = "wasm")]
pub mod wasm;

pub use abi::Arch;
pub use sandbox::{Config, Sandbox};

/// The crate's top-level error.
#[derive(Debug)]
pub enum Error {
    Vcpu(vcpu::VcpuError),
    Load(loader::LoadError),
    Image(image::ImageError),
    Config(String),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Vcpu(e) => write!(f, "{e}"),
            Self::Load(e) => write!(f, "{e}"),
            Self::Image(e) => write!(f, "{e}"),
            Self::Config(m) => write!(f, "configuration error: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<vcpu::VcpuError> for Error {
    fn from(e: vcpu::VcpuError) -> Self {
        Self::Vcpu(e)
    }
}
impl From<loader::LoadError> for Error {
    fn from(e: loader::LoadError) -> Self {
        Self::Load(e)
    }
}
impl From<image::ImageError> for Error {
    fn from(e: image::ImageError) -> Self {
        Self::Image(e)
    }
}
