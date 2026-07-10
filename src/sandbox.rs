//! The public entry point: build a sandbox, then run a command in it.
//!
//! [`Sandbox::run`] wires the pipeline end to end:
//!
//! 1. resolve + open the guest root image ([`crate::image`]),
//! 2. assemble the [`MountTable`]: overlay(squashfs root) + `/work` passthrough
//!    + tmpfs/proc/dev ([`crate::fs`]),
//! 3. create the guest address space and load the target ELF ([`crate::loader`]),
//! 4. select an execution backend and create the first vcpu ([`crate::vcpu`]),
//! 5. hand the vcpu to the [`Kernel`] run/serve loop ([`crate::kernel`]).
//!
//! Most steps are stubs today; the loop is what later phases fill in. Calling
//! `run()` now walks the pipeline and returns the first not-yet-implemented
//! frontier, which is exactly the next thing the ROADMAP asks us to build.

use std::path::PathBuf;

use crate::abi::Arch;
use crate::fs::MountTable;
use crate::image::{ImageRef, ImageStore};
use crate::kernel::Kernel;
use crate::vcpu::{self, GuestMemory};
use crate::Error;

/// Default guest RAM ceiling: 512 MiB.
const DEFAULT_MEM_BYTES: u64 = 512 * 1024 * 1024;
/// Guest base address for the flat process address space (Phase 2 refines this).
const GUEST_BASE: u64 = 0x1_0000;

/// A fully-specified sandbox run.
#[derive(Debug, Clone)]
pub struct Config {
    /// argv of the command to run inside the sandbox.
    pub command: Vec<String>,
    /// Host directory mounted at `/work` (defaults to the current dir).
    pub work_dir: PathBuf,
    /// The guest root image.
    pub image: ImageRef,
    /// Guest memory ceiling in bytes.
    pub mem_bytes: u64,
    /// Guest architecture (defaults to the host's native arch).
    pub arch: Arch,
}

impl Config {
    fn guest_arch_default() -> Arch {
        Arch::host_native().unwrap_or(Arch::Aarch64)
    }
}

/// Builder for a [`Sandbox`].
#[derive(Debug, Clone)]
pub struct SandboxBuilder {
    config: Config,
}

impl Default for SandboxBuilder {
    fn default() -> Self {
        let arch = Config::guest_arch_default();
        Self {
            config: Config {
                command: Vec::new(),
                work_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
                image: ImageRef::default_for(arch),
                mem_bytes: DEFAULT_MEM_BYTES,
                arch,
            },
        }
    }
}

impl SandboxBuilder {
    /// Set the command (argv) to run in the sandbox.
    #[must_use]
    pub fn command<I, S>(mut self, argv: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.config.command = argv.into_iter().map(Into::into).collect();
        self
    }

    /// Set the host directory exposed as `/work`.
    #[must_use]
    pub fn work_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.config.work_dir = dir.into();
        self
    }

    /// Set the guest memory ceiling in bytes.
    #[must_use]
    pub fn mem_bytes(mut self, bytes: u64) -> Self {
        self.config.mem_bytes = bytes;
        self
    }

    /// Finish building.
    #[must_use]
    pub fn build(self) -> Sandbox {
        Sandbox {
            config: self.config,
        }
    }

    /// Build and run in one step; returns the guest exit code.
    pub fn run(self) -> Result<i32, Error> {
        self.build().run()
    }
}

/// A configured sandbox, ready to run a command.
#[derive(Debug, Clone)]
pub struct Sandbox {
    config: Config,
}

impl Sandbox {
    #[must_use]
    pub fn builder() -> SandboxBuilder {
        SandboxBuilder::default()
    }

    #[must_use]
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Run the configured command inside the sandbox; returns its exit code.
    pub fn run(&self) -> Result<i32, Error> {
        if self.config.command.is_empty() {
            return Err(Error::Config("no command specified".into()));
        }

        // 1. Resolve the root image (Phase 11 downloads; today: must be cached).
        let store = ImageStore::default_location();
        let _root_image = store.ensure(&self.config.image)?;

        // 2. Assemble the mount table. Phase 4 mounts overlay(squashfs)+passthrough.
        let mounts = self.build_mounts();

        // 3. Guest address space + ELF load (Phase 2).
        let mut mem = GuestMemory::new(GUEST_BASE, self.config.mem_bytes);

        // 4. Pick a backend and (Phase 1) create the first vcpu.
        let backend = vcpu::select(self.config.arch)?;
        let mut vcpu = backend.new_vcpu(/*entry=*/ 0, /*stack=*/ 0)?;

        // 5. Run/serve loop.
        let mut kernel = Kernel::new(self.config.arch, mounts);
        let code = kernel.run(vcpu.as_mut(), &mut mem)?;
        Ok(code)
    }

    /// Build the default mount layout. Backends are added in Phase 4; for now
    /// this is the empty table that later phases populate.
    #[allow(clippy::unused_self)] // will read self.config (work_dir, image) once backends land
    fn build_mounts(&self) -> MountTable {
        // TODO(Phase 4):
        //   mounts.mount("/",     Overlay::new(Squashfs::open(root)?, Tmpfs::new()));
        //   mounts.mount("/work", Passthrough::new(&self.config.work_dir));
        //   mounts.mount("/tmp",  Tmpfs::new());
        //   mounts.mount("/proc", ProcFs::new());
        //   mounts.mount("/dev",  DevFs::new());
        MountTable::new()
    }
}
