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

use crate::Error;
use crate::abi::Arch;
use crate::fs::{DevFs, MountTable, TmpFs};
#[cfg(unix)]
use crate::fs::Passthrough;
use crate::image::{ImageRef, ImageStore};
use crate::kernel::Kernel;
use crate::loader::{ProcessSpec, load_static};
use crate::vcpu::interp::InterpBackend;
use crate::vcpu::mem::PAGE_SIZE;
use crate::vcpu::{self, Backend, GuestMemory};

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
    /// Force the software interpreter instead of the best hardware backend.
    /// Used by CI, the browser (wasm) target, and for portability.
    pub prefer_interp: bool,
    /// Extra host directories shared into the sandbox ("holes"): specific,
    /// chosen paths — never the whole home.
    pub binds: Vec<Bind>,
}

/// A host directory shared into the sandbox at a guest mount point.
#[derive(Debug, Clone)]
pub struct Bind {
    pub host: PathBuf,
    pub guest: String,
    pub read_only: bool,
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
                prefer_interp: false,
                binds: Vec::new(),
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

    /// Force the software interpreter backend (portable / wasm / CI).
    #[must_use]
    pub fn prefer_interp(mut self, yes: bool) -> Self {
        self.config.prefer_interp = yes;
        self
    }

    /// Share a specific host directory into the sandbox at `guest`, read-write.
    #[must_use]
    pub fn bind(mut self, host: impl Into<PathBuf>, guest: impl Into<String>) -> Self {
        self.config.binds.push(Bind {
            host: host.into(),
            guest: guest.into(),
            read_only: false,
        });
        self
    }

    /// Share a specific host directory into the sandbox at `guest`, read-only.
    #[must_use]
    pub fn bind_ro(mut self, host: impl Into<PathBuf>, guest: impl Into<String>) -> Self {
        self.config.binds.push(Bind {
            host: host.into(),
            guest: guest.into(),
            read_only: true,
        });
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
        let mem = GuestMemory::new(GUEST_BASE, self.config.mem_bytes);

        // 4. Pick a backend and (Phase 1) create the first vcpu.
        let backend = vcpu::select(self.config.arch)?;
        let vcpu = backend.new_vcpu(/*entry=*/ 0, /*stack=*/ 0)?;

        // 5. Run/serve loop.
        let mut kernel = Kernel::new(self.config.arch, mounts);
        let code = kernel.run(vcpu, mem)?;
        Ok(code)
    }

    /// Load and run a statically-linked ELF64 image, returning its exit code.
    ///
    /// This is the full pipeline — loader → backend → kernel run/serve loop —
    /// that [`Sandbox::run`] will call once image/filesystem resolution reads
    /// the target binary out of the guest root. Exposed now so it's testable
    /// and embeddable ahead of that.
    pub fn exec_elf(&self, elf: &[u8]) -> Result<i32, Error> {
        let arch = self.config.arch;
        let mut mem = GuestMemory::new(GUEST_BASE, round_up_page(self.config.mem_bytes));

        let argv = if self.config.command.is_empty() {
            vec!["prog".to_string()]
        } else {
            self.config.command.clone()
        };
        let spec = ProcessSpec {
            argv,
            envp: default_env(),
        };
        let img = load_static(&mut mem, elf, &spec)?;

        // Lay out heap and mmap in the gap between the image and the stack: the
        // heap grows up from the program break, mmap grows down from the stack,
        // meeting at a midpoint so the two arenas can't collide.
        let mid =
            page_align_down(img.program_break + (img.stack_bottom - img.program_break) / 2);

        let backend = self.backend()?;
        let vcpu = backend.new_vcpu(img.entry, img.stack_pointer)?;

        let mut kernel = Kernel::new(arch, self.build_mounts());
        kernel.set_cwd("/work");
        kernel.set_heap(img.program_break, mid);
        kernel.set_mmap_area(img.stack_bottom, mid);
        Ok(kernel.run(vcpu, mem)?)
    }

    /// Select the execution backend per config: the interpreter when forced,
    /// otherwise the best hardware backend for the host.
    fn backend(&self) -> Result<Box<dyn Backend>, Error> {
        if self.config.prefer_interp {
            Ok(Box::new(InterpBackend::new(self.config.arch)?))
        } else {
            Ok(vcpu::select(self.config.arch)?)
        }
    }

    /// Assemble the sandbox mount layout.
    ///
    /// `/` is a writable in-memory root today; once image resolution lands it
    /// becomes `Overlay::new(squashfs_lower, tmpfs_upper)` so many instances
    /// share one read-only squashfs. `/tmp` is its own tmpfs. `/work` and any
    /// configured binds are host passthroughs ("holes"). Synthetic `/proc`,
    /// `/sys`, `/dev` arrive in Phase 7.
    fn build_mounts(&self) -> MountTable {
        let mut mounts = MountTable::new();
        mounts.mount("/", Box::new(TmpFs::new()));
        mounts.mount("/tmp", Box::new(TmpFs::new()));
        mounts.mount("/dev", Box::new(DevFs::new()));

        #[cfg(unix)]
        {
            mounts.mount("/work", Box::new(Passthrough::new(self.config.work_dir.clone())));
            for b in &self.config.binds {
                let pt = if b.read_only {
                    Passthrough::read_only(b.host.clone())
                } else {
                    Passthrough::new(b.host.clone())
                };
                mounts.mount(b.guest.clone(), Box::new(pt));
            }
        }
        mounts
    }
}

/// A minimal default environment for the guest.
fn default_env() -> Vec<String> {
    vec![
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin".into(),
        "HOME=/root".into(),
        "TERM=xterm".into(),
        "PWD=/work".into(),
    ]
}

fn round_up_page(v: u64) -> u64 {
    v.div_ceil(PAGE_SIZE) * PAGE_SIZE
}

fn page_align_down(v: u64) -> u64 {
    v - v % PAGE_SIZE
}
