//! The public entry point: build a sandbox, then run a command in it.
//!
//! Two ways to run:
//!
//! * [`Sandbox::run`] is the `nixvm run` command line's: the guest gets the
//!   host's terminal as its stdio, the host's working directory at `/work`,
//!   starts in `/root`, kernel notices go to stderr, and the result is the
//!   exit code.
//! * [`Sandbox::execute`] is for embedding: the guest's stdio is what the
//!   builder was given ([`SandboxBuilder::stdin`], [`SandboxBuilder::stdout`],
//!   [`SandboxBuilder::stderr`]; nothing in, output discarded otherwise),
//!   nothing is printed, and the result is a [`RunOutcome`] — exit status,
//!   the limit that ended the run if one did, unsupported syscalls, the
//!   kernel's notices.
//!
//! Both honour the same configuration: the working directory
//! ([`SandboxBuilder::cwd`]), binds and filesystems at arbitrary guest paths
//! ([`SandboxBuilder::bind`], [`SandboxBuilder::mount`]) with or without the
//! implicit `/work` ([`SandboxBuilder::no_work_dir`]), the network
//! ([`SandboxBuilder::network`]), limits, the CPU backend.
//!
//! ```no_run
//! use nixvm::sandbox::{Capture, Network, Sandbox};
//!
//! let out = Capture::new();
//! let outcome = Sandbox::builder()
//!     .root_dir("/srv/alpine")
//!     .no_work_dir()
//!     .bind("/home/me/project", "/project")
//!     .cwd("/project")
//!     .network(Network::None)
//!     .wall_time(std::time::Duration::from_secs(30))
//!     .stdout(out.writer())
//!     .shell("echo hi; pwd")
//!     .execute()?;
//! assert_eq!(out.text(), "hi\n/project\n");
//! assert!(outcome.success());
//! # Ok::<(), nixvm::Error>(())
//! ```
//!
//! Either wires the pipeline end to end:
//!
//! 1. resolve + open the guest root image ([`crate::image`]),
//! 2. assemble the [`MountTable`]: overlay(squashfs root) + `/work` passthrough
//!    + tmpfs/proc/dev ([`crate::fs`]),
//! 3. create the guest address space and load the target ELF ([`crate::loader`]),
//! 4. select an execution backend and create the first vcpu ([`crate::vcpu`]),
//! 5. hand the vcpu to the [`Kernel`] run/serve loop ([`crate::kernel`]).
//!
//! Nothing here reads the process environment: the `NIXVM_*` switches come
//! from the settings installed in [`crate::env`] (none, unless the embedder
//! installs some), and everything else from the builder.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

mod embed;
pub use embed::{
    Capture, EgressFactory, FsFactory, LimitHit, Mount, Network, RunOutcome, Unsupported,
};
use embed::{FromShared, Io, ToShared};

use crate::Error;
use crate::abi::Arch;
#[cfg(any(unix, nixvm_linux))]
use crate::fs::Passthrough;
use crate::fs::{DevFs, MountFs, MountTable, Overlay, ProcFs, SysFs, TmpFs};
use crate::image::{ImageRef, ImageStore};
use crate::kernel::Kernel;
use crate::kernel::notice::{Notice, NoticeSink};
use crate::loader::{ProcessSpec, interp_path, load_dynamic, load_static};
use crate::vcpu::mem::PAGE_SIZE;
use crate::vcpu::{self, Backend, GuestMemory};

/// How the container comes up.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RunMode {
    /// The shell (or the given command) is PID 1. Network, `/etc`, and the
    /// environment are preconfigured host-side; there is no init and no service
    /// manager. The quickest way in.
    #[default]
    Nude,
    /// PID 1 is an init that runs the boot sequence (mounts, hostname, bring up
    /// loopback, then a login shell). Falls back to the shell if the root has no
    /// runnable init. (In-guest OpenRC/service scripts await dynamic linking; the
    /// boot-time host preconfiguration is applied in both modes today.)
    Booted,
}

/// Where the guest root filesystem (the overlay's read-only lower layer) comes
/// from.
#[derive(Debug, Clone, Default)]
pub enum RootSource {
    /// No root image: `/` is a bare writable tmpfs.
    #[default]
    Empty,
    /// A host directory used read-only as the lower layer (works without the
    /// `fstool` feature; the realistic "point me at an extracted rootfs" path).
    Dir(PathBuf),
    /// A host directory used directly as `/`, read-write: what the guest
    /// changes is changed on the host (to install packages into a root).
    DirWritable(PathBuf),
    /// A squashfs image file used read-only as the lower layer (needs `fstool`).
    Squashfs(PathBuf),
    /// A named image resolved through the [`ImageStore`] (squashfs; needs
    /// `fstool` and a cached image).
    Image(ImageRef),
    /// A Docker/OCI image (needs the `oci` feature): a registry name
    /// (`alpine:3.20`, `ghcr.io/o/i@sha256:…`, pulled with the `fetch`
    /// feature, blobs cached), or the path of a `docker save` archive or an
    /// OCI image layout (a tar or a directory). It's run as `docker run`
    /// would: [`Config::command`], when given, replaces its `Cmd`.
    Oci(String),
}

/// Default guest RAM ceiling: 8 GiB, reduced on a host without that much to spare
/// (see [`crate::limits::guest_memory_ceiling`]). The address space's layout is
/// [`GuestMemory::with_default_layout`]'s.
const DEFAULT_MEM_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// A fully-specified sandbox run.
#[derive(Debug, Clone)]
pub struct Config {
    /// argv of the command to run inside the sandbox.
    pub command: Vec<String>,
    /// Host directory mounted at `/work` (defaults to the current dir), unless
    /// [`Config::mount_work`] is off.
    pub work_dir: PathBuf,
    /// Whether [`Config::work_dir`] is mounted at `/work` (the default).
    pub mount_work: bool,
    /// The guest working directory of the first process; `None` is `/root`
    /// (`/` when booted, the image's own for an image). Made if missing.
    pub cwd: Option<String>,
    /// The guest root image (used when `root` is [`RootSource::Image`]).
    pub image: ImageRef,
    /// Where the read-only root filesystem comes from.
    pub root: RootSource,
    /// How the container boots.
    pub mode: RunMode,
    /// Explicit init binary for [`RunMode::Booted`] (default `/sbin/init`).
    pub init: Option<String>,
    /// Number of virtual CPUs (host worker threads); `1` is single-threaded.
    pub ncpus: usize,
    /// Guest memory ceiling in bytes.
    pub mem_bytes: u64,
    /// VM-wide resource ceilings.
    pub limits: crate::limits::Limits,
    /// Guest architecture (defaults to the host's native arch). Unless set
    /// explicitly ([`Config::arch_explicit`]), the PID-1 binary's ELF header
    /// decides: an x86-64 root runs as x86-64 on any host.
    pub arch: Arch,
    /// Whether [`Config::arch`] was chosen explicitly
    /// ([`SandboxBuilder::arch`]); then a PID-1 binary for another
    /// architecture is an error instead of switching the guest arch.
    pub arch_explicit: bool,
    /// Force the software interpreter instead of the best hardware backend.
    /// Used by CI, the browser (wasm) target, and for portability.
    pub prefer_interp: bool,
    /// Extra host directories shared into the sandbox ("holes"): specific,
    /// chosen paths — never the whole home.
    pub binds: Vec<Bind>,
    /// Filesystems of the embedder's, at guest paths (after the binds).
    pub mounts: Vec<Mount>,
    /// What the guest's sockets can reach; `None` leaves it to the installed
    /// `NIXVM_NET` setting ([`crate::env`]), which is loopback only unless
    /// set.
    pub network: Option<Network>,
    /// Extra `KEY=VALUE` entries appended to the guest environment, overriding
    /// the preconfigured defaults on a key collision.
    pub env: Vec<String>,
    /// Syscalls refused to every guest task; `None` leaves the kernel's own
    /// default (`NIXVM_DENY_SYSCALLS`).
    pub syscall_policy: Option<crate::policy::SyscallPolicy>,
    /// For an image root ([`RootSource::Oci`]): replaces its entrypoint.
    pub entrypoint: Option<Vec<String>>,
    /// The user PID 1 runs as (`user`, `uid`, `user:group`), looked up in the
    /// root's `/etc/passwd`; for an image root it replaces the image's user.
    pub user: Option<String>,
    /// The built-in Wayland display the guest gets, if any (see
    /// [`SandboxBuilder::display`]).
    pub display: Option<crate::display::Display>,
    /// The virtual GPU the guest gets (see [`SandboxBuilder::gpu`]); `None`
    /// leaves it to `NIXVM_GPU`.
    #[cfg(feature = "gpu")]
    pub gpu: Option<crate::gpu::GpuDevice>,
    /// Stdio and the notice sink (see [`SandboxBuilder::stdout`] and co).
    pub(crate) io: Io,
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
                mount_work: true,
                cwd: None,
                image: ImageRef::default_for(arch),
                root: RootSource::Empty,
                mode: RunMode::Nude,
                init: None,
                // NIXVM_CPUS=N runs guest compute on N host worker threads (the
                // SMP scheduler); unset or unparseable means 1 (serial). An
                // explicit `.ncpus()` on the builder still overrides this.
                ncpus: crate::env::var("NIXVM_CPUS")
                    .and_then(|s| s.trim().parse::<usize>().ok())
                    .map_or(1, |n| n.max(1)),
                mem_bytes: DEFAULT_MEM_BYTES,
                limits: crate::limits::Limits::for_this_host(),
                arch,
                arch_explicit: false,
                prefer_interp: false,
                binds: Vec::new(),
                mounts: Vec::new(),
                network: None,
                env: Vec::new(),
                syscall_policy: None,
                entrypoint: None,
                user: None,
                display: None,
                #[cfg(feature = "gpu")]
                gpu: None,
                io: Io::default(),
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

    /// Run `script` with `/bin/sh -c` (sets the [command](Self::command)).
    #[must_use]
    pub fn shell(self, script: impl Into<String>) -> Self {
        self.command(["/bin/sh".to_string(), "-c".to_string(), script.into()])
    }

    /// Set the host directory exposed as `/work`.
    #[must_use]
    pub fn work_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.config.work_dir = dir.into();
        self.config.mount_work = true;
        self
    }

    /// Don't mount anything at `/work`: the guest sees only the root and what
    /// is bound or mounted explicitly.
    #[must_use]
    pub fn no_work_dir(mut self) -> Self {
        self.config.mount_work = false;
        self
    }

    /// Start the first process in guest directory `dir` (absolute; made, with
    /// its parents, if the root lacks it). `PWD` follows unless set with
    /// [`env`](Self::env).
    #[must_use]
    pub fn cwd(mut self, dir: impl Into<String>) -> Self {
        self.config.cwd = Some(dir.into());
        self
    }

    /// Mount a filesystem of one's own at guest path `guest` (made, with its
    /// parents, if missing), e.g. a host directory behind a policy layer.
    /// `make` is called once per run.
    #[must_use]
    pub fn mount<F>(mut self, guest: impl Into<String>, make: F) -> Self
    where
        F: Fn() -> Box<dyn MountFs> + Send + Sync + 'static,
    {
        self.config.mounts.push(Mount {
            guest: guest.into(),
            fs: Arc::new(make),
        });
        self
    }

    /// What the guest's sockets can reach: [`Network::None`] (loopback only),
    /// [`Network::Host`], or an egress backend of one's own. Without this, the
    /// installed `NIXVM_NET` setting decides ([`crate::env`]; loopback only
    /// unless the embedder installed `NIXVM_NET=host`).
    #[must_use]
    pub fn network(mut self, network: Network) -> Self {
        self.config.network = Some(network);
        self
    }

    /// Feed the guest's stdin (fd 0) from `r`. Without this
    /// [`Sandbox::execute`] gives it an empty stdin, and [`Sandbox::run`]
    /// the host's. Clones of the builder share `r`.
    #[must_use]
    pub fn stdin(mut self, r: impl std::io::Read + Send + 'static) -> Self {
        self.config.io.stdin = Some(Arc::new(Mutex::new(r)));
        self
    }

    /// Send the guest's stdout (fd 1) to `w`. Without this
    /// [`Sandbox::execute`] discards it, and [`Sandbox::run`] writes it to the
    /// host's. Clones of the builder share `w`.
    #[must_use]
    pub fn stdout(mut self, w: impl std::io::Write + Send + 'static) -> Self {
        self.config.io.stdout = Some(Arc::new(Mutex::new(w)));
        self
    }

    /// Send the guest's stderr (fd 2) to `w` (see [`stdout`](Self::stdout)).
    #[must_use]
    pub fn stderr(mut self, w: impl std::io::Write + Send + 'static) -> Self {
        self.config.io.stderr = Some(Arc::new(Mutex::new(w)));
        self
    }

    /// Call `f` with each [notice](Notice) the kernel makes, as it's made —
    /// faults, limits, policy denials, traces. Without this
    /// [`Sandbox::execute`] prints nothing (the notices are in its
    /// [`RunOutcome`]), and [`Sandbox::run`] prints them on stderr.
    #[must_use]
    pub fn on_notice<F>(mut self, f: F) -> Self
    where
        F: Fn(&Notice) + Send + Sync + 'static,
    {
        self.config.io.notices = Some(Arc::new(f));
        self
    }

    /// Kill the guest once it has run for `limit` of wall-clock time (sets
    /// [`Limits::wall_time`](crate::limits::Limits::wall_time)).
    #[must_use]
    pub fn wall_time(mut self, limit: std::time::Duration) -> Self {
        self.config.limits.wall_time = Some(limit);
        self
    }

    /// Set the guest memory ceiling in bytes.
    #[must_use]
    pub fn mem_bytes(mut self, bytes: u64) -> Self {
        self.config.mem_bytes = bytes;
        self
    }

    /// Set the VM-wide resource ceilings. The default is
    /// [`Limits::for_this_host`](crate::limits::Limits::for_this_host).
    #[must_use]
    pub fn limits(mut self, limits: crate::limits::Limits) -> Self {
        self.config.limits = limits;
        self
    }

    /// Refuse syscalls to every guest task (see [`crate::policy`]). The default
    /// is what `NIXVM_DENY_SYSCALLS` says, or nothing.
    #[must_use]
    pub fn syscall_policy(mut self, policy: crate::policy::SyscallPolicy) -> Self {
        self.config.syscall_policy = Some(policy);
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

    /// Add a `KEY=VALUE` entry to the guest environment (overrides the
    /// preconfigured default for that key).
    #[must_use]
    pub fn env(mut self, kv: impl Into<String>) -> Self {
        self.config.env.push(kv.into());
        self
    }

    /// Use a host directory (an extracted rootfs) read-only as the guest root.
    #[must_use]
    pub fn root_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.config.root = RootSource::Dir(dir.into());
        self
    }

    /// Use a host directory as the guest root, read-write: the guest's
    /// changes land on the host ([`RootSource::DirWritable`]).
    #[must_use]
    pub fn root_dir_writable(mut self, dir: impl Into<PathBuf>) -> Self {
        self.config.root = RootSource::DirWritable(dir.into());
        self
    }

    /// Use a squashfs image file read-only as the guest root (needs `fstool`).
    #[must_use]
    pub fn root_squashfs(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.root = RootSource::Squashfs(path.into());
        self
    }

    /// Resolve the guest root through the image store (needs `fstool` + cache).
    #[must_use]
    pub fn root_image(mut self, image: ImageRef) -> Self {
        self.config.root = RootSource::Image(image);
        self
    }

    /// Choose the boot mode ([`RunMode::Nude`] or [`RunMode::Booted`]).
    #[must_use]
    pub fn mode(mut self, mode: RunMode) -> Self {
        self.config.mode = mode;
        self
    }

    /// Set an explicit init binary for [`RunMode::Booted`].
    #[must_use]
    pub fn init(mut self, path: impl Into<String>) -> Self {
        self.config.init = Some(path.into());
        self
    }

    /// Set the number of virtual CPUs (host worker threads for guest compute).
    #[must_use]
    pub fn ncpus(mut self, n: usize) -> Self {
        self.config.ncpus = n.max(1);
        self
    }

    /// Set the guest architecture (and the default root image's). Without
    /// this, the architecture follows the PID-1 binary's ELF header.
    #[must_use]
    pub fn arch(mut self, arch: Arch) -> Self {
        self.config.arch = arch;
        self.config.arch_explicit = true;
        self.config.image.arch = arch;
        self
    }

    /// Run a Docker/OCI image: a registry name or an archive path (see
    /// [`RootSource::Oci`]).
    #[must_use]
    pub fn image(mut self, name: impl Into<String>) -> Self {
        self.config.root = RootSource::Oci(name.into());
        self
    }

    /// Replace the image's entrypoint (`docker run --entrypoint`).
    #[must_use]
    pub fn entrypoint<I, S>(mut self, argv: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.config.entrypoint = Some(argv.into_iter().map(Into::into).collect());
        self
    }

    /// Run PID 1 as `user` (`docker run --user`), any root, not just an image.
    #[must_use]
    pub fn user(mut self, user: impl Into<String>) -> Self {
        self.config.user = Some(user.into());
        self
    }

    /// Give the guest a `width` × `height` Wayland display that nixvm
    /// composites itself ([`crate::display`]): its socket is
    /// [`crate::display::DEFAULT_SOCKET`], and the guest's environment gets
    /// `WAYLAND_DISPLAY` and an `XDG_RUNTIME_DIR` unless they're set. Read
    /// its frames through [`Sandbox::display`].
    #[must_use]
    pub fn display(mut self, width: u32, height: u32) -> Self {
        self.config.display = Some(crate::display::Display::new(width, height));
        self
    }

    /// Give the guest the virtual GPU `dev` (`/dev/dri/renderD128`) instead
    /// of whatever `NIXVM_GPU` asks for.
    #[cfg(feature = "gpu")]
    #[must_use]
    pub fn gpu(mut self, dev: crate::gpu::GpuDevice) -> Self {
        self.config.gpu = Some(dev);
        self
    }

    /// Finish building.
    #[must_use]
    pub fn build(self) -> Sandbox {
        Sandbox {
            config: self.config,
        }
    }

    /// Build and run in one step; returns the guest exit code (see
    /// [`Sandbox::run`]).
    pub fn run(self) -> Result<i32, Error> {
        self.build().run()
    }

    /// Build and run in one step for an embedder (see [`Sandbox::execute`]).
    pub fn execute(self) -> Result<RunOutcome, Error> {
        self.build().execute()
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

    /// The guest's display, if it has one ([`SandboxBuilder::display`]): a
    /// handle to read frames from and send input to, from another thread
    /// while [`Sandbox::run`] runs.
    #[must_use]
    pub fn display(&self) -> Option<crate::display::Display> {
        self.config.display.clone()
    }

    /// Boot the container and run PID 1 on the host's terminal, returning its
    /// exit code: what `nixvm run` does.
    ///
    /// Stdio not set on the builder is the host process's own (with terminal
    /// ioctls forwarded to the host tty when none is set), and kernel notices
    /// go to stderr unless [`SandboxBuilder::on_notice`] takes them. See
    /// [`Sandbox::execute`] for embedding.
    ///
    /// Assembles the root (overlay of the read-only image over a writable
    /// tmpfs), preconfigures `/etc` + the environment host-side, then loads and
    /// runs PID 1 — the shell in [`RunMode::Nude`], or init in
    /// [`RunMode::Booted`] (falling back to the shell if no init is present).
    /// The PID-1 binary is read out of the assembled guest root, so this needs a
    /// root that actually contains it (see [`SandboxBuilder::root_dir`] /
    /// `root_squashfs`). The binary must be a static ELF today — dynamic linking
    /// is not yet wired.
    pub fn run(&self) -> Result<i32, Error> {
        Ok(self.launch(true)?.exit_code)
    }

    /// Boot the container and run PID 1 for an embedder, returning how it
    /// went.
    ///
    /// Nothing is printed and nothing of the host process is used that the
    /// builder wasn't given: stdin is empty and stdout/stderr discarded
    /// unless set ([`SandboxBuilder::stdin`], [`SandboxBuilder::stdout`],
    /// [`SandboxBuilder::stderr`]), and the kernel's notices are collected in
    /// the [`RunOutcome`] (and passed to [`SandboxBuilder::on_notice`]'s
    /// callback). Otherwise as [`Sandbox::run`].
    pub fn execute(&self) -> Result<RunOutcome, Error> {
        self.launch(false)
    }

    /// [`Sandbox::run`] (`terminal`) or [`Sandbox::execute`].
    fn launch(&self, terminal: bool) -> Result<RunOutcome, Error> {
        #[cfg(feature = "oci")]
        let image = match &self.config.root {
            RootSource::Oci(name) => Some(self.load_image(name)?),
            _ => None,
        };
        #[cfg(feature = "oci")]
        let lower = match &image {
            Some(img) => {
                let tar = img.rootfs_tar()?;
                let fs = crate::fs::FsToolMount::from_tar(&tar)
                    .map_err(|e| Error::Config(format!("build the image's root: {e}")))?;
                Some(Box::new(fs) as Box<dyn MountFs>)
            }
            None => self.resolve_lower()?,
        };
        #[cfg(not(feature = "oci"))]
        let lower = self.resolve_lower()?;
        let mut mounts = self.build_mounts(lower);
        preconfigure(&mut mounts, self.network_enabled());

        let (_, argv) = self.pid1_program(&mut mounts);
        #[cfg_attr(not(feature = "oci"), allow(unused_mut))]
        let mut start = Pid1 {
            env: self.env(),
            cwd: self.cwd(),
            argv,
            uid: 0,
            gid: 0,
            groups: Vec::new(),
        };
        if let Some(user) = &self.config.user {
            let (uid, gid, home, groups) =
                resolve_user(user, &mut mounts).map_err(Error::Config)?;
            if !self.config.env.iter().any(|e| e.starts_with("HOME=")) {
                start.env.retain(|e| !e.starts_with("HOME="));
                start.env.push(format!("HOME={home}"));
            }
            start.uid = uid;
            start.gid = gid;
            start.groups = groups;
        }
        #[cfg(feature = "oci")]
        if let Some(img) = &image {
            let opts = crate::oci::RunOptions {
                entrypoint: self.config.entrypoint.clone(),
                cmd: self.config.command.clone(),
                env: self.config.env.clone(),
                user: self.config.user.clone(),
                working_dir: self.config.cwd.clone(),
            };
            let l = img.launch(&opts, &mut mounts)?;
            start = Pid1 {
                argv: l.argv,
                env: l.env,
                cwd: l.cwd,
                uid: l.uid,
                gid: l.gid,
                groups: l.groups,
            };
        }
        let Pid1 {
            argv,
            mut env,
            cwd,
            uid,
            gid,
            groups,
        } = start;
        if mounts.stat(&cwd).is_none() {
            mkdir_all(&mut mounts, &cwd);
        }
        if self.config.display.is_some() {
            display_env(&mut mounts, &mut env, uid, gid);
        }
        let argv = crate::vm::resolve_script(&mut mounts, argv).map_err(Error::Config)?;
        let path = argv[0].clone();
        let elf = mounts.read_file(&path).ok_or_else(|| {
            Error::Config(format!(
                "cannot read PID 1 `{path}` from the guest root (is the root populated?)"
            ))
        })?;

        let arch = self.guest_arch(&elf, &path)?;
        let mut mem = GuestMemory::with_default_layout(self.config.mem_bytes);
        // Page tables in the guest's own format, for its hardware MMU.
        mem.set_guest_arch(arch)
            .map_err(|e| Error::Config(format!("guest memory: {e:?}")))?;
        let spec = ProcessSpec {
            argv: argv.clone(),
            envp: env,
        };
        // Dynamic executables name their linker in PT_INTERP; load it alongside.
        let img = if let Some(interp) = interp_path(&elf) {
            let interp_elf = mounts.read_file(&interp).ok_or_else(|| {
                Error::Config(format!(
                    "{path}: dynamic linker `{interp}` not found in the guest root"
                ))
            })?;
            load_dynamic(&mut mem, &elf, &interp_elf, &spec)
                .map_err(|e| Error::Config(format!("{path}: dynamic load failed: {e}")))?
        } else {
            load_static(&mut mem, &elf, &spec)
                .map_err(|e| Error::Config(format!("{path}: load failed: {e}")))?
        };
        let mid = page_align_down(img.program_break + (img.stack_bottom - img.program_break) / 2);

        let backend = self.backend(arch)?;
        let vcpu = backend.new_vcpu(img.entry, img.stack_pointer)?;

        let mut kernel = Kernel::new(arch, mounts);
        kernel.set_limits(self.config.limits);
        if let Some(policy) = &self.config.syscall_policy {
            kernel.set_syscall_policy(policy);
        }
        kernel.set_ncpus(self.config.ncpus);
        let notices = self.install_io(&mut kernel, terminal);
        kernel.set_cwd(cwd);
        kernel.set_creds(uid, gid, &groups);
        kernel.set_exe(&path); // /proc/self/exe for pid 1
        kernel.set_cmdline(argv); // /proc/self/cmdline + comm for pid 1
        kernel.set_heap(img.program_break, mid);
        kernel.set_mmap_area(img.stack_bottom, mid);
        if let Some(display) = &self.config.display {
            display
                .attach(&kernel, crate::display::DEFAULT_SOCKET)
                .map_err(|e| Error::Config(format!("the display's socket: errno {}", -e)))?;
        }
        self.install_network(&mut kernel);
        self.install_gpu(&kernel);
        mem.calibrate_vdso(vcpu.as_ref());
        let exit_code = kernel.run(vcpu, mem)?;
        Ok(self.outcome(&kernel, exit_code, &notices, backend.name()))
    }

    /// Give `kernel` the builder's stdio and a notice sink that keeps what it
    /// sees (but traces) in the returned list and forwards it to the
    /// builder's callback — else, on the `terminal`, to stderr.
    fn install_io(&self, kernel: &mut Kernel, terminal: bool) -> Arc<Mutex<Vec<Notice>>> {
        let io = &self.config.io;
        match &io.stdin {
            Some(r) => kernel.set_stdin(Box::new(FromShared(Arc::clone(r)))),
            None if !terminal => kernel.set_stdin(Box::new(std::io::empty())),
            None => {}
        }
        match &io.stdout {
            Some(w) => kernel.set_stdout(Box::new(ToShared(Arc::clone(w)))),
            None if !terminal => kernel.set_stdout(Box::new(std::io::sink())),
            None => {}
        }
        match &io.stderr {
            Some(w) => kernel.set_stderr(Box::new(ToShared(Arc::clone(w)))),
            None if !terminal => kernel.set_stderr(Box::new(std::io::sink())),
            None => {}
        }
        // Terminal ioctls reach the host tty only when the guest's stdio is
        // the host's own.
        kernel.set_host_tty(
            terminal && io.stdin.is_none() && io.stdout.is_none() && io.stderr.is_none(),
        );
        let kept = Arc::new(Mutex::new(Vec::new()));
        let forward: Option<NoticeSink> = io
            .notices
            .clone()
            .or_else(|| terminal.then(crate::kernel::notice::stderr_sink));
        let keep = Arc::clone(&kept);
        kernel.set_notice_sink(Arc::new(move |n: &Notice| {
            if !matches!(n, Notice::Trace(_)) {
                keep.lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .push(n.clone());
            }
            if let Some(f) = &forward {
                f(n);
            }
        }));
        kept
    }

    /// How the run went, once [`Kernel::run`] returned `exit_code`.
    fn outcome(
        &self,
        kernel: &Kernel,
        exit_code: i32,
        notices: &Mutex<Vec<Notice>>,
        backend: &'static str,
    ) -> RunOutcome {
        for w in [&self.config.io.stdout, &self.config.io.stderr]
            .into_iter()
            .flatten()
        {
            let _ = w
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .flush();
        }
        let notices = std::mem::take(
            &mut *notices
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        let signal = kernel.pid1_signal();
        let mut unsupported: Vec<Unsupported> = kernel
            .unsupported()
            .into_iter()
            .map(|(nr, count)| Unsupported::Syscall { nr, count })
            .collect();
        unsupported.extend(kernel.unsupported_subcommands().into_iter().map(
            |((syscall, sub), count)| Unsupported::Subcommand {
                syscall,
                sub,
                count,
            },
        ));
        RunOutcome {
            exit_code,
            signal,
            limit: RunOutcome::limit_from_notices(&notices, signal),
            unsupported,
            notices,
            backend,
        }
    }

    /// Whether the guest gets more than its loopback.
    fn network_enabled(&self) -> bool {
        match &self.config.network {
            Some(n) => n.is_enabled() && egress_possible(n),
            None => crate::vm::egress_enabled(),
        }
    }

    /// Give `kernel` the network the builder asked for, else the one the
    /// installed `NIXVM_NET` names.
    fn install_network(&self, kernel: &mut Kernel) {
        match &self.config.network {
            None => crate::vm::install_egress(kernel),
            Some(Network::None) => {}
            Some(Network::Host) => {
                #[cfg(not(target_arch = "wasm32"))]
                kernel.set_egress(Box::new(crate::kernel::egress::HostEgress));
            }
            Some(Network::Egress(make)) => kernel.set_egress(make()),
        }
    }

    /// Load the image a [`RootSource::Oci`] names: an archive or layout on
    /// disk, else a pull from its registry. Unless the guest architecture was
    /// set explicitly, an image without one for this host's runs as the other
    /// (on the interpreter).
    #[cfg(feature = "oci")]
    fn load_image(&self, name: &str) -> Result<crate::oci::Image, Error> {
        use crate::oci::{Image, OciError};
        let arch = self.config.arch;
        let other = match arch {
            Arch::X86_64 => Arch::Aarch64,
            Arch::Aarch64 => Arch::X86_64,
        };
        let local = Path::new(name);
        let load = |a: Arch| -> Result<Image, OciError> {
            if local.is_dir() {
                return Image::from_dir(local, a);
            }
            if local.is_file() {
                return Image::from_archive(&std::fs::read(local)?, a);
            }
            #[cfg(feature = "fetch")]
            {
                crate::oci::pull(name, a)
            }
            #[cfg(not(feature = "fetch"))]
            {
                Err(OciError::Fetch(format!(
                    "{name} isn't a file, and pulling it from a registry needs the `fetch` feature"
                )))
            }
        };
        match load(arch) {
            Err(OciError::Platform(_)) if !self.config.arch_explicit => Ok(load(other)?),
            r => Ok(r?),
        }
    }

    /// PID 1's path and argv. Booted mode runs init (config override, else
    /// `/sbin/init`) when the root provides it, otherwise falls back to the
    /// shell/command like nude mode.
    fn pid1_program(&self, mounts: &mut MountTable) -> (String, Vec<String>) {
        let shell = || {
            if self.config.command.is_empty() {
                vec!["/bin/sh".to_string()]
            } else {
                self.config.command.clone()
            }
        };
        if self.config.mode == RunMode::Booted {
            let init = self
                .config
                .init
                .clone()
                .unwrap_or_else(|| "/sbin/init".to_string());
            if mounts.stat(&init).is_some() {
                return (init.clone(), vec![init]);
            }
            // No runnable init: fall back to the shell (still "booted" env).
        }
        let argv = shell();
        (argv[0].clone(), argv)
    }

    /// PID 1's working directory: the configured one, else `/` when booted
    /// and `/root` otherwise.
    fn cwd(&self) -> String {
        match &self.config.cwd {
            Some(dir) => dir.clone(),
            None if self.config.mode == RunMode::Booted => "/".into(),
            None => "/root".into(),
        }
    }

    /// The guest environment. Preconfigured so the shell is usable immediately
    /// in either mode (network is a virtual loopback that is always up).
    fn env(&self) -> Vec<String> {
        let mut env = vec![
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin".to_string(),
            "HOME=/root".to_string(),
            "TERM=xterm".to_string(),
            "HOSTNAME=nixvm".to_string(),
            "PS1=nixvm:\\w\\$ ".to_string(),
            format!("PWD={}", self.cwd()),
        ];
        if self.config.mode == RunMode::Booted {
            env.push("RUNLEVEL=3".to_string());
            env.push("PREVLEVEL=N".to_string());
        }
        // Caller-supplied entries win: drop any default sharing their key.
        for kv in &self.config.env {
            let key = kv.split('=').next().unwrap_or(kv);
            env.retain(|e| e.split('=').next() != Some(key));
            env.push(kv.clone());
        }
        env
    }

    /// Load and run a statically-linked ELF64 image, returning its exit code.
    ///
    /// This is the full pipeline — loader → backend → kernel run/serve loop —
    /// that [`Sandbox::run`] will call once image/filesystem resolution reads
    /// the target binary out of the guest root. Exposed now so it's testable
    /// and embeddable ahead of that.
    pub fn exec_elf(&self, elf: &[u8]) -> Result<i32, Error> {
        let arch = self.guest_arch(elf, "ELF image")?;
        let mut mem = GuestMemory::with_default_layout(self.config.mem_bytes);
        // Page tables in the guest's own format, for its hardware MMU.
        mem.set_guest_arch(arch)
            .map_err(|e| Error::Config(format!("guest memory: {e:?}")))?;

        let argv = if self.config.command.is_empty() {
            vec!["prog".to_string()]
        } else {
            self.config.command.clone()
        };
        let spec = ProcessSpec {
            argv: argv.clone(),
            envp: self.env(),
        };
        let img = load_static(&mut mem, elf, &spec)?;

        // Lay out heap and mmap in the gap between the image and the stack: the
        // heap grows up from the program break, mmap grows down from the stack,
        // meeting at a midpoint so the two arenas can't collide.
        let mid = page_align_down(img.program_break + (img.stack_bottom - img.program_break) / 2);

        let backend = self.backend(arch)?;
        let vcpu = backend.new_vcpu(img.entry, img.stack_pointer)?;

        let mut kernel = Kernel::new(arch, self.build_mounts(self.resolve_lower()?));
        kernel.set_ncpus(self.config.ncpus);
        let _ = self.install_io(&mut kernel, true);
        kernel.set_cwd(self.config.cwd.as_deref().unwrap_or("/work"));
        kernel.set_cmdline(argv);
        kernel.set_heap(img.program_break, mid);
        kernel.set_mmap_area(img.stack_bottom, mid);
        self.install_network(&mut kernel);
        self.install_gpu(&kernel);
        mem.calibrate_vdso(vcpu.as_ref());
        Ok(kernel.run(vcpu, mem)?)
    }

    /// The GPU the builder was given, else the one `NIXVM_GPU` asks for.
    #[cfg_attr(not(feature = "gpu"), allow(clippy::unused_self))]
    fn install_gpu(&self, kernel: &Kernel) {
        #[cfg(feature = "gpu")]
        if let Some(dev) = &self.config.gpu {
            let _ = kernel.attach_gpu(dev.clone());
            return;
        }
        crate::vm::install_gpu(kernel);
    }

    /// Select the execution backend per config: the interpreter when forced,
    /// otherwise the best hardware backend for the host.
    fn backend(&self, arch: Arch) -> Result<Box<dyn Backend>, Error> {
        Ok(vcpu::select_or_interpreter(
            arch,
            self.config.prefer_interp,
        )?)
    }

    /// The guest architecture for PID 1's `elf`: its ELF machine, unless the
    /// configuration fixed one explicitly — then a mismatch is an error (an
    /// x86-64 binary can't run as an aarch64 guest). An unrecognised header
    /// keeps the configured arch (the loader reports the real problem).
    fn guest_arch(&self, elf: &[u8], what: &str) -> Result<Arch, Error> {
        match Arch::from_elf(elf) {
            Some(a) if a != self.config.arch && self.config.arch_explicit => {
                Err(Error::Config(format!(
                    "{what} is an {} binary, but the guest architecture is set to {}",
                    a.as_str(),
                    self.config.arch.as_str()
                )))
            }
            Some(a) => Ok(a),
            None => Ok(self.config.arch),
        }
    }

    /// Assemble the sandbox mount layout.
    ///
    /// `/` is `Overlay::new(root_lower, tmpfs_upper)` — the read-only root image
    /// (a host dir or squashfs) under a writable in-memory upper, so many
    /// instances share one read-only lower — or a bare tmpfs when the root is
    /// [`RootSource::Empty`]. `/tmp` is its own tmpfs; `/dev`, `/proc`, `/sys`
    /// are synthetic; `/work` and any binds are host passthroughs ("holes").
    fn build_mounts(&self, lower: Option<Box<dyn MountFs>>) -> MountTable {
        let mut mounts = MountTable::new();
        match lower {
            #[cfg(any(unix, nixvm_linux))]
            Some(lower) if matches!(self.config.root, RootSource::DirWritable(_)) => {
                mounts.mount("/", lower);
            }
            Some(lower) => {
                mounts.mount("/", Box::new(Overlay::new(lower, Box::new(TmpFs::new()))));
            }
            None => mounts.mount("/", Box::new(TmpFs::new())),
        }
        mounts.mount("/tmp", Box::new(TmpFs::new()));
        mounts.mount("/dev", Box::new(DevFs::new()));
        mounts.mount("/proc", Box::new(ProcFs::new(self.config.ncpus)));
        mounts.mount("/sys", Box::new(SysFs::new(self.config.ncpus)));
        mounts.mount(
            "/sys/fs/cgroup",
            Box::new(crate::fs::CgroupFs::new(
                crate::fs::CgroupTree::shared(),
                crate::fs::cgroupfs::ROOT,
            )),
        );

        #[cfg(any(unix, nixvm_linux))]
        {
            if self.config.mount_work {
                mounts.mount(
                    "/work",
                    Box::new(Passthrough::new(self.config.work_dir.clone())),
                );
            }
            for b in &self.config.binds {
                let pt = if b.read_only {
                    Passthrough::read_only(b.host.clone())
                } else {
                    Passthrough::new(b.host.clone())
                };
                mkdir_all(&mut mounts, &b.guest);
                mounts.mount(b.guest.clone(), Box::new(pt));
            }
        }
        for m in &self.config.mounts {
            mkdir_all(&mut mounts, &m.guest);
            mounts.mount(m.guest.clone(), (m.fs)());
        }
        mounts
    }

    /// Resolve the overlay's read-only lower layer from the configured
    /// [`RootSource`], or `None` for a bare-tmpfs root.
    fn resolve_lower(&self) -> Result<Option<Box<dyn MountFs>>, Error> {
        match &self.config.root {
            RootSource::Empty => Ok(None),
            RootSource::DirWritable(dir) => {
                #[cfg(any(unix, nixvm_linux))]
                {
                    Ok(Some(Box::new(Passthrough::new(dir.clone()))))
                }
                #[cfg(not(any(unix, nixvm_linux)))]
                {
                    let _ = dir;
                    Err(Error::Config("a host-directory root requires unix".into()))
                }
            }
            RootSource::Dir(dir) => {
                #[cfg(any(unix, nixvm_linux))]
                {
                    Ok(Some(Box::new(Passthrough::read_only(dir.clone()))))
                }
                #[cfg(not(any(unix, nixvm_linux)))]
                {
                    let _ = dir;
                    Err(Error::Config("a host-directory root requires unix".into()))
                }
            }
            RootSource::Squashfs(path) => open_squashfs(path),
            // Loaded by `run`, which needs the image's config too.
            RootSource::Oci(name) => Err(Error::Config(format!(
                "running the image {name} needs the `oci` feature"
            ))),
            RootSource::Image(image) => {
                let store = ImageStore::default_location();
                let path = store
                    .ensure(image)
                    .map_err(|e| Error::Config(e.to_string()))?;
                open_squashfs(&path)
            }
        }
    }
}

/// `user[:group]` → (uid, gid, home, supplementary groups), names looked up
/// in the root's `/etc/passwd` and `/etc/group`. A numeric uid needn't be
/// listed; its gid is then its passwd entry's, or 0.
pub(crate) fn resolve_user(
    spec: &str,
    root: &mut MountTable,
) -> Result<(u32, u32, String, Vec<u32>), String> {
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
            return Err(format!("user {u:?} isn't in the root's /etc/passwd"));
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
                .ok_or_else(|| format!("group {g:?} isn't in the root's /etc/group"))?,
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

/// How PID 1 starts.
struct Pid1 {
    /// `argv[0]` is the program's path in the root.
    argv: Vec<String>,
    env: Vec<String>,
    cwd: String,
    uid: u32,
    gid: u32,
    groups: Vec<u32>,
}

/// Open a squashfs image as an overlay lower layer (needs the `fstool` feature).
fn open_squashfs(path: &Path) -> Result<Option<Box<dyn MountFs>>, Error> {
    #[cfg(feature = "fstool")]
    {
        let fs = crate::fs::FsToolMount::open_squashfs(path)
            .map_err(|e| Error::Config(format!("open squashfs {}: {e}", path.display())))?;
        Ok(Some(Box::new(fs)))
    }
    #[cfg(not(feature = "fstool"))]
    {
        let _ = path;
        Err(Error::Config(
            "a squashfs root requires the `fstool` feature".into(),
        ))
    }
}

/// Seed the writable upper layer with the handful of `/etc` files a shell
/// expects, and create `/root`. Existing files (from a real root image) are
/// never clobbered — this only fills gaps. This is the host-side stand-in for
/// boot-time service configuration until in-guest init can run.
/// Make directory `path` and its missing parents (best effort: what can't be
/// made shows up as an error when it's used).
fn mkdir_all(mounts: &mut MountTable, path: &str) {
    let mut at = String::new();
    for part in path.split('/').filter(|p| !p.is_empty()) {
        at.push('/');
        at.push_str(part);
        if mounts.stat(&at).is_none() {
            let _ = mounts.mkdir(&at, 0o755);
        }
    }
}

/// Whether `network` can be had in this build (no host sockets on wasm).
fn egress_possible(network: &Network) -> bool {
    !(cfg!(target_arch = "wasm32") && matches!(network, Network::Host))
}

/// What a guest with a display needs: `WAYLAND_DISPLAY` naming its socket,
/// and an `XDG_RUNTIME_DIR` (made, `0700` and owned by the user PID 1 runs
/// as) — unless the environment already sets them.
fn display_env(mounts: &mut MountTable, env: &mut Vec<String>, uid: u32, gid: u32) {
    let has = |env: &[String], key: &str| env.iter().any(|e| e.split('=').next() == Some(key));
    let _ = mounts.mkdir("/tmp", 0o1777);
    if !has(env, "WAYLAND_DISPLAY") {
        env.push(format!(
            "WAYLAND_DISPLAY={}",
            crate::display::DEFAULT_SOCKET
        ));
    }
    if !has(env, "XDG_RUNTIME_DIR") {
        let dir = format!("/run/user/{uid}");
        let _ = mounts.mkdir("/run", 0o755);
        let _ = mounts.mkdir("/run/user", 0o755);
        let _ = mounts.mkdir(&dir, 0o700);
        let _ = mounts.set_owner(&dir, Some(uid), Some(gid));
        env.push(format!("XDG_RUNTIME_DIR={dir}"));
    }
}

fn preconfigure(mounts: &mut MountTable, network: bool) {
    let _ = mounts.mkdir("/etc", 0o755);
    let _ = mounts.mkdir("/root", 0o700);
    // With host egress the guest resolver must reach a real (routable)
    // nameserver; the loopback-only default (Docker's embedded 127.0.0.11) would
    // get ENETUNREACH. A root that ships its own resolv.conf is never clobbered.
    let resolv = if network {
        "nameserver 1.1.1.1\nnameserver 8.8.8.8\n"
    } else {
        "nameserver 127.0.0.11\n"
    };
    let files: &[(&str, &str)] = &[
        ("/etc/hostname", "nixvm\n"),
        (
            "/etc/hosts",
            "127.0.0.1\tlocalhost nixvm\n::1\tlocalhost ip6-localhost\n",
        ),
        ("/etc/resolv.conf", resolv),
        (
            "/etc/os-release",
            "NAME=nixvm\nID=nixvm\nPRETTY_NAME=\"nixvm sandbox\"\nVERSION_ID=0\n",
        ),
        ("/etc/passwd", "root:x:0:0:root:/root:/bin/sh\n"),
        ("/etc/group", "root:x:0:\n"),
    ];
    for (path, content) in files {
        if mounts.stat(path).is_none() && mounts.create(path, 0o644).is_ok() {
            let _ = mounts.write_at(path, 0, content.as_bytes());
        }
    }
}

fn page_align_down(v: u64) -> u64 {
    v - v % PAGE_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `movz x0,#code ; movz x8,#93 ; svc #0` — the guest `exit(code)`.
    fn exit_words(code: u16) -> Vec<u8> {
        let words: [u32; 3] = [
            0xD280_0000 | (u32::from(code) << 5),
            0xD280_0BA8,
            0xD400_0001,
        ];
        let mut b = Vec::new();
        for w in words {
            b.extend_from_slice(&w.to_le_bytes());
        }
        b
    }

    /// A minimal aarch64 ELF: an optional `PT_INTERP` header plus one `PT_LOAD`
    /// mapping the whole file (headers included). `e_type` is ET_EXEC (2) or
    /// ET_DYN (3, a PIE with `p_vaddr = 0` that the loader biases). Entry points
    /// at the code, which follows the (optional) interpreter path string.
    fn build_elf(e_type: u16, vaddr: u64, interp: Option<&str>, code: &[u8]) -> Vec<u8> {
        let (ehsize, phsize) = (64usize, 56usize);
        let nph = if interp.is_some() { 2 } else { 1 };
        let mut interp_bytes = interp.map(|s| s.as_bytes().to_vec()).unwrap_or_default();
        if interp.is_some() {
            interp_bytes.push(0);
        }
        let interp_off = ehsize + nph * phsize;
        let code_off = (interp_off + interp_bytes.len()) as u64;
        let file_len = code_off + code.len() as u64;
        let entry = vaddr + code_off;

        let mut e = vec![0u8; interp_off];
        e[0..8].copy_from_slice(&[0x7f, b'E', b'L', b'F', 2, 1, 1, 0]);
        e[16..18].copy_from_slice(&e_type.to_le_bytes());
        e[18..20].copy_from_slice(&183u16.to_le_bytes()); // AArch64
        e[20..24].copy_from_slice(&1u32.to_le_bytes());
        e[24..32].copy_from_slice(&entry.to_le_bytes());
        e[32..40].copy_from_slice(&(ehsize as u64).to_le_bytes()); // e_phoff
        e[52..54].copy_from_slice(&(ehsize as u16).to_le_bytes());
        e[54..56].copy_from_slice(&(phsize as u16).to_le_bytes());
        e[56..58].copy_from_slice(&(nph as u16).to_le_bytes());

        let mut p = ehsize;
        if let Some(_s) = interp {
            e[p..p + 4].copy_from_slice(&3u32.to_le_bytes()); // PT_INTERP
            e[p + 4..p + 8].copy_from_slice(&4u32.to_le_bytes()); // PF_R
            e[p + 8..p + 16].copy_from_slice(&(interp_off as u64).to_le_bytes());
            e[p + 32..p + 40].copy_from_slice(&(interp_bytes.len() as u64).to_le_bytes());
            e[p + 40..p + 48].copy_from_slice(&(interp_bytes.len() as u64).to_le_bytes());
            p += phsize;
        }
        e[p..p + 4].copy_from_slice(&1u32.to_le_bytes()); // PT_LOAD
        e[p + 4..p + 8].copy_from_slice(&5u32.to_le_bytes()); // R+X
        e[p + 16..p + 24].copy_from_slice(&vaddr.to_le_bytes()); // p_vaddr
        e[p + 24..p + 32].copy_from_slice(&vaddr.to_le_bytes()); // p_paddr
        e[p + 32..p + 40].copy_from_slice(&file_len.to_le_bytes());
        e[p + 40..p + 48].copy_from_slice(&file_len.to_le_bytes());
        e[p + 48..p + 56].copy_from_slice(&0x1000u64.to_le_bytes());

        e.extend_from_slice(&interp_bytes);
        e.extend_from_slice(code);
        e
    }

    /// A static ET_EXEC `exit(code)` at the fixed guest base.
    fn build_exit_elf(code: u16) -> Vec<u8> {
        build_elf(2, 0x1_0000, None, &exit_words(code))
    }

    #[cfg(any(unix, nixvm_linux))]
    fn temp_root(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("nixvm-sb-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[cfg(any(unix, nixvm_linux))]
    #[test]
    fn nude_mode_boots_the_command_from_a_host_dir_root() {
        let dir = temp_root("nude");
        std::fs::write(dir.join("init"), build_exit_elf(42)).unwrap();
        let code = Sandbox::builder()
            .arch(Arch::Aarch64)
            .prefer_interp(true)
            .root_dir(&dir)
            .command(["/init"])
            .run()
            .unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(
            code, 42,
            "PID 1 read from the root and run to its exit code"
        );
    }

    /// PID 1 named through an *absolute* symlink (`/bin/sh -> /bin/busybox`)
    /// resolves inside the guest root, not on the host.
    #[cfg(any(unix, nixvm_linux))]
    #[test]
    fn pid1_through_an_absolute_symlink_resolves_in_the_root() {
        let dir = temp_root("symlink");
        std::fs::create_dir_all(dir.join("bin")).unwrap();
        std::fs::write(dir.join("bin/real"), build_exit_elf(42)).unwrap();
        std::os::unix::fs::symlink("/bin/real", dir.join("bin/sh")).unwrap();
        let code = Sandbox::builder()
            .prefer_interp(true)
            .root_dir(&dir)
            .command(["/bin/sh"])
            .run()
            .unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(code, 42);
    }

    /// Without `.arch()`, the guest arch follows PID 1's ELF header; pinning
    /// a different one is a clear error.
    #[cfg(any(unix, nixvm_linux))]
    #[test]
    fn guest_arch_follows_the_elf_unless_pinned() {
        let dir = temp_root("arch");
        std::fs::write(dir.join("init"), build_exit_elf(7)).unwrap();
        let code = Sandbox::builder()
            .prefer_interp(true)
            .root_dir(&dir)
            .command(["/init"])
            .run()
            .unwrap();
        assert_eq!(code, 7, "an aarch64 PID 1 runs as aarch64 on any host");
        let err = Sandbox::builder()
            .arch(Arch::X86_64)
            .prefer_interp(true)
            .root_dir(&dir)
            .command(["/init"])
            .run()
            .unwrap_err();
        std::fs::remove_dir_all(&dir).ok();
        assert!(matches!(err, Error::Config(m) if m.contains("aarch64 binary")));
        assert_eq!(Arch::from_name("AMD64"), Some(Arch::X86_64));
        assert_eq!(Arch::from_name("arm64"), Some(Arch::Aarch64));
        assert_eq!(Arch::from_name("riscv64"), None);
    }

    #[cfg(any(unix, nixvm_linux))]
    #[test]
    fn boots_a_dynamic_pid1_via_its_interpreter() {
        let dir = temp_root("dyn");
        std::fs::create_dir_all(dir.join("lib")).unwrap();
        // PID 1 /init is a dynamic ET_DYN naming interpreter /lib/ld; its own
        // code never runs (control goes to the interpreter's entry).
        std::fs::write(
            dir.join("init"),
            build_elf(3, 0, Some("/lib/ld"), &exit_words(0)),
        )
        .unwrap();
        // The "interpreter" is a PIE that just exits 55 — standing in for the
        // real ld-musl, which would instead relocate and jump to the exe.
        std::fs::write(dir.join("lib/ld"), build_elf(3, 0, None, &exit_words(55))).unwrap();

        let code = Sandbox::builder()
            .arch(Arch::Aarch64)
            .prefer_interp(true)
            .root_dir(&dir)
            .command(["/init"])
            .run()
            .unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(code, 55, "control transferred to the interpreter's entry");
    }

    #[cfg(any(unix, nixvm_linux))]
    #[test]
    fn dynamic_pid1_missing_interpreter_is_a_clear_error() {
        let dir = temp_root("dyn-noint");
        std::fs::write(
            dir.join("init"),
            build_elf(3, 0, Some("/lib/ld"), &exit_words(0)),
        )
        .unwrap();
        let err = Sandbox::builder()
            .arch(Arch::Aarch64)
            .prefer_interp(true)
            .root_dir(&dir)
            .command(["/init"])
            .run()
            .unwrap_err();
        std::fs::remove_dir_all(&dir).ok();
        assert!(matches!(err, Error::Config(m) if m.contains("dynamic linker")));
    }

    #[cfg(any(unix, nixvm_linux))]
    #[test]
    fn booted_mode_runs_sbin_init() {
        let dir = temp_root("booted");
        std::fs::create_dir_all(dir.join("sbin")).unwrap();
        std::fs::write(dir.join("sbin/init"), build_exit_elf(7)).unwrap();
        let code = Sandbox::builder()
            .arch(Arch::Aarch64)
            .prefer_interp(true)
            .root_dir(&dir)
            .mode(RunMode::Booted)
            .run()
            .unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(code, 7, "booted mode runs /sbin/init as PID 1");
    }

    #[cfg(any(unix, nixvm_linux))]
    #[test]
    fn same_result_on_four_cpus() {
        let dir = temp_root("smp");
        std::fs::write(dir.join("init"), build_exit_elf(19)).unwrap();
        let code = Sandbox::builder()
            .arch(Arch::Aarch64)
            .prefer_interp(true)
            .root_dir(&dir)
            .command(["/init"])
            .ncpus(4)
            .run()
            .unwrap();
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(code, 19);
    }

    #[test]
    fn preconfigure_seeds_etc_and_defaults_to_shell() {
        let sb = Sandbox::builder().build();
        let mut mounts = sb.build_mounts(sb.resolve_lower().unwrap());
        preconfigure(&mut mounts, false);
        assert!(
            mounts.stat("/etc/hostname").is_some(),
            "/etc/hostname seeded"
        );
        let (path, argv) = sb.pid1_program(&mut mounts);
        assert_eq!(path, "/bin/sh", "nude mode defaults PID 1 to the shell");
        assert_eq!(argv, vec!["/bin/sh".to_string()]);
    }

    #[test]
    fn missing_pid1_is_a_clear_error() {
        // Empty root, no shell present -> a descriptive error, not a panic.
        let err = Sandbox::builder()
            .arch(Arch::Aarch64)
            .prefer_interp(true)
            .command(["/bin/sh"])
            .run()
            .unwrap_err();
        assert!(matches!(err, Error::Config(_)));
    }
}
