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

use std::path::{Path, PathBuf};

use crate::Error;
use crate::abi::Arch;
#[cfg(unix)]
use crate::fs::Passthrough;
use crate::fs::{DevFs, MountFs, MountTable, Overlay, ProcFs, SysFs, TmpFs};
use crate::image::{ImageRef, ImageStore};
use crate::kernel::Kernel;
use crate::loader::{ProcessSpec, load_static};
use crate::vcpu::interp::InterpBackend;
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
    /// A squashfs image file used read-only as the lower layer (needs `fstool`).
    Squashfs(PathBuf),
    /// A named image resolved through the [`ImageStore`] (squashfs; needs
    /// `fstool` and a cached image).
    Image(ImageRef),
}

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
                root: RootSource::Empty,
                mode: RunMode::Nude,
                init: None,
                ncpus: 1,
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

    /// Use a host directory (an extracted rootfs) read-only as the guest root.
    #[must_use]
    pub fn root_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.config.root = RootSource::Dir(dir.into());
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

    /// Set the guest architecture.
    #[must_use]
    pub fn arch(mut self, arch: Arch) -> Self {
        self.config.arch = arch;
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

    /// Boot the container and run PID 1, returning its exit code.
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
        let mut mounts = self.build_mounts()?;
        preconfigure(&mut mounts);

        let (path, argv) = self.pid1_program(&mut mounts);
        let elf = read_mount_file(&mut mounts, &path).ok_or_else(|| {
            Error::Config(format!(
                "cannot read PID 1 `{path}` from the guest root (is the root populated?)"
            ))
        })?;

        let arch = self.config.arch;
        let mut mem = GuestMemory::new(GUEST_BASE, round_up_page(self.config.mem_bytes));
        let spec = ProcessSpec {
            argv,
            envp: self.env(),
        };
        let img = load_static(&mut mem, &elf, &spec).map_err(|_| {
            Error::Config(format!(
                "{path}: not a runnable static ELF (dynamic linking is not yet supported)"
            ))
        })?;
        let mid = page_align_down(img.program_break + (img.stack_bottom - img.program_break) / 2);

        let backend = self.backend()?;
        let vcpu = backend.new_vcpu(img.entry, img.stack_pointer)?;

        let mut kernel = Kernel::new(arch, mounts);
        kernel.set_ncpus(self.config.ncpus);
        kernel.set_cwd(if self.config.mode == RunMode::Booted {
            "/"
        } else {
            "/root"
        });
        kernel.set_heap(img.program_break, mid);
        kernel.set_mmap_area(img.stack_bottom, mid);
        Ok(kernel.run(vcpu, mem)?)
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

    /// The guest environment. Preconfigured so the shell is usable immediately
    /// in either mode (network is a virtual loopback that is always up).
    fn env(&self) -> Vec<String> {
        let mut env = vec![
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin".to_string(),
            "HOME=/root".to_string(),
            "TERM=xterm".to_string(),
            "HOSTNAME=nixvm".to_string(),
            "PS1=nixvm:\\w\\$ ".to_string(),
            format!(
                "PWD={}",
                if self.config.mode == RunMode::Booted {
                    "/"
                } else {
                    "/root"
                }
            ),
        ];
        if self.config.mode == RunMode::Booted {
            env.push("RUNLEVEL=3".to_string());
            env.push("PREVLEVEL=N".to_string());
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
        let arch = self.config.arch;
        let mut mem = GuestMemory::new(GUEST_BASE, round_up_page(self.config.mem_bytes));

        let argv = if self.config.command.is_empty() {
            vec!["prog".to_string()]
        } else {
            self.config.command.clone()
        };
        let spec = ProcessSpec {
            argv,
            envp: self.env(),
        };
        let img = load_static(&mut mem, elf, &spec)?;

        // Lay out heap and mmap in the gap between the image and the stack: the
        // heap grows up from the program break, mmap grows down from the stack,
        // meeting at a midpoint so the two arenas can't collide.
        let mid = page_align_down(img.program_break + (img.stack_bottom - img.program_break) / 2);

        let backend = self.backend()?;
        let vcpu = backend.new_vcpu(img.entry, img.stack_pointer)?;

        let mut kernel = Kernel::new(arch, self.build_mounts()?);
        kernel.set_ncpus(self.config.ncpus);
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
    /// `/` is `Overlay::new(root_lower, tmpfs_upper)` — the read-only root image
    /// (a host dir or squashfs) under a writable in-memory upper, so many
    /// instances share one read-only lower — or a bare tmpfs when the root is
    /// [`RootSource::Empty`]. `/tmp` is its own tmpfs; `/dev`, `/proc`, `/sys`
    /// are synthetic; `/work` and any binds are host passthroughs ("holes").
    fn build_mounts(&self) -> Result<MountTable, Error> {
        let mut mounts = MountTable::new();
        match self.resolve_lower()? {
            Some(lower) => {
                mounts.mount("/", Box::new(Overlay::new(lower, Box::new(TmpFs::new()))));
            }
            None => mounts.mount("/", Box::new(TmpFs::new())),
        }
        mounts.mount("/tmp", Box::new(TmpFs::new()));
        mounts.mount("/dev", Box::new(DevFs::new()));
        mounts.mount("/proc", Box::new(ProcFs::new()));
        mounts.mount("/sys", Box::new(SysFs::new()));

        #[cfg(unix)]
        {
            mounts.mount(
                "/work",
                Box::new(Passthrough::new(self.config.work_dir.clone())),
            );
            for b in &self.config.binds {
                let pt = if b.read_only {
                    Passthrough::read_only(b.host.clone())
                } else {
                    Passthrough::new(b.host.clone())
                };
                mounts.mount(b.guest.clone(), Box::new(pt));
            }
        }
        Ok(mounts)
    }

    /// Resolve the overlay's read-only lower layer from the configured
    /// [`RootSource`], or `None` for a bare-tmpfs root.
    fn resolve_lower(&self) -> Result<Option<Box<dyn MountFs>>, Error> {
        match &self.config.root {
            RootSource::Empty => Ok(None),
            RootSource::Dir(dir) => {
                #[cfg(unix)]
                {
                    Ok(Some(Box::new(Passthrough::read_only(dir.clone()))))
                }
                #[cfg(not(unix))]
                {
                    let _ = dir;
                    Err(Error::Config("a host-directory root requires unix".into()))
                }
            }
            RootSource::Squashfs(path) => open_squashfs(path),
            RootSource::Image(image) => {
                let store = ImageStore::default_location();
                let path = store.ensure(image).map_err(|e| Error::Config(e.to_string()))?;
                open_squashfs(&path)
            }
        }
    }
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
fn preconfigure(mounts: &mut MountTable) {
    let _ = mounts.mkdir("/etc", 0o755);
    let _ = mounts.mkdir("/root", 0o700);
    let files: &[(&str, &str)] = &[
        ("/etc/hostname", "nixvm\n"),
        (
            "/etc/hosts",
            "127.0.0.1\tlocalhost nixvm\n::1\tlocalhost ip6-localhost\n",
        ),
        ("/etc/resolv.conf", "nameserver 127.0.0.11\n"),
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

/// Read an entire file out of the mount table (PID-1 binary lookup).
fn read_mount_file(mounts: &mut MountTable, path: &str) -> Option<Vec<u8>> {
    let size = mounts.stat(path)?.size as usize;
    let mut buf = vec![0u8; size];
    let mut off = 0;
    while off < size {
        match mounts.read_at(path, off as u64, &mut buf[off..]) {
            Ok(0) => break,
            Ok(n) => off += n,
            Err(_) => return None,
        }
    }
    buf.truncate(off);
    Some(buf)
}

fn round_up_page(v: u64) -> u64 {
    v.div_ceil(PAGE_SIZE) * PAGE_SIZE
}

fn page_align_down(v: u64) -> u64 {
    v - v % PAGE_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal static aarch64 ELF whose entry is `exit(code)`:
    /// `movz x0,#code ; movz x8,#93 ; svc #0`. One PT_LOAD maps the whole file
    /// (headers included) at the guest base; the entry points at the code.
    fn build_exit_elf(code: u16) -> Vec<u8> {
        const BASE: u64 = 0x1_0000;
        let words: [u32; 3] = [
            0xD280_0000 | (u32::from(code) << 5), // movz x0, #code
            0xD280_0BA8,                          // movz x8, #93 (exit)
            0xD400_0001,                          // svc #0
        ];
        let mut code_bytes = Vec::new();
        for w in words {
            code_bytes.extend_from_slice(&w.to_le_bytes());
        }
        let (ehsize, phsize) = (64u64, 56u64);
        let code_off = ehsize + phsize;
        let file_len = code_off + code_bytes.len() as u64;
        let entry = BASE + code_off;

        let mut e = Vec::new();
        e.extend_from_slice(&[0x7f, b'E', b'L', b'F', 2, 1, 1, 0]);
        e.extend_from_slice(&[0u8; 8]);
        e.extend_from_slice(&2u16.to_le_bytes()); // e_type = ET_EXEC
        e.extend_from_slice(&183u16.to_le_bytes()); // e_machine = AArch64
        e.extend_from_slice(&1u32.to_le_bytes()); // e_version
        e.extend_from_slice(&entry.to_le_bytes()); // e_entry
        e.extend_from_slice(&ehsize.to_le_bytes()); // e_phoff
        e.extend_from_slice(&0u64.to_le_bytes()); // e_shoff
        e.extend_from_slice(&0u32.to_le_bytes()); // e_flags
        e.extend_from_slice(&(ehsize as u16).to_le_bytes()); // e_ehsize
        e.extend_from_slice(&(phsize as u16).to_le_bytes()); // e_phentsize
        e.extend_from_slice(&1u16.to_le_bytes()); // e_phnum
        e.extend_from_slice(&0u16.to_le_bytes()); // e_shentsize
        e.extend_from_slice(&0u16.to_le_bytes()); // e_shnum
        e.extend_from_slice(&0u16.to_le_bytes()); // e_shstrndx
        e.extend_from_slice(&1u32.to_le_bytes()); // p_type = PT_LOAD
        e.extend_from_slice(&5u32.to_le_bytes()); // p_flags = R+X
        e.extend_from_slice(&0u64.to_le_bytes()); // p_offset
        e.extend_from_slice(&BASE.to_le_bytes()); // p_vaddr
        e.extend_from_slice(&BASE.to_le_bytes()); // p_paddr
        e.extend_from_slice(&file_len.to_le_bytes()); // p_filesz
        e.extend_from_slice(&file_len.to_le_bytes()); // p_memsz
        e.extend_from_slice(&0x1000u64.to_le_bytes()); // p_align
        e.extend_from_slice(&code_bytes);
        e
    }

    #[cfg(unix)]
    fn temp_root(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("nixvm-sb-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[cfg(unix)]
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
        assert_eq!(code, 42, "PID 1 read from the root and run to its exit code");
    }

    #[cfg(unix)]
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

    #[cfg(unix)]
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
        let mut mounts = sb.build_mounts().unwrap();
        preconfigure(&mut mounts);
        assert!(mounts.stat("/etc/hostname").is_some(), "/etc/hostname seeded");
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
