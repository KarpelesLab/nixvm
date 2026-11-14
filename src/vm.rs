//! Interactive VM driver — boot a root image and drive it a step at a time,
//! feeding terminal input and draining output between steps.
//!
//! This is the engine behind the browser terminal (`crate::wasm`): the page
//! unpacks a distro root image (an Alpine minirootfs `.tar`) into an in-memory
//! filesystem, starts a shell, and then pumps keystrokes in and stdout/stderr
//! out — exactly what an xterm-style widget needs. It is deliberately
//! target-independent (no wasm, no unix) so it can be tested natively against a
//! real Alpine image (see `tests/alpine_boot.rs`) before it ever runs in a tab.

use crate::abi::Arch;
use crate::fs::{DevFs, MountTable, ProcFs, SysFs, TmpFs, tar};
use crate::kernel::{Kernel, Pumped};
use crate::loader::{ProcessSpec, interp_path, load_dynamic, load_static};
use crate::vcpu::GuestMemory;
use crate::vcpu::mem::PAGE_SIZE;
use std::sync::{Arc, Mutex};

/// Guest base address for the flat process address space.
const GUEST_BASE: u64 = 0x1_0000;

/// The optional network link [`Vm::finish`] takes; uninhabited without the
/// `tunnel` feature, so `None` is the only value there.
#[cfg(feature = "tunnel")]
type Tunnel = crate::tunnel::Tunnel;
#[cfg(not(feature = "tunnel"))]
type Tunnel = std::convert::Infallible;

/// How pid 1 starts: its argv (`argv[0]` an absolute path in the root),
/// environment, working directory and ids.
struct Start {
    argv: Vec<String>,
    env: Vec<String>,
    cwd: String,
    uid: u32,
    gid: u32,
    groups: Vec<u32>,
}

impl Start {
    /// The interactive shell's: root, at `/`, with the default environment.
    fn shell(argv: Vec<String>) -> Self {
        Self {
            argv,
            env: default_env(),
            cwd: "/".into(),
            uid: 0,
            gid: 0,
            groups: Vec::new(),
        }
    }
}

/// A [`std::io::Write`] that appends into a shared byte buffer — used to
/// capture guest fd 1/2 for the terminal.
struct CaptureSink(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for CaptureSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        /// Past this much waiting, hand control back so the embedder can drain it.
        const YIELD_ABOVE: usize = 16 << 20;
        /// An embedder that never drains loses output rather than the host's memory.
        const DROP_ABOVE: usize = 256 << 20;
        let mut held = self.0.lock().unwrap();
        if held.len() < DROP_ABOVE {
            held.extend_from_slice(buf);
        }
        if held.len() > YIELD_ABOVE {
            crate::vcpu::set_yield_deadline(Some(std::time::Duration::ZERO));
        }
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// What one [`Vm::pump`] produced: any output since the last pump, and whether
/// the guest exited.
#[derive(Debug, Default)]
pub struct Step {
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    /// `Some(code)` once pid 1 exited; `None` means it is parked for input.
    pub exit_code: Option<i32>,
    /// [`Vm::pump_for`] ran out of time with the guest still computing.
    pub busy: bool,
}

/// A booted, interactively-driven guest.
#[derive(Debug)]
pub struct Vm {
    kernel: Kernel,
    /// The guest's network link when booted with one ([`Vm::boot_squashfs_net`]).
    #[cfg(feature = "tunnel")]
    tunnel: Option<crate::tunnel::Tunnel>,
    stdout: Arc<Mutex<Vec<u8>>>,
    stderr: Arc<Mutex<Vec<u8>>>,
    finished: Option<i32>,
}

impl Vm {
    /// Boot `argv[0]` (e.g. `/bin/busybox`) from the root image `tar` (an
    /// uncompressed tar unpacked into an in-memory root), in interactive mode.
    /// Handles dynamic executables (reads the `PT_INTERP` linker from the same
    /// image). `mem_bytes` caps guest RAM.
    ///
    /// # Errors
    /// Returns a message if the image lacks `argv[0]`/its interpreter, the ELF
    /// is unsupported, or a backend fails to start.
    pub fn boot(tar_bytes: &[u8], argv: Vec<String>, mem_bytes: u64) -> Result<Self, String> {
        let mut mounts = MountTable::new();
        mounts.mount("/", Box::new(TmpFs::new()));
        tar::extract_into(&mut mounts, tar_bytes);
        Self::finish(mounts, Start::shell(argv), mem_bytes, None)
    }

    /// Boot from a `.tar` root image mounted as a **read-only squashfs lower**
    /// under a writable tmpfs upper (copy-on-write) — the real overlay layout.
    /// The tar is repacked into an in-memory squashfs via `fstool`, so the guest
    /// root stays compressed in RAM and decompresses on demand.
    ///
    /// # Errors
    /// Returns a message if the squashfs can't be built/opened, or as [`Vm::boot`].
    #[cfg(feature = "fstool")]
    pub fn boot_squashfs(
        tar_bytes: &[u8],
        argv: Vec<String>,
        mem_bytes: u64,
    ) -> Result<Self, String> {
        Self::boot_squashfs_with(tar_bytes, argv, Vec::new(), mem_bytes, None)
    }

    /// [`Vm::boot_squashfs`] with the guest's network carried over `tunnel`:
    /// it becomes the kernel's egress backend and the guest gets a
    /// `/etc/resolv.conf`. The embedder moves packets through
    /// [`Vm::tunnel`] (see [`crate::tunnel`]).
    ///
    /// # Errors
    /// As [`Vm::boot_squashfs`].
    #[cfg(all(feature = "fstool", feature = "tunnel"))]
    pub fn boot_squashfs_net(
        tar_bytes: &[u8],
        argv: Vec<String>,
        mem_bytes: u64,
        tunnel: crate::tunnel::Tunnel,
    ) -> Result<Self, String> {
        Self::boot_squashfs_with(tar_bytes, argv, Vec::new(), mem_bytes, Some(tunnel))
    }

    /// [`Vm::boot_squashfs`] with `env` (`KEY=VALUE`) added over the shell's
    /// default environment — e.g. [`display_env`].
    ///
    /// # Errors
    /// As [`Vm::boot_squashfs`].
    #[cfg(feature = "fstool")]
    pub fn boot_squashfs_env(
        tar_bytes: &[u8],
        argv: Vec<String>,
        env: Vec<String>,
        mem_bytes: u64,
    ) -> Result<Self, String> {
        Self::boot_squashfs_with(tar_bytes, argv, env, mem_bytes, None)
    }

    /// [`Vm::boot_squashfs_net`] with `env` (`KEY=VALUE`) added over the
    /// shell's default environment — e.g. [`display_env`] for a guest that
    /// talks to a [`Vm::attach_display`] compositor.
    ///
    /// # Errors
    /// As [`Vm::boot_squashfs`].
    #[cfg(all(feature = "fstool", feature = "tunnel"))]
    pub fn boot_squashfs_net_env(
        tar_bytes: &[u8],
        argv: Vec<String>,
        env: Vec<String>,
        mem_bytes: u64,
        tunnel: crate::tunnel::Tunnel,
    ) -> Result<Self, String> {
        Self::boot_squashfs_with(tar_bytes, argv, env, mem_bytes, Some(tunnel))
    }

    #[cfg(feature = "fstool")]
    fn boot_squashfs_with(
        tar_bytes: &[u8],
        argv: Vec<String>,
        env: Vec<String>,
        mem_bytes: u64,
        tunnel: Option<Tunnel>,
    ) -> Result<Self, String> {
        use crate::fs::{FsToolMount, Overlay};
        let lower = FsToolMount::from_tar(tar_bytes)
            .map_err(|e| format!("build squashfs from rootfs tar: {e}"))?;
        let mut mounts = MountTable::new();
        mounts.mount(
            "/",
            Box::new(Overlay::new(Box::new(lower), Box::new(TmpFs::new()))),
        );
        let mut start = Start::shell(argv);
        merge_env(&mut start.env, env);
        Self::finish(mounts, start, mem_bytes, tunnel)
    }

    /// Boot a Docker/OCI image: its layers flattened into a read-only root
    /// (an in-memory squashfs) under a writable tmpfs, and its entrypoint run
    /// as `docker run` would — command, environment, working directory and
    /// user from the image, overridden by `opts`. Drive it like any [`Vm`].
    ///
    /// # Errors
    /// A message if the layers don't make a root, the image names nothing to
    /// run or a program or user it lacks, or as [`Vm::boot`].
    #[cfg(feature = "oci")]
    pub fn boot_image(
        image: &crate::oci::Image,
        opts: &crate::oci::RunOptions,
        mem_bytes: u64,
    ) -> Result<Self, String> {
        Self::boot_image_with(image, opts, mem_bytes, None)
    }

    /// [`Vm::boot_image`] with the guest's network carried over `tunnel`, as
    /// [`Vm::boot_squashfs_net`].
    ///
    /// # Errors
    /// As [`Vm::boot_image`].
    #[cfg(all(feature = "oci", feature = "tunnel"))]
    pub fn boot_image_net(
        image: &crate::oci::Image,
        opts: &crate::oci::RunOptions,
        mem_bytes: u64,
        tunnel: crate::tunnel::Tunnel,
    ) -> Result<Self, String> {
        Self::boot_image_with(image, opts, mem_bytes, Some(tunnel))
    }

    #[cfg(feature = "oci")]
    fn boot_image_with(
        image: &crate::oci::Image,
        opts: &crate::oci::RunOptions,
        mem_bytes: u64,
        tunnel: Option<Tunnel>,
    ) -> Result<Self, String> {
        let mut mounts = image_root(image)?;
        let l = image.launch(opts, &mut mounts).map_err(|e| e.to_string())?;
        let start = Start {
            argv: l.argv,
            env: l.env,
            cwd: l.cwd,
            uid: l.uid,
            gid: l.gid,
            groups: l.groups,
        };
        Self::finish(mounts, start, mem_bytes, tunnel)
    }

    /// Mount the synthetic filesystems onto `mounts`, load `argv[0]` (following
    /// its `PT_INTERP` linker for dynamic executables), and boot it interactively.
    fn finish(
        mut mounts: MountTable,
        start: Start,
        mem_bytes: u64,
        tunnel: Option<Tunnel>,
    ) -> Result<Self, String> {
        let Start {
            argv,
            mut env,
            cwd,
            uid,
            gid,
            groups,
        } = start;
        if argv.is_empty() {
            return Err("empty argv".into());
        }
        let argv = resolve_script(&mut mounts, argv)?;
        let path = argv[0].clone();

        mounts.mount("/tmp", Box::new(TmpFs::new()));
        mounts.mount("/dev", Box::new(DevFs::new()));
        mounts.mount("/proc", Box::new(ProcFs::new(1)));
        mounts.mount("/sys", Box::new(SysFs::new(1)));
        mounts.mount(
            "/sys/fs/cgroup",
            Box::new(crate::fs::CgroupFs::new(
                crate::fs::CgroupTree::shared(),
                crate::fs::cgroupfs::ROOT,
            )),
        );
        install_runtime_dir(&mut mounts, &mut env, uid, gid);

        // With host egress enabled, give the guest resolver a nameserver
        // (a stock Alpine minirootfs ships no /etc/resolv.conf, so musl would
        // otherwise default to 127.0.0.1 and every lookup would fail).
        if egress_enabled() || tunnel.is_some() {
            install_resolv_conf(&mut mounts);
        }

        let elf = mounts
            .read_file(&path)
            .ok_or_else(|| format!("{path} not found in the root image"))?;
        let arch = Arch::from_elf(&elf).ok_or("unrecognized ELF machine type")?;

        // A large virtual extent over a `mem_bytes` physical pool: runtimes
        // reserve far more address space than they touch (Go's page-summary
        // and arena reservations need ~1 GiB, JS engines tens of GiB), and
        // only touched pages take a frame.
        let mem_bytes = round_up_page(mem_bytes);
        let mut mem = GuestMemory::new_split(GUEST_BASE, GUEST_VSIZE.max(mem_bytes), mem_bytes);
        // Page tables in the guest's own format, for its hardware MMU.
        mem.set_guest_arch(arch).map_err(|e| format!("{e:?}"))?;
        let spec = ProcessSpec {
            argv: argv.clone(),
            envp: env,
        };
        let loaded = if let Some(interp) = interp_path(&elf) {
            let interp_elf = mounts
                .read_file(&interp)
                .ok_or_else(|| format!("dynamic linker {interp} not found in the root image"))?;
            load_dynamic(&mut mem, &elf, &interp_elf, &spec)
        } else {
            load_static(&mut mem, &elf, &spec)
        };
        let img = loaded.map_err(|e| e.to_string())?;
        let mid = page_down(img.program_break + (img.stack_bottom - img.program_break) / 2);

        let backend = select_backend(arch)?;
        let vcpu = backend
            .new_vcpu(img.entry, img.stack_pointer)
            .map_err(|e| e.to_string())?;
        // Let the guest's vDSO read the clock without a syscall where the
        // backend can calibrate it (hardware; a no-op on the interpreter).
        crate::sandbox::calibrate_vdso(vcpu.as_ref(), &mem);

        let stdout = Arc::new(Mutex::new(Vec::new()));
        let stderr = Arc::new(Mutex::new(Vec::new()));
        let mut kernel = Kernel::new(arch, mounts);
        kernel.set_interactive(true);
        kernel.set_stdout(Box::new(CaptureSink(stdout.clone())));
        kernel.set_stderr(Box::new(CaptureSink(stderr.clone())));
        // The shell starts at `/` so the demo's hard-coded `/ $` prompt is
        // accurate and a bare `ls` lists the populated root (Alpine's /root
        // home is empty, which made `ls` look like it did nothing); an image
        // starts in its working directory.
        kernel.set_cwd(cwd);
        kernel.set_creds(uid, gid, &groups);
        kernel.set_exe(&path);
        kernel.set_cmdline(argv);
        kernel.set_heap(img.program_break, mid);
        kernel.set_mmap_area(img.stack_bottom, mid);
        install_egress(&mut kernel);
        install_gpu(&kernel);
        #[cfg(feature = "tunnel")]
        if let Some(t) = &tunnel {
            kernel.set_egress(t.egress());
        }
        kernel.boot(vcpu, mem);

        Ok(Self {
            kernel,
            #[cfg(feature = "tunnel")]
            tunnel,
            stdout,
            stderr,
            finished: None,
        })
    }

    /// Serve the guest a graphical display: `display` listens on
    /// [`crate::display::DEFAULT_SOCKET`] and runs as one of the kernel's
    /// services from the next pump on. Boot with [`display_env`] in the
    /// environment so the guest's Wayland clients find it.
    ///
    /// # Errors
    /// A message if the socket can't be made (one already exists there).
    pub fn attach_display(&self, display: &crate::display::Display) -> Result<(), String> {
        display
            .attach(&self.kernel, crate::display::DEFAULT_SOCKET)
            .map_err(|e| format!("attach the display: errno {}", -e))
    }

    /// Feed terminal input (keystrokes) to the guest's stdin.
    /// Give the guest the GPU `dev` (`/dev/dri/renderD128`).
    ///
    /// # Errors
    /// If it already has one.
    #[cfg(feature = "gpu")]
    pub fn attach_gpu(&self, dev: crate::gpu::GpuDevice) -> Result<(), String> {
        self.kernel
            .attach_gpu(dev)
            .map_err(|_| "a GPU is already attached".to_string())
    }

    pub fn write_stdin(&mut self, bytes: &[u8]) {
        self.kernel.feed_stdin(bytes);
    }

    /// Signal end-of-input (Ctrl-D).
    pub fn close_stdin(&mut self) {
        self.kernel.close_stdin();
    }

    /// Like [`Vm::pump`] but returning after about `budget` even if the guest
    /// is still computing (then [`Step::busy`] is set: pump again soon). For
    /// a single-threaded embedder that must keep its event loop alive.
    ///
    /// # Errors
    /// As [`Vm::pump`].
    pub fn pump_for(&mut self, budget: std::time::Duration) -> Result<Step, String> {
        let mut busy = false;
        if self.finished.is_none() {
            match self.kernel.pump_for(budget).map_err(|e| e.to_string())? {
                Pumped::Exited(code) => self.finished = Some(code),
                Pumped::Blocked => {}
                Pumped::Busy => busy = true,
            }
        }
        Ok(Step {
            stdout: std::mem::take(&mut *self.stdout.lock().unwrap()),
            stderr: std::mem::take(&mut *self.stderr.lock().unwrap()),
            exit_code: self.finished,
            busy,
        })
    }

    /// The guest's network link, when booted with one.
    #[cfg(feature = "tunnel")]
    #[must_use]
    pub fn tunnel(&self) -> Option<&crate::tunnel::Tunnel> {
        self.tunnel.as_ref()
    }

    /// Replace the VM-wide resource ceilings, which default to
    /// [`Limits::for_this_host`](crate::limits::Limits::for_this_host).
    pub fn set_limits(&mut self, limits: crate::limits::Limits) {
        self.kernel.set_limits(limits);
    }

    /// Refuse syscalls to every guest task, whatever it does (see
    /// [`crate::policy`]).
    pub fn set_syscall_policy(&mut self, policy: &crate::policy::SyscallPolicy) {
        self.kernel.set_syscall_policy(policy);
    }

    /// The syscalls the policy has refused lately, oldest first.
    #[must_use]
    pub fn denials(&self) -> Vec<crate::policy::Denial> {
        self.kernel.denials()
    }

    /// Whether the guest is parked reading the terminal, i.e. the last command
    /// finished and it wants input (see [`Kernel::awaiting_input`]).
    #[must_use]
    pub fn awaiting_input(&self) -> bool {
        self.kernel.awaiting_input()
    }

    /// Whether a parked guest will progress without input — a timer or the
    /// network — so the embedder should pump again soon.
    #[must_use]
    pub fn has_pending_work(&self) -> bool {
        self.finished.is_none() && self.kernel.has_pending_work()
    }

    /// Syscalls the guest attempted that nixvm doesn't implement: raw syscall
    /// number -> attempt count (see [`Kernel::unsupported`]).
    #[must_use]
    pub fn unsupported_syscalls(&self) -> std::collections::BTreeMap<u64, u64> {
        self.kernel.unsupported()
    }

    /// Subcommands of known syscalls that no handler recognized — `ioctl`
    /// requests, `fcntl`/`prctl` commands, socket options, … — as
    /// `(syscall name, subcommand) -> attempt count` (see
    /// [`Kernel::unsupported_subcommands`]).
    #[must_use]
    pub fn unsupported_subcommands(&self) -> std::collections::BTreeMap<(&'static str, u64), u64> {
        self.kernel.unsupported_subcommands()
    }

    /// Ctrl-C: interrupt the running command (see [`Kernel::interrupt`]).
    /// Returns whether a command was running. Pump afterwards.
    pub fn interrupt(&mut self) -> bool {
        self.kernel.interrupt()
    }

    /// Whether pid 1 has exited (with its code).
    #[must_use]
    pub fn exit_code(&self) -> Option<i32> {
        self.finished
    }

    /// Run the guest until it exits or parks for input, returning any output it
    /// produced. Call again after [`Vm::write_stdin`] to continue.
    ///
    /// # Errors
    /// Propagates a backend/scheduler failure (a guest fault ends the run via
    /// `exit_code`, not an error).
    pub fn pump(&mut self) -> Result<Step, String> {
        let mut busy = false;
        if self.finished.is_none() {
            match self.kernel.pump().map_err(|e| e.to_string())? {
                Pumped::Exited(code) => self.finished = Some(code),
                Pumped::Blocked => {}
                Pumped::Busy => busy = true,
            }
        }
        Ok(Step {
            stdout: std::mem::take(&mut *self.stdout.lock().unwrap()),
            stderr: std::mem::take(&mut *self.stderr.lock().unwrap()),
            exit_code: self.finished,
            busy,
        })
    }
}

/// Whether host-network egress is requested (`NIXVM_NET=host`) and available
/// (native only — the browser has no raw sockets, so this is always `false` on
/// wasm regardless of the env var; a WebSocket/pktkit transport will hook in
/// here later).
pub(crate) fn egress_enabled() -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        std::env::var_os("NIXVM_NET").is_some_and(|v| v == "host")
    }
    #[cfg(target_arch = "wasm32")]
    {
        false
    }
}

/// Install the host egress backend on `kernel` when [`egress_enabled`].
pub(crate) fn install_egress(kernel: &mut Kernel) {
    #[cfg(not(target_arch = "wasm32"))]
    if egress_enabled() {
        kernel.set_egress(Box::new(crate::kernel::egress::HostEgress));
    }
    #[cfg(target_arch = "wasm32")]
    let _ = kernel;
}

/// Attach the GPU `NIXVM_GPU` asks for (`egl`, `log`, `null`), when nixvm is
/// built with the `gpu` feature; otherwise, and by default, there is none.
pub(crate) fn install_gpu(kernel: &Kernel) {
    #[cfg(feature = "gpu")]
    if let Some(dev) = crate::gpu::GpuDevice::from_env() {
        let _ = kernel.attach_gpu(dev);
    }
    #[cfg(not(feature = "gpu"))]
    let _ = kernel;
}

/// An image's root: its layers flattened into an in-memory squashfs, read-only
/// under a writable tmpfs, with the `/etc/hostname` and `/etc/hosts` a
/// container runtime puts in where the image has none.
#[cfg(feature = "oci")]
pub(crate) fn image_root(image: &crate::oci::Image) -> Result<MountTable, String> {
    use crate::fs::{FsToolMount, Overlay};
    let tar = image.rootfs_tar().map_err(|e| e.to_string())?;
    let lower = FsToolMount::from_tar(&tar).map_err(|e| format!("build the image's root: {e}"))?;
    drop(tar);
    let mut mounts = MountTable::new();
    mounts.mount(
        "/",
        Box::new(Overlay::new(Box::new(lower), Box::new(TmpFs::new()))),
    );
    if mounts.stat("/etc").is_none() {
        let _ = mounts.mkdir("/etc", 0o755);
    }
    for (path, content) in [
        ("/etc/hostname", "nixvm\n"),
        (
            "/etc/hosts",
            "127.0.0.1\tlocalhost nixvm\n::1\tlocalhost ip6-localhost ip6-loopback\n",
        ),
    ] {
        if mounts.stat(path).is_none() && mounts.create(path, 0o644).is_ok() {
            let _ = mounts.write_at(path, 0, content.as_bytes());
        }
    }
    Ok(mounts)
}

/// PID 1's argv once a `#!` script is replaced by its interpreter, as
/// `execve` does it (Linux's binfmt_script): `[interpreter, its optional
/// argument, script, args…]`, up to 4 levels deep. A container image's
/// entrypoint is often a shell script.
///
/// # Errors
/// A message for a script whose interpreter line is empty, or too deep.
pub(crate) fn resolve_script(
    mounts: &mut MountTable,
    mut argv: Vec<String>,
) -> Result<Vec<String>, String> {
    for _ in 0..5 {
        let Some(path) = argv.first().cloned() else {
            return Err("empty argv".into());
        };
        let Some(target) = mounts.follow(&path) else {
            return Ok(argv);
        };
        let mut head = [0u8; 256];
        let n = mounts.read_at(&target, 0, &mut head).unwrap_or(0);
        let head = &head[..n];
        if !head.starts_with(b"#!") {
            return Ok(argv);
        }
        let line = head[2..].split(|&b| b == b'\n').next().unwrap_or_default();
        let line = String::from_utf8_lossy(line);
        let line = line.trim();
        let (interp, arg) = match line.split_once([' ', '\t']) {
            Some((i, a)) => (i, Some(a.trim()).filter(|a| !a.is_empty())),
            None => (line, None),
        };
        if interp.is_empty() {
            return Err(format!("{path}: a #! line with no interpreter"));
        }
        let mut next = vec![interp.to_string()];
        next.extend(arg.map(str::to_string));
        next.push(path);
        next.extend(argv.into_iter().skip(1));
        argv = next;
    }
    Err("#! interpreters nest too deep".into())
}

/// Write a minimal `/etc/resolv.conf` (public resolvers) into the guest root,
/// so musl's DNS has somewhere to send queries when egress is on.
fn install_resolv_conf(mounts: &mut MountTable) {
    const RESOLV: &[u8] = b"nameserver 1.1.1.1\nnameserver 8.8.8.8\n";
    if mounts.stat("/etc").is_none() {
        let _ = mounts.mkdir("/etc", 0o755);
    }
    // Overwrite any existing file: create is a no-op if present, so truncate
    // via a fresh create then write.
    let _ = mounts.create("/etc/resolv.conf", 0o644);
    let _ = mounts.write_at("/etc/resolv.conf", 0, RESOLV);
}

fn select_backend(arch: Arch) -> Result<Box<dyn crate::vcpu::Backend>, String> {
    // Prefer hardware virtualization where the host has it (KVM/HVF, probed
    // with an interpreter fallback — see `vcpu::select`); on wasm32 the
    // hardware backends don't build, so this always lands on the interpreter
    // there.
    crate::vcpu::select(arch).map_err(|e| e.to_string())
}

/// The environment a guest needs to reach the display
/// [`Vm::attach_display`] serves: `WAYLAND_DISPLAY` naming its socket. The
/// boot adds an `XDG_RUNTIME_DIR` (made 0700 for the guest's user) to an
/// environment that has `WAYLAND_DISPLAY` but none.
#[must_use]
pub fn display_env() -> Vec<String> {
    vec![format!(
        "WAYLAND_DISPLAY={}",
        crate::display::DEFAULT_SOCKET
    )]
}

/// Add `extra` (`KEY=VALUE`) to `env`, replacing the entries it names.
#[cfg(feature = "fstool")]
fn merge_env(env: &mut Vec<String>, extra: Vec<String>) {
    let key = |e: &str| e.split_once('=').map_or(e, |(k, _)| k).to_string();
    for e in extra {
        let k = key(&e);
        env.retain(|o| key(o) != k);
        env.push(e);
    }
}

/// Give a guest that talks Wayland its `XDG_RUNTIME_DIR`: where clients
/// keep their sockets and shared-memory files, and which libraries expect
/// to exist, private (0700) to the user. Defaults to `/run/user/<uid>` when
/// `env` has `WAYLAND_DISPLAY` but no `XDG_RUNTIME_DIR`; a directory named
/// but missing is made.
fn install_runtime_dir(mounts: &mut MountTable, env: &mut Vec<String>, uid: u32, gid: u32) {
    let var = |env: &[String], key: &str| {
        env.iter()
            .find_map(|e| e.strip_prefix(key)?.strip_prefix('=').map(str::to_string))
    };
    let dir = match var(env, "XDG_RUNTIME_DIR") {
        Some(d) => d,
        None if var(env, "WAYLAND_DISPLAY").is_some() => {
            let d = format!("/run/user/{uid}");
            env.push(format!("XDG_RUNTIME_DIR={d}"));
            d
        }
        None => return,
    };
    if !dir.starts_with('/') || mounts.stat(&dir).is_some() {
        return;
    }
    let mut at = String::new();
    for part in dir.split('/').filter(|p| !p.is_empty()) {
        at.push('/');
        at.push_str(part);
        if mounts.stat(&at).is_none() && mounts.mkdir(&at, 0o755).is_err() {
            return;
        }
    }
    let _ = mounts.set_mode(&dir, 0o700);
    let _ = mounts.set_owner(&dir, Some(uid), Some(gid));
}

fn default_env() -> Vec<String> {
    vec![
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin".to_string(),
        "HOME=/root".to_string(),
        "TERM=xterm-256color".to_string(),
        "PS1=nixvm:\\w\\$ ".to_string(),
        "PWD=/".to_string(),
        "USER=root".to_string(),
        "HOSTNAME=nixvm".to_string(),
    ]
}

/// Virtual extent of a [`Vm`]'s address space (see `sandbox::GUEST_VSIZE`):
/// its per-page metadata is sparse, so only what's touched costs anything.
const GUEST_VSIZE: u64 = 1 << 40;

fn round_up_page(v: u64) -> u64 {
    v.div_ceil(PAGE_SIZE) * PAGE_SIZE
}

fn page_down(v: u64) -> u64 {
    v - v % PAGE_SIZE
}
