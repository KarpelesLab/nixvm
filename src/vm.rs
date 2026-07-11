//! Interactive VM driver — boot a root image and drive it a step at a time,
//! feeding terminal input and draining output between steps.
//!
//! This is the engine behind the browser terminal ([`crate::wasm`]): the page
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

/// A [`std::io::Write`] that appends into a shared byte buffer — used to
/// capture guest fd 1/2 for the terminal.
struct CaptureSink(Arc<Mutex<Vec<u8>>>);

impl std::io::Write for CaptureSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
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
}

/// A booted, interactively-driven guest.
#[derive(Debug)]
pub struct Vm {
    kernel: Kernel,
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
        if argv.is_empty() {
            return Err("empty argv".into());
        }
        let path = argv[0].clone();

        let mut mounts = MountTable::new();
        mounts.mount("/", Box::new(TmpFs::new()));
        tar::extract_into(&mut mounts, tar_bytes);
        mounts.mount("/tmp", Box::new(TmpFs::new()));
        mounts.mount("/dev", Box::new(DevFs::new()));
        mounts.mount("/proc", Box::new(ProcFs::new()));
        mounts.mount("/sys", Box::new(SysFs::new()));

        let elf = read_mount_file(&mut mounts, &path)
            .ok_or_else(|| format!("{path} not found in the root image"))?;
        let arch = detect_arch(&elf).ok_or("unrecognized ELF machine type")?;

        let mut mem = GuestMemory::new(GUEST_BASE, round_up_page(mem_bytes));
        let spec = ProcessSpec {
            argv,
            envp: default_env(),
        };
        let loaded = if let Some(interp) = interp_path(&elf) {
            let interp_elf = read_mount_file(&mut mounts, &interp)
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

        let stdout = Arc::new(Mutex::new(Vec::new()));
        let stderr = Arc::new(Mutex::new(Vec::new()));
        let mut kernel = Kernel::new(arch, mounts);
        kernel.set_interactive(true);
        kernel.set_stdout(Box::new(CaptureSink(stdout.clone())));
        kernel.set_stderr(Box::new(CaptureSink(stderr.clone())));
        kernel.set_cwd("/root");
        kernel.set_heap(img.program_break, mid);
        kernel.set_mmap_area(img.stack_bottom, mid);
        kernel.boot(vcpu, mem);

        Ok(Self {
            kernel,
            stdout,
            stderr,
            finished: None,
        })
    }

    /// Feed terminal input (keystrokes) to the guest's stdin.
    pub fn write_stdin(&mut self, bytes: &[u8]) {
        self.kernel.feed_stdin(bytes);
    }

    /// Signal end-of-input (Ctrl-D).
    pub fn close_stdin(&mut self) {
        self.kernel.close_stdin();
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
        if self.finished.is_none() {
            match self.kernel.pump().map_err(|e| e.to_string())? {
                Pumped::Exited(code) => self.finished = Some(code),
                Pumped::Blocked => {}
            }
        }
        Ok(Step {
            stdout: std::mem::take(&mut *self.stdout.lock().unwrap()),
            stderr: std::mem::take(&mut *self.stderr.lock().unwrap()),
            exit_code: self.finished,
        })
    }
}

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

/// Peek the ELF header's `e_machine` (offset 18) to pick an arch.
fn detect_arch(elf: &[u8]) -> Option<Arch> {
    const EM_X86_64: u16 = 62;
    const EM_AARCH64: u16 = 183;
    match u16::from_le_bytes([*elf.get(18)?, *elf.get(19)?]) {
        EM_AARCH64 => Some(Arch::Aarch64),
        EM_X86_64 => Some(Arch::X86_64),
        _ => None,
    }
}

fn select_backend(arch: Arch) -> Result<Box<dyn crate::vcpu::Backend>, String> {
    use crate::vcpu::Backend;
    match arch {
        Arch::Aarch64 => crate::vcpu::interp::InterpBackend::new(arch)
            .map(|b| Box::new(b) as Box<dyn Backend>)
            .map_err(|e| e.to_string()),
        Arch::X86_64 => crate::vcpu::interp_x86::X86Backend::new(arch)
            .map(|b| Box::new(b) as Box<dyn Backend>)
            .map_err(|e| e.to_string()),
    }
}

fn default_env() -> Vec<String> {
    vec![
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin".to_string(),
        "HOME=/root".to_string(),
        "TERM=xterm-256color".to_string(),
        "PS1=nixvm:\\w\\$ ".to_string(),
        "PWD=/root".to_string(),
        "USER=root".to_string(),
        "HOSTNAME=nixvm".to_string(),
    ]
}

fn round_up_page(v: u64) -> u64 {
    v.div_ceil(PAGE_SIZE) * PAGE_SIZE
}

fn page_down(v: u64) -> u64 {
    v - v % PAGE_SIZE
}
