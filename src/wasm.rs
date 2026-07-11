//! Browser demo entry point.
//!
//! [`run_elf`] loads a statically-linked ELF64 image (aarch64 or x86-64,
//! auto-detected from the ELF header's `e_machine` field), runs it to
//! completion on the matching software interpreter
//! ([`crate::vcpu::interp`] for aarch64, [`crate::vcpu::interp_x86`] for
//! x86-64), and returns a JSON string with the guest's captured
//! stdout/stderr, exit code, and unsupported-syscall ledger, for
//! `web/index.html` to render. [`run_elf_with_argv`] is the same but lets the
//! caller supply `argv`.
//!
//! ## Manual verification
//!
//! `#[wasm_bindgen]` requires the `wasm32-unknown-unknown` target and
//! `wasm-pack`; neither runs under `cargo test`, so this module has no
//! automated test. To check it by hand:
//!
//! ```text
//! rustup target add wasm32-unknown-unknown
//! cargo install wasm-pack
//! wasm-pack build --target web --no-default-features --features wasm
//! # serve the repo root (module scripts need http(s)://, not file://), e.g.:
//! python3 -m http.server 8080
//! # then open http://localhost:8080/web/index.html, pick a static ELF, run it.
//! ```
//!
//! `.github/workflows/pages.yml` runs the same `wasm-pack build` in CI and
//! deploys `web/` + the `pkg/` output to GitHub Pages.
//!
//! ## Capturing guest stdout/stderr
//!
//! [`Kernel`] already exposes [`Kernel::set_stdout`]/[`Kernel::set_stderr`]
//! (see `src/kernel/mod.rs`) to redirect guest fd 1/2 away from the host's
//! real stdout/stderr — which don't exist in a browser tab — into an
//! in-memory sink. No new kernel seam was needed for this.
//!
//! [`crate::sandbox::Sandbox::exec_elf`] does not expose that hook (it always
//! wires the real host stdout/stderr), so this module bypasses `Sandbox` and
//! talks to [`crate::loader`], the [`crate::vcpu`] backends, and [`Kernel`]
//! directly — mirroring the wiring in `src/bin/run-elf.rs`.
//!
//! There is also no host filesystem to share into a browser tab, so the guest
//! root here is an in-memory [`crate::fs::TmpFs`] only (no `/work` passthrough
//! — that mount is `cfg(unix)`-gated and wouldn't build for wasm32 anyway).
//!
//! ## Diagnostics surfaced to the page
//!
//! Besides stdout/stderr/exit code, the JSON includes the guest architecture
//! nixvm detected and [`Kernel::unsupported`]'s ledger of syscalls the guest
//! attempted that nixvm doesn't implement (raw syscall number -> attempt
//! count) — there is no public step/instruction counter on [`Kernel`] to
//! surface a wall-independent step count, so this is the richest telemetry
//! available through its public API today.

#[cfg(target_arch = "wasm32")]
mod browser {
    use std::sync::{Arc, Mutex};

    use wasm_bindgen::prelude::*;

    use crate::abi::Arch;
    use crate::fs::{DevFs, MountTable, ProcFs, SysFs, TmpFs};
    use crate::kernel::Kernel;
    use crate::loader::{ProcessSpec, load_static};
    use crate::vcpu::Backend;
    use crate::vcpu::GuestMemory;
    use crate::vcpu::mem::PAGE_SIZE;

    /// Guest base address for the flat process address space (mirrors
    /// `sandbox::GUEST_BASE` / `run-elf`'s constant of the same value).
    const GUEST_BASE: u64 = 0x1_0000;
    /// Guest RAM ceiling for the browser demo: kept small since it backs a
    /// `Vec<u8>` in the wasm linear memory, not a real mmap.
    const MEM_BYTES: u64 = 256 * 1024 * 1024;
    /// Decompression-bomb guard for the `.tar.gz` root image (a minirootfs is
    /// well under this uncompressed).
    const MAX_ROOTFS_BYTES: u64 = 512 * 1024 * 1024;

    /// A [`std::io::Write`] sink that appends into a shared, lock-guarded byte
    /// buffer. Used in place of the host's real stdout/stderr to capture guest
    /// fd 1/2 for display on the page.
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

    /// Minimal JSON string escaping (no external JSON dep in the core crate).
    fn json_escape(s: &str) -> String {
        let mut out = String::with_capacity(s.len() + 2);
        for c in s.chars() {
            match c {
                '"' => out.push_str("\\\""),
                '\\' => out.push_str("\\\\"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                '\t' => out.push_str("\\t"),
                c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
                c => out.push(c),
            }
        }
        out
    }

    /// Everything gathered while attempting to run a guest image, in a shape
    /// that always serializes — even a load failure before a single guest
    /// instruction ran still reports what we know (e.g. the detected arch).
    struct RunOutcome {
        /// Detected guest architecture, once the ELF header could be parsed.
        arch: Option<&'static str>,
        /// Guest exit code, set only once the guest reached `exit`/`exit_group`
        /// (or the whole vcpu loop otherwise returned cleanly).
        exit_code: Option<i32>,
        stdout: String,
        stderr: String,
        /// Set when some stage failed: a bad ELF, an unsupported machine type,
        /// a backend that couldn't start, or a guest-side fault/illegal
        /// instruction that ended the run early. `stdout`/`stderr`/`unsupported`
        /// still hold whatever the guest produced before that happened.
        error: Option<String>,
        /// Unsupported syscall ledger: (raw guest syscall nr, attempt count),
        /// straight from [`Kernel::unsupported`].
        unsupported: Vec<(u64, u64)>,
    }

    impl RunOutcome {
        fn failed(msg: String) -> Self {
            Self {
                arch: None,
                exit_code: None,
                stdout: String::new(),
                stderr: String::new(),
                error: Some(msg),
                unsupported: Vec::new(),
            }
        }

        fn to_json(&self) -> String {
            let exit_code = self
                .exit_code
                .map_or_else(|| "null".to_string(), |c| c.to_string());
            let arch = self
                .arch
                .map_or_else(|| "null".to_string(), |a| format!("\"{a}\""));
            let error = self
                .error
                .as_deref()
                .map_or_else(|| "null".to_string(), |e| format!("\"{}\"", json_escape(e)));
            let mut unsupported = String::from("[");
            for (i, (nr, count)) in self.unsupported.iter().enumerate() {
                if i > 0 {
                    unsupported.push(',');
                }
                unsupported.push_str(&format!("{{\"nr\":{nr},\"count\":{count}}}"));
            }
            unsupported.push(']');
            format!(
                "{{\"ok\":{},\"arch\":{arch},\"exit_code\":{exit_code},\"stdout\":\"{}\",\"stderr\":\"{}\",\"error\":{error},\"unsupported_syscalls\":{unsupported}}}",
                self.error.is_none(),
                json_escape(&self.stdout),
                json_escape(&self.stderr),
            )
        }
    }

    /// Run a statically-linked ELF64 image (aarch64 or x86-64) on the
    /// software interpreter to completion, with the default single-element
    /// `argv` (see [`run_elf_with_argv`] to override it).
    ///
    /// Returns a JSON string:
    /// ```text
    /// {
    ///   "ok": bool,               // true iff the guest reached a definitive exit code
    ///   "arch": "aarch64"|"x86_64"|null,
    ///   "exit_code": <i32>|null,
    ///   "stdout": <str>,
    ///   "stderr": <str>,
    ///   "error": <str>|null,      // load/backend/fault message when ok is false
    ///   "unsupported_syscalls": [{"nr": <u64>, "count": <u64>}, ...]
    /// }
    /// ```
    /// `stdout`/`stderr`/`unsupported_syscalls` are populated with whatever
    /// the guest produced even when `ok` is `false` (e.g. it faulted partway
    /// through), so the page can still show partial output.
    #[wasm_bindgen]
    #[must_use]
    pub fn run_elf(bytes: &[u8]) -> String {
        console_error_panic_hook::set_once();
        run(bytes, Vec::new()).to_json()
    }

    /// Same as [`run_elf`], but lets the caller supply `argv` (guest
    /// `argv[0]` onward). An empty `argv` falls back to the same single
    /// `"prog"` element [`run_elf`] uses.
    #[wasm_bindgen]
    #[must_use]
    pub fn run_elf_with_argv(bytes: &[u8], argv: Vec<String>) -> String {
        console_error_panic_hook::set_once();
        run(bytes, argv).to_json()
    }

    /// An interactive guest terminal: boot a distro root image (an *uncompressed*
    /// tar — the page decompresses the `.tar.gz` via `DecompressionStream`) and
    /// drive a shell. Wraps [`crate::vm::Vm`].
    ///
    /// Usage from JS (with an xterm-style widget):
    /// ```js
    /// const term = new Terminal(tarBytes, ["/bin/busybox", "sh"]);
    /// term.write_stdin(new TextEncoder().encode("uname -a\n"));
    /// const out = term.pump();        // Uint8Array of stdout+stderr
    /// xterm.write(out);
    /// if (!term.is_running()) { /* exited with term.exit_code() */ }
    /// ```
    #[wasm_bindgen]
    pub struct Terminal {
        vm: crate::vm::Vm,
    }

    #[wasm_bindgen]
    impl Terminal {
        /// Boot `argv[0]` from `rootfs_targz` (a `.tar.gz` root image) in
        /// interactive mode. The gzip is decompressed in-process via `compcol`
        /// (no browser `DecompressionStream` needed). Throws if the archive is
        /// malformed or lacks `argv[0]` / its dynamic linker.
        #[wasm_bindgen(constructor)]
        pub fn new(rootfs_targz: &[u8], argv: Vec<String>) -> Result<Terminal, JsError> {
            console_error_panic_hook::set_once();
            let argv = if argv.is_empty() {
                vec!["/bin/busybox".to_string(), "sh".to_string()]
            } else {
                argv
            };
            let tar = crate::fs::tar::gunzip(rootfs_targz, MAX_ROOTFS_BYTES)
                .map_err(|e| JsError::new(&e))?;
            let vm = crate::vm::Vm::boot(&tar, argv, MEM_BYTES).map_err(|e| JsError::new(&e))?;
            Ok(Self { vm })
        }

        /// Feed keystrokes to the guest's stdin.
        pub fn write_stdin(&mut self, bytes: &[u8]) {
            self.vm.write_stdin(bytes);
        }

        /// Signal end-of-input (Ctrl-D).
        pub fn close_stdin(&mut self) {
            self.vm.close_stdin();
        }

        /// Run until the guest parks for input or exits; returns the bytes to
        /// write to the terminal (stdout then stderr). Call again after
        /// `write_stdin`.
        #[must_use]
        pub fn pump(&mut self) -> Vec<u8> {
            match self.vm.pump() {
                Ok(step) => {
                    let mut out = step.stdout;
                    out.extend_from_slice(&step.stderr);
                    out
                }
                Err(_) => Vec::new(),
            }
        }

        /// Whether the guest is still running (has not exited).
        #[must_use]
        pub fn is_running(&self) -> bool {
            self.vm.exit_code().is_none()
        }

        /// The guest exit code once it has exited (`-1` while still running).
        #[must_use]
        pub fn exit_code(&self) -> i32 {
            self.vm.exit_code().unwrap_or(-1)
        }
    }

    /// Peek the ELF header's `e_machine` field (offset 18, 2 bytes,
    /// little-endian) to pick a guest architecture, without duplicating the
    /// loader's own (private) full header parser. Mirrors `EM_AARCH64` (183)
    /// / `EM_X86_64` (62) from `src/loader.rs`; `load_static` re-validates the
    /// full header (magic, class, endianness, ...) right after this, so a
    /// bogus file still gets a proper [`crate::loader::LoadError`] message.
    fn detect_arch(elf: &[u8]) -> Option<Arch> {
        const EM_X86_64: u16 = 62;
        const EM_AARCH64: u16 = 183;
        let machine = u16::from_le_bytes([*elf.get(18)?, *elf.get(19)?]);
        match machine {
            EM_AARCH64 => Some(Arch::Aarch64),
            EM_X86_64 => Some(Arch::X86_64),
            _ => None,
        }
    }

    /// Pick the software interpreter backend for `arch`. On wasm32 there is
    /// no hardware-virtualization backend to prefer (`hvf`/`kvm` are host-OS
    /// gated and don't build here), so this always lands on the portable
    /// interpreters — aarch64 via [`crate::vcpu::interp`], x86-64 via
    /// [`crate::vcpu::interp_x86`] — the same split [`crate::vcpu::select`]
    /// falls back to off-macOS.
    fn select_backend(arch: Arch) -> Result<Box<dyn Backend>, String> {
        match arch {
            Arch::Aarch64 => crate::vcpu::interp::InterpBackend::new(arch)
                .map(|b| Box::new(b) as Box<dyn Backend>)
                .map_err(|e| e.to_string()),
            Arch::X86_64 => crate::vcpu::interp_x86::X86Backend::new(arch)
                .map(|b| Box::new(b) as Box<dyn Backend>)
                .map_err(|e| e.to_string()),
        }
    }

    fn run(elf: &[u8], argv: Vec<String>) -> RunOutcome {
        let Some(arch) = detect_arch(elf) else {
            return RunOutcome::failed(
                "not a recognized ELF64 image (expected an EM_AARCH64 or EM_X86_64 e_machine)"
                    .to_string(),
            );
        };
        let arch_str = arch.as_str();

        let mut mem = GuestMemory::new(GUEST_BASE, MEM_BYTES);
        let argv = if argv.is_empty() {
            vec!["prog".to_string()]
        } else {
            argv
        };
        let spec = ProcessSpec {
            argv,
            envp: vec![
                "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin".to_string(),
                "HOME=/root".to_string(),
                "TERM=xterm".to_string(),
                "PWD=/".to_string(),
            ],
        };
        let img = match load_static(&mut mem, elf, &spec) {
            Ok(i) => i,
            Err(e) => {
                let mut o = RunOutcome::failed(e.to_string());
                o.arch = Some(arch_str);
                return o;
            }
        };

        // Same heap/mmap midpoint split as sandbox::exec_elf / run-elf.
        let mid = page_down(img.program_break + (img.stack_bottom - img.program_break) / 2);

        let backend = match select_backend(arch) {
            Ok(b) => b,
            Err(e) => {
                let mut o = RunOutcome::failed(e);
                o.arch = Some(arch_str);
                return o;
            }
        };
        let vcpu = match backend.new_vcpu(img.entry, img.stack_pointer) {
            Ok(v) => v,
            Err(e) => {
                let mut o = RunOutcome::failed(e.to_string());
                o.arch = Some(arch_str);
                return o;
            }
        };

        let mut mounts = MountTable::new();
        mounts.mount("/", Box::new(TmpFs::new()));
        mounts.mount("/tmp", Box::new(TmpFs::new()));
        mounts.mount("/dev", Box::new(DevFs::new()));
        mounts.mount("/proc", Box::new(ProcFs::new()));
        mounts.mount("/sys", Box::new(SysFs::new()));

        let stdout_buf = Arc::new(Mutex::new(Vec::new()));
        let stderr_buf = Arc::new(Mutex::new(Vec::new()));

        let mut kernel = Kernel::new(arch, mounts);
        kernel.set_stdout(Box::new(CaptureSink(stdout_buf.clone())));
        kernel.set_stderr(Box::new(CaptureSink(stderr_buf.clone())));
        kernel.set_cwd("/");
        kernel.set_heap(img.program_break, mid);
        kernel.set_mmap_area(img.stack_bottom, mid);

        let result = kernel.run(vcpu, mem);

        let stdout = String::from_utf8_lossy(&stdout_buf.lock().unwrap()).into_owned();
        let stderr = String::from_utf8_lossy(&stderr_buf.lock().unwrap()).into_owned();
        let unsupported = kernel
            .unsupported()
            .iter()
            .map(|(&nr, &c)| (nr, c))
            .collect();

        let (exit_code, error) = match result {
            Ok(code) => (Some(code), None),
            Err(e) => (None, Some(e.to_string())),
        };

        RunOutcome {
            arch: Some(arch_str),
            exit_code,
            stdout,
            stderr,
            error,
            unsupported,
        }
    }

    fn page_down(v: u64) -> u64 {
        v - v % PAGE_SIZE
    }
}

#[cfg(target_arch = "wasm32")]
pub use browser::{Terminal, run_elf, run_elf_with_argv};
