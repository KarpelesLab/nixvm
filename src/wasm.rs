//! Browser demo entry point.
//!
//! `run_elf(bytes: &[u8]) -> String` loads a statically-linked ELF64 image,
//! runs it to completion on the software interpreter ([`crate::vcpu::interp`]),
//! and returns a JSON string with the guest's captured stdout/stderr and exit
//! code, for `web/index.html` to print into a `<pre>`.
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
//! talks to [`crate::loader`], [`crate::vcpu::interp::InterpBackend`], and
//! [`Kernel`] directly — mirroring the wiring in `src/bin/run-elf.rs`.
//!
//! There is also no host filesystem to share into a browser tab, so the guest
//! root here is an in-memory [`crate::fs::TmpFs`] only (no `/work` passthrough
//! — that mount is `cfg(unix)`-gated and wouldn't build for wasm32 anyway).

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
    use crate::vcpu::interp::InterpBackend;
    use crate::vcpu::mem::PAGE_SIZE;

    /// Guest base address for the flat process address space (mirrors
    /// `sandbox::GUEST_BASE` / `run-elf`'s constant of the same value).
    const GUEST_BASE: u64 = 0x1_0000;
    /// Guest RAM ceiling for the browser demo: kept small since it backs a
    /// `Vec<u8>` in the wasm linear memory, not a real mmap.
    const MEM_BYTES: u64 = 256 * 1024 * 1024;

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

    /// Run a statically-linked ELF64 image on the software interpreter to
    /// completion.
    ///
    /// Returns a JSON string:
    /// `{"ok":true,"exit_code":<i32>,"stdout":<str>,"stderr":<str>,"error":null}`
    /// on a completed run (the guest's own exit code may be nonzero — that's
    /// still `ok:true`), or
    /// `{"ok":false,"exit_code":null,"stdout":"","stderr":"","error":<str>}`
    /// if the interpreter itself couldn't load or run the image (bad ELF,
    /// unsupported machine type, a fault, …).
    #[wasm_bindgen]
    #[must_use]
    pub fn run_elf(bytes: &[u8]) -> String {
        console_error_panic_hook::set_once();
        match run(bytes) {
            Ok((code, out, err)) => format!(
                "{{\"ok\":true,\"exit_code\":{code},\"stdout\":\"{}\",\"stderr\":\"{}\",\"error\":null}}",
                json_escape(&out),
                json_escape(&err)
            ),
            Err(msg) => format!(
                "{{\"ok\":false,\"exit_code\":null,\"stdout\":\"\",\"stderr\":\"\",\"error\":\"{}\"}}",
                json_escape(&msg)
            ),
        }
    }

    fn run(elf: &[u8]) -> Result<(i32, String, String), String> {
        // The interpreter (src/vcpu/interp.rs) targets aarch64 today; the
        // browser demo runs aarch64 static binaries.
        let arch = Arch::Aarch64;

        let mut mem = GuestMemory::new(GUEST_BASE, MEM_BYTES);
        let spec = ProcessSpec {
            argv: vec!["prog".to_string()],
            envp: vec![
                "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin".to_string(),
                "HOME=/root".to_string(),
                "TERM=xterm".to_string(),
                "PWD=/".to_string(),
            ],
        };
        let img = load_static(&mut mem, elf, &spec).map_err(|e| e.to_string())?;

        // Same heap/mmap midpoint split as sandbox::exec_elf / run-elf.
        let mid = page_down(img.program_break + (img.stack_bottom - img.program_break) / 2);

        let backend = InterpBackend::new(arch).map_err(|e| e.to_string())?;
        let vcpu = backend
            .new_vcpu(img.entry, img.stack_pointer)
            .map_err(|e| e.to_string())?;

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

        let code = kernel.run(vcpu, mem).map_err(|e| e.to_string())?;

        let stdout = String::from_utf8_lossy(&stdout_buf.lock().unwrap()).into_owned();
        let stderr = String::from_utf8_lossy(&stderr_buf.lock().unwrap()).into_owned();
        Ok((code, stdout, stderr))
    }

    fn page_down(v: u64) -> u64 {
        v - v % PAGE_SIZE
    }
}

#[cfg(target_arch = "wasm32")]
pub use browser::run_elf;
