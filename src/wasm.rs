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
    /// How long one `Terminal::pump` runs the guest before handing the tab
    /// back to the browser: short enough to keep it responsive (a frame or
    /// two), long enough that the hand-offs cost little.
    const PUMP_SLICE: std::time::Duration = std::time::Duration::from_millis(30);

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
    /// // once !term.is_running(), term.exit_code() has the exit code
    /// ```
    ///
    /// Networking: the guest's sockets are carried over an IP-packet tunnel
    /// (see `crate::tunnel`). The page opens the transport (grouterd's
    /// WebSocket), calls `net_up` with the leased addresses from its hello,
    /// feeds each received packet to `net_input`, sends every packet
    /// `net_next_packet` yields, and calls `net_tick` about every 100 ms.
    /// While `has_pending_work()` the guest is waiting on a timer or the
    /// network and should be pumped again soon; `awaiting_input()` means the
    /// last command finished and the shell wants the next line.
    //
    // NOTE: keep this doc comment free of `*/` — wasm-bindgen copies it verbatim
    // into the generated `pkg/nixvm.js` as a `/** … */` JSDoc block, and a `*/`
    // inside (e.g. a JS block comment) would close that block early and produce
    // an "Unexpected token" syntax error in the module.
    #[wasm_bindgen]
    #[derive(Debug)]
    pub struct Terminal {
        vm: crate::vm::Vm,
        net: crate::tunnel::Tunnel,
        /// Packets taken from `net`, handed out one per `net_next_packet`.
        out: std::collections::VecDeque<Vec<u8>>,
        /// The last `pump` ran out of its time slice with the guest still
        /// computing (see `is_busy`).
        busy: bool,
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
            // Repack the tar into an in-memory squashfs (read-only lower) under a
            // writable tmpfs upper — the real copy-on-write overlay layout, so the
            // guest root stays compressed in RAM and decompresses on demand.
            let net = crate::tunnel::Tunnel::new();
            let vm = crate::vm::Vm::boot_squashfs_net(&tar, argv, MEM_BYTES, net.clone())
                .map_err(|e| JsError::new(&e))?;
            Ok(Self {
                vm,
                net,
                out: std::collections::VecDeque::new(),
                busy: false,
            })
        }

        /// [`Terminal::new`], on a [`Network`] the page already drives: the
        /// guest's sockets go over it, and either object can pump its packets.
        pub fn new_on(
            rootfs_targz: &[u8],
            argv: Vec<String>,
            net: &Network,
        ) -> Result<Terminal, JsError> {
            console_error_panic_hook::set_once();
            let argv = if argv.is_empty() {
                vec!["/bin/busybox".to_string(), "sh".to_string()]
            } else {
                argv
            };
            let tar = crate::fs::tar::gunzip(rootfs_targz, MAX_ROOTFS_BYTES)
                .map_err(|e| JsError::new(&e))?;
            let vm = crate::vm::Vm::boot_squashfs_net(&tar, argv, MEM_BYTES, net.net.clone())
                .map_err(|e| JsError::new(&e))?;
            Ok(Self {
                vm,
                net: net.net.clone(),
                out: std::collections::VecDeque::new(),
                busy: false,
            })
        }

        /// Boot a Docker/OCI image a finished [`ImagePull`] fetched, as
        /// `docker run` would: `args`, when given, replace its `Cmd`;
        /// `env` (`KEY=VALUE`) adds to its environment. Throws if the image
        /// names nothing to run or lacks the program or user it names.
        pub fn from_image(
            pull: ImagePull,
            args: Vec<String>,
            env: Vec<String>,
        ) -> Result<Terminal, JsError> {
            console_error_panic_hook::set_once();
            let image = pull.finish()?;
            Self::boot_image(&image, args, env)
        }

        /// Boot an image from an archive: `docker save` output (gzipped or
        /// not) or an OCI image layout tar, taking `arch`'s image
        /// (`amd64`/`arm64`) where it holds several. Otherwise as
        /// [`Terminal::from_image`].
        pub fn from_image_archive(
            archive: &[u8],
            arch: &str,
            args: Vec<String>,
            env: Vec<String>,
        ) -> Result<Terminal, JsError> {
            console_error_panic_hook::set_once();
            let arch = Arch::from_name(arch)
                .ok_or_else(|| JsError::new(&format!("unknown architecture {arch:?}")))?;
            let image = crate::oci::Image::from_archive(archive, arch)
                .map_err(|e| JsError::new(&e.to_string()))?;
            Self::boot_image(&image, args, env)
        }

        fn boot_image(
            image: &crate::oci::Image,
            args: Vec<String>,
            env: Vec<String>,
        ) -> Result<Terminal, JsError> {
            let opts = crate::oci::RunOptions {
                cmd: args,
                env,
                ..crate::oci::RunOptions::default()
            };
            let net = crate::tunnel::Tunnel::new();
            let vm = crate::vm::Vm::boot_image_net(image, &opts, MEM_BYTES, net.clone())
                .map_err(|e| JsError::new(&e))?;
            Ok(Self {
                vm,
                net,
                out: std::collections::VecDeque::new(),
                busy: false,
            })
        }

        /// Bring the network link up with the addresses from the tunnel's
        /// hello: `ipv4` (dotted quad) with its `ipv4_prefix` length, and/or
        /// `ipv6`, the leased range (`"2001:db8:1:2::/64"`, of which the guest
        /// takes `::1`). `mtu` bounds every packet the guest sends or asks for.
        #[allow(clippy::needless_pass_by_value)] // wasm-bindgen passes strings owned
        pub fn net_up(
            &mut self,
            ipv4: Option<String>,
            ipv4_prefix: u8,
            ipv6: Option<String>,
            mtu: u16,
        ) -> Result<(), JsError> {
            self.net
                .up(lease(ipv4.as_deref(), ipv4_prefix, ipv6.as_deref(), mtu)?);
            Ok(())
        }

        /// Boot an image an [`ImagePull`] fetched over `net`
        /// ([`ImagePull::pull_over`]), on that same link: the guest's sockets
        /// go over it too, and either object can pump its packets. Otherwise
        /// as [`Terminal::from_image`].
        pub fn from_image_on(
            pull: ImagePull,
            net: &Network,
            args: Vec<String>,
            env: Vec<String>,
        ) -> Result<Terminal, JsError> {
            console_error_panic_hook::set_once();
            let image = pull.finish()?;
            let opts = crate::oci::RunOptions {
                cmd: args,
                env,
                ..crate::oci::RunOptions::default()
            };
            let vm = crate::vm::Vm::boot_image_net(&image, &opts, MEM_BYTES, net.net.clone())
                .map_err(|e| JsError::new(&e))?;
            Ok(Self {
                vm,
                net: net.net.clone(),
                out: std::collections::VecDeque::new(),
                busy: false,
            })
        }

        /// The guest's public IPv6 address while the link is up (the one it
        /// took from the leased range), if IPv6 was leased. The IPv4 address
        /// is a shared, NATed inside address and not worth showing.
        #[must_use]
        pub fn net_ipv6(&self) -> Option<String> {
            self.net.lease()?.v6.map(|(addr, _)| addr.to_string())
        }

        /// Take the network link down (the transport closed).
        pub fn net_down(&mut self) {
            self.net.down();
        }

        /// A packet received from the transport.
        pub fn net_input(&mut self, packet: &[u8]) {
            self.net.inject(packet);
        }

        /// The next packet to send over the transport, if any.
        #[must_use]
        pub fn net_next_packet(&mut self) -> Option<Vec<u8>> {
            if self.out.is_empty() {
                self.out.extend(self.net.take_outbound());
            }
            self.out.pop_front()
        }

        /// Run the network timers (TCP retransmits, delayed ACKs); call about
        /// every 100 ms while the link is up.
        pub fn net_tick(&mut self) {
            self.net.tick();
        }

        /// Whether the shell finished the last command and wants input.
        #[must_use]
        pub fn awaiting_input(&self) -> bool {
            self.vm.awaiting_input()
        }

        /// Whether the parked guest is waiting on a timer or the network and
        /// should be pumped again soon.
        #[must_use]
        pub fn has_pending_work(&self) -> bool {
            self.vm.has_pending_work()
        }

        /// Feed keystrokes to the guest's stdin.
        pub fn write_stdin(&mut self, bytes: &[u8]) {
            self.vm.write_stdin(bytes);
        }

        /// Signal end-of-input (Ctrl-D).
        pub fn close_stdin(&mut self) {
            self.vm.close_stdin();
        }

        /// Ctrl-C: send SIGINT to the running command (not the shell).
        /// Returns whether a command was running; pump afterwards.
        pub fn interrupt(&mut self) -> bool {
            self.vm.interrupt()
        }

        /// Run until the guest parks for input or exits; returns the bytes to
        /// write to the terminal (stdout then stderr). Call again after
        /// `write_stdin`.
        #[must_use]
        pub fn pump(&mut self) -> Vec<u8> {
            // A short slice, then back to the page: the guest shares the tab's
            // only thread with the UI, its timers and the network tunnel, so a
            // compute-heavy command (apk verifying an index) must not freeze it.
            match self.vm.pump_for(PUMP_SLICE) {
                Ok(step) => {
                    self.busy = step.busy;
                    let mut out = step.stdout;
                    out.extend_from_slice(&step.stderr);
                    out
                }
                Err(_) => {
                    self.busy = false;
                    Vec::new()
                }
            }
        }

        /// Whether the last `pump` returned because its time slice ran out
        /// while the guest was still computing: pump again soon (after
        /// yielding to the browser), and don't treat it as waiting for input.
        #[must_use]
        pub fn is_busy(&self) -> bool {
            self.busy
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

    /// The tunnel lease `net_up` describes: `ipv4` (dotted quad) with its
    /// prefix length, and/or `ipv6`, the leased range (the guest takes `::1`).
    fn lease(
        ipv4: Option<&str>,
        ipv4_prefix: u8,
        ipv6: Option<&str>,
        mtu: u16,
    ) -> Result<crate::tunnel::Lease, JsError> {
        let v4 = match ipv4 {
            Some(a) => Some((
                a.parse::<std::net::Ipv4Addr>()
                    .map_err(|e| JsError::new(&format!("bad ipv4 {a:?}: {e}")))?,
                ipv4_prefix,
            )),
            None => None,
        };
        let v6 = match ipv6 {
            Some(r) => {
                let (addr, bits) = r.split_once('/').unwrap_or((r, "64"));
                let net = addr
                    .parse::<std::net::Ipv6Addr>()
                    .map_err(|e| JsError::new(&format!("bad ipv6 {r:?}: {e}")))?;
                let bits = bits.parse::<u8>().unwrap_or(64).min(128);
                Some((std::net::Ipv6Addr::from(u128::from(net) | 1), bits))
            }
            None => None,
        };
        Ok(crate::tunnel::Lease { v4, v6, mtu })
    }

    /// A network link carried over the page's packet transport (grouterd's
    /// WebSocket), before there's a guest to own it: [`ImagePull::pull_over`]
    /// pulls an image over it, and [`Terminal::from_image_on`] boots the image
    /// on it. The page pumps it as it would a [`Terminal`]'s: `up` with the
    /// hello's addresses, `input` for each packet received, send what
    /// `next_packet` yields, `tick` about every 100 ms.
    #[wasm_bindgen]
    #[derive(Debug)]
    pub struct Network {
        net: crate::tunnel::Tunnel,
        out: std::collections::VecDeque<Vec<u8>>,
    }

    #[wasm_bindgen]
    impl Network {
        #[wasm_bindgen(constructor)]
        #[must_use]
        pub fn new() -> Network {
            console_error_panic_hook::set_once();
            Self {
                net: crate::tunnel::Tunnel::new(),
                out: std::collections::VecDeque::new(),
            }
        }

        /// Bring the link up with the addresses from the transport's hello
        /// (as [`Terminal::net_up`]).
        #[allow(clippy::needless_pass_by_value)] // wasm-bindgen passes strings owned
        pub fn up(
            &mut self,
            ipv4: Option<String>,
            ipv4_prefix: u8,
            ipv6: Option<String>,
            mtu: u16,
        ) -> Result<(), JsError> {
            self.net
                .up(lease(ipv4.as_deref(), ipv4_prefix, ipv6.as_deref(), mtu)?);
            Ok(())
        }

        /// The public IPv6 address the link took from the leased range, while
        /// it's up and IPv6 was leased.
        #[must_use]
        pub fn ipv6(&self) -> Option<String> {
            self.net.lease()?.v6.map(|(addr, _)| addr.to_string())
        }

        /// Take the link down (the transport closed).
        pub fn down(&mut self) {
            self.net.down();
        }

        /// A packet received from the transport.
        pub fn input(&mut self, packet: &[u8]) {
            self.net.inject(packet);
        }

        /// The next packet to send over the transport, if any.
        #[must_use]
        pub fn next_packet(&mut self) -> Option<Vec<u8>> {
            if self.out.is_empty() {
                self.out.extend(self.net.take_outbound());
            }
            self.out.pop_front()
        }

        /// Run the network timers; call about every 100 ms while it's up.
        pub fn tick(&mut self) {
            self.net.tick();
        }
    }

    impl Default for Network {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Pulling a Docker/OCI image from a registry, one of two ways.
    ///
    /// Over a [`Network`] (the page's packet tunnel): call `pull_over(net)`
    /// between pumping the network until it returns `true`. nixvm does the
    /// DNS, TLS and HTTP itself, so CORS doesn't apply and no proxy is
    /// needed; then [`Terminal::from_image_on`] boots it on the same link.
    ///
    /// ```js
    /// const net = new Network();   // up() with the tunnel's hello, then pump
    /// const pull = new ImagePull("alpine:3.20", "amd64");
    /// while (!pull.pull_over(net)) await pumpPackets(net);
    /// const term = Terminal.from_image_on(pull, net, ["sh"], []);
    /// ```
    ///
    /// Or with the page doing the fetching: `next_request()` says what to
    /// `fetch()` (a JSON `{url, headers, digest}`, or `undefined` once done),
    /// and `feed()` takes the response. Registries send no CORS headers, so a
    /// page sends these requests through a proxy it trusts, and that proxy
    /// must pass the `WWW-Authenticate` header back. A request whose `digest`
    /// is set can be answered from a cache keyed by it; every blob is checked
    /// against its digest either way.
    ///
    /// ```js
    /// const pull = new ImagePull("alpine:3.20", "amd64");
    /// let req;
    /// while ((req = pull.next_request()) !== undefined) {
    ///   const { url, headers } = JSON.parse(req);
    ///   const r = await fetch(PROXY + encodeURIComponent(url), { headers });
    ///   pull.feed(r.status, r.headers.get("www-authenticate"),
    ///             r.headers.get("content-type"),
    ///             new Uint8Array(await r.arrayBuffer()));
    /// }
    /// const term = Terminal.from_image(pull, ["sh"], []);
    /// ```
    #[wasm_bindgen]
    #[derive(Debug)]
    pub struct ImagePull {
        mode: PullMode,
    }

    /// Who's fetching: the page, or nixvm over a [`Network`]. `Done` only
    /// while [`ImagePull::pull_over`] switches between them.
    #[derive(Debug)]
    #[allow(clippy::large_enum_variant)] // one per pull
    enum PullMode {
        Page(crate::oci::Pull),
        Net(crate::oci::net::NetPull),
        Done,
    }

    impl ImagePull {
        fn pull(&self) -> Option<&crate::oci::Pull> {
            match &self.mode {
                PullMode::Page(p) => Some(p),
                PullMode::Net(n) => Some(n.pull()),
                PullMode::Done => None,
            }
        }

        fn page(&mut self) -> Result<&mut crate::oci::Pull, JsError> {
            match &mut self.mode {
                PullMode::Page(p) => Ok(p),
                _ => Err(JsError::new("this pull is going over a Network")),
            }
        }

        fn finish(self) -> Result<crate::oci::Image, JsError> {
            match self.mode {
                PullMode::Page(p) => p.finish(),
                PullMode::Net(n) => n.finish(),
                PullMode::Done => Err(crate::oci::OciError::Fetch("the pull failed".into())),
            }
            .map_err(|e| JsError::new(&e.to_string()))
        }
    }

    #[wasm_bindgen]
    impl ImagePull {
        /// Start pulling `name` (`alpine:3.20`, `ghcr.io/o/i:tag`) for a
        /// guest of `arch` (`amd64` or `arm64`).
        #[wasm_bindgen(constructor)]
        pub fn new(name: &str, arch: &str) -> Result<ImagePull, JsError> {
            console_error_panic_hook::set_once();
            let arch = Arch::from_name(arch)
                .ok_or_else(|| JsError::new(&format!("unknown architecture {arch:?}")))?;
            let inner =
                crate::oci::Pull::new(name, arch).map_err(|e| JsError::new(&e.to_string()))?;
            Ok(Self {
                mode: PullMode::Page(inner),
            })
        }

        /// Log in to the registry (a token can go as the password). Before
        /// the pull starts.
        pub fn set_credentials(&mut self, user: &str, password: &str) -> Result<(), JsError> {
            self.page()?.set_credentials(user, password);
            Ok(())
        }

        /// Talk to `base` (a mirror) instead of the registry the name says.
        /// Before the pull starts.
        pub fn set_registry_base(&mut self, base: &str) -> Result<(), JsError> {
            self.page()?.set_registry_base(base);
            Ok(())
        }

        /// Pull over `net`, as far as it can go right now without waiting;
        /// `true` once everything is fetched. Call it again after pumping the
        /// network. Throws on a network, TLS or registry error.
        pub fn pull_over(&mut self, net: &Network) -> Result<bool, JsError> {
            if let PullMode::Page(_) = self.mode {
                let PullMode::Page(p) = std::mem::replace(&mut self.mode, PullMode::Done) else {
                    unreachable!()
                };
                self.mode = PullMode::Net(crate::oci::net::NetPull::new(p, net.net.egress()));
            }
            let PullMode::Net(n) = &mut self.mode else {
                return Err(JsError::new("the pull failed"));
            };
            n.poll().map_err(|e| JsError::new(&e.to_string()))
        }

        /// The next request, as JSON `{"url", "headers": {name: value},
        /// "digest": string|null}`; `undefined` once everything is fetched.
        #[must_use]
        pub fn next_request(&self) -> Option<String> {
            use crate::oci::json::quote;
            let PullMode::Page(p) = &self.mode else {
                return None;
            };
            let r = p.next_request()?;
            let headers: Vec<String> = r
                .headers
                .iter()
                .map(|(k, v)| format!("{}:{}", quote(k), quote(v)))
                .collect();
            Some(format!(
                r#"{{"url":{},"headers":{{{}}},"digest":{}}}"#,
                quote(&r.url),
                headers.join(","),
                r.digest.as_deref().map_or("null".to_string(), quote)
            ))
        }

        /// Hand over the response to the last request (after redirects):
        /// its status, `WWW-Authenticate` and `Content-Type` headers, and
        /// body. Throws on a registry error or bad content.
        pub fn feed(
            &mut self,
            status: u16,
            www_authenticate: Option<String>,
            content_type: Option<String>,
            body: Vec<u8>,
        ) -> Result<(), JsError> {
            let mut headers = Vec::new();
            if let Some(v) = www_authenticate {
                headers.push(("www-authenticate".to_string(), v));
            }
            if let Some(v) = content_type {
                headers.push(("content-type".to_string(), v));
            }
            self.page()?
                .feed(crate::oci::Response {
                    status,
                    headers,
                    body,
                })
                .map_err(|e| JsError::new(&e.to_string()))
        }

        /// `[layers fetched, layers in all, bytes fetched, bytes in all]`;
        /// the totals are known once the manifest is in.
        #[must_use]
        #[allow(clippy::cast_precision_loss)] // counts and sizes far below 2^53
        pub fn progress(&self) -> Vec<f64> {
            let (l, lt, b, bt) = self.pull().map_or((0, 0, 0, 0), crate::oci::Pull::progress);
            vec![l as f64, lt as f64, b as f64, bt as f64]
        }

        /// Whether everything has been fetched.
        #[must_use]
        pub fn is_done(&self) -> bool {
            self.pull().is_some_and(crate::oci::Pull::is_done)
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
        mounts.mount("/proc", Box::new(ProcFs::new(1)));
        mounts.mount("/sys", Box::new(SysFs::new(1)));
        mounts.mount(
            "/sys/fs/cgroup",
            Box::new(crate::fs::CgroupFs::new(
                crate::fs::CgroupTree::shared(),
                crate::fs::cgroupfs::ROOT,
            )),
        );

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
