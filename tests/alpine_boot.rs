//! End-to-end: boot a real Alpine root image interactively and run commands
//! through busybox `sh`, exactly as the browser terminal does.
//!
//! Skipped unless `NIXVM_ALPINE_TAR` points at an *uncompressed* Alpine
//! minirootfs `.tar` (the browser decompresses the `.tar.gz` itself), so CI —
//! which has no image — is unaffected. To run it:
//!
//! ```text
//! curl -O https://dl-cdn.alpinelinux.org/alpine/v3.20/releases/aarch64/alpine-minirootfs-3.20.10-aarch64.tar.gz
//! gunzip alpine-minirootfs-3.20.10-aarch64.tar.gz
//! NIXVM_ALPINE_TAR=$PWD/alpine-minirootfs-3.20.10-aarch64.tar cargo test --test alpine_boot -- --nocapture
//! ```

use nixvm::vm::Vm;

fn drain(vm: &mut Vm) -> String {
    // Pump to a quiescent point (blocked for input, or exited), collecting all
    // output. Bounded so a runaway can't hang the test.
    let mut out = Vec::new();
    for _ in 0..64 {
        let step = vm.pump().expect("pump");
        out.extend_from_slice(&step.stdout);
        out.extend_from_slice(&step.stderr);
        if step.exit_code.is_some() {
            break;
        }
        // Blocked with no new output means it's waiting on us.
        if step.stdout.is_empty() && step.stderr.is_empty() {
            break;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[test]
fn boots_alpine_and_runs_shell_commands() {
    let Ok(tar_path) = std::env::var("NIXVM_ALPINE_TAR") else {
        eprintln!("NIXVM_ALPINE_TAR not set; skipping live Alpine boot test");
        return;
    };
    let tar = std::fs::read(&tar_path).expect("read Alpine tar");

    let mut vm = Vm::boot(
        &tar,
        vec!["/bin/busybox".to_string(), "sh".to_string()],
        256 * 1024 * 1024,
    )
    .expect("boot Alpine busybox sh");

    // Let the shell start up and reach its first read of stdin.
    let boot = drain(&mut vm);
    eprintln!("--- boot output ---\n{boot}");

    // Type a command; the echoed output must come back.
    vm.write_stdin(b"echo hello-from-alpine\n");
    let out = drain(&mut vm);
    eprintln!("--- after echo ---\n{out}");
    assert!(
        out.contains("hello-from-alpine"),
        "shell should echo the command output, got: {out:?}"
    );

    // A second command, then exit. The rootfs may be either guest arch (the
    // env var picks it), so accept the machine name of both.
    vm.write_stdin(b"uname -m\n");
    let out2 = drain(&mut vm);
    eprintln!("--- after uname ---\n{out2}");
    assert!(
        out2.contains("aarch64") || out2.contains("x86_64"),
        "uname -m should print the guest machine, got: {out2:?}"
    );

    vm.write_stdin(b"exit\n");
    let _ = drain(&mut vm);
    assert!(vm.exit_code().is_some(), "shell should exit on `exit`");
}

/// Boot Alpine from a `.tar` repacked into an **in-memory squashfs** (read-only
/// lower) under a tmpfs upper — the real copy-on-write overlay layout, and the
/// path the browser demo takes. Gated on the `fstool` feature and
/// `NIXVM_ALPINE_TAR`.
#[cfg(feature = "fstool")]
#[test]
fn boots_alpine_from_in_memory_squashfs_overlay() {
    let Ok(tar_path) = std::env::var("NIXVM_ALPINE_TAR") else {
        eprintln!("NIXVM_ALPINE_TAR not set; skipping squashfs-overlay boot test");
        return;
    };
    let tar = std::fs::read(&tar_path).expect("read Alpine tar");

    let mut vm = Vm::boot_squashfs(
        &tar,
        vec!["/bin/busybox".to_string(), "sh".to_string()],
        256 * 1024 * 1024,
    )
    .expect("boot from in-memory squashfs overlay");
    let _ = drain(&mut vm);
    vm.write_stdin(b"cat /etc/alpine-release; echo squashfs-overlay-ok\n");
    let out = drain(&mut vm);
    eprintln!("--- squashfs overlay ---\n{out}");
    assert!(
        out.contains("squashfs-overlay-ok"),
        "shell runs from the squashfs-overlay root, got: {out:?}"
    );
    // The writable upper works: create a file, read it back.
    vm.write_stdin(b"echo hi > /tmp/x; cat /tmp/x\n");
    let out2 = drain(&mut vm);
    assert!(
        out2.contains("hi"),
        "tmpfs upper is writable, got: {out2:?}"
    );
    // A process killed by a signal must close its fds: the pipe's reader then
    // sees EOF and the pipeline finishes (it used to hang forever).
    vm.write_stdin(b"sh -c 'kill -9 $$' | wc -c; echo pipeline-done\n");
    let out3 = drain(&mut vm);
    assert!(
        out3.contains("pipeline-done"),
        "signal-killed writer's pipe reaches EOF, got: {out3:?}"
    );
    // `execve` of a `#!` script runs its interpreter (apk runs every package's
    // install scripts this way; they all failed with 127 before). `chroot`
    // execs directly, without the shell's own ENOEXEC fallback.
    vm.write_stdin(b"printf '#!/bin/sh -e\\necho shebang-$1\\n' > /tmp/s.sh; chmod +x /tmp/s.sh; chroot / /tmp/s.sh ok; echo exit-$?\n");
    let out4 = drain(&mut vm);
    assert!(
        out4.contains("shebang-ok") && out4.contains("exit-0"),
        "a #! script execs through its interpreter, got: {out4:?}"
    );
}

/// Live host-egress smoke test: boot Alpine with `NIXVM_NET=host` set and run
/// `apk update` against the real mirror over plain HTTP. Gated on **both**
/// `NIXVM_ALPINE_TAR` *and* `NIXVM_NET=host` (so CI, with neither, skips it)
/// and needs real outbound internet. Proves the full egress path: DNS over a
/// host UDP socket, TCP connect passthrough, and poll/read/write bridging.
#[cfg(feature = "fstool")]
#[test]
fn apk_update_over_host_egress() {
    let Ok(tar_path) = std::env::var("NIXVM_ALPINE_TAR") else {
        eprintln!("NIXVM_ALPINE_TAR not set; skipping egress test");
        return;
    };
    if std::env::var("NIXVM_NET").ok().as_deref() != Some("host") {
        eprintln!("NIXVM_NET != host; skipping egress test");
        return;
    }
    let tar = std::fs::read(&tar_path).expect("read Alpine tar");
    let mut vm = Vm::boot_squashfs(
        &tar,
        vec!["/bin/busybox".to_string(), "sh".to_string()],
        256 * 1024 * 1024,
    )
    .expect("boot");
    // Spin-pump: a guest blocked on async host I/O needs the driver to keep
    // pumping until the network completes.
    let drain = |vm: &mut Vm| -> String {
        let mut out = Vec::new();
        let mut idle = 0;
        for _ in 0..2_000_000 {
            let step = vm.pump().expect("pump");
            let got = !step.stdout.is_empty() || !step.stderr.is_empty();
            out.extend_from_slice(&step.stdout);
            out.extend_from_slice(&step.stderr);
            if step.exit_code.is_some() {
                break;
            }
            if got {
                idle = 0;
                continue;
            }
            idle += 1;
            if idle > 3000 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        String::from_utf8_lossy(&out).into_owned()
    };
    let _ = drain(&mut vm);
    // apk's aarch64 build starts up through NEON instructions the interpreter
    // doesn't decode yet (LD2/3/4 de-interleave, LDR-SIMD register offset), so
    // it can't run there regardless of networking. The egress path itself is
    // arch-agnostic (it lives in the kernel); assert the full apk flow only on
    // x86-64, where the interpreter is complete enough.
    vm.write_stdin(b"uname -m\n");
    let machine = drain(&mut vm);
    if !machine.contains("x86_64") {
        eprintln!(
            "guest is not x86_64 ({}); apk needs more NEON interpreter coverage, \
             skipping the apk assertion (egress itself is arch-agnostic)",
            machine.trim()
        );
        return;
    }

    // The stock repositories are https; the minirootfs has no CA certs, so
    // rewrite to http for this smoke test (egress itself is scheme-agnostic).
    vm.write_stdin(b"sed -i 's|https|http|' /etc/apk/repositories; apk update; echo DONE=$?\n");
    let out = drain(&mut vm);
    eprintln!("--- apk update ---\n{out}");
    assert!(
        out.contains("packages available") && out.contains("DONE=0"),
        "apk update should succeed over host egress, got: {out:?}"
    );
}

/// Same, but from the *compressed* `.tar.gz`, decompressed in-process via
/// `compcol` (the path the browser demo takes). Gated on the `targz` feature
/// and `NIXVM_ALPINE_TARGZ` pointing at the `.tar.gz`.
#[cfg(feature = "targz")]
#[test]
fn boots_alpine_from_targz_via_compcol() {
    let Ok(gz_path) = std::env::var("NIXVM_ALPINE_TARGZ") else {
        eprintln!("NIXVM_ALPINE_TARGZ not set; skipping compcol .tar.gz boot test");
        return;
    };
    let gz = std::fs::read(&gz_path).expect("read Alpine .tar.gz");
    let tar = nixvm::fs::tar::gunzip(&gz, 512 * 1024 * 1024).expect("compcol gunzip");

    let mut vm = Vm::boot(
        &tar,
        vec!["/bin/busybox".to_string(), "sh".to_string()],
        256 * 1024 * 1024,
    )
    .expect("boot from compcol-decompressed rootfs");
    let _ = drain(&mut vm);
    vm.write_stdin(b"echo compcol-decompressed-ok\n");
    let out = drain(&mut vm);
    assert!(
        out.contains("compcol-decompressed-ok"),
        "shell runs from the compcol-decompressed rootfs, got: {out:?}"
    );
}

/// Relative symlinks in the squashfs lower resolve (`/lib/libz.so.1 ->
/// libz.so.1.3.2`), so the dynamic linker can load apk's libz. A regression
/// here showed up in the browser as "Error loading shared library libz.so.1:
/// Symbolic link loop". Gated like the other live Alpine tests.
#[cfg(feature = "fstool")]
#[test]
fn squashfs_relative_symlinks_resolve_for_apk() {
    let Ok(tar_path) = std::env::var("NIXVM_ALPINE_TAR") else {
        eprintln!("NIXVM_ALPINE_TAR not set; skipping squashfs symlink test");
        return;
    };
    let tar = std::fs::read(&tar_path).expect("read Alpine tar");
    let mut vm = Vm::boot_squashfs(
        &tar,
        vec!["/bin/busybox".to_string(), "sh".to_string()],
        256 * 1024 * 1024,
    )
    .expect("boot from in-memory squashfs overlay");
    let _ = drain(&mut vm);
    vm.write_stdin(b"readlink /lib/libz.so.1; head -c 4 /lib/libz.so.1 | od -c | head -1; apk --version; echo done-$?\n");
    let out = drain(&mut vm);
    eprintln!("--- squashfs symlinks ---\n{out}");
    assert!(!out.contains("Symbolic link loop"), "symlink loop: {out:?}");
    assert!(out.contains("apk-tools"), "apk runs (loads libz): {out:?}");
}

/// busybox `ping` over the tunnel: the test plays the internet and answers
/// every ICMP echo request the guest sends, then checks ping saw the replies.
/// Also pings loopback, which the kernel answers itself. Gated on
/// `NIXVM_ALPINE_TAR`.
#[cfg(all(feature = "fstool", feature = "tunnel"))]
#[test]
fn ping_over_tunnel_and_loopback() {
    use nixvm::tunnel::{Lease, Tunnel};
    use std::net::Ipv4Addr;

    fn csum(data: &[u8]) -> u16 {
        let mut sum: u32 = data
            .chunks(2)
            .map(|c| u32::from(u16::from_be_bytes([c[0], *c.get(1).unwrap_or(&0)])))
            .sum();
        while sum >> 16 != 0 {
            sum = (sum & 0xffff) + (sum >> 16);
        }
        !(sum as u16)
    }

    /// Answer an IPv4 ICMP echo request with its echo reply.
    fn echo_reply(p: &[u8]) -> Option<Vec<u8>> {
        let hl = usize::from(p[0] & 0xf) * 4;
        if p[0] >> 4 != 4 || p[9] != 1 || p.get(hl) != Some(&8) {
            return None;
        }
        let mut r = p.to_vec();
        r[12..16].copy_from_slice(&p[16..20]);
        r[16..20].copy_from_slice(&p[12..16]);
        r[8] = 57; // TTL as seen after a few hops
        r[10..12].fill(0);
        let s = csum(&r[..hl]);
        r[10..12].copy_from_slice(&s.to_be_bytes());
        r[hl] = 0; // echo reply
        r[hl + 2..hl + 4].fill(0);
        let s = csum(&r[hl..]);
        r[hl + 2..hl + 4].copy_from_slice(&s.to_be_bytes());
        Some(r)
    }

    let Ok(tar_path) = std::env::var("NIXVM_ALPINE_TAR") else {
        eprintln!("NIXVM_ALPINE_TAR not set; skipping ping test");
        return;
    };
    let tar = std::fs::read(&tar_path).expect("read Alpine tar");
    let net = Tunnel::new();
    net.up(Lease {
        v4: Some((Ipv4Addr::new(100, 64, 0, 2), 10)),
        v6: None,
        mtu: 1400,
    });
    let mut vm = Vm::boot_squashfs_net(
        &tar,
        vec!["/bin/busybox".to_string(), "sh".to_string()],
        256 * 1024 * 1024,
        net.clone(),
    )
    .expect("boot with a tunnel");
    let _ = drain(&mut vm);

    let mut run = |cmd: &[u8]| -> String {
        vm.write_stdin(cmd);
        let mut out = Vec::new();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            let step = vm.pump().expect("pump");
            out.extend_from_slice(&step.stdout);
            out.extend_from_slice(&step.stderr);
            for p in net.take_outbound() {
                if let Some(r) = echo_reply(&p) {
                    net.inject(&r);
                }
            }
            if vm.awaiting_input() && !net.has_outbound() {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "timed out: {}",
                String::from_utf8_lossy(&out)
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        String::from_utf8_lossy(&out).into_owned()
    };

    let out = run(b"ping -c 2 8.8.8.8\n");
    eprintln!("--- ping 8.8.8.8 ---\n{out}");
    assert!(
        out.contains("64 bytes from 8.8.8.8"),
        "echo replies seen: {out:?}"
    );
    assert!(out.contains("2 packets received"), "both answered: {out:?}");

    let out = run(b"ping -c 1 127.0.0.1\n");
    eprintln!("--- ping 127.0.0.1 ---\n{out}");
    assert!(
        out.contains("1 packets received"),
        "loopback answered: {out:?}"
    );

    // Ctrl-C stops an endless ping: it prints its statistics (its SIGINT
    // handler runs) and the shell — not interrupted itself — reads again.
    vm.write_stdin(b"ping 8.8.8.8; echo after-ping\n");
    let mut out = Vec::new();
    let t = std::time::Instant::now();
    let mut interrupted = false;
    loop {
        let step = vm.pump().expect("pump");
        out.extend_from_slice(&step.stdout);
        out.extend_from_slice(&step.stderr);
        for p in net.take_outbound() {
            if let Some(r) = echo_reply(&p) {
                net.inject(&r);
            }
        }
        if !interrupted && t.elapsed() > std::time::Duration::from_millis(1500) {
            assert!(vm.interrupt(), "a command was running");
            interrupted = true;
        }
        if interrupted && vm.awaiting_input() {
            break;
        }
        assert!(
            t.elapsed().as_secs() < 20,
            "ping not interrupted: {}",
            String::from_utf8_lossy(&out)
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let out = String::from_utf8_lossy(&out);
    eprintln!("--- ping + ^C ---\n{out}");
    assert!(
        out.contains("packets transmitted"),
        "ping printed its stats: {out:?}"
    );
    assert!(out.contains("after-ping"), "the shell carried on: {out:?}");
    assert!(vm.exit_code().is_none(), "the shell survived ^C");
}

/// The real network path of the browser demo, natively: packets go through
/// grouterd's WebSocket tunnel via `scripts/tunnel-bridge.mjs`. Runs DNS +
/// HTTP (`apk update`) and ping against the live internet. Gated on
/// `NIXVM_TUNNEL_BRIDGE` (the bridge's `host:port`) and `NIXVM_ALPINE_TAR`;
/// each run consumes one (rationed) tunnel token.
#[cfg(all(feature = "fstool", feature = "tunnel"))]
#[test]
fn tunnel_live_apk_update() {
    use nixvm::tunnel::{Lease, Tunnel};
    use std::io::{Read, Write};
    use std::time::{Duration, Instant};

    let (Ok(bridge), Ok(tar_path)) = (
        std::env::var("NIXVM_TUNNEL_BRIDGE"),
        std::env::var("NIXVM_ALPINE_TAR"),
    ) else {
        eprintln!("NIXVM_TUNNEL_BRIDGE/NIXVM_ALPINE_TAR not set; skipping live tunnel test");
        return;
    };
    let mut link = std::net::TcpStream::connect(&bridge).expect("connect to tunnel-bridge");

    // Read one [kind][len][bytes] frame (blocking).
    fn read_frame(s: &mut std::net::TcpStream) -> Option<(u8, Vec<u8>)> {
        let mut hdr = [0u8; 5];
        s.read_exact(&mut hdr).ok()?;
        let mut b = vec![0u8; u32::from_be_bytes(hdr[1..5].try_into().unwrap()) as usize];
        s.read_exact(&mut b).ok()?;
        Some((hdr[0], b))
    }
    // A JSON string field of the hello (it is tiny and flat).
    fn field<'a>(json: &'a str, key: &str) -> Option<&'a str> {
        let at = json.find(&format!("\"{key}\":\""))? + key.len() + 4;
        Some(&json[at..at + json[at..].find('"')?])
    }

    let (kind, hello) = read_frame(&mut link).expect("hello");
    assert_eq!(kind, 0);
    let hello = String::from_utf8(hello).unwrap();
    eprintln!("hello: {hello}");
    let v4 = field(&hello, "ipv4").map(|a| (a.parse().unwrap(), 10));
    let v6 = field(&hello, "ipv6").map(|r| {
        let net: std::net::Ipv6Addr = r.split('/').next().unwrap().parse().unwrap();
        (std::net::Ipv6Addr::from(u128::from(net) | 1), 64)
    });
    let net = Tunnel::new();
    net.up(Lease { v4, v6, mtu: 1400 });

    let tar = std::fs::read(&tar_path).expect("read Alpine tar");
    let mut vm = Vm::boot_squashfs_net(
        &tar,
        vec!["/bin/busybox".to_string(), "sh".to_string()],
        256 * 1024 * 1024,
        net.clone(),
    )
    .expect("boot with a tunnel");
    let _ = drain(&mut vm);

    // Inbound frames arrive on a reader thread (the test plays the page's
    // event loop: pump, flush packets, feed arrivals, tick every 100 ms).
    let (tx, rx) = std::sync::mpsc::channel();
    let mut reader = link.try_clone().unwrap();
    std::thread::spawn(move || {
        while let Some((1, p)) = read_frame(&mut reader) {
            if tx.send(p).is_err() {
                break;
            }
        }
    });

    let mut run = |cmd: &[u8], secs: u64| -> String {
        vm.write_stdin(cmd);
        let mut out = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(secs);
        let mut last_tick = Instant::now();
        loop {
            let step = vm.pump().expect("pump");
            out.extend_from_slice(&step.stdout);
            out.extend_from_slice(&step.stderr);
            for p in net.take_outbound() {
                let mut f = vec![1u8];
                f.extend_from_slice(&(p.len() as u32).to_be_bytes());
                f.extend_from_slice(&p);
                link.write_all(&f).unwrap();
            }
            while let Ok(p) = rx.try_recv() {
                net.inject(&p);
            }
            if last_tick.elapsed() >= Duration::from_millis(100) {
                net.tick();
                last_tick = Instant::now();
            }
            if vm.awaiting_input() && !net.has_outbound() {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "timed out: {}",
                String::from_utf8_lossy(&out)
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        String::from_utf8_lossy(&out).into_owned()
    };

    let out = run(b"ping -c 2 1.1.1.1\n", 30);
    eprintln!("--- ping ---\n{out}");
    assert!(
        out.contains("packets received") && !out.contains(" 0 packets received"),
        "{out:?}"
    );

    let out = run(b"apk update; echo apk-exit-$?\n", 180);
    eprintln!("--- apk update ---\n{out}");
    assert!(out.contains("apk-exit-0"), "apk update succeeds: {out:?}");

    let out = run(
        b"wget -q -O - http://example.com | head -c 200; echo; echo wget-exit-$?\n",
        60,
    );
    eprintln!("--- wget ---\n{out}");
    assert!(out.contains("wget-exit-0"), "{out:?}");
}

/// `Vm::pump_for` hands control back on time while the guest computes (the
/// browser tab must not freeze during `apk`'s CPU-heavy work), and the
/// computation still finishes across calls. Gated on `NIXVM_ALPINE_TAR`.
#[cfg(feature = "fstool")]
#[test]
fn pump_for_yields_during_guest_computation() {
    use std::time::{Duration, Instant};
    let Ok(tar_path) = std::env::var("NIXVM_ALPINE_TAR") else {
        eprintln!("NIXVM_ALPINE_TAR not set; skipping pump_for test");
        return;
    };
    let tar = std::fs::read(&tar_path).expect("read Alpine tar");
    let mut vm = Vm::boot_squashfs(
        &tar,
        vec!["/bin/busybox".to_string(), "sh".to_string()],
        256 * 1024 * 1024,
    )
    .expect("boot");
    let _ = drain(&mut vm);
    // A pure-CPU loop: no syscalls for long stretches.
    vm.write_stdin(b"i=0; while [ $i -lt 3000 ]; do i=$((i+1)); done; echo loop-done\n");
    let budget = Duration::from_millis(30);
    let (mut out, mut busy_calls, mut longest) = (Vec::new(), 0, Duration::ZERO);
    let start = Instant::now();
    loop {
        let t = Instant::now();
        let step = vm.pump_for(budget).expect("pump_for");
        longest = longest.max(t.elapsed());
        out.extend_from_slice(&step.stdout);
        busy_calls += usize::from(step.busy);
        if !step.busy && vm.awaiting_input() {
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(120),
            "loop never finished"
        );
    }
    let out = String::from_utf8_lossy(&out);
    eprintln!(
        "pump_for: {busy_calls} busy returns, longest call {longest:?}, total {:?}",
        start.elapsed()
    );
    assert!(
        out.contains("loop-done"),
        "the computation completes: {out:?}"
    );
    assert!(busy_calls > 0, "a long computation yields at least once");
    assert!(
        longest < Duration::from_millis(500),
        "each call returns near its budget, longest {longest:?}"
    );
}
