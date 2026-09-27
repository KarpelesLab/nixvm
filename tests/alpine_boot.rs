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
}
