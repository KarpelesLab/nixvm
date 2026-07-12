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
