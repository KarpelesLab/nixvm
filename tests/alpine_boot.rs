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
