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

    // A second command, then exit.
    vm.write_stdin(b"uname -m\n");
    let out2 = drain(&mut vm);
    eprintln!("--- after uname ---\n{out2}");
    assert!(out2.contains("aarch64"), "uname -m should print aarch64");

    vm.write_stdin(b"exit\n");
    let _ = drain(&mut vm);
    assert!(vm.exit_code().is_some(), "shell should exit on `exit`");
}
