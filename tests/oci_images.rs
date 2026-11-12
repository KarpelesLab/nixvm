//! End-to-end: pull real images from Docker Hub and run them as `docker run`
//! would — the image's command, environment, working directory and user,
//! overridden the way `docker run`'s arguments do.
//!
//! These reach the network, so they're skipped unless `NIXVM_OCI_TESTS=1`
//! (CI's kvm job sets it). To run them:
//!
//! ```text
//! NIXVM_OCI_TESTS=1 cargo test --features oci,fetch --test oci_images -- --nocapture
//! ```
#![cfg(all(feature = "oci", feature = "fetch"))]

use nixvm::Arch;
use nixvm::oci::{self, RunOptions};
use nixvm::vm::Vm;

fn enabled() -> bool {
    let on = std::env::var_os("NIXVM_OCI_TESTS").is_some_and(|v| v == "1");
    if !on {
        eprintln!("NIXVM_OCI_TESTS not set; skipping the image pull tests");
    }
    on
}

fn arch() -> Arch {
    Arch::host_native().unwrap_or(Arch::X86_64)
}

/// Run to the end, returning (exit code, stdout + stderr).
fn run(vm: &mut Vm) -> (i32, String) {
    let mut out = Vec::new();
    for _ in 0..10_000 {
        let step = vm.pump().expect("pump");
        out.extend_from_slice(&step.stdout);
        out.extend_from_slice(&step.stderr);
        if let Some(code) = step.exit_code {
            return (code, String::from_utf8_lossy(&out).into_owned());
        }
        vm.close_stdin();
    }
    panic!("the guest didn't finish: {}", String::from_utf8_lossy(&out));
}

#[test]
fn an_image_runs_with_its_config_and_docker_run_overrides() {
    if !enabled() {
        return;
    }
    let img = oci::pull("alpine:3.20", arch()).expect("pull alpine");
    assert_eq!(img.config.arch(), Some(arch()));
    assert!(
        img.digest
            .as_deref()
            .is_some_and(|d| d.starts_with("sha256:"))
    );

    // The image's own Cmd (`/bin/sh`), given a script on stdin.
    let opts = RunOptions {
        cmd: vec![
            "sh".into(),
            "-c".into(),
            "cat /etc/alpine-release; id -u; pwd; echo $HOME $GREETING; echo oops >&2; exit 7"
                .into(),
        ],
        env: vec!["GREETING=hello".into()],
        ..RunOptions::default()
    };
    let mut vm = Vm::boot_image(&img, &opts, 512 << 20).expect("boot");
    let (code, out) = run(&mut vm);
    assert_eq!(code, 7, "{out}");
    assert!(out.starts_with("3.20."), "{out}");
    assert!(out.contains("\n0\n/\n/root hello\noops"), "{out}");

    // --user and --workdir, and --entrypoint.
    let opts = RunOptions {
        entrypoint: Some(vec!["id".into()]),
        user: Some("nobody".into()),
        working_dir: Some("/tmp".into()),
        ..RunOptions::default()
    };
    let mut vm = Vm::boot_image(&img, &opts, 512 << 20).expect("boot");
    let (code, out) = run(&mut vm);
    assert_eq!(code, 0, "{out}");
    assert!(
        out.starts_with("uid=65534(nobody) gid=65534(nobody)"),
        "{out}"
    );

    // A program the image lacks is an error, not a crash.
    let opts = RunOptions {
        entrypoint: Some(vec!["no-such-program".into()]),
        ..RunOptions::default()
    };
    let e = Vm::boot_image(&img, &opts, 512 << 20).unwrap_err();
    assert!(e.contains("not found on the image's PATH"), "{e}");
}

#[test]
fn a_script_entrypoint_and_layers_with_whiteouts_run() {
    if !enabled() {
        return;
    }
    // python:*-slim is several Debian layers, whiteouts and hard links
    // included, with an entrypoint-free `python3` Cmd; its own scripts run
    // through `#!`.
    let img = oci::pull("python:3.12-slim", arch()).expect("pull python");
    assert!(img.layers.len() > 1);
    let opts = RunOptions {
        cmd: vec![
            "sh".into(),
            "-c".into(),
            "python3 -c 'import sqlite3,ssl,json; print(\"py\", sqlite3.sqlite_version)'; pip3 --version | cut -c1-3".into(),
        ],
        ..RunOptions::default()
    };
    let mut vm = Vm::boot_image(&img, &opts, 1 << 30).expect("boot");
    let (code, out) = run(&mut vm);
    assert_eq!(code, 0, "{out}");
    assert!(out.contains("py 3.") && out.contains("pip"), "{out}");
}

/// The pull nixvm does over a guest-style link (the browser's tunnel) — its own
/// DNS, purecrypto TLS and HTTP, redirects to the CDN included — here over the
/// host's sockets.
#[cfg(feature = "oci-net")]
#[test]
fn a_pull_over_a_network_link_with_nixvm_doing_dns_tls_and_http() {
    if !enabled() {
        return;
    }
    let pull = oci::Pull::new("alpine:3.20", arch()).unwrap();
    let mut net = oci::net::NetPull::new(pull, Box::new(nixvm::kernel::egress::HostEgress));
    let start = std::time::Instant::now();
    while !net.poll().expect("pull") {
        assert!(start.elapsed().as_secs() < 120, "pull took too long");
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
    let img = net.finish().expect("image");
    assert_eq!(img.config.arch(), Some(arch()));
    let mut vm = Vm::boot_image(
        &img,
        &RunOptions {
            cmd: vec!["cat".into(), "/etc/alpine-release".into()],
            ..RunOptions::default()
        },
        256 << 20,
    )
    .expect("boot");
    let (code, out) = run(&mut vm);
    assert_eq!(code, 0, "{out}");
    assert!(out.starts_with("3.20."), "{out}");
}
