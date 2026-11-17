//! Root images an embedder runs: a released squashfs fetched and booted
//! straight from the file, and a persistent writable layer that keeps what one
//! guest did for the next.
//!
//! * `a_root_image_boots` needs `NIXVM_ROOT_IMAGE` (a path or URL of a
//!   squashfs root) and `NIXVM_ROOT_IMAGE_SHA256`; `NIXVM_ROOT_IMAGE_EXPECT`
//!   lists programs it must have (with `gcc` among them, a C program is built
//!   and run too). The images workflow runs it on every image it builds.
//! * `the_persistent_upper_outlives_the_guest` needs `NIXVM_ALPINE_TAR` (an
//!   uncompressed minirootfs `.tar`), as `tests/alpine_boot.rs`.
//!
//! Both skip themselves without these (CI checks they don't).

#![cfg(feature = "fstool")]

use nixvm::fs::{FsToolMount, MountFs, Overlay};
use nixvm::vm::Vm;

fn drain(vm: &mut Vm) -> String {
    let mut out = Vec::new();
    for _ in 0..256 {
        let step = vm.pump().expect("pump");
        out.extend_from_slice(&step.stdout);
        out.extend_from_slice(&step.stderr);
        if step.exit_code.is_some() || (step.stdout.is_empty() && step.stderr.is_empty()) {
            break;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Run `script` in a shell booted from `root`; everything it printed.
fn run(root: Box<dyn MountFs>, script: &str) -> String {
    let mut vm = Vm::boot_root(
        root,
        vec!["/bin/busybox".into(), "sh".into()],
        512 * 1024 * 1024,
    )
    .expect("boot");
    let _ = drain(&mut vm);
    vm.write_stdin(format!("{script}\necho END-OF-SCRIPT\nexit\n").as_bytes());
    let mut out = String::new();
    for _ in 0..64 {
        out.push_str(&drain(&mut vm));
        if vm.exit_code().is_some() {
            break;
        }
    }
    eprintln!("--- {script}\n{out}");
    assert!(
        out.contains("END-OF-SCRIPT"),
        "the script ran to the end: {out}"
    );
    out
}

struct TempDir(std::path::PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let p = std::env::temp_dir().join(format!("nixvm-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        Self(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[cfg(feature = "images")]
#[test]
fn a_root_image_boots() {
    use nixvm::image::{ImageRef, fetch, parse_hex};
    use std::fmt::Write as _;
    let (Ok(source), Ok(sha)) = (
        std::env::var("NIXVM_ROOT_IMAGE"),
        std::env::var("NIXVM_ROOT_IMAGE_SHA256"),
    ) else {
        eprintln!("NIXVM_ROOT_IMAGE/NIXVM_ROOT_IMAGE_SHA256 not set; skipping root image test");
        return;
    };
    let expect = std::env::var("NIXVM_ROOT_IMAGE_EXPECT").unwrap_or_default();
    let cache = TempDir::new("root-image-cache");
    let image = ImageRef {
        distro: "test".into(),
        version: "0".into(),
        arch: nixvm::Arch::X86_64, // only names the cache file
        sha256: Some(parse_hex(&sha).expect("a 64-digit sha256")),
        source: Some(source),
    };
    let lower = fetch(&image, &cache.0).expect("fetch and open the image");
    let root = Overlay::new(Box::new(lower), Box::new(nixvm::fs::TmpFs::new()));
    let mut script = String::from("cat /etc/alpine-release; apk --version\n");
    for p in expect.split_whitespace() {
        let _ = writeln!(script, "command -v {p} >/dev/null || echo MISSING-{p}");
    }
    if expect.split_whitespace().any(|p| p == "gcc") {
        script.push_str(
            "printf '#include <stdio.h>\\nint main(void){puts(\"built-by-gcc\");return 0;}\\n' > /tmp/h.c && gcc -O2 -o /tmp/h /tmp/h.c && /tmp/h\n",
        );
    }
    let out = run(Box::new(root), &script);
    assert!(out.contains("apk-tools"), "apk runs: {out}");
    assert!(
        !out.contains("MISSING-"),
        "every expected program is there: {out}"
    );
    if expect.split_whitespace().any(|p| p == "gcc") {
        assert!(out.contains("built-by-gcc"), "gcc builds a program: {out}");
    }
}

#[cfg(unix)]
#[test]
fn the_persistent_upper_outlives_the_guest() {
    let Ok(tar_path) = std::env::var("NIXVM_ALPINE_TAR") else {
        eprintln!("NIXVM_ALPINE_TAR not set; skipping persistent upper test");
        return;
    };
    let tar = std::fs::read(&tar_path).expect("read Alpine tar");
    let upper = TempDir::new("persistent-upper");
    let root = || {
        let lower = FsToolMount::from_tar(&tar).expect("squashfs from the tar");
        Box::new(Overlay::persistent(Box::new(lower), &upper.0).expect("the upper"))
    };

    // The first guest adds, changes and deletes, in the base image and not.
    run(
        root(),
        "echo kept > /etc/kept; echo changed >> /etc/motd; rm /etc/issue; \
         mkdir -p /opt/tool/bin; echo '#!/bin/sh' > /opt/tool/bin/t; chmod 755 /opt/tool/bin/t; \
         rm -r /media; mkdir /media; touch /media/fresh; mv /mnt /mnt2",
    );
    assert!(upper.0.join("etc/kept").is_file());
    assert!(
        upper.0.join("etc/.wh.issue").is_file(),
        "a whiteout on the host"
    );

    // The next guest, over a fresh copy of the image, sees all of it.
    let out = run(
        root(),
        "cat /etc/kept; tail -n1 /etc/motd; ls /etc/issue; ls -l /opt/tool/bin/t; \
         echo media:$(ls /media); ls -d /mnt; ls -d /mnt2 && echo mnt2-there",
    );
    assert!(out.contains("kept"), "{out}");
    assert!(out.contains("changed"), "{out}");
    assert!(out.contains("/etc/issue: No such file"), "{out}");
    assert!(out.contains("-rwxr-xr-x"), "{out}");
    assert!(out.contains("media:fresh\n"), "only the new /media: {out}");
    assert!(out.contains("/mnt: No such file"), "{out}");
    assert!(out.contains("mnt2-there"), "{out}");

    // And it can undo what the first one did.
    run(root(), "rm /etc/kept; rm -r /opt; mkdir /mnt");
    let out = run(root(), "ls /etc/kept; ls /opt; echo mnt:$(ls /mnt | wc -l)");
    assert!(out.contains("/etc/kept: No such file"), "{out}");
    assert!(out.contains("/opt: No such file"), "{out}");
    assert!(out.contains("mnt:0"), "{out}");
}
