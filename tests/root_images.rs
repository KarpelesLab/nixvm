//! Root images an embedder runs: an image pulled from a registry and booted
//! straight from its cached squashfs, and a persistent writable layer that
//! keeps what one guest did for the next.
//!
//! * `a_pulled_image_boots` needs `NIXVM_PULL_IMAGE`: an image reference, or
//!   `alpine` for `nixvm::image::ALPINE`; it pulls from the real registry,
//!   for `NIXVM_PULL_ARCH` (`x86_64`/`aarch64`; default the host's).
//!   With `NIXVM_NET=host` as well, it `apk add`s make into a persistent upper
//!   and checks that the next guest has it.
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
    // Long enough for an `apk add` over the network.
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(300);
    while std::time::Instant::now() < deadline {
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
fn a_pulled_image_boots() {
    nixvm::env::Env::from_process().install();
    let Ok(reference) = std::env::var("NIXVM_PULL_IMAGE") else {
        eprintln!("NIXVM_PULL_IMAGE not set; skipping the registry pull test");
        return;
    };
    let reference = if reference == "alpine" {
        nixvm::image::ALPINE.to_string()
    } else {
        reference
    };
    let arch = match std::env::var("NIXVM_PULL_ARCH").as_deref() {
        Ok("aarch64") => nixvm::Arch::Aarch64,
        Ok("x86_64") => nixvm::Arch::X86_64,
        _ if cfg!(target_arch = "aarch64") => nixvm::Arch::Aarch64,
        _ => nixvm::Arch::X86_64,
    };
    let cache = TempDir::new("pull-cache");
    let opts = nixvm::image::PullOptions::default;
    let pulled =
        nixvm::image::pull_image(&reference, arch, &cache.0, opts()).expect("pull the image");
    let size = std::fs::metadata(&pulled.path).expect("the root").len();
    eprintln!(
        "pulled {} as {}: {} ({size} bytes)",
        pulled.reference,
        pulled.manifest_digest,
        pulled.path.display()
    );
    let lower = pulled.open().expect("open the root");
    let root = Overlay::new(Box::new(lower), Box::new(nixvm::fs::TmpFs::new()));
    let out = run(Box::new(root), "cat /etc/alpine-release; apk --version");
    assert!(out.contains("apk-tools"), "apk runs: {out}");

    // Pulled again: from the cache, the same root.
    let again = nixvm::image::pull_image(&reference, arch, &cache.0, opts()).expect("again");
    assert_eq!(again.path, pulled.path);

    // With the network, what `apk add` installs in a persistent upper is
    // there for the next guest: a "dev image" as a layer of one's own.
    if std::env::var("NIXVM_NET").as_deref() == Ok("host") {
        let upper = TempDir::new("pull-upper");
        let root = || {
            let lower = again.open().expect("open the root");
            Box::new(Overlay::persistent(Box::new(lower), &upper.0).expect("the upper"))
        };
        // apk's exit status isn't checked: as a host user who isn't root, the
        // upper directory can't take apk's chown to root (EPERM), which apk
        // reports as an error after installing.
        let out = run(root(), "apk add --no-progress make 2>&1 | tail -n 5");
        assert!(
            out.contains("OK:") || out.contains("MiB in"),
            "apk add: {out}"
        );
        let out = run(root(), "make --version | head -n1");
        assert!(out.contains("GNU Make"), "make is still there: {out}");
    }
}

#[cfg(any(unix, nixvm_linux))]
#[test]
fn the_persistent_upper_outlives_the_guest() {
    nixvm::env::Env::from_process().install();
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
