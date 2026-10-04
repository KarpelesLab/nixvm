//! End-to-end: run real x86-64 Linux programs (an Alpine userland, e.g. the
//! minirootfs plus extra packages) on the x86-64 software interpreter.
//!
//! Skipped unless `NIXVM_X86_ROOT` names an x86-64 root directory. Each line of
//! the file named by `NIXVM_X86_CMDS` (default: a built-in busybox/musl smoke
//! list) is run as `/bin/sh -c <line>` in a fresh sandbox, with the host
//! directory `NIXVM_X86_WORK` (default: a temp dir) bound at `/work`; a line
//! starting with `!` must fail, otherwise it must exit 0.
//!
//! ```text
//! NIXVM_X86_ROOT=$PWD/alpine-x86_64 cargo test --release --test x86_programs -- --nocapture
//! ```

use nixvm::{Arch, Sandbox};

const DEFAULT_CMDS: &str = "\
echo hello from x86-64 | grep -q hello
test \"$(uname -m)\" = x86_64
seq 1 1000 | sort -rn | head -n 1 | grep -qx 1000
printf 'abc' | md5sum | grep -q 900150983cd24fb0d6963f7d28e17f72
printf 'abc' | sha256sum | grep -q ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
awk 'BEGIN { s = 0; for (i = 1; i <= 100; i++) s += i / 3.0; printf \"%.6f\\n\", s }' | grep -qx 1683.333333
awk 'BEGIN { printf \"%.10g\\n\", sin(1) + cos(2) + exp(1.5) + log(10) + sqrt(2) + atan2(1, 3) }' | grep -qx 8.945562428
dd if=/dev/zero bs=1k count=256 2>/dev/null | gzip -9 | gzip -d | wc -c | grep -qx 262144
seq 1 5000 | xargs -n 100 echo | wc -l | grep -qx 50
";

#[test]
fn x86_64_programs_run() {
    let Ok(root) = std::env::var("NIXVM_X86_ROOT") else {
        eprintln!("NIXVM_X86_ROOT not set; skipping x86-64 program tests");
        return;
    };
    let cmds = std::env::var("NIXVM_X86_CMDS").map_or_else(
        |_| DEFAULT_CMDS.to_string(),
        |p| std::fs::read_to_string(p).expect("read NIXVM_X86_CMDS"),
    );
    let work = std::env::var("NIXVM_X86_WORK").map_or_else(
        |_| {
            let d = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("x86_programs_work");
            std::fs::create_dir_all(&d).unwrap();
            d
        },
        std::path::PathBuf::from,
    );
    let mut failures = Vec::new();
    for line in cmds
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
    {
        let (expect_fail, cmd) = match line.strip_prefix('!') {
            Some(c) => (true, c.trim()),
            None => (false, line),
        };
        let started = std::time::Instant::now();
        let status = Sandbox::builder()
            .arch(Arch::X86_64)
            .root_dir(&root)
            .bind(&work, "/work")
            .mem_bytes(1 << 30)
            .env("PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
            .env("HOME=/work")
            // (`/bin/sh` is an absolute symlink, which the host-side ELF load
            // of the initial command doesn't resolve inside the root.)
            .command(["/bin/busybox", "sh", "-c", cmd])
            .run();
        let ok = match &status {
            Ok(code) => (*code == 0) != expect_fail,
            Err(_) => false,
        };
        eprintln!(
            "{} [{:>6.2}s] {cmd} -> {status:?}",
            if ok { "ok  " } else { "FAIL" },
            started.elapsed().as_secs_f64()
        );
        if !ok {
            failures.push(cmd.to_string());
        }
    }
    assert!(failures.is_empty(), "failed: {failures:#?}");
}
