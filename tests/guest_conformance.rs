//! Run the C programs in `tests/guest` on the VM, on each backend, and check that
//! hostile ones are contained.
//!
//! Needs `gcc` with static libraries on an x86-64 Linux host; it skips itself
//! otherwise, printing why. The same programs run on an aarch64 guest too when
//! a cross compiler is installed (`aarch64-linux-gnu-gcc`, or whatever
//! `NIXVM_AARCH64_CC` names). See `tests/guest/README.md`.
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::OnceLock;

const PROGRAMS: &[&str] = &[
    "sigshare",
    "sockopt",
    "queues",
    "rlimits",
    "async_signals",
    "ns_uts_ipc",
    "ns_net",
    "ns_pid",
    "ns_mount",
    "ns_user",
];

/// A directory of binaries, also used as the guest's root, so that a program can
/// `exec` itself there. `None` if there's no compiler to build them with.
fn built() -> Option<&'static Path> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| build_with("gcc", "guest_root"))
        .as_deref()
}

/// The same, built for an aarch64 guest by a cross compiler.
fn built_aarch64() -> Option<&'static Path> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| {
        let named = std::env::var("NIXVM_AARCH64_CC").ok();
        let candidates = named.into_iter().chain([
            "aarch64-linux-gnu-gcc".into(),
            "aarch64-unknown-linux-gnu-gcc".into(),
        ]);
        let found = candidates.into_iter().find(|cc| {
            Command::new(cc)
                .arg("--version")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .is_ok_and(|s| s.success())
        });
        let Some(cc) = found else {
            eprintln!("no aarch64 cross compiler, skipping the aarch64 guest");
            return None;
        };
        build_with(&cc, "guest_root_aarch64")
    })
    .as_deref()
}

fn build_with(cc: &str, dir_name: &str) -> Option<PathBuf> {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(dir_name);
    for sub in ["tmp", "etc", "proc", "dev", "sys", "mnt"] {
        std::fs::create_dir_all(dir.join(sub)).expect("create guest root");
    }
    for name in PROGRAMS.iter().chain(&["attacks"]) {
        let built = Command::new(cc)
            .args(["-static", "-O1", "-w", "-pthread", "-o"])
            .arg(dir.join(name))
            .arg(Path::new("tests/guest").join(format!("{name}.c")))
            .arg("-lrt")
            .output();
        match built {
            Ok(out) if out.status.success() => {}
            Ok(out) => {
                let why = String::from_utf8_lossy(&out.stderr);
                eprintln!("can't build {name} with {cc}, skipping: {why}");
                return None;
            }
            Err(e) => {
                eprintln!("no {cc}, skipping: {e}");
                return None;
            }
        }
    }
    // ns_user execs itself by this name.
    std::fs::copy(dir.join("ns_user"), dir.join("ns5")).expect("copy");
    Some(dir)
}

/// Run `program` in the VM with `env` set, killing it after a while so that a hang
/// is a failure rather than a stuck test.
fn run(root: &Path, program: &str, args: &[&str], env: &[(&str, &str)], seconds: &str) -> Output {
    run_on(
        env!("CARGO_BIN_EXE_run-elf-x86"),
        root,
        program,
        args,
        env,
        seconds,
    )
}

/// [`run`] with a given runner: `run-elf-x86`, or `run-elf` for an aarch64 guest.
fn run_on(
    runner: &str,
    root: &Path,
    program: &str,
    args: &[&str],
    env: &[(&str, &str)],
    seconds: &str,
) -> Output {
    let mut cmd = Command::new("timeout");
    cmd.args(["--foreground", seconds])
        .arg(runner)
        .arg(root.join(program))
        .arg(program)
        .args(args)
        .env("NIXVM_ROOT", root)
        .env_remove("NIXVM_INTERP")
        .env_remove("NIXVM_CPUS")
        .stdin(Stdio::null());
    for (key, value) in env {
        cmd.env(key, value);
    }
    cmd.output().expect("run the VM")
}

fn text(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

/// Every program, on one configuration.
fn conformance(env: &[(&str, &str)], seconds: &str) {
    let Some(root) = built() else { return };
    conformance_of(env!("CARGO_BIN_EXE_run-elf-x86"), root, env, seconds);
}

fn conformance_of(runner: &str, root: &Path, env: &[(&str, &str)], seconds: &str) {
    let mut failed = Vec::new();
    for name in PROGRAMS {
        let out = run_on(runner, root, name, &[], env, seconds);
        let text = text(&out);
        let passed = text.lines().any(|l| l == "PASS");
        let failures: Vec<&str> = text.lines().filter(|l| l.starts_with("FAIL")).collect();
        if !passed || !failures.is_empty() {
            let tail: Vec<&str> = text.lines().rev().take(6).collect();
            failed.push(format!("{name}: {failures:?} (last lines: {tail:?})"));
        }
    }
    assert!(failed.is_empty(), "{failed:#?}");
}

#[test]
fn programs_pass_on_the_default_backend() {
    conformance(&[], "120");
}

#[test]
fn programs_pass_on_the_interpreter() {
    conformance(&[("NIXVM_INTERP", "1")], "300");
}

#[test]
fn programs_pass_with_several_cpus() {
    conformance(&[("NIXVM_CPUS", "3")], "120");
}

/// On an aarch64 guest, which runs on the interpreter on this host.
#[test]
fn programs_pass_on_an_aarch64_guest() {
    let Some(root) = built_aarch64() else { return };
    conformance_of(env!("CARGO_BIN_EXE_run-elf"), root, &[], "300");
}

/// Each attempt must come back as an ordinary error in the guest, or a clean kill of
/// the guest. What must never happen is the VM's own process dying from a signal or
/// a panic, or running out the clock.
fn contained(args: &[&str], env: &[(&str, &str)], seconds: &str) -> String {
    let Some(root) = built() else {
        return String::new();
    };
    let out = run(root, "attacks", args, env, seconds);
    let text = text(&out);
    assert!(
        !text.contains("panicked"),
        "{args:?} panicked the VM:\n{text}"
    );
    assert!(
        !text.contains("memory allocation of"),
        "{args:?} aborted the VM:\n{text}"
    );
    let code = out.status.code();
    assert!(
        code.is_some(),
        "{args:?} killed the VM: {:?}\n{text}",
        out.status
    );
    assert_ne!(code, Some(124), "{args:?} ran out the clock:\n{text}");
    text
}

#[test]
fn absurd_sizes_are_errors_not_crashes() {
    for (attack, expect) in [
        ("ftruncate", "errno 28"),
        ("pwrite", "errno 27"),
        ("fallocate", "errno 28"),
        ("poll", "errno 22"),
        ("mincore", "errno 12"),
        ("recvmsg", "errno 90"),
        ("mempolicy", "errno 22"),
    ] {
        let out = contained(&[attack], &[], "60");
        if !out.is_empty() {
            assert!(out.contains(expect), "{attack}: want {expect}:\n{out}");
        }
    }
}

#[test]
fn a_fork_bomb_stops_at_the_task_limit() {
    let out = contained(&["forkbomb"], &[("NIXVM_LIMIT_TASKS", "200")], "60");
    if !out.is_empty() {
        assert!(out.contains("errno 11"), "fork should be EAGAIN:\n{out}");
        assert!(out.contains("stopped at 200"), "{out}");
    }
}

#[test]
fn filling_tmpfs_stops_at_the_memory_limit() {
    let limit = ("NIXVM_LIMIT_KERNEL_MEMORY", "67108864");
    let out = contained(&["tmpfs"], &[limit], "60");
    if !out.is_empty() {
        assert!(out.contains("errno 28"), "write should be ENOSPC:\n{out}");
    }
}

#[test]
fn a_loop_of_forks_outlasts_the_kvm_vcpu_id_limit() {
    let out = contained(&["forkloop", "6000"], &[], "180");
    if !out.is_empty() {
        assert!(out.contains("done 6000"), "{out}");
    }
}

#[test]
fn running_out_of_guest_ram_on_copy_on_write_kills_the_task_not_the_vm() {
    let out = contained(&["cowfill", "200"], &[], "120");
    if !out.is_empty() {
        assert!(
            out.contains("child status 0xb"),
            "expected a SIGSEGV:\n{out}"
        );
    }
}

#[test]
fn a_spinning_guest_is_stopped_by_the_time_limits() {
    for (limit, env) in [
        ("run time", ("NIXVM_LIMIT_WALL_SECONDS", "1")),
        ("CPU time", ("NIXVM_LIMIT_CPU_SECONDS", "1")),
    ] {
        let out = contained(&["spin"], &[env], "60");
        if !out.is_empty() {
            assert!(out.contains(limit), "{limit}:\n{out}");
            assert!(out.contains("Ok(137)"), "killed with SIGKILL:\n{out}");
        }
    }
}
