//! Run the C programs in `tests/guest` on the VM, on each backend, and check that
//! hostile ones are contained.
//!
//! Needs `gcc` with static libraries on an x86-64 Linux host; it skips itself
//! otherwise, printing why. See `tests/guest/README.md`.
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
    DIR.get_or_init(|| {
        let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("guest_root");
        for sub in ["tmp", "etc", "proc", "dev", "sys", "mnt"] {
            std::fs::create_dir_all(dir.join(sub)).expect("create guest root");
        }
        for name in PROGRAMS.iter().chain(&["attacks"]) {
            let built = Command::new("gcc")
                .args(["-static", "-O1", "-w", "-pthread", "-o"])
                .arg(dir.join(name))
                .arg(Path::new("tests/guest").join(format!("{name}.c")))
                .arg("-lrt")
                .output();
            match built {
                Ok(out) if out.status.success() => {}
                Ok(out) => {
                    let why = String::from_utf8_lossy(&out.stderr);
                    eprintln!("can't build {name}, skipping: {why}");
                    return None;
                }
                Err(e) => {
                    eprintln!("no gcc, skipping: {e}");
                    return None;
                }
            }
        }
        // ns_user execs itself by this name.
        std::fs::copy(dir.join("ns_user"), dir.join("ns5")).expect("copy");
        Some(dir)
    })
    .as_deref()
}

/// Run `program` in the VM with `env` set, killing it after a while so that a hang
/// is a failure rather than a stuck test.
fn run(root: &Path, program: &str, args: &[&str], env: &[(&str, &str)], seconds: &str) -> Output {
    let mut cmd = Command::new("timeout");
    cmd.args(["--foreground", seconds])
        .arg(env!("CARGO_BIN_EXE_run-elf-x86"))
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
    let mut failed = Vec::new();
    for name in PROGRAMS {
        let out = run(root, name, &[], env, seconds);
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
