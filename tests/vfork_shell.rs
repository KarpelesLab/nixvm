#![cfg(feature = "fstool")]
use nixvm::vm::Vm;

// Mimic the frontend EXACTLY: one pump() per Enter. Regression guard for the
// vfork bug where the child's execve clobbered the shared address space and
// corrupted the parent shell (vi/sh fighting for the console; later commands
// hanging). After the fix the shell survives an arbitrary command sequence.
fn one_pump(vm: &mut Vm, label: &str) -> String {
    let s = vm.pump().expect("pump");
    let out = String::from_utf8_lossy(&[s.stdout.clone(), s.stderr.clone()].concat()).into_owned();
    eprintln!("[{label}] exit_code={:?} out={out:?}", s.exit_code);
    out
}

#[test]
fn shell_survives_a_sequence_of_vfork_exec_commands() {
    let Ok(p) = std::env::var("NIXVM_ALPINE_TAR") else {
        return;
    };
    let tar = std::fs::read(&p).unwrap();
    let mut vm =
        Vm::boot_squashfs(&tar, vec!["/bin/busybox".into(), "sh".into()], 256 << 20).unwrap();
    one_pump(&mut vm, "boot");
    // Run several external commands back-to-back. Each forks+execs busybox; if
    // vfork shared (and execve clobbered) the shell's memory, the shell would
    // die after the first and later commands would produce nothing.
    for cmd in ["uname -a\n", "ls /\n", "uptime\n", "ls /\n", "echo alive\n"] {
        vm.write_stdin(cmd.as_bytes());
        let out = one_pump(&mut vm, cmd.trim_end());
        assert!(
            !out.is_empty(),
            "command {cmd:?} produced no output (shell corrupted?)"
        );
    }
}
