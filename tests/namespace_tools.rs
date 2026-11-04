//! End-to-end: real namespace tools (`unshare`, `nsenter`, `bwrap`) inside the VM.
//!
//! Skipped unless `NIXVM_BWRAP_TAR` names an uncompressed Alpine root that has
//! bubblewrap and util-linux-misc installed (`scripts/alpine-with-bwrap.sh` builds
//! one).
#![cfg(feature = "fstool")]

use nixvm::vm::Vm;

fn run(vm: &mut Vm, command: &str) -> String {
    let marker = "__DONE__";
    vm.write_stdin(format!("{command}; echo {marker}$?\n").as_bytes());
    let mut out = Vec::new();
    for _ in 0..4000 {
        let step = vm
            .pump_for(std::time::Duration::from_millis(50))
            .expect("pump");
        out.extend_from_slice(&step.stdout);
        out.extend_from_slice(&step.stderr);
        if String::from_utf8_lossy(&out).contains(marker) && !step.busy && vm.awaiting_input() {
            break;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[test]
fn real_namespace_tools_work() {
    let Ok(tar_path) = std::env::var("NIXVM_BWRAP_TAR") else {
        eprintln!("NIXVM_BWRAP_TAR not set; skipping namespace tools test");
        return;
    };
    let tar = std::fs::read(&tar_path).expect("read root tar");
    let mut vm = Vm::boot_squashfs(
        &tar,
        vec!["/bin/busybox".to_string(), "sh".to_string()],
        512 * 1024 * 1024,
    )
    .expect("boot");
    let cmds: &[(&str, &str)] = &[
        ("hostname outer", ""),
        ("unshare -Ur id -u", "0\n"),
        ("unshare -Uu -r sh -c 'hostname inner; hostname'", "inner\n"),
        ("hostname", "outer\n"),
        (
            "unshare -Ur --pid --fork --mount-proc sh -c 'echo pid=$$'",
            "pid=1\n",
        ),
        (
            "unshare -Urm sh -c 'mount -t tmpfs t /mnt && touch /mnt/x && ls /mnt'",
            "x\n",
        ),
        ("ls /mnt | wc -l", "0\n"),
        ("unshare -Urn sh -c 'ip link 2>&1 | grep -c tun0; true'", ""),
        // A new network namespace's lo starts down; `ip link set` brings it up.
        (
            "unshare -Urn sh -c 'ip link show lo | grep -c ,UP; ip link set lo up; ip link show lo | grep -c ,UP; ip addr show lo | grep -c 127.0.0.1'",
            "0\n1\n1\n",
        ),
        // bwrap brings it up itself.
        (
            "bwrap --ro-bind / / --dev /dev --proc /proc --unshare-net ip addr show lo | grep -c 127.0.0.1",
            "1\n",
        ),
        (
            "bwrap --ro-bind / / --dev /dev --proc /proc echo bwrap-basic",
            "bwrap-basic\n",
        ),
        (
            "bwrap --ro-bind / / --dev /dev --proc /proc --unshare-pid sh -c 'echo pid=$$'",
            "",
        ),
        (
            "bwrap --ro-bind / / --dev /dev --proc /proc --unshare-user --uid 1234 --gid 5678 id -u",
            "1234\n",
        ),
        (
            "bwrap --ro-bind / / --dev /dev --proc /proc --unshare-uts --hostname boxed hostname",
            "boxed\n",
        ),
        (
            "bwrap --ro-bind / / --dev /dev --proc /proc touch /etc/nope 2>&1 | grep -c 'Read-only'",
            "1\n",
        ),
        (
            "bwrap --ro-bind / / --dev /dev --proc /proc --tmpfs /tmp sh -c 'touch /tmp/ok && ls /tmp'",
            "ok\n",
        ),
        (
            "bwrap --ro-bind /usr /usr --ro-bind /bin /bin --ro-bind /lib /lib --proc /proc --dev /dev sh -c 'ls /etc 2>&1; echo isolated'",
            "isolated\n",
        ),
        (
            "bwrap --ro-bind / / --dev /dev --proc /proc --unshare-all sh -c 'id -u; hostname; echo all-ok'",
            "all-ok\n",
        ),
        (
            "bwrap --ro-bind / / --dev /dev --proc /proc --die-with-parent --new-session --unshare-pid --as-pid-1 sh -c 'echo pid=$$'",
            "pid=1\n",
        ),
    ];
    let mut failures = Vec::new();
    for (cmd, expect) in cmds {
        let out = run(&mut vm, cmd);
        eprintln!("$ {cmd}\n{out}");
        if !out.contains(expect) {
            failures.push((*cmd).to_string());
        }
    }
    assert!(failures.is_empty(), "failed: {failures:#?}");
}
