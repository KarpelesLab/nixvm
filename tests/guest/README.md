# Guest conformance programs

Small C programs that exercise kernel behaviour from inside a guest, each printing
`PASS` at the end. `tests/guest_conformance.rs` builds them with `gcc -static` and
runs them on the VM: on x86-64 on each backend, and on an aarch64 guest when an
aarch64 cross compiler is installed (`aarch64-linux-gnu-gcc`, or set
`NIXVM_AARCH64_CC`).

They're written to pass on real Linux too, which is how to check an expectation:

```text
gcc -static -pthread -o /tmp/ns_pid tests/guest/ns_pid.c
unshare -Urm /tmp/ns_pid                  # the ns_* programs want root in a user namespace
```

`ns_user` starts as an ordinary user, so run that one with no `unshare` at all, and
`ns_mount` with `unshare -Urmp --fork --mount-proc`, and `ns_net` with
`unshare -Urn sh -c 'ip link set lo up && ./ns_net'` (a new namespace's `lo` starts down). `cgroups` wants root and cgroup v2 (`sudo ./cgroups`; it makes a subtree of its own under
the root cgroup and removes it again). `rlimits` wants root too (`sudo ./rlimits`: it
raises hard limits and runs `SCHED_FIFO` tasks), and so does `sched` (`sudo ./sched`:
it runs real-time tasks on CPU 0 for a few seconds). `exec` needs nothing: it writes
tiny ELF images of its own to exec, and neither do `fifo`, `devfd`, `threads_mm`, `dualstack`, `madvise`, `execthread`, `memfd`, `sysv_shm`, `threadwait`, `connect_again`, `pty`, `exit_race`, `unix_scm`, `compat32` (x86-64: 32-bit code and `int $0x80`; build it non-PIE, as `-static` does) and `compat32_insn` (x86-64: the instructions only 32-bit code has — BCD, `BOUND`, `ARPL`, `LES`/`LDS`, segment bases and limits, 16-bit addressing, 16-bit code and stack segments; non-PIE too); `ia32` is an i386 program (`gcc -m32 -static -pthread -o ia32 ia32.c`; the VM runs it as an ELFCLASS32 process, and `compat32` execs it from 64-bit code when it is next to it), skipped where there is no 32-bit libc; `ptrace32` is another (`-m32`: ptrace and seccomp of i386 calls from an i386 tracer), and `ptrace_ia32` (x86-64, non-PIE) traces `int $0x80` from 64-bit code and, when it is next to it, `ptrace32` run as an i386 tracee; `ptrace_regs` (x86-64) checks what a debugger sees beyond the general registers and when — `si_errno`, a `SIGCONT` after a stop, FP state and `rip` at syscall stops, the debug registers and their traps — and `ptrace_regs32` (`-m32`) the same through the i386 paths; `creation` checks who owns what it makes only as root (`sudo ./creation`). CI runs those
on the runner as well as in the VM.
`userns_caps` and `sandbox_fs` run as anyone, making their own user namespaces;
as root `userns_caps` also checks how dropping uid 0 drops capabilities, so CI
runs both as root on the runner, and they're worth running as an ordinary user too.
`attacks.c` isn't a pass/fail
program: the Rust test checks that each thing it tries is contained.
`protect.c` isn't a Linux program either: it checks `Passthrough::protect` in a
`/work` the Rust test lays out and protects (`NIXVM_PROTECT`), and the test then
checks the host directory is unchanged.
