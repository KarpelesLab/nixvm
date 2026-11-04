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
`unshare -Urn sh -c 'ip link set lo up && ./ns_net'` (a new namespace's `lo` starts down). `attacks.c` isn't a pass/fail
program: the Rust test checks that each thing it tries is contained.
