# nixvm

**A portable, VM-style sandbox that runs a real Linux userland by emulating
Linux syscalls directly — not by emulating hardware.**

nixvm lets you run normally-dangerous tasks (`npm install`, `pip install`,
untrusted build scripts, `apk add`, …) inside an isolated Linux environment
that starts in milliseconds and runs anywhere — with or without hardware
virtualization.

## How it's different

A traditional VM (QEMU, Firecracker) boots a full guest Linux kernel and
emulates hardware: a CPU, interrupt controller, virtio devices, block devices.
That's heavy and slow to start.

nixvm takes the [gVisor](https://gvisor.dev/)-style approach instead: **there is
no guest kernel.** We ship a real Linux *userland* (Alpine by default) and run
its processes directly on the host CPU using hardware virtualization
(Hypervisor.framework on macOS/arm64, KVM on Linux). When a guest process
executes a `syscall`/`svc` instruction, the CPU traps out and **nixvm's own
"kernel" — written in Rust — services the syscall.** Files, memory, processes,
signals, and networking are all implemented in userspace, under our control and
resource limits.

A software CPU interpreter provides a fallback path so nixvm runs even where no
hardware virtualization is available (at reduced speed).

## Default sandbox layout

```
/           read-only Alpine root  (pre-packaged squashfs + copy-on-write overlay)
/work       the host's current working directory, passed through read-write
/tmp,/dev,  synthesized in-sandbox (tmpfs / devtmpfs / procfs / sysfs)
/proc,/sys
```

```sh
# Run a command inside the sandbox, with the current dir mounted at /work:
nixvm run -- npm install

# Interactive shell:
nixvm shell
```

## Targets

| Host              | Backend                    | Guest arch     |
| ----------------- | -------------------------- | -------------- |
| macOS / arm64     | Hypervisor.framework (HVF) | arm64          |
| Linux / arm64     | KVM                        | arm64          |
| Linux / x86-64    | KVM                        | x86-64         |
| anywhere          | software interpreter       | arm64 / x86-64 |

## Status

Early scaffold. See [ROADMAP.md](ROADMAP.md) for the phased plan.

## License

Apache-2.0 OR MIT, at your option.
