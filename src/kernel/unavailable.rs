//! Syscalls nixvm deliberately does not provide, and the exact "not here"
//! answer each one gives.
//!
//! Every number in both syscall tables decodes to a [`Sysno`]: the ones in this
//! file are *known* and refused on purpose, so they never land in the
//! [`Kernel::unsupported`] ledger (which is reserved for "the guest called
//! something nobody considered"). The errno for each mirrors what a real Linux
//! kernel answers when it was built without the feature (`ENOSYS` from a
//! `cond_syscall` stub), or what an unprivileged container answers when the
//! capability is missing (`EPERM`) — the two answers feature probes in libc,
//! systemd, util-linux, libaio/liburing users, … are written to fall back from.
//! Picking a *different* error (say `EINVAL` from an I/O-uring probe) can make
//! a caller conclude the feature exists but was misused, and abort instead of
//! falling back.

use super::{Kernel, err};
use crate::abi::arch::Sysno;
use crate::abi::errno::Errno;

impl Kernel {
    /// The fixed answer for a deliberately-unavailable syscall (see the module
    /// docs). Returns `None` for any `sys` that is *not* in the refused set, so
    /// the dispatcher can fall through to its real handlers.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_unavailable(&self, sys: Sysno, args: &[u64; 6]) -> Option<i64> {
        Some(match sys {
            // Kernel built without CONFIG_AIO / CONFIG_IO_URING: libaio users
            // (MySQL/MariaDB, fio) and liburing users (Tokio-uring, QEMU, newer
            // glibc-free runtimes) probe `io_setup`/`io_uring_setup` and fall
            // back to thread pools on ENOSYS.
            Sysno::IoSetup
            | Sysno::IoDestroy
            | Sysno::IoSubmit
            | Sysno::IoCancel
            | Sysno::IoGetevents
            | Sysno::IoPgetevents
            | Sysno::IoUringSetup
            | Sysno::IoUringEnter
            | Sysno::IoUringRegister
            // No PMU, no eBPF, no userfaultfd, no fanotify: each is a
            // `cond_syscall` that answers ENOSYS when its Kconfig is off.
            | Sysno::PerfEventOpen
            | Sysno::Bpf
            | Sysno::Userfaultfd
            | Sysno::FanotifyInit
            | Sysno::FanotifyMark
            // No loadable modules (CONFIG_MODULES=n), no BSD process
            // accounting, no disk quotas.
            | Sysno::InitModule
            | Sysno::FinitModule
            | Sysno::DeleteModule
            | Sysno::Acct
            | Sysno::Quotactl
            | Sysno::QuotactlFd
            // The new mount API: util-linux ≥ 2.39 tries `fsopen`/`open_tree`
            // first and falls back to classic mount(2) (which nixvm accepts)
            // on ENOSYS.
            | Sysno::OpenTree
            | Sysno::OpenTreeAttr
            | Sysno::MoveMount
            | Sysno::Fsopen
            | Sysno::Fsconfig
            | Sysno::Fsmount
            | Sysno::Fspick
            | Sysno::MountSetattr
            | Sysno::Statmount
            | Sysno::Listmount
            // Landlock not compiled in: `landlock_create_ruleset(NULL, 0,
            // LANDLOCK_CREATE_RULESET_VERSION)` probes see ENOSYS and run
            // unsandboxed, which is what every Landlock user is written to do.
            | Sysno::LandlockCreateRuleset
            | Sysno::LandlockAddRule
            | Sysno::LandlockRestrictSelf
            // No kernel keyring (CONFIG_KEYS=n): libkeyutils/Kerberos fall
            // back to file credential caches.
            | Sysno::AddKey
            | Sysno::RequestKey
            | Sysno::Keyctl
            // secretmem is off unless booted with `secretmem.enable=1`, in
            // which case Linux itself answers ENOSYS.
            | Sysno::MemfdSecret
            // No user shadow stacks; glibc only calls this when the ELF opts
            // in *and* the kernel advertised the feature.
            | Sysno::MapShadowStack
            // LSM introspection: no LSM is loaded.
            | Sysno::LsmGetSelfAttr
            | Sysno::LsmSetSelfAttr
            | Sysno::LsmListModules
            // FS_IOC_FSGETXATTR-as-a-syscall (6.17); `chattr`/`lsattr` use the
            // ioctl, which reports ENOTTY on our filesystems.
            | Sysno::FileGetattr
            | Sysno::FileSetattr
            // x86-64 legacy segment/TLS management: the 64-bit-only CPU model
            // has no LDT and no GDT TLS slots (`arch_prctl(ARCH_SET_FS)` is
            // the 64-bit TLS path). A kernel without CONFIG_MODIFY_LDT_SYSCALL
            // answers ENOSYS, which Wine/dosemu-style callers handle.
            | Sysno::ModifyLdt
            | Sysno::SetThreadArea
            | Sysno::GetThreadArea
            // `uselib` (CONFIG_USELIB=n on every modern distro) and `_sysctl`
            // (removed in 5.5) are ENOSYS on real kernels too.
            | Sysno::Uselib
            | Sysno::Sysctl
            // Only ever issued by the kernel's own uretprobe trampoline.
            | Sysno::Uretprobe
            // `sys_ni_syscall` slots: ENOSYS on every Linux.
            | Sysno::NiSyscall => err(Errno::ENOSYS),
            // Privileged machine-level operations a container is never granted
            // (CAP_SYS_BOOT / CAP_SYS_ADMIN dropped): the guest must not be
            // able to kexec, add swap, or pivot the root of the emulated
            // machine. `pivot_root` also fails EINVAL on Linux when `new_root`
            // is not a mount point, but the capability check comes first.
            Sysno::KexecLoad
            | Sysno::KexecFileLoad
            | Sysno::Swapon
            | Sysno::Swapoff
            | Sysno::PivotRoot => err(Errno::EPERM),
            // File handles: no backend can re-open a node from an opaque handle
            // (paths are the only identity), so — like a filesystem without
            // export support — `name_to_handle_at` is EOPNOTSUPP (systemd and
            // util-linux fall back to `statx`/mountinfo), and any handle given
            // to `open_by_handle_at` is necessarily stale.
            Sysno::NameToHandleAt => err(Errno::EOPNOTSUPP),
            Sysno::OpenByHandleAt => err(Errno::ESTALE),
            // `reboot` validates its magic numbers before the capability check
            // (EINVAL for a bad magic), then refuses: the guest cannot power
            // off or restart the emulated machine — init exits instead. The
            // Ctrl-Alt-Del toggles are harmless and accepted.
            Sysno::Reboot => {
                const MAGIC1: u64 = 0xfee1_dead;
                const MAGIC2: [u64; 4] = [672_274_793, 85_072_278, 369_367_448, 537_993_216];
                const CMD_CAD_ON: u64 = 0x89ab_cdef;
                const CMD_CAD_OFF: u64 = 0;
                if args[0] & 0xffff_ffff != MAGIC1 || !MAGIC2.contains(&(args[1] & 0xffff_ffff)) {
                    err(Errno::EINVAL)
                } else if matches!(args[2] & 0xffff_ffff, CMD_CAD_ON | CMD_CAD_OFF) {
                    0
                } else {
                    err(Errno::EPERM)
                }
            }
            _ => return None,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::abi::Arch;
    use crate::abi::arch::{Sysno, decode};
    use crate::abi::errno::Errno;
    use crate::fs::{MountTable, TmpFs};
    use crate::kernel::Kernel;

    fn kernel() -> Kernel {
        let mut mounts = MountTable::new();
        mounts.mount("/", Box::new(TmpFs::new()));
        Kernel::new(Arch::Aarch64, mounts)
    }

    #[test]
    fn every_listed_linux_syscall_number_decodes() {
        // The asm-generic table (aarch64) runs 0..=294 plus 424..=469, with
        // 244..=259 reserved for arch-specific calls arm64 doesn't define and
        // 38 (`renameat`) / 163-164 (`[gs]etrlimit`) present.
        for nr in (0..=243).chain(260..=294).chain(424..=469) {
            assert!(
                !matches!(decode(Arch::Aarch64, nr), Sysno::Unknown(_)),
                "aarch64 syscall {nr} is undecoded"
            );
        }
        // x86-64: 0..=335 contiguous (334 = rseq, 335 = uretprobe), then the
        // shared 424..=469 block.
        for nr in (0..=335).chain(424..=469) {
            assert!(
                !matches!(decode(Arch::X86_64, nr), Sysno::Unknown(_)),
                "x86-64 syscall {nr} is undecoded"
            );
        }
        // Holes in the tables stay unknown (they are not syscalls at all).
        assert_eq!(decode(Arch::X86_64, 400), Sysno::Unknown(400));
        assert_eq!(decode(Arch::Aarch64, 250), Sysno::Unknown(250));
    }

    #[test]
    fn refused_syscalls_answer_like_a_kernel_without_the_feature() {
        let k = kernel();
        let none = [0u64; 6];
        let e = |x: Errno| -i64::from(x.0);
        assert_eq!(
            k.sys_unavailable(Sysno::IoUringSetup, &none),
            Some(e(Errno::ENOSYS))
        );
        assert_eq!(
            k.sys_unavailable(Sysno::NiSyscall, &none),
            Some(e(Errno::ENOSYS))
        );
        assert_eq!(
            k.sys_unavailable(Sysno::Swapon, &none),
            Some(e(Errno::EPERM))
        );
        assert_eq!(
            k.sys_unavailable(Sysno::NameToHandleAt, &none),
            Some(e(Errno::EOPNOTSUPP))
        );
        // reboot: bad magic is EINVAL, a real command EPERM, CAD toggles OK.
        assert_eq!(
            k.sys_unavailable(Sysno::Reboot, &none),
            Some(e(Errno::EINVAL))
        );
        let power_off = [0xfee1_dead, 672_274_793, 0x4321_fedc, 0, 0, 0];
        assert_eq!(
            k.sys_unavailable(Sysno::Reboot, &power_off),
            Some(e(Errno::EPERM))
        );
        let cad_on = [0xfee1_dead, 672_274_793, 0x89ab_cdef, 0, 0, 0];
        assert_eq!(k.sys_unavailable(Sysno::Reboot, &cad_on), Some(0));
        // Anything else is not in the refused set.
        assert_eq!(k.sys_unavailable(Sysno::Read, &none), None);
    }
}
