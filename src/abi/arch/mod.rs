//! Per-architecture ABI tables.
//!
//! The kernel dispatches on a *normalized* [`Sysno`] so handlers are written
//! once; these tables map raw guest syscall numbers to that enum.

pub mod aarch64;
pub mod x86_64;

use crate::abi::Arch;

/// Architecture-neutral syscall identity.
///
/// Guest arm64 and x86-64 use different raw numbers for the same operation; the
/// kernel works in terms of this enum. Only a starter set is listed — entries
/// are added as handlers come online (see ROADMAP phases).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Sysno {
    Read,
    Write,
    Openat,
    Close,
    Brk,
    Mmap,
    Munmap,
    Mprotect,
    ExitGroup,
    Exit,
    SetTidAddress,
    Ioctl,
    Writev,
    Uname,
    Getpid,
    Getppid,
    Gettid,
    Getuid,
    Geteuid,
    Getgid,
    Getegid,
    ClockGettime,
    Lseek,
    Fstat,
    Newfstatat,
    Getdents64,
    Getcwd,
    Chdir,
    Getrandom,
    RtSigaction,
    RtSigprocmask,
    Futex,
    Fcntl,
    /// A raw guest number with no mapping yet — handled as ENOSYS but logged.
    Unknown(u64),
}

/// Decode a raw guest syscall number for `arch` into the normalized [`Sysno`].
#[must_use]
pub fn decode(arch: Arch, nr: u64) -> Sysno {
    match arch {
        Arch::Aarch64 => aarch64::decode(nr),
        Arch::X86_64 => x86_64::decode(nr),
    }
}
