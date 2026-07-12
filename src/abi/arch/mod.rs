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
    Mremap,
    Madvise,
    Mlock,
    Mlock2,
    Munlock,
    Mlockall,
    Munlockall,
    Msync,
    Mincore,
    ExitGroup,
    Exit,
    SetTidAddress,
    Ioctl,
    Readv,
    Writev,
    Setitimer,
    Uname,
    Getpid,
    Getppid,
    Gettid,
    Getuid,
    Geteuid,
    Getgid,
    Getegid,
    ClockGettime,
    Gettimeofday,
    ClockGetres,
    Nanosleep,
    ClockNanosleep,
    ClockSettime,
    Settimeofday,
    Adjtimex,
    ClockAdjtime,
    Time,
    Lseek,
    Fstat,
    Newfstatat,
    Getdents64,
    Getcwd,
    Chdir,
    Fchdir,
    Getrandom,
    RtSigaction,
    RtSigprocmask,
    RtSigpending,
    RtSigsuspend,
    RtSigtimedwait,
    RtSigreturn,
    Sigaltstack,
    Kill,
    Tkill,
    Tgkill,
    Futex,
    Fcntl,
    Flock,
    Sendmsg,
    Recvmsg,
    Pipe2,
    Dup,
    Dup2,
    Dup3,
    Clone,
    /// x86-64 only (aarch64 has no `fork` syscall — everything is `clone`).
    Fork,
    /// x86-64 only, like [`Sysno::Fork`].
    Vfork,
    /// x86-64 only (aarch64 has no path-based `stat` — only `newfstatat`).
    Stat,
    /// x86-64 only, like [`Sysno::Stat`].
    Lstat,
    /// x86-64 only: the legacy path-based file syscalls aarch64 replaced
    /// with their `*at` successors. Each dispatches as `<op>at(AT_FDCWD, …)`.
    Open,
    /// x86-64 only — `open(path, O_WRONLY|O_CREAT|O_TRUNC, mode)`.
    Creat,
    /// x86-64 only, like [`Sysno::Open`].
    Mkdir,
    /// x86-64 only — `unlinkat(AT_FDCWD, path, AT_REMOVEDIR)`.
    Rmdir,
    /// x86-64 only, like [`Sysno::Open`].
    Unlink,
    /// x86-64 only, like [`Sysno::Open`].
    Rename,
    /// x86-64 only, like [`Sysno::Open`].
    Symlink,
    /// x86-64 only, like [`Sysno::Open`].
    Readlink,
    Execve,
    Wait4,
    SetRobustList,
    Statfs,
    Fstatfs,
    Readlinkat,
    Symlinkat,
    Mkdirat,
    Unlinkat,
    Renameat,
    Renameat2,
    Faccessat,
    Faccessat2,
    Access,
    Fchmodat,
    Fchmod,
    Fchownat,
    Fchown,
    Utimensat,
    Umask,
    Getxattr,
    Lgetxattr,
    Fgetxattr,
    Socket,
    Socketpair,
    Bind,
    Listen,
    Accept4,
    Connect,
    Getsockname,
    Getpeername,
    Sendto,
    Recvfrom,
    Setsockopt,
    Getsockopt,
    Shutdown,
    SchedYield,
    SchedGetaffinity,
    SchedSetaffinity,
    SchedGetscheduler,
    SchedSetscheduler,
    SchedGetparam,
    SchedGetPriorityMax,
    SchedGetPriorityMin,
    Getrusage,
    Sysinfo,
    Prlimit64,
    Getrlimit,
    Setrlimit,
    Times,
    Getpriority,
    Setpriority,
    Getcpu,
    Prctl,
    Personality,
    Sethostname,
    Setdomainname,
    Capget,
    Capset,
    Membarrier,
    // ---- event-notification / readiness syscalls ----
    Poll,
    Ppoll,
    Select,
    Pselect6,
    EpollCreate,
    EpollCreate1,
    EpollCtl,
    EpollWait,
    EpollPwait,
    EpollPwait2,
    Eventfd,
    Eventfd2,
    TimerfdCreate,
    TimerfdSettime,
    TimerfdGettime,
    /// x86-64-only: set the FS/GS segment base (used for TLS). No aarch64
    /// equivalent (arm64 TLS goes through `TPIDR_EL0`, set directly by the
    /// vcpu on thread creation).
    ArchPrctl,
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
