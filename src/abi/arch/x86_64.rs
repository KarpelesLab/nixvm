//! x86-64 guest syscall table.
//!
//! On x86-64 the syscall number is in `rax`, args in `rdi, rsi, rdx, r10, r8,
//! r9`, return in `rax`.

use super::Sysno;

/// Map a raw x86-64 syscall number to the normalized [`Sysno`].
#[must_use]
pub fn decode(nr: u64) -> Sysno {
    match nr {
        0 => Sysno::Read,
        1 => Sysno::Write,
        257 => Sysno::Openat,
        3 => Sysno::Close,
        12 => Sysno::Brk,
        9 => Sysno::Mmap,
        11 => Sysno::Munmap,
        10 => Sysno::Mprotect,
        231 => Sysno::ExitGroup,
        60 => Sysno::Exit,
        218 => Sysno::SetTidAddress,
        16 => Sysno::Ioctl,
        20 => Sysno::Writev,
        63 => Sysno::Uname,
        39 => Sysno::Getpid,
        110 => Sysno::Getppid,
        186 => Sysno::Gettid,
        102 => Sysno::Getuid,
        107 => Sysno::Geteuid,
        104 => Sysno::Getgid,
        108 => Sysno::Getegid,
        228 => Sysno::ClockGettime,
        8 => Sysno::Lseek,
        5 => Sysno::Fstat,
        262 => Sysno::Newfstatat,
        217 => Sysno::Getdents64,
        79 => Sysno::Getcwd,
        80 => Sysno::Chdir,
        318 => Sysno::Getrandom,
        13 => Sysno::RtSigaction,
        14 => Sysno::RtSigprocmask,
        202 => Sysno::Futex,
        72 => Sysno::Fcntl,
        293 => Sysno::Pipe2,
        32 => Sysno::Dup,
        33 => Sysno::Dup2,
        292 => Sysno::Dup3,
        other => Sysno::Unknown(other),
    }
}
