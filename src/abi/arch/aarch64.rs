//! aarch64 guest syscall table (asm-generic numbering).
//!
//! On arm64 the syscall number is in `x8`, args in `x0..x5`, return in `x0`.

use super::Sysno;

/// Map a raw aarch64 syscall number to the normalized [`Sysno`].
#[must_use]
pub fn decode(nr: u64) -> Sysno {
    match nr {
        63 => Sysno::Read,
        64 => Sysno::Write,
        56 => Sysno::Openat,
        57 => Sysno::Close,
        214 => Sysno::Brk,
        222 => Sysno::Mmap,
        215 => Sysno::Munmap,
        226 => Sysno::Mprotect,
        94 => Sysno::ExitGroup,
        93 => Sysno::Exit,
        96 => Sysno::SetTidAddress,
        29 => Sysno::Ioctl,
        66 => Sysno::Writev,
        160 => Sysno::Uname,
        172 => Sysno::Getpid,
        173 => Sysno::Getppid,
        178 => Sysno::Gettid,
        174 => Sysno::Getuid,
        175 => Sysno::Geteuid,
        176 => Sysno::Getgid,
        177 => Sysno::Getegid,
        113 => Sysno::ClockGettime,
        62 => Sysno::Lseek,
        80 => Sysno::Fstat,
        79 => Sysno::Newfstatat,
        61 => Sysno::Getdents64,
        17 => Sysno::Getcwd,
        49 => Sysno::Chdir,
        278 => Sysno::Getrandom,
        other => Sysno::Unknown(other),
    }
}
