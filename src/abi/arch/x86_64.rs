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
        other => Sysno::Unknown(other),
    }
}
