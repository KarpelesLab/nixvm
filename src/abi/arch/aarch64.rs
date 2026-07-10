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
        other => Sysno::Unknown(other),
    }
}
