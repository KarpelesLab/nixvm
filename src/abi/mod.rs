//! Linux syscall ABI surface: the *facts* of the kernel ABI — error numbers,
//! syscall numbers, and C struct layouts — so the loader, kernel, and backends
//! all agree on the same numbers. Pure data + tiny helpers; nothing executes.

pub mod arch;
pub mod errno;

pub use errno::Errno;

/// Guest target architecture nixvm can host.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Arch {
    Aarch64,
    X86_64,
}

impl Arch {
    /// The architecture matching the host CPU, if nixvm can run it with
    /// hardware virtualization here.
    #[must_use]
    pub const fn host_native() -> Option<Self> {
        #[cfg(target_arch = "aarch64")]
        {
            Some(Self::Aarch64)
        }
        #[cfg(target_arch = "x86_64")]
        {
            Some(Self::X86_64)
        }
        #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
        {
            None
        }
    }

    /// The architecture an ELF image is built for, from its header's
    /// `e_machine` (offset 18): `EM_AARCH64` (183) or `EM_X86_64` (62).
    /// `None` for anything else, or a truncated header.
    #[must_use]
    pub fn from_elf(elf: &[u8]) -> Option<Self> {
        const EM_X86_64: u16 = 62;
        const EM_AARCH64: u16 = 183;
        match u16::from_le_bytes([*elf.get(18)?, *elf.get(19)?]) {
            EM_AARCH64 => Some(Self::Aarch64),
            EM_X86_64 => Some(Self::X86_64),
            _ => None,
        }
    }

    /// Parse an architecture name as users spell it: `aarch64`/`arm64`,
    /// `x86_64`/`x86-64`/`amd64` (case-insensitive).
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "aarch64" | "arm64" => Some(Self::Aarch64),
            "x86_64" | "x86-64" | "amd64" | "x64" => Some(Self::X86_64),
            _ => None,
        }
    }

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Aarch64 => "aarch64",
            Self::X86_64 => "x86_64",
        }
    }
}
