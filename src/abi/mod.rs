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

    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Aarch64 => "aarch64",
            Self::X86_64 => "x86_64",
        }
    }
}
