//! Turning an ELF file into a ready-to-run guest process image.
//!
//! Responsibilities:
//! * parse ELF64 headers and program headers,
//! * map `PT_LOAD` segments into [`GuestMemory`] with correct protections,
//! * for dynamic executables, load the `PT_INTERP` dynamic linker
//!   (musl's `ld-musl-*.so`) and hand *it* control,
//! * build the initial stack: `argc`, `argv`, `envp`, and the auxiliary vector
//!   (`AT_PHDR`, `AT_ENTRY`, `AT_RANDOM`, `AT_SYSINFO_EHDR` for the vDSO, …),
//! * report the entry PC and initial SP for the first [`crate::vcpu::Vcpu`].
//!
//! Static loading lands in Phase 2; dynamic-linker support in Phase 5.

use crate::vcpu::{GuestMemory, MemError};

/// Where to start executing and how the stack was laid out.
#[derive(Debug, Clone)]
pub struct LoadedImage {
    /// Entry PC (the interpreter's entry for dynamic executables).
    pub entry: u64,
    /// Initial stack pointer, pointing at `argc`.
    pub stack_pointer: u64,
    /// Program break (end of the highest `PT_LOAD`), where `brk` starts.
    pub program_break: u64,
}

/// What the guest should be started with.
#[derive(Debug, Clone)]
pub struct ProcessSpec {
    pub argv: Vec<String>,
    pub envp: Vec<String>,
}

#[derive(Debug)]
pub enum LoadError {
    NotElf,
    UnsupportedArch,
    Truncated,
    Mem(MemError),
    Unimplemented(&'static str),
}

impl core::fmt::Display for LoadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotElf => write!(f, "not an ELF64 file"),
            Self::UnsupportedArch => write!(f, "unsupported ELF machine type"),
            Self::Truncated => write!(f, "ELF file is truncated"),
            Self::Mem(e) => write!(f, "guest memory error: {e:?}"),
            Self::Unimplemented(w) => write!(f, "loader: {w} not implemented"),
        }
    }
}

impl std::error::Error for LoadError {}

/// Load a statically-linked ELF64 executable into `mem`.
///
/// Stub: implemented in Phase 2.
pub fn load_static(
    _mem: &mut GuestMemory,
    _elf: &[u8],
    _spec: &ProcessSpec,
) -> Result<LoadedImage, LoadError> {
    Err(LoadError::Unimplemented("static ELF loading (Phase 2)"))
}
