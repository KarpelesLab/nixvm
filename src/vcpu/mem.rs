//! Guest memory.
//!
//! For the hardware backends this is a host allocation mapped into the guest at
//! a fixed guest-virtual base (each guest process has a flat address space that
//! nixvm manages). For the interpreter it's a byte arena with a permission map.
//!
//! Phase 1 uses a simple flat model; a page-granular manager (region tree, COW,
//! file-backed maps) arrives in Phase 2 alongside `mmap`/`mprotect`.

/// Page protection bits (mirrors `PROT_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Prot(pub u8);

impl Prot {
    pub const NONE: Prot = Prot(0);
    pub const READ: Prot = Prot(1);
    pub const WRITE: Prot = Prot(2);
    pub const EXEC: Prot = Prot(4);

    #[must_use]
    pub const fn rw() -> Prot {
        Prot(Self::READ.0 | Self::WRITE.0)
    }
    #[must_use]
    pub const fn rx() -> Prot {
        Prot(Self::READ.0 | Self::EXEC.0)
    }
    #[must_use]
    pub const fn contains(self, other: Prot) -> bool {
        self.0 & other.0 == other.0
    }
}

#[derive(Debug)]
pub enum MemError {
    /// Address not mapped in the guest address space.
    Unmapped(u64),
    /// Access violates the region's protection.
    Protection { addr: u64, needed: Prot },
    /// Host-side allocation/mapping failure.
    Host(String),
}

/// The guest address space shared by every vcpu of one guest process.
///
/// Stub for the scaffold: real page management lands in Phase 2. The API is
/// what the kernel and backends will call.
#[derive(Debug)]
pub struct GuestMemory {
    base: u64,
    size: u64,
}

impl GuestMemory {
    /// Reserve a flat guest region `[base, base+size)`.
    #[must_use]
    pub fn new(base: u64, size: u64) -> Self {
        Self { base, size }
    }

    #[must_use]
    pub fn base(&self) -> u64 {
        self.base
    }
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Copy `buf` into guest memory at `addr`. (Phase 2 honors protections.)
    pub fn write(&mut self, _addr: u64, _buf: &[u8]) -> Result<(), MemError> {
        Err(MemError::Host("GuestMemory::write not implemented (Phase 2)".into()))
    }

    /// Copy `len` bytes out of guest memory at `addr`.
    pub fn read(&self, _addr: u64, _len: usize) -> Result<Vec<u8>, MemError> {
        Err(MemError::Host("GuestMemory::read not implemented (Phase 2)".into()))
    }
}
