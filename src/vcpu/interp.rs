//! Software CPU interpreter backend — the portable, no-acceleration fallback.
//!
//! Decodes and executes guest instructions in a loop, returning [`Exit::Syscall`]
//! when it decodes a syscall instruction. Slower than the hardware backends but
//! runs anywhere and on any guest arch (useful for CI and unsupported hosts).
//!
//! Phase 10 work. Stub for now.

use crate::abi::Arch;

use super::{Backend, GuestMemory, Vcpu, VcpuError};

#[derive(Debug)]
pub struct InterpBackend {
    guest: Arch,
}

impl InterpBackend {
    pub fn new(guest: Arch) -> Result<Self, VcpuError> {
        Ok(Self { guest })
    }
}

impl Backend for InterpBackend {
    fn name(&self) -> &'static str {
        "interp"
    }

    fn guest_arch(&self) -> Arch {
        self.guest
    }

    fn new_vcpu(
        &self,
        _mem: &GuestMemory,
        _entry: u64,
        _stack: u64,
    ) -> Result<Box<dyn Vcpu>, VcpuError> {
        Err(VcpuError::Backend(
            "software interpreter not implemented yet (ROADMAP Phase 10)".into(),
        ))
    }
}
