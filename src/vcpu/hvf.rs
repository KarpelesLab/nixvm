//! Hypervisor.framework backend (macOS / arm64) — the primary target.
//!
//! Plan (Phase 1): create a VM via `hv_vm_create`, map [`GuestMemory`] with
//! `hv_vm_map`, create a vcpu with `hv_vcpu_create`, and run guest code at
//! EL1/EL0. A guest `svc #0` traps out as `HV_EXIT_REASON_EXCEPTION` with an
//! ESR indicating a system call; we decode that into [`Exit::Syscall`] and
//! return to the kernel. This is where the crate's only `unsafe` FFI lives.
//!
//! Compile-time stub until the FFI bindings land.

use crate::abi::Arch;

use super::{Backend, Vcpu, VcpuError};

#[derive(Debug)]
pub struct HvfBackend {
    _priv: (),
}

impl HvfBackend {
    pub fn new() -> Result<Self, VcpuError> {
        // Phase 1: hv_vm_create() + capability probe here.
        Ok(Self { _priv: () })
    }
}

impl Backend for HvfBackend {
    fn name(&self) -> &'static str {
        "hvf"
    }

    fn guest_arch(&self) -> Arch {
        Arch::Aarch64
    }

    fn new_vcpu(&self, _entry: u64, _stack: u64) -> Result<Box<dyn Vcpu>, VcpuError> {
        Err(VcpuError::Backend(
            "hvf vcpu not implemented yet (ROADMAP Phase 1)".into(),
        ))
    }
}
