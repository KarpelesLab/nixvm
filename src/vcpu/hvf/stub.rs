//! The process-global EL1 exception stub.
//!
//! The guest runs at EL0; its `svc` traps to EL1, whose vector base is this
//! page. Every one of the 16 vector slots begins with `hvc #0`, so any exception
//! taken to EL1 — in normal operation only the guest `svc` at the 0x400 (sync,
//! lower EL AArch64) entry — bounces straight out to the host as an HVC exit.
//! Mapped once, at a fixed low IPA below the guest region so it never overlaps
//! guest RAM, and kept for the process's lifetime.

use super::sys::{self, HV_MEMORY_EXEC, HV_MEMORY_READ};
use super::vm::Vm;
use crate::vcpu::VcpuError;
use crate::vcpu::region::{HOST_PAGE, Region};
use std::sync::OnceLock;

/// Stub IPA — below `GUEST_BASE` (0x10000) so it never overlaps guest RAM, and
/// 0x800-aligned as `VBAR_EL1` requires.
const STUB_IPA: u64 = 0x4000;

static STUB: OnceLock<Result<u64, String>> = OnceLock::new();

/// Map the stub once (idempotent) and return its IPA, for `VBAR_EL1`.
pub fn ensure_mapped(vm: &Vm) -> Result<u64, VcpuError> {
    STUB.get_or_init(|| {
        let mut region = Region::new(HOST_PAGE);
        let hvc = 0xD400_0002u32.to_le_bytes(); // hvc #0
        for slot in 0..16usize {
            region.write(slot * 0x80, &hvc);
        }
        // SAFETY: `region` is a live HOST_PAGE allocation; make the code we just
        // wrote fetch-coherent before the guest can branch into it.
        unsafe { sys::sys_icache_invalidate(region.as_ptr().cast(), HOST_PAGE) };
        vm.map(
            region.as_ptr(),
            STUB_IPA,
            HOST_PAGE,
            HV_MEMORY_READ | HV_MEMORY_EXEC,
        )
        .map_err(|e| e.to_string())?;
        // Keep the stub mapped for the process's lifetime.
        std::mem::forget(region);
        Ok(STUB_IPA)
    })
    .as_ref()
    .map(|ipa| *ipa)
    .map_err(|e| VcpuError::Backend(e.clone()))
}
