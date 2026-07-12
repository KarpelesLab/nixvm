//! The process-global Hypervisor.framework VM.
//!
//! Apple Silicon permits only a tiny number of VMs per system (a 2-VM quota) and
//! `hv_vm_*` on arm64 operate on the calling process's single implicit VM — so
//! nixvm creates exactly one, lazily, shared by every guest process, and never
//! destroys it for the process's lifetime. [`vm`] returns it or the error from
//! `hv_vm_create` (notably `HV_DENIED` when the binary is not codesigned with the
//! `com.apple.security.hypervisor` entitlement), which the backend turns into a
//! graceful fallback to the interpreter.

// `map`/`unmap` are exercised by the bring-up test and the vcpu impl landing in
// the next step; allow them dead until then.
#![allow(dead_code)]

use super::sys::{
    HV_SUCCESS, hv_ipa_t, hv_return_t, hv_vm_create, hv_vm_map, hv_vm_protect, hv_vm_unmap,
};
use crate::vcpu::VcpuError;
use crate::vcpu::region::HOST_PAGE;
use std::sync::OnceLock;

/// Marker for "the process VM exists". arm64 `hv_vm_*` take no VM handle (the VM
/// is implicit per process), so this carries no state — only the guarantee that
/// `hv_vm_create` succeeded.
#[derive(Debug)]
pub struct Vm {
    _private: (),
}

static VM: OnceLock<Result<Vm, String>> = OnceLock::new();

/// The process VM, created on first call. Returns a backend error (never panics)
/// if the hypervisor is unavailable or the process is not entitled.
pub fn vm() -> Result<&'static Vm, VcpuError> {
    let created = VM.get_or_init(|| {
        // SAFETY: `hv_vm_create` is a valid one-time call per process; a null
        // config selects defaults. It returns a status we check rather than
        // producing any invalid state.
        let ret = unsafe { hv_vm_create(std::ptr::null()) };
        if ret == HV_SUCCESS {
            Ok(Vm { _private: () })
        } else {
            Err(format!("hv_vm_create failed (status {ret:#x})"))
        }
    });
    created.as_ref().map_err(|e| VcpuError::Backend(e.clone()))
}

fn check(ret: hv_return_t, what: &str) -> Result<(), VcpuError> {
    if ret == HV_SUCCESS {
        Ok(())
    } else {
        Err(VcpuError::Backend(format!(
            "{what} failed (status {ret:#x})"
        )))
    }
}

// `&self` is a capability token — holding a `&Vm` proves `hv_vm_create`
// succeeded — even though arm64 `hv_vm_*` take no VM handle.
#[allow(clippy::unused_self)]
impl Vm {
    /// Map `size` bytes of host memory at `host` into the guest at IPA `ipa`
    /// with `flags` ([`super::sys::HV_MEMORY_READ`] etc.). All three of `host`, `ipa`,
    /// `size` must be host-page (16 KiB) aligned.
    pub fn map(&self, host: *mut u8, ipa: u64, size: usize, flags: u64) -> Result<(), VcpuError> {
        assert_eq!(
            host as usize % HOST_PAGE,
            0,
            "hv_vm_map host addr alignment"
        );
        assert_eq!(ipa as usize % HOST_PAGE, 0, "hv_vm_map ipa alignment");
        assert_eq!(size % HOST_PAGE, 0, "hv_vm_map size alignment");
        // SAFETY: `host` points at a live `size`-byte allocation (a `Region`
        // owned by the caller for at least as long as the mapping); alignment is
        // asserted above.
        let ret = unsafe { hv_vm_map(host.cast(), ipa as hv_ipa_t, size, flags) };
        check(ret, "hv_vm_map")
    }

    /// Remove the mapping covering `[ipa, ipa+size)`.
    pub fn unmap(&self, ipa: u64, size: usize) -> Result<(), VcpuError> {
        // SAFETY: unmapping a guest IPA range never touches host memory; a range
        // that was not mapped simply returns an error we surface.
        let ret = unsafe { hv_vm_unmap(ipa as hv_ipa_t, size) };
        check(ret, "hv_vm_unmap")
    }

    /// Change the stage-2 permissions of `[ipa, ipa+size)`.
    #[allow(dead_code)] // used by the stage-2 COW milestone
    pub fn protect(&self, ipa: u64, size: usize, flags: u64) -> Result<(), VcpuError> {
        // SAFETY: adjusts stage-2 permissions only; no host memory is accessed.
        let ret = unsafe { hv_vm_protect(ipa as hv_ipa_t, size, flags) };
        check(ret, "hv_vm_protect")
    }
}
