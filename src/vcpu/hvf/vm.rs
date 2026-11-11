//! The process-global Hypervisor.framework VM and its guest-physical space.
//!
//! Apple Silicon permits only a tiny number of VMs per system and `hv_vm_*` on
//! arm64 operate on the calling process's single implicit VM — so nixvm creates
//! exactly one, lazily, shared by every guest, and never destroys it for the
//! process's lifetime. [`vm`] returns it or the error from `hv_vm_create`
//! (notably `HV_DENIED` when the binary is not codesigned with the
//! `com.apple.security.hypervisor` entitlement), which the backend turns into a
//! graceful fallback to the interpreter.
//!
//! The guest-physical (IPA) space holds every guest RAM pool in use: each arm64
//! pool ([`crate::vcpu::phys::PhysMem`]) owns its own window of it, chosen when
//! its page tables went arm64 ([`crate::vcpu::GuestMemory::set_guest_arch`]),
//! and its page-table entries already point into that window. [`ensure_pool`]
//! maps a pool's whole window once, the first time a vcpu runs on it; the pool
//! unmaps it as it goes (the release hook registered here). So several guests —
//! concurrent sandboxes, parallel tests — share the one VM without seeing each
//! other's memory: a guest's page tables only ever reach its own window.

use super::sys::{
    HV_MEMORY_EXEC, HV_MEMORY_READ, HV_MEMORY_WRITE, HV_SUCCESS, hv_ipa_t, hv_return_t,
    hv_vm_config_create, hv_vm_config_get_max_ipa_size, hv_vm_config_set_ipa_size, hv_vm_create,
    hv_vm_map, hv_vm_unmap,
};
use crate::vcpu::region::HOST_PAGE;
use crate::vcpu::{GuestMemory, VcpuError};
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

/// Marker for "the process VM exists". arm64 `hv_vm_*` take no VM handle (the VM
/// is implicit per process), so this carries no state — only the guarantee that
/// `hv_vm_create` succeeded.
#[derive(Debug)]
pub struct Vm {
    _private: (),
}

static VM: OnceLock<Result<Vm, String>> = OnceLock::new();

/// The pools mapped into the IPA space: window base → (host address, length).
static POOLS: Mutex<BTreeMap<u64, (usize, usize)>> = Mutex::new(BTreeMap::new());

/// The process VM, created on first call with the largest IPA space the host
/// offers (room for many pools' windows). Returns a backend error (never panics)
/// if the hypervisor is unavailable or the process is not entitled.
pub fn vm() -> Result<&'static Vm, VcpuError> {
    let created = VM.get_or_init(|| {
        // SAFETY: `hv_vm_config_create` returns a fresh config object (or null);
        // the IPA-size calls only read/write the out-pointer and that object,
        // and `hv_vm_create` takes a config or null for defaults, once per
        // process. Each status is checked rather than producing invalid state.
        let ret = unsafe {
            let config = hv_vm_config_create();
            let mut max_ipa = 0u32;
            if !config.is_null() && hv_vm_config_get_max_ipa_size(&raw mut max_ipa) == HV_SUCCESS {
                hv_vm_config_set_ipa_size(config, max_ipa);
            }
            hv_vm_create(config)
        };
        if ret == HV_SUCCESS {
            crate::vcpu::phys::on_window_release(release_pool);
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

/// Map `mem`'s physical pool into the IPA space at its window, if it isn't
/// already. Returns the window's base.
pub fn ensure_pool(mem: &GuestMemory) -> Result<u64, VcpuError> {
    let base = mem.phys_base();
    if base == 0 {
        return Err(VcpuError::Backend(
            "guest memory has no arm64 page tables (see GuestMemory::set_guest_arch)".into(),
        ));
    }
    let host = mem.phys_ptr();
    let len = mem.phys_len() as usize;
    let mut pools = POOLS.lock().unwrap();
    if let Some(&(at, size)) = pools.get(&base) {
        if (at, size) == (host as usize, len) {
            return Ok(base);
        }
        // A window this pool now owns still mapped for one that's gone (its
        // release raced this): drop it first.
        // SAFETY: unmapping a guest IPA range never touches host memory.
        check(unsafe { hv_vm_unmap(base, size) }, "hv_vm_unmap")?;
        pools.remove(&base);
    }
    assert_eq!(host as usize % HOST_PAGE, 0, "pool host alignment");
    assert_eq!(len % HOST_PAGE, 0, "pool length alignment");
    // SAFETY: `host` is the pool's live, host-page-aligned allocation of `len`
    // bytes; the pool unmaps it (`release_pool`) before that memory goes.
    let ret = unsafe {
        hv_vm_map(
            host.cast(),
            base as hv_ipa_t,
            len,
            HV_MEMORY_READ | HV_MEMORY_WRITE | HV_MEMORY_EXEC,
        )
    };
    check(ret, "hv_vm_map")?;
    pools.insert(base, (host as usize, len));
    Ok(base)
}

/// The pool whose IPA window is `base` is going: unmap it, if it was mapped.
fn release_pool(base: u64, _len: u64) {
    if let Some((_, size)) = POOLS.lock().unwrap().remove(&base) {
        // SAFETY: unmapping a guest IPA range never touches host memory; no vcpu
        // runs on a pool that is being dropped.
        unsafe { hv_vm_unmap(base as hv_ipa_t, size) };
    }
}
