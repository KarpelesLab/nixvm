//! Hypervisor.framework backend (macOS / arm64) — the primary target.
//!
//! Runs guest code on a real CPU core via Apple's Hypervisor.framework: one
//! process-global VM ([`vm`]) whose single guest-physical space holds a guest's
//! contiguous [`GuestMemory`] region (mapped MMU-off, so guest virtual == IPA),
//! with each guest thread on its own thread-bound `hv_vcpu`. A guest `svc` traps
//! to a tiny EL1 stub that `hvc`s out to the host, which the run loop decodes to
//! [`Exit::Syscall`]; stage-2 aborts become [`Exit::MemFault`]. All `unsafe` FFI
//! lives in [`sys`]; the process VM and its map/protect wrappers in [`vm`].
//!
//! Entitlement: creating the VM needs `com.apple.security.hypervisor`, so an
//! un-codesigned binary (CI, `cargo test`) gets a graceful error from
//! [`HvfBackend::new`] and the caller falls back to the interpreter.

mod sys;
mod vm;

use crate::abi::Arch;

use super::{Backend, Vcpu, VcpuError};

/// Backend handle. Its existence means the process VM was created successfully.
#[derive(Debug)]
pub struct HvfBackend {
    _private: (),
}

impl HvfBackend {
    /// Construct the backend handle. The process VM is created lazily (and its
    /// availability probed, for the interpreter fallback) when `select` wires
    /// this backend in — a later milestone step; for now the vcpu is not built.
    pub fn new() -> Result<Self, VcpuError> {
        Ok(Self { _private: () })
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
            "hvf vcpu not implemented yet (next milestone step)".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::sys::*;
    use super::vm::vm;
    use crate::vcpu::region::{HOST_PAGE, Region};

    /// Smallest possible bring-up: map one page of guest RAM, run `movz x0,#7 ;
    /// hvc #0` at EL1, and confirm the vcpu exits via an HVC exception with
    /// `x0 == 7`. Validates the whole FFI spine — VM create, map, vcpu create,
    /// run, exit decode, register read — before any of it is wired to the kernel.
    ///
    /// Ignored by default: needs a binary codesigned with the
    /// `com.apple.security.hypervisor` entitlement, and `NIXVM_HVF=1` to opt in.
    #[test]
    #[ignore = "requires the hypervisor entitlement + codesign; run with NIXVM_HVF=1"]
    fn bringup_mov_hvc() {
        // One 16 KiB page of guest RAM mapped at IPA 0x10000.
        const IPA: u64 = 0x1_0000;

        if std::env::var_os("NIXVM_HVF").is_none() {
            return;
        }
        let vm = vm().expect("create process VM (is the binary codesigned + entitled?)");

        let mut region = Region::new(HOST_PAGE);
        let program: [u32; 2] = [
            0xD28000E0, // movz x0, #7
            0xD4000002, // hvc  #0
        ];
        let mut bytes = Vec::new();
        for w in program {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        region.write(0, &bytes);
        // SAFETY: the region is a live HOST_PAGE-byte allocation.
        unsafe { sys_icache_invalidate(region.as_ptr().cast(), HOST_PAGE) };
        vm.map(
            region.as_ptr(),
            IPA,
            HOST_PAGE,
            HV_MEMORY_READ | HV_MEMORY_WRITE | HV_MEMORY_EXEC,
        )
        .expect("hv_vm_map");

        // SAFETY: single-threaded test; the vcpu is created and run on this
        // thread, `exit` points at framework-owned storage valid for its life.
        unsafe {
            let mut vcpu: hv_vcpu_t = 0;
            let mut exit: *mut hv_vcpu_exit_t = std::ptr::null_mut();
            assert_eq!(
                hv_vcpu_create(&raw mut vcpu, &raw mut exit, std::ptr::null()),
                HV_SUCCESS,
                "hv_vcpu_create"
            );
            assert_eq!(hv_vcpu_set_reg(vcpu, HV_REG_PC, IPA), HV_SUCCESS);
            // EL1t, DAIF masked — run the two instructions straight; HVC from EL1
            // exits directly to the host, so no EL1 stub is needed here.
            assert_eq!(hv_vcpu_set_reg(vcpu, HV_REG_CPSR, 0x3c4), HV_SUCCESS);

            assert_eq!(hv_vcpu_run(vcpu), HV_SUCCESS, "hv_vcpu_run");
            let ex = &*exit;
            assert_eq!(ex.reason, HV_EXIT_REASON_EXCEPTION, "exited via exception");
            let ec = (ex.exception.syndrome >> 26) & 0x3f;
            assert_eq!(ec, 0x16, "exception class is HVC");

            let mut x0: u64 = 0;
            assert_eq!(hv_vcpu_get_reg(vcpu, HV_REG_X0, &raw mut x0), HV_SUCCESS);
            assert_eq!(x0, 7, "guest set x0 = 7 before the hvc");

            assert_eq!(hv_vcpu_destroy(vcpu), HV_SUCCESS);
        }
    }
}
