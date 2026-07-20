//! A contiguous, host-page-aligned block of guest RAM.
//!
//! This is the single backing store for one guest address space
//! ([`crate::vcpu::mem::GuestMemory`]). It is allocated 16 KiB-aligned so a
//! hardware backend can hand the raw pointer straight to `hv_vm_map` — Apple
//! Silicon's host page size (and `hv_vm_map`'s required alignment for the host
//! pointer, guest IPA, and length) is 16 KiB. A stricter-than-needed alignment
//! is harmless on every other target.
//!
//! All guest-visible access goes through the safe copy methods below; the
//! `unsafe` required to allocate over-aligned memory and to move bytes in and
//! out of the raw allocation is confined to this module. This is one of the
//! crate's few sanctioned unsafe sites (alongside `vcpu::hvf` and the
//! `fs::passthrough` FFI): allocating page-aligned guest RAM is a primitive a
//! portable VM cannot express in safe std today.
//!
//! Concurrency: a hardware backend maps this memory into a guest and the guest
//! may mutate it directly, concurrently with host-side reads/writes. The kernel
//! serializes both behind the per-address-space `Mutex` (the big-kernel-lock
//! model), so only one actor touches a region at a time — which is what makes
//! the `Send` impl below sound.

use std::alloc::{self, Layout};
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};

/// Host page size on Apple Silicon and the alignment `hv_vm_map` requires of the
/// host pointer, guest IPA, and length. Used as the region's alignment on every
/// target.
pub const HOST_PAGE: usize = 16384;

/// A raw, zero-filled, `HOST_PAGE`-aligned allocation owning `len` bytes.
pub struct Region {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: `Region` owns its allocation uniquely; the raw pointer is never
// aliased by another `Region`. Guest/host access is serialized by the owning
// address space's `Mutex`, so moving a `Region` between threads (as the SMP
// scheduler does with a locked `GuestMemory`) races with nothing.
unsafe impl Send for Region {}

impl Region {
    /// Allocate `len` bytes — rounded up to a whole number of [`HOST_PAGE`]s and
    /// never zero — zero-filled and `HOST_PAGE`-aligned.
    #[must_use]
    pub fn new(len: usize) -> Self {
        let len = len.next_multiple_of(HOST_PAGE).max(HOST_PAGE);
        let layout = Layout::from_size_align(len, HOST_PAGE).expect("region layout");
        // SAFETY: `layout` has non-zero size. `alloc_zeroed` returns either a
        // valid zeroed allocation for `layout` or null, which we route to the
        // allocator's error handler (abort) rather than dereference.
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            alloc::handle_alloc_error(layout);
        }
        Self { ptr, len }
    }

    // Used by the hardware backend to size the `hv_vm_map`.
    #[must_use]
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The raw host pointer, for a hardware backend to `hv_vm_map`. Valid and
    /// `HOST_PAGE`-aligned for the region's lifetime.
    #[must_use]
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Copy `buf` into the region starting at byte offset `off`.
    ///
    /// Takes `&self`: like [`Region::write_u64_atomic`], the store targets the
    /// separate heap allocation `ptr` owns, not the borrowed struct fields, so it
    /// mutates nothing reachable through the shared reference. This is what lets a
    /// shared `Arc<PhysMem>` write to distinct guest frames without a `&mut` — the
    /// physical-RAM model, where concurrent writes to *different* frames race with
    /// nothing (the VMM serializes writes to the *same* frame via the kernel lock).
    ///
    /// # Panics
    /// If `[off, off + buf.len())` is out of range.
    pub fn write(&self, off: usize, buf: &[u8]) {
        assert!(
            off.checked_add(buf.len())
                .is_some_and(|end| end <= self.len),
            "region write out of bounds"
        );
        // SAFETY: the destination range is checked in-bounds above; `buf` is a
        // distinct allocation, so source and destination do not overlap.
        unsafe { ptr::copy_nonoverlapping(buf.as_ptr(), self.ptr.add(off), buf.len()) };
    }

    /// Copy `buf.len()` bytes out of the region starting at byte offset `off`.
    ///
    /// # Panics
    /// If `[off, off + buf.len())` is out of range.
    pub fn read(&self, off: usize, buf: &mut [u8]) {
        assert!(
            off.checked_add(buf.len())
                .is_some_and(|end| end <= self.len),
            "region read out of bounds"
        );
        // SAFETY: the source range is checked in-bounds above; `buf` is a
        // distinct allocation, so source and destination do not overlap.
        unsafe { ptr::copy_nonoverlapping(self.ptr.add(off), buf.as_mut_ptr(), buf.len()) };
    }

    /// Store `val` as a single atomic, aligned 64-bit write at byte offset `off`.
    ///
    /// `off` must be 8-byte aligned; since the region base is [`HOST_PAGE`]-aligned
    /// this holds for any 8-aligned offset. An aligned 64-bit store is atomic on
    /// x86-64, so a concurrent *reader* — notably a hardware page-table walker of a
    /// running sibling vcpu, when this region backs the KVM shadow page tables —
    /// observes either the whole old value or the whole new value, never a torn mix
    /// of bytes. That tear-freedom is what lets the SMP scheduler rewrite page-table
    /// leaves (an `mmap`/`mprotect`) while sibling vcpus execute `KVM_RUN` lockless.
    ///
    /// Takes `&self` deliberately: the store targets the separate heap allocation
    /// `ptr` owns, not the borrowed struct fields, so it mutates nothing reachable
    /// through the shared reference — and the atomicity guards the pointed-to bytes
    /// against the (non-Rust) hardware walker, not against another Rust thread.
    ///
    /// # Panics
    /// If `off` is not 8-aligned or `[off, off + 8)` is out of range.
    pub fn write_u64_atomic(&self, off: usize, val: u64) {
        assert!(
            off.is_multiple_of(8) && off.checked_add(8).is_some_and(|end| end <= self.len),
            "region atomic u64 store out of bounds or misaligned"
        );
        // SAFETY: `off` is 8-aligned and in-bounds and `ptr` is 16 KiB-aligned, so
        // `ptr + off` is a valid, naturally-aligned `*AtomicU64` into this region's
        // allocation (the cast_ptr_alignment lint can't see the runtime alignment
        // the assert enforces). The store is a single aligned mov; see the doc
        // comment for why `&self` is sound here.
        #[allow(clippy::cast_ptr_alignment)]
        unsafe {
            let p = self.ptr.add(off).cast::<AtomicU64>();
            (*p).store(val, Ordering::Release);
        }
    }

    /// Set the `len` bytes at `off` to `val`.
    ///
    /// Takes `&self` for the same reason as [`Region::write`].
    ///
    /// # Panics
    /// If `[off, off + len)` is out of range.
    pub fn fill(&self, off: usize, len: usize, val: u8) {
        assert!(
            off.checked_add(len).is_some_and(|end| end <= self.len),
            "region fill out of bounds"
        );
        // SAFETY: the range is checked in-bounds above.
        unsafe { ptr::write_bytes(self.ptr.add(off), val, len) };
    }

    /// Copy `len` bytes at offset `off` from `src` into `self`.
    ///
    /// Takes `&self` for the same reason as [`Region::write`].
    ///
    /// # Panics
    /// If the range is out of bounds for either region.
    #[allow(dead_code)] // exercised by tests; no lib caller since GuestMemory moved to the pool
    pub fn copy_from(&self, src: &Region, off: usize, len: usize) {
        let end = off.checked_add(len);
        assert!(
            end.is_some_and(|e| e <= self.len && e <= src.len),
            "region copy out of bounds"
        );
        // SAFETY: the range is in-bounds for both allocations; `self` and `src`
        // are distinct regions (the caller forks a fresh one), so no overlap.
        unsafe { ptr::copy_nonoverlapping(src.ptr.add(off), self.ptr.add(off), len) };
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.len, HOST_PAGE).expect("region layout");
        // SAFETY: `ptr` was returned by `alloc_zeroed` with exactly `layout` and
        // is freed exactly once (here, on drop).
        unsafe { alloc::dealloc(self.ptr, layout) };
    }
}

impl std::fmt::Debug for Region {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Region")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_is_aligned_zeroed_and_rounded() {
        let r = Region::new(100);
        assert_eq!(r.len(), HOST_PAGE, "rounded up to a whole host page");
        assert_eq!(r.as_ptr() as usize % HOST_PAGE, 0, "16 KiB-aligned");
        let mut buf = [0xffu8; 8];
        r.read(0, &mut buf);
        assert_eq!(buf, [0u8; 8], "zero-filled");
    }

    #[test]
    fn write_read_roundtrip_and_fill() {
        let r = Region::new(HOST_PAGE);
        r.write(16, &[1, 2, 3, 4]);
        let mut buf = [0u8; 4];
        r.read(16, &mut buf);
        assert_eq!(buf, [1, 2, 3, 4]);
        r.fill(16, 2, 0);
        r.read(16, &mut buf);
        assert_eq!(buf, [0, 0, 3, 4]);
    }

    #[test]
    fn copy_from_duplicates_bytes() {
        let a = Region::new(HOST_PAGE);
        a.write(0, &[9, 8, 7]);
        let b = Region::new(HOST_PAGE);
        b.copy_from(&a, 0, HOST_PAGE);
        let mut buf = [0u8; 3];
        b.read(0, &mut buf);
        assert_eq!(buf, [9, 8, 7]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn write_past_end_panics() {
        let r = Region::new(HOST_PAGE);
        r.write(HOST_PAGE - 2, &[1, 2, 3, 4]);
    }
}
