//! Physical guest RAM and a 4 KiB frame allocator.
//!
//! Phase 1 of the MMU refactor. Where [`crate::vcpu::mem::GuestMemory`] models
//! one process's *flat, identity-mapped* address space, this module provides the
//! substrate a *real* MMU model needs: a single, always-resident pool of
//! guest-*physical* RAM shared by every address space, plus an allocator that
//! hands out 4 KiB physical frames from it. Later phases build per-process page
//! tables over these frames (each process its own `CR3`) and register the whole
//! pool as one KVM memslot.
//!
//! Two types live here, deliberately kept as peers:
//!
//! * [`PhysMem`] — owns the backing [`Region`] and does host-side byte access at
//!   a *physical address* (`[0, size)`). It knows nothing about allocation; it is
//!   just addressable RAM, so a KVM memslot can be registered over the whole of
//!   it ([`PhysMem::as_ptr`]/[`PhysMem::len`], base 0) independently of who owns
//!   which frame.
//! * [`FrameAllocator`] — the free-frame bookkeeping (a free-list stack + a
//!   per-frame refcount array). [`FrameAllocator::alloc`] takes `&mut PhysMem` so
//!   it can zero the frame it hands back; the two are threaded together by the
//!   caller rather than one owning the other, which keeps `PhysMem` borrowable
//!   for the memslot registration without entangling it in the allocator borrow.
//!
//! Concurrency: allocation and page-table edits happen under the VMM's
//! big-kernel-lock — the kernel services one guest request at a time — so neither
//! type needs any internal locking; every mutating method takes `&mut self`. The
//! backing [`Region`] is still `Send`, so the whole pool moves between scheduler
//! threads under that lock exactly as a `GuestMemory` does today.
//!
//! This module is self-contained and not yet wired into the rest of the crate;
//! its entire public surface is exercised by the tests below. The allocator and
//! pool are used only from those tests until Phase 2/3 wire them in, so the
//! module-wide `allow(dead_code)` keeps a non-test release build warning-free.
#![allow(dead_code)]

use super::mem::PAGE_SIZE;
use super::region::Region;

/// Frame size — the guest page size, 4 KiB. A physical frame is one of these.
const FRAME: u64 = PAGE_SIZE;
/// Frame size as a `usize`, for offset math and metadata indexing.
const FRAME_SZ: usize = FRAME as usize;

/// Refcount sentinel marking a frame as *pinned*: reserved at bootstrap (or the
/// null frame 0) and permanently out of circulation. A pinned frame is never
/// handed out by [`FrameAllocator::alloc`] and never returns to the free list —
/// `incref`/`decref` are no-ops on it, so its count can neither overflow nor
/// drop to a "free" state. `u32::MAX` is safe as a sentinel: a frame shared by
/// four billion address spaces is not a real configuration.
const PINNED: u32 = u32::MAX;

/// A contiguous pool of guest-*physical* RAM, `[0, size)`, backed by one
/// host-page-aligned [`Region`]. Addresses passed to its methods are physical
/// addresses (`pa`), i.e. byte offsets from the pool base (which is always 0), so
/// a later KVM memslot mapping guest-physical `0` to [`PhysMem::as_ptr`] makes
/// every `pa` line up with the same host byte the interpreter sees here.
#[derive(Debug)]
pub struct PhysMem {
    region: Region,
    /// Pool size in bytes = `region.len()` rounded to a whole number of frames.
    size: usize,
}

// SAFETY: `PhysMem` is the shared guest physical RAM, held as an `Arc<PhysMem>`
// by every address space (and the KVM VM). Its only non-`Sync` member is the
// `Region`'s raw `*mut u8`. Sharing it across threads is exactly the physical-RAM
// model: two vcpus on different cores read/write distinct frames concurrently
// (as real RAM allows), and the VMM's big-kernel-lock serializes the host-side
// writes that could target the *same* frame. The byte methods take `&self` and
// touch only the owned heap allocation, never the borrowed struct.
unsafe impl Send for PhysMem {}
// SAFETY: see the `Send` note above.
unsafe impl Sync for PhysMem {}

impl PhysMem {
    /// Allocate a zero-filled pool holding `size` bytes of guest-physical RAM.
    /// `size` is rounded up to a whole number of frames (and, via [`Region`], to
    /// a whole host page). Pool base is 0.
    #[must_use]
    pub fn new(size: usize) -> Self {
        let size = size.next_multiple_of(FRAME_SZ).max(FRAME_SZ);
        let region = Region::new(size);
        // `Region::new` rounds up to a host page (16 KiB), a multiple of the
        // 4 KiB frame, so the whole allocation is frame-addressable.
        Self {
            size: region.len(),
            region,
        }
    }

    /// Physical base of the pool — always 0, the constant a KVM memslot maps
    /// guest-physical address 0 to. Present as a method so call sites read
    /// intent rather than a bare literal.
    #[must_use]
    #[allow(clippy::unused_self)]
    pub fn base(&self) -> u64 {
        0
    }

    /// Pool size in bytes (a whole number of frames).
    #[must_use]
    pub fn len(&self) -> usize {
        self.size
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Number of frames in the pool.
    #[must_use]
    pub fn nframes(&self) -> u64 {
        self.size as u64 / FRAME
    }

    /// Raw host pointer to physical address 0, for a KVM memslot to map the whole
    /// pool. Valid and host-page-aligned for the pool's lifetime.
    #[must_use]
    pub fn as_ptr(&self) -> *mut u8 {
        self.region.as_ptr()
    }

    /// Whether `[pa, pa + len)` lies entirely within the pool.
    fn in_bounds(&self, pa: u64, len: usize) -> bool {
        pa.checked_add(len as u64)
            .is_some_and(|end| end <= self.size as u64)
    }

    /// Copy `buf.len()` bytes out of the pool starting at physical address `pa`.
    ///
    /// # Panics
    /// If `[pa, pa + buf.len())` is out of the pool's bounds.
    pub fn read(&self, pa: u64, buf: &mut [u8]) {
        assert!(self.in_bounds(pa, buf.len()), "phys read out of bounds");
        self.region.read(pa as usize, buf);
    }

    /// Copy `bytes` into the pool starting at physical address `pa`.
    ///
    /// Takes `&self`: the pool is shared as an `Arc<PhysMem>` and written
    /// lock-free (the physical-RAM model — see [`Region::write`]); the VMM's
    /// kernel lock serializes writes that would target the same frame.
    ///
    /// # Panics
    /// If `[pa, pa + bytes.len())` is out of the pool's bounds.
    pub fn write(&self, pa: u64, bytes: &[u8]) {
        assert!(self.in_bounds(pa, bytes.len()), "phys write out of bounds");
        self.region.write(pa as usize, bytes);
    }

    /// Read a little-endian `u64` at physical address `pa` — for reading a
    /// page-table entry once page tables live in the pool.
    ///
    /// # Panics
    /// If `[pa, pa + 8)` is out of bounds.
    #[must_use]
    pub fn read_u64(&self, pa: u64) -> u64 {
        let mut b = [0u8; 8];
        self.read(pa, &mut b);
        u64::from_le_bytes(b)
    }

    /// Store `val` as a single atomic, aligned 64-bit write at physical address
    /// `pa` — the primitive for publishing a page-table entry to a concurrent
    /// hardware walker (see [`Region::write_u64_atomic`]).
    ///
    /// # Panics
    /// If `pa` is not 8-aligned or `[pa, pa + 8)` is out of bounds.
    pub fn write_u64_atomic(&self, pa: u64, val: u64) {
        assert!(self.in_bounds(pa, 8), "phys atomic u64 store out of bounds");
        self.region.write_u64_atomic(pa as usize, val);
    }

    /// Zero the whole frame containing `pa`. `pa` must be frame-aligned.
    ///
    /// # Panics
    /// If `pa` is not frame-aligned or its frame lies outside the pool.
    pub fn zero_frame(&self, pa: u64) {
        assert!(pa.is_multiple_of(FRAME), "zero_frame: pa not frame-aligned");
        assert!(self.in_bounds(pa, FRAME_SZ), "zero_frame out of bounds");
        self.region.fill(pa as usize, FRAME_SZ, 0);
    }

    /// Copy the frame at `src` over the frame at `dst` — the byte move a
    /// copy-on-write fault makes when it privatizes a shared frame. Both must be
    /// frame-aligned. A no-op copy onto itself is allowed.
    ///
    /// # Panics
    /// If either address is not frame-aligned or lies outside the pool.
    pub fn copy_frame(&self, dst: u64, src: u64) {
        assert!(
            dst.is_multiple_of(FRAME) && src.is_multiple_of(FRAME),
            "copy_frame: address not frame-aligned"
        );
        assert!(
            self.in_bounds(dst, FRAME_SZ) && self.in_bounds(src, FRAME_SZ),
            "copy_frame out of bounds"
        );
        if dst == src {
            return;
        }
        // Route through a stack buffer: `Region` exposes no same-allocation
        // slice, and this keeps the module free of new `unsafe`. One frame is
        // 4 KiB — cheap, and CoW copies are rare relative to plain accesses.
        let mut buf = [0u8; FRAME_SZ];
        self.region.read(src as usize, &mut buf);
        self.region.write(dst as usize, &buf);
    }
}

/// A 4 KiB physical-frame allocator over a [`PhysMem`] pool `[0, size)`.
///
/// Structure: a free-list stack of frame numbers ([`FrameAllocator::free_list`],
/// popped in `alloc`, pushed on the last `decref`) for O(1) alloc/free, plus a
/// per-frame refcount array ([`FrameAllocator::refcount`]) indexed by frame
/// number for copy-on-write sharing. The refcount array is the only real
/// overhead: `size / 4096` `u32`s — ~8 MiB for an 8 GiB pool, which is
/// acceptable for a structure consulted on every map/unmap/CoW fault.
///
/// Frame 0 (physical address 0) is permanently reserved as a null-frame guard so
/// that a zero page-table entry is unambiguously "not present" — it is never
/// handed out. Ranges pinned via [`FrameAllocator::reserve`] are likewise held
/// out of circulation for a bootstrap/control-block region a later phase owns.
///
/// Single-threaded by construction: every method takes `&mut self` and there is
/// no internal locking. The VMM's big-kernel-lock serializes all callers, so a
/// frame cannot be alloc'd or freed by two actors at once.
#[derive(Debug)]
pub struct FrameAllocator {
    /// `refcount[f]` for frame number `f`: 0 = free, `1..PINNED` = live and
    /// shared by that many owners, [`PINNED`] = reserved/null and permanent.
    refcount: Vec<u32>,
    /// Frame numbers currently free, as a LIFO stack. A frame is in this list
    /// iff its refcount is 0 and it is not pinned.
    free_list: Vec<u32>,
    /// Count of live (non-pinned, refcount ≥ 1) frames, tracked so `alloc_count`
    /// is O(1) rather than a scan of `refcount`.
    allocated: usize,
}

impl FrameAllocator {
    /// Build an allocator for a pool of `nframes` frames. Frame 0 is pinned
    /// immediately (the null-frame guard); every other frame starts free.
    ///
    /// `nframes` is normally `phys.nframes()`; the two are constructed together
    /// and must describe the same pool.
    #[must_use]
    pub fn new(nframes: u64) -> Self {
        let nframes = nframes as usize;
        assert!(nframes >= 1, "pool must have at least the null frame");
        let mut refcount = vec![0u32; nframes];
        refcount[0] = PINNED; // frame 0 is the null-frame guard, never handed out
        // Push high→low so `alloc` (pop) hands out ascending frame numbers, which
        // makes test expectations and dumps easier to read. Frame 0 is excluded.
        let free_list: Vec<u32> = (1..nframes as u32).rev().collect();
        Self {
            refcount,
            free_list,
            allocated: 0,
        }
    }

    /// Validate `pa`, returning its frame number, or `None` if `pa` is not
    /// frame-aligned or names a frame outside the pool. Callers treat `None` as a
    /// bug in the caller (a bad physical address) and ignore it.
    fn frame_of(&self, pa: u64) -> Option<usize> {
        if !pa.is_multiple_of(FRAME) {
            return None;
        }
        let f = (pa / FRAME) as usize;
        (f < self.refcount.len()).then_some(f)
    }

    /// Allocate a fresh frame: pop a free frame, zero its contents in `phys`, set
    /// its refcount to 1, and return its frame-aligned physical address. Returns
    /// `None` on exhaustion — the caller (an `mmap` growing the working set)
    /// turns that into `ENOMEM`; the allocator never panics on a full pool.
    ///
    /// `phys` must be the pool this allocator was built for; the frame is zeroed
    /// there so a fresh mapping reads as zero (`MAP_ANONYMOUS` semantics). `phys`
    /// is taken by shared reference — the allocator is the sole authority on which
    /// frame is free, so zeroing it races with no other owner.
    pub fn alloc(&mut self, phys: &PhysMem) -> Option<u64> {
        let f = self.free_list.pop()?;
        debug_assert_eq!(self.refcount[f as usize], 0, "free-list frame was not free");
        self.refcount[f as usize] = 1;
        self.allocated += 1;
        let pa = u64::from(f) * FRAME;
        phys.zero_frame(pa);
        Some(pa)
    }

    /// Add a reference to the frame at `pa` — a second address space now shares
    /// it (copy-on-write). No-op on a pinned frame. A bad or free `pa` is a
    /// caller bug: debug-asserted, ignored in release.
    pub fn incref(&mut self, pa: u64) {
        let Some(f) = self.frame_of(pa) else {
            debug_assert!(false, "incref: bad physical address {pa:#x}");
            return;
        };
        let rc = self.refcount[f];
        if rc == PINNED {
            return; // pinned frames have no meaningful refcount
        }
        debug_assert!(rc >= 1, "incref of a free frame {pa:#x}");
        // Saturate rather than wrap into PINNED; hitting u32::MAX real owners is
        // not a reachable state, but wrapping to 0 would silently free a live
        // frame, so guard it.
        self.refcount[f] = rc.saturating_add(1);
    }

    /// Drop a reference to the frame at `pa`; when the last reference goes the
    /// frame returns to the free list. No-op on a pinned frame. A bad, free, or
    /// double-freed `pa` is a caller bug: debug-asserted, ignored in release so a
    /// mis-accounted refcount never corrupts the free list.
    pub fn decref(&mut self, pa: u64) {
        let Some(f) = self.frame_of(pa) else {
            debug_assert!(false, "decref: bad physical address {pa:#x}");
            return;
        };
        let rc = self.refcount[f];
        if rc == PINNED {
            return; // reserved/null frames never free
        }
        debug_assert!(rc >= 1, "decref/free of an already-free frame {pa:#x}");
        if rc == 0 {
            return; // release-mode guard against the double-free above
        }
        self.refcount[f] = rc - 1;
        if rc == 1 {
            self.allocated -= 1;
            self.free_list.push(f as u32);
        }
    }

    /// Free a frame — the same single refcount path as [`FrameAllocator::decref`]
    /// (a frame shared by N owners is only reclaimed at the Nth free). Named
    /// `free` for call sites that own exactly one reference.
    pub fn free(&mut self, pa: u64) {
        self.decref(pa);
    }

    /// Current refcount of the frame at `pa` — 0 free, `n` live with `n` owners,
    /// [`PINNED`] (`u32::MAX`) reserved/null. Returns 0 for a bad/misaligned `pa`.
    /// For tests and diagnostics.
    #[must_use]
    pub fn refcount(&self, pa: u64) -> u32 {
        self.frame_of(pa).map_or(0, |f| self.refcount[f])
    }

    /// Permanently reserve the frames overlapping `[pa, pa + len)`, holding them
    /// out of circulation for a bootstrap/control-block region a later phase
    /// pins. Reserved frames are marked [`PINNED`]: never handed out by `alloc`,
    /// never refcounted, never freed. `pa` is rounded down and the end rounded up
    /// to frame boundaries. Intended for bootstrap, before any `alloc`; reserving
    /// a currently-free frame removes it from the free list (O(n), rare).
    ///
    /// # Panics
    /// If the range extends past the end of the pool.
    pub fn reserve(&mut self, pa: u64, len: u64) {
        if len == 0 {
            return;
        }
        let first = (pa / FRAME) as usize;
        let end = pa
            .checked_add(len)
            .expect("reserve range overflows")
            .div_ceil(FRAME) as usize;
        assert!(end <= self.refcount.len(), "reserve past end of pool");
        for f in first..end {
            if self.refcount[f] != PINNED && self.refcount[f] >= 1 {
                // Was a live allocation; drop it from the live count as it becomes
                // pinned. (Bootstrap normally reserves free frames, but stay
                // consistent if a caller pins something it allocated.)
                self.allocated -= 1;
            }
            self.refcount[f] = PINNED;
        }
        // A pinned frame must not sit in the free list.
        self.free_list.retain(|&f| self.refcount[f as usize] != PINNED);
    }

    /// Number of live (allocated, non-pinned) frames.
    #[must_use]
    pub fn alloc_count(&self) -> usize {
        self.allocated
    }

    /// Number of frames available to hand out.
    #[must_use]
    pub fn free_count(&self) -> usize {
        self.free_list.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small pool + its allocator, sized in frames so exhaustion is cheap.
    fn pool(nframes: u64) -> (PhysMem, FrameAllocator) {
        let phys = PhysMem::new(nframes as usize * FRAME_SZ);
        let alloc = FrameAllocator::new(phys.nframes());
        (phys, alloc)
    }

    #[test]
    fn pool_geometry_is_frame_rounded_and_based_at_zero() {
        let phys = PhysMem::new(FRAME_SZ + 1);
        assert_eq!(phys.base(), 0);
        assert_eq!(phys.len() % FRAME_SZ, 0, "whole number of frames");
        assert!(phys.len() >= 2 * FRAME_SZ, "rounded up past one frame");
        assert!(!phys.is_empty());
        assert!(!phys.as_ptr().is_null());
        assert_eq!(phys.nframes(), phys.len() as u64 / FRAME);
    }

    #[test]
    fn read_write_roundtrips_and_is_bounds_checked() {
        let phys = PhysMem::new(4 * FRAME_SZ);
        phys.write(0x40, &[1, 2, 3, 4]);
        let mut buf = [0u8; 4];
        phys.read(0x40, &mut buf);
        assert_eq!(buf, [1, 2, 3, 4]);
        // Untouched bytes read as zero.
        phys.read(0, &mut buf);
        assert_eq!(buf, [0, 0, 0, 0]);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn read_past_end_panics() {
        let phys = PhysMem::new(FRAME_SZ);
        let mut buf = [0u8; 4];
        // A read straddling the very end of the (rounded) pool is rejected.
        phys.read(phys.len() as u64 - 2, &mut buf);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn write_past_end_panics() {
        let phys = PhysMem::new(FRAME_SZ);
        phys.write(phys.len() as u64 - 2, &[1, 2, 3, 4]);
    }

    #[test]
    fn u64_atomic_store_and_read_roundtrip() {
        let phys = PhysMem::new(2 * FRAME_SZ);
        phys.write_u64_atomic(0x100, 0xdead_beef_cafe_babe);
        assert_eq!(phys.read_u64(0x100), 0xdead_beef_cafe_babe);
        // read_u64 also sees a plain byte write.
        phys.write(0x200, &7u64.to_le_bytes());
        assert_eq!(phys.read_u64(0x200), 7);
    }

    #[test]
    fn zero_and_copy_frame() {
        let phys = PhysMem::new(4 * FRAME_SZ);
        phys.write(FRAME, &[0xAB; 16]); // dirty frame 1
        phys.zero_frame(FRAME);
        let mut buf = [0xFFu8; 16];
        phys.read(FRAME, &mut buf);
        assert_eq!(buf, [0u8; 16], "zeroed");
        // Copy frame 2 (written) onto frame 3.
        phys.write(2 * FRAME, &[0x5A; 32]);
        phys.copy_frame(3 * FRAME, 2 * FRAME);
        let mut buf = [0u8; 32];
        phys.read(3 * FRAME, &mut buf);
        assert_eq!(buf, [0x5A; 32]);
    }

    #[test]
    fn alloc_returns_distinct_aligned_zeroed_frames() {
        let (mut phys, mut alloc) = pool(8);
        // Pre-dirty the whole pool so alloc must actively zero.
        for f in 0..8u64 {
            phys.write(f * FRAME, &[0xCC; 8]);
        }
        let a = alloc.alloc(&mut phys).unwrap();
        let b = alloc.alloc(&mut phys).unwrap();
        assert_ne!(a, b, "distinct frames");
        assert_ne!(a, 0, "frame 0 never handed out");
        assert_ne!(b, 0);
        assert!(a.is_multiple_of(FRAME) && b.is_multiple_of(FRAME), "frame-aligned");
        assert_eq!(phys.read_u64(a), 0, "handed-out frame is zeroed");
        assert_eq!(phys.read_u64(b), 0);
        assert_eq!(alloc.refcount(a), 1);
        assert_eq!(alloc.alloc_count(), 2);
    }

    #[test]
    fn free_returns_frame_to_the_pool_and_it_is_reused() {
        let (mut phys, mut alloc) = pool(4);
        let a = alloc.alloc(&mut phys).unwrap();
        let free_before = alloc.free_count();
        alloc.free(a);
        assert_eq!(alloc.refcount(a), 0, "freed");
        assert_eq!(alloc.free_count(), free_before + 1);
        assert_eq!(alloc.alloc_count(), 0);
        // The very next alloc reuses the just-freed frame (LIFO).
        let b = alloc.alloc(&mut phys).unwrap();
        assert_eq!(a, b, "freed frame is reused");
    }

    #[test]
    fn incref_decref_frees_only_at_zero() {
        let (mut phys, mut alloc) = pool(4);
        let a = alloc.alloc(&mut phys).unwrap(); // rc 1
        alloc.incref(a); // rc 2 — shared by two address spaces
        alloc.incref(a); // rc 3
        assert_eq!(alloc.refcount(a), 3);
        let free_now = alloc.free_count();
        alloc.decref(a); // rc 2
        alloc.decref(a); // rc 1
        assert_eq!(alloc.refcount(a), 1, "still live while shared");
        assert_eq!(alloc.free_count(), free_now, "not returned yet");
        assert_eq!(alloc.alloc_count(), 1);
        alloc.decref(a); // rc 0 -> freed
        assert_eq!(alloc.refcount(a), 0);
        assert_eq!(alloc.free_count(), free_now + 1);
        assert_eq!(alloc.alloc_count(), 0);
    }

    #[test]
    fn frame_zero_is_never_allocated_and_pinned() {
        let (mut phys, mut alloc) = pool(3);
        let allocatable = phys.nframes() as usize - 1; // all but the null frame
        assert_eq!(alloc.refcount(0), PINNED, "frame 0 pinned");
        // Drain the pool; frame 0 must never appear.
        let mut seen = Vec::new();
        while let Some(pa) = alloc.alloc(&mut phys) {
            assert_ne!(pa, 0);
            seen.push(pa);
        }
        assert_eq!(seen.len(), allocatable, "every frame but the null frame");
        // decref on the pinned null frame is a no-op, never adds it to the pool.
        alloc.decref(0);
        assert_eq!(alloc.refcount(0), PINNED);
        assert_eq!(alloc.free_count(), 0);
    }

    #[test]
    fn exhaustion_returns_none_without_panic() {
        let (mut phys, mut alloc) = pool(3);
        // Drain every allocatable frame, then confirm a full pool yields None.
        while alloc.alloc(&mut phys).is_some() {}
        assert_eq!(alloc.free_count(), 0);
        assert!(alloc.alloc(&mut phys).is_none(), "exhausted -> None");
        assert!(alloc.alloc(&mut phys).is_none(), "still None, no panic");
    }

    #[test]
    fn reserve_holds_a_range_out_of_circulation() {
        let (mut phys, mut alloc) = pool(8);
        // Reserve frames covering [FRAME, 3*FRAME) => frames 1 and 2.
        alloc.reserve(FRAME, 2 * FRAME);
        assert_eq!(alloc.refcount(FRAME), PINNED);
        assert_eq!(alloc.refcount(2 * FRAME), PINNED);
        // Drain and confirm reserved frames never come back.
        let mut seen = Vec::new();
        while let Some(pa) = alloc.alloc(&mut phys) {
            seen.push(pa);
        }
        assert!(!seen.contains(&FRAME), "reserved frame not handed out");
        assert!(!seen.contains(&(2 * FRAME)));
        // Pool has 8 frames: minus null(0) minus reserved(1,2) => 5 allocatable.
        assert_eq!(seen.len(), 5);
        // Pinned frames ignore incref/decref.
        alloc.decref(FRAME);
        assert_eq!(alloc.refcount(FRAME), PINNED);
    }

    #[test]
    fn reserve_rounds_out_to_frame_boundaries() {
        let (_phys, mut alloc) = pool(8);
        // A sub-frame range starting mid-frame 1 still pins whole frame 1.
        alloc.reserve(FRAME + 10, 4);
        assert_eq!(alloc.refcount(FRAME), PINNED);
        assert_eq!(alloc.refcount(2 * FRAME), 0, "next frame untouched");
    }

    #[test]
    fn refcount_of_bad_address_is_zero() {
        let (_phys, alloc) = pool(2);
        assert_eq!(alloc.refcount(1), 0, "misaligned pa");
        assert_eq!(alloc.refcount(100 * FRAME), 0, "out-of-pool frame");
    }
}
