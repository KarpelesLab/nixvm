//! Guest memory: one process's virtual address space over the shared pool.
//!
//! Phase 3 of the MMU refactor replaced the flat, identity-mapped backing with a
//! *real* MMU model. Guest RAM is one shared pool of physical frames
//! ([`super::phys::PhysMem`]) registered as a single KVM memslot; each process
//! owns a 4-level x86-64 page-table tree ([`super::pagetable::AddrSpace`], its own
//! `CR3`) over that pool. A [`GuestMemory`] bundles the two together plus the
//! per-page bookkeeping the kernel and loader need.
//!
//! The public surface is unchanged from the flat model — `read`/`write`/`map`/
//! `protect`/`unmap`/`fork`/`write_init`/… — so the kernel, loader, and
//! interpreter call it exactly as before; only the internals now translate every
//! access through the page tables and copy to/from the pool. Protection is
//! enforced by the page tables (the same tables the KVM hardware walker uses), so
//! a write to a read-only page, a fetch from an NX page, or a store to a
//! copy-on-write page all fault identically under the interpreter and under KVM.
//!
//! Copy-on-write: [`GuestMemory::fork`] shares every mapped frame read-only with
//! the child (via [`AddrSpace::fork_cow`]); the first write on either side
//! privatizes the frame ([`GuestMemory::cow_fault`], or transparently inside
//! `write`/`write_init`). Only touched pages ever consume a private frame.

use super::ctrl;
use super::pagetable::AddrSpace;
use super::phys::{FrameAllocator, PhysMem};
use super::region::HOST_PAGE;
use std::sync::{Arc, Mutex};

/// Guest page size advertised to the guest (`AT_PAGESZ`, `sysconf(_SC_PAGESIZE)`).
pub const PAGE_SIZE: u64 = 4096;

/// Page size as a `usize`, for indexing metadata.
const PS: usize = PAGE_SIZE as usize;

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
    pub const fn rwx() -> Prot {
        Prot(Self::READ.0 | Self::WRITE.0 | Self::EXEC.0)
    }
    #[must_use]
    pub const fn contains(self, other: Prot) -> bool {
        self.0 & other.0 == other.0
    }
    /// This protection with `WRITE` removed — the leaf a copy-on-write-shared
    /// frame carries so the next store faults.
    #[must_use]
    const fn read_only(self) -> Prot {
        Prot(self.0 & !Self::WRITE.0)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum MemError {
    /// Address (range) lies outside `[base, base + size)`.
    OutOfBounds(u64),
    /// Address is within bounds but the page is not mapped.
    Unmapped(u64),
    /// The page is mapped but the access violates its protection.
    Protection { addr: u64, needed: Prot },
    /// Host-side allocation/mapping failure (e.g. the frame pool is exhausted).
    Host(String),
}

impl MemError {
    /// The guest address the fault is attributed to. For page-granular faults
    /// (`Unmapped`, `Protection`) this is the base of the *offending* page.
    #[must_use]
    pub fn fault_addr(&self) -> u64 {
        match self {
            MemError::OutOfBounds(a) | MemError::Unmapped(a) => *a,
            MemError::Protection { addr, .. } => *addr,
            MemError::Host(_) => 0,
        }
    }
}

/// The guest address space for one process. Not `Clone` — use [`GuestMemory::fork`].
pub struct GuestMemory {
    base: u64,
    size: u64,
    /// The shared physical-RAM pool (one KVM memslot). Cloned across `fork`/
    /// `exec` so every address space and the memslot see the same frames.
    phys: Arc<PhysMem>,
    /// The shared frame allocator, locked briefly for map/unmap/fork/CoW.
    fa: Arc<Mutex<FrameAllocator>>,
    /// This process's page tables (its `CR3`).
    space: AddrSpace,
    /// Per-page *intended* protection (what `map`/`protect` requested), meaningful
    /// only when `mapped[i]`. The page-table leaf may be more restrictive than
    /// this — a copy-on-write-shared writable page reads `Prot::rw()` here but is
    /// installed read-only until privatized — so this is the authority on "was
    /// this page meant to be writable?" that [`GuestMemory::cow_fault`] needs.
    prot: Vec<Prot>,
    mapped: Vec<bool>,
    /// Per-page: loaded from a file (an ELF segment). `MADV_DONTNEED` preserves
    /// these; cleared when a page is re-`map`ped anonymously.
    file_backed: Vec<bool>,
    /// Set whenever a *present* page-table leaf is cleared or changed from the
    /// host (unmap, protect, copy-on-write privatize). A KVM vcpu running this
    /// address space must flush its TLB before its next run, or a stale entry
    /// would keep serving the old (now-unmapped or reused) frame. Consumed by
    /// [`GuestMemory::take_tlb_dirty`]; irrelevant to the interpreter (no TLB).
    tlb_dirty: bool,
    /// Physical address of this address space's private kernel-stack frame (the
    /// page mapped at [`ctrl::KSTACK_PAGE_VA`]). The CPU pushes the `#PF`
    /// exception frame here at CPL0; a KVM vcpu reads it back from this frame
    /// (not a shared control page) so concurrent faults on sibling vcpus never
    /// clobber one another. See [`GuestMemory::kstack_pa`].
    kstack_pa: u64,
}

impl std::fmt::Debug for GuestMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuestMemory")
            .field("base", &self.base)
            .field("size", &self.size)
            .field("cr3", &self.space.cr3())
            .finish_non_exhaustive()
    }
}

const fn round_up(v: u64, align: u64) -> u64 {
    v.div_ceil(align) * align
}

/// Allocate a fresh private kernel-stack frame and map it into `space` at
/// [`ctrl::KSTACK_PAGE_VA`], releasing any frame the mapping replaced (a
/// CoW-shared kstack inherited from `fork_cow`). Returns the new frame's physical
/// address. Every address space gets its own so concurrent `#PF`s under SMP push
/// their exception frames onto distinct pages.
fn install_kstack(space: &mut AddrSpace, fa: &mut FrameAllocator, phys: &PhysMem) -> u64 {
    let frame = fa.alloc(phys).expect("kstack: frame pool exhausted");
    if let Some(old) = ctrl::map_kstack(space, fa, phys, frame) {
        fa.free(old);
    }
    frame
}

impl GuestMemory {
    /// Reserve a flat guest region `[base, base + size)` backed by a fresh shared
    /// pool. `base` must be page-aligned; `size` is rounded up to a whole host
    /// page. The pool is sized to hold the region's data pages plus the page
    /// tables and the shared control block, with slack — so `fork`ing and small
    /// churn never exhaust it. Mapped-but-unwritten pages read as zero.
    ///
    /// This is the *only* constructor that mints a pool; every other address
    /// space in a run comes from [`GuestMemory::fork`] (which shares this pool) or
    /// [`GuestMemory::exec_reset`] (which rebuilds within it).
    #[must_use]
    pub fn new(base: u64, size: u64) -> Self {
        assert_eq!(base % PAGE_SIZE, 0, "base must be page-aligned");
        let size = size.max(PAGE_SIZE).next_multiple_of(HOST_PAGE as u64);
        let npages = (size / PAGE_SIZE) as usize;

        // Pool budget: one frame per guest page, plus a generous interior-table
        // allowance (a PT per 512 pages, higher levels above that — /64 is ~8×
        // the worst case for a single tree and comfortably covers a fork's second
        // tree), the control block, the null frame, and a fixed slack.
        let data = npages as u64;
        let tables = data / 64 + 64;
        let pool_frames = data + tables + ctrl::CTRL_FRAMES + 128;
        let phys = Arc::new(PhysMem::new((pool_frames * PAGE_SIZE) as usize));

        let mut fa = FrameAllocator::new(phys.nframes());
        ctrl::reserve_and_build(&mut fa, &phys);
        let mut space = AddrSpace::new(&mut fa, &phys).expect("pool too small for a PML4");
        ctrl::map_into(&mut space, &mut fa, &phys);
        let kstack_pa = install_kstack(&mut space, &mut fa, &phys);

        Self {
            base,
            size,
            phys,
            fa: Arc::new(Mutex::new(fa)),
            space,
            prot: vec![Prot::NONE; npages],
            mapped: vec![false; npages],
            file_backed: vec![false; npages],
            tlb_dirty: false,
            kstack_pa,
        }
    }

    // ---- KVM/backend hooks (crate-internal) ------------------------------

    /// The `CR3` value (PML4 physical address) for this address space.
    #[must_use]
    pub(crate) fn cr3(&self) -> u64 {
        self.space.cr3()
    }

    /// Host pointer to physical address 0 of the shared pool — the single KVM
    /// memslot's `userspace_addr`.
    #[must_use]
    pub(crate) fn phys_ptr(&self) -> *mut u8 {
        self.phys.as_ptr()
    }

    /// Size of the shared pool in bytes — the memslot's `memory_size`.
    #[must_use]
    pub(crate) fn phys_len(&self) -> u64 {
        self.phys.len() as u64
    }

    /// A clone of the shared pool handle, so the KVM VM can read the control block
    /// out of it on the fault path without a `GuestMemory` borrow.
    #[must_use]
    pub(crate) fn phys_arc(&self) -> Arc<PhysMem> {
        Arc::clone(&self.phys)
    }

    /// Physical address of this address space's private kernel-stack frame, so a
    /// KVM vcpu can read the pushed `#PF` exception frame lock-free (no
    /// `GuestMemory` borrow) on the SMP path. Changes on `fork`/`exec_reset`.
    #[must_use]
    pub(crate) fn kstack_pa(&self) -> u64 {
        self.kstack_pa
    }

    /// Take and clear the "page tables changed" flag: whether a present leaf was
    /// unmapped, re-protected, or copy-on-write-privatized since the last check.
    /// The scheduler flushes the running KVM vcpu's TLB when this is set, so a
    /// stale entry never serves an unmapped or replaced frame.
    pub(crate) fn take_tlb_dirty(&mut self) -> bool {
        std::mem::take(&mut self.tlb_dirty)
    }

    /// Number of live (allocated) frames in the shared pool — for leak checks.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn frames_in_use(&self) -> usize {
        self.fa.lock().unwrap().alloc_count()
    }

    // ---- geometry --------------------------------------------------------

    #[must_use]
    pub fn base(&self) -> u64 {
        self.base
    }
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Base guest address of page index `p`.
    fn page_base(&self, p: usize) -> u64 {
        self.base + (p as u64) * PAGE_SIZE
    }

    /// Page index for an in-bounds address.
    fn page_index(&self, addr: u64) -> Option<usize> {
        if addr < self.base || addr >= self.base + self.size {
            return None;
        }
        Some(((addr - self.base) / PAGE_SIZE) as usize)
    }

    /// Bounds-check `addr` (`OutOfBounds` otherwise).
    fn offset_ok(&self, addr: u64) -> Result<(), MemError> {
        if addr < self.base || addr >= self.base + self.size {
            return Err(MemError::OutOfBounds(addr));
        }
        Ok(())
    }

    /// Page indices `[first, last]` covering `[addr, addr + len)`.
    fn page_range(&self, addr: u64, len: usize) -> Result<(usize, usize), MemError> {
        if len == 0 {
            self.offset_ok(addr)?;
            let p = ((addr - self.base) / PAGE_SIZE) as usize;
            return Ok((p, p));
        }
        let end = addr
            .checked_add(len as u64 - 1)
            .ok_or(MemError::OutOfBounds(addr))?;
        self.offset_ok(addr)?;
        self.offset_ok(end)?;
        let first = ((addr - self.base) / PAGE_SIZE) as usize;
        let last = ((end - self.base) / PAGE_SIZE) as usize;
        Ok((first, last))
    }

    // ---- mapping ---------------------------------------------------------

    /// The effective leaf protection for page `p` pointing at `frame`: the
    /// intended protection, but read-only if the frame is copy-on-write-shared
    /// (so the next store faults and privatizes).
    fn leaf_prot(&self, p: usize, frame: u64, fa: &FrameAllocator) -> Prot {
        if self.prot[p].contains(Prot::WRITE) && fa.refcount(frame) > 1 {
            self.prot[p].read_only()
        } else {
            self.prot[p]
        }
    }

    /// Map (or remap) the pages covering `[addr, addr + len)` with `prot`.
    ///
    /// Demand-paged: this records the mapping (and drops any old backing so a
    /// remap reads as zero) but allocates **no** frames — a page's frame is minted
    /// on first touch (a write, or a fault). This is what lets a runtime reserve
    /// huge regions cheaply (JSC/Bun reserve multi-hundred-MiB — and `MAP_NORESERVE`
    /// gigabyte — arenas up front and commit them sparsely); eager allocation would
    /// exhaust the frame pool on the first such reservation.
    pub fn map(&mut self, addr: u64, len: u64, prot: Prot) -> Result<(), MemError> {
        if len == 0 {
            return Ok(());
        }
        let start = addr - addr % PAGE_SIZE;
        let end = round_up(addr + len, PAGE_SIZE);
        let (first, last) = self.page_range(start, (end - start) as usize)?;
        let mut fa = self.fa.lock().unwrap();
        for p in first..=last {
            // Fresh mapping: drop any existing backing so the page reads as zero.
            if self.mapped[p] {
                let va = self.base + (p as u64) * PAGE_SIZE;
                if let Some(frame) = self.space.unmap(va, &mut fa, &self.phys) {
                    fa.free(frame);
                }
            }
            self.mapped[p] = true;
            self.prot[p] = prot;
            self.file_backed[p] = false;
        }
        Ok(())
    }

    /// Ensure the page at `va` has a backing frame installed (demand paging): a
    /// no-op if already backed, otherwise a fresh zeroed frame with `prot`. Returns
    /// `false` only on pool exhaustion. Associated fn over the disjoint
    /// `space`/`phys` fields so the caller can hold the `fa` guard at once.
    fn ensure_backed(
        space: &mut AddrSpace,
        phys: &PhysMem,
        prot: Prot,
        fa: &mut FrameAllocator,
        va: u64,
    ) -> bool {
        if space.translate(va, phys).is_some() {
            return true;
        }
        let Some(frame) = fa.alloc(phys) else {
            return false;
        };
        match space.map(va, frame, prot, false, fa, phys) {
            Ok(_) => true,
            Err(_) => {
                fa.free(frame);
                false
            }
        }
    }

    /// Back a demand-paged page on a fault (the KVM `#PF` / not-present path).
    /// Returns `true` if `addr` was a mapped-but-unbacked page and is now backed,
    /// so the faulting access retries and succeeds; `false` if the fault is
    /// something else (unmapped, or an already-backed page — e.g. a copy-on-write
    /// or protection fault the caller must handle).
    pub fn demand_fault(&mut self, addr: u64) -> bool {
        let page = addr - addr % PAGE_SIZE;
        let Some(p) = self.page_index(page) else {
            return false;
        };
        if !self.mapped[p] || self.space.translate(page, &self.phys).is_some() {
            return false;
        }
        let mut fa = self.fa.lock().unwrap();
        Self::ensure_backed(&mut self.space, &self.phys, self.prot[p], &mut fa, page)
    }

    /// Mark `[addr, addr + len)` as file-backed (an ELF segment): `MADV_DONTNEED`
    /// preserves rather than zeroes these pages. Ignores pages outside the region.
    pub fn mark_file_backed(&mut self, addr: u64, len: u64) {
        if len == 0 || addr < self.base {
            return;
        }
        let start = ((addr - self.base) / PAGE_SIZE) as usize;
        let end = ((round_up(addr + len, PAGE_SIZE) - self.base) / PAGE_SIZE) as usize;
        for p in start..end.min(self.file_backed.len()) {
            self.file_backed[p] = true;
        }
    }

    /// Whether the page containing `addr` is file-backed.
    #[must_use]
    pub fn is_file_backed(&self, addr: u64) -> bool {
        if addr < self.base {
            return false;
        }
        let p = ((addr - self.base) / PAGE_SIZE) as usize;
        self.file_backed.get(p).copied().unwrap_or(false)
    }

    /// The effective (intended) protection of the page containing `addr`, or
    /// `None` if unmapped.
    #[must_use]
    pub fn page_prot(&self, addr: u64) -> Option<Prot> {
        let p = self.page_index(addr)?;
        self.mapped[p].then(|| self.prot[p])
    }

    /// Change protection on already-mapped pages covering `[addr, addr + len)`.
    pub fn protect(&mut self, addr: u64, len: u64, prot: Prot) -> Result<(), MemError> {
        if len == 0 {
            return Ok(());
        }
        let start = addr - addr % PAGE_SIZE;
        let end = round_up(addr + len, PAGE_SIZE);
        let (first, last) = self.page_range(start, (end - start) as usize)?;
        let fa = self.fa.lock().unwrap();
        for p in first..=last {
            if !self.mapped[p] {
                return Err(MemError::Unmapped(self.page_base(p)));
            }
            self.prot[p] = prot;
            let va = self.page_base(p);
            if let Some(t) = self.space.translate(va, &self.phys) {
                let frame = t.paddr & !(PAGE_SIZE - 1);
                let lp = self.leaf_prot(p, frame, &fa);
                if self.space.protect(va, lp, false, &self.phys) {
                    self.tlb_dirty = true; // a present leaf's permissions changed
                }
            }
        }
        Ok(())
    }

    /// Unmap the pages covering `[addr, addr + len)`, returning their frames to
    /// the pool.
    pub fn unmap(&mut self, addr: u64, len: u64) -> Result<(), MemError> {
        if len == 0 {
            return Ok(());
        }
        let start = addr - addr % PAGE_SIZE;
        let end = round_up(addr + len, PAGE_SIZE);
        let (first, last) = self.page_range(start, (end - start) as usize)?;
        let mut fa = self.fa.lock().unwrap();
        for p in first..=last {
            if self.mapped[p] {
                if let Some(frame) = self.space.unmap(self.page_base(p), &mut fa, &self.phys) {
                    fa.free(frame);
                    self.tlb_dirty = true; // a present leaf was cleared
                }
            }
            self.mapped[p] = false;
            self.prot[p] = Prot::NONE;
            self.file_backed[p] = false;
        }
        Ok(())
    }

    // ---- access ----------------------------------------------------------

    /// Verify `[addr, addr + len)` is mapped and every page grants `need`
    /// (intended protection).
    fn check(&self, addr: u64, len: usize, need: Prot) -> Result<(), MemError> {
        let (first, last) = self.page_range(addr, len)?;
        for p in first..=last {
            if !self.mapped[p] {
                return Err(MemError::Unmapped(self.page_base(p)));
            }
            if !self.prot[p].contains(need) {
                return Err(MemError::Protection {
                    addr: self.page_base(p),
                    needed: need,
                });
            }
        }
        Ok(())
    }

    /// Whether the page at `addr` is mapped and executable.
    #[must_use]
    pub fn can_exec(&self, addr: u64) -> bool {
        self.check(addr, 1, Prot::EXEC).is_ok()
    }

    /// Copy from the pool at guest `addr` into `buf`, translating per page. The
    /// caller must have verified access. A mapped-but-unbacked page (demand-paged,
    /// never touched) reads as zero without allocating a frame.
    fn copy_out(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemError> {
        let mut done = 0usize;
        while done < buf.len() {
            let cur = addr + done as u64;
            let page = cur - cur % PAGE_SIZE;
            let off = (cur - page) as usize;
            let n = (buf.len() - done).min(PS - off);
            match self.space.translate(cur, &self.phys) {
                Some(t) => self.phys.read(t.paddr, &mut buf[done..done + n]),
                None => buf[done..done + n].fill(0),
            }
            done += n;
        }
        Ok(())
    }

    /// Read `buf.len()` bytes from guest `addr` (requires `READ`).
    pub fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemError> {
        self.check(addr, buf.len(), Prot::READ)?;
        self.copy_out(addr, buf)
    }

    /// Read `len` bytes into a fresh `Vec` (requires `READ`).
    pub fn read_vec(&self, addr: u64, len: usize) -> Result<Vec<u8>, MemError> {
        let mut v = vec![0u8; len];
        self.read(addr, &mut v)?;
        Ok(v)
    }

    /// Ensure the page at `va` (intended protection `prot`) is backed by a private
    /// (unshared) frame its protection can write, privatizing a copy-on-write
    /// frame if needed. A no-op on an already-private page beyond restoring its
    /// leaf write bit. An associated fn over the disjoint `space`/`phys` fields so
    /// callers can hold the `fa` guard (which borrows the `fa` field) at once.
    /// Returns whether it changed a present leaf (so the caller flags the TLB).
    fn make_writable(space: &mut AddrSpace, phys: &PhysMem, prot: Prot, fa: &mut FrameAllocator, va: u64) -> bool {
        let Some(t) = space.translate(va, phys) else {
            return false;
        };
        let frame = t.paddr & !(PAGE_SIZE - 1);
        if fa.refcount(frame) > 1 {
            // Copy-on-write shared: privatize into a fresh frame.
            if let Some(new) = fa.alloc(phys) {
                phys.copy_frame(new, frame);
                if let Ok(old) = space.map(va, new, prot, false, fa, phys) {
                    if let Some(of) = old {
                        fa.decref(of);
                    }
                } else {
                    fa.free(new);
                }
            }
            true
        } else {
            // Private already: make sure the leaf carries the intended write bit
            // (a prior fork may have cleared it).
            space.protect(va, prot, false, phys)
        }
    }

    /// Copy `buf` into the pool at guest `addr`, privatizing copy-on-write pages
    /// first. The caller must have verified the pages are mapped (and, for a guest
    /// store, writable).
    fn copy_in(&mut self, addr: u64, buf: &[u8]) -> Result<(), MemError> {
        let (first, last) = self.page_range(addr, buf.len())?;
        {
            let mut fa = self.fa.lock().unwrap();
            let mut changed = false;
            for p in first..=last {
                let va = self.base + (p as u64) * PAGE_SIZE;
                // Demand-back the page (first touch), then privatize if it is a
                // copy-on-write share.
                if !Self::ensure_backed(&mut self.space, &self.phys, self.prot[p], &mut fa, va) {
                    return Err(MemError::Host("frame pool exhausted".into()));
                }
                changed |= Self::make_writable(&mut self.space, &self.phys, self.prot[p], &mut fa, va);
            }
            self.tlb_dirty |= changed;
        }
        let mut done = 0usize;
        while done < buf.len() {
            let cur = addr + done as u64;
            let page = cur - cur % PAGE_SIZE;
            let off = (cur - page) as usize;
            let n = (buf.len() - done).min(PS - off);
            let t = self
                .space
                .translate(cur, &self.phys)
                .ok_or(MemError::Unmapped(page))?;
            self.phys.write(t.paddr, &buf[done..done + n]);
            done += n;
        }
        Ok(())
    }

    /// Write `buf` to guest `addr` (requires `WRITE`). Kernel copy-out.
    pub fn write(&mut self, addr: u64, buf: &[u8]) -> Result<(), MemError> {
        self.check(addr, buf.len(), Prot::WRITE)?;
        self.copy_in(addr, buf)
    }

    /// Write `buf` on behalf of an executing guest store. Identical to
    /// [`GuestMemory::write`]; copy-on-write pages privatize transparently, so the
    /// interpreter's store to a shared page just succeeds (no kernel round-trip).
    pub fn write_trap(&mut self, addr: u64, buf: &[u8]) -> Result<(), MemError> {
        self.write(addr, buf)
    }

    /// Resolve a copy-on-write fault at `addr` (the KVM/hardware path). Privatizes
    /// the page and returns `true` so the faulting store retries and succeeds;
    /// returns `false` when the fault is genuine (unmapped, or a write to a page
    /// that was never meant to be writable), which the kernel turns into a signal.
    pub fn cow_fault(&mut self, addr: u64, write: bool) -> bool {
        if !write {
            return false;
        }
        let Some(p) = self.page_index(addr) else {
            return false;
        };
        if !self.mapped[p] || !self.prot[p].contains(Prot::WRITE) {
            return false;
        }
        let va = self.base + (p as u64) * PAGE_SIZE;
        let mut fa = self.fa.lock().unwrap();
        Self::make_writable(&mut self.space, &self.phys, self.prot[p], &mut fa, va);
        true
    }

    /// Write `buf` to guest `addr`, bypassing protection but requiring the pages
    /// be mapped. For host-side initialization (ELF segments into read-only pages,
    /// populating a file-backed `mmap`). Privatizes copy-on-write pages so it can
    /// never corrupt a shared frame.
    pub fn write_init(&mut self, addr: u64, buf: &[u8]) -> Result<(), MemError> {
        let (first, last) = self.page_range(addr, buf.len())?;
        for p in first..=last {
            if !self.mapped[p] {
                return Err(MemError::Unmapped(self.page_base(p)));
            }
        }
        // Demand-back each page and privatize any copy-on-write share, then write
        // straight to the frames (bypassing the leaf's protection — the host write
        // goes to the pool, not through the guest MMU).
        {
            let mut fa = self.fa.lock().unwrap();
            let mut changed = false;
            for p in first..=last {
                let va = self.base + (p as u64) * PAGE_SIZE;
                if !Self::ensure_backed(&mut self.space, &self.phys, self.prot[p], &mut fa, va) {
                    return Err(MemError::Host("frame pool exhausted".into()));
                }
                changed |= Self::make_writable(&mut self.space, &self.phys, self.prot[p], &mut fa, va);
            }
            self.tlb_dirty |= changed;
        }
        let mut done = 0usize;
        while done < buf.len() {
            let cur = addr + done as u64;
            let page = cur - cur % PAGE_SIZE;
            let off = (cur - page) as usize;
            let n = (buf.len() - done).min(PS - off);
            let t = self
                .space
                .translate(cur, &self.phys)
                .ok_or(MemError::Unmapped(page))?;
            self.phys.write(t.paddr, &buf[done..done + n]);
            done += n;
        }
        Ok(())
    }

    // ---- fork / exec / release ------------------------------------------

    /// Fork this address space for a new process: the child shares every mapped
    /// frame copy-on-write (both sides go read-only; the first store privatizes).
    /// Only touched pages ever cost a private frame. The child shares the pool.
    #[must_use]
    pub fn fork(&mut self) -> Self {
        let (child_space, parent_kstack, child_kstack) = {
            let mut fa = self.fa.lock().unwrap();
            let mut child = self
                .space
                .fork_cow(&mut fa, &self.phys)
                .expect("fork_cow: frame pool exhausted");
            // `fork_cow` cleared the write bit on *every* leaf, including the
            // shared supervisor control block. Restore it, RW-supervisor, on both
            // sides.
            ctrl::map_into(&mut self.space, &mut fa, &self.phys);
            ctrl::map_into(&mut child, &mut fa, &self.phys);
            // The kernel stack must be private per address space, not CoW-shared:
            // give the parent and child each a fresh frame (each `install_kstack`
            // releases the CoW-shared kstack `fork_cow` handed it). Its contents
            // are transient supervisor scratch, so nothing is lost by not copying.
            let parent_kstack = install_kstack(&mut self.space, &mut fa, &self.phys);
            let child_kstack = install_kstack(&mut child, &mut fa, &self.phys);
            (child, parent_kstack, child_kstack)
        };
        self.kstack_pa = parent_kstack;
        Self {
            base: self.base,
            size: self.size,
            phys: Arc::clone(&self.phys),
            fa: Arc::clone(&self.fa),
            space: child_space,
            prot: self.prot.clone(),
            mapped: self.mapped.clone(),
            file_backed: self.file_backed.clone(),
            tlb_dirty: false,
            kstack_pa: child_kstack,
        }
    }

    /// Tear down the current page tables (returning their frames to the pool) and
    /// install a fresh empty address space in the *same* pool — the `execve` seam.
    /// The `CR3` changes; a KVM vcpu picks the new value up on its next run. The
    /// loader then maps the new image into this same `GuestMemory`.
    pub(crate) fn exec_reset(&mut self) {
        self.reset_space();
    }

    /// Release this address space's frames back to the pool on process exit,
    /// leaving a minimal empty space behind (the task is a zombie and never runs
    /// again). Keeps the pool from filling with dead processes' frames.
    pub(crate) fn release(&mut self) {
        self.reset_space();
    }

    fn reset_space(&mut self) {
        let mut fa = self.fa.lock().unwrap();
        let mut ns = AddrSpace::new(&mut fa, &self.phys).expect("reset: pool exhausted");
        ctrl::map_into(&mut ns, &mut fa, &self.phys);
        let kstack_pa = install_kstack(&mut ns, &mut fa, &self.phys);
        let old = std::mem::replace(&mut self.space, ns);
        // The old space's private kstack frame is an ordinary (non-pinned) leaf,
        // so `destroy` frees it along with the rest; the shared control frames it
        // also references are pinned and survive.
        old.destroy(&mut fa, &self.phys);
        self.kstack_pa = kstack_pa;
        drop(fa);
        for x in &mut self.prot {
            *x = Prot::NONE;
        }
        for x in &mut self.mapped {
            *x = false;
        }
        for x in &mut self.file_backed {
            *x = false;
        }
    }

    // ---- fixed-width helpers --------------------------------------------

    /// Read a little-endian `u32` (requires `READ`).
    pub fn read_u32(&self, addr: u64) -> Result<u32, MemError> {
        let mut b = [0u8; 4];
        self.read(addr, &mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    /// Read a little-endian `u64` (requires `READ`).
    pub fn read_u64(&self, addr: u64) -> Result<u64, MemError> {
        let mut b = [0u8; 8];
        self.read(addr, &mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    /// Write a little-endian `u64` (requires `WRITE`).
    pub fn write_u64(&mut self, addr: u64, val: u64) -> Result<(), MemError> {
        self.write(addr, &val.to_le_bytes())
    }

    /// Read a NUL-terminated string starting at `addr` (requires `READ`),
    /// scanning at most `max` bytes.
    pub fn read_cstr(&self, addr: u64, max: usize) -> Result<Vec<u8>, MemError> {
        let mut out = Vec::new();
        for i in 0..max as u64 {
            let mut b = [0u8; 1];
            self.read(addr + i, &mut b)?;
            if b[0] == 0 {
                return Ok(out);
            }
            out.push(b[0]);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem() -> GuestMemory {
        // 64 KiB region at 0x1_0000 (page-aligned base).
        GuestMemory::new(0x1_0000, 16 * PAGE_SIZE)
    }

    #[test]
    fn unmapped_access_faults() {
        let m = mem();
        assert_eq!(m.read_u32(0x1_0000), Err(MemError::Unmapped(0x1_0000)));
    }

    #[test]
    fn out_of_bounds_faults() {
        let m = mem();
        assert!(matches!(m.read_u32(0x9_0000), Err(MemError::OutOfBounds(_))));
        assert!(matches!(m.read_u32(0x0_0000), Err(MemError::OutOfBounds(_))));
    }

    #[test]
    fn map_then_read_write_roundtrip() {
        let mut m = mem();
        m.map(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        m.write_u64(0x1_0010, 0xdead_beef_cafe_babe).unwrap();
        assert_eq!(m.read_u64(0x1_0010).unwrap(), 0xdead_beef_cafe_babe);
    }

    #[test]
    fn write_to_readonly_faults_but_write_init_succeeds() {
        let mut m = mem();
        m.map(0x1_0000, PAGE_SIZE, Prot::rx()).unwrap();
        assert_eq!(
            m.write(0x1_0000, &[1, 2, 3]),
            Err(MemError::Protection {
                addr: 0x1_0000,
                needed: Prot::WRITE,
            })
        );
        m.write_init(0x1_0000, &[1, 2, 3]).unwrap();
        assert_eq!(m.read_vec(0x1_0000, 3).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn cross_page_access_checks_every_page() {
        let mut m = mem();
        m.map(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        let boundary = 0x1_0000 + PAGE_SIZE - 8;
        assert!(matches!(m.read_u64(boundary + 4), Err(MemError::Unmapped(_))));
        m.map(0x1_0000 + PAGE_SIZE, PAGE_SIZE, Prot::rw()).unwrap();
        m.write_u64(boundary + 4, 42).unwrap();
        assert_eq!(m.read_u64(boundary + 4).unwrap(), 42);
    }

    #[test]
    fn read_cstr_stops_at_nul() {
        let mut m = mem();
        m.map(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        m.write(0x1_0000, b"hi\0rest").unwrap();
        assert_eq!(m.read_cstr(0x1_0000, 64).unwrap(), b"hi");
    }

    #[test]
    fn protect_changes_access() {
        let mut m = mem();
        m.map(0x1_0000, PAGE_SIZE, Prot::READ).unwrap();
        assert!(m.write(0x1_0000, &[9]).is_err());
        m.protect(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        m.write(0x1_0000, &[9]).unwrap();
        assert_eq!(m.read_vec(0x1_0000, 1).unwrap(), vec![9]);
    }

    #[test]
    fn mapped_but_unwritten_reads_zero() {
        let mut m = mem();
        m.map(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        assert_eq!(m.read_vec(0x1_0000, 8).unwrap(), vec![0u8; 8]);
    }

    #[test]
    fn spanning_read_write_across_pages() {
        let mut m = mem();
        m.map(0x1_0000, 2 * PAGE_SIZE, Prot::rw()).unwrap();
        m.write(0x1_0000 + PAGE_SIZE, &[0xEE]).unwrap();
        let boundary = 0x1_0000 + PAGE_SIZE - 2;
        assert_eq!(m.read_vec(boundary, 4).unwrap(), vec![0x00, 0x00, 0xEE, 0x00]);
    }

    #[test]
    fn unmap_then_remap_reads_zero() {
        let mut m = mem();
        m.map(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        m.write(0x1_0000, &[0x77]).unwrap();
        m.unmap(0x1_0000, PAGE_SIZE).unwrap();
        assert!(m.read_vec(0x1_0000, 1).is_err(), "unmapped is inaccessible");
        m.map(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        assert_eq!(m.read_vec(0x1_0000, 1).unwrap(), vec![0], "remap is zeroed");
    }

    #[test]
    fn fork_isolates_parent_and_child_both_directions() {
        let mut parent = mem();
        parent.map(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        parent.write(0x1_0000, &[0xAA]).unwrap();
        let mut child = parent.fork();
        assert_eq!(child.read_vec(0x1_0000, 1).unwrap(), vec![0xAA]);
        parent.write(0x1_0000, &[0xBB]).unwrap();
        child.write(0x1_0000, &[0xCC]).unwrap();
        assert_eq!(parent.read_vec(0x1_0000, 1).unwrap(), vec![0xBB]);
        assert_eq!(child.read_vec(0x1_0000, 1).unwrap(), vec![0xCC]);
    }

    #[test]
    fn fork_copies_only_mapped_pages() {
        let mut parent = mem();
        parent.map(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        parent
            .map(0x1_0000 + 2 * PAGE_SIZE, PAGE_SIZE, Prot::rw())
            .unwrap();
        parent.write(0x1_0000, &[1]).unwrap();
        parent.write(0x1_0000 + 2 * PAGE_SIZE, &[2]).unwrap();
        let child = parent.fork();
        assert_eq!(child.read_vec(0x1_0000, 1).unwrap(), vec![1]);
        assert_eq!(child.read_vec(0x1_0000 + 2 * PAGE_SIZE, 1).unwrap(), vec![2]);
        assert!(child.read_vec(0x1_0000 + PAGE_SIZE, 1).is_err());
    }

    #[test]
    fn write_trap_matches_write_and_cow_fault_privatizes() {
        let mut m = mem();
        m.map(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        m.write_trap(0x1_0000, &[0x5A]).unwrap();
        assert_eq!(m.read_vec(0x1_0000, 1).unwrap(), vec![0x5A]);
        // A write to genuinely read-only memory is a real fault (not CoW).
        m.protect(0x1_0000, PAGE_SIZE, Prot::READ).unwrap();
        assert!(m.write_trap(0x1_0000, &[0]).is_err());
        assert!(!m.cow_fault(0x1_0000, true), "read-only page is a genuine fault");
    }

    #[test]
    fn release_and_exec_reset_return_frames_to_the_pool() {
        let mut m = mem();
        let baseline = m.frames_in_use(); // empty space: PML4 + control-block tables
        // Touch several pages so real data frames are allocated (demand-paged).
        m.map(0x1_0000, 8 * PAGE_SIZE, Prot::rw()).unwrap();
        for i in 0..8 {
            m.write_u64(0x1_0000 + i * PAGE_SIZE, 0xdead).unwrap();
        }
        assert!(m.frames_in_use() > baseline, "touched pages consumed frames");
        // exit / execve must return every data + table frame the process held.
        m.release();
        assert_eq!(m.frames_in_use(), baseline, "release returned all frames");
        m.map(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        m.write_u64(0x1_0000, 1).unwrap();
        m.exec_reset();
        assert_eq!(m.frames_in_use(), baseline, "exec_reset returned all frames");
    }

    #[test]
    fn exec_reset_clears_mappings_and_reuses_pool() {
        let mut m = mem();
        m.map(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        m.write(0x1_0000, &[0x11]).unwrap();
        let old_cr3 = m.cr3();
        m.exec_reset();
        assert_ne!(m.cr3(), old_cr3, "execve installs fresh page tables");
        assert!(m.read_vec(0x1_0000, 1).is_err(), "old mapping is gone");
        m.map(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        assert_eq!(m.read_vec(0x1_0000, 1).unwrap(), vec![0], "fresh zero page");
    }
}
