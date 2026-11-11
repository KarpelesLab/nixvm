//! Per-process x86-64 page tables + a software page-walker.
//!
//! Phase 2 of the MMU refactor. [`crate::vcpu::phys`] (Phase 1) gave us a single
//! always-resident pool of guest-*physical* RAM plus a refcounting frame
//! allocator; this module builds a *real* 4-level x86-64 page-table tree over
//! those frames — one tree per process, each rooted at its own PML4 frame (its
//! `CR3`). An [`AddrSpace`] maps guest *virtual* addresses to physical frames and
//! carries a software walker ([`AddrSpace::translate`]) so the host side (the
//! interpreter, syscall copy-in/out, the loader) can resolve a guest pointer the
//! same way the hardware walker will once this is wired into KVM.
//!
//! # Why real page tables
//! The old model ([`crate::vcpu::mem::GuestMemory`]) keeps a flat protection
//! bitmap and rebuilds identity-mapped shadow tables from it. Here each process
//! instead *owns* a tree of `PhysMem` frames: interior tables (PML4 → PDPT → PD →
//! PT) and 4 KiB leaf pages. This is what lets two processes share a physical
//! frame copy-on-write (see [`AddrSpace::fork_cow`]) and what a per-process `CR3`
//! switch selects between.
//!
//! # 4 KiB leaves only
//! User memory is mapped exclusively with 4 KiB leaf pages — no huge pages. That
//! keeps the walker and the CoW machinery uniform (every leaf is one frame), at
//! the cost of a PT page per 2 MiB of mapped address space.
//!
//! # Atomic PTE publish
//! Every entry (interior or leaf) is installed with a single aligned 64-bit store
//! ([`PhysMem::write_u64_atomic`]). Once these tables back a KVM memslot, a
//! sibling vcpu's hardware walker runs lockless against them; an aligned store is
//! atomic on x86-64, so that walker sees either the whole old or the whole new
//! entry, never a torn mix. We adopt the same discipline now even though nothing
//! walks these concurrently yet, so the invariant holds by construction when
//! Phase 3 wires them in.
//!
//! # `destroy`, not `Drop`
//! An [`AddrSpace`] owns frames in the shared pool, but freeing them requires the
//! `FrameAllocator` and `PhysMem`, which `Drop` cannot borrow. So teardown is the
//! explicit by-value [`AddrSpace::destroy`]; simply dropping an `AddrSpace` leaks
//! every frame it owns. Callers *must* `destroy` it (a debug leak-check would be a
//! reasonable future addition).
//!
//! This module is self-contained and not yet wired into the rest of the crate;
//! its whole public surface is exercised by the tests below, so the module-wide
//! `allow(dead_code)` keeps a non-test release build warning-free (matching the
//! Phase 1 convention in [`crate::vcpu::phys`]).
#![allow(dead_code)]

use super::mem::{PAGE_SIZE, Prot};
use super::phys::{FrameAllocator, PhysMem};

/// The descriptor format of a page-table tree. Both are 4-level, 4 KiB-granule,
/// 48-bit trees of 512 eight-byte entries indexed by the same virtual-address
/// bits, so one walker serves both; only what the bits *in* an entry mean
/// differs. Each tree is in its guest architecture's format so that
/// architecture's hardware walker can use it (KVM/x86-64, HVF/arm64); the
/// software walker reads either. A pool's format lives on its [`PhysMem`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PtFormat {
    /// x86-64 long-mode PTEs.
    X86_64 = 0,
    /// arm64 VMSAv8-64 stage-1 descriptors (4 KiB granule).
    Aarch64 = 1,
}

impl PtFormat {
    pub(crate) fn from_u8(v: u8) -> Self {
        if v == Self::Aarch64 as u8 {
            Self::Aarch64
        } else {
            Self::X86_64
        }
    }

    /// The format a guest of `arch` walks.
    #[must_use]
    pub fn for_arch(arch: crate::abi::Arch) -> Self {
        match arch {
            crate::abi::Arch::Aarch64 => Self::Aarch64,
            crate::abi::Arch::X86_64 => Self::X86_64,
        }
    }
}

// Bit 0 is "present"/"valid" in both formats, so a zero entry is empty in both.
const P: u64 = 1 << 0;

// x86-64 PTE flag bits (identical across all four levels).
const RW: u64 = 1 << 1; // writable
const US: u64 = 1 << 2; // user-accessible
const NX: u64 = 1 << 63; // no-execute (needs EFER.NXE)
/// Mask selecting the frame's physical address out of an x86-64 PTE (bits
/// 12..=51). The low 12 bits are flags; the top 12 bits are NX + reserved.
/// Accessed/Dirty (bits 5, 6), set by the CPU, fall outside it and outside
/// what [`leaf_prot`] reads, so they never change the mapping we installed.
const ADDR_MASK: u64 = 0x000f_ffff_ffff_f000;

// arm64 stage-1 descriptor bits (4 KiB granule).
/// Bit 1: "table" at levels 0–2, "page" at level 3 — set in every entry we write.
const A_TABLE_PAGE: u64 = 1 << 1;
/// `AttrIndx` = 0: `MAIR_EL1` attribute 0, normal write-back memory.
const A_ATTR_NORMAL: u64 = 0 << 2;
/// `AP[1]`: EL0 may access the page.
const A_AP_EL0: u64 = 1 << 6;
/// `AP[2]`: read-only (at every level of privilege).
const A_AP_RO: u64 = 1 << 7;
/// `SH` = inner shareable.
const A_SH_INNER: u64 = 3 << 8;
/// The access flag. Set from the start: nothing here wants access-flag faults.
const A_AF: u64 = 1 << 10;
/// Privileged (EL1) execute-never.
const A_PXN: u64 = 1 << 53;
/// Unprivileged (EL0) execute-never.
const A_UXN: u64 = 1 << 54;
/// The output address of a descriptor (bits 12..=47).
const A_ADDR: u64 = 0x0000_ffff_ffff_f000;

/// Entries per table at every level (a 4 KiB table of 8-byte entries).
const ENTRIES: u64 = 512;

/// Number of paging levels (PML4=4, PDPT=3, PD=2, PT=1).
const PML4_LEVEL: u32 = 4;

/// Index into the level-`n` table for `vaddr`. Level 4 uses bits 39..47, 3 uses
/// 30..38, 2 uses 21..29, 1 uses 12..20 — 9 bits each.
fn table_index(vaddr: u64, level: u32) -> u64 {
    let shift = 12 + 9 * (level - 1);
    (vaddr >> shift) & 0x1ff
}

/// Interior (non-leaf) entry flags. x86-64: present, writable, user — effective
/// permission is the AND down the walk, so interior tables stay permissive and
/// the leaf carries the real protection. The same three bits are a valid arm64
/// table descriptor (valid + table; bit 2 is ignored there) with no
/// hierarchical restrictions.
const INTERIOR: u64 = P | RW | US;

/// How the entries of one pool's tables are encoded: its [`PtFormat`] and the
/// guest-physical address its physical address 0 appears at to the hardware
/// walker ([`PhysMem::base`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Enc {
    arm: bool,
    base: u64,
}

impl Enc {
    pub(crate) fn of(phys: &PhysMem) -> Self {
        Self {
            arm: phys.format() == PtFormat::Aarch64,
            base: phys.base(),
        }
    }

    /// The pool physical address an entry points at.
    fn addr(self, e: u64) -> u64 {
        let mask = if self.arm { A_ADDR } else { ADDR_MASK };
        (e & mask).wrapping_sub(self.base)
    }

    /// An interior entry pointing at the table at pool address `pa`.
    pub(crate) fn interior(self, pa: u64) -> u64 {
        (pa + self.base) | INTERIOR
    }

    /// A leaf mapping pool address `pa` with protection `prot`. `supervisor`
    /// makes the page reachable only from the kernel (ring 0 / EL1).
    pub(crate) fn leaf(self, pa: u64, prot: Prot, supervisor: bool) -> u64 {
        let pa = pa + self.base;
        if !self.arm {
            let mut f = P;
            if !supervisor {
                f |= US;
            }
            if prot.contains(Prot::WRITE) {
                f |= RW;
            }
            if !prot.contains(Prot::EXEC) {
                f |= NX;
            }
            return pa | f;
        }
        let mut f = P | A_TABLE_PAGE | A_ATTR_NORMAL | A_SH_INNER | A_AF;
        // A user page that grants nothing (PROT_NONE with a frame behind it) is
        // kept present but kernel-only, so EL0 faults on it.
        if !supervisor && prot.contains(Prot::READ) {
            f |= A_AP_EL0;
        }
        if !prot.contains(Prot::WRITE) {
            f |= A_AP_RO;
        }
        // The kernel never runs user pages, nor EL0 kernel ones.
        let exec = prot.contains(Prot::EXEC);
        if supervisor {
            f |= A_UXN | if exec { 0 } else { A_PXN };
        } else {
            f |= A_PXN
                | if exec && prot.contains(Prot::READ) {
                    0
                } else {
                    A_UXN
                };
        }
        pa | f
    }

    /// The protection of the present leaf `e`, and whether it is kernel-only.
    /// A present page is always readable (to the level that may access it).
    fn leaf_prot(self, e: u64) -> (Prot, bool) {
        let mut bits = Prot::READ.0;
        if !self.arm {
            if e & RW != 0 {
                bits |= Prot::WRITE.0;
            }
            if e & NX == 0 {
                bits |= Prot::EXEC.0;
            }
            return (Prot(bits), e & US == 0);
        }
        let supervisor = e & A_AP_EL0 == 0;
        if e & A_AP_RO == 0 {
            bits |= Prot::WRITE.0;
        }
        let xn = if supervisor { A_PXN } else { A_UXN };
        if e & xn == 0 {
            bits |= Prot::EXEC.0;
        }
        (Prot(bits), supervisor)
    }

    /// Whether the present leaf `e` is writable.
    fn writable(self, e: u64) -> bool {
        if self.arm {
            e & A_AP_RO == 0
        } else {
            e & RW != 0
        }
    }

    /// The present leaf `e` with write permission taken away.
    fn read_only(self, e: u64) -> u64 {
        if self.arm { e | A_AP_RO } else { e & !RW }
    }
}

/// The result of a successful software walk: the physical byte address the guest
/// virtual address resolves to, plus the leaf's effective protection and whether
/// it is supervisor-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Translation {
    /// Physical byte address: `frame_pa | (vaddr & 0xfff)`.
    pub paddr: u64,
    /// Effective protection of the leaf page.
    pub prot: Prot,
    /// `true` if the leaf's `US` bit is clear (kernel-only).
    pub supervisor: bool,
}

/// Why a [`AddrSpace::map`] could not install a mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapErr {
    /// The address space belongs to a process that has exited.
    Retired,
    /// `vaddr` or `frame_pa` was not 4 KiB-aligned; the offending address.
    Misaligned(u64),
    /// The frame allocator was exhausted while allocating an interior table. Any
    /// interior tables allocated before the failure stay linked into the tree
    /// (reachable, reused by a later `map`, and freed by `destroy`) — nothing is
    /// leaked, but the mapping was not installed.
    OutOfFrames,
}

/// One process's virtual address space: a 4-level x86-64 page-table tree whose
/// frames come from the shared [`FrameAllocator`]/[`PhysMem`] pool.
///
/// The `pml4` physical address is both the tree root and the identity of the
/// address space (it goes into KVM `sregs.cr3`). An `AddrSpace` must be torn down
/// with [`AddrSpace::destroy`], never merely dropped — see the module docs.
#[derive(Debug)]
pub struct AddrSpace {
    /// Physical address of the PML4 frame (the `CR3` value).
    pml4: u64,
    // TODO: a small direct-mapped software TLB (last N vaddr→paddr+prot),
    // flushed on every map/unmap/protect/fork_cow, to make repeated `translate`
    // cheap for the Phase 3 interpreter. Correctness first; omitted for now.
}

impl AddrSpace {
    /// Create an empty address space: allocate a zeroed PML4 frame. `None` if the
    /// allocator is exhausted.
    #[must_use]
    pub fn new(fa: &mut FrameAllocator, phys: &PhysMem) -> Option<Self> {
        let pml4 = fa.alloc(phys)?; // alloc returns a zeroed frame
        Some(Self { pml4 })
    }

    /// The PML4 physical address — the `CR3` value and the address space's
    /// identity.
    #[must_use]
    pub fn cr3(&self) -> u64 {
        self.pml4
    }

    /// A space holding no frames at all, for a process that has exited. Frame 0
    /// is the permanently pinned, zeroed null frame, so every lookup finds nothing
    /// and [`AddrSpace::map`] refuses.
    pub(crate) fn retired() -> Self {
        Self { pml4: 0 }
    }

    /// Read the entry at slot `idx` of the table at `table_pa`.
    fn entry(phys: &PhysMem, table_pa: u64, idx: u64) -> u64 {
        phys.read_u64(table_pa + idx * 8)
    }

    /// Follow the present interior entry at `table_pa[idx]`, or allocate a fresh
    /// zeroed sub-table and link it in (`INTERIOR` flags, atomic publish).
    /// Returns the child table's physical address, or `None` on exhaustion.
    fn next_or_alloc(
        enc: Enc,
        table_pa: u64,
        idx: u64,
        fa: &mut FrameAllocator,
        phys: &PhysMem,
    ) -> Option<u64> {
        let e = Self::entry(phys, table_pa, idx);
        if e & P != 0 {
            return Some(enc.addr(e));
        }
        let child = fa.alloc(phys)?;
        phys.write_u64_atomic(table_pa + idx * 8, enc.interior(child));
        Some(child)
    }

    /// Follow the present interior entry at `table_pa[idx]`, or `None` if it is
    /// not present (read-only walk; used by `translate`/`unmap`/`protect`).
    fn next(enc: Enc, phys: &PhysMem, table_pa: u64, idx: u64) -> Option<u64> {
        let e = Self::entry(phys, table_pa, idx);
        (e & P != 0).then(|| enc.addr(e))
    }

    /// Install `vaddr -> frame_pa` with protection `prot`, allocating any missing
    /// interior tables. Both addresses must be 4 KiB-aligned.
    ///
    /// If a leaf already existed at `vaddr` it is overwritten and its old
    /// `frame_pa` is returned as `Ok(Some(old))` so the caller can `decref` the
    /// clobbered frame; `Ok(None)` means the slot was previously empty. On
    /// allocation failure returns `Err(MapErr::OutOfFrames)` with no leak (see
    /// [`MapErr::OutOfFrames`]).
    pub fn map(
        &mut self,
        vaddr: u64,
        frame_pa: u64,
        prot: Prot,
        supervisor: bool,
        fa: &mut FrameAllocator,
        phys: &PhysMem,
    ) -> Result<Option<u64>, MapErr> {
        if self.pml4 == 0 {
            return Err(MapErr::Retired);
        }
        if !vaddr.is_multiple_of(PAGE_SIZE) {
            return Err(MapErr::Misaligned(vaddr));
        }
        if !frame_pa.is_multiple_of(PAGE_SIZE) {
            return Err(MapErr::Misaligned(frame_pa));
        }
        let enc = Enc::of(phys);
        let pdpt = Self::next_or_alloc(enc, self.pml4, table_index(vaddr, 4), fa, phys)
            .ok_or(MapErr::OutOfFrames)?;
        let pd = Self::next_or_alloc(enc, pdpt, table_index(vaddr, 3), fa, phys)
            .ok_or(MapErr::OutOfFrames)?;
        let pt = Self::next_or_alloc(enc, pd, table_index(vaddr, 2), fa, phys)
            .ok_or(MapErr::OutOfFrames)?;

        let leaf_pa = pt + table_index(vaddr, 1) * 8;
        let old = phys.read_u64(leaf_pa);
        let old_frame = (old & P != 0).then(|| enc.addr(old));

        // Single aligned store: a concurrent hardware walker sees whole-old or
        // whole-new, never a torn PTE.
        phys.write_u64_atomic(leaf_pa, enc.leaf(frame_pa, prot, supervisor));
        Ok(old_frame)
    }

    /// Walk to the leaf PTE physical address for `vaddr`, following only present
    /// interior entries. `None` if any interior level is not present.
    fn leaf_pte_pa(&self, enc: Enc, phys: &PhysMem, vaddr: u64) -> Option<u64> {
        let pdpt = Self::next(enc, phys, self.pml4, table_index(vaddr, 4))?;
        let pd = Self::next(enc, phys, pdpt, table_index(vaddr, 3))?;
        let pt = Self::next(enc, phys, pd, table_index(vaddr, 2))?;
        Some(pt + table_index(vaddr, 1) * 8)
    }

    /// Software 4-level walk: resolve `vaddr` to a physical byte address plus the
    /// leaf's protection. `None` if any level (including the leaf) is not present.
    #[must_use]
    pub fn translate(&self, vaddr: u64, phys: &PhysMem) -> Option<Translation> {
        let enc = Enc::of(phys);
        let leaf_pa = self.leaf_pte_pa(enc, phys, vaddr)?;
        let entry = phys.read_u64(leaf_pa);
        if entry & P == 0 {
            return None;
        }
        let (prot, supervisor) = enc.leaf_prot(entry);
        Some(Translation {
            paddr: enc.addr(entry) | (vaddr & (PAGE_SIZE - 1)),
            prot,
            supervisor,
        })
    }

    /// Rewrite the permission bits of an existing leaf (same frame). Returns
    /// whether `vaddr` was mapped. AD bits are dropped in the rewrite, which is
    /// correct — the CPU re-sets them on the next access.
    pub fn protect(&mut self, vaddr: u64, prot: Prot, supervisor: bool, phys: &PhysMem) -> bool {
        let enc = Enc::of(phys);
        let Some(leaf_pa) = self.leaf_pte_pa(enc, phys, vaddr) else {
            return false;
        };
        let entry = phys.read_u64(leaf_pa);
        if entry & P == 0 {
            return false;
        }
        phys.write_u64_atomic(leaf_pa, enc.leaf(enc.addr(entry), prot, supervisor));
        true
    }

    /// Whether every entry of the table at `table_pa` is zero (not present).
    fn table_is_empty(phys: &PhysMem, table_pa: u64) -> bool {
        (0..ENTRIES).all(|i| phys.read_u64(table_pa + i * 8) == 0)
    }

    /// Remove the mapping at `vaddr`. Clears the leaf PTE (atomic store of 0) and
    /// returns the frame it pointed at so the caller can `decref` it — this does
    /// *not* touch the data frame's refcount. `None` if `vaddr` was not mapped.
    ///
    /// Any interior table (PT, then PD, then PDPT) that becomes entirely empty as
    /// a result is freed back to `fa` and unlinked from its parent, so an address
    /// space does not accumulate empty tables after churn.
    pub fn unmap(&mut self, vaddr: u64, fa: &mut FrameAllocator, phys: &PhysMem) -> Option<u64> {
        // Resolve the full chain of tables, bailing if any level is absent.
        let enc = Enc::of(phys);
        let pdpt = Self::next(enc, phys, self.pml4, table_index(vaddr, 4))?;
        let pd = Self::next(enc, phys, pdpt, table_index(vaddr, 3))?;
        let pt = Self::next(enc, phys, pd, table_index(vaddr, 2))?;
        let leaf_pa = pt + table_index(vaddr, 1) * 8;
        let entry = phys.read_u64(leaf_pa);
        if entry & P == 0 {
            return None;
        }
        let frame = enc.addr(entry);
        phys.write_u64_atomic(leaf_pa, 0);

        // Walk back up: free each table that just became empty and clear the
        // parent slot that pointed at it. Stop at the first non-empty level (and
        // never free the PML4 — it lives for the address space's lifetime).
        if Self::table_is_empty(phys, pt) {
            phys.write_u64_atomic(pd + table_index(vaddr, 2) * 8, 0);
            fa.free(pt);
            if Self::table_is_empty(phys, pd) {
                phys.write_u64_atomic(pdpt + table_index(vaddr, 3) * 8, 0);
                fa.free(pd);
                if Self::table_is_empty(phys, pdpt) {
                    phys.write_u64_atomic(self.pml4 + table_index(vaddr, 4) * 8, 0);
                    fa.free(pdpt);
                }
            }
        }
        Some(frame)
    }

    /// Create a copy-on-write child of this address space.
    ///
    /// The child gets fresh interior tables (its own PML4/PDPT/PD/PT frames) but
    /// *shares* every mapped leaf frame with the parent: for each present leaf we
    /// `incref` the frame and give the child a read-only leaf pointing at it, then
    /// clear write in the parent's matching leaf too — so a store on either side
    /// faults and the (later-phase) fault handler privatizes the frame. NX/US are
    /// preserved on both sides.
    ///
    /// Failure semantics: the parent is left **untouched** on failure. The child
    /// is built (and its shared frames `incref`ed) in a first pass that never
    /// mutates the parent; only after that pass fully succeeds does a second pass
    /// clear write in the parent. On allocation failure the half-built child is
    /// `destroy`ed (freeing its interior tables and `decref`ing the frames the
    /// first pass shared, exactly undoing the increfs) and `None` is returned.
    #[must_use]
    pub fn fork_cow(&mut self, fa: &mut FrameAllocator, phys: &PhysMem) -> Option<Self> {
        let enc = Enc::of(phys);
        let child_pml4 = fa.alloc(phys)?;
        let child = Self { pml4: child_pml4 };
        if clone_subtree(enc, PML4_LEVEL, self.pml4, child_pml4, fa, phys).is_err() {
            // Undo everything the first pass did without having touched the parent.
            child.destroy(fa, phys);
            return None;
        }
        // Second pass, now that the child is fully built: make the parent's leaves
        // read-only so both sides fault-on-write.
        clear_leaf_write(enc, PML4_LEVEL, self.pml4, phys);
        Some(child)
    }

    /// Rewrite every entry of the tree from encoding `from` to `to` (another
    /// format, another guest-physical base) in place: the same tables, frames
    /// and protections, readable by the other walker. For a pool whose format
    /// changes after its first tables were built.
    pub(crate) fn reencode(&mut self, from: Enc, to: Enc, phys: &PhysMem) {
        if self.pml4 != 0 && from != to {
            reencode_subtree(from, to, PML4_LEVEL, self.pml4, phys);
        }
    }

    /// Tear down the whole tree: `decref` every mapped data frame and free every
    /// page-table frame (leaves up through the PML4) back to `fa`. Consumes the
    /// `AddrSpace` — see the module docs on why this can't be `Drop`.
    pub fn destroy(self, fa: &mut FrameAllocator, phys: &PhysMem) {
        if self.pml4 == 0 {
            return;
        }
        free_subtree(Enc::of(phys), PML4_LEVEL, self.pml4, fa, phys);
        fa.free(self.pml4);
    }
}

/// Recursively copy the parent table at `parent_pa` (level `level`) into the
/// freshly-allocated child table at `child_pa`. Interior levels allocate new
/// child sub-tables; the leaf level (`level == 1`) shares the parent's data
/// frames copy-on-write: `incref` and install a read-only child leaf. Returns
/// `Err(())` on allocator exhaustion, leaving the child partially built but fully
/// reachable (so [`AddrSpace::destroy`] can reclaim it and balance the increfs).
fn clone_subtree(
    enc: Enc,
    level: u32,
    parent_pa: u64,
    child_pa: u64,
    fa: &mut FrameAllocator,
    phys: &PhysMem,
) -> Result<(), ()> {
    for idx in 0..ENTRIES {
        let e = AddrSpace::entry(phys, parent_pa, idx);
        if e & P == 0 {
            continue;
        }
        if level == 1 {
            // Leaf: share the frame read-only. Preserve every bit (frame, user,
            // execute, …) except write, which goes so a write faults on both sides.
            fa.incref(enc.addr(e));
            phys.write_u64_atomic(child_pa + idx * 8, enc.read_only(e));
        } else {
            // Interior: allocate the child's own sub-table and recurse.
            let child_sub = fa.alloc(phys).ok_or(())?;
            phys.write_u64_atomic(child_pa + idx * 8, enc.interior(child_sub));
            clone_subtree(enc, level - 1, enc.addr(e), child_sub, fa, phys)?;
        }
    }
    Ok(())
}

/// [`AddrSpace::reencode`] for the table at `table_pa` (level `level`) and
/// everything below it.
fn reencode_subtree(from: Enc, to: Enc, level: u32, table_pa: u64, phys: &PhysMem) {
    for idx in 0..ENTRIES {
        let e = AddrSpace::entry(phys, table_pa, idx);
        if e & P == 0 {
            continue;
        }
        let pa = from.addr(e);
        let new = if level == 1 {
            let (prot, supervisor) = from.leaf_prot(e);
            to.leaf(pa, prot, supervisor)
        } else {
            reencode_subtree(from, to, level - 1, pa, phys);
            to.interior(pa)
        };
        phys.write_u64_atomic(table_pa + idx * 8, new);
    }
}

/// Clear the `RW` bit on every present leaf reachable from the table at
/// `table_pa` (level `level`), republishing each atomically. Used to make a
/// parent read-only after a CoW fork. Idempotent on already-read-only leaves.
fn clear_leaf_write(enc: Enc, level: u32, table_pa: u64, phys: &PhysMem) {
    for idx in 0..ENTRIES {
        let e = AddrSpace::entry(phys, table_pa, idx);
        if e & P == 0 {
            continue;
        }
        if level == 1 {
            if enc.writable(e) {
                phys.write_u64_atomic(table_pa + idx * 8, enc.read_only(e));
            }
        } else {
            clear_leaf_write(enc, level - 1, enc.addr(e), phys);
        }
    }
}

/// Recursively free every frame reachable from the table at `table_pa` (level
/// `level`), *excluding* `table_pa` itself (the caller frees that). At the leaf
/// level each present entry's data frame is `decref`ed; at interior levels each
/// child table is recursed into and then freed.
fn free_subtree(enc: Enc, level: u32, table_pa: u64, fa: &mut FrameAllocator, phys: &PhysMem) {
    for idx in 0..ENTRIES {
        let e = AddrSpace::entry(phys, table_pa, idx);
        if e & P == 0 {
            continue;
        }
        let child = enc.addr(e);
        if level == 1 {
            // Data frame: drop this address space's reference (CoW-shared frames
            // survive until their last owner frees them).
            fa.free(child);
        } else {
            free_subtree(enc, level - 1, child, fa, phys);
            fa.free(child);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: u64 = PAGE_SIZE;

    /// A small pool + allocator, sized in frames so counters stay cheap.
    fn pool(nframes: u64) -> (PhysMem, FrameAllocator) {
        let phys = PhysMem::new(nframes as usize * FRAME as usize);
        let fa = FrameAllocator::new(phys.nframes());
        (phys, fa)
    }

    /// Raw leaf PTE for `vaddr` (following present interiors), or 0 if any level
    /// is absent — for asserting the exact bits `map`/`protect` published.
    fn raw_leaf(space: &AddrSpace, phys: &PhysMem, vaddr: u64) -> u64 {
        match space.leaf_pte_pa(Enc::of(phys), phys, vaddr) {
            Some(pa) => phys.read_u64(pa),
            None => 0,
        }
    }

    /// [`pool`] with arm64 descriptors (and so an IPA window as its base).
    fn arm_pool(nframes: u64) -> (PhysMem, FrameAllocator) {
        let (phys, fa) = pool(nframes);
        phys.set_format(PtFormat::Aarch64);
        (phys, fa)
    }

    #[test]
    fn arm64_leaves_carry_the_hardware_bits() {
        let (phys, mut fa) = arm_pool(64);
        let base = phys.base();
        assert_ne!(base, 0, "an arm64 pool has an IPA window");
        let mut space = AddrSpace::new(&mut fa, &phys).unwrap();
        let (rw, rx, sup) = (
            fa.alloc(&phys).unwrap(),
            fa.alloc(&phys).unwrap(),
            fa.alloc(&phys).unwrap(),
        );
        space
            .map(0x1000, rw, Prot::rw(), false, &mut fa, &phys)
            .unwrap();
        space
            .map(0x2000, rx, Prot::rx(), false, &mut fa, &phys)
            .unwrap();
        space
            .map(0x3000, sup, Prot::rx(), true, &mut fa, &phys)
            .unwrap();
        let none = fa.alloc(&phys).unwrap();
        space
            .map(0x4000, none, Prot::NONE, false, &mut fa, &phys)
            .unwrap();

        let e = raw_leaf(&space, &phys, 0x1000);
        assert_eq!(e & A_ADDR, base + rw, "output address is in the window");
        assert_eq!(e & 3, 3, "valid page");
        assert_ne!(e & A_AF, 0, "access flag set");
        assert_ne!(e & A_AP_EL0, 0, "EL0 may access");
        assert_eq!(e & A_AP_RO, 0, "writable");
        assert_ne!(e & A_UXN, 0, "not executable");
        assert_ne!(e & A_PXN, 0, "never executable by the kernel");

        let e = raw_leaf(&space, &phys, 0x2000);
        assert_ne!(e & A_AP_RO, 0, "read-only");
        assert_eq!(e & A_UXN, 0, "executable at EL0");
        assert_ne!(e & A_PXN, 0);

        let e = raw_leaf(&space, &phys, 0x3000);
        assert_eq!(e & A_AP_EL0, 0, "kernel only");
        assert_ne!(e & A_UXN, 0);
        assert_eq!(e & A_PXN, 0, "kernel-executable");

        let e = raw_leaf(&space, &phys, 0x4000);
        assert_eq!(e & A_AP_EL0, 0, "PROT_NONE: no EL0 access at all");
        assert_ne!(e & A_UXN, 0);

        let t = space.translate(0x1abc, &phys).unwrap();
        assert_eq!(
            (t.paddr, t.prot, t.supervisor),
            (rw | 0xabc, Prot::rw(), false)
        );
        let t = space.translate(0x2000, &phys).unwrap();
        assert_eq!((t.prot, t.supervisor), (Prot::rx(), false));
        let t = space.translate(0x3000, &phys).unwrap();
        assert_eq!((t.prot, t.supervisor), (Prot::rx(), true));

        // Interior entries point into the window too.
        let l0 = phys.read_u64(space.cr3());
        assert_eq!(l0 & 3, 3, "valid table");
        assert!((l0 & A_ADDR) >= base);
        space.destroy(&mut fa, &phys);
    }

    #[test]
    fn arm64_fork_cow_takes_write_away_on_both_sides() {
        let (phys, mut fa) = arm_pool(64);
        let mut parent = AddrSpace::new(&mut fa, &phys).unwrap();
        let f = fa.alloc(&phys).unwrap();
        parent
            .map(0x5000, f, Prot::rw(), false, &mut fa, &phys)
            .unwrap();
        let child = parent.fork_cow(&mut fa, &phys).unwrap();
        for space in [&parent, &child] {
            let t = space.translate(0x5000, &phys).unwrap();
            assert_eq!(t.paddr, f);
            assert_eq!(t.prot, Prot::READ, "shared read-only");
        }
        assert_eq!(fa.refcount(f), 2);
        child.destroy(&mut fa, &phys);
        parent.destroy(&mut fa, &phys);
    }

    #[test]
    fn reencode_keeps_every_mapping() {
        let (phys, mut fa) = pool(64);
        let mut space = AddrSpace::new(&mut fa, &phys).unwrap();
        let maps = [
            (0x1000u64, Prot::rw(), false),
            (0x2000, Prot::rx(), false),
            (0x40_0000_0000, Prot::READ, false),
            (0x7f_ffff_f000, Prot::rx(), true),
        ];
        let mut frames = Vec::new();
        for &(va, prot, sup) in &maps {
            let f = fa.alloc(&phys).unwrap();
            frames.push(f);
            space.map(va, f, prot, sup, &mut fa, &phys).unwrap();
        }
        let check = |space: &AddrSpace, phys: &PhysMem| {
            for (&(va, prot, sup), &f) in maps.iter().zip(&frames) {
                let t = space.translate(va, phys).unwrap();
                assert_eq!((t.paddr, t.prot, t.supervisor), (f, prot, sup), "{va:#x}");
            }
        };
        let x86 = Enc::of(&phys);
        phys.set_format(PtFormat::Aarch64);
        let arm = Enc::of(&phys);
        space.reencode(x86, arm, &phys);
        check(&space, &phys);
        assert_eq!(
            raw_leaf(&space, &phys, 0x1000) & A_ADDR,
            arm.base + frames[0]
        );
        phys.set_format(PtFormat::X86_64);
        space.reencode(arm, Enc::of(&phys), &phys);
        check(&space, &phys);
        assert_eq!(raw_leaf(&space, &phys, 0x1000) & ADDR_MASK, frames[0]);
        space.destroy(&mut fa, &phys);
    }

    #[test]
    fn map_then_translate_and_raw_bits() {
        let (phys, mut fa) = pool(32);
        let mut space = AddrSpace::new(&mut fa, &phys).unwrap();
        let frame = fa.alloc(&phys).unwrap();

        let vaddr = 0x1234_5000;
        assert_eq!(
            space
                .map(vaddr, frame, Prot::rw(), false, &mut fa, &phys)
                .unwrap(),
            None,
            "fresh slot clobbers nothing"
        );

        let t = space.translate(vaddr + 0x10, &phys).unwrap();
        assert_eq!(t.paddr, frame + 0x10, "frame | offset");
        assert_eq!(t.prot, Prot::rw());
        assert!(!t.supervisor);

        // Raw PTE bits: present, writable, user, and NX set (no EXEC).
        let e = raw_leaf(&space, &phys, vaddr);
        assert_eq!(e & ADDR_MASK, frame);
        assert_ne!(e & P, 0, "present");
        assert_ne!(e & RW, 0, "writable");
        assert_ne!(e & US, 0, "user");
        assert_ne!(e & NX, 0, "NX set for non-exec");

        space.destroy(&mut fa, &phys);
    }

    #[test]
    fn exec_mapping_has_no_nx() {
        let (phys, mut fa) = pool(16);
        let mut space = AddrSpace::new(&mut fa, &phys).unwrap();
        let frame = fa.alloc(&phys).unwrap();
        space
            .map(0x4000, frame, Prot::rx(), false, &mut fa, &phys)
            .unwrap();
        let e = raw_leaf(&space, &phys, 0x4000);
        assert_eq!(e & NX, 0, "executable leaf clears NX");
        assert_eq!(e & RW, 0, "read-exec is not writable");
        let t = space.translate(0x4000, &phys).unwrap();
        assert_eq!(t.prot, Prot::rx());
        space.destroy(&mut fa, &phys);
    }

    #[test]
    fn supervisor_mapping_clears_us() {
        let (phys, mut fa) = pool(16);
        let mut space = AddrSpace::new(&mut fa, &phys).unwrap();
        let frame = fa.alloc(&phys).unwrap();
        space
            .map(0x8000, frame, Prot::rw(), true, &mut fa, &phys)
            .unwrap();
        let e = raw_leaf(&space, &phys, 0x8000);
        assert_eq!(e & US, 0, "supervisor leaf has US clear");
        let t = space.translate(0x8000, &phys).unwrap();
        assert!(t.supervisor, "translate reports supervisor-only");
        space.destroy(&mut fa, &phys);
    }

    #[test]
    fn spanning_slots_allocates_the_right_interior_tables() {
        let (phys, mut fa) = pool(64);
        let mut space = AddrSpace::new(&mut fa, &phys).unwrap();
        let base = fa.alloc_count(); // 1: the PML4

        // Same PML4/PDPT/PD, different PT slot: +1 PT + data frame (2 allocs).
        let f0 = fa.alloc(&phys).unwrap();
        space.map(0, f0, Prot::rw(), false, &mut fa, &phys).unwrap();
        assert_eq!(fa.alloc_count() - base, 1 /*data*/ + 3 /*PDPT,PD,PT*/);

        // Different PD index (bit 21): new PT only (share PDPT+PD) + data frame.
        let after0 = fa.alloc_count();
        let f1 = fa.alloc(&phys).unwrap();
        let v1 = 1u64 << 21;
        space
            .map(v1, f1, Prot::rw(), false, &mut fa, &phys)
            .unwrap();
        assert_eq!(fa.alloc_count() - after0, 1 /*data*/ + 1 /*PT*/);

        // Different PDPT index (bit 30): new PD + PT + data frame.
        let after1 = fa.alloc_count();
        let f2 = fa.alloc(&phys).unwrap();
        let v2 = 1u64 << 30;
        space
            .map(v2, f2, Prot::rw(), false, &mut fa, &phys)
            .unwrap();
        assert_eq!(fa.alloc_count() - after1, 1 /*data*/ + 2 /*PD,PT*/);

        // Different PML4 index (bit 39): new PDPT + PD + PT + data frame.
        let after2 = fa.alloc_count();
        let f3 = fa.alloc(&phys).unwrap();
        let v3 = 1u64 << 39;
        space
            .map(v3, f3, Prot::rw(), false, &mut fa, &phys)
            .unwrap();
        assert_eq!(
            fa.alloc_count() - after2,
            1 /*data*/ + 3 /*PDPT,PD,PT*/
        );

        // Every mapping resolves to its own frame.
        assert_eq!(space.translate(0, &phys).unwrap().paddr, f0);
        assert_eq!(space.translate(v1, &phys).unwrap().paddr, f1);
        assert_eq!(space.translate(v2, &phys).unwrap().paddr, f2);
        assert_eq!(space.translate(v3, &phys).unwrap().paddr, f3);

        // `map` transferred each freshly-allocated data frame's single reference
        // to the page table (it does not incref), so `destroy` frees the four data
        // frames along with every interior table and the PML4 — nothing is left,
        // and the test must not free the data frames again.
        space.destroy(&mut fa, &phys);
        assert_eq!(fa.alloc_count(), 0, "no frames leaked");
    }

    #[test]
    fn map_overwrite_returns_old_frame() {
        let (phys, mut fa) = pool(16);
        let mut space = AddrSpace::new(&mut fa, &phys).unwrap();
        let a = fa.alloc(&phys).unwrap();
        let b = fa.alloc(&phys).unwrap();
        space
            .map(0x2000, a, Prot::rw(), false, &mut fa, &phys)
            .unwrap();
        let old = space
            .map(0x2000, b, Prot::rx(), false, &mut fa, &phys)
            .unwrap();
        assert_eq!(old, Some(a), "overwrite returns the clobbered frame");
        assert_eq!(space.translate(0x2000, &phys).unwrap().paddr, b);
        space.destroy(&mut fa, &phys);
    }

    #[test]
    fn misaligned_map_is_rejected() {
        let (phys, mut fa) = pool(8);
        let mut space = AddrSpace::new(&mut fa, &phys).unwrap();
        assert_eq!(
            space.map(0x1001, 0x1000, Prot::rw(), false, &mut fa, &phys),
            Err(MapErr::Misaligned(0x1001))
        );
        assert_eq!(
            space.map(0x1000, 0x1001, Prot::rw(), false, &mut fa, &phys),
            Err(MapErr::Misaligned(0x1001))
        );
        space.destroy(&mut fa, &phys);
    }

    #[test]
    fn unmap_returns_frame_and_frees_empty_interior_tables() {
        let (phys, mut fa) = pool(16);
        let mut space = AddrSpace::new(&mut fa, &phys).unwrap();
        let baseline = fa.alloc_count(); // just the PML4
        let frame = fa.alloc(&phys).unwrap();
        space
            .map(0x9000, frame, Prot::rw(), false, &mut fa, &phys)
            .unwrap();
        // PML4 + PDPT + PD + PT + data.
        assert_eq!(fa.alloc_count(), baseline + 4);

        let got = space.unmap(0x9000, &mut fa, &phys);
        assert_eq!(got, Some(frame), "unmap returns the data frame");
        assert!(space.translate(0x9000, &phys).is_none(), "gone");
        // The now-empty PT/PD/PDPT were freed; only the PML4 (baseline) plus the
        // still-owned data frame remain — unmap doesn't decref the data frame.
        assert_eq!(
            fa.alloc_count(),
            baseline + 1,
            "empty interior tables reclaimed"
        );
        assert_eq!(fa.refcount(frame), 1, "caller still owns the data frame");

        assert_eq!(space.unmap(0x9000, &mut fa, &phys), None, "already gone");
        fa.free(frame);
        space.destroy(&mut fa, &phys);
        assert_eq!(fa.alloc_count(), 0);
    }

    #[test]
    fn unmap_keeps_shared_interior_tables() {
        let (phys, mut fa) = pool(16);
        let mut space = AddrSpace::new(&mut fa, &phys).unwrap();
        let f0 = fa.alloc(&phys).unwrap();
        let f1 = fa.alloc(&phys).unwrap();
        // Two leaves sharing the same PT (adjacent pages).
        space.map(0, f0, Prot::rw(), false, &mut fa, &phys).unwrap();
        space
            .map(FRAME, f1, Prot::rw(), false, &mut fa, &phys)
            .unwrap();
        let with_both = fa.alloc_count();

        // Unmapping one leaf must not free the still-shared PT/PD/PDPT. unmap
        // hands the frame back without decref'ing, so the live count is unchanged
        // by the unmap itself; the surviving mapping still resolves.
        assert_eq!(space.unmap(0, &mut fa, &phys), Some(f0));
        assert_eq!(fa.alloc_count(), with_both, "shared interior tables kept");
        assert_eq!(space.translate(FRAME, &phys).unwrap().paddr, f1);
        assert!(
            space.translate(0, &phys).is_none(),
            "the unmapped leaf is gone"
        );

        fa.free(f0); // caller decrefs the frame unmap handed back
        space.unmap(FRAME, &mut fa, &phys).unwrap();
        fa.free(f1);
        space.destroy(&mut fa, &phys);
        assert_eq!(fa.alloc_count(), 0);
    }

    #[test]
    fn protect_flips_rw_and_nx() {
        let (phys, mut fa) = pool(16);
        let mut space = AddrSpace::new(&mut fa, &phys).unwrap();
        let frame = fa.alloc(&phys).unwrap();
        space
            .map(0x5000, frame, Prot::rw(), false, &mut fa, &phys)
            .unwrap();

        assert!(space.protect(0x5000, Prot::rx(), false, &phys));
        let e = raw_leaf(&space, &phys, 0x5000);
        assert_eq!(e & RW, 0, "write cleared");
        assert_eq!(e & NX, 0, "exec set (NX cleared)");
        assert_eq!(e & ADDR_MASK, frame, "same frame kept");
        let t = space.translate(0x5000, &phys).unwrap();
        assert_eq!(t.prot, Prot::rx());

        // Back to read-only data (NX set again, no write).
        assert!(space.protect(0x5000, Prot::READ, false, &phys));
        let e = raw_leaf(&space, &phys, 0x5000);
        assert_ne!(e & NX, 0);
        assert_eq!(e & RW, 0);

        assert!(!space.protect(0x6000, Prot::rw(), false, &phys), "unmapped");
        space.destroy(&mut fa, &phys);
    }

    #[test]
    fn not_present_walk_returns_none_at_each_level() {
        let (phys, mut fa) = pool(16);
        let mut space = AddrSpace::new(&mut fa, &phys).unwrap();
        // Nothing mapped at all: PML4 slot empty.
        assert!(space.translate(0, &phys).is_none());

        // Map one page, then probe addresses that diverge at each interior level
        // — each shares the levels above but hits an absent entry below.
        let frame = fa.alloc(&phys).unwrap();
        space
            .map(0, frame, Prot::rw(), false, &mut fa, &phys)
            .unwrap();
        assert!(
            space.translate(1u64 << 39, &phys).is_none(),
            "absent PML4 entry"
        );
        assert!(
            space.translate(1u64 << 30, &phys).is_none(),
            "absent PDPT entry"
        );
        assert!(
            space.translate(1u64 << 21, &phys).is_none(),
            "absent PD entry"
        );
        assert!(
            space.translate(1u64 << 12, &phys).is_none(),
            "absent PT entry"
        );
        assert!(
            space.translate(0, &phys).is_some(),
            "the mapped page still resolves"
        );

        space.unmap(0, &mut fa, &phys);
        fa.free(frame);
        space.destroy(&mut fa, &phys);
    }

    #[test]
    fn fork_cow_shares_frames_read_only_and_no_leaks() {
        let (phys, mut fa) = pool(64);
        // Full baseline: an empty allocator before any AddrSpace exists.
        let alloc_baseline = fa.alloc_count();
        assert_eq!(alloc_baseline, 0);

        let mut parent = AddrSpace::new(&mut fa, &phys).unwrap();
        // Two parent mappings in different PD slots (exercise interior copying).
        let d0 = fa.alloc(&phys).unwrap();
        let d1 = fa.alloc(&phys).unwrap();
        parent
            .map(0x1000, d0, Prot::rw(), false, &mut fa, &phys)
            .unwrap();
        parent
            .map(1u64 << 21, d1, Prot::rwx(), false, &mut fa, &phys)
            .unwrap();

        let before_fork = fa.alloc_count();
        let child = parent.fork_cow(&mut fa, &phys).unwrap();

        // Child interior tables: PML4 + PDPT + PD + 2×PT = 5 new frames; the two
        // data frames are shared (incref, not alloc), so alloc_count grew by 5.
        assert_eq!(
            fa.alloc_count() - before_fork,
            5,
            "only child interior tables allocated"
        );

        // Child translates every parent mapping to the SAME frames.
        assert_eq!(child.translate(0x1000, &phys).unwrap().paddr, d0);
        assert_eq!(child.translate(1u64 << 21, &phys).unwrap().paddr, d1);

        // Both sides are read-only on the shared frames; refcount == 2 each.
        for &v in &[0x1000u64, 1u64 << 21] {
            assert!(
                !parent
                    .translate(v, &phys)
                    .unwrap()
                    .prot
                    .contains(Prot::WRITE),
                "parent RO"
            );
            assert!(
                !child
                    .translate(v, &phys)
                    .unwrap()
                    .prot
                    .contains(Prot::WRITE),
                "child RO"
            );
        }
        // NX/US preserved: the rwx page stays executable on both sides.
        assert!(
            child
                .translate(1u64 << 21, &phys)
                .unwrap()
                .prot
                .contains(Prot::EXEC)
        );
        assert!(
            parent
                .translate(1u64 << 21, &phys)
                .unwrap()
                .prot
                .contains(Prot::EXEC)
        );
        assert_eq!(fa.refcount(d0), 2);
        assert_eq!(fa.refcount(d1), 2);

        // Destroy the child: shared frames drop to 1 (still mapped in parent),
        // the child's interior frames are freed.
        child.destroy(&mut fa, &phys);
        assert_eq!(fa.refcount(d0), 1, "still owned by parent");
        assert_eq!(fa.refcount(d1), 1);
        assert_eq!(fa.alloc_count(), before_fork, "child interior reclaimed");

        // Destroy the parent: everything returns to the pre-test baseline.
        parent.destroy(&mut fa, &phys);
        assert_eq!(fa.refcount(d0), 0, "no owner left");
        assert_eq!(fa.refcount(d1), 0);
        assert_eq!(
            fa.alloc_count(),
            alloc_baseline,
            "no frames leaked anywhere"
        );
    }

    #[test]
    fn fork_cow_out_of_frames_leaves_parent_untouched() {
        // Sized so the parent + its mappings fit, but the child's interior tables
        // cannot all be allocated — fork_cow must fail cleanly.
        let (phys, mut fa) = pool(8);
        let mut parent = AddrSpace::new(&mut fa, &phys).unwrap();
        let d0 = fa.alloc(&phys).unwrap();
        parent
            .map(0x1000, d0, Prot::rw(), false, &mut fa, &phys)
            .unwrap();
        // Drain the pool so fork_cow's first interior alloc (after the child PML4)
        // eventually fails.
        let mut drained = Vec::new();
        while let Some(f) = fa.alloc(&phys) {
            drained.push(f);
        }
        // Free just enough for the child PML4 but not the whole interior chain.
        fa.free(drained.pop().unwrap());

        let live_before = fa.alloc_count();
        let d0_rc_before = fa.refcount(d0);
        assert!(
            parent.fork_cow(&mut fa, &phys).is_none(),
            "exhausted -> None"
        );

        // Parent leaf is unchanged (still writable) and the shared-frame refcount
        // is back to its pre-fork value — no half-applied CoW.
        assert!(
            parent
                .translate(0x1000, &phys)
                .unwrap()
                .prot
                .contains(Prot::WRITE),
            "parent still writable"
        );
        assert_eq!(fa.refcount(d0), d0_rc_before, "increfs undone");
        assert_eq!(
            fa.alloc_count(),
            live_before,
            "no frames leaked by the failed fork"
        );

        for f in drained {
            fa.free(f);
        }
        // `d0`'s single reference was transferred to the parent's page table by
        // `map`; `destroy` releases it, so the test must not free it again.
        parent.destroy(&mut fa, &phys);
    }
}
