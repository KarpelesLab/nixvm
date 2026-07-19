//! Guest page tables that enforce per-page protection.
//!
//! The rest of the KVM backend maps guest RAM as one flat slot; on its own that
//! would need page tables that mark every page present/writable/executable — the
//! uniformly-RWX identity map the control block used to build. That is both
//! wrong (real hardware faults on a write to code or a jump into the stack) and
//! a sandbox hole (injected data is executable).
//!
//! This module builds real 4 KiB page tables from [`GuestMemory`]'s protection
//! map: a page is present only if mapped, writable only with `PROT_WRITE`, and
//! carries the `NX` bit unless it has `PROT_EXEC`. A violating access raises a
//! guest `#PF`; with no IDT that becomes a triple fault → `KVM_EXIT_SHUTDOWN`,
//! which the vcpu already turns into a `MemFault` (from `cr2`) — so the kernel's
//! fault path (COW / signal / kill) runs, exactly as on real hardware.
//!
//! The tables live in their own guest-physical region (a dedicated memslot) so
//! page-table walks — which the CPU does by physical address — reach them. The
//! skeleton (PML4/PDPT/PDs, the control-block mapping) is built once; only the
//! PT leaf entries change, rebuilt in full on an address-space switch and
//! updated incrementally (per `GuestMemory::drain_pt_dirty`) on `mmap`/
//! `mprotect`/`munmap`.

use super::vm::{CTRL_GPA, IDENTITY_TOP};
use crate::vcpu::mem::{GuestMemory, PAGE_SIZE, Prot};
use crate::vcpu::region::Region;

/// Guest-physical base of the page-table region (its own memslot), just above
/// the identity window so it never collides with guest RAM or the control block.
pub const PT_AREA_GPA: u64 = IDENTITY_TOP;

const P: u64 = 1 << 0; // present
const RW: u64 = 1 << 1; // writable
const US: u64 = 1 << 2; // user-accessible
const PS: u64 = 1 << 7; // 2 MiB leaf (in a PD entry)
const NX: u64 = 1 << 63; // no-execute (needs EFER.NXE)
/// Accessed/Dirty bits: the CPU sets these in a leaf PTE as a side effect of a
/// walk, so a leaf read back from the region may carry them even though `sync`
/// never writes them. They are masked out before comparing an entry to its
/// desired value (see [`PageTables::leaf_is_stale`]).
const AD: u64 = (1 << 5) | (1 << 6);

const GIB: u64 = 1 << 30;
const MIB2: u64 = 2 << 20;

/// The PTE flags for a guest page with protection `prot` (present, user, plus
/// write/NX from the protection bits). The physical frame is OR-ed in by the
/// caller.
fn leaf_flags(prot: Prot) -> u64 {
    // Debug: NIXVM_NOWX forces every mapped page RWX (no W^X), to test whether
    // the protection page tables are corrupting/losing writes.
    if std::env::var_os("NIXVM_NOWX").is_some() {
        return P | RW | US;
    }
    let mut f = P | US;
    if prot.contains(Prot::WRITE) {
        f |= RW;
    }
    if !prot.contains(Prot::EXEC) {
        f |= NX;
    }
    f
}

/// The guest page tables backing one address space.
#[derive(Debug)]
pub struct PageTables {
    /// PML4, PDPT, per-GiB PDs, then the PT leaf pages — all in one region.
    region: Region,
    /// `[guest_base, guest_end)` the leaves cover.
    guest_base: u64,
    guest_end: u64,
    /// Number of PT leaf pages (one per 2 MiB of `[0, guest_end)`).
    n_pt: usize,
    /// Byte offset of the first PT leaf page within `region`.
    pt_off: usize,
    /// The (backing, layout) generations the leaves currently reflect.
    synced_backing: Option<u64>,
    synced_layout: u64,
}

impl PageTables {
    /// Build the fixed skeleton for a guest of `size` bytes based at `base`:
    /// PML4 → PDPT → one PD per covered GiB → a PT page per 2 MiB, plus a 2 MiB
    /// leaf mapping the control block. Leaves start all-zero (not present).
    pub fn new(base: u64, size: u64) -> Self {
        let guest_end = base + size;
        let n_pt = (guest_end.div_ceil(MIB2)) as usize; // PT pages for [0, guest_end)
        let n_pd_guest = (guest_end.div_ceil(GIB)) as usize; // guest PDs (GiB 0..)

        // Region layout: PML4(1) PDPT(1) guest-PDs(n_pd_guest) control-PD(1) PTs(n_pt).
        let pml4_off = 0usize;
        let pdpt_off = 0x1000;
        let pd0_off = 0x2000; // first guest PD
        let ctrl_pd_off = pd0_off + n_pd_guest * 0x1000;
        let pt_off = ctrl_pd_off + 0x1000;
        let total_pages = 2 + n_pd_guest + 1 + n_pt;
        let mut region = Region::new(total_pages * PAGE_SIZE as usize);

        // Every table entry is an aligned 64-bit store (see `Region::write_u64_atomic`):
        // uniform with the leaf updates below, and tear-free should this skeleton ever
        // be touched while a walker runs (it is not, today — the skeleton is built
        // before the memslot is registered).
        let put = |region: &mut Region, off: usize, v: u64| region.write_u64_atomic(off, v);

        // PML4[0] → PDPT.
        put(&mut region, pml4_off, (PT_AREA_GPA + pdpt_off as u64) | P | RW | US);

        // PDPT[g] → guest PD g, for each covered GiB.
        for g in 0..n_pd_guest {
            let pd_gpa = PT_AREA_GPA + (pd0_off + g * 0x1000) as u64;
            put(&mut region, pdpt_off + g * 8, pd_gpa | P | RW | US);
            // Each guest PD entry → its PT page (for 2 MiB slots within [0, guest_end)).
            for i in 0..512 {
                let slot = g * 512 + i; // global 2 MiB index
                if slot < n_pt {
                    let pt_gpa = PT_AREA_GPA + (pt_off + slot * 0x1000) as u64;
                    put(&mut region, pd0_off + g * 0x1000 + i * 8, pt_gpa | P | RW | US);
                }
            }
        }

        // The control block (GDT + syscall trampoline) must stay mapped, at its
        // own GiB, as one 2 MiB identity leaf.
        let ctrl_gib = (CTRL_GPA / GIB) as usize;
        let ctrl_slot = ((CTRL_GPA % GIB) / MIB2) as usize;
        let ctrl_pd_gpa = PT_AREA_GPA + ctrl_pd_off as u64;
        put(&mut region, pdpt_off + ctrl_gib * 8, ctrl_pd_gpa | P | RW | US);
        put(&mut region, ctrl_pd_off + ctrl_slot * 8, CTRL_GPA | P | RW | US | PS);

        Self {
            region,
            guest_base: base,
            guest_end,
            n_pt,
            pt_off,
            synced_backing: None,
            synced_layout: 0,
        }
    }

    /// Guest-physical address of the region, for the memslot.
    #[allow(clippy::unused_self)] // fixed placement; a method for call-site symmetry with size/host_base
    pub fn gpa(&self) -> u64 {
        PT_AREA_GPA
    }
    /// Size of the region.
    pub fn size(&self) -> u64 {
        self.region.len() as u64
    }
    /// Host pointer to the region (for `KVM_SET_USER_MEMORY_REGION`).
    pub fn host_base(&self) -> *mut u8 {
        self.region.as_ptr()
    }

    /// Set the PTE for the single guest page at `gpa` from its protection (or
    /// clear it if unmapped). No-op for addresses outside the covered range.
    fn set_page(&mut self, gpa: u64, prot: Option<Prot>) {
        if gpa < self.guest_base || gpa >= self.guest_end {
            return;
        }
        let slot = (gpa / MIB2) as usize; // which PT page
        let idx = ((gpa % MIB2) / PAGE_SIZE) as usize; // entry within it
        if slot >= self.n_pt {
            return;
        }
        let off = self.pt_off + slot * 0x1000 + idx * 8;
        let entry = match prot {
            Some(pr) => gpa | leaf_flags(pr),
            None => 0, // not present
        };
        // A single aligned 64-bit store: a sibling vcpu's hardware page walker,
        // running lockless under the SMP scheduler, sees the whole old or whole
        // new PTE, never a torn value.
        self.region.write_u64_atomic(off, entry);
    }

    /// Whether the shadow leaf PTE for the page containing `gpa` differs from what
    /// `mem`'s protection map currently wants — i.e. a [`PageTables::sync`] would
    /// change it. The SMP scheduler uses this to tell a *stale-shadow* fault (a
    /// sibling changed the mapping while this vcpu ran with page tables synced at
    /// its last dispatch — retry after reconcile) from a *genuine* protection or
    /// unmapped fault (the shadow already matches `mem`, so the access really is
    /// illegal). Returns `false` for an address outside the covered range: such a
    /// fault is always genuine.
    ///
    /// The comparison masks the CPU-managed Accessed/Dirty bits, which a walk sets
    /// in the leaf without `sync` ever writing them — otherwise a genuine
    /// read-only-write fault would look "stale" forever and retry endlessly.
    #[must_use]
    pub fn leaf_is_stale(&self, mem: &GuestMemory, gpa: u64) -> bool {
        let page = gpa & !(PAGE_SIZE - 1);
        if page < self.guest_base || page >= self.guest_end {
            return false;
        }
        let slot = (page / MIB2) as usize;
        let idx = ((page % MIB2) / PAGE_SIZE) as usize;
        if slot >= self.n_pt {
            return false;
        }
        let off = self.pt_off + slot * 0x1000 + idx * 8;
        let mut b = [0u8; 8];
        self.region.read(off, &mut b);
        let actual = u64::from_le_bytes(b) & !AD;
        let desired = match mem.page_prot(page) {
            Some(pr) => page | leaf_flags(pr),
            None => 0,
        };
        actual != desired
    }

    /// Rebuild every PT leaf from `mem`'s protection map (an address-space
    /// switch): clear all leaves, then set the mapped pages.
    fn rebuild_all(&mut self, mem: &GuestMemory) {
        self.region.fill(self.pt_off, self.n_pt * PAGE_SIZE as usize, 0);
        let mut gpa = self.guest_base;
        while gpa < self.guest_end {
            self.set_page(gpa, mem.page_prot(gpa));
            gpa += PAGE_SIZE;
        }
    }

    /// Update the leaves for the pages in `[first, last]` (inclusive, page
    /// addresses) from `mem` — the incremental path after `mmap`/`mprotect`.
    fn update_range(&mut self, mem: &GuestMemory, first: u64, last: u64) {
        let mut gpa = first;
        loop {
            self.set_page(gpa, mem.page_prot(gpa));
            if gpa >= last {
                break;
            }
            gpa += PAGE_SIZE;
        }
    }

    /// Bring the leaves into sync with `mem`. Returns whether the tables changed
    /// (the caller need not re-issue the memslot unless the region moved, which
    /// it never does). A backing-generation change means a different address
    /// space → full rebuild; otherwise a layout change applies the dirty ranges.
    pub fn sync(&mut self, mem: &mut GuestMemory) {
        let backing = mem.backing_generation();
        let layout = mem.layout_generation();
        if self.synced_backing != Some(backing) {
            self.rebuild_all(mem);
            let _ = mem.drain_pt_dirty(); // consumed by the full rebuild
            self.synced_backing = Some(backing);
            self.synced_layout = layout;
        } else if self.synced_layout != layout {
            for (first, last) in mem.drain_pt_dirty() {
                self.update_range(mem, first, last);
            }
            self.synced_layout = layout;
        }
    }
}
