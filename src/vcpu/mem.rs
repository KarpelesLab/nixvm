//! Guest memory.
//!
//! One guest process's flat physical/virtual address space at `[base, base +
//! size)`, backed by a single contiguous, host-page-aligned [`Region`]. Pages
//! are tracked (at 4 KiB granularity) as unmapped until `map`ped, each carrying
//! `PROT_*` bits; reads/writes are bounds- and permission-checked so a bad guest
//! pointer surfaces as [`MemError`] instead of corrupting host memory.
//!
//! The backing is one contiguous allocation so a hardware backend (HVF/KVM) can
//! `hv_vm_map` it straight into a guest — the guest then writes through to the
//! very bytes the kernel's syscall copy-in/out reads via [`GuestMemory::read`]/
//! [`GuestMemory::write`]. The software interpreter uses the same store.
//!
//! Copy-on-write: [`GuestMemory::fork`] gives the child its own region and
//! eagerly copies the parent's *resident* (mapped) pages — correct isolation,
//! bounded by the working set. The fault-driven COW seam is preserved in the API
//! ([`GuestMemory::write_trap`], [`GuestMemory::cow_fault`]) so the interpreter
//! and a hardware backend share one contract; lazy/shared COW over this unified
//! region (via a shared frame arena + stage-2/`mprotect` faults) is a later
//! milestone, so today `write_trap` never faults and `cow_fault` never fires.

use super::region::{HOST_PAGE, Region};
use std::sync::atomic::{AtomicU64, Ordering};

/// Guest page size advertised to the guest (`AT_PAGESZ`, `sysconf(_SC_PAGESIZE)`).
pub const PAGE_SIZE: u64 = 4096;

/// Page size as a `usize`, for indexing metadata.
const PS: usize = PAGE_SIZE as usize;

/// Backing-allocation identity. Every fresh region ([`GuestMemory::new`],
/// [`GuestMemory::fork`]) takes a globally-unique generation so a hardware
/// backend can tell — by comparing [`GuestMemory::backing_generation`] each
/// `run()` — whether the host mapping it established is still current or must be
/// re-issued (after a context switch to another process or an `execve` that
/// replaced the space). A monotonic counter, never reused, avoids the pointer
/// ABA problem a raw host-address token would have.
static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);

fn next_generation() -> u64 {
    NEXT_GENERATION.fetch_add(1, Ordering::Relaxed)
}

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
}

#[derive(Debug, PartialEq, Eq)]
pub enum MemError {
    /// Address (range) lies outside `[base, base + size)`.
    OutOfBounds(u64),
    /// Address is within bounds but the page is not mapped.
    Unmapped(u64),
    /// The page is mapped but the access violates its protection.
    Protection { addr: u64, needed: Prot },
    /// Host-side allocation/mapping failure.
    Host(String),
}

impl MemError {
    /// The guest address the fault is attributed to. For page-granular faults
    /// (`Unmapped`, `Protection`) this is the base of the *offending* page, not
    /// the access base — so a store spanning several pages faults at whichever
    /// page actually blocked it, which is what a COW retry loop and a real
    /// hardware fault (FAR) both need to make progress.
    #[must_use]
    pub fn fault_addr(&self) -> u64 {
        match self {
            MemError::OutOfBounds(a) | MemError::Unmapped(a) => *a,
            MemError::Protection { addr, .. } => *addr,
            MemError::Host(_) => 0,
        }
    }
}

/// The guest address space for one process. Not `Clone` — use [`GuestMemory::fork`],
/// which allocates the child its own region.
#[derive(Debug)]
pub struct GuestMemory {
    base: u64,
    size: u64,
    /// The one contiguous, host-page-aligned backing allocation.
    region: Region,
    /// Per-4 KiB-page protection; meaningful only when `mapped[i]`.
    prot: Vec<Prot>,
    mapped: Vec<bool>,
    generation: u64,
}

impl GuestMemory {
    /// Reserve a flat guest region `[base, base + size)`. `base` must be
    /// page-aligned; `size` is rounded up to a whole host page (16 KiB) so the
    /// backing can be mapped by a hardware backend. Mapped-but-unwritten pages
    /// read as zero.
    #[must_use]
    pub fn new(base: u64, size: u64) -> Self {
        assert_eq!(base % PAGE_SIZE, 0, "base must be page-aligned");
        let size = size.max(PAGE_SIZE).next_multiple_of(HOST_PAGE as u64);
        let npages = (size / PAGE_SIZE) as usize;
        Self {
            base,
            size,
            region: Region::new(size as usize),
            prot: vec![Prot::NONE; npages],
            mapped: vec![false; npages],
            generation: next_generation(),
        }
    }

    #[must_use]
    pub fn base(&self) -> u64 {
        self.base
    }
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Raw host pointer to guest `base`, for a hardware backend to `hv_vm_map`.
    /// Stable for this value's lifetime; changes across `fork`/`execve` (a new
    /// value with a new [`GuestMemory::backing_generation`]).
    #[must_use]
    pub fn host_base(&self) -> *mut u8 {
        self.region.as_ptr()
    }

    /// Identity of the current backing allocation (see [`NEXT_GENERATION`]).
    #[must_use]
    pub fn backing_generation(&self) -> u64 {
        self.generation
    }

    /// Host byte offset for a guest address, or `OutOfBounds`.
    fn offset(&self, addr: u64) -> Result<usize, MemError> {
        if addr < self.base || addr >= self.base + self.size {
            return Err(MemError::OutOfBounds(addr));
        }
        Ok((addr - self.base) as usize)
    }

    /// Base guest address of page index `p`.
    fn page_base(&self, p: usize) -> u64 {
        self.base + (p as u64) * PAGE_SIZE
    }

    /// Page indices `[first, last]` covering `[addr, addr + len)`.
    fn page_range(&self, addr: u64, len: usize) -> Result<(usize, usize), MemError> {
        if len == 0 {
            let _ = self.offset(addr)?;
            let p = ((addr - self.base) / PAGE_SIZE) as usize;
            return Ok((p, p));
        }
        let end = addr
            .checked_add(len as u64 - 1)
            .ok_or(MemError::OutOfBounds(addr))?;
        let _ = self.offset(addr)?;
        let _ = self.offset(end)?;
        let first = ((addr - self.base) / PAGE_SIZE) as usize;
        let last = ((end - self.base) / PAGE_SIZE) as usize;
        Ok((first, last))
    }

    /// Map (or remap) the pages covering `[addr, addr + len)` with `prot`. The
    /// range is rounded out to page boundaries and zero-filled — a fresh mapping
    /// reads as zero (`MAP_ANONYMOUS`/`MAP_FIXED` semantics).
    pub fn map(&mut self, addr: u64, len: u64, prot: Prot) -> Result<(), MemError> {
        if len == 0 {
            return Ok(());
        }
        let start = addr - addr % PAGE_SIZE;
        let end = round_up(addr + len, PAGE_SIZE);
        let (first, last) = self.page_range(start, (end - start) as usize)?;
        for p in first..=last {
            self.mapped[p] = true;
            self.prot[p] = prot;
        }
        self.region.fill(first * PS, (last - first + 1) * PS, 0);
        Ok(())
    }

    /// Change protection on already-mapped pages covering `[addr, addr + len)`.
    pub fn protect(&mut self, addr: u64, len: u64, prot: Prot) -> Result<(), MemError> {
        if len == 0 {
            return Ok(());
        }
        let start = addr - addr % PAGE_SIZE;
        let end = round_up(addr + len, PAGE_SIZE);
        let (first, last) = self.page_range(start, (end - start) as usize)?;
        for p in first..=last {
            if !self.mapped[p] {
                return Err(MemError::Unmapped(self.page_base(p)));
            }
            self.prot[p] = prot;
        }
        Ok(())
    }

    /// Unmap the pages covering `[addr, addr + len)`. The bytes are left intact
    /// but inaccessible; a later `map` zero-fills before granting access.
    pub fn unmap(&mut self, addr: u64, len: u64) -> Result<(), MemError> {
        if len == 0 {
            return Ok(());
        }
        let start = addr - addr % PAGE_SIZE;
        let end = round_up(addr + len, PAGE_SIZE);
        let (first, last) = self.page_range(start, (end - start) as usize)?;
        for p in first..=last {
            self.mapped[p] = false;
            self.prot[p] = Prot::NONE;
        }
        Ok(())
    }

    /// Verify `[addr, addr + len)` is mapped and every page grants `need`,
    /// returning the byte offset of `addr` within the region.
    fn check(&self, addr: u64, len: usize, need: Prot) -> Result<usize, MemError> {
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
        Ok((addr - self.base) as usize)
    }

    /// Read `buf.len()` bytes from guest `addr` (requires `READ`). Unwritten
    /// pages read as zero.
    pub fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemError> {
        let off = self.check(addr, buf.len(), Prot::READ)?;
        self.region.read(off, buf);
        Ok(())
    }

    /// Read `len` bytes into a fresh `Vec` (requires `READ`).
    pub fn read_vec(&self, addr: u64, len: usize) -> Result<Vec<u8>, MemError> {
        let mut v = vec![0u8; len];
        self.read(addr, &mut v)?;
        Ok(v)
    }

    /// Write `buf` to guest `addr` (requires `WRITE`). Entry point for
    /// kernel-side copy-out (syscall results).
    pub fn write(&mut self, addr: u64, buf: &[u8]) -> Result<(), MemError> {
        let off = self.check(addr, buf.len(), Prot::WRITE)?;
        self.region.write(off, buf);
        Ok(())
    }

    /// Write `buf` to guest `addr` on behalf of an executing guest store. Today
    /// identical to [`GuestMemory::write`]; the name marks the fault-driven COW
    /// seam (a future shared-page milestone makes this fault on a copy-on-write
    /// page, reporting the precise page so the vcpu can be resumed after
    /// [`GuestMemory::cow_fault`] privatizes it).
    pub fn write_trap(&mut self, addr: u64, buf: &[u8]) -> Result<(), MemError> {
        self.write(addr, buf)
    }

    /// Resolve a copy-on-write fault at `addr`. With eager-copy `fork` there is
    /// no shared page to privatize, so this always returns `false` — the kernel
    /// then treats the fault as a genuine `SIGSEGV`. The seam exists for the
    /// later shared-page COW milestone.
    #[allow(clippy::unused_self)]
    pub fn cow_fault(&mut self, _addr: u64, _write: bool) -> bool {
        false
    }

    /// Fork this address space for a new process: give the child its own region
    /// and eagerly copy every *resident* (mapped) page — correct isolation,
    /// bounded by the working set rather than the whole reservation.
    #[must_use]
    pub fn fork(&self) -> Self {
        let mut region = Region::new(self.size as usize);
        // Copy only mapped pages; unmapped ones stay zero in the fresh region.
        let mut p = 0;
        while p < self.mapped.len() {
            if !self.mapped[p] {
                p += 1;
                continue;
            }
            // Extend the run of contiguous mapped pages and copy it in one shot.
            let start = p;
            while p < self.mapped.len() && self.mapped[p] {
                p += 1;
            }
            region.copy_from(&self.region, start * PS, (p - start) * PS);
        }
        Self {
            base: self.base,
            size: self.size,
            region,
            prot: self.prot.clone(),
            mapped: self.mapped.clone(),
            generation: next_generation(),
        }
    }

    /// Write `buf` to guest `addr`, bypassing protection but still requiring the
    /// pages be mapped and in bounds. For host-side initialization (loading ELF
    /// segments into read-only pages, populating a file-backed `mmap`).
    pub fn write_init(&mut self, addr: u64, buf: &[u8]) -> Result<(), MemError> {
        let (first, last) = self.page_range(addr, buf.len())?;
        for p in first..=last {
            if !self.mapped[p] {
                return Err(MemError::Unmapped(self.page_base(p)));
            }
        }
        self.region.write((addr - self.base) as usize, buf);
        Ok(())
    }

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

const fn round_up(v: u64, align: u64) -> u64 {
    v.div_ceil(align) * align
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem() -> GuestMemory {
        // 64 KiB region at 0x1_0000 (16 KiB-aligned base).
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
        assert!(matches!(
            m.read_u32(0x9_0000),
            Err(MemError::OutOfBounds(_))
        ));
        assert!(matches!(
            m.read_u32(0x0_0000),
            Err(MemError::OutOfBounds(_))
        ));
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
        // Host-side init bypasses protection.
        m.write_init(0x1_0000, &[1, 2, 3]).unwrap();
        assert_eq!(m.read_vec(0x1_0000, 3).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn cross_page_access_checks_every_page() {
        let mut m = mem();
        // Map only the first of two pages a 16-byte read at the boundary spans.
        m.map(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        let boundary = 0x1_0000 + PAGE_SIZE - 8;
        assert!(matches!(
            m.read_u64(boundary + 4),
            Err(MemError::Unmapped(_))
        ));
        // Map the second page too; now it succeeds across the boundary.
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
        // Write only into the second page near the boundary; read straddles the
        // zero (unwritten) tail of page 1 and the written head of page 2.
        m.write(0x1_0000 + PAGE_SIZE, &[0xEE]).unwrap();
        let boundary = 0x1_0000 + PAGE_SIZE - 2;
        assert_eq!(
            m.read_vec(boundary, 4).unwrap(),
            vec![0x00, 0x00, 0xEE, 0x00]
        );
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
        // Child starts as a copy of the parent's resident bytes...
        assert_eq!(child.read_vec(0x1_0000, 1).unwrap(), vec![0xAA]);
        // ...then each side's writes are independent.
        parent.write(0x1_0000, &[0xBB]).unwrap();
        child.write(0x1_0000, &[0xCC]).unwrap();
        assert_eq!(parent.read_vec(0x1_0000, 1).unwrap(), vec![0xBB]);
        assert_eq!(child.read_vec(0x1_0000, 1).unwrap(), vec![0xCC]);
    }

    #[test]
    fn fork_copies_only_mapped_pages() {
        let mut parent = mem();
        // Map two non-adjacent pages, leaving a gap unmapped.
        parent.map(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        parent
            .map(0x1_0000 + 2 * PAGE_SIZE, PAGE_SIZE, Prot::rw())
            .unwrap();
        parent.write(0x1_0000, &[1]).unwrap();
        parent.write(0x1_0000 + 2 * PAGE_SIZE, &[2]).unwrap();
        let child = parent.fork();
        assert_eq!(child.read_vec(0x1_0000, 1).unwrap(), vec![1]);
        assert_eq!(
            child.read_vec(0x1_0000 + 2 * PAGE_SIZE, 1).unwrap(),
            vec![2]
        );
        // The gap is unmapped in both.
        assert!(child.read_vec(0x1_0000 + PAGE_SIZE, 1).is_err());
    }

    #[test]
    fn fork_and_new_take_distinct_backing_generations() {
        let a = mem();
        let b = mem();
        let c = a.fork();
        assert_ne!(a.backing_generation(), b.backing_generation());
        assert_ne!(a.backing_generation(), c.backing_generation());
        assert_ne!(b.backing_generation(), c.backing_generation());
        assert!(!a.host_base().is_null());
    }

    #[test]
    fn write_trap_matches_write_and_cow_fault_never_fires() {
        let mut m = mem();
        m.map(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        m.write_trap(0x1_0000, &[0x5A]).unwrap();
        assert_eq!(m.read_vec(0x1_0000, 1).unwrap(), vec![0x5A]);
        // A write to read-only memory faults through write_trap...
        m.protect(0x1_0000, PAGE_SIZE, Prot::READ).unwrap();
        assert!(m.write_trap(0x1_0000, &[0]).is_err());
        // ...and is a genuine fault (no COW page to privatize).
        assert!(!m.cow_fault(0x1_0000, true));
    }
}
