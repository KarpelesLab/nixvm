//! Guest memory.
//!
//! A host-backed region representing one guest process's physical/virtual
//! address space at `[base, base + size)`. Pages are tracked as unmapped until
//! `map`ped, each carrying `PROT_*` bits; reads/writes are bounds- and
//! permission-checked so a bad guest pointer surfaces as [`MemError`] instead of
//! corrupting host memory.
//!
//! Storage is **page-granular and copy-on-write**: each 4 KiB page is an
//! `Option<Arc<[u8; PAGE_SIZE]>>` (`None` = mapped-but-untouched, read as zero,
//! allocated on first write). [`GuestMemory::fork`] shares every page by cloning
//! the `Arc` table (no byte copy) and marks both parent and child copy-on-write;
//! the first mutation of a shared page privatizes just that page via
//! `Arc::make_mut`. Guest stores go through [`GuestMemory::write_trap`], which
//! *faults* (reporting the precise offending page) on a COW page so the vcpu can
//! be resumed by the kernel's page-fault handler after [`GuestMemory::cow_fault`]
//! privatizes it — the same seam a future hardware (HVF) backend will drive from
//! a real fault. Kernel-side copy-out (syscall results) uses [`GuestMemory::write`],
//! which resolves COW inline since there is no instruction to retry.

use std::sync::Arc;

/// Guest page size advertised to the guest (`AT_PAGESZ`, `sysconf(_SC_PAGESIZE)`).
pub const PAGE_SIZE: u64 = 4096;

/// Page size as a `usize`, for indexing host-side page buffers.
const PS: usize = PAGE_SIZE as usize;

/// One physical page of guest RAM, reference-counted so `fork` can share it
/// between address spaces until the first write privatizes it (`Arc::make_mut`).
type Page = Arc<[u8; PS]>;

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
    /// page actually blocked it, which is what the COW retry loop and a real
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
/// which additionally establishes the copy-on-write sharing invariant.
pub struct GuestMemory {
    base: u64,
    size: u64,
    /// Backing page for each page slot; `None` = never written (reads as zero).
    /// A page whose `cow` flag is clear is uniquely owned and free to mutate.
    pages: Vec<Option<Page>>,
    /// Per-page protection; meaningful only when `mapped[i]`.
    prot: Vec<Prot>,
    mapped: Vec<bool>,
    /// `cow[i] == true` ⇒ `pages[i]`'s `Arc` may be shared with another space;
    /// it must be privatized (`Arc::make_mut`) before any mutation. Set for
    /// shared pages by [`GuestMemory::fork`]; cleared the moment a page is
    /// privatized. `cow[i] == false` ⇒ `pages[i]` is uniquely owned.
    cow: Vec<bool>,
}

impl std::fmt::Debug for GuestMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mapped = self.mapped.iter().filter(|m| **m).count();
        let resident = self.pages.iter().filter(|p| p.is_some()).count();
        let shared = self.cow.iter().filter(|c| **c).count();
        f.debug_struct("GuestMemory")
            .field("base", &format_args!("{:#x}", self.base))
            .field("size", &self.size)
            .field("mapped_pages", &mapped)
            .field("resident_pages", &resident)
            .field("cow_pages", &shared)
            .field("total_pages", &self.mapped.len())
            .finish_non_exhaustive()
    }
}

impl GuestMemory {
    /// Reserve a flat guest region `[base, base + size)`. `base` and `size` must
    /// be page-aligned. All pages start unmapped; no page RAM is allocated until
    /// first write (the page table itself is `size / PAGE_SIZE` entries).
    #[must_use]
    pub fn new(base: u64, size: u64) -> Self {
        assert_eq!(base % PAGE_SIZE, 0, "base must be page-aligned");
        assert_eq!(size % PAGE_SIZE, 0, "size must be page-aligned");
        let npages = (size / PAGE_SIZE) as usize;
        Self {
            base,
            size,
            pages: (0..npages).map(|_| None).collect(),
            prot: vec![Prot::NONE; npages],
            mapped: vec![false; npages],
            cow: vec![false; npages],
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
    /// range is rounded out to page boundaries. Mapped pages read as zero until
    /// written (a fresh mapping drops any prior backing and clears COW sharing —
    /// `MAP_ANONYMOUS`/`MAP_FIXED` semantics).
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
            // Fresh mapping: zero-fill (drop backing) and drop any COW share so
            // a later write can't mutate a page still aliased by a fork sibling.
            self.pages[p] = None;
            self.cow[p] = false;
        }
        Ok(())
    }

    /// Change protection on already-mapped pages covering `[addr, addr + len)`.
    /// Contents and COW sharing are untouched (`mprotect` semantics).
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

    /// Unmap the pages covering `[addr, addr + len)`, releasing their backing
    /// (and any COW share).
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
            self.pages[p] = None;
            self.cow[p] = false;
        }
        Ok(())
    }

    /// Verify `[addr, addr + len)` is mapped and every page grants `need`,
    /// returning the covered page range. Does **not** consult `cow` — COW is a
    /// property of the writers, not of access validity.
    fn check(&self, addr: u64, len: usize, need: Prot) -> Result<(usize, usize), MemError> {
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
        Ok((first, last))
    }

    /// The per-page span of an access `[addr, addr+len)` intersected with page
    /// `p`: `(offset_within_page, len_within_page)`.
    fn page_span(&self, addr: u64, len: usize, p: usize) -> (usize, usize) {
        let pstart = self.page_base(p);
        let lo = addr.max(pstart) - pstart;
        let hi = (addr + len as u64).min(pstart + PAGE_SIZE) - pstart;
        (lo as usize, (hi - lo) as usize)
    }

    /// Ensure page `p` is allocated and uniquely owned, returning a mutable
    /// handle. Allocates a zero page if `None`; privatizes via `Arc::make_mut`
    /// if the `Arc` is shared. Clears the page's COW flag.
    fn ensure_owned_mut(&mut self, p: usize) -> &mut [u8; PS] {
        self.cow[p] = false;
        let page = self.pages[p].get_or_insert_with(|| Arc::new([0u8; PS]));
        Arc::make_mut(page)
    }

    /// Read `buf.len()` bytes from guest `addr` (requires `READ`). Unwritten
    /// (`None`) pages read as zero.
    pub fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemError> {
        let (first, last) = self.check(addr, buf.len(), Prot::READ)?;
        let mut done = 0usize;
        for p in first..=last {
            let (lo, n) = self.page_span(addr, buf.len(), p);
            match &self.pages[p] {
                Some(pg) => buf[done..done + n].copy_from_slice(&pg[lo..lo + n]),
                None => buf[done..done + n].fill(0),
            }
            done += n;
        }
        Ok(())
    }

    /// Read `len` bytes into a fresh `Vec` (requires `READ`).
    pub fn read_vec(&self, addr: u64, len: usize) -> Result<Vec<u8>, MemError> {
        let mut v = vec![0u8; len];
        self.read(addr, &mut v)?;
        Ok(v)
    }

    /// Copy `buf` across the covered pages, privatizing/allocating each as
    /// needed. Callers must have validated the range (mapped + `WRITE`) first.
    fn store(&mut self, addr: u64, buf: &[u8], first: usize, last: usize) {
        let mut done = 0usize;
        for p in first..=last {
            let (lo, n) = self.page_span(addr, buf.len(), p);
            if n == 0 {
                // Nothing lands on this page — don't allocate/privatize it just
                // to copy zero bytes (keeps lazy `None` pages lazy).
                continue;
            }
            let page = self.ensure_owned_mut(p);
            page[lo..lo + n].copy_from_slice(&buf[done..done + n]);
            done += n;
        }
    }

    /// Write `buf` to guest `addr` (requires `WRITE`), **resolving copy-on-write
    /// inline**: any shared page in the range is privatized before the store.
    /// This is the entry point for kernel-side copy-out (syscall results), where
    /// there is no faulting instruction to retry.
    pub fn write(&mut self, addr: u64, buf: &[u8]) -> Result<(), MemError> {
        let (first, last) = self.check(addr, buf.len(), Prot::WRITE)?;
        self.store(addr, buf, first, last);
        Ok(())
    }

    /// Write `buf` to guest `addr` on behalf of an executing guest store: like
    /// [`GuestMemory::write`], but a copy-on-write page **faults** instead of
    /// being resolved inline, reporting the *first* COW page in the range. The
    /// vcpu surfaces this as a memory fault; the kernel privatizes the page via
    /// [`GuestMemory::cow_fault`] and re-runs the instruction. Reporting the
    /// precise page (not the access base) guarantees a page-spanning store makes
    /// progress — each retry faults on the next still-shared page until none
    /// remain.
    pub fn write_trap(&mut self, addr: u64, buf: &[u8]) -> Result<(), MemError> {
        let (first, last) = self.check(addr, buf.len(), Prot::WRITE)?;
        for p in first..=last {
            if self.cow[p] {
                return Err(MemError::Protection {
                    addr: self.page_base(p),
                    needed: Prot::WRITE,
                });
            }
        }
        self.store(addr, buf, first, last);
        Ok(())
    }

    /// Resolve a copy-on-write fault at `addr`: if it names a mapped, writable,
    /// COW-shared page, privatize that page and return `true` (the faulting
    /// instruction should be retried). Returns `false` for a genuine fault —
    /// read, unmapped, read-only, or already-private page — which the caller
    /// turns into `SIGSEGV`.
    pub fn cow_fault(&mut self, addr: u64, write: bool) -> bool {
        if !write {
            return false;
        }
        let Ok(off) = self.offset(addr) else {
            return false;
        };
        let p = off / PS;
        if self.mapped[p] && self.prot[p].contains(Prot::WRITE) && self.cow[p] {
            self.ensure_owned_mut(p);
            true
        } else {
            false
        }
    }

    /// Fork this address space for a new process: share every resident page by
    /// cloning the `Arc` table (no byte copy) and mark both this space and the
    /// returned child copy-on-write, so the first write on *either* side
    /// privatizes only the touched page. Read-only shared pages (e.g. text) are
    /// marked COW too, harmlessly: a write to them fails the protection check
    /// before COW is ever consulted, yielding a real fault.
    #[must_use]
    pub fn fork(&mut self) -> Self {
        for p in 0..self.pages.len() {
            if self.pages[p].is_some() {
                self.cow[p] = true;
            }
        }
        Self {
            base: self.base,
            size: self.size,
            pages: self.pages.clone(),
            prot: self.prot.clone(),
            mapped: self.mapped.clone(),
            cow: self.cow.clone(),
        }
    }

    /// Write `buf` to guest `addr`, bypassing protection but still requiring the
    /// pages be mapped and in bounds, and still privatizing COW pages (so a
    /// prot-bypass host write can never mutate a page aliased by a fork sibling).
    /// For host-side initialization (loading ELF segments into read-only pages,
    /// populating a file-backed `mmap`).
    pub fn write_init(&mut self, addr: u64, buf: &[u8]) -> Result<(), MemError> {
        let (first, last) = self.page_range(addr, buf.len())?;
        for p in first..=last {
            if !self.mapped[p] {
                return Err(MemError::Unmapped(self.page_base(p)));
            }
        }
        self.store(addr, buf, first, last);
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

    /// Write a little-endian `u64` (requires `WRITE`). Resolves COW inline (host
    /// helper; not a guest store).
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
        // 64 KiB region at 0x1_0000.
        GuestMemory::new(0x1_0000, 16 * PAGE_SIZE)
    }

    /// Strong-count of the `Arc` backing the page containing `addr` (0 if the
    /// page is unallocated). Test-only window into COW sharing.
    fn page_refs(m: &GuestMemory, addr: u64) -> usize {
        let p = ((addr - m.base) / PAGE_SIZE) as usize;
        m.pages[p].as_ref().map_or(0, Arc::strong_count)
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
    fn lazy_none_page_reads_zero_and_allocates_on_write() {
        let mut m = mem();
        m.map(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        // Mapped but never written: reads zero, no page allocated.
        assert_eq!(m.read_vec(0x1_0000, 8).unwrap(), vec![0u8; 8]);
        assert_eq!(page_refs(&m, 0x1_0000), 0, "read must not allocate");
        m.write(0x1_0000, &[1]).unwrap();
        assert_eq!(page_refs(&m, 0x1_0000), 1, "write allocates a private page");
    }

    #[test]
    fn fork_shares_pages_until_written() {
        let mut parent = mem();
        parent.map(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        parent.write(0x1_0000, &[0xAA]).unwrap();
        assert_eq!(page_refs(&parent, 0x1_0000), 1);
        let child = parent.fork();
        // Both sides now share the one Arc (refcount 2), no byte copy.
        assert_eq!(page_refs(&parent, 0x1_0000), 2);
        assert_eq!(page_refs(&child, 0x1_0000), 2);
    }

    #[test]
    fn fork_then_parent_write_leaves_child_untouched() {
        let mut parent = mem();
        parent.map(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        parent.write(0x1_0000, &[0xAA]).unwrap();
        let child = parent.fork();
        // A guest store must fault while the page is COW-shared...
        assert!(parent.write_trap(0x1_0000, &[0xBB]).is_err());
        // ...the fault handler privatizes it...
        assert!(parent.cow_fault(0x1_0000, true));
        // ...and the retried store now lands.
        parent.write_trap(0x1_0000, &[0xBB]).unwrap();
        assert_eq!(parent.read_vec(0x1_0000, 1).unwrap(), vec![0xBB]);
        assert_eq!(child.read_vec(0x1_0000, 1).unwrap(), vec![0xAA]);
        // Parent now owns a private copy; child holds the original alone.
        assert_eq!(page_refs(&parent, 0x1_0000), 1);
        assert_eq!(page_refs(&child, 0x1_0000), 1);
    }

    #[test]
    fn fork_then_child_write_leaves_parent_untouched() {
        let mut parent = mem();
        parent.map(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        parent.write(0x1_0000, &[0xAA]).unwrap();
        let mut child = parent.fork();
        // Resolving write (as kernel copy-out would) privatizes inline.
        child.write(0x1_0000, &[0xCC]).unwrap();
        assert_eq!(child.read_vec(0x1_0000, 1).unwrap(), vec![0xCC]);
        assert_eq!(parent.read_vec(0x1_0000, 1).unwrap(), vec![0xAA]);
    }

    #[test]
    fn fork_three_way_isolation() {
        let mut a = mem();
        a.map(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        a.write(0x1_0000, &[1]).unwrap();
        let mut b = a.fork();
        let mut c = a.fork();
        assert_eq!(page_refs(&a, 0x1_0000), 3);
        a.write(0x1_0000, &[10]).unwrap();
        b.write(0x1_0000, &[20]).unwrap();
        c.write(0x1_0000, &[30]).unwrap();
        assert_eq!(a.read_vec(0x1_0000, 1).unwrap(), vec![10]);
        assert_eq!(b.read_vec(0x1_0000, 1).unwrap(), vec![20]);
        assert_eq!(c.read_vec(0x1_0000, 1).unwrap(), vec![30]);
    }

    #[test]
    fn cow_write_spanning_page_boundary_isolates_both_pages() {
        let mut parent = mem();
        parent.map(0x1_0000, 2 * PAGE_SIZE, Prot::rw()).unwrap();
        parent.write(0x1_0000, &[0x11]).unwrap();
        parent.write(0x1_0000 + PAGE_SIZE, &[0x22]).unwrap();
        let child = parent.fork();
        // 8-byte store straddling the page boundary: both pages are COW.
        let boundary = 0x1_0000 + PAGE_SIZE - 4;
        // First store faults on the FIRST cow page (the access base's page)...
        let e = parent.write_trap(boundary, &[0xFF; 8]).unwrap_err();
        assert_eq!(e.fault_addr(), 0x1_0000);
        assert!(parent.cow_fault(e.fault_addr(), true));
        // ...retry now faults on the SECOND page (precise addr => progress)...
        let e = parent.write_trap(boundary, &[0xFF; 8]).unwrap_err();
        assert_eq!(e.fault_addr(), 0x1_0000 + PAGE_SIZE);
        assert!(parent.cow_fault(e.fault_addr(), true));
        // ...and the third attempt lands.
        parent.write_trap(boundary, &[0xFF; 8]).unwrap();
        assert_eq!(parent.read_vec(boundary, 8).unwrap(), vec![0xFF; 8]);
        // Child sees neither page mutated.
        assert_eq!(child.read_vec(0x1_0000, 1).unwrap(), vec![0x11]);
        assert_eq!(child.read_vec(0x1_0000 + PAGE_SIZE, 1).unwrap(), vec![0x22]);
    }

    #[test]
    fn cow_fault_on_readonly_page_is_a_genuine_fault() {
        let mut parent = mem();
        parent.map(0x1_0000, PAGE_SIZE, Prot::rx()).unwrap();
        parent.write_init(0x1_0000, &[0x90]).unwrap();
        let _child = parent.fork();
        // A write to shared *read-only* memory fails the prot check (not COW)...
        assert!(matches!(
            parent.write_trap(0x1_0000, &[0x00]),
            Err(MemError::Protection { .. })
        ));
        // ...and the handler refuses it => SIGSEGV, not a silent copy.
        assert!(!parent.cow_fault(0x1_0000, true));
    }

    #[test]
    fn cow_fault_rejects_reads_and_unmapped() {
        let mut m = mem();
        m.map(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        m.write(0x1_0000, &[1]).unwrap();
        let _c = m.fork();
        assert!(!m.cow_fault(0x1_0000, false), "reads never COW");
        assert!(!m.cow_fault(0x9_0000, true), "out-of-bounds never COW");
    }

    #[test]
    fn write_init_on_shared_page_privatizes() {
        let mut parent = mem();
        parent.map(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        parent.write(0x1_0000, &[0xAA]).unwrap();
        let child = parent.fork();
        parent.write_init(0x1_0000, &[0xBB]).unwrap();
        assert_eq!(parent.read_vec(0x1_0000, 1).unwrap(), vec![0xBB]);
        assert_eq!(child.read_vec(0x1_0000, 1).unwrap(), vec![0xAA]);
    }

    #[test]
    fn unmap_drops_backing_and_reads_zero_after_remap() {
        let mut m = mem();
        m.map(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        m.write(0x1_0000, &[0x77]).unwrap();
        m.unmap(0x1_0000, PAGE_SIZE).unwrap();
        assert_eq!(page_refs(&m, 0x1_0000), 0, "unmap releases the page");
        m.map(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        assert_eq!(m.read_vec(0x1_0000, 1).unwrap(), vec![0], "remap is zeroed");
    }

    #[test]
    fn zero_length_write_does_not_allocate_or_privatize() {
        let mut parent = mem();
        parent.map(0x1_0000, 2 * PAGE_SIZE, Prot::rw()).unwrap();
        parent.write(0x1_0000, &[0xAA]).unwrap();
        let child = parent.fork();
        // An empty write must not privatize the shared page (no byte lands).
        parent.write(0x1_0000, &[]).unwrap();
        assert_eq!(page_refs(&parent, 0x1_0000), 2, "empty write kept sharing");
        // And an empty write to a mapped-but-untouched page must not allocate it.
        parent.write(0x1_1000, &[]).unwrap();
        assert_eq!(page_refs(&parent, 0x1_1000), 0, "empty write stayed lazy");
        let _ = child;
    }

    #[test]
    fn spanning_read_mixes_present_and_none_pages() {
        let mut m = mem();
        m.map(0x1_0000, 2 * PAGE_SIZE, Prot::rw()).unwrap();
        // Write only into the second page, near the boundary.
        m.write(0x1_0000 + PAGE_SIZE, &[0xEE]).unwrap();
        let boundary = 0x1_0000 + PAGE_SIZE - 2;
        // Read straddles an unwritten (zero) page and the written one.
        assert_eq!(
            m.read_vec(boundary, 4).unwrap(),
            vec![0x00, 0x00, 0xEE, 0x00]
        );
    }
}
