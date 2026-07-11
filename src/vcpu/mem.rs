//! Guest memory.
//!
//! A flat, host-backed region representing one guest process's physical/virtual
//! address space at `[base, base + size)`. Pages are tracked as unmapped until
//! `map`ped, each carrying `PROT_*` bits; reads/writes are bounds- and
//! permission-checked so a bad guest pointer surfaces as [`MemError`] instead of
//! corrupting host memory.
//!
//! This is the v1 model: one contiguous allocation, 4 KiB pages. A sparser
//! region tree and copy-on-write `fork` arrive with multi-process support
//! (ROADMAP Phase 6); the API here is what the loader, kernel and backends use.

/// Guest page size advertised to the guest (`AT_PAGESZ`, `sysconf(_SC_PAGESIZE)`).
pub const PAGE_SIZE: u64 = 4096;

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

/// The guest address space shared by every vcpu of one guest process.
#[derive(Clone)]
pub struct GuestMemory {
    base: u64,
    size: u64,
    bytes: Vec<u8>,
    /// Per-page protection; `prot[i]` is meaningful only when `mapped[i]`.
    prot: Vec<Prot>,
    mapped: Vec<bool>,
}

impl std::fmt::Debug for GuestMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mapped = self.mapped.iter().filter(|m| **m).count();
        f.debug_struct("GuestMemory")
            .field("base", &format_args!("{:#x}", self.base))
            .field("size", &self.size)
            .field("mapped_pages", &mapped)
            .field("total_pages", &self.mapped.len())
            .finish_non_exhaustive()
    }
}

impl GuestMemory {
    /// Reserve a flat guest region `[base, base + size)`. `base` and `size` must
    /// be page-aligned. All pages start unmapped and zeroed.
    #[must_use]
    pub fn new(base: u64, size: u64) -> Self {
        assert_eq!(base % PAGE_SIZE, 0, "base must be page-aligned");
        assert_eq!(size % PAGE_SIZE, 0, "size must be page-aligned");
        let npages = (size / PAGE_SIZE) as usize;
        Self {
            base,
            size,
            bytes: vec![0u8; size as usize],
            prot: vec![Prot::NONE; npages],
            mapped: vec![false; npages],
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
    /// range is rounded out to page boundaries. Newly-mapped pages keep their
    /// zero contents (backing store is zeroed at construction).
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
                return Err(MemError::Unmapped(self.base + (p as u64) * PAGE_SIZE));
            }
            self.prot[p] = prot;
        }
        Ok(())
    }

    /// Unmap the pages covering `[addr, addr + len)`.
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

    /// Verify `[addr, addr + len)` is mapped and every page grants `need`.
    fn check(&self, addr: u64, len: usize, need: Prot) -> Result<(), MemError> {
        let (first, last) = self.page_range(addr, len)?;
        for p in first..=last {
            if !self.mapped[p] {
                return Err(MemError::Unmapped(self.base + (p as u64) * PAGE_SIZE));
            }
            if !self.prot[p].contains(need) {
                return Err(MemError::Protection {
                    addr: self.base + (p as u64) * PAGE_SIZE,
                    needed: need,
                });
            }
        }
        Ok(())
    }

    /// Read `buf.len()` bytes from guest `addr` (requires `READ`).
    pub fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemError> {
        self.check(addr, buf.len(), Prot::READ)?;
        let off = (addr - self.base) as usize;
        buf.copy_from_slice(&self.bytes[off..off + buf.len()]);
        Ok(())
    }

    /// Read `len` bytes into a fresh `Vec` (requires `READ`).
    pub fn read_vec(&self, addr: u64, len: usize) -> Result<Vec<u8>, MemError> {
        let mut v = vec![0u8; len];
        self.read(addr, &mut v)?;
        Ok(v)
    }

    /// Write `buf` to guest `addr` (requires `WRITE`).
    pub fn write(&mut self, addr: u64, buf: &[u8]) -> Result<(), MemError> {
        self.check(addr, buf.len(), Prot::WRITE)?;
        let off = (addr - self.base) as usize;
        self.bytes[off..off + buf.len()].copy_from_slice(buf);
        Ok(())
    }

    /// Write `buf` to guest `addr`, bypassing protection but still requiring the
    /// pages be mapped and in bounds. For host-side initialization (loading ELF
    /// segments into read-only pages before the guest runs).
    pub fn write_init(&mut self, addr: u64, buf: &[u8]) -> Result<(), MemError> {
        let (first, last) = self.page_range(addr, buf.len())?;
        for p in first..=last {
            if !self.mapped[p] {
                return Err(MemError::Unmapped(self.base + (p as u64) * PAGE_SIZE));
            }
        }
        let off = (addr - self.base) as usize;
        self.bytes[off..off + buf.len()].copy_from_slice(buf);
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
        // 64 KiB region at 0x1_0000.
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
}
