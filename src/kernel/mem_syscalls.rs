//! Memory-management syscalls layered on top of the anonymous `mmap` arena:
//! `mremap`, `madvise`, and `mincore`. The `mlock`/`munlock` family, `mlockall`,
//! and `msync` model no swapping or dirty write-back, so they succeed as no-ops
//! directly in [`Kernel::dispatch`] rather than here.
//!
//! These handlers only touch [`GuestMemory`]'s public API plus the per-process
//! arena cursor via [`Kernel::alloc_mmap`]; they never alter fork/COW semantics
//! and never service file-backed mappings.

use super::{Kernel, err};
use crate::abi::errno::Errno;
use crate::vcpu::GuestMemory;
use crate::vcpu::mem::{MemError, PAGE_SIZE, Prot};

/// `MREMAP_MAYMOVE`: the kernel may relocate the mapping to satisfy a grow.
const MREMAP_MAYMOVE: u64 = 1;
/// `MADV_DONTNEED`: drop the pages; a later access reads fresh zeros.
const MADV_DONTNEED: u64 = 4;

/// Round `v` up to the next page boundary.
fn page_up(v: u64) -> u64 {
    v.div_ceil(PAGE_SIZE) * PAGE_SIZE
}

/// Round `v` down to its page boundary.
fn page_down(v: u64) -> u64 {
    v - v % PAGE_SIZE
}

/// Whether every page in `[start, end)` is unmapped (and thus in-bounds room we
/// could grow a mapping into). A mapped, protected, or out-of-bounds page all
/// count as "not free".
fn range_is_free(mem: &GuestMemory, start: u64, end: u64) -> bool {
    let mut p = start;
    while p < end {
        if !matches!(mem.read_vec(p, 1), Err(MemError::Unmapped(_))) {
            return false;
        }
        p += PAGE_SIZE;
    }
    true
}

impl Kernel {
    /// `mremap(old_addr, old_size, new_size, flags, new_addr)` — resize an
    /// existing anonymous mapping.
    ///
    /// Shrinking unmaps the tail and keeps the base. Growing tries to claim the
    /// following pages in place; if they are free it succeeds at the same
    /// address. When that is not possible and `MREMAP_MAYMOVE` is set, a fresh
    /// region is taken from the `mmap` arena, the old bytes are copied over, and
    /// the old range is unmapped (best-effort relocate).
    pub(super) fn sys_mremap(
        &mut self,
        old_addr: u64,
        old_size: u64,
        new_size: u64,
        flags: u64,
        _new_addr: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        if old_size == 0 || new_size == 0 {
            return err(Errno::EINVAL);
        }
        let old_addr = page_down(old_addr);
        let old_size = page_up(old_size);
        let new_size = page_up(new_size);

        if new_size <= old_size {
            // Shrink (or no-op): drop the tail, keep the base.
            let tail = old_addr + new_size;
            let _ = mem.unmap(tail, old_size - new_size);
            return old_addr as i64;
        }

        // Grow: first try to claim the following pages in place.
        let extra_start = old_addr + old_size;
        let extra_len = new_size - old_size;
        if range_is_free(mem, extra_start, extra_start + extra_len)
            && mem.map(extra_start, extra_len, Prot::rw()).is_ok()
        {
            return old_addr as i64;
        }

        // In-place grow is not clean; relocate only if allowed.
        if flags & MREMAP_MAYMOVE == 0 {
            return err(Errno::ENOMEM);
        }
        let Some(base) = self.alloc_mmap(new_size) else {
            return err(Errno::ENOMEM);
        };
        if mem.map(base, new_size, Prot::rw()).is_err() {
            return err(Errno::ENOMEM);
        }
        // Copy the old contents forward (best-effort: needs READ on the source).
        if let Ok(data) = mem.read_vec(old_addr, old_size as usize) {
            let _ = mem.write(base, &data);
        }
        let _ = mem.unmap(old_addr, old_size);
        base as i64
    }

    /// `madvise(addr, len, advice)` — advisory, so always succeeds. `MADV_DONTNEED`
    /// is honored by zeroing the mapped pages so the guest sees fresh zero pages;
    /// every other advice is ignored.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_madvise(
        &mut self,
        addr: u64,
        len: u64,
        advice: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        if advice == MADV_DONTNEED && len != 0 {
            let mut p = page_down(addr);
            let end = page_up(addr + len);
            let zero = [0u8; PAGE_SIZE as usize];
            while p < end {
                // Only mapped, writable pages take zeros; ignore everything else.
                let _ = mem.write(p, &zero);
                p += PAGE_SIZE;
            }
        }
        0
    }

    /// `mincore(addr, len, vec)` — report residency. Everything mapped here is
    /// resident, so write `1` for each page spanned by `[addr, addr + len)`.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_mincore(
        &mut self,
        addr: u64,
        len: u64,
        vec: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        if len == 0 {
            return 0;
        }
        let pages = ((page_up(addr + len) - page_down(addr)) / PAGE_SIZE) as usize;
        let resident = vec![1u8; pages];
        if mem.write(vec, &resident).is_err() {
            return err(Errno::EFAULT);
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::Arch;
    use crate::abi::arch::Sysno;
    use crate::fs::{MountTable, TmpFs};
    use crate::vcpu::{Exit, Vcpu, VcpuError};

    const PAGE: u64 = PAGE_SIZE;

    /// A no-op vcpu so we can exercise `dispatch` for the no-op syscalls.
    #[derive(Clone)]
    struct DummyVcpu;
    impl Vcpu for DummyVcpu {
        fn run(&mut self, _m: &mut GuestMemory) -> Result<Exit, VcpuError> {
            Ok(Exit::Halt)
        }
        fn syscall_nr(&self) -> u64 {
            0
        }
        fn syscall_args(&self) -> [u64; 6] {
            [0; 6]
        }
        fn set_syscall_ret(&mut self, _v: u64) {}
        fn reg(&self, _i: usize) -> u64 {
            0
        }
        fn set_reg(&mut self, _i: usize, _v: u64) {}
        fn pc(&self) -> u64 {
            0
        }
        fn set_pc(&mut self, _v: u64) {}
        fn sp(&self) -> u64 {
            0
        }
        fn set_sp(&mut self, _v: u64) {}
        fn set_tls(&mut self, _v: u64) {}
        fn fork(&self) -> Box<dyn Vcpu> {
            Box::new(self.clone())
        }
        fn reset(&mut self, _e: u64, _s: u64) {}
    }

    fn setup() -> (Kernel, GuestMemory) {
        let mut mounts = MountTable::new();
        mounts.mount("/", Box::new(TmpFs::new()));
        let mut kernel = Kernel::new(Arch::Aarch64, mounts);
        kernel.cur.pid = 1;
        let mem = GuestMemory::new(0x1_0000, 16 * PAGE);
        (kernel, mem)
    }

    #[test]
    fn mremap_grow_in_place_keeps_address_and_new_pages_work() {
        let (mut k, mut mem) = setup();
        // A 2-page mapping with 2 free pages after it.
        mem.map(0x1_0000, 2 * PAGE, Prot::rw()).unwrap();

        let ret = k.sys_mremap(0x1_0000, 2 * PAGE, 4 * PAGE, 0, 0, &mut mem);
        assert_eq!(ret, 0x1_0000, "grow-in-place returns the same address");

        // The freshly grown page is usable.
        let grown = 0x1_0000 + 2 * PAGE;
        mem.write_u64(grown, 0xabcd_ef01).unwrap();
        assert_eq!(mem.read_u64(grown).unwrap(), 0xabcd_ef01);
    }

    #[test]
    fn mremap_shrink_unmaps_tail() {
        let (mut k, mut mem) = setup();
        mem.map(0x1_0000, 4 * PAGE, Prot::rw()).unwrap();

        let ret = k.sys_mremap(0x1_0000, 4 * PAGE, 2 * PAGE, 0, 0, &mut mem);
        assert_eq!(ret, 0x1_0000, "shrink returns the old address");

        // The tail is gone: an access there now faults.
        let tail = 0x1_0000 + 2 * PAGE;
        assert!(matches!(mem.read_u64(tail), Err(MemError::Unmapped(_))));
        // The kept head still works.
        mem.write_u64(0x1_0000, 7).unwrap();
        assert_eq!(mem.read_u64(0x1_0000).unwrap(), 7);
    }

    #[test]
    fn mremap_maymove_relocates_when_blocked() {
        let (mut k, mut mem) = setup();
        k.set_mmap_area(0x1_0000 + 16 * PAGE, 0x1_0000);
        // 1-page mapping immediately followed by an occupied page, so an
        // in-place grow is impossible.
        mem.map(0x1_0000, PAGE, Prot::rw()).unwrap();
        mem.map(0x1_1000, PAGE, Prot::rw()).unwrap();
        mem.write_u64(0x1_0000, 0x1122_3344).unwrap();

        let ret = k.sys_mremap(0x1_0000, PAGE, 2 * PAGE, MREMAP_MAYMOVE, 0, &mut mem);
        assert_ne!(ret, 0x1_0000, "MAYMOVE relocated the mapping");
        assert!(ret >= 0);
        // Old bytes were copied to the new region.
        assert_eq!(mem.read_u64(ret as u64).unwrap(), 0x1122_3344);
        // The old range is unmapped.
        assert!(matches!(mem.read_u64(0x1_0000), Err(MemError::Unmapped(_))));
    }

    #[test]
    fn madvise_dontneed_zeros_pages() {
        let (mut k, mut mem) = setup();
        mem.map(0x1_0000, PAGE, Prot::rw()).unwrap();
        mem.write_u64(0x1_0010, 0xdead_beef).unwrap();

        assert_eq!(k.sys_madvise(0x1_0000, PAGE, MADV_DONTNEED, &mut mem), 0);
        assert_eq!(mem.read_u64(0x1_0010).unwrap(), 0, "page was zeroed");
    }

    #[test]
    fn mincore_reports_resident() {
        let (mut k, mut mem) = setup();
        mem.map(0x1_0000, 4 * PAGE, Prot::rw()).unwrap();
        let vec = 0x1_0000;
        assert_eq!(k.sys_mincore(0x1_1000, 2 * PAGE, vec, &mut mem), 0);
        assert_eq!(mem.read_vec(vec, 2).unwrap(), vec![1, 1]);
    }

    #[test]
    fn mlock_family_are_noops() {
        let (mut k, mut mem) = setup();
        let mut v = DummyVcpu;
        for s in [
            Sysno::Mlock,
            Sysno::Mlock2,
            Sysno::Munlock,
            Sysno::Mlockall,
            Sysno::Munlockall,
            Sysno::Msync,
        ] {
            assert_eq!(k.dispatch(s, 0, &[0; 6], &mut v, &mut mem), 0, "{s:?}");
        }
    }
}
