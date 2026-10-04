//! Memory-management syscalls layered on top of the anonymous `mmap` arena:
//! `mremap`, `madvise`, and `mincore`. The `mlock`/`munlock` family, `mlockall`,
//! and `msync` model no swapping or dirty write-back, so they succeed as no-ops
//! directly in [`Kernel::dispatch`] rather than here.
//!
//! These handlers only touch [`GuestMemory`]'s public API plus the per-process
//! arena cursor via [`Kernel::alloc_mmap`]; they never alter fork/COW semantics
//! and never service file-backed mappings.

use super::{Kernel, ServiceCtx, Shared, err};
use crate::abi::errno::Errno;
use crate::vcpu::GuestMemory;
use crate::vcpu::mem::{MemError, PAGE_SIZE, Prot};

/// `MREMAP_MAYMOVE`: the kernel may relocate the mapping to satisfy a grow.
const MREMAP_MAYMOVE: u64 = 1;
/// `MREMAP_FIXED`: relocate the mapping to a caller-chosen address (requires
/// `MREMAP_MAYMOVE`), replacing whatever is currently mapped there.
const MREMAP_FIXED: u64 = 2;
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
/// Whether every page in `[start, end)` is currently mapped.
fn range_is_mapped(mem: &GuestMemory, start: u64, end: u64) -> bool {
    let mut p = start;
    while p < end {
        if matches!(mem.read_vec(p, 1), Err(MemError::Unmapped(_))) {
            return false;
        }
        p += PAGE_SIZE;
    }
    true
}

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
    /// existing mapping.
    ///
    /// Shrinking unmaps the tail and keeps the base. Growing tries to claim the
    /// following pages in place; if they are free it succeeds at the same
    /// address. When that is not possible and `MREMAP_MAYMOVE` is set, a fresh
    /// region is taken from the `mmap` arena (or `new_addr` with
    /// `MREMAP_FIXED`) and the pages move there: a *shared* mapping's frames
    /// are remapped (so it keeps sharing with its other mappers), private
    /// pages are copied. The mapping keeps its protection, a writable shared
    /// file mapping's write-back follows it, and `MREMAP_DONTUNMAP` leaves the
    /// old range mapped (empty). Linux's argument checks apply: page-aligned
    /// addresses, known flags, `FIXED`/`DONTUNMAP` only with `MAYMOVE`,
    /// `DONTUNMAP` without resizing, non-overlapping `FIXED` ranges.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub(super) fn sys_mremap(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        old_addr: u64,
        old_size: u64,
        new_size: u64,
        flags: u64,
        new_addr: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const MREMAP_DONTUNMAP: u64 = 4;
        if old_size == 0 || new_size == 0 || !old_addr.is_multiple_of(PAGE_SIZE) {
            return err(Errno::EINVAL);
        }
        if flags & !(MREMAP_MAYMOVE | MREMAP_FIXED | MREMAP_DONTUNMAP) != 0
            || (flags & (MREMAP_FIXED | MREMAP_DONTUNMAP) != 0 && flags & MREMAP_MAYMOVE == 0)
        {
            return err(Errno::EINVAL);
        }
        let old_size = page_up(old_size);
        let new_size = page_up(new_size);
        let dontunmap = flags & MREMAP_DONTUNMAP != 0;
        if dontunmap && old_size != new_size {
            return err(Errno::EINVAL);
        }

        // The source must actually be mapped; Linux answers EFAULT otherwise.
        // musl's `pthread_getattr_np` finds the main thread's stack extent by
        // walking `mremap` *down* the stack until the call stops failing with
        // ENOMEM, so this is a hot path with a load-bearing errno — silently
        // succeeding here would both mis-size the guest's stack and map stray
        // pages under it.
        if !range_is_mapped(mem, old_addr, old_addr + old_size) {
            return err(Errno::EFAULT);
        }
        let prot = mem.page_prot(old_addr).unwrap_or(Prot::rw());

        if flags & MREMAP_FIXED != 0 {
            if !new_addr.is_multiple_of(PAGE_SIZE) {
                return err(Errno::EINVAL);
            }
            if new_addr < old_addr + old_size && old_addr < new_addr + new_size {
                return err(Errno::EINVAL); // overlapping ranges
            }
            sh.arena(cx).claim(new_addr, new_size);
            if let Err(e) = self.mremap_move(
                cx, old_addr, old_size, new_addr, new_size, prot, dontunmap, mem,
            ) {
                return e;
            }
            if !dontunmap {
                sh.arena(cx).free_range(old_addr, old_size);
            }
            return new_addr as i64;
        }

        if new_size <= old_size && !dontunmap {
            // Shrink (or no-op): drop the tail, keep the base. The tail goes
            // back to the arena — leaking it here would bleed the arena dry in
            // a guest that resizes buffers in a loop.
            let tail = old_addr + new_size;
            let freed = old_size - new_size;
            let _ = mem.unmap(tail, freed);
            sh.arena(cx).free_range(tail, freed);
            return old_addr as i64;
        }

        // Grow: first try to claim the following pages in place — only when
        // they are freed arena space (so nothing else owns them, and the arena
        // is told they are taken; an unmapped page above the arena may be the
        // stack guard gap or an image's neighbour).
        let extra_start = old_addr + old_size;
        let extra_len = new_size - old_size;
        if !dontunmap
            && sh.arena(cx).is_free(extra_start, extra_len)
            && range_is_free(mem, extra_start, extra_start + extra_len)
            && mem.map(extra_start, extra_len, prot).is_ok()
        {
            sh.arena(cx).claim(extra_start, extra_len);
            return old_addr as i64;
        }

        // In-place grow is not clean; relocate only if allowed.
        if flags & MREMAP_MAYMOVE == 0 {
            return err(Errno::ENOMEM);
        }
        let Some(base) = self.alloc_mmap(sh, cx, new_size) else {
            return err(Errno::ENOMEM);
        };
        if let Err(e) =
            self.mremap_move(cx, old_addr, old_size, base, new_size, prot, dontunmap, mem)
        {
            return e;
        }
        // The old block is ours again — without this, every relocating mremap
        // leaks its source and the arena runs out (Bun resizes buffers by the
        // thousand, which exhausted it and turned every later allocation into a
        // NULL the guest promptly dereferenced).
        if !dontunmap {
            sh.arena(cx).free_range(old_addr, old_size);
        }
        base as i64
    }

    /// Move `[old, old + old_size)` to `[new, new + new_size)` with `prot`:
    /// shared pages by remapping their frames, private ones by copying; then
    /// unmap the source (or, `dontunmap`, leave it mapped and empty). Keeps a
    /// writable shared file mapping's write-back pointed at its new home.
    #[allow(clippy::too_many_arguments, clippy::unused_self)]
    fn mremap_move(
        &self,
        cx: &mut ServiceCtx,
        old: u64,
        old_size: u64,
        new: u64,
        new_size: u64,
        prot: Prot,
        dontunmap: bool,
        mem: &mut GuestMemory,
    ) -> Result<(), i64> {
        let keep = old_size.min(new_size);
        if let Some(frames) = mem.shared_frames(old, keep) {
            mem.map_frames(new, &frames, prot)
                .map_err(|_| err(Errno::ENOMEM))?;
            if new_size > keep {
                mem.map_shared_anon(new + keep, new_size - keep, prot)
                    .map_err(|_| err(Errno::ENOMEM))?;
            }
        } else {
            mem.map(new, new_size, prot)
                .map_err(|_| err(Errno::ENOMEM))?;
            // Copy the contents forward (a write-only source can't be read
            // through the guest view; init writes ignore protection).
            if let Ok(data) = mem.read_vec(old, keep as usize) {
                let _ = mem.write_init(new, &data);
            }
        }
        if dontunmap {
            let _ = mem.map(old, old_size, prot);
        } else {
            let _ = mem.unmap(old, old_size);
        }
        for m in &mut cx.cur.shared_maps {
            if m.base == old {
                m.base = new;
                m.len = m.len.min(new_size);
            }
        }
        Ok(())
    }

    /// `madvise(addr, len, advice)`. After Linux's checks — a page-aligned
    /// `addr`, a known `advice` (`EINVAL`), and for the advice that acts on
    /// pages a fully mapped range (`ENOMEM`) — the advice that changes what
    /// the guest observes is honored:
    /// - `MADV_DONTNEED`/`MADV_FREE`: private anonymous pages read back as
    ///   zero (`FREE` may legally keep contents; zeroing is the conservative
    ///   answer); shared pages keep their contents, as Linux refaults them from
    ///   the shared object; file-backed (ELF) pages keep theirs too.
    /// - `MADV_REMOVE`: punches the backing out of shared memory (zeros).
    /// - `MADV_DONTFORK`/`MADV_WIPEONFORK` (and `DOFORK`/`KEEPONFORK`): the
    ///   fork policy, applied by the next `fork` (glibc's `arc4random` and
    ///   OpenSSL rely on WIPEONFORK so a child never reuses the parent's
    ///   random state).
    ///
    /// Everything else is a hint with nothing to act on in memory that is
    /// never paged out: accepted. `MADV_GUARD_INSTALL` (6.13) is `EINVAL`, as
    /// on kernels without it.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_madvise(
        &self,
        addr: u64,
        len: u64,
        advice: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const MADV_FREE: u64 = 8;
        const MADV_REMOVE: u64 = 9;
        const MADV_DONTFORK: u64 = 10;
        const MADV_DOFORK: u64 = 11;
        const MADV_WIPEONFORK: u64 = 18;
        const MADV_KEEPONFORK: u64 = 19;
        // NORMAL..DONTNEED, FREE..DODUMP, WIPEONFORK..COLLAPSE, HWPOISON,
        // SOFT_OFFLINE.
        let known = matches!(advice, 0..=4 | 8..=25 | 100 | 101);
        if !known || !addr.is_multiple_of(PAGE_SIZE) {
            return err(Errno::EINVAL);
        }
        if len == 0 {
            return 0;
        }
        let start = addr;
        let end = page_up(addr.saturating_add(len));
        // Linux refuses advice over a range with holes: an unmapped page
        // anywhere in it makes the call fail with ENOMEM. musl's allocator
        // relies on this to detect gaps.
        if !range_is_mapped(mem, start, end) {
            return err(Errno::ENOMEM);
        }
        match advice {
            MADV_DONTNEED | MADV_FREE | MADV_REMOVE => {
                let zero = [0u8; PAGE_SIZE as usize];
                let mut p = start;
                while p < end {
                    // File-backed (ELF-segment) pages keep their contents: on
                    // Linux MADV_DONTNEED discards the private copy and the
                    // next access reloads the file (Bun's embedded bytecode
                    // lives in such a segment). Shared pages are refaulted
                    // from the shared object, so they keep theirs too — except
                    // under MADV_REMOVE, which frees that backing.
                    let shared = mem.shared_phys(p).is_some();
                    if !mem.is_file_backed(p) && (!shared || advice == MADV_REMOVE) {
                        // Only mapped, writable pages take zeros; ignore the rest.
                        let _ = mem.write(p, &zero);
                    }
                    p += PAGE_SIZE;
                }
                0
            }
            MADV_DONTFORK | MADV_DOFORK | MADV_WIPEONFORK | MADV_KEEPONFORK => {
                let policy = match advice {
                    MADV_DONTFORK => crate::vcpu::mem::FORK_DONT,
                    MADV_WIPEONFORK => crate::vcpu::mem::FORK_WIPE,
                    _ => 0,
                };
                match mem.set_fork_policy(start, end - start, policy) {
                    Ok(()) => 0,
                    Err(_) => err(Errno::ENOMEM),
                }
            }
            _ => 0,
        }
    }

    /// `mincore(addr, len, vec)` — report per-page residency. Bit 0 of each byte
    /// is set only for pages that actually have a backing frame right now; a
    /// mapped-but-untouched (demand-paged, lazy) page reports 0, matching Linux.
    /// A range with any unmapped page is rejected with ENOMEM.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_mincore(&self, addr: u64, len: u64, vec: u64, mem: &mut GuestMemory) -> i64 {
        if len == 0 {
            return 0;
        }
        // addr must be page-aligned (Linux answers EINVAL otherwise).
        if !addr.is_multiple_of(PAGE_SIZE) {
            return err(Errno::EINVAL);
        }
        let start = page_down(addr);
        let end = page_up(addr + len);
        let pages = ((end - start) / PAGE_SIZE) as usize;
        let mut resident = vec![0u8; pages];
        for (i, r) in resident.iter_mut().enumerate() {
            let p = start + i as u64 * PAGE_SIZE;
            // A hole anywhere in the range makes the whole call ENOMEM.
            if mem.page_prot(p).is_none() {
                return err(Errno::ENOMEM);
            }
            *r = u8::from(mem.is_resident(p));
        }
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

    fn setup() -> (Kernel, GuestMemory, ServiceCtx) {
        let mut mounts = MountTable::new();
        mounts.mount("/", Box::new(TmpFs::new()));
        let mut kernel = Kernel::new(Arch::Aarch64, mounts);
        let mut cx = ServiceCtx::default();
        cx.cur.pid = 1;
        // These tests drive the handlers directly (no boot/run), so mm 0 needs
        // its mmap arena set up here.
        cx.cur.mm = 0;
        kernel.set_mmap_area(0x1_0000 + 16 * PAGE, 0x1_0000);
        kernel
            .shared
            .get_mut()
            .unwrap()
            .mmap_areas
            .push(crate::kernel::Arena::new(0x1_0000 + 16 * PAGE, 0x1_0000));
        let mem = GuestMemory::new(0x1_0000, 16 * PAGE);
        (kernel, mem, cx)
    }

    #[test]
    fn mremap_grow_in_place_keeps_address_and_new_pages_work() {
        let (k, mut mem, mut cx) = setup();
        let mut sh = k.shared.lock().unwrap();
        // Two arena blocks, `lo` right below `hi` (the arena grows down); free
        // `hi`, so `lo` can grow into it in place.
        let hi = k.alloc_mmap(&mut sh, &mut cx, 2 * PAGE).unwrap();
        mem.map(hi, 2 * PAGE, Prot::rw()).unwrap();
        let lo = k.alloc_mmap(&mut sh, &mut cx, 2 * PAGE).unwrap();
        mem.map(lo, 2 * PAGE, Prot::rw()).unwrap();
        assert_eq!(lo + 2 * PAGE, hi);
        mem.write_u64(lo, 0x1111).unwrap();
        assert_eq!(k.sys_munmap(&mut sh, &mut cx, hi, 2 * PAGE, &mut mem), 0);

        let ret = k.sys_mremap(&mut sh, &mut cx, lo, 2 * PAGE, 4 * PAGE, 0, 0, &mut mem);
        assert_eq!(ret, lo as i64, "grow-in-place returns the same address");
        mem.write_u64(hi, 0xabcd_ef01).unwrap();
        assert_eq!(mem.read_u64(hi).unwrap(), 0xabcd_ef01, "grown pages usable");

        // The regression: the arena must know those pages are taken again, or
        // the next mmap gets them and zero-fills the grown buffer (musl's
        // allocator then aborts on the wiped chunk header — apk update).
        let next = k.alloc_mmap(&mut sh, &mut cx, 2 * PAGE).unwrap();
        assert!(next + 2 * PAGE <= lo || next >= lo + 4 * PAGE, "no overlap");
        assert_eq!(mem.read_u64(lo).unwrap(), 0x1111);
    }

    #[test]
    fn mremap_grow_never_takes_pages_outside_the_arena() {
        let (k, mut mem, mut cx) = setup();
        let mut sh = k.shared.lock().unwrap();
        // The topmost arena block: above it is not arena space (the stack guard
        // gap in a real process), so growing must relocate, not extend.
        let top = k.alloc_mmap(&mut sh, &mut cx, PAGE).unwrap();
        mem.map(top, PAGE, Prot::rw()).unwrap();
        let ret = k.sys_mremap(&mut sh, &mut cx, top, PAGE, 2 * PAGE, 0, 0, &mut mem);
        assert_eq!(ret, err(Errno::ENOMEM), "no in-place grow past the arena");
    }

    #[test]
    fn mremap_shrink_unmaps_tail() {
        let (k, mut mem, mut cx) = setup();
        mem.map(0x1_0000, 4 * PAGE, Prot::rw()).unwrap();

        let ret = k.sys_mremap(
            &mut k.shared.lock().unwrap(),
            &mut cx,
            0x1_0000,
            4 * PAGE,
            2 * PAGE,
            0,
            0,
            &mut mem,
        );
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
        let (mut k, mut mem, mut cx) = setup();
        k.set_mmap_area(0x1_0000 + 16 * PAGE, 0x1_0000);
        // 1-page mapping immediately followed by an occupied page, so an
        // in-place grow is impossible.
        mem.map(0x1_0000, PAGE, Prot::rw()).unwrap();
        mem.map(0x1_1000, PAGE, Prot::rw()).unwrap();
        mem.write_u64(0x1_0000, 0x1122_3344).unwrap();

        let ret = k.sys_mremap(
            &mut k.shared.lock().unwrap(),
            &mut cx,
            0x1_0000,
            PAGE,
            2 * PAGE,
            MREMAP_MAYMOVE,
            0,
            &mut mem,
        );
        assert_ne!(ret, 0x1_0000, "MAYMOVE relocated the mapping");
        assert!(ret >= 0);
        // Old bytes were copied to the new region.
        assert_eq!(mem.read_u64(ret as u64).unwrap(), 0x1122_3344);
        // The old range is unmapped.
        assert!(matches!(mem.read_u64(0x1_0000), Err(MemError::Unmapped(_))));
    }

    #[test]
    fn mremap_fixed_honors_requested_destination() {
        let (k, mut mem, mut cx) = setup();
        // Source page with content, and a distinct reserved destination page.
        mem.map(0x1_0000, PAGE, Prot::rw()).unwrap();
        mem.write_u64(0x1_0000, 0x7777).unwrap();
        let dst = 0x1_5000;
        mem.map(dst, PAGE, Prot::rw()).unwrap();

        let ret = k.sys_mremap(
            &mut k.shared.lock().unwrap(),
            &mut cx,
            0x1_0000,
            PAGE,
            PAGE,
            MREMAP_MAYMOVE | MREMAP_FIXED,
            dst,
            &mut mem,
        );
        assert_eq!(ret, dst as i64, "MREMAP_FIXED lands at the requested addr");
        // Content moved to the destination; the source is unmapped.
        assert_eq!(mem.read_u64(dst).unwrap(), 0x7777);
        assert!(matches!(mem.read_u64(0x1_0000), Err(MemError::Unmapped(_))));
    }

    #[test]
    fn mremap_validates_arguments_like_linux() {
        let (k, mut mem, mut cx) = setup();
        mem.map(0x1_0000, 2 * PAGE, Prot::rw()).unwrap();
        let mut go = |old: u64, os: u64, ns: u64, fl: u64, na: u64, mem: &mut GuestMemory| {
            k.sys_mremap(
                &mut k.shared.lock().unwrap(),
                &mut cx,
                old,
                os,
                ns,
                fl,
                na,
                mem,
            )
        };
        let inval = -i64::from(Errno::EINVAL.0);
        assert_eq!(
            go(0x1_0010, PAGE, PAGE, 0, 0, &mut mem),
            inval,
            "unaligned old_addr"
        );
        assert_eq!(
            go(0x1_0000, PAGE, PAGE, 0x80, 0, &mut mem),
            inval,
            "unknown flag"
        );
        assert_eq!(
            go(0x1_0000, PAGE, PAGE, MREMAP_FIXED, 0x1_8000, &mut mem),
            inval,
            "FIXED w/o MAYMOVE"
        );
        assert_eq!(
            go(0x1_0000, PAGE, 2 * PAGE, 1 | 4, 0, &mut mem),
            inval,
            "DONTUNMAP resizing"
        );
        assert_eq!(
            go(
                0x1_0000,
                PAGE,
                PAGE,
                MREMAP_MAYMOVE | MREMAP_FIXED,
                0x1_1000 - PAGE / 2,
                &mut mem
            ),
            inval,
            "unaligned new_addr"
        );
        assert_eq!(
            go(
                0x1_0000,
                2 * PAGE,
                2 * PAGE,
                MREMAP_MAYMOVE | MREMAP_FIXED,
                0x1_1000,
                &mut mem
            ),
            inval,
            "overlapping FIXED ranges"
        );
    }

    #[test]
    fn mremap_keeps_protection_and_shared_frames() {
        let (k, mut mem, mut cx) = setup();
        // A read-only private page keeps PROT_READ at its new home.
        mem.map(0x1_0000, PAGE, Prot::rw()).unwrap();
        mem.write_u64(0x1_0000, 0x55).unwrap();
        mem.protect(0x1_0000, PAGE, Prot::READ).unwrap();
        let dst = 0x1_8000;
        let fl = MREMAP_MAYMOVE | MREMAP_FIXED;
        let ret = k.sys_mremap(
            &mut k.shared.lock().unwrap(),
            &mut cx,
            0x1_0000,
            PAGE,
            PAGE,
            fl,
            dst,
            &mut mem,
        );
        assert_eq!(ret, dst as i64);
        assert_eq!(mem.page_prot(dst), Some(Prot::READ));
        assert_eq!(mem.read_u64(dst).unwrap(), 0x55);

        // A shared page moves by frame: the new address aliases the same
        // physical memory (what a forked sibling still maps), not a copy.
        mem.map_shared_anon(0x1_2000, PAGE, Prot::rw()).unwrap();
        mem.write_u64(0x1_2000, 0x99).unwrap();
        let pa = mem.shared_phys(0x1_2000).unwrap();
        let dst2 = 0x1_a000;
        let ret = k.sys_mremap(
            &mut k.shared.lock().unwrap(),
            &mut cx,
            0x1_2000,
            PAGE,
            PAGE,
            fl,
            dst2,
            &mut mem,
        );
        assert_eq!(ret, dst2 as i64);
        assert_eq!(mem.shared_phys(dst2), Some(pa));
        assert_eq!(mem.read_u64(dst2).unwrap(), 0x99);

        // DONTUNMAP leaves the source mapped, empty.
        let dst3 = 0x1_c000;
        let ret = k.sys_mremap(
            &mut k.shared.lock().unwrap(),
            &mut cx,
            dst,
            PAGE,
            PAGE,
            MREMAP_MAYMOVE | 4 | MREMAP_FIXED,
            dst3,
            &mut mem,
        );
        assert_eq!(ret, dst3 as i64);
        assert_eq!(mem.read_u64(dst3).unwrap(), 0x55);
        assert_eq!(mem.read_u64(dst).unwrap(), 0);
    }

    #[test]
    fn madvise_dontneed_zeros_pages() {
        let (k, mut mem, _cx) = setup();
        mem.map(0x1_0000, PAGE, Prot::rw()).unwrap();
        mem.write_u64(0x1_0010, 0xdead_beef).unwrap();

        assert_eq!(k.sys_madvise(0x1_0000, PAGE, MADV_DONTNEED, &mut mem), 0);
        assert_eq!(mem.read_u64(0x1_0010).unwrap(), 0, "page was zeroed");
    }

    #[test]
    fn madvise_dontneed_over_unmapped_is_enomem() {
        let (k, mut mem, _cx) = setup();
        // One mapped page followed by an unmapped hole: DONTNEED across both must
        // fail ENOMEM and touch nothing.
        mem.map(0x1_0000, PAGE, Prot::rw()).unwrap();
        mem.write_u64(0x1_0000, 0x41).unwrap();
        assert_eq!(
            k.sys_madvise(0x1_0000, 2 * PAGE, MADV_DONTNEED, &mut mem),
            err(Errno::ENOMEM),
        );
        // The mapped page was NOT zeroed (the call discarded nothing).
        assert_eq!(mem.read_u64(0x1_0000).unwrap(), 0x41);
    }

    #[test]
    fn mincore_reports_real_residency() {
        let (k, mut mem, _cx) = setup();
        mem.map(0x1_0000, 4 * PAGE, Prot::rw()).unwrap();
        // Touch pages 0 and 2 so they get a backing frame; 1 and 3 stay lazy.
        mem.write_u64(0x1_0000, 1).unwrap();
        mem.write_u64(0x1_0000 + 2 * PAGE, 1).unwrap();
        let out = 0x1_0000; // report vector shares page 0 (already resident)
        assert_eq!(k.sys_mincore(0x1_0000, 4 * PAGE, out, &mut mem), 0);
        // 1 only for the touched (resident) pages, 0 for the demand-paged ones.
        assert_eq!(mem.read_vec(out, 4).unwrap(), vec![1, 0, 1, 0]);
    }

    #[test]
    fn mincore_over_unmapped_is_enomem() {
        let (k, mut mem, _cx) = setup();
        mem.map(0x1_0000, PAGE, Prot::rw()).unwrap();
        // Range extends into an unmapped page → ENOMEM.
        assert_eq!(
            k.sys_mincore(0x1_0000, 2 * PAGE, 0x1_0000, &mut mem),
            err(Errno::ENOMEM),
        );
    }

    #[test]
    fn msync_rejects_bad_flags() {
        let (k, mut mem, mut cx) = setup();
        let mut v = DummyVcpu;
        // MS_SYNC(4) | MS_ASYNC(1) together is mutually exclusive → EINVAL.
        assert_eq!(
            k.dispatch(
                &mut cx,
                Sysno::Msync,
                0,
                &[0, PAGE, 5, 0, 0, 0],
                &mut v,
                &mut mem
            ),
            err(Errno::EINVAL),
        );
        // An unknown flag bit → EINVAL.
        assert_eq!(
            k.dispatch(
                &mut cx,
                Sysno::Msync,
                0,
                &[0, PAGE, 0x10, 0, 0, 0],
                &mut v,
                &mut mem
            ),
            err(Errno::EINVAL),
        );
        // MS_SYNC alone is fine (no shared maps: a plain no-op success).
        assert_eq!(
            k.dispatch(
                &mut cx,
                Sysno::Msync,
                0,
                &[0, PAGE, 4, 0, 0, 0],
                &mut v,
                &mut mem
            ),
            0,
        );
    }

    #[test]
    fn mlock_family_are_noops() {
        let (k, mut mem, mut cx) = setup();
        let mut v = DummyVcpu;
        for s in [
            Sysno::Mlock,
            Sysno::Mlock2,
            Sysno::Munlock,
            Sysno::Munlockall,
            Sysno::Msync,
        ] {
            assert_eq!(
                k.dispatch(&mut cx, s, 0, &[0; 6], &mut v, &mut mem),
                0,
                "{s:?}"
            );
        }
        // mlockall needs MCL_CURRENT and/or MCL_FUTURE; MCL_ONFAULT alone or
        // nothing at all is EINVAL, as are unknown bits.
        let ml = |k: &Kernel, cx: &mut ServiceCtx, mem: &mut GuestMemory, f: u64| {
            k.dispatch(
                cx,
                Sysno::Mlockall,
                0,
                &[f, 0, 0, 0, 0, 0],
                &mut DummyVcpu,
                mem,
            )
        };
        assert_eq!(ml(&k, &mut cx, &mut mem, 1 | 2), 0);
        assert_eq!(ml(&k, &mut cx, &mut mem, 0), err(Errno::EINVAL));
        assert_eq!(ml(&k, &mut cx, &mut mem, 4), err(Errno::EINVAL));
        assert_eq!(ml(&k, &mut cx, &mut mem, 8 | 1), err(Errno::EINVAL));
        // mlock over an unmapped range is ENOMEM; mlock2's only flag is ONFAULT.
        let r = k.dispatch(
            &mut cx,
            Sysno::Mlock,
            0,
            &[0x7_0000, 4096, 0, 0, 0, 0],
            &mut v,
            &mut mem,
        );
        assert_eq!(r, err(Errno::ENOMEM));
        let r = k.dispatch(
            &mut cx,
            Sysno::Mlock2,
            0,
            &[0, 0, 2, 0, 0, 0],
            &mut v,
            &mut mem,
        );
        assert_eq!(r, err(Errno::EINVAL));
        // madvise: unknown advice / unaligned address are EINVAL.
        let r = k.dispatch(
            &mut cx,
            Sysno::Madvise,
            0,
            &[0x1_0000, 4096, 77, 0, 0, 0],
            &mut v,
            &mut mem,
        );
        assert_eq!(r, err(Errno::EINVAL));
        let r = k.dispatch(
            &mut cx,
            Sysno::Madvise,
            0,
            &[0x1_0001, 4096, 4, 0, 0, 0],
            &mut v,
            &mut mem,
        );
        assert_eq!(r, err(Errno::EINVAL));
    }
}
