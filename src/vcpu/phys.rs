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
use super::pagetable::PtFormat;
use super::region::Region;
use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

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
    /// The descriptor format of every page table built in this pool (a
    /// [`PtFormat`] as `u8`): the guest architecture's, so a hardware walker of
    /// that architecture can walk them. Chosen once, before the pool is shared
    /// (see [`crate::vcpu::GuestMemory::set_guest_arch`]).
    format: AtomicU8,
    /// Where the pool sits in guest-physical space as a *hardware* walker sees
    /// it: the address page-table entries and the translation base hold for
    /// physical address 0. 0 for an x86-64 pool (KVM maps each VM's pool at
    /// guest-physical 0); an arm64 pool gets its own window of the one
    /// process-wide Hypervisor.framework IPA space (see [`ipa_window`]), so
    /// several pools can be mapped at once. Every `pa` this type's methods take
    /// is still an offset from the pool's start.
    base: AtomicU64,
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
            format: AtomicU8::new(PtFormat::X86_64 as u8),
            base: AtomicU64::new(0),
        }
    }

    /// Where physical address 0 of the pool sits in the guest-physical space a
    /// hardware page-table walker uses (see the field doc): 0 for an x86-64
    /// pool, the pool's IPA window for an arm64 one.
    #[must_use]
    pub fn base(&self) -> u64 {
        self.base.load(Ordering::Relaxed)
    }

    /// The descriptor format of the page tables built in this pool.
    #[must_use]
    pub fn format(&self) -> PtFormat {
        PtFormat::from_u8(self.format.load(Ordering::Relaxed))
    }

    /// Switch the pool's page-table format, giving an arm64 pool a window of
    /// guest-physical space (and an x86-64 one base 0). The caller re-encodes
    /// every table already built in the pool.
    pub(crate) fn set_format(&self, format: PtFormat) {
        let old = self.base.swap(0, Ordering::Relaxed);
        if old != 0 {
            release_window(old, self.size as u64);
        }
        let base = match format {
            PtFormat::X86_64 => 0,
            PtFormat::Aarch64 => ipa_window(self.size as u64),
        };
        self.base.store(base, Ordering::Relaxed);
        self.format.store(format as u8, Ordering::Relaxed);
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

impl Drop for PhysMem {
    fn drop(&mut self) {
        let base = *self.base.get_mut();
        if base != 0 {
            release_window(base, self.size as u64);
        }
    }
}

/// Where the first IPA window starts: above 0, so a base of 0 always means
/// "no window".
const IPA_FIRST: u64 = 1 << 30;
/// IPA windows are aligned to 2 MiB (a hardware block size, and a multiple of
/// the 16 KiB `hv_vm_map` wants).
const IPA_ALIGN: u64 = 2 << 20;

/// The IPA windows in use: start → length, sorted.
static IPA_WINDOWS: Mutex<BTreeMap<u64, u64>> = Mutex::new(BTreeMap::new());

/// Called with `(base, len)` as an arm64 pool's IPA window is given up, before
/// its memory goes: how the hypervisor backend learns to unmap it.
static WINDOW_RELEASE: OnceLock<fn(u64, u64)> = OnceLock::new();

/// Register the function told when an IPA window is given up (first caller
/// wins; the hypervisor backend sets it once, when it creates its VM).
pub(crate) fn on_window_release(hook: fn(u64, u64)) {
    let _ = WINDOW_RELEASE.set(hook);
}

/// Take the lowest free IPA window of `len` bytes.
fn ipa_window(len: u64) -> u64 {
    let len = len.next_multiple_of(IPA_ALIGN);
    let mut windows = IPA_WINDOWS.lock().unwrap();
    let mut at = IPA_FIRST;
    for (&start, &size) in windows.iter() {
        if start >= at + len {
            break;
        }
        at = (start + size).next_multiple_of(IPA_ALIGN);
    }
    windows.insert(at, len);
    at
}

/// Give back the IPA window at `base`, telling the hypervisor backend first.
fn release_window(base: u64, len: u64) {
    if let Some(hook) = WINDOW_RELEASE.get() {
        hook(base, len);
    }
    IPA_WINDOWS.lock().unwrap().remove(&base);
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
    /// Frames ordinary allocations leave free, so that tearing down an address
    /// space, which needs a few frames to free many, can't be starved by a guest
    /// that filled the pool.
    reserve: usize,
    /// Whether allocations may currently dip into [`Self::reserve`].
    use_reserve: bool,
    /// The memory cgroup each live frame is charged to (an index into
    /// [`Self::memcgs`]; 0, the root, for most).
    owner: Vec<u16>,
    /// Memory cgroups: what each has charged, and its limits.
    memcgs: Vec<Memcg>,
    /// The memory cgroup new frames are charged to: the one of the address
    /// space allocating, set as it takes the allocator.
    pub charge_to: u16,
    /// The memory cgroup whose `memory.max` last refused a frame, until
    /// [`Self::take_memcg_oom`] collects it.
    oom: Option<u16>,
    /// Bit `slot % 64` is set while memory cgroup `slot` has a batch charged
    /// past `memory.high` waiting for [`Self::take_over_high`]; shared outside
    /// the lock so an interpreter can notice and hand the task to the kernel.
    pub due: Arc<AtomicU64>,
    /// The page cache, as far as memory cgroups see it (see [`FileCache`]).
    cache: FileCache,
}

/// The file pages the guest has read or written on its disks, as Linux's page
/// cache would hold them: each charged to the memory cgroup that brought it in
/// (it counts in `memory.current`, as `file` in `memory.stat`), and reclaimed —
/// oldest first, as memory cgroup limits, protections and the pool's size have
/// it (see [`FrameAllocator::reclaim`]). The data itself stays on the host,
/// which caches it; what's kept here is which pages are "in memory", which is
/// also what tells a read that has to go to the disk (and count as I/O for the
/// `io` controller) from one served from memory.
#[derive(Debug, Default)]
struct FileCache {
    /// Each file's cached pages (file: its disk and inode), and for each the
    /// memory cgroup it's charged to and when it was last used.
    files: HashMap<FileId, BTreeMap<u64, (u16, u64)>>,
    /// Per memory cgroup, its own cached pages, least recently used first.
    lru: Vec<BTreeMap<u64, (FileId, u64)>>,
    /// The last use handed out.
    clock: u64,
    /// Pages cached in all.
    pages: u64,
    /// Where the next reclaim starts among the memory cgroups, so none is
    /// always the first to give pages up (Linux's reclaim cookie).
    cursor: usize,
}

/// A cached file: its disk's device number and its inode there.
pub type FileId = (u64, u64);

/// Linux's `DEF_PRIORITY`: reclaim scans `1 >> priority` of each list, from
/// this down to all of it.
const DEF_PRIORITY: u32 = 12;
/// Linux's `SWAP_CLUSTER_MAX`: the batch reclaim works in.
const RECLAIM_BATCH: u64 = 32;

/// Pages Linux charges a memory cgroup at a time (`MEMCG_CHARGE_BATCH`): the
/// unit a `memory.high` penalty is reckoned in.
pub const MEMCG_CHARGE_BATCH: u64 = 64;

/// One memory cgroup as the allocator keeps it (see [`FrameAllocator::charge_to`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Memcg {
    /// The memory cgroup above, whose use includes this one's (0 at the top).
    pub parent: u16,
    /// Frames charged here and below.
    pub frames: u64,
    /// The most there have been.
    pub peak: u64,
    /// `memory.max` in frames: a frame past it is refused.
    pub max: u64,
    /// `memory.high` in frames: going past it is counted.
    pub high: u64,
    /// How often a frame was refused at `max` (`memory.events` `max`).
    pub max_events: u64,
    /// How often an allocation went past `high`.
    pub high_events: u64,
    /// File pages cached here and below (also in [`Self::frames`]).
    pub file: u64,
    /// `memory.min` and `memory.low` in pages: what reclaim leaves alone.
    pub min: u64,
    pub low: u64,
    /// How often its use below `memory.low` was reclaimed anyway, since
    /// nothing else could be (`memory.events` `low`).
    pub low_events: u64,
    /// Frames charged here while this or a memory cgroup above was over its
    /// `high`, not yet paid for (see [`FrameAllocator::take_over_high`]).
    pub over_high: u64,
    /// The slot is in use.
    pub live: bool,
}

impl Memcg {
    const ROOT: Self = Self {
        parent: 0,
        frames: 0,
        peak: 0,
        max: u64::MAX,
        high: u64::MAX,
        max_events: 0,
        high_events: 0,
        file: 0,
        min: 0,
        low: 0,
        low_events: 0,
        over_high: 0,
        live: true,
    };
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
            reserve: 0,
            use_reserve: false,
            owner: vec![0; nframes],
            memcgs: vec![Memcg::ROOT],
            charge_to: 0,
            oom: None,
            due: Arc::new(AtomicU64::new(0)),
            cache: FileCache::default(),
        }
    }

    /// The memory cgroups from `slot` up to the root (which isn't included:
    /// it has no limit, and its use is the pool's).
    fn memcg_chain(&self, slot: u16) -> impl Iterator<Item = usize> + '_ {
        let mut at = slot;
        std::iter::from_fn(move || {
            if at == 0 {
                return None;
            }
            let here = at as usize;
            at = self.memcgs.get(here).map_or(0, |m| m.parent);
            Some(here)
        })
    }

    /// Charge one frame to `slot` and those above it, unless that would pass a
    /// `memory.max` (counted, and remembered for [`Self::take_memcg_oom`]).
    /// Allocations made from the reserve, to tear an address space down, are
    /// never refused.
    fn charge(&mut self, slot: u16) -> bool {
        self.charge_page(slot, true)
    }

    /// [`Self::charge`], for a frame (`oom`: a refusal is an out-of-memory
    /// event) or a file page (not cached instead). A memory cgroup at its
    /// `memory.max` first reclaims what it can, as Linux's `try_charge` does.
    fn charge_page(&mut self, slot: u16, oom: bool) -> bool {
        let chain: Vec<usize> = self.memcg_chain(slot).collect();
        if !self.use_reserve {
            for &m in &chain {
                if self.memcgs[m].frames >= self.memcgs[m].max {
                    let over = self.memcgs[m].frames + 1 - self.memcgs[m].max;
                    self.reclaim(m as u16, over.max(RECLAIM_BATCH));
                }
            }
            if let Some(&full) = chain
                .iter()
                .find(|&&m| self.memcgs[m].frames >= self.memcgs[m].max)
            {
                self.memcgs[full].max_events += 1;
                if oom {
                    self.oom = Some(full as u16);
                }
                return false;
            }
        }
        let mut over = false;
        for m in chain {
            let g = &mut self.memcgs[m];
            g.frames += 1;
            g.peak = g.peak.max(g.frames);
            if g.frames > g.high {
                g.high_events += 1;
                over = true;
            }
        }
        if over {
            let g = &mut self.memcgs[slot as usize];
            g.over_high += 1;
            if g.over_high == MEMCG_CHARGE_BATCH {
                self.due.fetch_or(1 << (slot % 64), Ordering::Relaxed);
            }
        }
        true
    }

    /// Once [`MEMCG_CHARGE_BATCH`] or more frames have been charged to `slot` past a
    /// `memory.high` (its own or one above), take them: how many, and how far
    /// over its `high` the worst memory cgroup up the chain is now, as Linux's
    /// `calculate_overage` puts it (the excess over `high`, `<< 20`).
    pub fn take_over_high(&mut self, slot: u16) -> Option<(u64, u64)> {
        let pages = self.memcgs.get(slot as usize)?.over_high;
        if pages < MEMCG_CHARGE_BATCH {
            return None;
        }
        self.memcgs[slot as usize].over_high = 0;
        self.due.fetch_and(!(1 << (slot % 64)), Ordering::Relaxed);
        // Linux's `reclaim_high`: what's over each `memory.high` up the chain is
        // reclaimed first, and only what's left over is slept off.
        let chain: Vec<usize> = self.memcg_chain(slot).collect();
        for m in chain {
            let g = self.memcgs[m];
            if g.frames > g.high {
                self.reclaim(m as u16, g.frames - g.high);
            }
        }
        let overage = self
            .memcg_chain(slot)
            .map(|m| &self.memcgs[m])
            .filter(|g| g.frames > g.high)
            .map(|g| ((g.frames - g.high) << 20) / g.high.max(1))
            .max()
            .unwrap_or(0);
        Some((pages, overage))
    }

    fn uncharge(&mut self, slot: u16) {
        let chain: Vec<usize> = self.memcg_chain(slot).collect();
        for m in chain {
            let g = &mut self.memcgs[m];
            g.frames = g.frames.saturating_sub(1);
        }
    }

    /// A new memory cgroup below `parent`, or `None` if there are no slots
    /// left. A slot is reused once whatever was charged to it is gone.
    pub fn memcg_new(&mut self, parent: u16) -> Option<u16> {
        let fresh = Memcg {
            parent,
            ..Memcg::ROOT
        };
        if let Some(i) =
            (1..self.memcgs.len()).find(|&i| !self.memcgs[i].live && self.memcgs[i].frames == 0)
        {
            self.memcgs[i] = fresh;
            return u16::try_from(i).ok();
        }
        let i = u16::try_from(self.memcgs.len()).ok()?;
        self.memcgs.push(fresh);
        Some(i)
    }

    /// Reclaim file pages until `n` more pages fit in the pool beside those
    /// cached, as the guest's memory filling up makes Linux reclaim from every
    /// memory cgroup (respecting their protections).
    fn make_room(&mut self, n: u64) {
        let room = (self.free_list.len() as u64).saturating_sub(self.reserve as u64);
        if self.cache.pages + n > room {
            self.reclaim(0, (self.cache.pages + n - room).max(RECLAIM_BATCH));
        }
    }

    /// Note that pages `[first, first + n)` of file `file` were just read
    /// (`write`: written) by a task of memory cgroup `slot`, and say how many
    /// of them weren't in memory — what a read had to fetch from the disk. A
    /// page already cached is used again; one that isn't is cached and charged
    /// to `slot`, after reclaiming what its limits need, or left uncached if
    /// they can't be met.
    pub fn cache_access(&mut self, file: FileId, first: u64, n: u64, slot: u16) -> u64 {
        let mut missed = 0;
        for page in first..first.saturating_add(n) {
            self.cache.clock += 1;
            let now = self.cache.clock;
            if let Some((owner, used)) = self
                .cache
                .files
                .get_mut(&file)
                .and_then(|pages| pages.get_mut(&page))
            {
                let lru = &mut self.cache.lru[*owner as usize];
                lru.remove(used);
                lru.insert(now, (file, page));
                *used = now;
                continue;
            }
            missed += 1;
            self.make_room(1);
            if self.cache.pages as usize + self.allocated + self.reserve >= self.refcount.len()
                || !self.charge_page(slot, false)
            {
                continue;
            }
            let chain: Vec<usize> = self.memcg_chain(slot).collect();
            for m in chain {
                self.memcgs[m].file += 1;
            }
            let slot = slot as usize;
            if self.cache.lru.len() <= slot {
                self.cache.lru.resize_with(slot + 1, BTreeMap::new);
            }
            self.cache.lru[slot].insert(now, (file, page));
            self.cache
                .files
                .entry(file)
                .or_default()
                .insert(page, (slot as u16, now));
            self.cache.pages += 1;
        }
        missed
    }

    /// Drop file `file`'s cached pages from `first` on (all of them for 0): it
    /// was truncated, deleted, or `POSIX_FADV_DONTNEED` was asked for.
    pub fn cache_drop(&mut self, file: FileId, first: u64, end: u64) {
        let Some(pages) = self.cache.files.get_mut(&file) else {
            return;
        };
        let gone: Vec<(u64, (u16, u64))> = pages.range(first..end).map(|(&p, &v)| (p, v)).collect();
        for &(page, _) in &gone {
            pages.remove(&page);
        }
        if pages.is_empty() {
            self.cache.files.remove(&file);
        }
        for (_, (owner, used)) in gone {
            self.cache.lru[owner as usize].remove(&used);
            self.cache.pages -= 1;
            self.uncharge_file(owner);
        }
    }

    /// Pages the page cache holds in all.
    #[must_use]
    pub fn cached_pages(&self) -> u64 {
        self.cache.pages
    }

    fn uncharge_file(&mut self, slot: u16) {
        self.uncharge(slot);
        let chain: Vec<usize> = self.memcg_chain(slot).collect();
        for m in chain {
            self.memcgs[m].file = self.memcgs[m].file.saturating_sub(1);
        }
    }

    /// Evict memory cgroup `slot`'s `n` least recently used cached pages (or
    /// all it has); how many went.
    fn evict(&mut self, slot: u16, n: u64) -> u64 {
        let mut done = 0;
        while done < n {
            let Some((used, (file, page))) = self
                .cache
                .lru
                .get_mut(slot as usize)
                .and_then(BTreeMap::pop_first)
            else {
                break;
            };
            let _ = used;
            if let Some(pages) = self.cache.files.get_mut(&file) {
                pages.remove(&page);
                if pages.is_empty() {
                    self.cache.files.remove(&file);
                }
            }
            self.cache.pages -= 1;
            self.uncharge_file(slot);
            done += 1;
        }
        done
    }

    /// Reclaim at least `target` cached pages from memory cgroup `root` and
    /// below (0: the whole pool), as Linux's `shrink_node_memcgs` does: each
    /// memory cgroup is scanned in proportion to what it holds unprotected —
    /// one at or under its effective `memory.min` not at all, one under its
    /// `memory.low` only if nothing else would do (counted as a `low` event)
    /// — `1 >> priority` of it a pass, the priority falling until enough is
    /// found. Only the page cache can be reclaimed: there's no swap. Returns
    /// how many pages went.
    pub fn reclaim(&mut self, root: u16, target: u64) -> u64 {
        let order = self.subtree_of(root);
        let prot = self.protection(root, &order);
        let mut reclaimed = 0;
        for low_reclaim in [false, true] {
            let mut skipped = false;
            for priority in (0..=DEF_PRIORITY).rev() {
                let start = self.cache.cursor % order.len();
                for k in 0..order.len() {
                    let m = order[(start + k) % order.len()];
                    let (emin, elow) = prot[(start + k) % order.len()];
                    let usage = self.memcgs[m as usize].frames;
                    if m != root && usage <= emin {
                        continue;
                    }
                    if m != root && usage <= elow {
                        if !low_reclaim {
                            skipped = true;
                            continue;
                        }
                        self.memcgs[m as usize].low_events += 1;
                    }
                    let lru = self.cache.lru.get(m as usize).map_or(0, |l| l.len() as u64);
                    if lru == 0 {
                        continue;
                    }
                    // Linux's `apply_proportional_protection`.
                    let protection = if low_reclaim { emin } else { emin.max(elow) };
                    let mut scan = lru;
                    if protection > 0 {
                        scan = lru - lru * protection.min(usage) / (usage + 1);
                        scan = scan.max(RECLAIM_BATCH.min(lru));
                    }
                    // Never below what still protects it.
                    let floor = if m == root { 0 } else { protection };
                    let scan = (scan >> priority)
                        .min(target.saturating_sub(reclaimed).max(1))
                        .min(usage.saturating_sub(floor));
                    reclaimed += self.evict(m, scan);
                    if reclaimed >= target {
                        self.cache.cursor = self.cache.cursor.wrapping_add(1);
                        return reclaimed;
                    }
                }
            }
            self.cache.cursor = self.cache.cursor.wrapping_add(1);
            if !skipped {
                break;
            }
        }
        reclaimed
    }

    /// Memory cgroup `root` and those below it, parents first.
    fn subtree_of(&self, root: u16) -> Vec<u16> {
        let mut out = vec![root];
        let mut i = 0;
        while i < out.len() {
            let p = out[i];
            out.extend(
                (1..self.memcgs.len())
                    .filter(|&c| self.memcgs[c].live || self.memcgs[c].frames > 0)
                    .filter(|&c| self.memcgs[c].parent == p && c as u16 != p)
                    .map(|c| c as u16),
            );
            i += 1;
        }
        out
    }

    /// Each of `order`'s (a subtree, parents first) effective `memory.min`
    /// and `memory.low` for a reclaim from `root`, as Linux's
    /// `mem_cgroup_calculate_protection` works them out: the reclaim's root
    /// has none, those right below it what they're set to, and those further
    /// down what they're set to (no more than they use) — scaled down
    /// together when they claim more than their parent's own is.
    fn protection(&self, root: u16, order: &[u16]) -> Vec<(u64, u64)> {
        let mut eff: HashMap<u16, (u64, u64)> = HashMap::new();
        let used = |m: u16| self.memcgs[m as usize].frames;
        // What each parent's children claim (their use up to their setting).
        let mut claimed: HashMap<u16, (u64, u64)> = HashMap::new();
        for &m in &order[1..] {
            let g = &self.memcgs[m as usize];
            let c = claimed.entry(g.parent).or_default();
            c.0 += used(m).min(g.min);
            c.1 += used(m).min(g.low);
        }
        for &m in order {
            let g = &self.memcgs[m as usize];
            let e = if m == root {
                (0, 0)
            } else if g.parent == root {
                (g.min, g.low)
            } else {
                let parent = eff.get(&g.parent).copied().unwrap_or((0, 0));
                let siblings = claimed.get(&g.parent).copied().unwrap_or((0, 0));
                let scaled = |setting: u64, parent_eff: u64, siblings: u64| {
                    let protected = used(m).min(setting);
                    if siblings > parent_eff && siblings > 0 {
                        protected * parent_eff / siblings
                    } else {
                        protected
                    }
                };
                (
                    scaled(g.min, parent.0, siblings.0),
                    scaled(g.low, parent.1, siblings.1),
                )
            };
            eff.insert(m, e);
        }
        order.iter().map(|m| eff[m]).collect()
    }

    /// Reclaim memory cgroup `slot` down to its `memory.max` if it can (after
    /// the limit was lowered), and say whether it's still over.
    pub fn shrink_to_max(&mut self, slot: u16) -> bool {
        let Some(g) = self.memcgs.get(slot as usize).copied() else {
            return false;
        };
        if g.frames > g.max {
            self.reclaim(slot, g.frames - g.max);
        }
        self.memcgs[slot as usize].frames > self.memcgs[slot as usize].max
    }

    /// Reclaim up to `pages` from memory cgroup `slot` and below (a write to
    /// `memory.reclaim`); how many went.
    pub fn reclaim_pages(&mut self, slot: u16, pages: u64) -> u64 {
        self.reclaim(slot, pages)
    }

    /// Memory cgroup `slot` is gone. What's still charged to it stays counted
    /// above it until freed (Linux moves it to the parent; the totals are the
    /// same).
    pub fn memcg_remove(&mut self, slot: u16) {
        if let Some(m) = self.memcgs.get_mut(slot as usize).filter(|_| slot != 0) {
            m.live = false;
            m.max = u64::MAX;
            m.high = u64::MAX;
        }
    }

    /// Set memory cgroup `slot`'s `memory.max` and `memory.high`, in frames.
    pub fn memcg_set_limits(&mut self, slot: u16, max: u64, high: u64) {
        if let Some(m) = self.memcgs.get_mut(slot as usize).filter(|_| slot != 0) {
            m.max = max;
            m.high = high;
        }
    }

    /// Set memory cgroup `slot`'s `memory.low` and `memory.min`, in pages.
    pub fn memcg_set_protection(&mut self, slot: u16, low: u64, min: u64) {
        if let Some(m) = self.memcgs.get_mut(slot as usize).filter(|_| slot != 0) {
            m.low = low;
            m.min = min;
        }
    }

    /// Reset memory cgroup `slot`'s peak to what it uses now (a write to
    /// `memory.peak`).
    pub fn memcg_reset_peak(&mut self, slot: u16) {
        if let Some(m) = self.memcgs.get_mut(slot as usize) {
            m.peak = m.frames;
        }
    }

    /// Memory cgroup `slot` as it stands.
    #[must_use]
    pub fn memcg(&self, slot: u16) -> Option<Memcg> {
        self.memcgs.get(slot as usize).copied()
    }

    /// The memory cgroup whose `memory.max` refused a frame since the last
    /// call, if one did.
    pub fn take_memcg_oom(&mut self) -> Option<u16> {
        self.oom.take()
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
        self.make_room(1);
        if !self.use_reserve && self.free_list.len() <= self.reserve {
            return None;
        }
        if self.free_list.is_empty() || !self.charge(self.charge_to) {
            return None;
        }
        let f = self.free_list.pop()?;
        debug_assert_eq!(self.refcount[f as usize], 0, "free-list frame was not free");
        self.refcount[f as usize] = 1;
        self.owner[f as usize] = self.charge_to;
        self.allocated += 1;
        let pa = u64::from(f) * FRAME;
        phys.zero_frame(pa);
        Some(pa)
    }

    /// Keep `frames` free for allocations made under [`Self::set_use_reserve`].
    pub fn set_reserve(&mut self, frames: usize) {
        self.reserve = frames;
    }

    /// Let allocations dip into the reserve, or stop them doing so.
    pub fn set_use_reserve(&mut self, on: bool) {
        self.use_reserve = on;
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
            let owner = std::mem::take(&mut self.owner[f]);
            self.uncharge(owner);
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
                let owner = std::mem::take(&mut self.owner[f]);
                self.uncharge(owner);
            }
            self.refcount[f] = PINNED;
        }
        // A pinned frame must not sit in the free list.
        self.free_list
            .retain(|&f| self.refcount[f as usize] != PINNED);
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

impl crate::fs::cgroupfs::MemcgHost for FrameAllocator {
    fn memcg_create(&mut self, parent: u16) -> Option<u16> {
        self.memcg_new(parent)
    }
    fn memcg_drop(&mut self, slot: u16) {
        self.memcg_remove(slot);
    }
    fn memcg_limits(&mut self, slot: u16, max: u64, high: u64, low: u64, min: u64) {
        self.memcg_set_limits(slot, max, high);
        self.memcg_set_protection(slot, low, min);
    }
    fn memcg_use(&self, slot: u16) -> crate::fs::cgroupfs::MemUse {
        self.memcg(slot)
            .map(|m| crate::fs::cgroupfs::MemUse {
                current: m.frames * FRAME,
                peak: m.peak * FRAME,
                file: m.file * FRAME,
                high_events: m.high_events,
                max_events: m.max_events,
                low_events: m.low_events,
            })
            .unwrap_or_default()
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
    fn ordinary_allocations_leave_the_reserve_alone() {
        let (phys, mut fa) = pool(16);
        let total = fa.free_count();
        fa.set_reserve(4);
        for _ in 0..total - 4 {
            assert!(fa.alloc(&phys).is_some());
        }
        assert_eq!(fa.alloc(&phys), None, "stops at the reserve");
        fa.set_use_reserve(true);
        for _ in 0..4 {
            assert!(fa.alloc(&phys).is_some());
        }
        assert_eq!(fa.alloc(&phys), None, "and then it really is empty");
    }

    #[test]
    fn pool_geometry_is_frame_rounded_and_based_at_zero() {
        let phys = PhysMem::new(FRAME_SZ + 1);
        assert_eq!(phys.base(), 0);
        assert_eq!(phys.len() % FRAME_SZ, 0, "whole number of frames");
        assert!(phys.len() >= 2 * FRAME_SZ, "rounded up past one frame");
        assert_ne!(phys.len(), 0);
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
        let (phys, mut alloc) = pool(8);
        // Pre-dirty the whole pool so alloc must actively zero.
        for f in 0..8u64 {
            phys.write(f * FRAME, &[0xCC; 8]);
        }
        let a = alloc.alloc(&phys).unwrap();
        let b = alloc.alloc(&phys).unwrap();
        assert_ne!(a, b, "distinct frames");
        assert_ne!(a, 0, "frame 0 never handed out");
        assert_ne!(b, 0);
        assert!(
            a.is_multiple_of(FRAME) && b.is_multiple_of(FRAME),
            "frame-aligned"
        );
        assert_eq!(phys.read_u64(a), 0, "handed-out frame is zeroed");
        assert_eq!(phys.read_u64(b), 0);
        assert_eq!(alloc.refcount(a), 1);
        assert_eq!(alloc.alloc_count(), 2);
    }

    #[test]
    fn free_returns_frame_to_the_pool_and_it_is_reused() {
        let (phys, mut alloc) = pool(4);
        let a = alloc.alloc(&phys).unwrap();
        let free_before = alloc.free_count();
        alloc.free(a);
        assert_eq!(alloc.refcount(a), 0, "freed");
        assert_eq!(alloc.free_count(), free_before + 1);
        assert_eq!(alloc.alloc_count(), 0);
        // The very next alloc reuses the just-freed frame (LIFO).
        let b = alloc.alloc(&phys).unwrap();
        assert_eq!(a, b, "freed frame is reused");
    }

    #[test]
    fn memory_cgroups_are_charged_and_limited() {
        let (phys, mut fa) = pool(64);
        let a = fa.memcg_new(0).unwrap();
        let b = fa.memcg_new(a).unwrap();
        fa.memcg_set_limits(a, 3, 2);
        fa.charge_to = b;
        let f1 = fa.alloc(&phys).unwrap();
        let f2 = fa.alloc(&phys).unwrap();
        fa.charge_to = a;
        let f3 = fa.alloc(&phys).unwrap();
        assert_eq!(fa.memcg(a).unwrap().frames, 3, "b's count in a's");
        assert_eq!(fa.memcg(b).unwrap().frames, 2);
        assert_eq!(fa.memcg(a).unwrap().high_events, 1);
        fa.charge_to = b;
        assert_eq!(fa.alloc(&phys), None, "a's memory.max");
        assert_eq!(fa.take_memcg_oom(), Some(a));
        assert_eq!(fa.take_memcg_oom(), None);
        assert_eq!(fa.memcg(a).unwrap().max_events, 1);
        // A shared frame is charged once, to whoever allocated it, until its
        // last reference goes.
        fa.incref(f1);
        fa.free(f1);
        assert_eq!(fa.memcg(b).unwrap().frames, 2);
        fa.free(f1);
        assert_eq!(fa.memcg(b).unwrap().frames, 1);
        assert!(fa.alloc(&phys).is_some(), "room again");
        // Tearing down never fails on a limit.
        fa.set_use_reserve(true);
        assert!(fa.alloc(&phys).is_some());
        fa.set_use_reserve(false);
        assert_eq!(fa.memcg(a).unwrap().peak, 4);
        // A removed group's slot comes back once nothing is charged to it.
        fa.memcg_remove(b);
        assert_ne!(fa.memcg_new(a), Some(b));
        fa.free(f2);
        fa.free(f3);
        let _ = f3;
    }

    #[test]
    fn frames_charged_past_memory_high_are_handed_over_by_the_batch() {
        let (phys, mut fa) = pool(256);
        let a = fa.memcg_new(0).unwrap();
        let b = fa.memcg_new(a).unwrap();
        fa.memcg_set_limits(a, u64::MAX, 100);
        fa.charge_to = b;
        for _ in 0..100 {
            fa.alloc(&phys).unwrap();
        }
        assert_eq!(fa.take_over_high(b), None, "at memory.high, not past it");
        let bit = 1 << (b % 64);
        for _ in 0..MEMCG_CHARGE_BATCH - 1 {
            fa.alloc(&phys).unwrap();
        }
        assert_eq!(fa.due.load(Ordering::Relaxed) & bit, 0);
        assert_eq!(fa.take_over_high(b), None, "less than a batch");
        fa.alloc(&phys).unwrap();
        assert_ne!(fa.due.load(Ordering::Relaxed) & bit, 0, "a batch is due");
        // 164 frames against a's 100: 64% over, as a fraction << 20.
        assert_eq!(fa.take_over_high(b), Some((64, (64 << 20) / 100)));
        assert_eq!(fa.due.load(Ordering::Relaxed) & bit, 0);
        assert_eq!(fa.take_over_high(b), None);
    }

    #[test]
    fn the_page_cache_is_charged_and_reclaimed_as_protections_say() {
        let (phys, mut fa) = pool(4096);
        let parent = fa.memcg_new(0).unwrap();
        let a = fa.memcg_new(parent).unwrap();
        let b = fa.memcg_new(parent).unwrap();
        fa.memcg_set_limits(parent, 1000, u64::MAX);
        fa.memcg_set_protection(a, 600, 0);
        // A cached page is charged once, and a read of it again is a hit.
        assert_eq!(fa.cache_access((1, 7), 0, 500, a), 500);
        assert_eq!(fa.cache_access((1, 7), 0, 500, a), 0, "all cached");
        assert_eq!(fa.memcg(a).unwrap().file, 500);
        assert_eq!(fa.memcg(parent).unwrap().frames, 500);
        // The sibling reading past the parent's max reclaims its own pages,
        // not the protected ones.
        assert_eq!(fa.cache_access((1, 8), 0, 2000, b), 2000);
        assert_eq!(fa.memcg(a).unwrap().frames, 500, "memory.low kept them");
        assert!(fa.memcg(parent).unwrap().frames <= 1000);
        assert_eq!(fa.memcg(a).unwrap().low_events, 0);
        // An anonymous frame past the max takes cache rather than fail.
        fa.charge_to = b;
        assert!(fa.alloc(&phys).is_some());
        // With nothing unprotected left, protection below memory.low gives
        // (counted), but memory.min never does.
        fa.memcg_set_protection(a, 600, 300);
        assert!(fa.reclaim(parent, 10_000) > 0);
        assert_eq!(fa.memcg(a).unwrap().frames, 300, "down to memory.min");
        assert!(fa.memcg(a).unwrap().low_events > 0);
        // Dropping a file's pages uncharges them.
        fa.cache_drop((1, 7), 0, u64::MAX);
        assert_eq!(fa.memcg(a).unwrap().file, 0);
        assert_eq!(fa.memcg(a).unwrap().frames, 0);
    }

    #[test]
    fn incref_decref_frees_only_at_zero() {
        let (phys, mut alloc) = pool(4);
        let a = alloc.alloc(&phys).unwrap(); // rc 1
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
        let (phys, mut alloc) = pool(3);
        let allocatable = phys.nframes() as usize - 1; // all but the null frame
        assert_eq!(alloc.refcount(0), PINNED, "frame 0 pinned");
        // Drain the pool; frame 0 must never appear.
        let mut seen = Vec::new();
        while let Some(pa) = alloc.alloc(&phys) {
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
        let (phys, mut alloc) = pool(3);
        // Drain every allocatable frame, then confirm a full pool yields None.
        while alloc.alloc(&phys).is_some() {}
        assert_eq!(alloc.free_count(), 0);
        assert!(alloc.alloc(&phys).is_none(), "exhausted -> None");
        assert!(alloc.alloc(&phys).is_none(), "still None, no panic");
    }

    #[test]
    fn reserve_holds_a_range_out_of_circulation() {
        let (phys, mut alloc) = pool(8);
        // Reserve frames covering [FRAME, 3*FRAME) => frames 1 and 2.
        alloc.reserve(FRAME, 2 * FRAME);
        assert_eq!(alloc.refcount(FRAME), PINNED);
        assert_eq!(alloc.refcount(2 * FRAME), PINNED);
        // Drain and confirm reserved frames never come back.
        let mut seen = Vec::new();
        while let Some(pa) = alloc.alloc(&phys) {
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

    #[test]
    fn arm64_pools_get_disjoint_ipa_windows_given_back_on_drop() {
        let a = PhysMem::new(3 << 20);
        let b = PhysMem::new(1 << 20);
        a.set_format(PtFormat::Aarch64);
        b.set_format(PtFormat::Aarch64);
        let (ab, bb) = (a.base(), b.base());
        assert!(ab != 0 && bb != 0);
        assert_eq!(ab % IPA_ALIGN, 0);
        assert!(
            ab + a.len() as u64 <= bb || bb + b.len() as u64 <= ab,
            "windows overlap: {ab:#x}+{:#x} and {bb:#x}+{:#x}",
            a.len(),
            b.len()
        );
        drop(a);
        assert!(!IPA_WINDOWS.lock().unwrap().contains_key(&ab), "given back");
        assert!(IPA_WINDOWS.lock().unwrap().contains_key(&bb));
        b.set_format(PtFormat::X86_64);
        assert_eq!(b.base(), 0, "x86-64 tables address the pool from 0");
        assert!(!IPA_WINDOWS.lock().unwrap().contains_key(&bb));
    }
}
