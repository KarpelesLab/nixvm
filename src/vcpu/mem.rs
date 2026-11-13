//! Guest memory: one process's virtual address space over the shared pool.
//!
//! Phase 3 of the MMU refactor replaced the flat, identity-mapped backing with a
//! *real* MMU model. Guest RAM is one shared pool of physical frames
//! (`super::phys::PhysMem`) registered as a single KVM memslot; each process
//! owns a 4-level page-table tree (`super::pagetable::AddrSpace`, its own
//! `CR3`/`TTBR0`) over that pool, in its guest architecture's descriptor format
//! ([`GuestMemory::set_guest_arch`]). A [`GuestMemory`] bundles the two together plus the
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
//! the child (via `AddrSpace::fork_cow`); the first write on either side
//! privatizes the frame ([`GuestMemory::cow_fault`], or transparently inside
//! `write`/`write_init`). Only touched pages ever consume a private frame.

/// Note a page-table edit that can leave a stale TLB entry (fields only, so
/// it works while a lock guard borrows another part of the memory).
macro_rules! mark_tlb_dirty {
    ($m:expr) => {{
        $m.tlb_dirty = true;
        $m.tlb_gen = $m.tlb_gen.wrapping_add(1);
    }};
}

use super::ctrl;
use super::filemap::{FileMap, FileVma, PageError, Region, SHRINK_EPOCH};
use super::pagetable::{AddrSpace, Enc, PtFormat};
use super::pagevec::PageVec;
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
    /// The page maps a part of a file past its end: `SIGBUS` for the guest,
    /// `EFAULT` for a syscall.
    Bus(u64),
}

impl MemError {
    /// The guest address the fault is attributed to. For page-granular faults
    /// (`Unmapped`, `Protection`) this is the base of the *offending* page.
    #[must_use]
    pub fn fault_addr(&self) -> u64 {
        match self {
            MemError::OutOfBounds(a) | MemError::Unmapped(a) | MemError::Bus(a) => *a,
            MemError::Protection { addr, .. } => *addr,
            MemError::Host(_) => 0,
        }
    }
}

/// The guest address space for one process. Not `Clone` — use [`GuestMemory::fork`].
/// A process's heap: where it starts, the current program break (`brk`), and
/// the highest address the break may reach.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Heap {
    pub start: u64,
    pub brk: u64,
    pub limit: u64,
}

impl Heap {
    /// A heap starting (empty) at `start`, growing up to `limit`.
    #[must_use]
    pub fn new(start: u64, limit: u64) -> Self {
        Self {
            start,
            brk: start,
            limit,
        }
    }
}

pub struct GuestMemory {
    base: u64,
    size: u64,
    /// The shared physical-RAM pool (one KVM memslot). Cloned across `fork`/
    /// `exec` so every address space and the memslot see the same frames.
    phys: Arc<PhysMem>,
    /// The shared frame allocator, locked briefly for map/unmap/fork/CoW.
    fa: Arc<Mutex<FrameAllocator>>,
    /// The allocator's [`FrameAllocator::due`], readable without its lock.
    due: Arc<std::sync::atomic::AtomicU64>,
    /// This process's page tables (its `CR3`).
    space: AddrSpace,
    /// Per-page *intended* protection (what `map`/`protect` requested), meaningful
    /// only when `mapped[i]`. The page-table leaf may be more restrictive than
    /// this — a copy-on-write-shared writable page reads `Prot::rw()` here but is
    /// installed read-only until privatized — so this is the authority on "was
    /// this page meant to be writable?" that [`GuestMemory::cow_fault`] needs.
    prot: PageVec<Prot>,
    mapped: PageVec<bool>,
    /// How many entries of [`Self::mapped`] are set.
    mapped_count: usize,
    /// Per-page: loaded from a file (an ELF segment). `MADV_DONTNEED` preserves
    /// these; cleared when a page is re-`map`ped anonymously.
    file_backed: PageVec<bool>,
    /// Per-page `MAP_SHARED | MAP_ANONYMOUS` marker. These pages back a shared
    /// anonymous region: they are eagerly frame-backed at `mmap`, and across
    /// `fork` the child *aliases* the same frame writable (not copy-on-write),
    /// so parent and child see each other's stores — the standard shared-memory
    /// IPC primitive. `make_writable`/`leaf_prot` therefore never privatize them.
    shared_anon: PageVec<bool>,
    /// Pages with a `madvise` fork policy (sparse: almost always empty):
    /// [`FORK_DONT`] (`MADV_DONTFORK` — absent from a fork child) or
    /// [`FORK_WIPE`] (`MADV_WIPEONFORK` — zero-filled in a fork child).
    fork_policy: std::collections::BTreeMap<usize, u8>,
    /// Pages locked by `mlock`/`mlockall`, what `RLIMIT_MEMLOCK` charges. Not
    /// inherited by a fork child and dropped by `execve`, as on Linux.
    locks: super::pagelocks::PageLocks,
    /// Set whenever a *present* page-table leaf is cleared or changed from the
    /// host (unmap, protect, copy-on-write privatize). A KVM vcpu running this
    /// address space must flush its TLB before its next run, or a stale entry
    /// would keep serving the old (now-unmapped or reused) frame. Consumed by
    /// [`GuestMemory::take_tlb_dirty`]; irrelevant to the interpreter (no TLB).
    tlb_dirty: bool,
    /// Bumped with every such page-table edit ([`GuestMemory::tlb_generation`]):
    /// each vcpu of this address space — every thread's — flushes its TLB
    /// before running once this has moved since its last flush, not only the
    /// vcpu whose syscall made the edit.
    tlb_gen: u64,
    /// The program break and the window it may move in: the address space's,
    /// so a process's threads share it (as on Linux) and a fork copies it.
    pub heap: Heap,
    /// How many times this address space was handed a new program
    /// (`execve`): an SMP worker holding a thread from before checks it
    /// rather than run the old image's thread in the new image.
    pub exec_gen: u64,
    /// Set whenever the host puts bytes into a page the guest may execute, or a
    /// frame behind an executable mapping changes (a copy-on-write copy): an
    /// arm64 hardware vcpu must invalidate its instruction cache before running
    /// this space again, since stores on the host's side don't reach the
    /// instruction fetch path by themselves. Code the guest writes itself is its
    /// own to synchronize, as on Linux. Consumed by
    /// [`GuestMemory::take_icache_dirty`]; x86-64 and the interpreters need none.
    icache_dirty: bool,
    /// Physical address of this address space's private kernel-stack frame (the
    /// page mapped at [`ctrl::KSTACK_PAGE_VA`]). The CPU pushes the `#PF`
    /// exception frame here at CPL0; a KVM vcpu reads it back from this frame
    /// (not a shared control page) so concurrent faults on sibling vcpus never
    /// clobber one another. See [`GuestMemory::kstack_pa`].
    kstack_pa: u64,
    /// True once a second task shares this address space (`CLONE_VM`/`CLONE_THREAD`
    /// — a pthread). Concurrent threads of one address space can't run KVM in
    /// parallel: they share one page-table tree and one kstack frame, so
    /// simultaneous `#PF`s corrupt each other's exception frame and one thread's
    /// page-table edit leaves siblings' TLBs stale. The SMP scheduler therefore
    /// runs a shared space *serialized* (holding the per-space memory lock across
    /// the run, like the interpreter) instead of lockless. Distinct processes have
    /// distinct spaces/locks, so cross-process parallelism is unaffected. Sticky:
    /// stays set after a thread exits (harmless — just keeps that space serial).
    shared: bool,
    /// The memory cgroup (allocator slot) frames this address space allocates
    /// are charged to: its tasks' cgroup's, set by the kernel as they run.
    memcg: u16,
    /// Fault-around state (see [`GuestMemory::demand_fault`]): the page the
    /// last demand fault backed last, the direction the faults are walking
    /// (+1 up, -1 down — a stack), and how many pages the next fault in that
    /// direction backs.
    fault_last: u64,
    fault_dir: i64,
    fault_window: u64,
    /// Which incarnation of an address space this is: a new value for every
    /// page-table tree (`new`, `fork`, `execve`), never reused, so a hardware
    /// backend can tag its translations by address space (arm64 ASIDs) without
    /// confusing a tree with an earlier one at the same root frame.
    space_id: u64,
    /// File mappings (`mmap` of a file, a memfd, `/dev/shm`; a named SysV
    /// attachment), by first page: where a page's frame comes from when
    /// it's first touched, and what `/proc/<pid>/maps` calls it. See
    /// [`super::filemap`].
    files: std::collections::BTreeMap<usize, FileVma>,
    /// The [`SHRINK_EPOCH`] this space last dropped lost file pages at.
    shrink_seen: u64,
}

/// What [`GuestMemory::file_fault`] made of a fault.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFault {
    /// Not a fault on a file mapping's missing page.
    NotFile,
    /// The page is mapped now: run the access again.
    Resolved,
    /// The page is past the end of the file: `SIGBUS`.
    Bus,
}

/// The next [`GuestMemory::space_id`].
static NEXT_SPACE_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn new_space_id() -> u64 {
    NEXT_SPACE_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// The most pages one demand fault backs: 64 KiB, Linux's `fault_around_bytes`.
const FAULT_AROUND_MAX: u64 = 16;

impl std::fmt::Debug for GuestMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GuestMemory")
            .field("base", &self.base)
            .field("size", &self.size)
            .field("cr3", &self.space.cr3())
            .finish_non_exhaustive()
    }
}

/// [`GuestMemory::set_fork_policy`]: the page is not inherited by a fork child.
pub const FORK_DONT: u8 = 1;
/// [`GuestMemory::set_fork_policy`]: a fork child sees the page zero-filled.
pub const FORK_WIPE: u8 = 2;

/// Take the frame allocator `fa`, charging what's allocated to memory cgroup
/// `memcg` (an address space's: see [`GuestMemory::set_memcg`]).
fn lock_fa(fa: &Mutex<FrameAllocator>, memcg: u16) -> std::sync::MutexGuard<'_, FrameAllocator> {
    let mut fa = fa.lock().unwrap();
    fa.charge_to = memcg;
    fa
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
    /// `GuestMemory::exec_reset` (which rebuilds within it).
    #[must_use]
    pub fn new(base: u64, size: u64) -> Self {
        // The tied form (virtual bound == physical pool), used by tests.
        Self::new_split(base, size, size)
    }

    /// Reserve a guest address space whose *virtual* extent (`[base, base+vsize)`)
    /// is decoupled from the *physical* RAM pool (`phys_bytes`). A real MMU with
    /// demand paging needs this: a JS engine (JSC/V8) reserves tens of GiB of
    /// `MAP_NORESERVE` virtual address space up front but commits (touches) only a
    /// small working set, so the virtual bound must be large (else the mmap arena
    /// exhausts and a plain 128 MiB `mmap` fails with `ENOMEM`, which the engine
    /// treats as unreachable and traps) while the physical pool stays modest and
    /// is filled on first touch. `vsize` also sizes the per-page metadata vectors
    /// (~3 bytes/page — 48 MiB for a 64 GiB virtual space), so it is generous, not
    /// unbounded; the pool bounds the actual committed working set.
    #[must_use]
    pub fn new_split(base: u64, vsize: u64, phys_bytes: u64) -> Self {
        assert_eq!(base % PAGE_SIZE, 0, "base must be page-aligned");
        let size = vsize.max(PAGE_SIZE).next_multiple_of(HOST_PAGE as u64);
        let npages = (size / PAGE_SIZE) as usize;
        let phys_bytes = phys_bytes.min(crate::limits::guest_memory_ceiling());

        // Pool budget: the physical RAM (`phys_bytes`) holds the committed working
        // set — one frame per *touched* page — plus interior page tables (a PT per
        // 512 touched pages; the /16 allowance is generous headroom for sparse
        // mappings and a fork's second tree), the control block, the null frame,
        // and slack. This is bounded by physical RAM, NOT the (much larger)
        // virtual extent — untouched `MAP_NORESERVE` reservations cost no frame.
        let phys_pages = phys_bytes.next_multiple_of(HOST_PAGE as u64) / PAGE_SIZE;
        let pool_frames = phys_pages + phys_pages / 16 + ctrl::CTRL_FRAMES + 128;
        let phys = Arc::new(PhysMem::new((pool_frames * PAGE_SIZE) as usize));

        let mut fa = FrameAllocator::new(phys.nframes());
        // Well inside the slack the pool was given above, so a guest still gets
        // all of the RAM it was promised.
        fa.set_reserve(64);
        ctrl::reserve_and_build(&mut fa, &phys);
        let mut space = AddrSpace::new(&mut fa, &phys).expect("pool too small for a PML4");
        ctrl::map_into(&mut space, &mut fa, &phys);
        let kstack_pa = install_kstack(&mut space, &mut fa, &phys);

        Self {
            base,
            size,
            phys,
            due: Arc::clone(&fa.due),
            fa: Arc::new(Mutex::new(fa)),
            space,
            prot: PageVec::new(npages, Prot::NONE),
            mapped: PageVec::new(npages, false),
            mapped_count: 0,
            file_backed: PageVec::new(npages, false),
            shared_anon: PageVec::new(npages, false),
            fork_policy: std::collections::BTreeMap::new(),
            locks: super::pagelocks::PageLocks::default(),
            tlb_dirty: false,
            tlb_gen: 0,
            heap: Heap::default(),
            exec_gen: 0,
            icache_dirty: true,
            kstack_pa,
            shared: false,
            memcg: 0,
            fault_last: 0,
            fault_dir: 1,
            fault_window: 1,
            space_id: new_space_id(),
            files: std::collections::BTreeMap::new(),
            shrink_seen: SHRINK_EPOCH.load(std::sync::atomic::Ordering::Acquire),
        }
    }

    /// Build this space's page tables for a guest of `arch` from now on (its
    /// descriptor format: what `arch`'s hardware MMU walks), converting those
    /// already built. A new [`GuestMemory`] starts in the x86-64 format; an
    /// aarch64 guest that is to run on hardware switches before it runs. All the
    /// address spaces of a pool share one format, so this must come before the
    /// pool has a second (before a `fork`).
    ///
    /// # Errors
    /// The format would change once the pool is shared with another space.
    pub fn set_guest_arch(&mut self, arch: crate::abi::Arch) -> Result<(), MemError> {
        let want = PtFormat::for_arch(arch);
        if self.phys.format() == want {
            return Ok(());
        }
        if Arc::strong_count(&self.phys) != 1 {
            return Err(MemError::Host(
                "a guest's page-table format is chosen before its memory is shared".into(),
            ));
        }
        let from = Enc::of(&self.phys);
        self.phys.set_format(want);
        self.space.reencode(from, Enc::of(&self.phys), &self.phys);
        ctrl::build_arm_kernel_tables(&self.phys);
        mark_tlb_dirty!(self);
        Ok(())
    }

    /// The format of this space's page tables.
    #[must_use]
    pub fn page_table_format(&self) -> PtFormat {
        self.phys.format()
    }

    /// Charge frames this address space allocates from now on to memory cgroup
    /// `slot` (what it already has stays where it was charged, as on Linux).
    pub fn set_memcg(&mut self, slot: u16) {
        self.memcg = slot;
    }

    /// The memory cgroup whose `memory.max` refused a frame since the last
    /// call, if one did.
    #[must_use]
    pub fn take_memcg_oom(&self) -> Option<u16> {
        self.fa.lock().unwrap().take_memcg_oom()
    }

    /// Whether the kernel wants this space's task back as soon as it can be
    /// had: a batch was charged past its memory cgroup's `memory.high`, a
    /// penalty to pay (the frame allocator's `due`). Lock-free, for an
    /// interpreter to poll.
    #[must_use]
    pub fn kernel_due(&self) -> bool {
        self.due.load(std::sync::atomic::Ordering::Relaxed) & (1 << (self.memcg % 64)) != 0
    }

    /// The frame allocator's `take_over_high`: frames charged to `slot` past
    /// `memory.high` and the overage, once a batch is due.
    #[must_use]
    pub fn take_memcg_over_high(&self, slot: u16) -> Option<(u64, u64)> {
        self.fa.lock().unwrap().take_over_high(slot)
    }

    /// The frame allocator every address space shares.
    #[must_use]
    pub fn allocator(&self) -> Arc<Mutex<FrameAllocator>> {
        Arc::clone(&self.fa)
    }

    /// Run `f` on the frame allocator every address space shares, for the
    /// kernel's memory cgroup bookkeeping.
    pub fn with_allocator<T>(&self, f: impl FnOnce(&mut FrameAllocator) -> T) -> T {
        f(&mut self.fa.lock().unwrap())
    }

    // ---- KVM/backend hooks (crate-internal) ------------------------------

    /// The `CR3` value (PML4 physical address) for this address space.
    #[must_use]
    #[cfg_attr(
        not(any(
            all(target_os = "linux", target_arch = "x86_64"),
            all(target_os = "macos", target_arch = "aarch64")
        )),
        allow(dead_code)
    )] // hardware backends only
    pub(crate) fn cr3(&self) -> u64 {
        self.space.cr3()
    }

    /// Host pointer to physical address 0 of the shared pool — the single KVM
    /// memslot's `userspace_addr`.
    #[must_use]
    #[cfg_attr(
        not(any(
            all(target_os = "linux", target_arch = "x86_64"),
            all(target_os = "macos", target_arch = "aarch64")
        )),
        allow(dead_code)
    )] // hardware backends only
    pub(crate) fn phys_ptr(&self) -> *mut u8 {
        self.phys.as_ptr()
    }

    /// Size of the shared pool in bytes — the memslot's `memory_size`.
    #[must_use]
    #[cfg_attr(
        not(any(
            all(target_os = "linux", target_arch = "x86_64"),
            all(target_os = "macos", target_arch = "aarch64")
        )),
        allow(dead_code)
    )] // hardware backends only
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
    #[cfg_attr(
        not(all(target_os = "linux", target_arch = "x86_64")),
        allow(dead_code)
    )] // KVM-only
    pub(crate) fn kstack_pa(&self) -> u64 {
        self.kstack_pa
    }

    /// Whether a second task shares this address space (a `CLONE_VM` thread) — the
    /// SMP scheduler must run such a space serialized, not lockless. See [`shared`].
    #[must_use]
    pub(crate) fn is_shared(&self) -> bool {
        self.shared
    }

    /// Mark this address space as shared by more than one task (called when a
    /// `CLONE_VM`/`CLONE_THREAD` task is created against it). Sticky.
    pub(crate) fn mark_shared(&mut self) {
        self.shared = true;
    }

    /// Take and clear the "page tables changed" flag: whether a present leaf was
    /// unmapped, re-protected, or copy-on-write-privatized since the last check.
    /// The scheduler flushes the running KVM vcpu's TLB when this is set, so a
    /// stale entry never serves an unmapped or replaced frame.
    pub(crate) fn take_tlb_dirty(&mut self) -> bool {
        std::mem::take(&mut self.tlb_dirty)
    }

    /// How many TLB-staling page-table edits this address space has had. A
    /// vcpu that last flushed at another value holds entries that may be
    /// stale — one of a process's other threads unmapped, re-protected or
    /// replaced a page since — and must flush before it runs.
    #[must_use]
    pub fn tlb_generation(&self) -> u64 {
        self.tlb_gen
    }

    /// Take and clear the "code changed behind the guest's back" flag (see the
    /// field): whether the host wrote into a page the guest may execute since
    /// the last check.
    #[cfg_attr(
        not(all(target_os = "macos", target_arch = "aarch64")),
        allow(dead_code)
    )]
    pub(crate) fn take_icache_dirty(&mut self) -> bool {
        std::mem::take(&mut self.icache_dirty)
    }

    /// Where physical address 0 of the pool appears to a hardware page-table
    /// walker (see [`PhysMem::base`]).
    #[must_use]
    #[cfg_attr(
        not(all(target_os = "macos", target_arch = "aarch64")),
        allow(dead_code)
    )]
    pub(crate) fn phys_base(&self) -> u64 {
        self.phys.base()
    }

    /// This page-table tree's incarnation (see the field).
    #[must_use]
    #[cfg_attr(
        not(all(target_os = "macos", target_arch = "aarch64")),
        allow(dead_code)
    )]
    pub(crate) fn space_id(&self) -> u64 {
        self.space_id
    }

    /// `TTBR1_EL1` for this (arm64) pool: its upper-half kernel tables.
    #[must_use]
    #[cfg_attr(
        not(all(target_os = "macos", target_arch = "aarch64")),
        allow(dead_code)
    )]
    pub(crate) fn arm_ttbr1(&self) -> u64 {
        ctrl::arm_ttbr1(&self.phys)
    }

    /// Whether any page of `[first, last]` may be executed.
    fn any_exec(&self, first: usize, last: usize) -> bool {
        (first..=last).any(|p| self.prot[p].contains(Prot::EXEC))
    }

    /// Send guests' clock reads through the kernel from now on, by leaving the vDSO
    /// with no calibration to use.
    pub fn disable_vdso_clocks(&self) {
        super::ctrl::write_vvar(&self.phys, 0, 0, 0, 0, 0);
    }

    /// Total size of this space's mappings, what `RLIMIT_AS` measures.
    #[must_use]
    pub fn mapped_bytes(&self) -> u64 {
        self.mapped_count as u64 * PAGE_SIZE
    }

    /// Bytes locked by `mlock`/`mlockall`, what `RLIMIT_MEMLOCK` measures.
    #[must_use]
    pub fn locked_bytes(&self) -> u64 {
        self.locks.pages() as u64 * PAGE_SIZE
    }

    /// The page range `[first, end)` covering `[addr, addr + len)`, if in bounds.
    fn lock_span(&self, addr: u64, len: u64) -> Option<(usize, usize)> {
        let start = addr - addr % PAGE_SIZE;
        let end = round_up(addr.checked_add(len)?, PAGE_SIZE);
        let (first, last) = self.page_range(start, (end - start) as usize).ok()?;
        Some((first, last + 1))
    }

    /// How many bytes of `[addr, addr + len)` aren't locked yet.
    #[must_use]
    pub fn unlocked_bytes_in(&self, addr: u64, len: u64) -> u64 {
        if len == 0 {
            return 0;
        }
        self.lock_span(addr, len).map_or(0, |(first, end)| {
            (end - first - self.locks.locked_within(first, end)) as u64 * PAGE_SIZE
        })
    }

    /// Lock (`mlock`) or unlock (`munlock`) the pages of `[addr, addr + len)`.
    pub fn set_locked(&mut self, addr: u64, len: u64, locked: bool) {
        if len == 0 {
            return;
        }
        if let Some((first, end)) = self.lock_span(addr, len) {
            if locked {
                self.locks.insert(first, end);
            } else {
                self.locks.remove(first, end);
            }
        }
    }

    /// `mlockall`: lock every page mapped now if `current`, and every one mapped
    /// from now on if `future`.
    pub fn lock_all(&mut self, current: bool, future: bool) {
        if current {
            let mut run: Option<(usize, usize)> = None;
            let pages = self
                .mapped
                .non_default()
                .filter(|&(_, m)| m)
                .map(|(p, _)| p);
            for p in pages {
                run = match run {
                    Some((s, e)) if e == p => Some((s, p + 1)),
                    Some((s, e)) => {
                        self.locks.insert(s, e);
                        Some((p, p + 1))
                    }
                    None => Some((p, p + 1)),
                };
            }
            if let Some((s, e)) = run {
                self.locks.insert(s, e);
            }
        }
        self.locks.future = future;
    }

    /// `munlockall`: unlock everything and stop locking new mappings.
    pub fn unlock_all(&mut self) {
        self.locks.clear();
    }

    /// Whether `mlockall(MCL_FUTURE)` is in force.
    #[must_use]
    pub fn locks_future(&self) -> bool {
        self.locks.future
    }

    /// Number of live (allocated) frames in the shared pool — for leak checks.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn frames_in_use(&self) -> usize {
        lock_fa(&self.fa, self.memcg).alloc_count()
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
        // A shared-anonymous page is deliberately aliased across address spaces:
        // it stays writable even at refcount > 1 (writes go to the shared frame,
        // not a private copy).
        if !self.shared_anon[p] && self.prot[p].contains(Prot::WRITE) && fa.refcount(frame) > 1 {
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
        // (Dropped once the allocator is let go: a file's last reference
        // frees its frames.)
        let _gone = self.cut_files(first, last + 1);
        let mut fa = lock_fa(&self.fa, self.memcg);
        for p in first..=last {
            // Fresh mapping: drop any existing backing so the page reads as zero.
            if self.mapped[p] {
                let va = self.base + (p as u64) * PAGE_SIZE;
                if let Some(frame) = self.space.unmap(va, &mut fa, &self.phys) {
                    fa.free(frame);
                }
            }
            self.mapped_count += usize::from(!self.mapped[p]);
            self.mapped[p] = true;
            self.prot[p] = prot;
            self.file_backed[p] = false;
            self.shared_anon[p] = false; // a fresh mapping is private until tagged
            self.fork_policy.remove(&p);
        }
        // A mapping replaced by a new one is no longer locked.
        self.locks.remove(first, last + 1);
        drop(fa);
        Ok(())
    }

    /// Give the mapped pages of `[addr, addr + len)` a fork policy
    /// ([`FORK_DONT`], [`FORK_WIPE`], or 0 to inherit normally) — the
    /// `MADV_DONTFORK`/`MADV_WIPEONFORK` (and `DOFORK`/`KEEPONFORK`) advice.
    /// `MemError::Unmapped` if any page in the range isn't mapped.
    pub fn set_fork_policy(&mut self, addr: u64, len: u64, policy: u8) -> Result<(), MemError> {
        if len == 0 {
            return Ok(());
        }
        let start = addr - addr % PAGE_SIZE;
        let end = round_up(addr + len, PAGE_SIZE);
        let (first, last) = self.page_range(start, (end - start) as usize)?;
        if let Some(p) = (first..=last).find(|&p| !self.mapped[p]) {
            return Err(MemError::Unmapped(self.page_base(p)));
        }
        for p in first..=last {
            if policy == 0 {
                self.fork_policy.remove(&p);
            } else {
                self.fork_policy.insert(p, policy);
            }
        }
        Ok(())
    }

    /// Map a `MAP_SHARED | MAP_ANONYMOUS` region: like [`Self::map`], but tag the
    /// pages shared and **eagerly** back each with a frame. Eager backing is what
    /// makes cross-`fork` sharing work — a shared page must have a real frame at
    /// fork time so the child can alias it; a lazily-faulted page would mint a
    /// private frame per address space and defeat the sharing. Returns `ENOMEM`
    /// (as `Host`) if the pool can't back the whole region.
    pub fn map_shared_anon(&mut self, addr: u64, len: u64, prot: Prot) -> Result<(), MemError> {
        self.map(addr, len, prot)?;
        let start = addr - addr % PAGE_SIZE;
        let end = round_up(addr + len, PAGE_SIZE);
        let (first, last) = self.page_range(start, (end - start) as usize)?;
        let mut fa = lock_fa(&self.fa, self.memcg);
        for p in first..=last {
            self.shared_anon[p] = true;
            let va = self.base + (p as u64) * PAGE_SIZE;
            if !Self::ensure_backed(&mut self.space, &self.phys, self.prot[p], &mut fa, va) {
                return Err(MemError::Host("frame pool exhausted".into()));
            }
        }
        Ok(())
    }

    /// Allocate `n` zeroed frames from the shared pool for an owner that lives
    /// outside any one address space — a SysV shared-memory segment — each
    /// carrying one reference held by that owner (drop it with
    /// [`Self::release_frames`]). All-or-nothing: on exhaustion every frame
    /// taken so far is returned and `None` comes back.
    #[must_use]
    pub fn alloc_frames(&self, n: usize) -> Option<Vec<u64>> {
        let mut fa = lock_fa(&self.fa, self.memcg);
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            if let Some(f) = fa.alloc(&self.phys) {
                out.push(f);
            } else {
                for f in out {
                    fa.free(f);
                }
                return None;
            }
        }
        Some(out)
    }

    /// Drop the owner's reference to each of `frames` (see
    /// [`Self::alloc_frames`]); a frame still mapped somewhere lives on until
    /// its last mapping goes.
    pub fn release_frames(&self, frames: &[u64]) {
        let mut fa = lock_fa(&self.fa, self.memcg);
        for &f in frames {
            fa.free(f);
        }
    }

    /// How many owners hold `frame` right now — every mapping of it plus any
    /// outside owner (a SysV segment, the shared-file page cache). An outside
    /// owner seeing 1 knows no address space maps the frame any more.
    #[must_use]
    pub fn frame_refcount(&self, frame: u64) -> u32 {
        lock_fa(&self.fa, self.memcg).refcount(frame)
    }

    /// The frames behind `[addr, addr + len)` if *every* page there is a
    /// shared page with a frame installed (a `MAP_SHARED` region, a SysV
    /// segment): what `mremap` moves instead of copying, so the moved mapping
    /// still shares memory with the other mappers. `None` otherwise.
    #[must_use]
    pub fn shared_frames(&self, addr: u64, len: u64) -> Option<Vec<u64>> {
        let mut out = Vec::new();
        let mut p = addr;
        while p < addr.checked_add(len)? {
            out.push(self.shared_phys(p)? & !(PAGE_SIZE - 1));
            p += PAGE_SIZE;
        }
        Some(out)
    }

    /// The physical address behind `addr` if its page is a *shared* page — one
    /// aliased across address spaces (`MAP_SHARED`, a SysV segment) — else
    /// `None`. This is what a process-shared futex is keyed by: two processes
    /// waiting and waking on the same shared word see the same physical
    /// address even when it is mapped at different virtual addresses.
    #[must_use]
    pub fn shared_phys(&self, addr: u64) -> Option<u64> {
        let p = self.page_index(addr)?;
        if !self.mapped[p] || !self.shared_anon[p] {
            return None;
        }
        if let Some(t) = self.space.translate(addr, &self.phys) {
            return Some(t.paddr);
        }
        // A shared file mapping's page not touched here yet: the file's page.
        let (start, vma) = self.file_at(p)?;
        let obj = vma.map.obj.as_ref().filter(|_| vma.map.shared)?;
        let frame = obj
            .frame(vma.map.pgoff + (p - start) as u64, self.memcg)
            .ok()?;
        Some(frame + addr % PAGE_SIZE)
    }

    /// Map `frames` (one per page) at `addr` with `prot`, *sharing* them: each
    /// page takes its own reference on its frame, is tagged shared like a
    /// `MAP_SHARED | MAP_ANONYMOUS` page (writable at any refcount, aliased —
    /// not copied — across `fork`), and gives the reference back on
    /// `unmap`/`exec`/exit. This is how a SysV shared-memory segment
    /// (`shmat`) appears in several address spaces at once: every attachment
    /// points at the same physical frames. Any existing mapping in the range is
    /// replaced.
    pub fn map_frames(&mut self, addr: u64, frames: &[u64], prot: Prot) -> Result<(), MemError> {
        if frames.is_empty() {
            return Ok(());
        }
        let len = frames.len() as u64 * PAGE_SIZE;
        let (first, last) = self.page_range(addr, len as usize)?;
        let _gone = self.cut_files(first, last + 1);
        let mut fa = lock_fa(&self.fa, self.memcg);
        for (i, p) in (first..=last).enumerate() {
            let va = self.page_base(p);
            if self.mapped[p]
                && let Some(old) = self.space.unmap(va, &mut fa, &self.phys)
            {
                fa.free(old);
                mark_tlb_dirty!(self);
            }
            let frame = frames[i];
            fa.incref(frame);
            match self.space.map(va, frame, prot, false, &mut fa, &self.phys) {
                Ok(Some(old)) => fa.free(old),
                Ok(None) => {}
                Err(_) => {
                    fa.free(frame);
                    return Err(MemError::Host("frame pool exhausted".into()));
                }
            }
            self.mapped_count += usize::from(!self.mapped[p]);
            self.mapped[p] = true;
            self.prot[p] = prot;
            self.file_backed[p] = false;
            self.shared_anon[p] = true;
        }
        drop(fa);
        self.icache_dirty |= prot.contains(Prot::EXEC);
        Ok(())
    }

    // ---- file mappings ----------------------------------------------------

    /// The file mapping covering page `p`, with the page its first page is.
    fn file_at(&self, p: usize) -> Option<(usize, &FileVma)> {
        if self.files.is_empty() {
            return None;
        }
        let (&start, vma) = self.files.range(..=p).next_back()?;
        (p < start + vma.pages).then_some((start, vma))
    }

    /// Take the file mappings' pages `[first, end)` out of them, splitting a
    /// mapping that reaches past either end. What was taken is returned, for
    /// the caller to drop once it holds no allocator lock (a file's last
    /// reference frees its frames).
    fn cut_files(&mut self, first: usize, end: usize) -> Vec<FileVma> {
        if self.files.is_empty() {
            return Vec::new();
        }
        let mut gone = Vec::new();
        let keys: Vec<usize> = self
            .files
            .range(..end)
            .rev()
            .take_while(|(s, v)| *s + v.pages > first)
            .map(|(s, _)| *s)
            .collect();
        for start in keys {
            let vma = self.files.remove(&start).expect("listed");
            let vend = start + vma.pages;
            if start < first {
                self.files.insert(start, vma.slice(0, first - start));
            }
            if vend > end {
                self.files.insert(end, vma.slice(end - start, vend - end));
            }
            gone.push(vma);
        }
        gone
    }

    /// Map `[addr, addr + len)` (page-aligned) as `map` describes, with
    /// `prot`: the file's pages come in as they're first touched (and those
    /// it has already are put in now), past the end of the file is `SIGBUS`,
    /// and a shared mapping's stores are the file's — seen by every other
    /// mapper, kept across `fork`. A private one is copy-on-write over the
    /// file's pages: until a page is written, it reads what the file holds.
    /// Whatever was mapped in the range is replaced.
    pub fn map_file(
        &mut self,
        addr: u64,
        len: u64,
        prot: Prot,
        map: FileMap,
    ) -> Result<(), MemError> {
        if len == 0 {
            return Ok(());
        }
        self.map(addr, len, prot)?;
        let (first, last) = self.page_range(addr, len as usize)?;
        for p in first..=last {
            self.shared_anon[p] = map.shared;
            // (Keeps fault-around off the pages: they come from the file.)
            self.file_backed[p] = !map.shared;
        }
        let pages = last + 1 - first;
        let obj = map.obj.clone();
        let pgoff = map.pgoff;
        self.files.insert(first, FileVma::new(pages, map));
        let Some(obj) = obj else {
            return Ok(());
        };
        if prot == Prot::NONE {
            return Ok(());
        }
        // The pages the file has already: put them in now rather than take a
        // fault for each.
        for (idx, _) in obj.pages_in(pgoff, pgoff + pages as u64) {
            let p = first + (idx - pgoff) as usize;
            let Ok(frame) = obj.get(idx, self.memcg) else {
                continue;
            };
            self.install_file_page(p, frame);
        }
        self.icache_dirty |= prot.contains(Prot::EXEC);
        Ok(())
    }

    /// Put `frame` (a reference the caller took for this mapping) at page `p`.
    fn install_file_page(&mut self, p: usize, frame: u64) -> bool {
        let va = self.page_base(p);
        let mut fa = lock_fa(&self.fa, self.memcg);
        let lp = self.leaf_prot(p, frame, &fa);
        let Ok(old) = self.space.map(va, frame, lp, false, &mut fa, &self.phys) else {
            fa.free(frame);
            return false;
        };
        if let Some(old) = old {
            fa.free(old);
            mark_tlb_dirty!(self);
        }
        true
    }

    /// Name `[addr, addr + len)` in `/proc/<pid>/maps` as `map` says, its
    /// frames being already in place (a SysV attachment, shared anonymous
    /// memory). `map.obj` should be `None`.
    pub fn name_range(&mut self, addr: u64, len: u64, map: FileMap) -> Result<(), MemError> {
        if len == 0 {
            return Ok(());
        }
        let (first, last) = self.page_range(addr, len as usize)?;
        let _gone = self.cut_files(first, last + 1);
        self.files
            .insert(first, FileVma::new(last + 1 - first, map));
        Ok(())
    }

    /// Resolve a fault at `addr` on a file mapping's page that isn't in the
    /// page tables: put the file's page there ([`FileFault::Resolved`]), or
    /// find it past the end of the file ([`FileFault::Bus`]).
    pub fn file_fault(&mut self, addr: u64) -> FileFault {
        let page = addr - addr % PAGE_SIZE;
        let Some(p) = self.page_index(page) else {
            return FileFault::NotFile;
        };
        if !self.mapped[p] || self.prot[p] == Prot::NONE {
            return FileFault::NotFile;
        }
        if self.space.translate(page, &self.phys).is_some() {
            return FileFault::NotFile;
        }
        match self.back_file_page(p) {
            Ok(true) => {
                self.file_fault_around(p);
                FileFault::Resolved
            }
            Ok(false) => FileFault::NotFile,
            Err(_) => FileFault::Bus,
        }
    }

    /// After a file page's fault: put in the pages after it that the file
    /// already has (up to [`FAULT_AROUND_MAX`], within the mapping, the same
    /// page table and protection), as Linux's fault-around maps what's in the
    /// page cache — a code page the JIT wrote through its writable view,
    /// say, the first time its executable view runs it.
    fn file_fault_around(&mut self, p: usize) {
        let Some((start, vma)) = self.file_at(p) else {
            return;
        };
        let Some(obj) = vma.map.obj.clone() else {
            return;
        };
        let pgoff = vma.map.pgoff + (p - start) as u64;
        let room = (start + vma.pages - p - 1).min(FAULT_AROUND_MAX as usize - 1);
        let table_end = (self.page_base(p) | 0x1f_ffff) + 1;
        let prot = self.prot[p];
        for (idx, _) in obj.pages_in(pgoff + 1, pgoff + 1 + room as u64) {
            let q = p + (idx - pgoff) as usize;
            let va = self.page_base(q);
            if va >= table_end
                || !self.mapped[q]
                || self.prot[q] != prot
                || self.space.translate(va, &self.phys).is_some()
            {
                continue;
            }
            let Ok(frame) = obj.get(idx, self.memcg) else {
                break;
            };
            self.install_file_page(q, frame);
        }
    }

    /// Give page `p` its file page if it's a file mapping's and has none:
    /// `Ok(true)` if it did, `Ok(false)` if there was nothing to do (or no
    /// frame to be had), `Bus` past the end of the file.
    fn back_file_page(&mut self, p: usize) -> Result<bool, MemError> {
        let Some((start, vma)) = self.file_at(p) else {
            return Ok(false);
        };
        let Some(obj) = vma.map.obj.clone() else {
            return Ok(false);
        };
        let idx = vma.map.pgoff + (p - start) as u64;
        let exec = self.prot[p].contains(Prot::EXEC);
        match obj.get(idx, self.memcg) {
            Ok(frame) => {
                let done = self.install_file_page(p, frame);
                self.icache_dirty |= exec;
                Ok(done)
            }
            Err(PageError::Beyond) => Err(MemError::Bus(self.page_base(p))),
            Err(PageError::NoMemory) => Ok(false),
        }
    }

    /// [`Self::back_file_page`] for each of pages `[first, last]` not in the
    /// page tables: what a kernel copy into a file mapping needs first.
    fn back_file_pages(&mut self, first: usize, last: usize) -> Result<(), MemError> {
        if self.files.is_empty() {
            return Ok(());
        }
        for p in first..=last {
            if self.mapped[p]
                && self.file_at(p).is_some()
                && self
                    .space
                    .translate(self.page_base(p), &self.phys)
                    .is_none()
            {
                self.back_file_page(p)?;
            }
        }
        Ok(())
    }

    /// Drop the pages mapped files have lost since this space last looked
    /// (a file shrank under its mappings, or had a hole punched): touching
    /// one again finds the file as it is now — `SIGBUS` past its end, zeros
    /// in a hole. Cheap when nothing shrank. Called before the space runs.
    pub fn revalidate_files(&mut self) {
        let epoch = SHRINK_EPOCH.load(std::sync::atomic::Ordering::Acquire);
        if epoch == self.shrink_seen {
            return;
        }
        let seen = std::mem::replace(&mut self.shrink_seen, epoch);
        let mut lost = Vec::new();
        for (&start, vma) in &self.files {
            let Some(obj) = &vma.map.obj else { continue };
            if obj.shrunk_at() <= seen {
                continue;
            }
            for k in 0..vma.pages {
                let p = start + k;
                let va = self.page_base(p);
                let Some(t) = self.space.translate(va, &self.phys) else {
                    continue;
                };
                let frame = t.paddr & !(PAGE_SIZE - 1);
                let stale = match obj.peek(vma.map.pgoff + k as u64) {
                    Err(_) => true,
                    // A shared page must be the file's own; a private one may
                    // be its own copy.
                    Ok(f) => vma.map.shared && f != Some(frame),
                };
                if stale {
                    lost.push(va);
                }
            }
        }
        if lost.is_empty() {
            return;
        }
        let mut fa = lock_fa(&self.fa, self.memcg);
        for va in lost {
            if let Some(frame) = self.space.unmap(va, &mut fa, &self.phys) {
                fa.free(frame);
            }
        }
        mark_tlb_dirty!(self);
    }

    /// The file mapping at `addr`: what it maps, and the file page `addr` is.
    #[must_use]
    pub fn file_map_at(&self, addr: u64) -> Option<(FileMap, u64)> {
        let p = self.page_index(addr)?;
        let (start, vma) = self.file_at(p)?;
        Some((vma.map.clone(), vma.map.pgoff + (p - start) as u64))
    }

    /// Whether a shared file mapping in `[addr, addr + len)` can never be made
    /// writable (`VM_MAYWRITE` clear: the file wasn't opened for writing, or
    /// is sealed against it) — `mprotect`'s `EACCES`.
    #[must_use]
    pub fn write_forbidden(&self, addr: u64, len: u64) -> bool {
        let (Some(first), Some(last)) = (
            self.page_index(addr),
            self.page_index(addr + len.max(1) - 1),
        ) else {
            return false;
        };
        self.files
            .range(..=last)
            .rev()
            .take_while(|(s, v)| *s + v.pages > first)
            .any(|(_, v)| v.map.shared && !v.map.maywrite)
    }

    /// Every file mapping, as `(start, end, what it maps)`.
    #[must_use]
    pub fn file_maps(&self) -> Vec<(u64, u64, FileMap)> {
        self.files
            .iter()
            .map(|(&s, v)| {
                let start = self.page_base(s);
                (start, start + v.pages as u64 * PAGE_SIZE, v.map.clone())
            })
            .collect()
    }

    /// Move `[old, old + len)` to `[new, new + len)` (page-aligned, not
    /// overlapping) — `mremap`'s move: the pages keep their frames (a shared
    /// page keeps sharing, a private one moves without a copy), protection
    /// and file mappings. The old range is then unmapped, or with
    /// `dontunmap` stays mapped with nothing in it: a private anonymous page
    /// reads zero, a file mapping's page comes from the file again.
    pub fn move_range(
        &mut self,
        old: u64,
        new: u64,
        len: u64,
        dontunmap: bool,
    ) -> Result<(), MemError> {
        struct Page {
            mapped: bool,
            prot: Prot,
            shared: bool,
            file_backed: bool,
            frame: Option<u64>,
        }
        if len == 0 {
            return Ok(());
        }
        let (of, ol) = self.page_range(old, len as usize)?;
        let (nf, _) = self.page_range(new, len as usize)?;
        // What the old range holds, page by page, and its file mappings.
        let pages: Vec<Page> = (of..=ol)
            .map(|p| Page {
                mapped: self.mapped[p],
                prot: self.prot[p],
                shared: self.shared_anon[p],
                file_backed: self.file_backed[p],
                frame: self
                    .space
                    .translate(self.page_base(p), &self.phys)
                    .map(|t| t.paddr & !(PAGE_SIZE - 1)),
            })
            .collect();
        let vmas: Vec<(usize, FileVma)> = self
            .files
            .range(..=ol)
            .rev()
            .take_while(|(s, v)| *s + v.pages > of)
            .map(|(&s, v)| {
                let lo = s.max(of);
                let hi = (s + v.pages).min(ol + 1);
                (lo - of, v.slice(lo - s, hi - lo))
            })
            .collect();
        // The new range: mapped, tagged and named like the old.
        self.map(new, len, Prot::NONE)?;
        for (k, pg) in pages.iter().enumerate() {
            let p = nf + k;
            if !pg.mapped {
                self.mapped_count -= usize::from(self.mapped[p]);
                self.mapped[p] = false;
                continue;
            }
            self.prot[p] = pg.prot;
            self.shared_anon[p] = pg.shared;
            self.file_backed[p] = pg.file_backed;
        }
        for (off, vma) in &vmas {
            self.files.insert(nf + off, vma.clone());
        }
        {
            let mut fa = lock_fa(&self.fa, self.memcg);
            for (k, pg) in pages.iter().enumerate() {
                let Some(frame) = pg.frame.filter(|_| pg.mapped) else {
                    continue;
                };
                let p = nf + k;
                fa.incref(frame);
                let lp = self.leaf_prot(p, frame, &fa);
                if self
                    .space
                    .map(self.page_base(p), frame, lp, false, &mut fa, &self.phys)
                    .is_err()
                {
                    fa.free(frame);
                    return Err(MemError::Host("frame pool exhausted".into()));
                }
            }
        }
        self.icache_dirty = true;
        if !dontunmap {
            return self.unmap(old, len);
        }
        // The old range stays, empty: its frames go, and the private copies'
        // leaves with them; it keeps its protection and file mappings.
        let mut fa = lock_fa(&self.fa, self.memcg);
        for p in of..=ol {
            if let Some(frame) = self.space.unmap(self.page_base(p), &mut fa, &self.phys) {
                fa.free(frame);
            }
            // Shared anonymous memory has no file to come back from: what
            // stays is fresh private memory (Linux's anonymous DONTUNMAP).
            if self.file_at(p).is_none() {
                self.shared_anon[p] = false;
                self.file_backed[p] = false;
            }
        }
        mark_tlb_dirty!(self);
        drop(fa);
        self.locks.remove(of, ol + 1);
        Ok(())
    }

    /// The `/proc/<pid>/maps` regions: runs of mapped pages with one
    /// protection, sharing and backing, in address order (the program break's
    /// region one of its own).
    #[must_use]
    pub fn regions(&self) -> Vec<Region> {
        let mut out: Vec<Region> = Vec::new();
        let none: Arc<str> = Arc::from("");
        let heap = round_up(self.heap.start, PAGE_SIZE);
        let mut last_kind = (false, false);
        for (p, mapped) in self.mapped.non_default() {
            if !mapped {
                continue;
            }
            let start = self.page_base(p);
            let kind = (self.file_backed[p], start >= heap && start < self.heap.brk);
            let same_kind = std::mem::replace(&mut last_kind, kind) == kind;
            let (shared, offset, dev, ino, name) = match self.file_at(p) {
                Some((s, v)) => (
                    v.map.shared,
                    (v.map.pgoff + (p - s) as u64) * PAGE_SIZE,
                    v.map.dev,
                    v.map.ino,
                    Arc::clone(&v.map.name),
                ),
                None => (self.shared_anon[p], 0, 0, 0, Arc::clone(&none)),
            };
            if let Some(last) = out.last_mut()
                && same_kind
                && last.end == start
                && last.prot == self.prot[p]
                && last.shared == shared
                && last.name == name
                && last.dev == dev
                && last.ino == ino
                && (name.is_empty() || last.offset + (last.end - last.start) == offset)
            {
                last.end += PAGE_SIZE;
                continue;
            }
            out.push(Region {
                start,
                end: start + PAGE_SIZE,
                prot: self.prot[p],
                shared,
                offset,
                dev,
                ino,
                name,
            });
        }
        out
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
        if space.map(va, frame, prot, false, fa, phys).is_ok() {
            true
        } else {
            fa.free(frame);
            false
        }
    }

    /// Back a demand-paged page on a fault (the hardware `#PF` / translation
    /// fault path). Returns `true` if `addr` was a mapped-but-unbacked page and
    /// is now backed, so the faulting access retries and succeeds; `false` if
    /// the fault is something else (unmapped, or an already-backed page — e.g. a
    /// copy-on-write or protection fault the caller must handle).
    ///
    /// Fault-around: a guest touching fresh memory in order (a `memset`, a
    /// growing heap or stack) would take one exit per page, and on hardware the
    /// exit, not the work, is the cost. So when faults walk the space page after
    /// page, each backs a growing run of the pages ahead of it (up to
    /// 16, `FAULT_AROUND_MAX`) — pages already mapped there, unbacked, private
    /// anonymous with the same protection, within the same page table — taken
    /// only from frames nobody needs (the allocator's `alloc_spare`: never
    /// past a memory cgroup's `high` or `max`, never by reclaiming). A fault
    /// anywhere else starts over at one page, so sparse touches of a huge
    /// reservation still cost one frame each.
    pub fn demand_fault(&mut self, addr: u64) -> bool {
        let page = addr - addr % PAGE_SIZE;
        let Some(p) = self.page_index(page) else {
            return false;
        };
        if !self.mapped[p] || self.space.translate(page, &self.phys).is_some() {
            return false;
        }
        // A file mapping's page comes from the file (`file_fault`).
        if self.file_at(p).is_some_and(|(_, v)| v.map.obj.is_some()) {
            return false;
        }
        // A PROT_NONE page grants no access — never install a backing frame for
        // it. Returning `false` lets the read/write fault propagate as SIGSEGV
        // (guard pages, redzones, poisoned regions), matching the interpreter,
        // which enforces this in software via `check()`. Backing it would leave a
        // present, readable KVM leaf and let out-of-bounds reads sail through.
        if self.prot[p] == Prot::NONE {
            return false;
        }
        let mut fa = lock_fa(&self.fa, self.memcg);
        if !Self::ensure_backed(&mut self.space, &self.phys, self.prot[p], &mut fa, page) {
            return false;
        }
        // Walking on from the last run (up, or down for a stack)? Back more.
        let dir = if page.wrapping_add(PAGE_SIZE) == self.fault_last {
            -1
        } else {
            1
        };
        let walking = page == self.fault_last.wrapping_add_signed(dir * PAGE_SIZE as i64);
        let window = match (walking, dir == self.fault_dir) {
            (false, _) => 1,
            (true, true) => (self.fault_window * 2).min(FAULT_AROUND_MAX),
            (true, false) => 2,
        };
        let mut last = page;
        if !self.shared_anon[p] && !self.file_backed[p] {
            let prot = self.prot[p];
            for k in 1..window {
                let Some(va) = page.checked_add_signed(dir * (k * PAGE_SIZE) as i64) else {
                    break;
                };
                let same_table = va >> 21 == page >> 21;
                let Some(q) = self.page_index(va).filter(|_| same_table) else {
                    break;
                };
                if !self.mapped[q]
                    || self.prot[q] != prot
                    || self.shared_anon[q]
                    || self.file_backed[q]
                    || self.space.translate(va, &self.phys).is_some()
                {
                    break;
                }
                let Some(frame) = fa.alloc_spare(&self.phys) else {
                    break;
                };
                if self
                    .space
                    .map(va, frame, prot, false, &mut fa, &self.phys)
                    .is_err()
                {
                    fa.free(frame);
                    break;
                }
                last = va;
            }
        }
        self.fault_last = last;
        self.fault_dir = dir;
        self.fault_window = window;
        true
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

    /// Whether the page containing `addr` is *resident* — has a backing frame
    /// installed in the page tables right now. A mapped-but-never-touched
    /// (demand-paged) page is not resident until its first access mints a frame;
    /// an unmapped or out-of-bounds page is never resident. Read-only; installs
    /// nothing. This is exactly what `mincore` reports per page.
    #[must_use]
    pub fn is_resident(&self, addr: u64) -> bool {
        let page = addr - addr % PAGE_SIZE;
        if self.page_index(page).is_none() {
            return false;
        }
        self.space.translate(page, &self.phys).is_some()
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
        let fa = lock_fa(&self.fa, self.memcg);
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
                    mark_tlb_dirty!(self); // a present leaf's permissions changed
                }
            }
        }
        Ok(())
    }

    /// Drop the backing of the private anonymous pages covering
    /// `[addr, addr + len)`, keeping them mapped with their protection: the next touch reads
    /// zeros (Linux's `MADV_DONTNEED`), and the frames go back to the pool now.
    /// File-backed pages keep their contents (they'd be re-read from the file)
    /// and shared ones their frame. Unmapped pages are skipped.
    pub fn discard(&mut self, addr: u64, len: u64) -> Result<(), MemError> {
        if len == 0 {
            return Ok(());
        }
        let start = addr - addr % PAGE_SIZE;
        let end = round_up(addr + len, PAGE_SIZE);
        let (first, last) = self.page_range(start, (end - start) as usize)?;
        let in_file: Vec<bool> = (first..=last)
            .map(|p| self.file_at(p).is_some_and(|(_, v)| v.map.obj.is_some()))
            .collect();
        let mut fa = lock_fa(&self.fa, self.memcg);
        for p in first..=last {
            // A file mapping's page is read from the file again on the next
            // touch: a private copy is gone, a shared page is the file's.
            let file = in_file[p - first];
            if self.mapped[p]
                && (file || (!self.file_backed[p] && !self.shared_anon[p]))
                && let Some(frame) = self.space.unmap(self.page_base(p), &mut fa, &self.phys)
            {
                fa.free(frame);
                mark_tlb_dirty!(self); // a present leaf was cleared
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
        let _gone = self.cut_files(first, last + 1);
        let mut fa = lock_fa(&self.fa, self.memcg);
        for p in first..=last {
            if self.mapped[p]
                && let Some(frame) = self.space.unmap(self.page_base(p), &mut fa, &self.phys)
            {
                fa.free(frame);
                mark_tlb_dirty!(self); // a present leaf was cleared
            }
            self.mapped_count -= usize::from(self.mapped[p]);
            self.mapped[p] = false;
            self.prot[p] = Prot::NONE;
            self.file_backed[p] = false;
            self.shared_anon[p] = false;
            self.fork_policy.remove(&p);
        }
        self.locks.remove(first, last + 1);
        drop(fa);
        Ok(())
    }

    // ---- access ----------------------------------------------------------

    /// Verify `[addr, addr + len)` is mapped and every page grants `need`
    /// (intended protection).
    ///
    /// Ordinary guest memory lives in the flat `[base, base+size)` extent tracked
    /// by the per-page `prot`/`mapped` vectors. But some legitimately-mapped pages
    /// sit *above* that extent — the user-accessible control-block pages (the vDSO
    /// and its vvar, which the loader advertises to libc via `AT_SYSINFO_EHDR`).
    /// KVM reaches those through the hardware page-table walk; the interpreter
    /// reaches all memory through this method, so an out-of-extent address is
    /// validated against the page tables instead (which is also what [`Self::copy_out`]
    /// already translates through). Access is granted only to *user* (non-supervisor)
    /// leaves — exactly what a CPL3 guest could touch — so the supervisor control
    /// pages (GDT/IDT/kstack) stay unreadable.
    fn check(&self, addr: u64, len: usize, need: Prot) -> Result<(), MemError> {
        let end = addr
            .checked_add(len as u64)
            .ok_or(MemError::OutOfBounds(addr))?;
        if addr < self.base || end > self.base + self.size {
            return self.check_paged(addr, len, need);
        }
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

    /// [`Self::check`] for an address outside the flat extent: validate every page
    /// against the page tables, granting access only to present, user-accessible
    /// leaves that carry `need`. An address the tables don't map is genuinely out
    /// of bounds (there is no per-page slot for it), preserving the historical
    /// `OutOfBounds` fault for addresses beyond the guest's extent.
    fn check_paged(&self, addr: u64, len: usize, need: Prot) -> Result<(), MemError> {
        let mut cur = addr - addr % PAGE_SIZE;
        let end = addr.saturating_add(len.max(1) as u64);
        while cur < end {
            match self.space.translate(cur, &self.phys) {
                Some(t) if !t.supervisor && t.prot.contains(need) => {}
                Some(_) => {
                    return Err(MemError::Protection {
                        addr: cur,
                        needed: need,
                    });
                }
                None => return Err(MemError::OutOfBounds(addr)),
            }
            cur += PAGE_SIZE;
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
    /// A file mapping's page not yet in the page tables reads the file's page
    /// (`Bus` past the end of the file).
    fn copy_out(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemError> {
        let mut done = 0usize;
        while done < buf.len() {
            let cur = addr + done as u64;
            let page = cur - cur % PAGE_SIZE;
            let off = (cur - page) as usize;
            let n = (buf.len() - done).min(PS - off);
            match self.space.translate(cur, &self.phys) {
                Some(t) => self.phys.read(t.paddr, &mut buf[done..done + n]),
                None => match self.file_page(page)? {
                    Some(frame) => self.phys.read(frame + off as u64, &mut buf[done..done + n]),
                    None => buf[done..done + n].fill(0),
                },
            }
            done += n;
        }
        Ok(())
    }

    /// The frame a file mapping's page at `page` (not in the page tables)
    /// would map, `None` for a hole or for no file mapping there.
    fn file_page(&self, page: u64) -> Result<Option<u64>, MemError> {
        let Some(p) = self.page_index(page) else {
            return Ok(None);
        };
        let Some((start, vma)) = self.file_at(p) else {
            return Ok(None);
        };
        let Some(obj) = &vma.map.obj else {
            return Ok(None);
        };
        obj.peek(vma.map.pgoff + (p - start) as u64)
            .map_err(|_| MemError::Bus(page))
    }

    /// Read `buf.len()` bytes from guest `addr` (requires `READ`).
    /// Copying nothing never faults, whatever `addr` is, as with Linux's
    /// `copy_from_user`: a guest passes an empty buffer with any pointer.
    pub fn read(&self, addr: u64, buf: &mut [u8]) -> Result<(), MemError> {
        if buf.is_empty() {
            return Ok(());
        }
        self.check(addr, buf.len(), Prot::READ)?;
        self.copy_out(addr, buf)
    }

    /// Read `len` bytes into a fresh `Vec` (requires `READ`).
    pub fn read_vec(&self, addr: u64, len: usize) -> Result<Vec<u8>, MemError> {
        if len == 0 {
            return Ok(Vec::new());
        }
        // Validate before allocating: a guest-supplied length (a write of
        // SSIZE_MAX bytes from a bogus pointer) must fault, not make the host
        // try to allocate it.
        self.check(addr, len, Prot::READ)?;
        // Mapped isn't the same as backed: a huge untouched mapping passes the
        // check, and its length mustn't abort the host.
        let mut v = Vec::new();
        v.try_reserve_exact(len)
            .map_err(|e| MemError::Host(e.to_string()))?;
        v.resize(len, 0);
        self.read(addr, &mut v)?;
        Ok(v)
    }

    /// Ensure the page at `va` (intended protection `prot`) is backed by a private
    /// (unshared) frame its protection can write, privatizing a copy-on-write
    /// frame if needed. A no-op on an already-private page beyond restoring its
    /// leaf write bit. An associated fn over the disjoint `space`/`phys` fields so
    /// callers can hold the `fa` guard (which borrows the `fa` field) at once.
    /// Fails if a copy-on-write page can't be privatized for lack of a frame.
    /// Otherwise returns whether it changed a present leaf (so the caller flags the TLB).
    fn make_writable(
        space: &mut AddrSpace,
        phys: &PhysMem,
        prot: Prot,
        fa: &mut FrameAllocator,
        va: u64,
        shared_anon: bool,
    ) -> Result<bool, MemError> {
        let Some(t) = space.translate(va, phys) else {
            return Ok(false);
        };
        let frame = t.paddr & !(PAGE_SIZE - 1);
        if !shared_anon && fa.refcount(frame) > 1 {
            // Copy-on-write shared: privatize into a fresh frame. (A shared-anon
            // page is intentionally aliased — writes go to the shared frame, so
            // it takes the `else` branch and just gets its leaf write bit set.)
            // No frame must be an error: carrying on would write into the frame
            // another process still shares.
            let exhausted = || MemError::Host("frame pool exhausted".into());
            let new = fa.alloc(phys).ok_or_else(exhausted)?;
            phys.copy_frame(new, frame);
            let Ok(old) = space.map(va, new, prot, false, fa, phys) else {
                fa.free(new);
                return Err(exhausted());
            };
            if let Some(of) = old {
                fa.decref(of);
            }
            Ok(true)
        } else if t.prot == prot {
            // Private already, and the leaf says so: nothing changed (so no TLB
            // flush is owed — every kernel copy into a user buffer comes here).
            Ok(false)
        } else {
            // Private already: make sure the leaf carries the intended write bit
            // (a prior fork may have cleared it).
            Ok(space.protect(va, prot, false, phys))
        }
    }

    /// Copy `buf` into the pool at guest `addr`, privatizing copy-on-write pages
    /// first. The caller must have verified the pages are mapped (and, for a guest
    /// store, writable).
    fn copy_in(&mut self, addr: u64, buf: &[u8]) -> Result<(), MemError> {
        let (first, last) = self.page_range(addr, buf.len())?;
        self.back_file_pages(first, last)?;
        {
            let mut fa = lock_fa(&self.fa, self.memcg);
            let mut changed = false;
            for p in first..=last {
                let va = self.base + (p as u64) * PAGE_SIZE;
                // Demand-back the page (first touch), then privatize if it is a
                // copy-on-write share.
                if !Self::ensure_backed(&mut self.space, &self.phys, self.prot[p], &mut fa, va) {
                    return Err(MemError::Host("frame pool exhausted".into()));
                }
                changed |= Self::make_writable(
                    &mut self.space,
                    &self.phys,
                    self.prot[p],
                    &mut fa,
                    va,
                    self.shared_anon[p],
                )?;
            }
            if changed {
                mark_tlb_dirty!(self);
            }
        }
        self.icache_dirty |= self.any_exec(first, last);
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
    /// Like [`Self::read`], writing nothing never faults.
    pub fn write(&mut self, addr: u64, buf: &[u8]) -> Result<(), MemError> {
        if buf.is_empty() {
            return Ok(());
        }
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
        self.icache_dirty |= self.prot[p].contains(Prot::EXEC);
        let mut fa = lock_fa(&self.fa, self.memcg);
        // Out of frames is a real fault: reporting success here would retry the
        // store forever.
        match Self::make_writable(
            &mut self.space,
            &self.phys,
            self.prot[p],
            &mut fa,
            va,
            self.shared_anon[p],
        ) {
            Ok(changed) => {
                // The page has a new frame: a sibling thread's vcpu may still
                // translate it to the old one (now another process's copy),
                // and would read that.
                if changed {
                    mark_tlb_dirty!(self);
                }
                true
            }
            Err(_) => false,
        }
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
        self.back_file_pages(first, last)?;
        {
            let mut fa = lock_fa(&self.fa, self.memcg);
            let mut changed = false;
            for p in first..=last {
                let va = self.base + (p as u64) * PAGE_SIZE;
                if !Self::ensure_backed(&mut self.space, &self.phys, self.prot[p], &mut fa, va) {
                    return Err(MemError::Host("frame pool exhausted".into()));
                }
                changed |= Self::make_writable(
                    &mut self.space,
                    &self.phys,
                    self.prot[p],
                    &mut fa,
                    va,
                    self.shared_anon[p],
                )?;
            }
            if changed {
                mark_tlb_dirty!(self);
            }
        }
        self.icache_dirty |= self.any_exec(first, last);
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
    ///
    /// # Panics
    /// If the frame pool is exhausted. Guest-driven forks use
    /// [`GuestMemory::try_fork`]; this is for setting up known-small spaces.
    #[must_use]
    pub fn fork(&mut self) -> Self {
        self.try_fork().expect("fork: frame pool exhausted")
    }

    /// [`GuestMemory::fork`], or `None` with this space untouched if the frame
    /// pool can't hold the child's page tables.
    #[must_use]
    pub fn try_fork(&mut self) -> Option<Self> {
        let (child_space, parent_kstack, child_kstack) = {
            let mut fa = lock_fa(&self.fa, self.memcg);
            let mut child = self.space.fork_cow(&mut fa, &self.phys)?;
            // Past this point the fork can't fail, and it needs a couple more
            // frames: `fork_cow` just left us above the reserve, so these dip into
            // it by a bounded amount, and the next fork fails first.
            fa.set_use_reserve(true);
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
            fa.set_use_reserve(false);
            // `MAP_SHARED | MAP_ANONYMOUS` pages must stay a *single writable*
            // frame shared between parent and child, not copy-on-write. fork_cow
            // already increffed the frame and installed a (read-only) child leaf;
            // republish both sides' leaves writable so stores land in the shared
            // frame and each side sees the other's writes. These pages are eagerly
            // backed at mmap, so a frame always exists here.
            for (p, _) in self.shared_anon.non_default() {
                if self.mapped[p] {
                    let va = self.base + (p as u64) * PAGE_SIZE;
                    self.space.protect(va, self.prot[p], false, &self.phys);
                    child.protect(va, self.prot[p], false, &self.phys);
                }
            }
            // madvise fork policies: drop the child's leaf (its reference to
            // the parent's frame) for DONTFORK/WIPEONFORK pages — a WIPEONFORK
            // page stays mapped and demand-faults a fresh zero frame. (The
            // parent's own leaf was downgraded to copy-on-write by fork_cow; a
            // now-unshared frame is simply made writable again on first store.)
            for &p in self.fork_policy.keys() {
                let va = self.base + (p as u64) * PAGE_SIZE;
                if let Some(frame) = child.unmap(va, &mut fa, &self.phys) {
                    fa.free(frame);
                }
            }
            (child, parent_kstack, child_kstack)
        };
        // Every page is copy-on-write now, read-only in this space too: each
        // vcpu running it — every thread's, not only the forking one — must
        // drop the writable translations it holds, or a sibling thread's next
        // store goes straight into the frame the child now shares.
        mark_tlb_dirty!(self);
        self.kstack_pa = parent_kstack;
        let mut mapped = self.mapped.clone();
        let mut mapped_count = self.mapped_count;
        let mut shared_anon = self.shared_anon.clone();
        let mut fork_policy = std::collections::BTreeMap::new();
        for (&p, &policy) in &self.fork_policy {
            shared_anon[p] = false;
            if policy == FORK_DONT {
                mapped_count -= usize::from(mapped[p]);
                mapped[p] = false;
            } else {
                fork_policy.insert(p, policy);
            }
        }
        Some(Self {
            base: self.base,
            size: self.size,
            phys: Arc::clone(&self.phys),
            fa: Arc::clone(&self.fa),
            due: Arc::clone(&self.due),
            space: child_space,
            prot: self.prot.clone(),
            mapped,
            mapped_count,
            file_backed: self.file_backed.clone(),
            shared_anon,
            fork_policy,
            locks: super::pagelocks::PageLocks::default(),
            tlb_dirty: false,
            tlb_gen: 0,
            heap: self.heap,
            exec_gen: 0,
            icache_dirty: true,
            kstack_pa: child_kstack,
            shared: self.shared,
            memcg: self.memcg,
            fault_last: 0,
            fault_dir: 1,
            fault_window: 1,
            space_id: new_space_id(),
            files: self.files.clone(),
            shrink_seen: self.shrink_seen,
        })
    }

    /// Tear down the current page tables (returning their frames to the pool) and
    /// install a fresh empty address space in the *same* pool — the `execve` seam.
    /// The `CR3` changes; a KVM vcpu picks the new value up on its next run. The
    /// loader then maps the new image into this same `GuestMemory`.
    pub(crate) fn exec_reset(&mut self) {
        self.reset_space();
        self.exec_gen = self.exec_gen.wrapping_add(1);
    }

    /// Give up everything this address space holds when its last task exits,
    /// including its page tables and kernel stack. The
    /// task never runs again, but the slot outlives it, so anything kept here
    /// would add up over a long-lived process tree until the pool ran dry.
    pub(crate) fn retire(&mut self) {
        let mut fa = lock_fa(&self.fa, self.memcg);
        let old = std::mem::replace(&mut self.space, AddrSpace::retired());
        old.destroy(&mut fa, &self.phys);
        drop(fa);
        self.files.clear();
        self.kstack_pa = 0;
        self.prot.clear();
        self.mapped.clear();
        self.mapped_count = 0;
        self.file_backed.clear();
        self.shared_anon.clear();
        self.fork_policy.clear();
        mark_tlb_dirty!(self);
    }

    fn reset_space(&mut self) {
        let mut fa = lock_fa(&self.fa, self.memcg);
        // Exit and exec must always work, and this is what frees the frames a
        // full pool needs: the reserve guarantees the few it takes first. The
        // allocator lock is held throughout, so the old space's frames are back
        // before anyone else can dip in.
        fa.set_use_reserve(true);
        let mut ns = AddrSpace::new(&mut fa, &self.phys).expect("reset: pool exhausted");
        ctrl::map_into(&mut ns, &mut fa, &self.phys);
        let kstack_pa = install_kstack(&mut ns, &mut fa, &self.phys);
        let old = std::mem::replace(&mut self.space, ns);
        // The old space's private kstack frame is an ordinary (non-pinned) leaf,
        // so `destroy` frees it along with the rest; the shared control frames it
        // also references are pinned and survive.
        old.destroy(&mut fa, &self.phys);
        fa.set_use_reserve(false);
        self.kstack_pa = kstack_pa;
        drop(fa);
        self.files.clear();
        self.prot.clear();
        self.mapped.clear();
        self.mapped_count = 0;
        self.file_backed.clear();
        self.shared_anon.clear();
        self.fork_policy.clear();
        self.locks.clear();
        self.fault_last = 0;
        self.fault_window = 1;
        self.space_id = new_space_id();
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
    fn vdso_is_reachable_but_supervisor_control_pages_are_not() {
        // The vDSO/vvar sit in the control block *above* the flat extent (the
        // loader hands their address to libc via AT_SYSINFO_EHDR). A software
        // backend reaches memory through `check`, so those user pages must be
        // readable — the ELF header the guest parses starts with "\x7fELF".
        let m = mem();
        let ehdr = m.read_vec(ctrl::VDSO_VA, 4).expect("vDSO must be readable");
        assert_eq!(&ehdr, b"\x7fELF", "vDSO ELF magic at AT_SYSINFO_EHDR");
        assert!(m.can_exec(ctrl::VDSO_VA), "vDSO code page is executable");
        assert!(
            m.read_vec(ctrl::VVAR_VA, 8).is_ok(),
            "vvar page is readable"
        );
        // Supervisor-only control pages (GDT) stay unreachable from the guest.
        assert!(
            m.read_u32(ctrl::GDT_BASE).is_err(),
            "GDT is supervisor-only"
        );
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
        assert!(matches!(
            m.read_u64(boundary + 4),
            Err(MemError::Unmapped(_))
        ));
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
        assert_eq!(
            child.read_vec(0x1_0000 + 2 * PAGE_SIZE, 1).unwrap(),
            vec![2]
        );
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
        assert!(
            !m.cow_fault(0x1_0000, true),
            "read-only page is a genuine fault"
        );
    }

    #[test]
    fn fork_policies_wipe_or_drop_pages_in_the_child_only() {
        let mut m = mem();
        m.map(0x1_0000, 3 * PAGE_SIZE, Prot::rw()).unwrap();
        for i in 0..3 {
            m.write_u64(0x1_0000 + i * PAGE_SIZE, 0x1111 * (i + 1))
                .unwrap();
        }
        m.set_fork_policy(0x1_0000, PAGE_SIZE, FORK_WIPE).unwrap();
        m.set_fork_policy(0x1_0000 + PAGE_SIZE, PAGE_SIZE, FORK_DONT)
            .unwrap();
        assert!(
            m.set_fork_policy(0x1_0000 + 9 * PAGE_SIZE, PAGE_SIZE, FORK_DONT)
                .is_err()
        );
        let mut child = m.fork();
        // WIPEONFORK: zero in the child, intact in the parent (still writable).
        assert_eq!(child.read_u64(0x1_0000).unwrap(), 0);
        assert_eq!(m.read_u64(0x1_0000).unwrap(), 0x1111);
        m.write_u64(0x1_0000, 7).unwrap();
        // DONTFORK: absent from the child.
        assert!(child.read_u64(0x1_0000 + PAGE_SIZE).is_err());
        assert_eq!(m.read_u64(0x1_0000 + PAGE_SIZE).unwrap(), 0x2222);
        // An ordinary page is inherited.
        assert_eq!(child.read_u64(0x1_0000 + 2 * PAGE_SIZE).unwrap(), 0x3333);
        child.retire();
    }

    #[test]
    fn fork_and_retire_in_a_loop_leaks_no_frames() {
        let mut parent = mem();
        parent.map(0x1_0000, 4 * PAGE_SIZE, Prot::rw()).unwrap();
        parent.write_u64(0x1_0000, 1).unwrap();
        // The first pass settles the parent into copy-on-write shape.
        let mut first = parent.fork();
        first.retire();
        let settled = parent.frames_in_use();
        for _ in 0..50 {
            let mut child = parent.fork();
            child.write_u64(0x1_0000, 2).unwrap();
            child.retire();
            assert_eq!(parent.frames_in_use(), settled);
        }
        assert_eq!(parent.read_u64(0x1_0000).unwrap(), 1);
    }

    #[test]
    fn a_retired_space_finds_nothing_and_refuses_new_pages() {
        let mut m = mem();
        m.map(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        m.write_u64(0x1_0000, 1).unwrap();
        m.retire();
        assert!(m.read_u64(0x1_0000).is_err());
        assert!(m.write_u64(0x1_0000, 1).is_err());
    }

    #[test]
    fn a_fork_that_does_not_fit_in_the_pool_fails_and_changes_nothing() {
        let mut parent = GuestMemory::new(0x1_0000, 4 * 1024 * PAGE_SIZE);
        parent.map(0x1_0000, 16 * PAGE_SIZE, Prot::rw()).unwrap();
        parent.write_u64(0x1_0000, 7).unwrap();
        // Hold every frame the allocator will hand out.
        let held: Vec<u64> = {
            let mut fa = parent.fa.lock().unwrap();
            std::iter::from_fn(|| fa.alloc(&parent.phys)).collect()
        };
        let before = parent.frames_in_use();
        assert!(parent.try_fork().is_none());
        assert_eq!(parent.frames_in_use(), before);
        assert_eq!(parent.read_u64(0x1_0000).unwrap(), 7);
        // A write that would need a new frame fails too, rather than panicking,
        // and tearing the space down still works with the pool this full.
        assert!(parent.write_u64(0x1_0000 + PAGE_SIZE, 1).is_err());
        parent.release_frames(&held);
        assert!(parent.try_fork().is_some());
    }

    #[test]
    fn copy_on_write_with_no_frame_left_fails_without_touching_the_shared_page() {
        let mut parent = GuestMemory::new(0x1_0000, 4 * 1024 * PAGE_SIZE);
        parent.map(0x1_0000, PAGE_SIZE, Prot::rw()).unwrap();
        parent.write_u64(0x1_0000, 7).unwrap();
        let mut child = parent.fork();
        let held: Vec<u64> = {
            let mut fa = parent.fa.lock().unwrap();
            std::iter::from_fn(|| fa.alloc(&parent.phys)).collect()
        };
        assert!(child.write_u64(0x1_0000, 99).is_err());
        assert!(!child.cow_fault(0x1_0000, true), "reported, not retried");
        assert_eq!(parent.read_u64(0x1_0000).unwrap(), 7);
        parent.release_frames(&held);
        child.write_u64(0x1_0000, 99).unwrap();
        assert_eq!(parent.read_u64(0x1_0000).unwrap(), 7);
    }

    #[test]
    fn exec_reset_returns_frames_to_the_pool() {
        let mut m = mem();
        let baseline = m.frames_in_use(); // empty space: PML4 + control-block tables
        // Touch several pages so real data frames are allocated (demand-paged).
        m.map(0x1_0000, 8 * PAGE_SIZE, Prot::rw()).unwrap();
        for i in 0..8 {
            m.write_u64(0x1_0000 + i * PAGE_SIZE, 0xdead).unwrap();
        }
        assert!(
            m.frames_in_use() > baseline,
            "touched pages consumed frames"
        );
        // execve must return every data + table frame the process held.
        m.exec_reset();
        assert_eq!(
            m.frames_in_use(),
            baseline,
            "exec_reset returned all frames"
        );
    }

    #[test]
    fn shared_anon_pages_are_visible_across_fork() {
        let mut parent = mem();
        // A MAP_SHARED|ANON page and an ordinary (private/COW) page.
        parent
            .map_shared_anon(0x1_0000, PAGE_SIZE, Prot::rw())
            .unwrap();
        parent.map(0x1_1000, PAGE_SIZE, Prot::rw()).unwrap();
        parent.write_u64(0x1_0000, 100).unwrap();
        parent.write_u64(0x1_1000, 100).unwrap();

        let mut child = parent.fork();
        child.write_u64(0x1_0000, 999).unwrap();
        child.write_u64(0x1_1000, 999).unwrap();

        // The shared-anon page reflects the child's write; the private page stays
        // copy-on-write isolated.
        assert_eq!(
            parent.read_u64(0x1_0000).unwrap(),
            999,
            "shared-anon visible across fork"
        );
        assert_eq!(
            parent.read_u64(0x1_1000).unwrap(),
            100,
            "private page stays COW-isolated"
        );

        // Sharing is bidirectional: the parent's write is visible to the child.
        parent.write_u64(0x1_0000, 42).unwrap();
        assert_eq!(
            child.read_u64(0x1_0000).unwrap(),
            42,
            "shared-anon is bidirectional"
        );
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

    #[test]
    fn switching_to_arm64_tables_keeps_the_contents() {
        use crate::abi::Arch;
        let base = 0x1_0000;
        let mut m = GuestMemory::new(base, 64 * PAGE_SIZE);
        m.map(base, 4 * PAGE_SIZE, Prot::rw()).unwrap();
        m.write_init(base + 0x10, b"before").unwrap();
        assert_eq!(m.page_table_format(), PtFormat::X86_64);
        m.set_guest_arch(Arch::Aarch64).unwrap();
        assert_eq!(m.page_table_format(), PtFormat::Aarch64);
        assert_ne!(m.phys_base(), 0);
        assert_eq!(m.read_vec(base + 0x10, 6).unwrap(), b"before");
        m.write(base + 2 * PAGE_SIZE, b"after").unwrap();
        assert_eq!(m.read_vec(base + 2 * PAGE_SIZE, 5).unwrap(), b"after");
        // Copy-on-write still works in the new format.
        let mut child = m.fork();
        child.write(base + 0x10, b"child!").unwrap();
        assert_eq!(m.read_vec(base + 0x10, 6).unwrap(), b"before");
        assert_eq!(child.read_vec(base + 0x10, 6).unwrap(), b"child!");
        assert!(
            m.set_guest_arch(Arch::X86_64).is_err(),
            "a shared pool keeps its format"
        );
        m.set_guest_arch(Arch::Aarch64).unwrap(); // no change is fine
    }
}
