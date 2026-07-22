//! The supervisor control block, shared by every address space.
//!
//! Phase 3 of the MMU refactor retired the flat identity-mapped model. Guest
//! user memory now lives in real per-process page tables ([`super::pagetable`])
//! over one shared physical pool ([`super::phys`]). But a KVM guest at CPL3 still
//! needs a fixed supervisor scaffold to trap syscalls and faults into: a GDT, an
//! IDT (only `#PF`), a 64-bit TSS (with `RSP0` = the kernel stack top), the
//! `hlt; sysretq` syscall trampoline `IA32_LSTAR` points at, and the `#PF`
//! trampoline (`hlt`). This module owns that scaffold.
//!
//! Unlike the old control block (which also carried the identity page tables),
//! this one is *only* the six supervisor pages. They are built once into a small
//! run of reserved (pinned) frames near the base of the pool, and mapped —
//! supervisor-only (`US = 0`), at the same fixed virtual addresses they always
//! had ([`GDT_BASE`], [`LSTAR_VA`], …) — into *every* address space so a
//! `KVM_SET_SREGS` `cr3` switch never loses them.
//!
//! The virtual addresses are unchanged from the pre-refactor layout (the vcpu's
//! `sregs`/MSRs bake them in); only the *physical* backing moved from a separate
//! `Region` into pool frames. `read_u64` therefore resolves a control-block
//! virtual address to its pool physical address to read back the exception frame
//! the CPU pushes onto the kernel stack.

use super::mem::{PAGE_SIZE, Prot};
use super::phys::{FrameAllocator, PhysMem};
use super::pagetable::AddrSpace;

/// Top of the (old) identity window; the control block sits just below it, so
/// its virtual addresses stay far above any guest user mapping.
const IDENTITY_TOP: u64 = 64 << 30;

/// Virtual base of the control block — unchanged from the identity-model layout.
pub const CTRL_GPA: u64 = IDENTITY_TOP - (2 << 20);

// Virtual offsets of each page within the control window. These are frozen at
// their historical values so `GDT_BASE`/`LSTAR_VA`/… — which the vcpu's sregs,
// GDTR/IDTR, and LSTAR MSR reference — never move. (The 0x42000 gap once held
// the identity page tables; it is now simply unused virtual space.)
const GDT_OFF: u64 = 0x42000;
const TRAMP_OFF: u64 = 0x43000;
const FAULT_TRAMP_OFF: u64 = 0x44000;
const IDT_OFF: u64 = 0x45000;
const TSS_OFF: u64 = 0x46000;
const KSTACK_OFF: u64 = 0x47000;
const VDSO_OFF: u64 = 0x48000;
const VVAR_OFF: u64 = 0x49000;

/// The virtual address `IA32_LSTAR` points at: the `hlt; sysretq` trampoline.
pub const LSTAR_VA: u64 = CTRL_GPA + TRAMP_OFF;
/// Linear address of the GDT.
pub const GDT_BASE: u64 = CTRL_GPA + GDT_OFF;
/// The `#PF` trampoline's virtual address (the host recognizes a fault exit by
/// the vcpu `rip` landing just past it).
pub const FAULT_TRAMP_VA: u64 = CTRL_GPA + FAULT_TRAMP_OFF;
/// Linear address of the IDT and of the TSS.
pub const IDT_BASE: u64 = CTRL_GPA + IDT_OFF;
pub const TSS_BASE: u64 = CTRL_GPA + TSS_OFF;
/// Kernel stack top the TSS switches to on a CPL3→CPL0 exception (`RSP0`).
pub const KSTACK_TOP: u64 = CTRL_GPA + KSTACK_OFF + 0x1000;
/// Virtual base of the kernel-stack page. Unlike the rest of the control block
/// this page is **per address space** — each process maps its own frame here
/// (see [`map_kstack`]) so that under SMP two vcpus in *different* address spaces
/// taking a `#PF` at once push their CPU exception frames onto different physical
/// pages instead of clobbering one shared kstack (which corrupted the recovered
/// user state — demand paging makes faults frequent, so collisions were common).
///
/// This isolates distinct processes (the overwhelmingly common concurrent case).
/// `CLONE_THREAD` siblings that share one address space still share this frame;
/// making the kstack fully per-vcpu (a per-vcpu `RSP0`/TSS) is the follow-up for
/// heavily-threaded SMP guests.
pub const KSTACK_PAGE_VA: u64 = CTRL_GPA + KSTACK_OFF;

/// Virtual base of the vDSO ELF image (user-readable/executable) — the value the
/// loader advertises as `AT_SYSINFO_EHDR`. See [`super::vdso`].
pub const VDSO_VA: u64 = CTRL_GPA + VDSO_OFF;
/// Virtual base of the vDSO's "vvar" data page (user-readable): the host writes
/// the `rdtsc` → nanoseconds calibration here (see [`write_vvar`]) and the vDSO
/// code reads it.
pub const VVAR_VA: u64 = CTRL_GPA + VVAR_OFF;

/// GDT selector of the TSS descriptor (a 16-byte descriptor at slots 5–6).
pub const SEL_TSS: u16 = 0x28;
pub const SEL_KCODE: u16 = 0x08;
#[allow(dead_code)]
pub const SEL_KDATA: u16 = 0x10;
pub const SEL_UDATA: u16 = 0x18 | 3;
pub const SEL_UCODE: u16 = 0x20 | 3;
/// `IA32_STAR`: `sysretq` base selector in [63:48], `syscall` CS in [47:32].
pub const STAR_VALUE: u64 =
    (((SEL_UDATA as u64 & !3) - 8) | 3) << 48 | (SEL_KCODE as u64) << 32;
/// GDTR limit: 5 segment descriptors + a 16-byte TSS descriptor.
pub const GDT_LIMIT: u16 = 7 * 8 - 1;

/// Physical base of the control block's frames in the pool — the first
/// allocatable frame after the pinned null frame. Held out of circulation by
/// [`reserve_and_build`].
pub const CTRL_PHYS_BASE: u64 = PAGE_SIZE;
/// Number of *shared* control-block pages: GDT, syscall trampoline, `#PF`
/// trampoline, IDT, TSS, then the vDSO code page and its vvar data page. The
/// kernel stack is **not** here — it is per address space (see
/// [`KSTACK_PAGE_VA`] / [`map_kstack`]).
pub const CTRL_FRAMES: u64 = 7;
/// Physical-run index of the vDSO code frame and the vvar frame.
const VDSO_FIDX: usize = 5;
const VVAR_FIDX: usize = 6;
/// Bytes the control block reserves in the pool.
pub const CTRL_SIZE: u64 = CTRL_FRAMES * PAGE_SIZE;

/// One control-block page: its virtual offset within the window, its leaf
/// protection, and its index into the reserved physical run.
struct Page {
    voff: u64,
    prot: Prot,
}

/// The five shared pages, in physical order starting at [`CTRL_PHYS_BASE`].
const PAGES: [Page; 5] = [
    Page { voff: GDT_OFF, prot: Prot::rw() },
    Page { voff: TRAMP_OFF, prot: Prot::rx() },
    Page { voff: FAULT_TRAMP_OFF, prot: Prot::rx() },
    Page { voff: IDT_OFF, prot: Prot::rw() },
    Page { voff: TSS_OFF, prot: Prot::rw() },
];

/// Physical address of the control-block page at physical index `i`.
fn frame_pa(i: usize) -> u64 {
    CTRL_PHYS_BASE + i as u64 * PAGE_SIZE
}

/// Reserve the control-block frames (pinning them so `alloc`/CoW never touch
/// them) and write the GDT/IDT/TSS/trampolines into them. Call once, right after
/// the pool + allocator are built, before any user allocation.
pub fn reserve_and_build(fa: &mut FrameAllocator, phys: &PhysMem) {
    fa.reserve(CTRL_PHYS_BASE, CTRL_SIZE);

    let gdt_pa = frame_pa(0);
    let tramp_pa = frame_pa(1);
    let fault_pa = frame_pa(2);
    let idt_pa = frame_pa(3);
    let tss_pa = frame_pa(4);

    // GDT: null, kernel code64, kernel data, user data, user code64.
    let gdt: [u64; 5] = [
        0,
        0x00AF_9A00_0000_FFFF, // 0x08 kernel code (L=1, DPL0)
        0x00CF_9200_0000_FFFF, // 0x10 kernel data
        0x00CF_F200_0000_FFFF, // 0x18 user data  (DPL3)
        0x00AF_FA00_0000_FFFF, // 0x20 user code  (L=1, DPL3)
    ];
    for (i, d) in gdt.iter().enumerate() {
        phys.write(gdt_pa + i as u64 * 8, &d.to_le_bytes());
    }
    // TSS descriptor (16 bytes, GDT slots 5–6): a 64-bit available TSS at
    // TSS_BASE, limit 0x67.
    let base = TSS_BASE;
    let limit = 0x67u64;
    let lo = (limit & 0xFFFF)
        | ((base & 0xFFFF) << 16)
        | (((base >> 16) & 0xFF) << 32)
        | (0x8Bu64 << 40)
        | (((limit >> 16) & 0xF) << 48)
        | (((base >> 24) & 0xFF) << 56);
    let hi = (base >> 32) & 0xFFFF_FFFF;
    phys.write(gdt_pa + 5 * 8, &lo.to_le_bytes());
    phys.write(gdt_pa + 6 * 8, &hi.to_le_bytes());

    // The 64-bit TSS: only RSP0 (offset 4) matters.
    phys.write(tss_pa + 4, &KSTACK_TOP.to_le_bytes());
    phys.write(tss_pa + 102, &0x68u16.to_le_bytes()); // iomap base past the TSS

    // IDT entry 14 (#PF) → the fault trampoline (64-bit interrupt gate, CPL0).
    let off = FAULT_TRAMP_VA;
    let gate_lo = (off & 0xFFFF)
        | ((SEL_KCODE as u64) << 16)
        | (0x8Eu64 << 40)
        | (((off >> 16) & 0xFFFF) << 48);
    let gate_hi = (off >> 32) & 0xFFFF_FFFF;
    phys.write(idt_pa + 14 * 16, &gate_lo.to_le_bytes());
    phys.write(idt_pa + 14 * 16 + 8, &gate_hi.to_le_bytes());

    // Trampolines: the syscall one (`IA32_LSTAR`) and the #PF one.
    phys.write(tramp_pa, &[0xF4, 0x48, 0x0F, 0x07]); // hlt ; sysretq
    phys.write(fault_pa, &[0xF4]); // hlt

    // vDSO ELF image (its clock code reads the vvar page at VVAR_VA). The vvar
    // frame is left zeroed — `mult == 0` makes the vDSO fall back to the syscall
    // until [`write_vvar`] fills in the TSC calibration after the vcpu exists.
    phys.write(frame_pa(VDSO_FIDX), &super::vdso::build_image(VVAR_VA));
}

/// Write the vDSO's vvar calibration (TSC → nanoseconds) into the shared vvar
/// frame. Called once after the vcpu is created and the TSC is read, before the
/// guest runs. A non-zero `mult` also arms the vDSO fast path (it treats
/// `mult == 0` as "not calibrated" and uses the syscall instead).
pub fn write_vvar(phys: &PhysMem, mult: u64, shift: u64, base_tsc: u64, base_mono_ns: u64, base_wall_ns: u64) {
    use super::vdso::vvar;
    let pa = frame_pa(VVAR_FIDX);
    phys.write(pa + vvar::MULT, &mult.to_le_bytes());
    phys.write(pa + vvar::SHIFT, &shift.to_le_bytes());
    phys.write(pa + vvar::BASE_TSC, &base_tsc.to_le_bytes());
    phys.write(pa + vvar::BASE_MONO_NS, &base_mono_ns.to_le_bytes());
    phys.write(pa + vvar::BASE_WALL_NS, &base_wall_ns.to_le_bytes());
}

/// Map the shared control block, supervisor-only, at its fixed virtual addresses
/// into `space`. Idempotent: mapping over an existing (e.g. CoW-cleared) leaf
/// simply republishes the correct supervisor entry, and the reserved frames are
/// pinned so the returned old-frame is never freed. Does **not** map the kernel
/// stack — that is per address space (see [`map_kstack`]). Call after creating or
/// forking an address space.
pub fn map_into(space: &mut AddrSpace, fa: &mut FrameAllocator, phys: &PhysMem) {
    for (i, p) in PAGES.iter().enumerate() {
        let _ = space.map(CTRL_GPA + p.voff, frame_pa(i), p.prot, true, fa, phys);
    }
    // The vDSO code (user read/execute) and its vvar data (user read-only) —
    // mapped `supervisor = false` so the guest can call the clock code at CPL3.
    // These are the only user-accessible control-block pages.
    let _ = space.map(VDSO_VA, frame_pa(VDSO_FIDX), Prot::rx(), false, fa, phys);
    let _ = space.map(VVAR_VA, frame_pa(VVAR_FIDX), Prot::READ, false, fa, phys);
}

/// Map a private kernel-stack `frame` at [`KSTACK_PAGE_VA`] into `space`,
/// supervisor-RW. Each address space owns its own kstack frame so concurrent
/// `#PF`s on sibling vcpus never share (and clobber) one CPU exception-frame
/// page. Returns the frame this replaced (a CoW-shared kstack from `fork`, say),
/// for the caller to release.
pub fn map_kstack(
    space: &mut AddrSpace,
    fa: &mut FrameAllocator,
    phys: &PhysMem,
    frame: u64,
) -> Option<u64> {
    space
        .map(KSTACK_PAGE_VA, frame, Prot::rw(), true, fa, phys)
        .ok()
        .flatten()
}
