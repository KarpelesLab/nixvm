//! A minimal x86-64 vDSO so the guest can read the clock **without a syscall**.
//!
//! Clock-polling runtimes (Bun/JSC issues ~89% `clock_gettime`) otherwise pay a
//! full KVM exit per clock read (~6µs) where native hardware pays a vDSO read
//! (~150ns). We map a tiny ELF exporting the clock functions the real Linux
//! x86-64 vDSO provides — `__vdso_clock_gettime`, `__vdso_gettimeofday`,
//! `__vdso_clock_getres`, `__vdso_time`, `__vdso_getcpu` — plus a "vvar" data
//! page the host fills with a `rdtsc` → nanoseconds calibration; the vDSO code
//! reads `rdtsc` and scales it, all in guest userspace.
//!
//! libc finds the symbols via `AT_SYSINFO_EHDR` + the ELF's `DT_HASH`/
//! `DT_SYMTAB`. The symbols are unversioned: musl matches by name only, and
//! glibc accepts an unversioned symbol when the object carries no `DT_VERSYM`
//! (its `check_match` takes the "no version table → accept" path), so both use
//! the fast path. Every function falls back to the raw syscall for the cases the
//! calibration can't serve — an uncalibrated vvar, and the clocks the real vDSO
//! also punts (`CLOCK_PROCESS_CPUTIME_ID`/`CLOCK_THREAD_CPUTIME_ID` and anything
//! above `CLOCK_MONOTONIC_COARSE`) — so it only ever *adds* a fast path and never
//! returns a wrong answer for a clock it doesn't truly compute.
//!
//! Both the code and the vvar page live in the shared control block
//! ([`super::ctrl`]) mapped user-readable/executable into every address space at
//! a fixed high VA — outside the guest's 32 GiB range, so it never collides with
//! its heap/mmap/stack and is inherited by `fork`/`execve` for free.

/// Byte layout of the vvar data page the host fills (all little-endian). The
/// vDSO reads these; the host writes them once at startup after calibrating the
/// TSC. Kept in sync with the assembled code's `[r8+N]` offsets.
pub mod vvar {
    // Offset 0 is a reserved sequence field (for a future seqlock); the set-once
    // v1 does not use it.
    /// `mult`: `ns ≈ (tsc_delta * mult) >> shift`.
    pub const MULT: u64 = 8;
    /// `shift` (only the low byte is read by `shrd`).
    pub const SHIFT: u64 = 16;
    /// `base_tsc`: the `rdtsc` value captured at calibration.
    pub const BASE_TSC: u64 = 24;
    /// `base_mono_ns`: `CLOCK_MONOTONIC` nanoseconds at `base_tsc`.
    pub const BASE_MONO_NS: u64 = 32;
    /// `base_wall_ns`: `CLOCK_REALTIME` nanoseconds at `base_tsc`.
    pub const BASE_WALL_NS: u64 = 40;
}

/// The assembled vDSO function bytes (assembled from `vdso_x86.s`; see the source
/// tree for the `.s`). Each `movabs r8, <imm64>` placeholder (offsets
/// [`VVAR_PATCH`]) gets the real vvar VA written in at build time. Symbol entry
/// offsets are [`SYMS`].
const CODE: &[u8] = &[
    0x48, 0x83, 0xff, 0x06, 0x77, 0x62, 0x48, 0x83, 0xff, 0x02, 0x74, 0x5c, 0x48, 0x83, 0xff, 0x03,
    0x74, 0x56, 0x49, 0xb8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x49, 0x83, 0x78, 0x08,
    0x00, 0x74, 0x45, 0x0f, 0x31, 0x48, 0xc1, 0xe2, 0x20, 0x48, 0x09, 0xd0, 0x49, 0x2b, 0x40, 0x18,
    0x49, 0xf7, 0x60, 0x08, 0x41, 0x8a, 0x48, 0x10, 0x48, 0x0f, 0xad, 0xd0, 0x48, 0x83, 0xff, 0x00,
    0x74, 0x0c, 0x48, 0x83, 0xff, 0x05, 0x74, 0x06, 0x49, 0x03, 0x40, 0x20, 0xeb, 0x04, 0x49, 0x03,
    0x40, 0x28, 0x31, 0xd2, 0x48, 0xc7, 0xc1, 0x00, 0xca, 0x9a, 0x3b, 0x48, 0xf7, 0xf1, 0x48, 0x89,
    0x06, 0x48, 0x89, 0x56, 0x08, 0x31, 0xc0, 0xc3, 0xb8, 0xe4, 0x00, 0x00, 0x00, 0x0f, 0x05, 0xc3,
    0x48, 0x85, 0xf6, 0x74, 0x0d, 0xc7, 0x06, 0x00, 0x00, 0x00, 0x00, 0xc7, 0x46, 0x04, 0x00, 0x00,
    0x00, 0x00, 0x48, 0x85, 0xff, 0x74, 0x50, 0x49, 0xb8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22,
    0x11, 0x49, 0x83, 0x78, 0x08, 0x00, 0x74, 0x42, 0x0f, 0x31, 0x48, 0xc1, 0xe2, 0x20, 0x48, 0x09,
    0xd0, 0x49, 0x2b, 0x40, 0x18, 0x49, 0xf7, 0x60, 0x08, 0x41, 0x8a, 0x48, 0x10, 0x48, 0x0f, 0xad,
    0xd0, 0x49, 0x03, 0x40, 0x28, 0x31, 0xd2, 0x48, 0xc7, 0xc1, 0x00, 0xca, 0x9a, 0x3b, 0x48, 0xf7,
    0xf1, 0x48, 0x89, 0x07, 0x48, 0x89, 0xd0, 0x31, 0xd2, 0x48, 0xc7, 0xc1, 0xe8, 0x03, 0x00, 0x00,
    0x48, 0xf7, 0xf1, 0x48, 0x89, 0x47, 0x08, 0x31, 0xc0, 0xc3, 0xb8, 0x60, 0x00, 0x00, 0x00, 0x0f,
    0x05, 0xc3, 0x48, 0x83, 0xff, 0x06, 0x77, 0x23, 0x48, 0x83, 0xff, 0x02, 0x74, 0x1d, 0x48, 0x83,
    0xff, 0x03, 0x74, 0x17, 0x48, 0x85, 0xf6, 0x74, 0x0f, 0x48, 0xc7, 0x06, 0x00, 0x00, 0x00, 0x00,
    0x48, 0xc7, 0x46, 0x08, 0x01, 0x00, 0x00, 0x00, 0x31, 0xc0, 0xc3, 0xb8, 0xe5, 0x00, 0x00, 0x00,
    0x0f, 0x05, 0xc3, 0x49, 0xb8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x49, 0x83, 0x78,
    0x08, 0x00, 0x74, 0x32, 0x0f, 0x31, 0x48, 0xc1, 0xe2, 0x20, 0x48, 0x09, 0xd0, 0x49, 0x2b, 0x40,
    0x18, 0x49, 0xf7, 0x60, 0x08, 0x41, 0x8a, 0x48, 0x10, 0x48, 0x0f, 0xad, 0xd0, 0x49, 0x03, 0x40,
    0x28, 0x31, 0xd2, 0x48, 0xc7, 0xc1, 0x00, 0xca, 0x9a, 0x3b, 0x48, 0xf7, 0xf1, 0x48, 0x85, 0xff,
    0x74, 0x03, 0x48, 0x89, 0x07, 0xc3, 0xb8, 0xc9, 0x00, 0x00, 0x00, 0x0f, 0x05, 0xc3, 0x48, 0x85,
    0xff, 0x74, 0x06, 0xc7, 0x07, 0x00, 0x00, 0x00, 0x00, 0x48, 0x85, 0xf6, 0x74, 0x06, 0xc7, 0x06,
    0x00, 0x00, 0x00, 0x00, 0x31, 0xc0, 0xc3,
];
/// Offsets of the `movabs r8, VVAR_VA` immediates to patch (one per function that
/// reads the calibration).
const VVAR_PATCH: [usize; 3] = [20, 137, 277];

/// One exported function: its name (in dynstr) and its byte offset within [`CODE`].
struct Sym {
    name: &'static str,
    off: u64,
}
/// The exported symbol set, matching the real Linux x86-64 vDSO. Order defines
/// the `.dynsym` layout (index 1..=N; index 0 is the reserved null entry) and the
/// hash chain.
const SYMS: &[Sym] = &[
    Sym { name: "__vdso_clock_gettime", off: 0x0 },
    Sym { name: "__vdso_gettimeofday", off: 0x70 },
    Sym { name: "__vdso_clock_getres", off: 0xe2 },
    Sym { name: "__vdso_time", off: 0x113 },
    Sym { name: "__vdso_getcpu", off: 0x15e },
];

// ---- ELF-image layout (all within one 4 KiB page, ET_DYN, load base = 0) ----
// The ELF header sits at offset 0. Everything below is a fixed offset into the
// one-page image; `build_image` writes each region there. Sized for the 5-symbol
// table (see the overlap assertions in `build_image`).
const PHDR: u64 = 0x40; // 2 × 56-byte program headers
const HASH: u64 = 0xb0; // SysV hash: (2 + nbucket + nchain) × 4 = 36 bytes
const DYNSYM: u64 = 0xe0; // (1 + SYMS.len()) × 24-byte symbols
const DYNSTR: u64 = 0x170;
const DYNAMIC: u64 = 0x1d0; // 6 × 16-byte entries
const TEXT: u64 = 0x240;

/// Size of the vDSO ELF page (one page).
pub const PAGE: usize = 4096;

fn w64(buf: &mut [u8], off: u64, v: u64) {
    buf[off as usize..off as usize + 8].copy_from_slice(&v.to_le_bytes());
}
fn w32(buf: &mut [u8], off: u64, v: u32) {
    buf[off as usize..off as usize + 4].copy_from_slice(&v.to_le_bytes());
}
fn w16(buf: &mut [u8], off: u64, v: u16) {
    buf[off as usize..off as usize + 2].copy_from_slice(&v.to_le_bytes());
}

/// Build the 4 KiB vDSO ELF image, with the `movabs` clock-page immediates
/// patched to `vvar_va` (the absolute guest VA the vvar page is mapped at).
#[must_use]
pub fn build_image(vvar_va: u64) -> [u8; PAGE] {
    let mut b = [0u8; PAGE];

    // ELF header: ET_DYN, EM_X86_64.
    b[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
    b[4] = 2; // ELFCLASS64
    b[5] = 1; // ELFDATA2LSB
    b[6] = 1; // EV_CURRENT
    w16(&mut b, 16, 3); // e_type = ET_DYN
    w16(&mut b, 18, 62); // e_machine = EM_X86_64
    w32(&mut b, 20, 1); // e_version
    w64(&mut b, 32, PHDR); // e_phoff
    w16(&mut b, 52, 64); // e_ehsize
    w16(&mut b, 54, 56); // e_phentsize
    w16(&mut b, 56, 2); // e_phnum

    // Program header 0: PT_LOAD covering the whole page, R+X.
    w32(&mut b, PHDR, 1); // p_type = PT_LOAD
    w32(&mut b, PHDR + 4, 0x5); // p_flags = R|X
    w64(&mut b, PHDR + 8, 0); // p_offset
    w64(&mut b, PHDR + 16, 0); // p_vaddr
    w64(&mut b, PHDR + 24, 0); // p_paddr
    w64(&mut b, PHDR + 32, PAGE as u64); // p_filesz
    w64(&mut b, PHDR + 40, PAGE as u64); // p_memsz
    w64(&mut b, PHDR + 48, PAGE as u64); // p_align

    // Program header 1: PT_DYNAMIC.
    w32(&mut b, PHDR + 56, 2); // p_type = PT_DYNAMIC
    w32(&mut b, PHDR + 56 + 4, 0x4); // p_flags = R
    w64(&mut b, PHDR + 56 + 8, DYNAMIC); // p_offset
    w64(&mut b, PHDR + 56 + 16, DYNAMIC); // p_vaddr
    w64(&mut b, PHDR + 56 + 24, DYNAMIC); // p_paddr
    w64(&mut b, PHDR + 56 + 32, 6 * 16); // p_filesz
    w64(&mut b, PHDR + 56 + 40, 6 * 16); // p_memsz
    w64(&mut b, PHDR + 56 + 48, 8); // p_align

    let nsym = SYMS.len() as u64; // exported symbols (index 1..=nsym)
    let nchain = nsym + 1; // + the reserved null symbol at index 0

    // SysV hash: a single bucket whose chain threads every symbol (1→2→…→N→end),
    // so a lookup walks all names — fine for this tiny table and version-agnostic.
    w32(&mut b, HASH, 1); // nbucket
    w32(&mut b, HASH + 4, nchain as u32); // nchain (= total symbol slots)
    w32(&mut b, HASH + 8, 1); // bucket[0] = first symbol index
    w32(&mut b, HASH + 12, 0); // chain[0] (null symbol)
    for i in 1..=nsym {
        // chain[i] → i+1, and the last links to STN_UNDEF (0) to end the chain.
        let next = if i == nsym { 0 } else { i + 1 };
        w32(&mut b, HASH + 12 + i * 4, next as u32);
    }

    // Dynamic string table: a leading NUL, then each name NUL-terminated. Record
    // each name's offset for the matching .dynsym entry.
    let mut str_off = 1u64; // byte 0 is the empty string
    let mut name_offs = [0u64; 8];
    for (i, s) in SYMS.iter().enumerate() {
        name_offs[i] = str_off;
        let at = (DYNSTR + str_off) as usize;
        b[at..at + s.name.len()].copy_from_slice(s.name.as_bytes());
        str_off += s.name.len() as u64 + 1; // + the NUL terminator
    }
    let strsz = str_off;

    // Dynamic symbols. sym[0] is the reserved null entry (already zero); each
    // exported function follows. st_info = (STB_GLOBAL<<4)|STT_FUNC = 0x12;
    // st_shndx = 1 (defined in the single load segment).
    for (i, s) in SYMS.iter().enumerate() {
        let sym = DYNSYM + (i as u64 + 1) * 24;
        w32(&mut b, sym, name_offs[i] as u32); // st_name
        b[sym as usize + 4] = 0x12; // st_info
        w16(&mut b, sym + 6, 1); // st_shndx
        w64(&mut b, sym + 8, TEXT + s.off); // st_value
        let end = SYMS.get(i + 1).map_or(CODE.len() as u64, |n| n.off);
        w64(&mut b, sym + 16, end - s.off); // st_size
    }

    // Dynamic section: DT_HASH, DT_STRTAB, DT_SYMTAB, DT_STRSZ, DT_SYMENT, DT_NULL.
    let dyn_entries: [(u64, u64); 6] = [
        (4, HASH),    // DT_HASH
        (5, DYNSTR),  // DT_STRTAB
        (6, DYNSYM),  // DT_SYMTAB
        (10, strsz),  // DT_STRSZ
        (11, 24),     // DT_SYMENT
        (0, 0),       // DT_NULL
    ];
    for (i, (tag, val)) in dyn_entries.iter().enumerate() {
        let o = DYNAMIC + (i as u64) * 16;
        w64(&mut b, o, *tag);
        w64(&mut b, o + 8, *val);
    }

    // Guard the fixed layout against overlap as the table grows.
    debug_assert!(HASH + 8 + nchain * 4 <= DYNSYM, "hash table overlaps dynsym");
    debug_assert!(DYNSYM + nchain * 24 <= DYNSTR, "dynsym overlaps dynstr");
    debug_assert!(DYNSTR + strsz <= DYNAMIC, "dynstr overlaps dynamic");
    debug_assert!(TEXT as usize + CODE.len() <= PAGE, "code overruns the page");

    // Code, with the vvar VA patched into each `movabs` immediate.
    b[TEXT as usize..TEXT as usize + CODE.len()].copy_from_slice(CODE);
    for &p in &VVAR_PATCH {
        assert_eq!(&b[TEXT as usize + p..TEXT as usize + p + 8], &0x1122_3344_5566_7788u64.to_le_bytes(), "VVAR_PATCH offset must land on the movabs sentinel");
        w64(&mut b, TEXT + p as u64, vvar_va);
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse the built image the way musl's `__vdsosym` / glibc's `setup_vdso`
    /// do and confirm every exported symbol resolves to its code offset — a
    /// regression guard on the ELF.
    #[test]
    fn image_exports_all_clock_symbols() {
        let vvar = 0xF_FFE4_9000u64;
        let img = build_image(vvar);
        // ELF magic + ET_DYN.
        assert_eq!(&img[0..4], &[0x7f, b'E', b'L', b'F']);
        assert_eq!(u16::from_le_bytes([img[16], img[17]]), 3);

        // Walk PT_DYNAMIC → collect DT_HASH/DT_STRTAB/DT_SYMTAB.
        let phoff = u64::from_le_bytes(img[32..40].try_into().unwrap());
        let (mut hash, mut strtab, mut symtab) = (0u64, 0u64, 0u64);
        for i in 0..2u64 {
            let p = (phoff + i * 56) as usize;
            if u32::from_le_bytes(img[p..p + 4].try_into().unwrap()) == 2 {
                let mut d = u64::from_le_bytes(img[p + 8..p + 16].try_into().unwrap());
                loop {
                    let tag = u64::from_le_bytes(img[d as usize..d as usize + 8].try_into().unwrap());
                    let val = u64::from_le_bytes(img[d as usize + 8..d as usize + 16].try_into().unwrap());
                    match tag {
                        4 => hash = val,
                        5 => strtab = val,
                        6 => symtab = val,
                        0 => break,
                        _ => {}
                    }
                    d += 16;
                }
            }
        }
        assert!(hash != 0 && strtab != 0 && symtab != 0);

        // Walk the whole hash chain from bucket 0 and match names, exactly as a
        // dynamic linker resolving against DT_HASH would.
        let nbucket = u32::from_le_bytes(img[hash as usize..hash as usize + 4].try_into().unwrap());
        assert_eq!(nbucket, 1);
        let sym_name = |idx: u64| -> String {
            let s = (symtab + idx * 24) as usize;
            let n = u32::from_le_bytes(img[s..s + 4].try_into().unwrap()) as u64;
            let start = (strtab + n) as usize;
            let end = img[start..].iter().position(|&c| c == 0).unwrap() + start;
            String::from_utf8_lossy(&img[start..end]).into_owned()
        };
        let val = |idx: u64| u64::from_le_bytes(img[(symtab + idx * 24 + 8) as usize..(symtab + idx * 24 + 16) as usize].try_into().unwrap());
        // Follow bucket[0] → chain until STN_UNDEF, collecting (name → value).
        let mut resolved = std::collections::HashMap::new();
        let mut idx = u64::from(u32::from_le_bytes(img[hash as usize + 8..hash as usize + 12].try_into().unwrap()));
        while idx != 0 {
            resolved.insert(sym_name(idx), val(idx));
            let chain = hash + 8 + nbucket as u64 * 4 + idx * 4;
            idx = u64::from(u32::from_le_bytes(img[chain as usize..chain as usize + 4].try_into().unwrap()));
        }
        // Every exported symbol resolves to the right code offset.
        for s in SYMS {
            assert_eq!(resolved.get(s.name), Some(&(TEXT + s.off)), "symbol {}", s.name);
        }
        assert_eq!(resolved.len(), SYMS.len(), "chain reaches exactly the exports");

        // The vvar VA was patched into every movabs immediate.
        for &p in &VVAR_PATCH {
            let got = u64::from_le_bytes(img[(TEXT + p as u64) as usize..(TEXT + p as u64) as usize + 8].try_into().unwrap());
            assert_eq!(got, vvar);
        }
    }
}
