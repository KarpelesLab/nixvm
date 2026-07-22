//! A minimal x86-64 vDSO so the guest can read the clock **without a syscall**.
//!
//! Clock-polling runtimes (Bun/JSC issues ~89% `clock_gettime`) otherwise pay a
//! full KVM exit per clock read (~6µs) where native hardware pays a vDSO read
//! (~150ns). We map a tiny ELF exporting `__vdso_clock_gettime` /
//! `__vdso_gettimeofday`, plus a "vvar" data page the host fills with a `rdtsc`
//! → nanoseconds calibration; the vDSO code reads `rdtsc` and scales it, all in
//! guest userspace. musl finds the symbols via `AT_SYSINFO_EHDR` + the ELF's
//! `DT_HASH`/`DT_SYMTAB` (no versioned symbols — musl skips the version check
//! when the vDSO has no `DT_VERSYM`), and falls back to the raw syscall for any
//! clock the vDSO doesn't handle, so this only ever *adds* a fast path.
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

/// The assembled vDSO function bytes (see `vdso.s` in the repo history). Two
/// `movabs r8, <imm64>` placeholders (offsets [`VVAR_PATCH`]) get the real vvar
/// VA written in at build time. `__vdso_clock_gettime` at 0, `__vdso_gettimeofday`
/// at [`GETTIMEOFDAY_OFF`].
const CODE: &[u8] = &[
    0x48, 0x83, 0xff, 0x06, 0x77, 0x56, 0x49, 0xb8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11,
    0x49, 0x83, 0x78, 0x08, 0x00, 0x74, 0x45, 0x0f, 0x31, 0x48, 0xc1, 0xe2, 0x20, 0x48, 0x09, 0xd0,
    0x49, 0x2b, 0x40, 0x18, 0x49, 0xf7, 0x60, 0x08, 0x41, 0x8a, 0x48, 0x10, 0x48, 0x0f, 0xad, 0xd0,
    0x48, 0x83, 0xff, 0x00, 0x74, 0x0c, 0x48, 0x83, 0xff, 0x05, 0x74, 0x06, 0x49, 0x03, 0x40, 0x20,
    0xeb, 0x04, 0x49, 0x03, 0x40, 0x28, 0x31, 0xd2, 0x48, 0xc7, 0xc1, 0x00, 0xca, 0x9a, 0x3b, 0x48,
    0xf7, 0xf1, 0x48, 0x89, 0x06, 0x48, 0x89, 0x56, 0x08, 0x31, 0xc0, 0xc3, 0xb8, 0xe4, 0x00, 0x00,
    0x00, 0x0f, 0x05, 0xc3, 0x48, 0x85, 0xff, 0x74, 0x50, 0x49, 0xb8, 0x88, 0x77, 0x66, 0x55, 0x44,
    0x33, 0x22, 0x11, 0x49, 0x83, 0x78, 0x08, 0x00, 0x74, 0x42, 0x0f, 0x31, 0x48, 0xc1, 0xe2, 0x20,
    0x48, 0x09, 0xd0, 0x49, 0x2b, 0x40, 0x18, 0x49, 0xf7, 0x60, 0x08, 0x41, 0x8a, 0x48, 0x10, 0x48,
    0x0f, 0xad, 0xd0, 0x49, 0x03, 0x40, 0x28, 0x31, 0xd2, 0x48, 0xc7, 0xc1, 0x00, 0xca, 0x9a, 0x3b,
    0x48, 0xf7, 0xf1, 0x48, 0x89, 0x07, 0x48, 0x89, 0xd0, 0x31, 0xd2, 0x48, 0xc7, 0xc1, 0xe8, 0x03,
    0x00, 0x00, 0x48, 0xf7, 0xf1, 0x48, 0x89, 0x47, 0x08, 0x31, 0xc0, 0xc3, 0xb8, 0x60, 0x00, 0x00,
    0x00, 0x0f, 0x05, 0xc3,
];
/// Offsets of the two `movabs r8, VVAR_VA` immediates to patch.
const VVAR_PATCH: [usize; 2] = [8, 107];
/// Offset of `__vdso_gettimeofday` within [`CODE`].
const GETTIMEOFDAY_OFF: u64 = 0x64;

// ---- ELF-image layout (all within one 4 KiB page, ET_DYN, load base = 0) ----
// The ELF header sits at offset 0. Everything below is a fixed offset into the
// one-page image; `build_image` writes each region there.
const PHDR: u64 = 0x40; // 2 × 56-byte program headers
const HASH: u64 = 0xb0; // SysV hash
const DYNSYM: u64 = 0xc8; // 3 × 24-byte symbols
const DYNSTR: u64 = 0x110;
const DYNAMIC: u64 = 0x140; // 6 × 16-byte entries
const TEXT: u64 = 0x1a0;

const DYNSTR_BYTES: &[u8] = b"\0__vdso_clock_gettime\0__vdso_gettimeofday\0";
const NAME_CG: u64 = 1; // "__vdso_clock_gettime" offset in dynstr
const NAME_GTOD: u64 = 22; // "__vdso_gettimeofday" offset in dynstr

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

    // SysV hash: 1 bucket, 3-entry chain (sym0 unused, sym1→sym2→end).
    w32(&mut b, HASH, 1); // nbucket
    w32(&mut b, HASH + 4, 3); // nchain (= symbol count)
    w32(&mut b, HASH + 8, 1); // bucket[0] = first symbol index
    w32(&mut b, HASH + 12, 0); // chain[0]
    w32(&mut b, HASH + 16, 2); // chain[1] -> sym 2
    w32(&mut b, HASH + 20, 0); // chain[2] -> end

    // Dynamic symbols. sym[0] is the reserved null entry (already zero).
    // st_info = (STB_GLOBAL<<4)|STT_FUNC = 0x12; st_shndx = 1 (defined).
    let cg = DYNSYM + 24;
    w32(&mut b, cg, NAME_CG as u32);
    b[cg as usize + 4] = 0x12;
    w16(&mut b, cg + 6, 1); // st_shndx
    w64(&mut b, cg + 8, TEXT); // st_value = clock_gettime code
    w64(&mut b, cg + 16, GETTIMEOFDAY_OFF); // st_size
    let gt = DYNSYM + 48;
    w32(&mut b, gt, NAME_GTOD as u32);
    b[gt as usize + 4] = 0x12;
    w16(&mut b, gt + 6, 1);
    w64(&mut b, gt + 8, TEXT + GETTIMEOFDAY_OFF);
    w64(&mut b, gt + 16, CODE.len() as u64 - GETTIMEOFDAY_OFF);

    // Dynamic string table.
    b[DYNSTR as usize..DYNSTR as usize + DYNSTR_BYTES.len()].copy_from_slice(DYNSTR_BYTES);

    // Dynamic section: DT_HASH, DT_STRTAB, DT_SYMTAB, DT_STRSZ, DT_SYMENT, DT_NULL.
    let dyn_entries: [(u64, u64); 6] = [
        (4, HASH),                      // DT_HASH
        (5, DYNSTR),                    // DT_STRTAB
        (6, DYNSYM),                    // DT_SYMTAB
        (10, DYNSTR_BYTES.len() as u64), // DT_STRSZ
        (11, 24),                       // DT_SYMENT
        (0, 0),                         // DT_NULL
    ];
    for (i, (tag, val)) in dyn_entries.iter().enumerate() {
        let o = DYNAMIC + (i as u64) * 16;
        w64(&mut b, o, *tag);
        w64(&mut b, o + 8, *val);
    }

    // Code, with the vvar VA patched into both `movabs` immediates.
    b[TEXT as usize..TEXT as usize + CODE.len()].copy_from_slice(CODE);
    for &p in &VVAR_PATCH {
        w64(&mut b, TEXT + p as u64, vvar_va);
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse the built image the way musl's `__vdsosym` does and confirm both
    /// symbols resolve to their code offsets — a regression guard on the ELF.
    #[test]
    fn image_exports_the_clock_symbols() {
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

        // Walk the hash chain from bucket 0 and match names.
        let nchain = u32::from_le_bytes(img[hash as usize + 4..hash as usize + 8].try_into().unwrap());
        let sym_name = |idx: u64| -> String {
            let s = (symtab + idx * 24) as usize;
            let n = u32::from_le_bytes(img[s..s + 4].try_into().unwrap()) as u64;
            let start = (strtab + n) as usize;
            let end = img[start..].iter().position(|&c| c == 0).unwrap() + start;
            String::from_utf8_lossy(&img[start..end]).into_owned()
        };
        let val = |idx: u64| u64::from_le_bytes(img[(symtab + idx * 24 + 8) as usize..(symtab + idx * 24 + 16) as usize].try_into().unwrap());
        let mut found_cg = None;
        let mut found_gt = None;
        for idx in 1..u64::from(nchain) {
            match sym_name(idx).as_str() {
                "__vdso_clock_gettime" => found_cg = Some(val(idx)),
                "__vdso_gettimeofday" => found_gt = Some(val(idx)),
                _ => {}
            }
        }
        assert_eq!(found_cg, Some(TEXT));
        assert_eq!(found_gt, Some(TEXT + GETTIMEOFDAY_OFF));

        // The vvar VA was patched into both movabs immediates.
        for &p in &VVAR_PATCH {
            let got = u64::from_le_bytes(img[(TEXT + p as u64) as usize..(TEXT + p as u64) as usize + 8].try_into().unwrap());
            assert_eq!(got, vvar);
        }
    }
}
