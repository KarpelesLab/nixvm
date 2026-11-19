//! The 32-bit vDSO a compatibility-mode (ELF32, i386) process is given.
//!
//! Linux maps one into every ia32 task (`arch/x86/entry/vdso/vdso32`) and
//! names it in the auxiliary vector twice: `AT_SYSINFO_EHDR` (the ELF image)
//! and `AT_SYSINFO` (`__kernel_vsyscall`, the system-call entry glibc and musl
//! call through `%gs:0x10`). This one has what an i386 libc looks for that
//! isn't a clock:
//!
//! - `__kernel_vsyscall`, the fast system call: `push %ecx; push %edx; push
//!   %ebp`, then `syscall` with the second argument moved to `%ebp` (an AMD
//!   CPU), or `sysenter` with the stack pointer in `%ebp` (an Intel one), then
//!   `int $0x80` — the landing pad a fast system call returns just past, so
//!   that a restart (`ip -= 2`) lands on an `int $0x80` — and `pop`s and
//!   `ret`. Linux patches the instruction in for the CPU (`ALTERNATIVE_2`);
//!   [`build_image`] takes which.
//! - `__kernel_sigreturn` and `__kernel_rt_sigreturn`, where a signal handler
//!   installed without `SA_RESTORER` returns to.
//! - The clock functions, under `LINUX_2.6` as Linux's `vdso32.lds.S` exports
//!   them: `__vdso_clock_gettime`, `__vdso_clock_gettime64`,
//!   `__vdso_gettimeofday`, `__vdso_time`, `__vdso_clock_getres` and
//!   `__vdso_getcpu` (assembled from `vdso32_x86.s`). They read the 64-bit
//!   vDSO's vvar page — the same frame, mapped one page below the image (Linux's
//!   `[vvar]` sits there too) — and what the calibration can't serve goes to the
//!   system call through `__kernel_vsyscall`, as Linux's do.

/// How `__kernel_vsyscall` enters the kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FastEntry {
    /// `int $0x80` only (no fast instruction).
    Int80,
    /// `syscall`, as on an AMD CPU (`X86_FEATURE_SYSCALL32`).
    Syscall,
    /// `sysenter`, as on an Intel CPU (`X86_FEATURE_SYSENTER32`).
    Sysenter,
}

/// Size of the image: one page.
pub const PAGE: usize = 4096;

// ---- layout (ELF32, ET_DYN, load base 0) -----------------------------------
const EHDR_LEN: u32 = 52;
const PHDR: u32 = 0x34; // 2 × 32-byte program headers
const HASH: u32 = 0x80; // (3 + 1 + SYMS) × 4
const DYNSYM: u32 = 0xc0; // (1 + SYMS) × 16
const VERSYM: u32 = 0x160;
const VERDEF: u32 = 0x180; // 3 × (20 + 8)
const DYNSTR: u32 = 0x1e0;
const DYNAMIC: u32 = 0x300; // DYN_ENTRIES × 8
const DYN_ENTRIES: u32 = 9;
const TEXT: u32 = 0x400;
/// Where the clock code ([`CLOCK_CODE`]) starts: `vdso32_x86.s` reaches
/// `__kernel_vsyscall` and the vvar page relative to it.
const CLOCK: u32 = TEXT + 0x40;
const _: () = assert!(VERDEF + 3 * 28 <= DYNSTR, "verdef overlaps dynstr");
const _: () = assert!(
    DYNSYM + (SYMS.len() as u32 + 1) * 16 <= VERSYM,
    "dynsym overlaps versym"
);
const _: () = assert!(
    CLOCK as usize + CLOCK_CODE.len() <= PAGE,
    "the clock code runs off the page"
);

/// `__kernel_vsyscall`, within the image.
pub const VSYSCALL: u64 = TEXT as u64;
/// The `int80_landing_pad`, just past `__kernel_vsyscall`'s `int $0x80`.
pub const INT80_LANDING_PAD: u64 = TEXT as u64 + 9;
/// `__kernel_sigreturn`.
pub const SIGRETURN: u64 = TEXT as u64 + 0x10;
/// `__kernel_rt_sigreturn`.
pub const RT_SIGRETURN: u64 = TEXT as u64 + 0x20;

/// The code, at [`TEXT`]: the vsyscall (its fast instruction at offset 3,
/// four bytes, patched by [`build_image`]), then the two sigreturns.
const CODE: [u8; 0x28] = [
    // __kernel_vsyscall
    0x51, // push %ecx
    0x52, // push %edx
    0x55, // push %ebp
    0x90, 0x90, 0x90, 0x90, // the fast entry (or nops)
    0xcd, 0x80, // int $0x80
    // int80_landing_pad
    0x5d, // pop %ebp
    0x5a, // pop %edx
    0x59, // pop %ecx
    0xc3, // ret
    0x90, 0x90, 0x90, // pad
    // __kernel_sigreturn
    0x58, // pop %eax
    0xb8, 0x77, 0x00, 0x00, 0x00, // mov $119 (__NR_sigreturn), %eax
    0xcd, 0x80, // int $0x80
    0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, 0x90, // pad
    // __kernel_rt_sigreturn
    0xb8, 0xad, 0x00, 0x00, 0x00, // mov $173 (__NR_rt_sigreturn), %eax
    0xcd, 0x80, // int $0x80
    0x90,
];

/// The clock functions (assembled from `vdso32_x86.s`), at [`CLOCK`].
const CLOCK_CODE: &[u8] = &[
    0x55, 0x53, 0x56, 0x57, 0xe8, 0x00, 0x00, 0x00, 0x00, 0x5d, 0x81, 0xc5, 0xb7, 0xeb, 0xff, 0xff,
    0x8b, 0x45, 0x08, 0x0b, 0x45, 0x0c, 0x74, 0x62, 0x51, 0x0f, 0x31, 0x2b, 0x45, 0x18, 0x1b, 0x55,
    0x1c, 0x89, 0xd7, 0x89, 0xc6, 0xf7, 0x65, 0x08, 0x89, 0xc3, 0x89, 0xd1, 0x89, 0xf0, 0xf7, 0x65,
    0x0c, 0x01, 0xc1, 0x83, 0xd2, 0x00, 0x89, 0xd6, 0x6a, 0x00, 0x89, 0xf8, 0xf7, 0x65, 0x08, 0x01,
    0xc1, 0x11, 0xd6, 0x83, 0x14, 0x24, 0x00, 0x89, 0xf8, 0xf7, 0x65, 0x0c, 0x01, 0xc6, 0x11, 0x14,
    0x24, 0x5f, 0x8a, 0x45, 0x10, 0xa8, 0x20, 0x74, 0x06, 0x89, 0xcb, 0x89, 0xf1, 0x89, 0xfe, 0x89,
    0xca, 0x89, 0xc1, 0x89, 0xd8, 0x0f, 0xad, 0xd0, 0x0f, 0xad, 0xf2, 0x59, 0x03, 0x44, 0x0d, 0x00,
    0x13, 0x54, 0x0d, 0x04, 0x5f, 0x5e, 0x5b, 0x5d, 0xf8, 0xc3, 0x5f, 0x5e, 0x5b, 0x5d, 0xf9, 0xc3,
    0x53, 0xbb, 0x00, 0xca, 0x9a, 0x3b, 0x89, 0xc1, 0x89, 0xd0, 0x31, 0xd2, 0xf7, 0xf3, 0x91, 0xf7,
    0xf3, 0x87, 0xd1, 0x5b, 0xc3, 0x83, 0xf8, 0x06, 0x77, 0x1f, 0x83, 0xf8, 0x02, 0x74, 0x1a, 0x83,
    0xf8, 0x03, 0x74, 0x15, 0xb9, 0x20, 0x00, 0x00, 0x00, 0x85, 0xc0, 0x74, 0x05, 0x83, 0xf8, 0x05,
    0x75, 0x05, 0xb9, 0x28, 0x00, 0x00, 0x00, 0xf8, 0xc3, 0xf9, 0xc3, 0x8b, 0x44, 0x24, 0x04, 0xe8,
    0xd1, 0xff, 0xff, 0xff, 0x72, 0x18, 0xe8, 0x35, 0xff, 0xff, 0xff, 0x72, 0x11, 0xe8, 0xae, 0xff,
    0xff, 0xff, 0x8b, 0x54, 0x24, 0x08, 0x89, 0x02, 0x89, 0x4a, 0x04, 0x31, 0xc0, 0xc3, 0xb8, 0x09,
    0x01, 0x00, 0x00, 0xe9, 0xf8, 0x00, 0x00, 0x00, 0x8b, 0x44, 0x24, 0x04, 0xe8, 0xa4, 0xff, 0xff,
    0xff, 0x72, 0x24, 0xe8, 0x08, 0xff, 0xff, 0xff, 0x72, 0x1d, 0xe8, 0x81, 0xff, 0xff, 0xff, 0x56,
    0x8b, 0x74, 0x24, 0x0c, 0x89, 0x06, 0x89, 0x56, 0x04, 0x89, 0x4e, 0x08, 0xc7, 0x46, 0x0c, 0x00,
    0x00, 0x00, 0x00, 0x5e, 0x31, 0xc0, 0xc3, 0xb8, 0x93, 0x01, 0x00, 0x00, 0xe9, 0xbf, 0x00, 0x00,
    0x00, 0x8b, 0x44, 0x24, 0x08, 0x85, 0xc0, 0x74, 0x0d, 0xc7, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc7,
    0x40, 0x04, 0x00, 0x00, 0x00, 0x00, 0x83, 0x7c, 0x24, 0x04, 0x00, 0x74, 0x29, 0xb9, 0x28, 0x00,
    0x00, 0x00, 0xe8, 0xb9, 0xfe, 0xff, 0xff, 0x72, 0x20, 0xe8, 0x32, 0xff, 0xff, 0xff, 0x8b, 0x54,
    0x24, 0x04, 0x89, 0x02, 0x89, 0xc8, 0x31, 0xd2, 0xb9, 0xe8, 0x03, 0x00, 0x00, 0xf7, 0xf1, 0x8b,
    0x54, 0x24, 0x04, 0x89, 0x42, 0x04, 0x31, 0xc0, 0xc3, 0xb8, 0x4e, 0x00, 0x00, 0x00, 0xeb, 0x70,
    0xb9, 0x28, 0x00, 0x00, 0x00, 0xe8, 0x86, 0xfe, 0xff, 0xff, 0x72, 0x10, 0xe8, 0xff, 0xfe, 0xff,
    0xff, 0x8b, 0x4c, 0x24, 0x04, 0x85, 0xc9, 0x74, 0x02, 0x89, 0x01, 0xc3, 0x53, 0x8b, 0x5c, 0x24,
    0x08, 0xb8, 0x0d, 0x00, 0x00, 0x00, 0xe8, 0x25, 0xfe, 0xff, 0xff, 0x5b, 0xc3, 0x8b, 0x44, 0x24,
    0x04, 0xe8, 0xef, 0xfe, 0xff, 0xff, 0x72, 0x18, 0x8b, 0x44, 0x24, 0x08, 0x85, 0xc0, 0x74, 0x0d,
    0xc7, 0x00, 0x00, 0x00, 0x00, 0x00, 0xc7, 0x40, 0x04, 0x01, 0x00, 0x00, 0x00, 0x31, 0xc0, 0xc3,
    0xb8, 0x0a, 0x01, 0x00, 0x00, 0xeb, 0x19, 0x53, 0x8b, 0x5c, 0x24, 0x08, 0x8b, 0x4c, 0x24, 0x0c,
    0x8b, 0x54, 0x24, 0x10, 0xb8, 0x3e, 0x01, 0x00, 0x00, 0xe8, 0xe2, 0xfd, 0xff, 0xff, 0x5b, 0xc3,
    0x53, 0x8b, 0x5c, 0x24, 0x08, 0x8b, 0x4c, 0x24, 0x0c, 0xe8, 0xd2, 0xfd, 0xff, 0xff, 0x5b, 0xc3,
];

/// Where the vvar page is, below the image's base.
pub const VVAR_BELOW: u64 = 4096;

/// The exported symbols: name, offset, size, and version index (2 is
/// `LINUX_2.6`, 3 `LINUX_2.5`).
const SYMS: [(&str, u64, u64, u16); 9] = [
    ("__kernel_vsyscall", VSYSCALL, 13, 3),
    ("__kernel_sigreturn", SIGRETURN, 8, 3),
    ("__kernel_rt_sigreturn", RT_SIGRETURN, 7, 3),
    ("__vdso_clock_gettime", CLOCK as u64 + 0xbb, 0x2d, 2),
    ("__vdso_clock_gettime64", CLOCK as u64 + 0xe8, 0x39, 2),
    ("__vdso_gettimeofday", CLOCK as u64 + 0x121, 0x4f, 2),
    ("__vdso_time", CLOCK as u64 + 0x170, 0x2d, 2),
    ("__vdso_clock_getres", CLOCK as u64 + 0x19d, 0x2a, 2),
    ("__vdso_getcpu", CLOCK as u64 + 0x1c7, 0x19, 2),
];

const SONAME: &str = "linux-gate.so.1";
/// The version definitions after the base: index 2, then 3.
const VERSIONS: [&str; 2] = ["LINUX_2.6", "LINUX_2.5"];

fn w32(b: &mut [u8], off: u32, v: u32) {
    b[off as usize..off as usize + 4].copy_from_slice(&v.to_le_bytes());
}
fn w16(b: &mut [u8], off: u32, v: u16) {
    b[off as usize..off as usize + 2].copy_from_slice(&v.to_le_bytes());
}

/// Build the one-page image, `__kernel_vsyscall` entering by `entry`.
#[must_use]
pub fn build_image(entry: FastEntry) -> [u8; PAGE] {
    let mut b = [0u8; PAGE];
    b[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
    b[4] = 1; // ELFCLASS32
    b[5] = 1; // ELFDATA2LSB
    b[6] = 1; // EV_CURRENT
    w16(&mut b, 16, 3); // ET_DYN
    w16(&mut b, 18, 3); // EM_386
    w32(&mut b, 20, 1); // e_version
    w32(&mut b, 24, VSYSCALL as u32); // e_entry
    w32(&mut b, 28, PHDR); // e_phoff
    w16(&mut b, 40, EHDR_LEN as u16); // e_ehsize
    w16(&mut b, 42, 32); // e_phentsize
    w16(&mut b, 44, 2); // e_phnum
    w16(&mut b, 46, 40); // e_shentsize

    // PT_LOAD (the page, R+X), then PT_DYNAMIC.
    let ph = |b: &mut [u8], i: u32, ty: u32, off: u32, size: u32, flags: u32, align: u32| {
        let p = PHDR + i * 32;
        w32(b, p, ty);
        w32(b, p + 4, off); // p_offset
        w32(b, p + 8, off); // p_vaddr
        w32(b, p + 12, off); // p_paddr
        w32(b, p + 16, size); // p_filesz
        w32(b, p + 20, size); // p_memsz
        w32(b, p + 24, flags);
        w32(b, p + 28, align);
    };
    ph(&mut b, 0, 1, 0, PAGE as u32, 5, PAGE as u32);
    ph(&mut b, 1, 2, DYNAMIC, DYN_ENTRIES * 8, 4, 4);

    // SysV hash: one bucket chaining every symbol.
    let nsym = SYMS.len() as u32;
    let nchain = nsym + 1;
    w32(&mut b, HASH, 1);
    w32(&mut b, HASH + 4, nchain);
    w32(&mut b, HASH + 8, 1);
    for i in 1..=nsym {
        w32(&mut b, HASH + 12 + i * 4, if i == nsym { 0 } else { i + 1 });
    }

    // Strings: the names, the soname, the version.
    let mut strs = vec![0u8];
    let mut add = |s: &str| {
        let off = strs.len() as u32;
        strs.extend_from_slice(s.as_bytes());
        strs.push(0);
        off
    };
    let names: Vec<u32> = SYMS.iter().map(|(n, ..)| add(n)).collect();
    let soname = add(SONAME);
    let versions = VERSIONS.map(&mut add);
    let strsz = strs.len() as u32;
    b[DYNSTR as usize..DYNSTR as usize + strs.len()].copy_from_slice(&strs);

    // Symbols (Elf32_Sym: name, value, size, info, other, shndx).
    for (i, (&name, &(_, value, size, ver))) in names.iter().zip(SYMS.iter()).enumerate() {
        let s = DYNSYM + (i as u32 + 1) * 16;
        w32(&mut b, s, name);
        w32(&mut b, s + 4, value as u32);
        w32(&mut b, s + 8, size as u32);
        b[s as usize + 12] = 0x12; // STB_GLOBAL, STT_FUNC
        w16(&mut b, s + 14, 1);
        w16(&mut b, VERSYM + (i as u32 + 1) * 2, ver);
    }
    let defs = [
        (1u16, SONAME, soname),
        (0, VERSIONS[0], versions[0]),
        (0, VERSIONS[1], versions[1]),
    ];
    for (k, &(flags, name, off)) in defs.iter().enumerate() {
        let d = VERDEF + k as u32 * 28;
        w16(&mut b, d, 1);
        w16(&mut b, d + 2, flags);
        w16(&mut b, d + 4, k as u16 + 1);
        w16(&mut b, d + 6, 1);
        w32(&mut b, d + 8, super::vdso::elf_hash(name));
        w32(&mut b, d + 12, 20);
        w32(&mut b, d + 16, if k + 1 < defs.len() { 28 } else { 0 });
        w32(&mut b, d + 20, off);
    }

    let dyns: [(u32, u32); DYN_ENTRIES as usize] = [
        (4, HASH),
        (5, DYNSTR),
        (6, DYNSYM),
        (10, strsz),
        (11, 16),
        (0x6fff_fff0, VERSYM),
        (0x6fff_fffc, VERDEF),
        (0x6fff_fffd, 3),
        (0, 0),
    ];
    for (i, (tag, val)) in dyns.iter().enumerate() {
        w32(&mut b, DYNAMIC + i as u32 * 8, *tag);
        w32(&mut b, DYNAMIC + i as u32 * 8 + 4, *val);
    }
    debug_assert!(DYNSTR + strsz <= DYNAMIC);
    debug_assert!(HASH + 12 + nchain * 4 <= DYNSYM);
    debug_assert!(DYNSYM + nchain * 16 <= VERSYM);
    debug_assert!(VERSYM + nchain * 2 <= VERDEF);

    let t = TEXT as usize;
    b[t..t + CODE.len()].copy_from_slice(&CODE);
    let fast: [u8; 4] = match entry {
        FastEntry::Int80 => [0x90; 4],
        FastEntry::Syscall => [0x89, 0xcd, 0x0f, 0x05], // mov %ecx,%ebp; syscall
        FastEntry::Sysenter => [0x89, 0xe5, 0x0f, 0x34], // mov %esp,%ebp; sysenter
    };
    b[t + 3..t + 7].copy_from_slice(&fast);
    let c = CLOCK as usize;
    b[c..c + CLOCK_CODE.len()].copy_from_slice(CLOCK_CODE);
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_landing_pad_follows_the_int80() {
        let img = build_image(FastEntry::Syscall);
        let pad = INT80_LANDING_PAD as usize;
        assert_eq!(&img[pad - 2..pad], &[0xcd, 0x80]);
        assert_eq!(&img[pad - 4..pad - 2], &[0x0f, 0x05]);
        assert_eq!(img[SIGRETURN as usize], 0x58);
        assert_eq!(
            &img[RT_SIGRETURN as usize..RT_SIGRETURN as usize + 5],
            &[0xb8, 0xad, 0, 0, 0]
        );
    }

    #[test]
    fn symbols_resolve_through_the_hash_table() {
        let img = build_image(FastEntry::Sysenter);
        let r32 = |o: u32| u32::from_le_bytes(img[o as usize..o as usize + 4].try_into().unwrap());
        let nchain = r32(HASH + 4);
        let mut found = Vec::new();
        let mut i = r32(HASH + 8);
        while i != 0 {
            let s = DYNSYM + i * 16;
            let name_off = (DYNSTR + r32(s)) as usize;
            let end = img[name_off..].iter().position(|&c| c == 0).unwrap();
            let name = std::str::from_utf8(&img[name_off..name_off + end]).unwrap();
            found.push((name.to_string(), u64::from(r32(s + 4))));
            i = r32(HASH + 12 + i * 4);
            assert!(i < nchain);
        }
        assert_eq!(found[0], ("__kernel_vsyscall".to_string(), VSYSCALL));
        assert_eq!(found.len(), SYMS.len());
        // Each clock function's first byte is where the assembly put it.
        for (name, off) in &found[3..] {
            assert!(off >= &u64::from(CLOCK), "{name}");
            assert!(*off < u64::from(CLOCK) + CLOCK_CODE.len() as u64, "{name}");
        }
    }
}
