//! Turning an ELF file into a ready-to-run guest process image.
//!
//! Responsibilities:
//! * parse ELF64 headers and program headers (hand-rolled, no external dep),
//! * map `PT_LOAD` segments into [`GuestMemory`] with correct protections,
//! * build the initial stack: `argc`, `argv`, `envp`, and the auxiliary vector
//!   (`AT_PHDR`, `AT_ENTRY`, `AT_RANDOM`, `AT_EXECFN`, …),
//! * report the entry PC, initial SP, and program break.
//!
//! Static executables work now, including statically-linked position-independent
//! executables (`ET_DYN`, the default `musl-gcc -static-pie` output): the loader
//! picks a load bias, maps segments at `p_vaddr + bias`, and applies the
//! `R_*_RELATIVE` relocations a static PIE carries in its `.rela.dyn`. The
//! dynamic linker (`PT_INTERP` → `ld-musl`) and a vDSO arrive in ROADMAP Phase 5.

use crate::vcpu::mem::{PAGE_SIZE, Prot};
use crate::vcpu::{GuestMemory, MemError};

// ---- ELF constants -------------------------------------------------------

const ELF_MAGIC: [u8; 4] = [0x7f, b'E', b'L', b'F'];
const ELFCLASS64: u8 = 2;
const ELFDATA2LSB: u8 = 1;
const EM_X86_64: u16 = 62;
const EM_AARCH64: u16 = 183;

/// `e_type` for executables that require no relocation (fixed load address).
const ET_EXEC: u16 = 2;
/// `e_type` shared by shared objects and position-independent executables.
/// For a *static* PIE (no `PT_INTERP`), this is what modern musl toolchains
/// emit by default; the loader gives it a load bias and applies its
/// `R_*_RELATIVE` relocations itself, in place of a dynamic linker.
const ET_DYN: u16 = 3;

const PT_LOAD: u32 = 1;
const PT_DYNAMIC: u32 = 2;
const PF_X: u32 = 1;
const PF_W: u32 = 2;
const PF_R: u32 = 4;

const EHDR_LEN: usize = 64;
const PHDR_LEN: usize = 56;
/// Size of one `Elf64_Dyn` entry: `{ d_tag: u64, d_val/d_ptr: u64 }`.
const DYN_LEN: usize = 16;
/// Size of one `Elf64_Rela` entry: `{ r_offset, r_info, r_addend }`, all `u64`.
const RELA_LEN: usize = 24;
/// Size of one `Elf64_Rel` entry: `{ r_offset, r_info }`.
const REL_LEN: usize = 16;

// `PT_DYNAMIC` tags the loader cares about (ELF64 `Elf64_Dyn.d_tag`).
const DT_NULL: u64 = 0;
const DT_REL: u64 = 17;
const DT_RELSZ: u64 = 18;
const DT_RELENT: u64 = 19;
const DT_RELA: u64 = 7;
const DT_RELASZ: u64 = 8;
const DT_RELAENT: u64 = 9;
/// Count of `R_*_RELATIVE` entries at the front of `.rela.dyn`. A pure hint
/// (the loader classifies every entry by its own `r_type` instead), but
/// still recognised here so the table walk doesn't misparse it as unknown.
const DT_RELACOUNT: u64 = 0x6fff_fff9;

// Relocation types a static PIE can legally contain: `RELATIVE` (base +
// addend, always resolvable by the loader) and `IRELATIVE` (an ifunc
// resolver the loader would have to *execute* to resolve, which a
// host-side loader can't do before the guest's first instruction runs).
const R_X86_64_RELATIVE: u32 = 8;
const R_X86_64_IRELATIVE: u32 = 37;
const R_AARCH64_RELATIVE: u32 = 1027;
const R_AARCH64_IRELATIVE: u32 = 1032;

// Auxiliary-vector tags.
const AT_NULL: u64 = 0;
const AT_PHDR: u64 = 3;
const AT_PHENT: u64 = 4;
const AT_PHNUM: u64 = 5;
const AT_PAGESZ: u64 = 6;
const AT_BASE: u64 = 7;
const AT_FLAGS: u64 = 8;
const AT_ENTRY: u64 = 9;
const AT_UID: u64 = 11;
const AT_EUID: u64 = 12;
const AT_GID: u64 = 13;
const AT_EGID: u64 = 14;
const AT_CLKTCK: u64 = 17;
const AT_SECURE: u64 = 23;
const AT_RANDOM: u64 = 25;
const AT_EXECFN: u64 = 31;

/// Guest stack size reserved at the top of the address space.
const STACK_SIZE: u64 = 256 * 1024;

// ---- public API ----------------------------------------------------------

/// Where to start executing and how the stack was laid out.
#[derive(Debug, Clone)]
pub struct LoadedImage {
    /// Entry PC (the interpreter's entry for dynamic executables).
    pub entry: u64,
    /// Initial stack pointer, pointing at `argc`.
    pub stack_pointer: u64,
    /// Program break (end of the highest `PT_LOAD`), where `brk` starts.
    pub program_break: u64,
    /// Lowest address of the initial stack region (top of the address space
    /// minus the reserved stack). The mmap arena lives below this.
    pub stack_bottom: u64,
}

/// What the guest should be started with.
#[derive(Debug, Clone)]
pub struct ProcessSpec {
    pub argv: Vec<String>,
    pub envp: Vec<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum LoadError {
    NotElf,
    UnsupportedArch,
    Truncated,
    Malformed(&'static str),
    Mem(MemError),
    Unimplemented(&'static str),
}

impl core::fmt::Display for LoadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotElf => write!(f, "not an ELF64 file"),
            Self::UnsupportedArch => write!(f, "unsupported ELF machine type"),
            Self::Truncated => write!(f, "ELF file is truncated"),
            Self::Malformed(m) => write!(f, "malformed ELF: {m}"),
            Self::Mem(e) => write!(f, "guest memory error: {e:?}"),
            Self::Unimplemented(w) => write!(f, "loader: {w} not implemented"),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<MemError> for LoadError {
    fn from(e: MemError) -> Self {
        Self::Mem(e)
    }
}

/// Load a statically-linked ELF64 executable into `mem`.
///
/// Handles both fixed-address executables (`ET_EXEC`) and static
/// position-independent executables (`ET_DYN` with no `PT_INTERP`, i.e. a
/// static PIE): the latter get a nonzero load bias applied to every segment,
/// the entry point and the phdr address, followed by the `R_*_RELATIVE`
/// fixups recorded in `PT_DYNAMIC`.
pub fn load_static(
    mem: &mut GuestMemory,
    elf: &[u8],
    spec: &ProcessSpec,
) -> Result<LoadedImage, LoadError> {
    let ehdr = Ehdr::parse(elf)?;

    // A static PIE has no preferred load address (its PT_LOAD `p_vaddr`s are
    // normally already 0-based), so any page-aligned bias works as long as
    // every reference to a link-time address gets the same bias added and the
    // result stays inside the guest region. `GuestMemory` is one flat,
    // bounds-checked `[base, base + size)` window rather than a full 64-bit
    // address space, so we anchor the bias at the region's own base: that is
    // guaranteed page-aligned (`GuestMemory::new` asserts it) and keeps
    // biased addresses in the same place an `ET_EXEC` image with `p_vaddr`
    // starting at 0 would land, regardless of how large the caller's region
    // is. `ET_EXEC` images keep their existing, unbiased behavior.
    let bias: u64 = if ehdr.e_type == ET_DYN {
        mem.base()
    } else {
        0
    };

    // Collect every program header up front: PT_LOAD segments get mapped,
    // PT_DYNAMIC (if present) drives the relocation pass below, and the full
    // list is needed to translate the dynamic table's link-time vaddrs back
    // to file offsets.
    let mut phdrs = Vec::with_capacity(ehdr.phnum as usize);
    for i in 0..ehdr.phnum {
        let off = ehdr.phoff as usize + i as usize * PHDR_LEN;
        phdrs.push(Phdr::parse(elf, off)?);
    }

    let mut program_break = 0u64;
    let mut phdr_vaddr: Option<u64> = None;

    for ph in &phdrs {
        if ph.p_type != PT_LOAD || ph.memsz == 0 {
            continue;
        }
        let vaddr = ph
            .vaddr
            .checked_add(bias)
            .ok_or(LoadError::Malformed("segment vaddr + bias overflow"))?;

        // Map the segment's pages with its final protection, then load the file
        // bytes via write_init (which bypasses protection). The tail
        // [filesz, memsz) is .bss and stays zeroed.
        let prot = seg_prot(ph.flags);
        mem.map(vaddr, ph.memsz, prot)?;
        if ph.filesz > 0 {
            let file_end = ph
                .offset
                .checked_add(ph.filesz)
                .ok_or(LoadError::Malformed("segment offset overflow"))?;
            let bytes = elf
                .get(ph.offset as usize..file_end as usize)
                .ok_or(LoadError::Truncated)?;
            mem.write_init(vaddr, bytes)?;
        }

        program_break = program_break.max(round_up(vaddr + ph.memsz, PAGE_SIZE));

        // If this segment contains the program headers, record their (biased)
        // vaddr for AT_PHDR.
        if ph.offset <= ehdr.phoff && ehdr.phoff < ph.offset + ph.filesz {
            phdr_vaddr = Some(vaddr + (ehdr.phoff - ph.offset));
        }
    }

    if program_break == 0 {
        return Err(LoadError::Malformed("no loadable segments"));
    }

    if ehdr.e_type == ET_DYN {
        apply_relative_relocations(mem, elf, &phdrs, bias, ehdr.machine)?;
    }

    let entry = ehdr
        .entry
        .checked_add(bias)
        .ok_or(LoadError::Malformed("entry + bias overflow"))?;

    let stack_pointer = build_stack(mem, spec, &ehdr, entry, phdr_vaddr)?;

    Ok(LoadedImage {
        entry,
        stack_pointer,
        program_break,
        stack_bottom: (mem.base() + mem.size()) - STACK_SIZE,
    })
}

// ---- static-PIE relocations -----------------------------------------------

/// Apply `R_*_RELATIVE` fixups from `PT_DYNAMIC`'s `.rela.dyn`/`.rel.dyn` to
/// an already-mapped, already-populated image. `phdrs` must be the segment's
/// *unbiased* (link-time) program headers; `bias` is added to every link-time
/// address (segment vaddrs, `r_offset`, and the RELATIVE addend/base).
///
/// A static PIE only ever needs `RELATIVE` (and, in principle, `IRELATIVE`)
/// relocations — there is no symbol table to resolve against. `IRELATIVE`
/// entries require *executing* an ifunc resolver, which a host-side loader
/// cannot do before the guest's first instruction runs, so those are skipped;
/// any other relocation type is likewise skipped (it should not occur in a
/// static PIE and there is nothing sound the loader can do with it anyway).
fn apply_relative_relocations(
    mem: &mut GuestMemory,
    elf: &[u8],
    phdrs: &[Phdr],
    bias: u64,
    machine: u16,
) -> Result<(), LoadError> {
    let Some(dyn_ph) = phdrs.iter().find(|p| p.p_type == PT_DYNAMIC) else {
        // No PT_DYNAMIC: nothing to relocate.
        return Ok(());
    };

    let relative_type = match machine {
        EM_AARCH64 => R_AARCH64_RELATIVE,
        EM_X86_64 => R_X86_64_RELATIVE,
        _ => return Err(LoadError::UnsupportedArch),
    };
    let irelative_type = match machine {
        EM_AARCH64 => R_AARCH64_IRELATIVE,
        EM_X86_64 => R_X86_64_IRELATIVE,
        _ => return Err(LoadError::UnsupportedArch),
    };

    let mut dt_rela: Option<u64> = None;
    let mut dt_relasz: Option<u64> = None;
    let mut dt_relaent: Option<u64> = None;
    let mut dt_rel: Option<u64> = None;
    let mut dt_relsz: Option<u64> = None;
    let mut dt_relent: Option<u64> = None;
    // A pure hint (count of RELATIVE entries at the front of `.rela.dyn`).
    // Parsed for completeness/recognition, but not relied on: every entry's
    // own r_type is checked below regardless of this count.
    let mut dt_relacount: Option<u64> = None;

    let dyn_start = dyn_ph.offset as usize;
    let dyn_end = dyn_start
        .checked_add(dyn_ph.filesz as usize)
        .ok_or(LoadError::Malformed("PT_DYNAMIC size overflow"))?;
    let mut off = dyn_start;
    while off + DYN_LEN <= dyn_end {
        let tag = read_u64(elf, off)?;
        let val = read_u64(elf, off + 8)?;
        match tag {
            DT_NULL => break,
            DT_RELA => dt_rela = Some(val),
            DT_RELASZ => dt_relasz = Some(val),
            DT_RELAENT => dt_relaent = Some(val),
            DT_REL => dt_rel = Some(val),
            DT_RELSZ => dt_relsz = Some(val),
            DT_RELENT => dt_relent = Some(val),
            DT_RELACOUNT => dt_relacount = Some(val),
            // Every other tag (DT_SYMTAB, DT_STRTAB, DT_FLAGS, ...) is
            // irrelevant to a static loader that only ever applies RELATIVE
            // fixups.
            _ => {}
        }
        off += DYN_LEN;
    }
    let _ = dt_relacount; // recognized, not required (see comment above)

    if let (Some(rela_vaddr), Some(relasz)) = (dt_rela, dt_relasz) {
        let entsz = dt_relaent.unwrap_or(RELA_LEN as u64);
        if entsz as usize != RELA_LEN {
            return Err(LoadError::Malformed("unexpected DT_RELAENT"));
        }
        let file_off = vaddr_to_file_offset(phdrs, rela_vaddr)
            .ok_or(LoadError::Malformed("DT_RELA not backed by a PT_LOAD segment"))?;
        let count = relasz as usize / RELA_LEN;
        for i in 0..count {
            let e = (file_off as usize)
                .checked_add(i * RELA_LEN)
                .ok_or(LoadError::Malformed("DT_RELA entry overflow"))?;
            let r_offset = read_u64(elf, e)?;
            let r_info = read_u64(elf, e + 8)?;
            let r_addend = read_u64(elf, e + 16)?;
            let r_type = (r_info & 0xffff_ffff) as u32;

            if r_type == relative_type {
                let target = r_offset
                    .checked_add(bias)
                    .ok_or(LoadError::Malformed("relocation offset overflow"))?;
                let value = bias.wrapping_add(r_addend);
                mem.write_init(target, &value.to_le_bytes())?;
            } else if r_type == irelative_type {
                // Best-effort: cannot execute the resolver here. Skip; a
                // musl static-pie built without ifuncs never hits this path.
            }
            // Any other type has no place in a static PIE; skip it.
        }
    }

    // Elf64_Rel (no explicit addend) — parsed for completeness. musl's
    // static-pie relocator only ever emits Elf64_Rela, so this is untested by
    // musl output in practice but kept for spec-compliant loaders.
    if let (Some(rel_vaddr), Some(relsz)) = (dt_rel, dt_relsz) {
        let entsz = dt_relent.unwrap_or(REL_LEN as u64);
        if entsz as usize != REL_LEN {
            return Err(LoadError::Malformed("unexpected DT_RELENT"));
        }
        let file_off = vaddr_to_file_offset(phdrs, rel_vaddr)
            .ok_or(LoadError::Malformed("DT_REL not backed by a PT_LOAD segment"))?;
        let count = relsz as usize / REL_LEN;
        for i in 0..count {
            let e = (file_off as usize)
                .checked_add(i * REL_LEN)
                .ok_or(LoadError::Malformed("DT_REL entry overflow"))?;
            let r_offset = read_u64(elf, e)?;
            let r_info = read_u64(elf, e + 8)?;
            let r_type = (r_info & 0xffff_ffff) as u32;

            if r_type == relative_type {
                let target = r_offset
                    .checked_add(bias)
                    .ok_or(LoadError::Malformed("relocation offset overflow"))?;
                // Elf64_Rel has an implicit addend: whatever is already
                // stored at the target (usually the link-time vaddr).
                let existing = mem.read_u64(target)?;
                let value = bias.wrapping_add(existing);
                mem.write_init(target, &value.to_le_bytes())?;
            }
        }
    }

    Ok(())
}

/// Translate a link-time (unbiased) vaddr to a file offset via the `PT_LOAD`
/// segment that covers it, or `None` if no segment's file-backed range does.
fn vaddr_to_file_offset(phdrs: &[Phdr], vaddr: u64) -> Option<u64> {
    phdrs.iter().find_map(|ph| {
        if ph.p_type == PT_LOAD && vaddr >= ph.vaddr && vaddr < ph.vaddr + ph.filesz {
            Some(ph.offset + (vaddr - ph.vaddr))
        } else {
            None
        }
    })
}

// ---- header parsing ------------------------------------------------------

struct Ehdr {
    e_type: u16,
    machine: u16,
    entry: u64,
    phoff: u64,
    phnum: u16,
}

impl Ehdr {
    fn parse(elf: &[u8]) -> Result<Self, LoadError> {
        if elf.len() < EHDR_LEN {
            return Err(LoadError::Truncated);
        }
        if elf[0..4] != ELF_MAGIC {
            return Err(LoadError::NotElf);
        }
        if elf[4] != ELFCLASS64 || elf[5] != ELFDATA2LSB {
            return Err(LoadError::UnsupportedArch);
        }
        let e_type = read_u16(elf, 16)?;
        if e_type != ET_EXEC && e_type != ET_DYN {
            return Err(LoadError::Malformed("unsupported e_type"));
        }
        let machine = read_u16(elf, 18)?;
        if machine != EM_AARCH64 && machine != EM_X86_64 {
            return Err(LoadError::UnsupportedArch);
        }
        let phentsize = read_u16(elf, 54)?;
        if phentsize as usize != PHDR_LEN {
            return Err(LoadError::Malformed("unexpected e_phentsize"));
        }
        Ok(Self {
            e_type,
            machine,
            entry: read_u64(elf, 24)?,
            phoff: read_u64(elf, 32)?,
            phnum: read_u16(elf, 56)?,
        })
    }
}

struct Phdr {
    p_type: u32,
    flags: u32,
    offset: u64,
    vaddr: u64,
    filesz: u64,
    memsz: u64,
}

impl Phdr {
    fn parse(elf: &[u8], off: usize) -> Result<Self, LoadError> {
        if off + PHDR_LEN > elf.len() {
            return Err(LoadError::Truncated);
        }
        Ok(Self {
            p_type: read_u32(elf, off)?,
            flags: read_u32(elf, off + 4)?,
            offset: read_u64(elf, off + 8)?,
            vaddr: read_u64(elf, off + 16)?,
            filesz: read_u64(elf, off + 32)?,
            memsz: read_u64(elf, off + 40)?,
        })
    }
}

fn seg_prot(flags: u32) -> Prot {
    let mut p = Prot::NONE.0;
    if flags & PF_R != 0 {
        p |= Prot::READ.0;
    }
    if flags & PF_W != 0 {
        p |= Prot::WRITE.0;
    }
    if flags & PF_X != 0 {
        p |= Prot::EXEC.0;
    }
    Prot(p)
}

// ---- initial stack -------------------------------------------------------

fn build_stack(
    mem: &mut GuestMemory,
    spec: &ProcessSpec,
    ehdr: &Ehdr,
    entry: u64,
    phdr_vaddr: Option<u64>,
) -> Result<u64, LoadError> {
    let top = mem.base() + mem.size();
    let stack_bottom = top - STACK_SIZE;
    mem.map(stack_bottom, STACK_SIZE, Prot::rw())?;

    // String blob: argv then envp, each NUL-terminated, placed high.
    let mut blob = Vec::new();
    let mut arg_off = Vec::with_capacity(spec.argv.len());
    for a in &spec.argv {
        arg_off.push(blob.len() as u64);
        blob.extend_from_slice(a.as_bytes());
        blob.push(0);
    }
    let mut env_off = Vec::with_capacity(spec.envp.len());
    for e in &spec.envp {
        env_off.push(blob.len() as u64);
        blob.extend_from_slice(e.as_bytes());
        blob.push(0);
    }

    let str_base = (top - blob.len() as u64) & !0x7;
    let random_addr = (str_base - 16) & !0xf;
    let execfn = str_base + arg_off.first().copied().unwrap_or(0);

    let auxv: [(u64, u64); 16] = [
        (AT_PHDR, phdr_vaddr.unwrap_or(0)),
        (AT_PHENT, PHDR_LEN as u64),
        (AT_PHNUM, u64::from(ehdr.phnum)),
        (AT_PAGESZ, PAGE_SIZE),
        // No separate interpreter is ever loaded (static executables only,
        // including static PIEs relocated in-place by this loader), so
        // AT_BASE is always 0.
        (AT_BASE, 0),
        (AT_FLAGS, 0),
        (AT_ENTRY, entry),
        (AT_UID, 0),
        (AT_EUID, 0),
        (AT_GID, 0),
        (AT_EGID, 0),
        (AT_CLKTCK, 100),
        (AT_SECURE, 0),
        (AT_RANDOM, random_addr),
        (AT_EXECFN, execfn),
        (AT_NULL, 0),
    ];

    let nwords = 1                       // argc
        + spec.argv.len() + 1            // argv + NULL
        + spec.envp.len() + 1            // envp + NULL
        + auxv.len() * 2; // auxv pairs (incl. AT_NULL)
    let vec_bytes = nwords as u64 * 8;
    let sp = (random_addr - vec_bytes) & !0xf;

    if sp < stack_bottom {
        return Err(LoadError::Malformed("initial stack does not fit"));
    }

    // Populate the string/random areas, then the vector.
    mem.write_init(str_base, &blob)?;
    mem.write_init(random_addr, &[0u8; 16])?;

    let mut cur = sp;
    let mut push = |val: u64, mem: &mut GuestMemory| -> Result<(), LoadError> {
        mem.write_init(cur, &val.to_le_bytes())?;
        cur += 8;
        Ok(())
    };

    push(spec.argv.len() as u64, mem)?;
    for off in &arg_off {
        push(str_base + off, mem)?;
    }
    push(0, mem)?; // argv NULL
    for off in &env_off {
        push(str_base + off, mem)?;
    }
    push(0, mem)?; // envp NULL
    for (tag, val) in auxv {
        push(tag, mem)?;
        push(val, mem)?;
    }

    Ok(sp)
}

// ---- little-endian readers ----------------------------------------------

fn read_u16(b: &[u8], off: usize) -> Result<u16, LoadError> {
    let s = b.get(off..off + 2).ok_or(LoadError::Truncated)?;
    Ok(u16::from_le_bytes([s[0], s[1]]))
}
fn read_u32(b: &[u8], off: usize) -> Result<u32, LoadError> {
    let s = b.get(off..off + 4).ok_or(LoadError::Truncated)?;
    Ok(u32::from_le_bytes([s[0], s[1], s[2], s[3]]))
}
fn read_u64(b: &[u8], off: usize) -> Result<u64, LoadError> {
    let s = b.get(off..off + 8).ok_or(LoadError::Truncated)?;
    let mut a = [0u8; 8];
    a.copy_from_slice(s);
    Ok(u64::from_le_bytes(a))
}

const fn round_up(v: u64, align: u64) -> u64 {
    v.div_ceil(align) * align
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal static ELF64 with one RWX PT_LOAD segment carrying
    /// `code` at `vaddr`, entry == vaddr. The ELF header and program header sit
    /// at the front of the file and are covered by the segment (offset 0).
    fn tiny_elf(machine: u16, vaddr: u64, code: &[u8]) -> Vec<u8> {
        let mut f = vec![0u8; EHDR_LEN + PHDR_LEN];
        f[0..4].copy_from_slice(&ELF_MAGIC);
        f[4] = ELFCLASS64;
        f[5] = ELFDATA2LSB;
        f[6] = 1; // EI_VERSION
        f[16..18].copy_from_slice(&2u16.to_le_bytes()); // e_type = ET_EXEC
        f[18..20].copy_from_slice(&machine.to_le_bytes());
        f[20..24].copy_from_slice(&1u32.to_le_bytes()); // e_version
        let code_off = (EHDR_LEN + PHDR_LEN) as u64;
        f[24..32].copy_from_slice(&(vaddr + code_off).to_le_bytes()); // e_entry
        f[32..40].copy_from_slice(&(EHDR_LEN as u64).to_le_bytes()); // e_phoff
        f[52..54].copy_from_slice(&(EHDR_LEN as u16).to_le_bytes()); // e_ehsize
        f[54..56].copy_from_slice(&(PHDR_LEN as u16).to_le_bytes()); // e_phentsize
        f[56..58].copy_from_slice(&1u16.to_le_bytes()); // e_phnum

        // one PT_LOAD, RWX, offset 0, covering the whole file + code.
        let p = EHDR_LEN;
        let total = code_off + code.len() as u64;
        f[p..p + 4].copy_from_slice(&PT_LOAD.to_le_bytes());
        f[p + 4..p + 8].copy_from_slice(&(PF_R | PF_W | PF_X).to_le_bytes());
        f[p + 8..p + 16].copy_from_slice(&0u64.to_le_bytes()); // p_offset
        f[p + 16..p + 24].copy_from_slice(&vaddr.to_le_bytes()); // p_vaddr
        f[p + 24..p + 32].copy_from_slice(&vaddr.to_le_bytes()); // p_paddr
        f[p + 32..p + 40].copy_from_slice(&total.to_le_bytes()); // p_filesz
        f[p + 40..p + 48].copy_from_slice(&total.to_le_bytes()); // p_memsz
        f[p + 48..p + 56].copy_from_slice(&PAGE_SIZE.to_le_bytes()); // p_align

        f.extend_from_slice(code);
        f
    }

    fn spec() -> ProcessSpec {
        ProcessSpec {
            argv: vec!["prog".into(), "arg1".into()],
            envp: vec!["PATH=/bin".into()],
        }
    }

    /// Build a minimal static-PIE ELF64 (`ET_DYN`, no `PT_INTERP`): a single
    /// RWX `PT_LOAD` at link-time `p_vaddr = 0` covering the whole file (like
    /// `tiny_elf`, but 0-based since a PIE has no preferred load address),
    /// plus a `PT_DYNAMIC` segment describing one `R_*_RELATIVE` entry in a
    /// `.rela.dyn`-style table.
    ///
    /// File layout: `[Ehdr][Phdr×2][code][Dyn×4][Rela×1][reloc target u64]`.
    /// Returns `(file, unbiased entry, unbiased reloc-target vaddr, r_addend)`.
    fn tiny_pie_elf(machine: u16, code: &[u8]) -> (Vec<u8>, u64, u64, u64) {
        let phnum = 2u64;
        let headers_len = EHDR_LEN as u64 + phnum * PHDR_LEN as u64;
        let code_off = headers_len;
        let code_len = code.len() as u64;
        let dyn_off = code_off + code_len;
        let dyn_len = 4 * DYN_LEN as u64; // DT_RELA, DT_RELASZ, DT_RELAENT, DT_NULL
        let rela_off = dyn_off + dyn_len;
        let rela_len = RELA_LEN as u64; // one entry
        let reloc_target_off = rela_off + rela_len;
        let total = reloc_target_off + 8;

        let relative_type = match machine {
            EM_AARCH64 => R_AARCH64_RELATIVE,
            EM_X86_64 => R_X86_64_RELATIVE,
            _ => panic!("tiny_pie_elf: unsupported test machine"),
        };
        let r_addend = 0x1234u64;

        let mut f = vec![0u8; total as usize];
        f[0..4].copy_from_slice(&ELF_MAGIC);
        f[4] = ELFCLASS64;
        f[5] = ELFDATA2LSB;
        f[6] = 1; // EI_VERSION
        f[16..18].copy_from_slice(&ET_DYN.to_le_bytes());
        f[18..20].copy_from_slice(&machine.to_le_bytes());
        f[20..24].copy_from_slice(&1u32.to_le_bytes()); // e_version
        f[24..32].copy_from_slice(&code_off.to_le_bytes()); // e_entry
        f[32..40].copy_from_slice(&(EHDR_LEN as u64).to_le_bytes()); // e_phoff
        f[52..54].copy_from_slice(&(EHDR_LEN as u16).to_le_bytes()); // e_ehsize
        f[54..56].copy_from_slice(&(PHDR_LEN as u16).to_le_bytes()); // e_phentsize
        f[56..58].copy_from_slice(&(phnum as u16).to_le_bytes()); // e_phnum

        // Phdr 0: PT_LOAD, p_vaddr = p_offset = 0, covers the whole file.
        let p0 = EHDR_LEN;
        f[p0..p0 + 4].copy_from_slice(&PT_LOAD.to_le_bytes());
        f[p0 + 4..p0 + 8].copy_from_slice(&(PF_R | PF_W | PF_X).to_le_bytes());
        f[p0 + 8..p0 + 16].copy_from_slice(&0u64.to_le_bytes()); // p_offset
        f[p0 + 16..p0 + 24].copy_from_slice(&0u64.to_le_bytes()); // p_vaddr
        f[p0 + 24..p0 + 32].copy_from_slice(&0u64.to_le_bytes()); // p_paddr
        f[p0 + 32..p0 + 40].copy_from_slice(&total.to_le_bytes()); // p_filesz
        f[p0 + 40..p0 + 48].copy_from_slice(&total.to_le_bytes()); // p_memsz
        f[p0 + 48..p0 + 56].copy_from_slice(&PAGE_SIZE.to_le_bytes()); // p_align

        // Phdr 1: PT_DYNAMIC, vaddr == offset == dyn_off (PT_LOAD is 1:1
        // vaddr<->file-offset here since it starts at vaddr 0 / offset 0).
        let p1 = EHDR_LEN + PHDR_LEN;
        f[p1..p1 + 4].copy_from_slice(&PT_DYNAMIC.to_le_bytes());
        f[p1 + 4..p1 + 8].copy_from_slice(&(PF_R | PF_W).to_le_bytes());
        f[p1 + 8..p1 + 16].copy_from_slice(&dyn_off.to_le_bytes()); // p_offset
        f[p1 + 16..p1 + 24].copy_from_slice(&dyn_off.to_le_bytes()); // p_vaddr
        f[p1 + 24..p1 + 32].copy_from_slice(&dyn_off.to_le_bytes()); // p_paddr
        f[p1 + 32..p1 + 40].copy_from_slice(&dyn_len.to_le_bytes()); // p_filesz
        f[p1 + 40..p1 + 48].copy_from_slice(&dyn_len.to_le_bytes()); // p_memsz
        f[p1 + 48..p1 + 56].copy_from_slice(&8u64.to_le_bytes()); // p_align

        f[code_off as usize..(code_off + code_len) as usize].copy_from_slice(code);

        // Dynamic array: DT_RELA, DT_RELASZ, DT_RELAENT, DT_NULL.
        let d = dyn_off as usize;
        f[d..d + 8].copy_from_slice(&DT_RELA.to_le_bytes());
        f[d + 8..d + 16].copy_from_slice(&rela_off.to_le_bytes());
        f[d + 16..d + 24].copy_from_slice(&DT_RELASZ.to_le_bytes());
        f[d + 24..d + 32].copy_from_slice(&rela_len.to_le_bytes());
        f[d + 32..d + 40].copy_from_slice(&DT_RELAENT.to_le_bytes());
        f[d + 40..d + 48].copy_from_slice(&(RELA_LEN as u64).to_le_bytes());
        f[d + 48..d + 56].copy_from_slice(&DT_NULL.to_le_bytes());
        f[d + 56..d + 64].copy_from_slice(&0u64.to_le_bytes());

        // Rela entry: r_offset = reloc_target_off, r_info = relative_type
        // (r_sym == 0), r_addend.
        let r = rela_off as usize;
        f[r..r + 8].copy_from_slice(&reloc_target_off.to_le_bytes());
        f[r + 8..r + 16].copy_from_slice(&u64::from(relative_type).to_le_bytes());
        f[r + 16..r + 24].copy_from_slice(&r_addend.to_le_bytes());

        (f, code_off, reloc_target_off, r_addend)
    }

    #[test]
    fn rejects_non_elf() {
        let mut mem = GuestMemory::new(0x1_0000, 16 * PAGE_SIZE);
        let bytes = vec![b'x'; 128];
        assert!(matches!(
            load_static(&mut mem, &bytes, &spec()),
            Err(LoadError::NotElf)
        ));
    }

    #[test]
    fn loads_segment_and_reports_entry() {
        let vaddr = 0x1_0000u64;
        let code = 0xD400_0001u32.to_le_bytes(); // svc #0
        let elf = tiny_elf(EM_AARCH64, vaddr, &code);
        let mut mem = GuestMemory::new(vaddr, 128 * PAGE_SIZE);

        let img = load_static(&mut mem, &elf, &spec()).unwrap();

        let code_addr = vaddr + (EHDR_LEN + PHDR_LEN) as u64;
        assert_eq!(img.entry, code_addr);
        // The code is present at the entry and executable.
        assert_eq!(mem.read_u32(code_addr).unwrap(), 0xD400_0001);
        // Break is above the loaded image, page-aligned.
        assert!(img.program_break > code_addr);
        assert_eq!(img.program_break % PAGE_SIZE, 0);
    }

    #[test]
    fn stack_has_argc_argv_envp_and_auxv() {
        let vaddr = 0x1_0000u64;
        let elf = tiny_elf(EM_AARCH64, vaddr, &[0xD4, 0x00, 0x00, 0x01]);
        let mut mem = GuestMemory::new(vaddr, 128 * PAGE_SIZE);
        let img = load_static(&mut mem, &elf, &spec()).unwrap();
        let sp = img.stack_pointer;

        assert_eq!(sp % 16, 0, "sp must be 16-byte aligned");
        assert_eq!(mem.read_u64(sp).unwrap(), 2, "argc == 2");

        // argv[0] -> "prog"
        let argv0 = mem.read_u64(sp + 8).unwrap();
        assert_eq!(mem.read_cstr(argv0, 64).unwrap(), b"prog");
        // argv[1] -> "arg1", then NULL terminator.
        let argv1 = mem.read_u64(sp + 16).unwrap();
        assert_eq!(mem.read_cstr(argv1, 64).unwrap(), b"arg1");
        assert_eq!(mem.read_u64(sp + 24).unwrap(), 0, "argv NULL");

        // envp[0] -> "PATH=/bin", then NULL.
        let env0 = mem.read_u64(sp + 32).unwrap();
        assert_eq!(mem.read_cstr(env0, 64).unwrap(), b"PATH=/bin");
        assert_eq!(mem.read_u64(sp + 40).unwrap(), 0, "envp NULL");

        // Walk the auxv (starts after envp NULL) and find AT_ENTRY.
        let mut a = sp + 48;
        let mut found_entry = None;
        loop {
            let tag = mem.read_u64(a).unwrap();
            let val = mem.read_u64(a + 8).unwrap();
            if tag == AT_NULL {
                break;
            }
            if tag == AT_ENTRY {
                found_entry = Some(val);
            }
            a += 16;
        }
        assert_eq!(found_entry, Some(img.entry), "AT_ENTRY matches entry");
    }

    #[test]
    fn static_pie_loads_relocates_and_biases_entry() {
        let code = 0xD400_0001u32.to_le_bytes(); // svc #0, content is irrelevant
        let (elf, entry_unbiased, reloc_target_off, r_addend) = tiny_pie_elf(EM_AARCH64, &code);

        // A region base unrelated to any vaddr baked into the file: proves
        // the bias, not a coincidence of matching numbers, drove the result.
        let region_base = 0x40_0000u64;
        let mut mem = GuestMemory::new(region_base, 128 * PAGE_SIZE);

        // (1) it loads without error.
        let img = load_static(&mut mem, &elf, &spec()).unwrap();

        // load_static anchors a static PIE's bias at the guest region's base.
        let bias = region_base;

        // (3) entry equals the biased entry.
        assert_eq!(img.entry, bias + entry_unbiased);

        // (2) the relocated word at base + r_offset equals base + r_addend.
        let relocated = mem.read_u64(bias + reloc_target_off).unwrap();
        assert_eq!(relocated, bias + r_addend);

        // Break accounts for the bias and is page-aligned.
        assert!(img.program_break > bias);
        assert_eq!(img.program_break % PAGE_SIZE, 0);

        // AT_ENTRY and AT_BASE in the built stack agree with the biased image.
        let sp = img.stack_pointer;
        let mut a = sp + 8 * (1 + spec().argv.len() as u64 + 1 + spec().envp.len() as u64 + 1);
        let mut found_entry = None;
        let mut found_base = None;
        loop {
            let tag = mem.read_u64(a).unwrap();
            let val = mem.read_u64(a + 8).unwrap();
            if tag == AT_NULL {
                break;
            }
            if tag == AT_ENTRY {
                found_entry = Some(val);
            }
            if tag == AT_BASE {
                found_base = Some(val);
            }
            a += 16;
        }
        assert_eq!(found_entry, Some(img.entry));
        assert_eq!(found_base, Some(0), "no separate interpreter for a static PIE");
    }

    #[test]
    fn static_pie_relocates_on_x86_64_too() {
        let code = [0x90u8]; // nop, content is irrelevant
        let (elf, entry_unbiased, reloc_target_off, r_addend) = tiny_pie_elf(EM_X86_64, &code);
        let region_base = 0x80_0000u64;
        let mut mem = GuestMemory::new(region_base, 128 * PAGE_SIZE);

        let img = load_static(&mut mem, &elf, &spec()).unwrap();
        let bias = region_base;

        assert_eq!(img.entry, bias + entry_unbiased);
        let relocated = mem.read_u64(bias + reloc_target_off).unwrap();
        assert_eq!(relocated, bias + r_addend);
    }

    #[test]
    fn et_exec_still_loads_with_zero_bias() {
        // Existing ET_EXEC behavior is untouched: the region base and the
        // file's own p_vaddr coincide, and img.entry lands exactly on the
        // link-time vaddr (bias == 0), not the region base plus vaddr.
        let vaddr = 0x1_0000u64;
        let code = 0xD400_0001u32.to_le_bytes();
        let elf = tiny_elf(EM_AARCH64, vaddr, &code);
        let mut mem = GuestMemory::new(vaddr, 128 * PAGE_SIZE);

        let img = load_static(&mut mem, &elf, &spec()).unwrap();

        let code_addr = vaddr + (EHDR_LEN + PHDR_LEN) as u64;
        assert_eq!(img.entry, code_addr, "ET_EXEC entry is unbiased");
    }
}
