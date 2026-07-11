//! Turning an ELF file into a ready-to-run guest process image.
//!
//! Responsibilities:
//! * parse ELF64 headers and program headers (hand-rolled, no external dep),
//! * map `PT_LOAD` segments into [`GuestMemory`] with correct protections,
//! * build the initial stack: `argc`, `argv`, `envp`, and the auxiliary vector
//!   (`AT_PHDR`, `AT_ENTRY`, `AT_RANDOM`, `AT_EXECFN`, …),
//! * report the entry PC, initial SP, and program break.
//!
//! Static executables work now. The dynamic linker (`PT_INTERP` → `ld-musl`)
//! and a vDSO arrive in ROADMAP Phase 5.

use crate::vcpu::mem::{PAGE_SIZE, Prot};
use crate::vcpu::{GuestMemory, MemError};

// ---- ELF constants -------------------------------------------------------

const ELF_MAGIC: [u8; 4] = [0x7f, b'E', b'L', b'F'];
const ELFCLASS64: u8 = 2;
const ELFDATA2LSB: u8 = 1;
const EM_X86_64: u16 = 62;
const EM_AARCH64: u16 = 183;

const PT_LOAD: u32 = 1;
const PF_X: u32 = 1;
const PF_W: u32 = 2;
const PF_R: u32 = 4;

const EHDR_LEN: usize = 64;
const PHDR_LEN: usize = 56;

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
pub fn load_static(
    mem: &mut GuestMemory,
    elf: &[u8],
    spec: &ProcessSpec,
) -> Result<LoadedImage, LoadError> {
    let ehdr = Ehdr::parse(elf)?;

    let mut program_break = 0u64;
    let mut phdr_vaddr: Option<u64> = None;

    for i in 0..ehdr.phnum {
        let off = ehdr.phoff as usize + i as usize * PHDR_LEN;
        let ph = Phdr::parse(elf, off)?;
        if ph.p_type != PT_LOAD || ph.memsz == 0 {
            continue;
        }

        // Map the segment's pages with its final protection, then load the file
        // bytes via write_init (which bypasses protection). The tail
        // [filesz, memsz) is .bss and stays zeroed.
        let prot = seg_prot(ph.flags);
        mem.map(ph.vaddr, ph.memsz, prot)?;
        if ph.filesz > 0 {
            let file_end = ph
                .offset
                .checked_add(ph.filesz)
                .ok_or(LoadError::Malformed("segment offset overflow"))?;
            let bytes = elf
                .get(ph.offset as usize..file_end as usize)
                .ok_or(LoadError::Truncated)?;
            mem.write_init(ph.vaddr, bytes)?;
        }

        program_break = program_break.max(round_up(ph.vaddr + ph.memsz, PAGE_SIZE));

        // If this segment contains the program headers, record their vaddr for
        // AT_PHDR.
        if ph.offset <= ehdr.phoff && ehdr.phoff < ph.offset + ph.filesz {
            phdr_vaddr = Some(ph.vaddr + (ehdr.phoff - ph.offset));
        }
    }

    if program_break == 0 {
        return Err(LoadError::Malformed("no loadable segments"));
    }

    let stack_pointer = build_stack(mem, spec, &ehdr, phdr_vaddr)?;

    Ok(LoadedImage {
        entry: ehdr.entry,
        stack_pointer,
        program_break,
        stack_bottom: (mem.base() + mem.size()) - STACK_SIZE,
    })
}

// ---- header parsing ------------------------------------------------------

struct Ehdr {
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
        let machine = read_u16(elf, 18)?;
        if machine != EM_AARCH64 && machine != EM_X86_64 {
            return Err(LoadError::UnsupportedArch);
        }
        let phentsize = read_u16(elf, 54)?;
        if phentsize as usize != PHDR_LEN {
            return Err(LoadError::Malformed("unexpected e_phentsize"));
        }
        Ok(Self {
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
        (AT_BASE, 0),
        (AT_FLAGS, 0),
        (AT_ENTRY, ehdr.entry),
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
}
