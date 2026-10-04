//! A64 "Loads and Stores" (`op0 = x1x0`): every addressing form of the
//! general-purpose and SIMD&FP single-register loads/stores, pairs (including
//! the no-allocate `LDNP`/`STNP` and `LDPSW`), the unprivileged `LDTR*`/`STTR*`
//! (which behave as ordinary accesses at EL0), `PRFM`/`PRFUM` (no-ops),
//! literal loads, the exclusive/acquire-release family (`LDXR`/`STXR`/
//! `LDAXP`/`STLXP`/`LDAR`/`STLR`/…), the LSE atomics (`CAS`/`CASP`/`SWP`/
//! `LD<op>`/`ST<op>`, all sizes and ordering variants), and the Advanced SIMD
//! structure loads/stores (`LD1`–`LD4`/`ST1`–`ST4`, multiple and single
//! structure, `LD1R`–`LD4R`, with post-index).
//!
//! Atomics are plain read-modify-writes: the SMP layer runs one vcpu of an
//! address space at a time, holding its memory lock, so nothing can observe
//! the intermediate state. The same reasoning keeps the exclusive monitor a
//! simple per-vcpu flag (see `Aarch64Interp::excl_monitor`).

use super::alu::{extend_reg, ones, sign_extend};
use super::{Aarch64Interp, Step, reg_field};
use crate::vcpu::GuestMemory;

/// What a general-purpose `size:opc` load/store encoding does.
#[derive(Clone, Copy, PartialEq, Eq)]
enum GprOp {
    Store,
    /// Load, zero-extended.
    Load,
    /// Load, sign-extended to 64 bits.
    LoadSx64,
    /// Load, sign-extended to 32 bits (upper 32 bits zero).
    LoadSx32,
    Prefetch,
}

/// Decode `size:opc` of a general-purpose single-register load/store.
fn gpr_op(size: u32, opc: u32, prefetch_ok: bool) -> Option<GprOp> {
    Some(match (size, opc) {
        (_, 0b00) => GprOp::Store,
        (_, 0b01) => GprOp::Load,
        (0b11, 0b10) if prefetch_ok => GprOp::Prefetch,
        (0b00..=0b10, 0b10) => GprOp::LoadSx64,
        (0b00 | 0b01, 0b11) => GprOp::LoadSx32,
        _ => return None,
    })
}

/// Decode `size:opc` of a SIMD&FP single-register load/store into
/// `(log2 bytes, is_load)`.
fn fp_op(size: u32, opc: u32) -> Option<(u32, bool)> {
    if opc & 2 == 0 {
        Some((size, opc & 1 == 1))
    } else if size == 0 {
        Some((4, opc & 1 == 1))
    } else {
        None
    }
}

impl Aarch64Interp {
    pub(super) fn exec_ldst(&mut self, instr: u32, mem: &mut GuestMemory) -> Step {
        let vector = (instr >> 26) & 1 == 1;
        // SP alignment check (SCTLR_EL1.SA0, set by Linux): any access based
        // on a misaligned SP faults. (The literal form has no base register;
        // prefetches — PRFM unsigned-offset/register and PRFUM, the two
        // masks below — never fault.)
        if (instr >> 5) & 0x1f == 31
            && self.sp & 15 != 0
            && !((instr >> 28) & 3 == 0b01 && (instr >> 24) & 1 == 0)
            && instr & 0xFFC0_0000 != 0xF980_0000
            && instr & 0xFFC0_0000 != 0xF880_0000
        {
            return Step::Fault {
                addr: self.sp,
                write: false,
            };
        }
        match (instr >> 28) & 3 {
            0b00 => {
                if vector {
                    self.exec_simd_struct(instr, mem)
                } else if (instr >> 24) & 1 == 0 {
                    self.exec_exclusive(instr, mem)
                } else {
                    Step::Illegal // RCPC3 / MTE tag loads and stores
                }
            }
            0b01 => {
                if (instr >> 24) & 1 == 0 {
                    self.exec_literal(instr, mem)
                } else if !vector && (instr >> 21) & 1 == 0 && (instr >> 10) & 3 == 0 {
                    self.exec_rcpc2(instr, mem)
                } else {
                    Step::Illegal // MOPS, RCPC3 SIMD forms
                }
            }
            0b10 => self.exec_pair(instr, mem),
            _ => {
                if (instr >> 24) & 1 == 1 {
                    // Unsigned scaled 12-bit immediate offset.
                    let imm12 = u64::from((instr >> 10) & 0xfff);
                    self.exec_single(instr, mem, Addr::Unsigned(imm12))
                } else if (instr >> 21) & 1 == 0 {
                    let imm9 = sign_extend(u64::from((instr >> 12) & 0x1ff), 9);
                    let mode = match (instr >> 10) & 3 {
                        0b00 => Addr::Unscaled(imm9),
                        0b01 => Addr::Post(imm9),
                        0b10 => Addr::Unprivileged(imm9),
                        _ => Addr::Pre(imm9),
                    };
                    self.exec_single(instr, mem, mode)
                } else {
                    match (instr >> 10) & 3 {
                        0b00 if !vector => self.exec_atomic(instr, mem),
                        0b10 => self.exec_single(instr, mem, Addr::Register),
                        _ => Step::Illegal, // LDRAA/LDRAB (PAuth)
                    }
                }
            }
        }
    }

    /// Single-register loads/stores (GPR or SIMD&FP) in every addressing
    /// mode but literal.
    fn exec_single(&mut self, instr: u32, mem: &mut GuestMemory, mode: Addr) -> Step {
        let size = instr >> 30;
        let opc = (instr >> 22) & 3;
        let vector = (instr >> 26) & 1 == 1;
        let rt = reg_field(instr, 0);
        let rn = reg_field(instr, 5);
        let (scale, op) = if vector {
            if matches!(mode, Addr::Unprivileged(_)) {
                return Step::Illegal;
            }
            let Some((scale, load)) = fp_op(size, opc) else {
                return Step::Illegal;
            };
            (scale, if load { GprOp::Load } else { GprOp::Store })
        } else {
            let prefetch_ok =
                matches!(mode, Addr::Unsigned(_) | Addr::Unscaled(_) | Addr::Register);
            let Some(op) = gpr_op(size, opc, prefetch_ok) else {
                return Step::Illegal;
            };
            (size, op)
        };
        let base = self.read_sp(rn);
        let (addr, wback) = match mode {
            Addr::Unsigned(imm) => (base.wrapping_add(imm << scale), None),
            Addr::Unscaled(imm) | Addr::Unprivileged(imm) => (base.wrapping_add(imm as u64), None),
            Addr::Pre(imm) => {
                let a = base.wrapping_add(imm as u64);
                (a, Some(a))
            }
            Addr::Post(imm) => (base, Some(base.wrapping_add(imm as u64))),
            Addr::Register => {
                let option = (instr >> 13) & 7;
                if option & 2 == 0 {
                    return Step::Illegal;
                }
                let shift = if (instr >> 12) & 1 == 1 { scale } else { 0 };
                let off = extend_reg(self.read_x(reg_field(instr, 16)), option, shift);
                (base.wrapping_add(off), None)
            }
        };
        let step = if vector {
            self.ldst_vec(addr, scale, op == GprOp::Load, rt, mem)
        } else {
            match op {
                GprOp::Prefetch => Step::Next,
                GprOp::Store => self.store_x(addr, scale, rt, mem),
                GprOp::Load => self.load_x(addr, scale, rt, 0, mem),
                GprOp::LoadSx64 => self.load_x(addr, scale, rt, 64, mem),
                GprOp::LoadSx32 => self.load_x(addr, scale, rt, 32, mem),
            }
        };
        if let (Step::Next, Some(a)) = (&step, wback) {
            self.write_sp(rn, a);
        }
        step
    }

    /// Load `1 << scale` bytes into `X[rt]`, zero-extended (`sx == 0`) or
    /// sign-extended to 64 or 32 bits.
    #[inline]
    fn load_x(&mut self, addr: u64, scale: u32, rt: usize, sx: u32, mem: &mut GuestMemory) -> Step {
        let nbytes = 1usize << scale;
        let mut buf = [0u8; 8];
        if mem.read(addr, &mut buf[..nbytes]).is_err() {
            return Step::Fault { addr, write: false };
        }
        let raw = u64::from_le_bytes(buf);
        let val = match sx {
            0 => raw,
            64 => sign_extend(raw, 8 << scale) as u64,
            _ => sign_extend(raw, 8 << scale) as u64 & 0xffff_ffff,
        };
        self.write_x(rt, val);
        Step::Next
    }

    /// Store the low `1 << scale` bytes of `X[rt]`.
    #[inline]
    fn store_x(&mut self, addr: u64, scale: u32, rt: usize, mem: &mut GuestMemory) -> Step {
        let nbytes = 1usize << scale;
        let value = self.read_x(rt);
        self.note_store(addr, u128::from(value), nbytes);
        if let Err(e) = mem.write_trap(addr, &value.to_le_bytes()[..nbytes]) {
            return Step::Fault {
                addr: e.fault_addr(),
                write: true,
            };
        }
        Step::Next
    }

    /// Load (zero-extending) or store `1 << scale` bytes of SIMD register `rt`.
    pub(super) fn ldst_vec(
        &mut self,
        addr: u64,
        scale: u32,
        is_load: bool,
        rt: usize,
        mem: &mut GuestMemory,
    ) -> Step {
        let nbytes = 1usize << scale;
        if is_load {
            let mut buf = [0u8; 16];
            if mem.read(addr, &mut buf[..nbytes]).is_err() {
                return Step::Fault { addr, write: false };
            }
            self.v[rt] = u128::from_le_bytes(buf);
        } else {
            self.note_store(addr, self.v[rt], nbytes);
            if let Err(e) = mem.write_trap(addr, &self.v[rt].to_le_bytes()[..nbytes]) {
                return Step::Fault {
                    addr: e.fault_addr(),
                    write: true,
                };
            }
        }
        Step::Next
    }

    /// `LDR` (literal) / `LDRSW` (literal) / `PRFM` (literal), GPR and SIMD&FP.
    fn exec_literal(&mut self, instr: u32, mem: &mut GuestMemory) -> Step {
        let opc = instr >> 30;
        let rt = reg_field(instr, 0);
        let addr = self
            .pc
            .wrapping_add((sign_extend(u64::from((instr >> 5) & 0x7ffff), 19) << 2) as u64);
        if (instr >> 26) & 1 == 1 {
            if opc == 3 {
                return Step::Illegal;
            }
            return self.ldst_vec(addr, opc + 2, true, rt, mem);
        }
        match opc {
            0 => self.load_x(addr, 2, rt, 0, mem),
            1 => self.load_x(addr, 3, rt, 0, mem),
            2 => self.load_x(addr, 2, rt, 64, mem),
            _ => Step::Next, // PRFM (literal)
        }
    }

    /// Load/store pair: `LDP`/`STP`/`LDNP`/`STNP`/`LDPSW`, GPR and SIMD&FP,
    /// in no-allocate, post-index, signed-offset and pre-index forms.
    fn exec_pair(&mut self, instr: u32, mem: &mut GuestMemory) -> Step {
        let opc = instr >> 30;
        let vector = (instr >> 26) & 1 == 1;
        let mode = (instr >> 23) & 3; // 00 no-alloc, 01 post, 10 offset, 11 pre
        let load = (instr >> 22) & 1 == 1;
        let rt = reg_field(instr, 0);
        let rt2 = reg_field(instr, 10);
        let rn = reg_field(instr, 5);
        let (scale, sx) = if vector {
            if opc == 3 {
                return Step::Illegal;
            }
            (opc + 2, false)
        } else {
            match opc {
                0b00 => (2, false),
                0b01 if load && mode != 0 => (2, true), // LDPSW
                0b10 => (3, false),
                _ => return Step::Illegal, // STGP (MTE), LSE128
            }
        };
        let off = (sign_extend(u64::from((instr >> 15) & 0x7f), 7) << scale) as u64;
        let base = self.read_sp(rn);
        let addr = if mode == 0b01 {
            base
        } else {
            base.wrapping_add(off)
        };
        let nbytes = 1usize << scale;
        if load {
            let mut buf = [0u8; 32];
            if mem.read(addr, &mut buf[..2 * nbytes]).is_err() {
                return Step::Fault { addr, write: false };
            }
            let mut lo = [0u8; 16];
            let mut hi = [0u8; 16];
            lo[..nbytes].copy_from_slice(&buf[..nbytes]);
            hi[..nbytes].copy_from_slice(&buf[nbytes..2 * nbytes]);
            let (a, b) = (u128::from_le_bytes(lo), u128::from_le_bytes(hi));
            if vector {
                self.v[rt] = a;
                self.v[rt2] = b;
            } else if sx {
                self.write_x(rt, sign_extend(a as u64, 32) as u64);
                self.write_x(rt2, sign_extend(b as u64, 32) as u64);
            } else {
                self.write_x(rt, a as u64);
                self.write_x(rt2, b as u64);
            }
        } else {
            let (a, b) = if vector {
                (self.v[rt], self.v[rt2])
            } else {
                (u128::from(self.read_x(rt)), u128::from(self.read_x(rt2)))
            };
            let mut buf = [0u8; 32];
            buf[..nbytes].copy_from_slice(&a.to_le_bytes()[..nbytes]);
            buf[nbytes..2 * nbytes].copy_from_slice(&b.to_le_bytes()[..nbytes]);
            self.note_store(addr, a, 2 * nbytes);
            if let Err(e) = mem.write_trap(addr, &buf[..2 * nbytes]) {
                return Step::Fault {
                    addr: e.fault_addr(),
                    write: true,
                };
            }
        }
        match mode {
            0b01 => self.write_sp(rn, base.wrapping_add(off)),
            0b11 => self.write_sp(rn, addr),
            _ => {}
        }
        Step::Next
    }

    /// Load/store exclusive (register and pair), load-acquire/store-release,
    /// and the LSE compare-and-swap (`CAS*`, `CASP*`) family, which shares
    /// this encoding space: `size 001000 o2 L o1 Rs o0 Rt2 Rn Rt`.
    fn exec_exclusive(&mut self, instr: u32, mem: &mut GuestMemory) -> Step {
        let size = instr >> 30;
        let o2 = (instr >> 23) & 1;
        let load = (instr >> 22) & 1 == 1;
        let o1 = (instr >> 21) & 1;
        let o0 = (instr >> 15) & 1;
        let rs = reg_field(instr, 16);
        let rt2 = reg_field(instr, 10);
        let rn = reg_field(instr, 5);
        let rt = reg_field(instr, 0);
        let addr = self.read_sp(rn);
        let unallocated = match (o2, o1) {
            (1, 0) => o0 == 0,                                             // LDLAR/STLLR
            (1, 1) => rt2 != 31,                                           // CAS
            (0, 1) if size < 2 => rs & 1 != 0 || rt & 1 != 0 || rt2 != 31, // CASP
            _ => false,
        };
        if unallocated {
            return Step::Illegal;
        }
        // Ordered/atomic accesses must not cross a 16-byte boundary (FEAT_LSE2
        // single-copy atomicity; real cores raise an alignment fault).
        let bytes = if o1 == 1 && (o2 == 1 || size < 2) {
            if o2 == 1 {
                1u64 << size
            } else {
                8 << (size & 1)
            } // CAS / CASP
        } else if o1 == 1 {
            2u64 << size // LDXP/STXP
        } else {
            1u64 << size
        };
        if crosses_granule(addr, bytes) {
            return Step::Fault { addr, write: !load };
        }
        match (o2, o1) {
            (0, 0) => {
                // LDXR/LDAXR/STXR/STLXR (B/H/W/X).
                let nbytes = 1usize << size;
                if load {
                    let Some(v) = mem_read_sized(mem, addr, nbytes) else {
                        return Step::Fault { addr, write: false };
                    };
                    self.write_x(rt, v);
                    self.open_monitor(addr);
                } else if self.excl_check(addr) {
                    let v = self.read_x(rt);
                    if !self.mem_write_sized(mem, addr, nbytes, v) {
                        return Step::Fault { addr, write: true };
                    }
                    self.write_x(rs, 0);
                } else {
                    self.write_x(rs, 1);
                }
                Step::Next
            }
            (0, 1) if size >= 2 => {
                // LDXP/LDAXP/STXP/STLXP: 32- or 64-bit register pairs.
                let nbytes = 1usize << size; // per register
                if load {
                    let Some(a) = mem_read_sized(mem, addr, nbytes) else {
                        return Step::Fault { addr, write: false };
                    };
                    let addr2 = addr.wrapping_add(nbytes as u64);
                    let Some(b) = mem_read_sized(mem, addr2, nbytes) else {
                        return Step::Fault {
                            addr: addr2,
                            write: false,
                        };
                    };
                    self.write_x(rt, a);
                    self.write_x(rt2, b);
                    self.open_monitor(addr);
                } else if self.excl_check(addr) {
                    let (a, b) = (self.read_x(rt), self.read_x(rt2));
                    let mut buf = [0u8; 16];
                    buf[..nbytes].copy_from_slice(&a.to_le_bytes()[..nbytes]);
                    buf[nbytes..2 * nbytes].copy_from_slice(&b.to_le_bytes()[..nbytes]);
                    self.note_store(addr, 0, 2 * nbytes);
                    if let Err(e) = mem.write_trap(addr, &buf[..2 * nbytes]) {
                        return Step::Fault {
                            addr: e.fault_addr(),
                            write: true,
                        };
                    }
                    self.write_x(rs, 0);
                } else {
                    self.write_x(rs, 1);
                }
                Step::Next
            }
            (0, 1) => {
                // CASP/CASPA/CASPL/CASPAL: Rs and Rt must be even, and Rt2
                // all ones (real cores fault other values).
                if rs & 1 != 0 || rt & 1 != 0 || rt2 != 31 {
                    return Step::Illegal;
                }
                let nbytes = if size & 1 == 1 { 8 } else { 4 };
                self.cas_pair(addr, nbytes, rs, rt, mem)
            }
            (1, 0) => {
                if o0 == 0 {
                    return Step::Illegal; // LDLAR/STLLR (LORegions)
                }
                // LDAR/STLR (B/H/W/X).
                if load {
                    self.load_x(addr, size, rt, 0, mem)
                } else {
                    self.store_x(addr, size, rt, mem)
                }
            }
            _ => {
                if rt2 != 31 {
                    return Step::Illegal;
                }
                self.cas_single(addr, 1 << size, rs, rt, mem)
            }
        }
    }

    /// `STLUR*`/`LDAPUR*` (FEAT_LRCPC2): release/acquire-RCpc accesses with
    /// an unscaled signed 9-bit offset (`size 011001 opc 0 imm9 00 Rn Rt`).
    fn exec_rcpc2(&mut self, instr: u32, mem: &mut GuestMemory) -> Step {
        let size = instr >> 30;
        let opc = (instr >> 22) & 3;
        let rt = reg_field(instr, 0);
        let imm9 = sign_extend(u64::from((instr >> 12) & 0x1ff), 9);
        let addr = self.read_sp(reg_field(instr, 5)).wrapping_add(imm9 as u64);
        if opc >= 2 && (size == 3 || (size == 2 && opc == 3)) {
            return Step::Illegal;
        }
        if crosses_granule(addr, 1 << size) {
            return Step::Fault {
                addr,
                write: opc == 0,
            };
        }
        match (size, opc) {
            (_, 0b00) => self.store_x(addr, size, rt, mem),
            (_, 0b01) => self.load_x(addr, size, rt, 0, mem),
            (0b00..=0b10, 0b10) => self.load_x(addr, size, rt, 64, mem),
            (0b00 | 0b01, 0b11) => self.load_x(addr, size, rt, 32, mem),
            _ => Step::Illegal,
        }
    }

    /// Open the exclusive monitor on `addr` (`LDXR`/`LDAXR`/`LDXP`/`LDAXP`).
    fn open_monitor(&mut self, addr: u64) {
        self.excl_monitor = true;
        self.excl_addr = addr;
    }

    /// Consume the exclusive monitor for a store-exclusive to `addr`: whether
    /// the store may proceed.
    fn excl_check(&mut self, addr: u64) -> bool {
        let ok = self.excl_monitor && self.excl_addr == addr;
        self.excl_monitor = false;
        ok
    }

    /// LSE atomic memory operations: `LD<op>`/`ST<op>` and `SWP`
    /// (`size 111000 A R 1 Rs o3 opc 00 Rn Rt`).
    fn exec_atomic(&mut self, instr: u32, mem: &mut GuestMemory) -> Step {
        let nbytes = 1usize << (instr >> 30);
        let rs = reg_field(instr, 16);
        let rn = reg_field(instr, 5);
        let rt = reg_field(instr, 0);
        let o3 = (instr >> 15) & 1;
        let opc = (instr >> 12) & 7;
        let ldapr = o3 == 1 && opc == 0b100 && (instr >> 22) & 3 == 0b10 && rs == 31;
        if o3 == 1 && opc != 0 && !ldapr {
            return Step::Illegal; // LD64B/ST64B, …
        }
        let addr = self.read_sp(rn);
        if crosses_granule(addr, nbytes as u64) {
            return Step::Fault {
                addr,
                write: !ldapr,
            };
        }
        match (o3, opc) {
            (0, _) => self.ld_op(addr, nbytes, rs, rt, opc, mem),
            (1, 0) => self.swp(addr, nbytes, rs, rt, mem),
            // LDAPR/LDAPRB/LDAPRH (FEAT_LRCPC): A=1, R=0, Rs=11111.
            (1, 0b100) if (instr >> 22) & 3 == 0b10 && rs == 31 => {
                self.load_x(addr, instr >> 30, rt, 0, mem)
            }
            _ => Step::Illegal, // LD64B/ST64B, …
        }
    }

    /// Write the low `nbytes` bytes of `val` to `addr` through `note_store`.
    fn mem_write_sized(
        &mut self,
        mem: &mut GuestMemory,
        addr: u64,
        nbytes: usize,
        val: u64,
    ) -> bool {
        self.note_store(addr, u128::from(val), nbytes);
        mem.write_trap(addr, &val.to_le_bytes()[..nbytes]).is_ok()
    }

    /// `CAS*` (B/H/W/X): compare `Rs` with memory and, if equal, store `Rt`.
    /// `Rs` always receives the old value.
    fn cas_single(
        &mut self,
        addr: u64,
        nbytes: usize,
        rs: usize,
        rt: usize,
        mem: &mut GuestMemory,
    ) -> Step {
        let Some(old) = mem_read_sized(mem, addr, nbytes) else {
            return Step::Fault { addr, write: false };
        };
        if old == self.read_x(rs) & ones((nbytes * 8) as u32) {
            let new = self.read_x(rt);
            if !self.mem_write_sized(mem, addr, nbytes, new) {
                return Step::Fault { addr, write: true };
            }
        }
        self.write_x(rs, old);
        Step::Next
    }

    /// `CASP*`: compare-and-swap a register pair (`Rs`/`Rs+1` against
    /// memory, storing `Rt`/`Rt+1` on a match).
    fn cas_pair(
        &mut self,
        addr: u64,
        nbytes: usize,
        rs: usize,
        rt: usize,
        mem: &mut GuestMemory,
    ) -> Step {
        let Some(old0) = mem_read_sized(mem, addr, nbytes) else {
            return Step::Fault { addr, write: false };
        };
        let addr2 = addr.wrapping_add(nbytes as u64);
        let Some(old1) = mem_read_sized(mem, addr2, nbytes) else {
            return Step::Fault {
                addr: addr2,
                write: false,
            };
        };
        let mask = ones((nbytes * 8) as u32);
        if old0 == self.read_x(rs) & mask && old1 == self.read_x(rs + 1) & mask {
            let (new0, new1) = (self.read_x(rt), self.read_x(rt + 1));
            let mut buf = [0u8; 16];
            buf[..nbytes].copy_from_slice(&new0.to_le_bytes()[..nbytes]);
            buf[nbytes..2 * nbytes].copy_from_slice(&new1.to_le_bytes()[..nbytes]);
            self.note_store(addr, 0, 2 * nbytes);
            if let Err(e) = mem.write_trap(addr, &buf[..2 * nbytes]) {
                return Step::Fault {
                    addr: e.fault_addr(),
                    write: true,
                };
            }
        }
        self.write_x(rs, old0);
        self.write_x(rs + 1, old1);
        Step::Next
    }

    /// `SWP*`: store `Rs`, return the old value in `Rt`.
    fn swp(
        &mut self,
        addr: u64,
        nbytes: usize,
        rs: usize,
        rt: usize,
        mem: &mut GuestMemory,
    ) -> Step {
        let Some(old) = mem_read_sized(mem, addr, nbytes) else {
            return Step::Fault { addr, write: false };
        };
        let new = self.read_x(rs);
        if !self.mem_write_sized(mem, addr, nbytes, new) {
            return Step::Fault { addr, write: true };
        }
        self.write_x(rt, old);
        Step::Next
    }

    /// `LD<op>` (and the `ST<op>` aliases, `Rt == 31`): atomically apply `op`
    /// (000 ADD, 001 CLR, 010 EOR, 011 SET, 100 SMAX, 101 SMIN, 110 UMAX,
    /// 111 UMIN) with `Rs`; `Rt` receives the old value.
    fn ld_op(
        &mut self,
        addr: u64,
        nbytes: usize,
        rs: usize,
        rt: usize,
        op: u32,
        mem: &mut GuestMemory,
    ) -> Step {
        let Some(old) = mem_read_sized(mem, addr, nbytes) else {
            return Step::Fault { addr, write: false };
        };
        let bits = (nbytes * 8) as u32;
        let mask = ones(bits);
        let s = self.read_x(rs) & mask;
        let new = match op {
            0b000 => old.wrapping_add(s) & mask,
            0b001 => old & !s,
            0b010 => old ^ s,
            0b011 => old | s,
            0b100 => {
                if sign_extend(old, bits) >= sign_extend(s, bits) {
                    old
                } else {
                    s
                }
            }
            0b101 => {
                if sign_extend(old, bits) <= sign_extend(s, bits) {
                    old
                } else {
                    s
                }
            }
            0b110 => old.max(s),
            _ => old.min(s),
        };
        if !self.mem_write_sized(mem, addr, nbytes, new) {
            return Step::Fault { addr, write: true };
        }
        self.write_x(rt, old);
        Step::Next
    }

    /// Advanced SIMD load/store multiple structures and single structure
    /// (`LD1`–`LD4`, `ST1`–`ST4`, `LD1R`–`LD4R`), with optional post-index.
    fn exec_simd_struct(&mut self, instr: u32, mem: &mut GuestMemory) -> Step {
        if instr >> 31 != 0 {
            return Step::Illegal;
        }
        let q = (instr >> 30) & 1 == 1;
        let post = (instr >> 23) & 1 == 1;
        let single = (instr >> 24) & 1 == 1;
        let load = (instr >> 22) & 1 == 1;
        let r_bit = (instr >> 21) & 1;
        let rm = reg_field(instr, 16);
        let opcode = (instr >> 12) & 0xf;
        let size = (instr >> 10) & 3;
        let rn = reg_field(instr, 5);
        let rt = reg_field(instr, 0);
        // Without post-index, bits 20:16 are zero; multiple-structure forms
        // also have bit 21 clear.
        if (!post && rm != 0) || (!single && r_bit != 0) {
            return Step::Illegal;
        }
        let addr = self.read_sp(rn);
        let mut buf = [0u8; 64];
        let total;
        if single {
            let opc3 = opcode >> 1; // bits 15:13
            let s = opcode & 1; // bit 12
            let selem = (((opc3 & 1) << 1) | r_bit) + 1;
            let mut scale = opc3 >> 1;
            let mut replicate = false;
            let mut index = 0u32;
            match scale {
                3 => {
                    if !load || s != 0 {
                        return Step::Illegal;
                    }
                    scale = size;
                    replicate = true;
                }
                0 => index = (u32::from(q) << 3) | (s << 2) | size,
                1 => {
                    if size & 1 != 0 {
                        return Step::Illegal;
                    }
                    index = (u32::from(q) << 2) | (s << 1) | (size >> 1);
                }
                _ => {
                    if size & 2 != 0 {
                        return Step::Illegal;
                    }
                    if size & 1 == 0 {
                        index = (u32::from(q) << 1) | s;
                    } else {
                        if s != 0 {
                            return Step::Illegal;
                        }
                        index = u32::from(q);
                        scale = 3;
                    }
                }
            }
            let ebytes = 1usize << scale;
            total = ebytes * selem as usize;
            let esize = 8 * ebytes as u32;
            let emask = ones(esize);
            if load {
                if mem.read(addr, &mut buf[..total]).is_err() {
                    return Step::Fault { addr, write: false };
                }
                for s in 0..selem as usize {
                    let mut e = [0u8; 8];
                    e[..ebytes].copy_from_slice(&buf[s * ebytes..(s + 1) * ebytes]);
                    let elem = u128::from(u64::from_le_bytes(e));
                    let t = (rt + s) % 32;
                    if replicate {
                        let mut v = 0u128;
                        let lanes = if q { 128 / esize } else { 64 / esize };
                        for i in 0..lanes {
                            v |= elem << (i * esize);
                        }
                        self.v[t] = v;
                    } else {
                        let sh = index * esize;
                        self.v[t] = (self.v[t] & !(u128::from(emask) << sh)) | (elem << sh);
                    }
                }
            } else {
                for s in 0..selem as usize {
                    let t = (rt + s) % 32;
                    let elem = ((self.v[t] >> (index * esize)) as u64) & emask;
                    buf[s * ebytes..(s + 1) * ebytes]
                        .copy_from_slice(&elem.to_le_bytes()[..ebytes]);
                }
                self.note_store(addr, 0, total);
                if let Err(e) = mem.write_trap(addr, &buf[..total]) {
                    return Step::Fault {
                        addr: e.fault_addr(),
                        write: true,
                    };
                }
            }
        } else {
            let (rpt, selem) = match opcode {
                0b0000 => (1, 4),
                0b0010 => (4, 1),
                0b0100 => (1, 3),
                0b0110 => (3, 1),
                0b0111 => (1, 1),
                0b1000 => (1, 2),
                0b1010 => (2, 1),
                _ => return Step::Illegal,
            };
            if size == 3 && !q && selem != 1 {
                return Step::Illegal;
            }
            let ebytes = 1usize << size;
            let regbytes = if q { 16 } else { 8 };
            let elements = regbytes / ebytes;
            total = rpt * selem * regbytes;
            if load {
                if mem.read(addr, &mut buf[..total]).is_err() {
                    return Step::Fault { addr, write: false };
                }
                // De-interleave: memory element k of structure e goes to
                // register (r*selem + s) lane e.
                let mut regs = [[0u8; 16]; 4];
                let mut off = 0;
                for r in 0..rpt {
                    for e in 0..elements {
                        for s in 0..selem {
                            let reg = r + s; // rpt>1 implies selem==1 and vice versa
                            regs[reg][e * ebytes..(e + 1) * ebytes]
                                .copy_from_slice(&buf[off..off + ebytes]);
                            off += ebytes;
                        }
                    }
                }
                for (i, bytes) in regs.iter().enumerate().take(rpt.max(selem)) {
                    self.v[(rt + i) % 32] = u128::from_le_bytes(*bytes);
                }
            } else {
                let mut off = 0;
                for r in 0..rpt {
                    for e in 0..elements {
                        for s in 0..selem {
                            let reg = (rt + r + s) % 32;
                            let bytes = self.v[reg].to_le_bytes();
                            buf[off..off + ebytes]
                                .copy_from_slice(&bytes[e * ebytes..(e + 1) * ebytes]);
                            off += ebytes;
                        }
                    }
                }
                self.note_store(addr, 0, total);
                if let Err(e) = mem.write_trap(addr, &buf[..total]) {
                    return Step::Fault {
                        addr: e.fault_addr(),
                        write: true,
                    };
                }
            }
        }
        if post {
            let off = if rm == 31 {
                total as u64
            } else {
                self.read_x(rm)
            };
            self.write_sp(rn, addr.wrapping_add(off));
        }
        Step::Next
    }
}

/// Addressing mode of a single-register load/store.
#[derive(Clone, Copy)]
enum Addr {
    /// `[Xn, #imm12 << scale]`
    Unsigned(u64),
    /// `[Xn, #simm9]` (`LDUR`/`STUR`/`PRFUM`)
    Unscaled(i64),
    /// `[Xn, #simm9]` (`LDTR`/`STTR`: unprivileged — ordinary at EL0)
    Unprivileged(i64),
    /// `[Xn, #simm9]!`
    Pre(i64),
    /// `[Xn], #simm9`
    Post(i64),
    /// `[Xn, Rm{, extend {#amount}}]`
    Register,
}

/// Whether an access of `bytes` at `addr` crosses a 16-byte boundary, which
/// faults for ordered/atomic accesses.
#[inline]
fn crosses_granule(addr: u64, bytes: u64) -> bool {
    (addr & 15) + bytes > 16
}

/// Read `nbytes` (1, 2, 4 or 8) little-endian bytes, zero-extended; `None` on
/// a fault.
fn mem_read_sized(mem: &mut GuestMemory, addr: u64, nbytes: usize) -> Option<u64> {
    let mut buf = [0u8; 8];
    mem.read(addr, &mut buf[..nbytes]).ok()?;
    Some(u64::from_le_bytes(buf))
}
