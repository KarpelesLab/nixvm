//! The i386 system-call layer: an `int $0x80` (or a fast 32-bit entry)
//! from any task, and every call of an i386 program.
//!
//! [`crate::abi::arch::i386::entry`] says what each call is; this runs it.
//! A [`Entry::Call`] is its x86-64 counterpart with widened arguments, and
//! the structures it points to converted through a scratch copy below the
//! caller's stack, the way Linux's `compat_alloc_user_space` once did: the
//! x86-64 handler reads and writes that copy, and the converted result goes
//! back to the caller's structure. A call that blocks runs again from the
//! start when it is retried, scratch copy and all. A [`Entry::Compat`] has a
//! translation of its own here: the `socketcall` and `ipc` multiplexers, the
//! old `mmap`s, `msghdr`s and `iovec` arrays, `fd_set`s, `siginfo`, ...

mod calls;

use super::{Kernel, ServiceCtx, err};
use crate::abi::Arch;
use crate::abi::arch::i386::{self, Compat, Dir, Entry, Layout, Ret};

pub(in crate::kernel) use calls::siginfo_narrow;
use crate::abi::arch::{self};
use crate::abi::errno::Errno;
use crate::vcpu::mem::PAGE_SIZE;
use crate::vcpu::{GuestMemory, Vcpu};

/// `__USER32_CS` and `__USER_DS`.
const USER32_CS: u16 = 0x23;
const USER_DS: u16 = 0x2b;

/// Put a freshly loaded i386 program's vcpu in compatibility mode, as
/// Linux's `compat_start_thread`: `%cs` `__USER32_CS`, `%ss`, `%ds` and
/// `%es` `__USER_DS`, `%fs` and `%gs` null.
pub(super) fn start_thread_ia32(vcpu: &mut dyn Vcpu) {
    vcpu.set_user_segments(USER32_CS, USER_DS);
    vcpu.load_data_segment(3, USER_DS);
    vcpu.load_data_segment(0, USER_DS);
}

/// Rewrite the 32-bit vDSO for the entry the memory now says the CPU has
/// (the first program was loaded before its vcpu existed).
pub(super) fn remap_vdso32(mem: &mut GuestMemory) {
    if let Some(base) = mem.vdso32() {
        let image = crate::vcpu::vdso32::build_image(mem.fast32());
        let _ = mem.write_init(base, &image);
    }
}

/// Bytes kept clear below the caller's stack pointer before the scratch
/// area: the x86-64 red zone, for an `int $0x80` from 64-bit code.
const RED_ZONE: u64 = 256;
/// The most a call converts through the scratch area at once.
const SCRATCH_MAX: u64 = 1 << 20;

/// Scratch space below the caller's stack, handed out downwards.
struct Scratch {
    next: u64,
    floor: u64,
}

impl Scratch {
    fn new(sp: u64) -> Self {
        let top = (sp & !0xf).saturating_sub(RED_ZONE);
        Self {
            next: top,
            floor: top.saturating_sub(SCRATCH_MAX),
        }
    }
}

fn r32(b: &[u8], o: usize) -> u64 {
    u64::from(u32::from_le_bytes(b[o..o + 4].try_into().unwrap()))
}
fn r64(b: &[u8], o: usize) -> u64 {
    u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
}
fn w32(b: &mut [u8], o: usize, v: u64) {
    b[o..o + 4].copy_from_slice(&(v as u32).to_le_bytes());
}
fn w64(b: &mut [u8], o: usize, v: u64) {
    b[o..o + 8].copy_from_slice(&v.to_le_bytes());
}
const fn sx(v: u64) -> u64 {
    v as u32 as i32 as i64 as u64
}

/// A fault reading or writing the caller's memory: the call's `EFAULT`.
fn efault<T>(_: T) -> i64 {
    err(Errno::EFAULT)
}

impl Kernel {
    /// Carve `len` bytes (16-byte aligned) of scratch, its pages mapped and
    /// writable (growing the stack into them if need be), or `EFAULT`.
    fn scratch(
        &self,
        cx: &mut ServiceCtx,
        sc: &mut Scratch,
        len: u64,
        mem: &mut GuestMemory,
    ) -> Result<u64, i64> {
        let len = len.max(1).next_multiple_of(16);
        let at = sc.next.checked_sub(len).filter(|&a| a >= sc.floor);
        let Some(at) = at else {
            return Err(err(Errno::EFAULT));
        };
        let mut page = at & !(PAGE_SIZE - 1);
        while page < sc.next {
            if mem.page_prot(page).is_none() {
                self.grow_stack(cx, page, mem);
            }
            if !mem
                .page_prot(page)
                .is_some_and(|p| p.contains(crate::vcpu::Prot::WRITE))
            {
                return Err(err(Errno::EFAULT));
            }
            page += PAGE_SIZE;
        }
        sc.next = at;
        Ok(at)
    }

    /// Scratch holding `bytes`.
    fn scratch_with(
        &self,
        cx: &mut ServiceCtx,
        sc: &mut Scratch,
        bytes: &[u8],
        mem: &mut GuestMemory,
    ) -> Result<u64, i64> {
        let at = self.scratch(cx, sc, bytes.len() as u64, mem)?;
        mem.write(at, bytes).map_err(efault)?;
        Ok(at)
    }

    /// Run x86-64 call `nr`, as the i386 call it translates does: through
    /// the embedder's policy when `check` (a call the entry didn't name).
    fn native(
        &self,
        cx: &mut ServiceCtx,
        nr: u64,
        args: [u64; 6],
        check: bool,
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        if check
            && self.policy.is_active()
            && let Some(r) = self.policy_check(cx, nr, mem)
        {
            return r;
        }
        let sys = arch::decode(Arch::X86_64, nr);
        self.dispatch(cx, sys, nr, &args, vcpu, mem)
    }

    /// Run i386 call `entry` with argument registers `regs` (each zero-
    /// extended). `nested` for one a multiplexer (`socketcall`) made, which
    /// the policy hasn't seen yet.
    pub(super) fn ia32_call(
        &self,
        cx: &mut ServiceCtx,
        entry: Entry,
        regs: [u64; 6],
        nested: bool,
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        match entry {
            Entry::Absent => err(Errno::ENOSYS),
            Entry::Call {
                nr,
                signed,
                args,
                ptrs,
                ret,
            } => {
                let mut a = i386::widen(signed, args, regs);
                let r = if ptrs.is_empty() {
                    self.native(cx, nr, a, nested, vcpu, mem)
                } else {
                    match self.call_with_ptrs(cx, nr, &mut a, ptrs, nested, vcpu, mem) {
                        Ok(r) | Err(r) => r,
                    }
                };
                match ret {
                    Ret::Id16 if r > 0xffff => 65534,
                    _ => r,
                }
            }
            Entry::Compat { how, .. } => self.compat(cx, how, regs, vcpu, mem),
        }
    }

    /// A [`Entry::Call`] whose `ptrs` point to structures to convert.
    #[allow(clippy::too_many_arguments)]
    fn call_with_ptrs(
        &self,
        cx: &mut ServiceCtx,
        nr: u64,
        a: &mut [u64; 6],
        ptrs: &[i386::Ptr],
        nested: bool,
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> Result<i64, i64> {
        let mut sc = Scratch::new(vcpu.sp());
        let mut user = [0u64; 6];
        for p in ptrs {
            let i = usize::from(p.arg);
            user[i] = a[i] & 0xffff_ffff;
            if user[i] == 0 {
                continue;
            }
            let n = usize::from(p.count);
            let size64 = p.layout.size64 * n;
            let native = if matches!(p.dir, Dir::In | Dir::InOut | Dir::Rem) {
                let raw = mem
                    .read_vec(user[i], p.layout.size32 * n)
                    .map_err(efault)?;
                let mut out = Vec::with_capacity(size64);
                for k in 0..n {
                    let s = p.layout.size32;
                    out.extend(p.layout.widen(&raw[k * s..(k + 1) * s]));
                }
                out
            } else {
                vec![0u8; size64]
            };
            a[i] = self.scratch_with(cx, &mut sc, &native, mem)?;
        }
        let mut r = self.native(cx, nr, *a, nested, vcpu, mem);
        if cx.block {
            return Ok(r);
        }
        for p in ptrs {
            let i = usize::from(p.arg);
            let back = match p.dir {
                Dir::In => false,
                Dir::Out | Dir::InOut => r >= 0,
                Dir::Rem => r != err(Errno::EFAULT),
            };
            if user[i] == 0 || !back {
                continue;
            }
            match Self::narrow_back(mem, p.layout, usize::from(p.count), a[i], user[i]) {
                Ok(()) => {}
                Err(e) => r = e,
            }
        }
        Ok(r)
    }

    /// Convert `n` x86-64 structures at `from` to the caller's at `to`.
    fn narrow_back(
        mem: &mut GuestMemory,
        layout: &Layout,
        n: usize,
        from: u64,
        to: u64,
    ) -> Result<(), i64> {
        let raw = mem.read_vec(from, layout.size64 * n).map_err(efault)?;
        let mut out = Vec::with_capacity(layout.size32 * n);
        for k in 0..n {
            let s = layout.size64;
            let one = layout
                .narrow(&raw[k * s..(k + 1) * s])
                .map_err(|_| err(Errno::EOVERFLOW))?;
            out.extend(one);
        }
        mem.write(to, &out).map_err(efault)
    }

    /// Read `n` of the caller's `layout` structures at `at` and put their
    /// x86-64 forms in scratch.
    fn widen_in(
        &self,
        cx: &mut ServiceCtx,
        sc: &mut Scratch,
        layout: &Layout,
        n: usize,
        at: u64,
        mem: &mut GuestMemory,
    ) -> Result<u64, i64> {
        let raw = mem.read_vec(at, layout.size32 * n).map_err(efault)?;
        let mut out = Vec::with_capacity(layout.size64 * n);
        for k in 0..n {
            let s = layout.size32;
            out.extend(layout.widen(&raw[k * s..(k + 1) * s]));
        }
        self.scratch_with(cx, sc, &out, mem)
    }

    /// A call with a translation of its own.
    #[allow(clippy::too_many_lines)]
    fn compat(
        &self,
        cx: &mut ServiceCtx,
        how: Compat,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        let r = match how {
            Compat::Llseek => self.c_llseek(cx, a, vcpu, mem),
            Compat::OldMmap => {
                let Ok(raw) = mem.read_vec(a[0], 24) else {
                    return err(Errno::EFAULT);
                };
                let v: Vec<u64> = (0..6).map(|i| r32(&raw, i * 4)).collect();
                if !v[5].is_multiple_of(PAGE_SIZE) {
                    return err(Errno::EINVAL);
                }
                Ok(self.c_mmap(cx, [v[0], v[1], v[2], v[3], v[4], v[5]], vcpu, mem))
            }
            Compat::Mmap2 => {
                let off = a[5] * PAGE_SIZE;
                Ok(self.c_mmap(cx, [a[0], a[1], a[2], a[3], a[4], off], vcpu, mem))
            }
            Compat::Fcntl { is64 } => self.c_fcntl(cx, a, is64, vcpu, mem),
            Compat::Getdents => self.c_getdents(cx, a, vcpu, mem),
            Compat::Readdir => self.c_readdir(cx, a, vcpu, mem),
            Compat::Statfs64 { fd } => self.c_statfs64(cx, a, fd, vcpu, mem),
            Compat::Olduname | Compat::Oldolduname => {
                self.c_olduname(cx, a, how == Compat::Oldolduname, vcpu, mem)
            }
            Compat::Sysinfo => self.c_sysinfo(cx, a, vcpu, mem),
            Compat::Nice => Ok(self.c_nice(cx, a, vcpu, mem)),
            Compat::Stime => self.c_stime(cx, a, vcpu, mem),
            Compat::Signal => self.c_signal(cx, a, vcpu, mem),
            Compat::Sgetmask | Compat::Ssetmask => self.c_sgetmask(cx, how, a, vcpu, mem),
            Compat::Sigsuspend => self.c_sigsuspend(cx, a, vcpu, mem),
            Compat::Sigprocmask => self.c_sigprocmask(cx, a, vcpu, mem),
            Compat::Sigreturn => Ok(Self::sys_sigreturn_ia32(cx, false, vcpu, mem)),
            Compat::RtSigreturn => Ok(Self::sys_sigreturn_ia32(cx, true, vcpu, mem)),
            Compat::RtSigtimedwait { time64 } => self.c_sigtimedwait(cx, a, time64, vcpu, mem),
            Compat::RtSigqueueinfo => self.c_sigqueueinfo(cx, a, false, vcpu, mem),
            Compat::RtTgsigqueueinfo => self.c_sigqueueinfo(cx, a, true, vcpu, mem),
            Compat::Getgroups16 => self.c_getgroups16(cx, a, vcpu, mem),
            Compat::Setgroups16 => self.c_setgroups16(cx, a, vcpu, mem),
            Compat::OldSelect => {
                let Ok(raw) = mem.read_vec(a[0], 20) else {
                    return err(Errno::EFAULT);
                };
                let v = [
                    r32(&raw, 0),
                    r32(&raw, 4),
                    r32(&raw, 8),
                    r32(&raw, 12),
                    r32(&raw, 16),
                    0,
                ];
                self.c_select(cx, v, None, vcpu, mem)
            }
            Compat::Select => self.c_select(cx, a, None, vcpu, mem),
            Compat::Pselect6 { time64 } => self.c_select(cx, a, Some(time64), vcpu, mem),
            Compat::Socketcall => self.c_socketcall(cx, a, vcpu, mem),
            Compat::Ipc => self.c_ipc(cx, a, vcpu, mem),
            Compat::Futex => self.c_futex(cx, a, vcpu, mem),
            Compat::SchedSetaffinity => self.c_setaffinity(cx, a, vcpu, mem),
            Compat::SchedGetaffinity => self.c_getaffinity(cx, a, vcpu, mem),
            Compat::IoSubmit => self.c_io_submit(cx, a, vcpu, mem),
            Compat::Waitid => self.c_waitid(cx, a, vcpu, mem),
            Compat::SetRobustList => Ok(if a[1] == 12 {
                self.native(cx, 273, [a[0], 24, 0, 0, 0, 0], false, vcpu, mem)
            } else {
                err(Errno::EINVAL)
            }),
            Compat::GetRobustList => self.c_get_robust_list(cx, a, vcpu, mem),
            Compat::Rwv { write } => {
                let nr = if write { 20 } else { 19 };
                self.c_iov_call(cx, nr, a, 1, 2, vcpu, mem)
            }
            Compat::Prwv { write, flags } => {
                let nr = match (write, flags) {
                    (false, false) => 295,
                    (true, false) => 296,
                    (false, true) => 327,
                    (true, true) => 328,
                };
                // (fd, iov, vlen, pos_l, pos_h[, flags]): x86-64 takes the
                // position whole and the high half again (always 0).
                let pos = (a[3] & 0xffff_ffff) | a[4] << 32;
                let b = [a[0], a[1], a[2], pos, 0, a[5]];
                self.c_iov_call(cx, nr, b, 1, 2, vcpu, mem)
            }
            Compat::Vmsplice => self.c_iov_call(cx, 278, a, 1, 2, vcpu, mem),
            Compat::ProcessVm { write } => self.c_process_vm(cx, a, write, vcpu, mem),
            Compat::Sendmsg => self.c_sendmsg(cx, a, vcpu, mem),
            Compat::Recvmsg => self.c_recvmsg(cx, a, vcpu, mem),
            Compat::Sendmmsg => self.c_sendmmsg(cx, a, vcpu, mem),
            Compat::Recvmmsg { time64 } => self.c_recvmmsg(cx, a, time64, vcpu, mem),
            Compat::Getsockopt => self.c_getsockopt(cx, a, vcpu, mem),
            Compat::Setsockopt => self.c_setsockopt(cx, a, vcpu, mem),
            Compat::Semctl => self.c_semctl(cx, a, vcpu, mem),
            Compat::Shmctl => self.c_shmctl(cx, a, vcpu, mem),
            Compat::Shmat => Ok(self.native(cx, 30, [a[0], a[1], sx(a[2]), 0, 0, 0], false, vcpu, mem)),
            Compat::Msgsnd => self.c_msgsnd(cx, a, vcpu, mem),
            Compat::Msgrcv => self.c_msgrcv(cx, a, vcpu, mem),
            Compat::Msgctl => self.c_msgctl(cx, a, vcpu, mem),
            Compat::ArchPrctl => Ok(if (0x1011..=0x1030).contains(&a[0]) {
                self.native(cx, 158, a, false, vcpu, mem)
            } else {
                err(Errno::EINVAL)
            }),
            Compat::Prctl => self.c_prctl(cx, a, vcpu, mem),
            Compat::Seccomp => self.c_seccomp(cx, a, vcpu, mem),
        };
        match r {
            Ok(r) | Err(r) => r,
        }
    }
}
