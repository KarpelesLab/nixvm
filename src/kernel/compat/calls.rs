//! The i386 calls with a translation of their own (see [`super`]).

use super::{Scratch, efault, r32, r64, sx, w32, w64};
use crate::abi::arch::i386::{self, Compat, layouts as l};
use crate::abi::errno::Errno;
use crate::kernel::{Kernel, ServiceCtx, err};
use crate::vcpu::mem::PAGE_SIZE;
use crate::vcpu::{GuestMemory, Vcpu};

type R = Result<i64, i64>;

/// `MSG_CTRUNC`, `MSG_DONTWAIT`.
const MSG_CTRUNC: u64 = 0x8;
const SOL_SOCKET: u64 = 1;
const SCM_RIGHTS: u64 = 1;
/// `SCM_PIDFD`: one descriptor, the sender's pidfd (`SO_PASSPIDFD`).
const SCM_PIDFD: u64 = 4;
/// `IPC_64`: the caller uses the `*64_ds` structures.
const IPC_64: u64 = 0x100;

/// The x86-64 `siginfo_t` from the i386 one (`copy_siginfo_from_user32`):
/// the three `int`s, then the union, which moves from offset 12 to 16 and
/// widens its pointers and `long`s.
pub(in crate::kernel) fn siginfo_widen(c: &[u8]) -> [u8; 128] {
    let mut n = [0u8; 128];
    n[..12].copy_from_slice(&c[..12]);
    let (signo, code) = (r32(c, 0) as i32, r32(c, 8) as i32);
    let u = |o: usize| r32(c, 12 + o);
    match layout(signo, code) {
        Sil::Kill => {
            w32(&mut n, 16, u(0));
            w32(&mut n, 20, u(4));
        }
        Sil::Timer | Sil::Rt => {
            w32(&mut n, 16, u(0));
            w32(&mut n, 20, u(4));
            w64(&mut n, 24, u(8));
        }
        Sil::Chld => {
            w32(&mut n, 16, u(0));
            w32(&mut n, 20, u(4));
            w32(&mut n, 24, u(8));
            w64(&mut n, 32, sx(u(12)));
            w64(&mut n, 40, sx(u(16)));
        }
        Sil::Fault => {
            w64(&mut n, 16, u(0));
            n[24..26].copy_from_slice(&c[16..18]);
        }
        Sil::Poll => {
            w64(&mut n, 16, sx(u(0)));
            w32(&mut n, 24, u(4));
        }
        Sil::Sys => {
            w64(&mut n, 16, u(0));
            w32(&mut n, 24, u(4));
            w32(&mut n, 28, u(8));
        }
    }
    n
}

/// The i386 `siginfo_t` from the x86-64 one (`copy_siginfo_to_user32`).
pub(in crate::kernel) fn siginfo_narrow(n: &[u8]) -> [u8; 128] {
    let mut c = [0u8; 128];
    c[..12].copy_from_slice(&n[..12]);
    let (signo, code) = (r32(n, 0) as i32, r32(n, 8) as i32);
    match layout(signo, code) {
        Sil::Kill => {
            w32(&mut c, 12, r32(n, 16));
            w32(&mut c, 16, r32(n, 20));
        }
        Sil::Timer | Sil::Rt => {
            w32(&mut c, 12, r32(n, 16));
            w32(&mut c, 16, r32(n, 20));
            w32(&mut c, 20, r64(n, 24));
        }
        Sil::Chld => {
            w32(&mut c, 12, r32(n, 16));
            w32(&mut c, 16, r32(n, 20));
            w32(&mut c, 20, r32(n, 24));
            w32(&mut c, 24, r64(n, 32));
            w32(&mut c, 28, r64(n, 40));
        }
        Sil::Fault => {
            w32(&mut c, 12, r64(n, 16));
            c[16..18].copy_from_slice(&n[24..26]);
        }
        Sil::Poll => {
            w32(&mut c, 12, r64(n, 16));
            w32(&mut c, 16, r32(n, 24));
        }
        Sil::Sys => {
            w32(&mut c, 12, r64(n, 16));
            w32(&mut c, 16, r32(n, 24));
            w32(&mut c, 20, r32(n, 28));
        }
    }
    c
}

/// Which member of the `siginfo` union a signal and code use
/// (`siginfo_layout`).
enum Sil {
    Kill,
    Timer,
    Rt,
    Chld,
    Fault,
    Poll,
    Sys,
}

fn layout(signo: i32, code: i32) -> Sil {
    const SI_TIMER: i32 = -2;
    const SI_SIGIO: i32 = -5;
    if code == SI_TIMER {
        return Sil::Timer;
    }
    if code == SI_SIGIO {
        return Sil::Poll;
    }
    if code < 0 {
        return Sil::Rt;
    }
    // `SI_USER`, `SI_KERNEL` and above (a ptrace trap's `SIGTRAP | 0x80`,
    // its event codes) are the kill layout; a code past the signal's own
    // (`NSIGILL`, ...) is the poll one up to `NSIGPOLL`.
    if code == 0 || code >= 0x80 {
        return Sil::Kill;
    }
    let (limit, sil) = match signo {
        4 => (11, Sil::Fault),  // SIGILL
        5 => (6, Sil::Fault),   // SIGTRAP
        7 => (5, Sil::Fault),   // SIGBUS
        8 => (15, Sil::Fault),  // SIGFPE
        11 => (10, Sil::Fault), // SIGSEGV
        17 => (6, Sil::Chld),
        29 => (6, Sil::Poll),
        31 => (2, Sil::Sys),
        _ => (0, Sil::Kill),
    };
    if code <= limit {
        sil
    } else if code <= 6 {
        Sil::Poll
    } else {
        Sil::Kill
    }
}

/// What converting a `msghdr` set up: where the caller's is, and the x86-64
/// one's control buffer, for the way back.
struct Msg {
    user: u64,
    native: u64,
    control32: u64,
    controllen32: u64,
    control64: u64,
}

impl Kernel {
    pub(super) fn c_llseek(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let off = (a[1] << 32) | (a[2] & 0xffff_ffff);
        let r = self.native(cx, 8, [a[0], off, a[4], 0, 0, 0], false, vcpu, mem);
        if r < 0 {
            return Ok(r);
        }
        mem.write_u64(a[3], r as u64).map_err(efault)?;
        Ok(0)
    }

    /// `old_mmap` and `mmap2`, whose offset is in bytes here. Where the
    /// mapping goes is the x86-64 `mmap`'s business: below 4 GiB for an
    /// i386 call ([`Kernel::place`]), never past an i386 program's 4 GiB.
    pub(super) fn c_mmap(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        self.native(cx, 9, a, false, vcpu, mem)
    }

    pub(super) fn c_fcntl(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        is64: bool,
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let (fd, cmd, arg) = (a[0], a[1], a[2]);
        let (ncmd, layout) = match cmd {
            5..=7 => (cmd, &l::FLOCK),
            12..=14 if is64 => (cmd - 7, &l::FLOCK64),
            36..=38 if is64 => (cmd, &l::FLOCK64),
            12..=14 | 36..=38 => return Ok(err(Errno::EINVAL)),
            // The `int` arguments: F_DUPFD, F_SETFD, F_SETFL, F_SETOWN,
            // F_SETSIG, F_SETLEASE, F_NOTIFY, F_DUPFD_CLOEXEC,
            // F_SETPIPE_SZ, F_ADD_SEALS.
            0 | 2 | 4 | 8 | 10 | 1024 | 1026 | 1030 | 1031 | 1033 => {
                return Ok(self.native(cx, 72, [fd, cmd, sx(arg), 0, 0, 0], false, vcpu, mem));
            }
            _ => return Ok(self.native(cx, 72, [fd, cmd, arg, 0, 0, 0], false, vcpu, mem)),
        };
        let mut sc = Scratch::new(vcpu.sp());
        let p = self.widen_in(cx, &mut sc, layout, 1, arg, mem)?;
        let r = self.native(cx, 72, [fd, ncmd, p, 0, 0, 0], false, vcpu, mem);
        if r >= 0 && !cx.block && matches!(ncmd, 5 | 36) {
            Self::narrow_back(mem, layout, 1, p, arg)?;
        }
        Ok(r)
    }

    pub(super) fn c_getdents(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let (fd, dirp, count) = (a[0], a[1], a[2].min(1 << 18));
        let mut sc = Scratch::new(vcpu.sp());
        let buf = self.scratch(cx, &mut sc, count, mem)?;
        let r = self.native(cx, 78, [fd, buf, count, 0, 0, 0], false, vcpu, mem);
        if r <= 0 {
            return Ok(r);
        }
        let raw = mem.read_vec(buf, r as usize).map_err(efault)?;
        let mut out = Vec::new();
        let mut off = 0;
        let mut last_off = None;
        while off + 19 <= raw.len() {
            let ino = r64(&raw, off);
            let d_off = r64(&raw, off + 8);
            let reclen = usize::from(u16::from_le_bytes([raw[off + 16], raw[off + 17]]));
            if reclen < 20 || off + reclen > raw.len() {
                break;
            }
            let name = &raw[off + 18..off + reclen - 1];
            let name = &name[..name.iter().position(|&b| b == 0).unwrap_or(name.len())];
            let d_type = raw[off + reclen - 1];
            if ino > u64::from(u32::MAX) {
                if out.is_empty() {
                    return Ok(err(Errno::EOVERFLOW));
                }
                break;
            }
            let rl = (10 + name.len() + 2).next_multiple_of(4);
            let start = out.len();
            out.resize(start + rl, 0);
            w32(&mut out, start, ino);
            w32(&mut out, start + 4, d_off);
            out[start + 8..start + 10].copy_from_slice(&(rl as u16).to_le_bytes());
            out[start + 10..start + 10 + name.len()].copy_from_slice(name);
            out[start + rl - 1] = d_type;
            last_off = Some(d_off);
            off += reclen;
        }
        // Entries read past the last one handed back are read again.
        if off < raw.len()
            && let Some(pos) = last_off
        {
            self.native(cx, 8, [fd, pos, 0, 0, 0, 0], false, vcpu, mem);
        }
        mem.write(dirp, &out).map_err(efault)?;
        Ok(out.len() as i64)
    }

    /// `readdir`: one `old_linux_dirent`, then the position is past it.
    pub(super) fn c_readdir(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let mut sc = Scratch::new(vcpu.sp());
        let buf = self.scratch(cx, &mut sc, 4096, mem)?;
        let r = self.native(cx, 217, [a[0], buf, 4096, 0, 0, 0], true, vcpu, mem);
        if r <= 0 {
            return Ok(r);
        }
        let raw = mem.read_vec(buf, r as usize).map_err(efault)?;
        let (ino, d_off) = (r64(&raw, 0), r64(&raw, 8));
        let reclen = usize::from(u16::from_le_bytes([raw[16], raw[17]])).min(raw.len());
        let name = &raw[19..reclen];
        let name = &name[..name.iter().position(|&b| b == 0).unwrap_or(name.len())];
        self.native(cx, 8, [a[0], d_off, 0, 0, 0, 0], false, vcpu, mem);
        if ino > u64::from(u32::MAX) {
            return Ok(err(Errno::EOVERFLOW));
        }
        let mut out = vec![0u8; 10 + name.len() + 1];
        w32(&mut out, 0, ino);
        w32(&mut out, 4, d_off);
        out[8..10].copy_from_slice(&(name.len() as u16).to_le_bytes());
        out[10..10 + name.len()].copy_from_slice(name);
        mem.write(a[1], &out).map_err(efault)?;
        Ok(1)
    }

    pub(super) fn c_statfs64(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        fd: bool,
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        if a[1] != l::STATFS64.size32 as u64 {
            return Ok(err(Errno::EINVAL));
        }
        let mut sc = Scratch::new(vcpu.sp());
        let buf = self.scratch(cx, &mut sc, 120, mem)?;
        let nr = if fd { 138 } else { 137 };
        let r = self.native(cx, nr, [a[0], buf, 0, 0, 0, 0], false, vcpu, mem);
        if r >= 0 {
            Self::narrow_back(mem, &l::STATFS64, 1, buf, a[2])?;
        }
        Ok(r)
    }

    /// `olduname` (`struct old_utsname`: the first five fields) and
    /// `oldolduname` (five 9-byte fields).
    pub(super) fn c_olduname(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        oldold: bool,
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let mut sc = Scratch::new(vcpu.sp());
        let buf = self.scratch(cx, &mut sc, 390, mem)?;
        let r = self.native(cx, 63, [buf, 0, 0, 0, 0, 0], false, vcpu, mem);
        if r < 0 {
            return Ok(r);
        }
        let raw = mem.read_vec(buf, 390).map_err(efault)?;
        let out = if oldold {
            let mut o = vec![0u8; 45];
            for i in 0..5 {
                o[i * 9..i * 9 + 8].copy_from_slice(&raw[i * 65..i * 65 + 8]);
            }
            o
        } else {
            raw[..325].to_vec()
        };
        mem.write(a[0], &out).map_err(efault)?;
        Ok(0)
    }

    /// `sysinfo`, its memory figures scaled up a unit when they don't fit
    /// 32 bits (`compat_sys_sysinfo`).
    pub(super) fn c_sysinfo(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let mut sc = Scratch::new(vcpu.sp());
        let buf = self.scratch(cx, &mut sc, 112, mem)?;
        let r = self.native(cx, 99, [buf, 0, 0, 0, 0, 0], false, vcpu, mem);
        if r < 0 {
            return Ok(r);
        }
        let n = mem.read_vec(buf, 112).map_err(efault)?;
        let mut unit = r32(&n, 104).max(1);
        let mut shift = 0;
        if (r64(&n, 32) | r64(&n, 64)) >> 32 != 0 {
            while unit < PAGE_SIZE {
                unit <<= 1;
                shift += 1;
            }
        }
        let mut c = vec![0u8; 64];
        w32(&mut c, 0, r64(&n, 0));
        for i in 0..3 {
            w32(&mut c, 4 + i * 4, r64(&n, 8 + i * 8));
        }
        for (o32, o64) in [(16, 32), (20, 40), (24, 48), (28, 56), (32, 64), (36, 72)] {
            w32(&mut c, o32, r64(&n, o64) >> shift);
        }
        c[40..42].copy_from_slice(&n[80..82]);
        w32(&mut c, 44, r64(&n, 88) >> shift);
        w32(&mut c, 48, r64(&n, 96) >> shift);
        w32(&mut c, 52, unit);
        mem.write(a[0], &c).map_err(efault)?;
        Ok(0)
    }

    /// `nice(inc)`, as `getpriority`/`setpriority` of the caller.
    pub(super) fn c_nice(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        let inc = (sx(a[0]) as i64).clamp(-40, 40);
        let cur = self.native(cx, 140, [0; 6], true, vcpu, mem);
        if cur < 0 {
            return cur;
        }
        let nice = (20 - cur + inc).clamp(-20, 19);
        let r = self.native(cx, 141, [0, 0, nice as u64, 0, 0, 0], true, vcpu, mem);
        if r == err(Errno::EACCES) {
            err(Errno::EPERM)
        } else {
            r
        }
    }

    pub(super) fn c_stime(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let t = mem.read_u32(a[0]).map_err(efault)?;
        let mut tv = [0u8; 16];
        w64(&mut tv, 0, sx(u64::from(t)));
        let mut sc = Scratch::new(vcpu.sp());
        let p = self.scratch_with(cx, &mut sc, &tv, mem)?;
        Ok(self.native(cx, 164, [p, 0, 0, 0, 0, 0], false, vcpu, mem))
    }

    /// `signal(sig, handler)`: a one-shot, non-masking handler, and the old
    /// one back (`SA_ONESHOT | SA_NOMASK`).
    pub(super) fn c_signal(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let mut act = [0u8; 32];
        w64(&mut act, 0, a[1]);
        w64(&mut act, 8, 0x8000_0000 | 0x4000_0000);
        let mut sc = Scratch::new(vcpu.sp());
        let ap = self.scratch_with(cx, &mut sc, &act, mem)?;
        let op = self.scratch(cx, &mut sc, 32, mem)?;
        let r = self.native(cx, 13, [sx(a[0]), ap, op, 8, 0, 0], false, vcpu, mem);
        if r < 0 {
            return Ok(r);
        }
        Ok(i64::from(mem.read_u64(op).map_err(efault)? as u32))
    }

    /// `sgetmask()` / `ssetmask(mask)`: the first 32 signals' mask.
    pub(super) fn c_sgetmask(
        &self,
        cx: &mut ServiceCtx,
        how: Compat,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let mut sc = Scratch::new(vcpu.sp());
        let old = self.scratch(cx, &mut sc, 8, mem)?;
        let r = if how == Compat::Ssetmask {
            let new = self.scratch_with(cx, &mut sc, &(a[0] & 0xffff_ffff).to_le_bytes(), mem)?;
            self.native(cx, 14, [2, new, old, 8, 0, 0], false, vcpu, mem)
        } else {
            self.native(cx, 14, [0, 0, old, 8, 0, 0], false, vcpu, mem)
        };
        if r < 0 {
            return Ok(r);
        }
        Ok(i64::from(mem.read_u32(old).map_err(efault)?))
    }

    /// `sigsuspend(_, _, mask)` (x86's three-argument one).
    pub(super) fn c_sigsuspend(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let mut sc = Scratch::new(vcpu.sp());
        let set = self.scratch_with(cx, &mut sc, &(a[2] & 0xffff_ffff).to_le_bytes(), mem)?;
        Ok(self.native(cx, 130, [set, 8, 0, 0, 0, 0], false, vcpu, mem))
    }

    /// `sigprocmask(how, set, oset)` on the first 32 signals; `SIG_SETMASK`
    /// leaves the rest as they were (`compat_sig_setmask`).
    pub(super) fn c_sigprocmask(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let mut sc = Scratch::new(vcpu.sp());
        let cur = self.scratch(cx, &mut sc, 8, mem)?;
        let r = self.native(cx, 14, [0, 0, cur, 8, 0, 0], false, vcpu, mem);
        if r < 0 {
            return Ok(r);
        }
        let blocked = mem.read_u64(cur).map_err(efault)?;
        if a[1] != 0 {
            let set = u64::from(mem.read_u32(a[1]).map_err(efault)?);
            let new = match a[0] {
                0 => blocked | set,
                1 => blocked & !set,
                2 => (blocked & !0xffff_ffff) | set,
                _ => return Ok(err(Errno::EINVAL)),
            };
            let p = self.scratch_with(cx, &mut sc, &new.to_le_bytes(), mem)?;
            let r = self.native(cx, 14, [2, p, 0, 8, 0, 0], false, vcpu, mem);
            if r < 0 {
                return Ok(r);
            }
        }
        if a[2] != 0 {
            mem.write(a[2], &(blocked as u32).to_le_bytes())
                .map_err(efault)?;
        }
        Ok(0)
    }

    pub(super) fn c_sigtimedwait(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        time64: bool,
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let mut sc = Scratch::new(vcpu.sp());
        let info = if a[1] == 0 {
            0
        } else {
            self.scratch_with(cx, &mut sc, &[0u8; 128], mem)?
        };
        let ts = if a[2] == 0 || time64 {
            a[2]
        } else {
            self.widen_in(cx, &mut sc, &l::TIMESPEC, 1, a[2], mem)?
        };
        let r = self.native(cx, 128, [a[0], info, ts, a[3], 0, 0], false, vcpu, mem);
        if r > 0 && !cx.block && info != 0 {
            let n = mem.read_vec(info, 128).map_err(efault)?;
            mem.write(a[1], &siginfo_narrow(&n)).map_err(efault)?;
        }
        Ok(r)
    }

    /// `rt_sigqueueinfo(pid, sig, info)` / `rt_tgsigqueueinfo(tgid, tid,
    /// sig, info)`.
    pub(super) fn c_sigqueueinfo(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        tg: bool,
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let at = if tg { 3 } else { 2 };
        let c = mem.read_vec(a[at], 128).map_err(efault)?;
        let mut sc = Scratch::new(vcpu.sp());
        let p = self.scratch_with(cx, &mut sc, &siginfo_widen(&c), mem)?;
        let mut b = [sx(a[0]), sx(a[1]), sx(a[2]), 0, 0, 0];
        b[at] = p;
        Ok(self.native(cx, if tg { 297 } else { 129 }, b, false, vcpu, mem))
    }

    pub(super) fn c_getgroups16(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let n = sx(a[0]) as i64;
        if !(0..=65536).contains(&n) {
            return Ok(err(Errno::EINVAL));
        }
        let mut sc = Scratch::new(vcpu.sp());
        let buf = self.scratch(cx, &mut sc, n as u64 * 4, mem)?;
        let r = self.native(cx, 115, [n as u64, buf, 0, 0, 0, 0], false, vcpu, mem);
        if r <= 0 || n == 0 {
            return Ok(r);
        }
        let raw = mem.read_vec(buf, r as usize * 4).map_err(efault)?;
        let out: Vec<u8> = raw
            .chunks(4)
            .flat_map(|g| {
                let v = r32(g, 0);
                (if v > 0xffff { 65534 } else { v as u16 }).to_le_bytes()
            })
            .collect();
        mem.write(a[1], &out).map_err(efault)?;
        Ok(r)
    }

    pub(super) fn c_setgroups16(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let n = sx(a[0]) as i64;
        if !(0..=65536).contains(&n) {
            return Ok(err(Errno::EINVAL));
        }
        let raw = mem.read_vec(a[1], n as usize * 2).map_err(efault)?;
        let wide: Vec<u8> = raw
            .chunks(2)
            .flat_map(|g| {
                let v = u16::from_le_bytes([g[0], g[1]]);
                (if v == 0xffff { u32::MAX } else { u32::from(v) }).to_le_bytes()
            })
            .collect();
        let mut sc = Scratch::new(vcpu.sp());
        let p = self.scratch_with(cx, &mut sc, &wide, mem)?;
        Ok(self.native(cx, 116, [n as u64, p, 0, 0, 0, 0], false, vcpu, mem))
    }

    /// `select` (`time64` `None`: a `timeval`) and `pselect6` (a `timespec`,
    /// 32-bit unless `time64`, and the signal mask's pointer-and-size): the
    /// `fd_set`s are 32-bit words here, 64-bit ones there.
    pub(super) fn c_select(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        time64: Option<bool>,
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let n = sx(a[0]) as i64;
        if n < 0 {
            return Ok(err(Errno::EINVAL));
        }
        let n = n.min(1 << 20) as u64;
        let (b32, b64) = (n.div_ceil(32) * 4, n.div_ceil(64) * 8);
        let mut sc = Scratch::new(vcpu.sp());
        let mut b = [n, 0, 0, 0, 0, 0];
        for i in 1..=3 {
            if a[i] != 0 {
                let mut set = mem.read_vec(a[i], b32 as usize).map_err(efault)?;
                set.resize(b64 as usize, 0);
                b[i] = self.scratch_with(cx, &mut sc, &set, mem)?;
            }
        }
        let convert_ts = a[4] != 0 && time64 != Some(true);
        b[4] = if convert_ts {
            self.widen_in(cx, &mut sc, &l::TIMESPEC, 1, a[4], mem)?
        } else {
            a[4]
        };
        if time64.is_some() && a[5] != 0 {
            b[5] = self.widen_in(cx, &mut sc, &l::PTR_SIZE, 1, a[5], mem)?;
        }
        let nr = if time64.is_some() { 270 } else { 23 };
        let r = self.native(cx, nr, b, false, vcpu, mem);
        if cx.block {
            return Ok(r);
        }
        if r >= 0 {
            for i in 1..=3 {
                if a[i] != 0 {
                    let set = mem.read_vec(b[i], b32 as usize).map_err(efault)?;
                    mem.write(a[i], &set).map_err(efault)?;
                }
            }
        }
        if convert_ts && r != err(Errno::EFAULT) {
            Self::narrow_back(mem, &l::TIMESPEC, 1, b[4], a[4])?;
        }
        Ok(r)
    }

    /// `socketcall(call, args)`: the socket call `call` with its arguments
    /// in an array of 32-bit words, as the i386 call of its own it is.
    pub(super) fn c_socketcall(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        const NARGS: [u8; 21] = [
            0, 3, 3, 3, 2, 3, 3, 3, 4, 4, 4, 6, 6, 2, 5, 5, 3, 3, 4, 5, 4,
        ];
        let call = a[0];
        if !(1..=20).contains(&call) {
            return Ok(err(Errno::EINVAL));
        }
        let n = usize::from(NARGS[call as usize]);
        let raw = mem.read_vec(a[1], n * 4).map_err(efault)?;
        let mut regs = [0u64; 6];
        for (i, r) in regs.iter_mut().enumerate().take(n) {
            *r = r32(&raw, i * 4);
        }
        // accept, send and recv are accept4, sendto and recvfrom.
        let nr32 = match call {
            1 => 359,
            2 => 361,
            3 => 362,
            4 => 363,
            5 | 18 => 364,
            6 => 367,
            7 => 368,
            8 => 360,
            9 | 11 => 369,
            10 | 12 => 371,
            13 => 373,
            14 => 366,
            15 => 365,
            16 => 370,
            17 => 372,
            19 => 337,
            _ => 345,
        };
        let entry = i386::entry(nr32);
        Ok(self.ia32_call(cx, entry, regs, true, vcpu, mem))
    }

    /// `ipc(call, first, second, third, ptr, fifth)`: the SysV IPC call
    /// `call` (its low 16 bits; the high ones are a version).
    pub(super) fn c_ipc(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let (version, call) = (a[0] >> 16, a[0] & 0xffff);
        let (first, second, third, ptr, fifth) = (a[1], a[2], a[3], a[4], a[5]);
        let ent = |nr32| i386::entry(nr32);
        let regs = |v: [u64; 6]| v;
        Ok(match call {
            // semop: semtimedop without a timeout.
            1 => self.native(cx, 65, [sx(first), ptr, second, 0, 0, 0], true, vcpu, mem),
            2 => self.native(
                cx,
                64,
                [sx(first), sx(second), sx(third), 0, 0, 0],
                true,
                vcpu,
                mem,
            ),
            3 => {
                let arg = if ptr == 0 {
                    0
                } else {
                    u64::from(mem.read_u32(ptr).map_err(efault)?)
                };
                return self.c_semctl(cx, [first, second, third, arg, 0, 0], vcpu, mem);
            }
            4 => {
                let e = i386::Entry::Call {
                    nr: 220,
                    signed: 0b1,
                    args: None,
                    ptrs: const {
                        &[i386::Ptr {
                            arg: 3,
                            dir: i386::Dir::In,
                            layout: &l::TIMESPEC,
                            count: 1,
                        }]
                    },
                    ret: i386::Ret::Plain,
                };
                self.ia32_call(
                    cx,
                    e,
                    regs([first, ptr, second, fifth, 0, 0]),
                    true,
                    vcpu,
                    mem,
                )
            }
            11 => return self.c_msgsnd(cx, [first, ptr, second, third, 0, 0], vcpu, mem),
            12 => {
                let (msgp, msgtyp) = if version == 0 {
                    let k = mem.read_vec(ptr, 8).map_err(efault)?;
                    (r32(&k, 0), r32(&k, 4))
                } else {
                    (ptr, fifth)
                };
                return self.c_msgrcv(cx, [first, msgp, second, msgtyp, third, 0], vcpu, mem);
            }
            13 => self.native(cx, 68, [sx(first), sx(second), 0, 0, 0, 0], true, vcpu, mem),
            14 => return self.c_msgctl(cx, [first, second, ptr, 0, 0, 0], vcpu, mem),
            21 => {
                if version == 1 {
                    return Ok(err(Errno::EINVAL));
                }
                let r = self.ia32_call(
                    cx,
                    ent(397),
                    regs([first, ptr, second, 0, 0, 0]),
                    true,
                    vcpu,
                    mem,
                );
                if r < 0 {
                    return Ok(r);
                }
                mem.write(third, &(r as u32).to_le_bytes())
                    .map_err(efault)?;
                0
            }
            22 => self.native(cx, 67, [ptr, 0, 0, 0, 0, 0], true, vcpu, mem),
            23 => self.native(
                cx,
                29,
                [sx(first), second, sx(third), 0, 0, 0],
                true,
                vcpu,
                mem,
            ),
            24 => return self.c_shmctl(cx, [first, second, ptr, 0, 0, 0], vcpu, mem),
            _ => err(Errno::ENOSYS),
        })
    }

    /// `semctl(semid, semnum, cmd, arg)`, `arg` a 32-bit `union semun`.
    pub(super) fn c_semctl(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let (id, num, cmd, arg) = (sx(a[0]), sx(a[1]), a[2] & 0xffff_ffff, a[3] & 0xffff_ffff);
        let base = cmd & !IPC_64;
        let stat = matches!(base, 2 | 18 | 20); // IPC_STAT, SEM_STAT, SEM_STAT_ANY
        let set = base == 1; // IPC_SET
        if !(stat || set) {
            // SETVAL's is an int; the rest are pointers or ignored.
            let arg = if base == 16 { sx(arg) } else { arg };
            return Ok(self.ipc_ctl(cx, 66, [id, num, base, arg], vcpu, mem));
        }
        if cmd & IPC_64 == 0 {
            return Ok(err(Errno::EINVAL));
        }
        let mut sc = Scratch::new(vcpu.sp());
        let buf = if set {
            self.widen_in(cx, &mut sc, &l::SEMID64, 1, arg, mem)?
        } else {
            self.scratch_with(cx, &mut sc, &[0u8; 104], mem)?
        };
        let r = self.ipc_ctl(cx, 66, [id, num, base, buf], vcpu, mem);
        if r >= 0 && stat {
            Self::narrow_back(mem, &l::SEMID64, 1, buf, arg)?;
        }
        Ok(r)
    }

    /// A `*ctl` call made for the i386 one, so through the policy.
    fn ipc_ctl(
        &self,
        cx: &mut ServiceCtx,
        nr: u64,
        a: [u64; 4],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        self.native(cx, nr, [a[0], a[1], a[2], a[3], 0, 0], true, vcpu, mem)
    }

    pub(super) fn c_shmctl(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let (id, cmd, buf) = (sx(a[0]), a[1] & 0xffff_ffff, a[2] & 0xffff_ffff);
        let base = cmd & !IPC_64;
        let layout = match base {
            1 | 2 | 13 | 15 => &l::SHMID64, // IPC_SET, IPC_STAT, SHM_STAT(_ANY)
            3 => &l::SHMINFO64,             // IPC_INFO
            14 => &l::SHM_INFO,             // SHM_INFO
            _ => return Ok(self.ipc_ctl(cx, 31, [id, base, buf, 0], vcpu, mem)),
        };
        if cmd & IPC_64 == 0 && base != 14 {
            return Ok(err(Errno::EINVAL));
        }
        let mut sc = Scratch::new(vcpu.sp());
        let p = if base == 1 {
            self.widen_in(cx, &mut sc, layout, 1, buf, mem)?
        } else {
            self.scratch_with(cx, &mut sc, &vec![0u8; layout.size64], mem)?
        };
        let r = self.ipc_ctl(cx, 31, [id, base, p, 0], vcpu, mem);
        if r >= 0 && base != 1 {
            Self::narrow_back(mem, layout, 1, p, buf)?;
        }
        Ok(r)
    }

    pub(super) fn c_msgctl(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let (id, cmd, buf) = (sx(a[0]), a[1] & 0xffff_ffff, a[2] & 0xffff_ffff);
        let base = cmd & !IPC_64;
        if !matches!(base, 1 | 2 | 11 | 13) {
            // IPC_RMID; IPC_INFO and MSG_INFO's `struct msginfo` is `int`s.
            return Ok(self.ipc_ctl(cx, 71, [id, base, buf, 0], vcpu, mem));
        }
        if cmd & IPC_64 == 0 {
            return Ok(err(Errno::EINVAL));
        }
        let mut sc = Scratch::new(vcpu.sp());
        let p = if base == 1 {
            self.widen_in(cx, &mut sc, &l::MSQID64, 1, buf, mem)?
        } else {
            self.scratch_with(cx, &mut sc, &[0u8; 120], mem)?
        };
        let r = self.ipc_ctl(cx, 71, [id, base, p, 0], vcpu, mem);
        if r >= 0 && base != 1 {
            Self::narrow_back(mem, &l::MSQID64, 1, p, buf)?;
        }
        Ok(r)
    }

    /// `msgsnd(id, msgp, size, flags)`: the message's `long mtype` is 32-bit.
    pub(super) fn c_msgsnd(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let size = sx(a[2]) as i64;
        if !(0..=1 << 20).contains(&size) {
            return Ok(err(Errno::EINVAL));
        }
        let raw = mem.read_vec(a[1], 4 + size as usize).map_err(efault)?;
        let mut m = vec![0u8; 8 + size as usize];
        w64(&mut m, 0, sx(r32(&raw, 0)));
        m[8..].copy_from_slice(&raw[4..]);
        let mut sc = Scratch::new(vcpu.sp());
        let p = self.scratch_with(cx, &mut sc, &m, mem)?;
        Ok(self.native(
            cx,
            69,
            [sx(a[0]), p, size as u64, sx(a[3]), 0, 0],
            true,
            vcpu,
            mem,
        ))
    }

    /// `msgrcv(id, msgp, size, type, flags)`.
    pub(super) fn c_msgrcv(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let size = sx(a[2]) as i64;
        if !(0..=1 << 20).contains(&size) {
            return Ok(err(Errno::EINVAL));
        }
        let mut sc = Scratch::new(vcpu.sp());
        let p = self.scratch(cx, &mut sc, 8 + size as u64, mem)?;
        let b = [sx(a[0]), p, size as u64, sx(a[3]), sx(a[4]), 0];
        let r = self.native(cx, 70, b, true, vcpu, mem);
        if r < 0 || cx.block {
            return Ok(r);
        }
        let m = mem.read_vec(p, 8 + r as usize).map_err(efault)?;
        let mut out = vec![0u8; 4 + r as usize];
        w32(&mut out, 0, r64(&m, 0));
        out[4..].copy_from_slice(&m[8..]);
        mem.write(a[1], &out).map_err(efault)?;
        Ok(r)
    }

    pub(super) fn c_futex(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        // FUTEX_WAIT, LOCK_PI, WAIT_BITSET, WAIT_REQUEUE_PI, LOCK_PI2 take a
        // timeout; the others an `int` there (`val2`).
        let cmd = a[1] & 0x7f;
        let mut b = a;
        if matches!(cmd, 0 | 6 | 9 | 11 | 13) && a[3] != 0 {
            let mut sc = Scratch::new(vcpu.sp());
            b[3] = self.widen_in(cx, &mut sc, &l::TIMESPEC, 1, a[3], mem)?;
        }
        Ok(self.native(cx, 202, b, false, vcpu, mem))
    }

    pub(super) fn c_setaffinity(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let len = a[1].min(1 << 16);
        let mut set = mem.read_vec(a[2], len as usize).map_err(efault)?;
        set.resize(len.next_multiple_of(8) as usize, 0);
        let mut sc = Scratch::new(vcpu.sp());
        let p = self.scratch_with(cx, &mut sc, &set, mem)?;
        let b = [sx(a[0]), set.len() as u64, p, 0, 0, 0];
        Ok(self.native(cx, 203, b, false, vcpu, mem))
    }

    pub(super) fn c_getaffinity(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let len = a[1];
        if !len.is_multiple_of(4) {
            return Ok(err(Errno::EINVAL));
        }
        let len64 = len.min(1 << 16).next_multiple_of(8);
        let mut sc = Scratch::new(vcpu.sp());
        let p = self.scratch(cx, &mut sc, len64, mem)?;
        let r = self.native(cx, 204, [sx(a[0]), len64, p, 0, 0, 0], false, vcpu, mem);
        if r < 0 {
            return Ok(r);
        }
        let n = (r as u64).min(len);
        let set = mem.read_vec(p, n as usize).map_err(efault)?;
        mem.write(a[2], &set).map_err(efault)?;
        Ok(n as i64)
    }

    /// `io_submit(ctx, nr, iocbpp)`: an array of 32-bit pointers.
    pub(super) fn c_io_submit(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let n = sx(a[1]) as i64;
        if n < 0 {
            return Ok(err(Errno::EINVAL));
        }
        let n = n.min(1 << 16) as usize;
        let raw = mem.read_vec(a[2], n * 4).map_err(efault)?;
        let wide: Vec<u8> = raw
            .chunks(4)
            .flat_map(|p| r32(p, 0).to_le_bytes())
            .collect();
        let mut sc = Scratch::new(vcpu.sp());
        let p = self.scratch_with(cx, &mut sc, &wide, mem)?;
        Ok(self.native(cx, 209, [a[0], n as u64, p, 0, 0, 0], false, vcpu, mem))
    }

    pub(super) fn c_waitid(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let mut sc = Scratch::new(vcpu.sp());
        let info = if a[2] == 0 {
            0
        } else {
            self.scratch_with(cx, &mut sc, &[0u8; 128], mem)?
        };
        let ru = if a[4] == 0 {
            0
        } else {
            self.scratch_with(cx, &mut sc, &[0u8; 144], mem)?
        };
        let b = [sx(a[0]), sx(a[1]), info, sx(a[3]), ru, 0];
        let r = self.native(cx, 247, b, false, vcpu, mem);
        if r < 0 || cx.block {
            return Ok(r);
        }
        if info != 0 {
            let n = mem.read_vec(info, 128).map_err(efault)?;
            mem.write(a[2], &siginfo_narrow(&n)).map_err(efault)?;
        }
        if ru != 0 {
            Self::narrow_back(mem, &l::RUSAGE, 1, ru, a[4])?;
        }
        Ok(r)
    }

    pub(super) fn c_get_robust_list(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let mut sc = Scratch::new(vcpu.sp());
        let p = self.scratch(cx, &mut sc, 16, mem)?;
        let r = self.native(cx, 274, [sx(a[0]), p, p + 8, 0, 0, 0], false, vcpu, mem);
        if r < 0 {
            return Ok(r);
        }
        let head = mem.read_u64(p).map_err(efault)?;
        mem.write(a[1], &(head as u32).to_le_bytes())
            .map_err(efault)?;
        mem.write(a[2], &12u32.to_le_bytes()).map_err(efault)?;
        Ok(r)
    }

    /// A call taking an `iovec` array (argument `iov`) of `cnt` entries.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn c_iov_call(
        &self,
        cx: &mut ServiceCtx,
        nr: u64,
        a: [u64; 6],
        iov: usize,
        cnt: usize,
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let n = a[cnt] & 0xffff_ffff;
        if n > UIO_MAXIOV {
            return Ok(err(Errno::EINVAL));
        }
        let mut b = a;
        if n > 0 {
            let mut sc = Scratch::new(vcpu.sp());
            b[iov] = self.widen_in(cx, &mut sc, &l::IOVEC, n as usize, a[iov], mem)?;
        }
        Ok(self.native(cx, nr, b, false, vcpu, mem))
    }

    /// `process_vm_readv/writev(pid, lvec, liovcnt, rvec, riovcnt, flags)`.
    pub(super) fn c_process_vm(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        write: bool,
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let (ln, rn) = (a[2], a[4]);
        if ln > UIO_MAXIOV || rn > UIO_MAXIOV {
            return Ok(err(Errno::EINVAL));
        }
        let mut sc = Scratch::new(vcpu.sp());
        let mut b = [sx(a[0]), a[1], ln, a[3], rn, a[5]];
        if ln > 0 {
            b[1] = self.widen_in(cx, &mut sc, &l::IOVEC, ln as usize, a[1], mem)?;
        }
        if rn > 0 {
            b[3] = self.widen_in(cx, &mut sc, &l::IOVEC, rn as usize, a[3], mem)?;
        }
        Ok(self.native(cx, if write { 311 } else { 310 }, b, false, vcpu, mem))
    }

    /// The x86-64 `msghdr` for the caller's at `user`, in scratch: its
    /// `iovec`s widened, and its control messages for a send (or room for
    /// the received ones).
    fn msg_in(
        &self,
        cx: &mut ServiceCtx,
        sc: &mut Scratch,
        user: u64,
        send: bool,
        mem: &mut GuestMemory,
    ) -> Result<Msg, i64> {
        let m = mem.read_vec(user, 28).map_err(efault)?;
        let (name, namelen, iov, iovlen) = (r32(&m, 0), r32(&m, 4), r32(&m, 8), r32(&m, 12));
        let (control, controllen, flags) = (r32(&m, 16), r32(&m, 20), r32(&m, 24));
        if iovlen > UIO_MAXIOV {
            return Err(err(Errno::EMSGSIZE));
        }
        let iov64 = if iovlen == 0 {
            iov
        } else {
            self.widen_in(cx, sc, &l::IOVEC, iovlen as usize, iov, mem)?
        };
        let (control64, controllen64) = if control == 0 || controllen == 0 {
            (control, 0)
        } else if send {
            let raw = mem
                .read_vec(control, controllen.min(1 << 16) as usize)
                .map_err(efault)?;
            let c = cmsgs_widen(&raw)?;
            (self.scratch_with(cx, sc, &c, mem)?, c.len() as u64)
        } else {
            let room = controllen.min(1 << 16) * 2 + 64;
            (self.scratch(cx, sc, room, mem)?, room)
        };
        let mut n = vec![0u8; 56];
        w64(&mut n, 0, name);
        w32(&mut n, 8, namelen);
        w64(&mut n, 16, iov64);
        w64(&mut n, 24, iovlen);
        w64(&mut n, 32, control64);
        w64(&mut n, 40, controllen64);
        w32(&mut n, 48, flags);
        let native = self.scratch_with(cx, sc, &n, mem)?;
        Ok(Msg {
            user,
            native,
            control32: control,
            controllen32: controllen,
            control64: if send { 0 } else { control64 },
        })
    }

    /// Back to the caller's `msghdr` after a receive: the name's length, the
    /// control messages (narrowed, `MSG_CTRUNC` if they don't fit) and the
    /// flags.
    ///
    /// The descriptors of an `SCM_RIGHTS` that didn't fit are closed.
    fn msg_out(
        &self,
        cx: &mut ServiceCtx,
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
        m: &Msg,
    ) -> Result<(), i64> {
        let n = mem.read_vec(m.native, 56).map_err(efault)?;
        let mut flags = r32(&n, 48);
        mem.write(m.user + 4, &n[8..12]).map_err(efault)?;
        let mut used = 0;
        if m.control64 != 0 {
            let len = r64(&n, 40).min(1 << 17);
            let raw = mem.read_vec(m.control64, len as usize).map_err(efault)?;
            let (c, trunc, lost) = cmsgs_narrow(&raw, m.controllen32 as usize);
            if trunc {
                flags |= MSG_CTRUNC;
            }
            for fd in lost {
                self.native(cx, 3, [fd as u64, 0, 0, 0, 0, 0], false, vcpu, mem);
            }
            mem.write(m.control32, &c).map_err(efault)?;
            used = c.len() as u64;
        }
        mem.write(m.user + 20, &(used as u32).to_le_bytes())
            .map_err(efault)?;
        mem.write(m.user + 24, &(flags as u32).to_le_bytes())
            .map_err(efault)?;
        Ok(())
    }

    pub(super) fn c_sendmsg(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let mut sc = Scratch::new(vcpu.sp());
        let m = self.msg_in(cx, &mut sc, a[1], true, mem)?;
        Ok(self.native(cx, 46, [sx(a[0]), m.native, a[2], 0, 0, 0], true, vcpu, mem))
    }

    pub(super) fn c_recvmsg(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let mut sc = Scratch::new(vcpu.sp());
        let m = self.msg_in(cx, &mut sc, a[1], false, mem)?;
        let r = self.native(cx, 47, [sx(a[0]), m.native, a[2], 0, 0, 0], true, vcpu, mem);
        if r >= 0 && !cx.block {
            self.msg_out(cx, vcpu, mem, &m)?;
        }
        Ok(r)
    }

    /// `sendmmsg(fd, vec, vlen, flags)`: an array of `mmsghdr`s (32 bytes
    /// each here, 64 there).
    pub(super) fn c_sendmmsg(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let n = a[2].min(UIO_MAXIOV);
        let mut sc = Scratch::new(vcpu.sp());
        let vec = self.mmsg_in(cx, &mut sc, a[1], n, true, mem)?;
        let r = self.native(cx, 307, [sx(a[0]), vec.0, n, a[3], 0, 0], true, vcpu, mem);
        if r > 0 && !cx.block {
            for i in 0..r as u64 {
                let len = mem.read_u32(vec.0 + i * 64 + 56).map_err(efault)?;
                mem.write(a[1] + i * 32 + 28, &len.to_le_bytes())
                    .map_err(efault)?;
            }
        }
        Ok(r)
    }

    pub(super) fn c_recvmmsg(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        time64: bool,
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let n = a[2].min(UIO_MAXIOV);
        let mut sc = Scratch::new(vcpu.sp());
        let (vec, msgs) = self.mmsg_in(cx, &mut sc, a[1], n, false, mem)?;
        let ts = if a[4] == 0 || time64 {
            a[4]
        } else {
            self.widen_in(cx, &mut sc, &l::TIMESPEC, 1, a[4], mem)?
        };
        let r = self.native(cx, 299, [sx(a[0]), vec, n, a[3], ts, 0], true, vcpu, mem);
        if r > 0 && !cx.block {
            for (i, m) in msgs.iter().enumerate().take(r as usize) {
                self.msg_out(cx, vcpu, mem, m)?;
                let len = mem.read_u32(vec + i as u64 * 64 + 56).map_err(efault)?;
                mem.write(m.user + 28, &len.to_le_bytes()).map_err(efault)?;
            }
        }
        if ts != a[4] && r >= 0 && !cx.block {
            Self::narrow_back(mem, &l::TIMESPEC, 1, ts, a[4])?;
        }
        Ok(r)
    }

    /// The x86-64 `mmsghdr` array for the caller's `n` at `at`.
    fn mmsg_in(
        &self,
        cx: &mut ServiceCtx,
        sc: &mut Scratch,
        at: u64,
        n: u64,
        send: bool,
        mem: &mut GuestMemory,
    ) -> Result<(u64, Vec<Msg>), i64> {
        let mut msgs = Vec::with_capacity(n as usize);
        for i in 0..n {
            msgs.push(self.msg_in(cx, sc, at + i * 32, send, mem)?);
        }
        let mut arr = vec![0u8; n as usize * 64];
        for (i, m) in msgs.iter().enumerate() {
            let h = mem.read_vec(m.native, 56).map_err(efault)?;
            arr[i * 64..i * 64 + 56].copy_from_slice(&h);
        }
        let vec = self.scratch_with(cx, sc, &arr, mem)?;
        // Where each message's header now is, for the way back.
        for (i, m) in msgs.iter_mut().enumerate() {
            m.native = vec + i as u64 * 64;
        }
        Ok((vec, msgs))
    }

    pub(super) fn c_getsockopt(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let b = [sx(a[0]), sx(a[1]), sx(a[2]), a[3], a[4], 0];
        // SO_RCVTIMEO_OLD, SO_SNDTIMEO_OLD: a 32-bit `timeval`.
        if a[1] != SOL_SOCKET || !matches!(a[2], 20 | 21) || a[3] == 0 || a[4] == 0 {
            return Ok(self.native(cx, 55, b, true, vcpu, mem));
        }
        let len = mem.read_u32(a[4]).map_err(efault)?;
        let mut sc = Scratch::new(vcpu.sp());
        let val = self.scratch(cx, &mut sc, 16, mem)?;
        let lenp = self.scratch_with(cx, &mut sc, &16u32.to_le_bytes(), mem)?;
        let r = self.native(cx, 55, [b[0], b[1], b[2], val, lenp, 0], true, vcpu, mem);
        if r < 0 {
            return Ok(r);
        }
        let tv = mem.read_vec(val, 16).map_err(efault)?;
        let c = l::TIMESPEC.narrow(&tv).map_err(|_| err(Errno::EOVERFLOW))?;
        let n = (len as usize).min(8);
        mem.write(a[3], &c[..n]).map_err(efault)?;
        mem.write(a[4], &(n as u32).to_le_bytes()).map_err(efault)?;
        Ok(r)
    }

    pub(super) fn c_setsockopt(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let b = [sx(a[0]), sx(a[1]), sx(a[2]), a[3], a[4], 0];
        let layout = match (a[1], a[2]) {
            (SOL_SOCKET, 20 | 21) => &l::TIMESPEC,
            // SO_ATTACH_FILTER, SO_ATTACH_REUSEPORT_CBPF: a `sock_fprog`.
            (SOL_SOCKET, 26 | 51) => &l::SOCK_FPROG,
            _ => return Ok(self.native(cx, 54, b, true, vcpu, mem)),
        };
        if a[4] < layout.size32 as u64 {
            return Ok(err(Errno::EINVAL));
        }
        let mut sc = Scratch::new(vcpu.sp());
        let p = self.widen_in(cx, &mut sc, layout, 1, a[3], mem)?;
        let n = layout.size64 as u64;
        Ok(self.native(cx, 54, [b[0], b[1], b[2], p, n, 0], true, vcpu, mem))
    }

    /// `prctl`: `PR_SET_SECCOMP`'s filter is a 32-bit `sock_fprog`.
    pub(super) fn c_prctl(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let mut b = [sx(a[0]), a[1], a[2], a[3], a[4], 0];
        if a[0] == 22 && a[1] == 2 && a[2] != 0 {
            let mut sc = Scratch::new(vcpu.sp());
            b[2] = self.widen_in(cx, &mut sc, &l::SOCK_FPROG, 1, a[2], mem)?;
        }
        Ok(self.native(cx, 157, b, false, vcpu, mem))
    }

    /// `seccomp(op, flags, args)`: `SECCOMP_SET_MODE_FILTER`'s program is a
    /// 32-bit `sock_fprog`.
    pub(super) fn c_seccomp(
        &self,
        cx: &mut ServiceCtx,
        a: [u64; 6],
        vcpu: &mut dyn Vcpu,
        mem: &mut GuestMemory,
    ) -> R {
        let mut b = a;
        if a[0] == 1 && a[2] != 0 {
            let mut sc = Scratch::new(vcpu.sp());
            b[2] = self.widen_in(cx, &mut sc, &l::SOCK_FPROG, 1, a[2], mem)?;
        }
        Ok(self.native(cx, 317, b, false, vcpu, mem))
    }
}

/// `UIO_MAXIOV`.
const UIO_MAXIOV: u64 = 1024;

/// The x86-64 control messages for the caller's: 16-byte headers and
/// 8-byte alignment instead of 12 and 4 (`cmsghdr_from_user_compat_to_kern`).
fn cmsgs_widen(c: &[u8]) -> Result<Vec<u8>, i64> {
    let mut out = Vec::new();
    let mut off = 0;
    while off + 12 <= c.len() {
        let len = r32(c, off) as usize;
        if len < 12 || off + len > c.len() {
            return Err(err(Errno::EINVAL));
        }
        let data = &c[off + 12..off + len];
        let start = out.len();
        out.resize(start + 16, 0);
        w64(&mut out, start, 16 + data.len() as u64);
        out[start + 8..start + 16].copy_from_slice(&c[off + 4..off + 12]);
        out.extend_from_slice(data);
        out.resize(out.len().next_multiple_of(8), 0);
        off += len.next_multiple_of(4);
    }
    Ok(out)
}

/// The caller's control messages for x86-64's, as many as fit in `room`
/// bytes (and whether some didn't: `MSG_CTRUNC`), with the old timestamps'
/// `timeval`s and `timespec`s narrowed (`put_cmsg_compat`). An
/// `SCM_RIGHTS` that doesn't fit keeps the descriptors that do
/// (`scm_detach_fds_compat`); the others, which the x86-64 receive
/// installed, are returned for the caller to close: Linux never installs
/// them. An `SCM_PIDFD` is whole or not at all (`scm_pidfd_recv`, which
/// wants room for a compat header and the `int`), its pidfd likewise
/// returned when it isn't.
fn cmsgs_narrow(n: &[u8], room: usize) -> (Vec<u8>, bool, Vec<i32>) {
    let mut out = Vec::new();
    let mut lost = Vec::new();
    let mut trunc = false;
    let mut off = 0;
    while off + 16 <= n.len() {
        let len = r64(n, off) as usize;
        if len < 16 || off + len > n.len() {
            break;
        }
        let (level, ty) = (r32(n, off + 8), r32(n, off + 12));
        let (at, raw) = (off, &n[off + 16..off + len]);
        off += len.next_multiple_of(8);
        let rights = level == SOL_SOCKET && ty == SCM_RIGHTS;
        let pidfd = level == SOL_SOCKET && ty == SCM_PIDFD && raw.len() == 4;
        if trunc {
            if rights || pidfd {
                lost.extend(
                    raw.as_chunks::<4>()
                        .0
                        .iter()
                        .map(|f| i32::from_le_bytes(*f)),
                );
            }
            continue;
        }
        // SO_TIMESTAMP_OLD (a timeval), SO_TIMESTAMPNS_OLD (a timespec),
        // SO_TIMESTAMPING_OLD (three timespecs).
        let mut data: Vec<u8> = if level == SOL_SOCKET && matches!(ty, 29 | 35 | 37) {
            raw.chunks(16)
                .filter(|t| t.len() == 16)
                .flat_map(|t| l::TIMESPEC.narrow(t).unwrap_or_default())
                .collect()
        } else {
            raw.to_vec()
        };
        if out.len() + 12 + data.len() > room {
            trunc = true;
            if pidfd {
                lost.push(i32::from_le_bytes(data[..4].try_into().unwrap()));
            }
            if !rights {
                continue;
            }
            // The descriptors that fit after the header.
            let fit = room.saturating_sub(out.len() + 12) / 4;
            let fit = fit.min(data.len() / 4);
            lost.extend(
                data[fit * 4..]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|f| i32::from_le_bytes(*f)),
            );
            if fit == 0 {
                continue;
            }
            data.truncate(fit * 4);
        }
        let clen = 12 + data.len();
        let start = out.len();
        out.resize(start + 12, 0);
        w32(&mut out, start, clen as u64);
        out[start + 4..start + 12].copy_from_slice(&n[at + 8..at + 16]);
        out.extend_from_slice(&data);
        if out.len().next_multiple_of(4) <= room {
            out.resize(out.len().next_multiple_of(4), 0);
        }
    }
    (out, trunc, lost)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn control_messages_change_header_and_alignment() {
        // SCM_RIGHTS with one fd: 16 bytes here, 24 (20 + padding) there.
        let mut c = vec![0u8; 16];
        w32(&mut c, 0, 16);
        w32(&mut c, 4, 1);
        w32(&mut c, 8, 1);
        w32(&mut c, 12, 7);
        let n = cmsgs_widen(&c).unwrap();
        assert_eq!(n.len(), 24);
        assert_eq!(r64(&n, 0), 20);
        assert_eq!(r32(&n, 16), 7);
        let (back, trunc, _) = cmsgs_narrow(&n, 64);
        assert!(!trunc);
        assert_eq!(back, c);
        let (none, trunc, lost) = cmsgs_narrow(&n, 8);
        assert!(trunc && none.is_empty());
        assert_eq!(lost, [7]);
    }

    #[test]
    fn descriptors_that_dont_fit_are_given_back() {
        // SCM_RIGHTS with fds 5, 6, 7 (28 bytes there) and room for two.
        let mut n = vec![0u8; 32];
        w64(&mut n, 0, 28);
        w32(&mut n, 8, SOL_SOCKET);
        w32(&mut n, 12, SCM_RIGHTS);
        for (i, fd) in [5, 6, 7].into_iter().enumerate() {
            w32(&mut n, 16 + i * 4, fd);
        }
        let (c, trunc, lost) = cmsgs_narrow(&n, 20);
        assert!(trunc);
        assert_eq!(lost, [7]);
        assert_eq!((r32(&c, 0), c.len()), (20, 20));
        assert_eq!((r32(&c, 12), r32(&c, 16)), (5, 6));
    }

    #[test]
    fn a_pidfd_that_doesnt_fit_is_given_back() {
        // SCM_RIGHTS with fd 5, then SCM_PIDFD 9: 16 + 16 bytes there.
        let mut n = vec![0u8; 48];
        w64(&mut n, 0, 20);
        w32(&mut n, 8, SOL_SOCKET);
        w32(&mut n, 12, SCM_RIGHTS);
        w32(&mut n, 16, 5);
        w64(&mut n, 24, 20);
        w32(&mut n, 32, SOL_SOCKET);
        w32(&mut n, 36, SCM_PIDFD);
        w32(&mut n, 40, 9);
        let (c, trunc, lost) = cmsgs_narrow(&n, 32);
        assert!(!trunc && lost.is_empty());
        assert_eq!((c.len(), r32(&c, 24), r32(&c, 28)), (32, SCM_PIDFD, 9));
        let (c, trunc, lost) = cmsgs_narrow(&n, 31);
        assert!(trunc);
        assert_eq!((c.len(), lost.as_slice()), (16, &[9][..]));
    }

    #[test]
    fn siginfo_round_trips_a_child_and_a_fault() {
        let mut c = [0u8; 128];
        w32(&mut c, 0, 17);
        w32(&mut c, 8, 1); // CLD_EXITED
        w32(&mut c, 12, 42);
        w32(&mut c, 16, 1000);
        w32(&mut c, 20, 3);
        w32(&mut c, 24, 0xffff_ffff); // utime -1
        let n = siginfo_widen(&c);
        assert_eq!(r32(&n, 16), 42);
        assert_eq!(r64(&n, 32), u64::MAX);
        assert_eq!(siginfo_narrow(&n), c);
        let mut f = [0u8; 128];
        w32(&mut f, 0, 11);
        w32(&mut f, 8, 1);
        w64(&mut f, 16, 0xdead_beef);
        let c = siginfo_narrow(&f);
        assert_eq!(r32(&c, 12), 0xdead_beef);
    }
}
