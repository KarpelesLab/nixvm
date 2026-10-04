//! Process/thread attribute syscalls with real (if simple) state: scheduling
//! parameters (`sched_setparam`, `sched_rr_get_interval`, `sched_[gs]etattr`),
//! I/O priority, NUMA memory policy on a one-node machine, memory protection
//! keys, `mseal`, `membarrier`, `rseq`, capabilities, `adjtimex`,
//! `personality`, host/domain names, the robust-futex list head, `unshare`/
//! `setns`, and a few legacy x86-64 calls (`ustat`, `sysfs`, `vhangup`,
//! `iopl`/`ioperm`, `remap_file_pages`).
//!
//! The pattern throughout is the one the rest of the kernel follows for
//! attributes the cooperative scheduler doesn't act on: validate exactly as
//! Linux does (so a misuse fails the same way), record what was set, and report
//! it back — a program that sets a value and reads it back, or probes for a
//! feature, sees a coherent machine.

use super::{Fd, Kernel, ProcInfo, RunState, ServiceCtx, Shared, err};
use crate::abi::errno::Errno;
use crate::vcpu::GuestMemory;
use crate::vcpu::mem::PAGE_SIZE;

const SCHED_OTHER: u32 = 0;
const SCHED_FIFO: u32 = 1;
const SCHED_RR: u32 = 2;
const SCHED_BATCH: u32 = 3;
const SCHED_IDLE: u32 = 5;
const SCHED_DEADLINE: u32 = 6;
/// Every capability bit through `CAP_CHECKPOINT_RESTORE` (40), the last one
/// Linux defines: what root holds.
const CAP_LAST_CAP: u64 = 40;
const CAP_ALL: u64 = (1u64 << (CAP_LAST_CAP + 1)) - 1;

/// Run `f` on the task `pid` names (`0` = the caller), `ESRCH` if none.
fn with_task<R>(
    sh: &mut Shared,
    cx: &mut ServiceCtx,
    pid: u64,
    f: impl FnOnce(&mut ProcInfo) -> R,
) -> Result<R, i64> {
    let pid = pid as i32;
    if pid < 0 {
        return Err(err(Errno::EINVAL));
    }
    if pid == 0 || pid == cx.cur.pid {
        return Ok(f(&mut cx.cur));
    }
    sh.procs
        .iter_mut()
        .flatten()
        .find(|p| p.info.pid == pid && !matches!(p.info.run, RunState::Zombie(_)))
        .map(|p| f(&mut p.info))
        .ok_or_else(|| err(Errno::ESRCH))
}

/// The valid `sched_priority` range for a policy: 1..=99 for the real-time
/// ones, exactly 0 otherwise.
fn priority_ok(policy: u32, prio: u32) -> bool {
    match policy {
        SCHED_FIFO | SCHED_RR => (1..=99).contains(&prio),
        _ => prio == 0,
    }
}

/// Read a nodemask of `maxnode` bits at `ptr` (Linux reads `maxnode - 1`
/// bits — a historical off-by-one every libnuma accounts for). Any node beyond
/// node 0 is not online here: `EINVAL`.
fn read_nodemask(mem: &GuestMemory, ptr: u64, maxnode: u64) -> Result<u64, i64> {
    if ptr == 0 || maxnode <= 1 {
        return Ok(0);
    }
    let bits = maxnode - 1;
    if bits > 1 << 20 {
        return Err(err(Errno::EINVAL));
    }
    let words = bits.div_ceil(64);
    let mut mask0 = 0;
    for w in 0..words {
        let Ok(mut v) = mem.read_u64(ptr + w * 8) else {
            return Err(err(Errno::EFAULT));
        };
        // Only the low `bits` bits are significant.
        let valid = bits - w * 64;
        if valid < 64 {
            v &= (1u64 << valid) - 1;
        }
        if w == 0 {
            mask0 = v;
            if v & !1 != 0 {
                return Err(err(Errno::EINVAL));
            }
        } else if v != 0 {
            return Err(err(Errno::EINVAL));
        }
    }
    Ok(mask0)
}

/// Validate a memory-policy `mode` (with its `MPOL_F_*` flag bits) against
/// its nodemask, as `set_mempolicy`/`mbind` do. Returns the bare mode.
fn check_mpol(mode: u64, mask: u64) -> Result<u16, i64> {
    const MPOL_DEFAULT: u64 = 0;
    const MPOL_PREFERRED: u64 = 1;
    const MPOL_LOCAL: u64 = 4;
    const MPOL_WEIGHTED_INTERLEAVE: u64 = 6;
    const MPOL_F_NUMA_BALANCING: u64 = 1 << 13;
    const MPOL_F_RELATIVE_NODES: u64 = 1 << 14;
    const MPOL_F_STATIC_NODES: u64 = 1 << 15;
    let flags = mode & (MPOL_F_NUMA_BALANCING | MPOL_F_RELATIVE_NODES | MPOL_F_STATIC_NODES);
    let m = mode & !flags;
    if m > MPOL_WEIGHTED_INTERLEAVE
        || (flags & MPOL_F_STATIC_NODES != 0 && flags & MPOL_F_RELATIVE_NODES != 0)
    {
        return Err(err(Errno::EINVAL));
    }
    match m {
        // DEFAULT and LOCAL take no nodes; PREFERRED may name none (= local).
        MPOL_DEFAULT | MPOL_LOCAL if mask != 0 => Err(err(Errno::EINVAL)),
        MPOL_DEFAULT | MPOL_LOCAL | MPOL_PREFERRED => Ok(m as u16),
        // BIND / INTERLEAVE / PREFERRED_MANY / WEIGHTED_INTERLEAVE need nodes.
        _ if mask == 0 => Err(err(Errno::EINVAL)),
        _ => Ok(m as u16),
    }
}

impl Kernel {
    /// `sched_setparam(pid, param)`: the priority under the current policy.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_sched_setparam(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        pid: u64,
        param: u64,
        mem: &GuestMemory,
    ) -> i64 {
        if param == 0 {
            return err(Errno::EINVAL);
        }
        let Ok(prio) = mem.read_u32(param) else {
            return err(Errno::EFAULT);
        };
        match with_task(sh, cx, pid, |p| {
            if priority_ok(p.sched_policy as u32, prio) {
                p.sched_priority = prio as i32;
                0
            } else {
                err(Errno::EINVAL)
            }
        }) {
            Ok(r) | Err(r) => r,
        }
    }

    /// `sched_rr_get_interval(pid, tp)`: the round-robin quantum — 100 ms for
    /// `SCHED_RR`, none for `SCHED_FIFO`, and the fair class's ~4 ms slice.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_sched_rr_get_interval(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        pid: u64,
        tp: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let policy = match with_task(sh, cx, pid, |p| p.sched_policy as u32) {
            Ok(p) => p,
            Err(e) => return e,
        };
        let ns: u64 = match policy {
            SCHED_RR => 100_000_000,
            SCHED_FIFO => 0,
            _ => 4_000_000,
        };
        let mut b = [0u8; 16];
        b[8..16].copy_from_slice(&ns.to_le_bytes());
        if mem.write(tp, &b).is_err() {
            return err(Errno::EFAULT);
        }
        0
    }

    /// `sched_setattr(pid, struct sched_attr *, flags)`. The struct is
    /// size-versioned (48 bytes v0, 56 with the util-clamp fields); a short
    /// one is `E2BIG` with the expected size written back, a long one must be
    /// zero-padded.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_sched_setattr(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        pid: u64,
        attr: u64,
        flags: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const VER0: u32 = 48;
        const VER1: u32 = 56;
        const SCHED_FLAG_KEEP_POLICY: u64 = 0x08;
        const SCHED_FLAG_KEEP_PARAMS: u64 = 0x10;
        const SCHED_FLAG_ALL: u64 = 0x7f;
        if flags != 0 || attr == 0 {
            return err(Errno::EINVAL);
        }
        let Ok(mut size) = mem.read_u32(attr) else {
            return err(Errno::EFAULT);
        };
        if size == 0 {
            size = VER0;
        }
        if !(VER0..=4096).contains(&size) {
            let _ = mem.write(attr, &VER1.to_le_bytes());
            return err(Errno::E2BIG);
        }
        let Ok(raw) = mem.read_vec(attr, size as usize) else {
            return err(Errno::EFAULT);
        };
        if raw.len() > VER1 as usize && raw[VER1 as usize..].iter().any(|&b| b != 0) {
            let _ = mem.write(attr, &VER1.to_le_bytes());
            return err(Errno::E2BIG);
        }
        let u32_at = |o: usize| u32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(raw[o..o + 8].try_into().unwrap());
        let (policy, sflags, nice, prio) = (u32_at(4), u64_at(8), u32_at(16) as i32, u32_at(20));
        let (runtime, deadline, period) = (u64_at(24), u64_at(32), u64_at(40));
        if sflags & !SCHED_FLAG_ALL != 0 {
            return err(Errno::EINVAL);
        }
        if !matches!(
            policy,
            SCHED_OTHER | SCHED_FIFO | SCHED_RR | SCHED_BATCH | SCHED_IDLE | SCHED_DEADLINE
        ) && sflags & SCHED_FLAG_KEEP_POLICY == 0
        {
            return err(Errno::EINVAL);
        }
        if policy == SCHED_DEADLINE {
            // runtime <= deadline <= period (period 0 = deadline), runtime at
            // least 1 µs: Linux's __checkparam_dl().
            let period = if period == 0 { deadline } else { period };
            if deadline == 0 || runtime < 1024 || runtime > deadline || deadline > period {
                return err(Errno::EINVAL);
            }
        } else if !priority_ok(policy, prio) && sflags & SCHED_FLAG_KEEP_PARAMS == 0 {
            return err(Errno::EINVAL);
        }
        match with_task(sh, cx, pid, |p| {
            if sflags & SCHED_FLAG_KEEP_POLICY == 0 {
                p.sched_policy = policy as i32;
            }
            if sflags & SCHED_FLAG_KEEP_PARAMS == 0 {
                p.sched_priority = prio as i32;
                if matches!(policy, SCHED_OTHER | SCHED_BATCH) {
                    p.nice = nice.clamp(-20, 19);
                }
            }
        }) {
            Ok(()) => 0,
            Err(e) => e,
        }
    }

    /// `sched_getattr(pid, struct sched_attr *, size, flags)`.
    #[allow(clippy::unused_self, clippy::too_many_arguments)]
    pub(super) fn sys_sched_getattr(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        pid: u64,
        attr: u64,
        size: u64,
        flags: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        if flags != 0 || attr == 0 || !(48..=4096).contains(&size) {
            return err(Errno::EINVAL);
        }
        let (policy, prio, nice) = match with_task(sh, cx, pid, |p| {
            (p.sched_policy as u32, p.sched_priority as u32, p.nice)
        }) {
            Ok(v) => v,
            Err(e) => return e,
        };
        let n = size.min(56) as usize;
        let mut b = [0u8; 56];
        b[0..4].copy_from_slice(&(n as u32).to_le_bytes());
        b[4..8].copy_from_slice(&policy.to_le_bytes());
        b[16..20].copy_from_slice(&nice.to_le_bytes());
        b[20..24].copy_from_slice(&prio.to_le_bytes());
        b[52..56].copy_from_slice(&1024u32.to_le_bytes()); // sched_util_max
        if mem.write(attr, &b[..n]).is_err() {
            return err(Errno::EFAULT);
        }
        0
    }

    /// `ioprio_set(which, who, ioprio)` / `ioprio_get(which, who)`. `which` is
    /// `IOPRIO_WHO_PROCESS`/`PGRP`/`USER`; the priority is `class << 13 |
    /// level`. An unset priority reads as Linux's default: best-effort, level
    /// derived from `nice` (`(nice + 20) / 5`).
    #[allow(clippy::unused_self)]
    pub(super) fn sys_ioprio(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        which: u64,
        who: u64,
        set: Option<u64>,
    ) -> i64 {
        const IOPRIO_WHO_PROCESS: u64 = 1;
        const IOPRIO_WHO_PGRP: u64 = 2;
        const IOPRIO_WHO_USER: u64 = 3;
        const IOPRIO_CLASS_RT: u64 = 1;
        const IOPRIO_CLASS_BE: u64 = 2;
        if !matches!(
            which,
            IOPRIO_WHO_PROCESS | IOPRIO_WHO_PGRP | IOPRIO_WHO_USER
        ) {
            return err(Errno::EINVAL);
        }
        if let Some(prio) = set {
            let (class, level) = ((prio >> 13) & 0x7, prio & 0x1fff);
            if class > 3 || (class != 0 && class != 3 && level > 7) || prio > 0xffff {
                return err(Errno::EINVAL);
            }
            if class == IOPRIO_CLASS_RT && cx.cur.creds.euid != 0 {
                return err(Errno::EPERM);
            }
        }
        // PGRP/USER address a set of tasks; only the caller is modeled for
        // them (it is in its own group and is its own user).
        let pid = if which == IOPRIO_WHO_PROCESS { who } else { 0 };
        match with_task(sh, cx, pid, |p| {
            if let Some(prio) = set {
                p.ioprio = prio as u16;
                0
            } else if p.ioprio == 0 {
                let level = i64::from(((p.nice + 20) / 5).clamp(0, 7));
                ((IOPRIO_CLASS_BE << 13) as i64) | level
            } else {
                i64::from(p.ioprio)
            }
        }) {
            Ok(r) | Err(r) => r,
        }
    }

    /// `set_mempolicy(mode, nodemask, maxnode)`: one NUMA node (0) is online.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_set_mempolicy(
        &self,
        cx: &mut ServiceCtx,
        mode: u64,
        nodemask: u64,
        maxnode: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let mask = match read_nodemask(mem, nodemask, maxnode) {
            Ok(m) => m,
            Err(e) => return e,
        };
        match check_mpol(mode, mask) {
            Ok(m) => {
                cx.cur.mempolicy = (m, mask);
                0
            }
            Err(e) => e,
        }
    }

    /// `get_mempolicy(mode*, nodemask*, maxnode, addr, flags)`.
    #[allow(clippy::unused_self, clippy::too_many_arguments)]
    pub(super) fn sys_get_mempolicy(
        &self,
        cx: &ServiceCtx,
        modep: u64,
        nodemask: u64,
        maxnode: u64,
        addr: u64,
        flags: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const MPOL_F_NODE: u64 = 1;
        const MPOL_F_ADDR: u64 = 2;
        const MPOL_F_MEMS_ALLOWED: u64 = 4;
        const MPOL_INTERLEAVE: u16 = 3;
        if flags & !(MPOL_F_NODE | MPOL_F_ADDR | MPOL_F_MEMS_ALLOWED) != 0
            || (flags & MPOL_F_MEMS_ALLOWED != 0 && flags & (MPOL_F_NODE | MPOL_F_ADDR) != 0)
        {
            return err(Errno::EINVAL);
        }
        if nodemask != 0 && maxnode < 1 {
            return err(Errno::EINVAL);
        }
        if flags & MPOL_F_ADDR != 0 && mem.page_prot(addr).is_none() {
            return err(Errno::EFAULT);
        }
        let (policy, pmask) = cx.cur.mempolicy;
        let (mode, mask) = if flags & MPOL_F_MEMS_ALLOWED != 0 {
            (0, 1)
        } else if flags & MPOL_F_NODE != 0 {
            // The node `addr` lives on (with F_ADDR) or the next interleave
            // node: always node 0.
            if flags & MPOL_F_ADDR == 0 && policy != MPOL_INTERLEAVE {
                return err(Errno::EINVAL);
            }
            (0, pmask)
        } else {
            (i32::from(policy), pmask)
        };
        if modep != 0 && mem.write(modep, &mode.to_le_bytes()).is_err() {
            return err(Errno::EFAULT);
        }
        if nodemask != 0 {
            let words = (maxnode.saturating_sub(1)).div_ceil(64).max(1) as usize;
            let mut b = vec![0u8; words * 8];
            b[0..8].copy_from_slice(&mask.to_le_bytes());
            if mem.write(nodemask, &b).is_err() {
                return err(Errno::EFAULT);
            }
        }
        0
    }

    /// `mbind(addr, len, mode, nodemask, maxnode, flags)`: a per-range policy.
    /// With one node every policy places pages identically, so after Linux's
    /// validation (page-aligned start, mapped range, valid mode/nodes/flags)
    /// there is nothing to record.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_mbind(&self, a: &[u64; 6], mem: &GuestMemory) -> i64 {
        const MPOL_MF_STRICT: u64 = 1;
        const MPOL_MF_MOVE: u64 = 2;
        const MPOL_MF_MOVE_ALL: u64 = 4;
        let (addr, len, mode, nodemask, maxnode, flags) = (a[0], a[1], a[2], a[3], a[4], a[5]);
        if flags & !(MPOL_MF_STRICT | MPOL_MF_MOVE | MPOL_MF_MOVE_ALL) != 0
            || !addr.is_multiple_of(PAGE_SIZE)
        {
            return err(Errno::EINVAL);
        }
        let mask = match read_nodemask(mem, nodemask, maxnode) {
            Ok(m) => m,
            Err(e) => return e,
        };
        if let Err(e) = check_mpol(mode, mask) {
            return e;
        }
        let end = addr.saturating_add(len.div_ceil(PAGE_SIZE) * PAGE_SIZE);
        let mut p = addr;
        while p < end {
            if mem.page_prot(p).is_none() {
                return err(Errno::EFAULT);
            }
            p += PAGE_SIZE;
        }
        0
    }

    /// `move_pages(pid, count, pages, nodes, status, flags)`: report (or
    /// "move" to node 0) each page's node — 0 for a mapped page, `-EFAULT` for
    /// an unmapped one, `-ENODEV` for a request to an offline node. Only the
    /// caller's own address space is inspected; another process's pages are
    /// reported as node 0.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_move_pages(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        a: &[u64; 6],
        mem: &mut GuestMemory,
    ) -> i64 {
        const MPOL_MF_MOVE: u64 = 2;
        const MPOL_MF_MOVE_ALL: u64 = 4;
        let (pid, count, pages, nodes, status, flags) = (a[0], a[1], a[2], a[3], a[4], a[5]);
        if flags & !(MPOL_MF_MOVE | MPOL_MF_MOVE_ALL) != 0 {
            return err(Errno::EINVAL);
        }
        let own = match with_task(sh, cx, pid, |p| p.mm) {
            Ok(mm) => mm == cx.cur.mm,
            Err(e) => return e,
        };
        for i in 0..count.min(1 << 20) {
            let Ok(page) = mem.read_u64(pages + i * 8) else {
                return err(Errno::EFAULT);
            };
            let st: i32 = if nodes != 0 && mem.read_u32(nodes + i * 4).is_ok_and(|n| n != 0) {
                -19 // -ENODEV: only node 0 exists
            } else if own && mem.page_prot(page).is_none() {
                -14 // -EFAULT: not mapped
            } else {
                0
            };
            if status != 0 && mem.write(status + i * 4, &st.to_le_bytes()).is_err() {
                return err(Errno::EFAULT);
            }
        }
        0
    }

    /// `pkey_alloc(flags, access_rights)`: protection keys aren't available
    /// (no PKU/POE in the emulated CPUs), so after argument validation the
    /// answer is Linux's "no key free" — `ENOSPC`.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_pkey_alloc(&self, flags: u64, rights: u64) -> i64 {
        const PKEY_DISABLE_ACCESS: u64 = 1;
        const PKEY_DISABLE_WRITE: u64 = 2;
        if flags != 0 || rights & !(PKEY_DISABLE_ACCESS | PKEY_DISABLE_WRITE) != 0 {
            return err(Errno::EINVAL);
        }
        err(Errno::ENOSPC)
    }

    /// `pkey_mprotect(addr, len, prot, pkey)`: with no keys allocatable, only
    /// the implicit default key (0) and "no key" (-1) are valid — both mean a
    /// plain `mprotect`.
    pub(super) fn sys_pkey_mprotect(
        &self,
        sh: &Shared,
        cx: &ServiceCtx,
        a: &[u64; 6],
        mem: &mut GuestMemory,
    ) -> i64 {
        match a[3] as i32 {
            -1 | 0 => self.sys_mprotect_sealed(sh, cx, a[0], a[1], a[2], mem),
            _ => err(Errno::EINVAL),
        }
    }

    /// `mseal(addr, len, flags)`: forbid further changes to a mapping. The
    /// range must be page-aligned and fully mapped (`ENOMEM` otherwise); once
    /// sealed, `munmap`/`mprotect`/`mremap` over it fail `EPERM`.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_mseal(
        &self,
        sh: &mut Shared,
        cx: &ServiceCtx,
        addr: u64,
        len: u64,
        flags: u64,
        mem: &GuestMemory,
    ) -> i64 {
        if flags != 0 || !addr.is_multiple_of(PAGE_SIZE) {
            return err(Errno::EINVAL);
        }
        let len = len.div_ceil(PAGE_SIZE) * PAGE_SIZE;
        let Some(end) = addr.checked_add(len) else {
            return err(Errno::EINVAL);
        };
        let mut p = addr;
        while p < end {
            if mem.page_prot(p).is_none() {
                return err(Errno::ENOMEM);
            }
            p += PAGE_SIZE;
        }
        if len > 0 {
            sh.sealed.push((cx.cur.mm, addr, end));
        }
        0
    }

    /// Whether `[addr, addr + len)` overlaps a sealed range of the caller's
    /// address space (see [`Self::sys_mseal`]).
    pub(super) fn is_sealed(sh: &Shared, cx: &ServiceCtx, addr: u64, len: u64) -> bool {
        let end = addr.saturating_add(len);
        sh.sealed
            .iter()
            .any(|&(mm, s, e)| mm == cx.cur.mm && s < end && addr < e)
    }

    /// `mprotect` that honors `mseal`.
    pub(super) fn sys_mprotect_sealed(
        &self,
        sh: &Shared,
        cx: &ServiceCtx,
        addr: u64,
        len: u64,
        prot: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        if !sh.sealed.is_empty() && Self::is_sealed(sh, cx, addr, len) {
            return err(Errno::EPERM);
        }
        // PR_SET_MDWE: no writable+executable pages, and no gaining exec.
        if cx.cur.pr.mdwe & super::prctl::MDWE_REFUSE_EXEC_GAIN != 0 && prot & 4 != 0 {
            if prot & 2 != 0 {
                return err(Errno::EACCES);
            }
            let mut p = addr - addr % PAGE_SIZE;
            while p < addr.saturating_add(len) {
                if mem.page_prot(p).is_some_and(|q| q.0 & 4 == 0) {
                    return err(Errno::EACCES);
                }
                p += PAGE_SIZE;
            }
        }
        self.sys_mprotect(addr, len, prot, mem)
    }

    /// `membarrier(cmd, flags, cpu_id)`. Every thread of a process runs either
    /// serialized on one host thread or under the address-space lock, so any
    /// requested barrier already holds by the time the call returns; what is
    /// kept faithful is the command set (`QUERY`), the registration rule
    /// (private expedited commands before registering are `EPERM`), and
    /// `GET_REGISTRATIONS`.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_membarrier(
        &self,
        sh: &mut Shared,
        cx: &ServiceCtx,
        cmd: u64,
        flags: u64,
    ) -> i64 {
        const QUERY: u64 = 0;
        const GLOBAL: u64 = 1;
        const GLOBAL_EXPEDITED: u64 = 2;
        const REGISTER_GLOBAL_EXPEDITED: u64 = 4;
        const PRIVATE_EXPEDITED: u64 = 8;
        const REGISTER_PRIVATE_EXPEDITED: u64 = 16;
        const PRIVATE_EXPEDITED_SYNC_CORE: u64 = 32;
        const REGISTER_PRIVATE_EXPEDITED_SYNC_CORE: u64 = 64;
        const PRIVATE_EXPEDITED_RSEQ: u64 = 128;
        const REGISTER_PRIVATE_EXPEDITED_RSEQ: u64 = 256;
        const GET_REGISTRATIONS: u64 = 512;
        const ALL: u64 = 0x3ff;
        const FLAG_CPU: u64 = 1;
        let ok_flags = if cmd == PRIVATE_EXPEDITED_RSEQ {
            FLAG_CPU
        } else {
            0
        };
        if flags & !ok_flags != 0 {
            return err(Errno::EINVAL);
        }
        let reg = sh.membarrier.entry(cx.cur.mm).or_default();
        match cmd {
            QUERY => ALL as i64,
            GLOBAL => 0,
            REGISTER_GLOBAL_EXPEDITED
            | REGISTER_PRIVATE_EXPEDITED
            | REGISTER_PRIVATE_EXPEDITED_SYNC_CORE
            | REGISTER_PRIVATE_EXPEDITED_RSEQ => {
                *reg |= cmd;
                0
            }
            // Each expedited command needs its matching registration (the
            // REGISTER_* bit is the command's bit shifted left once).
            GLOBAL_EXPEDITED
            | PRIVATE_EXPEDITED
            | PRIVATE_EXPEDITED_SYNC_CORE
            | PRIVATE_EXPEDITED_RSEQ => {
                if cmd == GLOBAL_EXPEDITED || *reg & (cmd << 1) != 0 {
                    0
                } else {
                    err(Errno::EPERM)
                }
            }
            GET_REGISTRATIONS => *reg as i64,
            _ => err(Errno::EINVAL),
        }
    }

    /// `rseq(rseq, rseq_len, flags, sig)`: register (or, with
    /// `RSEQ_FLAG_UNREGISTER`, unregister) the thread's restartable-sequence
    /// area. On registration the kernel publishes the current CPU into
    /// `cpu_id_start`/`cpu_id` (and `node_id`/`mm_cid` when the area is large
    /// enough) — glibc's `sched_getcpu` reads it from there. A task is never
    /// migrated mid-slice here, so no critical section is ever aborted and the
    /// published CPU never goes stale (every task reports CPU 0, as `getcpu`).
    #[allow(clippy::unused_self)]
    pub(super) fn sys_rseq(
        &self,
        cx: &mut ServiceCtx,
        ptr: u64,
        len: u64,
        flags: u64,
        sig: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const RSEQ_FLAG_UNREGISTER: u64 = 1;
        const ORIG_LEN: u64 = 32;
        let sig = sig as u32;
        if flags & RSEQ_FLAG_UNREGISTER != 0 {
            if flags != RSEQ_FLAG_UNREGISTER {
                return err(Errno::EINVAL);
            }
            let Some((p, l, s)) = cx.cur.rseq else {
                return err(Errno::EINVAL);
            };
            if p != ptr || u64::from(l) != len {
                return err(Errno::EINVAL);
            }
            if s != sig {
                return err(Errno::EPERM);
            }
            // cpu_id = RSEQ_CPU_ID_UNINITIALIZED (-1); cpu_id_start = 0.
            let mut b = [0u8; 8];
            b[4..8].copy_from_slice(&(-1i32).to_le_bytes());
            let _ = mem.write(ptr, &b);
            cx.cur.rseq = None;
            return 0;
        }
        if flags != 0 {
            return err(Errno::EINVAL);
        }
        if let Some((p, l, s)) = cx.cur.rseq {
            return if p == ptr && u64::from(l) == len && s == sig {
                err(Errno::EBUSY)
            } else {
                err(Errno::EINVAL)
            };
        }
        if len < ORIG_LEN || !ptr.is_multiple_of(ORIG_LEN) || len > 4096 {
            return err(Errno::EINVAL);
        }
        // cpu_id_start, cpu_id = 0 (rseq_cs and flags left as the caller set
        // them); node_id @20 and mm_cid @24 if the area covers them.
        if mem.write(ptr, &[0u8; 8]).is_err() {
            return err(Errno::EFAULT);
        }
        if len >= 28 {
            let _ = mem.write(ptr + 20, &[0u8; 8]);
        }
        cx.cur.rseq = Some((ptr, len as u32, sig));
        0
    }

    /// Decode a capability header (`struct __user_cap_header_struct { u32
    /// version; i32 pid; }`): the number of 32-bit data words per set (1 for
    /// v1, 2 for v2/v3) and the pid. An unknown version gets the preferred one
    /// written back and fails `EINVAL`.
    fn cap_header(mem: &mut GuestMemory, hdr: u64) -> Result<(usize, i32), i64> {
        const V1: u32 = 0x1998_0330;
        const V2: u32 = 0x2007_1026;
        const V3: u32 = 0x2008_0522;
        let (Ok(version), Ok(pid)) = (mem.read_u32(hdr), mem.read_u32(hdr + 4)) else {
            return Err(err(Errno::EFAULT));
        };
        match version {
            V1 => Ok((1, pid as i32)),
            V2 | V3 => Ok((2, pid as i32)),
            _ => {
                let _ = mem.write(hdr, &V3.to_le_bytes());
                Err(err(Errno::EINVAL))
            }
        }
    }

    /// `capget(hdrp, datap)`: root holds every capability (effective and
    /// permitted, none inheritable) until it `capset`s them away or drops its
    /// uid; a non-root task holds none. A NULL `datap` just probes the version.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_capget(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        hdr: u64,
        data: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let (words, pid) = match Self::cap_header(mem, hdr) {
            Ok(v) => v,
            Err(e) if data == 0 && e == err(Errno::EINVAL) => return 0,
            Err(e) => return e,
        };
        if data == 0 {
            return 0;
        }
        if pid < 0 {
            return err(Errno::EINVAL);
        }
        let caps = match with_task(sh, cx, pid as u64, |p| task_caps(p)) {
            Ok(c) => c,
            Err(e) => return e,
        };
        let mut b = Vec::with_capacity(words * 12);
        for w in 0..words {
            for set in caps {
                b.extend_from_slice(&((set >> (32 * w)) as u32).to_le_bytes());
            }
        }
        if mem.write(data, &b).is_err() {
            return err(Errno::EFAULT);
        }
        0
    }

    /// `capset(hdrp, datap)`: set the caller's own capability sets. The new
    /// permitted set may only shrink and effective must stay within it
    /// (`EPERM` otherwise); another pid is `EPERM`, as on modern Linux.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_capset(
        &self,
        cx: &mut ServiceCtx,
        hdr: u64,
        data: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let (words, pid) = match Self::cap_header(mem, hdr) {
            Ok(v) => v,
            Err(e) => return e,
        };
        if pid != 0 && pid != cx.cur.pid {
            return err(Errno::EPERM);
        }
        let mut sets = [0u64; 3];
        for w in 0..words {
            for (i, set) in sets.iter_mut().enumerate() {
                let Ok(v) = mem.read_u32(data + (w * 12 + i * 4) as u64) else {
                    return err(Errno::EFAULT);
                };
                *set |= u64::from(v) << (32 * w);
            }
        }
        let [eff, perm, inh] = sets.map(|s| s & CAP_ALL);
        let [_, old_perm, _] = task_caps(&cx.cur);
        if perm & !old_perm != 0 || eff & !perm != 0 || inh & !(old_perm | inh) != 0 {
            return err(Errno::EPERM);
        }
        cx.cur.caps = Some([eff, perm, inh]);
        0
    }

    /// `adjtimex(buf)` / `clock_adjtime(clk, buf)`: read the kernel clock
    /// discipline. The host owns the clock, so the VM reports a synchronized
    /// clock (`TIME_OK`) and refuses any adjustment (`EPERM`, like
    /// `settimeofday`). Only `CLOCK_REALTIME` is adjustable on Linux; the other
    /// clocks are `EOPNOTSUPP`.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_adjtimex(&self, clk: u64, buf: u64, mem: &mut GuestMemory) -> i64 {
        const ADJ_OFFSET_SS_READ: u32 = 0xa001;
        if clk != 0 {
            return if clk > 11 {
                err(Errno::EINVAL)
            } else {
                err(Errno::EOPNOTSUPP)
            };
        }
        let Ok(modes) = mem.read_u32(buf) else {
            return err(Errno::EFAULT);
        };
        if modes != 0 && modes != ADJ_OFFSET_SS_READ {
            return err(Errno::EPERM);
        }
        let now = crate::clock::now_unix();
        let mut t = [0u8; 208];
        let mut put = |off: usize, v: i64| t[off..off + 8].copy_from_slice(&v.to_le_bytes());
        put(24, 500); // maxerror (µs)
        put(48, 2); // constant
        put(56, 1); // precision (µs)
        put(64, 32_768_000); // tolerance (ppm << 16)
        put(72, now.as_secs() as i64); // time.tv_sec
        put(80, i64::from(now.subsec_micros())); // time.tv_usec
        put(88, 10_000); // tick (µs)
        t[160..164].copy_from_slice(&37i32.to_le_bytes()); // tai offset
        if mem.write(buf, &t).is_err() {
            return err(Errno::EFAULT);
        }
        0 // TIME_OK
    }

    /// `personality(persona)`: `0xffffffff` queries; anything else sets.
    /// Returns the previous persona either way.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_personality(&self, cx: &mut ServiceCtx, persona: u64) -> i64 {
        let old = i64::from(cx.cur.personality);
        if persona as u32 != 0xffff_ffff {
            cx.cur.personality = persona as u32;
        }
        old
    }

    /// `sethostname`/`setdomainname(name, len)`: root-only, at most 64 bytes;
    /// what `uname` reports afterwards.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_setname(
        &self,
        sh: &mut Shared,
        cx: &ServiceCtx,
        name: u64,
        len: u64,
        domain: bool,
        mem: &GuestMemory,
    ) -> i64 {
        if len > 64 {
            return err(Errno::EINVAL);
        }
        if cx.cur.creds.euid != 0 {
            return err(Errno::EPERM);
        }
        let Ok(b) = mem.read_vec(name, len as usize) else {
            return err(Errno::EFAULT);
        };
        let s = String::from_utf8_lossy(&b).into_owned();
        if domain {
            sh.domainname = s;
        } else {
            sh.hostname = s;
        }
        0
    }

    /// `set_robust_list(head, len)` / `get_robust_list(pid, head*, len*)`.
    /// The head is recorded per thread and reported back; `len` must be the
    /// 24-byte `struct robust_list_head`.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_robust_list(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        a: &[u64; 6],
        get: bool,
        mem: &mut GuestMemory,
    ) -> i64 {
        const HEAD_LEN: u64 = 24;
        if !get {
            if a[1] != HEAD_LEN {
                return err(Errno::EINVAL);
            }
            cx.cur.robust_list = a[0];
            return 0;
        }
        let head = match with_task(sh, cx, a[0], |p| p.robust_list) {
            Ok(h) => h,
            Err(e) => return e,
        };
        if mem.write_u64(a[1], head).is_err() || mem.write_u64(a[2], HEAD_LEN).is_err() {
            return err(Errno::EFAULT);
        }
        0
    }

    /// `unshare(flags)`. `CLONE_FILES`/`CLONE_FS` really split a shared fd
    /// table / filesystem context off into a private copy; the namespace flags
    /// are accepted as no-ops for root exactly as `clone` accepts them (one
    /// global namespace of each kind) and refused for a non-root caller
    /// (`EPERM`; a user namespace needs no privilege). `CLONE_THREAD`/
    /// `CLONE_SIGHAND`/`CLONE_VM` are only valid for a single-threaded caller.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_unshare(&self, sh: &mut Shared, cx: &mut ServiceCtx, flags: u64) -> i64 {
        const CLONE_NEWTIME: u64 = 0x80;
        const CLONE_VM: u64 = 0x100;
        const CLONE_FS: u64 = 0x200;
        const CLONE_FILES: u64 = 0x400;
        const CLONE_SIGHAND: u64 = 0x800;
        const CLONE_THREAD: u64 = 0x1_0000;
        const CLONE_NEWNS: u64 = 0x2_0000;
        const CLONE_SYSVSEM: u64 = 0x4_0000;
        const CLONE_NEWCGROUP: u64 = 0x200_0000;
        const CLONE_NEWUTS: u64 = 0x400_0000;
        const CLONE_NEWIPC: u64 = 0x800_0000;
        const CLONE_NEWUSER: u64 = 0x1000_0000;
        const CLONE_NEWPID: u64 = 0x2000_0000;
        const CLONE_NEWNET: u64 = 0x4000_0000;
        const NS: u64 = CLONE_NEWTIME
            | CLONE_NEWNS
            | CLONE_NEWCGROUP
            | CLONE_NEWUTS
            | CLONE_NEWIPC
            | CLONE_NEWPID
            | CLONE_NEWNET;
        let valid = NS
            | CLONE_NEWUSER
            | CLONE_VM
            | CLONE_FS
            | CLONE_FILES
            | CLONE_SIGHAND
            | CLONE_THREAD
            | CLONE_SYSVSEM;
        if flags & !valid != 0 {
            return err(Errno::EINVAL);
        }
        let tgid = cx.cur.tgid;
        let threaded = sh
            .procs
            .iter()
            .flatten()
            .any(|p| p.info.tgid == tgid && !matches!(p.info.run, RunState::Zombie(_)));
        if flags & (CLONE_THREAD | CLONE_SIGHAND | CLONE_VM) != 0 && threaded {
            return err(Errno::EINVAL);
        }
        if flags & NS != 0 && cx.cur.creds.euid != 0 {
            return err(Errno::EPERM);
        }
        // A private fd table: the shared slot (checked out into `cur.fds` for
        // this slice) gets a copy for the siblings; this task moves to a fresh
        // slot its table is checked into at the end of the slice.
        let (files, fs) = (cx.cur.files, cx.cur.fs);
        if flags & CLONE_FILES != 0 && sh.procs.iter().flatten().any(|p| p.info.files == files) {
            let copy = cx.cur.fds.clone();
            for f in copy.values() {
                self.bump_pipe(f, true);
            }
            sh.file_tables[files] = Some(copy);
            cx.cur.files = sh.file_tables.len();
            sh.file_tables.push(None);
        }
        if flags & CLONE_FS != 0 && sh.procs.iter().flatten().any(|p| p.info.fs == fs) {
            sh.cwd_tables[fs] = Some(cx.cur.cwd.clone());
            cx.cur.fs = sh.cwd_tables.len();
            sh.cwd_tables.push(None);
        }
        0
    }

    /// `setns(fd, nstype)`: join the namespaces `fd` names. Only a pidfd can
    /// name namespaces here (there are no `/proc/<pid>/ns/*` files), and every
    /// process shares the one set of namespaces, so joining is a successful
    /// no-op; any other descriptor is not a namespace (`EINVAL`).
    #[allow(clippy::unused_self)]
    pub(super) fn sys_setns(&self, cx: &ServiceCtx, fd: u64, nstype: u64) -> i64 {
        const NS_ALL: u64 = 0x80
            | 0x2_0000
            | 0x200_0000
            | 0x400_0000
            | 0x800_0000
            | 0x1000_0000
            | 0x2000_0000
            | 0x4000_0000;
        if nstype & !NS_ALL != 0 {
            return err(Errno::EINVAL);
        }
        match cx.cur.fds.get(fd as i32) {
            Some(Fd::Pidfd(_)) => 0,
            Some(_) => err(Errno::EINVAL),
            None => err(Errno::EBADF),
        }
    }

    /// x86-64 `ustat(dev, struct ustat *)`: free blocks/inodes of the
    /// filesystem on `dev` — plenty of both, on every device.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_ustat(&self, ubuf: u64, mem: &mut GuestMemory) -> i64 {
        let mut b = [0u8; 32];
        b[0..4].copy_from_slice(&(1i32 << 20).to_le_bytes()); // f_tfree
        b[8..16].copy_from_slice(&(1u64 << 20).to_le_bytes()); // f_tinode
        if mem.write(ubuf, &b).is_err() {
            return err(Errno::EFAULT);
        }
        0
    }

    /// x86-64 `sysfs(option, arg1, arg2)`: the filesystem-type index (1: name
    /// → index, 2: index → name into a buffer, 3: count).
    #[allow(clippy::unused_self)]
    pub(super) fn sys_sysfs(&self, option: u64, a1: u64, a2: u64, mem: &mut GuestMemory) -> i64 {
        const TYPES: [&str; 9] = [
            "sysfs", "tmpfs", "proc", "devtmpfs", "devpts", "overlay", "squashfs", "ext4", "mqueue",
        ];
        match option {
            1 => {
                let Ok(name) = mem.read_cstr(a1, 256) else {
                    return err(Errno::EFAULT);
                };
                TYPES
                    .iter()
                    .position(|t| t.as_bytes() == name.as_slice())
                    .map_or(err(Errno::EINVAL), |i| i as i64)
            }
            2 => {
                let Some(t) = TYPES.get(a1 as usize) else {
                    return err(Errno::EINVAL);
                };
                let mut b = t.as_bytes().to_vec();
                b.push(0);
                if mem.write(a2, &b).is_err() {
                    return err(Errno::EFAULT);
                }
                0
            }
            3 => TYPES.len() as i64,
            _ => err(Errno::EINVAL),
        }
    }

    /// x86-64 `iopl(level)` / `ioperm(from, num, on)`: lowering privilege is
    /// always allowed; raising it needs `CAP_SYS_RAWIO` over real I/O ports,
    /// which an emulated machine can't grant — refused as for an unprivileged
    /// caller (`EPERM`).
    #[allow(clippy::unused_self)]
    pub(super) fn sys_ioport(&self, a: &[u64; 6], ioperm: bool) -> i64 {
        if ioperm {
            let (from, num, on) = (a[0], a[1], a[2]);
            if from.saturating_add(num) > 65536 {
                return err(Errno::EINVAL);
            }
            return if on != 0 { err(Errno::EPERM) } else { 0 };
        }
        match a[0] {
            0 => 0,
            1..=3 => err(Errno::EPERM),
            _ => err(Errno::EINVAL),
        }
    }
}

/// `[effective, permitted, inheritable]` for a task: what `capset` stored, or
/// root's full set / nothing for a non-root task.
fn task_caps(p: &ProcInfo) -> [u64; 3] {
    p.caps.unwrap_or(if p.creds.euid == 0 {
        [CAP_ALL, CAP_ALL, 0]
    } else {
        [0, 0, 0]
    })
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{BASE, call, e, setup};
    use crate::abi::arch::Sysno;
    use crate::abi::errno::Errno;

    #[test]
    fn sched_attr_param_and_rr_interval_roundtrip() {
        let (k, mut mem, mut v, mut cx) = setup();
        let (attr, out) = (BASE, BASE + 0x100);
        // SCHED_FIFO prio 10 through sched_setattr.
        let mut a = [0u8; 48];
        a[0..4].copy_from_slice(&48u32.to_le_bytes());
        a[4..8].copy_from_slice(&1u32.to_le_bytes());
        a[20..24].copy_from_slice(&10u32.to_le_bytes());
        mem.write(attr, &a).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::SchedSetattr,
                [0, attr, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::SchedGetscheduler,
                [0; 6]
            ),
            1
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::SchedGetattr,
                [0, out, 56, 0, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_u32(out).unwrap(), 56);
        assert_eq!(mem.read_u32(out + 20).unwrap(), 10);
        // An out-of-range priority for FIFO is EINVAL; sched_setparam too.
        mem.write(out, &0u32.to_le_bytes()).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::SchedSetparam,
                [0, out, 0, 0, 0, 0]
            ),
            e(Errno::EINVAL)
        );
        mem.write(out, &50u32.to_le_bytes()).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::SchedSetparam,
                [0, out, 0, 0, 0, 0]
            ),
            0
        );
        // A short struct is E2BIG with the expected size written back.
        mem.write(attr, &8u32.to_le_bytes()).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::SchedSetattr,
                [0, attr, 0, 0, 0, 0]
            ),
            e(Errno::E2BIG)
        );
        assert_eq!(mem.read_u32(attr).unwrap(), 56);
        // FIFO has no RR quantum.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::SchedRrGetInterval,
                [0, out, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_u64(out + 8).unwrap(), 0);
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::SchedRrGetInterval,
                [42, out, 0, 0, 0, 0]
            ),
            e(Errno::ESRCH)
        );
    }

    #[test]
    fn ioprio_numa_and_pkeys() {
        let (k, mut mem, mut v, mut cx) = setup();
        // Default ioprio: best-effort level 4 (nice 0).
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::IoprioGet,
                [1, 0, 0, 0, 0, 0]
            ),
            (2 << 13) | 4
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::IoprioSet,
                [1, 0, (3 << 13), 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::IoprioGet,
                [1, 0, 0, 0, 0, 0]
            ),
            3 << 13
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::IoprioSet,
                [9, 0, 0, 0, 0, 0]
            ),
            e(Errno::EINVAL)
        );
        // NUMA: bind to node 0 works, node 1 doesn't exist.
        let (mask, mode) = (BASE, BASE + 0x10);
        mem.write_u64(mask, 1).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::SetMempolicy,
                [2, mask, 65, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::GetMempolicy,
                [mode, mask, 65, 0, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_u32(mode).unwrap(), 2);
        mem.write_u64(mask, 2).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::SetMempolicy,
                [2, mask, 65, 0, 0, 0]
            ),
            e(Errno::EINVAL)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::SetMempolicy,
                [0, 0, 0, 0, 0, 0]
            ),
            0
        );
        // MEMS_ALLOWED reports node 0.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::GetMempolicy,
                [0, mask, 65, 0, 4, 0]
            ),
            0
        );
        assert_eq!(mem.read_u64(mask).unwrap(), 1);
        // mbind over a mapped page: OK; unaligned: EINVAL; unmapped: EFAULT.
        mem.write_u64(mask, 1).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Mbind,
                [BASE, 4096, 2, mask, 65, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Mbind,
                [BASE + 1, 4096, 2, mask, 65, 0]
            ),
            e(Errno::EINVAL)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Mbind,
                [BASE + 0x20_000, 4096, 2, mask, 65, 0]
            ),
            e(Errno::EFAULT)
        );
        // pkeys: none allocatable; key -1/0 is plain mprotect.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::PkeyAlloc,
                [0, 0, 0, 0, 0, 0]
            ),
            e(Errno::ENOSPC)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::PkeyAlloc,
                [1, 0, 0, 0, 0, 0]
            ),
            e(Errno::EINVAL)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::PkeyMprotect,
                [BASE, 4096, 3, u64::MAX, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::PkeyMprotect,
                [BASE, 4096, 3, 5, 0, 0]
            ),
            e(Errno::EINVAL)
        );
    }

    #[test]
    fn mseal_blocks_later_changes() {
        let (k, mut mem, mut v, mut cx) = setup();
        let page = BASE + 4 * 4096;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Mseal,
                [page, 4096, 1, 0, 0, 0]
            ),
            e(Errno::EINVAL)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Mseal,
                [BASE + 0x40_000, 4096, 0, 0, 0, 0]
            ),
            e(Errno::ENOMEM)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Mseal,
                [page, 4096, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Mprotect,
                [page, 4096, 1, 0, 0, 0]
            ),
            e(Errno::EPERM)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Munmap,
                [page, 4096, 0, 0, 0, 0]
            ),
            e(Errno::EPERM)
        );
        // A neighbouring page is unaffected.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Mprotect,
                [page - 4096, 4096, 3, 0, 0, 0]
            ),
            0
        );
    }

    #[test]
    fn membarrier_rseq_and_capabilities() {
        let (k, mut mem, mut v, mut cx) = setup();
        let q = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Membarrier,
            [0, 0, 0, 0, 0, 0],
        );
        assert_ne!(q & 8, 0, "PRIVATE_EXPEDITED supported");
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Membarrier,
                [8, 0, 0, 0, 0, 0]
            ),
            e(Errno::EPERM)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Membarrier,
                [16, 0, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Membarrier,
                [8, 0, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Membarrier,
                [512, 0, 0, 0, 0, 0]
            ),
            16
        );
        // rseq: register publishes cpu 0; re-register is EBUSY; unregister
        // with the wrong signature is EPERM.
        let area = BASE + 0x200;
        mem.write(area, &[0xffu8; 32]).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Rseq,
                [area, 32, 0, 0x5305_3053, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_u32(area + 4).unwrap(), 0);
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Rseq,
                [area, 32, 0, 0x5305_3053, 0, 0]
            ),
            e(Errno::EBUSY)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Rseq,
                [area, 32, 1, 7, 0, 0]
            ),
            e(Errno::EPERM)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Rseq,
                [area, 32, 1, 0x5305_3053, 0, 0]
            ),
            0
        );
        // capget: root has every capability; a version probe gets v3.
        let (hdr, data) = (BASE + 0x300, BASE + 0x400);
        mem.write(hdr, &[0u8; 8]).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Capget,
                [hdr, 0, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_u32(hdr).unwrap(), 0x2008_0522);
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Capget,
                [hdr, data, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_u32(data).unwrap(), u32::MAX, "effective, low word");
        assert_eq!(
            mem.read_u32(data + 12).unwrap(),
            0x1ff,
            "effective, high word"
        );
        // Drop everything but CAP_NET_RAW (13), then try to regain: EPERM.
        let mut d = [0u8; 24];
        d[0..4].copy_from_slice(&(1u32 << 13).to_le_bytes());
        d[4..8].copy_from_slice(&(1u32 << 13).to_le_bytes());
        mem.write(data, &d).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Capset,
                [hdr, data, 0, 0, 0, 0]
            ),
            0
        );
        d[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
        mem.write(data, &d).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Capset,
                [hdr, data, 0, 0, 0, 0]
            ),
            e(Errno::EPERM)
        );
    }

    #[test]
    fn adjtimex_reads_but_never_sets_and_names_reach_uname() {
        let (k, mut mem, mut v, mut cx) = setup();
        let buf = BASE;
        mem.write(buf, &[0u8; 208]).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Adjtimex,
                [buf, 0, 0, 0, 0, 0]
            ),
            0
        );
        assert!(
            mem.read_u64(buf + 72).unwrap() > 1_600_000_000,
            "time.tv_sec is now"
        );
        mem.write(buf, &1u32.to_le_bytes()).unwrap(); // ADJ_OFFSET
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Adjtimex,
                [buf, 0, 0, 0, 0, 0]
            ),
            e(Errno::EPERM)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::ClockAdjtime,
                [1, buf, 0, 0, 0, 0]
            ),
            e(Errno::EOPNOTSUPP)
        );
        // sethostname shows up in uname's nodename (field 1).
        mem.write(BASE + 0x100, b"box").unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Sethostname,
                [BASE + 0x100, 3, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Uname,
                [BASE + 0x200, 0, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_vec(BASE + 0x200 + 65, 4).unwrap(), b"box\0");
        // personality: set, then query returns it.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Personality,
                [0x0040_0000, 0, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Personality,
                [0xffff_ffff, 0, 0, 0, 0, 0]
            ),
            0x0040_0000
        );
    }

    #[test]
    fn groups_mounts_and_syslog() {
        use super::super::testutil::put_str;
        let (k, mut mem, mut v, mut cx) = setup();
        // Supplementary groups round-trip; a too-small getgroups is EINVAL.
        mem.write(BASE, &[5, 0, 0, 0, 7, 0, 0, 0]).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Setgroups,
                [2, BASE, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Getgroups,
                [0, 0, 0, 0, 0, 0]
            ),
            2
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Getgroups,
                [1, BASE + 0x10, 0, 0, 0, 0]
            ),
            e(Errno::EINVAL)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Getgroups,
                [8, BASE + 0x10, 0, 0, 0, 0]
            ),
            2
        );
        assert_eq!(mem.read_u32(BASE + 0x14).unwrap(), 7);
        // mount -t tmpfs over a directory, write in it, umount: gone.
        let (dir, ty, file) = (BASE + 0x100, BASE + 0x200, BASE + 0x300);
        put_str(&mut mem, dir, "/mnt");
        put_str(&mut mem, ty, "tmpfs");
        put_str(&mut mem, file, "/mnt/f");
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Mkdirat,
                [(-100i64) as u64, dir, 0o755, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Mount,
                [0, dir, ty, 0, 0, 0]
            ),
            0
        );
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Openat,
            [(-100i64) as u64, file, 0o102, 0o600, 0, 0],
        );
        assert!(fd >= 0);
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Umount2,
                [dir, 0, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Openat,
                [(-100i64) as u64, file, 0, 0, 0, 0]
            ),
            e(Errno::ENOENT)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Umount2,
                [dir, 0, 0, 0, 0, 0]
            ),
            e(Errno::EINVAL)
        );
        // A block-device filesystem has nothing to mount.
        put_str(&mut mem, ty, "ext4");
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Mount,
                [0, dir, ty, 0, 0, 0]
            ),
            e(Errno::ENODEV)
        );
        // syslog: the buffer size, no unknown actions.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Syslog,
                [10, 0, 0, 0, 0, 0]
            ),
            1 << 17
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Syslog,
                [11, 0, 0, 0, 0, 0]
            ),
            e(Errno::EINVAL)
        );
    }
}
