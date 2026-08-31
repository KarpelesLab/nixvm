//! Resource, scheduling, and process-attribute syscalls.
//!
//! These are mostly informational: they return success and/or write plausible
//! zeroed/static structs into guest memory. The guest's libc queries them at
//! startup (rlimits, cpu affinity, scheduler class, process name) and mostly
//! ignores the exact values, so a believable constant answer is enough.

use super::{Kernel, ServiceCtx, Shared, err};
use crate::abi::errno::Errno;
use crate::vcpu::GuestMemory;

/// A `set*id` argument's "leave unchanged" sentinel is `(uid_t)-1`, which the
/// guest passes zero-extended as `0xFFFF_FFFF`. Returns the new id, or `None`
/// to leave the field alone.
fn opt_id(v: u64) -> Option<u32> {
    let v = v as u32;
    (v != u32::MAX).then_some(v)
}

impl Kernel {
    /// `setuid(uid)`: a privileged process (euid 0) sets the real, effective,
    /// saved, and fs uid all to `uid`; an unprivileged one may only set the
    /// effective (and fs) uid to its real or saved uid, else `EPERM`.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_setuid(&self, cx: &mut ServiceCtx, uid: u32) -> i64 {
        let c = &mut cx.cur.creds;
        if c.euid == 0 {
            c.ruid = uid;
            c.euid = uid;
            c.suid = uid;
            c.fsuid = uid;
            0
        } else if uid == c.ruid || uid == c.suid {
            c.euid = uid;
            c.fsuid = uid;
            0
        } else {
            err(Errno::EPERM)
        }
    }

    /// `setgid(gid)` — the group analogue of [`Self::sys_setuid`] (privilege is
    /// still determined by the *user* euid).
    #[allow(clippy::unused_self)]
    pub(super) fn sys_setgid(&self, cx: &mut ServiceCtx, gid: u32) -> i64 {
        let privileged = cx.cur.creds.euid == 0;
        let c = &mut cx.cur.creds;
        if privileged {
            c.rgid = gid;
            c.egid = gid;
            c.sgid = gid;
            c.fsgid = gid;
            0
        } else if gid == c.rgid || gid == c.sgid {
            c.egid = gid;
            c.fsgid = gid;
            0
        } else {
            err(Errno::EPERM)
        }
    }

    /// `setresuid(ruid, euid, suid)` — set each id independently (`-1` leaves it
    /// unchanged). Privileged sets any; unprivileged only among its current
    /// real/effective/saved uids.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_setresuid(&self, cx: &mut ServiceCtx, r: u64, e: u64, s: u64) -> i64 {
        let c = cx.cur.creds;
        let privileged = c.euid == 0;
        let ok = |id: u32| privileged || id == c.ruid || id == c.euid || id == c.suid;
        for id in [opt_id(r), opt_id(e), opt_id(s)].into_iter().flatten() {
            if !ok(id) {
                return err(Errno::EPERM);
            }
        }
        let c = &mut cx.cur.creds;
        if let Some(v) = opt_id(r) {
            c.ruid = v;
        }
        if let Some(v) = opt_id(e) {
            c.euid = v;
            c.fsuid = v;
        }
        if let Some(v) = opt_id(s) {
            c.suid = v;
        }
        0
    }

    /// `setresgid(rgid, egid, sgid)` — the group analogue.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_setresgid(&self, cx: &mut ServiceCtx, r: u64, e: u64, s: u64) -> i64 {
        let c = cx.cur.creds;
        let privileged = c.euid == 0;
        let ok = |id: u32| privileged || id == c.rgid || id == c.egid || id == c.sgid;
        for id in [opt_id(r), opt_id(e), opt_id(s)].into_iter().flatten() {
            if !ok(id) {
                return err(Errno::EPERM);
            }
        }
        let c = &mut cx.cur.creds;
        if let Some(v) = opt_id(r) {
            c.rgid = v;
        }
        if let Some(v) = opt_id(e) {
            c.egid = v;
            c.fsgid = v;
        }
        if let Some(v) = opt_id(s) {
            c.sgid = v;
        }
        0
    }

    /// `setreuid(ruid, euid)` — set real and/or effective uid (`-1` leaves one
    /// unchanged); the saved uid follows the effective uid when either changes.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_setreuid(&self, cx: &mut ServiceCtx, r: u64, e: u64) -> i64 {
        let c = cx.cur.creds;
        let privileged = c.euid == 0;
        if let Some(r) = opt_id(r)
            && !(privileged || r == c.ruid || r == c.euid)
        {
            return err(Errno::EPERM);
        }
        if let Some(e) = opt_id(e)
            && !(privileged || e == c.ruid || e == c.euid || e == c.suid)
        {
            return err(Errno::EPERM);
        }
        let nc = &mut cx.cur.creds;
        if let Some(r) = opt_id(r) {
            nc.ruid = r;
        }
        if let Some(e) = opt_id(e) {
            nc.euid = e;
            nc.fsuid = e;
        }
        // The saved uid tracks the effective uid when the real uid is set or the
        // effective uid is changed to something other than the old real uid.
        if opt_id(r).is_some() || opt_id(e).is_some_and(|e| e != c.ruid) {
            nc.suid = nc.euid;
        }
        0
    }

    /// `setregid(rgid, egid)` — the group analogue of [`Self::sys_setreuid`].
    #[allow(clippy::unused_self)]
    pub(super) fn sys_setregid(&self, cx: &mut ServiceCtx, r: u64, e: u64) -> i64 {
        let c = cx.cur.creds;
        let privileged = c.euid == 0;
        if let Some(r) = opt_id(r)
            && !(privileged || r == c.rgid || r == c.egid)
        {
            return err(Errno::EPERM);
        }
        if let Some(e) = opt_id(e)
            && !(privileged || e == c.rgid || e == c.egid || e == c.sgid)
        {
            return err(Errno::EPERM);
        }
        let nc = &mut cx.cur.creds;
        if let Some(r) = opt_id(r) {
            nc.rgid = r;
        }
        if let Some(e) = opt_id(e) {
            nc.egid = e;
            nc.fsgid = e;
        }
        if opt_id(r).is_some() || opt_id(e).is_some_and(|e| e != c.rgid) {
            nc.sgid = nc.egid;
        }
        0
    }

    /// `setfsuid(fsuid)` / `setfsgid(fsgid)` — set the filesystem id, returning
    /// the *previous* one. Allowed if privileged or the id matches the caller's
    /// real/effective/saved/current-fs id; otherwise the change is silently
    /// ignored (the syscall never fails, it just returns the old value).
    #[allow(clippy::unused_self)]
    pub(super) fn sys_setfsuid(&self, cx: &mut ServiceCtx, fsuid: u64) -> i64 {
        let c = &mut cx.cur.creds;
        let old = c.fsuid;
        if let Some(v) = opt_id(fsuid)
            && (c.euid == 0 || v == c.ruid || v == c.euid || v == c.suid || v == c.fsuid)
        {
            c.fsuid = v;
        }
        i64::from(old)
    }

    /// `setfsgid(fsgid)` — the group analogue of [`Self::sys_setfsuid`].
    #[allow(clippy::unused_self)]
    pub(super) fn sys_setfsgid(&self, cx: &mut ServiceCtx, fsgid: u64) -> i64 {
        let c = &mut cx.cur.creds;
        let old = c.fsgid;
        if let Some(v) = opt_id(fsgid)
            && (c.euid == 0 || v == c.rgid || v == c.egid || v == c.sgid || v == c.fsgid)
        {
            c.fsgid = v;
        }
        i64::from(old)
    }

    /// `sched_get_priority_min`/`_max(policy)` — the valid real-time priority
    /// range. `SCHED_FIFO`(1)/`SCHED_RR`(2) use `1..=99`; the others (OTHER/
    /// BATCH/IDLE) have no RT priority, so both bounds are 0.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_sched_priority_bound(&self, policy: u64, max: bool) -> i64 {
        match policy {
            1 | 2 => i64::from(max) * 98 + 1, // 1 (min) or 99 (max)
            _ => 0,
        }
    }

    /// `sched_setscheduler(pid, policy, param)` — record the policy and its
    /// real-time priority (self only; other pids aren't modeled). Reported back
    /// by `sched_getscheduler`/`sched_getparam`.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_sched_setscheduler(&self, cx: &mut ServiceCtx, policy: i32, param: u64, mem: &GuestMemory) -> i64 {
        cx.cur.sched_policy = policy;
        if param != 0
            && let Ok(prio) = mem.read_u32(param)
        {
            cx.cur.sched_priority = prio as i32;
        }
        0
    }

    /// `setpriority(which, who, prio)` — record the nice value (clamped to
    /// −20..=19). `getpriority` reports it back as the kernel ABI `20 - nice`.
    /// Only `PRIO_PROCESS` on self is modeled; other targets succeed as a no-op.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_setpriority(&self, cx: &mut ServiceCtx, prio: i64) -> i64 {
        cx.cur.nice = (prio as i32).clamp(-20, 19);
        0
    }

    /// `sched_setaffinity(pid, cpusetsize, mask)` — record the affinity mask
    /// (self only). `sched_getaffinity` reports it back.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_sched_setaffinity(&self, cx: &mut ServiceCtx, size: u64, mask: u64, mem: &GuestMemory) -> i64 {
        let n = (size as usize).min(8);
        if n == 0 {
            return err(Errno::EINVAL);
        }
        let mut bytes = [0u8; 8];
        let Ok(v) = mem.read_vec(mask, n) else {
            return err(Errno::EFAULT);
        };
        bytes[..n].copy_from_slice(&v);
        let m = u64::from_le_bytes(bytes);
        if m == 0 {
            return err(Errno::EINVAL); // an empty set is rejected
        }
        cx.cur.affinity = m;
        0
    }

    /// `prctl(option, ...)` — process-attribute get/set. We model the process
    /// name (`PR_SET_NAME`/`PR_GET_NAME`, stored on the kernel) and treat every
    /// other option as a successful no-op.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_prctl(&self, cx: &mut ServiceCtx, args: &[u64; 6], mem: &mut GuestMemory) -> i64 {
        const PR_SET_PDEATHSIG: u64 = 1;
        const PR_GET_PDEATHSIG: u64 = 2;
        const PR_SET_NAME: u64 = 15;
        const PR_GET_NAME: u64 = 16;
        const PR_SET_DUMPABLE: u64 = 4;
        const PR_GET_DUMPABLE: u64 = 3;
        const PR_SET_NO_NEW_PRIVS: u64 = 38;
        const PR_GET_NO_NEW_PRIVS: u64 = 39;
        match args[0] {
            PR_SET_NAME => {
                if let Ok(name) = mem.read_cstr(args[1], 16) {
                    let n = name.len().min(15);
                    cx.cur.comm = String::from_utf8_lossy(&name[..n]).into_owned();
                }
                0
            }
            PR_GET_NAME => {
                // The kernel writes a fixed 16-byte, NUL-padded buffer.
                let mut buf = [0u8; 16];
                let b = cx.cur.comm.as_bytes();
                let n = b.len().min(15);
                buf[..n].copy_from_slice(&b[..n]);
                if mem.write(args[1], &buf).is_err() {
                    return err(Errno::EFAULT);
                }
                0
            }
            // no_new_privs is a set-once latch a re-check must see.
            PR_SET_NO_NEW_PRIVS => {
                cx.cur.no_new_privs = args[1] != 0;
                0
            }
            PR_GET_NO_NEW_PRIVS => i64::from(cx.cur.no_new_privs),
            // pdeathsig: store and report (delivery-on-parent-death not yet wired).
            PR_SET_PDEATHSIG => {
                cx.cur.pdeathsig = args[1];
                0
            }
            PR_GET_PDEATHSIG => {
                if mem.write(args[1], &(cx.cur.pdeathsig as i32).to_le_bytes()).is_err() {
                    return err(Errno::EFAULT);
                }
                0
            }
            // dumpable: store and report (we never emit cores, but sandboxes
            // toggle it and re-read the value, so it must round-trip). Valid: 0/1/2.
            PR_SET_DUMPABLE => {
                if args[1] > 2 {
                    return err(Errno::EINVAL);
                }
                cx.cur.dumpable = args[1];
                0
            }
            #[allow(clippy::cast_possible_wrap)]
            PR_GET_DUMPABLE => cx.cur.dumpable as i64,
            // PR_CAPBSET_READ, PR_SET_SECCOMP, PR_SET_CHILD_SUBREAPER, … : accepted no-ops.
            _ => 0,
        }
    }
}

/// `sched_getaffinity(pid, size, mask)` — report a single online CPU (bit 0),
/// returning the number of bytes written (`min(size, 8)`).
pub(super) fn sys_sched_getaffinity(bits: u64, size: u64, mask: u64, mem: &mut GuestMemory) -> i64 {
    let n = size.min(8) as usize;
    if n == 0 {
        return err(Errno::EINVAL);
    }
    // Report the task's affinity mask (its set, or the default all-CPUs set).
    let buf = bits.to_le_bytes();
    if mem.write(mask, &buf[..n]).is_err() {
        return err(Errno::EFAULT);
    }
    n as i64
}

/// `sched_getparam(pid, param)` — write a `sched_param { sched_priority = 0 }`.
pub(super) fn sys_sched_getparam(priority: i32, param: u64, mem: &mut GuestMemory) -> i64 {
    if mem.write(param, &priority.to_le_bytes()).is_err() {
        return err(Errno::EFAULT);
    }
    0
}

impl Kernel {
    /// `getrusage(who, buf)` — report the CPU time consumed so far in a `struct
    /// rusage` (144 bytes). The CPU time comes from the scheduler's per-task
    /// accounting — the *same* source as `clock_gettime(CLOCK_PROCESS_CPUTIME_ID)`
    /// so the two agree — landing in `ru_utime` (we don't split guest user vs.
    /// system time); the remaining counters (maxrss, faults, context switches)
    /// stay zero. `RUSAGE_CHILDREN` reports nothing (child accounting isn't
    /// tracked). Called with `sh` held (the B1 dispatch table).
    pub(super) fn sys_getrusage(&self, sh: &Shared, cx: &ServiceCtx, who: u64, buf: u64, mem: &mut GuestMemory) -> i64 {
        // `who` is a 32-bit `int`: RUSAGE_SELF(0), RUSAGE_CHILDREN(-1),
        // RUSAGE_THREAD(1). Read it via `as i32` so a `-1` the guest passed as a
        // zero-extended `0xFFFF_FFFF` is recognized.
        let cpu_ns = match who as i32 {
            1 => cx.cur.cpu_ns,             // RUSAGE_THREAD
            -1 => cx.cur.child_cpu_ns,      // RUSAGE_CHILDREN: reaped children's CPU
            _ => super::process_cpu_ns(sh, cx), // RUSAGE_SELF (0) and anything else
        };
        if mem.write(buf, &super::rusage_bytes(cpu_ns)).is_err() {
            return err(Errno::EFAULT);
        }
        0
    }

    /// `times(buf)` — report process CPU time in a `struct tms` (4 × i64 clock
    /// ticks) and return elapsed real time in ticks. Ticks are `USER_HZ` = 100 Hz
    /// (10 ms), as on Linux. `tms_utime` carries the process CPU time (the same
    /// per-task accounting as `getrusage`/`clock_gettime`); system and children
    /// fields are zero. Called with `sh` held.
    pub(super) fn sys_times(&self, sh: &Shared, cx: &ServiceCtx, buf: u64, mem: &mut GuestMemory) -> i64 {
        const TICK_NS: u128 = 10_000_000; // 1 tick = 10 ms (USER_HZ = 100)
        let mut tms = [0u8; 32];
        tms[0..8].copy_from_slice(&((super::process_cpu_ns(sh, cx) / TICK_NS) as i64).to_le_bytes()); // tms_utime
        // tms_stime stays zero; tms_cutime carries reaped children's CPU.
        tms[16..24].copy_from_slice(&((cx.cur.child_cpu_ns / TICK_NS) as i64).to_le_bytes()); // tms_cutime
        // tms_cstime stays zero.
        if mem.write(buf, &tms).is_err() {
            return err(Errno::EFAULT);
        }
        (crate::clock::now_monotonic().as_nanos() / TICK_NS) as i64 // real elapsed ticks
    }
}

/// `sysinfo(buf)` — write a `struct sysinfo` with 2 GiB total RAM, one process,
/// and `mem_unit = 1`.
pub(super) fn sys_sysinfo(buf: u64, mem: &mut GuestMemory) -> i64 {
    let data = encode_sysinfo();
    if mem.write(buf, &data).is_err() {
        return err(Errno::EFAULT);
    }
    0
}

/// `getcpu(cpu, node, tcache)` — always CPU 0 / NUMA node 0.
pub(super) fn sys_getcpu(cpu: u64, node: u64, mem: &mut GuestMemory) -> i64 {
    if cpu != 0 && mem.write(cpu, &0u32.to_le_bytes()).is_err() {
        return err(Errno::EFAULT);
    }
    if node != 0 && mem.write(node, &0u32.to_le_bytes()).is_err() {
        return err(Errno::EFAULT);
    }
    0
}

/// `capget(hdrp, datap)` — report an empty capability set.
pub(super) fn sys_capget(datap: u64, mem: &mut GuestMemory) -> i64 {
    if datap == 0 {
        return 0;
    }
    // Two `__user_cap_data_struct` entries (version 3), all bits clear.
    let zeros = [0u8; 24];
    if mem.write(datap, &zeros).is_err() {
        return err(Errno::EFAULT);
    }
    0
}

pub(super) const RLIMIT_NOFILE: u64 = 7;
/// The largest `RLIMIT_NOFILE` hard limit we'll grant. This bounds the fd
/// space a program believes it has — node/V8 binary-search-raise it and then
/// loop over `[0, soft)` marking every fd cloexec, so an unbounded raise turns
/// startup into a million-iteration spin.
pub(super) const NOFILE_HARD_CAP: u64 = 4096;

/// The constant soft/hard limit pair we report for the resources with a fixed
/// value (everything except `RLIMIT_NOFILE`, which the kernel tracks).
pub(super) fn rlimit_for(resource: u64) -> (u64, u64) {
    const RLIMIT_STACK: u64 = 3;
    const RLIMIT_NPROC: u64 = 6;
    const RLIM_INFINITY: u64 = u64::MAX;
    match resource {
        RLIMIT_STACK => (8 * 1024 * 1024, RLIM_INFINITY),
        RLIMIT_NPROC => (4096, 4096),
        _ => (RLIM_INFINITY, RLIM_INFINITY),
    }
}

/// Read a `struct rlimit { rlim_cur, rlim_max }` (2 x u64) from `addr`.
pub(super) fn read_rlimit(mem: &GuestMemory, addr: u64) -> Option<(u64, u64)> {
    Some((mem.read_u64(addr).ok()?, mem.read_u64(addr + 8).ok()?))
}

/// Write a `struct rlimit { rlim_cur, rlim_max }` (2 x u64) at `addr`.
pub(super) fn write_rlimit(mem: &mut GuestMemory, addr: u64, cur: u64, max: u64) -> i64 {
    let mut b = [0u8; 16];
    b[0..8].copy_from_slice(&cur.to_le_bytes());
    b[8..16].copy_from_slice(&max.to_le_bytes());
    if mem.write(addr, &b).is_err() {
        err(Errno::EFAULT)
    } else {
        0
    }
}

/// Encode a 64-bit `struct sysinfo` (112 bytes): 2 GiB total RAM at offset 32,
/// `procs = 1` at offset 80, `mem_unit = 1` at offset 104; everything else 0.
fn encode_sysinfo() -> [u8; 112] {
    const TOTAL_RAM: u64 = 2 * 1024 * 1024 * 1024;
    let mut b = [0u8; 112];
    // uptime (offset 0): seconds the VM has been running.
    b[0..8].copy_from_slice(&(crate::clock::now_monotonic().as_secs() as i64).to_le_bytes());
    b[32..40].copy_from_slice(&TOTAL_RAM.to_le_bytes()); // totalram
    // freeram (offset 40): report most of RAM free — 0 would make memory-sizing
    // programs (JVMs, databases) think the machine is out of memory.
    b[40..48].copy_from_slice(&(TOTAL_RAM * 3 / 4).to_le_bytes());
    b[80..82].copy_from_slice(&1u16.to_le_bytes()); // procs
    b[104..108].copy_from_slice(&1u32.to_le_bytes()); // mem_unit
    b
}

#[cfg(test)]
mod tests {
    use super::{Errno, Kernel, ServiceCtx, err, sys_sched_getaffinity, sys_sysinfo};
    use crate::abi::Arch;
    use crate::fs::{MountTable, TmpFs};
    use crate::vcpu::GuestMemory;
    use crate::vcpu::mem::{PAGE_SIZE, Prot};

    fn setup() -> (Kernel, GuestMemory) {
        let mut mounts = MountTable::new();
        mounts.mount("/", Box::new(TmpFs::new()));
        let kernel = Kernel::new(Arch::Aarch64, mounts);
        let mut mem = GuestMemory::new(0x1_0000, 16 * PAGE_SIZE);
        mem.map(0x1_0000, 4 * PAGE_SIZE, Prot::rw()).unwrap();
        (kernel, mem)
    }

    #[test]
    fn sysinfo_writes_totalram() {
        let (_k, mut mem) = setup();
        let buf = 0x1_0000;
        assert_eq!(sys_sysinfo(buf, &mut mem), 0);
        assert_eq!(mem.read_u64(buf + 32).unwrap(), 2 * 1024 * 1024 * 1024);
        assert_eq!(mem.read_u32(buf + 104).unwrap(), 1);
    }

    #[test]
    fn sched_getaffinity_writes_the_mask() {
        let (_k, mut mem) = setup();
        let mask = 0x1_0000;
        // The effective mask bits are written out (the dispatcher passes the
        // task's mask, or the default all-CPUs set).
        assert_eq!(sys_sched_getaffinity(0b1, 128, mask, &mut mem), 8);
        assert_eq!(mem.read_vec(mask, 1).unwrap()[0], 1);
        assert_eq!(sys_sched_getaffinity(0xf, 128, mask, &mut mem), 8);
        assert_eq!(mem.read_vec(mask, 1).unwrap()[0], 0xf);
    }

    #[test]
    fn prctl_set_get_name_roundtrips() {
        let (k, mut mem) = setup();
        let mut cx = ServiceCtx::for_test();
        let name = 0x1_0000;
        mem.write_init(name, b"myproc\0").unwrap();
        assert_eq!(k.sys_prctl(&mut cx, &[15, name, 0, 0, 0, 0], &mut mem), 0);
        let out = 0x1_1000;
        assert_eq!(k.sys_prctl(&mut cx, &[16, out, 0, 0, 0, 0], &mut mem), 0);
        assert_eq!(mem.read_vec(out, 6).unwrap(), b"myproc");
    }

    #[test]
    fn prctl_no_new_privs_and_pdeathsig_latch() {
        let (k, mut mem) = setup();
        let mut cx = ServiceCtx::for_test();
        // no_new_privs: set-once latch a re-check must observe.
        assert_eq!(k.sys_prctl(&mut cx, &[38, 1, 0, 0, 0, 0], &mut mem), 0);
        assert_eq!(k.sys_prctl(&mut cx, &[39, 0, 0, 0, 0, 0], &mut mem), 1);
        // pdeathsig: store and report.
        assert_eq!(k.sys_prctl(&mut cx, &[1, 15, 0, 0, 0, 0], &mut mem), 0);
        let out = 0x1_2000;
        assert_eq!(k.sys_prctl(&mut cx, &[2, out, 0, 0, 0, 0], &mut mem), 0);
        assert_eq!(mem.read_vec(out, 4).unwrap(), 15i32.to_le_bytes());
    }

    #[test]
    fn prctl_dumpable_roundtrips() {
        let (k, mut mem) = setup();
        let mut cx = ServiceCtx::for_test();
        // Default is SUID_DUMP_USER (1).
        assert_eq!(k.sys_prctl(&mut cx, &[3, 0, 0, 0, 0, 0], &mut mem), 1);
        // Set 0 (not dumpable) → reads back 0, not a hardcoded 1.
        assert_eq!(k.sys_prctl(&mut cx, &[4, 0, 0, 0, 0, 0], &mut mem), 0);
        assert_eq!(k.sys_prctl(&mut cx, &[3, 0, 0, 0, 0, 0], &mut mem), 0);
        // Out-of-range set is rejected.
        assert_eq!(k.sys_prctl(&mut cx, &[4, 3, 0, 0, 0, 0], &mut mem), err(Errno::EINVAL));
    }
}
