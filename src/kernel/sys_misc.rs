//! Resource, scheduling, and process-attribute syscalls.
//!
//! These are mostly informational: they return success and/or write plausible
//! zeroed/static structs into guest memory. The guest's libc queries them at
//! startup (rlimits, cpu affinity, scheduler class, process name) and mostly
//! ignores the exact values, so a believable constant answer is enough.

use super::{Kernel, ServiceCtx, Shared, err};
use crate::abi::errno::Errno;
use crate::vcpu::GuestMemory;

impl Kernel {
    /// `prctl(option, ...)` — process-attribute get/set. We model the process
    /// name (`PR_SET_NAME`/`PR_GET_NAME`, stored on the kernel) and treat every
    /// other option as a successful no-op.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_prctl(&self, sh: &mut Shared, args: &[u64; 6], mem: &mut GuestMemory) -> i64 {
        const PR_SET_NAME: u64 = 15;
        const PR_GET_NAME: u64 = 16;
        match args[0] {
            PR_SET_NAME => {
                if let Ok(name) = mem.read_cstr(args[1], 16) {
                    let mut buf = [0u8; 16];
                    let n = name.len().min(15);
                    buf[..n].copy_from_slice(&name[..n]);
                    sh.procname = buf;
                }
                0
            }
            PR_GET_NAME => {
                if mem.write(args[1], &sh.procname).is_err() {
                    return err(Errno::EFAULT);
                }
                0
            }
            // PR_SET_PDEATHSIG, PR_GET/SET_DUMPABLE, PR_CAPBSET_READ, ... : no-op.
            _ => 0,
        }
    }
}

/// `sched_getaffinity(pid, size, mask)` — report a single online CPU (bit 0),
/// returning the number of bytes written (`min(size, 8)`).
pub(super) fn sys_sched_getaffinity(size: u64, mask: u64, mem: &mut GuestMemory) -> i64 {
    let n = size.min(8) as usize;
    if n == 0 {
        return err(Errno::EINVAL);
    }
    let mut buf = vec![0u8; n];
    buf[0] = 1;
    if mem.write(mask, &buf).is_err() {
        return err(Errno::EFAULT);
    }
    n as i64
}

/// `sched_getparam(pid, param)` — write a `sched_param { sched_priority = 0 }`.
pub(super) fn sys_sched_getparam(param: u64, mem: &mut GuestMemory) -> i64 {
    if mem.write(param, &0i32.to_le_bytes()).is_err() {
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
    let mut b = [0u8; 112];
    b[32..40].copy_from_slice(&(2u64 * 1024 * 1024 * 1024).to_le_bytes());
    b[80..82].copy_from_slice(&1u16.to_le_bytes());
    b[104..108].copy_from_slice(&1u32.to_le_bytes());
    b
}

#[cfg(test)]
mod tests {
    use super::{Kernel, sys_sched_getaffinity, sys_sysinfo};
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
    fn sched_getaffinity_sets_bit0() {
        let (_k, mut mem) = setup();
        let mask = 0x1_0000;
        assert_eq!(sys_sched_getaffinity(128, mask, &mut mem), 8);
        assert_eq!(mem.read_vec(mask, 1).unwrap()[0], 1);
    }

    #[test]
    fn prctl_set_get_name_roundtrips() {
        let (k, mut mem) = setup();
        let name = 0x1_0000;
        mem.write_init(name, b"myproc\0").unwrap();
        assert_eq!(k.sys_prctl(&mut k.shared.lock().unwrap(), &[15, name, 0, 0, 0, 0], &mut mem), 0);
        let out = 0x1_1000;
        assert_eq!(k.sys_prctl(&mut k.shared.lock().unwrap(), &[16, out, 0, 0, 0, 0], &mut mem), 0);
        assert_eq!(mem.read_vec(out, 6).unwrap(), b"myproc");
    }
}
