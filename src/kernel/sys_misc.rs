//! Resource, scheduling, and process-attribute syscalls.
//!
//! These are mostly informational: they return success and/or write plausible
//! zeroed/static structs into guest memory. The guest's libc queries them at
//! startup (rlimits, cpu affinity, scheduler class, process name) and mostly
//! ignores the exact values, so a believable constant answer is enough.

use super::{Kernel, err};
use crate::abi::errno::Errno;
use crate::vcpu::GuestMemory;

impl Kernel {
    /// `prctl(option, ...)` — process-attribute get/set. We model the process
    /// name (`PR_SET_NAME`/`PR_GET_NAME`, stored on the kernel) and treat every
    /// other option as a successful no-op.
    pub(super) fn sys_prctl(&mut self, args: &[u64; 6], mem: &mut GuestMemory) -> i64 {
        const PR_SET_NAME: u64 = 15;
        const PR_GET_NAME: u64 = 16;
        match args[0] {
            PR_SET_NAME => {
                if let Ok(name) = mem.read_cstr(args[1], 16) {
                    let mut buf = [0u8; 16];
                    let n = name.len().min(15);
                    buf[..n].copy_from_slice(&name[..n]);
                    self.procname = buf;
                }
                0
            }
            PR_GET_NAME => {
                if mem.write(args[1], &self.procname).is_err() {
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

/// `getrusage(who, buf)` — write a zeroed `struct rusage` (144 bytes).
pub(super) fn sys_getrusage(buf: u64, mem: &mut GuestMemory) -> i64 {
    let zeros = [0u8; 144];
    if mem.write(buf, &zeros).is_err() {
        return err(Errno::EFAULT);
    }
    0
}

/// `times(buf)` — write a zeroed `struct tms` (4 x i64) and return 0 ticks.
pub(super) fn sys_times(buf: u64, mem: &mut GuestMemory) -> i64 {
    let zeros = [0u8; 32];
    if mem.write(buf, &zeros).is_err() {
        return err(Errno::EFAULT);
    }
    0
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

/// `prlimit64(pid, resource, new_limit, old_limit)` — ignore `new_limit`; if
/// `old_limit` is non-null, report the (constant) limit for `resource`.
pub(super) fn sys_prlimit64(resource: u64, old_limit: u64, mem: &mut GuestMemory) -> i64 {
    if old_limit == 0 {
        return 0;
    }
    let (cur, max) = rlimit_for(resource);
    write_rlimit(mem, old_limit, cur, max)
}

/// `getrlimit(resource, buf)` — report the (constant) limit for `resource`.
pub(super) fn sys_getrlimit(resource: u64, buf: u64, mem: &mut GuestMemory) -> i64 {
    let (cur, max) = rlimit_for(resource);
    write_rlimit(mem, buf, cur, max)
}

/// The constant soft/hard limit pair we report for a given `RLIMIT_*` resource.
fn rlimit_for(resource: u64) -> (u64, u64) {
    const RLIMIT_STACK: u64 = 3;
    const RLIMIT_NPROC: u64 = 6;
    const RLIMIT_NOFILE: u64 = 7;
    const RLIM_INFINITY: u64 = u64::MAX;
    match resource {
        RLIMIT_NOFILE => (1024, 4096),
        RLIMIT_STACK => (8 * 1024 * 1024, RLIM_INFINITY),
        RLIMIT_NPROC => (4096, 4096),
        _ => (RLIM_INFINITY, RLIM_INFINITY),
    }
}

/// Write a `struct rlimit { rlim_cur, rlim_max }` (2 x u64) at `addr`.
fn write_rlimit(mem: &mut GuestMemory, addr: u64, cur: u64, max: u64) -> i64 {
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
    fn prlimit64_reports_nofile_limit() {
        let (_k, mut mem) = setup();
        let buf = 0x1_0000;
        assert_eq!(super::sys_prlimit64(7, buf, &mut mem), 0);
        assert_eq!(mem.read_u64(buf).unwrap(), 1024);
        assert_eq!(mem.read_u64(buf + 8).unwrap(), 4096);
    }

    #[test]
    fn prlimit64_null_old_limit_is_noop() {
        let (_k, mut mem) = setup();
        assert_eq!(super::sys_prlimit64(7, 0, &mut mem), 0);
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
        let (mut k, mut mem) = setup();
        let name = 0x1_0000;
        mem.write_init(name, b"myproc\0").unwrap();
        assert_eq!(k.sys_prctl(&[15, name, 0, 0, 0, 0], &mut mem), 0);
        let out = 0x1_1000;
        assert_eq!(k.sys_prctl(&[16, out, 0, 0, 0, 0], &mut mem), 0);
        assert_eq!(mem.read_vec(out, 6).unwrap(), b"myproc");
    }
}
