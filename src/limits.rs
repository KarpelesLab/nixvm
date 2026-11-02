//! How much of the host a VM may use.
//!
//! A guest is untrusted: nothing it does should be able to take the host down,
//! whether by exhausting its memory or by crashing the process embedding nixvm.
//! [`Limits`] sets VM-wide ceilings, separate from the `rlimit`s a guest sets on
//! its own processes, and hitting one is an ordinary error the guest can handle.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// VM-wide resource ceilings. See [`Limits::for_this_host`] for the defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    /// Host memory the kernel may hold on the guest's behalf, in bytes: file
    /// data in tmpfs, pipe and socket queues, and the like. Guest RAM is
    /// separate; it is bounded by the size of the guest's physical pool.
    pub kernel_memory: u64,
    /// Tasks (threads and processes, counting zombies) that may exist at once.
    /// `fork` and `clone` beyond this fail with `EAGAIN`.
    pub tasks: u32,
    /// File descriptors open at once across the whole VM, on top of each
    /// process's own `RLIMIT_NOFILE`. Opening more fails with `ENFILE`.
    pub open_files: u32,
    /// How much the guest may grow what it stores in host directories, in bytes.
    /// Past this, writes there fail with `ENOSPC`. Whatever this says, a guest
    /// can't take a host filesystem below its last 5% of space or inodes.
    pub disk_bytes: u64,
    /// CPU time the whole VM may use across all tasks, or `None` for no limit.
    /// When it runs out every guest task is killed.
    pub cpu_time: Option<Duration>,
    /// How long the VM may run from its first use, or `None` for no limit. When
    /// it runs out every guest task is killed.
    pub wall_time: Option<Duration>,
}

impl Limits {
    /// Sensible defaults for the machine this runs on: a quarter of its RAM for
    /// kernel memory, at most 8 GiB and at least 256 MiB, 4096 tasks and 262144 open
    /// files. Each can be overridden with `NIXVM_LIMIT_KERNEL_MEMORY` (bytes),
    /// `NIXVM_LIMIT_TASKS` or `NIXVM_LIMIT_OPEN_FILES`. Disk, CPU time and run time
    /// have no default limit, since a host can't guess what a job needs, and can be
    /// set with `NIXVM_LIMIT_DISK_BYTES`, `NIXVM_LIMIT_CPU_SECONDS` and
    /// `NIXVM_LIMIT_WALL_SECONDS`.
    #[must_use]
    pub fn for_this_host() -> Self {
        const MIB: u64 = 1024 * 1024;
        let kernel_memory = (host_memory_bytes() / 4).clamp(256 * MIB, 8192 * MIB);
        Self {
            kernel_memory: env_u64("NIXVM_LIMIT_KERNEL_MEMORY").unwrap_or(kernel_memory),
            tasks: env_u64("NIXVM_LIMIT_TASKS").map_or(4096, |n| n.min(1 << 22) as u32),
            open_files: env_u64("NIXVM_LIMIT_OPEN_FILES")
                .map_or(1 << 18, |n| n.min(u64::from(u32::MAX)) as u32),
            disk_bytes: env_u64("NIXVM_LIMIT_DISK_BYTES").unwrap_or(u64::MAX),
            cpu_time: env_u64("NIXVM_LIMIT_CPU_SECONDS").map(Duration::from_secs),
            wall_time: env_u64("NIXVM_LIMIT_WALL_SECONDS").map(Duration::from_secs),
        }
    }
}

impl Default for Limits {
    fn default() -> Self {
        Self::for_this_host()
    }
}

/// The most guest RAM a VM may be given: three quarters of this machine's RAM,
/// since a guest can touch every page it's promised, or
/// `NIXVM_LIMIT_GUEST_MEMORY` (bytes).
#[must_use]
pub fn guest_memory_ceiling() -> u64 {
    env_u64("NIXVM_LIMIT_GUEST_MEMORY").unwrap_or_else(|| host_memory_bytes() / 4 * 3)
}

/// A positive integer from environment variable `name`.
fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok()?.trim().parse().ok()
}

/// Physical memory on this machine, or a conservative guess if it can't be told.
fn host_memory_bytes() -> u64 {
    const FALLBACK: u64 = 4 * 1024 * 1024 * 1024;
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|info| {
            let line = info.lines().find(|l| l.starts_with("MemTotal:"))?;
            let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
            Some(kib * 1024)
        })
        .unwrap_or(FALLBACK)
}

/// A shared counter of bytes in use against a ceiling. A charge that would go
/// over fails, so the caller can return an error rather than exhaust the host.
#[derive(Debug)]
pub struct Quota {
    limit: AtomicU64,
    used: AtomicU64,
}

impl Quota {
    /// A quota of `limit` bytes.
    #[must_use]
    pub fn new(limit: u64) -> Arc<Self> {
        Arc::new(Self {
            limit: AtomicU64::new(limit),
            used: AtomicU64::new(0),
        })
    }

    /// A quota that never refuses, for a filesystem not attached to a kernel.
    #[must_use]
    pub fn unlimited() -> Arc<Self> {
        Self::new(u64::MAX)
    }

    /// Change the ceiling. What is already charged stays charged, so usage can
    /// sit above a lowered limit until it is released.
    pub fn set_limit(&self, limit: u64) {
        self.limit.store(limit, Ordering::Relaxed);
    }

    /// Charge `bytes`, or return `false` and charge nothing if that would go
    /// over the limit.
    #[must_use]
    pub fn try_charge(&self, bytes: u64) -> bool {
        let limit = self.limit.load(Ordering::Relaxed);
        let mut used = self.used.load(Ordering::Relaxed);
        loop {
            let Some(total) = used.checked_add(bytes).filter(|&t| t <= limit) else {
                return false;
            };
            match self
                .used
                .compare_exchange_weak(used, total, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => return true,
                Err(actual) => used = actual,
            }
        }
    }

    /// Charge `bytes` regardless of the limit, for memory already allocated.
    pub fn force_charge(&self, bytes: u64) {
        self.used.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Give back `bytes` from an earlier charge.
    pub fn release(&self, bytes: u64) {
        let mut used = self.used.load(Ordering::Relaxed);
        loop {
            match self.used.compare_exchange_weak(
                used,
                used.saturating_sub(bytes),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(actual) => used = actual,
            }
        }
    }

    /// Bytes currently charged.
    #[must_use]
    pub fn used(&self) -> u64 {
        self.used.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_charge_over_the_limit_fails_and_charges_nothing() {
        let q = Quota::new(100);
        assert!(q.try_charge(60));
        assert!(!q.try_charge(41));
        assert_eq!(q.used(), 60);
        assert!(q.try_charge(40));
        q.release(50);
        assert_eq!(q.used(), 50);
    }

    #[test]
    fn a_huge_charge_does_not_wrap_around() {
        let q = Quota::unlimited();
        assert!(q.try_charge(10));
        assert!(!q.try_charge(u64::MAX));
        assert_eq!(q.used(), 10);
    }

    #[test]
    fn release_never_goes_below_zero() {
        let q = Quota::new(100);
        q.release(5);
        assert_eq!(q.used(), 0);
    }

    #[test]
    fn defaults_are_within_their_clamp() {
        let l = Limits::for_this_host();
        if std::env::var_os("NIXVM_LIMIT_KERNEL_MEMORY").is_none() {
            assert!((256 << 20..=8192 << 20).contains(&l.kernel_memory));
        }
    }
}
