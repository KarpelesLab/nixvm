//! Time and sleep syscalls.
//!
//! nixvm reads the host wall clock for the "get" calls. Sleeps use a
//! cooperative model: rather than block the host thread, a `nanosleep` treats
//! its interval as already elapsed and returns immediately (a real
//! scheduler-driven sleep is future work). Clock/time *setters* are refused
//! (`EPERM`) since the guest does not own the host clock.
//!
//! `clock_gettime` itself lives in the kernel module alongside the dispatcher.

use super::err;
use crate::abi::errno::Errno;
use crate::vcpu::GuestMemory;

/// Time since the UNIX epoch on the host clock (see [`crate::clock`] — the
/// platform-safe source that also works on wasm32).
fn host_now() -> std::time::Duration {
    crate::clock::now_unix()
}

/// `gettimeofday(tv, tz)` — write `struct timeval { i64 tv_sec; i64 tv_usec }`
/// (16 bytes) from the host clock. `tz` is obsolete and ignored.
pub(super) fn sys_gettimeofday(tv: u64, mem: &mut GuestMemory) -> i64 {
    if tv == 0 {
        return 0;
    }
    let now = host_now();
    let mut b = [0u8; 16];
    b[0..8].copy_from_slice(&(now.as_secs() as i64).to_le_bytes());
    b[8..16].copy_from_slice(&i64::from(now.subsec_micros()).to_le_bytes());
    match mem.write(tv, &b) {
        Ok(()) => 0,
        Err(_) => err(Errno::EFAULT),
    }
}

/// `clock_getres(clk_id, res)` — report a 1 ns resolution: if `res` is non-null
/// write `timespec { tv_sec: 0, tv_nsec: 1 }`.
pub(super) fn sys_clock_getres(res: u64, mem: &mut GuestMemory) -> i64 {
    if res == 0 {
        return 0;
    }
    let mut b = [0u8; 16];
    b[8..16].copy_from_slice(&1i64.to_le_bytes()); // tv_nsec = 1
    match mem.write(res, &b) {
        Ok(()) => 0,
        Err(_) => err(Errno::EFAULT),
    }
}

// `nanosleep`/`clock_nanosleep` live in the kernel module ([`Kernel::sys_nanosleep`]):
// unlike the pure time getters here, they suspend the caller on the scheduler's
// timed-wait, so they need the task context.

/// `time(tloc)` — return host unix seconds; if `tloc` is non-null also write it
/// there as an 8-byte value.
pub(super) fn sys_time(tloc: u64, mem: &mut GuestMemory) -> i64 {
    let secs = host_now().as_secs() as i64;
    if tloc != 0 && mem.write(tloc, &secs.to_le_bytes()).is_err() {
        return err(Errno::EFAULT);
    }
    secs
}
