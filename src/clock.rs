//! The one place nixvm reads the host wall clock.
//!
//! `std::time::SystemTime::now()` **panics** on `wasm32-unknown-unknown`
//! ("time not implemented on this platform"), and that panic poisons the
//! whole wasm instance — in the browser demo the first guest syscall that
//! touched the clock (busybox `ls` calls `clock_gettime`) killed the
//! terminal. Every clock read in the crate goes through [`now_unix`], which
//! picks a working source per platform:
//!
//! * native: `SystemTime`, as before;
//! * wasm32 with the `wasm` feature: JavaScript's `Date.now()` via a
//!   hand-declared wasm-bindgen import (millisecond resolution — the guest
//!   ABI reports nanoseconds, but a browser tab has no better source without
//!   `performance.now()` origin gymnastics);
//! * wasm32 without `wasm` (no JS bindings linked): a monotonic fake clock
//!   ticking 1 ms per read — wrong but total, so nothing can panic.

use std::time::Duration;

/// Time since the UNIX epoch on the best clock the platform offers
/// (saturating at 0 for a host clock set before 1970). This is `CLOCK_REALTIME`.
#[must_use]
pub fn now_unix() -> Duration {
    imp::now_unix()
}

/// A monotonic clock (`CLOCK_MONOTONIC`): non-decreasing time since an arbitrary,
/// fixed epoch (here the process start / host boot — unrelated to the wall clock,
/// so it is immune to wall-clock steps). Used for `CLOCK_MONOTONIC` and friends.
#[must_use]
pub fn now_monotonic() -> Duration {
    imp::now_monotonic()
}

/// Process CPU time consumed so far (`CLOCK_PROCESS_CPUTIME_ID`): advances only
/// while the process runs on a CPU, not while it sleeps.
#[must_use]
pub fn now_cpu_process() -> Duration {
    imp::now_cpu_process()
}

/// Thread CPU time consumed so far (`CLOCK_THREAD_CPUTIME_ID`).
#[must_use]
pub fn now_cpu_thread() -> Duration {
    imp::now_cpu_thread()
}

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    use std::time::Duration;

    pub fn now_unix() -> Duration {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
    }

    // On Linux the guest's clock ids match the host's (`CLOCK_MONOTONIC` = 1,
    // `CLOCK_PROCESS_CPUTIME_ID` = 2, `CLOCK_THREAD_CPUTIME_ID` = 3), so we read
    // the real host clocks directly — a since-boot monotonic and genuine CPU
    // time, exactly like a native process sees.
    #[cfg(target_os = "linux")]
    mod host {
        use std::time::Duration;
        #[repr(C)]
        struct Ts {
            sec: i64,
            nsec: i64,
        }
        unsafe extern "C" {
            fn clock_gettime(clk: i32, tp: *mut Ts) -> i32;
        }
        pub fn read(clk: i32) -> Option<Duration> {
            let mut ts = Ts { sec: 0, nsec: 0 };
            // SAFETY: `ts` is a valid, writable `timespec`; `clk` is a fixed,
            // valid POSIX clock id. `clock_gettime` writes only `ts`.
            (unsafe { clock_gettime(clk, &mut ts) } == 0)
                .then(|| Duration::new(ts.sec.max(0) as u64, ts.nsec.clamp(0, 999_999_999) as u32))
        }
    }

    /// Cross-platform monotonic fallback: elapsed since a fixed process origin.
    fn instant_monotonic() -> Duration {
        use std::sync::OnceLock;
        use std::time::Instant;
        static ORIGIN: OnceLock<Instant> = OnceLock::new();
        ORIGIN.get_or_init(Instant::now).elapsed()
    }

    pub fn now_monotonic() -> Duration {
        #[cfg(target_os = "linux")]
        if let Some(d) = host::read(1) {
            return d;
        }
        instant_monotonic()
    }

    pub fn now_cpu_process() -> Duration {
        #[cfg(target_os = "linux")]
        if let Some(d) = host::read(2) {
            return d;
        }
        // No portable CPU clock here: best-effort monotonic (still non-decreasing).
        instant_monotonic()
    }

    pub fn now_cpu_thread() -> Duration {
        #[cfg(target_os = "linux")]
        if let Some(d) = host::read(3) {
            return d;
        }
        instant_monotonic()
    }
}

#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
mod imp {
    use std::time::Duration;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        /// `Date.now()` — milliseconds since the UNIX epoch.
        #[wasm_bindgen(js_namespace = Date, js_name = now)]
        fn date_now() -> f64;
    }

    pub fn now_unix() -> Duration {
        Duration::from_millis(date_now() as u64)
    }

    // The browser has no CPU/monotonic clock we cheaply reach here; the wall
    // clock is total and non-panicking, which is all the demo needs.
    pub fn now_monotonic() -> Duration {
        now_unix()
    }
    pub fn now_cpu_process() -> Duration {
        now_unix()
    }
    pub fn now_cpu_thread() -> Duration {
        now_unix()
    }
}

#[cfg(all(target_arch = "wasm32", not(feature = "wasm")))]
mod imp {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    /// No JS to ask and no std clock: a monotonic counter that advances 1 ms
    /// per read keeps time-dependent guest code moving instead of panicking.
    static FAKE_MS: AtomicU64 = AtomicU64::new(1_700_000_000_000);

    pub fn now_unix() -> Duration {
        Duration::from_millis(FAKE_MS.fetch_add(1, Ordering::Relaxed))
    }
    pub fn now_monotonic() -> Duration {
        now_unix()
    }
    pub fn now_cpu_process() -> Duration {
        now_unix()
    }
    pub fn now_cpu_thread() -> Duration {
        now_unix()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn now_is_after_2020() {
        // A very loose sanity bound: the host clock reads as a real date.
        assert!(now_unix().as_secs() > 1_577_836_800, "clock reads as post-2020");
    }
}
