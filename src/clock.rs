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
//! * wasm32 with the `wasm` feature: JavaScript's `performance.now()` via a
//!   hand-declared wasm-bindgen import (sub-millisecond), anchored to the
//!   wall clock with one `Date.now()` reading for `CLOCK_REALTIME`;
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
            (unsafe { clock_gettime(clk, &raw mut ts) } == 0)
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
    use std::sync::OnceLock;
    use std::time::Duration;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        /// `Date.now()` — milliseconds since the UNIX epoch (whole ms).
        #[wasm_bindgen(js_namespace = Date, js_name = now)]
        fn date_now() -> f64;
        /// `performance.now()` — monotonic milliseconds since page load, with
        /// sub-millisecond resolution (browsers coarsen it to 5-100 µs).
        #[wasm_bindgen(js_namespace = performance, js_name = now)]
        fn perf_now() -> f64;
    }

    fn from_ms(ms: f64) -> Duration {
        Duration::from_nanos((ms.max(0.0) * 1e6) as u64)
    }

    /// `Date.now() - performance.now()`, taken once: the wall-clock time of
    /// the monotonic origin. Realtime is this plus `performance.now()`, so it
    /// keeps the monotonic clock's sub-millisecond resolution (`Date.now()`
    /// alone made every guest timing — ping's RTTs — whole milliseconds).
    fn epoch_offset_ms() -> f64 {
        static OFFSET: OnceLock<f64> = OnceLock::new();
        *OFFSET.get_or_init(|| date_now() - perf_now())
    }

    pub fn now_unix() -> Duration {
        from_ms(epoch_offset_ms() + perf_now())
    }

    pub fn now_monotonic() -> Duration {
        from_ms(perf_now())
    }
    // No per-process CPU clock in a tab; the monotonic clock is total and
    // non-panicking, which is all the demo needs.
    pub fn now_cpu_process() -> Duration {
        now_monotonic()
    }
    pub fn now_cpu_thread() -> Duration {
        now_monotonic()
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
        assert!(
            now_unix().as_secs() > 1_577_836_800,
            "clock reads as post-2020"
        );
    }
}
