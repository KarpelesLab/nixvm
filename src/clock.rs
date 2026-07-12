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
/// (saturating at 0 for a host clock set before 1970).
#[must_use]
pub fn now_unix() -> Duration {
    imp::now_unix()
}

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    use std::time::Duration;

    pub fn now_unix() -> Duration {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
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
