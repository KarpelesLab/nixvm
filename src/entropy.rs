//! Randomness for the guest: `getrandom(2)`, `/dev/random` and `/dev/urandom`,
//! the `AT_RANDOM` bytes libc seeds its stack canary from, and the
//! interpreter's `RDRAND`.
//!
//! All of it is purecrypto's [`OsRng`]: the host OS's generator natively
//! (`getrandom(2)` on Linux), and in the browser its `purecrypto.random_get`
//! import, which the page backs with `crypto.getRandomValues` (see
//! `web/public/purecrypto.js`; the wasm module hands that shim its
//! memory at start-up). Guests generate keys with these bytes, so they must
//! never be predictable.

use purecrypto::rng::{OsRng, RngCore};

/// Fill `buf` with random bytes.
pub fn fill(buf: &mut [u8]) {
    OsRng.fill_bytes(buf);
}

/// A random `u64`.
#[must_use]
pub fn u64() -> u64 {
    let mut b = [0u8; 8];
    fill(&mut b);
    u64::from_le_bytes(b)
}

#[cfg(test)]
mod tests {
    #[test]
    fn bytes_differ_from_call_to_call() {
        let (mut a, mut b) = ([0u8; 32], [0u8; 32]);
        super::fill(&mut a);
        super::fill(&mut b);
        assert_ne!(a, b);
        assert_ne!(a, [0; 32]);
        assert_ne!(super::u64(), super::u64());
    }
}
