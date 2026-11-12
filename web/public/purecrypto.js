// Host import for the `purecrypto` crate's browser entropy source, which the
// wasm module links but never calls: nixvm hands its TLS client entropy from
// `crypto.getRandomValues` itself (src/oci/net.rs). If anything ever does call
// this, fail loudly rather than hand out bytes that aren't random.
export function random_get() {
  throw new Error("purecrypto.random_get: unexpected call (nixvm supplies TLS entropy itself)");
}
