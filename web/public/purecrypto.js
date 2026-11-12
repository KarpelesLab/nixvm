// Host imports for the `purecrypto` crate's browser entropy source, where all
// of nixvm's randomness comes from: the guest's getrandom(2), /dev/urandom,
// its stack-canary seed, RDRAND, and the TLS that pulls images.
//
// `random_get(ptr, len)` fills `len` bytes of the wasm module's memory at
// `ptr` from `crypto.getRandomValues` (purecrypto asks for 16..=65536 bytes a
// call). nixvm hands over that memory as it starts, through `nixvm_memory`.
let memory = null;

export function nixvm_memory(m) {
  memory = m;
}

export function random_get(ptr, len) {
  if (!memory) throw new Error("purecrypto.random_get: called before nixvm handed over its memory");
  // A fresh view each call: the buffer is replaced when the memory grows.
  crypto.getRandomValues(new Uint8Array(memory.buffer, ptr, len));
}
