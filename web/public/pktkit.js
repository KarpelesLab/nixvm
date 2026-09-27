// Host imports for the `pktkit` crate on wasm32-unknown-unknown, which has no
// clock of its own: its TCP timers read `now_ms` (monotonic) and its ISNs /
// timestamps read `unix_ms` (wall clock). wasm-bindgen's glue imports these
// from the bare specifier "pktkit"; the import map in index.html points it
// here.
export function now_ms() {
  return performance.now();
}

export function unix_ms() {
  return Date.now();
}
