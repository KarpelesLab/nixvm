<script setup>
import "@xterm/xterm/css/xterm.css";

import { FitAddon } from "@xterm/addon-fit";
import { Terminal as XTerm } from "@xterm/xterm";
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from "vue";

// ---------------------------------------------------------------------------
// nixvm browser terminal.
//
// Boots a real Alpine Linux minirootfs — the user picks the guest
// architecture (arm64 or x86-64) first, then Start fetches the matching
// same-origin `rootfs-<arch>.tar.gz` and hands it to the wasm as-is (nixvm
// gunzips it itself via the `compcol` crate) — into the nixvm wasm
// sandbox's `Terminal` class (from `../../pkg/nixvm.js`, assembled next to
// this app's build output by `.github/workflows/pages.yml` — see
// `web/README.md` for how to get it locally) and drives an interactive
// `/bin/busybox sh` session. Both arches run on the same wasm build: the
// sandbox picks its aarch64 or x86-64 interpreter off the ELF headers
// inside the image.
//
// `busybox sh` here is not attached to a real TTY (no line editor, no
// local echo), so *this component* does the line editing: it buffers the
// current line, echoes typed characters to xterm itself, and only calls
// down into the guest (`write_stdin` + `pump`) on Enter / Ctrl-C / Ctrl-D.
// `pump()` runs the guest synchronously until it parks, so a CPU-bound
// command will briefly freeze the tab — that's a known, accepted trade-off
// of running a Linux userland on the main thread. A guest parked on a timer
// or the network is pumped again from a timer (and on every packet that
// arrives) until the shell is back reading the terminal; only then is the
// prompt shown.
//
// Networking: the guest's sockets go through nixvm's userspace TCP/IP stack
// (pktkit), which speaks raw IP packets. We relay those over a WebSocket to
// grouterd's tunnel endpoint: fetch a short-lived token, open
// `wss://…/tunnel/<token>`, read the JSON hello (leased addresses, MTU), then
// every binary message each way is one IP packet.
// ---------------------------------------------------------------------------

const PROMPT = "/ $ ";

/// Guest architectures the demo can boot. Both run on the same wasm
/// interpreter build — the arch is auto-detected from the ELFs inside the
/// selected rootfs image; the choice here only picks which Alpine image
/// (`rootfs-<arch>.tar.gz`, bundled by pages.yml) is fetched and booted.
const ARCHES = [
  { id: "aarch64", label: "arm64" },
  { id: "x86_64", label: "x86-64" },
];

/// Where a tunnel token comes from, and where the tunnel is opened with it.
const NET_TOKEN_URL = "https://ws.atonline.com/_special/rest/Network:jwt";
const NET_TUNNEL_URL = "wss://grouterd.atonline.com/tunnel/";
/// grouterd drops a tunnel after 5 minutes without traffic from us; a text
/// message counts as traffic and is otherwise ignored.
const NET_KEEPALIVE_MS = 60_000;
/// TCP timers (retransmits, delayed ACKs) run on this period.
const NET_TICK_MS = 100;
/// Tunnel close codes that are not worth reconnecting after with the same
/// session: another tunnel with our token replaced this one.
const NET_CLOSE_REPLACED = 4002;

const termEl = ref(null);
const status = ref("idle");
const arch = ref("aarch64");
const hasBooted = ref(false);
const statusMessages = {
  idle: "pick an architecture and press Start",
  downloading: "downloading Alpine rootfs…",
  decompressing: "decompressing rootfs…",
  loading: "loading WebAssembly module…",
  booting: "booting Alpine…",
  ready: "running",
  exited: "shell exited",
  error: "boot failed",
};
const statusText = computed(() => statusMessages[status.value] ?? status.value);

// Network link state, shown in the toolbar: off | connecting | online | error.
const netEnabled = ref(true);
const netState = ref("off");
const netAddr = ref("");
const netLabel = computed(() => {
  if (!netEnabled.value) return "net: off";
  switch (netState.value) {
    case "online":
      return `net: ${netAddr.value}`;
    case "connecting":
      return "net: connecting…";
    case "error":
      return "net: offline (retrying)";
    default:
      return "net: off";
  }
});
const netTitle = computed(() =>
  netEnabled.value
    ? "Guest networking via a WebSocket IP tunnel (grouterd). Click to disconnect."
    : "Guest networking is off. Click to connect.",
);
const bootingPhases = new Set(["downloading", "decompressing", "loading", "booting"]);
const rebootDisabled = computed(() => bootingPhases.has(status.value));
const bootLabel = computed(() => (hasBooted.value ? "Reboot" : "Start"));

let xterm = null;
let fitAddon = null;
let resizeObserver = null;
let guestTerm = null;

// Cached across reboots so hitting "Reboot" doesn't re-download a rootfs
// (one entry per arch) or re-instantiate the wasm module — only a fresh
// `Terminal` is created.
const cachedTars = new Map();
let cachedWasmModule = null;

let lineBuffer = "";
let atLineStart = true;
let busy = false;
// A command line was sent and the shell hasn't asked for the next one yet.
let commandRunning = false;
let pumpTimer = null;
let pumpDue = 0;
// Set once the wasm instance has trapped (panicked): the session is
// unrecoverable and every further guest call would throw. Cleared on reboot.
let guestDead = false;

const encoder = new TextEncoder();
const decoder = new TextDecoder();

// Resolve a site-relative path (e.g. "pkg/nixvm.js") against the *page's*
// URL. This matters for the dynamic `import()` below: unlike `fetch()`,
// which resolves a relative URL string against the document, a relative
// module specifier passed to `import()` resolves against the URL of the
// *importing module itself* — which, after bundling, is some hashed chunk
// under `assets/`, not the page. Resolving through `document.baseURI`
// first sidesteps that footgun for both calls.
function siteUrl(path) {
  return new URL(`${import.meta.env.BASE_URL}${path}`, document.baseURI).href;
}

function tick() {
  // Yield one microtask + a paint so status text updates before a
  // synchronous, potentially heavy `pump()` call blocks the main thread.
  return nextTick().then(() => new Promise((r) => requestAnimationFrame(r)));
}

function writeRaw(str) {
  if (!str) return;
  xterm.write(str);
  atLineStart = str.endsWith("\n") || str.endsWith("\r");
}

function writeBytes(bytes) {
  if (!bytes || bytes.length === 0) return;
  writeRaw(decoder.decode(bytes));
}

function writePrompt() {
  if (!atLineStart) xterm.write("\r\n");
  xterm.write(PROMPT);
  atLineStart = false;
}

function writeBanner(msg) {
  writeRaw(`${msg}\r\n`);
}

function writeErrorBanner(msg) {
  // 31 = red
  writeRaw(`\r\n\x1b[31m${msg}\x1b[0m\r\n`);
}

// The wasm sandbox is built with panic=abort, so a Rust panic (an
// unimplemented syscall path, a bug) traps the whole instance: from then on
// *every* call into it throws, and there's no recovering this session. Turn
// that into a visible message + a dead-but-rebootable terminal instead of a
// silent freeze at the prompt. The real panic text is in the browser console
// (console_error_panic_hook); we surface a pointer to it.
function surfaceGuestCrash(err) {
  if (guestDead) return;
  guestDead = true;
  netDisconnect();
  cancelPump();
  status.value = "error";
  const detail = err?.message ?? String(err);
  writeErrorBanner(
    `guest crashed: ${detail}\r\n(the sandbox panicked and can't continue — see the browser console for the Rust backtrace; click Reboot to start a new session)`,
  );
}

// Pump the guest after `delay` ms (sooner requests win over later ones).
function schedulePump(delay = 0) {
  const due = performance.now() + delay;
  if (pumpTimer !== null) {
    if (pumpDue <= due) return;
    clearTimeout(pumpTimer);
  }
  pumpDue = due;
  pumpTimer = setTimeout(runPump, delay);
}

function cancelPump() {
  if (pumpTimer !== null) clearTimeout(pumpTimer);
  pumpTimer = null;
}

// Run the guest until it parks, show its output, send its packets, and
// decide what's next: the prompt (the shell wants input), or another pump
// soon (it's waiting on a timer or the network).
function runPump() {
  pumpTimer = null;
  if (!guestTerm || guestDead || status.value !== "ready") return;
  let out;
  let awaiting;
  let pending;
  try {
    out = guestTerm.pump();
    writeBytes(out);
    flushNet();
    if (!guestTerm.is_running()) {
      const code = guestTerm.exit_code();
      status.value = "exited";
      commandRunning = false;
      writeRaw(`\r\n[ shell exited with code ${code} — click Reboot to start a new session ]\r\n`);
      return;
    }
    awaiting = guestTerm.awaiting_input();
    pending = guestTerm.has_pending_work();
  } catch (err) {
    surfaceGuestCrash(err);
    return;
  }
  if (commandRunning && awaiting) {
    commandRunning = false;
    writePrompt();
  }
  // Waiting on a timer / the network (foreground or background job): check
  // back soon. Packets arriving also trigger an immediate pump.
  if (pending) schedulePump(20);
}

async function afterStdinChanged() {
  commandRunning = true;
  await tick();
  cancelPump();
  runPump();
}

async function handleInput(data) {
  if (busy || guestDead || status.value !== "ready") return;
  // While a command runs, typed lines go to it as input (and Ctrl-C to it as
  // an interrupt); the prompt comes back once the shell is reading again.
  busy = true;
  try {
    for (const ch of data) {
      const code = ch.codePointAt(0);
      if (ch === "\r") {
        writeRaw("\r\n");
        const line = `${lineBuffer}\n`;
        lineBuffer = "";
        guestTerm.write_stdin(encoder.encode(line));
        await afterStdinChanged();
      } else if (code === 127 || code === 8) {
        // Backspace (DEL or BS, depending on platform/browser).
        if (lineBuffer.length > 0) {
          lineBuffer = lineBuffer.slice(0, -1);
          xterm.write("\b \b");
        }
      } else if (code === 3) {
        // Ctrl-C
        writeRaw("^C\r\n");
        lineBuffer = "";
        guestTerm.write_stdin(encoder.encode("\x03"));
        await afterStdinChanged();
      } else if (code === 4) {
        // Ctrl-D: EOF, only meaningful on an empty line.
        if (lineBuffer.length === 0) {
          guestTerm.close_stdin();
          await afterStdinChanged();
        }
      } else if (code === 27) {
        // Escape sequence (arrow keys, function keys, …) — this is not a
        // real line editor, so we don't support cursor movement/history.
        break;
      } else if (code < 32) {
        // Other control characters: ignore.
      } else {
        lineBuffer += ch;
        xterm.write(ch);
        atLineStart = false;
      }
      if (status.value !== "ready") break;
    }
  } finally {
    busy = false;
  }
}

// ---- networking -------------------------------------------------------

let ws = null;
let netGen = 0; // bumped on every (re)connect/disconnect; stale callbacks bail
let netTickTimer = null;
let netKeepaliveTimer = null;
let netRetryTimer = null;
let netRetryDelay = 2000;

// Send every packet the guest's stack has queued.
function flushNet() {
  if (!guestTerm || !ws || ws.readyState !== WebSocket.OPEN) return;
  for (;;) {
    const p = guestTerm.net_next_packet();
    if (p === undefined || p === null) break;
    ws.send(p);
  }
}

function netStopTimers() {
  clearInterval(netTickTimer);
  clearInterval(netKeepaliveTimer);
  clearTimeout(netRetryTimer);
  netTickTimer = netKeepaliveTimer = netRetryTimer = null;
}

// Drop the tunnel (if any) and take the guest's link down.
function netDisconnect() {
  netGen++;
  netStopTimers();
  const sock = ws;
  ws = null;
  if (sock) {
    sock.onopen = sock.onmessage = sock.onclose = sock.onerror = null;
    try {
      sock.close(1000);
    } catch {
      // already closed
    }
  }
  try {
    if (!guestDead) guestTerm?.net_down();
  } catch (err) {
    surfaceGuestCrash(err);
  }
  netState.value = "off";
  netAddr.value = "";
}

function netRetryLater() {
  netStopTimers();
  if (!netEnabled.value || !guestTerm || guestDead) return;
  netState.value = "error";
  const delay = netRetryDelay;
  netRetryDelay = Math.min(netRetryDelay * 2, 30_000);
  netRetryTimer = setTimeout(() => netConnect(), delay);
}

async function netConnect() {
  netDisconnect();
  if (!netEnabled.value || !guestTerm || guestDead) return;
  const gen = netGen;
  netState.value = "connecting";
  let token;
  try {
    const res = await fetch(NET_TOKEN_URL, { cache: "no-store" });
    if (!res.ok) throw new Error(`token request failed: ${res.status}`);
    const body = await res.json();
    token = body?.data?.token;
    if (!token) throw new Error("no token in response");
  } catch (err) {
    if (gen !== netGen) return;
    console.warn("nixvm: network token:", err);
    netRetryLater();
    return;
  }
  if (gen !== netGen || !guestTerm) return;

  const sock = new WebSocket(NET_TUNNEL_URL + encodeURIComponent(token));
  sock.binaryType = "arraybuffer";
  ws = sock;
  let hello = false;
  sock.onmessage = (ev) => {
    if (gen !== netGen || !guestTerm || guestDead) return;
    try {
      if (typeof ev.data === "string") {
        // The first text message is the hello: what we were leased.
        if (hello) return;
        hello = true;
        const h = JSON.parse(ev.data);
        guestTerm.net_up(h.ipv4 ?? undefined, h.ipv4_prefix ?? 32, h.ipv6 ?? undefined, h.mtu ?? 1400);
        netAddr.value = h.ipv4 ?? h.ipv6 ?? "up";
        netState.value = "online";
        netRetryDelay = 2000;
        netTickTimer = setInterval(() => {
          try {
            guestTerm?.net_tick();
            flushNet();
          } catch (err) {
            surfaceGuestCrash(err);
          }
        }, NET_TICK_MS);
        netKeepaliveTimer = setInterval(() => {
          if (sock.readyState === WebSocket.OPEN) sock.send("keepalive");
        }, NET_KEEPALIVE_MS);
        return;
      }
      guestTerm.net_input(new Uint8Array(ev.data));
      schedulePump(0);
    } catch (err) {
      surfaceGuestCrash(err);
    }
  };
  sock.onclose = (ev) => {
    if (gen !== netGen) return;
    ws = null;
    try {
      if (!guestDead) guestTerm?.net_down();
    } catch (err) {
      surfaceGuestCrash(err);
    }
    netAddr.value = "";
    if (ev.code === NET_CLOSE_REPLACED) {
      netStopTimers();
      netState.value = "off";
      netEnabled.value = false;
      return;
    }
    // Expired token, dropped connection, server restart: get a new token.
    netRetryLater();
  };
}

function toggleNet() {
  netEnabled.value = !netEnabled.value;
  if (netEnabled.value) {
    netRetryDelay = 2000;
    netConnect();
  } else {
    netDisconnect();
  }
}

async function fetchRootfsTarGz(archId) {
  const cached = cachedTars.get(archId);
  if (cached) return cached;
  // Fetch the compressed image as-is; nixvm's wasm decompresses the gzip
  // itself (via the `compcol` crate), so there's no DecompressionStream
  // dependency and it works in any wasm-capable browser.
  status.value = "downloading";
  await tick();
  const url = siteUrl(`rootfs-${archId}.tar.gz`);
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`failed to fetch ${url}: ${res.status} ${res.statusText}`);
  }
  const buf = await res.arrayBuffer();
  const tar = new Uint8Array(buf);
  cachedTars.set(archId, tar);
  return tar;
}

async function loadWasmModule() {
  if (cachedWasmModule) return cachedWasmModule;
  status.value = "loading";
  await tick();
  // The wasm-pack `pkg/` output doesn't exist in this Vite project's source
  // tree — it's produced by `wasm-pack build` and copied in next to the
  // built site by `.github/workflows/pages.yml` (or by hand for local dev,
  // see web/README.md). `@vite-ignore` tells Vite not to try to resolve
  // this import path at build time.
  const url = siteUrl("pkg/nixvm.js");
  const mod = await import(/* @vite-ignore */ url);
  await mod.default();
  cachedWasmModule = mod;
  return cachedWasmModule;
}

async function boot() {
  const archId = arch.value;
  guestDead = false;
  try {
    const targz = await fetchRootfsTarGz(archId);
    const mod = await loadWasmModule();
    status.value = "booting";
    lineBuffer = "";
    atLineStart = true;
    await tick();
    // The wasm Terminal takes the raw .tar.gz and gunzips it in-process; the
    // guest arch is auto-detected from the ELFs inside it. Construction or the
    // first pump can trap (panic=abort) — the catch below surfaces it.
    guestTerm = new mod.Terminal(targz, ["/bin/busybox", "sh"]);
    const out = guestTerm.pump();
    writeBanner(`nixvm — Alpine Linux (${archId}), running entirely in your browser.`);
    writeBanner('Type commands and press Enter. Try: uname -m; ls /; cat /etc/os-release');
    writeBytes(out);
    hasBooted.value = true;
    status.value = "ready";
    writePrompt();
    netConnect();
  } catch (err) {
    status.value = "error";
    writeErrorBanner(`boot failed: ${err?.message ?? err}`);
  }
}

async function reboot() {
  if (rebootDisabled.value) return;
  netDisconnect();
  cancelPump();
  commandRunning = false;
  try {
    guestTerm?.free?.();
  } catch {
    // already freed / never constructed — fine.
  }
  guestTerm = null;
  lineBuffer = "";
  atLineStart = true;
  busy = false;
  xterm.reset();
  status.value = "idle";
  await boot();
}

function fit() {
  if (!fitAddon) return;
  try {
    fitAddon.fit();
  } catch {
    // Container not laid out yet (e.g. mid-unmount) — ignore.
  }
}

onMounted(() => {
  xterm = new XTerm({
    cursorBlink: true,
    convertEol: true,
    fontSize: 14,
    fontFamily:
      'ui-monospace, "SF Mono", "Cascadia Code", "Fira Code", Menlo, Consolas, monospace',
    scrollback: 4000,
    theme: {
      background: "#0e1013",
      foreground: "#e6e8eb",
      cursor: "#8be9fd",
      selectionBackground: "#3a4453",
      black: "#0e1013",
      brightBlack: "#4a5262",
      red: "#ff6b6b",
      green: "#8ce99a",
      yellow: "#ffd43b",
      blue: "#74c0fc",
      magenta: "#d0a2f7",
      cyan: "#66d9e8",
      white: "#e6e8eb",
    },
  });
  fitAddon = new FitAddon();
  xterm.loadAddon(fitAddon);
  xterm.open(termEl.value);
  fit();
  xterm.onData(handleInput);

  resizeObserver = new ResizeObserver(() => fit());
  resizeObserver.observe(termEl.value);
  window.addEventListener("resize", fit);

  // No auto-boot: the guest architecture is chosen first, then Start fetches
  // and boots the matching Alpine image.
  writeBanner("nixvm — a real Alpine Linux userland, entirely in your browser.");
  writeBanner("Pick a guest architecture above, then press Start.");
});

onBeforeUnmount(() => {
  netDisconnect();
  cancelPump();
  resizeObserver?.disconnect();
  window.removeEventListener("resize", fit);
  try {
    guestTerm?.free?.();
  } catch {
    // ignore
  }
  xterm?.dispose();
});
</script>

<template>
  <div class="term-wrap">
    <div class="term-toolbar">
      <span class="status-dot" :class="`is-${status}`"></span>
      <span class="status-text">{{ statusText }}</span>
      <div class="arch-picker" role="radiogroup" aria-label="Guest architecture">
        <button
          v-for="a in ARCHES"
          :key="a.id"
          class="arch-btn"
          :class="{ 'is-selected': arch === a.id }"
          :disabled="rebootDisabled"
          role="radio"
          :aria-checked="arch === a.id"
          @click="arch = a.id"
        >
          {{ a.label }}
        </button>
      </div>
      <button
        class="net-btn"
        :class="`is-${netEnabled ? netState : 'off'}`"
        :title="netTitle"
        :disabled="!hasBooted"
        @click="toggleNet"
      >
        {{ netLabel }}
      </button>
      <button class="reboot-btn" :disabled="rebootDisabled" @click="hasBooted ? reboot() : boot()">
        {{ bootLabel }}
      </button>
    </div>
    <div ref="termEl" class="term-container"></div>
  </div>
</template>

<style scoped>
.term-wrap {
  display: flex;
  flex-direction: column;
  gap: 0.4rem;
  min-width: 0;
}

.term-toolbar {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.85rem;
  color: var(--muted);
}

.status-dot {
  width: 0.55rem;
  height: 0.55rem;
  border-radius: 50%;
  background: var(--muted);
  flex: none;
}

.status-dot.is-downloading,
.status-dot.is-decompressing,
.status-dot.is-loading,
.status-dot.is-booting {
  background: #ffd43b;
  animation: pulse 1.1s ease-in-out infinite;
}

.status-dot.is-ready {
  background: #8ce99a;
}

.status-dot.is-error {
  background: var(--danger);
}

.status-dot.is-exited {
  background: #74c0fc;
}

@keyframes pulse {
  0%,
  100% {
    opacity: 1;
  }
  50% {
    opacity: 0.35;
  }
}

.status-text {
  flex: 1 1 auto;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.arch-picker {
  flex: none;
  display: flex;
  border: 1px solid var(--panel-border);
  border-radius: 0.4rem;
  overflow: hidden;
}

.arch-btn {
  background: var(--panel);
  color: var(--muted);
  border: none;
  padding: 0.3rem 0.7rem;
  font-size: 0.8rem;
  font-family: inherit;
  cursor: pointer;
}

.arch-btn + .arch-btn {
  border-left: 1px solid var(--panel-border);
}

.arch-btn.is-selected {
  background: var(--panel-border);
  color: var(--accent-strong, var(--fg));
}

.arch-btn:hover:not(:disabled):not(.is-selected) {
  color: var(--fg);
}

.arch-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.net-btn {
  flex: none;
  background: var(--panel);
  color: var(--muted);
  border: 1px solid var(--panel-border);
  border-radius: 0.4rem;
  padding: 0.3rem 0.7rem;
  font-size: 0.8rem;
  font-family: inherit;
  cursor: pointer;
  max-width: 14rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.net-btn.is-online {
  color: #8ce99a;
}

.net-btn.is-connecting {
  color: #ffd43b;
}

.net-btn.is-error {
  color: var(--danger);
}

.net-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.reboot-btn {
  flex: none;
  background: var(--panel);
  color: var(--fg);
  border: 1px solid var(--panel-border);
  border-radius: 0.4rem;
  padding: 0.3rem 0.75rem;
  font-size: 0.8rem;
  font-family: inherit;
  cursor: pointer;
}

.reboot-btn:hover:not(:disabled) {
  border-color: var(--accent);
  color: var(--accent-strong);
}

.reboot-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.term-container {
  width: 100%;
  min-width: 0;
  height: min(60vh, 480px);
  min-height: 260px;
  background: #0e1013;
  border: 1px solid var(--panel-border);
  border-radius: 0.5rem;
  padding: 0.5rem;
  overflow: hidden;
}

/* xterm sizes its own internals via FitAddon; this just keeps the viewport
   from ever introducing page-level horizontal scroll. */
.term-container :deep(.xterm) {
  height: 100%;
}
</style>
