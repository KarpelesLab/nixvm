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
/// Tunnel close codes (grouterd): our token expired (get a new one), or
/// another tunnel with our token replaced this one (don't fight it).
const NET_CLOSE_EXPIRED = 4001;
const NET_CLOSE_REPLACED = 4002;
/// A token this close to expiring (ms) is replaced rather than reused.
const NET_TOKEN_MARGIN_MS = 60_000;
/// Consecutive tunnel attempts that die before the hello, with one token,
/// before we suspect the token and fetch a new one.
const NET_TOKEN_MAX_FAILS = 3;
/// Where this tab keeps its token, so a reload reuses it: tokens are limited
/// per day, and one is good for hours.
const NET_TOKEN_KEY = "nixvm.net.token";

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

// Network link state, shown in the toolbar:
// off | connecting | online | error | unavailable.
const netEnabled = ref(true);
const netState = ref("off");
// The token API refused us (daily limit reached, or down): shown as a banner.
const netUnavailable = ref(false);
const netAddr = ref("");
// When the tunnel, and the IPv6 address leased with it, expires (Unix ms;
// 0: unknown), and a clock ticking while online so the countdown updates.
const netExpires = ref(0);
const netNow = ref(Date.now());
// Briefly true after the address was copied, to say so on the button.
const netCopied = ref(false);
let netCopiedTimer = 0;

/// "01:52:59" — what is left until `until` (Unix ms), as a clock.
function remaining(until, now) {
  const s = Math.max(0, Math.floor((until - now) / 1000));
  const two = (n) => String(n).padStart(2, "0");
  return `${two(Math.floor(s / 3600))}:${two(Math.floor((s % 3600) / 60))}:${two(s % 60)}`;
}

const netLabel = computed(() => {
  if (!netEnabled.value) return "net: off";
  switch (netState.value) {
    case "online":
      if (netCopied.value) return "net: address copied";
      return netExpires.value
        ? `net: ${netAddr.value} · ${remaining(netExpires.value, netNow.value)}`
        : `net: ${netAddr.value}`;
    case "connecting":
      return "net: connecting…";
    case "error":
      return "net: offline (retrying)";
    case "unavailable":
      return "net: unavailable";
    default:
      return "net: off";
  }
});
const netTitle = computed(() => {
  if (!netEnabled.value) return "Guest networking is off. Click to connect.";
  const online = netState.value === "online";
  const addr = online && netAddr.value.includes(":") ? ` Guest IPv6: ${netAddr.value}.` : "";
  const until =
    online && netExpires.value
      ? ` Valid until ${new Date(netExpires.value).toLocaleTimeString()} (then a new tunnel, and a new address).`
      : "";
  const click = online && netAddr.value.includes(":") ? " Click to copy the address." : "";
  return `Guest networking via a WebSocket IP tunnel (grouterd).${addr}${until}${click}`;
});
const netCanCopy = computed(
  () => netEnabled.value && netState.value === "online" && netAddr.value.includes(":"),
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

// ---- diagnostics (`?netdebug` in the page URL) ------------------------
// Logs every tunnel packet, decoded, and a status line every 5 s, to the
// browser console — what is needed to tell a stalled network from a stalled
// guest when something hangs.
const netDebug = new URLSearchParams(location.search).has("netdebug");
const diag = { out: 0, in: 0, outB: 0, inB: 0, pumps: 0, lastPumpMs: 0, lastIn: 0, watchdogPumps: 0 };
const diagT0 = performance.now();

function describePacket(b) {
  const hex = (x) => x.toString(16);
  const v = b[0] >> 4;
  if (v === 4 && b.length >= 20) {
    const ihl = (b[0] & 15) * 4;
    const proto = b[9];
    const l4 = b.subarray(ihl);
    let d = `v4 ${b.subarray(12, 16).join(".")} > ${b.subarray(16, 20).join(".")} proto ${proto}`;
    const be16 = (o) => (l4[o] << 8) | l4[o + 1];
    if (proto === 6 && l4.length >= 20) d += ` ${be16(0)}>${be16(2)} flags ${hex(l4[13])} len ${b.length - ihl - (l4[12] >> 4) * 4}`;
    if (proto === 17 && l4.length >= 8) d += ` ${be16(0)}>${be16(2)} len ${be16(4) - 8}`;
    if (proto === 1 && l4.length >= 1) d += ` icmp type ${l4[0]}`;
    return d;
  }
  if (v === 6 && b.length >= 40) {
    const w = (o) => hex((b[o] << 8) | b[o + 1]);
    const addr = (o) => [...Array(8)].map((_, i) => w(o + 2 * i)).join(":");
    const nh = b[6];
    const l4 = b.subarray(40);
    const be16 = (o) => (l4[o] << 8) | l4[o + 1];
    let d = `v6 ${addr(8)} > ${addr(24)} nh ${nh}`;
    if (nh === 6 && l4.length >= 20) d += ` ${be16(0)}>${be16(2)} flags ${hex(l4[13])} len ${b.length - 40 - (l4[12] >> 4) * 4}`;
    if (nh === 17 && l4.length >= 8) d += ` ${be16(0)}>${be16(2)} len ${be16(4) - 8}`;
    if (nh === 58 && l4.length >= 1) d += ` icmp6 type ${l4[0]}`;
    return d;
  }
  return `?? ${b.length} B`;
}

function notePacket(dir, bytes) {
  if (dir === "out") {
    diag.out++;
    diag.outB += bytes.length;
  } else {
    diag.in++;
    diag.inB += bytes.length;
    diag.lastIn = performance.now();
  }
  if (netDebug) {
    const t = ((performance.now() - diagT0) / 1000).toFixed(3);
    console.debug(`[net ${t}s] ${dir === "out" ? "->" : "<-"} ${describePacket(bytes)}`);
  }
}

if (netDebug) {
  console.info("nixvm: netdebug on — tunnel packets and a status line every 5 s are logged here");
  setInterval(() => {
    if (!guestTerm || guestDead) return;
    let awaiting = "?";
    let pending = "?";
    try {
      awaiting = guestTerm.awaiting_input();
      pending = guestTerm.has_pending_work();
    } catch {
      // guest gone
    }
    const sinceIn = diag.lastIn ? `${((performance.now() - diag.lastIn) / 1000).toFixed(1)}s ago` : "never";
    console.info(
      `nixvm status: net ${netState.value} ws=${ws ? ws.readyState : "none"} | out ${diag.out} pkts ${diag.outB} B, in ${diag.in} pkts ${diag.inB} B, last in ${sinceIn} | pumps ${diag.pumps} (last ${diag.lastPumpMs.toFixed(1)} ms, watchdog ${diag.watchdogPumps}) | command running ${commandRunning}, awaiting input ${awaiting}, pending work ${pending}`,
    );
  }, 5000);
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
    const started = performance.now();
    out = guestTerm.pump();
    diag.pumps++;
    diag.lastPumpMs = performance.now() - started;
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
    if (guestTerm.is_busy()) {
      // The guest is computing: its time slice ran out. Let the browser
      // paint, handle input and the tunnel, then carry on right away.
      schedulePump(0);
      return;
    }
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
  if (pending) {
    schedulePump(20);
  } else if (commandRunning && !awaiting) {
    // A command is still running yet reports nothing to wait for: keep
    // pumping slowly anyway, so a wake-up the guest's own bookkeeping misses
    // can never leave the page waiting forever.
    diag.watchdogPumps++;
    schedulePump(250);
  }
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
        // Ctrl-C: SIGINT to the running command (the shell itself is left
        // alone, like an interactive shell). This terminal has no tty line
        // discipline, so a ^C byte on stdin would just be data.
        writeRaw("^C\r\n");
        lineBuffer = "";
        let wasRunning = false;
        try {
          wasRunning = guestTerm.interrupt();
        } catch (err) {
          surfaceGuestCrash(err);
          break;
        }
        if (wasRunning || commandRunning) {
          // Let it handle the signal; the prompt returns when the shell
          // reads again.
          commandRunning = true;
          cancelPump();
          runPump();
        } else {
          writePrompt();
        }
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
let netClockTimer = null;
let netRetryDelay = 2000;

// The tunnel token: `{ token, expires }` (expires in Unix seconds, from the
// token API). Reused across reconnects until it nears expiry — tokens are
// rationed per day, while one lasts hours.
let netToken = loadNetToken();
let netTokenFails = 0;

function loadNetToken() {
  try {
    const t = JSON.parse(sessionStorage.getItem(NET_TOKEN_KEY) ?? "null");
    return t && typeof t.token === "string" && Number.isFinite(t.expires) ? t : null;
  } catch {
    return null;
  }
}

function saveNetToken(t) {
  netToken = t;
  netTokenFails = 0;
  try {
    if (t) sessionStorage.setItem(NET_TOKEN_KEY, JSON.stringify(t));
    else sessionStorage.removeItem(NET_TOKEN_KEY);
  } catch {
    // storage unavailable (private mode): the in-memory copy still works
  }
}

function netTokenUsable() {
  return netToken !== null && netToken.expires * 1000 - Date.now() > NET_TOKEN_MARGIN_MS;
}

/// A token for the tunnel: the cached one while it is good, else a new one.
/// Resolves to the token string, or throws: `{ unavailable: true }` when the
/// API answered with a refusal (daily limit, service off), a plain error when
/// it couldn't be reached at all (offline — worth retrying).
async function netGetToken() {
  if (netTokenUsable()) return netToken.token;
  saveNetToken(null);
  const res = await fetch(NET_TOKEN_URL, { cache: "no-store" });
  let body = null;
  try {
    body = await res.json();
  } catch {
    // not JSON: treated as a refusal below
  }
  const data = body?.data;
  if (!res.ok || body?.result !== "success" || typeof data?.token !== "string") {
    const why = body?.error ?? body?.message ?? `HTTP ${res.status}`;
    throw Object.assign(new Error(`token API: ${why}`), { unavailable: true });
  }
  // Without an expiry, assume the documented two hours.
  const expires = Number.isFinite(data.expires) ? data.expires : Date.now() / 1000 + 7200;
  saveNetToken({ token: data.token, expires });
  return data.token;
}

// Send every packet the guest's stack has queued.
function flushNet() {
  if (!guestTerm || !ws || ws.readyState !== WebSocket.OPEN) return;
  for (;;) {
    const p = guestTerm.net_next_packet();
    if (p === undefined || p === null) break;
    notePacket("out", p);
    ws.send(p);
  }
}

function netStopTimers() {
  clearInterval(netTickTimer);
  clearInterval(netKeepaliveTimer);
  clearTimeout(netRetryTimer);
  clearInterval(netClockTimer);
  netTickTimer = netKeepaliveTimer = netRetryTimer = netClockTimer = null;
  netExpires.value = 0;
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
    token = await netGetToken();
  } catch (err) {
    if (gen !== netGen) return;
    console.warn("nixvm: network token:", err);
    if (err?.unavailable) {
      // Refused (e.g. today's free tokens are used up). Don't keep asking —
      // that only burns quota; the banner / net button retries on demand.
      netStopTimers();
      netState.value = "unavailable";
      netUnavailable.value = true;
      return;
    }
    netRetryLater(); // couldn't reach the API (offline): back off, retry
    return;
  }
  if (gen !== netGen || !guestTerm) return;
  netUnavailable.value = false;

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
        // Show the guest's public IPv6 address. The IPv4 one is a NATed
        // inside address (the same for every client), so not shown.
        netAddr.value = guestTerm.net_ipv6() ?? "online";
        // The address lasts as long as the tunnel: until `expires` (Unix s).
        netExpires.value = Number.isFinite(h.expires) ? h.expires * 1000 : 0;
        netNow.value = Date.now();
        netClockTimer = setInterval(() => {
          netNow.value = Date.now();
        }, 1000);
        netState.value = "online";
        netRetryDelay = 2000;
        netTokenFails = 0;
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
      const pkt = new Uint8Array(ev.data);
      notePacket("in", pkt);
      guestTerm.net_input(pkt);
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
    if (ev.code === NET_CLOSE_EXPIRED) {
      saveNetToken(null); // the next attempt fetches a fresh token
    } else if (!hello && ++netTokenFails >= NET_TOKEN_MAX_FAILS) {
      // Never got a hello with this token, repeatedly: it may have been
      // refused (the browser hides the HTTP status), so try a fresh one.
      saveNetToken(null);
    }
    // Dropped connection, server restart, expiry: reconnect, reusing the
    // token while it's still good.
    netRetryLater();
  };
}

function toggleNet() {
  // While refused, a click is "try again" rather than "turn off".
  if (netEnabled.value && netState.value === "unavailable") {
    netRetryNow();
    return;
  }
  netEnabled.value = !netEnabled.value;
  if (!netEnabled.value) netUnavailable.value = false;
  if (netEnabled.value) {
    netRetryDelay = 2000;
    netConnect();
  } else {
    netDisconnect();
  }
}

// The address part of the net control: copies the guest's IPv6 while online;
// otherwise it does what the power toggle would (connect, or retry).
async function netLabelClick() {
  if (!netCanCopy.value) {
    if (!netEnabled.value || netState.value === "unavailable") toggleNet();
    return;
  }
  try {
    await navigator.clipboard.writeText(netAddr.value);
  } catch {
    // No clipboard access (insecure context, denied): select the text so
    // it can be copied by hand.
    const sel = window.getSelection();
    if (sel && netLabelEl.value) sel.selectAllChildren(netLabelEl.value);
    return;
  }
  netCopied.value = true;
  clearTimeout(netCopiedTimer);
  netCopiedTimer = setTimeout(() => (netCopied.value = false), 1500);
}
const netLabelEl = ref(null);

function netRetryNow() {
  netUnavailable.value = false;
  netEnabled.value = true;
  netRetryDelay = 2000;
  netConnect();
}

function netDismissBanner() {
  netUnavailable.value = false;
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
      <div class="net-group" :class="`is-${netEnabled ? netState : 'off'}`">
        <button
          ref="netLabelEl"
          class="net-btn"
          :title="netTitle"
          :disabled="!hasBooted"
          @click="netLabelClick"
        >
          {{ netLabel }}
        </button>
        <button
          class="net-power"
          :title="netEnabled ? 'Turn guest networking off' : 'Turn guest networking on'"
          :aria-label="netEnabled ? 'Turn guest networking off' : 'Turn guest networking on'"
          :aria-pressed="netEnabled"
          :disabled="!hasBooted"
          @click="toggleNet"
        >
          ⏻
        </button>
      </div>
      <button class="reboot-btn" :disabled="rebootDisabled" @click="hasBooted ? reboot() : boot()">
        {{ bootLabel }}
      </button>
    </div>
    <div v-if="netUnavailable" class="net-banner" role="status">
      <span>
        Network is not available for free right now — the shell still works,
        just offline.
      </span>
      <button class="net-banner-btn" @click="netRetryNow">Retry</button>
      <button class="net-banner-btn" aria-label="Dismiss" @click="netDismissBanner">✕</button>
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

.net-group {
  flex: none;
  display: flex;
  min-width: 0;
}

.net-btn {
  flex: 0 1 auto;
  min-width: 0;
  border-top-right-radius: 0 !important;
  border-bottom-right-radius: 0 !important;
  /* Fixed-width digits: the countdown ticks every second without the
     button's width jittering. */
  font-variant-numeric: tabular-nums;
  background: var(--panel);
  color: var(--muted);
  border: 1px solid var(--panel-border);
  border-radius: 0.4rem;
  padding: 0.3rem 0.7rem;
  font-size: 0.8rem;
  font-family: inherit;
  cursor: pointer;
  max-width: 24rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.net-power {
  flex: none;
  background: var(--panel);
  color: var(--muted);
  border: 1px solid var(--panel-border);
  border-left: none;
  border-radius: 0 0.4rem 0.4rem 0;
  padding: 0.3rem 0.55rem;
  font-size: 0.8rem;
  font-family: inherit;
  cursor: pointer;
}

.net-power:hover:not(:disabled),
.net-btn:hover:not(:disabled) {
  border-color: var(--accent);
}

.net-power:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.is-online .net-btn,
.is-online .net-power {
  color: #8ce99a;
}

.is-connecting .net-btn,
.is-connecting .net-power {
  color: #ffd43b;
}

.is-error .net-btn,
.is-unavailable .net-btn {
  color: var(--danger);
}

.net-banner {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.45rem 0.7rem;
  font-size: 0.85rem;
  color: var(--fg);
  background: color-mix(in srgb, #ffd43b 14%, var(--panel));
  border: 1px solid color-mix(in srgb, #ffd43b 45%, var(--panel-border));
  border-radius: 0.4rem;
}

.net-banner span {
  flex: 1 1 auto;
  min-width: 0;
}

.net-banner-btn {
  flex: none;
  background: transparent;
  color: var(--fg);
  border: 1px solid var(--panel-border);
  border-radius: 0.35rem;
  padding: 0.2rem 0.55rem;
  font-size: 0.8rem;
  font-family: inherit;
  cursor: pointer;
}

.net-banner-btn:hover {
  border-color: var(--accent);
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
