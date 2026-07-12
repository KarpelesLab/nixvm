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
// `pump()` runs the guest synchronously to completion of that input, so a
// long-running command will briefly freeze the tab — that's a known,
// accepted trade-off of running a Linux userland on the main thread.
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

async function afterStdinChanged() {
  await tick();
  const out = guestTerm.pump();
  writeBytes(out);
  if (!guestTerm.is_running()) {
    const code = guestTerm.exit_code();
    status.value = "exited";
    writeRaw(`\r\n[ shell exited with code ${code} — click Reboot to start a new session ]\r\n`);
  } else {
    writePrompt();
  }
}

async function handleInput(data) {
  if (busy || status.value !== "ready") return;
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
  try {
    const targz = await fetchRootfsTarGz(archId);
    const mod = await loadWasmModule();
    status.value = "booting";
    lineBuffer = "";
    atLineStart = true;
    await tick();
    // The wasm Terminal takes the raw .tar.gz and gunzips it in-process; the
    // guest arch is auto-detected from the ELFs inside it.
    guestTerm = new mod.Terminal(targz, ["/bin/busybox", "sh"]);
    const out = guestTerm.pump();
    writeBanner(`nixvm — Alpine Linux (${archId}), running entirely in your browser.`);
    writeBanner('Type commands and press Enter. Try: uname -m; ls /; cat /etc/os-release');
    writeBytes(out);
    hasBooted.value = true;
    status.value = "ready";
    writePrompt();
  } catch (err) {
    status.value = "error";
    writeErrorBanner(`boot failed: ${err?.message ?? err}`);
  }
}

async function reboot() {
  if (rebootDisabled.value) return;
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
