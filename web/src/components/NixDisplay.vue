<script setup>
import { onBeforeUnmount, onMounted, ref } from "vue";

import { BUTTON_CODES, evdevKey } from "../keymap.js";

// ---------------------------------------------------------------------------
// The guest's graphical display, on a <canvas>.
//
// nixvm runs a Wayland compositor for the guest (`Terminal.enable_display`);
// this pane shows its frames and feeds it the user's keyboard and pointer.
// Each animation frame it asks for the frame (`display_frame()`, which hands
// back pixels only when they changed, and also tells the guest's clients a
// frame was shown, so their animations advance) and paints it with
// putImageData. One display pixel is one CSS pixel: the display is resized to
// the pane (debounced), so it's drawn 1:1.
//
// Keys go to the guest only while the canvas has the focus (click it); the
// terminal keeps them otherwise. Codes are physical keys (`KeyboardEvent.code`)
// mapped to Linux evdev codes; the guest's keymap does the rest.
// ---------------------------------------------------------------------------

const props = defineProps({
  // Returns the running wasm `Terminal` (with a display), or null.
  term: { type: Function, required: true },
});
// `input`: something was sent to the guest (pump it soon).
// `crash`: a call into the wasm threw (the instance trapped).
const emit = defineEmits(["input", "crash"]);

const paneEl = ref(null);
const canvasEl = ref(null);
const focused = ref(false);
const hasFrame = ref(false);

/// Smallest/largest display edge we ask for.
const MIN_EDGE = 64;
const MAX_EDGE = 4096;
/// Resizing the display reallocates the guest's buffers: wait for the pane to
/// settle first.
const RESIZE_DEBOUNCE_MS = 250;

let ctx = null;
let raf = 0;
let resizeObserver = null;
let resizeTimer = 0;
let lastTerm = null;
let requested = [0, 0];
// The latest pointer position not yet sent (moves are coalesced per frame).
let pendingMove = null;
// Keys held down on the canvas, released when it loses the focus.
const held = new Set();

/// The size the display should be to fill the pane, in CSS pixels.
function paneSize() {
  const el = paneEl.value;
  if (!el) return [640, 400];
  const clamp = (v) => Math.max(MIN_EDGE, Math.min(MAX_EDGE, Math.floor(v)));
  return [clamp(el.clientWidth), clamp(el.clientHeight)];
}

defineExpose({ paneSize });

/// Call into the guest; a throw means the wasm instance trapped.
function call(fn) {
  const t = props.term();
  if (!t) return undefined;
  try {
    return fn(t);
  } catch (err) {
    emit("crash", err);
    return undefined;
  }
}

function sent() {
  emit("input");
}

function flushMove() {
  if (!pendingMove) return;
  const [x, y] = pendingMove;
  pendingMove = null;
  call((t) => t.display_pointer_move(x, y));
  sent();
}

function frame() {
  raf = requestAnimationFrame(frame);
  const t = props.term();
  if (t !== lastTerm) {
    // A new guest (boot or reboot): it starts at the size it was made with.
    lastTerm = t;
    hasFrame.value = false;
    held.clear();
    requested = t ? (call((tt) => tt.display_size()) ?? [0, 0]) : [0, 0];
    if (t) scheduleResize();
  }
  if (!t) return;
  flushMove();
  const px = call((tt) => tt.display_frame());
  if (!px) return;
  const [w, h] = call((tt) => tt.display_size()) ?? [0, 0];
  if (!w || !h || px.length !== w * h * 4) return;
  const canvas = canvasEl.value;
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
    ctx = null;
  }
  ctx ??= canvas.getContext("2d");
  ctx.putImageData(new ImageData(new Uint8ClampedArray(px.buffer, px.byteOffset, px.length), w, h), 0, 0);
  hasFrame.value = true;
}

function scheduleResize() {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    const [w, h] = paneSize();
    if (w === requested[0] && h === requested[1]) return;
    requested = [w, h];
    call((t) => t.display_resize(w, h));
    sent();
  }, RESIZE_DEBOUNCE_MS);
}

/// The pointer's position on the display, in display pixels.
function displayPos(ev) {
  const canvas = canvasEl.value;
  const r = canvas.getBoundingClientRect();
  if (!r.width || !r.height) return [0, 0];
  return [((ev.clientX - r.left) * canvas.width) / r.width, ((ev.clientY - r.top) * canvas.height) / r.height];
}

function onPointerMove(ev) {
  pendingMove = displayPos(ev);
}

function onPointerDown(ev) {
  const code = BUTTON_CODES[ev.button];
  canvasEl.value.focus();
  if (code === undefined) return;
  ev.preventDefault();
  canvasEl.value.setPointerCapture?.(ev.pointerId);
  pendingMove = displayPos(ev);
  flushMove();
  call((t) => t.display_pointer_button(code, true));
  sent();
}

function onPointerUp(ev) {
  const code = BUTTON_CODES[ev.button];
  if (code === undefined) return;
  ev.preventDefault();
  pendingMove = displayPos(ev);
  flushMove();
  call((t) => t.display_pointer_button(code, false));
  sent();
}

function onPointerLeave() {
  pendingMove = null;
  call((t) => t.display_pointer_leave());
  sent();
}

/// Wheel deltas in pixels (lines and pages are converted).
function onWheel(ev) {
  ev.preventDefault();
  const unit = ev.deltaMode === 1 ? 16 : ev.deltaMode === 2 ? canvasEl.value.height : 1;
  flushMove();
  call((t) => t.display_scroll(ev.deltaX * unit, ev.deltaY * unit));
  sent();
}

function onKey(ev, pressed) {
  const code = evdevKey(ev);
  if (code === undefined) return;
  // Everything goes to the guest — Tab, Backspace, Ctrl-W — not the page.
  ev.preventDefault();
  ev.stopPropagation();
  // The guest's clients repeat held keys themselves.
  if (pressed && ev.repeat) return;
  if (pressed) held.add(code);
  else if (!held.delete(code)) return;
  call((t) => t.display_key(code, pressed));
  sent();
}

function onFocus() {
  focused.value = true;
}

function onBlur() {
  focused.value = false;
  // Keys still down when the focus left (Alt-Tab away) would stick otherwise.
  for (const code of held) call((t) => t.display_key(code, false));
  if (held.size) sent();
  held.clear();
}

onMounted(() => {
  resizeObserver = new ResizeObserver(() => scheduleResize());
  resizeObserver.observe(paneEl.value);
  // Wheel needs a non-passive listener to stop the page scrolling.
  canvasEl.value.addEventListener("wheel", onWheel, { passive: false });
  raf = requestAnimationFrame(frame);
});

onBeforeUnmount(() => {
  cancelAnimationFrame(raf);
  clearTimeout(resizeTimer);
  resizeObserver?.disconnect();
  canvasEl.value?.removeEventListener("wheel", onWheel);
});
</script>

<template>
  <div ref="paneEl" class="display-pane" :class="{ 'is-focused': focused }">
    <canvas
      ref="canvasEl"
      class="display-canvas"
      tabindex="0"
      aria-label="Guest display: click to send it your keyboard and mouse"
      @pointermove="onPointerMove"
      @pointerdown="onPointerDown"
      @pointerup="onPointerUp"
      @pointerleave="onPointerLeave"
      @keydown="onKey($event, true)"
      @keyup="onKey($event, false)"
      @focus="onFocus"
      @blur="onBlur"
      @contextmenu.prevent
    ></canvas>
    <div v-if="!hasFrame" class="display-empty">
      The guest's display appears here once it boots. Run a Wayland client in the terminal.
    </div>
    <div v-else-if="!focused" class="display-hint">click to type here</div>
  </div>
</template>

<style scoped>
.display-pane {
  position: relative;
  width: 100%;
  min-width: 0;
  height: min(60vh, 480px);
  min-height: 200px;
  background: #20242c;
  border: 1px solid var(--panel-border);
  border-radius: 0.5rem;
  overflow: hidden;
}

.display-pane.is-focused {
  border-color: var(--accent);
}

.display-canvas {
  display: block;
  outline: none;
  touch-action: none;
  cursor: default;
}

.display-empty {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1rem;
  text-align: center;
  color: var(--muted);
  font-size: 0.85rem;
  pointer-events: none;
}

.display-hint {
  position: absolute;
  right: 0.5rem;
  bottom: 0.4rem;
  padding: 0.1rem 0.45rem;
  border-radius: 0.3rem;
  background: rgb(0 0 0 / 45%);
  color: #cfd3da;
  font-size: 0.75rem;
  pointer-events: none;
}
</style>
