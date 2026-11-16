// Run a Wayland client on nixvm's built-in display in headless Chromium and
// see how it shows: the display's frames on a canvas as the demo page does
// (WebGL2 presentation when the GPU is on, putImageData otherwise), what the
// presentation costs per frame, and a screenshot of the canvas at the end.
//
//   node scripts/display-check.mjs --pkg <wasm-pack out dir, built with `gpu`> \
//        --rootfs <x86_64 root .tar.gz with Mesa and the client> \
//        [--no-gpu] [--cmd '<shell command>'] [--seconds 20] [--size 640x400] \
//        [--out <file.png>] [--expect '<regexp the output must match>']
//        [--resize WxH (halfway through)] [--pointer (move it all along)]
//        [--2d (with the GPU on a hidden canvas, copy frames out instead)]
//        [--interactive (type the command into an interactive shell)]
//
// `--cmd` defaults to es2gears_wayland. With the GPU (the default) the
// canvas's WebGL2 context is the guest's GPU (`attach_webgl2`) and frames are
// shown with `display_present`; with `--no-gpu` Mesa renders on llvmpipe and
// frames come out with `display_frame` + putImageData. Prints the guest's
// output, the compositor used, frames shown and their cost; exits non-zero if
// the guest's output matches `--fail` (default: EGL errors) or `--expect`
// doesn't match.
//
// Needs `npm install` (playwright-core) and a Chromium: playwright's own
// (`npx playwright-core install chromium`) or `$CHROME`.

import { createServer } from "node:http";
import { readFileSync, writeFileSync, existsSync, statSync } from "node:fs";
import { extname, join, resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { chromium } from "playwright-core";

const here = dirname(fileURLToPath(import.meta.url));
const { values: opt } = parseArgs({
  options: {
    pkg: { type: "string" },
    rootfs: { type: "string" },
    "no-gpu": { type: "boolean", default: false },
    cmd: { type: "string", default: "es2gears_wayland" },
    seconds: { type: "string", default: "20" },
    size: { type: "string", default: "640x400" },
    out: { type: "string" },
    resize: { type: "string" },
    pointer: { type: "boolean", default: false },
    "2d": { type: "boolean", default: false },
    interactive: { type: "boolean", default: false },
    expect: { type: "string" },
    fail: { type: "string", default: "EGL_BAD|eglSwapBuffers failed|Segmentation" },
  },
});
if (!opt.pkg || !opt.rootfs) {
  console.error("usage: display-check.mjs --pkg <dir> --rootfs <root.tar.gz> [--no-gpu] [--cmd <sh>] [--seconds N]");
  process.exit(2);
}
const [width, height] = opt.size.split("x").map(Number);

const HARNESS = `<!doctype html><meta charset="utf-8"><title>nixvm display check</title>
<script type="importmap">{ "imports": { "pktkit": "./pktkit.js", "purecrypto": "./purecrypto.js" } }</script>
<canvas id="c" width="1" height="1"></canvas>
<script type="module">
window.runGuest = async (cmd, useGpu, w, h, seconds, resize, pointer, use2d, interactive) => {
  const mod = await import("./pkg/nixvm.js");
  await mod.default();
  const tar = new Uint8Array(await (await fetch("./rootfs.tar.gz")).arrayBuffer());
  const net = new mod.Network();
  const env = mod.Terminal.display_env();
  // Interactive: an interactive shell is given the command as typed input,
  // as on the demo page.
  const argv = interactive ? ["/bin/busybox", "sh"] : ["/bin/sh", "-c", cmd];
  const term = mod.Terminal.new_on(tar, argv, net, env);
  if (interactive) term.write_stdin(new TextEncoder().encode(cmd + "\\n"));
  term.enable_display(w, h);
  const canvas = document.getElementById("c");
  let gl = null;
  let renderer = "none";
  if (useGpu) {
    // With --2d the GPU gets a hidden canvas and frames are copied out.
    const glCanvas = use2d ? document.createElement("canvas") : canvas;
    gl = glCanvas.getContext("webgl2", { alpha: false, antialias: false, depth: false, stencil: false, preserveDrawingBuffer: false });
    if (!gl) throw new Error("no WebGL2 in this browser");
    const dbg = gl.getExtension("WEBGL_debug_renderer_info");
    renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    term.attach_webgl2(gl);
  }
  const glDisplay = gl && !use2d;
  let ctx = null;
  const stats = { shown: 0, ms: 0, maxMs: 0, rafs: 0 };
  let done = false;
  // Show a frame the way the demo page does; returns whether one was drawn.
  const show = (force) => {
    const t0 = performance.now();
    let drew = false;
    if (glDisplay) {
      const [cw, ch] = term.display_current_size();
      if (canvas.width !== cw || canvas.height !== ch) {
        canvas.width = cw;
        canvas.height = ch;
        force = true;
      }
      drew = term.display_present(force);
    } else {
      const px = term.display_frame();
      if (px) {
        const [fw, fh] = term.display_size();
        if (canvas.width !== fw || canvas.height !== fh) {
          canvas.width = fw;
          canvas.height = fh;
          ctx = null;
        }
        ctx ??= canvas.getContext("2d");
        ctx.putImageData(new ImageData(new Uint8ClampedArray(px.buffer, px.byteOffset, px.length), fw, fh), 0, 0);
        drew = true;
      }
    }
    const dt = performance.now() - t0;
    if (drew) {
      stats.shown++;
      stats.ms += dt;
      stats.maxMs = Math.max(stats.maxMs, dt);
    }
    return drew;
  };
  const raf = () => {
    if (done) return;
    stats.rafs++;
    show(false);
    requestAnimationFrame(raf);
  };
  const t0 = performance.now();
  requestAnimationFrame(raf);
  const chunks = [];
  let resized = !resize;
  let k = 0;
  while (term.is_running() && performance.now() - t0 < seconds * 1000) {
    if (!resized && performance.now() - t0 > (seconds * 1000) / 2) {
      resized = true;
      term.display_resize(resize[0], resize[1]);
    }
    if (pointer) {
      k++;
      term.display_pointer_move(50 + (k % 200), 50 + ((k >> 1) % 100));
    }
    chunks.push(term.pump());
    await new Promise((r) => setTimeout(r, 0));
  }
  done = true;
  // The last frame, drawn and captured in one animation frame (the drawing
  // buffer is only valid until the page composites), with its center pixel.
  const cx = canvas.width >> 1;
  const cy = canvas.height >> 1;
  const [lastDrawn, png, center] = await new Promise((r) =>
    requestAnimationFrame(() => {
      const d = show(true);
      const c = new Uint8Array(4);
      if (glDisplay) {
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
        gl.readPixels(cx, canvas.height - 1 - cy, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, c);
      }
      r([d, canvas.toDataURL("image/png"), c]);
    }),
  );
  // With the GPU, check that against the frame read back from the GPU.
  let probe = "";
  if (glDisplay) {
    let f = null;
    for (let n = 0; n < 100 && !f; n++) {
      term.pump();
      f = term.display_frame();
    }
    f ??= new Uint8Array(0);
    const i = (cy * canvas.width + cx) * 4;
    probe = "last frame drawn " + lastDrawn + "; canvas center " + [...center] + ", frame center (a frame later) " + [...f.subarray(i, i + 4)];
  }
  const out = new TextDecoder().decode(new Uint8Array(chunks.flatMap((c) => [...c])));
  return {
    out,
    renderer,
    compositor: term.display_compositor ? term.display_compositor() : "software (old build)",
    probe,
    running: term.is_running(),
    stats,
    seconds: (performance.now() - t0) / 1000,
    png,
  };
};
window.harnessReady = true;
</script>`;

const TYPES = { ".js": "text/javascript", ".wasm": "application/wasm", ".html": "text/html", ".gz": "application/gzip" };
const publicDir = resolve(here, "../public");
function fileFor(url) {
  const p = decodeURIComponent(new URL(url, "http://x").pathname);
  if (p === "/rootfs.tar.gz") return resolve(opt.rootfs);
  if (p.startsWith("/pkg/")) return join(resolve(opt.pkg), p.slice(5).replace(/\.\.+/g, ""));
  if (p === "/pktkit.js" || p === "/purecrypto.js") return join(publicDir, p.slice(1));
  return null;
}
const server = createServer((req, res) => {
  if (req.url === "/" || req.url === "/index.html") {
    res.writeHead(200, { "content-type": "text/html" });
    return res.end(HARNESS);
  }
  const f = fileFor(req.url);
  if (!f || !existsSync(f) || !statSync(f).isFile()) {
    res.writeHead(404);
    return res.end();
  }
  res.writeHead(200, { "content-type": TYPES[extname(f)] ?? "application/octet-stream" });
  res.end(readFileSync(f));
});
await new Promise((r) => server.listen(0, "127.0.0.1", r));
const url = `http://127.0.0.1:${server.address().port}/`;

// WebGL2 without a GPU: SwiftShader through ANGLE.
const browser = await chromium.launch({
  executablePath: process.env.CHROME || undefined,
  headless: true,
  args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"],
});
let failed = false;
try {
  const page = await browser.newPage();
  let glErrors = 0;
  page.on("console", (m) => {
    if (m.type() !== "error" && m.type() !== "warning") return;
    if (m.text().includes("GL Driver Message") && m.text().includes("Performance")) return;
    if (/WebGL: [A-Z_]+:/.test(m.text())) glErrors++;
    console.log(`[browser ${m.type()}] ${m.text()}`);
  });
  page.on("pageerror", (e) => console.log(`[pageerror] ${e.message}`));
  await page.goto(url);
  await page.waitForFunction(() => window.harnessReady);
  page.setDefaultTimeout((Number(opt.seconds) + 600) * 1000);
  const script = `[ -e /bin/ls ] || /bin/busybox --install -s; ${opt.cmd}`;
  const resize = opt.resize ? opt.resize.split("x").map(Number) : null;
  const r = await page.evaluate(([c, g, w, h, s, z, p, d, i]) => window.runGuest(c, g, w, h, s, z, p, d, i), [
    script,
    !opt["no-gpu"],
    width,
    height,
    Number(opt.seconds),
    resize,
    opt.pointer,
    opt["2d"],
    opt.interactive,
  ]);
  console.log(r.out.trimEnd());
  const s = r.stats;
  console.log(
    `WebGL2: ${r.renderer}; compositor: ${r.compositor}; ${r.seconds.toFixed(1)} s, guest ${r.running ? "still running" : "exited"}`,
  );
  console.log(
    `frames shown: ${s.shown} of ${s.rafs} animation frames (${(s.shown / r.seconds).toFixed(1)}/s); ` +
      `cost per shown frame: ${(s.ms / Math.max(1, s.shown)).toFixed(2)} ms avg, ${s.maxMs.toFixed(2)} ms max`,
  );
  if (r.probe) console.log(r.probe);
  if (opt.out) {
    writeFileSync(opt.out, Buffer.from(r.png.split(",")[1], "base64"));
    console.log(`canvas saved to ${opt.out}`);
  }
  if (new RegExp(opt.fail).test(r.out)) {
    console.log(`FAIL: the output matches /${opt.fail}/`);
    failed = true;
  }
  if (opt.expect && !new RegExp(opt.expect).test(r.out)) {
    console.log(`FAIL: the output doesn't match /${opt.expect}/`);
    failed = true;
  }
  if (glErrors) {
    console.log(`FAIL: ${glErrors} WebGL errors`);
    failed = true;
  }
} catch (e) {
  console.log(`ERROR ${e.message}`);
  failed = true;
} finally {
  await browser.close();
  server.close();
}
process.exit(failed ? 1 : 0);
