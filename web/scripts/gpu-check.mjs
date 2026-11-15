// Run guest OpenGL on nixvm's WebGL2 GPU backend in headless Chromium, with
// no GPU needed (WebGL2 on SwiftShader), and check what it renders.
//
//   node scripts/gpu-check.mjs --pkg <wasm-pack out dir, built with `gpu`> \
//        --rootfs <x86_64 root .tar.gz with Mesa and the test programs> \
//        [--ref <dir>] [--out <dir>] [--no-gpu] [--cmd '<shell command>']
//
// Serves a bare harness page (no Vue app), boots the root with a canvas's
// WebGL2 context attached (`Terminal.attach_webgl2`; `--no-gpu` leaves it
// off, so Mesa falls back to llvmpipe in the same browser), and runs
// `/opt/gpu_triangle` and `/opt/gpu_scene` (tests/guest/*.c) or `--cmd`.
// Each picture comes back over the guest's stdout as base64 and is saved as
// `<out>/<prog>.ppm`; with `--ref`, it is compared with `<ref>/<prog>.ppm`
// (a native `NIXVM_GPU=egl` render) and the run fails past a small
// per-channel tolerance. The exit status says whether everything passed.
//
// Needs `npm install` (playwright-core) and a Chromium: playwright's own
// (`npx playwright-core install chromium`) or `$CHROME`.

import { createServer } from "node:http";
import { mkdirSync, readFileSync, writeFileSync, existsSync, statSync } from "node:fs";
import { extname, join, resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";

import { chromium } from "playwright-core";

const here = dirname(fileURLToPath(import.meta.url));
const { values: opt } = parseArgs({
  options: {
    pkg: { type: "string" },
    rootfs: { type: "string" },
    ref: { type: "string" },
    out: { type: "string", default: "gpu-check-out" },
    "no-gpu": { type: "boolean", default: false },
    cmd: { type: "string" },
    timeout: { type: "string", default: "600" },
    tolerance: { type: "string", default: "8" },
  },
});
if (!opt.pkg || !opt.rootfs) {
  console.error("usage: gpu-check.mjs --pkg <dir> --rootfs <root.tar.gz> [--ref <dir>] [--out <dir>] [--no-gpu] [--cmd <sh>]");
  process.exit(2);
}

const PROGS = ["gpu_triangle", "gpu_scene"];
// One shell command: each program renders to a file, which is printed as
// base64 between markers. (A root installed without running apk's scripts
// lacks busybox's applet links: make them first.)
const script =
  "[ -e /bin/ls ] || /bin/busybox --install -s; " +
  (opt.cmd ??
  PROGS.map(
    (p) =>
      `/opt/${p} /tmp/${p}.ppm; echo "@@EXIT ${p} $?"; ` +
      `[ -f /tmp/${p}.ppm ] && { echo "@@PPM ${p}"; base64 /tmp/${p}.ppm; echo "@@END"; }`,
  ).join("; "));

const HARNESS = `<!doctype html><meta charset="utf-8"><title>nixvm gpu check</title>
<script type="importmap">{ "imports": { "pktkit": "./pktkit.js", "purecrypto": "./purecrypto.js" } }</script>
<canvas id="c" width="1" height="1"></canvas>
<script type="module">
window.runGuest = async (cmd, useGpu) => {
  const mod = await import("./pkg/nixvm.js");
  await mod.default();
  const tar = new Uint8Array(await (await fetch("./rootfs.tar.gz")).arrayBuffer());
  const t0 = performance.now();
  const term = new mod.Terminal(tar, ["/bin/sh", "-c", cmd]);
  let renderer = "none";
  if (useGpu) {
    const gl = document.getElementById("c").getContext("webgl2");
    if (!gl) throw new Error("no WebGL2 in this browser");
    const dbg = gl.getExtension("WEBGL_debug_renderer_info");
    renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
    term.attach_webgl2(gl);
  }
  const chunks = [];
  while (term.is_running()) {
    chunks.push(term.pump());
    await new Promise((r) => setTimeout(r, 0));
  }
  const out = new TextDecoder().decode(new Uint8Array(chunks.flatMap((c) => [...c])));
  return { out, exit: term.exit_code(), renderer, seconds: (performance.now() - t0) / 1000 };
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
  // Chrome reports GL errors as console warnings ("WebGL: INVALID_ENUM: …"):
  // any of them fails the check. Performance notes ("GPU stall due to
  // ReadPixels") are just noise.
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
  page.setDefaultTimeout(Number(opt.timeout) * 1000);
  const r = await page.evaluate(([c, g]) => window.runGuest(c, g), [script, !opt["no-gpu"]]);
  console.log(`WebGL2 renderer: ${r.renderer}; guest exit ${r.exit} after ${r.seconds.toFixed(1)} s`);
  // Everything but the base64 bodies.
  console.log(r.out.replace(/@@PPM (\S+)\n[\s\S]*?@@END\n/g, "[$1.ppm]\n").trimEnd());
  if (r.exit !== 0 && opt.cmd) failed = true;
  if (glErrors) {
    console.log(`FAIL: ${glErrors} WebGL errors`);
    failed = true;
  }

  if (!opt.cmd) {
    mkdirSync(opt.out, { recursive: true });
    const tol = Number(opt.tolerance);
    for (const p of PROGS) {
      const exit = r.out.match(new RegExp(`@@EXIT ${p} (\\d+)`))?.[1];
      const m = r.out.match(new RegExp(`@@PPM ${p}\\n([\\s\\S]*?)@@END`));
      if (exit !== "0" || !m) {
        console.log(`FAIL ${p}: exit ${exit}, ${m ? "" : "no picture"}`);
        failed = true;
        continue;
      }
      const ppm = Buffer.from(m[1].replace(/\s+/g, ""), "base64");
      writeFileSync(join(opt.out, `${p}.ppm`), ppm);
      if (!opt.ref) continue;
      const cmp = comparePpm(ppm, readFileSync(join(opt.ref, `${p}.ppm`)), tol);
      // GL lets implementations rasterize edges and resolve depth ties a
      // little differently (SwiftShader vs. a host GPU or llvmpipe): allow
      // a few pixels off, everything else within the tolerance.
      const ok = cmp.sizeMatch && cmp.over <= Math.max(4, cmp.pixels / 1000);
      if (!ok) failed = true;
      console.log(
        `${ok ? "PASS" : "FAIL"} ${p}: ` +
          (cmp.sizeMatch
            ? `${cmp.differing} of ${cmp.pixels} pixels differ, ${cmp.over} by more than ${tol} (max ${cmp.maxDiff})`
            : "size differs"),
      );
    }
  }
} catch (e) {
  console.log(`ERROR ${e.message}`);
  failed = true;
} finally {
  await browser.close();
  server.close();
}
process.exit(failed ? 1 : 0);

/// Compare two binary PPMs (P6, maxval 255): pixels that differ at all, and
/// by more than `tol` in some channel.
function comparePpm(a, b, tol) {
  const parse = (buf) => {
    const head = buf.subarray(0, 64).toString("latin1").match(/^P6\s+(\d+)\s+(\d+)\s+255\s/);
    return head ? { w: +head[1], h: +head[2], px: buf.subarray(head[0].length) } : null;
  };
  const x = parse(a);
  const y = parse(b);
  if (!x || !y || x.w !== y.w || x.h !== y.h) return { sizeMatch: false };
  let maxDiff = 0;
  let differing = 0;
  let over = 0;
  for (let i = 0; i < x.px.length; i += 3) {
    let d = 0;
    for (let c = 0; c < 3; c++) d = Math.max(d, Math.abs(x.px[i + c] - y.px[i + c]));
    if (d) differing++;
    if (d > tol) over++;
    maxDiff = Math.max(maxDiff, d);
  }
  return { sizeMatch: true, maxDiff, differing, over, pixels: x.w * x.h };
}
