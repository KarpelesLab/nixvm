# nixvm browser demo

A Vue 3 + Vite app that boots a real Alpine Linux minirootfs into the nixvm
wasm sandbox and drives an interactive `busybox sh` session in an
[xterm.js](https://xtermjs.org/) terminal, entirely client-side. This is what
`.github/workflows/pages.yml` builds and deploys to GitHub Pages.

## How it's wired together

This directory is a self-contained Vite project (`package.json`,
`vite.config.js`, `index.html`, `src/`). It does **not** contain the two
biggest pieces the app needs at runtime — neither is a JS/npm artifact:

- **`pkg/`** — the wasm-bindgen output of `wasm-pack build --target web
  --no-default-features --features wasm`, run against `../src/wasm.rs` at
  the repo root. This is `nixvm.js` + `nixvm_bg.wasm`.
- **`rootfs-aarch64.tar.gz` / `rootfs-x86_64.tar.gz`** — upstream Alpine
  minirootfs tarballs (one per guest architecture the page offers),
  downloaded as-is (nothing repacked).

At runtime the app fetches/imports both **same-origin, next to the built
`index.html`**: `./pkg/nixvm.js` (dynamic `import()`) and the
`./rootfs-<arch>.tar.gz` matching the arch picked in the toolbar (`fetch`;
nixvm gunzips it in-process via the `compcol` crate).
Neither path is known to Vite at build time (they don't exist in this
source tree), so `vite.config.js` sets `base: './'` and the app resolves
both via `import.meta.env.BASE_URL` at runtime rather than as bundled
imports/assets.

In CI (`.github/workflows/pages.yml`), both are copied into `web/dist/`
*after* `vite build` runs, so they land alongside `index.html` and the
built JS/CSS bundle in the final Pages artifact.

## Local dev

```sh
npm install
npm run dev
```

This starts the Vite dev server, but **the app will fail to boot** unless
`pkg/` and the `rootfs-<arch>.tar.gz` images are also reachable at the site root — Vite's
`public/` directory is served as-is at `/`, so the easiest way to get a
fully working `npm run dev` is to build those two artifacts once yourself
and drop them into `public/` (already gitignored — see `.gitignore`):

```sh
# from the repo root
rustup target add wasm32-unknown-unknown
cargo install wasm-pack
wasm-pack build --target web --no-default-features --features wasm
cp pkg/nixvm.js pkg/nixvm_bg.wasm web/public/pkg/  # mkdir -p web/public/pkg first

# any Alpine minirootfs works for local testing (fetch one per arch you
# want the picker to offer); e.g.:
curl -sSL -o web/public/rootfs-aarch64.tar.gz \
  https://dl-cdn.alpinelinux.org/alpine/v3.20/releases/aarch64/alpine-minirootfs-3.20.3-aarch64.tar.gz
curl -sSL -o web/public/rootfs-x86_64.tar.gz \
  https://dl-cdn.alpinelinux.org/alpine/v3.20/releases/x86_64/alpine-minirootfs-3.20.3-x86_64.tar.gz
```

With those in place, `npm run dev` serves a fully working demo at
`http://localhost:5173/`.

## Production build

```sh
npm run build
```

Produces `web/dist/` (plain static files: `index.html`, hashed JS/CSS
assets). This is the command CI runs; CI additionally copies `pkg/` and
the rootfs images into `web/dist/` afterwards (see
`.github/workflows/pages.yml`) — running `npm run build` alone does **not**
produce a bootable site unless `web/public/pkg` and the
`web/public/rootfs-<arch>.tar.gz` images were already populated as above, in which case
Vite copies them into `dist/` automatically as static assets.

## Browser requirements

- `WebAssembly.instantiateStreaming` + ES modules (wasm-pack's `--target
  web` output).
- Top-level `await` (Vite `build.target: 'esnext'`).
All are supported by current Chrome, Edge, Firefox, and Safari; no
extra polyfills are included.

## UX notes

`busybox sh` is run under nixvm in interactive mode but is **not** attached
to a real TTY — no line editor, no local echo. `src/components/NixTerm.vue`
does that job itself: it echoes typed characters, buffers the current line,
and only calls into the guest (`write_stdin` + `pump`) on Enter, Ctrl-C, or
Ctrl-D-on-an-empty-line. `pump()` runs the guest synchronously to
completion of that input, so a long-running command briefly freezes the
tab — expected, and called out in the page's footer note.
