import { fileURLToPath, URL } from "node:url";

import vue from "@vitejs/plugin-vue";
import { defineConfig } from "vite";

// Vite config for the nixvm browser demo.
//
// - `base: './'` makes every emitted asset URL relative, so the site works
//   whether it's served at the Pages project root or (as here) under
//   `/nixvm/`, and so the app can resolve `./pkg/nixvm.js` and
//   `./rootfs.tar.gz` (both assembled next to `index.html` by
//   `.github/workflows/pages.yml`, *not* by this Vite build — see
//   `web/README.md`) purely from `import.meta.env.BASE_URL` at runtime.
// - `build.target: 'esnext'` is required for top-level `await` (used to
//   `await init()` the wasm module) and for `WebAssembly.instantiateStreaming`
//   as emitted by wasm-pack's `--target web` output.
export default defineConfig({
  base: "./",
  plugins: [vue()],
  build: {
    target: "esnext",
    outDir: "dist",
  },
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
});
