<script setup>
import NixTerm from "./components/NixTerm.vue";
</script>

<template>
  <div class="page">
    <header class="hero">
      <span class="badge">100% client-side · WebAssembly</span>
      <h1><span class="accent">nixvm</span> — Alpine Linux in your browser</h1>
      <p>
        This terminal boots a real Alpine Linux root filesystem inside a
        gVisor-style Linux syscall sandbox compiled to WebAssembly — no
        server, no container, nothing installed. Pick a guest CPU
        architecture (arm64 or x86-64 — each runs on its own software CPU
        interpreter), press Start, and everything, including the shell you
        type into, runs directly in this tab.
      </p>
      <div class="links">
        <a href="https://github.com/KarpelesLab/nixvm" target="_blank" rel="noopener">
          KarpelesLab/nixvm on GitHub
        </a>
        <a href="https://github.com/KarpelesLab/nixvm#readme" target="_blank" rel="noopener">
          How it works
        </a>
      </div>
    </header>

    <NixTerm />

    <footer class="note">
      Tip: this is <code>busybox sh</code> running non-interactively under
      nixvm — there's no real TTY, so line editing (backspace, Ctrl-C, Ctrl-D)
      is handled by this page, not the shell. Each command you run pumps the
      guest synchronously, so a long-running command will briefly freeze the
      tab; that's expected. Requires a modern browser (WebAssembly +
      top-level <code>await</code> support — recent Chrome, Edge, Firefox,
      or Safari).
    </footer>
  </div>
</template>
