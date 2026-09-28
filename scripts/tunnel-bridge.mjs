#!/usr/bin/env node
// Relay grouterd's WebSocket IP tunnel to a local TCP port, so native tests
// can exercise the exact network path the browser demo uses (see
// `tunnel_live_apk_update` in tests/alpine_boot.rs).
//
//   node scripts/tunnel-bridge.mjs 7777 &
//   NIXVM_TUNNEL_BRIDGE=127.0.0.1:7777 NIXVM_ALPINE_TAR=… \
//     cargo test --features fstool,tunnel --test alpine_boot tunnel_live -- --nocapture
//
// Fetches a tunnel token (they are rationed per day), opens
// wss://grouterd.atonline.com/tunnel/<token>, and accepts one TCP client.
// Framing on the TCP side, both ways: [kind u8][len u32 BE][bytes], where
// kind 0 is the tunnel's JSON hello (to the client only) and kind 1 is one IP
// packet. Needs Node 22+ (global WebSocket).

import net from "node:net";
import tls from "node:tls";
import { randomBytes } from "node:crypto";

const port = Number(process.argv[2] ?? 7777);
const res = await fetch("https://ws.atonline.com/_special/rest/Network:jwt", { cache: "no-store" });
const body = await res.json();
const token = body?.data?.token;
if (!token) {
  console.error("tunnel-bridge: no token:", JSON.stringify(body));
  process.exit(1);
}

function frame(kind, bytes) {
  const out = Buffer.alloc(5 + bytes.length);
  out[0] = kind;
  out.writeUInt32BE(bytes.length, 1);
  Buffer.from(bytes).copy(out, 5);
  return out;
}

// NIXVM_BRIDGE_RAW=nodelay|nagle: use a minimal WebSocket client over our
// own TLS socket instead of Node's WebSocket, with TCP_NODELAY on or off, to
// measure how the carrier's small writes (one per tunnel packet) behave.
const raw = process.env.NIXVM_BRIDGE_RAW;

function rawWebSocket(url, noDelay) {
  const u = new URL(url);
  const key = randomBytes(16).toString("base64");
  const ws = { readyState: 0, onopen: null, onmessage: null, onclose: null };
  const s = tls.connect({ host: u.hostname, port: 443, servername: u.hostname, ALPNProtocols: ["http/1.1"] });
  s.setNoDelay(noDelay);
  s.on("secureConnect", () => {
    s.write(`GET ${u.pathname} HTTP/1.1\r\nHost: ${u.hostname}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: ${key}\r\nSec-WebSocket-Version: 13\r\n\r\n`);
  });
  let buf = Buffer.alloc(0);
  let upgraded = false;
  s.on("data", (d) => {
    buf = Buffer.concat([buf, d]);
    if (!upgraded) {
      const end = buf.indexOf("\r\n\r\n");
      if (end < 0) return;
      upgraded = true;
      buf = buf.subarray(end + 4);
      ws.readyState = 1;
      ws.onopen?.();
    }
    for (;;) {
      if (buf.length < 2) return;
      const op = buf[0] & 0x0f;
      let len = buf[1] & 0x7f;
      let off = 2;
      if (len === 126) { if (buf.length < 4) return; len = buf.readUInt16BE(2); off = 4; }
      else if (len === 127) { if (buf.length < 10) return; len = Number(buf.readBigUInt64BE(2)); off = 10; }
      if (buf.length < off + len) return;
      const payload = buf.subarray(off, off + len);
      buf = buf.subarray(off + len);
      if (op === 1) ws.onmessage?.({ data: payload.toString() });
      else if (op === 2) ws.onmessage?.({ data: payload.buffer.slice(payload.byteOffset, payload.byteOffset + payload.length) });
      else if (op === 8) { ws.onclose?.({ code: payload.length >= 2 ? payload.readUInt16BE(0) : 1005 }); s.end(); }
      else if (op === 9) s.write(frame2(10, payload));
    }
  });
  s.on("close", () => ws.onclose?.({ code: 1006 }));
  function frame2(op, data) {
    const mask = randomBytes(4);
    const n = data.length;
    const hdr = n < 126 ? Buffer.from([0x80 | op, 0x80 | n]) : n < 65536 ? Buffer.from([0x80 | op, 0x80 | 126, n >> 8, n & 255]) : null;
    const body = Buffer.from(data);
    for (let i = 0; i < n; i++) body[i] ^= mask[i & 3];
    return Buffer.concat([hdr, mask, body]);
  }
  ws.send = (data) => s.write(frame2(typeof data === "string" ? 1 : 2, typeof data === "string" ? Buffer.from(data) : Buffer.from(data)));
  ws.close = () => s.end();
  return ws;
}

const server = net.createServer((sock) => {
  server.close(); // one client
  sock.setNoDelay(true);
  const url = `wss://grouterd.atonline.com/tunnel/${encodeURIComponent(token)}`;
  const ws = raw ? rawWebSocket(url, raw === "nodelay") : new WebSocket(url);
  if (!raw) ws.binaryType = "arraybuffer";
  const pending = [];
  ws.onopen = () => {
    for (const p of pending.splice(0)) ws.send(p);
  };
  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") sock.write(frame(0, Buffer.from(ev.data)));
    else sock.write(frame(1, new Uint8Array(ev.data)));
  };
  ws.onclose = (ev) => {
    console.error(`tunnel-bridge: websocket closed (${ev.code})`);
    sock.end();
    process.exit(0);
  };
  let buf = Buffer.alloc(0);
  sock.on("data", (chunk) => {
    buf = Buffer.concat([buf, chunk]);
    while (buf.length >= 5) {
      const len = buf.readUInt32BE(1);
      if (buf.length < 5 + len) break;
      const pkt = buf.subarray(5, 5 + len);
      buf = buf.subarray(5 + len);
      if (ws.readyState === 1) ws.send(pkt);
      else pending.push(pkt);
    }
  });
  sock.on("close", () => {
    ws.close();
    process.exit(0);
  });
});
server.listen(port, "127.0.0.1", () => console.error(`tunnel-bridge: listening on 127.0.0.1:${port}`));
