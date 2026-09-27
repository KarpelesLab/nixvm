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

const server = net.createServer((sock) => {
  server.close(); // one client
  const ws = new WebSocket(`wss://grouterd.atonline.com/tunnel/${encodeURIComponent(token)}`);
  ws.binaryType = "arraybuffer";
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
      if (ws.readyState === WebSocket.OPEN) ws.send(pkt);
      else pending.push(pkt);
    }
  });
  sock.on("close", () => {
    ws.close();
    process.exit(0);
  });
});
server.listen(port, "127.0.0.1", () => console.error(`tunnel-bridge: listening on 127.0.0.1:${port}`));
