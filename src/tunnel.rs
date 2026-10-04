//! Guest networking over an L3 packet tunnel.
//!
//! The browser has no sockets, but it can open a WebSocket, and grouterd's
//! `/tunnel/<token>` endpoint carries bare IP packets over one (a binary
//! message per packet, after a JSON "hello" naming the addresses leased to
//! us). This module is the guest-facing half of that: a [`Tunnel`] that is an
//! [`Egress`] backend for the socket layer, terminating guest TCP/UDP in
//! pktkit's userspace TCP/IP stack (`vclient`) and exchanging raw IP packets
//! with whoever drives it.
//!
//! It is deliberately transport-agnostic and sans-I/O, so it runs (and is
//! tested) natively as well as in wasm: the embedder
//!
//! * brings the link up with the leased addresses ([`Tunnel::up`]) and down
//!   again when the transport drops ([`Tunnel::down`]);
//! * hands every packet received from the transport to [`Tunnel::inject`];
//! * sends every packet [`Tunnel::take_outbound`] returns;
//! * calls [`Tunnel::tick`] about every 100 ms (TCP retransmits, delayed
//!   ACKs, keepalives) — there is no background thread on wasm to do it.
//!
//! Nothing here blocks. A guest `connect` is started with the handshake still
//! in flight and completes when a later pump sees it established (see the
//! `poll_connect` hook on [`HostConn`]); reads and writes return `WouldBlock`
//! until data or window arrives, and the kernel re-traps the syscall.
//!
//! One pktkit client runs per address family — a `vclient::Client` owns a
//! single prefix — and inbound packets are routed to the right one by IP
//! version.

use std::collections::{HashMap, VecDeque};
use std::io;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::{Arc, Mutex, Weak};

use pktkit::vclient::{Client, ClientConfig, TcpConn, UdpConn};
use pktkit::{IpPrefix, L3Device, Packet, Protocol, transport_checksum};

use crate::kernel::egress::{Datagram, Egress, HostConn, HostDgram, Link};

/// Most packets buffered for the transport before new ones are dropped: a
/// transport that stopped draining (a dead WebSocket) must not grow this
/// without bound. TCP retransmits whatever was lost.
const MAX_OUTBOUND: usize = 4096;

/// Addresses leased for the link, as the tunnel's hello announces them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lease {
    /// Our IPv4 address and its prefix length, if IPv4 is offered.
    pub v4: Option<(Ipv4Addr, u8)>,
    /// Our IPv6 address and its prefix length, if IPv6 is offered.
    pub v6: Option<(Ipv6Addr, u8)>,
    /// The link MTU: TCP MSS is clamped so no packet exceeds it.
    pub mtu: u16,
}

/// The packets waiting for the transport, shared with the pktkit clients'
/// output handlers.
type Outbound = Arc<Mutex<VecDeque<Vec<u8>>>>;

/// A guest network link carried over an IP-packet tunnel. Cheap to clone;
/// every clone is the same link. Install one on a kernel as its egress
/// backend with [`Tunnel::egress`].
#[derive(Clone, Default)]
pub struct Tunnel {
    inner: Arc<Mutex<Inner>>,
    outbound: Outbound,
}

#[derive(Default)]
struct Inner {
    v4: Option<Arc<Client>>,
    v6: Option<Arc<Client>>,
    lease: Option<Lease>,
    /// Open ICMP endpoints (guest raw/ping sockets), by family: each gets a
    /// copy of every inbound ICMP packet. Dropped endpoints are pruned lazily.
    icmp: Vec<(bool, Weak<IcmpQueue>)>,
    /// IPv4 identification for the ICMP packets we build.
    ip_id: u16,
}

/// Inbound ICMP packets waiting for one guest socket.
type IcmpQueue = Mutex<VecDeque<Vec<u8>>>;

/// Most inbound ICMP packets queued per endpoint before new ones are dropped.
const MAX_ICMP_QUEUE: usize = 256;

impl std::fmt::Debug for Tunnel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tunnel")
            .field("lease", &self.inner.lock().unwrap().lease)
            .finish_non_exhaustive()
    }
}

impl Tunnel {
    /// A tunnel whose link is down: guest connects to routable addresses fail
    /// with `ENETUNREACH` until [`Tunnel::up`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The egress backend to install on a kernel (see
    /// [`crate::kernel::Kernel::set_egress`]). Shares this tunnel's link.
    #[must_use]
    pub fn egress(&self) -> Box<dyn Egress> {
        Box::new(self.clone())
    }

    /// Bring the link up with the addresses in `lease`. Replaces any previous
    /// link: connections opened on it are gone (the remote end no longer
    /// routes to them), and the next guest connect uses the new addresses.
    pub fn up(&self, lease: Lease) {
        let mtu = lease.mtu.max(576);
        let mut inner = self.inner.lock().unwrap();
        inner.v4 = lease
            .v4
            .map(|(ip, bits)| self.client(IpPrefix::new(IpAddr::V4(ip), bits), mtu));
        inner.v6 = lease
            .v6
            .map(|(ip, bits)| self.client(IpPrefix::new(IpAddr::V6(ip), bits), mtu));
        inner.lease = Some(Lease { mtu, ..lease });
    }

    /// Take the link down (the transport closed). Guest connects fail with
    /// `ENETUNREACH` again; existing connections stall and time out.
    pub fn down(&self) {
        let mut inner = self.inner.lock().unwrap();
        for c in inner.v4.take().into_iter().chain(inner.v6.take()) {
            let _ = c.close();
        }
        inner.lease = None;
        self.outbound.lock().unwrap().clear();
    }

    /// The current lease, while the link is up.
    #[must_use]
    pub fn lease(&self) -> Option<Lease> {
        self.inner.lock().unwrap().lease
    }

    /// A packet received from the transport. Malformed packets and packets
    /// for a family the link doesn't have are dropped.
    pub fn inject(&self, packet: &[u8]) {
        let (client, mtu) = {
            let inner = self.inner.lock().unwrap();
            let Some(lease) = inner.lease else { return };
            let client = match packet.first().map(|b| b >> 4) {
                Some(4) => inner.v4.clone(),
                Some(6) => inner.v6.clone(),
                _ => None,
            };
            (client, lease.mtu)
        };
        if is_icmp(packet) {
            self.deliver_icmp(packet);
            return;
        }
        let Some(client) = client else { return };
        // Clamp the MSS a peer's SYN-ACK advertises, so what we send it fits
        // the link too (our own SYN is clamped on the way out).
        let mut packet = packet.to_vec();
        clamp_mss(&mut packet, mtu);
        let _ = client.send(Packet::from_slice(&packet));
    }

    /// Hand inbound ICMP `packet` to every open ICMP endpoint of its family.
    fn deliver_icmp(&self, packet: &[u8]) {
        let v6 = packet[0] >> 4 == 6;
        let mut inner = self.inner.lock().unwrap();
        inner.icmp.retain(|(_, w)| w.strong_count() > 0);
        for (fam, w) in &inner.icmp {
            if *fam == v6
                && let Some(q) = w.upgrade()
            {
                let mut q = q.lock().unwrap();
                if q.len() < MAX_ICMP_QUEUE {
                    q.push_back(packet.to_vec());
                }
            }
        }
    }

    /// Wrap ICMP message `msg` in an IP packet from our address to `dst` and
    /// queue it for the transport (computing the ICMPv6 checksum).
    fn send_icmp(&self, msg: &[u8], dst: IpAddr) -> io::Result<usize> {
        let unreachable = || io::Error::new(io::ErrorKind::NetworkUnreachable, "no route");
        let mut inner = self.inner.lock().unwrap();
        let lease = inner.lease.ok_or_else(unreachable)?;
        let pkt = match dst {
            IpAddr::V4(d) => {
                let (src, _) = lease.v4.ok_or_else(unreachable)?;
                inner.ip_id = inner.ip_id.wrapping_add(1);
                let total = 20 + msg.len();
                let mut p = Vec::with_capacity(total);
                p.extend_from_slice(&[0x45, 0]);
                p.extend_from_slice(&u16::try_from(total).map_err(|_| too_big())?.to_be_bytes());
                p.extend_from_slice(&inner.ip_id.to_be_bytes());
                p.extend_from_slice(&[0, 0, 64, 1, 0, 0]); // no frag, TTL 64, ICMP
                p.extend_from_slice(&src.octets());
                p.extend_from_slice(&d.octets());
                let sum = pktkit::checksum(&p[..20]);
                p[10..12].copy_from_slice(&sum.to_be_bytes());
                p.extend_from_slice(msg);
                p
            }
            IpAddr::V6(d) => {
                let (src, _) = lease.v6.ok_or_else(unreachable)?;
                if msg.len() < 4 {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "short ICMPv6"));
                }
                let mut body = msg.to_vec();
                body[2..4].fill(0);
                let sum = transport_checksum(Protocol::ICMPV6, src.into(), d.into(), &body);
                body[2..4].copy_from_slice(&sum.to_be_bytes());
                let mut p = Vec::with_capacity(40 + body.len());
                p.extend_from_slice(&[0x60, 0, 0, 0]);
                p.extend_from_slice(
                    &u16::try_from(body.len())
                        .map_err(|_| too_big())?
                        .to_be_bytes(),
                );
                p.extend_from_slice(&[58, 64]); // next header ICMPv6, hop limit 64
                p.extend_from_slice(&src.octets());
                p.extend_from_slice(&d.octets());
                p.extend_from_slice(&body);
                p
            }
        };
        if pkt.len() > usize::from(lease.mtu) {
            return Err(too_big());
        }
        drop(inner);
        let mut q = self.outbound.lock().unwrap();
        if q.len() < MAX_OUTBOUND {
            q.push_back(pkt);
        }
        Ok(msg.len())
    }

    /// Packets to send over the transport, oldest first.
    #[must_use]
    pub fn take_outbound(&self) -> Vec<Vec<u8>> {
        self.outbound.lock().unwrap().drain(..).collect()
    }

    /// Whether any packet is waiting for the transport.
    #[must_use]
    pub fn has_outbound(&self) -> bool {
        !self.outbound.lock().unwrap().is_empty()
    }

    /// Run the TCP timers (retransmission, persist, keepalive, TIME-WAIT,
    /// delayed ACKs). Call about every 100 ms while the link is up.
    pub fn tick(&self) {
        let clients: Vec<Arc<Client>> = {
            let inner = self.inner.lock().unwrap();
            inner.v4.iter().chain(inner.v6.iter()).cloned().collect()
        };
        for c in clients {
            c.tick();
        }
    }

    /// A pktkit client for `prefix` whose output lands in our outbound queue,
    /// MSS-clamped to `mtu`.
    fn client(&self, prefix: IpPrefix, mtu: u16) -> Arc<Client> {
        let client = Client::new(ClientConfig::default().prefix(prefix));
        let out = Arc::clone(&self.outbound);
        client.set_handler(Arc::new(move |pkt: &Packet| {
            let mut bytes = pkt.as_bytes().to_vec();
            clamp_mss(&mut bytes, mtu);
            let mut q = out.lock().unwrap();
            if q.len() < MAX_OUTBOUND {
                q.push_back(bytes);
            }
            Ok(())
        }));
        client
    }

    /// The client for destinations of family `v6`, if the link has one.
    fn client_for(&self, v6: bool) -> io::Result<Arc<Client>> {
        let inner = self.inner.lock().unwrap();
        let client = if v6 { &inner.v6 } else { &inner.v4 };
        client.clone().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NetworkUnreachable,
                if inner.lease.is_some() {
                    "no address of that family on the tunnel"
                } else {
                    "network link is down"
                },
            )
        })
    }
}

/// Parse a guest 16-byte IP + family + port into the socket address to use
/// on the wire: an IPv4-mapped IPv6 address (an `AF_INET6` socket reaching an
/// IPv4 host) goes out as plain IPv4.
fn sockaddr(ip: [u8; 16], v6: bool, port: u16) -> SocketAddr {
    let ip = if v6 {
        let a = Ipv6Addr::from(ip);
        a.to_ipv4_mapped().map_or(IpAddr::V6(a), IpAddr::V4)
    } else {
        IpAddr::V4(Ipv4Addr::new(ip[0], ip[1], ip[2], ip[3]))
    };
    SocketAddr::new(ip, port)
}

impl Egress for Tunnel {
    fn connect_tcp(&self, ip: [u8; 16], v6: bool, port: u16) -> io::Result<Box<dyn HostConn>> {
        let dest = sockaddr(ip, v6, port);
        let conn = self
            .client_for(dest.is_ipv6())?
            .dial_tcp_nonblocking(dest)?;
        conn.set_nonblocking(true);
        Ok(Box::new(TunnelConn {
            conn,
            pending: Vec::new(),
            eof: false,
        }))
    }

    fn open_udp(&self) -> io::Result<Box<dyn HostDgram>> {
        Ok(Box::new(TunnelDgram {
            tunnel: self.clone(),
            socks: HashMap::new(),
        }))
    }

    fn open_icmp(&self, v6: bool) -> io::Result<Box<dyn HostDgram>> {
        let queue = Arc::new(IcmpQueue::default());
        self.inner
            .lock()
            .unwrap()
            .icmp
            .push((v6, Arc::downgrade(&queue)));
        Ok(Box::new(TunnelIcmp {
            tunnel: self.clone(),
            queue,
        }))
    }

    fn link(&self) -> Option<Link> {
        let lease = self.lease()?;
        Some(Link {
            mtu: u32::from(lease.mtu),
            v4: lease.v4.map(|(ip, len)| (ip.octets(), len)),
            v6: lease.v6.map(|(ip, len)| (ip.octets(), len)),
        })
    }
}

fn too_big() -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, "packet exceeds the link MTU")
}

/// Whether IP `packet` carries ICMP (IPv4 protocol 1, IPv6 next header 58).
fn is_icmp(packet: &[u8]) -> bool {
    match packet.first().map(|b| b >> 4) {
        Some(4) => packet.len() >= 20 && packet[9] == 1,
        Some(6) => packet.len() >= 40 && packet[6] == 58,
        _ => false,
    }
}

/// A guest ICMP socket's endpoint on the tunnel (see [`Egress::open_icmp`]).
struct TunnelIcmp {
    tunnel: Tunnel,
    queue: Arc<IcmpQueue>,
}

impl std::fmt::Debug for TunnelIcmp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TunnelIcmp").finish_non_exhaustive()
    }
}

impl HostDgram for TunnelIcmp {
    fn send_to(&mut self, buf: &[u8], ip: [u8; 16], v6: bool, _port: u16) -> io::Result<usize> {
        self.tunnel.send_icmp(buf, sockaddr(ip, v6, 0).ip())
    }

    fn recv_from(&mut self) -> io::Result<Option<Datagram>> {
        let Some(pkt) = self.queue.lock().unwrap().pop_front() else {
            return Ok(None);
        };
        let (ip, v6) = if pkt[0] >> 4 == 6 {
            (<[u8; 16]>::try_from(&pkt[8..24]).unwrap(), true)
        } else {
            let mut ip = [0u8; 16];
            ip[..4].copy_from_slice(&pkt[12..16]);
            (ip, false)
        };
        Ok(Some((ip, v6, 0, pkt)))
    }
}

/// A guest TCP connection terminated in the tunnel's TCP stack.
struct TunnelConn {
    conn: TcpConn,
    /// Bytes already pulled out of the stack by a readiness check, delivered
    /// ahead of anything else (the stack has no peek).
    pending: Vec<u8>,
    /// A readiness check saw end-of-stream.
    eof: bool,
}

impl std::fmt::Debug for TunnelConn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TunnelConn")
            .field("peer", &self.conn.peer_addr())
            .finish_non_exhaustive()
    }
}

impl TunnelConn {
    /// Pull whatever the stack has buffered into `pending`, noting EOF.
    fn fill(&mut self) {
        if !self.pending.is_empty() || self.eof {
            return;
        }
        let mut buf = vec![0u8; 64 * 1024];
        match self.conn.read(&mut buf) {
            Ok(0) => self.eof = true,
            Ok(n) => {
                buf.truncate(n);
                self.pending = buf;
            }
            Err(_) => {}
        }
    }
}

impl HostConn for TunnelConn {
    fn recv(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if !self.pending.is_empty() {
            let n = buf.len().min(self.pending.len());
            buf[..n].copy_from_slice(&self.pending[..n]);
            self.pending.drain(..n);
            return Ok(n);
        }
        if self.eof {
            return Ok(0);
        }
        self.conn.read(buf)
    }

    fn send(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.conn.write(buf)
    }

    fn shutdown_write(&mut self) {
        // A FIN; the receive side stays open until the peer closes too.
        let _ = self.conn.close();
    }

    fn poll_readable(&mut self) -> bool {
        self.fill();
        !self.pending.is_empty() || self.eof
    }

    fn readable_len(&mut self) -> usize {
        self.fill();
        self.pending.len()
    }

    fn poll_connect(&mut self) -> io::Result<bool> {
        self.conn.poll_connect()
    }
}

/// A guest UDP socket over the tunnel. pktkit's UDP handles are connected
/// (one remote each), so this keeps one per destination the guest has sent
/// to and reads replies from all of them, keyed by the destination as the
/// guest named it (so a reply's source is reported in the same form).
struct TunnelDgram {
    tunnel: Tunnel,
    socks: HashMap<([u8; 16], bool, u16), UdpConn>,
}

impl std::fmt::Debug for TunnelDgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TunnelDgram")
            .field("peers", &self.socks.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

/// Most distinct destinations one guest UDP socket keeps a handle for; the
/// oldest is recycled past this (a resolver only ever talks to a few).
const MAX_UDP_PEERS: usize = 64;

impl HostDgram for TunnelDgram {
    fn send_to(&mut self, buf: &[u8], ip: [u8; 16], v6: bool, port: u16) -> io::Result<usize> {
        let key = (ip, v6, port);
        if !self.socks.contains_key(&key) {
            let dest = sockaddr(ip, v6, port);
            let conn = self.tunnel.client_for(dest.is_ipv6())?.dial_udp(dest)?;
            conn.set_nonblocking(true);
            if self.socks.len() >= MAX_UDP_PEERS
                && let Some(old) = self.socks.keys().next().copied()
            {
                self.socks.remove(&old);
            }
            self.socks.insert(key, conn);
        }
        self.socks[&key].send(buf)
    }

    fn recv_from(&mut self) -> io::Result<Option<Datagram>> {
        let mut buf = vec![0u8; 65_536];
        for (&(ip, v6, port), conn) in &self.socks {
            if let Ok(n) = conn.recv(&mut buf) {
                buf.truncate(n);
                return Ok(Some((ip, v6, port, buf)));
            }
        }
        Ok(None)
    }
}

/// Clamp the MSS option of a TCP SYN in IP packet `pkt` so segments fit an
/// `mtu`-sized link, fixing the TCP checksum. Anything else is left alone.
/// This is what a router on a small-MTU path does: the peer sends segments no
/// larger than our (clamped) MSS, and we send it none larger than its own.
fn clamp_mss(pkt: &mut [u8], mtu: u16) {
    let (ip_hdr, overhead, src, dst) = match pkt.first().map(|b| b >> 4) {
        Some(4) if pkt.len() >= 20 && pkt[9] == 6 => {
            let ihl = usize::from(pkt[0] & 0xf) * 4;
            let src = IpAddr::V4(Ipv4Addr::new(pkt[12], pkt[13], pkt[14], pkt[15]));
            let dst = IpAddr::V4(Ipv4Addr::new(pkt[16], pkt[17], pkt[18], pkt[19]));
            (ihl, 40, src, dst)
        }
        Some(6) if pkt.len() >= 40 && pkt[6] == 6 => {
            let src: [u8; 16] = pkt[8..24].try_into().unwrap();
            let dst: [u8; 16] = pkt[24..40].try_into().unwrap();
            (40, 60, IpAddr::V6(src.into()), IpAddr::V6(dst.into()))
        }
        _ => return,
    };
    let Some(tcp) = pkt.get_mut(ip_hdr..) else {
        return;
    };
    if tcp.len() < 20 || tcp[13] & 0x02 == 0 {
        return; // not a SYN
    }
    let max = mtu.saturating_sub(overhead);
    let data_off = (usize::from(tcp[12] >> 4) * 4).min(tcp.len());
    let mut i = 20;
    let mut changed = false;
    while i < data_off {
        match tcp[i] {
            0 => break, // end of options
            1 => i += 1,
            kind => {
                let Some(&len) = tcp.get(i + 1) else { break };
                let len = usize::from(len);
                if len < 2 || i + len > data_off {
                    break;
                }
                if kind == 2 && len == 4 {
                    let mss = u16::from_be_bytes([tcp[i + 2], tcp[i + 3]]);
                    if mss > max {
                        tcp[i + 2..i + 4].copy_from_slice(&max.to_be_bytes());
                        changed = true;
                    }
                }
                i += len;
            }
        }
    }
    if changed {
        tcp[16..18].fill(0);
        let sum = transport_checksum(Protocol::TCP, src, dst, tcp);
        tcp[16..18].copy_from_slice(&sum.to_be_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two tunnels wired back to back: what one sends, the other receives.
    fn shuttle(a: &Tunnel, b: &Tunnel) -> bool {
        let mut moved = false;
        for p in a.take_outbound() {
            b.inject(&p);
            moved = true;
        }
        for p in b.take_outbound() {
            a.inject(&p);
            moved = true;
        }
        moved
    }

    fn lease4(last: u8) -> Lease {
        Lease {
            v4: Some((Ipv4Addr::new(10, 0, 0, last), 24)),
            v6: Some((Ipv6Addr::new(0xfd00, 0, 0, 0, 0, 0, 0, u16::from(last)), 64)),
            mtu: 1400,
        }
    }

    #[test]
    fn link_down_is_unreachable() {
        let t = Tunnel::new();
        let e = t.connect_tcp([10, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], false, 80);
        assert_eq!(e.unwrap_err().kind(), io::ErrorKind::NetworkUnreachable);
    }

    #[test]
    fn tcp_connects_and_carries_data_both_ways() {
        let client = Tunnel::new();
        client.up(lease4(2));
        // The "internet": a second pktkit client, listening.
        let server = Tunnel::new();
        server.up(lease4(1));
        let srv = server.inner.lock().unwrap().v4.clone().unwrap();
        let listener = srv.listen_tcp(80).unwrap();
        listener.set_nonblocking(true);

        let mut ip = [0u8; 16];
        ip[..4].copy_from_slice(&[10, 0, 0, 1]);
        let mut conn = client.connect_tcp(ip, false, 80).unwrap();
        assert!(!conn.poll_connect().unwrap(), "handshake still in flight");

        // SYN must carry a clamped MSS (1400 - 40).
        let syn = client.take_outbound();
        assert_eq!(syn.len(), 1);
        let opts = &syn[0][40..];
        assert_eq!(&opts[..2], &[2, 4]);
        assert_eq!(u16::from_be_bytes([opts[2], opts[3]]), 1360);
        server.inject(&syn[0]);

        while shuttle(&client, &server) {}
        assert!(conn.poll_connect().unwrap(), "connected");
        let accepted = listener.accept().unwrap();
        accepted.set_nonblocking(true);

        assert_eq!(conn.send(b"GET / HTTP/1.0\r\n\r\n").unwrap(), 18);
        while shuttle(&client, &server) {}
        let mut buf = [0u8; 64];
        let n = accepted.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"GET / HTTP/1.0\r\n\r\n");

        // Bulk data larger than an MSS back to the guest, then close.
        let body: Vec<u8> = (0..20_000u32).map(|i| i as u8).collect();
        let mut sent = 0;
        let mut got = Vec::new();
        let mut rounds = 0;
        while got.len() < body.len() {
            if sent < body.len()
                && let Ok(n) = accepted.write(&body[sent..])
            {
                sent += n;
            }
            while shuttle(&client, &server) {}
            for p in client.take_outbound() {
                server.inject(&p);
            }
            let mut chunk = [0u8; 4096];
            while conn.poll_readable() {
                let n = conn.recv(&mut chunk).unwrap();
                if n == 0 {
                    break;
                }
                got.extend_from_slice(&chunk[..n]);
            }
            client.tick();
            server.tick();
            rounds += 1;
            assert!(rounds < 10_000, "transfer stalled at {} bytes", got.len());
        }
        assert_eq!(got, body);
        accepted.close().unwrap();
        while shuttle(&client, &server) {}
        assert!(conn.poll_readable(), "EOF is readable");
        assert_eq!(conn.recv(&mut buf).unwrap(), 0);
    }

    #[test]
    fn inbound_packets_never_exceed_the_mtu() {
        let client = Tunnel::new();
        client.up(lease4(2));
        let server = Tunnel::new();
        server.up(lease4(1));
        let srv = server.inner.lock().unwrap().v4.clone().unwrap();
        let listener = srv.listen_tcp(80).unwrap();
        listener.set_nonblocking(true);
        let mut ip = [0u8; 16];
        ip[..4].copy_from_slice(&[10, 0, 0, 1]);
        let conn = client.connect_tcp(ip, false, 80).unwrap();
        while shuttle(&client, &server) {}
        let accepted = listener.accept().unwrap();
        accepted.set_nonblocking(true);
        let _ = accepted.write(&[7u8; 8000]);
        for p in server.take_outbound() {
            assert!(p.len() <= 1400, "server sent a {}-byte packet", p.len());
        }
        drop(conn);
    }

    #[test]
    fn icmp_echo_goes_out_and_replies_come_back() {
        let t = Tunnel::new();
        t.up(lease4(2));
        let mut sock = t.open_icmp(false).unwrap();
        let mut dst = [0u8; 16];
        dst[..4].copy_from_slice(&[8, 8, 8, 8]);
        // Echo request, id 0x1234 seq 1, checksum left to the guest (zero here).
        let echo = [8u8, 0, 0, 0, 0x12, 0x34, 0, 1, b'h', b'i'];
        assert_eq!(sock.send_to(&echo, dst, false, 0).unwrap(), echo.len());
        let out = t.take_outbound();
        assert_eq!(out.len(), 1);
        let p = &out[0];
        assert_eq!(
            (p[0], p[9], &p[12..16], &p[16..20]),
            (0x45, 1, &[10, 0, 0, 2][..], &[8, 8, 8, 8][..])
        );
        assert_eq!(pktkit::checksum(&p[..20]), 0, "valid IPv4 header checksum");
        assert_eq!(&p[20..], &echo);
        // A reply (addresses swapped, type 0) reaches the endpoint whole.
        let mut reply = p.clone();
        reply[12..16].copy_from_slice(&[8, 8, 8, 8]);
        reply[16..20].copy_from_slice(&[10, 0, 0, 2]);
        reply[20] = 0;
        t.inject(&reply);
        let (from, v6, _, got) = sock.recv_from().unwrap().expect("reply");
        assert_eq!((&from[..4], v6), (&[8, 8, 8, 8][..], false));
        assert_eq!(got, reply);
        // Link down: unreachable.
        t.down();
        assert_eq!(
            sock.send_to(&echo, dst, false, 0).unwrap_err().kind(),
            io::ErrorKind::NetworkUnreachable
        );
    }

    #[test]
    fn udp_round_trip_reports_the_source() {
        let client = Tunnel::new();
        client.up(lease4(2));
        let server = Tunnel::new();
        server.up(lease4(1));
        let mut sock = client.open_udp().unwrap();
        let mut ip = [0u8; 16];
        ip[..4].copy_from_slice(&[10, 0, 0, 1]);
        assert_eq!(sock.send_to(b"query", ip, false, 53).unwrap(), 5);
        let out = client.take_outbound();
        assert_eq!(out.len(), 1);
        // Answer it by swapping addresses and ports in the query packet.
        let q = &out[0];
        let mut reply = q[..28].to_vec();
        reply.extend_from_slice(b"answer!");
        let total = reply.len() as u16;
        reply[2..4].copy_from_slice(&total.to_be_bytes());
        let (s, d) = (q[12..16].to_vec(), q[16..20].to_vec());
        reply[12..16].copy_from_slice(&d);
        reply[16..20].copy_from_slice(&s);
        reply[10..12].fill(0);
        let sum = pktkit::checksum(&reply[..20]);
        reply[10..12].copy_from_slice(&sum.to_be_bytes());
        let (sp, dp) = (q[20..22].to_vec(), q[22..24].to_vec());
        reply[20..22].copy_from_slice(&dp);
        reply[22..24].copy_from_slice(&sp);
        reply[24..26].copy_from_slice(&(total - 20).to_be_bytes());
        reply[26..28].fill(0); // no UDP checksum (legal over IPv4)
        client.inject(&reply);
        let (from, v6, port, data) = sock.recv_from().unwrap().expect("datagram");
        assert_eq!((&from[..4], v6, port), (&[10, 0, 0, 1][..], false, 53));
        assert_eq!(data, b"answer!");
        drop(server);
    }
}
