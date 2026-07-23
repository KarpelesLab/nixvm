//! Host network egress — the seam that lets guest sockets reach the real
//! internet, so `apk`/`curl`/`npm` work instead of only talking to endpoints
//! inside the VM.
//!
//! The in-VM loopback transport in [`super::net`] stays the default and the
//! only thing available with no egress backend installed. When a backend *is*
//! installed (`NIXVM_NET=host`, or [`crate::sandbox`]/[`crate::vm`] policy), a
//! guest `connect()` to a **routable** address (anything that isn't this VM's
//! own loopback/wildcard) is bridged onto a real host connection through the
//! [`Egress`] trait, and UDP datagrams to a routable address (DNS) go through a
//! host datagram socket.
//!
//! The trait boundary is deliberately narrow and transport-agnostic: the
//! native implementation ([`HostEgress`], `std::net`) lives here behind
//! `cfg(not(wasm32))`, and a future WebSocket/pktkit transport for the browser
//! slots in as another `impl Egress` with no change to the socket layer.
//!
//! Non-blocking is the contract. The kernel services syscalls on one thread,
//! so a host read/write must never park it: `recv`/`send`/`recv_from` return
//! `WouldBlock` and the kernel re-traps the guest syscall later (the same
//! block-and-retry path an empty in-VM socket queue uses). Connection setup is
//! the one exception — [`Egress::connect_tcp`] may block briefly (a bounded
//! `connect_timeout`) since the guest is mid-`connect()` anyway.

use std::fmt::Debug;
use std::io;

/// A live host-side stream connection a guest TCP socket is bridged onto.
pub trait HostConn: Send + Debug {
    /// Read into `buf`. `Ok(0)` is EOF (peer closed); `WouldBlock` means "no
    /// data yet, retry" — never a host-thread park.
    fn recv(&mut self, buf: &mut [u8]) -> io::Result<usize>;
    /// Write `buf`; may short-write. `WouldBlock` means "retry".
    fn send(&mut self, buf: &[u8]) -> io::Result<usize>;
    /// Half-close the write side (guest `shutdown(SHUT_WR)`).
    fn shutdown_write(&mut self);
    /// Whether a `recv` would return data or EOF right now (a non-blocking
    /// peek). Drives `poll`/`select` readiness so a guest that trusts `poll`
    /// (apk's http client) doesn't see a spurious "ready" then `EAGAIN`.
    fn poll_readable(&mut self) -> bool;
    /// Bytes available to read right now, for `ioctl(FIONREAD)`. The default is
    /// coarse (1 if any data, else 0); a transport that can peek the real count
    /// (native sockets) should override it so a client sizing a read buffer from
    /// `FIONREAD` doesn't under-read.
    fn readable_len(&mut self) -> usize {
        usize::from(self.poll_readable())
    }
}

/// One received datagram: `(source ip, v6, source port, payload)`.
pub type Datagram = ([u8; 16], bool, u16, Vec<u8>);

/// A host-side datagram socket for UDP egress (the DNS path, chiefly).
pub trait HostDgram: Send + Debug {
    /// Send `buf` to `(ip, v6, port)`.
    fn send_to(&mut self, buf: &[u8], ip: [u8; 16], v6: bool, port: u16) -> io::Result<usize>;
    /// Receive the next datagram, or `None` if none is ready (non-blocking).
    fn recv_from(&mut self) -> io::Result<Option<Datagram>>;
}

/// Opens host-side connections for the guest. One installed instance is shared
/// by the whole VM. The native impl is [`HostEgress`]; a browser transport
/// (WebSocket relay / pktkit) will be another implementor.
pub trait Egress: Send + Debug {
    /// Open a stream connection to `(ip, v6, port)`. May block briefly.
    fn connect_tcp(&self, ip: [u8; 16], v6: bool, port: u16) -> io::Result<Box<dyn HostConn>>;
    /// Open a datagram socket for UDP egress.
    fn open_udp(&self) -> io::Result<Box<dyn HostDgram>>;
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::HostEgress;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::{Egress, HostConn, HostDgram};
    use std::io::{self, Read, Write};
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpStream, UdpSocket};
    use std::time::Duration;

    /// Native host egress via `std::net` — the real internet, for CLI/test
    /// runs. Off by default; opt in per [`super`]'s policy.
    #[derive(Debug, Default)]
    pub struct HostEgress;

    /// Build a `SocketAddr` from the guest's 16-byte IP + family + port.
    fn sockaddr(ip: [u8; 16], v6: bool, port: u16) -> SocketAddr {
        if v6 {
            SocketAddr::new(IpAddr::V6(Ipv6Addr::from(ip)), port)
        } else {
            let v4 = Ipv4Addr::new(ip[0], ip[1], ip[2], ip[3]);
            SocketAddr::new(IpAddr::V4(v4), port)
        }
    }

    /// Decompose a host `SocketAddr` back into the guest 16-byte-IP form.
    fn from_sockaddr(a: SocketAddr) -> ([u8; 16], bool, u16) {
        match a {
            SocketAddr::V4(v4) => {
                let mut ip = [0u8; 16];
                ip[0..4].copy_from_slice(&v4.ip().octets());
                (ip, false, v4.port())
            }
            SocketAddr::V6(v6) => (v6.ip().octets(), true, v6.port()),
        }
    }

    impl Egress for HostEgress {
        fn connect_tcp(&self, ip: [u8; 16], v6: bool, port: u16) -> io::Result<Box<dyn HostConn>> {
            // A bounded blocking connect (the guest is mid-`connect()`), then
            // switch to non-blocking for all subsequent I/O.
            let stream = TcpStream::connect_timeout(&sockaddr(ip, v6, port), Duration::from_secs(10))?;
            stream.set_nonblocking(true)?;
            let _ = stream.set_nodelay(true);
            Ok(Box::new(TcpConn(stream)))
        }

        fn open_udp(&self) -> io::Result<Box<dyn HostDgram>> {
            let sock = UdpSocket::bind(("0.0.0.0", 0))?;
            sock.set_nonblocking(true)?;
            Ok(Box::new(UdpConn(sock)))
        }
    }

    #[derive(Debug)]
    struct TcpConn(TcpStream);

    impl HostConn for TcpConn {
        fn recv(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            self.0.read(buf)
        }
        fn send(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.write(buf)
        }
        fn shutdown_write(&mut self) {
            let _ = self.0.shutdown(std::net::Shutdown::Write);
        }
        fn poll_readable(&mut self) -> bool {
            // A non-blocking peek: any bytes (or a clean EOF, reported as
            // `Ok(0)`) means a subsequent recv won't block. `WouldBlock` means
            // nothing yet. Any other error also counts as "ready" so the guest
            // observes the error via recv rather than spinning.
            let mut b = [0u8; 1];
            match self.0.peek(&mut b) {
                Ok(_) => true,
                Err(e) => e.kind() != io::ErrorKind::WouldBlock,
            }
        }
        fn readable_len(&mut self) -> usize {
            // Peek into a page-sized buffer for the queued byte count. Capped at
            // the buffer — enough to size a read; a client re-checks after
            // draining. `WouldBlock`/errors report 0 (nothing to read now).
            let mut b = [0u8; 4096];
            self.0.peek(&mut b).unwrap_or(0)
        }
    }

    #[derive(Debug)]
    struct UdpConn(UdpSocket);

    impl HostDgram for UdpConn {
        fn send_to(&mut self, buf: &[u8], ip: [u8; 16], v6: bool, port: u16) -> io::Result<usize> {
            self.0.send_to(buf, sockaddr(ip, v6, port))
        }
        fn recv_from(&mut self) -> io::Result<Option<([u8; 16], bool, u16, Vec<u8>)>> {
            let mut buf = vec![0u8; 65_536];
            match self.0.recv_from(&mut buf) {
                Ok((n, from)) => {
                    buf.truncate(n);
                    let (ip, v6, port) = from_sockaddr(from);
                    Ok(Some((ip, v6, port, buf)))
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => Ok(None),
                Err(e) => Err(e),
            }
        }
    }
}
