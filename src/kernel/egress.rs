//! Host network egress — the seam that lets guest sockets reach the real
//! internet, so `apk`/`curl`/`npm` work instead of only talking to endpoints
//! inside the VM.
//!
//! The in-VM loopback transport in `super::net` stays the default and the
//! only thing available with no egress backend installed. When a backend *is*
//! installed (`NIXVM_NET=host`, or [`crate::sandbox`]/[`crate::vm`] policy), a
//! guest `connect()` to a **routable** address (anything that isn't this VM's
//! own loopback/wildcard) is bridged onto a real host connection through the
//! [`Egress`] trait, and UDP datagrams to a routable address (DNS) go through a
//! host datagram socket.
//!
//! The trait boundary is deliberately narrow and transport-agnostic: the
//! native implementation ([`HostEgress`], `std::net`) lives here behind
//! `cfg(not(wasm32))`, and the WebSocket/pktkit transport for the browser
//! (`crate::tunnel`) is another `impl Egress`.
//!
//! Non-blocking is the contract. The kernel services syscalls on one thread,
//! so a host read/write must never park it: `recv`/`send`/`recv_from` return
//! `WouldBlock` and the kernel re-traps the guest syscall later (the same
//! block-and-retry path an empty in-VM socket queue uses). Connection setup is
//! the one exception — [`Egress::connect_tcp`] may block briefly (a bounded
//! `connect_timeout`) since the guest is mid-`connect()` anyway — or, for a
//! transport that can't block at all (the browser tunnel, `crate::tunnel`),
//! return at once and report the handshake through [`HostConn::poll_connect`].

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
    /// Whether the connection's handshake has completed: `Ok(true)` once it is
    /// up, `Ok(false)` while still in progress, an error if it failed. A
    /// transport whose [`Egress::connect_tcp`] returns before the handshake
    /// finishes (a userspace TCP stack that must not block, like the wasm
    /// tunnel) reports progress here; the socket layer turns it into a
    /// blocking-connect re-trap or `EINPROGRESS`/`POLLOUT`/`SO_ERROR`. The
    /// default suits transports that only return established connections.
    fn poll_connect(&mut self) -> io::Result<bool> {
        Ok(true)
    }
    /// Apply a socket option the guest set to the real connection. A transport
    /// with no such control ignores it.
    fn set_option(&mut self, opt: ConnOption) {
        let _ = opt;
    }
    /// Bytes written but not yet acknowledged by the peer, if the transport can
    /// tell: what a lingering `close` waits to see drain. `None` means unknown,
    /// and then `close` doesn't wait.
    fn unsent(&mut self) -> Option<usize> {
        None
    }
}

/// A socket option a guest sets that also has to reach the real connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnOption {
    /// `TCP_NODELAY`.
    NoDelay(bool),
    /// `SO_KEEPALIVE`: probe an idle connection, so a dead peer is noticed.
    KeepAlive(bool),
    /// `TCP_KEEPIDLE`, `TCP_KEEPINTVL`, `TCP_KEEPCNT`: seconds idle before the
    /// first probe, seconds between probes, and how many go unanswered first.
    KeepIdle(u32),
    KeepInterval(u32),
    KeepCount(u32),
    /// `SO_LINGER` with a zero timeout: closing resets the connection rather
    /// than ending it cleanly. A nonzero timeout makes the guest's `close` wait
    /// for what it sent to be acknowledged (see [`HostConn::unsent`]); the host
    /// socket itself never lingers, since that would block the VM.
    ResetOnClose(bool),
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

/// The guest-visible network interface an [`Egress`] provides, shown to the
/// guest as `tun0` (by `ip addr`, `ip route`, `ifconfig`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Link {
    /// The interface MTU.
    pub mtu: u32,
    /// The guest's IPv4 address and prefix length, if it has one.
    pub v4: Option<([u8; 4], u8)>,
    /// The guest's IPv6 address and prefix length, if it has one.
    pub v6: Option<([u8; 16], u8)>,
}

/// Opens host-side connections for the guest. One installed instance is shared
/// by the whole VM. The native impl is [`HostEgress`]; a browser transport
/// (WebSocket relay / pktkit) will be another implementor.
pub trait Egress: Send + Debug {
    /// Open a stream connection to `(ip, v6, port)`. May block briefly, or
    /// return with the handshake still in flight (see
    /// [`HostConn::poll_connect`]). A `NetworkUnreachable` error means there is
    /// no route (the guest sees `ENETUNREACH`).
    fn connect_tcp(&self, ip: [u8; 16], v6: bool, port: u16) -> io::Result<Box<dyn HostConn>>;
    /// Open a datagram socket for UDP egress.
    fn open_udp(&self) -> io::Result<Box<dyn HostDgram>>;
    /// Open an ICMP (`v6`: ICMPv6) endpoint, for guest raw/ping sockets.
    /// `send_to` takes one ICMP message (header + body; the port is ignored)
    /// and sends it in an IP packet from our address — for ICMPv6 the
    /// transport fills in the checksum, which needs the pseudo-header. Every
    /// ICMP packet of that family that arrives is offered to every open
    /// endpoint: `recv_from` yields `(source, v6, 0, the whole IP packet)`,
    /// and the socket layer shapes it for the guest socket type. Unsupported
    /// by default (a host needs privileges for raw ICMP).
    fn open_icmp(&self, v6: bool) -> io::Result<Box<dyn HostDgram>> {
        let _ = v6;
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "ICMP egress not supported by this backend",
        ))
    }
    /// The interface this backend gives the guest, while it is up. `None` (the
    /// default) shows only `lo`, as when the host's own network is used.
    fn link(&self) -> Option<Link> {
        None
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::HostEgress;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::{ConnOption, Egress, HostConn, HostDgram};
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
            // The handshake happens on a thread of its own and the connection
            // reports it through `poll_connect`: a blocking connect here froze
            // the whole VM — every guest task, timers included — for as long
            // as an address took to answer (up to the timeout, for one that
            // never does), which a client trying several addresses (bun) turned
            // into a hang.
            let addr = sockaddr(ip, v6, port);
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::Builder::new()
                .name("nixvm-connect".into())
                .spawn(move || {
                    let _ = tx.send(TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT));
                })?;
            Ok(Box::new(TcpConn {
                state: ConnState::Connecting(rx),
                options: Vec::new(),
                shut_write: false,
            }))
        }

        fn open_udp(&self) -> io::Result<Box<dyn HostDgram>> {
            let sock = UdpSocket::bind(("0.0.0.0", 0))?;
            sock.set_nonblocking(true)?;
            Ok(Box::new(UdpConn(sock)))
        }
    }

    /// How long a host connect may take before it fails: about what Linux's
    /// SYN retries add up to.
    const CONNECT_TIMEOUT: Duration = Duration::from_secs(75);

    /// A host TCP connection: its handshake, on a helper thread, then the
    /// non-blocking stream.
    #[derive(Debug)]
    struct TcpConn {
        state: ConnState,
        /// Options set while still connecting, applied once it's up.
        options: Vec<ConnOption>,
        /// `shutdown(SHUT_WR)` while still connecting.
        shut_write: bool,
    }

    #[derive(Debug)]
    enum ConnState {
        Connecting(std::sync::mpsc::Receiver<io::Result<TcpStream>>),
        Up(TcpStream),
        /// The handshake failed; its error, until reported.
        Failed(Option<io::Error>),
    }

    fn apply_option(stream: &TcpStream, opt: ConnOption) {
        match opt {
            ConnOption::NoDelay(on) => {
                let _ = stream.set_nodelay(on);
            }
            #[cfg(unix)]
            other => super::sockopt::apply(stream, other),
            #[cfg(not(unix))]
            _ => {}
        }
    }

    impl TcpConn {
        /// The stream, once the handshake is done.
        fn stream(&mut self) -> Option<&mut TcpStream> {
            if let ConnState::Connecting(rx) = &self.state {
                match rx.try_recv() {
                    Ok(Ok(stream)) => {
                        let _ = stream.set_nonblocking(true);
                        let _ = stream.set_nodelay(true);
                        for opt in self.options.drain(..) {
                            apply_option(&stream, opt);
                        }
                        if self.shut_write {
                            let _ = stream.shutdown(std::net::Shutdown::Write);
                        }
                        self.state = ConnState::Up(stream);
                    }
                    Ok(Err(e)) => self.state = ConnState::Failed(Some(e)),
                    Err(std::sync::mpsc::TryRecvError::Empty) => {}
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        self.state =
                            ConnState::Failed(Some(io::Error::other("connect thread died")));
                    }
                }
            }
            match &mut self.state {
                ConnState::Up(stream) => Some(stream),
                _ => None,
            }
        }

        fn not_up(&mut self) -> io::Error {
            match &mut self.state {
                ConnState::Failed(e) => e
                    .take()
                    .unwrap_or_else(|| io::Error::from(io::ErrorKind::NotConnected)),
                _ => io::Error::from(io::ErrorKind::WouldBlock),
            }
        }
    }

    impl HostConn for TcpConn {
        fn recv(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            match self.stream() {
                Some(s) => s.read(buf),
                None => Err(self.not_up()),
            }
        }
        fn send(&mut self, buf: &[u8]) -> io::Result<usize> {
            match self.stream() {
                Some(s) => s.write(buf),
                None => Err(self.not_up()),
            }
        }
        fn shutdown_write(&mut self) {
            match self.stream() {
                Some(s) => {
                    let _ = s.shutdown(std::net::Shutdown::Write);
                }
                None => self.shut_write = true,
            }
        }
        fn poll_connect(&mut self) -> io::Result<bool> {
            if self.stream().is_some() {
                return Ok(true);
            }
            match &mut self.state {
                ConnState::Connecting(_) => Ok(false),
                _ => Err(self.not_up()),
            }
        }
        fn set_option(&mut self, opt: ConnOption) {
            match self.stream() {
                Some(s) => apply_option(s, opt),
                None => self.options.push(opt),
            }
        }
        #[cfg(unix)]
        fn unsent(&mut self) -> Option<usize> {
            super::sockopt::unsent(self.stream()?)
        }
        fn poll_readable(&mut self) -> bool {
            // A non-blocking peek: any bytes (or a clean EOF, reported as
            // `Ok(0)`) means a subsequent recv won't block. `WouldBlock` means
            // nothing yet. Any other error also counts as "ready" so the guest
            // observes the error via recv rather than spinning — a failed
            // handshake too; one still in progress has nothing to read.
            let mut b = [0u8; 1];
            match self.stream() {
                Some(s) => match s.peek(&mut b) {
                    Ok(_) => true,
                    Err(e) => e.kind() != io::ErrorKind::WouldBlock,
                },
                None => matches!(self.state, ConnState::Failed(_)),
            }
        }
        fn readable_len(&mut self) -> usize {
            // Peek into a page-sized buffer for the queued byte count. Capped at
            // the buffer — enough to size a read; a client re-checks after
            // draining. `WouldBlock`/errors report 0 (nothing to read now).
            let mut b = [0u8; 4096];
            self.stream().map_or(0, |s| s.peek(&mut b).unwrap_or(0))
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

/// Setting the options `std` has no call for on a host socket.
///
/// `unsafe`: `setsockopt(2)` is declared here by hand, since `std` doesn't expose
/// it and the crate takes no FFI dependency. Every call passes a live descriptor
/// borrowed from a `TcpStream` and a pointer to a local `c_int` with its size.
#[cfg(all(unix, not(target_arch = "wasm32")))]
mod sockopt {
    use super::ConnOption;
    use std::os::fd::AsRawFd;
    use std::os::raw::{c_int, c_void};

    unsafe extern "C" {
        fn setsockopt(fd: c_int, level: c_int, name: c_int, val: *const c_void, len: u32) -> c_int;
        #[cfg(target_os = "linux")]
        fn ioctl(fd: c_int, request: std::os::raw::c_ulong, ...) -> c_int;
        #[cfg(not(target_os = "linux"))]
        fn getsockopt(
            fd: c_int,
            level: c_int,
            name: c_int,
            val: *mut c_void,
            len: *mut u32,
        ) -> c_int;
    }

    /// Bytes sent on `sock` that the peer hasn't acknowledged yet (Linux's
    /// `SIOCOUTQ`, which counts the FIN too; the BSDs' `SO_NWRITE`).
    pub(super) fn unsent(sock: &impl AsRawFd) -> Option<usize> {
        let mut n: c_int = 0;
        #[cfg(target_os = "linux")]
        // SAFETY: a live descriptor, and `SIOCOUTQ` writes one int to `n`.
        let r = unsafe { ioctl(sock.as_raw_fd(), 0x5411, &raw mut n) };
        #[cfg(not(target_os = "linux"))]
        let r = {
            let mut len = std::mem::size_of::<c_int>() as u32;
            // SAFETY: a live descriptor, and `n` is `len` writable bytes.
            unsafe {
                getsockopt(
                    sock.as_raw_fd(),
                    consts::SOL_SOCKET,
                    0x1024, // SO_NWRITE
                    (&raw mut n).cast(),
                    &raw mut len,
                )
            }
        };
        if r == 0 {
            usize::try_from(n).ok()
        } else {
            None
        }
    }

    #[cfg(target_os = "linux")]
    mod consts {
        use std::os::raw::c_int;
        pub(super) const SOL_SOCKET: c_int = 1;
        pub(super) const SO_KEEPALIVE: c_int = 9;
        pub(super) const SO_LINGER: c_int = 13;
        pub(super) const TCP_KEEPIDLE: c_int = 4;
        pub(super) const TCP_KEEPINTVL: c_int = 5;
        pub(super) const TCP_KEEPCNT: c_int = 6;
    }
    #[cfg(not(target_os = "linux"))]
    mod consts {
        // The BSDs and macOS.
        use std::os::raw::c_int;
        pub(super) const SOL_SOCKET: c_int = 0xffff;
        pub(super) const SO_KEEPALIVE: c_int = 0x8;
        pub(super) const SO_LINGER: c_int = 0x80;
        #[cfg(target_vendor = "apple")]
        pub(super) const TCP_KEEPIDLE: c_int = 0x10; // TCP_KEEPALIVE
        #[cfg(not(target_vendor = "apple"))]
        pub(super) const TCP_KEEPIDLE: c_int = 0x100;
        #[cfg(target_vendor = "apple")]
        pub(super) const TCP_KEEPINTVL: c_int = 0x101;
        #[cfg(not(target_vendor = "apple"))]
        pub(super) const TCP_KEEPINTVL: c_int = 0x200;
        #[cfg(target_vendor = "apple")]
        pub(super) const TCP_KEEPCNT: c_int = 0x102;
        #[cfg(not(target_vendor = "apple"))]
        pub(super) const TCP_KEEPCNT: c_int = 0x400;
    }
    use consts::{SO_KEEPALIVE, SO_LINGER, SOL_SOCKET, TCP_KEEPCNT, TCP_KEEPIDLE, TCP_KEEPINTVL};
    const IPPROTO_TCP: c_int = 6;

    /// Set `opt` on `sock`. Failure is ignored: the guest's own view of the
    /// option is already recorded, and the connection works either way.
    pub(super) fn apply(sock: &impl AsRawFd, opt: ConnOption) {
        let int = |v: u32| c_int::try_from(v).unwrap_or(c_int::MAX);
        match opt {
            ConnOption::NoDelay(_) => {}
            ConnOption::KeepAlive(on) => set(sock, SOL_SOCKET, SO_KEEPALIVE, &[c_int::from(on)]),
            ConnOption::KeepIdle(s) => set(sock, IPPROTO_TCP, TCP_KEEPIDLE, &[int(s)]),
            ConnOption::KeepInterval(s) => set(sock, IPPROTO_TCP, TCP_KEEPINTVL, &[int(s)]),
            ConnOption::KeepCount(n) => set(sock, IPPROTO_TCP, TCP_KEEPCNT, &[int(n)]),
            // struct linger { int l_onoff; int l_linger; }
            ConnOption::ResetOnClose(on) => set(sock, SOL_SOCKET, SO_LINGER, &[c_int::from(on), 0]),
        }
    }

    fn set(sock: &impl AsRawFd, level: c_int, name: c_int, val: &[c_int]) {
        let len = std::mem::size_of_val(val) as u32;
        // SAFETY: the descriptor is open for the borrow's lifetime, and `val`
        // points at `len` readable bytes.
        unsafe {
            setsockopt(sock.as_raw_fd(), level, name, val.as_ptr().cast(), len);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{ConnOption, apply};
        use std::net::{TcpListener, TcpStream};
        use std::os::fd::AsRawFd;
        use std::os::raw::{c_int, c_void};

        unsafe extern "C" {
            fn getsockopt(
                fd: c_int,
                level: c_int,
                name: c_int,
                val: *mut c_void,
                len: *mut u32,
            ) -> c_int;
        }

        fn get(s: &TcpStream, level: c_int, name: c_int) -> [c_int; 2] {
            let mut v = [0 as c_int; 2];
            let mut len = std::mem::size_of_val(&v) as u32;
            // SAFETY: a live descriptor and a buffer of `len` writable bytes.
            let r = unsafe {
                getsockopt(
                    s.as_raw_fd(),
                    level,
                    name,
                    v.as_mut_ptr().cast(),
                    &raw mut len,
                )
            };
            assert_eq!(r, 0);
            v
        }

        #[test]
        fn options_reach_the_host_socket() {
            let l = TcpListener::bind("127.0.0.1:0").unwrap();
            let s = TcpStream::connect(l.local_addr().unwrap()).unwrap();
            apply(&s, ConnOption::KeepAlive(true));
            assert_ne!(get(&s, super::SOL_SOCKET, super::SO_KEEPALIVE)[0], 0);
            apply(&s, ConnOption::KeepIdle(42));
            assert_eq!(get(&s, super::IPPROTO_TCP, super::TCP_KEEPIDLE)[0], 42);
            apply(&s, ConnOption::KeepInterval(7));
            assert_eq!(get(&s, super::IPPROTO_TCP, super::TCP_KEEPINTVL)[0], 7);
            apply(&s, ConnOption::KeepCount(3));
            assert_eq!(get(&s, super::IPPROTO_TCP, super::TCP_KEEPCNT)[0], 3);
            apply(&s, ConnOption::ResetOnClose(true));
            assert_ne!(get(&s, super::SOL_SOCKET, super::SO_LINGER)[0], 0);
        }

        #[test]
        fn unsent_counts_what_the_peer_has_not_acknowledged() {
            use std::io::Write;
            let l = TcpListener::bind("127.0.0.1:0").unwrap();
            let mut s = TcpStream::connect(l.local_addr().unwrap()).unwrap();
            let (mut peer, _) = l.accept().unwrap();
            assert_eq!(super::unsent(&s), Some(0));
            // The peer never reads: once its window is full, sent bytes stay
            // unacknowledged.
            s.set_nonblocking(true).unwrap();
            let chunk = vec![7u8; 65536];
            while s.write(&chunk).is_ok() {}
            assert!(super::unsent(&s).unwrap() > 0);
            // Reading it all lets the rest be acknowledged.
            peer.set_nonblocking(true).unwrap();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
            let mut buf = vec![0u8; 1 << 20];
            while super::unsent(&s) != Some(0) {
                assert!(std::time::Instant::now() < deadline, "never drained");
                let _ = std::io::Read::read(&mut peer, &mut buf);
            }
        }
    }
}

#[cfg(all(test, unix, not(target_arch = "wasm32")))]
mod connect_tests {
    use super::{Egress, HostEgress};
    use std::net::TcpListener;
    use std::time::{Duration, Instant};

    /// A host connect returns at once and finishes in the background: a
    /// blocking one froze the whole VM, every guest task's timers included,
    /// for as long as the address took to answer.
    #[test]
    fn a_host_connect_never_blocks_the_vm() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut ip = [0u8; 16];
        ip[..4].copy_from_slice(&[127, 0, 0, 1]);
        let port = l.local_addr().unwrap().port();
        let mut c = HostEgress.connect_tcp(ip, false, port).unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while !c.poll_connect().unwrap() {
            assert!(Instant::now() < deadline, "the connect never completed");
            std::thread::sleep(Duration::from_millis(1));
        }
        let (mut peer, _) = l.accept().unwrap();
        assert_eq!(c.send(b"hi").unwrap(), 2);
        let mut b = [0u8; 2];
        std::io::Read::read_exact(&mut peer, &mut b).unwrap();
        assert_eq!(&b, b"hi");

        // An address that never answers (TEST-NET-1, RFC 5737): the call
        // still comes straight back, the handshake pending (or refused by the
        // host's routing at once).
        let started = Instant::now();
        let mut ip = [0u8; 16];
        ip[..4].copy_from_slice(&[192, 0, 2, 1]);
        if let Ok(mut c) = HostEgress.connect_tcp(ip, false, 9) {
            let pending = c.poll_connect();
            assert!(!matches!(pending, Ok(true)), "connected to a black hole?");
        }
        assert!(
            started.elapsed() < Duration::from_millis(500),
            "connect blocked"
        );
    }
}
