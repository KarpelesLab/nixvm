//! Local (in-kernel) sockets: `socketpair`, AF_UNIX stream sockets, and an
//! AF_INET/AF_INET6 loopback (TCP stream + UDP datagram) — all serviced
//! entirely inside nixvm with no host networking. External/internet traffic
//! is out of scope (it needs a userspace IP stack); this layer only connects
//! endpoints that both live in this VM, and only over the loopback addresses
//! `127.0.0.1`/`0.0.0.0` (v4) and `::1`/`::` (v6).
//!
//! A connected stream socket is a [`Pair`]: two byte buffers, one per
//! direction, with a small open-fd refcount per end — a bidirectional pipe.
//! `socket()` yields an unconnected [`Kind::Idle`] slot that `bind`/`listen`
//! turn into a [`Kind::Listener`] (an address plus a backlog of pending
//! connections) and that `connect` turns into a [`Pair`], queuing its index on
//! the target listener for `accept4` to hand back as the server-side end.
//!
//! A `SOCK_DGRAM` socket is a [`Dgram`]: an optional bound local address, an
//! optional `connect`-ed peer, and a queue of inbound `(source, payload)`
//! datagrams. `AF_INET`/`AF_INET6` port numbers live in a single namespace per
//! transport protocol (`tcp4`/`tcp6`/`udp4`/`udp6`), keyed only by family and
//! port — since only loopback/wildcard binds are accepted, the specific
//! address never needs to distinguish two sockets on the same port.

use std::collections::{BTreeMap, VecDeque};

use crate::abi::errno::Errno;
use crate::vcpu::GuestMemory;

use super::{Fd, Kernel, err};

const AF_UNIX: u16 = 1;
const AF_INET: u16 = 2;
const AF_INET6: u16 = 10;
const SOCK_STREAM: u64 = 1;
const SOCK_DGRAM: u64 = 2;
const SOCK_NONBLOCK: u64 = 0o4000;

const SOL_SOCKET: u64 = 1;
const SO_REUSEADDR: u64 = 2;
const SO_TYPE: u64 = 3;
// SO_ERROR (4) is handled by the wildcard arm below: always 0, since there is
// no pending-error model to report.
const SO_SNDBUF: u64 = 7;
const SO_RCVBUF: u64 = 8;

/// The kernel's socket table plus the AF_UNIX/AF_INET(6) address registries.
#[derive(Debug, Default)]
pub(super) struct Net {
    socks: Vec<Sock>,
    /// AF_UNIX path, or `"tcp4:port"`/`"tcp6:port"` -> listening slot.
    listeners: BTreeMap<String, usize>,
    /// `"udp4:port"`/`"udp6:port"` -> bound `Dgram` slot.
    dgram_ports: BTreeMap<String, usize>,
}

/// One entry in the socket table.
#[derive(Debug)]
struct Sock {
    domain: u16,
    kind: Kind,
    /// `SOCK_NONBLOCK` at creation time (there is no `fcntl(F_SETFL)` wiring
    /// into this module, so this is fixed at `socket()`/`accept4()`).
    nonblock: bool,
    /// `SO_REUSEADDR`, set via `setsockopt`; relaxes the `bind` `EADDRINUSE`
    /// check for `AF_INET`/`AF_INET6`.
    reuseaddr: bool,
}

/// The lifecycle state of a socket slot.
#[derive(Debug)]
enum Kind {
    /// Created by `socket()`, optionally `bind`-ed to `bound` but not yet
    /// listening or connected. Stream sockets only (`SOCK_DGRAM` uses
    /// [`Kind::Dgram`] instead).
    Idle { bound: Option<Addr> },
    /// `listen()`-ed: `backlog` holds the slot indices of pending connections
    /// (each already a [`Pair`]) waiting for `accept4`.
    Listener {
        addr: Option<Addr>,
        backlog: VecDeque<usize>,
    },
    /// A connected pair of byte streams (AF_UNIX, or AF_INET/AF_INET6 TCP).
    Pair(Pair),
    /// A `SOCK_DGRAM` endpoint (AF_INET/AF_INET6 UDP).
    Dgram(Dgram),
}

/// A connected socket: `to[e]` holds bytes destined for end `e` (so end `e`
/// reads `to[e]` and writes `to[1 - e]`). `refs[e]` counts open fds on end `e`;
/// `shut[e]` records a write-side `shutdown` from end `e`; `nonblock[e]`
/// mirrors the `O_NONBLOCK`/`SOCK_NONBLOCK` of the fd currently on end `e`.
/// `addrs[e]` is end `e`'s local `AF_INET`/`AF_INET6` address (so its peer
/// address is `addrs[1 - e]`); `None` for AF_UNIX pairs.
#[derive(Debug)]
struct Pair {
    to: [VecDeque<u8>; 2],
    refs: [usize; 2],
    shut: [bool; 2],
    nonblock: [bool; 2],
    addrs: [Option<InetAddr>; 2],
}

impl Pair {
    /// A freshly connected pair with one open reference on each end.
    fn new() -> Self {
        Self {
            to: [VecDeque::new(), VecDeque::new()],
            refs: [1, 1],
            shut: [false, false],
            nonblock: [false, false],
            addrs: [None, None],
        }
    }
}

/// A `SOCK_DGRAM` endpoint: `local` is the bound (or lazily ephemeral-assigned)
/// address, `peer` is the `connect()`-ed destination (if any), and `queue`
/// holds inbound `(source address, payload)` datagrams awaiting `recv`.
#[derive(Debug, Default)]
struct Dgram {
    local: Option<InetAddr>,
    peer: Option<InetAddr>,
    queue: VecDeque<(InetAddr, Vec<u8>)>,
}

/// A decoded `sockaddr`: an AF_UNIX path, or an AF_INET/AF_INET6 endpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Addr {
    Unix(String),
    Inet(InetAddr),
}

/// An AF_INET/AF_INET6 address. `ip` holds the IPv4 address in its low 4
/// bytes (rest zero) when `!v6`, or the full 16-byte IPv6 address when `v6`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct InetAddr {
    v6: bool,
    port: u16,
    ip: [u8; 16],
}

impl InetAddr {
    /// `INADDR_ANY`/`in6addr_any` (`0.0.0.0`/`::`).
    fn is_any(&self) -> bool {
        self.ip == [0u8; 16]
    }

    /// `127.0.0.0/8` (v4) or `::1` (v6).
    fn is_loopback(&self) -> bool {
        if self.v6 { self.ip == loopback_ip(true) } else { self.ip[0] == 127 }
    }

    /// This is a self-contained VM loopback: only the wildcard or loopback
    /// address may be bound (or connected to).
    fn valid_bind(&self) -> bool {
        self.is_any() || self.is_loopback()
    }
}

/// The concrete loopback address (`127.0.0.1` or `::1`) for a family.
fn loopback_ip(v6: bool) -> [u8; 16] {
    let mut ip = [0u8; 16];
    if v6 {
        ip[15] = 1;
    } else {
        ip[0] = 127;
        ip[3] = 1;
    }
    ip
}

/// The port-table key for `proto` (`"tcp"`/`"udp"`) and address `a`, e.g.
/// `"tcp4:8080"`. Deliberately ignores the specific address: only
/// loopback/wildcard binds are valid here, so a port number alone identifies
/// the endpoint.
fn route_key(proto: &str, a: InetAddr) -> String {
    format!("{proto}{}:{}", if a.v6 { 6 } else { 4 }, a.port)
}

impl Net {
    /// Adjust the open-fd refcount of the socket end `fd` refers to (a no-op for
    /// non-socket fds or unconnected sockets). Mirrors `Kernel::bump_pipe`.
    pub(super) fn bump(&mut self, fd: &Fd, inc: bool) {
        if let Fd::Socket { sock, end } = *fd
            && let Some(Kind::Pair(p)) = self.socks.get_mut(sock).map(|s| &mut s.kind)
        {
            if inc {
                p.refs[end] += 1;
            } else {
                p.refs[end] = p.refs[end].saturating_sub(1);
            }
        }
    }

    /// True if some *other* socket already occupies `proto`/`a`'s port.
    fn addr_in_use(&self, proto: &str, a: InetAddr, exclude: usize) -> bool {
        let key = route_key(proto, a);
        self.socks.iter().enumerate().any(|(i, s)| {
            if i == exclude {
                return false;
            }
            match &s.kind {
                Kind::Idle { bound: Some(Addr::Inet(b)) } if proto == "tcp" => {
                    route_key("tcp", *b) == key
                }
                Kind::Listener { addr: Some(Addr::Inet(b)), .. } if proto == "tcp" => {
                    route_key("tcp", *b) == key
                }
                Kind::Dgram(d) if proto == "udp" => {
                    d.local.is_some_and(|b| route_key("udp", b) == key)
                }
                _ => false,
            }
        })
    }

    /// The lowest free port >= 32768 for `proto`/`v6` (the standard Linux
    /// ephemeral range, trimmed for a small in-VM table).
    fn ephemeral_port(&self, proto: &str, v6: bool) -> u16 {
        for port in 32_768u32..=60_999 {
            let a = InetAddr { v6, port: port as u16, ip: [0; 16] };
            if !self.addr_in_use(proto, a, usize::MAX) {
                return port as u16;
            }
        }
        0
    }

    /// A fresh client-side local address: loopback of `v6`'s family, with a
    /// freshly allocated ephemeral TCP port.
    fn fresh_local(&self, v6: bool) -> InetAddr {
        InetAddr { v6, port: self.ephemeral_port("tcp", v6), ip: loopback_ip(v6) }
    }
}

impl Kernel {
    /// The `(slot, end)` a socket fd points at, if it is a socket.
    fn sock_of(&self, fd: u64) -> Option<(usize, usize)> {
        match self.cur.fds.get(fd as i32) {
            Some(Fd::Socket { sock, end }) => Some((*sock, *end)),
            _ => None,
        }
    }

    /// Ensure `SOCK_DGRAM` socket `sock` has a local address, lazily
    /// auto-assigning (and registering in the port table) an ephemeral
    /// loopback one if it was never `bind`-ed — mirroring the implicit local
    /// address Linux assigns to an unbound socket on first send.
    fn ensure_dgram_bound(&mut self, sock: usize) -> InetAddr {
        if let Kind::Dgram(d) = &self.net.socks[sock].kind
            && let Some(local) = d.local
        {
            return local;
        }
        let v6 = self.net.socks[sock].domain == AF_INET6;
        let port = self.net.ephemeral_port("udp", v6);
        let local = InetAddr { v6, port, ip: loopback_ip(v6) };
        if let Kind::Dgram(d) = &mut self.net.socks[sock].kind {
            d.local = Some(local);
        }
        self.net.dgram_ports.insert(route_key("udp", local), sock);
        local
    }

    /// `socket(domain, type, protocol)` — an unbound, unconnected endpoint.
    pub(super) fn sys_socket(&mut self, domain: u64, sotype: u64, _protocol: u64) -> i64 {
        let domain = domain as u16;
        if domain != AF_UNIX && domain != AF_INET && domain != AF_INET6 {
            return err(Errno::EAFNOSUPPORT);
        }
        let base_type = sotype & 0xf;
        if base_type != SOCK_STREAM && !(domain != AF_UNIX && base_type == SOCK_DGRAM) {
            return err(Errno::EOPNOTSUPP);
        }
        let nonblock = sotype & SOCK_NONBLOCK != 0;
        let kind = if base_type == SOCK_DGRAM {
            Kind::Dgram(Dgram::default())
        } else {
            Kind::Idle { bound: None }
        };
        let idx = self.net.socks.len();
        self.net.socks.push(Sock { domain, kind, nonblock, reuseaddr: false });
        i64::from(self.cur.fds.alloc(Fd::Socket { sock: idx, end: 0 }))
    }

    /// `socketpair(domain, type, protocol, sv)` — a connected AF_UNIX pair whose
    /// two fds are written to `sv[0]`/`sv[1]`.
    pub(super) fn sys_socketpair(
        &mut self,
        domain: u64,
        sotype: u64,
        _protocol: u64,
        sv: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        if domain as u16 != AF_UNIX {
            return err(Errno::EAFNOSUPPORT);
        }
        if sotype & 0xf != SOCK_STREAM {
            return err(Errno::EOPNOTSUPP);
        }
        let nonblock = sotype & SOCK_NONBLOCK != 0;
        let mut pair = Pair::new();
        pair.nonblock = [nonblock, nonblock];
        let idx = self.net.socks.len();
        self.net.socks.push(Sock {
            domain: AF_UNIX,
            kind: Kind::Pair(pair),
            nonblock,
            reuseaddr: false,
        });
        let fd0 = self.cur.fds.alloc(Fd::Socket { sock: idx, end: 0 });
        let fd1 = self.cur.fds.alloc(Fd::Socket { sock: idx, end: 1 });
        let mut b = [0u8; 8];
        b[0..4].copy_from_slice(&fd0.to_le_bytes());
        b[4..8].copy_from_slice(&fd1.to_le_bytes());
        if mem.write(sv, &b).is_err() {
            return err(Errno::EFAULT);
        }
        0
    }

    /// `bind(fd, addr, addrlen)` — record the local address of an idle
    /// (stream) or unbound (datagram) socket. For `AF_INET`/`AF_INET6`, port
    /// `0` auto-assigns an ephemeral port, and only the wildcard/loopback
    /// address is accepted (no host networking).
    pub(super) fn sys_bind(&mut self, fd: u64, addr: u64, addrlen: u64, mem: &GuestMemory) -> i64 {
        let Some((sock, _)) = self.sock_of(fd) else {
            return err(Errno::ENOTSOCK);
        };
        let Some(parsed) = read_sockaddr(mem, addr, addrlen) else {
            return err(Errno::EINVAL);
        };
        match parsed {
            Addr::Unix(_) => {
                if self.net.socks[sock].domain != AF_UNIX {
                    return err(Errno::EINVAL);
                }
                match &mut self.net.socks[sock].kind {
                    Kind::Idle { bound } => {
                        *bound = Some(parsed);
                        0
                    }
                    _ => err(Errno::EINVAL),
                }
            }
            Addr::Inet(mut a) => {
                let domain = self.net.socks[sock].domain;
                if (a.v6 && domain != AF_INET6) || (!a.v6 && domain != AF_INET) {
                    return err(Errno::EINVAL);
                }
                if !a.valid_bind() {
                    return err(Errno::EINVAL); // real errno: EADDRNOTAVAIL
                }
                let proto = match &self.net.socks[sock].kind {
                    Kind::Idle { .. } => "tcp",
                    Kind::Dgram(_) => "udp",
                    _ => return err(Errno::EINVAL),
                };
                if a.port == 0 {
                    a.port = self.net.ephemeral_port(proto, a.v6);
                } else if !self.net.socks[sock].reuseaddr && self.net.addr_in_use(proto, a, sock) {
                    return err(Errno::EINVAL); // real errno: EADDRINUSE
                }
                match &mut self.net.socks[sock].kind {
                    Kind::Idle { bound } => *bound = Some(Addr::Inet(a)),
                    Kind::Dgram(d) => d.local = Some(a),
                    _ => return err(Errno::EINVAL),
                }
                if proto == "udp" {
                    self.net.dgram_ports.insert(route_key("udp", a), sock);
                }
                0
            }
        }
    }

    /// `listen(fd, backlog)` — mark a bound socket as accepting connections.
    /// Auto-binds an ephemeral wildcard address first if `bind` was skipped
    /// (matching real Linux).
    pub(super) fn sys_listen(&mut self, fd: u64) -> i64 {
        let Some((sock, _)) = self.sock_of(fd) else {
            return err(Errno::ENOTSOCK);
        };
        let domain = self.net.socks[sock].domain;
        let mut bound = match &self.net.socks[sock].kind {
            Kind::Idle { bound } => bound.clone(),
            Kind::Listener { .. } => return 0,
            _ => return err(Errno::EINVAL),
        };
        if bound.is_none() && domain != AF_UNIX {
            let v6 = domain == AF_INET6;
            let port = self.net.ephemeral_port("tcp", v6);
            bound = Some(Addr::Inet(InetAddr { v6, port, ip: [0; 16] }));
        }
        let key = match &bound {
            Some(Addr::Unix(p)) => Some(p.clone()),
            Some(Addr::Inet(a)) => Some(route_key("tcp", *a)),
            None => None,
        };
        self.net.socks[sock].kind = Kind::Listener { addr: bound, backlog: VecDeque::new() };
        if let Some(k) = key {
            self.net.listeners.insert(k, sock);
        }
        0
    }

    /// `connect(fd, addr, addrlen)` — for a stream socket, connect an idle
    /// socket to a listener, turning this slot into the client end of a fresh
    /// pair and queuing it on the listener for `accept4`; for a datagram
    /// socket, just record the peer (no handshake, per UDP semantics).
    pub(super) fn sys_connect(
        &mut self,
        fd: u64,
        addr: u64,
        addrlen: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let Some((sock, end)) = self.sock_of(fd) else {
            return err(Errno::ENOTSOCK);
        };
        let Some(target) = read_sockaddr(mem, addr, addrlen) else {
            return err(Errno::EINVAL);
        };
        let domain = self.net.socks[sock].domain;
        let mismatched = match (&target, domain) {
            (Addr::Unix(_), AF_UNIX) => false,
            (Addr::Inet(a), AF_INET) => a.v6,
            (Addr::Inet(a), AF_INET6) => !a.v6,
            _ => true,
        };
        if mismatched {
            return err(Errno::EINVAL);
        }
        if let Addr::Inet(a) = &target
            && !a.valid_bind()
        {
            return err(Errno::ECONNREFUSED); // no route beyond this VM's loopback
        }

        if matches!(self.net.socks[sock].kind, Kind::Dgram(_)) {
            let Addr::Inet(peer) = target else {
                unreachable!("validated above")
            };
            self.ensure_dgram_bound(sock);
            if let Kind::Dgram(d) = &mut self.net.socks[sock].kind {
                d.peer = Some(peer);
            }
            return 0;
        }

        // An idle client end 0; anything else (already connected, a listener,
        // or the wrong end) is invalid.
        if !matches!(&self.net.socks[sock].kind, Kind::Idle { .. } if end == 0) {
            return err(Errno::EINVAL);
        }
        let key = match &target {
            Addr::Unix(p) => p.clone(),
            Addr::Inet(a) => route_key("tcp", *a),
        };
        let Some(&lidx) = self.net.listeners.get(&key) else {
            return err(Errno::ECONNREFUSED);
        };
        let listener_addr = match &self.net.socks[lidx].kind {
            Kind::Listener { addr, .. } => addr.clone(),
            _ => return err(Errno::ECONNREFUSED),
        };
        // Repurpose the client's idle slot as the connected pair, then queue its
        // index (the server-side end 1) on the listener's backlog.
        let mut pair = Pair::new();
        pair.nonblock[0] = self.net.socks[sock].nonblock;
        if domain != AF_UNIX {
            let v6 = domain == AF_INET6;
            let mut peer_addr = match listener_addr {
                Some(Addr::Inet(a)) => a,
                _ => InetAddr { v6, port: 0, ip: loopback_ip(v6) },
            };
            if peer_addr.is_any() {
                // Report the concrete loopback even if the server bound ANY.
                peer_addr.ip = loopback_ip(v6);
            }
            pair.addrs[0] = Some(self.net.fresh_local(v6));
            pair.addrs[1] = Some(peer_addr);
        }
        self.net.socks[sock].kind = Kind::Pair(pair);
        if let Kind::Listener { backlog, .. } = &mut self.net.socks[lidx].kind {
            backlog.push_back(sock);
        }
        0
    }

    /// `accept4(fd, addr, addrlen, flags)` — hand back the server-side end of a
    /// pending connection (blocking, like pipe read, when none is queued and
    /// the listening socket is not `O_NONBLOCK`).
    pub(super) fn sys_accept4(
        &mut self,
        fd: u64,
        addr: u64,
        addrlen: u64,
        flags: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some((sock, _)) = self.sock_of(fd) else {
            return err(Errno::ENOTSOCK);
        };
        let nonblock = self.net.socks[sock].nonblock;
        let pending = match &mut self.net.socks[sock].kind {
            Kind::Listener { backlog, .. } => backlog.pop_front(),
            _ => return err(Errno::EINVAL),
        };
        let Some(pidx) = pending else {
            if nonblock {
                return err(Errno::EAGAIN);
            }
            self.block = true; // no pending connection yet — re-trap later
            return 0;
        };
        let domain = self.net.socks[sock].domain;
        let peer = match &self.net.socks[pidx].kind {
            Kind::Pair(p) => p.addrs[0].map(Addr::Inet),
            _ => None,
        };
        write_sockaddr(mem, addr, addrlen, domain, peer.as_ref());
        if let Kind::Pair(p) = &mut self.net.socks[pidx].kind {
            p.nonblock[1] = flags & SOCK_NONBLOCK != 0;
        }
        i64::from(self.cur.fds.alloc(Fd::Socket {
            sock: pidx,
            end: 1,
        }))
    }

    /// `getsockname(fd, addr, addrlen)` — the local address (best-effort).
    pub(super) fn sys_getsockname(
        &mut self,
        fd: u64,
        addr: u64,
        addrlen: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some((sock, end)) = self.sock_of(fd) else {
            return err(Errno::ENOTSOCK);
        };
        let domain = self.net.socks[sock].domain;
        let resolved = match &self.net.socks[sock].kind {
            Kind::Idle { bound } => bound.clone(),
            Kind::Listener { addr, .. } => addr.clone(),
            Kind::Pair(p) => p.addrs[end].map(Addr::Inet),
            Kind::Dgram(d) => d.local.map(Addr::Inet),
        };
        write_sockaddr(mem, addr, addrlen, domain, resolved.as_ref())
    }

    /// `getpeername(fd, addr, addrlen)` — the peer address (best-effort).
    pub(super) fn sys_getpeername(
        &mut self,
        fd: u64,
        addr: u64,
        addrlen: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some((sock, end)) = self.sock_of(fd) else {
            return err(Errno::ENOTSOCK);
        };
        let domain = self.net.socks[sock].domain;
        match &self.net.socks[sock].kind {
            Kind::Pair(p) => write_sockaddr(mem, addr, addrlen, domain, p.addrs[1 - end].map(Addr::Inet).as_ref()),
            Kind::Dgram(d) if d.peer.is_some() => {
                write_sockaddr(mem, addr, addrlen, domain, d.peer.map(Addr::Inet).as_ref())
            }
            _ => err(Errno::EINVAL), // real errno: ENOTCONN
        }
    }

    /// `shutdown(fd, how)` — mark this end's write side closed so the peer sees
    /// EOF once it drains.
    pub(super) fn sys_shutdown(&mut self, fd: u64, _how: u64) -> i64 {
        let Some((sock, end)) = self.sock_of(fd) else {
            return err(Errno::ENOTSOCK);
        };
        match &mut self.net.socks[sock].kind {
            Kind::Pair(p) => {
                p.shut[end] = true;
                0
            }
            _ => err(Errno::EINVAL), // real errno: ENOTCONN
        }
    }

    /// `setsockopt(fd, level, optname, optval, optlen)` — currently only
    /// `SOL_SOCKET`/`SO_REUSEADDR` has an observable effect (it relaxes the
    /// `bind` `EADDRINUSE` check); everything else (`TCP_NODELAY`, buffer
    /// size hints, …) is accepted and silently ignored, since this is a
    /// virtual loopback with no real transport to tune.
    ///
    /// Not currently reachable from the syscall dispatch: `kernel::mod`
    /// resolves `Sysno::Setsockopt` to an unconditional no-op before it would
    /// reach here (see the `net.rs` module ownership note in this crate's
    /// task history). Kept implemented and unit-tested directly so the
    /// behavior exists once that wiring is added.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_setsockopt(
        &mut self,
        fd: u64,
        level: u64,
        optname: u64,
        optval: u64,
        optlen: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let Some((sock, _)) = self.sock_of(fd) else {
            return err(Errno::ENOTSOCK);
        };
        if level == SOL_SOCKET
            && optname == SO_REUSEADDR
            && optlen >= 4
            && let Ok(v) = mem.read_u32(optval)
        {
            self.net.socks[sock].reuseaddr = v != 0;
        }
        0
    }

    /// `getsockopt(fd, level, optname, optval, optlen)` — sane canned values
    /// for `SO_TYPE`/`SO_ERROR`/`SO_REUSEADDR`/`SO_RCVBUF`/`SO_SNDBUF`; `0` for
    /// anything else.
    ///
    /// Not currently reachable from the syscall dispatch, for the same reason
    /// as [`Kernel::sys_setsockopt`].
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_getsockopt(
        &mut self,
        fd: u64,
        level: u64,
        optname: u64,
        optval: u64,
        optlen_ptr: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some((sock, _)) = self.sock_of(fd) else {
            return err(Errno::ENOTSOCK);
        };
        let value: u32 = if level == SOL_SOCKET {
            match optname {
                SO_TYPE => match self.net.socks[sock].kind {
                    Kind::Dgram(_) => SOCK_DGRAM as u32,
                    _ => SOCK_STREAM as u32,
                },
                SO_REUSEADDR => u32::from(self.net.socks[sock].reuseaddr),
                SO_RCVBUF | SO_SNDBUF => 212_992,
                _ => 0, // includes SO_ERROR
            }
        } else {
            0
        };
        if optval != 0 {
            let _ = mem.write(optval, &value.to_le_bytes());
        }
        if optlen_ptr != 0 {
            let _ = mem.write(optlen_ptr, &4u32.to_le_bytes());
        }
        0
    }

    /// `sendto(fd, buf, len, flags, dest_addr, addrlen)` — for a datagram
    /// socket with an explicit destination, deliver straight into that port's
    /// inbound queue (fire-and-forget, like real UDP: no error if nothing is
    /// bound there); otherwise (no destination, or a stream socket) this is
    /// just `write`.
    ///
    /// Not currently reachable from the syscall dispatch: `kernel::mod`
    /// resolves both `Sysno::Write` and `Sysno::Sendto` to `sys_write` (3
    /// args only), so a real `sendto(2)` with a destination address never
    /// reaches this method today. Kept implemented and unit-tested directly
    /// so the address-aware behavior exists once that wiring is added.
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_sendto(
        &mut self,
        fd: u64,
        buf: u64,
        len: u64,
        _flags: u64,
        dest_addr: u64,
        dest_addrlen: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let Some((sock, end)) = self.sock_of(fd) else {
            return err(Errno::ENOTSOCK);
        };
        if dest_addr == 0 {
            let Ok(data) = mem.read_vec(buf, len as usize) else {
                return err(Errno::EFAULT);
            };
            return self.write_socket(sock, end, &data);
        }
        let Some(Addr::Inet(dest)) = read_sockaddr(mem, dest_addr, dest_addrlen) else {
            return err(Errno::EINVAL);
        };
        if !matches!(self.net.socks[sock].kind, Kind::Dgram(_)) {
            return err(Errno::EINVAL); // real errno: EOPNOTSUPP/EISCONN
        }
        if !dest.valid_bind() {
            return err(Errno::EINVAL); // no route beyond this VM's loopback
        }
        let Ok(data) = mem.read_vec(buf, len as usize) else {
            return err(Errno::EFAULT);
        };
        let src = self.ensure_dgram_bound(sock);
        let key = route_key("udp", dest);
        if let Some(&tgt) = self.net.dgram_ports.get(&key)
            && let Kind::Dgram(td) = &mut self.net.socks[tgt].kind
        {
            td.queue.push_back((src, data.clone()));
        }
        data.len() as i64
    }

    /// `recvfrom(fd, buf, len, flags, src_addr, addrlen)` — for a datagram
    /// socket, pop the next queued datagram and report its source address;
    /// for a stream socket this is `read` plus (best-effort) the peer address.
    ///
    /// Not currently reachable from the syscall dispatch, for the same reason
    /// as [`Kernel::sys_sendto`].
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_recvfrom(
        &mut self,
        fd: u64,
        buf: u64,
        len: u64,
        _flags: u64,
        src_addr: u64,
        src_addrlen: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some((sock, end)) = self.sock_of(fd) else {
            return err(Errno::ENOTSOCK);
        };
        if !matches!(self.net.socks[sock].kind, Kind::Dgram(_)) {
            let n = self.read_socket(sock, end, buf, len, mem);
            if n > 0 {
                let domain = self.net.socks[sock].domain;
                let peer = match &self.net.socks[sock].kind {
                    Kind::Pair(p) => p.addrs[1 - end].map(Addr::Inet),
                    _ => None,
                };
                write_sockaddr(mem, src_addr, src_addrlen, domain, peer.as_ref());
            }
            return n;
        }
        let nonblock = self.net.socks[sock].nonblock;
        let popped = match &mut self.net.socks[sock].kind {
            Kind::Dgram(d) => d.queue.pop_front(),
            _ => unreachable!("checked above"),
        };
        let Some((from, data)) = popped else {
            if nonblock {
                return err(Errno::EAGAIN);
            }
            self.block = true;
            return 0;
        };
        let domain = self.net.socks[sock].domain;
        write_sockaddr(mem, src_addr, src_addrlen, domain, Some(&Addr::Inet(from)));
        let n = (len as usize).min(data.len());
        if mem.write(buf, &data[..n]).is_err() {
            return err(Errno::EFAULT);
        }
        n as i64
    }

    /// Read from socket `sock`'s inbound queue for `end`. For a stream pair:
    /// empty with the peer still open -> block (or `EAGAIN` if `O_NONBLOCK`);
    /// empty with the peer closed -> EOF (0). For a datagram socket: pop the
    /// next queued datagram's payload (no address; that's `recvfrom`'s job),
    /// blocking/`EAGAIN`-ing the same way while the queue is empty (UDP sockets
    /// never see EOF). Mirrors `read_pipe`.
    pub(super) fn read_socket(
        &mut self,
        sock: usize,
        end: usize,
        buf: u64,
        count: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        if matches!(self.net.socks[sock].kind, Kind::Dgram(_)) {
            let nonblock = self.net.socks[sock].nonblock;
            let popped = match &mut self.net.socks[sock].kind {
                Kind::Dgram(d) => d.queue.pop_front(),
                _ => unreachable!("checked above"),
            };
            let Some((_, data)) = popped else {
                if nonblock {
                    return err(Errno::EAGAIN);
                }
                self.block = true;
                return 0;
            };
            let n = (count as usize).min(data.len());
            if mem.write(buf, &data[..n]).is_err() {
                return err(Errno::EFAULT);
            }
            return n as i64;
        }
        let (empty, peer_open, nonblock) = match &self.net.socks[sock].kind {
            Kind::Pair(p) => (
                p.to[end].is_empty(),
                p.refs[1 - end] > 0 && !p.shut[1 - end],
                p.nonblock[end],
            ),
            _ => return err(Errno::EINVAL),
        };
        if empty {
            if peer_open {
                if nonblock {
                    return err(Errno::EAGAIN);
                }
                self.block = true;
            }
            return 0;
        }
        let data: Vec<u8> = match &mut self.net.socks[sock].kind {
            Kind::Pair(p) => {
                let n = count.min(p.to[end].len() as u64) as usize;
                p.to[end].drain(..n).collect()
            }
            _ => return err(Errno::EINVAL),
        };
        if mem.write(buf, &data).is_err() {
            return err(Errno::EFAULT);
        }
        data.len() as i64
    }

    /// Append to socket `sock`'s outbound queue for `end`. For a stream pair,
    /// `EPIPE` if the peer end is fully closed. For a datagram socket, this is
    /// `send` (i.e. requires a `connect`-ed peer — `EPIPE` stands in for the
    /// real `ENOTCONN`, unavailable in this crate's errno set) and delivers
    /// fire-and-forget, like real UDP: no error if nothing is bound at the
    /// peer's port. Mirrors `write_pipe`.
    pub(super) fn write_socket(&mut self, sock: usize, end: usize, data: &[u8]) -> i64 {
        if matches!(self.net.socks[sock].kind, Kind::Dgram(_)) {
            let peer = match &self.net.socks[sock].kind {
                Kind::Dgram(d) => d.peer,
                _ => unreachable!("checked above"),
            };
            let Some(peer) = peer else {
                return err(Errno::EPIPE); // real errno: ENOTCONN
            };
            let src = self.ensure_dgram_bound(sock);
            let key = route_key("udp", peer);
            if let Some(&tgt) = self.net.dgram_ports.get(&key)
                && let Kind::Dgram(td) = &mut self.net.socks[tgt].kind
            {
                td.queue.push_back((src, data.to_vec()));
            }
            return data.len() as i64;
        }
        match &mut self.net.socks[sock].kind {
            Kind::Pair(p) => {
                if p.refs[1 - end] == 0 {
                    return err(Errno::EPIPE);
                }
                p.to[1 - end].extend(data.iter().copied());
                data.len() as i64
            }
            _ => err(Errno::EINVAL),
        }
    }
}

/// Decode a `sockaddr` into an [`Addr`]: the `sun_path` for AF_UNIX, or the
/// port/address for AF_INET (`struct sockaddr_in`) / AF_INET6
/// (`struct sockaddr_in6`). Abstract (leading-NUL) AF_UNIX names are treated
/// as their raw path.
fn read_sockaddr(mem: &GuestMemory, ptr: u64, addrlen: u64) -> Option<Addr> {
    if addrlen < 2 {
        return None;
    }
    let bytes = mem.read_vec(ptr, (addrlen as usize).min(128)).ok()?;
    let family = u16::from_le_bytes([bytes[0], bytes[1]]);
    match family {
        AF_UNIX => {
            let path = &bytes[2..];
            let end = path.iter().position(|&c| c == 0).unwrap_or(path.len());
            Some(Addr::Unix(String::from_utf8_lossy(&path[..end]).into_owned()))
        }
        AF_INET if bytes.len() >= 8 => {
            let port = u16::from_be_bytes([bytes[2], bytes[3]]);
            let mut ip = [0u8; 16];
            ip[0..4].copy_from_slice(&bytes[4..8]);
            Some(Addr::Inet(InetAddr { v6: false, port, ip }))
        }
        AF_INET6 if bytes.len() >= 24 => {
            let port = u16::from_be_bytes([bytes[2], bytes[3]]);
            let mut ip = [0u8; 16];
            ip.copy_from_slice(&bytes[8..24]);
            Some(Addr::Inet(InetAddr { v6: true, port, ip }))
        }
        _ => None,
    }
}

/// Encode `a` as a `struct sockaddr_in`/`sockaddr_in6` byte buffer.
fn encode_inet_sockaddr(a: InetAddr) -> Vec<u8> {
    if a.v6 {
        let mut b = vec![0u8; 28];
        b[0..2].copy_from_slice(&AF_INET6.to_le_bytes());
        b[2..4].copy_from_slice(&a.port.to_be_bytes());
        // b[4..8] flowinfo, b[24..28] scope_id: left zero.
        b[8..24].copy_from_slice(&a.ip);
        b
    } else {
        let mut b = vec![0u8; 16];
        b[0..2].copy_from_slice(&AF_INET.to_le_bytes());
        b[2..4].copy_from_slice(&a.port.to_be_bytes());
        b[4..8].copy_from_slice(&a.ip[0..4]);
        // b[8..16] sin_zero: left zero.
        b
    }
}

/// Write a best-effort `sockaddr` to `addr`, truncated to the caller's buffer,
/// updating the `socklen_t` at `addrlen_ptr`. A no-op when `addrlen_ptr` is
/// null. Always returns success (0).
fn write_sockaddr(
    mem: &mut GuestMemory,
    addr: u64,
    addrlen_ptr: u64,
    domain: u16,
    resolved: Option<&Addr>,
) -> i64 {
    if addrlen_ptr == 0 {
        return 0;
    }
    let buf = match resolved {
        Some(Addr::Unix(name)) => {
            let mut b = domain.to_le_bytes().to_vec();
            b.extend_from_slice(name.as_bytes());
            b.push(0);
            b
        }
        Some(Addr::Inet(a)) => encode_inet_sockaddr(*a),
        None => match domain {
            AF_INET => encode_inet_sockaddr(InetAddr { v6: false, port: 0, ip: [0; 16] }),
            AF_INET6 => encode_inet_sockaddr(InetAddr { v6: true, port: 0, ip: [0; 16] }),
            _ => domain.to_le_bytes().to_vec(),
        },
    };
    let cap = mem.read_u32(addrlen_ptr).unwrap_or(0) as usize;
    if addr != 0 {
        let n = buf.len().min(cap);
        let _ = mem.write(addr, &buf[..n]);
    }
    let _ = mem.write(addrlen_ptr, &(buf.len() as u32).to_le_bytes());
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::Arch;
    use crate::abi::arch::Sysno;
    use crate::fs::{MountTable, TmpFs};
    use crate::vcpu::mem::Prot;
    use crate::vcpu::{Exit, Vcpu, VcpuError};

    /// A no-op vcpu, matching the one in the `kernel` module's tests.
    #[derive(Clone)]
    struct DummyVcpu;
    impl Vcpu for DummyVcpu {
        fn run(&mut self, _m: &mut GuestMemory) -> Result<Exit, VcpuError> {
            Ok(Exit::Halt)
        }
        fn syscall_nr(&self) -> u64 {
            0
        }
        fn syscall_args(&self) -> [u64; 6] {
            [0; 6]
        }
        fn set_syscall_ret(&mut self, _v: u64) {}
        fn reg(&self, _i: usize) -> u64 {
            0
        }
        fn set_reg(&mut self, _i: usize, _v: u64) {}
        fn pc(&self) -> u64 {
            0
        }
        fn set_pc(&mut self, _v: u64) {}
        fn sp(&self) -> u64 {
            0
        }
        fn set_sp(&mut self, _v: u64) {}
        fn set_tls(&mut self, _v: u64) {}
        fn fork(&self) -> Box<dyn Vcpu> {
            Box::new(self.clone())
        }
        fn reset(&mut self, _e: u64, _s: u64) {}
    }

    const PAGE: u64 = 4096;

    fn setup() -> (Kernel, GuestMemory, DummyVcpu) {
        let mut mounts = MountTable::new();
        mounts.mount("/", Box::new(TmpFs::new()));
        let mut kernel = Kernel::new(Arch::Aarch64, mounts);
        kernel.cur.pid = 1;
        let mut mem = GuestMemory::new(0x1_0000, 16 * PAGE);
        mem.map(0x1_0000, 4 * PAGE, Prot::rw()).unwrap();
        (kernel, mem, DummyVcpu)
    }

    fn call(k: &mut Kernel, mem: &mut GuestMemory, v: &mut DummyVcpu, s: Sysno, a: [u64; 6]) -> i64 {
        k.dispatch(s, 0, &a, v, mem)
    }

    /// Write a `struct sockaddr_in` (AF_INET) at `ptr`.
    fn write_sockaddr_in(mem: &mut GuestMemory, ptr: u64, ip: [u8; 4], port: u16) {
        let mut b = [0u8; 16];
        b[0..2].copy_from_slice(&2u16.to_le_bytes());
        b[2..4].copy_from_slice(&port.to_be_bytes());
        b[4..8].copy_from_slice(&ip);
        mem.write_init(ptr, &b).unwrap();
    }

    /// Write a `struct sockaddr_in6` (AF_INET6) at `ptr`.
    fn write_sockaddr_in6(mem: &mut GuestMemory, ptr: u64, ip: [u8; 16], port: u16) {
        let mut b = [0u8; 28];
        b[0..2].copy_from_slice(&10u16.to_le_bytes());
        b[2..4].copy_from_slice(&port.to_be_bytes());
        b[8..24].copy_from_slice(&ip);
        mem.write_init(ptr, &b).unwrap();
    }

    /// The big-endian port field out of a `sockaddr_in`/`sockaddr_in6`.
    fn read_port(mem: &GuestMemory, ptr: u64) -> u16 {
        let b = mem.read_vec(ptr, 4).unwrap();
        u16::from_be_bytes([b[2], b[3]])
    }

    #[test]
    fn socketpair_roundtrip() {
        let (mut k, mut mem, mut v) = setup();
        let sv = 0x1_0000;
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Socketpair, [1, 1, 0, sv, 0, 0]), 0);
        let a = u64::from(mem.read_u32(sv).unwrap());
        let b = u64::from(mem.read_u32(sv + 4).unwrap());
        assert!(a >= 3 && b >= 3 && a != b);

        let msg = 0x1_1000;
        let out = 0x1_2000;
        mem.write_init(msg, b"hi").unwrap();
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Write, [a, msg, 2, 0, 0, 0]), 2);
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Read, [b, out, 2, 0, 0, 0]), 2);
        assert_eq!(mem.read_vec(out, 2).unwrap(), b"hi");

        // The other direction is empty with the peer still open -> blocks.
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Read, [a, out, 2, 0, 0, 0]), 0);
        assert!(k.block);
    }

    #[test]
    fn bind_listen_connect_accept_bidirectional() {
        let (mut k, mut mem, mut v) = setup();
        let addr = 0x1_1000;
        mem.write_init(addr, &1u16.to_le_bytes()).unwrap(); // AF_UNIX
        mem.write_init(addr + 2, b"/s\0").unwrap();
        let alen = 5u64;

        let srv = call(&mut k, &mut mem, &mut v, Sysno::Socket, [1, 1, 0, 0, 0, 0]) as u64;
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Bind, [srv, addr, alen, 0, 0, 0]), 0);
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Listen, [srv, 8, 0, 0, 0, 0]), 0);

        let cli = call(&mut k, &mut mem, &mut v, Sysno::Socket, [1, 1, 0, 0, 0, 0]) as u64;
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Connect, [cli, addr, alen, 0, 0, 0]), 0);
        let acc = call(&mut k, &mut mem, &mut v, Sysno::Accept4, [srv, 0, 0, 0, 0, 0]);
        assert!(acc >= 3, "accept returned a fd");
        let acc = acc as u64;

        let msg = 0x1_2000;
        let out = 0x1_3000;
        // client -> server
        mem.write_init(msg, b"ping").unwrap();
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Write, [cli, msg, 4, 0, 0, 0]), 4);
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Read, [acc, out, 4, 0, 0, 0]), 4);
        assert_eq!(mem.read_vec(out, 4).unwrap(), b"ping");
        // server -> client
        mem.write_init(msg, b"pong").unwrap();
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Write, [acc, msg, 4, 0, 0, 0]), 4);
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Read, [cli, out, 4, 0, 0, 0]), 4);
        assert_eq!(mem.read_vec(out, 4).unwrap(), b"pong");
    }

    #[test]
    fn connect_without_listener_is_refused() {
        let (mut k, mut mem, mut v) = setup();
        let addr = 0x1_1000;
        mem.write_init(addr, &1u16.to_le_bytes()).unwrap();
        mem.write_init(addr + 2, b"/nope\0").unwrap();
        let cli = call(&mut k, &mut mem, &mut v, Sysno::Socket, [1, 1, 0, 0, 0, 0]) as u64;
        let ret = call(&mut k, &mut mem, &mut v, Sysno::Connect, [cli, addr, 8, 0, 0, 0]);
        assert_eq!(ret, -i64::from(Errno::ECONNREFUSED.0));
    }

    #[test]
    fn write_to_socket_with_closed_peer_is_epipe() {
        let (mut k, mut mem, mut v) = setup();
        let sv = 0x1_0000;
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Socketpair, [1, 1, 0, sv, 0, 0]), 0);
        let end0 = u64::from(mem.read_u32(sv).unwrap());
        let end1 = u64::from(mem.read_u32(sv + 4).unwrap());

        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Close, [end1, 0, 0, 0, 0, 0]), 0);
        let msg = 0x1_1000;
        mem.write_init(msg, b"x").unwrap();
        let ret = call(&mut k, &mut mem, &mut v, Sysno::Write, [end0, msg, 1, 0, 0, 0]);
        assert_eq!(ret, -i64::from(Errno::EPIPE.0));
    }

    #[test]
    fn fstat_reports_socket_type() {
        let (mut k, mut mem, mut v) = setup();
        let sv = 0x1_0000;
        call(&mut k, &mut mem, &mut v, Sysno::Socketpair, [1, 1, 0, sv, 0, 0]);
        let a = u64::from(mem.read_u32(sv).unwrap());
        let st = 0x1_2000;
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Fstat, [a, st, 0, 0, 0, 0]), 0);
        let mode = mem.read_u32(st + 16).unwrap();
        assert_eq!(mode & 0o170_000, 0o140_000, "S_IFSOCK");
    }

    #[test]
    fn tcp_inet4_loopback_roundtrip() {
        let (mut k, mut mem, mut v) = setup();
        let addr = 0x1_1000;
        write_sockaddr_in(&mut mem, addr, [127, 0, 0, 1], 9000);
        let alen = 16u64;

        let srv = call(&mut k, &mut mem, &mut v, Sysno::Socket, [2, 1, 0, 0, 0, 0]) as u64;
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Bind, [srv, addr, alen, 0, 0, 0]), 0);
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Listen, [srv, 8, 0, 0, 0, 0]), 0);

        let cli = call(&mut k, &mut mem, &mut v, Sysno::Socket, [2, 1, 0, 0, 0, 0]) as u64;
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Connect, [cli, addr, alen, 0, 0, 0]), 0);
        let acc = call(&mut k, &mut mem, &mut v, Sysno::Accept4, [srv, 0, 0, 0, 0, 0]);
        assert!(acc >= 3, "accept returned a fd");
        let acc = acc as u64;

        let msg = 0x1_1200;
        let out = 0x1_1300;
        mem.write_init(msg, b"ping").unwrap();
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Write, [cli, msg, 4, 0, 0, 0]), 4);
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Read, [acc, out, 4, 0, 0, 0]), 4);
        assert_eq!(mem.read_vec(out, 4).unwrap(), b"ping");
        mem.write_init(msg, b"pong").unwrap();
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Write, [acc, msg, 4, 0, 0, 0]), 4);
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Read, [cli, out, 4, 0, 0, 0]), 4);
        assert_eq!(mem.read_vec(out, 4).unwrap(), b"pong");

        // getpeername on the client reports the server's bound port; getsockname
        // on the accepted end reports the same port back.
        let peer = 0x1_1400;
        let peerlen = 0x1_1500;
        mem.write_init(peerlen, &16u32.to_le_bytes()).unwrap();
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Getpeername, [cli, peer, peerlen, 0, 0, 0]), 0);
        assert_eq!(read_port(&mem, peer), 9000);
        assert_eq!(mem.read_vec(peer, 8).unwrap()[4..8], [127, 0, 0, 1]);

        mem.write_init(peerlen, &16u32.to_le_bytes()).unwrap();
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Getsockname, [acc, peer, peerlen, 0, 0, 0]), 0);
        assert_eq!(read_port(&mem, peer), 9000);
    }

    #[test]
    fn tcp_inet6_loopback_roundtrip() {
        let (mut k, mut mem, mut v) = setup();
        let addr = 0x1_1000;
        let mut ip = [0u8; 16];
        ip[15] = 1; // ::1
        write_sockaddr_in6(&mut mem, addr, ip, 9700);
        let alen = 28u64;

        let srv = call(&mut k, &mut mem, &mut v, Sysno::Socket, [10, 1, 0, 0, 0, 0]) as u64;
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Bind, [srv, addr, alen, 0, 0, 0]), 0);
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Listen, [srv, 8, 0, 0, 0, 0]), 0);

        let cli = call(&mut k, &mut mem, &mut v, Sysno::Socket, [10, 1, 0, 0, 0, 0]) as u64;
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Connect, [cli, addr, alen, 0, 0, 0]), 0);
        let acc = call(&mut k, &mut mem, &mut v, Sysno::Accept4, [srv, 0, 0, 0, 0, 0]);
        assert!(acc >= 3, "accept returned a fd");
        let acc = acc as u64;

        let msg = 0x1_1200;
        let out = 0x1_1300;
        mem.write_init(msg, b"v6ok").unwrap();
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Write, [cli, msg, 4, 0, 0, 0]), 4);
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Read, [acc, out, 4, 0, 0, 0]), 4);
        assert_eq!(mem.read_vec(out, 4).unwrap(), b"v6ok");
    }

    #[test]
    fn ephemeral_port_via_getsockname() {
        let (mut k, mut mem, mut v) = setup();
        let addr = 0x1_1000;
        write_sockaddr_in(&mut mem, addr, [127, 0, 0, 1], 0); // port 0: auto-assign
        let s = call(&mut k, &mut mem, &mut v, Sysno::Socket, [2, 1, 0, 0, 0, 0]) as u64;
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Bind, [s, addr, 16, 0, 0, 0]), 0);

        let name = 0x1_1200;
        let namelen = 0x1_1300;
        mem.write_init(namelen, &16u32.to_le_bytes()).unwrap();
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Getsockname, [s, name, namelen, 0, 0, 0]), 0);
        assert!(read_port(&mem, name) >= 32_768);
    }

    #[test]
    fn udp_connected_roundtrip_via_dispatch() {
        let (mut k, mut mem, mut v) = setup();
        let a_addr = 0x1_1000;
        write_sockaddr_in(&mut mem, a_addr, [127, 0, 0, 1], 9300);
        let b_addr = 0x1_1100;
        write_sockaddr_in(&mut mem, b_addr, [127, 0, 0, 1], 9400);

        let a = call(&mut k, &mut mem, &mut v, Sysno::Socket, [2, 2, 0, 0, 0, 0]) as u64;
        let b = call(&mut k, &mut mem, &mut v, Sysno::Socket, [2, 2, 0, 0, 0, 0]) as u64;
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Bind, [a, a_addr, 16, 0, 0, 0]), 0);
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Bind, [b, b_addr, 16, 0, 0, 0]), 0);
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Connect, [a, b_addr, 16, 0, 0, 0]), 0);
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Connect, [b, a_addr, 16, 0, 0, 0]), 0);

        let msg = 0x1_1200;
        let out = 0x1_1300;
        mem.write_init(msg, b"hi").unwrap();
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Write, [a, msg, 2, 0, 0, 0]), 2);
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Read, [b, out, 2, 0, 0, 0]), 2);
        assert_eq!(mem.read_vec(out, 2).unwrap(), b"hi");
    }

    #[test]
    fn udp_sendto_recvfrom_with_source_addr() {
        let (mut k, mut mem, _v) = setup();
        let a_addr = 0x1_1000;
        write_sockaddr_in(&mut mem, a_addr, [127, 0, 0, 1], 9100);
        let b_addr = 0x1_1100;
        write_sockaddr_in(&mut mem, b_addr, [127, 0, 0, 1], 9200);

        let a = k.sys_socket(2, 2, 0) as u64; // AF_INET, SOCK_DGRAM
        let b = k.sys_socket(2, 2, 0) as u64;
        assert_eq!(k.sys_bind(a, a_addr, 16, &mem), 0);
        assert_eq!(k.sys_bind(b, b_addr, 16, &mem), 0);

        let msg = 0x1_1200;
        mem.write_init(msg, b"hello").unwrap();
        assert_eq!(k.sys_sendto(a, msg, 5, 0, b_addr, 16, &mem), 5);

        let out = 0x1_1300;
        let src = 0x1_1400;
        let srclen = 0x1_1500;
        mem.write_init(srclen, &16u32.to_le_bytes()).unwrap();
        assert_eq!(k.sys_recvfrom(b, out, 5, 0, src, srclen, &mut mem), 5);
        assert_eq!(mem.read_vec(out, 5).unwrap(), b"hello");
        assert_eq!(read_port(&mem, src), 9100); // source is A's bound port
        assert_eq!(mem.read_vec(src, 8).unwrap()[4..8], [127, 0, 0, 1]);
    }

    #[test]
    fn setsockopt_reuseaddr_allows_rebind() {
        let (mut k, mut mem, _v) = setup();
        let addr = 0x1_1000;
        write_sockaddr_in(&mut mem, addr, [127, 0, 0, 1], 9500);

        let a = k.sys_socket(2, 1, 0) as u64;
        assert_eq!(k.sys_bind(a, addr, 16, &mem), 0);

        let b = k.sys_socket(2, 1, 0) as u64;
        // Without SO_REUSEADDR, binding the same port fails.
        assert_eq!(k.sys_bind(b, addr, 16, &mem), -i64::from(Errno::EINVAL.0));

        // Setting SO_REUSEADDR=1 on b lets the rebind through.
        let optval = 0x1_1600;
        mem.write_init(optval, &1u32.to_le_bytes()).unwrap();
        assert_eq!(k.sys_setsockopt(b, SOL_SOCKET, SO_REUSEADDR, optval, 4, &mem), 0);
        assert_eq!(k.sys_bind(b, addr, 16, &mem), 0);
    }

    #[test]
    fn accept4_nonblocking_returns_eagain() {
        let (mut k, mut mem, mut v) = setup();
        let addr = 0x1_1000;
        write_sockaddr_in(&mut mem, addr, [127, 0, 0, 1], 9600);
        let srv = call(&mut k, &mut mem, &mut v, Sysno::Socket, [2, 1 | SOCK_NONBLOCK, 0, 0, 0, 0]) as u64;
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Bind, [srv, addr, 16, 0, 0, 0]), 0);
        assert_eq!(call(&mut k, &mut mem, &mut v, Sysno::Listen, [srv, 8, 0, 0, 0, 0]), 0);
        let ret = call(&mut k, &mut mem, &mut v, Sysno::Accept4, [srv, 0, 0, 0, 0, 0]);
        assert_eq!(ret, -i64::from(Errno::EAGAIN.0));
        assert!(!k.block);
    }
}
