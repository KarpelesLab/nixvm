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

use super::egress::{Egress, HostConn, HostDgram};
use super::{Fd, Kernel, ServiceCtx, err};

impl Net {
    pub(super) fn set_egress(&mut self, egress: Box<dyn Egress>) {
        self.egress = Some(egress);
    }

    /// Set the `O_NONBLOCK` state of socket `sock` (end `end`) — the `fcntl`
    /// `F_SETFL` path. Updates the socket-level flag plus the per-end flag of a
    /// connected pair and a bridged host connection, so every `recv`/`send`
    /// path (dgram, pair, host) observes it. Silently ignores a stale index.
    pub(super) fn set_nonblock(&mut self, sock: usize, end: usize, nb: bool) {
        if let Some(s) = self.socks.get_mut(sock) {
            s.nonblock = nb;
            match &mut s.kind {
                Kind::Pair(p) => p.nonblock[end.min(1)] = nb,
                Kind::Host(h) => h.nonblock = nb,
                _ => {}
            }
        }
    }

    /// Current `O_NONBLOCK` state of socket `sock` (end `end`) — the `fcntl`
    /// `F_GETFL` path — reading the same per-kind flag the I/O paths consult.
    pub(super) fn is_nonblock(&self, sock: usize, end: usize) -> bool {
        self.socks.get(sock).is_some_and(|s| match &s.kind {
            Kind::Pair(p) => p.nonblock[end.min(1)],
            Kind::Host(h) => h.nonblock,
            _ => s.nonblock,
        })
    }

    /// Whether any live host-bridged connection exists: a TCP egress socket, or
    /// a UDP socket with a host peer. While one is open the guest may be blocked
    /// awaiting network data that arrives asynchronously, so an otherwise all-
    /// parked machine is not a deadlock — the scheduler polls for host readiness
    /// rather than erroring out (there is no host-side wakeup into the
    /// cooperative loop, so it must look for itself).
    pub(super) fn has_pending_host_io(&self) -> bool {
        self.socks.iter().any(|s| {
            matches!(
                &s.kind,
                Kind::Host(_) | Kind::Dgram(Dgram { host: Some(_), .. })
            )
        })
    }
}

impl Errno {
    /// Not (yet) in [`crate::abi::errno::Errno`]'s generic subset — only this
    /// module needs it so far (`getpeername`/`shutdown` on an unconnected
    /// socket). A second `impl Errno` block is legal: inherent impls may
    /// split across modules within the same crate.
    const ENOTCONN: Errno = Errno(107);
    /// `socket(AF_NETLINK, _, protocol)` with an unsupported `protocol`.
    const EPROTONOSUPPORT: Errno = Errno(93);
    /// A routable connect with no egress backend installed (loopback-only VM).
    const ENETUNREACH: Errno = Errno(101);
    /// A host `connect_tcp` that timed out.
    const ETIMEDOUT: Errno = Errno(110);
    /// A host socket read/write errored (connection reset).
    const ECONNRESET: Errno = Errno(104);
}

const AF_UNIX: u16 = 1;
const AF_INET: u16 = 2;
const AF_INET6: u16 = 10;
const AF_NETLINK: u16 = 16;
const SOCK_STREAM: u64 = 1;
const SOCK_DGRAM: u64 = 2;
const SOCK_RAW: u64 = 3;
const SOCK_NONBLOCK: u64 = 0o4000;
const NETLINK_ROUTE: u64 = 0;

const SOL_SOCKET: u64 = 1;
const IPPROTO_IP: u64 = 0;
const IPPROTO_TCP: u64 = 6;
const IPPROTO_IPV6: u64 = 41;

// `SOL_SOCKET` option names (asm-generic/socket.h).
const SO_REUSEADDR: u64 = 2;
const SO_TYPE: u64 = 3;
const SO_ERROR: u64 = 4;
const SO_BROADCAST: u64 = 6;
const SO_SNDBUF: u64 = 7;
const SO_RCVBUF: u64 = 8;
const SO_KEEPALIVE: u64 = 9;
const SO_LINGER: u64 = 13;
const SO_REUSEPORT: u64 = 15;
const SO_RCVTIMEO: u64 = 20;
const SO_SNDTIMEO: u64 = 21;
const SO_ACCEPTCONN: u64 = 30;
const SO_PROTOCOL: u64 = 38;
const SO_DOMAIN: u64 = 39;

// Protocol-level option names.
const IP_TOS: u64 = 1;
const TCP_NODELAY: u64 = 1;
const IPV6_V6ONLY: u64 = 26;

// `sendto`/`recvfrom` `flags` bits this module honors (linux/socket.h).
const MSG_PEEK: u64 = 0x02;
const MSG_TRUNC: u64 = 0x20;
const MSG_DONTWAIT: u64 = 0x40;
const MSG_WAITALL: u64 = 0x100;
// MSG_NOSIGNAL (0x4000) is a documented no-op: this virtual transport never
// raises SIGPIPE on a peer-closed write in the first place (it returns
// `EPIPE`), so there is nothing to suppress.

// `NETLINK_ROUTE` (rtnetlink) message types this module answers
// (linux/rtnetlink.h). Only enough of the protocol to let guest tools
// enumerate the always-up loopback interface: `RTM_GETLINK`/`RTM_GETADDR`
// dumps, and a minimal `RTM_GETROUTE` reply.
const RTM_NEWLINK: u16 = 16;
const RTM_GETLINK: u16 = 18;
const RTM_NEWADDR: u16 = 20;
const RTM_GETADDR: u16 = 22;
const RTM_GETROUTE: u16 = 26;
/// Generic netlink control messages (linux/netlink.h): an error/ACK, and the
/// end-of-dump marker.
const NLMSG_ERROR: u16 = 2;
const NLMSG_DONE: u16 = 3;

// `nlmsghdr.nlmsg_flags` bits this module inspects (linux/netlink.h).
const NLM_F_ACK: u16 = 0x04;
/// `NLM_F_ROOT | NLM_F_MATCH` — "dump the whole table", the flag pair
/// `ip`/`ifconfig`'s `RTM_GETLINK`/`RTM_GETADDR` requests always set.
const NLM_F_DUMP: u16 = 0x100 | 0x200;

/// `ifinfomsg.ifi_type` for the loopback device (linux/if_arp.h).
const ARPHRD_LOOPBACK: u16 = 772;
// `ifinfomsg.ifi_flags` / `IFF_*` bits (linux/if.h) set on `lo`.
const IFF_UP: u32 = 0x1;
const IFF_LOOPBACK: u32 = 0x8;
const IFF_RUNNING: u32 = 0x40;

// `IFLA_*` rtattr types (linux/if_link.h) filled in on the `RTM_NEWLINK` reply.
const IFLA_ADDRESS: u16 = 1;
const IFLA_IFNAME: u16 = 3;
const IFLA_MTU: u16 = 4;

// `IFA_*` rtattr types (linux/if_addr.h) filled in on the `RTM_NEWADDR` reply.
const IFA_ADDRESS: u16 = 1;
const IFA_LOCAL: u16 = 2;
const IFA_LABEL: u16 = 3;
/// `RT_SCOPE_HOST` (linux/rtnetlink.h): the scope of an address that is only
/// valid on this host, e.g. `127.0.0.1`.
const RT_SCOPE_HOST: u8 = 254;
/// The loopback interface's fixed `ifindex`: this module models exactly one
/// interface, so it never needs to be anything but `1`.
const LOOPBACK_IFINDEX: i32 = 1;

/// The kernel's socket table plus the AF_UNIX/AF_INET(6) address registries.
#[derive(Default)]
pub(super) struct Net {
    socks: Vec<Sock>,
    /// AF_UNIX path, or `"tcp4:port"`/`"tcp6:port"` -> listening slot.
    listeners: BTreeMap<String, usize>,
    /// `"udp4:port"`/`"udp6:port"` -> bound `Dgram` slot.
    dgram_ports: BTreeMap<String, usize>,
    /// Host-egress backend. `None` = loopback-only (the default, and the only
    /// possible mode with no backend installed); `Some` bridges guest
    /// connections to *routable* addresses onto real host sockets. Installed
    /// by [`Kernel::set_egress`].
    egress: Option<Box<dyn Egress>>,
}

impl std::fmt::Debug for Net {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Net")
            .field("socks", &self.socks)
            .field("listeners", &self.listeners)
            .field("dgram_ports", &self.dgram_ports)
            .field("egress", &self.egress.is_some())
            .finish()
    }
}

/// One entry in the socket table.
#[derive(Debug)]
struct Sock {
    domain: u16,
    kind: Kind,
    /// `SOCK_NONBLOCK` at creation time (there is no `fcntl(F_SETFL)` wiring
    /// into this module, so this is fixed at `socket()`/`accept4()`).
    nonblock: bool,
    /// `setsockopt`-controlled knobs, `SO_ERROR`, and buffer-size hints.
    opts: SockOpts,
}

/// The mutable `setsockopt`-controlled state of a socket, plus `SO_ERROR`.
/// Only `reuseaddr` has an observable effect on this virtual loopback (it
/// relaxes `bind`'s `EADDRINUSE` check); the rest are stored and echoed back
/// by `getsockopt` for compatibility with guest code that sets/reads them,
/// since there is no real transport here to actually tune. The many
/// independent `bool` flags genuinely are independent `setsockopt` knobs
/// (not encodable as a smaller state machine), hence the blanket allow.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug)]
struct SockOpts {
    reuseaddr: bool,
    reuseport: bool,
    keepalive: bool,
    broadcast: bool,
    /// `TCP_NODELAY` — stored only; every write is already delivered
    /// immediately, so Nagle's algorithm was never modeled to begin with.
    nodelay: bool,
    /// `IPV6_V6ONLY` — stored only; this module never models a dual-stack
    /// socket (an `AF_INET6` bind/connect always requires a v6 peer/local
    /// address), so there is no dual-stack behavior to gate.
    v6only: bool,
    linger_on: bool,
    linger_secs: i32,
    /// Raw `struct timeval` bytes (`tv_sec`, `tv_usec`), stored and echoed
    /// back verbatim; nothing in this module actually times out.
    rcvtimeo: [u8; 16],
    sndtimeo: [u8; 16],
    rcvbuf: u32,
    sndbuf: u32,
    tos: u32,
    /// `SO_ERROR`: read-and-cleared by `getsockopt`. Nothing in this
    /// synchronous virtual transport currently produces an asynchronous
    /// pending error, so this is always `0` today; the storage/clear
    /// machinery is here so `getsockopt(SO_ERROR)` behaves correctly if a
    /// future path (e.g. a failed background connect) ever sets it.
    error: i32,
}

impl Default for SockOpts {
    fn default() -> Self {
        Self {
            reuseaddr: false,
            reuseport: false,
            keepalive: false,
            broadcast: false,
            nodelay: false,
            v6only: false,
            linger_on: false,
            linger_secs: 0,
            rcvtimeo: [0; 16],
            sndtimeo: [0; 16],
            rcvbuf: 212_992,
            sndbuf: 212_992,
            tos: 0,
            error: 0,
        }
    }
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
    /// An `AF_NETLINK`/`NETLINK_ROUTE` endpoint.
    Netlink(Netlink),
    /// A stream socket bridged onto a real host connection (egress): the guest
    /// `connect`ed to a routable address and the host-egress backend was
    /// installed. Reads/writes go straight to [`HostConn`].
    Host(HostSock),
}

/// A guest stream socket bridged onto a real host connection.
struct HostSock {
    conn: Box<dyn HostConn>,
    /// The routable peer the guest connected to (for `getpeername`).
    peer: InetAddr,
    /// The guest's `shutdown(SHUT_WR)` has been forwarded to the host.
    wr_shut: bool,
    nonblock: bool,
}

// Only `peer` is meaningful in a dump; the boxed conn and flags are noise.
#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for HostSock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostSock").field("peer", &self.peer).finish()
    }
}

/// A connected socket: `to[e]` holds bytes destined for end `e` (so end `e`
/// reads `to[e]` and writes `to[1 - e]`). `refs[e]` counts open fds on end `e`.
/// `shut_wr[e]`/`shut_rd[e]` record a `shutdown(SHUT_WR)`/`shutdown(SHUT_RD)`
/// from end `e`: a shut write side makes further writes from `e` fail with
/// `EPIPE` and makes the peer's reads see EOF once `to[1-e]` drains; a shut
/// read side makes further reads from `e` return EOF (`0`) immediately,
/// regardless of what's still queued. `nonblock[e]` mirrors the
/// `O_NONBLOCK`/`SOCK_NONBLOCK` of the fd currently on end `e`. `addrs[e]` is
/// end `e`'s local `AF_INET`/`AF_INET6` address (so its peer address is
/// `addrs[1 - e]`); `None` for AF_UNIX pairs.
#[derive(Debug)]
struct Pair {
    to: [VecDeque<u8>; 2],
    refs: [usize; 2],
    shut_wr: [bool; 2],
    shut_rd: [bool; 2],
    nonblock: [bool; 2],
    addrs: [Option<InetAddr>; 2],
}

impl Pair {
    /// A freshly connected pair with one open reference on each end.
    fn new() -> Self {
        Self {
            to: [VecDeque::new(), VecDeque::new()],
            refs: [1, 1],
            shut_wr: [false, false],
            shut_rd: [false, false],
            nonblock: [false, false],
            addrs: [None, None],
        }
    }
}

/// A `SOCK_DGRAM` endpoint: `local` is the bound (or lazily ephemeral-assigned)
/// address, `peer` is the `connect()`-ed destination (if any), and `queue`
/// holds inbound `(source address, payload)` datagrams awaiting `recv`.
#[derive(Default)]
struct Dgram {
    local: Option<InetAddr>,
    peer: Option<InetAddr>,
    queue: VecDeque<(InetAddr, Vec<u8>)>,
    /// A host UDP socket, opened lazily the first time this datagram socket
    /// sends to a *routable* address with egress enabled. Once present, all
    /// sends go out through it and inbound host datagrams are drained into
    /// `queue` on recv. `None` = pure in-VM datagram socket.
    host: Option<Box<dyn HostDgram>>,
}

impl std::fmt::Debug for Dgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dgram")
            .field("local", &self.local)
            .field("peer", &self.peer)
            .field("queue", &self.queue)
            .field("host", &self.host.is_some())
            .finish()
    }
}

/// An `AF_NETLINK` endpoint (only `NETLINK_ROUTE` is modeled). There is no
/// "connection": a guest `send`/`sendto`/`write`s a request (an `nlmsghdr`
/// stream, one or more messages back to back) which is answered synchronously
/// by enqueuing the reply message(s) onto `queue`, in order, for a later
/// `recv`/`recvfrom`/`read` to drain — mirroring how a real `rtnetlink` dump
/// reply is a sequence of `nlmsghdr`s terminated by `NLMSG_DONE`, delivered
/// across one or more `recvmsg` calls depending on the caller's buffer size.
#[derive(Debug, Default)]
struct Netlink {
    /// This socket's `nl_pid`, assigned by `bind` (or lazily on first use);
    /// echoed back in `getsockname` and as every reply's `nlmsg_pid`.
    pid: u32,
    /// The `SOCK_RAW`/`SOCK_DGRAM` requested at `socket()` time, echoed back
    /// by `getsockopt(SO_TYPE)`.
    sotype: u64,
    /// Fully encoded (`nlmsghdr`-framed, `NLMSG_ALIGN`-ed) response messages
    /// awaiting `recv`; each entry is one complete message.
    queue: VecDeque<Vec<u8>>,
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
        if self.v6 {
            self.ip == loopback_ip(true)
        } else {
            self.ip[0] == 127
        }
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
                Kind::Idle {
                    bound: Some(Addr::Inet(b)),
                } if proto == "tcp" => route_key("tcp", *b) == key,
                Kind::Listener {
                    addr: Some(Addr::Inet(b)),
                    ..
                } if proto == "tcp" => route_key("tcp", *b) == key,
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
            let a = InetAddr {
                v6,
                port: port as u16,
                ip: [0; 16],
            };
            if !self.addr_in_use(proto, a, usize::MAX) {
                return port as u16;
            }
        }
        0
    }

    /// A fresh client-side local address: loopback of `v6`'s family, with a
    /// freshly allocated ephemeral TCP port.
    fn fresh_local(&self, v6: bool) -> InetAddr {
        InetAddr {
            v6,
            port: self.ephemeral_port("tcp", v6),
            ip: loopback_ip(v6),
        }
    }
}

impl Kernel {
    /// The `(slot, end)` a socket fd points at, if it is a socket.
    #[allow(clippy::unused_self)]
    fn sock_of(&self, cx: &ServiceCtx, fd: u64) -> Option<(usize, usize)> {
        match cx.cur.fds.get(fd as i32) {
            Some(Fd::Socket { sock, end }) => Some((*sock, *end)),
            _ => None,
        }
    }

    /// Ensure `SOCK_DGRAM` socket `sock` has a local address, lazily
    /// auto-assigning (and registering in the port table) an ephemeral
    /// loopback one if it was never `bind`-ed — mirroring the implicit local
    /// address Linux assigns to an unbound socket on first send.
    #[allow(clippy::unused_self)]
    fn ensure_dgram_bound(&self, net: &mut Net, sock: usize) -> InetAddr {
        if let Kind::Dgram(d) = &net.socks[sock].kind
            && let Some(local) = d.local
        {
            return local;
        }
        let v6 = net.socks[sock].domain == AF_INET6;
        let port = net.ephemeral_port("udp", v6);
        let local = InetAddr {
            v6,
            port,
            ip: loopback_ip(v6),
        };
        if let Kind::Dgram(d) = &mut net.socks[sock].kind {
            d.local = Some(local);
        }
        net.dgram_ports.insert(route_key("udp", local), sock);
        local
    }

    /// `socket(domain, type, protocol)` — an unbound, unconnected endpoint.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_socket(&self, net: &mut Net, cx: &mut ServiceCtx, domain: u64, sotype: u64, protocol: u64) -> i64 {
        let domain = domain as u16;
        if domain == AF_NETLINK {
            let base_type = sotype & 0xf;
            if base_type != SOCK_RAW && base_type != SOCK_DGRAM {
                return err(Errno::EOPNOTSUPP);
            }
            if protocol != NETLINK_ROUTE {
                return err(Errno::EPROTONOSUPPORT);
            }
            let nonblock = sotype & SOCK_NONBLOCK != 0;
            let kind = Kind::Netlink(Netlink {
                sotype: base_type,
                ..Netlink::default()
            });
            let idx = net.socks.len();
            net.socks.push(Sock {
                domain,
                kind,
                nonblock,
                opts: SockOpts::default(),
            });
            return i64::from(cx.cur.fds.alloc(Fd::Socket { sock: idx, end: 0 }));
        }
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
        let idx = net.socks.len();
        net.socks.push(Sock {
            domain,
            kind,
            nonblock,
            opts: SockOpts::default(),
        });
        i64::from(cx.cur.fds.alloc(Fd::Socket { sock: idx, end: 0 }))
    }

    /// `socketpair(domain, type, protocol, sv)` — a connected AF_UNIX pair whose
    /// two fds are written to `sv[0]`/`sv[1]`.
    #[allow(clippy::too_many_arguments, clippy::unused_self)]
    pub(super) fn sys_socketpair(
        &self, net: &mut Net, cx: &mut ServiceCtx,
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
        let idx = net.socks.len();
        net.socks.push(Sock {
            domain: AF_UNIX,
            kind: Kind::Pair(pair),
            nonblock,
            opts: SockOpts::default(),
        });
        let fd0 = cx.cur.fds.alloc(Fd::Socket { sock: idx, end: 0 });
        let fd1 = cx.cur.fds.alloc(Fd::Socket { sock: idx, end: 1 });
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
    pub(super) fn sys_bind(&self, net: &mut Net, cx: &mut ServiceCtx, fd: u64, addr: u64, addrlen: u64, mem: &GuestMemory) -> i64 {
        let Some((sock, _)) = self.sock_of(cx, fd) else {
            return err(Errno::ENOTSOCK);
        };
        if net.socks[sock].domain == AF_NETLINK {
            return self.bind_netlink(net, sock, addr, addrlen, mem);
        }
        let Some(parsed) = read_sockaddr(mem, addr, addrlen) else {
            return err(Errno::EINVAL);
        };
        match parsed {
            Addr::Unix(_) => {
                if net.socks[sock].domain != AF_UNIX {
                    return err(Errno::EINVAL);
                }
                match &mut net.socks[sock].kind {
                    Kind::Idle { bound } => {
                        *bound = Some(parsed);
                        0
                    }
                    _ => err(Errno::EINVAL),
                }
            }
            Addr::Inet(mut a) => {
                let domain = net.socks[sock].domain;
                if (a.v6 && domain != AF_INET6) || (!a.v6 && domain != AF_INET) {
                    return err(Errno::EINVAL);
                }
                if !a.valid_bind() {
                    return err(Errno::EINVAL); // real errno: EADDRNOTAVAIL
                }
                let proto = match &net.socks[sock].kind {
                    Kind::Idle { .. } => "tcp",
                    Kind::Dgram(_) => "udp",
                    _ => return err(Errno::EINVAL),
                };
                if a.port == 0 {
                    a.port = net.ephemeral_port(proto, a.v6);
                } else if !net.socks[sock].opts.reuseaddr
                    && net.addr_in_use(proto, a, sock)
                {
                    return err(Errno::EINVAL); // real errno: EADDRINUSE
                }
                match &mut net.socks[sock].kind {
                    Kind::Idle { bound } => *bound = Some(Addr::Inet(a)),
                    Kind::Dgram(d) => d.local = Some(a),
                    _ => return err(Errno::EINVAL),
                }
                if proto == "udp" {
                    net.dgram_ports.insert(route_key("udp", a), sock);
                }
                0
            }
        }
    }

    /// `listen(fd, backlog)` — mark a bound socket as accepting connections.
    /// Auto-binds an ephemeral wildcard address first if `bind` was skipped
    /// (matching real Linux).
    pub(super) fn sys_listen(&self, net: &mut Net, cx: &mut ServiceCtx, fd: u64) -> i64 {
        let Some((sock, _)) = self.sock_of(cx, fd) else {
            return err(Errno::ENOTSOCK);
        };
        let domain = net.socks[sock].domain;
        let mut bound = match &net.socks[sock].kind {
            Kind::Idle { bound } => bound.clone(),
            Kind::Listener { .. } => return 0,
            _ => return err(Errno::EINVAL),
        };
        if bound.is_none() && domain != AF_UNIX {
            let v6 = domain == AF_INET6;
            let port = net.ephemeral_port("tcp", v6);
            bound = Some(Addr::Inet(InetAddr {
                v6,
                port,
                ip: [0; 16],
            }));
        }
        let key = match &bound {
            Some(Addr::Unix(p)) => Some(p.clone()),
            Some(Addr::Inet(a)) => Some(route_key("tcp", *a)),
            None => None,
        };
        net.socks[sock].kind = Kind::Listener {
            addr: bound,
            backlog: VecDeque::new(),
        };
        if let Some(k) = key {
            net.listeners.insert(k, sock);
        }
        0
    }

    /// `connect(fd, addr, addrlen)` — for a stream socket, connect an idle
    /// socket to a listener, turning this slot into the client end of a fresh
    /// pair and queuing it on the listener for `accept4`; for a datagram
    /// socket, just record the peer (no handshake, per UDP semantics).
    pub(super) fn sys_connect(
        &self, net: &mut Net, cx: &mut ServiceCtx,
        fd: u64,
        addr: u64,
        addrlen: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let Some((sock, end)) = self.sock_of(cx, fd) else {
            return err(Errno::ENOTSOCK);
        };
        let Some(target) = read_sockaddr(mem, addr, addrlen) else {
            return err(Errno::EINVAL);
        };
        let domain = net.socks[sock].domain;
        let mismatched = match (&target, domain) {
            (Addr::Unix(_), AF_UNIX) => false,
            (Addr::Inet(a), AF_INET) => a.v6,
            (Addr::Inet(a), AF_INET6) => !a.v6,
            _ => true,
        };
        if mismatched {
            return err(Errno::EINVAL);
        }
        // A routable (non-loopback) destination: with a host-egress backend
        // installed, bridge it to the real internet; without one, it's
        // unreachable (loopback-only VM).
        let routable = matches!(&target, Addr::Inet(a) if !a.valid_bind());
        if routable {
            let Addr::Inet(dest) = target else {
                unreachable!("routable implies Inet")
            };
            if matches!(net.socks[sock].kind, Kind::Dgram(_)) {
                // UDP: remember the routable peer; the host socket is opened
                // lazily on first send (see `send_bytes`/`write_socket`).
                if net.egress.is_none() {
                    return err(Errno::ENETUNREACH);
                }
                self.ensure_dgram_bound(net, sock);
                if let Kind::Dgram(d) = &mut net.socks[sock].kind {
                    d.peer = Some(dest);
                }
                return 0;
            }
            return self.connect_host(net, sock, end, dest);
        }

        if matches!(net.socks[sock].kind, Kind::Dgram(_)) {
            let Addr::Inet(peer) = target else {
                unreachable!("validated above")
            };
            self.ensure_dgram_bound(net, sock);
            if let Kind::Dgram(d) = &mut net.socks[sock].kind {
                d.peer = Some(peer);
            }
            return 0;
        }

        // An idle client end 0; anything else (already connected, a listener,
        // or the wrong end) is invalid.
        if !matches!(&net.socks[sock].kind, Kind::Idle { .. } if end == 0) {
            return err(Errno::EINVAL);
        }
        let key = match &target {
            Addr::Unix(p) => p.clone(),
            Addr::Inet(a) => route_key("tcp", *a),
        };
        let Some(&lidx) = net.listeners.get(&key) else {
            return err(Errno::ECONNREFUSED);
        };
        let listener_addr = match &net.socks[lidx].kind {
            Kind::Listener { addr, .. } => addr.clone(),
            _ => return err(Errno::ECONNREFUSED),
        };
        // Repurpose the client's idle slot as the connected pair, then queue its
        // index (the server-side end 1) on the listener's backlog.
        let mut pair = Pair::new();
        pair.nonblock[0] = net.socks[sock].nonblock;
        if domain != AF_UNIX {
            let v6 = domain == AF_INET6;
            let mut peer_addr = match listener_addr {
                Some(Addr::Inet(a)) => a,
                _ => InetAddr {
                    v6,
                    port: 0,
                    ip: loopback_ip(v6),
                },
            };
            if peer_addr.is_any() {
                // Report the concrete loopback even if the server bound ANY.
                peer_addr.ip = loopback_ip(v6);
            }
            pair.addrs[0] = Some(net.fresh_local(v6));
            pair.addrs[1] = Some(peer_addr);
        }
        net.socks[sock].kind = Kind::Pair(pair);
        if let Kind::Listener { backlog, .. } = &mut net.socks[lidx].kind {
            backlog.push_back(sock);
        }
        0
    }

    /// Bridge a stream socket onto a real host connection to routable `dest`
    /// (egress). The socket must be an idle client end. Returns `0` on a
    /// completed connection (the host `connect` is synchronous here), else a
    /// negative errno.
    #[allow(clippy::unused_self)]
    fn connect_host(&self, net: &mut Net, sock: usize, end: usize, dest: InetAddr) -> i64 {
        if !matches!(&net.socks[sock].kind, Kind::Idle { .. } if end == 0) {
            return err(Errno::EINVAL);
        }
        let Some(egress) = &net.egress else {
            return err(Errno::ENETUNREACH);
        };
        match egress.connect_tcp(dest.ip, dest.v6, dest.port) {
            Ok(conn) => {
                let nonblock = net.socks[sock].nonblock;
                net.socks[sock].kind = Kind::Host(HostSock {
                    conn,
                    peer: dest,
                    wr_shut: false,
                    nonblock,
                });
                0
            }
            // Map the common connect failures to the errnos guest clients
            // branch on; anything else is a generic "connection refused".
            Err(e) => match e.kind() {
                std::io::ErrorKind::TimedOut => err(Errno::ETIMEDOUT),
                _ => err(Errno::ECONNREFUSED),
            },
        }
    }

    /// `accept4(fd, addr, addrlen, flags)` — hand back the server-side end of a
    /// pending connection (blocking, like pipe read, when none is queued and
    /// the listening socket is not `O_NONBLOCK`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_accept4(
        &self, net: &mut Net, cx: &mut ServiceCtx,
        fd: u64,
        addr: u64,
        addrlen: u64,
        flags: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some((sock, _)) = self.sock_of(cx, fd) else {
            return err(Errno::ENOTSOCK);
        };
        let nonblock = net.socks[sock].nonblock;
        let pending = match &mut net.socks[sock].kind {
            Kind::Listener { backlog, .. } => backlog.pop_front(),
            _ => return err(Errno::EINVAL),
        };
        let Some(pidx) = pending else {
            if nonblock {
                return err(Errno::EAGAIN);
            }
            cx.block = true; // no pending connection yet — re-trap later
            return 0;
        };
        let domain = net.socks[sock].domain;
        let peer = match &net.socks[pidx].kind {
            Kind::Pair(p) => p.addrs[0].map(Addr::Inet),
            _ => None,
        };
        write_sockaddr(mem, addr, addrlen, domain, peer.as_ref());
        if let Kind::Pair(p) = &mut net.socks[pidx].kind {
            p.nonblock[1] = flags & SOCK_NONBLOCK != 0;
        }
        i64::from(cx.cur.fds.alloc(Fd::Socket { sock: pidx, end: 1 }))
    }

    /// `bind(fd, addr, addrlen)` for an `AF_NETLINK` socket: parse a
    /// `struct sockaddr_nl` (`nl_family`, 2 bytes pad, `nl_pid`, `nl_groups`;
    /// this module's guest architectures are always little-endian, so all
    /// fields are read as such) and adopt its `nl_pid` as this
    /// socket's identity, auto-assigning a nonzero one (the `1`-based socket
    /// slot index — this module never needs it to match a real process id)
    /// when the guest asks for `0` ("let the kernel pick"), exactly like a
    /// real `AF_NETLINK` autobind.
    #[allow(clippy::unused_self)]
    fn bind_netlink(&self, net: &mut Net, sock: usize, addr: u64, addrlen: u64, mem: &GuestMemory) -> i64 {
        if addrlen < 8 {
            return err(Errno::EINVAL);
        }
        let Ok(b) = mem.read_vec(addr, 8) else {
            return err(Errno::EFAULT);
        };
        if u16::from_le_bytes([b[0], b[1]]) != AF_NETLINK {
            return err(Errno::EINVAL);
        }
        let requested = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
        let pid = if requested == 0 {
            sock as u32 + 1
        } else {
            requested
        };
        let Kind::Netlink(nl) = &mut net.socks[sock].kind else {
            return err(Errno::EINVAL);
        };
        nl.pid = pid;
        0
    }

    /// `getsockname(fd, addr, addrlen)` — the local address (best-effort).
    pub(super) fn sys_getsockname(
        &self, net: &mut Net, cx: &mut ServiceCtx,
        fd: u64,
        addr: u64,
        addrlen: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some((sock, end)) = self.sock_of(cx, fd) else {
            return err(Errno::ENOTSOCK);
        };
        let domain = net.socks[sock].domain;
        if domain == AF_NETLINK {
            let pid = match &net.socks[sock].kind {
                Kind::Netlink(nl) => nl.pid,
                _ => 0,
            };
            return write_netlink_sockaddr(mem, addr, addrlen, pid);
        }
        let resolved = match &net.socks[sock].kind {
            Kind::Idle { bound } => bound.clone(),
            Kind::Listener { addr, .. } => addr.clone(),
            Kind::Pair(p) => p.addrs[end].map(Addr::Inet),
            Kind::Dgram(d) => d.local.map(Addr::Inet),
            // A host-bridged socket has an ephemeral local address the guest
            // never bound and doesn't inspect; report the wildcard.
            Kind::Host(_) | Kind::Netlink(_) => None,
        };
        write_sockaddr(mem, addr, addrlen, domain, resolved.as_ref())
    }

    /// `getpeername(fd, addr, addrlen)` — the peer address (best-effort).
    pub(super) fn sys_getpeername(
        &self, net: &mut Net, cx: &mut ServiceCtx,
        fd: u64,
        addr: u64,
        addrlen: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some((sock, end)) = self.sock_of(cx, fd) else {
            return err(Errno::ENOTSOCK);
        };
        let domain = net.socks[sock].domain;
        match &net.socks[sock].kind {
            Kind::Pair(p) => write_sockaddr(
                mem,
                addr,
                addrlen,
                domain,
                p.addrs[1 - end].map(Addr::Inet).as_ref(),
            ),
            Kind::Dgram(d) if d.peer.is_some() => {
                write_sockaddr(mem, addr, addrlen, domain, d.peer.map(Addr::Inet).as_ref())
            }
            Kind::Host(h) => {
                let peer = Addr::Inet(h.peer);
                write_sockaddr(mem, addr, addrlen, domain, Some(&peer))
            }
            _ => err(Errno::ENOTCONN),
        }
    }

    /// `shutdown(fd, how)` — `SHUT_RD` (0) marks this end's read side closed
    /// (further reads return EOF immediately, regardless of what's still
    /// queued); `SHUT_WR` (1) marks the write side closed (further writes
    /// return `EPIPE`, and the peer sees EOF on read once it drains what's
    /// already queued); `SHUT_RDWR` (2) does both.
    pub(super) fn sys_shutdown(&self, net: &mut Net, cx: &mut ServiceCtx, fd: u64, how: u64) -> i64 {
        const SHUT_RD: u64 = 0;
        const SHUT_WR: u64 = 1;
        const SHUT_RDWR: u64 = 2;
        let Some((sock, end)) = self.sock_of(cx, fd) else {
            return err(Errno::ENOTSOCK);
        };
        match &mut net.socks[sock].kind {
            Kind::Pair(p) => {
                match how {
                    SHUT_RD => p.shut_rd[end] = true,
                    SHUT_WR => p.shut_wr[end] = true,
                    SHUT_RDWR => {
                        p.shut_rd[end] = true;
                        p.shut_wr[end] = true;
                    }
                    _ => return err(Errno::EINVAL),
                }
                0
            }
            Kind::Host(h) => {
                if matches!(how, SHUT_WR | SHUT_RDWR) {
                    h.wr_shut = true;
                    h.conn.shutdown_write();
                }
                0
            }
            _ => err(Errno::ENOTCONN),
        }
    }

    /// `setsockopt(fd, level, optname, optval, optlen)`. `SOL_SOCKET`
    /// `SO_REUSEADDR` has an observable effect (it relaxes the `bind`
    /// `EADDRINUSE` check); the rest of the options listed below are stored
    /// and echoed back by `getsockopt`, since this is a virtual loopback with
    /// no real transport to actually tune. Any option this module doesn't
    /// recognize is accepted and silently ignored (returns `0`) rather than
    /// erroring, matching how a real stack treats a great many `setsockopt`
    /// calls guest code makes speculatively.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_setsockopt(
        &self, net: &mut Net, cx: &mut ServiceCtx,
        fd: u64,
        level: u64,
        optname: u64,
        optval: u64,
        optlen: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let Some((sock, _)) = self.sock_of(cx, fd) else {
            return err(Errno::ENOTSOCK);
        };
        let opts = &mut net.socks[sock].opts;
        if level == SOL_SOCKET {
            match optname {
                SO_REUSEADDR if optlen >= 4 => {
                    if let Ok(v) = mem.read_u32(optval) {
                        opts.reuseaddr = v != 0;
                    }
                }
                SO_REUSEPORT if optlen >= 4 => {
                    if let Ok(v) = mem.read_u32(optval) {
                        opts.reuseport = v != 0;
                    }
                }
                SO_KEEPALIVE if optlen >= 4 => {
                    if let Ok(v) = mem.read_u32(optval) {
                        opts.keepalive = v != 0;
                    }
                }
                SO_BROADCAST if optlen >= 4 => {
                    if let Ok(v) = mem.read_u32(optval) {
                        opts.broadcast = v != 0;
                    }
                }
                SO_RCVBUF if optlen >= 4 => {
                    if let Ok(v) = mem.read_u32(optval) {
                        opts.rcvbuf = v;
                    }
                }
                SO_SNDBUF if optlen >= 4 => {
                    if let Ok(v) = mem.read_u32(optval) {
                        opts.sndbuf = v;
                    }
                }
                SO_LINGER if optlen >= 8 => {
                    if let Ok(b) = mem.read_vec(optval, 8) {
                        opts.linger_on = i32::from_le_bytes([b[0], b[1], b[2], b[3]]) != 0;
                        opts.linger_secs = i32::from_le_bytes([b[4], b[5], b[6], b[7]]);
                    }
                }
                SO_RCVTIMEO if optlen >= 16 => {
                    if let Ok(b) = mem.read_vec(optval, 16) {
                        opts.rcvtimeo.copy_from_slice(&b);
                    }
                }
                SO_SNDTIMEO if optlen >= 16 => {
                    if let Ok(b) = mem.read_vec(optval, 16) {
                        opts.sndtimeo.copy_from_slice(&b);
                    }
                }
                // SO_TYPE/SO_ERROR/SO_ACCEPTCONN/SO_DOMAIN/SO_PROTOCOL are
                // read-only in real Linux; anything else is unrecognized.
                // Either way: accept-and-ignore.
                _ => {}
            }
        } else if level == IPPROTO_TCP && optname == TCP_NODELAY && optlen >= 4 {
            if let Ok(v) = mem.read_u32(optval) {
                opts.nodelay = v != 0;
            }
        } else if level == IPPROTO_IPV6 && optname == IPV6_V6ONLY && optlen >= 4 {
            if let Ok(v) = mem.read_u32(optval) {
                opts.v6only = v != 0;
            }
        } else if level == IPPROTO_IP
            && optname == IP_TOS
            && optlen >= 4
            && let Ok(v) = mem.read_u32(optval)
        {
            opts.tos = v;
        }
        // Unknown level/optname combos: accept-and-ignore.
        0
    }

    /// `getsockopt(fd, level, optname, optval, optlen)` — sane canned/stored
    /// values for the options `setsockopt` above understands, plus
    /// `SO_TYPE`/`SO_ERROR`/`SO_ACCEPTCONN`/`SO_DOMAIN`/`SO_PROTOCOL`; `0` for
    /// anything else.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub(super) fn sys_getsockopt(
        &self, net: &mut Net, cx: &mut ServiceCtx,
        fd: u64,
        level: u64,
        optname: u64,
        optval: u64,
        optlen_ptr: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some((sock, _)) = self.sock_of(cx, fd) else {
            return err(Errno::ENOTSOCK);
        };
        // SO_LINGER and the timeval-shaped options are wider than the u32
        // fast path below, so they're handled (and returned) up front.
        if level == SOL_SOCKET && optname == SO_LINGER {
            let opts = &net.socks[sock].opts;
            let mut b = [0u8; 8];
            b[0..4].copy_from_slice(&i32::from(opts.linger_on).to_le_bytes());
            b[4..8].copy_from_slice(&opts.linger_secs.to_le_bytes());
            return write_optval(mem, optval, optlen_ptr, &b);
        }
        if level == SOL_SOCKET && (optname == SO_RCVTIMEO || optname == SO_SNDTIMEO) {
            let opts = &net.socks[sock].opts;
            let b = if optname == SO_RCVTIMEO {
                opts.rcvtimeo
            } else {
                opts.sndtimeo
            };
            return write_optval(mem, optval, optlen_ptr, &b);
        }
        let value: u32 = if level == SOL_SOCKET {
            match optname {
                SO_TYPE => match &net.socks[sock].kind {
                    Kind::Dgram(_) => SOCK_DGRAM as u32,
                    Kind::Netlink(nl) => nl.sotype as u32,
                    _ => SOCK_STREAM as u32,
                },
                SO_ERROR => {
                    let e = net.socks[sock].opts.error;
                    net.socks[sock].opts.error = 0; // read-and-cleared
                    e as u32
                }
                SO_REUSEADDR => u32::from(net.socks[sock].opts.reuseaddr),
                SO_REUSEPORT => u32::from(net.socks[sock].opts.reuseport),
                SO_KEEPALIVE => u32::from(net.socks[sock].opts.keepalive),
                SO_BROADCAST => u32::from(net.socks[sock].opts.broadcast),
                SO_RCVBUF => net.socks[sock].opts.rcvbuf,
                SO_SNDBUF => net.socks[sock].opts.sndbuf,
                SO_ACCEPTCONN => {
                    u32::from(matches!(net.socks[sock].kind, Kind::Listener { .. }))
                }
                SO_DOMAIN => u32::from(net.socks[sock].domain),
                SO_PROTOCOL => match (&net.socks[sock].kind, net.socks[sock].domain) {
                    (Kind::Dgram(_), _) => 17,                      // IPPROTO_UDP
                    (_, d) if d == AF_UNIX || d == AF_NETLINK => 0, // NETLINK_ROUTE == 0
                    _ => 6,                                         // IPPROTO_TCP
                },
                _ => 0,
            }
        } else if level == IPPROTO_TCP && optname == TCP_NODELAY {
            u32::from(net.socks[sock].opts.nodelay)
        } else if level == IPPROTO_IPV6 && optname == IPV6_V6ONLY {
            u32::from(net.socks[sock].opts.v6only)
        } else if level == IPPROTO_IP && optname == IP_TOS {
            net.socks[sock].opts.tos
        } else {
            0
        };
        write_optval(mem, optval, optlen_ptr, &value.to_le_bytes())
    }

    /// `sendto(fd, buf, len, flags, dest_addr, addrlen)` — for a datagram
    /// socket with an explicit destination, deliver straight into that port's
    /// inbound queue (fire-and-forget, like real UDP: no error if nothing is
    /// bound there); otherwise (no destination, or a stream socket) this is
    /// just `write`. `flags` (`MSG_DONTWAIT`/`MSG_NOSIGNAL`/…) has nothing to
    /// observably change here: a send into these in-memory queues never
    /// blocks and never raises `SIGPIPE` in the first place.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_sendto(
        &self, net: &mut Net, cx: &mut ServiceCtx,
        fd: u64,
        buf: u64,
        len: u64,
        _flags: u64,
        dest_addr: u64,
        dest_addrlen: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let Ok(data) = mem.read_vec(buf, len as usize) else {
            return err(Errno::EFAULT);
        };
        self.send_bytes(net, cx, fd, &data, dest_addr, dest_addrlen, mem)
    }

    /// The shared core of `sendto`/`sendmsg`: send an already-gathered `data`
    /// buffer on socket `fd`, optionally to `dest_addr` (a guest `sockaddr` of
    /// `dest_addrlen` bytes; `0` = no address, e.g. a connected socket).
    #[allow(clippy::too_many_arguments)]
    fn send_bytes(
        &self, net: &mut Net, cx: &mut ServiceCtx,
        fd: u64,
        data: &[u8],
        dest_addr: u64,
        dest_addrlen: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let Some((sock, end)) = self.sock_of(cx, fd) else {
            return err(Errno::ENOTSOCK);
        };
        if dest_addr == 0 || net.socks[sock].domain == AF_NETLINK {
            // An `AF_NETLINK` socket only ever has one valid peer (the
            // kernel), so a `dest_addr` sockaddr_nl, if given at all, carries
            // no information this module needs — it's the same request path
            // as a plain `write`.
            return self.write_socket(net, cx, sock, end, data);
        }
        let Some(Addr::Inet(dest)) = read_sockaddr(mem, dest_addr, dest_addrlen) else {
            return err(Errno::EINVAL);
        };
        if !matches!(net.socks[sock].kind, Kind::Dgram(_)) {
            return err(Errno::EINVAL); // real errno: EOPNOTSUPP/EISCONN
        }
        // A routable destination (DNS, chiefly) goes out through a real host
        // UDP socket when egress is enabled; without egress it's unreachable.
        if !dest.valid_bind() {
            if net.egress.is_none() {
                return err(Errno::ENETUNREACH);
            }
            return self.host_udp_send(net, sock, data, dest);
        }
        let src = self.ensure_dgram_bound(net, sock);
        let key = route_key("udp", dest);
        if let Some(&tgt) = net.dgram_ports.get(&key)
            && let Kind::Dgram(td) = &mut net.socks[tgt].kind
        {
            td.queue.push_back((src, data.to_vec()));
        }
        data.len() as i64
    }

    /// Readiness (`POLLIN`/`POLLOUT`) for a host-bridged socket slot, or `None`
    /// for a non-host socket (which uses the generic best-effort answer). Only
    /// host sockets get a precise readable answer (via a peek), so a guest that
    /// trusts `poll` (apk's http client) doesn't spin on spurious readiness.
    pub(super) fn host_socket_readiness(&self, net: &mut Net, sock: usize) -> Option<u32> {
        const POLLIN: u32 = 0x1;
        const POLLOUT: u32 = 0x4;
        if !matches!(
            net.socks.get(sock).map(|s| &s.kind),
            Some(Kind::Host(_) | Kind::Dgram(Dgram { host: Some(_), .. }))
        ) {
            return None;
        }
        if let Kind::Dgram(_) = &net.socks[sock].kind {
            self.host_udp_drain(net, sock);
            let readable =
                matches!(&net.socks[sock].kind, Kind::Dgram(d) if !d.queue.is_empty());
            return Some(POLLOUT | if readable { POLLIN } else { 0 });
        }
        let Kind::Host(h) = &mut net.socks[sock].kind else {
            return None;
        };
        let mut mask = if h.wr_shut { 0 } else { POLLOUT };
        if h.conn.poll_readable() {
            mask |= POLLIN;
        }
        Some(mask)
    }

    /// Send a datagram out through this socket's host UDP socket (opening it
    /// lazily on first use), for egress to a routable address.
    #[allow(clippy::unused_self)]
    fn host_udp_send(&self, net: &mut Net, sock: usize, data: &[u8], dest: InetAddr) -> i64 {
        if let Kind::Dgram(d) = &net.socks[sock].kind
            && d.host.is_none()
        {
            match net.egress.as_ref().map(|e| e.open_udp()) {
                Some(Ok(h)) => {
                    if let Kind::Dgram(d) = &mut net.socks[sock].kind {
                        d.host = Some(h);
                    }
                }
                _ => return err(Errno::ENETUNREACH),
            }
        }
        let Kind::Dgram(d) = &mut net.socks[sock].kind else {
            return err(Errno::EINVAL);
        };
        let Some(host) = d.host.as_mut() else {
            return err(Errno::ENETUNREACH);
        };
        match host.send_to(data, dest.ip, dest.v6, dest.port) {
            Ok(n) => n as i64,
            Err(_) => data.len() as i64, // UDP is fire-and-forget; swallow errors
        }
    }

    /// Drain any datagrams waiting on socket `sock`'s host UDP socket into its
    /// inbound queue, so a following `recvfrom`/`recvmsg` sees them. No-op for
    /// a non-host datagram socket.
    #[allow(clippy::unused_self)]
    fn host_udp_drain(&self, net: &mut Net, sock: usize) {
        let Kind::Dgram(d) = &mut net.socks[sock].kind else {
            return;
        };
        let Some(host) = d.host.as_mut() else {
            return;
        };
        // Pull whatever is ready (non-blocking), then push into the queue —
        // collect first so the host borrow ends before the queue borrow.
        let mut arrived = Vec::new();
        for _ in 0..64 {
            match host.recv_from() {
                Ok(Some((ip, v6, port, payload))) => {
                    arrived.push((InetAddr { v6, port, ip }, payload));
                }
                _ => break,
            }
        }
        if let Kind::Dgram(d) = &mut net.socks[sock].kind {
            d.queue.extend(arrived);
        }
    }

    /// `recvfrom(fd, buf, len, flags, src_addr, addrlen)` — for a datagram
    /// socket, pop (or, with `MSG_PEEK`, peek at) the next queued datagram and
    /// report its source address; for a stream socket this is a flag-aware
    /// `read` plus (best-effort) the peer address. `MSG_DONTWAIT` forces
    /// non-blocking for this call only; `MSG_TRUNC` (datagram only) makes the
    /// return value the full datagram length even if the caller's buffer was
    /// smaller (the copy itself is always truncated to `len`, matching real
    /// `recv` — this flag only changes what the *return value* reports).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_recvfrom(
        &self, net: &mut Net, cx: &mut ServiceCtx,
        fd: u64,
        buf: u64,
        len: u64,
        flags: u64,
        src_addr: u64,
        src_addrlen: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some((sock, end)) = self.sock_of(cx, fd) else {
            return err(Errno::ENOTSOCK);
        };
        if matches!(net.socks[sock].kind, Kind::Netlink(_)) {
            let nonblock = net.socks[sock].nonblock || flags & MSG_DONTWAIT != 0;
            let Some(data) = self.drain_netlink(net, sock, len as usize) else {
                if nonblock {
                    return err(Errno::EAGAIN);
                }
                cx.block = true;
                return 0;
            };
            // Every reply's source is the kernel, whose `nl_pid` is always `0`.
            write_netlink_sockaddr(mem, src_addr, src_addrlen, 0);
            let n = data.len();
            if mem.write(buf, &data).is_err() {
                return err(Errno::EFAULT);
            }
            return n as i64;
        }
        if !matches!(net.socks[sock].kind, Kind::Dgram(_)) {
            let n = self.recv_stream(net, cx, sock, end, buf, len, mem, flags);
            if n > 0 {
                let domain = net.socks[sock].domain;
                let peer = match &net.socks[sock].kind {
                    Kind::Pair(p) => p.addrs[1 - end].map(Addr::Inet),
                    _ => None,
                };
                write_sockaddr(mem, src_addr, src_addrlen, domain, peer.as_ref());
            }
            return n;
        }
        let nonblock = net.socks[sock].nonblock || flags & MSG_DONTWAIT != 0;
        let Some((from, data)) = self.recv_dgram_msg(net, sock, flags) else {
            if nonblock {
                return err(Errno::EAGAIN);
            }
            cx.block = true;
            return 0;
        };
        let domain = net.socks[sock].domain;
        write_sockaddr(mem, src_addr, src_addrlen, domain, Some(&Addr::Inet(from)));
        let n = (len as usize).min(data.len());
        if mem.write(buf, &data[..n]).is_err() {
            return err(Errno::EFAULT);
        }
        if flags & MSG_TRUNC != 0 {
            data.len() as i64
        } else {
            n as i64
        }
    }

    /// `sendmsg(fd, msghdr, flags)` — gather the `msg_iov` scatter/gather list
    /// into one buffer and send it, honoring `msg_name` as the destination
    /// (the datagram case). apk's HTTP/TLS client and musl's resolver use
    /// `sendmsg` rather than `sendto`.
    pub(super) fn sys_sendmsg(&self, net: &mut Net, cx: &mut ServiceCtx, fd: u64, msg: u64, flags: u64, mem: &GuestMemory) -> i64 {
        let Some(hdr) = MsgHdr::read(mem, msg) else {
            return err(Errno::EFAULT);
        };
        let mut data = Vec::new();
        for i in 0..hdr.iovlen {
            let ent = hdr.iov + i * 16;
            let (Ok(base), Ok(len)) = (mem.read_u64(ent), mem.read_u64(ent + 8)) else {
                return err(Errno::EFAULT);
            };
            match mem.read_vec(base, len as usize) {
                Ok(mut chunk) => data.append(&mut chunk),
                Err(_) => return err(Errno::EFAULT),
            }
        }
        let _ = flags;
        self.send_bytes(net, cx, fd, &data, hdr.name, u64::from(hdr.namelen), mem)
    }

    /// `sendmmsg(fd, msgvec, vlen, flags)` — send an array of `struct mmsghdr`
    /// `{ msghdr msg_hdr; u32 msg_len }` (64 bytes each), writing each sent
    /// byte count back into `msg_len`. Returns the number of messages sent.
    pub(super) fn sys_sendmmsg(&self, net: &mut Net, cx: &mut ServiceCtx, fd: u64, msgvec: u64, vlen: u64, mem: &mut GuestMemory) -> i64 {
        let mut sent = 0i64;
        for i in 0..vlen {
            let ent = msgvec + i * 64;
            let r = self.sys_sendmsg(net, cx, fd, ent, 0, mem);
            if r < 0 {
                return if sent > 0 { sent } else { r };
            }
            let _ = mem.write(ent + 56, &(r as u32).to_le_bytes());
            sent += 1;
        }
        sent
    }

    /// `recvmmsg(fd, msgvec, vlen, flags, timeout)` — receive into an array of
    /// `struct mmsghdr`, writing each received byte count into `msg_len`.
    /// Returns the number of messages received (stops at the first that would
    /// block, like the real syscall).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_recvmmsg(&self, net: &mut Net, cx: &mut ServiceCtx, fd: u64, msgvec: u64, vlen: u64, flags: u64, mem: &mut GuestMemory) -> i64 {
        let mut got = 0i64;
        for i in 0..vlen {
            let ent = msgvec + i * 64;
            let r = self.sys_recvmsg(net, cx, fd, ent, flags, mem);
            if r < 0 {
                // EAGAIN after at least one message is a normal stop, not an error.
                if got > 0 { return got; }
                return r;
            }
            let _ = mem.write(ent + 56, &(r as u32).to_le_bytes());
            got += 1;
            // Only force the first recv to block; the rest are best-effort.
        }
        got
    }

    /// `recvmsg(fd, msghdr, flags)` — receive one message into a scratch buffer
    /// sized to the `msg_iov` total, scatter it across the iovecs, and fill in
    /// `msg_name`/`msg_namelen` (source address) and `msg_flags`. Control data
    /// is not modeled: `msg_controllen` is cleared.
    pub(super) fn sys_recvmsg(
        &self, net: &mut Net, cx: &mut ServiceCtx,
        fd: u64,
        msg: u64,
        flags: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some(hdr) = MsgHdr::read(mem, msg) else {
            return err(Errno::EFAULT);
        };
        // Total scatter capacity, and the (base, len) list to scatter into.
        let mut iovs = Vec::with_capacity(hdr.iovlen as usize);
        let mut total = 0u64;
        for i in 0..hdr.iovlen {
            let ent = hdr.iov + i * 16;
            let (Ok(base), Ok(len)) = (mem.read_u64(ent), mem.read_u64(ent + 8)) else {
                return err(Errno::EFAULT);
            };
            iovs.push((base, len));
            total += len;
        }
        // Receive into a bounce buffer at a scratch guest address? No — reuse
        // recvfrom by receiving into the first iovec when there's exactly one
        // (the common case), else into a temporary host staging area via a
        // single-shot read. We stage in host memory: read the datagram/stream
        // bytes out through the existing path into iov[0]-sized reads.
        //
        // Simplest correct approach: pull up to `total` bytes with recvfrom
        // into the first iovec region is wrong when total spans many iovecs.
        // Instead gather into a host Vec by receiving into the largest single
        // iovec repeatedly is also wrong for datagrams. So: receive once into
        // a host buffer via a dedicated helper.
        let (src, mut got, msg_flags) = match self.recv_message(net, cx, fd, total, flags) {
            Ok(v) => v,
            Err(e) => return e,
        };
        // Scatter `got` across the iovecs.
        let mut off = 0usize;
        for (base, len) in iovs {
            if off >= got.len() {
                break;
            }
            let take = ((len as usize).min(got.len() - off)).min(got.len());
            if mem.write(base, &got[off..off + take]).is_err() {
                return err(Errno::EFAULT);
            }
            off += take;
        }
        got.truncate(off);
        // Fill msg_name (source address) and msg_flags; clear control length.
        // `msg_namelen` (socklen_t at msg+8) doubles as write_sockaddr's
        // in/out length word — it carries the caller's capacity in and the
        // written length out, exactly as recvmsg wants.
        if hdr.name != 0 && hdr.namelen > 0 {
            let domain = self
                .sock_of(cx, fd)
                .map_or(AF_INET, |(s, _)| net.socks[s].domain);
            write_sockaddr(mem, hdr.name, msg + 8, domain, src.as_ref());
        }
        // msg_controllen := 0 (offset 40), msg_flags := msg_flags (offset 48).
        let _ = mem.write(msg + 40, &0u64.to_le_bytes());
        let _ = mem.write(msg + 48, &(msg_flags as i32).to_le_bytes());
        off as i64
    }

    /// Receive one message's bytes (up to `cap`) from socket `fd` into a host
    /// `Vec`, returning the source address (datagram) and out `msg_flags`. The
    /// shared core of `recvmsg`, factored out of `recvfrom` so both can drive
    /// the netlink / stream / datagram paths without a guest bounce buffer.
    fn recv_message(
        &self, net: &mut Net, cx: &mut ServiceCtx,
        fd: u64,
        cap: u64,
        flags: u64,
    ) -> Result<(Option<Addr>, Vec<u8>, u64), i64> {
        let Some((sock, end)) = self.sock_of(cx, fd) else {
            return Err(err(Errno::ENOTSOCK));
        };
        match &net.socks[sock].kind {
            Kind::Netlink(_) => {
                let nonblock = net.socks[sock].nonblock || flags & MSG_DONTWAIT != 0;
                match self.drain_netlink(net, sock, cap as usize) {
                    Some(data) => Ok((None, data, 0)),
                    None => {
                        if nonblock {
                            Err(err(Errno::EAGAIN))
                        } else {
                            cx.block = true;
                            Ok((None, Vec::new(), 0))
                        }
                    }
                }
            }
            Kind::Dgram(_) => {
                let nonblock = net.socks[sock].nonblock || flags & MSG_DONTWAIT != 0;
                match self.recv_dgram_msg(net, sock, flags) {
                    Some((from, mut data)) => {
                        let truncated = data.len() as u64 > cap;
                        data.truncate(cap as usize);
                        let mf = if truncated { MSG_TRUNC } else { 0 };
                        Ok((Some(Addr::Inet(from)), data, mf))
                    }
                    None => {
                        if nonblock {
                            Err(err(Errno::EAGAIN))
                        } else {
                            cx.block = true;
                            Ok((None, Vec::new(), 0))
                        }
                    }
                }
            }
            Kind::Host(_) => {
                let bytes = self.host_recv(net, cx, sock, cap as usize, flags & MSG_DONTWAIT != 0)?;
                let peer = match &net.socks[sock].kind {
                    Kind::Host(h) => Some(Addr::Inet(h.peer)),
                    _ => None,
                };
                Ok((peer, bytes, 0))
            }
            _ => {
                // Stream: pull bytes directly out of the inbound queue (the
                // flag-aware `recv_stream` writes to guest memory, so replicate
                // its dequeue here against a host buffer instead).
                let data = self.recv_stream_bytes(net, cx, sock, end, cap, flags);
                match data {
                    Ok(bytes) => {
                        let peer = match &net.socks[sock].kind {
                            Kind::Pair(p) => p.addrs[1 - end].map(Addr::Inet),
                            _ => None,
                        };
                        Ok((peer, bytes, 0))
                    }
                    Err(e) => Err(e),
                }
            }
        }
    }

    /// Pop (or, for `MSG_PEEK`, peek at) datagram socket `sock`'s next queued
    /// inbound `(source, payload)`. A pure query: the caller decides the
    /// block/`EAGAIN` behavior when this returns `None` (an empty queue).
    fn recv_dgram_msg(&self, net: &mut Net, sock: usize, flags: u64) -> Option<(InetAddr, Vec<u8>)> {
        // Pull any host-arrived datagrams (egress replies, e.g. DNS) into the
        // queue first, so they're visible to this dequeue.
        self.host_udp_drain(net, sock);
        let peek = flags & MSG_PEEK != 0;
        match &mut net.socks[sock].kind {
            Kind::Dgram(d) if peek => d.queue.front().cloned(),
            Kind::Dgram(d) => d.queue.pop_front(),
            _ => unreachable!("checked by caller"),
        }
    }

    /// Read from socket `sock`'s inbound queue for `end`. For a stream pair:
    /// empty with the peer still open -> block (or `EAGAIN` if `O_NONBLOCK`);
    /// empty with the peer closed -> EOF (0). For a datagram socket: pop the
    /// next queued datagram's payload (no address; that's `recvfrom`'s job),
    /// blocking/`EAGAIN`-ing the same way while the queue is empty (UDP sockets
    /// never see EOF). Mirrors `read_pipe`. This is the plain `read()` path
    /// (no `MSG_*` flags); `recvfrom`/`recv` go through [`Self::recv_stream`]
    /// and [`Self::recv_dgram_msg`] instead, which are flag-aware.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn read_socket(
        &self, net: &mut Net, cx: &mut ServiceCtx,
        sock: usize,
        end: usize,
        buf: u64,
        count: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        if matches!(net.socks[sock].kind, Kind::Dgram(_)) {
            let nonblock = net.socks[sock].nonblock;
            let Some((_, data)) = self.recv_dgram_msg(net, sock, 0) else {
                if nonblock {
                    return err(Errno::EAGAIN);
                }
                cx.block = true;
                return 0;
            };
            let n = (count as usize).min(data.len());
            if mem.write(buf, &data[..n]).is_err() {
                return err(Errno::EFAULT);
            }
            return n as i64;
        }
        if matches!(net.socks[sock].kind, Kind::Netlink(_)) {
            let nonblock = net.socks[sock].nonblock;
            let Some(data) = self.drain_netlink(net, sock, count as usize) else {
                if nonblock {
                    return err(Errno::EAGAIN);
                }
                cx.block = true;
                return 0;
            };
            if mem.write(buf, &data).is_err() {
                return err(Errno::EFAULT);
            }
            return data.len() as i64;
        }
        if matches!(net.socks[sock].kind, Kind::Host(_)) {
            return match self.host_recv(net, cx, sock, count as usize, false) {
                Ok(data) => {
                    if mem.write(buf, &data).is_err() {
                        return err(Errno::EFAULT);
                    }
                    data.len() as i64
                }
                Err(e) => e,
            };
        }
        self.recv_stream(net, cx, sock, end, buf, count, mem, 0)
    }

    /// Read up to `cap` bytes from a host-bridged stream socket. `Ok(bytes)`
    /// (possibly empty on EOF); `Err(errno)` on `EAGAIN`/error. When the host
    /// side has no data yet and the socket is blocking, sets `self.block` and
    /// returns `Ok(empty)` so the caller returns 0 and the guest re-traps.
    #[allow(clippy::unused_self)]
    fn host_recv(&self, net: &mut Net, cx: &mut ServiceCtx, sock: usize, cap: usize, force_nonblock: bool) -> Result<Vec<u8>, i64> {
        let Kind::Host(h) = &mut net.socks[sock].kind else {
            return Err(err(Errno::EINVAL));
        };
        let nonblock = h.nonblock || force_nonblock;
        let mut buf = vec![0u8; cap.min(256 * 1024)];
        match h.conn.recv(&mut buf) {
            Ok(n) => {
                buf.truncate(n);
                Ok(buf) // n == 0 is a genuine EOF
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                if nonblock {
                    Err(err(Errno::EAGAIN))
                } else {
                    cx.block = true;
                    Ok(Vec::new())
                }
            }
            Err(_) => Err(err(Errno::ECONNRESET)),
        }
    }

    /// Drain up to `want` bytes' worth of complete, already-encoded messages
    /// from netlink socket `sock`'s reply queue, in order — packing as many
    /// whole messages as fit (never splitting one across two `recv` calls,
    /// leaving the remainder queued for the next read, like a real netlink
    /// socket packing a dump reply across skbs). `None` means the queue is
    /// currently empty (the caller decides block/`EAGAIN`); `Some` is always
    /// non-empty. If even the single next message doesn't fit `want`, it is
    /// truncated to `want` bytes and dropped from the queue (datagram
    /// semantics: an oversized read is truncated, not left partially queued).
    #[allow(clippy::unused_self)]
    fn drain_netlink(&self, net: &mut Net, sock: usize, want: usize) -> Option<Vec<u8>> {
        let Kind::Netlink(nl) = &mut net.socks[sock].kind else {
            unreachable!("checked by caller")
        };
        if nl.queue.is_empty() {
            return None;
        }
        let mut out = Vec::new();
        while let Some(next) = nl.queue.front() {
            if out.is_empty() && next.len() > want {
                let msg = nl.queue.pop_front().expect("front just checked Some");
                out.extend_from_slice(&msg[..want.min(msg.len())]);
                break;
            }
            if out.len() + next.len() > want {
                break;
            }
            let msg = nl.queue.pop_front().expect("front just checked Some");
            out.extend_from_slice(&msg);
        }
        Some(out)
    }

    /// Shared stream-socket receive path for plain `read()` (`flags == 0`)
    /// and `recvfrom()`/`recv()`. An immediate EOF (`0`) if this end's read
    /// side was `shutdown(SHUT_RD)`; otherwise the peer's queued bytes,
    /// honoring `MSG_PEEK` (leave the bytes queued instead of draining them),
    /// `MSG_DONTWAIT` (force non-blocking for this call only), and a
    /// best-effort `MSG_WAITALL` (don't return a short read while more is
    /// expected and the peer is still writable — ignored when non-blocking,
    /// matching real Linux). Mirrors `read_pipe`'s blocking convention:
    /// empty + peer open -> block (or `EAGAIN`); empty + peer closed -> EOF.
    #[allow(clippy::too_many_arguments)]
    fn recv_stream(
        &self, net: &mut Net, cx: &mut ServiceCtx,
        sock: usize,
        end: usize,
        buf: u64,
        count: u64,
        mem: &mut GuestMemory,
        flags: u64,
    ) -> i64 {
        if matches!(net.socks[sock].kind, Kind::Host(_)) {
            return match self.host_recv(net, cx, sock, count as usize, flags & MSG_DONTWAIT != 0) {
                Ok(data) => {
                    if mem.write(buf, &data).is_err() {
                        return err(Errno::EFAULT);
                    }
                    data.len() as i64
                }
                Err(e) => e,
            };
        }
        let (shut_rd, avail, peer_open, nonblock) = match &net.socks[sock].kind {
            Kind::Pair(p) => (
                p.shut_rd[end],
                p.to[end].len(),
                p.refs[1 - end] > 0 && !p.shut_wr[1 - end],
                p.nonblock[end] || flags & MSG_DONTWAIT != 0,
            ),
            _ => return err(Errno::EINVAL),
        };
        if shut_rd {
            return 0;
        }
        let short = flags & MSG_WAITALL != 0
            && !nonblock
            && avail > 0
            && avail < count as usize
            && peer_open;
        if avail == 0 || short {
            if peer_open {
                if nonblock {
                    return err(Errno::EAGAIN);
                }
                cx.block = true;
            }
            return 0;
        }
        let peek = flags & MSG_PEEK != 0;
        let data: Vec<u8> = match &mut net.socks[sock].kind {
            Kind::Pair(p) => {
                let n = (count as usize).min(p.to[end].len());
                if peek {
                    p.to[end].iter().take(n).copied().collect()
                } else {
                    p.to[end].drain(..n).collect()
                }
            }
            _ => return err(Errno::EINVAL),
        };
        if mem.write(buf, &data).is_err() {
            return err(Errno::EFAULT);
        }
        data.len() as i64
    }

    /// Host-buffer twin of [`Self::recv_stream`]: dequeue up to `count` stream
    /// bytes into a `Vec` (for `recvmsg`, which scatters across iovecs rather
    /// than writing one guest region). Same block/`EAGAIN`/EOF semantics.
    #[allow(clippy::unused_self)]
    fn recv_stream_bytes(
        &self, net: &mut Net, cx: &mut ServiceCtx,
        sock: usize,
        end: usize,
        count: u64,
        flags: u64,
    ) -> Result<Vec<u8>, i64> {
        let (shut_rd, avail, peer_open, nonblock) = match &net.socks[sock].kind {
            Kind::Pair(p) => (
                p.shut_rd[end],
                p.to[end].len(),
                p.refs[1 - end] > 0 && !p.shut_wr[1 - end],
                p.nonblock[end] || flags & MSG_DONTWAIT != 0,
            ),
            _ => return Err(err(Errno::EINVAL)),
        };
        if shut_rd {
            return Ok(Vec::new());
        }
        if avail == 0 {
            if peer_open {
                if nonblock {
                    return Err(err(Errno::EAGAIN));
                }
                cx.block = true;
            }
            return Ok(Vec::new());
        }
        let peek = flags & MSG_PEEK != 0;
        match &mut net.socks[sock].kind {
            Kind::Pair(p) => {
                let n = (count as usize).min(p.to[end].len());
                Ok(if peek {
                    p.to[end].iter().take(n).copied().collect()
                } else {
                    p.to[end].drain(..n).collect()
                })
            }
            _ => Err(err(Errno::EINVAL)),
        }
    }

    /// Append to socket `sock`'s outbound queue for `end`. For a stream pair,
    /// `EPIPE` if this end's write side was `shutdown(SHUT_WR)` or the peer
    /// end is fully closed. For a datagram socket, this is `send` (i.e.
    /// requires a `connect`-ed peer, else `ENOTCONN`) and delivers
    /// fire-and-forget, like real UDP: no error if nothing is bound at the
    /// peer's port. Mirrors `write_pipe`.
    pub(super) fn write_socket(&self, net: &mut Net, cx: &mut ServiceCtx, sock: usize, end: usize, data: &[u8]) -> i64 {
        if matches!(net.socks[sock].kind, Kind::Netlink(_)) {
            return self.handle_netlink_request(net, sock, data);
        }
        if matches!(net.socks[sock].kind, Kind::Dgram(_)) {
            let peer = match &net.socks[sock].kind {
                Kind::Dgram(d) => d.peer,
                _ => unreachable!("checked above"),
            };
            let Some(peer) = peer else {
                return err(Errno::ENOTCONN);
            };
            let src = self.ensure_dgram_bound(net, sock);
            let key = route_key("udp", peer);
            if let Some(&tgt) = net.dgram_ports.get(&key)
                && let Kind::Dgram(td) = &mut net.socks[tgt].kind
            {
                td.queue.push_back((src, data.to_vec()));
            }
            return data.len() as i64;
        }
        if let Kind::Host(h) = &mut net.socks[sock].kind {
            if h.wr_shut {
                return err(Errno::EPIPE);
            }
            return match h.conn.send(data) {
                Ok(n) => n as i64,
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    if h.nonblock {
                        err(Errno::EAGAIN)
                    } else {
                        // The host send buffer is full; ask the guest to retry.
                        cx.block = true;
                        0
                    }
                }
                Err(_) => err(Errno::ECONNRESET),
            };
        }
        match &mut net.socks[sock].kind {
            Kind::Pair(p) => {
                if p.shut_wr[end] || p.refs[1 - end] == 0 {
                    return err(Errno::EPIPE);
                }
                p.to[1 - end].extend(data.iter().copied());
                data.len() as i64
            }
            _ => err(Errno::EINVAL),
        }
    }

    /// Handle a guest's `NETLINK_ROUTE` request on netlink socket `sock`:
    /// `data` may hold one or more `nlmsghdr`-framed messages back to back
    /// (`NLMSG_ALIGN`-ed, per the wire format); each is answered independently
    /// and the reply message(s) are appended, in order, to the socket's reply
    /// queue for a later `recv` to drain. Returns `data.len()` (the whole
    /// request was consumed), matching a real `write`/`send`.
    #[allow(clippy::unused_self)]
    fn handle_netlink_request(&self, net: &mut Net, sock: usize, data: &[u8]) -> i64 {
        let pid = match &net.socks[sock].kind {
            Kind::Netlink(nl) => nl.pid,
            _ => 0,
        };
        let mut offset = 0usize;
        while offset + 16 <= data.len() {
            let hdr = &data[offset..offset + 16];
            let nlmsg_len = u32::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3]]) as usize;
            if nlmsg_len < 16 {
                break; // malformed: too short to even hold this header
            }
            let nlmsg_type = u16::from_le_bytes([hdr[4], hdr[5]]);
            let nlmsg_flags = u16::from_le_bytes([hdr[6], hdr[7]]);
            let nlmsg_seq = u32::from_le_bytes([hdr[8], hdr[9], hdr[10], hdr[11]]);
            let mut orig_hdr = [0u8; 16];
            orig_hdr.copy_from_slice(hdr);
            let want_ack = nlmsg_flags & NLM_F_ACK != 0;
            let dump = nlmsg_flags & NLM_F_DUMP == NLM_F_DUMP;

            let replies: Vec<Vec<u8>> = if nlmsg_type == RTM_GETLINK && dump {
                vec![
                    build_rtm_newlink(nlmsg_seq, pid),
                    encode_nlmsg(NLMSG_DONE, 0, nlmsg_seq, pid, &0i32.to_le_bytes()),
                ]
            } else if nlmsg_type == RTM_GETADDR && dump {
                vec![
                    build_rtm_newaddr_v4(nlmsg_seq, pid),
                    build_rtm_newaddr_v6(nlmsg_seq, pid),
                    encode_nlmsg(NLMSG_DONE, 0, nlmsg_seq, pid, &0i32.to_le_bytes()),
                ]
            } else if nlmsg_type == RTM_GETROUTE {
                // No routes beyond the implicit loopback one: just end the dump.
                vec![encode_nlmsg(
                    NLMSG_DONE,
                    0,
                    nlmsg_seq,
                    pid,
                    &0i32.to_le_bytes(),
                )]
            } else if want_ack {
                vec![encode_nlmsgerr(0, &orig_hdr, nlmsg_seq, pid)]
            } else {
                vec![encode_nlmsgerr(
                    -Errno::EOPNOTSUPP.0,
                    &orig_hdr,
                    nlmsg_seq,
                    pid,
                )]
            };
            if let Kind::Netlink(nl) = &mut net.socks[sock].kind {
                nl.queue.extend(replies);
            }
            offset += nlmsg_align(nlmsg_len);
        }
        data.len() as i64
    }
}

/// Decode a `sockaddr` into an [`Addr`]: the `sun_path` for AF_UNIX, or the
/// port/address for AF_INET (`struct sockaddr_in`) / AF_INET6
/// (`struct sockaddr_in6`). A `sun_path` whose first byte is NUL is Linux's
/// "abstract namespace": the name is every byte after that leading NUL up to
/// `addrlen` (embedded NULs allowed, no terminator — unlike a filesystem
/// path). The decoded [`Addr::Unix`] keeps that leading NUL character in its
/// `String`, which doubles as the `bind`/`listen`/`connect` lookup key in
/// [`Net::listeners`]: since a filesystem-path bind can never start with a
/// NUL byte, abstract and path names can never collide, and two guest
/// processes can rendezvous on an abstract name exactly like a path one — no
/// separate table needed.
/// The fields of a guest `struct msghdr` `sendmsg`/`recvmsg` need. 64-bit
/// layout: `msg_name`(0), `msg_namelen`(8, u32 + 4 pad), `msg_iov`(16),
/// `msg_iovlen`(24), `msg_control`(32), `msg_controllen`(40), `msg_flags`(48).
struct MsgHdr {
    name: u64,
    namelen: u32,
    iov: u64,
    iovlen: u64,
}

impl MsgHdr {
    fn read(mem: &GuestMemory, ptr: u64) -> Option<Self> {
        Some(Self {
            name: mem.read_u64(ptr).ok()?,
            namelen: mem.read_u32(ptr + 8).ok()?,
            iov: mem.read_u64(ptr + 16).ok()?,
            iovlen: mem.read_u64(ptr + 24).ok()?,
        })
    }
}

fn read_sockaddr(mem: &GuestMemory, ptr: u64, addrlen: u64) -> Option<Addr> {
    if addrlen < 2 {
        return None;
    }
    let bytes = mem.read_vec(ptr, (addrlen as usize).min(128)).ok()?;
    let family = u16::from_le_bytes([bytes[0], bytes[1]]);
    match family {
        AF_UNIX => {
            let path = &bytes[2..];
            if path.first() == Some(&0) {
                let len = (addrlen as usize).saturating_sub(2).min(path.len());
                Some(Addr::Unix(
                    String::from_utf8_lossy(&path[..len]).into_owned(),
                ))
            } else {
                let end = path.iter().position(|&c| c == 0).unwrap_or(path.len());
                Some(Addr::Unix(
                    String::from_utf8_lossy(&path[..end]).into_owned(),
                ))
            }
        }
        AF_INET if bytes.len() >= 8 => {
            let port = u16::from_be_bytes([bytes[2], bytes[3]]);
            let mut ip = [0u8; 16];
            ip[0..4].copy_from_slice(&bytes[4..8]);
            Some(Addr::Inet(InetAddr {
                v6: false,
                port,
                ip,
            }))
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
            AF_INET => encode_inet_sockaddr(InetAddr {
                v6: false,
                port: 0,
                ip: [0; 16],
            }),
            AF_INET6 => encode_inet_sockaddr(InetAddr {
                v6: true,
                port: 0,
                ip: [0; 16],
            }),
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

/// Write a `getsockopt` result: `value` to `optval` (best-effort) and its
/// length to the `socklen_t` at `optlen_ptr`. A no-op for a null pointer.
/// Always returns success (`0`).
fn write_optval(mem: &mut GuestMemory, optval: u64, optlen_ptr: u64, value: &[u8]) -> i64 {
    if optval != 0 {
        let _ = mem.write(optval, value);
    }
    if optlen_ptr != 0 {
        let _ = mem.write(optlen_ptr, &(value.len() as u32).to_le_bytes());
    }
    0
}

/// Round `len` up to a 4-byte boundary — `NLMSG_ALIGN` and `RTA_ALIGN` share
/// the same alignment (`NLMSG_ALIGNTO == RTA_ALIGNTO == 4`, linux/netlink.h).
fn nlmsg_align(len: usize) -> usize {
    (len + 3) & !3
}

/// Encode one complete `nlmsghdr` + `payload` message, `NLMSG_ALIGN`-ed (any
/// trailing pad bytes are zero). `nlmsg_len` records the *unpadded* length,
/// per the wire format — the padding exists only so a second message
/// concatenated right after this one starts on a 4-byte boundary.
fn encode_nlmsg(nlmsg_type: u16, flags: u16, seq: u32, pid: u32, payload: &[u8]) -> Vec<u8> {
    let total = 16 + payload.len();
    let mut b = vec![0u8; nlmsg_align(total)];
    b[0..4].copy_from_slice(&(total as u32).to_le_bytes());
    b[4..6].copy_from_slice(&nlmsg_type.to_le_bytes());
    b[6..8].copy_from_slice(&flags.to_le_bytes());
    b[8..12].copy_from_slice(&seq.to_le_bytes());
    b[12..16].copy_from_slice(&pid.to_le_bytes());
    b[16..total].copy_from_slice(payload);
    b
}

/// Encode one `rtattr` (`struct rtattr { len; type; } data...`),
/// `RTA_ALIGN`-ed. Concatenating several of these produces a valid rtattr
/// stream: each one's padded length is already a multiple of 4, so the next
/// attribute always starts aligned.
fn encode_rtattr(rta_type: u16, data: &[u8]) -> Vec<u8> {
    let len = 4 + data.len();
    let mut b = vec![0u8; nlmsg_align(len)];
    b[0..2].copy_from_slice(&(len as u16).to_le_bytes());
    b[2..4].copy_from_slice(&rta_type.to_le_bytes());
    b[4..4 + data.len()].copy_from_slice(data);
    b
}

/// An `NLMSG_ERROR` reply: `struct nlmsgerr { int error; struct nlmsghdr msg; }`
/// — `error` is `0` for a plain ACK, or a negative errno. `msg` is the
/// requesting message's header, copied back verbatim, matching the real
/// kernel's `netlink_ack`.
fn encode_nlmsgerr(error: i32, orig_hdr: &[u8; 16], seq: u32, pid: u32) -> Vec<u8> {
    let mut payload = vec![0u8; 4 + 16];
    payload[0..4].copy_from_slice(&error.to_le_bytes());
    payload[4..20].copy_from_slice(orig_hdr);
    encode_nlmsg(NLMSG_ERROR, 0, seq, pid, &payload)
}

/// `RTM_NEWLINK` describing the single, always-up loopback interface: index
/// `1`, `ARPHRD_LOOPBACK`, `IFF_UP|IFF_LOOPBACK|IFF_RUNNING`, plus
/// `IFLA_IFNAME="lo"`, `IFLA_MTU=65536`, and a 6-zero-byte `IFLA_ADDRESS`.
fn build_rtm_newlink(seq: u32, pid: u32) -> Vec<u8> {
    let flags = IFF_UP | IFF_LOOPBACK | IFF_RUNNING;
    let mut payload = vec![0u8; 16]; // struct ifinfomsg
    payload[2..4].copy_from_slice(&ARPHRD_LOOPBACK.to_le_bytes());
    payload[4..8].copy_from_slice(&LOOPBACK_IFINDEX.to_le_bytes());
    payload[8..12].copy_from_slice(&flags.to_le_bytes());
    payload.extend(encode_rtattr(IFLA_IFNAME, b"lo\0"));
    payload.extend(encode_rtattr(IFLA_MTU, &65_536u32.to_le_bytes()));
    payload.extend(encode_rtattr(IFLA_ADDRESS, &[0u8; 6]));
    encode_nlmsg(RTM_NEWLINK, 0, seq, pid, &payload)
}

/// `RTM_NEWADDR` for `127.0.0.1/8` on `lo`.
fn build_rtm_newaddr_v4(seq: u32, pid: u32) -> Vec<u8> {
    let mut payload = vec![0u8; 8]; // struct ifaddrmsg
    payload[0] = AF_INET as u8;
    payload[1] = 8; // ifa_prefixlen
    payload[3] = RT_SCOPE_HOST;
    payload[4..8].copy_from_slice(&(LOOPBACK_IFINDEX as u32).to_le_bytes());
    let ip = [127u8, 0, 0, 1];
    payload.extend(encode_rtattr(IFA_ADDRESS, &ip));
    payload.extend(encode_rtattr(IFA_LOCAL, &ip));
    payload.extend(encode_rtattr(IFA_LABEL, b"lo\0"));
    encode_nlmsg(RTM_NEWADDR, 0, seq, pid, &payload)
}

/// `RTM_NEWADDR` for `::1/128` on `lo`.
fn build_rtm_newaddr_v6(seq: u32, pid: u32) -> Vec<u8> {
    let mut payload = vec![0u8; 8]; // struct ifaddrmsg
    payload[0] = AF_INET6 as u8;
    payload[1] = 128; // ifa_prefixlen
    payload[4..8].copy_from_slice(&(LOOPBACK_IFINDEX as u32).to_le_bytes());
    let ip = loopback_ip(true);
    payload.extend(encode_rtattr(IFA_ADDRESS, &ip));
    payload.extend(encode_rtattr(IFA_LOCAL, &ip));
    payload.extend(encode_rtattr(IFA_LABEL, b"lo\0"));
    encode_nlmsg(RTM_NEWADDR, 0, seq, pid, &payload)
}

/// Write a `struct sockaddr_nl` (`nl_family`, 2 bytes pad, `nl_pid`,
/// `nl_groups=0`) to `addr`, truncated to the caller's buffer, updating the
/// `socklen_t` at `addrlen_ptr`. A no-op when `addrlen_ptr` is null. Always
/// returns success (`0`), mirroring [`write_sockaddr`].
fn write_netlink_sockaddr(mem: &mut GuestMemory, addr: u64, addrlen_ptr: u64, pid: u32) -> i64 {
    if addrlen_ptr == 0 {
        return 0;
    }
    let mut b = [0u8; 12];
    b[0..2].copy_from_slice(&AF_NETLINK.to_le_bytes());
    b[4..8].copy_from_slice(&pid.to_le_bytes());
    let cap = mem.read_u32(addrlen_ptr).unwrap_or(0) as usize;
    if addr != 0 {
        let n = b.len().min(cap);
        let _ = mem.write(addr, &b[..n]);
    }
    let _ = mem.write(addrlen_ptr, &(b.len() as u32).to_le_bytes());
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

    fn setup() -> (Kernel, GuestMemory, DummyVcpu, ServiceCtx) {
        let mut mounts = MountTable::new();
        mounts.mount("/", Box::new(TmpFs::new()));
        let kernel = Kernel::new(Arch::Aarch64, mounts);
        let mut cx = ServiceCtx::default();
        cx.cur.pid = 1;
        let mut mem = GuestMemory::new(0x1_0000, 16 * PAGE);
        mem.map(0x1_0000, 4 * PAGE, Prot::rw()).unwrap();
        (kernel, mem, DummyVcpu, cx)
    }

    fn call(
        k: &Kernel,
        cx: &mut ServiceCtx,
        mem: &mut GuestMemory,
        v: &mut DummyVcpu,
        s: Sysno,
        a: [u64; 6],
    ) -> i64 {
        k.dispatch(cx, s, 0, &a, v, mem)
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
        let (k, mut mem, mut v, mut cx) = setup();
        let sv = 0x1_0000;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Socketpair,
                [1, 1, 0, sv, 0, 0]
            ),
            0
        );
        let a = u64::from(mem.read_u32(sv).unwrap());
        let b = u64::from(mem.read_u32(sv + 4).unwrap());
        assert!(a >= 3 && b >= 3 && a != b);

        let msg = 0x1_1000;
        let out = 0x1_2000;
        mem.write_init(msg, b"hi").unwrap();
        assert_eq!(
            call(&k, &mut cx, &mut mem, &mut v, Sysno::Write, [a, msg, 2, 0, 0, 0]),
            2
        );
        assert_eq!(
            call(&k, &mut cx, &mut mem, &mut v, Sysno::Read, [b, out, 2, 0, 0, 0]),
            2
        );
        assert_eq!(mem.read_vec(out, 2).unwrap(), b"hi");

        // The other direction is empty with the peer still open -> blocks.
        assert_eq!(
            call(&k, &mut cx, &mut mem, &mut v, Sysno::Read, [a, out, 2, 0, 0, 0]),
            0
        );
        assert!(cx.block);
    }

    #[test]
    fn bind_listen_connect_accept_bidirectional() {
        let (k, mut mem, mut v, mut cx) = setup();
        let addr = 0x1_1000;
        mem.write_init(addr, &1u16.to_le_bytes()).unwrap(); // AF_UNIX
        mem.write_init(addr + 2, b"/s\0").unwrap();
        let alen = 5u64;

        let srv = call(&k, &mut cx, &mut mem, &mut v, Sysno::Socket, [1, 1, 0, 0, 0, 0]) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Bind,
                [srv, addr, alen, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Listen,
                [srv, 8, 0, 0, 0, 0]
            ),
            0
        );

        let cli = call(&k, &mut cx, &mut mem, &mut v, Sysno::Socket, [1, 1, 0, 0, 0, 0]) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Connect,
                [cli, addr, alen, 0, 0, 0]
            ),
            0
        );
        let acc = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Accept4,
            [srv, 0, 0, 0, 0, 0],
        );
        assert!(acc >= 3, "accept returned a fd");
        let acc = acc as u64;

        let msg = 0x1_2000;
        let out = 0x1_3000;
        // client -> server
        mem.write_init(msg, b"ping").unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [cli, msg, 4, 0, 0, 0]
            ),
            4
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Read,
                [acc, out, 4, 0, 0, 0]
            ),
            4
        );
        assert_eq!(mem.read_vec(out, 4).unwrap(), b"ping");
        // server -> client
        mem.write_init(msg, b"pong").unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [acc, msg, 4, 0, 0, 0]
            ),
            4
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Read,
                [cli, out, 4, 0, 0, 0]
            ),
            4
        );
        assert_eq!(mem.read_vec(out, 4).unwrap(), b"pong");
    }

    #[test]
    fn connect_without_listener_is_refused() {
        let (k, mut mem, mut v, mut cx) = setup();
        let addr = 0x1_1000;
        mem.write_init(addr, &1u16.to_le_bytes()).unwrap();
        mem.write_init(addr + 2, b"/nope\0").unwrap();
        let cli = call(&k, &mut cx, &mut mem, &mut v, Sysno::Socket, [1, 1, 0, 0, 0, 0]) as u64;
        let ret = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Connect,
            [cli, addr, 8, 0, 0, 0],
        );
        assert_eq!(ret, -i64::from(Errno::ECONNREFUSED.0));
    }

    #[test]
    fn write_to_socket_with_closed_peer_is_epipe() {
        let (k, mut mem, mut v, mut cx) = setup();
        let sv = 0x1_0000;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Socketpair,
                [1, 1, 0, sv, 0, 0]
            ),
            0
        );
        let end0 = u64::from(mem.read_u32(sv).unwrap());
        let end1 = u64::from(mem.read_u32(sv + 4).unwrap());

        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Close,
                [end1, 0, 0, 0, 0, 0]
            ),
            0
        );
        let msg = 0x1_1000;
        mem.write_init(msg, b"x").unwrap();
        let ret = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Write,
            [end0, msg, 1, 0, 0, 0],
        );
        assert_eq!(ret, -i64::from(Errno::EPIPE.0));
    }

    #[test]
    fn fstat_reports_socket_type() {
        let (k, mut mem, mut v, mut cx) = setup();
        let sv = 0x1_0000;
        call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Socketpair,
            [1, 1, 0, sv, 0, 0],
        );
        let a = u64::from(mem.read_u32(sv).unwrap());
        let st = 0x1_2000;
        assert_eq!(
            call(&k, &mut cx, &mut mem, &mut v, Sysno::Fstat, [a, st, 0, 0, 0, 0]),
            0
        );
        let mode = mem.read_u32(st + 16).unwrap();
        assert_eq!(mode & 0o170_000, 0o140_000, "S_IFSOCK");
    }

    #[test]
    #[allow(clippy::too_many_lines)] // a linear socket round-trip; splitting hurts readability
    fn tcp_inet4_loopback_roundtrip() {
        let (k, mut mem, mut v, mut cx) = setup();
        let addr = 0x1_1000;
        write_sockaddr_in(&mut mem, addr, [127, 0, 0, 1], 9000);
        let alen = 16u64;

        let srv = call(&k, &mut cx, &mut mem, &mut v, Sysno::Socket, [2, 1, 0, 0, 0, 0]) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Bind,
                [srv, addr, alen, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Listen,
                [srv, 8, 0, 0, 0, 0]
            ),
            0
        );

        let cli = call(&k, &mut cx, &mut mem, &mut v, Sysno::Socket, [2, 1, 0, 0, 0, 0]) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Connect,
                [cli, addr, alen, 0, 0, 0]
            ),
            0
        );
        let acc = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Accept4,
            [srv, 0, 0, 0, 0, 0],
        );
        assert!(acc >= 3, "accept returned a fd");
        let acc = acc as u64;

        let msg = 0x1_1200;
        let out = 0x1_1300;
        mem.write_init(msg, b"ping").unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [cli, msg, 4, 0, 0, 0]
            ),
            4
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Read,
                [acc, out, 4, 0, 0, 0]
            ),
            4
        );
        assert_eq!(mem.read_vec(out, 4).unwrap(), b"ping");
        mem.write_init(msg, b"pong").unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [acc, msg, 4, 0, 0, 0]
            ),
            4
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Read,
                [cli, out, 4, 0, 0, 0]
            ),
            4
        );
        assert_eq!(mem.read_vec(out, 4).unwrap(), b"pong");

        // getpeername on the client reports the server's bound port; getsockname
        // on the accepted end reports the same port back.
        let peer = 0x1_1400;
        let peerlen = 0x1_1500;
        mem.write_init(peerlen, &16u32.to_le_bytes()).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Getpeername,
                [cli, peer, peerlen, 0, 0, 0]
            ),
            0
        );
        assert_eq!(read_port(&mem, peer), 9000);
        assert_eq!(mem.read_vec(peer, 8).unwrap()[4..8], [127, 0, 0, 1]);

        mem.write_init(peerlen, &16u32.to_le_bytes()).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Getsockname,
                [acc, peer, peerlen, 0, 0, 0]
            ),
            0
        );
        assert_eq!(read_port(&mem, peer), 9000);
    }

    #[test]
    fn tcp_inet6_loopback_roundtrip() {
        let (k, mut mem, mut v, mut cx) = setup();
        let addr = 0x1_1000;
        let mut ip = [0u8; 16];
        ip[15] = 1; // ::1
        write_sockaddr_in6(&mut mem, addr, ip, 9700);
        let alen = 28u64;

        let srv = call(&k, &mut cx, &mut mem, &mut v, Sysno::Socket, [10, 1, 0, 0, 0, 0]) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Bind,
                [srv, addr, alen, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Listen,
                [srv, 8, 0, 0, 0, 0]
            ),
            0
        );

        let cli = call(&k, &mut cx, &mut mem, &mut v, Sysno::Socket, [10, 1, 0, 0, 0, 0]) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Connect,
                [cli, addr, alen, 0, 0, 0]
            ),
            0
        );
        let acc = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Accept4,
            [srv, 0, 0, 0, 0, 0],
        );
        assert!(acc >= 3, "accept returned a fd");
        let acc = acc as u64;

        let msg = 0x1_1200;
        let out = 0x1_1300;
        mem.write_init(msg, b"v6ok").unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [cli, msg, 4, 0, 0, 0]
            ),
            4
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Read,
                [acc, out, 4, 0, 0, 0]
            ),
            4
        );
        assert_eq!(mem.read_vec(out, 4).unwrap(), b"v6ok");
    }

    #[test]
    fn ephemeral_port_via_getsockname() {
        let (k, mut mem, mut v, mut cx) = setup();
        let addr = 0x1_1000;
        write_sockaddr_in(&mut mem, addr, [127, 0, 0, 1], 0); // port 0: auto-assign
        let s = call(&k, &mut cx, &mut mem, &mut v, Sysno::Socket, [2, 1, 0, 0, 0, 0]) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Bind,
                [s, addr, 16, 0, 0, 0]
            ),
            0
        );

        let name = 0x1_1200;
        let namelen = 0x1_1300;
        mem.write_init(namelen, &16u32.to_le_bytes()).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Getsockname,
                [s, name, namelen, 0, 0, 0]
            ),
            0
        );
        assert!(read_port(&mem, name) >= 32_768);
    }

    #[test]
    fn udp_connected_roundtrip_via_dispatch() {
        let (k, mut mem, mut v, mut cx) = setup();
        let a_addr = 0x1_1000;
        write_sockaddr_in(&mut mem, a_addr, [127, 0, 0, 1], 9300);
        let b_addr = 0x1_1100;
        write_sockaddr_in(&mut mem, b_addr, [127, 0, 0, 1], 9400);

        let a = call(&k, &mut cx, &mut mem, &mut v, Sysno::Socket, [2, 2, 0, 0, 0, 0]) as u64;
        let b = call(&k, &mut cx, &mut mem, &mut v, Sysno::Socket, [2, 2, 0, 0, 0, 0]) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Bind,
                [a, a_addr, 16, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Bind,
                [b, b_addr, 16, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Connect,
                [a, b_addr, 16, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Connect,
                [b, a_addr, 16, 0, 0, 0]
            ),
            0
        );

        let msg = 0x1_1200;
        let out = 0x1_1300;
        mem.write_init(msg, b"hi").unwrap();
        assert_eq!(
            call(&k, &mut cx, &mut mem, &mut v, Sysno::Write, [a, msg, 2, 0, 0, 0]),
            2
        );
        assert_eq!(
            call(&k, &mut cx, &mut mem, &mut v, Sysno::Read, [b, out, 2, 0, 0, 0]),
            2
        );
        assert_eq!(mem.read_vec(out, 2).unwrap(), b"hi");
    }

    #[test]
    fn udp_sendto_recvfrom_with_source_addr() {
        let (k, mut mem, _v, mut cx) = setup();
        let a_addr = 0x1_1000;
        write_sockaddr_in(&mut mem, a_addr, [127, 0, 0, 1], 9100);
        let b_addr = 0x1_1100;
        write_sockaddr_in(&mut mem, b_addr, [127, 0, 0, 1], 9200);

        let a = k.sys_socket(&mut k.net.lock().unwrap(), &mut cx, 2, 2, 0) as u64; // AF_INET, SOCK_DGRAM
        let b = k.sys_socket(&mut k.net.lock().unwrap(), &mut cx, 2, 2, 0) as u64;
        assert_eq!(k.sys_bind(&mut k.net.lock().unwrap(), &mut cx, a, a_addr, 16, &mem), 0);
        assert_eq!(k.sys_bind(&mut k.net.lock().unwrap(), &mut cx, b, b_addr, 16, &mem), 0);

        let msg = 0x1_1200;
        mem.write_init(msg, b"hello").unwrap();
        assert_eq!(k.sys_sendto(&mut k.net.lock().unwrap(), &mut cx, a, msg, 5, 0, b_addr, 16, &mem), 5);

        let out = 0x1_1300;
        let src = 0x1_1400;
        let srclen = 0x1_1500;
        mem.write_init(srclen, &16u32.to_le_bytes()).unwrap();
        assert_eq!(k.sys_recvfrom(&mut k.net.lock().unwrap(), &mut cx, b, out, 5, 0, src, srclen, &mut mem), 5);
        assert_eq!(mem.read_vec(out, 5).unwrap(), b"hello");
        assert_eq!(read_port(&mem, src), 9100); // source is A's bound port
        assert_eq!(mem.read_vec(src, 8).unwrap()[4..8], [127, 0, 0, 1]);
    }

    #[test]
    fn setsockopt_reuseaddr_allows_rebind() {
        let (k, mut mem, _v, mut cx) = setup();
        let addr = 0x1_1000;
        write_sockaddr_in(&mut mem, addr, [127, 0, 0, 1], 9500);

        let a = k.sys_socket(&mut k.net.lock().unwrap(), &mut cx, 2, 1, 0) as u64;
        assert_eq!(k.sys_bind(&mut k.net.lock().unwrap(), &mut cx, a, addr, 16, &mem), 0);

        let b = k.sys_socket(&mut k.net.lock().unwrap(), &mut cx, 2, 1, 0) as u64;
        // Without SO_REUSEADDR, binding the same port fails.
        assert_eq!(k.sys_bind(&mut k.net.lock().unwrap(), &mut cx, b, addr, 16, &mem), -i64::from(Errno::EINVAL.0));

        // Setting SO_REUSEADDR=1 on b lets the rebind through.
        let optval = 0x1_1600;
        mem.write_init(optval, &1u32.to_le_bytes()).unwrap();
        assert_eq!(
            k.sys_setsockopt(&mut k.net.lock().unwrap(), &mut cx, b, SOL_SOCKET, SO_REUSEADDR, optval, 4, &mem),
            0
        );
        assert_eq!(k.sys_bind(&mut k.net.lock().unwrap(), &mut cx, b, addr, 16, &mem), 0);
    }

    #[test]
    fn accept4_nonblocking_returns_eagain() {
        let (k, mut mem, mut v, mut cx) = setup();
        let addr = 0x1_1000;
        write_sockaddr_in(&mut mem, addr, [127, 0, 0, 1], 9600);
        let srv = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Socket,
            [2, 1 | SOCK_NONBLOCK, 0, 0, 0, 0],
        ) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Bind,
                [srv, addr, 16, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Listen,
                [srv, 8, 0, 0, 0, 0]
            ),
            0
        );
        let ret = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Accept4,
            [srv, 0, 0, 0, 0, 0],
        );
        assert_eq!(ret, -i64::from(Errno::EAGAIN.0));
        assert!(!cx.block);
    }

    #[test]
    fn setsockopt_getsockopt_rcvbuf_and_reuseaddr_roundtrip() {
        let (k, mut mem, _v, mut cx) = setup();
        let s = k.sys_socket(&mut k.net.lock().unwrap(), &mut cx, 2, 1, 0) as u64; // AF_INET, SOCK_STREAM

        let optval = 0x1_1000;
        mem.write_init(optval, &65_536u32.to_le_bytes()).unwrap();
        assert_eq!(
            k.sys_setsockopt(&mut k.net.lock().unwrap(), &mut cx, s, SOL_SOCKET, SO_RCVBUF, optval, 4, &mem),
            0
        );
        mem.write_init(optval, &1u32.to_le_bytes()).unwrap();
        assert_eq!(
            k.sys_setsockopt(&mut k.net.lock().unwrap(), &mut cx, s, SOL_SOCKET, SO_REUSEADDR, optval, 4, &mem),
            0
        );

        let out = 0x1_1100;
        let outlen = 0x1_1200;
        mem.write_init(outlen, &4u32.to_le_bytes()).unwrap();
        assert_eq!(
            k.sys_getsockopt(&mut k.net.lock().unwrap(), &mut cx, s, SOL_SOCKET, SO_RCVBUF, out, outlen, &mut mem),
            0
        );
        assert_eq!(mem.read_u32(out).unwrap(), 65_536);

        mem.write_init(outlen, &4u32.to_le_bytes()).unwrap();
        assert_eq!(
            k.sys_getsockopt(&mut k.net.lock().unwrap(), &mut cx, s, SOL_SOCKET, SO_REUSEADDR, out, outlen, &mut mem),
            0
        );
        assert_eq!(mem.read_u32(out).unwrap(), 1);
    }

    #[test]
    fn so_acceptconn_is_one_after_listen() {
        let (k, mut mem, mut v, mut cx) = setup();
        let addr = 0x1_1000;
        write_sockaddr_in(&mut mem, addr, [127, 0, 0, 1], 9800);
        let srv = call(&k, &mut cx, &mut mem, &mut v, Sysno::Socket, [2, 1, 0, 0, 0, 0]) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Bind,
                [srv, addr, 16, 0, 0, 0]
            ),
            0
        );

        let out = 0x1_1100;
        let outlen = 0x1_1200;
        mem.write_init(outlen, &4u32.to_le_bytes()).unwrap();
        assert_eq!(
            k.sys_getsockopt(&mut k.net.lock().unwrap(), &mut cx, srv, SOL_SOCKET, SO_ACCEPTCONN, out, outlen, &mut mem),
            0
        );
        assert_eq!(mem.read_u32(out).unwrap(), 0, "not listening yet");

        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Listen,
                [srv, 8, 0, 0, 0, 0]
            ),
            0
        );

        mem.write_init(outlen, &4u32.to_le_bytes()).unwrap();
        assert_eq!(
            k.sys_getsockopt(&mut k.net.lock().unwrap(), &mut cx, srv, SOL_SOCKET, SO_ACCEPTCONN, out, outlen, &mut mem),
            0
        );
        assert_eq!(mem.read_u32(out).unwrap(), 1, "listening");
    }

    #[test]
    fn msg_peek_returns_same_bytes_twice() {
        let (k, mut mem, mut v, mut cx) = setup();
        let sv = 0x1_0000;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Socketpair,
                [1, 1, 0, sv, 0, 0]
            ),
            0
        );
        let a = u64::from(mem.read_u32(sv).unwrap());
        let b = u64::from(mem.read_u32(sv + 4).unwrap());

        let msg = 0x1_1000;
        mem.write_init(msg, b"peekme").unwrap();
        assert_eq!(
            call(&k, &mut cx, &mut mem, &mut v, Sysno::Write, [a, msg, 6, 0, 0, 0]),
            6
        );

        let out = 0x1_2000;
        // Two MSG_PEEK reads in a row see the same bytes: nothing is consumed.
        assert_eq!(k.sys_recvfrom(&mut k.net.lock().unwrap(), &mut cx, b, out, 6, MSG_PEEK, 0, 0, &mut mem), 6);
        assert_eq!(mem.read_vec(out, 6).unwrap(), b"peekme");
        assert_eq!(k.sys_recvfrom(&mut k.net.lock().unwrap(), &mut cx, b, out, 6, MSG_PEEK, 0, 0, &mut mem), 6);
        assert_eq!(mem.read_vec(out, 6).unwrap(), b"peekme");

        // A real (non-peek) read now drains it...
        assert_eq!(
            call(&k, &mut cx, &mut mem, &mut v, Sysno::Read, [b, out, 6, 0, 0, 0]),
            6
        );
        assert_eq!(mem.read_vec(out, 6).unwrap(), b"peekme");
        // ...so a further read blocks (the peer end is still open).
        assert_eq!(
            call(&k, &mut cx, &mut mem, &mut v, Sysno::Read, [b, out, 6, 0, 0, 0]),
            0
        );
        assert!(cx.block);
    }

    #[test]
    fn af_unix_abstract_namespace_bind_connect_exchange() {
        let (k, mut mem, mut v, mut cx) = setup();
        let addr = 0x1_1000;
        mem.write_init(addr, &1u16.to_le_bytes()).unwrap(); // AF_UNIX
        // sun_path = "\0nixvm": a leading NUL marks an abstract-namespace name.
        mem.write_init(addr + 2, b"\0nixvm").unwrap();
        let alen = 2 + 6;

        let srv = call(&k, &mut cx, &mut mem, &mut v, Sysno::Socket, [1, 1, 0, 0, 0, 0]) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Bind,
                [srv, addr, alen, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Listen,
                [srv, 8, 0, 0, 0, 0]
            ),
            0
        );

        let cli = call(&k, &mut cx, &mut mem, &mut v, Sysno::Socket, [1, 1, 0, 0, 0, 0]) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Connect,
                [cli, addr, alen, 0, 0, 0]
            ),
            0
        );
        let acc = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Accept4,
            [srv, 0, 0, 0, 0, 0],
        );
        assert!(acc >= 3, "accept returned a fd");
        let acc = acc as u64;

        let msg = 0x1_2000;
        let out = 0x1_3000;
        mem.write_init(msg, b"hi").unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [cli, msg, 2, 0, 0, 0]
            ),
            2
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Read,
                [acc, out, 2, 0, 0, 0]
            ),
            2
        );
        assert_eq!(mem.read_vec(out, 2).unwrap(), b"hi");
    }

    #[test]
    fn shutdown_wr_then_peer_read_sees_eof() {
        const SHUT_WR: u64 = 1;
        let (k, mut mem, mut v, mut cx) = setup();
        let sv = 0x1_0000;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Socketpair,
                [1, 1, 0, sv, 0, 0]
            ),
            0
        );
        let a = u64::from(mem.read_u32(sv).unwrap());
        let b = u64::from(mem.read_u32(sv + 4).unwrap());

        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Shutdown,
                [a, SHUT_WR, 0, 0, 0, 0]
            ),
            0
        );

        // b's read sees immediate EOF (0), not a block, even though a's fd is
        // still open (only its write side was shut down).
        let out = 0x1_1000;
        assert_eq!(
            call(&k, &mut cx, &mut mem, &mut v, Sysno::Read, [b, out, 4, 0, 0, 0]),
            0
        );
        assert!(!cx.block);

        // a itself can no longer write.
        let msg = 0x1_2000;
        mem.write_init(msg, b"x").unwrap();
        let ret = call(&k, &mut cx, &mut mem, &mut v, Sysno::Write, [a, msg, 1, 0, 0, 0]);
        assert_eq!(ret, -i64::from(Errno::EPIPE.0));
    }

    #[test]
    fn getpeername_on_unconnected_returns_enotconn() {
        let (k, mut mem, mut v, mut cx) = setup();
        let s = call(&k, &mut cx, &mut mem, &mut v, Sysno::Socket, [2, 1, 0, 0, 0, 0]) as u64;
        let peer = 0x1_1000;
        let peerlen = 0x1_1100;
        mem.write_init(peerlen, &16u32.to_le_bytes()).unwrap();
        let ret = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Getpeername,
            [s, peer, peerlen, 0, 0, 0],
        );
        assert_eq!(ret, -i64::from(Errno::ENOTCONN.0));
    }

    // --- AF_NETLINK / NETLINK_ROUTE -----------------------------------------

    const NLM_F_REQUEST: u16 = 0x01;

    /// Write a bare `nlmsghdr` (no payload, `nlmsg_len=16`) at `ptr`.
    fn write_nlmsghdr(mem: &mut GuestMemory, ptr: u64, nlmsg_type: u16, flags: u16, seq: u32) {
        let mut b = [0u8; 16];
        b[0..4].copy_from_slice(&16u32.to_le_bytes());
        b[4..6].copy_from_slice(&nlmsg_type.to_le_bytes());
        b[6..8].copy_from_slice(&flags.to_le_bytes());
        b[8..12].copy_from_slice(&seq.to_le_bytes());
        mem.write_init(ptr, &b).unwrap();
    }

    /// Walk a buffer of `NLMSG_ALIGN`-ed `nlmsghdr` messages, asserting each
    /// one's alignment along the way, and return `(nlmsg_type, nlmsg_seq,
    /// payload)` per message.
    fn parse_nlmsgs(buf: &[u8]) -> Vec<(u16, u32, Vec<u8>)> {
        let mut out = Vec::new();
        let mut off = 0usize;
        while off + 16 <= buf.len() {
            let len = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()) as usize;
            assert!(len >= 16, "nlmsg_len must cover at least the header");
            let ty = u16::from_le_bytes(buf[off + 4..off + 6].try_into().unwrap());
            let seq = u32::from_le_bytes(buf[off + 8..off + 12].try_into().unwrap());
            let payload = buf[off + 16..off + len].to_vec();
            out.push((ty, seq, payload));
            let aligned = (len + 3) & !3;
            assert_eq!(aligned % 4, 0, "NLMSG_ALIGN must land on a 4-byte boundary");
            off += aligned;
        }
        assert_eq!(off, buf.len(), "messages must exactly tile the buffer");
        out
    }

    /// Find `rta_type`'s data within an rtattr stream starting at
    /// `payload[hdr_len..]`, asserting `RTA_ALIGN` along the way.
    fn find_rtattr(payload: &[u8], hdr_len: usize, rta_type: u16) -> Option<Vec<u8>> {
        let mut off = hdr_len;
        while off + 4 <= payload.len() {
            let len = u16::from_le_bytes(payload[off..off + 2].try_into().unwrap()) as usize;
            assert!(len >= 4, "rtattr len must cover at least its own header");
            let ty = u16::from_le_bytes(payload[off + 2..off + 4].try_into().unwrap());
            let data = payload[off + 4..off + len].to_vec();
            let aligned = (len + 3) & !3;
            assert_eq!(aligned % 4, 0, "RTA_ALIGN must land on a 4-byte boundary");
            if ty == rta_type {
                return Some(data);
            }
            off += aligned;
        }
        None
    }

    #[test]
    fn netlink_getlink_dump_reports_lo() {
        let (k, mut mem, mut v, mut cx) = setup();
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Socket,
            [
                u64::from(AF_NETLINK),
                SOCK_RAW | SOCK_NONBLOCK,
                NETLINK_ROUTE,
                0,
                0,
                0,
            ],
        ) as u64;
        assert!(fd >= 3);

        let req = 0x1_1000;
        write_nlmsghdr(&mut mem, req, RTM_GETLINK, NLM_F_REQUEST | NLM_F_DUMP, 42);
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [fd, req, 16, 0, 0, 0]
            ),
            16
        );

        let out = 0x1_2000;
        let n = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Read,
            [fd, out, 2048, 0, 0, 0],
        );
        assert!(n > 0, "expected a reply");
        let buf = mem.read_vec(out, n as usize).unwrap();
        let msgs = parse_nlmsgs(&buf);
        assert_eq!(msgs.len(), 2, "one RTM_NEWLINK, then NLMSG_DONE");

        let (ty, seq, payload) = &msgs[0];
        assert_eq!(*ty, RTM_NEWLINK);
        assert_eq!(*seq, 42);
        let ifname = find_rtattr(payload, 16, IFLA_IFNAME).expect("IFLA_IFNAME present");
        assert_eq!(ifname, b"lo\0");

        let (ty, seq, _) = &msgs[1];
        assert_eq!(*ty, NLMSG_DONE);
        assert_eq!(*seq, 42);
    }

    #[test]
    fn netlink_getaddr_dump_reports_127_0_0_1() {
        let (k, mut mem, mut v, mut cx) = setup();
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Socket,
            [
                u64::from(AF_NETLINK),
                SOCK_RAW | SOCK_NONBLOCK,
                NETLINK_ROUTE,
                0,
                0,
                0,
            ],
        ) as u64;

        let req = 0x1_1000;
        write_nlmsghdr(&mut mem, req, RTM_GETADDR, NLM_F_REQUEST | NLM_F_DUMP, 7);
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Sendto,
                [fd, req, 16, 0, 0, 0]
            ),
            16
        );

        let out = 0x1_2000;
        let n = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Recvfrom,
            [fd, out, 2048, 0, 0, 0],
        );
        assert!(n > 0, "expected a reply");
        let buf = mem.read_vec(out, n as usize).unwrap();
        let msgs = parse_nlmsgs(&buf);
        assert_eq!(*msgs.last().map(|(ty, ..)| ty).unwrap(), NLMSG_DONE);

        let v4 = msgs
            .iter()
            .find(|(ty, ..)| *ty == RTM_NEWADDR)
            .and_then(|(_, _, payload)| find_rtattr(payload, 8, IFA_LOCAL))
            .expect("an RTM_NEWADDR with IFA_LOCAL");
        assert_eq!(v4, [127, 0, 0, 1]);
    }

    #[test]
    fn netlink_unknown_type_yields_nlmsg_error() {
        let (k, mut mem, mut v, mut cx) = setup();
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Socket,
            [
                u64::from(AF_NETLINK),
                SOCK_RAW | SOCK_NONBLOCK,
                NETLINK_ROUTE,
                0,
                0,
                0,
            ],
        ) as u64;

        let req = 0x1_1000;
        write_nlmsghdr(&mut mem, req, 0xffff, NLM_F_REQUEST, 99);
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [fd, req, 16, 0, 0, 0]
            ),
            16
        );

        let out = 0x1_2000;
        let n = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Read,
            [fd, out, 2048, 0, 0, 0],
        );
        assert!(n > 0);
        let buf = mem.read_vec(out, n as usize).unwrap();
        let msgs = parse_nlmsgs(&buf);
        assert_eq!(msgs.len(), 1);
        let (ty, seq, payload) = &msgs[0];
        assert_eq!(*ty, NLMSG_ERROR);
        assert_eq!(*seq, 99);
        let error = i32::from_le_bytes(payload[0..4].try_into().unwrap());
        assert_eq!(error, -Errno::EOPNOTSUPP.0);
    }

    #[test]
    fn netlink_bind_and_getsockname_roundtrip_pid() {
        let (k, mut mem, mut v, mut cx) = setup();
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Socket,
            [
                u64::from(AF_NETLINK),
                SOCK_RAW | SOCK_NONBLOCK,
                NETLINK_ROUTE,
                0,
                0,
                0,
            ],
        ) as u64;

        let addr = 0x1_1000;
        let mut b = [0u8; 12];
        b[0..2].copy_from_slice(&AF_NETLINK.to_le_bytes());
        b[4..8].copy_from_slice(&4242u32.to_le_bytes());
        mem.write_init(addr, &b).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Bind,
                [fd, addr, 12, 0, 0, 0]
            ),
            0
        );

        let name = 0x1_2000;
        let namelen = 0x1_2100;
        mem.write_init(namelen, &12u32.to_le_bytes()).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Getsockname,
                [fd, name, namelen, 0, 0, 0]
            ),
            0
        );
        let b = mem.read_vec(name, 8).unwrap();
        assert_eq!(u16::from_le_bytes([b[0], b[1]]), AF_NETLINK);
        assert_eq!(u32::from_le_bytes([b[4], b[5], b[6], b[7]]), 4242);
    }

    #[test]
    fn socketpair_sendmsg_recvmsg_roundtrip() {
        // A connected UNIX socketpair: send via sendmsg (2 iovecs gathered)
        // and receive via recvmsg (scattered across 2 iovecs). Exercises the
        // msghdr/iovec plumbing apk's HTTP client relies on.
        let (k, mut mem, mut v, mut cx) = setup();
        let fds = 0x1_1000;
        assert_eq!(
            call(&k, &mut cx, &mut mem, &mut v, Sysno::Socketpair, [1, 1, 0, fds, 0, 0]),
            0
        );
        let a = u64::from(mem.read_u32(fds).unwrap());
        let b = u64::from(mem.read_u32(fds + 4).unwrap());

        // Two source chunks "he" + "llo", gathered by sendmsg.
        let (buf1, buf2) = (0x1_1100u64, 0x1_1110u64);
        mem.write_init(buf1, b"he").unwrap();
        mem.write_init(buf2, b"llo").unwrap();
        let iov_out = 0x1_1200;
        mem.write_init(iov_out, &buf1.to_le_bytes()).unwrap();
        mem.write_init(iov_out + 8, &2u64.to_le_bytes()).unwrap();
        mem.write_init(iov_out + 16, &buf2.to_le_bytes()).unwrap();
        mem.write_init(iov_out + 24, &3u64.to_le_bytes()).unwrap();
        let msg_out = 0x1_1300; // msghdr: name=0, namelen=0, iov, iovlen=2
        mem.write_init(msg_out + 16, &iov_out.to_le_bytes()).unwrap();
        mem.write_init(msg_out + 24, &2u64.to_le_bytes()).unwrap();
        assert_eq!(
            call(&k, &mut cx, &mut mem, &mut v, Sysno::Sendmsg, [a, msg_out, 0, 0, 0, 0]),
            5,
            "sendmsg gathers both iovecs"
        );

        // Receive scattered into a 2-byte then a 4-byte iovec.
        let (rb1, rb2) = (0x1_1400u64, 0x1_1410u64);
        let iov_in = 0x1_1500;
        mem.write_init(iov_in, &rb1.to_le_bytes()).unwrap();
        mem.write_init(iov_in + 8, &2u64.to_le_bytes()).unwrap();
        mem.write_init(iov_in + 16, &rb2.to_le_bytes()).unwrap();
        mem.write_init(iov_in + 24, &4u64.to_le_bytes()).unwrap();
        let msg_in = 0x1_1600;
        mem.write_init(msg_in + 16, &iov_in.to_le_bytes()).unwrap();
        mem.write_init(msg_in + 24, &2u64.to_le_bytes()).unwrap();
        assert_eq!(
            call(&k, &mut cx, &mut mem, &mut v, Sysno::Recvmsg, [b, msg_in, 0, 0, 0, 0]),
            5,
            "recvmsg returns the full message"
        );
        assert_eq!(mem.read_vec(rb1, 2).unwrap(), b"he");
        assert_eq!(mem.read_vec(rb2, 3).unwrap(), b"llo", "scattered across iovecs");
    }

    #[test]
    fn netlink_nonblocking_recv_with_empty_queue_is_eagain() {
        let (k, mut mem, mut v, mut cx) = setup();
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Socket,
            [
                u64::from(AF_NETLINK),
                SOCK_RAW | SOCK_NONBLOCK,
                NETLINK_ROUTE,
                0,
                0,
                0,
            ],
        ) as u64;
        let out = 0x1_1000;
        let ret = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Read,
            [fd, out, 64, 0, 0, 0],
        );
        assert_eq!(ret, -i64::from(Errno::EAGAIN.0));
        assert!(!cx.block);
    }
}
