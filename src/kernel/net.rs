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
//!
//! UDP multicast works between the VM's sockets: a datagram to a group reaches
//! every socket in the network namespace bound to its port that is in the
//! group (see [`Net::multicast_receivers`]). It never leaves the VM, egress or
//! not, so a guest can't reach the host's local network this way.

use std::collections::{BTreeMap, VecDeque};

use crate::abi::errno::Errno;
use crate::vcpu::GuestMemory;

use super::egress::{Egress, HostConn, HostDgram, Link};
use super::{Fd, Kernel, ServiceCtx, err};

/// One received message: `(source address, payload, out `msg_flags`,
/// `SCM_RIGHTS` fds delivered with it)`. Returned by [`Kernel::recv_message`].
type RecvMsg = (Option<Addr>, Vec<u8>, u64, Vec<Fd>);

impl Net {
    pub(super) fn set_egress(&mut self, egress: Box<dyn Egress>) {
        self.egress = Some(egress);
    }

    /// The egress backend's guest interface (`tun0`), while it is up.
    pub(super) fn link(&self) -> Option<Link> {
        self.egress.as_ref().and_then(|e| e.link())
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
    /// Note data moving through `sock` (and, for a connected datagram
    /// socketpair end, into its peer's queue) — see [`Sock::edge_seq`].
    pub(super) fn bump_edge(&mut self, sock: usize) {
        let Some(s) = self.socks.get_mut(sock) else {
            return;
        };
        s.edge_seq += 1;
        if let Kind::Dgram(Dgram {
            pair_peer: Some(p), ..
        }) = s.kind
            && let Some(t) = self.socks.get_mut(p)
        {
            t.edge_seq += 1;
        }
    }

    /// The socket's edge-trigger sequence ([`Sock::edge_seq`]), 0 if gone.
    pub(super) fn edge_seq(&self, sock: usize) -> u64 {
        self.socks.get(sock).map_or(0, |s| s.edge_seq)
    }

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
    /// A nonblocking TCP `connect` that has been started but not yet completed.
    const EINPROGRESS: Errno = Errno(115);
    /// A second nonblocking `connect` while the first is still in progress.
    const EALREADY: Errno = Errno(114);
    /// `connect` on a socket that is already connected.
    const EISCONN: Errno = Errno(106);
    /// A host socket read/write errored (connection reset).
    const ECONNRESET: Errno = Errno(104);
}

/// Shape an inbound ICMP IP packet for ICMP socket `icmp`: a raw IPv4 socket
/// reads the whole packet, raw IPv6 and ping sockets the ICMP message alone;
/// a ping socket only takes echo replies carrying its identifier. `None`
/// drops it (not for this socket, or malformed).
fn shape_icmp(icmp: Icmp, pkt: Vec<u8>) -> Option<Vec<u8>> {
    let v6 = pkt.first()? >> 4 == 6;
    let hl = if v6 {
        40
    } else {
        usize::from(pkt[0] & 0xf) * 4
    };
    let msg = pkt.get(hl..).filter(|m| m.len() >= 8)?;
    if !icmp.raw {
        let reply = if v6 { 129 } else { 0 };
        if msg[0] != reply || msg[4..6] != icmp.ident.to_be_bytes() {
            return None;
        }
    }
    Some(if icmp.raw && !v6 { pkt } else { msg.to_vec() })
}

/// Recompute an ICMPv4 message's checksum in place (RFC 792: over the whole
/// message, checksum field zeroed).
fn set_icmp4_checksum(msg: &mut [u8]) {
    msg[2..4].fill(0);
    let sum = inet_checksum(msg);
    msg[2..4].copy_from_slice(&sum.to_be_bytes());
}

/// The RFC 1071 Internet checksum of `data`.
fn inet_checksum(data: &[u8]) -> u16 {
    let mut sum: u32 = data
        .chunks(2)
        .map(|c| u32::from(u16::from_be_bytes([c[0], *c.get(1).unwrap_or(&0)])))
        .sum();
    while sum >> 16 != 0 {
        sum = (sum & 0xffff) + (sum >> 16);
    }
    !(sum as u16)
}

/// An IP packet carrying ICMP `msg` from local address `a` to itself — how a
/// loopback echo reply arrives (TTL/hop limit 64).
fn loopback_ip_packet(a: InetAddr, msg: &[u8]) -> Vec<u8> {
    let ip = if a.is_any() { loopback_ip(a.v6) } else { a.ip };
    let mut p = Vec::with_capacity(40 + msg.len());
    if a.v6 {
        p.extend_from_slice(&[0x60, 0, 0, 0]);
        p.extend_from_slice(&(msg.len() as u16).to_be_bytes());
        p.extend_from_slice(&[58, 64]);
        p.extend_from_slice(&ip);
        p.extend_from_slice(&ip);
    } else {
        p.extend_from_slice(&[0x45, 0]);
        p.extend_from_slice(&((20 + msg.len()) as u16).to_be_bytes());
        p.extend_from_slice(&[0, 0, 0, 0, 64, 1, 0, 0]);
        p.extend_from_slice(&ip[..4]);
        p.extend_from_slice(&ip[..4]);
        let sum = inet_checksum(&p[..20]);
        p[10..12].copy_from_slice(&sum.to_be_bytes());
    }
    p.extend_from_slice(msg);
    p
}

/// The errno a failed host connect maps to: the ones guest clients branch on
/// (fall back to the next address, retry later), else "connection refused".
fn connect_errno(e: &std::io::Error) -> i64 {
    match e.kind() {
        std::io::ErrorKind::TimedOut => err(Errno::ETIMEDOUT),
        std::io::ErrorKind::NetworkUnreachable | std::io::ErrorKind::HostUnreachable => {
            err(Errno::ENETUNREACH)
        }
        _ => err(Errno::ECONNREFUSED),
    }
}

const AF_UNIX: u16 = 1;
const AF_INET: u16 = 2;
const AF_INET6: u16 = 10;
const AF_NETLINK: u16 = 16;
const SOCK_STREAM: u64 = 1;
const SOCK_DGRAM: u64 = 2;
const SOCK_RAW: u64 = 3;
const SOCK_SEQPACKET: u64 = 5;
const SOCK_NONBLOCK: u64 = 0o4000;
/// `SOCK_CLOEXEC` in a `socket`/`socketpair`/`accept4` type/flags argument.
pub(super) const SOCK_CLOEXEC: u64 = 0o2000000;
const NETLINK_ROUTE: u64 = 0;

const SOL_SOCKET: u64 = 1;
const IPPROTO_IP: u64 = 0;
const IPPROTO_ICMP: u64 = 1;
const IPPROTO_TCP: u64 = 6;
const IPPROTO_ICMPV6: u64 = 58;
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
/// Which of a socket's two timeouts governs a call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SockDirection {
    Receive,
    Send,
}

/// The most iovecs one `sendmsg`/`recvmsg` takes (`UIO_MAXIOV`).
const UIO_MAXIOV: u64 = 1024;
const SO_RCVTIMEO_NEW: u64 = 66;
const SO_SNDTIMEO_NEW: u64 = 67;
const SO_PRIORITY: u64 = 12;
const SO_ACCEPTCONN: u64 = 30;
const SO_PROTOCOL: u64 = 38;
const SO_DOMAIN: u64 = 39;

// Protocol-level option names.
const IP_TOS: u64 = 1;
const TCP_NODELAY: u64 = 1;
const TCP_KEEPIDLE: u64 = 4;
const TCP_KEEPINTVL: u64 = 5;
const TCP_KEEPCNT: u64 = 6;

/// The options that also act on a host connection, applied when it's made if
/// the guest set them first.
const CONN_OPTIONS: &[(u64, u64)] = &[
    (IPPROTO_TCP, TCP_NODELAY),
    (SOL_SOCKET, SO_KEEPALIVE),
    (IPPROTO_TCP, TCP_KEEPIDLE),
    (IPPROTO_TCP, TCP_KEEPINTVL),
    (IPPROTO_TCP, TCP_KEEPCNT),
    (SOL_SOCKET, SO_LINGER),
];

/// What the guest's setting of `(level, optname)` means for a host
/// connection, if it set it and it means anything there. Only an explicit
/// choice counts: a transport's own default stands for a guest that never
/// asked.
fn conn_option(opts: &SockOpts, level: u64, optname: u64) -> Option<super::egress::ConnOption> {
    use super::egress::ConnOption;
    let raw = opts.extra.get(&(level, optname))?;
    let int = || {
        raw.get(..4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
    };
    Some(match (level, optname) {
        (IPPROTO_TCP, TCP_NODELAY) => ConnOption::NoDelay(opts.nodelay),
        (SOL_SOCKET, SO_KEEPALIVE) => ConnOption::KeepAlive(opts.keepalive),
        (IPPROTO_TCP, TCP_KEEPIDLE) => ConnOption::KeepIdle(int()?),
        (IPPROTO_TCP, TCP_KEEPINTVL) => ConnOption::KeepInterval(int()?),
        (IPPROTO_TCP, TCP_KEEPCNT) => ConnOption::KeepCount(int()?),
        (SOL_SOCKET, SO_LINGER) => {
            ConnOption::ResetOnClose(opts.linger_on && opts.linger_secs == 0)
        }
        _ => return None,
    })
}
const IPV6_V6ONLY: u64 = 26;

// `sendto`/`recvfrom` `flags` bits this module honors (linux/socket.h).
const MSG_PEEK: u64 = 0x02;
/// Ancillary data (an `SCM_RIGHTS` cmsg) was too large for the caller's
/// `msg_control` buffer and was truncated — set in `recvmsg`'s out `msg_flags`.
const MSG_CTRUNC: u64 = 0x08;
const MSG_TRUNC: u64 = 0x20;
const MSG_DONTWAIT: u64 = 0x40;
const MSG_WAITALL: u64 = 0x100;
/// `recvmsg` flag: set `FD_CLOEXEC` on every fd received via `SCM_RIGHTS`.
const MSG_CMSG_CLOEXEC: u64 = 0x4000_0000;

/// `cmsg_level`/`cmsg_type` for file-descriptor passing (SOL_SOCKET/SCM_RIGHTS).
const SCM_RIGHTS: i32 = 1;
const SCM_CREDENTIALS: i32 = 2;
const SO_PASSCRED: u64 = 16;
/// Size of a 64-bit `struct cmsghdr`: `cmsg_len`(size_t,8) + `cmsg_level`(int,4)
/// + `cmsg_type`(int,4). `CMSG_DATA` follows immediately (already 8-aligned).
const CMSG_HDR_LEN: usize = 16;
// MSG_NOSIGNAL (0x4000) is a documented no-op: this virtual transport never
// raises SIGPIPE on a peer-closed write in the first place (it returns
// `EPIPE`), so there is nothing to suppress.

// `NETLINK_ROUTE` (rtnetlink) message types this module answers
// (linux/rtnetlink.h). Only enough of the protocol to let guest tools
// enumerate the interfaces (`lo`, plus `tun0` while the egress backend
// provides a link): `RTM_GETLINK`/`RTM_GETADDR`/`RTM_GETROUTE` dumps.
const RTM_NEWLINK: u16 = 16;
const RTM_SETLINK: u16 = 19;
const RTM_GETLINK: u16 = 18;
const RTM_NEWADDR: u16 = 20;
const RTM_GETADDR: u16 = 22;
const RTM_NEWROUTE: u16 = 24;
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
/// `ifinfomsg.ifi_type` for a link without hardware addresses (a tun device).
const ARPHRD_NONE: u16 = 0xfffe;
// `ifinfomsg.ifi_flags` / `IFF_*` bits (linux/if.h).
const IFF_UP: u32 = 0x1;
const IFF_LOOPBACK: u32 = 0x8;
const IFF_POINTOPOINT: u32 = 0x10;
const IFF_RUNNING: u32 = 0x40;
const IFF_NOARP: u32 = 0x80;
const IFF_MULTICAST: u32 = 0x1000;
const IFF_LOWER_UP: u32 = 0x1_0000;

// `IFLA_*` rtattr types (linux/if_link.h) filled in on the `RTM_NEWLINK` reply.
const IFLA_ADDRESS: u16 = 1;
const IFLA_IFNAME: u16 = 3;
const IFLA_MTU: u16 = 4;
const IFLA_QDISC: u16 = 6;
const IFLA_TXQLEN: u16 = 13;
const IFLA_OPERSTATE: u16 = 16;
/// `IFLA_OPERSTATE` values (linux/if.h): `lo` reports "unknown".
const IF_OPER_UNKNOWN: u8 = 0;
const IF_OPER_DOWN: u8 = 2;
const IF_OPER_UP: u8 = 6;

// `RTA_*` rtattr types and `rtmsg` field values (linux/rtnetlink.h) for the
// `RTM_NEWROUTE` replies.
const RTA_DST: u16 = 1;
const RTA_OIF: u16 = 4;
const RTA_PREFSRC: u16 = 7;
const RTA_TABLE: u16 = 15;
const RT_TABLE_MAIN: u8 = 254;
const RTPROT_KERNEL: u8 = 2;
const RTPROT_STATIC: u8 = 4;
const RTN_UNICAST: u8 = 1;
const RT_SCOPE_UNIVERSE: u8 = 0;
const RT_SCOPE_LINK: u8 = 253;

// `IFA_*` rtattr types (linux/if_addr.h) filled in on the `RTM_NEWADDR` reply.
const IFA_ADDRESS: u16 = 1;
const IFA_LOCAL: u16 = 2;
const IFA_LABEL: u16 = 3;
/// `ifaddrmsg.ifa_flags` bit (linux/if_addr.h): a static, not leased, address.
const IFA_F_PERMANENT: u8 = 0x80;
/// `RT_SCOPE_HOST` (linux/rtnetlink.h): the scope of an address that is only
/// valid on this host, e.g. `127.0.0.1`.
const RT_SCOPE_HOST: u8 = 254;
/// The loopback interface's fixed `ifindex`.
const LOOPBACK_IFINDEX: i32 = 1;
/// `tun0`'s `ifindex`, while the egress backend provides a link.
const TUN_IFINDEX: i32 = 2;

/// The kernel's socket table plus the AF_UNIX/AF_INET(6) address registries.
#[derive(Default)]
pub(super) struct Net {
    socks: Vec<Sock>,
    /// AF_UNIX path, or `"tcp4:port"`/`"tcp6:port"` -> listening slot.
    listeners: BTreeMap<String, Vec<usize>>,
    /// Picks the next member of a `SO_REUSEPORT` listener group, round robin.
    next_listener: usize,
    /// `"udp4:port"`/`"udp6:port"` -> bound `Dgram` slot.
    pub(super) dgram_ports: BTreeMap<String, Vec<usize>>,
    /// AF_UNIX filesystem path -> bound datagram slot (e.g. syslog's /dev/log).
    dgram_paths: BTreeMap<String, usize>,
    /// Host-egress backend. `None` = loopback-only (the default, and the only
    /// possible mode with no backend installed); `Some` bridges guest
    /// connections to *routable* addresses onto real host sockets. Installed
    /// by [`Kernel::set_egress`].
    egress: Option<Box<dyn Egress>>,
    /// Last echo identifier handed to a ping socket.
    ping_ident: u16,
    /// Network namespaces whose `lo` has been set up or down. Unlisted, `lo` is
    /// up in the VM's first namespace and down in a new one, as on Linux.
    lo_state: BTreeMap<u32, bool>,
}

impl std::fmt::Debug for Net {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Net")
            .field("socks", &self.socks)
            .field("listeners", &self.listeners)
            .field("next_listener", &self.next_listener)
            .field("dgram_ports", &self.dgram_ports)
            .field("dgram_paths", &self.dgram_paths)
            .field("egress", &self.egress.is_some())
            .field("ping_ident", &self.ping_ident)
            .field("lo_state", &self.lo_state)
            .finish()
    }
}

/// One entry in the socket table.
#[derive(Debug)]
struct Sock {
    /// The network namespace this socket was created in, for good: `setns` moves
    /// the task, not its sockets. Ports and abstract names are per namespace.
    netns: u32,
    /// The routing key of the `AF_UNIX` name this socket is bound to.
    unix_bound_key: Option<String>,
    domain: u16,
    kind: Kind,
    /// `SOCK_NONBLOCK` at creation time (there is no `fcntl(F_SETFL)` wiring
    /// into this module, so this is fixed at `socket()`/`accept4()`).
    nonblock: bool,
    /// `setsockopt`-controlled knobs, `SO_ERROR`, and buffer-size hints.
    opts: SockOpts,
    /// Open fds on this socket while it is not a connected [`Pair`] (which
    /// counts per end): when the last one closes, the socket is released —
    /// a listener frees its port, a host connection is shut down.
    refs: usize,
    /// Bumped whenever data moves through this socket (a host read, a
    /// delivered write). It is part of the socket's edge-triggered epoll
    /// level: Linux raises a fresh `EPOLLET` edge on every arrival, even on an
    /// already-readable socket, and tokio relies on it — after a short read it
    /// assumes the socket is drained and waits for the next edge. A readiness
    /// mask alone stays `POLLIN|POLLOUT` across a read and a new arrival, so
    /// that edge would be lost and the waiter would sleep forever.
    edge_seq: u64,
}

/// The mutable `setsockopt`-controlled state of a socket, plus `SO_ERROR`.
/// [`Kernel::sys_setsockopt`] says which of these change behavior; the rest are
/// stored and echoed back
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
    /// `IPV6_V6ONLY`: off (the default), an `AF_INET6` stream socket listening
    /// on `::` takes IPv4 connections too (see [`Net::pick_dual_listener`]).
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
    /// Every other known option the guest set, by `(level, optname)`, stored
    /// verbatim (bounded) so `getsockopt` echoes it back.
    extra: BTreeMap<(u64, u64), Vec<u8>>,
}

impl SockOpts {
    /// A stored option's value as an integer: an `int`, or the single byte
    /// some IP options accept instead. `None` if it was never set.
    fn int_opt(&self, key: (u64, u64)) -> Option<i32> {
        let raw = self.extra.get(&key)?;
        Some(match raw.len() {
            0 => 0,
            1..=3 => i32::from(raw[0]),
            _ => i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]),
        })
    }
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
            // Stored as the *internal* buffer size; `getsockopt` reports twice
            // this, matching Linux (`net.core.{r,w}mem_default` = 212992, so the
            // internal default is half that). `setsockopt` overwrites it with the
            // caller's requested value, also doubled on readback.
            rcvbuf: 106_496,
            sndbuf: 106_496,
            tos: 0,
            error: 0,
            extra: BTreeMap::new(),
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
        /// Who called `listen()`: what a client connecting here sees as its peer.
        cred: Ucred,
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
    /// A connect whose outcome the guest hasn't observed yet: the transport
    /// returned with the handshake in flight ([`HostConn::poll_connect`]).
    /// Cleared when a (re-trapped or repeated) `connect` or
    /// `getsockopt(SO_ERROR)` reports the result.
    connecting: bool,
    /// When a lingering `close` (a nonzero `SO_LINGER`) gives up waiting for
    /// the peer to acknowledge what was sent, once one has started.
    linger_until: Option<u128>,
}

// Only `peer` is meaningful in a dump; the boxed conn and flags are noise.
#[allow(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for HostSock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostSock")
            .field("peer", &self.peer)
            .finish()
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
    /// Which ends are `AF_INET6` sockets on an IPv4 connection (a dual-stack
    /// listener's, or one that connected to a `::ffff:` address): they see
    /// its IPv4 addresses as IPv4-mapped ones ([`Pair::addr_seen_by`]).
    v6_view: [bool; 2],
    /// In-flight `SCM_RIGHTS` ancillary file descriptors, parallel to `to`:
    /// `scm[e]` rides the byte stream the reader on end `e` consumes. Each entry
    /// is `(pos, fds)` where `pos` is the number of stream bytes still queued
    /// *ahead* of the message these fds were sent with — so they are delivered
    /// with the `recvmsg` that first reads that message's data (Linux attaches
    /// ancillary data to the sending `sendmsg`'s bytes). Each `Fd` here holds a
    /// duplicated backing reference, transferred to a fresh guest fd on receipt
    /// or dropped if the bytes are consumed by a plain `read`/`recv`.
    scm: [VecDeque<(usize, Vec<Fd>)>; 2],
    /// A nonblocking connect on this end said EINPROGRESS and hasn't been
    /// asked again: the next connect() reports it done (0), later ones EISCONN.
    connect_pending: [bool; 2],
    /// The process on each end, for `SO_PEERCRED`: the creator of a
    /// `socketpair`, or the connecting process (end 0) and the listener's
    /// `listen()` caller (end 1), as Linux records them.
    cred: [Ucred; 2],
    /// A `SOCK_SEQPACKET` pair: the lengths of the messages queued in each
    /// `to[e]`, oldest first. Each send is one message, delivered whole or
    /// not at all; a read takes at most one, the rest of it discarded.
    msgs: Option<[VecDeque<usize>; 2]>,
    /// Who last sent toward each end, for `SCM_CREDENTIALS` (`SO_PASSCRED`).
    sender: [Ucred; 2],
}

/// `struct ucred`: a process's global pid, uid and gid when it set up a
/// connection.
#[derive(Debug, Clone, Copy, Default)]
struct Ucred {
    pid: i32,
    uid: u32,
    gid: u32,
}

impl Ucred {
    fn of(cx: &ServiceCtx) -> Self {
        Self {
            pid: cx.cur.tgid,
            uid: cx.cur.creds.euid,
            gid: cx.cur.creds.egid,
        }
    }
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
            v6_view: [false, false],
            scm: [VecDeque::new(), VecDeque::new()],
            connect_pending: [false; 2],
            cred: [Ucred::default(); 2],
            sender: [Ucred::default(); 2],
            msgs: None,
        }
    }

    /// How much of `to[e]` one read of up to `want` bytes takes: for a
    /// `SOCK_SEQPACKET` pair, no more than the next message.
    fn msg_read_len(&self, e: usize, want: usize) -> usize {
        match &self.msgs {
            Some(m) => want.min(m[e].front().copied().unwrap_or(0)),
            None => want,
        }
    }

    /// After a read drained `n` bytes from `to[e]`: a `SOCK_SEQPACKET`
    /// message's unread rest is discarded with it. Returns the `SCM_RIGHTS`
    /// descriptors the read carries off (see [`Self::scm_take`]) and whether
    /// a message was cut short (`MSG_TRUNC`).
    fn finish_read(&mut self, e: usize, n: usize) -> (Vec<Fd>, bool) {
        let Some(m) = &mut self.msgs else {
            return (self.scm_take(e, n), false);
        };
        let len = m[e].pop_front().unwrap_or(n).max(n);
        let rest = (len - n).min(self.to[e].len());
        self.to[e].drain(..rest);
        // Its descriptors ride the message's first byte.
        let fds = self.scm_take(e, n + rest);
        (fds, rest > 0)
    }

    /// End `which`'s address as end `viewer` sees it: an IPv4 address shown to
    /// an `AF_INET6` end is IPv4-mapped (`::ffff:a.b.c.d`), as on Linux.
    fn addr_seen_by(&self, which: usize, viewer: usize) -> Option<InetAddr> {
        let a = self.addrs[which]?;
        Some(if self.v6_view[viewer] && !a.v6 {
            a.mapped()
        } else {
            a
        })
    }

    /// Bytes readable from `to[e]` in one `recvmsg` without merging past an
    /// `SCM_RIGHTS` boundary: Linux does not coalesce an ancillary-bearing
    /// message with a following one. `want` is the caller's desired count.
    fn scm_read_len(&self, e: usize, want: usize) -> usize {
        match self.scm[e].front() {
            None => want,
            // The front message carries fds and starts now — deliver it, but
            // stop before the next ancillary-bearing message.
            Some(&(0, _)) => self.scm[e].get(1).map_or(want, |&(next, _)| want.min(next)),
            // Plain data ahead: read up to the ancillary boundary, no further.
            Some(&(pos, _)) => want.min(pos),
        }
    }

    /// After consuming `n` bytes from `to[e]`, pop and return the `SCM_RIGHTS`
    /// fd groups whose message boundary now falls within the consumed bytes,
    /// and shift the remaining entries' offsets down by `n`.
    fn scm_take(&mut self, e: usize, n: usize) -> Vec<Fd> {
        let mut out = Vec::new();
        let threshold = n.max(1); // n == 0 still delivers a zero-length message
        while self.scm[e].front().is_some_and(|&(pos, _)| pos < threshold) {
            if let Some((_, fds)) = self.scm[e].pop_front() {
                out.extend(fds);
            }
        }
        for (pos, _) in &mut self.scm[e] {
            *pos = pos.saturating_sub(n);
        }
        out
    }
}

/// A `SOCK_DGRAM` endpoint: `local` is the bound (or lazily ephemeral-assigned)
/// address, `peer` is the `connect()`-ed destination (if any), and `queue`
/// holds inbound `(source address, payload)` datagrams awaiting `recv`.
#[derive(Default)]
struct Dgram {
    local: Option<InetAddr>,
    peer: Option<InetAddr>,
    /// The slot index of the connected peer for a `socketpair(AF_UNIX,
    /// SOCK_DGRAM)` pair: a `write`/`send` on this socket delivers a datagram
    /// straight into that slot's `queue`, with no address namespace involved.
    /// `None` for a standalone (named/INET) datagram socket.
    pair_peer: Option<usize>,
    queue: VecDeque<(InetAddr, Vec<u8>)>,
    /// What [`Self::queue`] holds, by [`datagram_cost`].
    queued_bytes: usize,
    /// A host UDP socket, opened lazily the first time this datagram socket
    /// sends to a *routable* address with egress enabled. Once present, all
    /// sends go out through it and inbound host datagrams are drained into
    /// `queue` on recv. `None` = pure in-VM datagram socket.
    host: Option<Box<dyn HostDgram>>,
    /// Set for an ICMP socket (raw, or a `SOCK_DGRAM` "ping" socket): datagrams
    /// are ICMP messages, carried by the egress backend's ICMP endpoint (in
    /// `host`) or answered locally for a loopback echo.
    icmp: Option<Icmp>,
    /// The multicast groups joined (`IP_ADD_MEMBERSHIP`, `IPV6_JOIN_GROUP`,
    /// `MCAST_JOIN_GROUP`), as addresses with port 0.
    groups: Vec<InetAddr>,
}

/// What a queued datagram costs against a socket's receive buffer. The overhead
/// keeps empty datagrams from filling memory unnoticed.
fn datagram_cost(payload: &[u8]) -> usize {
    payload.len() + 256
}

impl Dgram {
    /// Queue an arriving datagram, or drop it if the receive buffer is full:
    /// what UDP does on Linux, where a sender never finds out either.
    fn enqueue(&mut self, from: InetAddr, payload: Vec<u8>) {
        /// Linux's default `net.core.rmem_max`, doubled as it does.
        const CAPACITY: usize = 2 * 212_992;
        let cost = datagram_cost(&payload);
        if self.queued_bytes + cost <= CAPACITY {
            self.queued_bytes += cost;
            self.queue.push_back((from, payload));
        }
    }
}

/// What kind of ICMP socket a [`Dgram`] is.
#[derive(Debug, Clone, Copy)]
struct Icmp {
    /// `SOCK_RAW`: the guest writes whole ICMP messages and reads every ICMP
    /// packet (IPv4 ones with their IP header, as Linux delivers them).
    /// Otherwise a ping socket: only echo requests out — stamped with `ident`
    /// — and only the matching echo replies in, without IP header.
    raw: bool,
    ident: u16,
}

impl std::fmt::Debug for Dgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dgram")
            .field("local", &self.local)
            .field("peer", &self.peer)
            .field("pair_peer", &self.pair_peer)
            .field("queue", &self.queue)
            .field("queued_bytes", &self.queued_bytes)
            .field("host", &self.host.is_some())
            .field("icmp", &self.icmp)
            .field("groups", &self.groups)
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
    /// The IPv4 address an IPv4-mapped IPv6 one (`::ffff:a.b.c.d`) stands for.
    fn unmapped(&self) -> Option<InetAddr> {
        if !self.v6 || self.ip[..10] != [0; 10] || self.ip[10..12] != [0xff, 0xff] {
            return None;
        }
        let mut ip = [0u8; 16];
        ip[..4].copy_from_slice(&self.ip[12..16]);
        Some(InetAddr {
            v6: false,
            port: self.port,
            ip,
        })
    }

    /// This IPv4 address as an IPv4-mapped IPv6 one.
    fn mapped(&self) -> InetAddr {
        let mut ip = [0u8; 16];
        ip[10] = 0xff;
        ip[11] = 0xff;
        ip[12..16].copy_from_slice(&self.ip[..4]);
        InetAddr {
            v6: true,
            port: self.port,
            ip,
        }
    }

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

    /// `224.0.0.0/4` (v4) or `ff00::/8` (v6).
    fn is_multicast(&self) -> bool {
        if self.v6 {
            self.ip[0] == 0xff
        } else {
            self.ip[0] & 0xf0 == 0xe0
        }
    }

    /// The all-hosts group every interface is in without joining:
    /// `224.0.0.1` or `ff02::1`.
    fn is_all_hosts(&self) -> bool {
        let mut all = [0u8; 16];
        if self.v6 {
            all[0] = 0xff;
            all[1] = 0x02;
            all[15] = 1;
        } else {
            all[..4].copy_from_slice(&[224, 0, 0, 1]);
        }
        self.ip == all
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
fn route_key(netns: u32, proto: &str, a: InetAddr) -> String {
    format!("{netns}/{proto}{}:{}", if a.v6 { 6 } else { 4 }, a.port)
}

/// The routing key for an `AF_UNIX` name. Abstract names (leading NUL) belong to
/// the network namespace. A path name is a file, so it's found through the
/// filesystem whatever the namespace.
fn unix_key(netns: u32, name: &str, identity: Option<&String>) -> String {
    if name.starts_with('\0') {
        format!("{netns}/{name}")
    } else {
        identity.map_or_else(|| name.to_string(), Clone::clone)
    }
}

impl Net {
    /// The network namespace socket `fd` belongs to.
    pub(super) fn netns_of(&self, fd: &Fd) -> Option<u32> {
        match *fd {
            Fd::Socket { sock, .. } => self.socks.get(sock).map(|s| s.netns),
            _ => None,
        }
    }

    /// Whether network namespace `ns`'s `lo` is up.
    pub(super) fn lo_up(&self, ns: u32) -> bool {
        self.lo_state.get(&ns).copied().unwrap_or(ns == 0)
    }

    /// What `sock`'s namespace says to a loopback destination while its `lo`
    /// is down: Linux has no route for IPv4 and no source address for IPv6.
    fn loopback_unreachable(&self, sock: usize, dest: &InetAddr) -> Option<i64> {
        if dest.valid_bind() && !self.lo_up(self.socks[sock].netns) {
            Some(err(if dest.v6 {
                Errno::EADDRNOTAVAIL
            } else {
                Errno::ENETUNREACH
            }))
        } else {
            None
        }
    }

    /// The interfaces namespace `ns` sees: its `lo`, and `tun0` only in the
    /// first namespace, while the egress backend provides a link.
    pub(super) fn ifaces_of(&self, ns: u32) -> Vec<Iface> {
        let link = self.link().filter(|_| ns == 0);
        ifaces(link.as_ref(), self.lo_up(ns))
    }

    /// Set namespace `ns`'s `lo` up or down.
    pub(super) fn set_lo(&mut self, ns: u32, up: bool) {
        self.lo_state.insert(ns, up);
    }

    /// What connects socket `sock` to the outside world. Only sockets in the VM's
    /// first network namespace have this: a new namespace is loopback only.
    fn egress_for(&self, sock: usize) -> Option<&dyn Egress> {
        if self.socks[sock].netns != 0 {
            return None;
        }
        self.egress.as_deref()
    }

    /// Adjust the open-fd refcount of the socket end `fd` refers to (a no-op for
    /// non-socket fds or unconnected sockets). Mirrors `Kernel::bump_pipe`.
    pub(super) fn bump(&mut self, fd: &Fd, inc: bool) {
        let Fd::Socket { sock, end } = *fd else {
            return;
        };
        let Some(s) = self.socks.get_mut(sock) else {
            return;
        };
        let n = match &mut s.kind {
            Kind::Pair(p) => &mut p.refs[end],
            _ => &mut s.refs,
        };
        if inc {
            *n += 1;
        } else {
            *n = n.saturating_sub(1);
            if s.refs == 0 && !matches!(s.kind, Kind::Pair(_)) {
                self.release(sock);
            }
        }
    }

    /// One step of a lingering `close` of `fd` at time `now`: `Some(when)` to
    /// wait and try again then, `None` to close now. Only the last fd on a
    /// connection to the host with a nonzero `SO_LINGER` waits: the first step
    /// sends the FIN, and the wait ends once the peer has acknowledged
    /// everything (as Linux's waits for the FIN's ack) or the timeout passes.
    /// A negative timeout is as good as forever, as on Linux.
    pub(super) fn linger_step(&mut self, fd: &Fd, now: u128) -> Option<u128> {
        const POLL_NS: u128 = 5_000_000;
        let Fd::Socket { sock, .. } = *fd else {
            return None;
        };
        let s = self.socks.get_mut(sock)?;
        let (on, secs) = (s.opts.linger_on, s.opts.linger_secs);
        let Kind::Host(h) = &mut s.kind else {
            return None;
        };
        if s.refs != 1 || !on || secs == 0 || h.connecting {
            return None;
        }
        let until = *h
            .linger_until
            .get_or_insert_with(|| now + u128::from(secs.cast_unsigned()) * 1_000_000_000);
        if !h.wr_shut {
            h.conn.shutdown_write();
            h.wr_shut = true;
        }
        if now >= until || h.conn.unsent().is_none_or(|n| n == 0) {
            return None;
        }
        Some(until.min(now + POLL_NS))
    }

    /// The last fd on (non-pair) socket `sock` closed: free what it holds. A
    /// listener leaves the port table (pending connections see the server
    /// end closed), a host connection is shut down and dropped, a datagram
    /// socket frees its port.
    fn release(&mut self, sock: usize) {
        let old = std::mem::replace(&mut self.socks[sock].kind, Kind::Idle { bound: None });
        match old {
            Kind::Listener { backlog, .. } => {
                self.listeners.retain(|_, group| {
                    group.retain(|&i| i != sock);
                    !group.is_empty()
                });
                for pending in backlog {
                    if let Some(Kind::Pair(p)) = self.socks.get_mut(pending).map(|s| &mut s.kind) {
                        p.refs[1] = 0;
                    }
                }
            }
            // A zero SO_LINGER asked for a reset instead: dropping the connection
            // without a clean shutdown first is what sends it.
            Kind::Host(mut h) if !h.wr_shut => {
                let opts = &self.socks[sock].opts;
                if !(opts.linger_on && opts.linger_secs == 0) {
                    h.conn.shutdown_write();
                }
            }
            Kind::Dgram(_) => self.dgram_ports.retain(|_, group| {
                group.retain(|&i| i != sock);
                !group.is_empty()
            }),
            _ => {}
        }
    }

    /// True if some *other* socket already occupies `proto`/`a`'s port.
    fn addr_in_use(&self, netns: u32, proto: &str, a: InetAddr, exclude: usize) -> bool {
        let key = route_key(netns, proto, a);
        // `exclude` is out of range when probing for a free ephemeral port, on
        // behalf of no socket: then nothing has opted in to sharing.
        let (my_reuseaddr, my_reuseport) = self
            .socks
            .get(exclude)
            .map_or((false, false), |s| (s.opts.reuseaddr, s.opts.reuseport));
        self.socks.iter().enumerate().any(|(i, s)| {
            if i == exclude {
                return false;
            }
            let (same_addr, listening) = match &s.kind {
                Kind::Idle {
                    bound: Some(Addr::Inet(b)),
                } if proto == "tcp" => (route_key(s.netns, "tcp", *b) == key, false),
                Kind::Listener {
                    addr: Some(Addr::Inet(b)),
                    ..
                } if proto == "tcp" => (route_key(s.netns, "tcp", *b) == key, true),
                Kind::Dgram(d) if proto == "udp" => (
                    d.local.is_some_and(|b| route_key(s.netns, "udp", b) == key),
                    false,
                ),
                _ => (false, false),
            };
            // Linux's rule: both sockets must have opted in. SO_REUSEPORT shares
            // a port outright; SO_REUSEADDR never shares one with a listener.
            same_addr
                && !(my_reuseport && s.opts.reuseport)
                && !(my_reuseaddr && s.opts.reuseaddr && !listening)
        })
    }

    /// Whether another socket already listens on `a`'s port in a way that
    /// stops `sock` from listening there too.
    fn listen_conflict(&self, a: InetAddr, sock: usize) -> bool {
        let reuseport = self.socks[sock].opts.reuseport;
        self.listeners
            .get(&route_key(self.socks[sock].netns, "tcp", a))
            .is_some_and(|group| {
                group
                    .iter()
                    .any(|&i| i != sock && !(reuseport && self.socks[i].opts.reuseport))
            })
    }

    /// The listener a new connection to `key` goes to: the next in turn if a
    /// `SO_REUSEPORT` group is sharing the port.
    fn pick_listener(&mut self, key: &str) -> Option<usize> {
        let group = self.listeners.get(key)?;
        let chosen = *group.get(self.next_listener % group.len())?;
        self.next_listener = self.next_listener.wrapping_add(1);
        Some(chosen)
    }

    /// The dual-stack listener an IPv4 connection to `a` (in `netns`) goes to
    /// when no IPv4 socket listens on its port: an `AF_INET6` one bound to
    /// `::` without `IPV6_V6ONLY` takes IPv4 connections too, as on Linux.
    fn pick_dual_listener(&mut self, netns: u32, a: InetAddr) -> Option<usize> {
        let key6 = route_key(netns, "tcp", InetAddr { v6: true, ..a });
        let group: Vec<usize> = self
            .listeners
            .get(&key6)?
            .iter()
            .copied()
            .filter(|&i| {
                !self.socks[i].opts.v6only
                    && matches!(&self.socks[i].kind, Kind::Listener { addr: Some(Addr::Inet(b)), .. } if b.is_any())
            })
            .collect();
        let chosen = *group.get(self.next_listener % group.len().max(1))?;
        self.next_listener = self.next_listener.wrapping_add(1);
        Some(chosen)
    }

    /// The socket a datagram from `src_port` to `key` goes to. A `SO_REUSEPORT`
    /// group spreads by source, so one flow stays on one socket; otherwise the
    /// most recent bind wins, as it does with `SO_REUSEADDR` on Linux.
    fn pick_dgram_target(&self, key: &str, src_port: u16) -> Option<usize> {
        // A socket bound to a multicast group's address takes only that
        // group's datagrams.
        let group: Vec<usize> = self
            .dgram_ports
            .get(key)?
            .iter()
            .copied()
            .filter(|&i| {
                !matches!(&self.socks[i].kind, Kind::Dgram(Dgram { local: Some(l), .. }) if l.is_multicast())
            })
            .collect();
        if group.is_empty() {
            return None;
        }
        if group.iter().all(|&i| self.socks[i].opts.reuseport) {
            group.get(usize::from(src_port) % group.len()).copied()
        } else {
            group.last().copied()
        }
    }

    /// The sockets in `sock`'s network namespace a datagram to multicast group
    /// `dest` reaches: bound to its port, at the wildcard address or the
    /// group's own, and in the group. A socket is in it if it joined it, or,
    /// with `IP_MULTICAST_ALL` (on by default, as on Linux), if any socket in
    /// the namespace did; everything is in the all-hosts group.
    fn multicast_receivers(&self, sock: usize, dest: InetAddr) -> Vec<usize> {
        let netns = self.socks[sock].netns;
        let group = InetAddr { port: 0, ..dest };
        let joined =
            |i: usize| matches!(&self.socks[i].kind, Kind::Dgram(d) if d.groups.contains(&group));
        let anyone_joined = dest.is_all_hosts()
            || self
                .socks
                .iter()
                .enumerate()
                .any(|(i, s)| s.netns == netns && joined(i));
        let Some(bound) = self.dgram_ports.get(&route_key(netns, "udp", dest)) else {
            return Vec::new();
        };
        bound
            .iter()
            .copied()
            .filter(|&i| {
                let Kind::Dgram(d) = &self.socks[i].kind else {
                    return false;
                };
                let at = d.local.is_some_and(|l| l.is_any() || l.ip == dest.ip);
                let all = self.socks[i]
                    .opts
                    .int_opt(if dest.v6 { (41, 29) } else { (0, 49) })
                    .unwrap_or(1)
                    != 0;
                at && (joined(i) || dest.is_all_hosts() || (all && anyone_joined))
            })
            .collect()
    }

    /// The lowest free port >= 32768 for `proto`/`v6` (the standard Linux
    /// ephemeral range, trimmed for a small in-VM table).
    fn ephemeral_port(&self, netns: u32, proto: &str, v6: bool) -> u16 {
        for port in 32_768u32..=60_999 {
            let a = InetAddr {
                v6,
                port: port as u16,
                ip: [0; 16],
            };
            if !self.addr_in_use(netns, proto, a, usize::MAX) {
                return port as u16;
            }
        }
        0
    }

    /// A fresh client-side local address: loopback of `v6`'s family, with a
    /// freshly allocated ephemeral TCP port.
    fn fresh_local(&self, netns: u32, v6: bool) -> InetAddr {
        InetAddr {
            v6,
            port: self.ephemeral_port(netns, "tcp", v6),
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

    /// Adjust the backing refcount of an in-flight `SCM_RIGHTS` fd. `net` is
    /// already held here, so a socket fd is bumped directly (via `net.bump`);
    /// routing it through `bump_pipe` would re-lock `net` and deadlock. Pipe
    /// (and any other refcounted) fds go through the shared `bump_pipe`, which
    /// only takes `pipes` — legal after `net` in the lock order.
    fn scm_ref(&self, net: &mut Net, fd: &Fd, inc: bool) {
        match fd {
            Fd::Socket { .. } => net.bump(fd, inc),
            other => self.bump_pipe(other, inc),
        }
    }

    /// Send a datagram to multicast group `dest`: it reaches the group's
    /// members in this network namespace, the sender included, unless the
    /// sender turned `IP_MULTICAST_LOOP` off. It never leaves the VM, even with
    /// egress: a guest doesn't get to talk to the host's local network.
    fn multicast_send(&self, net: &mut Net, sock: usize, data: &[u8], dest: InetAddr) -> i64 {
        if !net.lo_up(net.socks[sock].netns) {
            return err(Errno::ENETUNREACH);
        }
        let mut src = self.ensure_dgram_bound(net, sock);
        if src.is_any() || src.is_multicast() {
            src.ip = loopback_ip(src.v6);
        }
        let lo = if dest.v6 { (41, 19) } else { (0, 34) };
        if net.socks[sock].opts.int_opt(lo).unwrap_or(1) != 0 {
            for tgt in net.multicast_receivers(sock, dest) {
                if let Kind::Dgram(td) = &mut net.socks[tgt].kind {
                    td.enqueue(src, data.to_vec());
                }
            }
        }
        data.len() as i64
    }

    /// `IP_ADD_MEMBERSHIP`/`IP_DROP_MEMBERSHIP`, `IPV6_JOIN_GROUP`/
    /// `IPV6_LEAVE_GROUP` and `MCAST_JOIN_GROUP`/`MCAST_LEAVE_GROUP`: join or
    /// leave a multicast group, or `None` when `(level, optname)` isn't one.
    /// The interface named is ignored: the VM's sockets share one segment.
    fn set_membership(
        net: &mut Net,
        sock: usize,
        level: u64,
        optname: u64,
        optval: u64,
        optlen: u64,
        mem: &GuestMemory,
    ) -> Option<i64> {
        /// `IP_MAX_MEMBERSHIPS`, and the same for IPv6.
        const MAX_GROUPS: usize = 20;
        // (join, address family, offset of the group address, minimum optlen)
        let (join, v6, at, min) = match (level, optname) {
            (IPPROTO_IP, 35 | 36) => (optname == 35, false, 0, 8),
            (IPPROTO_IPV6, 20 | 21) => (optname == 20, true, 0, 20),
            // struct group_req: an interface index, then a sockaddr_storage
            // at offset 8, its address after the family and port.
            (IPPROTO_IP | IPPROTO_IPV6, 42 | 45) => {
                let v6 = mem.read_u32(optval + 8).ok()? as u16 == AF_INET6;
                (optname == 42, v6, if v6 { 16 } else { 12 }, 136)
            }
            _ => return None,
        };
        let domain = net.socks[sock].domain;
        if domain != AF_INET && domain != AF_INET6 {
            return Some(err(Errno::ENOPROTOOPT));
        }
        if !matches!(net.socks[sock].kind, Kind::Dgram(_)) {
            return Some(err(Errno::EPROTO));
        }
        if optlen < min {
            return Some(err(Errno::EINVAL));
        }
        let len = if v6 { 16 } else { 4 };
        let Ok(raw) = mem.read_vec(optval + at, len) else {
            return Some(err(Errno::EFAULT));
        };
        let mut group = InetAddr {
            v6,
            port: 0,
            ip: [0; 16],
        };
        group.ip[..len].copy_from_slice(&raw);
        if !group.is_multicast() {
            return Some(err(Errno::EINVAL));
        }
        let Kind::Dgram(d) = &mut net.socks[sock].kind else {
            unreachable!("checked above")
        };
        let had = d.groups.iter().position(|g| *g == group);
        Some(match (join, had) {
            (true, Some(_)) => err(Errno::EADDRINUSE),
            (true, None) if d.groups.len() >= MAX_GROUPS => err(Errno::ENOBUFS),
            (true, None) => {
                d.groups.push(group);
                0
            }
            (false, Some(i)) => {
                d.groups.remove(i);
                0
            }
            (false, None) => err(Errno::EADDRNOTAVAIL),
        })
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
        let port = net.ephemeral_port(net.socks[sock].netns, "udp", v6);
        let local = InetAddr {
            v6,
            port,
            ip: loopback_ip(v6),
        };
        if let Kind::Dgram(d) = &mut net.socks[sock].kind {
            d.local = Some(local);
        }
        net.dgram_ports
            .entry(route_key(net.socks[sock].netns, "udp", local))
            .or_default()
            .push(sock);
        local
    }

    /// `socket(domain, type, protocol)` — an unbound, unconnected endpoint.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_socket(
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
        domain: u64,
        sotype: u64,
        protocol: u64,
    ) -> i64 {
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
                netns: cx.cur.ns.net,
                unix_bound_key: None,
                domain,
                kind,
                nonblock,
                opts: SockOpts::default(),
                refs: 1,
                edge_seq: 0,
            });
            let fd = cx.cur.fds.alloc(Fd::Socket { sock: idx, end: 0 });
            cx.cur.fds.set_cloexec(fd, sotype & SOCK_CLOEXEC != 0);
            return i64::from(fd);
        }
        if domain != AF_UNIX && domain != AF_INET && domain != AF_INET6 {
            return err(Errno::EAFNOSUPPORT);
        }
        let base_type = sotype & 0xf;
        let nonblock = sotype & SOCK_NONBLOCK != 0;
        // ICMP: a raw socket (`ping` as root), or a `SOCK_DGRAM` ping socket
        // (unprivileged ping). The only raw protocol modeled is ICMP.
        let icmp_proto = if domain == AF_INET6 {
            IPPROTO_ICMPV6
        } else {
            IPPROTO_ICMP
        };
        if domain != AF_UNIX
            && (base_type == SOCK_RAW || (base_type == SOCK_DGRAM && protocol == icmp_proto))
        {
            if protocol != icmp_proto {
                return err(Errno::EPROTONOSUPPORT);
            }
            net.ping_ident = net.ping_ident.wrapping_add(1);
            let icmp = Icmp {
                raw: base_type == SOCK_RAW,
                ident: net.ping_ident,
            };
            // Open the backend endpoint now, so a raw socket sees replies (and
            // other ICMP) whether or not it has sent; without one, only
            // loopback echoes are answered.
            // Only the VM's first network namespace is connected to anything.
            let host = net
                .egress
                .as_ref()
                .filter(|_| cx.cur.ns.net == 0)
                .and_then(|e| e.open_icmp(domain == AF_INET6).ok());
            let idx = net.socks.len();
            net.socks.push(Sock {
                netns: cx.cur.ns.net,
                unix_bound_key: None,
                domain,
                kind: Kind::Dgram(Dgram {
                    icmp: Some(icmp),
                    host,
                    ..Dgram::default()
                }),
                nonblock,
                opts: SockOpts::default(),
                refs: 1,
                edge_seq: 0,
            });
            let fd = cx.cur.fds.alloc(Fd::Socket { sock: idx, end: 0 });
            cx.cur.fds.set_cloexec(fd, sotype & SOCK_CLOEXEC != 0);
            return i64::from(fd);
        }
        // Stream and datagram are supported for every domain (including AF_UNIX
        // datagram, which syslog's /dev/log uses); other types aren't.
        if base_type != SOCK_STREAM && base_type != SOCK_DGRAM {
            return err(Errno::EOPNOTSUPP);
        }
        let kind = if base_type == SOCK_DGRAM {
            Kind::Dgram(Dgram::default())
        } else {
            Kind::Idle { bound: None }
        };
        let idx = net.socks.len();
        net.socks.push(Sock {
            netns: cx.cur.ns.net,
            unix_bound_key: None,
            domain,
            kind,
            nonblock,
            opts: SockOpts::default(),
            refs: 1,
            edge_seq: 0,
        });
        let fd = cx.cur.fds.alloc(Fd::Socket { sock: idx, end: 0 });
        cx.cur.fds.set_cloexec(fd, sotype & SOCK_CLOEXEC != 0);
        i64::from(fd)
    }

    /// `socketpair(domain, type, protocol, sv)` — a connected AF_UNIX pair whose
    /// two fds are written to `sv[0]`/`sv[1]`.
    #[allow(clippy::too_many_arguments, clippy::unused_self)]
    pub(super) fn sys_socketpair(
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
        domain: u64,
        sotype: u64,
        _protocol: u64,
        sv: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        if domain as u16 != AF_UNIX {
            return err(Errno::EAFNOSUPPORT);
        }
        let base_type = sotype & 0xf;
        if ![SOCK_STREAM, SOCK_DGRAM, SOCK_SEQPACKET].contains(&base_type) {
            return err(Errno::EOPNOTSUPP);
        }
        let nonblock = sotype & SOCK_NONBLOCK != 0;
        // A datagram socketpair is two connected `Dgram` slots that deliver
        // straight into each other's queue (no address namespace); a stream
        // socketpair is a single two-ended `Pair`. The two fds returned differ
        // accordingly: (slotA end0, slotB end0) for dgram, (slot end0, slot
        // end1) for stream.
        let (idx0, idx1, end0, end1) = if base_type == SOCK_DGRAM {
            let a = net.socks.len();
            let b = a + 1;
            for peer in [b, a] {
                net.socks.push(Sock {
                    netns: cx.cur.ns.net,
                    unix_bound_key: None,
                    domain: AF_UNIX,
                    kind: Kind::Dgram(Dgram {
                        pair_peer: Some(peer),
                        ..Dgram::default()
                    }),
                    nonblock,
                    opts: SockOpts::default(),
                    refs: 1,
                    edge_seq: 0,
                });
            }
            (a, b, 0, 0)
        } else {
            let mut pair = Pair::new();
            pair.nonblock = [nonblock, nonblock];
            pair.cred = [Ucred::of(cx); 2];
            if base_type == SOCK_SEQPACKET {
                pair.msgs = Some([VecDeque::new(), VecDeque::new()]);
            }
            let idx = net.socks.len();
            net.socks.push(Sock {
                netns: cx.cur.ns.net,
                unix_bound_key: None,
                domain: AF_UNIX,
                kind: Kind::Pair(pair),
                nonblock,
                opts: SockOpts::default(),
                refs: 1,
                edge_seq: 0,
            });
            (idx, idx, 0, 1)
        };
        let fd0 = cx.cur.fds.alloc(Fd::Socket {
            sock: idx0,
            end: end0,
        });
        let fd1 = cx.cur.fds.alloc(Fd::Socket {
            sock: idx1,
            end: end1,
        });
        let cloexec = sotype & SOCK_CLOEXEC != 0;
        cx.cur.fds.set_cloexec(fd0, cloexec);
        cx.cur.fds.set_cloexec(fd1, cloexec);
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
    pub(super) fn sys_bind(
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
        fd: u64,
        addr: u64,
        addrlen: u64,
        mem: &GuestMemory,
    ) -> i64 {
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
            Addr::Unix(ref path) => {
                if net.socks[sock].domain != AF_UNIX {
                    return err(Errno::EINVAL);
                }
                let path = path.clone();
                net.socks[sock].unix_bound_key = Some(unix_key(
                    net.socks[sock].netns,
                    &path,
                    cx.unix_identity.as_ref(),
                ));
                match &mut net.socks[sock].kind {
                    Kind::Idle { bound } => {
                        *bound = Some(parsed);
                        0
                    }
                    // A datagram socket bound to a path receives datagrams sent to
                    // it (syslog's /dev/log, and unix-dgram IPC generally).
                    Kind::Dgram(_) => {
                        net.dgram_paths.insert(
                            unix_key(net.socks[sock].netns, &path, cx.unix_identity.as_ref()),
                            sock,
                        );
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
                let proto = match &net.socks[sock].kind {
                    Kind::Idle { .. } => "tcp",
                    Kind::Dgram(_) => "udp",
                    _ => return err(Errno::EINVAL),
                };
                // A UDP socket may bind a multicast group's address, to receive
                // only that group's datagrams on the port.
                if !a.valid_bind() && (proto != "udp" || !a.is_multicast()) {
                    return err(Errno::EADDRNOTAVAIL);
                }
                if a.port == 0 {
                    a.port = net.ephemeral_port(net.socks[sock].netns, proto, a.v6);
                } else if net.addr_in_use(net.socks[sock].netns, proto, a, sock) {
                    return err(Errno::EADDRINUSE);
                }
                match &mut net.socks[sock].kind {
                    Kind::Idle { bound } => *bound = Some(Addr::Inet(a)),
                    Kind::Dgram(d) => d.local = Some(a),
                    _ => return err(Errno::EINVAL),
                }
                if proto == "udp" {
                    let netns = net.socks[sock].netns;
                    let group = net
                        .dgram_ports
                        .entry(route_key(netns, "udp", a))
                        .or_default();
                    group.retain(|&i| i != sock);
                    group.push(sock);
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
            let port = net.ephemeral_port(net.socks[sock].netns, "tcp", v6);
            bound = Some(Addr::Inet(InetAddr {
                v6,
                port,
                ip: [0; 16],
            }));
        }
        let key = match &bound {
            Some(Addr::Unix(p)) => Some(
                net.socks[sock]
                    .unix_bound_key
                    .clone()
                    .unwrap_or_else(|| unix_key(net.socks[sock].netns, p, None)),
            ),
            Some(Addr::Inet(a)) => Some(route_key(net.socks[sock].netns, "tcp", *a)),
            None => None,
        };
        // Two sockets may share a port with SO_REUSEADDR while they are only
        // bound, but listening on it is exclusive unless they use SO_REUSEPORT.
        if let Some(Addr::Inet(a)) = &bound
            && net.listen_conflict(*a, sock)
        {
            return err(Errno::EADDRINUSE);
        }
        net.socks[sock].kind = Kind::Listener {
            addr: bound,
            backlog: VecDeque::new(),
            cred: Ucred::of(cx),
        };
        if let Some(k) = key {
            net.listeners.entry(k).or_default().push(sock);
        }
        0
    }

    /// `connect(fd, addr, addrlen)` — for a stream socket, connect an idle
    /// socket to a listener, turning this slot into the client end of a fresh
    /// pair and queuing it on the listener for `accept4`; for a datagram
    /// socket, just record the peer (no handshake, per UDP semantics).
    pub(super) fn sys_connect(
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
        fd: u64,
        addr: u64,
        addrlen: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let Some((sock, end)) = self.sock_of(cx, fd) else {
            return err(Errno::ENOTSOCK);
        };
        // Already bridged to a host connection: this is the re-trap of a
        // blocking connect waiting out its handshake, or the guest asking
        // again after a nonblocking one — either way report its progress.
        if let Kind::Host(h) = &mut net.socks[sock].kind {
            if !h.connecting {
                return err(Errno::EISCONN);
            }
            return match h.conn.poll_connect() {
                Ok(false) if h.nonblock => err(Errno::EALREADY),
                Ok(false) => {
                    cx.block = true;
                    0
                }
                Ok(true) => {
                    h.connecting = false;
                    0
                }
                Err(e) => {
                    h.connecting = false;
                    connect_errno(&e)
                }
            };
        }
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
        // A stream `AF_INET6` socket connecting to an IPv4-mapped address
        // (`::ffff:127.0.0.1`) makes an IPv4 connection, as on Linux.
        let target = match target {
            Addr::Inet(a) if !matches!(net.socks[sock].kind, Kind::Dgram(_)) => {
                Addr::Inet(a.unmapped().unwrap_or(a))
            }
            other => other,
        };
        if let Addr::Inet(a) = &target
            && let Some(e) = net.loopback_unreachable(sock, a)
        {
            return e;
        }
        // A routable (non-loopback) destination: with a host-egress backend
        // installed, bridge it to the real internet; without one, it's
        // unreachable (loopback-only VM).
        // A multicast group is a peer a datagram socket can have without
        // egress: its datagrams go to the group's members in this VM.
        if let Addr::Inet(dest) = &target
            && dest.is_multicast()
            && matches!(net.socks[sock].kind, Kind::Dgram(Dgram { icmp: None, .. }))
        {
            self.ensure_dgram_bound(net, sock);
            if let Kind::Dgram(d) = &mut net.socks[sock].kind {
                d.peer = Some(*dest);
            }
            return 0;
        }
        let routable = matches!(&target, Addr::Inet(a) if !a.valid_bind());
        if routable {
            let Addr::Inet(dest) = target else {
                unreachable!("routable implies Inet")
            };
            if matches!(net.socks[sock].kind, Kind::Dgram(_)) {
                // UDP: remember the routable peer; the host socket is opened
                // lazily on first send (see `send_bytes`/`write_socket`).
                if net.egress_for(sock).is_none() {
                    return err(Errno::ENETUNREACH);
                }
                self.ensure_dgram_bound(net, sock);
                if let Kind::Dgram(d) = &mut net.socks[sock].kind {
                    d.peer = Some(dest);
                }
                return 0;
            }
            return self.connect_host(net, cx, sock, end, dest);
        }

        if matches!(net.socks[sock].kind, Kind::Dgram(_)) {
            // An `AF_UNIX` datagram socket connecting to a path (syslog's
            // `/dev/log`) isn't modeled — `sendto` with the path is — so it's
            // what Linux says when no socket is there.
            let Addr::Inet(peer) = target else {
                return err(Errno::ENOENT);
            };
            self.ensure_dgram_bound(net, sock);
            if let Kind::Dgram(d) = &mut net.socks[sock].kind {
                d.peer = Some(peer);
            }
            return 0;
        }

        // An idle client end 0. A connected stream socket, or a listening one,
        // is EISCONN, as Linux says — what a nonblocking connect's caller
        // (hiredis, after EINPROGRESS) asks again to learn it has finished.
        match &mut net.socks[sock].kind {
            Kind::Idle { .. } if end == 0 => {}
            Kind::Pair(p) => {
                return if std::mem::take(&mut p.connect_pending[end]) {
                    0
                } else {
                    err(Errno::EISCONN)
                };
            }
            Kind::Listener { .. } => return err(Errno::EISCONN),
            _ => return err(Errno::EINVAL),
        }
        let key = match &target {
            Addr::Unix(p) => unix_key(net.socks[sock].netns, p, cx.unix_identity.as_ref()),
            Addr::Inet(a) => route_key(net.socks[sock].netns, "tcp", *a),
        };
        let netns = net.socks[sock].netns;
        let picked = net.pick_listener(&key).or_else(|| match &target {
            Addr::Inet(a) if !a.v6 => net.pick_dual_listener(netns, *a),
            _ => None,
        });
        let Some(lidx) = picked else {
            return err(Errno::ECONNREFUSED);
        };
        // A full accept queue: nobody is taking connections (Linux's SOMAXCONN).
        if let Kind::Listener { backlog, .. } = &net.socks[lidx].kind
            && backlog.len() >= 4096
        {
            return err(Errno::EAGAIN);
        }
        let (listener_addr, listener_cred) = match &net.socks[lidx].kind {
            Kind::Listener { addr, cred, .. } => (addr.clone(), *cred),
            _ => return err(Errno::ECONNREFUSED),
        };
        // Repurpose the client's idle slot as the connected pair, then queue its
        // index (the server-side end 1) on the listener's backlog.
        let mut pair = Pair::new();
        pair.cred = [Ucred::of(cx), listener_cred];
        pair.nonblock[0] = net.socks[sock].nonblock;
        if domain != AF_UNIX {
            // The connection's family is the target's (an IPv4-mapped one was
            // made plain IPv4 above); an `AF_INET6` end of an IPv4 connection
            // sees its addresses IPv4-mapped.
            let v6 = matches!(&target, Addr::Inet(a) if a.v6);
            let mut peer_addr = match listener_addr {
                Some(Addr::Inet(a)) if a.v6 == v6 => a,
                Some(Addr::Inet(a)) => InetAddr {
                    v6,
                    port: a.port,
                    ip: loopback_ip(v6),
                },
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
            pair.addrs[0] = Some(net.fresh_local(net.socks[sock].netns, v6));
            pair.addrs[1] = Some(peer_addr);
            pair.v6_view = [
                !v6 && domain == AF_INET6,
                !v6 && net.socks[lidx].domain == AF_INET6,
            ];
        }
        // The client end is held by every fd the idle socket had.
        pair.refs[0] = net.socks[sock].refs;
        let einprogress = domain != AF_UNIX && net.socks[sock].nonblock;
        pair.connect_pending[0] = einprogress;
        net.socks[sock].kind = Kind::Pair(pair);
        if let Kind::Listener { backlog, .. } = &mut net.socks[lidx].kind {
            backlog.push_back(sock);
        }
        // A nonblocking TCP connect can't report success synchronously: real
        // Linux returns EINPROGRESS and the socket becomes writable (POLLOUT)
        // with SO_ERROR==0 once the handshake finishes. This virtual pair is
        // already fully connected — hence immediately POLLOUT and SO_ERROR==0 —
        // so returning EINPROGRESS here alone reproduces the observable
        // sequence. AF_UNIX stream connects complete synchronously even when
        // nonblocking (Linux returns 0), so this only applies to AF_INET(6).
        if einprogress {
            return err(Errno::EINPROGRESS);
        }
        0
    }

    /// Bridge a stream socket onto a real host connection to routable `dest`
    /// (egress). The socket must be an idle client end. Returns `0` on a
    /// completed connection, else a negative errno. A transport that returns
    /// with the handshake still in flight leaves the socket `connecting`: a
    /// nonblocking socket gets `EINPROGRESS`, a blocking one re-traps (see the
    /// `Kind::Host` arm at the top of [`Self::sys_connect`]).
    #[allow(clippy::unused_self)]
    fn connect_host(
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
        sock: usize,
        end: usize,
        dest: InetAddr,
    ) -> i64 {
        if !matches!(&net.socks[sock].kind, Kind::Idle { .. } if end == 0) {
            return err(Errno::EINVAL);
        }
        let Some(egress) = net.egress_for(sock) else {
            return err(Errno::ENETUNREACH);
        };
        match egress.connect_tcp(dest.ip, dest.v6, dest.port) {
            Ok(mut conn) => {
                let nonblock = net.socks[sock].nonblock;
                let pending = match conn.poll_connect() {
                    Ok(up) => !up,
                    Err(e) => return connect_errno(&e),
                };
                // Only an explicit choice: a transport's own default stands
                // for a guest that never asked.
                let mut conn = conn;
                for &(level, optname) in CONN_OPTIONS {
                    if let Some(opt) = conn_option(&net.socks[sock].opts, level, optname) {
                        conn.set_option(opt);
                    }
                }
                net.socks[sock].kind = Kind::Host(HostSock {
                    conn,
                    peer: dest,
                    wr_shut: false,
                    nonblock,
                    connecting: pending,
                    linger_until: None,
                });
                match (pending, nonblock) {
                    (false, _) => 0,
                    (true, true) => err(Errno::EINPROGRESS),
                    (true, false) => {
                        cx.block = true; // re-trap until the handshake resolves
                        0
                    }
                }
            }
            Err(e) => connect_errno(&e),
        }
    }

    /// `accept4(fd, addr, addrlen, flags)` — hand back the server-side end of a
    /// pending connection (blocking, like pipe read, when none is queued and
    /// the listening socket is not `O_NONBLOCK`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_accept4(
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
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
            Kind::Pair(p) => p.addr_seen_by(0, 1).map(Addr::Inet),
            _ => None,
        };
        write_sockaddr(mem, addr, addrlen, domain, peer.as_ref());
        if let Kind::Pair(p) = &mut net.socks[pidx].kind {
            p.nonblock[1] = flags & SOCK_NONBLOCK != 0;
        }
        let fd = cx.cur.fds.alloc(Fd::Socket { sock: pidx, end: 1 });
        cx.cur.fds.set_cloexec(fd, flags & SOCK_CLOEXEC != 0);
        i64::from(fd)
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
    fn bind_netlink(
        &self,
        net: &mut Net,
        sock: usize,
        addr: u64,
        addrlen: u64,
        mem: &GuestMemory,
    ) -> i64 {
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
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
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
            Kind::Pair(p) => p.addr_seen_by(end, end).map(Addr::Inet),
            Kind::Dgram(d) => d.local.map(Addr::Inet),
            // A host-bridged socket has an ephemeral local address the guest
            // never bound and doesn't inspect; report the wildcard.
            Kind::Host(_) | Kind::Netlink(_) => None,
        };
        write_sockaddr(mem, addr, addrlen, domain, resolved.as_ref())
    }

    /// `getpeername(fd, addr, addrlen)` — the peer address (best-effort).
    pub(super) fn sys_getpeername(
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
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
                p.addr_seen_by(1 - end, end).map(Addr::Inet).as_ref(),
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
    pub(super) fn sys_shutdown(
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
        fd: u64,
        how: u64,
    ) -> i64 {
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

    /// Apply `SO_RCVTIMEO`/`SO_SNDTIMEO` to a socket call that just returned
    /// `ret`: if it wants to block on a socket with a timeout, arm the deadline
    /// on the first pass, and on the re-run after it passes fail with `EAGAIN`,
    /// as Linux does. Such a call isn't restarted after a signal (signal(7)).
    pub(super) fn apply_sock_timeout(
        &self,
        net: &Net,
        cx: &mut ServiceCtx,
        fd: u64,
        direction: SockDirection,
        ret: i64,
    ) -> i64 {
        let Some((sock, _)) = self.sock_of(cx, fd) else {
            return ret;
        };
        let opts = &net.socks[sock].opts;
        let raw = match direction {
            SockDirection::Receive => opts.rcvtimeo,
            SockDirection::Send => opts.sndtimeo,
        };
        let secs = i64::from_le_bytes(raw[0..8].try_into().unwrap());
        let usecs = i64::from_le_bytes(raw[8..16].try_into().unwrap());
        let timeout_ns = i128::from(secs) * 1_000_000_000 + i128::from(usecs) * 1_000;
        if timeout_ns <= 0 {
            return ret;
        }
        if !cx.block {
            cx.cur.wake_deadline = None;
            return ret;
        }
        cx.restartable = false;
        let now = super::poll::now_ns();
        match cx.cur.wake_deadline {
            Some(deadline) if now >= deadline => {
                cx.cur.wake_deadline = None;
                cx.block = false;
                err(Errno::EAGAIN)
            }
            Some(_) => ret,
            None => {
                cx.cur.wake_deadline = Some(now + timeout_ns as u128);
                ret
            }
        }
    }

    /// `setsockopt(fd, level, optname, optval, optlen)`.
    ///
    /// These take effect: `SO_REUSEADDR`/`SO_REUSEPORT` (`bind`, `listen` and
    /// connection routing), `SO_RCVTIMEO`/`SO_SNDTIMEO` (blocking socket calls),
    /// `TCP_NODELAY`, `SO_KEEPALIVE`, `TCP_KEEPIDLE`/`KEEPINTVL`/`KEEPCNT` and an
    /// abortive `SO_LINGER` (on the real connection, for a socket bridged to the
    /// host), `SO_SNDBUF`/`SO_RCVBUF` (clamped as Linux does, though the in-VM
    /// queues don't shrink to them).
    ///
    /// Other known options are stored and echoed back by `getsockopt` without
    /// changing how the socket behaves. [`super::sockopt::refused`] lists the
    /// ones that fail instead, and an option Linux doesn't know is `ENOPROTOOPT`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sys_setsockopt(
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
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
        // An option Linux doesn't know is ENOPROTOOPT; the read-only ones
        // (SO_TYPE, SO_ERROR, SO_PEERCRED, SO_ACCEPTCONN, SO_PROTOCOL, …) too.
        if !super::sockopt::known(level, optname)
            || (level == SOL_SOCKET
                && matches!(
                    optname,
                    3 | 4 | 17 | 28 | 30 | 31 | 38 | 39 | 57 | 59 | 71 | 77
                ))
        {
            if !super::sockopt::known(level, optname) {
                self.note_unsupported("setsockopt", (level << 32) | optname);
            }
            return err(Errno::ENOPROTOOPT);
        }
        if let Some(errno) = super::sockopt::refused(level, optname) {
            return err(errno);
        }
        // Linux takes a single byte for an IP-level integer option too.
        let int_len = if level == IPPROTO_IP { 1 } else { 4 };
        if super::sockopt::is_int(level, optname) && optlen < int_len {
            return err(Errno::EINVAL);
        }
        // SO_PRIORITY above 6 needs CAP_NET_ADMIN.
        if (level, optname) == (SOL_SOCKET, SO_PRIORITY)
            && mem
                .read_u32(optval)
                .is_ok_and(|v| v as i32 > 6 || (v as i32) < 0)
        {
            return err(Errno::EPERM);
        }
        // A timeout's microseconds must be below one second (EDOM on Linux).
        let is_timeo = level == SOL_SOCKET
            && matches!(
                optname,
                SO_RCVTIMEO | SO_SNDTIMEO | SO_RCVTIMEO_NEW | SO_SNDTIMEO_NEW
            );
        if is_timeo {
            if optlen < 16 {
                return err(Errno::EINVAL);
            }
            let Ok(usec) = mem.read_u64(optval + 8) else {
                return err(Errno::EFAULT);
            };
            if usec >= 1_000_000 {
                return err(Errno::EDOM);
            }
        }
        // Linux's ranges for the keepalive timers.
        if level == IPPROTO_TCP
            && let Some(max) = match optname {
                TCP_KEEPIDLE | TCP_KEEPINTVL => Some(32767),
                TCP_KEEPCNT => Some(127),
                _ => None,
            }
            && mem.read_u32(optval).is_ok_and(|v| v == 0 || v > max)
        {
            return err(Errno::EINVAL);
        }
        if let Some(ret) = Self::set_membership(net, sock, level, optname, optval, optlen, mem) {
            return ret;
        }
        let opts = &mut net.socks[sock].opts;
        if optlen <= 256
            && let Ok(mut raw) = mem.read_vec(optval, optlen as usize)
        {
            // Stored as the int getsockopt reads back.
            if raw.len() < 4 && super::sockopt::is_int(level, optname) {
                raw = i32::from(raw[0]).to_le_bytes().to_vec();
            }
            opts.extra.insert((level, optname), raw);
        }
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
                        opts.rcvbuf = super::sockopt::clamp_sock_buf(v);
                    }
                }
                SO_SNDBUF if optlen >= 4 => {
                    if let Ok(v) = mem.read_u32(optval) {
                        opts.sndbuf = super::sockopt::clamp_sock_buf(v);
                    }
                }
                SO_LINGER if optlen >= 8 => {
                    if let Ok(b) = mem.read_vec(optval, 8) {
                        opts.linger_on = i32::from_le_bytes([b[0], b[1], b[2], b[3]]) != 0;
                        opts.linger_secs = i32::from_le_bytes([b[4], b[5], b[6], b[7]]);
                    }
                }
                SO_RCVTIMEO | SO_RCVTIMEO_NEW if optlen >= 16 => {
                    if let Ok(b) = mem.read_vec(optval, 16) {
                        opts.rcvtimeo.copy_from_slice(&b);
                    }
                }
                SO_SNDTIMEO | SO_SNDTIMEO_NEW if optlen >= 16 => {
                    if let Ok(b) = mem.read_vec(optval, 16) {
                        opts.sndtimeo.copy_from_slice(&b);
                    }
                }
                // Every other known option: stored above, echoed by getsockopt.
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
            && let Some(v) = opts.int_opt((IPPROTO_IP, IP_TOS))
        {
            opts.tos = v as u32;
        }
        // Options with a meaning on the real connection go to it too.
        if let Some(opt) = conn_option(&net.socks[sock].opts, level, optname)
            && let Kind::Host(h) = &mut net.socks[sock].kind
        {
            h.conn.set_option(opt);
        }
        0
    }

    /// `getsockopt(fd, level, optname, optval, optlen)` — sane canned/stored
    /// values for the options `setsockopt` above understands, plus
    /// `SO_TYPE`/`SO_ERROR`/`SO_ACCEPTCONN`/`SO_DOMAIN`/`SO_PROTOCOL`; `0` for
    /// anything else.
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub(super) fn sys_getsockopt(
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
        fd: u64,
        level: u64,
        optname: u64,
        optval: u64,
        optlen_ptr: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some((sock, end)) = self.sock_of(cx, fd) else {
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
        if level == SOL_SOCKET
            && matches!(
                optname,
                SO_RCVTIMEO | SO_SNDTIMEO | SO_RCVTIMEO_NEW | SO_SNDTIMEO_NEW
            )
        {
            let opts = &net.socks[sock].opts;
            let b = if matches!(optname, SO_RCVTIMEO | SO_RCVTIMEO_NEW) {
                opts.rcvtimeo
            } else {
                opts.sndtimeo
            };
            return write_optval(mem, optval, optlen_ptr, &b);
        }
        let value: u32 = if level == SOL_SOCKET {
            match optname {
                SO_TYPE => match &net.socks[sock].kind {
                    Kind::Dgram(Dgram {
                        icmp: Some(Icmp { raw: true, .. }),
                        ..
                    }) => SOCK_RAW as u32,
                    Kind::Dgram(_) => SOCK_DGRAM as u32,
                    Kind::Netlink(nl) => nl.sotype as u32,
                    Kind::Pair(Pair { msgs: Some(_), .. }) => SOCK_SEQPACKET as u32,
                    _ => SOCK_STREAM as u32,
                },
                SO_ERROR => {
                    // A nonblocking host connect reports its outcome here once
                    // it resolves (the guest polled for POLLOUT, then asks).
                    if let Kind::Host(h) = &mut net.socks[sock].kind
                        && h.connecting
                    {
                        match h.conn.poll_connect() {
                            Ok(false) => {}
                            Ok(true) => h.connecting = false,
                            Err(e) => {
                                h.connecting = false;
                                net.socks[sock].opts.error = -connect_errno(&e) as i32;
                            }
                        }
                    }
                    let e = net.socks[sock].opts.error;
                    net.socks[sock].opts.error = 0; // read-and-cleared
                    e as u32
                }
                SO_REUSEADDR => u32::from(net.socks[sock].opts.reuseaddr),
                SO_REUSEPORT => u32::from(net.socks[sock].opts.reuseport),
                SO_KEEPALIVE => u32::from(net.socks[sock].opts.keepalive),
                SO_BROADCAST => u32::from(net.socks[sock].opts.broadcast),
                // Linux reports twice the internally stored buffer size (it
                // reserves the other half for bookkeeping/skb overhead), so a
                // read after `setsockopt` returns ~2× what was requested.
                SO_RCVBUF => net.socks[sock].opts.rcvbuf.saturating_mul(2),
                SO_SNDBUF => net.socks[sock].opts.sndbuf.saturating_mul(2),
                SO_ACCEPTCONN => u32::from(matches!(net.socks[sock].kind, Kind::Listener { .. })),
                SO_DOMAIN => u32::from(net.socks[sock].domain),
                SO_PROTOCOL => match (&net.socks[sock].kind, net.socks[sock].domain) {
                    (Kind::Dgram(_), _) => 17,                      // IPPROTO_UDP
                    (_, d) if d == AF_UNIX || d == AF_NETLINK => 0, // NETLINK_ROUTE == 0
                    _ => 6,                                         // IPPROTO_TCP
                },
                // SO_PEERCRED: `struct ucred { pid, uid, gid }` of the process
                // on the other end when the connection was made. A socket that
                // isn't connected has no peer: pid 0 and the overflow ids.
                17 => {
                    let peer = match &net.socks[sock].kind {
                        Kind::Pair(p) => p.cred[1 - end],
                        _ => Ucred {
                            pid: 0,
                            uid: 65534,
                            gid: 65534,
                        },
                    };
                    let mut b = [0u8; 12];
                    let pid = self.sender_pid_seen(cx, peer.pid);
                    b[0..4].copy_from_slice(&pid.to_le_bytes());
                    b[4..8].copy_from_slice(&peer.uid.to_le_bytes());
                    b[8..12].copy_from_slice(&peer.gid.to_le_bytes());
                    return write_optval(mem, optval, optlen_ptr, &b);
                }
                // SO_PEERSEC: no LSM labels.
                31 => return err(Errno::ENOPROTOOPT),
                // SO_COOKIE: a stable 64-bit id for the socket.
                57 => {
                    return write_optval(mem, optval, optlen_ptr, &(sock as u64 + 1).to_le_bytes());
                }
                _ => {
                    return self
                        .getsockopt_generic(net, sock, level, optname, optval, optlen_ptr, mem);
                }
            }
        } else if level == IPPROTO_TCP && optname == TCP_NODELAY {
            u32::from(net.socks[sock].opts.nodelay)
        } else if level == IPPROTO_IPV6 && optname == IPV6_V6ONLY {
            u32::from(net.socks[sock].opts.v6only)
        } else if level == IPPROTO_IP && optname == IP_TOS {
            net.socks[sock].opts.tos
        } else {
            return self.getsockopt_generic(net, sock, level, optname, optval, optlen_ptr, mem);
        };
        write_optval(mem, optval, optlen_ptr, &value.to_le_bytes())
    }

    /// `getsockopt` of an option without dedicated handling: the value the
    /// guest set, else Linux's default for a fresh socket, else zero; an
    /// option Linux doesn't know is `ENOPROTOOPT` (and recorded).
    #[allow(clippy::too_many_arguments)]
    fn getsockopt_generic(
        &self,
        net: &Net,
        sock: usize,
        level: u64,
        optname: u64,
        optval: u64,
        optlen_ptr: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        if !super::sockopt::known(level, optname) {
            self.note_unsupported("getsockopt", (level << 32) | optname);
            return err(Errno::ENOPROTOOPT);
        }
        let v = net.socks[sock]
            .opts
            .extra
            .get(&(level, optname))
            .cloned()
            .or_else(|| super::sockopt::default_value(level, optname))
            .unwrap_or_else(|| vec![0; 4]);
        write_optval(mem, optval, optlen_ptr, &v)
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
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
        fd: u64,
        buf: u64,
        len: u64,
        flags: u64,
        dest_addr: u64,
        dest_addrlen: u64,
        mem: &GuestMemory,
    ) -> i64 {
        const MSG_NOSIGNAL: u64 = 0x4000;
        let len = len.min(super::MAX_RW_COUNT);
        let Ok(data) = mem.read_vec(buf, len as usize) else {
            return err(Errno::EFAULT);
        };
        cx.dontwait = flags & MSG_DONTWAIT != 0;
        self.send_bytes(
            net,
            cx,
            fd,
            &data,
            dest_addr,
            dest_addrlen,
            flags & MSG_NOSIGNAL != 0,
            mem,
        )
    }

    /// The shared core of `sendto`/`sendmsg`: send an already-gathered `data`
    /// buffer on socket `fd`, optionally to `dest_addr` (a guest `sockaddr` of
    /// `dest_addrlen` bytes; `0` = no address, e.g. a connected socket).
    #[allow(clippy::too_many_arguments)]
    fn send_bytes(
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
        fd: u64,
        data: &[u8],
        dest_addr: u64,
        dest_addrlen: u64,
        nosignal: bool,
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
            return self.write_socket(net, cx, sock, end, data, nosignal);
        }
        let dest = match read_sockaddr(mem, dest_addr, dest_addrlen) {
            Some(Addr::Inet(d)) => d,
            // A unix datagram to a bound path: deliver to that socket's queue.
            // (An unbound/nonexistent path is silently dropped, as Linux does for
            // a datagram to a since-removed socket — no deadlock on the sender.)
            Some(Addr::Unix(path)) => {
                if !matches!(net.socks[sock].kind, Kind::Dgram(_)) {
                    return err(Errno::EINVAL);
                }
                let key = unix_key(net.socks[sock].netns, &path, cx.unix_identity.as_ref());
                if let Some(&tgt) = net.dgram_paths.get(&key)
                    && let Kind::Dgram(td) = &mut net.socks[tgt].kind
                {
                    // Unix datagrams carry no INET src; a zeroed placeholder (the
                    // recvfrom src is a unix path, usually ignored by the caller).
                    let src = InetAddr {
                        v6: false,
                        port: 0,
                        ip: [0; 16],
                    };
                    td.enqueue(src, data.to_vec());
                }
                return data.len() as i64;
            }
            _ => return err(Errno::EINVAL),
        };
        if !matches!(net.socks[sock].kind, Kind::Dgram(_)) {
            return err(Errno::EINVAL); // real errno: EOPNOTSUPP/EISCONN
        }
        if let Some(e) = net.loopback_unreachable(sock, &dest) {
            return e;
        }
        if matches!(
            net.socks[sock].kind,
            Kind::Dgram(Dgram { icmp: Some(_), .. })
        ) {
            return self.icmp_send(net, sock, data, dest);
        }
        if dest.is_multicast() {
            return self.multicast_send(net, sock, data, dest);
        }
        // A routable destination (DNS, chiefly) goes out through a real host
        // UDP socket when egress is enabled; without egress it's unreachable.
        if !dest.valid_bind() {
            if net.egress_for(sock).is_none() {
                return err(Errno::ENETUNREACH);
            }
            return self.host_udp_send(net, sock, data, dest);
        }
        let src = self.ensure_dgram_bound(net, sock);
        let key = route_key(net.socks[sock].netns, "udp", dest);
        if let Some(tgt) = net.pick_dgram_target(&key, src.port)
            && let Kind::Dgram(td) = &mut net.socks[tgt].kind
        {
            td.enqueue(src, data.to_vec());
        }
        data.len() as i64
    }

    /// Readiness (`POLLIN`/`POLLOUT`) for a host-bridged socket slot, or `None`
    /// for a non-host socket (which uses the generic best-effort answer). Only
    /// host sockets get a precise readable answer (via a peek), so a guest that
    /// trusts `poll` (apk's http client) doesn't spin on spurious readiness.
    /// Bytes available to read on socket `sock` (end `end`) right now — for
    /// `ioctl(FIONREAD/SIOCINQ)`. A bridged host stream peeks the real count; an
    /// in-VM pair reports its inbound queue; a datagram socket reports the next
    /// datagram's length (after draining the host UDP socket).
    pub(super) fn socket_readable_bytes(&self, net: &mut Net, sock: usize, end: usize) -> u64 {
        if matches!(net.socks.get(sock).map(|s| &s.kind), Some(Kind::Dgram(_))) {
            self.host_udp_drain(net, sock);
        }
        match net.socks.get_mut(sock).map(|s| &mut s.kind) {
            Some(Kind::Host(h)) => h.conn.readable_len() as u64,
            Some(Kind::Pair(p)) => p.to[end.min(1)].len() as u64,
            Some(Kind::Dgram(d)) => d
                .queue
                .front()
                .map_or(0, |(_, payload)| payload.len() as u64),
            _ => 0,
        }
    }

    /// Poll/epoll readiness for a socket fd's end `e`, covering host-bridged
    /// sockets, in-VM stream pairs, listeners, and datagram sockets. Crucially,
    /// an idle connected pair with **no buffered data is not readable** — the old
    /// blanket `POLLIN | POLLOUT` for in-VM sockets made level-triggered event
    /// loops (libuv/Node/Bun/tokio) busy-spin at 100% CPU: poll said "readable",
    /// the follow-up `recv` returned `EAGAIN`, and the loop repeated forever.
    pub(super) fn sock_readiness(&self, net: &mut Net, sock: usize, end: usize) -> u32 {
        const POLLIN: u32 = 0x1;
        const POLLOUT: u32 = 0x4;
        const POLLHUP: u32 = 0x10;
        // Host-bridged sockets (egress TCP, host-backed UDP) get a precise answer.
        if let Some(m) = self.host_socket_readiness(net, sock) {
            return m;
        }
        match net.socks.get(sock).map(|s| &s.kind) {
            Some(Kind::Pair(p)) => {
                let e = end.min(1);
                let peer_closed = p.refs[1 - e] == 0;
                // EOF (a read returns 0 without blocking): our inbound queue is
                // drained and the peer is gone / shut its write / we shut our read.
                let eof = p.to[e].is_empty() && (peer_closed || p.shut_wr[1 - e] || p.shut_rd[e]);
                let mut m = 0;
                if !p.to[e].is_empty() || eof {
                    m |= POLLIN;
                }
                let capacity = net.socks[sock].opts.sndbuf.saturating_mul(2) as usize;
                if !p.shut_wr[e] && p.to[1 - e].len() < capacity {
                    m |= POLLOUT;
                }
                if peer_closed {
                    m |= POLLHUP;
                }
                m
            }
            // A listener is readable exactly when a connection is pending to accept.
            Some(Kind::Listener { backlog, .. }) => {
                if backlog.is_empty() {
                    0
                } else {
                    POLLIN
                }
            }
            // A datagram socket is readable when a datagram is queued.
            Some(Kind::Dgram(d)) => POLLOUT | if d.queue.is_empty() { 0 } else { POLLIN },
            // Netlink replies are synchronous; an unconnected/idle socket is just
            // writable. Neither should report a spurious POLLIN.
            Some(Kind::Netlink(_)) => POLLIN | POLLOUT,
            _ => POLLOUT,
        }
    }

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
            let readable = matches!(&net.socks[sock].kind, Kind::Dgram(d) if !d.queue.is_empty());
            return Some(POLLOUT | if readable { POLLIN } else { 0 });
        }
        let Kind::Host(h) = &mut net.socks[sock].kind else {
            return None;
        };
        if h.connecting {
            // Mid-handshake: nothing to report yet; once it resolves the socket
            // is writable (and, on failure, in error — SO_ERROR says which).
            const POLLERR: u32 = 0x8;
            match h.conn.poll_connect() {
                Ok(false) => return Some(0),
                Ok(true) => {}
                Err(_) => return Some(POLLOUT | POLLERR),
            }
        }
        let mut mask = if h.wr_shut { 0 } else { POLLOUT };
        if h.conn.poll_readable() {
            mask |= POLLIN;
        }
        Some(mask)
    }

    /// Send ICMP message `data` from ICMP socket `sock` to `dest`. A ping socket
    /// only sends echo requests and stamps them with its identifier (and, for
    /// IPv4, the checksum). An echo to a local address is answered here, as
    /// the loopback interface would; anything else goes to the egress
    /// backend's ICMP endpoint.
    #[allow(clippy::unused_self)]
    fn icmp_send(&self, net: &mut Net, sock: usize, data: &[u8], dest: InetAddr) -> i64 {
        let Kind::Dgram(d) = &mut net.socks[sock].kind else {
            return err(Errno::EINVAL);
        };
        let Some(icmp) = d.icmp else {
            return err(Errno::EINVAL);
        };
        let (echo_req, echo_reply) = if dest.v6 { (128u8, 129u8) } else { (8u8, 0u8) };
        if data.len() < 8 {
            return err(Errno::EINVAL);
        }
        let mut msg = data.to_vec();
        if !icmp.raw {
            if msg[0] != echo_req {
                return err(Errno::EINVAL);
            }
            msg[4..6].copy_from_slice(&icmp.ident.to_be_bytes());
            if !dest.v6 {
                set_icmp4_checksum(&mut msg);
            }
        }
        if dest.valid_bind() {
            // Loopback: answer an echo request ourselves (and swallow the rest).
            if msg[0] == echo_req {
                let mut reply = msg;
                reply[0] = echo_reply;
                if !dest.v6 {
                    set_icmp4_checksum(&mut reply);
                }
                let pkt = loopback_ip_packet(dest, &reply);
                if let Some(p) = shape_icmp(icmp, pkt) {
                    let src = InetAddr {
                        v6: dest.v6,
                        port: 0,
                        ip: if dest.is_any() {
                            loopback_ip(dest.v6)
                        } else {
                            dest.ip
                        },
                    };
                    d.enqueue(src, p);
                }
            }
            return data.len() as i64;
        }
        let Some(host) = d.host.as_mut() else {
            return err(Errno::ENETUNREACH);
        };
        match host.send_to(&msg, dest.ip, dest.v6, 0) {
            Ok(_) => data.len() as i64,
            Err(e) if e.kind() == std::io::ErrorKind::InvalidInput => err(Errno::EMSGSIZE),
            Err(_) => err(Errno::ENETUNREACH),
        }
    }

    /// Send a datagram out through this socket's host UDP socket (opening it
    /// lazily on first use), for egress to a routable address.
    #[allow(clippy::unused_self)]
    fn host_udp_send(&self, net: &mut Net, sock: usize, data: &[u8], dest: InetAddr) -> i64 {
        if let Kind::Dgram(d) = &net.socks[sock].kind
            && d.host.is_none()
        {
            match net.egress_for(sock).map(Egress::open_udp) {
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
        let icmp = d.icmp;
        let mut arrived = Vec::new();
        for _ in 0..64 {
            match host.recv_from() {
                Ok(Some((ip, v6, port, payload))) => {
                    // An ICMP endpoint yields whole IP packets: shape each for
                    // this socket (and drop what a ping socket doesn't want).
                    let payload = match icmp {
                        Some(i) => match shape_icmp(i, payload) {
                            Some(p) => p,
                            None => continue,
                        },
                        None => payload,
                    };
                    arrived.push((InetAddr { v6, port, ip }, payload));
                }
                _ => break,
            }
        }
        if let Kind::Dgram(d) = &mut net.socks[sock].kind {
            for (src, payload) in arrived {
                d.enqueue(src, payload);
            }
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
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
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
            let Some(packet) = self.drain_netlink(net, sock, len as usize, flags) else {
                if nonblock {
                    return err(Errno::EAGAIN);
                }
                cx.block = true;
                return 0;
            };
            // Every reply's source is the kernel, whose `nl_pid` is always `0`.
            write_netlink_sockaddr(mem, src_addr, src_addrlen, 0);
            let n = packet.len().min(len as usize);
            if mem.write(buf, &packet[..n]).is_err() {
                return err(Errno::EFAULT);
            }
            return if flags & MSG_TRUNC != 0 {
                packet.len() as i64
            } else {
                n as i64
            };
        }
        if !matches!(net.socks[sock].kind, Kind::Dgram(_)) {
            let n = self.recv_stream(net, cx, sock, end, buf, len, mem, flags);
            if n > 0 {
                let domain = net.socks[sock].domain;
                let peer = match &net.socks[sock].kind {
                    Kind::Pair(p) => p.addr_seen_by(1 - end, end).map(Addr::Inet),
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
    pub(super) fn sys_sendmsg(
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
        fd: u64,
        msg: u64,
        flags: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let Some(hdr) = MsgHdr::read(mem, msg) else {
            return err(Errno::EFAULT);
        };
        if hdr.iovlen > UIO_MAXIOV {
            return err(Errno::EMSGSIZE);
        }
        let mut data = Vec::new();
        let mut remaining = super::MAX_RW_COUNT;
        for i in 0..hdr.iovlen {
            let ent = hdr.iov + i * 16;
            let (Ok(base), Ok(len)) = (mem.read_u64(ent), mem.read_u64(ent + 8)) else {
                return err(Errno::EFAULT);
            };
            // Entries may repeat one buffer, so bound the total, not each one.
            let len = len.min(remaining);
            remaining -= len;
            match mem.read_vec(base, len as usize) {
                Ok(mut chunk) => data.append(&mut chunk),
                Err(_) => return err(Errno::EFAULT),
            }
        }
        // Parse any SCM_RIGHTS ancillary data and translate each passed guest fd
        // to a duplicated kernel-side `Fd` (validated up front: an unknown fd
        // fails the whole sendmsg with EBADF, before anything is queued).
        let Some(guest_fds) = parse_scm_rights(mem, hdr.control, hdr.controllen) else {
            return err(Errno::EFAULT);
        };
        let mut passed: Vec<Fd> = Vec::with_capacity(guest_fds.len());
        for g in &guest_fds {
            match cx.cur.fds.get(*g) {
                Some(f) => passed.push(f.clone()),
                None => return err(Errno::EBADF),
            }
        }
        const MSG_NOSIGNAL: u64 = 0x4000;
        cx.dontwait = flags & MSG_DONTWAIT != 0;
        let r = self.send_bytes(
            net,
            cx,
            fd,
            &data,
            hdr.name,
            u64::from(hdr.namelen),
            flags & MSG_NOSIGNAL != 0,
            mem,
        );
        // On a successful send over a connected AF_UNIX stream pair, queue the
        // passed fds as ancillary data riding the just-written bytes: bump each
        // backing ref (the in-flight duplicate) and record the stream offset the
        // fds attach to, so the matching `recvmsg` materializes them.
        // A send still parked waiting for room queues its descriptors on the
        // re-run that completes it.
        if r >= 0
            && !cx.block
            && !passed.is_empty()
            && let Some((sock, end)) = self.sock_of(cx, fd)
            && matches!(net.socks[sock].kind, Kind::Pair(_))
        {
            for f in &passed {
                self.scm_ref(net, f, true);
            }
            if let Kind::Pair(p) = &mut net.socks[sock].kind {
                let reader = 1 - end;
                // Bytes already queued ahead of this message (this write appended
                // `r` bytes to the tail).
                let pos = p.to[reader].len().saturating_sub(r as usize);
                p.scm[reader].push_back((pos, passed));
            }
        }
        r
    }

    /// `sendmmsg(fd, msgvec, vlen, flags)` — send an array of `struct mmsghdr`
    /// `{ msghdr msg_hdr; u32 msg_len }` (64 bytes each), writing each sent
    /// byte count back into `msg_len`. Returns the number of messages sent.
    pub(super) fn sys_sendmmsg(
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
        fd: u64,
        msgvec: u64,
        vlen: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
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
    pub(super) fn sys_recvmmsg(
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
        fd: u64,
        msgvec: u64,
        vlen: u64,
        flags: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let mut got = 0i64;
        for i in 0..vlen {
            let ent = msgvec + i * 64;
            let r = self.sys_recvmsg(net, cx, fd, ent, flags, mem);
            if r < 0 {
                // EAGAIN after at least one message is a normal stop, not an error.
                if got > 0 {
                    return got;
                }
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
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
        fd: u64,
        msg: u64,
        flags: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some(hdr) = MsgHdr::read(mem, msg) else {
            return err(Errno::EFAULT);
        };
        // Total scatter capacity, and the (base, len) list to scatter into.
        if hdr.iovlen > UIO_MAXIOV {
            return err(Errno::EMSGSIZE);
        }
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
        let (src, mut got, mut msg_flags, scm_fds) =
            match self.recv_message(net, cx, fd, total, flags) {
                Ok(v) => v,
                Err(e) => return e,
            };
        // Nothing yet: the call waits (re-traps) with the caller's `msghdr`
        // untouched — writing back `msg_controllen = 0` now would leave the
        // re-run no room for the descriptors it is waiting for.
        if cx.block {
            return 0;
        }
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
        // A datagram or netlink message longer than the buffer: the rest is
        // lost, and with `MSG_TRUNC` the call says how long it was.
        let whole = got.len();
        got.truncate(off);
        // Fill msg_name (source address); `msg_namelen` (socklen_t at msg+8)
        // doubles as write_sockaddr's in/out length word — it carries the
        // caller's capacity in and the written length out, exactly as recvmsg
        // wants.
        if hdr.name != 0 && hdr.namelen > 0 {
            let domain = self
                .sock_of(cx, fd)
                .map_or(AF_INET, |(s, _)| net.socks[s].domain);
            write_sockaddr(mem, hdr.name, msg + 8, domain, src.as_ref());
        }
        // Emit any SCM_RIGHTS ancillary data into msg_control (offset 40 =
        // msg_controllen), setting msg_flags (offset 48).
        // With SO_PASSCRED, a message on a connected AF_UNIX socket says who sent it.
        let creds = self
            .sock_of(cx, fd)
            .and_then(|(s, end)| match &net.socks[s].kind {
                Kind::Pair(p)
                    if net.socks[s]
                        .opts
                        .int_opt((SOL_SOCKET, SO_PASSCRED))
                        .unwrap_or(0)
                        != 0
                        && (whole > 0 || !scm_fds.is_empty()) =>
                {
                    Some(p.sender[end])
                }
                _ => None,
            });
        let creds = creds.map(|c| {
            let pid = if c.pid == 0 {
                0
            } else {
                self.visible_pid(cx, c.pid)
            };
            (pid, c.uid, c.gid)
        });
        self.emit_scm_rights(
            net,
            cx,
            msg,
            &hdr,
            flags,
            scm_fds,
            creds,
            &mut msg_flags,
            mem,
        );
        let _ = mem.write(msg + 48, &(msg_flags as i32).to_le_bytes());
        if flags & MSG_TRUNC != 0 && msg_flags & MSG_TRUNC != 0 {
            whole as i64
        } else {
            off as i64
        }
    }

    /// Materialize `scm_fds` (in-flight SCM_RIGHTS fds delivered with this
    /// `recvmsg`) into fresh guest fds and write the `SCM_RIGHTS` cmsg into the
    /// caller's `msg_control` buffer. Fds that don't fit the buffer are dropped
    /// (their backing ref released) and `MSG_CTRUNC` is set. Always writes
    /// `msg_controllen` (offset 40); may set bits in `msg_flags`.
    #[allow(clippy::too_many_arguments)]
    fn emit_scm_rights(
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
        msg: u64,
        hdr: &MsgHdr,
        flags: u64,
        scm_fds: Vec<Fd>,
        creds: Option<(i32, u32, u32)>,
        msg_flags: &mut u64,
        mem: &mut GuestMemory,
    ) {
        // SCM_CREDENTIALS first, as Linux puts it: `struct ucred` in a
        // CMSG_SPACE(12) = 32-byte message.
        let mut used = 0usize;
        if let Some((pid, uid, gid)) = creds {
            const CRED_SPACE: usize = CMSG_HDR_LEN + 16;
            if hdr.control != 0 && hdr.controllen >= CRED_SPACE as u64 {
                let mut c = Vec::with_capacity(CRED_SPACE);
                c.extend_from_slice(&((CMSG_HDR_LEN + 12) as u64).to_le_bytes());
                c.extend_from_slice(&(SOL_SOCKET as i32).to_le_bytes());
                c.extend_from_slice(&SCM_CREDENTIALS.to_le_bytes());
                c.extend_from_slice(&pid.to_le_bytes());
                c.extend_from_slice(&uid.to_le_bytes());
                c.extend_from_slice(&gid.to_le_bytes());
                c.extend_from_slice(&[0; 4]);
                let _ = mem.write(hdr.control, &c);
                used = CRED_SPACE;
            } else {
                *msg_flags |= MSG_CTRUNC;
            }
        }
        if scm_fds.is_empty() {
            let _ = mem.write(msg + 40, &(used as u64).to_le_bytes());
            return;
        }
        // How many fds fit in the caller's control buffer.
        let room = (hdr.controllen as usize).saturating_sub(used);
        let capacity = if hdr.control != 0 && room >= CMSG_HDR_LEN {
            (room - CMSG_HDR_LEN) / 4
        } else {
            0
        };
        let deliver = capacity.min(scm_fds.len());
        if deliver < scm_fds.len() {
            *msg_flags |= MSG_CTRUNC;
        }
        // Release the backing ref of every fd that won't fit (they are lost,
        // exactly as Linux drops overflowing SCM_RIGHTS fds).
        for fd in &scm_fds[deliver..] {
            self.scm_ref(net, fd, false);
        }
        let cloexec = flags & MSG_CMSG_CLOEXEC != 0;
        let mut guest_fds: Vec<i32> = Vec::with_capacity(deliver);
        for fd in scm_fds.into_iter().take(deliver) {
            // The in-flight duplicate's ref transfers to this new guest fd, so
            // no extra bump is needed.
            let g = cx.cur.fds.alloc(fd);
            cx.cur.fds.set_cloexec(g, cloexec);
            guest_fds.push(g);
        }
        if deliver == 0 {
            let _ = mem.write(msg + 40, &(used as u64).to_le_bytes());
            return;
        }
        let cmsg_len = CMSG_HDR_LEN + 4 * deliver;
        let mut cbuf = Vec::with_capacity(cmsg_len);
        cbuf.extend_from_slice(&(cmsg_len as u64).to_le_bytes());
        cbuf.extend_from_slice(&(SOL_SOCKET as i32).to_le_bytes());
        cbuf.extend_from_slice(&SCM_RIGHTS.to_le_bytes());
        for g in &guest_fds {
            cbuf.extend_from_slice(&g.to_le_bytes());
        }
        let _ = mem.write(hdr.control + used as u64, &cbuf);
        let _ = mem.write(msg + 40, &((used + cmsg_len) as u64).to_le_bytes());
    }

    /// Receive up to `cap` bytes from socket `fd` into a host `Vec` — the
    /// byte-level `recv` that `splice` (socket → pipe) needs. Any SCM_RIGHTS
    /// fds riding the bytes are released (a splice carries no ancillary data,
    /// like a plain `read`). An empty `Ok` with `cx.block` set means "would
    /// block"; an empty `Ok` without it is EOF.
    pub(super) fn recv_fd_bytes(
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
        fd: u64,
        cap: u64,
        flags: u64,
    ) -> Result<Vec<u8>, i64> {
        let (_, mut data, _, fds) = self.recv_message(net, cx, fd, cap, flags)?;
        for f in &fds {
            self.scm_ref(net, f, false);
        }
        data.truncate(cap as usize);
        Ok(data)
    }

    /// Receive one message's bytes (up to `cap`) from socket `fd` into a host
    /// `Vec`, returning the source address (datagram) and out `msg_flags`. The
    /// shared core of `recvmsg`, factored out of `recvfrom` so both can drive
    /// the netlink / stream / datagram paths without a guest bounce buffer.
    fn recv_message(
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
        fd: u64,
        cap: u64,
        flags: u64,
    ) -> Result<RecvMsg, i64> {
        let Some((sock, end)) = self.sock_of(cx, fd) else {
            return Err(err(Errno::ENOTSOCK));
        };
        match &net.socks[sock].kind {
            Kind::Netlink(_) => {
                let nonblock = net.socks[sock].nonblock || flags & MSG_DONTWAIT != 0;
                match self.drain_netlink(net, sock, cap as usize, flags) {
                    Some(data) => {
                        let mf = if data.len() as u64 > cap {
                            MSG_TRUNC
                        } else {
                            0
                        };
                        Ok((None, data, mf, Vec::new()))
                    }
                    None => {
                        if nonblock {
                            Err(err(Errno::EAGAIN))
                        } else {
                            cx.block = true;
                            Ok((None, Vec::new(), 0, Vec::new()))
                        }
                    }
                }
            }
            Kind::Dgram(_) => {
                let nonblock = net.socks[sock].nonblock || flags & MSG_DONTWAIT != 0;
                match self.recv_dgram_msg(net, sock, flags) {
                    Some((from, data)) => {
                        let mf = if data.len() as u64 > cap {
                            MSG_TRUNC
                        } else {
                            0
                        };
                        Ok((Some(Addr::Inet(from)), data, mf, Vec::new()))
                    }
                    None => {
                        if nonblock {
                            Err(err(Errno::EAGAIN))
                        } else {
                            cx.block = true;
                            Ok((None, Vec::new(), 0, Vec::new()))
                        }
                    }
                }
            }
            Kind::Host(_) => {
                let bytes =
                    self.host_recv(net, cx, sock, cap as usize, flags & MSG_DONTWAIT != 0)?;
                let peer = match &net.socks[sock].kind {
                    Kind::Host(h) => Some(Addr::Inet(h.peer)),
                    _ => None,
                };
                Ok((peer, bytes, 0, Vec::new()))
            }
            _ => {
                // Stream: pull bytes directly out of the inbound queue (the
                // flag-aware `recv_stream` writes to guest memory, so replicate
                // its dequeue here against a host buffer instead), carrying any
                // SCM_RIGHTS fds that ride the consumed bytes.
                let (bytes, fds, cut) = self.recv_stream_bytes(net, cx, sock, end, cap, flags)?;
                let peer = match &net.socks[sock].kind {
                    Kind::Pair(p) => p.addr_seen_by(1 - end, end).map(Addr::Inet),
                    _ => None,
                };
                Ok((peer, bytes, if cut { MSG_TRUNC } else { 0 }, fds))
            }
        }
    }

    /// Pop (or, for `MSG_PEEK`, peek at) datagram socket `sock`'s next queued
    /// inbound `(source, payload)`. A pure query: the caller decides the
    /// block/`EAGAIN` behavior when this returns `None` (an empty queue).
    fn recv_dgram_msg(
        &self,
        net: &mut Net,
        sock: usize,
        flags: u64,
    ) -> Option<(InetAddr, Vec<u8>)> {
        // Pull any host-arrived datagrams (egress replies, e.g. DNS) into the
        // queue first, so they're visible to this dequeue.
        self.host_udp_drain(net, sock);
        let peek = flags & MSG_PEEK != 0;
        match &mut net.socks[sock].kind {
            Kind::Dgram(d) if peek => d.queue.front().cloned(),
            Kind::Dgram(d) => {
                let next = d.queue.pop_front();
                if let Some((_, payload)) = &next {
                    d.queued_bytes -= datagram_cost(payload);
                }
                next
            }
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
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
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
            let Some(mut data) = self.drain_netlink(net, sock, count as usize, 0) else {
                if nonblock {
                    return err(Errno::EAGAIN);
                }
                cx.block = true;
                return 0;
            };
            data.truncate(count as usize);
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
    fn host_recv(
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
        sock: usize,
        cap: usize,
        force_nonblock: bool,
    ) -> Result<Vec<u8>, i64> {
        let Kind::Host(h) = &mut net.socks[sock].kind else {
            return Err(err(Errno::EINVAL));
        };
        let nonblock = h.nonblock || force_nonblock;
        let mut buf = vec![0u8; cap.min(256 * 1024)];
        match h.conn.recv(&mut buf) {
            Ok(n) => {
                buf.truncate(n);
                net.socks[sock].edge_seq += 1;
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

    /// The next netlink "packet" queued for socket `sock`: as many whole
    /// messages as fit `want` bytes, or the first message alone. That's what a
    /// receive returns, the caller cutting it to its buffer; whatever doesn't
    /// fit is lost (datagram semantics). With `MSG_TRUNC` the packet is formed
    /// as for a 32 KiB buffer, the size a zero-length peek reports so the
    /// caller can size its buffer for the read that follows (iproute2 does
    /// this); with `MSG_PEEK` it stays queued.
    #[allow(clippy::unused_self)]
    fn drain_netlink(
        &self,
        net: &mut Net,
        sock: usize,
        want: usize,
        flags: u64,
    ) -> Option<Vec<u8>> {
        const TRUNC_PACKET: usize = 32 * 1024;
        let Kind::Netlink(nl) = &mut net.socks[sock].kind else {
            unreachable!("checked by caller")
        };
        let limit = if flags & MSG_TRUNC != 0 {
            want.max(TRUNC_PACKET)
        } else {
            want
        };
        let mut out = Vec::new();
        let mut taken = 0;
        for next in &nl.queue {
            if taken > 0 && out.len() + next.len() > limit {
                break;
            }
            out.extend_from_slice(next);
            taken += 1;
            if out.len() >= limit {
                break;
            }
        }
        if taken == 0 {
            return None;
        }
        if flags & MSG_PEEK == 0 {
            nl.queue.drain(..taken);
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
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
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
        // With MSG_TRUNC, a SOCK_SEQPACKET receive says how long the message was.
        let whole = match &net.socks[sock].kind {
            Kind::Pair(Pair { msgs: Some(m), .. }) if flags & MSG_TRUNC != 0 => {
                m[end].front().copied()
            }
            _ => None,
        };
        let data: Vec<u8> = match &mut net.socks[sock].kind {
            Kind::Pair(p) => {
                let n = p.msg_read_len(end, (count as usize).min(p.to[end].len()));
                if peek {
                    p.to[end].iter().take(n).copied().collect()
                } else {
                    p.to[end].drain(..n).collect()
                }
            }
            _ => return err(Errno::EINVAL),
        };
        // A plain `read`/`recv`/`recvfrom` can't carry ancillary data, so any
        // SCM_RIGHTS fds riding the consumed bytes are discarded (their backing
        // ref released), matching Linux.
        if !peek {
            let dropped = match &mut net.socks[sock].kind {
                Kind::Pair(p) => p.finish_read(end, data.len()).0,
                _ => Vec::new(),
            };
            for fd in &dropped {
                self.scm_ref(net, fd, false);
            }
        }
        if mem.write(buf, &data).is_err() {
            return err(Errno::EFAULT);
        }
        whole.unwrap_or(data.len()) as i64
    }

    /// Host-buffer twin of [`Self::recv_stream`]: dequeue up to `count` stream
    /// bytes into a `Vec` (for `recvmsg`, which scatters across iovecs rather
    /// than writing one guest region). Same block/`EAGAIN`/EOF semantics.
    #[allow(clippy::unused_self)]
    fn recv_stream_bytes(
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
        sock: usize,
        end: usize,
        count: u64,
        flags: u64,
    ) -> Result<(Vec<u8>, Vec<Fd>, bool), i64> {
        let (shut_rd, avail, peer_open, nonblock, scm_ready) = match &net.socks[sock].kind {
            Kind::Pair(p) => (
                p.shut_rd[end],
                p.to[end].len(),
                p.refs[1 - end] > 0 && !p.shut_wr[1 - end],
                p.nonblock[end] || flags & MSG_DONTWAIT != 0,
                matches!(p.scm[end].front(), Some(&(0, _))),
            ),
            _ => return Err(err(Errno::EINVAL)),
        };
        if shut_rd {
            return Ok((Vec::new(), Vec::new(), false));
        }
        // A pending zero-offset SCM group (fds sent with a zero-length message,
        // or whose preceding bytes are already drained) is deliverable even with
        // no data bytes queued.
        if avail == 0 && !scm_ready {
            if peer_open {
                if nonblock {
                    return Err(err(Errno::EAGAIN));
                }
                cx.block = true;
            }
            return Ok((Vec::new(), Vec::new(), false));
        }
        let peek = flags & MSG_PEEK != 0;
        match &mut net.socks[sock].kind {
            Kind::Pair(p) => {
                let want = (count as usize).min(p.to[end].len());
                let n = p.msg_read_len(end, p.scm_read_len(end, want));
                let data: Vec<u8> = if peek {
                    p.to[end].iter().take(n).copied().collect()
                } else {
                    p.to[end].drain(..n).collect()
                };
                // MSG_PEEK leaves the ancillary data queued too (a later real
                // recv still gets the fds); a real read transfers them out.
                let (fds, cut) = if peek {
                    let whole = p.msgs.as_ref().and_then(|m| m[end].front());
                    (Vec::new(), whole.is_some_and(|&len| len > n))
                } else {
                    p.finish_read(end, n)
                };
                Ok((data, fds, cut))
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
    pub(super) fn write_socket(
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
        sock: usize,
        end: usize,
        data: &[u8],
        nosignal: bool,
    ) -> i64 {
        let r = self.write_socket_inner(net, cx, sock, end, data, nosignal);
        if r > 0 {
            net.bump_edge(sock);
        }
        r
    }

    fn write_socket_inner(
        &self,
        net: &mut Net,
        cx: &mut ServiceCtx,
        sock: usize,
        end: usize,
        data: &[u8],
        nosignal: bool,
    ) -> i64 {
        if matches!(net.socks[sock].kind, Kind::Netlink(_)) {
            let privileged = self.capable_over(cx, cx.cur.ns.net, super::caps::CAP_NET_ADMIN);
            return self.handle_netlink_request(net, sock, data, privileged);
        }
        if matches!(net.socks[sock].kind, Kind::Dgram(_)) {
            // A connected `socketpair(AF_UNIX, SOCK_DGRAM)` end: deliver the
            // datagram straight into the peer slot's queue (no addresses).
            if let Kind::Dgram(Dgram {
                pair_peer: Some(tgt),
                ..
            }) = &net.socks[sock].kind
            {
                let tgt = *tgt;
                let src = InetAddr {
                    v6: false,
                    port: 0,
                    ip: [0; 16],
                };
                if let Kind::Dgram(td) = &mut net.socks[tgt].kind {
                    td.enqueue(src, data.to_vec());
                }
                return data.len() as i64;
            }
            let peer = match &net.socks[sock].kind {
                Kind::Dgram(d) => d.peer,
                _ => unreachable!("checked above"),
            };
            let Some(peer) = peer else {
                return err(Errno::ENOTCONN);
            };
            if matches!(
                net.socks[sock].kind,
                Kind::Dgram(Dgram { icmp: Some(_), .. })
            ) {
                return self.icmp_send(net, sock, data, peer);
            }
            // A connected socket's routable peer is reached through egress,
            // exactly like a `sendto` to it.
            if peer.is_multicast() {
                return self.multicast_send(net, sock, data, peer);
            }
            if !peer.valid_bind() {
                return self.host_udp_send(net, sock, data, peer);
            }
            let src = self.ensure_dgram_bound(net, sock);
            let key = route_key(net.socks[sock].netns, "udp", peer);
            if let Some(tgt) = net.pick_dgram_target(&key, src.port)
                && let Kind::Dgram(td) = &mut net.socks[tgt].kind
            {
                td.enqueue(src, data.to_vec());
            }
            return data.len() as i64;
        }
        if let Kind::Host(h) = &mut net.socks[sock].kind {
            if h.wr_shut {
                if !nosignal {
                    self.raise_sigpipe(cx);
                }
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
        // Linux doubles SO_SNDBUF for its bookkeeping, and so reports it.
        let capacity = net.socks[sock].opts.sndbuf.saturating_mul(2) as usize;
        match &mut net.socks[sock].kind {
            Kind::Pair(p) => {
                p.sender[1 - end] = Ucred::of(cx);
                if p.shut_wr[end] || p.refs[1 - end] == 0 {
                    cx.cur.io_progress = 0;
                    if !nosignal {
                        self.raise_sigpipe(cx);
                    }
                    return err(Errno::EPIPE);
                }
                // A SOCK_SEQPACKET send is one message, queued whole or not
                // at all.
                if let Some(m) = &mut p.msgs {
                    if data.len() > capacity {
                        return err(Errno::EMSGSIZE);
                    }
                    if capacity.saturating_sub(p.to[1 - end].len()) < data.len() {
                        if p.nonblock[end] || cx.dontwait {
                            return err(Errno::EAGAIN);
                        }
                        cx.block = true;
                        return 0;
                    }
                    if !data.is_empty() {
                        m[1 - end].push_back(data.len());
                        p.to[1 - end].extend(data.iter().copied());
                    }
                    return data.len() as i64;
                }
                // As with pipes: what fits goes in, and a blocking send parks to
                // finish the rest, so the guest sees one complete send.
                let done = (cx.cur.io_progress as usize).min(data.len());
                let room = capacity.saturating_sub(p.to[1 - end].len());
                let n = room.min(data.len() - done);
                p.to[1 - end].extend(data[done..done + n].iter().copied());
                let done = done + n;
                if done == data.len() {
                    cx.cur.io_progress = 0;
                    return done as i64;
                }
                if p.nonblock[end] || cx.dontwait {
                    cx.cur.io_progress = 0;
                    return if done > 0 {
                        done as i64
                    } else {
                        err(Errno::EAGAIN)
                    };
                }
                cx.cur.io_progress = done as u64;
                cx.block = true;
                0
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
    fn handle_netlink_request(
        &self,
        net: &mut Net,
        sock: usize,
        data: &[u8],
        privileged: bool,
    ) -> i64 {
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
            // The family the request asks about (the first byte of its
            // `rtgenmsg`/`ifaddrmsg`/`rtmsg`); `AF_UNSPEC` (0) means all.
            let family = if nlmsg_len > 16 {
                data.get(offset + 16).copied().map_or(0, u16::from)
            } else {
                0
            };
            let wants = |f: u16| family == 0 || family == f;
            // A new namespace has only its loopback interface.
            let ns = net.socks[sock].netns;
            let link = net.link().filter(|_| ns == 0);
            let done = || encode_nlmsg(NLMSG_DONE, 0, nlmsg_seq, pid, &0i32.to_le_bytes());
            let body = &data[offset..(offset + nlmsg_len).min(data.len())];

            let replies: Vec<Vec<u8>> = if matches!(nlmsg_type, RTM_NEWLINK | RTM_SETLINK) && !dump
            {
                // `ip link set lo up/down`: struct ifinfomsg's flags and change.
                let errno = set_link(net, ns, privileged, body);
                if errno != 0 || want_ack {
                    vec![encode_nlmsgerr(-errno, &orig_hdr, nlmsg_seq, pid)]
                } else {
                    Vec::new()
                }
            } else if nlmsg_type == RTM_GETLINK && dump {
                let mut r: Vec<Vec<u8>> = net
                    .ifaces_of(ns)
                    .iter()
                    .map(|i| build_rtm_newlink(nlmsg_seq, pid, i))
                    .collect();
                r.push(done());
                r
            } else if nlmsg_type == RTM_GETADDR && dump {
                let mut r = Vec::new();
                for i in net.ifaces_of(ns) {
                    if let Some((ip, len)) = i.v4.filter(|_| wants(AF_INET)) {
                        r.push(build_rtm_newaddr(nlmsg_seq, pid, &i, AF_INET, &ip, len));
                    }
                    if let Some((ip, len)) = i.v6.filter(|_| wants(AF_INET6)) {
                        r.push(build_rtm_newaddr(nlmsg_seq, pid, &i, AF_INET6, &ip, len));
                    }
                }
                r.push(done());
                r
            } else if nlmsg_type == RTM_GETROUTE && dump {
                // The main table: `tun0`'s on-link prefixes and the default
                // routes through it (loopback routes live in the local table).
                let mut r = Vec::new();
                if let Some(l) = &link {
                    if let Some((ip, len)) = l.v4.filter(|_| wants(AF_INET)) {
                        let net = mask_prefix(&ip, len);
                        let fam = AF_INET;
                        r.push(build_rtm_newroute(nlmsg_seq, pid, fam, None, None));
                        r.push(build_rtm_newroute(
                            nlmsg_seq,
                            pid,
                            fam,
                            Some((&net, len)),
                            Some(&ip),
                        ));
                    }
                    if let Some((ip, len)) = l.v6.filter(|_| wants(AF_INET6)) {
                        let net = mask_prefix(&ip, len);
                        let fam = AF_INET6;
                        r.push(build_rtm_newroute(
                            nlmsg_seq,
                            pid,
                            fam,
                            Some((&net, len)),
                            None,
                        ));
                        r.push(build_rtm_newroute(nlmsg_seq, pid, fam, None, None));
                    }
                }
                r.push(done());
                r
            } else if nlmsg_type == RTM_GETROUTE {
                // A route lookup (`ip route get`) isn't modelled: no answer.
                vec![done()]
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
                // Replies wait for a read; a guest that only ever asks gets told.
                if nl.queue.len() + replies.len() > 4096 {
                    return err(Errno::ENOBUFS);
                }
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
    /// `msg_control`(32) / `msg_controllen`(40): the ancillary-data buffer.
    control: u64,
    controllen: u64,
}

impl MsgHdr {
    fn read(mem: &GuestMemory, ptr: u64) -> Option<Self> {
        Some(Self {
            name: mem.read_u64(ptr).ok()?,
            namelen: mem.read_u32(ptr + 8).ok()?,
            iov: mem.read_u64(ptr + 16).ok()?,
            iovlen: mem.read_u64(ptr + 24).ok()?,
            control: mem.read_u64(ptr + 32).ok()?,
            controllen: mem.read_u64(ptr + 40).ok()?,
        })
    }
}

/// Round `len` up to the next 8-byte `CMSG_ALIGN` boundary.
fn cmsg_align(len: usize) -> usize {
    (len + 7) & !7
}

/// Parse the guest ancillary buffer `[control, control+controllen)` for
/// `SOL_SOCKET`/`SCM_RIGHTS` control messages, returning every passed guest fd
/// number in order. Non-SCM_RIGHTS cmsgs are skipped. Returns `None` only on a
/// faulting read of the buffer.
fn parse_scm_rights(mem: &GuestMemory, control: u64, controllen: u64) -> Option<Vec<i32>> {
    if control == 0 || controllen < CMSG_HDR_LEN as u64 {
        return Some(Vec::new());
    }
    let buf = mem.read_vec(control, controllen as usize).ok()?;
    let mut fds = Vec::new();
    let mut off = 0usize;
    while off + CMSG_HDR_LEN <= buf.len() {
        let cmsg_len = u64::from_le_bytes(buf[off..off + 8].try_into().ok()?) as usize;
        let level = i32::from_le_bytes(buf[off + 8..off + 12].try_into().ok()?);
        let ctype = i32::from_le_bytes(buf[off + 12..off + 16].try_into().ok()?);
        if cmsg_len < CMSG_HDR_LEN || off + cmsg_len > buf.len() {
            break;
        }
        if level == SOL_SOCKET as i32 && ctype == SCM_RIGHTS {
            let data = &buf[off + CMSG_HDR_LEN..off + cmsg_len];
            for chunk in data.as_chunks::<4>().0 {
                fds.push(i32::from_le_bytes(*chunk));
            }
        }
        // Advance to the next CMSG_ALIGN-ed header; guard against a zero stride.
        let step = cmsg_align(cmsg_len).max(CMSG_HDR_LEN);
        off += step;
    }
    Some(fds)
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
            // Netlink replies come from the kernel: a full `sockaddr_nl`
            // (family, pad, nl_pid 0 = the kernel, nl_groups 0). busybox `ip`
            // rejects anything but its 12-byte size ("sender address length
            // == 2").
            AF_NETLINK => {
                let mut b = vec![0u8; 12];
                b[0..2].copy_from_slice(&AF_NETLINK.to_le_bytes());
                b
            }
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
    // Copy no more than the caller's buffer holds (`*optlen` in), and report
    // how much was copied (`*optlen` out) — Linux truncates, never overruns.
    let cap = if optlen_ptr == 0 {
        value.len()
    } else {
        match mem.read_u32(optlen_ptr) {
            Ok(n) if (n as i32) < 0 => return err(Errno::EINVAL),
            Ok(n) => (n as usize).min(value.len()),
            Err(_) => return err(Errno::EFAULT),
        }
    };
    if optval != 0 && mem.write(optval, &value[..cap]).is_err() {
        return err(Errno::EFAULT);
    }
    if optlen_ptr != 0 {
        let _ = mem.write(optlen_ptr, &(cap as u32).to_le_bytes());
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

/// One guest-visible network interface, as netlink dumps and `SIOCGIF*`
/// ioctls describe it.
pub(super) struct Iface {
    name: &'static str,
    index: i32,
    /// `ifi_type` / the hardware address family (`ARPHRD_*`).
    hwtype: u16,
    flags: u32,
    mtu: u32,
    txqlen: u32,
    operstate: u8,
    v4: Option<([u8; 4], u8)>,
    v6: Option<([u8; 16], u8)>,
}

/// The guest's interfaces: `lo`, which has its addresses only while up, then
/// `tun0` while the egress backend provides a link (`link`).
fn ifaces(link: Option<&Link>, lo_up: bool) -> Vec<Iface> {
    let mut v = vec![Iface {
        name: "lo",
        index: LOOPBACK_IFINDEX,
        hwtype: ARPHRD_LOOPBACK,
        flags: if lo_up {
            IFF_UP | IFF_LOOPBACK | IFF_RUNNING | IFF_LOWER_UP
        } else {
            IFF_LOOPBACK
        },
        mtu: 65_536,
        txqlen: 1000,
        operstate: if lo_up { IF_OPER_UNKNOWN } else { IF_OPER_DOWN },
        v4: lo_up.then_some(([127, 0, 0, 1], 8)),
        v6: lo_up.then(|| (loopback_ip(true), 128)),
    }];
    if let Some(l) = link {
        v.push(Iface {
            name: "tun0",
            index: TUN_IFINDEX,
            hwtype: ARPHRD_NONE,
            flags: IFF_UP
                | IFF_POINTOPOINT
                | IFF_RUNNING
                | IFF_NOARP
                | IFF_MULTICAST
                | IFF_LOWER_UP,
            mtu: l.mtu,
            txqlen: 500,
            operstate: IF_OPER_UP,
            v4: l.v4,
            v6: l.v6,
        });
    }
    v
}

/// `ip` with all but its first `len` bits cleared: the network of a prefix.
fn mask_prefix<const N: usize>(ip: &[u8; N], len: u8) -> [u8; N] {
    let mut out = [0u8; N];
    for (i, (o, b)) in out.iter_mut().zip(ip).enumerate() {
        let keep = (usize::from(len)).saturating_sub(i * 8).min(8);
        *o = if keep == 0 {
            0
        } else {
            b & (0xffu8 << (8 - keep))
        };
    }
    out
}

/// An `RTM_NEWLINK`/`RTM_SETLINK` request (`body` is the whole message) in
/// namespace `ns`: bring `lo` up or down, the one change modelled. Returns 0
/// or a positive errno. `tun0` stays up, and the other interfaces don't exist.
fn set_link(net: &mut Net, ns: u32, privileged: bool, body: &[u8]) -> i32 {
    // struct ifinfomsg { u8 family, pad; u16 type; i32 index; u32 flags, change; }
    let Some(msg) = body.get(16..32) else {
        return Errno::EINVAL.0;
    };
    let index = i32::from_le_bytes(msg[4..8].try_into().unwrap());
    let flags = u32::from_le_bytes(msg[8..12].try_into().unwrap());
    let mut change = u32::from_le_bytes(msg[12..16].try_into().unwrap());
    if change == 0 {
        change = u32::MAX; // the old convention: every flag given is meant
    }
    // An interface named by IFLA_IFNAME instead of its index.
    let named_lo = || {
        let mut at = 32;
        while at + 4 <= body.len() {
            let len = usize::from(u16::from_le_bytes([body[at], body[at + 1]]));
            let ty = u16::from_le_bytes([body[at + 2], body[at + 3]]);
            if len < 4 {
                break;
            }
            if ty == IFLA_IFNAME {
                return body
                    .get(at + 4..at + len)
                    .is_some_and(|n| n.split(|&b| b == 0).next() == Some(b"lo"));
            }
            at += (len + 3) & !3;
        }
        false
    };
    let is_lo = index == LOOPBACK_IFINDEX || (index == 0 && named_lo());
    let is_tun = index == TUN_IFINDEX && ns == 0 && net.link().is_some();
    if !is_lo && !is_tun {
        return Errno::ENODEV.0;
    }
    if !privileged {
        return Errno::EPERM.0;
    }
    if change & IFF_UP == 0 {
        return 0;
    }
    let up = flags & IFF_UP != 0;
    if is_tun {
        return if up { 0 } else { Errno::EPERM.0 };
    }
    net.set_lo(ns, up);
    0
}

/// `RTM_NEWLINK` describing interface `i`.
fn build_rtm_newlink(seq: u32, pid: u32, i: &Iface) -> Vec<u8> {
    let mut payload = vec![0u8; 16]; // struct ifinfomsg
    payload[2..4].copy_from_slice(&i.hwtype.to_le_bytes());
    payload[4..8].copy_from_slice(&i.index.to_le_bytes());
    payload[8..12].copy_from_slice(&i.flags.to_le_bytes());
    let mut name = i.name.as_bytes().to_vec();
    name.push(0);
    payload.extend(encode_rtattr(IFLA_IFNAME, &name));
    payload.extend(encode_rtattr(IFLA_MTU, &i.mtu.to_le_bytes()));
    payload.extend(encode_rtattr(IFLA_TXQLEN, &i.txqlen.to_le_bytes()));
    payload.extend(encode_rtattr(IFLA_QDISC, b"noqueue\0"));
    payload.extend(encode_rtattr(IFLA_OPERSTATE, &[i.operstate]));
    if i.hwtype == ARPHRD_LOOPBACK {
        payload.extend(encode_rtattr(IFLA_ADDRESS, &[0u8; 6]));
    }
    encode_nlmsg(RTM_NEWLINK, 0, seq, pid, &payload)
}

/// `RTM_NEWADDR` for address `ip`/`len` (family `family`) on interface `i`.
fn build_rtm_newaddr(seq: u32, pid: u32, i: &Iface, family: u16, ip: &[u8], len: u8) -> Vec<u8> {
    let mut payload = vec![0u8; 8]; // struct ifaddrmsg
    payload[0] = family as u8;
    payload[1] = len; // ifa_prefixlen
    payload[2] = IFA_F_PERMANENT; // no lifetime: not "dynamic"
    payload[3] = if i.index == LOOPBACK_IFINDEX {
        RT_SCOPE_HOST
    } else {
        RT_SCOPE_UNIVERSE
    };
    payload[4..8].copy_from_slice(&(i.index as u32).to_le_bytes());
    payload.extend(encode_rtattr(IFA_ADDRESS, ip));
    payload.extend(encode_rtattr(IFA_LOCAL, ip));
    if family == AF_INET {
        let mut label = i.name.as_bytes().to_vec();
        label.push(0);
        payload.extend(encode_rtattr(IFA_LABEL, &label));
    }
    encode_nlmsg(RTM_NEWADDR, 0, seq, pid, &payload)
}

/// `RTM_NEWROUTE` in the main table via `tun0`: to `dst` (network, prefix
/// length) — an on-link kernel route — or, with `dst` `None`, the default
/// route. `src` is the preferred source address, if any.
fn build_rtm_newroute(
    seq: u32,
    pid: u32,
    family: u16,
    dst: Option<(&[u8], u8)>,
    src: Option<&[u8]>,
) -> Vec<u8> {
    let mut payload = vec![0u8; 12]; // struct rtmsg
    payload[0] = family as u8;
    payload[1] = dst.map_or(0, |(_, len)| len); // rtm_dst_len
    payload[4] = RT_TABLE_MAIN;
    payload[5] = if dst.is_some() {
        RTPROT_KERNEL
    } else {
        RTPROT_STATIC
    };
    // IPv4 routes without a gateway are on-link; IPv6 ones are global.
    payload[6] = if family == AF_INET {
        RT_SCOPE_LINK
    } else {
        RT_SCOPE_UNIVERSE
    };
    payload[7] = RTN_UNICAST;
    payload.extend(encode_rtattr(
        RTA_TABLE,
        &u32::from(RT_TABLE_MAIN).to_le_bytes(),
    ));
    if let Some((net, _)) = dst {
        payload.extend(encode_rtattr(RTA_DST, net));
    }
    if let Some(src) = src {
        payload.extend(encode_rtattr(RTA_PREFSRC, src));
    }
    payload.extend(encode_rtattr(RTA_OIF, &(TUN_IFINDEX as u32).to_le_bytes()));
    encode_nlmsg(RTM_NEWROUTE, 0, seq, pid, &payload)
}

/// Interface-query ioctls (`SIOCGIF*`) for the guest's interfaces — the same
/// `lo` and `tun0` (see [`ifaces`]) the netlink dump reports. They operate on
/// a `struct ifreq` (a 16-byte name then a 24-byte union) at `arg`, except
/// `SIOCGIFCONF` which uses `struct ifconf`. A query for an unknown interface
/// is `ENODEV`. Returns 0 on success or a negative errno; the caller has
/// already checked `arg`'s fd is a socket.
pub(super) fn iface_ioctl(
    req: u32,
    arg: u64,
    mem: &mut GuestMemory,
    net: &mut Net,
    ns: u32,
    privileged: bool,
) -> i64 {
    const SIOCGIFNAME: u32 = 0x8910;
    const SIOCGIFCONF: u32 = 0x8912;
    const SIOCGIFFLAGS: u32 = 0x8913;
    const SIOCSIFFLAGS: u32 = 0x8914;
    const SIOCGIFADDR: u32 = 0x8915;
    const SIOCGIFDSTADDR: u32 = 0x8917;
    const SIOCGIFBRDADDR: u32 = 0x8919;
    const SIOCGIFNETMASK: u32 = 0x891b;
    const SIOCGIFMTU: u32 = 0x8921;
    const SIOCGIFHWADDR: u32 = 0x8927;
    const SIOCGIFINDEX: u32 = 0x8933;
    const SIOCGIFTXQLEN: u32 = 0x8942;
    const IFR_UNION: u64 = 16; // offset of the union after `char ifr_name[16]`
    const IFREQ_SZ: u32 = 40;

    // Write a `struct sockaddr_in { family, port=0, addr, 8 pad }` into the union.
    let put_sin = |mem: &mut GuestMemory, ip: [u8; 4]| -> bool {
        let mut sa = [0u8; 16];
        sa[0..2].copy_from_slice(&AF_INET.to_le_bytes());
        sa[4..8].copy_from_slice(&ip);
        mem.write(arg + IFR_UNION, &sa).is_ok()
    };
    let all = net.ifaces_of(ns);

    if req == SIOCGIFCONF {
        // struct ifconf { int ifc_len; /*4 pad*/ char* ifc_buf; } — one ifreq
        // per interface with an IPv4 address.
        let (Ok(len), Ok(buf)) = (mem.read_u32(arg), mem.read_u64(arg + 8)) else {
            return err(Errno::EFAULT);
        };
        let with_v4: Vec<&Iface> = all.iter().filter(|i| i.v4.is_some()).collect();
        let need = IFREQ_SZ * with_v4.len() as u32;
        if buf == 0 {
            // Report the buffer size the list needs.
            let _ = mem.write(arg, &need.to_le_bytes());
            return 0;
        }
        let fit = with_v4.iter().take((len / IFREQ_SZ) as usize);
        let mut written = 0u32;
        for i in fit {
            let mut e = [0u8; IFREQ_SZ as usize];
            e[..i.name.len()].copy_from_slice(i.name.as_bytes());
            e[16..18].copy_from_slice(&AF_INET.to_le_bytes());
            e[20..24].copy_from_slice(&i.v4.map_or([0; 4], |(ip, _)| ip));
            if mem.write(buf + u64::from(written), &e).is_err() {
                return err(Errno::EFAULT);
            }
            written += IFREQ_SZ;
        }
        let _ = mem.write(arg, &written.to_le_bytes());
        return 0;
    }
    if req == SIOCGIFNAME {
        // Keyed by ifr_ifindex.
        let Ok(idx) = mem.read_u32(arg + IFR_UNION) else {
            return err(Errno::EFAULT);
        };
        let Some(i) = all.iter().find(|i| i.index as u32 == idx) else {
            return err(Errno::ENODEV);
        };
        let mut name = [0u8; 16];
        name[..i.name.len()].copy_from_slice(i.name.as_bytes());
        return if mem.write(arg, &name).is_ok() {
            0
        } else {
            err(Errno::EFAULT)
        };
    }
    // The remaining queries are keyed by ifr_name.
    let Ok(name) = mem.read_vec(arg, 16) else {
        return err(Errno::EFAULT);
    };
    let end = name.iter().position(|&c| c == 0).unwrap_or(16);
    let Some(i) = all.iter().find(|i| i.name.as_bytes() == &name[..end]) else {
        return err(Errno::ENODEV);
    };
    let v4 = |mem: &mut GuestMemory, f: fn([u8; 4], u8) -> [u8; 4]| -> Option<bool> {
        i.v4.map(|(ip, len)| put_sin(mem, f(ip, len)))
    };
    if req == SIOCSIFFLAGS {
        // `lo` can be brought up or down; `tun0` stays up.
        let Ok(raw) = mem.read_u32(arg + IFR_UNION) else {
            return err(Errno::EFAULT);
        };
        if !privileged {
            return err(Errno::EPERM);
        }
        let up = raw & IFF_UP != 0;
        if i.index == LOOPBACK_IFINDEX {
            net.set_lo(ns, up);
        } else if !up {
            return err(Errno::EPERM);
        }
        return 0;
    }
    let ok = match req {
        SIOCGIFINDEX => mem.write(arg + IFR_UNION, &i.index.to_le_bytes()).is_ok(),
        SIOCGIFFLAGS => {
            // ifr_flags is a short: the low 16 flag bits.
            let flags = i.flags as u16;
            mem.write(arg + IFR_UNION, &flags.to_le_bytes()).is_ok()
        }
        SIOCGIFMTU => mem.write(arg + IFR_UNION, &i.mtu.to_le_bytes()).is_ok(),
        SIOCGIFTXQLEN => mem.write(arg + IFR_UNION, &i.txqlen.to_le_bytes()).is_ok(),
        SIOCGIFADDR => match v4(mem, |ip, _| ip) {
            Some(ok) => ok,
            None => return err(Errno::EADDRNOTAVAIL),
        },
        SIOCGIFNETMASK => match v4(mem, |_, len| {
            (u32::MAX.checked_shl(32 - u32::from(len)).unwrap_or(0)).to_be_bytes()
        }) {
            Some(ok) => ok,
            None => return err(Errno::EADDRNOTAVAIL),
        },
        SIOCGIFBRDADDR | SIOCGIFDSTADDR => put_sin(mem, [0, 0, 0, 0]),
        SIOCGIFHWADDR => {
            // sockaddr with sa_family = the ARPHRD type and a zero address.
            let mut sa = [0u8; 16];
            sa[0..2].copy_from_slice(&i.hwtype.to_le_bytes());
            mem.write(arg + IFR_UNION, &sa).is_ok()
        }
        _ => return err(Errno::ENOTTY),
    };
    if ok { 0 } else { err(Errno::EFAULT) }
}

/// Whether `req` is an interface-query ioctl handled by [`iface_ioctl`].
pub(super) fn is_iface_ioctl(req: u32) -> bool {
    matches!(
        req,
        0x8910
            | 0x8912
            | 0x8913
            | 0x8914
            | 0x8915
            | 0x8917
            | 0x8919
            | 0x891b
            | 0x8921
            | 0x8927
            | 0x8933
            | 0x8942
    )
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
        fn fork(&self) -> Result<Box<dyn Vcpu>, crate::vcpu::VcpuError> {
            Ok(Box::new(self.clone()))
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
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [a, msg, 2, 0, 0, 0]
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
                [b, out, 2, 0, 0, 0]
            ),
            2
        );
        assert_eq!(mem.read_vec(out, 2).unwrap(), b"hi");

        // The other direction is empty with the peer still open -> blocks.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Read,
                [a, out, 2, 0, 0, 0]
            ),
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

        let srv = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Socket,
            [1, 1, 0, 0, 0, 0],
        ) as u64;
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

        let cli = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Socket,
            [1, 1, 0, 0, 0, 0],
        ) as u64;
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
    fn closing_the_last_listener_fd_frees_the_address() {
        // A unix listener, shared by a dup: the address stays live until the
        // last fd closes, then a connect is refused and a fresh bind works.
        let (k, mut mem, mut v, mut cx) = setup();
        let addr = 0x1_1000;
        mem.write_init(addr, &1u16.to_le_bytes()).unwrap();
        mem.write_init(addr + 2, b"/srv\0").unwrap();
        let mut sys = |cx: &mut ServiceCtx, mem: &mut GuestMemory, nr, a: [u64; 6]| {
            call(&k, cx, mem, &mut v, nr, a)
        };
        let srv = sys(&mut cx, &mut mem, Sysno::Socket, [1, 1, 0, 0, 0, 0]) as u64;
        assert_eq!(
            sys(&mut cx, &mut mem, Sysno::Bind, [srv, addr, 7, 0, 0, 0]),
            0
        );
        assert_eq!(
            sys(&mut cx, &mut mem, Sysno::Listen, [srv, 8, 0, 0, 0, 0]),
            0
        );
        let dup = sys(&mut cx, &mut mem, Sysno::Dup, [srv, 0, 0, 0, 0, 0]) as u64;
        assert_eq!(
            sys(&mut cx, &mut mem, Sysno::Close, [srv, 0, 0, 0, 0, 0]),
            0
        );
        let cli = sys(&mut cx, &mut mem, Sysno::Socket, [1, 1, 0, 0, 0, 0]) as u64;
        assert_eq!(
            sys(&mut cx, &mut mem, Sysno::Connect, [cli, addr, 7, 0, 0, 0]),
            0
        );
        assert_eq!(
            sys(&mut cx, &mut mem, Sysno::Close, [dup, 0, 0, 0, 0, 0]),
            0
        );
        // The pending connection sees the server side gone: EOF.
        let out = 0x1_2000;
        assert_eq!(
            sys(&mut cx, &mut mem, Sysno::Read, [cli, out, 4, 0, 0, 0]),
            0
        );
        let cli2 = sys(&mut cx, &mut mem, Sysno::Socket, [1, 1, 0, 0, 0, 0]) as u64;
        assert_eq!(
            sys(&mut cx, &mut mem, Sysno::Connect, [cli2, addr, 7, 0, 0, 0]),
            -i64::from(Errno::ECONNREFUSED.0)
        );
    }

    #[test]
    fn connect_without_listener_is_refused() {
        let (k, mut mem, mut v, mut cx) = setup();
        let addr = 0x1_1000;
        mem.write_init(addr, &1u16.to_le_bytes()).unwrap();
        mem.write_init(addr + 2, b"/nope\0").unwrap();
        let cli = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Socket,
            [1, 1, 0, 0, 0, 0],
        ) as u64;
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
    fn nonblocking_tcp_connect_returns_einprogress() {
        // A nonblocking connect to a loopback listener returns EINPROGRESS (not
        // 0), matching Linux; the (already fully connected) socket is then
        // immediately writable and SO_ERROR reads 0.
        let (k, mut mem, mut v, mut cx) = setup();
        let addr = 0x1_1000;
        write_sockaddr_in(&mut mem, addr, [127, 0, 0, 1], 9100);
        let srv = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Socket,
            [2, 1, 0, 0, 0, 0],
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
        // AF_INET, SOCK_STREAM | SOCK_NONBLOCK.
        let cli = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Socket,
            [2, SOCK_STREAM | SOCK_NONBLOCK, 0, 0, 0, 0],
        ) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Connect,
                [cli, addr, 16, 0, 0, 0]
            ),
            -i64::from(Errno::EINPROGRESS.0),
        );
        // SO_ERROR reads 0 (the virtual handshake already completed).
        let out = 0x1_1300;
        let outlen = 0x1_1400;
        mem.write_init(outlen, &4u32.to_le_bytes()).unwrap();
        assert_eq!(
            k.sys_getsockopt(
                &mut k.net.lock().unwrap(),
                &mut cx,
                cli,
                SOL_SOCKET,
                SO_ERROR,
                out,
                outlen,
                &mut mem
            ),
            0
        );
        assert_eq!(mem.read_u32(out).unwrap(), 0);
    }

    #[test]
    fn nonblocking_unix_connect_completes_synchronously() {
        // AF_UNIX stream connects complete immediately even when nonblocking:
        // Linux returns 0, not EINPROGRESS.
        let (k, mut mem, mut v, mut cx) = setup();
        let addr = 0x1_1000;
        mem.write_init(addr, &1u16.to_le_bytes()).unwrap();
        mem.write_init(addr + 2, b"/sock\0").unwrap();
        let srv = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Socket,
            [1, 1, 0, 0, 0, 0],
        ) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Bind,
                [srv, addr, 8, 0, 0, 0]
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
        let cli = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Socket,
            [1, SOCK_STREAM | SOCK_NONBLOCK, 0, 0, 0, 0],
        ) as u64;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Connect,
                [cli, addr, 8, 0, 0, 0]
            ),
            0
        );
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
    fn unix_dgram_socketpair_roundtrip() {
        // socketpair(AF_UNIX, SOCK_DGRAM) yields a connected datagram pair:
        // a write on one end is a discrete datagram read on the other, with
        // MSG_TRUNC when the receive buffer is too small.
        let (k, mut mem, mut v, mut cx) = setup();
        let sv = 0x1_0000;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Socketpair,
                [1, SOCK_DGRAM, 0, sv, 0, 0]
            ),
            0
        );
        let a = u64::from(mem.read_u32(sv).unwrap());
        let b = u64::from(mem.read_u32(sv + 4).unwrap());
        assert!(a >= 3 && b >= 3 && a != b, "two distinct fds: {a},{b}");

        let msg = 0x1_1000;
        mem.write_init(msg, b"0123456789").unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [b, msg, 10, 0, 0, 0]
            ),
            10
        );
        // Read the whole datagram on the other end.
        let out = 0x1_1100;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Read,
                [a, out, 16, 0, 0, 0]
            ),
            10
        );
        assert_eq!(mem.read_vec(out, 10).unwrap(), b"0123456789");

        // A second datagram read into a short buffer truncates (recvfrom with
        // MSG_TRUNC reports the true length).
        mem.write_init(msg, b"ABCDEFGHIJ").unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [b, msg, 10, 0, 0, 0]
            ),
            10
        );
        const MSG_TRUNC: u64 = 0x20;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Recvfrom,
                [a, out, 4, MSG_TRUNC, 0, 0]
            ),
            10,
            "MSG_TRUNC reports the full datagram length"
        );
        assert_eq!(mem.read_vec(out, 4).unwrap(), b"ABCD");
    }

    #[test]
    fn scm_rights_passes_an_fd_over_a_unix_socketpair() {
        // sendmsg an SCM_RIGHTS cmsg carrying a pipe read end over an AF_UNIX
        // stream socketpair; recvmsg on the peer must materialize a fresh fd and
        // emit the cmsg, and the received fd must read the pipe's data.
        let (k, mut mem, mut v, mut cx) = setup();

        // Connected AF_UNIX stream socketpair -> sa, sb.
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
        let sa = u64::from(mem.read_u32(sv).unwrap());
        let sb = u64::from(mem.read_u32(sv + 4).unwrap());

        // A pipe; write a marker into the write end, pass the read end.
        let pfd = 0x1_0010;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Pipe2,
                [pfd, 0, 0, 0, 0, 0]
            ),
            0
        );
        let pr = u64::from(mem.read_u32(pfd).unwrap());
        let pw = u64::from(mem.read_u32(pfd + 4).unwrap());
        let marker = 0x1_0020;
        mem.write_init(marker, b"XYZ").unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [pw, marker, 3, 0, 0, 0]
            ),
            3
        );

        // Build the send msghdr: 1 data byte + an SCM_RIGHTS cmsg carrying `pr`.
        let (m, iov, dat, ctl) = (0x1_0100u64, 0x1_0140u64, 0x1_0180u64, 0x1_01c0u64);
        mem.write_init(dat, b"A").unwrap();
        mem.write_init(iov, &dat.to_le_bytes()).unwrap();
        mem.write_init(iov + 8, &1u64.to_le_bytes()).unwrap();
        // cmsg: cmsg_len(20), SOL_SOCKET(1), SCM_RIGHTS(1), fd(pr).
        mem.write_init(ctl, &20u64.to_le_bytes()).unwrap();
        mem.write_init(ctl + 8, &1i32.to_le_bytes()).unwrap();
        mem.write_init(ctl + 12, &1i32.to_le_bytes()).unwrap();
        mem.write_init(ctl + 16, &(pr as i32).to_le_bytes())
            .unwrap();
        // msghdr fields: name/namelen/iov/iovlen/control/controllen/flags.
        for (o, val) in [
            (0u64, 0u64),
            (8, 0),
            (16, iov),
            (24, 1),
            (32, ctl),
            (40, 24),
            (48, 0),
        ] {
            mem.write_init(m + o, &val.to_le_bytes()).unwrap();
        }
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Sendmsg,
                [sa, m, 0, 0, 0, 0]
            ),
            1
        );

        // Build the recv msghdr and recvmsg on the peer.
        let (rm, riov, rdat, rctl) = (0x1_0200u64, 0x1_0240u64, 0x1_0280u64, 0x1_02c0u64);
        mem.write_init(riov, &rdat.to_le_bytes()).unwrap();
        mem.write_init(riov + 8, &4u64.to_le_bytes()).unwrap();
        for (o, val) in [
            (0u64, 0u64),
            (8, 0),
            (16, riov),
            (24, 1),
            (32, rctl),
            (40, 24),
            (48, 0),
        ] {
            mem.write_init(rm + o, &val.to_le_bytes()).unwrap();
        }
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Recvmsg,
                [sb, rm, 0, 0, 0, 0]
            ),
            1
        );
        assert_eq!(mem.read_vec(rdat, 1).unwrap(), b"A");
        // The cmsg is present: controllen == CMSG_LEN(4) == 20, level/type == 1.
        assert_eq!(mem.read_u64(rm + 40).unwrap(), 20, "msg_controllen");
        assert_eq!(mem.read_u32(rm + 48).unwrap(), 0, "no MSG_CTRUNC");
        assert_eq!(mem.read_u64(rctl).unwrap(), 20, "cmsg_len");
        assert_eq!(mem.read_u32(rctl + 8).unwrap(), 1, "SOL_SOCKET");
        assert_eq!(mem.read_u32(rctl + 12).unwrap(), 1, "SCM_RIGHTS");
        let gotfd = u64::from(mem.read_u32(rctl + 16).unwrap());
        assert!(gotfd >= 3, "materialized a fresh guest fd: {gotfd}");

        // The received fd reads the pipe marker.
        let out = 0x1_0300;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Read,
                [gotfd, out, 3, 0, 0, 0]
            ),
            3
        );
        assert_eq!(mem.read_vec(out, 3).unwrap(), b"XYZ");
    }

    #[test]
    fn scm_rights_ctrunc_when_control_buffer_too_small() {
        // A recvmsg whose msg_control can't hold the cmsg sets MSG_CTRUNC and
        // drops the fd (no cmsg emitted).
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
        let sa = u64::from(mem.read_u32(sv).unwrap());
        let sb = u64::from(mem.read_u32(sv + 4).unwrap());
        let pfd = 0x1_0010;
        call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Pipe2,
            [pfd, 0, 0, 0, 0, 0],
        );
        let pr = u64::from(mem.read_u32(pfd).unwrap());

        let (m, iov, dat, ctl) = (0x1_0100u64, 0x1_0140u64, 0x1_0180u64, 0x1_01c0u64);
        mem.write_init(dat, b"A").unwrap();
        mem.write_init(iov, &dat.to_le_bytes()).unwrap();
        mem.write_init(iov + 8, &1u64.to_le_bytes()).unwrap();
        mem.write_init(ctl, &20u64.to_le_bytes()).unwrap();
        mem.write_init(ctl + 8, &1i32.to_le_bytes()).unwrap();
        mem.write_init(ctl + 12, &1i32.to_le_bytes()).unwrap();
        mem.write_init(ctl + 16, &(pr as i32).to_le_bytes())
            .unwrap();
        for (o, val) in [
            (0u64, 0u64),
            (8, 0),
            (16, iov),
            (24, 1),
            (32, ctl),
            (40, 24),
            (48, 0),
        ] {
            mem.write_init(m + o, &val.to_le_bytes()).unwrap();
        }
        call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Sendmsg,
            [sa, m, 0, 0, 0, 0],
        );

        // recv with a control buffer too small for even the cmsg header.
        let (rm, riov, rdat, rctl) = (0x1_0200u64, 0x1_0240u64, 0x1_0280u64, 0x1_02c0u64);
        mem.write_init(riov, &rdat.to_le_bytes()).unwrap();
        mem.write_init(riov + 8, &4u64.to_le_bytes()).unwrap();
        for (o, val) in [
            (0u64, 0u64),
            (8, 0),
            (16, riov),
            (24, 1),
            (32, rctl),
            (40, 8),
            (48, 0),
        ] {
            mem.write_init(rm + o, &val.to_le_bytes()).unwrap();
        }
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Recvmsg,
                [sb, rm, 0, 0, 0, 0]
            ),
            1
        );
        const MSG_CTRUNC_BIT: u32 = 0x08;
        assert_eq!(mem.read_u64(rm + 40).unwrap(), 0, "no cmsg fit");
        assert_ne!(
            mem.read_u32(rm + 48).unwrap() & MSG_CTRUNC_BIT,
            0,
            "MSG_CTRUNC set"
        );
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
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Fstat,
                [a, st, 0, 0, 0, 0]
            ),
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

        let srv = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Socket,
            [2, 1, 0, 0, 0, 0],
        ) as u64;
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

        let cli = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Socket,
            [2, 1, 0, 0, 0, 0],
        ) as u64;
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

        let srv = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Socket,
            [10, 1, 0, 0, 0, 0],
        ) as u64;
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

        let cli = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Socket,
            [10, 1, 0, 0, 0, 0],
        ) as u64;
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
        let s = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Socket,
            [2, 1, 0, 0, 0, 0],
        ) as u64;
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

        let a = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Socket,
            [2, 2, 0, 0, 0, 0],
        ) as u64;
        let b = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Socket,
            [2, 2, 0, 0, 0, 0],
        ) as u64;
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
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [a, msg, 2, 0, 0, 0]
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
                [b, out, 2, 0, 0, 0]
            ),
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
        assert_eq!(
            k.sys_bind(&mut k.net.lock().unwrap(), &mut cx, a, a_addr, 16, &mem),
            0
        );
        assert_eq!(
            k.sys_bind(&mut k.net.lock().unwrap(), &mut cx, b, b_addr, 16, &mem),
            0
        );

        let msg = 0x1_1200;
        mem.write_init(msg, b"hello").unwrap();
        assert_eq!(
            k.sys_sendto(
                &mut k.net.lock().unwrap(),
                &mut cx,
                a,
                msg,
                5,
                0,
                b_addr,
                16,
                &mem
            ),
            5
        );

        let out = 0x1_1300;
        let src = 0x1_1400;
        let srclen = 0x1_1500;
        mem.write_init(srclen, &16u32.to_le_bytes()).unwrap();
        assert_eq!(
            k.sys_recvfrom(
                &mut k.net.lock().unwrap(),
                &mut cx,
                b,
                out,
                5,
                0,
                src,
                srclen,
                &mut mem
            ),
            5
        );
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
        assert_eq!(
            k.sys_bind(&mut k.net.lock().unwrap(), &mut cx, a, addr, 16, &mem),
            0
        );

        let b = k.sys_socket(&mut k.net.lock().unwrap(), &mut cx, 2, 1, 0) as u64;
        // Without SO_REUSEADDR, binding the same port fails.
        assert_eq!(
            k.sys_bind(&mut k.net.lock().unwrap(), &mut cx, b, addr, 16, &mem),
            -i64::from(Errno::EADDRINUSE.0)
        );

        // Both sockets have to opt in: SO_REUSEADDR=1 on a, then on b.
        let optval = 0x1_1600;
        mem.write_init(optval, &1u32.to_le_bytes()).unwrap();
        assert_eq!(
            k.sys_setsockopt(
                &mut k.net.lock().unwrap(),
                &mut cx,
                a,
                SOL_SOCKET,
                SO_REUSEADDR,
                optval,
                4,
                &mem
            ),
            0
        );
        assert_eq!(
            k.sys_setsockopt(
                &mut k.net.lock().unwrap(),
                &mut cx,
                b,
                SOL_SOCKET,
                SO_REUSEADDR,
                optval,
                4,
                &mem
            ),
            0
        );
        assert_eq!(
            k.sys_bind(&mut k.net.lock().unwrap(), &mut cx, b, addr, 16, &mem),
            0
        );
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
            k.sys_setsockopt(
                &mut k.net.lock().unwrap(),
                &mut cx,
                s,
                SOL_SOCKET,
                SO_RCVBUF,
                optval,
                4,
                &mem
            ),
            0
        );
        mem.write_init(optval, &1u32.to_le_bytes()).unwrap();
        assert_eq!(
            k.sys_setsockopt(
                &mut k.net.lock().unwrap(),
                &mut cx,
                s,
                SOL_SOCKET,
                SO_REUSEADDR,
                optval,
                4,
                &mem
            ),
            0
        );

        let out = 0x1_1100;
        let outlen = 0x1_1200;
        mem.write_init(outlen, &4u32.to_le_bytes()).unwrap();
        assert_eq!(
            k.sys_getsockopt(
                &mut k.net.lock().unwrap(),
                &mut cx,
                s,
                SOL_SOCKET,
                SO_RCVBUF,
                out,
                outlen,
                &mut mem
            ),
            0
        );
        // Linux reports twice the requested value (kernel bookkeeping overhead).
        assert_eq!(mem.read_u32(out).unwrap(), 131_072);

        mem.write_init(outlen, &4u32.to_le_bytes()).unwrap();
        assert_eq!(
            k.sys_getsockopt(
                &mut k.net.lock().unwrap(),
                &mut cx,
                s,
                SOL_SOCKET,
                SO_REUSEADDR,
                out,
                outlen,
                &mut mem
            ),
            0
        );
        assert_eq!(mem.read_u32(out).unwrap(), 1);
    }

    #[test]
    fn so_acceptconn_is_one_after_listen() {
        let (k, mut mem, mut v, mut cx) = setup();
        let addr = 0x1_1000;
        write_sockaddr_in(&mut mem, addr, [127, 0, 0, 1], 9800);
        let srv = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Socket,
            [2, 1, 0, 0, 0, 0],
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

        let out = 0x1_1100;
        let outlen = 0x1_1200;
        mem.write_init(outlen, &4u32.to_le_bytes()).unwrap();
        assert_eq!(
            k.sys_getsockopt(
                &mut k.net.lock().unwrap(),
                &mut cx,
                srv,
                SOL_SOCKET,
                SO_ACCEPTCONN,
                out,
                outlen,
                &mut mem
            ),
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
            k.sys_getsockopt(
                &mut k.net.lock().unwrap(),
                &mut cx,
                srv,
                SOL_SOCKET,
                SO_ACCEPTCONN,
                out,
                outlen,
                &mut mem
            ),
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
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Write,
                [a, msg, 6, 0, 0, 0]
            ),
            6
        );

        let out = 0x1_2000;
        // Two MSG_PEEK reads in a row see the same bytes: nothing is consumed.
        assert_eq!(
            k.sys_recvfrom(
                &mut k.net.lock().unwrap(),
                &mut cx,
                b,
                out,
                6,
                MSG_PEEK,
                0,
                0,
                &mut mem
            ),
            6
        );
        assert_eq!(mem.read_vec(out, 6).unwrap(), b"peekme");
        assert_eq!(
            k.sys_recvfrom(
                &mut k.net.lock().unwrap(),
                &mut cx,
                b,
                out,
                6,
                MSG_PEEK,
                0,
                0,
                &mut mem
            ),
            6
        );
        assert_eq!(mem.read_vec(out, 6).unwrap(), b"peekme");

        // A real (non-peek) read now drains it...
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Read,
                [b, out, 6, 0, 0, 0]
            ),
            6
        );
        assert_eq!(mem.read_vec(out, 6).unwrap(), b"peekme");
        // ...so a further read blocks (the peer end is still open).
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Read,
                [b, out, 6, 0, 0, 0]
            ),
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

        let srv = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Socket,
            [1, 1, 0, 0, 0, 0],
        ) as u64;
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

        let cli = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Socket,
            [1, 1, 0, 0, 0, 0],
        ) as u64;
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
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Read,
                [b, out, 4, 0, 0, 0]
            ),
            0
        );
        assert!(!cx.block);

        // a itself can no longer write.
        let msg = 0x1_2000;
        mem.write_init(msg, b"x").unwrap();
        let ret = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Write,
            [a, msg, 1, 0, 0, 0],
        );
        assert_eq!(ret, -i64::from(Errno::EPIPE.0));
    }

    #[test]
    fn getpeername_on_unconnected_returns_enotconn() {
        let (k, mut mem, mut v, mut cx) = setup();
        let s = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Socket,
            [2, 1, 0, 0, 0, 0],
        ) as u64;
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

    /// An egress backend that only provides a link, to show `tun0`.
    #[derive(Debug)]
    struct LinkOnly;
    impl Egress for LinkOnly {
        fn connect_tcp(&self, _: [u8; 16], _: bool, _: u16) -> std::io::Result<Box<dyn HostConn>> {
            Err(std::io::ErrorKind::Unsupported.into())
        }
        fn open_udp(&self) -> std::io::Result<Box<dyn HostDgram>> {
            Err(std::io::ErrorKind::Unsupported.into())
        }
        fn link(&self) -> Option<Link> {
            let mut v6 = [0u8; 16];
            v6[..4].copy_from_slice(&[0x26, 0x05, 0x57, 0xc0]);
            v6[15] = 1;
            Some(Link {
                mtu: 1400,
                v4: Some(([100, 64, 0, 2], 10)),
                v6: Some((v6, 64)),
            })
        }
    }

    /// An egress whose connections record the options applied to them.
    #[derive(Debug, Default)]
    struct Recording(std::sync::Arc<std::sync::Mutex<Vec<super::super::egress::ConnOption>>>);
    #[derive(Debug)]
    struct RecordingConn(std::sync::Arc<std::sync::Mutex<Vec<super::super::egress::ConnOption>>>);
    impl HostConn for RecordingConn {
        fn recv(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::ErrorKind::WouldBlock.into())
        }
        fn send(&mut self, b: &[u8]) -> std::io::Result<usize> {
            Ok(b.len())
        }
        fn shutdown_write(&mut self) {}
        fn poll_readable(&mut self) -> bool {
            false
        }
        fn set_option(&mut self, opt: super::super::egress::ConnOption) {
            self.0.lock().unwrap().push(opt);
        }
    }
    impl Egress for Recording {
        fn connect_tcp(&self, _: [u8; 16], _: bool, _: u16) -> std::io::Result<Box<dyn HostConn>> {
            Ok(Box::new(RecordingConn(self.0.clone())))
        }
        fn open_udp(&self) -> std::io::Result<Box<dyn HostDgram>> {
            Err(std::io::ErrorKind::Unsupported.into())
        }
    }

    #[test]
    fn keepalive_and_linger_reach_the_host_connection() {
        use super::super::egress::ConnOption;
        let (mut k, mut mem, mut v, mut cx) = setup();
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        k.set_egress(Box::new(Recording(seen.clone())));
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Socket,
            [u64::from(AF_INET), SOCK_STREAM, 0, 0, 0, 0],
        ) as u64;
        let val = 0x1_1000;
        let mut set = |level: u64, name: u64, bytes: &[u8]| {
            mem.write(val, bytes).unwrap();
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Setsockopt,
                [fd, level, name, val, bytes.len() as u64, 0],
            )
        };
        // Set before connecting: applied when the connection is made.
        assert_eq!(set(SOL_SOCKET, SO_KEEPALIVE, &1u32.to_le_bytes()), 0);
        assert_eq!(set(IPPROTO_TCP, TCP_KEEPIDLE, &30u32.to_le_bytes()), 0);
        // Out of Linux's range.
        assert_eq!(
            set(IPPROTO_TCP, TCP_KEEPCNT, &0u32.to_le_bytes()),
            err(Errno::EINVAL)
        );
        assert_eq!(
            set(IPPROTO_TCP, TCP_KEEPINTVL, &40000u32.to_le_bytes()),
            err(Errno::EINVAL)
        );
        let sa = 0x1_2000;
        let mut b = [0u8; 16];
        b[0..2].copy_from_slice(&AF_INET.to_le_bytes());
        b[2..4].copy_from_slice(&80u16.to_be_bytes());
        b[4..8].copy_from_slice(&[93, 184, 216, 34]);
        mem.write(sa, &b).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Connect,
                [fd, sa, 16, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            *seen.lock().unwrap(),
            [ConnOption::KeepAlive(true), ConnOption::KeepIdle(30)]
        );
        // Set on the live connection: applied at once. Only a zero linger resets.
        seen.lock().unwrap().clear();
        let mut set = |level: u64, name: u64, bytes: &[u8]| {
            mem.write(val, bytes).unwrap();
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Setsockopt,
                [fd, level, name, val, bytes.len() as u64, 0],
            )
        };
        assert_eq!(set(IPPROTO_TCP, TCP_KEEPCNT, &4u32.to_le_bytes()), 0);
        assert_eq!(set(SOL_SOCKET, SO_LINGER, &[1, 0, 0, 0, 5, 0, 0, 0]), 0);
        assert_eq!(set(SOL_SOCKET, SO_LINGER, &[1, 0, 0, 0, 0, 0, 0, 0]), 0);
        assert_eq!(set(SOL_SOCKET, SO_REUSEADDR, &1u32.to_le_bytes()), 0);
        assert_eq!(
            *seen.lock().unwrap(),
            [
                ConnOption::KeepCount(4),
                ConnOption::ResetOnClose(false),
                ConnOption::ResetOnClose(true)
            ]
        );
    }

    /// An egress whose connection reports a settable count of unacknowledged
    /// bytes, and whether its write side was shut.
    #[derive(Debug, Default, Clone)]
    struct Unacked(
        std::sync::Arc<(
            std::sync::atomic::AtomicUsize,
            std::sync::atomic::AtomicBool,
        )>,
    );
    use std::sync::atomic::Ordering;
    impl HostConn for Unacked {
        fn recv(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            Err(std::io::ErrorKind::WouldBlock.into())
        }
        fn send(&mut self, b: &[u8]) -> std::io::Result<usize> {
            Ok(b.len())
        }
        fn shutdown_write(&mut self) {
            self.0.1.store(true, Ordering::Relaxed);
        }
        fn poll_readable(&mut self) -> bool {
            false
        }
        fn unsent(&mut self) -> Option<usize> {
            Some(self.0.0.load(Ordering::Relaxed))
        }
    }
    impl Egress for Unacked {
        fn connect_tcp(&self, _: [u8; 16], _: bool, _: u16) -> std::io::Result<Box<dyn HostConn>> {
            Ok(Box::new(self.clone()))
        }
        fn open_udp(&self) -> std::io::Result<Box<dyn HostDgram>> {
            Err(std::io::ErrorKind::Unsupported.into())
        }
    }

    #[test]
    fn a_nonzero_linger_makes_close_wait_for_the_peer_to_acknowledge() {
        use std::sync::atomic::Ordering;
        let (mut k, mut mem, mut v, mut cx) = setup();
        let conn = Unacked::default();
        conn.0.0.store(100, Ordering::Relaxed);
        k.set_egress(Box::new(conn.clone()));
        let fd = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Socket,
            [u64::from(AF_INET), SOCK_STREAM, 0, 0, 0, 0],
        ) as u64;
        mem.write(0x1_1000, &[1, 0, 0, 0, 2, 0, 0, 0]).unwrap(); // on, 2 s
        let r = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Setsockopt,
            [fd, SOL_SOCKET, SO_LINGER, 0x1_1000, 8, 0],
        );
        assert_eq!(r, 0);
        write_sockaddr_in(&mut mem, 0x1_2000, [93, 184, 216, 34], 80);
        let r = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Connect,
            [fd, 0x1_2000, 16, 0, 0, 0],
        );
        assert_eq!(r, 0);
        // Something is unacknowledged: the FIN goes out, and close waits.
        call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Close,
            [fd, 0, 0, 0, 0, 0],
        );
        assert!(cx.block && cx.cur.fds.get(fd as i32).is_some());
        assert!(conn.0.1.load(Ordering::Relaxed), "the FIN was sent");
        let wake = cx.cur.wake_deadline.unwrap();
        assert!(
            wake <= super::super::poll::now_ns() + 10_000_000,
            "polled, not slept"
        );
        // Past the timeout, the wait ends even though it never drained.
        let f = cx.cur.fds.get(fd as i32).cloned().unwrap();
        let later = super::super::poll::now_ns() + 3_000_000_000;
        assert_eq!(k.net.lock().unwrap().linger_step(&f, later), None);
        // Once acknowledged, the close goes ahead.
        cx.block = false;
        conn.0.0.store(0, Ordering::Relaxed);
        let r = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Close,
            [fd, 0, 0, 0, 0, 0],
        );
        assert_eq!(r, 0);
        assert!(!cx.block && cx.cur.fds.get(fd as i32).is_none());
    }

    #[test]
    fn netlink_dumps_show_the_egress_link_as_tun0() {
        let (mut k, mut mem, mut v, mut cx) = setup();
        k.set_egress(Box::new(LinkOnly));
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
        let mut dump = |ty: u16| {
            let req = 0x1_1000;
            write_nlmsghdr(&mut mem, req, ty, NLM_F_REQUEST | NLM_F_DUMP, 1);
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Sendto,
                [fd, req, 16, 0, 0, 0],
            );
            let out = 0x1_2000;
            let n = call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Recvfrom,
                [fd, out, 4096, 0, 0, 0],
            );
            parse_nlmsgs(&mem.read_vec(out, n as usize).unwrap())
        };

        let links = dump(RTM_GETLINK);
        let names: Vec<Vec<u8>> = links
            .iter()
            .filter(|(ty, ..)| *ty == RTM_NEWLINK)
            .filter_map(|(_, _, p)| find_rtattr(p, 16, IFLA_IFNAME))
            .collect();
        assert_eq!(names, [b"lo\0".to_vec(), b"tun0\0".to_vec()]);

        let addrs: Vec<Vec<u8>> = dump(RTM_GETADDR)
            .iter()
            .filter(|(ty, ..)| *ty == RTM_NEWADDR)
            .filter_map(|(_, _, p)| find_rtattr(p, 8, IFA_LOCAL))
            .collect();
        assert!(addrs.contains(&vec![100, 64, 0, 2]), "{addrs:?}");
        assert!(
            addrs
                .iter()
                .any(|a| a.len() == 16 && a[..4] == [0x26, 0x05, 0x57, 0xc0])
        );

        // Per family: a default route and the on-link prefix.
        let dsts: Vec<Option<Vec<u8>>> = dump(RTM_GETROUTE)
            .iter()
            .filter(|(ty, ..)| *ty == RTM_NEWROUTE)
            .map(|(_, _, p)| find_rtattr(p, 12, RTA_DST))
            .collect();
        assert_eq!(dsts.len(), 4, "{dsts:?}");
        assert!(dsts.contains(&Some(vec![100, 64, 0, 0])));
        assert_eq!(dsts.iter().filter(|d| d.is_none()).count(), 2);
    }

    #[test]
    fn mask_prefix_clears_host_bits() {
        assert_eq!(mask_prefix(&[100, 64, 0, 2], 10), [100, 64, 0, 0]);
        assert_eq!(mask_prefix(&[100, 127, 255, 255], 10), [100, 64, 0, 0]);
        assert_eq!(mask_prefix(&[10, 1, 2, 3], 0), [0, 0, 0, 0]);
        assert_eq!(mask_prefix(&[10, 1, 2, 3], 32), [10, 1, 2, 3]);
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
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Socketpair,
                [1, 1, 0, fds, 0, 0]
            ),
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
        mem.write_init(msg_out + 16, &iov_out.to_le_bytes())
            .unwrap();
        mem.write_init(msg_out + 24, &2u64.to_le_bytes()).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Sendmsg,
                [a, msg_out, 0, 0, 0, 0]
            ),
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
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Recvmsg,
                [b, msg_in, 0, 0, 0, 0]
            ),
            5,
            "recvmsg returns the full message"
        );
        assert_eq!(mem.read_vec(rb1, 2).unwrap(), b"he");
        assert_eq!(
            mem.read_vec(rb2, 3).unwrap(),
            b"llo",
            "scattered across iovecs"
        );
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
