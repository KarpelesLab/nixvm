//! Local (in-kernel) sockets: `socketpair`, AF_UNIX stream sockets, and
//! AF_INET loopback — all serviced entirely inside nixvm with no host
//! networking. External/internet TCP is out of scope (it needs a userspace IP
//! stack); this layer only connects endpoints that both live in this VM.
//!
//! A connected socket is a [`Pair`]: two byte buffers, one per direction, with
//! a small open-fd refcount per end — a bidirectional pipe. `socket()` yields an
//! unconnected [`Kind::Idle`] slot that `bind`/`listen` turn into a
//! [`Kind::Listener`] (a path plus a backlog of pending connections) and that
//! `connect` turns into a [`Pair`], queuing its index on the target listener for
//! `accept4` to hand back as the server-side end.

use std::collections::{BTreeMap, VecDeque};

use crate::abi::errno::Errno;
use crate::vcpu::GuestMemory;

use super::{Fd, Kernel, err};

const AF_UNIX: u16 = 1;
const AF_INET: u16 = 2;
const SOCK_STREAM: u64 = 1;

/// The kernel's socket table plus the AF_UNIX/AF_INET listener registry.
#[derive(Debug, Default)]
pub(super) struct Net {
    socks: Vec<Sock>,
    /// Address key (an AF_UNIX path or `"inet:ip:port"`) -> listening slot.
    listeners: BTreeMap<String, usize>,
}

/// One entry in the socket table.
#[derive(Debug)]
struct Sock {
    domain: u16,
    kind: Kind,
}

/// The lifecycle state of a socket slot.
#[derive(Debug)]
enum Kind {
    /// Created by `socket()`, optionally `bind`-ed to `bound` but not yet
    /// listening or connected.
    Idle { bound: Option<String> },
    /// `listen()`-ed: `backlog` holds the slot indices of pending connections
    /// (each already a [`Pair`]) waiting for `accept4`.
    Listener {
        path: Option<String>,
        backlog: VecDeque<usize>,
    },
    /// A connected pair of byte streams.
    Pair(Pair),
}

/// A connected socket: `to[e]` holds bytes destined for end `e` (so end `e`
/// reads `to[e]` and writes `to[1 - e]`). `refs[e]` counts open fds on end `e`;
/// `shut[e]` records a write-side `shutdown` from end `e`.
#[derive(Debug)]
struct Pair {
    to: [VecDeque<u8>; 2],
    refs: [usize; 2],
    shut: [bool; 2],
}

impl Pair {
    /// A freshly connected pair with one open reference on each end.
    fn new() -> Self {
        Self {
            to: [VecDeque::new(), VecDeque::new()],
            refs: [1, 1],
            shut: [false, false],
        }
    }
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
}

impl Kernel {
    /// The `(slot, end)` a socket fd points at, if it is a socket.
    fn sock_of(&self, fd: u64) -> Option<(usize, usize)> {
        match self.cur.fds.get(fd as i32) {
            Some(Fd::Socket { sock, end }) => Some((*sock, *end)),
            _ => None,
        }
    }

    /// `socket(domain, type, protocol)` — an unbound, unconnected endpoint.
    pub(super) fn sys_socket(&mut self, domain: u64, sotype: u64, _protocol: u64) -> i64 {
        let domain = domain as u16;
        if domain != AF_UNIX && domain != AF_INET {
            return err(Errno::EAFNOSUPPORT);
        }
        if sotype & 0xf != SOCK_STREAM {
            return err(Errno::EOPNOTSUPP);
        }
        let idx = self.net.socks.len();
        self.net.socks.push(Sock {
            domain,
            kind: Kind::Idle { bound: None },
        });
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
        let idx = self.net.socks.len();
        self.net.socks.push(Sock {
            domain: AF_UNIX,
            kind: Kind::Pair(Pair::new()),
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

    /// `bind(fd, addr, addrlen)` — record the local address of an idle socket.
    pub(super) fn sys_bind(&mut self, fd: u64, addr: u64, addrlen: u64, mem: &GuestMemory) -> i64 {
        let Some((sock, _)) = self.sock_of(fd) else {
            return err(Errno::ENOTSOCK);
        };
        let Some(key) = read_sockaddr(mem, addr, addrlen) else {
            return err(Errno::EINVAL);
        };
        match &mut self.net.socks[sock].kind {
            Kind::Idle { bound } => {
                *bound = Some(key);
                0
            }
            _ => err(Errno::EINVAL),
        }
    }

    /// `listen(fd, backlog)` — mark a bound socket as accepting connections.
    pub(super) fn sys_listen(&mut self, fd: u64) -> i64 {
        let Some((sock, _)) = self.sock_of(fd) else {
            return err(Errno::ENOTSOCK);
        };
        let bound = match &self.net.socks[sock].kind {
            Kind::Idle { bound } => bound.clone(),
            Kind::Listener { .. } => return 0,
            Kind::Pair(_) => return err(Errno::EINVAL),
        };
        self.net.socks[sock].kind = Kind::Listener {
            path: bound.clone(),
            backlog: VecDeque::new(),
        };
        if let Some(p) = bound {
            self.net.listeners.insert(p, sock);
        }
        0
    }

    /// `connect(fd, addr, addrlen)` — connect an idle socket to a listener,
    /// turning this slot into the client end of a fresh pair and queuing it on
    /// the listener for `accept4`.
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
        // An idle client end 0; anything else (already connected, a listener, or
        // the wrong end) is invalid.
        if !matches!(&self.net.socks[sock].kind, Kind::Idle { .. } if end == 0) {
            return err(Errno::EINVAL);
        }
        let Some(key) = read_sockaddr(mem, addr, addrlen) else {
            return err(Errno::EINVAL);
        };
        let Some(&lidx) = self.net.listeners.get(&key) else {
            return err(Errno::ECONNREFUSED);
        };
        if !matches!(self.net.socks[lidx].kind, Kind::Listener { .. }) {
            return err(Errno::ECONNREFUSED);
        }
        // Repurpose the client's idle slot as the connected pair, then queue its
        // index (the server-side end 1) on the listener's backlog.
        self.net.socks[sock].kind = Kind::Pair(Pair::new());
        if let Kind::Listener { backlog, .. } = &mut self.net.socks[lidx].kind {
            backlog.push_back(sock);
        }
        0
    }

    /// `accept4(fd, addr, addrlen, flags)` — hand back the server-side end of a
    /// pending connection (blocking, like pipe read, when none is queued).
    pub(super) fn sys_accept4(
        &mut self,
        fd: u64,
        addr: u64,
        addrlen: u64,
        _flags: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some((sock, _)) = self.sock_of(fd) else {
            return err(Errno::ENOTSOCK);
        };
        let pending = match &mut self.net.socks[sock].kind {
            Kind::Listener { backlog, .. } => backlog.pop_front(),
            _ => return err(Errno::EINVAL),
        };
        let Some(pidx) = pending else {
            self.block = true; // no pending connection yet — re-trap later
            return 0;
        };
        write_sockaddr(mem, addr, addrlen, AF_UNIX, None);
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
        let Some((sock, _)) = self.sock_of(fd) else {
            return err(Errno::ENOTSOCK);
        };
        let domain = self.net.socks[sock].domain;
        let name = match &self.net.socks[sock].kind {
            Kind::Idle { bound } => bound.clone(),
            Kind::Listener { path, .. } => path.clone(),
            Kind::Pair(_) => None,
        };
        write_sockaddr(mem, addr, addrlen, domain, name.as_deref())
    }

    /// `getpeername(fd, addr, addrlen)` — the peer address (best-effort).
    pub(super) fn sys_getpeername(
        &mut self,
        fd: u64,
        addr: u64,
        addrlen: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Some((sock, _)) = self.sock_of(fd) else {
            return err(Errno::ENOTSOCK);
        };
        let domain = self.net.socks[sock].domain;
        match &self.net.socks[sock].kind {
            Kind::Pair(_) => write_sockaddr(mem, addr, addrlen, domain, None),
            _ => err(Errno::EINVAL), // ENOTCONN
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
            _ => err(Errno::EINVAL), // ENOTCONN
        }
    }

    /// Read from socket `sock`'s inbound queue for `end`. Empty with the peer
    /// still open -> block; empty with the peer closed -> EOF (0). Mirrors
    /// `read_pipe`.
    pub(super) fn read_socket(
        &mut self,
        sock: usize,
        end: usize,
        buf: u64,
        count: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let (empty, peer_open) = match &self.net.socks[sock].kind {
            Kind::Pair(p) => (
                p.to[end].is_empty(),
                p.refs[1 - end] > 0 && !p.shut[1 - end],
            ),
            _ => return err(Errno::EINVAL),
        };
        if empty {
            if peer_open {
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

    /// Append to socket `sock`'s outbound queue for `end` (`EPIPE` if the peer
    /// end is fully closed). Mirrors `write_pipe`.
    pub(super) fn write_socket(&mut self, sock: usize, end: usize, data: &[u8]) -> i64 {
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

/// Decode a `sockaddr` into an address key: the `sun_path` for AF_UNIX, or
/// `"inet:a.b.c.d:port"` for AF_INET. Abstract (leading-NUL) AF_UNIX names are
/// treated as their raw path.
fn read_sockaddr(mem: &GuestMemory, ptr: u64, addrlen: u64) -> Option<String> {
    if addrlen < 2 {
        return None;
    }
    let bytes = mem.read_vec(ptr, (addrlen as usize).min(128)).ok()?;
    let family = u16::from_le_bytes([bytes[0], bytes[1]]);
    match family {
        AF_UNIX => {
            let path = &bytes[2..];
            let end = path.iter().position(|&c| c == 0).unwrap_or(path.len());
            Some(String::from_utf8_lossy(&path[..end]).into_owned())
        }
        AF_INET if bytes.len() >= 8 => {
            let port = u16::from_be_bytes([bytes[2], bytes[3]]);
            let ip = &bytes[4..8];
            Some(format!(
                "inet:{}.{}.{}.{}:{port}",
                ip[0], ip[1], ip[2], ip[3]
            ))
        }
        _ => None,
    }
}

/// Write a best-effort `sockaddr` (family plus, for AF_UNIX, `name`) to `addr`,
/// truncated to the caller's buffer, updating the `socklen_t` at `addrlen_ptr`.
/// A no-op when either pointer is null. Always returns success (0).
fn write_sockaddr(
    mem: &mut GuestMemory,
    addr: u64,
    addrlen_ptr: u64,
    family: u16,
    name: Option<&str>,
) -> i64 {
    if addrlen_ptr == 0 {
        return 0;
    }
    let mut buf = Vec::new();
    buf.extend_from_slice(&family.to_le_bytes());
    if let Some(p) = name {
        buf.extend_from_slice(p.as_bytes());
        buf.push(0);
    }
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
}
