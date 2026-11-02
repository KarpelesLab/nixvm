//! The socket-option namespace: which `(level, optname)` pairs a Linux 6.x
//! kernel knows for the socket families nixvm provides, and what an unset one
//! reads back as.
//!
//! `setsockopt`/`getsockopt` (in `net.rs`) model the options with behavior
//! directly; everything else *known* is stored and echoed back verbatim (so a
//! program that sets `TCP_KEEPIDLE` and reads it back sees its value), or
//! reports Linux's default when never set. A pair Linux doesn't know is
//! `ENOPROTOOPT` — and recorded as an unsupported subcommand — rather than a
//! silent success, so the gap is visible.
//!
//! [`refused`] is the policy for the rest: options that would need host
//! privileges, reach past the sandbox's network boundary, or promise a security
//! property nixvm doesn't provide. These fail, because a guest told "done" would
//! go on trusting a filter or a device binding that doesn't exist.

use crate::abi::errno::Errno;

/// `SOL_SOCKET`.
pub(super) const SOL_SOCKET: u64 = 1;
/// The protocol levels.
const SOL_IP: u64 = 0;
const SOL_TCP: u64 = 6;
const SOL_UDP: u64 = 17;
const SOL_IPV6: u64 = 41;
const SOL_ICMPV6: u64 = 58;
const SOL_RAW: u64 = 255;
const SOL_PACKET: u64 = 263;
const SOL_NETLINK: u64 = 270;

/// Whether Linux knows `optname` at `level` (for the families nixvm has:
/// inet/inet6 TCP/UDP/ICMP, unix, netlink).
#[allow(clippy::match_same_arms)] // a table: one arm per level, kept separate
pub(super) fn known(level: u64, optname: u64) -> bool {
    match level {
        // SO_DEBUG .. SO_PEERPIDFD, minus the never-assigned 54 and 58.
        SOL_SOCKET => {
            (1..=21).contains(&optname)
                || ((25..=77).contains(&optname) && optname != 54 && optname != 58)
        }
        // IP_TOS .. IP_LOCAL_PORT_RANGE (with the multicast block).
        SOL_IP => (1..=27).contains(&optname) || (32..=51).contains(&optname),
        // TCP_NODELAY .. TCP_IS_MPTCP.
        SOL_TCP => (1..=43).contains(&optname),
        // UDP_CORK, UDP_ENCAP, UDP_NO_CHECK6_*, UDP_SEGMENT, UDP_GRO.
        SOL_UDP => matches!(optname, 1 | 100..=104),
        // IPV6_ADDRFORM .. IPV6_RECVERR_RFC4884.
        SOL_IPV6 => (1..=80).contains(&optname),
        SOL_ICMPV6 => optname == 1, // ICMPV6_FILTER
        SOL_RAW => optname == 1,    // ICMP_FILTER
        SOL_NETLINK => (1..=12).contains(&optname),
        SOL_PACKET => (1..=23).contains(&optname),
        _ => false,
    }
}

/// Why `setsockopt` won't accept `(level, optname)`, or `None` if it will.
///
/// `EPERM` is for what an unprivileged container is denied (`CAP_NET_ADMIN`,
/// `CAP_NET_RAW`), `ENOPROTOOPT` for features the kernel simply lacks here,
/// which is the answer programs already fall back from.
#[allow(clippy::match_same_arms)] // a table: grouped by what each option does
pub(super) fn refused(level: u64, optname: u64) -> Option<Errno> {
    match (level, optname) {
        // SO_BINDTODEVICE, SO_SNDBUFFORCE, SO_RCVBUFFORCE, SO_MARK,
        // SO_BINDTOIFINDEX: device binding, buffer limits, routing marks.
        (SOL_SOCKET, 25 | 32 | 33 | 36 | 62) => Some(Errno::EPERM),
        // IP_HDRINCL, IP_ROUTER_ALERT, IP_FREEBIND, IP_IPSEC_POLICY,
        // IP_XFRM_POLICY, IP_TRANSPARENT: forged headers, spoofed or unowned
        // addresses, kernel IPsec state.
        (SOL_IP, 3 | 5 | 15 | 16 | 17 | 19) => Some(Errno::EPERM),
        // IPV6_ROUTER_ALERT, IPV6_IPSEC_POLICY, IPV6_XFRM_POLICY,
        // IPV6_TRANSPARENT, IPV6_FREEBIND: the same, for IPv6.
        (SOL_IPV6, 22 | 34 | 35 | 75 | 78) => Some(Errno::EPERM),
        // TCP_REPAIR, TCP_REPAIR_QUEUE, TCP_QUEUE_SEQ, TCP_REPAIR_OPTIONS,
        // TCP_REPAIR_WINDOW, TCP_AO_REPAIR: raw control of connection state.
        (SOL_TCP, 19..=22 | 29 | 42) => Some(Errno::EPERM),
        // Socket filters (SO_ATTACH_FILTER, SO_DETACH_FILTER, SO_LOCK_FILTER,
        // SO_ATTACH_BPF, SO_ATTACH_REUSEPORT_CBPF/EBPF,
        // SO_DETACH_REUSEPORT_BPF): nothing would run the program, so accepting
        // one would let unfiltered traffic through a filter the guest trusts.
        (SOL_SOCKET, 26 | 27 | 44 | 50..=52 | 68) => Some(Errno::ENOPROTOOPT),
        // TCP_MD5SIG, TCP_ULP, TCP_MD5SIG_EXT, TCP_AO_*: authentication and
        // kTLS, which nothing here would apply to the wire.
        (SOL_TCP, 14 | 31 | 32 | 38..=41) => Some(Errno::ENOPROTOOPT),
        // There are no packet sockets.
        (SOL_PACKET, _) => Some(Errno::ENOPROTOOPT),
        _ => None,
    }
}

/// The value `SO_SNDBUF`/`SO_RCVBUF` store for a guest's request: Linux caps it
/// at `net.core.[rw]mem_max` and floors it at half the minimum buffer (the
/// stored value is doubled on read).
pub(super) fn clamp_sock_buf(requested: u32) -> u32 {
    const MEM_MAX: u32 = 212_992;
    const MIN: u32 = 2304;
    requested.clamp(MIN, MEM_MAX)
}

/// Options whose value is an `int` (`optlen < 4` is `EINVAL` on set).
#[allow(clippy::match_same_arms)] // a table: one arm per level
pub(super) fn is_int(level: u64, optname: u64) -> bool {
    match level {
        SOL_SOCKET => !matches!(
            optname,
            13 | 20 | 21 | 25 | 26 | 28 | 31 | 37 | 55 | 59 | 61 | 65..=67
        ),
        SOL_TCP => !matches!(optname, 11 | 13 | 14 | 26 | 28 | 29 | 31 | 32 | 33 | 35),
        SOL_IP => !matches!(optname, 4 | 32 | 35..=48),
        SOL_IPV6 => !matches!(optname, 17 | 20 | 21 | 50 | 57),
        _ => true,
    }
}

/// What a never-set known option reads back as (Linux's defaults for a
/// fresh socket), or `None` for "zero of the natural width".
#[allow(clippy::match_same_arms)] // a table: one arm per option, named
pub(super) fn default_value(level: u64, optname: u64) -> Option<Vec<u8>> {
    let int = |v: i32| Some(v.to_le_bytes().to_vec());
    match (level, optname) {
        (SOL_SOCKET, 18) => int(1),  // SO_RCVLOWAT
        (SOL_SOCKET, 19) => int(1),  // SO_SNDLOWAT
        (SOL_SOCKET, 46) => int(0),  // SO_BUSY_POLL
        (SOL_SOCKET, 49) => int(0),  // SO_INCOMING_CPU
        (SOL_IP, 2) => int(64),      // IP_TTL
        (SOL_IP, 10) => int(1),      // IP_MTU_DISCOVER: IP_PMTUDISC_WANT
        (SOL_IP, 14) => int(1500),   // IP_MTU
        (SOL_IP, 33) => int(1),      // IP_MULTICAST_TTL
        (SOL_IP, 34) => int(1),      // IP_MULTICAST_LOOP
        (SOL_TCP, 2) => int(1448),   // TCP_MAXSEG
        (SOL_TCP, 4) => int(7200),   // TCP_KEEPIDLE
        (SOL_TCP, 5) => int(75),     // TCP_KEEPINTVL
        (SOL_TCP, 6) => int(9),      // TCP_KEEPCNT
        (SOL_TCP, 7) => int(6),      // TCP_SYNCNT
        (SOL_TCP, 8) => int(60),     // TCP_LINGER2
        (SOL_TCP, 10) => int(32768), // TCP_WINDOW_CLAMP
        (SOL_TCP, 12) => int(1),     // TCP_QUICKACK
        (SOL_TCP, 13) => Some(b"cubic\0\0\0\0\0\0\0\0\0\0\0".to_vec()), // TCP_CONGESTION
        (SOL_TCP, 25) => int(-1),    // TCP_NOTSENT_LOWAT: unlimited
        (SOL_IPV6, 16) => int(64),   // IPV6_UNICAST_HOPS
        (SOL_IPV6, 18) => int(1),    // IPV6_MULTICAST_HOPS
        (SOL_IPV6, 19) => int(1),    // IPV6_MULTICAST_LOOP
        (SOL_IPV6, 23) => int(1),    // IPV6_MTU_DISCOVER
        (SOL_IPV6, 24) => int(1500), // IPV6_MTU
        (SOL_TCP, 11) => {
            // TCP_INFO: a `struct tcp_info` for an established connection;
            // tcpi_state = TCP_ESTABLISHED (1), and sane rto/mss figures.
            let mut b = vec![0u8; 104];
            b[0] = 1;
            b[8..12].copy_from_slice(&200_000u32.to_le_bytes()); // tcpi_rto (µs)
            b[16..20].copy_from_slice(&1448u32.to_le_bytes()); // tcpi_snd_mss
            b[20..24].copy_from_slice(&1448u32.to_le_bytes()); // tcpi_rcv_mss
            Some(b)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{BASE, call, e, setup};
    use super::*;

    #[test]
    fn known_set_covers_the_common_options_and_rejects_nonsense() {
        assert!(known(SOL_SOCKET, 9)); // SO_KEEPALIVE
        assert!(known(SOL_SOCKET, 17)); // SO_PEERCRED
        assert!(!known(SOL_SOCKET, 54));
        assert!(known(SOL_TCP, 4)); // TCP_KEEPIDLE
        assert!(known(SOL_IPV6, 26)); // IPV6_V6ONLY
        assert!(!known(SOL_TCP, 999));
        assert!(!known(12345, 1));
        assert_eq!(default_value(SOL_TCP, 6), Some(9i32.to_le_bytes().to_vec()));
        assert!(is_int(SOL_TCP, 4) && !is_int(SOL_SOCKET, 13));
    }

    #[test]
    fn refused_covers_the_dangerous_options_and_leaves_the_everyday_ones() {
        // Everyday options stay available.
        for (level, opt) in [
            (SOL_SOCKET, 2),  // SO_REUSEADDR
            (SOL_SOCKET, 15), // SO_REUSEPORT
            (SOL_SOCKET, 9),  // SO_KEEPALIVE
            (SOL_SOCKET, 7),  // SO_SNDBUF
            (SOL_SOCKET, 20), // SO_RCVTIMEO
            (SOL_TCP, 1),     // TCP_NODELAY
            (SOL_TCP, 4),     // TCP_KEEPIDLE
            (SOL_IP, 1),      // IP_TOS
            (SOL_IP, 2),      // IP_TTL
            (SOL_IPV6, 26),   // IPV6_V6ONLY
        ] {
            assert_eq!(refused(level, opt), None, "({level}, {opt})");
        }
        for (level, opt, errno) in [
            (SOL_SOCKET, 25, Errno::EPERM),       // SO_BINDTODEVICE
            (SOL_SOCKET, 36, Errno::EPERM),       // SO_MARK
            (SOL_SOCKET, 33, Errno::EPERM),       // SO_RCVBUFFORCE
            (SOL_SOCKET, 26, Errno::ENOPROTOOPT), // SO_ATTACH_FILTER
            (SOL_SOCKET, 50, Errno::ENOPROTOOPT), // SO_ATTACH_BPF
            (SOL_IP, 3, Errno::EPERM),            // IP_HDRINCL
            (SOL_IP, 19, Errno::EPERM),           // IP_TRANSPARENT
            (SOL_IPV6, 75, Errno::EPERM),         // IPV6_TRANSPARENT
            (SOL_TCP, 19, Errno::EPERM),          // TCP_REPAIR
            (SOL_TCP, 14, Errno::ENOPROTOOPT),    // TCP_MD5SIG
            (SOL_TCP, 31, Errno::ENOPROTOOPT),    // TCP_ULP
            (SOL_PACKET, 1, Errno::ENOPROTOOPT),  // PACKET_ADD_MEMBERSHIP
        ] {
            assert_eq!(refused(level, opt), Some(errno), "({level}, {opt})");
        }
    }

    /// `socket(AF_INET, ty)`, with `opts` set to 1 at `SOL_SOCKET`, bound to
    /// loopback `port`.
    fn bound_socket(h: &mut Harness, ty: u64, opts: &[u64], port: u16) -> (u64, i64) {
        use crate::abi::arch::Sysno;
        let s = h.call(Sysno::Socket, [2, ty, 0, 0, 0, 0]) as u64;
        h.mem.write(BASE, &1u32.to_le_bytes()).unwrap();
        for &opt in opts {
            h.call(Sysno::Setsockopt, [s, SOL_SOCKET, opt, BASE, 4, 0]);
        }
        let addr = BASE + 0x100;
        let mut sa = [0u8; 16];
        sa[0] = 2; // AF_INET
        sa[2..4].copy_from_slice(&port.to_be_bytes());
        sa[4..8].copy_from_slice(&[127, 0, 0, 1]);
        h.mem.write(addr, &sa).unwrap();
        (s, h.call(Sysno::Bind, [s, addr, 16, 0, 0, 0]))
    }

    struct Harness {
        k: crate::kernel::Kernel,
        mem: crate::vcpu::GuestMemory,
        v: super::super::testutil::DummyVcpu,
        cx: crate::kernel::ServiceCtx,
    }

    impl Harness {
        fn new() -> Self {
            let (k, mem, v, cx) = setup();
            Self { k, mem, v, cx }
        }

        fn call(&mut self, sys: crate::abi::arch::Sysno, args: [u64; 6]) -> i64 {
            call(&self.k, &mut self.cx, &mut self.mem, &mut self.v, sys, args)
        }
    }

    const SO_REUSEADDR: u64 = 2;
    const SO_REUSEPORT: u64 = 15;
    const SOCK_STREAM: u64 = 1;
    const SOCK_DGRAM: u64 = 2;

    #[test]
    fn dangerous_options_fail_through_the_syscall_and_are_not_stored() {
        use crate::abi::arch::Sysno;
        let mut h = Harness::new();
        let s = h.call(Sysno::Socket, [2, SOCK_STREAM, 0, 0, 0, 0]) as u64;
        h.mem.write(BASE, &1u32.to_le_bytes()).unwrap();
        let set =
            |h: &mut Harness, level, opt| h.call(Sysno::Setsockopt, [s, level, opt, BASE, 4, 0]);
        assert_eq!(set(&mut h, SOL_SOCKET, 36), e(Errno::EPERM), "SO_MARK");
        assert_eq!(
            set(&mut h, SOL_SOCKET, 50),
            e(Errno::ENOPROTOOPT),
            "SO_ATTACH_BPF"
        );
        assert_eq!(set(&mut h, SOL_TCP, 19), e(Errno::EPERM), "TCP_REPAIR");
        // Reading back shows it never took.
        let len = BASE + 0x10;
        h.mem.write_u64(len, 4).unwrap();
        h.call(Sysno::Getsockopt, [s, SOL_SOCKET, 36, BASE, len, 0]);
        assert_eq!(h.mem.read_u32(BASE).unwrap(), 0);
    }

    #[test]
    fn so_priority_above_six_needs_privilege() {
        use crate::abi::arch::Sysno;
        let mut h = Harness::new();
        let s = h.call(Sysno::Socket, [2, SOCK_STREAM, 0, 0, 0, 0]) as u64;
        h.mem.write(BASE, &6u32.to_le_bytes()).unwrap();
        let args = [s, SOL_SOCKET, 12, BASE, 4, 0];
        assert_eq!(h.call(Sysno::Setsockopt, args), 0);
        h.mem.write(BASE, &7u32.to_le_bytes()).unwrap();
        assert_eq!(h.call(Sysno::Setsockopt, args), e(Errno::EPERM));
    }

    #[test]
    fn a_port_is_shared_only_when_both_sockets_opted_in() {
        let mut h = Harness::new();
        let port = 9700;
        assert_eq!(bound_socket(&mut h, SOCK_STREAM, &[], port).1, 0);
        for opts in [&[][..], &[SO_REUSEADDR], &[SO_REUSEPORT]] {
            assert_eq!(
                bound_socket(&mut h, SOCK_STREAM, opts, port).1,
                e(Errno::EADDRINUSE),
                "the first socket didn't opt in ({opts:?})"
            );
        }
    }

    #[test]
    fn reuseaddr_shares_a_port_only_until_someone_listens() {
        use crate::abi::arch::Sysno;
        let mut h = Harness::new();
        let (port, opts) = (9701, &[SO_REUSEADDR][..]);
        let (a, r) = bound_socket(&mut h, SOCK_STREAM, opts, port);
        assert_eq!(r, 0);
        let (b, r) = bound_socket(&mut h, SOCK_STREAM, opts, port);
        assert_eq!(r, 0);
        assert_eq!(h.call(Sysno::Listen, [a, 8, 0, 0, 0, 0]), 0);
        assert_eq!(
            h.call(Sysno::Listen, [b, 8, 0, 0, 0, 0]),
            e(Errno::EADDRINUSE)
        );
        assert_eq!(
            bound_socket(&mut h, SOCK_STREAM, opts, port).1,
            e(Errno::EADDRINUSE),
            "no binding over a listener"
        );
    }

    #[test]
    fn reuseport_listeners_take_turns_accepting() {
        use crate::abi::arch::Sysno;
        let mut h = Harness::new();
        let (port, opts) = (9702, &[SO_REUSEPORT][..]);
        let (a, r) = bound_socket(&mut h, SOCK_STREAM, opts, port);
        assert_eq!(r, 0);
        let (b, r) = bound_socket(&mut h, SOCK_STREAM, opts, port);
        assert_eq!(r, 0);
        assert_eq!(h.call(Sysno::Listen, [a, 8, 0, 0, 0, 0]), 0);
        assert_eq!(h.call(Sysno::Listen, [b, 8, 0, 0, 0, 0]), 0);
        let addr = BASE + 0x100;
        for _ in 0..4 {
            let c = h.call(Sysno::Socket, [2, SOCK_STREAM, 0, 0, 0, 0]) as u64;
            assert_eq!(h.call(Sysno::Connect, [c, addr, 16, 0, 0, 0]), 0);
        }
        // Accept must find two connections waiting on each listener.
        for s in [a, b] {
            for _ in 0..2 {
                assert!(h.call(Sysno::Accept4, [s, 0, 0, 0, 0, 0]) >= 0);
                assert!(!h.cx.block);
            }
            h.call(Sysno::Accept4, [s, 0, 0, 0, 0, 0]);
            assert!(h.cx.block, "and no more than two");
            h.cx.block = false;
        }
    }

    #[test]
    fn closing_a_udp_socket_frees_its_port_in_the_routing_table() {
        use crate::abi::arch::Sysno;
        let mut h = Harness::new();
        let port = 9703;
        let (a, r) = bound_socket(&mut h, SOCK_DGRAM, &[], port);
        assert_eq!(r, 0);
        assert_eq!(h.call(Sysno::Close, [a, 0, 0, 0, 0, 0]), 0);
        assert_eq!(bound_socket(&mut h, SOCK_DGRAM, &[], port).1, 0);
        assert_eq!(h.k.net.lock().unwrap().dgram_ports.len(), 1);
    }

    #[test]
    fn recv_with_so_rcvtimeo_blocks_until_the_deadline_then_fails() {
        use crate::abi::arch::Sysno;
        let mut h = Harness::new();
        let (s, r) = bound_socket(&mut h, SOCK_DGRAM, &[], 9704);
        assert_eq!(r, 0);
        let tv = BASE + 0x200;
        h.mem.write_u64(tv, 5).unwrap(); // 5 s
        h.mem.write_u64(tv + 8, 0).unwrap();
        let set = [s, SOL_SOCKET, 20, tv, 16, 0];
        assert_eq!(h.call(Sysno::Setsockopt, set), 0);

        let recv = [s, BASE, 1, 0, 0, 0];
        assert_eq!(h.call(Sysno::Recvfrom, recv), 0);
        assert!(h.cx.block, "parks for now");
        assert!(!h.cx.restartable, "a timed socket call isn't restarted");
        let deadline = h.cx.cur.wake_deadline.expect("armed");

        // Re-run before the deadline: same deadline, still parked.
        h.cx.block = false;
        assert_eq!(h.call(Sysno::Recvfrom, recv), 0);
        assert!(h.cx.block);
        assert_eq!(h.cx.cur.wake_deadline, Some(deadline));

        // Re-run after it.
        h.cx.block = false;
        h.cx.cur.wake_deadline = Some(0);
        assert_eq!(h.call(Sysno::Recvfrom, recv), e(Errno::EAGAIN));
        assert!(!h.cx.block);
        assert_eq!(h.cx.cur.wake_deadline, None);
    }

    #[test]
    fn a_timeout_with_bad_microseconds_is_edom() {
        use crate::abi::arch::Sysno;
        let mut h = Harness::new();
        let s = h.call(Sysno::Socket, [2, SOCK_DGRAM, 0, 0, 0, 0]) as u64;
        h.mem.write_u64(BASE, 0).unwrap();
        h.mem.write_u64(BASE + 8, 1_000_000).unwrap();
        assert_eq!(
            h.call(Sysno::Setsockopt, [s, SOL_SOCKET, 20, BASE, 16, 0]),
            e(Errno::EDOM)
        );
    }

    #[test]
    fn socket_buffers_are_clamped_like_linux() {
        assert_eq!(clamp_sock_buf(1), 2304);
        assert_eq!(clamp_sock_buf(65_536), 65_536);
        assert_eq!(clamp_sock_buf(u32::MAX), 212_992);
    }

    #[test]
    fn options_are_stored_defaulted_truncated_or_refused() {
        use crate::abi::arch::Sysno;
        let (k, mut mem, mut v, mut cx) = setup();
        // A TCP socket (AF_INET, SOCK_STREAM).
        let s = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Socket,
            [2, 1, 0, 0, 0, 0],
        ) as u64;
        let (val, len) = (BASE, BASE + 0x10);
        // Unset TCP_KEEPCNT reads Linux's default 9.
        mem.write_u64(len, 4).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Getsockopt,
                [s, SOL_TCP, 6, val, len, 0]
            ),
            0
        );
        assert_eq!(mem.read_u32(val).unwrap(), 9);
        // Set TCP_KEEPIDLE, read it back.
        mem.write(val, &30u32.to_le_bytes()).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Setsockopt,
                [s, SOL_TCP, 4, val, 4, 0]
            ),
            0
        );
        mem.write(val, &0u32.to_le_bytes()).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Getsockopt,
                [s, SOL_TCP, 4, val, len, 0]
            ),
            0
        );
        assert_eq!(mem.read_u32(val).unwrap(), 30);
        // A short int option is EINVAL; an unknown one ENOPROTOOPT; a
        // read-only one can't be set.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Setsockopt,
                [s, SOL_TCP, 4, val, 2, 0]
            ),
            e(Errno::EINVAL)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Setsockopt,
                [s, SOL_TCP, 999, val, 4, 0]
            ),
            e(Errno::ENOPROTOOPT)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Setsockopt,
                [s, SOL_SOCKET, 3, val, 4, 0]
            ),
            e(Errno::ENOPROTOOPT)
        );
        // TCP_CONGESTION into a 4-byte buffer is truncated, not overrun.
        mem.write(val, &[0xffu8; 16]).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Getsockopt,
                [s, SOL_TCP, 13, val, len, 0]
            ),
            0
        );
        assert_eq!(mem.read_vec(val, 6).unwrap(), b"cubi\xff\xff");
        assert_eq!(mem.read_u32(len).unwrap(), 4);
    }
}
