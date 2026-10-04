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
    fn options_are_stored_defaulted_truncated_or_refused() {
        use super::super::testutil::{BASE, call, e, setup};
        use crate::abi::arch::Sysno;
        use crate::abi::errno::Errno;
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
