//! Tunnel throughput diagnostic: where does the time go on a bulk download
//! over grouterd's WebSocket tunnel? Takes the guest (and its interpreted TLS)
//! out of the picture: Rust drives pktkit's TCP directly through a
//! [`Tunnel`], and every TCP segment is traced (time, direction, seq/ack,
//! window, length) so stalls can be attributed — receive-window limited,
//! ACK-clocked, lossy, or the path simply not delivering.
//!
//! Gated on `NIXVM_TUNNEL_BRIDGE` (see `scripts/tunnel-bridge.mjs`); consumes
//! one tunnel token. Run with `--nocapture` to see the report:
//!
//! ```text
//! node scripts/tunnel-bridge.mjs 7777 &
//! NIXVM_TUNNEL_BRIDGE=127.0.0.1:7777 cargo test --release --features tunnel \
//!     --test tunnel_throughput -- --nocapture
//! ```
#![cfg(feature = "tunnel")]
// Rates and percentages from byte/packet counts; a report, not arithmetic.
#![allow(clippy::cast_precision_loss)]

use std::io::{Read, Write};
use std::net::{IpAddr, TcpStream, ToSocketAddrs};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use nixvm::tunnel::{Lease, Tunnel};

/// Default download: Alpine's community index from its CDN (Fastly).
/// `NIXVM_TUNNEL_TARGETS=host/path,host/path…` compares several servers.
const DEFAULT_TARGET: &str =
    "dl-cdn.alpinelinux.org/alpine/v3.20/community/aarch64/APKINDEX.tar.gz";

fn read_frame(s: &mut TcpStream) -> Option<(u8, Vec<u8>)> {
    let mut hdr = [0u8; 5];
    s.read_exact(&mut hdr).ok()?;
    let mut b = vec![0u8; u32::from_be_bytes(hdr[1..5].try_into().unwrap()) as usize];
    s.read_exact(&mut b).ok()?;
    Some((hdr[0], b))
}

fn field<'a>(json: &'a str, key: &str) -> Option<&'a str> {
    let at = json.find(&format!("\"{key}\":\""))? + key.len() + 4;
    Some(&json[at..at + json[at..].find('"')?])
}

/// One traced TCP segment (IPv4 only — the download uses the v4 lease).
#[derive(Clone, Copy)]
struct Seg {
    t: f64, // ms since start
    out: bool,
    seq: u32,
    ack: u32,
    win: u16,
    len: usize,
    flags: u8,
    /// The TCP checksum does not match the segment (inbound only).
    bad_csum: bool,
}

fn parse_tcp(p: &[u8], t: f64, out: bool) -> Option<Seg> {
    if p.first()? >> 4 != 4 || p[9] != 6 {
        return None;
    }
    let ihl = usize::from(p[0] & 0xf) * 4;
    let total = usize::from(u16::from_be_bytes([p[2], p[3]]));
    let tcp = p.get(ihl..total)?;
    let doff = usize::from(tcp[12] >> 4) * 4;
    let bad_csum = !out && {
        let src = IpAddr::from(<[u8; 4]>::try_from(&p[12..16]).unwrap());
        let dst = IpAddr::from(<[u8; 4]>::try_from(&p[16..20]).unwrap());
        let mut z = tcp.to_vec();
        z[16..18].fill(0);
        pktkit::transport_checksum(pktkit::Protocol::TCP, src, dst, &z)
            != u16::from_be_bytes([tcp[16], tcp[17]])
    };
    Some(Seg {
        bad_csum,
        t,
        out,
        seq: u32::from_be_bytes(tcp[4..8].try_into().unwrap()),
        ack: u32::from_be_bytes(tcp[8..12].try_into().unwrap()),
        win: u16::from_be_bytes([tcp[14], tcp[15]]),
        len: tcp.len().saturating_sub(doff),
        flags: tcp[13],
    })
}

/// Download `PATH` over the tunnel with pktkit's TCP; returns (bytes, secs, trace).
fn tunnel_download(
    net: &Tunnel,
    link: &mut TcpStream,
    rx: &mpsc::Receiver<Vec<u8>>,
    (host, path, ip): (&str, &str, IpAddr),
    tick: Duration,
) -> (usize, f64, Vec<Seg>) {
    let IpAddr::V4(v4) = ip else {
        panic!("v4 only")
    };
    let mut ipb = [0u8; 16];
    ipb[..4].copy_from_slice(&v4.octets());
    let egress = net.egress();
    let mut conn = egress.connect_tcp(ipb, false, 80).expect("connect");
    let start = Instant::now();
    let ms = |s: Instant| s.elapsed().as_secs_f64() * 1000.0;
    let mut trace = Vec::new();
    let mut sent_req = false;
    let mut got = 0usize;
    let mut last_tick = Instant::now();
    let req = format!("GET {path} HTTP/1.0\r\nHost: {host}\r\nConnection: close\r\n\r\n");
    loop {
        if !sent_req && conn.poll_connect().unwrap() {
            assert_eq!(conn.send(req.as_bytes()).unwrap(), req.len());
            sent_req = true;
        }
        let mut buf = vec![0u8; 65536];
        let mut eof = false;
        while conn.poll_readable() {
            let n = conn.recv(&mut buf).unwrap();
            if n == 0 {
                eof = true;
                break;
            }
            got += n;
        }
        for p in net.take_outbound() {
            if let Some(s) = parse_tcp(&p, ms(start), true) {
                trace.push(s);
            }
            let mut f = vec![1u8];
            f.extend_from_slice(&(p.len() as u32).to_be_bytes());
            f.extend_from_slice(&p);
            link.write_all(&f).unwrap();
        }
        if eof {
            break;
        }
        match rx.recv_timeout(Duration::from_millis(1)) {
            Ok(p) => {
                if let Some(s) = parse_tcp(&p, ms(start), false) {
                    trace.push(s);
                }
                net.inject(&p);
                while let Ok(p) = rx.try_recv() {
                    if let Some(s) = parse_tcp(&p, ms(start), false) {
                        trace.push(s);
                    }
                    net.inject(&p);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(e) => panic!("bridge gone: {e}"),
        }
        if last_tick.elapsed() >= tick {
            net.tick();
            last_tick = Instant::now();
        }
        if start.elapsed() > Duration::from_secs(60) {
            println!("  !! download stalled at {got} bytes after 60 s");
            break;
        }
    }
    (got, start.elapsed().as_secs_f64(), trace)
}

fn report(name: &str, bytes: usize, secs: f64, trace: &[Seg]) {
    // Full trace for offline analysis: NIXVM_TRACE_OUT=<dir>.
    if let Ok(dir) = std::env::var("NIXVM_TRACE_OUT")
        && !trace.is_empty()
    {
        let mut f = std::fs::File::create(format!(
            "{dir}/{}.csv",
            name.replace(|c: char| !c.is_alphanumeric(), "_")
        ))
        .unwrap();
        writeln!(f, "t_ms,dir,seq,ack,win,len,flags").unwrap();
        for s in trace {
            writeln!(
                f,
                "{:.3},{},{},{},{},{},{:#04x}",
                s.t,
                if s.out { "out" } else { "in" },
                s.seq,
                s.ack,
                s.win,
                s.len,
                s.flags
            )
            .unwrap();
        }
    }
    println!(
        "\n=== {name}: {bytes} bytes in {secs:.2}s = {:.0} KB/s",
        bytes as f64 / secs / 1024.0
    );
    if trace.is_empty() {
        return;
    }
    let inb: Vec<&Seg> = trace.iter().filter(|s| !s.out && s.len > 0).collect();
    let outs: Vec<&Seg> = trace.iter().filter(|s| s.out).collect();
    // Retransmissions seen inbound: data starting below the highest seq seen.
    let mut max_end = 0u32;
    let mut first = true;
    let mut retrans = 0;
    for s in &inb {
        let end = s.seq.wrapping_add(s.len as u32);
        if !first && (s.seq.wrapping_sub(max_end) as i32) < 0 {
            retrans += 1;
        }
        if first || (end.wrapping_sub(max_end) as i32) > 0 {
            max_end = end;
        }
        first = false;
    }
    let sizes: Vec<usize> = inb.iter().map(|s| s.len).collect();
    let max_seg = sizes.iter().max().copied().unwrap_or(0);
    let syn = trace.iter().find(|s| s.out && s.flags & 0x02 != 0);
    let synack = trace.iter().find(|s| !s.out && s.flags & 0x12 == 0x12);
    println!(
        "  segments: {} in (data), {} out; max data seg {} B; inbound retransmits {}",
        inb.len(),
        outs.len(),
        max_seg,
        retrans
    );
    if let (Some(a), Some(b)) = (syn, synack) {
        println!("  handshake RTT {:.1} ms (SYN -> SYN-ACK)", b.t - a.t);
    }
    let wins: Vec<u16> = outs.iter().map(|s| s.win).collect();
    println!(
        "  our advertised window (raw, unscaled): min {} max {}",
        wins.iter().min().unwrap_or(&0),
        wins.iter().max().unwrap_or(&0)
    );
    // ACK responsiveness: for each inbound data segment, time until we next
    // send a segment acking past it.
    let mut delays = Vec::new();
    for s in &inb {
        let end = s.seq.wrapping_add(s.len as u32);
        if let Some(a) = outs
            .iter()
            .find(|o| o.t >= s.t && (o.ack.wrapping_sub(end) as i32) >= 0)
        {
            delays.push(a.t - s.t);
        }
    }
    delays.sort_by(f64::total_cmp);
    if !delays.is_empty() {
        let pct = |p: f64| delays[((delays.len() - 1) as f64 * p) as usize];
        println!(
            "  our ACK delay: p50 {:.1} ms, p90 {:.1} ms, max {:.1} ms",
            pct(0.5),
            pct(0.9),
            delays[delays.len() - 1]
        );
    }
    // Inbound gaps: silences between data segments, and how much data we had
    // acked-but-not-yet-received window for (was the sender allowed to send?).
    let mut gaps: Vec<(f64, f64)> = inb.windows(2).map(|w| (w[1].t - w[0].t, w[0].t)).collect();
    gaps.sort_by(|a, b| b.0.total_cmp(&a.0));
    let total_gap: f64 = gaps.iter().filter(|g| g.0 > 20.0).map(|g| g.0).sum();
    println!(
        "  inbound silences > 20 ms: {} totalling {:.0} ms; largest: {}",
        gaps.iter().filter(|g| g.0 > 20.0).count(),
        total_gap,
        gaps.iter()
            .take(5)
            .map(|g| format!("{:.0}ms@{:.0}", g.0, g.1))
            .collect::<Vec<_>>()
            .join(", ")
    );
    // Per-100ms inbound byte histogram (first 3 s) to see the delivery shape.
    let mut buckets = vec![0usize; 30];
    for s in &inb {
        let i = (s.t / 100.0) as usize;
        if i < buckets.len() {
            buckets[i] += s.len;
        }
    }
    println!(
        "  KB per 100 ms (first 3 s): {}",
        buckets
            .iter()
            .map(|b| (b / 1024).to_string())
            .collect::<Vec<_>>()
            .join(" ")
    );
}

/// Proxy-only check, no TCP involved: bursts of 1000-byte ICMP echoes to
/// 1.1.1.1 through the tunnel; how many come back, and when.
fn icmp_bursts(
    net: &Tunnel,
    link: &mut TcpStream,
    rx: &mpsc::Receiver<Vec<u8>>,
    target: [u8; 4],
    payload: usize,
    bursts: &[usize],
) {
    let mut sock = net.egress().open_icmp(false).expect("icmp");
    let mut dst = [0u8; 16];
    dst[..4].copy_from_slice(&target);
    for (round, &n) in (1u16..).zip(bursts) {
        let start = Instant::now();
        for i in 0..n {
            let mut m = vec![8u8, 0, 0, 0, 0x4e, round as u8, (i >> 8) as u8, i as u8];
            m.extend(std::iter::repeat_n(0xa5, payload));
            let sum = pktkit::checksum(&m);
            m[2..4].copy_from_slice(&sum.to_be_bytes());
            sock.send_to(&m, dst, false, 0).unwrap();
        }
        for p in net.take_outbound() {
            let mut f = vec![1u8];
            f.extend_from_slice(&(p.len() as u32).to_be_bytes());
            f.extend_from_slice(&p);
            link.write_all(&f).unwrap();
        }
        let sent_ms = start.elapsed().as_secs_f64() * 1000.0;
        let mut times = Vec::new();
        while start.elapsed() < Duration::from_secs(3) {
            if let Ok(p) = rx.recv_timeout(Duration::from_millis(5)) {
                net.inject(&p);
            }
            while let Ok(Some((_, _, _, pkt))) = sock.recv_from() {
                let hl = usize::from(pkt[0] & 0xf) * 4;
                if pkt.get(hl) == Some(&0) && pkt.get(hl + 5) == Some(&(round as u8)) {
                    times.push(start.elapsed().as_secs_f64() * 1000.0);
                }
            }
        }
        let first = times.first().copied().unwrap_or(f64::NAN);
        let last = times.last().copied().unwrap_or(f64::NAN);
        println!(
            "ICMP to {target:?}: burst of {n:3} x {} B IP (sent in {sent_ms:.1} ms): {} replies ({:.0}% lost), first at {first:.0} ms, last at {last:.0} ms, {:.1} ms/reply",
            payload + 28,
            times.len(),
            100.0 * (1.0 - times.len() as f64 / n as f64),
            (last - first) / (times.len().max(2) - 1) as f64
        );
    }
}

/// Small pings sent 5 ms apart, each its own tunnel message: how long each
/// takes to come back. Small writes trickling onto the carrier are what TCP
/// ACKs look like; a Nagle + delayed-ACK interaction there shows up as most
/// of them waiting tens of milliseconds.
fn spaced_pings(net: &Tunnel, link: &mut TcpStream, rx: &mpsc::Receiver<Vec<u8>>) {
    let mut sock = net.egress().open_icmp(false).expect("icmp");
    let mut dst = [0u8; 16];
    dst[..4].copy_from_slice(&[1, 1, 1, 1]);
    let n = 20u16;
    let start = Instant::now();
    let mut sent_at = vec![0f64; usize::from(n)];
    let mut rtts = vec![f64::NAN; usize::from(n)];
    let mut next = 0u16;
    while start.elapsed() < Duration::from_secs(3) {
        let now_ms = start.elapsed().as_secs_f64() * 1000.0;
        if next < n && now_ms >= f64::from(next) * 5.0 {
            let mut m = vec![8u8, 0, 0, 0, 0x50, 0x50, 0, next as u8];
            m.extend(std::iter::repeat_n(0, 56));
            let sum = pktkit::checksum(&m);
            m[2..4].copy_from_slice(&sum.to_be_bytes());
            sock.send_to(&m, dst, false, 0).unwrap();
            for p in net.take_outbound() {
                let mut f = vec![1u8];
                f.extend_from_slice(&(p.len() as u32).to_be_bytes());
                f.extend_from_slice(&p);
                link.write_all(&f).unwrap();
            }
            sent_at[usize::from(next)] = now_ms;
            next += 1;
        }
        if let Ok(p) = rx.recv_timeout(Duration::from_micros(200)) {
            net.inject(&p);
        }
        while let Ok(Some((_, _, _, pkt))) = sock.recv_from() {
            let hl = usize::from(pkt[0] & 0xf) * 4;
            if pkt.get(hl) == Some(&0) && pkt.get(hl + 4) == Some(&0x50) {
                let i = usize::from(pkt[hl + 7]);
                if i < rtts.len() {
                    rtts[i] = start.elapsed().as_secs_f64() * 1000.0 - sent_at[i];
                }
            }
        }
    }
    println!(
        "spaced pings (5 ms apart) RTT ms: {}",
        rtts.iter()
            .map(|r| format!("{r:.0}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
}

#[test]
fn tunnel_live_throughput() {
    let Ok(bridge) = std::env::var("NIXVM_TUNNEL_BRIDGE") else {
        eprintln!("NIXVM_TUNNEL_BRIDGE not set; skipping");
        return;
    };
    let targets: Vec<(String, String, IpAddr)> = std::env::var("NIXVM_TUNNEL_TARGETS")
        .unwrap_or_else(|_| DEFAULT_TARGET.to_string())
        .split(',')
        .map(|t| {
            let (host, path) = t.split_once('/').unwrap();
            let ip = (host, 80)
                .to_socket_addrs()
                .unwrap()
                .find(std::net::SocketAddr::is_ipv4)
                .unwrap()
                .ip();
            (host.to_string(), format!("/{path}"), ip)
        })
        .collect();

    // Baselines: the same downloads over the host's own network.
    for (host, path, ip) in &targets {
        let t = Instant::now();
        let mut s = TcpStream::connect((*ip, 80)).unwrap();
        write!(
            s,
            "GET {path} HTTP/1.0\r\nHost: {host}\r\nConnection: close\r\n\r\n"
        )
        .unwrap();
        let mut body = Vec::new();
        s.read_to_end(&mut body).unwrap();
        report(
            &format!("host network: {host} ({ip})"),
            body.len(),
            t.elapsed().as_secs_f64(),
            &[],
        );
    }

    let mut link = TcpStream::connect(&bridge).expect("connect to tunnel-bridge");
    let (kind, hello) = read_frame(&mut link).expect("hello");
    assert_eq!(kind, 0);
    let hello = String::from_utf8(hello).unwrap();
    let v4 = field(&hello, "ipv4").map(|a| (a.parse().unwrap(), 10));
    let net = Tunnel::new();
    net.up(Lease {
        v4,
        v6: None,
        mtu: 1400,
    });
    let (tx, rx) = mpsc::channel();
    let mut reader = link.try_clone().unwrap();
    std::thread::spawn(move || {
        while let Some((1, p)) = read_frame(&mut reader) {
            if tx.send(p).is_err() {
                break;
            }
        }
    });
    let _ = link.set_nodelay(true);

    if std::env::var_os("NIXVM_SKIP_BURSTS").is_none() {
        spaced_pings(&net, &mut link, &rx);
        // Bursts at growing packet sizes (1400 B IP = the tunnel MTU, what
        // a full TCP segment is), to the direct path (Cloudflare) and to each
        // download server.
        for payload in [500, 1000, 1372] {
            icmp_bursts(&net, &mut link, &rx, [1, 1, 1, 1], payload, &[30]);
            for (_, _, ip) in &targets {
                let IpAddr::V4(v4) = ip else { continue };
                icmp_bursts(&net, &mut link, &rx, v4.octets(), payload, &[30]);
            }
        }
    }
    if std::env::var_os("NIXVM_SKIP_DOWNLOAD").is_some() {
        return;
    }
    // NIXVM_TUNNEL_MTUS=1400,700,440: repeat each download with the link MTU
    // (so our SYN's MSS) at each size. A server's back-to-back segments that
    // something upstream merges into one oversized packet (GRO) survive a
    // 1500-byte limit only when the merged size fits — and then show up here
    // as segments larger than the MSS we asked for.
    let mtus: Vec<u16> = std::env::var("NIXVM_TUNNEL_MTUS")
        .unwrap_or_else(|_| "1400".into())
        .split(',')
        .map(|m| m.parse().unwrap())
        .collect();
    for &mtu in &mtus {
        net.up(Lease { v4, v6: None, mtu });
        for (host, path, ip) in &targets {
            let tick = Duration::from_millis(10);
            let (n, secs, trace) = tunnel_download(&net, &mut link, &rx, (host, path, *ip), tick);
            let oversized = trace
                .iter()
                .filter(|s| !s.out && s.len > usize::from(mtu - 40))
                .count();
            report(&format!("tunnel mtu {mtu}: {host} ({ip})"), n, secs, &trace);
            let bad = trace.iter().filter(|s| s.bad_csum).count();
            let bad_big = trace
                .iter()
                .filter(|s| s.bad_csum && s.len > usize::from(mtu - 40))
                .count();
            println!(
                "  segments larger than our MSS ({}): {oversized}; bad TCP checksum: {bad} ({bad_big} of them oversized)",
                mtu - 40
            );
        }
    }
}
