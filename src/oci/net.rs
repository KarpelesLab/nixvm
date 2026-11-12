//! Pulling an image over a guest-style network link: any [`Egress`] — the
//! browser's packet tunnel ([`crate::tunnel::Tunnel`]) or the host's own
//! sockets — with nixvm doing the DNS, the TLS (purecrypto's) and the HTTP
//! itself. In a browser this is what gets around CORS: `fetch()` is never
//! used, the registry is reached over the tunnel's TCP like any guest
//! connection.
//!
//! [`NetPull::poll`] never blocks: it moves the transfer along as far as the
//! link allows and returns, so a single-threaded embedder calls it between
//! pumping the tunnel's packets. Each request is one connection
//! (`Connection: close`); redirects are followed (registries send blob
//! downloads to a CDN), dropping the registry's `Authorization` when the host
//! changes.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io;

use purecrypto::tls::{Config, Connection, RootCertStore, Step};
use purecrypto::x509::Time;

use super::{Image, OciError, Pull, Request, Response};
use crate::kernel::egress::{Egress, HostConn, HostDgram};

/// Public resolvers asked for names: Cloudflare's, then Google's.
const NAMESERVERS_V4: [[u8; 4]; 2] = [[1, 1, 1, 1], [8, 8, 8, 8]];
const NAMESERVERS_V6: [[u8; 16]; 2] = [
    [
        0x26, 0x06, 0x47, 0x00, 0x47, 0x00, 0, 0, 0, 0, 0, 0, 0, 0, 0x11, 0x11,
    ],
    [
        0x20, 0x01, 0x48, 0x60, 0x48, 0x60, 0, 0, 0, 0, 0, 0, 0, 0, 0x88, 0x88,
    ],
];
/// A DNS query unanswered this long is asked again (of the next server).
const DNS_RETRY_MS: u64 = 1500;
const DNS_TRIES: u32 = 6;
/// A transfer that moves no byte for this long fails.
const STALL_MS: u64 = 60_000;
const MAX_REDIRECTS: u32 = 8;

/// TLS's randomness: purecrypto's `OsRng`, the same as the guest's
/// ([`crate::entropy`]) — the OS, or the page's `crypto.getRandomValues`.
/// (purecrypto's own `os_rng()` shortcut needs its `std`, which the wasm build
/// leaves out.)
struct Entropy;

impl purecrypto::tls::EntropySource for Entropy {
    fn fill(&self, dest: &mut [u8]) {
        crate::entropy::fill(dest);
    }
}

fn now_ms() -> u64 {
    crate::clock::now_monotonic().as_millis() as u64
}

fn fetch_err(what: impl Into<String>) -> OciError {
    OciError::Fetch(what.into())
}

/// `scheme://host[:port]/path` split up.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Url {
    tls: bool,
    host: String,
    port: u16,
    /// Path and query, starting with `/`.
    path: String,
}

impl Url {
    fn parse(url: &str) -> Result<Self, OciError> {
        let (tls, rest) = if let Some(r) = url.strip_prefix("https://") {
            (true, r)
        } else if let Some(r) = url.strip_prefix("http://") {
            (false, r)
        } else {
            return Err(fetch_err(format!("not an http(s) URL: {url}")));
        };
        let (authority, path) = match rest.find(['/', '?']) {
            Some(i) if rest.as_bytes()[i] == b'/' => (&rest[..i], rest[i..].to_string()),
            Some(i) => (&rest[..i], format!("/{}", &rest[i..])),
            None => (rest, "/".to_string()),
        };
        let authority = authority.rsplit('@').next().unwrap_or(authority);
        let default = if tls { 443 } else { 80 };
        let (host, port) = if let Some(v6) = authority.strip_prefix('[') {
            let (h, after) = v6
                .split_once(']')
                .ok_or_else(|| fetch_err(format!("bad URL {url}")))?;
            let port = after.strip_prefix(':').map_or(Ok(default), str::parse);
            (
                h.to_string(),
                port.map_err(|_| fetch_err(format!("bad port in {url}")))?,
            )
        } else {
            match authority.rsplit_once(':') {
                Some((h, p)) => (
                    h.to_string(),
                    p.parse()
                        .map_err(|_| fetch_err(format!("bad port in {url}")))?,
                ),
                None => (authority.to_string(), default),
            }
        };
        if host.is_empty() {
            return Err(fetch_err(format!("no host in {url}")));
        }
        Ok(Self {
            tls,
            host: host.to_ascii_lowercase(),
            port,
            path,
        })
    }

    /// `location` (absolute, scheme-relative or a path) seen from `self`.
    fn join(&self, location: &str) -> Result<Self, OciError> {
        if location.starts_with("http://") || location.starts_with("https://") {
            return Self::parse(location);
        }
        let scheme = if self.tls { "https" } else { "http" };
        if let Some(rest) = location.strip_prefix("//") {
            return Self::parse(&format!("{scheme}://{rest}"));
        }
        let path = if location.starts_with('/') {
            location.to_string()
        } else {
            let dir = self.path.split('?').next().unwrap_or("/");
            let dir = &dir[..=dir.rfind('/').unwrap_or(0)];
            format!("{dir}{location}")
        };
        Ok(Self {
            path,
            ..self.clone()
        })
    }

    /// The host as an address, when it is one.
    fn literal(&self) -> Option<([u8; 16], bool)> {
        if let Ok(v4) = self.host.parse::<std::net::Ipv4Addr>() {
            return Some((to16(v4.octets()), false));
        }
        self.host
            .parse::<std::net::Ipv6Addr>()
            .ok()
            .map(|v6| (v6.octets(), true))
    }
}

fn to16(v4: [u8; 4]) -> [u8; 16] {
    let mut a = [0u8; 16];
    a[..4].copy_from_slice(&v4);
    a
}

/// A DNS query for `name`: type A (1) or AAAA (28), recursion desired.
fn dns_query(id: u16, name: &str, aaaa: bool) -> Vec<u8> {
    let mut q = Vec::with_capacity(name.len() + 18);
    q.extend_from_slice(&id.to_be_bytes());
    q.extend_from_slice(&[0x01, 0x00, 0, 1, 0, 0, 0, 0, 0, 0]);
    for label in name.trim_end_matches('.').split('.') {
        q.push(label.len().min(63) as u8);
        q.extend_from_slice(&label.as_bytes()[..label.len().min(63)]);
    }
    q.push(0);
    q.extend_from_slice(&(if aaaa { 28u16 } else { 1 }).to_be_bytes());
    q.extend_from_slice(&[0, 1]);
    q
}

/// The first address of the asked type in a DNS answer to query `id`;
/// `Err` when the server says the name doesn't exist.
fn dns_answer(msg: &[u8], id: u16, aaaa: bool) -> Result<Option<[u8; 16]>, OciError> {
    if msg.len() < 12 || u16::from_be_bytes([msg[0], msg[1]]) != id || msg[2] & 0x80 == 0 {
        return Ok(None);
    }
    let rcode = msg[3] & 0x0f;
    if rcode == 3 {
        return Err(fetch_err("no such host (NXDOMAIN)"));
    }
    let qd = u16::from_be_bytes([msg[4], msg[5]]);
    let an = u16::from_be_bytes([msg[6], msg[7]]);
    let mut i = 12;
    let skip_name = |i: &mut usize| -> Option<()> {
        loop {
            let len = *msg.get(*i)?;
            if len & 0xc0 == 0xc0 {
                *i += 2;
                return Some(());
            }
            *i += 1 + len as usize;
            if len == 0 {
                return Some(());
            }
        }
    };
    for _ in 0..qd {
        if skip_name(&mut i).is_none() {
            return Ok(None);
        }
        i += 4;
    }
    for _ in 0..an {
        if skip_name(&mut i).is_none() || i + 10 > msg.len() {
            return Ok(None);
        }
        let ty = u16::from_be_bytes([msg[i], msg[i + 1]]);
        let len = u16::from_be_bytes([msg[i + 8], msg[i + 9]]) as usize;
        i += 10;
        let Some(data) = msg.get(i..i + len) else {
            return Ok(None);
        };
        match (ty, aaaa) {
            (1, false) if len == 4 => return Ok(Some(to16([data[0], data[1], data[2], data[3]]))),
            (28, true) if len == 16 => {
                let mut a = [0u8; 16];
                a.copy_from_slice(data);
                return Ok(Some(a));
            }
            _ => {}
        }
        i += len;
    }
    if rcode != 0 {
        return Err(fetch_err(format!("DNS error (rcode {rcode})")));
    }
    Err(fetch_err(
        "the name has no address of the kind this link can reach",
    ))
}

/// An HTTP/1.1 response being read.
#[derive(Debug, Default)]
struct HttpResponse {
    buf: Vec<u8>,
    head: Option<(u16, Vec<(String, String)>)>,
    /// Where the body starts in `buf`.
    body_at: usize,
    body: Vec<u8>,
    framing: Framing,
    done: bool,
}

#[derive(Debug, Default, Clone, Copy)]
enum Framing {
    #[default]
    ToEof,
    Length(usize),
    Chunked,
}

impl HttpResponse {
    fn header(&self, name: &str) -> Option<&str> {
        self.head
            .as_ref()?
            .1
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    fn push(&mut self, data: &[u8]) -> Result<(), OciError> {
        self.buf.extend_from_slice(data);
        if self.head.is_none() {
            let Some(end) = self.buf.windows(4).position(|w| w == b"\r\n\r\n") else {
                if self.buf.len() > 64 << 10 {
                    return Err(fetch_err("an HTTP response head over 64 KiB"));
                }
                return Ok(());
            };
            let text = String::from_utf8_lossy(&self.buf[..end]).into_owned();
            let mut lines = text.split("\r\n");
            let status_line = lines.next().unwrap_or_default();
            let status = status_line
                .split(' ')
                .nth(1)
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| fetch_err(format!("not an HTTP response: {status_line:?}")))?;
            let headers: Vec<(String, String)> = lines
                .filter_map(|l| l.split_once(':'))
                .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
                .collect();
            self.head = Some((status, headers));
            self.body_at = end + 4;
            self.framing = if self
                .header("transfer-encoding")
                .is_some_and(|t| t.to_ascii_lowercase().contains("chunked"))
            {
                Framing::Chunked
            } else if let Some(n) = self.header("content-length").and_then(|n| n.parse().ok()) {
                Framing::Length(n)
            } else if status == 204 || status == 304 {
                Framing::Length(0)
            } else {
                Framing::ToEof
            };
        }
        self.decode_body()
    }

    fn decode_body(&mut self) -> Result<(), OciError> {
        match self.framing {
            Framing::ToEof => {
                self.body.extend_from_slice(&self.buf[self.body_at..]);
                self.body_at = self.buf.len();
            }
            Framing::Length(n) => {
                let take = (n - self.body.len()).min(self.buf.len() - self.body_at);
                self.body
                    .extend_from_slice(&self.buf[self.body_at..self.body_at + take]);
                self.body_at += take;
                self.done = self.body.len() == n;
            }
            Framing::Chunked => loop {
                let rest = &self.buf[self.body_at..];
                let Some(eol) = rest.windows(2).position(|w| w == b"\r\n") else {
                    break;
                };
                let size_text = String::from_utf8_lossy(&rest[..eol]);
                let size =
                    usize::from_str_radix(size_text.split(';').next().unwrap_or("").trim(), 16)
                        .map_err(|_| fetch_err("a bad chunk size"))?;
                if size == 0 {
                    // The last chunk; trailers, if any, end with an empty line.
                    if rest[eol + 2..].windows(2).any(|w| w == b"\r\n") {
                        self.done = true;
                    }
                    break;
                }
                if rest.len() < eol + 2 + size + 2 {
                    break;
                }
                self.body.extend_from_slice(&rest[eol + 2..eol + 2 + size]);
                self.body_at += eol + 2 + size + 2;
            },
        }
        // What's been decoded can go.
        if self.body_at > 1 << 20 {
            self.buf.drain(..self.body_at);
            self.body_at = 0;
        }
        Ok(())
    }

    fn eof(&mut self) -> Result<(), OciError> {
        match (self.head.is_some(), self.framing, self.done) {
            (true, Framing::ToEof, _) => {
                self.done = true;
                Ok(())
            }
            (_, _, true) => Ok(()),
            _ => Err(fetch_err("the connection closed mid-response")),
        }
    }
}

/// Where one request is.
enum Stage {
    Resolve {
        sock: Box<dyn HostDgram>,
        id: u16,
        aaaa: bool,
        sent_at: u64,
        tries: u32,
    },
    Connect(Box<dyn HostConn>),
    Stream {
        conn: Box<dyn HostConn>,
        tls: Option<Box<Connection>>,
        /// Bytes for the wire, not yet taken by the connection.
        out: Vec<u8>,
        /// Wire bytes TLS hasn't consumed yet.
        inbound: Vec<u8>,
        sent: bool,
        resp: HttpResponse,
    },
}

/// One request through to its response, redirects included.
struct Fetch {
    req: Request,
    url: Url,
    redirects: u32,
    stage: Option<Stage>,
    addr: Option<([u8; 16], bool)>,
    active_at: u64,
}

/// A pull driven over an [`Egress`]. See the module docs.
pub struct NetPull {
    pull: Pull,
    egress: Box<dyn Egress>,
    fetch: Option<Fetch>,
    resolved: BTreeMap<String, ([u8; 16], bool)>,
    next_id: u16,
}

impl core::fmt::Debug for NetPull {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("NetPull")
            .field("pull", &self.pull)
            .finish_non_exhaustive()
    }
}

impl NetPull {
    /// Drive `pull` over `egress`.
    #[must_use]
    pub fn new(pull: Pull, egress: Box<dyn Egress>) -> Self {
        let seed = (crate::clock::now_unix().subsec_nanos() & 0xffff) as u16;
        Self {
            pull,
            egress,
            fetch: None,
            resolved: BTreeMap::new(),
            next_id: seed | 1,
        }
    }

    /// The pull being driven (for its progress).
    #[must_use]
    pub fn pull(&self) -> &Pull {
        &self.pull
    }

    /// Move the pull along as far as the link allows right now, without
    /// waiting. `Ok(true)` once everything is fetched.
    ///
    /// # Errors
    /// The network's, TLS's or the registry's.
    pub fn poll(&mut self) -> Result<bool, OciError> {
        loop {
            if self.fetch.is_none() {
                let Some(req) = self.pull.next_request() else {
                    return Ok(true);
                };
                let url = Url::parse(&req.url)?;
                self.fetch = Some(Fetch {
                    req,
                    url,
                    redirects: 0,
                    stage: None,
                    addr: None,
                    active_at: now_ms(),
                });
            }
            match self.step()? {
                Some(resp) => {
                    self.fetch = None;
                    self.pull.feed(resp)?;
                }
                None => return Ok(false),
            }
        }
    }

    /// The image, once [`NetPull::poll`] said it's all here.
    ///
    /// # Errors
    /// As [`Pull::finish`].
    pub fn finish(self) -> Result<Image, OciError> {
        self.pull.finish()
    }

    /// Advance the current request; its response once it has one.
    fn step(&mut self) -> Result<Option<Response>, OciError> {
        let link = self.egress.link();
        let f = self.fetch.as_mut().expect("a request in flight");
        loop {
            let now = now_ms();
            if now.saturating_sub(f.active_at) > STALL_MS {
                return Err(fetch_err(format!("GET {}: timed out", f.req.url)));
            }
            match f.stage.take() {
                None => {
                    if let Some(a) = f
                        .url
                        .literal()
                        .or_else(|| self.resolved.get(&f.url.host).copied())
                    {
                        f.addr = Some(a);
                        let conn = self
                            .egress
                            .connect_tcp(a.0, a.1, f.url.port)
                            .map_err(|e| fetch_err(format!("connect to {}: {e}", f.url.host)))?;
                        f.stage = Some(Stage::Connect(conn));
                        continue;
                    }
                    // Ask for the address family the link has: AAAA on an
                    // IPv6-only one.
                    let aaaa = link.is_some_and(|l| l.v4.is_none() && l.v6.is_some());
                    let sock = self
                        .egress
                        .open_udp()
                        .map_err(|e| fetch_err(format!("DNS socket: {e}")))?;
                    self.next_id = self.next_id.wrapping_add(0x9e37);
                    f.stage = Some(Stage::Resolve {
                        sock,
                        id: self.next_id,
                        aaaa,
                        sent_at: 0,
                        tries: 0,
                    });
                }
                Some(Stage::Resolve {
                    mut sock,
                    id,
                    aaaa,
                    mut sent_at,
                    mut tries,
                }) => {
                    while let Some((_, _, _, msg)) = sock
                        .recv_from()
                        .map_err(|e| fetch_err(format!("DNS: {e}")))?
                    {
                        if let Some(ip) = dns_answer(&msg, id, aaaa)
                            .map_err(|e| fetch_err(format!("{}: {e}", f.url.host)))?
                        {
                            self.resolved.insert(f.url.host.clone(), (ip, aaaa));
                            f.active_at = now;
                            f.stage = None;
                            break;
                        }
                    }
                    if f.stage.is_none() && self.resolved.contains_key(&f.url.host) {
                        continue;
                    }
                    if sent_at == 0 || now.saturating_sub(sent_at) > DNS_RETRY_MS {
                        if tries >= DNS_TRIES {
                            return Err(fetch_err(format!("{}: no answer from DNS", f.url.host)));
                        }
                        let q = dns_query(id, &f.url.host, aaaa);
                        let i = tries as usize % 2;
                        let r = if aaaa {
                            sock.send_to(&q, NAMESERVERS_V6[i], true, 53)
                        } else {
                            sock.send_to(&q, to16(NAMESERVERS_V4[i]), false, 53)
                        };
                        r.map_err(|e| fetch_err(format!("DNS: {e}")))?;
                        sent_at = now.max(1);
                        tries += 1;
                    }
                    f.stage = Some(Stage::Resolve {
                        sock,
                        id,
                        aaaa,
                        sent_at,
                        tries,
                    });
                    return Ok(None);
                }
                Some(Stage::Connect(mut conn)) => {
                    match conn.poll_connect() {
                        Ok(true) => {}
                        Ok(false) => {
                            f.stage = Some(Stage::Connect(conn));
                            return Ok(None);
                        }
                        Err(e) => return Err(fetch_err(format!("connect to {}: {e}", f.url.host))),
                    }
                    f.active_at = now;
                    let tls = if f.url.tls {
                        let time = Time::from_unix(crate::clock::now_unix().as_secs());
                        let cfg = Config::builder()
                            .rng(std::sync::Arc::new(Entropy))
                            .roots(RootCertStore::with_embedded_roots())
                            .server_name(f.url.host.clone())
                            .alpn(vec![b"http/1.1".to_vec()])
                            .verification_time(time)
                            .build();
                        let c = Connection::client(&cfg)
                            .map_err(|e| fetch_err(format!("TLS to {}: {e:?}", f.url.host)))?;
                        Some(Box::new(c))
                    } else {
                        None
                    };
                    f.stage = Some(Stage::Stream {
                        conn,
                        tls,
                        out: Vec::new(),
                        inbound: Vec::new(),
                        sent: false,
                        resp: HttpResponse::default(),
                    });
                }
                Some(Stage::Stream {
                    mut conn,
                    mut tls,
                    mut out,
                    mut inbound,
                    mut sent,
                    mut resp,
                }) => {
                    let host = f.url.host.clone();
                    let tls_err =
                        |e: purecrypto::tls::Error| fetch_err(format!("TLS with {host}: {e:?}"));
                    let mut moved = false;
                    // The request, once the connection can carry it.
                    let ready = tls.as_ref().is_none_or(|t| t.is_handshake_complete());
                    if ready && !sent {
                        let mut head =
                            format!("GET {} HTTP/1.1\r\nHost: {}", f.url.path, f.url.host);
                        let default_port = if f.url.tls { 443 } else { 80 };
                        if f.url.port != default_port {
                            let _ = write!(head, ":{}", f.url.port);
                        }
                        head.push_str("\r\nUser-Agent: nixvm\r\nConnection: close\r\nAccept-Encoding: identity\r\n");
                        for (k, v) in &f.req.headers {
                            let _ = write!(head, "{k}: {v}\r\n");
                        }
                        head.push_str("\r\n");
                        match tls.as_mut() {
                            Some(t) => t.send(head.as_bytes()).map_err(tls_err)?,
                            None => out.extend_from_slice(head.as_bytes()),
                        }
                        sent = true;
                    }
                    if let Some(t) = tls.as_mut() {
                        if !t.is_handshake_complete() {
                            while let Step::WantWrite = t.drive().map_err(tls_err)? {
                                out.extend(t.pop().map_err(tls_err)?);
                            }
                        }
                        loop {
                            let chunk = t.pop().map_err(tls_err)?;
                            if chunk.is_empty() {
                                break;
                            }
                            out.extend(chunk);
                        }
                    }
                    // Out to the wire.
                    while !out.is_empty() {
                        match conn.send(&out) {
                            Ok(0) => break,
                            Ok(n) => {
                                out.drain(..n);
                                moved = true;
                            }
                            Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                            Err(e) => return Err(fetch_err(format!("send to {host}: {e}"))),
                        }
                    }
                    // In from the wire.
                    let mut buf = vec![0u8; 64 << 10];
                    let mut eof = false;
                    loop {
                        match conn.recv(&mut buf) {
                            Ok(0) => {
                                eof = true;
                                break;
                            }
                            Ok(n) => {
                                moved = true;
                                match tls.as_mut() {
                                    Some(_) => inbound.extend_from_slice(&buf[..n]),
                                    None => resp.push(&buf[..n])?,
                                }
                            }
                            Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                            Err(e) => return Err(fetch_err(format!("receive from {host}: {e}"))),
                        }
                    }
                    if let Some(t) = tls.as_mut() {
                        while !inbound.is_empty() {
                            let n = t.feed(&inbound).map_err(tls_err)?;
                            if n == 0 {
                                break;
                            }
                            inbound.drain(..n);
                        }
                        if t.is_handshake_complete() {
                            let mut plain = Vec::new();
                            while t.recv_into(&mut plain).map_err(tls_err)? > 0 {}
                            if !plain.is_empty() {
                                resp.push(&plain)?;
                            }
                            if t.received_close_notify() {
                                eof = true;
                            }
                        }
                    }
                    if eof && !resp.done {
                        resp.eof()?;
                    }
                    if moved {
                        f.active_at = now;
                    }
                    if resp.done {
                        let (status, headers) = resp.head.take().unwrap_or_default();
                        let location = headers
                            .iter()
                            .find(|(k, _)| k.eq_ignore_ascii_case("location"))
                            .map(|(_, v)| v.clone());
                        if let (301 | 302 | 303 | 307 | 308, Some(loc)) = (status, location) {
                            f.redirects += 1;
                            if f.redirects > MAX_REDIRECTS {
                                return Err(fetch_err(format!(
                                    "GET {}: too many redirects",
                                    f.req.url
                                )));
                            }
                            let next = f.url.join(&loc)?;
                            // The registry's token is for the registry only.
                            if next.host != f.url.host {
                                f.req
                                    .headers
                                    .retain(|(k, _)| !k.eq_ignore_ascii_case("authorization"));
                            }
                            f.url = next;
                            f.addr = None;
                            f.active_at = now;
                            continue; // stage None: resolve/connect anew
                        }
                        return Ok(Some(Response {
                            status,
                            headers,
                            body: std::mem::take(&mut resp.body),
                        }));
                    }
                    f.stage = Some(Stage::Stream {
                        conn,
                        tls,
                        out,
                        inbound,
                        sent,
                        resp,
                    });
                    if !moved {
                        return Ok(None);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls_and_redirects_resolve() {
        let u =
            Url::parse("https://registry-1.docker.io/v2/library/alpine/manifests/3.20").unwrap();
        assert_eq!(
            (u.tls, u.host.as_str(), u.port),
            (true, "registry-1.docker.io", 443)
        );
        assert_eq!(u.path, "/v2/library/alpine/manifests/3.20");
        let l = Url::parse("http://localhost:5000").unwrap();
        assert_eq!((l.tls, l.port, l.path.as_str()), (false, 5000, "/"));
        let v6 = Url::parse("https://[2001:db8::1]:8443/x?y").unwrap();
        assert_eq!(
            (v6.host.as_str(), v6.port, v6.path.as_str()),
            ("2001:db8::1", 8443, "/x?y")
        );
        assert!(v6.literal().is_some_and(|(_, is6)| is6));
        let q = Url::parse("https://h.example?x=1").unwrap();
        assert_eq!(q.path, "/?x=1");
        assert_eq!(
            u.join("https://cdn.example/b?sig=1").unwrap().host,
            "cdn.example"
        );
        assert_eq!(u.join("/other").unwrap().path, "/other");
        assert_eq!(
            u.join("sibling").unwrap().path,
            "/v2/library/alpine/manifests/sibling"
        );
        assert_eq!(u.join("//cdn2.example/p").unwrap().host, "cdn2.example");
        assert!(Url::parse("ftp://x").is_err());
    }

    #[test]
    fn dns_messages_round_trip() {
        let q = dns_query(0x1234, "example.com", false);
        assert_eq!(&q[..2], &[0x12, 0x34]);
        assert_eq!(&q[12..25], b"\x07example\x03com\x00");
        // An answer: the question, then a compressed-name A record.
        let mut a = q.clone();
        a[2] |= 0x80;
        a[7] = 1;
        a.extend_from_slice(&[0xc0, 0x0c, 0, 1, 0, 1, 0, 0, 0, 60, 0, 4, 93, 184, 215, 14]);
        let ip = dns_answer(&a, 0x1234, false).unwrap().unwrap();
        assert_eq!(&ip[..4], &[93, 184, 215, 14]);
        assert_eq!(
            dns_answer(&a, 0x9999, false).unwrap(),
            None,
            "someone else's answer"
        );
        let mut nx = q.clone();
        nx[2] |= 0x80;
        nx[3] = 3;
        assert!(dns_answer(&nx, 0x1234, false).is_err());
    }

    #[test]
    fn http_responses_are_framed() {
        let mut r = HttpResponse::default();
        r.push(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\nX: y\r\n\r\nhel")
            .unwrap();
        assert!(!r.done);
        r.push(b"lo").unwrap();
        assert!(r.done);
        assert_eq!(r.body, b"hello");
        assert_eq!(r.header("x"), Some("y"));

        let mut c = HttpResponse::default();
        c.push(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n4\r\nWiki\r\n5;ext=1\r\npedia\r",
        )
        .unwrap();
        assert!(!c.done);
        c.push(b"\n0\r\n\r\n").unwrap();
        assert!(c.done);
        assert_eq!(c.body, b"Wikipedia");

        let mut e = HttpResponse::default();
        e.push(b"HTTP/1.0 200 OK\r\n\r\nall of it").unwrap();
        e.eof().unwrap();
        assert!(e.done && e.body == b"all of it");

        let mut short = HttpResponse::default();
        short
            .push(b"HTTP/1.1 200 OK\r\nContent-Length: 9\r\n\r\nab")
            .unwrap();
        assert!(short.eof().is_err());
    }
}
