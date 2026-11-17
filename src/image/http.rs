//! A small blocking HTTP/1.1 client, for downloading root images.
//!
//! TLS is purecrypto's — pure Rust, the same as the OCI pull over a guest
//! network link uses — with its embedded CA roots, so a download
//! needs no C code, no libc of its own and no `curl` on the host. Where the
//! bytes travel is the embedder's [`Connector`]: the host's TCP
//! ([`TcpConnector`]) unless told otherwise.
//!
//! One request per connection (`Connection: close`), redirects followed (a
//! GitHub release asset is a redirect to its CDN), the body streamed to the
//! caller rather than held in memory.

use std::io::{self, BufRead, BufReader, Read, Write};
use std::sync::Arc;
use std::time::Duration;

use purecrypto::tls::{Config, Connection, RootCertStore, Step};
use purecrypto::x509::Time;

const MAX_REDIRECTS: u32 = 10;
/// A response head longer than this is refused.
const MAX_HEAD: usize = 64 << 10;

/// A byte stream to a server: what a [`Connector`] hands back.
pub trait Transport: Read + Write + Send {}

impl<T: Read + Write + Send> Transport for T {}

/// Opens connections for downloads: the host's TCP by default
/// ([`TcpConnector`]), or whatever the embedder routes them through (a proxy,
/// its own network stack).
pub trait Connector: Send + Sync {
    /// A connection to `host` (a name or an address literal) at `port`. TLS,
    /// when the URL asks for it, is layered on top by the client.
    ///
    /// # Errors
    /// The name doesn't resolve, or no address accepts the connection.
    fn connect(&self, host: &str, port: u16) -> io::Result<Box<dyn Transport>>;
}

/// [`Connector`] over the host's TCP (`std::net`), with a timeout on connecting
/// and on each read and write.
#[derive(Debug, Clone, Copy)]
pub struct TcpConnector {
    /// How long a connection attempt, or a read or write, may wait.
    pub timeout: Duration,
}

impl Default for TcpConnector {
    fn default() -> Self {
        Self {
            timeout: Duration::from_secs(60),
        }
    }
}

impl Connector for TcpConnector {
    fn connect(&self, host: &str, port: u16) -> io::Result<Box<dyn Transport>> {
        use std::net::{TcpStream, ToSocketAddrs};
        let mut last = io::Error::new(io::ErrorKind::NotFound, format!("{host}: no address"));
        for addr in (host, port).to_socket_addrs()? {
            match TcpStream::connect_timeout(&addr, self.timeout) {
                Ok(s) => {
                    s.set_read_timeout(Some(self.timeout))?;
                    s.set_write_timeout(Some(self.timeout))?;
                    let _ = s.set_nodelay(true);
                    return Ok(Box::new(s));
                }
                Err(e) => last = e,
            }
        }
        Err(last)
    }
}

/// `scheme://host[:port]/path` split up.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Url {
    pub tls: bool,
    pub host: String,
    pub port: u16,
    /// Path and query, starting with `/`.
    pub path: String,
}

impl Url {
    pub(crate) fn parse(url: &str) -> Result<Self, String> {
        let (tls, rest) = if let Some(r) = url.strip_prefix("https://") {
            (true, r)
        } else if let Some(r) = url.strip_prefix("http://") {
            (false, r)
        } else {
            return Err(format!("not an http(s) URL: {url}"));
        };
        let rest = rest.split('#').next().unwrap_or_default();
        let (authority, path) = match rest.find(['/', '?']) {
            Some(i) if rest.as_bytes()[i] == b'/' => (&rest[..i], rest[i..].to_string()),
            Some(i) => (&rest[..i], format!("/{}", &rest[i..])),
            None => (rest, "/".to_string()),
        };
        if authority.contains('@') {
            return Err(format!("credentials in a URL are not supported: {url}"));
        }
        let default = if tls { 443 } else { 80 };
        let (host, port) = if let Some(v6) = authority.strip_prefix('[') {
            let (h, after) = v6.split_once(']').ok_or_else(|| format!("bad URL {url}"))?;
            let port = match after.strip_prefix(':') {
                Some(p) => p.parse().map_err(|_| format!("bad port in {url}"))?,
                None => default,
            };
            (h.to_string(), port)
        } else {
            match authority.rsplit_once(':') {
                Some((h, p)) => (
                    h.to_string(),
                    p.parse().map_err(|_| format!("bad port in {url}"))?,
                ),
                None => (authority.to_string(), default),
            }
        };
        if host.is_empty() || host.contains(|c: char| c.is_ascii_whitespace() || c.is_control()) {
            return Err(format!("bad host in {url}"));
        }
        if path.contains(|c: char| c.is_ascii_whitespace() || c.is_control()) {
            return Err(format!("bad path in {url}"));
        }
        Ok(Self {
            tls,
            host: host.to_ascii_lowercase(),
            port,
            path,
        })
    }

    /// `location` (absolute, scheme-relative or a path) seen from `self`.
    fn join(&self, location: &str) -> Result<Self, String> {
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
        let mut next = self.clone();
        next.path = path;
        Self::parse(&next.to_string())
    }
}

impl core::fmt::Display for Url {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let scheme = if self.tls { "https" } else { "http" };
        let host = if self.host.contains(':') {
            format!("[{}]", self.host)
        } else {
            self.host.clone()
        };
        if self.port == if self.tls { 443 } else { 80 } {
            write!(f, "{scheme}://{host}{}", self.path)
        } else {
            write!(f, "{scheme}://{host}:{}{}", self.port, self.path)
        }
    }
}

/// TLS's randomness: purecrypto's `OsRng`, the same as the guest's
/// ([`crate::entropy`]).
struct Entropy;

impl purecrypto::tls::EntropySource for Entropy {
    fn fill(&self, dest: &mut [u8]) {
        crate::entropy::fill(dest);
    }
}

fn tls_err(e: &purecrypto::tls::Error) -> io::Error {
    io::Error::other(format!("TLS: {e:?}"))
}

/// A blocking TLS client stream over a [`Transport`].
struct TlsStream {
    io: Box<dyn Transport>,
    tls: Box<Connection>,
    /// Decrypted bytes not yet read.
    plain: Vec<u8>,
    at: usize,
}

impl TlsStream {
    /// Handshake with `host` over `io`, verifying its certificate against
    /// `roots`.
    fn connect(io: Box<dyn Transport>, host: &str, roots: RootCertStore) -> io::Result<Self> {
        let cfg = Config::builder()
            .rng(Arc::new(Entropy))
            .roots(roots)
            .server_name(host.to_string())
            .alpn(vec![b"http/1.1".to_vec()])
            .verification_time(Time::from_unix(crate::clock::now_unix().as_secs()))
            .build();
        let tls = Connection::client(&cfg).map_err(|e| tls_err(&e))?;
        let mut s = Self {
            io,
            tls: Box::new(tls),
            plain: Vec::new(),
            at: 0,
        };
        let mut buf = vec![0u8; 16 << 10];
        loop {
            s.flush_tls()?;
            if s.tls.is_handshake_complete() {
                return Ok(s);
            }
            let n = s.io.read(&mut buf)?;
            if n == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "TLS: the server closed the connection during the handshake",
                ));
            }
            s.feed(&buf[..n])?;
        }
    }

    /// Send whatever the engine has for the wire.
    fn flush_tls(&mut self) -> io::Result<()> {
        if !self.tls.is_handshake_complete() {
            while let Step::WantWrite = self.tls.drive().map_err(|e| tls_err(&e))? {
                let out = self.tls.pop().map_err(|e| tls_err(&e))?;
                self.io.write_all(&out)?;
            }
        }
        loop {
            let out = self.tls.pop().map_err(|e| tls_err(&e))?;
            if out.is_empty() {
                break;
            }
            self.io.write_all(&out)?;
        }
        self.io.flush()
    }

    fn feed(&mut self, mut wire: &[u8]) -> io::Result<()> {
        while !wire.is_empty() {
            let n = self.tls.feed(wire).map_err(|e| tls_err(&e))?;
            if n == 0 {
                break;
            }
            wire = &wire[n..];
        }
        Ok(())
    }
}

impl Read for TlsStream {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let mut buf = Vec::new();
        while self.at == self.plain.len() {
            self.plain.clear();
            self.at = 0;
            if self
                .tls
                .recv_into(&mut self.plain)
                .map_err(|e| tls_err(&e))?
                > 0
            {
                break;
            }
            if self.tls.received_close_notify() {
                return Ok(0);
            }
            if buf.is_empty() {
                buf = vec![0u8; 16 << 10];
            }
            let n = self.io.read(&mut buf)?;
            if n == 0 {
                // Closed without close_notify: the HTTP framing (or, for a
                // body read to the end, the digest) tells a cut-off transfer.
                return Ok(0);
            }
            self.feed(&buf[..n])?;
            // Post-handshake messages may want an answer (a key update).
            self.flush_tls()?;
        }
        let n = out.len().min(self.plain.len() - self.at);
        out[..n].copy_from_slice(&self.plain[self.at..self.at + n]);
        self.at += n;
        Ok(n)
    }
}

impl Write for TlsStream {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.tls.send(data).map_err(|e| tls_err(&e))?;
        self.flush_tls()?;
        Ok(data.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.flush_tls()
    }
}

/// How a response body ends.
#[derive(Debug, Clone, Copy)]
enum Framing {
    ToEof,
    Length(u64),
    Chunked,
}

/// A response's head: status and headers.
struct Head {
    status: u16,
    headers: Vec<(String, String)>,
}

impl Head {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    fn framing(&self) -> Framing {
        if self
            .header("transfer-encoding")
            .is_some_and(|t| t.to_ascii_lowercase().contains("chunked"))
        {
            Framing::Chunked
        } else if let Some(n) = self
            .header("content-length")
            .and_then(|n| n.trim().parse().ok())
        {
            Framing::Length(n)
        } else if self.status == 204 || self.status == 304 {
            Framing::Length(0)
        } else {
            Framing::ToEof
        }
    }
}

fn read_line(r: &mut dyn BufRead, limit: &mut usize) -> io::Result<String> {
    let mut line = Vec::new();
    loop {
        let buf = r.fill_buf()?;
        if buf.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "the connection closed mid-response",
            ));
        }
        let (take, done) = match buf.iter().position(|&b| b == b'\n') {
            Some(i) => (i + 1, true),
            None => (buf.len(), false),
        };
        line.extend_from_slice(&buf[..take]);
        r.consume(take);
        if line.len() > *limit {
            return Err(io::Error::other("an HTTP response head over 64 KiB"));
        }
        if done {
            *limit -= line.len();
            while line.last().is_some_and(|b| *b == b'\n' || *b == b'\r') {
                line.pop();
            }
            return Ok(String::from_utf8_lossy(&line).into_owned());
        }
    }
}

fn read_head(r: &mut dyn BufRead) -> io::Result<Head> {
    let mut limit = MAX_HEAD;
    let status_line = read_line(r, &mut limit)?;
    let status = status_line
        .strip_prefix("HTTP/1.")
        .and_then(|s| s.split(' ').nth(1))
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| io::Error::other(format!("not an HTTP response: {status_line:?}")))?;
    let mut headers = Vec::new();
    loop {
        let line = read_line(r, &mut limit)?;
        if line.is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            headers.push((k.trim().to_string(), v.trim().to_string()));
        }
    }
    Ok(Head { status, headers })
}

/// Copy a body framed by `framing` from `r` to `sink`; how many bytes.
fn copy_body(
    r: &mut dyn BufRead,
    framing: Framing,
    sink: &mut dyn FnMut(&[u8]) -> io::Result<()>,
) -> io::Result<u64> {
    let cut = || io::Error::new(io::ErrorKind::UnexpectedEof, "the transfer was cut off");
    let mut total = 0u64;
    let mut copy = |r: &mut dyn BufRead, mut left: Option<u64>| -> io::Result<()> {
        while left != Some(0) {
            let buf = r.fill_buf()?;
            if buf.is_empty() {
                return if left.is_none() { Ok(()) } else { Err(cut()) };
            }
            let n = left.map_or(buf.len(), |l| buf.len().min(l as usize));
            sink(&buf[..n])?;
            r.consume(n);
            total += n as u64;
            left = left.map(|l| l - n as u64);
        }
        Ok(())
    };
    match framing {
        Framing::ToEof => copy(r, None)?,
        Framing::Length(n) => copy(r, Some(n))?,
        Framing::Chunked => loop {
            let mut limit = MAX_HEAD;
            let line = read_line(r, &mut limit)?;
            let size = u64::from_str_radix(line.split(';').next().unwrap_or("").trim(), 16)
                .map_err(|_| io::Error::other("a bad chunk size"))?;
            if size == 0 {
                // Trailers, if any, end with an empty line.
                while !read_line(r, &mut limit)?.is_empty() {}
                break;
            }
            copy(r, Some(size))?;
            if !read_line(r, &mut limit)?.is_empty() {
                return Err(io::Error::other("a chunk without its CRLF"));
            }
        },
    }
    Ok(total)
}

/// Downloads over HTTP(S), through a [`Connector`].
pub(crate) struct Client<'a> {
    pub connector: &'a dyn Connector,
    pub roots: RootCertStore,
}

impl Client<'_> {
    /// `GET url`, following redirects, and stream the body of the final `200`
    /// to `sink`. `progress` hears (bytes so far, total if known) as it goes.
    pub(crate) fn get(
        &self,
        url: &str,
        sink: &mut dyn FnMut(&[u8]) -> io::Result<()>,
        progress: &mut dyn FnMut(u64, Option<u64>),
    ) -> Result<u64, String> {
        let mut url = Url::parse(url)?;
        for _ in 0..=MAX_REDIRECTS {
            let fail = |e: io::Error| format!("GET {url}: {e}");
            let io = self
                .connector
                .connect(&url.host, url.port)
                .map_err(|e| format!("connect to {}:{}: {e}", url.host, url.port))?;
            let mut io: Box<dyn Transport> = if url.tls {
                Box::new(TlsStream::connect(io, &url.host, self.roots.clone()).map_err(fail)?)
            } else {
                io
            };
            let host = if url.host.contains(':') {
                format!("[{}]", url.host)
            } else {
                url.host.clone()
            };
            let port = if url.port == if url.tls { 443 } else { 80 } {
                String::new()
            } else {
                format!(":{}", url.port)
            };
            let req = format!(
                "GET {} HTTP/1.1\r\nHost: {host}{port}\r\nUser-Agent: nixvm/{}\r\n\
                 Accept: */*\r\nAccept-Encoding: identity\r\nConnection: close\r\n\r\n",
                url.path,
                env!("CARGO_PKG_VERSION"),
            );
            io.write_all(req.as_bytes()).map_err(fail)?;
            io.flush().map_err(fail)?;
            let mut r = BufReader::with_capacity(64 << 10, io);
            let head = read_head(&mut r).map_err(fail)?;
            match head.status {
                200 => {
                    let framing = head.framing();
                    let total = match framing {
                        Framing::Length(n) => Some(n),
                        _ => None,
                    };
                    let mut got = 0u64;
                    progress(0, total);
                    let mut counted = |b: &[u8]| {
                        sink(b)?;
                        got += b.len() as u64;
                        progress(got, total);
                        Ok(())
                    };
                    return copy_body(&mut r, framing, &mut counted).map_err(fail);
                }
                301 | 302 | 303 | 307 | 308 => {
                    let loc = head.header("location").ok_or_else(|| {
                        format!("GET {url}: HTTP {} without a Location", head.status)
                    })?;
                    let next = url.join(loc)?;
                    if url.tls && !next.tls {
                        return Err(format!("GET {url}: redirected from HTTPS to {next}"));
                    }
                    url = next;
                }
                s => return Err(format!("GET {url}: HTTP {s}")),
            }
        }
        Err(format!("GET {url}: too many redirects"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls_parse_and_join() {
        let u = Url::parse("https://GitHub.com/o/r/releases/download/t/f.sqfs").unwrap();
        assert!(u.tls);
        assert_eq!((u.host.as_str(), u.port), ("github.com", 443));
        assert_eq!(u.path, "/o/r/releases/download/t/f.sqfs");
        let v6 = Url::parse("http://[::1]:8080?x=1").unwrap();
        assert_eq!(
            (v6.host.as_str(), v6.port, v6.path.as_str()),
            ("::1", 8080, "/?x=1")
        );
        assert_eq!(v6.to_string(), "http://[::1]:8080/?x=1");
        assert_eq!(
            u.join("https://cdn.example/a?sig=1").unwrap().to_string(),
            "https://cdn.example/a?sig=1"
        );
        assert_eq!(u.join("/x").unwrap().to_string(), "https://github.com/x");
        assert_eq!(
            u.join("g.sqfs").unwrap().to_string(),
            "https://github.com/o/r/releases/download/t/g.sqfs"
        );
        assert!(Url::parse("ftp://x/").is_err());
        assert!(Url::parse("http://u:p@x/").is_err());
        assert!(Url::parse("http://x/a b").is_err());
    }

    #[test]
    fn bodies_are_framed() {
        let mut out = Vec::new();
        let mut sink = |b: &[u8]| {
            out.extend_from_slice(b);
            Ok(())
        };
        let mut r: &[u8] = b"4;ext=1\r\nabcd\r\n2\r\nef\r\n0\r\nTrailer: x\r\n\r\n";
        assert_eq!(copy_body(&mut r, Framing::Chunked, &mut sink).unwrap(), 6);
        let mut r: &[u8] = b"hello, more";
        assert_eq!(copy_body(&mut r, Framing::Length(5), &mut sink).unwrap(), 5);
        let mut r: &[u8] = b"rest";
        assert_eq!(copy_body(&mut r, Framing::ToEof, &mut sink).unwrap(), 4);
        assert_eq!(out, b"abcdefhellorest");
        let mut r: &[u8] = b"abc";
        let e = copy_body(&mut r, Framing::Length(5), &mut |_| Ok(())).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::UnexpectedEof);
    }

    #[test]
    fn heads_parse() {
        let mut r: &[u8] = b"HTTP/1.1 302 Found\r\nLocation: /x\r\nContent-Length: 0\r\n\r\nbody";
        let h = read_head(&mut r).unwrap();
        assert_eq!(h.status, 302);
        assert_eq!(h.header("location"), Some("/x"));
        assert!(matches!(h.framing(), Framing::Length(0)));
        assert_eq!(r, b"body");
        let mut r: &[u8] = b"SSH-2.0-x\r\n\r\n";
        assert!(read_head(&mut r).is_err());
    }
}
