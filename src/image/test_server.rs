//! A small HTTP/1.1 server on loopback for the download and pull tests, plain
//! or over TLS (purecrypto's server side, with a self-signed certificate).

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

/// A request as the server saw it.
#[derive(Debug, Clone)]
pub(crate) struct Req {
    /// Path and query.
    pub path: String,
    /// Names lowercased.
    pub headers: Vec<(String, String)>,
}

impl Req {
    pub(crate) fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }
}

/// What to answer.
#[derive(Debug, Clone, Default)]
pub(crate) struct Resp {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    /// Send the body chunked rather than with a `Content-Length`.
    pub chunked: bool,
}

impl Resp {
    pub(crate) fn ok(body: impl Into<Vec<u8>>) -> Self {
        Self {
            status: 200,
            body: body.into(),
            ..Self::default()
        }
    }

    pub(crate) fn status(status: u16) -> Self {
        Self {
            status,
            ..Self::default()
        }
    }

    pub(crate) fn with(mut self, k: &str, v: &str) -> Self {
        self.headers.push((k.to_string(), v.to_string()));
        self
    }
}

pub(crate) type Handler = Arc<dyn Fn(&Req) -> Resp + Send + Sync>;

/// A running server: where it listens and the paths it was asked for.
pub(crate) struct Server {
    pub addr: SocketAddr,
    pub log: Arc<Mutex<Vec<String>>>,
}

impl Server {
    /// The paths asked for so far, and forget them.
    pub(crate) fn take_log(&self) -> Vec<String> {
        std::mem::take(&mut *self.log.lock().unwrap())
    }
}

/// Serve `handler` on a fresh loopback port, over TLS with `tls`'s identity
/// when given.
pub(crate) fn serve(
    tls: Option<(Vec<u8>, purecrypto::tls::SigningKey)>,
    handler: Handler,
) -> Server {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let log = Arc::new(Mutex::new(Vec::new()));
    let l = log.clone();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(conn) = conn else { return };
            let (tls, handler, log) = (tls.clone(), handler.clone(), l.clone());
            std::thread::spawn(move || {
                let mut stream: Box<dyn Stream> = match tls {
                    Some((cert, key)) => match tls_accept(conn, cert, key) {
                        Ok(s) => Box::new(s),
                        Err(_) => return,
                    },
                    None => Box::new(conn),
                };
                let _ = answer(&mut *stream, &*handler, &log);
            });
        }
    });
    Server { addr, log }
}

/// Where the handler of a [`serve_later`] server goes once it's made.
pub(crate) type Later = Arc<std::sync::OnceLock<Handler>>;

/// [`serve`], with the handler set afterwards (it may need the address).
pub(crate) fn serve_later(tls: Option<(Vec<u8>, purecrypto::tls::SigningKey)>) -> (Server, Later) {
    let later: Later = Arc::new(std::sync::OnceLock::new());
    let l = later.clone();
    let srv = serve(
        tls,
        Arc::new(move |r: &Req| (l.get().expect("a handler"))(r)),
    );
    (srv, later)
}

trait Stream: Read + Write + Send {}
impl<T: Read + Write + Send> Stream for T {}

/// Answer requests on one connection until the client closes it.
fn answer(
    stream: &mut dyn Stream,
    handler: &dyn Fn(&Req) -> Resp,
    log: &Mutex<Vec<String>>,
) -> io::Result<()> {
    loop {
        let mut r = BufReader::new(&mut *stream);
        let mut first = String::new();
        if r.read_line(&mut first)? == 0 {
            return Ok(());
        }
        let mut headers = Vec::new();
        loop {
            let mut l = String::new();
            r.read_line(&mut l)?;
            if l.trim().is_empty() {
                break;
            }
            if let Some((k, v)) = l.split_once(':') {
                headers.push((k.trim().to_ascii_lowercase(), v.trim().to_string()));
            }
        }
        drop(r);
        let req = Req {
            path: first.split(' ').nth(1).unwrap_or("").to_string(),
            headers,
        };
        log.lock().unwrap().push(req.path.clone());
        let resp = handler(&req);
        let mut out = format!("HTTP/1.1 {} X\r\n", resp.status).into_bytes();
        for (k, v) in &resp.headers {
            out.extend_from_slice(format!("{k}: {v}\r\n").as_bytes());
        }
        if resp.chunked {
            out.extend_from_slice(b"Transfer-Encoding: chunked\r\n\r\n");
            for c in resp.body.chunks(1000) {
                out.extend_from_slice(format!("{:x}\r\n", c.len()).as_bytes());
                out.extend_from_slice(c);
                out.extend_from_slice(b"\r\n");
            }
            out.extend_from_slice(b"0\r\n\r\n");
        } else {
            out.extend_from_slice(
                format!("Content-Length: {}\r\n\r\n", resp.body.len()).as_bytes(),
            );
            out.extend_from_slice(&resp.body);
        }
        stream.write_all(&out)?;
        stream.flush()?;
    }
}

/// A self-signed certificate (DER) for `name`, and its key.
pub(crate) fn self_signed(names: &[&str]) -> (Vec<u8>, purecrypto::tls::SigningKey) {
    use purecrypto::ec::{BoxedEcdsaPrivateKey, CurveId};
    use purecrypto::x509::{CertSigner, Certificate, DistinguishedName, Time, Validity};
    let key = BoxedEcdsaPrivateKey::generate(CurveId::P256, &mut purecrypto::rng::OsRng);
    let now = crate::clock::now_unix().as_secs();
    let validity = Validity::new(Time::from_unix(now - 3600), Time::from_unix(now + 86400));
    let cert = Certificate::self_signed_general(
        &CertSigner::Ecdsa(&key),
        &DistinguishedName::common_name(names[0]),
        &validity,
        1,
        false,
        names,
    )
    .unwrap()
    .to_der()
    .to_vec();
    (cert, purecrypto::tls::SigningKey::Ecdsa(key))
}

/// `der` as a PEM certificate.
pub(crate) fn pem(der: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut b64 = String::new();
    for c in der.chunks(3) {
        let n = (u32::from(c[0]) << 16)
            | (u32::from(*c.get(1).unwrap_or(&0)) << 8)
            | u32::from(*c.get(2).unwrap_or(&0));
        for i in 0..4 {
            b64.push(if i <= c.len() {
                T[(n >> (18 - 6 * i)) as usize & 63] as char
            } else {
                '='
            });
        }
    }
    let mut out = String::from("-----BEGIN CERTIFICATE-----\n");
    for line in b64.as_bytes().chunks(64) {
        out.push_str(std::str::from_utf8(line).unwrap());
        out.push('\n');
    }
    out.push_str("-----END CERTIFICATE-----\n");
    out
}

/// Connects every host name to one address: the test server's.
#[derive(Debug)]
pub(crate) struct To(pub SocketAddr);

impl rsurl::net::Connector for To {
    fn connect(
        &self,
        _host: &str,
        _port: u16,
        _timeout: Option<std::time::Duration>,
    ) -> rsurl::Result<Box<dyn rsurl::net::NetStream>> {
        Ok(Box::new(
            TcpStream::connect(self.0).map_err(rsurl::Error::Io)?,
        ))
    }
}

/// The server side of TLS over a connection, as a blocking stream.
struct ServerTls {
    io: TcpStream,
    tls: purecrypto::tls::Connection,
    plain: Vec<u8>,
    at: usize,
}

fn tls_err(e: &purecrypto::tls::Error) -> io::Error {
    io::Error::other(format!("{e:?}"))
}

fn tls_accept(
    io: TcpStream,
    cert: Vec<u8>,
    key: purecrypto::tls::SigningKey,
) -> io::Result<ServerTls> {
    use purecrypto::tls::{Config, Connection};
    let cfg = Config::builder().os_rng().identity(vec![cert], key).build();
    let tls = Connection::server(&cfg).map_err(|e| tls_err(&e))?;
    let mut s = ServerTls {
        io,
        tls,
        plain: Vec::new(),
        at: 0,
    };
    let mut buf = [0u8; 16384];
    while !s.tls.is_handshake_complete() {
        let n = s.io.read(&mut buf)?;
        if n == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        s.feed(&buf[..n])?;
    }
    Ok(s)
}

impl ServerTls {
    fn feed(&mut self, mut wire: &[u8]) -> io::Result<()> {
        while !wire.is_empty() {
            let n = self.tls.feed(wire).map_err(|e| tls_err(&e))?;
            if n == 0 {
                break;
            }
            wire = &wire[n..];
        }
        if !self.tls.is_handshake_complete() {
            while let purecrypto::tls::Step::WantWrite =
                self.tls.drive().map_err(|e| tls_err(&e))?
            {
                let out = self.tls.pop().map_err(|e| tls_err(&e))?;
                self.io.write_all(&out)?;
            }
        }
        self.out()
    }

    fn out(&mut self) -> io::Result<()> {
        loop {
            let out = self.tls.pop().map_err(|e| tls_err(&e))?;
            if out.is_empty() {
                return Ok(());
            }
            self.io.write_all(&out)?;
        }
    }
}

impl Read for ServerTls {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        let mut buf = [0u8; 16384];
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
            let n = self.io.read(&mut buf)?;
            if n == 0 {
                return Ok(0);
            }
            self.feed(&buf[..n])?;
        }
        let n = out.len().min(self.plain.len() - self.at);
        out[..n].copy_from_slice(&self.plain[self.at..self.at + n]);
        self.at += n;
        Ok(n)
    }
}

impl Write for ServerTls {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        self.tls.send(data).map_err(|e| tls_err(&e))?;
        self.out()?;
        Ok(data.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.out()
    }
}
