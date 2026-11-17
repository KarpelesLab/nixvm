//! Native HTTP(S) for the `fetch` feature: an image source URL
//! ([`crate::image`]) and registry pulls ([`crate::oci`]), over
//! [rsurl](https://crates.io/crates/rsurl) — pure Rust, TLS on purecrypto with
//! its embedded CA roots, so it builds where no libc or C compiler is (the
//! libc-free fullrust target).
//!
//! Redirects are followed (a registry's blobs live on a CDN), and
//! `Authorization` is dropped when one leaves the host. Bodies come back as
//! the wire had them (`Accept-Encoding: identity`, nothing decoded): what's
//! checked against a digest is what was sent.

use std::io::{self, Read};
use std::sync::Arc;
use std::time::Duration;

/// Opens the connections a download makes, instead of dialing TCP directly —
/// a proxy of the embedder's own, its own network stack, a test double.
/// rsurl's [`Connector`](rsurl::net::Connector): it hands back a plaintext
/// [`NetStream`](rsurl::net::NetStream) and TLS is layered on top. Over a
/// custom connector a response is read whole before it's handed over (rsurl
/// streams bodies only over its own TCP), and HTTP/1.1 is used.
pub use rsurl::net::{Connector, NetStream};

/// An error body read for a refused request is cut at this.
const MAX_ERROR_BODY: u64 = 64 << 10;

/// How requests are made: shared by every request of a download or pull.
#[derive(Clone, Default)]
pub(crate) struct Http {
    pub connector: Option<Arc<dyn Connector>>,
    /// A PEM file of CA certificates to trust *instead of* the embedded roots.
    pub ca_bundle: Option<String>,
    /// A directory of CA certificates (PEM) to trust *besides* them.
    pub ca_path: Option<String>,
    /// `http://host:port`, `socks5h://…` — curl's `--proxy`.
    pub proxy: Option<String>,
}

impl core::fmt::Debug for Http {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Http")
            .field("connector", &self.connector.is_some())
            .field("ca_bundle", &self.ca_bundle)
            .field("ca_path", &self.ca_path)
            .field("proxy", &self.proxy)
            .finish()
    }
}

/// A response's status and headers, and — unless the body went to a sink —
/// its body.
#[derive(Debug, Default)]
pub(crate) struct Reply {
    pub status: u16,
    #[cfg_attr(not(feature = "oci"), allow(dead_code))]
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Http {
    fn request(&self, url: &str, headers: &[(String, String)]) -> Result<rsurl::Request, String> {
        let mut r = rsurl::Request::get(url)
            .map_err(|e| format!("GET {url}: {e}"))?
            .follow_redirects(true)
            .max_redirs(10)
            .decompress(false)
            .idn(false)
            .connect_timeout(Duration::from_secs(30))
            .read_timeout(Some(Duration::from_secs(60)))
            .header("User-Agent", concat!("nixvm/", env!("CARGO_PKG_VERSION")))
            .header("Accept-Encoding", "identity");
        for (k, v) in headers {
            r = r.header(k, v);
        }
        if let Some(c) = &self.connector {
            r = r.connector(c.clone());
        }
        if let Some(p) = &self.ca_bundle {
            r = r.ca_bundle(p);
        }
        if let Some(p) = &self.ca_path {
            r = r.ca_path(p);
        }
        if let Some(p) = &self.proxy {
            r = r.proxy(p).map_err(|e| format!("proxy {p}: {e}"))?;
        }
        Ok(r)
    }

    /// `GET url`; the body of a `200` goes to `sink` as it arrives (and
    /// `progress` hears bytes so far and the total, when known), any other
    /// status's (up to 64 KiB) comes back in [`Reply::body`].
    pub(crate) fn get_to(
        &self,
        url: &str,
        headers: &[(String, String)],
        sink: &mut dyn FnMut(&[u8]) -> io::Result<()>,
        progress: &mut dyn FnMut(u64, Option<u64>),
    ) -> Result<Reply, String> {
        let fail = |e: &dyn core::fmt::Display| format!("GET {url}: {e}");
        let mut body = self
            .request(url, headers)?
            .send_reader()
            .map_err(|e| fail(&e))?;
        let mut reply = Reply {
            status: body.status(),
            headers: body.head().headers.clone(),
            body: Vec::new(),
        };
        if reply.status != 200 {
            let _ = (&mut body)
                .take(MAX_ERROR_BODY)
                .read_to_end(&mut reply.body);
            return Ok(reply);
        }
        let total = body
            .header("content-length")
            .and_then(|n| n.trim().parse().ok());
        let mut buf = vec![0u8; 64 << 10];
        let mut got = 0u64;
        progress(0, total);
        loop {
            let n = match body.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(fail(&e)),
            };
            sink(&buf[..n]).map_err(|e| fail(&e))?;
            got += n as u64;
            progress(got, total);
        }
        if let Some(t) = total
            && got != t
        {
            return Err(fail(&format!("cut off at {got} of {t} bytes")));
        }
        Ok(reply)
    }

    /// `GET url` into memory: at most `limit` bytes of a `200`'s body.
    #[cfg_attr(not(feature = "oci"), allow(dead_code))]
    pub(crate) fn get(
        &self,
        url: &str,
        headers: &[(String, String)],
        limit: usize,
    ) -> Result<Reply, String> {
        let mut data = Vec::new();
        let mut reply = self.get_to(
            url,
            headers,
            &mut |b| {
                if data.len() + b.len() > limit {
                    return Err(io::Error::other(format!("a response over {limit} bytes")));
                }
                data.extend_from_slice(b);
                Ok(())
            },
            &mut |_, _| {},
        )?;
        if reply.status == 200 {
            reply.body = data;
        }
        Ok(reply)
    }
}
