//! The Wayland wire format: how requests are decoded and events encoded.
//!
//! Every message is a header of two 32-bit words — the object id, then the
//! message size in bytes (header included) in the high 16 bits and the opcode
//! in the low 16 — followed by its arguments, each a multiple of 4 bytes, in
//! the host's byte order (little-endian on everything nixvm runs). File
//! descriptors don't travel in the bytes: they ride the socket as
//! `SCM_RIGHTS` and are taken in order by the messages that carry an `fd`.

use std::collections::VecDeque;

use crate::kernel::ServedFd;

/// The largest message nixvm accepts. libwayland's limit is 4096; newer ones
/// allow more, so leave room.
pub const MAX_MESSAGE: usize = 1 << 16;

/// A violation that ends the client's connection: sent to it as a
/// `wl_display.error` on `object` before the connection is closed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProtoError {
    pub object: u32,
    pub code: u32,
    pub message: String,
}

/// `wl_display.error` codes.
pub const ERR_INVALID_OBJECT: u32 = 0;
pub const ERR_INVALID_METHOD: u32 = 1;
pub const ERR_NO_MEMORY: u32 = 2;

impl ProtoError {
    pub fn new(object: u32, code: u32, message: impl Into<String>) -> Self {
        Self {
            object,
            code,
            message: message.into(),
        }
    }
}

/// A request's header, if `buf` holds a whole one: `(object, opcode, size)`.
///
/// # Errors
/// A size that can't be a message (shorter than its header, unaligned, or
/// beyond [`MAX_MESSAGE`]).
pub fn header(buf: &[u8]) -> Result<Option<(u32, u16, usize)>, ProtoError> {
    if buf.len() < 8 {
        return Ok(None);
    }
    let obj = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let word = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    let size = (word >> 16) as usize;
    let op = (word & 0xffff) as u16;
    if size < 8 || !size.is_multiple_of(4) || size > MAX_MESSAGE {
        return Err(ProtoError::new(1, ERR_INVALID_METHOD, "bad message size"));
    }
    if buf.len() < size {
        return Ok(None);
    }
    Ok(Some((obj, op, size)))
}

/// A request's arguments, read in order.
pub struct Args<'a> {
    body: &'a [u8],
    pos: usize,
    fds: &'a mut VecDeque<ServedFd>,
    /// The object the request is on, for errors.
    obj: u32,
}

impl<'a> Args<'a> {
    pub fn new(obj: u32, body: &'a [u8], fds: &'a mut VecDeque<ServedFd>) -> Self {
        Self {
            body,
            pos: 0,
            fds,
            obj,
        }
    }

    fn short(&self) -> ProtoError {
        ProtoError::new(self.obj, ERR_INVALID_METHOD, "truncated request")
    }

    /// A `uint` (also an `object` or `new_id`; 0 is the null object).
    pub fn uint(&mut self) -> Result<u32, ProtoError> {
        let b = self
            .body
            .get(self.pos..self.pos + 4)
            .ok_or_else(|| self.short())?;
        self.pos += 4;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    /// An `int`.
    #[allow(clippy::cast_possible_wrap)]
    pub fn int(&mut self) -> Result<i32, ProtoError> {
        self.uint().map(|v| v as i32)
    }

    /// An `array`'s bytes.
    pub fn array(&mut self) -> Result<&'a [u8], ProtoError> {
        let len = self.uint()? as usize;
        let padded = len.checked_add(3).ok_or_else(|| self.short())? & !3;
        let body: &'a [u8] = self.body;
        let b = body
            .get(self.pos..self.pos + padded)
            .ok_or_else(|| self.short())?;
        self.pos += padded;
        Ok(&b[..len])
    }

    /// A `string` that may be null.
    pub fn opt_string(&mut self) -> Result<Option<String>, ProtoError> {
        let b = self.array()?;
        if b.is_empty() {
            return Ok(None);
        }
        match b.split_last() {
            Some((0, s)) => Ok(Some(String::from_utf8_lossy(s).into_owned())),
            _ => Err(ProtoError::new(
                self.obj,
                ERR_INVALID_METHOD,
                "string not NUL-terminated",
            )),
        }
    }

    /// A `string`.
    pub fn string(&mut self) -> Result<String, ProtoError> {
        self.opt_string()?
            .ok_or_else(|| ProtoError::new(self.obj, ERR_INVALID_METHOD, "null string"))
    }

    /// An `fd`: the next descriptor that came over the socket.
    pub fn fd(&mut self) -> Result<ServedFd, ProtoError> {
        self.fds
            .pop_front()
            .ok_or_else(|| ProtoError::new(self.obj, ERR_INVALID_METHOD, "missing fd"))
    }
}

/// An event being encoded.
#[derive(Debug, Clone)]
pub struct Ev {
    buf: Vec<u8>,
}

impl Ev {
    /// An event `op` on object `obj`.
    pub fn new(obj: u32, op: u16) -> Self {
        let mut buf = Vec::with_capacity(32);
        buf.extend_from_slice(&obj.to_le_bytes());
        buf.extend_from_slice(&u32::from(op).to_le_bytes());
        Self { buf }
    }

    #[must_use]
    pub fn uint(mut self, v: u32) -> Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    #[must_use]
    pub fn int(mut self, v: i32) -> Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn fixed(self, v: f64) -> Self {
        self.int((v * 256.0).round() as i32)
    }

    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn array(mut self, b: &[u8]) -> Self {
        self.buf.extend_from_slice(&(b.len() as u32).to_le_bytes());
        self.buf.extend_from_slice(b);
        while !self.buf.len().is_multiple_of(4) {
            self.buf.push(0);
        }
        self
    }

    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn string(mut self, s: &str) -> Self {
        self.buf
            .extend_from_slice(&(s.len() as u32 + 1).to_le_bytes());
        self.buf.extend_from_slice(s.as_bytes());
        self.buf.push(0);
        while !self.buf.len().is_multiple_of(4) {
            self.buf.push(0);
        }
        self
    }

    /// The encoded message, its size filled in.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn finish(mut self) -> Vec<u8> {
        let size = self.buf.len() as u32;
        let op = u32::from_le_bytes([self.buf[4], self.buf[5], self.buf[6], self.buf[7]]);
        self.buf[4..8].copy_from_slice(&((size << 16) | (op & 0xffff)).to_le_bytes());
        self.buf
    }
}

/// An array of `u32`s, as `wl_array` carries them.
pub fn u32_array(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_event_round_trips_through_the_decoder() {
        let msg = Ev::new(7, 3)
            .uint(42)
            .int(-5)
            .fixed(1.5)
            .string("wl_seat")
            .array(&[1, 2, 3])
            .finish();
        assert_eq!(msg.len() % 4, 0);
        let (obj, op, size) = header(&msg).unwrap().unwrap();
        assert_eq!((obj, op, size), (7, 3, msg.len()));
        let mut fds = VecDeque::new();
        let mut a = Args::new(obj, &msg[8..], &mut fds);
        assert_eq!(a.uint().unwrap(), 42);
        assert_eq!(a.int().unwrap(), -5);
        assert_eq!(a.int().unwrap(), 384, "1.5 in 24.8 fixed point");
        assert_eq!(a.string().unwrap(), "wl_seat");
        assert_eq!(a.array().unwrap(), &[1, 2, 3]);
        assert!(a.uint().is_err(), "past the end is an error");
        assert!(a.fd().is_err(), "no fd came with it");
    }

    #[test]
    fn partial_and_bogus_headers() {
        let msg = Ev::new(1, 0).uint(9).finish();
        assert_eq!(header(&msg[..10]).unwrap(), None, "incomplete");
        assert_eq!(header(&msg[..4]).unwrap(), None, "no header yet");
        let mut bad = msg.clone();
        bad[6] = 6; // size 6: shorter than a header
        bad[7] = 0;
        assert!(header(&bad).is_err());
    }

    #[test]
    fn null_and_unterminated_strings() {
        let mut fds = VecDeque::new();
        let null = 0u32.to_le_bytes();
        assert_eq!(Args::new(1, &null, &mut fds).opt_string().unwrap(), None);
        assert!(Args::new(1, &null, &mut fds).string().is_err());
        let mut raw = 2u32.to_le_bytes().to_vec();
        raw.extend_from_slice(b"ab\0\0");
        assert!(Args::new(1, &raw, &mut fds).string().is_err());
    }
}
