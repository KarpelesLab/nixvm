//! Sockets nixvm itself serves inside the guest: a listening `AF_UNIX` socket
//! at a path in the guest's filesystem whose connections nixvm, not a guest
//! process, answers. The built-in Wayland display ([`crate::display`]) is
//! one: a guest client connects to `/tmp/wayland-0` like it would to a
//! compositor, and nixvm reads its requests and writes the events back.
//!
//! The host holds the listener and the server end (end 1) of every connected
//! [`Pair`] the way a guest fd would — one reference each, so the guest sees
//! a hangup when nixvm closes a connection and nixvm sees one when the last
//! guest fd on the client end closes. `SCM_RIGHTS` descriptors ride both ways
//! as [`ServedFd`]s, each owning the in-flight backing reference.
//!
//! Lock order: these take `vfs` and `net` (in that order), never while the
//! caller holds either.

use std::collections::VecDeque;

use super::{Addr, Kind, Sock, SockOpts, Ucred, unix_key};
use crate::abi::errno::Errno;
use crate::kernel::fd::{Fd, FileOffset};
use crate::kernel::{Kernel, err};

/// The mount namespace (and root) host-held sockets are named in: the VM's
/// first, which the guest's init starts in.
const HOST_NS: u32 = 0;

/// A file descriptor in flight between the guest and nixvm: received with a
/// message from a guest client, or made by nixvm to send to one. It owns one
/// backing reference, released by [`Kernel::served_drop_fd`] (or by sending it).
#[derive(Debug)]
pub struct ServedFd(Fd);

/// What [`Kernel::served_recv`] found on a connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServedRecv {
    /// Still connected (whatever was queued was appended).
    Open,
    /// Every guest fd on the client end is closed and nothing is left to read.
    Closed,
}

impl Kernel {
    /// Listen on `AF_UNIX` path `path` (absolute, in the guest's first mount
    /// namespace), making the socket's name there as `bind` would. Returns
    /// the listener's handle for [`Self::served_accept`].
    ///
    /// # Errors
    /// A negative errno: `EADDRINUSE` if `path` exists, or whatever creating
    /// a node there fails with.
    pub fn served_listen(&self, path: &str) -> Result<usize, i64> {
        let key = {
            let mut vfs = self.vfs.lock().unwrap();
            vfs.enter(HOST_NS, "/");
            if vfs.stat(path).is_some() {
                return Err(err(Errno::EADDRINUSE));
            }
            vfs.mknod(path, 0o140_777)
                .map_err(|e| -i64::from(e.raw_os_error().unwrap_or(Errno::EIO.0)))?;
            unix_key(0, path, Some(&vfs.identity(path)))
        };
        let mut net = self.net.lock().unwrap();
        let idx = net.socks.len();
        net.socks.push(Sock {
            netns: 0,
            unix_bound_key: Some(key.clone()),
            domain: super::AF_UNIX,
            kind: Kind::Listener {
                addr: Some(Addr::Unix(path.to_string())),
                backlog: VecDeque::new(),
                cred: Ucred::default(),
            },
            nonblock: true,
            opts: SockOpts::default(),
            refs: 1,
            edge_seq: 0,
            last_event: 0,
        });
        net.listeners.entry(key).or_default().push(idx);
        Ok(idx)
    }

    /// The next connection waiting on host listener `listener`, as a handle
    /// for [`Self::served_recv`]/[`Self::served_send`]/[`Self::served_close`].
    pub fn served_accept(&self, listener: usize) -> Option<usize> {
        let mut net = self.net.lock().unwrap();
        let Kind::Listener { backlog, .. } = &mut net.socks.get_mut(listener)?.kind else {
            return None;
        };
        backlog.pop_front()
    }

    /// Move everything queued toward nixvm on connection `conn` into `out`,
    /// and the descriptors that came with it into `fds`, in order.
    pub fn served_recv(
        &self,
        conn: usize,
        out: &mut Vec<u8>,
        fds: &mut Vec<ServedFd>,
    ) -> ServedRecv {
        let mut net = self.net.lock().unwrap();
        let Some(Kind::Pair(p)) = net.socks.get_mut(conn).map(|s| &mut s.kind) else {
            return ServedRecv::Closed;
        };
        let n = p.to[1].len();
        if n > 0 {
            out.extend(p.to[1].drain(..));
            fds.extend(p.scm_take(1, n).into_iter().map(ServedFd));
        }
        let closed = p.refs[0] == 0 || p.shut_wr[0];
        let p_empty = p.to[1].is_empty();
        if n > 0 {
            // The writer may have been waiting for room.
            net.note_event(conn);
        }
        if closed && p_empty {
            ServedRecv::Closed
        } else {
            ServedRecv::Open
        }
    }

    /// Queue `data` toward the guest client on connection `conn`, with `fds`
    /// riding its first byte. `false` (dropping `fds`) if the client is gone.
    pub fn served_send(&self, conn: usize, data: &[u8], fds: Vec<ServedFd>) -> bool {
        let mut net = self.net.lock().unwrap();
        let Some(Kind::Pair(p)) = net.socks.get_mut(conn).map(|s| &mut s.kind) else {
            drop(net);
            for f in fds {
                self.served_drop_fd(f);
            }
            return false;
        };
        if p.refs[0] == 0 || p.shut_rd[0] {
            drop(net);
            for f in fds {
                self.served_drop_fd(f);
            }
            return false;
        }
        if !fds.is_empty() {
            let pos = p.to[0].len();
            p.scm[0].push_back((pos, fds.into_iter().map(|f| f.0).collect()));
        }
        p.to[0].extend(data);
        net.bump_edge(conn);
        true
    }

    /// Close nixvm's end of connection `conn`: the client reads EOF.
    pub fn served_close(&self, conn: usize) {
        let mut net = self.net.lock().unwrap();
        // Whatever the client sent that nixvm never read goes with it.
        let dropped = match net.socks.get_mut(conn).map(|s| &mut s.kind) {
            Some(Kind::Pair(p)) => {
                let n = p.to[1].len();
                p.to[1].clear();
                p.scm_take(1, n)
            }
            _ => Vec::new(),
        };
        for fd in &dropped {
            self.scm_ref(&mut net, fd, false);
        }
        net.bump(&Fd::Socket { sock: conn, end: 1 }, false);
    }

    /// Release descriptor `fd` (what its holder's `close` would do).
    #[allow(clippy::needless_pass_by_value)] // consumes the reference it owns
    pub fn served_drop_fd(&self, fd: ServedFd) {
        let mut net = self.net.lock().unwrap();
        self.scm_ref(&mut net, &fd.0, false);
    }

    /// The size of the file `fd` is open on, if it is a regular file (or
    /// memfd) — what a shared-memory pool a client passed can be read up to.
    pub fn served_file_size(&self, fd: &ServedFd) -> Option<u64> {
        let Fd::File { path, .. } = &fd.0 else {
            return None;
        };
        let mut vfs = self.vfs.lock().unwrap();
        vfs.enter(HOST_NS, "/");
        vfs.stat(path).map(|a| a.size)
    }

    /// Read the file `fd` is open on at `off` into `buf`, as the guest's
    /// mappings of it see it (a shared mapping's stores included). Returns
    /// how many bytes were read; short at end of file, 0 if `fd` isn't a file.
    pub fn served_file_read(&self, fd: &ServedFd, off: u64, buf: &mut [u8]) -> usize {
        let Fd::File { path, .. } = &fd.0 else {
            return 0;
        };
        let mut vfs = self.vfs.lock().unwrap();
        vfs.enter(HOST_NS, "/");
        // A memfd's (and /dev/shm file's) contents are its pages.
        if let Some(id) = vfs.page_id(path)
            && let Some(obj) = self.shm.obj(id)
        {
            return obj.read(off, buf);
        }
        let mut got = 0;
        while got < buf.len() {
            match vfs.read_at(path, off + got as u64, &mut buf[got..]) {
                Ok(k) if k > 0 => got += k,
                _ => break,
            }
        }
        self.pc_after_read(&mut vfs, path, off, &mut buf[..got]);
        got
    }

    /// A read-only descriptor on the guest file at `path` (absolute, in the
    /// guest's first mount namespace), for sending to a client: a keymap
    /// nixvm wrote into the guest's filesystem, say.
    pub fn served_open_file(&self, path: &str) -> Option<ServedFd> {
        let mut vfs = self.vfs.lock().unwrap();
        vfs.enter(HOST_NS, "/");
        vfs.stat(path)?;
        let pin = self.pin_open(&mut vfs, path);
        Some(ServedFd(Fd::File {
            path: path.to_string(),
            offset: FileOffset::pinned(0, pin),
            readable: true,
            writable: false,
        }))
    }

    /// Write `data` to a new file at `path` (absolute, in the guest's first
    /// mount namespace; replaced if it exists), mode `mode`: what nixvm hands
    /// its clients by [`Self::served_open_file`], like a keyboard keymap.
    ///
    /// # Errors
    /// A negative errno from creating or writing the file.
    pub fn served_write_file(&self, path: &str, data: &[u8], mode: u32) -> Result<(), i64> {
        let errno = |e: std::io::Error| -i64::from(e.raw_os_error().unwrap_or(Errno::EIO.0));
        let mut vfs = self.vfs.lock().unwrap();
        vfs.enter(HOST_NS, "/");
        if vfs.stat(path).is_some() {
            vfs.unlink(path).map_err(errno)?;
        }
        vfs.create(path, mode).map_err(errno)?;
        let mut off = 0;
        while off < data.len() {
            match vfs.write_at(path, off as u64, &data[off..]) {
                Ok(0) => return Err(err(Errno::EIO)),
                Ok(k) => off += k,
                Err(e) => return Err(errno(e)),
            }
        }
        Ok(())
    }

    /// Bumps once per socket event in the VM: when it hasn't moved, no host
    /// connection has anything new to read.
    #[must_use]
    pub fn io_events(&self) -> u64 {
        self.io_events.load(std::sync::atomic::Ordering::Relaxed)
    }
}
