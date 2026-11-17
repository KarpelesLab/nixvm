//! The pieces of the embedding API: what a run is given besides its
//! configuration (stdio, extra filesystems, the network, a notice sink) and
//! what it returns ([`RunOutcome`]).

use std::fmt;
use std::io::{Read, Write};
use std::sync::{Arc, Mutex};

use crate::fs::MountFs;
use crate::kernel::notice::{Notice, NoticeSink, TimeLimit};

/// Makes a fresh filesystem for each run (see
/// [`SandboxBuilder::mount`](super::SandboxBuilder::mount)).
pub type FsFactory = Arc<dyn Fn() -> Box<dyn MountFs> + Send + Sync>;

/// Makes a fresh egress backend for each run (see [`Network::Egress`]).
pub type EgressFactory = Arc<dyn Fn() -> Box<dyn crate::kernel::egress::Egress> + Send + Sync>;

/// A filesystem mounted at a guest path by
/// [`SandboxBuilder::mount`](super::SandboxBuilder::mount).
#[derive(Clone)]
pub struct Mount {
    pub guest: String,
    pub fs: FsFactory,
}

impl fmt::Debug for Mount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mount")
            .field("guest", &self.guest)
            .finish_non_exhaustive()
    }
}

/// What the guest's sockets can reach beyond its own loopback.
#[derive(Clone, Default)]
pub enum Network {
    /// Loopback only: a connect to anywhere else fails with `ENETUNREACH`.
    #[default]
    None,
    /// The host's network, through host sockets
    /// ([`HostEgress`](crate::kernel::egress::HostEgress); not on wasm). The
    /// guest's `/etc/resolv.conf`, unless its root has one, names public
    /// resolvers.
    Host,
    /// A backend of the embedder's, made afresh for each run.
    Egress(EgressFactory),
}

impl Network {
    /// [`Network::Egress`] from a function making the backend.
    pub fn egress<F>(make: F) -> Self
    where
        F: Fn() -> Box<dyn crate::kernel::egress::Egress> + Send + Sync + 'static,
    {
        Self::Egress(Arc::new(make))
    }

    /// Whether the guest can reach anything beyond its loopback.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        !matches!(self, Self::None)
    }
}

impl fmt::Debug for Network {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::None => "None",
            Self::Host => "Host",
            Self::Egress(_) => "Egress(..)",
        })
    }
}

/// A writer shared between the builder's clones: every run writes to it.
pub(crate) type SharedWriter = Arc<Mutex<dyn Write + Send>>;
/// A reader shared between the builder's clones.
pub(crate) type SharedReader = Arc<Mutex<dyn Read + Send>>;

/// The guest's stdio and the notice sink, as the builder was given them.
#[derive(Clone, Default)]
pub(crate) struct Io {
    pub(crate) stdin: Option<SharedReader>,
    pub(crate) stdout: Option<SharedWriter>,
    pub(crate) stderr: Option<SharedWriter>,
    pub(crate) notices: Option<NoticeSink>,
}

impl fmt::Debug for Io {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Io")
            .field("stdin", &self.stdin.is_some())
            .field("stdout", &self.stdout.is_some())
            .field("stderr", &self.stderr.is_some())
            .field("notices", &self.notices.is_some())
            .finish()
    }
}

/// [`Write`] into a [`SharedWriter`].
pub(crate) struct ToShared(pub(crate) SharedWriter);

impl Write for ToShared {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .flush()
    }
}

/// [`Read`] from a [`SharedReader`].
pub(crate) struct FromShared(pub(crate) SharedReader);

impl Read for FromShared {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .read(buf)
    }
}

/// An in-memory buffer to capture a guest's output into: give
/// [`Capture::writer`] to [`SandboxBuilder::stdout`](super::SandboxBuilder::stdout)
/// (or `stderr`) and read what was written with [`Capture::contents`].
/// Clones share the buffer.
#[derive(Debug, Clone, Default)]
pub struct Capture(Arc<Mutex<Vec<u8>>>);

impl Capture {
    /// An empty buffer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A writer appending to this buffer.
    #[must_use]
    pub fn writer(&self) -> impl Write + Send + 'static {
        self.clone()
    }

    /// Everything written so far.
    #[must_use]
    pub fn contents(&self) -> Vec<u8> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Everything written so far, as text (invalid UTF-8 replaced).
    #[must_use]
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.contents()).into_owned()
    }
}

impl Write for Capture {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// A VM limit that ended the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LimitHit {
    /// [`Limits::wall_time`](crate::limits::Limits::wall_time) passed: every
    /// task was killed.
    WallTime,
    /// [`Limits::cpu_time`](crate::limits::Limits::cpu_time) ran out: every
    /// task was killed.
    CpuTime,
    /// The guest's RAM ran out and the out-of-memory killer killed the
    /// first process.
    Memory,
}

/// A system call (or a subcommand of one) the guest asked for that nixvm
/// doesn't implement, and how often.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Unsupported {
    /// A syscall number with no implementation (it got `ENOSYS`).
    Syscall { nr: u64, count: u64 },
    /// A subcommand no handler knows: an `ioctl` request, an `fcntl` or
    /// `prctl` command, a socket option (`level << 32 | name`), …
    Subcommand {
        syscall: &'static str,
        sub: u64,
        count: u64,
    },
}

impl fmt::Display for Unsupported {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Syscall { nr, count } => write!(f, "syscall {nr} ({count}x)"),
            Self::Subcommand {
                syscall,
                sub,
                count,
            } => write!(f, "{syscall} {sub:#x} ({count}x)"),
        }
    }
}

/// How a run went ([`Sandbox::execute`](super::Sandbox::execute)).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RunOutcome {
    /// The first process's exit code as a shell reports it in `$?`: its exit
    /// status, or `128 + signal` if a signal killed it.
    pub exit_code: i32,
    /// The signal that killed the first process, if one did.
    pub signal: Option<i32>,
    /// The VM limit that ended the run, if one did.
    pub limit: Option<LimitHit>,
    /// What the guest asked for that nixvm doesn't implement.
    pub unsupported: Vec<Unsupported>,
    /// What the kernel reported during the run (faults, limits, policy
    /// denials), in order. Traces are not kept here; a notice sink
    /// ([`SandboxBuilder::on_notice`](super::SandboxBuilder::on_notice)) sees
    /// them.
    pub notices: Vec<Notice>,
    /// The CPU backend that ran the guest (`kvm`, `hvf`, `interp`, …).
    pub backend: &'static str,
}

impl RunOutcome {
    /// Whether the first process exited with status 0.
    #[must_use]
    pub fn success(&self) -> bool {
        self.exit_code == 0 && self.signal.is_none()
    }

    /// Fill in [`RunOutcome::limit`] from the notices.
    pub(crate) fn limit_from_notices(notices: &[Notice], signal: Option<i32>) -> Option<LimitHit> {
        for n in notices {
            if let Notice::LimitReached(which) = n {
                return Some(match which {
                    TimeLimit::Wall => LimitHit::WallTime,
                    TimeLimit::Cpu => LimitHit::CpuTime,
                });
            }
        }
        let oom_pid1 = notices
            .iter()
            .any(|n| matches!(n, Notice::OutOfMemory { pid: 1, .. }));
        (oom_pid1 && signal == Some(9)).then_some(LimitHit::Memory)
    }
}
