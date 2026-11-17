//! What the kernel has to say about a run that isn't the guest's own output.
//!
//! A VM limit being reached, a guest process dying of a fault nobody handled,
//! the out-of-memory killer, a syscall the policy refused: Linux writes these
//! to its kernel log. nixvm hands each to the [`NoticeSink`] the embedder gave
//! the kernel ([`Kernel::set_notice_sink`](super::Kernel::set_notice_sink)), and
//! prints nothing itself — with no sink a notice is dropped, except
//! [`Notice::Trace`], which is only made when a debugging trace was asked
//! for (see [`crate::env`]) and then goes to stderr. Each notice's
//! [`Display`](std::fmt::Display) is the line the `nixvm` binary prints;
//! [`stderr_sink`] prints them that way.

use std::fmt;
use std::sync::Arc;

/// Receives every [`Notice`] a kernel makes, from whichever thread makes it.
pub type NoticeSink = Arc<dyn Fn(&Notice) + Send + Sync>;

/// A sink that prints each notice on its own line(s) of the host's stderr,
/// as the `nixvm` binaries do.
#[must_use]
pub fn stderr_sink() -> NoticeSink {
    Arc::new(|n: &Notice| eprintln!("{n}"))
}

/// Which VM-wide time ceiling ([`crate::limits::Limits`]) was reached.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeLimit {
    /// [`Limits::wall_time`](crate::limits::Limits::wall_time).
    Wall,
    /// [`Limits::cpu_time`](crate::limits::Limits::cpu_time).
    Cpu,
}

/// One thing the kernel reports. See the [module docs](self).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Notice {
    /// A VM-wide time ceiling passed; every guest task is being killed.
    LimitReached(TimeLimit),
    /// A page fault found no guest RAM left: the task was killed with
    /// `SIGKILL` (the out-of-memory killer).
    OutOfMemory {
        pid: i32,
        /// The task's `comm` (may be empty).
        comm: String,
        /// The address it faulted on.
        addr: u64,
        /// Guest RAM in use and free, in MiB, when it happened.
        used_mib: u64,
        free_mib: u64,
    },
    /// A guest task took a fault (memory, illegal instruction, …) it had no
    /// handler for, and dies of `signal`.
    Fault {
        pid: i32,
        signal: u32,
        /// What happened, e.g. `memory fault at 0x0 (write=false, pc=0x401000)`.
        what: String,
        /// Register, code and stack dump lines.
        context: Vec<String>,
    },
    /// The syscall policy refused a call and was asked to log it
    /// ([`crate::policy::SyscallPolicy::log_denials`]).
    Denied {
        pid: i32,
        /// The syscall's name.
        syscall: String,
        /// Its raw number.
        nr: u64,
        action: crate::policy::Action,
    },
    /// The configured syscall policy could not be read; every syscall is
    /// refused instead.
    BadPolicy(String),
    /// A debugging trace line (`NIXVM_TRACE`, `NIXVM_SCHEDTRACE`,
    /// `NIXVM_SIGTRACE`, …).
    Trace(String),
}

impl fmt::Display for Notice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LimitReached(which) => {
                let which = match which {
                    TimeLimit::Wall => "run time",
                    TimeLimit::Cpu => "CPU time",
                };
                write!(
                    f,
                    "nixvm: the VM's {which} limit was reached; killing the guest"
                )
            }
            Self::OutOfMemory {
                pid,
                comm,
                addr,
                used_mib,
                free_mib,
            } => {
                let comm = if comm.is_empty() {
                    String::new()
                } else {
                    format!(" ({comm})")
                };
                write!(
                    f,
                    "nixvm: out of memory: killed pid {pid}{comm}: no guest RAM left for its page \
                     at {addr:#x} ({used_mib} MiB in use, {free_mib} MiB free; --mem raises the \
                     limit)"
                )
            }
            Self::Fault {
                pid, what, context, ..
            } => {
                write!(f, "[fault] pid {pid} {what}")?;
                for line in context {
                    write!(f, "\n[fault]   {line}")?;
                }
                Ok(())
            }
            Self::Denied {
                pid,
                syscall,
                nr,
                action,
            } => write!(
                f,
                "nixvm: denied {syscall} (syscall {nr}) to pid {pid}: {action:?}"
            ),
            Self::BadPolicy(e) => write!(f, "nixvm: {e}; refusing every syscall"),
            Self::Trace(line) => f.write_str(line),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notices_read_as_the_binary_prints_them() {
        assert_eq!(
            Notice::LimitReached(TimeLimit::Wall).to_string(),
            "nixvm: the VM's run time limit was reached; killing the guest"
        );
        let fault = Notice::Fault {
            pid: 3,
            signal: 11,
            what: "general protection fault at 0x10".into(),
            context: vec!["regs: rax=0x0".into()],
        };
        assert_eq!(
            fault.to_string(),
            "[fault] pid 3 general protection fault at 0x10\n[fault]   regs: rax=0x0"
        );
    }
}
