//! The embedder's syscall policy: syscalls a guest may not make at all, whatever
//! it does inside the VM.
//!
//! A guest's own `seccomp` filters, capabilities and namespaces are the guest's
//! business; this is the host's, checked before any of them, for every task. A
//! rule names a syscall as Linux spells it (`mount`, `ptrace`, `rt_sigaction`),
//! or a group of them (`@mount`, `@network`, ...: see [`GROUPS`]), and says what
//! happens instead: the call fails with an errno (`EPERM` unless told
//! otherwise), or the process is killed.
//!
//! Every denial is counted and the most recent ones kept for
//! [`crate::kernel::Kernel::denials`], and with [`SyscallPolicy::log_denials`]
//! each is printed to stderr as it happens.
//!
//! From the environment, `NIXVM_DENY_SYSCALLS` takes the same rules as
//! [`SyscallPolicy::parse`] (`mount,@network:EACCES,ptrace:kill`) and
//! `NIXVM_LOG_DENIALS=1` turns logging on.

use crate::abi::Arch;
use crate::abi::arch::{Sysno, decode};
use crate::abi::errno::Errno;

/// What happens to a denied syscall.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// The call fails with this errno.
    Errno(Errno),
    /// The calling process is killed, as by `SIGKILL`.
    Kill,
}

/// Named groups of syscalls, for rules that mean a kind of thing.
pub const GROUPS: &[(&str, &[&str])] = &[
    (
        "@mount",
        &[
            "mount",
            "umount2",
            "pivot_root",
            "chroot",
            "fsopen",
            "fsmount",
            "fsconfig",
            "fspick",
            "move_mount",
            "open_tree",
            "mount_setattr",
        ],
    ),
    ("@namespaces", &["unshare", "setns"]),
    (
        "@network",
        &[
            "socket",
            "socketpair",
            "bind",
            "connect",
            "listen",
            "accept",
            "accept4",
            "sendto",
            "recvfrom",
            "sendmsg",
            "recvmsg",
            "sendmmsg",
            "recvmmsg",
        ],
    ),
    (
        "@debug",
        &[
            "ptrace",
            "process_vm_readv",
            "process_vm_writev",
            "kcmp",
            "pidfd_getfd",
            "perf_event_open",
        ],
    ),
    (
        "@clock",
        &["settimeofday", "clock_settime", "clock_adjtime", "adjtimex"],
    ),
    ("@keyring", &["add_key", "request_key", "keyctl"]),
];

/// The syscalls refused to every guest task, and how.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SyscallPolicy {
    /// `(syscall name, action)`, groups already expanded.
    rules: Vec<(String, Action)>,
    log: bool,
}

/// A rule's syscall name as it's compared: lowercase, no underscores, which is
/// how a Linux name lines up with [`Sysno`]'s (`rt_sigaction` and
/// `RtSigaction`).
fn key(name: &str) -> String {
    name.chars()
        .filter(|&c| c != '_')
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// The key of each syscall `arch` has, by raw number.
fn table(arch: Arch) -> Vec<(u64, Sysno, String)> {
    (0..1024u64)
        .filter_map(|nr| {
            let sys = decode(arch, nr);
            (!matches!(sys, Sysno::Unknown(_) | Sysno::NiSyscall))
                .then(|| (nr, sys, key(&format!("{sys:?}"))))
        })
        .collect()
}

/// Whether `name` is a syscall on either guest architecture.
fn is_syscall(name: &str) -> bool {
    let k = key(name);
    [Arch::X86_64, Arch::Aarch64]
        .iter()
        .any(|&a| table(a).iter().any(|(_, _, n)| *n == k))
}

impl SyscallPolicy {
    /// Nothing denied.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Refuse `name`, a syscall or a group, with `action`. `Err` names what
    /// isn't a syscall on either guest architecture, or an unknown group.
    pub fn deny(mut self, name: &str, action: Action) -> Result<Self, String> {
        let name = name.trim();
        if name.starts_with('@') {
            let Some((_, members)) = GROUPS.iter().find(|(g, _)| *g == name) else {
                return Err(format!("unknown syscall group {name:?}"));
            };
            for m in *members {
                self.rules.push(((*m).to_string(), action));
            }
            return Ok(self);
        }
        if !is_syscall(name) {
            return Err(format!("unknown syscall {name:?}"));
        }
        self.rules.push((name.to_string(), action));
        Ok(self)
    }

    /// Print each denial to stderr as it happens.
    #[must_use]
    pub fn log_denials(mut self, on: bool) -> Self {
        self.log = on;
        self
    }

    /// Whether denials are printed.
    #[must_use]
    pub fn logs(&self) -> bool {
        self.log
    }

    /// Whether nothing is denied.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Rules as `NIXVM_DENY_SYSCALLS` writes them: comma-separated syscalls or
    /// groups, each optionally followed by `:ERRNO` (a name like `EACCES`, or a
    /// number) or `:kill`. The default is `EPERM`.
    pub fn parse(spec: &str) -> Result<Self, String> {
        let mut policy = Self::new();
        for rule in spec.split(',').map(str::trim).filter(|r| !r.is_empty()) {
            let (name, action) = match rule.split_once(':') {
                None => (rule, Action::Errno(Errno::EPERM)),
                Some((name, how)) => (name, parse_action(how)?),
            };
            policy = policy.deny(name, action)?;
        }
        Ok(policy)
    }

    /// The policy `NIXVM_DENY_SYSCALLS` and `NIXVM_LOG_DENIALS` describe, if any.
    /// A rule that doesn't parse is an `Err`: a policy that silently allowed a
    /// misspelt syscall would be worse than none.
    pub fn from_env() -> Result<Self, String> {
        let log = std::env::var("NIXVM_LOG_DENIALS").is_ok_and(|v| !v.is_empty() && v != "0");
        let policy = match std::env::var("NIXVM_DENY_SYSCALLS") {
            Ok(spec) => Self::parse(&spec).map_err(|e| format!("NIXVM_DENY_SYSCALLS: {e}"))?,
            Err(_) => Self::new(),
        };
        Ok(policy.log_denials(log))
    }

    /// The action for each raw syscall number of `arch`, for the kernel to
    /// look up on every call.
    #[must_use]
    pub fn compile(&self, arch: Arch) -> CompiledPolicy {
        if self.rules.is_empty() {
            return CompiledPolicy {
                log: self.log,
                ..CompiledPolicy::default()
            };
        }
        let mut by_nr = vec![None; 1024];
        let mut names = vec![String::new(); 1024];
        for (nr, sys, k) in table(arch) {
            if let Some((_, action)) = self.rules.iter().rev().find(|(n, _)| key(n) == k) {
                by_nr[nr as usize] = Some(*action);
                names[nr as usize] = format!("{sys:?}");
            }
        }
        let active = by_nr.iter().any(Option::is_some);
        CompiledPolicy {
            by_nr,
            names,
            log: self.log,
            deny_all: false,
            active,
        }
    }

    /// What to enforce when the configured policy couldn't be read: every
    /// syscall kills, so a guest never runs with less confinement than asked.
    #[must_use]
    pub fn deny_everything() -> CompiledPolicy {
        CompiledPolicy {
            by_nr: Vec::new(),
            names: Vec::new(),
            log: true,
            deny_all: true,
            active: true,
        }
    }
}

fn parse_action(how: &str) -> Result<Action, String> {
    let how = how.trim();
    if how.eq_ignore_ascii_case("kill") {
        return Ok(Action::Kill);
    }
    if let Ok(n) = how.parse::<i32>() {
        return if (1..4096).contains(&n) {
            Ok(Action::Errno(Errno(n)))
        } else {
            Err(format!("errno {n} out of range"))
        };
    }
    Errno::from_name(&how.to_ascii_uppercase())
        .map(Action::Errno)
        .ok_or_else(|| format!("unknown errno {how:?}"))
}

/// A [`SyscallPolicy`] laid out by raw syscall number for one architecture.
#[derive(Debug, Clone, Default)]
pub struct CompiledPolicy {
    by_nr: Vec<Option<Action>>,
    names: Vec<String>,
    log: bool,
    deny_all: bool,
    /// Anything denied at all: checked on every syscall, so worked out once.
    active: bool,
}

impl CompiledPolicy {
    /// What happens to raw syscall `nr`, `None` to let it run, with its name.
    #[must_use]
    pub fn verdict(&self, nr: u64) -> Option<(Action, &str)> {
        if self.deny_all {
            return Some((Action::Kill, "any"));
        }
        let i = usize::try_from(nr).ok()?;
        let action = (*self.by_nr.get(i)?)?;
        Some((action, self.names[i].as_str()))
    }

    /// Whether anything is denied at all, so the common case skips the lookup.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.active
    }

    #[must_use]
    pub fn logs(&self) -> bool {
        self.log
    }
}

/// One refused syscall, for [`crate::kernel::Kernel::denials`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Denial {
    /// The process (thread group) that made it, by its id in the VM's first
    /// PID namespace.
    pub pid: i32,
    /// The syscall, as nixvm names it (`Mount`, `Ptrace`, ...).
    pub syscall: String,
    pub action: Action,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rules_parse_and_compile_by_name() {
        let p =
            SyscallPolicy::parse("mount, rt_sigaction:EACCES,ptrace:kill,@namespaces:13").unwrap();
        let x = p.compile(Arch::X86_64);
        assert_eq!(
            x.verdict(165).map(|v| v.0),
            Some(Action::Errno(Errno::EPERM))
        ); // mount
        assert_eq!(
            x.verdict(13).map(|v| v.0),
            Some(Action::Errno(Errno::EACCES))
        ); // rt_sigaction
        assert_eq!(x.verdict(101).map(|v| v.0), Some(Action::Kill)); // ptrace
        assert_eq!(x.verdict(272).map(|v| v.0), Some(Action::Errno(Errno(13)))); // unshare
        assert_eq!(x.verdict(0), None); // read
        assert!(x.is_active());
        // The same names on aarch64's numbers.
        let a = p.compile(Arch::Aarch64);
        assert_eq!(
            a.verdict(40).map(|v| v.0),
            Some(Action::Errno(Errno::EPERM))
        ); // mount
        assert_eq!(a.verdict(117).map(|v| v.0), Some(Action::Kill)); // ptrace
        assert_eq!(a.verdict(63), None); // read
    }

    #[test]
    fn mistakes_are_errors_not_silently_allowed() {
        assert!(SyscallPolicy::parse("mout").is_err());
        assert!(SyscallPolicy::parse("@nope").is_err());
        assert!(SyscallPolicy::parse("mount:EWHAT").is_err());
        assert!(SyscallPolicy::parse("mount:0").is_err());
        assert!(SyscallPolicy::parse("").unwrap().is_empty());
        assert!(!SyscallPolicy::new().compile(Arch::X86_64).is_active());
        let all = SyscallPolicy::deny_everything();
        assert_eq!(all.verdict(0).map(|v| v.0), Some(Action::Kill));
    }

    #[test]
    fn every_group_member_is_a_syscall() {
        for (group, members) in GROUPS {
            for m in *members {
                assert!(is_syscall(m), "{group}: {m}");
            }
        }
    }
}
