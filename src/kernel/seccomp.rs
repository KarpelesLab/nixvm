//! seccomp: `seccomp(2)` and `prctl(PR_SET_SECCOMP)`, *enforced*.
//!
//! Strict mode (`SECCOMP_SET_MODE_STRICT`) leaves a task only `read`,
//! `write`, `_exit` and `rt_sigreturn`; anything else kills the thread with
//! `SIGKILL`. Filter mode runs the installed classic-BPF programs over the
//! `struct seccomp_data` of every syscall (newest filter first, the most
//! restrictive verdict winning, as Linux combines them) and applies the
//! verdict: `ALLOW`/`LOG` run the call, `ERRNO(n)` fails it with `-n`
//! unexecuted, `TRAP` raises a forced `SIGSYS` (`si_code` `SYS_SECCOMP`),
//! `TRACE` finds no tracer and fails it `ENOSYS`, `KILL_THREAD` ends the
//! thread (the process, if it's the last one) and `KILL_PROCESS` the whole
//! process, with `SIGSYS`. Filters are inherited by `fork`/`clone` and kept
//! across `execve`, and installing one requires `no_new_privs` or
//! `CAP_SYS_ADMIN`. `SECCOMP_FILTER_FLAG_TSYNC` puts every thread of the
//! process under the caller's filters (failing with the first thread that
//! can't be, by ID or `ESRCH`), `LOG`/`SPEC_ALLOW` are accepted (there is no
//! audit log to write to, nor a speculation mitigation to skip).
//!
//! `USER_NOTIF` hands the syscall to a supervisor: a filter installed with
//! `SECCOMP_FILTER_FLAG_NEW_LISTENER` returns a listener descriptor (the
//! anonymous inode `seccomp notify`, close-on-exec) that can be passed to
//! another process over `SCM_RIGHTS`. A task whose syscall gets `USER_NOTIF`
//! queues a notification on its filter's listener and waits; the supervisor
//! polls the listener (`POLLIN` while a notification waits to be received,
//! `POLLOUT` while one is received but unanswered, `POLLHUP` once no task
//! uses the filter), takes notifications with `SECCOMP_IOCTL_NOTIF_RECV`,
//! checks one is still live with `NOTIF_ID_VALID`, may install descriptors in
//! the waiting task with `NOTIF_ADDFD` (at the lowest free number or a given
//! one, optionally as the reply itself), and answers with `NOTIF_SEND` — a
//! return value or error, or `SECCOMP_USER_NOTIF_FLAG_CONTINUE` to run the
//! syscall after all. As on Linux the waiting task is interruptible: a
//! handled signal drops its notification (later replies get `ENOENT`) and the
//! syscall is restarted (a fresh notification) or fails `EINTR`; with
//! `SECCOMP_FILTER_FLAG_WAIT_KILLABLE_RECV` only a fatal signal interrupts it
//! once the supervisor has received the notification. Closing the last
//! listener descriptor answers every outstanding notification `ENOSYS`, as do
//! later `USER_NOTIF` verdicts of the filter. The waiting is the kernel's
//! usual re-trap: the blocked syscall re-enters [`Kernel::seccomp_check`],
//! which finds its notification ([`SeccompState`]'s wait) rather than running
//! the filters again — Linux runs them once per syscall.
//!
//! Every filter is validated as Linux's `sk_chk_filter` + seccomp checks do
//! (known opcodes only, in-range forward jumps, aligned `seccomp_data`
//! loads, scratch-memory bounds, a final `RET`), so a malformed program is
//! `EINVAL` at install rather than misbehaving later.

use std::sync::{Arc, Mutex, Weak};

use super::{ExitCause, Fd, Kernel, ProcInfo, QueuedSig, RunState, ServiceCtx, Shared, err};
use crate::abi::Arch;
use crate::abi::errno::Errno;
use crate::vcpu::GuestMemory;

const SECCOMP_MODE_DISABLED: u8 = 0;
const SECCOMP_MODE_STRICT: u8 = 1;
const SECCOMP_MODE_FILTER: u8 = 2;

const SECCOMP_RET_KILL_PROCESS: u32 = 0x8000_0000;
const SECCOMP_RET_KILL_THREAD: u32 = 0x0000_0000;
const SECCOMP_RET_TRAP: u32 = 0x0003_0000;
const SECCOMP_RET_ERRNO: u32 = 0x0005_0000;
const SECCOMP_RET_USER_NOTIF: u32 = 0x7fc0_0000;
const SECCOMP_RET_TRACE: u32 = 0x7ff0_0000;
const SECCOMP_RET_LOG: u32 = 0x7ffc_0000;
const SECCOMP_RET_ALLOW: u32 = 0x7fff_0000;
const SECCOMP_RET_ACTION_FULL: u32 = 0xffff_0000;
const SECCOMP_RET_DATA: u32 = 0x0000_ffff;
/// `AUDIT_ARCH_*` values reported in `seccomp_data.arch`.
const AUDIT_ARCH_X86_64: u32 = 0xc000_003e;
const AUDIT_ARCH_AARCH64: u32 = 0xc000_00b7;
/// The longest program Linux accepts (`BPF_MAXINSNS`).
const BPF_MAXINSNS: usize = 4096;
/// The most instructions a syscall may run through across all of a task's
/// filters, each counting 4 extra (`MAX_INSNS_PER_PATH`).
const MAX_INSNS_PER_PATH: usize = (1 << 18) / 8;
/// `sizeof(struct seccomp_data)`.
const DATA_LEN: u32 = 64;

/// `SECCOMP_FILTER_FLAG_*`.
const FLAG_TSYNC: u64 = 1;
const FLAG_LOG: u64 = 2;
const FLAG_SPEC_ALLOW: u64 = 4;
const FLAG_NEW_LISTENER: u64 = 8;
const FLAG_TSYNC_ESRCH: u64 = 16;
const FLAG_WAIT_KILLABLE_RECV: u64 = 32;

/// `SECCOMP_USER_NOTIF_FLAG_CONTINUE`.
const USER_NOTIF_FLAG_CONTINUE: u32 = 1;
/// `SECCOMP_USER_NOTIF_FD_SYNC_WAKE_UP`.
const USER_NOTIF_FD_SYNC_WAKE_UP: u64 = 1;
/// `SECCOMP_ADDFD_FLAG_*`.
const ADDFD_FLAG_SETFD: u32 = 1;
const ADDFD_FLAG_SEND: u32 = 2;
/// `sizeof(struct seccomp_notif)`, `seccomp_notif_resp`, `seccomp_notif_addfd`.
const NOTIF_SIZE: usize = 80;
const RESP_SIZE: usize = 24;
const ADDFD_SIZE: usize = 24;

/// The listener ioctls (`'!'` type).
const IOCTL_NOTIF_RECV: u32 = 0xc050_2100;
const IOCTL_NOTIF_SEND: u32 = 0xc018_2101;
const IOCTL_NOTIF_ID_VALID: u32 = 0x4008_2102;
/// The `_IOR` spelling `NOTIF_ID_VALID` first shipped with, still accepted.
const IOCTL_NOTIF_ID_VALID_WRONG_DIR: u32 = 0x8008_2102;
const IOCTL_NOTIF_ADDFD: u32 = 0x4018_2103;
const IOCTL_NOTIF_SET_FLAGS: u32 = 0x4008_2104;

/// `O_CLOEXEC`, the one flag `seccomp_notif_addfd.newfd_flags` may carry.
const O_CLOEXEC: u32 = 0o2000000;

const SIGSYS: u64 = 31;
/// A notification not (or no longer) in the state an ioctl needs.
const EINPROGRESS: Errno = Errno(115);

/// One classic-BPF instruction (`struct sock_filter`).
#[derive(Clone, Copy, Debug)]
struct Insn {
    code: u16,
    jt: u8,
    jf: u8,
    k: u32,
}

/// One installed filter: its program, its install flags, and the listener a
/// `SECCOMP_FILTER_FLAG_NEW_LISTENER` install gave it. Shared (by `Arc`) by
/// every task it applies to, so the `Arc`'s count is Linux's `users`.
#[derive(Debug)]
pub(super) struct Filter {
    prog: Vec<Insn>,
    wait_killable: bool,
    listener: Option<Arc<Listener>>,
}

/// A task's seccomp state; cloned into children (the programs are shared,
/// what the task is in the middle of waiting for is not).
#[derive(Debug, Default)]
pub(super) struct SeccompState {
    mode: u8,
    /// Installed filters, oldest first.
    filters: Vec<Arc<Filter>>,
    /// How many filters were ever attached (`Seccomp_filters:`); kept when
    /// the task exits and drops them.
    count: u32,
    /// Bumped whenever another thread's `TSYNC` replaced the filters, so a
    /// copy of this state made before can tell.
    synced: u64,
    /// The notification the current syscall waits on.
    wait: Option<NotifWait>,
    /// The `NOTIF_ADDFD` the current ioctl waits on.
    addfd: Option<AddfdWait>,
    /// The current syscall was let through by a supervisor's `CONTINUE`; if
    /// it blocks, its re-traps are not filtered again.
    continued: bool,
}

impl Clone for SeccompState {
    fn clone(&self) -> Self {
        Self {
            mode: self.mode,
            filters: self.filters.clone(),
            count: self.count,
            synced: self.synced,
            wait: None,
            addfd: None,
            continued: false,
        }
    }
}

impl SeccompState {
    /// The mode `PR_GET_SECCOMP` reports (0/1/2).
    pub(super) fn mode(&self) -> u8 {
        self.mode
    }

    /// How many filters the task attached (`Seccomp_filters:`).
    pub(super) fn filter_count(&self) -> u32 {
        self.count
    }

    /// Whether any syscall needs checking.
    pub(super) fn active(&self) -> bool {
        self.mode != SECCOMP_MODE_DISABLED
    }

    /// [`Self::synced`].
    pub(super) fn synced(&self) -> u64 {
        self.synced
    }

    /// Take on another thread's filters (its `TSYNC`), keeping what this one is
    /// waiting on.
    fn adopt(&mut self, from: &Self) {
        self.mode = from.mode;
        self.filters.clone_from(&from.filters);
        self.count = from.count;
        self.synced = self.synced.wrapping_add(1);
    }

    /// Take on the filters another thread synchronized onto this one's
    /// stand-in while it ran (see [`Self::synced`]).
    pub(super) fn adopt_synced(&mut self, from: &Self) {
        self.adopt(from);
        self.synced = from.synced;
    }

    /// The current syscall waits on a notification its supervisor has
    /// received, under a `WAIT_KILLABLE_RECV` filter: only a fatal signal may
    /// interrupt it.
    pub(super) fn killable_wait(&self) -> bool {
        self.wait.as_ref().is_some_and(|w| w.killable_now)
    }

    /// The syscall is done (it returned rather than blocking again).
    pub(super) fn syscall_done(&mut self) {
        self.continued = false;
    }
}

/// What a task waiting on a notification remembers of it.
#[derive(Debug)]
struct NotifWait {
    listener: Arc<Listener>,
    id: u64,
    /// The filter is `WAIT_KILLABLE_RECV`.
    killable: bool,
    /// ... and the notification has been received.
    killable_now: bool,
}

/// What a supervisor waiting on an `ADDFD` remembers of it.
#[derive(Debug)]
struct AddfdWait {
    listener: Arc<Listener>,
    key: u64,
}

/// A filter's notification queue: what the listener descriptor reads.
#[derive(Debug, Default)]
pub(super) struct Listener {
    st: Mutex<NotifState>,
}

#[derive(Debug, Default)]
struct NotifState {
    /// The filter, to tell whether any task still uses it.
    filter: Weak<Filter>,
    next_id: u64,
    notifs: Vec<Notif>,
    /// The last listener descriptor was closed.
    detached: bool,
    /// `NOTIF_SET_FLAGS`.
    flags: u64,
    next_key: u64,
    /// What the `ADDFD`s of notifications that are gone came to, by key, for
    /// their supervisors to collect.
    done: Vec<(u64, i64)>,
}

impl NotifState {
    /// Notification `id` is over (answered, or its task stopped waiting): it
    /// goes, its `ADDFD`s' results (`ESRCH` for those never handled) kept for
    /// their supervisors.
    fn retire(&mut self, id: u64) -> Option<Notif> {
        let i = self.notifs.iter().position(|n| n.id == id)?;
        let mut n = self.notifs.remove(i);
        for a in n.addfd.drain(..) {
            self.done.push((a.key, a.ret.unwrap_or(err(Errno::ESRCH))));
        }
        Some(n)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NState {
    Init,
    Sent,
    Replied,
}

/// One notification (`struct seccomp_knotif`).
#[derive(Debug)]
struct Notif {
    id: u64,
    /// The waiting task, by global ID.
    tid: i32,
    data: [u8; 64],
    state: NState,
    val: i64,
    error: i32,
    flags: u32,
    /// `ADDFD` requests queued for the task, oldest first.
    addfd: Vec<AddFd>,
}

/// One `NOTIF_ADDFD` request.
#[derive(Debug)]
struct AddFd {
    key: u64,
    /// The descriptor to install (taken by the task as it handles it).
    file: Option<Fd>,
    setfd: bool,
    newfd: u32,
    cloexec: bool,
    send: bool,
    /// What the ioctl returns, once the task has handled it.
    ret: Option<i64>,
}

impl Listener {
    /// `poll` readiness of the listener descriptor.
    pub(super) fn poll(&self) -> u32 {
        const POLLIN: u32 = 0x1;
        const POLLOUT: u32 = 0x4;
        const POLLHUP: u32 = 0x10;
        const POLLRDNORM: u32 = 0x40;
        const POLLWRNORM: u32 = 0x100;
        let st = self.st.lock().unwrap();
        let mut r = 0;
        for n in &st.notifs {
            match n.state {
                NState::Init => r |= POLLIN | POLLRDNORM,
                NState::Sent => r |= POLLOUT | POLLWRNORM,
                NState::Replied => {}
            }
        }
        if st.filter.strong_count() == 0 {
            r |= POLLHUP;
        }
        r
    }
}

/// The listener descriptor's open file (`anon_inode:seccomp notify`): shared by
/// its `dup`s, `fork` copies and `SCM_RIGHTS` copies; the last one going
/// detaches the listener.
#[derive(Debug, Clone)]
pub struct NotifFile(Arc<NotifFileInner>);

#[derive(Debug)]
struct NotifFileInner(Arc<Listener>);

impl NotifFile {
    /// A stable identity for the open file.
    #[must_use]
    pub fn id(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }

    pub(super) fn listener(&self) -> &Arc<Listener> {
        &self.0.0
    }
}

impl Drop for NotifFileInner {
    /// `seccomp_notify_detach`: every notification not yet answered is
    /// answered `ENOSYS`, and the filter's later `USER_NOTIF`s fail so.
    fn drop(&mut self) {
        let mut st = self.0.st.lock().unwrap();
        st.detached = true;
        for n in &mut st.notifs {
            if n.state != NState::Replied {
                n.state = NState::Replied;
                n.error = -Errno::ENOSYS.0;
                n.val = 0;
                n.flags = 0;
            }
        }
        // The waiting tasks find their answer when they next re-check.
    }
}

/// `SECCOMP_LOG_*`: the actions `actions_logged` can name, in its order.
const LOG_NAMES: [&str; 8] = [
    "kill_process",
    "kill_thread",
    "trap",
    "errno",
    "user_notif",
    "trace",
    "log",
    "allow",
];
/// The bit of each [`LOG_NAMES`] entry.
const LOG_BITS: [u32; 8] = [1, 2, 4, 8, 1 << 7, 1 << 4, 1 << 5, 1 << 6];
/// `actions_logged` at boot: everything but `allow`.
pub(super) const DEFAULT_LOGGED: u32 = 0xbf;

/// `actions_logged`'s text for `mask`.
pub(super) fn logged_names(mask: u32) -> String {
    let names: Vec<&str> = LOG_NAMES
        .iter()
        .zip(LOG_BITS)
        .filter(|&(_, b)| mask & b != 0)
        .map(|(n, _)| *n)
        .collect();
    names.join(" ")
}

/// Drop what a task is waiting on (a notification, an `ADDFD`), as when its
/// wait is interrupted or it exits: its notification disappears (a reply
/// finds `ENOENT`), its queued `ADDFD`s fail `ESRCH`.
pub(super) fn abandon(st: &mut SeccompState) {
    if let Some(w) = st.wait.take() {
        w.listener.st.lock().unwrap().retire(w.id);
    }
    if let Some(w) = st.addfd.take() {
        let mut ls = w.listener.st.lock().unwrap();
        for n in &mut ls.notifs {
            n.addfd.retain(|a| a.key != w.key);
        }
        ls.done.retain(|&(k, _)| k != w.key);
    }
    st.continued = false;
}

/// A task is exiting (`seccomp_filter_release`): it stops using its filters
/// (a listener sees `POLLHUP` once nobody does) and stops waiting.
pub(super) fn release(info: &mut ProcInfo) {
    abandon(&mut info.seccomp);
    info.seccomp.filters.clear();
}

/// Validate a program the way Linux does before accepting it.
fn validate(prog: &[Insn]) -> bool {
    if prog.is_empty() || prog.len() > BPF_MAXINSNS {
        return false;
    }
    for (pc, i) in prog.iter().enumerate() {
        let ok = match i.code {
            // Always valid: ld/ldx imm and len; alu add sub mul or and lsh
            // rsh xor (K and X), div/mod by X, neg; ret k / ret a; tax / txa.
            0x00 | 0x01 | 0x80 | 0x81 | 0x04 | 0x14 | 0x24 | 0x44 | 0x54 | 0x64 | 0x74 | 0xa4
            | 0x0c | 0x1c | 0x2c | 0x3c | 0x4c | 0x5c | 0x6c | 0x7c | 0x9c | 0xac | 0x84 | 0x06
            | 0x16 | 0x07 | 0x87 => true,
            // An absolute 32-bit load must be aligned and inside seccomp_data.
            0x20 => i.k.is_multiple_of(4) && i.k < DATA_LEN,
            // Scratch memory has 16 slots.
            0x60 | 0x61 | 0x02 | 0x03 => i.k < 16,
            0x34 | 0x94 => i.k != 0, // div/mod by a constant 0
            // ja: an unconditional forward jump by k.
            0x05 => (pc + 1)
                .checked_add(i.k as usize)
                .is_some_and(|t| t < prog.len()),
            // jeq/jgt/jge/jset with K or X: both targets in range.
            0x15 | 0x25 | 0x35 | 0x45 | 0x1d | 0x2d | 0x3d | 0x4d => {
                pc + 1 + (i.jt as usize) < prog.len() && pc + 1 + (i.jf as usize) < prog.len()
            }
            _ => false,
        };
        if !ok {
            return false;
        }
    }
    matches!(prog.last().map(|i| i.code), Some(0x06 | 0x16))
}

/// Run one filter over `data` (the 64-byte `struct seccomp_data`).
fn run(prog: &[Insn], data: &[u8; 64]) -> u32 {
    let (mut a, mut x) = (0u32, 0u32);
    let mut m = [0u32; 16];
    let mut pc = 0usize;
    let word = |k: u32| u32::from_le_bytes(data[k as usize..k as usize + 4].try_into().unwrap());
    while let Some(i) = prog.get(pc) {
        pc += 1;
        let src = if i.code & 0x08 != 0 { x } else { i.k };
        match i.code {
            0x00 => a = i.k,
            0x01 => x = i.k,
            0x20 => a = word(i.k),
            0x80 => a = DATA_LEN,
            0x81 => x = DATA_LEN,
            0x60 => a = m[i.k as usize],
            0x61 => x = m[i.k as usize],
            0x02 => m[i.k as usize] = a,
            0x03 => m[i.k as usize] = x,
            0x04 | 0x0c => a = a.wrapping_add(src),
            0x14 | 0x1c => a = a.wrapping_sub(src),
            0x24 | 0x2c => a = a.wrapping_mul(src),
            0x34 | 0x3c => a = a.checked_div(src).unwrap_or(0),
            0x94 | 0x9c => a = a.checked_rem(src).unwrap_or(0),
            0x44 | 0x4c => a |= src,
            0x54 | 0x5c => a &= src,
            0x64 | 0x6c => a = a.checked_shl(src).unwrap_or(0),
            0x74 | 0x7c => a = a.checked_shr(src).unwrap_or(0),
            0xa4 | 0xac => a ^= src,
            0x84 => a = a.wrapping_neg(),
            0x05 => pc += i.k as usize,
            0x15 | 0x1d | 0x25 | 0x2d | 0x35 | 0x3d | 0x45 | 0x4d => {
                let hit = match i.code & 0xf0 {
                    0x10 => a == src,
                    0x20 => a > src,
                    0x30 => a >= src,
                    _ => a & src != 0,
                };
                pc += usize::from(if hit { i.jt } else { i.jf });
            }
            0x06 => return i.k,
            0x16 => return a,
            0x07 => x = a,
            0x87 => a = x,
            _ => return SECCOMP_RET_KILL_THREAD, // unreachable after validate()
        }
    }
    SECCOMP_RET_KILL_THREAD
}

/// Whether `ancestor`'s filters are where `chain` started from (Linux's
/// `is_ancestor`: no filters is everyone's ancestor).
fn is_ancestor(ancestor: &[Arc<Filter>], chain: &[Arc<Filter>]) -> bool {
    ancestor.len() <= chain.len() && ancestor.iter().zip(chain).all(|(a, b)| Arc::ptr_eq(a, b))
}

/// Read a `struct sock_fprog { u16 len; u64 filter; }`'s program, checking
/// what Linux checks in its order: the header can be read, the length is in
/// range, the task may install a filter at all, the program can be read and
/// is valid.
fn read_fprog(cx: &ServiceCtx, mem: &GuestMemory, ptr: u64) -> Result<Vec<Insn>, i64> {
    let (Ok(len), Ok(filter)) = (mem.read_vec(ptr, 2), mem.read_u64(ptr + 8)) else {
        return Err(err(Errno::EFAULT));
    };
    let len = usize::from(u16::from_le_bytes([len[0], len[1]]));
    if len == 0 || len > BPF_MAXINSNS {
        return Err(err(Errno::EINVAL));
    }
    if !cx.cur.no_new_privs && !cx.cur.caps.has(super::caps::CAP_SYS_ADMIN) {
        return Err(err(Errno::EACCES));
    }
    let Ok(raw) = mem.read_vec(filter, len * 8) else {
        return Err(err(Errno::EFAULT));
    };
    let prog: Vec<Insn> = raw
        .chunks(8)
        .map(|c| Insn {
            code: u16::from_le_bytes([c[0], c[1]]),
            jt: c[2],
            jf: c[3],
            k: u32::from_le_bytes([c[4], c[5], c[6], c[7]]),
        })
        .collect();
    if validate(&prog) {
        Ok(prog)
    } else {
        Err(err(Errno::EINVAL))
    }
}

impl Kernel {
    /// Enter a seccomp mode: the shared core of `seccomp(SET_MODE_*)` and
    /// `prctl(PR_SET_SECCOMP)`. `sh` (the task table) is needed only for
    /// `SECCOMP_FILTER_FLAG_TSYNC`, which only `seccomp(2)` can pass.
    pub(super) fn seccomp_set_mode(
        &self,
        sh: Option<&mut Shared>,
        cx: &mut ServiceCtx,
        mode: u64,
        args: u64,
        flags: u64,
        mem: &GuestMemory,
    ) -> i64 {
        match mode {
            1 => {
                // SECCOMP_MODE_STRICT: no flags, no args; can't leave filter mode.
                if flags != 0 || args != 0 {
                    return err(Errno::EINVAL);
                }
                let st = &mut cx.cur.seccomp;
                if st.mode == SECCOMP_MODE_FILTER {
                    return err(Errno::EINVAL);
                }
                st.mode = SECCOMP_MODE_STRICT;
                0
            }
            2 => self.seccomp_set_filter(sh, cx, args, flags, mem),
            _ => err(Errno::EINVAL),
        }
    }

    /// `seccomp_set_mode_filter`.
    fn seccomp_set_filter(
        &self,
        sh: Option<&mut Shared>,
        cx: &mut ServiceCtx,
        args: u64,
        flags: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let known = FLAG_TSYNC
            | FLAG_LOG
            | FLAG_SPEC_ALLOW
            | FLAG_NEW_LISTENER
            | FLAG_TSYNC_ESRCH
            | FLAG_WAIT_KILLABLE_RECV;
        if flags & !known != 0 {
            return err(Errno::EINVAL);
        }
        // TSYNC's failure (a thread ID) and NEW_LISTENER's success (an fd)
        // would be indistinguishable, unless TSYNC fails with ESRCH instead.
        if flags & FLAG_TSYNC != 0
            && flags & FLAG_NEW_LISTENER != 0
            && flags & FLAG_TSYNC_ESRCH == 0
        {
            return err(Errno::EINVAL);
        }
        if flags & FLAG_WAIT_KILLABLE_RECV != 0 && flags & FLAG_NEW_LISTENER == 0 {
            return err(Errno::EINVAL);
        }
        let prog = match read_fprog(cx, mem, args) {
            Ok(p) => p,
            Err(e) => return e,
        };
        let new_listener = flags & FLAG_NEW_LISTENER != 0;
        if new_listener {
            let soft = cx.cur.rlimits.soft(super::sys_misc::RLIMIT_NOFILE);
            if !cx.cur.fds.has_room_for(1, soft) {
                return err(Errno::EMFILE);
            }
        }
        let st = &cx.cur.seccomp;
        if st.mode == SECCOMP_MODE_STRICT {
            return err(Errno::EINVAL);
        }
        // Only one listener per filter chain.
        if new_listener && st.filters.iter().any(|f| f.listener.is_some()) {
            return err(Errno::EBUSY);
        }
        let total = st
            .filters
            .iter()
            .fold(prog.len(), |n, f| n + f.prog.len() + 4);
        if total > MAX_INSNS_PER_PATH {
            return err(Errno::ENOMEM);
        }
        let listener = new_listener.then(|| Arc::new(Listener::default()));
        let filter = Arc::new(Filter {
            prog,
            wait_killable: flags & FLAG_WAIT_KILLABLE_RECV != 0,
            listener: listener.clone(),
        });
        if flags & FLAG_TSYNC != 0 {
            let Some(sh) = sh else {
                return err(Errno::EINVAL);
            };
            if let Some(failed) = self.seccomp_tsync_blocker(sh, cx) {
                return if flags & FLAG_TSYNC_ESRCH != 0 {
                    err(Errno::ESRCH)
                } else {
                    failed
                };
            }
            Self::seccomp_attach(cx, filter);
            // Every other live thread takes on the caller's filters (and its
            // no_new_privs, so a thread can't be used to shed it).
            let (tgid, me, nnp) = (cx.cur.tgid, cx.cur.pid, cx.cur.no_new_privs);
            for p in sh.procs.iter_mut().flatten() {
                let i = &mut p.info;
                if i.tgid == tgid && i.pid != me && !matches!(i.run, RunState::Zombie(_)) {
                    i.seccomp.adopt(&cx.cur.seccomp);
                    i.no_new_privs |= nnp;
                }
            }
        } else {
            Self::seccomp_attach(cx, filter);
        }
        let Some(listener) = listener else {
            return 0;
        };
        {
            let mut ls = listener.st.lock().unwrap();
            ls.filter = Arc::downgrade(cx.cur.seccomp.filters.last().unwrap());
            // Linux starts each listener's IDs at a random point.
            ls.next_id = crate::entropy::u64();
        }
        let file = NotifFile(Arc::new(NotifFileInner(listener)));
        let fd = cx.cur.fds.alloc(Fd::SeccompNotify(file));
        cx.cur.fds.set_cloexec(fd, true);
        i64::from(fd)
    }

    /// Attach `filter` to the current task.
    fn seccomp_attach(cx: &mut ServiceCtx, filter: Arc<Filter>) {
        let st = &mut cx.cur.seccomp;
        st.mode = SECCOMP_MODE_FILTER;
        st.filters.push(filter);
        st.count = st.count.saturating_add(1);
    }

    /// `seccomp_can_sync_threads`: the first other live thread of the process
    /// that `TSYNC` can't move to the caller's filters (it's in strict mode,
    /// or has filters the caller's didn't grow from), as the caller numbers
    /// it — `-ESRCH` if it can't see it.
    fn seccomp_tsync_blocker(&self, sh: &Shared, cx: &ServiceCtx) -> Option<i64> {
        let mine = &cx.cur.seccomp;
        let failed = sh.procs.iter().flatten().map(|p| &p.info).find(|i| {
            i.tgid == cx.cur.tgid
                && i.pid != cx.cur.pid
                && !matches!(i.run, RunState::Zombie(_))
                && !(i.seccomp.mode == SECCOMP_MODE_DISABLED
                    || (i.seccomp.mode == SECCOMP_MODE_FILTER
                        && is_ancestor(&i.seccomp.filters, &mine.filters)))
        })?;
        let pid = self.visible_pid(cx, failed.pid);
        Some(if pid == 0 {
            err(Errno::ESRCH)
        } else {
            i64::from(pid)
        })
    }

    /// A write to `/proc/sys/kernel/seccomp/actions_logged`: the
    /// space-separated actions to log (`allow` may not be one), root only.
    pub(super) fn write_seccomp_logged(
        &self,
        cx: &ServiceCtx,
        buf: u64,
        count: u64,
        mem: &GuestMemory,
    ) -> i64 {
        if !Self::capable(cx, super::caps::CAP_SYS_ADMIN) {
            return err(Errno::EPERM);
        }
        // proc_dostring takes up to the buffer's size (the longest list).
        let Ok(bytes) = mem.read_vec(buf, count.min(64) as usize) else {
            return err(Errno::EFAULT);
        };
        let text = String::from_utf8_lossy(&bytes);
        let text = text.split(['\n', '\0']).next().unwrap_or("");
        let mut mask = 0;
        for name in text.split(' ') {
            if name.is_empty() {
                break;
            }
            match LOG_NAMES.iter().position(|n| *n == name) {
                Some(i) => mask |= LOG_BITS[i],
                None => return err(Errno::EINVAL),
            }
        }
        if mask & (1 << 6) != 0 {
            return err(Errno::EINVAL);
        }
        self.seccomp_logged
            .store(mask, std::sync::atomic::Ordering::Relaxed);
        count as i64
    }

    /// `seccomp(operation, flags, args)`.
    pub(super) fn sys_seccomp(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        op: u64,
        flags: u64,
        args: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const SET_MODE_STRICT: u64 = 0;
        const SET_MODE_FILTER: u64 = 1;
        const GET_ACTION_AVAIL: u64 = 2;
        const GET_NOTIF_SIZES: u64 = 3;
        // The operation and flags are `unsigned int`.
        let (op, flags) = (op & 0xffff_ffff, flags & 0xffff_ffff);
        match op {
            SET_MODE_STRICT => self.seccomp_set_mode(None, cx, 1, args, flags, mem),
            SET_MODE_FILTER => self.seccomp_set_mode(Some(sh), cx, 2, args, flags, mem),
            GET_ACTION_AVAIL => {
                if flags != 0 {
                    return err(Errno::EINVAL);
                }
                match mem.read_u32(args) {
                    Ok(
                        SECCOMP_RET_KILL_PROCESS
                        | SECCOMP_RET_KILL_THREAD
                        | SECCOMP_RET_TRAP
                        | SECCOMP_RET_ERRNO
                        | SECCOMP_RET_USER_NOTIF
                        | SECCOMP_RET_TRACE
                        | SECCOMP_RET_LOG
                        | SECCOMP_RET_ALLOW,
                    ) => 0,
                    Ok(_) => err(Errno::EOPNOTSUPP),
                    Err(_) => err(Errno::EFAULT),
                }
            }
            GET_NOTIF_SIZES => {
                if flags != 0 {
                    return err(Errno::EINVAL);
                }
                // struct seccomp_notif_sizes { u16 notif; u16 resp; u16 data }.
                let b = [NOTIF_SIZE as u8, 0, RESP_SIZE as u8, 0, DATA_LEN as u8, 0];
                if mem.write(args, &b).is_err() {
                    return err(Errno::EFAULT);
                }
                0
            }
            _ => err(Errno::EINVAL),
        }
    }

    /// Check syscall `raw` (with `args`, issued at `pc`) against the task's
    /// seccomp state. `None` lets it run; `Some(ret)` is the result to return
    /// without running it (the task may also have been killed, or be waiting
    /// for its supervisor's answer: `cx.block`).
    pub(super) fn seccomp_check(
        &self,
        cx: &mut ServiceCtx,
        sys: crate::abi::arch::Sysno,
        raw: u64,
        args: &[u64; 6],
        pc: u64,
        mem: &mut GuestMemory,
    ) -> Option<i64> {
        use crate::abi::arch::Sysno;
        const SIGKILL: i32 = 9;
        const SYS_SECCOMP: i32 = 1;
        if cx.cur.seccomp.mode == SECCOMP_MODE_STRICT {
            if matches!(
                sys,
                Sysno::Read | Sysno::Write | Sysno::Exit | Sysno::RtSigreturn
            ) {
                return None;
            }
            // `do_exit(SIGKILL)`: the thread, not its siblings.
            self.seccomp_kill_thread(cx, SIGKILL, false, mem);
            return Some(0);
        }
        // The re-trap of a syscall its supervisor let through, which blocked.
        if cx.cur.seccomp.continued {
            return None;
        }
        // The re-trap of a syscall waiting for its supervisor.
        if let Some(w) = cx.cur.seccomp.wait.take() {
            return self.seccomp_notif_wait(cx, w);
        }
        let arch = match self.arch {
            Arch::X86_64 => AUDIT_ARCH_X86_64,
            Arch::Aarch64 => AUDIT_ARCH_AARCH64,
        };
        let mut data = [0u8; 64];
        data[0..4].copy_from_slice(&(raw as u32).to_le_bytes());
        data[4..8].copy_from_slice(&arch.to_le_bytes());
        data[8..16].copy_from_slice(&pc.to_le_bytes());
        for (i, a) in args.iter().enumerate() {
            data[16 + i * 8..24 + i * 8].copy_from_slice(&a.to_le_bytes());
        }
        // Newest filter first; the most restrictive action wins (Linux
        // compares the action part as a signed value: KILL_PROCESS lowest),
        // the newest filter returning it on a tie.
        let (verdict, matched) = cx
            .cur
            .seccomp
            .filters
            .iter()
            .rev()
            .map(|f| (run(&f.prog, &data), Some(f)))
            .min_by_key(|(r, _)| (r & SECCOMP_RET_ACTION_FULL) as i32)
            .unwrap_or((SECCOMP_RET_ALLOW, None));
        let value = verdict & SECCOMP_RET_DATA;
        match verdict & SECCOMP_RET_ACTION_FULL {
            SECCOMP_RET_ALLOW | SECCOMP_RET_LOG => None,
            SECCOMP_RET_ERRNO => Some(-i64::from(value.min(4095))),
            SECCOMP_RET_TRAP => {
                // A forced SIGSYS with si_code SYS_SECCOMP; the _sigsys fields
                // (call address, syscall, arch) ride in the pid/uid/value
                // slots. Blocked or ignored, it's unblocked and defaulted,
                // so it kills.
                let info = QueuedSig {
                    code: SYS_SECCOMP,
                    pid: pc as u32 as i32,
                    uid: (pc >> 32) as u32,
                    value: u64::from(raw as u32) | u64::from(arch) << 32,
                };
                let bit = 1u64 << (SIGSYS - 1);
                if cx.cur.blocked & bit != 0 || cx.cur.handlers.get(SIGSYS).handler == 1 {
                    cx.cur.blocked &= !bit;
                    cx.cur.handlers.set(SIGSYS, super::SigAction::default());
                }
                cx.cur.pending |= bit;
                cx.cur.post_siginfo(SIGSYS, info);
                // `syscall_rollback`: the handler sees, and returns to, the
                // registers the syscall was made with.
                Some(match self.arch {
                    Arch::X86_64 => raw as i64,
                    Arch::Aarch64 => args[0] as i64,
                })
            }
            SECCOMP_RET_USER_NOTIF => {
                let filter = Arc::clone(matched.expect("a filter matched"));
                Some(self.seccomp_notif_start(cx, &filter, data))
            }
            SECCOMP_RET_TRACE => Self::seccomp_ret_trace(cx, value),
            SECCOMP_RET_KILL_THREAD => {
                self.seccomp_kill_thread(cx, SIGSYS as i32, true, mem);
                Some(0)
            }
            // KILL_PROCESS (and anything unknown, which Linux also treats so).
            _ => {
                let mut sh = self.shared.lock().unwrap();
                self.exit_group_with(&mut sh, cx, ExitCause::Signaled(SIGSYS as i32), mem);
                Some(0)
            }
        }
    }

    /// End the current thread with `sig` (`do_exit`), or the whole process if
    /// `group_if_last` and no other thread of it is alive (seccomp's
    /// `KILL_THREAD` on the last thread is a process kill with a core dump).
    fn seccomp_kill_thread(
        &self,
        cx: &mut ServiceCtx,
        sig: i32,
        group_if_last: bool,
        mem: &mut GuestMemory,
    ) {
        let mut sh = self.shared.lock().unwrap();
        let (tgid, me) = (cx.cur.tgid, cx.cur.pid);
        let others = sh.procs.iter().flatten().any(|p| {
            p.info.tgid == tgid && p.info.pid != me && !matches!(p.info.run, RunState::Zombie(_))
        });
        if others || !group_if_last {
            self.exit_task(&mut sh, cx, ExitCause::Signaled(sig), mem);
        } else {
            self.exit_group_with(&mut sh, cx, ExitCause::Signaled(sig), mem);
        }
    }

    /// Wake whoever may be waiting on a listener's change (a supervisor in
    /// `NOTIF_RECV` or `poll`, a task waiting on its answer).
    fn seccomp_wake(&self) {
        self.shared.lock().unwrap().unpark_all();
    }

    /// A `USER_NOTIF` verdict from `filter`: queue a notification for its
    /// supervisor and wait for the answer — or fail `ENOSYS` if it has no
    /// listener (any more).
    fn seccomp_notif_start(
        &self,
        cx: &mut ServiceCtx,
        filter: &Arc<Filter>,
        data: [u8; 64],
    ) -> i64 {
        let Some(listener) = filter.listener.clone() else {
            return err(Errno::ENOSYS);
        };
        let id = {
            let mut ls = listener.st.lock().unwrap();
            if ls.detached {
                return err(Errno::ENOSYS);
            }
            let id = ls.next_id;
            ls.next_id = ls.next_id.wrapping_add(1);
            ls.notifs.push(Notif {
                id,
                tid: cx.cur.pid,
                data,
                state: NState::Init,
                val: 0,
                error: 0,
                flags: 0,
                addfd: Vec::new(),
            });
            id
        };
        cx.cur.seccomp.wait = Some(NotifWait {
            listener,
            id,
            killable: filter.wait_killable,
            killable_now: false,
        });
        self.seccomp_wake();
        cx.block = true;
        0
    }

    /// The waiting syscall re-checks its notification: handle the `ADDFD`s
    /// queued for it, then return the answer, or wait on.
    fn seccomp_notif_wait(&self, cx: &mut ServiceCtx, mut w: NotifWait) -> Option<i64> {
        // Take the queued ADDFDs out (installing touches the pipe/socket
        // tables, which mustn't be locked inside a listener's lock).
        let pending: Vec<(u64, Fd, bool, u32, bool)> = {
            let mut ls = w.listener.st.lock().unwrap();
            let Some(n) = ls.notifs.iter_mut().find(|n| n.id == w.id) else {
                // Gone without an answer: can't happen, but don't hang.
                return Some(err(Errno::ENOSYS));
            };
            n.addfd
                .iter_mut()
                .filter(|a| a.ret.is_none())
                .filter_map(|a| {
                    let f = a.file.take()?;
                    Some((a.key, f, a.setfd, a.newfd, a.cloexec))
                })
                .collect()
        };
        let mut results = Vec::with_capacity(pending.len());
        for (key, file, setfd, newfd, cloexec) in pending {
            results.push((
                key,
                self.seccomp_install_fd(cx, file, setfd, newfd, cloexec),
            ));
        }
        let answered = !results.is_empty();
        let mut ls = w.listener.st.lock().unwrap();
        let Some(i) = ls.notifs.iter().position(|n| n.id == w.id) else {
            return Some(err(Errno::ENOSYS));
        };
        let n = &mut ls.notifs[i];
        for (key, ret) in results {
            if let Some(a) = n.addfd.iter_mut().find(|a| a.key == key) {
                a.ret = Some(ret);
                if a.send {
                    if ret < 0 {
                        // The reply didn't happen after all.
                        n.state = NState::Sent;
                    } else {
                        (n.flags, n.error, n.val) = (0, 0, ret);
                    }
                }
            }
        }
        if n.state == NState::Replied {
            let n = ls.retire(w.id).expect("found above");
            drop(ls);
            if answered {
                self.seccomp_wake();
            }
            if n.flags & USER_NOTIF_FLAG_CONTINUE != 0 {
                cx.cur.seccomp.continued = true;
                return None;
            }
            return Some(if n.error != 0 {
                i64::from(n.error)
            } else {
                n.val
            });
        }
        w.killable_now = w.killable && n.state == NState::Sent;
        drop(ls);
        cx.cur.seccomp.wait = Some(w);
        if answered {
            self.seccomp_wake();
        }
        cx.block = true;
        Some(0)
    }

    /// Install `file` in the current task for an `ADDFD`: at `newfd`
    /// (replacing what's there, `receive_fd_replace`) or the lowest free
    /// number (`receive_fd`). Returns the number or a negative errno.
    fn seccomp_install_fd(
        &self,
        cx: &mut ServiceCtx,
        file: Fd,
        setfd: bool,
        newfd: u32,
        cloexec: bool,
    ) -> i64 {
        let soft = cx.cur.rlimits.soft(super::sys_misc::RLIMIT_NOFILE);
        let n = if setfd {
            if u64::from(newfd) >= soft {
                return err(Errno::EBADF);
            }
            let n = newfd as i32;
            self.bump_pipe(&file, true);
            if let Some(old) = cx.cur.fds.insert(n, file) {
                self.bump_pipe(&old, false);
                self.epoll_forget(cx, n);
            }
            n
        } else {
            if !cx.cur.fds.has_room_for(1, soft) {
                return err(Errno::EMFILE);
            }
            self.bump_pipe(&file, true);
            cx.cur.fds.alloc(file)
        };
        cx.cur.fds.set_cloexec(n, cloexec);
        i64::from(n)
    }

    /// An `ioctl` on a listener descriptor.
    pub(super) fn seccomp_notify_ioctl(
        &self,
        cx: &mut ServiceCtx,
        file: &NotifFile,
        req: u32,
        arg: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let l = file.listener();
        match req {
            IOCTL_NOTIF_RECV => self.seccomp_notif_recv(cx, l, arg, mem),
            IOCTL_NOTIF_SEND => self.seccomp_notif_send(l, arg, mem),
            IOCTL_NOTIF_ID_VALID | IOCTL_NOTIF_ID_VALID_WRONG_DIR => {
                let Ok(id) = mem.read_u64(arg) else {
                    return err(Errno::EFAULT);
                };
                let ls = l.st.lock().unwrap();
                if ls
                    .notifs
                    .iter()
                    .any(|n| n.id == id && n.state == NState::Sent)
                {
                    0
                } else {
                    err(Errno::ENOENT)
                }
            }
            IOCTL_NOTIF_SET_FLAGS => {
                if arg & !USER_NOTIF_FD_SYNC_WAKE_UP != 0 {
                    return err(Errno::EINVAL);
                }
                l.st.lock().unwrap().flags = arg;
                0
            }
            // ADDFD is extensible: any size from the first version's up.
            _ if req & 0xffff == IOCTL_NOTIF_ADDFD & 0xffff => {
                self.seccomp_notif_addfd(cx, l, ((req >> 16) & 0x3fff) as usize, arg, mem)
            }
            _ => err(Errno::EINVAL),
        }
    }

    /// `SECCOMP_IOCTL_NOTIF_RECV`: take the oldest notification nobody has
    /// received, waiting for one.
    fn seccomp_notif_recv(
        &self,
        cx: &mut ServiceCtx,
        l: &Listener,
        arg: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        // The buffer must come zeroed (the struct is meant to grow).
        match mem.read_vec(arg, NOTIF_SIZE) {
            Ok(b) if b.iter().all(|&x| x == 0) => {}
            Ok(_) => return err(Errno::EINVAL),
            Err(_) => return err(Errno::EFAULT),
        }
        let (id, tid, data) = {
            let mut ls = l.st.lock().unwrap();
            let users = ls.filter.strong_count();
            match ls.notifs.iter_mut().find(|n| n.state == NState::Init) {
                Some(n) => {
                    n.state = NState::Sent;
                    (n.id, n.tid, n.data)
                }
                // Nobody left to send one.
                None if users == 0 => return err(Errno::ENOENT),
                None => {
                    cx.block = true;
                    return 0;
                }
            }
        };
        let mut out = [0u8; NOTIF_SIZE];
        out[0..8].copy_from_slice(&id.to_le_bytes());
        let pid = self.visible_pid(cx, tid);
        out[8..12].copy_from_slice(&pid.to_le_bytes());
        out[16..80].copy_from_slice(&data);
        if mem.write(arg, &out).is_err() {
            // Keep the notification alive for another try.
            let mut ls = l.st.lock().unwrap();
            if let Some(n) = ls.notifs.iter_mut().find(|n| n.id == id) {
                n.state = NState::Init;
            }
            return err(Errno::EFAULT);
        }
        0
    }

    /// `SECCOMP_IOCTL_NOTIF_SEND`: answer a received notification.
    fn seccomp_notif_send(&self, l: &Listener, arg: u64, mem: &GuestMemory) -> i64 {
        let Ok(b) = mem.read_vec(arg, RESP_SIZE) else {
            return err(Errno::EFAULT);
        };
        let id = u64::from_le_bytes(b[0..8].try_into().unwrap());
        let val = i64::from_le_bytes(b[8..16].try_into().unwrap());
        let error = i32::from_le_bytes(b[16..20].try_into().unwrap());
        let flags = u32::from_le_bytes(b[20..24].try_into().unwrap());
        if flags & !USER_NOTIF_FLAG_CONTINUE != 0 {
            return err(Errno::EINVAL);
        }
        if flags & USER_NOTIF_FLAG_CONTINUE != 0 && (error != 0 || val != 0) {
            return err(Errno::EINVAL);
        }
        {
            let mut ls = l.st.lock().unwrap();
            let Some(n) = ls.notifs.iter_mut().find(|n| n.id == id) else {
                return err(Errno::ENOENT);
            };
            // Exactly one reply.
            if n.state != NState::Sent {
                return err(EINPROGRESS);
            }
            n.state = NState::Replied;
            (n.error, n.val, n.flags) = (error, val, flags);
        }
        self.seccomp_wake();
        0
    }

    /// `SECCOMP_IOCTL_NOTIF_ADDFD`: install one of the caller's descriptors in
    /// the task waiting on a notification, and wait until it has.
    fn seccomp_notif_addfd(
        &self,
        cx: &mut ServiceCtx,
        l: &Arc<Listener>,
        size: usize,
        arg: u64,
        mem: &GuestMemory,
    ) -> i64 {
        // The re-trap of one already queued: done yet?
        if let Some(w) = cx.cur.seccomp.addfd.take() {
            if Arc::ptr_eq(&w.listener, l) {
                let mut ls = l.st.lock().unwrap();
                let mut found = None;
                for n in &mut ls.notifs {
                    if let Some(i) = n.addfd.iter().position(|a| a.key == w.key) {
                        found = Some(n.addfd[i].ret.inspect(|_| {
                            n.addfd.remove(i);
                        }));
                        break;
                    }
                }
                if found.is_none()
                    && let Some(i) = ls.done.iter().position(|&(k, _)| k == w.key)
                {
                    found = Some(Some(ls.done.remove(i).1));
                }
                return match found {
                    Some(Some(ret)) => ret,
                    Some(None) => {
                        drop(ls);
                        cx.cur.seccomp.addfd = Some(w);
                        cx.block = true;
                        0
                    }
                    // The task went away before handling it.
                    None => err(Errno::ESRCH),
                };
            }
            abandon(&mut SeccompState {
                addfd: Some(w),
                ..SeccompState::default()
            });
        }
        if !(ADDFD_SIZE..4096).contains(&size) {
            return err(Errno::EINVAL);
        }
        let Ok(b) = mem.read_vec(arg, size) else {
            return err(Errno::EFAULT);
        };
        // A newer, bigger struct is fine as long as what we don't know is 0.
        if b[ADDFD_SIZE..].iter().any(|&x| x != 0) {
            return err(Errno::E2BIG);
        }
        let word = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let id = u64::from_le_bytes(b[0..8].try_into().unwrap());
        let (flags, srcfd, newfd, newfd_flags) = (word(8), word(12), word(16), word(20));
        if newfd_flags & !O_CLOEXEC != 0 {
            return err(Errno::EINVAL);
        }
        if flags & !(ADDFD_FLAG_SETFD | ADDFD_FLAG_SEND) != 0 {
            return err(Errno::EINVAL);
        }
        if newfd != 0 && flags & ADDFD_FLAG_SETFD == 0 {
            return err(Errno::EINVAL);
        }
        let Some(file) = cx.cur.fds.get(srcfd as i32).cloned() else {
            return err(Errno::EBADF);
        };
        let send = flags & ADDFD_FLAG_SEND != 0;
        let key = {
            let mut ls = l.st.lock().unwrap();
            let key = ls.next_key;
            ls.next_key += 1;
            let Some(n) = ls.notifs.iter_mut().find(|n| n.id == id) else {
                return err(Errno::ENOENT);
            };
            // Not before the supervisor has it, nor after it's answered.
            if n.state != NState::Sent {
                return err(EINPROGRESS);
            }
            if send {
                // An atomic add-and-reply only with no other ADDFD queued.
                if !n.addfd.is_empty() {
                    return err(Errno::EBUSY);
                }
                n.state = NState::Replied;
            }
            n.addfd.push(AddFd {
                key,
                file: Some(file),
                setfd: flags & ADDFD_FLAG_SETFD != 0,
                newfd,
                cloexec: newfd_flags & O_CLOEXEC != 0,
                send,
                ret: None,
            });
            key
        };
        cx.cur.seccomp.addfd = Some(AddfdWait {
            listener: Arc::clone(l),
            key,
        });
        self.seccomp_wake();
        cx.block = true;
        0
    }
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{BASE, call, e, setup};
    use super::{Insn, run, validate};
    use crate::abi::arch::Sysno;
    use crate::abi::errno::Errno;

    /// Write a program at BASE + 0x100 and its sock_fprog at BASE.
    fn install(mem: &mut crate::vcpu::GuestMemory, prog: &[(u16, u8, u8, u32)]) {
        let mut raw = Vec::new();
        for &(code, jt, jf, k) in prog {
            raw.extend_from_slice(&code.to_le_bytes());
            raw.push(jt);
            raw.push(jf);
            raw.extend_from_slice(&k.to_le_bytes());
        }
        mem.write(BASE + 0x100, &raw).unwrap();
        let mut f = [0u8; 16];
        f[0..2].copy_from_slice(&(prog.len() as u16).to_le_bytes());
        f[8..16].copy_from_slice(&(BASE + 0x100).to_le_bytes());
        mem.write(BASE, &f).unwrap();
    }

    #[test]
    fn filter_errno_on_one_syscall_allows_the_rest() {
        let (k, mut mem, mut v, mut cx) = setup();
        // if (nr == 172 /* getpid */) return ERRNO(1); else return ALLOW.
        install(
            &mut mem,
            &[
                (0x20, 0, 0, 0),
                (0x15, 0, 1, 172),
                (0x06, 0, 0, 0x0005_0001),
                (0x06, 0, 0, 0x7fff_0000),
            ],
        );
        // Without no_new_privs a non-root task may not install a filter.
        cx.cur.creds.euid = 1000;
        cx.cur.caps = crate::kernel::caps::CapSets::none();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Seccomp,
                [1, 0, BASE, 0, 0, 0]
            ),
            e(Errno::EACCES)
        );
        cx.cur.no_new_privs = true;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Seccomp,
                [1, 0, BASE, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Prctl,
                [21, 0, 0, 0, 0, 0]
            ),
            2
        );
        // The check itself runs in the dispatcher's caller; drive it directly.
        assert_eq!(
            k.seccomp_check(&mut cx, Sysno::Getpid, 172, &[0; 6], 0, &mut mem),
            Some(-1)
        );
        assert_eq!(
            k.seccomp_check(&mut cx, Sysno::Gettid, 178, &[0; 6], 0, &mut mem),
            None
        );
        // A malformed program (no final RET) is EINVAL.
        install(&mut mem, &[(0x20, 0, 0, 0)]);
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Seccomp,
                [1, 0, BASE, 0, 0, 0]
            ),
            e(Errno::EINVAL)
        );
        // Action availability (USER_NOTIF included) and notif sizes.
        for (action, want) in [
            (0x7fc0_0000u32, 0),
            (0x0005_0000, 0),
            (0x7ff0_0000, 0),
            (0x7fe0_0000, e(Errno::EOPNOTSUPP)),
        ] {
            mem.write(BASE + 0x200, &action.to_le_bytes()).unwrap();
            assert_eq!(
                call(
                    &k,
                    &mut cx,
                    &mut mem,
                    &mut v,
                    Sysno::Seccomp,
                    [2, 0, BASE + 0x200, 0, 0, 0]
                ),
                want,
                "action {action:#x}"
            );
        }
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Seccomp,
                [3, 0, BASE + 0x200, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            mem.read_vec(BASE + 0x200, 6).unwrap(),
            [80, 0, 24, 0, 64, 0]
        );
    }

    #[test]
    fn a_listener_takes_a_notification_and_answers_it() {
        let (k, mut mem, mut v, mut cx) = setup();
        cx.cur.no_new_privs = true;
        // getpid (172 on aarch64) → USER_NOTIF, the rest allowed.
        install(
            &mut mem,
            &[
                (0x20, 0, 0, 0),
                (0x15, 0, 1, 172),
                (0x06, 0, 0, 0x7fc0_0000),
                (0x06, 0, 0, 0x7fff_0000),
            ],
        );
        let mut seccomp =
            |cx: &mut super::ServiceCtx, mem: &mut crate::vcpu::GuestMemory, flags: u64| {
                call(
                    &k,
                    cx,
                    mem,
                    &mut v,
                    Sysno::Seccomp,
                    [1, flags, BASE, 0, 0, 0],
                )
            };
        // WAIT_KILLABLE_RECV needs NEW_LISTENER; TSYNC with it needs TSYNC_ESRCH.
        assert_eq!(seccomp(&mut cx, &mut mem, 32), e(Errno::EINVAL));
        assert_eq!(seccomp(&mut cx, &mut mem, 8 | 1), e(Errno::EINVAL));
        let fd = seccomp(&mut cx, &mut mem, 8);
        assert!(fd >= 3, "listener fd {fd}");
        assert!(cx.cur.fds.is_cloexec(fd as i32));
        // A second listener in the same chain is EBUSY.
        assert_eq!(seccomp(&mut cx, &mut mem, 8), e(Errno::EBUSY));
        let Some(super::Fd::SeccompNotify(file)) = cx.cur.fds.get(fd as i32).cloned() else {
            panic!("not a listener");
        };
        let l = file.listener().clone();
        assert_eq!(l.poll(), 0);
        // The task's getpid waits for the supervisor.
        let r = k.seccomp_check(
            &mut cx,
            Sysno::Getpid,
            172,
            &[7, 0, 0, 0, 0, 0],
            0,
            &mut mem,
        );
        assert_eq!(r, Some(0));
        assert!(cx.block);
        assert_eq!(l.poll() & 0x1, 0x1, "POLLIN with a notification waiting");
        // The supervisor (here: the same task's context, for the ioctls)
        // receives it; the buffer must be zeroed.
        let w = cx.cur.seccomp.wait.take();
        let mut sup = super::ServiceCtx::for_test();
        sup.cur.pid = 2;
        sup.cur.fds.alloc(super::Fd::SeccompNotify(file.clone()));
        mem.write(BASE + 0x400, &[1u8; 80]).unwrap();
        assert_eq!(
            k.seccomp_notify_ioctl(
                &mut sup,
                &file,
                super::IOCTL_NOTIF_RECV,
                BASE + 0x400,
                &mut mem
            ),
            e(Errno::EINVAL)
        );
        mem.write(BASE + 0x400, &[0u8; 80]).unwrap();
        assert_eq!(
            k.seccomp_notify_ioctl(
                &mut sup,
                &file,
                super::IOCTL_NOTIF_RECV,
                BASE + 0x400,
                &mut mem
            ),
            0
        );
        let got = mem.read_vec(BASE + 0x400, 80).unwrap();
        let id = u64::from_le_bytes(got[0..8].try_into().unwrap());
        assert_eq!(u32::from_le_bytes(got[8..12].try_into().unwrap()), 1, "pid");
        assert_eq!(
            u32::from_le_bytes(got[16..20].try_into().unwrap()),
            172,
            "nr"
        );
        assert_eq!(
            u64::from_le_bytes(got[32..40].try_into().unwrap()),
            7,
            "arg0"
        );
        assert_eq!(l.poll() & 0x5, 0x4, "POLLOUT once received");
        // ID_VALID, then the answer.
        mem.write(BASE + 0x500, &id.to_le_bytes()).unwrap();
        let ioctl = |sup: &mut super::ServiceCtx, mem: &mut crate::vcpu::GuestMemory, r, a| {
            k.seccomp_notify_ioctl(sup, &file, r, a, mem)
        };
        assert_eq!(
            ioctl(
                &mut sup,
                &mut mem,
                super::IOCTL_NOTIF_ID_VALID,
                BASE + 0x500
            ),
            0
        );
        let mut resp = [0u8; 24];
        resp[0..8].copy_from_slice(&id.to_le_bytes());
        resp[8..16].copy_from_slice(&42i64.to_le_bytes());
        mem.write(BASE + 0x600, &resp).unwrap();
        assert_eq!(
            ioctl(&mut sup, &mut mem, super::IOCTL_NOTIF_SEND, BASE + 0x600),
            0
        );
        assert_eq!(
            ioctl(&mut sup, &mut mem, super::IOCTL_NOTIF_SEND, BASE + 0x600),
            e(super::EINPROGRESS),
            "exactly one reply"
        );
        cx.cur.seccomp.wait = w;
        cx.block = false;
        assert_eq!(
            k.seccomp_check(
                &mut cx,
                Sysno::Getpid,
                172,
                &[7, 0, 0, 0, 0, 0],
                0,
                &mut mem
            ),
            Some(42)
        );
        assert!(!cx.block);
        assert_eq!(
            ioctl(
                &mut sup,
                &mut mem,
                super::IOCTL_NOTIF_ID_VALID,
                BASE + 0x500
            ),
            e(Errno::ENOENT)
        );
        // The listener closed: USER_NOTIF fails ENOSYS.
        drop(sup);
        drop(file);
        cx.cur.fds.close(fd as i32);
        assert_eq!(
            k.seccomp_check(&mut cx, Sysno::Getpid, 172, &[0; 6], 0, &mut mem),
            Some(e(Errno::ENOSYS))
        );
        // With the task's filters gone, the listener reports POLLHUP.
        super::abandon(&mut cx.cur.seccomp);
        cx.cur.seccomp.filters.clear();
        assert_eq!(l.poll() & 0x10, 0x10);
    }

    #[test]
    fn the_most_restrictive_filter_wins_and_alu_works() {
        // A filter computing (nr & 0xff) + 1 == 0x10 via ALU before comparing.
        let prog = [
            Insn {
                code: 0x20,
                jt: 0,
                jf: 0,
                k: 0,
            },
            Insn {
                code: 0x54,
                jt: 0,
                jf: 0,
                k: 0xff,
            },
            Insn {
                code: 0x04,
                jt: 0,
                jf: 0,
                k: 1,
            },
            Insn {
                code: 0x15,
                jt: 0,
                jf: 1,
                k: 0x10,
            },
            Insn {
                code: 0x06,
                jt: 0,
                jf: 0,
                k: 0x0003_0000,
            },
            Insn {
                code: 0x06,
                jt: 0,
                jf: 0,
                k: 0x7fff_0000,
            },
        ];
        assert!(validate(&prog));
        let mut data = [0u8; 64];
        data[0] = 0x0f;
        assert_eq!(run(&prog, &data), 0x0003_0000);
        data[0] = 0x10;
        assert_eq!(run(&prog, &data), 0x7fff_0000);
        // Out-of-range loads and jumps are rejected.
        assert!(!validate(&[
            Insn {
                code: 0x20,
                jt: 0,
                jf: 0,
                k: 64
            },
            prog[5]
        ]));
        assert!(!validate(&[
            Insn {
                code: 0x15,
                jt: 5,
                jf: 0,
                k: 0
            },
            prog[5]
        ]));
    }
}
