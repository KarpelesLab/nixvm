//! `prctl(2)`: the process-attribute multiplexer, covering every option a
//! Linux 6.x kernel on arm64/x86-64 knows. Options with state the VM can
//! keep are stored per task and reported back; options for hardware or
//! kernel features that aren't there (SVE/SME vector lengths, pointer
//! authentication, tagged addresses, shadow stacks, syscall user dispatch,
//! core scheduling, other architectures' knobs) answer `EINVAL`, exactly as a
//! kernel built without them does — which is what their users probe for.
//! An option number nobody defines is `EINVAL` and is recorded as an
//! unsupported subcommand.

use super::{Kernel, ServiceCtx, err};
use crate::abi::Arch;
use crate::abi::errno::Errno;
use crate::vcpu::GuestMemory;

/// The last capability Linux defines (`CAP_CHECKPOINT_RESTORE`).
const CAP_LAST_CAP: u64 = 40;
const CAP_ALL: u64 = (1u64 << (CAP_LAST_CAP + 1)) - 1;

/// Per-task `prctl` state without a better home.
#[derive(Clone, Copy, Debug)]
#[allow(clippy::struct_excessive_bools)] // independent per-option latches
pub(super) struct PrctlState {
    keepcaps: bool,
    /// The capability bounding set (`PR_CAPBSET_READ`/`DROP`).
    bset: u64,
    /// Ambient capabilities (`PR_CAP_AMBIENT`).
    ambient: u64,
    securebits: u32,
    /// Timer slack in ns (default 50 µs).
    timerslack: u64,
    /// `PR_SET_CHILD_SUBREAPER`: orphaned descendants are reparented here
    /// rather than to init. Not inherited across `fork`.
    pub(super) child_subreaper: bool,
    thp_disable: bool,
    /// `PR_MCE_KILL` policy (`PR_MCE_KILL_DEFAULT` = 2).
    mce_kill: u64,
    io_flusher: bool,
    /// `PR_SET_MDWE` flags: deny write+execute mappings and exec gain.
    pub(super) mdwe: u64,
    memory_merge: bool,
    /// x86 `PR_SET_TSC` mode (`PR_TSC_ENABLE` = 1).
    tsc: u64,
}

impl Default for PrctlState {
    fn default() -> Self {
        Self {
            keepcaps: false,
            bset: CAP_ALL,
            ambient: 0,
            securebits: 0,
            timerslack: 50_000,
            child_subreaper: false,
            thp_disable: false,
            mce_kill: 2,
            io_flusher: false,
            mdwe: 0,
            memory_merge: false,
            tsc: 1,
        }
    }
}

/// `PR_MDWE_REFUSE_EXEC_GAIN`.
pub(super) const MDWE_REFUSE_EXEC_GAIN: u64 = 1;

impl Kernel {
    /// `prctl(option, arg2, arg3, arg4, arg5)`.
    #[allow(clippy::too_many_lines, clippy::match_same_arms)] // one arm per option, named for the record
    pub(super) fn sys_prctl(
        &self,
        cx: &mut ServiceCtx,
        args: &[u64; 6],
        mem: &mut GuestMemory,
    ) -> i64 {
        const PR_SET_PDEATHSIG: u64 = 1;
        const PR_GET_PDEATHSIG: u64 = 2;
        const PR_GET_DUMPABLE: u64 = 3;
        const PR_SET_DUMPABLE: u64 = 4;
        const PR_GET_KEEPCAPS: u64 = 7;
        const PR_SET_KEEPCAPS: u64 = 8;
        const PR_GET_TIMING: u64 = 13;
        const PR_SET_TIMING: u64 = 14;
        const PR_SET_NAME: u64 = 15;
        const PR_GET_NAME: u64 = 16;
        const PR_GET_SECCOMP: u64 = 21;
        const PR_SET_SECCOMP: u64 = 22;
        const PR_CAPBSET_READ: u64 = 23;
        const PR_CAPBSET_DROP: u64 = 24;
        const PR_GET_TSC: u64 = 25;
        const PR_SET_TSC: u64 = 26;
        const PR_GET_SECUREBITS: u64 = 27;
        const PR_SET_SECUREBITS: u64 = 28;
        const PR_SET_TIMERSLACK: u64 = 29;
        const PR_GET_TIMERSLACK: u64 = 30;
        const PR_TASK_PERF_EVENTS_DISABLE: u64 = 31;
        const PR_TASK_PERF_EVENTS_ENABLE: u64 = 32;
        const PR_MCE_KILL: u64 = 33;
        const PR_MCE_KILL_GET: u64 = 34;
        const PR_SET_MM: u64 = 35;
        const PR_SET_CHILD_SUBREAPER: u64 = 36;
        const PR_GET_CHILD_SUBREAPER: u64 = 37;
        const PR_SET_NO_NEW_PRIVS: u64 = 38;
        const PR_GET_NO_NEW_PRIVS: u64 = 39;
        const PR_GET_TID_ADDRESS: u64 = 40;
        const PR_SET_THP_DISABLE: u64 = 41;
        const PR_GET_THP_DISABLE: u64 = 42;
        const PR_CAP_AMBIENT: u64 = 47;
        const PR_GET_SPECULATION_CTRL: u64 = 52;
        const PR_SET_SPECULATION_CTRL: u64 = 53;
        const PR_SET_IO_FLUSHER: u64 = 57;
        const PR_GET_IO_FLUSHER: u64 = 58;
        const PR_SET_MDWE: u64 = 65;
        const PR_GET_MDWE: u64 = 66;
        const PR_SET_MEMORY_MERGE: u64 = 67;
        const PR_GET_MEMORY_MERGE: u64 = 68;
        const PR_SET_VMA: u64 = 0x5356_4d41;
        const PR_SET_PTRACER: u64 = 0x5961_6d61;
        let (a2, a3) = (args[1], args[2]);
        let root = cx.cur.creds.euid == 0;
        let put_i32 = |mem: &mut GuestMemory, at: u64, v: i32| -> i64 {
            if mem.write(at, &v.to_le_bytes()).is_ok() {
                0
            } else {
                err(Errno::EFAULT)
            }
        };
        let pr = &mut cx.cur.pr;
        match args[0] {
            PR_SET_NAME => {
                let Ok(name) = mem.read_cstr(a2, 16) else {
                    return err(Errno::EFAULT);
                };
                let n = name.len().min(15);
                cx.cur.comm = String::from_utf8_lossy(&name[..n]).into_owned();
                0
            }
            PR_GET_NAME => {
                // The kernel writes a fixed 16-byte, NUL-padded buffer.
                let mut buf = [0u8; 16];
                let b = cx.cur.comm.as_bytes();
                let n = b.len().min(15);
                buf[..n].copy_from_slice(&b[..n]);
                if mem.write(a2, &buf).is_err() {
                    return err(Errno::EFAULT);
                }
                0
            }
            // no_new_privs is a one-way latch: it can be set, never cleared.
            PR_SET_NO_NEW_PRIVS => {
                if a2 != 1 || args[2..5].iter().any(|&a| a != 0) {
                    return err(Errno::EINVAL);
                }
                cx.cur.no_new_privs = true;
                0
            }
            PR_GET_NO_NEW_PRIVS => i64::from(cx.cur.no_new_privs),
            // pdeathsig: delivered when the parent dies (see exit_task).
            PR_SET_PDEATHSIG => {
                if a2 > 64 {
                    return err(Errno::EINVAL);
                }
                cx.cur.pdeathsig = a2;
                0
            }
            PR_GET_PDEATHSIG => put_i32(mem, a2, cx.cur.pdeathsig as i32),
            // dumpable: we never emit cores, but sandboxes toggle and re-read
            // it. Settable values: 0/1 (2, SUID_DUMP_ROOT, is not settable).
            PR_SET_DUMPABLE => {
                if a2 > 1 {
                    return err(Errno::EINVAL);
                }
                cx.cur.dumpable = a2;
                0
            }
            #[allow(clippy::cast_possible_wrap)]
            PR_GET_DUMPABLE => cx.cur.dumpable as i64,
            PR_SET_KEEPCAPS => {
                if a2 > 1 {
                    return err(Errno::EINVAL);
                }
                pr.keepcaps = a2 == 1;
                0
            }
            PR_GET_KEEPCAPS => i64::from(pr.keepcaps),
            // Only statistical process timing exists.
            PR_GET_TIMING => 0,
            PR_SET_TIMING => {
                if a2 == 0 {
                    0
                } else {
                    err(Errno::EINVAL)
                }
            }
            PR_GET_SECCOMP => i64::from(cx.cur.seccomp.mode()),
            PR_SET_SECCOMP => self.seccomp_set_mode(cx, a2, a3, 0, mem),
            PR_CAPBSET_READ => {
                if a2 > CAP_LAST_CAP {
                    err(Errno::EINVAL)
                } else {
                    i64::from(pr.bset >> a2 & 1 == 1)
                }
            }
            PR_CAPBSET_DROP => {
                if a2 > CAP_LAST_CAP {
                    err(Errno::EINVAL)
                } else if !root {
                    err(Errno::EPERM)
                } else {
                    pr.bset &= !(1 << a2);
                    0
                }
            }
            PR_CAP_AMBIENT => {
                const IS_SET: u64 = 1;
                const RAISE: u64 = 2;
                const LOWER: u64 = 3;
                const CLEAR_ALL: u64 = 4;
                if a2 == CLEAR_ALL {
                    if a3 != 0 {
                        return err(Errno::EINVAL);
                    }
                    pr.ambient = 0;
                    return 0;
                }
                if a3 > CAP_LAST_CAP || args[3] != 0 || args[4] != 0 {
                    return err(Errno::EINVAL);
                }
                match a2 {
                    IS_SET => i64::from(pr.ambient >> a3 & 1 == 1),
                    // Raising needs the capability permitted and inheritable;
                    // root's default set has nothing inheritable.
                    RAISE => {
                        let inh = cx.cur.caps.map_or(0, |c| c[2]);
                        let perm = cx.cur.caps.map_or(if root { CAP_ALL } else { 0 }, |c| c[1]);
                        if inh >> a3 & 1 == 0 || perm >> a3 & 1 == 0 {
                            err(Errno::EPERM)
                        } else {
                            pr.ambient |= 1 << a3;
                            0
                        }
                    }
                    LOWER => {
                        pr.ambient &= !(1 << a3);
                        0
                    }
                    _ => err(Errno::EINVAL),
                }
            }
            PR_GET_SECUREBITS => i64::from(pr.securebits),
            PR_SET_SECUREBITS => {
                if !root {
                    err(Errno::EPERM)
                } else if a2 > 0xff {
                    err(Errno::EINVAL)
                } else {
                    pr.securebits = a2 as u32;
                    0
                }
            }
            PR_GET_TSC if self.arch == Arch::X86_64 => put_i32(mem, a2, pr.tsc as i32),
            PR_SET_TSC if self.arch == Arch::X86_64 => {
                if a2 == 1 || a2 == 2 {
                    pr.tsc = a2;
                    0
                } else {
                    err(Errno::EINVAL)
                }
            }
            PR_SET_TIMERSLACK => {
                pr.timerslack = if a2 == 0 { 50_000 } else { a2 };
                0
            }
            #[allow(clippy::cast_possible_wrap)]
            PR_GET_TIMERSLACK => pr.timerslack as i64,
            PR_TASK_PERF_EVENTS_DISABLE | PR_TASK_PERF_EVENTS_ENABLE => 0,
            PR_MCE_KILL => match (a2, a3) {
                (0, _) => {
                    pr.mce_kill = 2;
                    0
                }
                (1, 0..=2) => {
                    pr.mce_kill = a3;
                    0
                }
                _ => err(Errno::EINVAL),
            },
            #[allow(clippy::cast_possible_wrap)]
            PR_MCE_KILL_GET => pr.mce_kill as i64,
            // Rewriting the mm's bookkeeping (CRIU): only the size query has
            // an observable answer; the rest is accepted for privileged
            // callers.
            PR_SET_MM => {
                const PR_SET_MM_MAP_SIZE: u64 = 15;
                if !root {
                    return err(Errno::EPERM);
                }
                match a2 {
                    PR_SET_MM_MAP_SIZE => put_i32(mem, a3, 104),
                    1..=14 => 0,
                    _ => err(Errno::EINVAL),
                }
            }
            PR_SET_CHILD_SUBREAPER => {
                pr.child_subreaper = a2 != 0;
                0
            }
            PR_GET_CHILD_SUBREAPER => put_i32(mem, a2, i32::from(pr.child_subreaper)),
            PR_GET_TID_ADDRESS => {
                if mem.write_u64(a2, cx.cur.clear_child_tid).is_ok() {
                    0
                } else {
                    err(Errno::EFAULT)
                }
            }
            PR_SET_THP_DISABLE => {
                pr.thp_disable = a2 != 0;
                0
            }
            PR_GET_THP_DISABLE => i64::from(pr.thp_disable),
            // The emulated CPUs have no speculative-execution side channels to
            // mitigate: "not affected", and nothing to control.
            PR_GET_SPECULATION_CTRL => {
                if a2 > 2 {
                    err(Errno::EINVAL)
                } else {
                    0
                }
            }
            PR_SET_SPECULATION_CTRL => {
                if a2 > 2 {
                    err(Errno::EINVAL)
                } else {
                    err(Errno::ENXIO)
                }
            }
            PR_SET_IO_FLUSHER => {
                if !root {
                    err(Errno::EPERM)
                } else if a2 > 1 {
                    err(Errno::EINVAL)
                } else {
                    pr.io_flusher = a2 == 1;
                    0
                }
            }
            PR_GET_IO_FLUSHER => i64::from(pr.io_flusher),
            // Memory-deny-write-execute: once set it can't be cleared, and
            // mmap/mprotect then refuse W+X and exec gain (EACCES).
            PR_SET_MDWE => {
                if a2 & !3 != 0 || args[2..5].iter().any(|&a| a != 0) {
                    return err(Errno::EINVAL);
                }
                if pr.mdwe & MDWE_REFUSE_EXEC_GAIN != 0 && a2 & MDWE_REFUSE_EXEC_GAIN == 0 {
                    return err(Errno::EPERM);
                }
                pr.mdwe = a2;
                0
            }
            #[allow(clippy::cast_possible_wrap)]
            PR_GET_MDWE => pr.mdwe as i64,
            PR_SET_MEMORY_MERGE => {
                if root {
                    pr.memory_merge = a2 != 0;
                    0
                } else {
                    err(Errno::EPERM)
                }
            }
            PR_GET_MEMORY_MERGE => i64::from(pr.memory_merge),
            // Naming an anonymous mapping (shows in /proc/self/maps on Linux):
            // accepted, as the name has nowhere to surface here.
            PR_SET_VMA => {
                if a2 == 0 {
                    0
                } else {
                    err(Errno::EINVAL)
                }
            }
            // Yama ptrace scope: there is no ptrace, so any tracer is fine.
            PR_SET_PTRACER => 0,
            // Hardware/kernel features these CPUs and this kernel don't have:
            // unaligned/FP-emulation/endian/FP-exception knobs of other arches
            // (5,6,9-12,19,20,45,46), MPX (43,44), SVE (50,51), PAC (54,60,61),
            // tagged addresses (55,56), syscall user dispatch (59), core
            // scheduling (62), SME (63,64), RISC-V and powerpc controls
            // (69-73), shadow stacks (74-76), timer id restore (77), futex
            // hash (78).
            5
            | 6
            | 9..=12
            | 19
            | 20
            | 25
            | 26
            | 43..=46
            | 50
            | 51
            | 54..=56
            | 59..=64
            | 69..=78 => err(Errno::EINVAL),
            other => {
                self.note_unsupported("prctl", other);
                err(Errno::EINVAL)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::testutil::{BASE, call, e, setup};
    use crate::abi::arch::Sysno;
    use crate::abi::errno::Errno;

    #[test]
    fn prctl_state_roundtrips_and_unknowns_are_einval() {
        let (k, mut mem, mut v, mut cx) = setup();
        let p = |k: &super::Kernel,
                 cx: &mut super::ServiceCtx,
                 mem: &mut crate::vcpu::GuestMemory,
                 a: [u64; 6]| {
            call(
                k,
                cx,
                mem,
                &mut super::super::testutil::DummyVcpu,
                Sysno::Prctl,
                a,
            )
        };
        let _ = &mut v;
        // Child subreaper.
        assert_eq!(p(&k, &mut cx, &mut mem, [36, 1, 0, 0, 0, 0]), 0);
        assert_eq!(p(&k, &mut cx, &mut mem, [37, BASE, 0, 0, 0, 0]), 0);
        assert_eq!(mem.read_u32(BASE).unwrap(), 1);
        // Bounding set: all caps readable, dropping one sticks.
        assert_eq!(p(&k, &mut cx, &mut mem, [23, 21, 0, 0, 0, 0]), 1);
        assert_eq!(p(&k, &mut cx, &mut mem, [24, 21, 0, 0, 0, 0]), 0);
        assert_eq!(p(&k, &mut cx, &mut mem, [23, 21, 0, 0, 0, 0]), 0);
        assert_eq!(
            p(&k, &mut cx, &mut mem, [23, 99, 0, 0, 0, 0]),
            e(Errno::EINVAL)
        );
        // Timer slack.
        assert_eq!(p(&k, &mut cx, &mut mem, [30, 0, 0, 0, 0, 0]), 50_000);
        assert_eq!(p(&k, &mut cx, &mut mem, [29, 1000, 0, 0, 0, 0]), 0);
        assert_eq!(p(&k, &mut cx, &mut mem, [30, 0, 0, 0, 0, 0]), 1000);
        // no_new_privs is a one-way latch with strict arguments.
        assert_eq!(
            p(&k, &mut cx, &mut mem, [38, 0, 0, 0, 0, 0]),
            e(Errno::EINVAL)
        );
        assert_eq!(p(&k, &mut cx, &mut mem, [38, 1, 0, 0, 0, 0]), 0);
        assert_eq!(p(&k, &mut cx, &mut mem, [39, 0, 0, 0, 0, 0]), 1);
        // MDWE can't be relaxed once refusing exec gain.
        assert_eq!(p(&k, &mut cx, &mut mem, [65, 1, 0, 0, 0, 0]), 0);
        assert_eq!(
            p(&k, &mut cx, &mut mem, [65, 0, 0, 0, 0, 0]),
            e(Errno::EPERM)
        );
        // SVE length on a CPU without SVE; an option nobody defines.
        assert_eq!(
            p(&k, &mut cx, &mut mem, [51, 0, 0, 0, 0, 0]),
            e(Errno::EINVAL)
        );
        assert_eq!(
            p(&k, &mut cx, &mut mem, [4242, 0, 0, 0, 0, 0]),
            e(Errno::EINVAL)
        );
        assert_eq!(k.unsupported_subcommands().get(&("prctl", 4242)), Some(&1));
        // PR_GET_TID_ADDRESS.
        cx.cur.clear_child_tid = 0xabc0;
        assert_eq!(p(&k, &mut cx, &mut mem, [40, BASE, 0, 0, 0, 0]), 0);
        assert_eq!(mem.read_u64(BASE).unwrap(), 0xabc0);
    }
}
