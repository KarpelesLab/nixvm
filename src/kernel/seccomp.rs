//! seccomp: `seccomp(2)` and `prctl(PR_SET_SECCOMP)`, *enforced*.
//!
//! Strict mode (`SECCOMP_SET_MODE_STRICT`) leaves a task only `read`,
//! `write`, `_exit` and `rt_sigreturn`; anything else kills it with
//! `SIGKILL`. Filter mode runs the installed classic-BPF programs over the
//! `struct seccomp_data` of every syscall (newest filter first, the most
//! restrictive verdict winning, as Linux combines them) and applies the
//! verdict: `ALLOW`/`LOG` run the call, `ERRNO(n)` fails it with `-n`
//! unexecuted, `TRAP` raises `SIGSYS` (`si_code` `SYS_SECCOMP`) and fails it
//! `ENOSYS`, `TRACE` and `USER_NOTIF` find no tracer/listener and fail it
//! `ENOSYS`, `KILL_THREAD`/`KILL_PROCESS` terminate with `SIGSYS`. Filters are
//! inherited by `fork`/`clone` and kept across `execve`, and installing one
//! requires `no_new_privs` or `CAP_SYS_ADMIN`, so sandboxes (OpenSSH's
//! privsep child, systemd services, Chromium/Firefox, minijail, bubblewrap,
//! Python's `seccomp` users) get the confinement they ask for.
//!
//! Every filter is validated as Linux's `sk_chk_filter` + seccomp checks do
//! (known opcodes only, in-range forward jumps, aligned `seccomp_data`
//! loads, scratch-memory bounds, a final `RET`), so a malformed program is
//! `EINVAL` at install rather than misbehaving later. There is no user-space
//! notification listener (`SECCOMP_FILTER_FLAG_NEW_LISTENER` is `EINVAL`).

use std::sync::Arc;

use super::{ExitCause, Kernel, QueuedSig, ServiceCtx, Shared, err};
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
/// `sizeof(struct seccomp_data)`.
const DATA_LEN: u32 = 64;

/// One classic-BPF instruction (`struct sock_filter`).
#[derive(Clone, Copy, Debug)]
struct Insn {
    code: u16,
    jt: u8,
    jf: u8,
    k: u32,
}

/// A task's seccomp state; cloned into children (the programs are shared).
#[derive(Clone, Debug, Default)]
pub(super) struct SeccompState {
    mode: u8,
    /// Installed filters, oldest first.
    filters: Vec<Arc<Vec<Insn>>>,
}

impl SeccompState {
    /// The mode `PR_GET_SECCOMP` reports (0/1/2).
    pub(super) fn mode(&self) -> u8 {
        self.mode
    }

    /// Whether any syscall needs checking.
    pub(super) fn active(&self) -> bool {
        self.mode != SECCOMP_MODE_DISABLED
    }
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

impl Kernel {
    /// Read and validate a `struct sock_fprog { u16 len; u64 filter; }`.
    fn read_fprog(mem: &GuestMemory, ptr: u64) -> Result<Vec<Insn>, i64> {
        let (Ok(len), Ok(filter)) = (mem.read_vec(ptr, 2), mem.read_u64(ptr + 8)) else {
            return Err(err(Errno::EFAULT));
        };
        let len = usize::from(u16::from_le_bytes([len[0], len[1]]));
        if len == 0 || len > BPF_MAXINSNS {
            return Err(err(Errno::EINVAL));
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

    /// Enter a seccomp mode: the shared core of `seccomp(SET_MODE_*)` and
    /// `prctl(PR_SET_SECCOMP)`. (`SECCOMP_FILTER_FLAG_TSYNC` is applied by
    /// the caller, which holds the task table.)
    #[allow(clippy::unused_self)]
    pub(super) fn seccomp_set_mode(
        &self,
        cx: &mut ServiceCtx,
        mode: u64,
        args: u64,
        flags: u64,
        mem: &GuestMemory,
    ) -> i64 {
        let st = &mut cx.cur.seccomp;
        match mode {
            1 => {
                // SECCOMP_MODE_STRICT: no flags, no args; can't leave filter mode.
                if flags != 0 || args != 0 {
                    return err(Errno::EINVAL);
                }
                if st.mode == SECCOMP_MODE_FILTER {
                    return err(Errno::EINVAL);
                }
                st.mode = SECCOMP_MODE_STRICT;
                0
            }
            2 => {
                if st.mode == SECCOMP_MODE_STRICT {
                    return err(Errno::EINVAL);
                }
                if !cx.cur.no_new_privs && cx.cur.creds.euid != 0 {
                    return err(Errno::EACCES);
                }
                let prog = match Self::read_fprog(mem, args) {
                    Ok(p) => p,
                    Err(e) => return e,
                };
                let st = &mut cx.cur.seccomp;
                st.mode = SECCOMP_MODE_FILTER;
                st.filters.push(Arc::new(prog));
                0
            }
            _ => err(Errno::EINVAL),
        }
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
        const FLAG_TSYNC: u64 = 1;
        const FLAG_LOG: u64 = 2;
        const FLAG_SPEC_ALLOW: u64 = 4;
        const FLAG_NEW_LISTENER: u64 = 8;
        const FLAG_TSYNC_ESRCH: u64 = 16;
        const FLAG_WAIT_KILLABLE_RECV: u64 = 32;
        match op {
            SET_MODE_STRICT => self.seccomp_set_mode(cx, 1, args, flags, mem),
            SET_MODE_FILTER => {
                let known = FLAG_TSYNC
                    | FLAG_LOG
                    | FLAG_SPEC_ALLOW
                    | FLAG_NEW_LISTENER
                    | FLAG_TSYNC_ESRCH
                    | FLAG_WAIT_KILLABLE_RECV;
                // No notification listener can be handed out.
                if flags & !known != 0 || flags & FLAG_NEW_LISTENER != 0 {
                    return err(Errno::EINVAL);
                }
                let r = self.seccomp_set_mode(cx, 2, args, 0, mem);
                if r == 0 && flags & FLAG_TSYNC != 0 {
                    // Synchronize every thread of the group to our filters.
                    let (tgid, state) = (cx.cur.tgid, cx.cur.seccomp.clone());
                    for p in sh.procs.iter_mut().flatten() {
                        if p.info.tgid == tgid {
                            p.info.seccomp = state.clone();
                        }
                    }
                }
                r
            }
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
                // struct seccomp_notif_sizes { u16 notif = 80; u16 resp = 24;
                // u16 data = 64 }.
                let b = [80u8, 0, 24, 0, 64, 0];
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
    /// without running it (the task may also have been killed).
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
        const SIGSYS: u64 = 31;
        const SYS_SECCOMP: i32 = 1;
        if cx.cur.seccomp.mode == SECCOMP_MODE_STRICT {
            if matches!(
                sys,
                Sysno::Read | Sysno::Write | Sysno::Exit | Sysno::RtSigreturn
            ) {
                return None;
            }
            let mut sh = self.shared.lock().unwrap();
            self.exit_group_with(&mut sh, cx, ExitCause::Signaled(SIGKILL), mem);
            return Some(0);
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
        // compares the action part as a signed value: KILL_PROCESS lowest).
        let verdict = cx
            .cur
            .seccomp
            .filters
            .iter()
            .rev()
            .map(|f| run(f, &data))
            .min_by_key(|r| (r & SECCOMP_RET_ACTION_FULL) as i32)
            .unwrap_or(SECCOMP_RET_ALLOW);
        let value = verdict & SECCOMP_RET_DATA;
        match verdict & SECCOMP_RET_ACTION_FULL {
            SECCOMP_RET_ALLOW | SECCOMP_RET_LOG => None,
            SECCOMP_RET_ERRNO => Some(-i64::from(value.min(4095))),
            SECCOMP_RET_TRAP => {
                // SIGSYS with si_code SYS_SECCOMP; the _sigsys fields (call
                // address, syscall, arch) ride in the pid/uid/value slots.
                let info = QueuedSig {
                    code: SYS_SECCOMP,
                    pid: pc as u32 as i32,
                    uid: (pc >> 32) as u32,
                    value: u64::from(raw as u32) | u64::from(arch) << 32,
                };
                cx.cur.pending |= 1 << (SIGSYS - 1);
                cx.cur.post_siginfo(SIGSYS, info);
                Some(err(Errno::ENOSYS))
            }
            SECCOMP_RET_TRACE | SECCOMP_RET_USER_NOTIF => Some(err(Errno::ENOSYS)),
            // KILL_PROCESS / KILL_THREAD (and anything unknown, which Linux
            // also treats as a kill).
            _ => {
                let mut sh = self.shared.lock().unwrap();
                self.exit_group_with(&mut sh, cx, ExitCause::Signaled(SIGSYS as i32), mem);
                Some(0)
            }
        }
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
        // Action availability and notif sizes.
        mem.write(BASE + 0x200, &0x7fc0_0000u32.to_le_bytes())
            .unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Seccomp,
                [2, 0, BASE + 0x200, 0, 0, 0]
            ),
            e(Errno::EOPNOTSUPP)
        );
        mem.write(BASE + 0x200, &0x0005_0000u32.to_le_bytes())
            .unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Seccomp,
                [2, 0, BASE + 0x200, 0, 0, 0]
            ),
            0
        );
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
