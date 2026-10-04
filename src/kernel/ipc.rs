//! System V IPC: message queues (`msgget`/`msgsnd`/`msgrcv`/`msgctl`),
//! semaphore sets (`semget`/`semop`/`semtimedop`/`semctl`) and shared memory
//! (`shmget`/`shmat`/`shmdt`/`shmctl`).
//!
//! All three live in one machine-wide table ([`Ipc`], in [`Shared`]) — one IPC
//! namespace, keyed and permissioned as on Linux. Blocking operations (a full
//! queue, an empty one, a semaphore that can't proceed) use the kernel's re-trap
//! convention: the call sets `cx.block` and is re-executed when something may
//! have changed; every state change un-parks the waiters so they re-check.
//! These calls are never restarted after a signal handler (`EINTR` regardless
//! of `SA_RESTART`, per signal(7)), and a waiter whose object is removed under
//! it gets `EIDRM`.
//!
//! Shared memory is *real* sharing: a segment owns physical frames in the
//! pool every address space draws from, and `shmat` maps those same frames
//! (tagged shared, so `fork` aliases rather than copies them) into each
//! attacher — a store in one process is a store in all (see
//! [`GuestMemory::map_frames`]). Attachments are tracked per address space so
//! `shm_nattch`, `IPC_RMID`'s deferred destruction, `fork` inheritance and the
//! implicit detach at `execve`/exit all behave.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use super::{Kernel, ServiceCtx, Shared, err, poll};
use crate::abi::Arch;
use crate::abi::errno::Errno;
use crate::vcpu::GuestMemory;
use crate::vcpu::mem::{PAGE_SIZE, Prot};

const IPC_PRIVATE: i32 = 0;
const IPC_CREAT: u64 = 0o1000;
const IPC_EXCL: u64 = 0o2000;
const IPC_NOWAIT: u64 = 0o4000;
const IPC_RMID: u64 = 0;
const IPC_SET: u64 = 1;
const IPC_STAT: u64 = 2;
const IPC_INFO: u64 = 3;
/// `IPC_64`, ORed into `*ctl` commands by libcs: selects the 64-bit structs,
/// the only ones there are on these arches.
const IPC_64: u64 = 0x100;

// Limits (the Linux defaults, as reported by IPC_INFO).
const MSGMAX: u64 = 8192;
const MSGMNB: u64 = 16384;
const MSGMNI: i32 = 32000;
const SEMMSL: u64 = 32000;
const SEMMNI: i32 = 32000;
const SEMOPM: u64 = 500;
const SEMVMX: i32 = 32767;
const SHMMIN: u64 = 1;
const SHMMAX: u64 = 1 << 32;
const SHMMNI: i32 = 4096;

/// The `struct ipc64_perm` an object carries.
#[derive(Clone, Debug)]
struct Perm {
    key: i32,
    uid: u32,
    gid: u32,
    cuid: u32,
    cgid: u32,
    mode: u32,
    seq: u16,
}

impl Perm {
    fn new(key: i32, cx: &ServiceCtx, mode: u64, seq: u16) -> Self {
        let (uid, gid) = (cx.cur.creds.euid, cx.cur.creds.egid);
        Self {
            key,
            uid,
            gid,
            cuid: uid,
            cgid: gid,
            mode: (mode & 0o777) as u32,
            seq,
        }
    }

    /// `ipcperms`: may the caller access this object for `want` (`0o4` read,
    /// `0o2` write)? Root always may.
    fn allows(&self, cx: &ServiceCtx, want: u32) -> bool {
        let c = &cx.cur.creds;
        if c.euid == 0 {
            return true;
        }
        let bits = if c.euid == self.uid || c.euid == self.cuid {
            self.mode >> 6
        } else if c.egid == self.gid || c.egid == self.cgid {
            self.mode >> 3
        } else {
            self.mode
        };
        bits & want == want
    }

    /// May the caller change or remove the object (`IPC_SET`/`IPC_RMID`)?
    fn owner(&self, cx: &ServiceCtx) -> bool {
        let e = cx.cur.creds.euid;
        e == 0 || e == self.uid || e == self.cuid
    }

    /// Serialize as `struct ipc64_perm` (48 bytes).
    fn encode(&self, b: &mut [u8]) {
        b[0..4].copy_from_slice(&self.key.to_le_bytes());
        b[4..8].copy_from_slice(&self.uid.to_le_bytes());
        b[8..12].copy_from_slice(&self.gid.to_le_bytes());
        b[12..16].copy_from_slice(&self.cuid.to_le_bytes());
        b[16..20].copy_from_slice(&self.cgid.to_le_bytes());
        b[20..24].copy_from_slice(&self.mode.to_le_bytes());
        b[24..26].copy_from_slice(&self.seq.to_le_bytes());
    }

    /// Apply an `IPC_SET` from a `struct ipc64_perm` at `b`: uid, gid, and the
    /// permission bits of mode.
    fn set_from(&mut self, b: &[u8]) {
        self.uid = u32::from_le_bytes(b[4..8].try_into().unwrap());
        self.gid = u32::from_le_bytes(b[8..12].try_into().unwrap());
        let mode = u32::from_le_bytes(b[20..24].try_into().unwrap());
        self.mode = (self.mode & !0o777) | (mode & 0o777);
    }
}

fn now_s() -> i64 {
    crate::clock::now_unix().as_secs() as i64
}

/// One message queue.
#[derive(Debug)]
struct MsgQueue {
    perm: Perm,
    msgs: VecDeque<(i64, Vec<u8>)>,
    qbytes: u64,
    stime: i64,
    rtime: i64,
    ctime: i64,
    lspid: i32,
    lrpid: i32,
}

impl MsgQueue {
    fn cbytes(&self) -> u64 {
        self.msgs.iter().map(|m| m.1.len() as u64).sum()
    }
}

/// One semaphore set.
#[derive(Debug)]
struct SemSet {
    perm: Perm,
    vals: Vec<i32>,
    /// Pid of the last `semop` on each semaphore (`GETPID`).
    pids: Vec<i32>,
    otime: i64,
    ctime: i64,
}

/// One shared-memory segment.
#[derive(Debug)]
struct ShmSeg {
    perm: Perm,
    size: u64,
    /// The physical frames backing the segment (one reference each, owned by
    /// the segment; every attachment holds its own).
    frames: Vec<u64>,
    /// Current attachments: `(address space, address)`.
    attaches: Vec<(usize, u64)>,
    /// `IPC_RMID` was requested: destroy once the last attachment goes.
    removed: bool,
    atime: i64,
    dtime: i64,
    ctime: i64,
    cpid: i32,
    lpid: i32,
}

/// The machine's System V IPC namespace.
#[derive(Debug, Default)]
pub(super) struct Ipc {
    msq: BTreeMap<i32, MsgQueue>,
    sem: BTreeMap<i32, SemSet>,
    shm: BTreeMap<i32, ShmSeg>,
    /// Ids handed out so far, per kind; the next one is `index + seq * 32768`
    /// as on Linux, never reusing a live id.
    next: u32,
    /// Ids that were removed: a re-trapped waiter on one reports `EIDRM`
    /// rather than `EINVAL`.
    removed: BTreeSet<i32>,
    /// `SEM_UNDO` adjustments: `(tgid, semid, semnum) -> adj`, applied when the
    /// thread group exits.
    semadj: BTreeMap<(i32, i32, u16), i32>,
}

impl Ipc {
    fn alloc_id(&mut self) -> (i32, u16) {
        let n = self.next;
        self.next = self.next.wrapping_add(1);
        let seq = (n / 32768) as u16;
        (
            ((n % 32768) + u32::from(seq) * 32768) as i32 & i32::MAX,
            seq,
        )
    }

    /// Forget every attachment of address space `mm` (it is being torn down
    /// by `execve` or exit — the mapping itself goes with it), destroying any
    /// removed segment that thereby loses its last attachment.
    pub(super) fn detach_mm(&mut self, mm: usize, mem: &GuestMemory) {
        let mut dead = Vec::new();
        for (&id, s) in &mut self.shm {
            let before = s.attaches.len();
            s.attaches.retain(|a| a.0 != mm);
            if s.attaches.len() != before {
                s.dtime = now_s();
            }
            if s.removed && s.attaches.is_empty() {
                dead.push(id);
            }
        }
        for id in dead {
            if let Some(s) = self.shm.remove(&id) {
                mem.release_frames(&s.frames);
            }
        }
    }

    /// A `fork`: the child's copied address space `child` inherits every
    /// attachment of the parent's `parent` (the pages alias the same frames).
    pub(super) fn fork_mm(&mut self, parent: usize, child: usize) {
        for s in self.shm.values_mut() {
            let inherited: Vec<_> = s
                .attaches
                .iter()
                .filter(|a| a.0 == parent)
                .map(|a| (child, a.1))
                .collect();
            s.attaches.extend(inherited);
        }
    }

    /// An `munmap` of `[addr, addr + len)` in `mm` detaches any segment
    /// attached inside it (Linux counts that as a detach too).
    pub(super) fn unmapped(&mut self, mm: usize, addr: u64, len: u64, mem: &GuestMemory) {
        let end = addr.saturating_add(len);
        let mut dead = Vec::new();
        for (&id, s) in &mut self.shm {
            let size = s.size.div_ceil(PAGE_SIZE) * PAGE_SIZE;
            s.attaches
                .retain(|&(m, a)| !(m == mm && a < end && addr < a + size));
            if s.removed && s.attaches.is_empty() {
                dead.push(id);
            }
        }
        for id in dead {
            if let Some(s) = self.shm.remove(&id) {
                mem.release_frames(&s.frames);
            }
        }
    }

    /// Thread group `tgid` is gone: apply its `SEM_UNDO` adjustments.
    pub(super) fn exit_group(&mut self, tgid: i32) -> bool {
        let keys: Vec<_> = self
            .semadj
            .range((tgid, i32::MIN, 0)..=(tgid, i32::MAX, u16::MAX))
            .map(|(k, &v)| (*k, v))
            .collect();
        for ((_, id, num), adj) in &keys {
            if let Some(set) = self.sem.get_mut(id)
                && let Some(v) = set.vals.get_mut(*num as usize)
            {
                *v = (*v + adj).clamp(0, SEMVMX);
            }
            self.semadj.remove(&(tgid, *id, *num));
        }
        !keys.is_empty()
    }
}

/// Look up an object for `key`, or create it (the shared core of the three
/// `*get` calls). `create` builds the new object; `fits` vets an existing one
/// (`EINVAL` when e.g. the requested size exceeds it).
macro_rules! ipc_get {
    ($map:expr, $ipc:expr, $cx:expr, $key:expr, $flags:expr, $max:expr, $fits:expr, $create:expr) => {{
        let key = $key as i32;
        let existing = if key == IPC_PRIVATE {
            None
        } else {
            $map.iter()
                .find(|(_, o)| o.perm.key == key)
                .map(|(&id, _)| id)
        };
        match existing {
            Some(id) => {
                if $flags & IPC_CREAT != 0 && $flags & IPC_EXCL != 0 {
                    return err(Errno::EEXIST);
                }
                let o = &$map[&id];
                let want = ((($flags & 0o777) >> 6) & 0o6) as u32;
                if !o.perm.allows($cx, want) {
                    return err(Errno::EACCES);
                }
                if !$fits(o) {
                    return err(Errno::EINVAL);
                }
                i64::from(id)
            }
            None if key != IPC_PRIVATE && $flags & IPC_CREAT == 0 => err(Errno::ENOENT),
            None if $map.len() as i32 >= $max => err(Errno::ENOSPC),
            None => {
                let (id, seq) = $ipc.alloc_id();
                let perm = Perm::new(key, $cx, $flags, seq);
                match $create(perm) {
                    Ok(o) => {
                        $map.insert(id, o);
                        i64::from(id)
                    }
                    Err(e) => e,
                }
            }
        }
    }};
}

impl Kernel {
    /// The errno for an id that names no object: `EIDRM` if it was removed
    /// (a waiter's object vanished), else `EINVAL`.
    fn ipc_gone(ipc: &Ipc, id: i32) -> i64 {
        if ipc.removed.contains(&id) {
            err(Errno::EIDRM)
        } else {
            err(Errno::EINVAL)
        }
    }

    /// Park the caller in a System V wait (re-trap), never restarted.
    fn ipc_block(cx: &mut ServiceCtx) -> i64 {
        cx.block = true;
        cx.restartable = false;
        0
    }

    // ---- message queues ---------------------------------------------------

    /// `msgget(key, msgflg)`.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_msgget(&self, sh: &mut Shared, cx: &ServiceCtx, key: u64, flags: u64) -> i64 {
        let ipc = &mut sh.ipc;
        ipc_get!(
            ipc.msq,
            ipc,
            cx,
            key,
            flags,
            MSGMNI,
            |_: &MsgQueue| true,
            |perm| -> Result<MsgQueue, i64> {
                Ok(MsgQueue {
                    perm,
                    msgs: VecDeque::new(),
                    qbytes: MSGMNB,
                    stime: 0,
                    rtime: 0,
                    ctime: now_s(),
                    lspid: 0,
                    lrpid: 0,
                })
            }
        )
    }

    /// `msgsnd(msqid, msgp, msgsz, msgflg)`: append `{ long mtype; char
    /// mtext[msgsz]; }`, blocking (or `EAGAIN` with `IPC_NOWAIT`) while the
    /// queue's byte limit would be exceeded.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_msgsnd(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        a: &[u64; 6],
        mem: &GuestMemory,
    ) -> i64 {
        let (id, msgp, sz, flags) = (a[0] as i32, a[1], a[2], a[3]);
        if (sz as i64) < 0 || sz > MSGMAX || id < 0 {
            return err(Errno::EINVAL);
        }
        let Ok(mtype) = mem.read_u64(msgp) else {
            return err(Errno::EFAULT);
        };
        if (mtype as i64) < 1 {
            return err(Errno::EINVAL);
        }
        let Ok(text) = mem.read_vec(msgp + 8, sz as usize) else {
            return err(Errno::EFAULT);
        };
        let Some(q) = sh.ipc.msq.get_mut(&id) else {
            return Self::ipc_gone(&sh.ipc, id);
        };
        if !q.perm.allows(cx, 0o2) {
            return err(Errno::EACCES);
        }
        if q.cbytes() + sz > q.qbytes || q.msgs.len() as u64 >= q.qbytes {
            return if flags & IPC_NOWAIT != 0 {
                err(Errno::EAGAIN)
            } else {
                Self::ipc_block(cx)
            };
        }
        q.msgs.push_back((mtype as i64, text));
        q.stime = now_s();
        q.lspid = cx.cur.tgid;
        sh.unpark_all();
        0
    }

    /// `msgrcv(msqid, msgp, msgsz, msgtyp, msgflg)`: dequeue the first message
    /// selected by `msgtyp` (0: any; > 0: that type, or any other with
    /// `MSG_EXCEPT`; < 0: the lowest type ≤ |msgtyp|). A longer message is
    /// `E2BIG` unless `MSG_NOERROR` truncates it; none waiting blocks (or is
    /// `ENOMSG` with `IPC_NOWAIT`). `MSG_COPY` peeks at position `msgtyp`.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_msgrcv(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        a: &[u64; 6],
        mem: &mut GuestMemory,
    ) -> i64 {
        const MSG_NOERROR: u64 = 0o10000;
        const MSG_EXCEPT: u64 = 0o20000;
        const MSG_COPY: u64 = 0o40000;
        let (id, msgp, sz, typ, flags) = (a[0] as i32, a[1], a[2], a[3] as i64, a[4]);
        if (sz as i64) < 0 || id < 0 {
            return err(Errno::EINVAL);
        }
        if flags & MSG_COPY != 0 && (flags & IPC_NOWAIT == 0 || flags & MSG_EXCEPT != 0) {
            return err(Errno::EINVAL);
        }
        let Some(q) = sh.ipc.msq.get_mut(&id) else {
            return Self::ipc_gone(&sh.ipc, id);
        };
        if !q.perm.allows(cx, 0o4) {
            return err(Errno::EACCES);
        }
        let pick = if flags & MSG_COPY != 0 {
            usize::try_from(typ).ok().filter(|&i| i < q.msgs.len())
        } else if typ == 0 {
            (!q.msgs.is_empty()).then_some(0)
        } else if typ > 0 {
            q.msgs
                .iter()
                .position(|m| (m.0 == typ) != (flags & MSG_EXCEPT != 0))
        } else {
            let lim = typ.unsigned_abs() as i64;
            q.msgs
                .iter()
                .enumerate()
                .filter(|(_, m)| m.0 <= lim)
                .min_by_key(|(_, m)| m.0)
                .map(|(i, _)| i)
        };
        let Some(i) = pick else {
            return if flags & IPC_NOWAIT != 0 {
                err(Errno::ENOMSG)
            } else {
                Self::ipc_block(cx)
            };
        };
        let len = q.msgs[i].1.len() as u64;
        if len > sz && flags & MSG_NOERROR == 0 {
            return err(Errno::E2BIG);
        }
        let n = len.min(sz) as usize;
        let mut out = Vec::with_capacity(8 + n);
        out.extend_from_slice(&q.msgs[i].0.to_le_bytes());
        out.extend_from_slice(&q.msgs[i].1[..n]);
        if mem.write(msgp, &out).is_err() {
            return err(Errno::EFAULT);
        }
        if flags & MSG_COPY == 0 {
            q.msgs.remove(i);
            q.rtime = now_s();
            q.lrpid = cx.cur.tgid;
            sh.unpark_all();
        }
        n as i64
    }

    /// `msgctl(msqid, cmd, buf)`.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_msgctl(
        &self,
        sh: &mut Shared,
        cx: &ServiceCtx,
        id: u64,
        cmd: u64,
        buf: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const MSG_STAT: u64 = 11;
        const MSG_INFO: u64 = 12;
        const MSG_STAT_ANY: u64 = 13;
        let cmd = cmd & !IPC_64;
        let ipc = &mut sh.ipc;
        match cmd {
            IPC_INFO | MSG_INFO => {
                // struct msginfo: msgpool msgmap msgmax msgmnb msgmni msgssz
                // msgtql (int) msgseg (ushort).
                let (used, bytes) = ipc.msq.values().fold((0u64, 0u64), |(n, b), q| {
                    (n + q.msgs.len() as u64, b + q.cbytes())
                });
                let vals: [i32; 7] = if cmd == MSG_INFO {
                    [
                        ipc.msq.len() as i32,
                        used as i32,
                        MSGMAX as i32,
                        MSGMNB as i32,
                        MSGMNI,
                        16,
                        bytes as i32,
                    ]
                } else {
                    [
                        MSGMNI * MSGMNB as i32 / 1024,
                        MSGMNB as i32 / 16,
                        MSGMAX as i32,
                        MSGMNB as i32,
                        MSGMNI,
                        16,
                        MSGMNB as i32,
                    ]
                };
                let mut b = [0u8; 32];
                for (i, v) in vals.iter().enumerate() {
                    b[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
                b[28..30].copy_from_slice(&0xffffu16.to_le_bytes());
                if mem.write(buf, &b).is_err() {
                    return err(Errno::EFAULT);
                }
                ipc.msq.keys().copied().max().map_or(0, i64::from)
            }
            IPC_STAT | MSG_STAT | MSG_STAT_ANY => {
                let id = if cmd == IPC_STAT {
                    id as i32
                } else {
                    // *_STAT take a table index (here: the nth live queue).
                    match ipc.msq.keys().nth(id as usize) {
                        Some(&k) => k,
                        None => return err(Errno::EINVAL),
                    }
                };
                let Some(q) = ipc.msq.get(&id) else {
                    return err(Errno::EINVAL);
                };
                if cmd != MSG_STAT_ANY && !q.perm.allows(cx, 0o4) {
                    return err(Errno::EACCES);
                }
                let mut b = [0u8; 120];
                q.perm.encode(&mut b);
                let put = |b: &mut [u8; 120], o: usize, v: i64| {
                    b[o..o + 8].copy_from_slice(&v.to_le_bytes());
                };
                put(&mut b, 48, q.stime);
                put(&mut b, 56, q.rtime);
                put(&mut b, 64, q.ctime);
                put(&mut b, 72, q.cbytes() as i64);
                put(&mut b, 80, q.msgs.len() as i64);
                put(&mut b, 88, q.qbytes as i64);
                b[96..100].copy_from_slice(&q.lspid.to_le_bytes());
                b[100..104].copy_from_slice(&q.lrpid.to_le_bytes());
                if mem.write(buf, &b).is_err() {
                    return err(Errno::EFAULT);
                }
                if cmd == IPC_STAT { 0 } else { i64::from(id) }
            }
            IPC_SET => {
                let Ok(raw) = mem.read_vec(buf, 120) else {
                    return err(Errno::EFAULT);
                };
                let Some(q) = ipc.msq.get_mut(&(id as i32)) else {
                    return err(Errno::EINVAL);
                };
                if !q.perm.owner(cx) {
                    return err(Errno::EPERM);
                }
                let qbytes = u64::from_le_bytes(raw[88..96].try_into().unwrap());
                if qbytes > MSGMNB && cx.cur.creds.euid != 0 {
                    return err(Errno::EPERM);
                }
                q.perm.set_from(&raw);
                q.qbytes = qbytes;
                q.ctime = now_s();
                sh.unpark_all();
                0
            }
            IPC_RMID => {
                let id = id as i32;
                match ipc.msq.get(&id) {
                    None => err(Errno::EINVAL),
                    Some(q) if !q.perm.owner(cx) => err(Errno::EPERM),
                    Some(_) => {
                        ipc.msq.remove(&id);
                        ipc.removed.insert(id);
                        sh.unpark_all();
                        0
                    }
                }
            }
            _ => {
                self.note_unsupported("msgctl", cmd);
                err(Errno::EINVAL)
            }
        }
    }

    // ---- semaphores -------------------------------------------------------

    /// `semget(key, nsems, semflg)`.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_semget(
        &self,
        sh: &mut Shared,
        cx: &ServiceCtx,
        key: u64,
        nsems: u64,
        flags: u64,
    ) -> i64 {
        let nsems = nsems as i32;
        if nsems < 0 || nsems as u64 > SEMMSL {
            return err(Errno::EINVAL);
        }
        let ipc = &mut sh.ipc;
        ipc_get!(
            ipc.sem,
            ipc,
            cx,
            key,
            flags,
            SEMMNI,
            |s: &SemSet| nsems as usize <= s.vals.len(),
            |perm| -> Result<SemSet, i64> {
                if nsems == 0 {
                    return Err(err(Errno::EINVAL));
                }
                Ok(SemSet {
                    perm,
                    vals: vec![0; nsems as usize],
                    pids: vec![0; nsems as usize],
                    otime: 0,
                    ctime: now_s(),
                })
            }
        )
    }

    /// `semop`/`semtimedop(semid, sops, nsops[, timeout])`: apply every
    /// `struct sembuf { u16 sem_num; i16 sem_op; i16 sem_flg; }` atomically —
    /// all or nothing. If any operation can't proceed the caller blocks
    /// (`EAGAIN` if that operation has `IPC_NOWAIT`, or once the timeout
    /// passes); `SEM_UNDO` records the inverse for exit.
    #[allow(clippy::unused_self, clippy::too_many_lines)]
    pub(super) fn sys_semtimedop(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        a: &[u64; 6],
        timeout: u64,
        mem: &GuestMemory,
    ) -> i64 {
        const SEM_UNDO: i16 = 0o10000;
        let (id, sops, nsops) = (a[0] as i32, a[1], a[2]);
        if nsops == 0 || id < 0 {
            return err(Errno::EINVAL);
        }
        if nsops > SEMOPM {
            return err(Errno::E2BIG);
        }
        let Ok(raw) = mem.read_vec(sops, (nsops * 6) as usize) else {
            return err(Errno::EFAULT);
        };
        let ops: Vec<(u16, i16, i16)> = raw
            .as_chunks::<6>()
            .0
            .iter()
            .map(|c| {
                (
                    u16::from_le_bytes([c[0], c[1]]),
                    i16::from_le_bytes([c[2], c[3]]),
                    i16::from_le_bytes([c[4], c[5]]),
                )
            })
            .collect();
        // A finite timeout: seed the wall deadline once, reuse it on re-traps.
        let deadline = if timeout != 0 {
            if let Some(dl) = cx.cur.wake_deadline {
                Some(dl)
            } else {
                let (Ok(s), Ok(n)) = (mem.read_u64(timeout), mem.read_u64(timeout + 8)) else {
                    return err(Errno::EFAULT);
                };
                if (s as i64) < 0 || n >= 1_000_000_000 {
                    return err(Errno::EINVAL);
                }
                let dl = poll::now_ns() + u128::from(s) * 1_000_000_000 + u128::from(n);
                cx.cur.wake_deadline = Some(dl);
                Some(dl)
            }
        } else {
            None
        };
        let tgid = cx.cur.tgid;
        let Some(set) = sh.ipc.sem.get_mut(&id) else {
            return Self::ipc_gone(&sh.ipc, id);
        };
        let alters = ops.iter().any(|o| o.1 != 0);
        if !set.perm.allows(cx, if alters { 0o2 } else { 0o4 }) {
            return err(Errno::EACCES);
        }
        if ops.iter().any(|o| o.0 as usize >= set.vals.len()) {
            return err(Errno::EFBIG);
        }
        // Try the whole batch on a copy.
        let mut vals = set.vals.clone();
        let mut blocked = None;
        for (i, &(num, op, flg)) in ops.iter().enumerate() {
            let v = &mut vals[num as usize];
            if op > 0 {
                if *v + i32::from(op) > SEMVMX {
                    return err(Errno::ERANGE);
                }
                *v += i32::from(op);
            } else if op < 0 {
                if *v >= -i32::from(op) {
                    *v += i32::from(op);
                } else {
                    blocked = Some(i);
                    let _ = flg;
                    break;
                }
            } else if *v != 0 {
                blocked = Some(i);
                break;
            }
        }
        if let Some(i) = blocked {
            if ops[i].2 & IPC_NOWAIT as i16 != 0 {
                cx.cur.wake_deadline = None;
                return err(Errno::EAGAIN);
            }
            if deadline.is_some_and(|dl| poll::now_ns() >= dl) {
                cx.cur.wake_deadline = None;
                return err(Errno::EAGAIN);
            }
            cx.cur.sem_wait = Some((id, ops[i].0, ops[i].1 == 0));
            return Self::ipc_block(cx);
        }
        cx.cur.sem_wait = None;
        cx.cur.wake_deadline = None;
        set.vals = vals;
        for &(num, op, flg) in &ops {
            set.pids[num as usize] = tgid;
            if flg & SEM_UNDO != 0 && op != 0 {
                *sh.ipc.semadj.entry((tgid, id, num)).or_default() -= i32::from(op);
            }
        }
        if let Some(set) = sh.ipc.sem.get_mut(&id) {
            set.otime = now_s();
        }
        if alters {
            sh.unpark_all();
        }
        0
    }

    /// `semctl(semid, semnum, cmd, arg)` — `arg` is the `union semun` passed by
    /// value (an `int` for `SETVAL`, a pointer otherwise).
    #[allow(clippy::unused_self, clippy::too_many_lines)]
    pub(super) fn sys_semctl(
        &self,
        sh: &mut Shared,
        cx: &ServiceCtx,
        a: &[u64; 6],
        mem: &mut GuestMemory,
    ) -> i64 {
        const GETPID: u64 = 11;
        const GETVAL: u64 = 12;
        const GETALL: u64 = 13;
        const GETNCNT: u64 = 14;
        const GETZCNT: u64 = 15;
        const SETVAL: u64 = 16;
        const SETALL: u64 = 17;
        const SEM_STAT: u64 = 18;
        const SEM_INFO: u64 = 19;
        const SEM_STAT_ANY: u64 = 20;
        let (id, num, cmd, arg) = (a[0] as i32, a[1] as usize, a[2] & !IPC_64, a[3]);
        if matches!(cmd, IPC_INFO | SEM_INFO) {
            let ipc = &sh.ipc;
            // struct seminfo: semmap semmni semmns semmnu semmsl semopm semume
            // semusz semvmx semaem (10 ints).
            let total: usize = ipc.sem.values().map(|s| s.vals.len()).sum();
            let (semusz, semaem) = if cmd == SEM_INFO {
                (ipc.sem.len() as i32, total as i32)
            } else {
                (20, 16384)
            };
            let vals: [i32; 10] = [
                SEMMNI,
                SEMMNI,
                SEMMNI * SEMMSL as i32,
                SEMMNI,
                SEMMSL as i32,
                SEMOPM as i32,
                SEMOPM as i32,
                semusz,
                SEMVMX,
                semaem,
            ];
            let b: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
            if mem.write(arg, &b).is_err() {
                return err(Errno::EFAULT);
            }
            return ipc.sem.keys().copied().max().map_or(0, i64::from);
        }
        let real_id = if matches!(cmd, SEM_STAT | SEM_STAT_ANY) {
            match sh.ipc.sem.keys().nth(id as usize) {
                Some(&k) => k,
                None => return err(Errno::EINVAL),
            }
        } else {
            id
        };
        // Waiters counted from the parked tasks' recorded semaphore waits.
        let waiters = |zero: bool| -> i64 {
            sh.procs
                .iter()
                .flatten()
                .filter(|p| p.info.sem_wait == Some((id, num as u16, zero)) && p.info.parked)
                .count() as i64
        };
        let (ncnt, zcnt) = (waiters(false), waiters(true));
        let ipc = &mut sh.ipc;
        let Some(set) = ipc.sem.get_mut(&real_id) else {
            return Self::ipc_gone(ipc, real_id);
        };
        let n = set.vals.len();
        let need_num = matches!(cmd, GETVAL | GETPID | GETNCNT | GETZCNT | SETVAL);
        if need_num && num >= n {
            return err(Errno::EINVAL);
        }
        let read_perm = matches!(
            cmd,
            GETVAL | GETPID | GETNCNT | GETZCNT | GETALL | IPC_STAT | SEM_STAT
        );
        if read_perm && !set.perm.allows(cx, 0o4) {
            return err(Errno::EACCES);
        }
        if matches!(cmd, SETVAL | SETALL) && !set.perm.allows(cx, 0o2) {
            return err(Errno::EACCES);
        }
        match cmd {
            GETVAL => i64::from(set.vals[num]),
            GETPID => i64::from(set.pids[num]),
            GETNCNT => ncnt,
            GETZCNT => zcnt,
            GETALL => {
                let b: Vec<u8> = set
                    .vals
                    .iter()
                    .flat_map(|&v| (v as u16).to_le_bytes())
                    .collect();
                if mem.write(arg, &b).is_err() {
                    return err(Errno::EFAULT);
                }
                0
            }
            SETVAL => {
                let v = arg as i32;
                if !(0..=SEMVMX).contains(&v) {
                    return err(Errno::ERANGE);
                }
                set.vals[num] = v;
                set.ctime = now_s();
                ipc.semadj
                    .retain(|k, _| !(k.1 == id && k.2 as usize == num));
                sh.unpark_all();
                0
            }
            SETALL => {
                let Ok(raw) = mem.read_vec(arg, n * 2) else {
                    return err(Errno::EFAULT);
                };
                let vals: Vec<i32> = raw
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| i32::from(u16::from_le_bytes(*c)))
                    .collect();
                if vals.iter().any(|&v| v > SEMVMX) {
                    return err(Errno::ERANGE);
                }
                set.vals = vals;
                set.ctime = now_s();
                ipc.semadj.retain(|k, _| k.1 != id);
                sh.unpark_all();
                0
            }
            IPC_STAT | SEM_STAT | SEM_STAT_ANY => {
                // semid64_ds differs by arch: x86-64 pads after each time.
                let mut b = [0u8; 104];
                set.perm.encode(&mut b);
                let (ot, ct, ns, len) = match self.arch {
                    Arch::X86_64 => (48, 64, 80, 104),
                    Arch::Aarch64 => (48, 56, 64, 88),
                };
                b[ot..ot + 8].copy_from_slice(&set.otime.to_le_bytes());
                b[ct..ct + 8].copy_from_slice(&set.ctime.to_le_bytes());
                b[ns..ns + 8].copy_from_slice(&(n as u64).to_le_bytes());
                if mem.write(arg, &b[..len]).is_err() {
                    return err(Errno::EFAULT);
                }
                if cmd == IPC_STAT {
                    0
                } else {
                    i64::from(real_id)
                }
            }
            IPC_SET => {
                let Ok(raw) = mem.read_vec(arg, 48) else {
                    return err(Errno::EFAULT);
                };
                if !set.perm.owner(cx) {
                    return err(Errno::EPERM);
                }
                set.perm.set_from(&raw);
                set.ctime = now_s();
                0
            }
            IPC_RMID => {
                if !set.perm.owner(cx) {
                    return err(Errno::EPERM);
                }
                ipc.sem.remove(&id);
                ipc.removed.insert(id);
                ipc.semadj.retain(|k, _| k.1 != id);
                sh.unpark_all();
                0
            }
            _ => {
                self.note_unsupported("semctl", cmd);
                err(Errno::EINVAL)
            }
        }
    }

    // ---- shared memory ----------------------------------------------------

    /// `shmget(key, size, shmflg)`: the segment's frames are allocated up
    /// front from the shared pool (zeroed), so every attacher sees one copy.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_shmget(
        &self,
        sh: &mut Shared,
        cx: &ServiceCtx,
        key: u64,
        size: u64,
        flags: u64,
        mem: &GuestMemory,
    ) -> i64 {
        const SHM_HUGETLB: u64 = 0o4000;
        let ipc = &mut sh.ipc;
        let pid = cx.cur.tgid;
        ipc_get!(
            ipc.shm,
            ipc,
            cx,
            key,
            flags,
            SHMMNI,
            |s: &ShmSeg| size <= s.size,
            |perm| -> Result<ShmSeg, i64> {
                if !(SHMMIN..=SHMMAX).contains(&size) || flags & SHM_HUGETLB != 0 {
                    return Err(err(Errno::EINVAL));
                }
                let pages = size.div_ceil(PAGE_SIZE) as usize;
                let Some(frames) = mem.alloc_frames(pages) else {
                    return Err(err(Errno::ENOMEM));
                };
                Ok(ShmSeg {
                    perm,
                    size,
                    frames,
                    attaches: Vec::new(),
                    removed: false,
                    atime: 0,
                    dtime: 0,
                    ctime: now_s(),
                    cpid: pid,
                    lpid: 0,
                })
            }
        )
    }

    /// `shmat(shmid, shmaddr, shmflg)`: map the segment's frames into the
    /// caller — at a fresh address, or at `shmaddr` (rounded down with
    /// `SHM_RND`; an occupied range needs `SHM_REMAP`) — read-only with
    /// `SHM_RDONLY`, executable with `SHM_EXEC`.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_shmat(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        id: u64,
        addr: u64,
        flags: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const SHM_RDONLY: u64 = 0o10000;
        const SHM_RND: u64 = 0o20000;
        const SHM_REMAP: u64 = 0o40000;
        const SHM_EXEC: u64 = 0o100000;
        let id = id as i32;
        let Some(seg) = sh.ipc.shm.get(&id) else {
            return err(Errno::EINVAL);
        };
        let want = if flags & SHM_RDONLY != 0 { 0o4 } else { 0o6 };
        if !seg.perm.allows(cx, want) {
            return err(Errno::EACCES);
        }
        let len = seg.frames.len() as u64 * PAGE_SIZE;
        let frames = seg.frames.clone();
        let base = if addr == 0 {
            if flags & SHM_REMAP != 0 {
                return err(Errno::EINVAL);
            }
            match sh.arena(cx).alloc(len) {
                Some(b) => b,
                None => return err(Errno::ENOMEM),
            }
        } else {
            let base = if flags & SHM_RND != 0 {
                addr - addr % PAGE_SIZE
            } else if !addr.is_multiple_of(PAGE_SIZE) {
                return err(Errno::EINVAL);
            } else {
                addr
            };
            if flags & SHM_REMAP == 0 {
                let mut p = base;
                while p < base + len {
                    if mem.page_prot(p).is_some() {
                        return err(Errno::EINVAL);
                    }
                    p += PAGE_SIZE;
                }
            }
            sh.arena(cx).claim(base, len);
            base
        };
        let mut prot = if flags & SHM_RDONLY != 0 {
            Prot::READ
        } else {
            Prot::rw()
        };
        if flags & SHM_EXEC != 0 {
            prot = Prot(prot.0 | Prot::EXEC.0);
        }
        if mem.map_frames(base, &frames, prot).is_err() {
            return err(Errno::ENOMEM);
        }
        let mm = cx.cur.mm;
        let tgid = cx.cur.tgid;
        if let Some(seg) = sh.ipc.shm.get_mut(&id) {
            seg.attaches.push((mm, base));
            seg.atime = now_s();
            seg.lpid = tgid;
        }
        base as i64
    }

    /// `shmdt(shmaddr)`: detach the segment attached at exactly `shmaddr`.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_shmdt(
        &self,
        sh: &mut Shared,
        cx: &mut ServiceCtx,
        addr: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let mm = cx.cur.mm;
        let Some((&id, _)) = sh
            .ipc
            .shm
            .iter()
            .find(|(_, s)| s.attaches.contains(&(mm, addr)))
        else {
            return err(Errno::EINVAL);
        };
        let len = sh.ipc.shm[&id].frames.len() as u64 * PAGE_SIZE;
        let _ = mem.unmap(addr, len);
        sh.arena(cx).free_range(addr, len);
        let tgid = cx.cur.tgid;
        let seg = sh.ipc.shm.get_mut(&id).expect("found above");
        if let Some(i) = seg.attaches.iter().position(|a| *a == (mm, addr)) {
            seg.attaches.remove(i);
        }
        seg.dtime = now_s();
        seg.lpid = tgid;
        if seg.removed && seg.attaches.is_empty() {
            let seg = sh.ipc.shm.remove(&id).expect("present");
            mem.release_frames(&seg.frames);
        }
        0
    }

    /// `shmctl(shmid, cmd, buf)`.
    #[allow(clippy::unused_self, clippy::too_many_lines)]
    pub(super) fn sys_shmctl(
        &self,
        sh: &mut Shared,
        cx: &ServiceCtx,
        id: u64,
        cmd: u64,
        buf: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const SHM_LOCK: u64 = 11;
        const SHM_UNLOCK: u64 = 12;
        const SHM_STAT: u64 = 13;
        const SHM_INFO: u64 = 14;
        const SHM_STAT_ANY: u64 = 15;
        const SHM_DEST: u32 = 0o1000;
        let cmd = cmd & !IPC_64;
        let ipc = &mut sh.ipc;
        match cmd {
            IPC_INFO => {
                // struct shminfo64: shmmax shmmin shmmni shmseg shmall + 4 unused.
                let vals: [u64; 9] = [
                    SHMMAX,
                    SHMMIN,
                    SHMMNI as u64,
                    SHMMNI as u64,
                    SHMMAX / PAGE_SIZE,
                    0,
                    0,
                    0,
                    0,
                ];
                let b: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
                if mem.write(buf, &b).is_err() {
                    return err(Errno::EFAULT);
                }
                ipc.shm.keys().copied().max().map_or(0, i64::from)
            }
            SHM_INFO => {
                // struct shm_info: used_ids (int, padded), shm_tot, shm_rss,
                // shm_swp, swap_attempts, swap_successes.
                let pages: u64 = ipc.shm.values().map(|s| s.frames.len() as u64).sum();
                let mut b = [0u8; 48];
                b[0..4].copy_from_slice(&(ipc.shm.len() as i32).to_le_bytes());
                b[8..16].copy_from_slice(&pages.to_le_bytes());
                b[16..24].copy_from_slice(&pages.to_le_bytes());
                if mem.write(buf, &b).is_err() {
                    return err(Errno::EFAULT);
                }
                ipc.shm.keys().copied().max().map_or(0, i64::from)
            }
            IPC_STAT | SHM_STAT | SHM_STAT_ANY => {
                let id = if cmd == IPC_STAT {
                    id as i32
                } else {
                    match ipc.shm.keys().nth(id as usize) {
                        Some(&k) => k,
                        None => return err(Errno::EINVAL),
                    }
                };
                let Some(s) = ipc.shm.get(&id) else {
                    return err(Errno::EINVAL);
                };
                if cmd != SHM_STAT_ANY && !s.perm.allows(cx, 0o4) {
                    return err(Errno::EACCES);
                }
                let mut perm = s.perm.clone();
                if s.removed {
                    perm.mode |= SHM_DEST;
                }
                let mut b = [0u8; 112];
                perm.encode(&mut b);
                let put = |b: &mut [u8; 112], o: usize, v: i64| {
                    b[o..o + 8].copy_from_slice(&v.to_le_bytes());
                };
                put(&mut b, 48, s.size as i64);
                put(&mut b, 56, s.atime);
                put(&mut b, 64, s.dtime);
                put(&mut b, 72, s.ctime);
                b[80..84].copy_from_slice(&s.cpid.to_le_bytes());
                b[84..88].copy_from_slice(&s.lpid.to_le_bytes());
                put(&mut b, 88, s.attaches.len() as i64);
                if mem.write(buf, &b).is_err() {
                    return err(Errno::EFAULT);
                }
                if cmd == IPC_STAT { 0 } else { i64::from(id) }
            }
            IPC_SET => {
                let Ok(raw) = mem.read_vec(buf, 48) else {
                    return err(Errno::EFAULT);
                };
                let Some(s) = ipc.shm.get_mut(&(id as i32)) else {
                    return err(Errno::EINVAL);
                };
                if !s.perm.owner(cx) {
                    return err(Errno::EPERM);
                }
                s.perm.set_from(&raw);
                s.ctime = now_s();
                0
            }
            IPC_RMID => {
                let id = id as i32;
                let Some(s) = ipc.shm.get_mut(&id) else {
                    return err(Errno::EINVAL);
                };
                if !s.perm.owner(cx) {
                    return err(Errno::EPERM);
                }
                // The key is released immediately (a new shmget with it makes
                // a new segment); the memory lives until the last detach.
                s.removed = true;
                s.perm.key = IPC_PRIVATE;
                if s.attaches.is_empty() {
                    let s = ipc.shm.remove(&id).expect("present");
                    mem.release_frames(&s.frames);
                }
                0
            }
            SHM_LOCK | SHM_UNLOCK => {
                if ipc.shm.contains_key(&(id as i32)) {
                    0
                } else {
                    err(Errno::EINVAL)
                }
            }
            _ => {
                self.note_unsupported("shmctl", cmd);
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
    fn message_queue_send_receive_by_type() {
        let (k, mut mem, mut v, mut cx) = setup();
        let q = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Msgget,
            [0, 0o600, 0, 0, 0, 0],
        );
        assert!(q >= 0);
        let q = q as u64;
        let msg = BASE;
        for (t, body) in [(3u64, b"three"), (1, b"one!!"), (2, b"two!!")] {
            mem.write_u64(msg, t).unwrap();
            mem.write(msg + 8, body).unwrap();
            assert_eq!(
                call(
                    &k,
                    &mut cx,
                    &mut mem,
                    &mut v,
                    Sysno::Msgsnd,
                    [q, msg, 5, 0, 0, 0]
                ),
                0
            );
        }
        let out = BASE + 0x100;
        // Type 2 specifically.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Msgrcv,
                [q, out, 64, 2, 0, 0]
            ),
            5
        );
        assert_eq!(mem.read_vec(out + 8, 5).unwrap(), b"two!!");
        // Negative: the lowest type <= 3 is 1.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Msgrcv,
                [q, out, 64, (-3i64) as u64, 0, 0]
            ),
            5
        );
        assert_eq!(mem.read_u64(out).unwrap(), 1);
        // Too small without MSG_NOERROR: E2BIG; with it, truncated.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Msgrcv,
                [q, out, 2, 0, 0, 0]
            ),
            e(Errno::E2BIG)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Msgrcv,
                [q, out, 2, 0, 0o10000, 0]
            ),
            2
        );
        // Empty: IPC_NOWAIT → ENOMSG; blocking → parks.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Msgrcv,
                [q, out, 64, 0, 0o4000, 0]
            ),
            e(Errno::ENOMSG)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Msgrcv,
                [q, out, 64, 0, 0, 0]
            ),
            0
        );
        assert!(cx.block && !cx.restartable);
        cx.block = false;
        // IPC_STAT reports qbytes; RMID makes later calls EIDRM.
        let ds = BASE + 0x400;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Msgctl,
                [q, 2 | 0x100, ds, 0, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_u64(ds + 88).unwrap(), 16384);
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Msgctl,
                [q, 0, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Msgrcv,
                [q, out, 64, 0, 0, 0]
            ),
            e(Errno::EIDRM)
        );
    }

    #[test]
    fn keyed_get_create_exclusive_and_missing() {
        let (k, mut mem, mut v, mut cx) = setup();
        let key = 0x1234;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Semget,
                [key, 2, 0, 0, 0, 0]
            ),
            e(Errno::ENOENT)
        );
        let s = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Semget,
            [key, 2, 0o1600, 0, 0, 0],
        );
        assert!(s >= 0);
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Semget,
                [key, 2, 0, 0, 0, 0]
            ),
            s
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Semget,
                [key, 2, 0o3600, 0, 0, 0]
            ),
            e(Errno::EEXIST)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Semget,
                [key, 3, 0, 0, 0, 0]
            ),
            e(Errno::EINVAL)
        );
    }

    #[test]
    fn semaphores_are_atomic_and_undo_on_exit() {
        let (k, mut mem, mut v, mut cx) = setup();
        let s = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Semget,
            [0, 2, 0o600, 0, 0, 0],
        ) as u64;
        // SETVAL sem0 = 1.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Semctl,
                [s, 0, 16, 1, 0, 0]
            ),
            0
        );
        let sops = BASE;
        // {0, -1, 0} and {1, -1, IPC_NOWAIT}: the second can't proceed, so
        // nothing changes and the NOWAIT op makes it EAGAIN.
        let mut b = [0u8; 12];
        b[0..2].copy_from_slice(&0u16.to_le_bytes());
        b[2..4].copy_from_slice(&(-1i16).to_le_bytes());
        b[6..8].copy_from_slice(&1u16.to_le_bytes());
        b[8..10].copy_from_slice(&(-1i16).to_le_bytes());
        b[10..12].copy_from_slice(&0o4000i16.to_le_bytes());
        mem.write(sops, &b).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Semop,
                [s, sops, 2, 0, 0, 0]
            ),
            e(Errno::EAGAIN)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Semctl,
                [s, 0, 12, 0, 0, 0]
            ),
            1,
            "unchanged"
        );
        // A single decrement with SEM_UNDO succeeds…
        b[4..6].copy_from_slice(&0o10000i16.to_le_bytes());
        mem.write(sops, &b[..6]).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Semop,
                [s, sops, 1, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Semctl,
                [s, 0, 12, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Semctl,
                [s, 0, 11, 0, 0, 0]
            ),
            1,
            "GETPID"
        );
        // …and is undone when the thread group exits.
        assert!(k.shared.lock().unwrap().ipc.exit_group(1));
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Semctl,
                [s, 0, 12, 0, 0, 0]
            ),
            1
        );
        // A timed wait that can't proceed times out with EAGAIN.
        let ts = BASE + 0x100;
        mem.write(ts, &[0u8; 16]).unwrap();
        b[2..4].copy_from_slice(&(-5i16).to_le_bytes());
        b[4..6].copy_from_slice(&0i16.to_le_bytes());
        mem.write(sops, &b[..6]).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Semtimedop,
                [s, sops, 1, ts, 0, 0]
            ),
            e(Errno::EAGAIN)
        );
        // GETALL / SETALL / out-of-range values.
        mem.write(ts, &[7, 0, 9, 0]).unwrap();
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Semctl,
                [s, 0, 17, ts, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Semctl,
                [s, 1, 12, 0, 0, 0]
            ),
            9
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Semctl,
                [s, 0, 16, 40000, 0, 0]
            ),
            e(Errno::ERANGE)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Semctl,
                [s, 0, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Semop,
                [s, sops, 1, 0, 0, 0]
            ),
            e(Errno::EIDRM)
        );
    }

    #[test]
    fn shared_memory_is_really_shared_and_rmid_defers() {
        let (k, mut mem, mut v, mut cx) = setup();
        let id = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Shmget,
            [0, 10000, 0o600, 0, 0, 0],
        ) as u64;
        let a1 = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Shmat,
            [id, 0, 0, 0, 0, 0],
        ) as u64;
        let a2 = call(
            &k,
            &mut cx,
            &mut mem,
            &mut v,
            Sysno::Shmat,
            [id, 0, 0o10000, 0, 0, 0],
        ) as u64;
        assert_ne!(a1, a2);
        // A store through one attachment is visible through the other.
        mem.write(a1 + 8190, b"shared!").unwrap();
        assert_eq!(mem.read_vec(a2 + 8190, 7).unwrap(), b"shared!");
        // The read-only attachment really is read-only.
        assert!(mem.write(a2, b"x").is_err());
        // A forked address space aliases the same frames.
        let mut child = mem.fork();
        child.write(a1, b"from child").unwrap();
        assert_eq!(mem.read_vec(a1, 10).unwrap(), b"from child");
        child.release();
        let ds = BASE;
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Shmctl,
                [id, 2, ds, 0, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_u64(ds + 48).unwrap(), 10000, "shm_segsz");
        assert_eq!(mem.read_u64(ds + 88).unwrap(), 2, "shm_nattch");
        // RMID with attachments: still usable until the last detach.
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Shmctl,
                [id, 0, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Shmdt,
                [a2, 0, 0, 0, 0, 0]
            ),
            0
        );
        assert_eq!(mem.read_vec(a1, 10).unwrap(), b"from child");
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Shmdt,
                [a2, 0, 0, 0, 0, 0]
            ),
            e(Errno::EINVAL)
        );
        assert_eq!(
            call(
                &k,
                &mut cx,
                &mut mem,
                &mut v,
                Sysno::Shmdt,
                [a1, 0, 0, 0, 0, 0]
            ),
            0
        );
        assert!(
            k.shared.lock().unwrap().ipc.shm.is_empty(),
            "destroyed on last detach"
        );
        assert!(mem.read_vec(a1, 1).is_err(), "unmapped");
    }
}
