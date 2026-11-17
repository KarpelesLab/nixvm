//! Resource, scheduling, and process-attribute syscalls.
//!
//! These are mostly informational: they return success and/or write plausible
//! zeroed/static structs into guest memory. The guest's libc queries them at
//! startup (rlimits, cpu affinity, scheduler class, process name) and mostly
//! ignores the exact values, so a believable constant answer is enough.

use super::caps::{CAP_SETGID, CAP_SETUID};
use super::{Kernel, ServiceCtx, Shared, err};
use crate::abi::errno::Errno;
use crate::vcpu::GuestMemory;

/// A `set*id` argument's "leave unchanged" sentinel is `(uid_t)-1`, which the
/// guest passes zero-extended as `0xFFFF_FFFF`. Returns the new id, or `None`
/// to leave the field alone.
fn opt_id(v: u64) -> Option<u32> {
    let v = v as u32;
    (v != u32::MAX).then_some(v)
}

impl Kernel {
    /// `setuid(uid)`: with `CAP_SETUID` it sets the real, effective, saved and
    /// fs uid; without, only the effective (and fs) uid, to the real or saved
    /// one, else `EPERM`.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_setuid(&self, cx: &mut ServiceCtx, uid: u32) -> i64 {
        let privileged = cx.cur.caps.has(CAP_SETUID);
        let old = cx.cur.creds;
        let c = &mut cx.cur.creds;
        if privileged {
            c.ruid = uid;
            c.euid = uid;
            c.suid = uid;
            c.fsuid = uid;
        } else if uid == c.ruid || uid == c.suid {
            c.euid = uid;
            c.fsuid = uid;
        } else {
            return err(Errno::EPERM);
        }
        Self::uids_changed(cx, old);
        0
    }

    /// `setgid(gid)` — the group analogue of [`Self::sys_setuid`], with
    /// `CAP_SETGID`.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_setgid(&self, cx: &mut ServiceCtx, gid: u32) -> i64 {
        let privileged = cx.cur.caps.has(CAP_SETGID);
        let c = &mut cx.cur.creds;
        if privileged {
            c.rgid = gid;
            c.egid = gid;
            c.sgid = gid;
            c.fsgid = gid;
            0
        } else if gid == c.rgid || gid == c.sgid {
            c.egid = gid;
            c.fsgid = gid;
            0
        } else {
            err(Errno::EPERM)
        }
    }

    /// `setresuid(ruid, euid, suid)` — set each id independently (`-1` leaves it
    /// unchanged). With `CAP_SETUID` any; otherwise only among its current
    /// real/effective/saved uids.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_setresuid(&self, cx: &mut ServiceCtx, r: u64, e: u64, s: u64) -> i64 {
        let c = cx.cur.creds;
        let privileged = cx.cur.caps.has(CAP_SETUID);
        let ok = |id: u32| privileged || id == c.ruid || id == c.euid || id == c.suid;
        for id in [opt_id(r), opt_id(e), opt_id(s)].into_iter().flatten() {
            if !ok(id) {
                return err(Errno::EPERM);
            }
        }
        let nc = &mut cx.cur.creds;
        if let Some(v) = opt_id(r) {
            nc.ruid = v;
        }
        if let Some(v) = opt_id(e) {
            nc.euid = v;
            nc.fsuid = v;
        }
        if let Some(v) = opt_id(s) {
            nc.suid = v;
        }
        Self::uids_changed(cx, c);
        0
    }

    /// `setresgid(rgid, egid, sgid)` — the group analogue, with `CAP_SETGID`.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_setresgid(&self, cx: &mut ServiceCtx, r: u64, e: u64, s: u64) -> i64 {
        let c = cx.cur.creds;
        let privileged = cx.cur.caps.has(CAP_SETGID);
        let ok = |id: u32| privileged || id == c.rgid || id == c.egid || id == c.sgid;
        for id in [opt_id(r), opt_id(e), opt_id(s)].into_iter().flatten() {
            if !ok(id) {
                return err(Errno::EPERM);
            }
        }
        let c = &mut cx.cur.creds;
        if let Some(v) = opt_id(r) {
            c.rgid = v;
        }
        if let Some(v) = opt_id(e) {
            c.egid = v;
            c.fsgid = v;
        }
        if let Some(v) = opt_id(s) {
            c.sgid = v;
        }
        0
    }

    /// `setreuid(ruid, euid)` — set real and/or effective uid (`-1` leaves one
    /// unchanged); the saved uid follows the effective uid when either changes.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_setreuid(&self, cx: &mut ServiceCtx, r: u64, e: u64) -> i64 {
        let c = cx.cur.creds;
        let privileged = cx.cur.caps.has(CAP_SETUID);
        if let Some(r) = opt_id(r)
            && !(privileged || r == c.ruid || r == c.euid)
        {
            return err(Errno::EPERM);
        }
        if let Some(e) = opt_id(e)
            && !(privileged || e == c.ruid || e == c.euid || e == c.suid)
        {
            return err(Errno::EPERM);
        }
        let nc = &mut cx.cur.creds;
        if let Some(r) = opt_id(r) {
            nc.ruid = r;
        }
        if let Some(e) = opt_id(e) {
            nc.euid = e;
            nc.fsuid = e;
        }
        // The saved uid tracks the effective uid when the real uid is set or the
        // effective uid is changed to something other than the old real uid.
        if opt_id(r).is_some() || opt_id(e).is_some_and(|e| e != c.ruid) {
            nc.suid = nc.euid;
        }
        Self::uids_changed(cx, c);
        0
    }

    /// `setregid(rgid, egid)` — the group analogue of [`Self::sys_setreuid`].
    #[allow(clippy::unused_self)]
    pub(super) fn sys_setregid(&self, cx: &mut ServiceCtx, r: u64, e: u64) -> i64 {
        let c = cx.cur.creds;
        let privileged = cx.cur.caps.has(CAP_SETGID);
        if let Some(r) = opt_id(r)
            && !(privileged || r == c.rgid || r == c.egid)
        {
            return err(Errno::EPERM);
        }
        if let Some(e) = opt_id(e)
            && !(privileged || e == c.rgid || e == c.egid || e == c.sgid)
        {
            return err(Errno::EPERM);
        }
        let nc = &mut cx.cur.creds;
        if let Some(r) = opt_id(r) {
            nc.rgid = r;
        }
        if let Some(e) = opt_id(e) {
            nc.egid = e;
            nc.fsgid = e;
        }
        if opt_id(r).is_some() || opt_id(e).is_some_and(|e| e != c.rgid) {
            nc.sgid = nc.egid;
        }
        0
    }

    /// `setfsuid(fsuid)` / `setfsgid(fsgid)` — set the filesystem id, returning
    /// the *previous* one. Allowed with `CAP_SETUID` or if the id matches the
    /// caller's real/effective/saved/current-fs id; otherwise the change is
    /// silently ignored (the syscall never fails, it just returns the old value).
    #[allow(clippy::unused_self)]
    pub(super) fn sys_setfsuid(&self, cx: &mut ServiceCtx, fsuid: u64) -> i64 {
        let privileged = cx.cur.caps.has(CAP_SETUID);
        let c = &mut cx.cur.creds;
        let old = c.fsuid;
        if let Some(v) = opt_id(fsuid)
            && (privileged || v == c.ruid || v == c.euid || v == c.suid || v == c.fsuid)
        {
            c.fsuid = v;
            cx.cur.caps.after_setfsuid(old, v);
        }
        i64::from(old)
    }

    /// `setfsgid(fsgid)` — the group analogue of [`Self::sys_setfsuid`].
    #[allow(clippy::unused_self)]
    pub(super) fn sys_setfsgid(&self, cx: &mut ServiceCtx, fsgid: u64) -> i64 {
        let privileged = cx.cur.caps.has(CAP_SETGID);
        let c = &mut cx.cur.creds;
        let old = c.fsgid;
        if let Some(v) = opt_id(fsgid)
            && (privileged || v == c.rgid || v == c.egid || v == c.sgid || v == c.fsgid)
        {
            c.fsgid = v;
        }
        i64::from(old)
    }

    /// The capability fixups after the uids changed from `old`'s.
    fn uids_changed(cx: &mut ServiceCtx, old: super::Creds) {
        let new = cx.cur.creds;
        cx.cur.caps.after_setuid(
            (old.ruid, old.euid, old.suid, old.fsuid),
            (new.ruid, new.euid, new.suid, new.fsuid),
        );
    }

    /// `sched_get_priority_min`/`_max(policy)` — the valid real-time priority
    /// range. `SCHED_FIFO`(1)/`SCHED_RR`(2) use `1..=99`; the others (OTHER/
    /// BATCH/IDLE) have no RT priority, so both bounds are 0.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_sched_priority_bound(&self, policy: u64, max: bool) -> i64 {
        match policy {
            1 | 2 => i64::from(max) * 98 + 1, // 1 (min) or 99 (max)
            _ => 0,
        }
    }
}

impl Kernel {
    /// `setgroups(size, list)`: replace the supplementary groups — root only
    /// (`EPERM`), at most `NGROUPS_MAX` (65536) of them (`EINVAL`).
    #[allow(clippy::unused_self)]
    pub(super) fn sys_setgroups(
        &self,
        cx: &mut ServiceCtx,
        size: u64,
        list: u64,
        mem: &GuestMemory,
    ) -> i64 {
        if !cx.cur.caps.has(super::caps::CAP_SETGID) {
            return err(Errno::EPERM);
        }
        if size > 65536 {
            return err(Errno::EINVAL);
        }
        if size == 0 {
            cx.cur.groups.clear();
            return 0;
        }
        let Ok(raw) = mem.read_vec(list, size as usize * 4) else {
            return err(Errno::EFAULT);
        };
        // Kept as the VM sees them: each must be mapped where the caller is.
        let ns = cx.cur.ns.user;
        let users = self.users.lock().unwrap();
        let groups: Option<Vec<u32>> = raw
            .chunks(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .map(|g| users.to_global(ns, super::userns::IdKind::Gid, g))
            .collect();
        drop(users);
        match groups {
            Some(groups) => {
                cx.cur.groups = groups;
                0
            }
            None => err(Errno::EINVAL),
        }
    }

    /// The supplementary groups as the caller's user namespace sees them, the
    /// unmapped ones as the overflow gid.
    pub(super) fn visible_groups(&self, cx: &ServiceCtx) -> Vec<u32> {
        let ns = cx.cur.ns.user;
        if ns == 0 {
            return cx.cur.groups.clone();
        }
        let users = self.users.lock().unwrap();
        cx.cur
            .groups
            .iter()
            .map(|&g| users.seen_from_global(ns, super::userns::IdKind::Gid, g))
            .collect()
    }

    /// `getgroups(size, list)`: the supplementary groups; `size == 0` asks
    /// only for the count, a too-small `size` is `EINVAL`.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_getgroups(
        &self,
        cx: &ServiceCtx,
        size: u64,
        list: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        let n = cx.cur.groups.len();
        if size == 0 {
            return n as i64;
        }
        // An int: a "negative" size (top bit of the low 32) is EINVAL too.
        if size as u32 > i32::MAX as u32 || (size as usize) < n {
            return err(Errno::EINVAL);
        }
        let groups = self.visible_groups(cx);
        let b: Vec<u8> = groups.iter().flat_map(|g| g.to_le_bytes()).collect();
        if mem.write(list, &b).is_err() {
            return err(Errno::EFAULT);
        }
        n as i64
    }

    /// `syslog(type, buf, len)` — the kernel log. There is no kernel ring
    /// buffer: reads return nothing, the buffer size is reported, the console
    /// controls are accepted. Unknown actions are `EINVAL`; everything but the
    /// harmless size/unread queries needs root.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_syslog(
        &self,
        cx: &ServiceCtx,
        action: u64,
        _buf: u64,
        len: u64,
        _mem: &GuestMemory,
    ) -> i64 {
        const READ: u64 = 2;
        const READ_ALL: u64 = 3;
        const READ_CLEAR: u64 = 4;
        const SIZE_BUFFER: u64 = 10;
        let action = action as i32;
        if !(0..=10).contains(&action) {
            return err(Errno::EINVAL);
        }
        let action = action as u64;
        if !Self::capable(cx, super::caps::CAP_SYSLOG)
            && !Self::capable(cx, super::caps::CAP_SYS_ADMIN)
            && !matches!(action, READ_ALL | SIZE_BUFFER)
        {
            return err(Errno::EPERM);
        }
        match action {
            READ | READ_ALL | READ_CLEAR if len as u32 > i32::MAX as u32 => err(Errno::EINVAL),
            SIZE_BUFFER => 1 << 17,
            // Nothing unread, nothing to read, console controls accepted.
            _ => 0,
        }
    }

    /// x86-64 `arch_prctl(code, addr)`: the FS/GS bases and the CPUID and
    /// XSAVE-component controls. `ARCH_SET_FS` installs the TLS register and
    /// `ARCH_SET_GS` the GS base (Wine's TEB pointer; refused with `EINVAL`
    /// on a backend without one); CPUID faulting isn't
    /// available (`ENODEV` to enable it); the XSAVE queries report x87+SSE.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_arch_prctl(
        &self,
        cx: &mut ServiceCtx,
        code: u64,
        addr: u64,
        vcpu: &mut dyn crate::vcpu::Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        const ARCH_SET_GS: u64 = 0x1001;
        const ARCH_SET_FS: u64 = 0x1002;
        const ARCH_GET_FS: u64 = 0x1003;
        const ARCH_GET_GS: u64 = 0x1004;
        const ARCH_GET_CPUID: u64 = 0x1011;
        const ARCH_SET_CPUID: u64 = 0x1012;
        const ARCH_GET_XCOMP_SUPP: u64 = 0x1021;
        const ARCH_GET_XCOMP_PERM: u64 = 0x1022;
        const ARCH_REQ_XCOMP_PERM: u64 = 0x1023;
        const XFEATURES_X87_SSE: u64 = 3;
        let put = |mem: &mut GuestMemory, v: u64| {
            if mem.write_u64(addr, v).is_ok() {
                0
            } else {
                err(Errno::EFAULT)
            }
        };
        match code {
            ARCH_SET_FS => {
                vcpu.set_tls(addr);
                cx.cur.fs_base = addr;
                0
            }
            ARCH_GET_FS => put(mem, cx.cur.fs_base),
            ARCH_GET_GS => put(mem, vcpu.gs_base()),
            ARCH_GET_CPUID => 1,
            ARCH_SET_CPUID => {
                if addr == 1 {
                    0
                } else {
                    err(Errno::ENODEV)
                }
            }
            ARCH_GET_XCOMP_SUPP | ARCH_GET_XCOMP_PERM => put(mem, XFEATURES_X87_SSE),
            ARCH_REQ_XCOMP_PERM => {
                if addr < 2 {
                    0
                } else {
                    err(Errno::EOPNOTSUPP)
                }
            }
            // A canonical user address, as Linux checks (`TASK_SIZE_MAX`).
            ARCH_SET_GS if addr >= 0x7FFF_FFFF_F000 => err(Errno::EPERM),
            ARCH_SET_GS => {
                if vcpu.set_gs_base(addr) {
                    0
                } else {
                    err(Errno::EINVAL)
                }
            }
            other => {
                self.note_unsupported("arch_prctl", other);
                err(Errno::EINVAL)
            }
        }
    }

    /// i386 `set_thread_area(u_info)`: install one of the thread's three GDT
    /// TLS entries (12-14) from a `struct user_desc`, as Linux's
    /// `do_set_thread_area`. `entry_number` -1 takes the first empty one and
    /// writes its number back. Only data segments that are 32-bit and present
    /// are allowed, or an empty entry (`LDT_empty`/`LDT_zero`), which clears
    /// it. A segment register holding the entry's selector is reloaded.
    pub(super) fn sys_set_thread_area(
        u_info: u64,
        vcpu: &mut dyn crate::vcpu::Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        Self::do_set_thread_area(u_info, true, vcpu, mem)
    }

    /// The TLS of an i386 `clone`'s child (`CLONE_SETTLS`): the GDT entry
    /// `u_info` describes, which it must name (`set_new_tls`).
    pub(super) fn clone_thread_area(
        u_info: u64,
        vcpu: &mut dyn crate::vcpu::Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        Self::do_set_thread_area(u_info, false, vcpu, mem)
    }

    /// Linux's `do_set_thread_area`: entry -1 picks a free one only when
    /// `can_allocate`.
    fn do_set_thread_area(
        u_info: u64,
        can_allocate: bool,
        vcpu: &mut dyn crate::vcpu::Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        let mut raw = [0u8; 16];
        if mem.read(u_info, &mut raw).is_err() {
            return err(Errno::EFAULT);
        }
        let info = UserDesc::parse(&raw);
        if !info.tls_okay() {
            return err(Errno::EINVAL);
        }
        let mut idx = info.entry_number;
        if idx == u32::MAX && can_allocate {
            let Some(free) = (0..TLS_ENTRIES).find(|&i| vcpu.tls_desc(i) == 0) else {
                return err(Errno::ESRCH);
            };
            idx = (TLS_MIN + free) as u32;
            if mem.write(u_info, &idx.to_le_bytes()).is_err() {
                return err(Errno::EFAULT);
            }
        }
        let idx = idx as usize;
        if !(TLS_MIN..TLS_MIN + TLS_ENTRIES).contains(&idx) {
            return err(Errno::EINVAL);
        }
        if vcpu.set_tls_desc(idx - TLS_MIN, info.descriptor()) {
            0
        } else {
            err(Errno::ENOMEM)
        }
    }

    /// i386 `get_thread_area(u_info)`: the `struct user_desc` of TLS entry
    /// `u_info->entry_number` (Linux's `fill_user_desc`).
    pub(super) fn sys_get_thread_area(
        u_info: u64,
        vcpu: &mut dyn crate::vcpu::Vcpu,
        mem: &mut GuestMemory,
    ) -> i64 {
        let Ok(idx) = mem.read_u32(u_info) else {
            return err(Errno::EFAULT);
        };
        let idx = idx as usize;
        if !(TLS_MIN..TLS_MIN + TLS_ENTRIES).contains(&idx) {
            return err(Errno::EINVAL);
        }
        let out = UserDesc::from_descriptor(idx as u32, vcpu.tls_desc(idx - TLS_MIN));
        if mem.write(u_info, &out.to_bytes()).is_err() {
            return err(Errno::EFAULT);
        }
        0
    }
}

/// `GDT_ENTRY_TLS_MIN` and the number of TLS entries.
const TLS_MIN: usize = crate::vcpu::ctrl::GDT_ENTRY_TLS_MIN;
const TLS_ENTRIES: usize = crate::vcpu::ctrl::GDT_TLS_ENTRIES;

/// `struct user_desc` (`asm/ldt.h`): a segment as `set_thread_area` and
/// `modify_ldt` take it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(clippy::struct_excessive_bools)] // the struct's own bit fields
struct UserDesc {
    entry_number: u32,
    base_addr: u32,
    limit: u32,
    seg_32bit: bool,
    contents: u8,
    read_exec_only: bool,
    limit_in_pages: bool,
    seg_not_present: bool,
    useable: bool,
    lm: bool,
}

impl UserDesc {
    fn parse(b: &[u8; 16]) -> Self {
        let word = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let f = word(12);
        Self {
            entry_number: word(0),
            base_addr: word(4),
            limit: word(8),
            seg_32bit: f & 1 != 0,
            contents: ((f >> 1) & 3) as u8,
            read_exec_only: f & 8 != 0,
            limit_in_pages: f & 16 != 0,
            seg_not_present: f & 32 != 0,
            useable: f & 64 != 0,
            lm: f & 128 != 0,
        }
    }

    fn to_bytes(self) -> [u8; 16] {
        let f = u32::from(self.seg_32bit)
            | u32::from(self.contents) << 1
            | u32::from(self.read_exec_only) << 3
            | u32::from(self.limit_in_pages) << 4
            | u32::from(self.seg_not_present) << 5
            | u32::from(self.useable) << 6
            | u32::from(self.lm) << 7;
        let mut b = [0u8; 16];
        for (i, w) in [self.entry_number, self.base_addr, self.limit, f]
            .iter()
            .enumerate()
        {
            b[i * 4..i * 4 + 4].copy_from_slice(&w.to_le_bytes());
        }
        b
    }

    /// `LDT_empty` or `LDT_zero`: "no segment at all".
    fn is_empty(&self) -> bool {
        let none = self.base_addr == 0
            && self.limit == 0
            && self.contents == 0
            && !self.seg_32bit
            && !self.limit_in_pages
            && !self.useable;
        none && ((self.read_exec_only && self.seg_not_present)
            || (!self.read_exec_only && !self.seg_not_present))
    }

    /// `tls_desc_okay`: empty, or a present 32-bit data segment.
    fn tls_okay(&self) -> bool {
        self.is_empty() || (self.seg_32bit && self.contents <= 1 && !self.seg_not_present)
    }

    /// The 8-byte descriptor (`fill_ldt`; 0 for an empty one). The `lm` bit
    /// is never set.
    fn descriptor(&self) -> u64 {
        if self.is_empty() {
            return 0;
        }
        let base = u64::from(self.base_addr);
        let limit = u64::from(self.limit);
        let ty = (u64::from(!self.read_exec_only) << 1) | (u64::from(self.contents) << 2) | 1;
        (limit & 0xFFFF)
            | (base & 0xFF_FFFF) << 16
            | ty << 40
            | 1 << 44 // S: code/data
            | 3 << 45 // DPL 3
            | u64::from(!self.seg_not_present) << 47
            | ((limit >> 16) & 0xF) << 48
            | u64::from(self.useable) << 52
            | u64::from(self.seg_32bit) << 54
            | u64::from(self.limit_in_pages) << 55
            | (base >> 24) << 56
    }

    /// `fill_user_desc`: entry `idx` as `get_thread_area` shows it.
    fn from_descriptor(idx: u32, d: u64) -> Self {
        let ty = (d >> 40) & 0xF;
        Self {
            entry_number: idx,
            base_addr: (((d >> 16) & 0xFF_FFFF) | ((d >> 56) << 24)) as u32,
            limit: ((d & 0xFFFF) | ((d >> 32) & 0xF_0000)) as u32,
            seg_32bit: d & (1 << 54) != 0,
            contents: (ty >> 2) as u8,
            read_exec_only: ty & 2 == 0,
            limit_in_pages: d & (1 << 55) != 0,
            seg_not_present: d & (1 << 47) == 0,
            useable: d & (1 << 52) != 0,
            lm: d & (1 << 53) != 0,
        }
    }
}

#[cfg(test)]
mod user_desc_tests {
    use super::UserDesc;

    /// Wine's `%fs` for 32-bit code: base = the 32-bit TEB, a page limit,
    /// 32-bit. The descriptor is the one Linux's `fill_ldt` builds, and
    /// `get_thread_area` gives it back as it was set.
    #[test]
    fn a_tls_segment_round_trips_through_its_descriptor() {
        let u = UserDesc {
            entry_number: 12,
            base_addr: 0x1234_5678,
            limit: 0xFFF,
            seg_32bit: true,
            ..UserDesc::default()
        };
        assert!(u.tls_okay());
        let d = u.descriptor();
        assert_eq!(d, 0x1240_f334_5678_0fff);
        assert_eq!(UserDesc::from_descriptor(12, d), u);
        assert_eq!(UserDesc::parse(&u.to_bytes()), u);
    }

    #[test]
    fn empty_entries_and_bad_segments() {
        let zero = UserDesc::default();
        assert!(zero.tls_okay());
        assert_eq!(zero.descriptor(), 0);
        let empty = UserDesc {
            read_exec_only: true,
            seg_not_present: true,
            ..UserDesc::default()
        };
        assert_eq!(empty.descriptor(), 0);
        // 16-bit, code, or not-present segments are refused.
        let base = UserDesc {
            base_addr: 0x1000,
            limit: 1,
            seg_32bit: true,
            ..UserDesc::default()
        };
        assert!(base.tls_okay());
        assert!(
            !UserDesc {
                seg_32bit: false,
                ..base
            }
            .tls_okay()
        );
        assert!(
            !UserDesc {
                contents: 2,
                ..base
            }
            .tls_okay()
        );
        assert!(
            !UserDesc {
                seg_not_present: true,
                ..base
            }
            .tls_okay()
        );
    }
}

/// `sched_getparam(pid, param)` — write a `sched_param { sched_priority = 0 }`.
pub(super) fn sys_sched_getparam(priority: i32, param: u64, mem: &mut GuestMemory) -> i64 {
    if mem.write(param, &priority.to_le_bytes()).is_err() {
        return err(Errno::EFAULT);
    }
    0
}

impl Kernel {
    /// `getrusage(who, buf)` — report the CPU time consumed so far in a `struct
    /// rusage` (144 bytes). The CPU time comes from the scheduler's per-task
    /// accounting — the *same* source as `clock_gettime(CLOCK_PROCESS_CPUTIME_ID)`
    /// so the two agree — landing in `ru_utime` (we don't split guest user vs.
    /// system time); the remaining counters (maxrss, faults, context switches)
    /// stay zero. `RUSAGE_CHILDREN` reports nothing (child accounting isn't
    /// tracked). Called with `sh` held (the B1 dispatch table).
    #[allow(clippy::unused_self)]
    pub(super) fn sys_getrusage(
        &self,
        sh: &Shared,
        cx: &ServiceCtx,
        who: u64,
        buf: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        // `who` is a 32-bit `int`: RUSAGE_SELF(0), RUSAGE_CHILDREN(-1),
        // RUSAGE_THREAD(1). Read it via `as i32` so a `-1` the guest passed as a
        // zero-extended `0xFFFF_FFFF` is recognized.
        let cpu_ns = match who as i32 {
            1 => cx.cur.cpu_ns,                 // RUSAGE_THREAD
            -1 => cx.cur.child_cpu_ns,          // RUSAGE_CHILDREN: reaped children's CPU
            _ => super::process_cpu_ns(sh, cx), // RUSAGE_SELF (0) and anything else
        };
        if mem.write(buf, &super::rusage_bytes(cpu_ns)).is_err() {
            return err(Errno::EFAULT);
        }
        0
    }

    /// `times(buf)` — report process CPU time in a `struct tms` (4 × i64 clock
    /// ticks) and return elapsed real time in ticks. Ticks are `USER_HZ` = 100 Hz
    /// (10 ms), as on Linux. `tms_utime` carries the process CPU time (the same
    /// per-task accounting as `getrusage`/`clock_gettime`); system and children
    /// fields are zero. Called with `sh` held.
    #[allow(clippy::unused_self)]
    pub(super) fn sys_times(
        &self,
        sh: &Shared,
        cx: &ServiceCtx,
        buf: u64,
        mem: &mut GuestMemory,
    ) -> i64 {
        const TICK_NS: u128 = 10_000_000; // 1 tick = 10 ms (USER_HZ = 100)
        let mut tms = [0u8; 32];
        tms[0..8]
            .copy_from_slice(&((super::process_cpu_ns(sh, cx) / TICK_NS) as i64).to_le_bytes()); // tms_utime
        // tms_stime stays zero; tms_cutime carries reaped children's CPU.
        tms[16..24].copy_from_slice(&((cx.cur.child_cpu_ns / TICK_NS) as i64).to_le_bytes()); // tms_cutime
        // tms_cstime stays zero.
        if mem.write(buf, &tms).is_err() {
            return err(Errno::EFAULT);
        }
        (crate::clock::now_monotonic().as_nanos() / TICK_NS) as i64 // real elapsed ticks
    }
}

/// `sysinfo(buf)` — write a `struct sysinfo` with 2 GiB total RAM, one process,
/// and `mem_unit = 1`.
pub(super) fn sys_sysinfo(buf: u64, mem: &mut GuestMemory) -> i64 {
    let data = encode_sysinfo();
    if mem.write(buf, &data).is_err() {
        return err(Errno::EFAULT);
    }
    0
}

/// `getcpu(cpu, node, tcache)` — always CPU 0 / NUMA node 0.
pub(super) fn sys_getcpu(on: u16, cpu: u64, node: u64, mem: &mut GuestMemory) -> i64 {
    if cpu != 0 && mem.write(cpu, &u32::from(on).to_le_bytes()).is_err() {
        return err(Errno::EFAULT);
    }
    if node != 0 && mem.write(node, &0u32.to_le_bytes()).is_err() {
        return err(Errno::EFAULT);
    }
    0
}

pub(super) const RLIMIT_NOFILE: u64 = 7;
/// The largest `RLIMIT_NOFILE` hard limit we'll grant. This bounds the fd
/// space a program believes it has — node/V8 binary-search-raise it and then
/// loop over `[0, soft)` marking every fd cloexec, so an unbounded raise turns
/// startup into a million-iteration spin.
pub(super) const NOFILE_HARD_CAP: u64 = 4096;

pub(super) const RLIMIT_CPU: u64 = 0;
pub(super) const RLIMIT_FSIZE: u64 = 1;
pub(super) const RLIMIT_DATA: u64 = 2;
pub(super) const RLIMIT_STACK: u64 = 3;
pub(super) const RLIMIT_CORE: u64 = 4;
pub(super) const RLIMIT_NPROC: u64 = 6;
pub(super) const RLIMIT_MEMLOCK: u64 = 8;
pub(super) const RLIMIT_AS: u64 = 9;
pub(super) const RLIMIT_SIGPENDING: u64 = 11;
pub(super) const RLIMIT_MSGQUEUE: u64 = 12;
pub(super) const RLIMIT_NICE: u64 = 13;
pub(super) const RLIMIT_RTPRIO: u64 = 14;
pub(super) const RLIMIT_RTTIME: u64 = 15;
/// How many resources there are (`RLIM_NLIMITS`).
pub(super) const RLIM_NLIMITS: usize = 16;
pub(super) const RLIM_INFINITY: u64 = u64::MAX;

/// The limits a new VM's first process starts with.
pub(super) fn default_rlimits() -> [(u64, u64); RLIM_NLIMITS] {
    let mut limits = [(RLIM_INFINITY, RLIM_INFINITY); RLIM_NLIMITS];
    limits[RLIMIT_STACK as usize] = (8 * 1024 * 1024, RLIM_INFINITY);
    limits[RLIMIT_NPROC as usize] = (4096, 4096);
    limits[RLIMIT_NOFILE as usize] = (1024, NOFILE_HARD_CAP);
    limits[RLIMIT_SIGPENDING as usize] = (4096, 4096);
    limits[RLIMIT_MSGQUEUE as usize] = (819_200, 819_200);
    // Linux's defaults: no core files, 8 MiB of locked memory, and no raising
    // of priority or real-time scheduling without CAP_SYS_NICE.
    limits[RLIMIT_CORE as usize] = (0, RLIM_INFINITY);
    limits[RLIMIT_MEMLOCK as usize] = (8 << 20, 8 << 20);
    limits[RLIMIT_NICE as usize] = (0, 0);
    limits[RLIMIT_RTPRIO as usize] = (0, 0);
    limits
}

/// Read a `struct rlimit { rlim_cur, rlim_max }` (2 x u64) from `addr`.
pub(super) fn read_rlimit(mem: &GuestMemory, addr: u64) -> Option<(u64, u64)> {
    Some((mem.read_u64(addr).ok()?, mem.read_u64(addr + 8).ok()?))
}

/// Write a `struct rlimit { rlim_cur, rlim_max }` (2 x u64) at `addr`.
pub(super) fn write_rlimit(mem: &mut GuestMemory, addr: u64, cur: u64, max: u64) -> i64 {
    let mut b = [0u8; 16];
    b[0..8].copy_from_slice(&cur.to_le_bytes());
    b[8..16].copy_from_slice(&max.to_le_bytes());
    if mem.write(addr, &b).is_err() {
        err(Errno::EFAULT)
    } else {
        0
    }
}

/// Encode a 64-bit `struct sysinfo` (112 bytes): 2 GiB total RAM at offset 32,
/// `procs = 1` at offset 80, `mem_unit = 1` at offset 104; everything else 0.
fn encode_sysinfo() -> [u8; 112] {
    const TOTAL_RAM: u64 = 2 * 1024 * 1024 * 1024;
    let mut b = [0u8; 112];
    // uptime (offset 0): seconds the VM has been running.
    b[0..8].copy_from_slice(&(crate::clock::now_monotonic().as_secs() as i64).to_le_bytes());
    b[32..40].copy_from_slice(&TOTAL_RAM.to_le_bytes()); // totalram
    // freeram (offset 40): report most of RAM free — 0 would make memory-sizing
    // programs (JVMs, databases) think the machine is out of memory.
    b[40..48].copy_from_slice(&(TOTAL_RAM * 3 / 4).to_le_bytes());
    b[80..82].copy_from_slice(&1u16.to_le_bytes()); // procs
    b[104..108].copy_from_slice(&1u32.to_le_bytes()); // mem_unit
    b
}

#[cfg(test)]
mod tests {
    use super::{Errno, Kernel, ServiceCtx, err, sys_sysinfo};
    use crate::abi::Arch;
    use crate::fs::{MountTable, TmpFs};
    use crate::vcpu::GuestMemory;
    use crate::vcpu::mem::{PAGE_SIZE, Prot};

    fn setup() -> (Kernel, GuestMemory) {
        let mut mounts = MountTable::new();
        mounts.mount("/", Box::new(TmpFs::new()));
        let kernel = Kernel::new(Arch::Aarch64, mounts);
        let mut mem = GuestMemory::new(0x1_0000, 16 * PAGE_SIZE);
        mem.map(0x1_0000, 4 * PAGE_SIZE, Prot::rw()).unwrap();
        (kernel, mem)
    }

    #[test]
    fn sysinfo_writes_totalram() {
        let (_k, mut mem) = setup();
        let buf = 0x1_0000;
        assert_eq!(sys_sysinfo(buf, &mut mem), 0);
        assert_eq!(mem.read_u64(buf + 32).unwrap(), 2 * 1024 * 1024 * 1024);
        assert_eq!(mem.read_u32(buf + 104).unwrap(), 1);
    }

    #[test]
    fn prctl_set_get_name_roundtrips() {
        let (k, mut mem) = setup();
        let mut cx = ServiceCtx::for_test();
        let name = 0x1_0000;
        mem.write_init(name, b"myproc\0").unwrap();
        assert_eq!(k.sys_prctl(&mut cx, &[15, name, 0, 0, 0, 0], &mut mem), 0);
        let out = 0x1_1000;
        assert_eq!(k.sys_prctl(&mut cx, &[16, out, 0, 0, 0, 0], &mut mem), 0);
        assert_eq!(mem.read_vec(out, 6).unwrap(), b"myproc");
    }

    #[test]
    fn prctl_no_new_privs_and_pdeathsig_latch() {
        let (k, mut mem) = setup();
        let mut cx = ServiceCtx::for_test();
        // no_new_privs: set-once latch a re-check must observe.
        assert_eq!(k.sys_prctl(&mut cx, &[38, 1, 0, 0, 0, 0], &mut mem), 0);
        assert_eq!(k.sys_prctl(&mut cx, &[39, 0, 0, 0, 0, 0], &mut mem), 1);
        // pdeathsig: store and report.
        assert_eq!(k.sys_prctl(&mut cx, &[1, 15, 0, 0, 0, 0], &mut mem), 0);
        let out = 0x1_2000;
        assert_eq!(k.sys_prctl(&mut cx, &[2, out, 0, 0, 0, 0], &mut mem), 0);
        assert_eq!(mem.read_vec(out, 4).unwrap(), 15i32.to_le_bytes());
    }

    #[test]
    fn prctl_dumpable_roundtrips() {
        let (k, mut mem) = setup();
        let mut cx = ServiceCtx::for_test();
        // Default is SUID_DUMP_USER (1).
        assert_eq!(k.sys_prctl(&mut cx, &[3, 0, 0, 0, 0, 0], &mut mem), 1);
        // Set 0 (not dumpable) → reads back 0, not a hardcoded 1.
        assert_eq!(k.sys_prctl(&mut cx, &[4, 0, 0, 0, 0, 0], &mut mem), 0);
        assert_eq!(k.sys_prctl(&mut cx, &[3, 0, 0, 0, 0, 0], &mut mem), 0);
        // Out-of-range set is rejected.
        assert_eq!(
            k.sys_prctl(&mut cx, &[4, 3, 0, 0, 0, 0], &mut mem),
            err(Errno::EINVAL)
        );
    }
}
