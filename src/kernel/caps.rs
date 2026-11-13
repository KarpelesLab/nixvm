//! Capabilities: a task's privilege, as Linux divides it.
//!
//! Each task has the five capability sets and the securebits of
//! `security/commoncap.c`. They are relative to the task's user namespace: a
//! capability counts for a resource only if that namespace owns it, or is an
//! ancestor of the namespace that does ([`Kernel::ns_capable`]). Root's privilege
//! is these sets and nothing else, so a sandbox that drops them (`capset`,
//! `PR_CAPBSET_DROP`, a `setuid` away from root) is held to that, uid 0 or not.
//!
//! The sets change at the points Linux changes them: a uid change applies
//! `cap_emulate_setxuid` and the fsuid fixup, `execve` recomputes them by the
//! root rule (there are no file capabilities), and entering a user namespace gives
//! every capability in it.

use super::{Kernel, ServiceCtx};
use crate::abi::errno::Errno;

/// The last capability Linux defines (`CAP_CHECKPOINT_RESTORE`).
pub(super) const CAP_LAST_CAP: u32 = 40;
/// Every capability.
pub(super) const CAP_ALL: u64 = (1u64 << (CAP_LAST_CAP + 1)) - 1;

pub(super) const CAP_CHOWN: u32 = 0;
pub(super) const CAP_DAC_OVERRIDE: u32 = 1;
pub(super) const CAP_DAC_READ_SEARCH: u32 = 2;
pub(super) const CAP_FOWNER: u32 = 3;
pub(super) const CAP_FSETID: u32 = 4;
pub(super) const CAP_SETGID: u32 = 6;
pub(super) const CAP_SETUID: u32 = 7;
pub(super) const CAP_SETPCAP: u32 = 8;
const CAP_LINUX_IMMUTABLE: u32 = 9;
pub(super) const CAP_NET_ADMIN: u32 = 12;
pub(super) const CAP_IPC_LOCK: u32 = 14;
pub(super) const CAP_SYS_CHROOT: u32 = 18;
pub(super) const CAP_SYS_ADMIN: u32 = 21;
pub(super) const CAP_SYS_NICE: u32 = 23;
pub(super) const CAP_SYS_RESOURCE: u32 = 24;
pub(super) const CAP_SYS_TTY_CONFIG: u32 = 26;
const CAP_MKNOD: u32 = 27;
const CAP_MAC_OVERRIDE: u32 = 32;
pub(super) const CAP_SETFCAP: u32 = 31;
pub(super) const CAP_SYSLOG: u32 = 34;

/// What a change of fsuid to or from root turns off or on (`CAP_FS_MASK`).
const CAP_FS_MASK: u64 = 1 << CAP_CHOWN
    | 1 << CAP_DAC_OVERRIDE
    | 1 << CAP_DAC_READ_SEARCH
    | 1 << CAP_FOWNER
    | 1 << CAP_FSETID
    | 1 << CAP_LINUX_IMMUTABLE
    | 1 << CAP_MKNOD
    | 1 << CAP_MAC_OVERRIDE;

/// `SECBIT_NOROOT`: uid 0 gets nothing from `execve`.
const SECBIT_NOROOT: u32 = 1 << 0;
/// `SECBIT_NO_SETUID_FIXUP`: uid changes leave the sets alone.
const SECBIT_NO_SETUID_FIXUP: u32 = 1 << 2;
/// `SECBIT_KEEP_CAPS`: dropping every root uid keeps the permitted set.
const SECBIT_KEEP_CAPS: u32 = 1 << 4;
const SECBIT_KEEP_CAPS_LOCKED: u32 = 1 << 5;
/// `SECBIT_NO_CAP_AMBIENT_RAISE`.
const SECBIT_NO_CAP_AMBIENT_RAISE: u32 = 1 << 6;
/// Each flag above, with its lock bit just above it.
const SECURE_ALL_BITS: u32 = 0x55;
const SECURE_ALL_LOCKS: u32 = SECURE_ALL_BITS << 1;

/// The capability state of one task.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct CapSets {
    pub(super) effective: u64,
    pub(super) permitted: u64,
    pub(super) inheritable: u64,
    pub(super) bounding: u64,
    pub(super) ambient: u64,
    pub(super) securebits: u32,
}

/// A program file's capabilities, its `security.capability` attribute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FileCaps {
    pub(super) permitted: u64,
    pub(super) inheritable: u64,
    /// The program starts with its permitted set effective.
    pub(super) effective: bool,
}

impl FileCaps {
    /// Decode a `security.capability` value (`struct vfs_cap_data`, revision 1
    /// to 3). A revision-3 value counts only where its root id is root, here or
    /// in an ancestor namespace, which `root_ok` says.
    pub(super) fn parse(raw: &[u8], root_ok: impl Fn(u32) -> bool) -> Option<Self> {
        const REVISION_MASK: u32 = 0xff00_0000;
        const FLAGS_EFFECTIVE: u32 = 1;
        let word = |i: usize| {
            raw.get(i * 4..i * 4 + 4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        };
        let magic = word(0)?;
        let (permitted, inheritable) = match (magic & REVISION_MASK, raw.len()) {
            (0x0100_0000, 12) => (u64::from(word(1)?), u64::from(word(2)?)),
            (0x0200_0000, 20) | (0x0300_0000, 24) => {
                if raw.len() == 24 && !root_ok(word(5)?) {
                    return None;
                }
                let lo = (word(1)?, word(2)?);
                let hi = (word(3)?, word(4)?);
                (
                    u64::from(lo.0) | u64::from(hi.0) << 32,
                    u64::from(lo.1) | u64::from(hi.1) << 32,
                )
            }
            _ => return None,
        };
        Some(Self {
            permitted: permitted & CAP_ALL,
            inheritable: inheritable & CAP_ALL,
            effective: magic & FLAGS_EFFECTIVE != 0,
        })
    }
}

/// A task's user ids as the setuid fixups see them: `(real, effective, saved, fs)`.
pub(super) type Uids = (u32, u32, u32, u32);

impl CapSets {
    /// Every capability: root's, and anyone's in a user namespace it just made
    /// or joined.
    pub(super) const fn full() -> Self {
        Self {
            effective: CAP_ALL,
            permitted: CAP_ALL,
            inheritable: 0,
            bounding: CAP_ALL,
            ambient: 0,
            securebits: 0,
        }
    }

    /// No capabilities, as a task not started as root has.
    pub(super) const fn none() -> Self {
        Self {
            effective: 0,
            permitted: 0,
            ..Self::full()
        }
    }

    /// What a task started as `euid` has.
    pub(super) const fn for_uid(euid: u32) -> Self {
        if euid == 0 {
            Self::full()
        } else {
            Self::none()
        }
    }

    /// Whether `cap` is in the effective set.
    pub(super) fn has(&self, cap: u32) -> bool {
        self.effective >> cap & 1 == 1
    }

    /// `PR_GET_KEEPCAPS`.
    pub(super) fn keep_caps(&self) -> bool {
        self.securebits & SECBIT_KEEP_CAPS != 0
    }

    /// `PR_SET_KEEPCAPS`.
    pub(super) fn set_keep_caps(&mut self, keep: bool) -> Result<(), Errno> {
        if self.securebits & SECBIT_KEEP_CAPS_LOCKED != 0 {
            return Err(Errno::EPERM);
        }
        if keep {
            self.securebits |= SECBIT_KEEP_CAPS;
        } else {
            self.securebits &= !SECBIT_KEEP_CAPS;
        }
        Ok(())
    }

    /// `PR_SET_SECUREBITS`: needs `CAP_SETPCAP`; a locked bit can't change and a
    /// lock can't be released.
    pub(super) fn set_securebits(&mut self, bits: u64) -> Result<(), Errno> {
        let old = self.securebits;
        let new = u32::try_from(bits).map_err(|_| Errno::EPERM)?;
        if ((old & SECURE_ALL_LOCKS) >> 1) & (old ^ new) != 0
            || (old & SECURE_ALL_LOCKS) & !new != 0
            || new & !(SECURE_ALL_BITS | SECURE_ALL_LOCKS) != 0
            || !self.has(CAP_SETPCAP)
        {
            return Err(Errno::EPERM);
        }
        self.securebits = new;
        Ok(())
    }

    /// `PR_CAPBSET_DROP`.
    pub(super) fn drop_bounding(&mut self, cap: u32) -> Result<(), Errno> {
        if !self.has(CAP_SETPCAP) {
            return Err(Errno::EPERM);
        }
        self.bounding &= !(1 << cap);
        Ok(())
    }

    /// `PR_CAP_AMBIENT_RAISE`: the capability must be permitted and inheritable.
    pub(super) fn raise_ambient(&mut self, cap: u32) -> Result<(), Errno> {
        let bit = 1 << cap;
        if self.permitted & bit == 0
            || self.inheritable & bit == 0
            || self.securebits & SECBIT_NO_CAP_AMBIENT_RAISE != 0
        {
            return Err(Errno::EPERM);
        }
        self.ambient |= bit;
        Ok(())
    }

    /// `capset`: permitted only shrinks, effective stays within it, and
    /// inheritable grows only within the bounding set and, without
    /// `CAP_SETPCAP`, what was permitted.
    pub(super) fn capset(
        &mut self,
        effective: u64,
        permitted: u64,
        inheritable: u64,
    ) -> Result<(), Errno> {
        let allowed_inh = if self.has(CAP_SETPCAP) {
            CAP_ALL
        } else {
            self.inheritable | self.permitted
        };
        if inheritable & !allowed_inh != 0
            || inheritable & !(self.inheritable | self.bounding) != 0
            || permitted & !self.permitted != 0
            || effective & !permitted != 0
        {
            return Err(Errno::EPERM);
        }
        self.effective = effective;
        self.permitted = permitted;
        self.inheritable = inheritable;
        self.ambient &= permitted & inheritable;
        Ok(())
    }

    /// After a `set*uid` changed the uids from `old` to `new`
    /// (`cap_emulate_setxuid`, then the fsuid fixup).
    pub(super) fn after_setuid(&mut self, old: Uids, new: Uids) {
        if self.securebits & SECBIT_NO_SETUID_FIXUP != 0 {
            return;
        }
        let had_root = old.0 == 0 || old.1 == 0 || old.2 == 0;
        let has_root = new.0 == 0 || new.1 == 0 || new.2 == 0;
        if had_root && !has_root {
            if !self.keep_caps() {
                self.permitted = 0;
                self.effective = 0;
            }
            self.ambient = 0;
        }
        if old.1 == 0 && new.1 != 0 {
            self.effective = 0;
        }
        if old.1 != 0 && new.1 == 0 {
            self.effective = self.permitted;
        }
        self.after_setfsuid(old.3, new.3);
    }

    /// After the fsuid changed from `old` to `new`: the filesystem capabilities
    /// go with root's fsuid.
    pub(super) fn after_setfsuid(&mut self, old: u32, new: u32) {
        if self.securebits & SECBIT_NO_SETUID_FIXUP != 0 {
            return;
        }
        if old == 0 && new != 0 {
            self.effective &= !CAP_FS_MASK;
        }
        if old != 0 && new == 0 {
            self.effective |= self.permitted & CAP_FS_MASK;
        }
    }

    /// The sets a program starts with (`cap_bprm_creds_from_file`): what the
    /// file's capabilities and the bounding and inheritable sets allow; for
    /// root, real or effective, whatever the bounding and inheritable sets
    /// allow; and the ambient set, unless the program is set-id or has file
    /// capabilities. `ruid`/`euid` are the ids it runs with; with
    /// `no_new_privs`, it gains nothing it wasn't permitted.
    pub(super) fn on_exec(
        &mut self,
        ruid: u32,
        euid: u32,
        set_id: bool,
        file: Option<FileCaps>,
        no_new_privs: bool,
    ) {
        let old_permitted = self.permitted;
        let mut effective = false;
        self.permitted = 0;
        if let Some(f) = file {
            self.permitted = (self.bounding & f.permitted) | (self.inheritable & f.inheritable);
            effective = f.effective;
        }
        let set_uid_root = file.is_some() && euid == 0 && ruid != 0;
        if self.securebits & SECBIT_NOROOT == 0 && !set_uid_root {
            if euid == 0 || ruid == 0 {
                self.permitted = self.bounding | self.inheritable;
            }
            effective |= euid == 0;
        }
        if no_new_privs {
            self.permitted &= old_permitted;
        }
        if set_id || file.is_some() {
            self.ambient = 0;
        }
        self.permitted |= self.ambient;
        self.effective = if effective {
            self.permitted
        } else {
            self.ambient
        };
        self.securebits &= !SECBIT_KEEP_CAPS;
    }

    /// `/proc/<pid>/status`'s capability lines.
    pub(super) fn status_lines(&self) -> String {
        format!(
            "CapInh:\t{:016x}\nCapPrm:\t{:016x}\nCapEff:\t{:016x}\nCapBnd:\t{:016x}\nCapAmb:\t{:016x}\n",
            self.inheritable, self.permitted, self.effective, self.bounding, self.ambient
        )
    }
}

impl Kernel {
    /// Whether the caller holds `cap` over things user namespace `target` owns
    /// (Linux's `ns_capable`): its own namespace's capability, or `target` is
    /// nested in it below a namespace the caller's euid created, as a
    /// namespace's owner has every capability there.
    pub(super) fn ns_capable(&self, cx: &ServiceCtx, target: u32, cap: u32) -> bool {
        let mine = cx.cur.ns.user;
        if target == mine {
            return cx.cur.caps.has(cap);
        }
        let users = self.users.lock().unwrap();
        let mut ns = target;
        loop {
            if ns == mine {
                return cx.cur.caps.has(cap);
            }
            let Some((parent, owner)) = users.parent_and_owner(ns) else {
                return false;
            };
            if parent == mine && owner == cx.cur.creds.euid {
                return true;
            }
            ns = parent;
        }
    }

    /// Whether the caller holds `cap` over the whole VM (Linux's `capable`).
    pub(super) fn capable(cx: &ServiceCtx, cap: u32) -> bool {
        cx.cur.ns.user == 0 && cx.cur.caps.has(cap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dropping_root_uids_drops_capabilities_unless_kept() {
        let mut c = CapSets::full();
        c.after_setuid((0, 0, 0, 0), (1000, 1000, 1000, 1000));
        assert_eq!((c.effective, c.permitted), (0, 0));

        let mut c = CapSets::full();
        c.set_keep_caps(true).unwrap();
        c.after_setuid((0, 0, 0, 0), (1000, 1000, 1000, 1000));
        assert_eq!((c.effective, c.permitted), (0, CAP_ALL));
    }

    #[test]
    fn effective_follows_euid() {
        let mut c = CapSets::full();
        c.after_setuid((0, 0, 0, 0), (0, 1000, 0, 1000));
        assert_eq!(c.effective, 0);
        assert_eq!(c.permitted, CAP_ALL);
        c.after_setuid((0, 1000, 0, 1000), (0, 0, 0, 0));
        assert_eq!(c.effective, CAP_ALL);
    }

    #[test]
    fn fsuid_takes_the_filesystem_capabilities() {
        let mut c = CapSets::full();
        c.after_setfsuid(0, 5);
        assert_eq!(c.effective, CAP_ALL & !CAP_FS_MASK);
        c.after_setfsuid(5, 0);
        assert_eq!(c.effective, CAP_ALL);
    }

    #[test]
    fn exec_follows_the_root_rule() {
        let mut c = CapSets::full();
        c.bounding &= !(1 << CAP_SYS_ADMIN);
        c.on_exec(0, 0, false, None, false);
        assert!(!c.has(CAP_SYS_ADMIN));
        assert!(c.has(CAP_CHOWN));

        let mut c = CapSets::full();
        c.on_exec(1000, 1000, false, None, false);
        assert_eq!((c.effective, c.permitted), (0, 0));

        let mut c = CapSets::none();
        c.on_exec(1000, 0, true, None, false);
        assert_eq!(c.effective, CAP_ALL, "a set-uid-root program");
    }

    #[test]
    fn ambient_capabilities_survive_exec_as_non_root() {
        let mut c = CapSets::full();
        let bit = 1 << CAP_NET_ADMIN;
        c.set_keep_caps(true).unwrap();
        c.after_setuid((0, 0, 0, 0), (1000, 1000, 1000, 1000));
        c.capset(bit, CAP_ALL, bit).unwrap();
        c.raise_ambient(CAP_NET_ADMIN).unwrap();
        c.on_exec(1000, 1000, false, None, false);
        assert_eq!((c.effective, c.permitted), (bit, bit));
    }

    #[test]
    fn file_capabilities_grant_what_they_name() {
        // newuidmap's: cap_setuid=ep, revision 2.
        let raw = [
            1, 0, 0, 2, 0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        let f = FileCaps::parse(&raw, |_| true).unwrap();
        assert_eq!((f.permitted, f.effective), (1 << CAP_SETUID, true));
        let mut c = CapSets::none();
        c.on_exec(1000, 1000, false, Some(f), false);
        assert_eq!(
            (c.effective, c.permitted),
            (1 << CAP_SETUID, 1 << CAP_SETUID)
        );
        let mut c = CapSets::none();
        c.on_exec(1000, 1000, false, Some(f), true);
        assert_eq!(c.permitted, 0, "no_new_privs gains nothing");
        let mut rev3 = raw.to_vec();
        rev3[3] = 3;
        rev3.extend_from_slice(&1000u32.to_le_bytes());
        assert!(FileCaps::parse(&rev3, |root| root == 0).is_none());
        assert!(FileCaps::parse(&rev3, |root| root == 1000).is_some());
        assert!(FileCaps::parse(&raw[..19], |_| true).is_none());
    }

    #[test]
    fn capset_only_shrinks() {
        let mut c = CapSets::full();
        c.capset(0, 1, 0).unwrap();
        assert_eq!(c.capset(3, 3, 0), Err(Errno::EPERM));
        assert_eq!(c.capset(2, 1, 0), Err(Errno::EPERM));
    }

    #[test]
    fn locked_securebits_stay() {
        let mut c = CapSets::full();
        c.set_securebits(u64::from(SECBIT_KEEP_CAPS | SECBIT_KEEP_CAPS_LOCKED))
            .unwrap();
        assert_eq!(c.set_keep_caps(false), Err(Errno::EPERM));
        assert_eq!(c.set_securebits(0), Err(Errno::EPERM));
    }
}
