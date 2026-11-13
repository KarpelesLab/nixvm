//! Synthesized `/proc` filesystem.
//!
//! Presents the set of pseudo-files a Linux userland pokes at during startup
//! and everyday operation. There are two flavours of content:
//!
//! * **Static** files — `version`, `filesystems`, `mounts`, … — carry fixed,
//!   plausible bytes baked in at compile time. A few system files
//!   (`cpuinfo`, `stat`) are *rendered* rather than static, so their per-CPU
//!   blocks track the injected core count.
//! * **Per-process** files under `self/` — `cmdline`, `status`, `maps`, the
//!   `exe`/`cwd` symlinks, `fd/`, … — are rendered from a [`ProcData`] value.
//!   The backend ships sensible placeholders; the kernel injects the real
//!   running-process data later through [`ProcFs::set_self`], keeping this
//!   file free of any dependency on the process model.
//!
//! `/proc/<pid>` (for the injected pid) is a transparent alias of `/proc/self`
//! — every path under it is rewritten to its `self/...` equivalent before any
//! lookup — so the two trees always agree without doubling the storage.
//!
//! Layout is a small fixed tree (`""`, `self`, `sys`, `sys/kernel`); paths are
//! matched directly rather than through a stored map. `self/fd/<n>` entries
//! are the one dynamic exception, sized by however many descriptors the
//! kernel injected. The backend is read-only, so every mutating method keeps
//! the trait's `EROFS` default.

use std::fmt::Write as _;
use std::io;

use super::{Attrs, DirEntry, MountFs, NodeKind};

/// Unix mode type bit for a directory.
const S_IFDIR: u32 = 0o040_000;
/// Unix mode type bit for a regular file.
const S_IFREG: u32 = 0o100_000;
/// Unix mode type bit for a symbolic link.
const S_IFLNK: u32 = 0o120_000;

/// `self/root`'s symlink target. A sandboxed process is never chrooted in
/// this backend, so it always resolves to the filesystem root.
const ROOT_TARGET: &str = "/";

/// Every static path this backend knows, in a fixed order. The 1-based index
/// into this table is the node's inode, which keeps every inode distinct for
/// free. `self/fd/<n>` entries are not listed here — see [`fd_inode`].
const PATHS: &[&str] = &[
    "", // root directory — inode 1
    "self",
    "sys",
    "sys/kernel",
    "cpuinfo",
    "meminfo",
    "version",
    "uptime",
    "loadavg",
    "stat",
    "filesystems",
    "mounts",
    "cmdline",
    "sys/kernel/ostype",
    "sys/kernel/osrelease",
    "sys/kernel/hostname",
    "sys/kernel/domainname",
    "sys/kernel/overflowuid",
    "sys/kernel/overflowgid",
    "sys/kernel/pid_max",
    "self/cmdline",
    "self/status",
    "self/stat",
    "self/comm",
    "self/maps",
    "self/exe",
    "self/cwd",
    "self/auxv",
    "self/fd",
    "net",
    "net/tcp",
    "net/tcp6",
    "net/udp",
    "net/udp6",
    "net/unix",
    "net/dev",
    "net/route",
    "net/snmp",
    "net/protocols",
    "sys/net",
    "sys/net/core",
    "sys/net/core/somaxconn",
    "sys/net/ipv4",
    "sys/net/ipv4/tcp_rmem",
    "sys/net/ipv4/tcp_wmem",
    "sys/net/ipv4/ip_local_port_range",
    "sys/vm",
    "sys/vm/overcommit_memory",
    "sys/vm/max_map_count",
    "sys/fs",
    "sys/fs/file-max",
    "sys/fs/nr_open",
    "diskstats",
    "partitions",
    "swaps",
    "modules",
    "devices",
    "self/mountinfo",
    "self/mounts",
    "self/smaps",
    "self/statm",
    "self/limits",
    "self/cgroup",
    "self/io",
    "self/oom_score",
    "self/oom_score_adj",
    "self/wchan",
    "self/environ",
    "self/root",
    "self/sched",
    "self/schedstat",
    "self/syscall",
    "self/personality",
    "self/setgroups",
    "self/gid_map",
    "self/uid_map",
    "self/timens_offsets",
    "self/ns",
    "self/ns/cgroup",
    "self/ns/ipc",
    "self/ns/mnt",
    "self/ns/net",
    "self/ns/pid",
    "self/ns/pid_for_children",
    "self/ns/time",
    "self/ns/time_for_children",
    "self/ns/user",
    "self/ns/uts",
    "sys/kernel/random",
    "sys/kernel/random/boot_id",
    "sys/kernel/random/uuid",
    "sys/kernel/random/entropy_avail",
    "sys/kernel/random/poolsize",
    "sys/kernel/version",
    "sys/kernel/threads-max",
    "sys/kernel/ngroups_max",
    "sys/kernel/cap_last_cap",
    "sys/kernel/sched_rr_timeslice_ms",
    "sys/kernel/seccomp",
    "sys/kernel/seccomp/actions_avail",
    "sys/kernel/seccomp/actions_logged",
    "kallsyms",
    "keys",
    "key-users",
    "locks",
    "vmstat",
    "zoneinfo",
    "buddyinfo",
    "consoles",
    "self/task",
    "thread-self",
];

/// Static top-level file names, in `readdir("")` order.
const ROOT_FILES: &[&str] = &[
    "cpuinfo",
    "meminfo",
    "version",
    "uptime",
    "loadavg",
    "stat",
    "filesystems",
    "mounts",
    "cmdline",
    "diskstats",
    "partitions",
    "swaps",
    "modules",
    "devices",
    "kallsyms",
    "keys",
    "key-users",
    "locks",
    "vmstat",
    "zoneinfo",
    "buddyinfo",
    "consoles",
];

/// What `/proc/<pid>` shows about another task, as the reader numbers it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcTask {
    pub pid: u32,
    pub ppid: u32,
    /// The task tracing it (`TracerPid`), 0 for none.
    pub tracer: u32,
    pub comm: String,
    pub cmdline: Vec<u8>,
    /// The letter `ps` shows: `R`, `S`, `T`, `t` or `Z`.
    pub state: char,
    /// Who owns its `/proc/<pid>` entries: its effective ids, as the VM sees them.
    pub uid: u32,
    pub gid: u32,
    /// Its `status` file's `Uid:` and `Gid:` lines, in the reader's terms. Empty
    /// shows `uid`/`gid`.
    pub status_ids: String,
    /// Its user namespace's `uid_map`, `gid_map` and `setgroups`.
    pub id_files: [String; 3],
    /// Body of its `cgroup` file (`0::/path\n`).
    pub cgroup: String,
    /// Its `NoNewPrivs:`, `Seccomp:` and `Seccomp_filters:` status lines.
    pub sandbox: Sandbox,
    /// Body of its `syscall` file: the syscall it waits in, or `running`.
    pub syscall: String,
}

/// What `/proc/<pid>/status` says of a task's sandboxing: its `no_new_privs`
/// bit, seccomp mode and how many filters it attached.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Sandbox {
    pub no_new_privs: bool,
    pub seccomp: u8,
    pub filters: u32,
}

impl Sandbox {
    /// The status lines, as Linux prints them.
    #[must_use]
    pub fn lines(self) -> String {
        format!(
            "NoNewPrivs:\t{}\nSeccomp:\t{}\nSeccomp_filters:\t{}\n",
            u8::from(self.no_new_privs),
            self.seccomp,
            self.filters
        )
    }
}

impl ProcTask {
    /// The contents of `name` under its directory, for the files it has.
    fn file(&self, name: &str) -> Option<Vec<u8>> {
        Some(match name {
            "comm" => format!("{}\n", self.comm).into_bytes(),
            "cmdline" => self.cmdline.clone(),
            "cgroup" => self.cgroup.clone().into_bytes(),
            "uid_map" => self.id_files[0].clone().into_bytes(),
            "gid_map" => self.id_files[1].clone().into_bytes(),
            "setgroups" => self.id_files[2].clone().into_bytes(),
            "syscall" => self.syscall.clone().into_bytes(),
            "stat" => {
                // Only the leading fields `ps` and friends read carry anything.
                let rest = "0 ".repeat(45);
                format!(
                    "{} ({}) {} {} {} {} 0 {}\n",
                    self.pid,
                    self.comm,
                    self.state,
                    self.ppid,
                    self.pid,
                    self.pid,
                    rest.trim_end()
                )
                .into_bytes()
            }
            "status" => {
                let (uid, gid) = (self.uid, self.gid);
                let ids = if self.status_ids.is_empty() {
                    format!("Uid:\t{uid}\t{uid}\t{uid}\t{uid}\nGid:\t{gid}\t{gid}\t{gid}\t{gid}\n")
                } else {
                    self.status_ids.clone()
                };
                format!(
                    "Name:\t{}\nState:\t{} ({})\nTgid:\t{}\nPid:\t{}\nPPid:\t{}\nTracerPid:\t{}\n{ids}{}",
                    self.comm,
                    self.state,
                    state_desc(self.state),
                    self.pid,
                    self.pid,
                    self.ppid,
                    self.tracer,
                    self.sandbox.lines()
                )
                .into_bytes()
            }
            _ => return None,
        })
    }
}

/// Files under `/proc/<pid>` for a task other than the reader.
const TASK_FILES: &[&str] = &[
    "cgroup",
    "cmdline",
    "comm",
    "gid_map",
    "setgroups",
    "stat",
    "status",
    "syscall",
    "uid_map",
];

/// A fixed inode range for `/proc/<pid>` entries, which have no table entry.
fn task_inode(pid: u32, slot: u64) -> u64 {
    (1 << 40) + (u64::from(pid) << 8) + slot
}

/// Entries of `self/ns`, in the order Linux lists them.
const NS_FILES: &[&str] = &[
    "cgroup",
    "ipc",
    "mnt",
    "net",
    "pid",
    "pid_for_children",
    "time",
    "time_for_children",
    "user",
    "uts",
];

/// Per-process entry names under `self/`, in `readdir("self")` order. `exe`
/// and `cwd` are symlinks, `fd` is a directory, the rest are files.
const SELF_FILES: &[&str] = &[
    "ns",
    "task",
    "timens_offsets",
    "cmdline",
    "status",
    "stat",
    "comm",
    "maps",
    "exe",
    "cwd",
    "auxv",
    "fd",
    "mountinfo",
    "mounts",
    "smaps",
    "statm",
    "limits",
    "cgroup",
    "io",
    "oom_score",
    "oom_score_adj",
    "wchan",
    "environ",
    "root",
    "sched",
    "schedstat",
    "syscall",
    "personality",
    "setgroups",
    "gid_map",
    "uid_map",
];

/// Tunables exposed under `sys/kernel/`, excluding the `random/` subdirectory
/// (which [`ProcFs::readdir`] appends separately since it's a directory, not
/// a file).
const SYS_KERNEL_FILES: &[&str] = &[
    "ostype",
    "osrelease",
    "hostname",
    "domainname",
    "overflowuid",
    "overflowgid",
    "pid_max",
    "version",
    "threads-max",
    "ngroups_max",
    "cap_last_cap",
    "sched_rr_timeslice_ms",
];

/// File names under `sys/kernel/random/`, in `readdir("sys/kernel/random")`
/// order.
const SYS_KERNEL_RANDOM_FILES: &[&str] = &["boot_id", "uuid", "entropy_avail", "poolsize"];

/// File names under `net/`, in `readdir("net")` order.
const NET_FILES: &[&str] = &[
    "tcp",
    "tcp6",
    "udp",
    "udp6",
    "unix",
    "dev",
    "route",
    "snmp",
    "protocols",
];

/// Tunables exposed under `sys/net/core/`.
const SYS_NET_CORE_FILES: &[&str] = &["somaxconn"];

/// Tunables exposed under `sys/net/ipv4/`.
const SYS_NET_IPV4_FILES: &[&str] = &["tcp_rmem", "tcp_wmem", "ip_local_port_range"];

/// Tunables exposed under `sys/vm/`.
const SYS_VM_FILES: &[&str] = &["overcommit_memory", "max_map_count"];

/// Tunables exposed under `sys/fs/`.
const SYS_FS_FILES: &[&str] = &["file-max", "nr_open"];

// ---- static file bodies (each ends with a newline) ----

const MEMINFO: &str = "MemTotal:        2048000 kB\n\
MemFree:         1024000 kB\n\
MemAvailable:    1536000 kB\n\
Buffers:               0 kB\n\
Cached:           512000 kB\n\
SwapTotal:             0 kB\n\
SwapFree:              0 kB\n";

const VERSION: &str = "Linux version 6.1.0-nixvm (nixvm@nixvm) (gcc) #1 SMP nixvm\n";

const UPTIME: &str = "0.00 0.00\n";

const LOADAVG: &str = "0.00 0.00 0.00 1/1 1\n";

const FILESYSTEMS: &str = "nodev\ttmpfs\n\
nodev\tproc\n\
nodev\tsysfs\n\
nodev\tdevtmpfs\n\
nodev\tcgroup2\n";

const MOUNTS: &str = "tmpfs / tmpfs rw 0 0\n\
proc /proc proc rw 0 0\n\
sysfs /sys sysfs rw 0 0\n\
devtmpfs /dev devtmpfs rw 0 0\n";

/// `self/mountinfo`'s body, in the `36 35 98:0 /mnt1 /mnt2 rw,noatime … - fstype source opts`
/// format documented in `proc(5)`.
const MOUNTINFO: &str = "1 0 0:1 / / rw,relatime shared:1 - tmpfs tmpfs rw\n\
2 1 0:2 / /proc rw,nosuid,nodev,noexec,relatime shared:2 - proc proc rw\n\
3 1 0:3 / /sys rw,nosuid,nodev,noexec,relatime shared:3 - sysfs sysfs rw\n\
4 1 0:4 / /dev rw,nosuid,relatime shared:4 - devtmpfs devtmpfs rw\n";

/// `/proc/mounts`, one line per mount.
fn mounts_body(mounts: &[super::MountEntry]) -> String {
    let mut out = String::new();
    for m in mounts {
        let rw = if m.read_only { "ro" } else { "rw" };
        let _ = writeln!(out, "{} {} {} {rw} 0 0", m.source, m.point, m.fs_type);
    }
    out
}

/// `/proc/self/mountinfo`, in the format documented in `proc(5)`.
fn mountinfo_body(mounts: &[super::MountEntry]) -> String {
    let mut out = String::new();
    let under = |p: &str, q: &str| q == "/" || p == q || p.starts_with(&format!("{q}/"));
    for (i, m) in mounts.iter().enumerate() {
        // Its parent: the latest mount before it that it's below.
        let id = i + 1;
        let parent = (0..i)
            .rev()
            .find(|&j| under(&m.point, &mounts[j].point))
            .map_or(0, |j| j + 1);
        let rw = if m.read_only { "ro" } else { "rw" };
        // The optional fields: peer group, master, unbindable.
        let mut opt = String::new();
        if m.peer != 0 {
            let _ = write!(opt, " shared:{}", m.peer);
        }
        if m.master != 0 {
            let _ = write!(opt, " master:{}", m.master);
        }
        if m.unbindable {
            opt.push_str(" unbindable");
        }
        let (major, minor) = (super::blockdev::major(m.dev), super::blockdev::minor(m.dev));
        let _ = writeln!(
            out,
            "{id} {parent} {major}:{minor} /{0} {1} {rw},relatime{opt} - {2} {3} {rw}",
            m.subdir, m.point, m.fs_type, m.source,
        );
    }
    out
}

const CMDLINE: &str = "\n";

/// `sys/kernel/seccomp/actions_avail`: every filter action, as Linux names them.
const SECCOMP_ACTIONS_AVAIL: &str =
    "kill_process kill_thread trap errno user_notif trace log allow\n";
/// `sys/kernel/seccomp/actions_logged`'s default: all but `allow`.
const SECCOMP_ACTIONS_LOGGED: &str = "kill_process kill_thread trap errno user_notif trace log\n";

const OSTYPE: &str = "Linux\n";
const OSRELEASE: &str = "6.1.0-nixvm\n";
const HOSTNAME: &str = "nixvm\n";
const PID_MAX: &str = "32768\n";
/// `sys/kernel/version`'s body — distinct from top-level `version`
/// ([`VERSION`]): this one is just the build tag, as `proc(5)` documents.
const KERNEL_VERSION: &str = "#1 SMP nixvm\n";
const THREADS_MAX: &str = "127871\n";
const NGROUPS_MAX: &str = "65536\n";
/// Last valid capability index (`CAP_CHECKPOINT_RESTORE` on a modern kernel).
const CAP_LAST_CAP: &str = "40\n";
const SCHED_RR_TIMESLICE_MS: &str = "100\n";

// ---- sys/kernel/random/* ----

/// Fixed-but-valid UUIDs: a synthetic sandbox has no real boot entropy, but
/// userland only ever checks the shape, not the value.
const BOOT_ID: &str = "2f38a1c4-9b3e-4d3a-8f2b-6b6f2a9c1e10\n";
const RANDOM_UUID: &str = "b6d1a0f4-7c3d-4a2e-9f1b-3c5d7e9a2b41\n";
const ENTROPY_AVAIL: &str = "256\n";
const POOLSIZE: &str = "4096\n";

// ---- /proc/net/* ----

const NET_TCP_HEADER: &str = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n";

const NET_TCP6_HEADER: &str = "  sl  \
local_address                         remote_address                        st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n";

const NET_UDP_HEADER: &str = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when \
retrnsmt   uid  timeout inode ref pointer drops\n";

const NET_UDP6_HEADER: &str = "  sl  \
local_address                         remote_address                        st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode ref pointer drops\n";

const NET_UNIX: &str = "Num       RefCount Protocol Flags    Type St Inode Path\n";

const NET_DEV: &str = "Inter-|   Receive                                                |  Transmit\n\
 face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed\n\
    lo:       0       0    0    0    0     0          0         0        0       0    0    0    0     0       0          0\n";

const NET_ROUTE: &str =
    "Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT\n";

const NET_SNMP: &str = "Ip: Forwarding DefaultTTL InReceives InHdrErrors InAddrErrors ForwDatagrams InUnknownProtos InDiscards InDelivers OutRequests OutDiscards OutNoRoutes ReasmTimeout ReasmReqds ReasmOKs ReasmFails FragOKs FragFails FragCreates\n\
Ip: 1 64 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n\
Icmp: InMsgs InErrors InCsumErrors InDestUnreachs InTimeExcds InParmProbs InSrcQuenchs InRedirects InEchos InEchoReps InTimestamps InTimestampReps InAddrMasks InAddrMaskReps OutMsgs OutErrors OutDestUnreachs OutTimeExcds OutParmProbs OutSrcQuenchs OutRedirects OutEchos OutEchoReps OutTimestamps OutTimestampReps OutAddrMasks OutAddrMaskReps\n\
Icmp: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n\
Tcp: RtoAlgorithm RtoMin RtoMax MaxConn ActiveOpens PassiveOpens AttemptFails EstabResets CurrEstab InSegs OutSegs RetransSegs InErrs OutRsts InCsumErrors\n\
Tcp: 1 200 120000 -1 0 0 0 0 0 0 0 0 0 0 0\n\
Udp: InDatagrams NoPorts InErrors OutDatagrams RcvbufErrors SndbufErrors InCsumErrors IgnoredMulti\n\
Udp: 0 0 0 0 0 0 0 0\n";

const NET_PROTOCOLS: &str = "protocol  size sockets  memory press maxhdr  slab module     cl co di ac io in de sh ss gs se re sp bi br ha uh gp em\n\
TCP        2144   0       -1     no     20     yes   kernel     y  y  y  y  y  y  y  y  y  y  y  y  n  n  y  y  y  n\n\
UDP        1408   0       -1     no      0     yes   kernel     y  y  y  n  n  n  y  y  y  y  y  n  n  n  y  y  y  n\n";

// ---- /proc/sys/{net,vm,fs}/* ----

const SOMAXCONN: &str = "4096\n";
const TCP_RMEM: &str = "4096\t131072\t6291456\n";
const TCP_WMEM: &str = "4096\t16384\t4194304\n";
const IP_LOCAL_PORT_RANGE: &str = "32768\t60999\n";
const OVERCOMMIT_MEMORY: &str = "0\n";
const MAX_MAP_COUNT: &str = "65530\n";
const FILE_MAX: &str = "1048576\n";
const NR_OPEN: &str = "1048576\n";

// ---- misc top-level files ----

/// `diskstats` and `partitions` before the VM's disks are known: none.
const DISKSTATS: &str = "";
const PARTITIONS: &str = "major minor  #blocks  name\n\n";
const SWAPS: &str = "Filename\t\t\t\tType\t\tSize\t\tUsed\t\tPriority\n";
/// No modules are ever loaded in a synthetic kernel; an empty file is the
/// valid, real-world rendering of that state.
const MODULES: &str = "";
const DEVICES: &str = "Character devices:\n\
  1 mem\n\
  4 tty\n\
  5 /dev/tty\n\
  5 /dev/console\n\
 10 misc\n\
136 pts\n\
\n\
Block devices:\n\
  7 loop\n\
254 virtblk\n\
259 blkext\n";

/// No kernel modules ever register symbols in a synthetic kernel; a couple of
/// placeholder entries are a valid, minimal rendering (a fully empty file
/// would be valid too, but this exercises callers that expect at least one
/// parseable line).
const KALLSYMS: &str = "0000000000000000 t placeholder\n\
0000000000000000 t placeholder2\n";
/// No keys are ever created in a synthetic kernel; empty is the valid,
/// real-world rendering of that state.
const KEYS: &str = "";
const KEY_USERS: &str = "";
/// No file locks are ever held in a synthetic kernel; empty is valid.
const LOCKS: &str = "";

const VMSTAT: &str = "nr_free_pages 262144\n\
nr_zone_inactive_anon 0\n\
nr_zone_active_anon 0\n\
nr_zone_inactive_file 0\n\
nr_zone_active_file 0\n\
nr_zone_unevictable 0\n\
nr_slab_reclaimable 1024\n\
nr_slab_unreclaimable 2048\n\
pgfault 0\n\
pgmajfault 0\n";

const ZONEINFO: &str = "Node 0, zone   Normal\n\
  pages free     262144\n\
        min      1024\n\
        low      1280\n\
        high     1536\n\
        spanned  524288\n\
        present  524288\n\
        managed  512000\n";

const BUDDYINFO: &str =
    "Node 0, zone   Normal    100    50    25    12     6     3     1     0     0     0     0\n";

const CONSOLES: &str = "ttyS0                -W- (EC p a)    4:64\n";

// ---- self/* static bodies (independent of injected ProcData) ----

const LIMITS: &str = "Limit                     Soft Limit           Hard Limit           Units     \n\
Max cpu time              unlimited            unlimited            seconds   \n\
Max file size             unlimited            unlimited            bytes     \n\
Max data size             unlimited            unlimited            bytes     \n\
Max stack size            8388608              unlimited            bytes     \n\
Max core file size        0                    unlimited            bytes     \n\
Max resident set          unlimited            unlimited            bytes     \n\
Max processes             15746                15746                processes \n\
Max open files            1024                 4096                 files     \n\
Max locked memory         65536                65536                bytes     \n\
Max address space         unlimited            unlimited            bytes     \n\
Max file locks            unlimited            unlimited            locks     \n\
Max pending signals       15746                15746                signals   \n\
Max msgqueue size         819200               819200               bytes     \n\
Max nice priority         0                    0                              \n\
Max realtime priority     0                    0                              \n\
Max realtime timeout      unlimited            unlimited            us        \n";

const IO: &str = "rchar: 0\n\
wchar: 0\n\
syscr: 0\n\
syscw: 0\n\
read_bytes: 0\n\
write_bytes: 0\n\
cancelled_write_bytes: 0\n";

const OOM_SCORE: &str = "0\n";
const OOM_SCORE_ADJ: &str = "0\n";
/// `self/wchan` has no trailing newline on a real kernel — it's a single
/// symbol name (or `0` when idle), not a line-oriented text file.
const WCHAN: &str = "0";

/// `self/environ`'s body: NUL-separated `KEY=value` pairs, as the kernel
/// presents them (including the trailing NUL after the last entry).
/// [`ProcData`] carries no environment yet, so this is a minimal-but-plausible
/// static default rather than an injected value.
const ENVIRON: &str =
    "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\0HOME=/root\0TERM=xterm\0";
/// `self/schedstat`'s three whitespace-separated fields: time spent on the
/// CPU, time spent waiting to run, and timeslices run — all `0` since a
/// synthetic process hasn't actually been scheduled.
const SCHEDSTAT: &str = "0 0 0\n";
/// `self/personality`'s body: `PER_LINUX`, the standard Linux personality.
const PERSONALITY: &str = "0\n";
/// `self/setgroups`'s body: `allow`, the default before a user namespace
/// restricts `setgroups(2)`.
const SETGROUPS: &str = "allow\n";
/// Shared body for `self/uid_map` and `self/gid_map`: a single line mapping
/// the whole id range to itself (`id-inside-ns id-outside-ns length`).
const UID_GID_MAP: &str = "0 0 4294967295\n";

/// Running-process (and lightweight system) data backing the `self/` files
/// plus the CPU-count-sensitive system files.
///
/// The kernel builds one of these from the real process it is executing and
/// installs it with [`ProcFs::set_self`]; until then [`ProcData::default`]
/// supplies placeholders so every file still renders something plausible.
#[derive(Debug, Clone)]
pub struct ProcData {
    /// Every task the reader can see, for `/proc/<pid>`.
    pub tasks: Vec<ProcTask>,
    /// What the kernel says for `self/uid_map`, `self/gid_map`, `self/setgroups`
    /// and `self/timens_offsets`, once it has.
    pub uid_map: String,
    pub gid_map: String,
    pub setgroups: String,
    pub timens_offsets: String,
    /// Body of `self/limits`, and the UTS names for `sys/kernel/hostname` and
    /// `domainname`. Each is empty until the kernel has said, when the fixed
    /// text is shown instead.
    pub limits: String,
    pub hostname: String,
    pub domainname: String,
    /// Body of `self/cgroup`; empty until the kernel has said.
    pub cgroup: String,
    /// The reader's mounts, for `mounts` and `self/mountinfo`. Empty until the
    /// kernel has refreshed this, when the fixed text is shown instead.
    pub mounts: Vec<super::MountEntry>,
    /// Raw `argv` for `self/cmdline`, NUL-separated as the kernel presents it.
    pub cmdline: Vec<u8>,
    /// Absolute path the `self/exe` symlink resolves to.
    pub exe: String,
    /// Absolute path the `self/cwd` symlink resolves to.
    pub cwd: String,
    /// Body of `self/maps`. Empty means "not injected"; a minimal plausible
    /// map (image + heap + stack) is synthesized instead.
    pub maps: String,
    /// Short command name for `self/comm` (and the parenthesised field of
    /// `self/stat`).
    pub comm: String,
    /// The zeroth argument the process was launched with.
    pub argv0: String,
    /// Process id. Also the `<pid>` directory that mirrors `self/`. `0` means
    /// no numeric alias is published.
    pub pid: u32,
    /// Parent process id.
    pub ppid: u32,
    /// The task tracing it, 0 for none.
    pub tracer: u32,
    /// Real/effective/saved/filesystem uid (all reported equal, as a static
    /// sandbox has no reason to differ).
    pub uid: u32,
    /// Real/effective/saved/filesystem gid.
    pub gid: u32,
    /// Who owns `self/` and what's in it: the reader's effective ids, as the VM
    /// sees them.
    pub owner: (u32, u32),
    /// `status`'s `Uid:`, `Gid:` and `Groups:` lines, and its capability,
    /// `NoNewPrivs` and `Seccomp` lines, once the kernel has said (until then
    /// `uid`/`gid` are shown).
    pub status_ids: String,
    pub status_security: String,
    /// Single-letter run state (`R`, `S`, `D`, `Z`, `T`) for `stat`/`status`.
    pub state: char,
    /// Thread count.
    pub threads: u32,
    /// The reader's thread id, and every thread of its process, as it numbers
    /// them: `self/task/` and `thread-self`. Empty until the kernel has said.
    pub tid: u32,
    pub tids: Vec<u32>,
    /// Resident + virtual memory footprint, in kB, for `status`/`stat`.
    pub vm_size_kb: u64,
    pub vm_rss_kb: u64,
    /// Open file descriptors: `(fd number, symlink target)`, backing
    /// `self/fd/`. Empty means nothing is published there.
    pub fds: Vec<(u32, String)>,
    /// CPU core count backing the per-cpu blocks of `cpuinfo` and `stat`.
    pub nproc: usize,
    /// Raw `self/auxv` bytes (pairs of `(type, value)` `u64`s on a real
    /// kernel); empty is a valid, minimal rendering.
    pub auxv: Vec<u8>,
    /// The guest architecture `cpuinfo` describes; `None` until the kernel
    /// says (then the build's own architecture is assumed).
    pub arch: Option<crate::abi::Arch>,
    /// `self/syscall`: the read it is making.
    pub syscall: String,
    /// `sys/kernel/seccomp/actions_logged`'s body; `None` until the kernel
    /// has said (Linux's default is shown then).
    pub seccomp_logged: Option<String>,
}

impl Default for ProcData {
    fn default() -> Self {
        Self {
            tasks: Vec::new(),
            uid_map: String::new(),
            gid_map: String::new(),
            setgroups: String::new(),
            timens_offsets: String::new(),
            limits: String::new(),
            hostname: String::new(),
            domainname: String::new(),
            cgroup: String::new(),
            mounts: Vec::new(),
            cmdline: Vec::new(),
            exe: "/bin/busybox".to_string(),
            cwd: "/".to_string(),
            maps: String::new(),
            comm: "nixvm".to_string(),
            argv0: "busybox".to_string(),
            pid: 1,
            ppid: 0,
            tracer: 0,
            uid: 0,
            gid: 0,
            owner: (0, 0),
            status_ids: String::new(),
            status_security: String::new(),
            state: 'R',
            threads: 1,
            tid: 0,
            tids: Vec::new(),
            vm_size_kb: 4096,
            vm_rss_kb: 1024,
            fds: Vec::new(),
            nproc: 1,
            auxv: Vec::new(),
            arch: None,
            seccomp_logged: None,
            syscall: String::new(),
        }
    }
}

/// The live per-process fields the kernel refreshes into [`ProcData`] before
/// serving a `/proc/self` read (see [`ProcFs::update_self`]). A narrow subset
/// of [`ProcData`] — just the values that actually change per running task.
#[derive(Debug, Clone)]
pub struct ProcSelf {
    /// Every task the reader can see, for `/proc/<pid>`.
    pub tasks: Vec<ProcTask>,
    /// `self/uid_map`, `self/gid_map`, `self/setgroups` and `self/timens_offsets`.
    pub uid_map: String,
    pub gid_map: String,
    pub setgroups: String,
    pub timens_offsets: String,
    /// `self/limits`, and the caller's UTS names.
    pub limits: String,
    pub hostname: String,
    pub domainname: String,
    /// Body of `self/cgroup`.
    pub cgroup: String,
    /// The caller's mounts, for `mounts` and `self/mountinfo`.
    pub mounts: Vec<super::MountEntry>,
    pub comm: String,
    pub cmdline: Vec<u8>,
    pub exe: String,
    pub cwd: String,
    pub pid: u32,
    pub ppid: u32,
    /// The reader's effective ids as the VM sees them, which own `self/`.
    pub owner: (u32, u32),
    /// The reader's thread id and its process's threads (see [`ProcData`]).
    pub tid: u32,
    pub tids: Vec<u32>,
    /// `self/status`'s id lines and security lines (see [`ProcData`]).
    pub status_ids: String,
    pub status_security: String,
    /// The task tracing it, 0 for none.
    pub tracer: u32,
    /// Open descriptors as `(fd number, symlink target)`, backing `self/fd/`.
    pub fds: Vec<(u32, String)>,
    /// The task's saved auxiliary vector (`self/auxv`).
    pub auxv: Vec<u8>,
    /// The guest architecture (selects the `cpuinfo` shape and features).
    pub arch: Option<crate::abi::Arch>,
    /// `self/syscall`.
    pub syscall: String,
    /// `sys/kernel/seccomp/actions_logged`.
    pub seccomp_logged: Option<String>,
}

/// The synthesized `/proc` backend.
#[derive(Debug)]
pub struct ProcFs {
    data: ProcData,
    /// The VM's disks, for `partitions` and `diskstats`.
    disks: Option<super::blockdev::SharedDisks>,
}

impl Default for ProcFs {
    fn default() -> Self {
        Self::new(1)
    }
}

impl ProcFs {
    /// Build a procfs whose `cpuinfo`/`stat` render `ncpu` cores (nixvm's own
    /// count), consistent with `sysfs` and `sched_getaffinity`.
    #[must_use]
    pub fn new(ncpu: usize) -> Self {
        Self {
            data: ProcData {
                nproc: ncpu.max(1),
                ..ProcData::default()
            },
            disks: None,
        }
    }

    /// Install the running-process data backing the `self/` files (and the
    /// `<pid>` alias, and the per-cpu system files).
    pub fn set_self(&mut self, data: ProcData) {
        self.data = data;
    }

    /// Refresh only the *live* per-process fields backing `self/` from the task
    /// the kernel is about to serve a `/proc` read for. System-wide fields
    /// (`nproc`, the `cpuinfo`/`stat` cpu blocks) and coarse placeholders
    /// (`state`, memory footprint) are left as configured — this exists so
    /// `self/{comm,cmdline,stat,status,statm,exe}` stop reporting the stale
    /// default (`comm = "nixvm"`, empty `cmdline`) and instead name the running
    /// program. `pid`/`ppid` drive the `<pid>` alias too.
    pub fn update_self(&mut self, live: ProcSelf) {
        let d = &mut self.data;
        d.tasks = live.tasks;
        d.mounts = live.mounts;
        d.uid_map = live.uid_map;
        d.gid_map = live.gid_map;
        d.setgroups = live.setgroups;
        d.timens_offsets = live.timens_offsets;
        d.limits = live.limits;
        d.hostname = live.hostname;
        d.domainname = live.domainname;
        d.cgroup = live.cgroup;
        d.comm = live.comm;
        d.cmdline = live.cmdline;
        d.exe = live.exe;
        d.cwd = live.cwd;
        d.pid = live.pid;
        d.ppid = live.ppid;
        d.owner = live.owner;
        d.tid = live.tid;
        if !live.tids.is_empty() {
            d.threads = live.tids.len() as u32;
        }
        d.tids = live.tids;
        d.status_ids = live.status_ids;
        d.status_security = live.status_security;
        d.tracer = live.tracer;
        d.fds = live.fds;
        d.auxv = live.auxv;
        d.syscall = live.syscall;
        d.seccomp_logged = live.seccomp_logged;
        if live.arch.is_some() {
            d.arch = live.arch;
        }
        d.argv0 = d
            .cmdline
            .split(|&b| b == 0)
            .next()
            .map(|a| String::from_utf8_lossy(a).into_owned())
            .unwrap_or_default();
    }

    /// Rewrite a numeric-pid path to its `self/...` equivalent when it names
    /// the injected pid, leaving every other path untouched. This is the only
    /// place `<pid>` paths are special-cased; everything downstream only ever
    /// sees `self`-rooted or static paths.
    /// The task `rel` is under, if it's `<pid>` or `<pid>/...` for a task other
    /// than the reader, with what comes after.
    fn other_task<'a>(&self, rel: &'a str) -> Option<(&ProcTask, &'a str)> {
        let (first, tail) = rel.split_once('/').unwrap_or((rel, ""));
        let pid: u32 = first.parse().ok()?;
        let task = self.data.tasks.iter().find(|t| t.pid == pid)?;
        Some((task, tail))
    }

    /// `stat` under another task's directory.
    fn other_task_stat(task: &ProcTask, tail: &str) -> Option<Attrs> {
        let (kind, mode, size, slot) = match tail {
            "" | "ns" | "fd" | "task" => {
                let slot = match tail {
                    "" => 0,
                    "ns" => 1,
                    "fd" => 2,
                    _ => 3,
                };
                (NodeKind::Dir, S_IFDIR | 0o555, 0, slot)
            }
            _ if tail
                .strip_prefix("ns/")
                .is_some_and(|n| NS_FILES.contains(&n)) =>
            {
                (NodeKind::Symlink, S_IFLNK | 0o777, 0, 4)
            }
            name => {
                let body = task.file(name)?;
                (NodeKind::File, S_IFREG | 0o444, body.len() as u64, 5)
            }
        };
        Some(Attrs {
            kind,
            size,
            mode,
            uid: task.uid,
            gid: task.gid,
            atime: 0,
            mtime: 0,
            inode: task_inode(task.pid, slot),
            nlink: 1,
            rdev: 0,
            dev: 0,
        })
    }

    fn normalize(&self, rel: &str) -> String {
        let rel = self.normalize_pid(rel);
        // A thread's own directory shows what the process's does.
        if let Some(rest) = rel.strip_prefix("self/task/") {
            let (tid, tail) = rest.split_once('/').unwrap_or((rest, ""));
            if tid
                .parse::<u32>()
                .is_ok_and(|t| self.data.tids.contains(&t))
            {
                return if tail.is_empty() {
                    "self".to_string()
                } else {
                    format!("self/{tail}")
                };
            }
        }
        rel
    }

    /// `rel` with the reader's own `<pid>` spelled `self`.
    fn normalize_pid(&self, rel: &str) -> String {
        if self.data.pid != 0 {
            let pid_s = self.data.pid.to_string();
            if rel == pid_s {
                return "self".to_string();
            }
            if let Some(rest) = rel
                .strip_prefix(pid_s.as_str())
                .and_then(|r| r.strip_prefix('/'))
            {
                return format!("self/{rest}");
            }
        }
        rel.to_string()
    }

    /// The rendered bytes of a readable file, or `None` if `rel` is not a
    /// regular file (a directory, a symlink, or unknown).
    fn content(&self, rel: &str) -> Option<Vec<u8>> {
        match rel {
            "mounts" | "self/mounts" if !self.data.mounts.is_empty() => {
                return Some(mounts_body(&self.data.mounts).into_bytes());
            }
            "self/mountinfo" if !self.data.mounts.is_empty() => {
                return Some(mountinfo_body(&self.data.mounts).into_bytes());
            }
            // A map can legitimately be empty; `setgroups` says the kernel has spoken.
            "self/uid_map" if !self.data.setgroups.is_empty() => {
                return Some(self.data.uid_map.clone().into_bytes());
            }
            "self/gid_map" if !self.data.setgroups.is_empty() => {
                return Some(self.data.gid_map.clone().into_bytes());
            }
            "self/setgroups" if !self.data.setgroups.is_empty() => {
                return Some(self.data.setgroups.clone().into_bytes());
            }
            "self/timens_offsets" => {
                return Some(self.data.timens_offsets.clone().into_bytes());
            }
            "self/limits" if !self.data.limits.is_empty() => {
                return Some(self.data.limits.clone().into_bytes());
            }
            "self/cgroup" if !self.data.cgroup.is_empty() => {
                return Some(self.data.cgroup.clone().into_bytes());
            }
            "sys/kernel/hostname" if !self.data.hostname.is_empty() => {
                return Some(format!("{}\n", self.data.hostname).into_bytes());
            }
            "sys/kernel/seccomp/actions_logged" if self.data.seccomp_logged.is_some() => {
                let names = self.data.seccomp_logged.as_deref().unwrap_or_default();
                return Some(format!("{names}\n").into_bytes());
            }
            "sys/kernel/domainname" if !self.data.domainname.is_empty() => {
                return Some(format!("{}\n", self.data.domainname).into_bytes());
            }
            "cpuinfo" => {
                return Some(cpuinfo_body(self.data.nproc, self.data.arch).into_bytes());
            }
            "partitions" | "diskstats" if self.disks.is_some() => {
                let disks = self.disks.as_ref()?.lock().unwrap();
                let body = if rel == "partitions" {
                    disks.partitions()
                } else {
                    disks.diskstats()
                };
                return Some(body.into_bytes());
            }
            "stat" => return Some(stat_body(self.data.nproc).into_bytes()),
            _ => {}
        }
        if let Some(bytes) = static_content(rel) {
            return Some(bytes.to_vec());
        }
        self.self_content(rel)
    }

    /// Render a per-process file. `self/exe`, `self/cwd` and `self/fd/<n>`
    /// are deliberately excluded — they are symlinks, not readable files.
    fn self_content(&self, rel: &str) -> Option<Vec<u8>> {
        let d = &self.data;
        let body = match rel {
            "self/cmdline" => return Some(d.cmdline.clone()),
            "self/maps" => {
                let text = if d.maps.is_empty() {
                    default_maps(&d.exe)
                } else {
                    d.maps.clone()
                };
                return Some(text.into_bytes());
            }
            "self/auxv" => return Some(d.auxv.clone()),
            "self/syscall" => d.syscall.clone(),
            "self/comm" => format!("{}\n", d.comm),
            "self/stat" => self_stat_body(d),
            "self/statm" => statm_body(d),
            "self/sched" => sched_body(d),
            "self/smaps" => {
                let text = if d.maps.is_empty() {
                    default_maps(&d.exe)
                } else {
                    d.maps.clone()
                };
                smaps_body(&text)
            }
            "self/status" => {
                let ids = if d.status_ids.is_empty() {
                    format!(
                        "Uid:\t{uid}\t{uid}\t{uid}\t{uid}\nGid:\t{gid}\t{gid}\t{gid}\t{gid}\n",
                        uid = d.uid,
                        gid = d.gid
                    )
                } else {
                    d.status_ids.clone()
                };
                format!(
                    "Name:\t{comm}\n\
                     State:\t{state} ({state_desc})\n\
                     Tgid:\t{pid}\n\
                     Pid:\t{pid}\n\
                     PPid:\t{ppid}\n\
                     TracerPid:\t{tracer}\n\
                     {ids}\
                     Threads:\t{threads}\n\
                     VmSize:\t{vsize} kB\n\
                     VmRSS:\t{rss} kB\n\
                     {security}",
                    comm = d.comm,
                    state = d.state,
                    state_desc = state_desc(d.state),
                    pid = d.pid,
                    ppid = d.ppid,
                    tracer = d.tracer,
                    threads = d.threads,
                    vsize = d.vm_size_kb,
                    rss = d.vm_rss_kb,
                    security = d.status_security,
                )
            }
            _ => return None,
        };
        Some(body.into_bytes())
    }

    /// `self/fd/<n>`'s symlink target, if that descriptor is published.
    fn fd_target(&self, n: u32) -> Option<&str> {
        self.data
            .fds
            .iter()
            .find(|(fd, _)| *fd == n)
            .map(|(_, target)| target.as_str())
    }
}

/// Byte body of a static file, or `None` if `rel` is not one.
fn static_content(rel: &str) -> Option<&'static [u8]> {
    let text = match rel {
        "meminfo" => MEMINFO,
        "version" => VERSION,
        "uptime" => UPTIME,
        "loadavg" => LOADAVG,
        "filesystems" => FILESYSTEMS,
        "mounts" | "self/mounts" => MOUNTS,
        "cmdline" => CMDLINE,
        "sys/kernel/ostype" => OSTYPE,
        "sys/kernel/osrelease" => OSRELEASE,
        "sys/kernel/hostname" => HOSTNAME,
        "sys/kernel/seccomp/actions_avail" => SECCOMP_ACTIONS_AVAIL,
        "sys/kernel/seccomp/actions_logged" => SECCOMP_ACTIONS_LOGGED,
        "sys/kernel/domainname" => "(none)\n",
        "sys/kernel/overflowuid" | "sys/kernel/overflowgid" => "65534\n",
        "sys/kernel/pid_max" => PID_MAX,
        "sys/kernel/version" => KERNEL_VERSION,
        "sys/kernel/threads-max" => THREADS_MAX,
        "sys/kernel/ngroups_max" => NGROUPS_MAX,
        "sys/kernel/cap_last_cap" => CAP_LAST_CAP,
        "sys/kernel/sched_rr_timeslice_ms" => SCHED_RR_TIMESLICE_MS,
        "sys/kernel/random/boot_id" => BOOT_ID,
        "sys/kernel/random/uuid" => RANDOM_UUID,
        "sys/kernel/random/entropy_avail" => ENTROPY_AVAIL,
        "sys/kernel/random/poolsize" => POOLSIZE,
        "net/tcp" => NET_TCP_HEADER,
        "net/tcp6" => NET_TCP6_HEADER,
        "net/udp" => NET_UDP_HEADER,
        "net/udp6" => NET_UDP6_HEADER,
        "net/unix" => NET_UNIX,
        "net/dev" => NET_DEV,
        "net/route" => NET_ROUTE,
        "net/snmp" => NET_SNMP,
        "net/protocols" => NET_PROTOCOLS,
        "sys/net/core/somaxconn" => SOMAXCONN,
        "sys/net/ipv4/tcp_rmem" => TCP_RMEM,
        "sys/net/ipv4/tcp_wmem" => TCP_WMEM,
        "sys/net/ipv4/ip_local_port_range" => IP_LOCAL_PORT_RANGE,
        "sys/vm/overcommit_memory" => OVERCOMMIT_MEMORY,
        "sys/vm/max_map_count" => MAX_MAP_COUNT,
        "sys/fs/file-max" => FILE_MAX,
        "sys/fs/nr_open" => NR_OPEN,
        "diskstats" => DISKSTATS,
        "partitions" => PARTITIONS,
        "swaps" => SWAPS,
        "modules" => MODULES,
        "devices" => DEVICES,
        "kallsyms" => KALLSYMS,
        "keys" => KEYS,
        "key-users" => KEY_USERS,
        "locks" => LOCKS,
        "vmstat" => VMSTAT,
        "zoneinfo" => ZONEINFO,
        "buddyinfo" => BUDDYINFO,
        "consoles" => CONSOLES,
        "self/mountinfo" => MOUNTINFO,
        "self/limits" => LIMITS,
        "self/cgroup" => "0::/\n",
        "self/io" => IO,
        "self/oom_score" => OOM_SCORE,
        "self/oom_score_adj" => OOM_SCORE_ADJ,
        "self/wchan" => WCHAN,
        "self/environ" => ENVIRON,
        "self/schedstat" => SCHEDSTAT,
        "self/personality" => PERSONALITY,
        "self/setgroups" => SETGROUPS,
        "self/gid_map" | "self/uid_map" => UID_GID_MAP,
        _ => return None,
    };
    Some(text.as_bytes())
}

/// Render `cpuinfo`, one block per core, matching the injected `nproc`
/// (never fewer than one core).
fn cpuinfo_body(nproc: usize, arch: Option<crate::abi::Arch>) -> String {
    let arch = arch.unwrap_or(if cfg!(target_arch = "x86_64") {
        crate::abi::Arch::X86_64
    } else {
        crate::abi::Arch::Aarch64
    });
    let mut out = String::new();
    for i in 0..nproc.max(1) {
        write_cpuinfo_block(&mut out, i, arch);
    }
    out
}

/// arm64 `/proc/cpuinfo` feature names, indexed by `AT_HWCAP` bit
/// (`hwcap_str[]` in `arch/arm64/kernel/cpuinfo.c`).
const AARCH64_HWCAP_NAMES: [&str; 32] = [
    "fp", "asimd", "evtstrm", "aes", "pmull", "sha1", "sha2", "crc32", "atomics", "fphp",
    "asimdhp", "cpuid", "asimdrdm", "jscvt", "fcma", "lrcpc", "dcpop", "sha3", "sm3", "sm4",
    "asimddp", "sha512", "sve", "asimdfhm", "dit", "uscat", "ilrcpc", "flagm", "ssbs", "sb",
    "paca", "pacg",
];

/// arm64 `/proc/cpuinfo` feature names, indexed by `AT_HWCAP2` bit.
const AARCH64_HWCAP2_NAMES: [&str; 64] = [
    "dcpodp",
    "sve2",
    "sveaes",
    "svepmull",
    "svebitperm",
    "svesha3",
    "svesm4",
    "flagm2",
    "frint",
    "svei8mm",
    "svef32mm",
    "svef64mm",
    "svebf16",
    "i8mm",
    "bf16",
    "dgh",
    "rng",
    "bti",
    "mte",
    "ecv",
    "afp",
    "rpres",
    "mte3",
    "sme",
    "smei16i64",
    "smef64f64",
    "smei8i32",
    "smef16f32",
    "smeb16f32",
    "smef32f32",
    "smefa64",
    "wfxt",
    "ebf16",
    "sveebf16",
    "cssc",
    "rprfm",
    "sve2p1",
    "sme2",
    "sme2p1",
    "smei16i32",
    "smebi32i32",
    "smeb16b16",
    "smef16f16",
    "mops",
    "hbc",
    "sveb16b16",
    "lrcpc3",
    "lse128",
    "fpmr",
    "lut",
    "faminmax",
    "f8cvt",
    "f8fma",
    "f8dp4",
    "f8dp2",
    "f8e4m3",
    "f8e5m2",
    "smelutv2",
    "smef8f16",
    "smef8f32",
    "smesf8fma",
    "smesf8dp4",
    "smesf8dp2",
    "poe",
];

/// The arm64 `Features` line: every capability the loader puts in
/// `AT_HWCAP`/`AT_HWCAP2` (see [`crate::loader::hwcaps`]), named as Linux
/// names them, in Linux's order.
fn aarch64_features() -> String {
    let (hwcap, hwcap2) = crate::loader::hwcaps(crate::abi::Arch::Aarch64);
    let names = AARCH64_HWCAP_NAMES
        .iter()
        .enumerate()
        .filter(|&(bit, _)| hwcap >> bit & 1 == 1)
        .chain(
            AARCH64_HWCAP2_NAMES
                .iter()
                .enumerate()
                .filter(|&(bit, _)| hwcap2 >> bit & 1 == 1),
        )
        .map(|(_, n)| *n);
    names.collect::<Vec<_>>().join(" ")
}

/// Append one core's block to `cpuinfo`, in the shape of the guest's
/// architecture.
fn write_cpuinfo_block(out: &mut String, i: usize, arch: crate::abi::Arch) {
    if arch == crate::abi::Arch::X86_64 {
        let _ = write!(
            out,
            "processor\t: {i}\n\
             vendor_id\t: GenuineIntel\n\
             cpu family\t: 6\n\
             model name\t: nixvm Virtual CPU\n\
             flags\t\t: fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov \
             pat pse36 clflush mmx fxsr sse sse2 ht syscall nx lm constant_tsc \
             rep_good nopl cpuid pni pclmulqdq ssse3 cx16 sse4_1 sse4_2 x2apic \
             popcnt aes xsave avx rdrand hypervisor lahf_lm abm 3dnowprefetch\n\
             bogomips\t: 100.00\n\n"
        );
    } else {
        let _ = write!(
            out,
            "processor\t: {i}\n\
             BogoMIPS\t: 100.00\n\
             Features\t: {}\n\
             CPU implementer\t: 0x41\n\
             CPU architecture: 8\n\
             CPU variant\t: 0x0\n\
             CPU part\t: 0xd08\n\
             CPU revision\t: 0\n\n",
            aarch64_features()
        );
    }
}

/// Render `/proc/stat`, with one `cpuN` line per injected core.
fn stat_body(nproc: usize) -> String {
    let mut out = String::from("cpu  0 0 0 0 0 0 0 0 0 0\n");
    for i in 0..nproc.max(1) {
        let _ = writeln!(out, "cpu{i} 0 0 0 0 0 0 0 0 0 0");
    }
    out.push_str(
        "intr 0\n\
         ctxt 0\n\
         btime 0\n\
         processes 1\n\
         procs_running 1\n\
         procs_blocked 0\n",
    );
    out
}

/// A minimal but plausible `self/maps` body used when the kernel hasn't
/// injected real mapping data: a text/data image mapping for `exe`, plus a
/// heap and a stack region — the ranges a userland typically checks for.
fn default_maps(exe: &str) -> String {
    format!(
        "00400000-00401000 r-xp 00000000 00:00 0                          {exe}\n\
         00600000-00601000 rw-p 00000000 00:00 0                          {exe}\n\
         01000000-01021000 rw-p 00000000 00:00 0                          [heap]\n\
         7ffffffde000-7ffffffff000 rw-p 00000000 00:00 0                  [stack]\n"
    )
}

/// Human description of a single-letter run state, for `status`'s `State:`
/// line.
fn state_desc(state: char) -> &'static str {
    match state {
        'R' => "running",
        'S' => "sleeping",
        'D' => "disk sleep",
        'Z' => "zombie",
        'T' => "stopped",
        't' => "tracing stop",
        _ => "unknown",
    }
}

/// Push `val` onto `f` `n` times; used to fill the many always-zero `stat`
/// fields without a wall of repeated literals.
fn push_n(f: &mut Vec<String>, val: &str, n: usize) {
    for _ in 0..n {
        f.push(val.to_string());
    }
}

/// Render `self/stat`'s full 52-field line (see proc(5)), with the fields
/// nixvm actually tracks filled in and the rest zeroed.
fn self_stat_body(d: &ProcData) -> String {
    let pid = d.pid.to_string();
    let vsize = d.vm_size_kb * 1024;
    let rss = d.vm_rss_kb / 4; // rss is reported in pages, not kB
    let mut f: Vec<String> = Vec::with_capacity(52);
    f.push(pid.clone()); // 1 pid
    f.push(format!("({})", d.comm)); // 2 comm
    f.push(d.state.to_string()); // 3 state
    f.push(d.ppid.to_string()); // 4 ppid
    f.push(pid.clone()); // 5 pgrp
    f.push(pid); // 6 session
    f.push("0".to_string()); // 7 tty_nr
    f.push("-1".to_string()); // 8 tpgid
    f.push("0".to_string()); // 9 flags
    push_n(&mut f, "0", 4); // 10-13 minflt cminflt majflt cmajflt
    push_n(&mut f, "0", 4); // 14-17 utime stime cutime cstime
    f.push("20".to_string()); // 18 priority
    f.push("0".to_string()); // 19 nice
    f.push(d.threads.to_string()); // 20 num_threads
    f.push("0".to_string()); // 21 itrealvalue
    f.push("0".to_string()); // 22 starttime
    f.push(vsize.to_string()); // 23 vsize
    f.push(rss.to_string()); // 24 rss
    f.push("18446744073709551615".to_string()); // 25 rsslim
    push_n(&mut f, "0", 5); // 26-30 startcode endcode startstack kstkesp kstkeip
    push_n(&mut f, "0", 4); // 31-34 signal blocked sigignore sigcatch
    f.push("0".to_string()); // 35 wchan
    f.push("0".to_string()); // 36 nswap
    f.push("0".to_string()); // 37 cnswap
    f.push("17".to_string()); // 38 exit_signal
    f.push("0".to_string()); // 39 processor
    push_n(&mut f, "0", 2); // 40-41 rt_priority policy
    f.push("0".to_string()); // 42 delayacct_blkio_ticks
    push_n(&mut f, "0", 2); // 43-44 guest_time cguest_time
    push_n(&mut f, "0", 2); // 45-46 start_data end_data
    f.push("0".to_string()); // 47 start_brk
    push_n(&mut f, "0", 4); // 48-51 arg_start arg_end env_start env_end
    f.push("0".to_string()); // 52 exit_code
    debug_assert_eq!(f.len(), 52);
    format!("{}\n", f.join(" "))
}

/// Render `self/statm`'s 7 whitespace-separated page counts (see `proc(5)`):
/// `size resident shared text lib data dt`. Sizes are derived from the
/// injected kB figures assuming 4 kB pages.
fn statm_body(d: &ProcData) -> String {
    const PAGE_KB: u64 = 4;
    let size = (d.vm_size_kb / PAGE_KB).max(1);
    let resident = (d.vm_rss_kb / PAGE_KB).max(1);
    let shared = 0u64;
    let text = 1u64.min(size);
    let lib = 0u64;
    let data = size.saturating_sub(text);
    let dt = 0u64;
    format!("{size} {resident} {shared} {text} {lib} {data} {dt}\n")
}

/// Render `self/sched`'s human-readable scheduling-statistics dump (see
/// `sched_show_task()` in the kernel): a header line naming the process,
/// followed by the handful of `se.*`/`nr_*` fields userland tools actually
/// scrape, all zeroed since a synthetic process has no real scheduling
/// history.
fn sched_body(d: &ProcData) -> String {
    format!(
        "{comm} ({pid}, #threads: {threads})\n\
         ------------------------------------------------------------------\n\
         se.exec_start                               :               0.000000\n\
         se.vruntime                                 :               0.000000\n\
         se.sum_exec_runtime                         :               0.000000\n\
         nr_switches                                 :                      0\n\
         nr_voluntary_switches                       :                      0\n\
         nr_involuntary_switches                     :                      0\n\
         se.load.weight                              :                   1024\n\
         policy                                      :                      0\n\
         prio                                        :                    120\n",
        comm = d.comm,
        pid = d.pid,
        threads = d.threads,
    )
}

/// The `VmFlags` shorthand for a `self/maps` permission token (e.g. `r-xp`),
/// approximating what the kernel reports for a mapping with those
/// permissions.
fn vmflags_for(perms: &str) -> String {
    let mut flags = Vec::new();
    if perms.contains('r') {
        flags.push("rd");
    }
    if perms.contains('w') {
        flags.push("wr");
    }
    if perms.contains('x') {
        flags.push("ex");
    }
    flags.push("mr");
    flags.push("mw");
    flags.push("me");
    if perms.contains('w') {
        flags.push("dw");
    }
    flags.join(" ")
}

/// Render `self/smaps`: every line of `maps` (see [`default_maps`]) followed
/// by its per-mapping statistics block, sized from the line's own address
/// range.
fn smaps_body(maps: &str) -> String {
    let mut out = String::new();
    for line in maps.lines() {
        if line.is_empty() {
            continue;
        }
        let _ = writeln!(out, "{line}");
        let mut fields = line.split_whitespace();
        let range = fields.next().unwrap_or("");
        let perms = fields.next().unwrap_or("----");
        let size_kb = range
            .split_once('-')
            .and_then(|(s, e)| {
                let s = u64::from_str_radix(s, 16).ok()?;
                let e = u64::from_str_radix(e, 16).ok()?;
                Some(e.saturating_sub(s) / 1024)
            })
            .unwrap_or(4)
            .max(4);
        let writable = perms.contains('w');
        let (private_clean, private_dirty) = if writable { (0, size_kb) } else { (size_kb, 0) };
        let zero = 0u64;
        let page_kb = 4u64;
        let flags = vmflags_for(perms);
        let _ = write!(
            out,
            "Size:           {size_kb:>10} kB\n\
             KernelPageSize: {page_kb:>10} kB\n\
             MMUPageSize:    {page_kb:>10} kB\n\
             Rss:            {size_kb:>10} kB\n\
             Pss:            {size_kb:>10} kB\n\
             Shared_Clean:   {zero:>10} kB\n\
             Shared_Dirty:   {zero:>10} kB\n\
             Private_Clean:  {private_clean:>10} kB\n\
             Private_Dirty:  {private_dirty:>10} kB\n\
             Referenced:     {size_kb:>10} kB\n\
             Anonymous:      {zero:>10} kB\n\
             AnonHugePages:  {zero:>10} kB\n\
             Swap:           {zero:>10} kB\n\
             SwapPss:        {zero:>10} kB\n\
             Locked:         {zero:>10} kB\n\
             VmFlags: {flags}\n"
        );
    }
    out
}

/// The inode for a known static path (its 1-based position in [`PATHS`]).
fn inode_of(rel: &str) -> Option<u64> {
    PATHS.iter().position(|p| *p == rel).map(|i| i as u64 + 1)
}

/// Deterministic inode for a `self/fd/<n>` symlink, kept clear of the
/// (small, fixed) range [`inode_of`] hands out.
fn fd_inode(fd: u32) -> u64 {
    100_000 + u64::from(fd)
}

/// Whether `rel` (already [`ProcFs::normalize`]d) names one of the fixed
/// directories.
fn is_dir(rel: &str) -> bool {
    matches!(
        rel,
        "" | "self"
            | "self/fd"
            | "self/ns"
            | "self/task"
            | "sys"
            | "sys/kernel"
            | "sys/kernel/random"
            | "sys/kernel/seccomp"
            | "net"
            | "sys/net"
            | "sys/net/core"
            | "sys/net/ipv4"
            | "sys/vm"
            | "sys/fs"
    )
}

/// Build a directory entry, looking the inode up from the full path.
fn entry(name: &str, path: &str, kind: NodeKind) -> DirEntry {
    DirEntry {
        name: name.to_string(),
        kind,
        inode: inode_of(path).unwrap_or(0),
    }
}

/// Build directory entries for a flat list of file names under `prefix`
/// (e.g. `"net"` + `["tcp", ...]` -> `net/tcp`, …) — the common shape for
/// the many single-level tunable directories this backend exposes.
fn list_files(prefix: &str, names: &[&str]) -> Vec<DirEntry> {
    names
        .iter()
        .map(|n| {
            let path = format!("{prefix}/{n}");
            entry(n, &path, NodeKind::File)
        })
        .collect()
}

/// Copy `data[off..]` into `buf`, returning the byte count (0 at or past EOF).
fn read_slice(data: &[u8], off: u64, buf: &mut [u8]) -> usize {
    let off = off as usize;
    if off >= data.len() {
        return 0;
    }
    let n = buf.len().min(data.len() - off);
    buf[..n].copy_from_slice(&data[off..off + n]);
    n
}

fn enoent() -> io::Error {
    io::Error::from_raw_os_error(2) // ENOENT
}
fn eisdir() -> io::Error {
    io::Error::from_raw_os_error(21) // EISDIR
}
fn enotdir() -> io::Error {
    io::Error::from_raw_os_error(20) // ENOTDIR
}
fn einval() -> io::Error {
    io::Error::from_raw_os_error(22) // EINVAL
}

impl MountFs for ProcFs {
    fn fs_type(&self) -> &'static str {
        "proc"
    }

    fn attach_disks(&mut self, disks: &super::blockdev::SharedDisks) {
        self.disks = Some(disks.clone());
    }

    fn as_procfs_mut(&mut self) -> Option<&mut ProcFs> {
        Some(self)
    }

    fn stat(&mut self, rel: &str) -> Option<Attrs> {
        let rel = self.normalize(rel);
        let rel = rel.as_str();

        if let Some((task, tail)) = self.other_task(rel) {
            return Self::other_task_stat(task, tail);
        }

        if let Some(n) = rel
            .strip_prefix("self/fd/")
            .and_then(|s| s.parse::<u32>().ok())
        {
            let target = self.fd_target(n)?;
            return Some(Attrs {
                kind: NodeKind::Symlink,
                size: target.len() as u64,
                mode: S_IFLNK | 0o777,
                uid: 0,
                gid: 0,
                atime: 0,
                mtime: 0,
                inode: fd_inode(n),
                nlink: 1,
                rdev: 0,
                dev: 0,
            });
        }

        let inode = inode_of(rel)?;
        let (kind, mode, size) = if is_dir(rel) {
            (NodeKind::Dir, S_IFDIR | 0o555, 0)
        } else if rel == "self/exe" {
            (
                NodeKind::Symlink,
                S_IFLNK | 0o777,
                self.data.exe.len() as u64,
            )
        } else if rel == "self/cwd" {
            (
                NodeKind::Symlink,
                S_IFLNK | 0o777,
                self.data.cwd.len() as u64,
            )
        } else if rel == "self/root" {
            (NodeKind::Symlink, S_IFLNK | 0o777, ROOT_TARGET.len() as u64)
        } else if rel == "thread-self" {
            (NodeKind::Symlink, S_IFLNK | 0o777, 0)
        } else if rel
            .strip_prefix("self/ns/")
            .is_some_and(|n| NS_FILES.contains(&n))
        {
            // The kernel answers opens and readlinks of these itself, since the
            // link names a namespace rather than anything in this filesystem.
            (NodeKind::Symlink, S_IFLNK | 0o777, 0)
        } else {
            let data = self.content(rel)?;
            (NodeKind::File, S_IFREG | 0o444, data.len() as u64)
        };
        let (uid, gid) = if rel == "self" || rel.starts_with("self/") {
            self.data.owner
        } else {
            (0, 0)
        };
        // A directory's links are `.`, `..` and its subdirectories: a process's
        // `task/` has one per thread, which is how a program counts them.
        let nlink = if rel == "self/task" {
            2 + self.data.tids.len().max(1) as u32
        } else {
            1
        };
        Some(Attrs {
            kind,
            size,
            mode,
            uid,
            gid,
            atime: 0,
            mtime: 0,
            inode,
            nlink,
            rdev: 0,
            dev: 0,
        })
    }

    fn read_at(&mut self, rel: &str, off: u64, buf: &mut [u8]) -> io::Result<usize> {
        let rel = self.normalize(rel);
        let rel = rel.as_str();
        if let Some((task, tail)) = self.other_task(rel) {
            let body = task.file(tail).ok_or_else(enoent)?;
            return Ok(read_slice(&body, off, buf));
        }
        if is_dir(rel) {
            return Err(eisdir());
        }
        if rel == "self/exe"
            || rel == "self/cwd"
            || rel == "self/root"
            || rel.starts_with("self/fd/")
            || rel.starts_with("self/ns/")
        {
            return Err(einval()); // read on a symlink; use readlink
        }
        match self.content(rel) {
            Some(data) => Ok(read_slice(&data, off, buf)),
            None => Err(enoent()),
        }
    }

    fn readdir(&mut self, rel: &str) -> io::Result<Vec<DirEntry>> {
        let rel = self.normalize(rel);
        let rel = rel.as_str();
        if let Some((task, tail)) = self.other_task(rel) {
            let of = |name: &str, kind| DirEntry {
                name: name.to_string(),
                kind,
                inode: task_inode(task.pid, 6),
            };
            return match tail {
                "" => {
                    let mut out: Vec<DirEntry> =
                        TASK_FILES.iter().map(|n| of(n, NodeKind::File)).collect();
                    out.push(of("ns", NodeKind::Dir));
                    out.push(of("fd", NodeKind::Dir));
                    out.push(of("task", NodeKind::Dir));
                    Ok(out)
                }
                "ns" => Ok(NS_FILES.iter().map(|n| of(n, NodeKind::Symlink)).collect()),
                "fd" | "task" => Ok(Vec::new()),
                _ => Err(enotdir()),
            };
        }
        match rel {
            "" => {
                let mut out: Vec<DirEntry> = ROOT_FILES
                    .iter()
                    .map(|n| entry(n, n, NodeKind::File))
                    .collect();
                out.push(entry("self", "self", NodeKind::Dir));
                for task in self.data.tasks.iter().filter(|t| t.pid != self.data.pid) {
                    out.push(DirEntry {
                        name: task.pid.to_string(),
                        kind: NodeKind::Dir,
                        inode: task_inode(task.pid, 0),
                    });
                }
                if self.data.pid != 0 {
                    out.push(DirEntry {
                        name: self.data.pid.to_string(),
                        kind: NodeKind::Dir,
                        inode: inode_of("self").unwrap_or(0),
                    });
                }
                out.push(entry("sys", "sys", NodeKind::Dir));
                out.push(entry("net", "net", NodeKind::Dir));
                Ok(out)
            }
            "self/task" => {
                let tids = if self.data.tids.is_empty() {
                    vec![self.data.pid]
                } else {
                    self.data.tids.clone()
                };
                Ok(tids
                    .into_iter()
                    .map(|t| DirEntry {
                        name: t.to_string(),
                        kind: NodeKind::Dir,
                        inode: task_inode(t, 7),
                    })
                    .collect())
            }
            "self" => Ok(SELF_FILES
                .iter()
                .map(|n| {
                    let path = format!("self/{n}");
                    let kind = match *n {
                        "exe" | "cwd" | "root" => NodeKind::Symlink,
                        "fd" | "ns" | "task" => NodeKind::Dir,
                        _ => NodeKind::File,
                    };
                    entry(n, &path, kind)
                })
                .collect()),
            "self/ns" => Ok(NS_FILES
                .iter()
                .map(|n| entry(n, &format!("self/ns/{n}"), NodeKind::Symlink))
                .collect()),
            "self/fd" => Ok(self
                .data
                .fds
                .iter()
                .map(|(fd, _)| DirEntry {
                    name: fd.to_string(),
                    kind: NodeKind::Symlink,
                    inode: fd_inode(*fd),
                })
                .collect()),
            "sys" => Ok(vec![
                entry("kernel", "sys/kernel", NodeKind::Dir),
                entry("net", "sys/net", NodeKind::Dir),
                entry("vm", "sys/vm", NodeKind::Dir),
                entry("fs", "sys/fs", NodeKind::Dir),
            ]),
            "sys/kernel" => {
                let mut out = list_files("sys/kernel", SYS_KERNEL_FILES);
                out.push(entry("random", "sys/kernel/random", NodeKind::Dir));
                out.push(entry("seccomp", "sys/kernel/seccomp", NodeKind::Dir));
                Ok(out)
            }
            "sys/kernel/seccomp" => Ok(list_files(
                "sys/kernel/seccomp",
                &["actions_avail", "actions_logged"],
            )),
            "sys/kernel/random" => Ok(list_files("sys/kernel/random", SYS_KERNEL_RANDOM_FILES)),
            "net" => Ok(list_files("net", NET_FILES)),
            "sys/net" => Ok(vec![
                entry("core", "sys/net/core", NodeKind::Dir),
                entry("ipv4", "sys/net/ipv4", NodeKind::Dir),
            ]),
            "sys/net/core" => Ok(list_files("sys/net/core", SYS_NET_CORE_FILES)),
            "sys/net/ipv4" => Ok(list_files("sys/net/ipv4", SYS_NET_IPV4_FILES)),
            "sys/vm" => Ok(list_files("sys/vm", SYS_VM_FILES)),
            "sys/fs" => Ok(list_files("sys/fs", SYS_FS_FILES)),
            _ if rel.starts_with("self/fd/") => Err(enotdir()),
            _ if inode_of(rel).is_some() => Err(enotdir()),
            _ => Err(enoent()),
        }
    }

    fn readlink(&mut self, rel: &str) -> io::Result<String> {
        let rel = self.normalize(rel);
        let rel = rel.as_str();
        if rel == "self/exe" {
            return Ok(self.data.exe.clone());
        }
        if rel == "self/cwd" {
            return Ok(self.data.cwd.clone());
        }
        if rel == "self/root" {
            return Ok(ROOT_TARGET.to_string());
        }
        if rel == "thread-self" {
            let tid = if self.data.tid == 0 {
                self.data.pid
            } else {
                self.data.tid
            };
            return Ok(format!("{}/task/{tid}", self.data.pid));
        }
        if let Some(n) = rel
            .strip_prefix("self/fd/")
            .and_then(|s| s.parse::<u32>().ok())
        {
            return self.fd_target(n).map(str::to_string).ok_or_else(enoent);
        }
        if inode_of(rel).is_some() {
            return Err(einval()); // not a symlink
        }
        Err(enoent())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read an entire file by looping `read_at` to EOF.
    fn read_all(fs: &mut ProcFs, path: &str) -> Vec<u8> {
        let mut out = Vec::new();
        let mut off = 0u64;
        let mut buf = [0u8; 64];
        loop {
            let n = fs.read_at(path, off, &mut buf).unwrap();
            if n == 0 {
                break;
            }
            out.extend_from_slice(&buf[..n]);
            off += n as u64;
        }
        out
    }

    /// A fully populated sample, exercising every field the kernel is meant
    /// to eventually inject.
    fn sample() -> ProcData {
        ProcData {
            tasks: Vec::new(),
            uid_map: String::new(),
            gid_map: String::new(),
            setgroups: String::new(),
            timens_offsets: String::new(),
            limits: String::new(),
            hostname: String::new(),
            domainname: String::new(),
            cgroup: String::new(),
            mounts: Vec::new(),
            cmdline: b"prog\0--flag\0".to_vec(),
            exe: "/usr/bin/prog".to_string(),
            cwd: "/home/user".to_string(),
            maps: String::new(),
            comm: "prog".to_string(),
            argv0: "prog".to_string(),
            pid: 42,
            ppid: 7,
            tracer: 0,
            uid: 1000,
            gid: 1000,
            owner: (1000, 1000),
            tid: 0,
            tids: Vec::new(),
            status_ids: String::new(),
            status_security: String::new(),
            state: 'R',
            threads: 3,
            vm_size_kb: 8192,
            vm_rss_kb: 2048,
            fds: vec![
                (0, "/dev/tty".to_string()),
                (1, "/dev/tty".to_string()),
                (2, "pipe:[12345]".to_string()),
            ],
            nproc: 4,
            auxv: Vec::new(),
            arch: None,
            syscall: String::new(),
            seccomp_logged: None,
        }
    }

    fn with_tasks() -> ProcFs {
        let mut fs = ProcFs::new(1);
        fs.data.pid = 1;
        fs.data.tasks = vec![ProcTask {
            pid: 7,
            ppid: 1,
            tracer: 0,
            comm: "worker".to_string(),
            cmdline: b"worker\0-x\0".to_vec(),
            state: 'S',
            uid: 5,
            gid: 6,
            status_ids: String::new(),
            id_files: Default::default(),
            cgroup: "0::/a\n".to_string(),
            sandbox: Sandbox::default(),
            syscall: String::new(),
        }];
        fs
    }

    #[test]
    fn another_task_has_a_directory_of_its_own() {
        let mut fs = with_tasks();
        assert_eq!(fs.stat("7").unwrap().kind, NodeKind::Dir);
        assert_eq!(fs.stat("7").unwrap().uid, 5);
        assert_eq!(fs.stat("7/ns").unwrap().kind, NodeKind::Dir);
        assert_eq!(fs.stat("7/ns/pid").unwrap().kind, NodeKind::Symlink);
        assert!(fs.stat("7/nope").is_none());
        assert!(fs.stat("8").is_none(), "not visible, so not there");
        assert_eq!(read_all(&mut fs, "7/comm"), b"worker\n");
        assert_eq!(read_all(&mut fs, "7/cmdline"), b"worker\0-x\0");
        let stat = String::from_utf8(read_all(&mut fs, "7/stat")).unwrap();
        assert!(stat.starts_with("7 (worker) S 1 "), "{stat}");
        assert_eq!(stat.split_whitespace().count(), 52);
        let status = String::from_utf8(read_all(&mut fs, "7/status")).unwrap();
        assert!(status.contains("Uid:\t5\t5\t5\t5"), "{status}");
    }

    #[test]
    fn listing_proc_shows_every_visible_task() {
        let mut fs = with_tasks();
        let names: Vec<_> = fs
            .readdir("")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert!(names.contains(&"7".to_string()));
        let inner: Vec<_> = fs
            .readdir("7")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert!(inner.contains(&"ns".to_string()) && inner.contains(&"stat".to_string()));
        assert_eq!(fs.readdir("7/ns").unwrap().len(), NS_FILES.len());
    }

    #[test]
    fn version_contains_linux() {
        let mut fs = ProcFs::new(1);
        let data = read_all(&mut fs, "version");
        assert!(String::from_utf8_lossy(&data).contains("Linux"));
    }

    #[test]
    fn root_readdir_lists_files_and_dirs() {
        let mut fs = ProcFs::new(1);
        let names: Vec<String> = fs
            .readdir("")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert!(names.contains(&"cpuinfo".to_string()));
        assert!(names.contains(&"self".to_string()));
        assert!(names.contains(&"sys".to_string()));
        // Default ProcData carries pid 1, so its numeric alias is listed too.
        assert!(names.contains(&"1".to_string()));
    }

    #[test]
    fn self_exe_is_a_symlink() {
        let mut fs = ProcFs::new(1);
        let attrs = fs.stat("self/exe").unwrap();
        assert_eq!(attrs.kind, NodeKind::Symlink);
        assert_eq!(attrs.mode, S_IFLNK | 0o777);
        // readlink resolves to the exe path.
        assert_eq!(fs.readlink("self/exe").unwrap(), "/bin/busybox");
    }

    #[test]
    fn set_self_changes_cmdline() {
        let mut fs = ProcFs::new(1);
        // Placeholder cmdline is empty.
        assert_eq!(read_all(&mut fs, "self/cmdline").len(), 0);
        fs.set_self(sample());
        assert_eq!(read_all(&mut fs, "self/cmdline"), b"prog\0--flag\0");
        assert_eq!(fs.readlink("self/exe").unwrap(), "/usr/bin/prog");
        let comm = read_all(&mut fs, "self/comm");
        assert_eq!(comm, b"prog\n");
    }

    #[test]
    fn update_self_refreshes_live_fields_but_keeps_nproc() {
        let mut fs = ProcFs::new(4); // 4-cpu system files
        // Default placeholder before any refresh.
        assert_eq!(read_all(&mut fs, "self/comm"), b"nixvm\n");
        fs.update_self(ProcSelf {
            tasks: Vec::new(),
            uid_map: String::new(),
            gid_map: String::new(),
            setgroups: String::new(),
            timens_offsets: String::new(),
            limits: String::new(),
            hostname: String::new(),
            domainname: String::new(),
            cgroup: String::new(),
            mounts: Vec::new(),
            comm: "myprog".to_string(),
            cmdline: b"/bin/myprog\0-x\0".to_vec(),
            exe: "/bin/myprog".to_string(),
            cwd: "/tmp".to_string(),
            pid: 7,
            ppid: 3,
            owner: (0, 0),
            tid: 7,
            tids: vec![7],
            status_ids: String::new(),
            status_security: String::new(),
            tracer: 0,
            fds: vec![(0, "/dev/null".to_string()), (3, "pipe:[9]".to_string())],
            auxv: vec![6, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0],
            arch: None,
            syscall: String::new(),
            seccomp_logged: None,
        });
        assert_eq!(read_all(&mut fs, "self/auxv").len(), 16);
        assert_eq!(read_all(&mut fs, "self/comm"), b"myprog\n");
        assert_eq!(read_all(&mut fs, "self/cmdline"), b"/bin/myprog\0-x\0");
        assert_eq!(fs.readlink("self/exe").unwrap(), "/bin/myprog");
        // pid alias follows the refreshed pid.
        assert_eq!(read_all(&mut fs, "7/comm"), b"myprog\n");
        // The refreshed fd list backs self/fd/.
        assert_eq!(fs.readlink("self/fd/3").unwrap(), "pipe:[9]");
        // Status names the new program and pids.
        let status = String::from_utf8(read_all(&mut fs, "self/status")).unwrap();
        assert!(status.contains("Name:\tmyprog"));
        assert!(status.contains("Pid:\t7"));
        assert!(status.contains("PPid:\t3"));
        // The system-wide cpu count (nproc) is untouched: /proc/stat keeps cpu0..3.
        let stat = String::from_utf8(read_all(&mut fs, "stat")).unwrap();
        assert!(stat.contains("cpu3"));
    }

    #[test]
    fn directories_and_inodes() {
        let mut fs = ProcFs::new(1);
        assert_eq!(fs.stat("").unwrap().kind, NodeKind::Dir);
        assert_eq!(fs.stat("sys/kernel").unwrap().kind, NodeKind::Dir);
        assert_eq!(fs.stat("sys/kernel").unwrap().mode, S_IFDIR | 0o555);
        // Distinct inodes across entries.
        assert_ne!(
            fs.stat("cpuinfo").unwrap().inode,
            fs.stat("meminfo").unwrap().inode
        );
        // A regular file reports its rendered size.
        let meminfo = read_all(&mut fs, "meminfo");
        assert_eq!(fs.stat("meminfo").unwrap().size, meminfo.len() as u64);
        assert_eq!(fs.stat("meminfo").unwrap().mode, S_IFREG | 0o444);
    }

    #[test]
    fn unknown_path_errors() {
        let mut fs = ProcFs::new(1);
        assert!(fs.stat("nope").is_none());
        let mut buf = [0u8; 8];
        assert_eq!(
            fs.read_at("nope", 0, &mut buf).unwrap_err().raw_os_error(),
            Some(2)
        );
        assert_eq!(fs.readdir("nope").unwrap_err().raw_os_error(), Some(2));
    }

    #[test]
    fn read_only_backend() {
        assert!(ProcFs::new(1).read_only());
    }

    #[test]
    fn meminfo_contains_expected_fields() {
        let mut fs = ProcFs::new(1);
        let text = String::from_utf8(read_all(&mut fs, "meminfo")).unwrap();
        assert!(text.contains("MemTotal:"));
        assert!(text.contains("MemFree:"));
    }

    #[test]
    fn self_status_reflects_injected_data() {
        let mut fs = ProcFs::new(1);
        fs.set_self(sample());
        let text = String::from_utf8(read_all(&mut fs, "self/status")).unwrap();
        assert!(text.contains("Name:\tprog"));
        assert!(text.contains("Pid:\t42"));
        assert!(text.contains("PPid:\t7"));
        assert!(text.contains("Uid:\t1000\t1000\t1000\t1000"));
        assert!(text.contains("Gid:\t1000\t1000\t1000\t1000"));
        assert!(text.contains("Threads:\t3"));
        assert!(text.contains("VmSize:\t8192 kB"));
        assert!(text.contains("VmRSS:\t2048 kB"));
        assert!(text.contains("State:\tR (running)"));
    }

    #[test]
    fn self_stat_has_pid_and_comm() {
        let mut fs = ProcFs::new(1);
        fs.set_self(sample());
        let text = String::from_utf8(read_all(&mut fs, "self/stat")).unwrap();
        assert!(text.starts_with("42 (prog) R 7 "));
        assert_eq!(text.split_whitespace().count(), 52);
    }

    #[test]
    fn pid_dir_mirrors_self() {
        let mut fs = ProcFs::new(1);
        fs.set_self(sample());
        assert_eq!(read_all(&mut fs, "42/cmdline"), b"prog\0--flag\0");
        assert_eq!(fs.readlink("42/exe").unwrap(), "/usr/bin/prog");
        assert_eq!(fs.stat("42").unwrap().kind, NodeKind::Dir);
        let names: Vec<String> = fs
            .readdir("42")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert!(names.contains(&"fd".to_string()));
    }

    #[test]
    fn self_cwd_is_a_symlink() {
        let mut fs = ProcFs::new(1);
        fs.set_self(sample());
        assert_eq!(fs.stat("self/cwd").unwrap().kind, NodeKind::Symlink);
        assert_eq!(fs.readlink("self/cwd").unwrap(), "/home/user");
    }

    #[test]
    fn self_fd_lists_injected_descriptors() {
        let mut fs = ProcFs::new(1);
        fs.set_self(sample());
        assert_eq!(fs.stat("self/fd").unwrap().kind, NodeKind::Dir);
        let names: Vec<String> = fs
            .readdir("self/fd")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert_eq!(names.len(), 3);
        assert!(names.contains(&"0".to_string()));
        assert!(names.contains(&"2".to_string()));
        assert_eq!(fs.readlink("self/fd/2").unwrap(), "pipe:[12345]");
        assert_eq!(fs.stat("self/fd/0").unwrap().kind, NodeKind::Symlink);
    }

    #[test]
    fn self_fd_unknown_descriptor_is_enoent() {
        let mut fs = ProcFs::new(1);
        fs.set_self(sample());
        assert_eq!(
            fs.readlink("self/fd/99").unwrap_err().raw_os_error(),
            Some(2)
        );
        assert!(fs.stat("self/fd/99").is_none());
    }

    #[test]
    fn self_maps_falls_back_to_default_when_not_injected() {
        let mut fs = ProcFs::new(1);
        let text = String::from_utf8(read_all(&mut fs, "self/maps")).unwrap();
        assert!(text.contains("[stack]"));
        assert!(text.contains("[heap]"));
        assert!(text.contains("/bin/busybox"));
    }

    #[test]
    fn self_auxv_is_empty_by_default() {
        let mut fs = ProcFs::new(1);
        assert_eq!(read_all(&mut fs, "self/auxv").len(), 0);
    }

    #[test]
    fn cpuinfo_and_stat_track_nproc() {
        let mut fs = ProcFs::new(1);
        fs.set_self(sample()); // nproc = 4
        let cpuinfo = String::from_utf8(read_all(&mut fs, "cpuinfo")).unwrap();
        assert_eq!(cpuinfo.matches("processor\t:").count(), 4);
        let stat = String::from_utf8(read_all(&mut fs, "stat")).unwrap();
        assert!(stat.contains("cpu0 "));
        assert!(stat.contains("cpu3 "));
        assert!(!stat.contains("cpu4 "));
    }

    #[test]
    fn sys_kernel_files_present() {
        let mut fs = ProcFs::new(1);
        let text = String::from_utf8(read_all(&mut fs, "sys/kernel/hostname")).unwrap();
        assert_eq!(text, "nixvm\n");
        let pid_max = String::from_utf8(read_all(&mut fs, "sys/kernel/pid_max")).unwrap();
        assert_eq!(pid_max, "32768\n");
    }

    #[test]
    fn net_tcp_has_expected_header() {
        let mut fs = ProcFs::new(1);
        let text = String::from_utf8(read_all(&mut fs, "net/tcp")).unwrap();
        assert!(text.contains("sl"));
        assert!(text.contains("local_address"));
        assert!(text.contains("rem_address"));
        assert!(text.contains("st"));
        assert!(text.contains("uid"));
        assert!(text.contains("inode"));
    }

    #[test]
    fn net_dev_has_headers_and_lo_row() {
        let mut fs = ProcFs::new(1);
        let text = String::from_utf8(read_all(&mut fs, "net/dev")).unwrap();
        assert!(text.contains("Inter-|"));
        assert!(text.contains("face |bytes"));
        assert!(text.contains("lo:"));
    }

    #[test]
    fn net_unix_has_expected_header() {
        let mut fs = ProcFs::new(1);
        let text = String::from_utf8(read_all(&mut fs, "net/unix")).unwrap();
        assert!(text.starts_with("Num"));
        assert!(text.contains("RefCount"));
        assert!(text.contains("Protocol"));
        assert!(text.contains("Flags"));
        assert!(text.contains("Type"));
        assert!(text.contains("St"));
        assert!(text.contains("Inode"));
        assert!(text.contains("Path"));
    }

    #[test]
    fn net_directory_listing() {
        let mut fs = ProcFs::new(1);
        let names: Vec<String> = fs
            .readdir("net")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        for want in [
            "tcp",
            "tcp6",
            "udp",
            "udp6",
            "unix",
            "dev",
            "route",
            "snmp",
            "protocols",
        ] {
            assert!(names.contains(&want.to_string()), "missing {want}");
        }
        // Root readdir lists the new `net` directory.
        let root_names: Vec<String> = fs
            .readdir("")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert!(root_names.contains(&"net".to_string()));
    }

    #[test]
    fn sys_net_core_directory_listing() {
        let mut fs = ProcFs::new(1);
        let names: Vec<String> = fs
            .readdir("sys/net/core")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert!(names.contains(&"somaxconn".to_string()));
        let somaxconn = String::from_utf8(read_all(&mut fs, "sys/net/core/somaxconn")).unwrap();
        assert_eq!(somaxconn, "4096\n");
    }

    #[test]
    fn sys_net_ipv4_and_vm_and_fs_files() {
        let mut fs = ProcFs::new(1);
        assert_eq!(
            String::from_utf8(read_all(&mut fs, "sys/net/ipv4/tcp_rmem")).unwrap(),
            "4096\t131072\t6291456\n"
        );
        assert_eq!(
            String::from_utf8(read_all(&mut fs, "sys/vm/overcommit_memory")).unwrap(),
            "0\n"
        );
        assert_eq!(
            String::from_utf8(read_all(&mut fs, "sys/fs/file-max")).unwrap(),
            "1048576\n"
        );
    }

    #[test]
    fn self_statm_has_seven_numbers() {
        let mut fs = ProcFs::new(1);
        fs.set_self(sample());
        let text = String::from_utf8(read_all(&mut fs, "self/statm")).unwrap();
        let fields: Vec<&str> = text.split_whitespace().collect();
        assert_eq!(fields.len(), 7);
        for f in &fields {
            assert!(f.parse::<u64>().is_ok(), "not a number: {f}");
        }
    }

    #[test]
    fn self_limits_contains_max_open_files() {
        let mut fs = ProcFs::new(1);
        let text = String::from_utf8(read_all(&mut fs, "self/limits")).unwrap();
        assert!(text.contains("Max open files"));
        assert!(text.contains("Soft Limit"));
        assert!(text.contains("Hard Limit"));
    }

    #[test]
    fn self_io_and_oom_and_wchan() {
        let mut fs = ProcFs::new(1);
        let io = String::from_utf8(read_all(&mut fs, "self/io")).unwrap();
        assert!(io.contains("rchar:"));
        assert!(io.contains("read_bytes:"));
        assert_eq!(read_all(&mut fs, "self/oom_score"), b"0\n");
        assert_eq!(read_all(&mut fs, "self/oom_score_adj"), b"0\n");
        assert_eq!(read_all(&mut fs, "self/wchan"), b"0");
    }

    #[test]
    fn self_mountinfo_and_mounts() {
        let mut fs = ProcFs::new(1);
        let mountinfo = String::from_utf8(read_all(&mut fs, "self/mountinfo")).unwrap();
        assert!(mountinfo.contains(" / / "));
        assert!(mountinfo.contains(" - tmpfs tmpfs rw"));
        let mounts = String::from_utf8(read_all(&mut fs, "self/mounts")).unwrap();
        assert!(mounts.contains("tmpfs / tmpfs rw"));
    }

    #[test]
    fn self_smaps_tracks_maps() {
        let mut fs = ProcFs::new(1);
        let text = String::from_utf8(read_all(&mut fs, "self/smaps")).unwrap();
        assert!(text.contains("[stack]"));
        assert!(text.contains("Rss:"));
        assert!(text.contains("VmFlags:"));
    }

    #[test]
    fn pid_alias_reaches_new_self_files() {
        let mut fs = ProcFs::new(1);
        fs.set_self(sample());
        assert_eq!(read_all(&mut fs, "42/oom_score"), b"0\n");
        let statm = String::from_utf8(read_all(&mut fs, "42/statm")).unwrap();
        assert_eq!(statm.split_whitespace().count(), 7);
    }

    #[test]
    fn top_level_misc_files_present() {
        let mut fs = ProcFs::new(1);
        assert!(
            String::from_utf8(read_all(&mut fs, "partitions"))
                .unwrap()
                .contains("#blocks")
        );
        let disks = super::super::blockdev::SharedDisks::default();
        let dev = disks.lock().unwrap().add(1 << 30, false);
        disks.lock().unwrap().account(dev, false, 4096);
        fs.attach_disks(&disks);
        assert_eq!(
            String::from_utf8(read_all(&mut fs, "partitions")).unwrap(),
            "major minor  #blocks  name\n\n 254       0    1048576 vda\n"
        );
        assert!(
            String::from_utf8(read_all(&mut fs, "diskstats"))
                .unwrap()
                .starts_with(" 254       0 vda 1 0 8 0 0 0 0 0 ")
        );
        assert!(
            String::from_utf8(read_all(&mut fs, "swaps"))
                .unwrap()
                .contains("Filename")
        );
        assert_eq!(read_all(&mut fs, "modules").len(), 0);
        assert!(
            String::from_utf8(read_all(&mut fs, "devices"))
                .unwrap()
                .contains("Character devices:")
        );
    }

    #[test]
    fn self_environ_contains_path() {
        let mut fs = ProcFs::new(1);
        let text = String::from_utf8(read_all(&mut fs, "self/environ")).unwrap();
        assert!(text.contains("PATH="));
        // NUL-separated, as the kernel presents it.
        assert!(text.contains('\0'));
    }

    #[test]
    fn self_root_is_a_symlink_to_slash() {
        let mut fs = ProcFs::new(1);
        assert_eq!(fs.stat("self/root").unwrap().kind, NodeKind::Symlink);
        assert_eq!(fs.readlink("self/root").unwrap(), "/");
    }

    #[test]
    fn cpuinfo_contains_arch_feature_line() {
        let mut fs = ProcFs::new(1);
        let text = String::from_utf8(read_all(&mut fs, "cpuinfo")).unwrap();
        assert!(text.contains("Features") || text.contains("flags"));
    }

    /// The arm64 `Features` line names exactly the `AT_HWCAP`/`AT_HWCAP2`
    /// bits the loader advertises.
    #[test]
    fn aarch64_cpuinfo_features_match_hwcaps() {
        let mut fs = ProcFs::new(1);
        fs.data.arch = Some(crate::abi::Arch::Aarch64);
        let text = String::from_utf8(read_all(&mut fs, "cpuinfo")).unwrap();
        let line = text.lines().find(|l| l.starts_with("Features")).unwrap();
        let names: Vec<&str> = line.split(':').nth(1).unwrap().split_whitespace().collect();
        let (hwcap, hwcap2) = crate::loader::hwcaps(crate::abi::Arch::Aarch64);
        assert_eq!(
            names.len(),
            (hwcap.count_ones() + hwcap2.count_ones()) as usize
        );
        for n in [
            "fp", "asimd", "atomics", "sha3", "sb", "dcpodp", "i8mm", "bf16",
        ] {
            assert!(names.contains(&n), "{n} missing from {line}");
        }
        assert!(!names.contains(&"sve"), "sve is not advertised");
    }

    /// Checks the `8-4-4-4-12` hex shape `proc(5)` documents for
    /// `sys/kernel/random/{boot_id,uuid}`.
    fn is_uuid_shaped(s: &str) -> bool {
        let parts: Vec<&str> = s.split('-').collect();
        let widths = [8, 4, 4, 4, 12];
        parts.len() == widths.len()
            && parts
                .iter()
                .zip(widths)
                .all(|(p, w)| p.len() == w && p.chars().all(|c| c.is_ascii_hexdigit()))
    }

    #[test]
    fn random_uuid_matches_uuid_shape() {
        let mut fs = ProcFs::new(1);
        let uuid = String::from_utf8(read_all(&mut fs, "sys/kernel/random/uuid")).unwrap();
        assert!(is_uuid_shaped(uuid.trim()), "not UUID-shaped: {uuid}");
        let boot_id = String::from_utf8(read_all(&mut fs, "sys/kernel/random/boot_id")).unwrap();
        assert!(is_uuid_shaped(boot_id.trim()), "not UUID-shaped: {boot_id}");
    }

    #[test]
    fn random_entropy_and_poolsize() {
        let mut fs = ProcFs::new(1);
        assert_eq!(
            read_all(&mut fs, "sys/kernel/random/entropy_avail"),
            b"256\n"
        );
        assert_eq!(read_all(&mut fs, "sys/kernel/random/poolsize"), b"4096\n");
    }

    #[test]
    fn self_uid_map_contains_0_0() {
        let mut fs = ProcFs::new(1);
        let text = String::from_utf8(read_all(&mut fs, "self/uid_map")).unwrap();
        assert!(text.contains("0 0"));
        let text = String::from_utf8(read_all(&mut fs, "self/gid_map")).unwrap();
        assert!(text.contains("0 0"));
    }

    #[test]
    fn sys_kernel_extra_tunables_present() {
        let mut fs = ProcFs::new(1);
        assert_eq!(
            String::from_utf8(read_all(&mut fs, "sys/kernel/ngroups_max")).unwrap(),
            "65536\n"
        );
        assert_eq!(
            String::from_utf8(read_all(&mut fs, "sys/kernel/cap_last_cap")).unwrap(),
            "40\n"
        );
        assert!(
            String::from_utf8(read_all(&mut fs, "sys/kernel/threads-max"))
                .unwrap()
                .trim()
                .parse::<u64>()
                .is_ok()
        );
    }

    #[test]
    fn new_top_level_files_present() {
        let mut fs = ProcFs::new(1);
        assert!(read_all(&mut fs, "kallsyms").starts_with(b"0000000000000000"));
        assert_eq!(read_all(&mut fs, "keys").len(), 0);
        assert_eq!(read_all(&mut fs, "key-users").len(), 0);
        assert_eq!(read_all(&mut fs, "locks").len(), 0);
        let vmstat = String::from_utf8(read_all(&mut fs, "vmstat")).unwrap();
        assert!(vmstat.contains("nr_free_pages"));
        let zoneinfo = String::from_utf8(read_all(&mut fs, "zoneinfo")).unwrap();
        assert!(zoneinfo.contains("Node 0"));
        let buddyinfo = String::from_utf8(read_all(&mut fs, "buddyinfo")).unwrap();
        assert!(buddyinfo.contains("Node 0"));
        let consoles = String::from_utf8(read_all(&mut fs, "consoles")).unwrap();
        assert_ne!(consoles.len(), 0);
    }

    #[test]
    fn self_sched_and_schedstat_and_personality() {
        let mut fs = ProcFs::new(1);
        fs.set_self(sample());
        let sched = String::from_utf8(read_all(&mut fs, "self/sched")).unwrap();
        assert!(sched.starts_with("prog (42, #threads: 3)"));
        assert_eq!(read_all(&mut fs, "self/schedstat"), b"0 0 0\n");
        assert_eq!(read_all(&mut fs, "self/personality"), b"0\n");
        assert_eq!(read_all(&mut fs, "self/setgroups"), b"allow\n");
    }

    #[test]
    fn new_listings_include_environ_and_random() {
        let mut fs = ProcFs::new(1);
        let self_names: Vec<String> = fs
            .readdir("self")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert!(self_names.contains(&"environ".to_string()));
        assert!(self_names.contains(&"root".to_string()));
        assert!(self_names.contains(&"sched".to_string()));
        assert!(self_names.contains(&"uid_map".to_string()));

        let kernel_names: Vec<String> = fs
            .readdir("sys/kernel")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert!(kernel_names.contains(&"random".to_string()));

        let random_names: Vec<String> = fs
            .readdir("sys/kernel/random")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert!(random_names.contains(&"uuid".to_string()));
        assert!(random_names.contains(&"boot_id".to_string()));

        let root_names: Vec<String> = fs
            .readdir("")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        assert!(root_names.contains(&"kallsyms".to_string()));
        assert!(root_names.contains(&"vmstat".to_string()));
    }
}
