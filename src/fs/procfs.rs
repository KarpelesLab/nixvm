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
    "self/io",
    "self/oom_score",
    "self/oom_score_adj",
    "self/wchan",
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
];

/// Per-process entry names under `self/`, in `readdir("self")` order. `exe`
/// and `cwd` are symlinks, `fd` is a directory, the rest are files.
const SELF_FILES: &[&str] = &[
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
    "io",
    "oom_score",
    "oom_score_adj",
    "wchan",
];

/// Tunables exposed under `sys/kernel/`.
const SYS_KERNEL_FILES: &[&str] = &["ostype", "osrelease", "hostname", "pid_max"];

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
nodev\tdevtmpfs\n";

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

const CMDLINE: &str = "\n";

const OSTYPE: &str = "Linux\n";
const OSRELEASE: &str = "6.1.0-nixvm\n";
const HOSTNAME: &str = "nixvm\n";
const PID_MAX: &str = "32768\n";

// ---- /proc/net/* ----

const NET_TCP_HEADER: &str =
    "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid  timeout inode\n";

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

const NET_ROUTE: &str = "Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT\n";

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

const DISKSTATS: &str = "   8       0 vda 0 0 0 0 0 0 0 0 0 0 0\n";
const PARTITIONS: &str = "major minor  #blocks  name\n\n   8        0     102400 vda\n";
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
  8 sd\n\
  9 md\n\
259 blkext\n";

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

/// Running-process (and lightweight system) data backing the `self/` files
/// plus the CPU-count-sensitive system files.
///
/// The kernel builds one of these from the real process it is executing and
/// installs it with [`ProcFs::set_self`]; until then [`ProcData::default`]
/// supplies placeholders so every file still renders something plausible.
#[derive(Debug, Clone)]
pub struct ProcData {
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
    /// Real/effective/saved/filesystem uid (all reported equal, as a static
    /// sandbox has no reason to differ).
    pub uid: u32,
    /// Real/effective/saved/filesystem gid.
    pub gid: u32,
    /// Single-letter run state (`R`, `S`, `D`, `Z`, `T`) for `stat`/`status`.
    pub state: char,
    /// Thread count.
    pub threads: u32,
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
}

impl Default for ProcData {
    fn default() -> Self {
        Self {
            cmdline: Vec::new(),
            exe: "/bin/busybox".to_string(),
            cwd: "/".to_string(),
            maps: String::new(),
            comm: "nixvm".to_string(),
            argv0: "busybox".to_string(),
            pid: 1,
            ppid: 0,
            uid: 0,
            gid: 0,
            state: 'R',
            threads: 1,
            vm_size_kb: 4096,
            vm_rss_kb: 1024,
            fds: Vec::new(),
            nproc: 1,
            auxv: Vec::new(),
        }
    }
}

/// The synthesized `/proc` backend.
#[derive(Debug)]
pub struct ProcFs {
    data: ProcData,
}

impl Default for ProcFs {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcFs {
    #[must_use]
    pub fn new() -> Self {
        Self {
            data: ProcData::default(),
        }
    }

    /// Install the running-process data backing the `self/` files (and the
    /// `<pid>` alias, and the per-cpu system files).
    pub fn set_self(&mut self, data: ProcData) {
        self.data = data;
    }

    /// Rewrite a numeric-pid path to its `self/...` equivalent when it names
    /// the injected pid, leaving every other path untouched. This is the only
    /// place `<pid>` paths are special-cased; everything downstream only ever
    /// sees `self`-rooted or static paths.
    fn normalize(&self, rel: &str) -> String {
        if self.data.pid != 0 {
            let pid_s = self.data.pid.to_string();
            if rel == pid_s {
                return "self".to_string();
            }
            if let Some(rest) = rel.strip_prefix(pid_s.as_str()).and_then(|r| r.strip_prefix('/'))
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
            "cpuinfo" => return Some(cpuinfo_body(self.data.nproc).into_bytes()),
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
            "self/comm" => format!("{}\n", d.comm),
            "self/stat" => self_stat_body(d),
            "self/statm" => statm_body(d),
            "self/smaps" => {
                let text = if d.maps.is_empty() {
                    default_maps(&d.exe)
                } else {
                    d.maps.clone()
                };
                smaps_body(&text)
            }
            "self/status" => format!(
                "Name:\t{comm}\n\
                 State:\t{state} ({state_desc})\n\
                 Tgid:\t{pid}\n\
                 Pid:\t{pid}\n\
                 PPid:\t{ppid}\n\
                 Uid:\t{uid}\t{uid}\t{uid}\t{uid}\n\
                 Gid:\t{gid}\t{gid}\t{gid}\t{gid}\n\
                 Threads:\t{threads}\n\
                 VmSize:\t{vsize} kB\n\
                 VmRSS:\t{rss} kB\n",
                comm = d.comm,
                state = d.state,
                state_desc = state_desc(d.state),
                pid = d.pid,
                ppid = d.ppid,
                uid = d.uid,
                gid = d.gid,
                threads = d.threads,
                vsize = d.vm_size_kb,
                rss = d.vm_rss_kb,
            ),
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
        "sys/kernel/pid_max" => PID_MAX,
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
        "self/mountinfo" => MOUNTINFO,
        "self/limits" => LIMITS,
        "self/io" => IO,
        "self/oom_score" => OOM_SCORE,
        "self/oom_score_adj" => OOM_SCORE_ADJ,
        "self/wchan" => WCHAN,
        _ => return None,
    };
    Some(text.as_bytes())
}

/// Render `cpuinfo`, one block per core, matching the injected `nproc`
/// (never fewer than one core).
fn cpuinfo_body(nproc: usize) -> String {
    let mut out = String::new();
    for i in 0..nproc.max(1) {
        let _ = write!(
            out,
            "processor\t: {i}\n\
             BogoMIPS\t: 100.00\n\
             Features\t: fp asimd\n\
             CPU implementer\t: 0x41\n\
             CPU architecture: 8\n\
             CPU variant\t: 0x0\n\
             CPU part\t: 0xd08\n\
             CPU revision\t: 0\n\n"
        );
    }
    out
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
            | "sys"
            | "sys/kernel"
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
    fn stat(&mut self, rel: &str) -> Option<Attrs> {
        let rel = self.normalize(rel);
        let rel = rel.as_str();

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
                mtime: 0,
                inode: fd_inode(n),
                nlink: 1,
                rdev: 0,
            });
        }

        let inode = inode_of(rel)?;
        let (kind, mode, size) = if is_dir(rel) {
            (NodeKind::Dir, S_IFDIR | 0o555, 0)
        } else if rel == "self/exe" {
            (NodeKind::Symlink, S_IFLNK | 0o777, self.data.exe.len() as u64)
        } else if rel == "self/cwd" {
            (NodeKind::Symlink, S_IFLNK | 0o777, self.data.cwd.len() as u64)
        } else {
            let data = self.content(rel)?;
            (NodeKind::File, S_IFREG | 0o444, data.len() as u64)
        };
        Some(Attrs {
            kind,
            size,
            mode,
            uid: 0,
            gid: 0,
            mtime: 0,
            inode,
            nlink: 1,
            rdev: 0,
        })
    }

    fn read_at(&mut self, rel: &str, off: u64, buf: &mut [u8]) -> io::Result<usize> {
        let rel = self.normalize(rel);
        let rel = rel.as_str();
        if is_dir(rel) {
            return Err(eisdir());
        }
        if rel == "self/exe" || rel == "self/cwd" || rel.starts_with("self/fd/") {
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
        match rel {
            "" => {
                let mut out: Vec<DirEntry> = ROOT_FILES
                    .iter()
                    .map(|n| entry(n, n, NodeKind::File))
                    .collect();
                out.push(entry("self", "self", NodeKind::Dir));
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
            "self" => Ok(SELF_FILES
                .iter()
                .map(|n| {
                    let path = format!("self/{n}");
                    let kind = match *n {
                        "exe" | "cwd" => NodeKind::Symlink,
                        "fd" => NodeKind::Dir,
                        _ => NodeKind::File,
                    };
                    entry(n, &path, kind)
                })
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
            "sys/kernel" => Ok(SYS_KERNEL_FILES
                .iter()
                .map(|n| {
                    let path = format!("sys/kernel/{n}");
                    entry(n, &path, NodeKind::File)
                })
                .collect()),
            "net" => Ok(NET_FILES
                .iter()
                .map(|n| {
                    let path = format!("net/{n}");
                    entry(n, &path, NodeKind::File)
                })
                .collect()),
            "sys/net" => Ok(vec![
                entry("core", "sys/net/core", NodeKind::Dir),
                entry("ipv4", "sys/net/ipv4", NodeKind::Dir),
            ]),
            "sys/net/core" => Ok(SYS_NET_CORE_FILES
                .iter()
                .map(|n| {
                    let path = format!("sys/net/core/{n}");
                    entry(n, &path, NodeKind::File)
                })
                .collect()),
            "sys/net/ipv4" => Ok(SYS_NET_IPV4_FILES
                .iter()
                .map(|n| {
                    let path = format!("sys/net/ipv4/{n}");
                    entry(n, &path, NodeKind::File)
                })
                .collect()),
            "sys/vm" => Ok(SYS_VM_FILES
                .iter()
                .map(|n| {
                    let path = format!("sys/vm/{n}");
                    entry(n, &path, NodeKind::File)
                })
                .collect()),
            "sys/fs" => Ok(SYS_FS_FILES
                .iter()
                .map(|n| {
                    let path = format!("sys/fs/{n}");
                    entry(n, &path, NodeKind::File)
                })
                .collect()),
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
            cmdline: b"prog\0--flag\0".to_vec(),
            exe: "/usr/bin/prog".to_string(),
            cwd: "/home/user".to_string(),
            maps: String::new(),
            comm: "prog".to_string(),
            argv0: "prog".to_string(),
            pid: 42,
            ppid: 7,
            uid: 1000,
            gid: 1000,
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
        }
    }

    #[test]
    fn version_contains_linux() {
        let mut fs = ProcFs::new();
        let data = read_all(&mut fs, "version");
        assert!(String::from_utf8_lossy(&data).contains("Linux"));
    }

    #[test]
    fn root_readdir_lists_files_and_dirs() {
        let mut fs = ProcFs::new();
        let names: Vec<String> = fs.readdir("").unwrap().into_iter().map(|e| e.name).collect();
        assert!(names.contains(&"cpuinfo".to_string()));
        assert!(names.contains(&"self".to_string()));
        assert!(names.contains(&"sys".to_string()));
        // Default ProcData carries pid 1, so its numeric alias is listed too.
        assert!(names.contains(&"1".to_string()));
    }

    #[test]
    fn self_exe_is_a_symlink() {
        let mut fs = ProcFs::new();
        let attrs = fs.stat("self/exe").unwrap();
        assert_eq!(attrs.kind, NodeKind::Symlink);
        assert_eq!(attrs.mode, S_IFLNK | 0o777);
        // readlink resolves to the exe path.
        assert_eq!(fs.readlink("self/exe").unwrap(), "/bin/busybox");
    }

    #[test]
    fn set_self_changes_cmdline() {
        let mut fs = ProcFs::new();
        // Placeholder cmdline is empty.
        assert!(read_all(&mut fs, "self/cmdline").is_empty());
        fs.set_self(sample());
        assert_eq!(read_all(&mut fs, "self/cmdline"), b"prog\0--flag\0");
        assert_eq!(fs.readlink("self/exe").unwrap(), "/usr/bin/prog");
        let comm = read_all(&mut fs, "self/comm");
        assert_eq!(comm, b"prog\n");
    }

    #[test]
    fn directories_and_inodes() {
        let mut fs = ProcFs::new();
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
        let mut fs = ProcFs::new();
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
        assert!(ProcFs::new().read_only());
    }

    #[test]
    fn meminfo_contains_expected_fields() {
        let mut fs = ProcFs::new();
        let text = String::from_utf8(read_all(&mut fs, "meminfo")).unwrap();
        assert!(text.contains("MemTotal:"));
        assert!(text.contains("MemFree:"));
    }

    #[test]
    fn self_status_reflects_injected_data() {
        let mut fs = ProcFs::new();
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
        let mut fs = ProcFs::new();
        fs.set_self(sample());
        let text = String::from_utf8(read_all(&mut fs, "self/stat")).unwrap();
        assert!(text.starts_with("42 (prog) R 7 "));
        assert_eq!(text.split_whitespace().count(), 52);
    }

    #[test]
    fn pid_dir_mirrors_self() {
        let mut fs = ProcFs::new();
        fs.set_self(sample());
        assert_eq!(read_all(&mut fs, "42/cmdline"), b"prog\0--flag\0");
        assert_eq!(fs.readlink("42/exe").unwrap(), "/usr/bin/prog");
        assert_eq!(fs.stat("42").unwrap().kind, NodeKind::Dir);
        let names: Vec<String> = fs.readdir("42").unwrap().into_iter().map(|e| e.name).collect();
        assert!(names.contains(&"fd".to_string()));
    }

    #[test]
    fn self_cwd_is_a_symlink() {
        let mut fs = ProcFs::new();
        fs.set_self(sample());
        assert_eq!(fs.stat("self/cwd").unwrap().kind, NodeKind::Symlink);
        assert_eq!(fs.readlink("self/cwd").unwrap(), "/home/user");
    }

    #[test]
    fn self_fd_lists_injected_descriptors() {
        let mut fs = ProcFs::new();
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
        let mut fs = ProcFs::new();
        fs.set_self(sample());
        assert_eq!(
            fs.readlink("self/fd/99").unwrap_err().raw_os_error(),
            Some(2)
        );
        assert!(fs.stat("self/fd/99").is_none());
    }

    #[test]
    fn self_maps_falls_back_to_default_when_not_injected() {
        let mut fs = ProcFs::new();
        let text = String::from_utf8(read_all(&mut fs, "self/maps")).unwrap();
        assert!(text.contains("[stack]"));
        assert!(text.contains("[heap]"));
        assert!(text.contains("/bin/busybox"));
    }

    #[test]
    fn self_auxv_is_empty_by_default() {
        let mut fs = ProcFs::new();
        assert!(read_all(&mut fs, "self/auxv").is_empty());
    }

    #[test]
    fn cpuinfo_and_stat_track_nproc() {
        let mut fs = ProcFs::new();
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
        let mut fs = ProcFs::new();
        let text = String::from_utf8(read_all(&mut fs, "sys/kernel/hostname")).unwrap();
        assert_eq!(text, "nixvm\n");
        let pid_max = String::from_utf8(read_all(&mut fs, "sys/kernel/pid_max")).unwrap();
        assert_eq!(pid_max, "32768\n");
    }

    #[test]
    fn net_tcp_has_expected_header() {
        let mut fs = ProcFs::new();
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
        let mut fs = ProcFs::new();
        let text = String::from_utf8(read_all(&mut fs, "net/dev")).unwrap();
        assert!(text.contains("Inter-|"));
        assert!(text.contains("face |bytes"));
        assert!(text.contains("lo:"));
    }

    #[test]
    fn net_unix_has_expected_header() {
        let mut fs = ProcFs::new();
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
        let mut fs = ProcFs::new();
        let names: Vec<String> = fs
            .readdir("net")
            .unwrap()
            .into_iter()
            .map(|e| e.name)
            .collect();
        for want in ["tcp", "tcp6", "udp", "udp6", "unix", "dev", "route", "snmp", "protocols"] {
            assert!(names.contains(&want.to_string()), "missing {want}");
        }
        // Root readdir lists the new `net` directory.
        let root_names: Vec<String> = fs.readdir("").unwrap().into_iter().map(|e| e.name).collect();
        assert!(root_names.contains(&"net".to_string()));
    }

    #[test]
    fn sys_net_core_directory_listing() {
        let mut fs = ProcFs::new();
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
        let mut fs = ProcFs::new();
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
        let mut fs = ProcFs::new();
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
        let mut fs = ProcFs::new();
        let text = String::from_utf8(read_all(&mut fs, "self/limits")).unwrap();
        assert!(text.contains("Max open files"));
        assert!(text.contains("Soft Limit"));
        assert!(text.contains("Hard Limit"));
    }

    #[test]
    fn self_io_and_oom_and_wchan() {
        let mut fs = ProcFs::new();
        let io = String::from_utf8(read_all(&mut fs, "self/io")).unwrap();
        assert!(io.contains("rchar:"));
        assert!(io.contains("read_bytes:"));
        assert_eq!(read_all(&mut fs, "self/oom_score"), b"0\n");
        assert_eq!(read_all(&mut fs, "self/oom_score_adj"), b"0\n");
        assert_eq!(read_all(&mut fs, "self/wchan"), b"0");
    }

    #[test]
    fn self_mountinfo_and_mounts() {
        let mut fs = ProcFs::new();
        let mountinfo = String::from_utf8(read_all(&mut fs, "self/mountinfo")).unwrap();
        assert!(mountinfo.contains(" / / "));
        assert!(mountinfo.contains(" - tmpfs tmpfs rw"));
        let mounts = String::from_utf8(read_all(&mut fs, "self/mounts")).unwrap();
        assert!(mounts.contains("tmpfs / tmpfs rw"));
    }

    #[test]
    fn self_smaps_tracks_maps() {
        let mut fs = ProcFs::new();
        let text = String::from_utf8(read_all(&mut fs, "self/smaps")).unwrap();
        assert!(text.contains("[stack]"));
        assert!(text.contains("Rss:"));
        assert!(text.contains("VmFlags:"));
    }

    #[test]
    fn pid_alias_reaches_new_self_files() {
        let mut fs = ProcFs::new();
        fs.set_self(sample());
        assert_eq!(read_all(&mut fs, "42/oom_score"), b"0\n");
        let statm = String::from_utf8(read_all(&mut fs, "42/statm")).unwrap();
        assert_eq!(statm.split_whitespace().count(), 7);
    }

    #[test]
    fn top_level_misc_files_present() {
        let mut fs = ProcFs::new();
        assert!(String::from_utf8(read_all(&mut fs, "diskstats")).unwrap().contains("vda"));
        assert!(String::from_utf8(read_all(&mut fs, "partitions")).unwrap().contains("#blocks"));
        assert!(
            String::from_utf8(read_all(&mut fs, "swaps"))
                .unwrap()
                .contains("Filename")
        );
        assert!(read_all(&mut fs, "modules").is_empty());
        assert!(
            String::from_utf8(read_all(&mut fs, "devices"))
                .unwrap()
                .contains("Character devices:")
        );
    }
}
