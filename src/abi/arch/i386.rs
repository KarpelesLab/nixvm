//! The i386 system-call table, for `int $0x80` (and the fast 32-bit entries)
//! from any x86-64 task, and for an i386 (ELFCLASS32) program.
//!
//! Linux takes `int $0x80` through its ia32 entry from any task, 64-bit ones
//! included, and runs the i386 table's call (`syscall_32.tbl`) with the
//! number in `eax` and the arguments in `ebx`, `ecx`, `edx`, `esi`, `edi`,
//! `ebp`. A 64-bit Wine sets up the `%fs` its 32-bit code runs with this way
//! (`set_thread_area`, which only the i386 table has), and an i386 program
//! makes every call this way.
//!
//! Each call is an [`Entry`], data the kernel's compat layer
//! (`kernel::compat`) interprets:
//!
//! - most run as their x86-64 counterpart once their arguments are widened
//!   ([`Entry::Call`]): an `int` (or `long`, `off_t`, `pid_t`, ...) is
//!   sign-extended, everything else zero-extended (`SC_IA32_REGS_TO_ARGS`), a
//!   64-bit value split across two registers is joined ([`Arg::Pair`]: `pread64`,
//!   `ftruncate64`, `fallocate`, ...), and a structure laid out differently
//!   in the i386 ABI is converted on the way in and out through a scratch
//!   copy below the user's stack ([`Ptr`] and its [`Layout`]: `stat64`,
//!   `timespec`, `timeval`, `rlimit`, `rusage`, `statfs64`, `stack_t`, ...);
//! - the rest have a translation of their own ([`Entry::Compat`]): the
//!   `socketcall` and `ipc` multiplexers, `old_mmap`/`mmap2`, the signal
//!   calls and frames, `msghdr` and `iovec` arrays, `select`'s `fd_set`s,
//!   `fcntl64`'s locks, `getdents`, `execve`'s pointer arrays, `clone`'s
//!   argument order and TLS descriptor, ...
//!
//! What Linux has no 64-bit kernel entry for (`vm86`, `break`, `stty`, the
//! removed calls) is [`Entry::Absent`], `ENOSYS`.

/// Where one argument of the x86-64 call comes from, in the i386 registers
/// (numbered from 0: `ebx`, `ecx`, `edx`, `esi`, `edi`, `ebp`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arg {
    /// Register `n`, zero-extended (a pointer, an unsigned value).
    U(u8),
    /// Register `n`, sign-extended (an `int`, a `long`, an `off_t`).
    S(u8),
    /// A 64-bit value whose low half is register `lo` and high half `hi`.
    Pair(u8, u8),
    /// A 16-bit user or group ID in register `n`: `(u16)-1` is `-1`
    /// (`low2highuid`).
    Id16(u8),
    /// A constant.
    K(u64),
}

/// Which way a structure an argument points to goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    /// Read from the caller, converted for the call.
    In,
    /// Converted back to the caller when the call succeeds.
    Out,
    /// Both.
    InOut,
    /// Read, and converted back whatever the call returns (but `EFAULT`):
    /// what is left of a timeout (`nanosleep`'s remainder, `ppoll`'s).
    Rem,
}

/// How a field converts between the i386 and the x86-64 layouts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Sign-extended in, truncated out.
    S,
    /// Zero-extended in, truncated out.
    U,
    /// Zero-extended in; out, `EOVERFLOW` when the value doesn't fit.
    C,
    /// As [`Kind::C`], but all ones (an unknown count) goes out as all ones.
    CAll,
    /// A resource limit: `RLIM_INFINITY` is `COMPAT_RLIM_INFINITY` (all ones)
    /// both ways, and a limit too large saturates to it.
    Rlim,
    /// A limit for the old `getrlimit`: saturating at 0x7fffffff.
    RlimOld,
    /// A 16-bit user or group ID: in, `(u16)-1` is `-1`; out, an ID above
    /// 65535 is the overflow ID (65534, `high2lowuid`).
    Id16,
    /// A file size where an i386 `off_t` is: `EOVERFLOW` above 0x7fffffff
    /// (`MAX_NON_LFS`).
    Lfs,
    /// An old 16-bit device number: `EOVERFLOW` unless major and minor both
    /// fit a byte (`old_valid_dev`).
    OldDev,
    /// The high half of a 64-bit value whose low half is another field (the
    /// SysV IPC `*_high` times); ignored on the way in.
    Hi,
}

/// One field of a structure: its offset and size in the i386 layout and in
/// the x86-64 one, and how it converts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Field {
    pub o32: u16,
    pub s32: u8,
    pub o64: u16,
    pub s64: u8,
    pub kind: Kind,
}

/// A structure laid out differently in the i386 ABI: its sizes in both, and
/// the fields to convert (bytes not covered are zero in a converted copy).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub size32: usize,
    pub size64: usize,
    pub fields: &'static [Field],
}

/// An argument that points to a structure to convert: argument `arg` of the
/// x86-64 call (its i386 value is the user pointer; the call gets a converted
/// copy), `count` of them in a row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ptr {
    pub arg: u8,
    pub dir: Dir,
    pub layout: &'static Layout,
    pub count: u8,
}

/// What the call returns, converted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ret {
    /// As it is.
    Plain,
    /// A 16-bit user or group ID (`high2lowuid`).
    Id16,
}

/// The calls with a translation of their own, in `kernel::compat`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Compat {
    Stime,
    Nice,
    Signal,
    Sgetmask,
    Ssetmask,
    Sigsuspend,
    Sigprocmask,
    Sigreturn,
    RtSigreturn,
    RtSigtimedwait { time64: bool },
    RtSigqueueinfo,
    RtTgsigqueueinfo,
    Getgroups16,
    Setgroups16,
    OldSelect,
    Select,
    Pselect6 { time64: bool },
    Readdir,
    Getdents,
    OldMmap,
    Mmap2,
    Statfs64 { fd: bool },
    Socketcall,
    Ipc,
    Olduname,
    Oldolduname,
    Sysinfo,
    Llseek,
    Fcntl { is64: bool },
    Futex,
    SchedSetaffinity,
    SchedGetaffinity,
    IoSubmit,
    Waitid,
    SetRobustList,
    GetRobustList,
    Rwv { write: bool },
    Prwv { write: bool, flags: bool },
    Vmsplice,
    ProcessVm { write: bool },
    Sendmsg,
    Recvmsg,
    Sendmmsg,
    Recvmmsg { time64: bool },
    Getsockopt,
    Setsockopt,
    Semctl,
    Shmctl,
    Shmat,
    Msgsnd,
    Msgrcv,
    Msgctl,
    ArchPrctl,
    Prctl,
    Seccomp,
}

/// An i386 system call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Entry {
    /// The x86-64 call `nr`, its arguments the i386 ones with the `int`s
    /// (bit `i` of `signed` for argument `i`) sign-extended, or built by
    /// `args`; `ptrs` point to structures to convert, and `ret` is how the
    /// result goes back.
    Call {
        nr: u64,
        signed: u8,
        args: Option<&'static [Arg]>,
        ptrs: &'static [Ptr],
        ret: Ret,
    },
    /// A translation of its own; `nr` is the x86-64 call it amounts to, for
    /// the embedder's policy and tracing (`u64::MAX` for none).
    Compat { nr: u64, how: Compat },
    /// Not a call a 64-bit kernel has: `ENOSYS`.
    Absent,
}

impl Entry {
    /// The x86-64 call this one amounts to, for the embedder's policy,
    /// seccomp and tracing; `None` for an absent one.
    #[must_use]
    pub const fn nr64(&self) -> Option<u64> {
        match *self {
            Self::Call { nr, .. } | Self::Compat { nr, .. } => Some(nr),
            Self::Absent => None,
        }
    }
}

const fn n(nr: u64, signed: u8) -> Entry {
    Entry::Call {
        nr,
        signed,
        args: None,
        ptrs: &[],
        ret: Ret::Plain,
    }
}

const fn a(nr: u64, args: &'static [Arg]) -> Entry {
    Entry::Call {
        nr,
        signed: 0,
        args: Some(args),
        ptrs: &[],
        ret: Ret::Plain,
    }
}

const fn p(nr: u64, signed: u8, ptrs: &'static [Ptr]) -> Entry {
    Entry::Call {
        nr,
        signed,
        args: None,
        ptrs,
        ret: Ret::Plain,
    }
}

const fn ap(nr: u64, args: &'static [Arg], ptrs: &'static [Ptr]) -> Entry {
    Entry::Call {
        nr,
        signed: 0,
        args: Some(args),
        ptrs,
        ret: Ret::Plain,
    }
}

const fn c(nr: u64, how: Compat) -> Entry {
    Entry::Compat { nr, how }
}

const fn ptr(arg: u8, dir: Dir, layout: &'static Layout) -> Ptr {
    Ptr {
        arg,
        dir,
        layout,
        count: 1,
    }
}

const fn ptrn(arg: u8, dir: Dir, layout: &'static Layout, count: u8) -> Ptr {
    Ptr {
        arg,
        dir,
        layout,
        count,
    }
}

const fn f(o32: u16, s32: u8, o64: u16, s64: u8, kind: Kind) -> Field {
    Field {
        o32,
        s32,
        o64,
        s64,
        kind,
    }
}

/// `n` 32-bit words that are 64-bit ones in x86-64 (a structure of `long`s).
macro_rules! longs {
    ($kind:expr; $($i:expr),*) => {
        &[$(f($i * 4, 4, $i * 8, 8, $kind)),*]
    };
}

pub mod layouts {
    //! The structures the i386 ABI lays out differently.
    use super::Kind::{C, CAll, Hi, Id16, Lfs, OldDev, Rlim, RlimOld, S, U};
    use super::{Field, Layout, f};

    /// `struct old_timespec32` / `old_timeval32` / `utimbuf32`: two `long`s.
    pub const TIMESPEC: Layout = Layout {
        size32: 8,
        size64: 16,
        fields: longs!(S; 0, 1),
    };
    /// `old_itimerspec32` / `old_itimerval32`: two of them.
    pub const ITIMERSPEC: Layout = Layout {
        size32: 16,
        size64: 32,
        fields: longs!(S; 0, 1, 2, 3),
    };
    /// A `time_t` (`time`).
    pub const TIME: Layout = Layout {
        size32: 4,
        size64: 8,
        fields: &[f(0, 4, 0, 8, S)],
    };
    /// A `compat_off_t` (`sendfile`'s offset).
    pub const OFF: Layout = TIME;
    /// An `old_sigset_t`: the first 32 signals.
    pub const OLD_SIGSET: Layout = Layout {
        size32: 4,
        size64: 8,
        fields: &[f(0, 4, 0, 8, U)],
    };
    /// A `compat_aio_context_t`.
    pub const AIO_CONTEXT: Layout = Layout {
        size32: 4,
        size64: 8,
        fields: &[f(0, 4, 0, 8, U)],
    };
    /// A pointer and a size, as `pselect6`'s and `io_pgetevents`' signal
    /// mask argument is.
    pub const PTR_SIZE: Layout = Layout {
        size32: 8,
        size64: 16,
        fields: longs!(U; 0, 1),
    };
    /// `struct compat_rlimit`.
    pub const RLIMIT: Layout = Layout {
        size32: 8,
        size64: 16,
        fields: longs!(Rlim; 0, 1),
    };
    /// The old `getrlimit`'s.
    pub const OLD_RLIMIT: Layout = Layout {
        size32: 8,
        size64: 16,
        fields: longs!(RlimOld; 0, 1),
    };
    /// `struct compat_rusage`: two `timeval`s and fourteen `long`s.
    pub const RUSAGE: Layout = Layout {
        size32: 72,
        size64: 144,
        fields: longs!(S; 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17),
    };
    /// `struct compat_tms`: four `clock_t`s.
    pub const TMS: Layout = Layout {
        size32: 16,
        size64: 32,
        fields: longs!(S; 0, 1, 2, 3),
    };
    /// `struct stat64`, i386's (packed, 96 bytes), from x86-64's `struct
    /// stat` (`cp_stat64`).
    pub const STAT64: Layout = Layout {
        size32: 96,
        size64: 144,
        fields: &[
            f(0, 8, 0, 8, U),    // st_dev
            f(12, 4, 8, 8, U),   // __st_ino
            f(16, 4, 24, 4, U),  // st_mode
            f(20, 4, 16, 8, U),  // st_nlink
            f(24, 4, 28, 4, U),  // st_uid
            f(28, 4, 32, 4, U),  // st_gid
            f(32, 8, 40, 8, U),  // st_rdev
            f(44, 8, 48, 8, S),  // st_size
            f(52, 4, 56, 8, U),  // st_blksize
            f(56, 8, 64, 8, U),  // st_blocks
            f(64, 4, 72, 8, U),  // st_atime
            f(68, 4, 80, 8, U),  // st_atime_nsec
            f(72, 4, 88, 8, U),  // st_mtime
            f(76, 4, 96, 8, U),  // st_mtime_nsec
            f(80, 4, 104, 8, U), // st_ctime
            f(84, 4, 112, 8, U), // st_ctime_nsec
            f(88, 8, 8, 8, U),   // st_ino
        ],
    };
    /// `struct compat_stat` (`stat`, `lstat`, `fstat`: 64 bytes), with
    /// `cp_compat_stat`'s overflow checks.
    pub const STAT: Layout = Layout {
        size32: 64,
        size64: 144,
        fields: &[
            f(0, 4, 0, 8, OldDev),   // st_dev
            f(4, 4, 8, 8, C),        // st_ino
            f(8, 2, 24, 4, U),       // st_mode
            f(10, 2, 16, 8, C),      // st_nlink
            f(12, 2, 28, 4, Id16),   // st_uid
            f(14, 2, 32, 4, Id16),   // st_gid
            f(16, 4, 40, 8, OldDev), // st_rdev
            f(20, 4, 48, 8, Lfs),    // st_size
            f(24, 4, 56, 8, U),      // st_blksize
            f(28, 4, 64, 8, U),      // st_blocks
            f(32, 4, 72, 8, U),      // st_atime
            f(36, 4, 80, 8, U),
            f(40, 4, 88, 8, U), // st_mtime
            f(44, 4, 96, 8, U),
            f(48, 4, 104, 8, U), // st_ctime
            f(52, 4, 112, 8, U),
        ],
    };
    /// `struct __old_kernel_stat` (`oldstat`, ...: 32 bytes).
    pub const OLDSTAT: Layout = Layout {
        size32: 32,
        size64: 144,
        fields: &[
            f(0, 2, 0, 8, OldDev),   // st_dev
            f(2, 2, 8, 8, C),        // st_ino
            f(4, 2, 24, 4, U),       // st_mode
            f(6, 2, 16, 8, C),       // st_nlink
            f(8, 2, 28, 4, Id16),    // st_uid
            f(10, 2, 32, 4, Id16),   // st_gid
            f(12, 2, 40, 8, OldDev), // st_rdev
            f(16, 4, 48, 8, Lfs),    // st_size
            f(20, 4, 72, 8, U),      // st_atime
            f(24, 4, 88, 8, U),      // st_mtime
            f(28, 4, 104, 8, U),     // st_ctime
        ],
    };
    /// `struct compat_statfs` (64 bytes), with `put_compat_statfs`'s checks.
    pub const STATFS: Layout = Layout {
        size32: 64,
        size64: 120,
        fields: &[
            f(0, 4, 0, 8, U),      // f_type
            f(4, 4, 8, 8, C),      // f_bsize
            f(8, 4, 16, 8, C),     // f_blocks
            f(12, 4, 24, 8, C),    // f_bfree
            f(16, 4, 32, 8, C),    // f_bavail
            f(20, 4, 40, 8, CAll), // f_files
            f(24, 4, 48, 8, CAll), // f_ffree
            f(28, 8, 56, 8, U),    // f_fsid
            f(36, 4, 64, 8, U),    // f_namelen
            f(40, 4, 72, 8, C),    // f_frsize
            f(44, 4, 80, 8, U),    // f_flags
        ],
    };
    /// `struct compat_statfs64` (packed, 84 bytes).
    pub const STATFS64: Layout = Layout {
        size32: 84,
        size64: 120,
        fields: &[
            f(0, 4, 0, 8, U),   // f_type
            f(4, 4, 8, 8, C),   // f_bsize
            f(8, 8, 16, 8, U),  // f_blocks
            f(16, 8, 24, 8, U), // f_bfree
            f(24, 8, 32, 8, U), // f_bavail
            f(32, 8, 40, 8, U), // f_files
            f(40, 8, 48, 8, U), // f_ffree
            f(48, 8, 56, 8, U), // f_fsid
            f(56, 4, 64, 8, U), // f_namelen
            f(60, 4, 72, 8, C), // f_frsize
            f(64, 4, 80, 8, U), // f_flags
        ],
    };
    /// `compat_stack_t`.
    pub const STACK: Layout = Layout {
        size32: 12,
        size64: 24,
        fields: &[f(0, 4, 0, 8, U), f(4, 4, 8, 4, S), f(8, 4, 16, 8, U)],
    };
    /// `struct compat_sigevent` (64 bytes): the value, the signal, how to
    /// notify, and the thread (or function and attributes).
    pub const SIGEVENT: Layout = Layout {
        size32: 64,
        size64: 64,
        fields: &[
            f(0, 4, 0, 8, U),
            f(4, 4, 8, 4, S),
            f(8, 4, 12, 4, S),
            f(12, 4, 16, 8, U),
            f(16, 4, 24, 8, U),
        ],
    };
    /// `struct compat_mq_attr`.
    pub const MQ_ATTR: Layout = Layout {
        size32: 32,
        size64: 64,
        fields: longs!(S; 0, 1, 2, 3),
    };
    /// `struct old_timex32`.
    pub const TIMEX: Layout = Layout {
        size32: 128,
        size64: 208,
        fields: &[
            f(0, 4, 0, 4, U),    // modes
            f(4, 4, 8, 8, S),    // offset
            f(8, 4, 16, 8, S),   // freq
            f(12, 4, 24, 8, S),  // maxerror
            f(16, 4, 32, 8, S),  // esterror
            f(20, 4, 40, 4, S),  // status
            f(24, 4, 48, 8, S),  // constant
            f(28, 4, 56, 8, S),  // precision
            f(32, 4, 64, 8, S),  // tolerance
            f(36, 4, 72, 8, S),  // time.tv_sec
            f(40, 4, 80, 8, S),  // time.tv_usec
            f(44, 4, 88, 8, S),  // tick
            f(48, 4, 96, 8, S),  // ppsfreq
            f(52, 4, 104, 8, S), // jitter
            f(56, 4, 112, 4, S), // shift
            f(60, 4, 120, 8, S), // stabil
            f(64, 4, 128, 8, S), // jitcnt
            f(68, 4, 136, 8, S), // calcnt
            f(72, 4, 144, 8, S), // errcnt
            f(76, 4, 152, 8, S), // stbcnt
            f(80, 4, 160, 4, S), // tai
        ],
    };
    /// `struct compat_flock` (`F_GETLK`, `F_SETLK`, `F_SETLKW`).
    pub const FLOCK: Layout = Layout {
        size32: 16,
        size64: 32,
        fields: &[
            f(0, 2, 0, 2, S),
            f(2, 2, 2, 2, S),
            f(4, 4, 8, 8, Lfs),
            f(8, 4, 16, 8, S),
            f(12, 4, 24, 4, S),
        ],
    };
    /// `struct compat_flock64` (packed, 24 bytes: `F_GETLK64`, ...).
    pub const FLOCK64: Layout = Layout {
        size32: 24,
        size64: 32,
        fields: &[
            f(0, 2, 0, 2, S),
            f(2, 2, 2, 2, S),
            f(4, 8, 8, 8, S),
            f(12, 8, 16, 8, S),
            f(20, 4, 24, 4, S),
        ],
    };
    /// `struct compat_sigaction` (`rt_sigaction`).
    pub const SIGACTION: Layout = Layout {
        size32: 20,
        size64: 32,
        fields: &[
            f(0, 4, 0, 8, U),
            f(4, 4, 8, 8, U),
            f(8, 4, 16, 8, U),
            f(12, 8, 24, 8, U),
        ],
    };
    /// `struct compat_old_sigaction` (`sigaction`): the handler, the first 32
    /// signals' mask, the flags, the restorer.
    pub const OLD_SIGACTION: Layout = Layout {
        size32: 16,
        size64: 32,
        fields: &[
            f(0, 4, 0, 8, U),
            f(4, 4, 24, 8, U),
            f(8, 4, 8, 8, U),
            f(12, 4, 16, 8, U),
        ],
    };
    /// `struct compat_sock_fprog` (a classic BPF program).
    pub const SOCK_FPROG: Layout = Layout {
        size32: 8,
        size64: 16,
        fields: &[f(0, 2, 0, 2, U), f(4, 4, 8, 8, U)],
    };
    /// A 16-bit user or group ID (`getresuid16`'s).
    pub const ID16: Layout = Layout {
        size32: 2,
        size64: 4,
        fields: &[f(0, 2, 0, 4, Id16)],
    };
    /// `struct compat_iovec`.
    pub const IOVEC: Layout = Layout {
        size32: 8,
        size64: 16,
        fields: longs!(U; 0, 1),
    };
    /// `struct compat_robust_list_head`'s `long futex_offset` between two
    /// pointers.
    pub const ROBUST_LIST_HEAD: Layout = Layout {
        size32: 12,
        size64: 24,
        fields: &[f(0, 4, 0, 8, U), f(4, 4, 8, 8, S), f(8, 4, 16, 8, U)],
    };
    /// `struct compat_ipc64_perm`, the head of every SysV IPC `*64_ds`
    /// (36 bytes; x86-64's is 48, its `mode` 32-bit).
    const PERM: [Field; 7] = [
        f(0, 4, 0, 4, S),   // key
        f(4, 4, 4, 4, U),   // uid
        f(8, 4, 8, 4, U),   // gid
        f(12, 4, 12, 4, U), // cuid
        f(16, 4, 16, 4, U), // cgid
        f(20, 2, 20, 4, U), // mode
        f(24, 2, 24, 2, U), // seq
    ];
    /// `struct compat_shmid64_ds`.
    pub const SHMID64: Layout = Layout {
        size32: 84,
        size64: 112,
        fields: &[
            PERM[0],
            PERM[1],
            PERM[2],
            PERM[3],
            PERM[4],
            PERM[5],
            PERM[6],
            f(36, 4, 48, 8, U),  // shm_segsz
            f(40, 4, 56, 8, U),  // shm_atime
            f(44, 4, 56, 8, Hi), // shm_atime_high
            f(48, 4, 64, 8, U),  // shm_dtime
            f(52, 4, 64, 8, Hi),
            f(56, 4, 72, 8, U), // shm_ctime
            f(60, 4, 72, 8, Hi),
            f(64, 4, 80, 4, S), // shm_cpid
            f(68, 4, 84, 4, S), // shm_lpid
            f(72, 4, 88, 8, U), // shm_nattch
        ],
    };
    /// `struct compat_semid64_ds`.
    pub const SEMID64: Layout = Layout {
        size32: 64,
        size64: 104,
        fields: &[
            PERM[0],
            PERM[1],
            PERM[2],
            PERM[3],
            PERM[4],
            PERM[5],
            PERM[6],
            f(36, 4, 48, 8, U),  // sem_otime
            f(40, 4, 48, 8, Hi), // sem_otime_high
            f(44, 4, 64, 8, U),  // sem_ctime
            f(48, 4, 64, 8, Hi),
            f(52, 4, 80, 8, U), // sem_nsems
        ],
    };
    /// `struct compat_msqid64_ds`.
    pub const MSQID64: Layout = Layout {
        size32: 88,
        size64: 120,
        fields: &[
            PERM[0],
            PERM[1],
            PERM[2],
            PERM[3],
            PERM[4],
            PERM[5],
            PERM[6],
            f(36, 4, 48, 8, U), // msg_stime
            f(40, 4, 48, 8, Hi),
            f(44, 4, 56, 8, U), // msg_rtime
            f(48, 4, 56, 8, Hi),
            f(52, 4, 64, 8, U), // msg_ctime
            f(56, 4, 64, 8, Hi),
            f(60, 4, 72, 8, U),  // msg_cbytes
            f(64, 4, 80, 8, U),  // msg_qnum
            f(68, 4, 88, 8, U),  // msg_qbytes
            f(72, 4, 96, 4, S),  // msg_lspid
            f(76, 4, 100, 4, S), // msg_lrpid
        ],
    };
    /// `struct shminfo64` (`IPC_INFO` of `shmctl`): `long`s.
    pub const SHMINFO64: Layout = Layout {
        size32: 36,
        size64: 72,
        fields: longs!(U; 0, 1, 2, 3, 4, 5, 6, 7, 8),
    };
    /// `struct shm_info` (`SHM_INFO`): two `int`s, then `long`s.
    pub const SHM_INFO: Layout = Layout {
        size32: 24,
        size64: 48,
        fields: &[
            f(0, 4, 0, 4, S),
            f(4, 4, 8, 8, U),
            f(8, 4, 16, 8, U),
            f(12, 4, 24, 8, U),
            f(16, 4, 32, 8, U),
            f(20, 4, 40, 8, U),
        ],
    };
}

use Arg::{Id16 as I16, K, Pair, S, U};
use Compat as X;
use Dir::{In, InOut, Out, Rem};
use layouts as l;

/// Pointer arguments, shared by several calls.
const TS_IN1: &[Ptr] = &[ptr(1, In, &l::TIMESPEC)];
const TS_OUT1: &[Ptr] = &[ptr(1, Out, &l::TIMESPEC)];
const TS_IN_REM: &[Ptr] = &[ptr(0, In, &l::TIMESPEC), ptr(1, Rem, &l::TIMESPEC)];
const CLOCK_NANOSLEEP: &[Ptr] = &[ptr(2, In, &l::TIMESPEC), ptr(3, Rem, &l::TIMESPEC)];
const ITS_SET: &[Ptr] = &[ptr(2, In, &l::ITIMERSPEC), ptr(3, Out, &l::ITIMERSPEC)];
const ITS_GET: &[Ptr] = &[ptr(1, Out, &l::ITIMERSPEC)];
const ITV_SET: &[Ptr] = &[ptr(1, In, &l::ITIMERSPEC), ptr(2, Out, &l::ITIMERSPEC)];
const TV2_1: &[Ptr] = &[ptrn(1, In, &l::TIMESPEC, 2)];
const TV2_2: &[Ptr] = &[ptrn(2, In, &l::TIMESPEC, 2)];
const STAT64_1: &[Ptr] = &[ptr(1, Out, &l::STAT64)];
const STAT64_2: &[Ptr] = &[ptr(2, Out, &l::STAT64)];
const STAT_1: &[Ptr] = &[ptr(1, Out, &l::STAT)];
const OLDSTAT_1: &[Ptr] = &[ptr(1, Out, &l::OLDSTAT)];
const RLIMIT_IN: &[Ptr] = &[ptr(1, In, &l::RLIMIT)];
const RLIMIT_OUT: &[Ptr] = &[ptr(1, Out, &l::RLIMIT)];
const ID16_3: &[Ptr] = &[
    ptr(0, Out, &l::ID16),
    ptr(1, Out, &l::ID16),
    ptr(2, Out, &l::ID16),
];
const MQ_TS: &[Ptr] = &[ptr(4, In, &l::TIMESPEC)];

/// The x86-64 call (or translation) for i386 system call `nr`.
#[must_use]
#[allow(clippy::too_many_lines)]
pub const fn entry(nr: u64) -> Entry {
    match nr {
        // ---- arguments only -------------------------------------------------
        7 => a(61, &[S(0), U(1), S(2), K(0)]), // waitpid: wait4(pid, status, options, NULL)
        19 => n(8, 0b101),                     // lseek (compat_off_t)
        22 => a(166, &[U(0), K(0)]),           // umount: umount2(name, 0)
        92 => n(76, 0b10),                     // truncate (compat_off_t)
        93 => n(77, 0b10),                     // ftruncate
        180 => a(17, &[U(0), U(1), U(2), Pair(3, 4)]), // pread64
        181 => a(18, &[U(0), U(1), U(2), Pair(3, 4)]), // pwrite64
        193 => a(76, &[U(0), Pair(1, 2)]),     // truncate64
        194 => a(77, &[U(0), Pair(1, 2)]),     // ftruncate64
        225 => a(187, &[S(0), Pair(1, 2), U(3)]), // readahead
        239 => n(40, 0b11),                    // sendfile64
        250 => a(221, &[S(0), Pair(1, 2), U(3), S(4)]), // fadvise64
        272 => a(221, &[S(0), Pair(1, 2), Pair(3, 4), S(5)]), // fadvise64_64
        314 => a(277, &[S(0), Pair(1, 2), Pair(3, 4), U(5)]), // sync_file_range
        324 => a(285, &[S(0), S(1), Pair(2, 3), Pair(4, 5)]), // fallocate
        339 => a(301, &[S(0), U(1), Pair(2, 3), S(4), U(5)]), // fanotify_mark
        // 16-bit user and group IDs.
        16 => a(94, &[U(0), I16(1), I16(2)]),     // lchown16
        23 => a(105, &[I16(0)]),                  // setuid16
        46 => a(106, &[I16(0)]),                  // setgid16
        70 => a(113, &[I16(0), I16(1)]),          // setreuid16
        71 => a(114, &[I16(0), I16(1)]),          // setregid16
        95 => a(93, &[U(0), I16(1), I16(2)]),     // fchown16
        138 => a(122, &[I16(0)]),                 // setfsuid16
        139 => a(123, &[I16(0)]),                 // setfsgid16
        164 => a(117, &[I16(0), I16(1), I16(2)]), // setresuid16
        170 => a(119, &[I16(0), I16(1), I16(2)]), // setresgid16
        182 => a(92, &[U(0), I16(1), I16(2)]),    // chown16
        24 | 47 | 49 | 50 => Entry::Call {
            nr: match nr {
                24 => 102,
                47 => 104,
                49 => 107,
                _ => 108,
            },
            signed: 0,
            args: None,
            ptrs: &[],
            ret: Ret::Id16,
        }, // getuid16, getgid16, geteuid16, getegid16
        165 => p(118, 0, ID16_3),                 // getresuid16
        171 => p(120, 0, ID16_3),                 // getresgid16
        // The 32-bit ID calls are the native ones.
        198 => n(94, 0),    // lchown32
        199 => n(102, 0),   // getuid32
        200 => n(104, 0),   // getgid32
        201 => n(107, 0),   // geteuid32
        202 => n(108, 0),   // getegid32
        203 => n(113, 0),   // setreuid32
        204 => n(114, 0),   // setregid32
        205 => n(115, 0b1), // getgroups32
        206 => n(116, 0b1), // setgroups32
        207 => n(93, 0),    // fchown32
        208 => n(117, 0),   // setresuid32
        209 => n(118, 0),   // getresuid32
        210 => n(119, 0),   // setresgid32
        211 => n(120, 0),   // getresgid32
        212 => n(92, 0),    // chown32
        213 => n(105, 0),   // setuid32
        214 => n(106, 0),   // setgid32
        215 => n(122, 0),   // setfsuid32
        216 => n(123, 0),   // setfsgid32
        // The 64-bit time calls (`*_time64`) have x86-64's layouts.
        403 => n(228, 0b1),    // clock_gettime64
        404 => n(227, 0b1),    // clock_settime64
        405 => n(305, 0b1),    // clock_adjtime64
        406 => n(229, 0b1),    // clock_getres_time64
        407 => n(230, 0b11),   // clock_nanosleep_time64
        408 => n(224, 0b1),    // timer_gettime64
        409 => n(223, 0b11),   // timer_settime64
        410 => n(287, 0b1),    // timerfd_gettime64
        411 => n(286, 0b11),   // timerfd_settime64
        412 => n(280, 0b1001), // utimensat_time64
        414 => n(271, 0),      // ppoll_time64
        416 => p(333, 0b110, const { &[ptr(5, In, &l::PTR_SIZE)] }), // io_pgetevents_time64
        418 => n(242, 0b1),    // mq_timedsend_time64
        419 => n(243, 0b1),    // mq_timedreceive_time64
        420 => n(220, 0b1),    // semtimedop_time64
        422 => n(202, 0b100000), // futex_time64
        423 => n(148, 0b1),    // sched_rr_get_interval_time64
        // ---- structures -----------------------------------------------------
        13 => p(201, 0, const { &[ptr(0, Out, &l::TIME)] }), // time
        18 => p(4, 0, OLDSTAT_1),                            // oldstat
        28 => p(5, 0, OLDSTAT_1),                            // oldfstat
        84 => p(6, 0, OLDSTAT_1),                            // oldlstat
        30 => p(132, 0, const { &[ptr(1, In, &l::TIMESPEC)] }), // utime (utimbuf32)
        43 => p(100, 0, const { &[ptr(0, Out, &l::TMS)] }),  // times
        73 => ap(127, &[U(0), K(8)], const { &[ptr(0, Out, &l::OLD_SIGSET)] }), // sigpending
        75 => p(160, 0, RLIMIT_IN),                          // setrlimit
        76 => p(97, 0, const { &[ptr(1, Out, &l::OLD_RLIMIT)] }), // getrlimit (old)
        191 => p(97, 0, RLIMIT_OUT),                         // ugetrlimit
        77 => p(98, 0b1, const { &[ptr(1, Out, &l::RUSAGE)] }), // getrusage
        78 => p(96, 0, const { &[ptr(0, Out, &l::TIMESPEC)] }), // gettimeofday
        79 => p(164, 0, const { &[ptr(0, In, &l::TIMESPEC)] }), // settimeofday
        99 => p(137, 0, const { &[ptr(1, Out, &l::STATFS)] }), // statfs
        100 => p(138, 0, const { &[ptr(1, Out, &l::STATFS)] }), // fstatfs
        104 => p(38, 0b1, ITV_SET),                          // setitimer
        105 => p(36, 0b1, const { &[ptr(1, Out, &l::ITIMERSPEC)] }), // getitimer
        106 => p(4, 0, STAT_1),                              // stat
        107 => p(6, 0, STAT_1),                              // lstat
        108 => p(5, 0, STAT_1),                              // fstat
        114 => p(61, 0b101, const { &[ptr(3, Out, &l::RUSAGE)] }), // wait4
        124 => p(159, 0, const { &[ptr(0, InOut, &l::TIMEX)] }), // adjtimex
        343 => p(305, 0b1, const { &[ptr(1, InOut, &l::TIMEX)] }), // clock_adjtime
        161 => p(148, 0b1, TS_OUT1),                         // sched_rr_get_interval
        162 => p(35, 0, TS_IN_REM),                          // nanosleep
        186 => p(
            131,
            0,
            const { &[ptr(0, In, &l::STACK), ptr(1, Out, &l::STACK)] },
        ), // sigaltstack
        187 => p(40, 0b11, const { &[ptr(2, InOut, &l::OFF)] }), // sendfile
        195 => p(4, 0, STAT64_1),                            // stat64
        196 => p(6, 0, STAT64_1),                            // lstat64
        197 => p(5, 0, STAT64_1),                            // fstat64
        300 => p(262, 0b1001, STAT64_2),                     // fstatat64
        245 => p(206, 0, const { &[ptr(1, InOut, &l::AIO_CONTEXT)] }), // io_setup
        247 => p(208, 0b110, const { &[ptr(4, In, &l::TIMESPEC)] }), // io_getevents
        385 => p(
            333,
            0b110,
            const { &[ptr(4, In, &l::TIMESPEC), ptr(5, In, &l::PTR_SIZE)] },
        ), // io_pgetevents
        259 => p(222, 0b1, const { &[ptr(1, In, &l::SIGEVENT)] }), // timer_create
        260 => p(223, 0b11, ITS_SET),                        // timer_settime
        261 => p(224, 0b1, ITS_GET),                         // timer_gettime
        264 => p(227, 0b1, TS_IN1),                          // clock_settime
        265 => p(228, 0b1, TS_OUT1),                         // clock_gettime
        266 => p(229, 0b1, TS_OUT1),                         // clock_getres
        267 => p(230, 0b11, CLOCK_NANOSLEEP),                // clock_nanosleep
        271 => p(235, 0, TV2_1),                             // utimes
        299 => p(261, 0b1, TV2_2),                           // futimesat
        320 => p(280, 0b1001, TV2_2),                        // utimensat
        277 => p(240, 0b10, const { &[ptr(3, In, &l::MQ_ATTR)] }), // mq_open
        279 => p(242, 0b1, MQ_TS),                           // mq_timedsend
        280 => p(243, 0b1, MQ_TS),                           // mq_timedreceive
        281 => p(244, 0b1, const { &[ptr(1, In, &l::SIGEVENT)] }), // mq_notify
        282 => p(
            245,
            0b1,
            const { &[ptr(1, In, &l::MQ_ATTR), ptr(2, Out, &l::MQ_ATTR)] },
        ), // mq_getsetattr
        309 => p(271, 0, const { &[ptr(2, Rem, &l::TIMESPEC)] }), // ppoll
        325 => p(286, 0b11, ITS_SET),                        // timerfd_settime
        326 => p(287, 0b1, ITS_GET),                         // timerfd_gettime
        // Same layouts in both ABIs (`compat_sigset_t` is x86-64's `sigset_t`).
        175 => n(14, 0b1),    // rt_sigprocmask
        176 => n(127, 0),     // rt_sigpending
        179 => n(130, 0),     // rt_sigsuspend
        321 => n(282, 0b1),   // signalfd
        327 => n(289, 0b101), // signalfd4
        441 => n(441, 0b111), // epoll_pwait2
        371 => n(45, 0b1),    // recvfrom
        5 => n(2, 0b10),      // open
        295 => n(257, 0b101), // openat
        342 => n(304, 0b101), // open_by_handle_at
        54 => n(16, 0),       // ioctl
        62 => n(136, 0),      // ustat
        283 => n(246, 0),     // kexec_load
        288 => n(250, 0b1),   // keyctl
        123 => n(154, 0b1),   // modify_ldt
        243 => n(205, 0),     // set_thread_area
        244 => n(211, 0),     // get_thread_area
        // ---- translations of their own --------------------------------------
        // `execve` reads the pointer arrays at the caller's width; `clone`
        // takes its TLS (a `struct user_desc`) fourth, and with
        // `CLONE_SETTLS` it, and `clone3`'s, is a GDT entry for the child
        // (`set_new_tls` for an ia32 syscall).
        11 => n(59, 0),
        358 => n(322, 0b10001),
        120 => a(56, &[U(0), U(1), U(2), U(4), U(3)]),
        435 => n(435, 0),
        25 => c(164, X::Stime),
        34 => c(141, X::Nice),
        48 => c(13, X::Signal),
        // The handler gets an ia32 signal frame (`SA_IA32_ABI`).
        67 => ap(
            13,
            &[S(0), U(1), U(2), K(8)],
            const {
                &[
                    ptr(1, In, &l::OLD_SIGACTION),
                    ptr(2, Out, &l::OLD_SIGACTION),
                ]
            },
        ),
        174 => p(
            13,
            0b1,
            const { &[ptr(1, In, &l::SIGACTION), ptr(2, Out, &l::SIGACTION)] },
        ),
        68 => c(14, X::Sgetmask),
        69 => c(14, X::Ssetmask),
        72 => c(130, X::Sigsuspend),
        126 => c(14, X::Sigprocmask),
        119 => c(15, X::Sigreturn),
        173 => c(15, X::RtSigreturn),
        177 => c(128, X::RtSigtimedwait { time64: false }),
        421 => c(128, X::RtSigtimedwait { time64: true }),
        178 => c(129, X::RtSigqueueinfo),
        335 => c(297, X::RtTgsigqueueinfo),
        80 => c(115, X::Getgroups16),
        81 => c(116, X::Setgroups16),
        82 => c(23, X::OldSelect),
        142 => c(23, X::Select),
        308 => c(270, X::Pselect6 { time64: false }),
        413 => c(270, X::Pselect6 { time64: true }),
        89 => c(78, X::Readdir),
        141 => c(78, X::Getdents),
        90 => c(9, X::OldMmap),
        192 => c(9, X::Mmap2),
        268 => c(137, X::Statfs64 { fd: false }),
        269 => c(138, X::Statfs64 { fd: true }),
        102 => c(u64::MAX, X::Socketcall),
        117 => c(u64::MAX, X::Ipc),
        109 => c(63, X::Olduname),
        59 => c(63, X::Oldolduname),
        116 => c(99, X::Sysinfo),
        140 => c(8, X::Llseek),
        55 => c(72, X::Fcntl { is64: false }),
        221 => c(72, X::Fcntl { is64: true }),
        240 => c(202, X::Futex),
        241 => c(203, X::SchedSetaffinity),
        242 => c(204, X::SchedGetaffinity),
        248 => c(209, X::IoSubmit),
        284 => c(247, X::Waitid),
        311 => c(273, X::SetRobustList),
        312 => c(274, X::GetRobustList),
        145 => c(19, X::Rwv { write: false }),
        146 => c(20, X::Rwv { write: true }),
        333 => c(
            295,
            X::Prwv {
                write: false,
                flags: false,
            },
        ),
        334 => c(
            296,
            X::Prwv {
                write: true,
                flags: false,
            },
        ),
        378 => c(
            327,
            X::Prwv {
                write: false,
                flags: true,
            },
        ),
        379 => c(
            328,
            X::Prwv {
                write: true,
                flags: true,
            },
        ),
        316 => c(278, X::Vmsplice),
        347 => c(310, X::ProcessVm { write: false }),
        348 => c(311, X::ProcessVm { write: true }),
        370 => c(46, X::Sendmsg),
        372 => c(47, X::Recvmsg),
        345 => c(307, X::Sendmmsg),
        337 => c(299, X::Recvmmsg { time64: false }),
        417 => c(299, X::Recvmmsg { time64: true }),
        365 => c(55, X::Getsockopt),
        366 => c(54, X::Setsockopt),
        394 => c(66, X::Semctl),
        396 => c(31, X::Shmctl),
        397 => c(30, X::Shmat),
        400 => c(69, X::Msgsnd),
        401 => c(70, X::Msgrcv),
        402 => c(71, X::Msgctl),
        384 => c(158, X::ArchPrctl),
        172 => c(157, X::Prctl),
        354 => c(317, X::Seccomp),
        // ---- the rest are the native calls, with their `int`s sign-extended
        // (generated from the kernel's SYSCALL_DEFINE prototypes) -------------
        0 => n(219, 0b0),        // restart_syscall
        1 => n(60, 0b1),         // exit
        2 => n(57, 0b0),         // fork
        3 => n(0, 0b0),          // read
        4 => n(1, 0b0),          // write
        6 => n(3, 0b0),          // close
        8 => n(85, 0b0),         // creat
        9 => n(86, 0b0),         // link
        10 => n(87, 0b0),        // unlink
        12 => n(80, 0b0),        // chdir
        14 => n(133, 0b0),       // mknod
        15 => n(90, 0b0),        // chmod
        20 => n(39, 0b0),        // getpid
        21 => n(165, 0b0),       // mount
        27 => n(37, 0b0),        // alarm
        29 => n(34, 0b0),        // pause
        33 => n(21, 0b10),       // access
        36 => n(162, 0b0),       // sync
        37 => n(62, 0b11),       // kill
        38 => n(82, 0b0),        // rename
        39 => n(83, 0b0),        // mkdir
        40 => n(84, 0b0),        // rmdir
        41 => n(32, 0b0),        // dup
        42 => n(22, 0b0),        // pipe
        45 => n(12, 0b0),        // brk
        51 => n(163, 0b0),       // acct
        52 => n(166, 0b10),      // umount2
        57 => n(109, 0b11),      // setpgid
        60 => n(95, 0b1),        // umask
        61 => n(161, 0b0),       // chroot
        63 => n(33, 0b0),        // dup2
        64 => n(110, 0b0),       // getppid
        65 => n(111, 0b0),       // getpgrp
        66 => n(112, 0b0),       // setsid
        74 => n(170, 0b10),      // sethostname
        83 => n(88, 0b0),        // symlink
        85 => n(89, 0b100),      // readlink
        86 => n(134, 0b0),       // uselib
        87 => n(167, 0b10),      // swapon
        88 => n(169, 0b11),      // reboot
        91 => n(11, 0b0),        // munmap
        94 => n(91, 0b0),        // fchmod
        96 => n(140, 0b11),      // getpriority
        97 => n(141, 0b111),     // setpriority
        101 => n(173, 0b100),    // ioperm
        103 => n(103, 0b101),    // syslog
        110 => n(172, 0b0),      // iopl
        111 => n(153, 0b0),      // vhangup
        115 => n(168, 0b0),      // swapoff
        118 => n(74, 0b0),       // fsync
        121 => n(171, 0b10),     // setdomainname
        122 => n(63, 0b0),       // uname
        125 => n(10, 0b0),       // mprotect
        128 => n(175, 0b0),      // init_module
        129 => n(176, 0b0),      // delete_module
        131 => n(179, 0b0),      // quotactl
        132 => n(121, 0b1),      // getpgid
        133 => n(81, 0b0),       // fchdir
        135 => n(139, 0b1),      // sysfs
        136 => n(135, 0b0),      // personality
        143 => n(73, 0b0),       // flock
        144 => n(26, 0b100),     // msync
        147 => n(124, 0b1),      // getsid
        148 => n(75, 0b0),       // fdatasync
        150 => n(149, 0b0),      // mlock
        151 => n(150, 0b0),      // munlock
        152 => n(151, 0b1),      // mlockall
        153 => n(152, 0b0),      // munlockall
        154 => n(142, 0b1),      // sched_setparam
        155 => n(143, 0b1),      // sched_getparam
        156 => n(144, 0b11),     // sched_setscheduler
        157 => n(145, 0b1),      // sched_getscheduler
        158 => n(24, 0b0),       // sched_yield
        159 => n(146, 0b1),      // sched_get_priority_max
        160 => n(147, 0b1),      // sched_get_priority_min
        163 => n(25, 0b0),       // mremap
        168 => n(7, 0b100),      // poll
        183 => n(79, 0b0),       // getcwd
        184 => n(125, 0b0),      // capget
        185 => n(126, 0b0),      // capset
        190 => n(58, 0b0),       // vfork
        217 => n(155, 0b0),      // pivot_root
        218 => n(27, 0b0),       // mincore
        219 => n(28, 0b100),     // madvise
        220 => n(217, 0b0),      // getdents64
        224 => n(186, 0b0),      // gettid
        226 => n(188, 0b10000),  // setxattr
        227 => n(189, 0b10000),  // lsetxattr
        228 => n(190, 0b10001),  // fsetxattr
        229 => n(191, 0b0),      // getxattr
        230 => n(192, 0b0),      // lgetxattr
        231 => n(193, 0b1),      // fgetxattr
        232 => n(194, 0b0),      // listxattr
        233 => n(195, 0b0),      // llistxattr
        234 => n(196, 0b1),      // flistxattr
        235 => n(197, 0b0),      // removexattr
        236 => n(198, 0b0),      // lremovexattr
        237 => n(199, 0b1),      // fremovexattr
        238 => n(200, 0b11),     // tkill
        246 => n(207, 0b0),      // io_destroy
        249 => n(210, 0b0),      // io_cancel
        252 => n(231, 0b1),      // exit_group
        254 => n(213, 0b1),      // epoll_create
        255 => n(233, 0b111),    // epoll_ctl
        256 => n(232, 0b1101),   // epoll_wait
        257 => n(216, 0b0),      // remap_file_pages
        258 => n(218, 0b0),      // set_tid_address
        262 => n(225, 0b1),      // timer_getoverrun
        263 => n(226, 0b1),      // timer_delete
        270 => n(234, 0b111),    // tgkill
        274 => n(237, 0b0),      // mbind
        275 => n(239, 0b0),      // get_mempolicy
        276 => n(238, 0b1),      // set_mempolicy
        278 => n(241, 0b0),      // mq_unlink
        286 => n(248, 0b0),      // add_key
        287 => n(249, 0b0),      // request_key
        289 => n(251, 0b111),    // ioprio_set
        290 => n(252, 0b11),     // ioprio_get
        291 => n(253, 0b0),      // inotify_init
        292 => n(254, 0b1),      // inotify_add_watch
        293 => n(255, 0b11),     // inotify_rm_watch
        294 => n(256, 0b1),      // migrate_pages
        296 => n(258, 0b1),      // mkdirat
        297 => n(259, 0b1),      // mknodat
        298 => n(260, 0b10001),  // fchownat
        301 => n(263, 0b101),    // unlinkat
        302 => n(264, 0b101),    // renameat
        303 => n(265, 0b10101),  // linkat
        304 => n(266, 0b10),     // symlinkat
        305 => n(267, 0b1001),   // readlinkat
        306 => n(268, 0b1),      // fchmodat
        307 => n(269, 0b101),    // faccessat
        310 => n(272, 0b0),      // unshare
        313 => n(275, 0b101),    // splice
        315 => n(276, 0b11),     // tee
        317 => n(279, 0b100001), // move_pages
        318 => n(309, 0b0),      // getcpu
        319 => n(281, 0b1101),   // epoll_pwait
        322 => n(283, 0b11),     // timerfd_create
        323 => n(284, 0b0),      // eventfd
        328 => n(290, 0b10),     // eventfd2
        329 => n(291, 0b1),      // epoll_create1
        330 => n(292, 0b100),    // dup3
        331 => n(293, 0b10),     // pipe2
        332 => n(294, 0b1),      // inotify_init1
        336 => n(298, 0b1110),   // perf_event_open
        338 => n(300, 0b0),      // fanotify_init
        340 => n(302, 0b1),      // prlimit64
        341 => n(303, 0b10001),  // name_to_handle_at
        344 => n(306, 0b1),      // syncfs
        346 => n(308, 0b11),     // setns
        349 => n(312, 0b111),    // kcmp
        350 => n(313, 0b101),    // finit_module
        351 => n(314, 0b1),      // sched_setattr
        352 => n(315, 0b1),      // sched_getattr
        353 => n(316, 0b101),    // renameat2
        355 => n(318, 0b0),      // getrandom
        356 => n(319, 0b0),      // memfd_create
        357 => n(321, 0b1),      // bpf
        359 => n(41, 0b111),     // socket
        360 => n(53, 0b111),     // socketpair
        361 => n(49, 0b101),     // bind
        362 => n(42, 0b101),     // connect
        363 => n(50, 0b11),      // listen
        364 => n(288, 0b1001),   // accept4
        367 => n(51, 0b1),       // getsockname
        368 => n(52, 0b1),       // getpeername
        369 => n(44, 0b100001),  // sendto
        373 => n(48, 0b11),      // shutdown
        374 => n(323, 0b1),      // userfaultfd
        375 => n(324, 0b101),    // membarrier
        376 => n(325, 0b100),    // mlock2
        377 => n(326, 0b101),    // copy_file_range
        380 => n(329, 0b1000),   // pkey_mprotect
        381 => n(330, 0b0),      // pkey_alloc
        382 => n(331, 0b1),      // pkey_free
        383 => n(332, 0b1),      // statx
        386 => n(334, 0b100),    // rseq
        393 => n(64, 0b111),     // semget
        395 => n(29, 0b101),     // shmget
        398 => n(67, 0b0),       // shmdt
        399 => n(68, 0b11),      // msgget
        424 => n(424, 0b11),     // pidfd_send_signal
        425 => n(425, 0b0),      // io_uring_setup
        426 => n(426, 0b0),      // io_uring_enter
        427 => n(427, 0b0),      // io_uring_register
        428 => n(428, 0b1),      // open_tree
        429 => n(429, 0b101),    // move_mount
        430 => n(430, 0b0),      // fsopen
        431 => n(431, 0b10001),  // fsconfig
        432 => n(432, 0b1),      // fsmount
        433 => n(433, 0b1),      // fspick
        434 => n(434, 0b1),      // pidfd_open
        436 => n(436, 0b0),      // close_range
        437 => n(437, 0b1),      // openat2
        438 => n(438, 0b11),     // pidfd_getfd
        439 => n(439, 0b1101),   // faccessat2
        440 => n(440, 0b1001),   // process_madvise
        442 => n(442, 0b1),      // mount_setattr
        443 => n(443, 0b0),      // quotactl_fd
        444 => n(444, 0b0),      // landlock_create_ruleset
        447 => n(447, 0b0),      // memfd_secret
        448 => n(448, 0b1),      // process_mrelease
        449 => n(449, 0b10000),  // futex_waitv
        450 => n(450, 0b0),      // set_mempolicy_home_node
        451 => n(451, 0b0),      // cachestat
        452 => n(452, 0b1),      // fchmodat2
        453 => n(453, 0b0),      // map_shadow_stack
        454 => n(454, 0b100),    // futex_wake
        455 => n(455, 0b100000), // futex_wait
        456 => n(456, 0b1100),   // futex_requeue
        457 => n(457, 0b0),      // statmount
        458 => n(458, 0b0),      // listmount
        459 => n(459, 0b0),      // lsm_get_self_attr
        460 => n(460, 0b0),      // lsm_set_self_attr
        461 => n(461, 0b0),      // lsm_list_modules
        462 => n(462, 0b0),      // mseal
        463 => n(463, 0b1),      // setxattrat
        464 => n(464, 0b1),      // getxattrat
        465 => n(465, 0b1),      // listxattrat
        466 => n(466, 0b1),      // removexattrat
        467 => n(467, 0b1),      // open_tree_attr
        468 => n(468, 0b1),      // file_getattr
        469 => n(469, 0b1),      // file_setattr
        445 => n(445, 0b1),      // landlock_add_rule
        446 => n(446, 0b1),      // landlock_restrict_self
        // `vm86`, `vm86old`, the calls Linux removed or never had (`break`,
        // `stty`, `ftime`, `lookup_dcookie`, ...), `ptrace` (whose 32-bit
        // register and word layouts aren't translated), and numbers past
        // the table.
        _ => Entry::Absent,
    }
}

/// Widen i386 argument registers `regs` (each the 32-bit value,
/// zero-extended) for the x86-64 call: sign-extend the `int`s (bit `i` of
/// `signed`), or build them as `args` says.
#[must_use]
pub fn widen(signed: u8, args: Option<&[Arg]>, regs: [u64; 6]) -> [u64; 6] {
    let sx = |v: u64| i64::from(v as u32 as i32) as u64;
    let Some(args) = args else {
        let mut out = regs;
        for (i, a) in out.iter_mut().enumerate() {
            if signed & (1 << i) != 0 {
                *a = sx(*a);
            }
        }
        return out;
    };
    let mut out = [0u64; 6];
    for (o, a) in out.iter_mut().zip(args) {
        *o = match *a {
            Arg::U(r) => regs[usize::from(r)],
            Arg::S(r) => sx(regs[usize::from(r)]),
            Arg::Pair(lo, hi) => {
                (regs[usize::from(lo)] & 0xffff_ffff) | regs[usize::from(hi)] << 32
            }
            Arg::Id16(r) => {
                let v = regs[usize::from(r)] & 0xffff;
                if v == 0xffff { u64::from(u32::MAX) } else { v }
            }
            Arg::K(k) => k,
        };
    }
    out
}

/// Why a structure couldn't be converted for the i386 caller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Overflow;

fn get(b: &[u8], off: usize, size: usize) -> u64 {
    let mut w = [0u8; 8];
    w[..size].copy_from_slice(&b[off..off + size]);
    u64::from_le_bytes(w)
}

fn put(b: &mut [u8], off: usize, size: usize, v: u64) {
    b[off..off + size].copy_from_slice(&v.to_le_bytes()[..size]);
}

impl Layout {
    /// The x86-64 structure from the i386 one (`raw`: `size32` bytes).
    #[must_use]
    pub fn widen(&self, raw: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; self.size64];
        for fl in self.fields {
            let (o32, s32, o64, s64) = (
                usize::from(fl.o32),
                usize::from(fl.s32),
                usize::from(fl.o64),
                usize::from(fl.s64),
            );
            let v = get(raw, o32, s32);
            let bits = s32 * 8;
            let all = if bits == 64 {
                u64::MAX
            } else {
                (1u64 << bits) - 1
            };
            let w = match fl.kind {
                Kind::Hi => continue,
                Kind::S | Kind::Lfs => {
                    let shift = 64 - bits;
                    (((v << shift) as i64) >> shift) as u64
                }
                Kind::Rlim | Kind::RlimOld | Kind::Id16 if v == all => u64::MAX,
                _ => v,
            };
            put(&mut out, o64, s64, w);
        }
        out
    }

    /// The i386 structure from the x86-64 one (`raw`: `size64` bytes), or
    /// `Overflow` when a value doesn't fit and the call must fail with
    /// `EOVERFLOW`.
    ///
    /// # Errors
    /// A checked field's value doesn't fit.
    pub fn narrow(&self, raw: &[u8]) -> Result<Vec<u8>, Overflow> {
        let mut out = vec![0u8; self.size32];
        for fl in self.fields {
            let (o32, s32, o64, s64) = (
                usize::from(fl.o32),
                usize::from(fl.s32),
                usize::from(fl.o64),
                usize::from(fl.s64),
            );
            let v = get(raw, o64, s64);
            let bits = s32 * 8;
            let all = if bits == 64 {
                u64::MAX
            } else {
                (1u64 << bits) - 1
            };
            let in_all = if s64 == 8 {
                u64::MAX
            } else {
                (1u64 << (s64 * 8)) - 1
            };
            let w = match fl.kind {
                Kind::CAll if v == in_all => all,
                Kind::Rlim => v.min(all),
                Kind::RlimOld => v.min(0x7fff_ffff),
                Kind::Id16 if v > 0xffff => 65534,
                Kind::Hi => v >> 32,
                _ => v,
            };
            // The most a checked field may hold: `old_valid_dev` is a major
            // and minor below 256 in the new encoding (`minor & 0xff |
            // major << 8 | ...`).
            let max = match fl.kind {
                Kind::C | Kind::CAll => all,
                Kind::Lfs => 0x7fff_ffff,
                Kind::OldDev => 0xffff,
                _ => u64::MAX,
            };
            if w > max {
                return Err(Overflow);
            }
            put(&mut out, o32, s32, w);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i386_numbers_map_to_their_x86_64_calls() {
        assert_eq!(entry(243).nr64(), Some(205));
        assert_eq!(entry(4), n(1, 0));
        assert_eq!(entry(252), n(231, 1));
        // mmap2 takes a page offset: a translation of its own.
        assert_eq!(entry(192), c(9, Compat::Mmap2));
        assert_eq!(entry(166), Entry::Absent); // vm86
        assert_eq!(entry(17), Entry::Absent); // break
        assert_eq!(entry(1000), Entry::Absent);
    }

    #[test]
    fn int_arguments_are_sign_extended() {
        let a = widen(0b11, None, [0xffff_ffff, 9, 0x8000_0000, 0, 0, 0]);
        assert_eq!(a, [u64::MAX, 9, 0x8000_0000, 0, 0, 0]);
    }

    #[test]
    fn split_64_bit_values_are_joined() {
        let Entry::Call { args, .. } = entry(180) else {
            panic!()
        };
        let a = widen(0, args, [3, 0x1000, 10, 0x2345_6789, 0x1, 0]);
        assert_eq!(a, [3, 0x1000, 10, 0x1_2345_6789, 0, 0]);
        let Entry::Call { args, .. } = entry(324) else {
            panic!()
        };
        let a = widen(0, args, [5, 0xffff_ffff, 0, 1, 0x10, 0]);
        assert_eq!(a, [5, u64::MAX, 1 << 32, 0x10, 0, 0]);
    }

    #[test]
    fn sixteen_bit_ids_widen_minus_one() {
        let Entry::Call { args, .. } = entry(164) else {
            panic!()
        };
        let a = widen(0, args, [0xffff, 5, 0x1_ffff, 0, 0, 0]);
        assert_eq!(a, [0xffff_ffff, 5, 0xffff_ffff, 0, 0, 0]);
    }

    #[test]
    fn stat64_moves_and_narrows_fields() {
        let mut st = vec![0u8; 144];
        put(&mut st, 0, 8, 0x803); // dev
        put(&mut st, 8, 8, 0x1_0000_0002); // ino
        put(&mut st, 16, 8, 3); // nlink
        put(&mut st, 24, 4, 0o100644); // mode
        put(&mut st, 48, 8, 0x1_0000_0000); // size
        put(&mut st, 88, 8, 1234); // mtime
        let out = layouts::STAT64.narrow(&st).unwrap();
        assert_eq!(out.len(), 96);
        assert_eq!(get(&out, 0, 8), 0x803);
        assert_eq!(get(&out, 12, 4), 2); // __st_ino truncated
        assert_eq!(get(&out, 88, 8), 0x1_0000_0002); // st_ino whole
        assert_eq!(get(&out, 16, 4), 0o100644);
        assert_eq!(get(&out, 20, 4), 3);
        assert_eq!(get(&out, 44, 8), 0x1_0000_0000);
        assert_eq!(get(&out, 72, 4), 1234);
        // The old stat can't hold that size.
        assert_eq!(layouts::STAT.narrow(&st), Err(Overflow));
    }

    #[test]
    fn rlimits_keep_infinity() {
        let mut r = vec![0u8; 16];
        put(&mut r, 0, 8, u64::MAX);
        put(&mut r, 8, 8, 0x1_0000_0000);
        let out = layouts::RLIMIT.narrow(&r).unwrap();
        assert_eq!(get(&out, 0, 4), 0xffff_ffff);
        assert_eq!(get(&out, 4, 4), 0xffff_ffff);
        let back = layouts::RLIMIT.widen(&out);
        assert_eq!(get(&back, 0, 8), u64::MAX);
        let old = layouts::OLD_RLIMIT.narrow(&r).unwrap();
        assert_eq!(get(&old, 0, 4), 0x7fff_ffff);
    }

    #[test]
    fn timespecs_sign_extend() {
        let raw = [0xff, 0xff, 0xff, 0xff, 5, 0, 0, 0];
        let w = layouts::TIMESPEC.widen(&raw);
        assert_eq!(get(&w, 0, 8), u64::MAX);
        assert_eq!(get(&w, 8, 8), 5);
        assert_eq!(layouts::TIMESPEC.narrow(&w).unwrap(), raw.to_vec());
    }

    #[test]
    fn every_layout_stays_in_bounds() {
        use layouts::*;
        for l in [
            TIMESPEC,
            ITIMERSPEC,
            TIME,
            OLD_SIGSET,
            AIO_CONTEXT,
            PTR_SIZE,
            RLIMIT,
            OLD_RLIMIT,
            RUSAGE,
            TMS,
            STAT64,
            STAT,
            OLDSTAT,
            STATFS,
            STATFS64,
            STACK,
            SIGEVENT,
            MQ_ATTR,
            TIMEX,
            FLOCK,
            FLOCK64,
            ID16,
            IOVEC,
            ROBUST_LIST_HEAD,
            SHMID64,
            SEMID64,
            MSQID64,
            SHMINFO64,
            SHM_INFO,
            SIGACTION,
            OLD_SIGACTION,
            SOCK_FPROG,
        ] {
            for fl in l.fields {
                assert!(
                    usize::from(fl.o32) + usize::from(fl.s32) <= l.size32,
                    "{l:?}"
                );
                assert!(
                    usize::from(fl.o64) + usize::from(fl.s64) <= l.size64,
                    "{l:?}"
                );
            }
            let _ = l.narrow(&vec![0; l.size64]);
            let _ = l.widen(&vec![0; l.size32]);
        }
    }
}
