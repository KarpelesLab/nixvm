//! Linux error numbers.
//!
//! Syscall handlers return `-errno` in the guest result register; [`Errno`]
//! carries the positive value and encodes the negation at the boundary.

/// A Linux errno value (always positive here; negated at the syscall boundary).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Errno(pub i32);

impl Errno {
    /// Encode as the value returned in the syscall result register: the
    /// two's-complement negative errno, as an unsigned register word.
    #[must_use]
    pub const fn as_syscall_ret(self) -> u64 {
        (-(self.0 as i64)) as u64
    }
}

macro_rules! errnos {
    ($($name:ident = $val:expr),+ $(,)?) => {
        impl Errno {
            $(pub const $name: Errno = Errno($val);)+
        }
    };
}

// The generic subset shared across arm64/x86-64 (asm-generic/errno-base.h and
// errno.h). Expanded as handlers need them.
errnos! {
    EPERM = 1,
    ENOENT = 2,
    ESRCH = 3,
    EINTR = 4,
    EIO = 5,
    ENXIO = 6,
    E2BIG = 7,
    ENOEXEC = 8,
    EBADF = 9,
    ECHILD = 10,
    EAGAIN = 11,
    ENOMEM = 12,
    EACCES = 13,
    EFAULT = 14,
    EBUSY = 16,
    EEXIST = 17,
    EXDEV = 18,
    ENODEV = 19,
    ENOTDIR = 20,
    EISDIR = 21,
    EINVAL = 22,
    ENFILE = 23,
    EMFILE = 24,
    ENOTTY = 25,
    EFBIG = 27,
    ENOSPC = 28,
    ESPIPE = 29,
    EROFS = 30,
    EMLINK = 31,
    EPIPE = 32,
    ERANGE = 34,
    ENOSYS = 38,
    ENOTEMPTY = 39,
    ELOOP = 40,
    ENODATA = 61,
    ENOTSOCK = 88,
    EOPNOTSUPP = 95,
    EAFNOSUPPORT = 97,
    ECONNREFUSED = 111,
}
