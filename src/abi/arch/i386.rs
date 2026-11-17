//! The i386 system calls an x86-64 task can make with `int $0x80`.
//!
//! Linux takes `int $0x80` through its ia32 entry from any task, 64-bit ones
//! included, and runs the i386 table's call (`syscall_32.tbl`) with the
//! number in `eax` and the arguments in `ebx`, `ecx`, `edx`, `esi`, `edi`,
//! `ebp`. A 64-bit Wine sets up the `%fs` its 32-bit code runs with this way
//! (`set_thread_area`, which only the i386 table has).
//!
//! The calls here are the ones whose i386 entry point is the native one
//! (`sys_*`, not `compat_sys_*`) and whose arguments are plain integers,
//! descriptors or pointers to data laid out alike in both ABIs, so they run as
//! their x86-64 counterparts. The rest — anything taking a `long`, a
//! `time_t` or a structure that is narrower in the i386 ABI — would need the
//! compat layer, and answer `ENOSYS`.

/// The x86-64 number for i386 system call `nr`, with which of its arguments
/// are C `int`s (bit `i` for argument `i`), sign-extended from their low 32
/// bits; `None` when it isn't one of the calls this entry has.
#[must_use]
pub fn to_x86_64(nr: u64) -> Option<(u64, u8)> {
    Some(match nr {
        1 => (60, 0b1),      // exit
        3 => (0, 0),         // read
        4 => (1, 0),         // write
        6 => (3, 0),         // close
        20 => (39, 0),       // getpid
        27 => (37, 0),       // alarm
        29 => (34, 0),       // pause
        36 => (162, 0),      // sync
        37 => (62, 0b11),    // kill
        41 => (32, 0),       // dup
        42 => (22, 0),       // pipe
        45 => (12, 0),       // brk
        60 => (95, 0),       // umask
        63 => (33, 0),       // dup2
        64 => (110, 0),      // getppid
        65 => (111, 0),      // getpgrp
        66 => (112, 0),      // setsid
        91 => (11, 0),       // munmap
        118 => (74, 0),      // fsync
        123 => (154, 0b1),   // modify_ldt
        125 => (10, 0),      // mprotect
        132 => (121, 0b1),   // getpgid
        136 => (135, 0),     // personality
        147 => (124, 0b1),   // getsid
        148 => (75, 0),      // fdatasync
        150 => (149, 0),     // mlock
        151 => (150, 0),     // munlock
        158 => (24, 0),      // sched_yield
        199 => (102, 0),     // getuid32
        200 => (104, 0),     // getgid32
        201 => (107, 0),     // geteuid32
        202 => (108, 0),     // getegid32
        219 => (28, 0b100),  // madvise
        224 => (186, 0),     // gettid
        238 => (200, 0b11),  // tkill
        243 => (205, 0),     // set_thread_area
        244 => (211, 0),     // get_thread_area
        252 => (231, 0b1),   // exit_group
        270 => (234, 0b111), // tgkill
        318 => (309, 0),     // getcpu
        330 => (292, 0),     // dup3
        331 => (293, 0),     // pipe2
        436 => (436, 0),     // close_range
        _ => return None,
    })
}

/// The x86-64 view of i386 call `nr`'s arguments `args` (each the 32-bit
/// register, zero-extended): the `int` ones sign-extended.
#[must_use]
pub fn widen_args(signed: u8, args: [u64; 6]) -> [u64; 6] {
    let mut out = args;
    for (i, a) in out.iter_mut().enumerate() {
        if signed & (1 << i) != 0 {
            *a = i64::from(*a as u32 as i32) as u64;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i386_numbers_map_to_their_x86_64_calls() {
        assert_eq!(to_x86_64(243), Some((205, 0)));
        assert_eq!(to_x86_64(4), Some((1, 0)));
        assert_eq!(to_x86_64(252), Some((231, 1)));
        // mmap2 (192) takes a page offset: not the x86-64 mmap.
        assert_eq!(to_x86_64(192), None);
    }

    #[test]
    fn int_arguments_are_sign_extended() {
        let a = widen_args(0b11, [0xffff_ffff, 9, 0x8000_0000, 0, 0, 0]);
        assert_eq!(a, [u64::MAX, 9, 0x8000_0000, 0, 0, 0]);
    }
}
