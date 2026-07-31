# Source for the x86-64 vDSO code embedded in `vdso.rs` as `CODE`.
#
# Regenerate the byte array + symbol/patch offsets after editing:
#     as vdso_x86.s -o vdso_x86.o
#     objcopy -O binary -j .text vdso_x86.o vdso_x86.bin   # -> CODE bytes
#     nm vdso_x86.o                                        # -> SYMS offsets
# and find the 0x1122334455667788 sentinel occurrences for VVAR_PATCH.
#
# Each function fast-paths only what the vvar TSC calibration can serve and falls
# back to the raw syscall otherwise — matching the real Linux vDSO, and never
# returning a wrong answer for a clock it does not truly compute.
.intel_syntax noprefix
.text

# The vvar VA is a movabs immediate patched by build_image; use a recognizable
# sentinel so every occurrence can be found and rewritten.
.set VVAR_SENTINEL, 0x1122334455667788

# ---- __vdso_clock_gettime(clockid_t id in rdi, struct timespec *ts in rsi) ----
# Fast-path only the wall/monotonic clocks the vvar calibration covers:
#   0 REALTIME, 1 MONOTONIC, 4 MONOTONIC_RAW, 5 REALTIME_COARSE, 6 MONOTONIC_COARSE.
# CPU-time clocks (2,3), BOOTTIME/TAI/etc (>6), and an uncalibrated vvar fall
# through to the real syscall — exactly what the kernel vDSO does.
.globl __vdso_clock_gettime
__vdso_clock_gettime:
    cmp    rdi, 6
    ja     .Lcg_sys
    cmp    rdi, 2
    je     .Lcg_sys
    cmp    rdi, 3
    je     .Lcg_sys
    movabs r8, VVAR_SENTINEL
    cmp    qword ptr [r8+8], 0        # mult == 0 -> uncalibrated
    je     .Lcg_sys
    rdtsc
    shl    rdx, 32
    or     rax, rdx
    sub    rax, [r8+24]               # tsc - base_tsc
    mul    qword ptr [r8+8]           # * mult -> rdx:rax
    mov    cl, [r8+16]                # shift
    shrd   rax, rdx, cl               # >> shift  (ns since base_tsc)
    cmp    rdi, 0                      # REALTIME
    je     .Lcg_wall
    cmp    rdi, 5                      # REALTIME_COARSE
    je     .Lcg_wall
    add    rax, [r8+32]               # + base_mono_ns
    jmp    .Lcg_split
.Lcg_wall:
    add    rax, [r8+40]               # + base_wall_ns
.Lcg_split:
    xor    edx, edx
    mov    rcx, 1000000000
    div    rcx                        # rax = sec, rdx = nsec
    mov    [rsi], rax
    mov    [rsi+8], rdx
    xor    eax, eax
    ret
.Lcg_sys:
    mov    eax, 228                   # __NR_clock_gettime
    syscall
    ret

# ---- __vdso_gettimeofday(struct timeval *tv in rdi, struct timezone *tz in rsi) ----
.globl __vdso_gettimeofday
__vdso_gettimeofday:
    test   rsi, rsi                   # tz (obsolete): zero it if present
    je     .Lgtod_tv
    mov    dword ptr [rsi], 0         # tz_minuteswest
    mov    dword ptr [rsi+4], 0       # tz_dsttime
.Lgtod_tv:
    test   rdi, rdi
    je     .Lgtod_ok                  # tv == NULL: nothing to fill
    movabs r8, VVAR_SENTINEL
    cmp    qword ptr [r8+8], 0
    je     .Lgtod_sys
    rdtsc
    shl    rdx, 32
    or     rax, rdx
    sub    rax, [r8+24]
    mul    qword ptr [r8+8]
    mov    cl, [r8+16]
    shrd   rax, rdx, cl
    add    rax, [r8+40]               # + base_wall_ns
    xor    edx, edx
    mov    rcx, 1000000000
    div    rcx                        # rax = sec, rdx = nsec
    mov    [rdi], rax                 # tv_sec
    mov    rax, rdx
    xor    edx, edx
    mov    rcx, 1000
    div    rcx                        # rax = usec
    mov    [rdi+8], rax               # tv_usec
.Lgtod_ok:
    xor    eax, eax
    ret
.Lgtod_sys:
    mov    eax, 96                    # __NR_gettimeofday
    syscall
    ret

# ---- __vdso_clock_getres(clockid_t id in rdi, struct timespec *res in rsi) ----
# 1 ns resolution for the supported clocks (matching the kernel's answer); other
# clocks fall through to the syscall.
.globl __vdso_clock_getres
__vdso_clock_getres:
    cmp    rdi, 6
    ja     .Lcgr_sys
    cmp    rdi, 2
    je     .Lcgr_sys
    cmp    rdi, 3
    je     .Lcgr_sys
    test   rsi, rsi                   # res may be NULL
    je     .Lcgr_ok
    mov    qword ptr [rsi], 0         # tv_sec = 0
    mov    qword ptr [rsi+8], 1       # tv_nsec = 1
.Lcgr_ok:
    xor    eax, eax
    ret
.Lcgr_sys:
    mov    eax, 229                   # __NR_clock_getres
    syscall
    ret

# ---- __vdso_time(time_t *t in rdi) -> seconds ----
.globl __vdso_time
__vdso_time:
    movabs r8, VVAR_SENTINEL
    cmp    qword ptr [r8+8], 0
    je     .Ltime_sys
    rdtsc
    shl    rdx, 32
    or     rax, rdx
    sub    rax, [r8+24]
    mul    qword ptr [r8+8]
    mov    cl, [r8+16]
    shrd   rax, rdx, cl
    add    rax, [r8+40]               # + base_wall_ns
    xor    edx, edx
    mov    rcx, 1000000000
    div    rcx                        # rax = seconds
    test   rdi, rdi
    je     .Ltime_ret
    mov    [rdi], rax
.Ltime_ret:
    ret
.Ltime_sys:
    mov    eax, 201                   # __NR_time
    syscall
    ret

# ---- __vdso_getcpu(unsigned *cpu in rdi, unsigned *node in rsi, void *unused) ----
# Single-CPU guest: CPU 0, NUMA node 0.
.globl __vdso_getcpu
__vdso_getcpu:
    test   rdi, rdi
    je     .Lcpu_node
    mov    dword ptr [rdi], 0
.Lcpu_node:
    test   rsi, rsi
    je     .Lcpu_ok
    mov    dword ptr [rsi], 0
.Lcpu_ok:
    xor    eax, eax
    ret
