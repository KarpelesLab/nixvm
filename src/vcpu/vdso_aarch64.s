// Source for the aarch64 vDSO code embedded in `vdso.rs` as `CODE_AARCH64`.
//
// Regenerate the byte array + symbol/patch offsets after editing:
//     clang -target aarch64-linux-gnu -c vdso_aarch64.s -o vdso_aarch64.o
//     llvm-objcopy -O binary -j .text vdso_aarch64.o vdso_aarch64.bin  # -> bytes
//     llvm-nm vdso_aarch64.o                                          # -> offsets
// and find the 0x1122334455667788 sentinel for VVAR_PATCH_AARCH64.
//
// Same contract as the x86-64 one: the virtual counter (`cntvct_el0`, which
// counts host `mach_absolute_time` ticks under HVF) scaled by the vvar
// calibration serves the wall and monotonic clocks; everything else, and an
// uncalibrated vvar (the interpreter, whose counter isn't real time), takes the
// real syscall.
.text

// ---- __kernel_clock_gettime(clockid_t id in x0, struct timespec *ts in x1) ----
.globl __kernel_clock_gettime
__kernel_clock_gettime:
    cmp     x0, #6
    b.hi    1f
    cmp     x0, #2
    b.eq    1f
    cmp     x0, #3
    b.eq    1f
    ldr     x9, vvar
    ldr     x10, [x9, #8]          // mult; 0 = uncalibrated
    cbz     x10, 1f
    isb
    mrs     x11, cntvct_el0
    ldr     x12, [x9, #24]         // base count
    sub     x11, x11, x12
    mul     x13, x11, x10          // delta * mult, 128 bits
    umulh   x14, x11, x10
    ldr     x15, [x9, #16]         // shift (1..63)
    lsr     x13, x13, x15
    neg     x15, x15
    lsl     x14, x14, x15          // hi << (64 - shift)
    orr     x13, x13, x14          // ns since the base
    cmp     x0, #0                 // REALTIME
    ccmp    x0, #5, #4, ne         // or REALTIME_COARSE
    ldr     x12, [x9, #32]         // base monotonic ns
    ldr     x14, [x9, #40]         // base wall ns
    csel    x12, x14, x12, eq
    add     x13, x13, x12
    movz    x15, #0xca00
    movk    x15, #0x3b9a, lsl #16  // 1e9
    udiv    x16, x13, x15
    msub    x17, x16, x15, x13
    stp     x16, x17, [x1]
    mov     x0, #0
    ret
1:  mov     x8, #113               // __NR_clock_gettime
    svc     #0
    ret

// ---- __kernel_gettimeofday(struct timeval *tv in x0, struct timezone *tz in x1) ----
.globl __kernel_gettimeofday
__kernel_gettimeofday:
    cbz     x1, 2f
    str     xzr, [x1]              // tz_minuteswest, tz_dsttime = 0
2:  cbz     x0, 4f
    ldr     x9, vvar
    ldr     x10, [x9, #8]
    cbz     x10, 3f
    isb
    mrs     x11, cntvct_el0
    ldr     x12, [x9, #24]
    sub     x11, x11, x12
    mul     x13, x11, x10
    umulh   x14, x11, x10
    ldr     x15, [x9, #16]
    lsr     x13, x13, x15
    neg     x15, x15
    lsl     x14, x14, x15
    orr     x13, x13, x14
    ldr     x12, [x9, #40]         // base wall ns
    add     x13, x13, x12
    movz    x15, #0xca00
    movk    x15, #0x3b9a, lsl #16
    udiv    x16, x13, x15          // seconds
    msub    x17, x16, x15, x13     // ns
    mov     x15, #1000
    udiv    x17, x17, x15          // us
    stp     x16, x17, [x0]
4:  mov     x0, #0
    ret
3:  mov     x8, #169               // __NR_gettimeofday
    svc     #0
    ret

// ---- __kernel_clock_getres(clockid_t id in x0, struct timespec *res in x1) ----
.globl __kernel_clock_getres
__kernel_clock_getres:
    mov     x8, #114               // __NR_clock_getres
    svc     #0
    ret

// ---- __kernel_rt_sigreturn ----
.globl __kernel_rt_sigreturn
__kernel_rt_sigreturn:
    mov     x8, #139               // __NR_rt_sigreturn
    svc     #0

    .p2align 3
vvar:
    .quad   0x1122334455667788     // patched to the vvar page's VA
