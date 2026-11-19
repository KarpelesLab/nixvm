# Source for the 32-bit vDSO's clock functions, embedded in `vdso32.rs` as
# `CLOCK_CODE` at `CLOCK` (TEXT + 0x40) in the image.
#
# Regenerate the bytes and symbol offsets after editing:
#     as --32 vdso32_x86.s -o vdso32_x86.o
#     objcopy -O binary -j .text vdso32_x86.o vdso32_x86.bin   # -> CLOCK_CODE
#     nm vdso32_x86.o                                          # -> CLOCK_SYMS
#
# The functions read the same vvar page the 64-bit vDSO does (`vdso.rs`'s
# `vvar` offsets), mapped one page below the image, and fall back to the
# system call through `__kernel_vsyscall` (the image's fast entry) for what the
# calibration can't serve, as Linux's vdso32 does: an uncalibrated vvar, the
# CPU-time clocks, and anything above CLOCK_MONOTONIC_COARSE. cdecl: arguments
# on the stack, %ebx, %esi, %edi and %ebp preserved, the result in %eax.
.code32
.text

.set MULT, 8
.set SHIFT, 16
.set BASE_TSC, 24
.set MONO, 32
.set WALL, 40

.set NR_time, 13
.set NR_gettimeofday, 78
.set NR_clock_gettime, 265
.set NR_clock_getres, 266
.set NR_getcpu, 318
.set NR_clock_gettime64, 403

start:
# The image is laid out around this code: `__kernel_vsyscall` at TEXT, 0x40
# below, and the vvar page one page below the image (TEXT is 0x400).
.set VSYSCALL, start - 0x40
.set VVAR, start - 0x440 - 0x1000

# ---- ns_now: %ecx = the base to add (MONO or WALL) --------------------------
# Out: %edx:%eax = nanoseconds and CF clear, or CF set when the vvar holds no
# calibration. Clobbers %ecx.
ns_now:
    push %ebp
    push %ebx
    push %esi
    push %edi
    call 1f
1:  pop %ebp
    add $(VVAR - 1b), %ebp
    mov MULT(%ebp), %eax
    or MULT+4(%ebp), %eax
    jz .Luncal
    push %ecx
    rdtsc
    sub BASE_TSC(%ebp), %eax
    sbb BASE_TSC+4(%ebp), %edx
    mov %edx, %edi              # delta high
    mov %eax, %esi              # delta low
    # The 128-bit product delta * mult, in w3:w2:w1:w0 = (%esp):%esi:%ecx:%ebx.
    mull MULT(%ebp)             # dl * ml
    mov %eax, %ebx
    mov %edx, %ecx
    mov %esi, %eax
    mull MULT+4(%ebp)           # dl * mh
    add %eax, %ecx
    adc $0, %edx
    mov %edx, %esi
    push $0
    mov %edi, %eax
    mull MULT(%ebp)             # dh * ml
    add %eax, %ecx
    adc %edx, %esi
    adcl $0, (%esp)
    mov %edi, %eax
    mull MULT+4(%ebp)           # dh * mh
    add %eax, %esi
    adc %edx, (%esp)
    pop %edi                    # w3
    # >> shift: whole words first, then the bits.
    mov SHIFT(%ebp), %al
    test $32, %al
    jz 2f
    mov %ecx, %ebx
    mov %esi, %ecx
    mov %edi, %esi
2:  mov %ecx, %edx
    mov %eax, %ecx
    mov %ebx, %eax
    shrd %cl, %edx, %eax
    shrd %cl, %esi, %edx
    pop %ecx
    add (%ebp,%ecx), %eax
    adc 4(%ebp,%ecx), %edx
    pop %edi
    pop %esi
    pop %ebx
    pop %ebp
    clc
    ret
.Luncal:
    pop %edi
    pop %esi
    pop %ebx
    pop %ebp
    stc
    ret

# ---- split: %edx:%eax nanoseconds -> %edx:%eax seconds, %ecx nanoseconds ----
split:
    push %ebx
    mov $1000000000, %ebx
    mov %eax, %ecx
    mov %edx, %eax
    xor %edx, %edx
    div %ebx                    # high seconds, remainder in %edx
    xchg %eax, %ecx
    div %ebx                    # low seconds, nanoseconds in %edx
    xchg %edx, %ecx
    pop %ebx
    ret

# ---- clock_base: %eax = clock id -> %ecx = its base, or CF set -------------
# 0 REALTIME and 5 REALTIME_COARSE read the wall base; 1 MONOTONIC,
# 4 MONOTONIC_RAW and 6 MONOTONIC_COARSE the monotonic one.
clock_base:
    cmp $6, %eax
    ja 1f
    cmp $2, %eax
    je 1f
    cmp $3, %eax
    je 1f
    mov $MONO, %ecx
    test %eax, %eax
    je 2f
    cmp $5, %eax
    jne 3f
2:  mov $WALL, %ecx
3:  clc
    ret
1:  stc
    ret

# ---- __vdso_clock_gettime(clockid_t, struct old_timespec32 *) ---------------
.globl __vdso_clock_gettime
__vdso_clock_gettime:
    mov 4(%esp), %eax
    call clock_base
    jc 9f
    call ns_now
    jc 9f
    call split
    mov 8(%esp), %edx
    mov %eax, (%edx)
    mov %ecx, 4(%edx)
    xor %eax, %eax
    ret
9:  mov $NR_clock_gettime, %eax
    jmp sys2

# ---- __vdso_clock_gettime64(clockid_t, struct __kernel_timespec *) ---------
.globl __vdso_clock_gettime64
__vdso_clock_gettime64:
    mov 4(%esp), %eax
    call clock_base
    jc 9f
    call ns_now
    jc 9f
    call split
    push %esi
    mov 12(%esp), %esi
    mov %eax, (%esi)
    mov %edx, 4(%esi)
    mov %ecx, 8(%esi)
    movl $0, 12(%esi)
    pop %esi
    xor %eax, %eax
    ret
9:  mov $NR_clock_gettime64, %eax
    jmp sys2

# ---- __vdso_gettimeofday(struct old_timeval32 *, struct timezone *) --------
.globl __vdso_gettimeofday
__vdso_gettimeofday:
    mov 8(%esp), %eax
    test %eax, %eax
    jz 1f
    movl $0, (%eax)
    movl $0, 4(%eax)
1:  cmpl $0, 4(%esp)
    jz 2f
    mov $WALL, %ecx
    call ns_now
    jc 9f
    call split
    mov 4(%esp), %edx
    mov %eax, (%edx)
    mov %ecx, %eax
    xor %edx, %edx
    mov $1000, %ecx
    div %ecx
    mov 4(%esp), %edx
    mov %eax, 4(%edx)
2:  xor %eax, %eax
    ret
9:  mov $NR_gettimeofday, %eax
    jmp sys2

# ---- __vdso_time(old_time32_t *) -------------------------------------------
.globl __vdso_time
__vdso_time:
    mov $WALL, %ecx
    call ns_now
    jc 9f
    call split
    mov 4(%esp), %ecx
    test %ecx, %ecx
    jz 1f
    mov %eax, (%ecx)
1:  ret
9:  push %ebx
    mov 8(%esp), %ebx
    mov $NR_time, %eax
    call VSYSCALL
    pop %ebx
    ret

# ---- __vdso_clock_getres(clockid_t, struct old_timespec32 *) ---------------
# The calibrated clocks tick in nanoseconds, as the system call says.
.globl __vdso_clock_getres
__vdso_clock_getres:
    mov 4(%esp), %eax
    call clock_base
    jc 9f
    mov 8(%esp), %eax
    test %eax, %eax
    jz 1f
    movl $0, (%eax)
    movl $1, 4(%eax)
1:  xor %eax, %eax
    ret
9:  mov $NR_clock_getres, %eax
    jmp sys2

# ---- __vdso_getcpu(unsigned *, unsigned *, void *) -------------------------
.globl __vdso_getcpu
__vdso_getcpu:
    push %ebx
    mov 8(%esp), %ebx
    mov 12(%esp), %ecx
    mov 16(%esp), %edx
    mov $NR_getcpu, %eax
    call VSYSCALL
    pop %ebx
    ret

# ---- sys2: the system call %eax with the caller's two arguments -----------
# Reached by a jmp from a function's entry state (its arguments at 4(%esp)
# and 8(%esp)).
sys2:
    push %ebx
    mov 8(%esp), %ebx
    mov 12(%esp), %ecx
    call VSYSCALL
    pop %ebx
    ret
