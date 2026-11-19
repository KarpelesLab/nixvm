/* What a debugger sees and changes of a task beyond its general registers,
 * and when (x86-64; gcc -static):
 *  - si_errno: seccomp's SECCOMP_RET_TRAP puts its data there, for a
 *    64-bit call and an int $0x80 one; a handler sees it, and so does
 *    PTRACE_GETSIGINFO, and PTRACE_SETSIGINFO's goes to the handler.
 *    rt_sigqueueinfo's arrives through sigtimedwait, signalfd and a
 *    handler.
 *  - A task stopped in signal delivery and continued takes its SIGCONT
 *    before it runs more of its own code: a SIGCONT handler has run when
 *    the kill(SIGSTOP) returns; a tracer that seizes it in the stop sees
 *    PTRACE_EVENT_STOP with SIGSTOP, then (after PTRACE_LISTEN) with
 *    SIGTRAP, then the SIGCONT's signal-delivery-stop, then its next call.
 *  - The FP/SSE registers at syscall-entry and -exit stops: GETFPREGS, the
 *    NT_PRFPREG and NT_X86_XSTATE regsets read them, SETFPREGS and
 *    SETREGSET change what the task has when the call returns.
 *  - A tracer's rip at a syscall-entry or -exit stop is where the call
 *    returns to (and the exit stop shows it), for syscall and int $0x80;
 *    rcx keeps the address after the syscall; a changed eflags is the
 *    one the task gets back.
 *  - The debug registers in struct user's u_debugreg: what they read as,
 *    the writes Linux refuses (DR4/5, kernel addresses, I/O breakpoints,
 *    misaligned or odd-length ones), execute breakpoints, write and
 *    read/write watchpoints and single steps (SIGTRAP with TRAP_HWBKPT or
 *    TRAP_TRACE, and DR6 saying which), not inherited by a fork, and kept
 *    after a detach.
 */
#define _GNU_SOURCE
#include <elf.h>
#include <errno.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/ptrace.h>
#include <sys/signalfd.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <unistd.h>

static int failures;
#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("FAIL line %d: ", __LINE__);                                      \
      printf(__VA_ARGS__);                                                     \
      printf("\n");                                                            \
      failures++;                                                              \
    }                                                                          \
  } while (0)

#if defined(__x86_64__)

#ifndef TRAP_HWBKPT
#define TRAP_HWBKPT 4
#endif
#ifndef PTRACE_LISTEN
#define PTRACE_LISTEN 0x4208
#endif
#define SYS_SECCOMP_CODE 1
#define DR_OFF(n) (offsetof(struct user, u_debugreg) + (n) * 8)

/* An i386 system call from 64-bit code. */
static long int80(long nr) {
  long r;
  __asm__ volatile("int $0x80" : "=a"(r) : "a"(nr) : "memory", "r8", "r9",
                   "r10", "r11");
  return r;
}

static int wait_for(pid_t c) {
  int st = 0;
  CHECK(waitpid(c, &st, __WALL) == c, "waitpid: %m");
  return st;
}

/* A child that stops for its tracer (PTRACE_TRACEME, SIGSTOP), then runs
 * `body` and exits with what it returns. */
static pid_t traced(int (*body)(void)) {
  pid_t c = fork();
  if (c == 0) {
    ptrace(PTRACE_TRACEME, 0, 0, 0);
    raise(SIGSTOP);
    _exit(body());
  }
  int st = wait_for(c);
  CHECK(WIFSTOPPED(st) && WSTOPSIG(st) == SIGSTOP, "first stop %#x", st);
  ptrace(PTRACE_SETOPTIONS, c, 0, PTRACE_O_TRACESYSGOOD | PTRACE_O_EXITKILL);
  return c;
}

/* Let it go from its stop, passing `sig`, and run to its end: its exit
 * status, or -1 for anything else. */
static int finish_with(pid_t c, int sig0) {
  ptrace(PTRACE_CONT, c, 0, sig0);
  for (;;) {
    int st = wait_for(c);
    if (WIFEXITED(st))
      return WEXITSTATUS(st);
    if (!WIFSTOPPED(st))
      return -1;
    int sig = WSTOPSIG(st);
    ptrace(PTRACE_CONT, c, 0, sig == SIGTRAP || sig == (SIGTRAP | 0x80) ? 0 : sig);
  }
}

static int finish(pid_t c) { return finish_with(c, 0); }

/* Resume with PTRACE_SYSCALL until a syscall-entry stop of `nr`. */
static int to_entry(pid_t c, long nr) {
  for (int i = 0; i < 1000; i++) {
    ptrace(PTRACE_SYSCALL, c, 0, 0);
    int st = wait_for(c);
    if (!WIFSTOPPED(st))
      return 0;
    if (WSTOPSIG(st) != (SIGTRAP | 0x80))
      continue;
    struct user_regs_struct r;
    ptrace(PTRACE_GETREGS, c, 0, &r);
    if ((long)r.orig_rax == nr && r.rax == (unsigned long)-ENOSYS)
      return 1;
  }
  return 0;
}

/* Resume with PTRACE_SYSCALL into the exit stop of the call it's in. */
static int to_exit(pid_t c) {
  ptrace(PTRACE_SYSCALL, c, 0, 0);
  int st = wait_for(c);
  return WIFSTOPPED(st) && WSTOPSIG(st) == (SIGTRAP | 0x80);
}

/* ---- si_errno ---------------------------------------------------------- */

static volatile int sys_errno, sys_nr;
static volatile unsigned sys_arch;

static void on_sigsys(int sig, siginfo_t *si, void *uc) {
  (void)sig;
  (void)uc;
  sys_errno = si->si_errno;
  sys_nr = si->si_syscall;
  sys_arch = si->si_arch;
}

/* getppid traps with data 0x1234, the i386 getpid with 0x77. */
static void trap_filter(void) {
  struct sigaction sa = {.sa_sigaction = on_sigsys, .sa_flags = SA_SIGINFO};
  sigaction(SIGSYS, &sa, 0);
  struct sock_filter f[] = {
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, arch)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_I386, 0, 3),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, 20, 0, 4),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_TRAP | 0x77),
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_getppid, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_TRAP | 0x1234),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
  };
  struct sock_fprog prog = {sizeof f / sizeof f[0], f};
  prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
  if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog) != 0)
    _exit(99);
}

static int trapped(void) {
  int bad = 0;
  trap_filter();
  syscall(__NR_getppid);
  if (sys_errno != 0x1234 || sys_nr != __NR_getppid ||
      sys_arch != AUDIT_ARCH_X86_64)
    bad |= 1;
  int80(20);
  if (sys_errno != 0x77 || sys_nr != 20 || sys_arch != AUDIT_ARCH_I386)
    bad |= 2;
  return bad;
}

static int trapped_traced(void) {
  trap_filter();
  syscall(__NR_getppid);
  return sys_errno == 0x55 ? 0 : 1;
}

static void errno_checks(void) {
  pid_t c = fork();
  if (c == 0)
    _exit(trapped());
  int st = wait_for(c);
  CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "RET_TRAP's si_errno: %#x", st);

  /* The tracer sees it, and the one it sets is the handler's. */
  c = traced(trapped_traced);
  ptrace(PTRACE_CONT, c, 0, 0);
  st = wait_for(c);
  CHECK(WIFSTOPPED(st) && WSTOPSIG(st) == SIGSYS, "SIGSYS stop %#x", st);
  siginfo_t si;
  memset(&si, 0, sizeof si);
  CHECK(ptrace(PTRACE_GETSIGINFO, c, 0, &si) == 0, "GETSIGINFO: %m");
  CHECK(si.si_signo == SIGSYS && si.si_errno == 0x1234 &&
            si.si_code == SYS_SECCOMP_CODE && si.si_syscall == __NR_getppid &&
            si.si_arch == AUDIT_ARCH_X86_64,
        "SIGSYS siginfo: signo %d errno %#x code %d syscall %d arch %#x",
        si.si_signo, si.si_errno, si.si_code, si.si_syscall, si.si_arch);
  si.si_errno = 0x55;
  CHECK(ptrace(PTRACE_SETSIGINFO, c, 0, &si) == 0, "SETSIGINFO: %m");
  CHECK(finish_with(c, SIGSYS) == 0,
        "the handler didn't get the tracer's si_errno");

  /* rt_sigqueueinfo's si_errno, through each way of taking a signal. */
  sigset_t set;
  sigemptyset(&set);
  sigaddset(&set, SIGUSR2);
  sigprocmask(SIG_BLOCK, &set, 0);
  siginfo_t q;
  memset(&q, 0, sizeof q);
  q.si_signo = SIGUSR2;
  q.si_code = SI_QUEUE;
  q.si_errno = 33;
  q.si_pid = getpid();
  q.si_uid = getuid();
  q.si_value.sival_int = 5;
  CHECK(syscall(SYS_rt_sigqueueinfo, getpid(), SIGUSR2, &q) == 0,
        "rt_sigqueueinfo: %m");
  struct timespec zero = {0, 0};
  memset(&si, 0, sizeof si);
  CHECK(sigtimedwait(&set, &si, &zero) == SIGUSR2 && si.si_errno == 33 &&
            si.si_code == SI_QUEUE && si.si_value.sival_int == 5,
        "sigtimedwait: errno %d code %d", si.si_errno, si.si_code);
  syscall(SYS_rt_sigqueueinfo, getpid(), SIGUSR2, &q);
  int fd = signalfd(-1, &set, 0);
  struct signalfd_siginfo ss;
  memset(&ss, 0, sizeof ss);
  CHECK(fd >= 0 && read(fd, &ss, sizeof ss) == sizeof ss &&
            ss.ssi_signo == SIGUSR2 && ss.ssi_errno == 33 &&
            ss.ssi_code == SI_QUEUE && ss.ssi_int == 5,
        "signalfd: errno %d code %d", ss.ssi_errno, ss.ssi_code);
  close(fd);
  syscall(SYS_rt_sigqueueinfo, getpid(), SIGUSR2, &q);
  struct sigaction sa = {.sa_sigaction = on_sigsys, .sa_flags = SA_SIGINFO};
  sigaction(SIGUSR2, &sa, 0);
  sys_errno = 0;
  sigprocmask(SIG_UNBLOCK, &set, 0);
  CHECK(sys_errno == 33, "a handler's si_errno %d", sys_errno);
  signal(SIGUSR2, SIG_DFL);
}

/* ---- SIGCONT ------------------------------------------------------------ */

static volatile sig_atomic_t conts;

static void on_cont(int sig) {
  (void)sig;
  conts++;
}

static void sigcont_checks(void) {
  /* Its handler runs before kill returns: no other call in between. */
  pid_t c = fork();
  if (c == 0) {
    signal(SIGCONT, on_cont);
    syscall(SYS_kill, getpid(), SIGSTOP);
    _exit(conts == 1 ? 0 : 1);
  }
  int st;
  CHECK(waitpid(c, &st, WUNTRACED) == c && WIFSTOPPED(st), "stop %#x", st);
  kill(c, SIGCONT);
  st = wait_for(c);
  CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0,
        "the SIGCONT handler ran late: %#x", st);

  /* Seized in its stop, as strace starts a program. */
  c = fork();
  if (c == 0) {
    syscall(SYS_kill, getpid(), SIGSTOP);
    syscall(SYS_getpid);
    _exit(0);
  }
  CHECK(waitpid(c, &st, WSTOPPED) == c && WIFSTOPPED(st), "stop %#x", st);
  CHECK(ptrace(PTRACE_SEIZE, c, 0,
               PTRACE_O_TRACESYSGOOD | PTRACE_O_EXITKILL) == 0,
        "SEIZE: %m");
  CHECK(ptrace(PTRACE_INTERRUPT, c, 0, 0) == 0, "INTERRUPT: %m");
  kill(c, SIGCONT);
  st = wait_for(c);
  CHECK(st >> 8 == (SIGSTOP | PTRACE_EVENT_STOP << 8),
        "seized in a group stop: %#x", st);
  CHECK(ptrace(PTRACE_LISTEN, c, 0, 0) == 0, "LISTEN: %m");
  st = wait_for(c);
  CHECK(st >> 8 == (SIGTRAP | PTRACE_EVENT_STOP << 8), "after SIGCONT: %#x",
        st);
  ptrace(PTRACE_SYSCALL, c, 0, 0);
  st = wait_for(c);
  CHECK(st >> 8 == SIGCONT, "the SIGCONT's stop: %#x", st);
  siginfo_t si;
  memset(&si, 0, sizeof si);
  ptrace(PTRACE_GETSIGINFO, c, 0, &si);
  CHECK(si.si_signo == SIGCONT && si.si_code == SI_USER &&
            si.si_pid == getpid(),
        "SIGCONT siginfo %d %d %d", si.si_signo, si.si_code, si.si_pid);
  ptrace(PTRACE_SYSCALL, c, 0, 0);
  st = wait_for(c);
  struct user_regs_struct r;
  ptrace(PTRACE_GETREGS, c, 0, &r);
  CHECK(st >> 8 == (SIGTRAP | 0x80) && r.orig_rax == __NR_getpid,
        "then getpid's entry: %#x, orig_rax %lld", st, (long long)r.orig_rax);
  ptrace(PTRACE_DETACH, c, 0, 0);
  st = wait_for(c);
  CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "exit %#x", st);
}

/* ---- FP state at syscall stops -------------------------------------------- */

static const uint64_t FA[2] = {0x1111222233334444ull, 0x5555666677778888ull};
static const uint64_t FB[2] = {0x0123456789abcdefull, 0xfedcba9876543210ull};
static const uint64_t FC[2] = {0xc0c0c0c0c0c0c0c0ull, 0x0c0c0c0c0c0c0c0cull};

static int fp_body(void) {
  uint64_t o0[2], o1[2];
  unsigned short cw = 0x27f, after;
  __asm__ volatile("fldcw %0" ::"m"(cw));
  __asm__ volatile("movdqu (%2), %%xmm0\n\t"
                   "movdqu (%2), %%xmm1\n\t"
                   "mov %3, %%eax\n\t"
                   "syscall\n\t"
                   "movdqu %%xmm0, (%0)\n\t"
                   "movdqu %%xmm1, (%1)\n\t"
                   :
                   : "r"(o0), "r"(o1), "r"(FA), "i"(__NR_getppid)
                   : "rax", "rcx", "r11", "xmm0", "xmm1", "memory");
  __asm__ volatile("fnstcw %0" : "=m"(after));
  int bad = 0;
  if (memcmp(o0, FB, 16))
    bad |= 1;
  if (memcmp(o1, FC, 16))
    bad |= 2;
  if (after != 0x37f)
    bad |= 4;
  return bad;
}

static void fp_checks(void) {
  pid_t c = traced(fp_body);
  CHECK(to_entry(c, __NR_getppid), "no getppid entry stop");
  struct user_fpregs_struct fp;
  CHECK(ptrace(PTRACE_GETFPREGS, c, 0, &fp) == 0, "GETFPREGS: %m");
  CHECK(fp.cwd == 0x27f, "entry stop cwd %#x", fp.cwd);
  CHECK(memcmp(&fp.xmm_space[0], FA, 16) == 0 &&
            memcmp(&fp.xmm_space[4], FA, 16) == 0,
        "entry stop xmm0 %#x %#x", fp.xmm_space[0], fp.xmm_space[1]);
  unsigned char buf[4096];
  struct iovec iov = {buf, sizeof buf};
  CHECK(ptrace(PTRACE_GETREGSET, c, NT_PRFPREG, &iov) == 0 &&
            iov.iov_len == 512 && memcmp(buf, &fp, 512) == 0,
        "NT_PRFPREG at the entry stop: %zu bytes", iov.iov_len);
  iov = (struct iovec){buf, sizeof buf};
  CHECK(ptrace(PTRACE_GETREGSET, c, NT_X86_XSTATE, &iov) == 0 &&
            iov.iov_len >= 576 && memcmp(buf + 160, FA, 16) == 0 &&
            buf[0] == 0x7f && buf[1] == 2,
        "NT_X86_XSTATE at the entry stop: %zu bytes", iov.iov_len);
  memcpy(&fp.xmm_space[0], FB, 16);
  fp.cwd = 0x37f;
  CHECK(ptrace(PTRACE_SETFPREGS, c, 0, &fp) == 0, "SETFPREGS: %m");
  CHECK(to_exit(c), "no exit stop");
  memset(&fp, 0, sizeof fp);
  ptrace(PTRACE_GETFPREGS, c, 0, &fp);
  CHECK(memcmp(&fp.xmm_space[0], FB, 16) == 0 && fp.cwd == 0x37f,
        "exit stop: xmm0 %#x cwd %#x", fp.xmm_space[0], fp.cwd);
  memcpy(&fp.xmm_space[4], FC, 16);
  iov = (struct iovec){&fp, sizeof fp};
  CHECK(ptrace(PTRACE_SETREGSET, c, NT_PRFPREG, &iov) == 0, "SETREGSET: %m");
  int bad = finish(c);
  CHECK(bad == 0, "the task's FP state after the call: %d", bad);
}

/* ---- rip at syscall stops ------------------------------------------------- */

/* rip_call(nr, out): a syscall; out[0] its result, out[1] 1 straight after
 * it or 2 at rip_target (where only a tracer sends it), out[2] rcx there,
 * out[3] the flags. rip_call32 does the same with int $0x80. */
extern char rip_after[], rip_target[], rip_after32[], rip_target32[];
long rip_call(long nr, long *out);
long rip_call32(long nr, long *out);
__asm__(".text\n"
        ".globl rip_call\n"
        "rip_call:\n"
        "  mov %rdi, %rax\n"
        "  mov %rsi, %r8\n"
        "  syscall\n"
        ".globl rip_after\n"
        "rip_after:\n"
        "  movq $1, 8(%r8)\n"
        "  jmp 1f\n"
        ".globl rip_target\n"
        "rip_target:\n"
        "  movq $2, 8(%r8)\n"
        "  mov %rcx, 16(%r8)\n"
        "1:\n"
        "  mov %rax, (%r8)\n"
        "  pushfq\n"
        "  popq 24(%r8)\n"
        "  ret\n"
        ".globl rip_call32\n"
        "rip_call32:\n"
        "  push %rbx\n"
        "  mov %rdi, %rax\n"
        "  mov %rsi, %r8\n"
        "  int $0x80\n"
        ".globl rip_after32\n"
        "rip_after32:\n"
        "  movq $1, 8(%r8)\n"
        "  jmp 2f\n"
        ".globl rip_target32\n"
        "rip_target32:\n"
        "  movq $2, 8(%r8)\n"
        "2:\n"
        "  mov %rax, (%r8)\n"
        "  pop %rbx\n"
        "  ret\n");

static int rip_body(void) {
  long a[4] = {0}, b[4] = {0}, d[4] = {0}, e[4] = {0};
  int bad = 0;
  rip_call(__NR_getppid, a);
  if (a[0] != getppid() || a[1] != 2 || a[2] != (long)rip_after)
    bad |= 1;
  rip_call(__NR_getuid, b);
  if (b[0] != (long)getuid() || b[1] != 2)
    bad |= 2;
  rip_call32(64, d); /* i386 getppid */
  if (d[0] != getppid() || d[1] != 2)
    bad |= 4;
  rip_call(__NR_getpgrp, e);
  if (e[0] != getpgrp() || e[1] != 1 || !(e[3] & 1))
    bad |= 8;
  return bad;
}

static void rip_checks(void) {
  pid_t c = traced(rip_body);
  struct user_regs_struct r;
  /* Moved at the entry stop: the exit stop shows it, the call still runs. */
  CHECK(to_entry(c, __NR_getppid), "no getppid entry stop");
  ptrace(PTRACE_GETREGS, c, 0, &r);
  CHECK(r.rip == (unsigned long)rip_after && r.rcx == r.rip,
        "entry stop rip %#llx rcx %#llx", (unsigned long long)r.rip,
        (unsigned long long)r.rcx);
  r.rip = (unsigned long)rip_target;
  CHECK(ptrace(PTRACE_SETREGS, c, 0, &r) == 0, "SETREGS: %m");
  CHECK(to_exit(c), "no exit stop");
  ptrace(PTRACE_GETREGS, c, 0, &r);
  CHECK(r.rip == (unsigned long)rip_target && (long)r.rax == getpid(),
        "exit stop rip %#llx rax %lld", (unsigned long long)r.rip,
        (long long)r.rax);
  /* Moved at the exit stop. */
  CHECK(to_entry(c, __NR_getuid), "no getuid entry stop");
  CHECK(to_exit(c), "no exit stop");
  ptrace(PTRACE_GETREGS, c, 0, &r);
  CHECK(r.rip == (unsigned long)rip_after, "exit stop rip %#llx",
        (unsigned long long)r.rip);
  CHECK(ptrace(PTRACE_POKEUSER, c, offsetof(struct user, regs.rip),
               rip_target) == 0,
        "POKEUSER rip: %m");
  /* The i386 call, moved at its entry. */
  CHECK(to_entry(c, 64), "no int $0x80 getppid entry stop");
  ptrace(PTRACE_GETREGS, c, 0, &r);
  CHECK(r.rip == (unsigned long)rip_after32, "int $0x80 entry rip %#llx",
        (unsigned long long)r.rip);
  r.rip = (unsigned long)rip_target32;
  ptrace(PTRACE_SETREGS, c, 0, &r);
  /* Carry set at an entry stop, rip left alone. */
  CHECK(to_entry(c, __NR_getpgrp), "no getpgrp entry stop");
  ptrace(PTRACE_GETREGS, c, 0, &r);
  r.eflags |= 1;
  ptrace(PTRACE_SETREGS, c, 0, &r);
  int bad = finish(c);
  CHECK(bad == 0, "where the calls returned: %d", bad);
}

/* ---- debug registers ---------------------------------------------------- */

__attribute__((noinline)) static int hw_func(int x) {
  __asm__ volatile("" ::: "memory");
  return x * 3 + 1;
}

static volatile uint32_t wvar, rvar = 7;
static volatile uint64_t pad8;

static long peek_dr(pid_t c, int n) {
  errno = 0;
  return ptrace(PTRACE_PEEKUSER, c, DR_OFF(n), 0);
}

static long poke_dr(pid_t c, int n, unsigned long v) {
  return ptrace(PTRACE_POKEUSER, c, DR_OFF(n), v);
}

static int idle_body(void) { return hw_func(1) == 4 ? 0 : 1; }

static void dr_validation(void) {
  pid_t c = traced(idle_body);
  CHECK(peek_dr(c, 0) == 0 && errno == 0, "DR0 %#lx", peek_dr(c, 0));
  CHECK(peek_dr(c, 4) == 0 && peek_dr(c, 5) == 0, "DR4/DR5");
  CHECK((unsigned long)peek_dr(c, 6) == 0xffff0ff0, "DR6 %#lx", peek_dr(c, 6));
  CHECK(peek_dr(c, 7) == 0, "DR7 %#lx", peek_dr(c, 7));
  CHECK(poke_dr(c, 4, 1) == -1 && errno == EIO, "POKE DR4: %m");
  CHECK(poke_dr(c, 5, 1) == -1 && errno == EIO, "POKE DR5: %m");
  /* A kernel address, and the last byte of the 47-bit user half. */
  CHECK(poke_dr(c, 0, 0xffffffff81000000ul) == -1 && errno == EINVAL,
        "POKE DR0 kernel: %m");
  CHECK(poke_dr(c, 0, 0x7fffffffefffl) == 0, "POKE DR0 last byte: %m");
  CHECK(peek_dr(c, 0) == 0x7fffffffefffl, "DR0 %#lx", peek_dr(c, 0));
  /* An I/O breakpoint, an execute one 2 bytes long. */
  CHECK(poke_dr(c, 7, 0x20001) == -1 && errno == EINVAL, "I/O bp: %m");
  CHECK(poke_dr(c, 7, 0x40001) == -1 && errno == EINVAL, "2-byte exec: %m");
  CHECK(peek_dr(c, 7) == 0, "DR7 after refusals %#lx", peek_dr(c, 7));
  /* A 4-byte watchpoint at an odd address; then aligned. */
  CHECK(poke_dr(c, 1, (unsigned long)&wvar + 1) == 0, "POKE DR1: %m");
  CHECK(poke_dr(c, 7, 0xd00004) == -1 && errno == EINVAL, "misaligned: %m");
  CHECK(poke_dr(c, 1, (unsigned long)&wvar) == 0, "POKE DR1: %m");
  CHECK(poke_dr(c, 7, 0xd00004) == 0, "4-byte watchpoint: %m");
  CHECK(peek_dr(c, 7) == 0xd00004, "DR7 %#lx", peek_dr(c, 7));
  /* Now 4 bytes long: the address has to stay aligned. */
  CHECK(poke_dr(c, 1, (unsigned long)&wvar + 2) == -1 && errno == EINVAL,
        "DR1 misaligned for its length: %m");
  CHECK(peek_dr(c, 1) == (long)&wvar, "DR1 %#lx", peek_dr(c, 1));
  /* DR7's reserved bits read back as written. */
  CHECK(poke_dr(c, 7, 0x500) == 0 && peek_dr(c, 7) == 0x500, "DR7 %#lx",
        peek_dr(c, 7));
  CHECK(poke_dr(c, 6, 0) == 0 && peek_dr(c, 6) == 0, "DR6 %#lx",
        peek_dr(c, 6));
  /* struct user: words past the registers can't be written; the end. */
  CHECK(ptrace(PTRACE_POKEUSER, c, offsetof(struct user, u_fpvalid), 0) ==
                -1 && errno == EIO,
        "POKEUSER u_fpvalid: %m");
  /* The kernel's struct user is 928 bytes (glibc's stops before its
   * error_code and fault_address). */
  errno = 0;
  ptrace(PTRACE_PEEKUSER, c, 928, 0);
  CHECK(errno == EIO, "PEEKUSER at the end: %m");
  errno = 0;
  ptrace(PTRACE_PEEKUSER, c, 920, 0);
  CHECK(errno == 0, "PEEKUSER of the last word: %m");
  CHECK(finish(c) == 0, "exit");
}

static int exec_body(void) {
  int s = hw_func(1) + hw_func(2);
  return s == 4 + 7 ? 0 : 1;
}

static void hit(pid_t c, int code, unsigned long addr, long dr6, const char *what) {
  int st = wait_for(c);
  CHECK(WIFSTOPPED(st) && WSTOPSIG(st) == SIGTRAP, "%s: stop %#x", what, st);
  siginfo_t si;
  memset(&si, 0, sizeof si);
  ptrace(PTRACE_GETSIGINFO, c, 0, &si);
  struct user_regs_struct r;
  ptrace(PTRACE_GETREGS, c, 0, &r);
  CHECK(si.si_signo == SIGTRAP && si.si_code == code, "%s: code %d", what,
        si.si_code);
  CHECK((unsigned long)si.si_addr == r.rip, "%s: si_addr %p rip %#llx", what,
        si.si_addr, (unsigned long long)r.rip);
  if (addr)
    CHECK(r.rip == addr, "%s: rip %#llx want %#lx", what,
          (unsigned long long)r.rip, addr);
  CHECK(peek_dr(c, 6) == dr6, "%s: DR6 %#lx want %#lx", what, peek_dr(c, 6),
        dr6);
}

static int watch_body(void) {
  wvar = 42;
  uint32_t v = rvar;
  pad8 = v;
  return 0;
}

static int forks_body(void) {
  pid_t g = fork();
  if (g == 0)
    _exit(hw_func(1) == 4 ? 0 : 1);
  int st;
  waitpid(g, &st, 0);
  return WIFEXITED(st) && WEXITSTATUS(st) == 0 ? 0 : 1;
}

static volatile int trap_code;
static void *volatile trap_addr;
static void on_trap(int sig, siginfo_t *si, void *uc) {
  (void)sig;
  (void)uc;
  trap_code = si->si_code;
  trap_addr = si->si_addr;
}

static int detached_body(void) {
  int r = hw_func(1);
  return r == 4 && trap_code == TRAP_HWBKPT && trap_addr == (void *)hw_func
             ? 0
             : 1;
}

static void dr_hits(void) {
  /* An execute breakpoint, hit by both calls. */
  pid_t c = traced(exec_body);
  CHECK(poke_dr(c, 0, (unsigned long)hw_func) == 0, "DR0: %m");
  CHECK(poke_dr(c, 7, 1) == 0, "DR7: %m");
  ptrace(PTRACE_CONT, c, 0, 0);
  hit(c, TRAP_HWBKPT, (unsigned long)hw_func, 0xffff0ff1, "exec bp");
  ptrace(PTRACE_CONT, c, 0, 0);
  hit(c, TRAP_HWBKPT, (unsigned long)hw_func, 0xffff0ff1, "exec bp again");
  CHECK(poke_dr(c, 7, 0) == 0, "DR7 0: %m");
  CHECK(finish(c) == 0, "exec bp run");

  /* A write watchpoint (DR1) and a read/write one (DR2). */
  c = traced(watch_body);
  CHECK(poke_dr(c, 1, (unsigned long)&wvar) == 0, "DR1: %m");
  CHECK(poke_dr(c, 2, (unsigned long)&rvar) == 0, "DR2: %m");
  CHECK(poke_dr(c, 7, 0x4 | 0x10 | 0xd00000 | 0xf000000) == 0, "DR7: %m");
  ptrace(PTRACE_CONT, c, 0, 0);
  hit(c, TRAP_HWBKPT, 0, 0xffff0ff2, "write watchpoint");
  errno = 0;
  long w = ptrace(PTRACE_PEEKDATA, c, &wvar, 0);
  CHECK(errno == 0 && (uint32_t)w == 42, "the write is done: %#lx", w);
  ptrace(PTRACE_CONT, c, 0, 0);
  hit(c, TRAP_HWBKPT, 0, 0xffff0ff4, "read watchpoint");
  CHECK(finish(c) == 0, "watchpoint run");

  /* A single step: DR6 says BS. */
  c = traced(idle_body);
  ptrace(PTRACE_SINGLESTEP, c, 0, 0);
  hit(c, TRAP_TRACE, 0, 0xffff4ff0, "single step");
  CHECK(finish(c) == 0, "step run");

  /* A fork doesn't get them. */
  c = traced(forks_body);
  poke_dr(c, 0, (unsigned long)hw_func);
  poke_dr(c, 7, 1);
  CHECK(finish(c) == 0, "a fork child hit its parent's breakpoint");

  /* Still there after a detach: the task's own handler gets the trap. */
  struct sigaction sa = {.sa_sigaction = on_trap, .sa_flags = SA_SIGINFO};
  sigaction(SIGTRAP, &sa, 0);
  c = traced(detached_body);
  signal(SIGTRAP, SIG_DFL);
  poke_dr(c, 0, (unsigned long)hw_func);
  poke_dr(c, 7, 1);
  ptrace(PTRACE_DETACH, c, 0, 0);
  int st = wait_for(c);
  CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0,
        "after a detach, the breakpoint: %#x", st);
}

int main(void) {
  setvbuf(stdout, 0, _IONBF, 0);
  errno_checks();
  sigcont_checks();
  fp_checks();
  rip_checks();
  dr_validation();
  dr_hits();
  if (failures == 0)
    printf("PASS\n");
  return failures != 0;
}

#else
int main(void) {
  printf("PASS\n");
  return 0;
}
#endif
