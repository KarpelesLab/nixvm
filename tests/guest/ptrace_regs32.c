/* ptrace_regs for i386 (gcc -m32 -static): an i386 tracer of an i386
 * tracee, through Linux's compat paths.
 *  - si_errno: seccomp's SECCOMP_RET_TRAP data in the i386 siginfo, for a
 *    handler and for PTRACE_GETSIGINFO; PTRACE_SETSIGINFO's goes to the
 *    handler.
 *  - The FPU at syscall-entry and -exit stops: GETFPREGS (fsave),
 *    GETFPXREGS (fxsave) and their SET requests.
 *  - eip at a syscall-entry or -exit stop is where the call returns to, and
 *    a changed eflags is the one the task gets back.
 *  - struct user32's u_debugreg: what the registers read as, the writes
 *    Linux refuses, an execute breakpoint and a write watchpoint (SIGTRAP
 *    with TRAP_HWBKPT, DR6 saying which), and a single step's DR6.
 */
#define _GNU_SOURCE
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
#include <sys/syscall.h>
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

#if defined(__i386__)

#ifndef TRAP_HWBKPT
#define TRAP_HWBKPT 4
#endif
#define DR_OFF(n) (offsetof(struct user, u_debugreg) + (n) * 4)

static long int80(long nr) {
  long r;
  __asm__ volatile("int $0x80" : "=a"(r) : "a"(nr) : "memory");
  return r;
}

static int wait_for(pid_t c) {
  int st = 0;
  CHECK(waitpid(c, &st, __WALL) == c, "waitpid: %m");
  return st;
}

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
    if (r.orig_eax == nr && r.eax == -ENOSYS)
      return 1;
  }
  return 0;
}

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

/* getppid (64) traps with data 0x99. */
static void trap_filter(void) {
  struct sigaction sa = {.sa_sigaction = on_sigsys, .sa_flags = SA_SIGINFO};
  sigaction(SIGSYS, &sa, 0);
  struct sock_filter f[] = {
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, 64, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_TRAP | 0x99),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
  };
  struct sock_fprog prog = {sizeof f / sizeof f[0], f};
  prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
  if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &prog) != 0)
    _exit(99);
}

static int trapped(void) {
  trap_filter();
  int80(64);
  return sys_errno == 0x99 && sys_nr == 64 && sys_arch == AUDIT_ARCH_I386 ? 0
                                                                          : 1;
}

static int trapped_traced(void) {
  trap_filter();
  int80(64);
  return sys_errno == 0x55 ? 0 : 1;
}

static void errno_checks(void) {
  pid_t c = fork();
  if (c == 0)
    _exit(trapped());
  int st = wait_for(c);
  CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "RET_TRAP's si_errno: %#x", st);

  c = traced(trapped_traced);
  ptrace(PTRACE_CONT, c, 0, 0);
  st = wait_for(c);
  CHECK(WIFSTOPPED(st) && WSTOPSIG(st) == SIGSYS, "SIGSYS stop %#x", st);
  siginfo_t si;
  memset(&si, 0, sizeof si);
  CHECK(ptrace(PTRACE_GETSIGINFO, c, 0, &si) == 0, "GETSIGINFO: %m");
  CHECK(si.si_signo == SIGSYS && si.si_errno == 0x99 && si.si_code == 1 &&
            si.si_syscall == 64 && si.si_arch == AUDIT_ARCH_I386,
        "SIGSYS siginfo: signo %d errno %#x code %d syscall %d arch %#x",
        si.si_signo, si.si_errno, si.si_code, si.si_syscall, si.si_arch);
  si.si_errno = 0x55;
  CHECK(ptrace(PTRACE_SETSIGINFO, c, 0, &si) == 0, "SETSIGINFO: %m");
  CHECK(finish_with(c, SIGSYS) == 0,
        "the handler didn't get the tracer's si_errno");
}

/* ---- the FPU at syscall stops ------------------------------------------- */

static const uint32_t XA[4] = {0x11112222, 0x33334444, 0x55556666, 0x77778888};
static const uint32_t XB[4] = {0x01234567, 0x89abcdef, 0xfedcba98, 0x76543210};

static int fp_body(void) {
  uint32_t o[4];
  unsigned short cw = 0x27f, after;
  __asm__ volatile("fldcw %0" ::"m"(cw));
  __asm__ volatile("movdqu (%1), %%xmm0\n\t"
                   "mov $64, %%eax\n\t"
                   "int $0x80\n\t"
                   "movdqu %%xmm0, (%0)\n\t"
                   :
                   : "r"(o), "r"(XA)
                   : "eax", "xmm0", "memory");
  __asm__ volatile("fnstcw %0" : "=m"(after));
  int bad = 0;
  if (memcmp(o, XB, 16))
    bad |= 1;
  if (after != 0x37f)
    bad |= 2;
  return bad;
}

static void fp_checks(void) {
  pid_t c = traced(fp_body);
  CHECK(to_entry(c, 64), "no getppid entry stop");
  struct user_fpregs_struct fp;
  CHECK(ptrace(PTRACE_GETFPREGS, c, 0, &fp) == 0, "GETFPREGS: %m");
  CHECK((fp.cwd & 0xffff) == 0x27f, "entry stop fsave cwd %#lx", fp.cwd);
  struct user_fpxregs_struct fx;
  CHECK(ptrace(PTRACE_GETFPXREGS, c, 0, &fx) == 0, "GETFPXREGS: %m");
  CHECK(fx.cwd == 0x27f && memcmp(fx.xmm_space, XA, 16) == 0,
        "entry stop fxsave cwd %#x xmm0 %#lx", fx.cwd, fx.xmm_space[0]);
  fp.cwd = 0x37f;
  CHECK(ptrace(PTRACE_SETFPREGS, c, 0, &fp) == 0, "SETFPREGS: %m");
  CHECK(to_exit(c), "no exit stop");
  memset(&fx, 0, sizeof fx);
  ptrace(PTRACE_GETFPXREGS, c, 0, &fx);
  CHECK(fx.cwd == 0x37f && memcmp(fx.xmm_space, XA, 16) == 0,
        "exit stop: cwd %#x xmm0 %#lx", fx.cwd, fx.xmm_space[0]);
  memcpy(fx.xmm_space, XB, 16);
  CHECK(ptrace(PTRACE_SETFPXREGS, c, 0, &fx) == 0, "SETFPXREGS: %m");
  int bad = finish(c);
  CHECK(bad == 0, "the task's FPU after the call: %d", bad);
}

/* ---- eip at syscall stops ------------------------------------------------- */

/* eip_call(nr, out): an int $0x80; out[0] its result, out[1] 1 straight
 * after it or 2 at eip_target, out[2] the flags. */
extern char eip_after[], eip_target[];
long eip_call(long nr, long *out);
__asm__(".text\n"
        ".globl eip_call\n"
        "eip_call:\n"
        "  mov 4(%esp), %eax\n"
        "  mov 8(%esp), %edx\n"
        "  int $0x80\n"
        ".globl eip_after\n"
        "eip_after:\n"
        "  movl $1, 4(%edx)\n"
        "  jmp 1f\n"
        ".globl eip_target\n"
        "eip_target:\n"
        "  movl $2, 4(%edx)\n"
        "1:\n"
        "  mov %eax, (%edx)\n"
        "  pushf\n"
        "  popl 8(%edx)\n"
        "  ret\n");

static int eip_body(void) {
  long a[3] = {0}, b[3] = {0}, d[3] = {0};
  int bad = 0;
  eip_call(64, a); /* getppid */
  if (a[0] != getppid() || a[1] != 2)
    bad |= 1;
  eip_call(199, b); /* getuid32 */
  if (b[0] != (long)getuid() || b[1] != 2)
    bad |= 2;
  eip_call(65, d); /* getpgrp */
  if (d[0] != getpgrp() || d[1] != 1 || !(d[2] & 1))
    bad |= 4;
  return bad;
}

static void eip_checks(void) {
  pid_t c = traced(eip_body);
  struct user_regs_struct r;
  CHECK(to_entry(c, 64), "no getppid entry stop");
  ptrace(PTRACE_GETREGS, c, 0, &r);
  CHECK(r.eip == (long)eip_after, "entry stop eip %#lx", r.eip);
  r.eip = (long)eip_target;
  CHECK(ptrace(PTRACE_SETREGS, c, 0, &r) == 0, "SETREGS: %m");
  CHECK(to_exit(c), "no exit stop");
  ptrace(PTRACE_GETREGS, c, 0, &r);
  CHECK(r.eip == (long)eip_target && r.eax == getpid(),
        "exit stop eip %#lx eax %ld", r.eip, r.eax);
  CHECK(to_entry(c, 199), "no getuid32 entry stop");
  CHECK(to_exit(c), "no exit stop");
  CHECK(ptrace(PTRACE_POKEUSER, c, offsetof(struct user, regs.eip),
               eip_target) == 0,
        "POKEUSER eip: %m");
  CHECK(to_entry(c, 65), "no getpgrp entry stop");
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

static volatile uint32_t wvar;

static long peek_dr(pid_t c, int n) {
  errno = 0;
  return ptrace(PTRACE_PEEKUSER, c, DR_OFF(n), 0);
}

static long poke_dr(pid_t c, int n, unsigned long v) {
  return ptrace(PTRACE_POKEUSER, c, DR_OFF(n), v);
}

static int idle_body(void) { return hw_func(1) == 4 ? 0 : 1; }

static void hit(pid_t c, int code, unsigned long addr, unsigned long dr6,
                const char *what) {
  int st = wait_for(c);
  CHECK(WIFSTOPPED(st) && WSTOPSIG(st) == SIGTRAP, "%s: stop %#x", what, st);
  siginfo_t si;
  memset(&si, 0, sizeof si);
  ptrace(PTRACE_GETSIGINFO, c, 0, &si);
  struct user_regs_struct r;
  ptrace(PTRACE_GETREGS, c, 0, &r);
  CHECK(si.si_signo == SIGTRAP && si.si_code == code, "%s: code %d", what,
        si.si_code);
  CHECK((unsigned long)si.si_addr == (unsigned long)r.eip,
        "%s: si_addr %p eip %#lx", what, si.si_addr, r.eip);
  if (addr)
    CHECK((unsigned long)r.eip == addr, "%s: eip %#lx want %#lx", what, r.eip,
          addr);
  CHECK((unsigned long)peek_dr(c, 6) == dr6, "%s: DR6 %#lx want %#lx", what,
        peek_dr(c, 6), dr6);
}

static int exec_body(void) {
  int s = hw_func(1) + hw_func(2);
  wvar = 42;
  return s == 4 + 7 ? 0 : 1;
}

static void dr_checks(void) {
  pid_t c = traced(idle_body);
  CHECK(peek_dr(c, 0) == 0 && errno == 0, "DR0 %#lx", peek_dr(c, 0));
  CHECK((unsigned long)peek_dr(c, 6) == 0xffff0ff0, "DR6 %#lx", peek_dr(c, 6));
  CHECK(peek_dr(c, 7) == 0, "DR7 %#lx", peek_dr(c, 7));
  CHECK(poke_dr(c, 4, 1) == -1 && errno == EIO, "POKE DR4: %m");
  CHECK(poke_dr(c, 7, 0x20001) == -1 && errno == EINVAL, "I/O bp: %m");
  CHECK(poke_dr(c, 1, (unsigned long)&wvar + 1) == 0, "POKE DR1: %m");
  CHECK(poke_dr(c, 7, 0xd00004) == -1 && errno == EINVAL, "misaligned: %m");
  CHECK(peek_dr(c, 7) == 0, "DR7 after a refusal %#lx", peek_dr(c, 7));
  /* 8 bytes long: a 64-bit kernel takes it from a 32-bit tracer too. */
  CHECK(poke_dr(c, 1, (unsigned long)&wvar & ~7ul) == 0, "POKE DR1: %m");
  CHECK(poke_dr(c, 7, 0x900004) == 0, "8-byte watchpoint: %m");
  CHECK(poke_dr(c, 7, 0) == 0 && poke_dr(c, 1, 0) == 0, "clear: %m");
  ptrace(PTRACE_SINGLESTEP, c, 0, 0);
  hit(c, TRAP_TRACE, 0, 0xffff4ff0, "single step");
  CHECK(poke_dr(c, 6, 0) == 0 && peek_dr(c, 6) == 0, "DR6 %#lx",
        peek_dr(c, 6));
  CHECK(finish(c) == 0, "step run");

  c = traced(exec_body);
  CHECK(poke_dr(c, 0, (unsigned long)hw_func) == 0, "DR0: %m");
  CHECK(poke_dr(c, 1, (unsigned long)&wvar) == 0, "DR1: %m");
  CHECK(poke_dr(c, 7, 0x1 | 0x4 | 0xd00000) == 0, "DR7: %m");
  CHECK(peek_dr(c, 0) == (long)hw_func, "DR0 %#lx", peek_dr(c, 0));
  ptrace(PTRACE_CONT, c, 0, 0);
  hit(c, TRAP_HWBKPT, (unsigned long)hw_func, 0xffff0ff1, "exec bp");
  ptrace(PTRACE_CONT, c, 0, 0);
  hit(c, TRAP_HWBKPT, (unsigned long)hw_func, 0xffff0ff1, "exec bp again");
  ptrace(PTRACE_CONT, c, 0, 0);
  hit(c, TRAP_HWBKPT, 0, 0xffff0ff2, "write watchpoint");
  errno = 0;
  long w = ptrace(PTRACE_PEEKDATA, c, &wvar, 0);
  CHECK(errno == 0 && w == 42, "the write is done: %#lx", w);
  CHECK(finish(c) == 0, "breakpoint run");
}

int main(void) {
  setvbuf(stdout, 0, _IONBF, 0);
  errno_checks();
  fp_checks();
  eip_checks();
  dr_checks();
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
