/* What a debugger needs from ptrace beyond strace's syscall stops: single
 * steps, breakpoint traps (a breakpoint instruction in the code, and one the
 * tracer writes into the tracee's text), a fault stopping the tracee for the
 * tracer to repair, and the floating-point registers. Each case forks its own
 * tracee; a fork has the tracer's addresses, so the tracer names the
 * tracee's functions by its own pointers. */
#define _GNU_SOURCE
#include <elf.h>
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ptrace.h>
#include <sys/uio.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d, errno %d)\n", #c, __LINE__, errno); exit(1); } } while (0)

#if defined(__x86_64__)
typedef struct user_regs_struct regs_t;
static void get_regs(pid_t p, regs_t *r) { CHECK(ptrace(PTRACE_GETREGS, p, 0, r) == 0); }
static void set_regs(pid_t p, regs_t *r) { CHECK(ptrace(PTRACE_SETREGS, p, 0, r) == 0); }
static uintptr_t pc_of(pid_t p) { regs_t r; get_regs(p, &r); return r.rip; }
static void set_pc(pid_t p, uintptr_t v) { regs_t r; get_regs(p, &r); r.rip = v; set_regs(p, &r); }
/* Where the stop after a breakpoint at `bp` leaves the pc: past it. */
#define AFTER_BP(bp) ((bp) + 1)
#elif defined(__aarch64__)
#include <asm/ptrace.h>
typedef struct user_pt_regs regs_t;
static void get_regs(pid_t p, regs_t *r) {
  struct iovec io = {r, sizeof *r};
  CHECK(ptrace(PTRACE_GETREGSET, p, (void *)NT_PRSTATUS, &io) == 0);
}
static void set_regs(pid_t p, regs_t *r) {
  struct iovec io = {r, sizeof *r};
  CHECK(ptrace(PTRACE_SETREGSET, p, (void *)NT_PRSTATUS, &io) == 0);
}
static uintptr_t pc_of(pid_t p) { regs_t r; get_regs(p, &r); return r.pc; }
static void set_pc(pid_t p, uintptr_t v) { regs_t r; get_regs(p, &r); r.pc = v; set_regs(p, &r); }
/* A brk stops with the pc at the brk itself. */
#define AFTER_BP(bp) (bp)
#else
#error "unsupported architecture"
#endif

static int wait_stop(pid_t p) {
  int st;
  CHECK(waitpid(p, &st, __WALL) == p);
  if (!WIFSTOPPED(st)) printf("not stopped: status %#x\n", st);
  CHECK(WIFSTOPPED(st));
  return WSTOPSIG(st);
}

static int wait_exit(pid_t p) {
  int st;
  CHECK(waitpid(p, &st, __WALL) == p);
  if (!WIFEXITED(st)) printf("not exited: status %#x\n", st);
  CHECK(WIFEXITED(st));
  return WEXITSTATUS(st);
}

static pid_t traced_child(void (*body)(void)) {
  pid_t c = fork();
  CHECK(c >= 0);
  if (!c) {
    if (ptrace(PTRACE_TRACEME, 0, 0, 0) != 0) _exit(90);
    raise(SIGSTOP);
    body();
    _exit(0);
  }
  CHECK(wait_stop(c) == SIGSTOP);
  return c;
}

/* Leave the process, from anywhere (no stack needed): what the tracer
 * redirects a faulted tracee to. */
__attribute__((noinline, noreturn)) static void leave42(void) {
#if defined(__x86_64__)
  __asm__ volatile("mov $231, %eax\n\tmov $42, %edi\n\tsyscall");
#elif defined(__aarch64__)
  __asm__ volatile("mov x8, #94\n\tmov x0, #42\n\tsvc #0");
#endif
  __builtin_unreachable();
}

/* 1. Single steps: each is a SIGTRAP stop (TRAP_TRACE) one instruction on. */
static void step_body(void) {
  volatile int n = 0;
  for (int i = 0; i < 1000; i++) n += i;
  _exit(n == 499500 ? 0 : 1);
}
static void test_single_step(void) {
  pid_t c = traced_child(step_body);
  uintptr_t last = pc_of(c);
  int moved = 0;
  for (int i = 0; i < 50; i++) {
    CHECK(ptrace(PTRACE_SINGLESTEP, c, 0, 0) == 0);
    CHECK(wait_stop(c) == SIGTRAP);
    siginfo_t si;
    CHECK(ptrace(PTRACE_GETSIGINFO, c, 0, &si) == 0);
    CHECK(si.si_signo == SIGTRAP);
    uintptr_t pc = pc_of(c);
    if (pc != last) moved++;
    last = pc;
  }
  CHECK(moved >= 45); /* (a step out of a syscall may land where it was) */
  CHECK(ptrace(PTRACE_CONT, c, 0, 0) == 0);
  CHECK(wait_exit(c) == 0);
  puts("single step: 50 SIGTRAP stops, each an instruction on");
}

/* 2. A breakpoint instruction in the tracee stops it with SIGTRAP; the
 * tracer resumes it without the signal. */
static void bp_body(void) {
#if defined(__x86_64__)
  __asm__ volatile("int3");
#elif defined(__aarch64__)
  __asm__ volatile("brk #0");
#endif
  _exit(7);
}
static void test_breakpoint_insn(void) {
  pid_t c = traced_child(bp_body);
  CHECK(ptrace(PTRACE_CONT, c, 0, 0) == 0);
  CHECK(wait_stop(c) == SIGTRAP);
  uintptr_t pc = pc_of(c);
  errno = 0;
  /* The breakpoint is the instruction the stop is at (arm64) or just past. */
#if defined(__x86_64__)
  long insn = ptrace(PTRACE_PEEKTEXT, c, (void *)(pc - 1), 0);
  CHECK(errno == 0 && (insn & 0xff) == 0xcc);
#elif defined(__aarch64__)
  long insn = ptrace(PTRACE_PEEKTEXT, c, (void *)pc, 0);
  CHECK(errno == 0 && (insn & 0xffffffffL) == 0xd4200000L);
#endif
#if defined(__aarch64__)
  set_pc(c, pc + 4); /* step over the brk */
#endif
  CHECK(ptrace(PTRACE_CONT, c, 0, 0) == 0);
  CHECK(wait_exit(c) == 7);
  puts("breakpoint instruction: a SIGTRAP stop at it, then on");
}

/* 3. A breakpoint the tracer writes into the tracee's code. */
__attribute__((noinline)) static int target(int x) {
  __asm__ volatile("" ::: "memory");
  return x * 3;
}
static void poke_body(void) { _exit(target(5)); }
static void test_poked_breakpoint(void) {
  pid_t c = traced_child(poke_body);
  uintptr_t fn = (uintptr_t)&target;
  errno = 0;
  long orig = ptrace(PTRACE_PEEKTEXT, c, (void *)fn, 0);
  CHECK(errno == 0);
#if defined(__x86_64__)
  long patched = (orig & ~0xffL) | 0xcc; /* int3 */
#elif defined(__aarch64__)
  long patched = (orig & ~0xffffffffL) | 0xd4200000L; /* brk #0 */
#endif
  CHECK(ptrace(PTRACE_POKETEXT, c, (void *)fn, (void *)patched) == 0);
  CHECK(ptrace(PTRACE_PEEKTEXT, c, (void *)fn, 0) == patched);
  CHECK(ptrace(PTRACE_CONT, c, 0, 0) == 0);
  CHECK(wait_stop(c) == SIGTRAP);
  CHECK(pc_of(c) == AFTER_BP(fn));
  /* Put the code back and run the function from its start. */
  CHECK(ptrace(PTRACE_POKETEXT, c, (void *)fn, (void *)orig) == 0);
  set_pc(c, fn);
  CHECK(ptrace(PTRACE_CONT, c, 0, 0) == 0);
  CHECK(wait_exit(c) == 15);
  puts("poked breakpoint: hit, restored, and the function ran");
}

/* 4. A fault stops the tracee before any signal is delivered; the tracer
 * sees it, moves the pc, and the tracee goes on without the signal. */
static void fault_body(void) {
  *(volatile int *)0 = 1;
  _exit(1);
}
static void test_fault_stop(void) {
  pid_t c = traced_child(fault_body);
  CHECK(ptrace(PTRACE_CONT, c, 0, 0) == 0);
  CHECK(wait_stop(c) == SIGSEGV);
  siginfo_t si;
  CHECK(ptrace(PTRACE_GETSIGINFO, c, 0, &si) == 0);
  CHECK(si.si_signo == SIGSEGV && si.si_code == SEGV_MAPERR && si.si_addr == 0);
  set_pc(c, (uintptr_t)&leave42);
  CHECK(ptrace(PTRACE_CONT, c, 0, 0) == 0);
  CHECK(wait_exit(c) == 42);
  /* And a fault passed on with CONT kills as it would untraced. */
  c = traced_child(fault_body);
  CHECK(ptrace(PTRACE_CONT, c, 0, 0) == 0);
  CHECK(wait_stop(c) == SIGSEGV);
  CHECK(ptrace(PTRACE_CONT, c, 0, SIGSEGV) == 0);
  int st;
  CHECK(waitpid(c, &st, __WALL) == c && WIFSIGNALED(st) && WTERMSIG(st) == SIGSEGV);
  puts("fault: SIGSEGV stopped the tracee, which the tracer repaired");
}

/* 5. The floating-point registers: the tracee loads a pattern into its first
 * vector register and traps. */
static void fp_body(void) {
#if defined(__x86_64__)
  static const uint64_t pat[2] __attribute__((aligned(16))) = {0x1122334455667788ULL, 0x99aabbccddeeff00ULL};
  __asm__ volatile("movdqa %0, %%xmm0\n\tint3" : : "m"(pat) : "xmm0");
#elif defined(__aarch64__)
  static const uint64_t pat[2] __attribute__((aligned(16))) = {0x1122334455667788ULL, 0x99aabbccddeeff00ULL};
  __asm__ volatile("ldr q0, [%0]\n\tbrk #0" : : "r"(pat) : "v0");
#endif
  _exit(0);
}
static void test_fp_regs(void) {
  pid_t c = traced_child(fp_body);
  CHECK(ptrace(PTRACE_CONT, c, 0, 0) == 0);
  CHECK(wait_stop(c) == SIGTRAP);
  uint64_t lo, hi;
#if defined(__x86_64__)
  struct user_fpregs_struct fp;
  memset(&fp, 0, sizeof fp);
  CHECK(ptrace(PTRACE_GETFPREGS, c, 0, &fp) == 0);
  memcpy(&lo, &fp.xmm_space[0], 8);
  memcpy(&hi, &fp.xmm_space[2], 8);
  CHECK(fp.cwd == 0x37f); /* the x87 control word as the ABI leaves it */
#elif defined(__aarch64__)
  struct user_fpsimd_state fp;
  memset(&fp, 0, sizeof fp);
  struct iovec io = {&fp, sizeof fp};
  CHECK(ptrace(PTRACE_GETREGSET, c, (void *)NT_PRFPREG, &io) == 0);
  memcpy(&lo, &fp.vregs[0], 8);
  memcpy(&hi, (char *)&fp.vregs[0] + 8, 8);
  set_pc(c, pc_of(c) + 4);
#endif
  CHECK(lo == 0x1122334455667788ULL && hi == 0x99aabbccddeeff00ULL);
  CHECK(ptrace(PTRACE_CONT, c, 0, 0) == 0);
  CHECK(wait_exit(c) == 0);
  puts("fp registers: the tracee's vector register, as it set it");
}

static void run_all(void) {
  setvbuf(stdout, 0, _IONBF, 0);
  test_single_step();
  test_breakpoint_insn();
  test_poked_breakpoint();
  test_fault_stop();
  test_fp_regs();
  puts("PASS");
}

int main(void) {
  /* As pid 1 (the VM's init), run the cases in a child: they look for
   * ordinary process behaviour. */
  if (getpid() == 1) {
    pid_t c = fork();
    CHECK(c >= 0);
    if (!c) {
      run_all();
      return 0;
    }
    int st;
    CHECK(waitpid(c, &st, 0) == c);
    return WIFEXITED(st) ? WEXITSTATUS(st) : 1;
  }
  run_all();
  return 0;
}
