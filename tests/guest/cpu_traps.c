/* CPU exceptions in user code arrive as the signals Linux sends for them, with
 * the right si_code and si_addr and the registers intact, and execution can
 * go on afterwards: int3 (and the brk of arm64), an undefined instruction, a
 * NULL store, an integer divide by zero, and a privileged instruction. */
#define _GNU_SOURCE
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ucontext.h>

static sigjmp_buf jb;
static volatile int got_sig, got_code;
static void *volatile got_addr;
static volatile uintptr_t got_pc;

static void on_fault(int sig, siginfo_t *si, void *uc) {
  got_sig = sig;
  got_code = si->si_code;
  got_addr = si->si_addr;
#if defined(__x86_64__)
  got_pc = ((ucontext_t *)uc)->uc_mcontext.gregs[REG_RIP];
#elif defined(__aarch64__)
  got_pc = ((ucontext_t *)uc)->uc_mcontext.pc;
#endif
  siglongjmp(jb, 1);
}

/* A trap (int3, brk) returns from its handler to the next instruction. */
static volatile int traps;
static void on_trap(int sig, siginfo_t *si, void *uc) {
  (void)si;
  got_sig = sig;
  got_code = si->si_code;
#if defined(__x86_64__)
  got_pc = ((ucontext_t *)uc)->uc_mcontext.gregs[REG_RIP];
#elif defined(__aarch64__)
  /* brk is reported at itself; step over it. */
  got_pc = ((ucontext_t *)uc)->uc_mcontext.pc;
  ((ucontext_t *)uc)->uc_mcontext.pc += 4;
#endif
  traps++;
}

static int fails;
static void expect(int ok, const char *what) {
  printf("%s: %s (sig %d, code %d)\n", ok ? "ok" : "FAIL", what, got_sig, got_code);
  if (!ok) fails++;
}

static void reset(void) { got_sig = got_code = 0; got_addr = 0; got_pc = 0; }

int main(void) {
  struct sigaction sa;
  memset(&sa, 0, sizeof sa);
  sa.sa_flags = SA_SIGINFO | SA_NODEFER;
  sa.sa_sigaction = on_fault;
  sigaction(SIGSEGV, &sa, 0);
  sigaction(SIGILL, &sa, 0);
  sigaction(SIGFPE, &sa, 0);
  sigaction(SIGBUS, &sa, 0);
  sa.sa_sigaction = on_trap;
  sigaction(SIGTRAP, &sa, 0);

  /* A breakpoint: SIGTRAP, then the program carries on. */
  reset();
  uintptr_t at;
#if defined(__x86_64__)
  __asm__ volatile("lea 1f(%%rip), %0\n\tint3\n1:" : "=r"(at));
  /* Linux: si_code SI_KERNEL (0x80), the pc just past the int3. */
  expect(traps == 1 && got_sig == SIGTRAP && got_code == 0x80 && got_pc == at, "int3 is SIGTRAP past the int3, and returns");
#elif defined(__aarch64__)
  __asm__ volatile("adr %0, 1f\n1:\tbrk #0" : "=r"(at));
  /* Linux: si_code TRAP_BRKPT (1), the pc at the brk. */
  expect(traps == 1 && got_sig == SIGTRAP && got_code == 1 && got_pc == at, "brk is SIGTRAP at the brk, and returns");
#endif
  int again = traps;
#if defined(__x86_64__)
  __asm__ volatile("int3");
#elif defined(__aarch64__)
  __asm__ volatile("brk #0");
#endif
  expect(traps == again + 1, "a second breakpoint traps again");

  /* An undefined instruction: SIGILL ILL_ILLOPN (x86) / ILL_ILLOPC (arm64). */
  reset();
  if (!sigsetjmp(jb, 1)) {
#if defined(__x86_64__)
    __asm__ volatile("ud2");
#elif defined(__aarch64__)
    __asm__ volatile(".inst 0x00000000"); /* udf #0 */
#endif
  }
#if defined(__x86_64__)
  expect(got_sig == SIGILL && got_code == 2 && got_addr == (void *)got_pc, "ud2 is SIGILL ILL_ILLOPN at the instruction");
#elif defined(__aarch64__)
  expect(got_sig == SIGILL && got_code == 1 && got_addr == (void *)got_pc, "udf is SIGILL ILL_ILLOPC at the instruction");
#endif

  /* A NULL store: SIGSEGV SEGV_MAPERR at address 0. */
  reset();
  if (!sigsetjmp(jb, 1)) *(volatile int *)0 = 1;
  expect(got_sig == SIGSEGV && got_code == 1 && got_addr == 0, "a NULL store is SIGSEGV SEGV_MAPERR at 0");

  /* An integer divide by zero: SIGFPE FPE_INTDIV (x86; arm64 has no trap). */
#if defined(__x86_64__)
  reset();
  if (!sigsetjmp(jb, 1)) {
    long q, r;
    __asm__ volatile("xor %%edx, %%edx\n\tmov $1, %%eax\n\txor %%ecx, %%ecx\n\tdivl %%ecx" : "=a"(q), "=d"(r) : : "rcx");
  }
  expect(got_sig == SIGFPE && got_code == 1 && got_addr == (void *)got_pc, "a divide by zero is SIGFPE FPE_INTDIV at the instruction");

  /* A privileged instruction: SIGSEGV SI_KERNEL (#GP). */
  reset();
  if (!sigsetjmp(jb, 1)) __asm__ volatile("hlt");
  expect(got_sig == SIGSEGV && got_code == 0x80, "hlt is SIGSEGV SI_KERNEL");
#endif

  /* And the program is still whole. */
  volatile int x = 6;
  expect(x * 7 == 42, "the program runs on");
  if (fails) return 1;
  puts("PASS");
  return 0;
}
