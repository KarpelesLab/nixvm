/* What Wine asks of the kernel beyond what libc does (each check named after
 * the gap it once was):
 *  - arch_prctl(ARCH_SET_GS): x86-64 Wine keeps the TEB at %gs:0x30. The GS
 *    base is per thread, read back by ARCH_GET_GS, kept across a signal
 *    handler, inherited by a new thread and a fork child, and cleared by
 *    exec (checked by re-executing ourselves).
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

static int failures;
#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("FAIL " __VA_ARGS__);                                             \
      printf("\n");                                                            \
      failures++;                                                              \
    }                                                                          \
  } while (0)

#if defined(__x86_64__)
#include <asm/prctl.h>

static uint64_t teb[8] = {0x1111, 0x2222, 0x3333};

static uint64_t gs_read8(void) {
  uint64_t v;
  __asm__ volatile("movq %%gs:8, %0" : "=r"(v));
  return v;
}

static uint64_t get_gs(void) {
  uint64_t base = 0xdead;
  if (syscall(SYS_arch_prctl, ARCH_GET_GS, &base) != 0)
    return 0xbad;
  return base;
}

static volatile uint64_t in_handler;
static void on_usr1(int sig) {
  (void)sig;
  in_handler = gs_read8();
}

static void *thread_main(void *arg) {
  (void)arg;
  uint64_t inherited = get_gs();
  static uint64_t own[2] = {0, 0x7777};
  syscall(SYS_arch_prctl, ARCH_SET_GS, own);
  return (void *)(uintptr_t)(inherited == (uintptr_t)teb && gs_read8() == 0x7777);
}

static void gs_base(const char *self) {
  CHECK(syscall(SYS_arch_prctl, ARCH_SET_GS, teb) == 0, "ARCH_SET_GS refused");
  CHECK(get_gs() == (uintptr_t)teb, "ARCH_GET_GS gave %#lx", (long)get_gs());
  CHECK(gs_read8() == 0x2222, "%%gs:8 read %#lx", (long)gs_read8());

  signal(SIGUSR1, on_usr1);
  raise(SIGUSR1);
  CHECK(in_handler == 0x2222, "GS base in a handler: %%gs:8 = %#lx", (long)in_handler);
  CHECK(gs_read8() == 0x2222, "GS base lost across a handler");

  pthread_t t;
  void *ok = 0;
  pthread_create(&t, 0, thread_main, 0);
  pthread_join(t, &ok);
  CHECK(ok, "a thread doesn't inherit (or can't set) its GS base");
  CHECK(gs_read8() == 0x2222, "a thread's ARCH_SET_GS changed ours");

  pid_t pid = fork();
  if (pid == 0)
    _exit(gs_read8() == 0x2222 ? 0 : 1);
  int st = 0;
  waitpid(pid, &st, 0);
  CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "a fork child lost the GS base");

  CHECK(syscall(SYS_arch_prctl, ARCH_SET_GS, 0x8000000000000000ul) == -1,
        "a non-canonical GS base was accepted");

  pid = fork();
  if (pid == 0) {
    execl(self, self, "gs-after-exec", (char *)0);
    _exit(2);
  }
  waitpid(pid, &st, 0);
  CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "exec kept the GS base (%#x)", st);
}
#endif

int main(int argc, char **argv) {
#if defined(__x86_64__)
  if (argc > 1 && !strcmp(argv[1], "gs-after-exec"))
    return get_gs() == 0 ? 0 : 1;
  /* In the VM the program is /wine_abi, which /proc/self/exe may not name. */
  gs_base(access("/wine_abi", X_OK) == 0 ? "/wine_abi" : "/proc/self/exe");
#else
  (void)argc;
  (void)argv;
#endif
  if (failures)
    return 1;
  printf("PASS\n");
  return 0;
}
