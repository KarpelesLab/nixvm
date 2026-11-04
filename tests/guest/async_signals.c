#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <ucontext.h>
#include <unistd.h>
static int fails;
#define CHECK(c, ...) do { int ok_ = (c); printf("%s: ", ok_ ? "ok  " : "FAIL"); printf(__VA_ARGS__); puts(""); fflush(stdout); if (!ok_) fails++; } while (0)
static volatile sig_atomic_t ticks; static volatile unsigned long spin;
static void h(int s) { ticks++; }
int main(void) {
  pid_t p = fork(); if (!p) { for (;;); } usleep(100000); kill(p, SIGKILL); int st = 0; waitpid(p, &st, 0); CHECK(WIFSIGNALED(st) && WTERMSIG(st) == SIGKILL, "SIGKILL stops a spinner (%#x)", st);
  p = fork(); if (!p) { for (;;); } usleep(100000); kill(p, SIGTERM); waitpid(p, &st, 0); CHECK(WIFSIGNALED(st) && WTERMSIG(st) == SIGTERM, "SIGTERM too (%#x)", st);
  /* A handler runs on a syscall-free loop, and the loop's registers survive it. */
  signal(SIGALRM, h); ualarm(20000, 20000); unsigned long a = 0, b = 1, c = 2, d = 3; time_t s = time(0);
  while (ticks < 5 && time(0) - s < 4) { for (int i = 0; i < 1000000; i++) { a += b; c ^= a; d += c; b += 1; } }
  ualarm(0, 0); CHECK(ticks >= 5, "SIGALRM handler reaches a compute loop (%d)", ticks);
  /* Checksum computed with interruptions equals one without. */
  /* As long as it takes to be interrupted 20 times, so a slow interpreter isn't stuck on a fixed count. */
  unsigned long x = 1, y = 0, n = 0; ticks = 0; ualarm(2000, 2000); s = time(0);
  while (ticks < 20 && time(0) - s < 30) { for (int i = 0; i < 100000; i++) { x = x * 6364136223846793005UL + 1442695040888963407UL; y ^= x; } n += 100000; }
  ualarm(0, 0); unsigned long x2 = 1, y2 = 0; for (unsigned long i = 0; i < n; i++) { x2 = x2 * 6364136223846793005UL + 1442695040888963407UL; y2 ^= x2; } CHECK(y == y2 && ticks >= 20, "state survives %d async deliveries over %lu steps", ticks, n);
  /* Stop and continue a spinner. */
  p = fork(); if (!p) { for (;;); } usleep(50000); kill(p, SIGSTOP); waitpid(p, &st, WUNTRACED); CHECK(WIFSTOPPED(st), "SIGSTOP stops a spinner (%#x)", st); kill(p, SIGCONT); usleep(50000); kill(p, SIGKILL); waitpid(p, &st, 0); CHECK(WIFSIGNALED(st), "and it can be continued, then killed");
  puts(fails ? "SOME FAILED" : "PASS"); return fails != 0;
}
