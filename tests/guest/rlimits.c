#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
static int fails;
#define CHECK(c, ...) do { int ok_ = (c); printf("%s: ", ok_ ? "ok  " : "FAIL"); printf(__VA_ARGS__); puts(""); fflush(stdout); if (!ok_) fails++; } while (0)
static volatile sig_atomic_t got;
static void h(int s) { got = s; }
static int child_status(void (*f)(void)) { fflush(stdout); pid_t p = fork(); if (!p) { f(); _exit(0); } int st; waitpid(p, &st, 0); return st; }
static void *set_in_thread(void *a) { struct rlimit r = {123, 456}; setrlimit(RLIMIT_CORE, &r); return a; }
static void as_child(void) { struct rlimit r = {256 << 20, 256 << 20}; setrlimit(RLIMIT_AS, &r); void *p = mmap(0, 512UL << 20, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0); _exit(p == MAP_FAILED && errno == ENOMEM ? 7 : 1); }
static volatile char sink;
static void so(int n) { volatile char b[65536]; b[0] = n; sink = b[0]; so(n + 1); sink = b[1]; }
static void stack_child(void) { struct rlimit r = {1 << 20, 1 << 20}; setrlimit(RLIMIT_STACK, &r); so(0); _exit(1); }
static void cpu_child(void) { struct rlimit r = {1, RLIM_INFINITY}; setrlimit(RLIMIT_CPU, &r); signal(SIGXCPU, h); time_t s = time(0); while (!got && time(0) - s < 8) ; _exit(got == SIGXCPU ? 8 : 1); }
static void cpu_kill_child(void) { struct rlimit r = {1, 1}; setrlimit(RLIMIT_CPU, &r); time_t s = time(0); while (time(0) - s < 8) ; _exit(1); }
static void fsize_child(void) { struct rlimit r = {10, 10}; setrlimit(RLIMIT_FSIZE, &r); signal(SIGXFSZ, h); int f = open("/tmp/rl", O_CREAT | O_TRUNC | O_RDWR, 0600); long a = write(f, "012345678901234567890", 20); got = 0; errno = 0; long b = write(f, "x", 1); int e = errno; int sawsig = got == SIGXFSZ; int t = ftruncate(f, 100); int te = errno; _exit(a == 10 && b < 0 && e == EFBIG && sawsig && t < 0 && te == EFBIG ? 9 : 1); }
static void fsize_kill_child(void) { struct rlimit r = {10, 10}; setrlimit(RLIMIT_FSIZE, &r); int f = open("/tmp/rl2", O_CREAT | O_TRUNC | O_RDWR, 0600); write(f, "012345678901234567890", 20); write(f, "x", 1); _exit(1); }
static void nproc_child(void) { if (setresuid(1000, 1000, 1000)) _exit(2); struct rlimit r = {8, 8}; setrlimit(RLIMIT_NPROC, &r); int n = 0; for (;;) { pid_t p = fork(); if (p < 0) _exit(errno == EAGAIN && n > 0 && n <= 8 ? 10 : 3); if (!p) { pause(); _exit(0); } n++; if (n > 64) _exit(4); } }
static void priv_child(void) { if (setresuid(1000, 1000, 1000)) _exit(2); struct rlimit r = {100, 200}; if (setrlimit(RLIMIT_NOFILE, &r)) _exit(3); r.rlim_max = 300; r.rlim_cur = 100; if (setrlimit(RLIMIT_NOFILE, &r) == 0 || errno != EPERM) _exit(4); r.rlim_max = 150; if (setrlimit(RLIMIT_NOFILE, &r)) _exit(5); _exit(11); }
static void data_child(void) { struct rlimit r = {1 << 20, 1 << 20}; setrlimit(RLIMIT_DATA, &r); void *o = sbrk(0); _exit(sbrk(8 << 20) == (void *)-1 && errno == ENOMEM ? 12 : 1); }
int main(void) {
  struct rlimit r, q; CHECK(getrlimit(99, &r) < 0 && errno == EINVAL, "unknown resource EINVAL");
  getrlimit(RLIMIT_CORE, &r); q = r; q.rlim_cur = 5; q.rlim_max = 2; CHECK(setrlimit(RLIMIT_CORE, &q) < 0 && errno == EINVAL, "soft above hard EINVAL");
  r.rlim_cur = 77; r.rlim_max = 88; CHECK(!setrlimit(RLIMIT_CORE, &r), "setrlimit"); getrlimit(RLIMIT_CORE, &q); CHECK(q.rlim_cur == 77 && q.rlim_max == 88, "reads back %lu/%lu", q.rlim_cur, q.rlim_max);
  pthread_t t; pthread_create(&t, 0, set_in_thread, 0); pthread_join(t, 0); getrlimit(RLIMIT_CORE, &q); CHECK(q.rlim_cur == 123 && q.rlim_max == 456, "a thread's setrlimit applies to the process");
  fflush(stdout); pid_t c = fork(); if (!c) { struct rlimit s; getrlimit(RLIMIT_CORE, &s); s.rlim_cur = 1; int inh = s.rlim_max == 456; setrlimit(RLIMIT_CORE, &s); _exit(inh ? 21 : 1); } int st; waitpid(c, &st, 0); getrlimit(RLIMIT_CORE, &q); CHECK(WEXITSTATUS(st) == 21 && q.rlim_cur == 123, "fork inherits; child's change stays in the child");
  c = fork(); if (!c) { pause(); _exit(0); } r.rlim_cur = 9; r.rlim_max = 99; CHECK(!prlimit(c, RLIMIT_CORE, &r, 0), "prlimit on another process"); struct rlimit got2; CHECK(!prlimit(c, RLIMIT_CORE, 0, &got2) && got2.rlim_cur == 9, "reads it back"); getrlimit(RLIMIT_CORE, &q); CHECK(q.rlim_cur == 123, "doesn't touch the caller"); kill(c, 9); waitpid(c, 0, 0); CHECK(prlimit(c, RLIMIT_CORE, 0, &got2) < 0 && errno == ESRCH, "gone: ESRCH");
  st = child_status(as_child); CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 7, "RLIMIT_AS: mmap ENOMEM (%#x)", st);
  st = child_status(data_child); CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 12, "RLIMIT_DATA: brk fails (%#x)", st);
  st = child_status(stack_child); CHECK(WIFSIGNALED(st) && WTERMSIG(st) == SIGSEGV, "RLIMIT_STACK: overflow SIGSEGV (%#x)", st);
  st = child_status(fsize_child); CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 9, "RLIMIT_FSIZE: short write, then EFBIG + SIGXFSZ (%#x)", st);
  st = child_status(fsize_kill_child); CHECK(WIFSIGNALED(st) && WTERMSIG(st) == SIGXFSZ, "RLIMIT_FSIZE: SIGXFSZ kills by default (%#x)", st);
  st = child_status(cpu_child); CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 8, "RLIMIT_CPU soft: SIGXCPU (%#x)", st);
  st = child_status(cpu_kill_child); CHECK(WIFSIGNALED(st) && WTERMSIG(st) == SIGKILL, "RLIMIT_CPU hard: SIGKILL (%#x)", st);
  st = child_status(nproc_child); CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 10, "RLIMIT_NPROC: fork EAGAIN for non-root (%#x)", st);
  st = child_status(priv_child); CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 11, "only root raises a hard limit (%#x)", st);
  puts(fails ? "SOME FAILED" : "PASS"); return fails != 0;
}
