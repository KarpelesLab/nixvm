#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mqueue.h>
#include <sched.h>
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
/* The rest run as a user of their own, so on Linux nothing else of theirs is counted. */
static void become_user(void) { if (setresuid(4321, 4321, 4321)) _exit(2); }
static void memlock_child(void) {
  long pg = sysconf(_SC_PAGESIZE); struct rlimit r = {2 * pg, 2 * pg}; setrlimit(RLIMIT_MEMLOCK, &r); become_user();
  char *m = mmap(0, 4 * pg, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mlock(m, pg) || mlock(m + pg, pg)) _exit(3);
  if (mlock(m, 2 * pg)) _exit(4);                                  /* already locked: costs nothing */
  if (mlock(m + 2 * pg, pg) == 0 || errno != ENOMEM) _exit(5);
  if (munlock(m, pg) || mlock(m + 2 * pg, pg)) _exit(6);           /* unlocking frees room */
  if (mmap(0, 4 * pg, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS | MAP_LOCKED, -1, 0) != MAP_FAILED || errno != EAGAIN) _exit(7);
  if (mlockall(MCL_CURRENT) == 0 || errno != ENOMEM) _exit(8);
  munmap(m, 4 * pg); if (mlock(mmap(0, 2 * pg, PROT_READ, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0), 2 * pg)) _exit(9); /* munmap unlocks */
  r.rlim_cur = 0; setrlimit(RLIMIT_MEMLOCK, &r); if (mlock(m, pg) == 0 || errno != EPERM) _exit(10);
  _exit(13);
}
static void nice_child(void) {
  struct rlimit r = {30, 30}; setrlimit(RLIMIT_NICE, &r); setpriority(PRIO_PROCESS, 0, 5); become_user();
  if (setpriority(PRIO_PROCESS, 0, -5)) _exit(3);                  /* 20 - -5 = 25 <= 30 */
  if (setpriority(PRIO_PROCESS, 0, -15) == 0 || errno != EACCES) _exit(4);
  errno = 0; if (getpriority(PRIO_PROCESS, 0) != -5 || errno) _exit(5);
  if (setpriority(PRIO_PROCESS, 0, 10)) _exit(6);                  /* raising it is always allowed */
  r.rlim_cur = r.rlim_max = 0; setrlimit(RLIMIT_NICE, &r);
  if (setpriority(PRIO_PROCESS, 0, 9) == 0 || errno != EACCES) _exit(7);
  if (setpriority(PRIO_PROCESS, 1, 19) == 0 || errno != EPERM) _exit(8); /* not ours */
  if (setpriority(99, 0, 0) == 0 || errno != EINVAL) _exit(9);
  _exit(14);
}
static void rtprio_child(void) {
  struct rlimit r = {10, 10}; setrlimit(RLIMIT_RTPRIO, &r); become_user(); struct sched_param sp = {5};
  if (sched_setscheduler(0, SCHED_FIFO, &sp)) _exit(3);
  sp.sched_priority = 20; if (sched_setscheduler(0, SCHED_FIFO, &sp) == 0 || errno != EPERM) _exit(4);
  if (sched_setparam(0, &sp) == 0 || errno != EPERM) _exit(5);
  sp.sched_priority = 0; if (sched_setscheduler(0, SCHED_OTHER, &sp)) _exit(6);
  r.rlim_cur = r.rlim_max = 0; setrlimit(RLIMIT_RTPRIO, &r); sp.sched_priority = 1;
  if (sched_setscheduler(0, SCHED_RR, &sp) == 0 || errno != EPERM) _exit(7);
  if (sched_setscheduler(0, 77, &sp) == 0 || errno != EINVAL) _exit(8);
  _exit(15);
}
static void sigpending_child(void) {
  struct rlimit r = {3, 3}; setrlimit(RLIMIT_SIGPENDING, &r); become_user();
  sigset_t s; sigemptyset(&s); sigaddset(&s, SIGRTMIN); sigprocmask(SIG_BLOCK, &s, 0); union sigval v = {0};
  for (int i = 0; i < 3; i++) if (sigqueue(getpid(), SIGRTMIN, v)) _exit(3);
  if (sigqueue(getpid(), SIGRTMIN, v) == 0 || errno != EAGAIN) _exit(4);
  if (kill(getpid(), SIGRTMIN)) _exit(5);                          /* kill isn't refused */
  siginfo_t si; struct timespec z = {0, 0}; if (sigtimedwait(&s, &si, &z) != SIGRTMIN) _exit(6);
  if (sigqueue(getpid(), SIGRTMIN, v)) _exit(7);                   /* room again */
  _exit(16);
}
static void msgqueue_child(void) {
  struct mq_attr a = {0}; a.mq_maxmsg = 10; a.mq_msgsize = 1000; struct rlimit r = {25000, 25000}; setrlimit(RLIMIT_MSGQUEUE, &r); become_user();
  mqd_t q1 = mq_open("/rl-q1", O_CREAT | O_RDWR, 0600, &a), q2 = mq_open("/rl-q2", O_CREAT | O_RDWR, 0600, &a);
  if (q1 < 0 || q2 < 0) _exit(3);
  if (mq_open("/rl-q3", O_CREAT | O_RDWR, 0600, &a) >= 0 || errno != EMFILE) _exit(4);
  if (mq_open("/rl-q1", O_RDWR) < 0) _exit(5);                     /* opening one isn't charged */
  mq_unlink("/rl-q2"); mq_close(q2);
  if (mq_open("/rl-q3", O_CREAT | O_RDWR, 0600, &a) < 0) _exit(6); /* the freed queue's room */
  mq_unlink("/rl-q1"); mq_unlink("/rl-q3");
  _exit(17);
}
static void proc_limits_child(void) {
  struct rlimit r = {1234, 5678}; setrlimit(RLIMIT_MEMLOCK, &r); char b[4096] = {0}; int f = open("/proc/self/limits", O_RDONLY); read(f, b, sizeof b - 1);
  char *l = strstr(b, "Max locked memory"); unsigned long s = 0, h = 0; if (!l || sscanf(l + 25, "%lu %lu", &s, &h) != 2) _exit(3);
  if (strncmp(b, "Limit                     Soft Limit           Hard Limit           Units     \n", 79)) _exit(4);
  _exit(s == 1234 && h == 5678 ? 18 : 5);
}
static double cpu_s(clockid_t c) { struct timespec t; clock_gettime(c, &t); return t.tv_sec + t.tv_nsec / 1e9; }
static void spin_cpu(double s) { double end = cpu_s(CLOCK_THREAD_CPUTIME_ID) + s; while (cpu_s(CLOCK_THREAD_CPUTIME_ID) < end && !got) ; }
static void go_fifo(void) { struct sched_param sp = {1}; if (sched_setscheduler(0, SCHED_FIFO, &sp)) _exit(2); }
static void rttime_child(void) {
  struct rlimit r = {100000, RLIM_INFINITY}; setrlimit(RLIMIT_RTTIME, &r); signal(SIGXCPU, h); go_fifo();
  time_t s = time(0); while (!got && time(0) - s < 5) ;
  getrlimit(RLIMIT_RTTIME, &r); _exit(got == SIGXCPU && r.rlim_cur == 1100000 ? 19 : 1); /* the soft limit moves on a second */
}
static void rttime_kill_child(void) { struct rlimit r = {100000, 300000}; setrlimit(RLIMIT_RTTIME, &r); signal(SIGXCPU, SIG_IGN); go_fifo(); time_t s = time(0); while (time(0) - s < 5) ; _exit(1); }
static void rttime_sleep_child(void) {
  struct rlimit r = {200000, 400000}; setrlimit(RLIMIT_RTTIME, &r); signal(SIGXCPU, h); go_fifo();
  for (int i = 0; i < 12 && !got; i++) { spin_cpu(0.05); usleep(1000); } /* 600 ms of CPU, never 200 ms at a stretch */
  _exit(got ? 3 : 20);
}
static void rttime_fair_child(void) { struct rlimit r = {100000, 200000}; setrlimit(RLIMIT_RTTIME, &r); spin_cpu(0.4); _exit(21); } /* only real-time tasks count */
static void fork_cpu_child(void) { spin_cpu(0.3); fflush(stdout); pid_t c = fork(); if (!c) _exit(cpu_s(CLOCK_PROCESS_CPUTIME_ID) < 0.1 && cpu_s(CLOCK_THREAD_CPUTIME_ID) < 0.1 ? 22 : 1); int st; waitpid(c, &st, 0); _exit(WEXITSTATUS(st)); }
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
  st = child_status(memlock_child); CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 13, "RLIMIT_MEMLOCK: mlock, MAP_LOCKED and mlockall (%#x)", st);
  st = child_status(nice_child); CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 14, "RLIMIT_NICE bounds lowering the nice value (%#x)", st);
  st = child_status(rtprio_child); CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 15, "RLIMIT_RTPRIO bounds real-time priority (%#x)", st);
  st = child_status(rttime_child); CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 19, "RLIMIT_RTTIME soft: SIGXCPU, then a second more (%#x)", st);
  st = child_status(rttime_kill_child); CHECK(WIFSIGNALED(st) && WTERMSIG(st) == SIGKILL, "RLIMIT_RTTIME hard: SIGKILL (%#x)", st);
  st = child_status(rttime_sleep_child); CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 20, "RLIMIT_RTTIME: blocking starts the count over (%#x)", st);
  st = child_status(rttime_fair_child); CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 21, "RLIMIT_RTTIME: a fair-class task isn't limited (%#x)", st);
  st = child_status(fork_cpu_child); CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 22, "a fork child's CPU time starts at zero (%#x)", st);
  st = child_status(sigpending_child); CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 16, "RLIMIT_SIGPENDING: sigqueue EAGAIN (%#x)", st);
  st = child_status(msgqueue_child); CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 17, "RLIMIT_MSGQUEUE: mq_open EMFILE, freed by unlink and close (%#x)", st);
  st = child_status(proc_limits_child); CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 18, "/proc/self/limits shows the limits (%#x)", st);
  puts(fails ? "SOME FAILED" : "PASS"); return fails != 0;
}
