/* A child belongs to its whole process, not only to the thread that forked
 * it: another thread waits for it (.NET's Process.WaitForExit reaps from
 * its signal-handling thread, with waitid(P_ALL, WNOHANG|WNOWAIT) first),
 * and a child whose forking thread exits stays the process's; its getppid()
 * is the process. And the CPU time a reaped child reports (wait4's rusage,
 * then RUSAGE_CHILDREN) is its whole process's, the threads that exited
 * before it included: `time wine 7za ...` printed only the main thread's. */
#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d, errno %d)\n", #c, __LINE__, errno); exit(1); } } while (0)

static volatile pid_t forked;
static volatile int reaped;

static void *forker(void *arg) {
  forked = fork();
  if (forked == 0) {
    usleep(100000);
    _exit(getppid() == (pid_t)(long)arg ? 7 : 8);
  }
  return 0;
}

/* Forks, and lives on until the main thread has reaped the child. */
static void *forker_waits(void *arg) {
  forker(arg);
  while (!reaped) usleep(1000);
  return 0;
}

static void *spawn_and_exit(void *arg) {
  forker(arg);
  return 0; /* the forking thread is gone before its child exits */
}

/* Use `ms` of this thread's CPU time. */
static void *burn(void *arg) {
  struct timespec t;
  long ms = (long)arg;
  do {
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
  } while (t.tv_sec * 1000 + t.tv_nsec / 1000000 < ms);
  return 0;
}

static long tv_ms(struct timeval a, struct timeval b) {
  return (a.tv_sec + b.tv_sec) * 1000 + (a.tv_usec + b.tv_usec) / 1000;
}

int main(void) {
  pid_t me = getpid();
  pthread_t t;

  /* A thread forks; the main thread waits while it lives. */
  CHECK(pthread_create(&t, 0, forker_waits, (void *)(long)me) == 0);
  while (!forked) usleep(1000);
  siginfo_t si = {0};
  CHECK(waitid(P_ALL, 0, &si, WEXITED | WNOWAIT) == 0 && si.si_pid == forked);
  int status;
  CHECK(waitpid(forked, &status, 0) == forked);
  CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 7);
  reaped = 1;
  CHECK(pthread_join(t, 0) == 0);
  forked = 0;
  puts("a child another thread forked is the process's to wait for");

  /* The forking thread exited long before: still the process's child. */
  CHECK(pthread_create(&t, 0, spawn_and_exit, (void *)(long)me) == 0);
  CHECK(pthread_join(t, 0) == 0);
  CHECK(waitpid(forked, &status, 0) == forked);
  CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 7);
  puts("and stays so after that thread exits");

  /* Two threads use 200 ms each and are gone; the leader adds 50 ms. */
  pid_t c = fork();
  if (c == 0) {
    pthread_t b[2];
    for (int i = 0; i < 2; i++) pthread_create(&b[i], 0, burn, (void *)200L);
    for (int i = 0; i < 2; i++) pthread_join(b[i], 0);
    burn((void *)50L);
    _exit(0);
  }
  struct rusage ru;
  CHECK(wait4(c, &status, 0, &ru) == c);
  long ms = tv_ms(ru.ru_utime, ru.ru_stime);
  if (ms < 400) printf("wait4: child used %ld ms\n", ms);
  CHECK(ms >= 400);
  CHECK(getrusage(RUSAGE_CHILDREN, &ru) == 0);
  CHECK(tv_ms(ru.ru_utime, ru.ru_stime) >= 400);
  puts("a reaped child's CPU time counts its exited threads");
  puts("PASS");
  return 0;
}
