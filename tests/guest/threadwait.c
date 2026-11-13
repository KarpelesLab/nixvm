/* A child belongs to its whole process, not only to the thread that forked
 * it: another thread waits for it (.NET's Process.WaitForExit reaps from
 * its signal-handling thread, with waitid(P_ALL, WNOHANG|WNOWAIT) first),
 * and a child whose forking thread exits stays the process's; its getppid()
 * is the process. */
#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
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
  puts("PASS");
  return 0;
}
