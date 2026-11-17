/* A process goes while its threads are busy: one thread calls exit_group
 * (or execve) while its siblings make syscalls, touch their stacks and wait
 * on futexes, on other CPUs. Once the process is gone no thread of it runs
 * again: none comes back from a syscall into the torn-down address space and
 * faults there (which a SIGSEGV would make the process's exit status), and
 * the process exits as the exit_group said. */
#define _GNU_SOURCE
#include <errno.h>
#include <linux/futex.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d, errno %d)\n", #c, __LINE__, errno); exit(1); } } while (0)

#define THREADS 6
#define ROUNDS 40

static volatile int word;
static volatile int started;

/* Syscalls in a loop, each followed by user code on the stack and on a
 * mapping of its own that it keeps remapping. */
static void *busy(void *a) {
  long n = (long)a;
  volatile char buf[256];
  __sync_fetch_and_add(&started, 1);
  for (unsigned i = 0;; i++) {
    syscall(SYS_getpid);
    memset((char *)buf, (int)(i + n), sizeof buf);
    if (i % 16 == 0) {
      char *p = mmap(0, 65536, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
      if (p != MAP_FAILED) {
        p[0] = p[65535] = (char)i;
        munmap(p, 65536);
      }
    }
  }
  return 0;
}

/* Short futex waits on a word that nobody changes, then user code. */
static void *waiter(void *a) {
  (void)a;
  volatile char buf[256];
  struct timespec ts = {0, 200000};
  __sync_fetch_and_add(&started, 1);
  for (unsigned i = 0;; i++) {
    syscall(SYS_futex, &word, FUTEX_WAIT_PRIVATE, 0, &ts, 0, 0);
    memset((char *)buf, (int)i, sizeof buf);
  }
  return 0;
}

/* Woken in batches by the exiting thread right before it goes. */
static void *sleeper(void *a) {
  (void)a;
  volatile char buf[256];
  __sync_fetch_and_add(&started, 1);
  for (unsigned i = 0;; i++) {
    syscall(SYS_futex, &word, FUTEX_WAIT_PRIVATE, 0, 0, 0, 0);
    memset((char *)buf, (int)i, sizeof buf);
  }
  return 0;
}

static const char *self;

static void child(int round) {
  pthread_t t;
  for (long i = 0; i < THREADS; i++) {
    void *(*f)(void *) = i % 3 == 0 ? busy : i % 3 == 1 ? waiter : sleeper;
    CHECK(pthread_create(&t, 0, f, (void *)i) == 0);
  }
  while (started < THREADS) sched_yield();
  for (int i = 0; i < 20 + round % 7; i++) {
    syscall(SYS_futex, &word, FUTEX_WAKE_PRIVATE, THREADS, 0, 0, 0);
    syscall(SYS_getpid);
  }
  syscall(SYS_futex, &word, FUTEX_WAKE_PRIVATE, THREADS, 0, 0, 0);
  if (round % 4 == 3) {
    /* execve: the new image exits 42 at once. */
    char *argv[] = {(char *)self, "after-exec", 0};
    execv(self, argv);
    _exit(99);
  }
  syscall(SYS_exit_group, 42);
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "after-exec") == 0) return 42;
  self = access("/exit_race", X_OK) == 0 ? "/exit_race" : "/proc/self/exe";
  int rounds = argc > 1 ? atoi(argv[1]) : ROUNDS;
  for (int r = 0; r < rounds; r++) {
    pid_t pid = fork();
    CHECK(pid >= 0);
    if (pid == 0) child(r);
    int status;
    CHECK(waitpid(pid, &status, 0) == pid);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 42) {
      printf("FAIL: round %d: status %#x\n", r, status);
      exit(1);
    }
  }
  printf("PASS\n");
  return 0;
}
