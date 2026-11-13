/* execve from a thread that isn't the process's first: every other thread
 * goes, and the exec'ing thread becomes the process — the new image runs as
 * the process's PID (what its parent waits on), alone. */
#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d, errno %d)\n", #c, __LINE__, errno); exit(1); } } while (0)

static char pidarg[32];

static void *execer(void *a) {
  (void)a;
  /* (In the conformance root the program is at /execthread; the runner
   * there doesn't name it for /proc/self/exe.) */
  const char *self = access("/execthread", X_OK) == 0 ? "/execthread" : "/proc/self/exe";
  char *argv[] = {(char *)self, "after-exec", pidarg, 0};
  execv(self, argv);
  _exit(99);
}

static void *spinner(void *a) {
  (void)a;
  for (;;) pause();
}

int main(int argc, char **argv) {
  if (argc == 3 && strcmp(argv[1], "after-exec") == 0) {
    /* The new image: the old PID, the thread ID equal to it, one thread. */
    int tasks = 0;
    DIR *d = opendir("/proc/self/task");
    if (d) {
      struct dirent *e;
      while ((e = readdir(d)))
        if (e->d_name[0] != '.') tasks++;
      closedir(d);
    }
    int ok = getpid() == atoi(argv[2]) && syscall(SYS_gettid) == getpid() && (tasks == 0 || tasks == 1);
    printf("after exec: pid %d (was %s), tid %ld, %d task(s)\n", getpid(), argv[2], (long)syscall(SYS_gettid), tasks);
    return ok ? 42 : 1;
  }
  fflush(stdout);
  pid_t c = fork();
  CHECK(c >= 0);
  if (!c) {
    snprintf(pidarg, sizeof pidarg, "%d", getpid());
    pthread_t t1, t2;
    pthread_create(&t1, 0, spinner, 0);
    pthread_create(&t2, 0, execer, 0);
    for (;;) pause(); /* the main thread just waits to be taken away */
  }
  int st;
  CHECK(waitpid(c, &st, 0) == c);
  printf("the parent's waitpid(%d) got exit %d\n", c, WIFEXITED(st) ? WEXITSTATUS(st) : -1);
  CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 42);
  puts("PASS");
  return 0;
}
