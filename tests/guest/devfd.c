/* /proc/self/fd/N (and /dev/stdout, /dev/fd/N, which lead there) is a magic
 * link: opening it opens descriptor N's file again, not the name readlink
 * shows. Container images point their log files at /dev/stdout this way. */
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d, errno %d)\n", #c, __LINE__, errno); exit(1); } } while (0)

int main(void) {
  char buf[64], path[64], tmp[96];
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);

  /* A pipe's read end, reopened for writing, is a writer of the same pipe. */
  int p[2];
  CHECK(pipe(p) == 0);
  snprintf(path, sizeof path, "/proc/self/fd/%d", p[0]);
  int w = open(path, O_WRONLY);
  CHECK(w >= 0);
  CHECK(write(w, "via-proc", 8) == 8);
  CHECK(read(p[0], buf, 8) == 8 && memcmp(buf, "via-proc", 8) == 0);
  /* /dev/fd/N leads to the same place. */
  snprintf(path, sizeof path, "/dev/fd/%d", p[1]);
  int w2 = open(path, O_WRONLY);
  CHECK(w2 >= 0 && write(w2, "dev-fd", 6) == 6);
  CHECK(read(p[0], buf, 6) == 6 && memcmp(buf, "dev-fd", 6) == 0);
  /* The pipe's writers include the reopened ends: closing the original
   * write end leaves it open. */
  close(p[1]);
  close(w2);
  CHECK(write(w, "x", 1) == 1);
  close(w);
  CHECK(read(p[0], buf, sizeof buf) == 1);
  CHECK(read(p[0], buf, sizeof buf) == 0); /* now EOF */
  close(p[0]);
  puts("a pipe end reopened through /proc/self/fd");

  /* /dev/stdout is the process's stdout, whatever that is: here a pipe to
   * the parent. */
  CHECK(pipe(p) == 0);
  pid_t c = fork();
  if (!c) {
    dup2(p[1], 1);
    close(p[0]);
    close(p[1]);
    int o = open("/dev/stdout", O_WRONLY | O_CREAT | O_APPEND, 0644);
    _exit(o >= 0 && write(o, "stdout!", 7) == 7 ? 0 : 1);
  }
  close(p[1]);
  int st;
  CHECK(waitpid(c, &st, 0) == c && WIFEXITED(st) && WEXITSTATUS(st) == 0);
  CHECK(read(p[0], buf, sizeof buf) == 7 && memcmp(buf, "stdout!", 7) == 0);
  close(p[0]);
  puts("/dev/stdout opened with O_CREAT|O_APPEND reaches the pipe");

  /* A file is opened again by its path: a description of its own, from
   * offset 0, even after a rename. */
  snprintf(tmp, sizeof tmp, "/tmp/devfd-%d-%ld", getpid(), t.tv_nsec);
  int f = open(tmp, O_RDWR | O_CREAT | O_TRUNC, 0600);
  CHECK(f >= 0 && write(f, "abcdef", 6) == 6);
  snprintf(path, sizeof path, "/proc/self/fd/%d", f);
  int g = open(path, O_RDONLY);
  CHECK(g >= 0);
  CHECK(read(g, buf, 3) == 3 && memcmp(buf, "abc", 3) == 0);
  CHECK(lseek(f, 0, SEEK_CUR) == 6); /* f's offset is its own */
  errno = 0;
  CHECK(write(g, "x", 1) == -1 && errno == EBADF); /* opened read-only */
  close(g);
  close(f);
  unlink(tmp);
  puts("a file reopened through /proc/self/fd has its own offset and mode");

  /* A closed descriptor's link is gone; a socket can't be opened this way. */
  errno = 0;
  CHECK(open("/proc/self/fd/987", O_RDONLY) == -1 && errno == ENOENT);
  int s = socket(AF_UNIX, SOCK_STREAM, 0);
  CHECK(s >= 0);
  snprintf(path, sizeof path, "/proc/self/fd/%d", s);
  errno = 0;
  CHECK(open(path, O_RDWR) == -1 && errno == ENXIO);
  close(s);
  puts("closed fd ENOENT, socket ENXIO");
  puts("PASS");
  return 0;
}
