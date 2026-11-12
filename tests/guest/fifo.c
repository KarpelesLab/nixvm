/* Named FIFOs (mkfifo), as Linux's fifo_open has them: every open of the node
 * shares one pipe, found by the node itself — so a FIFO renamed or moved
 * while open is still the same pipe — and the opens wait for each other. */
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d, errno %d)\n", #c, __LINE__, errno); exit(1); } } while (0)

static long ms_since(struct timespec *t0) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (t.tv_sec - t0->tv_sec) * 1000 + (t.tv_nsec - t0->tv_nsec) / 1000000;
}

static void on_alarm(int s) { (void)s; }

int main(void) {
  struct timespec t0;
  clock_gettime(CLOCK_MONOTONIC, &t0);
  char d[64], f[128], moved[160], buf[64];
  snprintf(d, sizeof d, "/tmp/fifo-%d-%ld", getpid(), t0.tv_nsec);
  CHECK(mkdir(d, 0755) == 0);
  snprintf(f, sizeof f, "%s/p", d);
  CHECK(mkfifo(f, 0600) == 0);
  struct stat st;
  CHECK(stat(f, &st) == 0 && S_ISFIFO(st.st_mode));

  /* A blocking writer waits for a reader; the reader gets its bytes, then EOF. */
  pid_t p = fork();
  if (!p) {
    int w = open(f, O_WRONLY);
    _exit(w >= 0 && write(w, "hello", 5) == 5 ? 0 : 1);
  }
  usleep(200 * 1000);
  int st_child;
  CHECK(waitpid(p, &st_child, WNOHANG) == 0); /* still waiting for a reader */
  int r = open(f, O_RDONLY);
  CHECK(r >= 0);
  CHECK(read(r, buf, sizeof buf) == 5 && memcmp(buf, "hello", 5) == 0);
  CHECK(waitpid(p, &st_child, 0) == p && WIFEXITED(st_child) && WEXITSTATUS(st_child) == 0);
  CHECK(read(r, buf, sizeof buf) == 0);
  close(r);
  puts("blocking writer and reader meet");

  /* A blocking reader waits for a writer, too. */
  clock_gettime(CLOCK_MONOTONIC, &t0);
  p = fork();
  if (!p) {
    usleep(200 * 1000);
    int w = open(f, O_WRONLY);
    _exit(w >= 0 && write(w, "x", 1) == 1 ? 0 : 1);
  }
  r = open(f, O_RDONLY);
  long waited = ms_since(&t0);
  printf("reader waited %ld ms for the writer\n", waited);
  CHECK(r >= 0 && waited >= 150);
  CHECK(read(r, buf, 1) == 1 && buf[0] == 'x');
  CHECK(waitpid(p, &st_child, 0) == p && WEXITSTATUS(st_child) == 0);
  close(r);

  /* O_NONBLOCK: a reader goes at once (and reads EOF), a writer with no
   * reader is ENXIO. */
  r = open(f, O_RDONLY | O_NONBLOCK);
  CHECK(r >= 0);
  CHECK(read(r, buf, 1) == 0);
  close(r);
  errno = 0;
  CHECK(open(f, O_WRONLY | O_NONBLOCK) == -1 && errno == ENXIO);
  puts("O_NONBLOCK opens: reader goes, lone writer ENXIO");

  /* O_RDWR never waits, and reads back what it writes. */
  int rw = open(f, O_RDWR);
  CHECK(rw >= 0);
  CHECK((fcntl(rw, F_GETFL) & O_ACCMODE) == O_RDWR);
  CHECK(write(rw, "ab", 2) == 2);
  CHECK(read(rw, buf, 2) == 2 && memcmp(buf, "ab", 2) == 0);

  /* Each open has its own O_NONBLOCK. */
  int nb = open(f, O_RDONLY | O_NONBLOCK);
  int bl = open(f, O_RDONLY);
  CHECK(nb >= 0 && bl >= 0);
  CHECK(fcntl(nb, F_GETFL) & O_NONBLOCK);
  CHECK(!(fcntl(bl, F_GETFL) & O_NONBLOCK));
  errno = 0;
  CHECK(read(nb, buf, 1) == -1 && errno == EAGAIN);
  clock_gettime(CLOCK_MONOTONIC, &t0);
  p = fork();
  if (!p) {
    usleep(200 * 1000);
    _exit(write(rw, "z", 1) == 1 ? 0 : 1);
  }
  CHECK(read(bl, buf, 1) == 1 && buf[0] == 'z');
  waited = ms_since(&t0);
  printf("blocking read beside a non-blocking open waited %ld ms\n", waited);
  CHECK(waited >= 150);
  CHECK(waitpid(p, &st_child, 0) == p);
  close(nb);
  close(bl);

  /* Renamed into another directory while open: the new name is the same
   * pipe, and the descriptor is still the node. */
  struct stat fst;
  CHECK(fstat(rw, &fst) == 0 && S_ISFIFO(fst.st_mode));
  CHECK(fst.st_ino == st.st_ino && fst.st_dev == st.st_dev);
  snprintf(moved, sizeof moved, "%s/sub", d);
  CHECK(mkdir(moved, 0755) == 0);
  snprintf(moved, sizeof moved, "%s/sub/renamed", d);
  CHECK(rename(f, moved) == 0);
  int w2 = open(moved, O_WRONLY | O_NONBLOCK); /* rw is a reader */
  CHECK(w2 >= 0);
  CHECK(write(w2, "moved", 5) == 5);
  CHECK(read(rw, buf, 5) == 5 && memcmp(buf, "moved", 5) == 0);
  struct stat st2;
  CHECK(stat(moved, &st2) == 0 && st2.st_ino == st.st_ino);
  close(w2);
  CHECK(rename(moved, f) == 0);
  puts("a renamed FIFO is the same pipe");

  /* The contents go with the last close. */
  CHECK(write(rw, "lost", 4) == 4);
  close(rw);
  rw = open(f, O_RDWR | O_NONBLOCK);
  CHECK(rw >= 0);
  errno = 0;
  CHECK(read(rw, buf, sizeof buf) == -1 && errno == EAGAIN);
  close(rw);

  /* A waiting open a signal interrupts fails EINTR and stops counting as a
   * reader: a non-blocking writer then finds none. */
  struct sigaction sa = {0};
  sa.sa_handler = on_alarm; /* no SA_RESTART */
  sigaction(SIGALRM, &sa, 0);
  alarm(1);
  errno = 0;
  r = open(f, O_RDONLY);
  CHECK(r == -1 && errno == EINTR);
  errno = 0;
  CHECK(open(f, O_WRONLY | O_NONBLOCK) == -1 && errno == ENXIO);
  puts("an interrupted open is EINTR and gone");

  /* A reader already waiting in open() while the FIFO is renamed: a writer
   * opening the new name is the one it waits for. */
  p = fork();
  if (!p) {
    int rr = open(f, O_RDONLY);
    char c;
    _exit(rr >= 0 && read(rr, &c, 1) == 1 && c == 'r' ? 0 : 1);
  }
  usleep(200 * 1000);
  snprintf(moved, sizeof moved, "%s/sub/renamed", d);
  CHECK(rename(f, moved) == 0);
  int w3 = open(moved, O_WRONLY);
  CHECK(w3 >= 0 && write(w3, "r", 1) == 1);
  CHECK(waitpid(p, &st_child, 0) == p && WIFEXITED(st_child) && WEXITSTATUS(st_child) == 0);
  close(w3);
  CHECK(rename(moved, f) == 0);
  puts("an open waiting while the FIFO is renamed gets it");

  /* Two processes opening opposite ends blocking both get through. */
  p = fork();
  if (!p) {
    int w = open(f, O_WRONLY);
    _exit(w >= 0 ? 0 : 1);
  }
  r = open(f, O_RDONLY);
  CHECK(r >= 0);
  CHECK(waitpid(p, &st_child, 0) == p && WEXITSTATUS(st_child) == 0);
  CHECK(read(r, buf, 1) == 0); /* the writer has gone */
  close(r);

  unlink(f);
  snprintf(moved, sizeof moved, "%s/sub", d);
  rmdir(moved);
  rmdir(d);
  puts("PASS");
  return 0;
}
