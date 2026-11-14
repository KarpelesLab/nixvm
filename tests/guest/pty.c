/* forkpty: what a terminal emulator (havoc, foot, xterm) and `script` do to
 * start a shell. The child's pty becomes its controlling terminal
 * (login_tty: setsid + TIOCSCTTY), whose foreground group is the child's.
 * Each side closes the other's end, and the pty stays up while the copies
 * that are left are open: the parent's reads wait for the child's output,
 * the child reads what the parent types, and the parent sees EOF (EIO) only
 * once the child is gone. */
#include <errno.h>
#include <poll.h>
#include <pty.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <termios.h>
#include <unistd.h>

#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d, errno %d)\n", #c, __LINE__, errno); exit(1); } } while (0)

/* Read from the master until `want` shows up; 0 on EOF/error first. */
static int read_until(int m, char *buf, size_t cap, const char *want) {
  size_t got = 0;
  buf[0] = 0;
  while (got < cap - 1) {
    ssize_t n = read(m, buf + got, cap - 1 - got);
    if (n <= 0) return 0;
    got += n;
    buf[got] = 0;
    if (strstr(buf, want)) return 1;
  }
  return 0;
}

int main(void) {
  int m;
  pid_t pid = forkpty(&m, NULL, NULL, NULL);
  CHECK(pid >= 0);
  if (pid == 0) {
    /* Let the parent block reading first. */
    usleep(100000);
    int tty = isatty(0) && isatty(1) && isatty(2);
    int leader = getsid(0) == getpid();
    int fg = tcgetpgrp(0) == getpgrp();
    if (tty && leader && fg)
      printf("child-ok\n");
    else
      printf("child-bad tty=%d leader=%d fg=%d\n", tty, leader, fg);
    fflush(stdout);
    char line[64];
    if (!fgets(line, sizeof line, stdin)) _exit(2);
    printf("got:%s", line);
    fflush(stdout);
    _exit(0);
  }
  char buf[512];
  int ok = read_until(m, buf, sizeof buf, "child-ok");
  if (!ok) printf("the child said: %s\n", buf);
  CHECK(ok);
  CHECK(write(m, "ping\n", 5) == 5);
  ok = read_until(m, buf, sizeof buf, "got:ping");
  if (!ok) printf("the child said: %s\n", buf);
  CHECK(ok);
  int status;
  CHECK(waitpid(pid, &status, 0) == pid);
  CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
  /* Nobody has the slave open any more: the master reads EOF. */
  struct pollfd p = {.fd = m, .events = POLLIN};
  CHECK(poll(&p, 1, 1000) == 1 && (p.revents & (POLLHUP | POLLIN)));
  CHECK(read(m, buf, sizeof buf) <= 0);
  printf("PASS\n");
  return 0;
}
