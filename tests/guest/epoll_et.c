/* Edge-triggered epoll raises a fresh edge on every arrival, even while the fd
 * is still readable: a reader that consumed only part of the data (tokio takes
 * a short read to mean "drained") must still hear about the next write. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <unistd.h>

#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d, errno %d)\n", #c, __LINE__, errno); exit(1); } } while (0)

static void check(const char *what, int rd, int wr) {
  int ep = epoll_create1(0);
  CHECK(ep >= 0);
  struct epoll_event ev = {.events = EPOLLIN | EPOLLET, .data.fd = rd}, out;
  CHECK(epoll_ctl(ep, EPOLL_CTL_ADD, rd, &ev) == 0);
  CHECK(epoll_wait(ep, &out, 1, 0) == 0); /* nothing yet */
  CHECK(write(wr, "0123456789", 10) == 10);
  CHECK(epoll_wait(ep, &out, 1, 1000) == 1);
  CHECK(epoll_wait(ep, &out, 1, 0) == 0); /* the edge was consumed */
  char buf[16];
  CHECK(read(rd, buf, 5) == 5); /* half: still readable */
  CHECK(epoll_wait(ep, &out, 1, 0) == 0); /* a read is no arrival */
  CHECK(write(wr, "abcde", 5) == 5);
  /* The new data is an edge although the fd never stopped being readable. */
  CHECK(epoll_wait(ep, &out, 1, 1000) == 1 && out.data.fd == rd);
  CHECK(read(rd, buf, sizeof buf) == 10);
  CHECK(epoll_wait(ep, &out, 1, 0) == 0);
  close(ep);
  printf("%s: a write to a readable fd is a new edge\n", what);
}

int main(void) {
  int sv[2];
  CHECK(socketpair(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0, sv) == 0);
  check("socketpair", sv[0], sv[1]);
  int p[2];
  CHECK(pipe2(p, O_NONBLOCK) == 0);
  check("pipe", p[0], p[1]);
  puts("PASS");
  return 0;
}
