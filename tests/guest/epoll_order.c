/* epoll_wait hands back ready fds in the order they became ready, as Linux's
 * ready list does, and a reported level-triggered fd goes to the back of the
 * line. Xvfb depends on the first: handed a new connection before an older
 * client's hangup in the same batch, it took the hangup to mean its last
 * client had left, reset, and dropped the new client with it. */
#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d, errno %d)\n", #c, __LINE__, errno); exit(1); } } while (0)

int main(void) {
  /* An abstract listener (fd numbers below any accepted client's). */
  struct sockaddr_un a = {.sun_family = AF_UNIX};
  snprintf(a.sun_path + 1, sizeof a.sun_path - 1, "nixvm-epoll-order-%d", getpid());
  socklen_t alen = offsetof(struct sockaddr_un, sun_path) + 1 + strlen(a.sun_path + 1);
  int lis = socket(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK, 0);
  CHECK(lis >= 0);
  CHECK(bind(lis, (struct sockaddr *)&a, alen) == 0);
  CHECK(listen(lis, 8) == 0);

  int c1 = socket(AF_UNIX, SOCK_STREAM, 0);
  CHECK(connect(c1, (struct sockaddr *)&a, alen) == 0);
  int s1 = accept4(lis, NULL, NULL, SOCK_NONBLOCK);
  CHECK(s1 > lis);

  int ep = epoll_create1(0);
  struct epoll_event ev = {.events = EPOLLIN, .data.fd = lis};
  CHECK(epoll_ctl(ep, EPOLL_CTL_ADD, lis, &ev) == 0);
  ev.data.fd = s1;
  CHECK(epoll_ctl(ep, EPOLL_CTL_ADD, s1, &ev) == 0);
  struct epoll_event out[4];
  CHECK(epoll_wait(ep, out, 4, 0) == 0);

  /* The first client hangs up, then a second one connects. */
  CHECK(close(c1) == 0);
  int c2 = socket(AF_UNIX, SOCK_STREAM, 0);
  CHECK(connect(c2, (struct sockaddr *)&a, alen) == 0);
  CHECK(epoll_wait(ep, out, 4, 0) == 2);
  CHECK(out[0].data.fd == s1 && (out[0].events & EPOLLHUP));
  CHECK(out[1].data.fd == lis && out[1].events == EPOLLIN);
  printf("a hangup that came first is reported first\n");

  /* Both stay ready (level-triggered): one at a time, they take turns. */
  CHECK(epoll_wait(ep, out, 1, 0) == 1 && out[0].data.fd == s1);
  CHECK(epoll_wait(ep, out, 1, 0) == 1 && out[0].data.fd == lis);
  CHECK(epoll_wait(ep, out, 1, 0) == 1 && out[0].data.fd == s1);
  printf("level-triggered fds take turns\n");

  /* Fresh data makes a socket the newest arrival: it comes last. */
  int s2 = accept4(lis, NULL, NULL, SOCK_NONBLOCK);
  CHECK(s2 >= 0);
  CHECK(epoll_ctl(ep, EPOLL_CTL_DEL, s1, NULL) == 0);
  ev.data.fd = s2;
  CHECK(epoll_ctl(ep, EPOLL_CTL_ADD, s2, &ev) == 0);
  int c3 = socket(AF_UNIX, SOCK_STREAM, 0);
  CHECK(connect(c3, (struct sockaddr *)&a, alen) == 0); /* listener ready */
  CHECK(write(c2, "x", 1) == 1);                         /* then s2 */
  CHECK(epoll_wait(ep, out, 4, 0) == 2);
  CHECK(out[0].data.fd == lis && out[1].data.fd == s2);
  printf("ready fds come back oldest first\n");
  printf("PASS\n");
  return 0;
}
