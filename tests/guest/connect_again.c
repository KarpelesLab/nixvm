/* A nonblocking connect's caller asks again to learn it finished (hiredis,
 * redis-cli): once it has, the next connect() is 0 if the first said
 * EINPROGRESS, and every one after that — and one on a listening socket —
 * is EISCONN, not EINVAL. */
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <unistd.h>

#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d, errno %d)\n", #c, __LINE__, errno); exit(1); } } while (0)

int main(void) {
  int l = socket(AF_INET, SOCK_STREAM, 0);
  struct sockaddr_in a = {.sin_family = AF_INET, .sin_addr.s_addr = htonl(INADDR_LOOPBACK)};
  CHECK(bind(l, (struct sockaddr *)&a, sizeof a) == 0 && listen(l, 4) == 0);
  socklen_t len = sizeof a;
  CHECK(getsockname(l, (struct sockaddr *)&a, &len) == 0);
  int c = socket(AF_INET, SOCK_STREAM, 0);
  CHECK(fcntl(c, F_SETFL, O_NONBLOCK) == 0);
  int r = connect(c, (struct sockaddr *)&a, sizeof a);
  CHECK(r == 0 || errno == EINPROGRESS);
  struct pollfd p = {.fd = c, .events = POLLOUT};
  CHECK(poll(&p, 1, 5000) == 1 && (p.revents & POLLOUT));
  if (r == -1) CHECK(connect(c, (struct sockaddr *)&a, sizeof a) == 0);
  errno = 0;
  CHECK(connect(c, (struct sockaddr *)&a, sizeof a) == -1 && errno == EISCONN);
  errno = 0;
  CHECK(connect(l, (struct sockaddr *)&a, sizeof a) == -1 && errno == EISCONN);
  puts("connect again is EISCONN");
  puts("PASS");
  return 0;
}
