#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
static int fails;
#define CHECK(c, ...) do { int ok_ = (c); printf("%s: ", ok_ ? "ok  " : "FAIL"); printf(__VA_ARGS__); puts(""); if (!ok_) fails++; } while (0)
static struct sockaddr_in addr(int port) { struct sockaddr_in a = {0}; a.sin_family = AF_INET; a.sin_port = htons(port); a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); return a; }
static int sock(int type, int opt) { int s = socket(AF_INET, type, 0), one = 1; if (opt) setsockopt(s, SOL_SOCKET, opt, &one, sizeof one); return s; }
static int bnd(int s, int port) { struct sockaddr_in a = addr(port); return bind(s, (void *)&a, sizeof a); }
int main(void) {
  int a, b, r;
  a = sock(SOCK_STREAM, 0); b = sock(SOCK_STREAM, 0);
  CHECK(!bnd(a, 7001), "first bind");
  r = bnd(b, 7001); CHECK(r < 0 && errno == EADDRINUSE, "second bind EADDRINUSE (errno=%d)", errno);
  a = sock(SOCK_STREAM, SO_REUSEADDR); b = sock(SOCK_STREAM, SO_REUSEADDR);
  CHECK(!bnd(a, 7002) && !bnd(b, 7002), "REUSEADDR: both bind while idle");
  CHECK(!listen(a, 8), "first listen");
  r = listen(b, 8); CHECK(r < 0 && errno == EADDRINUSE, "REUSEADDR: second listen EADDRINUSE (errno=%d)", errno);
  int c = sock(SOCK_STREAM, SO_REUSEADDR);
  r = bnd(c, 7002); CHECK(r < 0 && errno == EADDRINUSE, "REUSEADDR: can't bind over a listener (errno=%d)", errno);
  a = sock(SOCK_STREAM, SO_REUSEPORT); b = sock(SOCK_STREAM, SO_REUSEPORT);
  CHECK(!bnd(a, 7003) && !bnd(b, 7003) && !listen(a, 8) && !listen(b, 8), "REUSEPORT: both listen");
  int got[2] = {0, 0}; struct sockaddr_in t = addr(7003);
  for (int i = 0; i < 4; i++) { int cl = socket(AF_INET, SOCK_STREAM, 0); connect(cl, (void *)&t, sizeof t); }
  struct pollfd p[2] = {{a, POLLIN, 0}, {b, POLLIN, 0}}; poll(p, 2, 100);
  for (int i = 0; i < 2; i++) while (1) { struct pollfd q = {i ? b : a, POLLIN, 0}; if (poll(&q, 1, 0) <= 0) break; if (accept(q.fd, 0, 0) < 0) break; got[i]++; }
  CHECK(got[0] + got[1] == 4, "REUSEPORT: every connection reaches a listener (%d, %d)", got[0], got[1]);
  a = sock(SOCK_DGRAM, SO_REUSEPORT); b = sock(SOCK_DGRAM, SO_REUSEPORT);
  CHECK(!bnd(a, 7004) && !bnd(b, 7004), "UDP REUSEPORT: both bind");
  int u = socket(AF_INET, SOCK_DGRAM, 0); t = addr(7004); sendto(u, "x", 1, 0, (void *)&t, sizeof t);
  char ch; int m = recv(a, &ch, 1, MSG_DONTWAIT) + recv(b, &ch, 1, MSG_DONTWAIT);
  CHECK(m == 0, "UDP REUSEPORT: datagram reached exactly one socket");
  close(a); close(b);
  a = sock(SOCK_DGRAM, 0); CHECK(!bnd(a, 7004), "UDP port free again after close");
  struct timeval tv = {0, 200000}; a = socket(AF_INET, SOCK_DGRAM, 0); bnd(a, 7005);
  setsockopt(a, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
  struct timeval rd; socklen_t l = sizeof rd; getsockopt(a, SOL_SOCKET, SO_RCVTIMEO, &rd, &l);
  CHECK(rd.tv_sec == 0 && rd.tv_usec > 195000 && rd.tv_usec < 205000, "SO_RCVTIMEO reads back (%ld)", (long)rd.tv_usec);
  struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0); r = recv(a, &ch, 1, 0); int e = errno; clock_gettime(CLOCK_MONOTONIC, &t1);
  long ms = (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_nsec - t0.tv_nsec) / 1000000;
  CHECK(r < 0 && e == EAGAIN && ms >= 190 && ms < 1000, "recv timed out EAGAIN after %ldms", ms);
  clock_gettime(CLOCK_MONOTONIC, &t0); r = read(a, &ch, 1); e = errno; clock_gettime(CLOCK_MONOTONIC, &t1);
  ms = (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_nsec - t0.tv_nsec) / 1000000;
  CHECK(r < 0 && e == EAGAIN && ms >= 190, "read() honors it too (%ldms)", ms);
  tv.tv_usec = 2000000; r = setsockopt(a, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv); CHECK(r < 0 && errno == EDOM, "bad timeval EDOM");
  a = socket(AF_INET, SOCK_STREAM, 0); int one = 1, v = 0; l = sizeof v;
  CHECK(!setsockopt(a, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one), "TCP_NODELAY accepted");
  getsockopt(a, IPPROTO_TCP, TCP_NODELAY, &v, &l); CHECK(v == 1, "TCP_NODELAY reads back");
  CHECK(!setsockopt(a, SOL_SOCKET, SO_KEEPALIVE, &one, sizeof one), "SO_KEEPALIVE accepted");
  v = 1 << 30; setsockopt(a, SOL_SOCKET, SO_RCVBUF, &v, sizeof v); getsockopt(a, SOL_SOCKET, SO_RCVBUF, &v, &l); CHECK(v > 0 && v < (1 << 30), "SO_RCVBUF clamped (%d)", v);
  puts(fails ? "SOME FAILED" : "PASS"); return fails != 0;
}
