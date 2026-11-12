/* Dual-stack IPv6 sockets: an AF_INET6 socket listening on :: without
 * IPV6_V6ONLY takes IPv4 connections too, seeing their addresses as
 * IPv4-mapped (::ffff:a.b.c.d) — what Node's listen(port) relies on — and an
 * AF_INET6 socket can connect to an IPv4 listener through a mapped address. */
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d, errno %d)\n", #c, __LINE__, errno); exit(1); } } while (0)

static int listen6(int v6only, int *port) {
  int s = socket(AF_INET6, SOCK_STREAM, 0);
  CHECK(s >= 0);
  CHECK(setsockopt(s, IPPROTO_IPV6, IPV6_V6ONLY, &v6only, sizeof v6only) == 0);
  struct sockaddr_in6 a = {.sin6_family = AF_INET6, .sin6_addr = IN6ADDR_ANY_INIT};
  CHECK(bind(s, (void *)&a, sizeof a) == 0 && listen(s, 4) == 0);
  socklen_t l = sizeof a;
  CHECK(getsockname(s, (void *)&a, &l) == 0);
  *port = ntohs(a.sin6_port);
  return s;
}

static int connect4(int port) {
  int c = socket(AF_INET, SOCK_STREAM, 0);
  CHECK(c >= 0);
  struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons(port)};
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  return connect(c, (void *)&a, sizeof a) == 0 ? c : -1;
}

int main(void) {
  char buf[64];
  int port;
  int l = listen6(0, &port);
  int c = connect4(port);
  CHECK(c >= 0);
  struct sockaddr_in6 peer;
  socklen_t pl = sizeof peer;
  int a = accept(l, (void *)&peer, &pl);
  CHECK(a >= 0);
  CHECK(peer.sin6_family == AF_INET6 && IN6_IS_ADDR_V4MAPPED(&peer.sin6_addr));
  inet_ntop(AF_INET6, &peer.sin6_addr, buf, sizeof buf);
  printf("dual-stack listener accepted an IPv4 client as %s\n", buf);
  CHECK(strcmp(buf, "::ffff:127.0.0.1") == 0);
  struct sockaddr_in6 self;
  socklen_t sl = sizeof self;
  CHECK(getsockname(a, (void *)&self, &sl) == 0 && IN6_IS_ADDR_V4MAPPED(&self.sin6_addr));
  struct sockaddr_in cpeer;
  socklen_t cl = sizeof cpeer;
  CHECK(getpeername(c, (void *)&cpeer, &cl) == 0 && cpeer.sin_family == AF_INET);
  CHECK(ntohl(cpeer.sin_addr.s_addr) == INADDR_LOOPBACK && ntohs(cpeer.sin_port) == port);
  CHECK(write(c, "hi", 2) == 2 && read(a, buf, 2) == 2 && memcmp(buf, "hi", 2) == 0);
  close(a);
  close(c);
  close(l);

  /* IPV6_V6ONLY: IPv6 only. */
  l = listen6(1, &port);
  errno = 0;
  CHECK(connect4(port) < 0 && errno == ECONNREFUSED);
  close(l);
  puts("an IPV6_V6ONLY listener refuses IPv4");

  /* An AF_INET6 socket reaching an IPv4 listener through ::ffff:127.0.0.1. */
  int l4 = socket(AF_INET, SOCK_STREAM, 0);
  struct sockaddr_in b = {.sin_family = AF_INET};
  b.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  CHECK(bind(l4, (void *)&b, sizeof b) == 0 && listen(l4, 4) == 0);
  socklen_t bl = sizeof b;
  CHECK(getsockname(l4, (void *)&b, &bl) == 0);
  int c6 = socket(AF_INET6, SOCK_STREAM, 0);
  struct sockaddr_in6 m = {.sin6_family = AF_INET6, .sin6_port = b.sin_port};
  CHECK(inet_pton(AF_INET6, "::ffff:127.0.0.1", &m.sin6_addr) == 1);
  CHECK(connect(c6, (void *)&m, sizeof m) == 0);
  struct sockaddr_in p4;
  socklen_t p4l = sizeof p4;
  a = accept(l4, (void *)&p4, &p4l);
  CHECK(a >= 0 && p4.sin_family == AF_INET && ntohl(p4.sin_addr.s_addr) == INADDR_LOOPBACK);
  struct sockaddr_in6 me;
  socklen_t mel = sizeof me;
  CHECK(getsockname(c6, (void *)&me, &mel) == 0 && IN6_IS_ADDR_V4MAPPED(&me.sin6_addr));
  puts("an AF_INET6 socket connects to an IPv4 listener through ::ffff:127.0.0.1");
  puts("PASS");
  return 0;
}
