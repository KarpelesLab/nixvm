/* UDP multicast between sockets of one host: joining and leaving groups,
 * delivery to every member (the sender too), binding a group's address,
 * IP_MULTICAST_LOOP and IP_MULTICAST_ALL, and the errors Linux gives. */
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d, errno %d)\n", #c, __LINE__, errno); exit(1); } } while (0)

static const char *GROUP = "239.7.7.7";

static int udp(int port, const char *at) {
  int s = socket(AF_INET, SOCK_DGRAM, 0), one = 1;
  CHECK(s >= 0);
  CHECK(setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one) == 0);
  struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons(port)};
  inet_pton(AF_INET, at, &a.sin_addr);
  CHECK(bind(s, (struct sockaddr *)&a, sizeof a) == 0);
  return s;
}

static int membership(int s, int op, const char *group) {
  struct ip_mreq m = {0};
  inet_pton(AF_INET, group, &m.imr_multiaddr);
  m.imr_interface.s_addr = htonl(INADDR_ANY);
  return setsockopt(s, IPPROTO_IP, op, &m, sizeof m);
}

static void send_to(int s, const char *group, int port, const char *msg) {
  struct sockaddr_in a = {.sin_family = AF_INET, .sin_port = htons(port)};
  inet_pton(AF_INET, group, &a.sin_addr);
  CHECK(sendto(s, msg, strlen(msg), 0, (struct sockaddr *)&a, sizeof a) == (ssize_t)strlen(msg));
}

/* What arrives on `s` within `ms`, or "" if nothing does. */
static const char *got(int s, int ms, int *from_port) {
  static char buf[64];
  struct pollfd p = {.fd = s, .events = POLLIN};
  if (poll(&p, 1, ms) != 1) return "";
  struct sockaddr_in from;
  socklen_t fl = sizeof from;
  ssize_t n = recvfrom(s, buf, sizeof buf - 1, 0, (struct sockaddr *)&from, &fl);
  CHECK(n >= 0);
  buf[n] = 0;
  if (from_port) *from_port = ntohs(from.sin_port);
  return buf;
}

int main(void) {
  const int port = 47123;
  int a = udp(port, "0.0.0.0"), b = udp(port, "0.0.0.0");
  CHECK(membership(a, IP_ADD_MEMBERSHIP, GROUP) == 0);
  CHECK(membership(b, IP_ADD_MEMBERSHIP, GROUP) == 0);
  CHECK(membership(a, IP_ADD_MEMBERSHIP, GROUP) == -1 && errno == EADDRINUSE);
  CHECK(membership(a, IP_ADD_MEMBERSHIP, "127.0.0.1") == -1 && errno == EINVAL);
  puts("join, twice, and a unicast address: ok");

  int tx = socket(AF_INET, SOCK_DGRAM, 0);
  struct sockaddr_in me = {.sin_family = AF_INET};
  socklen_t ml = sizeof me;
  CHECK(bind(tx, (struct sockaddr *)&me, sizeof me) == 0);
  getsockname(tx, (struct sockaddr *)&me, &ml);
  send_to(tx, GROUP, port, "one");
  int fp = 0;
  CHECK(!strcmp(got(a, 1000, &fp), "one") && fp == ntohs(me.sin_port));
  CHECK(!strcmp(got(b, 1000, 0), "one"));
  puts("every member gets a copy, from the sender's port: ok");

  /* A member that sends gets its own datagram back. */
  send_to(a, GROUP, port, "loop");
  CHECK(!strcmp(got(a, 1000, 0), "loop"));
  CHECK(!strcmp(got(b, 1000, 0), "loop"));
  puts("the sender, if a member, too: ok");

  /* IP_MULTICAST_LOOP off: nothing on this host gets it. */
  unsigned char off = 0;
  CHECK(setsockopt(tx, IPPROTO_IP, IP_MULTICAST_LOOP, &off, 1) == 0);
  int v = -1; socklen_t vl = sizeof v;
  CHECK(getsockopt(tx, IPPROTO_IP, IP_MULTICAST_LOOP, &v, &vl) == 0 && v == 0);
  send_to(tx, GROUP, port, "quiet");
  CHECK(!*got(a, 200, 0) && !*got(b, 200, 0));
  unsigned char on = 1;
  setsockopt(tx, IPPROTO_IP, IP_MULTICAST_LOOP, &on, 1);
  puts("IP_MULTICAST_LOOP off: ok");

  /* A socket bound to the port that never joined still gets the group's
   * datagrams while another socket is in it (IP_MULTICAST_ALL, on by
   * default), and doesn't with it off. */
  int c = udp(port, "0.0.0.0");
  send_to(tx, GROUP, port, "all");
  CHECK(!strcmp(got(c, 1000, 0), "all"));
  got(a, 1000, 0); got(b, 1000, 0);
  int zero = 0;
  CHECK(setsockopt(c, IPPROTO_IP, IP_MULTICAST_ALL, &zero, sizeof zero) == 0);
  send_to(tx, GROUP, port, "mine");
  CHECK(!strcmp(got(a, 1000, 0), "mine"));
  CHECK(!*got(c, 200, 0));
  got(b, 1000, 0);
  puts("IP_MULTICAST_ALL: ok");

  /* Another group on the same port reaches nobody. */
  send_to(tx, "239.7.7.8", port, "other");
  CHECK(!*got(a, 200, 0) && !*got(b, 200, 0));
  puts("a group nobody joined: ok");

  /* Leaving: the last one out, and nobody gets it. */
  CHECK(membership(a, IP_DROP_MEMBERSHIP, GROUP) == 0);
  CHECK(membership(a, IP_DROP_MEMBERSHIP, GROUP) == -1 && errno == EADDRNOTAVAIL);
  send_to(tx, GROUP, port, "b only");
  CHECK(!strcmp(got(b, 1000, 0), "b only"));
  CHECK(!strcmp(got(a, 1000, 0), "b only")); /* not a member, but IP_MULTICAST_ALL */
  CHECK(membership(b, IP_DROP_MEMBERSHIP, GROUP) == 0);
  send_to(tx, GROUP, port, "nobody");
  CHECK(!*got(a, 200, 0) && !*got(b, 200, 0));
  puts("leave: ok");

  /* Bound to the group's address: that group's datagrams, and no unicast. */
  int g = udp(port + 1, GROUP);
  struct group_req gr = {0};
  struct sockaddr_in *ga = (struct sockaddr_in *)&gr.gr_group;
  ga->sin_family = AF_INET;
  inet_pton(AF_INET, GROUP, &ga->sin_addr);
  CHECK(setsockopt(g, IPPROTO_IP, MCAST_JOIN_GROUP, &gr, sizeof gr) == 0);
  send_to(tx, GROUP, port + 1, "group");
  CHECK(!strcmp(got(g, 1000, 0), "group"));
  send_to(tx, "127.0.0.1", port + 1, "unicast");
  CHECK(!*got(g, 200, 0));
  CHECK(setsockopt(g, IPPROTO_IP, MCAST_LEAVE_GROUP, &gr, sizeof gr) == 0);
  puts("bound to the group, and MCAST_JOIN_GROUP: ok");

  /* A connected socket sends to the group with plain send(). */
  int d = udp(port + 2, "0.0.0.0");
  CHECK(membership(d, IP_ADD_MEMBERSHIP, GROUP) == 0);
  int cs = socket(AF_INET, SOCK_DGRAM, 0);
  struct sockaddr_in to = {.sin_family = AF_INET, .sin_port = htons(port + 2)};
  inet_pton(AF_INET, GROUP, &to.sin_addr);
  CHECK(connect(cs, (struct sockaddr *)&to, sizeof to) == 0);
  CHECK(send(cs, "conn", 4, 0) == 4);
  CHECK(!strcmp(got(d, 1000, 0), "conn"));
  puts("connected to a group: ok");

  /* Memberships are for datagram sockets. */
  int t = socket(AF_INET, SOCK_STREAM, 0);
  CHECK(membership(t, IP_ADD_MEMBERSHIP, GROUP) == -1 && errno == EPROTO);
  puts("not on a TCP socket: ok");

  /* IPv6: IPV6_JOIN_GROUP / IPV6_LEAVE_GROUP. */
  int r6 = socket(AF_INET6, SOCK_DGRAM, 0), t6 = socket(AF_INET6, SOCK_DGRAM, 0);
  struct sockaddr_in6 a6 = {.sin6_family = AF_INET6, .sin6_port = htons(port + 3)};
  CHECK(bind(r6, (struct sockaddr *)&a6, sizeof a6) == 0);
  struct ipv6_mreq m6 = {0};
  inet_pton(AF_INET6, "ff15::77", &m6.ipv6mr_multiaddr);
  CHECK(setsockopt(r6, IPPROTO_IPV6, IPV6_JOIN_GROUP, &m6, sizeof m6) == 0);
  inet_pton(AF_INET6, "ff15::77", &a6.sin6_addr);
  CHECK(sendto(t6, "six", 3, 0, (struct sockaddr *)&a6, sizeof a6) == 3);
  CHECK(!strcmp(got(r6, 1000, 0), "six"));
  CHECK(setsockopt(r6, IPPROTO_IPV6, IPV6_LEAVE_GROUP, &m6, sizeof m6) == 0);
  CHECK(sendto(t6, "gone", 4, 0, (struct sockaddr *)&a6, sizeof a6) == 4);
  CHECK(!*got(r6, 200, 0));
  puts("IPv6 groups: ok");

  puts("PASS");
  return 0;
}
