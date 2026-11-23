/* Ancillary data on AF_UNIX datagram and seqpacket sockets, as Linux's
 * af_unix.c and scm.c have it: SCM_RIGHTS descriptors ride the one datagram
 * they were sent with (never merged with the next), on socketpairs and on
 * named (path or abstract) sockets, connected or not; SCM_CREDENTIALS with
 * SO_PASSCRED; MSG_CTRUNC closing what doesn't fit; MSG_CMSG_CLOEXEC; the
 * SCM_MAX_FD limit; EBADF; empty datagrams that carry descriptors; in-flight
 * descriptors living until received or their queue goes away, a socket sent
 * over itself included; sendmmsg/recvmmsg with control data; and SCM_PIDFD
 * with SO_PASSPIDFD (after the other messages, whole or not at all, for a
 * sender alive or gone) and SO_PEERPIDFD; and SO_PASSRIGHTS (Linux 6.16): on
 * by default, each end's own, an accepted socket's from its listener, and
 * when it is off the sender of descriptors gets EPERM, on stream, datagram
 * and seqpacket sockets alike. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef SO_PASSPIDFD
#define SO_PASSPIDFD 76
#endif
#ifndef SO_PEERPIDFD
#define SO_PEERPIDFD 77
#endif
#ifndef SCM_PIDFD
#define SCM_PIDFD 0x04
#endif
#ifndef P_PIDFD
#define P_PIDFD 3
#endif
#ifndef SO_PASSRIGHTS
#define SO_PASSRIGHTS 83
#endif

#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d, errno %d)\n", #c, __LINE__, errno); exit(1); } } while (0)

#define MAXFD 260

/* Send `len` bytes of `data` with `n` descriptors (none if 0) to `to`. */
static ssize_t send_fds(int s, const void *data, size_t len, const int *fds, int n,
                        const struct sockaddr_un *to, socklen_t tolen)
{
  static char cbuf[CMSG_SPACE(MAXFD * sizeof(int))];
  struct iovec io = {(void *)data, len};
  struct msghdr m = {.msg_iov = &io, .msg_iovlen = 1};
  if (to) {
    m.msg_name = (void *)to;
    m.msg_namelen = tolen;
  }
  if (n > 0) {
    memset(cbuf, 0, sizeof cbuf);
    m.msg_control = cbuf;
    m.msg_controllen = CMSG_SPACE(n * sizeof(int));
    struct cmsghdr *c = CMSG_FIRSTHDR(&m);
    c->cmsg_level = SOL_SOCKET;
    c->cmsg_type = SCM_RIGHTS;
    c->cmsg_len = CMSG_LEN(n * sizeof(int));
    memcpy(CMSG_DATA(c), fds, n * sizeof(int));
  }
  return sendmsg(s, &m, 0);
}

struct got {
  ssize_t n;
  int flags;
  int nfds;
  int fds[MAXFD];
  size_t controllen;
  int has_cred;
  struct ucred cred;
  /* SCM_PIDFD's descriptor, or -1000 for none. */
  int pidfd;
  /* The control messages' types, in order. */
  int types[8];
  int ntypes;
  struct sockaddr_un from;
  socklen_t fromlen;
};

/* One recvmsg into `buf`, with a control buffer of `clen` bytes (no control
 * buffer at all if negative). */
static struct got recv_fds(int s, void *buf, size_t len, ssize_t clen, int flags)
{
  static char cbuf[CMSG_SPACE(MAXFD * sizeof(int)) + 64] __attribute__((aligned(8)));
  struct got g;
  memset(&g, 0, sizeof g);
  g.pidfd = -1000;
  struct iovec io = {buf, len};
  struct msghdr m = {.msg_iov = &io, .msg_iovlen = 1};
  m.msg_name = &g.from;
  m.msg_namelen = sizeof g.from;
  memset(cbuf, 0xaa, sizeof cbuf);
  if (clen >= 0) {
    m.msg_control = cbuf;
    m.msg_controllen = clen;
  }
  g.n = recvmsg(s, &m, flags);
  g.flags = m.msg_flags;
  g.controllen = m.msg_controllen;
  g.fromlen = m.msg_namelen;
  if (g.n < 0 || clen < 0)
    return g;
  for (struct cmsghdr *c = CMSG_FIRSTHDR(&m); c; c = CMSG_NXTHDR(&m, c)) {
    if (c->cmsg_level != SOL_SOCKET)
      continue;
    if (g.ntypes < 8)
      g.types[g.ntypes++] = c->cmsg_type;
    if (c->cmsg_type == SCM_PIDFD && c->cmsg_len == CMSG_LEN(sizeof(int)))
      memcpy(&g.pidfd, CMSG_DATA(c), sizeof(int));
    else if (c->cmsg_type == SCM_RIGHTS && c->cmsg_len >= CMSG_LEN(0)) {
      int k = (c->cmsg_len - CMSG_LEN(0)) / sizeof(int);
      memcpy(g.fds + g.nfds, CMSG_DATA(c), k * sizeof(int));
      g.nfds += k;
    } else if (c->cmsg_type == SCM_CREDENTIALS && c->cmsg_len == CMSG_LEN(sizeof(struct ucred))) {
      g.has_cred = 1;
      memcpy(&g.cred, CMSG_DATA(c), sizeof g.cred);
    }
  }
  return g;
}

/* Whether writing through `w` shows up on pipe read end `r`. */
static int same_pipe(int w, int r)
{
  char c = 'x';
  if (write(w, &c, 1) != 1)
    return 0;
  struct pollfd p = {r, POLLIN, 0};
  if (poll(&p, 1, 1000) != 1)
    return 0;
  return read(r, &c, 1) == 1 && c == 'x';
}

/* Whether pipe read end `r` sees EOF (every write end closed) within a while:
 * Linux's unix garbage collector runs from a workqueue. */
static int pipe_eof(int r)
{
  struct pollfd p = {r, POLLIN, 0};
  if (poll(&p, 1, 3000) != 1)
    return 0;
  char c;
  return read(r, &c, 1) == 0;
}

/* Linux collects unix socket cycles when some unix socket is released, from a
 * workqueue: release one, for that to happen. */
static void kick_gc(void)
{
  close(socket(AF_UNIX, SOCK_DGRAM, 0));
}

static int nonblocking_pipe_empty(int r)
{
  struct pollfd p = {r, POLLIN, 0};
  return poll(&p, 1, 50) == 0;
}

static int cloexec(int fd)
{
  return fcntl(fd, F_GETFD) & FD_CLOEXEC;
}

static socklen_t abstract(struct sockaddr_un *a, const char *tag)
{
  memset(a, 0, sizeof *a);
  a->sun_family = AF_UNIX;
  snprintf(a->sun_path + 1, sizeof a->sun_path - 1, "nixvm-unix-scm-%s-%d", tag, getpid());
  return offsetof(struct sockaddr_un, sun_path) + 1 + strlen(a->sun_path + 1);
}

static void test_dgram_pair(void)
{
  int sv[2], p[2];
  char buf[64];
  CHECK(socketpair(AF_UNIX, SOCK_DGRAM, 0, sv) == 0);
  CHECK(pipe(p) == 0);
  int two[2] = {p[1], p[0]};
  CHECK(send_fds(sv[0], "one", 3, &p[1], 1, NULL, 0) == 3);
  CHECK(send_fds(sv[0], "two", 3, NULL, 0, NULL, 0) == 3);
  CHECK(send_fds(sv[0], "three", 5, two, 2, NULL, 0) == 5);
  /* A buffer far bigger than any one of them: each still comes alone, with
   * just its own descriptors. */
  struct got g = recv_fds(sv[1], buf, sizeof buf, CMSG_SPACE(8 * sizeof(int)), 0);
  CHECK(g.n == 3 && memcmp(buf, "one", 3) == 0 && g.nfds == 1 && !(g.flags & MSG_CTRUNC));
  CHECK(g.controllen == CMSG_SPACE(sizeof(int)));
  CHECK(!cloexec(g.fds[0]) && same_pipe(g.fds[0], p[0]));
  close(g.fds[0]);
  g = recv_fds(sv[1], buf, sizeof buf, CMSG_SPACE(8 * sizeof(int)), 0);
  CHECK(g.n == 3 && memcmp(buf, "two", 3) == 0 && g.nfds == 0 && g.controllen == 0 && g.flags == 0);
  g = recv_fds(sv[1], buf, sizeof buf, CMSG_SPACE(8 * sizeof(int)), MSG_CMSG_CLOEXEC);
  CHECK(g.n == 5 && g.nfds == 2 && g.controllen == CMSG_SPACE(2 * sizeof(int)));
  CHECK(cloexec(g.fds[0]) && cloexec(g.fds[1]));
  CHECK(same_pipe(g.fds[0], g.fds[1]) && same_pipe(p[1], g.fds[1]));
  close(g.fds[0]);
  close(g.fds[1]);
  printf("socketpair dgram: fds ride their own datagram, MSG_CMSG_CLOEXEC\n");

  /* A datagram cut short still brings its descriptors. */
  CHECK(send_fds(sv[1], "truncated", 9, &p[1], 1, NULL, 0) == 9);
  g = recv_fds(sv[0], buf, 4, CMSG_SPACE(sizeof(int)), 0);
  CHECK(g.n == 4 && (g.flags & MSG_TRUNC) && g.nfds == 1);
  close(g.fds[0]);
  CHECK(send_fds(sv[1], "truncated", 9, &p[1], 1, NULL, 0) == 9);
  g = recv_fds(sv[0], buf, 4, CMSG_SPACE(sizeof(int)), MSG_TRUNC);
  CHECK(g.n == 9 && (g.flags & MSG_TRUNC) && g.nfds == 1);
  close(g.fds[0]);

  /* No data at all, but descriptors. */
  CHECK(send_fds(sv[0], "", 0, &p[1], 1, NULL, 0) == 0);
  CHECK(send_fds(sv[0], "next", 4, NULL, 0, NULL, 0) == 4);
  g = recv_fds(sv[1], buf, sizeof buf, CMSG_SPACE(sizeof(int)), 0);
  CHECK(g.n == 0 && g.nfds == 1 && same_pipe(g.fds[0], p[0]));
  close(g.fds[0]);
  g = recv_fds(sv[1], buf, sizeof buf, CMSG_SPACE(sizeof(int)), 0);
  CHECK(g.n == 4 && g.nfds == 0);
  printf("empty datagrams carry fds, truncated ones keep theirs\n");

  /* MSG_PEEK hands out copies and leaves the originals queued. */
  CHECK(send_fds(sv[0], "peek", 4, &p[1], 1, NULL, 0) == 4);
  g = recv_fds(sv[1], buf, sizeof buf, CMSG_SPACE(sizeof(int)), MSG_PEEK);
  CHECK(g.n == 4 && g.nfds == 1 && same_pipe(g.fds[0], p[0]));
  int peeked = g.fds[0];
  g = recv_fds(sv[1], buf, sizeof buf, CMSG_SPACE(sizeof(int)), 0);
  CHECK(g.n == 4 && g.nfds == 1 && g.fds[0] != peeked && same_pipe(g.fds[0], p[0]));
  close(peeked);
  close(g.fds[0]);

  /* A plain read() of a datagram that had descriptors drops them. */
  CHECK(send_fds(sv[0], "read", 4, &p[1], 1, NULL, 0) == 4);
  CHECK(read(sv[1], buf, sizeof buf) == 4);
  CHECK(send_fds(sv[0], "recv", 4, &p[1], 1, NULL, 0) == 4);
  CHECK(recv(sv[1], buf, sizeof buf, 0) == 4);
  close(sv[0]);
  close(sv[1]);
  close(p[0]);
  close(p[1]);
}

static void test_ctrunc(void)
{
  int sv[2], p[2];
  char buf[16];
  CHECK(socketpair(AF_UNIX, SOCK_DGRAM, 0, sv) == 0);
  CHECK(pipe(p) == 0);
  int three[3] = {p[1], p[1], p[1]};
  CHECK(send_fds(sv[0], "a", 1, three, 3, NULL, 0) == 1);
  CHECK(send_fds(sv[0], "b", 1, three, 3, NULL, 0) == 1);
  CHECK(send_fds(sv[0], "c", 1, three, 3, NULL, 0) == 1);
  close(p[1]);
  /* Room for one: the other two are closed. */
  struct got g = recv_fds(sv[1], buf, sizeof buf, CMSG_LEN(sizeof(int)), 0);
  CHECK(g.n == 1 && g.nfds == 1 && (g.flags & MSG_CTRUNC));
  CHECK(g.controllen == CMSG_LEN(sizeof(int)));
  int kept = g.fds[0];
  /* No control buffer: all three are closed. */
  g = recv_fds(sv[1], buf, sizeof buf, -1, 0);
  CHECK(g.n == 1 && (g.flags & MSG_CTRUNC) && g.controllen == 0);
  /* A buffer too small for even a header: likewise. */
  g = recv_fds(sv[1], buf, sizeof buf, sizeof(struct cmsghdr), 0);
  CHECK(g.n == 1 && (g.flags & MSG_CTRUNC) && g.nfds == 0);
  CHECK(nonblocking_pipe_empty(p[0]));
  close(kept);
  CHECK(pipe_eof(p[0]));
  close(p[0]);
  close(sv[0]);
  close(sv[1]);
  printf("MSG_CTRUNC closes the fds that don't fit\n");
}

static void test_limits(void)
{
  int sv[2], p[2];
  char buf[16];
  static int many[MAXFD];
  CHECK(socketpair(AF_UNIX, SOCK_DGRAM, 0, sv) == 0);
  CHECK(pipe(p) == 0);
  for (int i = 0; i < 254; i++)
    many[i] = p[1];
  errno = 0;
  CHECK(send_fds(sv[0], "x", 1, many, 254, NULL, 0) == -1 && errno == EINVAL);
  CHECK(send_fds(sv[0], "x", 1, many, 253, NULL, 0) == 1);
  struct got g = recv_fds(sv[1], buf, sizeof buf, CMSG_SPACE(MAXFD * sizeof(int)), 0);
  CHECK(g.n == 1 && g.nfds == 253 && !(g.flags & MSG_CTRUNC));
  for (int i = 0; i < g.nfds; i++)
    close(g.fds[i]);

  /* Two SCM_RIGHTS messages in one send: one list, limited as a whole. */
  char cbuf[CMSG_SPACE(200 * sizeof(int)) + CMSG_SPACE(60 * sizeof(int))] __attribute__((aligned(8)));
  memset(cbuf, 0, sizeof cbuf);
  struct iovec io = {"y", 1};
  struct msghdr m = {.msg_iov = &io, .msg_iovlen = 1, .msg_control = cbuf};
  m.msg_controllen = CMSG_SPACE(1 * sizeof(int)) + CMSG_SPACE(2 * sizeof(int));
  struct cmsghdr *c = CMSG_FIRSTHDR(&m);
  c->cmsg_level = SOL_SOCKET;
  c->cmsg_type = SCM_RIGHTS;
  c->cmsg_len = CMSG_LEN(sizeof(int));
  memcpy(CMSG_DATA(c), many, sizeof(int));
  c = CMSG_NXTHDR(&m, c);
  c->cmsg_level = SOL_SOCKET;
  c->cmsg_type = SCM_RIGHTS;
  c->cmsg_len = CMSG_LEN(2 * sizeof(int));
  memcpy(CMSG_DATA(c), many, 2 * sizeof(int));
  CHECK(sendmsg(sv[0], &m, 0) == 1);
  g = recv_fds(sv[1], buf, sizeof buf, CMSG_SPACE(8 * sizeof(int)), 0);
  CHECK(g.n == 1 && g.nfds == 3 && g.controllen == CMSG_SPACE(3 * sizeof(int)));
  for (int i = 0; i < g.nfds; i++)
    close(g.fds[i]);
  memset(cbuf, 0, sizeof cbuf);
  m.msg_controllen = CMSG_SPACE(200 * sizeof(int)) + CMSG_SPACE(60 * sizeof(int));
  c = CMSG_FIRSTHDR(&m);
  c->cmsg_level = SOL_SOCKET;
  c->cmsg_type = SCM_RIGHTS;
  c->cmsg_len = CMSG_LEN(200 * sizeof(int));
  memcpy(CMSG_DATA(c), many, 200 * sizeof(int));
  c = CMSG_NXTHDR(&m, c);
  c->cmsg_level = SOL_SOCKET;
  c->cmsg_type = SCM_RIGHTS;
  c->cmsg_len = CMSG_LEN(60 * sizeof(int));
  memcpy(CMSG_DATA(c), many, 60 * sizeof(int));
  errno = 0;
  CHECK(sendmsg(sv[0], &m, 0) == -1 && errno == EINVAL);

  /* A bad descriptor fails the whole send; nothing is queued. */
  int bad[2] = {p[1], 999};
  errno = 0;
  CHECK(send_fds(sv[0], "z", 1, bad, 2, NULL, 0) == -1 && errno == EBADF);
  /* So does a malformed control message, or one Linux doesn't know. */
  memset(cbuf, 0, sizeof cbuf);
  m.msg_controllen = CMSG_SPACE(sizeof(int));
  c = CMSG_FIRSTHDR(&m);
  c->cmsg_level = SOL_SOCKET;
  c->cmsg_type = 99;
  c->cmsg_len = CMSG_LEN(sizeof(int));
  errno = 0;
  CHECK(sendmsg(sv[0], &m, 0) == -1 && errno == EINVAL);
  c->cmsg_type = SCM_RIGHTS;
  c->cmsg_len = CMSG_SPACE(sizeof(int)) + 8;
  errno = 0;
  CHECK(sendmsg(sv[0], &m, 0) == -1 && errno == EINVAL);
  errno = 0;
  CHECK(recv(sv[1], buf, sizeof buf, MSG_DONTWAIT) == -1 && errno == EAGAIN);
  close(p[0]);
  close(p[1]);
  close(sv[0]);
  close(sv[1]);
  printf("SCM_MAX_FD is 253, EBADF and EINVAL send nothing\n");
}

static void test_named(const char *dir)
{
  char buf[64];
  int p[2];
  CHECK(pipe(p) == 0);
  struct sockaddr_un path = {.sun_family = AF_UNIX};
  snprintf(path.sun_path, sizeof path.sun_path, "%s/recv.sock", dir);
  socklen_t pathlen = offsetof(struct sockaddr_un, sun_path) + strlen(path.sun_path) + 1;
  int r = socket(AF_UNIX, SOCK_DGRAM, 0);
  CHECK(bind(r, (struct sockaddr *)&path, pathlen) == 0);

  /* Unconnected, unbound sender, to a path: the source has no name. */
  int s = socket(AF_UNIX, SOCK_DGRAM, 0);
  CHECK(send_fds(s, "path", 4, &p[1], 1, &path, pathlen) == 4);
  CHECK(send_fds(s, "plain", 5, NULL, 0, &path, pathlen) == 5);
  struct got g = recv_fds(r, buf, sizeof buf, CMSG_SPACE(4 * sizeof(int)), 0);
  CHECK(g.n == 4 && g.nfds == 1 && same_pipe(g.fds[0], p[0]));
  CHECK(g.fromlen == 0);
  close(g.fds[0]);
  g = recv_fds(r, buf, sizeof buf, CMSG_SPACE(4 * sizeof(int)), 0);
  CHECK(g.n == 5 && g.nfds == 0);

  /* A sender bound to an abstract name is seen by that name. */
  struct sockaddr_un sa;
  socklen_t salen = abstract(&sa, "sender");
  CHECK(bind(s, (struct sockaddr *)&sa, salen) == 0);
  CHECK(send_fds(s, "named", 5, &p[1], 1, &path, pathlen) == 5);
  g = recv_fds(r, buf, sizeof buf, CMSG_SPACE(4 * sizeof(int)), 0);
  CHECK(g.n == 5 && g.nfds == 1);
  CHECK(g.fromlen == salen && memcmp(&g.from, &sa, salen) == 0);
  close(g.fds[0]);

  /* To an abstract name. */
  struct sockaddr_un ra;
  socklen_t ralen = abstract(&ra, "recv");
  int r2 = socket(AF_UNIX, SOCK_DGRAM, 0);
  CHECK(bind(r2, (struct sockaddr *)&ra, ralen) == 0);
  CHECK(send_fds(s, "abs", 3, &p[1], 1, &ra, ralen) == 3);
  g = recv_fds(r2, buf, sizeof buf, CMSG_SPACE(4 * sizeof(int)), 0);
  CHECK(g.n == 3 && g.nfds == 1 && same_pipe(g.fds[0], p[0]));
  close(g.fds[0]);

  /* Connected to a path: send() needs no address; getpeername names it. */
  int c = socket(AF_UNIX, SOCK_DGRAM, 0);
  CHECK(connect(c, (struct sockaddr *)&path, pathlen) == 0);
  struct sockaddr_un peer;
  socklen_t peerlen = sizeof peer;
  CHECK(getpeername(c, (struct sockaddr *)&peer, &peerlen) == 0);
  CHECK(peerlen == pathlen && strcmp(peer.sun_path, path.sun_path) == 0);
  CHECK(send_fds(c, "conn", 4, &p[1], 1, NULL, 0) == 4);
  CHECK(send(c, "more", 4, 0) == 4);
  g = recv_fds(r, buf, sizeof buf, CMSG_SPACE(4 * sizeof(int)), 0);
  CHECK(g.n == 4 && memcmp(buf, "conn", 4) == 0 && g.nfds == 1);
  close(g.fds[0]);
  g = recv_fds(r, buf, sizeof buf, CMSG_SPACE(4 * sizeof(int)), 0);
  CHECK(g.n == 4 && memcmp(buf, "more", 4) == 0 && g.nfds == 0);

  /* A socket connected elsewhere takes datagrams only from its peer. */
  CHECK(connect(r2, (struct sockaddr *)&path, pathlen) == 0);
  errno = 0;
  CHECK(send_fds(c, "no", 2, NULL, 0, &ra, ralen) == -1 && errno == EPERM);

  /* Nobody there. */
  struct sockaddr_un none;
  socklen_t nonelen = abstract(&none, "nobody");
  errno = 0;
  CHECK(send_fds(s, "x", 1, &p[1], 1, &none, nonelen) == -1 && errno == ECONNREFUSED);
  struct sockaddr_un missing = {.sun_family = AF_UNIX};
  snprintf(missing.sun_path, sizeof missing.sun_path, "%s/missing.sock", dir);
  errno = 0;
  CHECK(sendto(s, "x", 1, 0, (struct sockaddr *)&missing, sizeof missing) == -1 && errno == ENOENT);
  errno = 0;
  CHECK(connect(c, (struct sockaddr *)&none, nonelen) == -1 && errno == ECONNREFUSED);

  /* The receiver goes away: its connected sender is refused, once. */
  close(r);
  errno = 0;
  CHECK(send(c, "gone", 4, 0) == -1 && errno == ECONNREFUSED);
  errno = 0;
  CHECK(send(c, "gone", 4, 0) == -1 && errno == ENOTCONN);
  unlink(path.sun_path);
  close(c);
  close(s);
  close(r2);
  close(p[0]);
  close(p[1]);
  printf("named dgram sockets: path and abstract, connected and not\n");
}

static void test_seqpacket(void)
{
  char buf[64];
  int sv[2], p[2];
  CHECK(pipe(p) == 0);
  CHECK(socketpair(AF_UNIX, SOCK_SEQPACKET, 0, sv) == 0);
  CHECK(send_fds(sv[0], "", 0, &p[1], 1, NULL, 0) == 0);
  int two[2] = {p[0], p[1]};
  CHECK(send_fds(sv[0], "second", 6, two, 2, NULL, 0) == 6);
  CHECK(send_fds(sv[0], "third", 5, NULL, 0, NULL, 0) == 5);
  struct got g = recv_fds(sv[1], buf, sizeof buf, CMSG_SPACE(8 * sizeof(int)), 0);
  CHECK(g.n == 0 && g.nfds == 1 && same_pipe(g.fds[0], p[0]));
  close(g.fds[0]);
  g = recv_fds(sv[1], buf, 3, CMSG_SPACE(8 * sizeof(int)), 0);
  CHECK(g.n == 3 && (g.flags & MSG_TRUNC) && g.nfds == 2);
  close(g.fds[0]);
  close(g.fds[1]);
  g = recv_fds(sv[1], buf, sizeof buf, CMSG_SPACE(8 * sizeof(int)), 0);
  CHECK(g.n == 5 && g.nfds == 0 && memcmp(buf, "third", 5) == 0);
  close(sv[0]);
  close(sv[1]);

  /* A named listener and its accepted connection. */
  struct sockaddr_un a;
  socklen_t alen = abstract(&a, "seqpacket");
  int l = socket(AF_UNIX, SOCK_SEQPACKET, 0);
  CHECK(l >= 0 && bind(l, (struct sockaddr *)&a, alen) == 0 && listen(l, 4) == 0);
  int type = 0;
  socklen_t tl = sizeof type;
  CHECK(getsockopt(l, SOL_SOCKET, SO_TYPE, &type, &tl) == 0 && type == SOCK_SEQPACKET);
  int c = socket(AF_UNIX, SOCK_SEQPACKET, 0);
  CHECK(connect(c, (struct sockaddr *)&a, alen) == 0);
  int s = accept(l, NULL, NULL);
  CHECK(s >= 0);
  CHECK(send_fds(c, "hello", 5, &p[1], 1, NULL, 0) == 5);
  CHECK(send_fds(c, "world", 5, &p[1], 1, NULL, 0) == 5);
  g = recv_fds(s, buf, sizeof buf, CMSG_SPACE(8 * sizeof(int)), 0);
  CHECK(g.n == 5 && memcmp(buf, "hello", 5) == 0 && g.nfds == 1);
  close(g.fds[0]);
  g = recv_fds(s, buf, sizeof buf, CMSG_SPACE(8 * sizeof(int)), 0);
  CHECK(g.n == 5 && memcmp(buf, "world", 5) == 0 && g.nfds == 1);
  close(g.fds[0]);
  /* A stream socket finds no such abstract name: they are per type. */
  int st = socket(AF_UNIX, SOCK_STREAM, 0);
  errno = 0;
  CHECK(connect(st, (struct sockaddr *)&a, alen) == -1 && errno == ECONNREFUSED);
  close(st);
  close(c);
  CHECK(recv(s, buf, sizeof buf, 0) == 0);
  close(s);
  close(l);
  close(p[0]);
  close(p[1]);
  printf("seqpacket: socketpair and listener, one message's fds each\n");
}

static void test_creds(void)
{
  char buf[16];
  int sv[2], p[2];
  CHECK(pipe(p) == 0);
  CHECK(socketpair(AF_UNIX, SOCK_DGRAM, 0, sv) == 0);
  int on = 1;
  CHECK(setsockopt(sv[1], SOL_SOCKET, SO_PASSCRED, &on, sizeof on) == 0);
  /* The sender asked for nothing: the receiver's SO_PASSCRED still gets them. */
  CHECK(send(sv[0], "c", 1, 0) == 1);
  struct got g = recv_fds(sv[1], buf, sizeof buf, 256, 0);
  CHECK(g.n == 1 && g.has_cred && g.nfds == 0);
  CHECK(g.cred.pid == getpid() && g.cred.uid == getuid() && g.cred.gid == getgid());
  CHECK(g.controllen == CMSG_SPACE(sizeof(struct ucred)));
  /* Credentials first, then the descriptors. */
  CHECK(send_fds(sv[0], "", 0, &p[1], 1, NULL, 0) == 0);
  g = recv_fds(sv[1], buf, sizeof buf, 256, 0);
  CHECK(g.n == 0 && g.has_cred && g.nfds == 1);
  CHECK(g.controllen == CMSG_SPACE(sizeof(struct ucred)) + CMSG_SPACE(sizeof(int)));
  close(g.fds[0]);
  /* Room for the credentials but not the descriptor. */
  CHECK(send_fds(sv[0], "f", 1, &p[1], 1, NULL, 0) == 1);
  g = recv_fds(sv[1], buf, sizeof buf, CMSG_SPACE(sizeof(struct ucred)), 0);
  CHECK(g.n == 1 && g.has_cred && g.nfds == 0 && (g.flags & MSG_CTRUNC));

  /* Explicit credentials: one's own are fine. */
  char cbuf[CMSG_SPACE(sizeof(struct ucred))] __attribute__((aligned(8)));
  memset(cbuf, 0, sizeof cbuf);
  struct iovec io = {"e", 1};
  struct msghdr m = {.msg_iov = &io, .msg_iovlen = 1, .msg_control = cbuf, .msg_controllen = sizeof cbuf};
  struct cmsghdr *c = CMSG_FIRSTHDR(&m);
  c->cmsg_level = SOL_SOCKET;
  c->cmsg_type = SCM_CREDENTIALS;
  c->cmsg_len = CMSG_LEN(sizeof(struct ucred));
  struct ucred me = {getpid(), getuid(), getgid()};
  memcpy(CMSG_DATA(c), &me, sizeof me);
  CHECK(sendmsg(sv[0], &m, 0) == 1);
  g = recv_fds(sv[1], buf, sizeof buf, 256, 0);
  CHECK(g.n == 1 && g.has_cred && g.cred.pid == getpid() && g.cred.uid == getuid());
  if (geteuid() != 0) {
    struct ucred other = {getpid(), getuid() + 1, getgid()};
    memcpy(CMSG_DATA(c), &other, sizeof other);
    errno = 0;
    CHECK(sendmsg(sv[0], &m, 0) == -1 && errno == EPERM);
  }
  close(sv[0]);
  close(sv[1]);

  /* An unbound sender with SO_PASSCRED is bound to an abstract name of its own. */
  struct sockaddr_un ra;
  socklen_t ralen = abstract(&ra, "creds");
  int r = socket(AF_UNIX, SOCK_DGRAM, 0);
  CHECK(bind(r, (struct sockaddr *)&ra, ralen) == 0);
  int s = socket(AF_UNIX, SOCK_DGRAM, 0);
  CHECK(setsockopt(s, SOL_SOCKET, SO_PASSCRED, &on, sizeof on) == 0);
  CHECK(sendto(s, "a", 1, 0, (struct sockaddr *)&ra, ralen) == 1);
  struct sockaddr_un me_a;
  socklen_t me_len = sizeof me_a;
  CHECK(getsockname(s, (struct sockaddr *)&me_a, &me_len) == 0);
  CHECK(me_len == offsetof(struct sockaddr_un, sun_path) + 6 && me_a.sun_path[0] == 0);
  g = recv_fds(r, buf, sizeof buf, 256, 0);
  CHECK(g.n == 1 && !g.has_cred && g.fromlen == me_len && memcmp(&g.from, &me_a, me_len) == 0);
  close(s);
  close(r);
  close(p[0]);
  close(p[1]);
  printf("SCM_CREDENTIALS with SO_PASSCRED, and autobind\n");
}

/* Whether pidfd `fd` is for a live process (pidfd_send_signal of 0). */
static int pidfd_alive(int fd)
{
  return syscall(424, fd, 0, NULL, 0) == 0;
}

/* Whether pidfd `fd` reads as exited (POLLIN). */
static int pidfd_exited(int fd)
{
  struct pollfd p = {fd, POLLIN, 0};
  return poll(&p, 1, 0) == 1 && (p.revents & POLLIN);
}

/* The lowest free descriptor number, to see nothing was left installed. */
static int next_fd(void)
{
  int fd = dup(0);
  close(fd);
  return fd;
}

static void test_pidfd(void)
{
  char buf[16];
  int sv[2], p[2], on = 1, v = -1;
  socklen_t vl = sizeof v;
  CHECK(pipe(p) == 0);
  CHECK(socketpair(AF_UNIX, SOCK_DGRAM, 0, sv) == 0);
  CHECK(getsockopt(sv[1], SOL_SOCKET, SO_PASSPIDFD, &v, &vl) == 0 && v == 0 && vl == sizeof v);
  CHECK(setsockopt(sv[1], SOL_SOCKET, SO_PASSPIDFD, &on, sizeof on) == 0);
  CHECK(getsockopt(sv[1], SOL_SOCKET, SO_PASSPIDFD, &v, &vl) == 0 && v == 1);
  /* SO_PASSPIDFD alone: an SCM_PIDFD for the sender, and no credentials. */
  CHECK(send(sv[0], "p", 1, 0) == 1);
  struct got g = recv_fds(sv[1], buf, sizeof buf, 256, 0);
  CHECK(g.n == 1 && !g.has_cred && g.nfds == 0 && g.ntypes == 1 && g.pidfd >= 0);
  CHECK(g.controllen == CMSG_SPACE(sizeof(int)) && !(g.flags & MSG_CTRUNC));
  CHECK(pidfd_alive(g.pidfd) && !pidfd_exited(g.pidfd));
  CHECK(fcntl(g.pidfd, F_GETFD) == FD_CLOEXEC);
  close(g.pidfd);
  /* With SO_PASSCRED and descriptors: credentials, descriptors, pidfd. */
  CHECK(setsockopt(sv[1], SOL_SOCKET, SO_PASSCRED, &on, sizeof on) == 0);
  CHECK(send_fds(sv[0], "q", 1, &p[1], 1, NULL, 0) == 1);
  g = recv_fds(sv[1], buf, sizeof buf, 256, 0);
  CHECK(g.n == 1 && g.has_cred && g.nfds == 1 && g.pidfd >= 0 && g.ntypes == 3);
  CHECK(g.types[0] == SCM_CREDENTIALS && g.types[1] == SCM_RIGHTS && g.types[2] == SCM_PIDFD);
  CHECK(g.controllen == CMSG_SPACE(sizeof(struct ucred)) + 2 * CMSG_SPACE(sizeof(int)));
  close(g.fds[0]);
  close(g.pidfd);
  /* Room for the credentials and a pidfd's header but not its int: none
   * made, and MSG_CTRUNC. */
  int before = next_fd();
  CHECK(send(sv[0], "r", 1, 0) == 1);
  g = recv_fds(sv[1], buf, sizeof buf, CMSG_SPACE(sizeof(struct ucred)) + CMSG_LEN(0) + 3, 0);
  CHECK(g.n == 1 && g.has_cred && g.pidfd == -1000 && (g.flags & MSG_CTRUNC));
  CHECK(g.controllen == CMSG_SPACE(sizeof(struct ucred)));
  CHECK(next_fd() == before);
  /* Exactly the room it needs. */
  CHECK(send(sv[0], "s", 1, 0) == 1);
  g = recv_fds(sv[1], buf, sizeof buf, CMSG_SPACE(sizeof(struct ucred)) + CMSG_LEN(sizeof(int)), 0);
  CHECK(g.n == 1 && g.has_cred && g.pidfd >= 0 && !(g.flags & MSG_CTRUNC));
  close(g.pidfd);
  /* No control buffer: MSG_CTRUNC, nothing made. */
  int off = 0;
  CHECK(setsockopt(sv[1], SOL_SOCKET, SO_PASSCRED, &off, sizeof off) == 0);
  CHECK(send(sv[0], "t", 1, 0) == 1);
  g = recv_fds(sv[1], buf, sizeof buf, -1, 0);
  CHECK(g.n == 1 && (g.flags & MSG_CTRUNC));
  CHECK(next_fd() == before);
  /* A datagram another process sent: its pidfd follows it, alive, then
   * exited; and one whose sender has gone (and been reaped) before it was
   * received is for an exited process (older kernels: -ESRCH instead). */
  int go[2];
  CHECK(pipe(go) == 0);
  pid_t child = fork();
  if (child == 0) {
    close(go[1]);
    send(sv[0], "c", 1, 0);
    char c;
    read(go[0], &c, 1);
    _exit(0);
  }
  g = recv_fds(sv[1], buf, sizeof buf, 256, 0);
  CHECK(g.n == 1 && g.pidfd >= 0);
  CHECK(pidfd_alive(g.pidfd) && !pidfd_exited(g.pidfd));
  close(go[1]);
  struct pollfd pp = {g.pidfd, POLLIN, 0};
  CHECK(poll(&pp, 1, 5000) == 1);
  siginfo_t si;
  memset(&si, 0, sizeof si);
  CHECK(waitid(P_PIDFD, g.pidfd, &si, WEXITED) == 0 && si.si_pid == child);
  close(g.pidfd);
  close(go[0]);
  child = fork();
  if (child == 0) {
    send(sv[0], "d", 1, 0);
    _exit(0);
  }
  CHECK(waitpid(child, NULL, 0) == child);
  g = recv_fds(sv[1], buf, sizeof buf, 256, 0);
  CHECK(g.n == 1 && (g.pidfd >= 0 || g.pidfd == -ESRCH));
  if (g.pidfd >= 0) {
    CHECK(pidfd_exited(g.pidfd) && !pidfd_alive(g.pidfd));
    close(g.pidfd);
  }
  close(sv[0]);
  close(sv[1]);

  /* A stream socketpair, and SO_PEERPIDFD on it: the pair's maker. */
  CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
  CHECK(setsockopt(sv[0], SOL_SOCKET, SO_PASSPIDFD, &on, sizeof on) == 0);
  CHECK(send(sv[1], "u", 1, 0) == 1);
  g = recv_fds(sv[0], buf, sizeof buf, 256, 0);
  CHECK(g.n == 1 && g.pidfd >= 0 && pidfd_alive(g.pidfd));
  close(g.pidfd);
  int pfd = -1;
  socklen_t pl = sizeof pfd;
  CHECK(getsockopt(sv[0], SOL_SOCKET, SO_PEERPIDFD, &pfd, &pl) == 0 && pfd >= 0 && pl == sizeof pfd);
  CHECK(pidfd_alive(pfd) && fcntl(pfd, F_GETFD) == FD_CLOEXEC);
  close(pfd);
  errno = 0;
  CHECK(setsockopt(sv[0], SOL_SOCKET, SO_PEERPIDFD, &on, sizeof on) == -1 && errno == ENOPROTOOPT);
  close(sv[0]);
  close(sv[1]);
  /* A datagram socketpair has a peer too; a connected datagram socket
   * doesn't, nor does one never connected. */
  CHECK(socketpair(AF_UNIX, SOCK_DGRAM, 0, sv) == 0);
  pl = sizeof pfd;
  CHECK(getsockopt(sv[1], SOL_SOCKET, SO_PEERPIDFD, &pfd, &pl) == 0 && pfd >= 0);
  close(pfd);
  struct ucred uc;
  socklen_t ul = sizeof uc;
  CHECK(getsockopt(sv[1], SOL_SOCKET, SO_PEERCRED, &uc, &ul) == 0 && uc.pid == getpid());
  close(sv[0]);
  close(sv[1]);
  int u = socket(AF_UNIX, SOCK_STREAM, 0);
  errno = 0;
  pl = sizeof pfd;
  CHECK(getsockopt(u, SOL_SOCKET, SO_PEERPIDFD, &pfd, &pl) == -1 && errno == ENODATA);
  close(u);

  /* Over a named socket: the connecting side's peer is whoever listened. */
  struct sockaddr_un sa;
  socklen_t salen = abstract(&sa, "peerpidfd");
  int ready[2], done[2];
  CHECK(pipe(ready) == 0 && pipe(done) == 0);
  child = fork();
  if (child == 0) {
    close(done[1]);
    int l = socket(AF_UNIX, SOCK_STREAM, 0);
    if (bind(l, (struct sockaddr *)&sa, salen) != 0 || listen(l, 4) != 0)
      _exit(1);
    /* A listening socket's peer is its own listen() caller. */
    int self = -1;
    socklen_t sl = sizeof self;
    if (getsockopt(l, SOL_SOCKET, SO_PEERPIDFD, &self, &sl) != 0 || !pidfd_alive(self))
      _exit(2);
    char c = 1;
    write(ready[1], &c, 1);
    int a = accept(l, NULL, NULL);
    /* The accepting side's peer is the connector (the parent). */
    int peer = -1;
    sl = sizeof peer;
    if (a < 0 || getsockopt(a, SOL_SOCKET, SO_PEERPIDFD, &peer, &sl) != 0 || !pidfd_alive(peer))
      _exit(3);
    read(done[0], &c, 1);
    _exit(0);
  }
  char c;
  CHECK(read(ready[0], &c, 1) == 1);
  int cl = socket(AF_UNIX, SOCK_STREAM, 0);
  CHECK(connect(cl, (struct sockaddr *)&sa, salen) == 0);
  pl = sizeof pfd;
  CHECK(getsockopt(cl, SOL_SOCKET, SO_PEERPIDFD, &pfd, &pl) == 0 && pfd >= 0 && pidfd_alive(pfd));
  close(done[1]);
  int st = -1;
  CHECK(waitpid(child, &st, 0) == child && WIFEXITED(st) && WEXITSTATUS(st) == 0);
  /* The peer has gone since: its pidfd says so. */
  CHECK(pidfd_exited(pfd));
  close(pfd);
  close(cl);
  close(ready[0]), close(ready[1]), close(done[0]);

  /* SO_PASSPIDFD is AF_UNIX's alone. */
  int t = socket(AF_INET, SOCK_STREAM, 0);
  errno = 0;
  CHECK(setsockopt(t, SOL_SOCKET, SO_PASSPIDFD, &on, sizeof on) == -1 && errno == EOPNOTSUPP);
  errno = 0;
  vl = sizeof v;
  CHECK(getsockopt(t, SOL_SOCKET, SO_PASSPIDFD, &v, &vl) == -1 && errno == EOPNOTSUPP);
  errno = 0;
  pl = sizeof pfd;
  CHECK(getsockopt(t, SOL_SOCKET, SO_PEERPIDFD, &pfd, &pl) == -1 && errno == ENODATA);
  close(t);
  close(p[0]);
  close(p[1]);
  printf("SCM_PIDFD with SO_PASSPIDFD, and SO_PEERPIDFD\n");
}

static void test_lifetime(void)
{
  char buf[16];
  int sv[2], p[2];

  /* Held only by a queued datagram: alive until it is received. */
  CHECK(pipe(p) == 0);
  CHECK(socketpair(AF_UNIX, SOCK_DGRAM, 0, sv) == 0);
  CHECK(send_fds(sv[0], "f", 1, &p[1], 1, NULL, 0) == 1);
  close(p[1]);
  close(sv[0]);
  CHECK(nonblocking_pipe_empty(p[0]));
  struct got g = recv_fds(sv[1], buf, sizeof buf, CMSG_SPACE(sizeof(int)), 0);
  CHECK(g.n == 1 && g.nfds == 1 && same_pipe(g.fds[0], p[0]));
  close(g.fds[0]);
  CHECK(pipe_eof(p[0]));
  close(sv[1]);
  close(p[0]);

  /* Closing the receiver with descriptors queued releases them. */
  CHECK(pipe(p) == 0);
  CHECK(socketpair(AF_UNIX, SOCK_DGRAM, 0, sv) == 0);
  CHECK(send_fds(sv[0], "f", 1, &p[1], 1, NULL, 0) == 1);
  close(p[1]);
  close(sv[1]);
  CHECK(pipe_eof(p[0]));
  close(sv[0]);
  close(p[0]);
  /* The same on a stream socket. */
  CHECK(pipe(p) == 0);
  CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
  CHECK(send_fds(sv[0], "f", 1, &p[1], 1, NULL, 0) == 1);
  close(p[1]);
  close(sv[1]);
  CHECK(pipe_eof(p[0]));
  close(sv[0]);
  close(p[0]);
  /* And on a bound datagram socket nobody connected to. */
  CHECK(pipe(p) == 0);
  struct sockaddr_un ra;
  socklen_t ralen = abstract(&ra, "lifetime");
  int r = socket(AF_UNIX, SOCK_DGRAM, 0);
  CHECK(bind(r, (struct sockaddr *)&ra, ralen) == 0);
  int s = socket(AF_UNIX, SOCK_DGRAM, 0);
  CHECK(send_fds(s, "f", 1, &p[1], 1, &ra, ralen) == 1);
  close(p[1]);
  close(r);
  CHECK(pipe_eof(p[0]));
  close(s);
  close(p[0]);
  printf("in-flight fds live until received or their queue closes\n");

  /* A socket sent over itself, with a pipe for company: once every descriptor
   * outside is closed, the cycle is collected. */
  for (int type = SOCK_STREAM; type <= SOCK_DGRAM; type++) {
    CHECK(pipe(p) == 0);
    CHECK(socketpair(AF_UNIX, type, 0, sv) == 0);
    int both[2] = {sv[1], p[1]};
    CHECK(send_fds(sv[0], "f", 1, both, 2, NULL, 0) == 1);
    close(p[1]);
    close(sv[0]);
    close(sv[1]);
    kick_gc();
    CHECK(pipe_eof(p[0]));
    close(p[0]);
  }
  /* Two sockets each holding the other. */
  CHECK(pipe(p) == 0);
  int a[2], b[2];
  CHECK(socketpair(AF_UNIX, SOCK_DGRAM, 0, a) == 0);
  CHECK(socketpair(AF_UNIX, SOCK_DGRAM, 0, b) == 0);
  int ab[2] = {b[1], p[1]};
  CHECK(send_fds(a[0], "f", 1, ab, 2, NULL, 0) == 1);
  CHECK(send_fds(b[0], "f", 1, &a[1], 1, NULL, 0) == 1);
  close(p[1]);
  close(a[0]);
  close(b[0]);
  close(a[1]);
  kick_gc();
  CHECK(nonblocking_pipe_empty(p[0]));
  close(b[1]);
  kick_gc();
  CHECK(pipe_eof(p[0]));
  close(p[0]);
  printf("a socket sent over itself is collected\n");
}

static void test_mmsg(void)
{
  int sv[2], p[2];
  CHECK(pipe(p) == 0);
  CHECK(socketpair(AF_UNIX, SOCK_DGRAM, 0, sv) == 0);
  struct mmsghdr mm[3];
  struct iovec io[3];
  char *msgs[3] = {"one", "two", "three"};
  char cb[3][CMSG_SPACE(2 * sizeof(int))] __attribute__((aligned(8)));
  memset(mm, 0, sizeof mm);
  memset(cb, 0, sizeof cb);
  for (int i = 0; i < 3; i++) {
    io[i].iov_base = msgs[i];
    io[i].iov_len = strlen(msgs[i]);
    mm[i].msg_hdr.msg_iov = &io[i];
    mm[i].msg_hdr.msg_iovlen = 1;
    if (i == 1)
      continue;
    int n = i == 0 ? 1 : 2;
    mm[i].msg_hdr.msg_control = cb[i];
    mm[i].msg_hdr.msg_controllen = CMSG_SPACE(n * sizeof(int));
    struct cmsghdr *c = CMSG_FIRSTHDR(&mm[i].msg_hdr);
    c->cmsg_level = SOL_SOCKET;
    c->cmsg_type = SCM_RIGHTS;
    c->cmsg_len = CMSG_LEN(n * sizeof(int));
    int fds[2] = {p[1], p[0]};
    memcpy(CMSG_DATA(c), fds, n * sizeof(int));
  }
  CHECK(sendmmsg(sv[0], mm, 3, 0) == 3);
  CHECK(mm[0].msg_len == 3 && mm[1].msg_len == 3 && mm[2].msg_len == 5);

  struct mmsghdr rm[4];
  struct iovec rio[4];
  char rb[4][16];
  char rc[4][CMSG_SPACE(4 * sizeof(int))] __attribute__((aligned(8)));
  memset(rm, 0, sizeof rm);
  for (int i = 0; i < 4; i++) {
    rio[i].iov_base = rb[i];
    rio[i].iov_len = sizeof rb[i];
    rm[i].msg_hdr.msg_iov = &rio[i];
    rm[i].msg_hdr.msg_iovlen = 1;
    rm[i].msg_hdr.msg_control = rc[i];
    rm[i].msg_hdr.msg_controllen = sizeof rc[i];
  }
  CHECK(recvmmsg(sv[1], rm, 4, MSG_DONTWAIT, NULL) == 3);
  CHECK(rm[0].msg_len == 3 && rm[1].msg_len == 3 && rm[2].msg_len == 5);
  CHECK(rm[0].msg_hdr.msg_controllen == CMSG_SPACE(sizeof(int)));
  CHECK(rm[1].msg_hdr.msg_controllen == 0);
  CHECK(rm[2].msg_hdr.msg_controllen == CMSG_SPACE(2 * sizeof(int)));
  int f0, f2[2];
  memcpy(&f0, CMSG_DATA(CMSG_FIRSTHDR(&rm[0].msg_hdr)), sizeof f0);
  memcpy(f2, CMSG_DATA(CMSG_FIRSTHDR(&rm[2].msg_hdr)), sizeof f2);
  CHECK(same_pipe(f0, p[0]) && same_pipe(f2[0], f2[1]));
  close(f0);
  close(f2[0]);
  close(f2[1]);
  close(sv[0]);
  close(sv[1]);
  close(p[0]);
  close(p[1]);
  printf("sendmmsg/recvmmsg carry each message's own fds\n");
}

static int passrights(int s)
{
  int v = -1;
  socklen_t l = sizeof v;
  if (getsockopt(s, SOL_SOCKET, SO_PASSRIGHTS, &v, &l) != 0)
    return -errno;
  return l == sizeof v ? v : -1000;
}

static int set_passrights(int s, int v)
{
  return setsockopt(s, SOL_SOCKET, SO_PASSRIGHTS, &v, sizeof v) == 0 ? 0 : -errno;
}

/* Whether nothing is queued on `s`. */
static int nothing_queued(int s)
{
  char c;
  return recv(s, &c, 1, MSG_DONTWAIT) == -1 && errno == EAGAIN;
}

static void test_passrights(const char *dir)
{
  int p[2];
  CHECK(pipe(p) == 0);
  /* Only AF_UNIX has it, and it takes an int. */
  int udp = socket(AF_INET, SOCK_DGRAM, 0);
  CHECK(udp >= 0);
  CHECK(passrights(udp) == -EOPNOTSUPP);
  CHECK(set_passrights(udp, 0) == -EOPNOTSUPP);
  close(udp);
  static const int types[] = {SOCK_STREAM, SOCK_DGRAM, SOCK_SEQPACKET};
  for (int t = 0; t < 3; t++) {
    int sv[2];
    CHECK(socketpair(AF_UNIX, types[t], 0, sv) == 0);
    CHECK(passrights(sv[0]) == 1 && passrights(sv[1]) == 1);
    short half = 0;
    CHECK(setsockopt(sv[1], SOL_SOCKET, SO_PASSRIGHTS, &half, sizeof half) == -1 && errno == EINVAL);
    /* Off on one end: descriptors to it are refused, nothing is queued, and
     * data alone still goes. */
    CHECK(set_passrights(sv[1], 0) == 0);
    CHECK(passrights(sv[1]) == 0 && passrights(sv[0]) == 1);
    errno = 0;
    CHECK(send_fds(sv[0], "x", 1, &p[0], 1, NULL, 0) == -1 && errno == EPERM);
    CHECK(nothing_queued(sv[1]));
    CHECK(send_fds(sv[0], "y", 1, NULL, 0, NULL, 0) == 1);
    char buf[4];
    struct got g = recv_fds(sv[1], buf, sizeof buf, 64, 0);
    CHECK(g.n == 1 && buf[0] == 'y' && g.nfds == 0);
    /* The other end still takes them. */
    CHECK(send_fds(sv[1], "z", 1, &p[0], 1, NULL, 0) == 1);
    g = recv_fds(sv[0], buf, sizeof buf, 64, 0);
    CHECK(g.n == 1 && g.nfds == 1 && same_pipe(p[1], g.fds[0]));
    close(g.fds[0]);
    /* On again (any non-zero value). */
    CHECK(set_passrights(sv[1], 7) == 0 && passrights(sv[1]) == 1);
    CHECK(send_fds(sv[0], "w", 1, &p[0], 1, NULL, 0) == 1);
    g = recv_fds(sv[1], buf, sizeof buf, 64, 0);
    CHECK(g.n == 1 && g.nfds == 1);
    close(g.fds[0]);
    close(sv[0]);
    close(sv[1]);
  }
  /* An accepted connection takes its listener's setting; the connecting
   * socket keeps its own. */
  for (int t = 0; t < 3; t += 2) {
    struct sockaddr_un a = {.sun_family = AF_UNIX};
    snprintf(a.sun_path, sizeof a.sun_path, "%s/rights%d", dir, t);
    int l = socket(AF_UNIX, types[t], 0);
    CHECK(l >= 0);
    CHECK(bind(l, (struct sockaddr *)&a, sizeof a) == 0 && listen(l, 4) == 0);
    CHECK(set_passrights(l, 0) == 0);
    int c = socket(AF_UNIX, types[t], 0);
    CHECK(connect(c, (struct sockaddr *)&a, sizeof a) == 0);
    int s = accept(l, NULL, NULL);
    CHECK(s >= 0);
    CHECK(passrights(s) == 0 && passrights(c) == 1);
    CHECK(send_fds(c, "x", 1, &p[0], 1, NULL, 0) == -1 && errno == EPERM);
    CHECK(send_fds(s, "x", 1, &p[0], 1, NULL, 0) == 1);
    char buf[4];
    struct got g = recv_fds(c, buf, sizeof buf, 64, 0);
    CHECK(g.n == 1 && g.nfds == 1);
    close(g.fds[0]);
    close(c);
    close(s);
    close(l);
    unlink(a.sun_path);
  }
  /* A named datagram socket that takes none refuses them from anyone. */
  struct sockaddr_un a = {.sun_family = AF_UNIX};
  snprintf(a.sun_path, sizeof a.sun_path, "%s/rightsd", dir);
  int r = socket(AF_UNIX, SOCK_DGRAM, 0);
  CHECK(r >= 0 && bind(r, (struct sockaddr *)&a, sizeof a) == 0);
  CHECK(set_passrights(r, 0) == 0);
  int w = socket(AF_UNIX, SOCK_DGRAM, 0);
  CHECK(w >= 0);
  CHECK(send_fds(w, "x", 1, &p[0], 1, &a, sizeof a) == -1 && errno == EPERM);
  CHECK(nothing_queued(r));
  CHECK(send_fds(w, "y", 1, NULL, 0, &a, sizeof a) == 1);
  close(w);
  close(r);
  unlink(a.sun_path);
  close(p[0]);
  close(p[1]);
}

int main(void)
{
  char dir[64];
  snprintf(dir, sizeof dir, "/tmp/unix-scm-%d", getpid());
  CHECK(mkdir(dir, 0700) == 0);
  test_dgram_pair();
  test_ctrunc();
  test_limits();
  test_named(dir);
  test_seqpacket();
  test_creds();
  test_lifetime();
  test_mmsg();
  test_pidfd();
  test_passrights(dir);
  rmdir(dir);
  puts("PASS");
  return 0;
}
