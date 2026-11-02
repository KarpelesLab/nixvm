#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>
static int fails;
#define CHECK(c, ...) do { int ok_ = (c); printf("%s: ", ok_ ? "ok  " : "FAIL"); printf(__VA_ARGS__); puts(""); fflush(stdout); if (!ok_) fails++; } while (0)
static struct sockaddr_in in4(int port) { struct sockaddr_in a = {0}; a.sin_family = AF_INET; a.sin_port = htons(port); a.sin_addr.s_addr = htonl(INADDR_LOOPBACK); return a; }
static int unix_addr(struct sockaddr_un *u, const char *name, int abs) { memset(u, 0, sizeof *u); u->sun_family = AF_UNIX; int n = strlen(name); if (abs) { memcpy(u->sun_path + 1, name, n); return offsetof(struct sockaddr_un, sun_path) + 1 + n; } strcpy(u->sun_path, name); return sizeof *u; }
#include <stddef.h>
static void lo_up(void) {} /* on real Linux see below */
#include <net/if.h>
#include <sys/ioctl.h>
static void bring_lo_up(void) { int s = socket(AF_INET, SOCK_DGRAM, 0); struct ifreq r = {0}; strcpy(r.ifr_name, "lo"); ioctl(s, SIOCGIFFLAGS, &r); r.ifr_flags |= IFF_UP; ioctl(s, SIOCSIFFLAGS, &r); close(s); }
int main(void) {
  int srv = socket(AF_INET, SOCK_STREAM, 0), one = 1; struct sockaddr_in a = in4(7200); CHECK(!bind(srv, (void *)&a, sizeof a) && !listen(srv, 8), "listener in the first namespace");
  int u = socket(AF_INET, SOCK_DGRAM, 0); struct sockaddr_in ua = in4(7201); bind(u, (void *)&ua, sizeof ua);
  struct sockaddr_un au; int alen = unix_addr(&au, "abs-sock", 1); int us = socket(AF_UNIX, SOCK_STREAM, 0); CHECK(!bind(us, (void *)&au, alen) && !listen(us, 8), "abstract unix listener");
  unlink("/tmp/nsp"); struct sockaddr_un pu; int plen = unix_addr(&pu, "/tmp/nsp", 0); int ps = socket(AF_UNIX, SOCK_STREAM, 0); CHECK(!bind(ps, (void *)&pu, plen) && !listen(ps, 8), "path unix listener");
  int sp[2]; socketpair(AF_UNIX, SOCK_STREAM, 0, sp);
  int keep = open("/proc/self/ns/net", O_RDONLY); CHECK(keep >= 0, "keep a handle on the first namespace");
  int p[2]; pipe(p); fflush(stdout); pid_t c = fork();
  if (!c) {
    CHECK(!unshare(CLONE_NEWNET), "unshare net"); bring_lo_up();
    int s = socket(AF_INET, SOCK_STREAM, 0); errno = 0; CHECK(connect(s, (void *)&a, sizeof a) < 0 && errno == ECONNREFUSED, "the listener isn't here: ECONNREFUSED (%d)", errno);
    int s2 = socket(AF_INET, SOCK_STREAM, 0); CHECK(!bind(s2, (void *)&a, sizeof a) && !listen(s2, 8), "the same port is free here");
    int c2 = socket(AF_INET, SOCK_STREAM, 0); CHECK(!connect(c2, (void *)&a, sizeof a), "and loopback works inside"); int acc = accept(s2, 0, 0); write(c2, "hi", 2); char b[3] = {0}; read(acc, b, 2); CHECK(!strcmp(b, "hi"), "data flows inside");
    int ud = socket(AF_INET, SOCK_DGRAM, 0); CHECK(!bind(ud, (void *)&ua, sizeof ua), "UDP port free here"); int uc = socket(AF_INET, SOCK_DGRAM, 0); sendto(uc, "in", 2, 0, (void *)&ua, sizeof ua); char ub[3] = {0}; CHECK(recv(ud, ub, 2, MSG_DONTWAIT) == 2 && !strcmp(ub, "in"), "UDP stays inside");
    int cu = socket(AF_UNIX, SOCK_STREAM, 0); errno = 0; CHECK(connect(cu, (void *)&au, alen) < 0 && errno == ECONNREFUSED, "abstract name isn't here (%d)", errno);
    int bu = socket(AF_UNIX, SOCK_STREAM, 0); CHECK(!bind(bu, (void *)&au, alen), "the abstract name is free here");
    int cp = socket(AF_UNIX, SOCK_STREAM, 0); CHECK(!connect(cp, (void *)&pu, plen), "a path name is a file: it still connects");
    CHECK(write(sp[0], "x", 1) == 1, "an old socket keeps working across namespaces");
    /* Unroutable from here. */
    struct sockaddr_in far = {0}; far.sin_family = AF_INET; far.sin_port = htons(80); inet_pton(AF_INET, "192.0.2.1", &far.sin_addr); int fs = socket(AF_INET, SOCK_STREAM, 0); errno = 0; CHECK(connect(fs, (void *)&far, sizeof far) < 0 && errno == ENETUNREACH, "no route out (%d)", errno);
    /* And back. */
    CHECK(!setns(keep, CLONE_NEWNET), "setns back"); int b1 = socket(AF_INET, SOCK_STREAM, 0); CHECK(!connect(b1, (void *)&a, sizeof a), "a new socket now reaches the first namespace"); CHECK(listen(s2, 8) == 0, "the inner one is still inner");
    _exit(fails);
  }
  int st; waitpid(c, &st, 0); fails += WIFEXITED(st) ? WEXITSTATUS(st) : 1;
  char x; CHECK(read(sp[1], &x, 1) == 1 && x == 'x', "socketpair crossed the namespace boundary");
  /* First namespace's ports are untouched. */
  int t = socket(AF_INET, SOCK_STREAM, 0); CHECK(bind(t, (void *)&a, sizeof a) < 0 && errno == EADDRINUSE, "first namespace still holds its port");
  puts(fails ? "SOME FAILED" : "PASS"); return fails != 0;
}
