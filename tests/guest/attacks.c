#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
static long report(const char *w, long r) { printf("%s -> %ld (errno %d %s)\n", w, r, r < 0 ? errno : 0, r < 0 ? strerror(errno) : ""); fflush(stdout); return r; }
int main(int argc, char **argv) {
  const char *m = argv[1]; setvbuf(stdout, 0, _IONBF, 0);
  int fd = open("/tmp/x", O_CREAT | O_RDWR, 0600);
  if (!strcmp(m, "ftruncate")) report("ftruncate 2^62", ftruncate(fd, 1L << 62));
  else if (!strcmp(m, "pwrite")) report("pwrite at 2^63-1", pwrite(fd, "x", 1, 0x7fffffffffffffffL));
  else if (!strcmp(m, "fallocate")) report("fallocate 2^62", fallocate(fd, 0, 0, 1L << 62));
  else if (!strcmp(m, "poll")) report("poll nfds huge", syscall(SYS_poll, 0, 1UL << 60, 0));
  else if (!strcmp(m, "mincore")) { char *p = mmap(0, 1L<<20, PROT_READ, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0); report("mincore huge", syscall(SYS_mincore, p, 1UL << 60, p)); }
  else if (!strcmp(m, "recvmsg")) { int s[2]; socketpair(AF_UNIX, SOCK_STREAM, 0, s); struct msghdr h = {0}; char b[1]; struct iovec v = {b, 1}; h.msg_iov = &v; h.msg_iovlen = 1UL << 60; report("recvmsg iovlen huge", recvmsg(s[0], &h, MSG_DONTWAIT)); }
  else if (!strcmp(m, "mempolicy")) { unsigned long n[8]; report("get_mempolicy maxnode huge", syscall(SYS_get_mempolicy, 0, n, 1UL << 60, 0, 0)); }
  else if (!strcmp(m, "forkloop")) { long i; long max = argc > 2 ? atol(argv[2]) : 6000; for (i = 0; i < max; i++) { pid_t p = fork(); if (p < 0) { report("fork", -1); break; } if (!p) _exit(0); waitpid(p, 0, 0); if (i % 2000 == 0) printf("forks %ld\n", i); } printf("done %ld\n", i); }
  else if (!strcmp(m, "forkbomb")) { long i; for (i = 0; i < 1000000; i++) { pid_t p = fork(); if (p < 0) { report("fork", -1); break; } if (!p) pause(); if (i % 2000 == 0) printf("tasks %ld\n", i); } printf("stopped at %ld\n", i); }
  else if (!strcmp(m, "pipe")) { int p[2]; pipe(p); fcntl(p[1], F_SETFL, O_NONBLOCK); size_t sz = 1UL << 30; char *b = mmap(0, sz, PROT_READ, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0); long tot = 0; for (int i = 0; i < 200; i++) { long r = write(p[1], b, sz); if (r < 0) { report("pipe write", r); break; } tot += r; if (r < (long)sz) break; } printf("pipe absorbed %ld MiB\n", tot >> 20); }
  else if (!strcmp(m, "tmpfs")) { static char b[1 << 20]; long i; for (i = 0; i < 1 << 20; i++) { long r = write(fd, b, sizeof b); if (r < 0) { report("tmpfs write", r); break; } if (i % 2048 == 0) printf("tmpfs %ld GiB\n", i >> 10); } printf("wrote %ld MiB\n", i); }
  else if (!strcmp(m, "fds")) { long i; for (i = 0; i < 5000000; i++) { if (dup(0) < 0) { report("dup", -1); break; } } printf("opened %ld\n", i); }
  else if (!strcmp(m, "memexit")) { size_t sz = strtoul(argv[2], 0, 0) << 20; char *p = mmap(0, sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0); if (p == MAP_FAILED) { report("mmap", -1); return 1; } puts("touching"); for (size_t i = 0; i < sz; i += 4096) p[i] = 1; puts("touched all; exiting"); return 0; }
  else if (!strcmp(m, "memfork")) { size_t sz = strtoul(argv[2], 0, 0) << 20; char *p = mmap(0, sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0); if (p == MAP_FAILED) { report("mmap", -1); return 1; } for (size_t i = 0; i < sz; i += 4096) p[i] = 1; puts("touched; forking"); pid_t c = fork(); report("fork", c); if (!c) _exit(0); return 0; }
  else if (!strcmp(m, "cowfill")) { size_t sz = strtoul(argv[2], 0, 0) << 20; char *p = mmap(0, sz, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0); if (p == MAP_FAILED) { report("mmap", -1); return 1; } for (size_t i = 0; i < sz; i += 4096) p[i] = 1; for (int d = 1; d <= 6; d++) { pid_t c = fork(); if (c < 0) { report("fork", -1); printf("depth %d could not fork\n", d); return 3; } if (c) { int st = 0; waitpid(c, &st, 0); printf("depth %d: child status %#x\n", d, st); return d == 1 ? 0 : (st == 0 ? 0 : 9); } for (size_t i = 0; i < sz; i += 4096) p[i] = d; printf("depth %d wrote everything\n", d); } _exit(0); }
  else if (!strcmp(m, "spin")) for (;;);
  return 0;
}
