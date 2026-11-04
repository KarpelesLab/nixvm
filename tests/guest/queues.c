#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <sys/wait.h>
#include <unistd.h>
static int fails;
#define CHECK(c, ...) do { int ok_ = (c); printf("%s: ", ok_ ? "ok  " : "FAIL"); printf(__VA_ARGS__); puts(""); fflush(stdout); if (!ok_) fails++; } while (0)
static unsigned sum(const unsigned char *b, size_t n) { unsigned s = 0; for (size_t i = 0; i < n; i++) s = s * 31 + b[i]; return s; }
struct job { int fd; size_t n; unsigned *out; size_t *got; };
static void *reader(void *a) { struct job *j = a; unsigned char *b = malloc(j->n); size_t got = 0; while (got < j->n) { ssize_t r = read(j->fd, b + got, j->n - got); if (r <= 0) break; got += r; usleep(200); } *j->got = got; *j->out = sum(b, got); free(b); return 0; }
static void h(int s) { (void)s; }
int main(void) {
  /* An empty iovec's pointer is never looked at (musl's exit flush passes one). */
  { int zp[2]; pipe(zp); struct iovec v[2] = {{"abc", 3}, {(void *)8, 0}}; CHECK(writev(zp[1], v, 2) == 3, "writev to a pipe with an empty bad segment");
    int sv[2]; socketpair(AF_UNIX, SOCK_STREAM, 0, sv); struct msghdr mh = {0}; mh.msg_iov = v; mh.msg_iovlen = 2; CHECK(sendmsg(sv[0], &mh, 0) == 3, "and sendmsg");
    CHECK(write(zp[1], (void *)8, 0) == 0 && read(zp[0], (void *)8, 0) == 0, "zero-length read/write with a bad pointer"); }
  int p[2]; pipe(p); long sz = fcntl(p[1], F_GETPIPE_SZ); CHECK(sz == 65536, "default capacity %ld", sz);
  struct pollfd w = {p[1], POLLOUT, 0}; CHECK(poll(&w, 1, 0) == 1, "empty pipe polls writable");
  fcntl(p[1], F_SETFL, O_NONBLOCK); char *big = malloc(1 << 20); for (int i = 0; i < 1 << 20; i++) big[i] = i * 7;
  long n = write(p[1], big, 1 << 20); CHECK(n == 65536, "nonblocking big write is short: %ld", n);
  w.revents = 0; CHECK(poll(&w, 1, 0) == 0, "full pipe does not poll writable");
  errno = 0; n = write(p[1], "x", 1); CHECK(n < 0 && errno == EAGAIN, "full nonblocking write EAGAIN (%d)", errno);
  char tmp[100]; read(p[0], tmp, 100); errno = 0; char block[4096] = {0}; n = write(p[1], block, 4096); CHECK(n < 0 && errno == EAGAIN, "small write stays atomic: nothing written (%ld)", n);
  n = write(p[1], "y", 1); CHECK(n == 1 || (n < 0 && errno == EAGAIN), "a byte fits, or Linux still holds the page");
  CHECK(fcntl(p[1], F_SETPIPE_SZ, 4096) < 0 && errno == EBUSY, "can't shrink below contents");
  close(p[0]); close(p[1]);
  pipe(p); CHECK(fcntl(p[1], F_SETPIPE_SZ, 1 << 19) == 1 << 19, "F_SETPIPE_SZ grows"); CHECK(fcntl(p[1], F_GETPIPE_SZ) == 1 << 19, "and reads back"); close(p[0]); close(p[1]);
  /* Blocking write of 1 MiB through a slow reader: one complete, intact write. */
  pipe(p); unsigned got_sum = 0; size_t got = 0; struct job j = {p[0], 1 << 20, &got_sum, &got}; pthread_t t; pthread_create(&t, 0, reader, &j);
  n = write(p[1], big, 1 << 20); pthread_join(t, 0);
  CHECK(n == 1 << 20, "blocking write completes in one call: %ld", n); CHECK(got == 1 << 20 && got_sum == sum((void *)big, 1 << 20), "data intact");
  close(p[0]); close(p[1]);
  /* writev too. */
  pipe(p); got = 0; got_sum = 0; j.fd = p[0]; pthread_create(&t, 0, reader, &j); struct iovec v[2] = {{big, 1 << 19}, {big + (1 << 19), 1 << 19}}; n = writev(p[1], v, 2); pthread_join(t, 0);
  CHECK(n == 1 << 20 && got_sum == sum((void *)big, 1 << 20), "writev completes intact: %ld", n); close(p[0]); close(p[1]);
  /* Interrupted write reports what it wrote. */
  pipe(p); struct sigaction sa = {0}; sa.sa_handler = h; sigaction(SIGALRM, &sa, 0); ualarm(50000, 0); n = write(p[1], big, 1 << 20); CHECK(n == 65536, "interrupted write returns the partial count: %ld", n); close(p[0]); close(p[1]);
  /* Reader gone mid-write: partial count, not EPIPE. */
  signal(SIGPIPE, SIG_IGN); pipe(p); pid_t c = fork(); if (!c) { close(p[1]); read(p[0], tmp, 100); _exit(0); } close(p[0]); n = write(p[1], big, 1 << 20); CHECK(n > 0 && n < 1 << 20, "reader gone mid-write: partial %ld", n); n = write(p[1], big, 10); CHECK(n < 0 && errno == EPIPE, "then EPIPE"); waitpid(c, 0, 0); close(p[1]);
  /* Sockets. */
  int s[2]; socketpair(AF_UNIX, SOCK_STREAM, 0, s); fcntl(s[0], F_SETFL, O_NONBLOCK); long tot = 0; while (tot < 64 << 20) { n = write(s[0], big, 1 << 20); if (n <= 0) break; tot += n; } CHECK(tot > 0 && tot < 8 << 20 && errno == EAGAIN, "socket queue is bounded: %ld then EAGAIN", tot);
  struct pollfd w2 = {s[0], POLLOUT, 0}; CHECK(poll(&w2, 1, 0) == 0, "full socket does not poll writable"); close(s[0]); close(s[1]);
  socketpair(AF_UNIX, SOCK_STREAM, 0, s); got = 0; got_sum = 0; struct job j2 = {s[1], 4 << 20, &got_sum, &got}; unsigned char *b4 = malloc(4 << 20); for (int i = 0; i < 4 << 20; i++) b4[i] = i * 13; pthread_create(&t, 0, reader, &j2); n = write(s[0], b4, 4 << 20); pthread_join(t, 0); CHECK(n == 4 << 20 && got_sum == sum(b4, 4 << 20), "blocking socket send completes intact: %ld", n); close(s[0]); close(s[1]);
  socketpair(AF_UNIX, SOCK_STREAM, 0, s); tot = 0; for (int i = 0; i < 64; i++) { n = send(s[0], big, 1 << 20, MSG_DONTWAIT); if (n <= 0) break; tot += n; } CHECK(tot > 0 && n < 0 && errno == EAGAIN, "MSG_DONTWAIT on a blocking socket: %ld then EAGAIN", tot); close(s[0]); close(s[1]);
  int u = socket(AF_INET, SOCK_DGRAM, 0); struct sockaddr_in sa2 = {0}; sa2.sin_family = AF_INET; sa2.sin_port = htons(7100); sa2.sin_addr.s_addr = htonl(0x7f000001); bind(u, (void *)&sa2, sizeof sa2); int o = socket(AF_INET, SOCK_DGRAM, 0); for (int i = 0; i < 2000; i++) sendto(o, big, 60000, 0, (void *)&sa2, sizeof sa2); int cnt = 0; while (recv(u, big, 65000, MSG_DONTWAIT) > 0) cnt++; CHECK(cnt > 0 && cnt < 2000, "UDP receive queue drops when full: kept %d of 2000", cnt);
  /* Descriptors. */
  struct rlimit r = {64, 4096}; setrlimit(RLIMIT_NOFILE, &r); int last = -1, k = 0; for (;;) { int d = dup(0); if (d < 0) break; last = d; k++; } CHECK(errno == EMFILE && last == 63, "dup stops at RLIMIT_NOFILE: EMFILE, last fd %d", last);
  CHECK(pipe(p) < 0 && errno == EMFILE, "pipe EMFILE too"); close(last); CHECK(pipe(p) < 0 && errno == EMFILE, "pipe needs two free"); int d = open("/tmp/z", O_CREAT | O_RDWR, 0600); CHECK(d == last, "one free: open fills it"); CHECK(open("/tmp/z2", O_CREAT | O_RDWR, 0600) < 0 && errno == EMFILE, "open EMFILE"); CHECK(socket(AF_INET, SOCK_STREAM, 0) < 0 && errno == EMFILE, "socket EMFILE");
  CHECK(dup2(0, 64) < 0 && errno == EBADF, "dup2 past the limit EBADF"); CHECK(fcntl(0, F_DUPFD, 64) < 0 && errno == EINVAL, "F_DUPFD past the limit EINVAL");
  puts(fails ? "SOME FAILED" : "PASS"); return fails != 0;
}
