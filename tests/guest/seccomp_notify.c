/* seccomp user notification and the rest of the seccomp API: a supervisor
 * receiving a target's syscalls over a listener passed with SCM_RIGHTS,
 * replies (values, errors, CONTINUE), ADDFD (plain, SETFD, SEND), reading
 * the target's memory through /proc/<pid>/mem, concurrent notifications,
 * signal interruption (and WAIT_KILLABLE_RECV), POLLHUP, closing the
 * listener; TSYNC, KILL_THREAD, the status and sysctl files. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <linux/filter.h>
#include <linux/sched.h>
#include <linux/seccomp.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef SECCOMP_FILTER_FLAG_WAIT_KILLABLE_RECV
#define SECCOMP_FILTER_FLAG_WAIT_KILLABLE_RECV (1UL << 5)
#endif
#ifndef SECCOMP_ADDFD_FLAG_SEND
#define SECCOMP_ADDFD_FLAG_SEND (1UL << 1)
#endif
#ifndef SECCOMP_FILTER_FLAG_TSYNC_ESRCH
#define SECCOMP_FILTER_FLAG_TSYNC_ESRCH (1UL << 4)
#endif
#ifndef SECCOMP_IOCTL_NOTIF_SET_FLAGS
#define SECCOMP_IOCTL_NOTIF_SET_FLAGS SECCOMP_IOW(4, __u64)
#endif

static int fails;
#define CHECK(c, ...)                                                          \
  do {                                                                         \
    if (!(c)) {                                                                \
      printf("FAIL: line %d: ", __LINE__);                                     \
      printf(__VA_ARGS__);                                                     \
      printf(" (errno %d)\n", errno);                                          \
      fflush(stdout);                                                          \
      fails++;                                                                 \
    }                                                                          \
  } while (0)

static long sc(unsigned op, unsigned flags, void *args) {
  return syscall(__NR_seccomp, op, flags, args);
}

/* nr == which ? action : ALLOW */
static long install(int nr, unsigned action, unsigned flags) {
  struct sock_filter f[] = {
      BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
      BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, nr, 0, 1),
      BPF_STMT(BPF_RET | BPF_K, action),
      BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
  };
  struct sock_fprog p = {4, f};
  return sc(SECCOMP_SET_MODE_FILTER, flags, &p);
}

static void send_fd(int sock, int fd) {
  char c = 'x';
  struct iovec io = {&c, 1};
  char buf[CMSG_SPACE(sizeof(int))] = {0};
  struct msghdr m = {0};
  m.msg_iov = &io;
  m.msg_iovlen = 1;
  m.msg_control = buf;
  m.msg_controllen = sizeof buf;
  struct cmsghdr *cm = CMSG_FIRSTHDR(&m);
  cm->cmsg_level = SOL_SOCKET;
  cm->cmsg_type = SCM_RIGHTS;
  cm->cmsg_len = CMSG_LEN(sizeof(int));
  memcpy(CMSG_DATA(cm), &fd, sizeof fd);
  if (sendmsg(sock, &m, 0) != 1) _exit(90);
}

static int recv_fd(int sock) {
  char c;
  struct iovec io = {&c, 1};
  char buf[CMSG_SPACE(sizeof(int))] = {0};
  struct msghdr m = {0};
  m.msg_iov = &io;
  m.msg_iovlen = 1;
  m.msg_control = buf;
  m.msg_controllen = sizeof buf;
  if (recvmsg(sock, &m, 0) != 1) return -1;
  struct cmsghdr *cm = CMSG_FIRSTHDR(&m);
  if (!cm || cm->cmsg_type != SCM_RIGHTS) return -1;
  int fd;
  memcpy(&fd, CMSG_DATA(cm), sizeof fd);
  return fd;
}

/* Fork a target that installs a getppid listener with `flags`, hands it over
 * a socket, and runs `body`; returns its pid and the listener. */
static pid_t spawn(unsigned flags, int (*body)(void), int *listener) {
  int sv[2];
  /* A SOCK_SEQPACKET pair, as the kernel's samples hand listeners over. */
  socketpair(AF_UNIX, SOCK_SEQPACKET, 0, sv);
  pid_t pid = fork();
  if (pid == 0) {
    close(sv[0]);
    prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
    long l = install(__NR_getppid, SECCOMP_RET_USER_NOTIF,
                     SECCOMP_FILTER_FLAG_NEW_LISTENER | flags);
    if (l < 0) _exit(91);
    send_fd(sv[1], l);
    close(l);
    _exit(body());
  }
  close(sv[1]);
  *listener = recv_fd(sv[0]);
  close(sv[0]);
  return pid;
}

static int status_of(pid_t pid) {
  int st;
  if (waitpid(pid, &st, 0) != pid) return -1;
  return WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
}

static int recv_notif(int l, struct seccomp_notif *n) {
  memset(n, 0, sizeof *n);
  return ioctl(l, SECCOMP_IOCTL_NOTIF_RECV, n);
}

static int reply(int l, __u64 id, __s64 val, __s32 error, __u32 flags) {
  struct seccomp_notif_resp r = {.id = id, .val = val, .error = error, .flags = flags};
  return ioctl(l, SECCOMP_IOCTL_NOTIF_SEND, &r);
}

/* ---- targets ---- */

static int t_values(void) {
  if (syscall(__NR_getppid) != 42) return 1;
  errno = 0;
  if (syscall(__NR_getppid) != -1 || errno != EPERM) return 2;
  if (syscall(__NR_getppid) <= 0) return 3;
  if (syscall(__NR_getppid) <= 0) return 4;
  return 0;
}

static int t_continue(void) {
  long r = syscall(__NR_getppid);
  return r > 0 && r != 42 ? 0 : 1;
}

static int t_addfd(void) {
  long n = syscall(__NR_getppid); /* answered with the fd ADDFD put in */
  if (n < 0) return 1;
  if (write((int)n, "a", 1) != 1) return 2;
  if (write(100, "b", 1) != 1) return 3;
  if (!(fcntl(100, F_GETFD) & FD_CLOEXEC)) return 4;
  long s = syscall(__NR_getppid); /* ADDFD|SEND: the call returns the fd */
  if (s < 0 || write((int)s, "c", 1) != 1) return 5;
  return 0;
}

static char secret[64] = "the target's secret";
static int t_mem(void) {
  long r = syscall(__NR_getppid, secret);
  if (r != 7) return 1;
  /* The supervisor wrote back into our memory. */
  return strcmp(secret, "rewritten by supervisor") ? 2 : 0;
}

static __thread long got;
static void *t_thread(void *a) {
  got = syscall(__NR_getppid, (long)a);
  return (void *)got;
}
static int t_threads(void) {
  pthread_t t1, t2;
  pthread_create(&t1, 0, t_thread, (void *)1);
  pthread_create(&t2, 0, t_thread, (void *)2);
  void *r1, *r2;
  pthread_join(t1, &r1);
  pthread_join(t2, &r2);
  return (long)r1 == 101 && (long)r2 == 102 ? 0 : 1;
}

static volatile sig_atomic_t usr1;
static void on_usr1(int s) { (void)s; usr1++; }
static int t_eintr(void) {
  struct sigaction sa = {0};
  sa.sa_handler = on_usr1; /* no SA_RESTART */
  sigaction(SIGUSR1, &sa, 0);
  errno = 0;
  long r = syscall(__NR_getppid);
  if (r != -1 || errno != EINTR || usr1 != 1) return 1;
  /* With SA_RESTART the call is made again: a new notification. */
  sa.sa_flags = SA_RESTART;
  sigaction(SIGUSR1, &sa, 0);
  r = syscall(__NR_getppid);
  return r == 9 && usr1 == 2 ? 0 : 2;
}

static int t_killable(void) {
  struct sigaction sa = {0};
  sa.sa_handler = on_usr1;
  sigaction(SIGUSR1, &sa, 0);
  long r = syscall(__NR_getppid);
  /* Not interrupted: the answer arrives, the handler runs after. */
  return r == 5 && usr1 == 1 ? 0 : 1;
}

static int t_enosys(void) {
  errno = 0;
  long r = syscall(__NR_getppid);
  return r == -1 && errno == ENOSYS ? 0 : 1;
}

static int t_exit(void) { return 0; }

static void wait_pollin(int l) {
  struct pollfd p = {l, POLLIN, 0};
  CHECK(poll(&p, 1, 5000) == 1 && (p.revents & POLLIN), "listener POLLIN");
}

static void wait_gone(int l, __u64 id) {
  for (int i = 0; i < 500; i++) {
    if (ioctl(l, SECCOMP_IOCTL_NOTIF_ID_VALID, &id) < 0 && errno == ENOENT) return;
    usleep(2000);
  }
  CHECK(0, "notification still valid after the signal");
}

/* ---- the rest of the API ---- */

static void *sibling(void *a) {
  int *stop = a;
  while (!__atomic_load_n(stop, __ATOMIC_SEQ_CST)) usleep(1000);
  errno = 0;
  long r = syscall(__NR_getppid);
  return (void *)(long)(r == -1 && errno == EACCES);
}

static void *own_filter(void *a) {
  int *stop = a;
  install(__NR_getuid, SECCOMP_RET_ERRNO | 1, 0);
  while (!__atomic_load_n(stop, __ATOMIC_SEQ_CST)) usleep(1000);
  return 0;
}

static void *kill_me(void *a) {
  (void)a;
  install(__NR_getppid, SECCOMP_RET_KILL_THREAD, 0);
  syscall(__NR_getppid);
  return (void *)1; /* not reached */
}

static int api_child(void) {
  prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
  /* TSYNC: the sibling ends up under the caller's new filter. */
  int stop = 0;
  pthread_t t;
  pthread_create(&t, 0, sibling, &stop);
  if (install(__NR_getppid, SECCOMP_RET_ERRNO | EACCES, SECCOMP_FILTER_FLAG_TSYNC) != 0)
    return 1;
  __atomic_store_n(&stop, 1, __ATOMIC_SEQ_CST);
  void *ok;
  pthread_join(t, &ok);
  if (!ok) return 2;
  /* A sibling with a filter of its own can't be synced: its tid, or ESRCH. */
  stop = 0;
  pid_t tid = 0;
  pthread_create(&t, 0, own_filter, &stop);
  usleep(50000);
  long r = install(__NR_getpid, SECCOMP_RET_ALLOW, SECCOMP_FILTER_FLAG_TSYNC);
  if (r <= 0) return 3;
  tid = r;
  errno = 0;
  r = install(__NR_getpid, SECCOMP_RET_ALLOW,
              SECCOMP_FILTER_FLAG_TSYNC | SECCOMP_FILTER_FLAG_TSYNC_ESRCH);
  if (r != -1 || errno != ESRCH) return 4;
  __atomic_store_n(&stop, 1, __ATOMIC_SEQ_CST);
  pthread_join(t, 0);
  if (tid == getpid()) return 5;
  /* KILL_THREAD ends the thread only. */
  pthread_create(&t, 0, kill_me, 0);
  pthread_join(t, &ok);
  if (ok == (void *)1) return 6;
  /* Flag combinations. */
  errno = 0;
  if (install(__NR_getpid, SECCOMP_RET_ALLOW, SECCOMP_FILTER_FLAG_WAIT_KILLABLE_RECV) != -1 ||
      errno != EINVAL)
    return 7;
  errno = 0;
  if (install(__NR_getpid, SECCOMP_RET_ALLOW,
              SECCOMP_FILTER_FLAG_TSYNC | SECCOMP_FILTER_FLAG_NEW_LISTENER) != -1 ||
      errno != EINVAL)
    return 8;
  if (install(__NR_getpid, SECCOMP_RET_ALLOW, SECCOMP_FILTER_FLAG_LOG | SECCOMP_FILTER_FLAG_SPEC_ALLOW))
    return 9;
  long l = install(__NR_getpid, SECCOMP_RET_ALLOW, SECCOMP_FILTER_FLAG_NEW_LISTENER);
  if (l < 0 || !(fcntl(l, F_GETFD) & FD_CLOEXEC)) return 10;
  errno = 0;
  if (install(__NR_getpid, SECCOMP_RET_ALLOW, SECCOMP_FILTER_FLAG_NEW_LISTENER) != -1 ||
      errno != EBUSY)
    return 11;
  char link[64] = {0}, path[48];
  snprintf(path, sizeof path, "/proc/self/fd/%ld", l);
  if (readlink(path, link, sizeof link - 1) < 0 || strcmp(link, "anon_inode:seccomp notify"))
    return 12;
  if (ioctl(l, SECCOMP_IOCTL_NOTIF_SET_FLAGS, 1) != 0) return 13;
  errno = 0;
  if (ioctl(l, SECCOMP_IOCTL_NOTIF_SET_FLAGS, 2) != -1 || errno != EINVAL) return 14;
  /* /proc/self/status and PR_GET_SECCOMP: 3 filters attached by now. */
  if (prctl(PR_GET_SECCOMP) != 2) return 15;
  FILE *f = fopen("/proc/self/status", "r");
  char line[256];
  int seen = 0;
  while (f && fgets(line, sizeof line, f)) {
    if (!strcmp(line, "NoNewPrivs:\t1\n")) seen |= 1;
    if (!strcmp(line, "Seccomp:\t2\n")) seen |= 2;
    if (!strcmp(line, "Seccomp_filters:\t3\n")) seen |= 4;
  }
  if (f) fclose(f);
  if (seen != 7) return 16;
  /* Strict mode can't be entered from filter mode. */
  errno = 0;
  if (sc(SECCOMP_SET_MODE_STRICT, 0, 0) != -1 || errno != EINVAL) return 17;
  return 0;
}

static int strict_child(void) {
  if (prctl(PR_SET_SECCOMP, SECCOMP_MODE_STRICT, 1234) != 0) return 1; /* arg ignored */
  syscall(__NR_getpid); /* SIGKILL */
  return 2;
}

int main(void) {
  setvbuf(stdout, 0, _IONBF, 0);
  /* SOCK_SEQPACKET keeps message boundaries: a short read loses the rest. */
  int sp[2];
  CHECK(socketpair(AF_UNIX, SOCK_SEQPACKET, 0, sp) == 0, "seqpacket socketpair");
  CHECK(write(sp[0], "hello", 5) == 5 && write(sp[0], "world!", 6) == 6, "seqpacket writes");
  char m[16] = {0};
  CHECK(read(sp[1], m, 3) == 3 && !memcmp(m, "hel", 3), "short read of a message");
  struct iovec mi = {m, sizeof m};
  struct msghdr mh = {.msg_iov = &mi, .msg_iovlen = 1};
  CHECK(recvmsg(sp[1], &mh, 0) == 6 && !memcmp(m, "world!", 6) && !(mh.msg_flags & MSG_TRUNC),
        "next message whole");
  CHECK(write(sp[0], "abcdef", 6) == 6, "seqpacket write");
  mi.iov_len = 2;
  CHECK(recvmsg(sp[1], &mh, 0) == 2 && (mh.msg_flags & MSG_TRUNC), "MSG_TRUNC");
  int type = 0;
  socklen_t tl = sizeof type;
  CHECK(getsockopt(sp[0], SOL_SOCKET, SO_TYPE, &type, &tl) == 0 && type == SOCK_SEQPACKET,
        "SO_TYPE %d", type);
  close(sp[0]);
  CHECK(read(sp[1], m, sizeof m) == 0, "EOF once the peer closes");
  close(sp[1]);

  /* Sizes and actions. */
  struct seccomp_notif_sizes sz;
  CHECK(sc(SECCOMP_GET_NOTIF_SIZES, 0, &sz) == 0 && sz.seccomp_notif == 80 &&
            sz.seccomp_notif_resp == 24 && sz.seccomp_data == 64,
        "notif sizes");
  unsigned actions[] = {SECCOMP_RET_KILL_PROCESS, SECCOMP_RET_KILL_THREAD, SECCOMP_RET_TRAP,
                        SECCOMP_RET_ERRNO, SECCOMP_RET_USER_NOTIF, SECCOMP_RET_TRACE,
                        SECCOMP_RET_LOG, SECCOMP_RET_ALLOW};
  for (unsigned i = 0; i < sizeof actions / sizeof *actions; i++)
    CHECK(sc(SECCOMP_GET_ACTION_AVAIL, 0, &actions[i]) == 0, "action %#x available", actions[i]);
  unsigned bogus = 0x7fe00000;
  CHECK(sc(SECCOMP_GET_ACTION_AVAIL, 0, &bogus) == -1 && errno == EOPNOTSUPP, "bogus action");
  char text[128] = {0};
  int fd = open("/proc/sys/kernel/seccomp/actions_avail", O_RDONLY);
  CHECK(fd >= 0 && read(fd, text, sizeof text - 1) > 0, "actions_avail");
  CHECK(!strcmp(text, "kill_process kill_thread trap errno user_notif trace log allow\n"),
        "actions_avail is %s", text);
  close(fd);

  /* Values and errors. */
  int l;
  struct seccomp_notif n;
  pid_t pid = spawn(0, t_values, &l);
  CHECK(l >= 0, "listener over SCM_RIGHTS");
  wait_pollin(l);
  memset(&n, 1, sizeof n);
  CHECK(ioctl(l, SECCOMP_IOCTL_NOTIF_RECV, &n) == -1 && errno == EINVAL, "RECV wants zeroes");
  CHECK(recv_notif(l, &n) == 0, "RECV");
  CHECK(n.pid == (__u32)pid, "notif pid %u, want %d", n.pid, pid);
  CHECK(n.data.nr == __NR_getppid, "notif nr %d", n.data.nr);
#ifdef __x86_64__
  CHECK(n.data.arch == 0xc000003e, "arch %#x", n.data.arch);
#endif
  struct pollfd p = {l, POLLIN | POLLOUT, 0};
  CHECK(poll(&p, 1, 0) == 1 && p.revents == POLLOUT, "POLLOUT while unanswered: %#x", p.revents);
  CHECK(ioctl(l, SECCOMP_IOCTL_NOTIF_ID_VALID, &n.id) == 0, "ID_VALID");
  CHECK(reply(l, n.id + 1000, 0, 0, 0) == -1 && errno == ENOENT, "SEND to a bad id");
  CHECK(reply(l, n.id, 1, 0, SECCOMP_USER_NOTIF_FLAG_CONTINUE) == -1 && errno == EINVAL,
        "CONTINUE with a value");
  CHECK(reply(l, n.id, 42, 0, 0) == 0, "SEND 42");
  CHECK(reply(l, n.id, 42, 0, 0) == -1 && (errno == EINPROGRESS || errno == ENOENT),
        "a second reply");
  CHECK(recv_notif(l, &n) == 0 && reply(l, n.id, 0, -EPERM, 0) == 0, "SEND -EPERM");
  for (int i = 0; i < 2; i++)
    CHECK(recv_notif(l, &n) == 0 && reply(l, n.id, 0, 0, SECCOMP_USER_NOTIF_FLAG_CONTINUE) == 0,
          "SEND CONTINUE");
  CHECK(status_of(pid) == 0, "target saw the answers");
  /* Its tasks gone: POLLHUP, and RECV has nothing to wait for. */
  p.events = POLLIN;
  CHECK(poll(&p, 1, 1000) == 1 && (p.revents & POLLHUP), "POLLHUP: %#x", p.revents);
  CHECK(recv_notif(l, &n) == -1 && errno == ENOENT, "RECV after the target is gone");
  close(l);

  pid = spawn(0, t_continue, &l);
  CHECK(recv_notif(l, &n) == 0 && reply(l, n.id, 0, 0, SECCOMP_USER_NOTIF_FLAG_CONTINUE) == 0,
        "CONTINUE");
  CHECK(status_of(pid) == 0, "CONTINUE ran the syscall");
  close(l);

  /* ADDFD: lowest free, SETFD 100 with O_CLOEXEC, then SEND. */
  int pp[2];
  CHECK(pipe(pp) == 0, "pipe");
  pid = spawn(0, t_addfd, &l);
  CHECK(recv_notif(l, &n) == 0, "RECV for ADDFD");
  struct seccomp_notif_addfd a = {.id = n.id, .srcfd = pp[1]};
  int got = ioctl(l, SECCOMP_IOCTL_NOTIF_ADDFD, &a);
  CHECK(got >= 0, "ADDFD");
  a.flags = SECCOMP_ADDFD_FLAG_SETFD;
  a.newfd = 100;
  a.newfd_flags = O_CLOEXEC;
  CHECK(ioctl(l, SECCOMP_IOCTL_NOTIF_ADDFD, &a) == 100, "ADDFD SETFD");
  a.newfd_flags = O_NONBLOCK;
  CHECK(ioctl(l, SECCOMP_IOCTL_NOTIF_ADDFD, &a) == -1 && errno == EINVAL, "ADDFD bad flags");
  a.newfd_flags = 0;
  a.flags = 0;
  a.newfd = 0;
  a.srcfd = 9999;
  CHECK(ioctl(l, SECCOMP_IOCTL_NOTIF_ADDFD, &a) == -1 && errno == EBADF, "ADDFD bad srcfd");
  CHECK(reply(l, n.id, got, 0, 0) == 0, "SEND the fd");
  CHECK(recv_notif(l, &n) == 0, "RECV for ADDFD|SEND");
  struct seccomp_notif_addfd as = {.id = n.id, .srcfd = pp[1], .flags = SECCOMP_ADDFD_FLAG_SEND};
  CHECK(ioctl(l, SECCOMP_IOCTL_NOTIF_ADDFD, &as) >= 0, "ADDFD|SEND");
  CHECK(reply(l, n.id, 0, 0, 0) == -1 && (errno == EINPROGRESS || errno == ENOENT),
        "no reply after ADDFD|SEND");
  CHECK(status_of(pid) == 0, "target used the added fds");
  close(pp[1]);
  char buf[8] = {0};
  CHECK(read(pp[0], buf, sizeof buf) == 3 && !strcmp(buf, "abc"), "pipe got %s", buf);
  close(pp[0]);
  close(l);

  /* /proc/<pid>/mem: read the argument string, write a new one. */
  pid = spawn(0, t_mem, &l);
  CHECK(recv_notif(l, &n) == 0, "RECV for mem");
  char path[64];
  snprintf(path, sizeof path, "/proc/%d/mem", pid);
  int mfd = open(path, O_RDWR);
  CHECK(mfd >= 0, "open %s", path);
  CHECK(ioctl(l, SECCOMP_IOCTL_NOTIF_ID_VALID, &n.id) == 0, "still valid after open");
  char s[64] = {0};
  CHECK(pread(mfd, s, sizeof s, n.data.args[0]) == sizeof s && !strcmp(s, "the target's secret"),
        "read target memory: %s", s);
  CHECK(lseek(mfd, n.data.args[0], SEEK_SET) == (off_t)n.data.args[0], "lseek mem");
  memset(s, 0, sizeof s);
  CHECK(read(mfd, s, 6) == 6 && !strcmp(s, "the ta"), "read at the offset");
  const char *w = "rewritten by supervisor";
  CHECK(pwrite(mfd, w, strlen(w) + 1, n.data.args[0]) == (ssize_t)strlen(w) + 1, "write mem");
  CHECK(pread(mfd, s, 8, 0) == -1 && errno == EIO, "unmapped address is EIO");
  close(mfd);
  CHECK(reply(l, n.id, 7, 0, 0) == 0, "SEND after mem");
  CHECK(status_of(pid) == 0, "target saw the write");
  close(l);

  /* Two notifications at once, answered in reverse order. */
  pid = spawn(0, t_threads, &l);
  struct seccomp_notif n1, n2;
  CHECK(recv_notif(l, &n1) == 0 && recv_notif(l, &n2) == 0, "two RECVs");
  CHECK(n1.id != n2.id && n1.pid != n2.pid, "distinct notifications");
  CHECK(reply(l, n2.id, 100 + n2.data.args[0], 0, 0) == 0, "reply second");
  CHECK(reply(l, n1.id, 100 + n1.data.args[0], 0, 0) == 0, "reply first");
  CHECK(status_of(pid) == 0, "each thread got its own answer");
  close(l);

  /* A signal interrupts the wait: the notification goes, then a restart
   * makes a new one. */
  pid = spawn(0, t_eintr, &l);
  CHECK(recv_notif(l, &n) == 0, "RECV before signal");
  kill(pid, SIGUSR1);
  wait_gone(l, n.id);
  CHECK(reply(l, n.id, 1, 0, 0) == -1 && errno == ENOENT, "reply to an interrupted one");
  CHECK(recv_notif(l, &n1) == 0, "RECV the second call");
  kill(pid, SIGUSR1);
  wait_gone(l, n1.id);
  CHECK(recv_notif(l, &n2) == 0 && n2.id != n1.id, "restart made a new notification");
  CHECK(reply(l, n2.id, 9, 0, 0) == 0, "reply after restart");
  CHECK(status_of(pid) == 0, "EINTR then restart");
  close(l);

  /* WAIT_KILLABLE_RECV: once received, a handled signal doesn't interrupt. */
  pid = spawn(SECCOMP_FILTER_FLAG_WAIT_KILLABLE_RECV, t_killable, &l);
  CHECK(recv_notif(l, &n) == 0, "RECV killable");
  kill(pid, SIGUSR1);
  usleep(100000);
  CHECK(ioctl(l, SECCOMP_IOCTL_NOTIF_ID_VALID, &n.id) == 0, "survives the signal");
  CHECK(reply(l, n.id, 5, 0, 0) == 0, "reply killable");
  CHECK(status_of(pid) == 0, "killable wait answered, then the handler ran");
  close(l);
  /* ... but SIGKILL still ends it. */
  pid = spawn(SECCOMP_FILTER_FLAG_WAIT_KILLABLE_RECV, t_killable, &l);
  CHECK(recv_notif(l, &n) == 0, "RECV killable 2");
  kill(pid, SIGKILL);
  CHECK(status_of(pid) == 128 + SIGKILL, "SIGKILL ends a killable wait");
  close(l);

  /* Closing the listener answers a waiting notification ENOSYS. */
  pid = spawn(0, t_enosys, &l);
  wait_pollin(l);
  close(l);
  CHECK(status_of(pid) == 0, "ENOSYS once the listener is closed");
  pid = spawn(0, t_exit, &l);
  CHECK(status_of(pid) == 0, "target without notifications");
  p.fd = l;
  p.events = POLLIN;
  CHECK(poll(&p, 1, 1000) == 1 && p.revents == POLLHUP, "POLLHUP only: %#x", p.revents);
  close(l);

  /* A listener left only in a CLONE_FILES table, its filter's task gone:
   * RECV has nobody to wait for. */
  {
    int pp2[2];
    CHECK(pipe(pp2) == 0, "pipe");
    struct clone_args ca = {.flags = CLONE_FILES, .exit_signal = SIGCHLD};
    pid = syscall(__NR_clone3, &ca, sizeof ca);
    if (pid == 0) {
      prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
      long lf = install(__NR_mknodat, SECCOMP_RET_USER_NOTIF, SECCOMP_FILTER_FLAG_NEW_LISTENER);
      if (lf < 0 || dup2(lf, 200) != 200) _exit(1);
      close(lf);
      close(pp2[1]);
      usleep(100000);
      _exit(0);
    }
    close(pp2[1]);
    char c;
    CHECK(read(pp2[0], &c, 1) == 0, "child closed the pipe");
    close(pp2[0]);
    CHECK(status_of(pid) == 0, "CLONE_FILES child");
    CHECK(recv_notif(200, &n) == -1 && errno == ENOENT, "RECV with the filter unused");
    close(200);
  }

  /* The rest of the API, in a child of its own. */
  pid = fork();
  if (!pid) _exit(api_child());
  int st = status_of(pid);
  CHECK(st == 0, "api child failed at step %d", st);
  pid = fork();
  if (!pid) _exit(strict_child());
  CHECK(status_of(pid) == 128 + SIGKILL, "strict mode kills");

  if (fails) {
    printf("%d failures\n", fails);
    return 1;
  }
  puts("PASS");
  return 0;
}
