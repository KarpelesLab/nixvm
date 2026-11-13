/* What sandbox setup tools (bubblewrap, the container runtimes) lean on, from
 * inside a user and mount namespace of their own:
 *  - creating a name that already exists on a read-only mount is EEXIST, not
 *    EROFS, so "make sure this directory exists" walks a read-only tree;
 *  - a mount target, chdir, or stat through /proc/self/fd/N acts on what that
 *    descriptor has open, wherever it is now;
 *  - a netlink peek with MSG_TRUNC says how long the next message is;
 *  - SOCK_SEQPACKET socket pairs keep message boundaries and carry fds
 *    (Chromium's sandbox IPC);
 *  - /proc/self/task links once per thread (how Chromium checks it's
 *    single-threaded before sandboxing itself);
 *  - what Chromium's namespace sandbox does next: a CLONE_FS child chroots
 *    into /proc/self/fdinfo/, which its parent shares; a tmpfs belongs to its
 *    mounter, with its mode= option; /proc/self/ns links can be access()ed.
 * Runs as anyone (as root it makes its namespaces all the same). */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mount.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#include <pthread.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>
static int fails;
#define CHECK(c, ...) do { int ok_ = (c); printf("%s: ", ok_ ? "ok  " : "FAIL"); printf(__VA_ARGS__); puts(""); fflush(stdout); if (!ok_) fails++; } while (0)
static int child(int (*f)(void)) { fflush(stdout); pid_t c = fork(); if (!c) _exit(f()); int st; waitpid(c, &st, 0); fails += WIFEXITED(st) ? WEXITSTATUS(st) : 1; return st; }
static int put(const char *path, const char *text) { int f = open(path, O_WRONLY); if (f < 0) return -1; int r = write(f, text, strlen(text)); int e = errno; close(f); errno = e; return r < 0 ? -1 : 0; }
static int sandbox(void) {
  int uid = getuid(), gid = getgid(); char l[64];
  if (unshare(CLONE_NEWUSER | CLONE_NEWNS)) return -1;
  put("/proc/self/setgroups", "deny"); sprintf(l, "0 %d 1\n", uid); put("/proc/self/uid_map", l); sprintf(l, "0 %d 1\n", gid); put("/proc/self/gid_map", l);
  return mount(0, "/", 0, MS_REC | MS_PRIVATE, 0);
}
/* A scratch tree on a tmpfs of our own, under the first writable place there is. */
static char base[256];
static int scratch(void) {
  const char *tries[] = {"/tmp", "/dev/shm", "/run", "/var/tmp", 0};
  for (int i = 0; tries[i]; i++) { sprintf(base, "%s/sbfs.%d", tries[i], getpid()); if (!mkdir(base, 0755)) return mount("tmpfs", base, "tmpfs", 0, 0); }
  return -1;
}
static int t_readonly(void) {
  char p[512], q[512];
  CHECK(!sandbox(), "user and mount namespace (%d)", errno); CHECK(!scratch(), "scratch tmpfs at %s (%d)", base, errno);
  sprintf(p, "%s/ro", base); mkdir(p, 0755); sprintf(q, "%s/ro/dir", base); mkdir(q, 0755); sprintf(q, "%s/ro/file", base); close(open(q, O_CREAT | O_WRONLY, 0644)); sprintf(q, "%s/ro/link", base); symlink("file", q);
  sprintf(q, "%s/view", base); mkdir(q, 0755);
  CHECK(!mount(p, q, 0, MS_BIND | MS_REC, 0), "bind it"); CHECK(!mount(0, q, 0, MS_BIND | MS_REMOUNT | MS_RDONLY, 0), "read-only (%d)", errno);
  sprintf(q, "%s/view/dir", base); CHECK(mkdir(q, 0755) < 0 && errno == EEXIST, "mkdir of an existing directory there: EEXIST (%d)", errno);
  sprintf(q, "%s/view/file", base); CHECK(mkdir(q, 0755) < 0 && errno == EEXIST, "of an existing file: EEXIST (%d)", errno);
  CHECK(mknod(q, S_IFREG | 0644, 0) < 0 && errno == EEXIST, "mknod too (%d)", errno);
  CHECK(symlink("x", q) < 0 && errno == EEXIST, "symlink too (%d)", errno);
  CHECK(open(q, O_CREAT | O_EXCL | O_WRONLY, 0644) < 0 && errno == EEXIST, "O_CREAT|O_EXCL too (%d)", errno);
  int fd = open(q, O_CREAT | O_RDONLY, 0644); CHECK(fd >= 0, "a plain O_CREAT open of it for reading works (%d)", errno); close(fd);
  sprintf(q, "%s/view/link", base); CHECK(mkdir(q, 0755) < 0 && errno == EEXIST, "a symlink's name: EEXIST (%d)", errno);
  sprintf(q, "%s/view/new", base); CHECK(mkdir(q, 0755) < 0 && errno == EROFS, "a new name: EROFS (%d)", errno);
  CHECK(symlink("x", q) < 0 && errno == EROFS, "symlink of a new name: EROFS (%d)", errno);
  sprintf(q, "%s/view", base); int dirfd = open(q, O_PATH | O_DIRECTORY);
  CHECK(mkdirat(dirfd, "dir", 0755) < 0 && errno == EEXIST, "mkdirat through a descriptor: EEXIST (%d)", errno);
  CHECK(mkdirat(dirfd, "new2", 0755) < 0 && errno == EROFS, "and EROFS for a new name (%d)", errno);
  return fails;
}
static int t_fd_paths(void) {
  char p[512], q[512], via[64]; struct stat a, b;
  CHECK(!sandbox(), "user and mount namespace (%d)", errno); CHECK(!scratch(), "scratch tmpfs (%d)", errno);
  sprintf(p, "%s/target", base); mkdir(p, 0755); sprintf(q, "%s/target/below", base); mkdir(q, 0755);
  int other = open(base, O_PATH | O_DIRECTORY); int fd = open(p, O_PATH | O_DIRECTORY | O_NOFOLLOW); CHECK(fd >= 0, "O_PATH descriptor of a directory");
  close(other);
  sprintf(via, "/proc/self/fd/%d", fd);
  CHECK(!stat(via, &a) && !fstat(fd, &b) && a.st_ino == b.st_ino && S_ISDIR(a.st_mode), "stat through /proc/self/fd/N is the directory (%lu %lu)", (unsigned long)a.st_ino, (unsigned long)b.st_ino);
  CHECK(!mount("tmpfs", via, "tmpfs", MS_NOSUID | MS_NODEV, "mode=0755"), "mount on /proc/self/fd/N (%d)", errno);
  sprintf(q, "%s/target/below", base); CHECK(access(q, F_OK) < 0 && errno == ENOENT, "the mount went on the directory: what was there is covered (%d)", errno);
  sprintf(q, "%s/target/fresh", base); CHECK(!mkdir(q, 0755), "and the new tmpfs is there to use");
  /* bubblewrap binds onto a fresh O_PATH descriptor of each destination. */
  sprintf(q, "%s/bindsrc", base); mkdir(q, 0755); sprintf(p, "%s/bindsrc/marker", base); close(open(p, O_CREAT | O_WRONLY, 0644));
  sprintf(p, "%s/target/fresh", base); int dst = open(p, O_PATH | O_DIRECTORY); sprintf(via, "/proc/self/fd/%d", dst);
  CHECK(!mount(q, via, 0, MS_BIND | MS_REC, 0), "bind onto /proc/self/fd/N (%d)", errno);
  sprintf(p, "%s/target/fresh/marker", base); CHECK(access(p, F_OK) == 0, "landed on the right directory (%d)", errno);
  int d2 = open(base, O_RDONLY | O_DIRECTORY); sprintf(via, "/proc/self/fd/%d", d2); CHECK(!chdir(via), "chdir through /proc/self/fd/N (%d)", errno);
  CHECK(access("bindsrc/marker", F_OK) == 0, "lands in that directory (%d)", errno);
  return fails;
}
static int t_netlink_peek(void) {
  int s = socket(AF_NETLINK, SOCK_RAW | SOCK_CLOEXEC, NETLINK_ROUTE); CHECK(s >= 0, "rtnetlink socket (%d)", errno);
  struct { struct nlmsghdr h; struct ifinfomsg i; } req = {{sizeof req, RTM_GETLINK, NLM_F_REQUEST | NLM_F_DUMP, 1, 0}, {AF_UNSPEC}};
  CHECK(send(s, &req, sizeof req, 0) == sizeof req, "dump the links");
  struct iovec iov = {0, 0}; struct msghdr m = {0}; m.msg_iov = &iov; m.msg_iovlen = 1;
  ssize_t n = recvmsg(s, &m, MSG_PEEK | MSG_TRUNC); CHECK(n > (ssize_t)sizeof(struct nlmsghdr), "a zero-length MSG_PEEK|MSG_TRUNC says how long the reply is (%zd)", n);
  CHECK(m.msg_flags & MSG_TRUNC, "and that it didn't fit");
  char *buf = malloc(n > 0 ? n : 1); iov.iov_base = buf; iov.iov_len = n > 0 ? n : 1; m.msg_flags = 0;
  ssize_t got = recvmsg(s, &m, 0); CHECK(got == n, "the read that follows gets just that much (%zd)", got);
  struct nlmsghdr *h = (struct nlmsghdr *)buf; CHECK(got > 0 && h->nlmsg_type == RTM_NEWLINK, "a link (%d)", got > 0 ? h->nlmsg_type : -1);
  while (recv(s, buf, n, MSG_DONTWAIT) > 0) {}
  CHECK(send(s, &req, sizeof req, 0) == sizeof req, "dump them again");
  char small[8]; ssize_t t = recv(s, small, sizeof small, MSG_TRUNC); CHECK(t > (ssize_t)sizeof small, "recv with MSG_TRUNC gives the full length too (%zd)", t);
  return fails;
}
static int t_seqpacket(void) {
  int sv[2]; CHECK(!socketpair(AF_UNIX, SOCK_SEQPACKET, 0, sv), "SOCK_SEQPACKET socketpair (%d)", errno);
  int type = 0; socklen_t l = sizeof type; getsockopt(sv[0], SOL_SOCKET, SO_TYPE, &type, &l); CHECK(type == SOCK_SEQPACKET, "SO_TYPE says so (%d)", type);
  CHECK(write(sv[0], "hello", 5) == 5 && write(sv[0], "wo", 2) == 2, "two messages");
  char b[64]; CHECK(read(sv[1], b, sizeof b) == 5, "a read takes one message, not both");
  CHECK(read(sv[1], b, sizeof b) == 2, "then the next");
  write(sv[0], "abcdef", 6); CHECK(recv(sv[1], b, 2, MSG_TRUNC) == 6, "MSG_TRUNC gives a message's full length");
  write(sv[0], "after", 5); CHECK(read(sv[1], b, sizeof b) == 5 && !memcmp(b, "after", 5), "the rest of a cut message is gone");
  write(sv[0], "xyz", 3); struct iovec iov = {b, 1}; struct msghdr m = {0}; m.msg_iov = &iov; m.msg_iovlen = 1;
  CHECK(recvmsg(sv[1], &m, 0) == 1 && (m.msg_flags & MSG_TRUNC), "recvmsg flags a cut message (%#x)", m.msg_flags);
  int fds[2]; pipe(fds); char cbuf[CMSG_SPACE(sizeof(int))] = {0}; struct msghdr sm = {0}; struct iovec siov = {"f", 1};
  sm.msg_iov = &siov; sm.msg_iovlen = 1; sm.msg_control = cbuf; sm.msg_controllen = sizeof cbuf;
  struct cmsghdr *c = CMSG_FIRSTHDR(&sm); c->cmsg_level = SOL_SOCKET; c->cmsg_type = SCM_RIGHTS; c->cmsg_len = CMSG_LEN(sizeof(int)); memcpy(CMSG_DATA(c), &fds[1], sizeof(int));
  CHECK(sendmsg(sv[0], &sm, 0) == 1, "send a descriptor (%d)", errno);
  char rb[CMSG_SPACE(sizeof(int))]; struct msghdr rm = {0}; struct iovec riov = {b, sizeof b}; rm.msg_iov = &riov; rm.msg_iovlen = 1; rm.msg_control = rb; rm.msg_controllen = sizeof rb;
  CHECK(recvmsg(sv[1], &rm, 0) == 1, "receive it"); struct cmsghdr *rc = CMSG_FIRSTHDR(&rm); int got = -1; if (rc && rc->cmsg_type == SCM_RIGHTS) memcpy(&got, CMSG_DATA(rc), sizeof(int));
  CHECK(got >= 0 && write(got, "p", 1) == 1 && read(fds[0], b, 1) == 1, "and it works");
  int on = 1; CHECK(!setsockopt(sv[1], SOL_SOCKET, SO_PASSCRED, &on, sizeof on), "SO_PASSCRED");
  fflush(stdout); pid_t kid = fork(); if (!kid) { write(sv[0], "c", 1); _exit(0); } waitpid(kid, 0, 0);
  char cb[CMSG_SPACE(sizeof(struct ucred))]; struct msghdr cm = {0}; struct iovec ciov = {b, sizeof b}; cm.msg_iov = &ciov; cm.msg_iovlen = 1; cm.msg_control = cb; cm.msg_controllen = sizeof cb;
  CHECK(recvmsg(sv[1], &cm, 0) == 1, "a message from a child"); struct cmsghdr *cc = CMSG_FIRSTHDR(&cm); struct ucred uc = {0};
  if (cc && cc->cmsg_type == SCM_CREDENTIALS) memcpy(&uc, CMSG_DATA(cc), sizeof uc);
  CHECK(cc && cc->cmsg_level == SOL_SOCKET && cc->cmsg_type == SCM_CREDENTIALS && uc.pid == kid && uc.uid == getuid() && uc.gid == getgid(), "says who sent it (SCM_CREDENTIALS: %d %d %d)", uc.pid, uc.uid, uc.gid);
  close(sv[0]); CHECK(read(sv[1], b, sizeof b) == 0, "EOF once the other end closes");
  return fails;
}
static void *sleeper(void *a) { (void)a; pause(); return 0; }
static int t_task_links(void) {
  struct stat s; CHECK(!stat("/proc/self/task", &s) && s.st_nlink == 3, "one thread: task/ has 3 links (%lu)", (unsigned long)s.st_nlink);
  pthread_t t; pthread_create(&t, 0, sleeper, 0); usleep(20000);
  CHECK(!stat("/proc/self/task", &s) && s.st_nlink == 4, "two threads: 4 (%lu)", (unsigned long)s.st_nlink);
  char p[64]; sprintf(p, "/proc/self/task/%d/status", getpid()); CHECK(access(p, R_OK) == 0, "task/<tid>/status (%d)", errno);
  return fails;
}
static char stack[65536];
static int chroot_fdinfo(void *a) { (void)a; return chroot("/proc/self/fdinfo/") || chdir("/") ? 1 : 0; }
static int t_chromium_sandbox(void) {
  char b[4096]; struct stat s;
  CHECK(!access("/proc/self/ns/user", F_OK) && !access("/proc/self/ns/pid", R_OK), "access() of /proc/self/ns links (%d)", errno);
  int fd = open("/dev/null", O_RDONLY); sprintf(b, "/proc/self/fdinfo/%d", fd); int f = open(b, O_RDONLY); int n = f < 0 ? -1 : read(f, b, sizeof b - 1); if (n > 0) b[n] = 0;
  CHECK(n > 0 && strstr(b, "pos:") && strstr(b, "flags:"), "/proc/self/fdinfo/N says pos and flags (%d)", n); close(f);
  CHECK(!stat("/proc/self/fdinfo", &s) && S_ISDIR(s.st_mode), "/proc/self/fdinfo is a directory");
  f = open("/proc/sys/fs/inotify/max_user_watches", O_RDONLY); CHECK(f >= 0 && read(f, b, 16) > 0, "/proc/sys/fs/inotify/max_user_watches (%d)", errno); close(f);
  CHECK(!sandbox(), "user and mount namespace (%d)", errno);
  CHECK(!scratch(), "scratch tmpfs (%d)", errno);
  CHECK(!stat(base, &s) && s.st_uid == 0 && (s.st_mode & 07777) == 01777, "a tmpfs root is its mounter's, mode 1777 (%d %o)", s.st_uid, s.st_mode & 07777);
  char m[300]; sprintf(m, "%s/m", base); mkdir(m, 0755); CHECK(!mount("tmpfs", m, "tmpfs", 0, "mode=0710"), "tmpfs with mode=0710");
  CHECK(!stat(m, &s) && (s.st_mode & 07777) == 0710, "takes it (%o)", s.st_mode & 07777);
  fflush(stdout); pid_t c = clone(chroot_fdinfo, stack + sizeof stack, CLONE_FS | SIGCHLD, 0); int st; waitpid(c, &st, 0);
  CHECK(c > 0 && WIFEXITED(st) && !WEXITSTATUS(st), "a CLONE_FS child chroots into /proc/self/fdinfo/ (%#x)", st);
  CHECK(access("/proc", F_OK) < 0 && access(base, F_OK) < 0, "and its parent is in that empty root too");
  return fails;
}
int main(void) {
  child(t_readonly); child(t_fd_paths); child(t_netlink_peek); child(t_seqpacket); child(t_task_links); child(t_chromium_sandbox);
  puts(fails ? "SOME FAILED" : "PASS"); return fails != 0;
}
