#define _GNU_SOURCE
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <mqueue.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <sys/sem.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <unistd.h>
static int fails;
#define CHECK(c, ...) do { int ok_ = (c); printf("%s: ", ok_ ? "ok  " : "FAIL"); printf(__VA_ARGS__); puts(""); fflush(stdout); if (!ok_) fails++; } while (0)
static void link_of(const char *p, char *b) { ssize_t n = readlink(p, b, 99); b[n < 0 ? 0 : n] = 0; }
static char *host(char *b) { struct utsname u; uname(&u); strcpy(b, u.nodename); return b; }
static int stack_fn(void *a) { char b[80]; sethostname("from-clone", 10); host(b); return strcmp(b, "from-clone") ? 1 : 0; }
static char stk[1 << 16];
int main(void) {
  char before[80], after[80], a[100], b[100];
  /* UTS */
  host(before); sethostname("first", 5); link_of("/proc/self/ns/uts", a); CHECK(strstr(a, "uts:[") == a, "uts link looks right: %s", a);
  fflush(stdout); pid_t c = fork();
  if (!c) { CHECK(!unshare(CLONE_NEWUTS), "unshare uts"); link_of("/proc/self/ns/uts", b); CHECK(strcmp(a, b), "new uts link differs"); host(after); CHECK(!strcmp(after, "first"), "inherits the name: %s", after); sethostname("second", 6); host(after); CHECK(!strcmp(after, "second"), "changes its own"); _exit(fails); }
  int st; waitpid(c, &st, 0); fails += WEXITSTATUS(st); host(after); CHECK(!strcmp(after, "first"), "parent's name untouched: %s", after);
  c = clone(stack_fn, stk + sizeof stk, CLONE_NEWUTS | SIGCHLD, 0); waitpid(c, &st, 0); CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "clone(CLONE_NEWUTS) child works in its own"); host(after); CHECK(!strcmp(after, "first"), "parent unaffected");
  /* setns and stat of links */
  struct stat s1, s2; CHECK(!stat("/proc/self/ns/uts", &s1), "stat through the link");
  int p[2], q[2]; pipe(p); pipe(q); c = fork();
  if (!c) { unshare(CLONE_NEWUTS); sethostname("joinme", 6); char x = 1; write(p[1], &x, 1); read(q[0], &x, 1); _exit(0); }
  char x; read(p[0], &x, 1); char path[64]; sprintf(path, "/proc/%d/ns/uts", c);
  CHECK(!stat(path, &s2) && s2.st_ino != s1.st_ino, "another process's link: stat differs (%lu vs %lu)", (unsigned long)s1.st_ino, (unsigned long)s2.st_ino);
  link_of(path, b); CHECK(strstr(b, "uts:[") == b && strcmp(a, b), "reads as a different namespace");
  int fd = open(path, O_RDONLY); CHECK(fd >= 0, "open it"); struct stat s3; CHECK(!fstat(fd, &s3) && s3.st_ino == s2.st_ino, "fstat matches");
  char check[100]; sprintf(check, "/proc/self/fd/%d", fd); link_of(check, b); CHECK(strstr(b, "uts:[") == b, "fd link reads as the namespace: %s", b);
  CHECK(setns(fd, CLONE_NEWIPC) < 0 && errno == EINVAL, "setns with the wrong type EINVAL");
  CHECK(!setns(fd, CLONE_NEWUTS), "setns"); host(after); CHECK(!strcmp(after, "joinme"), "now sees its name: %s", after); write(q[1], &x, 1); waitpid(c, 0, 0);
  { char t[80] = {0}; int f = open("/proc/sys/kernel/hostname", O_RDONLY); read(f, t, 79); close(f); CHECK(!strcmp(t, "joinme\n"), "/proc/sys/kernel/hostname is the namespace's: %s", t);
    /* Only root of the first user namespace may write sysctls. */
    int top = 0; f = open("/proc/self/uid_map", O_RDONLY); char um[100] = {0}; read(f, um, 99); close(f); unsigned a0, b0, c0; top = sscanf(um, "%u %u %u", &a0, &b0, &c0) == 3 && a0 == 0 && b0 == 0 && c0 == 4294967295u;
    f = open("/proc/sys/kernel/hostname", O_WRONLY); long w = f < 0 ? -1 : write(f, "viaproc\n", 8); int we = errno; close(f);
    if (top) { CHECK(w == 8, "write /proc/sys/kernel/hostname"); host(after); CHECK(!strcmp(after, "viaproc"), "uname sees it: %s", after);
      f = open("/proc/sys/kernel/domainname", O_WRONLY); CHECK(f >= 0 && write(f, "dom.example", 11) == 11, "write domainname"); close(f); getdomainname(t, sizeof t); CHECK(!strcmp(t, "dom.example"), "getdomainname sees it: %s", t); }
    else { CHECK(w < 0 && we == EACCES, "root of a user namespace can't write /proc/sys/kernel/hostname"); host(after); CHECK(!strcmp(after, "joinme"), "name unchanged: %s", after); } }
  CHECK(setns(0, 0) < 0 && errno == EINVAL, "setns on a non-namespace EINVAL"); CHECK(setns(9999, 0) < 0 && errno == EBADF, "setns bad fd EBADF");
  /* Directory listing */
  DIR *d = opendir("/proc/self/ns"); int n = 0, u = 0; struct dirent *e; while (d && (e = readdir(d))) { if (e->d_name[0] != '.') n++; if (!strcmp(e->d_name, "uts")) u = 1; } CHECK(n >= 8 && u, "/proc/self/ns lists %d entries", n);
  /* SysV IPC */
  int m = msgget(0x1234, IPC_CREAT | 0600), sh = shmget(0x1235, 4096, IPC_CREAT | 0600), se = semget(0x1236, 1, IPC_CREAT | 0600); CHECK(m >= 0 && sh >= 0 && se >= 0, "objects in the first namespace");
  fflush(stdout); c = fork();
  if (!c) { int f = 0; if (msgget(0x1234, 0) != m) f = 1; if (unshare(CLONE_NEWIPC)) _exit(99); errno = 0; if (msgget(0x1234, 0) != -1 || errno != ENOENT) f |= 2; if (shmget(0x1235, 0, 0) != -1 || errno != ENOENT) f |= 4; if (semget(0x1236, 0, 0) != -1 || errno != ENOENT) f |= 8; struct msqid_ds ds; if (msgctl(m, IPC_STAT, &ds) != -1 || errno != EINVAL) f |= 16; if (shmat(sh, 0, 0) != (void *)-1) f |= 32; int m2 = msgget(0x1234, IPC_CREAT | 0600); if (m2 < 0) f |= 64; _exit(f); }
  waitpid(c, &st, 0); CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "new IPC namespace sees none of them, can't reach them by id, and can reuse the key (bits %d)", WEXITSTATUS(st));
  CHECK(msgget(0x1234, 0) == m, "first namespace still has its own");
  msgctl(m, IPC_RMID, 0); shmctl(sh, IPC_RMID, 0); semctl(se, 0, IPC_RMID);
  /* mqueues */
  struct mq_attr at = {0}; at.mq_maxmsg = 1; at.mq_msgsize = 8; mqd_t q1 = syscall(SYS_mq_open, "mq1", O_CREAT | O_RDWR, 0600, &at); CHECK(q1 >= 0, "mq in first namespace");
  fflush(stdout); c = fork(); if (!c) { if (unshare(CLONE_NEWIPC)) _exit(99); int r = syscall(SYS_mq_open, "mq1", O_RDWR, 0, 0); _exit(r < 0 && errno == ENOENT ? 0 : 1); } waitpid(c, &st, 0); CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "mq names are per namespace (%#x)", st); syscall(SYS_mq_unlink, "mq1");
  /* Privilege */
  fflush(stdout); c = fork(); if (!c) { setresuid(1000, 1000, 1000); int r = unshare(CLONE_NEWUTS); _exit(r < 0 && errno == EPERM ? 0 : 1); } waitpid(c, &st, 0); CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "non-root can't (%#x)", st);
  puts(fails ? "SOME FAILED" : "PASS"); return fails != 0;
}
