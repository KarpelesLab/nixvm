/* cgroup v2: a hierarchy of our own under the root cgroup, moving processes,
 * pids.max, cgroup.events, cgroup.kill, cgroup.freeze, the limits on the
 * tree, CLONE_INTO_CGROUP, cgroup namespaces and /proc/<pid>/cgroup.
 *
 * Needs root and cgroup2 at /sys/fs/cgroup: run it as root on a real system
 * (it cleans up after itself). */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <linux/sched.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/vfs.h>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d, errno %d)\n", #c, __LINE__, errno); cleanup(); exit(1); } } while (0)

static char base[256], home[256];

static int put(const char *dir, const char *file, const char *text) {
  char p[512];
  snprintf(p, sizeof p, "%s/%s", dir, file);
  int fd = open(p, O_WRONLY);
  if (fd < 0) return -1;
  int n = write(fd, text, strlen(text));
  int e = errno;
  close(fd);
  errno = n < 0 ? e : 0; /* a stale errno would mislead a failing check */
  return n < 0 ? -1 : 0;
}

static const char *get(const char *dir, const char *file) {
  static char buf[4096];
  char p[512];
  snprintf(p, sizeof p, "%s/%s", dir, file);
  int fd = open(p, O_RDONLY);
  if (fd < 0) return "";
  int n = read(fd, buf, sizeof buf - 1);
  close(fd);
  buf[n < 0 ? 0 : n] = 0;
  return buf;
}

static void path(char *out, const char *name) { snprintf(out, 256, "%.200s/%.50s", base, name); }

static void cleanup(void) {
  if (!*base) return;
  put(home, "cgroup.procs", "0");
  const char *kids[] = {"zz/inner", "self", "lim", "kd", "fz", "nest/a", "nest", "into", "ns", "zz", 0};
  for (int i = 0; kids[i]; i++) {
    char p[256];
    path(p, kids[i]);
    rmdir(p);
  }
  rmdir(base);
}

/* The calling process's cgroup, from /proc/self/cgroup. */
static const char *my_cgroup(void) {
  static char buf[512];
  int fd = open("/proc/self/cgroup", O_RDONLY);
  int n = read(fd, buf, sizeof buf - 1);
  close(fd);
  buf[n < 0 ? 0 : n] = 0;
  char *nl = strchr(buf, '\n');
  if (nl) *nl = 0;
  return strncmp(buf, "0::", 3) ? buf : buf + 3;
}

int main(void) {
  struct statfs sf;
  CHECK(statfs("/sys/fs/cgroup", &sf) == 0 && sf.f_type == 0x63677270);
  puts("cgroup2 at /sys/fs/cgroup: ok");

  snprintf(home, sizeof home, "/sys/fs/cgroup%s", my_cgroup());
  snprintf(base, sizeof base, "/sys/fs/cgroup/nixvm-test-%d", getpid());
  CHECK(mkdir(base, 0755) == 0);
  CHECK(mkdir(base, 0755) == -1 && errno == EEXIST);
  if (!strstr(get("/sys/fs/cgroup", "cgroup.subtree_control"), "pids"))
    CHECK(put("/sys/fs/cgroup", "cgroup.subtree_control", "+pids") == 0);
  CHECK(strstr(get(base, "cgroup.controllers"), "pids"));
  CHECK(!strcmp(get(base, "cgroup.events"), "populated 0\nfrozen 0\n"));

  /* Into a leaf of our own, so base has no processes and can hand pids on. */
  char self[256], lim[256], p[256];
  path(self, "self");
  CHECK(mkdir(self, 0755) == 0);
  CHECK(put(self, "cgroup.procs", "0") == 0);
  char want[300];
  snprintf(want, sizeof want, "%s/self", base + strlen("/sys/fs/cgroup"));
  CHECK(!strcmp(my_cgroup(), want));
  char me[32];
  snprintf(me, sizeof me, "%d\n", getpid());
  CHECK(!strcmp(get(self, "cgroup.procs"), me));
  CHECK(!strcmp(get(base, "cgroup.events"), "populated 1\nfrozen 0\n"));
  CHECK(put(self, "cgroup.subtree_control", "+pids") == -1 && errno == ENOENT); /* base hasn't */
  CHECK(put(base, "cgroup.subtree_control", "+pids") == 0);
  /* pids is a threaded controller, so a cgroup with processes may enable it. */
  CHECK(put(self, "cgroup.subtree_control", "+pids") == 0);
  CHECK(put(self, "cgroup.subtree_control", "-pids") == 0);
  CHECK(!strcmp(get(base, "cgroup.subtree_control"), "pids\n"));
  /* But base hands pids to a populated child: no processes go in base. */
  CHECK(put(base, "cgroup.procs", "0") == -1 && errno == EBUSY);
  CHECK(put(base, "cgroup.subtree_control", "+nosuch") == -1 && errno == EINVAL);
  CHECK(rmdir(self) == -1 && errno == EBUSY);
  puts("moving ourselves, populated, no internal processes: ok");

  /* pids.max: a process that may have two tasks gets EAGAIN at a third. */
  path(lim, "lim");
  CHECK(mkdir(lim, 0755) == 0);
  CHECK(!strcmp(get(lim, "pids.max"), "max\n"));
  CHECK(put(lim, "pids.max", "2") == 0);
  CHECK(put(lim, "pids.max", "lots") == -1 && errno == EINVAL);
  int fds[2];
  CHECK(pipe(fds) == 0);
  pid_t c = fork();
  if (!c) {
    if (put(lim, "cgroup.procs", "0")) _exit(10);
    pid_t g = fork();
    if (!g) { pause(); _exit(0); }
    if (g < 0) _exit(11);
    pid_t h = fork();
    int failed = h < 0 && errno == EAGAIN;
    if (h == 0) _exit(0);
    char r = failed ? 'y' : 'n';
    if (write(fds[1], &r, 1) != 1) _exit(12);
    kill(g, SIGKILL);
    waitpid(g, 0, 0);
    _exit(failed ? 0 : 13);
  }
  char r;
  CHECK(read(fds[0], &r, 1) == 1 && r == 'y');
  int st;
  CHECK(waitpid(c, &st, 0) == c && WIFEXITED(st) && WEXITSTATUS(st) == 0);
  CHECK(!strcmp(get(lim, "pids.events"), "max 1\n"));
  CHECK(!strcmp(get(lim, "pids.current"), "0\n"));
  CHECK(!strcmp(get(lim, "cgroup.events"), "populated 0\nfrozen 0\n"));
  CHECK(rmdir(lim) == 0);
  puts("pids.max, pids.events, pids.current: ok");

  /* Another process's cgroup in /proc/<pid>/cgroup, and in cgroup.procs. */
  char kd[256];
  path(kd, "kd");
  CHECK(mkdir(kd, 0755) == 0);
  c = fork();
  if (!c) { pause(); _exit(0); }
  char num[32];
  snprintf(num, sizeof num, "%d", c);
  CHECK(put(kd, "cgroup.procs", num) == 0);
  snprintf(p, sizeof p, "/proc/%d/cgroup", c);
  int fd = open(p, O_RDONLY);
  char line[300] = {0};
  CHECK(fd >= 0 && read(fd, line, sizeof line - 1) > 0);
  close(fd);
  snprintf(want, sizeof want, "0::%s/kd\n", base + strlen("/sys/fs/cgroup"));
  CHECK(!strcmp(line, want));
  snprintf(want, sizeof want, "%d\n", c);
  CHECK(!strcmp(get(kd, "cgroup.procs"), want));
  CHECK(put(kd, "cgroup.threads", "0") == -1);
  puts("moving another process, /proc/<pid>/cgroup: ok");

  /* cgroup.kill. */
  CHECK(put(kd, "cgroup.kill", "1") == 0);
  CHECK(waitpid(c, &st, 0) == c && WIFSIGNALED(st) && WTERMSIG(st) == SIGKILL);
  CHECK(rmdir(kd) == 0);
  puts("cgroup.kill: ok");

  /* cgroup.freeze: a spinning child stops counting while frozen. */
  char fz[256];
  path(fz, "fz");
  CHECK(mkdir(fz, 0755) == 0);
  volatile long *counter = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  c = fork();
  if (!c) { for (;;) (*counter)++; }
  snprintf(num, sizeof num, "%d", c);
  CHECK(put(fz, "cgroup.procs", num) == 0);
  usleep(50000);
  CHECK(put(fz, "cgroup.freeze", "1") == 0);
  for (int i = 0; i < 100 && !strstr(get(fz, "cgroup.events"), "frozen 1"); i++) usleep(10000);
  CHECK(!strcmp(get(fz, "cgroup.events"), "populated 1\nfrozen 1\n"));
  usleep(20000);
  long before = *counter;
  usleep(100000);
  CHECK(*counter == before);
  CHECK(put(fz, "cgroup.freeze", "0") == 0);
  usleep(100000);
  CHECK(*counter != before);
  CHECK(put(fz, "cgroup.freeze", "1") == 0);
  kill(c, SIGKILL); /* a frozen task still dies */
  CHECK(waitpid(c, &st, 0) == c && WIFSIGNALED(st));
  CHECK(rmdir(fz) == 0);
  puts("cgroup.freeze: ok");

  /* Limits on the tree. */
  char nest[256], a[256];
  path(nest, "nest");
  CHECK(mkdir(nest, 0755) == 0);
  CHECK(put(nest, "cgroup.max.descendants", "1") == 0);
  path(a, "nest/a");
  CHECK(mkdir(a, 0755) == 0);
  path(p, "nest/b");
  CHECK(mkdir(p, 0755) == -1 && errno == EAGAIN);
  CHECK(put(nest, "cgroup.max.depth", "0") == 0);
  path(p, "nest/a/deeper");
  CHECK(mkdir(p, 0755) == -1 && errno == EAGAIN);
  CHECK(rmdir(nest) == -1 && errno == EBUSY);
  CHECK(!strcmp(get(nest, "cgroup.stat"), "nr_descendants 1\nnr_dying_descendants 0\n"));
  CHECK(rmdir(a) == 0 && rmdir(nest) == 0);
  puts("cgroup.max.descendants, cgroup.max.depth: ok");

  /* clone3 with CLONE_INTO_CGROUP. */
  char into[256];
  path(into, "into");
  CHECK(mkdir(into, 0755) == 0);
  int dfd = open(into, O_RDONLY | O_DIRECTORY);
  CHECK(dfd >= 0);
  struct clone_args ca = {.flags = CLONE_INTO_CGROUP, .exit_signal = SIGCHLD, .cgroup = dfd};
  c = syscall(SYS_clone3, &ca, sizeof ca);
  if (c == 0) {
    char w[300];
    snprintf(w, sizeof w, "%s/into", base + strlen("/sys/fs/cgroup"));
    _exit(strcmp(my_cgroup(), w) ? 1 : 0);
  }
  CHECK(c > 0);
  CHECK(waitpid(c, &st, 0) == c && WIFEXITED(st) && WEXITSTATUS(st) == 0);
  close(dfd);
  CHECK(rmdir(into) == 0);
  puts("CLONE_INTO_CGROUP: ok");

  /* A cgroup namespace is rooted where it's made, and a cgroup2 mount made
   * in it shows the tree from there. */
  char ns[256];
  path(ns, "ns");
  CHECK(mkdir(ns, 0755) == 0);
  c = fork();
  if (!c) {
    if (put(ns, "cgroup.procs", "0")) _exit(1);
    if (unshare(CLONE_NEWCGROUP)) _exit(2);
    if (strcmp(my_cgroup(), "/")) _exit(3);
    char w[300];
    snprintf(w, sizeof w, "/proc/%d/cgroup", getppid());
    int f = open(w, O_RDONLY);
    char l[300] = {0};
    if (f < 0 || read(f, l, sizeof l - 1) <= 0) _exit(4);
    if (strcmp(l, "0::/../self\n")) _exit(5);
    /* A cgroup2 mount made here has this namespace's root as its own. */
    if (unshare(CLONE_NEWNS) || mount(0, "/", 0, MS_REC | MS_PRIVATE, 0)) _exit(6);
    mkdir("/tmp/cgns", 0755);
    if (mount("none", "/tmp/cgns", "cgroup2", 0, 0)) _exit(7);
    char mine[32];
    snprintf(mine, sizeof mine, "%d\n", getpid());
    if (strcmp(get("/tmp/cgns", "cgroup.procs"), mine)) _exit(8);
    if (access("/tmp/cgns/ns", F_OK) == 0 || access("/tmp/cgns/self", F_OK) == 0) _exit(9);
    umount("/tmp/cgns");
    rmdir("/tmp/cgns");
    _exit(0);
  }
  CHECK(waitpid(c, &st, 0) == c && WIFEXITED(st));
  printf("cgroup namespace child: %d\n", WEXITSTATUS(st));
  CHECK(WEXITSTATUS(st) == 0);
  CHECK(rmdir(ns) == 0);
  puts("cgroup namespaces: ok");

  /* Renaming within the parent. */
  char zz[256];
  path(zz, "zz");
  CHECK(mkdir(zz, 0755) == 0);
  path(p, "zz/inner");
  CHECK(mkdir(p, 0755) == 0);
  char renamed[256];
  path(renamed, "zz/outer");
  CHECK(rename(p, renamed) == 0);
  CHECK(rmdir(renamed) == 0);
  puts("rename: ok");

  cleanup();
  CHECK(access(base, F_OK) == -1);
  puts("PASS");
  return 0;
}
