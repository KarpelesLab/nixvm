/* cgroup v2: a hierarchy of our own under the root cgroup, moving processes,
 * pids.max, the cpu controller (weights, cpu.max bandwidth), the memory
 * controller (memory.max and its out-of-memory kills), cgroup.events,
 * cgroup.kill, cgroup.freeze, the limits on the
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
#include <time.h>
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

static double secs(clockid_t c) { struct timespec t; clock_gettime(c, &t); return t.tv_sec + t.tv_nsec / 1e9; }

/* A number from a "key value" line of a flat-keyed file. */
static long long field(const char *text, const char *key) {
  char k[64];
  snprintf(k, sizeof k, "%s ", key);
  for (const char *l = text; l && *l; l = strchr(l, '\n') ? strchr(l, '\n') + 1 : 0)
    if (!strncmp(l, k, strlen(k))) return atoll(l + strlen(k));
  return -1;
}

static void path(char *out, const char *name) { snprintf(out, 256, "%.200s/%.50s", base, name); }

static void cleanup(void) {
  if (!*base) return;
  put(home, "cgroup.procs", "0");
  const char *kids[] = {"self", "lim", "kd", "fz", "cq", "mm/kid", "mm", "nest/a", "nest", "into", "ns", "zz", 0};
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

  /* The cpu controller: weights, and a cpu.max that holds a spinning child
   * to its share of each period. */
  if (!strstr(get("/sys/fs/cgroup", "cgroup.subtree_control"), "cpu"))
    CHECK(put("/sys/fs/cgroup", "cgroup.subtree_control", "+cpu") == 0);
  CHECK(put(base, "cgroup.subtree_control", "+cpu") == 0);
  CHECK(strstr(get(base, "cgroup.subtree_control"), "cpu pids"));
  char cq[256];
  path(cq, "cq");
  CHECK(mkdir(cq, 0755) == 0);
  CHECK(!strcmp(get(cq, "cpu.weight"), "100\n"));
  CHECK(!strcmp(get(cq, "cpu.max"), "max 100000\n"));
  CHECK(!strcmp(get(cq, "cpu.max.burst"), "0\n"));
  CHECK(!strcmp(get(cq, "cpu.idle"), "0\n"));
  CHECK(access("/sys/fs/cgroup/cpu.max", F_OK) == -1); /* not at the root */
  CHECK(put(cq, "cpu.weight", "300") == 0 && !strcmp(get(cq, "cpu.weight"), "300\n"));
  CHECK(put(cq, "cpu.weight", "0") == -1 && errno == ERANGE);
  CHECK(put(cq, "cpu.weight", "10001") == -1 && errno == ERANGE);
  CHECK(put(cq, "cpu.weight.nice", "0") == 0 && !strcmp(get(cq, "cpu.weight"), "100\n"));
  CHECK(put(cq, "cpu.weight.nice", "20") == -1 && errno == ERANGE);
  CHECK(put(cq, "cpu.idle", "1") == 0 && !strcmp(get(cq, "cpu.weight"), "1\n"));
  CHECK(put(cq, "cpu.weight", "50") == -1 && errno == EINVAL); /* idle */
  CHECK(put(cq, "cpu.idle", "0") == 0 && !strcmp(get(cq, "cpu.weight"), "100\n"));
  CHECK(put(cq, "cpu.idle", "2") == -1 && errno == EINVAL);
  CHECK(put(cq, "cpu.max", "500") == -1 && errno == EINVAL);           /* under 1 ms */
  CHECK(put(cq, "cpu.max", "10000 2000000") == -1 && errno == EINVAL); /* period over 1 s */
  CHECK(put(cq, "cpu.max", "lots") == -1 && errno == EINVAL);
  CHECK(put(cq, "cpu.max", "20000 100000") == 0 && !strcmp(get(cq, "cpu.max"), "20000 100000\n"));
  CHECK(put(cq, "cpu.max.burst", "30000") == -1 && errno == EINVAL);  /* more than the quota */
  CHECK(put(cq, "cpu.max.burst", "5000") == 0 && !strcmp(get(cq, "cpu.max.burst"), "5000\n"));
  CHECK(put(cq, "cpu.max.burst", "0") == 0);
  CHECK(put(cq, "cpu.max", "10000") == 0 && !strcmp(get(cq, "cpu.max"), "10000 100000\n")); /* period kept */
  volatile double *used = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
  c = fork();
  if (!c) {
    if (put(cq, "cgroup.procs", "0")) _exit(1);
    double end = secs(CLOCK_MONOTONIC) + 1.0;
    while (secs(CLOCK_MONOTONIC) < end) ;
    *used = secs(CLOCK_THREAD_CPUTIME_ID);
    _exit(0);
  }
  CHECK(waitpid(c, &st, 0) == c && WIFEXITED(st) && WEXITSTATUS(st) == 0);
  printf("10%% of a CPU for 1 s: %.3f s of CPU time\n", *used);
  CHECK(*used > 0.02 && *used < 0.3);
  const char *cs = get(cq, "cpu.stat");
  CHECK(field(cs, "nr_periods") >= 5 && field(cs, "nr_throttled") >= 5);
  CHECK(field(cs, "throttled_usec") > 300000 && field(cs, "nr_bursts") == 0);
  CHECK(field(get(cq, "cpu.stat.local"), "throttled_usec") > 300000);
  CHECK(field(get(base, "cgroup.stat"), "nr_subsys_cpu") >= 2);
  /* Taking cpu away from the children resets their settings. */
  CHECK(put(base, "cgroup.subtree_control", "-cpu") == 0);
  CHECK(access(cq, F_OK) == 0 && get(cq, "cpu.max")[0] == 0);
  CHECK(put(base, "cgroup.subtree_control", "+cpu") == 0);
  CHECK(!strcmp(get(cq, "cpu.max"), "max 100000\n"));
  CHECK(rmdir(cq) == 0);
  puts("cpu.weight, cpu.idle, cpu.max, cpu.stat: ok");

  /* The memory controller: what a cgroup's processes use, and memory.max,
   * past which one is killed. memory is a domain controller: it can't share a
   * cgroup with processes. */
  if (!strstr(get("/sys/fs/cgroup", "cgroup.subtree_control"), "memory"))
    CHECK(put("/sys/fs/cgroup", "cgroup.subtree_control", "+memory") == 0);
  CHECK(put(base, "cgroup.subtree_control", "+memory") == 0);
  CHECK(put(self, "cgroup.subtree_control", "+memory") == -1 && errno == EBUSY); /* we're in it */
  char mm[256], mk[256];
  path(mm, "mm");
  CHECK(mkdir(mm, 0755) == 0);
  CHECK(!strcmp(get(mm, "memory.max"), "max\n") && !strcmp(get(mm, "memory.high"), "max\n"));
  CHECK(!strcmp(get(mm, "memory.min"), "0\n") && !strcmp(get(mm, "memory.oom.group"), "0\n"));
  CHECK(get(mm, "memory.current")[0] >= '0');
  CHECK(put(mm, "memory.max", "4097") == 0 && !strcmp(get(mm, "memory.max"), "4096\n")); /* whole pages */
  CHECK(put(mm, "memory.max", "lots") == -1 && errno == EINVAL);
  CHECK(put(mm, "memory.high", "1G") == 0 && !strcmp(get(mm, "memory.high"), "1073741824\n"));
  CHECK(put(mm, "memory.oom.group", "2") == -1 && errno == EINVAL);
  CHECK(put(mm, "memory.max", "24M") == 0 && !strcmp(get(mm, "memory.max"), "25165824\n"));
  char sm[300];
  snprintf(sm, sizeof sm, "%s/memory.swap.max", mm);
  if (access(sm, F_OK) == 0) CHECK(put(mm, "memory.swap.max", "0") == 0); /* no swapping out of it */
  /* No processes where a domain controller is handed on. */
  path(mk, "mm/kid");
  CHECK(mkdir(mk, 0755) == 0);
  CHECK(put(mm, "cgroup.subtree_control", "+memory") == 0);
  CHECK(put(mm, "cgroup.procs", "0") == -1 && errno == EBUSY);
  CHECK(put(mm, "cgroup.subtree_control", "-memory") == 0);
  CHECK(rmdir(mk) == 0);
  int up[2], go[2];
  CHECK(pipe(up) == 0 && pipe(go) == 0);
  c = fork();
  if (!c) {
    if (put(mm, "cgroup.procs", "0")) _exit(1);
    char *m = mmap(0, 64 << 20, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    for (long i = 0; i < 8 << 20; i += 4096) m[i] = 1;
    char x = 'y';
    if (write(up[1], &x, 1) != 1 || read(go[0], &x, 1) != 1) _exit(2);
    for (long i = 0; i < 64 << 20; i += 4096) m[i] = 1; /* past memory.max */
    _exit(3);
  }
  CHECK(read(up[0], &r, 1) == 1);
  long long cur = atoll(get(mm, "memory.current"));
  printf("8 MiB touched: memory.current %lld\n", cur);
  CHECK(cur >= 8 << 20 && cur < 24 << 20);
  CHECK(field(get(mm, "memory.stat"), "anon") >= 8 << 20);
  CHECK(write(go[1], "g", 1) == 1);
  CHECK(waitpid(c, &st, 0) == c && WIFSIGNALED(st) && WTERMSIG(st) == SIGKILL);
  const char *ev = get(mm, "memory.events");
  CHECK(field(ev, "oom") >= 1 && field(ev, "oom_kill") == 1 && field(ev, "max") >= 1);
  CHECK(field(get(mm, "memory.events.local"), "oom_kill") == 1);
  long long peak = atoll(get(mm, "memory.peak"));
  CHECK(peak >= 20 << 20 && peak <= 25 << 20);
  CHECK(atoll(get(mm, "memory.current")) < 4 << 20); /* freed as it died */
  /* memory.oom.group: the kill takes the whole cgroup. */
  CHECK(put(mm, "memory.oom.group", "1") == 0);
  c = fork();
  if (!c) {
    if (put(mm, "cgroup.procs", "0")) _exit(1);
    pid_t g = fork();
    if (!g) { pause(); _exit(0); }
    char *m = mmap(0, 64 << 20, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    for (long i = 0; i < 64 << 20; i += 4096) m[i] = 1;
    _exit(3);
  }
  CHECK(waitpid(c, &st, 0) == c && WIFSIGNALED(st) && WTERMSIG(st) == SIGKILL);
  for (int i = 0; i < 100 && strcmp(get(mm, "cgroup.events"), "populated 0\nfrozen 0\n"); i++) usleep(10000);
  CHECK(!strcmp(get(mm, "cgroup.events"), "populated 0\nfrozen 0\n")); /* the grandchild too */
  CHECK(field(get(mm, "memory.events"), "oom_group_kill") == 1);
  CHECK(rmdir(mm) == 0);
  puts("memory.max, memory.current, memory.events, memory.oom.group: ok");

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
  CHECK(!strncmp(get(nest, "cgroup.stat"), "nr_descendants 1\n", 17));
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

  /* cgroup v2 can't rename a cgroup (v1 could). */
  char zz[256];
  path(zz, "zz");
  CHECK(mkdir(zz, 0755) == 0);
  path(p, "zz2");
  CHECK(rename(zz, p) == -1 && errno == EPERM);
  CHECK(rmdir(zz) == 0);
  puts("no rename: ok");

  cleanup();
  CHECK(access(base, F_OK) == -1);
  puts("PASS");
  return 0;
}
