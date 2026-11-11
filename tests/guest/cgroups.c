/* cgroup v2: a hierarchy of our own under the root cgroup, moving processes,
 * pids.max, the cpu controller (weights, cpu.max bandwidth), the memory
 * controller (memory.max and its out-of-memory kills, also on lowering it;
 * memory.high's throttling, the page cache, memory.reclaim and memory.low),
 * the io controller (io.max, io.stat, io.weight), cgroup.events,
 * cgroup.kill, cgroup.freeze, threaded subtrees, the limits on the
 * tree, CLONE_INTO_CGROUP, cgroup namespaces and /proc/<pid>/cgroup.
 *
 * Needs root and cgroup2 at /sys/fs/cgroup: run it as root on a real system
 * (it cleans up after itself). */
#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
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
#include <sys/sysmacros.h>
#include <sys/vfs.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d, errno %d)\n", #c, __LINE__, errno); cleanup(); exit(1); } } while (0)

static char base[256], home[256], scratch[256];

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
  if (*scratch) {
    char f[300];
    for (int i = 0; i < 3; i++) {
      snprintf(f, sizeof f, "%s.%d", scratch, i);
      unlink(f);
    }
  }
  if (!*base) return;
  put(home, "cgroup.procs", "0");
  const char *kids[] = {"self", "lim", "kd", "fz", "cq", "ga", "gb", "mm/kid", "mm", "io", "pr/a", "pr/b", "pr", "td/t1/t3", "td/t1", "td/t2", "td", "nest/a", "nest", "into", "ns", "zz", 0};
  for (int i = 0; kids[i]; i++) {
    char p[256];
    path(p, kids[i]);
    rmdir(p);
  }
  rmdir(base);
}

/* A thread that tells its tid down a pipe, then waits to be killed. */
static int tid_pipe[2];
static void *idle_thread(void *arg) {
  (void)arg;
  pid_t t = syscall(SYS_gettid);
  if (write(tid_pipe[1], &t, sizeof t) != sizeof t) _exit(9);
  for (;;) pause();
}

/* Whether a newline-separated list of ids has `id`. */
static int listed(const char *list, int id) {
  for (const char *l = list; l && *l; l = strchr(l, '\n') ? strchr(l, '\n') + 1 : 0)
    if (atoi(l) == id) return 1;
  return 0;
}

/* A number from the "key=value" field of a nested-keyed file's line for `dev`. */
static long long keyed(const char *text, const char *dev, const char *key) {
  char k[64];
  snprintf(k, sizeof k, " %s=", key);
  for (const char *l = text; l && *l; l = strchr(l, '\n') ? strchr(l, '\n') + 1 : 0) {
    if (strncmp(l, dev, strlen(dev)) || l[strlen(dev)] != ' ') continue;
    const char *end = strchr(l, '\n');
    const char *v = strstr(l, k);
    if (v && (!end || v < end)) return atoll(v + strlen(k));
  }
  return -1;
}

/* Write `mib` MiB to file `name`, flush it to the disk and drop it from the
 * page cache, so it's read from the disk again. */
static int make_file(const char *name, int mib) {
  static char chunk[64 << 10];
  memset(chunk, 'x', sizeof chunk);
  int fd = open(name, O_CREAT | O_TRUNC | O_WRONLY, 0600);
  if (fd < 0) return -1;
  for (int i = 0; i < mib * 16; i++)
    if (write(fd, chunk, sizeof chunk) != sizeof chunk) return -1;
  if (fsync(fd) || posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)) return -1;
  return close(fd);
}

/* Read all of `name`; how long it took, in milliseconds. */
static long read_file(const char *name) {
  static char chunk[64 << 10];
  double t0 = secs(CLOCK_MONOTONIC);
  int fd = open(name, O_RDONLY);
  if (fd < 0) return -1;
  while (read(fd, chunk, sizeof chunk) > 0) {}
  close(fd);
  return (long)((secs(CLOCK_MONOTONIC) - t0) * 1000);
}

/* In a child moved into cgroup `cg`: read file `name` (`write_mib` 0) or write
 * that many MiB to it and flush them; the milliseconds it took. */
static long io_in(const char *cg, const char *name, int write_mib) {
  int fds[2];
  if (pipe(fds)) return -1;
  pid_t c = fork();
  if (!c) {
    if (put(cg, "cgroup.procs", "0")) _exit(1);
    long ms;
    if (write_mib) {
      double t0 = secs(CLOCK_MONOTONIC);
      ms = make_file(name, write_mib) ? -1 : (long)((secs(CLOCK_MONOTONIC) - t0) * 1000);
    } else {
      ms = read_file(name);
    }
    if (write(fds[1], &ms, sizeof ms) != sizeof ms) _exit(2);
    _exit(0);
  }
  long ms = -1;
  if (read(fds[0], &ms, sizeof ms) != sizeof ms) ms = -1;
  int st;
  waitpid(c, &st, 0);
  close(fds[0]);
  close(fds[1]);
  return ms;
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
  /* cpu.weight shares the CPU between cgroups, each splitting its share among
   * its own tasks: on one CPU, two spinners in a cgroup of weight 200 get as
   * much together as twice one spinner of a cgroup of weight 100. */
  {
    char ga[256], gb[256];
    path(ga, "ga");
    path(gb, "gb");
    CHECK(mkdir(ga, 0755) == 0 && mkdir(gb, 0755) == 0);
    CHECK(put(ga, "cpu.weight", "200") == 0);
    volatile double *cpu = mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    int ready[2], start[2];
    CHECK(pipe(ready) == 0 && pipe(start) == 0);
    pid_t kids[3];
    for (int k = 0; k < 3; k++) {
      kids[k] = fork();
      if (!kids[k]) {
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(0, &one);
        if (put(k < 2 ? ga : gb, "cgroup.procs", "0") || sched_setaffinity(0, sizeof one, &one)) _exit(1);
        char x = 'r';
        if (write(ready[1], &x, 1) != 1 || read(start[0], &x, 1) != 1) _exit(2);
        double end = cpu[3];
        while (secs(CLOCK_MONOTONIC) < end) ;
        cpu[k] = secs(CLOCK_THREAD_CPUTIME_ID);
        _exit(0);
      }
    }
    for (int k = 0; k < 3; k++) CHECK(read(ready[0], &r, 1) == 1);
    cpu[3] = secs(CLOCK_MONOTONIC) + 1.5;
    CHECK(write(start[1], "ggg", 3) == 3);
    for (int k = 0; k < 3; k++) CHECK(waitpid(kids[k], &st, 0) == kids[k] && WIFEXITED(st) && WEXITSTATUS(st) == 0);
    double a = cpu[0] + cpu[1], b = cpu[2];
    printf("two tasks at weight 200 against one at 100: %.3f s and %.3f s of CPU (%.3f %.3f)\n", a, b, cpu[0], cpu[1]);
    CHECK(b > 0.1 && a / b > 1.4 && a / b < 2.8);
    close(ready[0]); close(ready[1]); close(start[0]); close(start[1]);
    CHECK(rmdir(ga) == 0 && rmdir(gb) == 0);
  }
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
  /* Lowering memory.max below what's in use kills, biggest first, until the
   * cgroup fits; the write returns once it does. */
  CHECK(put(mm, "memory.oom.group", "0") == 0 && put(mm, "memory.max", "max") == 0);
  long long kills = field(get(mm, "memory.events"), "oom_kill");
  pid_t small = fork();
  if (!small) {
    char x = 'y';
    if (put(mm, "cgroup.procs", "0") || write(up[1], &x, 1) != 1) _exit(1);
    pause();
    _exit(0);
  }
  CHECK(read(up[0], &r, 1) == 1);
  c = fork();
  if (!c) {
    if (put(mm, "cgroup.procs", "0")) _exit(1);
    char *m = mmap(0, 64 << 20, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    for (long i = 0; i < 8 << 20; i += 4096) m[i] = 1;
    char x = 'y';
    if (write(up[1], &x, 1) != 1) _exit(2);
    pause();
    _exit(3);
  }
  CHECK(read(up[0], &r, 1) == 1);
  CHECK(atoll(get(mm, "memory.current")) >= 8 << 20);
  CHECK(put(mm, "memory.max", "4M") == 0);
  CHECK(atoll(get(mm, "memory.current")) <= 4 << 20);
  CHECK(waitpid(c, &st, 0) == c && WIFSIGNALED(st) && WTERMSIG(st) == SIGKILL);
  long long now_kills = field(get(mm, "memory.events"), "oom_kill");
  int small_gone = waitpid(small, &st, WNOHANG) == small;
  printf("memory.max lowered: %lld kill(s), the small one %s\n", now_kills - kills,
         small_gone ? "killed too" : "kept");
  /* The big one goes first. Linux may take the small one too, if it looks
   * again before the big one's memory is all freed; nixvm frees it as the task
   * dies, so it stops at one. */
  CHECK(now_kills == kills + 1 + small_gone);
  if (!small_gone) {
    kill(small, SIGKILL);
    CHECK(waitpid(small, &st, 0) == small);
  }
  /* memory.high: past it, with nothing to reclaim, an allocating task sleeps
   * off its overage, longer the further over it is. */
  CHECK(put(mm, "memory.max", "max") == 0 && put(mm, "memory.high", "8M") == 0);
  long long highs = field(get(mm, "memory.events"), "high");
  c = fork();
  if (!c) {
    if (put(mm, "cgroup.procs", "0")) _exit(1);
    char *m = mmap(0, 16 << 20, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (long i = 0; i < (8 << 20) + (512 << 10); i += 4096) m[i] = 1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    long ms = (t1.tv_sec - t0.tv_sec) * 1000 + (t1.tv_nsec - t0.tv_nsec) / 1000000;
    if (write(up[1], &ms, sizeof ms) != sizeof ms) _exit(2);
    _exit(0);
  }
  long ms = 0;
  CHECK(read(up[0], &ms, sizeof ms) == sizeof ms);
  CHECK(waitpid(c, &st, 0) == c && WIFEXITED(st) && WEXITSTATUS(st) == 0);
  printf("half a MiB past memory.high: %ld ms\n", ms);
  CHECK(ms >= 150 && ms < 20000);
  CHECK(field(get(mm, "memory.events"), "high") > highs);
  CHECK(rmdir(mm) == 0);
  puts("memory.max, memory.current, memory.events, memory.oom.group: ok");

  /* Files on a disk: the io controller limits and counts the I/O to it, and
   * the page cache holding their data is charged to the memory cgroup that
   * read them. A scratch file in the working directory (in nixvm a host
   * directory, /work, which is a disk of its own). */
  struct stat ws;
  /* Unique beyond the pid: VMs running side by side all have a pid 1. */
  struct timespec uniq;
  clock_gettime(CLOCK_REALTIME, &uniq);
  snprintf(scratch, sizeof scratch, "%s/.nixvm-cgroups-%d-%ld",
           stat("/work", &ws) == 0 && access("/work", W_OK) == 0 ? "/work" : ".", getpid(),
           (long)(uniq.tv_nsec ^ uniq.tv_sec));
  char f0[300], f1[300], f2[300];
  snprintf(f0, sizeof f0, "%s.0", scratch);
  snprintf(f1, sizeof f1, "%s.1", scratch);
  snprintf(f2, sizeof f2, "%s.2", scratch);
  CHECK(make_file(f0, 2) == 0);
  CHECK(stat(f0, &ws) == 0);
  /* io.max names a whole disk: the one holding a partition, if it's on one. */
  char sys[300], real[4096], disk[32];
  snprintf(sys, sizeof sys, "/sys/dev/block/%u:%u", major(ws.st_dev), minor(ws.st_dev));
  int on_disk = realpath(sys, real) != 0;
  if (on_disk) {
    snprintf(sys, sizeof sys, "%.4000s/partition", real);
    if (access(sys, F_OK) == 0) *strrchr(real, '/') = 0;
    snprintf(sys, sizeof sys, "%.4000s", real);
    snprintf(disk, sizeof disk, "%s", get(sys, "dev"));
    if (strchr(disk, '\n')) *strchr(disk, '\n') = 0;
  }
  if (!on_disk || !strstr(get("/sys/fs/cgroup", "cgroup.controllers"), "io")) {
    puts("the working directory isn't on a disk, or no io controller: io skipped");
  } else {
    if (!strstr(get("/sys/fs/cgroup", "cgroup.subtree_control"), "io"))
      CHECK(put("/sys/fs/cgroup", "cgroup.subtree_control", "+io") == 0);
    CHECK(put(base, "cgroup.subtree_control", "+io") == 0);
    char io[256], line[128];
    path(io, "io");
    CHECK(mkdir(io, 0755) == 0);
    CHECK(!strcmp(get(io, "io.max"), ""));
    CHECK(!strncmp(get(io, "io.weight"), "default 100\n", 12));
    CHECK(put(io, "io.weight", "50") == 0 && !strncmp(get(io, "io.weight"), "default 50\n", 11));
    CHECK(put(io, "io.weight", "default 0") == -1 && errno == EINVAL);
    snprintf(line, sizeof line, "%s bogus=1", disk);
    CHECK(put(io, "io.max", line) == -1 && errno == EINVAL);
    snprintf(line, sizeof line, "%s rbps=0", disk);
    CHECK(put(io, "io.max", line) == -1 && errno == ERANGE);
    CHECK(put(io, "io.max", "4000:4000 rbps=1") == -1 && errno == ENODEV);
    snprintf(line, sizeof line, "%s rbps=max", disk);
    CHECK(put(io, "io.max", line) == 0 && !strcmp(get(io, "io.max"), ""));
    /* Writes past wbps wait for it: 2 MiB at 1 MiB/s. */
    snprintf(line, sizeof line, "%s wbps=1048576", disk);
    CHECK(put(io, "io.max", line) == 0);
    char want_max[128];
    snprintf(want_max, sizeof want_max, "%s rbps=max wbps=1048576 riops=max wiops=max\n", disk);
    CHECK(!strcmp(get(io, "io.max"), want_max));
    long ms = io_in(io, f1, 2);
    printf("2 MiB written at wbps 1 MiB/s: %ld ms\n", ms);
    CHECK(ms >= 1200 && ms < 15000);
    CHECK(keyed(get(io, "io.stat"), disk, "wbytes") >= 2 << 20);
    CHECK(keyed(get(io, "io.stat"), disk, "wios") > 0);
    /* And reads past rbps: the file is read from the disk, not the cache. */
    snprintf(line, sizeof line, "%s wbps=max rbps=1048576", disk);
    CHECK(put(io, "io.max", line) == 0);
    ms = io_in(io, f0, 0);
    printf("2 MiB read at rbps 1 MiB/s: %ld ms\n", ms);
    CHECK(ms >= 1200 && ms < 15000);
    CHECK(keyed(get(io, "io.stat"), disk, "rbytes") >= 2 << 20);
    CHECK(keyed(get("/sys/fs/cgroup", "io.stat"), disk, "rbytes") >= 2 << 20);
    /* What it read is in the page cache now, charged to it, and read again
     * from there it's no longer held back. */
    long long file = field(get(io, "memory.stat"), "file");
    printf("page cache charged after the read: %lld KiB\n", file >> 10);
    CHECK(file >= 1 << 20);
    CHECK(atoll(get(io, "memory.current")) >= file);
    ms = io_in(io, f0, 0);
    CHECK(ms >= 0 && ms < 500);
    /* memory.reclaim takes it back; asking for more than there is fails. */
    CHECK(put(io, "memory.reclaim", "1G") == -1 && errno == EAGAIN);
    CHECK(field(get(io, "memory.stat"), "file") < 512 << 10);
    CHECK(put(io, "memory.reclaim", "nonsense") == -1 && errno == EINVAL);
    CHECK(rmdir(io) == 0);
    puts("io.max, io.stat, io.weight, page cache, memory.reclaim: ok");
  }

  /* memory.low: reclaim within a parent's memory.max takes from the sibling
   * that isn't protected. */
  {
    char pr[256], pa[256], pb[256];
    path(pr, "pr");
    path(pa, "pr/a");
    path(pb, "pr/b");
    CHECK(mkdir(pr, 0755) == 0 && mkdir(pa, 0755) == 0 && mkdir(pb, 0755) == 0);
    CHECK(put(pr, "cgroup.subtree_control", "+memory") == 0);
    CHECK(put(pr, "memory.max", "16M") == 0 && put(pa, "memory.low", "12M") == 0);
    CHECK(make_file(f1, 8) == 0 && make_file(f2, 16) == 0);
    CHECK(io_in(pa, f1, 0) >= 0);
    long long a0 = atoll(get(pa, "memory.current"));
    CHECK(io_in(pb, f2, 0) >= 0 && io_in(pb, f2, 0) >= 0);
    long long a1 = atoll(get(pa, "memory.current")), b1 = atoll(get(pb, "memory.current"));
    printf("memory.low 12M beside a sibling reading 16M under 16M: %lld KiB kept of %lld, the sibling %lld KiB\n",
           a1 >> 10, a0 >> 10, b1 >> 10);
    if (a0 >= 6 << 20) { /* a page cache to protect (a disk to have read from) */
      CHECK(a1 >= a0 - (1 << 20));
      CHECK(b1 <= 9 << 20);
      CHECK(field(get(pa, "memory.events"), "low") == 0);
    }
    CHECK(rmdir(pa) == 0 && rmdir(pb) == 0 && rmdir(pr) == 0);
    puts("memory.low: ok");
  }
  unlink(f0);
  unlink(f1);
  unlink(f2);

  /* Threaded subtrees: a cgroup made "threaded" joins its parent's resource
   * domain, and a process's threads can then be spread across it. */
  char td[256], t1[256], t2[256], t3[256];
  path(td, "td");
  CHECK(mkdir(td, 0755) == 0);
  snprintf(t1, sizeof t1, "%.240s/t1", td);
  snprintf(t2, sizeof t2, "%.240s/t2", td);
  snprintf(t3, sizeof t3, "%.240s/t3", t1);
  CHECK(mkdir(t1, 0755) == 0 && mkdir(t2, 0755) == 0);
  CHECK(!strcmp(get(t1, "cgroup.type"), "domain\n"));
  CHECK(put(t1, "cgroup.type", "domain") == -1 && errno == EINVAL);
  CHECK(put(t1, "cgroup.type", "threaded") == 0);
  CHECK(put(t1, "cgroup.type", "threaded") == 0); /* already */
  CHECK(!strcmp(get(t1, "cgroup.type"), "threaded\n"));
  CHECK(!strcmp(get(td, "cgroup.type"), "domain threaded\n"));
  CHECK(!strcmp(get(t2, "cgroup.type"), "domain invalid\n")); /* a domain can't sit in a thread root */
  CHECK(put(t2, "cgroup.procs", "0") == -1 && errno == EOPNOTSUPP);
  CHECK(mkdir(t3, 0755) == 0 && !strcmp(get(t3, "cgroup.type"), "domain invalid\n"));
  CHECK(put(t3, "cgroup.type", "threaded") == 0);
  CHECK(rmdir(t3) == 0 && rmdir(t2) == 0);
  /* Only threaded controllers inside it; no domain one in the thread root. */
  CHECK(put(td, "cgroup.subtree_control", "+memory") == -1 && errno == EOPNOTSUPP);
  CHECK(put(td, "cgroup.subtree_control", "+cpu +pids") == 0);
  CHECK(!strcmp(get(t1, "cgroup.controllers"), "cpu pids\n"));
  CHECK(put(t1, "cgroup.subtree_control", "+pids") == 0 && put(t1, "cgroup.subtree_control", "-pids") == 0);
  /* A process in the thread root, with a second thread moved to t1. */
  CHECK(pipe(tid_pipe) == 0 && pipe(go) == 0);
  c = fork();
  if (!c) {
    pthread_t th;
    if (pthread_create(&th, 0, idle_thread, 0)) _exit(1);
    char x;
    if (read(go[0], &x, 1) != 1) _exit(2);
    _exit(0);
  }
  pid_t tid = 0;
  CHECK(read(tid_pipe[0], &tid, sizeof tid) == sizeof tid);
  snprintf(num, sizeof num, "%d", c);
  CHECK(put(td, "cgroup.procs", num) == 0); /* "domain threaded" takes processes */
  snprintf(num, sizeof num, "%d", tid);
  CHECK(put(t1, "cgroup.threads", num) == 0);
  CHECK(listed(get(t1, "cgroup.threads"), tid) && !listed(get(t1, "cgroup.threads"), c));
  CHECK(listed(get(td, "cgroup.threads"), c) && !listed(get(td, "cgroup.threads"), tid));
  CHECK(listed(get(td, "cgroup.procs"), c)); /* the process is the thread root's */
  CHECK(!strcmp(get(td, "cgroup.events"), "populated 1\nfrozen 0\n"));
  {
    char pp[512];
    snprintf(pp, sizeof pp, "%s/cgroup.procs", t1);
    int fd = open(pp, O_RDONLY);
    char b[64];
    CHECK(fd >= 0 && read(fd, b, sizeof b) == -1 && errno == EOPNOTSUPP);
    close(fd);
  }
  CHECK(put(t1, "cgroup.kill", "1") == -1 && errno == EOPNOTSUPP); /* killing is per process */
  CHECK(put(self, "cgroup.threads", num) == -1 && errno == EOPNOTSUPP); /* out of its domain */
  CHECK(put(t1, "cgroup.type", "threaded") == 0);
  /* A populated domain can't become threaded; no domain controller now. */
  CHECK(put(td, "cgroup.type", "threaded") == -1 && errno == EOPNOTSUPP);
  /* Writing cgroup.procs anywhere in the subtree moves every thread. */
  snprintf(num, sizeof num, "%d", c);
  CHECK(put(t1, "cgroup.procs", num) == 0);
  CHECK(listed(get(t1, "cgroup.threads"), c) && listed(get(t1, "cgroup.threads"), tid));
  CHECK(listed(get(td, "cgroup.procs"), c) && !strcmp(get(td, "cgroup.threads"), ""));
  CHECK(write(go[1], "g", 1) == 1);
  CHECK(waitpid(c, &st, 0) == c && WIFEXITED(st) && WEXITSTATUS(st) == 0);
  for (int i = 0; i < 100 && strcmp(get(td, "cgroup.events"), "populated 0\nfrozen 0\n"); i++) usleep(10000);
  CHECK(!strcmp(get(td, "cgroup.events"), "populated 0\nfrozen 0\n"));
  CHECK(!strcmp(get(td, "cgroup.type"), "domain threaded\n")); /* t1 is still threaded */
  CHECK(rmdir(t1) == 0);
  CHECK(!strcmp(get(td, "cgroup.type"), "domain\n")); /* back to a plain domain */
  CHECK(rmdir(td) == 0);
  close(tid_pipe[0]);
  close(tid_pipe[1]);
  puts("threaded subtrees, cgroup.type, cgroup.threads: ok");

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
