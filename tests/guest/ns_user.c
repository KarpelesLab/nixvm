#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <grp.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mount.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#include <linux/capability.h>
static int fails;
#define CHECK(c, ...) do { int ok_ = (c); printf("%s: ", ok_ ? "ok  " : "FAIL"); printf(__VA_ARGS__); puts(""); fflush(stdout); if (!ok_) fails++; } while (0)
static int child(int (*f)(void)) { fflush(stdout); pid_t c = fork(); if (!c) _exit(f()); int st; waitpid(c, &st, 0); fails += WIFEXITED(st) ? WEXITSTATUS(st) : 1; return st; }
static void as_user(void) { if (getuid() == 0) { setresgid(1000, 1000, 1000); setresuid(1000, 1000, 1000); } }
static int put(const char *path, const char *text) { int f = open(path, O_WRONLY); if (f < 0) return -1; int r = write(f, text, strlen(text)); int e = errno; close(f); errno = e; return r < 0 ? -1 : 0; }
static char *get(const char *p, char *b, int n) { b[0] = 0; int f = open(p, O_RDONLY); if (f >= 0) { int r = read(f, b, n - 1); b[r < 0 ? 0 : r] = 0; close(f); } return b; }
static long has_cap(int cap) { struct __user_cap_header_struct h = {_LINUX_CAPABILITY_VERSION_3, 0}; struct __user_cap_data_struct d[2]; if (syscall(SYS_capget, &h, d)) return -1; return (d[cap / 32].effective >> (cap % 32)) & 1; }
static int t_basics(void) {
  char b[200]; as_user(); int uid = getuid();
  CHECK(unshare(CLONE_NEWUTS) < 0 && errno == EPERM, "an ordinary user can't make a UTS namespace alone");
  CHECK(has_cap(CAP_SYS_ADMIN) == 0, "and has no capabilities");
  CHECK(!unshare(CLONE_NEWUSER), "but can make a user namespace");
  CHECK(getuid() == 65534 && geteuid() == 65534 && getgid() == 65534, "unmapped: overflow ids (%d %d)", getuid(), getgid());
  CHECK(has_cap(CAP_SYS_ADMIN) == 1, "with every capability there");
  CHECK(!strcmp(get("/proc/self/uid_map", b, 200), ""), "empty map (%s)", b);
  CHECK(setuid(0) < 0 && errno == EINVAL, "can't become an unmapped id: EINVAL (%d)", errno);
  CHECK(put("/proc/self/uid_map", "0 0 1\n") < 0 && errno == EPERM, "can't map another user's id (%d)", errno);
  char line[64]; sprintf(line, "0 %d 1\n", uid); CHECK(!put("/proc/self/uid_map", line), "maps its own");
  CHECK(put("/proc/self/uid_map", line) < 0 && errno == EPERM, "only once (%d)", errno);
  CHECK(getuid() == 0 && geteuid() == 0, "now it's uid 0 here (%d)", getuid());
  get("/proc/self/uid_map", b, 200); int i, o, n; CHECK(sscanf(b, "%d %d %d", &i, &o, &n) == 3 && i == 0 && o == uid && n == 1, "map reads back: %s", b);
  sprintf(line, "0 %d 1\n", 1000); CHECK(put("/proc/self/gid_map", line) < 0 && errno == EPERM, "gid_map needs setgroups denied (%d)", errno);
  CHECK(!strcmp(get("/proc/self/setgroups", b, 200), "allow\n"), "setgroups starts allowed");
  CHECK(!put("/proc/self/setgroups", "deny"), "deny it"); CHECK(!strcmp(get("/proc/self/setgroups", b, 200), "deny\n"), "reads deny");
  CHECK(put("/proc/self/setgroups", "allow") < 0 && errno == EPERM, "can't undo");
  sprintf(line, "0 %d 1\n", getegid() == 65534 ? 1000 : 1000); CHECK(!put("/proc/self/gid_map", line), "then gid_map works (%d)", errno); CHECK(getgid() == 0, "gid 0 here (%d)", getgid());
  gid_t none = 0; CHECK(setgroups(1, &none) < 0 && errno == EPERM, "setgroups refused (%d)", errno);
  CHECK(!setuid(0) && !setgid(0), "setuid to a mapped id works"); CHECK(setuid(5) < 0 && errno == EINVAL, "to an unmapped one EINVAL (%d)", errno);
  return fails;
}
static void map_root(int uid, int gid) { char l[64]; put("/proc/self/setgroups", "deny"); sprintf(l, "0 %d 1\n", uid); put("/proc/self/uid_map", l); sprintf(l, "0 %d 1\n", gid); put("/proc/self/gid_map", l); }
static int t_scope(void) {
  char b[200]; struct utsname u, v; as_user(); int uid = getuid(), gid = getgid(); uname(&u);
  CHECK(!unshare(CLONE_NEWUSER), "user namespace"); map_root(uid, gid);
  CHECK(sethostname("nope", 4) < 0 && errno == EPERM, "root here still can't rename a UTS namespace it doesn't own (%d)", errno);
  CHECK(mount("tmpfs", "/tmp", "tmpfs", 0, 0) < 0 && errno == EPERM, "nor mount in a mount namespace it doesn't own (%d)", errno);
  CHECK(!unshare(CLONE_NEWUTS), "its own UTS namespace"); CHECK(!sethostname("mine", 4), "that it can rename"); uname(&v); CHECK(!strcmp(v.nodename, "mine"), "and does");
  CHECK(!unshare(CLONE_NEWNS), "its own mount namespace"); mount(0, "/", 0, MS_REC | MS_PRIVATE, 0); CHECK(!mount("tmpfs", "/tmp", "tmpfs", 0, 0), "mount there (%d)", errno); close(open("/tmp/inside", O_CREAT | O_WRONLY, 0644));
  CHECK(!unshare(CLONE_NEWPID), "its own PID namespace"); fflush(stdout); pid_t c = fork(); if (!c) _exit(getpid() == 1 ? 0 : 1); int st; waitpid(c, &st, 0); CHECK(WIFEXITED(st) && !WEXITSTATUS(st), "works (%#x)", st);
  CHECK(!unshare(CLONE_NEWNET), "net"); CHECK(!unshare(CLONE_NEWIPC), "ipc");
  return fails;
}
static int t_one_call(void) { as_user(); CHECK(!unshare(CLONE_NEWUSER | CLONE_NEWUTS | CLONE_NEWNS), "unshare(USER|UTS|NS) in one call: the new user owns the rest (%d)", errno); map_root(1000, 1000); CHECK(!sethostname("x", 1), "so it can administer them (%d)", errno); return fails; }
static int t_clone(void) { as_user(); fflush(stdout); pid_t c = syscall(SYS_clone, CLONE_NEWUSER | SIGCHLD, 0, 0, 0, 0); if (!c) _exit(getuid() == 65534 && has_cap(CAP_SYS_ADMIN) == 1 ? 0 : 1); int st; waitpid(c, &st, 0); CHECK(WIFEXITED(st) && !WEXITSTATUS(st), "clone(CLONE_NEWUSER) child is unmapped with capabilities (%#x)", st); CHECK(getuid() != 65534 && has_cap(CAP_SYS_ADMIN) == 0, "the parent is unchanged"); return fails; }
static int t_files(void) {
  struct stat s; char b[80]; as_user(); int uid = getuid(), gid = getgid();
  mkdir("/tmp/nsf", 0777); chmod("/tmp/nsf", 0777); const char *own = "/tmp/nsf/own"; close(open(own, O_CREAT | O_WRONLY, 0644)); chown(own, uid, gid); stat(own, &s); int real_uid = s.st_uid;
  CHECK(!unshare(CLONE_NEWUSER), "user namespace"); CHECK(!stat(own, &s) && s.st_uid == 65534, "before a map, its own file shows overflow (%d)", s.st_uid);
  map_root(uid, gid); CHECK(!stat(own, &s) && s.st_uid == 0 && s.st_gid == 0, "after, its file is owned by 0 here (%d %d)", s.st_uid, s.st_gid);
  CHECK(!stat("/", &s) && (s.st_uid == 65534 || s.st_uid == 0), "/ owner (%d)", s.st_uid); if (uid != 0) CHECK(s.st_uid == 65534, "someone else's file shows overflow (%d)", s.st_uid);
  CHECK(chown(own, 5, -1) < 0 && errno == EINVAL, "chown to an unmapped id EINVAL (%d)", errno);
  CHECK(!chown(own, 0, 0), "chown to a mapped one (%d)", errno);
  return fails;
}
static int t_exec_helper(void) { execl(access("/ns5", X_OK) == 0 ? "/ns5" : "/proc/self/exe", "ns5", "--caps", (char *)0); return 99; }
static int t_exec(void) { as_user(); unshare(CLONE_NEWUSER); fflush(stdout); pid_t c = fork(); if (!c) t_exec_helper(); int st; waitpid(c, &st, 0); CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "exec as an unmapped user loses capabilities (%d)", WEXITSTATUS(st)); map_root(1000, 1000); fflush(stdout); c = fork(); if (!c) { execl(access("/ns5", X_OK) == 0 ? "/ns5" : "/proc/self/exe", "ns5", "--caps-root", (char *)0); _exit(99); } waitpid(c, &st, 0); CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "exec as root here keeps them (%d)", WEXITSTATUS(st)); return fails; }
static int t_locked(void) {
  /* The parent side (root in its own user namespace) prepares a read-only mount. */
  unshare(CLONE_NEWNS); mount(0, "/", 0, MS_REC | MS_PRIVATE, 0); mkdir("/tmp/lk", 0755); mount("tmpfs", "/tmp/lk", "tmpfs", 0, 0); mkdir("/tmp/lk/a", 0755); mkdir("/tmp/lk/hid", 0755); close(open("/tmp/lk/hid/secret", O_CREAT | O_WRONLY, 0644)); mkdir("/tmp/lk2", 0755); mount("tmpfs", "/tmp/lk2", "tmpfs", MS_RDONLY, 0); mount("tmpfs", "/tmp/lk/hid", "tmpfs", 0, 0); mount(0, "/tmp/lk2", 0, MS_REMOUNT | MS_RDONLY, 0);
  int pre = access("/tmp/lk/hid/secret", F_OK); CHECK(pre < 0, "a tmpfs hides the file");
  CHECK(!unshare(CLONE_NEWUSER | CLONE_NEWNS), "a less privileged user and mount namespace"); map_root(0, 0);
  mount(0, "/", 0, MS_REC | MS_PRIVATE, 0);
  CHECK(umount("/tmp/lk/hid") < 0 && errno == EINVAL, "can't uncover what an inherited mount hides: EINVAL (%d)", errno); CHECK(access("/tmp/lk/hid/secret", F_OK) < 0, "still hidden");
  CHECK(mount(0, "/tmp/lk2", 0, MS_REMOUNT, 0) < 0 && errno == EPERM, "can't make an inherited read-only mount writable: EPERM (%d)", errno);
  CHECK(!mount("tmpfs", "/tmp/lk/a", "tmpfs", 0, 0), "can stack new mounts on them"); CHECK(!umount("/tmp/lk/a"), "and remove those");
  return fails;
}
static int t_setns(void) {
  as_user(); int uid = getuid(), gid = getgid(); int base = open("/proc/self/ns/user", O_RDONLY); int p[2], q[2]; pipe(p); pipe(q); fflush(stdout); pid_t c = fork();
  if (!c) { unshare(CLONE_NEWUSER); map_root(uid, gid); char x = 1; write(p[1], &x, 1); read(q[0], &x, 1); _exit(0); }
  char x; read(p[0], &x, 1); char path[64]; sprintf(path, "/proc/%d/ns/user", c); int fd = open(path, O_RDONLY); CHECK(fd >= 0, "open a child's user namespace");
  CHECK(setns(base, CLONE_NEWUSER) < 0 && errno == EINVAL, "joining the one already in EINVAL (%d)", errno);
  CHECK(!setns(fd, CLONE_NEWUSER), "join the child's"); CHECK(has_cap(CAP_SYS_ADMIN) == 1 && getuid() == 0, "gets capabilities and a mapped id (%d)", getuid());
  CHECK(setns(base, CLONE_NEWUSER) < 0 && errno == EPERM, "can't go back outward (%d)", errno); x = 1; write(q[1], &x, 1); waitpid(c, 0, 0);
  return fails;
}
static int t_time(void) {
  char b[200]; struct timespec a1, a2; unshare(CLONE_NEWUSER); map_root(getuid(), getgid());
  CHECK(!unshare(CLONE_NEWTIME), "time namespace (%d)", errno); clock_gettime(CLOCK_MONOTONIC, &a1);
  CHECK(!put("/proc/self/timens_offsets", "monotonic 1000 0\nboottime 2000 0\n"), "set offsets before anything enters (%d)", errno);
  clock_gettime(CLOCK_MONOTONIC, &a2); CHECK(a2.tv_sec - a1.tv_sec < 100, "the caller's clock doesn't move");
  fflush(stdout); pid_t c = fork(); if (!c) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); long d = t.tv_sec - a1.tv_sec; clock_gettime(CLOCK_BOOTTIME, &t); struct timespec r; clock_gettime(CLOCK_REALTIME, &r); char o[100]; get("/proc/self/timens_offsets", o, 100); _exit(d >= 999 && d <= 1100 && strstr(o, "monotonic") ? 0 : 1); } int st; waitpid(c, &st, 0); CHECK(WIFEXITED(st) && !WEXITSTATUS(st), "a child sees the offset (%#x)", st);
  CHECK(put("/proc/self/timens_offsets", "monotonic 5 0\n") < 0, "not once a task has entered (%d)", errno);
  return fails;
}
int main(int argc, char **argv) {
  if (argc > 1 && !strcmp(argv[1], "--caps")) return has_cap(CAP_SYS_ADMIN) == 0 ? 0 : 1;
  if (argc > 1 && !strcmp(argv[1], "--caps-root")) return has_cap(CAP_SYS_ADMIN) == 1 ? 0 : 1;
  if (argc > 1 && !strcmp(argv[1], "--locked")) { child(t_locked); puts(fails ? "SOME FAILED" : "PASS"); return fails != 0; }
  child(t_basics); child(t_scope); child(t_one_call); child(t_clone); child(t_files); child(t_exec); child(t_setns); child(t_time);
  /* These need to start with privilege. */
  if (getuid() == 0) child(t_locked);
  puts(fails ? "SOME FAILED" : "PASS"); return fails != 0;
}
