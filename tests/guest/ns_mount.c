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
#include <sys/syscall.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>
static int fails;
#define CHECK(c, ...) do { int ok_ = (c); printf("%s: ", ok_ ? "ok  " : "FAIL"); printf(__VA_ARGS__); puts(""); fflush(stdout); if (!ok_) fails++; } while (0)
static int exists(const char *p) { struct stat s; return !stat(p, &s); }
static void put(const char *p, const char *t) { int f = open(p, O_CREAT | O_TRUNC | O_WRONLY, 0644); write(f, t, strlen(t)); close(f); }
static char *get(const char *p, char *b) { b[0] = 0; int f = open(p, O_RDONLY); if (f >= 0) { int n = read(f, b, 99); b[n < 0 ? 0 : n] = 0; close(f); } return b; }
static int in_child(int (*f)(void)) { fflush(stdout); pid_t c = fork(); if (!c) _exit(f()); int st; waitpid(c, &st, 0); fails += WIFEXITED(st) ? WEXITSTATUS(st) : 1; return st; }
static int has_mount(const char *point) { char b[8192] = {0}; int f = open("/proc/self/mountinfo", O_RDONLY); if (f < 0) return -1; read(f, b, sizeof b - 1); close(f); char needle[128]; sprintf(needle, " %s ", point); return strstr(b, needle) != 0; }
#define B "/tmp/nst"
static int t_private(void) {
  CHECK(!unshare(CLONE_NEWNS), "unshare mount namespace");
  mount(0, "/", 0, MS_REC | MS_PRIVATE, 0);
  CHECK(!mount("tmpfs", B "/m", "tmpfs", 0, 0), "mount a tmpfs inside");
  put(B "/m/inner", "x"); CHECK(exists(B "/m/inner"), "visible here"); CHECK(has_mount(B "/m") == 1, "mountinfo lists it");
  return fails;
}
static int t_bind(void) {
  unshare(CLONE_NEWNS); mount(0, "/", 0, MS_REC | MS_PRIVATE, 0);
  char b[100]; put(B "/src/f", "data");
  CHECK(!mount(B "/src", B "/dst", 0, MS_BIND, 0), "bind a directory"); CHECK(!strcmp(get(B "/dst/f", b), "data"), "reads through it");
  put(B "/dst/g", "two"); CHECK(!strcmp(get(B "/src/g", b), "two"), "writes show at the source");
  CHECK(!mount(0, B "/dst", 0, MS_REMOUNT | MS_BIND | MS_RDONLY, 0), "remount bind read-only");
  CHECK(open(B "/dst/h", O_CREAT | O_WRONLY, 0644) < 0 && errno == EROFS, "writes refused: EROFS (%d)", errno);
  CHECK(unlink(B "/dst/f") < 0 && errno == EROFS, "unlink refused"); CHECK(mkdir(B "/dst/d", 0755) < 0 && errno == EROFS, "mkdir refused"); CHECK(rename(B "/dst/f", B "/dst/f2") < 0 && errno == EROFS, "rename refused");
  put(B "/src/f3", "ok"); CHECK(!strcmp(get(B "/dst/f3", b), "ok"), "the source stays writable, and the mount shows it");
  CHECK(!umount(B "/dst"), "unmount it"); CHECK(!exists(B "/dst/f"), "gone");
  put(B "/file1", "filedata"); put(B "/file2", "other"); CHECK(!mount(B "/file1", B "/file2", 0, MS_BIND, 0), "bind a file onto a file"); CHECK(!strcmp(get(B "/file2", b), "filedata"), "it reads the source"); umount(B "/file2");
  /* Recursive */
  mount("tmpfs", B "/src/sub", "tmpfs", 0, 0); put(B "/src/sub/deep", "d");
  CHECK(!mount(B "/src", B "/dst", 0, MS_BIND, 0), "non-recursive bind"); CHECK(!exists(B "/dst/sub/deep"), "doesn't bring submounts"); umount(B "/dst");
  CHECK(!mount(B "/src", B "/dst", 0, MS_BIND | MS_REC, 0), "MS_REC bind"); CHECK(exists(B "/dst/sub/deep"), "brings submounts");
  CHECK(umount(B "/dst") < 0 && errno == EBUSY, "plain umount with something below: EBUSY (%d)", errno); CHECK(!umount2(B "/dst", MNT_DETACH), "MNT_DETACH takes it all"); CHECK(!exists(B "/dst/sub/deep"), "really gone");
  /* Move */
  mount("tmpfs", B "/m", "tmpfs", 0, 0); put(B "/m/a", "a"); CHECK(!mount(B "/m", B "/m2", 0, MS_MOVE, 0), "MS_MOVE"); CHECK(exists(B "/m2/a") && !exists(B "/m/a"), "moved"); umount(B "/m2");
  CHECK(mount("x", B "/nope", "nosuchfs", 0, 0) < 0, "unknown filesystem fails");
  return fails;
}
static int t_chroot(void) {
  char b[200];
  mkdir(B "/jail", 0755); mkdir(B "/jail/d", 0755); mkdir(B "/jail/tmp", 0755); put(B "/jail/d/in", "inside"); put(B "/secret", "outside");
  symlink("/d/in", B "/jail/abs"); symlink("../secret", B "/jail/esc"); symlink("../../secret", B "/jail/d/esc2"); symlink("/d", B "/jail/dirlink");
  CHECK(!chdir(B "/jail") && !chroot("."), "chroot"); CHECK(getcwd(b, sizeof b) && !strcmp(b, "/"), "cwd reads as / (%s)", b);
  CHECK(!strcmp(get("/d/in", b), "inside"), "absolute paths are inside"); CHECK(!strcmp(get("abs", b), "inside"), "an absolute symlink resolves inside"); CHECK(!strcmp(get("/dirlink/in", b), "inside"), "through a directory symlink");
  CHECK(!exists("/secret"), "can't see outside"); CHECK(!exists("../../../../secret"), "dot-dot stops at the root"); CHECK(!exists("/../secret"), "from the root too"); CHECK(!exists("esc"), "a symlink can't climb out either"); CHECK(!exists("/d/esc2"), "nor from deeper, with the exact depth"); char ob[100]; CHECK(open("esc", O_RDONLY) < 0, "and open() refuses it too");
  CHECK(!chdir("/d") && !chdir("../../..") && getcwd(b, sizeof b) && !strcmp(b, "/"), "chdir .. stops at the root (%s)", b);
  CHECK(!exists(B), "the outer directory isn't there");
  pid_t c = fork(); if (!c) { _exit(exists("/secret") || !exists("/d/in") ? 1 : 0); } int st; waitpid(c, &st, 0); CHECK(WIFEXITED(st) && !WEXITSTATUS(st), "a child stays inside (%#x)", st);
  CHECK(!chroot("d") && !exists("/d") && exists("/in"), "chroot again, narrower");
  return fails;
}
static int t_chroot_fd(void) { /* A cwd left outside isn't magic, as on Linux. */
  char b[200]; CHECK(!chroot(B "/jail") , "chroot without chdir"); long n = syscall(SYS_getcwd, b, sizeof b); CHECK(n > 0 && !strncmp(b, "(unreachable)", 13), "cwd outside reads (unreachable): %s", b); return fails; }
static int t_pivot(void) {
  char b[200];
  unshare(CLONE_NEWNS); mount(0, "/", 0, MS_REC | MS_PRIVATE, 0);
  CHECK(!mount("tmpfs", B "/new", "tmpfs", 0, 0), "new root is a mount"); mkdir(B "/new/old", 0755); mkdir(B "/new/proc", 0755); put(B "/new/hello", "in new root");
  CHECK(syscall(SYS_pivot_root, B "/src", B "/src/x") < 0 && errno == EINVAL, "new root must be a mount: EINVAL (%d)", errno);
  CHECK(syscall(SYS_pivot_root, B "/new", "/tmp") < 0 && (errno == EINVAL || errno == EBUSY), "put_old must be under it (%d)", errno);
  int fd = open(B "/new/hello", O_RDONLY); chdir(B "/new");
  CHECK(!syscall(SYS_pivot_root, ".", "old"), "pivot_root (%d)", errno); chdir("/");
  CHECK(!strcmp(get("/hello", b), "in new root"), "the new root is /"); CHECK(exists("/old/etc") || exists("/old/tmp"), "old root is at put_old");
  int n = read(fd, b, 5); CHECK(n == 5, "a file opened before still works (%d)", n); char l[64], tgt[200] = {0}; sprintf(l, "/proc/self/fd/%d", fd);
  CHECK(!mount("proc", "/proc", "proc", 0, 0), "mount /proc in the new root"); readlink(l, tgt, 199); CHECK(!strcmp(tgt, "/hello"), "its link reads the new path: %s", tgt);
  CHECK(has_mount("/old") >= 0, "mountinfo reads");
  CHECK(!umount2("/old", MNT_DETACH), "detach the old root"); CHECK(!exists("/old/etc") && !exists("/old/tmp"), "old root is gone"); CHECK(has_mount("/old/tmp") == 0, "mountinfo agrees");
  return fails;
}
static int t_pivot_dotdot(void) {
  unshare(CLONE_NEWNS); mount(0, "/", 0, MS_REC | MS_PRIVATE, 0); mount("tmpfs", B "/new", "tmpfs", 0, 0); put(B "/new/mark", "m"); chdir(B "/new");
  CHECK(!syscall(SYS_pivot_root, ".", "."), "pivot_root(., .) (%d)", errno); CHECK(!umount2(".", MNT_DETACH), "detach what's stacked on top"); chdir("/"); CHECK(exists("/mark"), "the new root is what's left"); CHECK(!exists("/etc") && !exists("/tmp"), "nothing of the old one");
  return fails;
}
static int t_setns_helper(void) {
  int keep = open("/proc/self/ns/mnt", O_RDONLY); CHECK(keep >= 0, "handle on the first namespace"); unshare(CLONE_NEWNS); mount(0, "/", 0, MS_REC | MS_PRIVATE, 0); mount("tmpfs", B "/m", "tmpfs", 0, 0); put(B "/m/only-here", "x"); chdir(B "/m");
  int of = open(B "/m/only-here", O_RDWR), od = open(B "/m", O_RDONLY | O_DIRECTORY); CHECK(of >= 0 && od >= 0, "open a file and a directory there");
  CHECK(!setns(keep, CLONE_NEWNS), "setns back"); CHECK(!exists(B "/m/only-here"), "that mount isn't in this namespace"); char b[200]; CHECK(getcwd(b, sizeof b) && !strcmp(b, "/"), "and the cwd is reset to the root (%s)", b);
  /* What was open stays the file it was opened on, as on Linux. */
  char r[8] = {0}; CHECK(pread(of, r, sizeof r - 1, 0) == 1 && r[0] == 'x', "the open file still reads its contents (%s)", r);
  CHECK(pwrite(of, "yz", 2, 1) == 2 && pread(of, r, 3, 0) == 3 && !memcmp(r, "xyz", 3), "and writes to it");
  struct stat st; CHECK(!fstat(of, &st) && st.st_size == 3, "fstat sees it (%ld)", (long)st.st_size);
  int rel = openat(od, "only-here", O_RDONLY); CHECK(rel >= 0, "openat through the old directory finds it");
  int made = openat(od, "made-later", O_CREAT | O_WRONLY, 0600); CHECK(made >= 0 && !exists(B "/m/made-later"), "a file made through it lands there, not here");
  return fails;
}
static int t_unix(void) {
  /* A socket found through the filesystem: same file, same socket, whatever the mount. */
  unlink(B "/src/sock"); int s = socket(AF_UNIX, SOCK_STREAM, 0); struct sockaddr_un a = {0}; a.sun_family = AF_UNIX; strcpy(a.sun_path, B "/src/sock"); CHECK(!bind(s, (void *)&a, sizeof a) && !listen(s, 4), "listen on a path");
  unshare(CLONE_NEWNS); mount(0, "/", 0, MS_REC | MS_PRIVATE, 0); int c = socket(AF_UNIX, SOCK_STREAM, 0); CHECK(!connect(c, (void *)&a, sizeof a), "connect from another mount namespace, same file");
  mount(B "/src", B "/dst", 0, MS_BIND, 0); struct sockaddr_un d = {0}; d.sun_family = AF_UNIX; strcpy(d.sun_path, B "/dst/sock"); int c2 = socket(AF_UNIX, SOCK_STREAM, 0); CHECK(!connect(c2, (void *)&d, sizeof d), "connect through a bind mount of it");
  mount("tmpfs", B "/src", "tmpfs", 0, 0); int c3 = socket(AF_UNIX, SOCK_STREAM, 0); CHECK(connect(c3, (void *)&a, sizeof a) < 0, "a tmpfs over it hides it (%d)", errno);
  return fails;
}
/* Whether the mountinfo line for `point` has `field` among its optional fields. */
static int mi_field(const char *point, const char *field) {
  static char b[16384]; memset(b, 0, sizeof b); int f = open("/proc/self/mountinfo", O_RDONLY); if (f < 0) return -1; read(f, b, sizeof b - 1); close(f);
  char needle[128]; sprintf(needle, " %s ", point); int found = 0;
  for (char *l = strtok(b, "\n"); l; l = strtok(0, "\n")) { char *at = strstr(l, needle); char *dash = strstr(l, " - "); if (at && dash && at < dash) { found = 0; char *o = strstr(at, field); if (o && o < dash) found = 1; } }
  return found;
}
static int t_propagation(void) {
  unshare(CLONE_NEWNS); mount(0, "/", 0, MS_REC | MS_PRIVATE, 0);
  mkdir(B "/p", 0755); CHECK(!mount("tmpfs", B "/p", "tmpfs", 0, 0), "a tmpfs to share"); for (const char *d = "abcdeu"; *d; d++) { char n[64]; sprintf(n, B "/p/%c", *d); mkdir(n, 0755); }
  CHECK(!mi_field(B "/p", "shared:"), "private to start with");
  CHECK(!mount(0, B "/p", 0, MS_SHARED, 0) && mi_field(B "/p", "shared:") == 1, "--make-shared, and mountinfo says so");
  CHECK(mount(0, B "/p/a", 0, MS_SHARED, 0) < 0 && errno == EINVAL, "not a mount point: EINVAL (%d)", errno);
  CHECK(mount(0, B "/p", 0, MS_SHARED | MS_PRIVATE, 0) < 0 && errno == EINVAL, "two types at once: EINVAL");
  int go[2], back[2]; pipe(go); pipe(back); char x = 1; int st;
  /* A peer in another namespace: mounts go both ways, and so do unmounts. */
  fflush(stdout); pid_t c = fork();
  if (!c) { unshare(CLONE_NEWNS); write(back[1], &x, 1); read(go[0], &x, 1); int seen = exists(B "/p/a/from-parent");
    mount("tmpfs", B "/p/b", "tmpfs", 0, 0); put(B "/p/b/from-child", "x"); write(back[1], &x, 1); read(go[0], &x, 1); int gone = !exists(B "/p/a/from-parent"); _exit(seen && gone ? 0 : 1 + 2 * seen + 4 * gone); }
  read(back[0], &x, 1); mount("tmpfs", B "/p/a", "tmpfs", 0, 0); put(B "/p/a/from-parent", "x"); write(go[1], &x, 1); read(back[0], &x, 1);
  CHECK(exists(B "/p/b/from-child"), "a mount in a peer namespace shows here");
  umount(B "/p/a"); write(go[1], &x, 1); waitpid(c, &st, 0); CHECK(WIFEXITED(st) && !WEXITSTATUS(st), "ours shows there, and so does its unmount (%#x)", st);
  /* A slave gets the master's mounts but keeps its own. */
  fflush(stdout); c = fork();
  if (!c) { unshare(CLONE_NEWNS); mount(0, B "/p", 0, MS_SLAVE, 0); int sl = mi_field(B "/p", "master:") == 1 && !mi_field(B "/p", "shared:"); write(back[1], &x, 1); read(go[0], &x, 1);
    int seen = exists(B "/p/c/from-master"); mount("tmpfs", B "/p/d", "tmpfs", 0, 0); put(B "/p/d/from-slave", "x"); write(back[1], &x, 1); read(go[0], &x, 1); _exit(sl && seen ? 0 : 1 + 2 * sl + 4 * seen); }
  read(back[0], &x, 1); mount("tmpfs", B "/p/c", "tmpfs", 0, 0); put(B "/p/c/from-master", "x"); write(go[1], &x, 1); read(back[0], &x, 1);
  CHECK(!exists(B "/p/d/from-slave"), "a slave's own mounts don't come back"); write(go[1], &x, 1); waitpid(c, &st, 0); CHECK(WIFEXITED(st) && !WEXITSTATUS(st), "a slave gets the master's mounts, and says master: (%#x)", st);
  /* A bind mount of a shared mount is its peer. */
  mkdir(B "/pbind", 0755); CHECK(!mount(B "/p", B "/pbind", 0, MS_BIND, 0) && mi_field(B "/pbind", "shared:") == 1, "a bind of a shared mount is shared");
  mount("tmpfs", B "/p/e", "tmpfs", 0, 0); put(B "/p/e/x", "x"); CHECK(exists(B "/pbind/e/x"), "and gets what's mounted under the original");
  CHECK(!mount(0, B "/pbind", 0, MS_PRIVATE, 0) && !mi_field(B "/pbind", "shared:"), "--make-private"); mount("tmpfs", B "/p/u", "tmpfs", 0, 0); put(B "/p/u/x", "x"); CHECK(!exists(B "/pbind/u/x"), "then nothing more arrives");
  /* Unbindable */
  mkdir(B "/ub", 0755); mkdir(B "/ubind", 0755); mount("tmpfs", B "/ub", "tmpfs", 0, 0); CHECK(!mount(0, B "/ub", 0, MS_UNBINDABLE, 0) && mi_field(B "/ub", "unbindable") == 1, "--make-unbindable");
  CHECK(mount(B "/ub", B "/ubind", 0, MS_BIND, 0) < 0 && errno == EINVAL, "and it can't be bind mounted: EINVAL (%d)", errno);
  /* Into a less privileged namespace, shared becomes slave. */
  fflush(stdout); c = fork(); if (!c) { if (unshare(CLONE_NEWUSER | CLONE_NEWNS)) _exit(9); _exit(mi_field(B "/p", "master:") == 1 && !mi_field(B "/p", "shared:") ? 0 : 1); }
  waitpid(c, &st, 0); CHECK(WIFEXITED(st) && !WEXITSTATUS(st), "a new user namespace's copy of a shared mount is a slave (%#x)", st);
  return fails;
}
int main(void) {
  mkdir(B, 0755); mount("tmpfs", B, "tmpfs", 0, 0); mkdir(B "/m", 0755); mkdir(B "/m2", 0755); mkdir(B "/src", 0755); mkdir(B "/src/sub", 0755); mkdir(B "/src/x", 0755); mkdir(B "/dst", 0755); mkdir(B "/new", 0755); put(B "/marker", "m");
  int st = in_child(t_private); CHECK(!exists(B "/m/inner"), "a mount in a child's namespace isn't visible outside"); CHECK(has_mount(B "/m") == 0, "nor in this mountinfo");
  in_child(t_bind); CHECK(!exists(B "/dst/f"), "binds stayed in their namespace"); CHECK(exists(B "/src/f"), "the file itself persists, it's the same filesystem");
  in_child(t_chroot); CHECK(exists(B "/secret"), "the jail didn't affect this process");
  in_child(t_chroot_fd); in_child(t_pivot); in_child(t_pivot_dotdot); CHECK(exists(B "/marker"), "pivots stayed in their namespaces");
  in_child(t_setns_helper); in_child(t_unix); in_child(t_propagation);
  /* Clone */
  fflush(stdout); pid_t c = syscall(SYS_clone, CLONE_NEWNS | SIGCHLD, 0, 0, 0, 0); if (!c) { mount(0, "/", 0, MS_REC | MS_PRIVATE, 0); mount("tmpfs", B "/m", "tmpfs", 0, 0); put(B "/m/via-clone", "x"); _exit(exists(B "/m/via-clone") ? 0 : 1); } waitpid(c, &st, 0); CHECK(WIFEXITED(st) && !WEXITSTATUS(st) && !exists(B "/m/via-clone"), "clone(CLONE_NEWNS) works the same (%#x)", st);
  puts(fails ? "SOME FAILED" : "PASS"); return fails != 0;
}
