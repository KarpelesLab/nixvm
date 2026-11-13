/* Capabilities as sandboxes use them: they're what privilege is, not uid 0, and
 * they change where Linux changes them (set*uid, execve, capset, the bounding and
 * ambient sets, securebits, entering a user namespace). Then setns, by namespace
 * file and by pidfd, which checks each namespace against them.
 *
 * Runs as anyone: the parts that start as root (dropping to another uid) run only
 * when it is root. */
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
#include <sys/xattr.h>
#include <unistd.h>
#include <linux/capability.h>
#include <linux/securebits.h>
static int fails;
#define CHECK(c, ...) do { int ok_ = (c); printf("%s: ", ok_ ? "ok  " : "FAIL"); printf(__VA_ARGS__); puts(""); fflush(stdout); if (!ok_) fails++; } while (0)
static int child(int (*f)(void)) { fflush(stdout); pid_t c = fork(); if (!c) _exit(f()); int st; waitpid(c, &st, 0); fails += WIFEXITED(st) ? WEXITSTATUS(st) : 1; return st; }
static int put(const char *path, const char *text) { int f = open(path, O_WRONLY); if (f < 0) return -1; int r = write(f, text, strlen(text)); int e = errno; close(f); errno = e; return r < 0 ? -1 : 0; }
static char *get(const char *p, char *b, int n) { b[0] = 0; int f = open(p, O_RDONLY); if (f >= 0) { int r = read(f, b, n - 1); b[r < 0 ? 0 : r] = 0; close(f); } return b; }
static struct __user_cap_data_struct caps[2];
static int getcaps(void) { struct __user_cap_header_struct h = {_LINUX_CAPABILITY_VERSION_3, 0}; return syscall(SYS_capget, &h, caps); }
static int setcaps(unsigned long long eff, unsigned long long perm, unsigned long long inh) { struct __user_cap_header_struct h = {_LINUX_CAPABILITY_VERSION_3, 0}; struct __user_cap_data_struct d[2] = {{eff, perm, inh}, {eff >> 32, perm >> 32, inh >> 32}}; return syscall(SYS_capset, &h, d); }
static unsigned long long eff(void) { getcaps(); return caps[0].effective | (unsigned long long)caps[1].effective << 32; }
static unsigned long long perm(void) { getcaps(); return caps[0].permitted | (unsigned long long)caps[1].permitted << 32; }
static int has(int cap) { return (eff() >> cap) & 1; }
#define BIT(c) (1ULL << (c))
static unsigned long long status_caps(const char *name) { char b[4096], *p; get("/proc/self/status", b, sizeof b); if (!(p = strstr(b, name))) return ~0ULL; return strtoull(p + strlen(name), 0, 16); }
static void map_root(int uid, int gid) { char l[64]; put("/proc/self/setgroups", "deny"); sprintf(l, "0 %d 1\n", uid); put("/proc/self/uid_map", l); sprintf(l, "0 %d 1\n", gid); put("/proc/self/gid_map", l); }
static const char *self_exe;
/* Run this program again as `--has <cap>`: its exit status says whether the new
 * program holds the capability. */
static int exec_has(int cap) { char n[16]; sprintf(n, "%d", cap); fflush(stdout); pid_t c = fork(); if (!c) { execl(self_exe, "userns_caps", "--has", n, (char *)0); _exit(99); } int st; waitpid(c, &st, 0); return WIFEXITED(st) ? WEXITSTATUS(st) : 98; }

static int t_root_drops(void) {
  CHECK(has(CAP_SETUID) && has(CAP_SYS_ADMIN), "root starts with every capability");
  CHECK(status_caps("CapEff:\t") == eff(), "status shows them (%llx)", status_caps("CapEff:\t"));
  CHECK(!setresuid(0, 1000, 0), "euid away from root"); CHECK(eff() == 0 && perm() != 0, "effective set cleared, permitted kept (%llx %llx)", eff(), perm());
  CHECK(!setresuid(0, 0, 0), "back to euid 0"); CHECK(has(CAP_SYS_ADMIN), "effective comes back from permitted");
  CHECK(!setresuid(1000, 1000, 1000), "every uid away from root"); CHECK(eff() == 0 && perm() == 0, "every capability gone (%llx %llx)", eff(), perm());
  CHECK(setuid(0) < 0 && errno == EPERM, "and can't come back (%d)", errno);
  return fails;
}
static int t_keepcaps(void) {
  CHECK(!prctl(PR_SET_KEEPCAPS, 1) && prctl(PR_GET_KEEPCAPS) == 1, "PR_SET_KEEPCAPS");
  CHECK(!setresuid(1000, 1000, 1000), "drop to 1000"); CHECK(eff() == 0 && perm() != 0, "kept permitted, not effective (%llx %llx)", eff(), perm());
  CHECK(setresgid(1000, 1000, 1000) < 0 && errno == EPERM, "without CAP_SETGID effective, setresgid fails (%d)", errno);
  CHECK(!setcaps(BIT(CAP_SETGID) | BIT(CAP_SYS_ADMIN), perm(), BIT(CAP_SYS_ADMIN)), "capset raises it from permitted");
  CHECK(!setresgid(1000, 1000, 1000), "then it works (%d)", errno);
  unsigned long long pm = perm(); CHECK(!setcaps(eff(), pm & ~BIT(CAP_CHOWN), BIT(CAP_SYS_ADMIN)), "drop CAP_CHOWN from permitted");
  CHECK(setcaps(eff(), pm, BIT(CAP_SYS_ADMIN)) < 0 && errno == EPERM, "permitted can't grow back (%d)", errno);
  CHECK(!prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_RAISE, CAP_SYS_ADMIN, 0, 0), "raise CAP_SYS_ADMIN into the ambient set (%d)", errno);
  CHECK(prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_RAISE, CAP_CHOWN, 0, 0) < 0 && errno == EPERM, "not one that isn't inheritable (%d)", errno);
  CHECK(exec_has(CAP_SYS_ADMIN) == 1, "a program run as 1000 keeps an ambient capability");
  CHECK(exec_has(CAP_SETGID) == 0, "and nothing else");
  CHECK(prctl(PR_GET_KEEPCAPS) == 1, "keepcaps holds until exec");
  return fails;
}
static int t_bounding(void) {
  CHECK(prctl(PR_CAPBSET_READ, CAP_SYS_ADMIN) == 1, "CAP_SYS_ADMIN bounded in");
  CHECK(!prctl(PR_CAPBSET_DROP, CAP_SYS_ADMIN), "drop it from the bounding set"); CHECK(prctl(PR_CAPBSET_READ, CAP_SYS_ADMIN) == 0, "reads dropped");
  CHECK(has(CAP_SYS_ADMIN), "it's still effective now");
  CHECK(exec_has(CAP_SYS_ADMIN) == 0, "but a program run as root doesn't get it");
  CHECK(exec_has(CAP_CHOWN) == 1, "and gets the rest");
  CHECK(!setcaps(eff() & ~BIT(CAP_SETPCAP), perm() & ~BIT(CAP_SETPCAP), 0), "drop CAP_SETPCAP");
  CHECK(prctl(PR_CAPBSET_DROP, CAP_CHOWN) < 0 && errno == EPERM, "then the bounding set can't shrink (%d)", errno);
  return fails;
}
static int t_securebits(void) {
  CHECK(!prctl(PR_SET_SECUREBITS, SECBIT_KEEP_CAPS | SECBIT_KEEP_CAPS_LOCKED), "lock keepcaps on (%d)", errno);
  CHECK(prctl(PR_GET_KEEPCAPS) == 1, "keepcaps reads on");
  CHECK(prctl(PR_SET_KEEPCAPS, 0) < 0 && errno == EPERM, "a locked bit can't change (%d)", errno);
  CHECK(prctl(PR_SET_SECUREBITS, 0) < 0 && errno == EPERM, "nor its lock (%d)", errno);
  return fails;
}
static int t_root_without_caps(void) {
  char host[64]; gethostname(host, sizeof host);
  CHECK(!setcaps(0, 0, 0), "uid 0 drops every capability");
  CHECK(getuid() == 0 && geteuid() == 0, "still uid 0");
  CHECK(sethostname(host, strlen(host)) < 0 && errno == EPERM, "no sethostname (%d)", errno);
  CHECK(unshare(CLONE_NEWUTS) < 0 && errno == EPERM, "no new namespaces (%d)", errno);
  CHECK(chroot("/") < 0 && errno == EPERM, "no chroot (%d)", errno);
  CHECK(setuid(5) < 0 && errno == EPERM, "no setuid (%d)", errno);
  gid_t g = 0; CHECK(setgroups(1, &g) < 0 && errno == EPERM, "no setgroups (%d)", errno);
  CHECK(exec_has(CAP_SYS_ADMIN) == 1, "exec as root gets the bounding set back");
  return fails;
}

/* As an ordinary user. */
/* Changing ids makes a process undumpable, which hides its /proc files from it. */
static void as_user(void) { if (getuid() == 0) { setgroups(0, 0); setresgid(1000, 1000, 1000); setresuid(1000, 1000, 1000); prctl(PR_SET_DUMPABLE, 1); } }
static int t_userns(void) {
  char b[4096]; struct stat s; as_user(); int uid = getuid(), gid = getgid();
  CHECK(!unshare(CLONE_NEWUSER), "user namespace"); CHECK(status_caps("CapEff:\t") == eff() && has(CAP_SYS_ADMIN), "every capability, which status shows");
  map_root(uid, gid);
  CHECK(strstr(get("/proc/self/status", b, sizeof b), "Uid:\t0\t0\t0\t0\n") != 0, "status shows the mapped ids");
  CHECK(!stat("/proc/self/status", &s) && s.st_uid == 0 && s.st_gid == 0, "its /proc entries are its own, through the map (%d %d)", s.st_uid, s.st_gid);
  gid_t gs[64]; int n = getgroups(64, gs), bad = 0; for (int i = 0; i < n; i++) bad += gs[i] != 0 && gs[i] != 65534; CHECK(n >= 0 && !bad, "supplementary groups show through the map (%d)", n);
  CHECK(!setcaps(eff() & ~BIT(CAP_SYS_ADMIN), perm() & ~BIT(CAP_SYS_ADMIN), 0), "drop CAP_SYS_ADMIN");
  CHECK(unshare(CLONE_NEWUTS) < 0 && errno == EPERM, "root here can't make namespaces without it (%d)", errno);
  CHECK(unshare(CLONE_NEWUSER) == 0, "but can make a user namespace");
  CHECK(has(CAP_SYS_ADMIN), "which gives every capability back, there");
  return fails;
}
static int t_map_needs_admin(void) {
  as_user(); int uid = getuid(); char l[64];
  CHECK(!unshare(CLONE_NEWUSER), "user namespace"); CHECK(!setcaps(0, 0, 0), "drop every capability");
  sprintf(l, "0 %d 1\n", uid); CHECK(put("/proc/self/uid_map", l) < 0 && errno == EPERM, "can't write its own map without CAP_SYS_ADMIN there (%d)", errno);
  return fails;
}
static int t_parent_maps(void) {
  as_user(); int uid = getuid(); int p[2], q[2]; pipe(p); pipe(q); fflush(stdout); pid_t c = fork();
  if (!c) { char x; unshare(CLONE_NEWUSER); write(p[1], "x", 1); read(q[0], &x, 1); _exit(getuid() == 0 ? 0 : 1); }
  char x, path[64], l[64]; read(p[0], &x, 1);
  sprintf(path, "/proc/%d/uid_map", c); sprintf(l, "0 %d 1\n", uid + 1);
  CHECK(put(path, l) < 0 && errno == EPERM, "the parent can't map someone else's id into it (%d)", errno);
  sprintf(l, "0 %d 1\n", uid); CHECK(!put(path, l), "but can map its own, from outside (%d)", errno);
  write(q[1], "x", 1); int st; waitpid(c, &st, 0); CHECK(WIFEXITED(st) && !WEXITSTATUS(st), "the child is uid 0 there (%#x)", st);
  return fails;
}
/* A copy of this program at `dst`, to give file capabilities. */
static int copy_self(const char *dst) {
  char buf[65536]; int in = open(self_exe, O_RDONLY), out = open(dst, O_CREAT | O_TRUNC | O_WRONLY, 0755); ssize_t n;
  if (in < 0 || out < 0) return -1;
  while ((n = read(in, buf, sizeof buf)) > 0) if (write(out, buf, n) != n) return -1;
  close(in); close(out); chmod(dst, 0755); return 0;
}
/* cap_net_raw=ep, as `setcap` writes it (struct vfs_cap_data, revision 2). */
static const unsigned int net_raw_ep[5] = {0x02000001, 1u << CAP_NET_RAW, 0, 0, 0};
static char caps_file[256];
static int exec_file_has(const char *file, int cap) { char n[16]; sprintf(n, "%d", cap); fflush(stdout); pid_t c = fork(); if (!c) { execl(file, "userns_caps", "--has", n, (char *)0); _exit(99); } int st; waitpid(c, &st, 0); return WIFEXITED(st) ? WEXITSTATUS(st) : 98; }
static int t_file_caps(void) {
  sprintf(caps_file, "/tmp/userns_caps.fc.%d", getpid()); CHECK(!copy_self(caps_file), "a copy of the program (%d)", errno);
  CHECK(!setxattr(caps_file, "security.capability", net_raw_ep, sizeof net_raw_ep, 0), "give it cap_net_raw=ep (%d)", errno);
  fflush(stdout); pid_t c = fork();
  if (!c) {
    setgroups(0, 0); setresgid(1000, 1000, 1000); CHECK(!setresuid(1000, 1000, 1000) && eff() == 0, "drop to 1000 with nothing");
    CHECK(exec_file_has(caps_file, CAP_NET_RAW) == 1, "run as 1000, it has the file's capability");
    CHECK(exec_file_has(caps_file, CAP_SYS_ADMIN) == 0, "and only that");
    CHECK(exec_has(CAP_NET_RAW) == 0, "a program without them gets nothing");
    CHECK(!prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) && exec_file_has(caps_file, CAP_NET_RAW) == 0, "nor does it under no_new_privs");
    _exit(fails);
  }
  int st; waitpid(c, &st, 0); fails += WIFEXITED(st) ? WEXITSTATUS(st) : 1;
  CHECK(!chown(caps_file, 0, 0), "chown it"); CHECK(getxattr(caps_file, "security.capability", 0, 0) < 0 && errno == ENODATA, "which takes its capabilities away (%d)", errno);
  unlink(caps_file); return fails;
}
static int t_ns_file_caps(void) {
  as_user(); int uid = getuid(), gid = getgid(); sprintf(caps_file, "/tmp/userns_caps.nsfc.%d", getpid()); CHECK(!copy_self(caps_file), "a copy of the program (%d)", errno);
  CHECK(setxattr(caps_file, "security.capability", net_raw_ep, sizeof net_raw_ep, 0) < 0 && errno == EPERM, "an ordinary user can't give it capabilities (%d)", errno);
  fflush(stdout); pid_t c = fork();
  if (!c) {
    unshare(CLONE_NEWUSER); map_root(uid, gid);
    struct stat s; CHECK(!stat(caps_file, &s) && s.st_uid == 0, "root in its own user namespace owns it");
    CHECK(!setxattr(caps_file, "security.capability", net_raw_ep, sizeof net_raw_ep, 0), "and can give it capabilities (%d)", errno);
    _exit(fails);
  }
  int st; waitpid(c, &st, 0); fails += WIFEXITED(st) ? WEXITSTATUS(st) : 1;
  unsigned int v[6] = {0}; int n = getxattr(caps_file, "security.capability", v, sizeof v);
  CHECK(n == 24 && (v[0] >> 24) == 3 && v[5] == (unsigned)uid, "outside, they're kept as the namespace's: revision 3, its root (%d %#x %u)", n, v[0], v[5]);
  CHECK(exec_file_has(caps_file, CAP_NET_RAW) == 0, "and don't count out here");
  fflush(stdout); c = fork();
  if (!c) { unshare(CLONE_NEWUSER); map_root(uid, gid); CHECK(exec_file_has(caps_file, CAP_NET_RAW) == 1, "but do in a namespace whose root that is"); _exit(fails); }
  waitpid(c, &st, 0); fails += WIFEXITED(st) ? WEXITSTATUS(st) : 1;
  unlink(caps_file); return fails;
}
static int t_new_files(void) {
  as_user(); int uid = getuid(), gid = getgid(); char f[256]; struct stat s; sprintf(f, "/tmp/userns_caps.nf.%d", getpid());
  close(open(f, O_CREAT | O_WRONLY, 0644)); CHECK(!stat(f, &s) && s.st_uid == (uid_t)uid && s.st_gid == (gid_t)gid, "a new file is its creator's (%d %d)", s.st_uid, s.st_gid); unlink(f);
  CHECK(!unshare(CLONE_NEWUSER), "user namespace"); map_root(uid, gid);
  close(open(f, O_CREAT | O_WRONLY, 0644)); CHECK(!stat(f, &s) && s.st_uid == 0 && s.st_gid == 0, "one made as root there is root's there (%d %d)", s.st_uid, s.st_gid); unlink(f);
  CHECK(!mkdir(f, 0755) && !stat(f, &s) && s.st_uid == 0, "so is a directory (%d)", s.st_uid); rmdir(f);
  return fails;
}
static int t_setns(void) {
  as_user(); int uid = getuid(), gid = getgid(); int p[2], q[2]; pipe(p); pipe(q);
  int my_uts = open("/proc/self/ns/uts", O_RDONLY); struct utsname u; uname(&u); char host[65]; strcpy(host, u.nodename);
  fflush(stdout); pid_t c = fork();
  if (!c) { char x; unshare(CLONE_NEWUSER | CLONE_NEWUTS); map_root(uid, gid); sethostname("kid", 3); write(p[1], "x", 1); read(q[0], &x, 1); _exit(0); }
  char x; read(p[0], &x, 1);
  int pfd = syscall(SYS_pidfd_open, c, 0); CHECK(pfd >= 0, "pidfd_open (%d)", errno);
  CHECK(setns(pfd, 0) < 0 && errno == EINVAL, "a pidfd needs namespace flags (%d)", errno);
  if (getuid() != 0) CHECK(setns(pfd, CLONE_NEWUTS) < 0 && errno == EPERM, "the child's UTS namespace alone needs privilege where the caller is (%d)", errno);
  fflush(stdout); pid_t g = fork();
  if (!g) {
    CHECK(!setns(pfd, CLONE_NEWUSER | CLONE_NEWUTS), "join its user and UTS namespaces through the pidfd (%d)", errno);
    uname(&u); CHECK(!strcmp(u.nodename, "kid"), "its hostname (%s)", u.nodename);
    CHECK(getuid() == 0 && has(CAP_SYS_ADMIN), "root there, with capabilities (%d)", getuid());
    CHECK(setns(my_uts, CLONE_NEWUTS) < 0 && errno == EPERM, "they don't reach the UTS namespace it came from (%d)", errno);
    CHECK(setns(pfd, CLONE_NEWUSER) < 0 && errno == EINVAL, "rejoining the user namespace it's in is EINVAL (%d)", errno);
    _exit(fails);
  }
  int st; waitpid(g, &st, 0); fails += WIFEXITED(st) ? WEXITSTATUS(st) : 1;
  uname(&u); CHECK(!strcmp(u.nodename, host), "the caller's own hostname is untouched");
  /* All or nothing: the user namespace is fine, the UTS one isn't. */
  fflush(stdout); g = fork();
  if (!g) {
    unshare(CLONE_NEWUSER); map_root(uid, gid); int mine = open("/proc/self/ns/user", O_RDONLY);
    CHECK(setns(pfd, CLONE_NEWUSER | CLONE_NEWUTS) < 0 && errno == EPERM, "a sibling namespace's root can't join through the pidfd (%d)", errno);
    int now = open("/proc/self/ns/user", O_RDONLY); struct stat s1, s2; fstat(mine, &s1); fstat(now, &s2);
    CHECK(s1.st_ino == s2.st_ino, "and nothing was joined");
    _exit(fails);
  }
  waitpid(g, &st, 0); fails += WIFEXITED(st) ? WEXITSTATUS(st) : 1;
  write(q[1], "x", 1); waitpid(c, 0, 0);
  return fails;
}

int main(int argc, char **argv) {
  if (argc > 2 && !strcmp(argv[1], "--has")) return has(atoi(argv[2]));
  self_exe = "/proc/self/exe";
  /* The conformance runner puts it at the guest's root; elsewhere, where it is. */
  static char exe[4096]; ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1); if (n > 0) { exe[n] = 0; self_exe = exe; }
  /* Somewhere every uid can run it from, even root without CAP_DAC_OVERRIDE
   * (a CI runner's work directory is private to its user). */
  static char copy[64]; int copied = 0;
  if (!access("/userns_caps", X_OK)) self_exe = "/userns_caps";
  else { sprintf(copy, "/tmp/userns_caps.self.%d", getpid()); if (!copy_self(copy)) { self_exe = copy; copied = 1; } }
  if (getuid() == 0) { child(t_root_drops); child(t_keepcaps); child(t_bounding); child(t_securebits); child(t_root_without_caps); child(t_file_caps); }
  child(t_ns_file_caps); child(t_new_files); child(t_userns); child(t_map_needs_admin); child(t_parent_maps); child(t_setns);
  if (copied) unlink(copy);
  puts(fails ? "SOME FAILED" : "PASS"); return fails != 0;
}
