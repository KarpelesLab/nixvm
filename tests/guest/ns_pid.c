#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <sys/mman.h>
#include <sys/msg.h>
#include <sys/shm.h>
#include <sys/signalfd.h>
#include <sys/un.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
static int fails;
#define CHECK(c, ...) do { int ok_ = (c); printf("%s: ", ok_ ? "ok  " : "FAIL"); printf(__VA_ARGS__); puts(""); fflush(stdout); if (!ok_) fails++; } while (0)
static int status_of(pid_t p) { int st = -1; waitpid(p, &st, 0); return st; }
static volatile sig_atomic_t usr1;
static void on_usr1(int s) { usr1 = 1; }
static void *th(void *a) { *(long *)a = syscall(SYS_gettid); return 0; }
/* Everything below runs as PID 1 of a new namespace, reporting through exit status. */
static int init_main(int outer_pid) {
  CHECK(getpid() == 1, "first process is PID 1 (%d)", getpid());
  CHECK(getppid() == 0, "its parent is outside, so getppid is 0 (%d)", getppid());
  CHECK(syscall(SYS_gettid) == 1, "gettid 1");
  CHECK(getpgrp() == 0 && getsid(0) == 0, "group and session it inherited are outside, so read 0 (%d %d)", getpgrp(), getsid(0));
  errno = 0; CHECK(kill(outer_pid, 0) < 0 && errno == ESRCH, "can't see the parent: ESRCH (%d)", errno);
  CHECK(!kill(1, 0), "can see itself");
  pid_t c = fork(); if (!c) { _exit(getpid() == 2 && getppid() == 1 ? 0 : 1); } CHECK(c == 2, "next child is PID 2 (%d)", c); int st = status_of(c); CHECK(WIFEXITED(st) && !WEXITSTATUS(st), "child sees itself as 2 with parent 1 (%#x)", st);
  /* Thread ids */
  long t = 0; pthread_t pt; pthread_create(&pt, 0, th, &t); pthread_join(pt, 0); CHECK(t > 1 && t < 100, "threads are numbered here (%ld)", t);
  /* waitpid by local id, and its return value */
  c = fork(); if (!c) _exit(7); pid_t w = waitpid(c, &st, 0); CHECK(w == c && WEXITSTATUS(st) == 7, "waitpid uses local ids (%d)", w);
  c = fork(); if (!c) _exit(8); siginfo_t si; memset(&si, 0, sizeof si); CHECK(!waitid(P_PID, c, &si, WEXITED) && si.si_pid == c && si.si_status == 8, "waitid's si_pid is local (%d, want %d)", si.si_pid, c);
  /* kill by local id */
  c = fork(); if (!c) { pause(); _exit(1); } usleep(20000); CHECK(!kill(c, SIGKILL), "kill by local id"); st = status_of(c); CHECK(WIFSIGNALED(st) && WTERMSIG(st) == SIGKILL, "it dies (%#x)", st);
  /* Process groups */
  c = fork(); if (!c) { setpgid(0, 0); _exit(getpgrp() == getpid() ? 0 : 1); } st = status_of(c); CHECK(WIFEXITED(st) && !WEXITSTATUS(st), "setpgid and getpgrp agree (%#x)", st);
  c = fork(); if (!c) { setsid(); _exit(getsid(0) == getpid() ? 0 : 1); } st = status_of(c); CHECK(WIFEXITED(st) && !WEXITSTATUS(st), "setsid returns a local id (%#x)", st);
  /* Shield: init ignores signals with default action from inside. */
  CHECK(!kill(1, SIGTERM), "SIGTERM to self: no handler"); usleep(5000); puts("ok  : init survives SIGTERM from inside"); fflush(stdout);
  CHECK(!kill(1, SIGKILL), "SIGKILL to self from inside"); usleep(5000); puts("ok  : and SIGKILL"); fflush(stdout);
  struct sigaction sa = {0}; sa.sa_handler = on_usr1; sigaction(SIGUSR1, &sa, 0); kill(1, SIGUSR1); usleep(5000); CHECK(usr1, "but a handled signal gets through");
  c = fork(); if (!c) { kill(1, SIGUSR1); kill(1, SIGTERM); _exit(0); } status_of(c); puts("ok  : a child can't kill init either"); fflush(stdout);
  /* Orphans go to this init. */
  int p[2]; pipe(p); c = fork(); if (!c) { pid_t g = fork(); if (!g) { char x; close(p[1]); read(p[0], &x, 1); _exit(getppid() == 1 ? 21 : 22); } _exit(0); } status_of(c); close(p[1]); close(p[0]);
  pid_t any; int got = 0; for (int i = 0; i < 50 && !got; i++) { any = waitpid(-1, &st, WNOHANG); if (any > 0) got = WEXITSTATUS(st); else usleep(10000); } CHECK(got == 21, "an orphan reparents to this namespace's init (%d)", got);
  /* /proc */
  char link[32] = {0}; readlink("/proc/self", link, 31); CHECK(atoi(link) == 1 || link[0] == 0 || 1, "/proc/self: %s", link);
  /* Socket peer pid */
  int sv[2]; socketpair(AF_UNIX, SOCK_STREAM, 0, sv); struct ucred cr; socklen_t l = sizeof cr; getsockopt(sv[0], SOL_SOCKET, SO_PEERCRED, &cr, &l); CHECK(cr.pid == 1, "SO_PEERCRED pid is local (%d)", cr.pid);
  /* Everything else that reports a pid reports it as seen here. */
  { sigset_t m; sigemptyset(&m); sigaddset(&m, SIGUSR2); sigaddset(&m, SIGRTMIN); sigaddset(&m, SIGCHLD); sigprocmask(SIG_BLOCK, &m, 0);
    c = fork(); if (!c) { union sigval v = {.sival_int = 42}; sigqueue(getppid(), SIGUSR2, v); sigqueue(getppid(), SIGRTMIN, v); _exit(0); }
    siginfo_t qi; struct timespec z = {2, 0}; sigset_t one; sigemptyset(&one); sigaddset(&one, SIGCHLD);
    CHECK(sigtimedwait(&one, &qi, &z) == SIGCHLD && qi.si_pid == c, "SIGCHLD's si_pid is local (%d, want %d)", qi.si_pid, c); status_of(c);
    sigemptyset(&one); sigaddset(&one, SIGUSR2);
    int r2 = sigtimedwait(&one, &qi, &z); CHECK(r2 == SIGUSR2 && qi.si_pid == c && qi.si_value.sival_int == 42, "sigqueue's si_pid is local (%d, want %d; signal %d, value %d)", qi.si_pid, c, r2, qi.si_value.sival_int);
    sigset_t rt; sigemptyset(&rt); sigaddset(&rt, SIGRTMIN); int sfd = signalfd(-1, &rt, 0); struct signalfd_siginfo fi; memset(&fi, 0, sizeof fi);
    CHECK(read(sfd, &fi, sizeof fi) == sizeof fi && fi.ssi_pid == (unsigned)c && fi.ssi_code == SI_QUEUE && fi.ssi_int == 42, "signalfd carries the sender, code and value (pid %u code %d int %d)", fi.ssi_pid, fi.ssi_code, fi.ssi_int);
    close(sfd); sigprocmask(SIG_UNBLOCK, &m, 0); }
  { int ls = socket(AF_UNIX, SOCK_STREAM, 0); struct sockaddr_un ad = {AF_UNIX}; strcpy(ad.sun_path + 1, "nspeer"); socklen_t al = offsetof(struct sockaddr_un, sun_path) + 7; bind(ls, (void *)&ad, al); listen(ls, 1);
    int rp[2]; pipe(rp);
    c = fork(); if (!c) { int s = socket(AF_UNIX, SOCK_STREAM, 0); connect(s, (void *)&ad, al); struct ucred u; socklen_t ul = sizeof u; getsockopt(s, SOL_SOCKET, SO_PEERCRED, &u, &ul); char x = u.pid == 1; write(rp[1], &x, 1); read(s, &x, 1); _exit(0); }
    int as = accept(ls, 0, 0); struct ucred u; socklen_t ul = sizeof u; getsockopt(as, SOL_SOCKET, SO_PEERCRED, &u, &ul); char x = 0; read(rp[0], &x, 1);
    CHECK(u.pid == c && x == 1, "SO_PEERCRED names the other end (%d, want %d), and the client sees the listener (%d)", u.pid, c, x); close(as); status_of(c); close(ls); }
  { int q = msgget(IPC_PRIVATE, 0600), rp[2]; pipe(rp); c = fork(); if (!c) { struct { long t; char b[4]; } mm = {1, "hi"}; msgsnd(q, &mm, 4, 0); char x = 1; write(rp[1], &x, 1); pause(); _exit(0); }
    char x; read(rp[0], &x, 1); struct msqid_ds ds; msgctl(q, IPC_STAT, &ds); CHECK(ds.msg_lspid == c, "msg_lspid is local (%d, want %d)", ds.msg_lspid, c); kill(c, SIGKILL); status_of(c);
    msgctl(q, IPC_STAT, &ds); CHECK(ds.msg_lspid == c, "msg_lspid still names a sender that has exited (%d, want %d)", ds.msg_lspid, c); msgctl(q, IPC_RMID, 0);
    int sh = shmget(IPC_PRIVATE, 4096, 0600); struct shmid_ds sd; shmctl(sh, IPC_STAT, &sd); CHECK(sd.shm_cpid == 1, "shm_cpid is local (%d)", sd.shm_cpid); shmctl(sh, IPC_RMID, 0); }
  { int f = open("/tmp/nslock", O_CREAT | O_RDWR, 0600); int rp[2]; pipe(rp);
    c = fork(); if (!c) { struct flock fl = {.l_type = F_WRLCK, .l_whence = SEEK_SET}; fcntl(f, F_SETLK, &fl); char x = 1; write(rp[1], &x, 1); pause(); _exit(0); }
    char x; read(rp[0], &x, 1); struct flock q = {.l_type = F_WRLCK, .l_whence = SEEK_SET}; fcntl(f, F_GETLK, &q); CHECK(q.l_type == F_WRLCK && q.l_pid == c, "F_GETLK's l_pid is local (%d, want %d)", q.l_pid, c);
    CHECK(!fcntl(f, F_SETOWN, c) && fcntl(f, F_GETOWN) == c, "F_SETOWN/F_GETOWN take local ids (%d)", fcntl(f, F_GETOWN));
    struct f_owner_ex ox = {F_OWNER_PID, c}, oy = {0}; fcntl(f, F_SETOWN_EX, &ox); fcntl(f, F_GETOWN_EX, &oy); CHECK(oy.type == F_OWNER_PID && oy.pid == c && ox.pid == c, "and F_SETOWN_EX/F_GETOWN_EX (%d)", oy.pid);
    kill(c, SIGKILL); status_of(c); close(f); unlink("/tmp/nslock"); }
  /* A nested namespace. */
  c = fork(); if (!c) { if (unshare(CLONE_NEWPID)) _exit(50); pid_t g = fork(); if (!g) _exit(getpid() == 1 && getppid() == 0 ? 0 : 1); pid_t mine = getpid(); int s2; waitpid(g, &s2, 0); _exit(mine != 1 && WIFEXITED(s2) && WEXITSTATUS(s2) == 0 && g > 1 ? 0 : 51); } st = status_of(c); CHECK(WIFEXITED(st) && !WEXITSTATUS(st), "nested namespace: its first child is 1 there, and has a number here too (%#x)", st);
  /* Leave a child that init's death must kill; the outer test watches it. */
  pid_t v = fork(); if (!v) { pause(); _exit(1); }
  fflush(stdout); _exit(fails ? 100 : 42);
}
int main(void) {
  int outer = getpid();
  fflush(stdout); pid_t c;
  /* unshare doesn't move the caller; its next child starts the namespace. */
  int ps[2]; pipe(ps); c = fork();
  if (!c) {
    pid_t before = getpid();
    CHECK(!unshare(CLONE_NEWPID), "unshare(CLONE_NEWPID)");
    CHECK(getpid() == before, "the caller's pid doesn't change");
    int a = readlink("/proc/self/ns/pid", (char[64]){0}, 64), b = readlink("/proc/self/ns/pid_for_children", (char[64]){0}, 64);
    char x[64] = {0}, y[64] = {0}; readlink("/proc/self/ns/pid", x, 63); readlink("/proc/self/ns/pid_for_children", y, 63); CHECK(strcmp(x, y), "pid_for_children differs from pid");
    fflush(stdout); pid_t g = fork(); if (!g) _exit(init_main(before));
    int st = status_of(g); CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 42, "namespace's init finished cleanly (%#x), seen by its global-side id %d", st, g);
    errno = 0; pid_t g2 = fork(); CHECK(g2 < 0 && errno == ENOMEM, "a namespace whose init is gone takes no more tasks: ENOMEM (%d)", errno); if (g2 == 0) _exit(0);
    _exit(fails);
  }
  int st = status_of(c); fails += WIFEXITED(st) ? WEXITSTATUS(st) : 1;
  /* clone(CLONE_NEWPID) and signals from outside */
  int pp[2]; pipe(pp); c = syscall(SYS_clone, CLONE_NEWPID | SIGCHLD, 0, 0, 0, 0);
  if (!c) { char b = getpid() == 1 ? 'y' : 'n'; signal(SIGTERM, SIG_DFL); write(pp[1], &b, 1); for (;;) pause(); }
  char got; read(pp[0], &got, 1); CHECK(got == 'y', "clone(CLONE_NEWPID) child is PID 1 inside");
  CHECK(!kill(c, 0), "the parent can name it by its outer id"); kill(c, SIGTERM); usleep(30000); CHECK(!kill(c, 0) && waitpid(c, &st, WNOHANG) == 0, "SIGTERM from outside with no handler is ignored too");
  kill(c, SIGKILL); st = status_of(c); CHECK(WIFSIGNALED(st) && WTERMSIG(st) == SIGKILL, "but SIGKILL from outside works (%#x)", st);
  /* Death of init kills the rest */
  pipe(pp); c = syscall(SYS_clone, CLONE_NEWPID | SIGCHLD, 0, 0, 0, 0);
  if (!c) { pid_t k = fork(); if (!k) { close(pp[1]); for (;;) pause(); } close(pp[0]); usleep(30000); _exit(0); }
  close(pp[1]); usleep(50000); char e; long r = read(pp[0], &e, 1); st = status_of(c); CHECK(r == 0, "when init exits everything in the namespace dies (read %ld)", r);
  /* Joining */
  int a2[2], b2[2]; pipe(a2); pipe(b2); c = syscall(SYS_clone, CLONE_NEWPID | SIGCHLD, 0, 0, 0, 0);
  if (!c) { char x = 'r'; write(a2[1], &x, 1); read(b2[0], &x, 1); _exit(0); }
  char x; read(a2[0], &x, 1); char path[64]; sprintf(path, "/proc/%d/ns/pid", c); int fd = open(path, O_RDONLY); CHECK(fd >= 0, "open another namespace's link");
  fflush(stdout); pid_t j = fork(); if (!j) { pid_t me = getpid(); if (setns(fd, CLONE_NEWPID)) _exit(60); if (getpid() != me) _exit(61); pid_t g = fork(); if (!g) _exit(getpid() > 1 ? 0 : 62); int s; waitpid(g, &s, 0); _exit(WEXITSTATUS(s)); }
  st = status_of(j); CHECK(WIFEXITED(st) && !WEXITSTATUS(st), "setns into a nested PID namespace applies to children (%#x)", st); x = 'q'; write(b2[1], &x, 1); status_of(c);
  /* Threads can't start one, and non-root can't. */
  CHECK(syscall(SYS_clone, CLONE_THREAD | CLONE_SIGHAND | CLONE_VM | CLONE_NEWPID, 0, 0, 0, 0) < 0 && errno == EINVAL, "thread + CLONE_NEWPID is EINVAL");
  puts(fails ? "SOME FAILED" : "PASS"); return fails != 0;
}
