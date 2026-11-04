/* Run with a syscall policy in force (see tests/guest_conformance.rs): tries what
   the policy denies, from every kind of task, and prints what came of each. Not a
   pass/fail program: the Rust test reads the lines. */
#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <sys/mount.h>
#include <sys/prctl.h>
#include <sys/ptrace.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
static void show(const char *what, long r) { printf("%s=%ld errno=%d\n", what, r, r < 0 ? errno : 0); fflush(stdout); }
static void *th(void *a) { errno = 0; show("thread-mount", mount("tmpfs", "/mnt", "tmpfs", 0, 0)); return a; }
int main(int argc, char **argv) {
  if (argc > 1 && !strcmp(argv[1], "exec")) { show("exec-mount", mount("tmpfs", "/mnt", "tmpfs", 0, 0)); return 0; }
  show("getpid", getpid() > 0);
  show("mount", mount("tmpfs", "/mnt", "tmpfs", 0, 0));
  show("unshare", unshare(CLONE_NEWNS));
  pthread_t t; pthread_create(&t, 0, th, 0); pthread_join(t, 0);
  fflush(stdout); pid_t c = fork(); if (!c) { show("child-mount", mount("tmpfs", "/mnt", "tmpfs", 0, 0)); _exit(0); } waitpid(c, 0, 0);
  /* A guest seccomp filter that allows everything changes nothing. */
  struct sock_filter allow[] = {BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW)}; struct sock_fprog prog = {1, allow};
  prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0); show("seccomp", syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER, 0, &prog)); show("after-seccomp-mount", mount("tmpfs", "/mnt", "tmpfs", 0, 0));
  fflush(stdout); c = fork(); if (!c) { execl("/policy", "policy", "exec", (char *)0); _exit(99); } int st; waitpid(c, &st, 0);
  /* Killed here, by the policy. */
  fflush(stdout); c = fork(); if (!c) { ptrace(PTRACE_TRACEME, 0, 0, 0); _exit(0); } waitpid(c, &st, 0); show("ptrace-child-killed", WIFSIGNALED(st) && WTERMSIG(st) == SIGKILL);
  puts("done"); return 0;
}
