/* ptrace(2) between guest processes, as strace and gdb use it: exec traps,
 * syscall-entry/exit stops (reading and rewriting the call), memory peek and
 * poke, signal interception, fork auto-attach, SEIZE/INTERRUPT, EXITKILL,
 * GET_SYSCALL_INFO, and seccomp's RET_TRACE. Each case forks its own tracee. */
#define _GNU_SOURCE
#include <elf.h>
#include <errno.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <signal.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/ptrace.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <unistd.h>

/* PTRACE_GET_SYSCALL_INFO's result, as the kernel lays it out (not every
 * libc declares it). */
struct syscall_info {
  unsigned char op;
  unsigned char pad[3];
  unsigned int arch;
  unsigned long long instruction_pointer, stack_pointer;
  union {
    struct { unsigned long long nr, args[6]; } entry;
    struct { long long rval; unsigned char is_error; } exit;
    struct { unsigned long long nr, args[6]; unsigned int ret_data; } seccomp;
  };
};
#ifndef PTRACE_GET_SYSCALL_INFO
#define PTRACE_GET_SYSCALL_INFO 0x420e
#endif
#define INFO_ENTRY 1
#define INFO_EXIT 2

#define CHECK(c) do { if (!(c)) { printf("FAIL: %s (line %d, errno %d)\n", #c, __LINE__, errno); exit(1); } } while (0)

/* The syscall number, first argument and return value of a stopped tracee,
 * in the architecture's own register layout. */
#if defined(__x86_64__)
typedef struct user_regs_struct regs_t;
static void get_regs(pid_t p, regs_t *r) { CHECK(ptrace(PTRACE_GETREGS, p, 0, r) == 0); }
static void set_regs(pid_t p, regs_t *r) { CHECK(ptrace(PTRACE_SETREGS, p, 0, r) == 0); }
static long nr_of(pid_t p) { regs_t r; get_regs(p, &r); return (long)r.orig_rax; }
static void set_nr(pid_t p, long nr) { regs_t r; get_regs(p, &r); r.orig_rax = nr; set_regs(p, &r); }
static long arg0_of(pid_t p) { regs_t r; get_regs(p, &r); return (long)r.rdi; }
static void set_arg0(pid_t p, long v) { regs_t r; get_regs(p, &r); r.rdi = v; set_regs(p, &r); }
static long ret_of(pid_t p) { regs_t r; get_regs(p, &r); return (long)r.rax; }
static void set_ret(pid_t p, long v) { regs_t r; get_regs(p, &r); r.rax = v; set_regs(p, &r); }
#elif defined(__aarch64__)
#include <asm/ptrace.h>
typedef struct user_pt_regs regs_t;
static void get_regs(pid_t p, regs_t *r) {
  struct iovec io = {r, sizeof *r};
  CHECK(ptrace(PTRACE_GETREGSET, p, (void *)NT_PRSTATUS, &io) == 0);
}
static void set_regs(pid_t p, regs_t *r) {
  struct iovec io = {r, sizeof *r};
  CHECK(ptrace(PTRACE_SETREGSET, p, (void *)NT_PRSTATUS, &io) == 0);
}
static long nr_of(pid_t p) {
  int nr;
  struct iovec io = {&nr, sizeof nr};
  CHECK(ptrace(PTRACE_GETREGSET, p, (void *)NT_ARM_SYSTEM_CALL, &io) == 0);
  return nr;
}
static void set_nr(pid_t p, long v) {
  int nr = (int)v;
  struct iovec io = {&nr, sizeof nr};
  CHECK(ptrace(PTRACE_SETREGSET, p, (void *)NT_ARM_SYSTEM_CALL, &io) == 0);
}
static long arg0_of(pid_t p) { regs_t r; get_regs(p, &r); return (long)r.regs[0]; }
static void set_arg0(pid_t p, long v) { regs_t r; get_regs(p, &r); r.regs[0] = v; set_regs(p, &r); }
static long ret_of(pid_t p) { return arg0_of(p); }
static void set_ret(pid_t p, long v) { set_arg0(p, v); }
#else
#error "unsupported architecture"
#endif

/* Wait for `p` to stop and return its status. */
static int wait_stop(pid_t p) {
  int st;
  CHECK(waitpid(p, &st, __WALL) == p);
  CHECK(WIFSTOPPED(st));
  return st;
}

static int wait_exit(pid_t p) {
  int st;
  CHECK(waitpid(p, &st, __WALL) == p);
  CHECK(WIFEXITED(st));
  return WEXITSTATUS(st);
}

/* Wait for a tracee to exit, passing on each signal it stops for (as a
 * tracer must: a traced parent stops for its child's SIGCHLD). */
static int wait_exit_passing(pid_t p) {
  for (;;) {
    int st;
    CHECK(waitpid(p, &st, __WALL) == p);
    if (WIFEXITED(st)) return WEXITSTATUS(st);
    CHECK(WIFSTOPPED(st));
    CHECK(ptrace(PTRACE_CONT, p, 0, WSTOPSIG(st)) == 0);
  }
}

/* Fork a tracee that asks to be traced and stops itself; return it stopped. */
static pid_t traced_child(void (*body)(void)) {
  pid_t c = fork();
  CHECK(c >= 0);
  if (!c) {
    if (ptrace(PTRACE_TRACEME, 0, 0, 0) != 0) _exit(90);
    raise(SIGSTOP);
    body();
    _exit(0);
  }
  int st = wait_stop(c);
  CHECK(WSTOPSIG(st) == SIGSTOP);
  return c;
}

/* Resume to the next syscall stop (entry or exit) and check it is one. */
static void to_syscall_stop(pid_t c) {
  CHECK(ptrace(PTRACE_SYSCALL, c, 0, 0) == 0);
  int st = wait_stop(c);
  CHECK(WSTOPSIG(st) == (SIGTRAP | 0x80));
}

/* Run to the entry of the next syscall numbered `nr`, stepping over others. */
static void to_entry_of(pid_t c, long nr) {
  for (int i = 0; i < 200; i++) {
    to_syscall_stop(c);
    long n = nr_of(c);
    if (n == nr) return;
    to_syscall_stop(c); /* that call's exit */
  }
  CHECK(!"syscall never reached");
}

/* 1. TRACEME then execve: the new image starts with a SIGTRAP stop. */
static char self_path[256];
static void exec_body(void) {
  char *argv[] = {self_path, "exec-child", 0};
  execv(self_path, argv);
  _exit(91);
}
static void test_exec(void) {
  pid_t c = traced_child(exec_body);
  CHECK(ptrace(PTRACE_CONT, c, 0, 0) == 0);
  int st = wait_stop(c);
  CHECK(WSTOPSIG(st) == SIGTRAP && (st >> 16) == 0);
  CHECK(ptrace(PTRACE_CONT, c, 0, 0) == 0);
  CHECK(wait_exit(c) == 7);
  puts("exec: the new image stopped with SIGTRAP");
}

/* 2. Syscall stops: read the call, rewrite an argument, skip a call and
 * forge its result, and see GET_SYSCALL_INFO agree. */
static void syscall_body(void) {
  pid_t pp = getppid();
  /* The tracer rewrites the argument 999999 (no such process: ESRCH) to 0
   * (ourselves), so this returns our process group. */
  long g = syscall(SYS_getpgid, 999999);
  /* The tracer skips this call and makes it return 4242. */
  long f = syscall(SYS_getpid);
  _exit(pp > 1 && g == getpgid(0) && f == 4242 ? 0 : 1);
}
static void test_syscalls(void) {
  pid_t c = traced_child(syscall_body);
  CHECK(ptrace(PTRACE_SETOPTIONS, c, 0, PTRACE_O_TRACESYSGOOD | PTRACE_O_EXITKILL) == 0);

  to_entry_of(c, SYS_getppid);
  struct syscall_info info;
  memset(&info, 0, sizeof info);
  long n = ptrace(PTRACE_GET_SYSCALL_INFO, c, (void *)sizeof info, &info);
  CHECK(n > 0);
  CHECK(info.op == INFO_ENTRY && info.entry.nr == SYS_getppid);
  to_syscall_stop(c); /* exit */
  CHECK(ret_of(c) == getpid());
  memset(&info, 0, sizeof info);
  CHECK(ptrace(PTRACE_GET_SYSCALL_INFO, c, (void *)sizeof info, &info) > 0);
  CHECK(info.op == INFO_EXIT && info.exit.rval == getpid() && !info.exit.is_error);

  to_entry_of(c, SYS_getpgid);
  CHECK(arg0_of(c) == 999999);
  set_arg0(c, 0);
  to_syscall_stop(c);
  CHECK(ret_of(c) > 0);

  to_entry_of(c, SYS_getpid);
  set_nr(c, -1); /* skip it */
  to_syscall_stop(c);
  set_ret(c, 4242);
  CHECK(ptrace(PTRACE_CONT, c, 0, 0) == 0);
  CHECK(wait_exit(c) == 0);
  puts("syscalls: entry/exit stops read, rewrote and skipped calls");
}

/* 3. PEEKDATA/POKEDATA on the tracee's memory (a fork: same addresses). */
static volatile long cell = 0x1111;
static void poke_body(void) {
  raise(SIGSTOP); /* let the tracer peek and poke */
  _exit(cell == 0x2222 ? 0 : 1);
}
static void test_peek_poke(void) {
  pid_t c = traced_child(poke_body);
  CHECK(ptrace(PTRACE_CONT, c, 0, 0) == 0);
  CHECK(WSTOPSIG(wait_stop(c)) == SIGSTOP);
  errno = 0;
  long v = ptrace(PTRACE_PEEKDATA, c, (void *)&cell, 0);
  CHECK(errno == 0 && v == 0x1111);
  CHECK(ptrace(PTRACE_POKEDATA, c, (void *)&cell, (void *)0x2222) == 0);
  CHECK(ptrace(PTRACE_PEEKDATA, c, (void *)&cell, 0) == 0x2222);
  CHECK(ptrace(PTRACE_CONT, c, 0, 0) == 0);
  CHECK(wait_exit(c) == 0);
  puts("peek/poke: read and wrote the tracee's memory");
}

/* 4. Signals stop the tracee first; the tracer suppresses, inspects and
 * replaces them. */
static volatile int got_usr1, got_usr2;
static void on_usr1(int s) { (void)s; got_usr1++; }
static void on_usr2(int s) { (void)s; got_usr2++; }
static void signal_body(void) {
  signal(SIGUSR1, on_usr1);
  signal(SIGUSR2, on_usr2);
  kill(getpid(), SIGUSR1); /* suppressed by the tracer */
  kill(getpid(), SIGUSR2); /* replaced by SIGUSR1 */
  _exit(got_usr1 == 1 && got_usr2 == 0 ? 0 : 1);
}
static void test_signals(void) {
  pid_t c = traced_child(signal_body);
  CHECK(ptrace(PTRACE_CONT, c, 0, 0) == 0);
  CHECK(WSTOPSIG(wait_stop(c)) == SIGUSR1);
  CHECK(ptrace(PTRACE_CONT, c, 0, 0) == 0);
  CHECK(WSTOPSIG(wait_stop(c)) == SIGUSR2);
  siginfo_t si;
  CHECK(ptrace(PTRACE_GETSIGINFO, c, 0, &si) == 0);
  CHECK(si.si_signo == SIGUSR2 && si.si_pid == c);
  CHECK(ptrace(PTRACE_CONT, c, 0, SIGUSR1) == 0);
  CHECK(wait_exit(c) == 0);
  puts("signals: intercepted, suppressed and replaced");
}

/* 5. PTRACE_O_TRACEFORK: the child's fork stops with an event, and the new
 * process starts traced, stopped. */
static void fork_body(void) {
  pid_t g = fork();
  if (g < 0) _exit(2);
  if (!g) _exit(5);
  int st;
  if (waitpid(g, &st, 0) != g || !WIFEXITED(st) || WEXITSTATUS(st) != 5) _exit(3);
  _exit(0);
}
static void test_fork(void) {
  pid_t c = traced_child(fork_body);
  CHECK(ptrace(PTRACE_SETOPTIONS, c, 0, PTRACE_O_TRACEFORK) == 0);
  CHECK(ptrace(PTRACE_CONT, c, 0, 0) == 0);
  int st = wait_stop(c);
  CHECK(WSTOPSIG(st) == SIGTRAP && (st >> 16) == PTRACE_EVENT_FORK);
  unsigned long g = 0;
  CHECK(ptrace(PTRACE_GETEVENTMSG, c, 0, &g) == 0 && g > 0);
  st = wait_stop((pid_t)g);
  CHECK(WSTOPSIG(st) == SIGSTOP);
  CHECK(ptrace(PTRACE_CONT, (pid_t)g, 0, 0) == 0);
  CHECK(ptrace(PTRACE_CONT, c, 0, 0) == 0);
  CHECK(wait_exit((pid_t)g) == 5); /* the tracer hears of its exit */
  CHECK(wait_exit_passing(c) == 0); /* and the real parent reaped it too */
  puts("fork: the new process was attached and stopped");
}

/* 6. SEIZE then INTERRUPT: a PTRACE_EVENT_STOP without any signal. */
static void test_seize(void) {
  pid_t c = fork();
  CHECK(c >= 0);
  if (!c) {
    for (;;) pause();
  }
  CHECK(ptrace(PTRACE_SEIZE, c, 0, PTRACE_O_EXITKILL) == 0);
  CHECK(ptrace(PTRACE_INTERRUPT, c, 0, 0) == 0);
  int st = wait_stop(c);
  CHECK((st >> 16) == PTRACE_EVENT_STOP && WSTOPSIG(st) == SIGTRAP);
  CHECK(ptrace(PTRACE_KILL, c, 0, 0) == 0);
  CHECK(waitpid(c, &st, __WALL) == c && WIFSIGNALED(st) && WTERMSIG(st) == SIGKILL);
  puts("seize: interrupted with PTRACE_EVENT_STOP, then killed");
}

/* 7. PTRACE_O_EXITKILL: a tracee dies with its tracer. */
static void test_exitkill(void) {
  CHECK(prctl(PR_SET_CHILD_SUBREAPER, 1) == 0);
  int fds[2];
  CHECK(pipe(fds) == 0);
  pid_t t = fork();
  CHECK(t >= 0);
  if (!t) {
    pid_t g = fork();
    if (g < 0) _exit(2);
    if (!g) {
      if (ptrace(PTRACE_TRACEME, 0, 0, 0) != 0) _exit(90);
      raise(SIGSTOP);
      for (;;) pause();
    }
    int st;
    if (waitpid(g, &st, 0) != g || !WIFSTOPPED(st)) _exit(3);
    if (ptrace(PTRACE_SETOPTIONS, g, 0, PTRACE_O_EXITKILL) != 0) _exit(4);
    if (write(fds[1], &g, sizeof g) != sizeof g) _exit(5);
    _exit(0); /* the tracer goes; the tracee must go with it */
  }
  close(fds[1]);
  pid_t g;
  CHECK(read(fds[0], &g, sizeof g) == sizeof g);
  CHECK(wait_exit(t) == 0);
  int st;
  CHECK(waitpid(g, &st, 0) == g); /* reparented to us, the subreaper */
  CHECK(WIFSIGNALED(st) && WTERMSIG(st) == SIGKILL);
  CHECK(prctl(PR_SET_CHILD_SUBREAPER, 0) == 0);
  puts("exitkill: the tracee was killed when its tracer exited");
}

/* 8. seccomp's SECCOMP_RET_TRACE hands the call to the tracer as an event. */
static void seccomp_body(void) {
  struct sock_filter f[] = {
    BPF_STMT(BPF_LD | BPF_W | BPF_ABS, offsetof(struct seccomp_data, nr)),
    BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, SYS_getppid, 0, 1),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_TRACE | 0x77),
    BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
  };
  struct sock_fprog prog = {sizeof f / sizeof f[0], f};
  if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) _exit(2);
  if (syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER, 0, &prog) != 0) _exit(3);
  pid_t pp = (pid_t)syscall(SYS_getppid); /* the tracer lets it run */
  _exit(pp > 1 ? 0 : 1);
}
static void test_seccomp_trace(void) {
  pid_t c = traced_child(seccomp_body);
  CHECK(ptrace(PTRACE_SETOPTIONS, c, 0, PTRACE_O_TRACESECCOMP) == 0);
  CHECK(ptrace(PTRACE_CONT, c, 0, 0) == 0);
  int st = wait_stop(c);
  CHECK(WSTOPSIG(st) == SIGTRAP && (st >> 16) == PTRACE_EVENT_SECCOMP);
  unsigned long msg = 0;
  CHECK(ptrace(PTRACE_GETEVENTMSG, c, 0, &msg) == 0 && msg == 0x77);
  CHECK(nr_of(c) == SYS_getppid);
  CHECK(ptrace(PTRACE_CONT, c, 0, 0) == 0);
  CHECK(wait_exit(c) == 0);
  puts("seccomp: RET_TRACE stopped the tracee with PTRACE_EVENT_SECCOMP");
}

int main(int argc, char **argv) {
  if (argc == 2 && strcmp(argv[1], "exec-child") == 0) return 7;
  /* (The conformance root holds the program at /ptrace.) */
  snprintf(self_path, sizeof self_path, "%s", access("/ptrace", X_OK) == 0 ? "/ptrace" : "/proc/self/exe");
  setvbuf(stdout, 0, _IONBF, 0);
  /* As init (pid 1, as in the VM) every tracee's parent would be pid 1: run
   * the cases one level down, where a parent pid means a real one. */
  if (getpid() == 1) {
    pid_t k = fork();
    CHECK(k >= 0);
    if (k) {
      int st;
      CHECK(waitpid(k, &st, 0) == k);
      return WIFEXITED(st) ? WEXITSTATUS(st) : 1;
    }
  }
  test_exec();
  test_syscalls();
  test_peek_poke();
  test_signals();
  test_fork();
  test_seize();
  test_exitkill();
  test_seccomp_trace();
  puts("PASS");
  return 0;
}
