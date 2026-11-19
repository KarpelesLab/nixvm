/* ptrace and seccomp of i386 system calls, from a 64-bit tracer (x86-64,
 * non-PIE: gcc -static):
 *  - A 64-bit task's int $0x80: its syscall stops show the i386 number in
 *    orig_rax and its arguments in rbx, rcx, rdx, rsi, rdi (cs stays 0x33,
 *    and GETREGSET NT_PRSTATUS the 64-bit regs); GET_SYSCALL_INFO says
 *    AUDIT_ARCH_I386. The tracer changes the number and the arguments, and
 *    orig_rax -1 skips the call.
 *  - seccomp of int $0x80 from 64-bit code: the filters see AUDIT_ARCH_I386
 *    and the i386 numbers (RET_ERRNO, RET_TRAP's si_arch and si_syscall,
 *    USER_NOTIF's data), strict mode allows the i386 write but not getpid.
 *  - An i386 program (ptrace32, when it's next to this one) traced by this
 *    64-bit tracer: cs 0x23, orig_rax the i386 number, rbx..rbp its
 *    arguments, NT_PRSTATUS the 32-bit regs (the tracee's view), and a
 *    changed call.
 */
#define _GNU_SOURCE
#include <elf.h>
#include <errno.h>
#include <linux/audit.h>
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <signal.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/prctl.h>
#include <sys/ptrace.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <unistd.h>

static int failures;
#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("FAIL line %d: ", __LINE__);                                      \
      printf(__VA_ARGS__);                                                     \
      printf("\n");                                                            \
      failures++;                                                              \
    }                                                                          \
  } while (0)

#if defined(__x86_64__)

/* An i386 system call from 64-bit code: ebx, ecx, edx, esi, edi. */
static long int80(long nr, long b, long c, long d, long s, long di) {
  long r;
  __asm__ volatile("int $0x80"
                   : "=a"(r)
                   : "a"(nr), "b"(b), "c"(c), "d"(d), "S"(s), "D"(di)
                   : "memory", "r8", "r9", "r10", "r11");
  return r;
}

#define TAG_REGS 0x7a900001
#define TAG_ARG 0x7a900002
#define TAG_SKIP 0x7a900003

static void child64(void) {
  ptrace(PTRACE_TRACEME, 0, 0, 0);
  raise(SIGSTOP);
  int bad = 0;
  long ppid = getppid();
  /* i386 getpid (20), which the tracer makes getppid (64). */
  if (int80(20, 0x11, 0x22, 0x33, 0x44, TAG_REGS) != ppid)
    bad |= 1;
  /* dup(999), the argument made 0. */
  if (int80(41, 999, 0, 0, 0, TAG_ARG) < 0)
    bad |= 2;
  if (int80(20, 0, 0, 0, 0, TAG_SKIP) != 4242)
    bad |= 4;
  _exit(bad);
}

static void trace_int80(void) {
  pid_t c = fork();
  if (c == 0)
    child64();
  int st, entering = 1, seen = 0, native = 0;
  long tag = 0;
  waitpid(c, &st, 0);
  ptrace(PTRACE_SETOPTIONS, c, 0, PTRACE_O_TRACESYSGOOD);
  ptrace(PTRACE_SYSCALL, c, 0, 0);
  while (waitpid(c, &st, 0) == c && WIFSTOPPED(st)) {
    if (WSTOPSIG(st) != (SIGTRAP | 0x80)) {
      ptrace(PTRACE_SYSCALL, c, 0, WSTOPSIG(st));
      continue;
    }
    struct user_regs_struct r;
    ptrace(PTRACE_GETREGS, c, 0, &r);
    struct __ptrace_syscall_info info;
    memset(&info, 0, sizeof info);
    ptrace(PTRACE_GET_SYSCALL_INFO, c, sizeof info, &info);
    CHECK(r.cs == 0x33, "cs %#llx", r.cs);
    int i386 = (r.rdi & 0xffffff00) == 0x7a900000;
    if (entering) {
      tag = i386 ? (long)r.rdi : 0;
      CHECK(info.arch == (i386 ? AUDIT_ARCH_I386 : AUDIT_ARCH_X86_64),
            "arch %#x for call %lld", info.arch, r.orig_rax);
      if (!i386)
        native++;
      if (tag == TAG_REGS) {
        seen |= 1;
        CHECK(r.orig_rax == 20 && r.rbx == 0x11 && r.rcx == 0x22 &&
                  r.rdx == 0x33 && r.rsi == 0x44 && r.rax == -ENOSYS,
              "regs %lld %#llx %#llx %#llx %#llx", r.orig_rax, r.rbx, r.rcx,
              r.rdx, r.rsi);
        CHECK(info.entry.nr == 20 && info.entry.args[0] == 0x11 &&
                  info.entry.args[1] == 0x22 && info.entry.args[4] == TAG_REGS,
              "info nr %llu args %#llx", (unsigned long long)info.entry.nr,
              (unsigned long long)info.entry.args[0]);
        unsigned char buf[512];
        struct iovec iov = {buf, sizeof buf};
        CHECK(ptrace(PTRACE_GETREGSET, c, NT_PRSTATUS, &iov) == 0 &&
                  iov.iov_len == sizeof r,
              "NT_PRSTATUS of a 64-bit task: %zu bytes", iov.iov_len);
        CHECK(ptrace(PTRACE_POKEUSER, c, offsetof(struct user, regs.orig_rax),
                     64) == 0,
              "POKEUSER: %m");
      } else if (tag == TAG_ARG) {
        seen |= 2;
        r.rbx = 0;
        ptrace(PTRACE_SETREGS, c, 0, &r);
      } else if (tag == TAG_SKIP) {
        seen |= 4;
        r.orig_rax = -1;
        r.rax = 4242;
        ptrace(PTRACE_SETREGS, c, 0, &r);
      }
    } else {
      if (tag == TAG_REGS)
        CHECK(r.orig_rax == 64 && (pid_t)r.rax == getpid(),
              "getppid at exit: %lld returned %lld", r.orig_rax, r.rax);
      if (tag)
        CHECK(info.arch == AUDIT_ARCH_I386, "arch at exit %#x", info.arch);
      tag = 0;
    }
    entering = !entering;
    ptrace(PTRACE_SYSCALL, c, 0, 0);
  }
  CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "int $0x80 tracee: %#x", st);
  CHECK(seen == 7 && native > 0, "saw %#x, %d x86-64 calls", seen, native);
}

/* ---- seccomp of int $0x80 ------------------------------------------- */

#define LD(off) BPF_STMT(BPF_LD | BPF_W | BPF_ABS, (off))
#define JEQ(k, t, f) BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, (k), (t), (f))
#define RET(k) BPF_STMT(BPF_RET | BPF_K, (k))
#define NR offsetof(struct seccomp_data, nr)
#define ARCH offsetof(struct seccomp_data, arch)

static long filter(struct sock_filter *f, unsigned short n, unsigned flags) {
  struct sock_fprog prog = {n, f};
  prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
  return syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER, flags, &prog);
}

static volatile int sys_seen, sys_nr;
static volatile unsigned sys_arch;

static void on_sigsys(int sig, siginfo_t *si, void *uc) {
  (void)uc;
  sys_seen = sig;
  sys_nr = si->si_syscall;
  sys_arch = si->si_arch;
}

static int in_child(int (*f)(void)) {
  pid_t c = fork();
  if (c == 0)
    _exit(f());
  int st;
  waitpid(c, &st, 0);
  return st;
}

/* i386 getpid (20) is an errno, i386 getuid32 (199) a trap; the x86-64
 * calls with those numbers (writev, gettimeofday...) are left alone. */
static int filtered(void) {
  struct sock_filter f[] = {
      LD(ARCH),
      JEQ(AUDIT_ARCH_I386, 0, 5),
      LD(NR),
      JEQ(20, 0, 1),
      RET(SECCOMP_RET_ERRNO | 33),
      JEQ(199, 0, 1),
      RET(SECCOMP_RET_TRAP),
      RET(SECCOMP_RET_ALLOW),
  };
  if (filter(f, sizeof f / sizeof f[0], 0) != 0)
    return 128;
  int bad = 0;
  if (int80(20, 0, 0, 0, 0, 0) != -33)
    bad |= 1;
  if (getpid() <= 0)
    bad |= 2;
  struct sigaction sa = {.sa_sigaction = on_sigsys, .sa_flags = SA_SIGINFO};
  sigaction(SIGSYS, &sa, 0);
  long r = int80(199, 0, 0, 0, 0, 0);
  if (sys_seen != SIGSYS || sys_nr != 199 || sys_arch != AUDIT_ARCH_I386 ||
      r != 199)
    bad |= 4;
  /* The x86-64 number 199 is fremovexattr: not trapped. */
  errno = 0;
  sys_seen = 0;
  syscall(199, "/nonexistent", "user.x");
  if (sys_seen)
    bad |= 8;
  return bad;
}

static int notified(void) {
  struct sock_filter f[] = {
      LD(ARCH),
      JEQ(AUDIT_ARCH_I386, 0, 3),
      LD(NR),
      JEQ(20, 0, 1),
      RET(SECCOMP_RET_USER_NOTIF),
      RET(SECCOMP_RET_ALLOW),
  };
  int fd = filter(f, sizeof f / sizeof f[0], SECCOMP_FILTER_FLAG_NEW_LISTENER);
  if (fd < 0)
    return 128;
  pid_t c = fork();
  if (c == 0)
    _exit(int80(20, 0x77, 0x88, 0, 0, 0) == 555 ? 0 : 1);
  struct seccomp_notif req;
  memset(&req, 0, sizeof req);
  if (ioctl(fd, SECCOMP_IOCTL_NOTIF_RECV, &req) != 0)
    return 64;
  int bad = 0;
  if (req.data.arch != AUDIT_ARCH_I386 || req.data.nr != 20 ||
      req.data.args[0] != 0x77 || req.data.args[1] != 0x88)
    bad |= 1;
  struct seccomp_notif_resp resp = {.id = req.id, .val = 555};
  if (ioctl(fd, SECCOMP_IOCTL_NOTIF_SEND, &resp) != 0)
    bad |= 2;
  int st;
  waitpid(c, &st, 0);
  if (!WIFEXITED(st) || WEXITSTATUS(st) != 0)
    bad |= 4;
  return bad;
}

static char out[2];

static int strict(void) {
  int p[2];
  if (pipe(p) != 0)
    return 1;
  prctl(PR_SET_SECCOMP, SECCOMP_MODE_STRICT);
  /* The i386 write, then the x86-64 one. */
  if (int80(4, p[1], (long)"x", 1, 0, 0) != 1)
    syscall(SYS_exit, 3);
  if (write(p[1], "y", 1) != 1)
    syscall(SYS_exit, 4);
  if (int80(3, p[0], (long)out, 2, 0, 0) != 2)
    syscall(SYS_exit, 5);
  int80(20, 0, 0, 0, 0, 0); /* i386 getpid: killed */
  syscall(SYS_exit, 6);
  return 7;
}

static void seccomp_checks(void) {
  int st = in_child(filtered);
  CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "filtered int $0x80: %#x", st);
  st = in_child(notified);
  CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "USER_NOTIF: %#x", st);
  st = in_child(strict);
  CHECK(WIFSIGNALED(st) && WTERMSIG(st) == SIGKILL, "strict mode: %#x", st);
}

/* ---- a 64-bit tracer of an i386 program --------------------------------- */

/* The i386 user_regs_struct. */
struct regs32 {
  uint32_t ebx, ecx, edx, esi, edi, ebp, eax;
  uint32_t ds, es, fs, gs, orig_eax, eip, cs, eflags, esp, ss;
};

static void trace_ia32(void) {
  char self[512], path[600];
  ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
  if (n <= 0)
    return;
  self[n] = 0;
  char *slash = strrchr(self, '/');
  if (!slash)
    return;
  *slash = 0;
  snprintf(path, sizeof path, "%s/ptrace32", self);
  if (access(path, X_OK) != 0)
    return;
  pid_t c = fork();
  if (c == 0) {
    ptrace(PTRACE_TRACEME, 0, 0, 0);
    char *argv[] = {path, "tracee", NULL};
    execv(path, argv);
    _exit(99);
  }
  int st, entering = 1, seen = 0, stops = 0;
  long tag = 0;
  waitpid(c, &st, 0);
  CHECK(WIFSTOPPED(st) && WSTOPSIG(st) == SIGTRAP, "exec stop %#x", st);
  ptrace(PTRACE_SETOPTIONS, c, 0, PTRACE_O_TRACESYSGOOD);
  ptrace(PTRACE_SYSCALL, c, 0, 0);
  while (waitpid(c, &st, 0) == c && WIFSTOPPED(st)) {
    if (WSTOPSIG(st) != (SIGTRAP | 0x80)) {
      ptrace(PTRACE_SYSCALL, c, 0, WSTOPSIG(st));
      continue;
    }
    stops++;
    struct user_regs_struct r;
    ptrace(PTRACE_GETREGS, c, 0, &r);
    struct __ptrace_syscall_info info;
    memset(&info, 0, sizeof info);
    ptrace(PTRACE_GET_SYSCALL_INFO, c, sizeof info, &info);
    CHECK(r.cs == 0x23 && r.ss == 0x2b, "cs %#llx ss %#llx", r.cs, r.ss);
    CHECK(info.arch == AUDIT_ARCH_I386, "arch %#x", info.arch);
    /* The tracee's view: an i386 task's general regs are 68 bytes. */
    unsigned char buf[sizeof r];
    struct iovec iov = {buf, sizeof buf};
    CHECK(ptrace(PTRACE_GETREGSET, c, NT_PRSTATUS, &iov) == 0 &&
              iov.iov_len == sizeof(struct regs32),
          "NT_PRSTATUS of an i386 task: %zu bytes", iov.iov_len);
    struct regs32 r32;
    memcpy(&r32, buf, sizeof r32);
    CHECK(r32.orig_eax == (uint32_t)r.orig_rax && r32.ebx == (uint32_t)r.rbx &&
              r32.eip == (uint32_t)r.rip && r32.cs == 0x23,
          "regs32 orig_eax %u ebx %#x eip %#x", r32.orig_eax, r32.ebx, r32.eip);
    if (entering) {
      CHECK(r.orig_rax < 0x1000 && r.rax == -ENOSYS,
            "orig_rax %lld rax %lld at entry", r.orig_rax, r.rax);
      tag = r.rbx == 0x1111 && r.orig_rax == 20;
      if (tag) {
        seen |= 1;
        CHECK(r.rcx == 0x2222 && r.rdx == 0x3333 && r.rsi == 0x4444 &&
                  r.rdi == 0x5555 && r.rbp == 0x6666,
              "args %#llx %#llx %#llx %#llx %#llx", r.rcx, r.rdx, r.rsi, r.rdi,
              r.rbp);
        CHECK(info.entry.nr == 20 && info.entry.args[0] == 0x1111 &&
                  info.entry.args[5] == 0x6666,
              "info nr %llu", (unsigned long long)info.entry.nr);
        r.orig_rax = 64; /* getppid */
        CHECK(ptrace(PTRACE_SETREGS, c, 0, &r) == 0, "SETREGS: %m");
      }
    } else if (tag) {
      seen |= 2;
      CHECK(r.orig_rax == 64 && (pid_t)r.rax == getpid(), "getppid: %lld",
            r.rax);
      tag = 0;
    }
    entering = !entering;
    ptrace(PTRACE_SYSCALL, c, 0, 0);
  }
  CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "the i386 tracee: %#x", st);
  CHECK(seen == 3 && stops > 4, "saw %#x in %d stops", seen, stops);
}

int main(void) {
  setvbuf(stdout, 0, _IONBF, 0);
  trace_int80();
  seccomp_checks();
  trace_ia32();
  if (!failures)
    printf("PASS\n");
  return failures != 0;
}

#else
int main(void) {
  printf("PASS\n");
  return 0;
}
#endif
