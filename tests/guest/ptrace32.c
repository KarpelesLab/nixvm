/* ptrace and seccomp of i386 system calls, from an i386 program
 * (gcc -m32 -static): Linux's compat paths.
 *  - A 32-bit tracer of a 32-bit tracee: GETREGS/SETREGS with
 *    user_regs_struct (32-bit), PEEKUSER/POKEUSER at struct user offsets
 *    (EIO past it or off a word), PEEKDATA/POKEDATA of 4-byte words,
 *    GETFPREGS (the fsave image), SETFPREGS, GETFPXREGS (fxsave),
 *    GET_THREAD_AREA/SET_THREAD_AREA, GETSIGINFO (compat siginfo),
 *    GETEVENTMSG (a 4-byte word), GETREGSET NT_PRSTATUS (68 bytes),
 *    NT_PRFPREG (108), NT_PRXFPREG, NT_X86_XSTATE, NT_386_TLS, and
 *    GET_SYSCALL_INFO (AUDIT_ARCH_I386).
 *  - Syscall stops: the tracer sees and changes the i386 number (orig_eax)
 *    and arguments (ebx, ecx, edx, esi, edi, ebp); orig_eax -1 skips the
 *    call with the eax the tracer left; eax at the exit stop is the result.
 *  - seccomp filters see AUDIT_ARCH_I386, the i386 numbers and arguments:
 *    RET_ERRNO, RET_TRAP (SIGSYS with si_syscall, si_arch), KILL_PROCESS,
 *    RET_TRACE (the tracer changes the call), USER_NOTIF (the notification's
 *    data), and strict mode, which allows only the i386 read, write, exit
 *    and sigreturn (not rt_sigreturn).
 * `ptrace32 tracee` is the program ptrace_ia32 (x86-64) traces.
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

#if defined(__i386__)

#ifndef NT_PRXFPREG
#define NT_PRXFPREG 0x46e62b7f
#endif
#ifndef NT_386_TLS
#define NT_386_TLS 0x200
#endif
#ifndef NT_X86_XSTATE
#define NT_X86_XSTATE 0x202
#endif

/* struct user_desc (asm/ldt.h). */
struct udesc {
  unsigned int entry_number;
  unsigned int base_addr;
  unsigned int limit;
  unsigned int seg_32bit : 1;
  unsigned int contents : 2;
  unsigned int read_exec_only : 1;
  unsigned int limit_in_pages : 1;
  unsigned int seg_not_present : 1;
  unsigned int useable : 1;
};

/* An i386 system call with all six argument registers, through int $0x80:
 * v[0] the number, v[1..6] ebx, ecx, edx, esi, edi, ebp. */
static long int80(long nr, long b, long c, long d, long s, long di, long bp) {
  long v[7] = {nr, b, c, d, s, di, bp};
  long r = (long)v;
  __asm__ volatile("push %%ebp\n\t"
                   "push %%ebx\n\t"
                   "push %%esi\n\t"
                   "push %%edi\n\t"
                   "mov 4(%%eax), %%ebx\n\t"
                   "mov 8(%%eax), %%ecx\n\t"
                   "mov 12(%%eax), %%edx\n\t"
                   "mov 16(%%eax), %%esi\n\t"
                   "mov 20(%%eax), %%edi\n\t"
                   "mov 24(%%eax), %%ebp\n\t"
                   "mov (%%eax), %%eax\n\t"
                   "int $0x80\n\t"
                   "pop %%edi\n\t"
                   "pop %%esi\n\t"
                   "pop %%ebx\n\t"
                   "pop %%ebp"
                   : "+a"(r)
                   :
                   : "ecx", "edx", "memory");
  return r;
}

/* Tags in %ebp, which say which call the tracer is looking at. */
#define TAG_REGS 0x7a900001
#define TAG_ARG 0x7a900002
#define TAG_SKIP 0x7a900003
#define TAG_RET 0x7a900004

static volatile unsigned words[2] = {0x11111111, 0x22222222};

static unsigned short fpu_cw(void) {
  unsigned short cw;
  __asm__ volatile("fnstcw %0" : "=m"(cw));
  return cw;
}

static void set_fpu_cw(unsigned short cw) {
  __asm__ volatile("fldcw %0" ::"m"(cw));
}

static unsigned short gs_sel(void) {
  unsigned short s;
  __asm__ volatile("mov %%gs, %0" : "=r"(s));
  return s;
}

/* The traced child: its exit status says which of its checks failed. */
static void tracee(void) {
  int bad = 0;
  ptrace(PTRACE_TRACEME, 0, 0, 0);
  set_fpu_cw(0x27f);
  raise(SIGSTOP);
  /* The tracer set the control word, a word of memory and TLS entry 13. */
  if (fpu_cw() != 0x7f)
    bad |= 1;
  set_fpu_cw(0x37f);
  if (words[0] != 0xaabbccdd || words[1] != 0x22222222)
    bad |= 2;
  struct udesc d = {.entry_number = 13};
  if (syscall(SYS_get_thread_area, &d) != 0 || d.base_addr != 0x12345000)
    bad |= 4;
  long ppid = getppid();
  /* getpid, made getppid at its entry. */
  if (int80(20, 0x11, 0x22, 0x33, 0x44, 0x55, TAG_REGS) != ppid)
    bad |= 8;
  /* dup(999), whose argument becomes 0. */
  if (int80(41, 999, 0, 0, 0, 0, TAG_ARG) < 0)
    bad |= 16;
  /* Skipped, returning what the tracer left in eax. */
  if (int80(20, 0, 0, 0, 0, 0, TAG_SKIP) != 4242)
    bad |= 32;
  /* getuid32, whose result the tracer replaces at the exit stop. */
  if (int80(199, 0, 0, 0, 0, 0, TAG_RET) != 777)
    bad |= 64;
  _exit(bad);
}

static void check_stop(pid_t c) {
  struct user_regs_struct r;
  CHECK(ptrace(PTRACE_GETREGS, c, 0, &r) == 0, "GETREGS: %m");
  CHECK(r.xcs == 0x23 && r.xss == 0x2b, "cs %#lx ss %#lx", r.xcs, r.xss);
  CHECK(r.xds == 0x2b && r.xes == 0x2b, "ds %#lx es %#lx", r.xds, r.xes);
  CHECK(r.xgs == gs_sel(), "gs %#lx, mine %#x", r.xgs, gs_sel());
  CHECK(r.eflags & 0x202, "eflags %#lx", r.eflags);
  /* Stopped by the tgkill raise made. */
  CHECK(r.orig_eax == 270 || r.orig_eax == 238, "orig_eax %ld", r.orig_eax);
  errno = 0;
  long eip = ptrace(PTRACE_PEEKUSER, c, offsetof(struct user, regs.eip), 0);
  CHECK(errno == 0 && eip == r.eip, "PEEKUSER eip %#lx, %#lx", eip, r.eip);
  long esp = ptrace(PTRACE_PEEKUSER, c, offsetof(struct user, regs.esp), 0);
  CHECK(esp == r.esp, "PEEKUSER esp %#lx", esp);
  errno = 0;
  long past = ptrace(PTRACE_PEEKUSER, c, sizeof(struct user), 0);
  CHECK(errno == 0 && past == 0, "PEEKUSER at the end: %ld %m", past);
  errno = 0;
  ptrace(PTRACE_PEEKUSER, c, sizeof(struct user) + 4, 0);
  CHECK(errno == EIO, "PEEKUSER past the end: %m");
  errno = 0;
  ptrace(PTRACE_PEEKUSER, c, 2, 0);
  CHECK(errno == EIO, "PEEKUSER off a word: %m");
  /* A selector with RPL 0, and a null %cs. */
  CHECK(ptrace(PTRACE_POKEUSER, c, offsetof(struct user, regs.xds), 0x10) ==
                -1 && errno == EIO,
        "POKEUSER ds 0x10: %m");
  CHECK(ptrace(PTRACE_POKEUSER, c, offsetof(struct user, regs.xcs), 0) == -1 &&
            errno == EIO,
        "POKEUSER cs 0: %m");

  /* The general regset: 68 bytes, the same registers. */
  struct user_regs_struct rs;
  memset(&rs, 0xee, sizeof rs);
  struct iovec iov = {&rs, sizeof rs + 32};
  CHECK(ptrace(PTRACE_GETREGSET, c, NT_PRSTATUS, &iov) == 0, "GETREGSET: %m");
  CHECK(iov.iov_len == 68 && sizeof rs == 68, "NT_PRSTATUS is %zu bytes",
        iov.iov_len);
  CHECK(memcmp(&rs, &r, sizeof r) == 0, "NT_PRSTATUS differs from GETREGS");
  iov.iov_len = 6;
  CHECK(ptrace(PTRACE_GETREGSET, c, NT_PRSTATUS, &iov) == -1 && errno == EINVAL,
        "a length off a word: %m");

  /* Memory, a 32-bit word at a time. */
  errno = 0;
  long w = ptrace(PTRACE_PEEKDATA, c, &words[0], 0);
  CHECK(errno == 0 && (unsigned long)w == 0x11111111, "PEEKDATA %#lx", w);
  w = ptrace(PTRACE_PEEKTEXT, c, &words[1], 0);
  CHECK((unsigned long)w == 0x22222222, "PEEKTEXT %#lx", w);
  CHECK(ptrace(PTRACE_POKEDATA, c, &words[0], 0xaabbccdd) == 0, "POKEDATA: %m");

  /* The FPU: the fsave image, its control word the one the child set. */
  struct user_fpregs_struct fp;
  CHECK(ptrace(PTRACE_GETFPREGS, c, 0, &fp) == 0, "GETFPREGS: %m");
  CHECK(sizeof fp == 108 && fp.cwd == 0xffff027f, "fsave cwd %#lx", fp.cwd);
  CHECK((fp.fcs & 0xffff) == 0x23, "fsave fcs %#lx", fp.fcs);
  CHECK((fp.twd & 0xffff) == 0xffff, "fsave twd %#lx (all empty)", fp.twd);
  struct user_fpxregs_struct fx;
  CHECK(ptrace(PTRACE_GETFPXREGS, c, 0, &fx) == 0, "GETFPXREGS: %m");
  CHECK(fx.cwd == 0x27f && (fx.mxcsr & 0xffc0) == 0x1f80,
        "fxsave cwd %#x mxcsr %#lx", fx.cwd, fx.mxcsr);
  unsigned char buf[4096];
  iov = (struct iovec){buf, sizeof buf};
  CHECK(ptrace(PTRACE_GETREGSET, c, NT_PRFPREG, &iov) == 0 &&
            iov.iov_len == 108 && memcmp(buf, &fp, 108) == 0,
        "NT_PRFPREG: %zu bytes", iov.iov_len);
  iov = (struct iovec){buf, sizeof buf};
  CHECK(ptrace(PTRACE_GETREGSET, c, NT_PRXFPREG, &iov) == 0 &&
            iov.iov_len == 512 && memcmp(buf, &fx, 160) == 0,
        "NT_PRXFPREG: %zu bytes", iov.iov_len);
  iov = (struct iovec){buf, sizeof buf};
  CHECK(ptrace(PTRACE_GETREGSET, c, NT_X86_XSTATE, &iov) == 0 &&
            iov.iov_len >= 576 && iov.iov_len < sizeof buf,
        "NT_X86_XSTATE: %zu bytes", iov.iov_len);
  uint64_t xcr0;
  memcpy(&xcr0, buf + 464, 8);
  CHECK((xcr0 & 3) == 3 && buf[0] == 0x7f && buf[1] == 2,
        "xstate xcr0 %#llx cwd %#x", (unsigned long long)xcr0,
        buf[0] | buf[1] << 8);
  fp.cwd = 0x7f;
  CHECK(ptrace(PTRACE_SETFPREGS, c, 0, &fp) == 0, "SETFPREGS: %m");
  memset(&fx, 0, sizeof fx);
  ptrace(PTRACE_GETFPXREGS, c, 0, &fx);
  CHECK(fx.cwd == 0x7f && (fx.mxcsr & 0xffc0) == 0x1f80,
        "after SETFPREGS: cwd %#x mxcsr %#lx", fx.cwd, fx.mxcsr);

  /* The TLS entries: the child's 12 is its copy of mine. */
  struct udesc mine = {.entry_number = 12}, its;
  syscall(SYS_get_thread_area, &mine);
  memset(&its, 0, sizeof its);
  CHECK(ptrace(PTRACE_GET_THREAD_AREA, c, 12, &its) == 0, "GET_THREAD_AREA: %m");
  CHECK(its.entry_number == 12 && its.base_addr == mine.base_addr &&
            its.seg_32bit && its.limit_in_pages,
        "entry %u base %#x, mine %#x", its.entry_number, its.base_addr,
        mine.base_addr);
  CHECK(ptrace(PTRACE_GET_THREAD_AREA, c, 15, &its) == -1 && errno == EINVAL,
        "GET_THREAD_AREA 15: %m");
  struct udesc set = {.entry_number = 0,
                      .base_addr = 0x12345000,
                      .limit = 0xfffff,
                      .seg_32bit = 1,
                      .limit_in_pages = 1,
                      .useable = 1};
  CHECK(ptrace(PTRACE_SET_THREAD_AREA, c, 13, &set) == 0, "SET_THREAD_AREA: %m");
  struct udesc tls[4];
  memset(tls, 0, sizeof tls);
  iov = (struct iovec){tls, sizeof tls};
  CHECK(ptrace(PTRACE_GETREGSET, c, NT_386_TLS, &iov) == 0 &&
            iov.iov_len == 48 && tls[0].entry_number == 12 &&
            tls[1].entry_number == 13 && tls[1].base_addr == 0x12345000,
        "NT_386_TLS: %zu bytes, %u %u %#x", iov.iov_len, tls[0].entry_number,
        tls[1].entry_number, tls[1].base_addr);

  /* The signal, in the 32-bit siginfo. */
  siginfo_t si;
  memset(&si, 0, sizeof si);
  CHECK(ptrace(PTRACE_GETSIGINFO, c, 0, &si) == 0, "GETSIGINFO: %m");
  CHECK(si.si_signo == SIGSTOP && si.si_code == SI_TKILL && si.si_pid == c,
        "siginfo %d code %d pid %d", si.si_signo, si.si_code, si.si_pid);
}

static void tracer(void) {
  pid_t c = fork();
  if (c == 0)
    tracee();
  int st;
  CHECK(waitpid(c, &st, 0) == c && WIFSTOPPED(st) && WSTOPSIG(st) == SIGSTOP,
        "first stop %#x", st);
  check_stop(c);
  CHECK(ptrace(PTRACE_SETOPTIONS, c, 0,
               PTRACE_O_TRACESYSGOOD | PTRACE_O_TRACEEXIT) == 0,
        "SETOPTIONS: %m");
  ptrace(PTRACE_SYSCALL, c, 0, 0);
  int entering = 1, seen = 0;
  long tag = 0;
  for (;;) {
    if (waitpid(c, &st, 0) != c || !WIFSTOPPED(st))
      break;
    if (st >> 8 == (SIGTRAP | PTRACE_EVENT_EXIT << 8)) {
      uint64_t msg = ~0ull;
      CHECK(ptrace(PTRACE_GETEVENTMSG, c, 0, &msg) == 0, "GETEVENTMSG: %m");
      /* A 32-bit word, the exit status (0 if every check passed). */
      CHECK(msg >> 32 == 0xffffffff && (msg & 0xffff00ff) == 0,
            "event message %#llx", (unsigned long long)msg);
      seen |= 16;
      ptrace(PTRACE_CONT, c, 0, 0);
      continue;
    }
    if (WSTOPSIG(st) != (SIGTRAP | 0x80)) {
      ptrace(PTRACE_SYSCALL, c, 0, WSTOPSIG(st));
      continue;
    }
    struct user_regs_struct r;
    ptrace(PTRACE_GETREGS, c, 0, &r);
    struct __ptrace_syscall_info info;
    memset(&info, 0, sizeof info);
    long n = ptrace(PTRACE_GET_SYSCALL_INFO, c, sizeof info, &info);
    CHECK(n > 0 && info.op == (entering ? PTRACE_SYSCALL_INFO_ENTRY
                                        : PTRACE_SYSCALL_INFO_EXIT),
          "syscall info op %d at %s", info.op, entering ? "entry" : "exit");
    CHECK(info.arch == AUDIT_ARCH_I386, "syscall info arch %#x", info.arch);
    if (entering) {
      tag = r.ebp;
      CHECK(r.eax == -ENOSYS, "eax at entry %ld", r.eax);
      CHECK(info.entry.nr == (unsigned long)r.orig_eax &&
                info.entry.args[0] == (unsigned long)r.ebx &&
                info.entry.args[5] == (unsigned long)r.ebp,
            "info nr %llu args %#llx %#llx", (unsigned long long)info.entry.nr,
            (unsigned long long)info.entry.args[0],
            (unsigned long long)info.entry.args[5]);
      if (tag == TAG_REGS) {
        seen |= 1;
        CHECK(r.orig_eax == 20 && r.ebx == 0x11 && r.ecx == 0x22 &&
                  r.edx == 0x33 && r.esi == 0x44 && r.edi == 0x55,
              "regs: %ld %#lx %#lx %#lx %#lx %#lx", r.orig_eax, r.ebx, r.ecx,
              r.edx, r.esi, r.edi);
        CHECK(info.entry.args[1] == 0x22 && info.entry.args[4] == 0x55 &&
                  info.instruction_pointer == (unsigned long)r.eip &&
                  info.stack_pointer == (unsigned long)r.esp,
              "info args/ip/sp");
        long o = ptrace(PTRACE_PEEKUSER, c, offsetof(struct user, regs.orig_eax), 0);
        CHECK(o == 20, "PEEKUSER orig_eax %ld", o);
        CHECK(ptrace(PTRACE_POKEUSER, c, offsetof(struct user, regs.orig_eax),
                     SYS_getppid) == 0,
              "POKEUSER orig_eax: %m");
      } else if (tag == TAG_ARG) {
        seen |= 2;
        CHECK(r.orig_eax == 41 && r.ebx == 999, "dup(%ld)", r.ebx);
        r.ebx = 0;
        CHECK(ptrace(PTRACE_SETREGS, c, 0, &r) == 0, "SETREGS: %m");
      } else if (tag == TAG_SKIP) {
        seen |= 4;
        r.orig_eax = -1;
        r.eax = 4242;
        ptrace(PTRACE_SETREGS, c, 0, &r);
      }
    } else {
      if (tag == TAG_REGS)
        CHECK(r.orig_eax == SYS_getppid && r.eax == getpid(),
              "the changed call: %ld returned %ld", r.orig_eax, r.eax);
      if (tag == TAG_ARG)
        CHECK(info.exit.is_error == 0 && info.exit.rval >= 0,
              "dup(0) returned %lld", (long long)info.exit.rval);
      if (tag == TAG_SKIP)
        CHECK(r.orig_eax == -1 && r.eax == 4242, "skipped: %ld %ld",
              r.orig_eax, r.eax);
      if (tag == TAG_RET) {
        seen |= 8;
        CHECK(r.orig_eax == 199 && r.eax == getuid(), "getuid32: %ld",
              r.eax);
        ptrace(PTRACE_POKEUSER, c, offsetof(struct user, regs.eax), 777);
      }
      tag = 0;
    }
    entering = !entering;
    ptrace(PTRACE_SYSCALL, c, 0, 0);
  }
  CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "the tracee: status %#x", st);
  CHECK(seen == 31, "saw %#x of the stops", seen);
}

/* Single steps of 32-bit code: each a SIGTRAP stop at the next
 * instruction (TRAP_TRACE), and a step over int $0x80 stops after it. */
static void steps(void) {
  pid_t c = fork();
  if (c == 0) {
    ptrace(PTRACE_TRACEME, 0, 0, 0);
    raise(SIGSTOP);
    volatile int x = 0;
    for (int i = 0; i < 20; i++)
      x += i;
    int80(20, 0, 0, 0, 0, 0, 0);
    _exit(x == 190 ? 0 : 1);
  }
  int st, moved = 0, n = 0;
  waitpid(c, &st, 0);
  struct user_regs_struct r;
  ptrace(PTRACE_GETREGS, c, 0, &r);
  long eip = r.eip;
  for (; n < 200; n++) {
    if (ptrace(PTRACE_SINGLESTEP, c, 0, 0) != 0)
      break;
    if (waitpid(c, &st, 0) != c || !WIFSTOPPED(st) || WSTOPSIG(st) != SIGTRAP)
      break;
    ptrace(PTRACE_GETREGS, c, 0, &r);
    moved += r.eip != eip;
    eip = r.eip;
    CHECK(r.xcs == 0x23, "step %d: cs %#lx", n, r.xcs);
  }
  CHECK(n >= 60 && moved >= n - 5, "%d steps, %d moved, status %#x", n, moved,
        st);
  if (WIFSTOPPED(st)) {
    siginfo_t si;
    ptrace(PTRACE_GETSIGINFO, c, 0, &si);
    CHECK(si.si_signo == SIGTRAP, "step siginfo %d code %d", si.si_signo,
          si.si_code);
    ptrace(PTRACE_CONT, c, 0, 0);
    waitpid(c, &st, 0);
  }
  CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "stepped child: %#x", st);
}

/* ---- seccomp ---------------------------------------------------------- */

#define LD(off) BPF_STMT(BPF_LD | BPF_W | BPF_ABS, (off))
#define JEQ(k, t, f) BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, (k), (t), (f))
#define RET(k) BPF_STMT(BPF_RET | BPF_K, (k))
#define NR offsetof(struct seccomp_data, nr)
#define ARCH offsetof(struct seccomp_data, arch)
#define ARG0 offsetof(struct seccomp_data, args[0])

static long filter(struct sock_filter *f, unsigned short n, unsigned flags) {
  struct sock_fprog prog = {n, f};
  prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
  return syscall(SYS_seccomp, SECCOMP_SET_MODE_FILTER, flags, &prog);
}

static volatile int sys_seen;
static volatile int sys_nr;
static volatile unsigned sys_arch;
static volatile void *sys_addr;

static void on_sigsys(int sig, siginfo_t *si, void *uc) {
  (void)uc;
  sys_seen = sig;
  sys_nr = si->si_syscall;
  sys_arch = si->si_arch;
  sys_addr = si->si_call_addr;
}

/* Run `f` in a child; its exit status. */
static int in_child(int (*f)(void)) {
  pid_t c = fork();
  if (c == 0)
    _exit(f());
  int st;
  waitpid(c, &st, 0);
  return st;
}

static int filtered(void) {
  struct sock_filter f[] = {
      LD(ARCH),
      JEQ(AUDIT_ARCH_I386, 1, 0),
      RET(SECCOMP_RET_KILL_PROCESS),
      LD(NR),
      JEQ(20, 0, 1),
      RET(SECCOMP_RET_ERRNO | 77),
      JEQ(199, 0, 1),
      RET(SECCOMP_RET_TRAP | 5),
      JEQ(96, 0, 3),
      LD(ARG0),
      JEQ(0x1234, 0, 1),
      RET(SECCOMP_RET_ERRNO | 55),
      RET(SECCOMP_RET_ALLOW),
  };
  if (filter(f, sizeof f / sizeof f[0], 0) != 0) {
    printf("FAIL seccomp: %m\n");
    return 1;
  }
  int bad = 0;
  if (int80(20, 0, 0, 0, 0, 0, 0) != -77)
    bad |= 1;
  /* The same call through the vDSO's fast entry. */
  errno = 0;
  if (syscall(SYS_getpid) != -1 || errno != 77)
    bad |= 2;
  if (int80(96, 0x1234, 0, 0, 0, 0, 0) != -55)
    bad |= 4;
  if (int80(96, 0, 0, 0, 0, 0, 0) < 0)
    bad |= 8;
  struct sigaction sa = {.sa_sigaction = on_sigsys, .sa_flags = SA_SIGINFO};
  sigaction(SIGSYS, &sa, 0);
  long r = int80(199, 0, 0, 0, 0, 0, 0);
  /* The handler returns to the call's registers: eax is its number. */
  if (sys_seen != SIGSYS || sys_nr != 199 || sys_arch != AUDIT_ARCH_I386 ||
      r != 199 || !sys_addr)
    bad |= 16;
  return bad;
}

static int killed(void) {
  struct sock_filter f[] = {
      LD(NR),
      JEQ(64, 0, 1),
      RET(SECCOMP_RET_KILL_PROCESS),
      RET(SECCOMP_RET_ALLOW),
  };
  filter(f, sizeof f / sizeof f[0], 0);
  int80(64, 0, 0, 0, 0, 0, 0);
  return 1;
}

static int strict_write(void) {
  int p[2];
  pipe(p);
  prctl(PR_SET_SECCOMP, SECCOMP_MODE_STRICT);
  if (int80(4, p[1], (long)"x", 1, 0, 0, 0) != 1)
    int80(1, 3, 0, 0, 0, 0, 0);
  int80(20, 0, 0, 0, 0, 0, 0); /* getpid: killed */
  int80(1, 4, 0, 0, 0, 0, 0);
  return 5;
}

static int strict_exit(void) {
  prctl(PR_SET_SECCOMP, SECCOMP_MODE_STRICT);
  int80(1, 9, 0, 0, 0, 0, 0); /* exit(9) */
  return 1;
}

static int strict_rt_sigreturn(void) {
  prctl(PR_SET_SECCOMP, SECCOMP_MODE_STRICT);
  int80(173, 0, 0, 0, 0, 0, 0); /* only the old sigreturn is allowed */
  return 1;
}

static int traced_filter(void) {
  pid_t c = fork();
  if (c == 0) {
    ptrace(PTRACE_TRACEME, 0, 0, 0);
    raise(SIGSTOP);
    struct sock_filter f[] = {
        LD(NR),
        JEQ(20, 0, 1),
        RET(SECCOMP_RET_TRACE | 42),
        RET(SECCOMP_RET_ALLOW),
    };
    filter(f, sizeof f / sizeof f[0], 0);
    _exit(int80(20, 0x5150, 0, 0, 0, 0, 0) == getppid() ? 0 : 1);
  }
  int bad = 0, st;
  waitpid(c, &st, 0);
  ptrace(PTRACE_SETOPTIONS, c, 0, PTRACE_O_TRACESECCOMP);
  ptrace(PTRACE_CONT, c, 0, 0);
  waitpid(c, &st, 0);
  if (!WIFSTOPPED(st) || st >> 8 != (SIGTRAP | PTRACE_EVENT_SECCOMP << 8))
    return 64;
  unsigned long msg = 0;
  ptrace(PTRACE_GETEVENTMSG, c, 0, &msg);
  struct user_regs_struct r;
  ptrace(PTRACE_GETREGS, c, 0, &r);
  struct __ptrace_syscall_info info;
  memset(&info, 0, sizeof info);
  ptrace(PTRACE_GET_SYSCALL_INFO, c, sizeof info, &info);
  if (msg != 42 || r.orig_eax != 20 || r.ebx != 0x5150)
    bad |= 1;
  if (info.op != PTRACE_SYSCALL_INFO_SECCOMP || info.arch != AUDIT_ARCH_I386 ||
      info.seccomp.nr != 20 || info.seccomp.args[0] != 0x5150 ||
      info.seccomp.ret_data != 42)
    bad |= 2;
  r.orig_eax = SYS_getppid;
  ptrace(PTRACE_SETREGS, c, 0, &r);
  ptrace(PTRACE_CONT, c, 0, 0);
  waitpid(c, &st, 0);
  if (!WIFEXITED(st) || WEXITSTATUS(st) != 0)
    bad |= 4;
  return bad;
}

static int notified(void) {
  struct sock_filter f[] = {
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
    _exit(int80(20, 0xabc, 0xdef, 0, 0, 0, 0) == 999 ? 0 : 1);
  struct seccomp_notif req;
  memset(&req, 0, sizeof req);
  int bad = 0;
  if (ioctl(fd, SECCOMP_IOCTL_NOTIF_RECV, &req) != 0)
    return 64;
  if (req.data.arch != AUDIT_ARCH_I386 || req.data.nr != 20 ||
      req.data.args[0] != 0xabc || req.data.args[1] != 0xdef ||
      req.pid != (unsigned)c)
    bad |= 1;
  struct seccomp_notif_resp resp = {.id = req.id, .val = 999};
  if (ioctl(fd, SECCOMP_IOCTL_NOTIF_SEND, &resp) != 0)
    bad |= 2;
  int st;
  waitpid(c, &st, 0);
  if (!WIFEXITED(st) || WEXITSTATUS(st) != 0)
    bad |= 4;
  return bad;
}

static void seccomp_checks(void) {
  int st = in_child(filtered);
  CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "filtered i386 calls: %#x", st);
  st = in_child(killed);
  CHECK(WIFSIGNALED(st) && WTERMSIG(st) == SIGSYS, "KILL_PROCESS: %#x", st);
  st = in_child(strict_write);
  CHECK(WIFSIGNALED(st) && WTERMSIG(st) == SIGKILL, "strict getpid: %#x", st);
  st = in_child(strict_exit);
  CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 9, "strict exit: %#x", st);
  st = in_child(strict_rt_sigreturn);
  CHECK(WIFSIGNALED(st) && WTERMSIG(st) == SIGKILL, "strict rt_sigreturn: %#x",
        st);
  st = in_child(traced_filter);
  CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "RET_TRACE: %#x", st);
  st = in_child(notified);
  CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "USER_NOTIF: %#x", st);
}

/* Traced by ptrace_ia32, a 64-bit tracer: getpid with these registers,
 * which it turns into getppid. */
static int traced_by_64(void) {
  long r = int80(20, 0x1111, 0x2222, 0x3333, 0x4444, 0x5555, 0x6666);
  return r == getppid() ? 0 : 1;
}

int main(int argc, char **argv) {
  setvbuf(stdout, 0, _IONBF, 0);
  if (argc > 1 && strcmp(argv[1], "tracee") == 0)
    return traced_by_64();
  tracer();
  steps();
  seccomp_checks();
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
