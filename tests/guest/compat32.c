/* 32-bit code in a 64-bit process, as Wine's WoW64 runs it:
 *  - The selectors are Linux's: %cs = __USER_CS (0x33), %ss = __USER_DS
 *    (0x2b), %ds/%es/%fs/%gs null; __USER32_CS (0x23) is the 32-bit code
 *    segment.
 *  - int $0x80 is the i386 system call, from 64-bit code too: getpid is 20,
 *    and set_thread_area (243) and get_thread_area (244) manage the three GDT
 *    TLS entries (12-14), which a thread and a fork child inherit. The x86-64
 *    numbers for them (205, 211) are ENOSYS. A segment register holding a
 *    TLS selector is reloaded when its entry changes.
 *  - lretq, iretq and a far call reach 0x23; ljmp and lret come back to 0x33.
 *    32-bit code runs as 32-bit code (0x40 is inc %eax, push is 4 bytes) and
 *    can make int $0x80 calls.
 *  - A fault in 32-bit code reaches a 64-bit handler with CS 0x23 in its
 *    ucontext, and the sigreturn goes back to 32-bit code where the handler
 *    said; so does a signal from another thread while 32-bit code runs.
 * Non-PIE (gcc -static), so the code and data are below 4 GiB.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <ucontext.h>
#include <unistd.h>

static int failures;
#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("FAIL " __VA_ARGS__);                                             \
      printf("\n");                                                            \
      failures++;                                                              \
    }                                                                          \
  } while (0)

#if defined(__x86_64__)

/* struct user_desc (asm/ldt.h). */
struct user_desc {
  unsigned int entry_number;
  unsigned int base_addr;
  unsigned int limit;
  unsigned int seg_32bit : 1;
  unsigned int contents : 2;
  unsigned int read_exec_only : 1;
  unsigned int limit_in_pages : 1;
  unsigned int seg_not_present : 1;
  unsigned int useable : 1;
  unsigned int lm : 1;
};

/* An i386 system call from 64-bit code (the struct must be below 4 GiB). */
static long int80(long nr, long a, long b, long c) {
  long ret;
  __asm__ volatile("int $0x80"
                   : "=a"(ret)
                   : "a"(nr), "b"(a), "c"(b), "d"(c)
                   : "memory");
  return ret;
}

static unsigned short sel(int which) {
  unsigned short s = 0xffff;
  switch (which) {
  case 0: __asm__ volatile("mov %%cs,%0" : "=r"(s)); break;
  case 1: __asm__ volatile("mov %%ss,%0" : "=r"(s)); break;
  case 2: __asm__ volatile("mov %%ds,%0" : "=r"(s)); break;
  case 3: __asm__ volatile("mov %%es,%0" : "=r"(s)); break;
  case 4: __asm__ volatile("mov %%fs,%0" : "=r"(s)); break;
  case 5: __asm__ volatile("mov %%gs,%0" : "=r"(s)); break;
  }
  return s;
}

static uint32_t blob[4] = {0x11111111, 0x22222222, 0x33333333, 0x44444444};
static struct user_desc desc;

static int set_tls(int entry, void *base) {
  memset(&desc, 0, sizeof desc);
  desc.entry_number = entry;
  desc.base_addr = (uint32_t)(uintptr_t)base;
  desc.limit = 0xfff;
  desc.seg_32bit = 1;
  return (int)int80(243, (long)&desc, 0, 0);
}

static uint32_t gs_read(int off) {
  uint32_t v;
  __asm__ volatile("movl %%gs:(%1),%0" : "=r"(v) : "r"((long)off));
  return v;
}

static void load_gs(unsigned short s) { __asm__ volatile("mov %0,%%gs" ::"r"(s)); }

static int tls_entry;
static void *thread_tls(void *arg) {
  (void)arg;
  struct user_desc *d = &desc;
  memset(d, 0, sizeof *d);
  d->entry_number = tls_entry;
  long r = int80(244, (long)d, 0, 0);
  return (void *)(uintptr_t)(r == 0 && d->base_addr == (uint32_t)(uintptr_t)blob);
}

static void tls_checks(void) {
  CHECK(int80(20, 0, 0, 0) == getpid(), "int $0x80 getpid");
  CHECK(syscall(205, &desc) == -1 && errno == ENOSYS, "x86-64 set_thread_area is ENOSYS");
  CHECK(syscall(211, &desc) == -1 && errno == ENOSYS, "x86-64 get_thread_area is ENOSYS");

  CHECK(set_tls(-1, blob) == 0, "set_thread_area(-1)");
  tls_entry = desc.entry_number;
  CHECK(tls_entry >= 12 && tls_entry <= 14, "allocated entry %d", tls_entry);
  memset(&desc, 0, sizeof desc);
  desc.entry_number = tls_entry;
  CHECK(int80(244, (long)&desc, 0, 0) == 0, "get_thread_area");
  CHECK(desc.base_addr == (uint32_t)(uintptr_t)blob && desc.limit == 0xfff &&
            desc.seg_32bit && !desc.contents && !desc.read_exec_only &&
            !desc.seg_not_present,
        "get_thread_area reads back base %#x limit %#x", desc.base_addr, desc.limit);
  desc.entry_number = 15;
  CHECK(int80(244, (long)&desc, 0, 0) == -EINVAL, "get_thread_area(15) is EINVAL");
  memset(&desc, 0, sizeof desc);
  desc.entry_number = tls_entry;
  desc.seg_32bit = 1;
  desc.contents = 2; /* code */
  CHECK(int80(243, (long)&desc, 0, 0) == -EINVAL, "a code segment is EINVAL");

  /* Load %gs with the selector: its base is the descriptor's. */
  unsigned short s = (unsigned short)(tls_entry << 3 | 3);
  load_gs(s);
  CHECK(sel(5) == s, "%%gs is %#x", sel(5));
  CHECK(gs_read(4) == 0x22222222, "%%gs:4 is %#x", gs_read(4));
  /* Moving the entry reloads %gs. */
  CHECK(set_tls(tls_entry, &blob[2]) == 0, "set_thread_area again");
  CHECK(gs_read(4) == 0x44444444, "%%gs follows the entry: %#x", gs_read(4));
  CHECK(set_tls(tls_entry, blob) == 0, "set_thread_area back");
  load_gs(0);

  pthread_t t;
  void *ok = 0;
  pthread_create(&t, 0, thread_tls, 0);
  pthread_join(t, &ok);
  CHECK(ok, "a new thread has the TLS entry");
  pid_t pid = fork();
  if (pid == 0) {
    memset(&desc, 0, sizeof desc);
    desc.entry_number = tls_entry;
    _exit(int80(244, (long)&desc, 0, 0) == 0 &&
                  desc.base_addr == (uint32_t)(uintptr_t)blob
              ? 0
              : 1);
  }
  int st = -1;
  waitpid(pid, &st, 0);
  CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 0, "a fork child has the TLS entry");
}

/* ---- 32-bit code ---------------------------------------------------------
 * Entered with %eax = an argument, %ds/%es = 0x2b, %esp = the low stack.
 * Returns (to 64-bit) with %eax = the result. */
uint64_t saved_rsp;
uint32_t stack32[4096] __attribute__((aligned(16)));
uint32_t cs32, esp_step, pid32, fault_resumed, async_target;
volatile uint32_t spins32;
struct {
  uint32_t off;
  uint16_t sel;
} __attribute__((packed)) far32;

/* Common entry/exit, around the 32-bit body at `label`. */
#define TO32_PROLOGUE                                                          \
  "push %rbx\n\tpush %rbp\n\tpush %r12\n\tpush %r13\n\tpush %r14\n\tpush %r15\n\t" \
  "mov %rsp,saved_rsp(%rip)\n\t"                                               \
  "mov $0x2b,%ecx\n\tmov %ecx,%ds\n\tmov %ecx,%es\n\t"                         \
  "lea stack32+16384(%rip),%rsp\n\t"                                           \
  "mov %edi,%eax\n\t"
#define BACK64                                                                 \
  "mov saved_rsp(%rip),%rsp\n\t"                                               \
  "xor %ecx,%ecx\n\tmov %ecx,%ds\n\tmov %ecx,%es\n\t"                          \
  "pop %r15\n\tpop %r14\n\tpop %r13\n\tpop %r12\n\tpop %rbp\n\tpop %rbx\n\t"  \
  "ret\n\t"

uint32_t via_lretq(uint32_t arg);
uint32_t via_iretq(uint32_t arg);
uint32_t via_lcall(uint32_t arg);
uint32_t fault32(uint32_t arg);
uint32_t spin32(uint32_t arg);

__asm__(".text\n"
        /* lretq to 0x23; the 32-bit body ljmps back to 0x33. */
        ".globl via_lretq\nvia_lretq:\n\t" TO32_PROLOGUE
        "pushq $0x23\n\tpushq $body32\n\tlretq\n"
        /* iretq to 0x23 (a 5-quadword frame), ljmp back. */
        ".globl via_iretq\nvia_iretq:\n\t" TO32_PROLOGUE
        "mov %rsp,%rdx\n\t"
        "pushq $0x2b\n\tpushq %rdx\n\tpushfq\n\tpushq $0x23\n\tpushq $body32\n\tiretq\n"
        /* lcall (m16:32) to 0x23; the 32-bit side lrets. */
        ".globl via_lcall\nvia_lcall:\n\t" TO32_PROLOGUE
        "movl $call32,far32(%rip)\n\tmovw $0x23,far32+4(%rip)\n\t"
        "lcall *far32(%rip)\n\t" BACK64
        /* A fault in 32-bit code. */
        ".globl fault32\nfault32:\n\t" TO32_PROLOGUE
        "pushq $0x23\n\tpushq $fault_body32\n\tlretq\n"
        /* A 32-bit loop that waits for a signal from another thread. */
        ".globl spin32\nspin32:\n\t" TO32_PROLOGUE
        "pushq $0x23\n\tpushq $spin_body32\n\tlretq\n"

        ".code32\n"
        "body32:\n\t"
        "mov %cs,%ecx\n\tmov %ecx,cs32\n\t"
        ".byte 0x40\n\t" /* inc %eax: a REX prefix in 64-bit code */
        "mov %esp,%ebx\n\tpush %eax\n\tsub %esp,%ebx\n\tmov %ebx,esp_step\n\tpop %edx\n\t"
        "imul $3,%eax,%eax\n\t"
        "push %eax\n\t"
        "mov $20,%eax\n\tint $0x80\n\tmov %eax,pid32\n\t" /* i386 getpid */
        "pop %eax\n\t"
        "ljmp $0x33,$back64\n"
        "call32:\n\t"
        "mov %cs,%ecx\n\tmov %ecx,cs32\n\t"
        "add $5,%eax\n\t"
        "lret\n"
        "fault_body32:\n\t"
        "movl $0,fault_resumed\n\t"
        "xor %ecx,%ecx\n\t"
        ".globl fault_insn32\nfault_insn32:\n\t"
        "movl (%ecx),%ecx\n\t" /* faults: the handler skips it (2 bytes) */
        "movl $1,fault_resumed\n\t"
        "ljmp $0x33,$back64\n"
        "spin_body32:\n\t"
        "1:\tincl spins32\n\t"
        "cmpl %eax,async_target\n\t"
        "ja 1b\n\t"
        ".globl spin_end32\nspin_end32:\n\t"
        "mov spins32,%eax\n\t"
        "ljmp $0x33,$back64\n"
        ".code64\n"
        "back64:\n\t" BACK64);

extern char fault_insn32[], spin_body32[], spin_end32[];

static volatile int seg_hits, seg_cs, handler_cs;
static void on_segv(int sig, siginfo_t *si, void *ctx) {
  (void)sig;
  (void)si;
  ucontext_t *uc = ctx;
  seg_hits++;
  seg_cs = uc->uc_mcontext.gregs[REG_CSGSFS] & 0xffff;
  handler_cs = sel(0);
  if (uc->uc_mcontext.gregs[REG_RIP] == (uintptr_t)fault_insn32)
    uc->uc_mcontext.gregs[REG_RIP] += 2;
}

static volatile int usr_hits, usr_in32;
static void on_usr1(int sig, siginfo_t *si, void *ctx) {
  (void)sig;
  (void)si;
  ucontext_t *uc = ctx;
  uint64_t rip = uc->uc_mcontext.gregs[REG_RIP];
  usr_hits++;
  if ((uc->uc_mcontext.gregs[REG_CSGSFS] & 0xffff) == 0x23 &&
      rip >= (uintptr_t)spin_body32 && rip <= (uintptr_t)spin_end32) {
    usr_in32++;
    async_target = 0; /* let the loop finish */
  }
}

static volatile int spinning;
static pthread_t spinner;
static void *signaller(void *arg) {
  (void)arg;
  while (!spinning)
    usleep(100);
  for (int i = 0; i < 2000 && !usr_in32; i++) {
    pthread_kill(spinner, SIGUSR1);
    usleep(1000);
  }
  async_target = 0; /* give up */
  return 0;
}

/* More threads with a TLS entry in use, all alive at once, than a GDT a
 * thread would allow (the KVM backend once kept 512): each has its own
 * %gs base. */
#define MANY_TLS 520
static uint32_t many_words[MANY_TLS];
static int many_ready;
static void *many_tls(void *arg) {
  int i = (int)(uintptr_t)arg;
  /* Its own user_desc, below 4 GiB (a static array: non-PIE). */
  static struct user_desc descs[MANY_TLS];
  struct user_desc *d = &descs[i];
  memset(d, 0, sizeof *d);
  d->entry_number = -1;
  d->base_addr = (uint32_t)(uintptr_t)&many_words[i];
  d->limit = 3;
  d->seg_32bit = 1;
  long r = int80(243, (long)d, 0, 0);
  /* All alive at once (polled: a 520-thread barrier is a futex storm). */
  __atomic_add_fetch(&many_ready, 1, __ATOMIC_SEQ_CST);
  struct timespec ms = {0, 20000000};
  while (__atomic_load_n(&many_ready, __ATOMIC_SEQ_CST) < MANY_TLS)
    nanosleep(&ms, NULL);
  if (r != 0)
    return (void *)1;
  load_gs((unsigned short)(d->entry_number << 3 | 3));
  sched_yield();
  uint32_t v = gs_read(0);
  load_gs(0);
  return (void *)(uintptr_t)(v == many_words[i] ? 0 : 2);
}

static void many_tls_checks(void) {
  pthread_t t[MANY_TLS];
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, 64 * 1024);
  int made = 0;
  for (int i = 0; i < MANY_TLS; i++) {
    many_words[i] = 0x5000 + i;
    if (pthread_create(&t[i], &attr, many_tls, (void *)(uintptr_t)i) != 0)
      break;
    made++;
  }
  CHECK(made == MANY_TLS, "made %d threads", made);
  if (made != MANY_TLS) {
    /* The others would wait forever: let them go. */
    __atomic_add_fetch(&many_ready, MANY_TLS, __ATOMIC_SEQ_CST);
    for (int i = 0; i < made; i++)
      pthread_join(t[i], NULL);
    return;
  }
  int bad = 0;
  for (int i = 0; i < made; i++) {
    void *r;
    pthread_join(t[i], &r);
    bad += r != NULL;
  }
  CHECK(bad == 0, "%d of %d threads lost their TLS entry", bad, made);
}

/* modify_ldt from 64-bit code: an LDT data segment for %gs. */
static uint32_t ldt_words[2] = {0xaaaa5555, 0x12345678};
static void ldt_checks(void) {
  struct user_desc d;
  memset(&d, 0, sizeof d);
  d.entry_number = 3;
  d.base_addr = (uint32_t)(uintptr_t)ldt_words;
  d.limit = 7;
  d.seg_32bit = 1;
  CHECK(syscall(SYS_modify_ldt, 1, &d, sizeof d) == 0, "modify_ldt write");
  unsigned short s = 3 << 3 | 4 | 3;
  load_gs(s);
  CHECK(gs_read(4) == 0x12345678, "LDT %%gs:4 is %#x", gs_read(4));
  load_gs(0);
  /* Its result is an int in a zero-extended register, errors too. */
  CHECK(syscall(SYS_modify_ldt, 7, &d, 0) == 0xffffffdaL, "modify_ldt(7) is (unsigned)-ENOSYS");
  static unsigned char buf[32]; /* below 4 GiB, for int $0x80 */
  CHECK(syscall(SYS_modify_ldt, 0, buf, sizeof buf) == sizeof buf, "read_ldt");
  /* The same through int $0x80. */
  CHECK(int80(123, 0, (long)buf, sizeof buf) == sizeof buf, "int $0x80 modify_ldt");
}

/* An i386 mmap2 from a 64-bit task goes below 4 GiB (in_32bit_syscall). */
static void mmap32_checks(void) {
  long r;
  __asm__ volatile("push %%rbp\n\t"
                   "xor %%ebp, %%ebp\n\t"
                   "int $0x80\n\t"
                   "pop %%rbp"
                   : "=a"(r)
                   : "a"(192L), "b"(0L), "c"(1L << 20), "d"(3L), "S"(0x22L),
                     "D"(-1L)
                   : "memory");
  CHECK(r > 0 && (unsigned long)r < 0x100000000ul && r % 4096 == 0,
        "int $0x80 mmap2 from 64-bit code: %#lx", r);
  if (r > 0 && (unsigned long)r < 0x100000000ul) {
    *(volatile char *)r = 1;
    munmap((void *)r, 1 << 20);
  }
}


static void mode_checks(void) {
  cs32 = 0;
  uint32_t r = via_lretq(6);
  CHECK(r == 21, "lretq/ljmp: 32-bit code computed %u, not 21", r);
  CHECK(cs32 == 0x23, "%%cs in 32-bit code is %#x", cs32);
  CHECK(esp_step == 4, "a 32-bit push moves %%esp by %u", esp_step);
  CHECK(pid32 == (uint32_t)getpid(), "int $0x80 from 32-bit code: %u", pid32);
  CHECK(sel(0) == 0x33 && sel(2) == 0, "back in 64-bit mode: cs %#x ds %#x", sel(0), sel(2));

  cs32 = 0;
  r = via_iretq(1);
  CHECK(r == 6 && cs32 == 0x23, "iretq to 32-bit code: %u, cs %#x", r, cs32);
  cs32 = 0;
  r = via_lcall(1);
  CHECK(r == 6 && cs32 == 0x23, "lcall/lret: %u, cs %#x", r, cs32);

  struct sigaction sa = {0};
  sa.sa_sigaction = on_segv;
  sa.sa_flags = SA_SIGINFO;
  sigaction(SIGSEGV, &sa, 0);
  r = fault32(0);
  CHECK(seg_hits == 1, "SIGSEGV from 32-bit code: %d", seg_hits);
  CHECK(seg_cs == 0x23, "the ucontext has CS %#x", seg_cs);
  CHECK(handler_cs == 0x33, "the handler runs in 64-bit mode: %#x", handler_cs);
  CHECK(fault_resumed == 1, "sigreturn resumed the 32-bit code after the fault");
  signal(SIGSEGV, SIG_DFL);

  sa.sa_sigaction = on_usr1;
  sigaction(SIGUSR1, &sa, 0);
  async_target = 0xffffffff;
  spinner = pthread_self();
  pthread_t t;
  pthread_create(&t, 0, signaller, 0);
  spinning = 1;
  r = spin32(1);
  pthread_join(t, 0);
  CHECK(usr_in32 >= 1, "a signal interrupted 32-bit code (%d of %d)", usr_in32, usr_hits);
  CHECK(r == spins32 && r > 0, "the 32-bit loop resumed and finished: %u", r);
}

/* An i386 program exec'd from this 64-bit one, when it's next to it (the
 * guest tests build `ia32` with `gcc -m32`): it runs as an ELFCLASS32
 * process, with its arguments and environment. */
static void exec_ia32(void) {
  char self[512], path[600];
  ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
  if (n <= 0)
    return;
  self[n] = 0;
  char *slash = strrchr(self, '/');
  if (!slash)
    return;
  *slash = 0;
  snprintf(path, sizeof path, "%s/ia32", self);
  if (access(path, X_OK) != 0)
    return;
  pid_t c = fork();
  if (c == 0) {
    char *argv[] = {path, "exec-check", "second", NULL};
    char *envp[] = {"IA32_ENV=yes", NULL};
    execve(path, argv, envp);
    _exit(99);
  }
  int status = 0;
  CHECK(waitpid(c, &status, 0) == c && WIFEXITED(status) &&
            WEXITSTATUS(status) == 77,
        "exec of the i386 program: status %#x", status);
}

int main(int argc, char **argv) {
  setvbuf(stdout, 0, _IONBF, 0);
  /* `compat32 many`: only the many-threads check (slow on a debug VM). */
  if (argc > 1 && strcmp(argv[1], "many") == 0) {
    many_tls_checks();
    if (!failures)
      printf("PASS\n");
    return failures != 0;
  }
  CHECK(sel(0) == 0x33, "%%cs is %#x", sel(0));
  CHECK(sel(1) == 0x2b, "%%ss is %#x", sel(1));
  CHECK(sel(2) == 0 && sel(3) == 0 && sel(4) == 0 && sel(5) == 0,
        "%%ds %#x %%es %#x %%fs %#x %%gs %#x", sel(2), sel(3), sel(4), sel(5));
  CHECK((uintptr_t)&main < 0x80000000u && (uintptr_t)stack32 < 0x80000000u,
        "built as non-PIE, below 2 GiB");
  tls_checks();
  mode_checks();
  ldt_checks();
  mmap32_checks();
  exec_ia32();
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
