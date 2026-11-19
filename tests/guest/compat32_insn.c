/* The instructions only 32-bit code has, run in compatibility mode from a
 * 64-bit process (a far call to __USER32_CS, 0x23, as compat32 does):
 *  - DAA/DAS/AAA/AAS for every AL and incoming CF/AF, and AAM/AAD (base 10
 *    and 7), against the Intel SDM's definitions (the result and the flags the
 *    SDM defines); AAM 0 is a divide error (SIGFPE). SALC.
 *  - BOUND within the bounds, and outside them (SIGSEGV).
 *  - ARPL (MOVSXD in 64-bit code).
 *  - LES/LDS/LGS loading __USER_DS (0x2b) and the offset.
 *  - Segment bases and limits: %ds/%es loaded with a set_thread_area
 *    descriptor (a base, a 4 KiB limit) are relative to its base; an access
 *    past the limit is a #GP (SIGSEGV).
 *  - 16-bit addressing (the 0x67 prefix): [bx+si], [bx+disp] wrapping at
 *    64 KiB, a store through [di], [bp+di] in %ss by default (here a fault
 *    near address 0) and in %ds with an override, rep movsb on CX/SI/DI, and
 *    loop on CX.
 *  - 16-bit code, as Wine's Win16 thunks run it: a far call from 32-bit code
 *    to a 16-bit LDT code segment (modify_ldt), whose default operand and
 *    address size are 16 bits (0x66 and 0x67 make them 32); a 16-bit LDT
 *    stack segment, where pushes, pops, near calls and far calls (out to
 *    32-bit code and back) move %sp only; loop and rep stos on CX/DI; a near
 *    jump past 0xffff wrapping %ip to the start of the segment; a fault
 *    (int $0x21) whose 64-bit handler resumes the 16-bit code; and a far
 *    return to 32-bit code.
 * Non-PIE (gcc -static), so the code and data are below 4 GiB.
 */
#define _GNU_SOURCE
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
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

/* ---- the switch to 32-bit code --------------------------------------------
 * call32(fn) far-calls the 32-bit function fn (which ends with lret) on a
 * low stack, with %ds/%es = 0x2b, and returns to 64-bit code with %ds/%es
 * null again. Inputs and outputs go through the globals below. */
uint64_t saved_rsp;
uint32_t stack32[4096] __attribute__((aligned(16)));
uint32_t in_ax, in_cx, in_fl, out_ax, out_bx, out_cx, out_dx, out_fl;
uint32_t tls_sel;
uint32_t r_1, r_2, r_cx, r_si, r_di, r_loops, r_cx2, r_bpds, r_lim;
struct {
  uint32_t off;
  uint16_t sel;
} __attribute__((packed)) far32, farp, far16;
int32_t bnds[2];

void call32(void *fn);

/* Load the flags and %eax, run `insn`, store %eax and the flags. */
#define FLAGS_BODY(name, insn)                                                 \
  ".globl " #name "\n" #name ":\n\t"                                           \
  "mov in_fl,%eax\n\tpush %eax\n\tpopf\n\tmov in_ax,%eax\n\t" insn "\n\t"      \
  "pushf\n\tpop %ecx\n\tmov %ecx,out_fl\n\tmov %eax,out_ax\n\tlret\n"

__asm__(".text\n"
        ".globl call32\ncall32:\n\t"
        "push %rbx\n\tpush %rbp\n\tpush %r12\n\tpush %r13\n\tpush %r14\n\tpush %r15\n\t"
        "mov %rsp,saved_rsp(%rip)\n\t"
        "mov $0x2b,%ecx\n\tmov %ecx,%ds\n\tmov %ecx,%es\n\t"
        "lea stack32+16384(%rip),%rsp\n\t"
        "movl %edi,far32(%rip)\n\tmovw $0x23,far32+4(%rip)\n\t"
        "lcall *far32(%rip)\n\t"
        "mov saved_rsp(%rip),%rsp\n\t"
        "xor %ecx,%ecx\n\tmov %ecx,%ds\n\tmov %ecx,%es\n\tmov %ecx,%gs\n\t"
        "pop %r15\n\tpop %r14\n\tpop %r13\n\tpop %r12\n\tpop %rbp\n\tpop %rbx\n\t"
        "ret\n"

        ".code32\n"
        FLAGS_BODY(f_daa, "daa")
        FLAGS_BODY(f_das, "das")
        FLAGS_BODY(f_aaa, "aaa")
        FLAGS_BODY(f_aas, "aas")
        FLAGS_BODY(f_aam10, "aam")
        FLAGS_BODY(f_aad10, "aad")
        FLAGS_BODY(f_aam7, "aam $7")
        FLAGS_BODY(f_aad7, "aad $7")
        FLAGS_BODY(f_salc, ".byte 0xd6")
        /* aam $0: #DE; the SIGFPE handler resumes past it. */
        FLAGS_BODY(f_aam0, ".globl aam0_insn\naam0_insn: .byte 0xd4, 0\n"
                           ".globl aam0_end\naam0_end:")
        /* bound %eax / %ax, with the bounds pair in bnds. */
        ".globl f_bound\nf_bound:\n\t"
        "movl $0,out_bx\n\tmov in_ax,%eax\n\t"
        ".globl bound_insn\nbound_insn: bound %eax,bnds\n"
        ".globl bound_end\nbound_end:\n\t"
        "movl $1,out_bx\n\tlret\n"
        /* arpl %ax,%cx */
        ".globl f_arpl\nf_arpl:\n\t"
        "mov in_fl,%eax\n\tpush %eax\n\tpopf\n\t"
        "mov in_ax,%eax\n\tmov in_cx,%ecx\n\tarpl %ax,%cx\n\t"
        "pushf\n\tpop %edx\n\tmov %edx,out_fl\n\tmov %ecx,out_cx\n\t"
        /* ...and to memory. */
        "mov in_ax,%eax\n\tarpl %ax,out_dx\n\tlret\n"
        /* les/lds/lgs from farp. */
        ".globl f_les\nf_les:\n\t"
        "xor %eax,%eax\n\tmov %eax,%es\n\tmov %eax,%gs\n\t"
        "les farp,%ebx\n\tmov %ebx,out_bx\n\tmov %es,%eax\n\tmov %eax,out_ax\n\t"
        "lgs farp,%edx\n\tmov %edx,out_dx\n\tmov %gs,%eax\n\tmov %eax,out_fl\n\t"
        "lds farp,%ecx\n\tmov %ecx,out_cx\n\tlret\n"
        /* %ds/%es on the TLS descriptor (base = buf), 16-bit addressing. */
        ".globl f_addr16\nf_addr16:\n\t"
        "push %ebp\n\t"
        "mov tls_sel,%eax\n\tmov %eax,%ds\n\tmov %eax,%es\n\t"
        "mov $8,%ebx\n\tmov $4,%esi\n\t"
        "addr16 mov (%bx,%si),%eax\n\tpush %eax\n\t"     /* buf+12 */
        "mov $0x10004,%ebx\n\t"
        ".byte 0x67,0x8b,0x87,0xfe,0xff\n\tpush %eax\n\t" /* [bx+0xfffe]: buf+2 */
        "mov $0x30,%edi\n\tmov $0xa5a5a5a5,%eax\n\t"
        "addr16 mov %eax,(%di)\n\t"                      /* buf+0x30 */
        "mov $0x55550010,%ecx\n\tmov $0x66660000,%esi\n\tmov $0x77770100,%edi\n\t"
        "cld\n\taddr16 rep movsb\n\t"                    /* buf -> buf+0x100 */
        "push %ecx\n\tpush %esi\n\tpush %edi\n\t"
        "mov $0x10003,%ecx\n\txor %eax,%eax\n"
        "1:\tinc %eax\n\taddr16 loop 1b\n\t"
        "push %eax\n\tpush %ecx\n\t"
        "mov $0x2b,%eax\n\tmov %eax,%ds\n\tmov %eax,%es\n\t"
        "popl r_cx2\n\tpopl r_loops\n\tpopl r_di\n\tpopl r_si\n\tpopl r_cx\n\t"
        "popl r_2\n\tpopl r_1\n\t"
        "pop %ebp\n\tlret\n"
        /* [bp+di] is in %ss (flat: near address 0, a fault), in %ds with ds:. */
        ".globl f_bpss\nf_bpss:\n\t"
        "push %ebp\n\t"
        "mov tls_sel,%eax\n\tmov %eax,%ds\n\t"
        "xor %edi,%edi\n\tmov $0x10,%ebp\n\txor %ecx,%ecx\n"
        ".globl bpss_insn\nbpss_insn: addr16 mov (%bp,%di),%eax\n"
        ".globl bpss_end\nbpss_end:\n\t"
        "addr16 mov %ds:(%bp,%di),%ecx\n\t"
        "mov $0x2b,%eax\n\tmov %eax,%ds\n\tmov %ecx,r_bpds\n\t"
        "pop %ebp\n\tlret\n"
        /* The last dword below the 4 KiB limit is in; one byte on is #GP. */
        ".globl f_limit\nf_limit:\n\t"
        "mov tls_sel,%eax\n\tmov %eax,%ds\n\t"
        "mov 0xffc,%edx\n"
        ".globl limit_insn\nlimit_insn: mov 0xffd,%ecx\n"
        ".globl limit_end\nlimit_end:\n\t"
        "mov $0x2b,%eax\n\tmov %eax,%ds\n\tmov %edx,r_lim\n\tlret\n"
        /* ---- 16-bit code: f_to16 far-calls c16 (copied to code16, the
         * base of LDT code segment 0, selector 7), whose data and stack
         * segment is LDT entry 1 (selector 0xf, based at area16). */
        ".globl f_to16\nf_to16:\n\t"
        "push %ebp\n\tpush %ebx\n\tpush %esi\n\tpush %edi\n\t"
        "lcall *far16\n\t"
        "pop %edi\n\tpop %esi\n\tpop %ebx\n\tpop %ebp\n\tlret\n"
        /* Called from 16-bit code, on the 16-bit stack. */
        ".globl c32_helper\nc32_helper:\n\t"
        "mov $0x77,%ecx\n\tpush %ecx\n\tpop %edx\n\tlret\n"
        ".code16\n"
        ".globl c16_start\nc16_start:\n"
        /* int $0x21, as Win16 code makes: a #GP (SIGSEGV) whose handler
         * resumes the 16-bit code past it. */
        ".globl c16_int21\nc16_int21: int $0x21\n\t"
        "mov $0xf,%ax\n\tmov %ax,%ds\n\tmov %ax,%es\n\t"
        "movl %esp,(0)\n\tmov %ss,(4)\n\t"
        "mov %ax,%ss\n\tmovl $0x55558000,%esp\n\t"
        /* The default operand size is 16: inc %ax leaves the top of %eax. */
        "movl $0xaaaaffff,%eax\n\tinc %ax\n\tmovl %eax,(0x10)\n\t"
        /* A 16-bit stack: pushes and pops move %sp. */
        "mov $0x1234,%ax\n\tpush %ax\n\tpushl $0x11223344\n\t"
        "movl %esp,(0x14)\n\tpopl %ebx\n\tpop %cx\n\t"
        "movl %ebx,(0x18)\n\tmov %cx,(0x1c)\n\tmovl %esp,(0x20)\n\t"
        /* 16-bit addressing by default, 32-bit under 0x67. */
        "mov $0x40,%bx\n\tmov $2,%si\n\tmovw $0xbeef,(%bx,%si)\n\t"
        "movl $0x48,%ebx\n\taddr32 movw $0xcafe,(%ebx)\n\t"
        /* A near call and return, on %ip and %sp. */
        "call 1f\n\tjmp 2f\n"
        "1:\tmovl %esp,(0x24)\n\tret\n"
        /* loop and rep stos count in %cx and move %di. */
        "2:\tmovl $0x10003,%ecx\n\txor %dx,%dx\n"
        "3:\tinc %dx\n\tloop 3b\n\t"
        "movl %ecx,(0x28)\n\tmov %dx,(0x2c)\n\t"
        "movl $0x12340004,%ecx\n\tmovl $0x56780050,%edi\n\tmov $0x5a,%al\n\t"
        "cld\n\trep stosb\n\tmovl %ecx,(0x30)\n\tmovl %edi,(0x34)\n\t"
        /* Out to 32-bit code and back, on the 16-bit stack. */
        "lcalll $0x23,$c32_helper\n\t"
        "movl %ecx,(0x38)\n\tmovl %edx,(0x44)\n\tmovl %esp,(0x3c)\n\t"
        /* To 0xfff0, whose jmp (written there by wrap_stub) wraps to c16_back. */
        ".byte 0xe9\n\t.word 0xfff0 - (. + 2 - c16_start)\n"
        ".globl c16_back\nc16_back:\n\t"
        "movw $0x600d,(0x60)\n\t"
        "mov (4),%ss\n\tmovl (0),%esp\n\t"
        "mov $0x2b,%ax\n\tmov %ax,%ds\n\tmov %ax,%es\n\t"
        "lretl\n"
        ".globl c16_end\nc16_end:\n"
        ".code64\n");


extern char f_daa[], f_das[], f_aaa[], f_aas[], f_aam10[], f_aad10[], f_aam7[],
    f_aad7[], f_salc[], f_aam0[], f_bound[], f_arpl[], f_les[], f_addr16[],
    f_bpss[], f_limit[];
extern char aam0_insn[], aam0_end[], bound_insn[], bound_end[], bpss_insn[],
    bpss_end[], limit_insn[], limit_end[];
extern char f_to16[], c16_start[], c16_int21[], c16_back[], c16_end[];
static volatile int int21_hits;

/* ---- faults: resume past the instruction, counting --------------------- */
static struct {
  char *insn, *end;
  int sig;
  volatile int hits;
} resume[] = {
    {aam0_insn, aam0_end, SIGFPE, 0},
    {bound_insn, bound_end, SIGSEGV, 0},
    {bpss_insn, bpss_end, SIGSEGV, 0},
    {limit_insn, limit_end, SIGSEGV, 0},
};
enum { R_AAM0, R_BOUND, R_BPSS, R_LIMIT };

static void on_fault(int sig, siginfo_t *si, void *ctx) {
  (void)si;
  ucontext_t *uc = ctx;
  greg_t *rip = &uc->uc_mcontext.gregs[REG_RIP];
  /* int $0x21 in the 16-bit code segment (LDT entry 0): %ip is the offset. */
  if (sig == SIGSEGV && (uc->uc_mcontext.gregs[REG_CSGSFS] & 0xffff) == 7 &&
      *rip == c16_int21 - c16_start) {
    int21_hits++;
    *rip += 2;
    return;
  }
  for (unsigned i = 0; i < sizeof resume / sizeof resume[0]; i++)
    if (resume[i].sig == sig && *rip == (greg_t)(uintptr_t)resume[i].insn &&
        (uc->uc_mcontext.gregs[REG_CSGSFS] & 0xffff) == 0x23) {
      resume[i].hits++;
      *rip = (greg_t)(uintptr_t)resume[i].end;
      return;
    }
  /* Anything else is a real crash. */
  signal(sig, SIG_DFL);
}

/* ---- reference results (Intel SDM) -------------------------------------- */
enum { CF = 1, PF = 4, AF = 0x10, ZF = 0x40, SF = 0x80 };

static uint32_t szp(uint8_t v) {
  return (v == 0 ? ZF : 0) | (v & 0x80 ? SF : 0) |
         (__builtin_parity(v) ? 0 : PF);
}

/* DAA/DAS: AL and CF, AF, SF, ZF, PF. */
static void ref_daa(int sub, uint32_t ax, int cf, int af, uint32_t *rax,
                    uint32_t *rfl) {
  uint8_t al = ax, old = al;
  uint32_t f = 0;
  if ((al & 0xf) > 9 || af) {
    if (sub && (al < 6 || cf))
      f |= CF;
    al = sub ? al - 6 : al + 6;
    f |= AF;
  }
  if (old > 0x99 || cf) {
    al = sub ? al - 0x60 : al + 0x60;
    f |= CF;
  }
  *rax = (ax & 0xff00) | al;
  *rfl = f | szp(al);
}

/* AAA/AAS: AX and CF, AF. */
static void ref_aaa(int sub, uint32_t ax, int af, uint32_t *rax,
                    uint32_t *rfl) {
  uint32_t f = 0;
  if ((ax & 0xf) > 9 || af) {
    ax = sub ? ax - 6 - 0x100 : ax + 0x106;
    f = AF | CF;
  }
  *rax = ax & 0xff0f;
  *rfl = f;
}

static uint32_t run_flags(char *fn, uint32_t ax, uint32_t fl) {
  in_ax = ax;
  in_fl = 0x202 | fl;
  call32(fn);
  return out_fl;
}

static void bcd_checks(void) {
  int bad = 0;
  static const uint8_t ahs[] = {0x00, 0x12, 0xff};
  for (int k = 0; k < 4; k++) {
    char *fn = (char *[]){f_daa, f_das, f_aaa, f_aas}[k];
    const char *name = (const char *[]){"daa", "das", "aaa", "aas"}[k];
    for (unsigned h = 0; h < sizeof ahs; h++)
      for (uint32_t al = 0; al < 256; al++)
        for (int cfaf = 0; cfaf < 4; cfaf++) {
          uint32_t ax = (uint32_t)ahs[h] << 8 | al, wax, wfl, mask;
          int cf = cfaf & 1, af = cfaf >> 1;
          /* The other flags start set, to see the defined ones change. */
          uint32_t fl = run_flags(fn, 0xabcd0000 | ax,
                                  (cf ? CF : 0) | (af ? AF : 0) | PF | ZF | SF);
          if (k < 2) {
            ref_daa(k, ax, cf, af, &wax, &wfl);
            mask = CF | AF | SF | ZF | PF;
          } else {
            ref_aaa(k == 3, ax, af, &wax, &wfl);
            mask = CF | AF;
          }
          if (out_ax != (0xabcd0000 | wax) || (fl & mask) != wfl) {
            if (bad++ < 10)
              printf("FAIL %s ax %#x cf %d af %d: ax %#x flags %#x, want "
                     "ax %#x flags %#x\n",
                     name, ax, cf, af, out_ax, fl & mask, 0xabcd0000 | wax, wfl);
          }
        }
  }
  /* AAM/AAD in base 10 and 7: SF, ZF, PF defined. */
  for (int k = 0; k < 4; k++) {
    char *fn = (char *[]){f_aam10, f_aad10, f_aam7, f_aad7}[k];
    unsigned base = k < 2 ? 10 : 7;
    int aad = k & 1;
    for (uint32_t ax = 0; ax < 0x10000; ax += 7) {
      uint32_t fl = run_flags(fn, ax, 0);
      uint8_t al = ax, ah = ax >> 8;
      uint32_t want = aad ? (uint8_t)(al + ah * base)
                          : (uint32_t)(al / base) << 8 | al % base;
      if ((out_ax & 0xffff) != want || (fl & (SF | ZF | PF)) != szp(want)) {
        if (bad++ < 10)
          printf("FAIL %s %u ax %#x: %#x flags %#x\n", aad ? "aad" : "aam",
                 base, ax, out_ax, fl);
      }
    }
  }
  CHECK(!bad, "%d BCD results differ", bad);

  run_flags(f_aam0, 0x1234, 0);
  CHECK(resume[R_AAM0].hits == 1, "aam $0 raised SIGFPE %d times",
        resume[R_AAM0].hits);
  CHECK(out_ax == 0x1234, "aam $0 left ax %#x", out_ax);

  run_flags(f_salc, 0x11223344, CF);
  CHECK(out_ax == 0x112233ff && (out_fl & CF), "salc with CF: %#x", out_ax);
  run_flags(f_salc, 0x11223344, 0);
  CHECK(out_ax == 0x11223300 && !(out_fl & CF), "salc without CF: %#x", out_ax);
}

static void misc_checks(void) {
  bnds[0] = -5;
  bnds[1] = 100;
  static const struct {
    int32_t v;
    int in;
  } b[] = {{0, 1}, {-5, 1}, {100, 1}, {-6, 0}, {101, 0}, {-2147483647 - 1, 0}};
  for (unsigned i = 0; i < sizeof b / sizeof b[0]; i++) {
    int before = resume[R_BOUND].hits;
    in_ax = (uint32_t)b[i].v;
    call32(f_bound);
    int faulted = resume[R_BOUND].hits - before;
    CHECK(faulted == !b[i].in && out_bx == 1, "bound %d in -5..100: %s",
          b[i].v, faulted ? "SIGSEGV" : "no fault");
  }

  in_ax = 0x2b;
  in_cx = 0xabcd0028;
  in_fl = 0x202;
  out_dx = 0x77770029;
  call32(f_arpl);
  CHECK(out_cx == 0xabcd002b && (out_fl & ZF), "arpl raised %#x, ZF %d",
        out_cx, !!(out_fl & ZF));
  CHECK(out_dx == 0x7777002b, "arpl to memory: %#x", out_dx);
  in_ax = 0x28;
  in_cx = 0x2b;
  in_fl = 0x202 | ZF;
  call32(f_arpl);
  CHECK(out_cx == 0x2b && !(out_fl & ZF), "arpl kept %#x, ZF %d", out_cx,
        !!(out_fl & ZF));

  farp.off = 0x12345678;
  farp.sel = 0x2b;
  out_ax = out_bx = out_cx = out_dx = out_fl = 0;
  call32(f_les);
  CHECK(out_bx == 0x12345678 && (out_ax & 0xffff) == 0x2b, "les: %#x, es %#x",
        out_bx, out_ax);
  CHECK(out_dx == 0x12345678 && (out_fl & 0xffff) == 0x2b, "lgs: %#x, gs %#x",
        out_dx, out_fl);
  CHECK(out_cx == 0x12345678, "lds: %#x", out_cx);
}

/* A TLS (GDT) descriptor based at buf, 4 KiB, for %ds and %es. */
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
static struct user_desc desc;
static uint8_t buf[4096] __attribute__((aligned(4096)));

static long int80(long nr, long a) {
  long ret;
  __asm__ volatile("int $0x80" : "=a"(ret) : "a"(nr), "b"(a) : "memory");
  return ret;
}

static void segment_checks(void) {
  memset(&desc, 0, sizeof desc);
  desc.entry_number = -1;
  desc.base_addr = (uint32_t)(uintptr_t)buf;
  desc.limit = 0xfff;
  desc.seg_32bit = 1;
  long r = int80(243, (long)&desc); /* i386 set_thread_area */
  CHECK(r == 0, "set_thread_area: %ld", r);
  if (r)
    return;
  tls_sel = desc.entry_number << 3 | 3;
  for (int i = 0; i < 4096; i++)
    buf[i] = (uint8_t)(i * 7 + 1);

  call32(f_addr16);
  uint32_t w12, w2, w30, w0, w100, wffc;
  memcpy(&w12, buf + 12, 4);
  memcpy(&w2, buf + 2, 4);
  memcpy(&w30, buf + 0x30, 4);
  CHECK(r_1 == w12, "[bx+si] in %%ds: %#x, want %#x", r_1, w12);
  CHECK(r_2 == w2, "[bx+0xfffe] wraps to 2: %#x, want %#x", r_2, w2);
  CHECK(w30 == 0xa5a5a5a5, "store through [di]: %#x", w30);
  memcpy(&w0, buf, 4);
  memcpy(&w100, buf + 0x100, 4);
  CHECK(w100 == w0 && !memcmp(buf, buf + 0x100, 16) && buf[0x110] == (uint8_t)(0x110 * 7 + 1),
        "rep movsb copied 16 bytes from %%ds:si to %%es:di");
  CHECK(r_cx == 0x55550000 && r_si == 0x66660010 && r_di == 0x77770110,
        "rep movsb on cx/si/di: ecx %#x esi %#x edi %#x", r_cx, r_si, r_di);
  CHECK(r_loops == 3 && r_cx2 == 0x10000, "loop on cx: %u loops, ecx %#x",
        r_loops, r_cx2);

  call32(f_bpss);
  uint32_t w10;
  memcpy(&w10, buf + 0x10, 4);
  CHECK(resume[R_BPSS].hits == 1, "[bp+di] uses %%ss (a fault near 0): %d",
        resume[R_BPSS].hits);
  CHECK(r_bpds == w10, "ds:[bp+di]: %#x, want %#x", r_bpds, w10);

  call32(f_limit);
  memcpy(&wffc, buf + 0xffc, 4);
  CHECK(r_lim == wffc, "the last dword in the limit: %#x, want %#x", r_lim, wffc);
  CHECK(resume[R_LIMIT].hits == 1, "past the limit is SIGSEGV: %d",
        resume[R_LIMIT].hits);
}

/* ---- 16-bit code --------------------------------------------------------- */
static uint8_t code16[65536] __attribute__((aligned(4096)));
static uint8_t area16[65536] __attribute__((aligned(4096)));

static uint32_t at16(unsigned off) {
  uint32_t v;
  memcpy(&v, area16 + off, 4);
  return v;
}

static void code16_checks(void) {
  memcpy(code16, c16_start, c16_end - c16_start);
  /* At 0xfff0: jmp rel16, to c16_back once %ip wraps at 64 KiB. */
  uint16_t back = c16_back - c16_start, rel = back - 0xfff3;
  code16[0xfff0] = 0xe9;
  memcpy(code16 + 0xfff1, &rel, 2);
  CHECK(mprotect(code16, sizeof code16, PROT_READ | PROT_WRITE | PROT_EXEC) == 0,
        "mprotect the 16-bit code");
  struct user_desc d;
  memset(&d, 0, sizeof d);
  d.entry_number = 0;
  d.base_addr = (uint32_t)(uintptr_t)code16;
  d.limit = 0xffff;
  d.contents = 2; /* code */
  long r = syscall(SYS_modify_ldt, 1, &d, sizeof d);
  CHECK(r == 0, "modify_ldt: the 16-bit code segment: %ld", r);
  memset(&d, 0, sizeof d);
  d.entry_number = 1;
  d.base_addr = (uint32_t)(uintptr_t)area16;
  d.limit = 0xffff;
  r |= syscall(SYS_modify_ldt, 1, &d, sizeof d);
  CHECK(r == 0, "modify_ldt: the 16-bit data and stack segment");
  if (r)
    return;
  far16.off = 0;
  far16.sel = 7;
  call32(f_to16);
  CHECK(int21_hits == 1, "int $0x21 in 16-bit code: %d SIGSEGVs", int21_hits);
  CHECK(at16(0x60) == 0x600d, "the 16-bit code ran to its end: %#x", at16(0x60));
  CHECK(at16(0x10) == 0xaaaa0000, "inc %%ax at operand size 16: %#x", at16(0x10));
  CHECK(at16(0x14) == 0x55557ffa, "pushes on a 16-bit stack: %%esp %#x", at16(0x14));
  CHECK(at16(0x18) == 0x11223344 && (at16(0x1c) & 0xffff) == 0x1234,
        "popl %#x, pop %#x", at16(0x18), at16(0x1c) & 0xffff);
  CHECK(at16(0x20) == 0x55558000, "%%esp after the pops: %#x", at16(0x20));
  CHECK((at16(0x42) & 0xffff) == 0xbeef, "[bx+si]: %#x", at16(0x42) & 0xffff);
  CHECK((at16(0x48) & 0xffff) == 0xcafe, "addr32 [ebx]: %#x", at16(0x48) & 0xffff);
  CHECK(at16(0x24) == 0x55557ffe, "a near call pushes 2 bytes: %%esp %#x",
        at16(0x24));
  CHECK(at16(0x28) == 0x10000 && (at16(0x2c) & 0xffff) == 3,
        "loop on cx: ecx %#x, %u loops", at16(0x28), at16(0x2c) & 0xffff);
  CHECK(at16(0x30) == 0x12340000 && at16(0x34) == 0x56780054 &&
            at16(0x50) == 0x5a5a5a5a,
        "rep stosb on cx/di: ecx %#x edi %#x, %#x", at16(0x30), at16(0x34),
        at16(0x50));
  CHECK(at16(0x38) == 0x77 && at16(0x44) == 0x77 && at16(0x3c) == 0x55558000,
        "a far call to 32-bit code on the 16-bit stack: ecx %#x edx %#x esp %#x",
        at16(0x38), at16(0x44), at16(0x3c));
}

int main(void) {
  setvbuf(stdout, 0, _IONBF, 0);
  CHECK((uintptr_t)&main < 0x80000000u && (uintptr_t)stack32 < 0x80000000u,
        "built as non-PIE, below 2 GiB");
  struct sigaction sa = {0};
  sa.sa_sigaction = on_fault;
  sa.sa_flags = SA_SIGINFO;
  sigaction(SIGSEGV, &sa, 0);
  sigaction(SIGFPE, &sa, 0);
  bcd_checks();
  misc_checks();
  segment_checks();
  code16_checks();
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
